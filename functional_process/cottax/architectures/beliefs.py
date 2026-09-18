"""Drawing a belief table: quantiles to coordinates to batched env values.

`configurations.kinds.BELIEFS` says *what* is uncertain and how (a `Belief` per
boundary input: `uniform`, `relative`, `lognormal` or `factor`, with its parameters).
This module says how a draw becomes a value the graph reads -- the numeric half of
`paper_tests/uq.Input` and `uq.Model.coordinates` / `env`, as functions:

- `coordinate(belief, u, nominal)`: the quantile `u` in [0, 1] as the sampled
  **coordinate** -- the value itself, or the factor for a `factor` row (numpy).
- `nominal_coordinate(belief, nominal)`: the coordinate of the nominal point.
- `value(belief, x, nominal)`: the coordinate as the env value, with a leading batch
  axis kept (jax): a `factor` row scales its nominal array as one.
- `sobol(beliefs, n, seed)`: one scrambled Sobol' set of quantiles, [n, k].
- `coordinates(beliefs, u_q, nominal)`: quantiles to coordinates, row by row.
- `theta_env(beliefs, var_of, theta_coords, nominal)`: the batched env, one `[rows, ...]`
  array per sampled path.

The `dummy` row (a path that maps to nothing) is drawn like any other and dropped by
`theta_env`, so a sample set is the same whether or not an estimator wants the noise
floor it exists for. **The nominal point as the last row** is `ouu.sample`'s
convention, not this module's: `coordinates` maps whatever quantiles it is handed.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from cottax.names import PathMap
from scipy.stats import norm, qmc

from functional_process.configurations.kinds import BELIEFS, Belief

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

KINDS = ("uniform", "relative", "lognormal", "factor")
"""The distributions a `Belief.kind` may name."""


def coordinate(belief: Belief, u, nominal):
    """The sampled coordinate at quantile `u` -- the value for `uniform` / `relative` /
    `lognormal`, the factor for `factor`. `u` may carry a batch axis.

    Raises
    ------
    ValueError
        If `belief.kind` is not one of `KINDS`.
    """
    u = np.asarray(u, dtype=float)
    if belief.kind in {"uniform", "factor"}:
        return belief.a + (belief.b - belief.a) * u
    if belief.kind == "relative":
        return np.asarray(nominal, dtype=float) * (1.0 + belief.a * (2.0 * u - 1.0))
    if belief.kind == "lognormal":
        return np.asarray(nominal, dtype=float) * np.exp(belief.a * norm.ppf(u))
    raise ValueError(f"{belief.path}: kind {belief.kind!r} is not one of {KINDS}")


def nominal_coordinate(belief: Belief, nominal) -> float:
    """The coordinate of the nominal point: `1.0` for a `factor` row, else the nominal
    value itself.
    """
    return 1.0 if belief.kind == "factor" else float(np.asarray(nominal))


def value(belief: Belief, x, nominal):
    """The env value for coordinate `x` (a leading batch axis is kept): the coordinate
    itself, or for a `factor` row the nominal array scaled by it.
    """
    x = jnp.asarray(x)
    if belief.kind != "factor":
        return x
    scale = jnp.asarray(nominal)
    return x[..., None] * scale[None, :] if jnp.ndim(x) else x * scale


def describe(belief: Belief) -> str:
    """The distribution in one short phrase, for a table."""
    if belief.kind == "uniform":
        return f"U[{belief.a:g}, {belief.b:g}]"
    if belief.kind == "relative":
        return f"+-{100 * belief.a:g} %"
    if belief.kind == "lognormal":
        return f"lognormal, sigma = {belief.a:.3g}"
    if belief.kind == "factor":
        return f"x U[{belief.a:g}, {belief.b:g}]"
    return belief.kind


def sobol(beliefs: Iterable[Belief], n: int, seed: int = 0) -> np.ndarray:
    """One scrambled Sobol' set of `n` quantile rows over `beliefs`, [n, k]. `n` a
    power of two keeps the set balanced; scipy warns otherwise and the draw still
    stands.
    """
    k = len(tuple(beliefs))
    return qmc.Sobol(d=k, scramble=True, seed=seed).random(n)


def coordinates(
    beliefs: Iterable[Belief], u_q: np.ndarray, nominal: Mapping[str, object]
) -> np.ndarray:
    """Quantiles `u_q` [rows, k] to coordinates `x` [rows, k], one column per belief;
    `nominal[path]` is what a relative row is relative to (a row absent there --
    `dummy` -- is absolute).
    """
    beliefs = tuple(beliefs)
    u_q = np.asarray(u_q, dtype=float)
    x = np.empty_like(u_q)
    for j, belief in enumerate(beliefs):
        x[:, j] = coordinate(belief, u_q[:, j], nominal.get(belief.path, 1.0))
    return x


def nominal_coordinates(
    beliefs: Iterable[Belief], nominal: Mapping[str, object]
) -> np.ndarray:
    """The nominal point as one row of coordinates, [k]."""
    return np.array([nominal_coordinate(b, nominal.get(b.path, 1.0)) for b in beliefs])


def theta_env(
    beliefs: Iterable[Belief],
    var_of: Mapping[str, object],
    theta_coords: np.ndarray,
    nominal: Mapping[str, object],
) -> PathMap:
    """`{VarPath: [rows, ...] value}` for coordinates `theta_coords` [rows, k]: the
    batched env over the sampled paths. A belief whose path is not in `var_of`
    (`dummy`) is skipped.
    """
    values = {}
    for j, belief in enumerate(beliefs):
        var = var_of.get(belief.path)
        if var is None:
            continue
        values[var] = value(
            belief, jnp.asarray(theta_coords[:, j]), nominal[belief.path]
        )
    return PathMap(values)


def hfact_belief(sigma: float) -> dict:
    """What lognormal(`sigma`) says of the confinement factor: the 90 % interval and
    the mean over the worst (lowest, and highest) decile.
    """
    z = norm.ppf(0.9)
    return {
        "sigma": sigma,
        "interval_90": [
            float(math.exp(-norm.ppf(0.95) * sigma)),
            float(math.exp(norm.ppf(0.95) * sigma)),
        ],
        "lowest_decile_mean": float(math.exp(sigma**2 / 2) * norm.cdf(-z - sigma) / 0.1),
        "highest_decile_mean": float(math.exp(sigma**2 / 2) * norm.sf(z - sigma) / 0.1),
    }


def sampled(
    beliefs: Iterable[Belief] = BELIEFS, held: Iterable[str] = ()
) -> tuple[Belief, ...]:
    """`beliefs` minus the rows in `held` (spellings), the `dummy` row kept."""
    held = set(held)
    return tuple(b for b in beliefs if b.path not in held)


__all__ = [
    "KINDS",
    "coordinate",
    "coordinates",
    "describe",
    "hfact_belief",
    "nominal_coordinate",
    "nominal_coordinates",
    "sampled",
    "sobol",
    "theta_env",
    "value",
]
