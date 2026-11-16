"""ALC's coordination by Anderson mixing: `scipy.optimize.anderson` on the sweep.

**The paper baseline** (from 2026-11-16, `alc_report.md` section 7): this driver at
scipy's own defaults (history `M` 5, regularisation `w0` 0.01, `alpha` 1, no line
search), `f_tol` 1e-6, and every ALC weight at `INITIAL_WEIGHT` 0.5. `benchmark.py`
runs it; `alc.Coordinator` remains for the plain sweeps of the earlier runs.

The same coordination as `alc.Coordinator` (contract `alc.AugmentedLagrangian`), driven
differently. A sweep with the method-of-multipliers update is a map of the
coordination's state,

    G(copies, v) = (where the copies go after one sweep, v + 2 w o w o c),

and ALC's answer is its fixed point: the copies consistent and `v` stationary, so
`c = 0`. `alc.Coordinator` iterates `G` one sweep at a time (with the inexact inner loop
or, `inner_tolerance = inf`, plainly). This driver hands `F(z) = G(z) - z` to scipy's
Anderson mixing, which takes each next point from a combination of the last `M`
(point, residual) pairs. It takes no derivative: each `F` is one sweep, so it is the
SLSQPs' own `jax.jacfwd` inside the disciplines and nothing across them.

**Coordinates.** Anderson mixes in `z = (copy / |copy at start|, v / (2 w^2))`. In those
units `F` is each copy's relative movement over the sweep and the inconsistency `c`
itself, so `f_tol` bounds both, as `alc.Coordinator`'s `tolerance` and
`settled_tolerance` do (`max |F| <= f_tol`). `w` stays at its start (`beta = 1`).
COE's movement is not part of the test.

**As `SlsqpDriver` wraps scipy's SLSQP.** The sweep reading is split into its arrays
(the coordination's context) and the rest; the arrays cross a `jax.pure_callback`, and on
the host one jitted walk of the interior (the three disciplines, their SLSQPs being host
callbacks in their turn) is `F`. The walk is compiled once per graph: it is built when
the graph is traced, so a warm solve reuses it.

**Safeguard.** A point at which a sweep fails (a discipline's model non-finite, as at
w0 = 0.3, `alc_report.md` section 6) or returns a non-finite value is not taken: the
mixing restarts from the last good point with an empty history, up to `restarts` times.
`warmup` plain sweeps (`z <- G(z)`) may precede the mixing, to leave the transient first.

`benchmark.py` and `alc.py` run the baseline; by hand, at other settings:

    python paper_tests/distributed_architectures/anderson.py [--m 5] [--w0 0.5] [--warmup 0]
"""

from __future__ import annotations

import argparse

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from cottax.core.driven import Start, Steps
from functional_process.cottax.architectures.reading import PortDriver

import alc as A
import partition as C

SCALE_FLOOR = 1e-6
"""Below this magnitude a copy's start does not set its unit; the floor does."""

MAX_SWEEPS = 3000
"""The sweep budget, as `alc.MAX_SWEEPS`."""

INITIAL_WEIGHT = 0.5
"""Every ALC weight, at the start and throughout (`beta = 1`), in the baseline: the
fastest stable plain start (T11–T13) and, with Anderson, 262 sweeps against 994 at 1
(T20, T23). Within the range Eq. 29 would give here (0.2–0.5)."""


class _Cap(Exception):
    """The sweep budget is spent."""


class _Failed(Exception):
    """A sweep failed, or returned a non-finite value, at the point asked."""


class AndersonCoordinator(PortDriver[A.AugmentedLagrangian]):
    """`scipy.optimize.anderson` on ALC's sweep map, as a host callback (module
    docstring). Fields as scipy names them where they are scipy's: `M` the history,
    `w0` its regularisation (not ALC's initial weight), `alpha` the initial Jacobian
    `-1/alpha` (1: the first step is a plain sweep), `line_search` (`None` or
    `"armijo"`; each trial point is a sweep)."""

    multiplier: object
    weight: object
    merit: object
    M: int = 5
    w0: float = 0.01
    alpha: float = 1.0
    line_search: str | None = None
    f_tol: float = 1e-6
    warmup: int = 0
    restarts: int = 3
    max_sweeps: int = MAX_SWEEPS
    callback: object = None
    """`f(sweep, max_abs_c, settled, updated, max_weight, max_abs_dv, merit) -> None`
    after each sweep, as `alc.Coordinator` calls it (`updated` always: every sweep
    updates `v`)."""

    needs = frozenset({Start})
    provides = frozenset({Steps})
    computes = (Steps,)

    def read(self, posed, context) -> A.Sweep:
        return A.Sweep(posed, context)

    def solve(self, sweep: A.Sweep, data) -> tuple:
        start = tuple(jnp.asarray(u, dtype=float) for u in data[Start])
        flat0, unravel = ravel_pytree(start)
        names = list(sweep.unknowns)
        v_at, w_at, m_at = (names.index(p) for p in (self.multiplier, self.weight, self.merit))
        ends = np.cumsum([0, *(int(np.size(u)) for u in start)])
        span = [np.arange(ends[i], ends[i + 1]) for i in range(len(start))]
        copy_idx = np.concatenate([span[i] for i in range(len(start)) if i not in {v_at, w_at, m_at}])
        v_idx, w_idx, m_idx = span[v_at], span[w_at], span[m_at]
        dynamic, static = eqx.partition(sweep, eqx.is_array)

        @jax.jit
        def walk(dyn, flat):
            """One sweep at `flat`: where every unknown goes, and `c`."""
            u = unravel(flat)
            moved, held = eqx.combine(dyn, static)(*u)
            moved = tuple(jnp.asarray(a, dtype=b.dtype).reshape(b.shape)
                          for a, b in zip(moved, u, strict=True))
            c = jnp.concatenate([jnp.ravel(jnp.asarray(h, dtype=float)) for h in held])
            return ravel_pytree(moved)[0], c

        host = _host(self, walk, copy_idx, v_idx, w_idx, m_idx)
        flat, steps = jax.pure_callback(
            host, (jax.ShapeDtypeStruct(flat0.shape, flat0.dtype),
                   jax.ShapeDtypeStruct((), np.int32)),
            dynamic, flat0, vmap_method="sequential")
        return (*unravel(flat), steps)


def _host(s: AndersonCoordinator, walk, copy_idx, v_idx, w_idx, m_idx):
    """The host side: scipy's Anderson on `F(z) = G(z) - z`, with the safeguard."""
    from scipy.optimize import NoConvergence, anderson

    def host(dyn, flat_start):
        flat_start = np.asarray(flat_start, dtype=float)
        unit = np.maximum(np.abs(flat_start[copy_idx]), SCALE_FLOOR)
        w = flat_start[w_idx]
        two_w2 = 2.0 * w * w
        n = len(copy_idx)

        def to_flat(z):
            flat = flat_start.copy()
            flat[copy_idx] = z[:n] * unit
            flat[v_idx] = z[n:] * two_w2
            return flat

        count = [0]
        last: dict = {}   # the last good sweep: its point and where it went

        def F(z):
            if count[0] >= s.max_sweeps:
                raise _Cap
            flat = to_flat(z)
            count[0] += 1
            try:
                moved, c = (np.asarray(a, dtype=float) for a in walk(dyn, flat))
            except Exception as failure:
                raise _Failed(f"sweep {count[0]}: {type(failure).__name__}") from failure
            if not (np.all(np.isfinite(moved)) and np.all(np.isfinite(c))):
                raise _Failed(f"sweep {count[0]}: non-finite")
            step = moved[copy_idx] / unit - z[:n]
            residual = np.concatenate([step, c])
            went = moved.copy()
            went[v_idx] = flat[v_idx] + two_w2 * c   # the multiplier update, Eq. 24
            last.update(z=z.copy(), went=went, residual=residual)
            if s.callback is not None:
                s.callback(count[0], np.max(np.abs(c)), np.max(np.abs(step)), True,
                           np.max(w), np.max(np.abs(two_w2 * c)), moved[m_idx][0])
            return residual

        z = np.concatenate([flat_start[copy_idx] / unit, flat_start[v_idx] / two_w2])
        restarts = 0
        try:
            for _ in range(s.warmup):
                z = z + F(z)
            while True:
                try:
                    anderson(F, z, M=s.M, w0=s.w0, alpha=s.alpha, line_search=s.line_search,
                             f_tol=s.f_tol, maxiter=s.max_sweeps)
                    break   # converged: `last` is the converged sweep
                except _Failed as failure:
                    if restarts >= s.restarts or not last:
                        raise
                    restarts += 1
                    print(f"  anderson: restart {restarts} from the last good point ({failure})",
                          flush=True)
                    z = last["z"] + last["residual"]   # a plain sweep from the last good point
        except (_Cap, NoConvergence):
            pass
        except _Failed:
            if not last:
                raise
        return last["went"], np.int32(count[0])

    return host


def coordinator(log=None, **settings) -> AndersonCoordinator:
    """The driver for `alc.driven(coordinator=...)`: as it stands, the baseline;
    `functools.partial(coordinator, M=..., ...)` for other settings."""
    return AndersonCoordinator(A.MULTIPLIER, A.WEIGHT, A.SEEN_MERIT, callback=log, **settings)


def main(argv=None):
    import functools

    import benchmark as B

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--reg", type=float, default=0.01, help="scipy's w0")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--w0", type=float, default=INITIAL_WEIGHT, help="ALC's initial weight")
    ap.add_argument("--max-sweeps", type=int, default=MAX_SWEEPS)
    args = ap.parse_args(argv)
    configuration = C.load(C.CONFIGURATION)
    make = functools.partial(coordinator, M=args.m, w0=args.reg, alpha=args.alpha,
                             warmup=args.warmup, max_sweeps=args.max_sweeps)
    built = B.build_alc(configuration, initial_weight=args.w0, coordinator=make)
    out, seconds = C.timed(built["graph"], built["inputs"])
    row = B.read(configuration, built, out)
    print(f"{row['sweeps']} sweeps in {seconds:.1f} s (cold), COE {row['coe']:.7f}, "
          f"max violation {row['violation']:.1e}")
    print("  ", {k: f"{p['iterations']} ({p['solves']})" for k, p in row["problems"].items()})


if __name__ == "__main__":
    main()
