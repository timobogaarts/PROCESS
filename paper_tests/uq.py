"""The deterministic PROCESS optimum under uncertainty: `stellarator_helias`'s design
fixed at PROCESS's converged `ixc`, one closure kept inside the MDA -- the power balance
(`^cond.constraints.c2`) closed by the plasma density -- and 26 boundary inputs drawn
from stated distributions. Sobol' indices by Saltelli sampling and a plain Monte Carlo,
every evaluation one row of the batched closed MDA on the GPU (`close_conditions.closed`
vmapped over the uncertain leaves, as `--batch` vmaps over the design).

The formulation. PROCESS's answer is a point: eight design values at which two equalities
hold and twelve inequalities are satisfied, with every other input at the file's value.
Here the design is that point, `hfact` -- PROCESS's closure variable for the power
balance -- becomes an *uncertain input* (a confinement-scaling multiplier is a belief,
not a knob), and the plasma runs at whatever density balances its power:
`RootFind((c2,), (nd_plasma_electrons_vol_avg,))`, the 22-node cycle of the pairing
table, flattened and driven by `SafeguardedNewtonDriver`. The net-electric equality
`c16` is *not* closed: net electric power is an output, and `c16`'s residual against its
target (`.constraints.p_plant_electric_net_required_mw`) is reported like an inequality.
Every inequality is reported as its residual (`g <= 0` satisfied, PROCESS's normalised
residual before the sign flip its VMCON wrapper applies) and as a probability of
violation, so the question the plain Monte Carlo answers is how much of the
deterministic optimum's feasibility is luck.

The inputs are `INPUTS` -- a table, meant to be edited: path, distribution, parameters.
A `dummy` input that nothing reads is in the table on purpose (its indices are the
estimator's noise floor).

    G=~/miniconda3/envs/process_port_gpu/bin/python; export XLA_PYTHON_CLIENT_PREALLOCATE=false
    JAX_PLATFORMS=cuda $G paper_tests/uq.py                 # ~2 min: 237k + 100k evaluations
    JAX_PLATFORMS=cuda $G paper_tests/uq.py --n-base 65536 --n-mc 200000 --variant no-hfact
                                                             # ~5 min: 1.9 M + 200k evaluations
    JAX_PLATFORMS=cpu  $G paper_tests/uq.py --smoke          # N=64, chunk 256: exercises everything
    $G paper_tests/uq.py --render [--variant V]              # tables + figures from the saved samples

`--variant`: `nominal` (the table as stated), `hfact-5pct` (hfact lognormal(0.05)),
`no-hfact` (hfact fixed at its nominal and removed from the inputs). Output names carry
the variant as a suffix; the nominal keeps the bare names.

Outputs: `out/uq_inputs.tex`, `out/uq_sobol.{csv,tex}`, `out/uq_mc.{csv,tex}`,
`out/uq.json` (timings, convergence, the nominal check), `out/uq_tornado_*.png`,
`out/uq_hist_net_power.png`, `out/uq_hist_cost.png`. The raw evaluations
(`uq_saltelli.npz`, `uq_mc.npz`) land in `--samples` (default
`~/.cache/functional_process/uq/`), not in `out/`.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)  # before any array: PROCESS is float64

import close_conditions as cc  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from common import OUT, ROOT, sci, write_csv, write_json, write_tex  # noqa: E402
from cottax.names import PathMap  # noqa: E402
from cottax.problem import Converged, Steps  # noqa: E402
from scipy.stats import norm, qmc  # noqa: E402

from functional_process.cottax import mdf, sand  # noqa: E402
from functional_process.cottax.mda_harness import CACHE_DIR  # noqa: E402
from functional_process.cottax.sand_harness import reference_run  # noqa: E402

PAIRING = {"^cond.constraints.c2": ".physics.nd_plasma_electrons_vol_avg"}
"""The one closure: the power balance by the density (22-node cycle, scaled
sensitivity 0.33 in `close_conditions.py --table`)."""

SAMPLES = Path(os.environ.get("FP_UQ_SAMPLES", CACHE_DIR / "uq"))

LN2_HALF = math.log(2.0) / 2.0
"""`lognormal(ln 2 / 2)`: a factor of 0.5-2 at two sigma."""


# ---------------------------------------------------------------- the input table


@dataclasses.dataclass(frozen=True)
class Input:
    """One uncertain boundary input.

    `kind`: `uniform` -- uniform on [a, b] (absolute); `relative` -- uniform in
    nominal x [1 - a, 1 + a]; `lognormal` -- nominal x exp(a z), z standard normal;
    `factor` -- nominal x uniform[a, b], the nominal an array scaled as one (the
    sampled coordinate is the factor). `path` is the boundary input's spelling or
    `dummy`, which maps to nothing.
    """

    path: str
    kind: str
    a: float = 0.0
    b: float = 0.0
    note: str = ""

    @property
    def label(self) -> str:
        return self.path.split(".")[-1] if self.path != "dummy" else "dummy"

    def coordinate(self, u, nominal):
        """The sampled coordinate `x` (the value, or the factor) at quantile `u`."""
        if self.kind == "uniform":
            return self.a + (self.b - self.a) * u
        if self.kind == "relative":
            return nominal * (1.0 + self.a * (2.0 * u - 1.0))
        if self.kind == "lognormal":
            return nominal * np.exp(self.a * norm.ppf(u))
        if self.kind == "factor":
            return self.a + (self.b - self.a) * u
        raise ValueError(self.kind)

    def nominal_coordinate(self, nominal):
        return 1.0 if self.kind == "factor" else float(nominal)

    def value(self, x, nominal):
        """The env value for coordinate `x` (`x` may carry a leading batch axis)."""
        if self.kind == "factor":
            return jnp.asarray(x)[..., None] * jnp.asarray(nominal)[None, :] \
                if jnp.ndim(x) else jnp.asarray(x) * jnp.asarray(nominal)
        return jnp.asarray(x)

    def describe(self) -> str:
        if self.kind == "uniform":
            return f"U[{self.a:g}, {self.b:g}]"
        if self.kind == "relative":
            return f"±{100 * self.a:g} %"
        if self.kind == "lognormal":
            return f"lognormal, σ = {self.a:.3g}"
        if self.kind == "factor":
            return f"× U[{self.a:g}, {self.b:g}]"
        return self.kind


INPUTS = (
    Input(".physics.hfact", "lognormal", 0.15, note="confinement multiplier; PROCESS's closure variable, here a belief"),
    Input(".physics.alphan", "relative", 0.20, note="density profile exponent"),
    Input(".physics.alphat", "relative", 0.20, note="temperature profile exponent"),
    Input(".physics.f_p_alpha_plasma_deposited", "uniform", 0.90, 0.99, note="alpha power deposited in the plasma"),
    Input(".physics.f_temp_plasma_ion_electron", "uniform", 0.85, 1.0, note="$T_i / T_e$"),
    Input(".physics.f_sync_reflect", "uniform", 0.5, 0.7, note="synchrotron wall reflectivity"),
    Input(".impurity_radiation.f_nd_impurity_electron_array[13]", "lognormal", LN2_HALF,
          note="tungsten fraction (index 13, the only seeded non-alpha impurity)"),
    Input(".stellarator.bmn", "lognormal", LN2_HALF, note="residual field ripple"),
    Input(".stellarator.f_asym", "uniform", 1.0, 1.2, note="divertor heat-load asymmetry"),
    Input(".stellarator.f_rad", "uniform", 0.75, 0.95, note="radiated fraction in the SOL / divertor"),
    Input(".fwbs.f_p_blkt_multiplication", "uniform", 1.25, 1.45, note="blanket energy multiplication"),
    Input(".fwbs.declfw", "relative", 0.20, note="first-wall neutron decay length"),
    Input(".fwbs.declblkt", "relative", 0.20, note="blanket neutron decay length"),
    Input(".fwbs.fhole", "uniform", 0.0, 0.05, note="first-wall hole fraction"),
    Input(".fwbs.dr_fw_wall", "uniform", 0.002, 0.004, note="first-wall armour thickness [m]"),
    Input(".tfcoil.temp_tf_cryo", "uniform", 4.2, 4.8, note="coil temperature [K]"),
    Input(".tfcoil.dx_tf_wp_insulation", "uniform", 0.008, 0.012, note="winding-pack insulation [m]"),
    Input(".constraints.f_j_tf_wp_critical_max", "uniform", 0.7, 0.9, note="allowed fraction of the critical current"),
    Input(".heat_transport.eta_turbine", "uniform", 0.33, 0.42, note="thermal-to-electric efficiency"),
    Input(".current_drive.eta_ecrh_injector_wall_plug", "uniform", 0.4, 0.6, note="ECRH wall-plug efficiency"),
    Input(".heat_transport.f_p_fw_coolant_pump_total_heat", "relative", 0.30, note="FW pumping power fraction"),
    Input(".divertor.tdiv", "uniform", 3.0, 8.0, note="divertor plasma temperature [eV]"),
    Input(".costs.f_t_plant_available", "uniform", 0.65, 0.85, note="availability"),
    Input(".costs.life_plant", "uniform", 30.0, 50.0, note="plant life [y]"),
    Input(".costs.discount_rate", "uniform", 0.04, 0.08, note="discount rate"),
    Input(".costs.ucsc", "factor", 0.7, 1.5, note="superconductor unit costs, the (9,) array scaled together"),
    Input("dummy", "uniform", 0.0, 1.0, note="read by nothing: the estimators' noise floor"),
)

VARIANTS = {
    "nominal": "hfact lognormal(0.15), the table as stated",
    "hfact-5pct": "hfact lognormal(0.05), everything else unchanged",
    "no-hfact": "hfact fixed at its nominal and removed from the inputs (k = 25 + dummy)",
}


def inputs_for(variant: str) -> tuple:
    """`INPUTS` under `variant`."""
    if variant == "nominal":
        return INPUTS
    if variant == "hfact-5pct":
        return tuple(dataclasses.replace(i, a=0.05) if i.path == ".physics.hfact" else i for i in INPUTS)
    if variant == "no-hfact":
        return tuple(i for i in INPUTS if i.path != ".physics.hfact")
    raise ValueError(f"variant {variant!r}; one of {sorted(VARIANTS)}")


def named(stem: str, variant: str) -> str:
    """`uq_sobol` -> `uq_sobol_no-hfact`; the nominal variant keeps the bare name."""
    return stem if variant == "nominal" else f"{stem}_{variant}"


OUTPUTS = (
    (".heat_transport.p_plant_electric_net_mw", "net electric power [MW]"),
    ("^cond.numerics.objf", "objective (coe / 100)"),
    (".costs.coe", "cost of electricity [$/MWh]"),
    (".costs.concost", "constructed cost [M$]"),
    (".physics.p_fusion_total_mw", "fusion power [MW]"),
    (".physics.nd_plasma_electrons_vol_avg", "closed density [m^-3]"),
    ("^cond.constraints.c16", "c16: net power vs target (residual)"),
    ("^cond.constraints.c2", "c2: power balance (closed, ~0)"),
)
"""Scalar outputs read off the run env, plus every inequality residual (`INEQUALITIES`,
found on the built problem) and the closing Newton's `Steps` / `Converged`."""

CONSTRAINT_NAMES = {
    2: "power balance", 8: "neutron wall load", 16: "net electric power", 17: "radiation fraction",
    18: "divertor heat load", 24: "beta limit", 32: "TF conduit stress", 34: "TF dump voltage",
    35: "TF J_wp quench protection", 62: "alpha / energy confinement time", 65: "VV stress at quench",
    67: "radiation wall load", 82: "build: toroidal consistency", 83: "build: radial consistency",
}
"""The `constraint_<n>` docstrings of `core/solver/constraints.py`, shortened."""


# ---------------------------------------------------------------- the model


@dataclasses.dataclass
class Model:
    """The closed MDA at PROCESS's optimum, and the pruned, batched program over the
    uncertain leaves.
    """

    live: object
    built: cc.Closed
    point: PathMap
    """Every schedule input at the nominal (the design at PROCESS's `x`, the `^guess`
    of the density at its root)."""
    inputs: tuple
    """`INPUTS` that are inputs of this schedule; `dummy` kept."""
    dropped: list
    var_of: dict
    """spelling -> VarPath, inputs and outputs alike."""
    nominal: dict
    """input spelling -> nominal value (numpy)."""
    wanted: tuple
    """output VarPaths the pruned program returns, in `columns` order."""
    columns: tuple
    """spellings of `wanted`."""
    inequalities: tuple
    guess: object
    """The density's `^guess` port."""
    unknowns: tuple = ()
    """Every unknown of the closing problem (the density and the residualised
    Picards' copies), with `guesses` their `^guess` ports and `u0` their roots at
    the nominal."""
    guesses: dict = dataclasses.field(default_factory=dict)
    u0: np.ndarray | None = None
    sensitivity: np.ndarray | None = None
    """d unknowns / d coordinate at the nominal, [unknowns, inputs] (the predictor)."""
    dene0: float = 0.0
    _programs: dict = dataclasses.field(default_factory=dict)

    def x0(self) -> np.ndarray:
        return np.array([i.nominal_coordinate(self.nominal.get(i.path, 1.0)) for i in self.inputs])

    def coordinates(self, U: np.ndarray) -> np.ndarray:
        """Quantiles `U` [M, k] -> coordinates `X` [M, k]."""
        X = np.empty_like(U)
        for j, i in enumerate(self.inputs):
            X[:, j] = i.coordinate(U[:, j], self.nominal.get(i.path, 1.0))
        return X

    def predicted(self, X: np.ndarray, predict: str) -> np.ndarray:
        """Every closing unknown's start per row, [rows, unknowns]: `linear` --
        `u0 + J (x - x0)`; `loglinear` -- `u0 * prod (x_i / x_i0) ** s_i` with `s_i`
        the scaled sensitivity, the linear term where `x_i0 = 0`. Clipped to
        [0.3, 3] x `u0`. All of the problem's unknowns move together: a density start
        far from the nominal with the ion-density copies left at it is an inconsistent
        point the safeguarded Newton stalls at.
        """
        x0 = self.x0()[None, :]
        u0 = self.u0[None, :]
        J = self.sensitivity  # [unknowns, inputs]
        if predict == "linear":
            pred = u0 + (X - x0) @ J.T
        elif predict == "loglinear":
            nonzero = np.abs(x0[0]) > 0
            s_scaled = np.where(nonzero[None, :], J * x0 / self.u0[:, None], 0.0)  # [unknowns, inputs]
            ratio = np.where(nonzero[None, :], X / np.where(nonzero, x0, 1.0), 1.0)
            log_pred = np.log(u0) + np.log(np.maximum(ratio, 1e-12)) @ s_scaled.T
            log_pred += ((X - x0) @ np.where(nonzero[None, :], 0.0, J).T) / u0
            pred = np.exp(log_pred)
        else:
            raise ValueError(predict)
        return np.clip(pred, 0.3 * u0, 3.0 * u0)

    def env(self, X: np.ndarray, predict: str | None) -> tuple[PathMap, PathMap]:
        """The batched env for coordinates `X` and its `in_axes`; `predict` is
        `None` (the nominal root as the start), `linear` or `loglinear`.
        """
        values = dict(self.point.items())
        batched = set()
        for j, i in enumerate(self.inputs):
            if i.path == "dummy":
                continue
            v = self.var_of[i.path]
            values[v] = i.value(jnp.asarray(X[:, j]), self.nominal[i.path])
            batched.add(v)
        if predict and self.sensitivity is not None:
            pred = self.predicted(X, predict)
            for j, u in enumerate(self.unknowns):
                values[self.guesses[u]] = jnp.asarray(pred[:, j])
                batched.add(self.guesses[u])
        axes = PathMap((v, 0 if v in batched else None) for v in self.point)
        return PathMap(values), axes

    def program(self, predict: str | None):
        """The jitted, vmapped, pruned run over a chunk (cached per `predict`, so the
        Saltelli and Monte Carlo passes share one compile).
        """
        if predict not in self._programs:
            run = self.built.problem.traceable.run
            wanted = self.wanted

            def pruned(given):
                out = run(given)
                return jnp.stack([jnp.asarray(out[w], dtype=jnp.float64).reshape(()) for w in wanted])

            _, axes = self.env(np.zeros((1, len(self.inputs))), predict)
            self._programs[predict] = jax.jit(jax.vmap(pruned, in_axes=(axes,)))
        return self._programs[predict]


def build(pairing=PAIRING, variant: str = "nominal") -> tuple[Model, dict]:
    """Open the session, close `c2` by the density, put the design at PROCESS's
    answer, prime, and check the nominal against PROCESS's converged run.
    """
    began = time.perf_counter()
    live = cc.open_live()
    built = cc.closed(live, pairing, flatten=True)
    problem = built.problem
    run = reference_run(str(ROOT / cc.PATH))
    process_x = {sand.iteration_variable_path(i): v for i, v in run.converged.items()}
    env = mdf.seed(problem, live.cold)
    schedule_inputs = set(problem.eager.inputs)
    for var, value in process_x.items():
        if var in schedule_inputs:
            env[var] = jnp.asarray(float(value))
    closing = next(iter(built.pairings.values()))
    ports = mdf.guess_ports(problem)
    for var in built.pairings.values():  # every closing variable starts at PROCESS's answer
        env[next(g for g, u in ports.items() if u == var)] = jnp.asarray(float(process_x[var]))
    guess = next(g for g, u in ports.items() if u == closing)
    env, primed = mdf.prime(problem, env)
    point = PathMap(mdf._inputs_only(problem, env).items())
    var_of = {v.spelling: v for v in point}
    var_of.update({k.spelling: k for k in primed})
    build_s = time.perf_counter() - began

    inputs, dropped, nominal = [], [], {}
    for i in inputs_for(variant):
        if i.path == "dummy":
            inputs.append(i)
            continue
        if i.path not in var_of or var_of[i.path] not in set(point):
            dropped.append(i.path)
            continue
        inputs.append(i)
        nominal[i.path] = np.asarray(point[var_of[i.path]])
    place = next(iter(built.places.values()))  # one place, also when two conditions are closed together
    inequalities = tuple(problem.report["inequalities"])
    wanted = [var_of[s] for s, _ in OUTPUTS] + list(inequalities)
    wanted += [Steps.name_for(place), Converged.name_for(place)]
    # Every other driver's verdict, so a chunk's validity mask sees all of them.
    prefix = Converged.name_for(place).spelling.split("^problem", 1)[0]
    others = sorted(
        (k for k in primed if k.spelling.startswith(prefix) and k != Converged.name_for(place)),
        key=lambda k: k.spelling,
    )
    wanted += others
    columns = tuple(k.spelling for k in wanted)
    model = Model(
        live=live, built=built, point=point, inputs=tuple(inputs), dropped=dropped, var_of=var_of,
        nominal=nominal, wanted=tuple(wanted), columns=columns, inequalities=inequalities, guess=guess,
    )
    d = run.data
    at = {s: float(np.asarray(primed[var_of[s]])) for s, _ in OUTPUTS}
    reference = {
        ".heat_transport.p_plant_electric_net_mw": float(d.heat_transport.p_plant_electric_net_mw),
        ".costs.coe": float(d.costs.coe),
        ".costs.concost": float(d.costs.concost),
        ".physics.p_fusion_total_mw": float(d.physics.p_fusion_total_mw),
        ".physics.nd_plasma_electrons_vol_avg": float(d.physics.nd_plasma_electrons_vol_avg),
        "^cond.numerics.objf": float(d.costs.coe) / 100.0,
    }
    check = {
        "variant": variant,
        "variant_note": VARIANTS[variant],
        "build_s": build_s,
        "design": {k.spelling: float(v) for k, v in process_x.items()},
        "closing": closing.spelling,
        "pairing": {c.spelling: v.spelling for c, v in built.pairings.items()},
        "root_find_at_nominal": {c.spelling: r for c, r in built.root_find_reports(primed).items()},
        "port_at_nominal": at,
        "process_converged": reference,
        "relative_difference": {k: at[k] / reference[k] - 1.0 for k in reference},
        "inequalities_at_nominal": {c.spelling: float(np.asarray(primed[c])) for c in inequalities},
        "c16_target_mw": float(np.asarray(point[var_of[".constraints.p_plant_electric_net_required_mw"]])),
        "dropped_inputs": dropped,
        "n_inputs": len(inputs),
        "n_schedule_inputs": len(point),
        "n_outputs_full": len(primed),
        "n_outputs_pruned": len(wanted),
        "other_driver_verdicts": [k.spelling for k in others],
        "blocking": cc.describe(built.blocking),
    }
    model.dene0 = at[".physics.nd_plasma_electrons_vol_avg"]
    ports = mdf.guess_ports(problem)
    model.unknowns = tuple(problem.graph[place].unknowns)
    model.guesses = {u: g for g, u in ports.items() if u in set(model.unknowns)}
    model.u0 = np.array([float(np.asarray(primed[u])) for u in model.unknowns])
    check["closing_unknowns"] = {u.spelling: float(v) for u, v in zip(model.unknowns, model.u0, strict=True)}
    return model, check


def sensitivity(model: Model) -> tuple[np.ndarray, float]:
    """D density / d coordinate at the nominal by one `jax.jacfwd` of the unbatched
    closed MDA (through the root find's `custom_root`), one column per input.
    """
    run = jax.jit(model.built.problem.traceable.run)
    x0 = jnp.asarray(model.x0())

    def roots(x):
        values = dict(model.point.items())
        for j, i in enumerate(model.inputs):
            if i.path == "dummy":
                continue
            values[model.var_of[i.path]] = i.value(x[j], model.nominal[i.path])
        out = run(PathMap(values))
        return jnp.stack([out[u] for u in model.unknowns])

    began = time.perf_counter()
    grad = jax.block_until_ready(jax.jacfwd(roots)(x0))
    seconds = time.perf_counter() - began
    return np.asarray(grad), seconds


# ---------------------------------------------------------------- evaluation


def evaluate(model: Model, X: np.ndarray, chunk: int, predict: str | None, label: str) -> dict:
    """Every row of `X` through the batched closed MDA, `chunk` rows per call (the
    last chunk padded with the nominal so that one program serves every call).
    """
    m, _k = X.shape
    wanted = model.wanted
    fn = model.program(predict)
    steps_col = len(OUTPUTS) + len(model.inequalities)
    Y = np.empty((m, len(wanted)))
    walls, first = [], None
    x0 = model.x0()
    for start in range(0, m, chunk):
        rows = X[start:start + chunk]
        n = rows.shape[0]
        if n < chunk:
            rows = np.concatenate([rows, np.repeat(x0[None, :], chunk - n, axis=0)])
        env, _ = model.env(rows, predict)
        began = time.perf_counter()
        out = jax.block_until_ready(fn(env))
        wall = time.perf_counter() - began
        if first is None:
            first = wall
        else:
            walls.append(wall)
        Y[start:start + n] = np.asarray(out)[:n]
        del out
        steps = Y[start:start + n, steps_col]
        conv = Y[start:start + n, steps_col + 1]
        print(
            f"[{label}] rows {start:7d}-{start + n:7d}: {wall * 1e3:7.1f} ms, "
            f"steps max {int(steps.max())} mean {steps.mean():.2f}, "
            f"unconverged {int((conv < 0.5).sum())}/{n}", flush=True,
        )
    return {"Y": Y, "first_call_s": first, "walls": walls, "chunk": chunk, "predict": predict}


def valid_rows(model: Model, Y: np.ndarray) -> np.ndarray:
    """A row counts if every driver reported convergence and every reported output is
    finite.
    """
    n_fixed = len(OUTPUTS) + len(model.inequalities)
    converged = np.all(Y[:, n_fixed + 1:] > 0.5, axis=1)
    finite = np.all(np.isfinite(Y[:, :n_fixed]), axis=1)
    return converged & finite


COE_CAP = 1000.0
"""`coe` saturates at ~1e21 $/MWh when the net power is <= 0 (`costs.py`'s
`kwhpy = max(kwhpy, 1e-10)` guard), which would make its variance the variance of
an indicator; `coe_capped` is `min(coe, COE_CAP)`, a bounded quantity of interest whose
Sobol' indices mean something."""

SKIP_SOBOL = ("^cond.constraints.c2", "^cond.numerics.objf")
"""Not analysed: the closed residual (zero by construction) and `objf` (= coe / 100)."""


def analysis(model: Model, Y: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """The analysed columns: every scalar output and inequality residual of `Y`, plus
    the derived `coe_capped`.
    """
    names = [c for c in model.columns if not (c.startswith("^driver_out.") or c.startswith("^steps.") or c.startswith("^converged."))]
    Ya = Y[:, [model.columns.index(c) for c in names]]
    coe = Y[:, model.columns.index(".costs.coe")]
    Ya = np.concatenate([Ya, np.minimum(coe, COE_CAP)[:, None]], axis=1)
    names.append("coe_capped")
    return Ya, names


# ---------------------------------------------------------------- Saltelli / Sobol'


def saltelli(k: int, n: int, seed: int = 0) -> np.ndarray:
    """The (k + 2) n quantile rows: A, B, then AB_1 .. AB_k (A with column i from B),
    A and B the two halves of one scrambled 2k-dimensional Sobol' sequence.
    """
    if n & (n - 1):
        raise ValueError(f"n = {n} is not a power of two (Sobol' balance)")
    base = qmc.Sobol(d=2 * k, scramble=True, seed=seed).random(n)
    A, B = base[:, :k], base[:, k:]
    blocks = [A, B]
    for i in range(k):
        AB = A.copy()
        AB[:, i] = B[:, i]
        blocks.append(AB)
    return np.concatenate(blocks, axis=0)


def sobol_indices(fA, fB, fAB):
    """Saltelli 2010's first-order estimator and Jansen's total-order estimator.

    `fA`, `fB` [n]; `fAB` [k, n]. Variance from the pooled A and B samples.
    """
    pooled = np.concatenate([fA, fB])
    var = pooled.var(ddof=1)
    if var <= 0.0:
        return np.zeros(fAB.shape[0]), np.zeros(fAB.shape[0])
    s1 = np.mean(fB[None, :] * (fAB - fA[None, :]), axis=1) / var
    st = 0.5 * np.mean((fA[None, :] - fAB) ** 2, axis=1) / var
    return s1, st


def bootstrap(fA, fB, fAB, replicates: int, seed: int = 0):
    """95 % percentile intervals over resamples of the base rows."""
    rng = np.random.default_rng(seed)
    n = fA.shape[0]
    s1s, sts = [], []
    for _ in range(replicates):
        idx = rng.integers(0, n, size=n)
        s1, st = sobol_indices(fA[idx], fB[idx], fAB[:, idx])
        s1s.append(s1)
        sts.append(st)
    s1s, sts = np.array(s1s), np.array(sts)
    return (np.percentile(s1s, 2.5, axis=0), np.percentile(s1s, 97.5, axis=0),
            np.percentile(sts, 2.5, axis=0), np.percentile(sts, 97.5, axis=0))


def sobol_table(model: Model, Y: np.ndarray, n: int, replicates: int) -> tuple[list, dict]:
    """Rows: one per (input, output) with S1, ST and their CIs; the complete-case
    mask over A, B and every AB_i applied row-wise.
    """
    k = len(model.inputs)
    valid = valid_rows(model, Y)
    keep = valid[:n] & valid[n:2 * n]
    for i in range(k):
        keep &= valid[(2 + i) * n:(3 + i) * n]
    kept = int(keep.sum())
    rows, summary = [], {"n": n, "k": k, "rows_kept": kept, "rows_dropped": n - kept,
                         "evaluations": Y.shape[0], "unconverged_evaluations": int((~valid).sum())}
    began = time.perf_counter()
    Ya, names = analysis(model, Y)
    for col, name in enumerate(names):
        if name in SKIP_SOBOL:
            continue
        f = Ya[:, col]
        fA, fB = f[:n][keep], f[n:2 * n][keep]
        fAB = np.stack([f[(2 + i) * n:(3 + i) * n][keep] for i in range(k)])
        s1, st = sobol_indices(fA, fB, fAB)
        lo1, hi1, loT, hiT = bootstrap(fA, fB, fAB, replicates)
        for j, inp in enumerate(model.inputs):
            rows.append({
                "output": name, "input": inp.path, "label": inp.label,
                "S1": float(s1[j]), "S1_lo": float(lo1[j]), "S1_hi": float(hi1[j]),
                "ST": float(st[j]), "ST_lo": float(loT[j]), "ST_hi": float(hiT[j]),
            })
        summary.setdefault("sum_S1", {})[name] = float(s1.sum())
        summary.setdefault("sum_ST", {})[name] = float(st.sum())
        summary.setdefault("variance", {})[name] = float(np.concatenate([fA, fB]).var(ddof=1))
    summary["estimators_s"] = time.perf_counter() - began
    return rows, summary


# ---------------------------------------------------------------- Monte Carlo


def monte_carlo_table(model: Model, Y: np.ndarray, check: dict) -> tuple[list, dict]:
    valid = valid_rows(model, Y)
    Ya, names = analysis(model, Y)
    Yv = Ya[valid]
    nominal_of = dict(check["port_at_nominal"])
    nominal_of.update(check["inequalities_at_nominal"])
    nominal_of["coe_capped"] = min(nominal_of[".costs.coe"], COE_CAP)
    rows = []
    for col, name in enumerate(names):
        f = Yv[:, col]
        row = {
            "output": name, "label": cname(name), "nominal": nominal_of.get(name),
            "mean": float(f.mean()), "sd": float(f.std(ddof=1)),
            "q05": float(np.percentile(f, 5)), "q50": float(np.percentile(f, 50)),
            "q95": float(np.percentile(f, 95)), "min": float(f.min()), "max": float(f.max()),
        }
        if name.startswith("^cond.constraints.") and name != "^cond.constraints.c2":
            row["p_violated"] = float((f > 0.0).mean())
            row["p_worse_than_nominal"] = float((f > nominal_of[name]).mean())
        rows.append(row)
    net = Yv[:, names.index(".heat_transport.p_plant_electric_net_mw")]
    target = check["c16_target_mw"]
    ineq = [names.index(c.spelling) for c in model.inequalities]
    g = Yv[:, ineq]
    g0 = np.array([nominal_of[c.spelling] for c in model.inequalities])
    all_ok = np.all(g <= 0.0, axis=1)
    no_worse = np.all(g <= np.maximum(g0, 0.0)[None, :], axis=1)
    summary = {
        "samples": int(Y.shape[0]), "valid": int(valid.sum()), "invalid": int((~valid).sum()),
        "target_mw": target,
        "p_net_below_target": float((net < target).mean()),
        "p_net_below_nominal": float((net < nominal_of[".heat_transport.p_plant_electric_net_mw"]).mean()),
        "p_net_nonpositive": float((net <= 0.0).mean()),
        "p_all_inequalities_satisfied": float(all_ok.mean()),
        "p_all_satisfied_and_net_above_target": float((all_ok & (net >= target)).mean()),
        "p_no_inequality_worse_than_nominal": float(no_worse.mean()),
        "p_no_worse_and_net_above_nominal": float((no_worse & (net >= nominal_of[".heat_transport.p_plant_electric_net_mw"])).mean()),
        "p_violated": {c.spelling: float((Yv[:, i] > 0.0).mean()) for c, i in zip(model.inequalities, ineq, strict=True)},
        "p_worse_than_nominal": {c.spelling: float((Yv[:, i] > nominal_of[c.spelling]).mean()) for c, i in zip(model.inequalities, ineq, strict=True)},
        "mean_violations_per_sample": float((g > 0.0).sum(axis=1).mean()),
        "max_abs_c2": float(np.max(np.abs(Yv[:, names.index("^cond.constraints.c2")]))),
    }
    return rows, summary


# ---------------------------------------------------------------- rendering


def tex(text: str) -> str:
    """A label as LaTeX text: the characters that would break a tabular escaped."""
    return (text.replace("_", r"\_").replace("^", r"\^{}").replace("%", r"\%").replace("$", r"\$")
            .replace("m\\^{}-3", r"m$^{-3}$"))


def cname(spelling: str) -> str:
    """`^cond.constraints.c24` -> `c24 (beta limit)`."""
    if spelling.startswith("^cond.constraints.c"):
        cid = int(spelling.rsplit("c", 1)[1])
        return f"c{cid} ({CONSTRAINT_NAMES.get(cid, '')})"
    if spelling == "coe_capped":
        return f"coe capped at {COE_CAP:g} [$/MWh]"
    return dict(OUTPUTS).get(spelling, spelling)


def worst_margin(check: dict) -> str:
    """The inequality with the smallest margin at the nominal (largest g <= 0)."""
    at = check["inequalities_at_nominal"]
    return max(at, key=lambda c: at[c])


def render_inputs(model: Model, variant: str) -> None:
    header = ["input", "distribution", "nominal", "note"]
    body = []
    for i in model.inputs:
        nominal = model.nominal.get(i.path)
        shown = "--" if nominal is None else (
            f"{float(nominal):.4g}" if np.ndim(nominal) == 0 else "(" + ", ".join(f"{v:g}" for v in np.asarray(nominal)) + ")"
        )
        body.append([r"\texttt{" + tex(i.path) + "}",
                     tex(i.describe()).replace("±", r"$\pm$").replace("σ", r"$\sigma$").replace("×", r"$\times$"),
                     shown, i.note])
    write_tex("uq.py", header, body, name=named("uq_inputs", variant), align="llrl",
              caption_note="the uncertain inputs; nominal = the input file's or the vendored default's value; "
                           "lognormal(sigma): nominal x exp(sigma z)")


def render_sobol(model: Model, rows: list, check: dict, variant: str) -> None:
    write_csv("uq.py", rows, name=named("uq_sobol", variant))
    net, cost, capital = ".heat_transport.p_plant_electric_net_mw", "coe_capped", ".costs.concost"
    worst = worst_margin(check)
    by = {(r["input"], r["output"]): r for r in rows}
    order = sorted(model.inputs, key=lambda i: -by[i.path, net]["ST"])
    header = ["input", r"$S_1$ net", r"$S_T$ net", r"$S_1$ coe", r"$S_T$ coe", r"$S_1$ concost", r"$S_T$ concost",
              f"$S_1$ {worst.rsplit('.', 1)[1]}", f"$S_T$ {worst.rsplit('.', 1)[1]}"]

    def cell(r, key):
        lo, hi = r[f"{key}_lo"], r[f"{key}_hi"]
        return f"{r[key]:.3f} [{lo:.3f}, {hi:.3f}]"

    body = []
    for i in order:
        body.append([r"\texttt{" + tex(i.label) + "}",
                     *(cell(by[i.path, o], key) for o in (net, cost, capital, worst) for key in ("S1", "ST"))])
    write_tex("uq.py", header, body, name=named("uq_sobol", variant), align="lrrrrrrrr",
              caption_note=f"Sobol' indices (Saltelli 2010 S1, Jansen ST) with bootstrap 95 % CIs, "
                           f"sorted by ST for net power; coe capped at {COE_CAP:g} $/MWh (its 1e21 sentinel at "
                           f"net power <= 0 is an indicator otherwise); last pair: {cname(worst)}, "
                           f"the worst-margin inequality at the nominal")


def render_mc(rows: list, summary: dict, variant: str) -> None:
    write_csv("uq.py", rows, name=named("uq_mc", variant))
    header = ["output", "nominal", "mean", "sd", r"$q_{5}$", r"$q_{50}$", r"$q_{95}$", "P(g > 0)", r"P(g > g$_\mathrm{nom}$)"]
    body = []
    for r in rows:
        if r["output"] in ("^cond.constraints.c2", "^cond.numerics.objf"):
            continue
        f = (lambda v: f"{v:.4g}") if abs(r["mean"]) < 1e6 else sci
        body.append([tex(cname(r["output"])),
                     f(r["nominal"]) if r["nominal"] is not None else "--",
                     f(r["mean"]), f(r["sd"]), f(r["q05"]), f(r["q50"]), f(r["q95"]),
                     f"{r['p_violated']:.3f}" if "p_violated" in r else "--",
                     f"{r['p_worse_than_nominal']:.3f}" if "p_worse_than_nominal" in r else "--"])
    write_tex("uq.py", header, body, name=named("uq_mc", variant), align="lrrrrrrrr",
              caption_note=(f"plain Monte Carlo, {summary['valid']} valid of {summary['samples']} samples; "
                            f"P(net < {summary['target_mw']:g} MW) = {summary['p_net_below_target']:.3f}; "
                            f"P(all inequalities satisfied) = {summary['p_all_inequalities_satisfied']:.3f}; "
                            "residuals g <= 0 satisfied"))


BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, RULE = "#0b0b0b", "#52514e", "#e6e5e2"


def _style(ax):
    ax.grid(True, which="major", color=RULE, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(RULE)
    ax.tick_params(colors=MUTED, labelsize=8)


def tornado(rows: list, output: str, title: str, name: str, top: int = 12) -> Path:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    of = sorted((r for r in rows if r["output"] == output), key=lambda r: r["ST"])[-top:]
    fig, ax = plt.subplots(figsize=(6, 0.32 * len(of) + 1.2))
    y = np.arange(len(of))
    ax.barh(y, [r["ST"] for r in of], color=ORANGE, height=0.62, label="$S_T$ (total)")
    ax.barh(y, [r["S1"] for r in of], color=BLUE, height=0.34, label="$S_1$ (first order)")
    ax.errorbar([r["ST"] for r in of], y, xerr=[[r["ST"] - r["ST_lo"] for r in of], [r["ST_hi"] - r["ST"] for r in of]],
                fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in of], fontsize=8, color=INK)
    ax.set_xlabel("Sobol' index", color=MUTED, fontsize=9)
    ax.set_xlim(left=0)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    _style(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=150)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    return path


def histogram(values: np.ndarray, marks: dict, xlabel: str, title: str, name: str, note: str = "") -> Path:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import matplotlib.transforms as transforms  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    lo, hi = np.percentile(values, [1.0, 97.5])
    lo, hi = min(lo, min(x for x, _ in marks.values())), max(hi, max(x for x, _ in marks.values()))
    beyond = float(((values < lo) | (values > hi)).mean())
    ax.hist(np.clip(values, lo, hi), bins=80, range=(lo, hi), color=BLUE, alpha=0.9, edgecolor="white", lw=0.3)
    blended = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for i, (label, (x, colour)) in enumerate(marks.items()):
        ax.axvline(x, color=colour, lw=1.4, ls="--" if i else "-")
        ax.text(x, 0.80 - 0.09 * i, " " + label, transform=blended, color=colour, fontsize=8, va="top")
    lines = []
    if beyond > 0:
        lines.append(f"{100 * beyond:.1f} % of samples beyond the axis, piled at its ends")
    if note:
        lines.append(note)
    if lines:
        ax.text(0.99, 0.98, "\n".join(lines), transform=ax.transAxes, ha="right", va="top",
                color=MUTED, fontsize=7, wrap=True)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel("samples", color=MUTED, fontsize=9)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    _style(ax)
    fig.tight_layout()
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=150)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    return path


def render_figures(model: Model, sobol_rows: list, Ymc: np.ndarray, check: dict, mc_summary: dict, variant: str) -> list:
    net, cost, capital = ".heat_transport.p_plant_electric_net_mw", "coe_capped", ".costs.concost"
    worst = worst_margin(check)
    tag = "" if variant == "nominal" else f" [{variant}]"
    paths = [
        tornado(sobol_rows, net, "Sobol' indices: net electric power" + tag, named("uq_tornado_net_power", variant)),
        tornado(sobol_rows, cost, f"Sobol' indices: cost of electricity (capped at {COE_CAP:g} $/MWh)" + tag, named("uq_tornado_cost", variant)),
        tornado(sobol_rows, capital, "Sobol' indices: constructed cost" + tag, named("uq_tornado_concost", variant)),
        tornado(sobol_rows, worst, f"Sobol' indices: {cname(worst)} residual" + tag, named(f"uq_tornado_{worst.rsplit('.', 1)[1]}", variant)),
    ]
    valid = valid_rows(model, Ymc)
    n_at = check["port_at_nominal"]
    paths.append(histogram(
        Ymc[valid, model.columns.index(net)],
        {f"target {check['c16_target_mw']:g} MW": (check["c16_target_mw"], ORANGE),
         f"nominal {n_at[net]:.0f} MW": (n_at[net], INK)},
        "net electric power [MW]",
        f"Net electric power over the inputs' uncertainty{tag}: P(< target) = {mc_summary['p_net_below_target']:.2f}",
        named("uq_hist_net_power", variant),
    ))
    coe = Ymc[valid, model.columns.index(".costs.coe")]
    finite_cost = coe < 1e6  # below the 1e21 sentinel of a non-positive net power
    paths.append(histogram(
        coe[finite_cost],
        {f"nominal {n_at['.costs.coe']:.1f} $/MWh": (n_at[".costs.coe"], INK)},
        "cost of electricity [$/MWh]",
        "Cost of electricity over the inputs' uncertainty" + tag,
        named("uq_hist_cost", variant),
        note=f"{100 * (1 - finite_cost.mean()):.1f} % of samples have net power <= 0\n(coe undefined) and are not shown",
    ))
    return paths


# ---------------------------------------------------------------- main


def run(n: int, mc: int, chunk: int, replicates: int, predict: str, seed: int, samples: Path, variant: str) -> None:
    samples.mkdir(parents=True, exist_ok=True)
    setup_began = time.perf_counter()
    timings: dict = {"backend": jax.default_backend(), "variant": variant, "chunk": chunk, "n": n, "mc": mc, "bootstrap_replicates": replicates}
    model, check = build(variant=variant)
    timings["build_s"] = check["build_s"]
    print(json.dumps({k: v for k, v in check.items() if k != "blocking"}, indent=1, default=str))
    print("\n".join(check["blocking"]))
    if model.dropped:
        print("dropped (not boundary inputs of this schedule):", model.dropped)
    k = len(model.inputs)

    # The predictor: d density / d input at the nominal.
    grad, timings["sensitivity_s"] = sensitivity(model)
    model.sensitivity = grad
    print(f"d dene / d x at the nominal ({timings['sensitivity_s']:.1f} s):")
    for j, i in enumerate(model.inputs):
        g = grad[0, j]
        print(f"   {i.path:60} {g: .4e}  scaled {g * model.x0()[j] / model.dene0: .4e}")
    timings["scaled_sensitivity_of_density"] = {i.path: float(grad[0, j] * model.x0()[j] / model.dene0) for j, i in enumerate(model.inputs)}

    began = time.perf_counter()
    U = saltelli(k, n, seed)
    X = model.coordinates(U)
    timings["sampling_saltelli_s"] = time.perf_counter() - began
    print(f"Saltelli: {X.shape[0]} rows ({k} + 2) x {n}, sampled in {timings['sampling_saltelli_s']:.2f} s")

    # Choose the start: a calibration chunk each way (the same rows), keep the one
    # with fewer Newton steps.
    use_predict = None if predict in ("none", "auto") else predict
    if predict == "auto":
        steps_col = len(OUTPUTS) + len(model.inequalities)
        trial = {}
        for p in (None, "linear", "loglinear"):
            r = evaluate(model, X[:chunk], chunk, p, f"calibrate predict={p}")
            valid = valid_rows(model, r["Y"])
            trial[p] = {"first_call_s": r["first_call_s"], "steps_mean": float(r["Y"][:, steps_col].mean()),
                        "steps_max": int(r["Y"][:, steps_col].max()), "unconverged": int((~valid).sum())}
            r2 = evaluate(model, X[:chunk], chunk, p, f"calibrate predict={p} warm")
            trial[p]["warm_s"] = r2["first_call_s"]
        timings["calibration"] = {str(p): v for p, v in trial.items()}
        # Fewest Newton steps among the starts that lose no point.
        best = min((p for p in trial if trial[p]["unconverged"] <= trial[None]["unconverged"]),
                   key=lambda p: (trial[p]["steps_mean"], trial[p]["warm_s"]))
        use_predict = best
        print(f"calibration: {trial}; start kept: {use_predict}")
    timings["predicted_start"] = use_predict
    # Setup: build, sensitivity, sampling and the calibration (three compiles) --
    # everything before the first production evaluation.
    timings["setup_s"] = time.perf_counter() - setup_began

    eval_began = time.perf_counter()
    result = evaluate(model, X, chunk, use_predict, "saltelli")
    Y = result["Y"]
    timings["saltelli_first_call_s"] = result["first_call_s"]
    timings["saltelli_walls_s"] = result["walls"]
    timings["saltelli_calls"] = 1 + len(result["walls"])
    timings["saltelli_evaluations"] = int(Y.shape[0])
    timings["saltelli_evaluation_s"] = time.perf_counter() - eval_began
    np.savez(samples / "uq_saltelli.npz", Y=Y, columns=np.array(model.columns),
             inputs=np.array([i.path for i in model.inputs]), n=n, seed=seed)

    began = time.perf_counter()
    rng = np.random.default_rng(seed + 1)
    Umc = rng.random((mc, k))
    Xmc = model.coordinates(Umc)
    timings["sampling_mc_s"] = time.perf_counter() - began
    eval_began = time.perf_counter()
    result = evaluate(model, Xmc, chunk, use_predict, "mc")
    Ymc = result["Y"]
    timings["mc_first_call_s"] = result["first_call_s"]
    timings["mc_walls_s"] = result["walls"]
    timings["mc_evaluations"] = int(Ymc.shape[0])
    timings["mc_evaluation_s"] = time.perf_counter() - eval_began
    timings["evaluation_only_s"] = timings["saltelli_evaluation_s"] + timings["mc_evaluation_s"]
    np.savez(samples / "uq_mc.npz", Y=Ymc, columns=np.array(model.columns),
             inputs=np.array([i.path for i in model.inputs]), seed=seed)

    render(model, check, timings, Y, n, Ymc, replicates, variant)


def render(model, check, timings, Y, n, Ymc, replicates, variant):
    """Estimators, tables, figures and `out/uq.json` from the evaluations."""
    sobol_rows, sobol_summary = sobol_table(model, Y, n, replicates)
    mc_rows, mc_summary = monte_carlo_table(model, Ymc, check)
    steps_col = len(OUTPUTS) + len(model.inequalities)
    for label, YY in (("saltelli", Y), ("mc", Ymc)):
        valid = valid_rows(model, YY)
        timings[f"{label}_steps_mean"] = float(YY[:, steps_col].mean())
        timings[f"{label}_steps_max"] = int(YY[:, steps_col].max())
        timings[f"{label}_unconverged"] = int((~valid).sum())
        timings[f"{label}_unconverged_fraction"] = float((~valid).mean())
    if "saltelli_walls_s" in timings:
        walls = timings["saltelli_walls_s"] + timings["mc_walls_s"]
        chunk = timings["chunk"]
        total_eval = timings["saltelli_evaluations"] + timings["mc_evaluations"]
        total_wall = sum(walls) + timings["saltelli_first_call_s"] + timings["mc_first_call_s"]
        timings["evaluations_per_s_warm"] = chunk * len(walls) / sum(walls) if walls else None
        timings["evaluations_per_s_including_first_calls"] = total_eval / total_wall
        if "evaluation_only_s" in timings:
            timings["evaluations_per_s_evaluation_only"] = total_eval / timings["evaluation_only_s"]
        timings["warm_wall_per_call_s"] = float(np.median(walls)) if walls else None
        timings["us_per_evaluation_warm"] = 1e6 * float(np.median(walls)) / chunk if walls else None
    render_inputs(model, variant)
    render_sobol(model, sobol_rows, check, variant)
    render_mc(mc_rows, mc_summary, variant)
    figures = render_figures(model, sobol_rows, Ymc, check, mc_summary, variant)
    # The "top 10 for net power and cost" and the dummy's floor, in the JSON too.
    net, cost = ".heat_transport.p_plant_electric_net_mw", "coe_capped"
    tops = {}
    for o in (net, cost, ".costs.concost", worst_margin(check)):
        of = sorted((r for r in sobol_rows if r["output"] == o), key=lambda r: -r["ST"])[:10]
        tops[o] = [{k2: r[k2] for k2 in ("label", "S1", "S1_lo", "S1_hi", "ST", "ST_lo", "ST_hi")} for r in of]
    dummy = {r["output"]: (r["S1"], r["ST"]) for r in sobol_rows if r["input"] == "dummy"}
    payload = {
        "check": check, "timings": timings, "sobol": sobol_summary, "top": tops, "dummy": dummy,
        "mc": mc_summary, "figures": [str(p) for p in figures],
        "inputs": [dataclasses.asdict(i) | {"nominal": (None if i.path not in model.nominal else np.asarray(model.nominal[i.path]).tolist())} for i in model.inputs],
    }
    write_json("uq.py", payload, name=named("uq", variant))
    print(json.dumps({"timings": timings, "sobol": {k2: v for k2, v in sobol_summary.items() if k2 != "variance"},
                      "mc": mc_summary, "dummy": dummy}, indent=1, default=str))
    for o, rows in tops.items():
        print(f"\ntop 10 by ST for {cname(o)}:")
        for r in rows:
            print(f"   {r['label']:40} S1 {r['S1']: .3f} [{r['S1_lo']: .3f}, {r['S1_hi']: .3f}]   ST {r['ST']: .3f} [{r['ST_lo']: .3f}, {r['ST_hi']: .3f}]")


def rerender(samples: Path, replicates: int, variant: str) -> None:
    """Tables and figures from the saved evaluations; no GPU, the model rebuilt on
    the CPU only for its names and the nominal check.
    """
    model, check = build(variant=variant)
    s = np.load(samples / "uq_saltelli.npz")
    m = np.load(samples / "uq_mc.npz")
    if tuple(s["columns"]) != model.columns or tuple(s["inputs"]) != tuple(i.path for i in model.inputs):
        raise ValueError("the saved samples were made with a different table; rerun")
    previous = OUT / f"{named('uq', variant)}.json"
    timings = json.loads(previous.read_text()).get("timings", {}) if previous.exists() else {}
    render(model, check, timings, s["Y"], int(s["n"]), m["Y"], replicates, variant)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    def option(name, default, cast=int):
        return cast(argv[argv.index(name) + 1]) if name in argv else default

    smoke = "--smoke" in argv
    variant = option("--variant", "nominal", str)
    if variant not in VARIANTS:
        raise SystemExit(f"--variant {variant!r}: one of {sorted(VARIANTS)}")
    n = option("--n-base", 64 if smoke else 8192)  # Saltelli's N; (k + 2) N evaluations
    if n & (n - 1):
        raise SystemExit(f"--n-base {n} must be a power of two (Sobol' balance)")
    mc = option("--n-mc", 512 if smoke else 100_000)
    chunk = option("--chunk", 256 if smoke else 16384)
    replicates = option("--boot", 20 if smoke else 200)
    predict = option("--predict", "auto", str)  # auto | none | linear | loglinear
    seed = option("--seed", 0)
    default_samples = SAMPLES / (("smoke" if smoke else "full") + ("" if variant == "nominal" else f"_{variant}"))
    samples = option("--samples", default_samples, Path)
    if "--render" in argv:
        rerender(samples, replicates, variant)
        return 0
    run(n, mc, chunk, replicates, predict, seed, samples, variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
