"""Optimisation under uncertainty (OUU) on the closed stellarator MDA: `uq.py`'s
model (the power balance `c2` closed inside the MDA by the plasma density, 26 uncertain
boundary inputs), with the design freed again and a robust problem stated over one
fixed Sobol' sample set (sample-average approximation, common random numbers).

**Formulation.** Design `x`: the run's `ixc` design places that are neither closed
(`nd_plasma_electrons_vol_avg`, answered by the root find) nor uncertain (`hfact`, an
input of `uq.INPUTS`) -- six values: `b_plasma_toroidal_on_axis`, `rmajor`,
`temp_plasma_electron_vol_avg_kev`, `t_tf_superconductor_quench`,
`f_a_tf_turn_cable_copper`, `f_nd_alpha_thermal_electron`, bounded by the input file's
`boundl` / `boundu`. Uncertain `theta`: `uq.INPUTS` with `uq.py`'s distributions
(`--inputs all`), or the physics entries alone with the four economic ones
(`f_t_plant_available`, `life_plant`, `discount_rate`, `ucsc`) and the dummy held at
their nominal (`--inputs physics`, the default); one scrambled Sobol' set of N points
drawn once and kept fixed, **plus the nominal point as row N**, under this script's own
belief table (`belief_table`: `--table new`, the default, has hfact lognormal(`--hfact-sigma`,
0.10) and the tungsten fraction and field ripple uniform +-20 %; `--table old` is
`uq.INPUTS` as stated), so every batched call
also returns the nominal design's answer. For every sample the closed MDA runs (one
`jax.vmap` with the sample axis on theta and the Newton starts only, `x` shared) and
returns cost, the inequality residuals and the closing Newton's verdict.

    minimise   f(x)   = F[coe](x)  +  FAILED_F failed(x)
    subject to CVaR_alpha[g_j] <= 0     for the twelve inequalities j (+ c16 with --with-c16)
               failed(x)       <= eps

with `F` one of `--objective`: `mean` (the default), E[coe | converged and net electric
power > 0]; `median`, the sample median of coe over converged samples (a sort, so
differentiable almost everywhere); `levelised`, `sum_i w_i coe_i kwh_i / sum_i w_i kwh_i`,
the ratio of expectations, `kwh_i = net_mw_i f_t_plant_available_i` guarded below by the
cost model's own `max(kwhpy, 1e-10)`; `nominal`, coe at the nominal inputs (row N: no
batch in the objective, the batch only in the constraints). `w_i` = 1 where the closing
Newton converged and every output is finite, 0 otherwise; `CVaR_alpha` the
Rockafellar-Uryasev sample estimator, the mean of the worst `ceil((1 - alpha) N)`
converged samples (`lax.top_k`); `failed` the fraction of samples masked out -- and **a
failed sample is infeasible**: it enters every CVaR at `FAILED_G` and costs `FAILED_F / N`,
because SLSQP's first-iteration merit is `f` alone and a design where nothing converges
must not read as free. `coe` is `1e21` wherever the net electric power is non-positive
(the cost model's guard), so `E[coe]` unconditioned is an indicator scaled by 1e21; the
`mean` measure conditions on net power > 0 and the `levelised` one weights by the energy.

**Differentiation.** `--jac fwd` (the default) is `jax.jacfwd` of the statistics vector
(1 + n_g + 1 outputs, 6 inputs): the scaling sweep measured it at 2x the forward call
and 4x its memory, against 13x and 15x for `jax.jacrev`, which with six inputs is the
wrong mode. Both work as they stand: every loop in the closed MDA carries an implicit
derivative (the closing Newton under `lax.custom_root`, the Picards and the coil root
find under optimistix's `ImplicitAdjoint`). `--chunk C` runs the sample set in chunks of
C under `lax.map`, each chunk's body under `jax.checkpoint`, which bounds reverse mode's
memory to one chunk's residuals; the statistics are computed on the concatenated rows,
so chunked and unchunked are the same function.

**Optimiser.** `scipy.optimize.minimize(method='SLSQP')` on the jitted statistics and
their Jacobian, value and Jacobian from one compiled call, `x` scaled by its start, and
**with move limits**: SLSQP has no trust region, and unboxed its first QP step on this
problem lands on a bound corner where every sample's Newton stalls; so each SLSQP call is
boxed to +-`--move-limit` of its start and re-centred while its answer sits on a face of
the box. A call that ends infeasible halves the box and restarts from the best feasible
point. The run has converged when SLSQP reports success strictly inside its box, or a
re-centred call moves by less than `--tol` (relative, in scaled units). Warm starts:
every sample's Newton starts from its converged root at the **incumbent** (SLSQP's
accepted iterate, via the `callback`), never from the last trial point -- a line-search
trial far out would otherwise hand garbage roots to the next, nearer trial, which then
fails to converge for no reason of the design's (the first version did exactly that).
`c16` (net electric power against its 1000 MW target) is a CVaR constraint only with
`--with-c16`: with it the robust problem is infeasible.

**Evidence.** `--evidence` evaluates the deterministic and the robust design on the fixed
sample set and on a fresh Sobol' set (`--fresh-seed`), from cold starts and once more
warm, and writes per-constraint violation fractions, the failed and any-violation
fractions, per-sample feasibility (converged, net > 0, every g <= 0) with coe statistics
over the feasible samples, coe and net power percentiles, and the design deltas with the
active bounds -- the per-sample columns beside it as `.npz` for the histograms.

    PY=~/miniconda/envs/process_port/bin/python; export JAX_PLATFORMS=cpu
    $PY paper_tests/ouu.py --scaling [--sizes 64,256,1024,4096] [--modes fwd,jacrev,jacfwd] [--fresh]
    $PY paper_tests/ouu.py --smoke [--n 256] [--alpha 0.9] [--max-iter 200] [--objective mean]
                           [--inputs physics] [--jac fwd] [--chunk C] [--tol 1e-4] [--evidence] [--name STEM] [--tag T]
    $PY paper_tests/ouu.py --decompose [--n 256] [--inputs all] [--robust out/ouu_smoke.json] [--table old] [--name STEM]
    $PY paper_tests/ouu.py --evidence --robust out/ouu_alpha0.9.json --alpha 0.9 [--fresh-seed 1]
    $PY paper_tests/ouu.py --plot [--from paper_tests/out_cluster]
    $PY paper_tests/ouu.py --measure --n 256 --mode jacrev    # one configuration, one process

Outputs: `out/ouu_scaling.csv`, `out/ouu_smoke.json` (or `--name`), `out/ouu_decomposition.json`,
`out/ouu_evidence_<alpha><tag>.{json,npz}`, `out/ouu_price_of_robustness.png`,
`out/ouu_histograms_0.9.png`, `out/ouu_logs/*.log`.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)  # before any array: PROCESS is float64

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import uq  # noqa: E402
from common import OUT, write_csv, write_json  # noqa: E402
from cottax.names import PathMap  # noqa: E402
from cottax.problem import Converged, Steps  # noqa: E402
from jax import lax  # noqa: E402
from scipy.stats import qmc  # noqa: E402

ALPHA = 0.9
EPS_FAILED = 0.02
"""Largest tolerated fraction of samples whose closed MDA did not converge."""
FAILED_G = 10.0
"""A failed sample is infeasible: it enters every CVaR at this (normalised) violation,
so a design where the closed MDA stops converging cannot look feasible. A constant, so
it carries no gradient; what pulls the design back is the converged samples' own."""
G_TOL = 2.0e-3
"""A sample violates a constraint at `g > G_TOL` (normalised: 0.2 % of the limit), not
`g > 0`: a constraint an optimiser holds active sits a hair above zero and would
otherwise read as violated in every sample -- the radial build `c83` at +2e-9 after
SLSQP, and PROCESS's own converged point has `c24` at +1.5e-3 and `c35` at +1.6e-4
(VMCON's slack), so the deterministic design is feasible only to this tolerance."""
FAILED_F = 1.0e3
"""Cost added to `f` per unit failed fraction (and `f` itself where nothing converged):
about ten times the deterministic coe, so a line search rejects a point whose samples
fail rather than reading their absence as a cheaper plant."""

C16 = "^cond.constraints.c16"
TE = ".physics.temp_plasma_electron_vol_avg_kev"
LOGS = OUT / "ouu_logs"

OBJECTIVES = ("mean", "median", "levelised", "nominal", "rated")
"""`rated`: `mean_i(annual cost_i) / (rated energy)`, the expected annual cost per
MWh the plant is rated for (`p_plant_electric_net_required_mw` x nominal availability)
-- the expectation over every converged sample of a finite, differentiable quantity,
which `mean` (coe ~ 1 / net) is not; a shortfall is the CVaR constraint on `c16`'s."""
ECONOMIC = (".costs.f_t_plant_available", ".costs.life_plant", ".costs.discount_rate", ".costs.ucsc")
"""`uq.INPUTS` entries `--inputs physics` holds at their nominal (with the dummy)."""
INPUT_SETS = ("all", "physics")
JAC_MODES = ("fwd", "rev")
TABLES = ("new", "old", "build")
HFACT_SIGMA = 0.10
PAIRINGS = {
    "one": uq.PAIRING,
    "two": {"^cond.constraints.c2": ".physics.nd_plasma_electrons_vol_avg",
            "^cond.constraints.c16": ".physics.f_nd_alpha_thermal_electron"},
    "te": {"^cond.constraints.c2": ".physics.nd_plasma_electrons_vol_avg",
           "^cond.constraints.c16": ".physics.temp_plasma_electron_vol_avg_kev"},
}
"""`--pairing`: which equalities the MDA closes per sample, and by what. `one` is
`uq.PAIRING` (the power balance by the density); `two` adds the net electric power
(`c16`, an equality in the file) closed by the thermal alpha fraction, the
`close_conditions.PAIRINGS` choice -- the two root finds share a cycle and are
flattened into one square problem over both. With `two` the operating point is
recourse: per sample the density and the alpha fraction are what the realised physics
requires, and only `temp_plasma_electron_vol_avg_kev` stays a shared operating knob."""
BUILD_LEAVES = (
    ".fwbs.dr_fw_wall", ".tfcoil.dx_tf_wp_insulation", ".fwbs.fhole", ".constraints.f_j_tf_wp_critical_max",
)
"""Sampled leaves the `build` table holds at their nominal: the three manufacturing
scatters (`decision_kinds.md`: build decisions sampled as beliefs) and the designer's
current margin the winding-pack sizing rule reads (`output_kinds.md` §1a).
`paper_tests/stage_check.py` lists every sampled leaf that reaches a claimed build
output: these four are the ones behind the coil, the radial build and the first wall,
so with them fixed every sample has one build, and what still varies per sample is
physics, operation and cost -- and the three rules the hand-off leaves per sample (the
component lifetimes, the vacuum and heat-exchanger counts)."""
RELATIVE_20 = (".impurity_radiation.f_nd_impurity_electron_array[13]", ".stellarator.bmn")
"""`uq.INPUTS` entries the `new` table draws uniform in nominal x [0.8, 1.2] rather
than lognormal(ln 2 / 2)."""


def belief_table(table: str = "new", hfact_sigma: float = HFACT_SIGMA) -> tuple:
    """This script's own belief table over `uq.INPUTS`' paths: the `old` table is
    `uq.INPUTS` as stated (hfact lognormal(0.15), tungsten and ripple lognormal(ln 2 / 2));
    the `new` one has hfact lognormal(`hfact_sigma`) and tungsten and ripple relative
    +-20 %. `uq.py` is not edited: `build` swaps the model's inputs for these."""
    if table not in TABLES:
        raise ValueError(f"--table {table!r}; one of {TABLES}")
    if table == "old":
        return uq.INPUTS
    rows = []
    for i in uq.INPUTS:
        if table == "build" and i.path in BUILD_LEAVES:
            continue
        if i.path == ".physics.hfact":
            i = dataclasses.replace(i, a=hfact_sigma)
        elif i.path in RELATIVE_20:
            i = dataclasses.replace(i, kind="relative", a=0.20, b=0.0, note=i.note + " (relative +-20 %)")
        rows.append(i)
    return tuple(rows)


def hfact_belief(sigma: float) -> dict:
    """What lognormal(`sigma`) says of the confinement factor: the 90 % interval and
    the mean over the worst (lowest, and highest) decile."""
    from scipy.stats import norm  # noqa: PLC0415

    z = norm.ppf(0.9)
    return {
        "sigma": sigma,
        "interval_90": [float(math.exp(-norm.ppf(0.95) * sigma)), float(math.exp(norm.ppf(0.95) * sigma))],
        "lowest_decile_mean": float(math.exp(sigma**2 / 2) * norm.cdf(-z - sigma) / 0.1),
        "highest_decile_mean": float(math.exp(sigma**2 / 2) * norm.sf(z - sigma) / 0.1),
    }


# ---------------------------------------------------------------- the problem


@dataclasses.dataclass(frozen=True)
class RecourseBound:
    """A closing variable's bound as a per-sample inequality: the recourse exists only
    where the root the closure found is a physical operating point (an alpha fraction
    below zero delivers the rated power on paper and nowhere else). Normalised by the
    bound's range, so it sits in the CVaR beside the graph's own inequalities."""

    var: object
    side: str
    lower: float
    upper: float

    @property
    def spelling(self) -> str:
        return f"^bound{self.var.spelling}.{self.side}"

    def residual(self, value):
        span = self.upper - self.lower
        return (self.lower - value) / span if self.side == "lower" else (value - self.upper) / span


@dataclasses.dataclass
class Ouu:
    model: uq.Model
    check: dict
    design: tuple
    """The six design places, in `ixc` order."""
    ixc: tuple
    """Their PROCESS `ixc` numbers."""
    x0: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    constraints: tuple
    """Inequality VarPaths under CVaR: the twelve, plus `c16` with `--with-c16`."""
    columns: tuple
    """Per-sample output columns: coe, net, f_avail, concost, *constraints, steps,
    converged, *unknowns."""
    alpha: float
    n: int
    seed: int
    rated_mw: float
    """`p_plant_electric_net_required_mw`, `c16`'s target."""
    alpha16: float | None
    te_grid: np.ndarray | None
    """`--te-recourse K`: T_e is not a design place but chosen per sample from these K
    values -- the cheapest feasible one (converged, net > 0, every g <= 0), else the
    least infeasible -- so the operating temperature is recourse and the outer
    derivative is the envelope theorem's, at the chosen point (`stop_gradient` on the
    choice)."""
    inputs: str
    objective: str
    table: str
    hfact_sigma: float
    pairing: str
    Theta: np.ndarray
    """Coordinates [N + 1, k], `uq.Model.coordinates` of the scrambled Sobol' set with
    the nominal point as the last row."""
    theta: PathMap
    """The batched env values of the uncertain inputs ([N + 1, ...] each)."""
    starts0: np.ndarray
    """[N + 1, unknowns]: every row started from the nominal root."""
    build_s: float

    @property
    def n_g(self) -> int:
        return len(self.constraints)

    @property
    def m(self) -> int:
        """How many worst samples a CVaR averages."""
        return max(1, int(math.ceil((1.0 - self.alpha) * self.n)))

    @property
    def m16(self) -> int:
        """How many worst samples `c16`'s CVaR averages: `alpha16`'s, where one is given
        (a shortfall of the rated power is a softer requirement than a physics limit),
        else `alpha`'s."""
        alpha = self.alpha if self.alpha16 is None else self.alpha16
        return max(1, int(math.ceil((1.0 - alpha) * self.n)))


def build(n: int, alpha: float = ALPHA, seed: int = 0, with_c16: bool = False,
          inputs: str = "physics", objective: str = "mean", table: str = "new",
          hfact_sigma: float = HFACT_SIGMA, pairing: str = "one", alpha16: float | None = None,
          te_recourse: int = 0, te_range: tuple | None = None) -> Ouu:
    began = time.perf_counter()
    if inputs not in INPUT_SETS:
        raise ValueError(f"--inputs {inputs!r}; one of {INPUT_SETS}")
    if objective not in OBJECTIVES:
        raise ValueError(f"--objective {objective!r}; one of {OBJECTIVES}")
    if pairing not in PAIRINGS:
        raise ValueError(f"--pairing {pairing!r}; one of {tuple(PAIRINGS)}")
    if with_c16 and pairing != "one":
        raise ValueError(f"--with-c16 states c16 as a CVaR constraint; --pairing {pairing} closes it per sample")
    model, check = uq.build(PAIRINGS[pairing])
    by_path = {i.path: i for i in belief_table(table, hfact_sigma)}
    model.inputs = tuple(by_path[i.path] for i in model.inputs if i.path in by_path)
    if inputs == "physics":
        model.inputs = tuple(i for i in model.inputs if i.path not in ECONOMIC and i.path != "dummy")
    uncertain = {model.var_of[i.path] for i in model.inputs if i.path != "dummy"}
    design = tuple(v for v in model.built.design if v not in uncertain)
    from functional_process.cottax import sand  # noqa: PLC0415

    ixc_of = {sand.iteration_variable_path(i): i for i in model.live.reference.ixc}
    bounds = {v: (lo, hi) for v, lo, hi in model.live.reference.bounds}
    te_grid = None
    if te_recourse:  # T_e per sample: off the design, onto a grid the recourse picks from
        te = model.var_of[TE]
        design = tuple(v for v in design if v != te)
        lo, hi = te_range if te_range else bounds[te]
        te_grid = np.linspace(lo, hi, te_recourse)
    x0 = np.array([float(np.asarray(model.point[v])) for v in design])
    lower = np.array([bounds[v][0] for v in design])
    upper = np.array([bounds[v][1] for v in design])
    ineq = tuple(model.inequalities)
    constraints = (*ineq, model.var_of[C16]) if with_c16 else ineq
    for closing in model.built.pairings.values():  # the recourse's own bounds, where it has any
        if closing in bounds and len(model.built.pairings) > 1:
            lo, hi = bounds[closing]
            constraints += (RecourseBound(closing, "lower", lo, hi), RecourseBound(closing, "upper", lo, hi))
    place = next(iter(model.built.places.values()))
    columns = (
        model.var_of[".costs.coe"], model.var_of[".heat_transport.p_plant_electric_net_mw"],
        model.var_of[".costs.f_t_plant_available"], model.var_of[".costs.concost"],
        *constraints, Steps.name_for(place), Converged.name_for(place), *model.unknowns,
    )
    Theta, theta, starts0 = sample(model, n, seed)
    return Ouu(
        model=model, check=check, design=design, ixc=tuple(ixc_of[v] for v in design), x0=x0, lower=lower, upper=upper,
        constraints=constraints, columns=columns, alpha=alpha, n=n, seed=seed, rated_mw=check["c16_target_mw"], alpha16=alpha16,
        te_grid=te_grid,
        inputs=inputs, objective=objective,
        table=table, hfact_sigma=hfact_sigma if table != "old" else 0.15, pairing=pairing,
        Theta=Theta, theta=theta, starts0=starts0, build_s=time.perf_counter() - began,
    )


def sample(model: uq.Model, n: int, seed: int) -> tuple[np.ndarray, PathMap, np.ndarray]:
    """One scrambled Sobol' set of `n` points over every input of `model` (a dummy
    column is drawn and ignored), the nominal point appended as row `n`: coordinates,
    the batched env, and every row's start at the nominal root."""
    k = len(model.inputs)
    U = qmc.Sobol(d=k, scramble=True, seed=seed).random(n)
    Theta = np.vstack([model.coordinates(U), model.x0()[None, :]])
    return Theta, theta_env(model, Theta), np.tile(model.u0, (n + 1, 1))


def theta_env(model: uq.Model, Theta: np.ndarray) -> PathMap:
    """`{uncertain VarPath: [rows, ...] value}` for coordinates `Theta` [rows, k]."""
    values = {}
    for j, i in enumerate(model.inputs):
        if i.path == "dummy":
            continue
        values[model.var_of[i.path]] = i.value(jnp.asarray(Theta[:, j]), model.nominal[i.path])
    return PathMap(values)


# ---------------------------------------------------------------- the functions


def make(ouu: Ouu, jac: str = "fwd", chunks: int | None = None) -> dict:
    """The jittable functions over `(x_flat, theta, starts)`; `theta` and `starts`
    are arguments rather than closed-over constants so that nothing of size N is baked
    into an executable. Every batched array has N + 1 rows, the nominal point last.
    """
    if jac not in JAC_MODES:
        raise ValueError(f"--jac {jac!r}; one of {JAC_MODES}")
    model = ouu.model
    run = model.built.problem.traceable.run
    point = dict(model.point.items())
    design = ouu.design
    guesses = [model.guesses[u] for u in model.unknowns]
    columns = ouu.columns
    n_g, m, objective = ouu.n_g, ouu.m, ouu.objective
    m16 = ouu.m16
    i16 = next((i for i, c in enumerate(ouu.constraints) if c.spelling == C16), None)
    c_coe, c_net, c_avail, c_concost = 0, 1, 2, 3
    c_g = slice(4, 4 + n_g)
    c_steps, c_conv = 4 + n_g, 5 + n_g
    c_u = slice(6 + n_g, 6 + n_g + len(guesses))
    tiny = 1.0e-10 / (1.0e3 * 24.0 * 365.2425)
    """The cost model's `max(kwhpy, 1e-10)` guard, in MW-years of net energy."""
    rated_kwh = ouu.rated_mw * float(np.asarray(point[model.var_of[".costs.f_t_plant_available"]]))
    """The rated net energy in the same unit: `p_plant_electric_net_required_mw` at the
    nominal availability (an economic input, at its nominal in every table)."""

    te_var, te_grid = model.var_of[TE], ouu.te_grid

    def at_point(x_flat, theta_row, start_row, te=None):
        values = dict(point)
        values.update(theta_row.items())
        values.update(zip(design, [x_flat[j] for j in range(len(design))], strict=True))
        values.update(zip(guesses, [start_row[j] for j in range(len(guesses))], strict=True))
        if te is not None:
            values[te_var] = te
        out = run(PathMap(values))
        return jnp.stack([
            jnp.asarray(c.residual(out[c.var]) if isinstance(c, RecourseBound) else out[c], dtype=jnp.float64).reshape(())
            for c in columns
        ])

    def one(x_flat, theta_row, start_row):
        return at_point(x_flat, theta_row, start_row)

    def choose(rows):
        """One sample's K rows -> the row at its recourse: the cheapest feasible T_e,
        else the least infeasible one."""
        conv = (rows[:, c_conv] > 0.5) & jnp.all(jnp.isfinite(rows[:, :c_steps]), axis=1)
        g = rows[:, c_g]
        feasible = conv & (rows[:, c_net] > 0) & jnp.all(g <= G_TOL, axis=1)
        worst = jnp.where(conv, jnp.max(g, axis=1), FAILED_G)
        score = jnp.where(feasible, rows[:, c_coe], 1.0e6 + worst)
        k = lax.stop_gradient(jnp.argmin(score))
        return rows[k]

    def vmapped(x_flat, theta, starts):
        if te_grid is None:
            return jax.vmap(one, in_axes=(None, 0, 0))(x_flat, theta, starts)
        # One flat batch over (sample, T_e) pairs -- a vmap inside a vmap trips the
        # solver's error branch at lowering -- then the recourse picks per sample.
        k, n = len(te_grid), starts.shape[0]
        theta_k = jax.tree_util.tree_map(lambda a: jnp.repeat(a, k, axis=0), theta)
        starts_k = jnp.repeat(starts, k, axis=0)
        te_k = jnp.tile(jnp.asarray(te_grid), n)
        rows = jax.vmap(at_point, in_axes=(None, 0, 0, 0))(x_flat, theta_k, starts_k, te_k)
        return jax.vmap(choose)(rows.reshape(n, k, -1))

    def run_batch(x_flat, theta, starts, chunks: int | None = chunks):
        """[N + 1, columns]: every row's closed MDA at design `x_flat`; with `chunks`
        the N samples go through `lax.map` in that many checkpointed pieces and the
        nominal row on its own."""
        if not chunks:
            return vmapped(x_flat, theta, starts)
        n = starts.shape[0] - 1
        if n % chunks:
            raise ValueError(f"N = {n} is not a multiple of chunks = {chunks}")
        size = n // chunks
        theta_s = jax.tree_util.tree_map(lambda a: a[:-1].reshape(chunks, size, *a.shape[1:]), theta)
        theta_n = jax.tree_util.tree_map(lambda a: a[-1], theta)
        starts_s = starts[:-1].reshape(chunks, size, -1)

        @jax.checkpoint
        def body(chunk):
            th, st = chunk
            return vmapped(x_flat, th, st)

        Ys = lax.map(body, (theta_s, starts_s)).reshape(n, -1)
        last = vmapped(x_flat, jax.tree_util.tree_map(lambda a: a[None], theta_n), starts[-1:])
        return jnp.concatenate([Ys, last], axis=0)

    def measures(Y_all):
        """Every objective measure and the constraints' CVaRs from the rows."""
        Y, Yn = Y_all[:-1], Y_all[-1]
        finite = jnp.all(jnp.isfinite(Y[:, : c_steps]), axis=1)
        w = jnp.where((Y[:, c_conv] > 0.5) & finite, 1.0, 0.0)
        sum_w = jnp.sum(w)
        count = jnp.asarray(Y.shape[0], float)
        failed = 1.0 - sum_w / count
        coe, net = Y[:, c_coe], Y[:, c_net]
        kwh = jnp.maximum(net * Y[:, c_avail], tiny)              # proportional to kwhpy
        annual = jnp.where(w > 0, coe, 0.0) * kwh                  # finite: coe = 1e9 ann / kwhpy
        levelised = jnp.sum(w * annual) / jnp.maximum(jnp.sum(w * kwh), 1.0e-10)
        rated = jnp.sum(w * annual) / jnp.maximum(sum_w, 1.0) / rated_kwh
        positive = (w > 0) & (net > 0)
        n_pos = jnp.sum(jnp.where(positive, 1.0, 0.0))
        mean = jnp.where(n_pos > 0, _masked_mean(coe, positive), FAILED_F)
        median = _median(coe, w > 0)
        nominal_ok = (Yn[c_conv] > 0.5) & jnp.all(jnp.isfinite(Yn[: c_steps]))
        nominal = jnp.where(nominal_ok, Yn[c_coe], FAILED_F)
        g = jnp.where(w[:, None] > 0, Y[:, c_g], FAILED_G)         # [N, n_g]
        g_cvar = jnp.mean(lax.top_k(g.T, m)[0], axis=1)
        if i16 is not None and m16 != m:
            g_cvar = g_cvar.at[i16].set(jnp.mean(lax.top_k(g[:, i16], m16)[0]))
        value = {"mean": mean, "median": median, "levelised": levelised, "nominal": nominal, "rated": rated}[objective]
        f = jnp.where(sum_w > 0, value, FAILED_F) + FAILED_F * failed
        u = jnp.where(w[:, None] > 0, Y[:, c_u], jnp.nan)
        conv = w > 0
        diagnostics = {
            "steps_mean": jnp.mean(Y[:, c_steps]), "steps_max": jnp.max(Y[:, c_steps]),
            "coe_nominal": Yn[c_coe], "net_nominal_mw": Yn[c_net], "levelised": levelised, "rated": rated,
            "mean_coe_where_converged_and_positive_net": mean, "median_coe_where_converged": median,
            "mean_concost": _masked_mean(Y[:, c_concost], conv),
            "mean_net_mw": _masked_mean(net, conv),
            "p_net_nonpositive": jnp.mean(jnp.where(net <= 0, 1.0, 0.0)),
            "p_violated": jnp.mean(jnp.where(Y[:, c_g] > 0, 1.0, 0.0), axis=0),
            "mean_g": jnp.mean(Y[:, c_g], axis=0),
        }
        return f, g_cvar, failed, u, diagnostics

    def statistics_full(x_flat, theta, starts):
        Y_all = run_batch(x_flat, theta, starts)
        f, g_cvar, failed, u, diagnostics = measures(Y_all)
        u_all = jnp.concatenate([u, Y_all[-1:, c_u]], axis=0)
        u_next = jnp.where(jnp.isnan(u_all), starts, u_all)
        return f, g_cvar, failed, lax.stop_gradient(u_next), diagnostics

    def statistics(x_flat, theta, starts):
        """`(f, g_cvar, failed_fraction)` at design `x_flat`."""
        f, g_cvar, failed, _u, _d = statistics_full(x_flat, theta, starts)
        return f, g_cvar, failed

    def statistics_vec(x_flat, theta, starts):
        f, g_cvar, failed = statistics(x_flat, theta, starts)
        return jnp.concatenate([f[None], g_cvar, failed[None]])

    def statistics_chunked_vec(x_flat, theta, starts, chunks: int):
        f, g_cvar, failed, _u, _d = measures(run_batch(x_flat, theta, starts, chunks))
        return jnp.concatenate([f[None], g_cvar, failed[None]])

    def value_jac_starts(x_flat, theta, starts):
        """One compiled call for SLSQP: the statistics vector, its Jacobian (forward
        or reverse, `jac`; one trace of the model, primal reused), the next starts,
        diagnostics.
        """

        def twice(x):
            f, g_cvar, failed, u_next, diagnostics = statistics_full(x, theta, starts)
            out = jnp.concatenate([f[None], g_cvar, failed[None]])
            return out, (out, u_next, diagnostics)

        differentiate = jax.jacfwd if jac == "fwd" else jax.jacrev
        jacobian, (out, u_next, diagnostics) = differentiate(twice, has_aux=True)(x_flat)
        return out, jacobian, u_next, diagnostics

    return {
        "run_batch": run_batch,
        "statistics": statistics,
        "statistics_vec": statistics_vec,
        "statistics_full": statistics_full,
        "statistics_chunked_vec": statistics_chunked_vec,
        "value_jac_starts": value_jac_starts,
        "layout": {"c_coe": c_coe, "c_net": c_net, "c_avail": c_avail, "c_concost": c_concost, "c_g": (4, 4 + n_g),
                   "c_steps": c_steps, "c_conv": c_conv, "c_u": (6 + n_g, 6 + n_g + len(guesses)), "tiny": tiny},
    }


def _masked_mean(v, mask):
    w = jnp.where(mask, 1.0, 0.0)
    return jnp.sum(w * jnp.where(mask, v, 0.0)) / jnp.maximum(jnp.sum(w), 1.0)


def _median(v, mask):
    """The sample median of `v` over `mask` (the two middle values averaged): a sort,
    so differentiable wherever the order is strict."""
    s = jnp.sort(jnp.where(mask, v, jnp.inf))
    k = jnp.sum(jnp.where(mask, 1, 0))
    lo = s[jnp.maximum((k - 1) // 2, 0)]
    hi = s[jnp.maximum(k // 2, 0)]
    return jnp.where(k > 0, 0.5 * (lo + hi), FAILED_F)


# ---------------------------------------------------------------- measurement


MODES = ("fwd", "jacrev", "jacfwd", "grad", "jacrev_chunked")
"""`fwd`: `statistics`; `jacrev` / `jacfwd`: its Jacobian (1 + n_g + 1 outputs, 6
inputs); `grad`: `jax.grad` of `f` alone (one cotangent); `jacrev_chunked`:
`jacrev` of the chunked statistics in chunks of `--chunk`."""


def program(fns: dict, mode: str, chunk: int | None):
    if mode == "fwd":
        return jax.jit(fns["statistics_vec"])
    if mode == "jacrev":
        return jax.jit(jax.jacrev(fns["statistics_vec"]))
    if mode == "jacfwd":
        return jax.jit(jax.jacfwd(fns["statistics_vec"]))
    if mode == "grad":
        return jax.jit(jax.grad(lambda x, th, st: fns["statistics"](x, th, st)[0]))
    if mode == "jacrev_chunked":
        def chunked(x, th, st):
            return fns["statistics_chunked_vec"](x, th, st, (st.shape[0] - 1) // chunk)
        return jax.jit(jax.jacrev(chunked))
    raise ValueError(mode)


def executable_memory(fn, *args) -> dict:
    """XLA's own accounting of the compiled executable: the temporaries it holds
    during one call (`temp_mb`, where reverse mode's residuals live), its arguments,
    outputs and code -- separable from the process's RSS, which the compiler's own
    working set dominates on the CPU."""
    try:
        analysis = fn.lower(*args).compile().memory_analysis()
    except Exception as failure:  # noqa: BLE001
        return {"memory_analysis": f"{type(failure).__name__}: {failure}"[:120]}
    if analysis is None:
        return {}
    mb = 2.0**20
    return {
        "temp_mb": analysis.temp_size_in_bytes / mb,
        "argument_mb": analysis.argument_size_in_bytes / mb,
        "output_mb": analysis.output_size_in_bytes / mb,
        "code_mb": analysis.generated_code_size_in_bytes / mb,
    }


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def device_peak_mb() -> float | None:
    try:
        stats = jax.local_devices()[0].memory_stats()
        return None if not stats else stats.get("peak_bytes_in_use", 0) / 2**20
    except Exception:  # noqa: BLE001
        return None


def measure(n: int, mode: str, chunk: int | None, repeats: int = 3, alpha: float = ALPHA, seed: int = 0,
            inputs: str = "all", objective: str = "levelised") -> dict:
    """One configuration in this process: build, compile, time, peak RSS."""
    row = {"N": n, "mode": mode, "backend": jax.default_backend(), "chunk": chunk if mode == "jacrev_chunked" else None}
    ouu = build(n, alpha=alpha, seed=seed, inputs=inputs, objective=objective)
    fns = make(ouu)
    row["build_s"] = ouu.build_s
    row["rss_after_build_mb"] = rss_mb()
    fn = program(fns, mode, chunk)
    x0 = jnp.asarray(ouu.x0)
    starts = jnp.asarray(ouu.starts0)
    began = time.perf_counter()
    out = jax.block_until_ready(fn(x0, ouu.theta, starts))
    row["first_call_s"] = time.perf_counter() - began
    row["rss_after_compile_mb"] = rss_mb()
    row.update(executable_memory(fn, x0, ouu.theta, starts))
    walls = []
    for _ in range(repeats):
        began = time.perf_counter()
        out = jax.block_until_ready(fn(x0, ouu.theta, starts))
        walls.append(time.perf_counter() - began)
    row["warm_s"] = min(walls)
    row["warm_us_per_sample"] = 1e6 * min(walls) / n
    row["rss_peak_mb"] = rss_mb()
    row["rss_increment_mb"] = row["rss_peak_mb"] - row["rss_after_build_mb"]
    row["device_peak_mb"] = device_peak_mb()
    arr = np.asarray(out)
    row["output_shape"] = list(arr.shape)
    row["finite"] = bool(np.all(np.isfinite(arr)))
    if mode == "fwd":
        row["f"] = float(arr[0])
        row["max_g_cvar"] = float(arr[1:-1].max())
        row["failed_fraction"] = float(arr[-1])
    elif mode in ("jacrev", "jacfwd", "jacrev_chunked"):
        row["max_abs_jac"] = float(np.abs(arr).max())
        row["jac_f"] = arr[0].tolist()
    elif mode == "grad":
        row["jac_f"] = arr.tolist()
    return row


def scaling(sizes, modes, chunk: int, timeout_s: float, python: str, fresh: bool = False) -> list[dict]:
    """Every (N, mode) in a fresh subprocess, rows collected into `out/ouu_scaling.csv`.
    Rows already in `out/ouu_scaling.json` with status `ok` are kept and their
    configurations skipped unless `fresh`, so a sweep can be resumed or extended."""
    LOGS.mkdir(parents=True, exist_ok=True)
    rows = []
    previous = OUT / "ouu_scaling.json"
    if not fresh and previous.exists():
        rows = [r for r in json.loads(previous.read_text()) if r.get("status") == "ok"]
    finished = {(r["N"], r["mode"]) for r in rows}
    env = dict(os.environ, JAX_PLATFORMS=os.environ.get("JAX_PLATFORMS", "cpu"), XLA_PYTHON_CLIENT_PREALLOCATE="false")
    for n in sizes:
        for mode in modes:
            if (mode == "jacrev_chunked" and n <= chunk) or (n, mode) in finished:
                continue
            log = LOGS / f"ouu_measure_{n}_{mode}.log"
            cmd = [python, __file__, "--measure", "--n", str(n), "--mode", mode, "--chunk", str(chunk)]
            began = time.perf_counter()
            try:
                done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, env=env, cwd=str(Path(__file__).resolve().parent.parent))
                log.write_text(done.stdout + "\n--- stderr ---\n" + done.stderr)
                line = next((ln for ln in reversed(done.stdout.splitlines()) if ln.startswith("OUU_RESULT ")), None)
                if done.returncode != 0 or line is None:
                    tail = (done.stderr.strip().splitlines() or ["?"])[-1][:160]
                    row = {"N": n, "mode": mode, "status": f"exit {done.returncode}: {tail}"}
                else:
                    row = json.loads(line[len("OUU_RESULT "):])
                    row["status"] = "ok"
            except subprocess.TimeoutExpired as failure:
                log.write_text((failure.stdout or b"").decode() if isinstance(failure.stdout, bytes) else (failure.stdout or ""))
                row = {"N": n, "mode": mode, "status": f"timeout after {timeout_s:.0f} s"}
            row["subprocess_s"] = time.perf_counter() - began
            rows.append(row)
            print(json.dumps(row, default=str), flush=True)
            rows.sort(key=lambda r: (r["N"], MODES.index(r["mode"])))
            write_csv("ouu.py", with_ratios(rows), name="ouu_scaling")
            write_json("ouu.py", with_ratios(rows), name="ouu_scaling")
    rows = with_ratios(rows)
    write_csv("ouu.py", rows, name="ouu_scaling")
    write_json("ouu.py", rows, name="ouu_scaling")
    return rows


def with_ratios(rows: list[dict]) -> list[dict]:
    """Reverse / forward ratios for time and memory, per N."""
    by = {(r["N"], r["mode"]): r for r in rows if r.get("status") == "ok"}
    out = []
    for r in rows:
        r = dict(r)
        base = by.get((r["N"], "fwd"))
        if base and r.get("status") == "ok" and r["mode"] != "fwd":
            r["time_over_fwd"] = r["warm_s"] / base["warm_s"]
            r["rss_increment_over_fwd"] = (r["rss_increment_mb"] / base["rss_increment_mb"]) if base["rss_increment_mb"] > 0 else None
            r["rss_peak_over_fwd"] = r["rss_peak_mb"] / base["rss_peak_mb"]
            if "temp_mb" in r and base.get("temp_mb"):
                r["temp_over_fwd"] = r["temp_mb"] / base["temp_mb"]
        out.append(r)
    return out


# ---------------------------------------------------------------- summaries of a sample set


def summarise(ouu: Ouu, layout: dict, Y_all: np.ndarray, alpha_extra: float = 0.9) -> dict:
    """Everything the evidence reads off one batched call at one design: the nominal
    row, the objective measures, failure and violation fractions, per-sample
    feasibility (converged, net > 0, every g <= 0) and the coe statistics over the
    feasible samples, percentiles, and the constraints' CVaRs (with a failed sample at
    `FAILED_G`, as the optimiser sees them)."""
    Y_all = np.asarray(Y_all, dtype=float)
    Y, Yn = Y_all[:-1], Y_all[-1]
    c_coe, c_net, c_avail = layout["c_coe"], layout["c_net"], layout["c_avail"]
    g0, g1 = layout["c_g"]
    c_steps, c_conv = layout["c_steps"], layout["c_conv"]
    finite = np.all(np.isfinite(Y[:, :c_steps]), axis=1)
    conv = (Y[:, c_conv] > 0.5) & finite
    coe, net, g = Y[:, c_coe], Y[:, c_net], Y[:, g0:g1]
    n = Y.shape[0]
    positive = conv & (net > 0)
    violates = np.where(conv[:, None], g > G_TOL, True)
    feasible = positive & ~np.any(violates, axis=1)
    kwh = np.maximum(net * Y[:, c_avail], layout["tiny"])
    annual = np.where(conv, coe, 0.0) * kwh
    g_masked = np.where(conv[:, None], g, FAILED_G)
    names = [c.spelling for c in ouu.constraints]

    def cvar(alpha):
        m = max(1, int(math.ceil((1.0 - alpha) * n)))
        worst = -np.sort(-g_masked, axis=0)[:m]
        return dict(zip(names, np.mean(worst, axis=0).tolist(), strict=True))

    def percentiles(v, mask):
        v = v[mask]
        if v.size == 0:
            return {"p5": None, "p50": None, "p95": None, "mean": None, "median": None, "count": 0}
        p5, p50, p95 = np.percentile(v, [5, 50, 95])
        return {"p5": float(p5), "p50": float(p50), "p95": float(p95), "mean": float(np.mean(v)), "median": float(np.median(v)), "count": int(v.size)}

    nominal_ok = bool(Yn[c_conv] > 0.5 and np.all(np.isfinite(Yn[:c_steps])))
    return {
        "n": int(n),
        "nominal": {"converged": nominal_ok, "coe": float(Yn[c_coe]), "net_mw": float(Yn[c_net]),
                    "g": dict(zip(names, Yn[g0:g1].tolist(), strict=True)),
                    "feasible": bool(nominal_ok and Yn[c_net] > 0 and np.all(Yn[g0:g1] <= G_TOL))},
        "failed_fraction": float(1.0 - conv.mean()),
        "p_net_nonpositive": float(np.mean(net <= 0)),
        "feasible_fraction": float(feasible.mean()),
        "any_violation_fraction": float(np.mean(np.any(violates, axis=1))),
        "any_violation_fraction_among_converged": float(np.mean(np.any(g[conv] > G_TOL, axis=1))) if conv.any() else None,
        "p_violated": dict(zip(names, np.where(conv[:, None], g > G_TOL, True).mean(axis=0).tolist(), strict=True)),
        "p_violated_among_converged": dict(zip(names, (g[conv] > G_TOL).mean(axis=0).tolist(), strict=True)) if conv.any() else None,
        "mean_g_among_converged": dict(zip(names, g[conv].mean(axis=0).tolist(), strict=True)) if conv.any() else None,
        "cvar": {f"{ouu.alpha:g}": cvar(ouu.alpha), f"{alpha_extra:g}": cvar(alpha_extra)},
        "levelised": float(np.sum(annual[conv]) / max(np.sum(kwh[conv]), 1e-10)) if conv.any() else None,
        "rated": float(np.mean(annual[conv]) / (ouu.rated_mw * float(Yn[c_avail]))) if conv.any() else None,
        "mean_coe_where_converged_and_positive_net": float(np.mean(coe[positive])) if positive.any() else None,
        "median_coe_where_converged": float(np.median(coe[conv])) if conv.any() else None,
        "coe_where_converged_and_positive_net": percentiles(coe, positive),
        "coe_where_feasible": percentiles(coe, feasible),
        "net_mw_where_converged": percentiles(net, conv),
        "steps_mean": float(np.mean(Y[:, c_steps])), "steps_max": float(np.max(Y[:, c_steps])),
    }


def per_sample(layout: dict, Y_all: np.ndarray) -> dict:
    """The columns the histograms draw, per sample: coe, net, g, converged, feasible."""
    Y = np.asarray(Y_all, dtype=float)[:-1]
    g0, g1 = layout["c_g"]
    finite = np.all(np.isfinite(Y[:, : layout["c_steps"]]), axis=1)
    conv = (Y[:, layout["c_conv"]] > 0.5) & finite
    g = Y[:, g0:g1]
    feasible = conv & (Y[:, layout["c_net"]] > 0) & np.all(g <= G_TOL, axis=1)
    return {"coe": Y[:, layout["c_coe"]], "net": Y[:, layout["c_net"]], "g": g, "converged": conv, "feasible": feasible}


def evaluate(fns: dict, x: np.ndarray, theta: PathMap, starts0: np.ndarray, x_from: np.ndarray | None = None,
             steps: int = 8, eps: float = EPS_FAILED, max_calls: int = 64) -> tuple[np.ndarray, np.ndarray, dict]:
    """The batched rows at design `x`: once from the nominal root (`Y_cold`), and once
    warm (`Y_warm`, what the optimiser saw). Warm means from the converged roots of a
    **continuation** from `x_from` (the deterministic design, where the nominal root is
    a start every sample converges from) in `steps` pieces, each piece's Newton started
    from the previous piece's roots and halved while it loses more than `eps` of the
    samples -- the chain of warm starts the optimiser itself walked. Without `x_from`,
    from the cold pass's own converged roots. A last pass from the converged roots
    either way. Returns `(Y_cold, Y_warm, timing)`."""
    run = fns["_run_batch_jit"]
    layout = fns["layout"]
    u0, u1 = layout["c_u"]
    x = np.asarray(x, float)
    calls = {"n": 0}

    def call(xi, starts):
        began = time.perf_counter()
        Y = np.asarray(jax.block_until_ready(run(jnp.asarray(xi), theta, jnp.asarray(starts))))
        calls["n"] += 1
        conv = (Y[:, layout["c_conv"]] > 0.5) & np.all(np.isfinite(Y[:, : layout["c_steps"]]), axis=1)
        return Y, conv, time.perf_counter() - began

    Y_cold, conv, cold_s = call(x, starts0)
    starts = np.where(conv[:, None], Y_cold[:, u0:u1], starts0)
    path_failed = []
    if x_from is not None and not np.allclose(x_from, x):
        x_from = np.asarray(x_from, float)
        Y0, conv0, _ = call(x_from, starts0)
        starts = np.where(conv0[:, None], Y0[:, u0:u1], starts0)
        base_failed = 1.0 - conv0.mean()
        x_cur, fraction, halvings = x_from.copy(), 1.0 / steps, 0
        while not np.allclose(x_cur, x) and calls["n"] < max_calls:
            x_try = x if fraction >= 1.0 - 1e-12 else x_cur + fraction * (x - x_from)
            if np.linalg.norm(x_try - x_from) >= np.linalg.norm(x - x_from) - 1e-12:
                x_try = x
            Yi, conv_i, _ = call(x_try, starts)
            failed_i = 1.0 - conv_i.mean()
            if failed_i > base_failed + eps and halvings < 6:
                fraction, halvings = fraction / 2, halvings + 1
                continue
            path_failed.append(float(failed_i))
            x_cur = x_try
            starts = np.where(conv_i[:, None], Yi[:, u0:u1], starts)
    Y_warm, conv, warm_s = call(x, starts)
    starts = np.where(conv[:, None], Y_warm[:, u0:u1], starts)
    Y_warm, conv, warm_s = call(x, starts)
    return Y_cold, Y_warm, {"cold_s": cold_s, "warm_s": warm_s, "calls": calls["n"], "path_failed": path_failed,
                            "failed_cold": float(1.0 - ((Y_cold[:, layout["c_conv"]] > 0.5).mean()))}


def design_table(ouu: Ouu, x_det: np.ndarray, x_rob: np.ndarray, tol: float = 1e-3) -> list[dict]:
    """Per `ixc` entry: bounds, both designs, the delta, and which bound is active
    (within `tol` of the range)."""
    rows = []
    for i, v in enumerate(ouu.design):
        lo, hi = float(ouu.lower[i]), float(ouu.upper[i])
        span = hi - lo
        rows.append({
            "ixc": int(ouu.ixc[i]), "place": v.spelling, "lower": lo, "upper": hi,
            "deterministic": float(x_det[i]), "robust": float(x_rob[i]),
            "delta": float(x_rob[i] - x_det[i]), "delta_relative": float(x_rob[i] / x_det[i] - 1.0),
            "active_bound": "upper" if hi - x_rob[i] < tol * span else "lower" if x_rob[i] - lo < tol * span else None,
            "deterministic_active_bound": "upper" if hi - x_det[i] < tol * span else "lower" if x_det[i] - lo < tol * span else None,
        })
    return rows


def robust_design_of(payload: dict, design: tuple) -> tuple[np.ndarray, str]:
    """The robust design a run's JSON records: `robust` where the run wrote one, else
    its best feasible model call, else the final point."""
    if payload.get("robust"):
        return np.array([payload["robust"]["x"][v.spelling] for v in design]), payload["robust"]["source"]
    if payload.get("best_feasible"):
        return np.array(payload["best_feasible"]["x"], dtype=float), "best_feasible"
    return np.array([payload["final"]["x"][v.spelling] for v in design]), "final"


def evidence(ouu: Ouu, fns: dict, x_det: np.ndarray, x_rob: np.ndarray, fresh_seed: int,
             name: str, source: str, extra: dict | None = None) -> dict:
    """Both designs on the fixed sample set and on a fresh Sobol' set: the summaries,
    the design deltas, and the per-sample columns as `<name>.npz`."""
    fns.setdefault("_run_batch_jit", jax.jit(fns["run_batch"]))
    layout = fns["layout"]
    _Theta_f, theta_fresh, starts_fresh = sample(ouu.model, ouu.n, fresh_seed)
    sets = {"fixed": (ouu.theta, ouu.starts0), "fresh": (theta_fresh, starts_fresh)}
    designs = {"deterministic": np.asarray(x_det, float), "robust": np.asarray(x_rob, float)}
    result = {
        "n": ouu.n, "alpha": ouu.alpha, "seed": ouu.seed, "fresh_seed": fresh_seed, "inputs": ouu.inputs,
        "objective": ouu.objective, "table": ouu.table, "hfact_sigma": ouu.hfact_sigma, "hfact_belief": hfact_belief(ouu.hfact_sigma),
        "robust_source": source, "backend": jax.default_backend(),
        "constraints": [c.spelling for c in ouu.constraints],
        "design": design_table(ouu, x_det, x_rob),
        "sets": {}, "timing": {},
    }
    if extra:
        result.update(extra)
    columns = {}
    for set_name, (theta, starts0) in sets.items():
        result["sets"][set_name] = {}
        for design_name, x in designs.items():
            Y_cold, Y_warm, timing = evaluate(fns, x, theta, starts0, x_from=None if design_name == "deterministic" else designs["deterministic"])
            summary = summarise(ouu, layout, Y_warm)
            summary["failed_fraction_cold_start"] = timing["failed_cold"]
            summary["continuation"] = {"calls": timing["calls"], "path_failed": timing["path_failed"]}
            result["sets"][set_name][design_name] = summary
            result["timing"][f"{set_name}/{design_name}"] = timing
            for key, value in per_sample(layout, Y_warm).items():
                columns[f"{set_name}/{design_name}/{key}"] = value
            print(f"[evidence] {set_name:5s} {design_name:13s}: failed {summary['failed_fraction']:.4f}  feasible {summary['feasible_fraction']:.4f}  "
                  f"any violation {summary['any_violation_fraction']:.4f}  coe|feasible mean {_fmt(summary['coe_where_feasible']['mean'])}  "
                  f"coe|net>0 mean {_fmt(summary['mean_coe_where_converged_and_positive_net'])}  nominal coe {summary['nominal']['coe']:.2f}", flush=True)
    write_json("ouu.py", result, name=name)
    np.savez_compressed(OUT / f"{name}.npz", **columns)
    return result


def sweep_te(ouu: Ouu, fns: dict, x_rob: np.ndarray, name: str, points: int = 16, source: str = "") -> dict:
    """What a per-sample choice of `temp_plasma_electron_vol_avg_kev` would gain at the
    robust design: T_e swept over `points` values between its bounds, every other
    design place at `x_rob`, each batch warm-started from the previous point's roots,
    and per sample the cheapest point that is feasible (converged, net > 0, every g <= 0
    -- the graph's inequalities, and `c16` where it is one). The shared T_e's own row
    is the `robust` reference. Since the shared T_e is a restriction of the recourse,
    the gain reported is a **lower bound** on the value of T_e recourse."""
    fns.setdefault("_run_batch_jit", jax.jit(fns["run_batch"]))
    run, layout = fns["_run_batch_jit"], fns["layout"]
    u0, u1 = layout["c_u"]
    c_coe, c_net, c_conv = layout["c_coe"], layout["c_net"], layout["c_conv"]
    g0, g1 = layout["c_g"]
    te = next((i for i, v in enumerate(ouu.design) if v.spelling == ".physics.temp_plasma_electron_vol_avg_kev"), None)
    if te is None:
        return {"skipped": "T_e is not a design place here"}
    lo, hi = float(ouu.lower[te]), float(ouu.upper[te])
    grid = np.linspace(lo, hi, points)
    # The reference: the robust design itself, warm along a continuation from the deterministic one.
    _Y_cold, Y_ref, timing = evaluate(fns, x_rob, ouu.theta, ouu.starts0, x_from=ouu.x0)
    Y_ref = np.asarray(Y_ref)
    names = [c.spelling for c in ouu.constraints]
    graph_only = [i for i, c in enumerate(ouu.constraints) if c.spelling != C16]

    def feasible_of(Y, cols):
        conv = (Y[:, c_conv] > 0.5) & np.all(np.isfinite(Y[:, : layout["c_steps"]]), axis=1)
        return conv & (Y[:, c_net] > 0) & np.all(Y[:, g0:g1][:, cols] <= G_TOL, axis=1), conv

    def walk(order):
        """The grid walked in `order` from the reference roots; rows [points, N + 1, columns]."""
        starts = np.where((Y_ref[:, c_conv] > 0.5)[:, None], Y_ref[:, u0:u1], ouu.starts0)
        rows = {}
        for k in order:
            x = x_rob.copy()
            x[te] = grid[k]
            Y = np.asarray(jax.block_until_ready(run(jnp.asarray(x), ouu.theta, jnp.asarray(starts))))
            conv = Y[:, c_conv] > 0.5
            starts = np.where(conv[:, None], Y[:, u0:u1], starts)
            rows[k] = Y
        return rows

    # Down from the robust T_e and up from it, so every point is reached by continuation.
    k_rob = int(np.argmin(np.abs(grid - x_rob[te])))
    rows = walk(range(k_rob, -1, -1))
    rows.update(walk(range(k_rob, points)))
    Ys = np.stack([rows[k] for k in range(points)])  # [points, N + 1, columns]
    n = Ys.shape[1] - 1
    result = {"name": name, "source": source, "grid_kev": grid.tolist(), "t_e_shared_kev": float(x_rob[te]),
              "n": n, "alpha": ouu.alpha, "constraints": names, "reference_continuation": timing}
    for label, cols in (("all_constraints", list(range(len(names)))), ("graph_inequalities", graph_only)):
        feas_ref, conv_ref = feasible_of(Y_ref[:-1], cols)
        feas = np.stack([feasible_of(Ys[k, :-1], cols)[0] for k in range(points)])  # [points, N]
        coe = np.where(feas, Ys[:, :-1, c_coe], np.inf)
        best = np.argmin(coe, axis=0)  # per sample
        any_feasible = np.isfinite(coe[best, np.arange(n)])
        coe_best = coe[best, np.arange(n)]
        gained = feas_ref & any_feasible
        result[label] = {
            "feasible_fraction_shared": float(feas_ref.mean()),
            "feasible_fraction_swept": float(any_feasible.mean()),
            "coe_mean_where_feasible_shared": float(Y_ref[:-1, c_coe][feas_ref].mean()) if feas_ref.any() else None,
            "coe_mean_where_feasible_swept": float(coe_best[any_feasible].mean()) if any_feasible.any() else None,
            "coe_mean_over_samples_feasible_both": {
                "shared": float(Y_ref[:-1, c_coe][gained].mean()) if gained.any() else None,
                "swept": float(coe_best[gained].mean()) if gained.any() else None,
                "count": int(gained.sum()),
            },
            "chosen_t_e_kev": {"p5": float(np.percentile(grid[best[any_feasible]], 5)) if any_feasible.any() else None,
                               "p50": float(np.percentile(grid[best[any_feasible]], 50)) if any_feasible.any() else None,
                               "p95": float(np.percentile(grid[best[any_feasible]], 95)) if any_feasible.any() else None,
                               "at_shared_fraction": float(np.mean(best[any_feasible] == k_rob)) if any_feasible.any() else None},
            "feasible_fraction_per_grid_point": feas.mean(axis=1).tolist(),
            "failed_fraction_per_grid_point": [float(1.0 - feasible_of(Ys[k, :-1], cols)[1].mean()) for k in range(points)],
        }
    write_json("ouu", result, name)
    a = result["all_constraints"]
    print(f"[sweep_te] {name}: shared T_e {x_rob[te]:.2f} keV feasible {a['feasible_fraction_shared']:.3f} -> swept {a['feasible_fraction_swept']:.3f}; "
          f"coe|feasible {a['coe_mean_where_feasible_shared']} -> {a['coe_mean_where_feasible_swept']}; "
          f"on samples feasible both ways ({a['coe_mean_over_samples_feasible_both']['count']}): "
          f"{a['coe_mean_over_samples_feasible_both']['shared']} -> {a['coe_mean_over_samples_feasible_both']['swept']}", flush=True)
    return result


def _fmt(v) -> str:
    return "--" if v is None else f"{v:.2f}"


def decompose(n: int, robust: Path, inputs: str, seed: int, alpha: float, with_c16: bool,
              table: str = "new", hfact_sigma: float = HFACT_SIGMA, name: str = "ouu_decomposition",
              pairing: str = "one") -> dict:
    """The deterministic design and a run's robust design on one N-sample set: every
    measure side by side (`out/<name>.json`)."""
    ouu = build(n, alpha=alpha, seed=seed, with_c16=with_c16, inputs=inputs, objective="mean", table=table,
                hfact_sigma=hfact_sigma, pairing=pairing)
    fns = make(ouu)
    fns["_run_batch_jit"] = jax.jit(fns["run_batch"])
    payload = json.loads(Path(robust).read_text())
    x_rob, source = robust_design_of(payload, ouu.design)
    rows = {}
    for label, x in (("deterministic", ouu.x0), ("robust", x_rob)):
        _Y_cold, Y_warm, timing = evaluate(fns, x, ouu.theta, ouu.starts0, x_from=None if label == "deterministic" else ouu.x0)
        rows[label] = summarise(ouu, fns["layout"], Y_warm)
        rows[label]["failed_fraction_cold_start"] = timing["failed_cold"]
        rows[label]["continuation"] = {"calls": timing["calls"], "path_failed": timing["path_failed"]}
    result = {
        "n": n, "seed": seed, "inputs": inputs, "alpha": alpha, "table": ouu.table, "hfact_sigma": ouu.hfact_sigma,
        "hfact_belief": hfact_belief(ouu.hfact_sigma), "n_inputs": len(ouu.model.inputs),
        "belief_table": [dataclasses.asdict(i) for i in ouu.model.inputs],
        "robust_from": str(robust), "robust_source": source,
        "robust_f_recorded": payload.get("robust", payload.get("best_feasible") or {}).get("f"),
        "deterministic_coe_at_nominal": ouu.check["port_at_nominal"][".costs.coe"],
        "design": design_table(ouu, ouu.x0, x_rob),
        "rows": rows,
    }
    write_json("ouu.py", result, name=name)
    print(f"table {ouu.table}, hfact sigma {ouu.hfact_sigma}, inputs {inputs} ({len(ouu.model.inputs)} entries), N {n}, robust design from {robust} ({source})")
    print(decomposition_table(result))
    return result


def decomposition_table(result: dict) -> str:
    d, r = result["rows"]["deterministic"], result["rows"]["robust"]
    lines = [f"{'measure':44s} {'deterministic':>14s} {'robust':>14s}"]

    def row(label, a, b, fmt="{:14.3f}"):
        lines.append(f"{label:44s} {fmt.format(a) if a is not None else '--':>14s} {fmt.format(b) if b is not None else '--':>14s}")

    row("coe at nominal inputs [$/MWh]", d["nominal"]["coe"], r["nominal"]["coe"])
    row("levelised (ratio of expectations)", d["levelised"], r["levelised"])
    row("mean coe | converged, net > 0", d["mean_coe_where_converged_and_positive_net"], r["mean_coe_where_converged_and_positive_net"])
    row("median coe | converged", d["median_coe_where_converged"], r["median_coe_where_converged"])
    row("mean coe | feasible", d["coe_where_feasible"]["mean"], r["coe_where_feasible"]["mean"])
    row("median coe | feasible", d["coe_where_feasible"]["median"], r["coe_where_feasible"]["median"])
    row("coe p5 / p95 | feasible", d["coe_where_feasible"]["p5"], r["coe_where_feasible"]["p5"])
    row("  ", d["coe_where_feasible"]["p95"], r["coe_where_feasible"]["p95"])
    row("failed fraction (warm, continuation)", d["failed_fraction"], r["failed_fraction"], "{:14.4f}")
    row("failed fraction (cold, nominal root)", d["failed_fraction_cold_start"], r["failed_fraction_cold_start"], "{:14.4f}")
    row("P(net power <= 0)", d["p_net_nonpositive"], r["p_net_nonpositive"], "{:14.4f}")
    row("feasible fraction", d["feasible_fraction"], r["feasible_fraction"], "{:14.4f}")
    row("any violation fraction", d["any_violation_fraction"], r["any_violation_fraction"], "{:14.4f}")
    row("net power p5 / p50 / p95 [MW]", d["net_mw_where_converged"]["p5"], r["net_mw_where_converged"]["p5"], "{:14.1f}")
    row("  ", d["net_mw_where_converged"]["p50"], r["net_mw_where_converged"]["p50"], "{:14.1f}")
    row("  ", d["net_mw_where_converged"]["p95"], r["net_mw_where_converged"]["p95"], "{:14.1f}")
    lines.append("")
    lines.append(f"{'constraint':28s} {'P(g>0) det':>11s} {'P(g>0) rob':>11s} {'CVaR0.9 det':>12s} {'CVaR0.9 rob':>12s}")
    for c in d["p_violated"]:
        lines.append(f"{c.replace('^cond.constraints.', ''):28s} {d['p_violated'][c]:11.4f} {r['p_violated'][c]:11.4f} "
                     f"{d['cvar']['0.9'][c]:12.4f} {r['cvar']['0.9'][c]:12.4f}")
    lines.append("")
    lines.append(f"{'ixc':>4s} {'place':44s} {'lower':>8s} {'upper':>8s} {'det':>10s} {'robust':>10s} {'delta':>10s} active")
    for e in result["design"]:
        lines.append(f"{e['ixc']:4d} {e['place']:44s} {e['lower']:8.4g} {e['upper']:8.4g} {e['deterministic']:10.4f} {e['robust']:10.4f} "
                     f"{e['delta']:+10.4f} {e['active_bound'] or ''}")
    return "\n".join(lines)


# ---------------------------------------------------------------- the SLSQP run


def smoke(n: int, alpha: float, max_iter: int, with_c16: bool, eps: float, seed: int, warm: bool,
          delta: float = 0.25, objective: str = "mean", inputs: str = "physics", jac: str = "fwd",
          chunks: int | None = None, tol: float = 1e-4, ftol: float = 1e-6, gtol: float = 1e-4,
          name: str = "ouu_smoke", with_evidence: bool = False, fresh_seed: int = 1, tag: str = "",
          max_outer: int = 60, delta_min: float = 1e-3, table: str = "new", hfact_sigma: float = HFACT_SIGMA,
          ftol_outer: float = 1e-5, pairing: str = "one", start: Path | None = None,
          alpha16: float | None = None, te_recourse: int = 0, te_range: tuple | None = None) -> dict:
    from scipy.optimize import minimize  # noqa: PLC0415

    ouu = build(n, alpha=alpha, seed=seed, with_c16=with_c16, inputs=inputs, objective=objective, table=table,
                hfact_sigma=hfact_sigma, pairing=pairing, alpha16=alpha16, te_recourse=te_recourse, te_range=te_range)
    fns = make(ouu, jac=jac, chunks=chunks)
    fused = jax.jit(fns["value_jac_starts"])
    forward = jax.jit(fns["statistics_vec"])
    x0, scale = ouu.x0, ouu.x0.copy()
    theta = ouu.theta
    starts = ouu.starts0
    if start is not None:  # a second start, from another run's robust design, warm along the way
        x0, _ = robust_design_of(json.loads(Path(start).read_text()), ouu.design)
        fns["_run_batch_jit"] = jax.jit(fns["run_batch"])
        _Y_cold, Y_warm, timing = evaluate(fns, x0, theta, ouu.starts0, x_from=ouu.x0)
        u0, u1 = fns["layout"]["c_u"]
        u = np.asarray(Y_warm[:, u0:u1])
        conv = np.asarray(Y_warm[:, fns["layout"]["c_conv"]]) > 0.5
        starts = np.where(conv[:, None], u, ouu.starts0)
        print(f"start {start}: continuation from the deterministic design in {timing['calls']} calls, "
              f"failed {timing['failed_cold']:.3f} cold, path failed {timing['path_failed']}", flush=True)
    state = {"starts": jnp.asarray(starts), "cache": {}, "calls": 0, "trace": [], "u_next": {}}
    n_g = ouu.n_g
    print(f"built in {ouu.build_s:.1f} s: N {n} (+ nominal row), alpha {alpha} (m = {ouu.m}), objective {objective}, "
          f"inputs {inputs} ({len(ouu.model.inputs)} entries), table {ouu.table} (hfact sigma {ouu.hfact_sigma}), "
          f"pairing {ouu.pairing} ({len(ouu.design)} design places: {[v.spelling.rsplit('.', 1)[-1] for v in ouu.design]}"
          f"{'' if ouu.te_grid is None else f', T_e recourse on {len(ouu.te_grid)} points'}), "
          f"jac {jac}, chunks {chunks}, backend {jax.default_backend()}", flush=True)

    # The forward call alone, timed: compile, then warm.
    began = time.perf_counter()
    jax.block_until_ready(forward(jnp.asarray(x0), theta, state["starts"]))
    forward_compile_s = time.perf_counter() - began
    walls = []
    for _ in range(3):
        began = time.perf_counter()
        jax.block_until_ready(forward(jnp.asarray(x0), theta, state["starts"]))
        walls.append(time.perf_counter() - began)
    forward_warm_s = min(walls)
    print(f"forward call: compile {forward_compile_s:.1f} s, warm {forward_warm_s * 1e3:.1f} ms ({1e6 * forward_warm_s / n:.2f} us/sample)", flush=True)

    def at(xs: np.ndarray):
        """`(values, jacobian_scaled, diagnostics, wall)` at scaled design `xs`,
        cached per point; the next starts kept aside under the point's key."""
        key = xs.tobytes()
        if key not in state["cache"]:
            while len(state["cache"]) >= 3:
                old = next(iter(state["cache"]))
                del state["cache"][old]
                state["u_next"].pop(old, None)
            x = jnp.asarray(xs * scale)
            began = time.perf_counter()
            out, jacobian, u_next, diagnostics = jax.block_until_ready(fused(x, theta, state["starts"]))
            wall = time.perf_counter() - began
            state["calls"] += 1
            state["u_next"][key] = u_next
            state["cache"][key] = (np.asarray(out), np.asarray(jacobian) * scale[None, :], {k: np.asarray(v).tolist() for k, v in diagnostics.items()}, wall)
        return state["cache"][key]

    def adopt(xs: np.ndarray):
        """Warm-start every sample from its root at `xs` -- SLSQP's accepted iterate."""
        if warm:
            at(np.asarray(xs))
            state["starts"] = state["u_next"][np.asarray(xs).tobytes()]

    def objective_fn(xs):
        return float(at(np.asarray(xs))[0][0])

    def objective_jac(xs):
        return at(np.asarray(xs))[1][0]

    def ineq(xs):
        v = at(np.asarray(xs))[0]
        return np.concatenate([-v[1 : 1 + n_g], [eps - v[-1]]])

    def ineq_jac(xs):
        j = at(np.asarray(xs))[1]
        return np.concatenate([-j[1 : 1 + n_g], -j[-1:]], axis=0)

    trace = state["trace"]

    def record(xs):
        v, _j, d, wall = at(np.asarray(xs))
        trace.append({"call": len(trace) + 1, "f": float(v[0]), "max_g_cvar": float(v[1 : 1 + n_g].max()),
                      "failed_fraction": float(v[-1]), "x": (np.asarray(xs) * scale).tolist(),
                      "steps_mean": d["steps_mean"], "steps_max": d["steps_max"], "model_s": wall})
        print(f"[slsqp] call {len(trace):3d}: f {v[0]:9.4f} $/MWh  max CVaR g {v[1:1 + n_g].max(): .4f}  failed {v[-1]:.4f}  Newton steps mean {d['steps_mean']:.2f} max {d['steps_max']:.0f}  ({wall:.2f} s)", flush=True)

    def objective_recorded(xs):
        record(xs)
        return objective_fn(xs)

    def feasible_at(v) -> bool:
        return bool(v[1 : 1 + n_g].max() <= gtol and v[-1] <= eps)

    began = time.perf_counter()
    v0, j0, d0, first_call_s = at(x0 / scale)
    print(f"first fused value+jac{jac} call: {first_call_s:.1f} s (compile); f0 {v0[0]:.4f}, max CVaR g {v0[1:1 + n_g].max():.4f}, failed {v0[-1]:.4f}", flush=True)
    # SLSQP with move limits: each call boxed to x (1 -+ delta) around its start, the
    # box re-centred on its answer while the answer sits on a face of the box (and not
    # on the input file's own bound); halved and restarted from the best feasible point
    # when a call ends infeasible. Converged when SLSQP succeeds strictly inside the
    # box, or a re-centred call moves less than `tol`.
    x = x0 / scale
    best, stalls = None, 0
    total_nit, outer, result = 0, [], None
    converged, reason = False, "iteration limit reached"
    while total_nit < max_iter and len(outer) < max_outer:
        lo = np.maximum(ouu.lower / scale, x * (1.0 - delta))
        hi = np.minimum(ouu.upper / scale, x * (1.0 + delta))
        result = minimize(
            objective_recorded, x, jac=objective_jac, bounds=list(zip(lo, hi, strict=True)),
            constraints=[{"type": "ineq", "fun": ineq, "jac": ineq_jac}],
            method="SLSQP", options={"maxiter": max_iter - total_nit, "ftol": ftol}, callback=adopt,
        )
        nit = max(int(result.nit), 1)
        total_nit += nit
        x_new = np.asarray(result.x, dtype=float)
        # SLSQP leaves its answer ~1e-6 inside a bound, so a face is read to 1e-5 of
        # the box's half-width; a face that is the input file's own bound is not one.
        face_tol = 1e-5 * delta
        on_face = bool(np.any((np.abs(x_new - lo) < face_tol * np.abs(x)) & (lo > ouu.lower / scale + 1e-12))
                       | np.any((np.abs(x_new - hi) < face_tol * np.abs(x)) & (hi < ouu.upper / scale - 1e-12)))
        v, _j, _d, _w = at(x_new)
        feasible = feasible_at(v)
        moved = float(np.max(np.abs(x_new - x)))
        entry = {"nit": int(result.nit), "nfev": int(result.nfev), "njev": int(result.njev), "status": int(result.status),
                 "message": str(result.message), "on_move_limit": on_face, "move_limit": delta, "f": float(v[0]),
                 "max_g_cvar": float(v[1 : 1 + n_g].max()), "failed_fraction": float(v[-1]), "feasible": feasible,
                 "moved": moved, "x": (x_new * scale).tolist()}
        outer.append(entry)
        print(f"[outer {len(outer)}] {result.message}; nit {result.nit}, box +-{delta:.3g}, on move limit: {on_face}, "
              f"feasible: {feasible}, moved {moved:.2e}, f {v[0]:.4f}, max CVaR g {v[1:1 + n_g].max():.4f}", flush=True)
        previous_best = best["f"] if best is not None else None
        if feasible and (best is None or v[0] < best["f"]):
            best = {"f": float(v[0]), "x": x_new.copy(), "starts": state["u_next"].get(x_new.tobytes(), state["starts"]), "outer": len(outer)}
        if not feasible or result.status not in (0, 8, 9):
            # Ended infeasible, or SLSQP gave up (incompatible constraints): a smaller
            # box from the best feasible point. A status-8 line search ("positive
            # directional derivative") at a feasible point is not that: the median and
            # the CVaRs are piecewise smooth, so the linear model is wrong at a kink,
            # and the call is re-centred like any other.
            delta *= 0.5
            if best is not None:
                x = best["x"].copy()
                state["starts"] = best["starts"]
            if delta < delta_min:
                reason = f"move limit shrunk below {delta_min:g} without a feasible success"
                break
            entry["action"] = f"restart from best feasible with box +-{delta:.3g}"
            continue
        x = x_new
        adopt(x_new)
        if result.status == 0 and not on_face:
            converged, reason = True, "SLSQP success strictly inside the move box"
            break
        if moved < tol:
            converged, reason = True, f"re-centred call moved {moved:.2e} < tol {tol:g}"
            break
        if previous_best is not None and v[0] >= previous_best - ftol_outer * abs(previous_best):
            stalls += 1
            if stalls >= 2:
                converged, reason = True, f"two re-centred calls improved f by less than {ftol_outer:g} relative"
                break
        else:
            stalls = 0
        if result.status == 9:
            reason = "iteration limit reached"
            break
    else:
        reason = "iteration limit reached" if total_nit >= max_iter else f"outer call limit {max_outer} reached"
    wall = time.perf_counter() - began
    vf, jf, df, _ = at(x)
    final_feasible = feasible_at(vf)
    if final_feasible and (best is None or vf[0] <= best["f"] + 1e-12):
        robust_x, source = x * scale, "final"
    elif best is not None:
        robust_x, source = best["x"] * scale, f"best feasible (outer call {best['outer']})"
    else:
        robust_x, source = x * scale, "final (infeasible)"
    xf = x * scale
    names = [c.spelling for c in ouu.constraints]
    g0, gf = v0[1 : 1 + n_g], vf[1 : 1 + n_g]
    active = [names[i] for i in range(n_g) if abs(gf[i]) < 1e-3]
    violated = [names[i] for i in range(n_g) if gf[i] > 1e-3]
    at_bound = [ouu.design[i].spelling for i in range(len(xf)) if abs(xf[i] - ouu.lower[i]) < 1e-6 * max(1, abs(ouu.lower[i])) or abs(xf[i] - ouu.upper[i]) < 1e-6 * max(1, abs(ouu.upper[i]))]
    model_walls = sorted(t["model_s"] for t in trace[1:]) or [first_call_s]
    payload = {
        "n": n, "alpha": alpha, "m_worst": ouu.m, "alpha16": alpha16, "m16": ouu.m16, "te_recourse": te_recourse,
        "te_grid": None if ouu.te_grid is None else ouu.te_grid.tolist(), "eps_failed": eps, "with_c16": with_c16, "warm_starts": warm,
        "objective": objective, "inputs": inputs, "n_inputs": len(ouu.model.inputs), "table": ouu.table,
        "hfact_sigma": ouu.hfact_sigma, "hfact_belief": hfact_belief(ouu.hfact_sigma), "jac": jac, "chunks": chunks,
        "pairing": ouu.pairing, "closed": ouu.check["pairing"], "uncertain": [i.path for i in ouu.model.inputs],
        "started_from": str(start) if start is not None else "deterministic", "x_start": x0.tolist(),
        "max_iter": max_iter, "move_limit": delta, "tol": tol, "ftol": ftol, "gtol": gtol, "seed": seed, "backend": jax.default_backend(),
        "device": str(jax.local_devices()[0].device_kind) if jax.local_devices() else None,
        "design": [v.spelling for v in ouu.design],
        "ixc": list(ouu.ixc),
        "bounds": {v.spelling: [float(lo), float(hi)] for v, lo, hi in zip(ouu.design, ouu.lower, ouu.upper, strict=True)},
        "deterministic": {
            "coe_dollar_per_mwh": ouu.check["port_at_nominal"][".costs.coe"],
            "objf": ouu.check["port_at_nominal"]["^cond.numerics.objf"],
            "x": ouu.x0.tolist(),
            "inequalities_at_nominal": ouu.check["inequalities_at_nominal"],
        },
        "start": {"f": float(v0[0]), "g_cvar": dict(zip(names, g0.tolist(), strict=True)), "failed_fraction": float(v0[-1]),
                  "diagnostics": d0, "jac_f": j0[0].tolist()},
        "final": {"f": float(vf[0]), "g_cvar": dict(zip(names, gf.tolist(), strict=True)), "failed_fraction": float(vf[-1]),
                  "feasible": final_feasible,
                  "x": dict(zip([v.spelling for v in ouu.design], xf.tolist(), strict=True)),
                  "x_over_x0": (xf / ouu.x0).tolist(), "diagnostics": df, "active": active, "violated": violated,
                  "design_at_bound": at_bound},
        "robust": {"x": dict(zip([v.spelling for v in ouu.design], robust_x.tolist(), strict=True)), "source": source,
                   "f": float(vf[0]) if source == "final" else (best["f"] if best is not None else float(vf[0]))},
        "converged": converged, "reason": reason,
        "scipy": {"success": bool(result.success) if result is not None else False, "status": int(result.status) if result is not None else None,
                  "message": str(result.message) if result is not None else None,
                  "major_iterations": total_nit, "outer_calls": len(outer)},
        "outer": outer,
        "best_feasible": best_feasible(trace, eps, gtol),
        "model_calls": state["calls"], "compile_s": first_call_s, "wall_s": wall, "build_s": ouu.build_s,
        "timing": {"forward_compile_s": forward_compile_s, "forward_warm_s": forward_warm_s,
                   "forward_us_per_sample": 1e6 * forward_warm_s / n,
                   "fused_compile_s": first_call_s, "fused_warm_median_s": float(np.median(model_walls)),
                   "fused_warm_min_s": float(model_walls[0]), "fused_over_forward": float(np.median(model_walls) / forward_warm_s)},
        "trace": trace,
    }
    write_json("ouu.py", payload, name=name)
    print(json.dumps({k: v for k, v in payload.items() if k not in ("trace", "bounds", "outer")}, indent=1, default=str), flush=True)
    print(f"[done] converged: {converged} ({reason}); {total_nit} major iterations in {len(outer)} outer calls, "
          f"{state['calls']} model calls, {wall:.1f} s; robust design from {source}, f {payload['robust']['f']:.4f}", flush=True)
    if with_evidence:
        evidence(ouu, fns, ouu.x0, robust_x, fresh_seed, f"ouu_evidence_{alpha:g}{tag}", source,
                 extra={"run": name, "converged": converged, "reason": reason, "robust_f": payload["robust"]["f"]})
    return payload


def best_feasible(trace: list, eps: float, tol: float = 1e-3) -> dict | None:
    """The cheapest model call at which every CVaR constraint held (to `tol`) and the
    failed fraction was within `eps` -- SLSQP's last iterate need not be its best."""
    feasible = [t for t in trace if t["max_g_cvar"] <= tol and t["failed_fraction"] <= eps]
    return min(feasible, key=lambda t: t["f"]) if feasible else None


# ---------------------------------------------------------------- figures


def plot(source: Path, alphas=(0.5, 0.75, 0.9, 0.95, 0.99), histogram_alpha: float = 0.9, tag: str = "") -> None:
    """`out/ouu_price_of_robustness.png` and `out/ouu_histograms_<alpha>.png` from the
    evidence files under `source`."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    det_colour, rob_colour = "#4a4a4a", "#c0504d"
    rows = []
    for alpha in alphas:
        path = source / f"ouu_evidence_{alpha:g}{tag}.json"
        if not path.exists():
            print(f"[plot] no {path}", flush=True)
            continue
        e = json.loads(path.read_text())
        rows.append((alpha, e))
    if rows:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
        for ax, key, title in ((axes[0], "mean_coe_where_converged_and_positive_net", "mean coe | converged, net > 0"),
                               (axes[1], "coe_where_feasible", "mean coe | feasible")):
            def pick(summary):
                return summary[key] if key != "coe_where_feasible" else summary[key]["mean"]

            det_fixed = [pick(e["sets"]["fixed"]["deterministic"]) for _a, e in rows]
            det_fresh = [pick(e["sets"]["fresh"]["deterministic"]) for _a, e in rows]
            ax.axhline(np.nanmean([v for v in det_fixed if v is not None]), color=det_colour, lw=1.2, label="deterministic, fixed set")
            ax.axhline(np.nanmean([v for v in det_fresh if v is not None]), color=det_colour, lw=1.2, ls="--", label="deterministic, fresh set")
            a = [alpha for alpha, _e in rows]
            ax.plot(a, [pick(e["sets"]["fixed"]["robust"]) for _a, e in rows], "o-", color=rob_colour, label="robust, fixed set")
            ax.plot(a, [pick(e["sets"]["fresh"]["robust"]) for _a, e in rows], "s--", color=rob_colour, mfc="white", label="robust, fresh set")
            for alpha, e in rows:
                if not e.get("converged", True):
                    ax.annotate("not converged", (alpha, pick(e["sets"]["fixed"]["robust"])), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
            ax.set_xlabel(r"$\alpha$ (CVaR level)")
            ax.set_ylabel("coe [$/MWh]")
            ax.set_title(title, fontsize=10)
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(OUT / "ouu_price_of_robustness.png", dpi=160)
        plt.close(fig)
        print(f"[plot] wrote {OUT / 'ouu_price_of_robustness.png'}", flush=True)

    path = source / f"ouu_evidence_{histogram_alpha:g}{tag}.json"
    npz = source / f"ouu_evidence_{histogram_alpha:g}{tag}.npz"
    if not (path.exists() and npz.exists()):
        print(f"[plot] no {path} / {npz}", flush=True)
        return
    e = json.loads(path.read_text())
    cols = np.load(npz)
    names = [c.replace("^cond.constraints.", "") for c in e["constraints"]]
    det, rob = e["sets"]["fixed"]["deterministic"], e["sets"]["fixed"]["robust"]
    worst = sorted(range(len(names)), key=lambda i: -max(det["p_violated"][e["constraints"][i]], rob["p_violated"][e["constraints"][i]]))[:3]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4))
    feasible_colour, infeasible_colour = {"deterministic": "#7f7f7f", "robust": "#d98c8a"}, {"deterministic": "#2b2b2b", "robust": "#8b1a1a"}

    def stacked(ax, key, transform, label, bins=40, log=False):
        """coe (or another column) at both designs, each stacked feasible / infeasible
        in one set of bins."""
        values = {d: transform(cols[f"fixed/{d}/{key}"], cols[f"fixed/{d}/converged"]) for d in ("deterministic", "robust")}
        finite = np.concatenate([v[np.isfinite(v)] for v in values.values()])
        if finite.size == 0:
            return
        lo, hi = np.percentile(finite, [0.5, 99.5])
        edges = np.linspace(lo, hi, bins + 1)
        width = (edges[1] - edges[0]) / 2.0
        for j, d in enumerate(("deterministic", "robust")):
            v = np.clip(values[d], lo, hi)
            feasible = cols[f"fixed/{d}/feasible"]
            ok = np.isfinite(values[d])
            h_f, _ = np.histogram(v[ok & feasible], bins=edges)
            h_i, _ = np.histogram(v[ok & ~feasible], bins=edges)
            left = edges[:-1] + j * width
            ax.bar(left, h_f, width=width, align="edge", color=feasible_colour[d], label=f"{d}, feasible")
            ax.bar(left, h_i, width=width, align="edge", bottom=h_f, color=infeasible_colour[d], label=f"{d}, infeasible")
        ax.set_xlabel(label)
        ax.set_ylabel("samples")
        if log:
            ax.set_yscale("log")

    n_samples = e["n"]
    stacked(axes[0, 0], "coe", lambda v, c: np.where(c & (v < 1e6), v, np.nan), "coe [$/MWh] (converged, < 1e6)")
    axes[0, 0].set_title(f"coe, N = {n_samples}, alpha = {histogram_alpha:g}", fontsize=10)
    stacked(axes[0, 1], "net", lambda v, c: np.where(c, v, np.nan), "net electric power [MW]")
    axes[0, 1].set_title("net electric power", fontsize=10)
    axes[0, 2].axis("off")
    text = [f"feasible fraction: det {det['feasible_fraction']:.3f}, robust {rob['feasible_fraction']:.3f}",
            f"failed fraction:   det {det['failed_fraction']:.3f}, robust {rob['failed_fraction']:.3f}",
            f"P(net <= 0):       det {det['p_net_nonpositive']:.3f}, robust {rob['p_net_nonpositive']:.3f}",
            f"coe | feasible mean: det {_fmt(det['coe_where_feasible']['mean'])}, robust {_fmt(rob['coe_where_feasible']['mean'])}",
            f"coe | net > 0 mean:  det {_fmt(det['mean_coe_where_converged_and_positive_net'])}, robust {_fmt(rob['mean_coe_where_converged_and_positive_net'])}",
            f"coe at nominal:      det {det['nominal']['coe']:.1f}, robust {rob['nominal']['coe']:.1f}"]
    axes[0, 2].text(0.0, 0.95, "\n".join(text), va="top", ha="left", fontsize=8, family="monospace", transform=axes[0, 2].transAxes)
    for ax, i in zip(axes[1], worst, strict=False):
        stacked(ax, "g", lambda v, c, i=i: np.where(c, v[:, i], np.nan), f"{names[i]} residual (g <= 0 satisfied)")
        ax.axvline(0.0, color="black", lw=0.8)
        ax.set_title(f"{names[i]}: P(g>0) det {det['p_violated'][e['constraints'][i]]:.3f}, robust {rob['p_violated'][e['constraints'][i]]:.3f}", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / f"ouu_histograms_{histogram_alpha:g}{tag}.png", dpi=160)
    plt.close(fig)
    print(f"[plot] wrote {OUT / f'ouu_histograms_{histogram_alpha:g}{tag}.png'}", flush=True)


# ---------------------------------------------------------------- main


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    def option(name, default, cast=int):
        return cast(argv[argv.index(name) + 1]) if name in argv else default

    common = {"inputs": option("--inputs", "physics", str), "objective": option("--objective", "mean", str),
              "table": option("--table", "new", str), "hfact_sigma": option("--hfact-sigma", HFACT_SIGMA, float),
              "pairing": option("--pairing", "one", str)}
    if "--measure" in argv:
        row = measure(option("--n", 256), option("--mode", "fwd", str), option("--chunk", 256), alpha=option("--alpha", ALPHA, float), seed=option("--seed", 0),
                      inputs=option("--inputs", "all", str), objective=option("--objective", "levelised", str))
        print("OUU_RESULT " + json.dumps(row, default=str), flush=True)
        return 0
    if "--scaling" in argv:
        sizes = tuple(int(s) for s in option("--sizes", "64,256,1024,4096", str).split(","))
        modes = tuple(option("--modes", "fwd,jacrev,jacfwd,grad,jacrev_chunked", str).split(","))
        if unknown := set(modes) - set(MODES):
            raise SystemExit(f"--modes: no such mode {sorted(unknown)}; one of {MODES}")
        scaling(sizes, modes, option("--chunk", 256), option("--timeout", 1800.0, float), sys.executable, "--fresh" in argv)
        return 0
    if "--smoke" in argv:
        smoke(option("--n", 256), option("--alpha", ALPHA, float), option("--max-iter", 200), "--with-c16" in argv,
              option("--eps", EPS_FAILED, float), option("--seed", 0), "--no-warm" not in argv, option("--move-limit", 0.25, float),
              jac=option("--jac", "fwd", str), chunks=option("--chunk", None), tol=option("--tol", 1e-4, float),
              ftol=option("--ftol", 1e-6, float), gtol=option("--gtol", 1e-4, float), name=option("--name", "ouu_smoke", str),
              with_evidence="--evidence" in argv, fresh_seed=option("--fresh-seed", 1), tag=option("--tag", "", str),
              start=Path(option("--start", "", str)) if "--start" in argv else None,
              alpha16=option("--alpha16", None, float) if "--alpha16" in argv else None,
              te_recourse=option("--te-recourse", 0),
              te_range=tuple(float(v) for v in option("--te-range", "", str).split(",")) if "--te-range" in argv else None, **common)
        return 0
    if "--decompose" in argv:
        decompose(option("--n", 256), Path(option("--robust", str(OUT / "ouu_smoke.json"), str)), option("--inputs", "all", str),
                  option("--seed", 0), option("--alpha", ALPHA, float), "--with-c16" in argv,
                  table=common["table"], hfact_sigma=common["hfact_sigma"], name=option("--name", "ouu_decomposition", str),
                  pairing=common["pairing"])
        return 0
    if "--evidence" in argv:
        robust = Path(option("--robust", "", str))
        if not robust.is_file():
            raise SystemExit("--evidence needs --robust <run JSON>")
        payload = json.loads(robust.read_text())
        alpha = option("--alpha", payload.get("alpha", ALPHA), float)
        ouu = build(option("--n", payload.get("n", 256)), alpha=alpha, seed=option("--seed", payload.get("seed", 0)),
                    with_c16="--with-c16" in argv, inputs=option("--inputs", payload.get("inputs", "physics"), str),
                    objective=option("--objective", payload.get("objective", "mean"), str),
                    table=option("--table", payload.get("table", "new"), str), hfact_sigma=option("--hfact-sigma", payload.get("hfact_sigma", HFACT_SIGMA), float),
                    pairing=option("--pairing", payload.get("pairing", "one"), str), alpha16=payload.get("alpha16"),
                    te_recourse=payload.get("te_recourse", 0),
                    te_range=(payload["te_grid"][0], payload["te_grid"][-1]) if payload.get("te_grid") else None)
        x_rob, source = robust_design_of(payload, ouu.design)
        evidence(ouu, make(ouu), ouu.x0, x_rob, option("--fresh-seed", 1), f"ouu_evidence_{alpha:g}{option('--tag', '', str)}", source,
                 extra={"run": str(robust), "converged": payload.get("converged"), "reason": payload.get("reason"),
                        "robust_f": payload.get("robust", {}).get("f")})
        return 0
    if "--sweep-te" in argv:
        robust = Path(option("--robust", "", str))
        if not robust.is_file():
            raise SystemExit("--sweep-te needs --robust <run JSON>")
        payload = json.loads(robust.read_text())
        alpha = option("--alpha", payload.get("alpha", ALPHA), float)
        ouu = build(option("--n", payload.get("n", 256)), alpha=alpha, seed=option("--seed", payload.get("seed", 0)),
                    with_c16=payload.get("with_c16", False), inputs=option("--inputs", payload.get("inputs", "physics"), str),
                    objective=option("--objective", payload.get("objective", "mean"), str),
                    table=option("--table", payload.get("table", "new"), str), hfact_sigma=option("--hfact-sigma", payload.get("hfact_sigma", HFACT_SIGMA), float),
                    pairing=option("--pairing", payload.get("pairing", "one"), str), alpha16=payload.get("alpha16"),
                    te_recourse=payload.get("te_recourse", 0),
                    te_range=(payload["te_grid"][0], payload["te_grid"][-1]) if payload.get("te_grid") else None)
        x_rob, source = robust_design_of(payload, ouu.design)
        sweep_te(ouu, make(ouu), x_rob, option("--name", f"ouu_sweep_te_{robust.stem}", str), points=option("--points", 16), source=source)
        return 0
    if "--plot" in argv:
        plot(Path(option("--from", str(OUT.parent / "out_cluster"), str)), histogram_alpha=option("--alpha", 0.9, float), tag=option("--tag", "", str))
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
