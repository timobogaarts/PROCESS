"""A driven, pulsed tokamak under uncertainty: the graph rewrites the flexibility
study of `large_tokamak_nof` needs, and its two-stage assembly.

PROCESS's DEMO is *driven*, not ignited: 75 MW of ECRH heating during the burn
(`p_hcd_primary_extra_heat_mw`) plus the current-drive power a 40 % non-inductive
fraction needs, capped by `c30` at `p_hcd_injected_max = 200` MW, with the confinement
multiplier `hfact` the iteration variable that closes the power balance `c2`. The
stellarator study (`ouu.two_stage`, `paper_tests/flexibility.py`) fixes the build and
lets the operator re-optimise per belief draw; this module is what the tokamak needs
on top of it, each piece one graph operation on the problem graph (`mdf.mdf_graph`:
the cut MDA with the condition and objective nodes):

- **`installed_power`** -- PROCESS conflates the heating power *used* in the burn with
  the heating power *installed*: the cost account `c2231 = ucech x
  p_hcd_ecrh_injected_total_mw` (`process/models/costs/costs.py:1866-1880`) bills the
  used power, and `c30` caps it by an input nothing costs. Two `Rewire`s make the
  installed power one boundary input, `INSTALLED` (`.current_drive.p_hcd_installed_mw`),
  a first-stage build decision: the cost node reads it in place of the used power, and
  `c30` reads it in place of `p_hcd_injected_max` -- so per world the operator may use up
  to what was bought, and what was bought is what is paid for. The used power stays
  where it was for everything else (the recirculating power, `process/models/power.py`,
  reads the used power's wall-plug electricity, as it should).
- **`beta_limit_factor`** -- `c24` compares the beta against `beta_vol_avg_max`, which
  on the tokamak branch is not an input: `.tokamak.plasma_beta.limit` computes it from
  `beta_norm_max`, itself computed by `.tokamak.plasma_beta.norm_max` as Wesson's
  `4 l_i` (`i_beta_norm_max = 1`, the default; the file's `beta_norm_max = 3.0` is dead).
  A belief about the beta limit's threshold has nothing to sample, so one `Redefine`
  gives the Wesson node a second read, `F_BETA_NORM_MAX` (`.physics.f_beta_norm_max`,
  nominal 1), and the body `f x 4 l_i`.
- **`freeze`** -- a build quantity a model computes from per-world quantities (the PF
  coil currents and turns from the plasma's equilibrium, the CS swing from the flux the
  world's plasma consumes) is re-sized in every world, which violates
  non-anticipativity; `lift.lift` takes a *root find* apart, and these are explicit
  rules. `freeze` is the explicit rule's lift: a `Cut` of the place for the readers
  that size the build from it, minting `^built.<place>` -- a first-stage boundary
  input, fixed at the deterministic design -- while the per-world requirement is still
  computed and, where the caller says so, compared against what was built as an
  inequality `^cond.built.<place> = (required - built) / built <= 0`.

`two_stage` is `ouu.two_stage` for this problem: the same `TwoStage` record, the same
`ouu.make`, but the closure is `closing.close_conditions` under the study's rewrites
(three pairings: `c1` by the beta, `c2` by the heating power, `c62` by the thermal alpha
fraction), the kind table is `configurations.kinds_tokamak`'s, and the design is the
build plus the installed power.
"""

from __future__ import annotations

import dataclasses
import time
from typing import TYPE_CHECKING

import jax

jax.config.update("jax_enable_x64", True)  # before any array: PROCESS is float64

import numpy as np  # noqa: E402
from cottax.core.plan.graphops import Redefine, Rewire  # noqa: E402
from cottax.execution import RunnableGraph  # noqa: E402
from cottax.execution.schedule import Schedule  # noqa: E402
from cottax.pytree.names import MintKey, PathMap, prefix_path  # noqa: E402
from cottax.pytree.nodes import ImplementedFunction  # noqa: E402
from cottax.pytree.plan import Insert, Plan  # noqa: E402
from cottax.pytree.problem import Converged, Steps, unknowns_of  # noqa: E402
from cottax.pytree.rewrites import Cut  # noqa: E402
from cottax.pytree.spec import NodePath, VarPath  # noqa: E402
from jax.tree_util import GetAttrKey  # noqa: E402

from functional_process.configurations import kinds_tokamak  # noqa: E402
from functional_process.configurations.kinds import Belief, Kind  # noqa: E402
from functional_process.cottax.architectures import (  # noqa: E402
    beliefs as beliefs_,
)
from functional_process.cottax.architectures import (  # noqa: E402
    closing,
    mdf,
    ouu,
    sand,
    stages,
)
from functional_process.cottax.architectures.mda import guess_sources  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Iterable

    from cottax.pytree.graph import Graph

NAME = "large_tokamak_nof"

# ---------------------------------------------------------------- the ops


class Retarget(Rewire):
    """Point named reads of named nodes at other places: `((node, old, new), ...)`.
    cottax's `Rewire` with the renaming said outright (`Cut` derives its own from a
    producer; here the new place need not be produced by anyone -- it may be a new
    boundary input).
    """

    renamings: tuple[tuple[NodePath, VarPath, VarPath], ...]

    def renaming(self, graph):
        out: dict = {}
        for node, old, new in self.renamings:
            out.setdefault(node, {})[old] = new
        return out

    def __repr__(self) -> str:
        return (
            "retarget("
            + ", ".join(
                f"{n.spelling}: {o.spelling} -> {w.spelling}"
                for n, o, w in self.renamings
            )
            + ")"
        )


class Replace(Redefine):
    """One node redefined in place: `node` gets `definition`."""

    node: NodePath
    definition: object

    def redefined(self, graph):
        return {self.node: self.definition}

    def __repr__(self) -> str:
        return f"replace({self.node.spelling})"


def _node(*names: str) -> NodePath:
    return NodePath(tuple(GetAttrKey(n) for n in names))


def _var(*names: str) -> VarPath:
    return VarPath(tuple(GetAttrKey(n) for n in names))


# ---------------------------------------------------------------- installed power

INSTALLED = _var("current_drive", "p_hcd_installed_mw")
"""The installed heating and current-drive power (MW): a first-stage build decision,
what the plant bought."""
USED_ECRH = _var("current_drive", "p_hcd_ecrh_injected_total_mw")
"""What the cost node billed: the ECRH power used (heating + current drive)."""
USED_TOTAL = _var("current_drive", "p_hcd_injected_total_mw")
"""What `c30` caps: every injected power."""
CAP = _var("current_drive", "p_hcd_injected_max")
"""`c30`'s limit input, 200 MW in the file, costed by nothing."""
COST_NODE = _node("costs", "power_injection_cost")
CONSTRAINT30 = _node("Constraint30")
HEATING = _var("current_drive", "p_hcd_primary_extra_heat_mw")
"""The heating power during the burn, 75 MW in the file: the closure of `c2`."""
CD_POWER = _var("current_drive", "p_hcd_primary_injected_mw")
"""The current-drive power, from `f_c_plasma_non_inductive` as in PROCESS."""


def installed_power(graph: Graph) -> Graph:
    """The cost node and `c30` re-wired to `INSTALLED`: what is bought is what is paid
    for and what the operator may use.
    """
    return Retarget((
        (COST_NODE, USED_ECRH, INSTALLED),
        (CONSTRAINT30, CAP, INSTALLED),
    )).apply(graph)


# ---------------------------------------------------------------- the beta limit

F_BETA_NORM_MAX = _var("physics", "f_beta_norm_max")
"""A factor on Wesson's `beta_N,max = 4 l_i`: nominal 1, the beta limit's threshold as
a belief."""
LI = _var("physics", "ind_plasma_internal_norm")
BETA_NORM_MAX = _var("physics", "beta_norm_max")
NORM_MAX = _node("tokamak", "plasma_beta", "norm_max")


@dataclasses.dataclass(frozen=True)
class ScaledWesson:
    """`f * 4 l_i`: Wesson's limit with a factor on it."""

    def __call__(self, ind_plasma_internal_norm, f_beta_norm_max):
        return f_beta_norm_max * 4.0 * ind_plasma_internal_norm


def beta_limit_factor(graph: Graph) -> Graph:
    """`.tokamak.plasma_beta.norm_max` redefined to read `F_BETA_NORM_MAX` too."""
    node = graph[NORM_MAX]
    if tuple(v.spelling for v in node.reads) != (LI.spelling,) or tuple(
        v.spelling for v in node.owns
    ) != (BETA_NORM_MAX.spelling,):
        raise ValueError(
            f"{NORM_MAX.spelling} is not Wesson's node (reads {[v.spelling for v in node.reads]}, "
            f"owns {[v.spelling for v in node.owns]}); the factor is written for `4 l_i`"
        )
    return Replace(
        NORM_MAX,
        ImplementedFunction((LI, F_BETA_NORM_MAX), (BETA_NORM_MAX,), ScaledWesson()),
    ).apply(graph)


# ---------------------------------------------------------------- freezing a build output

BUILT = MintKey("built")
"""The namespace a frozen build output's copy is minted in: `.pf_coil.n_pf_coil_turns`
-> `^built.pf_coil.n_pf_coil_turns`."""
COND = MintKey("cond")
CHECK = _node("Built")


def built_of(var: VarPath) -> VarPath:
    """`^built.<var>`."""
    return prefix_path(var, BUILT)


def check_condition_for(var: VarPath) -> VarPath:
    """`^cond.built.<var>`: the capacity inequality of a frozen place."""
    return prefix_path(VarPath((GetAttrKey("built"), *var.segments)), COND)


def check_place_for(var: VarPath) -> NodePath:
    """`.Built.<the place's last component>`."""
    return NodePath((*CHECK.segments, GetAttrKey(var.spelling.split(".")[-1])))


@dataclasses.dataclass(frozen=True)
class Capacity:
    """`(|required| - |built|) / max|built|`, elementwise then the worst: negative when
    what the world asks for fits in what was built. Magnitudes, since a PF coil's
    current has a sign; array-valued places (one per coil) reduce to their maximum so
    the condition is one scalar, normalised like PROCESS's own residuals.
    """

    sign: float = 1.0

    def __call__(self, required, built):
        import jax.numpy as jnp  # noqa: PLC0415

        required, built = jnp.abs(jnp.asarray(required)), jnp.abs(jnp.asarray(built))
        gap = self.sign * (required - built) / jnp.maximum(jnp.max(built), 1e-30)
        return jnp.max(gap)


@dataclasses.dataclass(frozen=True)
class Freeze:
    """One build output to freeze: the place, which readers are rewired (`None`: all
    of them), and whether a capacity check is stated (`check`: `None` for none;
    `"above"` when the world's requirement must not exceed what was built; `"below"`
    when it must not fall short).
    """

    place: str
    check: str | None = None
    why: str = ""
    readers: tuple[str, ...] | None = None


def freeze(
    graph: Graph,
    var: VarPath,
    *,
    check: str | None = None,
    readers: Iterable[str] | None = None,
) -> tuple[Graph, dict]:
    """`var`'s readers (all of them, or the node spellings in `readers`) rewired to
    `^built.<var>` (a `Cut` with the `BUILT` mint), so the value the rest of the
    machine is sized from is a first-stage input, and the producer still computes
    the per-world requirement. With `check`, a node `.Built.<name>` is inserted
    owning `^cond.built.<var>`, the capacity inequality.

    Returns the graph and a report: the ops, the readers rewired, the condition.

    Raises
    ------
    KeyError
        If nothing produces `var`, or nothing reads it.
    """
    wanted = None if readers is None else set(readers)
    readers = tuple(
        n
        for n, d in graph.definitions.items()
        if var in d.reads and (wanted is None or n.spelling in wanted)
    )
    if not readers:
        raise KeyError(f"nothing reads {var.spelling}, so there is nothing to freeze")
    plan = Plan(graph) + Cut(var, readers, BUILT)
    condition = None
    if check is not None:
        condition = check_condition_for(var)
        plan += Insert(
            PathMap((
                (
                    check_place_for(var),
                    ImplementedFunction(
                        (var, built_of(var)),
                        (condition,),
                        Capacity(1.0 if check == "above" else -1.0),
                    ),
                ),
            ))
        )
    report = {
        "place": var.spelling,
        "built": built_of(var).spelling,
        "readers": tuple(r.spelling for r in readers),
        "check": check,
        "condition": None if condition is None else condition.spelling,
        "ops": tuple(repr(op) for op in plan.ops),
    }
    return plan.graph, report


FREEZES: tuple[Freeze, ...] = (
    Freeze(
        ".pf_coil.c_pf_cs_coils_peak_ma",
        check="above",
        why="the PF and CS coils' rated currents: what the coils, their conductor, "
        "their masses and their power supplies are built for; per world the "
        "equilibrium asks for its own currents, which must fit (|I_world| <= |I_built| "
        "per coil, `^cond.built.pf_coil.c_pf_cs_coils_peak_ma`)",
        readers=(
            ".costs.pf_magnet_cost",
            ".tokamak.pf_coil.sizes",
            ".tokamak.pf_coil.masses",
            ".power.pf_coil_power",
        ),
    ),
    *(
        Freeze(
            f".pf_coil.{name}[{k}]",
            why="the PF coil's design peak field, what its conductor is sized at; the "
            "world's field is still computed, and the current check stands for it",
            readers=(
                ".tokamak.pf_coil.masses",
                ".tokamak.pf_coil.strand_critical_current",
            ),
        )
        for name in ("b_pf_coil_peak", "bpf2")
        for k in range(6)
    ),
    Freeze(
        ".physics.ind_plasma",
        why="the plasma self-inductance is geometry and l_i, both held; its owner also "
        "owns the volt-seconds, so it reads as second stage by node granularity alone",
        readers=(".tokamak.pf_coil.inductance",),
    ),
    Freeze(
        ".fwbs.f_ster_div_single",
        why="the divertor's solid-angle fraction is geometry; its owner reads the "
        "separatrix power",
    ),
    Freeze(
        ".first_wall.a_fw_inboard",
        why="first-wall area: geometry, owned beside the neutron flux",
    ),
    Freeze(
        ".first_wall.a_fw_outboard",
        why="first-wall area: geometry, owned beside the neutron flux",
    ),
    Freeze(
        ".first_wall.a_fw_total",
        why="first-wall area: geometry, owned beside the neutron flux",
    ),
    Freeze(
        ".heat_transport.helpow",
        why="the cryoplant load sizes the cryogenic building; built once, at the nominal",
        readers=(".buildings.sizing",),
    ),
    *(
        Freeze(
            f".vacuum.{name}",
            why="the vacuum system is built once (the stellarator's `QUANTILE` decision); "
            "no active constraint checks the world's pumping requirement",
        )
        for name in (
            "n_vac_pumps_high",
            "n_vv_vacuum_ducts",
            "dlscal",
            "m_vv_vacuum_duct_shield",
            "dia_vv_vacuum_ducts",
        )
    ),
)
"""Every build output the stage split found re-sized per world, and what was done:
one capacity check (the coil currents) and fourteen exact freezes (geometry a node
happens to own beside a per-world quantity, and the cost-only vacuum system)."""


# ---------------------------------------------------------------- the study's rewrite


def study_rewrite(graph: Graph) -> Graph:
    """`installed_power` then `beta_limit_factor`: what `close_conditions` is handed."""
    return beta_limit_factor(installed_power(graph))


PAIRINGS: dict[str, str] = {
    "^cond.constraints.c2": HEATING.spelling,
    "^cond.constraints.c62": ".physics.f_nd_alpha_thermal_electron",
    "^cond.constraints.c1": ".physics.beta_total_vol_avg",
}
"""What closes what, per world: the beta consistency by the beta (a two-node cycle:
PROCESS's `ixc 5` is a copy of a computed quantity), the power balance by the heating
power (six nodes: the injected power, the loss power, the scaling, its tail), the helium
particle balance -- `c62`, an inequality held with equality -- by the thermal alpha
fraction (28 nodes: the dilution, the fusion rates, the radiation, the confinement
time). `c2` and `c62` share a cycle and are flattened into one square problem with the
fuel-ion Picard's two copies."""

CLOSING_BOUNDS: dict[str, tuple[float, float]] = {
    HEATING.spelling: (0.0, 200.0),
}
"""Bounds for the closing variable the reference has none for: the heating power
cannot be negative (a world whose balance asks for negative heating at the operator's
point has no operating point there), and the file's cap is the span the lower bound is
normalised by (`ouu.RecourseBound`)."""

KNOBS: tuple[str, ...] = kinds_tokamak.KNOBS


PROCESS_PAIRINGS: dict[str, str] = {
    "^cond.constraints.c2": HEATING.spelling,
    "^cond.constraints.c1": ".physics.beta_total_vol_avg",
}
"""PROCESS's own point, re-closed: `c2` by the heating power and `c1` by the beta, the
helium fraction left at PROCESS's value -- what `process_point` evaluates, the
deterministic design as PROCESS reports it, in the port, under the study's rewrites."""


def process_point(session, design_values, closing_values, driver=None) -> dict:
    """The port at PROCESS's converged design: `close_conditions` under
    `PROCESS_PAIRINGS` and `study_rewrite`, seeded at `design_values` (one per
    `ixc` minus the two closing variables) and `closing_values`, primed once.
    Returns the primed env's values by spelling (`out`), the heating power the balance
    asks for at PROCESS's `hfact` (`heating`, 75 MW if the port reproduces PROCESS's
    `c2`), the current-drive power (`cd`), their sum (`installed`: what the study
    installs, zero margin), the closing reports and every condition's value.
    """
    built = closing.close_conditions(
        session,
        PROCESS_PAIRINGS,
        flatten=True,
        driver=driver,
        rewrite=study_rewrite,
        bounds=CLOSING_BOUNDS,
    )
    live = built.session
    env = closing.seed(built, live.reference.cold, design_values, closing_values)
    env[F_BETA_NORM_MAX] = mdf._not_weak(1.0)
    # Only `c30` and the cost node read the installed power; primed twice, the
    # second time at the used power, so the cost and `c30` are PROCESS's own.
    env[INSTALLED] = mdf._not_weak(200.0)
    _primed, out = mdf.prime(built.problem, env)
    heating = float(np.asarray(out[HEATING]))
    cd = float(np.asarray(out[CD_POWER]))
    env[INSTALLED] = mdf._not_weak(heating + cd)
    _primed, out = mdf.prime(built.problem, env)
    values = {v.spelling: np.asarray(out[v]) for v in out}
    return {
        "built": built,
        "out": values,
        "heating": heating,
        "cd": cd,
        "installed": heating + cd,
        "reports": {c.spelling: r for c, r in built.root_find_reports(out).items()},
        "conditions": {
            c.spelling: float(np.asarray(out[c]))
            for c in (
                *built.report["equalities"],
                *built.report["inequalities"],
                built.report["objective"],
            )
        },
    }


def close(session, driver=None) -> closing.Closed:
    """`closing.close_conditions` under `study_rewrite` and `PAIRINGS`, flattened,
    driven by `closing.safeguarded()` unless `driver` says otherwise.
    """
    return closing.close_conditions(
        session,
        PAIRINGS,
        flatten=True,
        driver=driver,
        rewrite=study_rewrite,
        bounds=CLOSING_BOUNDS,
    )


def seed(
    built: closing.Closed, design_values, closing_values, *, installed=None
) -> dict:
    """`closing.seed` plus the new inputs: `F_BETA_NORM_MAX` at 1, `INSTALLED` at
    `installed` (or, `None`, at the used power the primed nominal finds: the heating
    power that closes `c2` plus the current-drive power -- zero margin, PROCESS's own
    point).
    """
    live = built.session
    env = closing.seed(built, live.reference.cold, design_values, closing_values)
    env[F_BETA_NORM_MAX] = mdf._not_weak(1.0)
    if installed is None:
        env[INSTALLED] = mdf._not_weak(0.0)
        _primed, out = mdf.prime(built.problem, env)
        installed = float(np.asarray(out[HEATING])) + float(np.asarray(out[CD_POWER]))
    env[INSTALLED] = mdf._not_weak(float(installed))
    return env


# ---------------------------------------------------------------- the two-stage problem


def two_stage(
    session,
    *,
    beliefs: Iterable[Belief] = kinds_tokamak.BELIEFS,
    held: Iterable[str] = (),
    alpha: float = ouu.ALPHA,
    n: int = 256,
    seed_: int = 0,
    design_values=None,
    closing_values=None,
    installed=None,
    freezes: Iterable[Freeze] = (),
    knobs: Iterable[str] = KNOBS,
    extra_columns: Iterable[str] = (),
    driver=None,
) -> ouu.TwoStage:
    """`ouu.two_stage` for the driven tokamak. The closed MDA is `close(session)`;
    `freezes` are applied to it (`freeze`, each frozen place's `^built.*` copy seeded
    at the nominal's own value and joining the design as a build place, each capacity
    check joining the constraints); the design is the closed problem's minus every
    sampled place, plus `INSTALLED`, plus the frozen copies; the split is at the
    sampled leaves, the closing start ports and `knobs` -- the operating leaves the
    operator moves per world.

    Returns an `ouu.TwoStage` that `ouu.make(model, hoist=...)` compiles.
    """
    began = time.perf_counter()
    built = close(session, driver)
    live = built.session
    env = seed(built, design_values, closing_values, installed=installed)
    frozen_reports: dict = {}
    frozen_vars: list = []
    freezes = tuple(freezes)
    if freezes:
        _primed, roots = mdf.prime(built.problem, env)
        driven, undriven = built.graph, built.problem.graph
        conditions = []
        for f in freezes:
            var = next(v for v in driven.graph.owners if v.spelling == f.place)
            driven, report = freeze(driven, var, check=f.check, readers=f.readers)
            undriven, _ = freeze(undriven, var, check=f.check, readers=f.readers)
            report["why"] = f.why
            frozen_reports[f.place] = report
            frozen_vars.append(built_of(var))
            env[built_of(var)] = roots[var]
            if report["condition"] is not None:
                conditions.append(
                    next(
                        c
                        for c in driven.graph.owners
                        if c.spelling == report["condition"]
                    )
                )
        old = built.problem
        schedule = Schedule(RunnableGraph(driven))
        report = dict(
            old.report,
            inequalities=tuple(old.report["inequalities"]) + tuple(conditions),
            freezes=frozen_reports,
            blocks=len(driven.graph.components),
        )
        problem = dataclasses.replace(
            old,
            graph=undriven,
            eager=schedule,
            traceable=schedule,
            design=tuple(old.design) + tuple(frozen_vars),
            conditions=tuple(old.conditions) + tuple(conditions),
            n_inequality=old.n_inequality + len(conditions),
            report=report,
        )
        built = dataclasses.replace(built, problem=problem, design=problem.design)
    problem = built.problem
    env, primed = mdf.prime(problem, env)
    point = PathMap(mdf._inputs_only(problem, env).items())
    var_of = {v.spelling: v for v in point}
    var_of.update({k.spelling: k for k in primed})

    held = tuple(held)
    rows, dropped, nominal = [], [], {}
    for belief in beliefs_.sampled(tuple(beliefs), held):
        if belief.path == "dummy":
            rows.append(belief)
            continue
        var = var_of.get(belief.path)
        if var is None or var not in point:
            dropped.append(belief.path)
            continue
        rows.append(belief)
        nominal[belief.path] = np.asarray(point[var])
    rows = tuple(rows)
    uncertain = {var_of[b.path] for b in rows if b.path != "dummy"}

    # The design: the closed problem's places minus every sampled one, plus the
    # installed power. A frozen copy (`^built.*`, possibly an array -- one value per
    # PF coil) is a build input held at the nominal in `point`, not a design place:
    # nothing here optimises the build, and the flat design vector is scalar.
    design = tuple(
        v
        for v in built.design
        if v not in uncertain and not v.spelling.startswith("^built")
    ) + (INSTALLED,)
    bounds = {v: (lo, hi) for v, lo, hi in live.reference.bounds}
    ixc_of = {sand.iteration_variable_path(i): i for i in live.reference.ixc}
    bounds[INSTALLED] = (0.0, 1000.0)
    ixc_of[INSTALLED] = 0
    x0 = np.array([float(np.asarray(point[v])) for v in design])
    lower = np.array([bounds[v][0] for v in design])
    upper = np.array([bounds[v][1] for v in design])

    constraints = tuple(problem.report["inequalities"])
    for var in built.closing:
        lo_hi = bounds.get(var) or CLOSING_BOUNDS.get(var.spelling)
        if lo_hi is not None and var.spelling in kinds_tokamak.RECOURSE_BOUNDED:
            lo, hi = lo_hi
            constraints += (ouu.RecourseBound(var, "lower", lo, hi),)
    place = next(iter(built.places.values()))
    unknowns, guesses = _closing_ports(built)
    extra = ouu.resolve_columns(extra_columns, var_of)
    u0 = np.array([float(np.asarray(primed[u])) for u in unknowns])
    Theta, theta = ouu.sample(rows, var_of, nominal, n, seed_)
    starts0 = np.tile(u0, (n + 1, 1))

    table = dict(kinds_tokamak.KINDS)
    varying = tuple(g.spelling for g in guesses) + tuple(knobs)
    for spelling in varying:
        table.setdefault(spelling, Kind.OPERATING)
    table[INSTALLED.spelling] = Kind.BUILD
    table[F_BETA_NORM_MAX.spelling] = Kind.BELIEF
    for var in frozen_vars:
        table[var.spelling] = Kind.BUILD
    for var in built.design:
        table.setdefault(var.spelling, Kind.BUILD)
    graph = built.graph
    split = stages.split(
        graph,
        stages.leaves(
            graph,
            table,
            sampled=tuple(b.path for b in rows if b.path != "dummy"),
            varying=varying,
        ),
    )
    first = Schedule(RunnableGraph(stages.first_stage_graph(graph, split)))
    recourse = Schedule(RunnableGraph(stages.recourse_graph(graph, split)))
    verdicts = ouu.other_verdicts(recourse, place)
    columns = (
        var_of[ouu.COE],
        var_of[ouu.NET],
        var_of[ouu.AVAIL],
        var_of[ouu.CONCOST],
        *constraints,
        Steps.name_for(place),
        Converged.name_for(place),
        *unknowns,
        *verdicts,
        *extra,
    )
    return ouu.TwoStage(
        closed=built,
        closure="flattened",
        point=point,
        nominal_out=primed,
        var_of=var_of,
        beliefs=rows,
        dropped=tuple(dropped),
        nominal=nominal,
        design=design,
        ixc=tuple(ixc_of[v] for v in design),
        x0=x0,
        lower=lower,
        upper=upper,
        constraints=constraints,
        columns=columns,
        unknowns=unknowns,
        guesses=guesses,
        place=place,
        verdicts=verdicts,
        extra=extra,
        alpha=alpha,
        n=n,
        seed=seed_,
        rated_mw=float(np.asarray(point[var_of[ouu.RATED]])),
        alpha16=None,
        te_grid=None,
        objective="levelised",
        pairing="tokamak",
        held=held,
        Theta=Theta,
        theta=theta,
        starts0=starts0,
        stages=split,
        first=first,
        recourse=recourse,
        build_s=time.perf_counter() - began,
    )


def _closing_ports(built: closing.Closed):
    """Every closing problem's unknowns and their start ports, parallel, the combined
    `c2`/`c62` problem first (it is `ouu`'s `place`, whose `Steps`/`Converged` are
    columns) and `c1`'s after -- its verdict is among `verdicts`.
    """
    ports = {u: g for g, u in guess_sources(built.graph).items()}
    unknowns: list = []
    for p in dict.fromkeys(built.places.values()):
        for u in unknowns_of(built.graph[p]):
            if u not in unknowns:
                unknowns.append(u)
    return tuple(unknowns), tuple(ports[u] for u in unknowns)


__all__ = [
    "BUILT",
    "CAP",
    "CD_POWER",
    "CLOSING_BOUNDS",
    "COST_NODE",
    "FREEZES",
    "F_BETA_NORM_MAX",
    "HEATING",
    "INSTALLED",
    "KNOBS",
    "NAME",
    "NORM_MAX",
    "PAIRINGS",
    "PROCESS_PAIRINGS",
    "USED_ECRH",
    "USED_TOTAL",
    "Capacity",
    "Freeze",
    "Replace",
    "Retarget",
    "ScaledWesson",
    "beta_limit_factor",
    "built_of",
    "check_condition_for",
    "close",
    "freeze",
    "installed_power",
    "process_point",
    "seed",
    "study_rewrite",
    "two_stage",
]
