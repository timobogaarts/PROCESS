"""What the sequential split and ALC share: the Helias problem partitioned (by reach for
the sequential split, by design context into build, operation and economics for ALC),
the AAO base every recipe starts from, and several problems driven and valued as
`study` drives and values its one top.

Nothing here decides an architecture. `partition` and `disciplines` read the graph;
`base` reads AAO's declared problem and AAO's own recipe on it; `drive`, `values` and
`report` are `study.block_plan`, `study.block_values` and `study.answer` for several tops.

Run from `PROCESS/` with

    JAX_PLATFORMS=cpu PYTHONPATH=~/projects/jaxgraph/src:. \
        ~/anaconda3/envs/func_PROCESS_env/bin/python paper_tests/distributed_architectures/<script>.py
"""

from __future__ import annotations

import copy
import dataclasses
import time

import jax

jax.config.update("jax_enable_x64", True)

import cottax.assemblies.jax as cx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from cottax.core import Plan, Violation, unminted  # noqa: E402
from cottax.core.frozen import OrderedMap  # noqa: E402
from cottax.relational import programme  # noqa: E402

from functional_process.configurations import load  # noqa: E402
from functional_process.cottax import study  # noqa: E402
from functional_process.cottax.architectures.contracts import (  # noqa: E402
    SURFACE,
    Converged,
    PortMinimisation,
    Start,
    Status,
    Steps,
)
from functional_process.cottax.architectures.drivers import VmconDriver  # noqa: E402
from functional_process.cottax.input.native import state_of  # noqa: E402
from functional_process.cottax.models.constraints import MERIT, held  # noqa: E402
from functional_process.cottax.problem import TOP, problem_model  # noqa: E402
from functional_process.cottax.truth import ground_truth  # noqa: E402

CONFIGURATION = "stellarator_helias"

MAGNET_CONSTRAINTS = (32, 34, 35, 65, 82, 83)
"""The TF-coil constraints: conduit stress, dump voltage, hot-spot J_wp/J_prot,
vacuum-vessel stress at quench, and the two stellarator coil limits. Every other
constraint of the file, and the merit, is the plasma's."""


def cid(constraint) -> int:
    """A constraint declaration's PROCESS id, off the place it holds: `.constraints.c<id>`."""
    return int(held(constraint).spelling.rsplit(".c", 1)[1])


def short(place) -> str:
    return place.spelling.rsplit(".", 1)[1]


# ---------------------------------------------------------------- the partition, read off the graph


@dataclasses.dataclass(frozen=True)
class Partition:
    """The file's problem split by reach on the machine's graph (no problem stated).

    The magnet side is every model a magnet constraint reads, at any depth, and every
    model that reads a magnet-local variable and no plasma one, with what those read:
    each depends on the magnet's design alone. The plasma side is every other model
    that depends on the design. A model depending on no design variable is context,
    run once before either."""

    design: OrderedMap
    """Every design variable of the file, with its bounds, in the file's order."""
    magnet_design: tuple
    """What the magnet constraints reach: the shared and the magnet-local variables."""
    plasma_design: tuple
    """The rest."""
    magnet_constraints: tuple
    plasma_equalities: tuple
    plasma_inequalities: tuple
    magnet_nodes: frozenset
    plasma_nodes: frozenset
    crossing: OrderedMap
    """Each place the magnet side computes or owns (a magnet design variable) that a
    plasma model reads, with the plasma models that read it: the coupling, one way."""
    back: tuple
    """Every place the plasma side computes that a magnet model reads: none, for a split
    that runs magnet first."""

    @property
    def shared(self) -> tuple:
        """The magnet's design variables a plasma model reads directly."""
        return tuple(d for d in self.magnet_design if d in self.crossing)

    @property
    def magnet_local(self) -> tuple:
        return tuple(d for d in self.magnet_design if d not in self.crossing)


def partition(configuration, magnet_ids=MAGNET_CONSTRAINTS) -> Partition:
    problem = configuration.problem
    graph = study.machine_model(configuration).graph
    owners = graph.graph.owners
    design = OrderedMap(problem.design.items())
    reach = {d: set(graph.graph.reach([d])) for d in design}
    depends = {n: {d for d in design if n in reach[d]} for n in graph.nodes}

    constraints = (*problem.equalities, *problem.inequalities)
    magnet_c = tuple(c for c in constraints if cid(c) in magnet_ids)
    reached = set().union(*(depends[owners[held(c)]] for c in magnet_c))
    magnet_design = tuple(d for d in design if d in reached)
    plasma_design = tuple(d for d in design if d not in reached)
    # Every magnet variable reaches the merit, so reach alone does not say which are
    # shared: a shared one is read directly by a model that depends on the plasma too.
    shared = _read_by_plasma_directly(graph, depends, magnet_design)
    local = set(magnet_design) - shared

    seeds = {owners[held(c)] for c in magnet_c}
    seeds |= {n for n in graph.nodes if depends[n] & local and depends[n] <= set(magnet_design)}
    closed = seeds | set(graph.graph.ancestors(list(seeds)))
    magnet_nodes = frozenset(n for n in closed if depends[n])
    plasma_nodes = frozenset(n for n in graph.nodes if depends[n] and n not in magnet_nodes)
    if bad := [n for n in magnet_nodes if depends[n] & set(plasma_design)]:
        raise ValueError(f"magnet models depending on a plasma variable: {bad}")

    magnet_owned = {p for p, o in owners.items() if o in magnet_nodes} | set(magnet_design)
    crossing: dict = {}
    for n in sorted(plasma_nodes, key=str):
        for r in graph[n].reads:
            if r in magnet_owned:
                crossing.setdefault(r, []).append(n)
    order = [*magnet_design, *sorted((p for p in crossing if p not in design), key=lambda p: p.spelling)]
    crossing = OrderedMap((p, tuple(crossing[p])) for p in order if p in crossing)
    back = tuple(sorted({r for n in magnet_nodes for r in graph[n].reads
                         if owners.get(r) in plasma_nodes}, key=lambda p: p.spelling))
    return Partition(
        design=design,
        magnet_design=magnet_design,
        plasma_design=plasma_design,
        magnet_constraints=magnet_c,
        plasma_equalities=tuple(c for c in problem.equalities if c not in magnet_c),
        plasma_inequalities=tuple(c for c in problem.inequalities if c not in magnet_c),
        magnet_nodes=magnet_nodes,
        plasma_nodes=plasma_nodes,
        crossing=crossing,
        back=back,
    )


def _read_by_plasma_directly(graph, depends, magnet_design) -> set:
    """The magnet variables read by a model that also depends on a non-magnet variable:
    the shared ones, `b_t` and `rmajor` on Helias."""
    m = set(magnet_design)
    return {r for n in graph.nodes if depends[n] - m for r in graph[n].reads if r in m}


def reach_table(configuration) -> str:
    """Which design variables each constraint and the merit depend on, on the machine's
    graph: the table the split is read off."""
    problem = configuration.problem
    graph = study.machine_model(configuration).graph
    owners = graph.graph.owners
    design = list(problem.design)
    reach = {d: set(graph.graph.reach([d])) for d in design}
    width = 18
    lines = [" " * width + "".join(f"{short(d)[:9]:>10}" for d in design)]
    rows = [MERIT, *(held(c) for c in (*problem.equalities, *problem.inequalities))]
    for place in rows:
        o = owners[place]
        lines.append(f"{short(place):{width}}" + "".join(
            f"{'X' if o in reach[d] else '.':>10}" for d in design))
    return "\n".join(lines)


# ---------------------------------------------------------------- build, operation, economics


BUILD, OPERATION, ECONOMICS = "build", "operation", "economics"
DISCIPLINES = (BUILD, OPERATION, ECONOMICS)
"""In the order the data flows: nothing a later one computes is read by an earlier one."""

DISCIPLINE_CONSTRAINTS = {
    BUILD: MAGNET_CONSTRAINTS,
    OPERATION: (2, 8, 17, 18, 24, 62, 67),
    ECONOMICS: (16,),
}
"""Build: the coil limits. Operation: power balance, beta, the alpha confinement ratio,
neutron and radiation wall loads, radiation fraction, divertor heat load. Economics:
net electric output, beside COE."""

ECONOMICS_AREAS = (".costs.", ".power.", ".buildings.", ".availability.", ".vacuum.",
                   ".stellarator.fwbs.")
"""The plant: costing, the plant power balance, buildings, availability, pumping, and the
blanket and shield sized to the loads operation hands over."""

NOT_ECONOMICS = (".stellarator.fwbs.cryostat_and_vv_geometry",)
"""The vessel and cryostat geometry: build's, though filed under `fwbs`."""

ALSO_BUILD = (".stellarator.coils.tf_cryo_area",)
"""The coil cryostat area: a coil dimension, though the coil constraints do not read it."""

CONSTANT_LINKS = frozenset({
    ".build.dr_fw_inboard", ".build.dr_fw_outboard", ".physics.eps", ".physics.m_fuel_amu",
    ".stellarator.f_st_n_coils", ".tfcoil.n_tf_coils"})
"""Places one discipline computes and another reads that no design variable moves (each
bit-identical with every design variable perturbed by 1 %, `constant_links` re-checks):
read where they are computed, no copy and no consistency constraint."""


@dataclasses.dataclass(frozen=True)
class Link:
    """One quantity `source` computes (or owns, a design variable) and `holder` reads:
    `holder` holds a copy of it, read by `readers` in place of `place`, and `place` is
    held equal to the copy."""

    place: object
    source: str
    holder: str
    readers: tuple


@dataclasses.dataclass(frozen=True)
class Disciplines:
    """The file's problem split by design context: each design-dependent model the
    constraints or the merit need, each constraint and the merit given to one discipline,
    and the quantities crossing between them, one way only."""

    design: OrderedMap
    """Every design variable of the file, with its bounds, in the file's order."""
    designs: dict
    """Each discipline's own design variables; economics has none, only copies."""
    constraints: dict
    """Each discipline's constraint declarations, equalities first."""
    nodes: dict
    """Each discipline's models, constraint bodies, and (economics) the merit."""
    reporting: frozenset
    """Design-dependent models nothing of the problem reads: evaluated after."""
    links: tuple
    """Every `Link`, source then holder in `DISCIPLINES` order, design variables first."""
    constant: tuple
    """The crossing places in `CONSTANT_LINKS`, read without a copy."""

    def links_between(self, source, holder) -> tuple:
        return tuple(k for k in self.links if (k.source, k.holder) == (source, holder))

    def copies_held_by(self, holder) -> tuple:
        return tuple(k for k in self.links if k.holder == holder)


def _discipline_of(node, part: Partition) -> str:
    s = str(node)
    if s.startswith("['c"):
        c = int(s.strip("[]'c"))
        return next(d for d, ids in DISCIPLINE_CONSTRAINTS.items() if c in ids)
    if s == "['merit']":
        return ECONOMICS
    if s.startswith(ECONOMICS_AREAS) and not s.startswith(NOT_ECONOMICS):
        return ECONOMICS
    if node in part.magnet_nodes or s.startswith(ALSO_BUILD):
        return BUILD
    return OPERATION


def disciplines(configuration) -> Disciplines:
    """Build, operation and economics on the machine's graph. Refused if a discipline
    reads anything a later one computes, or if a cycle spans two."""
    problem = configuration.problem
    part = partition(configuration)
    graph = study.machine_model(configuration).graph
    owners = graph.graph.owners
    design = part.design
    reach = {d: set(graph.graph.reach([d])) for d in design}
    depends = {n: {d for d in design if n in reach[d]} for n in graph.nodes}
    constraints = (*problem.equalities, *problem.inequalities)
    ends = {owners[MERIT], *(owners[held(c)] for c in constraints)}
    needed = ends | set(graph.graph.ancestors(list(ends)))
    side = {n: _discipline_of(n, part) for n in graph.nodes if depends[n] and n in needed}
    reporting = frozenset(n for n in graph.nodes if depends[n] and n not in needed)
    designs = {BUILD: part.magnet_design, OPERATION: part.plasma_design, ECONOMICS: ()}
    owned_by = {d: k for k, ds in designs.items() for d in ds}

    rank = {d: i for i, d in enumerate(DISCIPLINES)}
    found: dict = {}
    constant = set()
    for n in sorted(side, key=str):
        for r in graph[n].reads:
            o = owners.get(r)
            source = side.get(o) if o is not None else owned_by.get(r)
            if source is None or source == side[n]:
                continue
            if r.spelling in CONSTANT_LINKS:
                constant.add(r)
                continue
            if rank[source] > rank[side[n]]:
                raise ValueError(f"{side[n]} model {n} reads {r.spelling}, which {source} "
                                 f"computes: the split is not one-way")
            found.setdefault((source, side[n], r), []).append(n)
    for cycle in graph.graph.cycles:
        if len({side.get(n) for n in cycle} - {None}) > 1:
            raise ValueError(f"a cycle spans {sorted({side.get(n) for n in cycle} - {None})}")

    def order(key):
        source, holder, place = key
        return (rank[source], rank[holder], place not in design,
                list(design).index(place) if place in design else 0, place.spelling)

    links = tuple(Link(p, s, h, tuple(found[(s, h, p)])) for s, h, p in sorted(found, key=order))
    by_id = {cid(c): c for c in constraints}
    return Disciplines(
        design=design,
        designs=designs,
        constraints={d: tuple(by_id[i] for i in sorted(ids, key=lambda i: (
            by_id[i] not in problem.equalities, i))) for d, ids in DISCIPLINE_CONSTRAINTS.items()},
        nodes={d: frozenset(n for n, s in side.items() if s == d) for d in DISCIPLINES},
        reporting=reporting,
        links=links,
        constant=tuple(sorted(constant, key=lambda p: p.spelling)),
    )


def constant_links(configuration, cold=None, step=0.01) -> dict:
    """Per place of `CONSTANT_LINKS`, the largest relative change over the MDA re-solved
    with each design variable moved by `step`: zero where the place is constant."""
    cold = study.cold_state(configuration) if cold is None else cold
    state = state_of(configuration)
    model = study.machine_model(configuration)
    mda = study.mda_graph(model, study.RULE)
    base_inputs = study.values(mda, state, model.given, cold)
    base = study.solve(mda, base_inputs)
    places = [p for p in base if getattr(p, "spelling", None) in CONSTANT_LINKS]
    worst = {p.spelling: 0.0 for p in places}
    for d in configuration.problem.design:
        moved = OrderedMap((k, v * (1 + step) if k == d else v) for k, v in base_inputs.items())
        out = study.solve(mda, moved)
        for p in places:
            a, b = np.asarray(base[p]), np.asarray(out[p])
            worst[p.spelling] = max(worst[p.spelling],
                                    float(np.max(np.abs(b - a) / np.maximum(np.abs(a), 1e-300))))
    return worst


# ---------------------------------------------------------------- the base every recipe starts from


@dataclasses.dataclass(frozen=True)
class Base:
    """The file's problem as AAO declares it (`problem.problem_model`: the machine, every
    constraint body, the merit, and `Opt` over all eight variables), read for the
    distributed recipes.

    The Sellar base holds each bound and each constraint as a requirement node of its
    own, which a plan combines into the problem that owns its variables. This one holds
    them all inside `Opt`, and no op splits a condition node; so a recipe deletes `Opt`
    and inserts each of its problems with its share of `Opt`'s relations, taken from
    here unchanged. What SAND does inside a problem is taken from AAO's own recipe
    (`cx.absolute.sand` on this base): the same cuts and closures, and the implicit
    models' solves combined in."""

    model: object
    objective: object
    """`Opt`'s objective."""
    relations: tuple
    """`Opt`'s relations, in its order: the bounds, then the constraints."""
    bounds: OrderedMap
    """Per design variable, its bound relations in `Opt`."""
    constraints: OrderedMap
    """Per PROCESS constraint id, its relation in `Opt`."""
    parameters: tuple
    """`Opt`'s parameters: what its bounds are read from."""
    closures: tuple
    """AAO's `FixedPointCut`s, in order: the cycles on the optimiser's cycle."""
    solves: tuple
    """The implicit models' solves AAO combines into `Opt`."""

    @property
    def graph(self):
        return self.model.graph

    def share(self, design, ids) -> tuple:
        """`Opt`'s relations a problem over `design` subject to constraints `ids` holds,
        in `Opt`'s order."""
        mine = [*(r for x in design for r in self.bounds[x]),
                *(self.constraints[i] for i in ids)]
        return tuple(r for r in self.relations if any(r is m for m in mine))

    def owner(self, discipline_nodes: dict, label) -> str:
        """The discipline a closure (by its `at`) or a solve belongs to: for a closure,
        the one whose models its cuts move onto the copies; for a solve, the one whose
        models read what it determines."""
        g = self.graph
        node = next((c for c in self.closures if c.at == label), None)
        if node is not None:
            readers = {r for cut in node.cuts for r in cut.readers}
        else:
            readers = {r for u in g[label].unknowns for r in g.graph.readers.get(u, ())}
        found = [k for k, ns in discipline_nodes.items() if readers & ns]
        if len(found) != 1:
            raise ValueError(f"{label} is read by {found or 'no discipline'}")
        return found[0]


def base(configuration, rule=study.RULE) -> Base:
    from cottax.core.maths.plan import Combine
    from cottax.relational.rewrites import FixedPointCut

    model = problem_model(configuration)
    graph = model.graph
    top = graph[TOP]
    by_held = {held(c): cid(c) for c in (*configuration.problem.equalities,
                                         *configuration.problem.inequalities)}
    bounds = OrderedMap((x, tuple(r for r in top.relations
                                  if study._is_bound(top, r) and x in (r.lhs, r.rhs)))
                        for x in top.unknowns)
    constraints = OrderedMap((by_held[r.lhs], r) for r in top.relations if r.lhs in by_held)
    counted = sum(map(len, bounds.values())) + len(constraints)
    if counted != len(top.relations):
        raise ValueError(f"`Opt` holds {len(top.relations)} relations, {counted} read")
    ops = [o for stage in cx.absolute.sand(TOP, rule).staged(graph) for o in stage.ops]
    closures = tuple(o for o in ops if isinstance(o, FixedPointCut))
    at = {c.at for c in closures}
    solves = tuple(n for o in ops if isinstance(o, Combine) for n in o.nodes
                   if n != TOP and n not in at)
    return Base(model, top.objectives[0], tuple(top.relations), bounds, constraints,
                tuple(top.parameters), closures, solves)


def condition(objective=None, relations=(), parameters=()):
    """A condition node determining nothing, as a recipe inserts one: minimising
    `objective` (if any), holding `relations` over `parameters`."""
    objectives = () if objective is None else (cx.Objective(objective, cx.Min),)
    return cx.RelationalCondition(relations=tuple(relations), objectives=objectives,
                                  unknowns=(), parameters=tuple(parameters))


def requirement_of(closure):
    """AAO's closure of a `FixedPointCut` (`^copy.x = x` per cut), determining nothing:
    the `Insert` that, after its `Cut`s, closes the loop, to be combined into the
    problem that owns its copies."""
    return condition(relations=closure.definition.relations)


def held_parameters(b: Base, relations) -> tuple:
    """`Opt`'s parameters the `relations` read: what a problem holding them must list."""
    reads = {p for r in relations for p in (r.lhs, r.rhs) if p is not None}
    return tuple(p for p in b.parameters if p in reads)


def insert(label, node):
    """`node` inserted at `label`: one binding."""
    from cottax.core.maths.plan import Insert
    return Insert(OrderedMap(((label, node),)))


# ---------------------------------------------------------------- several problems at the root


class TopsShareCycle(Violation):
    top: str
    others: str
    error = ValueError
    template = ("{top} and {others} lie on one cycle, so neither can run before the other: "
                "one driver answers one problem, so `Combine` them into one (the "
                "monolithic problem) or `Nest` one inside the other (a bilevel one). "
                "Which is a modelling decision; the split as stated is not sequential")


def refuse_shared_cycles(graph, tops) -> None:
    """Refuse `tops`, problems at the root, if two lie on one cycle: neither could run
    before the other. cottax refuses such a graph too, when it is driven ("several
    problems on one cycle with nothing saying which is outer"); this names the two
    problems, before anything is driven."""
    for i, top in enumerate(tops):
        cycle = set(graph.cycle_of(top))
        if on := [str(o) for o in tops[i + 1:] if o in cycle]:
            raise TopsShareCycle(str(top), ", ".join(on)).raised


def node(name: str):
    """A node at the model's root, as `problem.TOP` is."""
    return cx.absolute.node(cx.N[name])


# ---------------------------------------------------------------- driven, as study drives its top


def recorder(trace):
    return study._recorder(trace)


def drive(plan: Plan, tops, env, traces: dict, *, max_iter=study.MAX_ITER, optimiser=VmconDriver,
          scale_conditions=True, also=(), options=None) -> Plan:
    """Each top answered by `optimiser` reporting its steps, verdict and status, its
    iterates recorded into `traces[top]`, its coupling conditions scaled by `env` (an
    MDA at the design), `options` passed to each; each `(problem, contract,
    implementation)` of `also` as it says; every other problem driven as `study.driven`
    drives it."""
    implementations = {}
    for problem, contract, implementation in also:
        plan = plan + SURFACE.assign(problem, contract)
        implementations[problem] = implementation
    for t in tops:
        top = plan.graph[t]
        nlp = programme(top)
        plan = plan + SURFACE.assign(t, PortMinimisation(reports=(Steps, Converged, Status)))
        traces[t] = []
        implementations[t] = optimiser(
            n_equality=len(nlp.equalities),
            n_inequality=len(nlp.inequalities),
            condition_scale=study.condition_scale(top, env) if scale_conditions else (),
            callback=recorder(traces[t]),
            max_iter=max_iter,
            **(options or {}),
        )
    return study.driven(plan + cx.absolute.implement(implementations))


# ---------------------------------------------------------------- valued, as study values its top


def mda_env(configuration, cold=None):
    """`(state, env)`: the configuration's state and one MDA at its design."""
    cold = study.cold_state(configuration) if cold is None else cold
    state = state_of(configuration)
    model = study.machine_model(configuration)
    mda = study.mda_graph(model, study.RULE)
    return state, study.solve(mda, study.values(mda, state, model.given, cold))


def values(graph, state, given, env, design, start=None) -> OrderedMap:
    """A value at every boundary input of `graph`: `given`, else a design variable's
    start from `start` (default the state), every other start and every cut copy from
    `env` at the place it copies (every mint stripped), anything else from the state,
    else `env`. What `study.block_values` does for SAND's top, for several."""
    start = {} if start is None else start
    started = study.starts(graph)
    out = {}
    for var in graph.graph.boundary_inputs:
        if var in given:
            out[var] = given[var]
            continue
        unknown = started.get(var)
        source = var if unknown is None else unknown
        if unknown is not None and unknown in design:
            value = start[unknown] if unknown in start else ground_truth(state, unknown)
        elif unknown is not None or unminted(var) != var:
            place = study._place(source)
            value = start.get(place, env[place] if place in env else env.get(source, 0.0))
        else:
            try:
                value = ground_truth(state, source)
            except (AttributeError, KeyError):
                value = env.get(source, 0.0)
        out[var] = study._typed(value)
    return OrderedMap(out)


def report(graph, out, top, trace) -> dict:
    """`status`, `iterations`, `objf`, `max_eq`, `min_ie` of one top off a run."""
    node_ = graph[top]
    reported = {role: np.asarray(out[slot[0]])
                for role, slot in zip(node_.contract.reports, node_.reports, strict=True)}
    iterations = int(reported[Steps])
    status = ("converged" if reported[Converged] else "no-step" if not iterations
              else "cap" if iterations >= study.MAX_ITER else "stopped")
    objf, max_eq, min_ie = trace[-1][2:] if trace else (None, None, None)
    return {"status": status, "vmcon_status": int(reported[Status]), "iterations": iterations,
            "objf": objf, "max_eq": max_eq, "min_ie": min_ie}


def timed(graph, inputs) -> tuple:
    began = time.perf_counter()
    out = study.solve(graph, inputs)
    jax.block_until_ready(jax.tree.leaves(list(out.values())))
    return out, time.perf_counter() - began


def number(out, place) -> float:
    return float(np.asarray(out[place]))


def aao(configuration, cold=None, start=None, optimiser=VmconDriver) -> dict:
    """The file's own problem under SAND (the base and AAO's recipe), solved by
    `optimiser` (the pin's VMCON by default): the AAO reference; from `start`
    (`{design variable: value}`) where given, else the file's start."""
    state = None if start is None else state_with(configuration, start)
    return study.study(configuration, "SAND", cold, state=state, optimiser=optimiser)


def state_with(configuration, values) -> object:
    """The configuration's state with `values` (`{place: value}`) written in: where
    every solve valued from it starts."""
    state = copy.deepcopy(state_of(configuration))
    for place, value in values.items():
        area, name = place.spelling.strip(".").split(".")
        state.areas[area].values[name] = value
        state.values[(area, name)] = value
    return state


__all__ = [
    "CONFIGURATION", "MAGNET_CONSTRAINTS", "MERIT", "TOP", "Partition", "aao", "cid",
    "drive", "held", "jnp", "load", "mda_env", "node", "number", "partition", "problem_model",
    "reach_table", "report", "short", "timed", "values",
]
