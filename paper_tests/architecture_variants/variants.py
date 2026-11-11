"""The architecture variants of a configuration: per coupling block, whether its MDA is
resolved in an MDA of its own (`F`, as MDF) or by the optimiser (`O`, as IDF and SAND),
and per implicit model on the optimiser's cycle, whether its solve stays local (`L`, as
MDF and IDF) or is absorbed by the optimiser (`O`, as SAND).

MDF, IDF and SAND are cottax `Monolithic` architectures that share every stage --
absorb what the design reaches into the top, resolve the cycles off the top's with an
MDA, cut every cycle of the scope by the rule, close each cut group with a fixed-point
closure, nest what lies inside another's body -- and differ in two things only:
`combining`, where each cycle's problems go, and `boxes`, which implicit models the
cutting sees whole. `Hybrid` is one more final of the same kind, deciding both per
cycle and per implicit model; it emits nothing but `Combine` and `Nest`, so every
variant passes the same refusals and executability checks as the three presets, and
at the corners it emits their op sequences exactly (`structure.py` checks the graphs
are equal before anything is timed).

A variant is labelled by one letter per block, in binding order, then `-` and one
letter per switchable implicit model: `FFF` is MDF, `OOO` IDF (= SAND with no implicit
model), `OO-L` IDF and `OO-O` SAND on a machine with one implicit model.
"""

from __future__ import annotations

import dataclasses
import itertools
from dataclasses import field

import cottax.assemblies.jax as cx
from cottax.core import FunctionalNode, Nest, Plan
from cottax.consumers.architectures.architectures import (
    Boxes,
    Monolithic,
    OnCycle,
    combined,
    one_mda,
)
from cottax.assemblies.jax import RootFinding
from cottax.relational import programme
from cottax.relational.consumers.execution.jax_solvers import Newton

from functional_process.configurations import Optimisation
from functional_process.cottax import study
from functional_process.cottax.architectures.contracts import (
    ROLE_PLACES,
    SURFACE,
    Converged,
    PortFixedPointIteration,
    PortMinimisation,
    PortRootFinding,
    Status,
    Steps,
)
from functional_process.cottax.architectures.drivers import SlsqpDriver
from functional_process.cottax.problem import TOP, problem_model


# ---------------------------------------------------------------- the architecture


class AllBut(Boxes):
    """Every problem of the graph handed seen whole, but `names`: `Disciplines` when
    `names` is empty, `Bodies` when it holds them all."""

    names: frozenset

    def folded(self, graph):
        return tuple(p for p in graph.problems if p not in self.names)

    def __repr__(self) -> str:
        return f"all-but({', '.join(sorted(map(str, self.names)))})"


class Hybrid(Monolithic):
    """Per cycle of the top's scope: its closures folded into one MDA nested in `top`
    where `mda` names the cycle (by its first closure, the name MDF gives that MDA),
    else combined into `top`. Per implicit model: combined into `top` where `absorbed`
    names it, else nested where MDF or IDF nests it -- in its cycle's MDA, or in `top`.

    The ops come out in the presets' own order: one `Combine` into the top over the top,
    then per cycle its closures and its absorbed models (SAND's and IDF's order), then
    per cycle MDF's `one_mda` or the nests IDF writes."""

    mda: frozenset = field(default=frozenset(), kw_only=True)
    absorbed: frozenset = field(default=frozenset(), kw_only=True)

    @property
    def boxes(self) -> Boxes:
        return AllBut(self.absorbed)

    @property
    def combines_declared(self) -> bool:
        return bool(self.absorbed)

    def combining(self, opened, closures):
        to_top, placed = [self.top], []
        for cycle in self.cycles_under(opened, closures):
            local = tuple(p for p in cycle.declared if p not in self.absorbed)
            taken = tuple(p for p in cycle.declared if p in self.absorbed)
            if cycle.closures and cycle.closures[0] in self.mda:
                to_top.extend(taken)
                placed.extend(one_mda(self, opened, OnCycle(cycle.closures, local), self.top))
            else:
                to_top.extend((*cycle.closures, *taken))
                placed.extend(Nest(p, self.top) for p in local)
        return (*combined(self, opened, self.top, tuple(to_top)), *placed)

    def __repr__(self) -> str:
        return (f"{self._repr()[:-1]}, mda={sorted(map(str, self.mda))}, "
                f"absorbed={sorted(map(str, self.absorbed))})")


# ---------------------------------------------------------------- the variants


@dataclasses.dataclass(frozen=True)
class Switches:
    """What can be switched on one configuration, read off its MDF graph: the MDAs MDF
    nests in the top (one per coupling block on the optimiser's cycle) and the implicit
    models it nests in the top or in one of those MDAs."""

    blocks: tuple
    implicit: tuple


@dataclasses.dataclass(frozen=True)
class Variant:
    label: str
    mda: frozenset
    absorbed: frozenset
    preset: str
    """`MDF`, `IDF`, `SAND`, `IDF/SAND` (the two coincide), or `` for a hybrid."""


def _plan_under(configuration, op):
    return Plan(problem_model(configuration).graph) + op


def switches(configuration, rule=study.RULE) -> Switches:
    graph = _plan_under(configuration, cx.absolute.mdf(TOP, rule)).graph
    within = dict(graph.within)
    blocks = tuple(p for p in graph.problems if within.get(p) == TOP and _is_closure(p))
    implicit = tuple(p for p in graph.problems
                     if not _is_closure(p) and p != TOP and within.get(p) in (TOP, *blocks))
    return Switches(blocks, implicit)


def _is_closure(label) -> bool:
    return "closure" in str(label)


def variants(configuration, rule=study.RULE) -> tuple[Variant, ...]:
    """Every variant of `configuration`, blocks varying slowest. A root find has an MDF
    and nothing else among the presets (`study.arms`): its all-`O` variant, the block
    combined into the top's Newton, is a hybrid."""
    sw = switches(configuration, rule)
    optimises = isinstance(configuration.problem, Optimisation)
    out = []
    for mda_bits in itertools.product("FO", repeat=len(sw.blocks)):
        for abs_bits in itertools.product("LO", repeat=len(sw.implicit)):
            label = "".join(mda_bits) + ("-" + "".join(abs_bits) if sw.implicit else "")
            mda = frozenset(b for b, c in zip(sw.blocks, mda_bits) if c == "F")
            absorbed = frozenset(m for m, c in zip(sw.implicit, abs_bits) if c == "O")
            all_f, all_o = set(mda_bits) <= {"F"}, set(mda_bits) <= {"O"}
            none_abs, all_abs = not absorbed, len(absorbed) == len(sw.implicit)
            preset = ("MDF" if all_f and none_abs else
                      "" if not optimises else
                      "IDF/SAND" if all_o and not sw.implicit else
                      "IDF" if all_o and none_abs else
                      "SAND" if all_o and all_abs else "")
            out.append(Variant(label, mda, absorbed, preset))
    return tuple(out)


def by_label(configuration, label, rule=study.RULE) -> Variant:
    for v in variants(configuration, rule):
        if v.label == label:
            return v
    raise KeyError(f"{configuration.name} has no variant {label!r}")


def hybrid(variant: Variant, rule=study.RULE) -> Hybrid:
    return Hybrid(rule, cx.absolute.closing, TOP, mda=variant.mda, absorbed=variant.absorbed)


# ---------------------------------------------------------------- driven and valued, as `study`


def driven_reporting_steps(plan, fixed_point=study.picard):
    """`study.driven`, every problem it drives asked for its `Steps` as well: the same
    drivers (cottax's optimistix `Picard` and `Newton`, at the port's tolerances), whose
    `Steps` is optimistix's `num_steps`. Only the counting copy of a variant is driven
    so (`InnerCounter` in `solve.py`); the timed graph is `study.driven`'s."""
    plan = plan + cx.Drive(
        (fixed_point(), Newton()),
        (PortFixedPointIteration(reports=(Steps,)), PortRootFinding(reports=(Steps,))),
        ROLE_PLACES,
    )
    return study._supplied(plan)


@dataclasses.dataclass(frozen=True)
class Inner:
    """One problem nested under the top: what it is, how many models its own cycle
    holds (what one of its steps re-runs), and where its `Steps` report lands."""

    label: object
    kind: str
    """`fixed-point` (a block's MDA) or `root-find` (an implicit model's Newton)."""
    size: int
    steps: object


def inner_problems(graph) -> tuple[Inner, ...]:
    """Every problem nested, at any depth, under the top of a graph driven by
    `driven_reporting_steps`, in graph order."""
    within = dict(graph.within)

    def under_top(p):
        while p in within:
            p = within[p]
            if p == TOP:
                return True
        return False

    out = []
    for p in graph.problems:
        if p == TOP or not under_top(p):
            continue
        node = graph[p]
        slots = dict(zip(node.contract.reports, node.reports, strict=True))
        kind = "fixed-point" if isinstance(node.contract, PortFixedPointIteration) else "root-find"
        size = sum(isinstance(graph[n], FunctionalNode) for n in graph.cycle_of(p))
        out.append(Inner(p, kind, size, slots[Steps][0]))
    return tuple(out)


def top_models(graph) -> int:
    """How many models one evaluation of the top's problem runs once: its cycle's."""
    return sum(isinstance(graph[n], FunctionalNode) for n in graph.cycle_of(TOP))


def variant_plan(configuration, variant, env, trace, rule=study.RULE, optimiser=SlsqpDriver,
                 reports=True, *, model=None, drive=study.driven) -> Plan:
    """`study.block_plan` with `Hybrid` for `study._architecture(arm)`; nothing else
    changed: the top by `optimiser` at `study.MAX_ITER`, its coupling conditions
    scaled by `env`, every other problem driven by `drive` (`study.driven`; the
    counting copy passes `driven_reporting_steps`)."""
    model = problem_model(configuration) if model is None else model
    plan = Plan(model.graph) + hybrid(variant, rule)
    top = plan.graph[TOP]
    if isinstance(configuration.problem, Optimisation):
        nlp = programme(top)
        contract = PortMinimisation(reports=(Steps, Converged, Status) if reports else ())
        implementation = optimiser(
            n_equality=len(nlp.equalities),
            n_inequality=len(nlp.inequalities),
            condition_scale=study.condition_scale(top, env),
            callback=study._recorder(trace),
            max_iter=study.MAX_ITER,
        )
    else:
        contract, implementation = RootFinding(steps=reports), Newton()
    plan = plan + SURFACE.assign(TOP, contract)
    return drive(plan + cx.absolute.implement({TOP: implementation}))


def nested_given(variant: Variant) -> bool:
    """`study.assemble`'s `nested_given=arm in ("MDF", "IDF")`: on for every variant
    but SAND's corner, where nothing is nested under the top. Where IDF and SAND
    coincide the paper prints the SAND row, so that corner is SAND's (off); the two
    rows' iterations agree on every such machine in `../architectures/out`."""
    return variant.preset not in ("SAND", "IDF/SAND")


def assemble_variant(configuration, variant, cold=None, *, rule=study.RULE,
                     optimiser=SlsqpDriver, state=None) -> study.Assembled:
    """`study.assemble` for a variant: the MDA at the file's design solved once, the
    variant's graph built and valued from it."""
    cold = study.cold_state(configuration) if cold is None else cold
    state = study.state_of(configuration) if state is None else state
    model = study.machine_model(configuration)
    mda = study.mda_graph(model, rule)
    env = study.solve(mda, study.values(mda, state, model.given, cold))
    trace: list = []
    block_model = problem_model(configuration)
    graph = variant_plan(configuration, variant, env, trace, rule=rule, optimiser=optimiser,
                         model=block_model).graph
    inputs = study.block_values(graph, state, block_model.given, env,
                                set(configuration.problem.design), nested_given=nested_given(variant))
    return study.Assembled(variant.label, graph, inputs, env, trace)
