"""The two-stage split of a graph for optimisation under uncertainty.

A two-stage formulation makes the build decisions once, before the machine exists, and
lets the operating point settle per realisation of the physics; a belief is drawn. On
the graph that is a question `reach` answers: every node downstream of a belief leaf
or an operating leaf is **second stage** -- it takes a different value in every sample
-- and everything else is **first stage**, a function of the build alone, evaluated
once per design and hoisted out of the batch over belief samples.

Three things here, in that order:

- **`split`**: the stage of every node, from the graph and a `Leaves` -- which boundary
  inputs are beliefs and which are operating variables, picked out of
  `graph.graph.boundary_inputs` by `configurations.kinds.KINDS` (`leaves`).
- **`violations`**: the check `paper_tests/stage_check.py` ran. A model output that
  `output_kinds.md` classes as a build decision closed by a rule (a duct diameter, a
  pump count, a winding-pack width) must be the same in every sample; one whose owner
  is second stage is a non-anticipativity violation, and the leaves that reach it say
  what would have to be held at its nominal to make it first stage. `report` is the
  table `stage_check.main` printed.
- **The hoist**: `first_stage_graph` / `recourse_graph` are the two subgraphs, and
  `hoisted` runs the first once and hands back a callable over the second whose
  boundary carries the first stage's outputs. `first_env` and `recourse(...)` together
  reproduce `evaluate.run_schedule` on the whole graph at the same inputs.

**A block never straddles the boundary.** `reach` is closed under descendants, so the
second stage is descendant-closed and the first stage ancestor-closed; every node of a
cycle is a descendant of every other, so a strongly connected component -- a driven
block, its problem node, whatever is nested in it -- lands whole in one stage: second
if any node of it is reached, first otherwise. That is the rule, and it is not applied
here, it falls out of the definition. `Graph.subgraph` keeps a nesting pair only when
both ends are kept and re-checks that both are on one cycle, so the subgraphs carry
the whole graph's nesting and drivers unchanged.

The counts this file measures are in `tests/architectures/test_stages.py`; the
handoff's table (`~/jaxgraph`, `plans/handoff_2026-09-17.md`) was measured on the
graph *with* `.vacuum.duct_diameter_root_find` (156 nodes) and before `kinds.py` added
the two section-4 lifetimes to `CLAIMED_BUILD_OUTPUTS`, which is why the test states
both graphs.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import TYPE_CHECKING

from cottax.answerable import AnswerableGraph
from cottax.evaluation.schedule import Schedule

from functional_process.configurations.kinds import (
    BELIEFS,
    CLAIMED_BUILD_OUTPUTS,
    KINDS,
    PAIRINGS,
    Kind,
)
from functional_process.cottax.architectures.evaluate import inputs_only, run_schedule

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from cottax.graph import Graph
    from cottax.spec import NodePath, VarPath

# ------------------------------------------------------------------ the leaves


class Stage(enum.Enum):
    """Which stage a node is evaluated in."""

    FIRST = "first"
    """A function of the build alone: evaluated once per design."""
    SECOND = "second"
    """Reached from a belief leaf: takes a different value in every sample."""
    RECOURSE = "recourse"
    """Reached from an operating leaf but from no belief: varies with the operating
    point the sample settles at, so it is second stage too -- kept apart so the
    count says how much of the second stage the recourse alone drags in."""


@dataclasses.dataclass(frozen=True)
class Leaves:
    """The boundary inputs of a graph that vary per sample, and the ones that do not.

    `belief` and `operating` are what `split` asks `reach` about; `build` is the
    complement `KINDS` classes as a build decision, recorded so a caller can see what
    the first stage is a function of.
    """

    belief: tuple[VarPath, ...]
    operating: tuple[VarPath, ...]
    build: tuple[VarPath, ...]

    @property
    def varying(self) -> tuple[VarPath, ...]:
        """Every leaf that varies per sample: the beliefs, then the operating ones."""
        return (*self.belief, *self.operating)


def sampled_paths(held: Iterable[str] = ()) -> tuple[str, ...]:
    """The spellings `BELIEFS` draws, bar `dummy` (which maps to nothing) and bar
    `held` -- `BUILD_LEAVES` for the handoff's `build` table, `ECONOMIC` when only the
    physics is uncertain.
    """
    held = set(held)
    return tuple(b.path for b in BELIEFS if b.path != "dummy" and b.path not in held)


RECOURSE_PATHS: tuple[str, ...] = tuple(PAIRINGS["two"].values())
"""The operating leaves the MDA closes per sample -- the density and the thermal alpha
fraction -- which is what `stage_check` took as the operating set of its `sampled`
and `build` rows: every other operating leaf is held at its nominal there."""


def leaves(
    graph: Graph,
    kinds: Mapping[str, Kind] = KINDS,
    *,
    sampled: Iterable[str] | None = None,
    varying: Iterable[str] | None = None,
) -> Leaves:
    """The `Leaves` of `graph`: its boundary inputs sorted by `kinds` (spelling ->
    `Kind`), in the graph's own binding order.

    `sampled` restricts the belief set to those spellings -- the rows a study actually
    draws (`sampled_paths()`), rather than everything `kinds` classes as a belief;
    `varying` likewise restricts the operating set (`RECOURSE_PATHS`). Either given as
    an iterable of spellings; a spelling that is not a boundary input of `graph` is
    ignored, since the study can draw it without the graph reading it.

    Raises
    ------
    KeyError
        If a boundary input of `graph` is not in `kinds` at all: the sort is a
        hand-made table and a leaf it does not know has no stage.
    """
    inputs = graph.graph.boundary_inputs
    if unknown := [v.spelling for v in inputs if v.spelling not in kinds]:
        raise KeyError(
            f"{len(unknown)} boundary input(s) have no kind in the table, starting "
            f"with {unknown[:5]}; a leaf without a kind has no stage"
        )
    keep_belief = None if sampled is None else set(sampled)
    keep_operating = None if varying is None else set(varying)
    return Leaves(
        belief=tuple(
            v
            for v in inputs
            if (keep_belief is None and kinds[v.spelling] is Kind.BELIEF)
            or (keep_belief is not None and v.spelling in keep_belief)
        ),
        operating=tuple(
            v
            for v in inputs
            if (keep_operating is None and kinds[v.spelling] is Kind.OPERATING)
            or (keep_operating is not None and v.spelling in keep_operating)
        ),
        build=tuple(v for v in inputs if kinds[v.spelling] is Kind.BUILD),
    )


# ------------------------------------------------------------------ the split


@dataclasses.dataclass(frozen=True)
class Counts:
    """How many nodes, and how many owned places, one stage holds."""

    nodes: int
    owned: int


@dataclasses.dataclass(frozen=True)
class Stages:
    """The stage of every node of a graph, from `split`."""

    leaves: Leaves
    first: tuple[NodePath, ...]
    """Reached from no varying leaf, in binding order."""
    second: tuple[NodePath, ...]
    """Reached from a belief leaf, in binding order."""
    recourse: tuple[NodePath, ...]
    """Reached from an operating leaf and from no belief, in binding order."""
    stage: Mapping[NodePath, Stage]
    reach_of_leaf: Mapping[VarPath, frozenset[NodePath]]
    """Every varying leaf -> the nodes it reaches."""
    counts: Mapping[Stage, Counts]

    @property
    def varying(self) -> tuple[NodePath, ...]:
        """Every node that takes a different value per sample, in binding order:
        the second stage and the recourse together.
        """
        return tuple(n for n, s in self.stage.items() if s is not Stage.FIRST)

    @property
    def n_nodes(self) -> int:
        """How many nodes the graph has, every stage together."""
        return len(self.stage)

    @property
    def first_stage_fraction(self) -> float:
        """The share of nodes that can be hoisted out of the batch."""
        return len(self.first) / self.n_nodes if self.n_nodes else 0.0

    def responsible(self, node: NodePath) -> tuple[str, ...]:
        """The spellings of every varying leaf that reaches `node`, sorted."""
        return tuple(
            sorted(
                v.spelling
                for v, reached in self.reach_of_leaf.items()
                if node in reached
            )
        )


def split(graph: Graph, leaves: Leaves) -> Stages:
    """Every node of `graph` sorted into its stage: `second` is `reach` from the
    belief leaves, `recourse` is `reach` from the operating leaves less that, `first`
    is the rest.
    """
    dependencies = graph.graph
    second = frozenset(dependencies.reach(leaves.belief))
    recourse = frozenset(dependencies.reach(leaves.operating)) - second
    stage = {
        n: (
            Stage.SECOND
            if n in second
            else Stage.RECOURSE
            if n in recourse
            else Stage.FIRST
        )
        for n in graph.nodes
    }
    counts = {
        which: Counts(
            nodes=sum(1 for s in stage.values() if s is which),
            owned=sum(len(graph[n].owns) for n, s in stage.items() if s is which),
        )
        for which in Stage
    }
    return Stages(
        leaves=leaves,
        first=tuple(n for n, s in stage.items() if s is Stage.FIRST),
        second=tuple(n for n, s in stage.items() if s is Stage.SECOND),
        recourse=tuple(n for n, s in stage.items() if s is Stage.RECOURSE),
        stage=stage,
        reach_of_leaf={
            leaf: frozenset(dependencies.reach([leaf])) for leaf in leaves.varying
        },
        counts=counts,
    )


# ------------------------------------------------------------- the violations


@dataclasses.dataclass(frozen=True)
class Violation:
    """A claimed build output whose owner is second stage."""

    place: str
    section: str
    """Which table of `output_kinds.md` claims it (`CLAIMED_BUILD_OUTPUTS`' value)."""
    owner: NodePath
    stage: Stage
    responsible: tuple[str, ...]
    """The varying leaves that reach the owner, sorted by spelling."""

    @property
    def n_responsible(self) -> int:
        """How many varying leaves reach the owner."""
        return len(self.responsible)


@dataclasses.dataclass(frozen=True)
class Confirmed:
    """A claimed build output whose owner is first stage."""

    place: str
    section: str
    owner: NodePath


def owned_by_spelling(graph: Graph) -> dict[str, tuple[VarPath, NodePath]]:
    """`spelling -> (variable, owner)` over every owned place of `graph`."""
    return {var.spelling: (var, owner) for var, owner in graph.graph.owners.items()}


def violations(
    graph: Graph,
    stages: Stages,
    claimed: Mapping[str, str] = CLAIMED_BUILD_OUTPUTS,
) -> tuple[tuple[Violation, ...], tuple[Confirmed, ...]]:
    """Every place in `claimed` (spelling -> section) that `graph` owns, sorted into
    the ones whose owner is second stage (or recourse) -- the violations, most
    responsible leaves first -- and the ones confirmed first stage. A claimed spelling
    `graph` does not own is skipped: absent in this configuration, or a shorthand
    cell.
    """
    owned = owned_by_spelling(graph)
    hits, clean = [], []
    for spelling, section in sorted(claimed.items()):
        if spelling not in owned:
            continue
        _var, owner = owned[spelling]
        stage = stages.stage[owner]
        if stage is Stage.FIRST:
            clean.append(Confirmed(spelling, section, owner))
        else:
            hits.append(
                Violation(spelling, section, owner, stage, stages.responsible(owner))
            )
    hits.sort(key=lambda v: (-v.n_responsible, v.place))
    return tuple(hits), tuple(clean)


def _truncate(items: Iterable[str], n: int = 8) -> str:
    items = list(items)
    if len(items) <= n:
        return ", ".join(items)
    return ", ".join(items[:n]) + f" +{len(items) - n} more"


def report(stages: Stages, hits: Iterable[Violation], label: str = "") -> str:
    """The summary line and the violation table `stage_check.main` printed, as text."""
    hits = tuple(hits)
    counts = stages.counts
    first, second, recourse = (
        counts[Stage.FIRST],
        counts[Stage.SECOND],
        counts[Stage.RECOURSE],
    )
    summary = (
        f"{label or 'stages':>8}: {len(stages.leaves.belief):3d} belief + "
        f"{len(stages.leaves.operating)} operating leaves | nodes 1st/2nd/recourse "
        f"{first.nodes}/{second.nodes}/{recourse.nodes} of {stages.n_nodes} "
        f"| owned 1st/2nd/recourse {first.owned}/{second.owned}/{recourse.owned} "
        f"| 1st-stage frac {stages.first_stage_fraction:.2f} "
        f"| violations {len(hits)}"
    )
    lines = [
        f"  {v.place:45s} {v.section:4s} <- {v.owner.spelling:40s} "
        f"({v.n_responsible} leaves: {_truncate(v.responsible)})"
        for v in hits
    ]
    return "\n".join([summary, *lines])


# ------------------------------------------------------------------- the hoist


def first_stage_graph(graph: Graph, stages: Stages) -> Graph:
    """The subgraph of first-stage nodes: a function of the build leaves alone, with
    every driver and nesting it had in `graph`.
    """
    return graph.subgraph(stages.first)


def recourse_graph(graph: Graph, stages: Stages) -> Graph:
    """The subgraph of every node that varies per sample -- second stage and recourse
    -- whose boundary inputs now include the first-stage outputs it reads.
    """
    return graph.subgraph(stages.varying)


def hoisted_inputs(graph: Graph, stages: Stages) -> tuple[VarPath, ...]:
    """The first-stage outputs the recourse graph reads: what the hoist hands across
    the boundary, in binding order.
    """
    first = set(first_stage_graph(graph, stages).graph.owned_variables)
    return tuple(
        v for v in recourse_graph(graph, stages).graph.boundary_inputs if v in first
    )


@dataclasses.dataclass(frozen=True)
class Hoisted:
    """A graph split for the batch: the first stage run once, the recourse runnable
    any number of times over it. From `hoisted`.
    """

    first: Schedule
    recourse: Schedule
    first_env: Mapping[VarPath, object]
    """Everything the first stage computed, and its inputs."""
    inputs: Mapping[VarPath, object]
    """The whole graph's inputs, as handed to `hoisted`."""
    run: Callable = run_schedule

    def __call__(self, overrides: Mapping[VarPath, object] | None = None) -> dict:
        """Run the recourse over the first stage's answer, with `overrides` -- one
        sample's values at the varying leaves -- over the nominal inputs. Returns the
        recourse schedule's env: its inputs (the first stage's outputs among them) and
        everything it computed.

        Raises
        ------
        ValueError
            If `overrides` writes a place the first stage computed or a build leaf it
            read: a value that differs per sample there is a violation of the split,
            not an input to the recourse.
        """
        env = {**self.inputs, **self.first_env}
        if overrides:
            first_read = set(self.first.inputs) | (
                set(self.first_env) - set(self.inputs)
            )
            if stale := [v.spelling for v in overrides if v in first_read]:
                raise ValueError(
                    f"override(s) at {stale}, which the first stage read or computed: "
                    f"a value that changes per sample there is a violation of the "
                    f"split, not a recourse input"
                )
            env.update(overrides)
        return self.run(self.recourse, inputs_only(self.recourse, env))


def hoisted(
    graph: Graph,
    stages: Stages,
    env: Mapping[VarPath, object],
    run: Callable = run_schedule,
) -> Hoisted:
    """Evaluate the first stage of `graph` once from `env` (the whole graph's inputs),
    and return the `Hoisted` whose call runs the recourse over that answer.

    `run(schedule, inputs) -> env` is `evaluate.run_schedule` by default.
    """
    first = Schedule(AnswerableGraph(first_stage_graph(graph, stages)))
    recourse = Schedule(AnswerableGraph(recourse_graph(graph, stages)))
    first_env = run(first, inputs_only(first, env))
    return Hoisted(
        first=first, recourse=recourse, first_env=first_env, inputs=dict(env), run=run
    )
