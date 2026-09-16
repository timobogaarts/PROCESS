"""Grouping this port's graph by the *prefix* of its node names, and drawing the two
orderings -- by provenance (where a node was written) and by structure (the run order at
every nesting level, with every solve boxed inside the solve it is nested in, ringed by
the kind of problem it answers and named with its driver: the solve strategy).
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
import dataclasses
import json
import warnings

import networkx as nx

from jax.tree_util import DictKey, GetAttrKey

from cottax.abstract import Eq, is_problem, undriven
from cottax.blocking import Blocking
from cottax.graph import Graph
from cottax.spec import NodePath, VarPath
from cottax.problem import ConditionNode, Driven, shape_of
from cottax.names import is_minted, unminted
from cottax.visualization.sequencing import sequenced
from cottax.visualization.xdsm import (
    PROBLEM_TYPE_TEXT,
    Formatter,
    NoFormat,
    _xesc,
    problems_at,
)
from cottax.visualization.xdsm_html import HtmlDoc

type Group = tuple[str, ...]

UNGROUPED: Group = ()

CONDITIONS: Group = ("conditions",)
"""The group the optimiser's condition nodes are drawn in -- `.Constraint<n>` and
`.Objective`, top-level names `sand.constraint_nodes` / `objective_nodes` mint with no
subsystem of their own. They are what the optimiser reads; a reader looks for them."""

OPTIMISER: Group = ("optimiser",)
"""The group the outer problem is drawn in: `.Opt`, `.RootFind`, `^problem.sand`."""

FIXED_GROUP_COLOUR = {CONDITIONS: "#c1704f", OPTIMISER: "#77609a"}
"""Colours these two groups take regardless of `PALETTE`'s order: terracotta for the
conditions (the paper's variable colour, off every subsystem hue) and the optimiser's
violet, the same the kind ring uses."""


def _synthetic(path: NodePath) -> "Group | None":
    """`CONDITIONS` / `OPTIMISER` for the names those groups collect, else `None`."""
    keys = _tree_keys(path)
    if is_minted(path):
        return OPTIMISER if unminted(path).spelling in (".sand",) else None
    if len(keys) != 1:
        return None
    leaf = keys[0]
    if leaf.startswith("Constraint") or leaf.startswith("Objective"):
        return CONDITIONS
    if leaf in ("Opt", "RootFind"):
        return OPTIMISER
    return None
"""The group of a node whose name carries no prefix at all."""

UNGROUPED_LABEL = "(ungrouped)"


# ================================================================== reading the prefix
def _tree_keys(path: NodePath) -> tuple[str, ...]:
    """The leading run of namespace keys of `path`, once any minted root is dropped."""
    out: list[str] = []
    for key in unminted(path).segments:
        if isinstance(key, DictKey) and isinstance(key.key, str):
            out.append(key.key)
        elif isinstance(key, GetAttrKey):
            out.append(key.name)
        else:
            break
    return tuple(out)


def _cut_owner(
    path: NodePath, owners: "Mapping[VarPath, NodePath]"
) -> "NodePath | None":
    """The node that owns the variable a problem-over-a-variable was minted over, or
    `None`.
    """
    keys = unminted(path).segments
    for end in range(len(keys), 0, -1):
        var = VarPath(keys[:end])
        if var in owners:
            return owners[var]
    return None


def group_of(
    path: NodePath,
    *,
    depth: int | None = None,
    among: "Iterable[NodePath] | None" = None,
    owners: "Mapping[VarPath, NodePath] | None" = None,
) -> Group:
    """The group `path` declares it belongs to: its leading keys, without its own name.
    """
    if depth is not None and depth < 1:
        raise ValueError(f"depth must be at least 1, not {depth}")
    synthetic = _synthetic(path)
    if synthetic is not None:
        return synthetic
    if among is not None and is_minted(path) and unminted(path) not in among:
        if owners is not None:
            owner = _cut_owner(path, owners)
            if owner is not None:
                return group_of(owner, depth=depth, among=among, owners=owners)
        return UNGROUPED
    keys = _tree_keys(path)
    own = max(len(keys) - 1, 0)
    return keys[: own if depth is None else min(depth, own)]


def group_label(group: Group) -> str:
    """How a group is written: dotted, like the names it is a prefix of."""
    return ".".join(group) if group else UNGROUPED_LABEL


def top_of(group: Group) -> Group:
    """The subsystem a group is in: its first key."""
    return group[:1]


def containing(groups: "Iterable[Group]") -> Group:
    """The smallest namespace holding every one of `groups`: their longest common
    prefix.
    """
    named = [g for g in groups if g != UNGROUPED]
    if not named:
        return UNGROUPED
    common = named[0]
    for group in named[1:]:
        keep = 0
        while keep < min(len(common), len(group)) and common[keep] == group[keep]:
            keep += 1
        common = common[:keep]
    return common


def hierarchical(groups: "Iterable[Group]") -> tuple[Group, ...]:
    """`groups` re-ordered so a namespace is followed by the namespaces inside it."""
    groups = tuple(dict.fromkeys(groups))
    rank = {g: i for i, g in enumerate(groups)}
    return tuple(
        sorted(
            groups,
            key=lambda g: tuple(
                rank.get(g[: i + 1], len(groups)) for i in range(len(g))
            ),
        )
    )


# ================================================================== the two orderings
def group_sequence(
    names: Iterable[NodePath],
    *,
    depth: int | None = None,
    owners: "Mapping[VarPath, NodePath] | None" = None,
) -> tuple[Group, ...]:
    """Every group present, in **first-appearance order** over `names`."""
    names = tuple(names)
    among = frozenset(names)
    seen: dict[Group, None] = {}
    for name in names:
        seen.setdefault(group_of(name, depth=depth, among=among, owners=owners), None)
    return tuple(seen)


def dependency_group_sequence(
    graph: Graph, *, depth: int | None = None
) -> tuple[Group, ...]:
    """Every group present, in **dependency order**: contract each group, then sort
    that.
    """
    owners = graph.owners
    among = frozenset(graph.nodes)
    at = {
        name: group_of(name, depth=depth, among=among, owners=owners)
        for name in graph.nodes
    }
    declared = group_sequence(graph.nodes, depth=depth, owners=owners)
    rank = {g: i for i, g in enumerate(declared)}

    contracted = nx.DiGraph()
    contracted.add_nodes_from(declared)
    for name in graph.nodes:
        for var in graph[name].reads:
            source = owners.get(var)
            if source is None:
                continue
            if at[source] != at[name]:
                contracted.add_edge(at[source], at[name])

    condensed = nx.condensation(contracted)
    order = nx.lexicographical_topological_sort(
        condensed, key=lambda c: min(rank[g] for g in condensed.nodes[c]["members"])
    )
    return tuple(
        g
        for c in order
        for g in sorted(condensed.nodes[c]["members"], key=rank.__getitem__)
    )


def provenance_order(
    names: Iterable[NodePath],
    *,
    depth: int | None = None,
    groups: Sequence[Group] | None = None,
    owners: "Mapping[VarPath, NodePath] | None" = None,
) -> tuple[NodePath, ...]:
    """`names` regrouped so every member of a group is adjacent, groups in `groups`'
    order.
    """
    names = tuple(names)
    among = frozenset(names)
    order = (
        tuple(groups)
        if groups is not None
        else group_sequence(names, depth=depth, owners=owners)
    )
    index = {g: i for i, g in enumerate(order)}
    missing = {
        group_of(n, depth=depth, among=among, owners=owners) for n in names
    } - set(index)
    if missing:
        raise KeyError(
            f"group(s) {sorted(group_label(g) for g in missing)} are in the graph but not "
            f"in the `groups` order given"
        )
    return tuple(
        sorted(
            names,
            key=lambda n: index[group_of(n, depth=depth, among=among, owners=owners)],
        )
    )


def structure_order(blocking: Blocking) -> tuple[NodePath, ...]:
    """The order the graph actually runs in, **at every level**: `blocking`'s blocks in
    their run order, each block's body in the run order of *its* level, and so on down
    through `Blocking.inner`.

    Two things are done to the blocking's own member order, and both are borrowed from
    cottax's XDSM rather than invented here:

    - the body of every block -- the block with its problems taken out -- is put in the
      SCC order of that body wherever the stored order would draw a read from a later
      member (`visualization.sequencing.sequenced`, which is recursive over `inner`, so a
      nested level's interior is ordered by *its* blocking and not by the parent's
      binding order);
    - the problem a level answers is drawn **first** in its block, ahead of what it
      drives (`xdsm._problem_first`'s rule): it is what the level is *for*, and pinning
      it to the head is what leaves the interior contiguous, so a nested box can be a
      rectangle. A block that answers no single problem -- none declared, or several
      un-nested ones -- puts every problem it holds ahead of every body, in stored order.

    Which linear extension of the body is drawn is a presentation choice, and this makes
    no promise beyond *one the body could be run in*: every edge among body members below
    the diagonal, and only the reads that pass through a problem above it.
    """
    return tuple(_run_order(sequenced(blocking)))


def answered_at(blocking: Blocking) -> tuple[NodePath | None, ...]:
    """`xdsm.problems_at`, without its warning: the problem each block answers at its
    own level, `None` where none is or where several are declared and none nested.

    The warning is right for a drawing that then shows nothing at that block, and wrong
    here: a block declaring three un-nested problems is *drawn* -- every one of the
    three marked by its kind, the block ringed as coupled with no solve named -- and the
    page is the place a reader sees that the nesting is missing. Repeating cottax's
    advice on the console, once per call and with the whole block spelt out, is noise
    over a picture that already says it.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*declares \d+ problems.*")
        return problems_at(blocking)


def _run_order(blocking: Blocking) -> Iterator[NodePath]:
    """`structure_order`'s recursion: one level, its nested levels in place."""
    for block, held, lead in zip(blocking.blocks, blocking.inner, answered_at(blocking)):
        if held is not None:
            # The level's problem is the one member its interior does not hold; the
            # interior is a blocking of its own and states the rest of the order.
            inside = frozenset(held.graph.nodes)
            yield from (name for name in block if name not in inside)
            yield from _run_order(held)
            continue
        if lead is not None:
            head = [lead]
        else:
            # Every problem ahead of every body, the outer kind first: an optimiser is
            # what a reader looks for at the top left of a block.
            head = sorted(
                (name for name in block if is_problem(blocking.graph[name])),
                key=lambda name: 0 if problem_kind(blocking.graph[name]) in ("optimise", COMBINED) else 1,
            )
        yield from head
        yield from (name for name in block if name not in head)


# ================================================================== the measurement
@dataclasses.dataclass(frozen=True)
class BlockGrouping:
    """One block of a `Blocking`, and which groups it is made of."""

    members: tuple[NodePath, ...]
    groups: tuple[Group, ...]
    """The distinct groups its members declare, `UNGROUPED` included, in member order."""

    @property
    def real(self) -> int:
        """Members that are not minted names -- the block's coupling, in §11's sense."""
        return sum(1 for m in self.members if not is_minted(m))

    @property
    def named_groups(self) -> tuple[Group, ...]:
        return tuple(g for g in self.groups if g != UNGROUPED)

    @property
    def container(self) -> Group:
        """The smallest namespace holding every named group its members declare."""
        return containing(self.groups)

    @property
    def spans(self) -> bool:
        """Whether its members declare more than one named group at all."""
        return len(self.named_groups) > 1

    @property
    def crosses(self) -> bool:
        """Whether it couples namespaces that **nothing smaller than the machine
        contains**.
        """
        return self.spans and self.container == UNGROUPED

    @property
    def nests(self) -> bool:
        """Spans namespaces, but all inside one -- `physics` with `physics.profiles`."""
        return self.spans and self.container != UNGROUPED


@dataclasses.dataclass(frozen=True)
class GroupingReport:
    """What provenance and structure agree and disagree about, on one graph."""

    depth: int | None
    groups: tuple[Group, ...]
    sizes: Mapping[Group, int]
    runs: Mapping[Group, int]
    """How many maximal contiguous stretches each group occupies in the **run** order.
    """

    blocks: tuple[BlockGrouping, ...]
    cross_group_edges: int
    """Node-to-node edges whose endpoints are in different named groups."""

    cross_subsystem_edges: int
    """Of those, the ones whose endpoints are in different *subsystems* (`top_of`)."""

    @property
    def coupled(self) -> tuple[BlockGrouping, ...]:
        """The blocks that genuinely couple: more than one non-minted node."""
        return tuple(b for b in self.blocks if b.real > 1)

    @property
    def crossing(self) -> tuple[BlockGrouping, ...]:
        """Of those, the ones no single namespace contains -- see
        `BlockGrouping.crosses`.
        """
        return tuple(b for b in self.coupled if b.crosses)

    @property
    def nesting(self) -> tuple[BlockGrouping, ...]:
        """Of those, the ones spanning namespaces that one namespace still contains."""
        return tuple(b for b in self.coupled if b.nests)

    @property
    def levels(self) -> int:
        """How deep the grouping actually goes: the longest group present."""
        return max((len(g) for g in self.groups), default=0)

    def summary(self) -> str:
        grain = "the tree" if self.depth is None else f"depth {self.depth}"
        return (
            f"{len(self.groups)} group(s) over {self.levels} level(s), by {grain}; "
            f"{len(self.blocks)} block(s), {len(self.coupled)} with more than one real "
            f"node, {len(self.crossing)} of those crossing a subsystem boundary and "
            f"{len(self.nesting)} spanning namespaces inside one; "
            f"{self.cross_group_edges} cross-group edge(s), "
            f"{self.cross_subsystem_edges} between subsystems; "
            f"{sum(1 for g in self.groups if self.runs.get(g, 0) == 1)}/"
            f"{len(self.groups)} group(s) contiguous in the run order"
        )


def grouping_report(blocking: Blocking, *, depth: int | None = None) -> GroupingReport:
    """Measure provenance against structure on `blocking`: § 11's table, for any graph.
    """
    graph = blocking.graph
    order = structure_order(blocking)
    owners = graph.owners
    groups = group_sequence(graph.nodes, depth=depth, owners=owners)
    among = frozenset(graph.nodes)
    at = {
        name: group_of(name, depth=depth, among=among, owners=owners)
        for name in graph.nodes
    }

    sizes = {g: 0 for g in groups}
    for name in graph.nodes:
        sizes[at[name]] += 1

    runs = {g: 0 for g in groups}
    previous: Group | None = None
    for name in order:
        if at[name] != previous:
            runs[at[name]] += 1
        previous = at[name]

    blocks = tuple(
        BlockGrouping(tuple(block), tuple(dict.fromkeys(at[name] for name in block)))
        for block in blocking.blocks
    )

    crossing = {
        (owners[var], name)
        for name in graph.nodes
        for var in graph[name].reads
        if var in owners
        and at[owners[var]] != at[name]
        and at[owners[var]] != UNGROUPED
        and at[name] != UNGROUPED
    }
    between = {
        (source, target)
        for source, target in crossing
        if top_of(at[source]) != top_of(at[target])
    }
    return GroupingReport(
        depth, groups, sizes, runs, blocks, len(crossing), len(between)
    )


# ================================================================== the strategy
COMBINED = "combined"
"""The kind of a problem several were `Combine`d into -- see `problem_kind`."""

KIND_ORDER = (
    "optimise", COMBINED, "root-find", "fixed-point", "feasibility", "declared", "stated"
)
"""Every kind a problem row can be marked with, in the order a legend lists them:
cottax's own shape slugs (`xdsm.PROBLEM_TYPE_ORDER`, plus `shape_of`'s two fallthroughs)
with `combined` beside the optimise it is a special case of."""

KIND_TEXT = {
    **PROBLEM_TYPE_TEXT,
    COMBINED: "combined (several problems as one)",
    "declared": "declared, matching no shape",
}
"""What a legend entry says of each kind."""

UNDRIVEN = "undriven"
"""What `driver_name` says of a problem no algorithm has been `Assign`ed to."""


def problem_kind(node) -> str | None:
    """The kind a problem row is marked with: `None` for a node with a body, else
    `cottax.problem.shape_of`'s slug -- except `combined`.

    **`combined` is read off structure, and the reading is a heuristic.** `Combine`
    leaves no mark on the node it builds: the join is `Condition.__add__`, which
    concatenates the two statements' relations, and the result is a `Condition` like any
    other. What *does* survive is the concatenation: the `Optimise(...)` constructor
    writes every equality into **one** `Eq` relation against zero (and every inequality
    into one `Le`), so an objective sitting over **two or more** `Eq` relations against
    zero can only have been written by hand or by a join -- and in this port it is the
    join, SAND's `^problem.sand`, every inner fixed point residualised and folded under
    the file's optimiser. A fixed point over several relations is *not* called combined:
    a join of pairings is still a fixed point, which is what `is_fixed_point` says of it.
    """
    if not isinstance(node, ConditionNode):
        return None
    kind = shape_of(node)
    problem = undriven(node)
    residual_relations = sum(
        1 for r in problem.relations if r.op is Eq and r.against_zero
    )
    if kind == "optimise" and residual_relations > 1:
        return COMBINED
    return kind


def driver_name(node) -> str | None:
    """Which algorithm answers a problem: the driver's class name, `UNDRIVEN` for a
    problem none has been `Assign`ed to yet, `None` for a node that is not a problem."""
    if not isinstance(node, ConditionNode):
        return None
    return type(node.driver).__name__ if isinstance(node, Driven) else UNDRIVEN


@dataclasses.dataclass(frozen=True)
class Solve:
    """One block at one level of the solve strategy: what is solved together, how deep,
    inside which other solve, and by what."""

    members: tuple[NodePath, ...]
    level: int
    """Nesting depth: 0 for a block of the top-level blocking, 1 inside one of those."""

    parent: int | None
    """Index into `solve_levels`' tuple of the solve this one is nested in."""

    problem: NodePath | None
    """The problem answered at this level -- `xdsm.problems_at`'s answer: `None` for a
    block that only runs, and for one that declares several problems and nests none."""

    kind: str | None
    driver: str | None

    @property
    def real(self) -> int:
        return sum(1 for m in self.members if not is_minted(m))

    @property
    def driven(self) -> bool:
        return self.problem is not None

    @property
    def coupled(self) -> bool:
        """More than one real node: `BlockGrouping.real > 1`, the coupling § 11 counts.
        """
        return self.real > 1


def solve_levels(blocking: Blocking) -> tuple[Solve, ...]:
    """Every block of `blocking` that is driven or coupled, **at every nesting level**,
    outermost first and in run order within a level.

    A block that is neither -- one node, or nothing to solve -- is left out, at every
    level; what is kept is every solve (a driven block, whatever its size: a nesting is a
    claim about the solve, and the point of drawing it is to see the claim) and every
    coupled block nothing drives (`GroupingReport.coupled`'s blocks, the cycles a cut has
    not reached). Read off `Blocking.inner` and `xdsm.problems_at`, so it never refuses:
    a blocking no schedule could be built for still has levels, and a picture of one is
    the picture worth having.
    """
    out: list[Solve] = []

    def walk(level: Blocking, depth: int, parent: int | None) -> None:
        for block, held, lead in zip(level.blocks, level.inner, answered_at(level)):
            node = level.graph[lead] if lead is not None else None
            solve = Solve(
                tuple(block), depth, parent, lead, problem_kind(node), driver_name(node)
            )
            above = parent
            if solve.coupled or solve.driven:
                out.append(solve)
                above = len(out) - 1
            if held is not None:
                walk(held, depth + 1, above)

    walk(blocking, 0, None)
    return tuple(out)


# ================================================================== the drawing
PALETTE = (
    "#4c78a8",  # steel blue
    "#f58518",  # orange
    "#3fa7b8",  # cyan
    "#e45756",  # red
    "#cc79a7",  # reddish pink
    "#5c9ecf",  # sky
    "#f2a35e",  # apricot
    "#ff9da6",  # salmon
    "#8ab6d6",  # powder blue
    "#bab0ac",  # warm grey
    "#d98c8c",  # dusty rose
    "#6b8fb8",  # dusty blue
)
"""One colour per group, mid-luminance so every one of them reads on white and on black.
**No green, ocher or purple**: those three hues are the problem kinds' (`KIND_COLOUR`,
the paper's rings) and nothing else on the page may wear them. The previous palette's
green, yellow, brown, mint and teal all read as one of the three and are gone.
"""

UNGROUPED_COLOUR = "#8c8c8c"

KIND_COLOUR = {
    "optimise": "#77609a",
    COMBINED: "#77609a",
    "root-find": "#96771f",
    "fixed-point": "#4f7a37",
    "feasibility": "#5e5e5e",
    "declared": "#9d9d9d",
    "stated": "#9d9d9d",
}

KIND_FILL = {
    "optimise": "#dbcfec",
    COMBINED: "#dbcfec",
    "root-find": "#ead3a0",
    "fixed-point": "#c4dcb2",
    "feasibility": "#dcdcdc",
    "declared": "#e6e6e6",
    "stated": "#e6e6e6",
}
"""The paper's problem-box **fills** (`latex_xdsm.py`'s `.prob-*`), beside `KIND_COLOUR`'s
strokes: a problem row's diagonal and marks take the fill, its label and its box's ring
the stroke -- the same pairing an XDSM box has, so the two drawings read alike."""
"""One colour per problem kind -- a box's ring and a problem row's label. **The paper's
own** (`graph_paper/figures/latex_xdsm.py`, the terracotta palette's `ring-*` rules:
violet optimise, ocher root-find, olive fixed-point, grey feasibility / unanswered), so a
DSM here and an XDSM there say one kind in one colour; a combined problem is an
optimiser and takes its violet. `PALETTE` above was nudged away from these three hues
for the same reason. A coupled block nothing drives keeps the page's accent."""

TIER_OVERLAY = (None, "hatch-stripe", "hatch-dot")
"""What a group beyond the palette's length is drawn with, on top of its recycled
colour.
"""


def group_style(index: int) -> tuple[str, str | None]:
    """The colour and overlay texture the `index`-th subsystem is drawn with."""
    return PALETTE[index % len(PALETTE)], TIER_OVERLAY[
        (index // len(PALETTE)) % len(TIER_OVERLAY)
    ]


TINT_LADDER = (0.0, 0.34, -0.26, 0.52, -0.44, 0.18, -0.13, 0.62, -0.55, 0.44, -0.35)
"""How far each namespace *inside* a subsystem is shaded away from the subsystem's own
hue.
"""


def shade(colour: str, amount: float) -> str:
    """`colour` blended `amount` of the way to white (positive) or to black (negative)."""
    r, g, b = (int(colour[i : i + 2], 16) for i in (1, 3, 5))
    if amount >= 0:
        mixed = (round(c + (255 - c) * amount) for c in (r, g, b))
    else:
        mixed = (round(c * (1 + amount)) for c in (r, g, b))
    return "#" + "".join(f"{c:02x}" for c in mixed)


@dataclasses.dataclass(frozen=True)
class Shade:
    """How one group is painted: its own tint, its texture, and the hue it is a tint of."""

    colour: str
    overlay: str | None
    base: str
    """The subsystem's undiluted hue -- what a *label* is written in."""


def group_palette(groups: Iterable[Group]) -> dict[Group, Shade]:
    """One `Shade` per group **and per namespace above one**, keyed by group."""
    named = tuple(dict.fromkeys(g for g in groups if g != UNGROUPED))
    closure = tuple(dict.fromkeys(g[: i + 1] for g in named for i in range(len(g))))
    subsystems = tuple(dict.fromkeys(top_of(g) for g in closure))
    # The two synthetic groups take fixed colours and no slot of the palette, so a
    # subsystem's hue does not shift when an optimiser is inserted.
    real = [top for top in subsystems if top not in FIXED_GROUP_COLOUR]
    base = {top: group_style(i) for i, top in enumerate(real)}
    base.update({top: (FIXED_GROUP_COLOUR[top], None) for top in subsystems if top in FIXED_GROUP_COLOUR})
    out = {UNGROUPED: Shade(UNGROUPED_COLOUR, None, UNGROUPED_COLOUR)}
    for top in subsystems:
        colour, overlay = base[top]
        family = [g for g in hierarchical(closure) if top_of(g) == top]
        for rank, group in enumerate(family):
            tint = TINT_LADDER[rank % len(TINT_LADDER)]
            out[group] = Shade(shade(colour, tint), overlay, colour)
    return out


TIP_VARS = 12
"""How many variable names one hover may list before it stops and says how many are
left.
"""


def _edges(graph: Graph) -> dict[tuple[NodePath, NodePath], list[VarPath]]:
    """Node -> node, with the variables that flow along each. The boundary is dropped."""
    owners = graph.owners
    out: dict[tuple[NodePath, NodePath], list[VarPath]] = {}
    for name in graph.nodes:
        for var in dict.fromkeys(graph[name].reads):
            source = owners.get(var)
            if source is None or source == name:
                continue
            out.setdefault((source, name), []).append(var)
    return out


MODES = ("provenance", "structure")
"""The two pages one struct can be drawn as. *Provenance* is the ordering that scatters
a block, so it boxes only the coupled blocks of the top level and rings their members;
*structure* is the run order, where every solve is contiguous, so it boxes every solve
at every level -- a 2-row fixed point included -- and shades depth."""


def _matrix_struct(
    blocking: Blocking,
    order: Sequence[NodePath],
    *,
    depth: int | None,
    formatter: Formatter,
    mode: str = "provenance",
) -> dict:
    """Everything the page draws, as plain data: rows, cells, group bands, block boxes,
    and -- new with the solve strategy -- each row's nesting depth and kind, the boxes of
    every nested level, the depth bands and the kind legend.

    `mode` (`MODES`) decides one thing in the data: whether a driven block of the top
    level with nothing coupled in it (a fixed point beside the one node it is minted
    over) gets a box. Every other key is the same in both modes, so a caller reading the
    struct for its numbers -- the paper's print generator -- sees exactly what it saw
    before this argument existed.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    graph = blocking.graph
    order = tuple(order)
    if set(order) != set(graph.nodes):
        raise ValueError(
            f"the order given holds {len(set(order))} of the graph's {len(graph.nodes)} "
            f"node(s) -- a DSM is a permutation of the whole graph, not a selection"
        )

    owners = graph.owners
    among = frozenset(graph.nodes)
    at = {
        name: group_of(name, depth=depth, among=among, owners=owners)
        for name in graph.nodes
    }
    groups = group_sequence(graph.nodes, depth=depth, owners=owners)
    # Hue is keyed on the *subsystem*, not on the group: see `top_of`. Depth is a tint of
    # that hue (`group_palette`), so `stellarator.coils` is a lighter stellarator blue in
    # a second ribbon lane rather than a colour of its own that says nothing about where
    # it lives.
    subsystems = tuple(dict.fromkeys(top_of(g) for g in groups))
    palette = group_palette(groups)
    hue = {name: palette[at[name]] for name in graph.nodes}
    # A problem row is its KIND's colour outright -- label, diagonal and the marks of what
    # it owns -- so the three problem types jump out of a matrix of subsystem hues,
    # which `PALETTE` keeps clear of violet, ocher and olive for exactly this reason.
    for name in graph.nodes:
        kind = problem_kind(graph[name])
        if kind is not None:
            hue[name] = Shade(KIND_FILL.get(kind, UNGROUPED_COLOUR), None,
                              KIND_COLOUR.get(kind, UNGROUPED_COLOUR))
    index = {name: i for i, name in enumerate(order)}

    # A row carries the node's **declared ports**, not the union of the edges this matrix
    # happens to draw for it. The two differ, and the difference is the whole reason to
    # prefer the declaration: a node reads boundary variables nothing in the graph owns
    # and owns variables nothing downstream reads, and neither appears as a mark anywhere
    # in the matrix. Reading a node's inputs off its incoming cells would therefore
    # describe the picture rather than the node -- and a diagonal hover is the one place
    # a reader is asking about the node itself. `reads`/`owns` are `NodeDefinition`'s own
    # ports (`cottax/spec.py`), the same two lists `xdsm_struct` ships as
    # `reads`/`writes`, so the two diagrams answer this question from one source.
    def _ports(vars_: Sequence[VarPath]) -> tuple[list[str], int]:
        seen = list(dict.fromkeys(vars_))
        return [formatter.var(v) for v in seen[:TIP_VARS]], len(seen)

    # The solve strategy: every driven or coupled block at every nesting level. A row's
    # depth is how many of those hold it -- 0 outside every solve, 1 in a top-level one,
    # 2 in a level nested inside that -- and is the same whichever mode draws it.
    solves = solve_levels(blocking)
    row_depth = {name: 0 for name in order}
    for solve in solves:
        for member in solve.members:
            row_depth[member] = max(row_depth[member], solve.level + 1)

    rows = []
    for name in order:
        reads, n_reads = _ports(graph[name].reads)
        writes, n_writes = _ports(graph[name].owns)
        rows.append({
            "name": formatter.node((name, graph[name])),
            "group": group_label(at[name]),
            "colour": hue[name].colour,
            "base": hue[name].base,
            "overlay": hue[name].overlay,
            "problem": isinstance(graph[name], ConditionNode),
            "minted": is_minted(name),
            "reads": reads,
            "nr": n_reads,
            "writes": writes,
            "nw": n_writes,
            "depth": row_depth[name],
            "kind": problem_kind(graph[name]),
            "driver": driver_name(graph[name]),
        })

    cells = [
        {
            "r": index[source],  # outputs in rows: this row produces ...
            "c": index[target],  # ... what this column reads
            "n": len(shared),
            "v": [formatter.var(v) for v in shared[:TIP_VARS]],
            "colour": hue[source].colour,
        }
        for (source, target), shared in _edges(graph).items()
    ]

    # One ribbon lane per level of the tree, outermost nearest the labels: a run of
    # `stellarator` in lane 0 with `coils` and `fwbs` drawn inside it in lane 1. This is
    # where `depth` used to be spent -- truncating the grain to make one flat ribbon
    # drawable -- and nesting the lanes is what buys the grain back without a knob. A
    # node shallower than the lane contributes nothing to it, which is how a ragged tree
    # draws: `costs` has lane 0 and simply no lane 1.
    levels = max((len(at[name]) for name in order), default=0)
    bands: list[dict] = []
    for level in range(levels):
        run: Group | None = None
        start = 0
        for i, name in enumerate((*order, None)):
            group = at[name] if name is not None else None
            key = (
                group[: level + 1] if group is not None and len(group) > level else None
            )
            if key != run:
                if run is not None:
                    bands.append({
                        "level": level,
                        "from": start,
                        "to": i - 1,
                        "label": group_label(run) if level == 0 else run[-1],
                        "full": group_label(run),
                        "colour": palette[run].colour,
                        "base": palette[run].base,
                        "overlay": palette[run].overlay,
                    })
                run, start = key, i

    report = grouping_report(blocking, depth=depth)
    # One box per solve. The top level keeps its old rule -- a box is a *coupled* block
    # -- unless the page is the structure one, which boxes every solve; a nested level is
    # boxed whenever it is a solve, since nesting is what the structure page exists to
    # show. So the level-0 boxes of a provenance struct are `report.coupled`'s blocks,
    # in the same order, with the same keys plus the strategy's.
    kept = [
        s
        for s in solves
        if s.coupled or (s.driven and (s.level > 0 or mode == "structure"))
    ]
    renumber = {id(s): i for i, s in enumerate(kept)}
    boxes = []
    for b in kept:
        grouping = BlockGrouping(b.members, tuple(dict.fromkeys(at[m] for m in b.members)))
        parent = None if b.parent is None else renumber.get(id(solves[b.parent]))
        boxes.append({
            "from": min(index[m] for m in b.members),
            "to": max(index[m] for m in b.members),
            "size": len(b.members),
            "real": b.real,
            "crosses": grouping.crosses,
            "nests": grouping.nests,
            "container": group_label(grouping.container),
            "contiguous": (
                max(index[m] for m in b.members) - min(index[m] for m in b.members) + 1
            )
            == len(b.members),
            "groups": [group_label(g) for g in grouping.groups],
            "members": [formatter.node((m, graph[m])) for m in b.members],
            "at": sorted(index[m] for m in b.members),
            "level": b.level,
            "parent": parent,
            "problem": None if b.problem is None else index[b.problem],
            "kind": b.kind,
            "driver": b.driver,
        })

    # The nesting depth as a ribbon: runs of rows at one depth, the way `bands` runs
    # rows of one group. What the structure page shows on its axes instead of the
    # namespace, since the run order has already made every solve contiguous.
    depth_bands: list[dict] = []
    run_depth: int | None = None
    start = 0
    for i, name in enumerate((*order, None)):
        d = row_depth[name] if name is not None else None
        if d != run_depth:
            if run_depth is not None:
                depth_bands.append({"from": start, "to": i - 1, "depth": run_depth})
            run_depth, start = d, i

    present = {}
    for row in rows:
        if row["kind"] is not None:
            present[row["kind"]] = present.get(row["kind"], 0) + 1
    kinds = [
        {
            "kind": k,
            "colour": KIND_COLOUR.get(k, UNGROUPED_COLOUR),
            "text": KIND_TEXT.get(k, k),
            "count": present[k],
        }
        for k in KIND_ORDER
        if k in present
    ] + [
        {"kind": k, "colour": UNGROUPED_COLOUR, "text": k, "count": present[k]}
        for k in present
        if k not in KIND_ORDER
    ]

    legend = [
        {
            "label": group_label(g) if len(g) <= 1 else g[-1],
            "full": group_label(g),
            "level": max(len(g) - 1, 0),
            "colour": palette[g].colour,
            "overlay": palette[g].overlay,
            "size": report.sizes[g],
            "runs": report.runs[g],
        }
        for g in hierarchical(groups)
    ]
    return {
        "rows": rows,
        "cells": cells,
        "bands": bands,
        "boxes": boxes,
        "legend": legend,
        "summary": report.summary(),
        "backward": sum(c["n"] for c in cells if c["r"] > c["c"]),
        "reads": sum(c["n"] for c in cells),
        "coupled": len(report.coupled),
        "crossing": len(report.crossing),
        "nesting": len(report.nesting),
        "levels": levels,
        "crossEdges": report.cross_group_edges,
        "subsystemEdges": report.cross_subsystem_edges,
        "tiers": len(TIER_OVERLAY),
        "recycled": len(subsystems) > len(PALETTE) * len(TIER_OVERLAY),
        "mode": mode,
        "depthBands": depth_bands,
        "depths": max((r["depth"] for r in rows), default=0),
        "kinds": kinds,
        "solves": len(kept),
    }


_PAGE = r"""<meta charset="utf-8"><title>__TITLE__</title>
<style>
:root {
  --bg:#ffffff; --fg:#1a1a1a; --dim:#6b6b6b; --rule:#d9d9d9; --panel:#f6f6f6;
  --diag:#00000018; --accent:#d62728; --crosshair:#00000010;
}
@media (prefers-color-scheme: dark) {
  :root { --bg:#14161a; --fg:#e8e8e8; --dim:#9aa0a6; --rule:#33383f; --panel:#1c1f24;
          --diag:#ffffff18; --accent:#ff6b6b; --crosshair:#ffffff12; }
}
* { box-sizing:border-box; }
html { height:100%; }
body { margin:0; height:100%; display:flex; flex-direction:column;
  background:var(--bg); color:var(--fg);
  font:13px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
header { flex:0 0 auto; padding:12px 18px 10px; border-bottom:1px solid var(--rule); }
h1 { margin:0; font-size:16px; font-weight:600; letter-spacing:.01em; }
#wrap { flex:1 1 auto; min-height:0; display:flex; align-items:stretch; }
#side { width:280px; min-width:280px; overflow:auto; padding:12px 14px;
  border-right:1px solid var(--rule); background:var(--panel); }
#side h2 { font-size:11px; text-transform:uppercase; letter-spacing:.08em;
  color:var(--dim); margin:14px 0 6px; font-weight:600; }
#side h2:first-child { margin-top:0; }
.lg { display:grid; grid-template-columns:14px 1fr auto; gap:5px 8px; align-items:center; }
.sw { width:14px; height:14px; border-radius:3px; }
.lg .nm { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:11px;
  overflow:hidden; text-overflow:ellipsis; }
.lg .ct { color:var(--dim); font-size:11px; font-variant-numeric:tabular-nums; }
.scatter { color:var(--accent); font-weight:600; }
#stage { flex:1; position:relative; overflow:hidden; cursor:grab; }
#stage.drag { cursor:grabbing; }
#tip { position:fixed; pointer-events:none; z-index:9; background:var(--bg);
  border:1px solid var(--rule); border-radius:6px; padding:6px 8px; font-size:11px;
  max-width:44ch; box-shadow:0 4px 14px #0003; display:none;
  font-family:ui-monospace,Menlo,monospace; white-space:pre-wrap; }
/* The tooltip's three voices, spelled the way `cottax`'s XDSM tip spells its own
   (`xdsm_html.py`, the `'h'` line class): a section header is bold and underlined, its
   items are the indented lines below it. `.dim` is the supporting context a hover leads
   *away* from -- which nodes, which row -- and `.fb` is the one thing on a DSM worth an
   accent, a mark below the diagonal. */
#tip .h { font-weight:700; text-decoration:underline; }
#tip .dim { color:var(--dim); }
#tip .fb { color:var(--accent); font-weight:700; }
#bar { position:absolute; right:10px; top:10px; z-index:5; display:flex; gap:6px; }
#bar button { font:11px inherit; padding:4px 9px; border-radius:5px; cursor:pointer;
  border:1px solid var(--rule); background:var(--bg); color:var(--fg); }
text { fill:var(--fg); }
.lbl { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:8.5px; }
.lbl.mint { fill:var(--dim); }
.lbl.prob { font-weight:700; }
.grid { stroke:var(--rule); stroke-width:.4; }
.sep { stroke:var(--fg); stroke-width:.7; opacity:.35; }
.fb { stroke:none; }  /* below the diagonal is what says feedback; full weight, no ring */
.box { fill:none; stroke:#9d9d9d; stroke-width:1.4; }
.box.ok { stroke:#9d9d9d; opacity:.7; }
/* A solve's box on the structure page: its ring is the kind of problem the level
   answers (`.solve`, stroke set per box), and its *area* is one more coat of the page's
   ink over whatever it sits in (`.area`) -- so a level nested two deep is darker than the
   level around it, and depth reads as shading without a second palette. */
.area { fill:var(--fg); fill-opacity:.07; stroke:none; }
.solve { fill:none; }
.slabel { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:8px;
  font-weight:600; }
.gname { font-size:10px; font-weight:600; }
.dname { font-size:9px; fill:var(--dim); }
.tree { display:grid; grid-template-columns:14px 1fr auto; gap:4px 8px; align-items:center; }
.tree .sw { border:2.5px solid; border-radius:3px; background:none; }
.tree .nm { font-family:ui-monospace,Menlo,monospace; font-size:11px; white-space:nowrap;
  overflow:hidden; text-overflow:ellipsis; }
.tree .ct { color:var(--dim); font-size:11px; font-variant-numeric:tabular-nums; }
#cross { pointer-events:none; }
#cross rect { fill:var(--fg); opacity:.07; }
</style>
<header><h1>__TITLE__</h1></header>
<div id="wrap">
  <div id="side"></div>
  <div id="stage"><div id="bar"><button id="fit">Fit</button><button id="reset">1:1</button></div>
    <svg id="svg"></svg></div>
</div>
<div id="tip"></div>
<script>
const D = __DATA__;
const CELL = 14, PAD = 6, BAND = 11, GAP = 4;
const svg = document.getElementById('svg'), stage = document.getElementById('stage');
const tip = document.getElementById('tip');
const NS = 'http://www.w3.org/2000/svg';
const el = (tag, a = {}) => { const e = document.createElementNS(NS, tag);
  for (const k in a) e.setAttribute(k, a[k]); return e; };

const n = D.rows.length;
const charW = 5.1;
const lblW = Math.min(340, 20 + charW * Math.max(...D.rows.map(r => r.name.length)));
/* Which page this is (`MODES` in grouping.py). The structure page bands its axes by
   nesting depth, boxes every solve at every level and shades depth; the provenance page
   is the picture it always was, namespace bands and the top level's coupled blocks. */
const STRUCT = D.mode === 'structure';
const KIND_COLOUR = Object.fromEntries(D.kinds.map(k => [k.kind, k.colour]));
/* Depth as ink: every level of nesting is one more coat of `AREA` over the last, so the
   opacity a depth reads at is what `d` coats stack to -- the same number the boxes'
   overlapping areas produce on the matrix, so the lane, the legend and the picture agree. */
const AREA = 0.07;
/* The depth *lane* and its legend use a steeper ramp than the areas: an 11px lane at 7%
   grey is invisible, and what the lane has to do is be read against its neighbour. */
const depthAlpha = d => 0.03 + 0.14 * d;
/* Which boxes this page draws. Every box the struct carries is a solve or a coupled
   block at some level; the provenance page draws the top level only (a nested level
   scattered over a provenance ordering is a bounding box over most of the matrix, and
   its rings would ring the same rows its parent's already do), the structure page draws
   them all, outermost first so a child's area lands on top of its parent's. */
const BOXES = STRUCT ? D.boxes : D.boxes.filter(b => b.level === 0);
const boxLabel = b => (b.kind || 'coupled') + ' \u00b7 ' + (b.driver || 'undriven');
const boxDepth = b => { let d = 0;
  for (let p = b.parent; p !== null && p !== undefined; p = BOXES[p].parent) d++; return d; };
const BANDS = STRUCT
  ? D.depthBands.map(b => ({level: 0, from: b.from, to: b.to, label: 'depth ' + b.depth,
      full: 'nesting depth ' + b.depth, colour: 'var(--fg)', alpha: depthAlpha(b.depth),
      base: 'var(--dim)', overlay: null, depth: b.depth}))
  : D.bands;
/* One ribbon lane per level, so the axis is as wide as the tree is deep. Labels get a
   gutter column per level for the same reason: an inner stretch sits inside its outer
   one, so their labels share a y and would collide in a single column. */
const LEVELS = Math.max(1, ...BANDS.map(b => b.level + 1));
const BANDW = BAND * LEVELS;
const lvlW = [];
for (const b of BANDS) lvlW[b.level] = Math.max(lvlW[b.level] || 0, b.label.length);
const lvlX = []; let gut = 6;
for (let L = 0; L < LEVELS; L++) { lvlX[L] = gut; gut += 8 + 6.2 * (lvlW[L] || 0); }
/* The structure page names every solve in a second gutter column, level-indented, on
   the row its box starts at: inside the matrix a label of any size covers marks, and the
   marks are the picture. */
const solveX = gut;
if (STRUCT && BOXES.length)
  gut += 12 + 4.9 * Math.max(...BOXES.map(b => boxLabel(b).length + 7 + 1.6 * boxDepth(b)));
const gutter = 10 + gut;
const X0 = lblW + BANDW + GAP, Y0 = lblW + BANDW + GAP;
const W = X0 + n * CELL + gutter + PAD, H = Y0 + n * CELL + PAD;
const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;');

const defs = el('defs');
defs.innerHTML =
  '<pattern id="hatch-stripe" width="4" height="4" patternUnits="userSpaceOnUse"' +
  ' patternTransform="rotate(45)"><rect width="4" height="4" fill="none"/>' +
  '<line x1="0" y1="0" x2="0" y2="4" stroke="#fff" stroke-opacity=".55" stroke-width="1.6"/></pattern>' +
  '<pattern id="hatch-dot" width="4" height="4" patternUnits="userSpaceOnUse">' +
  '<circle cx="1.6" cy="1.6" r="1" fill="#fff" fill-opacity=".6"/></pattern>';
svg.appendChild(defs);
const root = el('g'); svg.appendChild(root);
svg.setAttribute('width', '100%'); svg.setAttribute('height', '100%');

/* ---- group ribbons, one per axis, plus the separators between groups ---- */
const band = el('g', {class: 'band'});
for (const b of BANDS) {
  const len = (b.to - b.from + 1) * CELL;
  const off = lblW + b.level * BAND;
  /* Depth is drawn as tint, not as hue -- `group_palette` picked it, so a lane's colour
     is the group's own and not this level's opacity: every namespace of one subsystem
     is a shade of that subsystem's colour, and none of them is a second palette. (On the
     structure page the lane *is* depth, and its opacity is `depthAlpha`.) */
  for (const axis of [0, 1]) {
    const x = axis ? X0 + b.from * CELL : off, y = axis ? off : Y0 + b.from * CELL;
    const w = axis ? len : BAND, h = axis ? BAND : len;
    band.appendChild(el('rect', {x, y, width: w, height: h, fill: b.colour, rx: 2,
      'fill-opacity': b.alpha === undefined ? 1 : b.alpha}));
    if (b.overlay)
      band.appendChild(el('rect', {x, y, width: w, height: h, fill: `url(#${b.overlay})`,
        rx: 2}));
  }
  /* Separators mark subsystems only. A rule across the whole matrix for every level of
     every namespace would be a grid, not a separator -- and on the structure page the
     boxes are the separators, so the depth lane draws none. */
  for (const p of (b.level || STRUCT ? [] : [b.from, b.to + 1])) {
    band.appendChild(el('line', {class: 'sep', x1: X0 + p * CELL, y1: lblW,
      x2: X0 + p * CELL, y2: Y0 + n * CELL}));
    band.appendChild(el('line', {class: 'sep', x1: lblW, y1: Y0 + p * CELL,
      x2: X0 + n * CELL, y2: Y0 + p * CELL}));
  }
  /* the group's name, in the right-hand gutter, once per stretch: the ribbon is 11px
     wide and nothing legible fits inside it, and a picture whose colours are the whole
     point must not need its legend to be readable at all.

     Only for a stretch of two rows or more. A one-row stretch's label would overlap its
     neighbour's and neither would be readable -- and in the run order, where a scattered
     group is *made of* one-row stretches, that is most of them. The colour, the legend
     and the row's own dotted name all still say which group it is; what would be lost is
     only a third copy of it, and what would be gained is a gutter of mush. */
  if (b.to > b.from) {
    /* `b.base`, not `b.colour`: the pale end of a tint family is chosen to be a
       distinguishable *fill*, which is not the same as being readable as 10px text. */
    const name = el('text', {class: STRUCT ? 'dname' : 'gname',
      x: X0 + n * CELL + lvlX[b.level],
      y: Y0 + (b.from + (b.to - b.from + 1) / 2) * CELL + 3.5, fill: b.base});
    name.textContent = b.label;
    band.appendChild(name);
  }
}
root.appendChild(band);

/* ---- row and column labels ---- */
const labels = el('g');
/* A problem row is the one kind of row the strategy is *about*, so its label and its
   label is written in its kind's colour: the group stays on its diagonal, its marks and
   its band, and what the row answers is what a reader of the label is looking for. */
const kindOf = r => r.kind && KIND_COLOUR[r.kind];
D.rows.forEach((r, i) => {
  const y = Y0 + i * CELL + CELL / 2 + 3;
  const cls = 'lbl' + (r.minted ? ' mint' : '') + (r.kind ? ' prob' : '');
  const paint = kindOf(r) ? {fill: kindOf(r)} : {};
  const t = el('text', {class: cls, x: lblW - 4, y, 'text-anchor': 'end', ...paint});
  t.textContent = r.name; labels.appendChild(t);
  const x = X0 + i * CELL + CELL / 2 + 3;
  const u = el('text', {class: cls, x, y: lblW - 4, 'text-anchor': 'start',
    transform: `rotate(-90 ${x} ${lblW - 4})`, ...paint});
  u.textContent = r.name; labels.appendChild(u);
});
root.appendChild(labels);

/* ---- the matrix ---- */
const grid = el('g');
for (let i = 0; i <= n; i++) {
  grid.appendChild(el('line', {class: 'grid', x1: X0, y1: Y0 + i * CELL,
    x2: X0 + n * CELL, y2: Y0 + i * CELL}));
  grid.appendChild(el('line', {class: 'grid', x1: X0 + i * CELL, y1: Y0,
    x2: X0 + i * CELL, y2: Y0 + n * CELL}));
}
root.appendChild(grid);

/* ---- the solves' areas: one coat of ink per level, under the diagonal and the marks ---- */
const ringOf = b => b.kind ? (KIND_COLOUR[b.kind] || 'var(--dim)') : null;
const areas = el('g');
if (STRUCT) for (const b of BOXES) {
  const x = X0 + b.from * CELL, w = (b.to - b.from + 1) * CELL;
  areas.appendChild(el('rect', {x, y: Y0 + b.from * CELL, width: w, height: w,
    class: 'area', rx: 3}));
}
root.appendChild(areas);

/* The diagonal keeps the row's GROUP hue on both pages, exactly as the provenance page
   paints it -- the kind is on the label and the box ring, and a reader scanning the
   diagonal is reading provenance. A problem row's cell is outlined rather than filled. */
const diag = el('g');
D.rows.forEach((r, i) => {
  const x = X0 + i * CELL, y = Y0 + i * CELL;
  diag.appendChild(el('rect', {x, y, width: CELL, height: CELL,
    fill: r.colour, 'fill-opacity': r.problem ? .9 : .95,
    stroke: r.problem ? r.base : 'none', 'stroke-width': 1.4}));
  if (r.overlay)
    diag.appendChild(el('rect', {x, y, width: CELL, height: CELL, fill: `url(#${r.overlay})`}));
});
root.appendChild(diag);

const marks = el('g');
for (const c of D.cells) {
  const fb = c.r > c.c;                       /* outputs in rows -> feedback below */
  const m = el('rect', {x: X0 + c.c * CELL + 2.5, y: Y0 + c.r * CELL + 2.5,
    width: CELL - 5, height: CELL - 5, rx: 1.5, fill: c.colour,
    'fill-opacity': fb ? 1 : .7, class: fb ? 'fb' : ''});
  m.dataset.i = JSON.stringify(c);
  marks.appendChild(m);
}
root.appendChild(marks);

/* ---- one outline per genuinely coupled block, plus a ring on each of its members ----

   The box alone is not enough and in the provenance ordering it is close to useless: a
   block whose members are scattered has a *bounding* box, which here spans 157 of 161
   rows and says only "somewhere in here". The rings are what actually locate the block,
   and a block drawn as k rings inside one dashed box is exactly the fact worth seeing --
   these nodes are one solve, and this ordering has strewn them across the design. */
/* On the structure page a solve's ring is its problem's kind, and its width falls with
   its depth -- the outer optimiser's ring is the heaviest, the root find two levels in
   the lightest -- so nesting reads at a glance even where the areas' shading is subtle.
   A coupled block nothing drives keeps the accent ring it always had: it is a cycle a
   cut has not reached, and that is the one thing on the page that is wrong rather than
   nested. */
const boxes = el('g');
for (const b of BOXES) {
  const x = X0 + b.from * CELL, y = Y0 + b.from * CELL, w = (b.to - b.from + 1) * CELL;
  /* Rings are the structure page's: a solve in its kind's colour, a cycle nothing yet
     answers in the paper's grey. The provenance page draws no rings -- an SCC's members
     are located there by their marks, and a red ring said nothing the marks did not. */
  if (!STRUCT) continue;
  const kind = ringOf(b);
  const style = kind
    ? {class: 'solve', stroke: kind, 'stroke-width': Math.max(1, 2.8 - 0.6 * b.level)}
    : {class: 'box' + (b.crosses ? '' : ' ok')};
  const r = el('rect', {x: x - 1.5, y: y - 1.5, width: w + 3, height: w + 3, rx: 3,
    'stroke-opacity': b.contiguous ? 1 : .5,
    'stroke-dasharray': b.contiguous ? 'none' : '4 3', ...style});
  r.dataset.i = JSON.stringify({box: b});
  boxes.appendChild(r);
  /* The rings locate a scattered block; a contiguous solve on the structure page is
     located by its box, and ringing each of 123 members would only thicken the diagonal. */
  if (!STRUCT || !b.contiguous) for (const i of b.at) {
    const ring = el('rect', {x: X0 + i * CELL - 1.5, y: Y0 + i * CELL - 1.5,
      width: CELL + 3, height: CELL + 3, rx: 3, ...style});
    ring.dataset.i = JSON.stringify({box: b});
    boxes.appendChild(ring);
  }
  /* The box's name -- kind and driver -- in the gutter, on the row the box starts at
     (its problem's row, when it has one), indented by how deep it is nested; a hairline
     from the box's right edge leads the eye across. */
  if (STRUCT) {
    const gx = X0 + n * CELL + solveX + 8 * boxDepth(b), gy = y + CELL / 2 + 3;
    boxes.appendChild(el('line', {x1: x + w + 1.5, y1: y + CELL / 2, x2: gx - 3,
      y2: y + CELL / 2, stroke: kind || '#9d9d9d', 'stroke-width': .5,
      'stroke-opacity': .5, 'stroke-dasharray': '2 3'}));
    const t = el('text', {class: 'slabel', x: gx, y: gy, fill: kind || '#9d9d9d'});
    t.textContent = boxLabel(b) + ' (' + b.size + ')';
    t.dataset.i = JSON.stringify({box: b});
    boxes.appendChild(t);
  }
}
root.appendChild(boxes);

const cross = el('g', {id: 'cross'});
const cx = el('rect', {width: 0, height: 0}), cy = el('rect', {width: 0, height: 0});
cross.appendChild(cx); cross.appendChild(cy); root.appendChild(cross);

/* ---- legend ---- */
const side = document.getElementById('side');
side.innerHTML =
  '<h2>Groups' + (D.recycled ? ' <span class="scatter">(colours recycled)</span>' : '') +
  '</h2><div class="lg">' + D.legend.map(g =>
    `<div class="sw" style="background:${g.colour}${
      g.overlay === 'hatch-stripe'
        ? ';background-image:repeating-linear-gradient(45deg,#fff9 0 1.5px,#0000 1.5px 4px)'
        : g.overlay === 'hatch-dot'
        ? ';background-image:radial-gradient(#fff9 22%,#0000 23%);background-size:4px 4px'
        : ''}"></div>` +
    `<div class="nm" style="padding-left:${g.level * 11}px" ` +
    `title="${esc(g.full)}">${esc(g.label)}</div>` +
    `<div class="ct" title="nodes / contiguous stretches in the run order">${g.size}` +
    (g.runs > 1 ? ` <span class="scatter">&times;${g.runs}</span>` : '') + '</div>'
  ).join('') + '</div>';
/* The strategy, on the structure page: every solve as a tree, indented by its level --
   what answers what, inside what, by which algorithm -- and the two keys the boxes are
   drawn in, kind (ring) and depth (shade). The kinds are listed on both pages, since a
   problem row is marked by its kind wherever it sits. */
if (STRUCT && BOXES.length) {
  side.innerHTML = '<h2>Solve strategy</h2><div class="tree">' + BOXES.map(b => {
    const colour = ringOf(b) || '#9d9d9d';
    return `<div class="sw" style="border-color:${colour}"></div>` +
      `<div class="nm" style="padding-left:${boxDepth(b) * 11}px" title="${esc(
        b.problem === null ? 'nothing drives this block' : D.rows[b.problem].name)}">` +
      `${esc(boxLabel(b))}</div><div class="ct" title="nodes in the block">${b.size}</div>`;
  }).join('') + '</div>' + side.innerHTML;
}
if (D.kinds.length) {
  side.innerHTML += '<h2>Problem kinds</h2><div class="tree">' + D.kinds.map(k =>
    `<div class="sw" style="border-color:${k.colour}"></div>` +
    `<div class="nm" title="${esc(k.text)}">${esc(k.kind)}</div>` +
    `<div class="ct" title="problem rows of this kind">${k.count}</div>`).join('') + '</div>';
}
if (STRUCT) {
  const depths = [];
  for (let d = 0; d <= D.depths; d++) depths.push(d);
  side.innerHTML += '<h2>Nesting depth</h2><div class="lg">' + depths.map(d =>
    `<div class="sw" style="background:var(--fg);opacity:${depthAlpha(d).toFixed(3)}"></div>` +
    `<div class="nm">depth ${d}</div>` +
    `<div class="ct" title="rows at this depth">${D.rows.filter(r => r.depth === d).length}</div>`
  ).join('') + '</div>';
}

/* ---- hover ---- */
function show(e, html) {
  tip.innerHTML = html; tip.style.display = 'block';
  const r = tip.getBoundingClientRect();
  tip.style.left = Math.min(e.clientX + 14, innerWidth - r.width - 8) + 'px';
  tip.style.top = Math.min(e.clientY + 14, innerHeight - r.height - 8) + 'px';
}
/* Everything between the two markers below is *pure*: it reads `D` and `esc` and touches
   no DOM, no event and no layout. That is what makes the wording testable without a
   browser -- `functional_process/tests/test_dsm_tooltips.py` slices this block out of a
   rendered page by those markers, evaluates it beside the page's own `D`, and asserts on
   the literal string. Keep it that way: anything here that reached for `tip` or an event
   would take the tooltip's text back out of reach of the only check there is on it. */
/* TOOLTIP-TEXT-BEGIN */
/* A capped list of names as its own indented block, with the tail counted rather than
   drawn -- the one idiom every list in this tooltip uses (a node's ports, a cell's
   shared variables), so `... N more` always means the same thing. `total` is the real
   length; `vars` is what survived `TIP_VARS` back in `_matrix_struct`. */
function varList(vars, total) {
  return vars.map(v => '  ' + esc(v)).join('\n') +
    (total > vars.length ? `\n  <span class="dim">... ${total - vars.length} more</span>` : '');
}
/* A node's own i/o, from its *declared* ports -- the same `reads:` / `writes:` pair
   cottax's XDSM tip shows for a diagonal box, and for the same reason: the diagonal is
   where a reader is asking about the node, not about a coupling. A name repeated twice
   (row and column, which on the diagonal are the same node) answered nothing. */
function portList(label, vars, total) {
  if (!total) return `<span class="h">${label}:</span> <span class="dim">none</span>`;
  return `<span class="h">${label}:</span> ${total}\n` + varList(vars, total);
}
/* A problem row says what it is and who answers it, on one line under the header --
   only a problem row, so a body's tip reads exactly as it did. */
function solveLine(r) {
  return r.kind ? `<span class="h">solve:</span> ${esc(r.kind)} &middot; ${esc(r.driver)}\n` : '';
}
function nodeTip(i) {
  const r = D.rows[i];
  return `<b>${esc(r.name)}</b>\n` +
    `<span class="dim">${esc(r.group)} &middot; row/col ${i}</span>\n` +
    solveLine(r) +
    portList('reads', r.reads, r.nr) + '\n' + portList('writes', r.writes, r.nw);
}
/* Variables first. The question at a mark is *which* variable couples these two nodes;
   the two names are already on the row and column headers and under the crosshair, so
   leading with them spent the tip's first line on what the reader could already see.
   They stay, as the footer, spelt `from`/`to` so the direction needs no memory of which
   axis is which -- and the feedback marking, the one thing a DSM exists to show, keeps
   its arrow and its accent down there where it qualifies the coupling rather than
   announcing it.

   One variable is its own headline; several get a count above the list, so the first
   line is the size of the coupling either way. */
function cellTip(d) {
  const r = D.rows[d.r], c = D.rows[d.c], fb = d.r > d.c;
  const head = d.n === 1 ? `<b>${esc(d.v[0])}</b>`
    : `<b>${d.n} variables</b>\n` + varList(d.v, d.n);
  return head + '\n' +
    (fb ? '<span class="fb">&#8595; feeds backwards</span>'
        : '<span class="dim">feeds forwards</span>') + '\n' +
    `<span class="dim">  from  ${esc(r.name)}\n  to    ${esc(c.name)}</span>`;
}
/* TOOLTIP-TEXT-END */
stage.addEventListener('mousemove', e => {
  const t = e.target;
  if (t.dataset && t.dataset.i) {
    const d = JSON.parse(t.dataset.i);
    if (d.box) {
      show(e, `<b>${esc(boxLabel(d.box))}</b>\n` +
        `<span class="dim">level ${d.box.level} &middot; ` +
        `block of ${d.box.size} (${d.box.real} not minted)</span>\n` +
        (d.box.problem === null ? '' : `answers: ${esc(D.rows[d.box.problem].name)}\n`) +
        `groups: ${esc(d.box.groups.join(', '))}\n` +
        `contained in: ${esc(d.box.container)}${d.box.crosses ? ' (nothing)' : ''}\n` +
        esc(d.box.members.join('\n')));
    } else {
      show(e, cellTip(d));
    }
    return;
  }
  const p = pt(e);
  const i = Math.floor((p.y - Y0) / CELL), j = Math.floor((p.x - X0) / CELL);
  if (i >= 0 && i < n && j >= 0 && j < n) {
    cx.setAttribute('x', X0 + j * CELL); cx.setAttribute('y', Y0);
    cx.setAttribute('width', CELL); cx.setAttribute('height', n * CELL);
    cy.setAttribute('x', X0); cy.setAttribute('y', Y0 + i * CELL);
    cy.setAttribute('width', n * CELL); cy.setAttribute('height', CELL);
    /* On the diagonal row and column are one node, so the generic two-name readout said
       the same thing twice. `nodeTip` answers the question that cell actually poses. */
    show(e, i === j ? nodeTip(i)
      : `row ${i}  <b>${esc(D.rows[i].name)}</b>\ncol ${j}  <b>${esc(D.rows[j].name)}</b>`);
  } else {
    cx.setAttribute('width', 0); cy.setAttribute('width', 0);
    tip.style.display = 'none';
  }
});
stage.addEventListener('mouseleave', () => { tip.style.display = 'none'; });

/* ---- pan / zoom ---- */
let k = 1, tx = 0, ty = 0;
const apply = () => root.setAttribute('transform', `translate(${tx} ${ty}) scale(${k})`);
function pt(e) { const r = stage.getBoundingClientRect();
  return {x: (e.clientX - r.left - tx) / k, y: (e.clientY - r.top - ty) / k}; }
stage.addEventListener('wheel', e => {
  e.preventDefault();
  const p = pt(e), f = Math.exp(-e.deltaY * 0.0016), nk = Math.min(12, Math.max(0.06, k * f));
  const r = stage.getBoundingClientRect();
  tx = e.clientX - r.left - p.x * nk; ty = e.clientY - r.top - p.y * nk; k = nk; apply();
}, {passive: false});
let drag = null;
stage.addEventListener('mousedown', e => { drag = {x: e.clientX - tx, y: e.clientY - ty};
  stage.classList.add('drag'); });
addEventListener('mouseup', () => { drag = null; stage.classList.remove('drag'); });
addEventListener('mousemove', e => { if (!drag) return;
  tx = e.clientX - drag.x; ty = e.clientY - drag.y; apply(); });
function fit() {
  const r = stage.getBoundingClientRect();
  k = Math.min((r.width - 24) / W, (r.height - 24) / H);
  tx = (r.width - W * k) / 2; ty = (r.height - H * k) / 2; apply();
}
document.getElementById('fit').onclick = fit;
document.getElementById('reset').onclick = () => { k = 1; tx = 12; ty = 12; apply(); };
addEventListener('resize', fit);
fit();
</script>"""


def render_grouped_dsm_html(
    blocking: Blocking,
    *,
    order: Sequence[NodePath] | None = None,
    depth: int | None = None,
    title: str = "Grouped DSM",
    file_name: str = "dsm_grouped",
    outdir: str = ".",
    write: bool = False,
    formatter: Formatter = NoFormat(),
    mode: str | None = None,
) -> HtmlDoc:
    """`blocking`'s graph as a DSM in `order`, every row coloured by the group its name
    declares.

    `mode` (`MODES`) picks the page: the *provenance* page bands the axes by namespace
    and boxes the top level's coupled blocks; the *structure* page bands them by nesting
    depth and boxes every solve at every level, shaded by depth and ringed by kind.
    `None` decides by the order: structure if `order` is `structure_order(blocking)`,
    provenance otherwise.
    """
    order = structure_order(blocking) if order is None else tuple(order)
    if mode is None:
        mode = "structure" if order == structure_order(blocking) else "provenance"
    struct = _matrix_struct(blocking, order, depth=depth, formatter=formatter, mode=mode)
    # The placeholders are spent **before** the data goes in, not after: a node whose name
    # happened to spell `__TITLE__` would otherwise have the title substituted into the
    # middle of the graph. `</` is broken up for the same reason one level down -- a name
    # holding `</script>` would end the script tag early.
    page = _PAGE.replace("__TITLE__", _xesc(title)).replace(
        "__DATA__", json.dumps(struct).replace("</", "<\\/")
    )
    doc = HtmlDoc("<!doctype html>\n" + page)
    if write:
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, f"{file_name}.html")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(str(doc))
        doc.path = path
    return doc
