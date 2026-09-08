"""Grouping this port's graph by the *prefix* of its node names, and drawing the two
orderings.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
import dataclasses
import json

import networkx as nx

from jax.tree_util import DictKey, GetAttrKey

from cottax.blocking import Blocking
from cottax.graph import Graph
from cottax.spec import ConditionNode, NodePath, VarPath
from cottax.tools.minting import is_minted, unminted
from cottax.visualization.xdsm import Formatter, NoFormat, _xesc
from cottax.visualization.xdsm_html import HtmlDoc

type Group = tuple[str, ...]

UNGROUPED: Group = ()
"""The group of a node whose name carries no prefix at all."""

UNGROUPED_LABEL = "(ungrouped)"


# ================================================================== reading the prefix
def _tree_keys(path: NodePath) -> tuple[str, ...]:
    """The leading run of namespace keys of `path`, once any minted root is dropped."""
    out: list[str] = []
    for key in unminted(path).keys:
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
    keys = unminted(path).keys
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
    """`blocking`'s own order, flattened: the order the graph actually runs in."""
    return tuple(name for block in blocking.blocks for name in block)


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


# ================================================================== the drawing
PALETTE = (
    "#4c78a8",
    "#f58518",
    "#54a24b",
    "#e45756",
    "#b279a2",
    "#72b7b2",
    "#eeca3b",
    "#9d7660",
    "#ff9da6",
    "#8dd3c7",
    "#bab0ac",
    "#5c9ecf",
)
"""One colour per group, mid-luminance so every one of them reads on white and on black.
"""

UNGROUPED_COLOUR = "#8c8c8c"

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
    base = {top: group_style(i) for i, top in enumerate(subsystems)}
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


def _matrix_struct(
    blocking: Blocking,
    order: Sequence[NodePath],
    *,
    depth: int | None,
    formatter: Formatter,
) -> dict:
    """Everything the page draws, as plain data: rows, cells, group bands, block boxes.
    """
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
    boxes = [
        {
            "from": min(index[m] for m in b.members),
            "to": max(index[m] for m in b.members),
            "size": len(b.members),
            "real": b.real,
            "crosses": b.crosses,
            "nests": b.nests,
            "container": group_label(b.container),
            "contiguous": (
                max(index[m] for m in b.members) - min(index[m] for m in b.members) + 1
            )
            == len(b.members),
            "groups": [group_label(g) for g in b.groups],
            "members": [formatter.node((m, graph[m])) for m in b.members],
            "at": sorted(index[m] for m in b.members),
        }
        for b in report.coupled
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
.grid { stroke:var(--rule); stroke-width:.4; }
.sep { stroke:var(--fg); stroke-width:.7; opacity:.35; }
.fb { stroke:var(--accent); stroke-width:1.1; }
.box { fill:none; stroke:var(--accent); stroke-width:1.4; }
.box.ok { stroke:var(--fg); opacity:.55; }
.gname { font-size:10px; font-weight:600; }
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
/* One ribbon lane per level, so the axis is as wide as the tree is deep. Labels get a
   gutter column per level for the same reason: an inner stretch sits inside its outer
   one, so their labels share a y and would collide in a single column. */
const LEVELS = Math.max(1, ...D.bands.map(b => b.level + 1));
const BANDW = BAND * LEVELS;
const lvlW = [];
for (const b of D.bands) lvlW[b.level] = Math.max(lvlW[b.level] || 0, b.label.length);
const lvlX = []; let gut = 6;
for (let L = 0; L < LEVELS; L++) { lvlX[L] = gut; gut += 8 + 6.2 * (lvlW[L] || 0); }
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
for (const b of D.bands) {
  const len = (b.to - b.from + 1) * CELL;
  const off = lblW + b.level * BAND;
  /* Depth is drawn as tint, not as hue -- `group_palette` picked it, so a lane's colour
     is the group's own and not this level's opacity: every namespace of one subsystem
     is a shade of that subsystem's colour, and none of them is a second palette. */
  for (const axis of [0, 1]) {
    const x = axis ? X0 + b.from * CELL : off, y = axis ? off : Y0 + b.from * CELL;
    const w = axis ? len : BAND, h = axis ? BAND : len;
    band.appendChild(el('rect', {x, y, width: w, height: h, fill: b.colour, rx: 2}));
    if (b.overlay)
      band.appendChild(el('rect', {x, y, width: w, height: h, fill: `url(#${b.overlay})`,
        rx: 2}));
  }
  /* Separators mark subsystems only. A rule across the whole matrix for every level of
     every namespace would be a grid, not a separator. */
  for (const p of (b.level ? [] : [b.from, b.to + 1])) {
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
    const name = el('text', {class: 'gname', x: X0 + n * CELL + lvlX[b.level],
      y: Y0 + (b.from + (b.to - b.from + 1) / 2) * CELL + 3.5, fill: b.base});
    name.textContent = b.label;
    band.appendChild(name);
  }
}
root.appendChild(band);

/* ---- row and column labels ---- */
const labels = el('g');
D.rows.forEach((r, i) => {
  const y = Y0 + i * CELL + CELL / 2 + 3;
  const t = el('text', {class: 'lbl' + (r.minted ? ' mint' : ''), x: lblW - 4, y,
    'text-anchor': 'end'});
  t.textContent = r.name; labels.appendChild(t);
  const x = X0 + i * CELL + CELL / 2 + 3;
  const u = el('text', {class: 'lbl' + (r.minted ? ' mint' : ''), x, y: lblW - 4,
    'text-anchor': 'start', transform: `rotate(-90 ${x} ${lblW - 4})`});
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

const diag = el('g');
D.rows.forEach((r, i) => {
  const x = X0 + i * CELL, y = Y0 + i * CELL;
  diag.appendChild(el('rect', {x, y, width: CELL, height: CELL, fill: r.colour,
    'fill-opacity': r.problem ? .45 : .95,
    stroke: r.problem ? r.colour : 'none', 'stroke-width': 1.4}));
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
const boxes = el('g');
for (const b of D.boxes) {
  const x = X0 + b.from * CELL, w = (b.to - b.from + 1) * CELL;
  const r = el('rect', {x: x - 1.5, y: Y0 + b.from * CELL - 1.5, width: w + 3, height: w + 3,
    class: 'box' + (b.crosses ? '' : ' ok'), rx: 3,
    'stroke-opacity': b.contiguous ? 1 : .5,
    'stroke-dasharray': b.contiguous ? 'none' : '4 3'});
  r.dataset.i = JSON.stringify({box: b});
  boxes.appendChild(r);
  for (const i of b.at) {
    const ring = el('rect', {x: X0 + i * CELL - 1.5, y: Y0 + i * CELL - 1.5,
      width: CELL + 3, height: CELL + 3, rx: 3, class: 'box' + (b.crosses ? '' : ' ok')});
    ring.dataset.i = JSON.stringify({box: b});
    boxes.appendChild(ring);
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
function nodeTip(i) {
  const r = D.rows[i];
  return `<b>${esc(r.name)}</b>\n` +
    `<span class="dim">${esc(r.group)} &middot; row/col ${i}</span>\n` +
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
      show(e, `<b>block of ${d.box.size} (${d.box.real} not minted)</b>\n` +
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
) -> HtmlDoc:
    """`blocking`'s graph as a DSM in `order`, every row coloured by the group its name
    declares.
    """
    order = structure_order(blocking) if order is None else order
    struct = _matrix_struct(blocking, order, depth=depth, formatter=formatter)
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
