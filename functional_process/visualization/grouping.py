'''
Grouping this port's graph by the *prefix* of its node names, and drawing the two orderings.

**This is the port's, not cottax's.** An earlier draft put it in
`cottax.visualization`; it does not belong there. Reading a group off a name is leading-key
matching, not a graph primitive, and the rest is a bespoke DSM renderer for one picture --
cottax already has `ragraph_dsm.render_dsm_html`, and a library whose stated instinct is to
stay thin does not want a second one. If the two hooks this needed (row order from outside,
cell colour by group rather than by edge kind) turn out to be generally wanted, they belong
as parameters on the renderer cottax already ships, not as a parallel path.

Grouping a graph by the *prefix* of its node names, and drawing the two orderings.

A hierarchical `NodePath` carries two facts at once. `physics.profiles.DensityProfile`
says both **who declared this** (`physics.profiles`) and **what it is** (`DensityProfile`),
and the first of those is a grouping nobody had to write down separately: it is already in
the name. This module reads it back out, and draws the one picture that makes it worth
having -- the same graph ordered by that grouping beside the same graph ordered by
`Blocking`.

**The two orderings answer different questions and neither is derived from the other.**
Provenance is about ownership, naming and configuration; structure is about scheduling.
Where they agree the human name is trustworthy. Where they disagree the disagreement is
the signal: a group scattered across the run order is a *label*, not a module, and a block
spanning groups is a genuine cross-group feedback loop. `grouping_report` measures exactly
those two, and `render_grouped_dsm_html` draws them.

Three rules are load-bearing and are stated once.

**A group is the leading namespace keys of a name, minus the node's own final key.**
Both key kinds count, because in a `NodePath` both are namespace positions: a
`GetAttrKey` is a slot in the machine tree and a `DictKey` a mapping key, and the kind
follows the container rather than marking naming apart from grouping. A name with no
such prefix -- a flat `['Build']` -- is `UNGROUPED` and says so, rather than being filed
under whatever its first key happened to spell.

*(This rule used to be "the leading `DictKey`s, asked by kind", with a `GetAttrKey` read
as a place in the caller's pytree. `model_tree_design.md` §8 step 3 inverted that: slots
mint `GetAttrKey`s deliberately, so the kind stopped separating a model-tree position
from a variable place and every machine node silently fell to `UNGROUPED`. See
`_tree_keys`.)*

**A node minted over a *variable* place is `UNGROUPED`, and that is asked of the graph,
not of a key kind.** `^problem.fwbs.f_ster_div_single` is a `FixedPointCut`'s problem
named after the variable it cuts; `.fwbs` is a `DataStructure` area, and grouping by it
would invent a subsystem. `group_of`'s `among` is how the question is put -- a minted
name whose unminted form is not itself a node.

**A minted name is unwrapped first, so a minted node inherits the group of what it was
minted over.** `^problem.stellarator.coils.Intersect` is the driver of
`stellarator.coils.Intersect` and belongs beside it; drawing it anywhere else would put a
block's own problem outside the block. This is `is_minted`/`unminted` asked once, so the
rule holds for every namespace at once and no caller carries the list.

**A block's *real* size is its non-minted membership.** A node paired with the problem
minted over it is a two-node block that couples nothing -- counting it as coupling would
report every declared `FixedPoint` in the graph as a feedback loop. `BlockGrouping.real`
is that count, and the headline figures filter on it.
'''

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
import dataclasses
import json

from jax.tree_util import DictKey, GetAttrKey

from cottax.blocking import Blocking
from cottax.graph import Graph
from cottax.spec import DeclaredNode, NodePath, VarPath
from cottax.tools.minting import is_minted, unminted
from cottax.visualization.xdsm import Formatter, NoFormat, _xesc
from cottax.visualization.xdsm_html import HtmlDoc

type Group = tuple[str, ...]

UNGROUPED: Group = ()
'''
The group of a node whose name carries no prefix at all.

Not an error and not a bucket named `'other'`: an empty prefix is exactly what a flat name
*says*, and a picture that spelt it as a group would invent a containment the name does
not claim. Every flat graph is entirely `UNGROUPED`, which is the honest reading of a
graph whose names have no tree in them yet.
'''

UNGROUPED_LABEL = '(ungrouped)'


# ================================================================== reading the prefix
def _tree_keys(path: NodePath) -> tuple[str, ...]:
    '''
    The leading run of namespace keys of `path`, once any minted root is dropped.

    **Both key kinds count, because in a `NodePath` both are namespace positions.** A
    `GetAttrKey` is a slot in the machine tree (`.stellarator.coils.coil_current`) and a
    `DictKey` a mapping key (`['MiscPlantEquipmentCost']`, a switch arm) -- the kind
    follows the container, which is cottax's `node_and_names` rule, not a distinction
    between naming and grouping.

    **This tested for `DictKey` alone, and `model_tree_design.md` §8 step 3 made that
    wrong in the worst way: silently.** Before step 3 a model-tree position was a
    `DictKey` and a `GetAttrKey` could only be a variable place, so the kind was a sound
    proxy for "is this a position in a model tree". Step 3 made slots mint `GetAttrKey`s
    on purpose (that design's §3.1, so a node name is a working address into the tree),
    which inverted the proxy: every machine node stopped having any leading key at all
    and fell to `UNGROUPED`, while `grouped` still exited 0 and wrote both files. The
    report said `1 group(s) at depth 1` where there are six, and a picture whose whole
    subject is the grouping drew one colour.

    What the kind stood in for -- model-tree position against a node minted over a
    *variable* place -- is now asked directly, by `group_of`'s `among`.
    '''
    out: list[str] = []
    for key in unminted(path).keys:
        if isinstance(key, DictKey) and isinstance(key.key, str):
            out.append(key.key)
        elif isinstance(key, GetAttrKey):
            out.append(key.name)
        else:
            break
    return tuple(out)


def group_of(path: NodePath, *, depth: int = 1,
             among: 'Iterable[NodePath] | None' = None) -> Group:
    '''
    The group `path` declares it belongs to: its leading keys, without its own name.

    `depth` is the grain. `depth=1` gives the top-level subsystem
    (`stellarator.coils.Intersect -> ('stellarator',)`), `depth=2` the one below it
    (`('stellarator', 'coils')`). A name shallower than `depth` gives whatever prefix it
    has, so a two-level and a three-level name can be grouped together without either
    being special-cased.

    The node's own final key is never part of its group -- `['Build']` is `UNGROUPED`,
    not a group of one called `Build`.

    **`among` is the node set, and it separates a minted model-tree position from a
    minted variable place.** A name minted over a node unmints to a name that is *in* the
    graph (`^problem.stellarator.coils.intersect`); one minted over a variable does not
    (`^problem.fwbs.f_ster_div_single` -- `.fwbs` is a `DataStructure` area, and reading
    it as a group invents a subsystem). Omitting `among` does not ask, and reads every
    minted name as a tree position -- the honest fallback for a caller with no graph to
    consult. Measured on the reference graph: one node of 159 is in the second category,
    and unasked it forms a phantom `fwbs` group of one, a seventh subsystem in the legend
    beside the six real ones.
    '''
    if depth < 1:
        raise ValueError(f'depth must be at least 1, not {depth}')
    if among is not None and is_minted(path) and unminted(path) not in among:
        return UNGROUPED
    keys = _tree_keys(path)
    return keys[: min(depth, max(len(keys) - 1, 0))]


def group_label(group: Group) -> str:
    '''How a group is written: dotted, like the names it is a prefix of.'''
    return '.'.join(group) if group else UNGROUPED_LABEL


# ================================================================== the two orderings
def group_sequence(names: Iterable[NodePath], *, depth: int = 1) -> tuple[Group, ...]:
    '''
    Every group present, in **first-appearance order** over `names`.

    Stable and declared: a graph's node order is its binding order, so this is the order
    the groups were first written down in, and it does not move when an unrelated node is
    added. Deliberately not alphabetical -- sorting would impose an order the declaration
    never claimed, and the point of the provenance view is to show the one it did.
    '''
    names = tuple(names)
    among = frozenset(names)
    seen: dict[Group, None] = {}
    for name in names:
        seen.setdefault(group_of(name, depth=depth, among=among), None)
    return tuple(seen)


def provenance_order(
    names: Iterable[NodePath],
    *,
    depth: int = 1,
    groups: Sequence[Group] | None = None,
) -> tuple[NodePath, ...]:
    '''
    `names` regrouped so every member of a group is adjacent, groups in declared order.

    Within a group the input order is kept, so the only thing this changes is which nodes
    sit next to which. **It is not a run order** and makes no claim to be one: nothing
    here consults an edge. Marks above the diagonal of a DSM drawn in this order are
    therefore *not* feedback -- they are places where provenance and dependency disagree,
    which is the comparison the picture exists to make.

    `groups` overrides the group order (`group_sequence`'s otherwise); every group present
    must appear in it.
    '''
    names = tuple(names)
    among = frozenset(names)
    order = tuple(groups) if groups is not None else group_sequence(names, depth=depth)
    index = {g: i for i, g in enumerate(order)}
    missing = {group_of(n, depth=depth, among=among) for n in names} - set(index)
    if missing:
        raise KeyError(
            f'group(s) {sorted(group_label(g) for g in missing)} are in the graph but not '
            f'in the `groups` order given'
        )
    return tuple(sorted(names, key=lambda n: index[group_of(n, depth=depth, among=among)]))


def structure_order(blocking: Blocking) -> tuple[NodePath, ...]:
    '''
    `blocking`'s own order, flattened: the order the graph actually runs in.

    The top level only. A block's interior is its own blocking and draws its own picture;
    at this level a block is contiguous by construction, which is what lets it be one
    square on the diagonal.
    '''
    return tuple(name for block in blocking.blocks for name in block)


# ================================================================== the measurement
@dataclasses.dataclass(frozen=True)
class BlockGrouping:
    '''One block of a `Blocking`, and which groups it is made of.'''

    members: tuple[NodePath, ...]
    groups: tuple[Group, ...]
    '''The distinct groups its members declare, `UNGROUPED` included, in member order.'''

    @property
    def real(self) -> int:
        '''Members that are not minted names -- the block's coupling, in §11's sense.'''
        return sum(1 for m in self.members if not is_minted(m))

    @property
    def named_groups(self) -> tuple[Group, ...]:
        return tuple(g for g in self.groups if g != UNGROUPED)

    @property
    def crosses(self) -> bool:
        '''Whether it spans more than one *named* group: a cross-group feedback loop.'''
        return len(self.named_groups) > 1


@dataclasses.dataclass(frozen=True)
class GroupingReport:
    '''What provenance and structure agree and disagree about, on one graph.'''

    depth: int
    groups: tuple[Group, ...]
    sizes: Mapping[Group, int]
    runs: Mapping[Group, int]
    '''
    How many maximal contiguous stretches each group occupies in the **run** order.

    One means the group is also a schedulable unit -- provenance and structure agree about
    it. Many means its members are interleaved with other groups' throughout the schedule:
    the group is a *label*, not a module. This is the second of § 11.2's two signals, and
    the only one a count of blocks cannot give (almost every block is a single node, so
    "how many blocks does this group touch" just re-counts its members).
    '''

    blocks: tuple[BlockGrouping, ...]
    cross_group_edges: int
    '''Node-to-node edges whose endpoints are in different named groups.'''

    @property
    def coupled(self) -> tuple[BlockGrouping, ...]:
        '''The blocks that genuinely couple: more than one non-minted node.'''
        return tuple(b for b in self.blocks if b.real > 1)

    @property
    def crossing(self) -> tuple[BlockGrouping, ...]:
        '''Of those, the ones spanning more than one group.'''
        return tuple(b for b in self.coupled if b.crosses)

    def summary(self) -> str:
        return (
            f'{len(self.groups)} group(s) at depth {self.depth}; '
            f'{len(self.blocks)} block(s), {len(self.coupled)} with more than one real '
            f'node, {len(self.crossing)} of those crossing a group boundary; '
            f'{self.cross_group_edges} cross-group edge(s); '
            f'{sum(1 for g in self.groups if self.runs.get(g, 0) == 1)}/'
            f'{len(self.groups)} group(s) contiguous in the run order'
        )


def grouping_report(blocking: Blocking, *, depth: int = 1) -> GroupingReport:
    '''
    Measure provenance against structure on `blocking`: § 11's table, for any graph.

    Takes a `Blocking` rather than a `Graph` because "structure" is a partition somebody
    chose -- `Blocking.scc` is the finest honest one, `Blocking.fused` a coarser one, and
    which was meant is not this function's decision to make.
    '''
    graph = blocking.graph
    order = structure_order(blocking)
    groups = group_sequence(graph.nodes, depth=depth)
    among = frozenset(graph.nodes)
    at = {name: group_of(name, depth=depth, among=among) for name in graph.nodes}

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

    owners = graph.owners
    crossing = {
        (owners[var], name)
        for name in graph.nodes
        for var in graph[name].reads
        if var in owners and at[owners[var]] != at[name]
        and at[owners[var]] != UNGROUPED and at[name] != UNGROUPED
    }
    return GroupingReport(depth, groups, sizes, runs, blocks, len(crossing))


# ================================================================== the drawing
PALETTE = (
    '#4c78a8', '#f58518', '#54a24b', '#e45756', '#b279a2', '#72b7b2',
    '#eeca3b', '#9d7660', '#ff9da6', '#8dd3c7', '#bab0ac', '#5c9ecf',
)
'''
One colour per group, mid-luminance so every one of them reads on white and on black.

Chosen for that constraint rather than for prettiness: a palette tuned for a light page
goes to mud on a dark one, and this picture is drawn in whichever the reader's system
asks for. A greyscale reader still has the group ribbon's written label beside every row.
'''

UNGROUPED_COLOUR = '#8c8c8c'

TIER_OVERLAY = (None, 'hatch-stripe', 'hatch-dot')
'''
What a group beyond the palette's length is drawn with, on top of its recycled colour.

More groups than colours is a real case and silently recycling would make two groups
indistinguishable at exactly the moment there are too many to hold in your head. So the
colour recycles and a texture is laid over it: the 13th group is the 1st's blue under
stripes, the 25th the same blue under dots. Past that the texture recycles too and the
picture says so in its legend, because three tiers of texture is already more than a
reader can keep apart and pretending otherwise would be worse than admitting it.
'''


def group_style(index: int) -> tuple[str, str | None]:
    '''The colour and overlay texture the `index`-th group is drawn with.'''
    return PALETTE[index % len(PALETTE)], TIER_OVERLAY[
        (index // len(PALETTE)) % len(TIER_OVERLAY)
    ]


def _edges(graph: Graph) -> dict[tuple[NodePath, NodePath], list[VarPath]]:
    '''Node -> node, with the variables that flow along each. The boundary is dropped.'''
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
    depth: int,
    formatter: Formatter,
) -> dict:
    '''
    Everything the page draws, as plain data: rows, cells, group bands, block boxes.

    Built here and shipped as JSON, exactly as `render_xdsm_html` ships `xdsm_struct` --
    the browser lays out and paints, and no structural decision is taken there.
    '''
    graph = blocking.graph
    order = tuple(order)
    if set(order) != set(graph.nodes):
        raise ValueError(
            f'the order given holds {len(set(order))} of the graph\'s {len(graph.nodes)} '
            f'node(s) -- a DSM is a permutation of the whole graph, not a selection'
        )

    among = frozenset(graph.nodes)
    at = {name: group_of(name, depth=depth, among=among) for name in graph.nodes}
    groups = group_sequence(graph.nodes, depth=depth)
    palette = {g: group_style(i) for i, g in enumerate(groups)}
    palette[UNGROUPED] = (UNGROUPED_COLOUR, None)
    index = {name: i for i, name in enumerate(order)}

    rows = [
        {
            'name': formatter.node((name, graph[name])),
            'group': group_label(at[name]),
            'colour': palette[at[name]][0],
            'overlay': palette[at[name]][1],
            'problem': isinstance(graph[name], DeclaredNode),
            'minted': is_minted(name),
        }
        for name in order
    ]

    cells = [
        {
            'r': index[target],          # inputs in rows: this row reads ...
            'c': index[source],          # ... from this column
            'n': len(shared),
            'v': [formatter.var(v) for v in shared[:12]],
            'colour': palette[at[source]][0],
        }
        for (source, target), shared in _edges(graph).items()
    ]

    bands: list[dict] = []
    start = 0
    for i, name in enumerate(order):
        if i + 1 == len(order) or at[order[i + 1]] != at[name]:
            bands.append({
                'from': start, 'to': i,
                'label': group_label(at[name]),
                'colour': palette[at[name]][0],
                'overlay': palette[at[name]][1],
            })
            start = i + 1

    report = grouping_report(blocking, depth=depth)
    boxes = [
        {
            'from': min(index[m] for m in b.members),
            'to': max(index[m] for m in b.members),
            'size': len(b.members),
            'real': b.real,
            'crosses': b.crosses,
            'contiguous': (max(index[m] for m in b.members)
                           - min(index[m] for m in b.members) + 1) == len(b.members),
            'groups': [group_label(g) for g in b.groups],
            'members': [formatter.node((m, graph[m])) for m in b.members],
            'at': sorted(index[m] for m in b.members),
        }
        for b in report.coupled
    ]

    legend = [
        {
            'label': group_label(g),
            'colour': palette[g][0],
            'overlay': palette[g][1],
            'size': report.sizes[g],
            'runs': report.runs[g],
        }
        for g in groups
    ]
    return {
        'rows': rows, 'cells': cells, 'bands': bands, 'boxes': boxes,
        'legend': legend, 'summary': report.summary(),
        'backward': sum(c['n'] for c in cells if c['r'] < c['c']),
        'reads': sum(c['n'] for c in cells),
        'coupled': len(report.coupled), 'crossing': len(report.crossing),
        'crossEdges': report.cross_group_edges,
        'tiers': len(TIER_OVERLAY),
        'recycled': len(groups) > len(PALETTE) * len(TIER_OVERLAY),
    }


_PAGE = r'''<meta charset="utf-8"><title>__TITLE__</title>
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
header { flex:0 0 auto; padding:14px 18px 10px; border-bottom:1px solid var(--rule); }
h1 { margin:0 0 3px; font-size:16px; font-weight:600; letter-spacing:.01em; }
.sub { color:var(--dim); font-size:12px; max-width:120ch; }
.sub b { color:var(--fg); font-weight:600; }
.sub code { font-family:ui-monospace,Menlo,monospace; }
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
.note { color:var(--dim); font-size:11px; margin:6px 0 0; }
.note code { font-family:ui-monospace,Menlo,monospace; }
.scatter { color:var(--accent); font-weight:600; }
#stage { flex:1; position:relative; overflow:hidden; cursor:grab; }
#stage.drag { cursor:grabbing; }
#tip { position:fixed; pointer-events:none; z-index:9; background:var(--bg);
  border:1px solid var(--rule); border-radius:6px; padding:6px 8px; font-size:11px;
  max-width:44ch; box-shadow:0 4px 14px #0003; display:none;
  font-family:ui-monospace,Menlo,monospace; white-space:pre-wrap; }
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
<header>
  <h1>__TITLE__</h1>
  <div class="sub">__SUB__</div>
</header>
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
const gutter = 16 + 6.2 * Math.max(...D.bands.map(b => b.label.length));
const X0 = lblW + BAND + GAP, Y0 = lblW + BAND + GAP;
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
  for (const axis of [0, 1]) {
    const x = axis ? X0 + b.from * CELL : lblW, y = axis ? lblW : Y0 + b.from * CELL;
    const w = axis ? len : BAND, h = axis ? BAND : len;
    band.appendChild(el('rect', {x, y, width: w, height: h, fill: b.colour, rx: 2}));
    if (b.overlay)
      band.appendChild(el('rect', {x, y, width: w, height: h, fill: `url(#${b.overlay})`, rx: 2}));
  }
  for (const p of [b.from, b.to + 1]) {
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
    const name = el('text', {class: 'gname', x: X0 + n * CELL + 6,
      y: Y0 + (b.from + (b.to - b.from + 1) / 2) * CELL + 3.5, fill: b.colour});
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
  const fb = c.r < c.c;                       /* inputs in rows -> feedback above */
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
const scat = D.legend.filter(g => g.runs > 1).length;
side.innerHTML =
  '<h2>Groups</h2><div class="lg">' + D.legend.map(g =>
    `<div class="sw" style="background:${g.colour}${
      g.overlay === 'hatch-stripe'
        ? ';background-image:repeating-linear-gradient(45deg,#fff9 0 1.5px,#0000 1.5px 4px)'
        : g.overlay === 'hatch-dot'
        ? ';background-image:radial-gradient(#fff9 22%,#0000 23%);background-size:4px 4px'
        : ''}"></div>` +
    `<div class="nm">${esc(g.label)}</div>` +
    `<div class="ct" title="nodes / contiguous stretches in the run order">${g.size}` +
    (g.runs > 1 ? ` <span class="scatter">&times;${g.runs}</span>` : '') + '</div>'
  ).join('') + '</div>' +
  `<p class="note">Right-hand figure is the group's node count; a red <span class="scatter">&times;k</span> ` +
  `is how many separate stretches its members occupy in the <b>run</b> order. ` +
  `<b>1 stretch</b> means provenance and structure agree about that group; many means the ` +
  `group is a label, not a schedulable unit. Here ${scat} of ${D.legend.length} group(s) are scattered.</p>` +
  '<h2>Reading it</h2><p class="note">Inputs in <b>rows</b>: a mark at (row <i>r</i>, column ' +
  '<i>c</i>) means <i>r</i> reads something <i>c</i> owns, coloured by <i>c</i>\'s group. ' +
  'So marks <b>below</b> the diagonal are feed-forward and marks <b>above</b> it, outlined in ' +
  'red, run backwards in this ordering (ragraph\'s <code>IR_FAD</code>; the plotly ' +
  '<code>dsm.html</code> uses the mirrored <code>IC_FBD</code>).</p>' +
  '<p class="note">A <b>ring</b> on a diagonal cell marks a node in a block that genuinely ' +
  'couples (more than one non-minted node), and the outline around them is that block. ' +
  '<b>Red</b> means the block <b>spans</b> more than one group -- a cross-group feedback ' +
  'loop, the most interesting thing this comparison can find; grey means it is contained ' +
  'in one group. A <b>faint dashed</b> outline is a block whose members are not adjacent ' +
  'in this ordering: it is a bounding box, not the block, and the rings inside it are ' +
  'where the block actually is.</p>' +
  '<h2>Totals</h2><p class="note"><b>' + D.backward + ' of ' + D.reads +
  '</b> variable reads run <b>backwards</b> in this ordering (the marks above the ' +
  'diagonal). In the run order that figure is exactly the coupling inside the blocks; ' +
  'in any other ordering the excess is how far that ordering disagrees with what the ' +
  'graph depends on.</p><p class="note">' + D.summary + '</p>' +
  (D.recycled ? '<p class="note scatter">More groups than colours &times; textures: ' +
    'colours are recycled and two groups may look alike. Read the ribbon label.</p>' : '');

/* ---- hover ---- */
function show(e, html) {
  tip.innerHTML = html; tip.style.display = 'block';
  const r = tip.getBoundingClientRect();
  tip.style.left = Math.min(e.clientX + 14, innerWidth - r.width - 8) + 'px';
  tip.style.top = Math.min(e.clientY + 14, innerHeight - r.height - 8) + 'px';
}
stage.addEventListener('mousemove', e => {
  const t = e.target;
  if (t.dataset && t.dataset.i) {
    const d = JSON.parse(t.dataset.i);
    if (d.box) {
      show(e, `<b>block of ${d.box.size} (${d.box.real} not minted)</b>\n` +
        `groups: ${esc(d.box.groups.join(', '))}\n` + esc(d.box.members.join('\n')));
    } else {
      const r = D.rows[d.r], c = D.rows[d.c];
      show(e, `<b>${esc(r.name)}</b>\n  ${d.r < d.c ? '&#8593; reads backwards' : 'reads'} ` +
        `${d.n} var(s) from\n<b>${esc(c.name)}</b>\n` + esc(d.v.join('\n')) +
        (d.n > d.v.length ? `\n... ${d.n - d.v.length} more` : ''));
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
    show(e, `row ${i}  <b>${esc(D.rows[i].name)}</b>\ncol ${j}  <b>${esc(D.rows[j].name)}</b>`);
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
</script>'''


def render_grouped_dsm_html(
    blocking: Blocking,
    *,
    order: Sequence[NodePath] | None = None,
    depth: int = 1,
    title: str = 'Grouped DSM',
    subtitle: str = '',
    file_name: str = 'dsm_grouped',
    outdir: str = '.',
    write: bool = False,
    formatter: Formatter = NoFormat(),
) -> HtmlDoc:
    '''
    `blocking`'s graph as a DSM in `order`, every row coloured by the group its name declares.

    `order` defaults to `structure_order(blocking)` -- the order the graph runs in. Pass
    `provenance_order(blocking.graph.nodes, depth=depth)` for the other view; the two
    together are the comparison this module exists for, and drawing them as two files with
    everything else held fixed is what makes them comparable at a glance.

    Self-contained: one HTML file, no CDN, no external asset, SVG drawn by ~200 lines of
    vanilla JS with wheel-zoom and drag-pan. Deliberately **not** `render_dsm_html`, whose
    figure is plotly's: that route colours cells by *edge kind* and orders rows by the
    ragraph node list, and neither is reachable from the outside, which is precisely the
    two things this picture needs to control. It is a different instrument, not a
    replacement -- `render_dsm_html` keeps the hierarchy folding and the variable-level
    modes this one has no answer for.

    The convention is **inputs in rows, feedback above the diagonal** (ragraph's
    `IR_FAD`), which is the mirror of the `IC_FBD` `render_dsm` draws. Stated in the
    page's own legend rather than left to be inferred, because a silently mirrored DSM is
    read backwards without anything looking wrong.
    '''
    order = structure_order(blocking) if order is None else order
    struct = _matrix_struct(blocking, order, depth=depth, formatter=formatter)
    # The placeholders are spent **before** the data goes in, not after: a node whose name
    # happened to spell `__TITLE__` would otherwise have the title substituted into the
    # middle of the graph. `</` is broken up for the same reason one level down -- a name
    # holding `</script>` would end the script tag early.
    page = (_PAGE
            .replace('__TITLE__', _xesc(title))
            .replace('__SUB__', subtitle or _xesc(struct['summary']))
            .replace('__DATA__', json.dumps(struct).replace('</', '<\\/')))
    doc = HtmlDoc('<!doctype html>\n' + page)
    if write:
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, f'{file_name}.html')
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write(str(doc))
        doc.path = path
    return doc
