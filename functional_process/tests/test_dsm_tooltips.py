"""What the grouped DSM's hover actually says, asserted on the literal text.

Two things are pinned here and they are different in kind.

**The payload** (`_matrix_struct`) is the structural half: a row carries the node's
*declared* ports -- `NodeDefinition.reads`/`.owns`, the same two lists `cottax`'s
`xdsm_struct` ships as `reads`/`writes` -- capped at `TIP_VARS`, and **not** the union of
the edges this matrix happens to draw. That distinction is the point of
`test_a_row_s_ports_are_declared_not_read_off_the_matrix`: a node reads boundary
variables nobody in the graph owns and owns variables nobody reads, and neither is a mark
anywhere in the picture. A hover over the diagonal is the one place a reader is asking
about the *node*; answering from the matrix would describe the drawing instead.

**The wording** is the other half, and it is JavaScript, which is why these tests do
something unusual: they slice the page's own tooltip functions out of the rendered HTML
(between the `TOOLTIP-TEXT-BEGIN`/`-END` markers `grouping.py` puts there for exactly
this), evaluate them beside the page's own `D` in a real JS engine, and compare the
returned string to a literal. There is no browser here, so the alternative was asserting
that some substring appears in a template -- which pins the source and not the sentence.
The engine is `quickjs`; where it is not installed the wording tests skip and the payload
tests still run.
"""

import json
import re

import pytest
from jax.tree_util import GetAttrKey

from cottax.blocking import Blocking
from cottax.graph import Graph
from cottax.interfaces.spelling import xDSMFormatterFlat
from cottax.spec import ImplementedFunction, In, NodePath, Out, VarPath
from cottax.tools.path import path_map
from functional_process.cottax.visualization.grouping import (
    TIP_VARS,
    _matrix_struct,
    render_grouped_dsm_html,
)

SPELLING = xDSMFormatterFlat()


def V(*keys) -> VarPath:
    return VarPath(tuple(GetAttrKey(k) for k in keys))


def N(*keys) -> NodePath:
    # `GetAttrKey`, not `DictKey`: the real graph's node names are machine-tree slots and
    # spell with a leading dot (`.buildings.sizing`), so the literals asserted below are
    # the literals the published pages actually show.
    return NodePath(tuple(GetAttrKey(k) for k in keys))


def call(reads, owns):
    return ImplementedFunction(
        inputs=tuple(In(r) for r in reads),
        outputs=tuple(Out(o) for o in owns),
        fn=lambda *a: None,
    )


# A graph built to hold, at known positions, exactly the four cells the wording has to
# read well at: a node with two ports, a node with more ports than `TIP_VARS`, a
# one-variable feed-forward and a many-variable feedback. Nothing about it is a claim
# about PROCESS -- the real pages are checked separately, below, for carrying the payload
# at all.
_LEAN = N("g", "lean")  # 1 read, 1 write -- the "few ports" diagonal
_WIDE = N("g", "wide")  # far more reads than `TIP_VARS` -- the "many ports" diagonal
_SINK = N("h", "sink")  # owns the whole feedback bundle `wide` reads back

_EXTERNAL = [V("ext", f"in{i:02d}") for i in range(TIP_VARS + 2)]  # owned by nobody
_BACK = [V("h", f"back{i:02d}") for i in range(TIP_VARS + 4)]  # the feedback bundle
_WIDE_READS = [V("g", "lean_out"), *_EXTERNAL, *_BACK]


@pytest.fixture
def fixture_graph() -> Graph:
    return Graph(
        path_map({
            _LEAN: call([V("ext", "seed")], [V("g", "lean_out")]),
            _WIDE: call(_WIDE_READS, [V("g", "wide_a"), V("g", "wide_b")]),
            _SINK: call([V("g", "wide_a")], _BACK),
        })
    )


@pytest.fixture
def order():
    """Rows in the order the fixture declares them, so indices below are readable."""
    return (_LEAN, _WIDE, _SINK)


@pytest.fixture
def struct(fixture_graph, order):
    return _matrix_struct(
        Blocking.scc(fixture_graph), order, depth=None, formatter=SPELLING
    )


# ================================================================= the payload
def test_a_row_carries_the_node_s_declared_reads_and_writes(struct):
    lean, wide, sink = struct["rows"]
    assert (lean["reads"], lean["nr"]) == ([".ext.seed"], 1)
    assert (lean["writes"], lean["nw"]) == ([".g.lean_out"], 1)
    assert (sink["reads"], sink["nr"]) == ([".g.wide_a"], 1)
    # Signature/declaration order, not sorted -- the same order `xdsm_struct` ships.
    assert wide["reads"][0] == ".g.lean_out"
    assert wide["reads"][1] == ".ext.in00"


def test_a_long_port_list_is_capped_and_the_count_is_the_real_one(struct):
    wide = struct["rows"][1]
    assert wide["nr"] == len(_WIDE_READS) == 2 * TIP_VARS + 7
    assert len(wide["reads"]) == TIP_VARS
    sink = struct["rows"][2]
    assert sink["nw"] == TIP_VARS + 4
    assert len(sink["writes"]) == TIP_VARS


def test_a_row_s_ports_are_declared_not_read_off_the_matrix(struct):
    """The reason the declared route was taken, stated as a test.

    `.ext.seed` and `.ext.in00` are read by nodes in this graph and owned by none of
    them, and `.g.wide_b` is owned and read by nobody -- so the union of the cells
    touching a row would show neither. A diagonal hover has to answer for the node.
    """
    supplied = {v for cell in struct["cells"] for v in cell["v"]}
    lean, wide, sink = struct["rows"]
    assert ".ext.seed" in lean["reads"]
    assert ".ext.seed" not in supplied
    assert ".ext.in00" in wide["reads"]
    assert ".ext.in00" not in supplied
    assert ".g.wide_b" in wide["writes"]
    assert ".g.wide_b" not in supplied


def test_a_cell_still_carries_its_shared_variables_capped_the_same_way(struct):
    back = next(c for c in struct["cells"] if c["r"] == 2 and c["c"] == 1)
    assert back["n"] == TIP_VARS + 4
    assert len(back["v"]) == TIP_VARS


# ============================================================== the wording
def _payload(html: str) -> str:
    """The page's `const D = {...};` literal, back out of the rendered file.

    `</` is written `<\\/` on the way in (a node named `</script>` would otherwise end
    the script tag early -- see `render_grouped_dsm_html`); undone here so the text is
    JSON again.
    """
    data = re.search(r"^const D = (.*);$", html, re.MULTILINE).group(1)
    return data.replace("<\\/", "</")


def _tooltips(html: str):
    """The page's own `nodeTip`/`cellTip`, evaluated over the page's own `D`."""
    quickjs = pytest.importorskip("quickjs")
    data = _payload(html)
    esc = re.search(r"^const esc = .*$", html, re.MULTILINE).group(0)
    body = html.split("/* TOOLTIP-TEXT-BEGIN */", maxsplit=1)[1].split(
        "/* TOOLTIP-TEXT-END */", maxsplit=1
    )[0]
    ctx = quickjs.Context()
    ctx.eval(f"const D = {data};\n{esc}\n{body}")
    return (
        lambda i: ctx.eval(f"nodeTip({i})"),
        lambda cell: ctx.eval(f"cellTip({json.dumps(cell)})"),
        json.loads(data),
    )


@pytest.fixture
def page(fixture_graph, order):
    return str(
        render_grouped_dsm_html(
            Blocking.scc(fixture_graph),
            order=order,
            title="tooltips",
            formatter=SPELLING,
        )
    )


def test_a_diagonal_hover_shows_the_node_s_io_not_its_name_twice(page):
    node_tip, _, _ = _tooltips(page)
    assert node_tip(0) == (
        "<b>.g.lean</b>\n"
        '<span class="dim">g &middot; row/col 0</span>\n'
        '<span class="h">reads:</span> 1\n'
        "  .ext.seed\n"
        '<span class="h">writes:</span> 1\n'
        "  .g.lean_out"
    )


def test_a_diagonal_hover_with_many_ports_stops_and_says_how_many_are_left(page):
    node_tip, _, _ = _tooltips(page)
    tip = node_tip(1)
    listed = [ln for ln in tip.split("\n") if ln.startswith("  .")]
    assert len(listed) == TIP_VARS + 2  # capped reads, plus both writes
    assert f'  <span class="dim">... {TIP_VARS + 7} more</span>' in tip
    assert tip.startswith(
        "<b>.g.wide</b>\n"
        '<span class="dim">g &middot; row/col 1</span>\n'
        f'<span class="h">reads:</span> {2 * TIP_VARS + 7}\n'
        "  .g.lean_out\n"
        "  .ext.in00\n"
    )
    assert tip.endswith('<span class="h">writes:</span> 2\n  .g.wide_a\n  .g.wide_b')


def test_a_node_with_no_reads_says_none_rather_than_drawing_an_empty_block():
    """A source node -- every `initialisation.*` node in the real graph is one."""
    graph = Graph(path_map({N("g", "src"): call([], [V("g", "out")])}))
    page = str(
        render_grouped_dsm_html(
            Blocking.scc(graph), order=(N("g", "src"),), formatter=SPELLING
        )
    )
    node_tip, _, _ = _tooltips(page)
    assert node_tip(0) == (
        "<b>.g.src</b>\n"
        '<span class="dim">g &middot; row/col 0</span>\n'
        '<span class="h">reads:</span> <span class="dim">none</span>\n'
        '<span class="h">writes:</span> 1\n'
        "  .g.out"
    )


def test_a_forward_cell_with_one_variable_leads_with_that_variable(page):
    _, cell_tip, data = _tooltips(page)
    cell = next(c for c in data["cells"] if c["r"] == 0 and c["c"] == 1)
    assert cell_tip(cell) == (
        "<b>.g.lean_out</b>\n"
        '<span class="dim">feeds forwards</span>\n'
        '<span class="dim">  from  .g.lean\n  to    .g.wide</span>'
    )


def test_a_backward_cell_leads_with_the_variables_and_keeps_the_feedback_mark(page):
    _, cell_tip, data = _tooltips(page)
    cell = next(c for c in data["cells"] if c["r"] == 2 and c["c"] == 1)
    tip = cell_tip(cell)
    assert tip.startswith(f"<b>{TIP_VARS + 4} variables</b>\n  .h.back00\n  .h.back01\n")
    assert tip.endswith(
        '  <span class="dim">... 4 more</span>\n'
        '<span class="fb">&#8595; feeds backwards</span>\n'
        '<span class="dim">  from  .h.sink\n  to    .g.wide</span>'
    )
    # The node names are still there, but neither is on the first line any more.
    assert not tip.split("\n")[0].endswith("</b>") or "variables" in tip.split("\n")[0]


# =========================================== what must not have regressed
def test_the_block_branch_and_the_crosshair_are_untouched(page):
    """Neither is reachable from `_tooltips`; both are pinned as source, deliberately."""
    assert "if (d.box) {" in page
    assert "block of ${d.box.size} (${d.box.real} not minted)" in page
    assert "contained in: ${esc(d.box.container)}" in page
    # The crosshair still tracks the pointer over the matrix, diagonal included.
    assert "cx.setAttribute('x', X0 + j * CELL)" in page
    assert "cy.setAttribute('y', Y0 + i * CELL)" in page
    # The generic two-name readout survives for a *blank* cell off the diagonal.
    assert "i === j ? nodeTip(i)" in page


def test_the_page_is_still_self_contained(page):
    """Nothing the new tooltip needs is fetched -- no CDN, no font, no external asset.

    The one absolute URL in the file is the SVG namespace, which is an identifier and
    never dereferenced; asserting that is the *only* one is the check, since a blanket
    "no http" would have to be relaxed for it and then means nothing.
    """
    for external in ("<script src", "<link ", "@import", "url(http", "srcset"):
        assert external not in page
    assert re.findall(r"https?://\S*", page) == ["http://www.w3.org/2000/svg';"]


# ======================================= the four published pages carry the payload
@pytest.mark.parametrize("machine", [None, "tokamak"])
def test_the_uncut_pages_carry_per_node_ports(monkeypatch, tmp_path, machine):
    """End to end through `render_xdsm.grouped_uncut`, the entry point that writes them.

    Written to `tmp_path`; this test does not touch the committed pages. What it asserts
    is only that the real payload reaches the real page -- every row has the four new
    keys, and the widest node in PROCESS's graph is genuinely wider than the cap, which
    is what makes the cap load-bearing rather than decorative.
    """
    from functional_process.cottax import render_xdsm
    from functional_process.cottax.boundary import TOKAMAK_INPUT_FILE

    monkeypatch.setattr(render_xdsm, "OUTDIR", tmp_path)
    render_xdsm.grouped_uncut(
        input_file=TOKAMAK_INPUT_FILE if machine == "tokamak" else None
    )
    suffix = "_tokamak" if machine == "tokamak" else ""
    for stem in (f"dsm_provenance_uncut{suffix}", f"dsm_scc_uncut{suffix}"):
        html = (tmp_path / f"{stem}.html").read_text(encoding="utf-8")
        rows = json.loads(_payload(html))["rows"]
        assert rows, stem
        assert all({"reads", "nr", "writes", "nw"} <= set(r) for r in rows), stem
        # Every node owns at least one variable (cottax refuses otherwise), so a row with
        # no writes would mean the payload lost them somewhere.
        assert all(r["nw"] >= 1 for r in rows), stem
        assert max(r["nr"] for r in rows) > TIP_VARS, stem
        assert all(len(r["reads"]) <= TIP_VARS for r in rows), stem
