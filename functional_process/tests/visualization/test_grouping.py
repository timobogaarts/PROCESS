"""
Grouping a graph by the prefix of its names, and the two orderings drawn from it.

What is tested is the *rule*, not the drawing: which keys of a name are a group and which
are the node, what a minted name inherits, what an ordering is allowed to claim, and that
the two headline figures (a group scattered across the schedule, a block spanning groups)
come out of a graph built to have exactly one of each. The SVG the page paints is left to
be looked at -- there is no browser here, and asserting on markup would pin the drawing
rather than the reading.
"""

import json
import re

import pytest
from jax.tree_util import DictKey, GetAttrKey

from cottax.blocking import Blocking
from cottax.graph import Graph
from cottax.interfaces.spelling import xDSMFormatterFlat
from cottax.spec import ImplementedFunction, In, NodePath, Out, VarPath
from cottax.tools.minting import MintKey
from cottax.tools.path import path_map
from functional_process.cottax.visualization.grouping import (
    PALETTE,
    TIER_OVERLAY,
    UNGROUPED,
    _matrix_struct,
    containing,
    dependency_group_sequence,
    group_label,
    group_of,
    group_sequence,
    group_palette,
    group_style,
    grouping_report,
    hierarchical,
    provenance_order,
    render_grouped_dsm_html,
    shade,
    structure_order,
    top_of,
)


def V(*keys) -> VarPath:
    return VarPath(tuple(GetAttrKey(k) for k in keys))


def N(*keys) -> NodePath:
    return NodePath(tuple(DictKey(k) for k in keys))


def M(ns: str, *keys) -> NodePath:
    """A name minted in `ns` over a model-tree position."""
    return NodePath((MintKey(ns), *(DictKey(k) for k in keys)))


def call(reads, owns):
    return ImplementedFunction(
        inputs=tuple(In(r) for r in reads),
        outputs=tuple(Out(o) for o in owns),
        fn=lambda *a: None,
    )


# ============================================================== reading the prefix
def test_the_node_s_own_key_is_never_its_group():
    """A flat name has no group: `['Build']` says nothing about who owns `Build`."""
    assert group_of(N("Build")) == UNGROUPED
    assert group_label(UNGROUPED) == "(ungrouped)"


def test_the_grain_is_the_tree_s_own_by_default():
    """No argument: a node's group is the namespace it lives in, however deep that is."""
    assert group_of(N("stellarator", "coils", "Intersect")) == ("stellarator", "coils")
    assert group_of(N("costs", "Acc22")) == ("costs",)
    assert group_of(N("physics", "profiles", "parameterisation", "OnAxis")) == (
        "physics",
        "profiles",
        "parameterisation",
    )


def test_depth_selects_the_grain_and_only_ever_truncates():
    name = N("stellarator", "coils", "Intersect")
    assert group_of(name, depth=1) == ("stellarator",)
    assert group_of(name, depth=2) == ("stellarator", "coils")
    assert group_label(group_of(name, depth=2)) == "stellarator.coils"
    # It saturates at the node's own parent, so past the tree's depth it stops moving --
    # which is why an integer was never the grain, only a zoom on it.
    assert group_of(name, depth=99) == group_of(name) == ("stellarator", "coils")


def test_a_name_shallower_than_the_depth_gives_what_it_has():
    """Two- and three-level names group together without either being special-cased."""
    assert group_of(N("physics", "DensityProfile"), depth=2) == ("physics",)


def test_a_minted_name_inherits_the_group_of_what_it_was_minted_over():
    """At the tree's grain that is the node's own namespace, not its subsystem: a
    problem drawn beside `stellarator.coils.Intersect` belongs in `stellarator.coils`."""
    minted = M("problem", "stellarator", "coils", "Intersect")
    assert group_of(minted) == ("stellarator", "coils")
    assert group_of(minted, depth=1) == ("stellarator",)


def test_a_name_minted_over_a_variable_place_is_ungrouped():
    """
    `^problem.physics.proton_rate_density` is minted over a `VarPath`, not over a node.

    Its second key spells `.physics`, which is a place in the caller's data structure and
    not a position in a model tree -- reading it as a group would file the node under a
    group that does not exist.

    **Asked of the graph, not of the key kind.** This used to rely on `GetAttrKey`
    meaning "a variable place"; `model_tree_design.md` §8 step 3 made a `GetAttrKey` the
    spelling of a machine *slot* too, so the kind cannot separate the two any more. What
    can is whether the unminted name is itself a node -- which is what `among` asks.
    """
    minted = NodePath((
        MintKey("problem"),
        GetAttrKey("physics"),
        GetAttrKey("proton_rate_density"),
    ))
    real = NodePath((GetAttrKey("physics"), GetAttrKey("density_profile")))

    assert group_of(minted, among=[real]) == UNGROUPED
    assert group_of(minted, among=[]) == UNGROUPED
    # The other half of the rule: a name minted over a node that *is* there keeps its
    # group, so a block's problem sits beside its block.
    assert group_of(NodePath((MintKey("problem"), *real.keys)), among=[real]) == (
        "physics",
    )
    # Nothing to ask: the prefix is read as written. Pinned so the fallback is
    # deliberate rather than discovered.
    assert group_of(minted) == ("physics",)


def test_a_name_minted_over_a_variable_place_falls_back_to_its_owner_s_group():
    """
    `group_of`'s other half: when unminting a problem's name does not land on a node,
    `owners` lets it land on the group of whoever *does* compute the variable it was
    minted over, instead of falling straight to `UNGROUPED`.

    Mirrors the two `FixedPointCut` problems the reference driven graph actually has --
    `^problem.fwbs.f_ster_div_single` (cut over one variable, named after its own place,
    so the direct lookup in `owners` hits immediately) and
    `^problem.physics.proton_rate_density.cycle` (cut over two, named after the first
    plus one extra `.cycle` key, so `_cut_owner` has to trim before it hits) -- and
    checks the phantom `fwbs`/one-off `physics` group the guard exists to prevent does
    not appear in `group_sequence` either.
    """
    divertor = NodePath((GetAttrKey("stellarator"), GetAttrKey("divertor")))
    fusion_rates = NodePath((GetAttrKey("physics"), GetAttrKey("fusion_rates")))
    fusion_totals = NodePath((
        GetAttrKey("physics"),
        GetAttrKey("fusion_totals_no_beam"),
    ))
    f_ster = V("fwbs", "f_ster_div_single")
    proton = V("physics", "proton_rate_density")
    fusden = V("physics", "fusden_alpha_total")

    single = NodePath((
        MintKey("problem"),
        GetAttrKey("fwbs"),
        GetAttrKey("f_ster_div_single"),
    ))
    cyclic = NodePath((
        MintKey("problem"),
        GetAttrKey("physics"),
        GetAttrKey("proton_rate_density"),
        GetAttrKey("cycle"),
    ))

    graph = Graph(
        path_map({
            divertor: call([], [f_ster]),
            fusion_rates: call([], [proton]),
            fusion_totals: call([], [fusden]),
            single: call([f_ster], [V("hat", "fwbs", "f_ster_div_single")]),
            cyclic: call(
                [proton, fusden],
                [
                    V("hat", "physics", "proton_rate_density"),
                    V("hat", "physics", "fusden_alpha_total"),
                ],
            ),
        })
    )
    owners = graph.owners
    among = frozenset(graph.nodes)

    assert group_of(single, among=among, owners=owners) == ("stellarator",)
    assert group_of(cyclic, among=among, owners=owners) == ("physics",)
    # Without `owners` the guard still holds: no `graph` to consult means no fallback,
    # and the honest answer stays `UNGROUPED` rather than inventing a phantom group
    # (`fwbs`, or a one-off `physics.proton_rate_density`) from the minted name itself.
    assert group_of(single, among=among) == UNGROUPED
    assert group_of(cyclic, among=among) == UNGROUPED

    groups = group_sequence(graph.nodes, owners=owners)
    assert ("fwbs",) not in groups
    assert set(groups) == {("stellarator",), ("physics",)}


def test_depth_must_be_at_least_one():
    with pytest.raises(ValueError, match="at least 1"):
        group_of(N("a", "b"), depth=0)


# ============================================================== the orderings
@pytest.fixture
def coupled():
    """
    Three groups, one cycle across two of them, and a chain that interleaves the others.

    `a.p -> b.q -> a.r -> a.p` is a genuine cross-group SCC. After it the chain
    `c.s -> b.u -> c.t` forces the run order to alternate `c`, `b`, `c`, so `c` is a group
    that is not a schedulable unit -- which is the other half of what the comparison is
    for. Both facts are in one graph deliberately: a picture that surfaces only one of
    them has missed.
    """
    return Graph(
        path_map({
            N("a", "p"): call([V("r")], [V("p")]),
            N("b", "q"): call([V("p")], [V("q")]),
            N("a", "r"): call([V("q")], [V("r")]),
            N("c", "s"): call([V("p")], [V("s")]),
            N("b", "u"): call([V("s")], [V("u")]),
            N("c", "t"): call([V("u")], [V("t")]),
        })
    )


def test_group_sequence_is_first_appearance_not_alphabetical(coupled):
    assert group_sequence(coupled.nodes) == (("a",), ("b",), ("c",))


def test_provenance_order_makes_every_group_adjacent(coupled):
    order = provenance_order(coupled.nodes)
    assert [group_of(n) for n in order] == [
        ("a",),
        ("a",),
        ("b",),
        ("b",),
        ("c",),
        ("c",),
    ]
    assert set(order) == set(coupled.nodes)


def test_provenance_order_keeps_the_input_order_inside_a_group(coupled):
    order = provenance_order(coupled.nodes)
    assert order[:2] == (N("a", "p"), N("a", "r"))


def test_provenance_order_takes_a_declared_group_order(coupled):
    order = provenance_order(coupled.nodes, groups=[("c",), ("b",), ("a",)])
    assert [group_of(n) for n in order][0] == ("c",)


def test_provenance_order_refuses_a_group_order_that_misses_a_group(coupled):
    with pytest.raises(KeyError, match="'a'"):
        provenance_order(coupled.nodes, groups=[("b",), ("c",)])


def test_structure_order_is_the_blocking_s_own(coupled):
    blocking = Blocking.scc(coupled)
    assert structure_order(blocking) == tuple(
        name for block in blocking.blocks for name in block
    )


# =========================================================== the dependency group axis
_LAYERED = {
    N("z", "total"): call([V("m"), V("n")], [V("z")]),
    N("a", "src"): call([], [V("a")]),
    N("n", "y"): call([V("a")], [V("n")]),
    N("m", "x"): call([V("a")], [V("m")]),
}
"""
Four groups in a chain with the **sink declared first** -- the shape of the complaint.

`z` reads from `m` and `n` and nothing in the graph reads from `z`, yet it is bound
first, so `group_sequence` puts it at the top left of the matrix and the picture reads as
"everything depends on `z`". That is `costs` in the port's own graph, in miniature. `m`
and `n` are both immediately downstream of `a` and independent of each other, so they are
also the tie that the fallback to declaration order has to break.
"""


@pytest.fixture
def layered():
    return Graph(path_map(dict(_LAYERED)))


def test_the_dependency_axis_puts_a_sink_group_last(layered):
    """The complaint, and the fix, in one assertion pair."""
    assert group_sequence(layered.nodes) == (("z",), ("a",), ("n",), ("m",))
    assert dependency_group_sequence(layered) == (("a",), ("n",), ("m",), ("z",))


def test_the_dependency_axis_is_a_topological_order_of_the_group_graph(layered):
    """
    Every cross-group edge points forward, which is the whole claim the axis makes.

    Checked on a fixture with no group-level cycle, so "forward" is unconditional here;
    the coupled case is `test_mutually_dependent_groups_collapse_into_one_scc`, where
    there is no forward to point in.
    """
    order = dependency_group_sequence(layered)
    assert set(order) == set(group_sequence(layered.nodes))
    rank = {g: i for i, g in enumerate(order)}
    owners = layered.owners
    for name in layered.nodes:
        for var in layered[name].reads:
            source = owners.get(var)
            if source is None or group_of(source) == group_of(name):
                continue
            assert rank[group_of(source)] < rank[group_of(name)], (
                f"{group_label(group_of(source))} -> {group_label(group_of(name))} "
                f"runs backwards in {[group_label(g) for g in order]}"
            )


def test_the_tie_break_is_declaration_order_and_is_stable(layered):
    """
    `m` and `n` are at the same level; declaration order decides, and keeps deciding.

    Both halves matter because these diagrams are regenerated and eyeballed against the
    previous version: an axis that answered differently on a second call, or shuffled
    when an unrelated node joined a group, would make every re-render unreadable. The
    added node reads `a` and is in `m`, so it adds no group edge that was not there.
    """
    assert dependency_group_sequence(layered) == dependency_group_sequence(layered)
    grown = Graph(path_map({**_LAYERED, N("m", "extra"): call([V("a")], [V("extra")])}))
    assert dependency_group_sequence(grown) == dependency_group_sequence(layered)


def test_mutually_dependent_groups_collapse_into_one_scc():
    """
    `a` and `b` feed each other, so there is no order between them: emit both, adjacent.

    The point is that this does not raise. A topological sort of the *contracted* graph
    has to survive a cycle in it -- two subsystems that genuinely exchange values are the
    normal case, not an error -- and the honest answer is "adjacent, in declaration
    order", which is what condensing before sorting gives. The whole SCC still sorts
    ahead of the sink `z` that reads it.
    """
    mutual = Graph(
        path_map({
            N("z", "sink"): call([V("q")], [V("z")]),
            N("a", "p"): call([V("r")], [V("p")]),
            N("b", "q"): call([V("p")], [V("q")]),
            N("a", "r"): call([V("q")], [V("r")]),
        })
    )
    assert dependency_group_sequence(mutual) == (("a",), ("b",), ("z",))


def test_a_wholly_coupled_group_graph_falls_back_to_declaration_order(coupled):
    """`a <-> b <-> c`: one SCC, nothing to order, so the axis is `group_sequence`."""
    assert dependency_group_sequence(coupled) == group_sequence(coupled.nodes)


def test_provenance_order_takes_the_dependency_axis(layered):
    """
    How `render_xdsm.grouped` wires the two together -- through the existing `groups=`.

    `provenance_order`'s signature does not change: the dependency axis is just another
    group order to hand it, and rows *inside* a group are still in declaration order,
    which is where the provenance reading is actually about how the file was written.
    """
    order = provenance_order(layered.nodes, groups=dependency_group_sequence(layered))
    assert [group_of(n) for n in order] == [("a",), ("n",), ("m",), ("z",)]
    assert set(order) == set(layered.nodes)


# ============================================================== the measurement
def test_the_report_finds_the_crossing_block_and_the_scattered_group(coupled):
    report = grouping_report(Blocking.scc(coupled))
    (block,) = report.coupled
    assert set(block.members) == {N("a", "p"), N("b", "q"), N("a", "r")}
    assert block.real == 3 and block.crosses
    assert report.crossing == (block,)
    # not one group of the three is a contiguous stretch of the run order: `a` and `b` are
    # interleaved inside the block that couples them, and `c` straddles `b.u` after it.
    assert report.runs == {("a",): 2, ("b",): 2, ("c",): 2}
    assert report.cross_group_edges == 5


def test_a_minted_problem_beside_its_node_is_not_coupling():
    """
    A node and the problem minted over it are two members and one *real* one.

    Counting the pair as coupling would report every declared `FixedPoint` in a graph as a
    feedback loop, which is § 11's reason for saying "SCCs with more than one real node".
    """
    graph = Graph(
        path_map({
            N("g", "x"): call([V("u")], [V("c")]),
            M("problem", "g", "x"): call([V("c")], [V("u")]),
        })
    )
    (block,) = grouping_report(Blocking.scc(graph)).blocks
    assert len(block.members) == 2 and block.real == 1
    assert grouping_report(Blocking.scc(graph)).coupled == ()


def test_an_ungrouped_member_does_not_make_a_block_cross():
    graph = Graph(
        path_map({
            N("g", "x"): call([V("u")], [V("c")]),
            N("g", "y"): call([V("c")], [V("d")]),
            N("loose"): call([V("d")], [V("u")]),
        })
    )
    (block,) = grouping_report(Blocking.scc(graph)).coupled
    assert block.real == 3
    assert not block.crosses and UNGROUPED in block.groups


# ============================================================== the drawing
def test_colours_recycle_under_a_texture_rather_than_silently(coupled):
    """More groups than colours must not make two of them look the same."""
    first, later = group_style(0), group_style(len(PALETTE))
    assert first[0] == later[0] and first[1] is None and later[1] == TIER_OVERLAY[1]


def test_the_struct_puts_a_read_in_its_producer_s_row(coupled):
    """Outputs in rows (`IC_FBD`): `c.s` reads `p` from `a.p`, so the mark is
    (row a.p, col c.s) -- and feedback therefore falls *below* the diagonal, the same way
    round as the plotly `dsm.html` and as everyone else's DSM."""
    blocking = Blocking.scc(coupled)
    order = structure_order(blocking)
    struct = _matrix_struct(blocking, order, depth=1, formatter=xDSMFormatterFlat())
    at = {name: i for i, name in enumerate(order)}
    (cell,) = [
        c
        for c in struct["cells"]
        if c["r"] == at[N("a", "p")] and c["c"] == at[N("c", "s")]
    ]
    assert cell["v"] == [".p"]


def test_backward_is_the_lower_triangle(coupled):
    """The flip is only real if the count agrees with it: what runs backwards in this
    ordering is what sits below the diagonal, not above."""
    blocking = Blocking.scc(coupled)
    struct = _matrix_struct(
        blocking, structure_order(blocking), depth=1, formatter=xDSMFormatterFlat()
    )
    assert struct["backward"] == sum(c["n"] for c in struct["cells"] if c["r"] > c["c"])
    assert struct["backward"] >= 1  # the cycle has to run backwards somewhere


def test_the_struct_is_a_permutation_of_the_whole_graph(coupled):
    blocking = Blocking.scc(coupled)
    with pytest.raises(ValueError, match="permutation"):
        _matrix_struct(
            blocking, blocking.graph.nodes[:2], depth=1, formatter=xDSMFormatterFlat()
        )


def test_the_page_is_self_contained_and_carries_its_own_data(coupled, tmp_path):
    doc = render_grouped_dsm_html(
        Blocking.scc(coupled),
        file_name="g",
        outdir=str(tmp_path),
        write=True,
        formatter=xDSMFormatterFlat(),
    )
    page = str(doc)
    assert doc.path == str(tmp_path / "g.html")
    assert "//" not in re.sub(r"https?://", "", page)  # no protocol-relative asset
    assert "src=" not in page and "<link" not in page
    data = json.loads(re.search(r"const D = (\{.*?\});\n", page, re.S).group(1))
    assert len(data["rows"]) == len(coupled.nodes)
    assert data["backward"] >= 1  # the cycle has to run backwards


def test_the_two_orderings_differ_only_in_their_rows(coupled):
    """The comparison is only a comparison if everything but the order is held fixed."""
    blocking = Blocking.scc(coupled)
    fmt = xDSMFormatterFlat()
    by_structure = _matrix_struct(
        blocking, structure_order(blocking), depth=1, formatter=fmt
    )
    by_provenance = _matrix_struct(
        blocking, provenance_order(coupled.nodes), depth=1, formatter=fmt
    )
    assert by_structure["legend"] == by_provenance["legend"]
    assert by_structure["reads"] == by_provenance["reads"]
    assert {r["name"] for r in by_structure["rows"]} == {
        r["name"] for r in by_provenance["rows"]
    }


# ====================================================== containment, not distinctness
def test_containing_is_the_longest_common_prefix():
    assert containing([("a", "b"), ("a", "b", "c")]) == ("a", "b")
    assert containing([("a", "b"), ("a", "c")]) == ("a",)
    assert containing([("a",), ("b",)]) == UNGROUPED
    assert containing([]) == UNGROUPED


def test_containing_ignores_ungrouped_members():
    """A flat name says nothing about containment, so it must not drag it to the root --
    the same reading `named_groups` takes, and what keeps
    `test_an_ungrouped_member_does_not_make_a_block_cross` true."""
    assert containing([("a", "b"), UNGROUPED]) == ("a", "b")


def test_top_of_is_the_subsystem():
    assert top_of(("stellarator", "coils")) == ("stellarator",)
    assert top_of(UNGROUPED) == UNGROUPED


@pytest.fixture
def nested():
    """
    A loop spanning `p` and `p.q` -- two groups, one subtree, and *not* a crossing.

    The shape the reference machine actually has: its one multi-group SCC spans
    `physics`, `physics.profiles` and `physics.profiles.parameterisation`. Reported as a
    cross-group loop it reads as two subsystems coupling, which is false; reported at
    `depth=1` it vanishes entirely, which is also false.
    """
    return Graph(
        path_map({
            N("p", "x"): call([V("c")], [V("a")]),
            N("p", "q", "y"): call([V("a")], [V("b")]),
            N("p", "q", "r", "z"): call([V("b")], [V("c")]),
            N("t", "w"): call([V("a")], [V("w")]),
        })
    )


def test_a_block_inside_one_subtree_does_not_cross(nested):
    (block,) = grouping_report(Blocking.scc(nested)).coupled
    assert block.real == 3
    assert block.spans and block.nests and not block.crosses
    assert block.container == ("p",)
    assert grouping_report(Blocking.scc(nested)).crossing == ()
    assert grouping_report(Blocking.scc(nested)).nesting == (block,)


def test_a_block_across_subsystems_still_crosses(coupled):
    (block,) = grouping_report(Blocking.scc(coupled)).coupled
    assert block.spans and block.crosses and not block.nests
    assert block.container == UNGROUPED


def test_edges_are_counted_twice_over_group_and_over_subsystem(nested):
    """`p.x -> p.q.y` is a cross-group edge and is not two subsystems talking. Both
    figures are reported because at the tree's grain the first alone misleads."""
    report = grouping_report(Blocking.scc(nested))
    assert report.cross_group_edges > report.cross_subsystem_edges
    assert report.cross_subsystem_edges == 1  # only `p.x -> t.w`


def test_the_summary_says_which_grain_it_used(nested):
    blocking = Blocking.scc(nested)
    assert "by the tree" in grouping_report(blocking).summary()
    assert "by depth 1" in grouping_report(blocking, depth=1).summary()
    assert grouping_report(blocking).levels == 3


# ============================================================== the nested ribbon
def test_the_ribbon_has_one_lane_per_level(nested):
    blocking = Blocking.scc(nested)
    struct = _matrix_struct(
        blocking, structure_order(blocking), depth=None, formatter=xDSMFormatterFlat()
    )
    assert struct["levels"] == 3
    assert {b["level"] for b in struct["bands"]} == {0, 1, 2}
    outer = [b for b in struct["bands"] if b["level"] == 0]
    assert {b["label"] for b in outer} == {"p", "t"}
    # An inner lane is labelled by its own key, and its full path is kept for the title.
    inner = [b for b in struct["bands"] if b["level"] == 1]
    assert [(b["label"], b["full"]) for b in inner] == [("q", "p.q")]


def test_a_shallow_name_simply_has_no_inner_lane(nested):
    """How a ragged tree draws: `t` has a lane 0 and no lane 1, rather than a padded
    one or a special case."""
    blocking = Blocking.scc(nested)
    struct = _matrix_struct(
        blocking, structure_order(blocking), depth=None, formatter=xDSMFormatterFlat()
    )
    rows = {r["name"]: r for r in struct["rows"]}
    covered = {
        b["level"]
        for b in struct["bands"]
        if b["from"] <= list(rows).index("t.w") <= b["to"]
    }
    assert covered == {0}


def test_hue_is_the_subsystem_s_so_a_subtree_reads_as_one_thing(nested):
    """One hue per subsystem, and depth spent as a *tint* of it: the three namespaces of
    `p` are three shades of one colour, and `t` is a different colour entirely."""
    blocking = Blocking.scc(nested)
    struct = _matrix_struct(
        blocking, structure_order(blocking), depth=None, formatter=xDSMFormatterFlat()
    )
    rows = {r["name"]: r for r in struct["rows"]}
    assert rows["p.x"]["base"] == rows["p.q.y"]["base"] == rows["p.q.r.z"]["base"]
    assert rows["t.w"]["base"] != rows["p.x"]["base"]
    tints = {rows[n]["colour"] for n in ("p.x", "p.q.y", "p.q.r.z")}
    assert len(tints) == 3  # ... and the three are told apart


def test_a_subsystem_keeps_the_undiluted_hue_and_its_children_are_shaded(nested):
    """The namespace itself is the palette colour; what is inside it is shaded away from
    it. A child that came out the same colour as its parent would make the ribbon's inner
    lane invisible."""
    palette = group_palette([("p",), ("p", "q"), ("p", "q", "r"), ("t",)])
    assert palette[("p",)].colour == palette[("p",)].base
    assert palette[("p", "q")].colour != palette[("p",)].colour
    assert palette[("p", "q")].base == palette[("p",)].base


def test_shade_goes_to_white_and_to_black():
    assert shade("#808080", 0.0) == "#808080"
    assert shade("#808080", 1.0) == "#ffffff"
    assert shade("#808080", -1.0) == "#000000"


def test_an_intermediate_namespace_is_coloured_even_with_no_node_of_its_own():
    """The ribbon draws a lane per level, so a namespace nothing lives directly in still
    needs a colour -- otherwise a subsystem has a hole in the middle of it."""
    palette = group_palette([("p", "q", "r")])
    assert set(palette) >= {("p",), ("p", "q"), ("p", "q", "r")}


def test_the_legend_is_hierarchical_and_indented(nested):
    blocking = Blocking.scc(nested)
    struct = _matrix_struct(
        blocking, structure_order(blocking), depth=None, formatter=xDSMFormatterFlat()
    )
    assert [(g["full"], g["level"]) for g in struct["legend"]] == [
        ("p", 0),
        ("p.q", 1),
        ("p.q.r", 2),
        ("t", 0),
    ]


def test_hierarchical_puts_a_namespace_before_what_is_inside_it():
    assert hierarchical([("t",), ("p", "q"), ("p",)]) == (("t",), ("p",), ("p", "q"))
