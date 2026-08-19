'''
Grouping a graph by the prefix of its names, and the two orderings drawn from it.

What is tested is the *rule*, not the drawing: which keys of a name are a group and which
are the node, what a minted name inherits, what an ordering is allowed to claim, and that
the two headline figures (a group scattered across the schedule, a block spanning groups)
come out of a graph built to have exactly one of each. The SVG the page paints is left to
be looked at -- there is no browser here, and asserting on markup would pin the drawing
rather than the reading.
'''

import json
import re

import pytest
from jax.tree_util import DictKey, GetAttrKey

from cottax.blocking import Blocking
from cottax.graph import Graph
from cottax.interfaces.spelling import xDSMFormatterFlat
from cottax.spec import CallableNode, In, NodePath, Out, VarPath
from cottax.tools.minting import MintKey
from cottax.tools.path import path_map
from functional_process.visualization.grouping import (PALETTE, TIER_OVERLAY, UNGROUPED, _matrix_struct,
                                           group_label, group_of, group_sequence,
                                           group_style, grouping_report, provenance_order,
                                           render_grouped_dsm_html, structure_order)


def V(*keys) -> VarPath:
    return VarPath(tuple(GetAttrKey(k) for k in keys))


def N(*keys) -> NodePath:
    return NodePath(tuple(DictKey(k) for k in keys))


def M(ns: str, *keys) -> NodePath:
    '''A name minted in `ns` over a model-tree position.'''
    return NodePath((MintKey(ns), *(DictKey(k) for k in keys)))


def call(reads, owns):
    return CallableNode(
        inputs=tuple(In(r) for r in reads),
        outputs=tuple(Out(o) for o in owns),
        fn=lambda *a: None,
    )


# ============================================================== reading the prefix
def test_the_node_s_own_key_is_never_its_group():
    '''A flat name has no group: `['Build']` says nothing about who owns `Build`.'''
    assert group_of(N('Build')) == UNGROUPED
    assert group_label(UNGROUPED) == '(ungrouped)'


def test_depth_selects_the_grain():
    name = N('stellarator', 'coils', 'Intersect')
    assert group_of(name, depth=1) == ('stellarator',)
    assert group_of(name, depth=2) == ('stellarator', 'coils')
    assert group_label(group_of(name, depth=2)) == 'stellarator.coils'


def test_a_name_shallower_than_the_depth_gives_what_it_has():
    '''Two- and three-level names group together without either being special-cased.'''
    assert group_of(N('physics', 'DensityProfile'), depth=2) == ('physics',)


def test_a_minted_name_inherits_the_group_of_what_it_was_minted_over():
    assert group_of(M('problem', 'stellarator', 'coils', 'Intersect')) == ('stellarator',)


def test_a_name_minted_over_a_variable_place_is_ungrouped():
    '''
    `^problem.physics.proton_rate_density` is minted over a `VarPath`, not over a node.

    Its second key spells `.physics`, which is a place in the caller's data structure and
    not a position in a model tree -- reading it as a group would file the node under a
    group that does not exist. Asked by *kind*, so this holds for every namespace at once.
    '''
    minted = NodePath((MintKey('problem'), GetAttrKey('physics'),
                       GetAttrKey('proton_rate_density')))
    assert group_of(minted) == UNGROUPED


def test_depth_must_be_at_least_one():
    with pytest.raises(ValueError, match='at least 1'):
        group_of(N('a', 'b'), depth=0)


# ============================================================== the orderings
@pytest.fixture
def coupled():
    '''
    Three groups, one cycle across two of them, and a chain that interleaves the others.

    `a.p -> b.q -> a.r -> a.p` is a genuine cross-group SCC. After it the chain
    `c.s -> b.u -> c.t` forces the run order to alternate `c`, `b`, `c`, so `c` is a group
    that is not a schedulable unit -- which is the other half of what the comparison is
    for. Both facts are in one graph deliberately: a picture that surfaces only one of
    them has missed.
    '''
    return Graph(path_map({
        N('a', 'p'): call([V('r')], [V('p')]),
        N('b', 'q'): call([V('p')], [V('q')]),
        N('a', 'r'): call([V('q')], [V('r')]),
        N('c', 's'): call([V('p')], [V('s')]),
        N('b', 'u'): call([V('s')], [V('u')]),
        N('c', 't'): call([V('u')], [V('t')]),
    }))


def test_group_sequence_is_first_appearance_not_alphabetical(coupled):
    assert group_sequence(coupled.nodes) == (('a',), ('b',), ('c',))


def test_provenance_order_makes_every_group_adjacent(coupled):
    order = provenance_order(coupled.nodes)
    assert [group_of(n) for n in order] == [('a',), ('a',), ('b',), ('b',), ('c',), ('c',)]
    assert set(order) == set(coupled.nodes)


def test_provenance_order_keeps_the_input_order_inside_a_group(coupled):
    order = provenance_order(coupled.nodes)
    assert order[:2] == (N('a', 'p'), N('a', 'r'))


def test_provenance_order_takes_a_declared_group_order(coupled):
    order = provenance_order(coupled.nodes, groups=[('c',), ('b',), ('a',)])
    assert [group_of(n) for n in order][0] == ('c',)


def test_provenance_order_refuses_a_group_order_that_misses_a_group(coupled):
    with pytest.raises(KeyError, match="'a'"):
        provenance_order(coupled.nodes, groups=[('b',), ('c',)])


def test_structure_order_is_the_blocking_s_own(coupled):
    blocking = Blocking.scc(coupled)
    assert structure_order(blocking) == tuple(
        name for block in blocking.blocks for name in block)


# ============================================================== the measurement
def test_the_report_finds_the_crossing_block_and_the_scattered_group(coupled):
    report = grouping_report(Blocking.scc(coupled))
    (block,) = report.coupled
    assert set(block.members) == {N('a', 'p'), N('b', 'q'), N('a', 'r')}
    assert block.real == 3 and block.crosses
    assert report.crossing == (block,)
    # not one group of the three is a contiguous stretch of the run order: `a` and `b` are
    # interleaved inside the block that couples them, and `c` straddles `b.u` after it.
    assert report.runs == {('a',): 2, ('b',): 2, ('c',): 2}
    assert report.cross_group_edges == 5


def test_a_minted_problem_beside_its_node_is_not_coupling():
    '''
    A node and the problem minted over it are two members and one *real* one.

    Counting the pair as coupling would report every declared `FixedPoint` in a graph as a
    feedback loop, which is § 11's reason for saying "SCCs with more than one real node".
    '''
    graph = Graph(path_map({
        N('g', 'x'): call([V('u')], [V('c')]),
        M('problem', 'g', 'x'): call([V('c')], [V('u')]),
    }))
    (block,) = grouping_report(Blocking.scc(graph)).blocks
    assert len(block.members) == 2 and block.real == 1
    assert grouping_report(Blocking.scc(graph)).coupled == ()


def test_an_ungrouped_member_does_not_make_a_block_cross():
    graph = Graph(path_map({
        N('g', 'x'): call([V('u')], [V('c')]),
        N('g', 'y'): call([V('c')], [V('d')]),
        N('loose'): call([V('d')], [V('u')]),
    }))
    (block,) = grouping_report(Blocking.scc(graph)).coupled
    assert block.real == 3
    assert not block.crosses and UNGROUPED in block.groups


# ============================================================== the drawing
def test_colours_recycle_under_a_texture_rather_than_silently(coupled):
    '''More groups than colours must not make two of them look the same.'''
    first, later = group_style(0), group_style(len(PALETTE))
    assert first[0] == later[0] and first[1] is None and later[1] == TIER_OVERLAY[1]


def test_the_struct_puts_a_read_in_its_reader_s_row(coupled):
    '''Inputs in rows: `c.s` reads `p` from `a.p`, so the mark is (row c.s, col a.p).'''
    blocking = Blocking.scc(coupled)
    order = structure_order(blocking)
    struct = _matrix_struct(blocking, order, depth=1, formatter=xDSMFormatterFlat())
    at = {name: i for i, name in enumerate(order)}
    (cell,) = [c for c in struct['cells']
               if c['r'] == at[N('c', 's')] and c['c'] == at[N('a', 'p')]]
    assert cell['v'] == ['.p']


def test_the_struct_is_a_permutation_of_the_whole_graph(coupled):
    blocking = Blocking.scc(coupled)
    with pytest.raises(ValueError, match='permutation'):
        _matrix_struct(blocking, blocking.graph.nodes[:2], depth=1,
                       formatter=xDSMFormatterFlat())


def test_the_page_is_self_contained_and_carries_its_own_data(coupled, tmp_path):
    doc = render_grouped_dsm_html(Blocking.scc(coupled), file_name='g',
                                  outdir=str(tmp_path), write=True,
                                  formatter=xDSMFormatterFlat())
    page = str(doc)
    assert doc.path == str(tmp_path / 'g.html')
    assert '//' not in re.sub(r'https?://', '', page)  # no protocol-relative asset
    assert 'src=' not in page and '<link' not in page
    data = json.loads(re.search(r'const D = (\{.*?\});\n', page, re.S).group(1))
    assert len(data['rows']) == len(coupled.nodes)
    assert data['backward'] >= 1                       # the cycle has to run backwards


def test_the_two_orderings_differ_only_in_their_rows(coupled):
    '''The comparison is only a comparison if everything but the order is held fixed.'''
    blocking = Blocking.scc(coupled)
    fmt = xDSMFormatterFlat()
    by_structure = _matrix_struct(blocking, structure_order(blocking), depth=1, formatter=fmt)
    by_provenance = _matrix_struct(blocking, provenance_order(coupled.nodes), depth=1,
                                   formatter=fmt)
    assert by_structure['legend'] == by_provenance['legend']
    assert by_structure['reads'] == by_provenance['reads']
    assert {r['name'] for r in by_structure['rows']} == {r['name'] for r in by_provenance['rows']}
