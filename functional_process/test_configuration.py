"""Every topology switch's arms assemble, and the switch really changes the topology.

The unit tests under `models/` check that a ported *function* matches PROCESS. Nothing
there can catch the failure this module is about: an arm that is ported and tested but
never assembled into any graph, which is exactly the state `LowhybHeating` and
`AFwTotalNoPowerflow` were in before `configuration.py` existed. A node no configuration
selects is dead code that passes its own tests.
"""

import pytest

from functional_process.configuration import Alternative, Configuration, Switch
from functional_process.models.stellarator.density_limits import SudoDensityLimit
from functional_process.models.stellarator.initialization import PulseDurations
from functional_process.total_process import TOPOLOGY_SWITCHES, graph_for


def _ported_values(switch):
    return [a.value for a in switch.alternatives if a.unported is None]


ARMS = [
    pytest.param(switch, value, id=f"{switch.path}={value}")
    for switch in TOPOLOGY_SWITCHES
    for value in _ported_values(switch)
]


@pytest.mark.parametrize(("switch", "value"), ARMS)
def test_every_ported_arm_assembles(switch, value):
    """Each ported arm is reachable from some configuration, and builds a real graph.

    This is the check that keeps a ported-but-unwired node from going unnoticed: if an
    arm's nodes collide with `COMMON` or with another switch's arm, `to_graph` raises
    here rather than the arm being quietly left out of `total_process.py`.
    """
    graph = graph_for(Configuration({switch.path: value}))
    assert graph.definitions, f"{switch.path} == {value} assembled an empty graph"


@pytest.mark.parametrize("switch", TOPOLOGY_SWITCHES, ids=lambda s: s.path)
def test_arms_are_mutually_exclusive(switch):
    """Arms filed under one switch collide on ownership -- otherwise they aren't arms.

    Two nodes that own disjoint outputs could coexist in one graph, so filing them as
    alternatives would wrongly delete one from every configuration.
    """
    switch.check_arms_are_exclusive()


@pytest.mark.parametrize("switch", TOPOLOGY_SWITCHES, ids=lambda s: s.path)
def test_arms_select_different_node_sets(switch):
    """A switch that selects the same nodes whatever its value is not a topology switch.

    It would belong in the other category of `naming_convention.md` § "switches are not
    ports" -- a static kwarg on one node -- not here.
    """
    selected = {
        value: {
            name.path_str()
            for name in graph_for(Configuration({switch.path: value})).nodes
        }
        for value in _ported_values(switch)
    }
    distinct = {frozenset(nodes) for nodes in selected.values()}
    assert len(distinct) == len(selected), (
        f"{switch.path} selects the same node set for two different values: {selected}"
    )


def test_ipowerflow_decides_whether_the_graph_has_a_cycle():
    """The `ipowerflow` arms differ in *reads*, not just formula, and that flips an SCC.

    `AFwTotalWithPowerflow` reads `.fwbs.f_ster_div_single`, which `Divertor` owns, while
    `Divertor` reads `.first_wall.a_fw_total`, which both arms own. So `ipowerflow != 0`
    is genuinely coupled and `ipowerflow == 0` is not.

    Pinned as a test because it is the concrete counterexample to modelling a switch as
    one fused node branching internally: no such node could express "this configuration
    has no cycle", and the whole point of `_audit/next_steps.md` § 5 is to ask exactly
    that question of the real graph.

    Checks for this specific SCC's presence/absence in `.cycles`, not indices into it or
    overall `.is_acyclic` -- the graph also carries several *unconditional* declared
    `FixedPoint` self-loops by now (`EtathLiqStep`, `DeltaEtaStep`, `WindingPackJTfWp`,
    and others -- each a genuine Shape-B single-node loop, not a modelling accident, see
    `_audit/next_steps.md` §5), so both configurations are `not is_acyclic` and the
    number/order of entries in `.cycles` is not stable across unrelated registrations.
    """
    coupled = graph_for(Configuration({".heat_transport.ipowerflow": 1}))
    uncoupled = graph_for(Configuration({".heat_transport.ipowerflow": 0}))

    divertor_cycle = {"['Divertor']", "['AFwTotalWithPowerflow']"}
    assert divertor_cycle in [{n.path_str() for n in c} for c in coupled.cycles]
    assert divertor_cycle not in [{n.path_str() for n in c} for c in uncoupled.cycles]


def test_default_configuration_matches_process_defaults():
    """`GRAPH` is the graph a silent IN.DAT produces, not an arbitrary pick.

    The defaults are cited from `process/data_structure/` in `TOPOLOGY_SWITCHES`; this
    asserts the assembled default agrees with choosing each of them explicitly.
    """
    explicit = Configuration({s.path: s.default for s in TOPOLOGY_SWITCHES})
    assert {n.path_str() for n in graph_for().nodes} == {
        n.path_str() for n in graph_for(explicit).nodes
    }


def test_unported_arm_says_why():
    """Requesting a known-but-unported branch names the blocker instead of silently
    assembling a graph with a hole where that arm's outputs belong."""
    with pytest.raises(NotImplementedError, match="culnbi"):
        graph_for(Configuration({".stellarator.isthtr": 3}))


def test_unknown_switch_value_is_rejected():
    with pytest.raises(ValueError, match="not a known alternative"):
        graph_for(Configuration({".stellarator.isthtr": 99}))


def test_unknown_switch_path_is_rejected():
    """A typo'd key would otherwise be silently ignored, assembling the default graph
    while the caller believes they configured something."""
    with pytest.raises(ValueError, match="unknown switch"):
        graph_for(Configuration({".stellarator.is_thtr": 1}))


def test_non_exclusive_arms_are_rejected():
    """The guard works -- two arms owning nothing in common are not alternatives."""
    bogus = Switch(
        path=".stellarator.made_up",
        default=0,
        alternatives=(
            Alternative(value=0, declarations=(SudoDensityLimit,)),
            Alternative(value=1, declarations=(PulseDurations,)),
        ),
    )
    with pytest.raises(ValueError, match="no output in common"):
        bogus.check_arms_are_exclusive()


def test_alternative_cannot_be_both_ported_and_unported():
    with pytest.raises(ValueError, match="one or the other"):
        Alternative(value=0, declarations=(object(),), unported="reason")


def test_switch_default_must_be_a_declared_alternative():
    with pytest.raises(ValueError, match="not among the declared alternatives"):
        Switch(path=".a.b", default=7, alternatives=(Alternative(value=0),))
