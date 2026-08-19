"""Every topology switch's arms assemble, and the switch really changes the topology.

The unit tests under `models/` check that a ported *function* matches PROCESS. Nothing
there can catch the failure this module is about: an arm that is ported and tested but
never assembled into any graph, which is exactly the state `LowhybHeating` and
`AFwTotalNoPowerflow` were in before `configuration.py` existed. A node no configuration
selects is dead code that passes its own tests.
"""

import pytest

from cottax.interfaces.pytree_namespace_module import spell_flat

from functional_process.configuration import Alternative, Configuration, Switch
from functional_process.models.stellarator.density_limits import SudoDensityLimit
from functional_process.models.stellarator.initialization import PulseDurations
from functional_process.total_process import (
    PROCESS_DEFAULT_CONFIGURATION,
    REFERENCE_CONFIGURATION,
    REFERENCE_INPUT_FILE,
    TOPOLOGY_SWITCHES,
    graph_for,
)


def _ported_values(switch):
    return [a.value for a in switch.alternatives if a.is_ported]


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

    # Spelled with `spell_flat`, not `path_str()`: node names are hierarchical now
    # (`stellarator.Divertor`), and jax's own bracket spelling of a three-key path is
    # unreadable. `AFwTotalWithPowerflow` is still bare because switch arms are not yet
    # placed in the model tree -- see `path_refactor.md` §B.4.
    divertor_cycle = {"stellarator.Divertor", "AFwTotalWithPowerflow"}
    assert divertor_cycle in [{spell_flat(n) for n in c} for c in coupled.cycles]
    assert divertor_cycle not in [{spell_flat(n) for n in c} for c in uncoupled.cycles]


def test_process_default_configuration_matches_process_defaults():
    """`PROCESS_DEFAULT_CONFIGURATION` is the graph a silent IN.DAT produces.

    The defaults are cited from `process/data_structure/` in `TOPOLOGY_SWITCHES`; this
    asserts assembling with no choices agrees with choosing each of them explicitly.

    **This used to assert the same of `graph_for()`'s no-argument form.** That is no
    longer what the bare form means -- see `REFERENCE_CONFIGURATION`'s docstring for the
    five registration bugs that default caused -- but the property itself is unchanged
    and still worth pinning, so it moved onto the configuration that still claims it.
    """
    explicit = Configuration({s.path: s.default for s in TOPOLOGY_SWITCHES})
    assert {n.path_str() for n in graph_for(PROCESS_DEFAULT_CONFIGURATION).nodes} == {
        n.path_str() for n in graph_for(explicit).nodes
    }


def test_switch_defaults_match_process():
    """Every `Switch.default` really is the value `process/data_structure/` declares.

    `Switch.default`'s docstring claims this and each declaration cites a `file:line`,
    but nothing checked it -- the citation could rot silently, and the whole point of
    keeping `PROCESS_DEFAULT_CONFIGURATION` meaningful is that this property holds.
    Read off the live dataclass rather than re-parsing the source, so a renamed or
    retyped field fails here instead of drifting.
    """
    from process.core.model import DataStructure

    data = DataStructure()
    for switch in TOPOLOGY_SWITCHES:
        if "," in switch.path:
            # A comma-joined `path` is a *synthetic* lookup key for a joint dispatch on
            # two PROCESS switches at once, and its `value`/`default` are arm indices,
            # not literal field values (see that `Switch`'s own comment block in
            # `total_process.py`). There is no single field to compare against.
            continue
        area, _, field = switch.path.lstrip(".").partition(".")
        actual = getattr(getattr(data, area), field)
        assert switch.default == actual, (
            f"{switch.path}: declared default {switch.default} but "
            f"DataStructure().{area}.{field} is {actual}"
        )


def test_reference_configuration_matches_the_input_file():
    """`REFERENCE_CONFIGURATION` says what `REFERENCE_INPUT_FILE` actually sets.

    **This is the check that closes the bug class.** Five registration errors in this
    project came from a value copied off PROCESS's bare defaults instead of the run
    being modelled, each found only afterwards by the MDA harness. Parsing the input
    file makes "the assembled graph matches the run it is validated against" a checked
    property rather than a thing someone remembered.

    Both directions are asserted: every switch the file sets explicitly must appear in
    `REFERENCE_CONFIGURATION` with that value (so a newly-`Switch`ed variable the file
    already sets cannot be quietly registered at PROCESS's default), and every entry in
    `REFERENCE_CONFIGURATION` must be one the file really sets (so a stale entry cannot
    outlive the file changing).
    """
    import re
    from pathlib import Path

    text = Path(REFERENCE_INPUT_FILE).read_text()
    in_file = {}
    for line in text.splitlines():
        m = re.match(r"\s*([A-Za-z_]\w*)\s*=\s*(-?\d+)\s*(\*.*)?$", line)
        if m:
            in_file[m.group(1)] = int(m.group(2))

    for switch in TOPOLOGY_SWITCHES:
        if "," in switch.path:
            continue  # synthetic joint-dispatch key -- see the test above
        field = switch.path.rpartition(".")[2]
        declared = REFERENCE_CONFIGURATION.choices.get(switch.path)
        if field in in_file:
            assert declared == in_file[field], (
                f"{REFERENCE_INPUT_FILE} sets {field} = {in_file[field]}, but "
                f"REFERENCE_CONFIGURATION says {declared!r} for {switch.path}"
            )
        else:
            assert declared is None, (
                f"REFERENCE_CONFIGURATION pins {switch.path} = {declared}, but "
                f"{REFERENCE_INPUT_FILE} never sets {field} -- it would fall through "
                f"to Switch.default ({switch.default})"
            )


def test_unported_arm_says_why():
    """Requesting a known-but-unported branch names the blocker instead of silently
    assembling a graph with a hole where that arm's outputs belong.
    """
    with pytest.raises(NotImplementedError, match="culnbi"):
        graph_for(Configuration({".stellarator.isthtr": 3}))


def test_unknown_switch_value_is_rejected():
    with pytest.raises(ValueError, match="not a known alternative"):
        graph_for(Configuration({".stellarator.isthtr": 99}))


def test_unknown_switch_path_is_rejected():
    """A typo'd key would otherwise be silently ignored, assembling the default graph
    while the caller believes they configured something.
    """
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


@pytest.mark.parametrize(
    "kwargs",
    [
        {"declarations": (object(),), "unported": "reason"},
        {"declarations": (object(),), "unproduced": "reason"},
        {"unported": "reason", "unproduced": "reason"},
    ],
)
def test_alternative_is_exactly_one_of_ported_refused_or_empty(kwargs):
    with pytest.raises(ValueError, match="exactly one of"):
        Alternative(value=0, **kwargs)


def test_alternative_must_say_which_kind_it_is():
    """An arm with nothing at all is indistinguishable from an oversight -- an empty
    arm must say `unproduced=<why>`, not just omit everything.
    """
    with pytest.raises(ValueError, match="declares nothing at all"):
        Alternative(value=0)


def test_switch_default_must_be_a_declared_alternative():
    with pytest.raises(ValueError, match="not among the declared alternatives"):
        Switch(
            path=".a.b",
            default=7,
            alternatives=(Alternative(value=0, declarations=(SudoDensityLimit,)),),
        )


def test_unproduced_arm_assembles_as_empty_rather_than_raising():
    """`unproduced` is `unported`'s quieter sibling: it contributes no nodes but does
    not refuse. Pinned on the real instance, `.costs.i_cost_model == 1` -- PROCESS's own
    default, and therefore the arm `GRAPH` is built from at import time, which is why it
    cannot be `unported` (see that switch's own comment block).
    """
    switch = next(s for s in TOPOLOGY_SWITCHES if s.path == ".costs.i_cost_model")
    assert switch.choose(1) == ()  # assembles, contributes nothing, does not raise

    # ...and at graph level: the arm's absence is the only difference from the 1990 arm.
    choices = dict(REFERENCE_CONFIGURATION.choices)
    with_1990 = graph_for(Configuration({**choices, ".costs.i_cost_model": 0}))
    with_none = graph_for(Configuration({**choices, ".costs.i_cost_model": 1}))
    assert {n.path_str() for n in with_none.nodes} < {
        n.path_str() for n in with_1990.nodes
    }


def test_the_1990_cost_model_arm_is_the_only_producer_of_coe():
    """`.costs.coe` -- the `i_figure_merit == 6` objective -- has a producer under
    `i_cost_model == 0` and, honestly, none at all under PROCESS's own default. The
    second half is the claim `unproduced` exists to make checkable: a consumer of
    `.costs.coe` must surface as an unowned input there, not be silently satisfied by
    the 1990 model's formula.
    """
    coe = ".costs.coe"
    owners = {
        value: {
            v.path_str()
            for v in graph_for(Configuration({".costs.i_cost_model": value})).owners
        }
        for value in (0, 1)
    }
    assert coe in owners[0]
    assert coe not in owners[1]
