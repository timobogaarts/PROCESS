"""The machine tree: what each slot may hold, and that `Machine()` means PROCESS.

**Exclusivity is not tested here any more, because it is no longer testable.** It used
to be the point of this file: `Switch.check_arms_are_exclusive` detected "these nodes
cannot coexist" by watching their owned outputs collide, and several tests existed to
police that detection. One slot holds one occupant, so exclusivity is by construction and
there is nothing left to check. What replaced those tests is the question they were a
proxy for -- *after* choosing an occupant, does every remaining read still have an owner?
-- which is `model_tree_design.md` §6's boundary postcondition, and is `§8` step 5's work,
not this file's.
"""

import functools
import re

import equinox as eqx
import pytest
from cottax.interfaces.pytree_namespace_module import spell_flat, to_graph

from functional_process.total_process import (
    BLANKET_MASSES,
    BLANKET_SHIELD_POWER,
    BUILDING_SIZING,
    CONFINEMENT_TIME,
    COST_MODEL,
    ELECTRIC_PRODUCTION,
    FW_AREA,
    HEATING,
    PROFILE_PARAMETERISATION,
    REFERENCE_INPUT_FILE,
    REFERENCE_MACHINE,
    REFERENCE_MACHINE_SWITCHES,
    TF_POWER,
    UNPORTED,
    AFwTotalNoPowerflow,
    AFwTotalWithPowerflow,
    Machine,
    machine_from_indat,
    switches_from_indat,
)
from process.data_structure.physics_variables import (
    ConfinementRadiationLossModel,
    ConfinementTimeModel,
    PlasmaIgnitionModel,
)


def occupant_class(entry):
    """The class a registry entry builds, seeing through a settings-carrying partial."""
    return entry.func if isinstance(entry, functools.partial) else entry


def _plain(entry):
    return entry()


def _confinement(entry):
    """`ConfinementTime`'s three settings, which both occupants of that slot carry."""
    return entry(
        i_confinement_time=ConfinementTimeModel.ISS04_STELLARATOR,
        i_rad_loss=ConfinementRadiationLossModel.CORE_ONLY,
        i_plasma_ignited=PlasmaIgnitionModel.IGNITED,
    )


SLOTS = [
    # (field, registry, where the occupant sits, PROCESS's own default, how to build)
    (
        "istell",
        CONFINEMENT_TIME,
        lambda m: m.physics.confinement_time.model,
        0,
        _confinement,
    ),
    ("isthtr", HEATING, lambda m: m.stellarator.heating, 1, _plain),
    ("ipowerflow", FW_AREA, lambda m: m.stellarator.fw_area, 1, _plain),
    (
        "i_plasma_pedestal",
        PROFILE_PARAMETERISATION,
        lambda m: m.physics.profiles.parameterisation,
        1,
        _plain,
    ),
    ("i_bldgs_size", BUILDING_SIZING, lambda m: m.buildings.sizing, 0, _plain),
    ("i_tf_sup", TF_POWER, lambda m: m.power.tf_power, 1, _plain),
    (
        "ireactor",
        ELECTRIC_PRODUCTION,
        lambda m: m.availability.electric_production,
        1,
        _plain,
    ),
    (
        "blktmodel_ipowerflow",
        BLANKET_SHIELD_POWER,
        lambda m: m.stellarator.fwbs.blanket_shield_power,
        2,
        _plain,
    ),
    (
        "blktmodel_blkttype",
        BLANKET_MASSES,
        lambda m: m.stellarator.fwbs.blanket_masses,
        2,
        _plain,
    ),
    ("i_cost_model", COST_MODEL, lambda m: m.costs, None, _plain),
]

OCCUPANTS = [
    (field, registry, where, value, build)
    for field, registry, where, _, build in SLOTS
    for value in registry
]


@pytest.mark.parametrize(
    ("field", "registry", "where", "value", "build"),
    OCCUPANTS,
    ids=[f"{f}={v}" for f, _, _, v, _b in OCCUPANTS],
)
def test_every_registered_occupant_assembles(field, registry, where, value, build):
    """Every occupant any registry can produce builds a non-empty graph in its slot.

    The old form of this test assembled a whole `Configuration` per arm. Swapping one
    slot is the sharper question and the cheaper one: it isolates the occupant from every
    other choice, so a failure names the slot rather than a configuration.
    """
    machine = eqx.tree_at(
        where, REFERENCE_MACHINE, build(registry[value]), is_leaf=lambda x: x is None
    )
    graph = to_graph(machine)
    assert graph.definitions, f"{field} == {value} assembled an empty graph"


@pytest.mark.parametrize(
    ("field", "registry", "where", "build"),
    [(f, r, w, b) for f, r, w, _, b in SLOTS if len(r) > 1],
    ids=[f for f, r, _, _, _b in SLOTS if len(r) > 1],
)
def test_occupants_of_one_slot_differ(field, registry, where, build):
    """Two occupants of one slot must actually differ in the graph they produce.

    **Compared by ports, not by node names, and that is the change step 3 forced.** Node
    identity is the slot path now, so swapping an occupant renames nothing -- the old
    form of this test compared node *name sets* and would pass vacuously for every
    single-node slot. What still distinguishes two occupants is what they read and own.

    A slot whose occupants have identical ports is a slot that decides nothing: either
    the family is spurious, or one occupant is a mis-registration of the other.
    `i_tf_sup == 2` is the recorded case where PROCESS really does run the identical
    branch, and it is refused rather than registered twice for exactly this reason.
    """
    ports = {}
    for value, entry in registry.items():
        machine = eqx.tree_at(
            where, REFERENCE_MACHINE, build(entry), is_leaf=lambda x: x is None
        )
        graph = to_graph(machine)
        ports[value] = frozenset(
            (
                name.path_str(),
                frozenset(i.var.path_str() for i in node.inputs),
                frozenset(o.var.path_str() for o in node.outputs),
            )
            for name, node in graph.definitions.items()
        )
    assert len(set(ports.values())) == len(ports), (
        f"{field}: occupants {sorted(ports)} read and own exactly the same things"
    )


def test_ipowerflow_decides_whether_the_graph_has_a_cycle():
    """The `fw_area` occupants differ in *reads*, not just formula, and that flips an SCC.

    `AFwTotalWithPowerflow` reads `.fwbs.f_ster_div_single`, which `divertor` owns, while
    `divertor` reads `.first_wall.a_fw_total`, which both occupants own. So
    `ipowerflow != 0` is genuinely coupled and `ipowerflow == 0` is not.

    Pinned because it is the concrete counterexample to modelling a switch as one fused
    node branching internally: no such node could express *"this configuration has no
    cycle"*.

    Checks this specific SCC's presence and absence, not indices into `.cycles` or overall
    `.is_acyclic` -- the graph carries several unconditional declared `FixedPoint`
    self-loops, so both machines are `not is_acyclic`.
    """

    def graph_with(occupant):
        return to_graph(
            eqx.tree_at(lambda m: m.stellarator.fw_area, REFERENCE_MACHINE, occupant)
        )

    # `.stellarator.divertor` is a slot, so it is snake_case and carries no class name
    # (`model_tree_design.md` §3.2 -- identity is the place); `.stellarator.fw_area` is
    # likewise the slot, whichever occupant fills it, which is the property that makes
    # this pair comparable at all.
    cycle = {".stellarator.divertor", ".stellarator.fw_area"}
    coupled = graph_with(AFwTotalWithPowerflow())
    uncoupled = graph_with(AFwTotalNoPowerflow())
    assert cycle in [{spell_flat(n) for n in c} for c in coupled.cycles]
    assert cycle not in [{spell_flat(n) for n in c} for c in uncoupled.cycles]


@pytest.mark.parametrize(
    ("field", "registry", "where", "default"),
    [(f, r, w, d) for f, r, w, d, _b in SLOTS if "_" not in f or f.startswith("i_")],
    ids=[f for f, _, _, _d, _b in SLOTS if "_" not in f or f.startswith("i_")],
)
def test_machine_defaults_are_process_defaults(field, registry, where, default):
    """`Machine()`'s field defaults **are** PROCESS's bare defaults, slot by slot.

    Read off the live `DataStructure` dataclass rather than re-parsing source, so a
    renamed or retyped field fails here instead of drifting. This is the contract that
    used to be `Switch.default`'s, and it is the reason a silent IN.DAT still reproduces
    the run PROCESS itself would do.

    An absent occupant (`costs = None` at `i_cost_model == 1`) is checked as absence:
    the registry has no entry, and the slot holds `None`.
    """
    from process.core.model import DataStructure

    if field in ("blktmodel_ipowerflow", "blktmodel_blkttype"):
        pytest.skip("a synthetic joint key -- no single DataStructure field to compare")

    area = {
        "istell": "stellarator",
        "isthtr": "stellarator",
        "ipowerflow": "heat_transport",
        "i_plasma_pedestal": "physics",
        "i_bldgs_size": "buildings",
        "i_tf_sup": "tfcoil",
        "ireactor": "costs",
        "i_cost_model": "costs",
    }[field]
    actual = int(getattr(getattr(DataStructure(), area), field))
    occupant = where(Machine())
    if actual not in registry:
        assert occupant is None, (
            f"{field}: PROCESS defaults to {actual}, which has no occupant, so "
            f"Machine()'s slot must be None -- it holds {type(occupant).__name__}"
        )
    else:
        assert type(occupant) is occupant_class(registry[actual]), (
            f"{field}: PROCESS defaults to {actual}, which the registry maps to "
            f"{occupant_class(registry[actual]).__name__}, but Machine() holds "
            f"{type(occupant).__name__}"
        )


def test_reference_machine_matches_the_input_file():
    """`REFERENCE_MACHINE_SWITCHES` says what `REFERENCE_INPUT_FILE` actually sets.

    **This is the check that closes the bug class.** Five registration errors in this
    project came from a value copied off PROCESS's bare defaults instead of the run being
    modelled, each found only afterwards by the MDA harness. Parsing the input file makes
    *"the assembled machine matches the run it is validated against"* a checked property
    rather than something someone remembered.

    Both directions: every switch the file sets that this port has a slot for must be
    transcribed, and every transcribed entry must be one the file really sets.
    """
    in_file = switches_from_indat(REFERENCE_INPUT_FILE)
    ours = {f for f, _, _, _d, _b in SLOTS if "_" not in f or f.startswith("i_")}

    for field, value in REFERENCE_MACHINE_SWITCHES.items():
        assert in_file.get(field) == value, (
            f"REFERENCE_MACHINE_SWITCHES says {field} = {value}, but "
            f"{REFERENCE_INPUT_FILE} says {in_file.get(field)!r}"
        )
    for field in ours:
        if field in in_file:
            assert field in REFERENCE_MACHINE_SWITCHES, (
                f"{REFERENCE_INPUT_FILE} sets {field} = {in_file[field]}, but "
                f"REFERENCE_MACHINE_SWITCHES does not transcribe it -- it would fall "
                f"through to the Machine() default"
            )


def test_reference_machine_is_what_the_factory_builds():
    """`machine_from_indat` on the reference file picks the occupants the file names."""
    assert type(REFERENCE_MACHINE.physics.confinement_time.model) is occupant_class(
        CONFINEMENT_TIME[REFERENCE_MACHINE_SWITCHES["istell"]]
    )
    assert type(REFERENCE_MACHINE.physics.profiles.parameterisation) is occupant_class(
        PROFILE_PARAMETERISATION[REFERENCE_MACHINE_SWITCHES["i_plasma_pedestal"]]
    )
    assert type(REFERENCE_MACHINE.costs) is occupant_class(
        COST_MODEL[REFERENCE_MACHINE_SWITCHES["i_cost_model"]]
    )
    assert REFERENCE_MACHINE.stellarator.machine_config is not None


@pytest.mark.parametrize(("field", "value"), sorted(UNPORTED), ids=str)
def test_a_refused_value_says_why(tmp_path, field, value):
    """Asking for an unported value raises, and the message carries the recorded reason.

    The reason strings are audit content -- they moved verbatim out of the
    `Alternative(unported=...)` declarations this replaced. A refusal that did not name
    one would be indistinguishable from a value PROCESS never had.
    """
    if "_" in field and not field.startswith("i_"):
        pytest.skip("joint key -- exercised through the two integers it derives from")
    indat = tmp_path / "IN.DAT"
    indat.write_text(f"{field} = {value}\n")
    with pytest.raises(NotImplementedError, match=re.escape(f"{field} == {value}")):
        machine_from_indat(indat)


def test_an_unknown_value_is_rejected_naming_what_exists(tmp_path):
    """A typo'd value fails loudly rather than falling through to a default -- the one
    property of `Switch.choose` worth keeping verbatim.
    """
    indat = tmp_path / "IN.DAT"
    indat.write_text("isthtr = 99\n")
    with pytest.raises(ValueError, match="not a known value"):
        machine_from_indat(indat)


def test_an_absent_occupant_assembles_as_nothing(tmp_path):
    """`costs = None` contributes no nodes and does not refuse.

    Absence is `refusal`'s quieter sibling and the two sit on one switch:
    `i_cost_model == 1` (PROCESS's own default) is absent because the honest answer is
    *"this configuration computes no cost of electricity"*; `== 2` is refused because
    assembling anything would be a guess at a model the caller supplies at runtime.
    """
    assert Machine().costs is None
    without = to_graph(
        eqx.tree_at(
            lambda m: m.costs, REFERENCE_MACHINE, None, is_leaf=lambda x: x is None
        )
    )
    assert {n.path_str() for n in without.nodes} < {
        n.path_str() for n in to_graph(REFERENCE_MACHINE).nodes
    }


def test_the_1990_cost_model_is_the_only_producer_of_coe():
    """`.costs.coe` -- the `i_figure_merit == 6` objective -- has a producer with the
    1990 occupant and, honestly, none at all at PROCESS's default. The second half is the
    claim absence exists to make checkable: a consumer of `.costs.coe` must surface as an
    unowned input there, not be silently satisfied by the 1990 model's formula.
    """
    with_costs = to_graph(REFERENCE_MACHINE)
    without = to_graph(
        eqx.tree_at(
            lambda m: m.costs, REFERENCE_MACHINE, None, is_leaf=lambda x: x is None
        )
    )
    assert ".costs.coe" in {v.path_str() for v in with_costs.owners}
    assert ".costs.coe" not in {v.path_str() for v in without.owners}
