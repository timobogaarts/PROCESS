"""The machine tree: what each slot may hold, and that only an IN.DAT may fill one.

**There is no bare tree to test any more.** `StellaratorProcess()` raises: every slot
`machine_from_indat` fills lost its default, so the tree carries no configuration of its
own and there is nothing to compare against PROCESS's `*_variables.py` defaults. What
replaced that test is the factory's own refusals -- a silent IN.DAT, an unported value,
a typo -- since the factory is now the only thing that builds a machine.

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
    # (field, registry, where the occupant sits, how to build one)
    # No "PROCESS's own default" column any more: no slot the factory fills has a
    # default, so there is nothing for one to be compared against. Where PROCESS's
    # default matters it is the `switches.get` fallback inside `machine_from_indat`,
    # exercised through `test_a_silent_indat_is_refused_naming_istell`.
    (
        "istell",
        CONFINEMENT_TIME,
        lambda m: m.physics.confinement_time.model,
        _confinement,
    ),
    ("isthtr", HEATING, lambda m: m.stellarator.heating, _plain),
    ("ipowerflow", FW_AREA, lambda m: m.stellarator.fw_area, _plain),
    (
        "i_plasma_pedestal",
        PROFILE_PARAMETERISATION,
        lambda m: m.physics.profiles.parameterisation,
        _plain,
    ),
    ("i_bldgs_size", BUILDING_SIZING, lambda m: m.buildings.sizing, _plain),
    ("i_tf_sup", TF_POWER, lambda m: m.power.tf_power, _plain),
    (
        "ireactor",
        ELECTRIC_PRODUCTION,
        lambda m: m.availability.electric_production,
        _plain,
    ),
    (
        "blktmodel_ipowerflow",
        BLANKET_SHIELD_POWER,
        lambda m: m.stellarator.fwbs.blanket_shield_power,
        _plain,
    ),
    (
        "blktmodel_blkttype",
        BLANKET_MASSES,
        lambda m: m.stellarator.fwbs.blanket_masses,
        _plain,
    ),
    ("i_cost_model", COST_MODEL, lambda m: m.costs, _plain),
]

SINGLE_FIELDS = [f for f, _r, _w, _b in SLOTS if not f.startswith("blktmodel_")]
"""The slots addressed by one integer -- the joint keys are derived, not written."""

BASELINE_INDAT = {"istell": 6, "i_cost_model": 0}
"""The least an IN.DAT must say for `machine_from_indat` to get past the two slots whose
PROCESS default is refused. Written into every temp file below so a test about one field
fails on that field and not on `istell`."""


def write_indat(tmp_path, **switches):
    """A temp IN.DAT setting `BASELINE_INDAT` plus `switches` (which win on a clash)."""
    indat = tmp_path / "IN.DAT"
    indat.write_text(
        "".join(f"{f} = {v}\n" for f, v in {**BASELINE_INDAT, **switches}.items())
    )
    return indat


OCCUPANTS = [
    (field, registry, where, value, build)
    for field, registry, where, build in SLOTS
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
    [(f, r, w, b) for f, r, w, b in SLOTS if len(r) > 1],
    ids=[f for f, r, _w, _b in SLOTS if len(r) > 1],
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

    Slots with one occupant are skipped by the `len(r) > 1` filter, which since the
    tokamak arm was deleted includes `istell` -- a one-member family decides nothing
    *here*, but it is still the slot the whole device hangs from.
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


def test_a_silent_indat_is_refused_naming_istell(tmp_path):
    """An IN.DAT that sets nothing yields no machine at all -- it names `istell`.

    **Replaces `test_machine_defaults_are_process_defaults`**, whose premise --
    *"`Machine()`'s field defaults are PROCESS's bare defaults"* -- is gone by
    construction: no slot the factory fills has a default, so there is no bare tree whose
    defaults could be read. That contract was never true either. PROCESS defaults
    `i_confinement_time = 34` and `i_plasma_ignited = 0`; the tree carried `38` and `1`,
    because the reference run's values had been transcribed into the slot's *constructor
    kwargs*, where the old test -- which compared occupant classes only -- could not see
    them. A test that cannot fail on the thing it is named for is worse than no test.

    What is checkable is the refusal. PROCESS's own default is `istell = 0`, a tokamak;
    this tree has no tokamak, so a silent file is refused with that reason rather than
    quietly assembling stellarator geometry under a tokamak confinement law.
    """
    indat = tmp_path / "IN.DAT"
    indat.write_text("")
    with pytest.raises(NotImplementedError, match=re.escape("istell == 0")):
        machine_from_indat(indat)


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

    for field, value in REFERENCE_MACHINE_SWITCHES.items():
        assert in_file.get(field) == value, (
            f"REFERENCE_MACHINE_SWITCHES says {field} = {value}, but "
            f"{REFERENCE_INPUT_FILE} says {in_file.get(field)!r}"
        )
    for field in SINGLE_FIELDS:
        if field in in_file:
            assert field in REFERENCE_MACHINE_SWITCHES, (
                f"{REFERENCE_INPUT_FILE} sets {field} = {in_file[field]}, but "
                f"REFERENCE_MACHINE_SWITCHES does not transcribe it -- it would fall "
                f"through to PROCESS's own default"
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

    Every file is written over `BASELINE_INDAT`, because two of PROCESS's own defaults
    (`istell = 0`, `i_cost_model = 1`) are themselves refused: without it every case
    here would fail on whichever of those the constructor reached first, rather than on
    the value under test.
    """
    if field.startswith("blktmodel_"):
        pytest.skip("joint key -- exercised through the two integers it derives from")
    indat = write_indat(tmp_path, **{field: value})
    with pytest.raises(NotImplementedError, match=re.escape(f"{field} == {value}")):
        machine_from_indat(indat)


def test_an_unknown_value_is_rejected_naming_what_exists(tmp_path):
    """A typo'd value fails loudly rather than falling through to a default -- the one
    property of `Switch.choose` worth keeping verbatim.
    """
    with pytest.raises(ValueError, match="not a known value"):
        machine_from_indat(write_indat(tmp_path, isthtr=99))


def test_the_default_cost_model_is_refused_with_its_reason(tmp_path):
    """`i_cost_model == 1` raises, and says `costs_2015.py` is what is missing.

    **This used to be a test of absence**: the slot was `Costs | None` and PROCESS's own
    default filled it with `None`, on the reasoning that *"this configuration computes no
    cost of electricity"* is the honest answer. It is refused now, because the tree has
    no optional slots left -- a graph silently missing `.costs.coe` and `.costs.concost`
    is exactly the sort of thing that should be said out loud rather than assembled.
    `== 2` sits on the same switch and was already refused, for the other reason: it
    injects a `Model` at runtime that this graph has never seen.
    """
    with pytest.raises(NotImplementedError, match=re.escape("i_cost_model == 1")) as exc:
        machine_from_indat(write_indat(tmp_path, i_cost_model=1))
    assert "costs_2015.py" in str(exc.value)
    assert UNPORTED["i_cost_model", 1] in str(exc.value)


def test_the_1990_cost_model_is_the_only_producer_of_coe():
    """`.costs.coe` -- the `i_figure_merit == 6` objective -- has exactly one producer.

    Removing the occupant is a **structural what-if**, not a configuration this tree
    admits any more: `i_cost_model == 1` is refused, so `costs = None` is reachable only
    by `eqx.tree_at`. The claim is still worth pinning, and it is about the graph rather
    than about any switch -- deleting a node makes its outputs surface as unowned inputs
    at the consumers, instead of being silently satisfied by some other node's formula.
    `.costs.coe` is the sharpest instance, being an objective.
    """
    with_costs = to_graph(REFERENCE_MACHINE)
    without = to_graph(
        eqx.tree_at(
            lambda m: m.costs, REFERENCE_MACHINE, None, is_leaf=lambda x: x is None
        )
    )
    assert ".costs.coe" in {v.path_str() for v in with_costs.owners}
    assert ".costs.coe" not in {v.path_str() for v in without.owners}
