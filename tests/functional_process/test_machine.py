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
import os
import re

import equinox as eqx
import pytest
from cottax.interfaces.pytree_namespace_module import spell_flat, to_graph

from functional_process import boundary as fp_boundary
from functional_process.boundary import orphaned_by
from functional_process.indat import (
    BLANKET_MASSES,
    BLANKET_SHIELD_POWER,
    BUILDING_SIZING,
    CONFINEMENT_SCALING,
    CRYO_Q_NUC,
    ENERGY_STORAGE,
    EnergyStorageCostUnpulsed,
    ThermalStorageModel,
    CONFINEMENT_TAIL,
    PLASMA_POWER_LOSS,
    COST_MODEL,
    COST_OF_ELECTRICITY,
    ELECTRIC_PRODUCTION,
    FW_AREA,
    GRAPH,
    HEATING,
    PROFILE_PARAMETERISATION,
    REFERENCE_INPUT_FILE,
    REFERENCE_MACHINE,
    REFERENCE_MACHINE_SWITCHES,
    ST_INIT_I_PLASMA_PEDESTAL,
    ProfileParameterisationPedestal,
    TF_POWER,
    UNPORTED,
    AFwTotalNoPowerflow,
    AFwTotalWithPowerflow,
    machine_from_indat,
    switches_from_indat,
)
from functional_process.total_process import TokamakProcess
from process.data_structure.physics_variables import (
    ConfinementRadiationLossModel,
    ConfinementTimeModel,
    PlasmaIgnitionModel,
)
from process.models.tfcoil.base import TFConductorModel


def occupant_class(entry):
    """The class a registry entry builds, seeing through a settings-carrying partial."""
    return entry.func if isinstance(entry, functools.partial) else entry


def _plain(entry):
    return entry()


def _electric_production(entry):
    """`ELECTRIC_PRODUCTION`'s entries are builders taking `.tfcoil.i_tf_sup`, which
    `machine_from_indat` resolves from the `tf_power` slot and threads in rather than
    letting the occupant hardcode. The reference machine's value is what these swap
    tests hold it at.
    """
    return entry(TFConductorModel.SUPERCONDUCTING)


def _costs(entry):
    """`Costs` has two slots of its own -- `cost_of_electricity`, whose occupant may be
    `None`, and `energy_storage_cost` -- so it cannot be default-constructed. Built with
    the reference machine's.
    """
    return entry(
        cost_of_electricity=REFERENCE_MACHINE.costs.cost_of_electricity,
        energy_storage_cost=REFERENCE_MACHINE.costs.energy_storage_cost,
    )


def _profile_parameterisation(entry):
    """The parabolic arm has a slot of its own -- `ecrh_density_limit`, which is the
    `EcrhDensityLimit` node on a stellarator and `None` on a tokamak, because
    `st_d_limit_ecrh` is reached only from `st_phys`. So it cannot be
    default-constructed either; built with the reference (stellarator) machine's, which
    is the arm these swap tests are about.

    The pedestal arm has no such slot and takes no argument, so the `getattr` fallback
    rather than a second table: it is the same registry.
    """
    if entry is ProfileParameterisationPedestal:
        return entry()
    return entry(
        ecrh_density_limit=REFERENCE_MACHINE.physics.profiles.parameterisation.ecrh_density_limit
    )


SLOTS = [
    # (field, registry, where the occupant sits, how to build one)
    # No "PROCESS's own default" column any more: no slot the factory fills has a
    # default, so there is nothing for one to be compared against. Where PROCESS's
    # default matters it is the `switches.get` fallback inside `machine_from_indat`,
    # exercised through `test_a_silent_indat_is_still_refused_but_no_longer_on_istell`.
    (
        "i_confinement_time",
        CONFINEMENT_SCALING,
        lambda m: m.physics.confinement_time.scaling,
        _plain,
    ),
    (
        "i_rad_loss",
        CONFINEMENT_TAIL,
        lambda m: m.physics.confinement_time.tail,
        _plain,
    ),
    (
        "i_plasma_ignited_i_rad_loss",
        PLASMA_POWER_LOSS,
        lambda m: m.physics.confinement_time.power_loss,
        _plain,
    ),
    ("isthtr", HEATING, lambda m: m.stellarator.heating, _plain),
    ("ipowerflow", FW_AREA, lambda m: m.stellarator.fw_area, _plain),
    (
        "i_plasma_pedestal",
        PROFILE_PARAMETERISATION,
        lambda m: m.physics.profiles.parameterisation,
        _profile_parameterisation,
    ),
    ("i_bldgs_size", BUILDING_SIZING, lambda m: m.buildings.sizing, _plain),
    ("i_tf_sup", TF_POWER, lambda m: m.power.tf_power, _plain),
    (
        "ireactor",
        ELECTRIC_PRODUCTION,
        lambda m: m.availability.electric_production,
        _electric_production,
    ),
    (
        "ireactor_ipnet",
        COST_OF_ELECTRICITY,
        lambda m: m.costs.cost_of_electricity,
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
    ("i_cost_model", COST_MODEL, lambda m: m.costs, _costs),
    (
        "inuclear_i_tf_sup",
        CRYO_Q_NUC,
        lambda m: m.power.cryo_q_nuc,
        lambda entry: None if entry is None else entry(),
    ),
    (
        "i_pulsed_plant_istore",
        ENERGY_STORAGE,
        lambda m: m.costs.energy_storage_cost,
        lambda entry: (
            entry()
            if entry is EnergyStorageCostUnpulsed
            else entry(istore=ThermalStorageModel.ELECTROWATT_OPTION_1)
        ),
    ),
]

SINGLE_FIELDS = [
    f
    for f, _r, _w, _b in SLOTS
    if not f.startswith("blktmodel_") and f != "ireactor_ipnet"
]
"""The slots addressed by one integer -- the joint keys are derived, not written.

`ireactor_ipnet` joins the two `blktmodel_*` keys here: `_cost_of_electricity_arm`
turns `.costs.ireactor` and `.costs.ipnet` into one arm index, so there is no single
integer named `ireactor_ipnet` for an IN.DAT to set."""

BASELINE_INDAT = {"istell": 6, "i_cost_model": 0, "i_plasma_ignited": 1}
"""The least an IN.DAT must say for `machine_from_indat` to get past the slots whose
PROCESS default is refused. Written into every temp file below so a test about one field
fails on that field and not on `istell`.

`i_plasma_ignited` joined them when the confinement node became slots: PROCESS's own
default is `NON_IGNITED`, whose head arm adds injected heating and therefore reads a
variable the written arm does not, so it is a real branch this port has not written.
Refusing is the same honest answer `istell` gives -- a machine assembled from the wrong
arm's reads is exactly the invented-edge defect the split removes."""


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


SWAP_PIN = os.path.join(os.path.dirname(fp_boundary.__file__), "reference_swaps.txt")
"""Which reads each alternative occupant leaves without a producer.

Generated -- `FP_WRITE_SWAP_PIN=1 $PY -m pytest tests/functional_process/test_machine.py
-k orphans` -- and never hand-edited, same discipline as the boundary pin beside it.
"""


def _swap_orphans():
    """Every registered occupant, and what swapping it in orphans."""
    out = {}
    for field, registry, where, build in SLOTS:
        for value, occupant in registry.items():
            machine = eqx.tree_at(where, REFERENCE_MACHINE, build(occupant),
                                  is_leaf=lambda x: x is None)
            for var in orphaned_by(GRAPH, to_graph(machine)):
                out.setdefault(f"{field}={value}", []).append(var.path_str())
    return {k: sorted(v) for k, v in out.items()}


def test_swapping_an_occupant_orphans_only_what_is_recorded():
    """**The swap contract**: swap the occupant, rebuild, and account for every read
    that lost its owner.

    `next_steps.md` §12.2 designed this and nothing implemented it until
    `functional_process/boundary.py`. The hazard is *partial overlap* -- an occupant
    owning a subset of what it replaces, leaving the difference with no producer and its
    consumers silently reading PROCESS's `DataStructure`. Same defect class as a missing
    producer: eight recorded instances, none ever found by a check.

    **The first run of this check found that every multi-arm slot in the tree has one**
    -- six slots, thirty-seven reads. That is not a bug list yet and the test does not
    pretend it is one: under `isthtr = 2` there is no ECRH power to compute, and PROCESS
    genuinely leaves `.current_drive.p_hcd_ecrh_injected_total_mw` at its initialised
    value. What the port cannot say today is *which* of the thirty-seven are that and
    which are a producer someone forgot, because a read served by a `DataStructure`
    default and a read served by nothing look identical from inside the graph. They stop
    looking identical exactly when the boundary becomes declared rather than implied,
    which is where this port is going.

    So it is pinned rather than asserted away: the set is recorded, a **new** overlap
    fails, and the recorded ones are a work list with a name. Deliberately *not*
    asserted: that two occupants own the same set. §12.2 rejects that outright --
    `i_cost_model`'s arms genuinely compute different things and forcing a common set
    means inventing fields that exist only to satisfy a test.
    """
    orphans = _swap_orphans()
    if os.environ.get("FP_WRITE_SWAP_PIN"):
        with open(SWAP_PIN, "w", encoding="utf-8") as handle:
            handle.write(
                "# Reads each alternative occupant leaves with no producer, against the\n"
                "# reference machine. Generated by FP_WRITE_SWAP_PIN=1 pytest -k orphans;\n"
                "# do not hand-edit. A new line here is a new partial overlap -- see\n"
                "# test_machine.test_swapping_an_occupant_orphans_only_what_is_recorded.\n"
            )
            for arm, paths in sorted(orphans.items()):
                for path in paths:
                    handle.write(f"{arm} {path}\n")

    with open(SWAP_PIN, encoding="utf-8") as handle:
        pinned = {}
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                arm, _, path = line.partition(" ")
                pinned.setdefault(arm, []).append(path)

    assert orphans == {k: sorted(v) for k, v in pinned.items()}


@pytest.mark.parametrize(
    ("field", "registry", "where", "build"),
    [(f, r, w, b) for f, r, w, b in SLOTS if len(r) > 1],
    ids=[f for f, r, _w, _b in SLOTS if len(r) > 1],
)
def test_occupants_of_one_slot_differ(field, registry, where, build):
    """Two occupants of one slot must actually be different models.

    **Compared by class, not by ports, and that is a deliberate loosening.** The earlier
    form asserted the occupants' *ports* differ, on the reasoning that a slot whose
    occupants read and own the same things decides nothing. That reasoning holds only
    while a switch may also live as a static kwarg: it is what makes "these two branches
    read the same variables, so keep the kwarg" (`switch_kwarg_survey.md` band (c)) the
    right answer, and the by-ports check was that policy's enforcement.

    **The policy is now that no switch is a static kwarg, whatever its reads.** A switch
    value selects an occupant, full stop -- including two occupants that read and own the
    same things and differ only in a literal (Account 225.3's two ELECTROWATT designs are
    the standing example: same two reads, different itemised sum). Under that rule the
    by-ports assertion is not a safety net, it is a blocker on the conversion, so what is
    checked is what still means something: **each value selects a distinct occupant
    class**, so a registry cannot quietly map two values to one entry.

    That is weaker, and the gap it opens is worth naming: it no longer catches a family
    whose occupants are genuinely identical in *behaviour* too. Nothing checks that, and
    nothing cheaply can -- two bodies differing in one constant are indistinguishable
    without evaluating them. `i_tf_sup == 2`, the recorded case where PROCESS runs the
    byte-identical branch to `== 0`, is still handled by refusing it rather than
    registering it twice; that refusal is now the only thing standing there.
    """
    occupants = {value: entry for value, entry in registry.items()}
    distinct = {id(entry) for entry in occupants.values()}
    assert len(distinct) == len(occupants), (
        f"{field}: occupants {sorted(occupants)} are not distinct models -- two values "
        f"map to one entry, so at least one of them decides nothing"
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


def test_a_silent_indat_is_still_refused_but_no_longer_on_istell(tmp_path):
    """An IN.DAT that sets nothing yields no machine -- and the reason moved to
    `i_cost_model`.

    **Replaces `test_machine_defaults_are_process_defaults`**, whose premise --
    *"`Machine()`'s field defaults are PROCESS's bare defaults"* -- is gone by
    construction: no slot the factory fills has a default, so there is no bare tree whose
    defaults could be read. That contract was never true either. PROCESS defaults
    `i_confinement_time = 34` and `i_plasma_ignited = 0`; the tree carried `38` and `1`,
    because the reference run's values had been transcribed into the slot's *constructor
    kwargs*, where the old test -- which compared occupant classes only -- could not see
    them. A test that cannot fail on the thing it is named for is worse than no test.

    **Re-targeted, not deleted, when `TokamakProcess` landed.** This test was
    `test_a_silent_indat_is_refused_naming_istell`, and its stated ground was *"PROCESS's
    own default is `istell = 0`, a tokamak; this tree has no tokamak"*. It has one now --
    an empty `Tokamak` namespace inside a `TokamakProcess`, which assembles the shared
    subsystems and nothing stellarator-specific -- so `istell == 0` is a device rather
    than a refusal and asserting otherwise would be asserting a fact that stopped being
    true. What the test was *for* survives untouched and is what it asserts now:

    1. **PROCESS's bare defaults still do not silently assemble.** The refusal a silent
       file now gets is `i_plasma_ignited = 0` with `i_rad_loss = 1` -- PROCESS's own
       defaults, and the confinement head arm that adds injected heating, which this
       port has not written. That is the same refusal `large_tokamak_eval.IN.DAT`
       itself gets (`_audit/tokamak_boundary.md` §"What blocked the real file"), so it
       is not an artefact of an empty file: it is the first thing a real conventional
       tokamak asks for that this port has not got. `i_cost_model = 1` (KOVARI_2014) is
       refused behind it, which is why `BASELINE_INDAT` sets both.
    2. **The device is still resolved first.** A file whose *only* content is a refused
       `istell` value reports `istell`, not `i_cost_model`, even though `i_cost_model`'s
       default is refused too and the constructor would reach it. That ordering is the
       property the old name was really guarding, and it is the half of the old test
       that had nothing to do with the tokamak.
    3. **The default device is the one PROCESS names.** Given only the two switches whose
       PROCESS defaults this port refuses, a file that never mentions `istell` builds a
       `TokamakProcess` -- not a `StellaratorProcess`, and not an error.
    """
    indat = tmp_path / "IN.DAT"
    indat.write_text("")
    with pytest.raises(
        NotImplementedError, match=re.escape("i_plasma_ignited_i_rad_loss == -1")
    ):
        machine_from_indat(indat)
    ignited = tmp_path / "IGNITED.DAT"
    ignited.write_text("i_plasma_ignited = 1\n")
    with pytest.raises(NotImplementedError, match=re.escape("i_cost_model == 1")):
        machine_from_indat(ignited)

    preset = tmp_path / "PRESET.DAT"
    preset.write_text("istell = 3\n")
    with pytest.raises(NotImplementedError, match=re.escape("istell == 3")):
        machine_from_indat(preset)

    silent_device = tmp_path / "TOK.DAT"
    silent_device.write_text("i_cost_model = 0\ni_plasma_ignited = 1\n")
    assert type(machine_from_indat(silent_device)) is TokamakProcess


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
    confinement = REFERENCE_MACHINE.physics.confinement_time
    assert type(confinement.scaling) is occupant_class(
        CONFINEMENT_SCALING[
            ConfinementTimeModel(REFERENCE_MACHINE_SWITCHES["i_confinement_time"])
        ]
    )
    assert type(confinement.tail) is occupant_class(
        CONFINEMENT_TAIL[
            ConfinementRadiationLossModel(REFERENCE_MACHINE_SWITCHES["i_rad_loss"])
        ]
    )
    # The head is a joint dispatch, so what it proves is that both integers reached it.
    assert type(confinement.power_loss) is occupant_class(PLASMA_POWER_LOSS[0])
    # `i_plasma_pedestal` is the one slot the file does *not* decide: `st_init` forces
    # it to `0` on every stellarator run, so the factory reads
    # `ST_INIT_I_PLASMA_PEDESTAL` and not the file. The reference file happens to agree,
    # which is why this assertion could be written either way and is deliberately
    # written both -- the two must not be allowed to drift apart silently.
    assert REFERENCE_MACHINE_SWITCHES["i_plasma_pedestal"] == ST_INIT_I_PLASMA_PEDESTAL
    assert type(REFERENCE_MACHINE.physics.profiles.parameterisation) is occupant_class(
        PROFILE_PARAMETERISATION[ST_INIT_I_PLASMA_PEDESTAL]
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
    if field.startswith(("blktmodel_", "i_plasma_ignited_", "i_pulsed_plant_")):
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
