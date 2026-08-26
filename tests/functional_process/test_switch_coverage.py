"""Pins a known gap in `total_process.py`'s switch handling -- deliberately, not a fix.

`machine_from_indat`'s own docstring calls itself **"the only place in this port an
`i_*` integer is ever read"**. It is not: `REFERENCE_INPUT_FILE`
(`tests/regression/input_files/stellarator_helias.IN.DAT`) explicitly sets switches the
factory never looks at --

    i_p_coolant_pumping = 1     i_thermal_electric_conversion = 2   i_plasma_ignited = 1
    i_plant_availability = 0

The first three are instead hand-transcribed as a static constructor kwarg on one or
more slots of `REFERENCE_MACHINE`, matching the file's value *by coincidence of careful
authorship*, not by anything that would notice if a value drifted or a slot were missed.
The fourth, `i_plant_availability`, is not transcribed anywhere at all: PROCESS's own
dispatch on it never runs, because `Stellarator.run()` calls `self.availability.avail()`
directly (`stellarator.py:175`), bypassing `.costs.i_plant_availability` entirely.

**The list shrinks as the binding policy is applied.** `i_confinement_time`/`i_rad_loss`
(the confinement split) and `i_tf_sc_mat` (`_audit/next_steps.md` §14.5) each left it by
becoming a slot the factory fills, which is the intended direction of travel: a switch
read from the file cannot drift from it.

This module is the inventory of that gap, `SWITCH_INVENTORY`, plus tests that keep the
inventory honest against the live tree and the live file, plus -- since step 4d --
`COHERENCE_CASES` and `test_no_slot_contradicts_a_factory_switch`, which are about the
sharper half of the same question: not *"is this transcription right for the reference
run"* but *"can two places in one machine give different answers to one switch"*. They
could, in four ways, and `switch_kwarg_survey.md` band (a) is the record of it.
**None of the inventory tests fix the gap** --
`test_hardcoded_values_agree_with_the_reference_file` passes today because the
remaining transcriptions are, in fact, all correct today. What changes is that a future
drift (a transcription going stale, a new switch joining the transcribed/bypassed set
without a decision, a slot silently losing its static kwarg) now fails a test instead of
waiting to be found by the MDA harness the way `i_confinement_time`/
`i_thermal_electric_conversion`/`i_p_coolant_pumping` originally were
(`total_process.py`'s own `graph_for` docstring).

Import discipline: only symbols the tree refactor in progress does not touch --
`machine_from_indat`, `switches_from_indat`, `REFERENCE_INPUT_FILE`, `REFERENCE_MACHINE`,
`UNPORTED`. No `Machine`/`StellaratorProcess`, no `COMMON` -- this module reaches into
the tree only through `REFERENCE_MACHINE` attribute access and fresh
`machine_from_indat` calls of its own.
"""

import dataclasses
import re
from pathlib import Path

import pytest

from functional_process.indat import (
    REFERENCE_INPUT_FILE,
    REFERENCE_MACHINE,
    ST_INIT_I_PLASMA_PEDESTAL,
    UNPORTED,
    machine_from_indat,
    switches_from_indat,
)
from functional_process.models.switch_enums import IFEModel


@dataclasses.dataclass(frozen=True)
class ReadByFactory:
    """`machine_from_indat` reads this switch from the file itself -- no transcription,
    nothing to pin beyond "the value it read matches the file", which
    `test_inventory_matches_the_reference_file` checks like any other entry.
    """

    value: int


@dataclasses.dataclass(frozen=True)
class Slot:
    """One place in `REFERENCE_MACHINE` where a switch's value is hardcoded.

    `path` is a plain attribute path from a `Machine`/tree root (`m.<path>`), spelled
    without the leading dot PROCESS's own `VarPath`s use, since it walks Python
    attributes on the assembled tree, not a `.area.name` `DataStructure` field.
    `getter` performs that walk; kept as a callable rather than re-deriving it from
    `path` by `functools.reduce(getattr, ...)` so a typo in `path` cannot silently
    point the test somewhere other than what it prints.
    """

    path: str
    getter: object  # Callable[[REFERENCE_MACHINE-shaped tree], int-like]


@dataclasses.dataclass(frozen=True)
class Hardcoded:
    """Not read from the file at all: transcribed as a static constructor kwarg on
    every slot in `slots`, each expected to carry `value`.
    """

    value: int
    slots: tuple[Slot, ...]


@dataclasses.dataclass(frozen=True)
class ForcedByProcess:
    """Set in the file, read by nothing in the factory *on purpose*: PROCESS itself
    overwrites it before any model runs, so the file's value is dead and assembling from
    it would build a configuration PROCESS cannot produce.

    `value` is what the file says; `forced_to` is what PROCESS makes of it. The two
    agree for `REFERENCE_INPUT_FILE` and are recorded separately anyway, so a file that
    started disagreeing would show up here as a diff rather than as a silently-ignored
    input.
    """

    value: int
    forced_to: int
    reason: str


@dataclasses.dataclass(frozen=True)
class Bypassed:
    """Set in the file, consulted by nothing: the PROCESS dispatch this switch would
    drive is itself skipped by a hardcoded solve-time call, so there is no slot for the
    value to be transcribed onto, right or wrong. `value` is still recorded, purely so a
    future change to what the reference file sets it to is visible as a diff here rather
    than silent.
    """

    value: int
    reason: str


SWITCH_INVENTORY: dict[str, ReadByFactory | Hardcoded | ForcedByProcess | Bypassed] = {
    # --- Read by machine_from_indat, and (incidentally) set explicitly by the
    # reference file rather than left at its default. `switches_from_indat` line
    # references are `stellarator_helias.IN.DAT`'s.
    "istell": ReadByFactory(6),  # :137
    "isthtr": ReadByFactory(1),  # :138 -- equals the bare default, set anyway
    "i_plasma_pedestal": ForcedByProcess(  # :127
        value=0,
        forced_to=ST_INIT_I_PLASMA_PEDESTAL,
        reason=(
            "process/models/stellarator/initialization.py:31 -- st_init sets "
            "data.physics.i_plasma_pedestal = 0 unconditionally on every istell != 0 "
            "run, in the same block that zeroes the central solenoid. So the file's "
            "value cannot decide the .physics.profiles.parameterisation slot, and "
            "machine_from_indat reads ST_INIT_I_PLASMA_PEDESTAL instead of the file. "
            "It used to read the file: an IN.DAT saying istell = 6, "
            "i_plasma_pedestal = 1 assembled ProfileParameterisationPedestal for a run "
            "PROCESS executes with parabolic profiles."
        ),
    ),
    "i_cost_model": ReadByFactory(0),  # :248
    "ireactor": ReadByFactory(1),  # :260 -- equals the bare default, set anyway
    # --- Read by the factory, having been hand-transcribed as static kwargs until
    # `_audit/next_steps.md` §14.2's binding policy converted them.
    #
    # **`Hardcoded` has no entries left.** Every switch this file sets that decides a
    # model now decides a *slot*, so `machine_from_indat` reads it from the file and
    # there is no transcription for this inventory to police. That is the stronger
    # check of the two: a value read from the file cannot drift from it, which is the
    # whole defect class this module exists for (`i_confinement_time` was `34` against
    # the file's `38` once, found by the MDA harness and by nothing here).
    "i_p_coolant_pumping": ReadByFactory(1),  # :198
    "i_thermal_electric_conversion": ReadByFactory(2),  # :203
    "i_plasma_ignited": ReadByFactory(1),  # :126
    "i_tf_sc_mat": ReadByFactory(1),  # :235 -- ITER Nb3Sn
    # `Hardcoded` on `winding_pack_intersect_inputs` until `_audit/next_steps.md` §14.5
    # made that node a slot with eight occupants. The eight branches of
    # `jcrit_from_material` read genuinely different `.tfcoil.*` fields, so the single
    # branching node declared six reads that are dead at this value -- one of them
    # `.tfcoil.j_tf_wp`, the sole back-edge closing the coils SCC. Read from the file
    # now, like every other switch that selects a class.
    "i_confinement_time": ReadByFactory(38),  # :129 -- ISS04
    "i_rad_loss": ReadByFactory(1),  # :128 -- CORE_ONLY
    # Both were `Hardcoded` here until
    # the confinement node became slots. They are **read by the factory** now, not
    # transcribed into a static kwarg, so there is no hardcoded field left for this
    # inventory to police -- `REFERENCE_MACHINE_SWITCHES` carries them instead and
    # `test_reference_machine_matches_the_input_file` is what checks them against the
    # file. That is the stronger check of the two: a value read from the file cannot
    # drift from it, which is the whole defect class this module exists for
    # (`i_confinement_time` was `34` against the file's `38` once, found by the MDA
    # harness and by nothing here).
    # --- Set by the file, read by nothing, transcribed nowhere.
    "i_plant_availability": Bypassed(  # :258
        value=0,
        reason=(
            "Stellarator.run() calls self.availability.avail() directly "
            "(stellarator.py:175), bypassing .costs.i_plant_availability's dispatch "
            "entirely -- `Avail` is the node exercised at solve time regardless of "
            "this switch's value. See total_process.py's Availability.avail comment."
        ),
    ),
}
"""Every switch `REFERENCE_INPUT_FILE` sets that bears on which node occupies a slot in
`REFERENCE_MACHINE` -- a record of a known gap, deliberately pinned so it cannot grow
silently. Twelve entries: five the factory reads (and which happen to be set explicitly
rather than left at their default), five hand-transcribed as static kwargs (matching the
file today, checked nowhere before this module), one -- `i_plasma_pedestal` -- the
factory deliberately *stopped* reading because PROCESS overwrites it, and one --
`i_plant_availability` -- set by the file and consulted by no code path in this port at
all.

Deliberately excludes the six further switches `machine_from_indat` reads that this
particular file leaves at their PROCESS default (`ipowerflow`, `blktmodel`, `blkttype`,
`i_tf_sup`, `i_bldgs_size`, `ipnet`, see `FACTORY_READ_SWITCHES` below) -- they are not
switches "the reference IN.DAT sets", so they have no value to pin here. Also excludes
ten more integers `switches_from_indat` picks up from this same file that are not
model-tree switches at all -- see `OUT_OF_SCOPE_INTEGERS`.
"""

FACTORY_READ_SWITCHES = frozenset({
    "istell",
    "isthtr",
    "ipowerflow",
    "blktmodel",
    "blkttype",
    "i_cost_model",
    "i_tf_sup",
    "i_tf_sc_mat",
    "i_bldgs_size",
    "ireactor",
    "ipnet",
    # --- read by the factory since `_audit/next_steps.md` §14.2's switch conversion.
    # Every one of these was an `eqx.field(static=True)` on one or more nodes; each now
    # decides a slot, so the factory reads it and no occupant transcribes it.
    "ife",
    "itart",
    "supercond_cost_model",
    "istore",
    "i_beta_fast_alpha",
    "i_plasma_ignited",
    "i_pflux_fw_neutron",
    "i_pf_energy_storage_source",
    "i_pf_conductor",
    "ibkt_life",
    "i_blkt_dual_coolant",
    "i_blanket_type",
    "secondary_cycle_liq",
    "i_thermal_electric_conversion",
    "i_p_coolant_pumping",
})
"""Every switch name `machine_from_indat` reads via `switches.get(...)` that decides a
slot of the **shared** subsystems or of the stellarator device -- the ones this module's
inventory is about. Read from `indat.py` itself (`switches.get("<name>"` /
`pick("<name>"`), and exercised dynamically by
`test_factory_switches_actually_change_the_assembled_machine` /
`test_factory_switches_that_only_prove_themselves_by_a_refusal` below, so membership is
checked against the factory's real behaviour rather than its source text.

**One caveat, and it is about the machine rather than the factory**: `i_pf_conductor`
is read (into `_cryo_q_loads_arm`/`_cryo_loads_arm`) but cannot change *this* machine's
occupants on its own, because `i_tf_sup == 1` already makes PROCESS's cryoplant guard
true (`power.py:1054-1057`) -- so it has no `_CHANGES_A_SLOT` row and its read is
evidenced by `indat.py` alone.

**Fifteen names joined in `_audit/next_steps.md` §14.2's switch conversion**, and every
one of them was an `eqx.field(static=True)` on one or more nodes before it: `ife`,
`itart`, `supercond_cost_model`, `istore`, `i_beta_fast_alpha`, `i_plasma_ignited`,
`i_pflux_fw_neutron`, `i_pf_energy_storage_source`, `i_pf_conductor`, `ibkt_life`,
`i_blkt_dual_coolant`, `i_blanket_type`, `secondary_cycle_liq`,
`i_thermal_electric_conversion`, `i_p_coolant_pumping`. The tokamak device reads a
further set of its own (`i_hcd_primary`, `i_plasma_geometry`, ...) which this inventory
does not cover -- `machine_survey.py` is the instrument for those.

Superset of `SWITCH_INVENTORY`'s `ReadByFactory` entries: the names not set by
`REFERENCE_INPUT_FILE` have no value for that inventory to pin.

`i_plasma_pedestal` **is not here**, although the file sets it and the tree has a slot
for it: `machine_from_indat` reads `ST_INIT_I_PLASMA_PEDESTAL` for that slot instead,
because `st_init` overwrites the field on every stellarator run. It is a
`ForcedByProcess` entry in `SWITCH_INVENTORY` and is pinned by
`test_a_process_forced_switch_cannot_move_the_machine`, which asserts the opposite of
what a membership here would claim.
"""

OUT_OF_SCOPE_INTEGERS = {
    # Regex artifacts of `_INDAT_INTEGER`: PROCESS's active-constraint/iteration-
    # variable-ID arrays are written as repeated `name = value` lines, and the crude
    # single-assignment parser (deliberately not a full IN.DAT parser, see
    # `switches_from_indat`'s own docstring) only keeps the last one -- not switch data.
    "icc": "constraint-ID array, not a switch -- switches_from_indat keeps only the "
    "last repeated `icc = ...` line",
    "ixc": "iteration-variable-ID array, same regex artifact as icc",
    # Plain numeric inputs that happen to be written without a decimal point, so the
    # integer-only regex catches them even though they are not switches.
    "n_equality_constraints": "a count derived from icc, not a switch",
    "p_plant_electric_net_required_mw": "a physical target power, not a switch",
    "pflux_div_heat_load_max_mw": "a physical bound, not a switch",
    "f_t_alpha_energy_confinement_min": "a physical bound, not a switch",
    # Real PROCESS switches, but outside machine_from_indat's remit entirely: they
    # govern the solver/objective layer (which numerics.i_figure_merit/run mode/
    # iteration cap), not which model occupies a slot in the Machine tree -- CLAUDE.md's
    # own mapping table files numerics.i_figure_merit under "the objective condition of
    # an Optimise problem", not a node.
    "i_process_run_mode": "solver run-mode switch, not a Machine-assembly switch",
    "maxcal": "solver iteration cap, not a switch at all",
    "i_figure_merit": "selects the optimiser's objective, not a model occupant",
    # Real PROCESS switches consulted by ported cost nodes as ordinary numeric reads
    # (`calculate_divertor_cost(..., ifueltyp)`, `calculate_indirect_costs(..., lsa)`
    # in models/costs/costs.py) -- table/multiplier indices inside an otherwise pure
    # formula, not a choice of which node exists. No machine_from_indat dispatch is
    # missing for them; they are wired as plain `FromExactly`s like any other read.
    "ifueltyp": "read as an ordinary numeric input by ported cost nodes, not a "
    "topology switch",
    "lsa": "read as an ordinary numeric input (a 1-4 table index) by ported cost "
    "nodes, not a topology switch",
}
"""The rest of what `switches_from_indat(REFERENCE_INPUT_FILE)` returns once
`SWITCH_INVENTORY`'s twelve entries are accounted for -- eleven more integer-looking
assignments, none of them a Machine-tree switch, each with why. Exists so
`test_inventory_matches_the_reference_file` can assert *exact* coverage of every integer
the file sets rather than a fuzzy subset check, and so a genuinely new switch added to
the file has nowhere to hide: it must land in `SWITCH_INVENTORY` or gain a reasoned entry
here, not neither.
"""


def test_inventory_matches_the_reference_file():
    """`SWITCH_INVENTORY` (plus `OUT_OF_SCOPE_INTEGERS`) accounts for every integer
    `switches_from_indat` finds in `REFERENCE_INPUT_FILE`, and every switch
    `SWITCH_INVENTORY` claims the file sets is one the file actually sets, at the value
    recorded. This is the test that fails the moment someone adds an eighth switch to
    the input file without deciding how this port handles it, or lets one of the twelve
    recorded values drift from the file underneath the inventory.
    """
    file_switches = switches_from_indat(REFERENCE_INPUT_FILE)

    accounted_for = set(SWITCH_INVENTORY) | set(OUT_OF_SCOPE_INTEGERS)
    unaccounted = set(file_switches) - accounted_for
    assert not unaccounted, (
        f"{sorted(unaccounted)}: REFERENCE_INPUT_FILE sets these integers and neither "
        "SWITCH_INVENTORY nor OUT_OF_SCOPE_INTEGERS says what they are -- a new switch "
        "(or a new plain integer input) has appeared in the reference file since this "
        "inventory was last updated. Decide how it is handled and add it to one of the "
        "two, do not silently ignore it."
    )

    phantom = set(SWITCH_INVENTORY) - set(file_switches)
    assert not phantom, (
        f"{sorted(phantom)}: SWITCH_INVENTORY claims REFERENCE_INPUT_FILE sets these, "
        "but switches_from_indat does not find them there any more -- the inventory has "
        "gone stale against the file it is meant to be transcribing."
    )

    stale_values = {
        name: (entry.value, file_switches[name])
        for name, entry in SWITCH_INVENTORY.items()
        if entry.value != file_switches[name]
    }
    assert not stale_values, (
        f"{stale_values}: (recorded, actual) value pairs disagree -- "
        "REFERENCE_INPUT_FILE changed under this inventory, or the inventory was "
        "transcribed wrong to begin with."
    )


@pytest.mark.parametrize(
    ("switch", "slot"),
    [
        pytest.param(switch, slot, id=f"{switch}@{slot.path}")
        for switch, entry in SWITCH_INVENTORY.items()
        if isinstance(entry, Hardcoded)
        for slot in entry.slots
    ],
)
def test_hardcoded_values_agree_with_the_reference_file(switch, slot):
    """The real teeth of this module: read `switch`'s value out of the live
    `REFERENCE_MACHINE` tree at `slot.path`, and check it against what
    `REFERENCE_INPUT_FILE` actually sets `switch` to -- not against `SWITCH_INVENTORY`'s
    own recorded value, which `test_inventory_matches_the_reference_file` already pins
    to the file, so a slip in *this* comparison could not hide behind a slip in that one.

    Every one of the six hardcoded switches agrees with the file today (that is what the
    finding this module records is about: the transcriptions are correct, but nothing
    was checking them). **If this test fails, that is a live bug in `total_process.py`
    -- a slot's static kwarg has drifted from the reference run -- and should be reported
    as one, not silenced by editing the assertion.**

    Enum-valued fields (every slot here is `eqx.field(static=True)`-typed as a PROCESS
    `IntEnum`) compare via `int(...)`, since PROCESS's own switch values are what
    `switches_from_indat` returns.
    """
    file_value = switches_from_indat(REFERENCE_INPUT_FILE)[switch]
    live_value = int(slot.getter(REFERENCE_MACHINE))
    assert live_value == file_value, (
        f"REFERENCE_MACHINE.{slot.path} == {live_value}, but REFERENCE_INPUT_FILE sets "
        f"{switch} = {file_value} -- this slot's static kwarg has drifted from the "
        "reference run it is supposed to match."
    )


def _field_names(obj, seen=None):
    """Every dataclass field name reachable from `obj`, walking `eqx.Module`/plain
    dataclass instances, tuples/lists and dict values. Used only to prove a negative --
    that no slot anywhere in the tree carries a given field name -- which is a stronger,
    more direct check than trying to enumerate "everywhere it is not" by hand.

    Yields
    ------
    str
        Every dataclass field name found while walking `obj`, in traversal order,
        possibly with duplicates.
    """
    if seen is None:
        seen = set()
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            yield f.name
            yield from _field_names(getattr(obj, f.name), seen)
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            yield from _field_names(item, seen)
    elif isinstance(obj, dict):
        for item in obj.values():
            yield from _field_names(item, seen)


def test_i_plant_availability_has_no_occupant_anywhere():
    """Pins the `Bypassed` claim concretely: no field named `i_plant_availability`
    exists anywhere in `REFERENCE_MACHINE`'s tree, on any node.

    This is the sharpest test in the module for the seventh switch precisely because
    there is no slot to read a *value* from -- unlike the six `Hardcoded` switches,
    honesty here cannot be "the value agrees", only "the field is genuinely absent, not
    merely unchecked". If a future pass wires `Avail`/`WardTaylorAvailability` up to
    `.costs.i_plant_availability`, this test starts failing -- correctly, since that is
    exactly the moment `i_plant_availability` should move from `Bypassed` to `Hardcoded`
    or `ReadByFactory` in `SWITCH_INVENTORY` above.
    """
    names = set(_field_names(REFERENCE_MACHINE))
    # Sanity check on the walker itself: every one of the six genuinely-hardcoded switch
    # names must show up (they do have slots), or this test would pass for the wrong
    # reason -- a walker that finds nothing would also "confirm" i_plant_availability's
    # absence.
    hardcoded_names = {
        name for name, entry in SWITCH_INVENTORY.items() if isinstance(entry, Hardcoded)
    }
    missing_positive_control = hardcoded_names - names
    assert not missing_positive_control, (
        f"{missing_positive_control}: these are known to be hardcoded fields "
        f"(SWITCH_INVENTORY), but _field_names did not find them -- the walker itself "
        "is broken, so the absence check below would be meaningless."
    )
    assert "i_plant_availability" not in names, (
        "REFERENCE_MACHINE now has a field named i_plant_availability somewhere -- "
        "SWITCH_INVENTORY's Bypassed entry for it is stale. Update SWITCH_INVENTORY to "
        "Hardcoded/ReadByFactory as appropriate, with the new slot(s)."
    )


_INDAT_ASSIGNMENT = re.compile(r"^(\s*)([A-Za-z_]\w*)(\s*=\s*)-?\d+", re.MULTILINE)


def _indat_with_override(tmp_path, name, value):
    """A copy of `REFERENCE_INPUT_FILE` with one integer switch's line replaced (or, if
    the switch is absent -- left at its PROCESS default -- appended). The minimal
    perturbation needed to prove `machine_from_indat` actually consults a given switch's
    *value*, rather than merely mentioning its name somewhere in the module.
    """
    text = Path(REFERENCE_INPUT_FILE).read_text()
    pattern = re.compile(rf"^(\s*){re.escape(name)}(\s*=\s*)-?\d+", re.MULTILINE)
    new_text, count = pattern.subn(rf"\g<1>{name}\g<2>{value}", text, count=1)
    if count == 0:
        new_text = text + f"\n{name} = {value}\n"
    out = tmp_path / "perturbed.IN.DAT"
    out.write_text(new_text)
    return str(out)


# (switch, override value, probe) -- probe takes a Machine and returns something that
# is expected to differ between REFERENCE_MACHINE and the perturbed one. A type name is
# enough: every occupant pair below is a different class (or None), never two instances
# of the same class with different fields, so "the type changed" is unambiguous proof
# the switch's value reached the slot.
_CHANGES_A_SLOT = (
    ("isthtr", 2, lambda m: type(m.stellarator.heating).__name__),
    ("i_tf_sup", 0, lambda m: type(m.power.tf_power).__name__),
    (
        "i_tf_sc_mat",
        5,
        lambda m: type(m.stellarator.coils.winding_pack_intersect_inputs).__name__,
    ),
    ("i_bldgs_size", 1, lambda m: type(m.buildings.sizing).__name__),
    ("ireactor", 0, lambda m: type(m.availability.electric_production).__name__),
    ("ipnet", 1, lambda m: type(m.costs.cost_of_electricity).__name__),
    (
        "ipowerflow",
        0,
        lambda m: type(m.stellarator.fwbs.blanket_shield_power).__name__,
    ),
    # --- the switches `_audit/next_steps.md` §14.2 turned from static kwargs into
    # slots. One row each, probing the occupant the switch now selects; `None` is a
    # legitimate probe value, as `ipnet`'s row already established.
    (
        "i_beta_fast_alpha",
        0,
        lambda m: type(m.physics.fast_alpha_beta).__name__,
    ),
    (
        "i_plasma_ignited",
        0,
        lambda m: type(m.physics.plasma_composition).__name__,
    ),
    (
        "i_pflux_fw_neutron",
        2,
        lambda m: type(m.stellarator.neutron_wall_load).__name__,
    ),
    (
        "i_tf_sc_mat",
        5,
        lambda m: type(m.stellarator.coils.coils_mass).__name__,
    ),
    (
        "i_pf_energy_storage_source",
        1,
        lambda m: type(m.power.acpow).__name__,
    ),
    ("ibkt_life", 1, lambda m: type(m.availability.avail).__name__),
    (
        "supercond_cost_model",
        1,
        lambda m: type(m.costs.tf_magnet_cost_superconducting).__name__,
    ),
    (
        "i_thermal_electric_conversion",
        0,
        lambda m: type(m.power.eta_turbine).__name__,
    ),
    (
        "secondary_cycle_liq",
        2,
        lambda m: type(m.power.etath_liq).__name__,
    ),
    ("itart", 1, lambda m: type(m.availability.cplife_avail).__name__),
)
"""Seven of the `FACTORY_READ_SWITCHES`: each has a second registered occupant that
`_indat_with_override` can select on its own, so the "really reads it" proof is
"the assembled tree's occupant type changes".

**`istell` left this table and went back to `_CAUSES_A_REFUSAL`**, and the round trip is
worth reading as a measurement rather than as churn. It was a refusal when `istell == 0`
had no tokamak to be; it moved here when `TokamakProcess` landed with twenty-five empty
slots, because an empty device asks nothing and therefore refuses nothing; and it is a
refusal again now that fourteen of those slots have occupants -- because the perturbed
machine is a real what-if and deliberately not a curated one. Taking
`stellarator_helias.IN.DAT` and changing only `istell` yields a *tokamak* carrying that
file's `i_plasma_ignited = 1`, and the tokamak separatrix-power family has no ignited
arm written. So the row still proves the value is consulted, from the other table.

**That is the interesting half.** An empty device slot cannot disagree with an input
file about anything; a filled one can, and immediately did. The refusal names
`i_plasma_ignited_separatrix`, a slot in `.tokamak.physics` that did not exist when this
row was written, which is exactly the evidence that filling the device slot made the
factory answerable for more of the file than it was before.

What that row used to demonstrate is not lost: `test_machine.
test_a_silent_indat_is_still_refused_but_no_longer_on_istell` asserts that a file
carrying only PROCESS's refused defaults builds a `TokamakProcess` with a populated
`.tokamak.build` and TF coil, which is a stronger "a different thing was built" than a
type name.

`i_plasma_pedestal` **left this list** and is not a `FACTORY_READ_SWITCHES` member any
more: `st_init` forces it to `0` on every stellarator run, so the factory reads
`ST_INIT_I_PLASMA_PEDESTAL` and the file's value decides nothing. The *inverse* claim is
pinned instead, by `test_a_process_forced_switch_cannot_move_the_machine`.

`ipnet` **joined it**, and it is the one row here whose occupant is `None`:
`.costs.cost_of_electricity` is a slot now, keyed on `ireactor` and `ipnet` jointly, and
`ipnet = 1` ("let go < 0 (no c-o-e)") empties it -- which is what PROCESS does, since
`Costs.run()` calls `coelc()` only when `ireactor == 1 and ipnet == 0`. `NoneType` is
the probed type name, and that is the point: absence is an occupant.

`i_tf_sc_mat` **joined this list** when `winding_pack_intersect_inputs` became a slot
(`_audit/next_steps.md` §14.5); it was a `Hardcoded` static kwarg before, checked only
for agreeing with the file. `5` (WST Nb3Sn) is one of the seven arms that are not the
reference run's, chosen for having no material read of its own -- the probe is the
occupant's class name, so any of the seven would do.

`ipowerflow` **moved here from `_CAUSES_A_REFUSAL`**, and the old row was pinning a bug.
`BlanketShieldPowerExponential` is a ported occupant of the joint blanket/shield-power
slot -- arm 1 of `_blanket_shield_power_arm`, `blktmodel == 0 & ipowerflow == 0` -- and
it was unreachable, because the joint key used to be derived by passing `blktmodel`'s
*value* through where an arm index was wanted, defaulted to an illegal `2`. Every
`ipowerflow = 0` machine was refused with the reason recorded for a different arm
(`blktmodel == 1`'s `blanket_neutronics()`). The probe is deliberately the joint slot and
not `fw_area`, which `ipowerflow` also selects on its own: `fw_area` would pass whether
or not the joint key were fixed."""

_CAUSES_A_REFUSAL = (
    ("ife", 1, ("ife", IFEModel.INERTIAL_CONFINEMENT)),
    ("istell", 0, ("i_plasma_ignited_separatrix", 1)),
    ("istell", 3, ("istell", 3)),
    ("i_cost_model", 1, ("i_cost_model", 1)),
    ("blktmodel", 1, ("blktmodel_ipowerflow", 0)),
    ("blkttype", 1, ("blktmodel_blkttype", 1)),
)
"""Five refusals, for three different reasons -- and `istell` appears twice, at two
values, for two unrelated reasons.

`i_cost_model`: the switch has exactly one registered occupant, so its *other* value
cannot select a second one -- there is none to select. `i_cost_model == 1` is
KOVARI_2014, unported. It was spelled as a slot holding `None` until the tree stopped
carrying optional slots; it is a refusal now, which is why it moved here from
`_CHANGES_A_SLOT`.

`istell`: **two rows, and they refuse for reasons that have nothing to do with each
other.** `3` is one of the five hardcoded machine presets (`_ISTELL_PRESET_REASON`),
still unported -- a refusal *about the device*, and what an unrecognised device must
hit. `0` is a device this port has, and it refuses about the *physics* the file asks for:
re-reading `stellarator_helias.IN.DAT` as a tokamak carries its `i_plasma_ignited = 1`
into `.tokamak.physics.separatrix_power`, whose ignited arm is not written. The second
row is only possible because the tokamak device slot has occupants now; see
`_CHANGES_A_SLOT`'s note for why that is the result rather than a regression.

`blktmodel`/`blkttype`: each feeds a *joint* dispatch (`.fwbs.blktmodel` x
`.heat_transport.ipowerflow`, and x `.fwbs.blkttype`) alongside a second switch, and the
arm the pair selects has no occupant. `blktmodel == 1` is `blanket_neutronics()` in
*both* dispatches (arm 0 of each); the shield-power slot is resolved first, so that is
the reason that surfaces, and the key recorded here says so. `blkttype == 1` is the
liquid-breeder sub-arm, which only exists inside `blktmodel == 0`.

**Both keys changed with the arm-index fix**, and one of them was previously wrong about
what it was proving: `blktmodel = 1` used to refuse at the *mass* slot citing the
liquid-breeder reason, which happened only after the shield-power slot had already
chosen `BlanketShieldPowerExponential` -- a node written for `ipowerflow == 0`, not for
`blktmodel == 1`. The test passed while the assembly was selecting the wrong node.

Either way this is still proof the value is consulted: an ignored value could not turn a
successful assembly into a documented refusal. So this list is not a lesser check than
`_CHANGES_A_SLOT`, only a differently-shaped one."""


@pytest.mark.parametrize(
    ("switch", "value", "probe"), _CHANGES_A_SLOT, ids=[c[0] for c in _CHANGES_A_SLOT]
)
def test_factory_switches_actually_change_the_assembled_machine(
    tmp_path, switch, value, probe
):
    """One of `FACTORY_READ_SWITCHES` per case: assembling a machine from a copy of
    `REFERENCE_INPUT_FILE` with only `switch` changed produces a *different* occupant at
    the slot `probe` inspects than `REFERENCE_MACHINE` has. This is the dynamic half of
    confirming `machine_from_indat` reads what it claims to -- a grep for `switches.get`
    shows the name appears in the source, this shows changing the file's value actually
    changes what gets built, so the read is not dead code.
    """
    baseline = probe(REFERENCE_MACHINE)
    indat = _indat_with_override(tmp_path, switch, value)
    perturbed = machine_from_indat(indat)
    assert probe(perturbed) != baseline, (
        f"setting {switch} = {value} did not change the probed slot's occupant type "
        f"(still {baseline!r}) -- machine_from_indat may not actually be reading "
        f"{switch}."
    )


def test_a_process_forced_switch_cannot_move_the_machine(tmp_path):
    """`i_plasma_pedestal`'s inverse of the test above: overriding it in the IN.DAT
    changes **nothing** in the assembled machine, because PROCESS overwrites it.

    `st_init` (`process/models/stellarator/initialization.py:31`) assigns
    `data.physics.i_plasma_pedestal = 0` on every `istell != 0` run, so a file saying
    `i_plasma_pedestal = 1` is a file PROCESS runs with parabolic profiles. The factory
    used to read that `1` and assemble `ProfileParameterisationPedestal` for it. The
    machines are compared whole, not at the one slot: a forced switch must not move
    *any* part of the tree, and comparing the repr is the cheapest total comparison.

    Deliberately not a refusal. Rejecting the file would make this port decline an input
    PROCESS runs happily; reproducing the forcing is what modelling the run means. That
    the value is ignored is recorded in `SWITCH_INVENTORY`'s `ForcedByProcess` entry and
    in `ST_INIT_I_PLASMA_PEDESTAL`'s own docstring, so it is ignored *visibly*.
    """
    for value in (0, 1):
        perturbed = machine_from_indat(
            _indat_with_override(tmp_path, "i_plasma_pedestal", value)
        )
        assert repr(perturbed) == repr(REFERENCE_MACHINE), (
            f"i_plasma_pedestal = {value} moved the assembled machine, but st_init "
            f"forces the field to {ST_INIT_I_PLASMA_PEDESTAL} on every stellarator run "
            "-- the factory is reading the file again."
        )


def _static_fields_named(obj, name, path="", seen=None):
    """Every `eqx.field(static=True)` field called `name` anywhere in `obj`, as
    (attribute path, value) pairs.

    `_field_names`' sibling, and the reason it is separate: that one proves a name is
    *absent*, this one collects the values a name is held at so they can be compared
    against each other and against what the factory decided.

    Yields
    ------
    tuple[str, object]
        The attribute path to the field, and the value held there.
    """
    if seen is None:
        seen = set()
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            value = getattr(obj, f.name)
            if f.name == name and f.metadata.get("static"):
                yield (f"{path}.{f.name}", value)
            yield from _static_fields_named(value, name, f"{path}.{f.name}", seen)
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            yield from _static_fields_named(item, name, path, seen)
    elif isinstance(obj, dict):
        for item in obj.values():
            yield from _static_fields_named(item, name, path, seen)


COHERENCE_CASES = (
    # (switch, what the IN.DAT says, what the factory must resolve it to)
    ("i_p_coolant_pumping", 0, 0),
    ("i_p_coolant_pumping", 1, 1),
    ("i_blkt_dual_coolant", 0, 0),
    ("i_blkt_dual_coolant", 1, 1),
    ("i_blkt_dual_coolant", 2, 2),
    ("i_thermal_electric_conversion", 0, 0),
    ("i_thermal_electric_conversion", 1, 1),
    ("i_thermal_electric_conversion", 2, 2),
    ("i_thermal_electric_conversion", 3, 3),
    ("i_thermal_electric_conversion", 4, 4),
)
"""Every (switch, file value) pair that assembles, for the switches that both drive a
slot **and** appear as a static field on some occupant -- with the value the factory is
required to resolve it to.

The third column is the switch's own value everywhere except `i_plasma_pedestal`, whose
resolved value is `ST_INIT_I_PLASMA_PEDESTAL` at either file value: PROCESS overwrites
it, so the tree must hold what PROCESS will hold and not what the file said. That is the
one place these three columns are allowed to disagree, and it is the reason the table
has three columns rather than two.

`blktmodel`/`blkttype`/`istell`/`isthtr`/`i_bldgs_size`/`i_cost_model` are absent because
no occupant carries a static field of that name --
`test_every_factory_switch_with_a_static_field_is_covered` is what keeps that true."""


@pytest.mark.parametrize(
    ("switch", "file_value", "resolved"),
    COHERENCE_CASES,
    ids=[f"{sw}={v}" for sw, v, _r in COHERENCE_CASES],
)
def test_no_slot_contradicts_a_factory_switch(tmp_path, switch, file_value, resolved):
    """**The mechanical coherence check.** For a switch that decides which node exists,
    no *other* slot in the assembled machine may hold a different answer to the same
    question.

    Walks the whole machine `machine_from_indat` builds for one override and compares
    every static field named after the switch against the value the factory resolved.
    One walk per case beats one assertion per known site: it needs no list of the sites,
    so a site added later is covered the day it is added.

    This is the test that would have caught all four of `switch_kwarg_survey.md` band
    (a)'s live incoherences, none of which any gate could see on the reference machine:

    * `i_tf_sup = 0` built `TfPowerResistive` at `power.tf_power` and left five nodes
      (`power.cryo_q_nuc_step`, `cryo_q_loads_step`, `cryo_loads`,
      `availability.electric_production`, `cplife_avail`) saying `SUPERCONDUCTING`;
    * `ipowerflow = 0` built `AFwTotalNoPowerflow` and `BlanketShieldPowerExponential`
      and left `stellarator.neutron_wall_load` /
      `radiated_wall_load_and_fraction` saying `COMPREHENSIVE_2014`;
    * `ireactor = 0` built `PowerProfilesOverTime` and kept a `CostOfElectricity`
      carrying `ireactor = 1` -- a node whose own `__check_init__` says it must not
      exist on that value;
    * `i_plasma_pedestal = 1` built the pedestal arm for a run `st_init` forces to
      parabolic.
    """
    machine = machine_from_indat(_indat_with_override(tmp_path, switch, file_value))
    held = dict(_static_fields_named(machine, switch))
    contradictions = {
        path: int(value) for path, value in held.items() if int(value) != resolved
    }
    assert not contradictions, (
        f"{switch} = {file_value} resolves to {resolved}, but the assembled machine "
        f"holds {contradictions} -- a slot is answering a switch the factory already "
        "answered. A switch that decides which node exists belongs in one place; where "
        "an occupant genuinely needs the value, thread it from machine_from_indat "
        "rather than transcribing it."
    )


def test_every_factory_switch_with_a_static_field_is_covered():
    """`COHERENCE_CASES` covers every factory-read switch that any occupant also spells
    as a static field -- so a slot that newly hardcodes one cannot go unchecked.

    Doubles as the positive control for `_static_fields_named`: if the walker found
    nothing, the intersection would be empty and this would fail rather than letting
    every case above pass vacuously.
    """
    names = set(_field_names(REFERENCE_MACHINE))
    forced = {
        name
        for name, entry in SWITCH_INVENTORY.items()
        if isinstance(entry, ForcedByProcess)
    }
    should_be_covered = (FACTORY_READ_SWITCHES | forced) & names
    covered = {switch for switch, _v, _r in COHERENCE_CASES}
    assert should_be_covered == covered, (
        f"uncovered: {sorted(should_be_covered - covered)}; stale: "
        f"{sorted(covered - should_be_covered)}. A switch that both decides a slot and "
        "appears as a static kwarg is exactly the shape that goes incoherent -- add "
        "every value of it that assembles to COHERENCE_CASES."
    )


@pytest.mark.parametrize(
    ("switch", "value", "unported_key"),
    _CAUSES_A_REFUSAL,
    ids=[c[0] for c in _CAUSES_A_REFUSAL],
)
def test_factory_switches_that_only_prove_themselves_by_a_refusal(
    tmp_path, switch, value, unported_key
):
    """The four switches from `_CAUSES_A_REFUSAL`: `REFERENCE_MACHINE`
    itself assembles without error, but overriding just `switch` (holding the other
    member of its joint key, where it has one, at the reference file's value) selects an
    arm `UNPORTED` documents as not-yet-ported -- so `machine_from_indat` raises
    `NotImplementedError` where it previously did not. An ignored switch could not turn
    a successful assembly into a documented refusal, so this is still real evidence the
    value is read, just shaped as "assembly now fails" rather than "a different node was
    selected".
    """
    indat = _indat_with_override(tmp_path, switch, value)
    with pytest.raises(NotImplementedError, match=re.escape(UNPORTED[unported_key])):
        machine_from_indat(indat)
