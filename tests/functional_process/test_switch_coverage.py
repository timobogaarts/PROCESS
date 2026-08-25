"""Pins a known gap in `total_process.py`'s switch handling -- deliberately, not a fix.

`machine_from_indat`'s own docstring calls itself **"the only place in this port an
`i_*` integer is ever read"**. It is not: `REFERENCE_INPUT_FILE`
(`tests/regression/input_files/stellarator_helias.IN.DAT`) explicitly sets seven
switches the factory never looks at --

    i_p_coolant_pumping = 1     i_thermal_electric_conversion = 2   i_plasma_ignited = 1
    i_tf_sc_mat = 1             i_confinement_time = 38             i_rad_loss = 1
    i_plant_availability = 0

Six of the seven are instead hand-transcribed as a static constructor kwarg on one or
more slots of `REFERENCE_MACHINE`, matching the file's value *by coincidence of careful
authorship*, not by anything that would notice if a value drifted or a slot were missed.
The seventh, `i_plant_availability`, is not transcribed anywhere at all: PROCESS's own
dispatch on it never runs, because `Stellarator.run()` calls `self.availability.avail()`
directly (`stellarator.py:175`), bypassing `.costs.i_plant_availability` entirely.

This module is the inventory of that gap, `SWITCH_INVENTORY`, plus tests that keep the
inventory honest against the live tree and the live file. **None of these tests fix the
gap** -- `test_hardcoded_values_agree_with_the_reference_file` passes today because the
six transcriptions are, in fact, all correct today. What changes is that a future drift
(a transcription going stale, a new switch joining the six-transcribed/one-bypassed set
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

from functional_process.total_process import (
    REFERENCE_INPUT_FILE,
    REFERENCE_MACHINE,
    UNPORTED,
    machine_from_indat,
    switches_from_indat,
)


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
class Bypassed:
    """Set in the file, consulted by nothing: the PROCESS dispatch this switch would
    drive is itself skipped by a hardcoded solve-time call, so there is no slot for the
    value to be transcribed onto, right or wrong. `value` is still recorded, purely so a
    future change to what the reference file sets it to is visible as a diff here rather
    than silent.
    """

    value: int
    reason: str


SWITCH_INVENTORY: dict[str, ReadByFactory | Hardcoded | Bypassed] = {
    # --- Read by machine_from_indat, and (incidentally) set explicitly by the
    # reference file rather than left at its default. `switches_from_indat` line
    # references are `stellarator_helias.IN.DAT`'s.
    "istell": ReadByFactory(6),  # :137
    "isthtr": ReadByFactory(1),  # :138 -- equals the bare default, set anyway
    "i_plasma_pedestal": ReadByFactory(0),  # :127
    "i_cost_model": ReadByFactory(0),  # :248
    "ireactor": ReadByFactory(1),  # :260 -- equals the bare default, set anyway
    # --- Hand-transcribed as a static kwarg, matching the file today, checked nowhere.
    "i_p_coolant_pumping": Hardcoded(  # :198
        value=1,
        slots=(
            Slot(
                "power.component_thermal_powers.i_p_coolant_pumping",
                lambda m: m.power.component_thermal_powers.i_p_coolant_pumping,
            ),
            Slot(
                "power.delta_eta_step.i_p_coolant_pumping",
                lambda m: m.power.delta_eta_step.i_p_coolant_pumping,
            ),
            Slot(
                "power.p_fw_div_heat_deposited_mw_step.i_p_coolant_pumping",
                lambda m: m.power.p_fw_div_heat_deposited_mw_step.i_p_coolant_pumping,
            ),
            Slot(
                "power.p_fw_blkt_coolant_pump_mw_step.i_p_coolant_pumping",
                lambda m: m.power.p_fw_blkt_coolant_pump_mw_step.i_p_coolant_pumping,
            ),
            Slot(
                "availability.electric_production.i_p_coolant_pumping",
                lambda m: m.availability.electric_production.i_p_coolant_pumping,
            ),
        ),
    ),
    "i_thermal_electric_conversion": Hardcoded(  # :203
        value=2,
        slots=(
            Slot(
                "power.component_thermal_powers.i_thermal_electric_conversion",
                lambda m: m.power.component_thermal_powers.i_thermal_electric_conversion,
            ),
            Slot(
                "power.delta_eta_step.i_thermal_electric_conversion",
                lambda m: m.power.delta_eta_step.i_thermal_electric_conversion,
            ),
            Slot(
                "power.eta_turbine_step.i_thermal_electric_conversion",
                lambda m: m.power.eta_turbine_step.i_thermal_electric_conversion,
            ),
            Slot(
                "power.temp_turbine_coolant_in_step.i_thermal_electric_conversion",
                lambda m: (
                    m.power.temp_turbine_coolant_in_step.i_thermal_electric_conversion
                ),
            ),
        ),
    ),
    "i_plasma_ignited": Hardcoded(  # :126
        value=1,
        slots=(
            Slot(
                "stellarator.heating_and_radiation_power.i_plasma_ignited",
                lambda m: m.stellarator.heating_and_radiation_power.i_plasma_ignited,
            ),
            Slot(
                "physics.confinement_time.model.i_plasma_ignited",
                lambda m: m.physics.confinement_time.model.i_plasma_ignited,
            ),
            Slot(
                "physics.plasma_composition.i_plasma_ignited",
                lambda m: m.physics.plasma_composition.i_plasma_ignited,
            ),
        ),
    ),
    "i_tf_sc_mat": Hardcoded(  # :235
        value=1,
        slots=(
            Slot(
                "stellarator.coils.winding_pack_intersect_inputs.i_tf_sc_mat",
                lambda m: m.stellarator.coils.winding_pack_intersect_inputs.i_tf_sc_mat,
            ),
        ),
    ),
    "i_confinement_time": Hardcoded(  # :129
        value=38,
        slots=(
            Slot(
                "physics.confinement_time.model.i_confinement_time",
                lambda m: m.physics.confinement_time.model.i_confinement_time,
            ),
        ),
    ),
    "i_rad_loss": Hardcoded(  # :128
        value=1,
        slots=(
            Slot(
                "physics.confinement_time.model.i_rad_loss",
                lambda m: m.physics.confinement_time.model.i_rad_loss,
            ),
        ),
    ),
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
rather than left at their default), six hand-transcribed as static kwargs (matching the
file today, checked nowhere before this module), and one -- `i_plant_availability` --
set by the file and consulted by no code path in this port at all.

Deliberately excludes the five further switches `machine_from_indat` reads that this
particular file leaves at their PROCESS default (`ipowerflow`, `blktmodel`, `blkttype`,
`i_tf_sup`, `i_bldgs_size`, see `FACTORY_READ_SWITCHES` below) -- they are not switches
"the reference IN.DAT sets", so they have no value to pin here. Also excludes ten more
integers `switches_from_indat` picks up from this same file that are not model-tree
switches at all -- see `OUT_OF_SCOPE_INTEGERS`.
"""

FACTORY_READ_SWITCHES = frozenset({
    "istell",
    "isthtr",
    "ipowerflow",
    "blktmodel",
    "blkttype",
    "i_plasma_pedestal",
    "i_cost_model",
    "i_tf_sup",
    "i_bldgs_size",
    "ireactor",
})
"""Every switch name `machine_from_indat` itself reads via `switches.get(...)` --
confirmed by grep (`switches.get("<name>"` / `pick("<name>"`, ten hits, no more) and
exercised dynamically, one by one, by
`test_factory_switches_actually_change_the_assembled_machine` below -- so this list is
checked against the factory's real behaviour, not merely its source text.
Superset of `SWITCH_INVENTORY`'s five `ReadByFactory` entries: the other five names here
(`ipowerflow`, `blktmodel`, `blkttype`, `i_tf_sup`, `i_bldgs_size`) are read the same way
but are not set by `REFERENCE_INPUT_FILE`, so `SWITCH_INVENTORY` -- "switches the file
sets" -- has no entry for them.
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
    (
        "i_plasma_pedestal",
        1,
        lambda m: type(m.physics.profiles.parameterisation).__name__,
    ),
    ("i_tf_sup", 0, lambda m: type(m.power.tf_power).__name__),
    ("i_bldgs_size", 1, lambda m: type(m.buildings.sizing).__name__),
    ("ireactor", 0, lambda m: type(m.availability.electric_production).__name__),
    (
        "ipowerflow",
        0,
        lambda m: type(m.stellarator.fwbs.blanket_shield_power).__name__,
    ),
)
"""Six of the ten `FACTORY_READ_SWITCHES`: each has a second registered occupant that
`_indat_with_override` can select on its own, so the "really reads it" proof is
"the assembled tree's occupant type changes".

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
    ("istell", 0, ("istell", 0)),
    ("i_cost_model", 1, ("i_cost_model", 1)),
    ("blktmodel", 1, ("blktmodel_ipowerflow", 0)),
    ("blkttype", 1, ("blktmodel_blkttype", 1)),
)
"""The remaining four `FACTORY_READ_SWITCHES`, refused for two different reasons.

`istell`/`i_cost_model`: the switch has exactly one registered occupant, so its *other*
value cannot select a second one -- there is none to select. `istell == 0` is a tokamak,
which this tree has no namespace for, and `i_cost_model == 1` is KOVARI_2014, unported.
Both were spelled as a slot holding `None` until the tree stopped carrying optional
slots; they are refusals now, which is why they moved here from `_CHANGES_A_SLOT`.

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
