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

import dataclasses
import functools
import os
import re
from pathlib import Path

import equinox as eqx
import pytest
from cottax.interfaces.pytree_namespace_module import spell_flat, to_graph

from functional_process import boundary as fp_boundary
from functional_process.boundary import orphaned_by
from functional_process.indat import (
    ACPOW,
    AVAIL,
    BLANKET_MASSES,
    BLANKET_SHIELD_POWER,
    BUILDING_SIZING,
    COILS_MASS_MATERIAL,
    CONFINEMENT_SCALING,
    CONFINEMENT_TAIL,
    COST_MODEL,
    COST_OF_ELECTRICITY,
    CPLIFE,
    CRYO_LOADS,
    CRYO_Q_LOADS,
    CRYO_Q_NUC,
    ELECTRIC_PRODUCTION,
    ENERGY_STORAGE,
    ETA_TURBINE,
    ETATH_LIQ,
    FAST_ALPHA_BETA,
    FW_AREA,
    GRAPH,
    HEATING,
    HEATING_AND_RADIATION_POWER,
    NEUTRON_WALL_LOAD,
    P_FW_BLKT_COOLANT_PUMP,
    P_FW_DIV_HEAT_DEPOSITED,
    PLASMA_COMPOSITION,
    PLASMA_POWER_LOSS,
    PROFILE_PARAMETERISATION,
    RADIATED_WALL_LOAD,
    REFERENCE_INPUT_FILE,
    REFERENCE_MACHINE,
    REFERENCE_MACHINE_SWITCHES,
    ST_INIT_I_PLASMA_PEDESTAL,
    TEMP_TURBINE_COOLANT_IN,
    TF_MAGNET_COST_SUPERCONDUCTING,
    TF_POWER,
    UNPORTED,
    WINDING_PACK_MATERIAL,
    AFwTotalNoPowerflow,
    AFwTotalWithPowerflow,
    PedestalSeparatrixDensities,
    ProfileParameterisationPedestal,
    graph_for,
    machine_from_indat,
    switches_from_indat,
)
from functional_process.cottax.pfcoil.namespace import PFCoilSphericalTokamak
from functional_process.cottax.tfcoil.superconducting import (
    CiccAveragedTurnGeometryFromCurrentPerTurn,
    CiccIntegerTurnGeometry,
    SuperconductingTfWpGeometryDoubleRectangular,
    SuperconductingTfWpGeometryRectangular,
)
from functional_process.total_process import TokamakProcess
from process.data_structure.physics_variables import (
    ConfinementRadiationLossModel,
    ConfinementTimeModel,
)


def occupant_class(entry):
    """The class a registry entry builds, seeing through a settings-carrying partial."""
    return entry.func if isinstance(entry, functools.partial) else entry


def _plain(entry):
    return entry()


def _maybe_absent(entry):
    """A registry whose `None` entry means "nothing owns these fields" -- the shape
    `_audit/next_steps.md` §14.4 established for `inuclear` and this wave extended to
    seven more slots.
    """
    return None if entry is None else entry()


def _costs(entry):
    """`Costs` has six slots of its own -- `cost_of_electricity`, whose occupant may
    be `None`, `energy_storage_cost`, `tf_magnet_cost_superconducting`, and
    `reactor_structure_cost`, `pf_coil_power_conditioning_cost` and `pf_magnet_cost`,
    all three `None` here -- so it cannot be default-constructed. Built with the
    reference machine's.

    The third joined when `.costs.supercond_cost_model` stopped being an
    `eqx.field(static=True)` (`_audit/next_steps.md` §14.2); the last three when
    Accounts 221.4, 225.2 and 222.2 came back as *device*-decided slots (2026-08-30),
    which is why all three are `None` on the reference machine -- a stellarator has no
    reactor structure to cost, no PF coil system to cost, and never calls `Power.run`,
    so it has no PF coil power supply to condition either.
    """
    return entry(
        cost_of_electricity=REFERENCE_MACHINE.costs.cost_of_electricity,
        energy_storage_cost=REFERENCE_MACHINE.costs.energy_storage_cost,
        tf_magnet_cost_superconducting=(
            REFERENCE_MACHINE.costs.tf_magnet_cost_superconducting
        ),
        reactor_structure_cost=REFERENCE_MACHINE.costs.reactor_structure_cost,
        pf_coil_power_conditioning_cost=(
            REFERENCE_MACHINE.costs.pf_coil_power_conditioning_cost
        ),
        pf_magnet_cost=REFERENCE_MACHINE.costs.pf_magnet_cost,
    )


def _profile_parameterisation(entry):
    """The parabolic arm has a slot of its own -- `ecrh_density_limit`, which is the
    `EcrhDensityLimit` node on a stellarator and `None` on a tokamak, because
    `st_d_limit_ecrh` is reached only from `st_phys`. So it cannot be
    default-constructed either; built with the reference (stellarator) machine's, which
    is the arm these swap tests are about.

    The pedestal arm has a slot of its own too since 2026-08-27 --
    `pedestal_separatrix`, `.physics.i_nd_plasma_pedestal_separatrix`'s two inverse
    occupants, a switch nested under this one. Built with PROCESS's own default
    (`GREENWALD_FRACTION`), which is what both reference files select; the swap tests
    here are about the *outer* switch, and `test_a_refused_value_says_why` covers the
    inner one on its own.
    """
    if entry is ProfileParameterisationPedestal:
        return entry(pedestal_separatrix=PedestalSeparatrixDensities())
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
        # The eight-arm slot, and the one whose occupant decides whether the coils
        # block is a cycle: only the Bi-2212 arm reads `.tfcoil.j_tf_wp`
        # (`_audit/next_steps.md` §14.5).
        "i_tf_sc_mat",
        WINDING_PACK_MATERIAL,
        lambda m: m.stellarator.coils.winding_pack_intersect_inputs,
        _plain,
    ),
    (
        "i_plasma_pedestal",
        PROFILE_PARAMETERISATION,
        lambda m: m.physics.profiles.parameterisation,
        _profile_parameterisation,
    ),
    ("i_bldgs_size", BUILDING_SIZING, lambda m: m.buildings.sizing, _plain),
    ("i_tf_sup", TF_POWER, lambda m: m.power.tf_power, _plain),
    (
        # Five arms now, not two: `_electric_production_arm` folds `ireactor` together
        # with the two *joint* conditions that were four static kwargs on the reactor
        # occupant (`_audit/next_steps.md` §14.2).
        "electric_production_arm",
        ELECTRIC_PRODUCTION,
        lambda m: m.availability.electric_production,
        _plain,
    ),
    (
        "ireactor_ipnet_itart",
        COST_OF_ELECTRICITY,
        lambda m: m.costs.cost_of_electricity,
        _plain,
    ),
    (
        "blktmodel_ipowerflow_i_p_coolant_pumping",
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
        _plain,
    ),
    (
        "supercond_cost_model",
        TF_MAGNET_COST_SUPERCONDUCTING,
        lambda m: m.costs.tf_magnet_cost_superconducting,
        _plain,
    ),
    # --- the slots this wave's switch conversion created -------------------------
    #
    # Every one of them was a static kwarg on a node until `_audit/next_steps.md`
    # §14.2. They are listed here rather than left out because this table is what
    # `test_every_registered_occupant_assembles` and the swap contract iterate: a slot
    # missing from it is a slot nothing checks.
    ("i_beta_fast_alpha", FAST_ALPHA_BETA, lambda m: m.physics.fast_alpha_beta, _plain),
    (
        "i_plasma_ignited",
        PLASMA_COMPOSITION,
        lambda m: m.physics.plasma_composition,
        _plain,
    ),
    (
        "i_pflux_fw_neutron_ipowerflow",
        NEUTRON_WALL_LOAD,
        lambda m: m.stellarator.neutron_wall_load,
        _plain,
    ),
    (
        "i_pflux_fw_neutron_ipowerflow",
        RADIATED_WALL_LOAD,
        lambda m: m.stellarator.radiated_wall_load_and_fraction,
        _plain,
    ),
    (
        "i_plasma_ignited_stellarator",
        HEATING_AND_RADIATION_POWER,
        lambda m: m.stellarator.heating_and_radiation_power,
        _plain,
    ),
    (
        "i_tf_sc_mat_mass",
        COILS_MASS_MATERIAL,
        lambda m: m.stellarator.coils.coils_mass,
        _plain,
    ),
    (
        "i_pf_energy_storage_source",
        ACPOW,
        lambda m: m.power.acpow,
        _plain,
    ),
    ("eta_turbine_arm", ETA_TURBINE, lambda m: m.power.eta_turbine, _maybe_absent),
    ("secondary_cycle_liq", ETATH_LIQ, lambda m: m.power.etath_liq, _maybe_absent),
    (
        "temp_turbine_coolant_in_arm",
        TEMP_TURBINE_COOLANT_IN,
        lambda m: m.power.temp_turbine_coolant_in,
        _maybe_absent,
    ),
    (
        "p_fw_div_heat_deposited_arm",
        P_FW_DIV_HEAT_DEPOSITED,
        lambda m: m.power.p_fw_div_heat_deposited_mw,
        _maybe_absent,
    ),
    (
        "p_fw_blkt_coolant_pump_arm",
        P_FW_BLKT_COOLANT_PUMP,
        lambda m: m.power.p_fw_blkt_coolant_pump_mw,
        _maybe_absent,
    ),
    ("cryo_q_loads_arm", CRYO_Q_LOADS, lambda m: m.power.cryo_q_loads, _maybe_absent),
    ("cryo_loads_arm", CRYO_LOADS, lambda m: m.power.cryo_loads, _plain),
    ("ibkt_life", AVAIL, lambda m: m.availability.avail, _plain),
    ("cplife_arm", CPLIFE, lambda m: m.availability.cplife_avail, _maybe_absent),
]

SINGLE_FIELDS = [
    f
    for f, _r, _w, _b in SLOTS
    if not f.startswith("blktmodel_")
    and not f.endswith("_arm")
    and f
    not in (
        "ireactor_ipnet_itart",
        "i_pflux_fw_neutron_ipowerflow",
        "i_plasma_ignited_stellarator",
        "i_tf_sc_mat_mass",
    )
]
"""The slots addressed by one integer -- the joint keys are derived, not written.

`ireactor_ipnet_itart` joins the two `blktmodel_*` keys here:
`_cost_of_electricity_arm` turns `.costs.ireactor`, `.costs.ipnet` and `.physics.itart`
into one arm index, so there is no single integer of that name for an IN.DAT to set."""

TOKAMAK_BASELINE_INDAT = {
    "i_cost_model": 0,
    "i_hcd_primary": 10,
    "i_p_coolant_pumping": 3,
    "i_pulsed_plant": 1,
    "pulsetimings": 0,
    "i_bootstrap_current": 4,
    "i_density_limit": 7,
    "n_pf_coil_groups": 4,
    "i_pf_location": "2,2,3,3",
    "n_pf_coils_in_group": "1,1,2,2",
    "i_pf_superconductor": 3,
}
"""The least an IN.DAT must say for `machine_from_indat` to build a **tokamak**.

Eleven entries now, every one of them a PROCESS default this port refuses: the 2015
cost model, ITER neutral beam heating, mechanical coolant pumping, a continuous plant,
`pulsetimings = 1` once the plant is pulsed -- and, since waves 2/3's consolidation
filled the last eleven `Tokamak` slots, the Wilson bootstrap scaling
(`i_bootstrap_current = 3`), the ASDEX-New density limit (`i_density_limit = 8`), the
three-group PF coil topology (`n_pf_coil_groups = 3` with the default location
pattern) and the ITER-Nb3Sn PF conductor (`i_pf_superconductor = 1`), each replaced by
the reference run's own value. Written as data rather than
as a curated file, so that a case about one field fails on that field. The two list
entries are strings because that is how an IN.DAT spells them.

`large_tokamak_eval.IN.DAT` sets every one of these explicitly, which is the useful
sanity check on this dict: it is the minimum, and a real conventional tokamak input
file already exceeds it.
"""

DERIVED_UNPORTED_KEYS = {
    # Arm indices, not switch values. Each is exercised through the integers its `_*_arm`
    # function reads -- the same reason the three joint keys below are skipped.
    "centrepost_neutronics_arm",
    "cicc_turn_geometry_arm",
    "croco_turn_geometry_arm",
    "divertor_geometry_arm",
    "divertor_heat_load_arm",
    "first_wall_arm",
    "hcd_primary_powers_arm",
    "pf_coil_system_arm",
    "plasma_geometry_arm",
    "pulse_ramp_times_arm",
    "r_cp_top_arm",
    "structure_arm",
    "surface_poloidal_field_arm",
    "tf_coil_shape_arm",
    "tf_field_and_force_arm",
    "tf_inboard_radii_arm",
    "tf_stress_arm",
    "vacuum_vessel_arm",
    # Per-slot names for a switch that is read at more than one slot with different
    # dispositions, so the key is the slot and not the integer.
    "i_tf_sup_build",
    "i_tf_inside_cs_vacuum_shield",
    "i_plasma_ignited_separatrix",
    # Two-switch keys whose *value* is a `(i_str_wp, i_tf_sc_mat)` pair, so no single
    # IN.DAT line selects one. Exercised through the two integers by
    # `test_superconducting.py`'s `test_the_two_superconductor_slots_are_total` and
    # `test_i_str_wp_zero_is_refused_end_to_end`, which is what this skip is trading
    # against.
    "i_str_wp_i_tf_sc_mat_cicc_sc_properties",
    "i_str_wp_i_tf_sc_mat_temp_margin",
    # The CroCo namespace's two, same shape and the same trade: totality over both
    # 2 x 9 products is `test_croco.py::test_every_unwritten_croco_material_is_refused_
    # with_a_reason`, and the end-to-end refusal is
    # `test_a_croco_machine_refuses_an_unwritten_tape_material`.
    "i_str_wp_i_tf_sc_mat_croco_sc_properties",
    "i_str_wp_i_tf_sc_mat_croco_temp_margin",
}
"""`UNPORTED` keys that no IN.DAT integer can select directly.

Three kinds, and the distinction is worth keeping visible: a value the factory
**derives** (`n_divertors`), an **arm index** several switches jointly select, and a
**per-slot name** for one integer that two slots answer differently (`.tfcoil.i_tf_sup`
decides `power.tf_power` *and* two `.tokamak.build` nodes; `.physics.i_plasma_ignited`
decides three slots in three subsystems; `.build.i_tf_inside_cs` decides both the CS-to-TF
radial slice and the vacuum-vessel one, and a file that sets it is refused at the earlier
of the two, with that slot's message -- so the later slot's refusal is unreachable
through the integer and is covered where the integer is,
`test_the_tf_inboard_radii_arms_are_refused_through_their_real_integers`).
None of the three is a thing a file can set, so the refusal is reached through the
integers it derives from -- which the survey and switch-coverage tests do.

**Every entry must still name a live `UNPORTED` refusal**, which
`test_the_skip_list_holds_no_entry_that_is_now_ported` checks. A skip whose refusal has
since been ported is dead weight that reads like coverage: it silently defers a case
that no longer exists, and the wave that ported the arm has no reason to look here.
Five entries had rotted that way by 2026-08-27 (`n_divertors`, `fw_blkt_vv_shape_arm`,
`nuclear_heating_renormalisation_arm`, `i_tf_shape_build`, `itart_hcpb`) and were
removed with that check; the consolidation brief had spotted two of the five by hand,
which is the argument for measuring it instead."""

TOKAMAK_ONLY_UNPORTED_FIELDS = {
    "i_beta_norm_max",
    "i_blanket_type",
    "i_bootstrap_current",
    "i_density_limit",
    "i_diamagnetic_current",
    "i_hcd_calculations",
    "i_hcd_primary",
    "i_hcd_secondary",
    "i_ind_plasma_internal_norm",
    "i_l_h_threshold",
    "i_len_sol_outboard_power_decay",
    "i_p_coolant_pumping",
    "i_pfirsch_schluter_current",
    "i_plasma_current",
    "i_plasma_geometry",
    "i_plasma_ignited_separatrix",
    "i_ecrh_wave_mode",
}
"""`UNPORTED` fields the **stellarator** branch never reads.

`machine_from_indat` resolves everything only a stellarator asks *below* the device
branch, and everything only a tokamak asks inside `_tokamak_device`, so a refusal in the
second is unreachable from a `istell = 6` file. These cases are written over
`TOKAMAK_BASELINE_INDAT` instead."""

NESTED_UNPORTED_COMPANIONS = {"i_ecrh_wave_mode": {"i_hcd_primary": 13}}
"""Switches whose `UNPORTED` refusal only exists inside another switch's value.

`i_ecrh_wave_mode` is the first: `_hcd_primary_efficiency` consults it only when
`i_hcd_primary == 13` (the wave-mode `if` sits *inside*
`electron_cyclotron_freethy`, `current_drive.py:1074-1079`), and the baseline sets
`10`, so a case about the wave mode must also select the value that nests it or the
integer is silently ignored -- which is PROCESS's own behaviour, not a test artefact.
The companion values are overlaid on `TOKAMAK_BASELINE_INDAT` below, under the field
being tested so the field under test still wins a clash."""

BASELINE_INDAT = {
    "istell": 6,
    "i_cost_model": 0,
    "i_plasma_ignited": 1,
    "i_p_coolant_pumping": 1,
}
"""The least an IN.DAT must say for `machine_from_indat` to get past the slots whose
PROCESS default is refused. Written into every temp file below so a test about one field
fails on that field and not on `istell`.

`i_plasma_ignited` joined them when the confinement node became slots: PROCESS's own
default is `NON_IGNITED`, whose head arm adds injected heating and therefore reads a
variable the written arm does not, so it is a real branch this port has not written.
Refusing is the same honest answer `istell` gives -- a machine assembled from the wrong
arm's reads is exactly the invented-edge defect the split removes.

`i_p_coolant_pumping` joined on 2026-08-31 for the identical reason and it is the
starkest of the three: `fwbs_variables.py:249`'s default is `2` (`MECHANICAL`), and
`stellarator.py:924-928` *raises* on that value -- **PROCESS itself will not run a
stellarator that leaves the switch alone**, so neither will the factory. `1`
(`FRACTION_OF_HEAT`) is `stellarator_helias.IN.DAT`'s own value and the arm the
reference machine takes."""


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


TOKAMAK_MACHINE = machine_from_indat(
    str(
        Path(fp_boundary.__file__).resolve().parent.parent
        / fp_boundary.TOKAMAK_INPUT_FILE
    )
)
"""`large_tokamak_eval.IN.DAT`'s machine -- the second base `SLOTS` never had.

`SLOTS` swaps occupants into `REFERENCE_MACHINE`, which is a **stellarator**, so no
tokamak-only slot can appear in it: `eqx.tree_at` has nowhere to put one. That is why
the tokamak's registries were not merely missing from the list, they were unreachable
from it, and the meta-tests skipped the whole group in silence rather than failing.
"""


def _croco_machine():
    """`large_tokamak_eval.IN.DAT` rewound onto the CroCo turn -- the **third** base.

    `TOKAMAK_MACHINE` and `REFERENCE_MACHINE` are cable-in-conduit and stellarator
    machines respectively, so no slot in either can hold a `CrocoSuperconductingTfCoil`
    occupant and `_slot_for` cannot derive the CroCo registries from them -- the same
    unreachability `TOKAMAK_MACHINE` was added to fix one device up. Built here rather
    than from an ST input file because neither tracked spherical tokamak assembles yet
    (the PF coil system and `i_tf_stress_model`), and a base has to be a machine.

    Two lines of difference: `i_tf_turn_type = 2` selects the namespace and
    `i_tf_sc_mat = 9` is the one tape material with an occupant.
    """
    import tempfile

    source = (
        Path(fp_boundary.__file__).resolve().parent.parent
        / fp_boundary.TOKAMAK_INPUT_FILE
    )
    text = "\n".join(
        "i_tf_sc_mat = 9" if line.startswith("i_tf_sc_mat") else line
        for line in source.read_text().splitlines()
    )
    path = Path(tempfile.mkdtemp()) / "croco.IN.DAT"
    path.write_text(text + "\ni_tf_turn_type = 2\n")
    return machine_from_indat(str(path))


CROCO_MACHINE = _croco_machine()
"""The CroCo tokamak, so `_slot_for` can reach the three `CROCO_*` registries."""


def _slots_source():
    """The text of the `SLOTS` literal itself -- *only* the literal.

    Read from this file rather than introspected because `SLOTS` holds its registries as
    objects, and the question is which ones are *named* there. Bounded at the closing
    bracket deliberately: an unbounded split takes the whole rest of the file with it,
    and then every registry named in a comment below would count as covered.
    """
    return Path(__file__).read_text().split("SLOTS = [")[1].split("\n]")[0]


def _slot_registries():
    """`{name: registry}` for every `indat.py` dict that maps a switch value to an
    occupant **class** -- measured off the module rather than listed, so a registry
    added by a porting wave is covered the day it lands.

    The filter is exactly "every value is a class or `None`", which is what makes an
    entry swappable into a slot. It excludes `UNPORTED` (values are reason strings),
    `ITERATION_VARIABLES` and `REFERENCE_MACHINE_SWITCHES` (values are data) without
    naming any of them.
    """
    from functional_process import indat

    return {
        name: value
        for name, value in vars(indat).items()
        if name.isupper()
        and isinstance(value, dict)
        and value
        and all(v is None or isinstance(v, type) for v in value.values())
    }


def _filled_slots(machine, prefix=()):
    """`{attribute path: occupant}` for every non-namespace slot of a machine tree."""
    from cottax.interfaces.pytree_namespace_module import ModelNamespace

    out = {}
    for field in dataclasses.fields(machine):
        if field.metadata.get("static"):
            continue
        value = getattr(machine, field.name)
        path = (*prefix, field.name)
        if isinstance(value, ModelNamespace):
            out.update(_filled_slots(value, path))
        else:
            out[path] = value
    return out


REGISTRY_SLOT_OVERRIDES = {
    # A *second* registry feeding a slot another registry already reaches: the ECRH
    # O-mode efficiency is chosen by `i_ecrh_wave_mode` **inside** `i_hcd_primary = 13`,
    # so its occupant is a `primary_efficiency` occupant and no separate slot exists for
    # the derivation to find.
    "HCD_PRIMARY_EFFICIENCY_FREETHY": ("tokamak", "current_drive", "primary_efficiency"),
}
"""Registry -> slot, for the cases `_slot_for` cannot derive. Kept to one."""

NAMESPACE_VALUED_REGISTRIES = {
    "BLANKET_MODEL": "the entry is the whole `CcfeHcpb` subsystem namespace",
    "CS_COIL": "the entry is the whole `CSCoil` subsystem namespace",
    "DEVICE": "the entry is the device itself (`TokamakProcess`/`StellaratorProcess`)",
    "HCD_CALCULATIONS": "the entry is the whole `TokamakCurrentDrive` namespace",
    "PF_COIL": "the entry is the whole `PFCoil` subsystem namespace",
}
"""Registries whose values are namespaces rather than node occupants.

Swapping one replaces a *subsystem*, not a slot's occupant, so the occupant cases below
cannot express it and say so instead of skipping quietly. Their arms are covered where
they are chosen -- `machine_from_indat`'s own refusal tests -- and a joint swap harness
for subsystems would be a different check, not a longer version of this one."""

REGISTRY_COMPANIONS = {
    # The spherical-tokamak renormalisation arms and the centrepost neutronics slot are
    # one choice in two places: `CentrepostNeutronicsAbsent` (the conventional
    # tokamak's occupant) **owns** `.fwbs.p_cp_shield_nuclear_heat_mw`, and so do arms
    # 2 and 3, so swapping either alone is a duplicate producer and cottax refuses it.
    # Measured, not assumed -- the refusal names both nodes. Same shape as
    # `NESTED_UNPORTED_COMPANIONS`: a case that only exists jointly is stated jointly.
    ("NUCLEAR_HEATING_RENORMALISATION", 2): (
        ("tokamak", "ccfe_hcpb", "centrepost_neutronics"),
        1,
        "CENTREPOST_NEUTRONICS",
    ),
    ("NUCLEAR_HEATING_RENORMALISATION", 3): (
        ("tokamak", "ccfe_hcpb", "centrepost_neutronics"),
        1,
        "CENTREPOST_NEUTRONICS",
    ),
}
"""`(registry, value) -> (companion slot, companion value, companion registry)`."""


def _build_occupant(name, entry, machine):
    """One occupant instance, or `None` for an absent one.

    Two registry entries need constructor arguments, and in both cases the arguments are
    *shapes* the machine already carries -- read off the tree rather than written here,
    so they cannot drift from the graph the occupant is swapped into.
    `SauterBootstrapCurrentFraction` needs the profile grid's length; `PF_MAGNET_COST`'s
    two arms need the PF loop's bounds and its conductor branch, which the occupant
    already standing in that slot has.
    """
    if entry is None:
        return None
    if name == "BOOTSTRAP_CURRENT" and entry.__name__.startswith("Sauter"):
        grid = machine.physics.profiles.profile_grid
        return entry(n_plasma_profile_elements=grid.n_plasma_profile_elements)
    if name == "PF_MAGNET_COST":
        standing = machine.costs.pf_magnet_cost
        # `iohcl` left this list on 2026-08-31: it is the registry's second *key* now,
        # not a kwarg (`_audit/switch_consultation_audit.md` §2), so every arm here
        # takes the same two fields.
        return entry(
            n_cs_pf_coils=standing.n_cs_pf_coils,
            i_pf_conductor=standing.i_pf_conductor,
        )
    return entry()


def _slot_for(name, registry, machine):
    """The attribute path of the slot `registry`'s occupants fill, or `None`.

    Derived twice over rather than typed: by **occupant type** (which slot currently
    holds an instance of one of the registry's classes), then, for a slot standing empty
    on this machine, by **name** (`DX_TF_SIDE_CASE_MIN` -> `dx_tf_side_case_min`, whose
    `False` arm is `None` and whose `True` arm is what the reference file does not take).
    A list of lambdas would have to be maintained by every future wave; this cannot go
    stale.
    """
    if name in REGISTRY_SLOT_OVERRIDES:
        return REGISTRY_SLOT_OVERRIDES[name]
    slots = _filled_slots(machine)
    classes = tuple(v for v in registry.values() if v is not None)
    for path, occupant in slots.items():
        if isinstance(occupant, classes):
            return path
    for path in slots:
        if path[-1] == name.lower():
            return path
    return None


def _derived_occupants():
    """Every registry arm not already in `SLOTS`, with the slot it fills."""
    from functional_process import indat

    in_slots = {
        name for name in _slot_registries() if re.search(rf"\b{name}\b", _slots_source())
    }
    cases = []
    for name, registry in sorted(_slot_registries().items()):
        if name in in_slots or name in NAMESPACE_VALUED_REGISTRIES:
            continue
        for base in (TOKAMAK_MACHINE, REFERENCE_MACHINE, CROCO_MACHINE):
            path = _slot_for(name, registry, base)
            if path is not None:
                break
        if path is None:
            continue
        for value, entry in registry.items():
            companion = REGISTRY_COMPANIONS.get((name, value))
            if companion is not None:
                companion = (
                    companion[0],
                    getattr(indat, companion[2])[companion[1]],
                )
            cases.append((name, path, value, entry, base, companion))
    return cases


DERIVED_OCCUPANTS = _derived_occupants()
"""The 135 registry arms `SLOTS` cannot reach, and where each one goes."""


@pytest.mark.parametrize(
    ("registry", "path", "value", "entry", "base", "companion"),
    DERIVED_OCCUPANTS,
    ids=[f"{n}={v}" for n, _p, v, _e, _b, _c in DERIVED_OCCUPANTS],
)
def test_every_derived_occupant_assembles(registry, path, value, entry, base, companion):
    """`test_every_registered_occupant_assembles`, for the slots `SLOTS` cannot hold.

    Same question -- swap one occupant, rebuild, get a non-empty graph -- against the
    tokamak base and with the slot derived rather than written down. This is the check
    the consolidation brief asked for as "`SLOTS` does not cover the tokamak TF
    registries", and the measurement made the ask bigger: it is **58 registries and 135
    arms**, most of the tokamak, not the eight TF ones.

    What it does *not* do is extend the swap-orphan pin (`reference_swaps.txt`) to a
    second device. That pin is generated evidence about which reads a swap orphans, and
    a tokamak half is a separate decision with its own regeneration -- deferred out
    loud rather than half-done.
    """
    occupant = _build_occupant(registry, entry, base)
    machine = eqx.tree_at(
        lambda m: functools.reduce(getattr, path, m),
        base,
        occupant,
        is_leaf=lambda x: x is None,
    )
    if companion is not None:
        companion_path, companion_entry = companion
        machine = eqx.tree_at(
            lambda m: functools.reduce(getattr, companion_path, m),
            machine,
            _build_occupant(registry, companion_entry, base),
            is_leaf=lambda x: x is None,
        )
    graph = to_graph(machine)
    assert graph.definitions, f"{registry} == {value} assembled an empty graph"


def test_no_slot_registry_is_covered_by_nothing():
    """Every switch-value-to-occupant registry in `indat.py` is exercised by one of the
    three cases above, or is named as a namespace with a reason.

    The failure this closes is the one the consolidation brief describes: a registry
    absent from `SLOTS` is not a failing test, it is **no test**, and the group stays
    silently uncovered for as long as nobody re-reads the list. Deriving the coverage
    from `indat.py`'s own contents turns "somebody must remember" into "a new registry
    fails until it is placed".
    """
    covered = {name for name, *_ in DERIVED_OCCUPANTS} | set(NAMESPACE_VALUED_REGISTRIES)
    covered |= {
        name for name in _slot_registries() if re.search(rf"\b{name}\b", _slots_source())
    }
    missing = sorted(set(_slot_registries()) - covered)
    assert not missing, (
        f"{missing} map switch values to occupants but no occupant case reaches them -- "
        f"add them to `SLOTS` (stellarator base), let `_slot_for` derive them, or name "
        f"them in `NAMESPACE_VALUED_REGISTRIES` with the reason"
    )


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
            machine = eqx.tree_at(
                where, REFERENCE_MACHINE, build(occupant), is_leaf=lambda x: x is None
            )
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
                handle.writelines(f"{arm} {path}\n" for path in paths)

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
       file gets is `i_cost_model = 1` (KOVARI_2014, unported), and it is not the last
       one in the way: `TOKAMAK_BASELINE_INDAT` is the full list of PROCESS defaults a
       tokamak has to override before this port will build one, and it has five entries.

       **The refusal this test used to assert moved, and the move is the result.** It
       was `i_plasma_ignited = 0` with `i_rad_loss = 1` -- PROCESS's own defaults, and
       the confinement head arm that adds injected heating. `tokamak_boundary.md`
       §"What blocked the real file" recorded that as the first thing a real conventional
       tokamak asks for that this port had not got. It has it now
       (`PlasmaPowerLossNonIgnitedCoreRadiation`), so the refusal a silent file gets is a
       different one, further down. That is what progress looks like from here: the
       *first* refusal moves, and the list of overrides a tokamak needs gets shorter.
    2. **The device is still resolved first.** A file whose *only* content is an
       `istell` the factory cannot answer reports `istell`, not `i_cost_model`, even
       though `i_cost_model`'s default is refused too and the constructor would reach it.
       That ordering is the property the old name was really guarding, and it is the half
       of the old test that had nothing to do with the tokamak.

       **The probe changed from a refusal to a typo, because there are no refused
       `istell` values left** (2026-08-30): `1`-`5`, the five machine presets, build
       `StellaratorProcess` now, so the only `istell` that can fail is one PROCESS has
       never had. `7` therefore raises `ValueError` from `_slot_occupant`'s "not a known
       value" branch rather than `NotImplementedError` from its `UNPORTED` branch. The
       error *class* is weaker evidence about the port's frontier and exactly as strong
       about the ordering, which is all this assertion was ever for -- a bare file whose
       `istell` were read second would report `i_cost_model == 1` instead, as the first
       assertion above shows it does when `istell` is absent.
    3. **The default device is the one PROCESS names.** Given only the switches whose
       PROCESS defaults this port refuses, a file that never mentions `istell` builds a
       `TokamakProcess` -- not a `StellaratorProcess`, and not an error. It is a real
       tokamak now rather than an empty one: fourteen of `Tokamak`'s twenty-five slots
       are filled, so this assertion exercises the whole tokamak factory and not just
       the device branch.
    """
    indat = tmp_path / "IN.DAT"
    indat.write_text("")
    with pytest.raises(NotImplementedError, match=re.escape("i_cost_model == 1")):
        machine_from_indat(indat)

    no_such_device = tmp_path / "PRESET.DAT"
    no_such_device.write_text("istell = 7\n")
    with pytest.raises(ValueError, match=re.escape("istell == 7")):
        machine_from_indat(no_such_device)

    silent_device = tmp_path / "TOK.DAT"
    silent_device.write_text(
        "".join(f"{f} = {v}\n" for f, v in TOKAMAK_BASELINE_INDAT.items())
    )
    machine = machine_from_indat(silent_device)
    assert type(machine) is TokamakProcess
    assert machine.tokamak.build is not None
    assert machine.tokamak.cicc_superconducting_tf_coil is not None


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


def test_a_switch_that_decides_two_slots_decides_both(tmp_path):
    """`i_tf_turns_integer` reaches *both* of its consequences, not just one.

    The bug class this closes (2026-08-27, `low_aspect_ratio_DEMO.IN.DAT`): the factory
    read `i_tf_turns_integer` for the `i_tf_wp_geom` `UNSET` resolution, so
    `machine_survey` truthfully reported *"the factory dispatches on it"* -- while the
    turn-geometry slot never consulted it and silently kept the averaged occupant. The
    port then computed a square 0.0568 m turn where PROCESS's converged answer is a
    0.0547 x 0.0591 m rectangle, 4e-2 on the turn dimensions and 42 % on
    `m_tf_coil_superconductor` through a near-cancellation. `next_steps.md` §14.11's
    failure mode, in its survey-blind variant: a switch is "dispatched on" only when
    **every** slot it decides is dispatched.

    Two assertions per arm, one per consequence, plus the real file that found it.
    """
    base = "".join(f"{f} = {v}\n" for f, v in TOKAMAK_BASELINE_INDAT.items())

    integer = tmp_path / "INTEGER.DAT"
    integer.write_text(base + "i_tf_turns_integer = 1\n")
    coil = machine_from_indat(integer).tokamak.cicc_superconducting_tf_coil
    assert type(coil.cicc_turn_geometry) is CiccIntegerTurnGeometry
    assert (
        type(coil.superconducting_tf_wp_geometry)
        is SuperconductingTfWpGeometryRectangular
    )

    averaged = tmp_path / "AVERAGED.DAT"
    averaged.write_text(base)  # `i_tf_turns_integer` unset: PROCESS's default `0`
    coil = machine_from_indat(averaged).tokamak.cicc_superconducting_tf_coil
    assert type(coil.cicc_turn_geometry) is CiccAveragedTurnGeometryFromCurrentPerTurn
    assert (
        type(coil.superconducting_tf_wp_geometry)
        is SuperconductingTfWpGeometryDoubleRectangular
    )

    demo = Path(REFERENCE_INPUT_FILE).parent / "low_aspect_ratio_DEMO.IN.DAT"
    coil = machine_from_indat(demo).tokamak.cicc_superconducting_tf_coil
    assert type(coil.cicc_turn_geometry) is CiccIntegerTurnGeometry


@pytest.mark.parametrize(("field", "value"), sorted(UNPORTED), ids=str)
def test_a_refused_value_says_why(tmp_path, field, value):
    """Asking for an unported value raises, and the message carries the recorded reason.

    The reason strings are audit content -- they moved verbatim out of the
    `Alternative(unported=...)` declarations this replaced. A refusal that did not name
    one would be indistinguishable from a value PROCESS never had.

    Every file is written over a baseline, because several of PROCESS's own defaults are
    themselves refused: without it every case here would fail on whichever of those the
    constructor reached first, rather than on the value under test. **Which baseline
    depends on the device**, since `machine_from_indat` resolves a stellarator's
    switches and a tokamak's in two disjoint branches -- see
    `TOKAMAK_ONLY_UNPORTED_FIELDS`.
    """
    if field in DERIVED_UNPORTED_KEYS or field.startswith((
        "blktmodel_",
        "i_plasma_ignited_i_",
        "i_pulsed_plant_",
    )):
        pytest.skip("derived key -- exercised through the integers it derives from")
    if field in TOKAMAK_ONLY_UNPORTED_FIELDS:
        indat = tmp_path / "TOK.DAT"
        indat.write_text(
            "".join(
                # A baseline entry may be a comma-list spelled as a string
                # (`i_pf_location`); everything else -- including the enum member
                # under test -- normalises through `int`.
                f"{f} = {v if isinstance(v, str) else int(v)}\n"
                for f, v in {
                    **TOKAMAK_BASELINE_INDAT,
                    **NESTED_UNPORTED_COMPANIONS.get(field, {}),
                    field: value,
                }.items()
            )
        )
    else:
        indat = write_indat(tmp_path, **{field: value})
    with pytest.raises(NotImplementedError, match=re.escape(f"{field} == {value}")):
        machine_from_indat(indat)


def test_the_skip_list_holds_no_entry_that_is_now_ported():
    """`DERIVED_UNPORTED_KEYS` may only name fields `UNPORTED` still refuses.

    The skip exists because no IN.DAT line selects a derived key directly. Once the arm
    behind such a key is *ported*, its `UNPORTED` rows go away and the entry stops
    skipping anything -- but it stays in the file, reading like a deliberate exemption
    for a case that no longer exists. Nothing else notices: the parametrisation is over
    `UNPORTED`, so a stale entry never runs and never fails.

    This is the guard for the rot rather than a periodic re-read of the list. It is
    deliberately one-directional -- a *new* derived key that needs skipping shows up as
    a failing refusal test, which is loud on its own.
    """
    refused = {field for field, _ in UNPORTED}
    stale = sorted(DERIVED_UNPORTED_KEYS - refused)
    assert not stale, (
        f"{stale} no longer name any `UNPORTED` refusal -- the arms behind them are "
        f"ported, so the skip defers nothing and should be deleted"
    )


def test_the_tf_inboard_radii_arms_are_refused_through_their_real_integers(tmp_path):
    """`tf_inboard_radii_arm`'s two refused arms, reached the way a file reaches them.

    The arm index itself is a `DERIVED_UNPORTED_KEYS` entry no file can set, so this is
    the per-integer coverage that skip defers to: `i_tf_inside_cs = 1` selects arm -1
    (TF inside the CS), still refused, over the tokamak baseline. Added with the
    cold-boundary wave's `TfInboardRadiiTfOutsideCs` (2026-08-27); the `-2`
    (`i_cs_precomp = 0`) case it originally asserted refused was ported the same day
    (`TfInboardRadiiNoCsPrecomp`, the ST frontier wave), so that integer is asserted to
    assemble instead -- the same file, one flip of one switch, both dispositions pinned.
    """
    for extra, arm in ((("i_tf_inside_cs", 1), -1),):
        indat = tmp_path / f"TOK_{extra[0]}.DAT"
        indat.write_text(
            "".join(
                f"{f} = {v}\n"
                for f, v in {**TOKAMAK_BASELINE_INDAT, extra[0]: extra[1]}.items()
            )
        )
        with pytest.raises(
            NotImplementedError, match=re.escape(f"tf_inboard_radii_arm == {arm}")
        ):
            machine_from_indat(indat)
    indat = tmp_path / "TOK_i_cs_precomp.DAT"
    indat.write_text(
        "".join(
            f"{f} = {v}\n"
            for f, v in {**TOKAMAK_BASELINE_INDAT, "i_cs_precomp": 0}.items()
        )
    )
    machine_from_indat(indat)  # ported arm -2: assembles, no refusal


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


SPHERICAL_TOKAMAK_PF_SWITCHES = {
    "iohcl": 0,
    "n_pf_coil_groups": 4,
    "i_pf_location": (2, 3, 3, 4),
    "n_pf_coils_in_group": (2, 2, 2, 2),
    "itart": 1,
    "itartpf": 1,
    "i_pf_current": 1,
    "i_pf_conductor": 0,
    "i_pf_superconductor": 9,
    "i_cs_superconductor": 1,
    "i_r_pf_outside_tf_placement": 1,
}
"""The PF coil system as both `spherical_tokamak_eval.IN.DAT` and `st_regression.IN.DAT`
set it, byte for byte. `i_tf_shape` is added per-test because it is an enum member."""

CONVENTIONAL_PF_SWITCHES = {
    "iohcl": 1,
    "n_pf_coil_groups": 4,
    "i_pf_location": (2, 2, 3, 3),
    "n_pf_coils_in_group": (1, 1, 2, 2),
    "itart": 0,
    "itartpf": 0,
    "i_pf_current": 1,
    "i_pf_conductor": 0,
    "i_pf_superconductor": 3,
    "i_cs_superconductor": 1,
    "i_r_pf_outside_tf_placement": 0,
}
"""`large_tokamak_eval.IN.DAT`'s, likewise."""


def test_both_ported_pf_coil_systems_deviate_on_nothing():
    """The two supported PF configurations are each accepted whole, and named.

    Until 2026-08-30 the spherical tokamaks' configuration deviated on **five** of the
    dimensions this function checks (`-1`, `-2`, `-3`, `-6`, `-7`) and this test pinned
    that tuple, with a docstring saying it would shrink the day one of the five was
    ported. All five are ported, so it is empty; what the test pins now is the pair of
    accepted configurations and the arm each resolves to.

    `-1` is gone from the dimensions altogether. `.build.iohcl` used to be refused when
    zero; it is now the *family selector* -- `-2`, `-6` and `-7` are each measured
    against the occupant set written for this machine's own `iohcl`, which is what keeps
    a mixed configuration (no solenoid, conventional `i_pf_location`) refused rather
    than silently reaching a namespace with no such occupant. That mix is the third
    assertion below.
    """
    from functional_process.indat import (
        TFCoilShapeModel,
        _pf_coil_system_arm,
        _pf_coil_system_deviations,
    )

    spherical = {
        **SPHERICAL_TOKAMAK_PF_SWITCHES,
        "i_tf_shape": TFCoilShapeModel.PICTURE_FRAME,
    }
    conventional = {
        **CONVENTIONAL_PF_SWITCHES,
        "i_tf_shape": TFCoilShapeModel.D_SHAPE,
    }
    assert _pf_coil_system_deviations(**spherical) == ()
    assert _pf_coil_system_arm(**spherical) == 2
    assert _pf_coil_system_deviations(**conventional) == ()
    assert _pf_coil_system_arm(**conventional) == 0

    # A configuration that borrows half of each is refused, and says so on every
    # dimension it borrowed. Without the family split this would pass three
    # independent membership tests and then look for occupants that do not exist.
    mixed = {**conventional, "iohcl": 0}
    assert _pf_coil_system_deviations(**mixed) == (-2, -6, -7)


def test_the_pf_coil_refusal_still_names_every_deviating_dimension(tmp_path):
    """A file that misses on several PF dimensions at once is told about all of them.

    `_pf_coil_system_arm` short-circuits -- it returns the first refused dimension and
    never evaluates the rest -- which is correct for *choosing* an occupant and wrong
    for *sizing* the work, and `consolidation_round_3.md` §5 is about exactly that. The
    spherical tokamaks used to be this test's subject and are now accepted, so the
    subject is the mixed configuration instead: no central solenoid, but a conventional
    machine's coil topology, superconductors and outside-TF placement.

    Reached through `machine_from_indat` rather than through the helper, because the
    point is what a *user* sees: one `NotImplementedError` whose first paragraph is
    byte-for-byte the one `_slot_occupant` used to raise, followed by the others and
    the count. Written over `TOKAMAK_BASELINE_INDAT` for the same reason
    `test_a_refused_value_says_why` is -- several PROCESS defaults are themselves
    refused, and this case is about the PF package, not about whichever of those the
    constructor would reach first.
    """
    indat = tmp_path / "MIXEDPF.DAT"
    indat.write_text(
        "".join(
            f"{f} = {v if isinstance(v, str) else int(v)}\n"
            for f, v in {**TOKAMAK_BASELINE_INDAT, "iohcl": 0}.items()
        )
    )
    with pytest.raises(NotImplementedError) as refusal:
        machine_from_indat(indat)
    message = str(refusal.value)
    assert "pf_coil_system_arm == -2 is a real PROCESS branch" in message
    for arm in (-6, -7):
        assert f"AND pf_coil_system_arm == {arm}:" in message
    assert "3 of the six dimensions" in message


def test_a_spherical_tokamak_pf_system_assembles_without_a_central_solenoid(tmp_path):
    """`iohcl = 0` gives an empty `.tokamak.cs_coil` and the eight-coil PF namespace.

    The measured shape of blocker `-1`: absence, not a variant. `pfcoil()` skips
    `ohcalc` outright when there is no solenoid (`pfcoil.py:1048-1050`), so none of
    `.tokamak.cs_coil`'s seven nodes has a PROCESS counterpart -- the slot is `None`,
    the way `models/tokamak/namespace.py`'s `water_use` is, and the PF occupants on this
    arm declare *fewer reads* rather than reading the fields a solenoid would have
    written.

    Written over `TOKAMAK_BASELINE_INDAT` rather than over the real ST file for the
    reason the refusal test gives: the two tracked spherical tokamaks are still refused
    above this point, on the CroCo TF turn (`i_tf_turn_type = 2`) and the two REBCO tape
    slots behind it, which are a different package.
    """
    indat = tmp_path / "STPF.DAT"
    indat.write_text(
        "".join(
            f"{f} = {v if isinstance(v, str) else int(v)}\n"
            for f, v in {
                **TOKAMAK_BASELINE_INDAT,
                "iohcl": 0,
                "i_pf_location": "2,3,3,4",
                "n_pf_coils_in_group": "2,2,2,2",
                "i_pf_superconductor": 9,
                "i_r_pf_outside_tf_placement": 1,
            }.items()
        )
    )
    machine = machine_from_indat(indat)
    assert machine.tokamak.cs_coil is None
    assert isinstance(machine.tokamak.pf_coil, PFCoilSphericalTokamak)

    # And the fields `ohcalc` would have written are boundary inputs, not zeros some
    # node claims to have computed.
    owners = {v.path_str() for v in graph_for(machine).owners}
    for cs_field in (
        ".pf_coil.a_cs_poloidal",
        ".pf_coil.a_cs_cable_space",
        ".pf_coil.b_cs_peak_flat_top_end",
        ".pf_coil.temp_cs_superconductor_margin",
    ):
        assert cs_field not in owners
    # What `pfcoil()` itself still writes on this arm does have a producer.
    assert ".pf_coil.f_j_cs_start_end_flat_top" in owners
    assert ".pf_coil.m_pf_coil_conductor_total" in owners


NODE_COUNTS = {
    "large_tokamak_nof": 247,
    "large_tokamak_eval": 249,
    "low_aspect_ratio_DEMO": 247,
    "spherical_tokamak_eval": 244,
    "st_regression": 244,
    "stellarator_helias": 154,
    "helias_5b": 154,
}
"""Re-pinned 2026-09-02, and the arithmetic is written down because a bare number is
what let this drift unnoticed for two days.

**+1 on every tokamak, and only the tokamaks**, for `FusionGain` -- the node that gave
`.current_drive.big_q_plasma` a producer and closed the last inert condition on
`st_regression` (`_audit/next_steps.md` §28.3). It sits in `TokamakCurrentDrive`, so
neither stellarator sees it. That is the whole of the difference from the four
movements below.

**+3 on every tokamak**, measured by diffing the node set against the graph at
`60d9ba88` (2026-08-31), the commit that last set these numbers -- three ports landed
since and nothing was removed:

    .tokamak.build.r_cp_top
    .tokamak.physics.psep_over_r_metric
    .tokamak.radiated_wall_load

**-1 more on the two spherical files**, which is today's `burn_time` gate: `Pulse.run`
does not compute the burn time when `i_pulsed_plant = 0`, so the slot is empty on a
steady-state machine and the node is not in the graph. That is why they read `243` where
the other three tokamaks read `246`/`248`.

Both stellarators are unmoved at `154`; none of the four changes is theirs."""
"""`_audit/next_steps.md` §23.6's table, which was recorded there and frozen nowhere.

Two of the seven were pinned in `test_process_free_import.py` and the other five were
"measured the same way and recorded here rather than frozen into the test". This is the
freeze, and it exists because the presence-flag fix immediately below is exactly the
kind of change that moves a count silently.

**Moved once since, by `models/initialisation`** -- the port of `init.py` and `st_init`'s
thirteen `off` writes (`_audit/init_audit.md` §5b). Every configuration gains exactly the
occupants its own switches ask for, and
`test_the_seed_nodes_account_for_every_moved_node_count` asserts the split rather than
leaving it to the totals:

- the three single-null pulsed tokamaks: **+5**, the `init.py` occupants whose fields
  their own nodes read;
- the two spherical tokamaks: **+7** -- those five, plus `esbldgm3` (they are not
  pulsed) and the double-null upper-build identity;
- the two stellarators: **+4** -- `eff_tf_cryo`, `esbldgm3`, and `st_init`'s two.

`TfConductorYoungsModulus` is one node for two fields, which is why the tokamak count
is five and not six.
"""


@pytest.mark.parametrize("stem", sorted(NODE_COUNTS), ids=sorted(NODE_COUNTS))
def test_every_tracked_configuration_assembles_at_its_recorded_node_count(stem):
    path = str(Path(REFERENCE_INPUT_FILE).resolve().parent / f"{stem}.IN.DAT")
    assert len(graph_for(machine_from_indat(path)).definitions) == NODE_COUNTS[stem]


SEED_NODE_COUNTS = {
    "large_tokamak_nof": 5,
    "large_tokamak_eval": 5,
    "low_aspect_ratio_DEMO": 5,
    "spherical_tokamak_eval": 7,
    "st_regression": 7,
    "stellarator_helias": 4,
    "helias_5b": 4,
}
"""How many `.initialisation.*` nodes each configuration's own switches ask for.

Asserted separately from `NODE_COUNTS` for the reason the presence-flag test below
exists: a total is not an account. Two slot changes cancelled exactly once already, and
a count that moves by the right amount for the wrong reason is the failure this file is
supposed to catch.
"""


@pytest.mark.parametrize("stem", sorted(SEED_NODE_COUNTS), ids=sorted(SEED_NODE_COUNTS))
def test_the_seed_nodes_account_for_every_moved_node_count(stem):
    """`models/initialisation` is the whole of `NODE_COUNTS`' one movement.

    The split is not uniform and that is the content: `init.py` writes different fields
    on different machines, and `st_init` writes on one device only. A tokamak gets the
    five `init.py` occupants whose fields its own nodes read; a double-null tokamak gets
    two more (`esbldgm3`, because it is not pulsed, and the upper-build identity); a
    stellarator gets two of the five plus `st_init`'s pair, because its graph has no TF
    stress chain and no beam.
    """
    path = str(Path(REFERENCE_INPUT_FILE).resolve().parent / f"{stem}.IN.DAT")
    graph = graph_for(machine_from_indat(path))
    seed_nodes = [
        node
        for node in graph.definitions
        if node.path_str().startswith(".initialisation")
    ]
    assert len(seed_nodes) == SEED_NODE_COUNTS[stem], sorted(
        n.path_str() for n in seed_nodes
    )


def test_the_presence_flags_swap_two_nodes_and_add_none(tmp_path):
    """Why the two spherical tokamaks still count 234 after gaining a producer.

    `init.py:925-930`'s two presence flags were stuck at `False` on every file
    (`init_audit.md` §3) and are now read from the text, which flips both on four of
    the seven. That changes two slots at once and the changes cancel:

    - `DX_TF_SIDE_CASE_MIN[True]` is a node where `[False]` is `None` -- **+1**, and it
      is the missing producer `.tfcoil.dx_tf_side_case_min` (`next_steps.md` §22.6).
    - `DR_TF_PLASMA_CASE[True]` is an `ExplicitFunction` where `[False]` is a
      `FixedPointFunction` -- a node that reads what it owns, so it mints a second
      `^problem` node for its own cut. **-1**.

    Net zero on the two spherical tokamaks, and *neither* half of that is a coincidence
    worth trusting silently, which is why it is asserted rather than left to the total.
    """
    directory = Path(REFERENCE_INPUT_FILE).resolve().parent
    fraction = graph_for(machine_from_indat(str(directory / "st_regression.IN.DAT")))
    from_input = graph_for(
        machine_from_indat(str(directory / "large_tokamak_eval.IN.DAT"))
    )

    def nodes_named(graph, fragment):
        return {str(n) for n in graph.definitions if fragment in str(n)}

    assert len(nodes_named(fraction, "dx_tf_side_case_min")) == 1
    assert not nodes_named(from_input, "dx_tf_side_case_min")
    assert len(nodes_named(fraction, "dr_tf_plasma_case")) == 1
    assert len(nodes_named(from_input, "dr_tf_plasma_case")) == 2
    assert any("^problem" in n for n in nodes_named(from_input, "dr_tf_plasma_case"))
