"""Each model variant the converter chose for a switch is the branch PROCESS's own
dispatch takes at the configuration's switch values.

A switch picks a model; no node reads one. For every slot whose occupant a switch
chooses, the chosen declaration, fed random values at the places it reads, agrees with
the original dispatching function called with the same values and the configuration's
switch values -- so the converter picked the right variant, and the variant is the
branch.

An index that only selects a table entry is not a switch either: the declaration reads
the entry, derived from the values by `native`'s derivations. Every declaration that
reads one is fed the entries derived from the same values the dispatch is called with,
and agrees with the dispatch called with the configuration's index.
"""

import inspect

import numpy as np
import pytest

from functional_process import configurations
from functional_process.configurations import NAMES
from functional_process.cottax.input.indat import switches_from_indat
from functional_process.cottax.input.native import (
    _level_of_safety_assurance,
    _table_entries,
    layered_values,
)
from functional_process.models.costs import costs as C
from functional_process.models.pfcoil import masses as pf_masses
from functional_process.models.power import thermal_cryo as T
from functional_process.models.stellarator.coils.mass import calculate_coils_mass
from functional_process.models.stellarator.stellarator_fwbs_s1_s5 import (
    calculate_fw_blanket_shield_geometry,
)
from functional_process.models.tfcoil import superconducting as tf_superconducting
from functional_process.models.vacuum.vacuum import calculate_vacuum_pumping_old

SWITCHES = (
    "ireactor",
    "itart",
    "i_tf_sup",
    "ifueltyp",
    "ife",
    "i_hcd_primary",
    "i_shld_primary_heat",
    "ipowerflow",
    "i_pf_conductor",
    "i_vac_pump_dwell",
    "i_vacuum_pump_type",
    # Not switched by this work, but the thermal arms' dispatch reads them too.
    "i_p_coolant_pumping",
    "i_blkt_dual_coolant",
    "i_thermal_electric_conversion",
    # The PF and TF magnet costs' dispatch.
    "supercond_cost_model",
)
"""Every switch a dispatch below is called with, valued from the configuration."""


def _indexed(fn, **entries):
    """`fn`, taking each of `entries`' parameters `name=(table, index)` as the table and
    the index instead and passing `table[index - 1]`: the lookup PROCESS makes inside
    the model, for a model the port always handed the entry.
    """
    looked_up = {n for pair in entries.values() for n in pair}
    names = [n for n in inspect.signature(fn).parameters if n not in entries]
    names += dict.fromkeys(n for pair in entries.values() for n in pair)

    def dispatch(**kwargs):
        entry = {
            name: np.asarray(kwargs[table])[int(kwargs[index]) - 1]
            for name, (table, index) in entries.items()
        }
        rest = {k: v for k, v in kwargs.items() if k not in looked_up}
        return fn(**rest, **entry)

    dispatch.__signature__ = inspect.Signature([
        inspect.Parameter(n, inspect.Parameter.KEYWORD_ONLY) for n in names
    ])
    dispatch.__name__ = fn.__name__
    return dispatch


TF_DENSITY = {"den_tf_sc_material": ("dcond", "i_tf_sc_mat")}
PF_DENSITY = {"den_pf_conductor": ("dcond", "i_pf_superconductor")}
CS_DENSITY = {"den_cs_conductor": ("dcond", "i_cs_superconductor")}

SLOTS = {
    ("costs", "convert_fpy_to_calendar"): C.convert_fpy_to_calendar,
    ("costs", "structures_cost"): C.calculate_structures_cost,
    ("costs", "turbine_plant_equipment_cost"): C.calculate_turbine_plant_equipment_cost,
    ("costs", "heat_rejection_cost"): C.calculate_heat_rejection_cost,
    ("costs", "tf_coil_power_conditioning_cost"): (
        C.calculate_tf_coil_power_conditioning_cost
    ),
    ("costs", "first_wall_cost"): C.calculate_first_wall_cost,
    ("costs", "blanket_cost"): C.calculate_blanket_cost,
    ("costs", "divertor_cost"): C.calculate_divertor_cost,
    ("costs", "power_injection_cost"): C.calculate_power_injection_cost,
    ("costs", "magnets_cost"): C.calculate_magnets_cost,
    ("costs", "power_conditioning_cost"): C.calculate_power_conditioning_cost,
    ("costs", "vacuum_system_cost"): C.calculate_vacuum_system_cost,
    ("costs", "cost_of_electricity"): C.calculate_cost_of_electricity,
    ("costs", "pf_magnet_cost"): C.calculate_pf_magnet_cost,
    ("power", "component_thermal_powers"): T.calculate_component_thermal_powers_owned,
    ("power", "delta_eta_step"): T.calculate_delta_eta_next,
    ("vacuum", "vacuum_old"): calculate_vacuum_pumping_old,
    ("stellarator", "fwbs", "fw_blanket_shield_geometry"): (
        calculate_fw_blanket_shield_geometry
    ),
}
"""Each switched slot -> PROCESS's dispatch over its variants."""

INDEXED = {
    ("costs", "indirect_costs"): C.calculate_indirect_costs,
    ("costs", "reactor_structure_cost"): C.calculate_reactor_structure_cost,
    ("costs", "vacuum_vessel_assembly_cost"): C.calculate_vacuum_vessel_assembly_cost,
    ("costs", "reactor_cooling_system_cost"): C.calculate_reactor_cooling_system_cost,
    ("costs", "switchyard_cost"): C.calculate_switchyard_cost,
    ("costs", "transformers_cost"): C.calculate_transformers_cost,
    ("costs", "low_voltage_cost"): C.calculate_low_voltage_cost,
    ("costs", "diesel_generators_cost"): C.calculate_diesel_generators_cost,
    (
        "costs",
        "auxiliary_facility_power_cost",
    ): C.calculate_auxiliary_facility_power_cost,
    ("costs", "misc_plant_equipment_cost"): C.calculate_misc_plant_equipment_cost,
    ("costs", "shield_cost"): C.calculate_shield_cost,
    ("costs", "tf_magnet_cost_superconducting"): (
        C.calculate_tf_magnet_cost_superconducting
    ),
    ("costs", "auxiliary_component_cooling_cost"): (
        C.calculate_auxiliary_component_cooling_cost
    ),
    ("costs", "cryogenic_system_cost"): C.calculate_cryogenic_system_cost,
    ("tokamak", "pf_coil", "masses"): {
        "calculate_pf_coil_masses_from_elements": _indexed(
            pf_masses.calculate_pf_coil_masses_from_elements, **PF_DENSITY, **CS_DENSITY
        ),
        "calculate_pf_coil_masses_no_central_solenoid_bound": _indexed(
            pf_masses.calculate_pf_coil_masses_no_central_solenoid_bound, **PF_DENSITY
        ),
    },
    (
        "tokamak",
        "cicc_superconducting_tf_coil",
        "superconducting_tf_coil_areas_and_masses",
    ): {
        "_superconducting_tf_coil_areas_and_masses_conventional": _indexed(
            tf_superconducting.superconducting_tf_coil_areas_and_masses_conventional,
            **TF_DENSITY,
        ),
        "_superconducting_tf_coil_areas_and_masses_spherical_tokamak": _indexed(
            tf_superconducting.superconducting_tf_coil_areas_and_masses_spherical_tokamak,
            **TF_DENSITY,
        ),
    },
    ("stellarator", "coils", "coils_mass"): _indexed(calculate_coils_mass, **TF_DENSITY),
}
"""Each slot whose declaration reads a table entry an index selects -> PROCESS's
signature, which takes the index (the switched slots above read entries too). The
masses have no PROCESS signature of their own: `_indexed` makes the lookup PROCESS makes
inside `pfcoil` / `sctfcoil` / `stellarator`."""

NOT_PLACES = {
    "i_p_coolant_pumping": 2,
    "i_blkt_dual_coolant": 0,
    "supercond_cost_model": 0,
}
"""Switches that are not a place a configuration values, and PROCESS's defaults
(`fwbs_variables.py:249`, `:526`, `cost_variables.py:552`); the file's own value
wins."""

FIXED = {"qtorus": 0.0}
"""Values the declarations fix at `Vacuum.run`'s one call site."""

PF_COIL_ARRAYS = (
    "r_pf_coil_middle",
    "c_pf_cs_coils_peak_ma",
    "r_pf_coil_inner",
    "r_pf_coil_outer",
    "z_pf_coil_upper",
    "z_pf_coil_lower",
    "b_pf_coil_peak",
    "bpf2",
)
"""Computed per-coil arrays no configuration values, shaped as `n_pf_coil_turns`."""


def _chosen(name, slots):
    """Every slot of `slots` `name`'s machine fills.

    Yields
    ------
    tuple
        `(slot, declaration, dispatch)`.
    """
    machine = configurations.load(name).machine
    for slot, dispatches in slots.items():
        occupant = machine
        for field in slot:
            occupant = getattr(occupant, field, None)
        if occupant is None:
            continue
        dispatch = (
            dispatches[occupant.fn.__name__]
            if isinstance(dispatches, dict)
            else dispatches
        )
        yield slot, occupant, dispatch


def _switch_values(name):
    """Every switch in `SWITCHES`, at the value the configuration's graph read before
    the switches became variants: the file's own, else PROCESS's default.
    """
    values, _sources = layered_values(configurations.load(name).values)
    by_field = dict(NOT_PLACES)
    for (_area, field), value in values.items():
        if field in SWITCHES:
            by_field[field] = int(value)
    stated = switches_from_indat(f"tests/regression/input_files/{name}.IN.DAT")
    by_field |= {k: int(v) for k, v in stated.items() if k in NOT_PLACES}
    return by_field, values


def _random_like(value, rng):
    """`value` perturbed by up to 10 % per entry, so every model is evaluated near a
    point it is valid at; an integer (an index, a count) kept, and a zero moved off
    zero."""
    if isinstance(value, bool | int | np.integer):
        return value
    array = np.asarray(value)
    if array.dtype.kind in "iub":
        return value
    base = np.where(array == 0.0, 1.0, array)
    perturbed = base * rng.uniform(0.9, 1.1, size=array.shape)
    return float(perturbed) if array.ndim == 0 else perturbed


def _derived(arguments, state):
    """The table entries `native` derives, from the values the dispatch is called with:
    each place at its argument where the dispatch takes one, else at the state's value.
    """
    values = {place: arguments.get(place[1], value) for place, value in state.items()}
    out = {}
    _level_of_safety_assurance(values, out)
    _table_entries(values, out)
    return {field: value for (_area, field), value in out.items()}


def _agree(name, slots):
    """Each declaration of `slots` `name`'s machine fills, fed at the places it reads,
    agrees with its dispatch called with the configuration's switches and indices, to
    rtol 1e-12.

    Returns
    -------
    int
        How many declarations were checked.
    """
    switches, state = _switch_values(name)
    by_leaf = {field: value for (_area, field), value in state.items()}
    rng = np.random.default_rng(0)
    checked = 0
    for slot, declaration, dispatch in _chosen(name, slots):
        places = dict(declaration.reads.items())
        leaf_of = {}
        for parameter, place in places.items():
            leaf = getattr(place.segments[-1], "name", None)
            leaf_of[parameter] = leaf if leaf is not None else parameter
        arguments = {}
        for parameter in inspect.signature(dispatch).parameters:
            if parameter in switches:
                arguments[parameter] = switches[parameter]
            elif parameter in FIXED:
                arguments[parameter] = FIXED[parameter]
            elif parameter in PF_COIL_ARRAYS:
                shape = np.shape(by_leaf["n_pf_coil_turns"])
                arguments[parameter] = rng.uniform(0.5, 2.0, size=shape)
            else:
                arguments[parameter] = _random_like(by_leaf.get(parameter, 1.0), rng)
        derived = _derived(arguments, state)
        values = []
        for parameter in places:
            leaf = leaf_of[parameter]
            if leaf in arguments:
                values.append(arguments[leaf])
            elif leaf in derived:
                values.append(derived[leaf])
            else:
                values.append(arguments[parameter])
        got = declaration.fn(*values)
        want = dispatch(**arguments)
        got = got if isinstance(got, tuple) else (got,)
        want = want if isinstance(want, tuple) else (want,)
        assert len(got) == len(want), slot
        for g, w in zip(got, want, strict=True):
            assert np.all(np.isfinite(np.asarray(w, dtype=float))), slot
            np.testing.assert_allclose(
                np.asarray(g, dtype=float),
                np.asarray(w, dtype=float),
                rtol=1e-12,
                err_msg=f"{'.'.join(slot)}: {declaration.fn.__name__} is not "
                f"{dispatch.__name__}'s branch at {switches}",
            )
        checked += 1
    return checked


@pytest.mark.parametrize("name", NAMES)
def test_each_variant_is_the_branch_processes_switches_take(name):
    """The chosen declaration, fed at the places it reads, agrees with PROCESS's
    dispatch called with the configuration's switches, to rtol 1e-12.
    """
    assert _agree(name, SLOTS)


@pytest.mark.parametrize("name", NAMES)
def test_each_table_entry_is_the_one_processes_index_selects(name):
    """Each declaration reading a derived table entry, fed the entries derived from the
    same values, agrees with PROCESS's signature called with the configuration's index,
    to rtol 1e-12.
    """
    assert _agree(name, INDEXED)
