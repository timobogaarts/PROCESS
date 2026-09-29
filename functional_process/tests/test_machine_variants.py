"""Each model variant the converter chose for a switch is the branch PROCESS's own
dispatch takes at the configuration's switch values.

A switch picks a model; no node reads one. For every slot whose occupant a switch
chooses, the chosen declaration, fed random values at the places it reads, agrees with
the original dispatching function called with the same values and the configuration's
switch values -- so the converter picked the right variant, and the variant is the
branch.
"""

import inspect

import numpy as np
import pytest

from functional_process import configurations
from functional_process.configurations import NAMES
from functional_process.cottax.input.indat import switches_from_indat
from functional_process.cottax.input.native import layered_values
from functional_process.cottax.models.costs import costs as cost_declarations
from functional_process.models.costs import costs as C
from functional_process.models.power import thermal_cryo as T
from functional_process.models.stellarator.stellarator_fwbs_s1_s5 import (
    calculate_fw_blanket_shield_geometry,
)
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
)
"""Every switch a dispatch below is called with, valued from the configuration."""

PF_MAGNET_COST = {
    "PfMagnetCostPerKg": C.calculate_pf_magnet_cost_per_kg,
    "PfMagnetCostPerKgCsWstNb3Sn": C.calculate_pf_magnet_cost_per_kg,
    "PfMagnetCostPerKgNoCentralSolenoid": (
        C.calculate_pf_magnet_cost_per_kg_no_central_solenoid
    ),
    "PfMagnetCostPerKam": C.calculate_pf_magnet_cost_per_kam,
    "PfMagnetCostPerKamNoCentralSolenoid": (
        C.calculate_pf_magnet_cost_per_kam_no_central_solenoid
    ),
}

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
    ("costs", "pf_magnet_cost"): PF_MAGNET_COST,
    ("power", "component_thermal_powers"): T.calculate_component_thermal_powers_owned,
    ("power", "delta_eta_step"): T.calculate_delta_eta_next,
    ("vacuum", "vacuum_old"): calculate_vacuum_pumping_old,
    ("stellarator", "fwbs", "fw_blanket_shield_geometry"): (
        calculate_fw_blanket_shield_geometry
    ),
}
"""Each switched slot -> PROCESS's dispatch over its variants."""

NOT_PLACES = {"i_p_coolant_pumping": 2, "i_blkt_dual_coolant": 0}
"""Switches that are not a place a configuration values, and PROCESS's defaults
(`fwbs_variables.py:249`, `:526`); the file's own value wins."""

FIXED = {"qtorus": 0.0}
"""Values the declarations fix at `Vacuum.run`'s one call site."""

BROKEN_BEFORE = {
    "PfMagnetCostPerKgNoCentralSolenoid": (
        "its body passes `calculate_pf_magnet_cost_per_kg_no_central_solenoid` no "
        "`den_pf_conductor` (the 2026-08-31 split never reached this occupant), so a "
        "call is one argument short; unchanged by the switch work"
    ),
}
"""Declarations that cannot be called, for a reason that predates the variants. Pinned
as failing, so that a fix shows up here."""

PF_COIL_ARRAYS = ("r_pf_coil_middle", "c_pf_cs_coils_peak_ma")
"""Computed per-coil arrays no configuration values, shaped as `n_pf_coil_turns`."""


def _chosen(name):
    """Every switched slot `name`'s machine fills.

    Yields
    ------
    tuple
        `(slot, declaration, dispatch)`.
    """
    machine = configurations.load(name).machine
    for slot, dispatches in SLOTS.items():
        occupant = machine
        for field in slot:
            occupant = getattr(occupant, field, None)
        if occupant is None:
            continue
        dispatch = (
            dispatches[_declared_name(occupant)]
            if isinstance(dispatches, dict)
            else dispatches
        )
        yield slot, occupant, dispatch


def _declared_name(declaration):
    """The name `declaration` is bound to among the cost declarations, if any."""
    return next(
        (k for k, v in vars(cost_declarations).items() if v is declaration), None
    )


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
    """A random value shaped like `value`; an integer (an index, a count) kept."""
    if isinstance(value, bool | int | np.integer):
        return value
    array = np.asarray(value)
    if array.dtype.kind in "iub":
        return value
    if array.ndim == 0:
        return float(rng.uniform(0.5, 2.0))
    return rng.uniform(0.5, 2.0, size=array.shape)


@pytest.mark.parametrize("name", NAMES)
def test_each_variant_is_the_branch_processes_switches_take(name):
    """The chosen declaration, fed at the places it reads, agrees with PROCESS's
    dispatch called with the configuration's switches, to rtol 1e-12.
    """
    switches, state = _switch_values(name)
    by_leaf = {field: value for (_area, field), value in state.items()}
    rng = np.random.default_rng(0)
    checked = 0
    for slot, declaration, dispatch in _chosen(name):
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
        values = []
        for parameter in places:
            leaf = leaf_of[parameter]
            values.append(arguments[leaf] if leaf in arguments else arguments[parameter])
        if _declared_name(declaration) in BROKEN_BEFORE:
            with pytest.raises(TypeError):
                declaration.fn(*values)
            continue
        got = declaration.fn(*values)
        want = dispatch(**arguments)
        got = got if isinstance(got, tuple) else (got,)
        want = want if isinstance(want, tuple) else (want,)
        assert len(got) == len(want), slot
        for g, w in zip(got, want, strict=True):
            np.testing.assert_allclose(
                np.asarray(g, dtype=float),
                np.asarray(w, dtype=float),
                rtol=1e-12,
                err_msg=f"{'.'.join(slot)}: {declaration.fn.__name__} is not "
                f"{dispatch.__name__}'s branch at {switches}",
            )
        checked += 1
    assert checked
