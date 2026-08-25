"""Harness cases for the ported `coils/mass.py` (registry #12).

No matching PROCESS unit test exists for `calculate_coils_mass` or its 8 sub-functions
(grepped `tests/unit/models/stellarator/test_stellarator.py`) -- fuzz-only, same as
`forces.py`.
"""

from functional_process._harness import Tier1Contract
from functional_process.models.stellarator.coils.mass import calculate_coils_mass
from process.core.model import DataStructure
from process.models.stellarator.coils.mass import (
    calculate_coils_mass as _process_calculate_coils_mass,
)


def _reference_coils_mass(
    a_tf_wp_with_insulation,
    a_tf_wp_no_insulation,
    len_tf_coil,
    a_tf_coil_inboard_case,
    den_tf_coil_case,
    den_tf_wp_turn_insulation,
    n_tf_coil_turns,
    a_tf_turn_cable_space_no_void,
    f_a_tf_turn_cable_space_extra_void,
    f_a_tf_turn_cable_copper,
    a_tf_wp_coolant_channels,
    den_tf_sc_material,
    a_tf_turn_steel,
    den_steel,
    a_tf_coil_wp_turn_insulation,
    n_tf_coils,
):
    """Call PROCESS's `calculate_coils_mass` through the port's signature.

    `i_tf_sc_mat` is fixed at its default (1); `dcond` is set uniformly to
    `den_tf_sc_material` so `dcond[i_tf_sc_mat - 1]` equals the sample value
    regardless of which material index is selected -- isolates the lookup from the
    switch, consistent with the port treating the indexed scalar as an explicit arg.
    """
    data = DataStructure()
    data.tfcoil.len_tf_coil = len_tf_coil
    data.tfcoil.a_tf_coil_inboard_case = a_tf_coil_inboard_case
    data.tfcoil.den_tf_coil_case = den_tf_coil_case
    data.tfcoil.den_tf_wp_turn_insulation = den_tf_wp_turn_insulation
    data.tfcoil.n_tf_coil_turns = n_tf_coil_turns
    data.tfcoil.a_tf_turn_cable_space_no_void = a_tf_turn_cable_space_no_void
    data.tfcoil.f_a_tf_turn_cable_space_extra_void = f_a_tf_turn_cable_space_extra_void
    data.tfcoil.f_a_tf_turn_cable_copper = f_a_tf_turn_cable_copper
    data.tfcoil.a_tf_wp_coolant_channels = a_tf_wp_coolant_channels
    data.tfcoil.dcond = [den_tf_sc_material] * len(data.tfcoil.dcond)
    data.tfcoil.a_tf_turn_steel = a_tf_turn_steel
    data.fwbs.den_steel = den_steel
    data.tfcoil.a_tf_coil_wp_turn_insulation = a_tf_coil_wp_turn_insulation
    data.tfcoil.n_tf_coils = n_tf_coils

    _process_calculate_coils_mass(a_tf_wp_with_insulation, a_tf_wp_no_insulation, data)

    return (
        data.tfcoil.m_tf_coil_case,
        data.tfcoil.m_tf_coil_wp_insulation,
        data.tfcoil.m_tf_coil_superconductor,
        data.tfcoil.m_tf_coil_copper,
        data.tfcoil.m_tf_wp_steel_conduit,
        data.tfcoil.m_tf_coil_wp_turn_insulation,
        data.tfcoil.m_tf_coil_conductor,
        data.tfcoil.m_tf_coils_total,
    )


class TestCoilsMass(Tier1Contract):
    audit_record = "models/stellarator/coils/mass.md"
    reference = _reference_coils_mass
    ported = calculate_coils_mass

    fuzz_bounds = {
        "a_tf_wp_with_insulation": (0.05, 5.0),
        "a_tf_wp_no_insulation": (0.01, 4.0),
        "len_tf_coil": (1.0, 5000.0),
        "a_tf_coil_inboard_case": (0.01, 3.0),
        "den_tf_coil_case": (1000.0, 10000.0),
        "den_tf_wp_turn_insulation": (500.0, 3000.0),
        "n_tf_coil_turns": (1.0, 2000.0),
        "a_tf_turn_cable_space_no_void": (1.0e-6, 0.01),
        "f_a_tf_turn_cable_space_extra_void": (0.0, 0.6),
        "f_a_tf_turn_cable_copper": (0.0, 0.95),
        "a_tf_wp_coolant_channels": (0.0, 0.001),
        "den_tf_sc_material": (5000.0, 9000.0),
        "a_tf_turn_steel": (1.0e-6, 0.01),
        "den_steel": (1000.0, 10000.0),
        "a_tf_coil_wp_turn_insulation": (1.0e-6, 0.5),
        "n_tf_coils": (1.0, 100.0),
    }
