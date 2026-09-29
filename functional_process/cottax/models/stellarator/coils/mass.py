"""Pure-functional port of `process/models/stellarator/coils/mass.py` (registry #12)."""

from functional_process.cottax.paths import (
    fwbs,
    tfcoil,
)
from functional_process.models.stellarator.coils.mass import (
    calculate_coils_mass,
)
from functional_process.vocabulary import (
    constants,  # noqa: F401
)

# `calculate_coils_mass`, given the superconductor's density
# `.tfcoil.den_tf_sc_material` -- the entry of `.tfcoil.dcond` at `.tfcoil.i_tf_sc_mat`,
# derived from a configuration's values (`cottax.input.native`), so one occupant serves
# every material.


def _coils_mass(
    den_tf_sc_material,
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
    a_tf_turn_steel,
    a_tf_coil_wp_turn_insulation,
    n_tf_coils,
    den_steel,
):
    """The whole calculation, given this material's density."""
    return calculate_coils_mass(
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
    )


_COILS_MASS_OUTPUTS = (
    "m_tf_coil_case",
    "m_tf_coil_wp_insulation",
    "m_tf_coil_superconductor",
    "m_tf_coil_copper",
    "m_tf_wp_steel_conduit",
    "m_tf_coil_wp_turn_insulation",
    "m_tf_coil_conductor",
    "m_tf_coils_total",
)


CoilsMass = tfcoil.function(
    _coils_mass,
    *_COILS_MASS_OUTPUTS,
    reads={"den_steel": fwbs},
)
