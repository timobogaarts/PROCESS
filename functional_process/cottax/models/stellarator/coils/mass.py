"""Pure-functional port of `process/models/stellarator/coils/mass.py` (registry #12)."""

from cottax.interfaces import V

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

# The CoilsMass family:
# The `calculate_coils_mass` family -- one occupant per `.tfcoil.i_tf_sc_mat`
# value, differing only in which element of `.tfcoil.dcond` is the superconductor
# density.


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


# `i_tf_sc_mat == ITER_NB3SN` (1) -- ITER Nb3Sn -- PROCESS's own default and the
# reference run's.
IterNb3snCoilsMass = tfcoil.function(
    _coils_mass,
    *_COILS_MASS_OUTPUTS,
    reads={"den_tf_sc_material": V.tfcoil.dcond[0], "den_steel": fwbs},
)


# `i_tf_sc_mat == BI2212` (2) -- Bi-2212.
Bi2212CoilsMass = tfcoil.function(
    _coils_mass,
    *_COILS_MASS_OUTPUTS,
    reads={"den_tf_sc_material": V.tfcoil.dcond[1], "den_steel": fwbs},
)


# `i_tf_sc_mat == OLD_LUBELL_NBTI` (3) -- old Lubell NbTi.
OldLubellNbtiCoilsMass = tfcoil.function(
    _coils_mass,
    *_COILS_MASS_OUTPUTS,
    reads={"den_tf_sc_material": V.tfcoil.dcond[2], "den_steel": fwbs},
)


# `i_tf_sc_mat == USER_DEFINED_NB3SN` (4) -- user-defined ITER Nb3Sn.
UserDefinedNb3snCoilsMass = tfcoil.function(
    _coils_mass,
    *_COILS_MASS_OUTPUTS,
    reads={"den_tf_sc_material": V.tfcoil.dcond[3], "den_steel": fwbs},
)


# `i_tf_sc_mat == WST_NB3SN` (5) -- WST Nb3Sn.
WstNb3snCoilsMass = tfcoil.function(
    _coils_mass,
    *_COILS_MASS_OUTPUTS,
    reads={"den_tf_sc_material": V.tfcoil.dcond[4], "den_steel": fwbs},
)


# `i_tf_sc_mat == CROCO_REBCO` (6) -- CroCo REBCO.
CrocoRebcoCoilsMass = tfcoil.function(
    _coils_mass,
    *_COILS_MASS_OUTPUTS,
    reads={"den_tf_sc_material": V.tfcoil.dcond[5], "den_steel": fwbs},
)


# `i_tf_sc_mat == DURHAM_NBTI` (7) -- Durham Ginzburg-Landau NbTi.
DurhamNbtiCoilsMass = tfcoil.function(
    _coils_mass,
    *_COILS_MASS_OUTPUTS,
    reads={"den_tf_sc_material": V.tfcoil.dcond[6], "den_steel": fwbs},
)


# `i_tf_sc_mat == DURHAM_REBCO` (8) -- Durham REBCO.
DurhamRebcoCoilsMass = tfcoil.function(
    _coils_mass,
    *_COILS_MASS_OUTPUTS,
    reads={"den_tf_sc_material": V.tfcoil.dcond[7], "den_steel": fwbs},
)
