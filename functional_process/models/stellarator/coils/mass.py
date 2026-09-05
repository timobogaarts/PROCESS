"""Pure physics functions extracted from
`functional_process.cottax.stellarator.coils.mass`, per `_audit/formulas_split.md`
step 2 phase A. The graph declarations that wrap these stay in that sibling
module.
"""

from functional_process.vocabulary import constants


def calculate_coils_mass(
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
    """Total coil mass, and the intermediate component masses that feed it.

    Ports `calculate_coils_mass`'s 8-step chain (see module docstring).

    Parameters
    ----------
    a_tf_wp_with_insulation, a_tf_wp_no_insulation :
        Winding pack area, with/without insulation (m2).
    len_tf_coil :
        TF coil length (m). `.tfcoil.len_tf_coil`.
    a_tf_coil_inboard_case :
        TF coil case area (m2). `.tfcoil.a_tf_coil_inboard_case`.
    den_tf_coil_case :
        Case material density (kg/m3). `.tfcoil.den_tf_coil_case`.
    den_tf_wp_turn_insulation :
        Turn/ground insulation density (kg/m3). `.tfcoil.den_tf_wp_turn_insulation`.
    n_tf_coil_turns :
        Turns per coil. `.tfcoil.n_tf_coil_turns`.
    a_tf_turn_cable_space_no_void :
        Cable space area per turn, no void (m2). `.tfcoil.a_tf_turn_cable_space_no_void`.
    f_a_tf_turn_cable_space_extra_void :
        Extra-void fraction of cable space. `.tfcoil.f_a_tf_turn_cable_space_extra_void`.
    f_a_tf_turn_cable_copper :
        Copper fraction of cable conductor. `.tfcoil.f_a_tf_turn_cable_copper`.
    a_tf_wp_coolant_channels :
        Coolant channel area (m2, 0 for a stellarator). `.tfcoil.a_tf_wp_coolant_channels`.
    den_tf_sc_material :
        Superconductor density (kg/m3) -- `.tfcoil.dcond[i_tf_sc_mat - 1]`, already
        indexed by the material switch (see module docstring).
    a_tf_turn_steel :
        Steel conduit area per turn (m2). `.tfcoil.a_tf_turn_steel`.
    den_steel :
        Steel density (kg/m3). `.fwbs.den_steel`.
    a_tf_coil_wp_turn_insulation :
        Turn insulation area, already `n_tf_coil_turns`-scaled (m2).
        `.tfcoil.a_tf_coil_wp_turn_insulation`.
    n_tf_coils :
        Number of TF coils. `.tfcoil.n_tf_coils`.

    Returns
    -------
    :
        `(m_tf_coil_case, m_tf_coil_wp_insulation, m_tf_coil_superconductor,
        m_tf_coil_copper, m_tf_wp_steel_conduit, m_tf_coil_wp_turn_insulation,
        m_tf_coil_conductor, m_tf_coils_total)` -- all masses (kg).
    """
    m_tf_coil_case = len_tf_coil * a_tf_coil_inboard_case * den_tf_coil_case

    m_tf_coil_wp_insulation = (
        len_tf_coil
        * (a_tf_wp_with_insulation - a_tf_wp_no_insulation)
        * den_tf_wp_turn_insulation
    )

    m_tf_coil_superconductor = (
        len_tf_coil
        * n_tf_coil_turns
        * a_tf_turn_cable_space_no_void
        * (1.0 - f_a_tf_turn_cable_space_extra_void)
        * (1.0 - f_a_tf_turn_cable_copper)
        - len_tf_coil * a_tf_wp_coolant_channels
    ) * den_tf_sc_material

    m_tf_coil_copper = (
        len_tf_coil
        * n_tf_coil_turns
        * a_tf_turn_cable_space_no_void
        * (1.0 - f_a_tf_turn_cable_space_extra_void)
        * f_a_tf_turn_cable_copper
        - len_tf_coil * a_tf_wp_coolant_channels
    ) * constants.DEN_COPPER

    m_tf_wp_steel_conduit = len_tf_coil * n_tf_coil_turns * a_tf_turn_steel * den_steel

    m_tf_coil_wp_turn_insulation = (
        len_tf_coil * a_tf_coil_wp_turn_insulation * den_tf_wp_turn_insulation
    )

    m_tf_coil_conductor = (
        m_tf_coil_superconductor
        + m_tf_coil_copper
        + m_tf_wp_steel_conduit
        + m_tf_coil_wp_turn_insulation
    )

    m_tf_coils_total = (
        m_tf_coil_case + m_tf_coil_conductor + m_tf_coil_wp_insulation
    ) * n_tf_coils

    return (
        m_tf_coil_case,
        m_tf_coil_wp_insulation,
        m_tf_coil_superconductor,
        m_tf_coil_copper,
        m_tf_wp_steel_conduit,
        m_tf_coil_wp_turn_insulation,
        m_tf_coil_conductor,
        m_tf_coils_total,
    )
