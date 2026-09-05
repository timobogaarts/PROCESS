"""Pure physics functions extracted from
`functional_process.cottax.stellarator.coils.forces`, per `_audit/formulas_split.md`
step 2 phase A. The graph declarations that wrap these stay in that sibling
module.
"""


def calculate_max_force_density(
    a_tf_wp_no_insulation,
    stella_config_max_force_density,
    f_st_i_total,
    f_st_n_coils,
    b_tf_inboard_peak_symmetric,
    stella_config_wp_bmax,
    stella_config_wp_area,
):
    """Maximum force density in the TF coil winding pack, from scaling (MN/m3)."""
    return (
        stella_config_max_force_density
        * f_st_i_total
        / f_st_n_coils
        * b_tf_inboard_peak_symmetric
        / stella_config_wp_bmax
        * stella_config_wp_area
        / a_tf_wp_no_insulation
    )


def calculate_max_force_density_mnm(
    stella_config_max_force_density_mnm,
    f_st_i_total,
    f_st_n_coils,
    b_tf_inboard_peak_symmetric,
    stella_config_wp_bmax,
):
    """Maximum force per unit length in the TF coil winding pack, from scaling (MN/m)."""
    return (
        stella_config_max_force_density_mnm
        * f_st_i_total
        / f_st_n_coils
        * b_tf_inboard_peak_symmetric
        / stella_config_wp_bmax
    )


def calculate_maximum_stress(max_force_density, dr_tf_wp_with_insulation):
    """Approximate maximum stress (needed for constraint 32), in Pa.

    `max_force_density` is `calculate_max_force_density`'s return value -- see module
    docstring for why this is an explicit argument rather than a re-read off `data`.
    """
    return max_force_density * dr_tf_wp_with_insulation * 1.0e6


def calculate_max_lateral_force_density(
    a_tf_wp_no_insulation,
    stella_config_max_lateral_force_density,
    f_st_i_total,
    f_st_n_coils,
    b_tf_inboard_peak_symmetric,
    stella_config_wp_bmax,
    stella_config_wp_area,
):
    """Maximum lateral force density in the TF coil winding pack, from scaling (MN/m3)."""
    return (
        stella_config_max_lateral_force_density
        * f_st_i_total
        / f_st_n_coils
        * b_tf_inboard_peak_symmetric
        / stella_config_wp_bmax
        * stella_config_wp_area
        / a_tf_wp_no_insulation
    )


def calculate_max_radial_force_density(
    a_tf_wp_no_insulation,
    stella_config_max_radial_force_density,
    f_st_i_total,
    f_st_n_coils,
    b_tf_inboard_peak_symmetric,
    stella_config_wp_bmax,
    stella_config_wp_area,
):
    """Maximum radial force density in the TF coil winding pack, from scaling (MN/m3)."""
    return (
        stella_config_max_radial_force_density
        * f_st_i_total
        / f_st_n_coils
        * b_tf_inboard_peak_symmetric
        / stella_config_wp_bmax
        * stella_config_wp_area
        / a_tf_wp_no_insulation
    )


def calculate_centering_force_max_mn(
    stella_config_centering_force_max_mn,
    f_st_i_total,
    f_st_n_coils,
    b_tf_inboard_peak_symmetric,
    stella_config_wp_bmax,
    stella_config_coillength,
    n_tf_coils,
    len_tf_coil,
):
    """Maximum centering force in the TF coils, from scaling (MN)."""
    return (
        stella_config_centering_force_max_mn
        * f_st_i_total
        / f_st_n_coils
        * b_tf_inboard_peak_symmetric
        / stella_config_wp_bmax
        * stella_config_coillength
        / n_tf_coils
        / len_tf_coil
    )


def calculate_centering_force_min_mn(
    stella_config_centering_force_min_mn,
    f_st_i_total,
    f_st_n_coils,
    b_tf_inboard_peak_symmetric,
    stella_config_wp_bmax,
    stella_config_coillength,
    n_tf_coils,
    len_tf_coil,
):
    """Minimum centering force in the TF coils, from scaling (MN)."""
    return (
        stella_config_centering_force_min_mn
        * f_st_i_total
        / f_st_n_coils
        * b_tf_inboard_peak_symmetric
        / stella_config_wp_bmax
        * stella_config_coillength
        / n_tf_coils
        / len_tf_coil
    )


def calculate_centering_force_avg_mn(
    stella_config_centering_force_avg_mn,
    f_st_i_total,
    f_st_n_coils,
    b_tf_inboard_peak_symmetric,
    stella_config_wp_bmax,
    stella_config_coillength,
    n_tf_coils,
    len_tf_coil,
):
    """Average centering force in the TF coils, from scaling (MN)."""
    return (
        stella_config_centering_force_avg_mn
        * f_st_i_total
        / f_st_n_coils
        * b_tf_inboard_peak_symmetric
        / stella_config_wp_bmax
        * stella_config_coillength
        / n_tf_coils
        / len_tf_coil
    )
