"""Pure-functional port of `process/models/stellarator/coils/forces.py` (registry #11).

Audit record: `functional_process/_audit/units/models/stellarator/coils/forces.md`. All
seven source functions share one shape: a handful of
`.stellarator_config.*`/`.stellarator.*`/ `.tfcoil.*` reads combined by straight-line
arithmetic, no switches, no internal loop, no calls into other models -- the cleanest
file in the coil subsystem. `calculate_maximum_stress` is the one exception worth a
note: the source reads `.tfcoil.max_force_density` off `data`, which is itself written
by `calculate_max_force_density` earlier in the same caller (`st_coil`) -- ported here
as an explicit `max_force_density` argument instead of an implicit `data` read, so the
two functions compose by ordinary argument-passing rather than through a shared mutable
object."""

from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.paths import stellarator, stellarator_config, tfcoil


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


class MaxForceDensity(ExplicitFunction):
    """cottax node: `calculate_max_force_density`."""

    max_force_density = OutputInto(tfcoil)

    def __call__(
        self,
        a_tf_wp_no_insulation=From(tfcoil),
        stella_config_max_force_density=From(stellarator_config),
        f_st_i_total=From(stellarator),
        f_st_n_coils=From(stellarator),
        b_tf_inboard_peak_symmetric=From(tfcoil),
        stella_config_wp_bmax=From(stellarator_config),
        stella_config_wp_area=From(stellarator_config),
    ):
        return calculate_max_force_density(
            a_tf_wp_no_insulation,
            stella_config_max_force_density,
            f_st_i_total,
            f_st_n_coils,
            b_tf_inboard_peak_symmetric,
            stella_config_wp_bmax,
            stella_config_wp_area,
        )


class MaximumStress(ExplicitFunction):
    """cottax node: `calculate_maximum_stress`.

    Reads `max_force_density` as a real graph edge from `MaxForceDensity`, not a
    `data`-mediated re-read -- see module docstring.
    """

    sig_tf_wp = OutputInto(tfcoil)

    def __call__(
        self,
        max_force_density=From(tfcoil),
        dr_tf_wp_with_insulation=From(tfcoil),
    ):
        return calculate_maximum_stress(max_force_density, dr_tf_wp_with_insulation)
