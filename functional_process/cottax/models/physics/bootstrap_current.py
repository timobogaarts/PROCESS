"""Pure-functional port of the tokamak bootstrap-current chain."""

from cottax.assemblies.jax import V

from functional_process.cottax.paths import current_drive, physics
from functional_process.cottax.stated import states
from functional_process.models.physics.bootstrap_current import (
    _beta_poloidal_sauter,
    _beta_poloidal_total_sauter,
    _calculate_l31_32_coefficient,
    _calculate_l31_coefficient,
    _calculate_l34_alpha_31_coefficient,
    _electron_collisionality_sauter,
    _ion_collisionality_sauter,
    _trapped_particle_fraction_sauter,
    bootstrap_fraction_sauter,
    calculate_plasma_current_fractions,
    diamagnetic_fraction_scene,
    enforce_bootstrap_current_fraction_max,
    ps_fraction_scene,
    sauter_bootstrap_current_fraction,
)

__all__ = [
    "_beta_poloidal_sauter",
    "_beta_poloidal_total_sauter",
    "_calculate_l31_32_coefficient",
    "_calculate_l31_coefficient",
    "_calculate_l34_alpha_31_coefficient",
    "_electron_collisionality_sauter",
    "_ion_collisionality_sauter",
    "_trapped_particle_fraction_sauter",
    "bootstrap_fraction_sauter",
    "enforce_bootstrap_current_fraction_max",
]


# The BootstrapCurrentFractionScaling family:
# The family that owns `.current_drive.f_c_plasma_bootstrap` under
# `i_bootstrap_current` (`bootstrap_current.py:250-262`, `:264-298`).


# `i_bootstrap_current == SAUTER` (4) -- the arm `large_tokamak_eval` takes.
# `n_plasma_profile_elements`: the profile grid's resolution, read at
# `.physics.n_plasma_profile_elements` (PROCESS's own field, 201).
def _sauter_bootstrap_current_fraction(
    n_plasma_profile_elements,
    radius_plasma_profile_norm,
    nd_plasma_electron_profile,
    temp_plasma_electron_profile_kev,
    a_plasma_poloidal,
    rminor,
    rmajor,
    nd_plasma_ions_total_vol_avg,
    nd_plasma_electrons_vol_avg,
    temp_plasma_ion_vol_avg_kev,
    temp_plasma_electron_vol_avg_kev,
    n_charge_plasma_effective_vol_avg,
    q0,
    q95,
    m_ions_total_amu,
    f_plasma_fuel_helium3,
    b_plasma_toroidal_on_axis,
    plasma_current,
    cboot,
    f_c_plasma_bootstrap_max,
):
    return sauter_bootstrap_current_fraction(
        n_plasma_profile_elements=n_plasma_profile_elements,
        radius_plasma_profile_norm=radius_plasma_profile_norm,
        nd_plasma_electron_profile=nd_plasma_electron_profile,
        temp_plasma_electron_profile_kev=temp_plasma_electron_profile_kev,
        a_plasma_poloidal=a_plasma_poloidal,
        rminor=rminor,
        rmajor=rmajor,
        nd_plasma_ions_total_vol_avg=nd_plasma_ions_total_vol_avg,
        nd_plasma_electrons_vol_avg=nd_plasma_electrons_vol_avg,
        temp_plasma_ion_vol_avg_kev=temp_plasma_ion_vol_avg_kev,
        temp_plasma_electron_vol_avg_kev=temp_plasma_electron_vol_avg_kev,
        n_charge_plasma_effective_vol_avg=n_charge_plasma_effective_vol_avg,
        q0=q0,
        q95=q95,
        m_ions_total_amu=m_ions_total_amu,
        f_plasma_fuel_helium3=f_plasma_fuel_helium3,
        b_plasma_toroidal_on_axis=b_plasma_toroidal_on_axis,
        plasma_current=plasma_current,
        cboot=cboot,
        f_c_plasma_bootstrap_max=f_c_plasma_bootstrap_max,
    )


SauterBootstrapCurrentFraction = physics.function(
    _sauter_bootstrap_current_fraction,
    V.current_drive.f_c_plasma_bootstrap_sauter,
    "j_plasma_bootstrap_sauter_profile",
    V.current_drive.f_c_plasma_bootstrap,
)


# The PlasmaDiamagneticCurrentFraction family:
# The family that owns `.current_drive.f_c_plasma_diamagnetic` under
# `i_diamagnetic_current` (`plasma_current.py:1081-1094`).


# `i_diamagnetic_current == NONE` (0) -- the default, and this input's value.
# `f_c_plasma_diamagnetic`: the diamagnetic current fraction PROCESS never assigns on
# this arm, read at `^stated.current_drive.f_c_plasma_diamagnetic`.
NoDiamagneticCurrent = states(current_drive, "f_c_plasma_diamagnetic")


# `i_diamagnetic_current == SCENE_FIT` (2) -- both tracked spherical tokamaks.
SceneDiamagneticCurrent = current_drive.function(
    diamagnetic_fraction_scene,
    "f_c_plasma_diamagnetic",
    reads={"beta": V.physics.beta_total_vol_avg},
)


# The PlasmaPfirschSchluterCurrentFraction family:
# The family that owns `.current_drive.f_c_plasma_pfirsch_schluter` under
# `i_pfirsch_schluter_current` (`physics.py:538-541`).


# `i_pfirsch_schluter_current == 0` -- the default, and this input's value.
# `f_c_plasma_pfirsch_schluter`: the Pfirsch-Schluter current fraction PROCESS never
# assigns on this arm, read at `^stated.current_drive.f_c_plasma_pfirsch_schluter`.
NoPfirschSchluterCurrent = states(current_drive, "f_c_plasma_pfirsch_schluter")


# `i_pfirsch_schluter_current == 1` -- both tracked spherical tokamaks.
ScenePfirschSchluterCurrent = current_drive.function(
    ps_fraction_scene,
    "f_c_plasma_pfirsch_schluter",
    reads={
        "beta": V.physics.beta_total_vol_avg,
    },
)


# cottax node: `calculate_plasma_current_fractions`, ports declared.
PlasmaCurrentFractions = current_drive.function(
    calculate_plasma_current_fractions,
    "f_c_plasma_internal",
    V.physics.f_c_plasma_auxiliary,
    V.physics.f_c_plasma_inductive,
)
