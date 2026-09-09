"""Pure-functional port of the tokamak bootstrap-current chain."""

import equinox as eqx
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.cottax.stated import StatesValues
from functional_process.cottax.paths import current_drive, physics
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


class BootstrapCurrentFractionScaling(ExplicitFunction):
    """The family that owns `.current_drive.f_c_plasma_bootstrap` under
    `i_bootstrap_current` (`bootstrap_current.py:250-262`, `:264-298`).
    """


class SauterBootstrapCurrentFraction(BootstrapCurrentFractionScaling):
    """`i_bootstrap_current == SAUTER` (4) -- the arm `large_tokamak_eval` takes."""

    n_plasma_profile_elements: int = eqx.field(static=True)

    f_c_plasma_bootstrap_sauter = OutputInto(current_drive)
    j_plasma_bootstrap_sauter_profile = OutputInto(physics)
    f_c_plasma_bootstrap = OutputInto(current_drive)

    def __call__(
        self,
        radius_plasma_profile_norm=From(physics),
        nd_plasma_electron_profile=From(physics),
        temp_plasma_electron_profile_kev=From(physics),
        a_plasma_poloidal=From(physics),
        rminor=From(physics),
        rmajor=From(physics),
        nd_plasma_ions_total_vol_avg=From(physics),
        nd_plasma_electrons_vol_avg=From(physics),
        temp_plasma_ion_vol_avg_kev=From(physics),
        temp_plasma_electron_vol_avg_kev=From(physics),
        n_charge_plasma_effective_vol_avg=From(physics),
        q0=From(physics),
        q95=From(physics),
        m_ions_total_amu=From(physics),
        f_plasma_fuel_helium3=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
        plasma_current=From(physics),
        cboot=From(current_drive),
        f_c_plasma_bootstrap_max=From(current_drive),
    ):
        return sauter_bootstrap_current_fraction(
            n_plasma_profile_elements=self.n_plasma_profile_elements,
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


class PlasmaDiamagneticCurrentFraction(ExplicitFunction):
    """The family that owns `.current_drive.f_c_plasma_diamagnetic` under
    `i_diamagnetic_current` (`plasma_current.py:1081-1094`).
    """


class NoDiamagneticCurrent(PlasmaDiamagneticCurrentFraction, StatesValues):
    """`i_diamagnetic_current == NONE` (0) -- the default, and this input's value."""

    f_c_plasma_diamagnetic = OutputInto(current_drive)
    """The diamagnetic current fraction PROCESS never assigns on this arm, read at
    `^stated.current_drive.f_c_plasma_diamagnetic`.
    """


class SceneDiamagneticCurrent(PlasmaDiamagneticCurrentFraction):
    """`i_diamagnetic_current == SCENE_FIT` (2) -- both tracked spherical tokamaks."""

    f_c_plasma_diamagnetic = OutputInto(current_drive)

    def __call__(
        self,
        beta_total_vol_avg=From(physics),
        q95=From(physics),
        q0=From(physics),
    ):
        return diamagnetic_fraction_scene(beta=beta_total_vol_avg, q95=q95, q0=q0)


class PlasmaPfirschSchluterCurrentFraction(ExplicitFunction):
    """The family that owns `.current_drive.f_c_plasma_pfirsch_schluter` under
    `i_pfirsch_schluter_current` (`physics.py:538-541`).
    """


class NoPfirschSchluterCurrent(PlasmaPfirschSchluterCurrentFraction, StatesValues):
    """`i_pfirsch_schluter_current == 0` -- the default, and this input's value."""

    f_c_plasma_pfirsch_schluter = OutputInto(current_drive)
    """The Pfirsch-Schluter current fraction PROCESS never assigns on this arm, read at
    `^stated.current_drive.f_c_plasma_pfirsch_schluter`.
    """


class ScenePfirschSchluterCurrent(PlasmaPfirschSchluterCurrentFraction):
    """`i_pfirsch_schluter_current == 1` -- both tracked spherical tokamaks."""

    f_c_plasma_pfirsch_schluter = OutputInto(current_drive)

    def __call__(self, beta_total_vol_avg=From(physics)):
        return ps_fraction_scene(beta=beta_total_vol_avg)


class PlasmaCurrentFractions(ExplicitFunction):
    """cottax node: `calculate_plasma_current_fractions`, ports declared."""

    f_c_plasma_internal = OutputInto(current_drive)
    f_c_plasma_auxiliary = OutputInto(physics)
    f_c_plasma_inductive = OutputInto(physics)

    def __call__(
        self,
        f_c_plasma_bootstrap=From(current_drive),
        f_c_plasma_diamagnetic=From(current_drive),
        f_c_plasma_pfirsch_schluter=From(current_drive),
        f_c_plasma_non_inductive=From(physics),
    ):
        return calculate_plasma_current_fractions(
            f_c_plasma_bootstrap=f_c_plasma_bootstrap,
            f_c_plasma_diamagnetic=f_c_plasma_diamagnetic,
            f_c_plasma_pfirsch_schluter=f_c_plasma_pfirsch_schluter,
            f_c_plasma_non_inductive=f_c_plasma_non_inductive,
        )
