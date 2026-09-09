"""Pure-functional port of `process/models/physics/current_drive.py`'s `CurrentDrive`.
"""

from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    OutputInto,
)

from functional_process.cottax.stated import StatesValues
from functional_process.cottax.paths import current_drive, heat_transport, physics
from functional_process.models.physics.current_drive import (
    calculate_current_drive_ecrh_primary_no_secondary,
    calculate_current_drive_freethy_ecrh_primary_no_secondary,
    electron_cyclotron_primary_powers,
    freethy_electron_cyclotron_efficiency,
    fusion_gain,
    hcd_electric_total_mw,
    hcd_injected_power_total_mw,
    hcd_primary_injected_power_mw,
    hcd_secondary_driven_current,
    user_input_electron_cyclotron_efficiency,
)
from functional_process.vocabulary import PlasmaIgnitionModel

__all__ = [
    "calculate_current_drive_ecrh_primary_no_secondary",
    "calculate_current_drive_freethy_ecrh_primary_no_secondary",
]


class HcdPrimaryEfficiency(ExplicitFunction):
    """The family that owns `.current_drive.eta_cd_hcd_primary`: one occupant per model.
    """


class HcdPrimaryEfficiencyUserInputEcrh(HcdPrimaryEfficiency):
    """`i_hcd_primary == 10` (`USER_INPUT_ELECTRON_CYCLOTRON`)."""

    eta_cd_hcd_primary = OutputInto(current_drive)

    def __call__(
        self,
        eta_cd_norm_ecrh=From(current_drive),
        nd_plasma_electrons_vol_avg=From(physics),
        rmajor=From(physics),
    ):
        return user_input_electron_cyclotron_efficiency(
            eta_cd_norm_ecrh=eta_cd_norm_ecrh,
            nd_plasma_electrons_vol_avg=nd_plasma_electrons_vol_avg,
            rmajor=rmajor,
        )


class HcdPrimaryEfficiencyFreethyEcrhOMode(HcdPrimaryEfficiency):
    """`i_hcd_primary == 13` (`FREETHY_ELECTRON_CYCLOTRON`), O-mode."""

    eta_cd_hcd_primary = OutputInto(current_drive)

    def __call__(
        self,
        temp_plasma_electron_vol_avg_kev=From(physics),
        n_charge_plasma_effective_vol_avg=From(physics),
        rmajor=From(physics),
        nd_plasma_electrons_vol_avg=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
        n_ecrh_harmonic=From(current_drive),
        feffcd=From(current_drive),
    ):
        return freethy_electron_cyclotron_efficiency(
            temp_plasma_electron_vol_avg_kev=temp_plasma_electron_vol_avg_kev,
            n_charge_plasma_effective_vol_avg=n_charge_plasma_effective_vol_avg,
            rmajor=rmajor,
            nd_plasma_electrons_vol_avg=nd_plasma_electrons_vol_avg,
            b_plasma_toroidal_on_axis=b_plasma_toroidal_on_axis,
            n_ecrh_harmonic=n_ecrh_harmonic,
            feffcd=feffcd,
            i_ecrh_wave_mode=0,
        )


class HcdSecondaryHeating(ExplicitFunction):
    """The family that owns what the *secondary* heating system contributes."""


class HcdSecondaryHeatingNone(HcdSecondaryHeating, StatesValues):
    """`i_hcd_secondary == 0` (`NO_CURRENT_DRIVE`): the secondary contributes zero."""

    eta_cd_hcd_secondary = OutputInto(current_drive)
    p_hcd_secondary_extra_heat_mw = OutputInto(current_drive)
    p_hcd_secondary_electric_mw = OutputInto(heat_transport)


class HcdSecondaryDrivenCurrent(ExplicitFunction):
    """cottax node: `hcd_secondary_driven_current`, ports declared."""

    c_hcd_secondary_driven = OutputInto(current_drive)
    f_c_plasma_hcd_secondary = OutputInto(current_drive)

    def __call__(
        self,
        eta_cd_hcd_secondary=From(current_drive),
        p_hcd_secondary_injected_mw=From(current_drive),
        plasma_current=From(physics),
    ):
        return hcd_secondary_driven_current(
            eta_cd_hcd_secondary=eta_cd_hcd_secondary,
            p_hcd_secondary_injected_mw=p_hcd_secondary_injected_mw,
            plasma_current=plasma_current,
        )


class HcdPrimaryInjectedPower(ExplicitFunction):
    """cottax node: `hcd_primary_injected_power_mw`, ports declared."""

    p_hcd_primary_injected_mw = OutputInto(current_drive)

    def __call__(
        self,
        f_c_plasma_auxiliary=From(physics),
        f_c_plasma_hcd_secondary=From(current_drive),
        plasma_current=From(physics),
        eta_cd_hcd_primary=From(current_drive),
    ):
        return hcd_primary_injected_power_mw(
            f_c_plasma_auxiliary=f_c_plasma_auxiliary,
            f_c_plasma_hcd_secondary=f_c_plasma_hcd_secondary,
            plasma_current=plasma_current,
            eta_cd_hcd_primary=eta_cd_hcd_primary,
        )


class HcdPrimaryPowers(ExplicitFunction):
    """The family that owns the primary system's wall-plug and per-technology powers."""


class HcdPrimaryPowersElectronCyclotronNoSecondary(HcdPrimaryPowers):
    """Primary method `ELECTRON_CYCLOTRON` (`i_hcd_primary` 3, 7, 10, 13), secondary 0.
    """

    p_hcd_ecrh_injected_total_mw = OutputInto(current_drive)
    p_hcd_ecrh_electric_mw = OutputInto(current_drive)
    eta_hcd_primary_injector_wall_plug = OutputInto(current_drive)
    p_hcd_primary_electric_mw = OutputInto(heat_transport)

    def __call__(
        self,
        p_hcd_primary_injected_mw=From(current_drive),
        p_hcd_primary_extra_heat_mw=From(current_drive),
        eta_ecrh_injector_wall_plug=From(current_drive),
    ):
        return electron_cyclotron_primary_powers(
            p_hcd_ecrh_injected_secondary_mw=0.0,
            p_hcd_primary_injected_mw=p_hcd_primary_injected_mw,
            p_hcd_primary_extra_heat_mw=p_hcd_primary_extra_heat_mw,
            eta_ecrh_injector_wall_plug=eta_ecrh_injector_wall_plug,
        )


class HcdInjectedPowerTotal(ExplicitFunction):
    """cottax node: `hcd_injected_power_total_mw`, ports declared. Switch independent."""

    p_hcd_injected_total_mw = OutputInto(current_drive)

    def __call__(
        self,
        p_hcd_primary_injected_mw=From(current_drive),
        p_hcd_primary_extra_heat_mw=From(current_drive),
        p_hcd_secondary_injected_mw=From(current_drive),
        p_hcd_secondary_extra_heat_mw=From(current_drive),
    ):
        return hcd_injected_power_total_mw(
            p_hcd_primary_injected_mw=p_hcd_primary_injected_mw,
            p_hcd_primary_extra_heat_mw=p_hcd_primary_extra_heat_mw,
            p_hcd_secondary_injected_mw=p_hcd_secondary_injected_mw,
            p_hcd_secondary_extra_heat_mw=p_hcd_secondary_extra_heat_mw,
        )


class HcdElectricTotal(ExplicitFunction):
    """The family that owns `.heat_transport.p_hcd_electric_total_mw`."""


class HcdElectricTotalNonIgnited(HcdElectricTotal):
    """`i_plasma_ignited == 0` (`NON_IGNITED`): the sum of the two systems' wall plugs.
    """

    p_hcd_electric_total_mw = OutputInto(heat_transport)

    def __call__(
        self,
        p_hcd_primary_electric_mw=From(heat_transport),
        p_hcd_secondary_electric_mw=From(heat_transport),
    ):
        return hcd_electric_total_mw(
            p_hcd_primary_electric_mw=p_hcd_primary_electric_mw,
            p_hcd_secondary_electric_mw=p_hcd_secondary_electric_mw,
            i_plasma_ignited=PlasmaIgnitionModel.NON_IGNITED,
        )


class HcdElectricTotalIgnited(HcdElectricTotal):
    """`i_plasma_ignited == 1` (`IGNITED`): zero, and two reads that are not reads."""

    p_hcd_electric_total_mw = OutputInto(heat_transport)

    def __call__(self):
        return hcd_electric_total_mw(
            p_hcd_primary_electric_mw=0.0,
            p_hcd_secondary_electric_mw=0.0,
            i_plasma_ignited=PlasmaIgnitionModel.IGNITED,
        )


class FusionGain(ExplicitFunction):
    """cottax node: `fusion_gain`, ports declared."""

    big_q_plasma = OutputInto(current_drive)

    def __call__(
        self,
        p_fusion_total_mw=From(physics),
        p_hcd_injected_total_mw=From(current_drive),
        p_beam_orbit_loss_mw=From(current_drive),
        p_plasma_ohmic_mw=From(physics),
    ):
        return fusion_gain(
            p_fusion_total_mw=p_fusion_total_mw,
            p_hcd_injected_total_mw=p_hcd_injected_total_mw,
            p_beam_orbit_loss_mw=p_beam_orbit_loss_mw,
            p_plasma_ohmic_mw=p_plasma_ohmic_mw,
        )
