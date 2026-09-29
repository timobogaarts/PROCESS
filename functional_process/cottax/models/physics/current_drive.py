"""Pure-functional port of `process/models/physics/current_drive.py`'s `CurrentDrive`."""

from cottax.interfaces import V

from functional_process.cottax.paths import current_drive, heat_transport, physics
from functional_process.cottax.stated import states
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


# The HcdPrimaryEfficiency family:
# The family that owns `.current_drive.eta_cd_hcd_primary`: one occupant per model.


# `i_hcd_primary == 10` (`USER_INPUT_ELECTRON_CYCLOTRON`).
HcdPrimaryEfficiencyUserInputEcrh = current_drive.function(
    user_input_electron_cyclotron_efficiency,
    "eta_cd_hcd_primary",
    reads={
        "nd_plasma_electrons_vol_avg": physics,
        "rmajor": physics,
    },
)


# `i_hcd_primary == 13` (`FREETHY_ELECTRON_CYCLOTRON`), O-mode.
def _hcd_primary_efficiency_freethy_ecrh_o_mode(
    temp_plasma_electron_vol_avg_kev,
    n_charge_plasma_effective_vol_avg,
    rmajor,
    nd_plasma_electrons_vol_avg,
    b_plasma_toroidal_on_axis,
    n_ecrh_harmonic,
    feffcd,
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


HcdPrimaryEfficiencyFreethyEcrhOMode = current_drive.function(
    _hcd_primary_efficiency_freethy_ecrh_o_mode,
    "eta_cd_hcd_primary",
    reads={
        "temp_plasma_electron_vol_avg_kev": physics,
        "n_charge_plasma_effective_vol_avg": physics,
        "rmajor": physics,
        "nd_plasma_electrons_vol_avg": physics,
        "b_plasma_toroidal_on_axis": physics,
    },
)


# The HcdSecondaryHeating family:
# The family that owns what the *secondary* heating system contributes.


# `i_hcd_secondary == 0` (`NO_CURRENT_DRIVE`): the secondary contributes zero.
HcdSecondaryHeatingNone = states(
    current_drive,
    "eta_cd_hcd_secondary",
    "p_hcd_secondary_extra_heat_mw",
    V.heat_transport.p_hcd_secondary_electric_mw,
)


# cottax node: `hcd_secondary_driven_current`, ports declared.
HcdSecondaryDrivenCurrent = current_drive.function(
    hcd_secondary_driven_current,
    "c_hcd_secondary_driven",
    "f_c_plasma_hcd_secondary",
    reads={
        "plasma_current": physics,
    },
)


# cottax node: `hcd_primary_injected_power_mw`, ports declared.
HcdPrimaryInjectedPower = current_drive.function(
    hcd_primary_injected_power_mw,
    "p_hcd_primary_injected_mw",
    reads={
        "f_c_plasma_auxiliary": physics,
        "plasma_current": physics,
    },
)


# The HcdPrimaryPowers family:
# The family that owns the primary system's wall-plug and per-technology powers.


# Primary method `ELECTRON_CYCLOTRON`.
#
# `i_hcd_primary` 3, 7, 10, 13), secondary 0.
def _hcd_primary_powers_electron_cyclotron_no_secondary(
    p_hcd_primary_injected_mw, p_hcd_primary_extra_heat_mw, eta_ecrh_injector_wall_plug
):
    return electron_cyclotron_primary_powers(
        p_hcd_ecrh_injected_secondary_mw=0.0,
        p_hcd_primary_injected_mw=p_hcd_primary_injected_mw,
        p_hcd_primary_extra_heat_mw=p_hcd_primary_extra_heat_mw,
        eta_ecrh_injector_wall_plug=eta_ecrh_injector_wall_plug,
    )


HcdPrimaryPowersElectronCyclotronNoSecondary = current_drive.function(
    _hcd_primary_powers_electron_cyclotron_no_secondary,
    "p_hcd_ecrh_injected_total_mw",
    "p_hcd_ecrh_electric_mw",
    "eta_hcd_primary_injector_wall_plug",
    V.heat_transport.p_hcd_primary_electric_mw,
)


# cottax node: `hcd_injected_power_total_mw`, ports declared. Switch independent.
HcdInjectedPowerTotal = current_drive.function(
    hcd_injected_power_total_mw, "p_hcd_injected_total_mw"
)


# The HcdElectricTotal family:
# The family that owns `.heat_transport.p_hcd_electric_total_mw`.


# `i_plasma_ignited == 0`.
#
# `NON_IGNITED`): the sum of the two systems' wall plugs.
def _hcd_electric_total_non_ignited(
    p_hcd_primary_electric_mw, p_hcd_secondary_electric_mw
):
    return hcd_electric_total_mw(
        p_hcd_primary_electric_mw=p_hcd_primary_electric_mw,
        p_hcd_secondary_electric_mw=p_hcd_secondary_electric_mw,
        i_plasma_ignited=PlasmaIgnitionModel.NON_IGNITED,
    )


HcdElectricTotalNonIgnited = heat_transport.function(
    _hcd_electric_total_non_ignited, "p_hcd_electric_total_mw"
)


# `i_plasma_ignited == 1` (`IGNITED`): zero, and two reads that are not reads.
def _hcd_electric_total_ignited():
    return hcd_electric_total_mw(
        p_hcd_primary_electric_mw=0.0,
        p_hcd_secondary_electric_mw=0.0,
        i_plasma_ignited=PlasmaIgnitionModel.IGNITED,
    )


HcdElectricTotalIgnited = heat_transport.function(
    _hcd_electric_total_ignited, "p_hcd_electric_total_mw"
)


# cottax node: `fusion_gain`, ports declared.
FusionGain = current_drive.function(
    fusion_gain,
    "big_q_plasma",
    reads={
        "p_fusion_total_mw": physics,
        "p_plasma_ohmic_mw": physics,
    },
)
