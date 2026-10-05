"""Pure-functional port of `process/models/physics/confinement_time.py`."""

from cottax.relational_pytree import V

from functional_process.cottax.paths import physics
from functional_process.models.physics.confinement_time import (
    calculate_confinement_time,
    calculate_double_and_triple_product,
    calculate_iter_physics_basis_elongation,
    christiansen_confinement_time,
    confinement_from_scaling,
    confinement_scaling_inputs,
    ds03_confinement_time,
    goldston_confinement_time,
    gyro_reduced_bohm_confinement_time,
    hubbard_lower_confinement_time,
    hubbard_nominal_confinement_time,
    hubbard_upper_confinement_time,
    iss04_stellarator_confinement_time,
    iss95_stellarator_confinement_time,
    iter_89_0_confinement_time,
    iter_89p_confinement_time,
    iter_93h_confinement_time,
    iter_96p_confinement_time,
    iter_h90_p_amended_confinement_time,
    iter_h90_p_confinement_time,
    iter_h97p_confinement_time,
    iter_h97p_elmy_confinement_time,
    iter_ipb98y1_confinement_time,
    iter_ipb98y2_confinement_time,
    iter_ipb98y3_confinement_time,
    iter_ipb98y4_confinement_time,
    iter_ipb98y_confinement_time,
    iter_pb98py_confinement_time,
    itpa20_confinement_time,
    itpa20_il_confinement_time,
    jaeri_confinement_time,
    kaye_big_confinement_time,
    kaye_confinement_time,
    kaye_goldston_confinement_time,
    lackner_gottardi_confinement_time,
    lackner_gottardi_stellarator_confinement_time,
    lang_high_density_confinement_time,
    menard_nstx_confinement_time,
    menard_nstx_petty08_hybrid_confinement_time,
    merezhkin_muhkovatov_confinement_time,
    mirnov_confinement_time,
    murari_confinement_time,
    ncst_confinement_time,
    neo_alcator_confinement_time,
    neo_kaye_confinement_time,
    nstx_gyro_bohm_confinement_time,
    paz_soldan_nt_confinement_time,
    petty08_confinement_time,
    plasma_power_loss_mw,
    rebut_lallia_confinement_time,
    riedel_h_confinement_time,
    riedel_l_confinement_time,
    shimomura_confinement_time,
    sudo_et_al_confinement_time,
    t10_confinement_time,
    valovic_elmy_confinement_time,
)
from functional_process.vocabulary import (
    ConfinementRadiationLossModel,
    PlasmaIgnitionModel,
)

__all__ = [
    "calculate_confinement_time",
    "christiansen_confinement_time",
    "ds03_confinement_time",
    "goldston_confinement_time",
    "gyro_reduced_bohm_confinement_time",
    "hubbard_lower_confinement_time",
    "hubbard_nominal_confinement_time",
    "hubbard_upper_confinement_time",
    "iss95_stellarator_confinement_time",
    "iter_89_0_confinement_time",
    "iter_89p_confinement_time",
    "iter_93h_confinement_time",
    "iter_96p_confinement_time",
    "iter_h90_p_amended_confinement_time",
    "iter_h90_p_confinement_time",
    "iter_h97p_confinement_time",
    "iter_h97p_elmy_confinement_time",
    "iter_ipb98y1_confinement_time",
    "iter_ipb98y3_confinement_time",
    "iter_ipb98y4_confinement_time",
    "iter_ipb98y_confinement_time",
    "iter_pb98py_confinement_time",
    "itpa20_confinement_time",
    "itpa20_il_confinement_time",
    "jaeri_confinement_time",
    "kaye_big_confinement_time",
    "kaye_confinement_time",
    "kaye_goldston_confinement_time",
    "lackner_gottardi_confinement_time",
    "lackner_gottardi_stellarator_confinement_time",
    "lang_high_density_confinement_time",
    "menard_nstx_confinement_time",
    "menard_nstx_petty08_hybrid_confinement_time",
    "merezhkin_muhkovatov_confinement_time",
    "mirnov_confinement_time",
    "murari_confinement_time",
    "ncst_confinement_time",
    "neo_alcator_confinement_time",
    "neo_kaye_confinement_time",
    "nstx_gyro_bohm_confinement_time",
    "paz_soldan_nt_confinement_time",
    "petty08_confinement_time",
    "rebut_lallia_confinement_time",
    "riedel_h_confinement_time",
    "riedel_l_confinement_time",
    "shimomura_confinement_time",
    "sudo_et_al_confinement_time",
    "t10_confinement_time",
    "valovic_elmy_confinement_time",
]


# cottax node: `calculate_iter_physics_basis_elongation`, ports declared.
IterPhysicsBasisElongation = physics.function(
    calculate_iter_physics_basis_elongation, "kappa_ipb"
)


# The unit conversions every scaling law takes as arguments.
ConfinementScalingInputs = physics.function(
    confinement_scaling_inputs, "nd_plasma_electron_line_19", "cur_plasma_ma"
)


# The PlasmaPowerLoss family:
# The family that owns `.physics.p_plasma_loss_mw`: the head, one occupant per arm.


# `i_plasma_ignited == IGNITED` and `i_rad_loss == CORE_ONLY` -- both runs' arm.
def _plasma_power_loss_ignited_core_radiation(
    f_p_alpha_plasma_deposited,
    p_alpha_total_mw,
    p_non_alpha_charged_mw,
    p_plasma_ohmic_mw,
    pden_plasma_core_rad_mw,
    vol_plasma,
):
    return plasma_power_loss_mw(
        f_p_alpha_plasma_deposited=f_p_alpha_plasma_deposited,
        p_alpha_total_mw=p_alpha_total_mw,
        p_non_alpha_charged_mw=p_non_alpha_charged_mw,
        p_plasma_ohmic_mw=p_plasma_ohmic_mw,
        p_hcd_injected_total_mw=0.0,
        pden_plasma_rad_mw=0.0,
        pden_plasma_core_rad_mw=pden_plasma_core_rad_mw,
        vol_plasma=vol_plasma,
        i_plasma_ignited=PlasmaIgnitionModel.IGNITED,
        i_rad_loss=ConfinementRadiationLossModel.CORE_ONLY,
    )


PlasmaPowerLossIgnitedCoreRadiation = physics.function(
    _plasma_power_loss_ignited_core_radiation, "p_plasma_loss_mw"
)


# `i_plasma_ignited == NON_IGNITED` and `i_rad_loss == CORE_ONLY`.
def _plasma_power_loss_non_ignited_core_radiation(
    f_p_alpha_plasma_deposited,
    p_alpha_total_mw,
    p_non_alpha_charged_mw,
    p_plasma_ohmic_mw,
    p_hcd_injected_total_mw,
    pden_plasma_core_rad_mw,
    vol_plasma,
):
    return plasma_power_loss_mw(
        f_p_alpha_plasma_deposited=f_p_alpha_plasma_deposited,
        p_alpha_total_mw=p_alpha_total_mw,
        p_non_alpha_charged_mw=p_non_alpha_charged_mw,
        p_plasma_ohmic_mw=p_plasma_ohmic_mw,
        p_hcd_injected_total_mw=p_hcd_injected_total_mw,
        pden_plasma_rad_mw=0.0,
        pden_plasma_core_rad_mw=pden_plasma_core_rad_mw,
        vol_plasma=vol_plasma,
        i_plasma_ignited=PlasmaIgnitionModel.NON_IGNITED,
        i_rad_loss=ConfinementRadiationLossModel.CORE_ONLY,
    )


PlasmaPowerLossNonIgnitedCoreRadiation = physics.function(
    _plasma_power_loss_non_ignited_core_radiation,
    "p_plasma_loss_mw",
)


# The ConfinementTimeScaling family:
# The family that owns `.physics.t_electron_confinement`: one occupant per law.


# ISS04 stellarator scaling. `ConfinementTimeModel.ISS04_STELLARATOR` (38).
Iss04ConfinementTime = physics.function(
    iss04_stellarator_confinement_time,
    "t_electron_confinement",
)


# IPB98(y,2) ELMy H-mode scaling.
IterIpb98y2ConfinementTime = physics.function(
    iter_ipb98y2_confinement_time,
    "t_electron_confinement",
    reads={
        "afuel": V.physics.m_fuel_amu,
    },
)


# The ConfinementTail family:
# The family that owns everything downstream of the chosen law.


# `i_rad_loss == CORE_ONLY`: `hstar` degrades on synchrotron plus inner radiation.
def _confinement_tail_core_radiation(
    t_electron_confinement,
    hfact,
    p_plasma_loss_mw,
    pden_plasma_sync_mw,
    p_plasma_inner_rad_mw,
    vol_plasma,
    eden_plasma_ions_thermal_vol_avg,
    eden_plasma_electrons_thermal_vol_avg,
    e_plasma_beta,
):
    return confinement_from_scaling(
        t_electron_confinement=t_electron_confinement,
        hfact=hfact,
        p_plasma_loss_mw=p_plasma_loss_mw,
        i_rad_loss=ConfinementRadiationLossModel.CORE_ONLY,
        pden_plasma_sync_mw=pden_plasma_sync_mw,
        p_plasma_inner_rad_mw=p_plasma_inner_rad_mw,
        pden_plasma_rad_mw=0.0,
        vol_plasma=vol_plasma,
        eden_plasma_ions_thermal_vol_avg=eden_plasma_ions_thermal_vol_avg,
        eden_plasma_electrons_thermal_vol_avg=eden_plasma_electrons_thermal_vol_avg,
        e_plasma_beta=e_plasma_beta,
    )


ConfinementTailCoreRadiation = physics.function(
    _confinement_tail_core_radiation,
    "pden_electron_transport_loss_mw",
    "pden_ion_transport_loss_mw",
    "t_electron_energy_confinement",
    "t_ion_energy_confinement",
    "t_energy_confinement",
    "hstar",
    "t_energy_confinement_beta",
)


# cottax node: `calculate_double_and_triple_product`, ports declared.
DoubleAndTripleProduct = physics.function(
    calculate_double_and_triple_product,
    "ntau",
    "nTtau",
    reads={"temp_plasma_electrons_vol_avg_kev": "temp_plasma_electron_vol_avg_kev"},
)
