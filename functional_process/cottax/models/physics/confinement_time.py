"""Pure-functional port of `process/models/physics/confinement_time.py`."""

from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    FromExactly,
    OutputInto,
)

from functional_process.cottax.paths import current_drive, physics, stellarator
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


class IterPhysicsBasisElongation(ExplicitFunction):
    """cottax node: `calculate_iter_physics_basis_elongation`, ports declared."""

    kappa_ipb = OutputInto(physics)

    def __call__(
        self,
        vol_plasma=From(physics),
        rmajor=From(physics),
        rminor=From(physics),
    ):
        return calculate_iter_physics_basis_elongation(vol_plasma, rmajor, rminor)


class ConfinementScalingInputs(ExplicitFunction):
    """The unit conversions every scaling law takes as arguments."""

    nd_plasma_electron_line_19 = OutputInto(physics)
    cur_plasma_ma = OutputInto(physics)

    def __call__(
        self,
        nd_plasma_electron_line=From(physics),
        plasma_current=From(physics),
    ):
        return confinement_scaling_inputs(nd_plasma_electron_line, plasma_current)


class PlasmaPowerLoss(ExplicitFunction):
    """The family that owns `.physics.p_plasma_loss_mw`: the head, one occupant per arm.
    """


class PlasmaPowerLossIgnitedCoreRadiation(PlasmaPowerLoss):
    """`i_plasma_ignited == IGNITED` and `i_rad_loss == CORE_ONLY` -- both runs' arm."""

    p_plasma_loss_mw = OutputInto(physics)

    def __call__(
        self,
        f_p_alpha_plasma_deposited=From(physics),
        p_alpha_total_mw=From(physics),
        p_non_alpha_charged_mw=From(physics),
        p_plasma_ohmic_mw=From(physics),
        pden_plasma_core_rad_mw=From(physics),
        vol_plasma=From(physics),
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


class PlasmaPowerLossNonIgnitedCoreRadiation(PlasmaPowerLoss):
    """`i_plasma_ignited == NON_IGNITED` and `i_rad_loss == CORE_ONLY`."""

    p_plasma_loss_mw = OutputInto(physics)

    def __call__(
        self,
        f_p_alpha_plasma_deposited=From(physics),
        p_alpha_total_mw=From(physics),
        p_non_alpha_charged_mw=From(physics),
        p_plasma_ohmic_mw=From(physics),
        p_hcd_injected_total_mw=From(current_drive),
        pden_plasma_core_rad_mw=From(physics),
        vol_plasma=From(physics),
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


class ConfinementTimeScaling(ExplicitFunction):
    """The family that owns `.physics.t_electron_confinement`: one occupant per law."""


class Iss04ConfinementTime(ConfinementTimeScaling):
    """ISS04 stellarator scaling. `ConfinementTimeModel.ISS04_STELLARATOR` (38)."""

    t_electron_confinement = OutputInto(physics)

    def __call__(
        self,
        rminor=From(physics),
        rmajor=From(physics),
        nd_plasma_electron_line_19=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
        p_plasma_loss_mw=From(physics),
        iotabar=FromExactly(stellarator.iotabar),
    ):
        return iss04_stellarator_confinement_time(
            rminor,
            rmajor,
            nd_plasma_electron_line_19,
            b_plasma_toroidal_on_axis,
            p_plasma_loss_mw,
            iotabar,
        )


class IterIpb98y2ConfinementTime(ConfinementTimeScaling):
    """IPB98(y,2) ELMy H-mode scaling."""

    t_electron_confinement = OutputInto(physics)

    def __call__(
        self,
        cur_plasma_ma=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
        nd_plasma_electron_line_19=From(physics),
        p_plasma_loss_mw=From(physics),
        rmajor=From(physics),
        kappa_ipb=From(physics),
        aspect=From(physics),
        afuel=FromExactly(physics.m_fuel_amu),
    ):
        return iter_ipb98y2_confinement_time(
            cur_plasma_ma,
            b_plasma_toroidal_on_axis,
            nd_plasma_electron_line_19,
            p_plasma_loss_mw,
            rmajor,
            kappa_ipb,
            aspect,
            afuel,
        )


class ConfinementTail(ExplicitFunction):
    """The family that owns everything downstream of the chosen law."""


class ConfinementTailCoreRadiation(ConfinementTail):
    """`i_rad_loss == CORE_ONLY`: `hstar` degrades on synchrotron plus inner radiation.
    """

    pden_electron_transport_loss_mw = OutputInto(physics)
    pden_ion_transport_loss_mw = OutputInto(physics)
    t_electron_energy_confinement = OutputInto(physics)
    t_ion_energy_confinement = OutputInto(physics)
    t_energy_confinement = OutputInto(physics)
    hstar = OutputInto(physics)
    t_energy_confinement_beta = OutputInto(physics)

    def __call__(
        self,
        t_electron_confinement=From(physics),
        hfact=From(physics),
        p_plasma_loss_mw=From(physics),
        pden_plasma_sync_mw=From(physics),
        p_plasma_inner_rad_mw=From(physics),
        vol_plasma=From(physics),
        eden_plasma_ions_thermal_vol_avg=From(physics),
        eden_plasma_electrons_thermal_vol_avg=From(physics),
        e_plasma_beta=From(physics),
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


class DoubleAndTripleProduct(ExplicitFunction):
    """cottax node: `calculate_double_and_triple_product`, ports declared."""

    ntau = OutputInto(physics)
    nTtau = OutputInto(physics)

    def __call__(
        self,
        nd_plasma_electrons_vol_avg=From(physics),
        temp_plasma_electron_vol_avg_kev=From(physics),
        t_energy_confinement=From(physics),
    ):
        return calculate_double_and_triple_product(
            nd_plasma_electrons_vol_avg,
            temp_plasma_electron_vol_avg_kev,
            t_energy_confinement,
        )
