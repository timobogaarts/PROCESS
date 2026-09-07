"""Pure-functional port of PROCESS's constraint set
(`process/core/solver/constraints.py`).
"""

import jax.numpy as jnp

from functional_process.vocabulary import constants
from functional_process.vocabulary import TFCSRadialConfiguration
from functional_process.vocabulary import PlasmaIgnitionModel
from functional_process.vocabulary import DensityLimitModel
from functional_process.vocabulary import BetaComponentLimits
from functional_process.vocabulary import TFConductorModel


def leq(value, bound):
    """`value <= bound`."""
    residual = value - bound
    normalised_residual = (value / bound) - 1.0
    return residual, normalised_residual, value, bound


def geq(value, bound):
    """`value >= bound`."""
    residual = bound - value
    normalised_residual = 1.0 - (value / bound)
    return residual, normalised_residual, value, bound


def eq(value, bound):
    """`value == bound`."""
    residual = value - bound
    normalised_residual = 1.0 - (value / bound)
    return residual, normalised_residual, value, bound


def calculate_plasma_beta(pres_plasma, b_field):
    """Plasma beta from pressure and field."""
    return 2.0 * constants.RMU0 * pres_plasma / (b_field**2)


def constraint_1(
    beta_fast_alpha,
    beta_beam,
    nd_plasma_electrons_vol_avg,
    temp_plasma_electron_density_weighted_kev,
    nd_plasma_ions_total_vol_avg,
    temp_plasma_ion_density_weighted_kev,
    b_plasma_total,
    beta_total_vol_avg,
):
    """Relationship between beta, temperature and density."""
    beta_thermal_total_vol_avg = calculate_plasma_beta(
        pres_plasma=(
            constants.KILOELECTRON_VOLT
            * (
                nd_plasma_electrons_vol_avg * temp_plasma_electron_density_weighted_kev
                + nd_plasma_ions_total_vol_avg * temp_plasma_ion_density_weighted_kev
            )
        ),
        b_field=b_plasma_total,
    )
    return eq(
        beta_fast_alpha + beta_beam + beta_thermal_total_vol_avg,
        beta_total_vol_avg,
    )


def constraint_2(
    i_rad_loss,
    i_plasma_ignited,
    pden_electron_transport_loss_mw,
    pden_ion_transport_loss_mw,
    pden_plasma_rad_mw,
    pden_plasma_core_rad_mw,
    f_p_alpha_plasma_deposited,
    pden_alpha_total_mw,
    pden_non_alpha_charged_mw,
    pden_plasma_ohmic_mw,
    p_hcd_injected_total_mw,
    vol_plasma,
):
    """Global power balance equation (total)."""
    pscaling = pden_electron_transport_loss_mw + pden_ion_transport_loss_mw
    if i_rad_loss == 0:
        pnumerator = pscaling + pden_plasma_rad_mw
    elif i_rad_loss == 1:
        pnumerator = pscaling + pden_plasma_core_rad_mw
    else:
        pnumerator = pscaling

    if PlasmaIgnitionModel(i_plasma_ignited) == PlasmaIgnitionModel.NON_IGNITED:
        pdenom = (
            f_p_alpha_plasma_deposited * pden_alpha_total_mw
            + pden_non_alpha_charged_mw
            + pden_plasma_ohmic_mw
            + p_hcd_injected_total_mw / vol_plasma
        )
    else:
        pdenom = (
            f_p_alpha_plasma_deposited * pden_alpha_total_mw
            + pden_non_alpha_charged_mw
            + pden_plasma_ohmic_mw
        )

    return eq(pnumerator, pdenom)


def constraint_3(
    i_plasma_ignited,
    pden_ion_transport_loss_mw,
    pden_ion_electron_equilibration_mw,
    f_p_alpha_plasma_deposited,
    f_pden_alpha_ions_mw,
    p_hcd_injected_ions_mw,
    vol_plasma,
):
    """Global power balance equation for ions."""
    lhs = pden_ion_transport_loss_mw + pden_ion_electron_equilibration_mw

    if PlasmaIgnitionModel(i_plasma_ignited) == PlasmaIgnitionModel.NON_IGNITED:
        rhs = (
            f_p_alpha_plasma_deposited * f_pden_alpha_ions_mw
            + p_hcd_injected_ions_mw / vol_plasma
        )
    else:
        rhs = f_p_alpha_plasma_deposited * f_pden_alpha_ions_mw

    return eq(lhs, rhs)


def constraint_4(
    i_rad_loss,
    i_plasma_ignited,
    pden_electron_transport_loss_mw,
    pden_plasma_rad_mw,
    pden_plasma_core_rad_mw,
    f_p_alpha_plasma_deposited,
    f_pden_alpha_electron_mw,
    pden_ion_electron_equilibration_mw,
    p_hcd_injected_electrons_mw,
    vol_plasma,
):
    """Global power balance equation for electrons."""
    pscaling = pden_electron_transport_loss_mw
    if i_rad_loss == 0:
        pnumerator = pscaling + pden_plasma_rad_mw
    elif i_rad_loss == 1:
        pnumerator = pscaling + pden_plasma_core_rad_mw
    else:
        pnumerator = pscaling

    if PlasmaIgnitionModel(i_plasma_ignited) == PlasmaIgnitionModel.NON_IGNITED:
        pdenom = (
            f_p_alpha_plasma_deposited * f_pden_alpha_electron_mw
            + pden_ion_electron_equilibration_mw
            + p_hcd_injected_electrons_mw / vol_plasma
        )
    else:
        pdenom = (
            f_p_alpha_plasma_deposited * f_pden_alpha_electron_mw
            + pden_ion_electron_equilibration_mw
        )

    return eq(pnumerator, pdenom)


def constraint_5(
    i_density_limit,
    nd_plasma_electron_line,
    nd_plasma_electrons_vol_avg,
    nd_plasma_electrons_max,
    f_nd_plasma_electron_limit_max,
):
    """Electron density upper limit."""
    bound = nd_plasma_electrons_max * f_nd_plasma_electron_limit_max
    if i_density_limit == DensityLimitModel.GREENWALD:
        return leq(nd_plasma_electron_line, bound)
    return leq(nd_plasma_electrons_vol_avg, bound)


def constraint_6(beta_poloidal_eps, beta_poloidal_eps_max):
    """Epsilon beta-poloidal upper limit."""
    return leq(beta_poloidal_eps, beta_poloidal_eps_max)


def constraint_7(i_plasma_ignited, nd_beam_ions_out, nd_beam_ions):
    """Hot beam ion density consistency."""
    if PlasmaIgnitionModel(i_plasma_ignited) == PlasmaIgnitionModel.IGNITED:
        raise ValueError("constraint_7: do not use if i_plasma_ignited=IGNITED")

    return eq(nd_beam_ions_out, nd_beam_ions)


def constraint_8(pflux_fw_neutron_mw, pflux_fw_neutron_max_mw):
    """Neutron wall load upper limit."""
    return leq(pflux_fw_neutron_mw, pflux_fw_neutron_max_mw)


def constraint_9(p_fusion_total_mw, p_fusion_total_max_mw):
    """Fusion power upper limit."""
    return leq(p_fusion_total_mw, p_fusion_total_max_mw)


def constraint_11(rbld, rmajor):
    """Radial build consistency (equality)."""
    return eq(rbld, rmajor)


def constraint_12(vs_cs_pf_total_pulse, vs_plasma_total_required):
    """Volt-second capability lower limit."""
    return geq(vs_cs_pf_total_pulse, vs_plasma_total_required)


def constraint_13(t_plant_pulse_burn, t_burn_min):
    """Burn time lower limit."""
    return geq(t_plant_pulse_burn, t_burn_min)


def constraint_14(n_beam_decay_lengths_core, n_beam_decay_lengths_core_required):
    """Neutral beam e-decay lengths to plasma centre (equality)."""
    return eq(n_beam_decay_lengths_core, n_beam_decay_lengths_core_required)


def constraint_15(p_plasma_separatrix_mw, p_l_h_threshold_mw, f_h_mode_margin):
    """L-H power threshold limit (H-mode enforcement)."""
    return geq(p_plasma_separatrix_mw, p_l_h_threshold_mw * f_h_mode_margin)


def constraint_16(p_plant_electric_net_mw, p_plant_electric_net_required_mw):
    """Net electric power lower limit."""
    return geq(p_plant_electric_net_mw, p_plant_electric_net_required_mw)


def constraint_17(
    istell,
    f_p_plasma_separatrix_rad,
    f_p_plasma_separatrix_rad_max,
    psolradmw,
    p_plasma_heating_total_mw,
):
    """Plasma radiation fraction upper limit."""
    if istell != 0:
        f_rad_sol = psolradmw / p_plasma_heating_total_mw
        value = f_p_plasma_separatrix_rad - f_rad_sol
    else:
        value = f_p_plasma_separatrix_rad

    return leq(value, f_p_plasma_separatrix_rad_max)


def constraint_18(pflux_div_heat_load_mw, pflux_div_heat_load_max_mw):
    """Divertor heat load upper limit."""
    return leq(pflux_div_heat_load_mw, pflux_div_heat_load_max_mw)


def constraint_19(p_cp_resistive_mw, p_tf_leg_resistive_mw, mvalim):
    """MVA (power) upper limit: resistive TF coil set."""
    totmva = p_cp_resistive_mw + p_tf_leg_resistive_mw
    return leq(totmva, mvalim)


def constraint_20(radius_beam_tangency, radius_beam_tangency_max):
    """Neutral beam tangency radius upper limit."""
    return leq(radius_beam_tangency, radius_beam_tangency_max)


def constraint_21(rminor, rminor_min):
    """Minor radius lower limit."""
    return geq(rminor, rminor_min)


def constraint_22(p_l_h_threshold_mw, f_l_mode_margin, p_plasma_separatrix_mw):
    """L-H power threshold limit, to enforce L-mode."""
    return geq(p_l_h_threshold_mw, f_l_mode_margin * p_plasma_separatrix_mw)


def constraint_23(
    rminor,
    dr_fw_plasma_gap_outboard,
    dr_fw_outboard,
    dr_blkt_outboard,
    f_r_conducting_wall,
):
    """Conducting shell radius / rminor upper limit."""
    rcw = rminor + dr_fw_plasma_gap_outboard + dr_fw_outboard + dr_blkt_outboard
    return leq(rcw, f_r_conducting_wall * rminor)


def constraint_24(
    i_beta_component,
    istell,
    beta_total_vol_avg,
    beta_thermal_vol_avg,
    beta_beam,
    beta_toroidal_vol_avg,
    beta_vol_avg_max,
):
    """Beta upper limit."""
    if i_beta_component == BetaComponentLimits.TOTAL or istell != 0:
        value = beta_total_vol_avg
    elif i_beta_component == BetaComponentLimits.THERMAL:
        value = beta_thermal_vol_avg
    elif i_beta_component == BetaComponentLimits.THERMAL_AND_BEAM:
        value = beta_thermal_vol_avg + beta_beam
    elif i_beta_component == BetaComponentLimits.TOROIDAL:
        value = beta_toroidal_vol_avg
    else:
        raise ValueError(
            f"constraint_24: i_beta_component={i_beta_component!r} is not a member of "
            f"BetaComponentLimits"
        )

    return leq(value, beta_vol_avg_max)


def constraint_25(b_tf_inboard_peak_with_ripple, b_tf_inboard_max):
    """Peak toroidal field upper limit."""
    return leq(b_tf_inboard_peak_with_ripple, b_tf_inboard_max)


def constraint_26(j_cs_flat_top_end, j_cs_critical_flat_top_end, fjohc):
    """Central Solenoid current density upper limit at end-of-flattop."""
    return leq(j_cs_flat_top_end / j_cs_critical_flat_top_end, fjohc)


def constraint_27(j_cs_pulse_start, j_cs_critical_pulse_start, fjohc0):
    """Central Solenoid current density upper limit at beginning-of-pulse."""
    return leq(j_cs_pulse_start / j_cs_critical_pulse_start, fjohc0)


def constraint_28(i_plasma_ignited, big_q_plasma, big_q_plasma_min):
    """Fusion gain (big Q) lower limit."""
    if PlasmaIgnitionModel(i_plasma_ignited) != PlasmaIgnitionModel.NON_IGNITED:
        raise ValueError("constraint_28: not valid if i_plasma_ignited != NON_IGNITED")

    return geq(big_q_plasma, big_q_plasma_min)


def constraint_29(rmajor, rminor, rinboard):
    """Inboard major radius consistency."""
    return eq(rmajor - rminor, rinboard)


def constraint_30(p_hcd_injected_total_mw, p_hcd_injected_max):
    """Injection power upper limit."""
    return leq(p_hcd_injected_total_mw, p_hcd_injected_max)


def constraint_31(sig_tf_case, sig_tf_case_max):
    """TF coil case stress upper limit (SCTF)."""
    return leq(sig_tf_case, sig_tf_case_max)


def constraint_32(sig_tf_wp, sig_tf_wp_max):
    """TF coil conduit stress upper limit (SCTF)."""
    return leq(sig_tf_wp, sig_tf_wp_max)


def constraint_33(j_tf_wp, j_tf_wp_critical, f_j_tf_wp_critical_max):
    """TF coil operating/critical current density upper limit (SCTF)."""
    return leq(j_tf_wp, j_tf_wp_critical * f_j_tf_wp_critical_max)


def constraint_34(v_tf_coil_dump_quench_kv, v_tf_coil_dump_quench_max_kv):
    """TF coil dump voltage upper limit (SCTF)."""
    return leq(v_tf_coil_dump_quench_kv, v_tf_coil_dump_quench_max_kv)


def constraint_35(j_tf_wp, j_tf_wp_quench_heat_max):
    """TF coil J_wp upper limit for quench protection."""
    return leq(j_tf_wp, j_tf_wp_quench_heat_max)


def constraint_36(
    temp_tf_superconductor_margin,
    temp_tf_superconductor_margin_min,
):
    """TF coil superconductor temperature margin lower limit."""
    return geq(temp_tf_superconductor_margin, temp_tf_superconductor_margin_min)


def constraint_37(
    eta_cd_norm_hcd_primary,
    eta_cd_norm_hcd_primary_max,
):
    """Current drive gamma upper limit."""
    return leq(eta_cd_norm_hcd_primary, eta_cd_norm_hcd_primary_max)


def constraint_39(
    temp_fw_peak,
    temp_fw_max,
):
    """First wall temperature upper limit."""
    return leq(temp_fw_peak, temp_fw_max)


def constraint_40(
    p_hcd_injected_total_mw,
    p_hcd_injected_min_mw,
):
    """Auxiliary power lower limit."""
    return geq(p_hcd_injected_total_mw, p_hcd_injected_min_mw)


def constraint_41(
    t_plant_pulse_plasma_current_ramp_up,
    t_current_ramp_up_min,
):
    """Plasma current ramp-up time lower limit."""
    return geq(t_plant_pulse_plasma_current_ramp_up, t_current_ramp_up_min)


def constraint_42(
    t_plant_pulse_total,
    t_cycle_min,
):
    """Cycle time lower limit."""
    return geq(t_plant_pulse_total, t_cycle_min)


def constraint_43(
    i_tf_sup,
    temp_cp_average,
    tcpav2,
):
    """Average centrepost temperature consistency equation (TART)."""
    if i_tf_sup == TFConductorModel.WATER_COOLED_COPPER:
        temp_cp_average -= constants.TEMP_ROOM
        tcpav2 -= constants.TEMP_ROOM

    return eq(temp_cp_average, tcpav2)


def constraint_44(
    i_tf_sup,
    temp_cp_max,
    temp_cp_peak,
):
    """Centrepost temperature upper limit (TART)."""
    if i_tf_sup == TFConductorModel.WATER_COOLED_COPPER:
        temp_cp_max -= constants.TEMP_ROOM
        temp_cp_peak -= constants.TEMP_ROOM

    return leq(temp_cp_peak, temp_cp_max)


def constraint_45(itart, q95, q95_min):
    """Edge safety factor lower limit (TART)."""
    if itart == 0:
        raise ValueError("constraint_45: itart=0 -- constraint 45 requires itart=1")

    return geq(q95, q95_min)


def constraint_46(itart, eps, plasma_current, c_tf_total):
    """I_p / I_rod upper limit (TART)."""
    if itart == 0:
        raise ValueError("constraint_46: itart=0 -- constraint 46 requires itart=1")

    cratmx = 1.0 + 4.91 * (eps - 0.62)
    return leq(plasma_current / c_tf_total, cratmx)


def constraint_48(beta_poloidal_vol_avg, beta_poloidal_max):
    """Poloidal beta upper limit."""
    return leq(beta_poloidal_vol_avg, beta_poloidal_max)


def constraint_51(vs_plasma_ramp_required, vs_cs_pf_total_ramp):
    """Startup flux = available startup flux (equality)."""
    return eq(abs(vs_plasma_ramp_required), vs_cs_pf_total_ramp)


def constraint_53(flu_tf_neutron_fast_peak, flu_tf_neutron_fast_max):
    """Fast neutron fluence on TF coil, upper limit."""
    return leq(flu_tf_neutron_fast_peak, flu_tf_neutron_fast_max)


def constraint_54(ptfnucpm3, ptfnucmax):
    """Peak TF coil nuclear heating, upper limit."""
    return leq(ptfnucpm3, ptfnucmax)


def constraint_56(p_plasma_separatrix_rmajor_mw, p_plasma_separatrix_rmajor_max_mw):
    """P_sep / R0 upper limit."""
    return leq(p_plasma_separatrix_rmajor_mw, p_plasma_separatrix_rmajor_max_mw)


def constraint_59(f_p_beam_shine_through, f_p_beam_shine_through_max):
    """Neutral beam shine-through fraction upper limit."""
    return leq(f_p_beam_shine_through, f_p_beam_shine_through_max)


def constraint_60(temp_cs_superconductor_margin, temp_cs_superconductor_margin_min):
    """Central Solenoid s/c temperature margin lower limit."""
    return geq(temp_cs_superconductor_margin, temp_cs_superconductor_margin_min)


def constraint_61(f_t_plant_available, f_t_plant_available_min):
    """Plant availability lower limit."""
    return geq(f_t_plant_available, f_t_plant_available_min)


def constraint_62(f_t_alpha_energy_confinement, f_t_alpha_energy_confinement_min):
    """Lower limit on the ratio of alpha-particle to energy confinement times."""
    return geq(f_t_alpha_energy_confinement, f_t_alpha_energy_confinement_min)


def constraint_63(n_iter_vacuum_pumps, n_tf_coils):
    """Upper limit on the number of high-vacuum pumps (`i_vacuum_pumping = simple`)."""
    return leq(n_iter_vacuum_pumps, n_tf_coils)


def constraint_64(
    n_charge_plasma_effective_vol_avg, n_charge_plasma_effective_vol_avg_max
):
    """Upper limit on volume-averaged plasma effective charge (Zeff)."""
    return leq(n_charge_plasma_effective_vol_avg, n_charge_plasma_effective_vol_avg_max)


def constraint_65(vv_stress_quench, max_vv_stress):
    """Upper limit on vacuum vessel stress during a TF coil quench."""
    return leq(vv_stress_quench, max_vv_stress)


def constraint_66(peakpoloidalpower, maxpoloidalpower):
    """Upper limit on rate of change of energy in the poloidal field."""
    return leq(peakpoloidalpower, maxpoloidalpower)


def constraint_67(pflux_fw_rad_max_mw, pflux_fw_rad_max):
    """Simple upper limit on radiation wall load."""
    return leq(pflux_fw_rad_max_mw, pflux_fw_rad_max)


def constraint_68(
    i_q95_fixed,
    p_plasma_separatrix_mw,
    b_plasma_toroidal_on_axis,
    q95,
    q95_fixed,
    aspect,
    rmajor,
    p_div_bt_q_aspect_rmajor_mw,
    p_div_bt_q_aspect_rmajor_max_mw,
):
    """Upper limit on Psep scaling (PsepBt / q95*A*R0)."""
    if i_q95_fixed == 1:
        value = (p_plasma_separatrix_mw * b_plasma_toroidal_on_axis) / (
            q95_fixed * aspect * rmajor
        )
    else:
        value = p_div_bt_q_aspect_rmajor_mw

    return leq(value, p_div_bt_q_aspect_rmajor_max_mw)


def constraint_72(
    i_tf_bucking,
    i_tf_inside_cs,
    stress_shear_cs_peak,
    sig_tf_cs_bucked,
    stress_cs_steel_max,
):
    """Upper limit on Central Solenoid Tresca yield stress."""
    if i_tf_bucking >= 2 and i_tf_inside_cs == TFCSRadialConfiguration.TF_OUTSIDE_CS:
        # `jnp.maximum`, not the builtin: `max(a, b)` evaluates `b > a` and calls
        # `bool()` on the result, which raises `TracerBoolConversionError` the moment
        # either operand is traced -- so this line made the whole tokamak MDF problem
        # untraceable while SAND's eager path walked straight past it
        # (`_audit/optimise_design.md` §16). It also *silently* discarded a `nan`:
        # `nan > a` is `False`, so the builtin returns `a` and the missing-value alarm
        # `sand_harness.UNWRITTEN_BY_PROCESS` exists to raise never fired here.
        value = jnp.maximum(stress_shear_cs_peak, sig_tf_cs_bucked)
    else:
        value = stress_shear_cs_peak

    return leq(value, stress_cs_steel_max)


def constraint_73(p_plasma_separatrix_mw, p_l_h_threshold_mw, p_hcd_injected_total_mw):
    """Lower limit: separatrix power >= L-H threshold power + auxiliary power."""
    return geq(p_plasma_separatrix_mw, p_l_h_threshold_mw + p_hcd_injected_total_mw)


def constraint_74(temp_croco_quench, temp_croco_quench_max):
    """Upper limit on TF coil quench temperature."""
    return leq(temp_croco_quench, temp_croco_quench_max)


def constraint_75(coppera_m2, tf_coppera_m2_max):
    """Upper limit on TF coil current / copper area."""
    return leq(coppera_m2, tf_coppera_m2_max)


def constraint_76(
    kappa,
    triang,
    aspect,
    p_plasma_separatrix_mw,
    nd_plasma_electron_max_array_7,
    nd_plasma_separatrix_electron,
):
    """Upper limit for the Eich critical separatrix density model."""
    alpha_crit = (kappa**1.2) * (1.0 + 1.5 * triang)
    nd_plasma_separatrix_electron_eich_max = (
        5.9
        * alpha_crit
        * (aspect ** (-2.0 / 7.0))
        * (((1.0 + (kappa**2.0)) / 2.0) ** (-6.0 / 7.0))
        * ((p_plasma_separatrix_mw * 1.0e6) ** (-11.0 / 70.0))
        * nd_plasma_electron_max_array_7
    )

    return leq(nd_plasma_separatrix_electron, nd_plasma_separatrix_electron_eich_max)


def constraint_77(c_tf_turn, c_tf_turn_max):
    """Maximum TF coil current per turn upper limit."""
    return leq(c_tf_turn, c_tf_turn_max)


def constraint_78(fzactual, fzmin):
    """Reinke criterion, divertor impurity fraction lower limit."""
    return geq(fzactual, fzmin)


def constraint_79(b_cs_peak_flat_top_end, b_cs_peak_pulse_start, b_cs_limit_max):
    """Maximum central solenoid (CS) field."""
    peak = jnp.maximum(b_cs_peak_flat_top_end, b_cs_peak_pulse_start)
    return leq(peak, b_cs_limit_max)


def constraint_80(p_plasma_separatrix_mw, p_plasma_separatrix_min_mw):
    """Lower limit on power crossing the separatrix."""
    return geq(p_plasma_separatrix_mw, p_plasma_separatrix_min_mw)


def constraint_81(nd_plasma_electron_on_axis, nd_plasma_pedestal_electron):
    """Lower limit ensuring central density exceeds the pedestal density."""
    return geq(nd_plasma_electron_on_axis, nd_plasma_pedestal_electron)


def constraint_82(toroidalgap, dx_tf_inboard_out_toroidal):
    """Toroidal consistency of the stellarator build."""
    return geq(toroidalgap, dx_tf_inboard_out_toroidal)


def constraint_83(available_radial_space, required_radial_space):
    """Radial consistency of the stellarator build."""
    return geq(available_radial_space, required_radial_space)


def constraint_84(beta_total_vol_avg, beta_vol_avg_min):
    """Lower limit of plasma beta."""
    return geq(beta_total_vol_avg, beta_vol_avg_min)


def constraint_85(
    i_cp_lifetime, cplife, cplife_input, life_div_fpy, life_blkt_fpy, life_plant
):
    """Equality constraint for the centrepost (CP) lifetime."""
    if i_cp_lifetime == 0:
        bound = cplife_input
    elif i_cp_lifetime == 1:
        bound = life_div_fpy
    elif i_cp_lifetime == 2:
        bound = life_blkt_fpy
    elif i_cp_lifetime == 3:
        bound = life_plant
    else:
        raise ValueError(
            f"constraint_85: i_cp_lifetime={i_cp_lifetime!r} is not in {{0, 1, 2, 3}}"
        )

    return eq(cplife, bound)


def constraint_86(dx_tf_turn_general, t_turn_tf_max):
    """Upper limit on TF winding-pack turn edge length."""
    return leq(dx_tf_turn_general, t_turn_tf_max)


def constraint_87(p_cryo_plant_electric_mw, p_cryo_plant_electric_max_mw):
    """TF coil cryogenic power upper limit."""
    return leq(p_cryo_plant_electric_mw, p_cryo_plant_electric_max_mw)


def constraint_88(str_wp, str_wp_max):
    """TF coil vertical strain upper limit (absolute value)."""
    return leq(abs(str_wp), str_wp_max)


def constraint_89(copperaoh_m2, copperaoh_m2_max):
    """Central Solenoid (OH) coil current / copper area upper limit."""
    return leq(copperaoh_m2, copperaoh_m2_max)


def constraint_90(n_cycle, n_cycle_min, ibkt_life, bkt_life_csf, bktcycles):
    """Lower limit for CS coil stress load cycles."""
    if ibkt_life == 1 and bkt_life_csf == 1:
        n_cycle_min = bktcycles

    return geq(n_cycle, n_cycle_min)


def constraint_91(
    i_plasma_ignited,
    p_hcd_primary_extra_heat_mw,
    powerht_constraint,
    powerscaling_constraint,
):
    """ECRH ignition heating-power lower limit."""
    if PlasmaIgnitionModel(i_plasma_ignited) == PlasmaIgnitionModel.NON_IGNITED:
        value = powerht_constraint + p_hcd_primary_extra_heat_mw
    else:
        value = powerht_constraint

    return geq(value, powerscaling_constraint)


def constraint_92(f_plasma_fuel_deuterium, f_plasma_fuel_tritium, f_plasma_fuel_helium3):
    """D/T/He3 fuel fraction consistency (must sum to 1)."""
    return eq(
        f_plasma_fuel_deuterium + f_plasma_fuel_tritium + f_plasma_fuel_helium3,
        1.0,
    )
