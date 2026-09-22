"""The decision kinds of `large_tokamak_nof`: which boundary input is a belief, a
build decision, an operating variable, numerics, or a constraint limit; the belief
table the tokamak flexibility study draws from, in its two groups; and the two-stage
split -- what is built once, what the operator sets per world, what closes what.

The tokamak counterpart of `kinds.py` (the stellarator's), same shape, and like it
data only: no cottax, no jax. The 412 boundary inputs of the problem graph
(`indat.graph_for(machine)` cut by `mda.SCHEME`, the 26 `icc` condition nodes and the
objective inserted -- `paper_tests/decision_kinds_tokamak.md` is the census, listed
programmatically) take the stellarator table's kind where the spelling is shared
(244 of them: the 1990 cost coefficients, the plasma beliefs, the radial build, the
switches) and the sort below for the 168 the stellarator graph never reads (the PF
coils, the central solenoid, the pulsed-plant timings, the pedestal, the tokamak's
own limits). Two rows differ from the stellarator table: the xenon fraction
(`f_nd_impurity_electron_array[12]`, PROCESS's `ixc 135`) is an operator knob here,
not a belief; and `hfact` stays a belief, though PROCESS's converged design holds it at
its upper bound of 1.2.

The two places the study adds to the graph (`architectures.driven`) are not
boundary inputs of the file's graph and are given their kind there:
`.current_drive.p_hcd_installed_mw` (build) and `.physics.f_beta_norm_max` (belief).
"""

from __future__ import annotations

from functional_process.configurations.kinds import Belief, Kind

KINDS: dict[str, Kind] = {
    # -- belief (180)
    "^stated.tfcoil.eff_tf_cryo": Kind.BELIEF,
    ".costs.csi": Kind.BELIEF,
    ".costs.cland": Kind.BELIEF,
    ".costs.ucrb": Kind.BELIEF,
    ".costs.UCMB": Kind.BELIEF,
    ".costs.UCWS": Kind.BELIEF,
    ".costs.UCTR": Kind.BELIEF,
    ".costs.UCEL": Kind.BELIEF,
    ".costs.UCAD": Kind.BELIEF,
    ".costs.UCCO": Kind.BELIEF,
    ".costs.UCSH": Kind.BELIEF,
    ".costs.UCCR": Kind.BELIEF,
    ".costs.cturbb": Kind.BELIEF,
    ".costs.UCFWA": Kind.BELIEF,
    ".costs.UCFWS": Kind.BELIEF,
    ".costs.UCFWPS": Kind.BELIEF,
    ".costs.fkind": Kind.BELIEF,
    ".costs.ucblbe": Kind.BELIEF,
    ".costs.ucblli2o": Kind.BELIEF,
    ".costs.ucblss": Kind.BELIEF,
    ".costs.ucblvd": Kind.BELIEF,
    ".costs.ucshld": Kind.BELIEF,
    ".costs.ucpens": Kind.BELIEF,
    ".costs.UCGSS": Kind.BELIEF,
    ".costs.ucdiv": Kind.BELIEF,
    ".costs.ucsc": Kind.BELIEF,
    ".costs.uccu": Kind.BELIEF,
    ".costs.cconshtf": Kind.BELIEF,
    ".costs.cconfix": Kind.BELIEF,
    ".costs.ucwindtf": Kind.BELIEF,
    ".costs.uccase": Kind.BELIEF,
    ".costs.UCINT": Kind.BELIEF,
    ".costs.cconshpf": Kind.BELIEF,
    ".tfcoil.dcond[2]": Kind.BELIEF,
    ".tfcoil.dcond[0]": Kind.BELIEF,
    ".costs.ucwindpf": Kind.BELIEF,
    ".costs.ucfnc": Kind.BELIEF,
    ".costs.uccryo": Kind.BELIEF,
    ".costs.ucech": Kind.BELIEF,
    ".costs.uclh": Kind.BELIEF,
    ".costs.ucich": Kind.BELIEF,
    ".costs.ucnbi": Kind.BELIEF,
    ".costs.fcdfuel": Kind.BELIEF,
    ".costs.UCCPMP": Kind.BELIEF,
    ".costs.UCTPMP": Kind.BELIEF,
    ".costs.UCBPMP": Kind.BELIEF,
    ".costs.UCDUCT": Kind.BELIEF,
    ".costs.UCVALV": Kind.BELIEF,
    ".costs.UCVDSH": Kind.BELIEF,
    ".costs.UCVIAC": Kind.BELIEF,
    ".costs.uctfps": Kind.BELIEF,
    ".costs.uctfbr": Kind.BELIEF,
    ".costs.uctfsw": Kind.BELIEF,
    ".costs.UCTFDR": Kind.BELIEF,
    ".costs.UCTFGR": Kind.BELIEF,
    ".costs.UCTFIC": Kind.BELIEF,
    ".costs.uctfbus": Kind.BELIEF,
    ".costs.ucbus": Kind.BELIEF,
    ".costs.ucpfps": Kind.BELIEF,
    ".costs.ucpfic": Kind.BELIEF,
    ".costs.ucpfb": Kind.BELIEF,
    ".costs.ucpfbs": Kind.BELIEF,
    ".costs.ucpfbk": Kind.BELIEF,
    ".costs.ucpfdr1": Kind.BELIEF,
    ".costs.ucpfcb": Kind.BELIEF,
    ".costs.uchts": Kind.BELIEF,
    ".costs.UCPHX": Kind.BELIEF,
    ".costs.UCAHTS": Kind.BELIEF,
    ".heat_transport.vachtmw": Kind.BELIEF,
    ".heat_transport.p_tritium_plant_electric_mw": Kind.BELIEF,
    ".costs.uccry": Kind.BELIEF,
    ".costs.ucf1": Kind.BELIEF,
    ".costs.UCFPR": Kind.BELIEF,
    ".costs.UCDTC": Kind.BELIEF,
    ".costs.UCNBV": Kind.BELIEF,
    ".costs.uciac": Kind.BELIEF,
    ".costs.ucme": Kind.BELIEF,
    ".costs.ucturb": Kind.BELIEF,
    ".costs.UCSWYD": Kind.BELIEF,
    ".costs.UCPP": Kind.BELIEF,
    ".costs.UCAP": Kind.BELIEF,
    ".costs.UCLV": Kind.BELIEF,
    ".costs.UCDGEN": Kind.BELIEF,
    ".costs.UCAF": Kind.BELIEF,
    ".costs.ucmisc": Kind.BELIEF,
    ".costs.uchrs": Kind.BELIEF,
    ".costs.cfind": Kind.BELIEF,
    ".costs.cowner": Kind.BELIEF,
    ".costs.fcontng": Kind.BELIEF,
    ".costs.fcap0": Kind.BELIEF,
    ".costs.fcr0": Kind.BELIEF,
    ".costs.discount_rate": Kind.BELIEF,
    ".costs.fcap0cp": Kind.BELIEF,
    ".costs.ucoam": Kind.BELIEF,
    ".costs.ucfuel": Kind.BELIEF,
    ".costs.uche3": Kind.BELIEF,
    ".costs.ucwst": Kind.BELIEF,
    ".costs.decomf": Kind.BELIEF,
    ".costs.dintrt": Kind.BELIEF,
    ".costs.dtlife": Kind.BELIEF,
    ".physics.f_p_alpha_plasma_deposited": Kind.BELIEF,
    ".physics.plasma_res_factor": Kind.BELIEF,
    ".physics.csawth": Kind.BELIEF,
    ".physics.ejima_coeff": Kind.BELIEF,
    ".physics.ind_plasma_internal_norm": Kind.BELIEF,
    ".physics.q0": Kind.BELIEF,
    ".current_drive.cboot": Kind.BELIEF,
    ".current_drive.eta_cd_norm_ecrh": Kind.BELIEF,
    ".current_drive.eta_ecrh_injector_wall_plug": Kind.BELIEF,
    ".tfcoil.den_tf_wp_turn_insulation": Kind.BELIEF,
    ".tfcoil.den_tf_coil_case": Kind.BELIEF,
    ".fwbs.den_steel": Kind.BELIEF,
    ".tfcoil.eyoung_steel": Kind.BELIEF,
    ".tfcoil.poisson_steel": Kind.BELIEF,
    ".tfcoil.poisson_cond_axial": Kind.BELIEF,
    ".tfcoil.poisson_cond_trans": Kind.BELIEF,
    ".tfcoil.poisson_ins": Kind.BELIEF,
    ".tfcoil.eyoung_copper": Kind.BELIEF,
    ".tfcoil.poisson_copper": Kind.BELIEF,
    ".tfcoil.str_cs_con_res": Kind.BELIEF,
    ".cs_fatigue.paris_coefficient": Kind.BELIEF,
    ".cs_fatigue.paris_power_law": Kind.BELIEF,
    ".cs_fatigue.walker_coefficient": Kind.BELIEF,
    ".cs_fatigue.fracture_toughness": Kind.BELIEF,
    ".physics.rad_fraction_sol": Kind.BELIEF,
    ".physics.ffwal": Kind.BELIEF,
    ".constraints.f_fw_rad_max": Kind.BELIEF,
    ".fwbs.fvolsi": Kind.BELIEF,
    ".fwbs.fvolso": Kind.BELIEF,
    ".fwbs.fvoldw": Kind.BELIEF,
    ".divertor.den_div_structure": Kind.BELIEF,
    ".ccfe_hcpb.fw_armour_u_nuc_heating": Kind.BELIEF,
    ".fwbs.f_p_blkt_multiplication": Kind.BELIEF,
    ".primary_pumping.gamma_he": Kind.BELIEF,
    ".fwbs.etaiso": Kind.BELIEF,
    ".heat_transport.f_p_shld_coolant_pump_total_heat": Kind.BELIEF,
    ".heat_transport.f_p_div_coolant_pump_total_heat": Kind.BELIEF,
    ".physics.f_nd_plasma_pedestal_greenwald": Kind.BELIEF,
    ".physics.f_nd_plasma_separatrix_greenwald": Kind.BELIEF,
    ".physics.radius_plasma_pedestal_temp_norm": Kind.BELIEF,
    ".physics.temp_plasma_pedestal_kev": Kind.BELIEF,
    ".physics.temp_plasma_separatrix_kev": Kind.BELIEF,
    ".physics.alphat": Kind.BELIEF,
    ".physics.tbeta": Kind.BELIEF,
    ".physics.radius_plasma_pedestal_density_norm": Kind.BELIEF,
    ".physics.alphan": Kind.BELIEF,
    ".physics.f_temp_plasma_ion_electron": Kind.BELIEF,
    ".physics.hfact": Kind.BELIEF,
    ".physics.f_sync_reflect": Kind.BELIEF,
    ".impurity_radiation.f_nd_impurity_electron_array[2]": Kind.BELIEF,
    ".impurity_radiation.f_nd_impurity_electron_array[3]": Kind.BELIEF,
    ".impurity_radiation.f_nd_impurity_electron_array[4]": Kind.BELIEF,
    ".impurity_radiation.f_nd_impurity_electron_array[5]": Kind.BELIEF,
    ".impurity_radiation.f_nd_impurity_electron_array[6]": Kind.BELIEF,
    ".impurity_radiation.f_nd_impurity_electron_array[7]": Kind.BELIEF,
    ".impurity_radiation.f_nd_impurity_electron_array[8]": Kind.BELIEF,
    ".impurity_radiation.f_nd_impurity_electron_array[9]": Kind.BELIEF,
    ".impurity_radiation.f_nd_impurity_electron_array[10]": Kind.BELIEF,
    ".impurity_radiation.f_nd_impurity_electron_array[11]": Kind.BELIEF,
    ".impurity_radiation.f_nd_impurity_electron_array[13]": Kind.BELIEF,
    ".impurity_radiation.temp_impurity_keV_array": Kind.BELIEF,
    ".impurity_radiation.pden_impurity_lz_nd_temp_array": Kind.BELIEF,
    ".physics.tauratio": Kind.BELIEF,
    ".physics.f_nd_protium_electrons": Kind.BELIEF,
    ".impurity_radiation.impurity_arr_zav": Kind.BELIEF,
    ".impurity_radiation.m_impurity_amu_array": Kind.BELIEF,
    ".pf_coil.rhopfbus": Kind.BELIEF,
    ".pf_power.f_p_pf_energy_store_loss": Kind.BELIEF,
    ".pf_power.f_p_pf_psu_loss": Kind.BELIEF,
    ".pf_coil.etapsu": Kind.BELIEF,
    ".tfcoil.rho_tf_bus": Kind.BELIEF,
    ".heat_transport.etatf": Kind.BELIEF,
    ".fwbs.eta_coolant_pump_electric": Kind.BELIEF,
    ".fwbs.qnuc": Kind.BELIEF,
    ".vacuum.outgrat_fw": Kind.BELIEF,
    ".heat_transport.p_plant_electric_base": Kind.BELIEF,
    ".heat_transport.pflux_plant_floor_electric": Kind.BELIEF,
    ".heat_transport.eta_turbine": Kind.BELIEF,
    ".costs.abktflnc": Kind.BELIEF,
    ".costs.adivflnc": Kind.BELIEF,
    # -- build (132)
    ".costs.life_plant": Kind.BUILD,
    ".buildings.triv": Kind.BUILD,
    ".tfcoil.i_tf_sc_mat": Kind.BUILD,
    ".tfcoil.n_tf_coils": Kind.BUILD,
    ".pf_coil.fcupfsu": Kind.BUILD,
    ".pf_coil.f_a_pf_coil_void": Kind.BUILD,
    ".pf_coil.j_pf_coil_wp_peak": Kind.BUILD,
    ".pf_coil.f_a_cs_void": Kind.BUILD,
    ".pf_coil.fcuohsu": Kind.BUILD,
    ".tfcoil.i_tf_sup": Kind.BUILD,
    ".tfcoil.c_tf_turn": Kind.BUILD,
    ".fwbs.i_blkt_coolant_type": Kind.BUILD,
    ".physics.rmajor": Kind.BUILD,
    ".physics.aspect": Kind.BUILD,
    ".physics.kappa": Kind.BUILD,
    ".physics.triang": Kind.BUILD,
    ".physics.b_plasma_toroidal_on_axis": Kind.BUILD,
    ".build.plsepi": Kind.BUILD,
    ".build.plsepo": Kind.BUILD,
    ".build.plleni": Kind.BUILD,
    ".build.plleno": Kind.BUILD,
    ".divertor.betai": Kind.BUILD,
    ".divertor.betao": Kind.BUILD,
    ".divertor.dz_divertor": Kind.BUILD,
    ".build.dz_shld_lower": Kind.BUILD,
    ".build.dz_vv_lower": Kind.BUILD,
    ".build.dz_shld_vv_gap": Kind.BUILD,
    ".build.dz_shld_thermal": Kind.BUILD,
    ".build.dr_tf_shld_gap": Kind.BUILD,
    ".build.dz_vv_upper": Kind.BUILD,
    ".build.dz_shld_upper": Kind.BUILD,
    ".build.dr_shld_blkt_gap": Kind.BUILD,
    ".build.dz_fw_plasma_gap": Kind.BUILD,
    ".build.dr_blkt_inboard": Kind.BUILD,
    ".build.dr_blkt_outboard": Kind.BUILD,
    ".tfcoil.dr_tf_wp_with_insulation": Kind.BUILD,
    ".tfcoil.dr_tf_nose_case": Kind.BUILD,
    ".build.dr_bore": Kind.BUILD,
    ".build.dr_cs": Kind.BUILD,
    ".build.fseppc": Kind.BUILD,
    ".build.fcspc": Kind.BUILD,
    ".build.sigallpc": Kind.BUILD,
    ".build.dr_cs_tf_gap": Kind.BUILD,
    ".build.dr_shld_thermal_inboard": Kind.BUILD,
    ".build.dr_shld_vv_gap_inboard": Kind.BUILD,
    ".build.dr_vv_inboard": Kind.BUILD,
    ".build.dr_shld_inboard": Kind.BUILD,
    ".build.dr_fw_plasma_gap_inboard": Kind.BUILD,
    ".build.dr_fw_plasma_gap_outboard": Kind.BUILD,
    ".build.dr_shld_outboard": Kind.BUILD,
    ".tfcoil.dx_tf_wp_insulation": Kind.BUILD,
    ".tfcoil.dx_tf_wp_insertion_gap": Kind.BUILD,
    ".build.dr_vv_outboard": Kind.BUILD,
    ".build.gapomin": Kind.BUILD,
    ".build.dr_shld_thermal_outboard": Kind.BUILD,
    ".tfcoil.ripple_b_tf_plasma_edge_max": Kind.BUILD,
    ".tfcoil.dx_tf_side_case_min": Kind.BUILD,
    ".tfcoil.f_a_tf_turn_cable_space_extra_void": Kind.BUILD,
    ".tfcoil.f_a_tf_turn_cable_copper": Kind.BUILD,
    ".tfcoil.f_vforce_inboard": Kind.BUILD,
    ".tfcoil.dx_tf_turn_insulation": Kind.BUILD,
    ".tfcoil.dia_tf_turn_coolant_channel": Kind.BUILD,
    ".tfcoil.dx_tf_turn_steel": Kind.BUILD,
    ".tfcoil.theta1_coil": Kind.BUILD,
    ".tfcoil.theta1_vv": Kind.BUILD,
    ".tfcoil.t_tf_superconductor_quench": Kind.BUILD,
    ".build.dr_vv_shells": Kind.BUILD,
    ".tfcoil.rrr_tf_cu": Kind.BUILD,
    ".tfcoil.t_tf_quench_detection": Kind.BUILD,
    ".constraints.flu_tf_neutron_fast_max": Kind.BUILD,
    ".tfcoil.layer_ins": Kind.BUILD,
    ".pf_coil.dr_pf_tf_outboard_out_offset": Kind.BUILD,
    ".pf_coil.rpf2": Kind.BUILD,
    ".pf_coil.zref": Kind.BUILD,
    ".pf_coil.j_cs_flat_top_end": Kind.BUILD,
    ".pf_coil.f_j_cs_start_pulse_end_flat_top": Kind.BUILD,
    ".pf_coil.c_pf_coil_turn_peak_input": Kind.BUILD,
    ".pf_coil.pf_current_safety_factor": Kind.BUILD,
    ".pf_coil.sigpfcf": Kind.BUILD,
    ".pf_coil.sigpfcalw": Kind.BUILD,
    ".pf_coil.f_a_cs_turn_steel": Kind.BUILD,
    ".pf_coil.f_z_cs_tf_internal": Kind.BUILD,
    ".pf_coil.f_dr_dz_cs_turn": Kind.BUILD,
    ".pf_coil.radius_cs_turn_corners": Kind.BUILD,
    ".cs_fatigue.residual_sig_hoop": Kind.BUILD,
    ".cs_fatigue.t_crack_vertical": Kind.BUILD,
    ".cs_fatigue.sf_vertical_crack": Kind.BUILD,
    ".cs_fatigue.sf_radial_crack": Kind.BUILD,
    ".cs_fatigue.sf_fast_fracture": Kind.BUILD,
    ".divertor.n_divertors": Kind.BUILD,
    ".divertor.f_div_flux_expansion": Kind.BUILD,
    ".divertor.deg_div_field_plate": Kind.BUILD,
    ".fwbs.f_a_fw_outboard_hcd": Kind.BUILD,
    ".fwbs.radius_fw_channel": Kind.BUILD,
    ".fwbs.dr_fw_wall": Kind.BUILD,
    ".fwbs.dx_fw_module": Kind.BUILD,
    ".divertor.fdiva": Kind.BUILD,
    ".divertor.f_vol_div_coolant": Kind.BUILD,
    ".divertor.dx_div_plate": Kind.BUILD,
    ".fwbs.f_a_blkt_cooling_channels": Kind.BUILD,
    ".fwbs.vfshld": Kind.BUILD,
    ".fwbs.fw_armour_thickness": Kind.BUILD,
    ".fwbs.breeder_f": Kind.BUILD,
    ".fwbs.breeder_multiplier": Kind.BUILD,
    ".fwbs.vfcblkt": Kind.BUILD,
    ".fwbs.vfpblkt": Kind.BUILD,
    ".fwbs.dr_pf_cryostat": Kind.BUILD,
    ".build.f_z_cryostat": Kind.BUILD,
    ".build.dr_cryostat": Kind.BUILD,
    ".buildings.rxcl": Kind.BUILD,
    ".buildings.trcl": Kind.BUILD,
    ".buildings.row": Kind.BUILD,
    ".buildings.wgt": Kind.BUILD,
    ".buildings.shmf": Kind.BUILD,
    ".buildings.clh2": Kind.BUILD,
    ".buildings.stcl": Kind.BUILD,
    ".buildings.rbvfac": Kind.BUILD,
    ".buildings.rbwt": Kind.BUILD,
    ".buildings.rbrt": Kind.BUILD,
    ".buildings.fndt": Kind.BUILD,
    ".buildings.hcwt": Kind.BUILD,
    ".buildings.hccl": Kind.BUILD,
    ".buildings.wgt2": Kind.BUILD,
    ".buildings.mbvfac": Kind.BUILD,
    ".buildings.wsvfac": Kind.BUILD,
    ".buildings.pfbldgm3": Kind.BUILD,
    ".buildings.esbldgm3": Kind.BUILD,
    ".buildings.pibv": Kind.BUILD,
    ".buildings.conv": Kind.BUILD,
    ".buildings.admv": Kind.BUILD,
    ".buildings.shov": Kind.BUILD,
    ".constraints.f_j_tf_wp_critical_max": Kind.BUILD,
    # -- operating (24)
    ".costs.f_t_plant_available": Kind.OPERATING,
    ".tfcoil.temp_tf_cryo": Kind.OPERATING,
    ".physics.f_plasma_fuel_tritium": Kind.OPERATING,
    ".physics.f_plasma_fuel_helium3": Kind.OPERATING,
    ".physics.q95": Kind.OPERATING,
    ".physics.nd_plasma_electrons_vol_avg": Kind.OPERATING,
    ".physics.temp_plasma_electron_vol_avg_kev": Kind.OPERATING,
    ".times.t_plant_pulse_fusion_ramp": Kind.OPERATING,
    ".physics.beta_total_vol_avg": Kind.OPERATING,
    ".physics.f_c_plasma_non_inductive": Kind.OPERATING,
    ".current_drive.p_hcd_primary_extra_heat_mw": Kind.OPERATING,
    ".tfcoil.tftmp": Kind.OPERATING,
    ".pf_coil.temp_cs_superconductor_operating": Kind.OPERATING,
    ".times.t_plant_pulse_coil_precharge": Kind.OPERATING,
    ".times.t_plant_pulse_dwell": Kind.OPERATING,
    ".primary_pumping.p_he": Kind.OPERATING,
    ".primary_pumping.dp_he": Kind.OPERATING,
    ".primary_pumping.t_in_bb": Kind.OPERATING,
    ".primary_pumping.t_out_bb": Kind.OPERATING,
    ".physics.f_plasma_fuel_deuterium": Kind.OPERATING,
    ".impurity_radiation.f_nd_impurity_electron_array[12]": Kind.OPERATING,
    ".physics.f_nd_alpha_thermal_electron": Kind.OPERATING,
    ".vacuum.pres_vv_chamber_base": Kind.OPERATING,
    ".vacuum.pres_div_chamber_burn": Kind.OPERATING,
    # -- numerics (46)
    "^stated.tfcoil.eyoung_ins": Kind.NUMERICS,
    "^stated.tfcoil.eyoung_cond_axial": Kind.NUMERICS,
    "^stated.tfcoil.eyoung_cond_trans": Kind.NUMERICS,
    "^stated.pf_coil.rho_pf_coil": Kind.NUMERICS,
    "^stated.physics.f_nd_beam_electron": Kind.NUMERICS,
    ".physics.itart": Kind.NUMERICS,
    ".costs.lsa": Kind.NUMERICS,
    ".costs.ireactor": Kind.NUMERICS,
    ".costs.ifueltyp": Kind.NUMERICS,
    ".ife.ife": Kind.NUMERICS,
    ".pf_coil.i_pf_superconductor": Kind.NUMERICS,
    ".pf_coil.i_cs_superconductor": Kind.NUMERICS,
    ".current_drive.i_hcd_primary": Kind.NUMERICS,
    ".current_drive.p_hcd_lowhyb_injected_total_mw": Kind.NUMERICS,
    ".current_drive.p_beam_injected_mw": Kind.NUMERICS,
    ".vacuum.i_vacuum_pump_type": Kind.NUMERICS,
    ".physics.f_vol_plasma": Kind.NUMERICS,
    ".physics.alphaj": Kind.NUMERICS,
    ".physics.beta_beam": Kind.NUMERICS,
    ".current_drive.f_c_plasma_bootstrap_max": Kind.NUMERICS,
    "^stated.current_drive.f_c_plasma_diamagnetic": Kind.NUMERICS,
    "^stated.current_drive.f_c_plasma_pfirsch_schluter": Kind.NUMERICS,
    "^stated.current_drive.eta_cd_hcd_secondary": Kind.NUMERICS,
    "^stated.current_drive.p_hcd_secondary_extra_heat_mw": Kind.NUMERICS,
    "^stated.heat_transport.p_hcd_secondary_electric_mw": Kind.NUMERICS,
    ".current_drive.p_hcd_secondary_injected_mw": Kind.NUMERICS,
    ".current_drive.p_beam_orbit_loss_mw": Kind.NUMERICS,
    ".pf_coil.alfapf": Kind.NUMERICS,
    "^stated.fwbs.pnuc_cp_tf": Kind.NUMERICS,
    "^stated.fwbs.p_cp_shield_nuclear_heat_mw": Kind.NUMERICS,
    "^stated.fwbs.pnuc_cp": Kind.NUMERICS,
    "^stated.fwbs.neut_flux_cp": Kind.NUMERICS,
    ".primary_pumping.f_p_fw_blkt_pump": Kind.NUMERICS,
    ".physics.p_beam_alpha_mw": Kind.NUMERICS,
    ".impurity_radiation.radius_plasma_core_norm": Kind.NUMERICS,
    ".impurity_radiation.f_p_plasma_core_rad_reduction": Kind.NUMERICS,
    ".physics.burnup_in": Kind.NUMERICS,
    ".current_drive.f_beam_tritium": Kind.NUMERICS,
    ".heat_transport.p_blkt_breeder_pump_mw": Kind.NUMERICS,
    ".fwbs.f_nuc_pow_bz_liq": Kind.NUMERICS,
    ".current_drive.p_beam_shine_through_mw": Kind.NUMERICS,
    ".heat_transport.i_shld_primary_heat": Kind.NUMERICS,
    ".fwbs.outlet_temp_liq": Kind.NUMERICS,
    ".tfcoil.temp_cp_coolant_inlet": Kind.NUMERICS,
    ".vacuum.i_vac_pump_dwell": Kind.NUMERICS,
    ".constraints.q95_fixed": Kind.NUMERICS,
    # -- derived (11)
    ".costs.cplife": Kind.DERIVED,
    ".fwbs.m_blkt_vanadium": Kind.DERIVED,
    ".tfcoil.tfcmw": Kind.DERIVED,
    ".tfcoil.m_tf_bus": Kind.DERIVED,
    ".heat_transport.p_fw_div_heat_deposited_mw": Kind.DERIVED,
    ".tfcoil.res_tf_leg": Kind.DERIVED,
    ".heat_transport.p_fw_coolant_pump_mw": Kind.DERIVED,
    ".heat_transport.p_blkt_coolant_pump_mw": Kind.DERIVED,
    ".fwbs.p_fw_hcd_nuclear_heat_mw": Kind.DERIVED,
    ".fwbs.life_fw_fpy": Kind.DERIVED,
    ".tfcoil.sig_tf_cs_bucked": Kind.DERIVED,
    # -- limit (19)
    ".current_drive.p_hcd_injected_max": Kind.LIMIT,
    ".constraints.f_h_mode_margin": Kind.LIMIT,
    ".constraints.p_plant_electric_net_required_mw": Kind.LIMIT,
    ".constraints.b_tf_inboard_max": Kind.LIMIT,
    ".constraints.fjohc": Kind.LIMIT,
    ".constraints.fjohc0": Kind.LIMIT,
    ".tfcoil.v_tf_coil_dump_quench_max_kv": Kind.LIMIT,
    ".tfcoil.temp_tf_superconductor_margin_min": Kind.LIMIT,
    ".tfcoil.temp_cs_superconductor_margin_min": Kind.LIMIT,
    ".constraints.f_t_alpha_energy_confinement_min": Kind.LIMIT,
    ".tfcoil.max_vv_stress": Kind.LIMIT,
    ".pf_coil.stress_cs_steel_max": Kind.LIMIT,
    ".constraints.p_div_bt_q_aspect_rmajor_max_mw": Kind.LIMIT,
    ".tfcoil.sig_tf_case_max": Kind.LIMIT,
    ".tfcoil.sig_tf_wp_max": Kind.LIMIT,
    ".constraints.f_nd_plasma_electron_limit_max": Kind.LIMIT,
    ".constraints.pflux_fw_neutron_max_mw": Kind.LIMIT,
    ".constraints.p_fusion_total_max_mw": Kind.LIMIT,
    ".constraints.t_burn_min": Kind.LIMIT,
}
"""Every boundary input of `large_tokamak_nof`'s problem graph -> its kind: 180
belief (73 of them 1990 cost-model coefficients, 12 PF-coil cost coefficients), 132
build, 24 operating, 46 numerics, 11 derived-but-read-from-outside (zero here), 19
constraint limits. Decisions worth naming: `q95` is operating (the plasma current is
set on the day) but held at the design value in the study, since the CS and PF system
are sized to it; `f_c_plasma_non_inductive` likewise; `beta_total_vol_avg` (`ixc 5`)
is the copy of a computed quantity PROCESS iterates on, an operating state; the
pedestal (`temp_plasma_pedestal_kev`, `f_nd_plasma_pedestal_greenwald`, the profile
radii, `tbeta`) is a profile belief, not a knob; `ind_plasma_internal_norm` is a belief
(held: the beta limit's threshold is sampled through `f_beta_norm_max`, not through
`l_i`, which also enters the volt-seconds); `j_cs_flat_top_end` (`ixc 37`) is the CS's
design current density, build; the fatigue model's material coefficients are beliefs
and its safety factors build."""


DESIGN_KINDS: dict[int, Kind] = {
    2: Kind.BUILD,  # b_plasma_toroidal_on_axis: the coil current and the magnet
    3: Kind.BUILD,  # rmajor: the machine size (the figure of merit, at its lower bound 8 m)
    4: Kind.OPERATING,  # temp_plasma_electron_vol_avg_kev: heating on the day -- a knob
    5: Kind.OPERATING,  # beta_total_vol_avg: the copy of a computed quantity, closes c1
    6: Kind.OPERATING,  # nd_plasma_electrons_vol_avg: fuelling -- a knob, under Greenwald (c5)
    10: Kind.BELIEF,  # hfact: a confinement multiplier is not chosen; PROCESS holds it at 1.2, the bound
    16: Kind.BUILD,  # dr_cs: the central solenoid's radial thickness
    18: Kind.OPERATING,  # q95: the plasma current; held at the design value (see the note above)
    29: Kind.BUILD,  # dr_bore: the machine bore
    37: Kind.BUILD,  # j_cs_flat_top_end: the CS design current density at the end of flat-top
    44: Kind.OPERATING,  # f_c_plasma_non_inductive: the current-drive share; held, it fixes the CD power
    56: Kind.BUILD,  # t_tf_superconductor_quench: dump / protection design
    57: Kind.BUILD,  # dr_tf_nose_case: TF coil nose case thickness
    58: Kind.BUILD,  # dx_tf_turn_steel: TF conduit thickness
    59: Kind.BUILD,  # f_a_tf_turn_cable_copper: conductor design
    60: Kind.BUILD,  # c_tf_turn: TF current per turn
    109: Kind.OPERATING,  # f_nd_alpha_thermal_electron: the helium fraction, closes c62 per world
    122: Kind.BUILD,  # f_a_cs_turn_steel: CS steel fraction
    135: Kind.OPERATING,  # f_nd_impurity_electron_array[12]: the xenon seeding -- a knob
    140: Kind.BUILD,  # dr_tf_wp_with_insulation: the winding-pack radial thickness
}
"""The twenty `ixc` iteration variables of the reference run, in the file's order ->
kind: 12 build, 7 operating, 1 belief. Of the seven operating: three are the
operator's knobs (`KNOBS`), two close a condition per world (`beta_total_vol_avg` c1,
`f_nd_alpha_thermal_electron` c62), two are held at the design value (`q95`,
`f_c_plasma_non_inductive`)."""

DESIGN_PLACES: dict[int, str] = {
    2: ".physics.b_plasma_toroidal_on_axis",
    3: ".physics.rmajor",
    4: ".physics.temp_plasma_electron_vol_avg_kev",
    5: ".physics.beta_total_vol_avg",
    6: ".physics.nd_plasma_electrons_vol_avg",
    10: ".physics.hfact",
    16: ".build.dr_cs",
    18: ".physics.q95",
    29: ".build.dr_bore",
    37: ".pf_coil.j_cs_flat_top_end",
    44: ".physics.f_c_plasma_non_inductive",
    56: ".tfcoil.t_tf_superconductor_quench",
    57: ".tfcoil.dr_tf_nose_case",
    58: ".tfcoil.dx_tf_turn_steel",
    59: ".tfcoil.f_a_tf_turn_cable_copper",
    60: ".tfcoil.c_tf_turn",
    109: ".physics.f_nd_alpha_thermal_electron",
    122: ".pf_coil.f_a_cs_turn_steel",
    135: ".impurity_radiation.f_nd_impurity_electron_array[12]",
    140: ".tfcoil.dr_tf_wp_with_insulation",
}
"""`ixc` id -> the place it owns, the same rows."""

BUILD_IXC: tuple[int, ...] = (2, 3, 16, 29, 37, 56, 57, 58, 59, 60, 122, 140)
"""The twelve build `ixc`: fixed at PROCESS's converged design in the study."""

HELD_OPERATING: tuple[int, ...] = (18, 44)
"""`q95` and `f_c_plasma_non_inductive`: operating by kind, held at the design value
-- the plasma current and the current-drive share define the pulse scenario the CS
and PF coils are sized for."""

LIMIT_OF: dict[int, str] = {
    30: ".current_drive.p_hcd_injected_max",  # -> `.current_drive.p_hcd_installed_mw` after the rewire
    15: ".constraints.f_h_mode_margin",
    16: ".constraints.p_plant_electric_net_required_mw",
    24: ".physics.f_beta_norm_max",  # the factor on Wesson's 4 l_i; the threshold itself is computed
    25: ".constraints.b_tf_inboard_max",
    26: ".constraints.fjohc",
    27: ".constraints.fjohc0",
    33: ".constraints.f_j_tf_wp_critical_max",
    34: ".tfcoil.v_tf_coil_dump_quench_max_kv",
    36: ".tfcoil.temp_tf_superconductor_margin_min",
    60: ".tfcoil.temp_cs_superconductor_margin_min",
    62: ".constraints.f_t_alpha_energy_confinement_min",
    65: ".tfcoil.max_vv_stress",
    72: ".pf_coil.stress_cs_steel_max",
    68: ".constraints.p_div_bt_q_aspect_rmajor_max_mw",
    31: ".tfcoil.sig_tf_case_max",
    32: ".tfcoil.sig_tf_wp_max",
    5: ".constraints.f_nd_plasma_electron_limit_max",
    8: ".constraints.pflux_fw_neutron_max_mw",
    9: ".constraints.p_fusion_total_max_mw",
    13: ".constraints.t_burn_min",
}
"""`icc` id -> the limit input it compares against, for 21 of the 23 inequalities.
The other two compare two outputs: `c35` (`j_tf_wp` against the computed quench
limit) and `c81` (the on-axis density against the pedestal's). `c24`'s is listed as
the factor the study adds: on the tokamak branch `beta_vol_avg_max` is computed from
Wesson's `4 l_i` and the file's `beta_norm_max = 3.0` is dead."""

LIMIT_ON: dict[int, Kind] = {
    30: Kind.BUILD,  # installed power: what was bought
    15: Kind.NUMERICS,  # a trivial margin (1.0) on the L-H threshold
    16: Kind.BUILD,  # plant spec: the required net electric power (400 MW)
    24: Kind.BELIEF,  # the beta limit's coefficient
    25: Kind.BUILD,  # magnet spec: the peak field
    26: Kind.BUILD,  # margin: CS current density at end of flat-top
    27: Kind.BUILD,  # margin: CS current density at pulse start
    33: Kind.BUILD,  # margin: TF operating / critical current (1.0 here)
    34: Kind.BUILD,  # protection spec: dump voltage
    36: Kind.BUILD,  # spec: TF temperature margin
    60: Kind.BUILD,  # spec: CS temperature margin
    62: Kind.BELIEF,  # He exhaust: the minimum tau_He* / tau_E
    65: Kind.BUILD,  # allowable: vacuum-vessel stress at quench
    72: Kind.BUILD,  # allowable: CS steel stress
    68: Kind.BUILD,  # divertor spec: the PsepB/qAR guideline
    31: Kind.BUILD,  # allowable: TF case stress
    32: Kind.BUILD,  # allowable: TF conduit stress
    5: Kind.BELIEF,  # the allowed Greenwald fraction (1.2): a physics belief, held
    8: Kind.BUILD,  # materials spec: neutron wall load
    9: Kind.BUILD,  # plant spec: the fusion power cap
    13: Kind.BUILD,  # plant spec: the minimum burn time
}
"""`icc` id -> the kind its limit is a limit *on*: 17 build specifications and
margins, 3 physics beliefs (two sampled, one held), 1 trivial bound."""


# ---------------------------------------------------------------- the belief table

PLASMA = "plasma"
"""The parameters of the 0-D closure that stands in for a transport solver."""
LIMITS = "limits"
"""Constraint thresholds that are physics beliefs."""

BELIEFS: tuple[Belief, ...] = (
    Belief(
        ".physics.hfact",
        "lognormal",
        0.10,
        0.0,
        "confinement multiplier (H98y2); PROCESS's closure variable of the power balance, at its bound 1.2 in the converged design",
    ),
    Belief(".physics.alphan", "relative", 0.20, 0.0, "density profile exponent"),
    Belief(".physics.alphat", "relative", 0.20, 0.0, "temperature profile exponent"),
    Belief(".physics.f_temp_plasma_ion_electron", "uniform", 0.85, 1.0, "$T_i / T_e$"),
    Belief(
        ".physics.f_p_alpha_plasma_deposited",
        "uniform",
        0.90,
        0.99,
        "alpha power deposited in the plasma",
    ),
    Belief(
        ".impurity_radiation.f_nd_impurity_electron_array[13]",
        "relative",
        0.20,
        0.0,
        "tungsten fraction (index 13, the seeded non-xenon impurity, 5e-6); xenon (index 12) is the operator's",
    ),
    Belief(
        ".physics.f_beta_norm_max",
        "relative",
        0.20,
        0.0,
        "the beta limit's threshold: a factor on Wesson's beta_N,max = 4 l_i (`architectures.driven.beta_limit_factor`)",
    ),
    Belief(
        ".constraints.f_t_alpha_energy_confinement_min",
        "uniform",
        3.0,
        6.0,
        "rho* = tau_He* / tau_E, the helium exhaust requirement (5 in the file)",
    ),
    Belief("dummy", "uniform", 0.0, 1.0, "read by nothing: the estimators' noise floor"),
)
"""The belief table: the plasma group (six rows) and the limits group (two rows),
the same rule as the stellarator's (`kinds.BELIEFS`): parameters of the 0-D closure
that stands in for a transport solver, and constraint thresholds that are physics
beliefs. Everything else is at the file's value."""

GROUP: dict[str, str] = {
    ".physics.hfact": PLASMA,
    ".physics.alphan": PLASMA,
    ".physics.alphat": PLASMA,
    ".physics.f_temp_plasma_ion_electron": PLASMA,
    ".physics.f_p_alpha_plasma_deposited": PLASMA,
    ".impurity_radiation.f_nd_impurity_electron_array[13]": PLASMA,
    ".physics.f_beta_norm_max": LIMITS,
    ".constraints.f_t_alpha_energy_confinement_min": LIMITS,
}
"""Belief path -> its group."""


# ---------------------------------------------------------------- the two-stage split

TE = ".physics.temp_plasma_electron_vol_avg_kev"
NE = ".physics.nd_plasma_electrons_vol_avg"
XE = ".impurity_radiation.f_nd_impurity_electron_array[12]"
KNOBS: tuple[str, ...] = (NE, TE, XE)
"""The operator's recourse: the density (`ixc 6`, under Greenwald `c5`), the electron
temperature (`ixc 4`) and the xenon seeding (`ixc 135`, the divertor / radiation
knob)."""

HEATING = ".current_drive.p_hcd_primary_extra_heat_mw"
INSTALLED = ".current_drive.p_hcd_installed_mw"

PAIRINGS: dict[str, dict[str, str]] = {
    "tokamak": {
        "^cond.constraints.c2": HEATING,
        "^cond.constraints.c62": ".physics.f_nd_alpha_thermal_electron",
        "^cond.constraints.c1": ".physics.beta_total_vol_avg",
    },
}
"""What the MDA closes per world, and by what: the power balance by the heating
power (`hfact` is a belief now; at the operator's (n, T) the balance says what the
auxiliary power must be), the helium particle balance -- `c62`, an inequality held
with equality: n_He = rho* tau_E S_alpha -- by the thermal alpha fraction, and the beta
consistency by the beta (PROCESS's `ixc 5` is the copy of a computed quantity)."""

RECOURSE_BOUNDED: tuple[str, ...] = (HEATING,)
"""Closing variables whose lower bound is a per-world inequality: the heating power
cannot be negative -- a world whose balance asks for negative heating at the
operator's point has no operating point there."""

BUILD_LEAVES: tuple[str, ...] = ()
"""The stellarator's build table held four sampled rows (manufacturing scatter and
the designer's current margin) at their nominal; the tokamak table samples none of
them, so nothing is held here."""

ECONOMIC: tuple[str, ...] = ()
"""No economic row is sampled."""
