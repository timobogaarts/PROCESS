"""The solve environment, built from the input file alone -- no `DataStructure`."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np

from functional_process.cottax.importer import ArrayInput, Imported, read_indat
from functional_process.vocabulary.iteration_variables import ITERATION_VARIABLES


class Full(NamedTuple):
    """A constant array in `DATACLASS_DEFAULTS`, expanded on read."""

    value: float
    shape: tuple[int, ...]


def _expand(value):
    """A `Full` as its array; anything else unchanged."""
    return (
        np.full(value.shape, value.value, dtype=float) if type(value) is Full else value
    )


# --------------------------------------------------------------- the vendored defaults

DATACLASS_DEFAULTS: dict[tuple[str, str], Any] = {
    ("build", "dr_blkt_inboard"): 0.115,
    ("build", "dr_blkt_outboard"): 0.235,
    ("build", "dr_bore"): 1.42,
    ("build", "dr_cryostat"): 0.07,
    ("build", "dr_cs"): 0.811,
    ("build", "dr_cs_tf_gap"): 0.08,
    ("build", "dr_fw_plasma_gap_inboard"): 0.14,
    ("build", "dr_fw_plasma_gap_outboard"): 0.15,
    ("build", "dr_shld_blkt_gap"): 0.05,
    ("build", "dr_shld_inboard"): 0.69,
    ("build", "dr_shld_outboard"): 1.05,
    ("build", "dr_shld_thermal_inboard"): 0.05,
    ("build", "dr_shld_thermal_outboard"): 0.05,
    ("build", "dr_shld_vv_gap_inboard"): 0.155,
    ("build", "dr_tf_inboard"): 0.0,
    ("build", "dr_tf_shld_gap"): 0.05,
    ("build", "dr_vv_inboard"): 0.07,
    ("build", "dr_vv_outboard"): 0.07,
    ("build", "dr_vv_shells"): 0.12,
    ("build", "dz_fw_plasma_gap"): 0.6,
    ("build", "dz_shld_lower"): 0.7,
    ("build", "dz_shld_thermal"): 0.05,
    ("build", "dz_shld_upper"): 0.6,
    ("build", "dz_shld_vv_gap"): 0.163,
    ("build", "dz_vv_lower"): 0.07,
    ("build", "dz_vv_upper"): 0.07,
    ("build", "dz_xpoint_divertor"): 0.0,
    ("build", "f_dr_tf_outboard_inboard"): 1.19,
    ("build", "f_z_cryostat"): 4.268,
    ("build", "fcspc"): 0.6,
    ("build", "fseppc"): 350000000.0,
    ("build", "gapomin"): 0.234,
    ("build", "i_blkt_inboard"): 1,
    ("build", "i_cs_precomp"): 1,
    ("build", "i_r_cp_top"): 0,
    ("build", "i_tf_inside_cs"): 0,
    ("build", "iohcl"): 1,
    ("build", "plleni"): 1.0,
    ("build", "plleno"): 1.0,
    ("build", "plsepi"): 1.0,
    ("build", "plsepo"): 1.5,
    ("build", "r_cp_top"): 0.0,
    ("build", "r_tf_inboard_mid"): 0.0,
    ("build", "sigallpc"): 300000000.0,
    ("buildings", "admv"): 100000.0,
    ("buildings", "clh2"): 15.0,
    ("buildings", "conv"): 60000.0,
    ("buildings", "dz_tf_cryostat"): 2.5,
    ("buildings", "esbldgm3"): 1000.0,
    ("buildings", "fndt"): 2.0,
    ("buildings", "hccl"): 5.0,
    ("buildings", "hcwt"): 1.5,
    ("buildings", "i_bldgs_size"): 0,
    ("buildings", "mbvfac"): 2.8,
    ("buildings", "pfbldgm3"): 20000.0,
    ("buildings", "pibv"): 20000.0,
    ("buildings", "rbrt"): 1.0,
    ("buildings", "rbvfac"): 1.6,
    ("buildings", "rbwt"): 2.0,
    ("buildings", "row"): 4.0,
    ("buildings", "rxcl"): 4.0,
    ("buildings", "shmf"): 0.5,
    ("buildings", "shov"): 100000.0,
    ("buildings", "stcl"): 3.0,
    ("buildings", "trcl"): 1.0,
    ("buildings", "triv"): 40000.0,
    ("buildings", "wgt"): 500000.0,
    ("buildings", "wgt2"): 100000.0,
    ("buildings", "wsvfac"): 1.9,
    ("ccfe_hcpb", "fw_armour_u_nuc_heating"): 6.25e-07,
    ("constraints", "b_tf_inboard_max"): 12.0,
    ("constraints", "f_fw_rad_max"): 3.33,
    ("constraints", "f_h_mode_margin"): 1.0,
    ("constraints", "f_j_tf_wp_critical_max"): 0.7,
    ("constraints", "f_nd_plasma_electron_limit_max"): 1.0,
    ("constraints", "f_p_plasma_separatrix_rad_max"): 1.0,
    ("constraints", "f_t_alpha_energy_confinement_min"): 5.0,
    ("constraints", "fjohc"): 0.7,
    ("constraints", "fjohc0"): 0.7,
    ("constraints", "flu_tf_neutron_fast_max"): 1e23,
    ("constraints", "i_q95_fixed"): 0,
    ("constraints", "p_div_bt_q_aspect_rmajor_max_mw"): 9.5,
    ("constraints", "p_fusion_total_max_mw"): 1500.0,
    ("constraints", "p_hcd_injected_min_mw"): 0.1,
    ("constraints", "p_plant_electric_net_required_mw"): 1000.0,
    ("constraints", "p_plasma_separatrix_rmajor_max_mw"): 25.0,
    ("constraints", "pflux_fw_neutron_max_mw"): 1.0,
    ("constraints", "pflux_fw_rad_max"): 1.0,
    ("constraints", "pflux_fw_rad_max_mw"): 0.0,
    ("constraints", "q95_fixed"): 3.0,
    ("constraints", "t_burn_min"): 1.0,
    ("costs", "UCAD"): 180.0,
    ("costs", "UCAF"): 1500000.0,
    ("costs", "UCAHTS"): 31.0,
    ("costs", "UCAP"): 17.0,
    ("costs", "UCBPMP"): 292500.0,
    ("costs", "UCCO"): 350.0,
    ("costs", "UCCPMP"): 390000.0,
    ("costs", "UCCR"): 460.0,
    ("costs", "UCDGEN"): 1700000.0,
    ("costs", "UCDTC"): 13.0,
    ("costs", "UCDUCT"): 42250.0,
    ("costs", "UCEL"): 380.0,
    ("costs", "UCFPR"): 44000000.0,
    ("costs", "UCFWA"): 60000.0,
    ("costs", "UCFWPS"): 10000000.0,
    ("costs", "UCFWS"): 53000.0,
    ("costs", "UCGSS"): 35.0,
    ("costs", "UCINT"): 35.0,
    ("costs", "UCLV"): 16.0,
    ("costs", "UCMB"): 260.0,
    ("costs", "UCNBV"): 1000.0,
    ("costs", "UCPHX"): 15.0,
    ("costs", "UCPP"): 48.0,
    ("costs", "UCSH"): 115.0,
    ("costs", "UCSWYD"): 18400000.0,
    ("costs", "UCTFDR"): 0.000175,
    ("costs", "UCTFGR"): 5000.0,
    ("costs", "UCTFIC"): 10000.0,
    ("costs", "UCTPMP"): 110500.0,
    ("costs", "UCTR"): 370.0,
    ("costs", "UCVALV"): 390000.0,
    ("costs", "UCVDSH"): 26.0,
    ("costs", "UCVIAC"): 1300000.0,
    ("costs", "UCWS"): 460.0,
    ("costs", "abktflnc"): 5.0,
    ("costs", "adivflnc"): 7.0,
    ("costs", "c2214"): 0.0,
    ("costs", "c2222"): 0.0,
    ("costs", "c2252"): 0.0,
    ("costs", "cconfix"): 80.0,
    ("costs", "cconshpf"): 70.0,
    ("costs", "cconshtf"): 75.0,
    ("costs", "cfind"): [0.244, 0.244, 0.244, 0.29],
    ("costs", "cland"): 19.2,
    ("costs", "cowner"): 0.15,
    ("costs", "cplife"): 0.0,
    ("costs", "cpstcst"): 0.0,
    ("costs", "csi"): 16.0,
    ("costs", "cturbb"): 38.0,
    ("costs", "decomf"): 0.1,
    ("costs", "dintrt"): 0.0,
    ("costs", "discount_rate"): 0.0435,
    ("costs", "dtlife"): 0.0,
    ("costs", "f_t_plant_available"): 0.75,
    ("costs", "fcap0"): 1.165,
    ("costs", "fcap0cp"): 1.08,
    ("costs", "fcdfuel"): 0.1,
    ("costs", "fcontng"): 0.195,
    ("costs", "fcr0"): 0.0966,
    ("costs", "fkind"): 1.0,
    ("costs", "i_cost_model"): 1,
    ("costs", "i_plant_availability"): 2,
    ("costs", "ibkt_life"): 0,
    ("costs", "ifueltyp"): 0,
    ("costs", "ireactor"): 1,
    ("costs", "life_dpa"): 50,
    ("costs", "life_plant"): 30.0,
    ("costs", "lsa"): 4,
    ("costs", "output_costs"): 1,
    ("costs", "ucblbe"): 260.0,
    ("costs", "ucblli2o"): 600.0,
    ("costs", "ucblss"): 90.0,
    ("costs", "ucblvd"): 200.0,
    ("costs", "ucbus"): 0.123,
    ("costs", "uccase"): 50.0,
    ("costs", "uccry"): 93000.0,
    ("costs", "uccryo"): 32.0,
    ("costs", "uccu"): 75.0,
    ("costs", "ucdiv"): 280000.0,
    ("costs", "ucech"): 3.0,
    ("costs", "ucf1"): 22300000.0,
    ("costs", "ucfnc"): 35.0,
    ("costs", "ucfuel"): 3.45,
    ("costs", "uche3"): 1000000.0,
    ("costs", "uchrs"): 87900000.0,
    ("costs", "uchts"): [15.3, 19.1],
    ("costs", "uciac"): 150000000.0,
    ("costs", "ucich"): 3.0,
    ("costs", "uclh"): 3.3,
    ("costs", "ucme"): 125000000.0,
    ("costs", "ucmisc"): 25000000.0,
    ("costs", "ucnbi"): 3.3,
    ("costs", "ucoam"): [68.8, 68.8, 68.8, 74.4],
    ("costs", "ucpens"): 32.0,
    ("costs", "ucpfb"): 210.0,
    ("costs", "ucpfbk"): 16600.0,
    ("costs", "ucpfbs"): 4900.0,
    ("costs", "ucpfcb"): 75000.0,
    ("costs", "ucpfdr1"): 150.0,
    ("costs", "ucpfic"): 10000.0,
    ("costs", "ucpfps"): 35000.0,
    ("costs", "ucrb"): 400.0,
    ("costs", "ucsc"): [600.0, 600.0, 300.0, 600.0, 600.0, 600.0, 300.0, 1200.0, 1200.0],
    ("costs", "ucshld"): 32.0,
    ("costs", "uctfbr"): 1.22,
    ("costs", "uctfbus"): 100.0,
    ("costs", "uctfps"): 24.0,
    ("costs", "uctfsw"): 1.0,
    ("costs", "ucturb"): [230000000.0, 245000000.0],
    ("costs", "ucwindpf"): 465.0,
    ("costs", "ucwindtf"): 480.0,
    ("costs", "ucwst"): [0.0, 3.94, 5.91, 7.88],
    ("cs_fatigue", "bkt_life_csf"): 0.0,
    ("cs_fatigue", "dr_cs_turn_conduit"): 0.07,
    ("cs_fatigue", "dz_cs_turn_conduit"): 0.022,
    ("cs_fatigue", "fracture_toughness"): 200.0,
    ("cs_fatigue", "n_cycle_min"): 20000.0,
    ("cs_fatigue", "paris_coefficient"): 6.5e-13,
    ("cs_fatigue", "paris_power_law"): 3.5,
    ("cs_fatigue", "residual_sig_hoop"): 240000000.0,
    ("cs_fatigue", "sf_fast_fracture"): 1.5,
    ("cs_fatigue", "sf_radial_crack"): 2.0,
    ("cs_fatigue", "sf_vertical_crack"): 2.0,
    ("cs_fatigue", "t_crack_vertical"): 0.00089,
    ("cs_fatigue", "walker_coefficient"): 0.436,
    ("current_drive", "cboot"): 1.0,
    ("current_drive", "e_beam_kev"): 1000.0,
    ("current_drive", "eta_cd_norm_ecrh"): 0.35,
    ("current_drive", "eta_ecrh_injector_wall_plug"): 0.3,
    ("current_drive", "f_beam_tritium"): 1e-06,
    ("current_drive", "f_c_plasma_bootstrap_max"): 0.9,
    ("current_drive", "feffcd"): 1.0,
    ("current_drive", "i_ecrh_wave_mode"): 0,
    ("current_drive", "i_hcd_calculations"): 1,
    ("current_drive", "i_hcd_primary"): 5,
    ("current_drive", "n_ecrh_harmonic"): 2.0,
    ("current_drive", "p_beam_injected_mw"): 0.0,
    ("current_drive", "p_beam_orbit_loss_mw"): 0.0,
    ("current_drive", "p_beam_shine_through_mw"): 0.0,
    ("current_drive", "p_hcd_beam_injected_total_mw"): 0.0,
    ("current_drive", "p_hcd_injected_max"): 150.0,
    ("current_drive", "p_hcd_lowhyb_injected_total_mw"): 0.0,
    ("current_drive", "p_hcd_primary_extra_heat_mw"): 0.0,
    ("current_drive", "p_hcd_secondary_injected_mw"): 0.0,
    ("divertor", "anginc"): 0.262,
    ("divertor", "betai"): 1.0,
    ("divertor", "betao"): 1.0,
    ("divertor", "deg_div_field_plate"): 1.0,
    ("divertor", "den_div_structure"): 10000.0,
    ("divertor", "dx_div_plate"): 0.035,
    ("divertor", "dz_divertor"): 0.2,
    ("divertor", "f_div_flux_expansion"): 2.0,
    ("divertor", "f_vol_div_coolant"): 0.3,
    ("divertor", "fdiva"): 1.11,
    ("divertor", "i_div_heat_load"): 2,
    ("divertor", "n_divertors"): 2,
    ("divertor", "pflux_div_heat_load_max_mw"): 5.0,
    ("divertor", "prn1"): 0.285,
    ("divertor", "tdiv"): 2.0,
    ("divertor", "xpertin"): 2.0,
    ("fwbs", "breeder_f"): 0.5,
    ("fwbs", "breeder_multiplier"): 0.75,
    ("fwbs", "declblkt"): 0.075,
    ("fwbs", "declfw"): 0.075,
    ("fwbs", "declshld"): 0.075,
    ("fwbs", "den_steel"): 7800.0,
    ("fwbs", "dr_fw_wall"): 0.003,
    ("fwbs", "dr_pf_cryostat"): 0.5,
    ("fwbs", "dx_fw_module"): 0.02,
    ("fwbs", "eta_coolant_pump_electric"): 0.95,
    ("fwbs", "etaiso"): 0.85,
    ("fwbs", "f_a_blkt_cooling_channels"): 0.25,
    ("fwbs", "f_a_fw_outboard_hcd"): 0.0,
    ("fwbs", "f_nuc_pow_bz_liq"): 0.66,
    ("fwbs", "f_p_blkt_multiplication"): 1.269,
    ("fwbs", "f_ster_div_single"): 0.115,
    ("fwbs", "fblbe"): 0.6,
    ("fwbs", "fblli2o"): 0.08,
    ("fwbs", "fbllipb"): 0.68,
    ("fwbs", "fblss"): 0.09705,
    ("fwbs", "fblvd"): 0.0,
    ("fwbs", "fhole"): 0.0,
    ("fwbs", "fvoldw"): 1.74,
    ("fwbs", "fvolsi"): 1.0,
    ("fwbs", "fvolso"): 0.64,
    ("fwbs", "fw_armour_thickness"): 0.005,
    ("fwbs", "fwclfr"): 0.15,
    ("fwbs", "i_blanket_type"): 1,
    ("fwbs", "i_blkt_coolant_type"): 1,
    ("fwbs", "i_fw_blkt_shared_coolant"): 0,
    ("fwbs", "i_fw_blkt_vv_shape"): 2,
    ("fwbs", "i_fw_coolant_type"): 1,
    ("fwbs", "i_p_coolant_pumping"): 2,
    ("fwbs", "i_thermal_electric_conversion"): 0,
    ("fwbs", "inuclear"): 0,
    ("fwbs", "life_fw_fpy"): 0.0,
    ("fwbs", "m_blkt_vanadium"): 0.0,
    ("fwbs", "outlet_temp_liq"): 720.0,
    ("fwbs", "p_cp_shield_nuclear_heat_mw"): 0.0,
    ("fwbs", "p_div_rad_total_mw"): 0.0,
    ("fwbs", "p_fw_hcd_nuclear_heat_mw"): 0.0,
    ("fwbs", "qnuc"): 0.0,
    ("fwbs", "radius_fw_channel"): 0.006,
    ("fwbs", "vfcblkt"): 0.05295,
    ("fwbs", "vfpblkt"): 0.1,
    ("fwbs", "vfshld"): 0.25,
    ("globals", "maxcal"): 200,
    (
        "globals",
        "runtitle",
    ): "Run Title (change this line using input variable 'runtitle')",
    ("heat_transport", "eta_turbine"): 0.35,
    ("heat_transport", "etatf"): 0.9,
    ("heat_transport", "f_p_blkt_coolant_pump_total_heat"): 0.005,
    ("heat_transport", "f_p_div_coolant_pump_total_heat"): 0.005,
    ("heat_transport", "f_p_fw_coolant_pump_total_heat"): 0.005,
    ("heat_transport", "f_p_shld_coolant_pump_total_heat"): 0.005,
    ("heat_transport", "i_shld_primary_heat"): 1,
    ("heat_transport", "ipowerflow"): 1,
    ("heat_transport", "p_blkt_breeder_pump_mw"): 0.0,
    ("heat_transport", "p_blkt_coolant_pump_mw"): 0.0,
    ("heat_transport", "p_div_coolant_pump_mw"): 0.0,
    ("heat_transport", "p_fw_coolant_pump_mw"): 0.0,
    ("heat_transport", "p_fw_div_heat_deposited_mw"): 0.0,
    ("heat_transport", "p_plant_electric_base"): 5000000.0,
    ("heat_transport", "p_shld_coolant_pump_mw"): 0.0,
    ("heat_transport", "p_tritium_plant_electric_mw"): 15.0,
    ("heat_transport", "peakmva"): 0.0,
    ("heat_transport", "pflux_plant_floor_electric"): 150.0,
    ("heat_transport", "vachtmw"): 0.5,
    ("ife", "ife"): 0,
    ("impurity_radiation", "f_nd_impurity_electron_array"): Full(0.0, (14,)),
    ("impurity_radiation", "f_nd_impurity_electrons"): [
        1.0,
        0.1,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    ("impurity_radiation", "f_p_plasma_core_rad_reduction"): 1.0,
    ("impurity_radiation", "impurity_arr_zav"): Full(0.0, (14, 200)),
    ("impurity_radiation", "m_impurity_amu_array"): Full(0.0, (14,)),
    ("impurity_radiation", "pden_impurity_lz_nd_temp_array"): Full(0.0, (14, 200)),
    ("impurity_radiation", "radius_plasma_core_norm"): 0.6,
    ("impurity_radiation", "temp_impurity_keV_array"): Full(0.0, (14, 200)),
    ("numerics", "boundl"): Full(9e-99, (177,)),
    ("numerics", "boundu"): Full(9e99, (177,)),
    ("numerics", "epsfcn"): 0.001,
    ("numerics", "epsvmc"): 1e-06,
    ("numerics", "i_figure_merit"): 7,
    ("numerics", "i_process_run_mode"): 1,
    ("numerics", "n_equality_constraints"): -1,
    ("pf_coil", "alfapf"): 5e-10,
    ("pf_coil", "c_pf_coil_turn_peak_input"): Full(40000.0, (22,)),
    ("pf_coil", "dr_pf_tf_outboard_out_offset"): 1.5,
    ("pf_coil", "etapsu"): 0.9,
    ("pf_coil", "f_a_cs_turn_steel"): 0.5,
    ("pf_coil", "f_a_cs_void"): 0.3,
    ("pf_coil", "f_a_pf_coil_void"): Full(0.3, (22,)),
    ("pf_coil", "f_dr_dz_cs_turn"): 3.1818181818181817,
    ("pf_coil", "f_j_cs_start_pulse_end_flat_top"): 0.9,
    ("pf_coil", "f_z_cs_tf_internal"): 0.71,
    ("pf_coil", "fcuohsu"): 0.7,
    ("pf_coil", "fcupfsu"): 0.69,
    ("pf_coil", "i_cs_superconductor"): 1,
    ("pf_coil", "i_pf_conductor"): 0,
    ("pf_coil", "i_pf_location"): [2, 2, 3, 0, 0, 0, 0, 0, 0, 0],
    ("pf_coil", "i_pf_superconductor"): 1,
    ("pf_coil", "i_r_pf_outside_tf_placement"): 0,
    ("pf_coil", "ind_pf_cs_plasma_mutual"): Full(0.0, (22, 22)),
    ("pf_coil", "j_cs_flat_top_end"): 18500000.0,
    ("pf_coil", "j_pf_coil_wp_peak"): Full(30000000.0, (22,)),
    ("pf_coil", "m_pf_coil_max"): 0.0,
    ("pf_coil", "n_pf_coil_groups"): 3,
    ("pf_coil", "n_pf_coil_turns"): Full(0.0, (22,)),
    ("pf_coil", "n_pf_coils_in_group"): [1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ("pf_coil", "p_pf_electric_supplies_mw"): 0.0,
    ("pf_coil", "pf_current_safety_factor"): 1.0,
    ("pf_coil", "r_pf_coil_outer_max"): 0.0,
    ("pf_coil", "radius_cs_turn_corners"): 0.003,
    ("pf_coil", "rho_pf_coil"): 2.5e-08,
    ("pf_coil", "rhopfbus"): 3.93e-08,
    ("pf_coil", "rpf2"): -1.63,
    ("pf_coil", "rref"): Full(7.0, (10,)),
    ("pf_coil", "sigpfcalw"): 500.0,
    ("pf_coil", "sigpfcf"): 1.0,
    ("pf_coil", "stress_cs_steel_max"): 400000000.0,
    ("pf_coil", "stress_hoop_cs_inner"): 0.0,
    ("pf_coil", "temp_cs_superconductor_operating"): 4.75,
    ("pf_coil", "zref"): [3.6, 1.2, 2.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ("pf_power", "ensxpfm"): 0.0,
    ("pf_power", "f_p_pf_energy_store_loss"): 0.1,
    ("pf_power", "f_p_pf_psu_loss"): 0.1,
    ("pf_power", "srcktpm"): 0.0,
    ("physics", "alphaj"): 1.0,
    ("physics", "alphan"): 0.25,
    ("physics", "alphat"): 0.5,
    ("physics", "aspect"): 2.907,
    ("physics", "b_plasma_toroidal_on_axis"): 5.68,
    ("physics", "beta_beam"): 0.0,
    ("physics", "beta_norm_max"): 3.5,
    ("physics", "beta_thermal_vol_avg"): 0.0,
    ("physics", "beta_toroidal_vol_avg"): 0.0,
    ("physics", "beta_total_vol_avg"): 0.042,
    ("physics", "beta_vol_avg_max"): 0.0,
    ("physics", "beta_vol_avg_min"): 0.0,
    ("physics", "burnup_in"): 0.0,
    ("physics", "csawth"): 1.0,
    ("physics", "dlamie"): 0.0,
    ("physics", "ejima_coeff"): 0.4,
    ("physics", "f_c_plasma_non_inductive"): 1.0,
    ("physics", "f_nd_alpha_thermal_electron"): 0.1,
    ("physics", "f_nd_beam_electron"): 0.005,
    ("physics", "f_nd_plasma_pedestal_greenwald"): 0.85,
    ("physics", "f_nd_plasma_separatrix_greenwald"): 0.5,
    ("physics", "f_nd_protium_electrons"): 0.0,
    ("physics", "f_p_alpha_plasma_deposited"): 0.95,
    ("physics", "f_p_div_lower"): 1.0,
    ("physics", "f_plasma_fuel_deuterium"): 0.5,
    ("physics", "f_plasma_fuel_helium3"): 0.0,
    ("physics", "f_plasma_fuel_tritium"): 0.5,
    ("physics", "f_sync_reflect"): 0.6,
    ("physics", "f_temp_plasma_electron_density_vol_avg"): 0.0,
    ("physics", "f_temp_plasma_ion_electron"): 1.0,
    ("physics", "f_vol_plasma"): 1.0,
    ("physics", "ffwal"): 0.92,
    ("physics", "fkzohm"): 1.0,
    ("physics", "fusden_alpha_total"): 0.0,
    ("physics", "hfact"): 1.0,
    ("physics", "i_alphaj"): 0,
    ("physics", "i_beta_component"): 0,
    ("physics", "i_beta_fast_alpha"): 1,
    ("physics", "i_beta_norm_max"): 1,
    ("physics", "i_bootstrap_current"): 3,
    ("physics", "i_confinement_time"): 34,
    ("physics", "i_density_limit"): 8,
    ("physics", "i_diamagnetic_current"): 0,
    ("physics", "i_ind_plasma_internal_norm"): 0,
    ("physics", "i_pfirsch_schluter_current"): 0,
    ("physics", "i_plasma_current"): 4,
    ("physics", "i_plasma_geometry"): 0,
    ("physics", "i_plasma_ignited"): 0,
    ("physics", "i_plasma_pedestal"): 1,
    ("physics", "i_rad_loss"): 1,
    ("physics", "i_single_null"): 1,
    ("physics", "ind_plasma_internal_norm"): 0.9,
    ("physics", "itart"): 0,
    ("physics", "itartpf"): 0,
    ("physics", "kappa"): 1.792,
    ("physics", "m_s_limit"): 0.3,
    ("physics", "nd_plasma_electrons_vol_avg"): 9.8e19,
    ("physics", "nd_plasma_pedestal_electron"): 4e19,
    ("physics", "nd_plasma_separatrix_electron"): 3e19,
    ("physics", "p_beam_alpha_mw"): 0.0,
    ("physics", "p_plasma_ohmic_mw"): 0.0,
    ("physics", "p_plasma_separatrix_rmajor_mw"): 0.0,
    ("physics", "pden_plasma_ohmic_mw"): 0.0,
    ("physics", "plasma_current"): 0.0,
    ("physics", "plasma_res_factor"): 1.0,
    ("physics", "proton_rate_density"): 0.0,
    ("physics", "psolradmw"): 0.0,
    ("physics", "q0"): 1.0,
    ("physics", "q95"): 0.0,
    ("physics", "rad_fraction_sol"): 0.8,
    ("physics", "radius_plasma_pedestal_density_norm"): 1.0,
    ("physics", "radius_plasma_pedestal_temp_norm"): 1.0,
    ("physics", "rmajor"): 8.14,
    ("physics", "tauratio"): 1.0,
    ("physics", "tbeta"): 2.0,
    ("physics", "temp_plasma_electron_vol_avg_kev"): 12.9,
    ("physics", "temp_plasma_ion_vol_avg_kev"): 12.9,
    ("physics", "temp_plasma_pedestal_kev"): 1.0,
    ("physics", "temp_plasma_separatrix_kev"): 0.1,
    ("physics", "triang"): 0.36,
    ("power", "delta_eta"): 0.0,
    ("primary_pumping", "dp_he"): 550000.0,
    ("primary_pumping", "f_p_fw_blkt_pump"): 1.0,
    ("primary_pumping", "gamma_he"): 1.667,
    ("primary_pumping", "p_he"): 8000000.0,
    ("primary_pumping", "t_in_bb"): 573.13,
    ("primary_pumping", "t_out_bb"): 773.13,
    ("pulse", "i_pulsed_plant"): 0,
    ("stellarator", "bmn"): 0.001,
    ("stellarator", "f_asym"): 1.0,
    ("stellarator", "f_rad"): 0.85,
    ("stellarator", "f_st_coil_aspect"): 1.0,
    ("stellarator", "f_w"): 0.5,
    ("stellarator", "fdivwet"): 0.333333333333333,
    ("stellarator", "flpitch"): 0.001,
    ("stellarator", "iotabar"): 1.0,
    ("stellarator", "istell"): 0,
    ("stellarator", "isthtr"): 1,
    ("stellarator", "m_res"): 5,
    ("stellarator", "max_gyrotron_frequency"): 1000000000.0,
    ("stellarator", "n_res"): 5,
    ("stellarator", "shear"): 0.5,
    ("superconducting_tfcoil", "dx_tf_croco_strand_copper"): 0.0025,
    ("superconducting_tfcoil", "dx_tf_hts_tape_copper"): 0.0001,
    ("superconducting_tfcoil", "dx_tf_hts_tape_hastelloy"): 5e-05,
    ("superconducting_tfcoil", "dx_tf_hts_tape_rebco"): 1e-06,
    ("superconducting_tfcoil", "i_tf_turn_type"): 1,
    ("tfcoil", "a_tf_wp_coolant_channels"): 0.0,
    ("tfcoil", "c_tf_turn"): 70000.0,
    ("tfcoil", "casths_fraction"): 0.06,
    ("tfcoil", "dcond"): [
        6080.0,
        6080.0,
        6070.0,
        6080.0,
        6080.0,
        8500.0,
        6070.0,
        8500.0,
        8500.0,
    ],
    ("tfcoil", "den_tf_coil_case"): 8000.0,
    ("tfcoil", "den_tf_wp_turn_insulation"): 1800.0,
    ("tfcoil", "dia_tf_turn_coolant_channel"): 0.005,
    ("tfcoil", "dr_tf_nose_case"): 0.3,
    ("tfcoil", "dr_tf_plasma_case"): 0.0,
    ("tfcoil", "dr_tf_wp_with_insulation"): 0.0,
    ("tfcoil", "dx_tf_side_case_min"): 0.0,
    ("tfcoil", "dx_tf_turn_general"): 0.0,
    ("tfcoil", "dx_tf_turn_insulation"): 0.0008,
    ("tfcoil", "dx_tf_turn_steel"): 0.008,
    ("tfcoil", "dx_tf_wp_insertion_gap"): 0.01,
    ("tfcoil", "dx_tf_wp_insulation"): 0.018,
    ("tfcoil", "dx_tf_wp_primary_toroidal"): 0.0,
    ("tfcoil", "eff_tf_cryo"): -1.0,
    ("tfcoil", "eyoung_cond_axial"): 660000000.0,
    ("tfcoil", "eyoung_cond_trans"): 0.0,
    ("tfcoil", "eyoung_copper"): 117000000000.0,
    ("tfcoil", "eyoung_ins"): 100000000.0,
    ("tfcoil", "eyoung_res_tf_buck"): 150000000000.0,
    ("tfcoil", "eyoung_steel"): 205000000000.0,
    ("tfcoil", "f_a_tf_turn_cable_copper"): 0.69,
    ("tfcoil", "f_a_tf_turn_cable_space_extra_void"): 0.4,
    ("tfcoil", "f_dr_tf_plasma_case"): 0.05,
    ("tfcoil", "f_vforce_inboard"): 0.5,
    ("tfcoil", "i_cp_joints"): -1,
    ("tfcoil", "i_tf_bucking"): -1,
    ("tfcoil", "i_tf_case_geom"): 0,
    # The two `resolve_eyoung_cond` reads. Added 2026-09-04 (`_audit/optimise_design.md`
    # §34): a stated value's resolution asks the state for its switch, and until these
    # were here every tokamak row reported them on `state.missing`. The *number* was
    # right either way -- `indat._stated_get` falls back to the same
    # `I_TF_COND_EYOUNG_*_DEFAULT` `imported.get` uses -- but a miss list is a work list,
    # and answering it here is doing the work rather than logging it.
    ("tfcoil", "i_tf_cond_eyoung_axial"): 0,
    ("tfcoil", "i_tf_cond_eyoung_trans"): 1,
    ("tfcoil", "i_tf_sc_mat"): 1,
    ("tfcoil", "i_tf_shape"): 0,
    ("tfcoil", "i_tf_stress_model"): 1,
    ("tfcoil", "i_tf_sup"): 1,
    ("tfcoil", "i_tf_tresca"): 0,
    ("tfcoil", "i_tf_turns_integer"): 0,
    ("tfcoil", "i_tf_wp_geom"): -1,
    ("tfcoil", "layer_ins"): 0.0,
    ("tfcoil", "m_tf_bus"): 0.0,
    ("tfcoil", "max_vv_stress"): 143000000.0,
    ("tfcoil", "n_tf_coils"): 16.0,
    ("tfcoil", "n_tf_wp_layers"): 20,
    ("tfcoil", "n_tf_wp_pancakes"): 10,
    ("tfcoil", "poisson_cond_axial"): 0.3,
    ("tfcoil", "poisson_cond_trans"): 0.3,
    ("tfcoil", "poisson_copper"): 0.35,
    ("tfcoil", "poisson_ins"): 0.34,
    ("tfcoil", "poisson_steel"): 0.3,
    ("tfcoil", "res_tf_leg"): 0.0,
    ("tfcoil", "rho_tf_bus"): 1.86e-08,
    ("tfcoil", "ripple_b_tf_plasma_edge_max"): 1.0,
    ("tfcoil", "rrr_tf_cu"): 100.0,
    ("tfcoil", "sig_tf_case_max"): 600000000.0,
    ("tfcoil", "sig_tf_cs_bucked"): 0.0,
    ("tfcoil", "sig_tf_wp_max"): 600000000.0,
    ("tfcoil", "str_cs_con_res"): -0.005,
    ("tfcoil", "t_tf_quench_detection"): 3.0,
    ("tfcoil", "t_tf_superconductor_quench"): 10.0,
    ("tfcoil", "temp_cp_coolant_inlet"): 313.15,
    ("tfcoil", "temp_cs_superconductor_margin_min"): 0.0,
    ("tfcoil", "temp_tf_cryo"): 4.5,
    ("tfcoil", "temp_tf_superconductor_margin_min"): 0.0,
    ("tfcoil", "tfcmw"): 0.0,
    ("tfcoil", "tftmp"): 4.5,
    ("tfcoil", "theta1_coil"): 45.0,
    ("tfcoil", "theta1_vv"): 1.0,
    ("tfcoil", "tmargmin"): 0.0,
    ("tfcoil", "v_tf_coil_dump_quench_max_kv"): 20.0,
    ("times", "pulsetimings"): 1,
    ("times", "t_plant_pulse_burn"): Full(1000.0, ()),
    ("times", "t_plant_pulse_coil_precharge"): 15.0,
    ("times", "t_plant_pulse_dwell"): 1800.0,
    ("times", "t_plant_pulse_fusion_ramp"): 10.0,
    ("times", "t_plant_pulse_plasma_current_ramp_down"): 15.0,
    ("times", "t_plant_pulse_plasma_current_ramp_up"): 30.0,
    ("vacuum", "i_vac_pump_dwell"): 0,
    ("vacuum", "i_vacuum_pump_type"): 1,
    ("vacuum", "outgrat_fw"): 1.3e-08,
    ("vacuum", "pres_div_chamber_burn"): 0.36,
    ("vacuum", "pres_vv_chamber_base"): 0.0005,
}
"""`(area, field) -> PROCESS's `DataStructure` dataclass default`."""

# ------------------------------------------------------------------------- the state


@dataclass
class _Area:
    """One `data.<area>`. Answers the fields it holds; records the ones it does not."""

    name: str
    values: dict[str, Any]
    missing: list[tuple[str, str]]

    def __getattr__(self, field_name: str):
        """The field, or an `AttributeError` **and a row in `missing`**."""
        if field_name.startswith("__"):
            raise AttributeError(field_name)
        if field_name in self.values:
            return self.values[field_name]
        self.missing.append((self.name, field_name))
        raise AttributeError(
            f"`.{self.name}.{field_name}` has no native answer -- no IN.DAT line sets "
            f"it and it is not in `DATACLASS_DEFAULTS`"
        )


@dataclass
class NativeState:
    """What a solve reads instead of PROCESS's `DataStructure`."""

    areas: dict[str, _Area]
    values: dict[tuple[str, str], Any]
    missing: list[tuple[str, str]] = field(default_factory=list)
    sources: dict[tuple[str, str], str] = field(default_factory=dict)
    """`indat` or `defaults` per answered place -- §22.6's `source` column, minus the
    `process` row that no longer exists here.
    """

    def __getattr__(self, area_name: str):
        """The area, or an `AttributeError`."""
        if area_name.startswith("__"):
            raise AttributeError(area_name)
        if area_name in self.areas:
            return self.areas[area_name]
        raise AttributeError(f"no area `{area_name}`")

    def get(self, area: str, name: str, default=None):
        """One place's value, without going through the area objects."""
        return self.values.get((area, name), default)


def _array_from(imported_value: ArrayInput, default):
    """One `IN.DAT` array assignment applied to its dataclass default."""
    base = np.asarray(_expand(default), dtype=float)
    out = np.zeros_like(base) if imported_value.zero_filled else base.copy()
    flat = out.reshape(-1)
    for index, value in imported_value.elements:
        if index < flat.size:
            flat[index] = value
    return out


# ------------------------------------------------------------------- the derivations

_Places = dict[tuple[str, str], Any]
"""`(area, field) -> value`."""

#
# The third source, and the last one. `_audit/init_audit.md` counts **zero** genuine
# parse-time inputs among `init.py`'s 35 writes: every one of them is a sentinel
# resolved, a presence flag, or a *derivation* from something the file already states.
# So a place `init.py` writes is not a place a native state has to be told about -- it is
# a place it has to *work out*, and answering it with the dataclass default is answering
# a field PROCESS's own initialisation has already overwritten before any model runs.
#
# These run **after** the file's values, because that is where they run in PROCESS:
# `SingleRun.initialise` calls `initialise_imprad` and then `init_process`, and
# `init_process` parses the `IN.DAT` before applying any of the rules below.


def _initialise_imprad(values: _Places, out: _Places) -> None:
    """`process/main.py:430`'s four tables -- `init_audit.md` §5's fifth source."""
    from functional_process.models.physics.impurity_radiation import (  # noqa: PLC0415
        M_IMPURITY_AMU_ARRAY,
        impurity_tables,
    )

    for name, table in impurity_tables().items():
        out["impurity_radiation", name] = table.copy()
    out["impurity_radiation", "m_impurity_amu_array"] = np.asarray(
        M_IMPURITY_AMU_ARRAY, dtype=float
    )


def _alias_impurity_fractions(values: _Places, out: _Places) -> None:
    """`init.py:381-384` -- `f_nd_impurity_electron_array[i]` from
    `f_nd_impurity_electrons[i]`.
    """
    declared = values.get(("impurity_radiation", "f_nd_impurity_electrons"))
    if declared is None:
        return
    out["impurity_radiation", "f_nd_impurity_electron_array"] = np.asarray(
        declared, dtype=float
    ).copy()


def _single_or_double_null(values: _Places, out: _Places) -> None:
    """`init.py:606-617` -- `.physics.i_single_null` decides four fields, not one."""
    from functional_process.cottax.indat import _n_divertors  # noqa: PLC0415
    from functional_process.vocabulary.enums import DivertorNumberModels  # noqa: PLC0415

    i_single_null = values.get(("physics", "i_single_null"))
    if i_single_null is None:
        return
    out["divertor", "n_divertors"] = _n_divertors(int(i_single_null))
    if DivertorNumberModels(int(i_single_null)) is not DivertorNumberModels.DOUBLE_NULL:
        return
    for upper, lower in (
        ("dz_fw_plasma_gap", "dz_xpoint_divertor"),
        ("dz_shld_upper", "dz_shld_lower"),
        ("dz_vv_upper", "dz_vv_lower"),
    ):
        if ("build", lower) in values:
            out["build", upper] = values["build", lower]


def _deprecated_temperature_margin_alias(values: _Places, out: _Places) -> None:
    """`init.py:1171-1190` -- `tmargmin` is a deprecated alias for two fields."""
    tmargmin = values.get(("tfcoil", "tmargmin"))
    if tmargmin is None or float(tmargmin) <= 0.0001:
        return
    out["tfcoil", "temp_tf_superconductor_margin_min"] = float(tmargmin)
    out["tfcoil", "temp_cs_superconductor_margin_min"] = float(tmargmin)


DERIVATIONS = (
    _initialise_imprad,
    _alias_impurity_fractions,
    _single_or_double_null,
    _deprecated_temperature_margin_alias,
)
"""The rules a native state applies over the file's own values, in PROCESS's own order.
"""


def native_values(input_file: str | Imported) -> tuple[dict, dict, list]:
    """`(values, sources)` -- defaults, then the file's values, then `DERIVATIONS`."""
    imported = input_file if isinstance(input_file, Imported) else read_indat(input_file)
    values: dict[tuple[str, str], Any] = {}
    sources: dict[tuple[str, str], str] = {}
    unshapeable: list[tuple[str, str]] = []
    for place, default in DATACLASS_DEFAULTS.items():
        values[place] = _expand(default)
        sources[place] = "defaults"
    for place, stated in imported.values.items():
        if isinstance(stated, ArrayInput):
            default = DATACLASS_DEFAULTS.get(place)
            if default is None:
                unshapeable.append(place)
                continue
            values[place] = _array_from(stated, default)
        else:
            values[place] = stated
        sources[place] = "indat"
    derived: _Places = {}
    for derive in DERIVATIONS:
        derive(values, derived)
    values.update(derived)
    for place in derived:
        sources[place] = "derived"
    return values, sources, unshapeable


def native_state(input_file: str | Imported) -> NativeState:
    """The `DataStructure` stand-in for one input file. **The interface.**"""
    values, sources, unshapeable = native_values(input_file)
    missing: list[tuple[str, str]] = list(unshapeable)
    areas: dict[str, _Area] = {}
    for (area, name), value in values.items():
        areas.setdefault(area, _Area(area, {}, missing)).values[name] = value
    return NativeState(areas=areas, values=values, missing=missing, sources=sources)


# ------------------------------------------------------------------- the problem side


def _pedestal_temperature_bound(ixc, state, low: float, high: float):
    """`init.py:444-459` -- iteration variable 4's lower bound, raised off the pedestal.
    """
    from functional_process.cottax.indat import (
        ST_INIT_I_PLASMA_PEDESTAL,
    )

    if 4 not in {int(i) for i in ixc}:
        return low, high
    if int(state.get("ife", "ife", 0)) == 1:
        return low, high
    if int(state.get("numerics", "i_process_run_mode", 1)) != 1:  # OPTIMISATION
        return low, high
    pedestal = int(state.get("physics", "i_plasma_pedestal", 1))
    if int(state.get("stellarator", "istell", 0)) != 0:
        pedestal = ST_INIT_I_PLASMA_PEDESTAL
    if pedestal != 1:
        return low, high
    raised = float(state.get("physics", "temp_plasma_pedestal_kev", 1.0)) * 1.001
    return (low, high) if low >= raised else (raised, max(high, raised))


def native_bounds(ixc, imported, state=None):
    """`((VarPath, lower, upper), ...)` -- `ReferenceRun.bounds`, from the file."""
    from functional_process.cottax.sand import iteration_variable_path  # noqa: PLC0415

    lower = imported.get("numerics", "boundl")
    upper = imported.get("numerics", "boundu")
    out = []
    for i in ixc:
        variable = ITERATION_VARIABLES[int(i)]
        low, high = float(variable.lower_bound), float(variable.upper_bound)
        if isinstance(lower, ArrayInput):
            low = float(lower.as_dict().get(int(i) - 1, low))
        if isinstance(upper, ArrayInput):
            high = float(upper.as_dict().get(int(i) - 1, high))
        if int(i) == 4 and state is not None:
            low, high = _pedestal_temperature_bound(ixc, state, low, high)
        out.append((iteration_variable_path(int(i)), low, high))
    return tuple(out)


@dataclass
class NativeReference:
    """`sand_harness.ReferenceRun`'s shape, with nothing from PROCESS in it."""

    data: object
    cold: object
    ixc: list
    icc: list
    n_equality: int
    i_figure_merit: int
    bounds: tuple
    solver_iterations: int | None = None
    convergence_parameter: float | None = None
    solve_seconds: float = 0.0


def native_reference(input_file: str) -> NativeReference:
    """Everything a cold solve needs for one input file, PROCESS-free."""
    from functional_process.cottax.indat import problem_from_indat  # noqa: PLC0415

    imported = read_indat(input_file)
    problem = problem_from_indat(imported)
    ixc = sorted(int(i) for i in problem.ixc)
    n_equality = problem.n_equality_constraints
    if n_equality is None:
        # `init.py`'s `-1` sentinel: the equalities are what is left over. Measured to
        # fire on 0 of 7 tracked files (§23.7), so this arm is untested by any of them.
        n_equality = len(problem.icc) - (problem.n_inequality_constraints or 0)
    state = native_state(imported)
    return NativeReference(
        data=state,
        cold=state,
        ixc=ixc,
        icc=[int(i) for i in problem.icc],
        n_equality=int(n_equality),
        i_figure_merit=int(problem.i_figure_merit or 7),
        bounds=native_bounds(ixc, imported, state),
    )


CONFIGURATIONS = (
    "tests/regression/input_files/stellarator_helias.IN.DAT",
    "tests/regression/input_files/helias_5b.IN.DAT",
    "tests/regression/input_files/large_tokamak_nof.IN.DAT",
    "tests/regression/input_files/large_tokamak_eval.IN.DAT",
    "tests/regression/input_files/low_aspect_ratio_DEMO.IN.DAT",
    "tests/regression/input_files/spherical_tokamak_eval.IN.DAT",
    "tests/regression/input_files/st_regression.IN.DAT",
)
"""Every `tests/regression/input_files/*.IN.DAT` except `IFE.IN.DAT`."""


def stem(input_file: str) -> str:
    """`.../helias_5b.IN.DAT` -> `helias_5b`, the name a configuration is known by."""
    return pathlib.PurePath(input_file).name.removesuffix(".IN.DAT")
