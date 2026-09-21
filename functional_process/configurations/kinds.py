"""The decision kinds of `stellarator_helias`: which boundary input is a belief, a
build decision, an operating variable, numerics, or a constraint limit; which model
outputs are really build decisions closed by a rule, and what was decided about each;
and the belief table an uncertainty study draws from.

PROCESS's all-at-once solve (VMCON over `ixc` / `icc`) does not distinguish these: a
confinement multiplier sits in the design column beside the major radius. A two-stage
formulation under uncertainty has to -- build decisions are made once, before the
machine exists; operating variables are set per realisation of the physics; beliefs
are drawn -- and a build quantity that a model re-sizes per belief sample violates
non-anticipativity. This module is the sort, as data: no cottax, no jax, nothing that
interprets a tag. The architecture that reads it decides what a kind means.

Every table here was extracted from `paper_tests/decision_kinds.md`,
`paper_tests/output_kinds.md`, `paper_tests/uq.py` and `paper_tests/ouu.py`, then the
2026-09-17 decisions applied; each name below says which. Spellings are the port's
(`.area.field`, `^stated.area.field`, `^guess.area.field`, `^cond.constraints.c<n>`),
exactly as the tables write them.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class Kind(enum.Enum):
    """What kind of decision a boundary input is.

    `decision_kinds.md`'s five kinds plus one for a constraint's limit input, which the
    table lists apart (section 2) and counts with the kind it is a limit *on*
    (`LIMIT_ON`).
    """

    BELIEF = "belief"
    """Nobody chooses it and nobody knows it exactly: a physics or engineering
    parameter, a material constant, a cost coefficient."""
    BUILD = "build"
    """Fixed before the machine exists: a size, a thickness, a material, a margin the
    designer sets."""
    OPERATING = "operating"
    """Set in operation, given the machine and the realised physics: a density, a
    temperature, a fuelling fraction, a coolant setting."""
    NUMERICS = "numerics"
    """A switch, a solver start, a pin the port restates, or a tokamak-only field the
    stellarator graph reads at zero."""
    DERIVED = "derived"
    """Computed by PROCESS's full pipeline but by no node of the stellarator graph, so
    read from the cold `DataStructure`; every one is zero here."""
    LIMIT = "limit"
    """A constraint's limit input (`LIMIT_OF`): a specification or a physics bound
    the constraint compares an output against."""


KINDS: dict[str, Kind] = {
    # -- beliefs (decision_kinds.md section 3.1: 74 physics and engineering, then the 73
    #    1990 cost-model coefficients)
    ".constraints.f_fw_rad_max": Kind.BELIEF,
    # Reclassified 2026-09-17: the table listed it as a belief because `uq.INPUTS` samples
    # it, but the allowed fraction of the critical current is a margin the designer sets,
    # read by the winding-pack sizing rule -- a build decision. Held at nominal in the
    # `build` belief table (`BUILD_LEAVES`).
    ".constraints.f_j_tf_wp_critical_max": Kind.BUILD,
    ".costs.abktflnc": Kind.BELIEF,
    ".costs.adivflnc": Kind.BELIEF,
    ".costs.cfind": Kind.BELIEF,
    ".costs.cowner": Kind.BELIEF,
    ".costs.decomf": Kind.BELIEF,
    ".costs.dintrt": Kind.BELIEF,
    ".costs.discount_rate": Kind.BELIEF,
    ".costs.dtlife": Kind.BELIEF,
    ".costs.fcap0": Kind.BELIEF,
    ".costs.fcap0cp": Kind.BELIEF,
    ".costs.fcdfuel": Kind.BELIEF,
    ".costs.fcontng": Kind.BELIEF,
    ".costs.fcr0": Kind.BELIEF,
    ".costs.fkind": Kind.BELIEF,
    ".current_drive.eta_ecrh_injector_wall_plug": Kind.BELIEF,
    ".divertor.den_div_structure": Kind.BELIEF,
    ".divertor.xpertin": Kind.BELIEF,
    ".fwbs.declblkt": Kind.BELIEF,
    ".fwbs.declfw": Kind.BELIEF,
    ".fwbs.declshld": Kind.BELIEF,
    ".fwbs.den_steel": Kind.BELIEF,
    ".fwbs.eta_coolant_pump_electric": Kind.BELIEF,
    ".fwbs.f_p_blkt_multiplication": Kind.BELIEF,
    ".fwbs.fvoldw": Kind.BELIEF,
    ".fwbs.fvolsi": Kind.BELIEF,
    ".fwbs.fvolso": Kind.BELIEF,
    ".heat_transport.eta_turbine": Kind.BELIEF,
    ".heat_transport.etatf": Kind.BELIEF,
    ".heat_transport.f_p_blkt_coolant_pump_total_heat": Kind.BELIEF,
    ".heat_transport.f_p_div_coolant_pump_total_heat": Kind.BELIEF,
    ".heat_transport.f_p_fw_coolant_pump_total_heat": Kind.BELIEF,
    ".heat_transport.f_p_shld_coolant_pump_total_heat": Kind.BELIEF,
    ".heat_transport.p_plant_electric_base": Kind.BELIEF,
    ".heat_transport.p_tritium_plant_electric_mw": Kind.BELIEF,
    ".heat_transport.pflux_plant_floor_electric": Kind.BELIEF,
    ".heat_transport.vachtmw": Kind.BELIEF,
    ".impurity_radiation.f_nd_impurity_electron_array[10]": Kind.BELIEF,
    ".impurity_radiation.f_nd_impurity_electron_array[11]": Kind.BELIEF,
    ".impurity_radiation.f_nd_impurity_electron_array[12]": Kind.BELIEF,
    ".impurity_radiation.f_nd_impurity_electron_array[13]": Kind.BELIEF,
    ".impurity_radiation.f_nd_impurity_electron_array[2]": Kind.BELIEF,
    ".impurity_radiation.f_nd_impurity_electron_array[3]": Kind.BELIEF,
    ".impurity_radiation.f_nd_impurity_electron_array[4]": Kind.BELIEF,
    ".impurity_radiation.f_nd_impurity_electron_array[5]": Kind.BELIEF,
    ".impurity_radiation.f_nd_impurity_electron_array[6]": Kind.BELIEF,
    ".impurity_radiation.f_nd_impurity_electron_array[7]": Kind.BELIEF,
    ".impurity_radiation.f_nd_impurity_electron_array[8]": Kind.BELIEF,
    ".impurity_radiation.f_nd_impurity_electron_array[9]": Kind.BELIEF,
    ".impurity_radiation.impurity_arr_zav": Kind.BELIEF,
    ".impurity_radiation.m_impurity_amu_array": Kind.BELIEF,
    ".impurity_radiation.pden_impurity_lz_nd_temp_array": Kind.BELIEF,
    ".impurity_radiation.temp_impurity_keV_array": Kind.BELIEF,
    ".physics.alphan": Kind.BELIEF,
    ".physics.alphat": Kind.BELIEF,
    ".physics.f_nd_protium_electrons": Kind.BELIEF,
    ".physics.f_p_alpha_plasma_deposited": Kind.BELIEF,
    ".physics.f_sync_reflect": Kind.BELIEF,
    ".physics.f_temp_plasma_ion_electron": Kind.BELIEF,
    ".physics.ffwal": Kind.BELIEF,
    ".physics.hfact": Kind.BELIEF,
    ".physics.tauratio": Kind.BELIEF,
    ".stellarator.bmn": Kind.BELIEF,
    ".stellarator.f_asym": Kind.BELIEF,
    ".stellarator.f_w": Kind.BELIEF,
    ".stellarator.fdivwet": Kind.BELIEF,
    ".stellarator.flpitch": Kind.BELIEF,
    ".tfcoil.dcond[0]": Kind.BELIEF,
    ".tfcoil.den_tf_coil_case": Kind.BELIEF,
    ".tfcoil.den_tf_wp_turn_insulation": Kind.BELIEF,
    ".tfcoil.rho_tf_bus": Kind.BELIEF,
    ".vacuum.outgrat_fw": Kind.BELIEF,
    "^stated.tfcoil.eff_tf_cryo": Kind.BELIEF,
    ".costs.UCAD": Kind.BELIEF,
    ".costs.UCAF": Kind.BELIEF,
    ".costs.UCAHTS": Kind.BELIEF,
    ".costs.UCAP": Kind.BELIEF,
    ".costs.UCBPMP": Kind.BELIEF,
    ".costs.UCCO": Kind.BELIEF,
    ".costs.UCCPMP": Kind.BELIEF,
    ".costs.UCCR": Kind.BELIEF,
    ".costs.UCDGEN": Kind.BELIEF,
    ".costs.UCDTC": Kind.BELIEF,
    ".costs.UCDUCT": Kind.BELIEF,
    ".costs.UCEL": Kind.BELIEF,
    ".costs.UCFPR": Kind.BELIEF,
    ".costs.UCFWA": Kind.BELIEF,
    ".costs.UCFWPS": Kind.BELIEF,
    ".costs.UCFWS": Kind.BELIEF,
    ".costs.UCGSS": Kind.BELIEF,
    ".costs.UCINT": Kind.BELIEF,
    ".costs.UCLV": Kind.BELIEF,
    ".costs.UCMB": Kind.BELIEF,
    ".costs.UCNBV": Kind.BELIEF,
    ".costs.UCPHX": Kind.BELIEF,
    ".costs.UCPP": Kind.BELIEF,
    ".costs.UCSH": Kind.BELIEF,
    ".costs.UCSWYD": Kind.BELIEF,
    ".costs.UCTFDR": Kind.BELIEF,
    ".costs.UCTFGR": Kind.BELIEF,
    ".costs.UCTFIC": Kind.BELIEF,
    ".costs.UCTPMP": Kind.BELIEF,
    ".costs.UCTR": Kind.BELIEF,
    ".costs.UCVALV": Kind.BELIEF,
    ".costs.UCVDSH": Kind.BELIEF,
    ".costs.UCVIAC": Kind.BELIEF,
    ".costs.UCWS": Kind.BELIEF,
    ".costs.cconfix": Kind.BELIEF,
    ".costs.cconshtf": Kind.BELIEF,
    ".costs.cland": Kind.BELIEF,
    ".costs.csi": Kind.BELIEF,
    ".costs.cturbb": Kind.BELIEF,
    ".costs.ucblbe": Kind.BELIEF,
    ".costs.ucblli2o": Kind.BELIEF,
    ".costs.ucblss": Kind.BELIEF,
    ".costs.ucblvd": Kind.BELIEF,
    ".costs.ucbus": Kind.BELIEF,
    ".costs.uccase": Kind.BELIEF,
    ".costs.uccry": Kind.BELIEF,
    ".costs.uccryo": Kind.BELIEF,
    ".costs.uccu": Kind.BELIEF,
    ".costs.ucdiv": Kind.BELIEF,
    ".costs.ucech": Kind.BELIEF,
    ".costs.ucf1": Kind.BELIEF,
    ".costs.ucfuel": Kind.BELIEF,
    ".costs.uche3": Kind.BELIEF,
    ".costs.uchrs": Kind.BELIEF,
    ".costs.uchts": Kind.BELIEF,
    ".costs.uciac": Kind.BELIEF,
    ".costs.ucich": Kind.BELIEF,
    ".costs.uclh": Kind.BELIEF,
    ".costs.ucme": Kind.BELIEF,
    ".costs.ucmisc": Kind.BELIEF,
    ".costs.ucnbi": Kind.BELIEF,
    ".costs.ucoam": Kind.BELIEF,
    ".costs.ucpens": Kind.BELIEF,
    ".costs.ucrb": Kind.BELIEF,
    ".costs.ucsc": Kind.BELIEF,
    ".costs.ucshld": Kind.BELIEF,
    ".costs.uctfbr": Kind.BELIEF,
    ".costs.uctfbus": Kind.BELIEF,
    ".costs.uctfps": Kind.BELIEF,
    ".costs.uctfsw": Kind.BELIEF,
    ".costs.ucturb": Kind.BELIEF,
    ".costs.ucwindtf": Kind.BELIEF,
    ".costs.ucwst": Kind.BELIEF,

    # -- build decisions (section 3.2)
    ".build.dr_blkt_inboard": Kind.BUILD,
    ".build.dr_blkt_outboard": Kind.BUILD,
    ".build.dr_cryostat": Kind.BUILD,
    ".build.dr_fw_plasma_gap_inboard": Kind.BUILD,
    ".build.dr_fw_plasma_gap_outboard": Kind.BUILD,
    ".build.dr_shld_blkt_gap": Kind.BUILD,
    ".build.dr_shld_inboard": Kind.BUILD,
    ".build.dr_shld_outboard": Kind.BUILD,
    ".build.dr_shld_vv_gap_inboard": Kind.BUILD,
    ".build.dr_vv_inboard": Kind.BUILD,
    ".build.dr_vv_outboard": Kind.BUILD,
    ".build.dz_shld_vv_gap": Kind.BUILD,
    ".build.dz_vv_lower": Kind.BUILD,
    ".build.dz_vv_upper": Kind.BUILD,
    ".build.gapomin": Kind.BUILD,
    ".buildings.admv": Kind.BUILD,
    ".buildings.clh2": Kind.BUILD,
    ".buildings.conv": Kind.BUILD,
    ".buildings.dz_tf_cryostat": Kind.BUILD,
    ".buildings.fndt": Kind.BUILD,
    ".buildings.hccl": Kind.BUILD,
    ".buildings.hcwt": Kind.BUILD,
    ".buildings.mbvfac": Kind.BUILD,
    ".buildings.pfbldgm3": Kind.BUILD,
    ".buildings.pibv": Kind.BUILD,
    ".buildings.rbrt": Kind.BUILD,
    ".buildings.rbvfac": Kind.BUILD,
    ".buildings.rbwt": Kind.BUILD,
    ".buildings.row": Kind.BUILD,
    ".buildings.rxcl": Kind.BUILD,
    ".buildings.shmf": Kind.BUILD,
    ".buildings.shov": Kind.BUILD,
    ".buildings.stcl": Kind.BUILD,
    ".buildings.trcl": Kind.BUILD,
    ".buildings.triv": Kind.BUILD,
    ".buildings.wgt": Kind.BUILD,
    ".buildings.wgt2": Kind.BUILD,
    ".buildings.wsvfac": Kind.BUILD,
    # Confirmed build 2026-09-17 (a plant-life choice) although `uq.INPUTS` samples it.
    ".costs.life_plant": Kind.BUILD,
    ".divertor.anginc": Kind.BUILD,
    ".divertor.dx_div_plate": Kind.BUILD,
    ".divertor.f_vol_div_coolant": Kind.BUILD,
    ".divertor.n_divertors": Kind.BUILD,
    # Confirmed build 2026-09-17: sampled by `uq.INPUTS` as manufacturing scatter, but a
    # build decision; held at nominal in the `build` belief table (`BUILD_LEAVES`).
    ".fwbs.dr_fw_wall": Kind.BUILD,
    ".fwbs.dr_pf_cryostat": Kind.BUILD,
    ".fwbs.f_a_fw_outboard_hcd": Kind.BUILD,
    ".fwbs.fblbe": Kind.BUILD,
    ".fwbs.fblli2o": Kind.BUILD,
    ".fwbs.fblss": Kind.BUILD,
    ".fwbs.fblvd": Kind.BUILD,
    # Confirmed build 2026-09-17: sampled as scatter, held at nominal (`BUILD_LEAVES`).
    ".fwbs.fhole": Kind.BUILD,
    ".fwbs.i_blkt_coolant_type": Kind.BUILD,
    ".fwbs.radius_fw_channel": Kind.BUILD,
    ".fwbs.vfshld": Kind.BUILD,
    ".physics.b_plasma_toroidal_on_axis": Kind.BUILD,
    ".physics.kappa": Kind.BUILD,
    ".physics.rmajor": Kind.BUILD,
    ".stellarator.f_st_coil_aspect": Kind.BUILD,
    ".stellarator.iotabar": Kind.BUILD,
    ".stellarator.m_res": Kind.BUILD,
    ".stellarator.max_gyrotron_frequency": Kind.BUILD,
    ".stellarator.n_res": Kind.BUILD,
    ".stellarator.shear": Kind.BUILD,
    ".tfcoil.dr_tf_nose_case": Kind.BUILD,
    ".tfcoil.dx_tf_turn_general": Kind.BUILD,
    ".tfcoil.dx_tf_turn_insulation": Kind.BUILD,
    ".tfcoil.dx_tf_turn_steel": Kind.BUILD,
    # Confirmed build 2026-09-17: sampled as scatter, held at nominal (`BUILD_LEAVES`).
    ".tfcoil.dx_tf_wp_insulation": Kind.BUILD,
    ".tfcoil.f_a_tf_turn_cable_copper": Kind.BUILD,
    ".tfcoil.f_a_tf_turn_cable_space_extra_void": Kind.BUILD,
    ".tfcoil.i_tf_sc_mat": Kind.BUILD,
    ".tfcoil.i_tf_sup": Kind.BUILD,
    ".tfcoil.t_tf_quench_detection": Kind.BUILD,
    ".tfcoil.t_tf_superconductor_quench": Kind.BUILD,
    ".tfcoil.tmargmin": Kind.BUILD,
    "^stated.times.t_plant_pulse_burn": Kind.BUILD,

    # -- operating variables (section 3.3)
    ".costs.f_t_plant_available": Kind.OPERATING,
    ".current_drive.p_hcd_primary_extra_heat_mw": Kind.OPERATING,
    # Confirmed operating 2026-09-17 (a detachment target) although `uq.INPUTS` samples it.
    ".divertor.tdiv": Kind.OPERATING,
    ".physics.f_nd_alpha_thermal_electron": Kind.OPERATING,
    ".physics.f_plasma_fuel_deuterium": Kind.OPERATING,
    ".physics.f_plasma_fuel_helium3": Kind.OPERATING,
    ".physics.f_plasma_fuel_tritium": Kind.OPERATING,
    ".physics.nd_plasma_electrons_vol_avg": Kind.OPERATING,
    ".physics.temp_plasma_electron_vol_avg_kev": Kind.OPERATING,
    # Confirmed operating 2026-09-17 (impurity seeding sets it) although sampled.
    ".stellarator.f_rad": Kind.OPERATING,
    ".tfcoil.temp_tf_cryo": Kind.OPERATING,
    ".tfcoil.tftmp": Kind.OPERATING,
    ".times.t_plant_pulse_dwell": Kind.OPERATING,
    ".times.t_plant_pulse_fusion_ramp": Kind.OPERATING,
    ".vacuum.pres_div_chamber_burn": Kind.OPERATING,
    ".vacuum.pres_vv_chamber_base": Kind.OPERATING,

    # -- numerics, switches, pins and inert tokamak-only reads (section 3.4). A solver's
    # start (`^guess.*`) is numerics by kind, whichever variable the scheme cut
    # (`stages.leaves`), so none is listed.
    ".costs.ifueltyp": Kind.NUMERICS,
    ".costs.ireactor": Kind.NUMERICS,
    ".costs.lsa": Kind.NUMERICS,
    ".current_drive.e_beam_kev": Kind.NUMERICS,
    ".current_drive.f_beam_tritium": Kind.NUMERICS,
    ".current_drive.i_hcd_primary": Kind.NUMERICS,
    ".current_drive.p_beam_injected_mw": Kind.NUMERICS,
    ".current_drive.p_beam_orbit_loss_mw": Kind.NUMERICS,
    ".current_drive.p_beam_shine_through_mw": Kind.NUMERICS,
    ".current_drive.p_hcd_beam_injected_total_mw": Kind.NUMERICS,
    ".current_drive.p_hcd_lowhyb_injected_total_mw": Kind.NUMERICS,
    ".fwbs.f_nuc_pow_bz_liq": Kind.NUMERICS,
    ".fwbs.outlet_temp_liq": Kind.NUMERICS,
    ".heat_transport.i_shld_primary_heat": Kind.NUMERICS,
    ".heat_transport.ipowerflow": Kind.NUMERICS,
    ".heat_transport.p_blkt_breeder_pump_mw": Kind.NUMERICS,
    ".ife.ife": Kind.NUMERICS,
    ".impurity_radiation.f_p_plasma_core_rad_reduction": Kind.NUMERICS,
    ".impurity_radiation.radius_plasma_core_norm": Kind.NUMERICS,
    ".pf_coil.m_pf_coil_max": Kind.NUMERICS,
    ".pf_coil.p_pf_electric_supplies_mw": Kind.NUMERICS,
    ".pf_coil.r_pf_coil_outer_max": Kind.NUMERICS,
    ".pf_power.ensxpfm": Kind.NUMERICS,
    ".pf_power.srcktpm": Kind.NUMERICS,
    ".physics.alphaj": Kind.NUMERICS,
    ".physics.beta_beam": Kind.NUMERICS,
    ".physics.burnup_in": Kind.NUMERICS,
    ".physics.dlamie": Kind.NUMERICS,
    ".physics.itart": Kind.NUMERICS,
    ".physics.p_beam_alpha_mw": Kind.NUMERICS,
    ".physics.p_plasma_ohmic_mw": Kind.NUMERICS,
    ".physics.pden_plasma_ohmic_mw": Kind.NUMERICS,
    ".physics.plasma_current": Kind.NUMERICS,
    ".tfcoil.a_tf_wp_coolant_channels": Kind.NUMERICS,
    ".tfcoil.temp_cp_coolant_inlet": Kind.NUMERICS,
    ".vacuum.ceff_i": Kind.NUMERICS,
    ".vacuum.i_vac_pump_dwell": Kind.NUMERICS,
    ".vacuum.i_vacuum_pump_type": Kind.NUMERICS,
    ".vacuum.l1": Kind.NUMERICS,
    ".vacuum.l2": Kind.NUMERICS,
    ".vacuum.l3": Kind.NUMERICS,
    ".vacuum.xmult_i": Kind.NUMERICS,
    "^stated.build.dr_cs": Kind.NUMERICS,
    "^stated.build.dr_cs_tf_gap": Kind.NUMERICS,
    "^stated.buildings.esbldgm3": Kind.NUMERICS,
    "^stated.costs.c2253": Kind.NUMERICS,
    "^stated.physics.nd_plasma_pedestal_electron": Kind.NUMERICS,
    "^stated.physics.nd_plasma_separatrix_electron": Kind.NUMERICS,
    "^stated.physics.radius_plasma_pedestal_density_norm": Kind.NUMERICS,
    "^stated.physics.radius_plasma_pedestal_temp_norm": Kind.NUMERICS,
    "^stated.physics.tbeta": Kind.NUMERICS,
    "^stated.physics.temp_plasma_pedestal_kev": Kind.NUMERICS,
    "^stated.physics.temp_plasma_separatrix_kev": Kind.NUMERICS,
    "^stated.times.t_plant_pulse_coil_precharge": Kind.NUMERICS,
    "^stated.times.t_plant_pulse_plasma_current_ramp_down": Kind.NUMERICS,
    "^stated.times.t_plant_pulse_plasma_current_ramp_up": Kind.NUMERICS,

    # -- derived quantities read from outside: missing producers, all zero (section 3.5)
    ".build.r_tf_inboard_mid": Kind.DERIVED,
    ".costs.c2214": Kind.DERIVED,
    ".costs.c2222": Kind.DERIVED,
    ".costs.c2252": Kind.DERIVED,
    ".costs.cplife": Kind.DERIVED,
    ".fwbs.p_cp_shield_nuclear_heat_mw": Kind.DERIVED,
    ".fwbs.p_div_rad_total_mw": Kind.DERIVED,
    ".heat_transport.peakmva": Kind.DERIVED,
    ".physics.beta_thermal_vol_avg": Kind.DERIVED,
    ".physics.beta_toroidal_vol_avg": Kind.DERIVED,
    ".tfcoil.m_tf_bus": Kind.DERIVED,
    ".tfcoil.res_tf_leg": Kind.DERIVED,
    ".tfcoil.tfcmw": Kind.DERIVED,

    # -- constraint limits (section 2), the ten boundary inputs of the problem graph the
    #    constraint nodes read and no model does; `LIMIT_ON` says what each is a limit on
    ".constraints.p_plant_electric_net_required_mw": Kind.LIMIT,
    ".physics.beta_vol_avg_max": Kind.LIMIT,
    ".constraints.pflux_fw_neutron_max_mw": Kind.LIMIT,
    ".constraints.f_p_plasma_separatrix_rad_max": Kind.LIMIT,
    ".divertor.pflux_div_heat_load_max_mw": Kind.LIMIT,
    ".constraints.pflux_fw_rad_max": Kind.LIMIT,
    ".constraints.f_t_alpha_energy_confinement_min": Kind.LIMIT,
    ".tfcoil.sig_tf_wp_max": Kind.LIMIT,
    ".tfcoil.v_tf_coil_dump_quench_max_kv": Kind.LIMIT,
    ".tfcoil.max_vv_stress": Kind.LIMIT,
}
"""Every boundary input of `stellarator_helias`'s problem graph (324: the driven MDA
graph's 311 plus the 10 constraint limits and 3 quantities the constraint nodes read
that no stellarator node produces) -> its kind. Sections 3.1-3.5 of
`decision_kinds.md` in their order, the eight `ixc` places among them, then section 2's
limits; the section 2 spellings are the bare names of that table resolved to the area
PROCESS declares the input in (`vocabulary.input_variables`), as the constraint nodes
resolve them. One row moved from the table's classification, marked where it stands:
`f_j_tf_wp_critical_max`, belief -> build. Per kind: 146 belief (73 of them 1990
cost-model coefficients), 77 build, 16 operating, 62 numerics, 13 derived, 10 limits."""


DESIGN_KINDS: dict[int, Kind] = {
    2: Kind.BUILD,  # b_plasma_toroidal_on_axis: sets the coil current and the magnet
    3: Kind.BUILD,  # rmajor: the machine size
    4: Kind.OPERATING,  # temp_plasma_electron_vol_avg_kev: heating and fuelling on the day
    6: Kind.OPERATING,  # nd_plasma_electrons_vol_avg: fuelling; closes the power balance
    10: Kind.BELIEF,  # hfact: a confinement multiplier is not chosen
    109: Kind.OPERATING,  # f_nd_alpha_thermal_electron: fuelling / He exhaust
    59: Kind.BUILD,  # f_a_tf_turn_cable_copper: conductor design
    56: Kind.BUILD,  # t_tf_superconductor_quench: dump-resistor / protection design
}
"""The eight `ixc` iteration variables of the reference run, in the file's order
(`decision_kinds.md` section 1) -> kind: 4 build, 3 operating, 1 belief. The place each
id addresses is `vocabulary.iteration_variables.ITERATION_VARIABLES`'s."""

DESIGN_PLACES: dict[int, str] = {
    2: ".physics.b_plasma_toroidal_on_axis",
    3: ".physics.rmajor",
    4: ".physics.temp_plasma_electron_vol_avg_kev",
    6: ".physics.nd_plasma_electrons_vol_avg",
    10: ".physics.hfact",
    109: ".physics.f_nd_alpha_thermal_electron",
    59: ".tfcoil.f_a_tf_turn_cable_copper",
    56: ".tfcoil.t_tf_superconductor_quench",
}
"""`ixc` id -> the place it owns, the same rows."""

LIMIT_OF: dict[int, str] = {
    16: ".constraints.p_plant_electric_net_required_mw",
    24: ".physics.beta_vol_avg_max",
    8: ".constraints.pflux_fw_neutron_max_mw",
    17: ".constraints.f_p_plasma_separatrix_rad_max",
    18: ".divertor.pflux_div_heat_load_max_mw",
    67: ".constraints.pflux_fw_rad_max",
    62: ".constraints.f_t_alpha_energy_confinement_min",
    32: ".tfcoil.sig_tf_wp_max",
    34: ".tfcoil.v_tf_coil_dump_quench_max_kv",
    65: ".tfcoil.max_vv_stress",
}
"""`icc` id -> the limit input it compares against, for the ten of the fourteen active
constraints whose limit is a boundary input (`decision_kinds.md` section 2). The other
four have none: c2 (the power balance) and c82 / c83 (build consistency) compare two
outputs, c35's limit (`j_tf_wp_quench_heat_max`) is computed."""

LIMIT_ON: dict[int, Kind] = {
    16: Kind.BUILD,  # plant spec: the required net electric power
    24: Kind.BELIEF,  # the stellarator beta limit
    8: Kind.BUILD,  # materials spec: neutron wall load
    17: Kind.NUMERICS,  # a trivial bound (1.0) on the radiation fraction
    18: Kind.BUILD,  # target spec: divertor heat load
    67: Kind.BUILD,  # materials spec: radiation wall load
    62: Kind.BELIEF,  # He exhaust: the minimum tau_He* / tau_E
    32: Kind.BUILD,  # allowable: winding-pack stress
    34: Kind.BUILD,  # protection spec: dump voltage
    65: Kind.BUILD,  # allowable: vacuum-vessel stress at quench
}
"""`icc` id -> the kind its limit is a limit *on*, the same rows: 7 build
specifications, 2 physics beliefs, 1 trivial bound."""


CLAIMED_BUILD_OUTPUTS: dict[str, str] = {
    # 1a. TF coil / magnet
    ".physics.aspect": "1a",
    ".tfcoil.n_tf_coils": "1a",
    ".stellarator.r_coil_major": "1a",
    ".stellarator.r_coil_minor": "1a",
    ".stellarator.coilcurrent": "1a",
    ".stellarator.f_st_i_total": "1a",
    ".stellarator.wp_width_r_min": "1a",
    ".tfcoil.dr_tf_wp_with_insulation": "1a",
    ".tfcoil.j_tf_wp": "1a",
    ".tfcoil.dx_tf_wp_primary_toroidal": "1a",
    ".tfcoil.dx_tf_wp_secondary_toroidal": "1a",
    ".tfcoil.dr_tf_plasma_case": "1a",
    ".tfcoil.dx_tf_side_case_min": "1a",
    # 1b. Radial / vertical build, first wall, blanket, shield, vessel, divertor, vacuum
    ".build.dr_fw_inboard": "1b",
    ".build.dr_fw_outboard": "1b",
    ".build.dr_tf_outboard": "1b",
    ".build.dr_shld_vv_gap_outboard": "1b",
    ".fwbs.life_fw_fpy": "1b",
    ".vacuum.n_vac_pumps_high": "1b",
    ".vacuum.dia_vv_vacuum_ducts": "1b",
    ".vacuum.d_duct": "1b",
    # 1c. Plant equipment count
    ".heat_transport.n_primary_heat_exchangers": "1c",
    # 1d. Initialisation / pinning
    ".build.dr_cs": "1d",
    ".build.dr_cs_tf_gap": "1d",
    ".buildings.esbldgm3": "1d",
    # 4. The two component lifetimes the borderline cases class as build with
    #    `life_fw_fpy`; counted among section 3's fourteen sizing choices, tabled nowhere
    ".fwbs.life_blkt_fpy": "4",
    ".costs.life_div_fpy": "4",
}
"""Every owned place `output_kinds.md` flags as a build decision closed by a rule (a
definition or a sizing choice) -> the section that flags it. Sections 1a-1d are the
tables; the two lifetimes come from section 4, which classes them build without a
table row. A cell that named two places (`r_coil_major`, `r_coil_minor`) is two
entries, spelled in full; the node a cell named beside a place is not one."""


class Decision(enum.Enum):
    """What was decided about a sizing choice for the two-stage formulation."""

    LIFT = "lift"
    """Make it a first-stage design variable, and the rule's equality an inequality
    enforced as a chance constraint."""
    RECOURSE = "recourse"
    """Keep it per sample: it is an operating decision, made with the machine running."""
    QUANTILE = "quantile"
    """Size it once at the alpha-quantile of the sampled quantity its rule reads."""
    NONE = "none"
    """Leave the rule as it is: it reads only build inputs, so it is first-stage
    already."""


SIZING_CHOICES: dict[str, Decision] = {
    # Winding pack: its rule reads the coil temperature and the critical-current
    # fraction, so the coil is re-sized per sample. With `f_j_tf_wp_critical_max` a build
    # margin, only the coil temperature reaches it; lift the pack thickness to an outer
    # build variable (PROCESS ixc 140) and the rule to `j_tf_wp <= f j_c` (icc 33), a
    # chance constraint, the safe side a wider pack. icc 35 stays as it is.
    ".stellarator.wp_width_r_min": Decision.LIFT,
    ".tfcoil.dr_tf_wp_with_insulation": Decision.LIFT,
    ".tfcoil.j_tf_wp": Decision.LIFT,
    # Component lifetimes: replacement happens when the fluence is reached, a decision
    # made with the machine running -- operating, per sample. Overrides
    # `output_kinds.md` section 4's move of the blanket and divertor lifetimes to build.
    ".fwbs.life_fw_fpy": Decision.RECOURSE,
    ".fwbs.life_blkt_fpy": Decision.RECOURSE,
    ".costs.life_div_fpy": Decision.RECOURSE,
    # Vacuum pumping and the heat-exchanger count: one first-stage number each, sized at
    # the alpha-quantile of the fusion / primary power the rule reads; the effect is
    # cost-only.
    ".vacuum.n_vac_pumps_high": Decision.QUANTILE,
    ".vacuum.dia_vv_vacuum_ducts": Decision.QUANTILE,
    ".vacuum.d_duct": Decision.QUANTILE,
    ".heat_transport.n_primary_heat_exchangers": Decision.QUANTILE,
    # Placeholders (case thicknesses equal to the nose case, outboard TF leg equal to
    # the inboard, shield-vessel gap at its minimum): they read only build inputs, so
    # they are first-stage already. Switching on PROCESS ixc 172 / 75 / 31 would move
    # the deterministic optimum, which is a separate question.
    ".tfcoil.dr_tf_plasma_case": Decision.NONE,
    ".tfcoil.dx_tf_side_case_min": Decision.NONE,
    ".build.dr_tf_outboard": Decision.NONE,
    ".build.dr_shld_vv_gap_outboard": Decision.NONE,
}
"""The fourteen sizing choices of `output_kinds.md` section 3 -- an inequality a rule
holds at zero slack -- and the decision on each, agreed 2026-09-17: one lift, one
reclassification, one quantile rule, and four left alone."""


@dataclass(frozen=True)
class Belief:
    """One uncertain boundary input and its distribution.

    `kind`: `uniform` -- uniform on [a, b]; `relative` -- uniform in
    nominal x [1 - a, 1 + a]; `lognormal` -- nominal x exp(a z), z standard normal;
    `factor` -- nominal x uniform[a, b], an array scaled as one. `path` is the boundary
    input's spelling, or `dummy`, which maps to nothing. The same fields as
    `paper_tests/uq.Input`; how a quantile becomes a value is the architecture's.
    """

    path: str
    kind: str
    a: float = 0.0
    b: float = 0.0
    note: str = ""


BELIEFS: tuple[Belief, ...] = (
    Belief(".physics.hfact", "lognormal", 0.10, 0.0, "confinement multiplier; PROCESS's closure variable, here a belief"),
    Belief(".physics.alphan", "relative", 0.20, 0.0, "density profile exponent"),
    Belief(".physics.alphat", "relative", 0.20, 0.0, "temperature profile exponent"),
    Belief(".physics.f_p_alpha_plasma_deposited", "uniform", 0.90, 0.99, "alpha power deposited in the plasma"),
    Belief(".physics.f_temp_plasma_ion_electron", "uniform", 0.85, 1.0, "$T_i / T_e$"),
    Belief(".physics.f_sync_reflect", "uniform", 0.5, 0.7, "synchrotron wall reflectivity"),
    Belief(".impurity_radiation.f_nd_impurity_electron_array[13]", "relative", 0.20, 0.0, "tungsten fraction (index 13, the only seeded non-alpha impurity) (relative +-20 %)"),
    Belief(".stellarator.bmn", "relative", 0.20, 0.0, "residual field ripple (relative +-20 %)"),
    Belief(".stellarator.f_asym", "uniform", 1.0, 1.2, "divertor heat-load asymmetry"),
    Belief(".stellarator.f_rad", "uniform", 0.75, 0.95, "radiated fraction in the SOL / divertor"),
    Belief(".fwbs.f_p_blkt_multiplication", "uniform", 1.25, 1.45, "blanket energy multiplication"),
    Belief(".fwbs.declfw", "relative", 0.20, 0.0, "first-wall neutron decay length"),
    Belief(".fwbs.declblkt", "relative", 0.20, 0.0, "blanket neutron decay length"),
    Belief(".fwbs.fhole", "uniform", 0.0, 0.05, "first-wall hole fraction"),
    Belief(".fwbs.dr_fw_wall", "uniform", 0.002, 0.004, "first-wall armour thickness [m]"),
    Belief(".tfcoil.temp_tf_cryo", "uniform", 4.2, 4.8, "coil temperature [K]"),
    Belief(".tfcoil.dx_tf_wp_insulation", "uniform", 0.008, 0.012, "winding-pack insulation [m]"),
    Belief(".constraints.f_j_tf_wp_critical_max", "uniform", 0.7, 0.9, "allowed fraction of the critical current"),
    Belief(".heat_transport.eta_turbine", "uniform", 0.33, 0.42, "thermal-to-electric efficiency"),
    Belief(".current_drive.eta_ecrh_injector_wall_plug", "uniform", 0.4, 0.6, "ECRH wall-plug efficiency"),
    Belief(".heat_transport.f_p_fw_coolant_pump_total_heat", "relative", 0.30, 0.0, "FW pumping power fraction"),
    Belief(".divertor.tdiv", "uniform", 3.0, 8.0, "divertor plasma temperature [eV]"),
    Belief(".costs.f_t_plant_available", "uniform", 0.65, 0.85, "availability"),
    Belief(".costs.life_plant", "uniform", 30.0, 50.0, "plant life [y]"),
    Belief(".costs.discount_rate", "uniform", 0.04, 0.08, "discount rate"),
    Belief(".costs.ucsc", "factor", 0.7, 1.5, "superconductor unit costs, the (9,) array scaled together"),
    Belief("dummy", "uniform", 0.0, 1.0, "read by nothing: the estimators' noise floor"),
)
"""The belief table, canonical: `paper_tests/ouu.belief_table("new", 0.10)` -- `uq.INPUTS`
with `hfact` lognormal(0.10) rather than 0.15 and the tungsten fraction and field ripple
uniform within 20 % rather than lognormal(ln 2 / 2). The `dummy` row, read by nothing,
is the estimators' noise floor and stays. Nine of the paths are not beliefs by `KINDS`
(`ECONOMIC`'s `life_plant`, `BUILD_LEAVES`, `tdiv`, `f_rad`, `temp_tf_cryo`,
`f_t_plant_available`): sampled as scatter, or as a belief about operation."""

ECONOMIC: tuple[str, ...] = (
    ".costs.f_t_plant_available",
    ".costs.life_plant",
    ".costs.discount_rate",
    ".costs.ucsc",
)
"""The four economic rows of `BELIEFS`, held at their nominal when only the physics is
uncertain (`ouu.ECONOMIC`)."""

BUILD_LEAVES: tuple[str, ...] = (
    ".fwbs.dr_fw_wall",
    ".tfcoil.dx_tf_wp_insulation",
    ".fwbs.fhole",
    ".constraints.f_j_tf_wp_critical_max",
)
"""The sampled rows the `build` belief table holds at their nominal (`ouu.BUILD_LEAVES`):
the three manufacturing scatters and the designer's current margin. The stage check
found these four to be the sampled leaves behind every coil, radial-build and first-wall
place; with them fixed, every sample has one build, and what still varies is physics,
operation, cost, and the four accepted second-stage outputs: `life_fw_fpy` (recourse)
and `n_vac_pumps_high` / `dia_vv_vacuum_ducts` / `n_primary_heat_exchangers`
(cost-only)."""


C16 = "^cond.constraints.c16"
"""The net electric power condition: an equality in the file, a plant requirement."""
TE = ".physics.temp_plasma_electron_vol_avg_kev"
"""The one operating knob with no equation to close once c2 and c16 are taken."""

PAIRINGS: dict[str, dict[str, str]] = {
    "one": {"^cond.constraints.c2": ".physics.nd_plasma_electrons_vol_avg"},
    "two": {
        "^cond.constraints.c2": ".physics.nd_plasma_electrons_vol_avg",
        "^cond.constraints.c16": ".physics.f_nd_alpha_thermal_electron",
    },
    "te": {
        "^cond.constraints.c2": ".physics.nd_plasma_electrons_vol_avg",
        C16: TE,
    },
}
"""Which equalities the MDA closes per sample, and by what (`ouu.PAIRINGS`): `one` is
the power balance by the density; `two` adds the net electric power closed by the
thermal alpha fraction, `te` by the electron temperature. `one` is the settled form:
closing c16 by the alpha fraction needs a negative fraction in 42 % of samples at the
deterministic design, so c16 is a chance constraint instead, and the temperature and
the alpha fraction are shared operating set-points."""

HISTORIC_PAIRING: dict[str, str] = {
    "^cond.constraints.c2": ".physics.hfact",
    "^cond.constraints.c16": ".physics.f_nd_alpha_thermal_electron",
}
"""The first session's choice (`close_conditions.PAIRINGS`): the power balance closed
by `hfact`, the smallest cycle (three nodes), and c16 by the alpha fraction.
Superseded -- a confinement multiplier is a belief, not a knob, and once it is sampled
the density is what closes the balance (`PAIRINGS["one"]`)."""
