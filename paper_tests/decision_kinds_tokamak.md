# Decision kinds in the PROCESS large-tokamak reference run

`tests/regression/input_files/large_tokamak_nof.IN.DAT` (the generic EU-DEMO-shaped
tokamak: `istell = 0`, single null, IPB98(y,2) confinement, **driven** -- `i_plasma_ignited
= 0`, 75 MW of ECRH heating plus current drive for a 40 % non-inductive fraction --
**pulsed** (`i_pulsed_plant = 1`, `t_burn_min = 7200 s`), 1990 cost model, figure of merit
1 = minimise the major radius), read through the cottax port. The tokamak counterpart of
`decision_kinds.md` (the stellarator's), listed the same way, programmatically:
`indat.graph_for(machine)` (245 nodes, 390 boundary inputs, 3 problems the models declare
themselves) -> `mda.cut_graph` under `mda.SCHEME` (248 nodes: three cycles opened by three
`^mda.*` fixed points -- the winding-pack toroidal width, the fuel / total ion densities,
the CS flux swing with the burn time -- 222 components) -> `mdf.mdf_graph` (the 26 `icc`
condition nodes and the objective inserted: **275 nodes, 412 boundary inputs**). The
`Optimise` PROCESS solves owns the 20 `ixc` places and reads `.numerics.objf` and
`.constraints.c<n>` for the 26 `icc`, 3 of them equalities. Values are the cold
`DataStructure`; converged values are PROCESS's own (`common.process_reference`, VMCON, 8
iterations, `objf = 1.6 = 0.2 x 8 m`).

Kinds as in the stellarator census: **belief**, **build decision**, **operating variable**,
**numerics / switch**, **derived / output**, and a constraint *limit* listed apart in section 2.
The sort is `functional_process/configurations/kinds_tokamak.py`: 244 of the 412 spellings
are shared with the stellarator graph and keep its kind (one exception below), the other 168
were sorted here.

Headline: **412 boundary inputs = 180 belief (97 of them cost-model and financing
coefficients, 83 physics / engineering) + 132 build decisions + 24 operating variables + 46
numerics / switches + 11 derived-but-read-from-outside (missing producers, all zero) + 19
constraint limits.** Of the 20 `ixc`: 12 build, 7 operating, 1 belief. Two places the study
adds to the graph are not in the file's boundary and get their kind in
`architectures/driven.py`: the installed H&CD power (build) and the beta-limit factor (belief).

Three things the tokamak file says that the stellarator's did not, and that shape the
split:

- **PROCESS's design demands the best confinement the file allows.** `hfact` (ixc 10) closes
  the power balance and its converged value is **1.2, the upper bound**. The 75 MW of heating
  is the file's number, not a solved one; what the balance solves for is how good the
  confinement must be, and the answer is "as good as permitted".
- **The figure of merit sits on its bound.** `rmajor` (ixc 3) is bounded [8, 9] and the
  optimum is 8.000: any feasible point of the 8 m machine is optimal, so the *operating*
  point PROCESS reports (T_e 12.65 keV, n at 1.2 x Greenwald, f_He 0.0857, Xe 6.2e-4) is one
  feasible point, not a cost optimum. Section 4 and the study's report say what that does to
  the deterministic comparison.
- **Two numbers PROCESS conflates.** The cost account `c2231 = ucech x
  p_hcd_ecrh_injected_total_mw` (`process/models/costs/costs.py:1866-1880`) bills the heating
  power *used* in the burn, and `c30` caps the used power by `p_hcd_injected_max = 200 MW`, an
  input nothing costs. The installed power -- what is bought -- is not a quantity of the
  model; the study adds it (section 4).

## 1. The `ixc` iteration variables

Bounds are the file's `boundl` / `boundu` where set, else `ITERATION_VARIABLES`' defaults
(marked *default*). Start / converged are PROCESS's own run.

| ixc | place | bounds | start -> converged | kind | why |
|---|---|---|---|---|---|
| 2 | `.physics.b_plasma_toroidal_on_axis` | [0.01, 30] *default* | 5.7 -> 4.956 | **build** | sets the TF coil current and the whole magnet; cannot change after the coils are wound |
| 3 | `.physics.rmajor` | [8, 9] | 8 -> 8 | **build** | the machine size and the figure of merit -- at its lower bound, so PROCESS's answer is *any* feasible point of the 8 m machine |
| 4 | `.physics.temp_plasma_electron_vol_avg_kev` | [5.5055, 100] | 12 -> 12.65 | **operating** | set by heating and fuelling on the day; a recourse knob |
| 5 | `.physics.beta_total_vol_avg` | [0.001, 1] *default* | 0.03 -> 0.03888 | **operating** | PROCESS's copy of a computed quantity (c1 says beta = beta(n, T, B)): the iteration variable is a cut, closed per world by a two-node root find |
| 6 | `.physics.nd_plasma_electrons_vol_avg` | [2e+19, 1e+21] *default* | 7.5e+19 -> 8.108e+19 | **operating** | fuelling; a recourse knob, bounded by Greenwald (c5, at 1.2 x n_GW in the converged design) |
| 10 | `.physics.hfact` | [0.1, 1.2] | 1.1 -> 1.2 | **belief** | a confinement multiplier is not chosen; PROCESS uses it as the closure knob for c2 and its answer sits on the upper bound 1.2 -- the design demands the best confinement the file allows |
| 16 | `.build.dr_cs` | [0.3, 10] | 0.5 -> 0.5908 | **build** | the central solenoid's radial thickness |
| 18 | `.physics.q95` | [3, 50] | 3.5 -> 3.349 | **operating** | the plasma current, set on the day -- but held at the design value here: the CS flux and the PF coil set are sized to it, and letting it move per world would re-size them or need every coil rule turned into a capacity check; the three knobs the study frees are the ones that leave the coil set alone |
| 29 | `.build.dr_bore` | [0.1, 10] *default* | 2 -> 2.04 | **build** | the machine bore |
| 37 | `.pf_coil.j_cs_flat_top_end` | [100000, 1e+08] *default* | 1.5e+07 -> 1.744e+07 | **build** | the CS design current density at the end of flat-top: what the CS conductor is built for (c26 checks it against the critical current) |
| 44 | `.physics.f_c_plasma_non_inductive` | [0.001, 1] *default* | 0.4 -> 0.4423 | **operating** | the non-inductive current share; held at the design value (0.442), since it fixes the current-drive power and the pulse scenario the CS is sized for |
| 56 | `.tfcoil.t_tf_superconductor_quench` | [0.1, 100] *default* | 25 -> 19.18 | **build** | TF dump time: protection design (c34, c35, c65) |
| 57 | `.tfcoil.dr_tf_nose_case` | [0.05, 1] *default* | 0.5 -> 0.2016 | **build** | TF nose case thickness (c31) |
| 58 | `.tfcoil.dx_tf_turn_steel` | [0.008, 0.1] | 0.008 -> 0.008085 | **build** | TF conduit thickness (c32) |
| 59 | `.tfcoil.f_a_tf_turn_cable_copper` | [0.5, 0.94] | 0.8 -> 0.8998 | **build** | conductor design: copper fraction of the cable (c35) |
| 60 | `.tfcoil.c_tf_turn` | [65000, 90000] | 6.5e+04 -> 8.259e+04 | **build** | TF current per turn (with 140, the winding pack) |
| 109 | `.physics.f_nd_alpha_thermal_electron` | [0.05, 0.1] | 0.1 -> 0.08566 | **operating** | the helium fraction: a state the particle balance sets, not a knob -- closes c62 per world (n_He = rho* tau_E S_alpha). PROCESS leaves it to the optimiser, which parks it at 0.0857 where rho* = 7.1, well above the 5 the file requires |
| 122 | `.pf_coil.f_a_cs_turn_steel` | [0.001, 0.95] *default* | 0.8 -> 0.6928 | **build** | CS steel fraction (c72, c60) |
| 135 | `.impurity_radiation.f_nd_impurity_electron_array[12]` | [1e-08, 0.01] *default* | 0.00038 -> 0.0006169 | **operating** | the xenon seeding: the divertor / radiation knob, a recourse knob |
| 140 | `.tfcoil.dr_tf_wp_with_insulation` | [0.4, 2] | 0.5 -> 0.5709 | **build** | the winding-pack radial thickness; here an iteration variable, where the stellarator solved it inside the coil model (the rule the stellarator study had to lift is an ixc here, so no lift is needed) |

The winding pack is the one build rule the stellarator study had to *lift* out of a model
(`lift.lift_winding_pack`); in this file it is an `ixc` (140) with `c33` its inequality --
PROCESS already states it the way the lift restates it. What has to be lifted here is
elsewhere (section 4: the PF coil set).

## 2. The `icc` constraints

`n_equality_constraints = 3`; the rest are `g <= 0` in PROCESS's normalised form. "at
PROCESS's x (port)" is the port's value of the condition at PROCESS's converged design
(`driven.process_point`: `c2` closed by the heating power, which comes out at 75.00 MW, and
`c1` by the beta -- the port reproduces PROCESS's power balance -- with the installed power
set to the used 84.8 MW). Negative is satisfied; the four small positive values are where
the port's models differ from PROCESS at that order (`c11` the radial build +8e-4, `c36` the TF
temperature margin +7e-3, `c31` the TF case stress +1.3e-2, `c13` the burn time +1e-4: 7199 s
against 7200), not violations PROCESS reports.

| icc | meaning | class | limit (value) | limit kind | at PROCESS's x (port) | note |
|---|---|---|---|---|---|---|
| 1 | beta consistency: beta_total = beta_th(n, T, B) + beta_fast_alpha + beta_beam | **closure / consistency** | -- | -- | -1.3e-15 | closed per world by `beta_total_vol_avg` (ixc 5), a two-node cycle |
| 2 | global power balance: transport loss + core radiation = alpha + charged + ohmic + injected | **closure / consistency** | -- | -- | +2.7e-11 | PROCESS closes it with `hfact`; here by the heating power (six-node cycle) |
| 11 | radial build consistency: the build sums to rmajor | **consistency (geometry)** | -- | -- | +8.1e-04 | build only, first stage; +8e-4 at PROCESS's x in the port (the port's radial build differs at that order) |
| 30 | injected power <= maximum | **capacity** | `p_hcd_injected_max` = 200 MW -> `p_hcd_installed_mw` | build (installed) | +0.0e+00 | after the rewire the limit is the installed power, 84.8 MW at the deterministic point (zero margin); per world |
| 15 | L-H threshold: P_sep >= f x P_LH | **physics limit** | `f_h_mode_margin` = 1 | numerics (trivial) | -7.6e-01 | per world; inactive (-0.76) |
| 16 | net electric power >= required | **requirement** | `p_plant_electric_net_required_mw` = 400 | build (plant spec) | -1.9e-04 | an inequality in this file (an equality in the stellarator's); active at PROCESS's optimum; per world |
| 24 | beta <= beta_max | **physics limit** | computed: Wesson 4 l_i = 3.6 -> x `f_beta_norm_max` | belief | -2.5e-01 | the file's `beta_norm_max = 3.0` is dead (`i_beta_norm_max = 1`); the factor is the sampled belief; per world; inactive (-0.25) at the optimum |
| 25 | peak TF field <= max | **engineering margin** | `b_tf_inboard_max` = 14 T | build | -1.7e-01 | build only |
| 26 | CS current density at end of flat-top <= critical | **engineering margin** | `fjohc` = 1 | build (margin) | -5.1e-01 | reads the CS peak field, which is per world through the swing; per world |
| 27 | CS current density at pulse start <= critical | **engineering margin** | `fjohc0` = 1 | build (margin) | -3.8e-01 | per world: the pulse-start current is the world's flux swing |
| 33 | TF I_op / I_crit <= f | **engineering margin** | `f_j_tf_wp_critical_max` = 1 | build (margin) | -2.6e-01 | build only |
| 34 | TF dump voltage <= max | **engineering margin** | `v_tf_coil_dump_quench_max_kv` = 10 | build (protection spec) | -1.7e-01 | build only |
| 35 | J_wp <= quench-protection J_max | **engineering margin** | (computed) | -- | -4.4e-05 | build only; active (-4e-5) |
| 36 | TF temperature margin >= min | **engineering margin** | `temp_tf_superconductor_margin_min` = 1.5 K | build (spec) | +7.2e-03 | build only; +7e-3 at PROCESS's x in the port (a deliberate divergence of the port's critical-current fit) |
| 60 | CS temperature margin >= min | **engineering margin** | `temp_cs_superconductor_margin_min` = 1.5 K | build (spec) | -1.1e-04 | per world through the CS peak field; active (-1e-4) |
| 62 | tau_He* / tau_E >= min | **physics limit -> closure** | `f_t_alpha_energy_confinement_min` = 5 | belief | -4.2e-01 | held with equality per world by the helium fraction; inactive at PROCESS's point (rho* = 7.1) |
| 65 | VV stress at quench <= max | **engineering margin** | `max_vv_stress` = 1.43e8 Pa | build (allowable) | -6.7e-01 | build only |
| 72 | CS stress <= max | **engineering margin** | `stress_cs_steel_max` = 7.5e8 Pa | build (allowable) | -1.6e-05 | per world through the CS peak field; active (-1.6e-5) |
| 81 | on-axis density >= pedestal density | **consistency (profile)** | -- | -- | -6.8e-01 | per world; inactive (-0.68) |
| 68 | P_sep B / (q95 A R) <= max | **physics / divertor limit** | `p_div_bt_q_aspect_rmajor_max_mw` = 10 | build (divertor spec) | -1.3e-05 | per world; active at PROCESS's optimum (-1.3e-5) |
| 31 | TF case stress <= max | **engineering margin** | `sig_tf_case_max` = 7.5e8 Pa | build (allowable) | +1.3e-02 | build only; +1.3e-2 at PROCESS's x in the port |
| 32 | TF conduit stress <= max | **engineering margin** | `sig_tf_wp_max` = 7.5e8 Pa | build (allowable) | -1.2e-01 | build only |
| 5 | line density <= f x n_Greenwald | **physics limit** | `f_nd_plasma_electron_limit_max` = 1.2 | belief (held) | -4.2e-06 | per world; active (-4e-6): PROCESS runs at 1.2 x Greenwald |
| 8 | neutron wall load <= max | **engineering margin** | `pflux_fw_neutron_max_mw` = 2 | build (materials spec) | -4.8e-01 | per world; inactive (-0.48) |
| 9 | fusion power <= max | **plant cap** | `p_fusion_total_max_mw` = 3000 | build (plant spec) | -4.5e-01 | per world; inactive (-0.45) |
| 13 | burn time >= min | **requirement (pulsed plant)** | `t_burn_min` = 7200 s | build (plant spec) | +1.1e-04 | per world: the burn time is the CS flux left after the world's ramp-up and consumption at the world's loop voltage; active (+1e-4) |

Active at PROCESS's optimum (|g| < 2e-4): `c16` (net power, 400 MW), `c35` (quench
protection), `c60` (CS temperature margin), `c72` (CS stress), `c68` (PsepB/qAR), `c5` (1.2 x
Greenwald), `c13` (the burn time) and, by construction after the rewire, `c30`. Seven of the
eight are *operating* limits the operator meets per world; `c35` is the build's.

The 19 limits that are boundary inputs are counted as `limit` in the headline: 16
engineering specifications and margins (build), 2 physics beliefs (`f_t_alpha_energy_confinement_min`,
sampled, and `f_nd_plasma_electron_limit_max`, held), 1 trivial bound (`f_h_mode_margin` = 1).
`c24`'s threshold is computed (Wesson's `beta_N,max = 4 l_i` with `l_i = 0.9` an input), and
`c35`'s and `c81`'s compare two outputs.

## 3. Every other boundary input, by kind (compressed)

`kinds_tokamak.KINDS` is the full table; `decision_kinds.md` section 3 has the stellarator's
row-by-row meanings for the 244 shared spellings. Per kind:

| kind | count | what |
|---|---|---|
| belief | 180 | 97 cost-model and financing coefficients (`.costs.UC*`, `uc*`, `c*`, `fcr0`, `discount_rate`, ...); 83 physics / engineering, listed below |
| build | 132 | the radial and vertical build (`.build.*`, 38), the TF coil design (`.tfcoil.*`, 22), the PF coils and the CS (`.pf_coil.*`, 22), the blanket / first wall / shield / divertor (`.fwbs.*`, `.divertor.*`, 19), the buildings (21), the plasma shape (`aspect`, `kappa`, `triang`), the fatigue safety factors, `n_tf_coils = 16`, `life_plant` |
| operating | 24 | listed below: the seven operating `ixc`, the heating power, the fuel mix, the coil temperatures, the pulse timings, the coolant settings, the vacuum pressures, the availability |
| numerics | 46 | switches (`i_*`, `lsa`, `ireactor`, ...), the `^stated.*` pins the port restates (14), inert zero reads (`p_beam_*`, `p_hcd_secondary_*`), smoothing parameters (`alfapf`) |
| derived | 11 | read at zero, produced by no node of this graph: `.costs.cplife` = 0; `.fwbs.m_blkt_vanadium` = 0; `.tfcoil.tfcmw` = 0; `.tfcoil.m_tf_bus` = 0; `.heat_transport.p_fw_div_heat_deposited_mw` = 0; `.tfcoil.res_tf_leg` = 0; `.heat_transport.p_fw_coolant_pump_mw` = 0; `.heat_transport.p_blkt_coolant_pump_mw` = 0; `.fwbs.p_fw_hcd_nuclear_heat_mw` = 0; `.fwbs.life_fw_fpy` = 0; `.tfcoil.sig_tf_cs_bucked` = 0; |
| limit | 19 | `.current_drive.p_hcd_injected_max` = 200; `.constraints.f_h_mode_margin` = 1; `.constraints.p_plant_electric_net_required_mw` = 400; `.constraints.b_tf_inboard_max` = 14; `.constraints.fjohc` = 1; `.constraints.fjohc0` = 1; `.tfcoil.v_tf_coil_dump_quench_max_kv` = 10; `.tfcoil.temp_tf_superconductor_margin_min` = 1.5; `.tfcoil.temp_cs_superconductor_margin_min` = 1.5; `.constraints.f_t_alpha_energy_confinement_min` = 5; `.tfcoil.max_vv_stress` = 1.43e+08; `.pf_coil.stress_cs_steel_max` = 7.5e+08; `.constraints.p_div_bt_q_aspect_rmajor_max_mw` = 10; `.tfcoil.sig_tf_case_max` = 7.5e+08; `.tfcoil.sig_tf_wp_max` = 7.5e+08; `.constraints.f_nd_plasma_electron_limit_max` = 1.2; `.constraints.pflux_fw_neutron_max_mw` = 2; `.constraints.p_fusion_total_max_mw` = 3000; `.constraints.t_burn_min` = 7200; |

### 3.1 The 83 physics / engineering beliefs

Values are the cold `DataStructure`; `readers` is how many nodes of the problem graph read
the place. The eight rows the study samples are in **section 4**; every other row is held at
its value.

| path | value | readers |
|---|---|---|
| `^stated.tfcoil.eff_tf_cryo` | 0.13 | 1 |
| `.tfcoil.dcond[2]` | 6070 | 2 |
| `.tfcoil.dcond[0]` | 6080 | 3 |
| `.heat_transport.vachtmw` | 0.5 | 3 |
| `.heat_transport.p_tritium_plant_electric_mw` | 15 | 3 |
| `.physics.f_p_alpha_plasma_deposited` | 0.95 | 6 |
| `.physics.plasma_res_factor` | 0.7 | 1 |
| `.physics.csawth` | 1 | 1 |
| `.physics.ejima_coeff` | 0.3 | 1 |
| `.physics.ind_plasma_internal_norm` | 0.9 | 3 |
| `.physics.q0` | 1 | 1 |
| `.current_drive.cboot` | 1 | 1 |
| `.current_drive.eta_cd_norm_ecrh` | 0.3 | 1 |
| `.current_drive.eta_ecrh_injector_wall_plug` | 0.5 | 1 |
| `.tfcoil.den_tf_wp_turn_insulation` | 1800 | 1 |
| `.tfcoil.den_tf_coil_case` | 8000 | 1 |
| `.fwbs.den_steel` | 7800 | 6 |
| `.tfcoil.eyoung_steel` | 2.05e+11 | 1 |
| `.tfcoil.poisson_steel` | 0.3 | 2 |
| `.tfcoil.poisson_cond_axial` | 0.3 | 1 |
| `.tfcoil.poisson_cond_trans` | 0.3 | 1 |
| `.tfcoil.poisson_ins` | 0.34 | 1 |
| `.tfcoil.eyoung_copper` | 1.17e+11 | 1 |
| `.tfcoil.poisson_copper` | 0.35 | 1 |
| `.tfcoil.str_cs_con_res` | -0.005 | 2 |
| `.cs_fatigue.paris_coefficient` | 6.5e-13 | 1 |
| `.cs_fatigue.paris_power_law` | 3.5 | 1 |
| `.cs_fatigue.walker_coefficient` | 0.436 | 1 |
| `.cs_fatigue.fracture_toughness` | 200 | 1 |
| `.physics.rad_fraction_sol` | 0.8 | 1 |
| `.physics.ffwal` | 0.92 | 2 |
| `.constraints.f_fw_rad_max` | 3.33 | 1 |
| `.fwbs.fvolsi` | 1 | 1 |
| `.fwbs.fvolso` | 0.64 | 1 |
| `.fwbs.fvoldw` | 1.74 | 1 |
| `.divertor.den_div_structure` | 10000 | 1 |
| `.ccfe_hcpb.fw_armour_u_nuc_heating` | 6.25e-07 | 1 |
| `.fwbs.f_p_blkt_multiplication` | 1.269 | 1 |
| `.primary_pumping.gamma_he` | 1.667 | 1 |
| `.fwbs.etaiso` | 0.9 | 1 |
| `.heat_transport.f_p_shld_coolant_pump_total_heat` | 0.005 | 1 |
| `.heat_transport.f_p_div_coolant_pump_total_heat` | 0.005 | 1 |
| `.physics.f_nd_plasma_pedestal_greenwald` | 0.85 | 1 |
| `.physics.f_nd_plasma_separatrix_greenwald` | 0.5 | 1 |
| `.physics.radius_plasma_pedestal_temp_norm` | 0.94 | 2 |
| `.physics.temp_plasma_pedestal_kev` | 5.5 | 2 |
| `.physics.temp_plasma_separatrix_kev` | 0.1 | 2 |
| `.physics.alphat` | 1.45 | 5 |
| `.physics.tbeta` | 2 | 3 |
| `.physics.radius_plasma_pedestal_density_norm` | 0.94 | 2 |
| `.physics.alphan` | 1 | 5 |
| `.physics.f_temp_plasma_ion_electron` | 1 | 2 |
| `.physics.hfact` | 1.1 | 1 |
| `.physics.f_sync_reflect` | 0.6 | 1 |
| `.impurity_radiation.f_nd_impurity_electron_array[2]` | 0 | 3 |
| `.impurity_radiation.f_nd_impurity_electron_array[3]` | 0 | 3 |
| `.impurity_radiation.f_nd_impurity_electron_array[4]` | 0 | 3 |
| `.impurity_radiation.f_nd_impurity_electron_array[5]` | 0 | 3 |
| `.impurity_radiation.f_nd_impurity_electron_array[6]` | 0 | 3 |
| `.impurity_radiation.f_nd_impurity_electron_array[7]` | 0 | 3 |
| `.impurity_radiation.f_nd_impurity_electron_array[8]` | 0 | 3 |
| `.impurity_radiation.f_nd_impurity_electron_array[9]` | 0 | 3 |
| `.impurity_radiation.f_nd_impurity_electron_array[10]` | 0 | 3 |
| `.impurity_radiation.f_nd_impurity_electron_array[11]` | 0 | 3 |
| `.impurity_radiation.f_nd_impurity_electron_array[13]` | 5e-06 | 3 |
| `.impurity_radiation.temp_impurity_keV_array` | array(14, 200) | 3 |
| `.impurity_radiation.pden_impurity_lz_nd_temp_array` | array(14, 200) | 1 |
| `.physics.tauratio` | 1 | 1 |
| `.physics.f_nd_protium_electrons` | 0 | 1 |
| `.impurity_radiation.impurity_arr_zav` | array(14, 200) | 2 |
| `.impurity_radiation.m_impurity_amu_array` | array(14,) | 1 |
| `.pf_coil.rhopfbus` | 3.93e-08 | 1 |
| `.pf_power.f_p_pf_energy_store_loss` | 0.1 | 1 |
| `.pf_power.f_p_pf_psu_loss` | 0.1 | 1 |
| `.pf_coil.etapsu` | 0.9 | 1 |
| `.tfcoil.rho_tf_bus` | 1.86e-08 | 1 |
| `.heat_transport.etatf` | 0.9 | 1 |
| `.fwbs.eta_coolant_pump_electric` | 0.87 | 1 |
| `.fwbs.qnuc` | 13000 | 2 |
| `.vacuum.outgrat_fw` | 1.3e-08 | 1 |
| `.heat_transport.p_plant_electric_base` | 5e+06 | 1 |
| `.heat_transport.pflux_plant_floor_electric` | 150 | 1 |
| `.heat_transport.eta_turbine` | 0.4 | 1 |

Decisions worth naming (the rest follow the stellarator table): the pedestal
(`temp_plasma_pedestal_kev`, `f_nd_plasma_pedestal_greenwald`, the profile radii, `tbeta`)
is a profile belief, not a knob -- with `i_plasma_pedestal = 1` the pedestal density is
0.85 x Greenwald and the on-axis values follow from the volume averages the operator sets;
`ind_plasma_internal_norm` (`l_i = 0.9`) is a belief held at its value -- it enters both the
beta limit (Wesson) and the volt-seconds, and the study samples the beta limit's threshold
through a factor on the coefficient instead, so the flux consumption is not moved by the same
draw; `plasma_res_factor`, `ejima_coeff`, `csawth`, `q0`, `cboot` are physics coefficients;
`eta_cd_norm_ecrh` (the current-drive efficiency) and `eta_ecrh_injector_wall_plug` are
engineering beliefs held at the file's values; the fatigue model's material coefficients are
beliefs and its safety factors build.

### 3.2 The 24 operating variables

| path | value | readers |
|---|---|---|
| `.costs.f_t_plant_available` | 0.8 | 3 |
| `.tfcoil.temp_tf_cryo` | 4.5 | 2 |
| `.physics.f_plasma_fuel_tritium` | 0.5 | 3 |
| `.physics.f_plasma_fuel_helium3` | 0 | 4 |
| `.physics.q95` | 3.5 | 4 |
| `.physics.nd_plasma_electrons_vol_avg` | 7.5e+19 | 14 |
| `.physics.temp_plasma_electron_vol_avg_kev` | 12 | 9 |
| `.times.t_plant_pulse_fusion_ramp` | 10 | 5 |
| `.physics.beta_total_vol_avg` | 0.03 | 6 |
| `.physics.f_c_plasma_non_inductive` | 0.4 | 1 |
| `.current_drive.p_hcd_primary_extra_heat_mw` | 75 | 2 |
| `.tfcoil.tftmp` | 4.75 | 3 |
| `.pf_coil.temp_cs_superconductor_operating` | 4.75 | 2 |
| `.times.t_plant_pulse_coil_precharge` | 500 | 4 |
| `.times.t_plant_pulse_dwell` | 1800 | 3 |
| `.primary_pumping.p_he` | 8e+06 | 1 |
| `.primary_pumping.dp_he` | 550000 | 1 |
| `.primary_pumping.t_in_bb` | 573.13 | 1 |
| `.primary_pumping.t_out_bb` | 773.13 | 1 |
| `.physics.f_plasma_fuel_deuterium` | 0.5 | 3 |
| `.impurity_radiation.f_nd_impurity_electron_array[12]` | 0.00038 | 3 |
| `.physics.f_nd_alpha_thermal_electron` | 0.1 | 1 |
| `.vacuum.pres_vv_chamber_base` | 0.0005 | 1 |
| `.vacuum.pres_div_chamber_burn` | 0.36 | 1 |

`q95` and `f_c_plasma_non_inductive` are operating by kind and **held at the design value**
in the study (section 4). The two coil temperatures, the pulse timings (`t_plant_pulse_dwell`,
`_coil_precharge`, `_fusion_ramp`), the coolant settings, the fuel mix and the availability
are operating settings held at the file's values, as the stellarator study held them.

## 4. The two-stage split, as implemented

`architectures/driven.py` (the graph operations and the assembly), `kinds_tokamak.py`
(the tables), `paper_tests/flexibility_tokamak.py` (the driver). Every step is a cottax op
on the problem graph of section 0 (275 nodes, 412 inputs, 3 equalities, 23 inequalities):

| # | step | ops | nodes | inputs | inequalities |
|---|---|---|---|---|---|
| 0 | the problem graph | `mda.SCHEME` (3 `FixedPointCut`s) + `Insert(icc x26, objective)` | 275 | 412 | 23 |
| 1 | the installed power | `Rewire(.costs.power_injection_cost: p_hcd_ecrh_injected_total_mw -> p_hcd_installed_mw)`, `Rewire(.Constraint30: p_hcd_injected_max -> p_hcd_installed_mw)` | 275 | 412 (+1, -1: `p_hcd_injected_max` is read by nothing now) | 23 |
| 2 | the beta limit's factor | `Redefine(.tokamak.plasma_beta.norm_max)`: reads `(l_i, f_beta_norm_max)`, body `f x 4 l_i` | 275 | 413 | 23 |
| 3 | the closures | `Insert(RootFind)` x3 at `.Close.c2` / `.Close.c62` / `.Close.c1`; the three cycles touch, so the fuel-ion Picard is `Residualise`d and the three root finds and it are `Combine`d into `^problem.Close.c2` (5 unknowns: the heating power, the two cut copies, the helium fraction, the beta); `nested_inside`; `Assign(safeguarded())` | 276 | 421 (driven: 8 `^guess.*` start ports) | **22** (`c62` closed) |
| 4 | the freezes | `Cut(c_pf_cs_coils_peak_ma, readers=(sizes, masses, pf_magnet_cost, pf_coil_power), mint ^built)` + `Insert(.Built.c_pf_cs_coils_peak_ma)`; 12 `Cut`s of the PF peak fields (readers: masses, strand_critical_current); `Cut(ind_plasma, readers=(inductance,))`, `Cut(f_ster_div_single)`, 3 `Cut`s of the first-wall areas, `Cut(helpow, readers=(buildings.sizing,))`, 5 `Cut`s of the vacuum system -- 24 `^built.*` inputs | 277 | 445 | 23 (+ the coil capacity) |
| 5 | the recourse bound | the heating power's lower bound as a per-world inequality (`ouu.RecourseBound`, normalised by the 200 MW cap) | | | 24 |
| 6 | the split | `stages.split` at 8 sampled leaves + 5 start ports + 3 knobs: **135 first-stage / 140 second-stage / 2 recourse nodes of 277** (113 / 161 / 2 of 276 before step 4; 385 first-stage owned places against 263); 8 of the 24 inequalities are constants given the build and are not posed to the operator | | | 16 posed |

**Beliefs (8 sampled, two groups).** PLASMA: `hfact` lognormal sigma 0.10 (0.05 / 0.02 on
the cluster), `alphan` +-20 %, `alphat` +-20 %, `f_temp_plasma_ion_electron` U[0.85, 1],
`f_p_alpha_plasma_deposited` U[0.90, 0.99], the tungsten fraction
`f_nd_impurity_electron_array[13]` +-20 % (the file seeds xenon at index 12 -- `ixc 135`, an
operator knob -- and tungsten at index 13, 5e-6). LIMITS: `f_beta_norm_max` +-20 % (the
`Redefine` of step 2: the beta limit's threshold is Wesson's `4 l_i` = 3.6, the file's
`beta_norm_max = 3.0` is dead because `i_beta_norm_max` defaults to 1), and
`f_t_alpha_energy_confinement_min` U[3, 6] (5 in the file). All eight are boundary inputs of
the problem graph (the factor after step 2); all eight reach 154-159 of the 276 nodes but
the beta factor, which reaches 3.

**Build (first stage, fixed at PROCESS's converged design).** The twelve build `ixc` (B, R,
`dr_cs`, `dr_bore`, `j_cs_flat_top_end`, `t_quench`, `dr_tf_nose_case`, `dx_tf_turn_steel`,
`f_a_tf_turn_cable_copper`, `c_tf_turn`, `f_a_cs_turn_steel`, `dr_tf_wp_with_insulation`), the
two held operating `ixc` (`q95`, `f_c_plasma_non_inductive`), every other input at the file's
value, and the **installed H&CD power** `p_hcd_installed_mw` = the heating power the balance
asks for at PROCESS's design (75.00 MW in the port) + the current-drive power (9.82 MW) =
**84.8 MW, zero margin**. The eight build constraints (`c25`, `c31`, `c32`, `c33`, `c34`,
`c35`, `c36`, `c65`) are functions of the build alone and are evaluated once.

**Closures per world.** `c2` by the heating power (a six-node cycle: the injected power, the
loss power, the scaling, its tail, the condition), `c62` by the helium fraction (28 nodes: the
dilution, the fusion rates, the radiation, the confinement time -- `c62` is an inequality
held with equality, `n_He = rho* tau_E S_alpha`), `c1` by the beta (two nodes: `ixc 5` is the
copy of a computed quantity). One square problem of five unknowns, `closing.safeguarded()`
(capped, backtracked, Broyden-updated Newton), 7 steps at the nominal to 1e-13.

**Operator knobs (recourse).** The density (`ixc 6`, under `c5`), the temperature (`ixc 4`),
the xenon fraction (`ixc 135`), scaled by their nominal values for SLSQP. Per world: phase 1
`psi = min_z max_j g_j` over the **16 posed** conditions (`c30`, `c15`, `c16`, `c24`, `c26`,
`c27`, `c60`, `c72`, `c81`, `c68`, `c5`, `c8`, `c9`, `c13`, the coil capacity, the heating
power's lower bound), then `min coe` s.t. `g <= max(tol, psi)`.

**Pulsed-plant constraints.** `c13` (the burn time) is per world: the burn time is what is
left of the CS flux after the world's ramp-up consumption at the world's loop voltage
(`v_plasma_loop_burn` reads the plasma resistance, per world through T_e and Z_eff, and the
inductive current fraction, per world through the bootstrap fraction); `stages.split` puts
`.tokamak.pulse.burn_time`, the flux-swing Picard and `.Constraint13` in the second stage.
So are `c26`, `c27`, `c60`, `c72` (the CS at the world's swing) -- they bound operability, and
`c60` / `c72` are active in every operable world. `c11` (the radial build) is first stage and
is not posed.

**Lifts.** `stages.violations` needs a claimed-build-output table the tokamak never had;
the split was read directly: with the beliefs and the three knobs varying, 161 of 276
nodes were second stage, among them the **whole PF coil system** (`equilibrium_currents`
reads the poloidal beta -> the time-point currents -> the waveform's peak currents -> the
coil sizes, turns, masses, peak fields, inductances), the cryostat (reads the PF coil
positions), the structure masses, the buildings, the vacuum system, and the first-wall /
blanket masses (a node owning the first-wall *area* beside the neutron flux). None is a
root find, so `lift.lift` does not apply; `driven.freeze` is the explicit rule's lift: a
`Cut` of the place for the readers that size the build from it, minting `^built.<place>`, a
first-stage input at the nominal, and -- for the coil currents -- an inserted capacity
inequality `^cond.built.pf_coil.c_pf_cs_coils_peak_ma = max_coil (|I_world| - |I_built|) /
max|I_built| <= 0`. After the 24 freezes the first stage is 135 nodes and the second stage
holds the plasma, the current drive, the divertor, the CS at the world's swing, the burn
time, the power flows, the nuclear heating, the lifetimes and the costs (the construction
cost varies per world only through the cost-only items the stellarator census also left:
the heat exchangers, the pumping and cryogenic power ratings, the fuel processing).
