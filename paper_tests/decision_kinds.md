# Decision kinds in the PROCESS stellarator reference run

`tests/regression/input_files/stellarator_helias.IN.DAT` (HELIAS 5-B, `istell = 6`, ISS04,
ignited, 1990 cost model, figure of merit 6 = minimise cost of electricity), read through the
cottax port. Everything below was listed programmatically:
`indat.GRAPH` -> `mda.driven_graph` (156 nodes; boundary 311 = 289 `input` + 6 `guess` + 16
`stated`) and `boundary.problem_graph` (the same graph with the 14 constraint nodes and the
objective inserted; boundary **324**, the 13 extra being 10 constraint limits and 3 quantities
the constraint nodes read that no stellarator node produces). The `Optimise` node built by
`sand.optimise_graph` owns the 8 `ixc` places and reads `^cond.numerics.objf` plus
`^cond.constraints.c<n>` for the 14 `icc`. Values are the cold, initialised `DataStructure`
(`SingleRun(...).data` before the solve); converged values are PROCESS's own (cached
`reference_run`).

Kinds: **belief** (nobody chooses, nobody knows exactly), **build decision** (fixed before the
machine exists), **operating variable** (set in operation given the machine and the realised
physics), **numerics / switch**, **derived / output**. A constraint *limit* is listed apart in §2
and counted with the kind it is a limit *on*.

Headline: 324 boundary inputs of the problem graph = 147 belief (73 of them 1990
cost-model coefficients) + 76 build decisions + 16 operating variables +
62 numerics / switches / inert tokamak-only reads + 13 derived-but-read-from-outside
(missing producers, all zero) + 10 constraint limits. Of the 8 `ixc`: 4 build, 3 operating, 1 belief.

## 1. The `ixc` iteration variables

Bounds are the file's `boundl`/`boundu` where set, else `ITERATION_VARIABLES`' defaults (marked
*default*). Start / converged are PROCESS's own run.

| ixc | place | meaning | bounds | start -> converged | kind | why |
|---|---|---|---|---|---|---|
| 2 | `.physics.b_plasma_toroidal_on_axis` | toroidal field on axis (T) | [4, 10] | 5.5 -> 4.70 | **build** | sets the coil current (`coilcurrent` = f(B, R)) and the whole magnet; cannot change after the coils are wound |
| 3 | `.physics.rmajor` | plasma major radius (m) | [10, 30] | 20 -> 26.7 | **build** | the machine size; every coil and build length scales from it (`f_st_rmajor`) |
| 4 | `.physics.temp_plasma_electron_vol_avg_kev` | volume-averaged T_e (keV) | [3, 15] | 7 -> 5.67 | **operating** | set by heating and fuelling on the day; recourse candidate |
| 6 | `.physics.nd_plasma_electrons_vol_avg` | volume-averaged n_e (m^-3) | [3e19, 3e20] | 2e20 -> 1.75e20 | **operating** | fuelling; the knob uq.py/ouu.py already close the power balance with (scaled sensitivity 0.33 on c2) |
| 10 | `.physics.hfact` | H-factor on ISS04 confinement | [0.1 *default*, 1.2] | 1.0 -> 1.056 | **belief** | a confinement multiplier is not chosen; PROCESS uses it as the closure knob for c2 (scaled sensitivity 1.0), which is exactly why the deterministic optimum is luck |
| 109 | `.physics.f_nd_alpha_thermal_electron` | thermal alpha density / n_e | [1e-4, 0.4] | 0.1 -> 0.0336 | **operating** | fixed by fuelling / He exhaust for a given confinement; PROCESS lets the optimiser pick it, which makes it a de-facto closure of c16 (sensitivity 0.42) rather than a physics quantity |
| 59 | `.tfcoil.f_a_tf_turn_cable_copper` | copper fraction of the cable conductor | [0.3, 0.9] | 0.7 -> 0.738 | **build** | conductor design, fixed when the cable is made; it trades quench protection (c35) against WP size |
| 56 | `.tfcoil.t_tf_superconductor_quench` | TF fast-discharge (dump) time (s) | [1, 50] | 35 -> 35.3 | **build** | dump-resistor / protection design; its only leverage is on c34/c35/c65 (VV stress), sensitivity on c16 is 8e-5 |

Commented-out in the file and therefore fixed: `aspect` (ixc 1; the port overrides the file's
11.1 with the configuration's `aspect_ref` = 12.3 via `DefaultAspectRatio`), `f_st_coil_aspect`
(ixc 176, = 1.0), `f_j_tf_wp_critical_max` (ixc 50, = 0.8). All three are build decisions
the file has frozen.

## 2. The `icc` constraints

`n_equality_constraints = 2`; the rest are `g <= 0` in PROCESS's normalised form. "p_viol"
is the fraction of `uq.py`'s 100k-sample Monte Carlo (design at the deterministic optimum, c2
closed by the density, 26 uncertain inputs) that violates the constraint (`out/uq_mc.csv`).

| icc | meaning (ported `constraint_<n>`) | reads | class | limit (value) | limit kind | p_viol | note |
|---|---|---|---|---|---|---|---|
| 2 | global power balance: transport + core radiation loss = alpha + non-alpha charged + ohmic + auxiliary | `pden_*_transport_loss`, `pden_plasma_core_rad`, `f_p_alpha_plasma_deposited`, `pden_alpha_total`, `p_hcd_injected_total`, `vol_plasma` | **closure / consistency** | -- | -- | 0 (closed) | to be closed per sample; PROCESS closes it with `hfact`, uq/ouu with the density |
| 16 | net electric power >= required | `p_plant_electric_net_mw` | **requirement (equality in this file)** | `p_plant_electric_net_required_mw` = 1000 MW | build (plant spec) | 0.55 | a target, not a physics limit; as an equality it is a closure candidate for the second operating knob (T_e or f_nd_alpha), or an inequality-with-CVaR (`ouu.py --with-c16`: infeasible robustly) |
| 24 | beta <= beta_max (stellarator branch uses total beta) | `beta_total_vol_avg` | **physics limit** | `beta_vol_avg_max` = 0.04 | belief (stellarator beta limit) | 0.58 | chance constraint; the limit value itself is a belief |
| 8 | neutron wall load <= max | `pflux_fw_neutron_mw` | **engineering margin** | `pflux_fw_neutron_max_mw` = 1.5 MW/m2 | build (materials spec) | 0.29 | belief-sensitive through `ffwal`, fusion power; keep the limit at nominal, treat as probabilistic |
| 17 | radiation fraction <= max (stellarator: `f_p_plasma_separatrix_rad - psolradmw/p_heating`) | `f_p_plasma_separatrix_rad`, `psolradmw`, `p_plasma_heating_total_mw` | **physics limit** | `f_p_plasma_separatrix_rad_max` = 1.0 | numerics (trivial bound) | 0.007 | almost inert at this optimum |
| 18 | divertor heat load <= max | `pflux_div_heat_load_mw` | **engineering margin** | `pflux_div_heat_load_max_mw` = 10 MW/m2 | build (target spec) | 0.11 | belief-sensitive (`f_asym`, `f_rad`, `bmn`, `xpertin`, `tdiv`); probabilistic |
| 67 | radiation wall load <= max | `pflux_fw_rad_max_mw` (= `f_fw_rad_max` x mean) | **engineering margin** | `pflux_fw_rad_max` = 1.2 MW/m2 | build (materials spec) | 0.26 | peaking factor `f_fw_rad_max` = 3.33 is a belief |
| 82 | toroidal gap between coils >= coil toroidal thickness | `toroidalgap`, `dx_tf_inboard_out_toroidal` | **consistency (geometry)** | -- | -- | 0 | build geometry; deterministic given the build decisions, keep at nominal |
| 83 | available radial space >= required (plasma-coil distance vs blanket+shield+VV+gaps) | `available_radial_space`, `required_radial_space` | **consistency (geometry)** | -- | -- | 0.50 | active at the optimum (residual -1e-5) and split 50/50 only by `dr_fw_wall` scatter; deterministic given the build -- keep, not a chance constraint |
| 62 | tau_He* / tau_E >= min | `f_t_alpha_energy_confinement` | **physics limit** | `f_t_alpha_energy_confinement_min` = 4 | belief (He exhaust) | 0.16 | chance constraint; the 4 is itself a belief |
| 32 | TF conduit (WP) stress <= max | `sig_tf_wp` | **engineering margin** | `sig_tf_wp_max` = 4e8 Pa | build (allowable) | 0 | inactive (-0.69) |
| 34 | TF dump voltage <= max | `v_tf_coil_dump_quench_kv` | **engineering margin** | `v_tf_coil_dump_quench_max_kv` = 12 kV | build (protection spec) | 0 | inactive (-0.86) |
| 35 | J_wp <= quench-protection J_max | `j_tf_wp`, `j_tf_wp_quench_heat_max` | **engineering margin** | (computed) | -- | 0.50 | active at the optimum (1.6e-4); moves with `temp_tf_cryo`, `dx_tf_wp_insulation`, `f_j_tf_wp_critical_max`; probabilistic unless those are held |
| 65 | VV shear stress at quench <= max | `vv_stress_quench` | **engineering margin** | `max_vv_stress` = 9.3e7 Pa | build (allowable) | 0 | inactive (-0.98) |

The 10 limits are boundary inputs of the problem graph and are the 10 rows counted as
`limit` in the headline: 8 engineering specs (build), 2 physics beliefs (`beta_vol_avg_max`,
`f_t_alpha_energy_confinement_min`); `f_p_plasma_separatrix_rad_max = 1` is a trivial bound.

## 3. Every other boundary input, by kind

`set by`: `IN.DAT` = the file sets it explicitly; blank = PROCESS default; `ixc` = an iteration
variable (already in §1, repeated here for completeness of the boundary). Values are the cold
DataStructure.

### 3.1 Beliefs (147; 74 physics / engineering + 73 cost coefficients)

| path | value | meaning | set by |
|---|---|---|---|
| `.constraints.f_fw_rad_max` | 3.33 | radiation wall-load peaking factor (3.33) |  |
| `.constraints.f_j_tf_wp_critical_max` | 0.8 | allowed I/Ic fraction; sizes the winding pack (uq.py samples it); a designer margin in practice | IN.DAT |
| `.costs.abktflnc` | 15 | allowable neutron fluence of FW/blanket (MW-y/m2): materials belief that sets blanket life | IN.DAT |
| `.costs.adivflnc` | 25 | allowable divertor heat fluence (MW-y/m2) | IN.DAT |
| `.costs.cfind` | [0.244, 0.244, 0.244, 0.29] | financing: discount rate / cost of money / contingency (economic belief) |  |
| `.costs.cowner` | 0.15 | financing: discount rate / cost of money / contingency (economic belief) |  |
| `.costs.decomf` | 0.1 | financing: discount rate / cost of money / contingency (economic belief) |  |
| `.costs.dintrt` | 0 | financing: discount rate / cost of money / contingency (economic belief) | IN.DAT |
| `.costs.discount_rate` | 0.06 | financing: discount rate / cost of money / contingency (economic belief) | IN.DAT |
| `.costs.dtlife` | 0 | financing: discount rate / cost of money / contingency (economic belief) |  |
| `.costs.fcap0` | 1.15 | financing: discount rate / cost of money / contingency (economic belief) | IN.DAT |
| `.costs.fcap0cp` | 1.06 | financing: discount rate / cost of money / contingency (economic belief) | IN.DAT |
| `.costs.fcdfuel` | 0.1 | financing: discount rate / cost of money / contingency (economic belief) |  |
| `.costs.fcontng` | 0.15 | financing: discount rate / cost of money / contingency (economic belief) | IN.DAT |
| `.costs.fcr0` | 0.065 | financing: discount rate / cost of money / contingency (economic belief) | IN.DAT |
| `.costs.fkind` | 1 | financing: discount rate / cost of money / contingency (economic belief) | IN.DAT |
| `.current_drive.eta_ecrh_injector_wall_plug` | 0.5 | ECRH wall-plug efficiency | IN.DAT |
| `.divertor.den_div_structure` | 1e+04 | divertor structure density (material constant) |  |
| `.divertor.xpertin` | 1.5 | SOL perpendicular transport coefficient (m2/s) | IN.DAT |
| `.fwbs.declblkt` | 0.1 | neutron deposition decay length (m), stellarator-only 1-D model | IN.DAT |
| `.fwbs.declfw` | 0.1 | neutron deposition decay length (m), stellarator-only 1-D model | IN.DAT |
| `.fwbs.declshld` | 0.056 | neutron deposition decay length (m), stellarator-only 1-D model | IN.DAT |
| `.fwbs.den_steel` | 7800 | steel density (material constant) | IN.DAT |
| `.fwbs.eta_coolant_pump_electric` | 1 | primary coolant pump electrical efficiency | IN.DAT |
| `.fwbs.f_p_blkt_multiplication` | 1.35 | blanket energy multiplication | IN.DAT |
| `.fwbs.fvoldw` | 1.74 | area coverage factor (VV / inboard shield / outboard shield volume): modelling belief |  |
| `.fwbs.fvolsi` | 1 | area coverage factor (VV / inboard shield / outboard shield volume): modelling belief |  |
| `.fwbs.fvolso` | 0.64 | area coverage factor (VV / inboard shield / outboard shield volume): modelling belief |  |
| `.heat_transport.eta_turbine` | 0.375 | thermal-to-electric efficiency (user-set, i_thermal_electric_conversion=2) | IN.DAT |
| `.heat_transport.etatf` | 0.9 | AC-to-resistive conversion efficiency of TF supplies |  |
| `.heat_transport.f_p_blkt_coolant_pump_total_heat` | 0.033 | pumping power as a fraction of component thermal power | IN.DAT |
| `.heat_transport.f_p_div_coolant_pump_total_heat` | 0.107 | pumping power as a fraction of component thermal power | IN.DAT |
| `.heat_transport.f_p_fw_coolant_pump_total_heat` | 0.033 | pumping power as a fraction of component thermal power | IN.DAT |
| `.heat_transport.f_p_shld_coolant_pump_total_heat` | 0 | pumping power as a fraction of component thermal power | IN.DAT |
| `.heat_transport.p_plant_electric_base` | 5e+06 | base plant electric load / floor-area load (W, W/m2): balance-of-plant belief |  |
| `.heat_transport.p_tritium_plant_electric_mw` | 15 | tritium plant / vacuum plant electric load (MW): balance-of-plant belief |  |
| `.heat_transport.pflux_plant_floor_electric` | 150 | base plant electric load / floor-area load (W, W/m2): balance-of-plant belief |  |
| `.heat_transport.vachtmw` | 0.5 | tritium plant / vacuum plant electric load (MW): balance-of-plant belief |  |
| `.impurity_radiation.f_nd_impurity_electron_array[10]` | 0 | impurity fraction n_Z/n_e (only [13] tungsten = 1e-5 is non-zero) |  |
| `.impurity_radiation.f_nd_impurity_electron_array[11]` | 0 | impurity fraction n_Z/n_e (only [13] tungsten = 1e-5 is non-zero) |  |
| `.impurity_radiation.f_nd_impurity_electron_array[12]` | 0 | impurity fraction n_Z/n_e (only [13] tungsten = 1e-5 is non-zero) |  |
| `.impurity_radiation.f_nd_impurity_electron_array[13]` | 1e-05 | impurity fraction n_Z/n_e (only [13] tungsten = 1e-5 is non-zero) |  |
| `.impurity_radiation.f_nd_impurity_electron_array[2]` | 0 | impurity fraction n_Z/n_e (only [13] tungsten = 1e-5 is non-zero) |  |
| `.impurity_radiation.f_nd_impurity_electron_array[3]` | 0 | impurity fraction n_Z/n_e (only [13] tungsten = 1e-5 is non-zero) |  |
| `.impurity_radiation.f_nd_impurity_electron_array[4]` | 0 | impurity fraction n_Z/n_e (only [13] tungsten = 1e-5 is non-zero) |  |
| `.impurity_radiation.f_nd_impurity_electron_array[5]` | 0 | impurity fraction n_Z/n_e (only [13] tungsten = 1e-5 is non-zero) |  |
| `.impurity_radiation.f_nd_impurity_electron_array[6]` | 0 | impurity fraction n_Z/n_e (only [13] tungsten = 1e-5 is non-zero) |  |
| `.impurity_radiation.f_nd_impurity_electron_array[7]` | 0 | impurity fraction n_Z/n_e (only [13] tungsten = 1e-5 is non-zero) |  |
| `.impurity_radiation.f_nd_impurity_electron_array[8]` | 0 | impurity fraction n_Z/n_e (only [13] tungsten = 1e-5 is non-zero) |  |
| `.impurity_radiation.f_nd_impurity_electron_array[9]` | 0 | impurity fraction n_Z/n_e (only [13] tungsten = 1e-5 is non-zero) |  |
| `.impurity_radiation.impurity_arr_zav` | array(14, 200) | atomic data table (ADAS-derived Lz, <Z>, masses on a temperature grid): physics belief with small stated error |  |
| `.impurity_radiation.m_impurity_amu_array` | array(14) | atomic data table (ADAS-derived Lz, <Z>, masses on a temperature grid): physics belief with small stated error |  |
| `.impurity_radiation.pden_impurity_lz_nd_temp_array` | array(14, 200) | atomic data table (ADAS-derived Lz, <Z>, masses on a temperature grid): physics belief with small stated error |  |
| `.impurity_radiation.temp_impurity_keV_array` | array(14, 200) | atomic data table (ADAS-derived Lz, <Z>, masses on a temperature grid): physics belief with small stated error |  |
| `.physics.alphan` | 0.35 | density profile exponent (parabolic profiles) | IN.DAT |
| `.physics.alphat` | 1.2 | temperature profile exponent | IN.DAT |
| `.physics.f_nd_protium_electrons` | 0 | seeded protium fraction (0) |  |
| `.physics.f_p_alpha_plasma_deposited` | 0.95 | alpha power deposited in the plasma (0.95) |  |
| `.physics.f_sync_reflect` | 0.6 | synchrotron wall reflectivity | IN.DAT |
| `.physics.f_temp_plasma_ion_electron` | 0.95 | T_i/T_e | IN.DAT |
| `.physics.ffwal` | 0.92 | plasma-surface to first-wall area factor for the neutron wall load (0.92) |  |
| `.physics.hfact` | 1 | ixc 10: H-factor on ISS04 confinement -- PROCESS's closure knob, a belief here | ixc |
| `.physics.tauratio` | 1 | He/pellet particle-confinement ratio (1.0) |  |
| `.stellarator.bmn` | 0.001 | residual radial field perturbation (ripple) | IN.DAT |
| `.stellarator.f_asym` | 1.1 | divertor heat-load peaking factor | IN.DAT |
| `.stellarator.f_w` | 0.5 | island size fraction factor | IN.DAT |
| `.stellarator.fdivwet` | 0.3333 | wetted fraction of the divertor area (1/3) |  |
| `.stellarator.flpitch` | 0.001 | field-line pitch at the target (rad) | IN.DAT |
| `.tfcoil.dcond[0]` | 6080 | material density / resistivity constant |  |
| `.tfcoil.den_tf_coil_case` | 8000 | material density / resistivity constant |  |
| `.tfcoil.den_tf_wp_turn_insulation` | 1800 | material density / resistivity constant |  |
| `.tfcoil.rho_tf_bus` | 1.86e-08 | material density / resistivity constant |  |
| `.vacuum.outgrat_fw` | 1.3e-08 | first-wall outgassing rate |  |
| `^stated.tfcoil.eff_tf_cryo` | 0.13 | cryoplant efficiency vs Carnot (0.13); a belief the port pins |  |

Cost-model coefficients (73; all belief, economic; `uq.py` samples only `ucsc`,
`discount_rate`, `life_plant`, `f_t_plant_available`):

| path | value | meaning |
|---|---|---|
| `.costs.UCAD` | 180 | unit cost for administration buildings |
| `.costs.UCAF` | 1.5e+06 | unit cost for aux facility power equipment |
| `.costs.UCAHTS` | 31 | unit cost for aux heat transport equipment |
| `.costs.UCAP` | 17 | unit cost of auxiliary transformer |
| `.costs.UCBPMP` | 2.92e+05 | vacuum system backing pump cost |
| `.costs.UCCO` | 350 | unit cost for control buildings |
| `.costs.UCCPMP` | 3.9e+05 | vacuum system cryopump cost |
| `.costs.UCCR` | 460 | unit cost for cryogenic building |
| `.costs.UCDGEN` | 1.7e+06 | cost per 8 MW diesel generator |
| `.costs.UCDTC` | 13 | detritiation, air cleanup cost |
| `.costs.UCDUCT` | 4.22e+04 | vacuum system duct cost |
| `.costs.UCEL` | 380 | unit cost for electrical equipment building |
| `.costs.UCFPR` | 4.4e+07 | cost of 60g/day tritium processing unit |
| `.costs.UCFWA` | 6e+04 | first wall armour cost |
| `.costs.UCFWPS` | 1e+07 | first wall passive stabiliser cost |
| `.costs.UCFWS` | 5.3e+04 | first wall structure cost |
| `.costs.UCGSS` | 35 | cost of reactor structure |
| `.costs.UCINT` | 35 | superconductor intercoil structure cost |
| `.costs.UCLV` | 16 | low voltage system cost |
| `.costs.UCMB` | 260 | unit cost for reactor maintenance building |
| `.costs.UCNBV` | 1000 | cost of nuclear building ventilation |
| `.costs.UCPHX` | 15 | primary heat transport cost |
| `.costs.UCPP` | 48 | cost of primary power transformers |
| `.costs.UCSH` | 115 | cost of shops and warehouses |
| `.costs.UCSWYD` | 1.84e+07 | switchyard equipment costs |
| `.costs.UCTFDR` | 0.000175 | cost of TF coil dump resistors |
| `.costs.UCTFGR` | 5000 | additional cost of TF coil dump resistors |
| `.costs.UCTFIC` | 1e+04 | cost of TF coil instrumentation and control |
| `.costs.UCTPMP` | 1.1e+05 | cost of turbomolecular pump |
| `.costs.UCTR` | 370 | cost of tritium building |
| `.costs.UCVALV` | 3.9e+05 | vacuum system valve cost |
| `.costs.UCVDSH` | 26 | vacuum duct shield cost |
| `.costs.UCVIAC` | 1.3e+06 | vacuum system instrumentation and control cost |
| `.costs.UCWS` | 460 | cost of active assembly shop |
| `.costs.cconfix` | 80 | fixed cost of superconducting cable |
| `.costs.cconshtf` | 75 | cost of TF coil steel conduit/sheath |
| `.costs.cland` | 19.2 | cost of land |
| `.costs.csi` | 16 | allowance for site costs |
| `.costs.cturbb` | 38 | cost of turbine building |
| `.costs.ucblbe` | 260 | unit cost for blanket beryllium |
| `.costs.ucblli2o` | 600 | unit cost for blanket Li_2O |
| `.costs.ucblss` | 90 | unit cost for blanket stainless steel |
| `.costs.ucblvd` | 280 | unit cost for blanket vanadium |
| `.costs.ucbus` | 0.123 | cost of aluminium bus for TF coil |
| `.costs.uccase` | 50 | cost of superconductor case |
| `.costs.uccry` | 9.3e+04 | heat transport system cryoplant costs |
| `.costs.uccryo` | 32 | unit cost for vacuum vessel |
| `.costs.uccu` | 75 | unit cost for copper in superconducting cable |
| `.costs.ucdiv` | 5e+05 | cost of divertor blade |
| `.costs.ucech` | 3 | ECH system cost |
| `.costs.ucf1` | 2.23e+07 | cost of fuelling system |
| `.costs.ucfuel` | 3.45 | unit cost of D-T fuel |
| `.costs.uche3` | 1e+06 | cost of helium-3 |
| `.costs.uchrs` | 8.79e+07 | cost of heat rejection system |
| `.costs.uchts` | [15.3, 19.1] | cost of heat transport system equipment per loop |
| `.costs.uciac` | 1.5e+08 | cost of instrumentation, control & diagnostics |
| `.costs.ucich` | 3 | ICH system cost |
| `.costs.uclh` | 3.3 | lower hybrid system cost |
| `.costs.ucme` | 3e+08 | cost of maintenance equipment |
| `.costs.ucmisc` | 2.5e+07 | miscellaneous plant allowance |
| `.costs.ucnbi` | 3.3 | NBI system cost |
| `.costs.ucoam` | [68.8, 68.8, 68.8, 74.4] | annual cost of operation and maintenance |
| `.costs.ucpens` | 32 | penetration shield cost |
| `.costs.ucrb` | 400 | cost of reactor building |
| `.costs.ucsc` | array(9) | - |
| `.costs.ucshld` | 32 | cost of shield structural steel |
| `.costs.uctfbr` | 1.22 | cost of TF coil breakers |
| `.costs.uctfbus` | 100 | cost of TF coil bus |
| `.costs.uctfps` | 24 | cost of TF coil power supplies |
| `.costs.uctfsw` | 1 | cost of TF coil slow dump switches |
| `.costs.ucturb` | [2.3e+08, 2.45e+08] | cost of turbine plant equipment |
| `.costs.ucwindtf` | 480 | cost of TF coil superconductor windings |
| `.costs.ucwst` | [0, 3.94, 5.91, 7.88] | cost of waste disposal |

### 3.2 Build decisions (76)

Flag **[optimiser ought to own]**: every `.build.dr_*` / `dz_*` / gap, `dr_tf_nose_case`, the four
turn / insulation dimensions, `f_a_tf_turn_cable_space_extra_void`, `tmargmin` and the blanket
composition are fixed by the file although they are exactly the thicknesses and coil parameters
a design optimisation decides (PROCESS has `ixc` ids for several: 31, 57, 58, 61, 73, 74, 93,
94). The 15 `.build.*` entries and `dr_fw_wall` / `radius_fw_channel` set `required_radial_space`
and so c83, the one constraint that is active and split 50/50 in the Monte Carlo.

| path | value | meaning | set by |
|---|---|---|---|
| `.build.dr_blkt_inboard` | 0.41 | radial-build thickness fixed by the file; ought to be an optimiser variable (ixc 73/74/93/94/61/31 exist for some) | IN.DAT |
| `.build.dr_blkt_outboard` | 0.63 | radial-build thickness fixed by the file; ought to be an optimiser variable (ixc 73/74/93/94/61/31 exist for some) | IN.DAT |
| `.build.dr_cryostat` | 0.05 | radial-build thickness fixed by the file; ought to be an optimiser variable (ixc 73/74/93/94/61/31 exist for some) | IN.DAT |
| `.build.dr_fw_plasma_gap_inboard` | 0.3 | radial-build thickness fixed by the file; ought to be an optimiser variable (ixc 73/74/93/94/61/31 exist for some) | IN.DAT |
| `.build.dr_fw_plasma_gap_outboard` | 0.3 | radial-build thickness fixed by the file; ought to be an optimiser variable (ixc 73/74/93/94/61/31 exist for some) | IN.DAT |
| `.build.dr_shld_blkt_gap` | 0.05 | radial-build thickness fixed by the file; ought to be an optimiser variable (ixc 73/74/93/94/61/31 exist for some) |  |
| `.build.dr_shld_inboard` | 0.3 | radial-build thickness fixed by the file; ought to be an optimiser variable (ixc 73/74/93/94/61/31 exist for some) | IN.DAT |
| `.build.dr_shld_outboard` | 0.3 | radial-build thickness fixed by the file; ought to be an optimiser variable (ixc 73/74/93/94/61/31 exist for some) | IN.DAT |
| `.build.dr_shld_vv_gap_inboard` | 0.25 | radial-build thickness fixed by the file; ought to be an optimiser variable (ixc 73/74/93/94/61/31 exist for some) | IN.DAT |
| `.build.dr_vv_inboard` | 0.6 | radial-build thickness fixed by the file; ought to be an optimiser variable (ixc 73/74/93/94/61/31 exist for some) | IN.DAT |
| `.build.dr_vv_outboard` | 0.6 | radial-build thickness fixed by the file; ought to be an optimiser variable (ixc 73/74/93/94/61/31 exist for some) | IN.DAT |
| `.build.dz_shld_vv_gap` | 0.163 | radial-build thickness fixed by the file; ought to be an optimiser variable (ixc 73/74/93/94/61/31 exist for some) |  |
| `.build.dz_vv_lower` | 0.6 | radial-build thickness fixed by the file; ought to be an optimiser variable (ixc 73/74/93/94/61/31 exist for some) | IN.DAT |
| `.build.dz_vv_upper` | 0.6 | radial-build thickness fixed by the file; ought to be an optimiser variable (ixc 73/74/93/94/61/31 exist for some) | IN.DAT |
| `.build.gapomin` | 0.25 | radial-build thickness fixed by the file; ought to be an optimiser variable (ixc 73/74/93/94/61/31 exist for some) | IN.DAT |
| `.buildings.admv` | 1e+05 | plant civil layout (volumes, clearances, factors); low leverage on coe |  |
| `.buildings.clh2` | 15 | plant civil layout (volumes, clearances, factors); low leverage on coe |  |
| `.buildings.conv` | 6e+04 | plant civil layout (volumes, clearances, factors); low leverage on coe |  |
| `.buildings.dz_tf_cryostat` | 2.5 | plant civil layout (volumes, clearances, factors); low leverage on coe |  |
| `.buildings.fndt` | 2 | plant civil layout (volumes, clearances, factors); low leverage on coe |  |
| `.buildings.hccl` | 5 | plant civil layout (volumes, clearances, factors); low leverage on coe |  |
| `.buildings.hcwt` | 1.5 | plant civil layout (volumes, clearances, factors); low leverage on coe |  |
| `.buildings.mbvfac` | 2.8 | plant civil layout (volumes, clearances, factors); low leverage on coe |  |
| `.buildings.pfbldgm3` | 2e+04 | plant civil layout (volumes, clearances, factors); low leverage on coe |  |
| `.buildings.pibv` | 2e+04 | plant civil layout (volumes, clearances, factors); low leverage on coe |  |
| `.buildings.rbrt` | 1 | plant civil layout (volumes, clearances, factors); low leverage on coe |  |
| `.buildings.rbvfac` | 1.6 | plant civil layout (volumes, clearances, factors); low leverage on coe |  |
| `.buildings.rbwt` | 2 | plant civil layout (volumes, clearances, factors); low leverage on coe |  |
| `.buildings.row` | 4 | plant civil layout (volumes, clearances, factors); low leverage on coe |  |
| `.buildings.rxcl` | 4 | plant civil layout (volumes, clearances, factors); low leverage on coe |  |
| `.buildings.shmf` | 0.5 | plant civil layout (volumes, clearances, factors); low leverage on coe |  |
| `.buildings.shov` | 1e+05 | plant civil layout (volumes, clearances, factors); low leverage on coe |  |
| `.buildings.stcl` | 3 | plant civil layout (volumes, clearances, factors); low leverage on coe |  |
| `.buildings.trcl` | 1 | plant civil layout (volumes, clearances, factors); low leverage on coe |  |
| `.buildings.triv` | 4e+04 | plant civil layout (volumes, clearances, factors); low leverage on coe |  |
| `.buildings.wgt` | 5e+05 | plant civil layout (volumes, clearances, factors); low leverage on coe |  |
| `.buildings.wgt2` | 1e+05 | plant civil layout (volumes, clearances, factors); low leverage on coe |  |
| `.buildings.wsvfac` | 1.9 | plant civil layout (volumes, clearances, factors); low leverage on coe |  |
| `.costs.life_plant` | 40 | plant life (y): a design/financing choice (uq.py samples it as a belief) | IN.DAT |
| `.divertor.anginc` | 0.03 | field-line angle of incidence on the target (rad): a divertor geometry choice | IN.DAT |
| `.divertor.dx_div_plate` | 0.035 | divertor plate thickness / coolant fraction / count |  |
| `.divertor.f_vol_div_coolant` | 0.3 | divertor plate thickness / coolant fraction / count |  |
| `.divertor.n_divertors` | 1 | divertor plate thickness / coolant fraction / count |  |
| `.fwbs.dr_fw_wall` | 0.003 | first-wall channel wall thickness (m) (uq.py samples it as manufacturing scatter) |  |
| `.fwbs.dr_pf_cryostat` | 0.5 | coil-to-cryostat radial clearance (m) |  |
| `.fwbs.f_a_fw_outboard_hcd` | 0 | first-wall area fraction taken by H&CD/diagnostics |  |
| `.fwbs.fblbe` | 0.3663 | blanket composition by volume (Be / Li2O / steel / V) | IN.DAT |
| `.fwbs.fblli2o` | 0.1491 | blanket composition by volume (Be / Li2O / steel / V) | IN.DAT |
| `.fwbs.fblss` | 0.0985 | blanket composition by volume (Be / Li2O / steel / V) | IN.DAT |
| `.fwbs.fblvd` | 0 | blanket composition by volume (Be / Li2O / steel / V) | IN.DAT |
| `.fwbs.fhole` | 0 | first-wall hole fraction (uq.py samples it) | IN.DAT |
| `.fwbs.i_blkt_coolant_type` | 1 | blanket coolant type (switch): a coolant choice |  |
| `.fwbs.radius_fw_channel` | 0.006 | first-wall coolant channel radius (m) |  |
| `.fwbs.vfshld` | 0.4 | shield coolant void fraction | IN.DAT |
| `.physics.b_plasma_toroidal_on_axis` | 5.5 | ixc 2: field on axis (T) | ixc |
| `.physics.kappa` | 1.792 | elongation 1.79: a property of the HELIAS coil set (fixed configuration constant, not chosen freely) |  |
| `.physics.rmajor` | 20 | ixc 3: major radius (m) | ixc |
| `.stellarator.f_st_coil_aspect` | 1 | coil aspect-ratio scaling factor (1.0; ixc 176 exists, commented out) | IN.DAT |
| `.stellarator.iotabar` | 1 | rotational transform / shear / resonance numbers: properties of the fixed HELIAS configuration | IN.DAT |
| `.stellarator.m_res` | 5 | rotational transform / shear / resonance numbers: properties of the fixed HELIAS configuration |  |
| `.stellarator.max_gyrotron_frequency` | 1e+09 | maximal gyrotron frequency (Hz): heating-system specification (sets the ECRH density limit) |  |
| `.stellarator.n_res` | 5 | rotational transform / shear / resonance numbers: properties of the fixed HELIAS configuration |  |
| `.stellarator.shear` | 0.5 | rotational transform / shear / resonance numbers: properties of the fixed HELIAS configuration | IN.DAT |
| `.tfcoil.dr_tf_nose_case` | 0.06 | coil case nose thickness (m); ixc 57 exists -- fixed by the file, ought to be optimised (side/plasma case follow by formula) | IN.DAT |
| `.tfcoil.dx_tf_turn_general` | 0.056 | turn / conduit / insulation dimensions (m); ixc 58 exists for the steel -- fixed by the file (uq.py samples dx_tf_wp_insulation) | IN.DAT |
| `.tfcoil.dx_tf_turn_insulation` | 0.002 | turn / conduit / insulation dimensions (m); ixc 58 exists for the steel -- fixed by the file (uq.py samples dx_tf_wp_insulation) | IN.DAT |
| `.tfcoil.dx_tf_turn_steel` | 0.0012 | turn / conduit / insulation dimensions (m); ixc 58 exists for the steel -- fixed by the file (uq.py samples dx_tf_wp_insulation) | IN.DAT |
| `.tfcoil.dx_tf_wp_insulation` | 0.01 | turn / conduit / insulation dimensions (m); ixc 58 exists for the steel -- fixed by the file (uq.py samples dx_tf_wp_insulation) | IN.DAT |
| `.tfcoil.f_a_tf_turn_cable_copper` | 0.7 | ixc 59: copper fraction of the cable | ixc |
| `.tfcoil.f_a_tf_turn_cable_space_extra_void` | 0.3 | cable void (He) fraction | IN.DAT |
| `.tfcoil.i_tf_sc_mat` | 1 | superconductor choice (1 = ITER Nb3Sn) / conductor type (1 = SC): a material choice spelt as a switch | IN.DAT |
| `.tfcoil.i_tf_sup` | 1 | superconductor choice (1 = ITER Nb3Sn) / conductor type (1 = SC): a material choice spelt as a switch |  |
| `.tfcoil.t_tf_quench_detection` | 0.5 | quench detection + activation delay (s): protection-system spec | IN.DAT |
| `.tfcoil.t_tf_superconductor_quench` | 35 | ixc 56: fast discharge (dump) time (s) | ixc |
| `.tfcoil.tmargmin` | 1.5 | minimum superconductor temperature margin (K): a designer-set margin | IN.DAT |
| `^stated.times.t_plant_pulse_burn` | 3.16e+07 | steady-state burn = 1 y; the pulse-schedule convention of a stellarator plant |  |

### 3.3 Operating variables (16)

| path | value | meaning | set by |
|---|---|---|---|
| `.costs.f_t_plant_available` | 0.75 | plant availability; borderline (a target the operator meets, or a belief about reliability -- uq.py samples it) | IN.DAT |
| `.current_drive.p_hcd_primary_extra_heat_mw` | 0 | auxiliary heating power (MW); 0 because the plasma is declared ignited (ixc 11 would free it) | IN.DAT |
| `.divertor.tdiv` | 5 | divertor plasma temperature (eV): a detachment/operating target; borderline (uq.py samples it as a belief) | IN.DAT |
| `.physics.f_nd_alpha_thermal_electron` | 0.1 | ixc 109: thermal alpha fraction n_alpha/n_e -- set by fuelling / He exhaust | ixc |
| `.physics.f_plasma_fuel_deuterium` | 0.5 | fuel mix D/T/He3 (0.5/0.5/0): a fuelling choice |  |
| `.physics.f_plasma_fuel_helium3` | 0 | fuel mix D/T/He3 (0.5/0.5/0): a fuelling choice |  |
| `.physics.f_plasma_fuel_tritium` | 0.5 | fuel mix D/T/He3 (0.5/0.5/0): a fuelling choice |  |
| `.physics.nd_plasma_electrons_vol_avg` | 2e+20 | ixc 6: volume-averaged electron density | ixc |
| `.physics.temp_plasma_electron_vol_avg_kev` | 7 | ixc 4: volume-averaged electron temperature (keV) | ixc |
| `.stellarator.f_rad` | 0.85 | radiated fraction in the SOL (0.85): impurity seeding sets it, so operating; borderline (uq.py samples it as a belief) | IN.DAT |
| `.tfcoil.temp_tf_cryo` | 4.5 | coil / helium operating temperature (K): set by the cryoplant; borderline build vs operating (uq.py samples temp_tf_cryo) | IN.DAT |
| `.tfcoil.tftmp` | 4.5 | coil / helium operating temperature (K): set by the cryoplant; borderline build vs operating (uq.py samples temp_tf_cryo) | IN.DAT |
| `.times.t_plant_pulse_dwell` | 1800 | pulse schedule (dwell 1800 s, ramp 10 s); nearly inert against a 1-y burn |  |
| `.times.t_plant_pulse_fusion_ramp` | 10 | pulse schedule (dwell 1800 s, ramp 10 s); nearly inert against a 1-y burn |  |
| `.vacuum.pres_div_chamber_burn` | 0.36 | chamber pressures during burn / dwell (Pa) |  |
| `.vacuum.pres_vv_chamber_base` | 0.0005 | chamber pressures during burn / dwell (Pa) |  |

### 3.4 Numerics, switches and inert reads (62)

Includes the 6 `^guess` starts, the 16 `^stated` pins, and every tokamak-only field (beam, PF,
centrepost, plasma current) the stellarator graph reads at zero.

| path | value | meaning | set by |
|---|---|---|---|
| `^guess.fwbs.f_ster_div_single` | 0.115 | solver start for a driven unknown; not a decision |  |
| `^guess.physics.fusden_alpha_total` | 0 | solver start for a driven unknown; not a decision |  |
| `^guess.physics.proton_rate_density` | 0 | solver start for a driven unknown; not a decision |  |
| `^guess.physics.temp_plasma_ion_vol_avg_kev` | 12.9 | solver start for a driven unknown; not a decision |  |
| `^guess.power.delta_eta` | 0 | solver start for a driven unknown; not a decision |  |
| `^guess.vacuum.d_duct` | n/a | solver start for a driven unknown; not a decision |  |
| `.costs.ifueltyp` | 0 | cost-model switch | IN.DAT |
| `.costs.ireactor` | 1 | cost-model switch | IN.DAT |
| `.costs.lsa` | 2 | cost-model switch | IN.DAT |
| `.current_drive.e_beam_kev` | 1000 | NBI parameter; no beam on this machine, inert |  |
| `.current_drive.f_beam_tritium` | 1e-06 | NBI parameter; no beam on this machine, inert |  |
| `.current_drive.i_hcd_primary` | 5 | primary H&CD switch (ECRH); consumed as a switch |  |
| `.current_drive.p_beam_injected_mw` | 0 | NBI / lower-hybrid injected power; neither exists here, zero and inert |  |
| `.current_drive.p_beam_orbit_loss_mw` | 0 | NBI parameter; no beam on this machine, inert |  |
| `.current_drive.p_beam_shine_through_mw` | 0 | NBI parameter; no beam on this machine, inert |  |
| `.current_drive.p_hcd_beam_injected_total_mw` | 0 | NBI parameter; no beam on this machine, inert |  |
| `.current_drive.p_hcd_lowhyb_injected_total_mw` | 0 | NBI / lower-hybrid injected power; neither exists here, zero and inert |  |
| `.fwbs.f_nuc_pow_bz_liq` | 0.66 | dual-coolant blanket parameter; no liquid breeder here, inert |  |
| `.fwbs.outlet_temp_liq` | 720 | dual-coolant blanket parameter; no liquid breeder here, inert |  |
| `.heat_transport.i_shld_primary_heat` | 1 | power-flow model switch |  |
| `.heat_transport.ipowerflow` | 1 | power-flow model switch |  |
| `.heat_transport.p_blkt_breeder_pump_mw` | 0 | secondary breeder pump power; no liquid breeder, inert (0) |  |
| `.ife.ife` | 0 | IFE switch (0) |  |
| `.impurity_radiation.f_p_plasma_core_rad_reduction` | 1 | radiation-model definition (core radius 0.6, core-radiation fraction subtracted 1.0) | IN.DAT |
| `.impurity_radiation.radius_plasma_core_norm` | 0.6 | radiation-model definition (core radius 0.6, core-radiation fraction subtracted 1.0) | IN.DAT |
| `.pf_coil.m_pf_coil_max` | 0 | PF/CS quantity; a stellarator has none, zero and inert |  |
| `.pf_coil.p_pf_electric_supplies_mw` | 0 | PF/CS quantity; a stellarator has none, zero and inert |  |
| `.pf_coil.r_pf_coil_outer_max` | 0 | PF/CS quantity; a stellarator has none, zero and inert |  |
| `.pf_power.ensxpfm` | 0 | PF/CS quantity; a stellarator has none, zero and inert |  |
| `.pf_power.srcktpm` | 0 | PF/CS quantity; a stellarator has none, zero and inert |  |
| `.physics.alphaj` | 1 | current-profile index; no plasma current, inert |  |
| `.physics.beta_beam` | 0 | beam beta; no beam, inert |  |
| `.physics.burnup_in` | 0 | 0 = compute burn-up / Coulomb log internally (switch-valued sentinel) |  |
| `.physics.dlamie` | 0 | 0 = compute burn-up / Coulomb log internally (switch-valued sentinel) |  |
| `.physics.itart` | 0 | spherical-tokamak switch (0) |  |
| `.physics.p_beam_alpha_mw` | 0 | beam / ohmic / current quantity; a stellarator has none, zero and inert |  |
| `.physics.p_plasma_ohmic_mw` | 0 | beam / ohmic / current quantity; a stellarator has none, zero and inert |  |
| `.physics.pden_plasma_ohmic_mw` | 0 | beam / ohmic / current quantity; a stellarator has none, zero and inert |  |
| `.physics.plasma_current` | 0 | beam / ohmic / current quantity; a stellarator has none, zero and inert |  |
| `.tfcoil.a_tf_wp_coolant_channels` | 0 | WP coolant channel area, 0 (unused slot) |  |
| `.tfcoil.temp_cp_coolant_inlet` | 313.1 | centrepost coolant inlet T; no centrepost, inert |  |
| `.vacuum.ceff_i` | n/a | constant of the old vacuum-duct model (not on the DataStructure) |  |
| `.vacuum.i_vac_pump_dwell` | 0 | vacuum pump switch (type / dwell pumping) |  |
| `.vacuum.i_vacuum_pump_type` | 1 | vacuum pump switch (type / dwell pumping) |  |
| `.vacuum.l1` | n/a | constant of the old vacuum-duct model (not on the DataStructure) |  |
| `.vacuum.l2` | n/a | constant of the old vacuum-duct model (not on the DataStructure) |  |
| `.vacuum.l3` | n/a | constant of the old vacuum-duct model (not on the DataStructure) |  |
| `.vacuum.xmult_i` | n/a | constant of the old vacuum-duct model (not on the DataStructure) |  |
| `^stated.build.dr_cs` | 0 | pin of a PROCESS default the port restates; inert on this machine |  |
| `^stated.build.dr_cs_tf_gap` | 0 | pin of a PROCESS default the port restates; inert on this machine |  |
| `^stated.buildings.esbldgm3` | 0 | pin of a PROCESS default the port restates; inert on this machine |  |
| `^stated.costs.c2253` | 0 | pin of a PROCESS default the port restates; inert on this machine |  |
| `^stated.physics.nd_plasma_pedestal_electron` | 0 | pin of a PROCESS default the port restates; inert on this machine |  |
| `^stated.physics.nd_plasma_separatrix_electron` | 0 | pin of a PROCESS default the port restates; inert on this machine |  |
| `^stated.physics.radius_plasma_pedestal_density_norm` | 1 | pin of a PROCESS default the port restates; inert on this machine |  |
| `^stated.physics.radius_plasma_pedestal_temp_norm` | 1 | pin of a PROCESS default the port restates; inert on this machine |  |
| `^stated.physics.tbeta` | 2 | pin of a PROCESS default the port restates; inert on this machine |  |
| `^stated.physics.temp_plasma_pedestal_kev` | 0 | pin of a PROCESS default the port restates; inert on this machine |  |
| `^stated.physics.temp_plasma_separatrix_kev` | 0 | pin of a PROCESS default the port restates; inert on this machine |  |
| `^stated.times.t_plant_pulse_coil_precharge` | 0 | pin of a PROCESS default the port restates; inert on this machine |  |
| `^stated.times.t_plant_pulse_plasma_current_ramp_down` | 0 | pin of a PROCESS default the port restates; inert on this machine |  |
| `^stated.times.t_plant_pulse_plasma_current_ramp_up` | 0 | pin of a PROCESS default the port restates; inert on this machine |  |

### 3.5 Derived quantities read from outside (13, all zero: missing producers)

These are computed by PROCESS's full pipeline but no node of the stellarator graph owns them, so
the port reads them from the cold DataStructure. Every one is 0 here and inert; they are listed
because a reader of `boundary_inputs` would take them for decisions.

| path | value | meaning | set by |
|---|---|---|---|
| `.build.r_tf_inboard_mid` | 0 | tokamak-only field, unset (0); read by buildings.tf_coil_envelope -- a missing producer, inert |  |
| `.costs.c2214` | 0 | cost accumulator, zero here: a missing producer of the 1990 cost model, inert |  |
| `.costs.c2222` | 0 | cost accumulator, zero here: a missing producer of the 1990 cost model, inert |  |
| `.costs.c2252` | 0 | cost accumulator, zero here: a missing producer of the 1990 cost model, inert |  |
| `.costs.cplife` | 0 | centrepost lifetime, tokamak-only (0) |  |
| `.fwbs.p_cp_shield_nuclear_heat_mw` | 0 | zero, read by `power.component_thermal_powers`: centrepost nuclear heat / divertor radiation no stellarator node produces (missing producer) |  |
| `.fwbs.p_div_rad_total_mw` | 0 | zero, read by `power.component_thermal_powers`: centrepost nuclear heat / divertor radiation no stellarator node produces (missing producer) |  |
| `.heat_transport.peakmva` | 0 | peak MVA, zero: missing producer, inert |  |
| `.physics.beta_thermal_vol_avg` | 0 | zero, read by Constraint24 but unused on the stellarator branch (istell != 0 takes beta_total): missing producer, inert |  |
| `.physics.beta_toroidal_vol_avg` | 0 | zero, read by Constraint24 but unused on the stellarator branch (istell != 0 takes beta_total): missing producer, inert |  |
| `.tfcoil.m_tf_bus` | 0 | bus mass / leg resistance / supply power: zero, missing producers of the cost & power models, inert |  |
| `.tfcoil.res_tf_leg` | 0 | bus mass / leg resistance / supply power: zero, missing producers of the cost & power models, inert |  |
| `.tfcoil.tfcmw` | 0 | bus mass / leg resistance / supply power: zero, missing producers of the cost & power models, inert |  |

### 3.6 Model outputs that are in reality build decisions

Not on the boundary: the graph derives them by formula from `rmajor`, `b_plasma_toroidal_on_axis`
and the HELIAS 5-B configuration file (`stellarator_helias.stella_conf.json`), where a designer
would choose them.

| owned place | producing node | how it is derived | what a designer would decide |
|---|---|---|---|
| `.physics.aspect`, `.physics.rminor`, `.physics.eps` | `default_aspect_ratio`, `stellarator_scaling_factors` | `aspect = aspect_ref` (12.3) of the configuration; `rminor = rmajor / aspect` | the aspect ratio (ixc 1 is commented out) |
| `.tfcoil.n_tf_coils` | `stellarator_scaling_factors` | `coilspermodule x symmetry` = 50, scaled | number of coils |
| `.stellarator.r_coil_major`, `r_coil_minor` | `stellarator_scaling_factors` | `coil_rmajor/rminor` of the configuration x `f_st_rmajor`, `f_st_coil_aspect` | coil size and plasma-coil distance |
| `.physics.kappa` (boundary input, 1.79) | -- | frozen configuration constant | -- (fixed with the coil set) |
| `.stellarator.coilcurrent`, `f_st_i_total` | `coils.coil_current` | `i0 x f_st_b x f_st_rmajor / f_st_n_coils` | coil current follows B and R |
| `.tfcoil.dr_tf_wp_with_insulation`, `dx_tf_wp_primary_toroidal`, `j_tf_wp`, `n_tf_coil_turns`, `a_tf_wp_*` | `coils.winding_pack_intersect_inputs` + the `wp_width_r_min` root find + `winding_pack_total_size_post` | the WP is sized so that `j_tf_wp = f_j_tf_wp_critical_max x j_critical(B_peak, tftmp + tmargmin)` | WP cross-section and current density (ixc 13/14-style choices in tokamak files) |
| `.tfcoil.dr_tf_plasma_case`, `dx_tf_side_case_min`, `.build.dr_tf_inboard` | `coils.coil_casing`, `coil_radial_thickness` | fixed fractions of `dr_tf_nose_case` | case thicknesses |
| `.build.dr_fw_inboard`, `dr_fw_outboard` | `stellarator.build` | `2 x radius_fw_channel + 2 x dr_fw_wall` (= 0.018 m) | first-wall thickness |
| `.tfcoil.e_tf_magnetic_stored_total_gj`, `.build.z_tf_inside_half`, `len_tf_coil`, `tfcryoarea` | `coils.*` | configuration constants scaled by `r_coil_minor` | -- (follow the coil set) |
| `.fwbs.life_fw_fpy` | `fwbs.fw_blanket_shield_geometry` | `abktflnc / pflux_fw_neutron_mw` | blanket replacement schedule (a belief x an output) |
| `.times.t_plant_pulse_burn` (`^stated`, 3.156e7 s) | `initialisation.stellarator_pulse_times` | 1 year, pinned | steady-state operation |
| `.build.dr_cs`, `dr_cs_tf_gap` (`^stated`, 0) | `initialisation.stellarator_solenoid_absent` | no solenoid | -- |

## 4. Recommended two-stage split for OUU

Outer (first stage, decided once, before the machine exists): **`rmajor`, `b_plasma_toroidal_on_axis`,
`f_a_tf_turn_cable_copper`, `t_tf_superconductor_quench`** -- the 4 build `ixc`. Optionally add
the frozen build decisions of §3.2 that c83 and c35 actually depend on (`dr_blkt_*`,
`dr_shld_*`, `dr_vv_*`, `dr_tf_nose_case`, `dx_tf_wp_insulation`, `tmargmin`) if the paper wants
the radial build owned by the optimiser rather than the file.

Recourse (second stage, per sample, given the realised beliefs):

| knob | closes | why this pairing |
|---|---|---|
| `nd_plasma_electrons_vol_avg` (ixc 6) | c2, power balance | what `uq.py` / `ouu.py` already do (`PAIRING`); 22-node cycle, scaled sensitivity 0.33, root exists at every outer iterate |
| `f_nd_alpha_thermal_electron` (ixc 109) | c16, net electric power = 1000 MW | `close_conditions.PAIRINGS`' choice for c16: sensitivity 0.42, root exists from cold; the alternative `temp_plasma_electron_vol_avg_kev` (sensitivity -1.9) is stronger but VMCON's path with T_e at its 3 keV bound loses the c2 root in `hfact` |
| `temp_plasma_electron_vol_avg_kev` (ixc 4) | free operating variable | either a second-stage decision variable (minimise expected coe per sample subject to the physics inequalities) or held at the nominal; it is the one operating knob with no equation to close once c2 and c16 are taken |

Beliefs (sampled): `hfact` (sigma 0.10 lognormal in `ouu.belief_table`), plus `uq.INPUTS`'
physics rows (`alphan`, `alphat`, `f_p_alpha_plasma_deposited`, `f_temp_plasma_ion_electron`,
`f_sync_reflect`, tungsten fraction, `bmn`, `f_asym`, `f_rad`, `f_p_blkt_multiplication`,
`declfw`, `declblkt`, `fhole`, `dr_fw_wall`, `temp_tf_cryo`, `dx_tf_wp_insulation`,
`f_j_tf_wp_critical_max`, `eta_turbine`, `eta_ecrh_injector_wall_plug`,
`f_p_fw_coolant_pump_total_heat`, `tdiv`) and the four economic ones. Note three of those
(`dr_fw_wall`, `dx_tf_wp_insulation`, `fhole`) are build decisions sampled as manufacturing
scatter, and `tdiv`, `f_rad`, `temp_tf_cryo` are operating variables sampled as beliefs -- say so
in the paper or move them.

Constraint set after recourse:

| stays probabilistic (chance / CVaR) | deterministic given the outer design (keep at nominal) | closed per sample |
|---|---|---|
| c24 beta (p 0.58), c8 neutron wall load (0.29), c67 radiation wall load (0.26), c62 He/energy confinement (0.16), c18 divertor heat load (0.11), c35 J_wp quench protection (0.50, only if `temp_tf_cryo` / `dx_tf_wp_insulation` / `f_j_tf_wp_critical_max` stay sampled), c17 radiation fraction (0.007) | c82 toroidal build, c83 radial build (deterministic once `dr_fw_wall` is fixed), c32 WP stress, c34 dump voltage, c65 VV stress (all functions of the 4 outer `ixc`, the file's build thicknesses and `t_quench`; inactive by 0.7-1.0 except c83) | c2 by the density; c16 by `f_nd_alpha_thermal_electron` (or reported as an inequality with CVaR, which `ouu.py` found infeasible at alpha 0.9) |

Ambiguities decided here: `f_j_tf_wp_critical_max`, `tmargmin`, `tftmp`/`temp_tf_cryo` are
margins a designer sets (build) but `uq.py` samples the first and third -- classified as the
task's list says (allowed current fraction = belief; margin = build; coil temperature = operating,
borderline). `f_t_plant_available` is operating (borderline). `life_plant` is build (a plant-life
choice), although `uq.py` samples it. Material constants (`den_steel`, `dcond`, `rho_tf_bus`,
atomic data tables) are counted as beliefs with negligible uncertainty rather than as a sixth
kind. `kappa`, `iotabar`, `shear`, `m_res`, `n_res` are build decisions only in the sense that
they are fixed with the HELIAS coil set; nothing in this run can move them. `c16` is stated as an
equality in the file, so it is classed as a closure here; physically it is a plant requirement.
