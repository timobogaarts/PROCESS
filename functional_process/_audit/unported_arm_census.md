# Which `UNPORTED` refusals a file that says nothing would hit — 2026-08-31

**Analysis only.** Nothing under `functional_process/*.py`, `process/*.py` or `tests/` was
changed by this pass.

## The question, and why defaults are the right instrument

`indat.UNPORTED` holds **219 rows over 50 switch axes**, and every one of them was shaped by
the same eight regression files. Sampling more files from this repo cannot correct that bias
(§0 below verifies the claim). But an `IN.DAT` does not choose most of its switches at all:
it selects PROCESS's *default* for everything it is silent about, and most files are silent
about most switches. So the measurement that actually predicts what an unseen file hits is:

> for each `UNPORTED` `(switch, value)` row, is that value the switch's **default**?

An axis whose default is refused bites **every** file silent on it. An axis whose default is
ported is only reached by a file that asks.

## 0. The premise, checked: the repo's other inputs carry no new arms

**[measured]** Parsed every `name = <integer>` assignment out of the five
`examples/data/*IN.DAT`, three `tests/integration/data/*IN.DAT` and one
`tests/unit/data/*IN.DAT`, and differenced the resulting `(name, value)` set against the
union over the eight `tests/regression/input_files/*IN.DAT`.

The eight regression files carry **159 distinct integer `switch = value` pairs**. Across all
nine other input files the pairs *not* already in that set are exactly four:

| pair | file | what it is |
|---|---|---|
| `isweep = 6` | `examples/data/scan_example_file_IN.DAT` | scan machinery |
| `ifispact = 0` | `tests/integration/data/ref_IN.DAT` | a legacy flag `process/` no longer branches on |
| `ixc = 42`, `ixc = 61` | `tests/integration/data/ref_IN.DAT` | iteration variables, not topology |

**Zero new physics arms.** The claim in the brief is correct as stated. `ref_IN.DAT` is the
only file worth a second look — 73 switch lines against `large_tokamak`'s 69 — and its extra
four are the rows above. The bias is real and this repo cannot sample its way out of it.

## 1. Method

**[measured]** Two probes.

1. **Direct axes.** For each `UNPORTED` axis that names a real `IN.DAT` switch, the
   declaration was read out of `process/data_structure/*_variables.py` and the default
   compared against the axis's refused values.
2. **Derived axes.** The `*_arm` predicates in `indat.py` were **called** with PROCESS's own
   defaults for every switch in their signature, and the returned arm index looked up in
   `UNPORTED`. Calling the predicate rather than reasoning about it is the point: the arm
   indices are computed, and eleven of them are joint dispatches over two or three switches.

**Sentinel resolution was applied, not assumed** (`_audit/init_audit.md` §2a). For the
reference configuration `itart = 0`, `i_tf_sup = 1`:

| field | dataclass default | value used here | resolved at |
|---|---|---|---|
| `i_tf_shape` | `0` (`DEFAULT`) | `1` `D_SHAPE` | `init.py:775` (the non-TART site; `:728` is the ST one) |
| `i_tf_bucking` | `-1` | `1` | `init.py:891-895` |
| `i_tf_wp_geom` | `-1` (`UNSET`) | `1` `DOUBLE_RECTANGULAR` (because `i_tf_turns_integer = 0`) | `init.py:977-989` |
| `i_cp_joints` | `-1` | **stays `-1`** — `init.py:752-756` sits on the TART path only | — |

Two of those change an answer: `i_tf_bucking` read naively as `-1` would put `tf_stress_arm`
on `(1, -1, 0)`, which is in no registry, instead of `(1, 1, 0)`, which is ported.
`init_audit.md`'s two "real-looking" sentinels (`eyoung_ins`, `eyoung_cond_axial`) are
material properties, not switch axes, and do not key any slot.

**Inferred, not measured:** the port-cost column (LOC, whole-`Model`-vs-formula) and the tier-2
ranking. Both are argued from evidence named per row, never from a count alone.

## 2. Headline

| | axes | rows |
|---|---|---|
| the factory dispatches on | **88** | — |
| ... with at least one `UNPORTED` row | 50 | 219 |
| ... of which **default-and-unported** (tier 1) | **8** | 8 |
| ... of which **non-default-and-unported** (tier 2) | 42 | 211 |
| ... fully ported, no `UNPORTED` row (tier 3) | 38 | 0 |

**Tier 1 is 8 axes out of 88, and three of the eight need no porting work at all.** That is
the headline and it is a good result: the refusal table is *not* littered with traps for a
silent file. Of the eight, one is already in flight, one is a refusal PROCESS itself agrees
with, one needs a node class over a formula this port has already ported and tested, and one
is a topology no real tokamak input leaves unstated. The genuinely new model work a silent
unseen file forces is **three arms**: Wilson bootstrap, the 2015 cost model, and the ITER
neutral-beam heating block.

## 3. Tier 1 — default-and-unported

Ranked by how much unwritten work the arm actually is, most first.

| # | axis | unported value(s) | default | declared at | cost to port | evidence |
|---|---|---|---|---|---|---|
| 1 | `i_cost_model` | **1**, 2 | **1** `KOVARI_2014` | `cost_variables.py:327` | **whole `Model` package**, `process/models/costs/costs_2015.py` = **1227 LOC**, zero cottax nodes today | measured LOC; `UNPORTED` reason |
| 2 | `i_hcd_primary` | **5**, 0-4, 6-8, 12 | **5** `ITER_NEUTRAL_BEAM` | `current_drive_variables.py:190` | **~160 LOC of formula**: `NeutralBeam.iternb` (`current_drive.py:146-235`, ~90) + the beam wall-plug block (`:2191-2260`, ~70). Not a new `Model` | measured line spans; `UNPORTED` reason |
| 3 | `pf_coil_system_arm` | **-2**, -3..-7 | **-2** — `n_pf_coil_groups = 3`, `i_pf_location = (2,2,3)`, `n_pf_coils_in_group = (1,1,2)` | `pfcoil_variables.py:320`, `:220`, `:310` | a **third `PFCoilTopology`** and a third set of the package's **13 node instances** — the largest of the eight | predicate evaluated at defaults; `UNPORTED` -2 reason |
| 4 | `i_bootstrap_current` | **3**, 1, 2, 5-13 | **3** `WILSON` | `physics_variables.py:818` | **one `@staticmethod` formula**, `bootstrap_current.py:368-497` = **~130 LOC**. `bootstrap_current.md` § 'not ported in this pass' notes each arm needs its own occupant *and* its own harness contract | measured span; `UNPORTED` reason |
| 5 | `i_p_coolant_pumping` | **2**, 0, 1 | **2** `MECHANICAL` | `fwbs_variables.py:249` | **[IN FLIGHT — do not re-analyse]** arm 0 being ported now; arm 2 blocked on `next_steps.md` §5's CoolProp tracing policy and an SCC in `fw_temp`, not on missing formulas | brief; `UNPORTED` reason |
| 6 | `i_density_limit` | **8**, 1-6 | **8** `ASDEX_NEW` | `physics_variables.py:863` | **near-zero in `process/`**: the formula is already ported and Tier-1-tested against PROCESS's own staticmethod (`density_limit.md` '## UNPORTED'). Needs a node class + one registry line. Cheapest tier-1 row by a wide margin | `UNPORTED` reason states it verbatim |
| 7 | `hcd_primary_powers_arm` | **-1** | **-1**, from `(i_hcd_primary=5, i_hcd_secondary=0)` | derived; `current_drive_variables.py:190`, `:206` | **not independent work** — the same fact as row 2 seen at the joint slot. It becomes arm `0` the moment a file sets an EC primary; it is the combinatorial `+=` accumulator (`current_drive.py:2147` over `:1955`) that makes it a product | predicate evaluated at defaults |
| 8 | `blktmodel_ipowerflow_i_p_coolant_pumping` | **4** | **4**, from `(blktmodel=0, ipowerflow=1, i_p_coolant_pumping=2)` | `fwbs_variables.py:479`, `heat_transport_variables.py:94`, `fwbs_variables.py:249` | **zero — no port work exists.** `stellarator.py:924-928` raises `ProcessValueError` on this cell. The refusal is correct and the `UNPORTED` text already says so. Stellarator-only slot, so a tokamak never reaches it | predicate evaluated at defaults; `UNPORTED` reason |

### Qualifiers that keep the list honest

- **Row 3 is the softest.** `n_pf_coil_groups`/`i_pf_location`/`n_pf_coils_in_group` are coil-count
  *topology*, not switches — five of the eight regression files state them explicitly and the
  three that do not are the two stellarators and IFE, none of which has a PF coil system a
  tokamak would recognise. So the default is refused, but a real tokamak input file that
  omits its PF layout is not a thing anyone has observed. Ranked 3rd on cost, not on
  likelihood.
- **Rows 7 and 8 are not separate work** and rows 5 and 6 are nearly free. **The tier-1 list
  that actually costs anything is rows 1, 2 and 4** — 1227 LOC of costs, ~160 of neutral
  beam, ~130 of Wilson.
- **A doc slip found in passing, not fixed** (read-only pass): `_blanket_shield_power_arm`'s
  docstring says "Arm 2 is PROCESS's own default (`blktmodel = 0`, `ipowerflow = 1`,
  `i_p_coolant_pumping = 1`)". `i_p_coolant_pumping`'s dataclass default is `2`
  (`fwbs_variables.py:249`), not `1`; `1` is the reference *stellarator run's* value. The
  arithmetic in the function is right and the `UNPORTED` row for arm 4 states the default
  correctly — only that one docstring sentence is wrong. Owed to whoever owns `indat.py`.

## 4. Tier 2 — non-default-and-unported

**42 axes, 211 rows.** Only reachable by a file that asks. Ranked by three independent
evidence sources, named per row:

- **(R)** a tracked regression file already sets the axis to an unported value — the
  strongest signal, because it is a configuration someone actually wrote;
- **(D)** `documentation/source/` recommends the arm — evidence about what a *new* user would
  be told to write;
- **(U)** `tests/unit/models/` exercises the arm directly — evidence PROCESS's own maintainers
  treat it as live.

### 4a. Top rows — ranked

| rank | axis | unported | default (declared at) | evidence |
|---|---|---|---|---|
| 1 | `i_tf_sc_mat` | 9 `HAZELTON_ZHAI_REBCO` | 1 (`tfcoil_variables.py:246`) | **(R)** `spherical_tokamak_eval` **and** `st_regression` both set `9`. Two of eight tracked files, and both STs. **(U)** 15 unit hits. The strongest tier-2 row there is |
| 2 | `ife` | 1 `INERTIAL_CONFINEMENT` | 0 (`ife_variables.py:253`) | **(R)** `IFE.IN.DAT` sets it. **Cost: `process/models/ife.py` = 2488 LOC + `ife_variables.py` = 617** — a whole device, not an arm. High evidence, highest cost |
| 3 | `i_beta_norm_max` = 3, `i_ind_plasma_internal_norm` = 2 | 3 / 2 | 1 (`physics_variables.py:945`) / 0 (`:948`) | **(D)** the docs pair these two explicitly for STs: `plasma_beta.md:277-279` "only recommended for spherical tokamaks … use with `i_ind_plasma_internal_norm = 2`", and `plasma_inductance.md:57-59` says the same in reverse. **A mismatch worth flagging**: both tracked STs set `0`/`0`, so the docs' recommended ST configuration is *not* the one this port was shaped by |
| 4 | `i_plasma_current` | 1,2,3,5,6,7,8 | 4 (`physics_variables.py:843`) | **(U)** 12 unit hits, **(D)** 2 doc files with per-arm recommendations (`plasma_current.md:576-578`). Seven closed-form scalings; no tracked file selects a refused one (all set 4 or 9, both ported) |
| 5 | `i_rad_loss` | 0, 2 | 1 (`physics_variables.py:954`) | **(U)** 19 unit hits — the highest tier-2 unit-test count on a physics axis. No **(R)**: both stellarators set the ported `1` |
| 6 | `i_tf_sup` | 2 `HELIUM_COOLED_ALUMINIUM` | 1 (`tfcoil_variables.py:261`) | **(U)** 86 unit hits, by far the most of any axis. No **(R)**. See §24.6's LATENT item — this axis is *partially* enforced (`2` refused, `0` silently accepted), which no `UNPORTED`-keyed table can show |
| 7 | `i_plasma_geometry` | 1-9, 11, 12 | 0 (`physics_variables.py:971`) | **(U)** 4, **(D)** 3. `low_aspect_ratio_DEMO` sets `10`, which is ported — near-miss, not a hit |
| 8 | `i_hcd_secondary` | 1-8, 10, 12, 13 | 0 `NO_CURRENT_DRIVE` | `current_drive_variables.py:206` | **(D)** 11 doc files, **(U)** 7. Coupled to tier-1 row 7: any non-zero secondary re-opens the `hcd_primary_powers_arm` product |
| 9 | `i_blanket_type` | 5 | 1 (`fwbs_variables.py:70`) | **(U)** 8; three regression files set the ported `1` |
| 10 | `isthtr` | 3 | 1 (`stellarator_variables.py:87`) | neither stellarator sets `3`; **(D)** 1 doc file. Low |

### 4b. The rest — no (R), no (D), little or no (U)

`i_l_h_threshold` (default 19, `physics_variables.py:1234`; 15 refused arms but **0 unit
hits** and one doc file), `i_diamagnetic_current` (default 0, `:856` — both STs set the
*ported* `2`), `i_ecrh_wave_mode` (0, `current_drive_variables.py:116`; 0 unit, 0 doc hits),
`i_hcd_calculations` (1, `:223`), `i_len_sol_outboard_power_decay` (1, `:1718`; 0 unit, 0 doc
hits — the weakest axis in the table), `i_pf_energy_storage_source` (2,
`pf_power_variables.py:18`), `i_tf_sup_build` (keys `i_tf_sup`, default 1),
`i_tf_inside_cs_vacuum_shield` (keys `i_tf_inside_cs`, default 0, `build_variables.py:189`),
`i_plasma_ignited_separatrix` (keys `i_plasma_ignited`, default 0, `physics_variables.py:881`).

**All 22 derived `*_arm` axes not already in tier 1** land here, each verified by evaluating
the predicate at defaults: `blktmodel_blkttype` → 2, `centrepost_neutronics_arm` → 0,
`cicc_turn_geometry_arm` → 0, `croco_turn_geometry_arm` → 0, `divertor_geometry_arm` → 0,
`divertor_heat_load_arm` → 0, `first_wall_arm` → 0, `i_plasma_ignited_i_rad_loss` → 1,
`i_pulsed_plant_istore` → 0, `plasma_geometry_arm` → 0, `pulse_ramp_times_arm` → 0,
`structure_arm` → 0, `surface_poloidal_field_arm` → 0, `tf_coil_shape_arm` → 0,
`tf_field_and_force_arm` → `False`, `tf_inboard_radii_arm` → 0, `vacuum_vessel_arm` → 0,
`tf_stress_arm` → `(1, 1, 0)` **[IN FLIGHT — not re-analysed]**, and the four
`i_str_wp_i_tf_sc_mat_*` registries → `(1, 1)`, which is ported on both CICC slots.

**The whole CroCo half of that last group is unreachable by default**:
`i_tf_turn_type` defaults to `1` `CABLE_IN_CONDUIT` (`superconducting_tf_coil_variables.py:194`),
and `caller.py:307-313` only reaches `CROCOSuperconductingTFCoil` on `2`. So
`i_str_wp_i_tf_sc_mat_croco_sc_properties`, `..._croco_temp_margin` and
`croco_turn_geometry_arm` are gated behind a non-default switch **before** their own arm is
asked — 35 of the 211 tier-2 rows, all double-gated.

## 5. Tier 3 — fully ported axes

**38 axes with no `UNPORTED` row at all**, for completeness of the 88:

`cplife_arm`, `cryo_loads_arm`, `cryo_q_loads_arm`, `dr_tf_inboard_winding_pack`,
`electric_production_arm`, `eta_turbine_arm`, `fw_blkt_vv_shape_arm`, `i_alphaj`,
`i_beta_fast_alpha`, `i_confinement_time`, `i_cs_superconductor`, `i_f_dr_tf_plasma_case`,
`i_nd_plasma_pedestal_separatrix`, `i_pfirsch_schluter_current`, `i_pflux_fw_neutron_ipowerflow`,
`i_plasma_ignited`, `i_plasma_pedestal`, `i_single_null`, `i_tf_case_geom`, `i_tf_shape_build`,
`i_tf_wp_geom`, `ibkt_life`, `inuclear_i_tf_sup`, `ipowerflow`, `ireactor_ipnet_itart`,
`istell`, `itart_hcpb`, `itart_i_tf_sc_mat_sc_tf_masses`, `n_divertors`,
`nuclear_heating_renormalisation_arm`, `p_fw_blkt_coolant_pump_arm`,
`p_fw_div_heat_deposited_arm`, `peak_b_ripple_arm`, `secondary_cycle_liq`,
`supercond_cost_model`, `temp_turbine_coolant_in_arm`, `tf_self_inductance_arm`,
`tfc_sidewall_is_fraction`.

Four axes are refused via `_refuse_unported_switch` rather than `_slot_occupant` and so do not
appear in `machine_survey.slot_registries()`: `i_cost_model`, `i_p_coolant_pumping`, `ife`,
`isthtr`. All four are counted above.

## 6. Blind spots of this method — stated, not hidden

1. **A default is not a probability.** This measures what a file *silent* on a switch gets. It
   says nothing about which switches real users habitually set, and the two evidence sources
   used for tier 2 (unit tests, docs) are proxies for that, not measurements of it.
2. **It is keyed on `UNPORTED`, and `next_steps.md` §24.6 proved that table does not bound the
   class.** A partially-enforced axis (`i_tf_sup`: `2` refused, `0` silently accepted with a
   wrong answer) and a fully-enforced one are indistinguishable in any table with these rows
   as its index. The `PfMagnetCost`/`iohcl` LIVE defect of §24.6 would not appear anywhere in
   this census. **This document ranks the refusals; it does not find the missing ones.**
3. **One configuration was evaluated**, `itart = 0`, `i_tf_sup = 1` — the conventional
   superconducting tokamak. A spherical or resistive machine resolves four sentinels
   differently (`i_tf_shape` → `PICTURE_FRAME`, `i_cp_joints` → `0`/`1`, `i_tf_bucking` → `0`
   on copper, `i_tf_wp_geom`) and its tier-1 list may differ. Re-running §1's second probe with
   a different `D` dict is a few minutes' work and was not done.
4. **`ife`'s 1016 unit-test "hits" are a substring artefact** (`life`, `different`), caught by
   inspection; it is not used as evidence above. Every other unit-hit count in §4 is a raw
   `grep -o` count and could carry a smaller version of the same error — treat those numbers as
   ordinal, not cardinal.
5. **Port-cost figures are line spans, not effort estimates.** `costs_2015.py`'s 1227 LOC is
   measured; that this makes it the most expensive tier-1 row is inferred.
