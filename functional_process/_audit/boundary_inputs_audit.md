# Boundary (unowned) inputs of the driven graph — are they all genuinely inputs?

**Question asked**: `driven_graph(GRAPH)` has 379 unowned inputs; the reference input file
sets 79 of them; 300 are not set by the file. `.tfcoil.len_tf_coil` is one of those 300
and PROCESS *computes* it (`process/models/stellarator/coils/calculate.py:87`) — a real
cut edge, a quantity this port takes as given that PROCESS derives. How many more are
there, and does closing any of them create a cycle?

**Short answer**: **12**, out of 300. Not 300, not 93. Every other one of the 300 is
either never assigned anywhere in `process/` (207), assigned only to a constant during
initialisation (35), or assigned only in a code path this configuration never takes (46).
And **none of the 12 hides a cross-subsystem cycle** — checked by measuring node-level
reachability, not by inspection. The one that does touch a cycle
(`.physics.fusden_alpha_total`) lands *inside* the SCC that `mda.CUTS` already cuts.

**But this audit found a different, worse thing on the way**: a boundary input that is not
merely missing a producer but is bound to the **wrong field**
(`ProfileValues.rho` → `.neoclassics.r_eff`, which PROCESS never writes and which is
permanently `0.0`, while PROCESS passes the literal `0.6`), and the reason the MDA harness
cannot see it — `mda_harness.compare` **silently drops all 29 array-valued outputs** with
`continue`, counting them as neither agreement, disagreement, unverifiable, nor error. See
§6. That is the same "tautological agreement" blind spot the switch audit found, in a
second, independent form.

*Status: investigation only. **No registration change, no model edit, no harness edit was
made.*** `$PY -m pytest functional_process -q` after this audit: **3597 passed, 2909
skipped** (the number moved from `next_steps.md` §8.4's 3418 because of §9 and of a
concurrently-running session, not because of anything here — nothing here changed code).

---

## 1. The measurement, reproduced

Reproduced exactly, and the "set by the input file" half was tightened from name-matching
to **name-and-area** matching through PROCESS's own `INPUT_VARIABLES` table
(`process/core/input.py`), so the count cannot be inflated by two areas sharing a field
name:

| | count |
|---|---|
| `driven_graph(GRAPH).unowned_inputs` | **379** |
| set by `tests/regression/input_files/stellarator_helias.IN.DAT` | **79** |
| not set by the input file | **300** |

Method: parse every `name[(index)] = ...` LHS in the input file (122 distinct names, all
122 present in `INPUT_VARIABLES` — no unparsed lines, no unknown names), then for each
unowned `VarPath` `.area.field` require both `field ∈ file-names` and
`INPUT_VARIABLES[field].module == area`. **Zero** paths matched by name but not by area,
so the 79/300 split is the same either way. `GRAPH` here is `graph_for()`'s default,
`REFERENCE_CONFIGURATION` (`total_process.py:1182-1187`) — the helias run's own switch
choices (`istell=6`, `isthtr=1`, `i_plasma_pedestal=0`, `i_cost_model=0`, `ireactor=1`).
146 nodes in `GRAPH`, 148 after `driven_graph`'s two `FixedPointCut`s.

One caveat on what "set by the input file" means, worth recording because it looks like a
miss and is not: 12 of the 300 are
`.impurity_radiation.f_nd_impurity_electron_array[2..13]`, and the input file **does** set
those numbers — under the *other* name `f_nd_impurity_electrons(3..14)`
(`stellarator_helias.IN.DAT:210-223`), which `check_process` copies element-by-element into
the array field (`process/core/init.py:382-385`). They are genuine inputs; the count is
name-faithful, not substance-faithful, in exactly this one place.

## 2. Narrowing 300 → 93, mechanically

**Assignment idiom check first, not assumed.** A first pass with the obvious regex
(`\.<field>\s*(\[...\])?\s*=`) was wrong in both directions, so the authoritative scan is
an **`ast` walk of every `.py` under `process/`** collecting every `Assign`/`AugAssign`/
`AnnAssign` target, recursing through `Tuple`/`List`/`Starred`/`Subscript`, and
reconstructing the full dotted target so `.area.field` must match, not just `field`:

- the regex **missed tuple-unpack writes** — `(\n  self.data.physics.beta_beam,\n  ...,\n) = reactions.beam_fusion(...)`
  (`process/models/stellarator/stellarator.py:2010-2013`) has no `=` after the attribute.
  Three fields recovered this way: `.physics.beta_beam`, `.physics.p_beam_alpha_mw`,
  `.physics.p_plasma_ohmic_mw`.
- the regex **counted f-strings and log messages as writes** —
  `f"{self.data.current_drive.i_hcd_primary = }"` (`process/models/physics/current_drive.py:1802`),
  `"...(data.tfcoil.i_tf_sup = 1)"` (`process/core/init.py:678`),
  `f"{self.data.times.t_plant_pulse_burn = }"` (`process/core/solver/evaluators.py:80`).
- area-blind matching also counted `self.ife = IFE(...)` (`process/main.py:673`) as a write
  to `.ife.ife`, and `self.plasma_current = PlasmaCurrent()` (`:702`) as a write to
  `.physics.plasma_current`. Reconstructing the dotted path removes both.

Two further false-negative mechanisms were checked and found absent, rather than assumed
away:

- **`setattr`** — 6 non-test call sites in `process/` (`process/core/input.py:1434,1482`
  the IN.DAT parser; `process/models/stellarator/preset_config.py:262` the stellarator
  config-file loader; `process/core/scan.py:118` and
  `process/core/solver/iteration_variables.py:382,395` the scan/iteration-variable
  writers). All four *kinds* are input mechanisms, not model computation. Recorded
  per-field below where relevant.
- **area-alias locals** (`bld = self.data.build; bld.dr_cs = ...`) — 23 such bindings exist
  (`process/models/vacuum.py:57-59`, `process/models/tfcoil/superconducting.py` ×12,
  `process/models/pfcoil.py` ×4, …) but they keep the dot, so the scan sees them. Checked
  separately whether any of the 207 never-assigned fields is ever bound to a bare local
  that is then mutated: 7 such bindings exist and **all 7 are scalar reads out**
  (`wt = self.data.buildings.wgt`, `bc20m = data.tfcoil.b_crit_upper_nbti`, …), none a
  mutable-array alias.

Result:

| | count |
|---|---|
| **never assigned anywhere** in `process/` outside `data_structure/` | **207** |
| **assigned somewhere** — the candidate set | **93** |

The 207 are genuine inputs by construction and are summarised (not enumerated
individually) in §3. The 93 are classified one at a time in §4.

## 3. The 207 never-assigned — genuine inputs, category (a), summarised

Split by whether PROCESS's own input table would even accept them from an IN.DAT:

| | count | what they are |
|---|---|---|
| IN.DAT-settable, this run leaves at the `*_variables.py` default | **138** | ordinary inputs |
| not IN.DAT-settable | **69** | see below |

The 69 non-input ones break down completely, with no residue:

- **35 `.costs.*`** — 34 `UC*` unit-cost constants plus `sc_mat_cost_0`, module-level
  constants in `process/data_structure/cost_variables.py` (e.g. `UCAD: float = 180.0`,
  `:566`; `sc_mat_cost_0`, `:770`). Never written, by design.
- **25 `.stellarator_config.stella_config_*`** — loaded from the machine's JSON
  configuration file by `process/models/stellarator/preset_config.py:258-266`
  (`setattr(data.stellarator_config, f"stella_config_{k.lower()}", v)`), reached on this
  run's path via `st_new_config` (`process/models/stellarator/stellarator.py:156`). A
  config-file load is an input, and the harness seeds it correctly.
- **5 `.vacuum.l1`/`l2`/`l3`/`ceff_i`/`xmult_i`** — the deliberately-minted
  `DuctDiameterRootFind` locals already documented in `mda_harness.py`'s module docstring
  and `EXCLUDED_NODE_NAMES`. Real PROCESS keeps them as locals inside
  `_solve_vacuum_pumping_old`'s per-species Newton loop.
- **4 singles**, each checked individually:
  - `.tfcoil.j_crit_str_0` — a constant list in `process/data_structure/tfcoil_variables.py:353`,
    read at `process/models/costs/costs.py:1496,1662,1742`. Input.
  - `.fwbs.f_nuc_pow_bz_liq` — default `0.66` (`process/data_structure/fwbs_variables.py:653`).
    `process/models/blankets/blanket_library.py:2205` assigns a **local** of the same name,
    not the field. Input.
  - `.current_drive.p_beam_injected_mw` — read at `process/models/costs/costs.py:1915` and
    reported at `process/models/physics/current_drive.py:2622,2824`; never written anywhere
    in `process/`. Input (arguably a dead field, but correct as a boundary input).
  - **`.neoclassics.r_eff` — a genuine input in PROCESS's data structure, and a real port
    defect in this graph.** See §6; this is the one entry in the 207 that is not simply
    "fine".

## 4. The 93 candidates, classified individually

Categories as asked: **(a)** genuine input — never assigned, or assigned only to a
constant; **(b)** computed by a model this port has not registered; **(c)** computed by a
model this port *has* registered, whose node does not declare that `Output`; **(d)**
computed only in a code path this configuration never takes.

**The arbiter for (d) is `process/core/caller.py:272-275`**, which for `istell != 0`
calls `self.models.stellarator.run()` and **returns immediately** — no `plasma_geom`, no
`build`, no `physics`, no TF-coil model, no `pfcoil`, no `pulse`, no `divertor`, no
`fw`/`shield`/`vacuum_vessel`, no blanket model. Inside that, the call sequence is
`process/models/stellarator/stellarator.py:114-186`: `st_new_config`, `st_geom`, `st_phys`,
`st_density_limits`, `st_coil`, `st_build`, `st_strc`, `st_fwbs`, `st_div`,
`power.tfpwr`, `power.component_thermal_powers`, `power.calculate_cryo_loads`,
`buildings.run`, `vacuum.run`, `power.acpow`, `power.plant_electric_production`,
`availability.avail`, `costs.run`. Notably **`Power.pfpwr` is not in that list** (its only
callers are `Power.run`, `power.py:54,81`), which alone accounts for 19 of the 93.

### 4a. Category (a) — 35, assigned only to a constant

| VarPath | evidence | value |
|---|---|---|
| `.build.dr_cs` | `process/models/stellarator/initialization.py:23` (`st_init`) | `0.0` — no central solenoid |
| `.build.dr_cs_tf_gap` | `initialization.py:26` | `0.0` |
| `.physics.kappa95` | `initialization.py:33` | `1.0` |
| `.physics.triang` | `initialization.py:34` | `0.0` |
| `.times.t_plant_pulse_coil_precharge` | `initialization.py:43` | `0.0` |
| `.times.t_plant_pulse_plasma_current_ramp_up` | `initialization.py:44` | `0.0` |
| `.times.t_plant_pulse_burn` | `initialization.py:45` | `3.15576e7` (one year) |
| `.times.t_plant_pulse_plasma_current_ramp_down` | `initialization.py:46` | `0.0` |
| `.buildings.triv` | `process/core/init.py:370` (`check_process`) | `0.0` |
| `.heat_transport.p_tritium_plant_electric_mw` | `init.py:371` | `0.0` |
| `.buildings.esbldgm3` | `init.py:827` | `0.0` |
| `.divertor.n_divertors` | `init.py:609,617` | `2` or `1`, chosen at init from a switch |
| `.physics.f_nd_beam_electron` | `init.py:1145,1147` | `0.0` |
| `.impurity_radiation.f_nd_impurity_electron_array[2..13]` (12) | `init.py:383` | copied from the IN.DAT array, see §1 |
| `.impurity_radiation.m_impurity_amu_array` | `process/models/physics/impurity_radiation.py:322` (`init_imp_element`) | loaded from PROCESS's impurity data files at startup |
| `.impurity_radiation.temp_impurity_keV_array` | `impurity_radiation.py:374` | ″ |
| `.impurity_radiation.pden_impurity_lz_nd_temp_array` | `impurity_radiation.py:375` | ″ |
| `.impurity_radiation.impurity_arr_zav` | `impurity_radiation.py:376` | ″ |
| `.physics.radius_plasma_pedestal_density_norm` | `process/models/physics/plasma_profiles.py:112` | `1.0` — L-mode reset, constant either way |
| `.physics.nd_plasma_pedestal_electron` | `plasma_profiles.py:115` | `0.0` ″ |
| `.physics.nd_plasma_separatrix_electron` | `plasma_profiles.py:116` | `0.0` ″ |
| `.physics.tbeta` | `plasma_profiles.py:117` | `2.0` ″ |
| `.structure.fncmass` | `process/models/stellarator/stellarator.py:334` (`st_strc`) | `0.0` — "set to zero to avoid double-counting" (that method's own docstring) |
| `.structure.gsmass` | `stellarator.py:337` | `0.0e0  # ? Not sure about this.` (source comment verbatim) |

Two notes on the borderline rows, since (a) vs (d) is the distinction the brief asked to
keep sharp:

- The four `plasma_profiles.py:112-117` rows are inside an **error-recovery block** that
  fires only when the L-mode consistency check at `:92-100` fails. Whether it fires or not,
  the field ends up at the same L-mode constant, so (a) either way. Their *other* producer,
  `NeProfile.set_pedestal_and_separatrix_values` (`process/models/physics/profiles.py:318,325`),
  has exactly one caller — `Physics.run` (`process/models/physics/physics.py:368`) — which
  is off this run's path, so it contributes nothing.
- `.structure.fncmass`/`.gsmass` are written **every iteration** by `st_strc`, not once at
  init, but always to the literal `0.0`. Classified (a) because a node owning them would
  be a constant node; recorded here so a later pass does not "discover" them as a gap.
  (Their tokamak producer `process/models/structure.py:48,52` is off-path.)

### 4b. Category (d) — 46, off this configuration's path

Grouped by why; every row verified against the call sequence above, not inferred from the
file name.

**`Power.pfpwr` never runs (19 fields).** Its only callers are `Power.run`
(`process/models/power.py:54,81`); `stellarator.py` calls `tfpwr`,
`component_thermal_powers`, `calculate_cryo_loads`, `acpow`, `plant_electric_production`
and never `run` or `pfpwr`. Same for the whole `PFCoil`/`CSCoil` model, which
`caller.py:319` reaches only on the tokamak branch.

- from `Power.pfpwr`: `.pf_power.acptmax` (`power.py:576,590`), `.pf_power.ensxpfm` (`:562`),
  `.pf_power.pfckts` (`:572`), `.pf_power.spfbusl` (`:575`), `.pf_power.srcktpm` (`:352,411`),
  `.pf_power.vpfskv` (`:571`), `.heat_transport.peakmva` (`:569`),
  `.pf_coil.p_pf_electric_supplies_mw` (`:604`)
- from `PFCoil.pfcoil` (`process/models/pfcoil.py`): `.pf_coil.j_crit_str_pf` (`:900,902`),
  `.pf_coil.j_pf_coil_wp_peak` (`:764`), `.pf_coil.m_pf_coil_max` (`:852,1016`),
  `.pf_coil.m_pf_coil_structure_total` (`:1054,1061`), `.pf_coil.n_pf_coil_turns`
  (`:607,757,808,1079`), `.pf_coil.r_pf_coil_middle` (`:182,667`),
  `.pf_coil.r_pf_coil_outer_max` (`:738,840`)
- from `PFCoil.waveform`: `.pf_coil.c_pf_cs_coils_peak_ma` (`pfcoil.py:2891,2904,2918,3264,3274`)
- from `CSCoil.ohcalc`: `.pf_coil.a_cs_cable_space` (`pfcoil.py:3548,3557`),
  `.pf_coil.f_a_pf_coil_void` (`:3322`), `.pf_coil.j_crit_str_cs` (`:3622,3626`)

**Tokamak-only physics models (6).** `Physics.run` and `PlasmaGeom.run` are
`caller.py:284,290`, tokamak branch only; `st_geom`
(`stellarator.py:276-318`) writes `vol_plasma`/`a_plasma_surface`/`a_plasma_poloidal`/
`a_plasma_surface_outboard` and no elongation or current field at all.

`.physics.alphaj` (`physics.py:338,343`), `.physics.dlamie` (`:279`), `.physics.qstar`
(`:303`), `.physics.plasma_current` (`:286`), `.physics.p_plasma_ohmic_mw` (`:771`),
`.physics.kappa` (`plasma_geometry.py:251,272,285,300,309,330,353,396,411,428`).

`.physics.plasma_current` is worth calling out even though its classification is
unambiguous: four registered nodes read it (`ProfileFactors`,
`AuxiliaryPhysicsQuantities`, `DimensionlessPlasmaParameters`,
`StellaratorConfinementTime`) and on a stellarator PROCESS itself never computes it, so
whatever value the converged run carries is the input value, not a derived one.

**Switch arms this run does not take (12).**

| VarPath | producer | why it does not fire |
|---|---|---|
| `.tfcoil.m_tf_bus` | `Power.tfpwr`, `power.py:2146` | inside `if i_tf_sup != 1` (`:2131`); this run is `i_tf_sup = 1`, `tfcoil_variables.py:261`'s default, not set in the IN.DAT |
| `.tfcoil.tfcmw` | `Power.tfpwr`, `power.py:2192` | ″ |
| `.costs.cpstcst` | `Costs.acc2221`, `costs.py:1461,1463,1466` | inside `if i_tf_sup != SUPERCONDUCTING` (`:1448-1450`) |
| `.fwbs.p_div_rad_total_mw` | `st_fwbs`, `stellarator.py:630` | inside `if blktmodel == 1` (`:608`) → `if ipowerflow == 1` (`:611`); this run is `blktmodel = 0`. Independently confirmed by `unit_registry.md` row 1E1, which records this same field as "read-but-never-assigned … leaves it permanently at its `0.0` default whenever `blktmodel != 1 & ipowerflow == 1`" |
| `.fwbs.temp_blkt_coolant_out` | `st_fwbs`, `stellarator.py:805,814` | inside `if i_blkt_coolant_type == CoolantType.WATER` (`:803`); this run is `1` (`HELIUM`, `fwbs_variables.py:279`), and neither `i_blkt_coolant_type` nor `blkttype` appears in the IN.DAT |
| `.physics.beta_beam` | `st_phys`, `stellarator.py:2011` | inside `if p_hcd_beam_injected_total_mw != 0 and i_plasma_ignited == NON_IGNITED` (`:2006-2008`); `isthtr = 1` (ECRH) means `st_heat` never sets a beam power (`heating.py:36-50`), and `i_plasma_ignited = 1` on this run (`next_steps.md` §8.2) — false twice over |
| `.physics.p_beam_alpha_mw` | `stellarator.py:2013` | ″ |
| `.current_drive.p_hcd_beam_injected_total_mw` | `st_heat`, `heating.py:74` | `isthtr == 3` arm only |
| `.current_drive.p_beam_orbit_loss_mw` | `heating.py:78` | ″ |
| `.current_drive.p_hcd_lowhyb_injected_total_mw` | `heating.py:53` | `isthtr == 2` arm only |
| `.current_drive.p_beam_shine_through_mw` | `CurrentDrive.current_drive`, `current_drive.py:1999,2200` | that model is never called on the stellarator path |
| `.fwbs.i_blkt_coolant_type` | `CCFE_HCPB.run`, `hcpb.py:45` | `blktmodel == 1` only |

**Models never reached on this path (9).**

`.build.r_tf_inboard_mid` (`process/models/build.py:1728`, `Build.calculate_radial_build`;
`st_build` has no counterpart write), `.buildings.dz_tf_cryostat`
(`process/models/cryostat.py:58`; `st_fwbs` computes its own cryostat geometry inline at
`stellarator.py:1276-1296` and does not write this field), `.fwbs.neut_flux_cp`
(`hcpb.py:128,148`), `.fwbs.p_cp_shield_nuclear_heat_mw` (`hcpb.py:137,146,267`),
`.heat_transport.p_blkt_breeder_pump_mw`
(`process/models/blankets/blanket_library.py:2632`),
`.tfcoil.a_tf_wp_coolant_channels` (`process/models/tfcoil/superconducting.py:2474,3909`),
`.tfcoil.j_crit_str_tf` (`superconducting.py:2944,2978,3005,3041,3082,3110,4512`),
`.tfcoil.p_cp_coolant_pump_elec` (`process/models/tfcoil/base.py:1481`),
`.tfcoil.res_tf_leg` (`process/models/tfcoil/resistive.py:606`).

### 4c. Categories (b) and (c) — the 12 real cut edges

These are the findings. Each is a value PROCESS computes on **this** run's path that the
port takes as given.

#### (c1) `.tfcoil.len_tf_coil` — the one the brief named. Confirmed.

- **Producer**: `st_coil`'s inline geometry block,
  `process/models/stellarator/coils/calculate.py:86-91`:
  `len_tf_coil = stella_config_coillength * (r_coil_minor / stella_config_coil_rminor) / n_tf_coils`.
- **Registered readers (4)**: `StructureMasses`, `PlasmaFacingCoilArea`, `CoilsMass`,
  `TfMagnetCostSuperconducting`.
- **Why (c), not (b)**: the port *has* decomposed `st_coil` into registered nodes, and it
  has already carved one node out of this very block — `ZTfInsideHalf`
  (`functional_process/models/stellarator/coils/calculate.py:1302-1370`) owns
  `.build.z_tf_inside_half` from `calculate.py:79-84`, the four lines immediately above.
  The formula is already ported and already runs, at
  `functional_process/models/stellarator/coils/calculate.py:1599-1603`, inside the eager
  `st_coil` orchestrator that `total_process.py` deliberately does not register. Nothing is
  missing but a node.
- **Cycle risk: none.** Measured. The producer's inputs are
  `.stellarator_config.stella_config_coillength` (boundary, and currently read by nothing —
  it is not even in the 379), `.stellarator_config.stella_config_coil_rminor` (boundary),
  `.stellarator.r_coil_minor` and `.tfcoil.n_tf_coils` (both owned by
  `StellaratorScalingFactors`). `StellaratorScalingFactors` is **not** reachable from any of
  the four readers, so the closure is acyclic. The hypothesis that `len_tf_coil` hides
  plasma↔TF↔build feedback is **not** supported by the graph.
- **Design decision required before closing it** — and this is the interesting part.
  `calculate.md:124` and `calculate.py:1404,1449-1463` record a real PROCESS bug at exactly
  this field: `st_coil` calls `calculate_plasma_facing_coil_area(data)` at
  `process/models/stellarator/coils/calculate.py:68`, which **reads `data.tfcoil.len_tf_coil`
  19 lines before `:87` writes it**, so `tfsai`/`tfsao` are computed from the *previous*
  `Caller` round's value. The eager port preserves this faithfully with a separate
  `len_tf_coil_stale` parameter. The declared `PlasmaFacingCoilArea` node, however, binds
  `.tfcoil.len_tf_coil` directly (`calculate.py:227`), so adding a producer would silently
  switch it from stale to fresh. At the converged fixed point stale == fresh, so the
  harness number should not move — but the distinction the audit record went to trouble to
  preserve would be quietly erased. Either accept it explicitly (documented), or give
  `PlasmaFacingCoilArea` a `FixedPointFunction`-shaped stale binding.
- **Sibling gap in the same three lines**: `.tfcoil.tfcryoarea` (`calculate.py:92-101`) and
  `min_bending_radius` (`:105-110`) are unported for the same reason. `tfcryoarea` is not in
  the 379 only because its sole would-be readers, `Cryo`/`CryoLoads`, are unregistered — see
  (b7)/(b8). Close them together.

#### (c2) `.physics.p_plasma_inner_rad_mw`

- **Producer**: `st_phys`, `process/models/stellarator/stellarator.py:2160-2162`:
  `p_plasma_inner_rad_mw = pden_plasma_core_rad_mw * vol_plasma`.
- **Registered reader (1)**: `StellaratorConfinementTime`.
- **Why (c)**: both operands are already owned — `.physics.pden_plasma_core_rad_mw` by
  `PlasmaRadiationPowers` (`functional_process/models/physics/radiation_power.py:721`),
  `.physics.vol_plasma` by `StellaratorPlasmaGeometry` — and the immediately following lines
  (`:2175-2220`) are ported and registered as `HeatingAndRadiationPower`
  (`functional_process/models/stellarator/stellarator_B_st_phys.py:233-528`, whose docstring
  states it also folds in `:2167`). Only `:2152-2165` fell between the two nodes. Chunk 1B is
  registered; this output is not declared by any of its nodes.
- **Cycle risk: none.** Measured — neither `PlasmaRadiationPowers` nor
  `StellaratorPlasmaGeometry` is reachable from `StellaratorConfinementTime`.
- **Blocker, not optional**: the same unported stretch contains PROCESS's zero-clip,
  `pden_plasma_core_rad_mw = max(pden_plasma_core_rad_mw, 0.0)`
  (`stellarator.py:2152-2155`). `PlasmaRadiationPowers`'s own docstring already flags that
  it deliberately does **not** clip ("that `max(..., 0)` is in `st_phys` … ported here it
  would double-own the fields"), so the value it owns is the *unclipped* one. Wiring
  `p_plasma_inner_rad_mw` straight off it would be arithmetically wrong whenever the clip is
  active. `mda_harness.KNOWN_MINT_VALUES` already records the same clip as the reason
  `pden_impurity_core_rad_total_mw` gets no reconstruction. The clip needs a node (or an
  explicit "measured inactive on this run" argument) first. Its sibling
  `.physics.p_plasma_outer_rad_mw` (`:2164`) is in the same position but currently has no
  reader, so it does not appear in the 379.

#### (b1)–(b4), (b5), (b6) `.fwbs.m_blkt_li2o`, `m_blkt_beryllium`, `m_blkt_steel_total`, `m_blkt_vanadium`, `whtshld`, `wpenshld`

- **Producer**: `st_fwbs`, `process/models/stellarator/stellarator.py:1068-1085` (blanket
  component masses, inside `if blktmodel == 0` at `:1056` — **this run's arm**) and
  `:1196-1206` (`whtshld = vol_shld_total * den_steel * (1 - vfshld)`;
  `wpenshld = whtshld`, both unconditional in `st_fwbs`).
- **Registered readers**: `BlanketCost` (the four masses), `Bldgs` + `ShieldCost`
  (`whtshld`), `ShieldCost` (`wpenshld`).
- **Why (b)**: this is section **S4** `blanket_shield_fw_coolant_mass` (1045-1274) of the
  `st_fwbs` synthesis. `unit_registry.md`:63 records S4 as *unported*, "needs no further
  audit, just S2/S3's signatures" — and S2 (`stellarator_fwbs_s2.py`) and S3
  (`DivertorPlateMass`) have since been ported and registered
  (`unit_registry.md`:65-66). The stated blocker is gone.
- **Cycle risk: none, all six.** Measured. The only owned operands are
  `.fwbs.vol_blkt_total` and `.fwbs.vol_shld_total`, both owned by
  `FwBlanketShieldGeometry` (S1, registered), which is not reachable from `BlanketCost`,
  `ShieldCost` or `Bldgs`. Everything else (`fblli2o`, `fblbe`, `fblss`, `fblvd`,
  `den_steel`, `vfshld`) is a boundary input and stays one.
- Six fields for one unit — the best fields-per-unit ratio in this list.

#### (b7), (b8) `.physics.fusden_total`, `.physics.fusden_alpha_total`

- **Producer**: `st_phys`, `process/models/stellarator/stellarator.py:2030-2042` (beam arm)
  and `:2050-2052` (no-beam arm). On **this** run the beam arm is dead (see (d), `beta_beam`)
  and the writes are the two identities `fusden_total = fusden_plasma`,
  `fusden_alpha_total = fusden_plasma_alpha`.
- **Registered readers**: `AuxiliaryPhysicsQuantities` (both), `PlasmaComposition`
  (`fusden_alpha_total`).
- **Why (b)**: `stellarator.py:2002-2054` is unported. Chunk 1B's ported ranges are
  `1916-1919`, `1930-1968`, `1971-1976`, `1991-2001`, `2095-2117`, `2175-2220`, `2223-2257`,
  `2282-2290` (each stated in its own function docstring in
  `functional_process/models/stellarator/stellarator_B_st_phys.py`); `2002-2094` is a hole.
  Both operands are already owned by `FusionRates` and — measured — currently have **zero
  readers** in the graph, i.e. the port computes them and then throws them away while
  reading the boundary value of the same quantity two lines later.
- **CYCLE RISK: YES, and it is the only one of the 12.** Measured per-reader:
  `FusionRates` **is** reachable from `PlasmaComposition` (so the edge
  `FusionRates → PlasmaComposition` closes a loop) and **is not** reachable from
  `AuxiliaryPhysicsQuantities`. Crucially, the loop it closes is **already an SCC**:
  `{DensityProfile, FusionRates, PlasmaComposition, ParabolicOnAxisDensities}`, the one
  `mda.CUTS`'s `.physics.proton_rate_density` already cuts into a `FixedPoint` driven by
  `PicardDriver`. So closing this edge **adds an edge inside an existing driven block; it
  does not create a new SCC and does not cross a subsystem boundary.**
- **Second-order consideration**: `PlasmaComposition` branches on
  `fusden_alpha_total < 1e-6` as a "not yet calculated" bootstrap
  (`functional_process/models/physics/physics_B_composition.py:203-210`). Turning that from a
  seeded boundary constant into a driven unknown makes the `PicardDriver`'s starting guess
  load-bearing on a branch, not just on a value.

#### (b9), (b10) `.heat_transport.helpow`, `.heat_transport.p_cryo_plant_electric_mw`

- **Producer**: `Power.calculate_cryo_loads`, `process/models/power.py:1049-1108` —
  **called on this run's path**, `stellarator.py:168`.
- **Registered readers**: `Bldgs`, `CryogenicSystemCost` (`helpow`); `Acpow`,
  `PlantElectricProductionReactor`, `AuxiliaryComponentCoolingCost`
  (`p_cryo_plant_electric_mw`).
- **Why (b)**: **ported but deliberately unregistered.** `Cryo` and `CryoLoads` exist in
  `functional_process/models/power_B_thermal_cryo.py` (as of this audit at `:1491` and
  `:1531`; that file is being edited concurrently, so treat the line numbers as indicative
  and the class names as authoritative) and both declare these `Output`s.
  `total_process.py:1127-1134` records exactly why they are not registered: each reads and
  owns `.fwbs.qnuc` (and `CryoLoads` also `.power.qss`/`qac`/`qcl`/`qmisc`), confirmed
  directly via `to_graph`, and no `FixedPointFunction` split has been written. Same Shape-B
  gap `next_steps.md` §5 tracks; `unit_registry.md`:107 records it too.
- **Cycle risk: none.** Measured over their full declared input sets: the owned inputs are
  `.structure.coldmass` (`StructureMasses`), `.fwbs.p_tf_nuclear_heat_mw`
  (`DetailedPowerflowBlanketShieldPower`), `.times.t_plant_pulse_plasma_present`
  (`PulseDurations`), `.tfcoil.c_tf_turn` (`WindingPackTotalSizePost`), `.tfcoil.n_tf_coils`
  (`StellaratorScalingFactors`) — none reachable from any of the five readers.
- **Prerequisite**: `.tfcoil.tfcryoarea` (see (c1)) is one of their inputs and has no
  producer. Registering these without (c1) trades two boundary inputs for one new one.

## 5. Ranking by cycle risk — the structural answer

The check is not "does the formula look coupled" but: build the node-level DAG
(`producer → consumer` for every owned variable), take the readers of the boundary
variable, take their transitive descendants, and ask whether the would-be producer's own
inputs are owned by anything in that set. Anything in that set means closing the edge
creates a cycle.

| finding | closes a cycle? | which |
|---|---|---|
| `.physics.fusden_alpha_total` | **yes** | `FusionRates → PlasmaComposition`, **inside the existing `{DensityProfile, FusionRates, PlasmaComposition, ParabolicOnAxisDensities}` SCC** already cut by `mda.CUTS` |
| `.physics.fusden_total` | no | reader `AuxiliaryPhysicsQuantities` only |
| `.tfcoil.len_tf_coil` | no | producer inputs owned by `StellaratorScalingFactors`, upstream of all four readers |
| `.physics.p_plasma_inner_rad_mw` | no | |
| `.fwbs.m_blkt_li2o` / `beryllium` / `steel_total` / `vanadium` | no | |
| `.fwbs.whtshld` / `wpenshld` | no | |
| `.heat_transport.helpow` / `p_cryo_plant_electric_mw` | no | |

**So: closing every one of the 12 would leave the "zero SCCs crossing a subsystem
boundary" observation intact.** The 12 multi-node SCCs in the driven graph today are
`DuctDiameterRootFind`, `EtathLiqStep`, `TempTurbineCoolantInStep`, `IonVolAvgTemperature`,
`{WindingPackIntersectInputs, Intersect, WindingPackTotalSizePost}`,
`{DensityProfile, FusionRates, PlasmaComposition, ParabolicOnAxisDensities}`,
`{Divertor, AFwTotalWithPowerflow}`, `CplifeAvail`, `PFwDivHeatDepositedMwStep`,
`PFwBlktCoolantPumpMwStep`, `DeltaEtaStep`, `EtaTurbineStep` — and none of the 12 findings
joins any two of them.

**Caveat on the strength of that claim, stated rather than glossed.** The reachability
check is over *currently registered* nodes. Two of the 12 need genuinely new nodes (S4's
masses, `st_phys:2002-2054`) whose own further inputs were enumerated from the PROCESS
source and checked, so those are sound. But the check cannot see coupling that would only
appear once *other*, still-unregistered producers are added — most obviously the whole
`.pf_coil`/`.pf_power` subsystem, which on a stellarator PROCESS genuinely never runs, and
`Power.pfpwr`, whose output `.pf_power.ensxpfm` is an input to `CryoLoads`. The honest
statement is **"no cross-subsystem cycle appears among the producers this configuration
actually executes"**, not "PROCESS has no cross-subsystem feedback".

## 6. Two things found on the way that are worse than a missing producer

### 6.1 `ProfileValues.rho` is bound to a field PROCESS never writes — the `q95`/`iotabar` bug class again

`.neoclassics.r_eff` sits in the 207 "never assigned" list, and by the letter of the
classification that makes it a genuine input. It is not.

PROCESS's only call site is
`process/models/stellarator/neoclassics.py:289-293`:

```python
self.init_neoclassics(
    0.6,
    self.data.stellarator_config.stella_config_epseff,
    self.data.stellarator.iotabar,
)
```

The first argument, `r_effin`, is the **literal `0.6`**. It is used at `:46`
(`init_profile_values_from_PROCESS(r_effin)`) and never stored. The field
`.neoclassics.r_eff` exists (`process/data_structure/neoclassics_variables.py:87`,
`r_eff: float = 0.0`) and is written **nowhere in `process/`** — the only field of the 379
that is never even *mentioned* outside `data_structure/`.

This port's `ProfileValues` binds
`rho=Input(lambda s: s.neoclassics.r_eff)`
(`functional_process/models/stellarator/neoclassics.py:649`). Its own docstring is candid
about the situation ("even though the `rho=0.6` argument used at that one call site is
itself a literal, not read from `data`") but the binding still resolves, at run time, to
the boundary seed — which is `0.0`, PROCESS's untouched default, not `0.6`.

It matters. Evaluating the ported `calculate_profile_values` at both values with the same
other arguments:

```
rho=0.0  densities[:2]=[1.0e20, 5.0e19]  dr_densities[:2]=[-0.0, -0.0]
rho=0.6  densities[:2]=[8.0e19, 4.0e19]  dr_densities[:2]=[-4.17e19, -2.08e19]
```

The radial derivatives are identically zero at `rho = 0`. Everything downstream of
`.neoclassics.dr_densities`/`dr_temperatures` is therefore computed at the wrong point in
the profile. **Not fixed here** (no edits), but the fix is one line and is exactly the
shape `next_steps.md` §8.3 already settled once for `ConfinementTime`'s `q95`/`iotabar`:
`rho` is not an `Input` at all, it is a static `0.6` on the node — which also removes
`.neoclassics.r_eff` from the boundary set entirely.

### 6.2 `mda_harness.compare` silently drops every array-valued output

The reason §6.1 has never shown up as a harness disagreement:

```python
try:
    got_f = float(np.asarray(got))
    expected_f = float(np.asarray(expected))
except (TypeError, ValueError):
    continue  # non-scalar or non-numeric field, skip
```
— `functional_process/mda_harness.py:695-699`.

That `continue` is *before* any bookkeeping. A non-scalar output is counted as **neither**
`agreements`, `disagreements`, `unverifiable`, **nor** `errors`. It vanishes.

Measured, over `driven_graph(GRAPH).owners` resolved against a `DataStructure()`:
**487 owned variables — 430 scalar-comparable, 28 with no backing field (the 21 the harness
already reports as errors, plus mints), and 29 array-valued and silently skipped.**

```
.impurity_radiation.n_charge_impurity_profile        CalculateEffectiveChargeIonisationProfiles
.neoclassics.densities / temperatures /
  dr_densities / dr_temperatures                     ProfileValues            <- §6.1 hides here
.physics.fusrat_plasma_dt_profile / dd_triton /
  dd_helion / dhe3_profile                           FusionRates
.physics.gradient_length_ne / gradient_length_te     ParabolicGradientLengths
.physics.n_charge_plasma_effective_profile           CalculateEffectiveChargeIonisationProfiles
.physics.pres_plasma_electron_profile / fuel /
  ion_total / thermal_total / thermal_vol_avg        ProfileFactors
.physics.temp_plasma_electron_line_avg_kev           ParabolicProfileValues
.power.*_profile_mw  (11 fields)                     PlantElectricProductionReactor
```

Six percent of the graph's outputs are outside the harness's field of view, and nothing in
its report says so. The minimal honest fix is to compare arrays elementwise where shapes
match and, where they do not, record a fourth outcome rather than `continue` — so that a
skipped output is at least *visible*. (Not made here: `mda_harness.py` is harness code and
the brief was investigation-only.)

### 6.3 Observed, not diagnosed: the harness has moved since §8.4

`run_mda_harness.py`, run once at the end of this audit:
**404 agreements, 32 disagreements (0 in driven blocks), 3 unverifiable, 0 ungrounded
inputs, 21 errors, 55 static switch kwargs checked / 0 mismatched.** `next_steps.md` §8.4
records 237/2/65/2/11. The agreement count rising is expected (§9 registered ~43 cost
nodes via `i_cost_model = 0`), but **`PlantElectricProductionReactor` is now 9 fields off,
worst `.heat_transport.p_plant_electric_base_total_mw` at `rel_diff = 1.968e-01`, and
`.costs.coe` is off by `1.733e-02` where §9 recorded `1.704e-06`.** A concurrently-running
session owns `power_B_thermal_cryo.py` and `core/solver/drivers.py`, so this is recorded as
an observation with a timestamp and explicitly **not** attributed to anything; whoever
picks it up should re-run before diagnosing.

## 7. Prioritised closing plan

Ordered by (value ÷ risk), with the expected harness effect stated so it can be checked
rather than assumed.

1. **`.neoclassics.r_eff` → static `0.6` on `ProfileValues`** (§6.1). One line, no new
   node, no cycle, removes a boundary input, and fixes a *wrong answer* rather than a
   missing edge. Do this first because it is the only item that is currently incorrect
   rather than merely incomplete. Expected harness effect: **none visible** until item 2 is
   done — which is the point.
2. **Make `mda_harness.compare` stop swallowing arrays** (§6.2). Until this exists there is
   no way to verify item 1, and 29 outputs stay unmeasured. Expected effect: up to +29
   compared outputs; at least `ProfileValues`'s four should move from invisible to
   agreeing-after-item-1.
3. **`.fwbs` S4 masses — 6 fields, one unit, zero cycle risk.** Port
   `stellarator.py:1045-1274`'s mass block as ordinary `ExplicitFunction`s reading
   `FwBlanketShieldGeometry`'s `vol_blkt_total`/`vol_shld_total`. `unit_registry.md`:63's
   stated blocker (S2/S3 signatures) is discharged. Expected effect: −6 boundary inputs,
   +6 comparable outputs, no new SCC. Watch `BlanketCost`/`ShieldCost`/`Bldgs`, which
   currently consume the seeded values and would start consuming computed ones.
4. **`.tfcoil.len_tf_coil` (+ `.tfcoil.tfcryoarea`)** — a `ZTfInsideHalf`-shaped node pair
   in the same file, formulas already written at
   `functional_process/models/stellarator/coils/calculate.py:1599-1608`. **Needs the design
   decision in (c1)** about `PlasmaFacingCoilArea`'s stale read; do not close it silently.
   Expected effect: −1 boundary input (−2 once `Cryo`/`CryoLoads` land), +2 comparable
   outputs.
5. **`.physics.fusden_total` / `.fusden_alpha_total`** — port `stellarator.py:2002-2054`.
   Cheap arithmetically (two identities on this run), but it is the **one** item that
   touches a driven block, and it turns a `PicardDriver` starting guess into something a
   branch predicate depends on. Do it after 3 and 4 so that a harness move can be
   attributed.
6. **`.physics.p_plasma_inner_rad_mw`** — **blocked on a decision, not on work**: the zero
   clip at `stellarator.py:2152-2155` must be given an owner first, or `PlasmaRadiationPowers`
   would feed an unclipped value into the product. Resolve that (it also unblocks
   `mda_harness.KNOWN_MINT_VALUES`'s deliberately-absent
   `pden_impurity_core_rad_total_mw` entry) and this becomes a one-line node.
7. **`Cryo`/`CryoLoads`** — largest design cost, lowest ratio. Needs the Shape-B
   `FixedPointFunction` split for `.fwbs.qnuc` and `.power.qss`/`qac`/`qcl`/`qmisc`
   (`total_process.py:1127-1134`), *and* item 4 for `.tfcoil.tfcryoarea`. Note the file is
   concurrently owned. Expected effect: −2 boundary inputs, +9 comparable outputs, +1 or +2
   new driven blocks.

Items needing a **design decision** rather than a patch, collected: (4) stale-vs-fresh
`len_tf_coil`; (6) who owns the `pden_plasma_core_rad_mw` clip; (7) the `qnuc`/`q*`
self-loop split. `unit_registry.md`'s recorded duplicate-ownership conflicts
(`CpLifetimeSuperconducting`/`CplifeAvail`, `NuclearHeatingMagnets`/`ScTfCoilNuclearHeating`,
`Build`/`ZTfInsideHalf`) are **not** among them — none of the 12 findings collides with an
existing owner.

## 8. What this audit did not reach

- **The 207 never-assigned were classified in aggregate, not one at a time** (§3), on the
  strength of the AST scan plus the four bucket checks. Four singles were checked
  individually and one of them (`.neoclassics.r_eff`) turned out to be a defect — so the
  aggregate treatment is exactly where another defect of that shape would still be hiding.
  The 138 IN.DAT-settable ones are the safest; the 25 `stellarator_config` and 35 `costs`
  constants were verified structurally (one loader, one constants file) rather than
  per-field.
- **Only value-producing writes were sought.** A field PROCESS assigns *through a
  mechanism the AST cannot see* — `exec`, dict-driven `setattr` beyond the six audited call
  sites, or a C/Fortran extension — would read as never-assigned. No such mechanism was
  found, but the search was for the ones that exist, not a proof that none does.
- **Correctness of the values at the boundary was not checked**, except where §6.1 forced
  it. A boundary input can be a genuine input and still be seeded from the wrong field; the
  MDA harness cannot detect that either, for the same reason it could not detect
  `.neoclassics.r_eff`.
- **The 32 current harness disagreements were not diagnosed** (§6.3) — out of scope, and
  the relevant files are concurrently owned.
- **Other configurations were not measured.** Everything here is
  `REFERENCE_CONFIGURATION`. Every category-(d) row is (d) *for this run*; a
  `blktmodel = 1`, `isthtr = 3`, `i_tf_sup = 0` or tokamak configuration would reclassify
  many of the 46 as real gaps. The (d) table records the switch and the value each time so
  that re-running the classification for another configuration is mechanical.
