# Cold boundary — where every input's value comes from, and why 11 roots go non-finite

**What this file is.** `tokamak_boundary.md` counted what the tokamak graph reads and
does not produce (the boundary). This file answers the next two questions about that
boundary, which turn out to be one question: **where does each entry's value come from
on a cold start** (an `IN.DAT` and `SingleRun.__init__`, no solve, no model pass), and
**why the cold tokamak MDA produces non-finite values at 11 root nodes** even though
every one of its 183 schedule steps runs and every driven block converges. The answer
to the second is a six-variable subset of the answer to the first.

The measurement that motivated this: the cold tokamak MDA (boundary grounded from
`SingleRun.__init__` on `large_tokamak_eval.IN.DAT`; guesses from the cold fields plus
PROCESS's `first_call` literals for the PF pair) runs **183/183 steps with 0 failures
and 0 ungrounded inputs, but leaves 104 owned outputs non-finite, from 11 roots** — a
root being a non-finite output whose declared reads are all finite. Re-running with
0-seeds instead of the `first_call` literals reproduces the same 11 roots, so these are
singularities at cold *input values* (x/0, 0/0, exp of nan), not driver failures.

## Measurement basis

Everything here was measured against `~/jaxgraph` pinned at **`db4f025`** (the current
HEAD) via `git archive`, on the PROCESS working tree carrying the
tokamak wave: **tokamak graph = 203 nodes**, boundary pinned at **349 `input` + 11
`guess`** (`reference_boundary_tokamak.txt`), and the live driven graph matches that
pin exactly (pin-only: none, live-only: none). Note this is the *working tree*: the
last commit alone (`6f85c59d`) assembles 165 nodes and shows a different root set (14),
so this record is meaningless against any other state of the port — regenerate rather
than reuse its numbers.

The 11-root run is on the MDA harness's graph, i.e. after
`mda_harness._without_excluded` (drops `duct_diameter_root_find`): 344 inputs + 10
guesses there. The provenance census below is on the **full** driven graph (349 + 11);
the five inputs and one guess the exclusion removes are all bucket 2 / the `d_duct`
start, and no conclusion moves between the two.

Cold data throughout means: `SingleRun(input_file, "vmcon").data` — `initialise()`
(which calls `initialise_imprad` then `init.init_process`) and `Models()` construction
have run; no model `run()` has.

## Task A — the 11 roots

### The headline

All 11 roots are one defect class: **(a) missing producer**. Each traces to a boundary
input that PROCESS computes in an in-scope model *before first use* within one pass of
`Caller._call_models_once`'s authored order, and that the port has no node for — so the
cold `DataStructure`'s bare default `0.0` flows into a division or an exponent. None is
init-computed-but-degenerate (b): the census below puts all six culprit variables in
bucket 2 (bare default survives `init_process` untouched). None is a genuine cold
singularity of PROCESS itself (c): in every case the PROCESS producer runs earlier in
the same pass than the consumer, and the producer's own first-pass inputs are
non-degenerate file literals and defaults — verified end-to-end by running the full
evaluation (`sr.run()`), after which all six hold finite non-zero values.

**Six boundary zeros, four unported producers, eleven roots:**

| boundary zero (cold `0.0`, bucket 2) | PROCESS producer | producer file:line | producer runs at (`caller.py`) | consumer runs at | roots fed |
|---|---|---|---|---|---|
| `.build.r_tf_inboard_in`, `.build.r_tf_inboard_out` | `Build.calculate_radial_build` | `process/models/build.py:1720-1726`, `:1731-1735` (TF-inside-CS arm `:1692`) | `:288` | `:306` (`cicc_sctfcoil`) | 1–3 |
| `.physics.res_plasma` | `Physics.plasma_ohmic_heating` | `process/models/physics/physics.py:1663-1679` (written back `:768-773`) | `:290` | `:322` (`pulse`, via `v_plasma_loop_burn`) | 4 |
| `.pf_coil.vs_cs_pf_total_burn` | `PFCoil.vsec` | `process/models/pfcoil.py:1706-1708` (called from `run`, `:79`) | `:319` | `:322` (`pulse`) | 4 |
| `.build.dr_fw_inboard`, `.build.dr_fw_outboard` | `Fw.set_fw_geometry` | `process/models/fw.py:347-352` (called from `run`, `:110`) | `:327` | `:345` (`ccfe_hcpb`) | 5–11 |

First-pass non-degeneracy of the producers, for the (c) ruling: `r_tf_inboard_in` =
`dr_bore + dr_cs + dr_cs_precomp + dr_cs_tf_gap` (all file literals or defaults;
`dr_bore = 2.0038`, `dr_cs = 0.5468`, `IN.DAT:62,69`); `r_tf_inboard_out` adds
`dr_tf_inboard = 1.2` (`IN.DAT:74` — a file literal, so PROCESS does *not* need a TF
result to place the TF coil on pass 1); `dr_fw_inboard` = `2*radius_fw_channel +
2*dr_fw_wall` = `2*0.006 + 2*0.003 = 0.018` (both dataclass defaults,
`fwbs_variables.py:291,294`; the run reproduces exactly `0.018`); `res_plasma` is IPDG89 resistivity off file-literal geometry and
temperatures; `vsec` consumes same-pass `pfcoil` outputs. Converged values for the
record: `r_tf_inboard_in = 2.6986`, `r_tf_inboard_out = 3.8986`, `res_plasma =
4.0496e-9`, `vs_cs_pf_total_burn = -280.19` (negative — `calculate_burn_time` takes
`abs()`), `dr_fw_* = 0.018`.

### Per-root evidence

| # | non-finite output | owner (`.tokamak.` elided) | the arithmetic (port) | PROCESS same | cold value | degenerate read(s) |
|---|---|---|---|---|---|---|
| 1 | `.tfcoil.j_tf_coil_full_area` | `cicc_superconducting_tf_coil.tf_current` | `c_tf_total / a_tf_inboard_total` — `functional_process/models/tfcoil/base.py:363` | `process/models/tfcoil/base.py:424` (`tf_current`, def `:375`) | `2.127e8 / 0 = inf` | `.tfcoil.a_tf_inboard_total = 0` (produced) |
| 2 | `.superconducting_tfcoil.f_a_tf_coil_inboard_steel` | `cicc_superconducting_tf_coil.cicc_inboard_areas_and_fractions` | `n_tf_coils * a_tf_coil_inboard_steel / a_tf_inboard_total` — `functional_process/models/tfcoil/superconducting.py:804` | `process/models/tfcoil/superconducting.py:3646-3648` (`tf_cicc_inboard_areas_and_fractions`, def `:3600`) | `16·(−0.0417)/0 = −inf` | same |
| 3 | `.superconducting_tfcoil.f_a_tf_coil_inboard_insulation` | same node | `functional_process/models/tfcoil/superconducting.py:809-811` | `process/models/tfcoil/superconducting.py:3656-3658` | `+inf` | same |
| 4 | `.times.t_plant_pulse_burn` | `pulse.burn_time` | `abs(vs_cs_pf_total_burn)/v_plasma_loop_burn − t_plant_pulse_fusion_ramp` — `functional_process/models/pulse.py:121` | `process/models/pulse.py:302-304` (`calculate_burn_time`, def `:276`, called `:158`) | `abs(0)/0 − 10 = nan` | `.pf_coil.vs_cs_pf_total_burn = 0` **and** `.physics.v_plasma_loop_burn = 0` |
| 5 | `.fwbs.f_a_fw_coolant_inboard` | `ccfe_hcpb.first_wall_coolant_void_fractions` | `π·radius_fw_channel² / (dx_fw_module · dr_fw_inboard)` — `functional_process/models/blankets/hcpb.py:119-122` | `process/models/blankets/hcpb.py:483-490` | `π·0.006²/(0.02·0) = inf` | `.build.dr_fw_inboard = 0` |
| 6 | `.fwbs.f_a_fw_coolant_outboard` | same node | `= inboard` (`hcpb.py:122`) | `hcpb.py:490` | `inf` | same |
| 7 | `.ccfe_hcpb.armour_density` | `ccfe_hcpb.nuclear_heating_magnets` | `DEN_TUNGSTEN·(1 − vffwm)`, `vffwm` the same `/dr_fw_inboard` — `functional_process/models/blankets/hcpb.py:421,424` | `process/models/blankets/hcpb.py:493-497` | `(1 − inf) → −inf` | `.build.dr_fw_inboard = 0` |
| 8 | `.ccfe_hcpb.fw_density` | same node | `den_steel·(1 − vffwm)` — `hcpb.py:425` | `hcpb.py:498` | `−inf` | same |
| 9 | `.ccfe_hcpb.x_blanket` | same node | `(armour_density·fw_armour_thickness + fw_density·(dr_fw_inboard+dr_fw_outboard)/2 + …)/1000` — `hcpb.py:441-446` | `hcpb.py:535-541` | `−inf·0.005 + (−inf)·0 = nan` | both `dr_fw_* = 0` (the `(−inf)·0` term is what makes it `nan` rather than `−inf`) |
| 10 | `.ccfe_hcpb.tfc_nuc_heating` | same node | `e·exp(−a·x_blanket)·exp(−b·x_shield)·m_tf_coils_total` — `hcpb.py:453-455` | `hcpb.py:564-569` (`itart == 0` arm) | `exp(−2.83·nan) = nan` | in-node, downstream of 9 |
| 11 | `.ccfe_hcpb.p_tf_nuclear_heat_mw_unnormalised` | same node | `tfc_nuc_heating·(p_fusion_total_mw/1000)/1e6` — `hcpb.py:457-459` | `hcpb.py:572-576` | `nan` | in-node, downstream of 10 |

Two chains behind the table, measured by walking each zero-valued read up the graph to
the boundary:

- **Roots 1–3.** `.tfcoil.a_tf_inboard_total` is *produced*, by
  `cicc_superconducting_tf_coil.tf_global_geometry` — circular case
  (`i_tf_case_geom` defaults to `0`), `π·(r_tf_inboard_out² − r_tf_inboard_in²)`
  (`functional_process/models/tfcoil/base.py:186-189`; PROCESS
  `process/models/tfcoil/base.py:284`) — and is `0` because **both radii are boundary
  zeros**. So the port's TF geometry node is fine; what is missing is the radial build
  that feeds it. PROCESS even has explicit negative-area error reporting downstream of
  this arithmetic (`superconducting.py:2495-2513`) that a PROCESS run never triggers
  because `build.run` has always gone first.
- **Root 4** needs *both* zeros: with only `res_plasma` restored the expression is
  `abs(0)/v − 10 = −10` (finite garbage); with only `vs_cs_pf_total_burn` restored it
  is `280/0 = inf`. `v_plasma_loop_burn` itself is produced in-port
  (`plasma_inductance.volt_seconds`: `plasma_current · res_plasma ·
  f_c_plasma_inductive`, `functional_process/models/physics/plasma_inductance.py:204`;
  PROCESS `physics.py:4878` in `calculate_volt_second_requirements`, def `:4768`) — it
  is zero only because `res_plasma` is.

A caution on reading "11 roots, 104 non-finite": the same six zeros also poison a lot
of *finite* output. `r_b_tf_inboard_peak = −0.078`, `a_tf_coil_inboard_case =
−0.0836`, `m_tf_coils_total = −8.09e5` are all downstream of the TF zeros and all
garbage while passing every finiteness check. The root count measures where the
non-finiteness *enters*, not how much of the cold answer is wrong.

## Task B — the provenance census

### Method

For each of the 349 pinned `input` entries (guesses excluded — a `^guess.*` port has
no `DataStructure` field), three values were compared elementwise: a bare
`DataStructure()`, a snapshot deep-copied at the instant `SingleRun.initialise()` had
finished but `Models()` had not yet been constructed (taken by monkeypatching
`Models.__init__`), and the final cold `SingleRun(...).data`. The `IN.DAT` was parsed
for assigned names (lhs lowercased; `name(k)` → element `k−1`;
`f_nd_impurity_electrons(k)` is the input spelling of storage
`.impurity_radiation.f_nd_impurity_electron_array[k−1]`; a whole-array pin row counts
as in-file when its elements are assigned — `zref` is the one case). Buckets:

1. **file literal** — the name is assigned in `large_tokamak_eval.IN.DAT`;
2. **bare default** — not in the file, cold value == the dataclass default;
3. **init-computed** — not in the file, cold ≠ default, and the change is present in
   the post-`initialise()` snapshot;
4. **written during `SingleRun.__init__` by something other than `initialise()`** —
   cold ≠ default but == the snapshot (i.e. a `Models()` construction side effect).

### The counts

| bucket | n | |
|---|---|---|
| 1 file literal | **90** | 89 by direct name + `zref` (elements `zref(1..10)`, `IN.DAT:254-263`) |
| 2 bare default | **252** | includes all six of Task A's zeros |
| 3 init-computed | **7** | full list below |
| 4 `Models()`-written | **0** | measured, not assumed — the snapshot equals cold on all 349 |
| | **349** | |

Twenty of the 90 file literals assign a value **equal to the dataclass default**
(`n_tf_coils = 16`, `q0 = 1.0`, ten zero `f_nd_impurity_electrons` entries, …) — each
was checked against the parsed file value and all are benign restatements, not lost
assignments.

### Bucket 3 — the whole init debt, by name

| variable | writer | what it is |
|---|---|---|
| `.impurity_radiation.m_impurity_amu_array` | `initialise_imprad`, `process/models/physics/impurity_radiation.py:27` (element writes `:322`), called from `SingleRun.initialise` (`process/main.py:430`, *before* `init_process` at `:432`) | species masses, from the packaged element data |
| `.impurity_radiation.temp_impurity_keV_array` | same, `:374` | Lz-table temperature grid, read from `process/data/lz_non_corona_14_elements/` |
| `.impurity_radiation.pden_impurity_lz_nd_temp_array` | same, `:375` | Lz radiative-loss tables, same files |
| `.impurity_radiation.impurity_arr_zav` | same, `:376` | average-charge tables, same files |
| `.divertor.n_divertors` | `init.init_process`, `process/core/init.py:607-617` | `2` → `1` from `i_single_null = 1` (`IN.DAT:307`) |
| `.tfcoil.eff_tf_cryo` | `process/core/init.py:931-936` | sentinel `−1` → `0.13` (ITER cryoplant efficiency for SC TF) |
| `.physics.f_nd_beam_electron` | `process/core/init.py:1142-1147` | `0.005` → `0.0` (no NBI on this run) |

Bucket 4 is **empty**. The recorded stellarator example of bucket 3 (`iohcl` from
`st_init`) has no tokamak counterpart on this run: everything init writes to this
boundary is either device-independent data loading (the four impurity tables) or a
one-line switch-derived default (the last three).

### The supplementary measurement the bucket definitions cannot see

Bucket 4 as defined ("a model writes it before first use") cannot appear in a cold
diff — no model has run. The operational form of the question is: **which boundary
inputs does a PROCESS run overwrite?** Running the full evaluation (`sr.run()` on the
same `SingleRun`; `i_process_run_mode = -2`, fsolve over the two `ixc` variables) and
diffing the 349 before/after:

**31 of the 349 change.** Two are the run's own iteration variables
(`.physics.temp_plasma_electron_vol_avg_kev`, `.physics.nd_plasma_electrons_vol_avg` —
`ixc = 4, 6`; the solver moved them, not a model). The other **29 are model-written
boundary inputs — the port's real missing-producer exposure**, of which Task A's six
are exactly the members whose cold value is a *degenerate* zero:

```
.blanket.deg_blkt_inboard_poloidal_plasma     0        -> 127.797
.build.dr_cs_bore                             1.42     -> 2.00384
.build.dr_fw_inboard                          0        -> 0.018      <- Task A
.build.dr_fw_outboard                         0        -> 0.018      <- Task A
.build.dr_tf_inner_bore                       0        -> 11.6798
.build.dz_blkt_upper                          0        -> 0.85
.build.dz_tf_upper_lower_midplane             0        -> -1.23388
.build.r_tf_inboard_in                        0        -> 2.69861    <- Task A
.build.r_tf_inboard_mid                       0        -> 3.29861
.build.r_tf_inboard_out                       0        -> 3.89861    <- Task A
.build.z_tf_top                               0        -> 8.78433
.buildings.dz_tf_cryostat                     2.5      -> 5.65835
.costs.c2214                                  0        -> 44.7704
.costs.c2222                                  0        -> 643.072
.costs.c2252                                  0        -> 39.6536
.fwbs.dewmkg                                  0        -> 1.44940e7
.fwbs.p_div_rad_total_mw                      0        -> 15.4195
.heat_transport.peakmva                       0        -> 79.2176
.pf_coil.p_pf_electric_supplies_mw            0        -> 3.42872
.pf_coil.vs_cs_pf_total_burn                  0        -> -280.188   <- Task A
.pf_power.ensxpfm                             0        -> 29349.1
.pf_power.srcktpm                             0        -> 971.625
.physics.beta_poloidal_vol_avg                0        -> 1.32822
.physics.dlamie                               0        -> 17.8343
.physics.nd_plasma_pedestal_electron          5e19     -> 6.12234e19   (file literal)
.physics.nd_plasma_separatrix_electron        2e19     -> 3.60137e19   (file literal)
.physics.p_plasma_ohmic_mw                    0        -> 0.603722
.physics.pflux_plasma_surface_neutron_avg_mw  0        -> 1.09115
.physics.res_plasma                           0        -> 4.04956e-9  <- Task A
```

Two of the 29 are **file literals a model overwrites**:
`nd_plasma_pedestal_electron` / `nd_plasma_separatrix_electron`, written by the
pedestal profile arm (`process/models/physics/profiles.py:318-325`, the
Greenwald-fraction scaling). This is the tokamak's measured instance of exactly the
defect `mdf.seed`'s docstring records on the stellarator — there it was the *L-mode
reset* of the same variable pair (`plasma_profiles.py:110-117`) that the port lacked,
found because a cold and a converged `DataStructure` disagreed in precisely those two
entries. Same variables, opposite profile arm, third method of detection.

The remaining 23 are stale-but-finite on a cold start: the silent form of what the 11
roots make loud.

## Method notes

- Probes: a per-step cold-MDA runner with root attribution (a copy of the
  orchestrating session's `cold_mda.py`), a zero-chase walker (from each root's
  zero-valued read, recurse through producers to the boundary-zero frontier), and the
  census script (bare/`after-initialise`/cold three-way diff plus the `sr.run()`
  overwrite diff). All are session scratchpad scripts, deliberately not committed —
  each is ~100 lines over `functional_process.mda`/`mda_harness` internals and this
  record carries the method; the numbers should be regenerated against whatever state
  needs auditing, not replayed.
- The frontier walk terminates fast: every non-finite chain bottoms out in the six
  boundary zeros within two hops (`a_tf_inboard_total` and `v_plasma_loop_burn` are
  the only produced-but-zero intermediates).
- `Blocking.scc` + `schedule_for` on the excluded driven graph: 183 steps, 0
  failures, 0 cascades — the failure mode of this graph is *silently finite-looking
  wrong numbers plus 104 loud non-finite ones*, not exceptions.
- One flag worth keeping: cold `.build.dr_cs_bore = 1.42` is neither the file value
  nor stale-innocent — the dataclass default happens to be non-zero, and a run moves
  it to `dr_bore = 2.00384`. Non-zero defaults make stale reads harder to spot than
  the zeros that produced the 11 roots.

## What this implies

**For §13.8 (`indat_to_python`).** 342 of the 349 boundary inputs — 98% — are pure
data on a cold start: 90 file literals plus 252 dataclass defaults that survive
`init_process` untouched. An `indat_to_python` emitter plus the existing
`data_structure` defaults therefore *already* determine nearly the whole boundary
without any PROCESS machinery. The entire irreducible `init_process` dependency at
this boundary is **seven variables**: four are impurity data tables loaded from
packaged files (`initialise_imprad` — a data asset to ship, not init logic to port),
and three are one-line switch-derived rules (`i_single_null` → `n_divertors`;
`eff_tf_cryo` sentinel; `f_nd_beam_electron` zeroing) that belong in the machine
factory next to the switches they read. "Fully cold without PROCESS" is seven
variables away, and four of the seven are file reads.

**But cold-correct is not converged-correct.** The census's buckets say where a value
comes from *before the run*; the 29-name overwrite list says which of those values
PROCESS would replace *during* one. On a cold start the port reads a stale value for
all 29 — six of them degenerate zeros that generate every one of the 104 non-finite
values through 11 roots and exactly **four unported producers**. In fix order by
payoff: `Fw.set_fw_geometry` (`fw.py:347-352` — two lines of arithmetic) removes 7 of
the 11 roots; the `r_tf_inboard_in/out` slice of `Build.calculate_radial_build`
removes 3 more; `res_plasma` (`plasma_ohmic_heating`) and `PFCoil.vsec` together
remove the last. The 23 finite-but-stale entries are the same defect without the alarm
— the two overwritten *file literals* among them (the pedestal density pair) being the
proof, on the tokamak this time, that no input-file-based check can see this class.
The boundary pin counts growth; the overwrite diff is the instrument that says which
of the standing 349 are real inputs and which are producers the port still owes.


## Addendum, 2026-08-27 (same day, cold-boundary wave): the four producers landed

The fix order above was executed in full, in payoff order, on this working tree:

1. `Fw.set_fw_geometry` -> `models/fw.py::FirstWallGeometry`, new `Tokamak` slot
   `first_wall_geometry` (11 -> 4 roots, 104 -> 56 non-finite).
2. The CS-to-TF radial slice -> `models/build.py::TfInboardRadiiTfOutsideCs`, new
   `Build` slot `tf_inboard_radii` -- taken at `build.py:1691` rather than `:1720`, so
   `dr_cs_bore` (a standing stale input, `1.42` cold for `2.00384` converged, read by
   `CSFluxSwing`) and `dr_cs_precomp` are produced too (4 -> 1 root).
3. `Physics.plasma_ohmic_heating` -> `models/physics/physics.py::PlasmaOhmicHeating`,
   fifth `TokamakPhysics` slot `ohmic_heating` -- chained-comparison defect reproduced.
4. `PFCoil.vsec` + the `c_pf_coil_turn` tail of `pfcoil()` ->
   `models/pfcoil/volt_seconds.py` (unit #52), two new `PFCoil` slots. Registering it
   merged the PF ring and the volt-second/burn-time ring into one nine-node SCC whose
   cut is the standing `mda.CUTS` trio, measured sufficient and minimal
   (`test_mda.py::test_the_merged_pf_volt_second_burn_time_cycle_keeps_its_cuts`).

Post state, same probe, same input: **185/185 steps, 0 failures, 0 ungrounded,
0 non-finite, 0 roots.** Boundary pin moved 349+11 -> 347+11: nine rows closed
(the six zeros of Task A plus `dr_cs_bore`, `p_plasma_ohmic_mw`, and `r_tf_inboard_mid`)
against seven genuine new reads (`dr_bore`, `dr_cs_tf_gap`, `fseppc`, `fcspc`,
`sigallpc`, `dr_fw_wall`, `plasma_res_factor` -- all file literals or bucket-2
defaults, none degenerate in a division or exponent). Warm harness: tokamak 597 -> 611
agreements (+14 = exactly the new owned outputs, all agreeing), disagreements 16 -> 16
(same members), errors 20 -> 20; stellarator bit-identical (472/34/3/25). The 29-name
overwrite list above now stands at 20 (nine closed); the census's bucket counts are
for the pre-wave tree and should be regenerated, not patched.
