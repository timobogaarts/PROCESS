# Does assembly actually consult the switches `UNPORTED` refuses?

**Measured 2026-08-31** against the working tree at `715dbe48`, all seven
`run_cold_matrix.CONFIGURATIONS`. Analysis only: nothing in `functional_process/` or
`process/` was changed by this pass, and the guard test it proposes is proposed, not
added.

## The question

`indat.UNPORTED` holds 219 `(field, value)` rows across 50 switch axes, and a file that
selects one of them is refused at assembly. That mechanism works. It is not the whole
story, because **a refusal only fires where a `_slot_occupant` call for that field is
reached.** A node registered on a branch that never asks the switch computes one arm
unconditionally, and assembly sees nothing to refuse.

The proven instance the brief was written around — `helias_5b.IN.DAT` setting
`i_p_coolant_pumping = 0` and assembling anyway, with
`stellarator_fwbs_s2.py` always computing `FRACTION_OF_HEAT` and pump power landing at
15.58 MW against PROCESS's 176.0 — is the `EcrhDensityLimit` class again: a node whose
formula is wrong for the configuration actually being run.

**That instance is now closed** (a concurrent pass landed
`_blanket_shield_power_arm(blktmodel, ipowerflow, i_p_coolant_pumping)` and
`DetailedPowerflowBlanketShieldPowerUserInputPumping` while this audit was running; the
occupant `helias_5b` now assembles is measured below). This record is about **how large
the class is**, and the answer is: **small, but not empty, and it does not live where the
brief expected.**

## Method, and exactly what each column is

Four probes, all run through `~/miniconda/envs/process_port/bin/python` with
`jax_enable_x64` on before any array:

1. **Consultation trace.** `indat._slot_occupant`, `indat._refuse_unported_switch` and
   `indat.switches_from_indat` are monkey-patched (in a scratch script, not in the tree)
   so that one `machine_from_indat` call per configuration records (a) every
   `(field, value)` a slot was resolved against and (b) every `IN.DAT` key the assembly
   actually read. **Measured**, per configuration.
2. **Refusal-enforcement sweep.** For every `UNPORTED` row whose field is a real `IN.DAT`
   switch name (22 axes, 96 `(field, value)` pairs), each of the seven files is rewritten
   with that switch forced to that refused value and re-assembled. A pair that assembles
   is a refusal the configuration does not enforce. 672 assemblies. **Measured.**
3. **Pinned-kwarg introspection.** `machine_survey.pinned_switches(graph)` on each
   assembled graph, compared against the file's own value for the same name.
   **Measured.**
4. **Discrimination — is the unenforced row a defect?** For each unenforced axis, grep
   PROCESS for readers *restricted to the call path the configuration actually takes*.
   `Stellarator.run` (`process/models/stellarator/stellarator.py`) calls `st_*` plus
   `costs`, `availability`, `power.tfpwr` / `component_thermal_powers` /
   `calculate_cryo_loads` / `acpow` / `plant_electric_production`, `buildings` and
   `vacuum` — and nothing in `physics.py`, `current_drive.py`, `fw.py`, `hcpb.py` or
   `divertor.py`. A switch no module on that path reads is one PROCESS ignores too, and
   an unenforced refusal for it is correct, not a gap. **This column is inference from a
   grep plus a reading of `Stellarator.run`, not a run.**

### Blind spots of the method

- **Derived arm indices cannot be swept.** Twenty-eight of the fifty axes
  (`pf_coil_system_arm`, `tf_stress_arm`, `first_wall_arm`, …) are keyed on an arm index
  computed inside `machine_from_indat`; no `IN.DAT` names them, so probe 2 cannot reach
  them. They are covered only by probe 1, which shows *where* the dispatch runs, not
  whether it should have run somewhere else. This is
  `machine_survey.assembly_verdict`'s own recorded blind spot, unchanged.
- **Assembly is not evaluation.** Every finding here is about which occupant class the
  factory selects. A node that selects correctly and then computes the wrong arm
  *internally* (a `jnp.where` on a static field with the wrong value) is invisible to all
  four probes except through probe 3.
- **One switch per perturbation.** Probe 2 changes one integer at a time. A refusal that
  only fires on a *combination* the sweep never builds is not tested.
- **The seventh file is not a control.** All seven configurations assemble today, so
  "assembles" carries no information on its own; only the *pair* (refused value, still
  assembles) does.

## The full table — 50 axes

Configuration keys: `sh` stellarator_helias, `h5` helias_5b, `ltn` large_tokamak_nof,
`lte` large_tokamak_eval, `lar` low_aspect_ratio_DEMO, `ste` spherical_tokamak_eval,
`str` st_regression.

| axis | refused arms | assembly dispatches on it in | refused value still assembles on |
|---|---|---|---|
| `blktmodel_blkttype` | 2 | sh h5 | derived arm; not sweepable |
| `blktmodel_ipowerflow_i_p_coolant_pumping` | 2 | sh h5 | derived arm; not sweepable |
| `centrepost_neutronics_arm` | 2 | ltn lte lar ste str | derived arm; not sweepable |
| `cicc_turn_geometry_arm` | 2 | ltn lte lar | derived arm; not sweepable |
| `croco_turn_geometry_arm` | 3 | ste str | derived arm; not sweepable |
| `divertor_geometry_arm` | 1 | ltn lte lar ste str | derived arm; not sweepable |
| `divertor_heat_load_arm` | 2 | ltn lte lar ste str | derived arm; not sweepable |
| `first_wall_arm` | 2 | ltn lte lar ste str | derived arm; not sweepable |
| `hcd_primary_powers_arm` | 1 | ltn lte lar ste str | derived arm; not sweepable |
| `i_beta_norm_max` | 4 | ltn lte lar ste str | 4/4 on sh h5 |
| `i_blanket_type` | 1 | ltn lte lar ste str | 1/1 on sh h5 |
| `i_bootstrap_current` | 12 | ltn lte lar ste str | 12/12 on sh h5 |
| `i_cost_model` | 2 | all seven | none — enforced everywhere |
| `i_density_limit` | 7 | ltn lte lar ste str | 7/7 on sh h5 |
| `i_diamagnetic_current` | 1 | ltn lte lar ste str | 1/1 on sh h5 |
| `i_ecrh_wave_mode` | 1 | ste str | 1/1 on sh h5 ltn lte lar |
| `i_hcd_calculations` | 1 | ltn lte lar ste str | 1/1 on sh h5 |
| `i_hcd_primary` | 10 | ltn lte lar | 10/10 on sh h5 |
| `i_hcd_secondary` | 11 | ltn lte lar ste str | 11/11 on sh h5 |
| `i_ind_plasma_internal_norm` | 1 | ltn lte lar ste str | 1/1 on sh h5 |
| `i_l_h_threshold` | 15 | ltn lte lar ste str | 15/15 on sh h5 |
| `i_len_sol_outboard_power_decay` | 2 | ltn lte lar ste str | 2/2 on sh h5 |
| `i_p_coolant_pumping` | 3 | ltn lte lar ste str | 2/3 on sh h5 — **see §1** |
| `i_pf_energy_storage_source` | 1 | all seven | none — enforced everywhere |
| `i_plasma_current` | 7 | ltn lte lar ste str | 7/7 on sh h5 |
| `i_plasma_geometry` | 11 | ltn lte lar ste str | 11/11 on sh h5 |
| `i_plasma_ignited_i_rad_loss` | 1 | all seven | derived arm; not sweepable |
| `i_plasma_ignited_separatrix` | 1 | ltn lte lar ste str | derived arm; not sweepable |
| `i_pulsed_plant_istore` | 1 | all seven | derived arm; not sweepable |
| `i_rad_loss` | 2 | all seven | none — enforced everywhere |
| `i_str_wp_i_tf_sc_mat_cicc_sc_properties` | 13 | ltn lte lar | derived arm; not sweepable |
| `i_str_wp_i_tf_sc_mat_croco_sc_properties` | 17 | ste str | derived arm; not sweepable |
| `i_str_wp_i_tf_sc_mat_croco_temp_margin` | 17 | ste str | derived arm; not sweepable |
| `i_str_wp_i_tf_sc_mat_temp_margin` | 14 | ltn lte lar | derived arm; not sweepable |
| `i_tf_inside_cs_vacuum_shield` | 1 | ltn lte lar ste str | derived arm; not sweepable |
| `i_tf_sc_mat` | 1 | sh h5 | 1/1 on ste str |
| `i_tf_sup` | 1 | all seven | none — enforced everywhere; **but see §3** |
| `i_tf_sup_build` | 2 | ltn lte lar ste str | derived arm; not sweepable |
| `ife` | 1 | **no slot** (a bare `_refuse_unported_switch` guard, read on all seven) | none — enforced everywhere |
| `isthtr` | 1 | sh h5 | 1/1 on ltn lte lar ste str |
| `pf_coil_system_arm` | 6 | ltn lte lar ste str | derived arm; not sweepable |
| `plasma_geometry_arm` | 1 | ltn lte lar ste str | derived arm; not sweepable |
| `pulse_ramp_times_arm` | 2 | ltn lte lar ste str | derived arm; not sweepable |
| `structure_arm` | 3 | ltn lte lar ste str | derived arm; not sweepable |
| `surface_poloidal_field_arm` | 1 | ltn lte lar ste str | derived arm; not sweepable |
| `tf_coil_shape_arm` | 2 | ltn lte lar ste str | derived arm; not sweepable |
| `tf_field_and_force_arm` | 1 | ltn lte lar ste str | derived arm; not sweepable |
| `tf_inboard_radii_arm` | 1 | ltn lte lar ste str | derived arm; not sweepable |
| `tf_stress_arm` | 21 | ltn lte lar ste str | derived arm; not sweepable |
| `vacuum_vessel_arm` | 1 | ltn lte lar ste str | derived arm; not sweepable |

### Reading the "still assembles" column

Eighteen of the twenty-two sweepable axes are unenforced on **exactly the two
stellarators or exactly the five tokamaks, and nowhere else.** That partition is not an
accident and it is not (mostly) a defect: it is the device branch in
`machine_from_indat`. Of those eighteen:

- **Fifteen are correct.** No module on the configuration's own PROCESS call path reads
  the switch, so PROCESS ignores it too. Measured by grep over the stellarator call path
  (§ Method probe 4): `i_beta_norm_max`, `i_bootstrap_current`, `i_density_limit`,
  `i_diamagnetic_current`, `i_hcd_secondary`, `i_ind_plasma_internal_norm`,
  `i_l_h_threshold`, `i_len_sol_outboard_power_decay`, `i_plasma_current`,
  `i_plasma_geometry` — all tokamak-physics switches, none read anywhere the stellarator
  run reaches. `isthtr` is the mirror image (stellarator-only, unenforced on the five
  tokamaks). `i_ecrh_wave_mode` is read at exactly one site,
  `process/models/physics/current_drive.py:1767`, inside the `i_hcd_primary == 13`
  branch, so it is correctly unasked on the three files that set `i_hcd_primary = 10`.
  `i_blanket_type == 5` (DCLL) names an occupant of `.tokamak.ccfe_hcpb`, a namespace a
  stellarator does not have. `i_tf_sc_mat == 9`'s row says in its own words that it is
  *"scoped to the critical-surface slots — `WINDING_PACK_MATERIAL` and
  `COILS_MASS_MATERIAL`"*, which are the stellarator coil slots; the two ST files reach
  the CroCo registries instead, which do hold `(i_str_wp = 1, 9)`.
  `i_hcd_calculations == 0` is forced by PROCESS itself
  (`process/models/stellarator/initialization.py:39` assigns it unconditionally) and no
  stellarator node computes current drive at all.
- **One is now fixed** — `i_p_coolant_pumping`, §1.
- **Two are the class this audit was asked to bound** — but neither was found by the
  sweep. They were found by probe 3, and they are §2 and §3.

---

## §1. `i_p_coolant_pumping` — the seed instance, closed while this ran

`i_p_coolant_pumping` values 0 and 1 still assemble on both stellarators, and that is now
**correct**: `_blanket_shield_power_arm` became a three-switch joint dispatch and
`BLANKET_SHIELD_POWER` gained arm 3. Measured on the current tree:

| file | `i_p_coolant_pumping` | `.stellarator.fwbs.blanket_shield_power` |
|---|---|---|
| `stellarator_helias` | 1 (`FRACTION_OF_HEAT`) | `DetailedPowerflowBlanketShieldPower` |
| `helias_5b` | 0 (`USER_INPUT`) | `DetailedPowerflowBlanketShieldPowerUserInputPumping` |

Value 2 (`MECHANICAL`) and 3 now refuse on the stellarators through the joint arm's
arm 4, which is where PROCESS itself raises (`stellarator.py:924-928`).

**The `UNPORTED` rows `("i_p_coolant_pumping", 0)` and `(…, 1)` have outlived their cause
on the stellarator branch** and now bind only the tokamak `PUMPING_POWER` slot. They are
still correct there; the reason text describes `hcpb.py`, not `stellarator.py`, so
nothing is stale in what it *says*. Recorded because the same pair of rows now means two
different things depending on which branch reaches them, and the next reader should not
infer from row 0's presence that the stellarator still refuses it.

**This validates the method.** Probe 1 shows `i_p_coolant_pumping` as *read on both
stellarators but resolved through no slot of its own name* — which is precisely the
signature the gap had before the fix, and precisely the signature §2 and §3 still have.

## §2. `PfMagnetCost` is pinned to a central solenoid two tracked files do not have — **FIXED 2026-08-31**

**The one live wrong answer this audit found.**

`indat.py:5244-5255` builds the PF magnet cost occupant with two literals:

```python
build=lambda cls: cls(
    n_cs_pf_coils=N_CS_PF_COILS,                       # 7 = REFERENCE_TOPOLOGY
    iohcl=CentralSolenoidConfiguration.PRESENT,        # 1
    i_pf_conductor=i_pf_conductor,                     # threaded from the file
)
```

`spherical_tokamak_eval.IN.DAT` and `st_regression.IN.DAT` both set **`iohcl = 0`** (no
central solenoid). Measured by probe 3 on their assembled graphs:

```
spherical_tokamak_eval   iohcl  pinned=[1]  <== DISAGREES with file value 0
st_regression            iohcl  pinned=[1]  <== DISAGREES with file value 0
```

And the same machine's PF coil *system* reads the switch correctly —
`indat.py:3935`, `topology = REFERENCE_TOPOLOGY if has_cs else
SPHERICAL_TOKAMAK_TOPOLOGY`. So one assembled machine holds both answers:

| | PF coil system (`_pf_coil_system_arm`) | `PfMagnetCost` (pinned) |
|---|---|---|
| central solenoid | absent (`iohcl = 0`) | present (`iohcl = 1`) |
| `n_cs_pf_coils` | 8 (`SPHERICAL_TOKAMAK_TOPOLOGY`) | 7 (`N_CS_PF_COILS`) |

`costs.py` uses both: `_n_pf_coils_costed(n_cs_pf_coils, iohcl)` returns
`n_cs_pf_coils - 1 if iohcl == 1 else n_cs_pf_coils` (`costs.py:1858-1866`), and two
separate CS cost terms are gated on `is_superconducting and iohcl == 1`
(`:1947`, `:2036`). So on both ST files the port costs **6 PF coils plus a central
solenoid that does not exist**, where PROCESS costs 8 PF coils and no solenoid.

**Both files sit in `reference_cold_matrix.txt` as converged.** The cost error does not
feed the PF coil system, so it will not move the build; it moves `.costs.*` and anything
downstream of them, which on an `i_figure_merit` keyed to cost is the objective itself.

**The comment above the call is stale and says so out loud**: *"`-1` refuses
`iohcl != 1` … so `N_CS_PF_COILS` is the count and not a guess."* That refusal was
retired by the no-central-solenoid pass (commit `253c426a`, *"one refusal that outlived
its cause"*), and the pin it was justifying was not revisited. **This is the same failure
shape as the `i_p_coolant_pumping` instance, arrived at from the opposite direction: not
a refusal that never fired, but a refusal that stopped firing and left a hardcoded answer
standing behind it.**

**Closed 2026-08-31.** Not by threading the switch — by a second occupant, which is
what §14.2 requires ("no switch is a static kwarg, whatever its reads") and what the
reads independently say: four fields (`.pf_coil.i_cs_superconductor`,
`.pf_coil.a_cs_cable_space`, `.pf_coil.f_a_cs_void`, `.pf_coil.fcuohsu`, plus
`.pf_coil.j_crit_str_cs` on the `PER_KAM` arm) are read *only* inside `acc2222`'s
`iohcl == 1` block, and `provider`'s pin records `a_cs_cable_space` as **`unwritten`** on
`spherical_tokamak_eval` — the old node declared an edge to a field the run never fills.
`PF_MAGNET_COST` is now keyed on `_pf_magnet_cost_arm(supercond_cost_model, iohcl)` over
four occupants (`PfMagnetCostPerKg{,NoCentralSolenoid}` and the `PerKam` pair);
`n_cs_pf_coils` stays a static kwarg (a count, not a switch) and comes from the shared
`_pf_coil_topology(iohcl)` the PF coil system is measured against, so the two cannot
disagree again.

Measured against PROCESS's own cold-start `.costs.c2222`:

| file | before (pinned `iohcl=1`, `n=7`) | after | PROCESS |
|---|---|---|---|
| `spherical_tokamak_eval` | 404.666198 | **425.364301** | 425.3643005724115 |
| `st_regression` | 502.801374 | **528.814606** | 528.8146061533076 |
| `large_tokamak_nof` (control, `iohcl=1`) | 591.848936 | 591.848936 | 591.8489361664515 |

`c22221` and `c22222` are the two terms that move; `c22223`/`c22224` do not depend on
the loop bound. The guard §6 proposes landed with the fix as
`tests/functional_process/test_switch_coverage.py::
test_no_pinned_switch_contradicts_its_own_input_file`, and failed on exactly the two ST
rows before it.

## §3. Two stellarator nodes assume a superconducting TF that nothing refuses — **LATENT**

`i_tf_sup == 2` (aluminium) is refused on all seven files. `i_tf_sup == 0` (resistive) is
**not in `UNPORTED` at all**, and the only slot a stellarator dispatches it into is
`TF_POWER`, which holds an arm 0. Measured: `stellarator_helias.IN.DAT` with
`i_tf_sup = 0` appended **assembles**, while all five tokamaks refuse it through
`i_tf_sup_build == 0`.

Diffing the two assembled stellarator trees (`i_tf_sup` 1 → 0) gives exactly three
occupant changes, all in `.power`:

```
.power.cryo_q_loads   CryoQLoadsSuperconductingTf -> CryoQLoadsResistiveTf
.power.cryo_q_nuc     CryoQNuc                    -> None
.power.tf_power       TfPowerSuperconducting      -> TfPowerResistive
```

Everything else is unchanged — including:

- `.stellarator.fwbs.blanket_shield_power` = `DetailedPowerflowBlanketShieldPower`, whose
  own docstring says *"`i_tf_sup != SUPERCONDUCTING` — dropped … This function always
  computes `p_tf_nuclear_heat_mw` as if `i_tf_sup == SUPERCONDUCTING`"*
  (`stellarator_fwbs_s2.py:197-199`), and whose module docstring names the same drop at
  line 32;
- `models/stellarator/tf_nuclear_heating.py`, which states at line 5 that it *"ports only
  the SUPERCONDUCTING branch of the source's `i_tf_sup` switch"*;
- the whole `.stellarator.coils` namespace — `IterNb3snCoilsMass`,
  `IterNb3snWindingPackIntersectInputs`, `QuenchProtection`, `MaximumStress` — a
  superconducting coil set on a machine declared resistive.

So a resistive stellarator assembles and silently receives superconducting TF nuclear
heating and a superconducting coil set. **Latent**: no tracked input file sets
`i_tf_sup = 0` on a stellarator, and both stellarators set 1. It is the same class as
`helias_5b`'s, one refusal short of being live.

Note the asymmetry that makes it invisible to `machine_survey`: the axis reports as
*enforced everywhere* in the table above, because the only value `UNPORTED` names (2) is
in fact refused everywhere. **A partially-enforced axis and a fully-enforced one are
indistinguishable in a table keyed on `UNPORTED`'s rows** — which is the structural
lesson of this audit and the reason §4 proposes the guard it does.

## §4. What was checked and found clean

Recorded so the next pass does not re-derive it.

- **`i_blkt_coolant_type = 2` (water) assembles on all seven.** Correct: PROCESS's
  `CCFEHCPB.run` assigns `CoolantType.HELIUM` unconditionally at
  `process/models/blankets/hcpb.py:45`, so the water arm is dead code, not dormant.
  `blankets/hcpb.py`'s own record already says so.
- **`i_fw_coolant_type = 'water'` assembles on all seven.** Not an instance: the only
  PROCESS readers are `fw.py`'s `fw_temp` and `blanket_library.py`'s coolant properties,
  both CoolProp-bound and neither ported. Nothing in the port assumes an arm; the
  computation is absent, which is a different (and already recorded) gap.
- **`ipowerflow = 0` assembles on all seven.** Correct on tokamaks:
  `grep -rlw ipowerflow process/` returns only the two data-structure modules,
  `stellarator/stellarator.py`, `stellarator/build.py` and `core/input.py`.
  `process/models/power.py` does not read it at all, so there is no tokamak arm to miss.
- **`i_single_null = 0` assembles on all seven**, and 2 is rejected as not a legal enum
  value. Both correct — `_n_divertors` has occupants for both legal arms.
- **`ife`** is the one axis with no slot at all: a bare `_refuse_unported_switch` guard,
  read on every configuration and correctly firing only at `ife = 1`.
- **`BUILDING_SIZING[1]`** pins `i_hcd_primary=CurrentDriveModel.ITER_NEUTRAL_BEAM`
  (`indat.py:2046`) rather than threading the file's value into `BldgsSizes`, whose
  `is_neutral_beam` is derived from it (`buildings/buildings.py:1043-1044`). **Latent and
  weaker than §3**: no tracked file sets `i_bldgs_size = 1` (five are silent, the two ST
  files set 0), so the node is never instantiated. Worth threading when the arm is first
  reached, not before; noted here so it is not re-found as a surprise.

## §5. The size of the class, stated plainly

**Two instances, one live and one latent** (plus one weaker latent, §4's
`BUILDING_SIZING`), out of 50 axes and 219 rows. The class is **small**, and the
enforcement mechanism is in much better shape than the seed instance suggested: 15 of the
18 unenforced axes are unenforced *correctly*, for a reason that grep can confirm in one
line each.

What the two instances share, and what the sweep could not see, is that **neither is a
missing `UNPORTED` row.** Both are hardcoded answers standing where a refusal used to be
(§2) or where no refusal was ever written because the switch's other arm looked like
absence rather than a variant (§3). A table of refusals cannot catch either. Probe 3 —
introspecting the assembled graph's static kwargs and diffing them against the file —
caught both, and caught them in seconds.

## §6. Proposed guard (not added)

`machine_survey.survey` already computes everything needed and then discards it. Its
`pinned` branch — the one that emits `DISAGREES, tree holds [...]` — is only reached for
switches **not** in `factory_fields()`:

```python
if name in fields:          # `iohcl` lands here …
    ...
elif name in pinned:        # … so this never runs for it
```

`iohcl` is in `factory_fields()` (`indat.py:4859` reads it via `switches.get`) *and*
pinned on `PfMagnetCost`, so §2 was invisible to the survey by one `elif`. **The
proposed change is to run the pinned-value check for every switch that is pinned,
whether or not a slot also dispatches on it**, and to report a contradiction as its own
verdict rather than folding it into `factory`.

The test that would have failed on §2, stated for whoever picks this up
(`tests/functional_process/test_switch_coverage.py` is its natural home, beside
`test_no_slot_contradicts_a_factory_switch`, which asks this question of
`REFERENCE_INPUT_FILE` only):

> For each of `provider.CONFIGURATIONS`, assemble the machine, take
> `machine_survey.pinned_switches(graph)`, and assert that every pinned switch whose name
> the file sets holds that file's value.

It is seven assemblies, it needs no PROCESS run, and on the tree as it stood it failed
on exactly two rows (`iohcl` on both ST files) — so it landed *with* §2's fix.

**The `elif` half of this proposal was tried and reverted.** `survey`'s `graph` defaults
to `indat.GRAPH`, the *reference* machine built from `stellarator_helias`, so lifting the
pinned-value check out from behind the `elif` compares every other file's switches
against a stellarator's pins: it immediately reported `large_tokamak_eval`'s
`i_p_coolant_pumping = 3` as contradicted by a `1` that has nothing to do with that file,
and broke three `test_machine_survey` pins that were right. A pinned-value check is only
meaningful against the machine *that file* assembles, which is exactly what the new test
does and what `survey` structurally cannot. `Row.verdict`'s docstring carries this.

A second, weaker guard would catch §3, and is worth stating even though it is more
work: a node that drops a switch arm says so in its docstring
(`stellarator_fwbs_s2.py:197`, `tf_nuclear_heating.py:5`). **Those sentences are the
port's only record that an assumption exists**, and nothing reads them. Turning them into
a declared attribute — `assumes = {"i_tf_sup": SUPERCONDUCTING}` on the node class,
checked at assembly against the file — would make §3's class mechanically detectable
instead of grep-detectable. Out of scope here; recorded as the shape of the real fix.

## Reproducing this

Scratch scripts (not in the tree; rewrite them from the descriptions above if needed):
the consultation trace patches the three `indat` symbols named in probe 1; the sweep
appends `field = value` to a copy of each `IN.DAT` after deleting any existing assignment
of that name, then calls `machine_from_indat`; the pinned check is four lines around
`machine_survey.pinned_switches`. Total runtime, all four probes, under ten minutes.
