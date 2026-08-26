# Tokamak boundary — what a second device is missing, by name

**What this file is.** `tokamak_scope.md` counted the *switch* decisions a conventional
tokamak adds (17 new). `tokamak_call_surface.md` counted the *code* it reaches (38 files,
338 functions, 28 591 entered lines). Neither could count **variables**, because a
boundary is a property of an assembled graph and there was no tokamak graph to take one
of. This file is that count. It is `next_steps.md` §13.9's ask and
`tokamak_scope.md` §"The order this implies" step 4, and it replaces an estimate with a
measurement.

`tokamak_scope.md` §"Not built, and why" declined to build the device class, on the
ground that *"an empty device class cannot assemble — `physics.confinement_time`'s
registry is keyed on `istell` and has no tokamak entry"*. That registry is keyed on
`i_confinement_time` now, so the objection is spent, and the scaffold it said was worth
"exactly one line of work once the first slot has an occupant" is built:
`models/tokamak/namespace.py` (`Tokamak`, twenty-five slots, all empty) and
`total_process.py`'s `TokamakProcess` (a **sibling** of `StellaratorProcess`, not a
variant of it).

## Regenerating

```bash
SP=$(mktemp -d); (cd ~/jaxgraph && git archive HEAD src/cottax) | tar -x -C $SP
PYTHONPATH=$SP/src $PY -c '
import jax; jax.config.update("jax_enable_x64", True)
from functional_process.indat import machine_from_indat, graph_for
from functional_process import boundary as B
# NB: the file itself is refused at .physics.confinement_time.power_loss -- see below.
# Append one line, `i_plasma_ignited = 1`, to a copy of it and point this at that.
m = machine_from_indat("<large_tokamak_eval.IN.DAT + i_plasma_ignited = 1>")
g = graph_for(m); print(len(g.definitions), "nodes", len(B.boundary(g)), "boundary")
for kind, v in B.boundary(g):
    print(kind, v.path_str(), [n.path_str() for n in B.readers_of(g, v)])'
```

Everything below was taken against `~/jaxgraph` at **`b7c5572`** ("driver is now
declarable and first-class graph citizen"), pinned with `git archive` rather than read off
the live tree, because `mda.py`/`sand.py`/`boundary.py` were mid-rewrite for that commit's
API while this was measured. **No driver layer is involved**: every number here is off the
*declared* graph (`indat.graph_for(machine)`), so `guess` is `0` on both machines and
`input` is the whole boundary.

## The headline

| | nodes | boundary (all `input`) | cycles |
|---|---|---|---|
| tokamak (`large_tokamak_eval`) | **100** | **314** | 11 |
| stellarator (`stellarator_helias`, the reference) | 159 | 316 | 13 |
| shared boundary reads | | **239** | |
| on the tokamak only | | **75** | |
| on the stellarator only | | 77 | |

**75 is the number this file exists to produce**, and the answer to "how much does a
second device cost, in variables" is *not* 314. 239 of the tokamak's 314 boundary reads
are the same reads the stellarator already has and the same work items already tracked.
The tokamak's own debt is 75 — and, as §"The twelve that are simply inputs" and §"The
one that is already declared" below show, **58 of those 75 are the debt**: 12 are
variables PROCESS itself computes nowhere on a tokamak, 1 is a slot this tree already
declares empty on purpose, and 4 belong to a *shared* subsystem rather than to the
device. Read as a work list, the file has four sections and only the first two are work:

| | n | what it is |
|---|---|---|
| an empty `Tokamak` slot would produce it | **58** | the work list, 11 slots |
| a shared subsystem's pedestal arm would produce it | **4** | one node, mirror of an existing one |
| PROCESS computes it nowhere on a tokamak | 12 | permanent boundary, not debt |
| the tree already declares the slot empty | 1 | `inuclear = 1`, working as designed |

Node counts per subsystem, for the record:

| | costs | physics | power | availability | vacuum | buildings | device |
|---|---|---|---|---|---|---|---|
| tokamak | 40 | 33 | 18 | 4 | 3 | 2 | 0 |
| stellarator | 40 | 37 | 19 | 4 | 3 | 2 | 54 |

The three differences are all explained and none is a surprise: `physics` 37 → 33 is the
pedestal profile arm (3 nodes) replacing the parabolic one (7); `power` 19 → 18 is
`cryo_q_nuc`, whose slot is empty because `large_tokamak_eval.IN.DAT:170` sets
`inuclear = 1` and PROCESS's own comment is *"if inuclear = 1: qnuc is input"*; `device`
54 → 0 is the whole point of the exercise.

**The cyclic structure survives the device swap.** The tokamak has 11 cycles to the
stellarator's 13: ten of each are two-node `^problem` self-loops on the same driven nodes,
and both machines carry the *same* six-node physics SCC (`fusion_power_totals_mw` →
`fusion_totals_no_beam` → the on-axis-density node → `density_profile` → `fusion_rates` →
`plasma_composition`), with only the profile-arm occupant differing. The stellarator's two
extra are `power.cryo_q_nuc`'s self-loop and the four-node
`winding_pack_intersect`/`intersect` coil SCC, both device-specific. So on the evidence so
far a second device changes *which* nodes exist and not the shape of the coupling.

## What blocked the real file, and why it was not worked around

**`large_tokamak_eval.IN.DAT` does not assemble.** It is refused, once, at
`.physics.confinement_time.power_loss`:

```
i_plasma_ignited_i_rad_loss == -1 is a real PROCESS branch but is not ported: the head
of `calculate_confinement_time` is written for an ignited plasma losing core radiation
only ... The other five combinations are real PROCESS branches reading genuinely
different variables (injected heating when not ignited; ...) and none is written yet.
```

The file never sets `i_plasma_ignited`, so it takes PROCESS's own default `0`
(`physics_variables.py:881`, NON_IGNITED). The stellarator reference run sets `1`
(`stellarator_helias.IN.DAT:126`), so the only written occupant,
`PlasmaPowerLossIgnitedCoreRadiation`, does not fit a conventional tokamak's own
reference input. **That is a genuine discovery of this pass** — `tokamak_scope.md`'s 17
new decisions do not include `i_plasma_ignited`, because it is not a *new* switch: it is
one this port already reads, pinned to the one value both stellarator runs happen to use.

It is one occupant, and a small one. The NON_IGNITED core-only head is the same formula
with one extra term, so the arm needs exactly one new class reading exactly one extra
variable, `.current_drive.p_hcd_injected_total_mw` — a variable that is **already on the
tokamak boundary** (attributed to `.tokamak.current_drive` below). So writing the arm adds
a reader to a boundary entry that exists, and adds nothing new to this table.

Since a machine was still needed to measure, the numbers above are taken on
`large_tokamak_eval.IN.DAT` **plus the single line `i_plasma_ignited = 1`**, and nothing
else. Stated rather than smoothed over, because it is a divergence from the run being
modelled and the reader has to be able to price it. The exact price is one row: the true
`large_tokamak_eval` boundary is these 314 entries **plus a second reader on
`.current_drive.p_hcd_injected_total_mw`** and no new variable at all, since nothing in
this port produces it either way. The refusal was left in place, not weakened: an
unwritten arm assembled from the written one's reads is the invented-edge defect the
confinement split exists to remove, and `BASELINE_INDAT` in `test_machine.py` has recorded
that `i_plasma_ignited = 0` is refused since before this pass.

## The 58 that are the work list

Grouped by the empty slot that would produce them. **Attribution is mechanical, not
curated**: an `ast` walk over the 38 files of `tokamak_call_surface.md` §B collects every
`Assign`/`AugAssign` target of the form `[self.]data.<area>.<name>` — tuple-unpacked
write-backs included, which is how `structure.py`, `fw.py` and `tfcoil/base.py` write most
of their outputs and which a regex over the same files silently missed for ten of these
rows. Readers are `boundary.readers_of`.

| slot | reads | what fills it |
|---|---|---|
| `.tokamak.ccfe_hcpb` | 16 | `blankets/hcpb.py::CCFE_HCPB` (+ `blanket_library.py` by inheritance) |
| `.tokamak.cicc_superconducting_tf_coil` | 10 | `tfcoil/superconducting.py::CICCSuperconductingTFCoil` (+ `tfcoil/base.py`) |
| `.tokamak.physics` | 8 | `physics/physics.py::Physics`, the tokamak arm |
| `.tokamak.build` | 6 | `build.py::Build` |
| `.tokamak.plasma_geom` | 5 | `physics/plasma_geometry.py::PlasmaGeom` |
| `.tokamak.current_drive` | 3 | `physics/current_drive.py::CurrentDrive` |
| `.tokamak.first_wall` | 3 | `fw.py::FirstWall` |
| `.tokamak.structure` | 3 | `structure.py::Structure` |
| `.tokamak.divertor` | 2 | `divertor.py::Divertor` |
| `.tokamak.cryostat` | 1 | `cryostat.py::Cryostat` |
| `.tokamak.vacuum_vessel` | 1 | `vacuum.py::VacuumVessel` |

**58 reads across 11 of the 25 slots.** The other fourteen (`plasma_inductance`, `plasma_beta`, `plasma_current`, `bootstrap_current`, `l_h_transition`, `scrape_off_layer`, `density_limit`, `plasma_fields`, `pf_coil`, `cs_coil`, `cs_fatigue`, `pulse`, `shield`, `water_use`) are attributed **zero** boundary reads, and that is a real result rather than a gap in the method: nothing currently in this graph reads what they produce. `pfcoil.py` is the largest wholly-unported model in the scope (3525 entered lines, 24 functions) and it contributes nothing here, because `model_tree_design.md` §8 step 4c *deleted* its consumers -- Accounts 222.2 and 225.2 -- when the stellarator had no PF coils to cost. **The boundary does not see a producer nobody asks for.** So this table is a lower bound on the work and an exact count of the work *the graph as it stands* is blocked on; restoring the deleted cost nodes is what makes `pf_coil` appear in it, and `cost_boundary_inputs.md` category (d) already carries the producer `file:line` for each.


### `.tokamak.ccfe_hcpb` — 16

| variable | read by |
|---|---|
| `.divertor.a_div_surface_total` | `.costs.divertor_cost` |
| `.fwbs.m_blkt_beryllium` | `.costs.blanket_cost` |
| `.fwbs.m_blkt_li2o` | `.costs.blanket_cost` |
| `.fwbs.m_blkt_steel_total` | `.costs.blanket_cost` |
| `.fwbs.p_blkt_nuclear_heat_total_mw` | `.costs.reactor_cooling_system_cost`, `.power.component_thermal_powers`, `.power.delta_eta_step` |
| `.fwbs.p_fw_hcd_rad_total_mw` | `.power.component_thermal_powers` |
| `.fwbs.p_fw_nuclear_heat_total_mw` | `.power.component_thermal_powers`, `.power.delta_eta_step`, `.power.p_fw_div_heat_deposited_mw_step` |
| `.fwbs.p_fw_rad_total_mw` | `.power.component_thermal_powers`, `.power.delta_eta_step`, `.power.p_fw_div_heat_deposited_mw_step` |
| `.fwbs.p_shld_nuclear_heat_mw` | `.costs.reactor_cooling_system_cost`, `.power.component_thermal_powers`, `.power.delta_eta_step` |
| `.fwbs.p_tf_nuclear_heat_mw` | `.availability.electric_production` |
| `.fwbs.whtshld` | `.costs.shield_cost`, `.buildings.sizing` |
| `.fwbs.wpenshld` | `.costs.shield_cost` |
| `.heat_transport.p_blkt_coolant_pump_mw` | `.power.component_thermal_powers`, `.power.delta_eta_step`, `.power.p_fw_blkt_coolant_pump_mw_step` |
| `.heat_transport.p_div_coolant_pump_mw` | `.power.component_thermal_powers`, `.power.delta_eta_step`, `.power.p_fw_div_heat_deposited_mw_step` |
| `.heat_transport.p_fw_coolant_pump_mw` | `.power.component_thermal_powers`, `.power.delta_eta_step`, `.power.p_fw_div_heat_deposited_mw_step`, `.power.p_fw_blkt_coolant_pump_mw_step` |
| `.heat_transport.p_shld_coolant_pump_mw` | `.power.component_thermal_powers`, `.power.delta_eta_step` |

The single biggest attributable gap, and it is the first wall / blanket / shield chain,
not the plasma. `blankets/hcpb.py::CCFE_HCPB` is 7 entered functions and 956 entered
lines with `blankets/blanket_library.py` (14 functions, 822 lines) reached through it by
inheritance. Three of the seven are ported (unit #13) and none is registered: on a
stellarator they are reachable only through `blanket_neutronics()`, whose live PROCESS
call-site bug blocks that arm. A tokamak reaches them directly, so **the registration
that unit #13 has been waiting for has a graph to go into now.**

Every reader is in `costs` or `power` -- the two subsystems `tokamak_call_surface.md` §C
measured as fully shared. That is the shape of the whole table: the ported balance of
plant is *already wired to consume* what an unported device model would produce, which is
why these entries are boundary reads rather than absent edges.


### `.tokamak.cicc_superconducting_tf_coil` — 10

| variable | read by |
|---|---|
| `.tfcoil.c_tf_turn` | `.costs.tf_coil_power_conditioning_cost`, `.power.tf_power`, `.power.cryo_q_loads_step` |
| `.tfcoil.e_tf_magnetic_stored_total_gj` | `.costs.tf_coil_power_conditioning_cost`, `.power.tf_power` |
| `.tfcoil.len_tf_coil` | `.costs.tf_magnet_cost_superconducting` |
| `.tfcoil.m_tf_coil_case` | `.costs.tf_magnet_cost_superconducting` |
| `.tfcoil.m_tf_coil_copper` | `.costs.tf_magnet_cost_superconducting` |
| `.tfcoil.m_tf_coil_superconductor` | `.costs.tf_magnet_cost_superconducting` |
| `.tfcoil.m_tf_coils_total` | `.buildings.tf_coil_envelope` |
| `.tfcoil.n_tf_coil_turns` | `.costs.tf_magnet_cost_superconducting` |
| `.tfcoil.tfcryoarea` | `.power.cryo_q_loads_step` |
| `.tfcoil.v_tf_coil_dump_quench_kv` | `.costs.tf_coil_power_conditioning_cost`, `.power.tf_power` |

Ten reads, and the slot behind them is 19 entered functions / 2457 entered lines in
`tfcoil/superconducting.py` plus 8 / 753 in `tfcoil/base.py`, reached by inheritance. Two
of the ten (`.tfcoil.len_tf_coil`, `.tfcoil.tfcryoarea`) are the *same variables* the
stellarator's `coils/calculate.py` produces -- `LenTfCoil` and `TfCryoArea` were both
landed specifically to take them off the stellarator's boundary
(`boundary_inputs_audit.md` §7 items 4 and 7). So this is not a new variable to
understand, it is the same variable computed by a different device's coil model, which
is exactly the shape a slot is for.

**This slot carries the only CoolProp obstacle in the tokamak scope.** `quench.py` (436
lines, 450 CoolProp calls per `_call_models_once`) hangs off
`quench_heat_protection_current_density`. `.tfcoil.v_tf_coil_dump_quench_kv` above is on
that chain. So this slot is *scheduled* differently from the rest of the table: it waits
on `next_steps.md` §5's unresolved wrapping policy, not merely on someone writing it.


### `.tokamak.physics` — 8

| variable | read by |
|---|---|
| `.physics.b_plasma_surface_poloidal_average` | `.physics.fast_alpha_beta` |
| `.physics.e_plasma_beta` | `.physics.confinement_time.tail`, `.physics.dimensionless_plasma_parameters` |
| `.physics.p_plasma_inner_rad_mw` | `.physics.confinement_time.tail` |
| `.physics.p_plasma_rad_mw` | `.physics.radiation_fraction` |
| `.physics.p_plasma_separatrix_mw` | `.power.component_thermal_powers`, `.power.delta_eta_step`, `.power.p_fw_div_heat_deposited_mw_step` |
| `.physics.pden_plasma_core_rad_mw` | `.physics.confinement_time.power_loss` |
| `.times.t_plant_pulse_plasma_present` | `.power.cryo_q_loads_step` |
| `.times.t_plant_pulse_total` | `.costs.cost_of_electricity`, `.availability.avail` |

The tokamak arm of `physics.py`, and note what is in the list: two `.times.*` fields.
`.times.t_plant_pulse_total` and `.times.t_plant_pulse_plasma_present` are written inside
`physics.py`, which is where `pulsetimings` -- decision 15 -- has its **only read in all
of `process/models/**`** (`physics.py:476`). So the pulse-timing decision and two of this
slot's eight reads are the same piece of work.

`.physics.e_plasma_beta` is the one row here with a stellarator counterpart already
ported: `StellaratorBetaAndStoredEnergy` owns it there, and `PlasmaBeta`
(`.tokamak.plasma_beta`, empty) is what owns it on a tokamak -- decision 7's site.


### `.tokamak.build` — 6

| variable | read by |
|---|---|
| `.build.dr_tf_inboard` | `.buildings.tf_coil_envelope`, `.vacuum.vacuum_old` |
| `.build.dr_tf_outboard` | `.buildings.tf_coil_envelope` |
| `.build.r_shld_inboard_inner` | `.buildings.sizing`, `.vacuum.vacuum_old` |
| `.build.r_shld_outboard_outer` | `.buildings.sizing` |
| `.build.r_tf_outboard_mid` | `.buildings.tf_coil_envelope` |
| `.build.z_tf_inside_half` | `.buildings.sizing`, `.buildings.tf_coil_envelope` |

Six reads, all consumed by `buildings` and `vacuum`, and all of them radial-build
geometry. `build.py` is 6 entered functions but **2306 of its 2360 lines** -- the widest
entered-LOC-to-function ratio in the whole scope -- so this is six variables sitting on
top of one very large, wholly unported model. `.build.z_tf_inside_half` is the one the
stellarator already produces (`ZTfInsideHalf`, carved out of `st_coil` for exactly this
reason), which is again the same field from a different device's model.


### `.tokamak.plasma_geom` — 5

| variable | read by |
|---|---|
| `.physics.a_plasma_poloidal` | `.physics.profiles.profile_factors` |
| `.physics.a_plasma_surface` | `.vacuum.vacuum_old` |
| `.physics.eps` | `.physics.dimensionless_plasma_parameters` |
| `.physics.rminor` | `.physics.confinement_time.elongation`, `.physics.synchrotron_radiation_power`, `.vacuum.vacuum_old` |
| `.physics.vol_plasma` | `.physics.fusion_power_totals_mw`, `.physics.confinement_time.elongation`, `.physics.confinement_time.power_loss`, `.physics.confinement_time.tail`, `.physics.set_fusion_powers`, `.physics.synchrotron_radiation_power`, `.physics.auxiliary_physics_quantities`, `.physics.electron_thermal_energy`, `.physics.ion_thermal_energy`, `.physics.dimensionless_plasma_parameters`, `.vacuum.vacuum_old` |

Five reads and one of them, `.physics.vol_plasma`, has **eleven readers** -- the most
of any entry in this table. Plasma volume is what every extensive physics quantity is
formed from, and on a stellarator `StellaratorPlasmaGeometry` produces it. This is the
single highest-leverage row in the file: one node in `physics/plasma_geometry.py`
(7 entered functions, 549 lines, 1 of them already ported) closes reads for
`fusion_power_totals_mw`, three confinement slots, `set_fusion_powers`,
`synchrotron_radiation_power`, `auxiliary_physics_quantities`, both thermal-energy nodes,
`dimensionless_plasma_parameters` and `vacuum_old`.


### `.tokamak.current_drive` — 3

| variable | read by |
|---|---|
| `.current_drive.p_hcd_ecrh_injected_total_mw` | `.costs.power_injection_cost` |
| `.current_drive.p_hcd_injected_total_mw` | `.physics.total_plasma_heating_power`, `.power.component_thermal_powers` |
| `.heat_transport.p_hcd_electric_total_mw` | `.costs.heat_rejection_cost`, `.power.component_thermal_powers`, `.power.acpow`, `.availability.electric_production` |

Three reads, and `.current_drive.p_hcd_injected_total_mw` is the one the unwritten
NON_IGNITED confinement head would also read -- see §"What blocked the real file". Site
of decision 10 (`i_hcd_primary = 10`, 14 reads in `current_drive.py` and further reads
inside `buildings.py`, `costs/costs.py` and `build.py`, all three already ported).


### `.tokamak.first_wall` — 3

| variable | read by |
|---|---|
| `.first_wall.a_fw_total` | `.costs.first_wall_cost` |
| `.physics.p_fw_alpha_mw` | `.power.component_thermal_powers`, `.power.delta_eta_step`, `.power.p_fw_div_heat_deposited_mw_step` |
| `.physics.pflux_fw_neutron_mw` | `.availability.avail`, `.availability.cplife_avail` |

`fw.py`, 6 entered functions / 299 entered lines. It imports `FluidProperties` but
reaches CoolProp **zero** times on this run: every site is behind
`.fwbs.i_p_coolant_pumping == MECHANICAL` (2) and the file sets 3. Dormant, not absent --
a second tokamak IN.DAT wakes it (`tokamak_call_surface.md` §D).


### `.tokamak.structure` — 3

| variable | read by |
|---|---|
| `.structure.aintmass` | `.costs.tf_magnet_cost_superconducting` |
| `.structure.clgsmass` | `.costs.tf_magnet_cost_superconducting` |
| `.structure.coldmass` | `.power.cryo_q_loads_step` |

Three reads, and this is the slot `cost_boundary_inputs.md` predicted. `costs` dropped
Account 221.4 (reactor structure) because `st_strc` sets `.structure.fncmass`/`.gsmass` to
a literal `0.0` on a stellarator, so `ReactorStructureCost` computed an exact zero and
landed on the right number by luck. `structure.py::Structure` is 2 entered functions / 200
lines and writes all five fields by tuple unpacking -- the pattern a regex attribution
missed and the `ast` one catches.


### `.tokamak.divertor` — 2

| variable | read by |
|---|---|
| `.divertor.pflux_div_heat_load_mw` | `.availability.avail` |
| `.fwbs.p_div_nuclear_heat_total_mw` | `.power.component_thermal_powers`, `.power.delta_eta_step`, `.power.p_fw_div_heat_deposited_mw_step` |

`models/divertor.py::Divertor` -- **not** `models/stellarator/divertor.py`, which is
ported (unit #4) and is half of the stellarator graph's one non-problem cycle. Two
different models of two different devices' divertors, and the slot is what keeps them
apart. `.divertor.pflux_div_heat_load_mw` has a second writer in `availability.py`, a
fallback assignment, so the `ast` attribution reports both; the producer is `divertor.py`.


### `.tokamak.cryostat` — 1

| variable | read by |
|---|---|
| `.fwbs.r_cryostat_inboard` | `.buildings.sizing` |

One read. `models/cryostat.py` is 2 entered functions / 69 lines and is **not** the
stellarator's, which is `stellarator.py:1282-1330` (unit #1 chunk S5, ported, and already
a slot of `.stellarator.fwbs`). The cheapest slot in the table.


### `.tokamak.vacuum_vessel` — 1

| variable | read by |
|---|---|
| `.fwbs.m_vv` | `.costs.vacuum_vessel_assembly_cost` |

One read, and a **confirmed registry prediction**: unit #16 recorded `VacuumVessel` as
*"confirmed unreachable on the stellarator pipeline, no action needed"*. The tokamak
reaches it at `caller.py:331`. Its file-mate `Vacuum` is ported and shared.

## The four that are a *shared* subsystem's gap, not the tokamak's

These are not attributable to any `Tokamak` slot. They are a hole in
`.physics.profiles.parameterisation`'s **pedestal** occupant, which the tokamak is the
first machine to select: `large_tokamak_eval.IN.DAT:291` sets `i_plasma_pedestal = 1`,
where `st_init` forces every stellarator run to `0`.


| variable | read by |
|---|---|
| `.physics.f_temp_plasma_electron_density_vol_avg` | `.physics.plasma_composition` |
| `.physics.nd_plasma_electron_line` | `.physics.confinement_time.inputs`, `.physics.dimensionless_plasma_parameters` |
| `.physics.temp_plasma_electron_density_weighted_kev` | `.physics.profiles.profile_factors`, `.physics.electron_thermal_energy`, `.physics.fast_alpha_beta` |
| `.physics.temp_plasma_ion_density_weighted_kev` | `.physics.profiles.profile_factors`, `.physics.ion_thermal_energy`, `.physics.fast_alpha_beta` |

`ProfileParameterisationParabolic` has seven nodes and `ProfileParameterisationPedestal`
three, and the four missing ones are exactly `ParabolicProfileValues`' counterpart:
`plasma_profiles.py:217/218/226/234` are inside `pedestal_parameterisation` and compute
these four, while `:126-158` are inside `parabolic_parameterisation` and are what
`ParabolicProfileValues` ports. So the pedestal arm needs one node, the mirror of one
that already exists and is validated.

**This is the first thing found by assembling a second device that a switch survey could
not have found**, and it is worth naming as a method result: `tokamak_scope.md` classed
`i_plasma_pedestal` as *"the factory dispatches on it — already a slot"*, which was true
and was not the whole story. A slot with a registered occupant on both arms can still
have one arm that produces less than the other, and only the boundary sees it. That is
`model_tree_design.md` §6's ragged-arm hazard, caught by the check written for it.

## The twelve that are simply inputs

**PROCESS computes these nowhere on the tokamak path.** They will never gain a producer,
so they are not work items and must not be counted as debt — the boundary is *supposed*
to hold them. Seven are the pedestal arm's own inputs and five have no writer anywhere in
the traced surface.

### Seven pedestal-arm inputs


| variable | read by |
|---|---|
| `.physics.nd_plasma_pedestal_electron` | `.physics.profiles.parameterisation.pedestal_on_axis_densities`, `.physics.profiles.density_profile` |
| `.physics.nd_plasma_separatrix_electron` | `.physics.profiles.parameterisation.pedestal_on_axis_densities`, `.physics.profiles.density_profile` |
| `.physics.radius_plasma_pedestal_density_norm` | `.physics.profiles.parameterisation.pedestal_on_axis_densities`, `.physics.profiles.density_profile` |
| `.physics.radius_plasma_pedestal_temp_norm` | `.physics.profiles.parameterisation.pedestal_temperature_profile`, `.physics.profiles.parameterisation.pedestal_on_axis_temperatures` |
| `.physics.tbeta` | `.physics.profiles.parameterisation.pedestal_temperature_profile`, `.physics.profiles.parameterisation.pedestal_on_axis_temperatures`, `.physics.synchrotron_radiation_power` |
| `.physics.temp_plasma_pedestal_kev` | `.physics.profiles.parameterisation.pedestal_temperature_profile`, `.physics.profiles.parameterisation.pedestal_on_axis_temperatures` |
| `.physics.temp_plasma_separatrix_kev` | `.physics.profiles.parameterisation.pedestal_temperature_profile`, `.physics.profiles.parameterisation.pedestal_on_axis_temperatures` |

All seven are written **only** by `plasma_profiles.py:111-117`, the L-mode reset inside
`parabolic_parameterisation` — which is why `LModeProfileReset` is a node of the parabolic
arm and correctly has no pedestal counterpart. On the pedestal arm PROCESS reads them from
the input file, and `large_tokamak_eval.IN.DAT:292-299` sets all seven explicitly, each
with the comment `(i_plasma_pedestal==1)`. So this is not seven missing producers, it is
seven inputs the stellarator happens to overwrite with zeros and a tokamak does not.

One caveat carried rather than dropped: `.physics.temp_plasma_separatrix_kev` has a second
writer, `physics.py:1109`'s `reinke_tsep`, behind the Reinke criterion. Not on this run,
and the input file's own comment says so (*"calculated if reinke"*). A run with that
constraint active moves this row into the work list.

### Five with no producer anywhere on the traced surface


| variable | read by |
|---|---|
| `.fwbs.life_fw_fpy` | `.availability.avail` |
| `.fwbs.m_blkt_vanadium` | `.costs.blanket_cost` |
| `.fwbs.p_fw_hcd_nuclear_heat_mw` | `.power.component_thermal_powers` |
| `.physics.aspect` | `.physics.confinement_time.scaling`, `.physics.synchrotron_radiation_power`, `.physics.auxiliary_physics_quantities` |
| `.tfcoil.n_tf_coils` | `.costs.tf_magnet_cost_superconducting`, `.costs.tf_coil_power_conditioning_cost`, `.power.tf_power`, `.power.cryo_q_loads_step`, `.buildings.sizing`, `.buildings.tf_coil_envelope`, `.vacuum.vacuum_old` |

Each was checked individually against the whole of `process/`, not just the 38 files:

* `.physics.aspect` — **iteration variable 1** (`iteration_variables.py:45`), set at
  `large_tokamak_eval.IN.DAT:276`. The only model that ever writes it is
  `stellarator.py:220`, and only when `1 not in ixc`. On a tokamak it is a design
  variable, which is what a boundary input is for.
* `.tfcoil.n_tf_coils` — set at `:377`. Written only by `stellarator.py:228`. Seven
  readers, the most of any input row, and all of them in ported subsystems: this is
  decision 17, and §E's point that six of the seventeen decisions have their reads inside
  files the port has already reproduced is visible here as a boundary entry with no work
  attached.
* `.fwbs.life_fw_fpy` — written only by `stellarator.py:515` and `ife.py`. PROCESS itself
  knows: `availability.py:159` carries the comment *"For some reason life_fw_fpy is not
  always calculated"* and guards `< 0.0001`. So the tokamak path genuinely leaves it at
  its `fwbs_variables.py:55` default of `0.0`.
* `.fwbs.m_blkt_vanadium` — written only by `stellarator.py:1083` and `ife.py`. A
  CCFE-HCPB blanket has no vanadium; `BlanketCost` multiplies the default `0.0` by
  `ucblvd`.
* `.fwbs.p_fw_hcd_nuclear_heat_mw` — written only by `blankets/dcll.py:223` (which needs
  `i_blanket_type = 5`) and by `stellarator.py`. Dormant on this run's CCFE-HCPB blanket.

Three of these five — `life_fw_fpy`, `m_blkt_vanadium`, `p_fw_hcd_nuclear_heat_mw` — are
**produced on the stellarator and are inputs on the tokamak**. That is the sharpest single
correction this file makes to the intuition that a second device only adds work: part of
the boundary growth is permanent and is PROCESS's own doing, not this port's.

## The one that is already declared


| variable | read by |
|---|---|
| `.fwbs.qnuc` | `.power.cryo_q_loads_step`, `.power.cryo_loads` |

`.power.cryo_q_nuc` is `None` on this machine, because `large_tokamak_eval.IN.DAT:170`
sets `inuclear = 1` and PROCESS's own comment at `power.py:1825` is *"Issue #511: if
inuclear = 1: qnuc is input"*. So this row is the tree working exactly as designed:
`CRYO_Q_NUC = {0: CryoQNuc, 1: None}`, and the empty slot **states** what
`sand.degenerate_fixed_points` used to have to recover at runtime by differentiating a
residual. It is on the boundary because it is an input, not because anything is missing,
and it is listed apart from the twelve above only because the tree says so structurally
rather than by absence of a writer.

## The 77 the tokamak does *not* have

Symmetric and worth one line: 77 of the stellarator's 316 boundary reads are absent from
the tokamak's, and every one belongs to a node the tokamak does not have — 15
`.stellarator.*` machine-config and profile parameters, 25 `.tfcoil.*` superconductor and
winding-pack inputs read by `coils/`, 18 `.fwbs.*` blanket-composition fractions, 8
`.build.*` thicknesses, 6 `.divertor.*`, and the rest. Nothing is lost by the swap that a
tokamak still wants. The two boundaries are 239 shared, 75 tokamak-only, 77
stellarator-only — which is the honest way to say that the two devices' *inputs* overlap
about as much as their models do.

## What this file does not settle

* **One input file, one point.** Same caveat `tokamak_call_surface.md` §F carries:
  `.fwbs.i_p_coolant_pumping = 2` or `.fwbs.i_blkt_coolant_type = 2` would wake three
  dormant CoolProp modules, `i_tf_sup = 0` would swap the whole TF slot, and
  `i_blanket_type = 5` would replace `.tokamak.ccfe_hcpb` — the largest attributable group
  here — with `blankets/dcll.py`. A second tokamak IN.DAT is a different table.
* **The boundary is of the *declared* graph.** No `Drive` and no `Start` port, so `guess`
  is 0 by construction. Once the tokamak's problems are declared the split will matter and
  the `input` half is the half that must not grow.
* **The constraint and objective reads are not in it.** `tokamak_call_surface.md` §F
  records that `objective_function` and `constraint_eqns` sit outside
  `_call_models_once` entirely; they are ported, but they are not in this graph, so their
  reads are not counted here. A tokamak's active `icc`/`ixc` set will add to both sides of
  this ledger.
* **Nothing here says a tokamak runs.** It says exactly which named variables stand
  between this graph and one, and which of them never will.
