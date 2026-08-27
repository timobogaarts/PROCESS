---
kind: model-unit
status: draft
confidence: high
---

**Ported: the whole tokamak call path (7/7 entered functions).** This record was written
for the *stellarator* scope, where only three of `CCFE_HCPB`'s methods were reachable and
only through `blanket_neutronics()`. It is now extended for the tokamak, which reaches
`CCFE_HCPB.run()` directly (`process/core/caller.py:345`) and for which
`_audit/tokamak_boundary.md` §`.tokamak.ccfe_hcpb` is the single biggest attributable
boundary gap at **16 reads**. Everything below the "tokamak extension" heading is new;
the original three-function audit is preserved beneath it because its evidence is still
the evidence, and because open question #1 (a live PROCESS `TypeError`) is unresolved and
must not be lost.

**Headline results of the extension:**

- **14 of the slot's 16 boundary variables get a producer.** The other two --
  `.heat_transport.p_fw_coolant_pump_mw` and `.heat_transport.p_blkt_coolant_pump_mw` --
  have **no producer anywhere in PROCESS at this run's `i_p_coolant_pumping`**, measured.
  See "two boundary variables with no producer" below; this contradicts
  `tokamak_boundary.md` and the contradiction is a finding, not an oversight.
- **The two apparent cycles inside this model both dissolve.** No `FixedPointFunction` is
  needed: assembled with `blanket_library.py`'s four nodes, `Blocking.scc` returns **15
  blocks of size 1**, in exactly PROCESS's own call order.
- **`itart` is no longer a traced `jnp.where` argument.** The original record recommended
  keeping it as one; `next_steps.md` §14.2's binding policy (later than that
  recommendation) withdraws that option, and acting on it removed three dead reads from
  the conventional arm and one from the spherical.

---

# Tokamak extension (this pass)

## source, re-measured

`process/models/blankets/hcpb.py`, 1663 lines, **956 entered** across **7 entered
functions**. Re-measured with the same `sys.setprofile` hook `tokamak_call_surface.md`
§"Regenerating" specifies, over one `Caller._call_models_once` on
`tests/regression/input_files/large_tokamak_eval.IN.DAT`, and it reproduced §B's seven
exactly:

| entered function | lines | ported as |
|---|---|---|
| `CCFE_HCPB.run` | 42-283 | **two** nodes -- `CentrepostNeutronicsAbsent` (`:143-148`) and `NuclearHeatingRenormalisationSingleNullConventional` (`:195-276`). The rest of `run()` is orchestration and becomes the tree's slot order. |
| `CCFE_HCPB.component_masses` | 285-461 | `DivertorSurfaceAndPlateMassSingleNull` (`:353-367`) + `ComponentMasses` (everything else) |
| `CCFE_HCPB.nuclear_heating_magnets` | 463-609 | `FirstWallCoolantVoidFractions` (`:483-490`) + `NuclearHeatingMagnetsConventional` / `…SphericalTokamak` |
| `CCFE_HCPB.nuclear_heating_fw` | 611-651 | `NuclearHeatingFw` |
| `CCFE_HCPB.nuclear_heating_blanket` | 653-698 | `NuclearHeatingBlanket` |
| `CCFE_HCPB.nuclear_heating_shield` | 700-769 | `NuclearHeatingShieldConventional` / `…SphericalTokamak` |
| `CCFE_HCPB.powerflow_calc` | 771-1006 | `FirstWallRadiationPowers` (`:780-814`) + `PumpingPowerMechanicalWithPressureDrop` (`:864-918`) |

Not entered, therefore not ported: `st_cp_angle_fraction` (`:1008-1080`),
`st_tf_centrepost_fast_neut_flux` (`:1082-1134`), `st_centrepost_nuclear_heating`
(`:1136-1287`), `write_output` (`:1289-1662`), `output` (`:38-40`). The first three are
the `itart == 1` centrepost chain and are the reason `itart == 1` is UNPORTED for this
whole slot.

`blankets/blanket_library.py` is reached through this file by inheritance
(`hcpb.py:25`) and is ported separately -- see
`_audit/units/models/blankets/blanket_library.md`.

## the two cycles, and why neither needs a `FixedPointFunction`

`next_steps.md` §5 records that four apparent self-loops in this project dissolved on
inspection. These are the fifth and sixth, and both are *two-node* cycles in PROCESS's
own call order rather than self-loops, which is why they are worth spelling out.

**Cycle 1 -- `component_masses` <-> `nuclear_heating_magnets`.** `run()` calls
`component_masses()` at `:150` and `nuclear_heating_magnets()` at `:155`.

- `component_masses` **reads** `.fwbs.f_a_fw_coolant_inboard`/`_outboard` at `:317`,
  `:320`, `:332`, `:335`, `:383`, `:386`.
- `nuclear_heating_magnets` **writes** them at `:483` and `:490` -- five lines later in
  execution order, one pass too late.
- `nuclear_heating_magnets` in turn reads `.fwbs.m_blkt_total` (`:500`), `.fwbs.whtshld`
  (`:503`), which `component_masses` writes at `:450` and `:370`.

So PROCESS reads a one-pass-stale value and relies on `Caller.call_models`
(`caller.py:100-133`) re-running the entire pipeline up to ten times to repair it.

**It dissolves because the write depends on nothing the read's block produces.**
`f_a_fw_coolant_inboard = pi * radius_fw_channel**2 / (dx_fw_module * dr_fw_inboard)`, and
`.fwbs.radius_fw_channel` and `.fwbs.dx_fw_module` have **no producer anywhere under
`process/models/`** -- `grep -rn "\.radius_fw_channel\s*=[^=]" process/models/` and the
same for `dx_fw_module` return nothing; both are pure inputs. `.build.dr_fw_inboard` is
`build.py`'s. Lifting the three lines into `FirstWallCoolantVoidFractions` and ordering it
first makes the graph acyclic **and** gives PROCESS's converged answer in one pass.

**Cycle 2 -- `component_masses` with itself, on `.divertor.a_div_surface_total`.** Read at
`:299` (the divertor coolant volume), written at `:353`. Same shape, same resolution: the
write reads only `.divertor.fdiva`, `.physics.rmajor`, `.physics.rminor`, none of which
this block produces, so `DivertorSurfaceAndPlateMassSingleNull` is split out and ordered
first.

**This is the same defect `st_fwbs` chunk 1E2 found on the stellarator side and could not
dissolve** (`unit_registry.md` row 4, and the S3 sub-chunk of the `st_fwbs` synthesis):
there, `st_fwbs` runs *before* `st_div` every iteration and reads
`.divertor.a_div_surface_total` as the *previous* `run()` wrote it, bootstrapped by a
hardcoded `50.0` on the true first call -- a genuine two-node SCC with `Divertor`. The
tokamak's is not the same cycle: here the producer is `component_masses` itself, in the
same function, and the value is geometric. Worth recording that the same `VarPath`
produces a real SCC on one device and a dissolvable one on the other.

**Measured result.** With `blanket_library.py`'s four nodes and this file's eleven
assembled into one `ModelNamespace`, `cottax.blocking.Blocking.scc` gives 15 blocks, max
size 1, in the order
`FirstWallCoolantVoidFractions, DivertorSurfaceAndPlateMassSingleNull,
CentrepostNeutronicsAbsent, FirstWallRadiationPowers, BlanketHalfHeightSingleNull,
EllipticalBlanketAreas, EllipticalBlanketVolumes, BlanketCoverageFactorsSingleNull,
ComponentMasses, NuclearHeatingMagnetsConventional, NuclearHeatingFw,
NuclearHeatingBlanket, NuclearHeatingShieldConventional,
NuclearHeatingRenormalisationSingleNullConventional,
PumpingPowerMechanicalWithPressureDrop` -- which is PROCESS's own call order with the two
cycles unwound.

## two boundary variables with no producer -- `tokamak_boundary.md` contradicted

`_audit/tokamak_boundary.md` §`.tokamak.ccfe_hcpb` lists
`.heat_transport.p_fw_coolant_pump_mw` and `.heat_transport.p_blkt_coolant_pump_mw` among
the sixteen this slot owes. **On the reference run nothing writes either of them**, and
the port cannot produce what PROCESS does not compute. Evidence:

- Every write site of either field in `process/models/`:
  `stellarator.py:644`/`:652`/`:908`/`:918`, `blanket_library.py:2574`/`:2588`, and
  `hcpb.py:820-821` (inside `powerflow_calc`'s `i_p_coolant_pumping == 1` arm). The
  stellarator sites are out of scope; `blanket_library.py:2574`/`:2588` are inside
  `thermo_hydraulic_model`, which is reached only at `i_p_coolant_pumping == 2` and is
  **not entered** on this run (`tokamak_call_surface.md` §D); `hcpb.py:820` is arm 1.
  This run is arm **3**.
- Measured: after four `Caller._call_models_once` passes both fields are exactly `0.0`,
  their `heat_transport_variables.py:73`/`:85` dataclass defaults.
- PROCESS's own consumer agrees. `power.component_thermal_powers` forms
  `.primary_pumping.p_fw_blkt_coolant_pump_mw` from those two fields **only** when
  `i_p_coolant_pumping not in {MECHANICAL, MECHANICAL_WITH_PRESSURE_DROP}`
  (`power.py:821-829`); at arm 3 it reads the `primary_pumping` field that
  `powerflow_calc:895` wrote instead.

`tokamak_boundary.md` names its own method as "mechanical, not curated" -- an `ast` walk
over `Assign` targets across the 38 reached files. That walk cannot see which *arm* a
write is in, so it attributed arms 1 and 2's writes to a run taking arm 3. The two reads
are real, but they are the **power subsystem's** `i_p_coolant_pumping` union-of-arms
defect -- `next_steps.md` §14.9 item 2, "the last contradicted switch, 5 slots" -- not
this slot's missing work. The port's answer at arm 3 is
`PumpingPowerMechanicalWithPressureDrop`, which owns
`.primary_pumping.p_fw_blkt_coolant_pump_mw`; the arm-3 occupants of
`power.component_thermal_powers`/`delta_eta_step`/`p_fw_div_heat_deposited_mw_step`/
`p_fw_blkt_coolant_pump_mw_step` should read *that*, and should not declare the two
`heat_transport` fields at all.

## data footprint -- the new functions

Only the functions this pass added or changed are tabulated; the original three's tables
are preserved below unchanged except where noted.

### `component_masses` (`:285-461`) -> `DivertorSurfaceAndPlateMassSingleNull` + `ComponentMasses`

Not a `@staticmethod`; genuine `self.data` extraction.

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.divertor.a_div_surface_total` | read `:299`, **write** `:353`/`:361` | **cycle 2** | split: the write is `DivertorSurfaceAndPlateMassSingleNull`'s, the read is `ComponentMasses`' |
| `.divertor.f_vol_div_coolant` | read `:300`, `:365` | explicit-arg | read by both halves |
| `.divertor.dx_div_plate` | read `:301`, `:366` | explicit-arg | read by both halves |
| `.divertor.fdiva` | read `:354` | explicit-arg | divertor half only |
| `.divertor.den_div_structure` | read `:364` | explicit-arg | divertor half only |
| `.divertor.n_divertors` | read `:360` | **switch** | selects the occupant; not a port |
| `.physics.rmajor`, `.physics.rminor` | read `:357-358` | explicit-arg | divertor half only |
| `.divertor.m_div_plate` | write `:362` | own-write (returned) | |
| `.fwbs.vol_blkt_total` | read `:306`, `:419`, `:425`, `:444` | explicit-arg | `blanket_library.BlanketCoverageFactorsSingleNull` |
| `.fwbs.f_a_blkt_cooling_channels` | read `:306` | explicit-arg | |
| `.fwbs.vol_shld_total` | read `:310`, `:371` | explicit-arg | `shield.py:138` |
| `.fwbs.vfshld` | read `:310`, `:373` | explicit-arg | |
| `.first_wall.a_fw_inboard`, `_outboard` | read `:315`/`:318`, `:330`/`:333`, `:381`/`:384` | explicit-arg | `fw.py:89-91` |
| `.first_wall.a_fw_total` | read `:337` | explicit-arg | `fwclfr`'s denominator |
| `.build.dr_fw_inboard`, `_outboard` | read `:316`/`:319`, `:331`/`:334`, `:339`, `:382`/`:385` | explicit-arg | |
| `.fwbs.f_a_fw_coolant_inboard`, `_outboard` | read `:317`/`:320`, `:332`/`:335`, `:383`/`:386` | **cycle 1** | now `FirstWallCoolantVoidFractions`' outputs |
| `.fwbs.den_steel` | read `:372`, `:391`, `:446` | explicit-arg | |
| `.physics.a_plasma_surface` | read `:396` | explicit-arg | |
| `.fwbs.fw_armour_thickness` | read `:396` | explicit-arg | |
| `.fwbs.breeder_f` | read `:404`, `:411`; **write** `:404`, `:405` | **self-write, dissolved** | see the deviation below |
| `.fwbs.breeder_multiplier` | read `:411`, `:416` | explicit-arg | |
| `.fwbs.vfcblkt`, `.fwbs.vfpblkt` | read `:438-439` | explicit-arg | |
| `.fwbs.m_fw_blkt_div_coolant_total` | write `:325` | own-write (returned) | |
| `.fwbs.fwclfr` | write `:329` | own-write (returned) | PROCESS's own comment: only used by the retired `fispact.f90`/`safety.f90` |
| `.fwbs.whtshld` | write `:370` | own-write (returned) | **boundary target** (`.costs.shield_cost`, `.buildings.sizing`) |
| `.fwbs.wpenshld` | write `:377` | own-write (returned) | **boundary target**; a copy of `whtshld` |
| `.fwbs.vol_fw_total` | write `:380` | own-write (returned) | |
| `.fwbs.m_fw_total` | write `:390` | own-write (returned) | consumed by `NuclearHeatingFw` |
| `.fwbs.fw_armour_vol`, `.fwbs.fw_armour_mass` | write `:395`, `:400` | own-write (returned) | |
| `.fwbs.f_vol_blkt_li4sio4`, `f_vol_blkt_tibe12` | write `:410`, `:415` | own-write (returned) | |
| `.fwbs.m_blkt_tibe12`, `m_blkt_li4sio4` | write `:418`, `:424` | own-write (returned) | |
| `.fwbs.m_blkt_beryllium`, `m_blkt_li2o` | write `:430`, `:431` | own-write (returned) | **boundary targets**; PROCESS issue #327 aliases of the two above |
| `.fwbs.f_vol_blkt_steel` | write `:434` | own-write (returned) | |
| `.fwbs.m_blkt_steel_total` | write `:443` | own-write (returned) | **boundary target** |
| `.fwbs.m_blkt_total` | write `:450` | own-write (returned) | consumed by magnets and blanket heating |
| `.fwbs.armour_fw_bl_mass` | write `:457` | own-write (returned) | |

### `nuclear_heating_fw` (`:611-651`) -> `NuclearHeatingFw`

`@staticmethod`, already pure.

| VarPath | read/write | classification |
|---|---|---|
| `.fwbs.m_fw_total` | read (via caller `:158`) | explicit-arg |
| `.ccfe_hcpb.fw_armour_u_nuc_heating` | read (via caller `:159`) | explicit-arg -- an input; nothing under `process/models/` writes it |
| `.physics.p_fusion_total_mw` | read (via caller `:160`) | explicit-arg |
| `.ccfe_hcpb.p_fw_nuclear_heat_total_mw_unnormalised` | write (return) | own-write (**minted**; PROCESS stores it in `.fwbs.p_fw_nuclear_heat_total_mw`, which `:220` overwrites) |

`raise ProcessValueError` on a negative result (`:646-650`) is a domain guard; the port
returns `nan` and the contract declares `reference_domain_errors`.

### `run()`'s `itart` else-arm (`:143-148`) -> `CentrepostNeutronicsAbsent`

Four literal zeros, no reads. `.fwbs.pnuc_cp_tf` is read by `power.py:1095` and
`tfcoil/base.py:1264`; `.fwbs.neut_flux_cp` by `availability.py:1553`/`:1557`;
`.fwbs.p_cp_shield_nuclear_heat_mw` by `power.py:940` and by this file's own
`powerflow_calc:852`/`:908`. `.fwbs.pnuc_cp` has no reader outside `hcpb.py`'s own
reporting (`:1460`) and is owned anyway, because PROCESS writes it and a field PROCESS
writes with no owner is a boundary input in disguise.

### `run()`'s renormalisation (`:195-276`) -> `NuclearHeatingRenormalisationSingleNullConventional`

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.ccfe_hcpb.p_{fw,blkt,shld,tf}_…_unnormalised` | read | explicit-arg | the four minted names |
| `.divertor.n_divertors` | read `:215` | **switch** | selects the occupant |
| `.physics.itart` | read `:103` | **switch** | selects the occupant; on this arm `f_geom_cp` is the literal `0` of `:144` and `.fwbs.pnuc_cp_tf` the literal `0` of `:145`, so `:263`'s `+ pnuc_cp_tf` and `:268`'s `f_geom_cp * p_neutron_total_mw` are both provably inert and are **not declared as reads** |
| `.fwbs.f_ster_div_single` | read `:215` | explicit-arg | `divertor.py:42` |
| `.fwbs.f_p_blkt_multiplication` | read `:225`, `:236`, `:248`, `:260`, `:273` | explicit-arg | |
| `.physics.p_neutron_total_mw` | read `:227`, `:238`, `:250`, `:262`, `:275` | explicit-arg | |
| `.ccfe_hcpb.pnuc_tot_blk_sector` | write `:195`, read `:223`/`:234`/`:246`/`:258` | own-write + `local-intermediate` | written once, read back four times in the same straight-line block |
| `.fwbs.p_fw_nuclear_heat_total_mw` | write `:220` | own-write (returned) | **boundary target** |
| `.fwbs.p_blkt_nuclear_heat_total_mw` | write `:231` | own-write (returned) | **boundary target** |
| `.fwbs.p_shld_nuclear_heat_mw` | write `:243` | own-write (returned) | **boundary target** |
| `.fwbs.p_tf_nuclear_heat_mw` | write `:255` | own-write (returned) | **boundary target**; see the dual-producer note below |
| `.fwbs.p_cp_shield_nuclear_heat_mw` | write `:267` | `redundant-duplicate-write` | identical `0.0` to `:146`; owned by `CentrepostNeutronicsAbsent`, not here |
| `.fwbs.p_blkt_multiplication_mw` | write `:272` | own-write (returned) | read by `physics.py:2471`/`:2706` and `power.py:1184` |

**Dual ownership of `.fwbs.p_tf_nuclear_heat_mw` -- recorded, not a conflict here.**
`functional_process/models/stellarator/tf_nuclear_heating.py`'s `ScTfCoilNuclearHeating`
owns the same `VarPath` (`tf_nuclear_heating.py:184`). On the tokamak tree the ownership
is clean: that node is a slot of `Stellarator`'s joint `blktmodel`/`ipowerflow` switch
(`models/stellarator/namespace.py:186`) and a `TokamakProcess` has no `Stellarator`
namespace at all. The story is worth carrying because it has already produced one real
bug: `unit_registry.md`'s S2 row records `ScTfCoilNuclearHeating` sitting
*unconditionally* in `COMMON` while its formula is correct only for `ipowerflow == 0`,
the `EcrhDensityLimit` bug class. Two devices, two producers, one field -- any future
machine assembling both must choose, and the choice is a device fact, not a switch value.

### `powerflow_calc` (`:771-1006`) -> `FirstWallRadiationPowers` + `PumpingPowerMechanicalWithPressureDrop`

Prologue (`:780-814`), unconditional -- **not** behind `i_p_coolant_pumping`:

| VarPath | read/write | note |
|---|---|---|
| `.physics.p_plasma_rad_mw` | read `:781`, `:786` | |
| `.fwbs.f_a_fw_outboard_hcd` | read `:781` | `0.0` on this run, so `p_fw_hcd_rad_total_mw` is identically zero here; the harness fuzzes it away from zero deliberately |
| `.fwbs.p_div_rad_total_mw` | read `:787` | `divertor.py:52` |
| `.first_wall.a_fw_outboard`, `a_fw_total` | read `:807-808`, `:813` | |
| `.current_drive.p_beam_orbit_loss_mw` | read `:809` | |
| `.physics.p_fw_alpha_mw` | read `:810` | |
| `.fwbs.p_fw_hcd_rad_total_mw` | write `:780` | **boundary target** |
| `.fwbs.p_fw_rad_total_mw` | write `:785` | **boundary target** |
| `.fwbs.psurffwo`, `.fwbs.psurffwi` | write `:805`, `:812` | consumed by the pumping arm |

Arm 3 (`:864-918`): reads `.primary_pumping.p_he`/`dp_he`/`gamma_he`/`t_in_bb`/`t_out_bb`/
`f_p_fw_blkt_pump`, `.fwbs.etaiso`, the four renormalised powers, `.fwbs.psurffwi`/
`psurffwo`, `.fwbs.p_cp_shield_nuclear_heat_mw`, `.fwbs.p_div_nuclear_heat_total_mw`,
`.fwbs.p_div_rad_total_mw`, `.physics.p_plasma_separatrix_mw`,
`.heat_transport.f_p_shld_coolant_pump_total_heat`/`f_p_div_coolant_pump_total_heat`;
writes `.primary_pumping.p_fw_blkt_coolant_pump_mw` (`:895`),
`.heat_transport.p_shld_coolant_pump_mw` (`:904`, **boundary target**),
`.heat_transport.p_div_coolant_pump_mw` (`:911`, **boundary target**).

## switches touched

| switch | value on the reference run | source | occupant written | UNPORTED values, with reason |
|---|---|---|---|---|
| `.physics.itart` | 0 | default, `physics_variables.py:994` | conventional arms of magnets/shield/renormalisation, plus `CentrepostNeutronicsAbsent` | `== 1` needs `st_cp_angle_fraction`/`st_tf_centrepost_fast_neut_flux`/`st_centrepost_nuclear_heating` (`:1008-1287`, not entered, not ported) **and** `blanket_library`'s D-shaped geometry. The two ST *nuclear-heating* arms are written and tested (preserving unit #13's coverage) but the slot cannot be filled at `itart == 1` until that chain exists. |
| `.divertor.n_divertors` | 1 | IN.DAT-derived | single-null arms of the divertor mass and the renormalisation | `== 2` doubles `a_div_surface_total` (`:360-361`) and changes `f_geom_blanket` (`:215`). Not written. |
| `.fwbs.i_p_coolant_pumping` | 3 | IN.DAT:172 | `PumpingPowerMechanicalWithPressureDrop` | `0` (`USER_INPUT`, no arm at all in `powerflow_calc`), `1` (`FRACTION_OF_HEAT`, `:817-838`, a *different owned set* -- see below), `2` (`MECHANICAL`, `:840-862`, reaches `primary_coolant_properties`/`thermo_hydraulic_model` and hence **CoolProp**, `next_steps.md` §5's unresolved wrapping policy). |
| `.fwbs.i_blkt_coolant_type` | 1 (`HELIUM`) | **assigned by `run()` itself at `:45`** | n/a -- one arm reachable | `WATER` (2) selects `powerflow_calc:793-801`, the only CoolProp site inside `CCFE_HCPB`. It is **dead code on this model's path**, not dormant: `run()` sets `HELIUM` unconditionally at `:45` and nothing between there and `:793` changes it. Stronger than `tokamak_call_surface.md` §D's "dormant behind this run's switch values" -- no input file revives this one through `CCFE_HCPB.run()`. |
| `.fwbs.i_blanket_type` | 1 (`CCFE_HCPB`) | default, `fwbs_variables.py:70` | this whole file | `== 5` routes to `blankets/dcll.py` (`caller.py:349`), a different slot occupant entirely. |

**`i_p_coolant_pumping`'s arms do not own the same set**, which is the shape
`next_steps.md` §12.2 ("alternatives are keyed on output -- nearly") and §14.7's swap
contract already flag. Arm 1 owns `.heat_transport.p_fw_coolant_pump_mw`,
`p_blkt_coolant_pump_mw`, `p_shld_coolant_pump_mw`, `p_div_coolant_pump_mw`; arm 3 owns
`.primary_pumping.p_fw_blkt_coolant_pump_mw` plus the last two of those. Any `Switch`
over this slot has a partial overlap by construction, and so do the four `power.py` nodes
that consume it.

## deviations from PROCESS

1. **`.fwbs.breeder_f` is not owned by any node.** PROCESS clamps it in place at
   `:404-405` -- a read of a field followed by a write to it, a cottax self-loop. Kept as
   a local because the clamp is idempotent and `.fwbs.breeder_f`'s **only** reader
   anywhere under `process/models/` is `:411`, two lines later (grep for `\.breeder_f`
   returns `:404`, `:405`, `:411`, and `iteration_variables.py:102`). It is also inert on
   any run where the solver owns the field: iteration variable 108 declares bounds
   `(0.060, 1.0)`, strictly inside `[1e-10, 1.0]`. **Consequence to know about**: after a
   port run `.fwbs.breeder_f` holds the *unclamped* input where PROCESS holds the clamped
   one. They differ only for inputs outside the solver's own declared bounds.
2. **The four `nuclear_heating_*` nodes own minted `_unnormalised` names under
   `.ccfe_hcpb`**, not the real `.fwbs` fields. `run()` overwrites all four at
   `:220-264`, and the overwritten value is what every downstream consumer reads, so the
   raw and the final are two different quantities PROCESS happens to store in one slot.
   Same shape as `plasma_physics.py`'s `pden_plasma_core_rad_mw_unclipped` mint. The
   original record already flagged this for `p_tf_nuclear_heat_mw` alone; it is true of
   all four.
3. **`calculate_nuclear_heating_magnets_*` recomputes the FW void fraction as a local.**
   `FirstWallCoolantVoidFractions` owns the field; the magnets function keeps PROCESS's
   own expression as `vffwm` so its harness case stays a 1:1 diff against
   `nuclear_heating_magnets(False)` rather than a transcription. One formula, two nodes,
   one owner -- the duplicate is a local, not a second write.
4. **`nuclear_heating_fw` returns `nan` where PROCESS raises `ProcessValueError`**
   (`:646-650`), per `test_harness.md`'s domain-guard policy. The untaken branch is a
   constant, so no NaN reaches the tangent.
5. **`nuclear_heating_blanket`'s `logger.error`** (`:689-696`) is dropped -- a Python side
   effect conditioned on a traced value, with no effect on the return.
6. **`pfactor`'s power law goes through `safe_pow`** (`:868-874`). The exponent is
   `(gamma_he - 1) / gamma_he ~= 0.4`, in the `0 < p < 1` band of `next_steps.md` §9's
   trap. The base is far from zero on any realistic input, so this changes no value; it is
   what `test_gradient_finite_at_zero` demands and it is free.
7. **`run()`, `component_volumes`, `component_masses` and `powerflow_calc` are not nodes.**
   Each is orchestration or a straight-line block that two nodes divide between them; the
   tree's slot order carries what their statement order carried.

## tier signal

All eleven nodes: **tier 1**. No internal iteration anywhere on this path, no `scipy`, no
`copy.deepcopy`, no reachable CoolProp call, no data-dependent early exit or loop.

## calls into other models

- `pumping_powers_as_fractions` (`process/models/engineering/ivc_functions.py:27-96`) --
  called only from `powerflow_calc`'s arm 1, which is UNPORTED. Not in this port.
- `calculate_pipe_bend_radius` (same file) -- called from `run():81`, writes
  `.fwbs.radius_blkt_channel_90_bend`/`_180_bend`, neither of which reaches any of the
  sixteen boundary variables. Out of the minimal closure.
- `primary_coolant_properties`/`thermo_hydraulic_model` (`blanket_library.py`) -- arm 2
  only, not entered.
- `FluidProperties.of` (CoolProp) -- `:794` only, behind the dead `WATER` arm.

## JAX-difficulty flags

- **Switch branches** -- all resolved structurally (one occupant per value). No
  `jnp.where` remains on any switch, which is the change from the original record.
- **`vv_density`'s guarded division** (`:509-512`) -- `minor`, `workaround-known`. The
  source's Python `if` short-circuits the division; a traced `jnp.where` evaluates both
  branches, so the port substitutes the *denominator*
  (`jnp.where(vol_vv == 0.0, 1.0, vol_vv)`) rather than selecting the result.
- **`safe_pow` at `pfactor`** -- see deviation 6.
- **Divisions with no guard**: `fwclfr`'s `a_fw_total * 0.5 * (dr_fw_inboard +
  dr_fw_outboard)` (`:337-339`), `blanket_density`'s `vol_blkt_total`,
  `shield_density`'s `vol_shld_total`, the renormalisation's `pnuc_tot_blk_sector`, and
  `1 - fpump`. All are non-zero on any physical operating point and PROCESS guards none of
  them either; `test_gradient_finite_at_zero` skips each, because zeroing the denominator
  makes the *value* non-finite too, which is the out-of-domain case that test excludes by
  design.

## harness

`tests/functional_process/models/blankets/test_hcpb.py`, eleven contracts. Two adapter
techniques are worth naming because they generalise:

- **`_RenormalisationOnly`**, a `CCFE_HCPB` subclass with every step of `run()` stubbed
  except `:103-148` and `:195-276`, and the four unnormalised powers injected where the
  real `nuclear_heating_*` calls would have written them. This is how a block that is
  *lines inside a method* rather than a method gets a real reference instead of a
  transcription. `next_steps.md` §14.3 names the general problem -- "any split finer than
  PROCESS's own has no 1:1 reference" -- and this is one way to buy one back.
- **`powerflow_calc`'s two nodes share one contract**, because PROCESS's method
  *overwrites* `psurffwi`/`psurffwo` (`:805-814`) before its pumping arm reads them, so
  seeding them and calling the method would test nothing. The port's two functions are
  composed in the graph's own order and all seven outputs diffed at once.

Sample provenance: `tests/unit/models/blankets/test_ccfe_hcpb.py`'s parametrised cases,
plus a reference-run point per contract read off the assembled `DataStructure` after four
`_call_models_once` passes on `large_tokamak_eval.IN.DAT`. The renormalisation contract's
four unnormalised powers were recomputed from that run's inputs through PROCESS's own four
routines and sum to `1411.3788234012184`, the `.ccfe_hcpb.pnuc_tot_blk_sector` measured on
the same run to the last digit -- an independent check that the sample sits *on* the
operating point.

## open questions (tokamak extension)

1. **`.heat_transport.p_fw_coolant_pump_mw`/`p_blkt_coolant_pump_mw`.** No producer at
   `i_p_coolant_pumping == 3`; the port needs the *power* subsystem's arm-3 occupants to
   read `.primary_pumping.p_fw_blkt_coolant_pump_mw` instead. Whether that is a change to
   `power.py`'s existing nodes or four new occupants is `next_steps.md` §14.9 item 2's
   call, not this unit's.
2. **`n_divertors`: switch or value?** Same question `blanket_library.md` raises; this
   file branches on it in two places (`:360`, `:215`) and `models/vacuum/vacuum.py` reads
   it as a plain value. One policy line settles both.
3. **Registering `NuclearHeatingMagnetsSphericalTokamak`/`NuclearHeatingShieldSphericalTokamak`.**
   Written and tested, but a machine at `itart == 1` also needs the centrepost chain and
   the D-shaped blanket geometry, neither ported. They should stay written-and-unregistered
   until then, and the reason should live wherever `indat.py` records UNPORTED values.

---

# The original stellarator-scope record (unchanged below this line)

**Ported (3/3).** All three in-scope methods are self-contained tier-1 and are ported in
`hcpb.py`/`test_hcpb.py`: `nuclear_heating_blanket` and `nuclear_heating_shield` were
already `@staticmethod`s in the source (no `self.data` access at all — the pure port is
almost a verbatim copy); `nuclear_heating_magnets` is a `self`-bound instance method with
a genuine `self.data` footprint and gets the usual "close the data back-door" extraction,
same shape as `PlasmaDensityLimit.calculate_density_limit`/`st_sudo_density_limit`.

**A live call-site bug was found while tracing the stellarator caller — see "open
questions" #1. It does not block this port** (the port targets the staticmethods'
declared signatures, not the broken call site), but it directly affects `st_fwbs`'s S2
sub-computation, which is what this unit was the sole blocker for, so it is flagged
prominently rather than buried.

## source

`process/models/blankets/hcpb.py` (1663 lines total). Registry unit #13, scope: the
three named methods only — `nuclear_heating_blanket` (654-698), `nuclear_heating_shield`
(700-769), `nuclear_heating_magnets` (463-609) — per `unit_registry.md`'s row 13 and
`next_steps.md` §4b's dispatch table.

**Transitive-closure check** (same discipline as unit #10/#22, unit #20/#23): none of the
three methods calls any other `Model` method, any other method within `hcpb.py` itself
(including the fourth sibling `nuclear_heating_fw`, out of scope — the three in-scope
methods neither call it nor are called by it), or any module-level function outside
`numpy`. All three are dead-ends — no further scope expansion needed. Confirmed by direct
read of all three bodies (`nuclear_heating_magnets`: 463-609; `nuclear_heating_blanket`:
653-698; `nuclear_heating_shield`: 700-769) and by grep (`grep -n "self\.\(nuclear_heating\|hcpb\)"` inside the file returns
only the three's own `def` lines and their call sites in `run()`).

**Call sites** (for context, not part of this unit's own footprint):
- `CCFE_HCPB.run()` (tokamak path), lines 150-182: calls all three with correctly
  supplied keyword arguments matching their declared signatures — `component_masses()`
  first (150), then
  `nuclear_heating_magnets(output=output)` (155), then `nuclear_heating_fw` (157), then
  `nuclear_heating_blanket(m_blkt_total=..., p_fusion_total_mw=...)` (164),
  then `nuclear_heating_shield(itart=..., ..., x_blanket=self.data.ccfe_hcpb.x_blanket,
  ...)` (174) — note `x_blanket` here is `nuclear_heating_magnets`'s own output,
  confirming the real call order `magnets → shield` (shield needs magnets' `x_blanket`).
  **All four of these are now in scope and ported — see the tokamak extension above.**
- `stellarator.py`'s `blanket_neutronics()` (`stellarator.py:422-461`, part of unit #1's
  `st_fwbs` S2 per the synthesis doc), calls `self.hcpb.nuclear_heating_blanket()` (440,
  **zero arguments**), `self.hcpb.nuclear_heating_magnets(False)` (443, correct — matches
  `nuclear_heating_magnets(self, output: bool)`), `self.hcpb.nuclear_heating_shield()`
  (458, **zero arguments**). See open question #1 — the zero-argument calls do not match
  either staticmethod's declared signature.

## data footprint

### `nuclear_heating_blanket(m_blkt_total, p_fusion_total_mw)`

Already pure in the source (`@staticmethod`, no `self` access at all).

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.fwbs.m_blkt_total` | read (via caller's kwarg) | explicit-arg | plain parameter |
| `.physics.p_fusion_total_mw` | read (via caller's kwarg) | explicit-arg | plain parameter |
| `.ccfe_hcpb.p_blkt_nuclear_heat_total_mw_unnormalised` | write (return) | own-write; **minted** in the tokamak extension (was `.fwbs.p_blkt_nuclear_heat_total_mw`) | first return value |
| `.ccfe_hcpb.exp_blanket` | write (via caller's return-value assignment) | own-write (returned) | second return value |

`logger.error(...)` on `p_blkt_nuclear_heat_total_mw < 1` is a diagnostic side effect,
not a data write — see JAX-difficulty flags.

### `nuclear_heating_shield(itart, dr_shld_outboard, dr_shld_inboard, shield_density, whtshld, x_blanket, p_fusion_total_mw)`

Already pure in the source (`@staticmethod`, no `self` access at all).

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.itart` | read (via caller's kwarg) | **switch** (was: explicit-arg) | gates a 2-way formula branch; now selects between the two occupants |
| `.build.dr_shld_outboard` | read | explicit-arg | |
| `.build.dr_shld_inboard` | read | explicit-arg | **read by the `itart == 0` arm only** — the conventional occupant declares it, the spherical one does not |
| `.ccfe_hcpb.shield_density` | read | explicit-arg | produced by `nuclear_heating_magnets` at the real call site (both `run()` and `blanket_neutronics()` call magnets before shield) — an ordinary graph edge |
| `.fwbs.whtshld` | read | explicit-arg | shield mass, produced by `component_masses` (ported in the tokamak extension) |
| `.ccfe_hcpb.x_blanket` | read | explicit-arg | also produced by `nuclear_heating_magnets` — second edge from that node into this one |
| `.physics.p_fusion_total_mw` | read | explicit-arg | |
| `.ccfe_hcpb.p_shld_nuclear_heat_mw_unnormalised` | write (return) | own-write; **minted** in the tokamak extension | 1st return value |
| `.ccfe_hcpb.exp_shield1` | write (return) | own-write (returned) | 2nd |
| `.ccfe_hcpb.exp_shield2` | write (return) | own-write (returned) | 3rd |
| `.ccfe_hcpb.shld_u_nuc_heating` | write (return) | own-write (returned) | 4th |

### `nuclear_heating_magnets(self, output)`

Not a `@staticmethod` — genuine `self.data` extraction needed. 21 reads (verified against
`tests/unit/models/blankets/test_ccfe_hcpb.py::test_nuclear_heating_magnets`'s fixture
field list as a cross-check, which independently enumerates the same 21 — including
`dr_fw_outboard`, easy to miss on a first read since it only appears inside the
`x_blanket` formula's `(dr_fw_inboard + dr_fw_outboard) / 2.0` term, not in the
inboard-only void-fraction calculation two lines above it).

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.fwbs.radius_fw_channel` | read | explicit-arg | no producer anywhere in `process/models/` — a pure input |
| `.fwbs.dx_fw_module` | read | explicit-arg | likewise |
| `.build.dr_fw_inboard` | read | explicit-arg | used twice: void-fraction denominator and `x_blanket`'s FW term |
| `.build.dr_fw_outboard` | read | explicit-arg | **easy to miss** — only appears in `x_blanket`'s FW term |
| `.fwbs.den_steel` | read | explicit-arg | |
| `.fwbs.m_blkt_total` | read | explicit-arg | `component_masses`' output |
| `.fwbs.vol_blkt_total` | read | explicit-arg | `blanket_library`'s output |
| `.fwbs.whtshld` | read | explicit-arg | `component_masses`' output |
| `.fwbs.vol_shld_total` | read | explicit-arg | `shield.py:138` |
| `.build.dr_vv_inboard` | read | explicit-arg | |
| `.build.dr_vv_outboard` | read | explicit-arg | |
| `.fwbs.m_vv` | read | explicit-arg | `vacuum.py:799` |
| `.fwbs.vol_vv` | read | explicit-arg | denominator of `vv_density`; guarded in the port |
| `.physics.itart` | read | **switch** (was: explicit-arg) | two 2-way branches; now selects between the two occupants |
| `.build.dr_blkt_outboard` | read | explicit-arg | |
| `.build.dr_blkt_inboard` | read | explicit-arg | **conventional occupant only** |
| `.build.dr_shld_outboard` | read | explicit-arg | |
| `.build.dr_shld_inboard` | read | explicit-arg | **conventional occupant only** |
| `.fwbs.fw_armour_thickness` | read | explicit-arg | |
| `.tfcoil.whttflgs` | read | explicit-arg | **spherical occupant only** |
| `.tfcoil.m_tf_coils_total` | read | explicit-arg | **conventional occupant only** |
| `.physics.p_fusion_total_mw` | read | explicit-arg | |
| `.fwbs.f_a_fw_coolant_inboard` | write (return) | **moved** to `FirstWallCoolantVoidFractions` in the tokamak extension | |
| `.fwbs.f_a_fw_coolant_outboard` | write (return) | **moved**; numerically identical to the inboard value (source: `f_a_fw_coolant_outboard = f_a_fw_coolant_inboard`) | |
| `.ccfe_hcpb.armour_density` | write (return) | own-write (returned) | |
| `.ccfe_hcpb.fw_density` | write (return) | own-write (returned) | |
| `.ccfe_hcpb.blanket_density` | write (return) | own-write (returned) | |
| `.ccfe_hcpb.shield_density` | write (return) | own-write (returned) | consumed by `nuclear_heating_shield` |
| `.ccfe_hcpb.vv_density` | write (return) | own-write (returned) | |
| `.ccfe_hcpb.x_blanket` | write (return) | own-write (returned) | consumed by `nuclear_heating_shield` |
| `.ccfe_hcpb.x_shield` | write (return) | own-write (returned) | **not** read by `nuclear_heating_shield` (that function computes its own, differently-scaled shield exponent) |
| `.ccfe_hcpb.tfc_nuc_heating` | write (return) | own-write (returned) | |
| `.ccfe_hcpb.p_tf_nuclear_heat_mw_unnormalised` | write (return) | own-write; **minted** in the tokamak extension | **re-normalised later** by `run()`'s lines 253-264 — this node owns only the *first* write. That prediction is what the extension acted on. |

Local intermediates, all `local-intermediate` per schema: `vffwm`, `d_vv_all`,
`th_blanket_av`/`th_shield_av`.

## tier signal

All three: **tier 1**. No internal iteration anywhere, no calls into other models.

## switches touched (original entry, superseded by the tokamak extension's table)

- `.physics.itart` — **new, not in `switches.md`'s original 10.** The original
  recommendation here was **keep-static** (a plain traced argument with `jnp.where`),
  on the ground that PROCESS's own staticmethod signature already unified both branches
  and the port's job was to reproduce that function rather than redesign it.
  **That recommendation is withdrawn.** `next_steps.md` §14.2's binding policy — no switch
  is a static kwarg, whatever its reads — post-dates it, and acting on it is strictly
  better here anyway: the branches' reads-sets genuinely differ, so the unified version
  declared `.build.dr_blkt_inboard`, `.build.dr_shld_inboard` and `.tfcoil.m_tf_coils_total`
  as edges a spherical tokamak does not have, and `.tfcoil.whttflgs` as one a conventional
  tokamak does not. See the extension's switch table.

## calls into other models

None among the three. All are dead-ends — confirmed by direct read and by grep.

## JAX-difficulty flags (original)

- **`logger.error(...)` inside `nuclear_heating_blanket`** — `minor`,
  `workaround-known`: dropped in the port.
- **`itart`'s three 2-way branches** — **resolved structurally** in the tokamak extension
  (one occupant per value); no `jnp.where` remains.
- **`vv_density`'s guarded division** — `minor`, `workaround-known`; the port guards the
  denominator so the untaken branch stays finite.
- No CoolProp, no `scipy`, no `copy.deepcopy`, no data-dependent early exit or loop.

## open questions (original)

1. **A live call-site bug**: `stellarator.py`'s `blanket_neutronics()` (lines 440, 458)
   calls `self.hcpb.nuclear_heating_blanket()` and `self.hcpb.nuclear_heating_shield()`
   with **zero arguments**, but both are `@staticmethod`s requiring 2 and 7 explicit
   keyword arguments respectively (confirmed: neither has a default value for any
   parameter). This would raise `TypeError: nuclear_heating_blanket() missing 2 required
   positional arguments` the moment a stellarator run with `blktmodel == 1` actually
   executes `blanket_neutronics()`. **Not exercised by any existing test**:
   `tests/unit/models/stellarator/test_stellarator.py` only ever sets `blktmodel=0`
   (confirmed by grep — three hits, all `blktmodel=0`), so this branch appears to be
   dead in practice, not merely untested in principle. Two candidate fixes exist (pass the
   same explicit keyword arguments `run()`'s own call sites already use at lines 164-182;
   or treat it as a genuine PROCESS bug to report upstream) — **not resolved here**.
   **The tokamak extension does not touch this**: a tokamak reaches every one of these
   methods through `CCFE_HCPB.run()`'s own correct call sites, never through
   `blanket_neutronics()`, so the tokamak port is unaffected by the bug and the bug is
   unaffected by the port. The stellarator's `blktmodel == 1` arm remains blocked on it.
2. **`itart`'s switches-table gap** — recorded here as a new finding, not added to
   `switches.md` itself. The extension's switch table supersedes the recommendation.
3. **`.ccfe_hcpb.x_shield`'s consumer**, if any, beyond reporting — **answered by the
   tokamak trace**: grep across the 38 reached files finds no read of
   `.ccfe_hcpb.x_shield` outside `nuclear_heating_magnets` itself and
   `write_output` (`hcpb.py:584`). It is a reporting output. `NuclearHeatingMagnets*`
   still owns it, because PROCESS writes it and an unowned written field is a boundary
   input in disguise.


## 2026-08-27 — the `n_divertors == 2` arms ported (double-null wave)

This file's two `n_divertors` sites, named by `indat.py`'s now-deleted
`('n_divertors', 2)` refusal, both have occupants.

| slot | single-null occupant | double-null occupant |
|---|---|---|
| `.tokamak.ccfe_hcpb.divertor_surface_and_plate_mass` | `DivertorSurfaceAndPlateMassSingleNull` | `DivertorSurfaceAndPlateMassDoubleNull` |
| `.tokamak.ccfe_hcpb.nuclear_heating_renormalisation` | `NuclearHeatingRenormalisationSingleNullConventional` | `NuclearHeatingRenormalisationDoubleNullConventional` |

New family bases `DivertorSurfaceAndPlateMass` and `NuclearHeatingRenormalisation` carry
the slot annotations in `models/blankets/namespace.py`.

**Divertor surface and plate mass (`hcpb.py:353-367`).** `if n_divertors == 2:
a_div_surface_total *= 2.0` — two divertors, twice the plate area, and twice the mass that
follows. The arms read the same six fields and differ by one literal factor; that is still
an occupant and not an `n_divertors` multiplier, under `next_steps.md` §14.2's `istore`
precedent. The read/own conflict recorded above — this node writes
`.divertor.a_div_surface_total` while `ComponentMasses` reads it, so it is ordered before
`ComponentMasses` — is unchanged and applies to both arms.

**Nuclear-heating renormalisation (`hcpb.py:213-217`).** PROCESS spells `f_geom_blanket`
as `1 - n_divertors * f_ster_div_single - f_geom_cp` — a *multiplication*, not an `if`,
which is exactly why the two arms are the same shape with a different constant. At
`n_divertors == 2` and `f_geom_cp` the conventional arm's literal `0` (`:144`), it is
`1 - 2 * f_ster_div_single`. Everything else, including the two deliberate
non-declarations `itart == 0` buys (`+ pnuc_cp_tf` provably `+ 0` at `:263`;
`f_geom_cp * p_neutron_total_mw` provably `0` at `:268`), is unchanged.

**This slot is a 2x2 and only its `itart == 0` row is written**, which is why
`_nuclear_heating_renormalisation_arm` now asks `itart` *first*: a spherical machine
refuses on `('itart_hcpb', 1)` however its divertors are counted, and reporting the
divertor count instead would name a precondition the port now meets. Consequence worth
stating plainly: **neither spherical-tokamak input file reaches the new renormalisation
arm**, because both set `itart = 1`. It was written because `hcpb.py:215` is one of the
double-null wave's named sites and because a conventional double-null machine needs it;
it does not by itself move those two files' frontier. Open question 3's answer stands
unchanged.

**Tests.** `TestDivertorSurfaceAndPlateMassDoubleNull` and
`TestNuclearHeatingRenormalisationDoubleNull`, both Tier 1, both reusing their
single-null sibling's legacy point and fuzz box — with `f_ster_div_single` capped below
`0.5`, where `1 - 2 * f_ster_div_single` changes sign and the renormalisation stops
meaning anything. The renormalisation reference is the existing `_RenormalisationOnly`
subclass driven at `n_divertors = 2`; `_seed_component_masses` and `_run_renormalisation`
gained an `n_divertors` parameter defaulting to `1`, so no existing case moved. Green at
`--fp-gradients --fp-fuzz 40`.

No new boundary input.
