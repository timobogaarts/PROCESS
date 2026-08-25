# The switches that are still constructor kwargs — a survey, not a conversion

**Status: survey only. No code changed.** This is the per-case evidence
`model_tree_design.md` §8 step 4b's follow-up (ii) records as outstanding — "30 slots
still pass a switch as a static constructor kwarg … that is step 6's job, family by
family" — and step 6's own list ("the six recorded split-default deviations plus any
formula family §4's refined criterion catches") turns out to be a small subset of what
is actually there. Nothing here decides step 6's order by itself; it supplies the two
facts that should decide it, measured rather than estimated: **which switches make a node
declare reads the run being modelled does not have**, and **which switches are resolved
in two places that can disagree**.

The position being audited against is `machine_from_indat`'s own, quoted because every
finding below is an instance of it:

> The rejected alternative was one node owning the union of every variant's ports and
> branching internally. That would make a node read `eta_ecrh_injector_wall_plug` *and*
> `eta_lowhyb_injector_wall_plug` regardless of which is live, **inventing graph edges
> that do not exist in the run being modelled**, and would put a non-differentiable
> integer on a port.

Both halves of that sentence are violated by the tree as it stands: 79 invented reads
across 23 slots, and 10 declared ports carrying a switch integer.

## 1. What was measured, and how

Everything numeric below comes from one of four scripted measurements against the live
`REFERENCE_MACHINE`/`GRAPH` and PROCESS's own converged answer for the reference run.
No claim here was read off a name.

1. **Which slots carry a switch.** Walk `REFERENCE_MACHINE` as a `ModelNamespace` tree;
   for every occupant, take `dataclasses.fields(occ)` where `f.metadata["static"]`.
   *Deliberately dynamic, not an AST walk over `total_process.py`'s class bodies* — see
   §2 for the two slots an AST walk cannot see.
2. **Declared reads and writes.** `GRAPH.definitions[name].inputs/.outputs`, i.e. the
   ports as assembled, not as written.
3. **Dead declared reads.** Seed every declared input from the converged
   `DataStructure` (`mda_harness.converged_data(REFERENCE_INPUT_FILE)` +
   `_ground_truth`), build `jax.make_jaxpr` of the node's own callable (`__call__`, or
   `.step` for a `FixedPointFunction`), and report every `invar` that appears in no
   equation and in no `outvar`. A read that survives constant-folding as `0.0 * x`
   counts as **live** — the measurement is deliberately conservative, so every "dead"
   below is dead with certainty and the true count may be higher.
4. **Attribution to a switch.** Re-run (3) with the occupant rebuilt via
   `dataclasses.replace` at every other value of the switch's `IntEnum`, and diff the
   dead sets. A read dead here and live at some other value is **invented by the
   switch**. For the 16 slots carrying two or more switches the sweep is over the full
   product of their enums, not one switch at a time — single-switch flips are blind to
   jointly-gated reads, and three of the cases below (`availability.electric_production`,
   `availability.cplife_avail`, `costs.pf_magnet_cost`) are only visible jointly.

The scripts live in this session's scratchpad, not in the repo; they are ~40 lines each
and are described precisely enough above to be rebuilt. Any of them is a candidate to
become a real test — see §7.

**One methodological limit, stated up front.** Where the other arm *raises*
(`NotImplementedError`/`ValueError`, seven `ife` sites, `istore == 3`,
`i_plasma_pedestal != 0`, `i_tf_sc_mat == 9`, `i_confinement_time` 0 and 25), method (4)
cannot measure its reads. Those rows report the reads-difference the raise message and
the PROCESS source assert, marked *(unmeasured)*.

## 2. The count: **32**, not 30

`model_tree_design.md` step 4b's "30 slots" is a count of *class-body* constructor
kwargs: 32 slots in `total_process.py` are written as `Name(...)` with arguments, less
`profile_grid` (`n_plasma_profile_elements`) and `impurity_radiation_totals`
(`imp_indices`), which §3(b)/(c) exempts as shape/membership. That derivation is
correct as far as it goes and reproduces exactly.

Walking the *assembled* tree instead finds **36 slots carrying `eqx.field(static=True)`
fields, of which 32 carry at least one PROCESS switch**. The four that carry statics but
no switch are `physics.profiles.profile_grid` (shape), `physics.impurity_radiation_totals`
(membership), `stellarator.machine_config` (the stellarator config payload) and
`stellarator.profile_values` (`rho`, a sample point). `costs.pf_magnet_cost` carries
`n_cs_pf_coils`, a count, *alongside* three genuine switches, so it counts as a switch
slot.

The two slots the class-body walk cannot see are the ones **`machine_from_indat` builds**,
so their kwargs are in the factory rather than in a slot default:

- `physics.confinement_time.model` — built with `i_confinement_time=ISS04_STELLARATOR`,
  `i_rad_loss=CORE_ONLY`, `i_plasma_ignited=IGNITED` inside the `istell` resolution;
- `availability.electric_production` — `ELECTRIC_PRODUCTION[1]` is a
  `functools.partial(PlantElectricProductionReactor, itart=…, i_tf_sup=…,
  i_blkt_dual_coolant=…, i_p_coolant_pumping=…)`.

Both are *switched slots that additionally hardcode four more switches inside their
occupant*, which is the shape most likely to be missed by any source-level count.
`test_switch_coverage.py` (moved to `tests/functional_process/` by a concurrent pass
while this survey was being written; cited by basename throughout) already lists both
(under `i_plasma_ignited`,
`i_confinement_time`, `i_rad_loss`, `i_p_coolant_pumping`), so the test file and this
survey agree; only step 4b's prose is one derivation short.

**59 (slot, switch) pairs over 26 distinct switches.** The largest fan-outs are `ife`
(7 slots), `i_tf_sup` (5), `i_p_coolant_pumping` (5), `i_thermal_electric_conversion` (4),
`itart` (4).

## 3. Summary table — one row per (slot, switch) pair

Columns, in the brief's order and compressed:

- **val** — the hardcoded value, as an integer.
- **ar** — arity: how many values the switch has in PROCESS (`data_structure/*.py`
  docstring or the upstream `IntEnum`); `*` marks a value the port refuses.
- **br** — does the model body branch on it? `Y` = a real `if`; `G` = guard-only (the
  other arm raises, so there is one live formula); `P` = precondition asserted in
  `__check_init__`, never consulted by the body; `n` = passed and never branched on.
- **reads** — do the branches differ in declared reads? `live` = they differ *and* the
  difference shows as dead declared reads on the reference machine (the count is how
  many); `off` = they differ, but only between two arms neither of which is the
  reference one (nothing invented here); `no` = no value of this switch changes any
  declared read; `?` = unmeasurable, other arm raises.
- **writes** — do the branches differ in declared writes? (`no` throughout; see §4.0.)
- **cp** — does a counterpart class already exist? `Y` / `~` (exists but is not a usable
  occupant, see the note) / `-`.
- **split** — is this switch *also* resolved elsewhere? `R` = also drives a registry
  slot; `p` = also read as a declared port by some node; `Rp` = both.

| slot | switch | val | ar | br | reads | writes | cp | split |
|---|---|---:|---:|:--:|---|---|:--:|:--:|
| `costs.first_wall_cost` | `ife` | 0 | 2 | G | ? (0 live) | no | - | p |
| `costs.blanket_cost` | `ife` | 0 | 2 | G | ? (0 live) | no | - | p |
| `costs.shield_cost` | `ife` | 0 | 2 | G | ? (0 live) | no | - | p |
| `costs.power_injection_cost` | `ife` | 0 | 2 | G | ? (0 live) | no | - | p |
| `costs.auxiliary_component_cooling_cost` | `ife` | 0 | 2 | G | ? (0 live) | no | - | p |
| `costs.fuel_processing_cost` | `ife` | 0 | 2 | G | ? (0 live) | no | - | p |
| `costs.cost_of_electricity` | `ife` | 0 | 2 | G | ? (0 live) | no | - | p |
| `costs.cost_of_electricity` | `itart` | 0 | 2 | Y | **live (3)** | no | - | p |
| `costs.cost_of_electricity` | `ireactor` | 1 | 2 | P | no | no | Y | Rp |
| `costs.cost_of_electricity` | `ipnet` | 0 | 2 | P | no | no | - | - |
| `costs.tf_magnet_cost_superconducting` | `supercond_cost_model` | 0 | 2 | Y | **live (3)** | no | - | - |
| `costs.pf_magnet_cost` | `iohcl` | 0 | 2 | Y | **live (11)** | no | - | - |
| `costs.pf_magnet_cost` | `i_pf_conductor` | 0 | 2 | Y | off | no | - | - |
| `costs.pf_magnet_cost` | `supercond_cost_model` | 0 | 2 | Y | off | no | - | - |
| `costs.energy_storage_cost` | `i_pulsed_plant` | 0 | 2 | Y | no | no | - | - |
| `costs.energy_storage_cost` | `istore` | 1 | 3\* | Y | ? (0 live) | no | - | - |
| `stellarator.coils.winding_pack_intersect_inputs` | `i_tf_sc_mat` | 1 | 9\* | Y | **live (6)** | no | ~ | p |
| `stellarator.neutron_wall_load` | `i_pflux_fw_neutron` | 1 | 2 | Y | **live (4)** | no | - | - |
| `stellarator.neutron_wall_load` | `ipowerflow` | 1 | 2 | Y | off | no | - | Rp |
| `stellarator.radiated_wall_load_and_fraction` | `i_pflux_fw_neutron` | 1 | 2 | Y | **live (4)** | no | - | - |
| `stellarator.radiated_wall_load_and_fraction` | `ipowerflow` | 1 | 2 | Y | off | no | - | Rp |
| `stellarator.heating_and_radiation_power` | `i_plasma_ignited` | 1 | 2 | Y | **live (1)** | no | - | - |
| `physics.profiles.…parabolic.ecrh_density_limit` | `i_plasma_pedestal` | 0 | 2 | G | ? (0 live) | no | - | R |
| `physics.confinement_time.model` | `i_confinement_time` | 38 | 52\* | Y | **live (10)** | no | - | - |
| `physics.confinement_time.model` | `i_rad_loss` | 1 | 3 | Y | **live (1)** | no | - | - |
| `physics.confinement_time.model` | `i_plasma_ignited` | 1 | 2 | Y | **live (1)** | no | - | - |
| `physics.fast_alpha_beta` | `i_beta_fast_alpha` | 1 | 2 | Y | no | no | - | - |
| `physics.plasma_composition` | `i_plasma_ignited` | 1 | 2 | Y | **live (1)** | no | - | - |
| `power.component_thermal_powers` | `i_p_coolant_pumping` | 1 | 4 | Y | **live (1)** | no | - | - |
| `power.component_thermal_powers` | `i_blkt_dual_coolant` | 0 | 3 | Y | **live (1)** | no | - | - |
| `power.component_thermal_powers` | `i_thermal_electric_conversion` | 2 | 5 | Y | **live (2)** | no | ~ | - |
| `power.component_thermal_powers` | `i_blanket_type` | 1 | 2 | Y | no | no | ~ | - |
| `power.component_thermal_powers` | `secondary_cycle_liq` | 4 | 2\* | Y | no | no | ~ | - |
| `power.delta_eta_step` | `i_p_coolant_pumping` | 1 | 4 | Y | **live (1)** | no | - | - |
| `power.delta_eta_step` | `i_blkt_dual_coolant` | 0 | 3 | Y | **live (1)** | no | - | - |
| `power.delta_eta_step` | `i_thermal_electric_conversion` | 2 | 5 | Y | no | no | - | - |
| `power.eta_turbine_step` | `i_thermal_electric_conversion` | 2 | 5 | Y | **live (2)** | no | ~ | - |
| `power.eta_turbine_step` | `i_blanket_type` | 1 | 2 | Y | no | no | ~ | - |
| `power.etath_liq_step` | `secondary_cycle_liq` | 4 | 2\* | Y | **live (1)** | no | ~ | - |
| `power.temp_turbine_coolant_in_step` | `i_thermal_electric_conversion` | 2 | 5 | Y | **live (1)** | no | ~ | - |
| `power.temp_turbine_coolant_in_step` | `i_blanket_type` | 1 | 2 | Y | no | no | ~ | - |
| `power.temp_turbine_coolant_in_step` | `secondary_cycle_liq` | 4 | 2\* | Y | **live (1)** | no | ~ | - |
| `power.p_fw_div_heat_deposited_mw_step` | `i_p_coolant_pumping` | 1 | 4 | Y | **live (1)** | no | - | - |
| `power.p_fw_blkt_coolant_pump_mw_step` | `i_p_coolant_pumping` | 1 | 4 | Y | **live (1)** | no | - | - |
| `power.cryo_q_nuc_step` | `i_tf_sup` | 1 | 3 | Y | **live (1)** | no | - | Rp |
| `power.cryo_q_nuc_step` | `inuclear` | 0 | 2 | Y | **live (1)** | no | - | - |
| `power.cryo_q_loads_step` | `i_tf_sup` | 1 | 3 | Y | **live (4)** | no | - | Rp |
| `power.cryo_q_loads_step` | `i_pf_conductor` | 0 | 2 | Y | off | no | - | - |
| `power.cryo_loads` | `i_tf_sup` | 1 | 3 | Y | **live (4)** | no | - | Rp |
| `power.cryo_loads` | `i_pf_conductor` | 0 | 2 | Y | off | no | - | - |
| `power.acpow` | `i_pf_energy_storage_source` | 2 | 3 | Y | **live (1)** | no | - | - |
| `availability.electric_production` | `itart` | 0 | 2 | Y | **live (1)** | no | ~ | p |
| `availability.electric_production` | `i_tf_sup` | 1 | 3 | Y | (joint) | no | ~ | Rp |
| `availability.electric_production` | `i_blkt_dual_coolant` | 0 | 3 | Y | **live (2)** | no | ~ | - |
| `availability.electric_production` | `i_p_coolant_pumping` | 1 | 4 | Y | (joint) | no | ~ | - |
| `availability.avail` | `ibkt_life` | 0 | 2 | Y | **live (1)** | no | - | - |
| `availability.avail` | `itart` | 0 | 2 | Y | **live (1)** | no | - | p |
| `availability.cplife_avail` | `i_tf_sup` | 1 | 3 | Y | (joint) | no | ~ | Rp |
| `availability.cplife_avail` | `itart` | 0 | 2 | Y | **live (6, joint)** | no | ~ | p |

"(joint)" marks a switch whose read effect is gated by another switch on the same node,
so the count is attributed to the pair, not split: `electric_production`'s
`.tfcoil.p_cp_coolant_pump_elec` needs `itart == 1 and i_tf_sup == 0`; its
`.heat_transport.etath_liq`/`.power.p_blkt_liquid_breeder_heat_deposited_mw` need
`i_blkt_dual_coolant > 0 and i_p_coolant_pumping == MECHANICAL`; `cplife_avail`'s six
need `itart == 1` before `i_tf_sup` can choose between them.

**Totals.** Of 345 declared reads on the 32 switch-carrying slots, **79 are dead on the
reference machine's live arm and recoverable by changing one of that slot's own
switches** — invented edges, in `machine_from_indat`'s sense, 23 % of the surface. Nine
slots invent nothing: the six pure-`ife` cost nodes, `energy_storage_cost`,
`ecrh_density_limit` and `fast_alpha_beta`.

A further 15 declared reads are dead on the reference machine but *not* recoverable at
any value of that node's switches, so they are a different defect and out of this
survey's remit; they are listed in §6 so they are not lost.

## 4. Per-case notes

### 4.0 The writes column is uniformly `no`, and that is worth one paragraph

No switch on any of the 32 slots changes the node's declared output set: a branch can
zero an output (`jnp.where(ireactor == 1, computed, 0.0)` in `structures_cost` /
`turbine_plant_equipment_cost`) or raise, but never omit one. Ragged output sets exist
only among *registry* occupants (`BLANKET_MASSES`'s unported arms write
`.fwbs.wtbllipb`/`.m_blkt_lithium` in place of `.m_blkt_li2o`/`.m_blkt_beryllium`), which
is why `model_tree_design.md` §4's "ragged arms are fine" was written about slots and not
about kwargs. **Conversion of a static kwarg therefore never costs a boundary-set change
from the writes side** — which removes the main reason a conversion would be expensive,
and is the single most useful fact in this survey for scheduling. The only case where an
occupant *should* acquire a smaller output set is `costs.cost_of_electricity` (§4.2),
where PROCESS does not call `coelc()` at all on the other arm.

### 4.1 `i_tf_sup` — a live incoherence, confirmed by running it

`i_tf_sup` is resolved in **seven** places: the `TF_POWER` registry
(`power.tf_power`), five static kwargs (`power.cryo_q_nuc_step`,
`power.cryo_q_loads_step`, `power.cryo_loads`, `availability.electric_production`,
`availability.cplife_avail`), and one declared *port*
(`.costs.tf_coil_power_conditioning_cost` reads `.tfcoil.i_tf_sup`). Measured:

```
machine_from_indat(<reference file with i_tf_sup = 0>)
  -> power.tf_power           = TfPowerResistive
     power.cryo_loads.i_tf_sup = 1     (still SUPERCONDUCTING)
     availability.electric_production.i_tf_sup = 1
     ... and three more
```

The assembled machine is resistive in one place and superconducting in five. This is not
a hypothetical: `test_switch_coverage._CHANGES_A_SLOT` **exercises exactly this
override** (`("i_tf_sup", 0, lambda m: type(m.power.tf_power).__name__)`) and asserts only
that `tf_power`'s occupant changed. Nothing asserts the rest of the tree followed.

### 4.2 `ireactor` — a second live incoherence, and a node that should not exist

`ireactor` drives the `ELECTRIC_PRODUCTION` registry, is a static on
`costs.cost_of_electricity`, and is a declared port on three further cost nodes
(`structures_cost`, `turbine_plant_equipment_cost`, `heat_rejection_cost`). Measured:
`i_reactor = 0` assembles `PowerProfilesOverTime` at `availability.electric_production`
while `costs.cost_of_electricity` keeps `ireactor=CALCULATED` — and `CostOfElectricity`'s
own `__check_init__` says that value is a *precondition*:

> PROCESS calls `coelc()` only when `ireactor == 1 and ipnet == 0` … On any other
> configuration PROCESS leaves `.costs.coe` at whatever it already held, **so this node
> must not exist**.

The declaration is right and the tree ignores it: on `ireactor = 0` the graph still
contains a node that its own constructor says must not exist, still the sole owner of
`.costs.coe`, still fed by a `p_plant_electric_net_mw` that the profile-only occupant does
not compute. `ireactor`/`ipnet` on this node are the clearest case in the survey of a
switch that is **not a branch at all but a node-existence condition** — the thing
`model_tree_design.md` §4 spells as a slot whose occupant is absent, and which the tree
lost the ability to spell when all four `X | None`s were removed in step 4b. Converting
this one requires deciding how the tree says "no occupant" again, which step 4b
deliberately closed off; see §6.

### 4.3 `ipowerflow`, `blktmodel`, `blkttype` — the joint keys are arm indices, and the factory feeds them switch values

`ipowerflow` is resolved in five places: `FW_AREA` (`stellarator.fw_area`), the
`BLANKET_SHIELD_POWER` joint key, two static kwargs (`neutron_wall_load`,
`radiated_wall_load_and_fraction`) and one port
(`.stellarator.fwbs.fw_blanket_shield_geometry` reads `.heat_transport.ipowerflow`). The
same split as §4.1 — but it is currently **unobservable**, because `ipowerflow = 0`
refuses to assemble at all, for a reason that is itself a defect.

`unit_registry.md` rows for the two joint switches define the keys as **arm indices**:

| key | `blktmodel_ipowerflow` | `blktmodel_blkttype` |
|---|---|---|
| 0 | `blktmodel == 1` (unported) | `blktmodel != 0` (unported) |
| 1 | `blktmodel != 1 & ipowerflow == 0` → `BlanketShieldPowerExponential` | liquid breeder, `blkttype ∈ {1,2}` (unported) |
| 2 | `blktmodel != 1 & ipowerflow == 1` → `DetailedPowerflow…` (PROCESS's default) | `blktmodel == 0 & blkttype ∉ {1,2}` → `BlanketComponentMasses` |

`machine_from_indat` derives them as

```python
blktmodel = switches.get("blktmodel", 2)                    # PROCESS's default is 0
... blktmodel if switches.get("ipowerflow", 1) else 0       # blktmodel_ipowerflow
... blktmodel if switches.get("blkttype", 3) >= 3 else 1    # blktmodel_blkttype
```

i.e. it passes `blktmodel`'s **value** where an **arm index** is required, and papers over
the mismatch with a default of `2` — which is not a legal value of `blktmodel` at all
(`fwbs_variables.py:479`: `blktmodel ∈ {0, 1}`). It gives the right occupant for the
reference run only because the reference file never mentions `blktmodel`. Measured, by
writing one line into a copy of the reference file:

| override | actual | correct |
|---|---|---|
| `blktmodel = 0` (PROCESS's own default, stated) | `NotImplementedError`, citing the `blktmodel == 1` arm's reason | `DetailedPowerflowBlanketShieldPower` + `BlanketComponentMasses` — i.e. the reference machine |
| `blktmodel = 1` | assembles `BlanketShieldPowerExponential` (the `ipowerflow == 0` arm) for the shield-power slot, then refuses at the mass slot citing the *liquid-breeder* reason | refuse at both, citing the `blktmodel == 1` reasons |
| `ipowerflow = 0` | `NotImplementedError` | `BlanketShieldPowerExponential` — **a ported arm, currently unreachable** |
| `blkttype = 1` | `NotImplementedError`, liquid-breeder reason | same (this one is right) |

The second row is the serious one: an override that PROCESS reads as "KIT HCPB
neutronics" silently selects a node written for a *different* switch's arm. This is the
`ScTfCoilNuclearHeating` bug class `unit_registry.md` records as found-and-fixed once
already, reintroduced by the key derivation rather than by a registration.

`test_switch_coverage.py` does not catch it and cannot, as written:
`_CAUSES_A_REFUSAL`'s `("blktmodel", 1, ("blktmodel_blkttype", 1))` passes because the
*mass* slot refuses, which happens after the shield-power slot has already chosen wrongly;
and its `("ipowerflow", 0, ("blktmodel_ipowerflow", 0))` case pins the third row above as
correct behaviour. Those two parametrisations are where I disagree with that module —
not with its inventory, which is complete and correct for the six switches it covers.

### 4.4 `i_plasma_pedestal` — the factory reads a value PROCESS overrides

`machine_from_indat` reads `i_plasma_pedestal` from the IN.DAT to pick
`PROFILE_PARAMETERISATION`. But `process/models/stellarator/initialization.py:31`
(`st_init`) sets `data.physics.i_plasma_pedestal = 0` **unconditionally whenever
`istell != 0`**, along with `data.build.iohcl = 0`. So on any stellarator run the file's
value is dead: an IN.DAT saying `istell = 6, i_plasma_pedestal = 1` makes this port
assemble `ProfileParameterisationPedestal` while PROCESS runs parabolic profiles. The
reference file says `0` and the two agree today; nothing checks that they must.
(The hardcoded `i_plasma_pedestal=PARABOLIC_PROFILE` on `ecrh_density_limit` is a
different matter — that node lives *inside* `ProfileParameterisationParabolic`, so its
kwarg is redundant with its own container and cannot disagree.)

### 4.5 `ife` on seven cost slots — the branches differ, but the difference is a refusal

All seven `ife` kwargs guard a `raise NotImplementedError` whose message names the other
arm's reads (`.ife.fwmatm`, `.ife.blmatm`, `.ife.shmatm`, `.ife.ifedrv`/`.cdriv0..3`,
`.ife.tdspmw`/`.tfacmw`, `.ife.uctarg`/`.reprat`). Measured: **zero** dead declared
reads on any of the seven. So the branches do differ in reads — in PROCESS — but the
port has one live formula per node and invents nothing. Conversion to occupants buys no
graph change; what it buys is *where the refusal happens* (§5, band c).

Note the inconsistency this survey found alongside: `.ife.ife` is simultaneously a static
kwarg on seven nodes and a **declared port** on three others (`costs.divertor_cost`,
`costs.magnets_cost`, `costs.power_conditioning_cost`, which use `jnp.where(ife == 1, 0.0,
…)`). Same switch, two spellings, one of them a non-differentiable integer on a port.

### 4.6 `i_tf_sc_mat` on `winding_pack_intersect_inputs` — the invented edge is a cycle

Six declared reads are dead at `ITER_NB3SN`: `.tfcoil.b_crit_upper_nbti`, `.tfcoil.bcritsc`,
`.tfcoil.fhts`, `.tfcoil.t_crit_nbti`, `.tfcoil.tcritsc`, `.tfcoil.j_tf_wp` — each live on
exactly one other material (`fhts`/`j_tf_wp` on Bi-2212, `bcritsc`/`tcritsc` on
user-defined Nb3Sn, the two NbTi criticals on Durham GL). The sixth is the interesting one.

`.tfcoil.j_tf_wp` is consumed only by `_critical_current_density_by_material`'s
`i_tf_sc_mat == 2` branch. It is also, measured, **the sole back-edge closing the coils
SCC**:

```
GRAPH.cycles contains
  ['.stellarator.coils.winding_pack_intersect_inputs',
   '.stellarator.coils.intersect',
   '^problem.stellarator.coils.intersect',
   '.stellarator.coils.winding_pack_total_size_post']

owner of .tfcoil.j_tf_wp  -> .stellarator.coils.winding_pack_total_size_post
readers of .tfcoil.j_tf_wp -> .stellarator.coils.winding_pack_intersect_inputs   (the only one)
reachability from …total_size_post to …intersect_inputs with that one edge deleted -> False
```

So on the run being modelled the four-node driven block is a two-node one
(`intersect` + its `^problem`), and `winding_pack_total_size_post` computes a
`.tfcoil.j_tf_wp` that nothing reads. `machine_from_indat`'s docstring already claims
"**a switch can decide whether the graph has a cycle**" and cites
`Stellarator.fw_area` for it; this is a second instance, and unlike `fw_area` it is
currently getting the answer wrong.

### 4.7 Two fixed points that are the identity map on the reference machine

Nine of the fourteen cycles in `GRAPH` are `FixedPointFunction` self-loops on
switch-carrying slots. For seven of them the self-read is *dead* on the live arm, so the
fixed point converges in one iteration — structurally misleading but numerically sound
(`delta_eta_step`, `temp_turbine_coolant_in_step`, `etath_liq_step`,
`p_fw_div_heat_deposited_mw_step`, `p_fw_blkt_coolant_pump_mw_step`, `cryo_q_nuc_step`,
`cryo_q_loads_step`). For the other two the self-read is the *only* live read, i.e. the
step is the identity and **every value is a fixed point**:

- **`availability.cplife_avail`** (`itart = CONVENTIONAL`). `calculate_cplife_next` opens
  with `if itart != 1: return cplife`. Measured: 6 of 7 declared reads dead, the survivor
  being `.costs.cplife` itself. The `FixedPoint` problem owns `^cond.costs.cplife` and
  determines nothing.
- **`power.eta_turbine_step`** (`i_thermal_electric_conversion = USER_INPUT`).
  `calculate_plant_thermal_efficiency`'s `USER_INPUT` arm is
  `return (eta_turbine, temp_turbine_coolant_in)`. Measured: `.heat_transport.eta_turbine`
  is the only live read; `delta_eta` and `temp_blkt_coolant_out` are dead. This one has a
  live downstream: `availability.electric_production` reads `.heat_transport.eta_turbine`
  to form `p_plant_electric_gross_mw`, which reaches `.costs.coe`, this run's objective.
  So a quantity the objective depends on is nominally *driven* while being, on this
  configuration, an unowned boundary input in disguise.

Whether either is numerically harmful depends on what the driver seeds them with, which
this survey did not measure (§6).

### 4.8 `CplifeAvail` — has the recorded reason expired? **Yes, and it does not help.**

`total_process.py`'s comment and `CplifeAvail`'s docstring both give one reason for not
registering `CpLifetimeSuperconducting`/`CpLifetimeResistive`:

> this node's `FixedPoint` problem *also* wants to own `.costs.cplife`, and only one
> producer of one `VarPath` may exist in any graph that registers both together.

**That reason has expired.** Occupants of one slot never coexist, by construction
(`model_tree_design.md` §2, "One field, one occupant: exclusive by construction"), so two
candidate owners of `.costs.cplife` placed in the *same slot* are not a conflict and
`Graph` would never see them together. The comment is a survivor of the flat-registry
era, when "registering both" was the only way to express an alternative.

**But the existing pair still cannot be the occupants**, for a reason the comment does not
give: they compute a different quantity. `CpLifetime{Superconducting,Resistive}` are
`ExplicitFunction`s returning the *fresh* centrepost lifetime;
`CplifeAvail.step` returns `calculate_cplife_lifetime_adjustment(fresh, life_plant,
f_t_plant_available)`, and passes the previous value straight through when `itart != 1`.
Swapping one in would silently drop the availability adjustment. Conversion here means
writing two new `FixedPointFunction` occupants (or one base plus two arms via §8.3's
`_rebound_signature` pattern) that keep the adjustment — the existing pair stays what its
own comment already calls it, "valid, independently useful standalone nodes".

And the switch that actually matters at this slot is **`itart`, not `i_tf_sup`**: on
`itart = 0` the node is the identity (§4.7), so the honest conversion is an `itart`-keyed
slot whose `CONVENTIONAL_ASPECT_RATIO` occupant is *absent*, leaving `.costs.cplife` as
the ordinary boundary input it really is on this run. `i_tf_sup` then keys a second,
nested choice inside the `SPHERICAL` occupant, which is exactly the nesting
`model_tree_design.md` §2 says the tree makes expressible.

### 4.9 `i_confinement_time` — 52 values, and the union node reads 32 fields for a formula that uses 12

`ConfinementTimeModel` has 52 members (0–51). Value 0 (`USER_INPUT`) is dead code in
PROCESS itself and the port faithfully raises on it; 25 raises "Scaling removed"; 51 is
double-named (`NCST`/`PAZ_SOLDAN_NT`). So 49 reachable scalings, one node.

Measured across all 49: 13 of 32 declared reads are dead at `ISS04_STELLARATOR`, 12 of
them recoverable at some other value (`aspect`, `eps`, `kappa`, `kappa95`, `m_fuel_amu`,
`m_ions_total_amu`, `n_charge_plasma_effective_vol_avg`, `qstar`,
`temp_plasma_electron_density_weighted_kev`, `triang` from `i_confinement_time`;
`pden_plasma_rad_mw` from `i_rad_loss`; `p_hcd_injected_total_mw` from
`i_plasma_ignited`). The thirteenth, `.physics.tauee_in`, is dead at **every** value —
it is `USER_INPUT`'s read, and `USER_INPUT` is unreachable.

This is the high-arity family, and it is the one case where "one occupant per value" is
plainly the wrong shape: 49 classes for a port with one live member. See §5 band (d).

### 4.10 `n_cs_pf_coils = 0` on `costs.pf_magnet_cost` — not a switch, and the biggest single distortion

Not a switch (a count, correctly exempted by §3(b)), but it is why this node's numbers
look the way they do and it deserves recording. `st_init` gives the stellarator no central
solenoid (`iohcl = 0`) and the reference machine has `n_cs_pf_coils = 0`, so both loops in
`calculate_pf_magnet_cost` execute zero times. Measured: **20 of 27 declared reads dead**,
of which 11 are recoverable by `iohcl` and the remaining 9 only by giving the machine PF
coils. Re-measured at `n_cs_pf_coils = 3` the dead count falls to 8, all recoverable by
`iohcl`/`supercond_cost_model`.

The honest reading is that a stellarator has no PF magnet cost to compute and this node's
27-port declaration asserts a dependence on the PF coil system that this device does not
have. That is a slot-shaped problem (an absent occupant), not a kwarg-shaped one, and it
is not step 6's — recorded here so it is not mistaken for one.

### 4.11 Switches spelled as ports — 10 sites, 6 switches

Directly against `machine_from_indat`'s "would put a non-differentiable integer on a
port", and against `naming_convention.md`'s "switches are not ports":

| node | port |
|---|---|
| `.costs.divertor_cost` | `.ife.ife` |
| `.costs.magnets_cost` | `.ife.ife` |
| `.costs.power_conditioning_cost` | `.ife.ife` |
| `.costs.convert_fpy_to_calendar` | `.physics.itart` |
| `.costs.structures_cost` | `.costs.ireactor` |
| `.costs.turbine_plant_equipment_cost` | `.costs.ireactor` |
| `.costs.heat_rejection_cost` | `.costs.ireactor` |
| `.costs.tf_magnet_cost_superconducting` | `.tfcoil.i_tf_sc_mat` |
| `.costs.tf_coil_power_conditioning_cost` | `.tfcoil.i_tf_sup` |
| `.stellarator.fwbs.fw_blanket_shield_geometry` | `.heat_transport.ipowerflow` |

Each of these six switches is *also* a static kwarg somewhere, so the tree currently
holds two different, unrelated answers to "what is `i_tf_sup`". Two of them
(`i_tf_sc_mat`, `i_tf_sup`) index into an array (`ucsc[i_tf_sc_mat - 1]`), which is why
seeding the jaxpr required integer dtypes — a float on that port is a trace error, which
is the same fact stated as a symptom. Three unregistered nodes add more of the same
(`BldgsSizes` reads `.tfcoil.i_tf_sup`; `PlantElectricProduction` takes `ireactor`
statically).

## 5. Recommended conversion order

Four bands. Within a band, order by the size of the invented fan-in.

### Band (a) — live incoherences. Do these first; they are bugs, not tidying

Not conversions at all in three of the five cases: the tree already has a slot, and the
defect is that the same switch is *also* answered somewhere else.

1. **`blktmodel`/`blkttype`/`ipowerflow` joint keys** (§4.3). Derive the arm index from
   the switch values instead of passing `blktmodel` through, and delete the illegal
   default of `2`. Concretely: `key1 = 0 if blktmodel == 1 else (2 if ipowerflow else 1)`,
   `key2 = 1 if blkttype in {1,2} else (0 if blktmodel else 2)`, both with
   `switches.get(..., <PROCESS's real default>)`. This alone makes a ported arm
   (`BlanketShieldPowerExponential`) reachable and stops one override selecting the wrong
   node. It also requires re-pinning two `test_switch_coverage._CAUSES_A_REFUSAL` rows,
   which currently assert the wrong behaviour.
2. **`i_tf_sup`** (§4.1). Five static kwargs must follow the registry's answer. The
   cheapest correct step is not conversion but *threading*: `machine_from_indat` already
   reads `i_tf_sup`; pass it to the five slots instead of letting each hardcode `1`. That
   turns a five-way incoherence into a one-line dependency and is strictly smaller than
   splitting five nodes into occupant pairs — which is band (b)'s job anyway.
3. **`ireactor`/`ipnet` on `costs.cost_of_electricity`** (§4.2). This is a node-existence
   condition, and its conversion is blocked on the tree regaining a way to say "no
   occupant" — step 4b removed all four `| None`s deliberately. Until that is decided, the
   minimum honest fix is to have `machine_from_indat` refuse `ireactor = 0` outright
   (an `UNPORTED` entry) rather than assemble a machine whose own constructor comment says
   it must not exist. Same for the three `.costs.ireactor` ports.
4. **`ipowerflow`** (§4.3), once (1) makes `ipowerflow = 0` reachable: thread it to the
   two `stellarator.*_wall_load*` slots the way (2) threads `i_tf_sup`.
5. **`i_plasma_pedestal`** (§4.4). Either stop reading it from the IN.DAT (it is forced to
   `0` by `st_init` on every stellarator run) or assert that the file agrees with what
   `st_init` will do. The first is more honest: `PROFILE_PARAMETERISATION` has only one
   reachable arm on this device.

"Convert to occupants" is the wrong description of this band. Four of the five are
*deletions of a duplicate answer*, and they are cheap.

### Band (b) — branches differ in reads. 23 slots, 79 invented edges

Conversion here means, concretely, per `model_tree_design.md` §4 case 1/2: a family base
(usually the existing class, made abstract or kept as the arm the reference run uses), one
occupant subclass per PROCESS value with **only that arm's `From`/`FromExactly`
declarations**, a registry line in `machine_from_indat`, and an `UNPORTED` entry for every
value with no occupant. The node keeps its slot path, so no consumer moves
(§3.2/§4's "promoting a dial to a family later is a *local* edit").

**(b1) — invented edges that change the block structure. Highest value; do these first.**

| slot | switch | what conversion changes |
|---|---|---|
| `stellarator.coils.winding_pack_intersect_inputs` | `i_tf_sc_mat` | drops `.tfcoil.j_tf_wp`, collapsing a 4-node driven block to 2 (§4.6) |
| `availability.cplife_avail` | `itart` (then `i_tf_sup`) | removes a `FixedPoint` that is the identity map (§4.7, §4.8) |
| `power.eta_turbine_step` | `i_thermal_electric_conversion` | removes a second identity `FixedPoint`, on a variable the objective depends on (§4.7) |
| `power.{delta_eta,temp_turbine_coolant_in,etath_liq,p_fw_div_heat_deposited_mw,p_fw_blkt_coolant_pump_mw,cryo_q_nuc,cryo_q_loads}_step` | as tabled | seven `FixedPoint` self-loops whose self-read is dead; each becomes an ordinary `ExplicitFunction` on the live arm |

**(b2) — large invented fan-in, no block change.** `physics.confinement_time.model` (12 —
but see band (d)), `costs.pf_magnet_cost` (11, `iohcl`),
`stellarator.{neutron_wall_load,radiated_wall_load_and_fraction}` (4 each, and the same
switch, so one family serves both), `power.cryo_loads` (4),
`power.component_thermal_powers` (4 across three switches),
`costs.tf_magnet_cost_superconducting` (3), `costs.cost_of_electricity` (3, `itart`),
`availability.electric_production` (3).

**(b3) — one- or two-edge cases.** `power.acpow` (`i_pf_energy_storage_source`: 1 edge
each way, complementary — `peakmva` vs `fmgdmw` — the cleanest and smallest conversion in
the whole survey and a good pilot), `stellarator.heating_and_radiation_power`,
`physics.plasma_composition`, `availability.avail` (`ibkt_life`).

The `i_pflux_fw_neutron` pair deserves a note: it kills `.first_wall.a_fw_total` on both
`stellarator.*_wall_load*` nodes, and `.first_wall.a_fw_total` is `stellarator.fw_area`'s
own output — so this switch invents an edge *out of a slot the factory already resolves*.
Converting it is the only place in band (b) where the invented edge crosses a registry
boundary.

### Band (c) — reads-identical; convert for §4's reasons, not this survey's

Nine slots invent nothing. Splitting them changes no edge and no number.

- **`ife` × 7** (§4.5), **`istore == 3`**, **`i_plasma_pedestal` on `ecrh_density_limit`**.
  These are `G` rows: the other arm raises inside the node body. **Conversion is the
  wrong answer here; relocation is the right one.** A refusal belongs in `UNPORTED`, where
  `machine_from_indat` states it once, at assembly, naming the switch — not seven times,
  at trace time, inside seven node bodies. The reason strings are already written and
  would move verbatim, exactly as step 4 moved 52 `Alternative(unported=)` strings.
  `i_plasma_pedestal` on `ecrh_density_limit` should simply be deleted: its container
  occupant already encodes it and cannot disagree.
- **`i_beta_fast_alpha`** — the textbook `model_tree_design.md` §4 "formula family":
  two published formulas over identical inputs. Convert on §4's extensibility argument
  (an enum family has no cheap escape when a third member needs a read the family cannot
  express), not because the graph gains anything. Already on step 6's list.
- **`i_blanket_type`, `secondary_cycle_liq` on `component_thermal_powers`,
  `i_pf_conductor` on `cryo_*`, `ipowerflow` on the wall-load pair, `i_pulsed_plant`** —
  `no`/`off` rows. Leave as enum-typed statics. They are nested *under* another switch on
  the same node (`i_blanket_type` only matters when `i_thermal_electric_conversion ∈
  {0,1,3}`; `ipowerflow` only when `i_pflux_fw_neutron == 2`), so they should become
  fields of the *outer* switch's occupants when band (b) converts those, and disappear
  from the arms where they mean nothing. That is a consequence of band (b), not separate
  work.

### Band (d) — high-arity families, needing a different treatment entirely

- **`i_confinement_time`, 49 reachable values, 1 live** (§4.9). One occupant per value is
  not the answer. The answer that matches the port's scope: a `ConfinementTime` family
  base carrying the shared pre-dispatch (the loss-power assembly, `kappa_ipb`, the unit
  conversions) and **one occupant for the scaling the port actually runs**, with
  `UNPORTED` entries generated for the other 48 from the existing `elif` chain. That takes
  32 declared reads down to ~20 for the run being modelled, costs one class, and leaves
  the family open in exactly the way §4 argues for. The 48 unwritten arms are not a debt
  this port owes: `unit_registry.md` scopes it to a stellarator.
- **`i_tf_sc_mat`, 9 values, 8 implemented** (§4.6). Counterpart classes exist — the eight
  `Jcrit*` nodes in `coils/coils.py` — and they are **the wrong shape**, for the reason
  `total_process.py`'s own docstring gives: the dispatch happens inside
  `winding_pack_curves`'s 200-point sampling loop, where `b_max` is an array and there is
  no scalar call site to bind them to. So conversion here is 8 occupants of a
  `winding_pack_intersect_inputs` family (or a nested `jcrit` sub-slot inside it), *not*
  registering the 8 existing nodes, and the docstring's refusal to register them stands.
  Only the ITER-Nb3Sn occupant needs writing to close §4.6's cycle.
- **`i_thermal_electric_conversion` (5) × `i_blanket_type` (2) × `secondary_cycle_liq`
  (2), on four slots.** The four `power.*` slots all route into the same two dispatchers
  (`calculate_plant_thermal_efficiency`, `_2`), so they are one family read four times,
  not four families. Convert once and let the four slots share it — otherwise the same
  5-way split gets written four times and the four copies can drift.

## 6. What this survey could not determine

1. **The reads-set of any arm the port refuses.** Method (4) needs both arms traceable.
   Sixteen rows are marked `?` or rest on a raise message: the seven `ife` sites,
   `istore == 3`, `i_plasma_pedestal != 0`, `i_tf_sc_mat == 9`, `i_confinement_time`
   0 and 25, `ireactor != 1`, `ipnet != 0`. To determine them: read PROCESS's own arm and
   record its reads by hand, the way the raise messages already do informally. That is
   audit work, not measurement, and it is the *only* thing standing between those rows and
   an `UNPORTED` entry with a real reads-set attached.
2. **Whether the two identity fixed points (§4.7) are numerically harmful.** That depends
   entirely on what `mda.default_drivers`/`sand` seed `.heat_transport.eta_turbine` and
   `.costs.cplife` with, and whether the seed is PROCESS's converged value or a default.
   Determining it: run `run_mda_harness.py` and read whether either appears in the
   ungrounded/driven accounting, then perturb the seed and see whether `.costs.coe` moves.
   One afternoon; not attempted here because it is a numerical question and this is a
   structural survey.
3. **Whether removing the `j_tf_wp` read changes any converged value** (§4.6). Only the
   topology was measured (reachability with that one edge deleted). The value question
   needs the SAND/MDA gate re-run with the read removed, which is a code change this
   survey is not allowed to make.
4. **Reads hidden behind a *data-dependent* condition.** The jaxpr method sees only
   Python-level branching. A read that survives as `jnp.where(cond, x, y)` counts as live
   even when `cond` is constant for this run — so every "dead" count here is a lower
   bound. `costs.energy_storage_cost` is the known instance:
   `.heat_transport.p_plant_electric_net_mw` is multiplied by a c2253 that
   `i_pulsed_plant = CONTINUOUS` has already set to `0.0`, so the edge is inert in value
   and in gradient but is reported live. Determining the true set needs a symbolic
   constant-propagation pass over the jaxpr, or a gradient probe.
5. **`physics.impurity_radiation_totals` was not traced at all** — three of its inputs
   (`.physics.radius_plasma_profile_norm`, `.nd_plasma_electron_profile`,
   `.temp_plasma_electron_profile_kev`) have no backing `DataStructure` field, so
   `_ground_truth` cannot seed them. It carries no switch, so this costs the survey
   nothing, but the same gap will block any future harness that wants per-node tracing.
6. **The 15 reads that are dead at every value of their node's switches** — the residue
   of §3's totals. Six sit on `power.component_thermal_powers` and
   `power.delta_eta_step` (`.heat_transport.eta_turbine`, `.etath_liq`,
   `.temp_turbine_coolant_in`, `.p_fw_div_heat_deposited_mw`, `.power.delta_eta`): the
   node reads the field, recomputes it internally, and **discards the result**, because
   the recomputed value is owned by one of the six `*Step` nodes instead. So
   `ComponentThermalPowers` is wired into four fixed-point loops it does not consume.
   One (`.physics.tauee_in` on `confinement_time.model`) is `USER_INPUT`'s read, and
   `USER_INPUT` is dead code in PROCESS. Nine are `costs.pf_magnet_cost`'s, from
   `n_cs_pf_coils = 0` (§4.10). None of these is a switch defect, and this survey makes
   no recommendation about them beyond recording that they exist and that band (b) will
   not fix them.
7. **Whether any of the 10 switch-valued ports (§4.11) is load-bearing for a gradient.**
   They are integers on differentiable ports; whether `jacfwd` ever traces through one was
   not checked.

## 7. One test this survey would like to leave behind

Measurement (3)/(4) is a test, not a script: *for every slot carrying a static switch,
no declared read may be dead at the value that slot actually holds*, with an explicit
allow-list of the ones that are (79 today). It fails the moment a conversion lands
(correctly — the allow-list shrinks) and the moment a new union node is registered. It is
strictly stronger than `switch_audit`'s value check and strictly stronger than
`test_switch_coverage.test_hardcoded_values_agree_with_the_reference_file`, which can only
see the six switches the IN.DAT happens to set.

On which: this survey checked **all 26** hardcoded switches against PROCESS's own
converged `DataStructure` for the reference run, not just the six the file sets. All 26
agree. Six differ from PROCESS's *bare* default — `i_confinement_time` (38 vs 34),
`i_p_coolant_pumping` (1 vs 2), `i_plasma_ignited` (1 vs 0), `i_thermal_electric_conversion`
(2 vs 0), `i_plasma_pedestal` (0 vs 1) and **`iohcl` (0 vs 1)**. The last is the one worth
the paragraph: `iohcl` is set by neither the IN.DAT nor `machine_from_indat`. Its `0` comes
from `process/models/stellarator/initialization.py:24`, i.e. from PROCESS's own stellarator
initialiser, so no test that compares against the input file can ever see it, and nothing
would notice if `st_init` changed. That is the concrete argument for making the converged
`DataStructure` — not the IN.DAT — the oracle for hardcoded switch values.
