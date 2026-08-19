---
kind: model-unit
status: draft
confidence: medium-high
---

## Update: port pass (see `stellarator_B_st_phys.py` / `test_stellarator_B_st_phys.py`)

This pass cross-checked the ~13 sub-computations this record already lists against what
has actually landed in `functional_process/models/` since it was written (most of it —
`physics_A_pure_formulas.py`, `physics_B_composition.py`, `confinement_time.py`,
`exhaust.py`, `radiation_power.py`, `fusion_reactions.py` — did not exist yet, or was
incomplete, when this record's original pass ran). Result: **almost every delegated
sub-call already has a ported node**; what remained genuinely new was the arithmetic
that lives directly in `st_phys`'s own body (never delegated), ported as eight tier-1
nodes below.

### Already-ported sub-calls (no new work — cross-reference only)

| `st_phys` call | ported node | file |
|---|---|---|
| `reactions.FusionReactionRate(...)` (`.deuterium_branching()`, `.calculate_fusion_rates()`, `.set_physics_variables()`) | `FusionRates` | `models/physics/fusion_reactions.py` |
| `reactions.set_fusion_powers(...)` | `SetFusionPowers` | `models/physics/fusion_reactions.py` |
| `self.beta.fast_alpha_beta(...)` (confirmed: `self.beta` *is* `self.plasma_beta`, `PlasmaBeta.fast_alpha_beta`, per open question 2 below) | `FastAlphaBeta` | `models/physics/physics_A_pure_formulas.py` |
| `rether(...)` | `IonElectronEquilibration` | `models/physics/physics_A_pure_formulas.py` |
| `physics_funcs.calculate_radiation_powers(...)` | `PlasmaRadiationPowers` | `models/physics/radiation_power.py` |
| `self.physics.calaculate_stored_thermal_energy(...)` (×2, electron and ion) | `ElectronThermalEnergy`, `IonThermalEnergy` | `models/physics/physics_A_pure_formulas.py` |
| `self.physics.confinement.calculate_confinement_time(...)` | `ConfinementTime` | `models/physics/confinement_time.py` |
| `self.physics.confinement.calculate_double_and_triple_product(...)` | `DoubleAndTripleProduct` | `models/physics/confinement_time.py` |
| `self.physics.calculate_total_plasma_heating_power(...)` | `TotalPlasmaHeatingPower` | `models/physics/physics_A_pure_formulas.py` |
| `self.physics.exhaust.calculate_radiation_fraction(...)` | `RadiationFraction` | `models/physics/exhaust.py` |
| `self.physics.phyaux(...)` | `AuxiliaryPhysicsQuantities` | `models/physics/physics_A_pure_formulas.py` |

This is direct evidence for `../../CLAUDE.md`'s "most of PROCESS is probably not
actually cyclic once dependencies are made explicit" — `st_phys` looked like it needed
a large tier-3 composition effort, and it turns out the composition's *leaves* were
already done by other forks working the physics-unit registry rows in parallel.

### Still audit-only (entangled with a not-yet-ported unit, or self-referential)

- **`self.physics.plasma_composition()`** — `physics_B_composition.md`'s own finding:
  `plasma_composition` reads and would need to write `.physics.first_call` in the same
  call (a self-loop), so it gets no node in that file either. Unchanged by this pass.
- **`self.plasma_profile.run()`** — `plasma_profiles.md`'s scope-gap finding: both
  branches call into `profiles.py`'s `neprofile.run()`/`teprofile.run()` classes and
  read fields those write; only downstream arithmetic is ported
  (`ProfileFactors`/`ParabolicGradientLengths`), not the orchestrating `run()` itself.
  Unchanged by this pass.
- **`self.neoclassics.calc_neoclassics()`** — `neoclassics.py` ports only two of the
  method's pieces (`ProfileValues`, `EffectiveThermalDiffusivity`); the 26-tuple
  orchestrator itself has no node. Unchanged by this pass.
- **`reactions.beam_fusion(...)`** (lines 2006-2047, the beam-active branch of the
  fusion totals) — `fusion_reactions.md`: blocked on the `scipy.integrate.quad` call
  (non-JAX-traceable, and only ~1e-6 accurate even in PROCESS's own hands — four orders
  outside tier-1 tolerance). Because this branch is entangled with an unported unit, the
  whole `if p_hcd_beam_injected_total_mw != 0 and not ignited: ... else: ...` block
  (lines 2004-2053, fusden_total/fusden_alpha_total/p_dt_total_mw derivation) is left
  audit-only rather than ported here. **Note for whoever ports `beam_fusion`**: the
  `else` branch alone (`fusden_total = fusden_plasma`, etc.) is trivial and could be
  folded into one node with the beam branch via ordinary Python `if` on
  `p_hcd_beam_injected_total_mw != 0`/`i_plasma_ignited` treated as static — same
  convention this pass used elsewhere — once `p_beam_alpha_mw`/`nd_beam_ions_out` have
  a producer.

### New findings from this pass

1. **Two more staleness channels than previously confirmed**, same shape as
   `b_plasma_surface_poloidal_average`. `.physics.beta_fast_alpha` is read at line 1931
   (inside the `beta_total_vol_avg` formula) *before* `self.beta.fast_alpha_beta(...)`
   computes and writes it at line 2079; `.physics.beta_beam` is read at line 1932
   *before* `reactions.beam_fusion(...)` conditionally writes it at line 2011. Both are
   read-before-write within the same `st_phys` call, exactly the Picard-truncation
   pattern this record already identified for the poloidal field — this record's
   earlier "how narrow" section ("the only confirmed channel...") undersold it; there
   are (at least) three, not one. Does not change the tier-2 recommendation for
   `power_at_ignition_point` (still narrow, still the same mechanism), but broadens what
   its residual should account for.

   **Superseded by item 3 below**: "read-before-write within the same `st_phys` call"
   is true textually (both reads sit above both writes in the source), but a follow-up
   dependency check (below) found this framing overstated the shape — unlike
   `power_at_ignition_point`'s genuine 2-call Picard truncation and the
   `b_plasma_surface_poloidal_average` self-loop this node is deliberately split away
   from, `beta_fast_alpha`/`beta_beam` are **not** a fixed point at all once the graph
   is assembled: neither producer's inputs feed back through this node's outputs. Kept
   here rather than rewritten, per this record's existing convention of layering
   findings rather than erasing superseded ones (see item 2's own "may not be a cycle
   at all" walking back a stronger claim).
2. **The poloidal-field "cycle" may not be a cycle at all**, once the graph is actually
   assembled. `calculate_poloidal_field_from_rotational_transform` (this pass's new
   node, ports lines 1971-1976) depends only on `rminor`/`b_plasma_toroidal_on_axis`/
   `rmajor`/`iotabar` — none of which depend on anything `st_phys` itself computes (in
   particular, not on `calculate_total_field`'s output or on `beta_total_vol_avg`/
   `e_plasma_beta`/`rho_star`). If that node is wired as the producer of
   `.physics.b_plasma_surface_poloidal_average` and `calculate_total_field` as its
   reader, a single acyclic pass already gives a self-consistent answer — no Picard
   iteration needed for *this* channel specifically. This does not resolve the whole
   question (the two `beta_fast_alpha`/`beta_beam` channels above are real
   producer-before-consumer orderings inside one call, and `plasma_composition`/
   `plasma_profile.run()` are still unaudited for self-reference of their own), but it
   is worth flagging precisely because `power_at_ignition_point`'s "call twice" trick
   may be truncating a fixed point that a correctly-ordered acyclic graph wouldn't need
   to iterate at all. Left for the later composition pass to confirm with the full
   graph — not resolved here, per this task's framing.
3. **`beta_fast_alpha`/`beta_beam`, resolved: both are Shape A (ordinary acyclic
   cross-node edges), not Shape B (genuine single-node self-loops)** — the distinction
   `_audit/next_steps.md` §5 draws between `Divertor`/`DivertorPlateMass` (confirmed not
   a real cycle once actually checked) and true `FixedPointFunction` cases. Checked
   with exactly that method — read the real producer's inputs and ask whether they
   depend on anything this node (the consumer) itself outputs:

   - **`beta_fast_alpha`**: sole owner is `FastAlphaBeta`
     (`models/physics/physics_A_pure_formulas.py`, already registered in
     `total_process.py`) — `fast_alpha_beta`'s inputs
     (`b_plasma_surface_poloidal_average`, `b_plasma_toroidal_on_axis`, several
     density/temperature averages, `pden_alpha_total_mw`, `pden_plasma_alpha_mw`,
     `f_plasma_fuel_deuterium`) never include `beta_total_vol_avg`/`e_plasma_beta`/
     `rho_star` (`StellaratorBetaAndRhoStar`'s own outputs) or `beta_beam`. No cycle.
     Verified directly: `to_graph([FastAlphaBeta(i_beta_fast_alpha=1),
     StellaratorBetaAndRhoStar()])` assembles with no error (two ordinary nodes, no
     "reads ... which it also owns"). **No `FixedPointFunction` was built — none is
     needed.** `StellaratorBetaAndRhoStar`'s existing `Input(lambda s:
     s.physics.beta_fast_alpha)` already points at `FastAlphaBeta`'s real `VarPath`;
     this is a plain registrable edge (`FastAlphaBeta` upstream of
     `StellaratorBetaAndRhoStar`), noted here for the later consolidation/wiring pass,
     not registered by this pass (out of boundary, `total_process.py` untouched).
   - **`beta_beam`**: sole owner in the PROCESS source is `reactions.beam_fusion(...)`
     (`process/models/physics/fusion_reactions.py`), which has **no ported node at
     all** yet (blocked on a non-JAX-traceable `scipy.integrate.quad` call — see the
     "still audit-only" table above and `fusion_reactions.md`). Its inputs
     (`beamfus0`, `betbm0`, `b_plasma_total`, `c_beam_total`,
     `nd_plasma_electrons_vol_avg`, `nd_plasma_fuel_ions_vol_avg`, `dlamie`,
     `e_beam_kev`, `f_plasma_fuel_deuterium`, `f_plasma_fuel_tritium`,
     `f_beam_tritium`, `temp_plasma_electron_density_weighted_kev`, `vol_plasma`,
     `n_charge_plasma_effective_mass_weighted_vol_avg`) likewise never include
     `beta_total_vol_avg`/`e_plasma_beta`/`rho_star`/`beta_beam` itself, so this is
     also Shape A by the same reasoning — but this can only be confirmed by reading
     `beam_fusion`'s signature, not by an actual `to_graph()` call (there is no node to
     assemble yet). **Different answer from `beta_fast_alpha` only in *readiness*, not
     in *shape***: once `beam_fusion` is ported (whoever picks that up — see the
     "still audit-only" table's note on it), it wires as an ordinary upstream producer
     of `.physics.beta_beam`, no fixed point required, no rework of this file needed.

   Net: **no `FixedPointFunction` was built anywhere in this pass.** The only genuine
   Shape-B self-loop this unit has found so far remains `plasma_composition`'s
   `.physics.first_call` self-reference (`physics_B_composition.md`'s finding,
   out of this file's boundary, being handled by a different agent) — `st_phys` itself,
   once `b_plasma_surface_poloidal_average` (item 2) and `beta_fast_alpha`/`beta_beam`
   (this item) are wired as ordinary edges, has no self-loop of its own.

### New tier-1 ports (this pass)

Eight nodes, all in `stellarator_B_st_phys.py`, all `ExplicitFunction`s, no internal
iteration, no calls into any other unit — the arithmetic that lives directly in
`st_phys`'s own body and was not already covered by another unit's node:

| node | ports (owns) | source lines |
|---|---|---|
| `TotalField` | `.physics.b_plasma_total` | 1916-1919 |
| `PoloidalFieldFromRotationalTransform` | `.physics.b_plasma_surface_poloidal_average` | 1971-1976 |
| `StellaratorBetaAndRhoStar` | `.physics.beta_total_vol_avg`, `.physics.e_plasma_beta`, `.physics.rho_star` | 1930-1968 |
| `FusionPowerTotalsMw` | `.physics.p_plasma_dt_mw`, `.physics.p_dhe3_total_mw`, `.physics.p_dd_total_mw` | 1991-2001 |
| `NeutronWallLoad` | `.physics.pflux_fw_neutron_mw` | 2095-2117 |
| `HeatingAndRadiationPower` | `.physics.p_plasma_rad_mw`, `.physics.psolradmw`, `.physics.p_plasma_separatrix_mw`, `.physics.p_fw_alpha_mw` | 2175-2220 |
| `RadiatedWallLoadAndFraction` | `.physics.pflux_fw_rad_mw`, `.constraints.pflux_fw_rad_max_mw`, `.physics.rad_fraction_total` | 2223-2257 |
| `ThermalEnergyTotals` | `.physics.eden_plasma_thermal_vol_avg`, `.physics.e_plasma_thermal_total` | 2282-2290 |

**`TotalField`/`PoloidalFieldFromRotationalTransform` split, not merged**: both touch
`.physics.b_plasma_surface_poloidal_average` (one reads it, one owns it) — merging them
into one node would make that node both read and own the same `VarPath`, which
`~/jaxgraph/CLAUDE.md`'s graph rules forbid (a self-loop). Kept as two ordinary nodes;
see "New findings" above for why the resulting edge may not even need to be a cycle.

**Switches** (`i_pflux_fw_neutron`, `heat_transport.ipowerflow`, `i_plasma_ignited`):
kept as static `int`s (`eqx.field(static=True)`, plain Python `if`/`elif` in the
function body), following the precedent already set by `ConfinementTime.i_rad_loss` and
`FastAlphaBeta.i_beta_fast_alpha` rather than mechanically applying
`naming_convention.md`'s default "differing reads-set -> split" rule. `NeutronWallLoad`
and `RadiatedWallLoadAndFraction` each get three harness contracts (one per branch) to
exercise all three reads-sets despite being one node — same idea as
`TestFastAlphaBetaIpdg89`/`TestFastAlphaBetaWard`.

**`powht` has no `VarPath`** (a plain Python local in the PROCESS source, never written
to `data`) — folded into `HeatingAndRadiationPower`'s body rather than exposed as a
separate port, per `_audit/schema.md`'s `local-intermediate` classification.
`pden_plasma_rad_mw * vol_plasma` is computed twice in the source (once inside `powht`,
once as `.physics.p_plasma_rad_mw`'s first write) — folded into one expression in the
port; `minor` note, not a correctness issue.

**JAX-difficulty**: all eight nodes are ordinary arithmetic plus `jnp.maximum` clamps
and static-switch `if`/`elif` — no blockers, no `workaround-known` items beyond the
already-flagged switch pattern. `RadiatedWallLoadAndFraction.rad_fraction_total`'s
denominator is unguarded (matches the PROCESS source, which would raise
`ZeroDivisionError` at the same point — not reachable at any sampled operating point, so
left as an open question rather than patched).

**Tests**: `test_stellarator_B_st_phys.py`, all `Tier1Contract`. None of the eight
expressions is separately callable in PROCESS (all are inline locals inside `st_phys`'s
~570-line body), so each reference is an independent from-source transcription rather
than a call into a PROCESS method — same situation
`stellarator_D_structure.md`'s `calculate_intercoil_mass_scaling_reference` was in.
`~/miniconda3/envs/process_port/bin/python -m pytest
functional_process/models/stellarator/test_stellarator_B_st_phys.py -q --fp-gradients`:
**325 passed**, no skips beyond the standard opt-in gate.

## Update: `StellaratorBetaAndStoredEnergy` — the registerable form, added this pass

`StellaratorBetaAndRhoStar` was never registered, because its `.physics.rho_star` output
collides with `DimensionlessPlasmaParameters`'s own — a **redundant duplicate write in
PROCESS itself** (`st_phys` and `outplas` compute it from the same inputs by the same
expression), not a modelling disagreement, so one of the two writes can simply be dropped.
Dropping the whole *node* to achieve that also cost `.physics.beta_total_vol_avg` and
`.physics.e_plasma_beta` their only producer, as collateral: only `rho_star` was ever in
conflict.

`StellaratorBetaAndStoredEnergy` is the same pure function with the third return value
discarded, owning exactly the two uncontested outputs. Registered in `COMMON`;
`StellaratorBetaAndRhoStar` stays written and unregistered.

Splitting `calculate_stellarator_beta_and_rho_star` in two was considered and rejected:
`rho_star`'s formula shares no sub-expression with the other two, so a split buys nothing
the discard does not (XLA eliminates the dead `sqrt`), and it would invalidate this
record's data-footprint row for a block PROCESS genuinely computes as one.

**Why it was worth doing.** `.physics.beta_total_vol_avg` is constraint 24's only
argument, and constraint 24 is one of the 14 active constraints of
`stellarator_helias.IN.DAT`. Without a producer it could not be assembled into an
`Optimise` at all, and `_audit/optimise_design.md` §2.4 had it recorded as permanently
INERT. See §10.7 there.

**And what it exposed.** Registering it pulled `FastAlphaBeta` into the differentiated set
for the first time (nothing downstream of `beta_fast_alpha` had previously fed a
condition), and its whole Jacobian row came back non-finite:
`jnp.sqrt(jnp.maximum(0.0, temp_sum_20 - 0.65))` in
`physics_A_pure_formulas.fast_alpha_beta` is the standard clamped-`sqrt` autodiff trap,
and the clamp is *active* on this run (`temp_sum_20 = 0.6449`). Fixed there with the
double `jnp.where`; value-identical. The general lesson is worth carrying: a gradient
defect only becomes visible when something downstream of it reads into a condition, so
closing a producer gap routinely exposes one.

## source
`process/models/stellarator/stellarator.py`, lines 1886-2456: `st_phys()`. Chunk 1B of
unit #1 (see `../../_audit/unit_registry.md`) — **priority chunk**, read in full.

## Scope correction found here (same class of miss as chunk 1A's `coils/` finding)

The original "methods in scope" list for shared files (registry rows 9-18, now
renumbered — see 1A's `coils/` correction above them) was built by grepping
`self.<attr>.<method>` call sites, which misses **module-level function imports**.
`st_phys` calls three such functions not previously in scope:

- `process/models/physics/fusion_reactions.py` — `FusionReactionRate` (class,
  `.deuterium_branching()`, `.calculate_fusion_rates()`, `.set_physics_variables()`
  methods), `beam_fusion()`, `set_fusion_powers()`. **Not in registry at all — new unit
  needed.**
- `process/models/physics/radiation_power.py` — `calculate_radiation_powers()`. **Not in
  registry at all — new unit needed.**
- `rether()` — imported from `process/models/physics/physics.py` (already registry unit
  #9, but `rether` is **missing from that unit's "methods in scope" list**; the list only
  has `plasma_composition`, `calaculate_stored_thermal_energy`,
  `calculate_effective_charge_ionisation_profiles`, `calculate_total_plasma_heating_power`,
  `outplas`, `phyaux`).

Registry updated below (new rows 9a, 9b under the shared-files table; `rether` added to
unit 9's method list) rather than deferred, per the precedent set by chunk 1A's `coils/`
correction.

## data footprint

**True external inputs** (read, never written, anywhere in this method) — ~55 fields
across `physics`, `current_drive`, `fwbs`, `first_wall`, `heat_transport`, `stellarator`,
`constraints`, `numerics`. Listed by group rather than one row each (a full one-row-per-
field table for a 570-line function isn't a useful reading of the schema — flagging this
as a practical schema note below); every one classifies as **explicit-arg** except the
three noted separately:

- `physics`: `aspect`, `rminor`, `rmajor`, `eps`, `kappa`, `kappa95`, `hfact`,
  `i_confinement_time`, `i_plasma_ignited`, `i_beta_fast_alpha`, `i_pflux_fw_neutron`,
  `vol_plasma`, `m_fuel_amu`, `m_ions_total_amu`, `dlamie`, `f_plasma_fuel_deuterium`,
  `f_plasma_fuel_tritium`, `f_alpha_electron`, `f_alpha_ion`,
  `f_p_alpha_plasma_deposited`, `pden_non_alpha_charged_mw`, `pden_plasma_neutron_mw`,
  `pden_plasma_alpha_mw`, `p_plasma_ohmic_mw`, `f_sync_reflect`, `tbeta`, `qstar`,
  `n_charge_plasma_effective_vol_avg`, `n_charge_plasma_effective_mass_weighted_vol_avg`,
  `plasma_current`, `burnup_in`, `tauratio`, `nd_plasma_alphas_thermal_vol_avg`,
  `nd_plasma_fuel_ions_vol_avg`, `nd_plasma_ions_total_vol_avg`,
  `temp_plasma_electron_density_weighted_kev`, `temp_plasma_ion_density_weighted_kev`,
  `temp_plasma_ion_vol_avg_kev`, `nd_plasma_electron_line`, `nd_plasma_electron_on_axis`.
- `current_drive`: `p_hcd_beam_injected_total_mw`, `c_beam_total`, `e_beam_kev`,
  `f_beam_tritium`, `p_hcd_injected_total_mw`.
- `fwbs`: `fhole`, `f_a_fw_outboard_hcd`, `f_ster_div_single`.
- `first_wall`: `a_fw_total`.
- `heat_transport`: `ipowerflow`.
- `stellarator`: `f_rad`, `iotabar`.
- `constraints`: `f_fw_rad_max`.
- `numerics`: `ixc` (read only for the `5 in ixc` guard — see JAX-difficulty flags, this
  is a data-dependent `raise`, not a value computation).

**The three inputs `power_at_ignition_point` overwrites on its proxy before calling
`st_phys` twice** — load-bearing for the tier-2 question, see the dedicated section below:

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.nd_plasma_electrons_vol_avg` | read only, never written in this method | explicit-arg | identical across both of `power_at_ignition_point`'s calls (nothing between the two calls re-sets it) |
| `.physics.b_plasma_toroidal_on_axis` | read only, never written in this method | explicit-arg | same — identical across both calls |
| `.physics.temp_plasma_electron_vol_avg_kev` | read only (line 2342, `calculate_double_and_triple_product`) | explicit-arg | **does not feed `p_plasma_loss_mw` or `pscalingmw`** within this method — see "why two calls" below |

**The order-dependency that actually explains most of what changes between the two
calls**:

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.b_plasma_surface_poloidal_average` | **read** at line 1918 (to compute `b_plasma_total`), then **written** at lines 1971-1976 (from `iotabar`/geometry) — in that order, within one execution | **implicit-io** | on call 1, the line-1918 read sees whatever was left on `self.data` from *before* `st_phys` was ever entered (stale); on call 2 (same proxy object, same `self.data`), it sees call 1's own line-1971 write. This is the mechanism, not a side detail. |
| `.physics.b_plasma_total` | write (1916-1919, from the stale-or-fresh field above) | — | differs between call 1 and call 2 purely because of the above; feeds `beta_total_vol_avg` (1930), `e_plasma_beta` (1944), `rho_star` (1953), `beam_fusion` (2017), `fast_alpha_beta` (2081), `calculate_radiation_powers` (2136) |

**Local intermediates** (computed and consumed within this same straight-line method —
would become ordinary Python locals in the pure port, not ports): everything else
`self.data.physics.*` this method writes and later reads back in the same call
(`beta_total_vol_avg`, `e_plasma_beta`, `rho_star`, `fusden_total`,
`fusden_alpha_total`, `p_plasma_dt_mw`, `pden_alpha_total_mw`, `p_alpha_total_mw`,
`p_plasma_rad_mw` and its components, `powht`/`p_plasma_separatrix_mw`,
`eden_plasma_*_thermal_vol_avg`, `e_plasma_*_thermal`, confinement outputs, etc.) — not
enumerated row by row per the same practical-schema note as the true-inputs list.

**The two outputs `power_at_ignition_point` actually harvests**:

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.p_plasma_loss_mw` | write, from `confinement_time_data.p_plasma_loss_mw` (line 2335) | — | causally depends on `nd_plasma_electrons_vol_avg` and `b_plasma_toroidal_on_axis` directly (both are arguments to `calculate_confinement_time`, lines 2296-2297) — these two proxy-set inputs are real, not dead |
| `.physics.pscalingmw` | write (lines 2354-2357, from `p_electron_transport_loss_mw` + `p_ion_transport_loss_mw`, themselves from the same `confinement_time_data`) | — | same causal chain as above |

## proposed signature(s)

**Not attempting one signature for the whole 570 lines** — this method is internally a
long acyclic composition of ~13 sub-computations (plasma composition, profile, field,
beta, heat, fusion, beam fusion, radiation, confinement, thermal energy ×2, neoclassics,
phyaux), each already close to tier-1 shape via the calls it makes. The honest port
structure is **tier-3 composition of ~13 tier-1 nodes**, not one tier-2 function — forcing
one signature here would hide exactly the acyclic structure the audit is supposed to
surface. Recommend: port each sub-computation as its own function (most already are,
via the `self.physics.*`/`reactions.*`/`physics_funcs.*` calls this method makes) and
`st_phys` itself becomes the composition/wiring, not a computation of its own — consistent
with how `naming_convention.md` treats composition.

The two fields `power_at_ignition_point` needs (`p_plasma_loss_mw`, `pscalingmw`) are
reachable from the true-inputs list above plus the `b_plasma_total` chain — **unblocks
`density_limits.md`'s provisional tier-2 signature**, though I'm not editing that file
(different unit's record); the "TBD" inputs it flagged as needed resolve to: the full
true-inputs list above, plus explicit handling of the `b_plasma_surface_poloidal_average`
staleness (see next section — this is the crux of that unit's tier-2 design, not a detail).

## tier signal

`st_phys` itself: **no internal iteration** — confirmed by full read, single
straight-line pass, no `while`/`scipy.optimize`/`fsolve`. In isolation this method is
tier-3 (acyclic composition of many sub-calls), not tier-2.

**But `power_at_ignition_point`'s "call it twice" pattern turns it into one anyway** —
this needs restating from what `density_limits.md` could see: it is not "an unrelated
convergence loop wrapped around a black box." Because both calls run on the *same* proxy
object (`self.data` persists across the two calls, not re-supplied), and `st_phys` reads
`b_plasma_surface_poloidal_average` before overwriting it, **calling `st_phys` twice on
one proxy is literally two steps of fixed-point (Picard) iteration**, with `st_phys`
itself playing the role of the update function. This reframes
`density_limits.md`'s "empirically tuned, not derived" finding more precisely: it's not
an arbitrary magic number, it's an *unverified 2-step truncation* of a real (if narrow)
fixed point.

**How narrow**: within this chunk's scope, the *only* confirmed channel by which call 2
can differ from call 1 is the `b_plasma_surface_poloidal_average` → `b_plasma_total` →
(`beta_total_vol_avg`, `e_plasma_beta`, `rho_star`, `beam_fusion`, `fast_alpha_beta`,
`calculate_radiation_powers`) chain — because the two directly proxy-set inputs
(`nd_plasma_electrons_vol_avg`, `b_plasma_toroidal_on_axis`) are read-only in this method
and unchanged between calls, and the third (`temp_plasma_electron_vol_avg_kev`) doesn't
reach either harvested output within this method. **Caveat, not resolved here**: the two
calls at the very top of `st_phys` — `self.physics.plasma_composition()` (registry unit
9) and `self.plasma_profile.run()` (registry unit 12, formerly 11) — are both still
`pending`; if either has its own self-referential state across calls, that's a second
channel this chunk's scope can't see. Recommend confirming with a direct numerical
experiment once ported (call the composed function 1×, 2×, 5× on identical inputs and
diff `p_plasma_loss_mw`/`pscalingmw`) rather than further hand-tracing — cheaper and more
conclusive than reading two more large files by hand. **Feeding straight into
`test_harness.md`'s tier-2 worked example** — recommend adding this as the concrete
numerical-experiment recommendation there.

## switches touched

`i_confinement_time`, `i_plasma_ignited`, `i_beta_fast_alpha`, `i_pflux_fw_neutron` — all
already rows in `../../core/solver/switches.md` from the pilot switch batch; this chunk
is exactly the file their "pending — needs model-unit audit" notes were waiting on.
Cross-reference, don't duplicate — see that file for per-switch reads-set detail once
updated as a follow-up (not done in this pass, out of directive scope).

One switch not previously catalogued: **`heat_transport.ipowerflow`** (line 2101, 2229)
— a plain int compared to `0`, gates the same three-way branch shape
(`i_pflux_fw_neutron==1` / `ipowerflow==0` / else) twice in this method for two different
quantities (`pflux_fw_neutron_mw`, `pflux_fw_rad_mw`). Same reads-set in both branches of
this *particular* three-way split (both else-branches read `fhole`+`a_fw_total` vs.
`fhole`+`f_a_fw_outboard_hcd`+`f_ster_div_single`+`a_fw_total` — genuinely differ) — split
candidate, new switch row needed in `switches.md`.

## calls into other models

- `self.physics.plasma_composition()` — registry unit 9 (`physics.py`), pending.
- `self.plasma_profile.run()` — registry unit 12 (`plasma_profiles.py`), pending.
- `st_heat(self, False, self.data)` — registry unit 5 (`heating.py`, already fully in
  scope as a whole stellarator-subpackage file), pending.
- `reactions.FusionReactionRate(...)`, `.deuterium_branching()`, `.calculate_fusion_rates()`,
  `.set_physics_variables()`, `reactions.beam_fusion()`, `reactions.set_fusion_powers()` —
  **new unit needed**, `physics/fusion_reactions.py` (not previously scoped).
- `self.beta.fast_alpha_beta(...)` — `self.beta` is a `PlasmaBeta` instance; not
  previously listed as an injected sub-model in `main.py`'s `Models.__init__` grep
  (that grep only checked `Stellarator.__init__`'s own injected attributes:
  availability/buildings/vacuum/costs/power/plasma_profile/hcpb/current_drive/physics/
  neoclassics/plasma_beta/plasma_bootstrap — `plasma_beta` *is* in that list, so `self.beta`
  is presumably `self.plasma_beta` under a different local name inside this class; **worth
  a quick confirm**, not re-derived here since it's a one-line check, not a new unit).
- `rether(...)` — `physics/physics.py`, registry unit 9, but **not in that unit's
  in-scope method list** — added below.
- `physics_funcs.calculate_radiation_powers(...)` — **new unit needed**,
  `physics/radiation_power.py` (not previously scoped).
- `self.physics.calaculate_stored_thermal_energy(...)` ×2, `self.physics.confinement.calculate_confinement_time(...)`,
  `self.physics.confinement.calculate_double_and_triple_product(...)`,
  `self.physics.calculate_total_plasma_heating_power(...)`,
  `self.physics.exhaust.calculate_radiation_fraction(...)`, `self.physics.phyaux(...)` —
  all already-scoped methods of registry units 9-11, pending.
- `self.neoclassics.calc_neoclassics()` — registry unit... (neoclassics is
  `Stellarator.__init__`'s injected `neoclassics`, i.e. `process/models/stellarator/neoclassics.py`
  — that's already-in-scope file #7, pending, not the shared-file list).
- `self.st_phys_output(...)` (conditional on `output`) — chunk 1G, **already drafted,
  confirmed purely reporting**, no action needed.

## JAX-difficulty flags

- **`if 5 in self.data.numerics.ixc: raise ProcessValueError(...)`** (lines 1922-1926) —
  `workaround-known` (`needs-lax-cond-or-where`, or more precisely: this is a *modelling
  precondition* — `ixc` membership is known at graph-build time, not per-evaluation, so
  this should become a build-time assertion when assembling the stellarator graph, not a
  runtime-traced branch at all). Also directly explains the docstring's "Raises
  ProcessValueError if beta is in ixc and istell>0" and the note already flagged in
  `switches.md`'s `istell` entry about a beta/`ixc` interaction — this is that interaction,
  now located precisely.
- **Three-way `if/elif/else` on `i_pflux_fw_neutron`/`ipowerflow`** (×2, see switches
  section) — ordinary switch-shaped branching, `workaround-known`, not `blocker`.
- **`max(0.00001e0, powht)` / `max(0.0e0, ...)` clamps** (lines 2182-2184, 2153-2158,
  2197, 2212-2214) — `minor`, same pattern already flagged in `density_limits.md` as
  evidence of upstream numerical instability being floored rather than fixed at the
  source; recurs here, worth noting the pattern is not a one-off.
- No CoolProp calls, no ragged/dynamic-shape numpy, no `copy.deepcopy` in this chunk
  itself (the deepcopy is `power_at_ignition_point`'s, already flagged there).

## open questions

1. **Practical schema note, not a content gap**: a full one-row-per-field data-footprint
   table doesn't scale to a 570-line, ~55-true-input method — grouped listing used
   instead, full rows reserved for the fields that are actually load-bearing for a
   downstream decision (the three proxy-set inputs, the staleness pair, the two harvested
   outputs). Suggest the schema explicitly bless this for large chunks rather than leaving
   it to each fork's judgment.
2. **`self.beta` vs `self.plasma_beta`** — almost certainly the same object under a
   shorthand local name set somewhere in `__init__`/`st_new_config`, not re-derived here
   (out of this chunk's line range); flagging so whoever audits chunk 1A/1C confirms it in
   one line rather than it being silently assumed.
3. Two new shared-file units needed (`fusion_reactions.py`, `radiation_power.py`) and one
   existing unit's method list needs `rether` added — done in the registry edit
   accompanying this record, not left open.

## Derivative-safe power laws (`safe_pow` / `safe_sqrt`)

2 square roots in this file have been rewritten from `x ** p` / `jnp.sqrt(x)` to
`models/safe_math.py`'s `safe_pow(x, p)` / `safe_sqrt(x)`.

**Why.** For `0 < p < 1` the function is continuous at `x == 0` and its derivative is
not: `d/dx x**p = p * x**(p-1) -> +inf`. JAX's JVP then returns `inf` along the
direction that perturbs `x` and `nan` (`inf * 0`) along every other, so the *value* is
right everywhere and the *Jacobian row* is poisoned. That is the defect class
`_audit/next_steps.md` §9 records; the most recent instance produced 46 non-finite
Jacobian cells and stalled a cold optimiser start at zero SQP steps, reported by the
solver as "the problem seems to be non-convex".

**Value identity, checked not asserted.** `safe_pow`/`safe_sqrt` dispatch on `x == 0`
and evaluate the identical expression otherwise, so every `x != 0` result is bit-for-bit
what it was, and the `x == 0` result is `0.0 ** p` / `sqrt(0.0)` -- again exactly what
the bare expression returns. Verified two ways: a hex-exact diff of every Tier-1
contract's output over every declared sample plus eight fresh fuzz draws (3655 points,
zero differing bits), and `run_mda_harness.py` unchanged at 492 agreements / 34
disagreements. PROCESS itself does not raise at `x == 0` here -- it is plain Python
`float.__pow__` / `numpy.sqrt`, both of which return `0.0` -- and the reference was
re-evaluated at each boundary point to confirm it returns the port's number.

**What changed is only the derivative at exactly `x == 0`**, which becomes `0` instead
of `inf`/`nan` -- the same convention JAX already uses at `jnp.maximum`'s kink.

`Tier1Contract.test_gradient_finite_at_zero` (`--fp-gradients`) now checks the whole
class automatically: it zeroes each differentiable argument in turn and requires a
finite Jacobian wherever the value is finite.
