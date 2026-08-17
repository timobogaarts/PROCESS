---
kind: model-unit
status: draft
confidence: medium-high
---

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
