---
kind: model-unit
status: draft
confidence: medium
---

## source
`process/models/stellarator/neoclassics.py` (841 lines, full file in scope). One class,
`Neoclassics(Model)`: an orchestrator (`init_neoclassics`, `calc_neoclassics`) over 13
small, cleanly-separated pure methods, plus `run`/`output` (both empty — `Neoclassics`
"doesn't need to be run", per its own docstrings; not part of the audit).

**Dispatched to look for a self-contained tier-2 unit; found none.** Despite the file's
size, there is no `fsolve`/Newton/bisection-style internal loop and no hardcoded-
iteration-count pattern anywhere in it (contrast `power_at_ignition_point`/`st_phys`'s
"call twice" Picard step). `init_neoclassics` and `calc_neoclassics` are both **straight-
line pipelines**: each of the ~13 sub-methods is called exactly once, in a fixed order,
writing to `self.data.neoclassics.*` and reading the previous method's write back —
tier-3-shaped (acyclic composition), not tier-2. Every individual sub-method, once its
`self.data` reads are made explicit, is tier-1: pure math, no internal solve, no calls
into another not-yet-ported unit. Given this, the immediate-porting practice applies at
the level of each of the 13 sub-methods, not the file as a whole — see below.

**No chunking needed.** Unlike `stellarator.py`, every method here already has a clean,
short, single-purpose boundary (13-100 lines each) — there is no ambiguous line range to
split. One record covers the whole file, following `density_limits.md`'s precedent (one
record, several proposed signatures).

## data footprint

Methods are grouped by call order in `init_neoclassics`/`calc_neoclassics`. `.neoclassics.roots`/
`.weights` (30-point Gauss-Laguerre quadrature, hardcoded array literals with no PROCESS
input) are treated as **fixed numerical-method constants**, not ports — same treatment as
stellarator_F's `coef`/`decay` tables. `.neoclassics.no_roots` is a derived property
(`len(roots)`), likewise not a port.

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.temp_plasma_electron_on_axis_kev`, `.temp_plasma_ion_on_axis_kev`, `.alphat`, `.nd_plasma_electron_on_axis`, `.f_plasma_fuel_deuterium`, `.nd_plasma_ions_on_axis`, `.nd_plasma_alphas_thermal_vol_avg`, `.alphan`, `.rminor` | read | explicit-arg | `init_profile_values_from_PROCESS`; `rho` itself is a plain argument (`0.6` at the one call site in `calc_neoclassics`, not read from `data`) |
| `.neoclassics.densities`, `.temperatures`, `.dr_densities`, `.dr_temperatures` | write | explicit-arg | return value of the above |
| `.neoclassics.kt` | write | explicit-arg | `neoclassics_calc_KT(roots, temperatures)` |
| `.neoclassics.densities`, `.temperatures` (read again) | read | implicit-io | `neoclassics_calc_nu` — reads the two fields `init_profile_values_from_PROCESS` just wrote, one call earlier in the same straight-line pipeline; genuinely order-dependent (not `local-intermediate`: this read is in a *different* method, only correct because `init_neoclassics` calls them in this exact order) |
| `.neoclassics.nu` | write | explicit-arg | return value |
| `.neoclassics.roots`, `.temperatures` (read again) | read | implicit-io | `neoclassics_calc_nu_star` |
| `.neoclassics.iota` | read | **implicit-io — external, unresolved** | never written anywhere in this file (grep confirmed: zero assignments to `self.data.neoclassics.iota`). Must be set by another unit before `Neoclassics.run()`'s callers reach this method — likely `stellarator.py`'s geometry setup (`st_geom`, unit 1C) or `preset_config.py` (unit #8, still pending). Flagging as an open question rather than guessing which. |
| `.neoclassics.nu_star` | write | explicit-arg | return value |
| `.neoclassics.nu_star_averaged` | write | explicit-arg | `neoclassics_calc_nu_star_fromT(iotain)` — note `iota` here **is** a plain argument (`iotain`, forwarded from `init_neoclassics`'s own argument), unlike the same-named `.neoclassics.iota` field read by `neoclassics_calc_nu_star`/`neoclassics_calc_D11_plateau` above/below. Two different values under confusingly identical names — flagged explicitly in the port's docstrings so this isn't lost. |
| `.neoclassics.vd` | write | explicit-arg | `neoclassics_calc_vd(roots, temperatures, rmajor, b_plasma_toroidal_on_axis)` — reads `.physics.rmajor`/`.physics.b_plasma_toroidal_on_axis` directly (explicit-arg), plus `.neoclassics.temperatures` again (implicit-io, same pipeline-order pattern as above) |
| `.neoclassics.d11_plateau` | write | explicit-arg | `neoclassics_calc_D11_plateau` — reads `.neoclassics.kt`/`.vd` (implicit-io) plus `.physics.rmajor`/`.neoclassics.iota` (the same unresolved external field as above) |
| `.neoclassics.d11_mono` | write | explicit-arg | `neoclassics_calc_d11_mono(eps_eff, vd, nu)` — `eps_eff` is `.stellarator_config.stella_config_epseff`, explicit-arg; `vd`/`nu` implicit-io |
| `.neoclassics.d111`, `.d112`, `.d113` | write | explicit-arg | `calc_integrated_radial_transport_coeffs(index=1/2/3)` — same function, called 3x with a different literal `index`; `index` is a precondition/static argument (which of three output fields you get), not a `VarPath` — see naming_convention.md's "switches are not ports" and the port's docstring |
| `.neoclassics.er` | read | **implicit-io — external, unresolved** | never written anywhere in this file, same situation as `.neoclassics.iota`. Read by both `neoclassics_calc_gamma_flux` and `neoclassics_calc_q_flux`. |
| `.neoclassics.gamma_flux` | write | explicit-arg | `neoclassics_calc_gamma_flux` — reads `.neoclassics.densities`/`temperatures`/`dr_densities`/`dr_temperatures`/`d111`/`d112` (implicit-io) plus `.er` (external, unresolved) |
| `.neoclassics.q_flux` | write | explicit-arg | `neoclassics_calc_q_flux` — same shape as `gamma_flux`, using `d112`/`d113` |
| `.physics.vol_plasma`, `.stellarator.f_st_rmajor`, `.impurity_radiation.radius_plasma_core_norm`, `.physics.rminor`, `.stellarator_config.stella_config_rminor_ref`, `.physics.a_plasma_surface`, `.physics.f_p_alpha_plasma_deposited`, `.physics.pden_alpha_total_mw`, `.physics.pden_plasma_core_rad_mw`, `.physics.nd_plasma_electron_on_axis`, `.physics.temp_plasma_electron_on_axis_kev`, `.physics.alphat`, `.physics.alphan` | read | explicit-arg | `st_calc_eff_chi` — **fully independent** of the `.neoclassics.*` pipeline above; no read or write of any `.neoclassics.*` field. Return value (`chi_PROCESS_e`) is a **local** in `calc_neoclassics`, never stored to `data` — same "invented VarPath" situation as `density_limits.py`'s `EcrhDensityLimit` outputs. Re-verified in the MDA triage (`_audit/next_steps.md` §8.1) with the full chain: assigned at `process/models/stellarator/neoclassics.py:396`, returned as the 22nd element of `calc_neoclassics`'s tuple (`neoclassics.py:425`), unpacked into a local at `process/models/stellarator/stellarator.py:2426`, and from there reaching only `st_phys_output` (`stellarator.py:2439`) which prints it via `po.ovarre` (`stellarator.py:2512-2513`). `process/data_structure/neoclassics_variables.py` has no `chi_*` field of any kind. |

`calc_neoclassics()` itself (the second orchestrator, called from outside this file —
not yet located, since no caller of `Neoclassics.calc_neoclassics`/`.init_neoclassics`
turned up in `stellarator.py`'s scope; likely `preset_config.py`, unit #8, still
`pending`) composes ~13 of the above fields into ~27 summary values (`q_PROCESS`,
`chi_neo_e`, `nu_star_e`, etc.) via more read-only arithmetic on `.neoclassics.*` and
`.physics.*`/`.impurity_radiation.*` fields, calling `st_calc_eff_chi` once more at the
end. Tier-3 composition, not audited field-by-field here (out of the "port tier-1/2
immediately" scope for this dispatch) — flagged as the natural next unit once `Neoclassics`'s
own 13 pieces are all ported.

## proposed signature(s)

**Ported and harness-tested** (both fully scalar-argument, both legacy samples from
existing PROCESS unit tests):

```python
def calculate_profile_values(
    rho, temp_plasma_electron_on_axis_kev, temp_plasma_ion_on_axis_kev, alphat,
    nd_plasma_electron_on_axis, f_plasma_fuel_deuterium, nd_plasma_ions_on_axis,
    nd_plasma_alphas_thermal_vol_avg, alphan, rminor,
) -> tuple[Array, Array, Array, Array]:  # (densities, temperatures, dr_densities, dr_temperatures), each (4,)
    ...

def calculate_effective_thermal_diffusivity(
    vol_plasma, f_st_rmajor, radius_plasma_core_norm, rminor, stella_config_rminor_ref,
    a_plasma_surface, f_p_alpha_plasma_deposited, pden_alpha_total_mw,
    pden_plasma_core_rad_mw, nd_plasma_electron_on_axis, temp_plasma_electron_on_axis_kev,
    alphat, alphan,
) -> float:  # chi_PROCESS_e
    ...
```

**Ported, tier-1, but not harness-tested** (array-valued arguments — see JAX-difficulty
flags and open questions):

```python
def calculate_kt(temperatures) -> Array  # (4, 30)
def calculate_collision_frequency(densities, temperatures) -> Array  # (4, 30)
def calculate_normalized_collision_frequency(temperatures, nu, iota, rmajor) -> Array  # (4, 30)
def calculate_normalized_collision_frequency_from_temperature(
    iota, temp_plasma_electron_vol_avg_kev, temp_plasma_ion_vol_avg_kev,
    nd_plasma_electrons_vol_avg, nd_plasma_fuel_ions_vol_avg, f_plasma_fuel_deuterium,
    nd_plasma_alphas_thermal_vol_avg, rmajor,
) -> Array  # (4,)
def calculate_drift_velocity(temperatures, rmajor, b_plasma_toroidal_on_axis) -> Array  # (4, 30)
def calculate_plateau_transport_coefficient(kt, vd, rmajor, iota) -> Array  # (4, 30)
def calculate_monoenergetic_transport_coefficient(eps_eff, vd, nu) -> Array  # (4, 30)
def calculate_integrated_radial_transport_coefficient(d11_mono, index) -> Array  # (4,), index in {1,2,3} static
def calculate_gamma_flux(densities, temperatures, dr_densities, dr_temperatures, d111, d112, er) -> Array  # (4,)
def calculate_q_flux(densities, temperatures, dr_densities, dr_temperatures, d112, d113, er) -> Array  # (4,)
```
All 10 are in `functional_process/models/stellarator/neoclassics.py`, faithfully ported
(value-for-value translation of the source, confirmed by hand against the listing above —
not independently value-checked against PROCESS, since the harness cannot yet do so).

## cottax node

**Written** for the two tested functions: `ProfileValues`, `EffectiveThermalDiffusivity`
(`ExplicitFunction`s), registered in `functional_process/total_process.py`.

```python
class ProfileValues(ExplicitFunction):
    densities = Output(lambda s: s.neoclassics.densities)
    temperatures = Output(lambda s: s.neoclassics.temperatures)
    dr_densities = Output(lambda s: s.neoclassics.dr_densities)
    dr_temperatures = Output(lambda s: s.neoclassics.dr_temperatures)

    def __call__(self, rho=Input(lambda s: s.neoclassics.r_eff), ..., rminor=Input(...)):
        return calculate_profile_values(rho, ..., rminor)


class EffectiveThermalDiffusivity(ExplicitFunction):
    chi_process_e = Output(lambda s: s.neoclassics.chi_process_e)  # invented VarPath, see data footprint

    def __call__(self, vol_plasma=Input(...), ..., alphan=Input(...)):
        return calculate_effective_thermal_diffusivity(vol_plasma, ..., alphan)
```

`ProfileValues.rho`'s `Input` maps it to `.neoclassics.r_eff` — a guess, not confirmed:
the only call site (`calc_neoclassics`) passes the literal `0.6`, not a `data` read, so
there is no existing PROCESS storage location for "the rho this call uses" to port. If
`preset_config.py`'s audit (unit #8) turns up a real `.neoclassics.r_eff`/similar field
this should feed from, correct this mapping then.

**Not wrapped** (see JAX-difficulty flags below): the other 10 functions. Adding
`ExplicitFunction`s for them now would put untested nodes in the graph.

## tier signal

- `init_profile_values_from_PROCESS`, `st_calc_eff_chi`: **tier 1**, harness-tested.
- The other 11 `neoclassics_calc_*`/`calc_integrated_radial_transport_coeffs` methods:
  **tier 1** by classification (no internal solve, no calls to another unit) but
  currently untestable by the harness (array-valued arguments — see below).
- `init_neoclassics`, `calc_neoclassics`: **tier 3** — acyclic composition of the above,
  not a solve. Not audited as a signature here (out of this dispatch's immediate-porting
  scope); the call order in `init_neoclassics` (lines 112-142 of the source) is the exact
  wiring a tier-3 pass over this unit would need to reproduce.

## switches touched
None. No `data.<area>.i_*` field read anywhere in this file.

## calls into other models
None — every method reads only `self.data.*`, no call to another `Model`'s method.

## JAX-difficulty flags
- **Harness gap, not a property of these functions**: `Tier1Contract`'s gradient-agreement
  test (`_harness/contracts.py`) differentiates one argument at a time via
  `float(sample.kwargs[name])` in `_jacobian`/`_reference_along` — this raises on any
  argument that isn't a scalar. 10 of this file's 13 pure functions take at least one
  length-4 (species) or length-30 (Gauss-Laguerre grid) array argument. This is expected
  to recur throughout the rest of the audit wherever a function is naturally vectorised
  over species or a quadrature grid (`fusion_reactions.py`/`radiation_power.py`, still
  pending, are candidates) — worth fixing once, in the harness, rather than working
  around per-unit. Two directions that would close it: (a) a per-component fuzz+
  differentiate scheme (flatten the array, treat each component as its own differentiable
  argument), or (b) a whole-array `jax.jacfwd`/finite-difference comparison instead of
  the current per-scalar-argument one. Not attempted here — out of this dispatch's scope
  (shared harness infrastructure, not this unit).
- `for j in range(4): for k in range(4): ...` double loops in `neoclassics_calc_nu`/
  `neoclassics_calc_nu_star_fromT` — `minor`, unrolls fine under `jax.jit` (4x4=16
  iterations, static range), ported as plain Python loops accumulating into a
  `.at[...].add(...)` update rather than in-place mutation.
- `if xk < 200.0: expxk = exp(-xk) else: 0.0` in `neoclassics_calc_nu_star_fromT` —
  `workaround-known`, ported as `jnp.where` (see `calculate_normalized_collision_
  frequency_from_temperature`'s docstring).

## open questions
1. **`.neoclassics.iota` and `.neoclassics.er` are read but never written anywhere in
   this file.** Both must come from another unit — candidates are `stellarator.py`'s
   `st_geom` (unit 1C, drafted) for `iota`, and an as-yet-unlocated producer for `er`
   (radial electric field; no obvious candidate seen yet in any audited chunk). Whoever
   next audits `preset_config.py` (unit #8) or reviews 1C should check for these writes
   specifically.
2. **`iota` is genuinely two different things under one name in this file.**
   `neoclassics_calc_nu_star`/`neoclassics_calc_D11_plateau` read `.neoclassics.iota`
   (the unresolved external field above); `neoclassics_calc_nu_star_fromT` instead takes
   `iota` as a plain forwarded argument (`iotain`, from `init_neoclassics`'s own
   signature) — these are not asserted to be the same value at call time, just
   confusingly homonymous. Ported as two distinctly-named parameters
   (`calculate_normalized_collision_frequency`'s `iota` vs.
   `calculate_normalized_collision_frequency_from_temperature`'s `iota`) to avoid
   silently conflating them; flagging in case they're supposed to be the same quantity
   and this is itself a latent inconsistency in the source.
3. **Who calls `Neoclassics.init_neoclassics`/`.calc_neoclassics` at all?** Not found
   anywhere in `stellarator.py`'s scope (grep for `self.neoclassics.` in
   `process/models/stellarator/stellarator.py` returns nothing) — despite `Neoclassics`
   being one of `Stellarator`'s injected sub-models per the unit registry's scope rule.
   Likely `preset_config.py` (unit #8, pending) or `initialization.py` (unit #6,
   pending) — flagging for whoever audits those next, since it also settles open
   question 1's `iota`/`er` producers if the caller sets them just beforehand.
4. **The harness gap in the JAX-difficulty flags is the actual blocker for finishing
   this unit's tier-1 half**, not any property of the remaining 10 functions. Recommend
   treating it as its own small piece of harness work rather than deferring it
   indefinitely, since every subsequent species-vectorised unit will hit the same wall.
