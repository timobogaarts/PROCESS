---
kind: model-unit
status: draft
confidence: high
---

## source

`process/models/physics/physics.py`, five separate ranges, registry unit #9, chunk A:

- `rether` (module-level function, not a method): lines 103-152.
- `Physics.phyaux` (`@staticmethod`, `@nb.njit(cache=True)`): lines 1493-1602.
- `Physics.calculate_total_plasma_heating_power` (`@staticmethod`): lines 3538-3568.
- `Physics.calaculate_stored_thermal_energy` (`@staticmethod`, PROCESS's own
  misspelling — matches source, see `unit_registry.md` row #9): lines 3573-3609.
- `PlasmaBeta.fast_alpha_beta` (`@staticmethod`, `@nb.njit(cache=True)`): lines
  4265-4392.

## scope note: why these five are one chunk

`physics.py`'s in-scope method list (registry row #9) is not a contiguous line range —
unlike `stellarator.py`'s chunks 1A-1G, which split one 2600-line file by *position*,
these eight methods are scattered across a 6931-line file and belong to two different
classes (`Physics`, `PlasmaBeta`). Grouping by *position* would produce meaningless
boundaries here; grouping by *tier characteristic* does not. This chunk is every
in-scope method that is **already a fully pure function in the source** — no
`self.data` access anywhere in the body, explicit arguments in, explicit values out —
confirmed by reading each one, not assumed from the `@staticmethod` decorator (three of
the five are `@staticmethod`, one is a bare module function, and `phyaux`/`fast_alpha_beta`
are additionally `@nb.njit`, i.e. PROCESS's own numba compiler already requires them to
be free of Python-object/`self` access). `PlasmaBeta.fast_alpha_beta` sits on a
different class from the other four, but it is put here rather than a separate file
because its *port* is identical in kind to the other four's — no entanglement, no audit
judgment left to make beyond translating `if`/`min`/`max` to their traced equivalents.

The remaining three in-scope methods (`plasma_composition`,
`calculate_effective_charge_ionisation_profiles`, `outplas`) are **not** in this shape —
see `physics_B_composition.md` and `physics_C_outplas.md`.

## data footprint

All five functions: **zero `self.data` reads, zero `self.data` writes** inside the
function bodies themselves — every value is an explicit parameter or local. The table
below is instead the *call-site* footprint: which `VarPath` each parameter is bound to
where PROCESS calls the function, and which `VarPath` each return value is written to.
This is what the `cottax` node classes below encode as `Input`/`Output`; the audit
schema's five read/write classifications (`explicit-arg` etc.) do not really apply here
since the function itself performs no `self.data` access — every row is `explicit-arg`
by construction, both call sites (`stellarator.py` and `physics.py`'s own `Physics.run()`)
bind the same fields, confirmed by grepping both.

| function | VarPath (arg binding / return) | read/write | note |
|---|---|---|---|
| `rether` | `.physics.alphan`, `.physics.alphat`, `.physics.nd_plasma_electrons_vol_avg`, `.physics.dlamie`, `.physics.temp_plasma_electron_vol_avg_kev`, `.physics.temp_plasma_ion_vol_avg_kev`, `.physics.n_charge_plasma_effective_mass_weighted_vol_avg` | read | `stellarator.py:2121-2128` |
| `rether` | `.physics.pden_ion_electron_equilibration_mw` | write | `stellarator.py:2121` |
| `phyaux` | `.physics.aspect`, `.physics.nd_plasma_fuel_ions_vol_avg`, `.physics.fusden_total`, `.physics.fusden_alpha_total`, `.physics.plasma_current`, `.physics.nd_plasma_alphas_thermal_vol_avg`, `.physics.t_energy_confinement`, `.physics.vol_plasma`, `.physics.burnup_in`, `.physics.tauratio` | read | `stellarator.py:2377-2398` |
| `phyaux` | `sbar` | — | **not a `VarPath`**: the stellarator caller passes the Python literal `1.0e0` (`stellarator.py:2378`), never a `data` field. Kept as a plain field on the node (default `1.0`), not hardcoded, so it stays visible — see "open questions". |
| `phyaux` | `.physics.burnup`, `.physics.figmer`, `.physics.molflow_plasma_fuelling_required`, `.physics.rndfuel`, `.physics.t_alpha_confinement`, `.physics.f_t_alpha_energy_confinement` | write | `stellarator.py:2384-2398` |
| `phyaux` | `.physics.fusrat` | write, **but discarded at this call site** | `stellarator.py:2390` assigns the third return value to a bare local `_fusrat`, never to `data`. `physics.py:961` (unit #22, tokamak, out of stellarator scope) *does* write it to `.physics.fusrat` — the field exists and is real, just unproduced on this pipeline. Not a `redundant-duplicate-write` (only one call site in scope writes anything at all); flagged as a dropped output instead. |
| `calculate_total_plasma_heating_power` | `.physics.f_p_alpha_plasma_deposited`, `.physics.p_alpha_total_mw`, `.physics.p_non_alpha_charged_mw`, `.physics.p_plasma_ohmic_mw`, `.current_drive.p_hcd_injected_total_mw` | read | `stellarator.py:2358-2364` |
| `calculate_total_plasma_heating_power` | `.physics.p_plasma_heating_total_mw` | write | `stellarator.py:2360` |
| `calaculate_stored_thermal_energy` (electron binding) | `.physics.vol_plasma`, `.physics.nd_plasma_electrons_vol_avg`, `.physics.temp_plasma_electron_density_weighted_kev` | read | `stellarator.py:2264-2270` |
| `calaculate_stored_thermal_energy` (electron binding) | `.physics.eden_plasma_electrons_thermal_vol_avg`, `.physics.e_plasma_electrons_thermal` | write | `stellarator.py:2264-2266` |
| `calaculate_stored_thermal_energy` (ion binding) | `.physics.vol_plasma`, `.physics.nd_plasma_ions_total_vol_avg`, `.physics.temp_plasma_ion_density_weighted_kev` | read | `stellarator.py:2273-2279` |
| `calaculate_stored_thermal_energy` (ion binding) | `.physics.eden_plasma_ions_thermal_vol_avg`, `.physics.e_plasma_ions_thermal` | write | `stellarator.py:2273-2275` |
| `fast_alpha_beta` | `.physics.b_plasma_surface_poloidal_average`, `.physics.b_plasma_toroidal_on_axis`, `.physics.nd_plasma_electrons_vol_avg`, `.physics.nd_plasma_fuel_ions_vol_avg`, `.physics.nd_plasma_ions_total_vol_avg`, `.physics.temp_plasma_electron_density_weighted_kev`, `.physics.temp_plasma_ion_density_weighted_kev`, `.physics.pden_alpha_total_mw`, `.physics.pden_plasma_alpha_mw`, `.physics.i_beta_fast_alpha`, `.physics.f_plasma_fuel_deuterium` | read | `stellarator.py:2079-2090` |
| `fast_alpha_beta` | `.physics.beta_fast_alpha` | write | `stellarator.py:2079` |

Two locally-computed (not `data`-derived) quantities inside `fast_alpha_beta`'s body,
`beta_thermal`/`fact`/`fact2`, are ordinary `local-intermediate`s (straight-line, no
branch between computation and use) and are not separately tabulated.

## proposed signature(s)

```python
def rether(alphan, alphat, nd_plasma_electrons_vol_avg, dlamie, te,
           temp_plasma_ion_vol_avg_kev, n_charge_plasma_effective_mass_weighted_vol_avg): ...

def phyaux(aspect, nd_plasma_fuel_ions_vol_avg, fusden_total, fusden_alpha_total,
           plasma_current, sbar, nd_plasma_alphas_thermal_vol_avg, t_energy_confinement,
           vol_plasma, burnup_in, tauratio) -> tuple[float, ...]: ...

def calculate_total_plasma_heating_power(f_p_alpha_plasma_deposited, p_alpha_total_mw,
           p_non_alpha_charged_mw, p_plasma_ohmic_mw, p_hcd_injected_total_mw) -> float: ...

def calaculate_stored_thermal_energy(vol_plasma, nd_plasma_vol_avg,
           temp_plasma_density_weighted_vol_avg_kev) -> tuple[float, float]: ...

def fast_alpha_beta(b_plasma_poloidal_average, b_plasma_toroidal_on_axis,
           nd_plasma_electrons_vol_avg, nd_plasma_fuel_ions_vol_avg,
           nd_plasma_ions_total_vol_avg, temp_plasma_electron_density_weighted_kev,
           temp_plasma_ion_density_weighted_kev, pden_alpha_total_mw,
           pden_plasma_alpha_mw, i_beta_fast_alpha, f_plasma_fuel_deuterium) -> float: ...
```
Argument order and spelling copied verbatim from the source signatures. Implemented in
`physics_A_pure_formulas.py`.

## cottax node

Six node classes in `physics_A_pure_formulas.py`: `IonElectronEquilibration`,
`AuxiliaryPhysicsQuantities`, `TotalPlasmaHeatingPower`, `ElectronThermalEnergy`,
`IonThermalEnergy`, `FastAlphaBeta`. Two nodes (`ElectronThermalEnergy`/
`IonThermalEnergy`) wrap the *same* underlying function at two different `VarPath`
bindings — the source function is species-agnostic, but a `cottax` node is named by
what it owns, and PROCESS calls it twice with two disjoint output sets, so one node
class cannot cover both without inventing a shared "species" dimension PROCESS's own
call sites don't have.

## tier signal
1 (explicit pure function) for all five — no internal iteration, no calls into any
other unit, no `self.data` access in the body at all (the strongest form of tier-1 in
this codebase: nothing to promote, only to translate).

## switches touched
- `i_beta_fast_alpha` (`.physics.i_beta_fast_alpha`, values `0`/`1`): kept as a
  **static kwarg** on `FastAlphaBeta` (`traceability_policy.md`'s "exception: static
  kwarg" case) — both branches read exactly the same six variables
  (`nd_plasma_fuel_ions_vol_avg`, `nd_plasma_electrons_vol_avg`,
  `temp_plasma_electron_density_weighted_kev`, `temp_plasma_ion_density_weighted_kev`,
  plus the two already-computed `beta_thermal`/`fact2`) and differ only in two
  coefficients and whether a `sqrt`/`max(0, ...)` guard is applied — a genuine
  solver-method choice, not alternate physics with a different input set. See
  `core/solver/switches.md` for the registry row (added by the coordinating session,
  not this record).

## calls into other models
None. All five functions are self-contained — no calls to any other `Model`, no calls
into `impurity_radiation.py`/`fusion_reactions.py`/etc.

## JAX-difficulty flags
- `needs-lax-cond-or-where` (`phyaux`, both `if`/`else` selections; `fast_alpha_beta`,
  the `f_plasma_fuel_deuterium < 1.0` selection) — severity **workaround-known**,
  resolved with `jnp.where`.
- **Denominator-guard-required** (new flag, not yet in `traceability_policy.md`'s
  list): both `phyaux` and `fast_alpha_beta` have a `jnp.where` whose *unselected*
  branch would divide by a value that is exactly zero precisely because it was not
  selected (`fusden_alpha_total == 0` selects `t_alpha_confinement = 0` in `phyaux`;
  `pden_plasma_alpha_mw == 0` on the "negligible alpha production" branch in
  `fast_alpha_beta`, per PROCESS's own comment). Severity **workaround-known**, but
  worth flagging as its own category — `_audit/test_harness.md`'s pilot retrospective
  found this exact failure mode (`jnp.where` computing a NaN gradient through a
  correct-looking value) by injecting a `stop_gradient` bug; here it is naturally
  present in the source formula rather than injected, so it had to be actively guarded
  rather than discovered by the harness. Recommend adding this as a named category to
  `traceability_policy.md`'s "Dynamic shape / mutation idioms" section, since it will
  recur (`plasma_composition`, chunk B of this same file, has at least one more
  instance — see that record).
- `phyaux`'s `molflow_plasma_fuelling_required = rndfuel / burnup` can be a genuine
  `0/0` at a legitimate PROCESS input (`burnup_in == 0` **and**
  `nd_plasma_alphas_thermal_vol_avg == 0`, e.g. a cold-start/no-fusion point) — PROCESS
  itself raises `ZeroDivisionError` there (plain Python division), which is exactly
  `test_harness.md`'s domain-guard shape: the port returns `nan` instead of raising.
  Declared as `reference_domain_errors = (ZeroDivisionError,)` in the test case, sample
  `phyaux-no-alphas`. Severity **minor** (already the harness's standard pattern).
- Minor: `fast_alpha_beta`'s `jnp.sqrt(jnp.maximum(0.0, ...))` has a formally
  infinite gradient exactly at the clamp boundary (`temp_sum_20 == 0.65`) — inherent
  to the source formula (PROCESS's own `np.sqrt(max(0.0, ...))` has the identical
  property), not introduced by porting. Not expected to be hit by fuzzing (a
  probability-zero boundary), noted for completeness.

## open questions
1. `phyaux`'s `sbar` parameter is a real argument in the general signature but PROCESS's
   only in-scope caller (`stellarator.py:2378`) always passes the literal `1.0`, never a
   `data` field. Kept as a plain default-valued field on `AuxiliaryPhysicsQuantities`
   rather than folding it into the body as a hardcoded constant, so a future tokamak
   port (unit #22, out of scope here, whose own `physics.py:962` call site also appears
   to pass `1.0e0` based on a spot check — not verified in depth, out of scope) can
   override it if it ever turns out not to be a universal constant. Not resolved further
   here.
2. `phyaux`'s `fusrat` output has no producer role in the stellarator graph (discarded
   at the only in-scope call site) but is a real `DataStructure` field written by the
   tokamak path. `AuxiliaryPhysicsQuantities.fusrat` is declared anyway, matching the
   function's true output set — whether `total_process.py`'s stellarator `GRAPH` should
   wire it to anything is the coordinating session's call, not this record's.


## Update: `fast_alpha_beta`'s clamped square root — a real autodiff defect, fixed

`i_beta_fast_alpha != 0` (the Ward branch) computed
`jnp.sqrt(jnp.maximum(0.0, temp_sum_20 - 0.65))`. That is value-correct and returns `nan`
from `jax.jacfwd` **on the clamped branch**: `sqrt` has an infinite derivative at zero and
`inf * 0` is `nan`. Exactly the trap `_audit/next_steps.md` §9 records for
`costs.py:2874-2888`'s clamped net-electric-power square root, and the fix is the same
standard **double `jnp.where`** — the inner one keeps a finite argument out of `sqrt`'s
differentiation rule, the outer one selects the value.

**Not hypothetical: the clamp is active on the reference run.** `temp_sum_20` is `0.6449`
against the `0.65` threshold, so `fast_alpha_beta`'s derivative was `nan` every time it
was differentiated. It went unnoticed because nothing downstream of
`.physics.beta_fast_alpha` fed a condition until `StellaratorBetaAndStoredEnergy` was
registered and constraint 24 started reading `.physics.beta_total_vol_avg`; at that point
it was the only non-finite row of the SAND Jacobian, and it made the whole SQP
unsolvable. See `_audit/optimise_design.md` §10.5b.

Every value test passed throughout, before and after — which is the point. This is the
second instance in this project of "a `jnp.maximum` guard in front of a function with an
unbounded derivative *at the guard point*", and it should now be treated as a known
pattern to grep for rather than a surprise.
