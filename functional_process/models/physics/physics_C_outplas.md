---
kind: model-unit
status: draft
confidence: high
---

## source
`process/models/physics/physics.py`, `Physics.outplas`, lines 1783-2877 (1095 lines —
by far the largest single method in the in-scope list). Registry unit #9, chunk C.

## scope note: 1095 lines, one computation

`outplas` is PROCESS's plasma-physics reporting routine — `self.outfile`/`self.mfile`
throughout, called only from `Physics.output()` (`physics.py:219-223`) and, in the
stellarator pipeline, `stellarator.py`'s `output=True` branch (step 5, per
`stellarator_A_orchestration.md`). Confirmed by direct search: `grep -n "self\.data\.[a-zA-Z_.]* = "`
over the full 1095-line body finds exactly **three** assignments, all in the first 40
lines (`physics.py:1790-1822`), and every one of the remaining ~1050 lines is either a
`process_output` call (`po.ovarre`/`po.oheadr`/`po.osubhd`/`po.oblnkl`/`po.ocmmnt`/
`po.oshead`) reading `self.data.*` for display, or a call into another injected
sub-model's own `.output()` method (`self.geometry.output()`, `self.current.output()`,
`self.fields.output()`, `self.beta.output_beta_information()`,
`self.output_temperature_density_profile_info()` — a same-class method **not** in this
unit's scope, confirmed separately to be a pure reporting shell (one write, to a
different-scope field, at `physics.py:3323`, well outside this method) —
`self.density_limit.output()`, `self.exhaust.output()`, `self.scrape_off_layer.output()`,
`self.plasma_transition.output()`, `self.plasma_bootstrap_current.output()`,
`self.dia_current.output()`). None of these are in scope for this unit (they belong to
other, already-scoped-elsewhere units, or are out of scope entirely) and none of them
compute anything this method could not do without — `outplas` never reads a value one
of them writes.

This is the same "reporting isn't quite inert" pattern already found in
`density_limits.py`'s `output()`, `coils/output.py`, and `stellarator_E3`'s
reporting tail — except here the inert part is unusually large (1095 lines) and the
non-inert part unusually small (3 lines), rather than the other way round.

**Four local Python variables computed purely for display/diagnostic logging**
(`p_plasma_imbalance_mw`, `p_reactor_imbalance_mw`, `p_electric_imbalance`,
`p_plasma_imbalance_mw` — power-balance in/out sums, each compared to `0.1` and logged
via `logger.error` if exceeded) are **not** ported: none is written to `self.data`, none
is read anywhere else in this function or consumed by any other unit, and each is a
straight-line sum/difference of already-computed `self.data` fields with no reuse
potential — the same judgement call `coils/output.py`'s record already made for its own
"several trivial inline arithmetic expressions for display only, none worth
extracting."

## data footprint

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.dlamie`, `.vol_plasma`, `.rmajor`, `.b_plasma_toroidal_on_axis`, `.eps`, `.nd_plasma_electron_line`, `.kappa`, `.e_plasma_beta`, `.plasma_current`, `.m_ions_total_amu` | read | explicit-arg | all ten read once each, straight-line, no branch |
| `.physics.nu_star`, `.rho_star`, `.beta_mcdonald` | write | explicit-arg | the only writes in the entire 1095-line method |

Everything else this method touches (several hundred `self.data.*` reads across the
remaining ~1050 lines) is read-only, for display purposes, and out of scope — not
tabulated individually; the audit's judgement is that none of it is a computation to
port, confirmed by the write-count grep above, not by inspection of a sample.

## proposed signature(s)
```python
def calculate_dimensionless_plasma_parameters(
    dlamie, vol_plasma, rmajor, b_plasma_toroidal_on_axis, eps,
    nd_plasma_electron_line, kappa, e_plasma_beta, plasma_current, m_ions_total_amu,
) -> tuple[float, float, float]: ...
```
Implemented in `physics_C_outplas.py`.

## cottax node
`DimensionlessPlasmaParameters` (`ExplicitFunction`) in `physics_C_outplas.py`. No other
node is proposed for this unit — the reporting body has nothing left to wrap.

## tier signal
1 — explicit pure function, three unrelated dimensionless-parameter formulas
(`nu_star`, `rho_star`, `beta_mcdonald`) sharing an argument list, no branches, no calls.

## switches touched
None inside the ported computation. `outplas`'s *reporting* body branches on
`.stellarator.istell` (whether to call `self.fields.output()`/`self.density_limit.output()`)
and `.divertor.n_divertors` (single vs. double null display) — both purely display-path
decisions, out of scope for this record since nothing downstream of them is ported.

## calls into other models
None, in the ported computation. The surrounding (unported) reporting body calls
`self.geometry.output()`, `self.current.output()`, `self.fields.output()`,
`self.beta.output_beta_information()`, `self.output_temperature_density_profile_info()`,
`self.density_limit.output()`, `self.exhaust.output()`, `self.scrape_off_layer.output()`,
`self.plasma_transition.output()`, `self.plasma_bootstrap_current.output()`,
`self.dia_current.output()` — all reporting-only calls on other units' own models, none
of which this unit's scope covers or needs.

## JAX-difficulty flags
None. `nu_star`'s `jnp.sqrt(eps)` and `rho_star`'s `jnp.sqrt(...)` both take physically
non-negative arguments in every real PROCESS configuration (`eps` is an inverse aspect
ratio; the `rho_star` radicand is a product of positive densities/temperatures/masses) —
no domain guard added, consistent with `rether`'s treatment in chunk A.

## open questions
None.
