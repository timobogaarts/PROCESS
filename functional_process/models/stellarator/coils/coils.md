---
kind: model-unit
status: reviewed
confidence: high
---

**Ported (partial).** `coils.py` / `test_coils.py`: `j_crit_cable_from_fraction` and
`bmax_from_awp`, both tier-1, tests passing (legacy + fuzz). `jcrit_from_material` and
`intersect` are **not** ported — see below and open questions.

## source
`process/models/stellarator/coils/coils.py` (303 lines, full file in scope). 4 module-
level functions: `j_crit_cable_from_fraction`, `jcrit_from_material`, `intersect`,
`bmax_from_awp`. Called only from `coils/calculate.py` (registry unit #9, `st_coil`'s
winding-pack solve loop) — no calls in the other direction, and no calls between these
four functions and `forces.py`/`mass.py`/`quench.py` (grepped: none of those three files
import from `coils.py` or vice versa; each of the four files in this batch is called
directly by unit #9, not by each other).

## data footprint

| VarPath | read/write | classification | note |
|---|---|---|---|
| — | — | — | `j_crit_cable_from_fraction` takes no `data` at all — already pure in the source. |
| `.stellarator_config.stella_config_a1` | read | explicit-arg (via `data` back-door) | `bmax_from_awp` |
| `.stellarator_config.stella_config_a2` | read | explicit-arg (via `data` back-door) | same |

`jcrit_from_material`'s and `intersect`'s footprints are call-site-dependent (see below)
— not tabulated here since neither is ported this pass.

## proposed signature(s)

Ported, tier-1, as written (see `coils.py`):
```python
def j_crit_cable_from_fraction(j_crit_sc, f_tf_conductor_copper, f_he) -> float:
    ...

def bmax_from_awp(
    wp_width_radial, current, n_tf_coils, r_coil_major, r_coil_minor,
    stella_config_a1, stella_config_a2,
) -> float:
    ...
```

**Not ported — `jcrit_from_material`.** A genuine 8-way switch on `i_tf_sc_mat`
(`process/data_structure/tfcoil_variables.py:246`, default 1), each branch calling a
different function in `process.models.superconductors` (`itersc`, `bi2212`,
`jcrit_nbti`, `western_superconducting_nb3sn`, `jcrit_rebco`, `gl_nbti`, `gl_rebco`) with
a genuinely different reads-set — branch 4 reads `b_crit_sc`/`t_crit_sc` that no other
branch touches, branch 7 reads `b_crit_upper_nbti`/`t_crit_nbti` that no other branch
touches, branches 1/3/5 use fixed `bc20m`/`tc0m` literals instead of reading either.
Per `traceability_policy.md`'s split-by-default: **split**, into up to 8 functions. Not
attempted here because every branch's actual arithmetic lives in
`process/models/superconductors.py`, which is not yet a registry unit — porting a branch
now would mean porting an unaudited module's formula sight-unseen. **Recommend adding
`process/models/superconductors.py` as a new registry unit** (it's shared with tokamak TF
coil code too, per a quick grep of its importers — scoping that precisely is its own small
task, not done here).

**Not ported — `intersect`.** A generic Newton-Raphson-style root-finder over two
tabulated `(x, y)` curves: fixed 100-iteration cap, early `break` on
`abs(y01 - y02) < epsy`, `np.interp` calls each iteration. Self-contained (no calls into
other models) — a real tier-2 candidate by this audit's own "self-contained internal
solve" bar. Not ported this pass because its unknowns are whole arrays (`x1, y1, x2, y2`
are the tabulated curves, `xin` the initial guess), which doesn't fit the harness's
scalar-kwarg sample/fuzz machinery (`_harness/sampling.py`) without a real design pass —
flagged as a priority item for whoever picks this up next, not rushed into a fragile
`ImplicitFunction`/`FixedPointFunction` wrap. See open questions for the residual shape
this would need.

## tier signal
- `j_crit_cable_from_fraction`: **tier 1** — pure, no `data`, no branch.
- `bmax_from_awp`: **tier 1** — pure once the 2-field `data` back-door is closed (same
  shape as `st_sudo_density_limit` in `density_limits.py`).
- `jcrit_from_material`: tier 1 *per branch*, once split — see above, blocked on
  `process.models.superconductors` being audited.
- `intersect`: **tier 2**, self-contained, not yet ported (array-valued unknowns — see
  above).

## switches touched
- `i_tf_sc_mat` (`.tfcoil.i_tf_sc_mat`) — **new, not in `switches.md`'s original 10.**
  Add as its own entry: split-by-default recommendation per the reads-set evidence above,
  blocked on `process.models.superconductors`'s own audit for a final decision (this
  file only shows *that* the branches differ, not each branch's full reads-set, since the
  actual formulas are one level down). Also read (same field) inside `mass.py`'s
  `superconductor()` as a plain array index (`data.tfcoil.dcond[i_tf_sc_mat - 1]`) — a
  data-table lookup, not a formula branch; see `mass.md`'s note on the same field for why
  that use is treated differently.

## calls into other models
- `jcrit_from_material` calls `process.models.superconductors.{itersc, bi2212,
  jcrit_nbti, western_superconducting_nb3sn, jcrit_rebco, gl_nbti, gl_rebco}` — none of
  these audited yet (not a registry unit as of this pass).
- Neither ported function calls anything outside this file.

## JAX-difficulty flags
- `jcrit_from_material`: `if b_max > bc20m: j_crit_sc = 1.0e-9` (branches 1, 3) is a
  data-dependent branch on a *continuous* traced value, not a switch — `minor`,
  `needs-lax-cond-or-where`, standard `jnp.where` fix once this function is in scope.
- `intersect`: the `for _i in range(100): ... break` loop is exactly the "fixed
  iteration count standing in for real convergence" pattern already flagged twice
  elsewhere in this audit (`power_at_ignition_point`, `stellarator.py`'s `output=True`
  double-call) — `workaround-known` for the iteration count itself
  (`lax.while_loop`/`lax.fori_loop` with a real convergence check), but the *early
  `break`* on a data-dependent condition is the harder part: JAX has no early-exit
  `break`, so this needs to become a proper `while_loop` cond, not a mechanical
  `fori_loop` swap. `blocker` for a faithful line-for-line port, `workaround-known` for a
  from-scratch tier-2 `RootFind` reformulation (which is what this audit recommends
  anyway, per `CLAUDE.md`'s general stance against porting PROCESS's own ad hoc
  iteration schemes unchanged).
- `intersect`'s `logger.error(...)` calls on out-of-range `x` are diagnostic side
  effects on a data-dependent condition — not traceable as written, but not needed in a
  proper `RootFind` reformulation either (a real root-finder reports non-convergence
  through its own return status, not a log line).

## open questions
1. **Should `process/models/superconductors.py` be added as a new registry unit?**
   Recommended above — not added to `_audit/unit_registry.md` by this fork per the
   dispatch's scope (only units #10-12/#14 are mine to edit); flagging for whoever
   consolidates this batch.
2. **`intersect`'s residual formulation, if/when it's ported as tier-2:** the natural
   shape is `RootFind` over `x` with residual `y1_interp(x) - y2_interp(x)`, closed
   form once `x1, y1, x2, y2` are fixed arrays (they're tabulated data, not iteration
   unknowns) — `x` is the one real unknown. Whether the harness needs an array-valued
   `Sample.kwargs` extension to test this cleanly, or whether `x1`/`y1`/`x2`/`y2` should
   be treated as `static_argnames` (fixed per call site, not differentiated) is a
   harness-design question, not resolved here.
3. **`i_tf_sc_mat`'s split, once `process.models.superconductors` is audited**: whether
   all 8 branches are actually reachable in the stellarator pipeline, or whether some
   are tokamak-only dead paths in this scope — not checked here (would need
   `preset_config.py`/input-file survey, out of this fork's scope).
