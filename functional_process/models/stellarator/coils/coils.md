---
kind: model-unit
status: draft
confidence: high
---

**Ported (3/4).** `coils.py` / `test_coils.py`: `j_crit_cable_from_fraction` and
`bmax_from_awp` (tier-1, ported previously), plus `intersect` (tier-2, ported this pass —
see below). `jcrit_from_material` remains **not** ported — still blocked on
`process/models/superconductors.py` not being an audited registry unit; re-confirmed by
this pass, not re-litigated (see below).

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
| — | — | — | `intersect` takes no `data` at all — already pure in the source (see below). |

`jcrit_from_material`'s footprint is call-site-dependent (not tabulated here, since it
isn't ported this pass — see below).

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

Ported, tier-2, this pass:
```python
def intersect_residual(x, x1, y1, x2, y2) -> float:
    """y1_interp(x) - y2_interp(x); vanishes at the crossing."""
    ...

def intersect(x1, y1, x2, y2, xin) -> float:
    """The x at which the two curves cross, found by bisection + Newton polish."""
    ...
```
`intersect` was already pure in the source — no `data.*` reads, no calls to any other
model. Its args are exactly PROCESS's: `x1, y1, x2, y2` (tabulated curves), `xin`
(starting guess).

**Not ported — `jcrit_from_material`.** Re-confirmed this pass (grepped
`functional_process` and `process/models/superconductors.py` again — the module is still
1289 lines, still not an audited registry unit, still not touched by any parallel fork).
A genuine 8-way switch on `i_tf_sc_mat`
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

## cottax node

**Actually written** for `j_crit_cable_from_fraction`/`bmax_from_awp`'s siblings in other
units, but **not** for any of the three functions in this file, including the newly
ported `intersect` — and for the same reason in all three cases. Every one of their real
call-site arguments (`coilcurrent`, `wp_width_r_min`, `r_coil_major`, `r_coil_minor` for
`bmax_from_awp`; `j_crit_sc`/`f_tf_conductor_copper`/`f_he` for
`j_crit_cable_from_fraction`, called from inside `jcrit_from_material`; and, for
`intersect`, `wp_width_r` (used as both `x1` and `x2`), `lhs`, `rhs`, `wp_width_r_min`) is
a *local* computed inside `winding_pack_total_size`'s solve loop (`coils/calculate.py`,
unit #9), not an established `.area.field` this audit has independently verified —
wrapping any of them as a node now would assert a wiring this pass has no basis for (see
`schema.md`: "skip this section... while open questions about the signature itself are
unresolved"). Correct home for all three nodes is wherever unit #9 declares its own
solve — `calculate.md`'s open question #1 already raises this exact tension for
`coilcurrent`.

The natural declaration, once unit #9 mints real `VarPath`s for its locals (the same
minting `calculate.py`'s `CoilCurrent` node already did for `coilcurrent`), is a
pytree-namespace `ImplicitFunction` pairing `intersect_residual` with a `RootFind` over
one unknown:
```python
class Intersect(ImplicitFunction):
    wp_width_r_min = Output(lambda s: s.stellarator.wp_width_r_min)

    def residual(
        self,
        x1=Input(lambda s: s.stellarator.wp_width_r),
        y1=Input(lambda s: s.stellarator.lhs),
        x2=Input(lambda s: s.stellarator.wp_width_r),
        y2=Input(lambda s: s.stellarator.rhs),
    ):
        return intersect_residual(<the unknown>, x1, y1, x2, y2)  # sketch only
```
**Open question this sketch surfaces**: `xin` has no place in this shape at all. A
`RootFind`'s starting guess comes from whatever `Drive`s the block (see `evaluate.py`'s
`Drive.__call__`: `guess = env[unknowns] if started else None`, handed to the driver
positionally), not from an `In` on the residual body — so PROCESS's `xin` argument simply
does not survive into the node-graph port as a port. That is fine for `intersect` itself
(the pure function below still takes and uses it, faithfully), but it means the *node*
wrap, once written, has one fewer declared input than the function it wraps — worth
flagging explicitly, since `_audit/naming_convention.md` has no category yet for "an
argument that is real in the pure function but has no port in the node."

## tier signal
- `j_crit_cable_from_fraction`: **tier 1** — pure, no `data`, no branch.
- `bmax_from_awp`: **tier 1** — pure once the 2-field `data` back-door is closed (same
  shape as `st_sudo_density_limit` in `density_limits.py`).
- `jcrit_from_material`: tier 1 *per branch*, once split — see above, blocked on
  `process.models.superconductors` being audited.
- `intersect`: **tier 2, ported this pass.** Self-contained (no calls into other
  models, no `data` access at all — confirmed by reading the full 100-odd lines): a
  genuine internal Newton-Raphson-style solve over two tabulated `(x, y)` curves, fixed
  100-iteration cap with an early `break` on `abs(y01 - y02) < epsy`. Exactly the "internal
  iterative loop closing over state local to one model" `test_harness.md`'s tier-2 section
  describes, and — per that same section's own framing — this is the first unit in the
  registry to actually exercise `Tier2Contract`'s residual-based pass criterion (see
  `test_coils.py`).

## switches touched
- `i_tf_sc_mat` (`.tfcoil.i_tf_sc_mat`) — **new, not in `switches.md`'s original 10.**
  Add as its own entry: split-by-default recommendation per the reads-set evidence above,
  blocked on `process.models.superconductors`'s own audit for a final decision (this
  file only shows *that* the branches differ, not each branch's full reads-set, since the
  actual formulas are one level down). Also read (same field) inside `mass.py`'s
  `superconductor()` as a plain array index (`data.tfcoil.dcond[i_tf_sc_mat - 1]`) — a
  data-table lookup, not a formula branch; see `mass.md`'s note on the same field for why
  that use is treated differently.
- `intersect` touches no switch — genuinely pure over its five array/scalar arguments.

## calls into other models
- `jcrit_from_material` calls `process.models.superconductors.{itersc, bi2212,
  jcrit_nbti, western_superconducting_nb3sn, jcrit_rebco, gl_nbti, gl_rebco}` — none of
  these audited yet (not a registry unit as of this pass).
- Neither `intersect` nor the two previously-ported functions call anything outside this
  file.

## JAX-difficulty flags
- `jcrit_from_material`: `if b_max > bc20m: j_crit_sc = 1.0e-9` (branches 1, 3) is a
  data-dependent branch on a *continuous* traced value, not a switch — `minor`,
  `needs-lax-cond-or-where`, standard `jnp.where` fix once this function is in scope.
- `intersect`, **resolved this pass**: the source's `for _i in range(100): ... break`
  loop with a data-dependent early exit had no faithful `jax`-traceable translation (JAX
  has no early-exit `break`), and its post-loop clamp-and-`logger.error` bail-out on
  leaving `[xmin, xmax]` is a diagnostic side effect on a data-dependent condition — not
  traceable as written either. The port does not attempt a line-for-line translation of
  either: it re-poses the same defining equation (`intersect_residual`) and drives it with
  `optimistix.root_find`'s `Bisection` (bracket = the curves' full x-overlap, which is a
  valid sign-changing bracket exactly whenever a crossing exists there — no `xin`-
  dependent windowing, so, unlike PROCESS's local Newton-Raphson, a bad `xin` cannot walk
  the solve off the domain or onto the wrong crossing) followed by a fixed few `jax.grad`-
  based Newton corrections once bisection has localised `x` into the correct linear
  segment of the piecewise-linear interpolation (see `coils.py`'s
  `_intersect_newton_polish` docstring for why a Newton step is *exact*, not approximate,
  once there). `throw=False` on the `root_find` call means non-convergence is reported
  through `optimistix`'s own `result` field rather than raised or logged — the traced
  equivalent of PROCESS's `logger.error`, per `traceability_policy.md`.
- `jnp.interp` (used by `intersect_residual`) is piecewise-linear and therefore not
  everywhere-differentiable (a kink at each tabulated `x` node) — `minor`, and not a
  concern for this port specifically: `Tier2Contract` has no gradient-agreement test (see
  `test_harness.md`), so no check here differentiates through the interpolation's kinks.
  Worth flagging for whoever eventually drives the `ImplicitFunction`/`RootFind` pair
  sketched above through an actual `jacfwd`, e.g. for a tier-4 MDA's own sensitivities.

## open questions
1. **Should `process/models/superconductors.py` be added as a new registry unit?**
   Still recommended, still not added to `_audit/unit_registry.md` (out of this fork's
   scope) — flagging again for whoever consolidates this batch, since `intersect`
   landing removes the *other* blocker on `winding_pack_total_size` (unit #9), leaving
   `jcrit_from_material`/`process.models.superconductors` as the one dependency left.
2. **`xin`'s disappearance from the node wrap** (see `## cottax node` above) — not
   resolved here, flagged as a naming-convention gap.
3. **`i_tf_sc_mat`'s split, once `process.models.superconductors` is audited**: whether
   all 8 branches are actually reachable in the stellarator pipeline, or whether some
   are tokamak-only dead paths in this scope — not checked here (would need
   `preset_config.py`/input-file survey, out of this fork's scope).
4. **`winding_pack_total_size` (unit #9) unblocking**: with `intersect` now ported,
   `winding_pack_total_size` (`calculate.md`) is blocked on exactly one remaining thing —
   `jcrit_from_material` (open question 1 above), not on anything in this file anymore.
   It also still needs its own locals (`wp_width_r`, `lhs`, `rhs`, `wp_width_r_min`,
   `coilcurrent`) minted as real `VarPath`s before any of the three nodes sketched in
   this record or in `calculate.md`'s own notes can be written — a design step, not a
   blocked dependency. Not attempted here; `winding_pack_total_size` is unit #9's to
   port, not this fork's.
