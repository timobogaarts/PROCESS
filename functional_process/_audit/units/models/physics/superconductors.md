---
kind: model-unit
status: draft
confidence: high
---

**Ported (7/7 in-scope functions, plus one transitively-required shared helper).**
`superconductors.py` / `test_superconductors.py`: `jcrit_rebco`, `itersc`, `jcrit_nbti`,
`bi2212`, `gl_nbti`, `gl_rebco`, `western_superconducting_nb3sn` — all **tier 1**, not
tier 2 as the registry anticipated (see "tier signal" below for why). `bottura_scaling`,
a pure helper `itersc` and `western_superconducting_nb3sn` both call, is ported alongside
them (same transitive-inclusion precedent as `fusion_reactions.md`'s
`bosch_hale_reactivity`) but is not itself one of the seven named units. Verified against
live PROCESS calls at full float64 precision, including every branch (see "tier signal"),
before writing the harness cases.

## source

`process/models/superconductors.py` (1289 lines total). Registry unit #22, scoped to
these 7 material-model functions by the dispatch task (~603 LOC estimated there; measured
in this pass — including `bottura_scaling`, which two of the seven call, with full
docstrings — at closer to 745 LOC across the 8 in-scope defs):

| function | lines |
|---|---|
| `jcrit_rebco` | 167-254 |
| `itersc` | 296-378 |
| `jcrit_nbti` | 381-446 |
| `bi2212` | 449-529 |
| `gl_nbti` | 532-625 |
| `gl_rebco` | 628-725 |
| `western_superconducting_nb3sn` | 852-930 |
| `bottura_scaling` (transitively in scope — called by `itersc`, `western_superconducting_nb3sn`) | 933-1088 |

**Out of scope, not audited here** (the registry's own scope note already excludes them,
confirmed by re-reading): `current_sharing_rebco` (257-293, the file's one
`scipy.optimize.newton` call — likely the source of the registry's "tier-2 leaning"
expectation, see "tier signal"), `hijc_rebco` (728-849), `CroCoCableGeometry`/
`calculate_croco_cable_geometry` (1091-1196), `superconductor_current_density_margin`
(1199-1289, a second, differently-scoped dispatcher — see "switches touched"), and the
`SuperconductorShape`/`SuperconductorType`/`SuperconductorMaterial`/`SuperconductorModel`
enums (18-165, metadata only, not read by any of the 7 in-scope functions).

**Called from** (all found by grep, cross-checked against every importer of
`process.models.superconductors`):

- `process/models/stellarator/coils/coils.py::jcrit_from_material` (unit #10, stellarator,
  **in scope** for this pipeline) — the sole reason this unit exists (found by unit #10's
  own audit as a third-level scope miss, see `coils.md`).
- `process/models/tfcoil/superconducting.py` and `process/models/pfcoil.py` — tokamak/PF
  TF-coil code, **out of the stellarator scope** this registry covers
  (`test_harness.md`'s "Scope note"), not audited or traced further here.

## data footprint

**None of the 7 functions (nor `bottura_scaling`) reads `self.data` at all** — every one
is already a plain module-level function over explicit scalar arguments, exactly like
`coils.md`'s `intersect`/`j_crit_cable_from_fraction`. There is no `data`-backdoor to
close; the proposed signatures below are the source signatures, unchanged.

| VarPath | read/write | classification | note |
|---|---|---|---|
| — | — | — | no `data.*` access in any of the 8 in-scope functions |

Provenance of each function's real arguments, for context (not a data footprint in the
schema sense, since these are already plain parameters): `jcrit_from_material`
(`coils.py`, unit #10, unported) supplies `b_max`/`t_helium` — themselves locals of unit
#9's `winding_pack_total_size` solve loop — plus, for two branches, `data.tfcoil.*` fields
read directly at the `jcrit_from_material` call site (`b_crit_sc`/`t_crit_sc` for
`i_tf_sc_mat == 4`, `b_crit_upper_nbti`/`t_crit_nbti` for `i_tf_sc_mat == 7` — see
"switches touched"). None of that provenance changes this unit's own signatures.

## proposed signature(s)

Ported, tier-1, unchanged from source (parameter names kept identical — already following
the naming convention):

```python
def jcrit_rebco(
    temp_conductor, b_conductor
) -> tuple: ...  # (j_critical, validity, b_c20max, temp_c0max)


def bottura_scaling(
    csc,
    p,
    q,
    c_a1,
    c_a2,
    epsilon_0a,
    temp_conductor,
    b_conductor,
    epsilon,
    b_c20max,
    temp_c0max,
) -> tuple: ...  # (j_scaling, b_critical, temp_critical)


def itersc(
    temp_conductor, b_conductor, strain, b_c20max, temp_c0max
) -> tuple: ...  # (j_critical, b_critical, temp_critical)


def jcrit_nbti(
    temp_conductor, b_conductor, c0, b_c20max, temp_c0max
) -> tuple: ...  # (j_critical, temp_critical)


def bi2212(
    b_conductor, jstrand, temp_conductor, f_strain
) -> tuple: ...  # (j_critical, temp_margin)


def gl_nbti(
    temp_conductor, b_conductor, strain, b_c20max, t_c0
) -> tuple: ...  # (j_critical, b_critical, t_critical)


def gl_rebco(
    temp_conductor, b_conductor, strain, b_c20max, t_c0
) -> tuple: ...  # (j_critical, b_critical, temp_critical)


def western_superconducting_nb3sn(
    temp_conductor, b_conductor, strain, b_c20max, temp_c0max
) -> tuple: ...  # (j_critical, b_critical, t_critical)
```

## cottax node

**None written**, for the same reason as every function in `coils.md`
(`j_crit_cable_from_fraction`, `bmax_from_awp`, `intersect`): every real call site's
arguments are locals inside `jcrit_from_material` (unit #10, unported) — themselves
downstream of unit #9's own unminted solve-loop locals — not established `.area.field`
`VarPath`s this audit can independently verify. Wrapping any of the 8 functions as an
`ExplicitFunction` now would assert a wiring this pass has no basis for (`schema.md`:
"skip this section... while open questions about the signature itself are unresolved").

The natural home, once unit #10 mints real `VarPath`s for `jcrit_from_material`'s locals
(`b_max`, `t_helium`, and the per-branch `bc20m`/`tc0m` — literal in five of the eight
`i_tf_sc_mat` branches, `data.tfcoil.*` reads in two, absent entirely in the other one, see
"switches touched"), is one `ExplicitFunction` node per branch — i.e. per material model —
rather than one node wrapping the whole switch, matching `traceability_policy.md`'s
split-by-default and `coils.md`'s own recommendation. That design step belongs to whoever
ports `jcrit_from_material`, not to this unit.

## tier signal

**All 8 in-scope functions are tier 1** — explicit pure functions, no internal iteration,
no `self.data` access. This **corrects** the registry's/`next_steps.md`'s "tier-2 leaning
(`scipy.optimize`)" expectation: the only `scipy.optimize` call in the whole 1289-line
file is inside `current_sharing_rebco` (`optimize.newton`, line 282), which is not one of
the 7 in-scope functions and is not called by any of them — it calls `jcrit_rebco`, not
the other way round. `jcrit_rebco` itself, and the other six, are ordinary closed-form
algebraic formulas with Python `if`/`else` branches on continuous inputs (needs
`jnp.where`, not a solver) and, for `itersc`/`western_superconducting_nb3sn`, one shared
non-iterative helper (`bottura_scaling`, itself branchy but not looping).

Verified directly (not assumed) against live `process.models.superconductors` calls,
component by component, at every branch of every function, before writing the harness
cases below — full float64 agreement in every case tried, including:

- `jcrit_rebco`'s three arms: in-validity-range/out-of-range, `b_conductor < birr` /
  `>= birr`, `temp_conductor < temp_c0max` / `>=`.
- `bottura_scaling`'s "inside critical surface" / "outside" branch (exercised through
  `itersc` at an extreme `(t, b)` point), and its normal/abnormal `temp_critical` arm.
- `jcrit_nbti`'s `bratio < 1` / `>= 1` arms.
- `gl_nbti`'s `b_reduced <= 1` / `> 1` arms.
- `bi2212`'s in-range point and its raise condition (see "JAX-difficulty flags").

Also verified `jax.jacfwd` is finite at every one of the above points (both branches of
every `jnp.where`), confirming the domain guards described in "JAX-difficulty flags" below
actually close off the NaN-through-untaken-branch failure mode `test_gradient_finite`
exists to catch, not just that the port's *value* happens to agree.

## switches touched

- **`i_tf_sc_mat` (`.tfcoil.i_tf_sc_mat`) — not yet in `switches.md` (10 original entries),
  first formally recorded here** (also referenced provisionally by `coils.md`, which found
  it first but deferred the actual reads-set evidence to this unit's audit — that evidence
  is below). **Not touched by any of the 8 functions in this file** — `i_tf_sc_mat` is
  read and dispatched on entirely inside `jcrit_from_material` (`coils.py`, unit #10,
  which calls into this unit, not the reverse), so this unit's own functions have no
  switch dependency of their own. Recorded here per the task's instruction (not yet
  formally recorded elsewhere) and not added to `total_process.py`'s
  `TOPOLOGY_SWITCHES`/`total_process.py` — that consolidation is the coordinating
  session's job.

  **Values seen and per-value reads-set** (from `jcrit_from_material`,
  `process/models/stellarator/coils/coils.py:52-162`), confirming and completing
  `coils.md`'s partial finding:

  | `i_tf_sc_mat` | function called | `bc20m`/`tc0m` source | extra reads |
  |---|---|---|---|
  | 1 | `itersc` | literal `32.97`/`16.06` | — |
  | 2 | `bi2212` | n/a (function takes no `b_c20max`/`temp_c0max`) | `f_hts` (→ `f_strain`) |
  | 3 | `jcrit_nbti` | literal `15.0`/`9.3` | `c0 = 1.0` (literal) |
  | 4 | `itersc` (again) | `data.tfcoil.b_crit_sc`/`t_crit_sc` | — |
  | 5 | `western_superconducting_nb3sn` | literal `32.97`/`16.06` (same as 1) | — |
  | 6 | `jcrit_rebco` | n/a (function takes neither) | — (but see open question 1: the call site itself looks broken) |
  | 7 | `gl_nbti` | `data.tfcoil.b_crit_upper_nbti`/`t_crit_nbti` | — |
  | 8 | `gl_rebco` | literal `429`/`185` | — |

  Reads-sets genuinely differ per branch (branch 4 alone reads `b_crit_sc`/`t_crit_sc`,
  branch 7 alone reads `b_crit_upper_nbti`/`t_crit_nbti`, branches 1/3/5/8 use fixed
  literals, branches 2/6 use neither) — per `traceability_policy.md`'s split-by-default,
  this is unambiguous evidence for **split**, one function/node per value, confirming
  `coils.md`'s provisional call now that the actual formulas (this unit) have been read.

  **Entangled switches**: none found — no other switch changes which of these 8 branches
  is reachable or what it reads.

  **A second, inconsistent dispatcher exists for the same switch** —
  `superconductor_current_density_margin` (this file, 1199-1289, out of scope, called from
  `pfcoil.py`/`tfcoil/superconducting.py`, both out of the stellarator pipeline) maps
  `i_tf_sc_mat` values `{1, 3, 4, 5, 7, 8, 9}` to `{itersc, jcrit_nbti, itersc,
  western_superconducting_nb3sn, gl_nbti, gl_rebco, hijc_rebco}` — **missing values 2
  (`bi2212`) and 6 (`jcrit_rebco`) entirely**, and **including value 9 (`hijc_rebco`)**,
  which `jcrit_from_material` never reaches at all. Both dispatchers key off the same
  `SuperconductorModel` `IntEnum` (this file, lines 73-124: 1=`ITER_NB3SN`, 2=`BI2212`,
  3=`OLD_LUBELL_NBTI`, 4=`USER_DEFINED_NB3SN`, 5=`WST_NB3SN`, 6=`CROCO_REBCO`,
  7=`DURHAM_NBTI`, 8=`DURHAM_REBCO`, 9=`HAZELTON_ZHAI_REBCO`), so the two dispatchers
  disagree about which values are even legal for their respective callers — not a bug in
  this unit's own code, but real evidence that `i_tf_sc_mat`'s valid range is genuinely
  caller-dependent (stellarator TF coil vs. PF coil / tokamak TF coil), which is exactly
  why `hijc_rebco`/`current_sharing_rebco`/`superconductor_current_density_margin` are
  correctly out of scope here: they are only reachable from callers outside the
  stellarator pipeline this registry covers.

  **Split decision: split.** **Confidence: high** — reads-set evidence is direct, from
  reading the actual dispatcher body, not inferred.

## calls into other models

None. This unit is the leaf of its call chain: `jcrit_from_material` (unit #10) calls
*into* these 8 functions; none of the 8 calls out to any other `Model` or registry unit.
(`itersc`/`western_superconducting_nb3sn` call `bottura_scaling`, which is in this same
file and ported alongside them — not a cross-unit call.)

## JAX-difficulty flags

All `workaround-known` / `minor` — no blockers found in any of the 7 in-scope functions.

- **Data-dependent `if`/`else` branches on continuous inputs**, in every function except
  `gl_rebco`: `jcrit_rebco` (validity range, `temp_conductor < temp_c0max`,
  `b_conductor < birr`), `bottura_scaling` (`f_b_conductor_critical_no_temp < 1.0`,
  "inside/outside critical surface"), `jcrit_nbti` (`bratio < 1`), `gl_nbti`
  (`b_reduced <= 1`) — `needs-lax-cond-or-where`, `workaround-known`, all converted to
  `jnp.where` in the port. Each one required guarding the *untaken* branch's base against
  a negative value under a non-integer power (e.g. `bottura_scaling`'s
  `(1 - f_b_conductor_critical_no_temp) ** (1/1.52)`, which is NaN for
  `f_b_conductor_critical_no_temp > 1` even though that branch is never selected there) —
  the exact `jnp.where`-leaks-NaN-through-the-untaken-branch failure mode
  `test_gradient_finite` exists to catch. Verified by hand with `jax.jacfwd` at every
  branch boundary before writing the harness cases (see "tier signal") — every gradient
  came back finite, not just the value.
- **`logger.error(...)` diagnostic calls on out-of-range/artificially-lowered inputs**
  (`jcrit_rebco`, `bottura_scaling` ×2) — side effects with **no return-value
  consequence** (the source computes the same formula regardless of whether the log
  fires), dropped in the port. `minor`.
- **`bi2212`'s `raise ProcessValueError(...)`** outside the fit's validity range
  (`temp_conductor > 20.0` or `b_conductor < 6.0` or `b > 104.0`, computed *after* both
  return values, so the port's arithmetic is identical either way) — a traced function
  cannot raise, so the port masks both outputs to `jnp.nan` there instead, same idiom
  `plasma_profiles.py`'s `_gradient_length` already established
  (`Tier1Contract.reference_domain_errors`). Verified directly: PROCESS raises at
  `(b_conductor=3.0, jstrand=2e7, temp_conductor=25.0, f_strain=0.2)`, the port returns
  `(nan, nan)` at the same point. `workaround-known`.
- **`gl_nbti`/`gl_rebco`'s fractional powers of `(1 - b_reduced)`** outside physical
  validity (`b_reduced > 1`, i.e. operating above the critical field) — `gl_nbti`'s source
  itself branches here (ported, see above); `gl_rebco`'s source does **not** branch at all
  and always applies the fractional exponent `q = 2.5`. For `b_reduced > 1` this makes the
  *source* itself (plain Python floats, not `numpy`) return a **complex number**
  (`(-x)**2.5` is complex under Python's `**`), not a NaN — a pre-existing PROCESS
  oddity, not introduced by this port. `jnp`'s `**` instead returns `NaN` for the same
  input, so the port's behaviour would diverge from the reference's *type* (complex vs.
  NaN) at that unphysical corner. Not fuzzed there (kept within `b_reduced < 1`, the same
  "smooth middle only" convention `fusion_reactions.md`'s
  `TestBeamFusionCrossSection` uses) — `minor`, flagged for whoever eventually fuzzes this
  wider.
- No CoolProp calls, no `copy.deepcopy`, no `scipy.optimize`/`fsolve` in any of the 8
  in-scope functions (confirmed by re-reading all eight in full, not just grepping) — this
  is the correction to the registry's tier expectation (see "tier signal").

## open questions

1. **`jcrit_from_material`'s `i_tf_sc_mat == 6` call site looks broken**, found as a side
   effect of tracing every caller for the switch table above, not part of this unit's own
   scope to fix: `process/models/stellarator/coils/coils.py:136` calls
   `superconductors.jcrit_rebco(t_helium, b_max, 0)` — **three** positional arguments —
   but `jcrit_rebco`'s signature (this file, confirmed unchanged, line 167) takes exactly
   **two** (`temp_conductor`, `b_conductor`). As written, executing `i_tf_sc_mat == 6`
   would raise `TypeError: jcrit_rebco() takes 2 positional arguments but 3 were given`.
   Not reproduced or fixed here (out of this file's boundary — `coils.py` belongs to unit
   #10), but worth flagging loudly for whoever ports `jcrit_from_material`: either this
   branch is genuinely dead/never executed in the stellarator pipeline (in which case
   `i_tf_sc_mat == 6`/`jcrit_rebco` may be unreachable in practice, similar to unit #21's
   `i_nd_plasma_pedestal_separatrix` nodes turning out unreachable from stellarator), or
   this is a live bug in current PROCESS. This unit's own `jcrit_rebco` port (2-argument
   signature, matching the source function actually defined) is unaffected either way —
   it's the *caller* that looks wrong, not the callee ported here.
2. **Whether all 8 (or 9, counting `hijc_rebco`) `i_tf_sc_mat` values are actually
   reachable from real stellarator input files** — not checked here (would need a
   `preset_config.py`/input-file survey), same open item `coils.md` already left
   unresolved (its open question 3).

## Derivative-safe power laws (`safe_pow` / `safe_sqrt`)

4 fractional power laws and 3 square roots in this file have been rewritten from `x ** p` / `jnp.sqrt(x)` to
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


## 2026-08-30 (evening) -- `hijc_rebco` ported

`hijc_rebco` (`process/models/superconductors.py:728-849`) is the eighth in-scope
function and was the one gap this record left: Wolf et al.'s parameterisation with
Hazelton and Zhai's fit values, the arm `superconpf`/`supercon` take at
`i_*_superconductor == 9`. Ported for `models/pfcoil/superconductor.py`'s REBCO PF
strand occupant (both spherical tokamaks set `i_pf_superconductor = 9`);
`next_steps.md` §18.5 records that the CroCo TF package needs the same function, so
this landing may collide with that wave's.

**Its `if b_critical > b_conductor` is not a switch and does not become an occupant.**
Both arms are the same expression with the sign of the last bracket reversed, and
PROCESS says why in its own comment (`:818-822`): above the critical field
`1 - b/b_c` is negative and its fractional power `q = 0.9` would be `nan`, so the sign
is flipped to keep a real -- and deliberately negative -- critical current. Written as
one `jnp.where`-free `safe_pow(|1 - b/b_c|, q)`, which is the same number on both sides
and forms no `nan` in an untaken branch to leak into the tangent. `safe_pow` guards both
fractional powers of the reduced field.

Verified **bit-exact** against PROCESS over 200 random `(temp_conductor, b_conductor)`
points spanning both sides of the critical field (worst relative difference `0.0`), with
finite gradients in both arguments. **A harness contract is owed** -- see
`next_steps.md`.
