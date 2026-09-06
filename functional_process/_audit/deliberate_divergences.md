# Deliberate divergences from PROCESS

**What this file is.** The port's contract is faithfulness to PROCESS -- regression
agreement within a percentage tolerance is the only oracle (`CLAUDE.md` § Difficulties).
Where the port *knowingly* computes something different, that has to be findable in one
place rather than only in the docstring of whichever function does it. **This is that
place.**

**Scope, deliberately narrow.** Only entries where the port's *behaviour* differs: a
different value, a different derivative, or a different failure. **Not** listed here:

- things not ported yet or ruled out of scope (`unit_registry.md` and `next_steps.md`
  track those, and there are many);
- extra outputs that add information without changing any PROCESS quantity;
- refactors, renames and structural choices with no numerical consequence.

Each entry says what differs, why, and what the evidence is that it is safe. **A divergence
without a receipt does not belong here** -- it belongs in `next_steps.md` as a problem.

Created 2026-09-06 by sweeping the model layer for `deliberate`/`unlike PROCESS`/`diverge`
markers plus the changes made during that session. **It is an initial register, not a
proven-complete one**: the sweep was keyword-based, so a divergence recorded in prose that
uses none of those words would have been missed.

---

## Behavioural divergences

### 1. `WARD_KINK_SMOOTHING` -- a smoothed square-root kink

`models/physics/pure_formulas.py:13,372`. `_fast_alpha_fraction_ward`'s
`sqrt(temp_sum_20 - 0.65)` has derivative zero below the threshold and unbounded above.
Replaced with a smoothed form at `1.0e-3`.

**Why**: the optimiser sat 5.5e-08 from the kink and crossed it on 46 % of steps.
**Receipt**: `stellarator_helias` SAND went from 87-333 iterations and one catastrophic
stop to a stable 24, with every +-1 ulp draw agreeing on `objf` to fifteen digits.
`optimise_design.md` §47; three rejected alternatives in `tried_and_rejected.md`.
**Cost**: values differ from PROCESS by O(1e-3) *in the smoothing region only*.

### 2. `solve_duct_diameter` and siblings -- `stop_gradient` plus one live Newton step

`models/vacuum/vacuum.py:286-313`, and 4 `stop_gradient` sites in `models/` overall. The
loop runs with its parameters frozen, then one differentiable Newton step is taken from the
converged root.

**Why**: gives the exact implicit-function-theorem derivative instead of differentiating a
truncated iteration. **The value is unchanged**; only the derivative differs -- and it
differs by being *correct* rather than by being different. `optimise_design.md` §59.

### 3. `solve_duct_geometry` -- `vmap` over 64 candidates instead of a `while_loop`

`models/vacuum/vacuum.py:316-399`. PROCESS's unbounded `while True` first-fit search became
a fixed 64-candidate `vmap` plus `argmax`.

**Why**: `lax.while_loop` has no transpose rule, so it blocked reverse-mode AD on the whole
graph -- and therefore every off-the-shelf `optimistix` solver (`gauss_newton.py:176` uses
`jax.jacrev`). **Receipt**: seven of eight test cases bit-identical, `nflag` correct on all
eight, and the eighth's 1.11e-16 is the *old* loop's fusion rounding rather than a change in
what is computed. HLO got smaller and the worst trip count 2.6x faster.
`optimise_design.md` §59. **Also caps at 64 where PROCESS has no cap** -- `0.9**64 ~ 1.2e-3`,
so any physical input resolves far sooner; noted in the function's own docstring.

### 4. `intersect` -- full-overlap bracket rather than PROCESS's local window

`models/stellarator/coils/coils.py:313`. Keeps PROCESS's defining equation and interpolation
but brackets over the full overlap and drives it with a convergence-checked bisection plus
Newton polish, where PROCESS uses a local window and guess-clamping.

**Why**: PROCESS's guess-clamping only ever seeds a starting point; the bracket makes the
root find actually convergence-checked. **Caveat, and it is live**: this function's output is
piecewise-smooth with a kink at each of ~200 interpolation breakpoints, which
`optimise_design.md` §72 identifies as the mechanism behind the `stellarator_helias` SAND
pathology. The divergence is not the cause -- PROCESS's version is piecewise-linear too --
but this is the site.

### 5. `scrape_off_layer.py` -- raises where PROCESS does not

`models/physics/scrape_off_layer.py:111`. A **declared divergence**: raises for `0` where
PROCESS silently continues.

### 6. `vacuum.py` -- `a1` staleness not reproduced

`models/vacuum/vacuum.py:328`. PROCESS reads the fits-or-not test's `a1` from the diameter
*before* that Newton step's update; the port computes it from the diameter actually
returned. **Why**: the staleness is bounded by the loop's own 1 % convergence tolerance and
is an artefact of assignment order, not intent. Flagged in `_audit/units/.../vacuum.md`.

### 7. `intersect_residual` -- a C1 monotone cubic where PROCESS interpolates linearly

`models/stellarator/coils/coils.py`. `pchip_interp` (Fritsch-Carlson) replaces `jnp.interp`
in the residual `intersect` solves.

**Why**: piecewise-linear interpolation makes that residual only *piecewise* smooth -- its
derivative jumps at every one of ~200 tabulated breakpoints -- and `stellarator_helias`'s
SAND arm exposes exactly that residual to the outer SQP as an equality.

**Receipt** (`optimise_design.md` §86, §89), ten +-ulp draws at the original resolution:

| | piecewise-linear | C1 |
|---|---|---|
| SLSQP | 8/10, 235-429 iterations, **2 hard caps** | **10/10, 83-101** |
| VMCON | -- | 10/10, **exactly 43 every draw** |

**Cost, and it is the largest in this file.** A different interpolant through the same
points is a different function: the crossing moves **8.1e-05** relative, propagating to
~**5e-04** on TF coil masses, areas and current densities, and to **149 cold-state rows**
(76 on `helias_5b`, 73 on `stellarator_helias`) recorded in `cold_start.ACCEPTED` under
`C1_INTERPOLANT`. Well inside the 5 % regression tolerance; five orders outside what the
port's own tier-3 faithfulness test asks, so that test now names the 22 affected fields
explicitly and holds them to 2e-03 while everything else stays at `rtol = 1e-9`.

**Scope**: **stellarator-only**. `intersect` has no tokamak caller and no tokamak row moved
-- checked, not assumed.

**What it does not fix**: SLSQP and VMCON still settle 1.0e-04 apart, stably across all ten
draws. That gap survives this change and the resolution alternative alike, and is open.

**The alternative it was chosen over**: raising `_N_WINDING_PACK_SAMPLES`. That perturbs
`objf` by 1.06e-03 -- an order of magnitude more -- and does not remove the kinks, only
shrink them; §86's sweep is non-monotone, with N = 1500 still capping.

### 8. `helias_5b.IN.DAT` -- `icc = 11` removed

An **input file**, not a model. Constraint 11 (`rbld == rmajor`) is a tautology on the
stellarator build path: `dr_bore = rmajor - S` then `rbld = dr_bore + S` over the same ten
terms, so `d(rbld)/d(rmajor) = 1` exactly and the equation can never bind.

**Receipt**: PROCESS's own regression run differs in exactly **one row** --
`n_equality_constraints+n_inequality_constraints`, 5 -> 4 -- with every physics variable
unchanged. `optimise_design.md` §46, §48, §52. **Note**: the tracked reference MFILE in
`process-tracking-data` needs regenerating; until it is, `pytest tests/regression -k
helias_5b` fails on that single metadata row.

---

## Considered and NOT done

Recorded so the reasoning is not re-derived.

- **De-ratcheting `dr_tf_plasma_case`** (`optimise_design.md` §77, §78). The input arm reads
  the value it writes back, making a ratchet whose fixed point is non-unique. Tested:
  replacing the self-read with the file's stated value is **bit-identical** on both affected
  configurations. **Not done because it buys nothing measurable**: every arm on both
  configurations already converges, SLSQP included, and PROCESS genuinely does ratchet (it
  re-runs the pipeline over a mutated `DataStructure`), so changing it is a divergence with
  no receipt on the other side of the ledger.
- **Promoting that clamp to a constraint** (§77). Would make the model smooth, but requires
  `dr_tf_plasma_case` to become a design variable -- a degree of freedom PROCESS does not
  have -- to give the optimiser something to act on. A change to the problem statement, for
  a kink that is currently costing no solver anything.
- **Smoothing the TF case clamp.** Wrong in kind: unlike the Ward kink, this one is real
  geometry (an arc's sagitta) doing its job.
