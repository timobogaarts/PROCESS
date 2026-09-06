# The `Optimise` layer — design

**Status: built.** This was the design-and-measurement record for wiring PROCESS's
constraints, objective and iteration variables into a real `cottax.problem.Optimise` over
the driven graph, and for choosing/benchmarking the SQP driver around it
(`functional_process/sand.py`, `core/solver/drivers.py`, `sand_harness.py`,
`run_sand_harness.py`, `test_sand.py`, `mdf.py`). All of it landed. The port's SQP driver
reproduces PROCESS's answers on the tracked configurations (see `unit_registry.md` /
`next_steps.md` for current per-configuration status) and is measured at roughly 9-12x
PROCESS's wall clock end-to-end and ~180x per iteration, with exact forward-mode
gradients replacing PROCESS's finite differences.

This file grew to 12,000+ lines of investigation across dozens of numbered sections while
that was being worked out — SAND vs. MDF, driver choice, jit/compile strategy, a long
chase of one stellarator configuration's solve instability, and a kink-smoothing retune.
None of that narrative is needed to use or extend the driver layer today. What is worth
keeping — the specific things tried and rejected, with the numbers that killed them, so
nobody re-spends the time re-deriving them — has been extracted to
`_audit/tried_and_rejected.md`. The full investigation, section by section, remains in
git history (`git show be7756fc:functional_process/_audit/optimise_design.md` and its
ancestors, or `git log -- functional_process/_audit/optimise_design.md`); this file's
sections are cited by number from many code comments and other audit records, and none of
that numbering has been changed by this cut — it is simply no longer reproduced here.

## 39. `session.py`'s 29-compile assembly-and-prime overhead, diagnosed and partly fixed (2026-09-05)

Two questions, both answered by instrumenting `jax._src.compiler.backend_compile_and_load`
(module name + call stack per compile) rather than guessing, per `session.py`'s own
docstring recording a full `open_session(...).mdf()` at **29 compiles / 19.3 s** cold,
**0 / 0.9 s** warm, on `stellarator_helias`.

**Q1 — why the fused schedule traced twice.** `mdf.prime` and `mdf.solve` both call
`run_schedule(mdf.eager, ...)` on the *same* `Schedule` object, and `schedule_verdict`
confirms the whole-schedule jit succeeds both times (`(True, None)`) — so this was never
the eager-walk fallback. Diffing the two calls' input envs leaf by leaf found 13 keys
(every design variable, and every `^guess.*` port once `prime` has written it) whose
`shape`/`dtype` agreed but `weak_type` did not: `True` from `mdf.seed`'s
`jnp.asarray(python_float)`, `False` from `VmconDriver`'s answer, which crosses a NumPy
array (`drivers._sqp_callback`'s host round trip). `weak_type` is part of the abstract
value `eqx.filter_jit` keys its cache on, same as shape and dtype, so `solve`'s re-seeded
env was a structurally different pytree at the same names over the same `Schedule` —
exactly `_root_find_seed`'s lesson (one leaf's identity/structure mismatches, the whole
memo misses) in a different guise. Fix: `mdf._not_weak` forces `weak_type` off on every
value `mdf.seed` hands the schedule (`jax.lax.convert_element_type`/an explicit `dtype=`
on `jnp.asarray` — checked, both strip the bit; the latter is one XLA program instead of
two, using `numpy.asarray(value).dtype` to read the target dtype because
`jax.numpy.result_type` refuses a bare Python `list`, which `grounded` is for every
array-valued field). [measured] `stellarator_helias` MDF: 29 -> 27 compiles, first solve
19-22 s -> ~17.5 s, second solve unchanged at 0 compiles / ~0.7-0.9 s.

**Q2 — what the 25 trivial compiles are.** Traced (via the compile call stack) to three
sources, not one: **8** are `mdf.seed`'s own eager `jnp.asarray` staging of every input
(now folded to one op per (shape, dtype) signature by the Q1 fix, same count); **5** are
inside `core/solver/drivers.py`, `optimistix`'s own trace-time constant-folding the first
time `SeededNewtonDriver`'s Newton solver is traced (`equinox._enum`'s `__ne__` under
`implicit_jvp`, concrete-valued so jax evaluates it eagerly while building the jaxpr, not
part of it); the remaining **~12** are also in `drivers.py`, `VmconDriver`/`host_cache`
setup (`ravel_pytree(start)`'s reshape/concatenate, `_sqp_callback`'s unravel/split/
reshape of the driver's answer, `optimistix.fixed_point`'s own setup). **The "stated
ports" hypothesis is refuted**: `models/stated.py`'s ~21 declarations are not
distinguishable from any other `mdf.seed` input in the trace — they are plain Python
floats exactly like every other `ground_truth` read, and all ~350 of `mdf.seed`'s inputs
already dedup to a handful of (shape, dtype) programs, not one each. No fix applied to
the `drivers.py` sources (17 of 25): out of this worktree's scope (another agent's file),
and each is a one-time ~7 ms cost paid once per `Session` object, not per scan point —
confirmed by `session.series()` showing 0 compiles on a repeat solve, including a re-seed
from a different `cold` `DataStructure`.

**Gate.** `_x` and `objf` compared as `float.hex()` before/after the Q1 fix on
`stellarator_helias` MDF: identical to the last bit. `tests/functional_process
tests/unit` unchanged (see commit).

## 40. `VmconDriver.fused` landed on by default (2026-09-05)

`host_cache.bind`'s fused `values_and_jacobian` program (§31.30) was held off since
2026-09-03 because turning it on flipped `stellarator_helias` cold SAND from converged
to stopped — a real row, but §31.32 showed the arm was unstable to *any* last-bit
Jacobian change, fused or not, and `WARD_KINK_SMOOTHING = 1e-3` (landed since, see
`tried_and_rejected.md`'s Ward kink entry) closed off the mechanism: the same arm now
takes 24 iterations on every `+-1` ulp draw. That removed the one reason `fused` was
held back, so it was re-measured rather than re-argued.

**Re-measured [2026-09-05, measured].** Full seven-configuration cold matrix
(`--native --compare-process`, twelve MDF/SAND rows) with `fused` on is **bit-for-bit
identical** to the tracked `reference_cold_matrix.txt` — every status stays `converged`,
every printed digit of `objf`/`max|eq|`/`min ie`/`worst dx` is unchanged. No row was
re-baselined because no row moved. The fused-vs-split Jacobian still differs at the last
bit or two (unchanged from §31.30.4/§31.32.1's finding — not a bitwise spelling, and
never claimed to be), but nothing downstream of that difference is visible on any row
today.

**What it buys, SAND arm only, `--native` seeding, isolated via `host_cache`'s three
programs directly [measured 2026-09-05]:**

| | `values` + `jacobian` | fused | removed | per-call split | per-call fused |
|---|---|---|---|---|---|
| `stellarator_helias` SAND | 5 648 + 11 480 = 17 128 lines | 11 595 lines | 5 533 (32%) | 2.560 ms | 1.572 ms (-39%) |
| `large_tokamak_nof` SAND | 15 810 + 40 301 = 56 111 lines | 40 555 lines | 15 556 (28%) | 9.459 ms | 8.237 ms (-13%) |

Line counts are `lower(...).as_text()` StableHLO line counts — a property of the program,
not of machine load. Per-call times are medians over every call one solve makes (47 and
19 calls respectively), each set collected within one process. Whole-row wall clock is
not quoted; §31.30.1 already found it unusable under concurrent load.

`VmconDriver.fused` now defaults to `True`. `tests/functional_process tests/unit` still
pass (see the commit that landed this). `tried_and_rejected.md`'s fused entry records the
held-back-then-shipped history in full.

## 41. Where a warm solve's time actually goes (2026-09-05)

§40 landed `fused` and left the obvious question unasked: **is a block call expensive
because the model is expensive, or because calling it is?** Measured on today's tree,
medians over 15 calls at the converged point, each configuration in its own process,
persistent compilation cache disabled.

| | design vars | `values` | `jacobian` | fused | split (v+j) | fused saves |
|---|---|---|---|---|---|---|
| `stellarator_helias` MDF | 8 | 0.752 ms | 1.189 ms | **1.107 ms** | 1.941 ms | 42.9 % |
| `stellarator_helias` SAND | 14 | 0.880 ms | 0.975 ms | **0.913 ms** | 1.855 ms | 50.8 % |
| `large_tokamak_nof` MDF | 20 | 1.730 ms | 14.300 ms | **13.739 ms** | 16.030 ms | 14.3 % |
| `large_tokamak_nof` SAND | 27 | 1.040 ms | 5.529 ms | **5.560 ms** | 6.569 ms | 15.4 % |

**It is work, not overhead.** `jax`'s own bare-dispatch floor — a jitted `z * 2.0` on one
scalar, measured in the same process — is **0.016–0.024 ms**, two orders of magnitude
below `values`. There is no dispatch problem left at this boundary; §31.14's 14.5×
`bind` win already took it.

**The Jacobian is the cost, and forward mode is already doing well on it.** A tangent
costs about **0.4 of a primal evaluation** (tokamak MDF: 14.3 ms of Jacobian against
20 × 1.73 = 34.6 ms if each tangent cost a full primal), which is `vmap` amortising the
shared work across tangents. So the tokamak's `jacobian/values` ratio of 8.3 is near the
floor *for a 20-column Jacobian*; it is not slack.

**This reframes the "0.8 ms stellarator / 3 ms tokamak" floor targets.** Those are
*value* evaluation costs, and on values we are **at or below both**: 0.75–0.88 ms
stellarator, 1.04–1.73 ms tokamak. What exceeds them is the Jacobian, which is a
different quantity and scales with the design-variable count.

**Why `fused` helps the stellarator twice as much as the tokamak.** It removes exactly
one primal evaluation. Where the Jacobian is 1.1–1.6× the value (stellarator) that is
half the call; where it is 5–8× (tokamak) it is a seventh. Both are worth having, and
neither is where the remaining time is.

**The remaining time is host-side, and it is `cvxpy`.** A warm solve, XLA calls timed
against total wall:

| | warm wall | in XLA | calls | host |
|---|---|---|---|---|
| `stellarator_helias` MDF | 695 ms | 164 ms (24 %) | 81 | **531 ms** |
| `stellarator_helias` SAND | 639 ms | 87 ms (14 %) | 47 | **552 ms** |
| `large_tokamak_nof` MDF | 793 ms | 354 ms (45 %) | 13 | **439 ms** |
| `large_tokamak_nof` SAND | 571 ms | 146 ms (26 %) | 19 | **425 ms** |

`cProfile` names it: **`pyvmcon` rebuilds the `cvxpy` QP from scratch every SQP
iteration** — 41 `clarabel_conif.new_solver` constructions for 41 iterations, ~20 000
`cvxpy` canonicalisation calls in one solve. That is upstream of this port, and it is now
the largest single item in a warm solve on three of the four arms. A parametrised (DPP)
problem reused across iterations is the fix if it is ever worth taking.

**Per-call cost is point-dependent, so a solve average is not this table.** The same
tokamak MDF block averaged **27.2 ms** over the 13 calls of a real solve against 13.7 ms
median at the converged point: the block contains an inner solve whose trip count moves
with the iterate. Quote the median-at-a-point for comparing *programs* and the
solve-average for comparing *solves*; they are not the same number.

**Compile, for the record, on the same tree** (true cold, no persistent cache): 2 programs
above 100 ms on either MDF arm (16 896 + 8 465 lines = 9.5 s stellarator; 53 010 + 23 679
= 33.4 s tokamak), 6–7 on the SAND arms, and **every trivial compile put together is
186–449 ms**. §39's "not worth chasing" holds with the count now measured rather than
estimated.

## 42. SLSQP across the whole matrix, and two corrections (2026-09-05)

`next_steps.md` has carried *"the SLSQP driver has never been run across the full
seven-configuration matrix"* since it was written. It has now been run, and the run
corrected two things this file said.

**How it is run.** `run_cold_matrix.py --slsqp`, which changes the driver **class** and
nothing else — same assembly, same bounds, same tolerance, same iteration cap, same
seeding, same scoring, and the equality/inequality counts still read off
`Optimise.equalities`/`.inequalities` rather than counted by a caller. `optimiser` is
threaded through `mda.default_drivers`, `sand.sand_schedule`, `mdf.driver`/`mdf.solve`
and the harness as a *class*, deliberately: a caller handing in a built driver would have
had to count the conditions itself, which is the one thing §4.1 says never to do. The
published table is `reference_slsqp_matrix.txt`; its header names the optimiser on its own
line beside `SEEDED` and `SCORED`.

The two `i_process_run_mode = -2` rows (`large_tokamak_eval`, `spherical_tokamak_eval`)
come out **byte-identical** to the VMCON table, which is the check that the flag reaches
only `Optimise` blocks: those two state a `RootFind` and have no optimiser to change.

**Result: 9 of 12 converged, 1 hit the cap, 2 failed** — and where it converges it agrees
with VMCON on the answer while reaching *better* constraint residuals:

| | SLSQP it | VMCON it | SLSQP `max|eq|` | VMCON `max|eq|` |
|---|---|---|---|---|
| `stellarator_helias` MDF | **27** | 41 | 4.85e-10 | 5.56e-10 |
| `stellarator_helias` SAND | cap(500) | 24 | 3.95e-04 | 7.13e-06 |
| `helias_5b` MDF | **fails at it 1** | 4 | 1.16e-01 | 1.93e-12 |
| `helias_5b` SAND | **fails at it 1** | 7 | 3.92e+03 | 1.10e-13 |
| `large_tokamak_nof` MDF | 8 | 7 | **7.04e-12** | 2.40e-06 |
| `large_tokamak_nof` SAND | 13 | 10 | **1.39e-13** | 7.17e-06 |
| `low_aspect_ratio_DEMO` MDF | 12 | 11 | 4.44e-15 | 9.99e-15 |
| `low_aspect_ratio_DEMO` SAND | **17** | 79 | 1.70e-11 | 5.29e-13 |
| `st_regression` MDF | 11 | 10 | 7.64e-12 | 2.66e-15 |
| `st_regression` SAND | 10 | 10 | 8.00e-14 | 7.32e-13 |

The objectives agree to the printed digits everywhere both converge. **This is the oracle
the open item wanted**: on the SAND arms there is no PROCESS answer to compare against at
all, and two independent SQPs landing on the same optimum is the strongest evidence
available that the optimum is the problem's and not the solver's.

`helias_5b` is the one real failure and scipy says why: **"Singular matrix C in LSQ
subproblem"** at iteration 1, both arms. That is a rank-deficient constraint Jacobian at
the cold start, which VMCON's own QP happens to survive and scipy's does not. It is
evidence about the *problem*, not about scipy — the distinction this whole exercise
exists to draw — and it is a new open item.

**Host cost: `sqp` reads 0.0 s on all twelve SLSQP arms**, against 0.0-2.2 s for VMCON.
scipy's SLSQP is Fortran and allocates nothing per iteration; `pyvmcon` builds a fresh
`cvxpy` problem every iteration (§41). On `low_aspect_ratio_DEMO` SAND that is 2.2 s and
79 iterations against 0.0 s and 17.

**But SLSQP pays for it in evaluations, which §41 predicted and this measures.** Its line
search calls `fun` at points it never asks a derivative for, and `SlsqpDriver.at` computes
the Jacobian at *every* distinct point regardless — `nfev/nit` is 1.1-1.5 where it
converges and **7.0** on the capped stellarator SAND arm (3 518 evaluations for 500
iterations). At §41's 5.5-8.3x Jacobian-to-value ratio that is the expensive half. So the
two are not ranked: SLSQP is cheaper per iteration and can be far more expensive per
solve.

### Correction 1: `fused` does move the matrix, on two configurations

§40 recorded the seven-configuration matrix as **bit-for-bit identical** with `fused` on.
That was measured on a tree without §39's `mdf._not_weak`, and on today's `main` it is
false: **four cells move**, all in the residual columns, on `low_aspect_ratio_DEMO` and
`st_regression`.

| | with `fused` | pin (`fused` off) |
|---|---|---|
| `low_aspect_ratio_DEMO` MDF `max|eq|` | 7.33e-15 | 9.99e-15 |
| `low_aspect_ratio_DEMO` SAND `max|eq|` | 1.74e-13 | 5.29e-13 |
| `st_regression` MDF `max|eq|` | 2.89e-15 | 2.66e-15 |
| `st_regression` SAND `max|eq|` / `min ie` | 7.32e-13 / 3.71e-13 | 2.90e-13 / 3.73e-13 |

Every other cell — every `status`, `SQP` count, `objf` and `worst dx` on all twelve
rows — is unchanged, and the moved cells are residuals at `1e-13`-`1e-15`, orders below
any tolerance the table applies. So the *conclusion* of §40 stands and the default stays;
what was wrong is the word "identical".

**Ruled out before concluding, because three explanations were available.** Two uncached
runs of `st_regression` agree with each other to the last digit, so it is not run-to-run
nondeterminism. A run *with* a fresh persistent cache reproduces the uncached numbers, so
it is not the compilation cache — which also **falsifies `main`'s own docstring claim of
"bitwise-identical rows" under `--cache`** in the other direction: the cache is innocent
here, and the claim was never tested this way. Setting `fused = False` on today's tree
reproduces the pin exactly. That is the variable.

`reference_cold_matrix.txt` is therefore re-baselined on today's `main`, which is what it
is for: the file states the tree's own measurement, and it had stopped doing so.

### Correction 2: a `nan` convergence value turned every SLSQP success into a failure

`SlsqpDriver`'s callback had to grow `VmconDriver`'s four-argument shape for
`run_cold_matrix._recorder` to read it. The fourth argument is VMCON's convergence
parameter, which scipy forms no equivalent of, and the first attempt passed `nan` as the
honest answer.

It is not the honest answer. `_status` compares that number to a tolerance and nothing
else reads it, and `nan <= tol` is `False` — so **every SLSQP row printed `stopped`,
including the nine where scipy had said "Optimization terminated successfully"** and the
residuals were up to five orders better than VMCON's. The first table generated from this
flag said SLSQP converged on none of twelve rows; it converges on nine. The final callback
now carries `0.0` for `result.success` and `inf` otherwise, documented at the call site as
an encoding of scipy's verdict and not a convergence measure.

The general lesson is worth the paragraph: a sentinel is only honest if every consumer
reads it as one. `nan` chosen for caution produced a confidently wrong table, which is the
failure mode this port's whole "refuse rather than report healthy" discipline exists to
avoid — and it produced the *opposite* error, reporting sick when healthy, which is
cheaper but not free.

### Correction 3: "SLSQP must not fuse" is an argument about the wrong thing

Stated in `scaled_problem`'s docstring, in `SlsqpDriver`'s own `at`-cache comment, and
(added earlier the same day) in `VmconDriver._Problem.__call__`: scipy's SLSQP calls `fun`
alone during its line search, so a fused program would pay a whole Jacobian at every trial
point.

True of what scipy **asks** for. False of what the driver **computes**: `at` caches
`(evaluate(x), jacobian(x))` at every distinct point, derivative requested or not. So the
saving the split pair exists to protect was never being taken, and fusing would be
*cheaper* for SLSQP too — §41's table has the split at 16.03 ms against 13.74 ms fused on
`large_tokamak_nof` MDF.

scipy's own counters measure the waste, and it is not uniform: `nfev/nit` is 1.1-1.5 where
SLSQP converges, so the extra Jacobians are a rounding error there and nobody noticed. On
the capped `stellarator_helias` SAND arm it is 7.0 — `nfev 3518` against `njev 501`, about
**3 000 Jacobians nobody asked for**, ~5.4 s of a 22.9 s arm.

The repair is a **lazy `at`** rather than a fused one, and it is not made here: it changes
what the driver evaluates, so every row of `reference_slsqp_matrix.txt` would have to be
re-measured. Filed in `next_steps.md`. All three comments are corrected to say what is
actually true, which is that `fused` is a per-driver field because the two drivers may
want different answers — not because the answer is settled for the other one.

## 43. The lazy `at`, and where the evaluation floor actually is (2026-09-05)

§42 filed a lazy `SlsqpDriver.at` as future work. It is done, and it turned out to be two
defects rather than one.

**Defect 1, the one that was filed.** `at` cached `(evaluate(x), jacobian(x))` together, so
every point scipy's line search touched derived a Jacobian nobody asked for. Split into
`values_at`/`jacobian_at` over a two-slot cache (two, because the line search walks past
the accepted iterate before the callback runs, and one slot would evict exactly the point
the callback asks about).

**Defect 2, found only by counting programs afterwards.** The per-iterate callback built a
full `pyvmcon.Result`, which meant a Jacobian at every accepted iterate — and **no callback
in this tree reads `df`, `deq` or `die`**. `run_cold_matrix._recorder` reads `f`, `eq`,
`ie` and nothing else. So the callback was, by itself, doubling the derivative cost of a
solve. Replaced by `_Iterate`, duck-compatible with `Result` but with the three derivative
attributes as properties.

Counted on `stellarator_helias`, both arms, `--native`, by wrapping `host_cache.bind`:

| | Jacobian programs run | scipy's `njev` |
|---|---|---|
| before | ~3 557 (one per distinct point) | 527 |
| after defect 1 | 1 014 | 527 |
| after defect 2 | **527** | 527 |

Exactly what was asked for, and no more. **The answers are unchanged**: all twelve rows of
`reference_slsqp_matrix.txt` are identical before and after.

**What it bought, and a correction to §42's estimate.** §42 put the waste at ~5.4 s of the
capped `stellarator_helias` SAND arm, by multiplying ~3 000 spared Jacobians by §41's
1.8 ms. Measured, that arm's wall clock went **22.2 s to 19.5 s** and its `model` phase
3.2 s to 2.3 s. Real, and smaller than the arithmetic — per-call cost inside a solve is not
the median-at-a-point §41 measured, which is §41's own warning applied to me. Every other
arm moves within run-to-run noise, exactly as the `nfev/nit` of 1.1-1.5 predicts.

### Are we at the evaluation floor?

Three separate floors, and the honest answer differs for each.

**1. Redundant work: yes, this is the floor now.** Nothing computes a value or a
derivative that its caller did not ask for, on either driver. VMCON asks for both at every
point and gets them from one fused program (§40); SLSQP asks for value-only at line-search
points and gets exactly that. There is no third thing to remove — the next reduction has to
come from asking for less, not from wasting less.

**2. Per-call cost: near the floor for the shape of the problem.** §41 measured jax's
bare-dispatch floor at 0.016-0.024 ms against a `values` call of 0.75-1.73 ms, so the
boundary is free. A forward-mode tangent runs at ~0.4 of a primal evaluation, which is
`vmap` amortising about as well as it can. What is left is the *program*: 53 010 emitted
lines for one tokamak MDF block. That is the lever for both compile time and per-call time,
and it is untouched.

**3. Whole-solve evaluation cost: no, and the gap is not evaluation.** A warm solve is
0.36-0.79 s, of which XLA is 87-354 ms (§41). The stellarator arms are **already at
87-164 ms of actual evaluation** — at or under a 100 ms target — and the tokamak MDF arm is
178 ms at the converged-point median (354 ms averaged over a real solve, whose calls are
not all at the converged point). So the evaluation half is close to done. The other
425-552 ms is host: `pyvmcon` rebuilding its `cvxpy` QP every iteration, and
`host_cache.call`'s own 1.8 ms and `__eq__`'s 0.5 ms per call. **Getting a solve to 100 ms
is a host-side problem now, not an evaluation one** — and the SLSQP arms, whose `sqp` phase
reads 0.0 s on all twelve, are the evidence that most of that host cost is one library's
allocation pattern rather than anything structural.

## 44. The `model` column is not evaluation time (2026-09-06)

The cold matrix's `model` column reads 1.5 s for `stellarator_helias` MDF under VMCON,
against 41 SQP iterations. That invites the reading "41 iterations at ~35 ms each", and
it is wrong: **the median call is 1.000 ms and the p90 is 1.2 ms.**

Measured by wrapping `host_cache.bind`'s programs for a whole row (both arms,
`stellarator_helias`, `--native`) and keeping every call time rather than a mean:

| | |
|---|---|
| calls | 128 |
| median | **1.000 ms** |
| p90 | 1.2 ms |
| max | 9 108.7 ms |
| calls over 10 ms | **2 of 128** |
| time held by those 2 | **15.79 s of 15.92 s** |

The two are the first call of each arm, carrying that arm's compile. Averaging over them
gives 53.6 ms/call, which is what a naive reading of the column reproduces and which
describes no call that actually happened. The same programs, called 15 times at the row's
final point in the same process, run at **0.732 / 0.929 / 0.763 ms** (`values` /
`jacobian` / `values_and_jacobian`) — §41's numbers, confirmed from inside a real solve
rather than beside one.

**So what is in `model`, if the arithmetic is 0.13 s?** Splitting every call into its
phases (`phase_timing.totals()` sampled either side of each call) answers it exactly. Of
128 calls, only two are not ~1.4 ms:

| call | wall | trace | lower | compile | **rest** |
|---|---|---|---|---|---|
| first, MDF arm | 9 194 ms | 1 369 | 464 | 6 013 | **1 348** |
| first, SAND arm | 6 662 ms | 406 | 287 | 5 233 | **737** |
| each of the other 126 | 1.4 ms | 0 | 0 | 0 | 1.4 |

`rest` over all 128 calls is **2.21 s**, and the 126 ordinary calls hold **0.130 s** of it.
So `model` is 2.08 s of first-call `rest` plus 0.13 s of evaluation, which is the 2.6 s
the two arms report.

**And `rest` is not "Python overhead" — it is trace and lower work the two patches cannot
see.** `cProfile` of that first call, by `tottime`: `jax._src.core.bind` (40 396 calls,
4.67 s cumulative), `partial_eval.default_process_primitive` (19 311),
`batching.process_primitive` (`jacfwd`'s `vmap`, 2.52 s cumulative),
`core.eval_jaxpr` (847 calls, 2.98 s cumulative) and `mlir.jaxpr_subcomp` (0.81 s).
`phase_timing` patches `trace_to_jaxpr_dynamic` and `lower_jaxpr_to_module`; `jacfwd`
re-enters primitive binding and `eval_jaxpr` *outside* those entry points — §31's finding
that `jvp_jaxpr` re-interprets under the same tracer stack rather than rewriting a jaxpr,
showing up here as mis-attributed time.

**So the table under-reports trace/lower and over-reports `model`.** On `stellarator_helias`
the honest split of a 13.6 s cold row is roughly **4.4 s trace+lower, 8.3 s compile,
0.13 s arithmetic** — about **97 % compilation in the broad sense and 1 % evaluation**,
where the column suggested 2.6 s of "model". The matrix header has always labelled
`model`/`other` **UNVERIFIED** for exactly this reason; this is what that warning was
worth.

**Consequence for how this table is read.** A row's `model` figure does not scale with
iteration count and must not be divided by one. To compare *solvers* use the iteration
count and `sqp`; to compare *programs* use §41's median-at-a-point; to compare *rows* use
the total. §43's own warning — "per-call cost is point-dependent, so a solve average is
not this table" — was about a factor of two between two honest measurements. This is the
larger version of the same mistake: a mean over a distribution whose top two samples are
compiles.


## 45. Where the memory goes: two programs a row, and a floor that saturates (2026-09-06)

`next_steps.md`'s open item 3 asked for a measurement before a theory. A whole-matrix
pass had died with `LLVM ERROR: Unable to allocate section memory` on a request for
**63 bytes**, so the failure was XLA's CPU section allocator having nothing left to map
rather than one oversized program, and what was unknown was the resident cost **per
compiled program**, whether `jax.clear_caches()` returns it, and why a few dozen programs
a configuration exhaust a 15 GB box.

Measured by patching the one entry point `phase_timing` already names for compilation
(`compiler.backend_compile_and_load`) and recording, per program, the `VmRSS` delta
across the compile call and the character count of the StableHLO module handed to it,
then running `run_warm_matrix.measure` per row with `VmRSS` sampled before, after, and
after `clear_caches` + `gc.collect()` + `malloc_trim(0)`. Probe:
`_audit/rss_per_program.py`. VMCON, all seven configurations in one process each time, in
`CONFIGURATIONS` order (MB; `large_tokamak_eval` and `spherical_tokamak_eval` state no
SAND arm):

| configuration | arm | programs | grown by compiles | before | after | **trimmed** | HWM |
|---|---|---:|---:|---:|---:|---:|---:|
| `stellarator_helias` | MDF | 26 | 621 | 392 | 1091 | 594 | 1091 |
| `helias_5b` | MDF | 24 | 408 | 594 | 1027 | 605 | 1091 |
| `large_tokamak_nof` | MDF | 24 | 1221 | 605 | 1954 | 752 | 1955 |
| `large_tokamak_eval` | MDF | 20 | 703 | 752 | 1511 | 796 | 1955 |
| `low_aspect_ratio_DEMO` | MDF | 24 | 1217 | 796 | 2113 | 845 | 2156 |
| `spherical_tokamak_eval` | MDF | 17 | 709 | 845 | 1602 | 879 | 2156 |
| `st_regression` | MDF | 21 | 796 | 879 | 1727 | **851** | 2156 |
| `stellarator_helias` | SAND | 33 | 616 | 392 | 1087 | 575 | 1087 |
| `helias_5b` | SAND | 34 | 374 | 575 | 971 | 580 | 1087 |
| `large_tokamak_nof` | SAND | 40 | 1377 | 580 | 2091 | 712 | 2091 |
| `low_aspect_ratio_DEMO` | SAND | 40 | 1419 | 717 | 2209 | 769 | 2226 |
| `st_regression` | SAND | 34 | 957 | 770 | 1766 | **766** | 2226 |

**1. Two programs a row hold almost all of it.** Every row's growth is concentrated in
two multi-megabyte modules: on `st_regression` MDF they are 447 MB and 444 MB of a 920 MB
row, and the other 19 programs cost 29 MB between them. The small programs -- the 200-450
character scalar reshapes §39 counted -- cost **1-2 MB each regardless of size**, which is
allocator granularity rather than module content. That per-program floor is ~30 MB at 20
programs a row: not the problem.

**2. The floor saturates; it does not leak.** Post-trim, MDF goes
594 -> 605 -> 752 -> 796 -> 845 -> 879 -> **851** and SAND
575 -> 580 -> 712 -> 769 -> **766**: both climb for three or four rows and then flatten,
with the last row *below* the one before it. Peak RSS over a seven-configuration pass is
**2.16 GB** (MDF) and **2.23 GB** (SAND). So `clear_caches` + `malloc_trim` does bound the
pass, the residue is a plateau and not an accumulation, and the original OOM is explained
by its absence: `run_warm_matrix` runs **four** measurements per configuration (two arms x
two drivers), and without the trim each leaves its ~1 GB resident, which reaches 15 GB
somewhere in the fourth configuration -- exactly where it died. `large_tokamak_nof` SAND,
the row named in that failure, now peaks at 2.09 GB and trims back to 712 MB.

### Correction: StableHLO character count is not the predictor it was taken for

§31.16 is quoted in `tried_and_rejected.md` as **~200 bytes of peak RSS per character of
pre-optimisation StableHLO**, and this measurement says that constant is an average over
one *class* of module rather than a property of modules. The 24 large programs here split
cleanly in two:

- **expensive**, 97-437 B/char (median ~262) -- the two per row that carry the growth;
- **cheap**, 5-46 B/char -- a third multi-megabyte module present in every SAND row.
  `large_tokamak_nof` SAND lowers a **1 331 233**-character module that costs **7.9 MB**,
  next to a 3 143 770-character one that costs **934.5 MB**. That is a 50x spread in
  B/char between two modules in the same row.

So a module's character count predicts its resident cost only within a class, and the
~200 B/char figure should be read as "the expensive class averages ~260, and some large
modules cost almost nothing" -- not as a rate to multiply a row's total HLO by. What
distinguishes the two classes was not investigated; the cheap ones are the SAND-only
third module, which is a lead, not an answer.

**What this closes and what it does not.** Item 3 is answered: the mechanism is resident
compiled executables, the workaround already in both runners is the correct fix rather
than a papering-over, and a pass is bounded with it at ~2.2 GB. What it does *not* say is
that a row's 1 GB is necessary. The lever is the two expensive modules, which is a
lowering question (§31.2's ~52 StableHLO lines per assembled node), not an allocator one.

## 46. `helias_5b`'s singular LSQ subproblem: an equality with nothing to solve (2026-09-06)

`next_steps.md`'s open item 1: `helias_5b` fails under SLSQP on **both** arms at iteration
1 with scipy status 6, *"Singular matrix C in LSQ subproblem"*, while VMCON converges the
same configuration. Nobody had looked at **which** constraints are dependent. Diagnosed
by hooking `drivers.scaled_problem` -- the one builder both drivers go through -- and
reading the Jacobian at the first point either driver evaluates.

**One row, and it is exactly zero.** MDF arm, three design variables
(`ixc = 4, 6, 10` -> `.physics.temp_plasma_electron_vol_avg_kev`,
`.physics.nd_plasma_electrons_vol_avg`, `.physics.hfact`), six condition rows:

| condition | \|row\| | value |
|---|---:|---:|
| `^cond.numerics.objf` | 2.010e-01 | +7.583e-01 |
| `^cond.constraints.c2` | 8.275e-01 | -6.870e-03 |
| **`^cond.constraints.c11`** | **0.000e+00** | **+1.110e-16** |
| `^cond.constraints.c16` | 3.932e+00 | +1.161e-01 |
| `^cond.constraints.c84` | 6.828e+00 | -3.452e+00 |
| `^cond.constraints.c24` | 1.366e+00 | -1.097e-01 |

`c11`'s three entries are `'0.0', '0.0', '0.0'` by `repr`, not "small" -- **exact**, not a
conditioning artefact. Its residual is already `1.1e-16`, so the constraint is satisfied
and carries no gradient. The **equality block alone** -- `[c2, c11, c16]`, which is the
matrix scipy factorises for its LSQ search direction -- is therefore 3x3 of **rank 2**,
smallest singular value 4.1e-17. SAND is the same story one size up: 9x9, rank 8,
smallest singular value 3.3e-15, same row. The **full** matrix stays full column rank in
both arms (MDF 6x3 rank 3, singular values 8.00 / 0.791 / 3.15e-02), which is why the zero
row is invisible to anything that does not factorise the equalities on their own.

**Why the row is zero, structurally.** PROCESS constraint 11 is the radial-build
consistency equation, `rbld == rmajor`. In `helias_5b`'s graph `.physics.rmajor` is in
`graph.unowned_inputs` -- a boundary input, a constant for this configuration -- and
`.build.rbld` is produced by `.stellarator.build` from inputs disjoint from all three
iteration variables. So the residual is constant with respect to every column the problem
has. The zero row is guaranteed by construction.

**And it is the input file's defect, faithfully ported.** `helias_5b.IN.DAT` lists
`icc = 11` and does **not** list `ixc = 3` (`rmajor`, `iteration_variables.py:47`). Its
sibling `stellarator_helias.IN.DAT` is the exact complement: `ixc = 3` present, `icc = 11`
absent. So one file gives the optimiser the major radius as a design variable without the
consistency equation that would pin it, and the other states the consistency equation
without the variable it exists to determine. PROCESS itself sees the same zero row --
finite differences over the same three `ixc` move that residual no more than AD does --
and its VMCON tolerates it, so nothing flagged it in twenty years of use.

**Half of this was already written down, and the link was not.**
`drivers._refuse_inert_objective`'s docstring names this very row -- *"`helias_5b`'s
equality 11 compares a radial build against `.physics.rmajor` on a file whose three
iteration variables are the temperature, the density and `hfact`"* -- as its example of a
zero row that is deliberately **named and not refused on**, because a constraint the
design cannot steer is common and often intended. So the zero row was known. What was not
known is that it is the whole of scipy's *"Singular matrix C in LSQ subproblem"*, and
therefore that "named and not refused on" has a cost: it is fine for VMCON and fatal for
SLSQP. The policy is still right -- refusing here would fail a configuration VMCON solves
correctly -- but the diagnosis should reach the user through the error rather than through
this file.

**Verdict: SLSQP is right and VMCON is permissive.** `pyvmcon` reaches its step through a
`cvxpy` QP that never needs the equality Jacobian to be independent on its own, and the
constraint being already satisfied means nothing forces the issue; scipy's SLSQP
factorises that block densely and aborts the moment it meets a zero row. The failure is a
solver exposing a pre-existing structural defect in the problem statement, not a port bug
and not a scaling bug -- **an equality that is present in name only.**

**Secondary, and not the cause.** In the SAND arm the inequality rows `c84` (lower beta
limit) and `c24` (upper beta limit) are exactly anti-parallel, `cos = -1.0` -- they are
two bounds on the same `beta_total_vol_avg`, and the scaled rows differ by a factor of
exactly 5. Neither is near its bound at the cold start (`-3.452` and `-0.110`), so neither
is plausibly in SLSQP's working set at the failing iteration.

**Rejected on the way: an apparent rank collapse in the *unscaled* SAND Jacobian**
(rank 3 of 9). The unknown columns span ~20 orders of magnitude in absolute units
(`nd_plasma_electrons_vol_avg` column norm 7e-3 against
`temp_plasma_ion_vol_avg_kev`'s 2.6e17), so `numpy.linalg.matrix_rank`'s default tolerance
swamps genuine O(1) singular values. A units artefact in the diagnostic, not a second
degeneracy -- the scaled matrix, which is what both drivers actually condition on, has
only the one exact zero row.

## 47. `stellarator_helias` SAND under SLSQP: a period-2 cycle on one residual (2026-09-06)

`next_steps.md`'s open item 2: this arm hits the 500-iteration cap at 4 019 block calls
where VMCON converges in 24, with `nfev/nit` of 7.0 against 1.1-1.5 everywhere SLSQP
converges. The question was what the line search is failing to make progress on.
Instrumented by monkey-patching `run_cold_matrix._recorder` and `SAND_MAX_ITER`
in-process and running the existing `build_sand`/`solve_sand` capped at 120 iterations
(33-36 s), which reproduces the ratio: **785 evaluations / 120 iterations = 6.54**.

**It oscillates *and* creeps, in ramp-and-reset cycles.** The per-iteration evaluation
count is not flat noise -- it climbs from 1-2 to 10-11 and then resets, with resets at
iterations ~18, ~39, ~62 and ~102. Inside each ramp, one equality residual and two
inequalities lock into a clean **period-2 zigzag**: `^cond.stellarator.wp_width_r_min`
alternates `-0.0152, -0.0052, -0.0163, -0.0050, -0.0144, -0.0046, ...` while `c62`
alternates feasible (~0) and violated (~0.03-0.04) on the opposite parity, with `c35`
along for the ride. The envelope does shrink -- the low branch goes -0.0163 -> -0.0144 ->
... -> -0.0067 over ~20 iterations -- so this is not a stalled cycle, it is a decaying one
at a rate nowhere near reaching 1e-8 within 500 steps. The objective drifts inside
`[1.2182, 1.2188]` for the entire tail, which is why the capped row still reports
`objf = 1.21846128` against the converged `1.21848284`.

`wp_width_r_min` is a named suspect independently: `drivers.py`'s own docstring flags it
as *"the only one whose units are genuinely not its unknown's"*.

**Three candidate causes, all measured and all rejected.**

- **Not a bound.** None of the 8 bounded design unknowns approach a bound across 120
  iterations; the closest is `f_nd_alpha_thermal_electron` at 6.4 % of its range, and
  most sit at 15-85 %. An active-set thrash on a bound was the obvious reading of
  "7 evaluations per iteration" and it is wrong.
- **Not the Jacobian.** Central finite differences (`h = 1e-6`, all 14 columns) against
  the analytic `jacfwd` Jacobian at three points along both the SLSQP and the VMCON
  trajectory agree to a max relative error of 1e-4-1e-6, median 0 -- FD truncation, not a
  defect. Expected, and worth the five minutes to remove from the board.
- **Not the scale spread, which was the leading hypothesis and is falsified by its own
  control.** Verified independently by capturing `driver.condition_scale` and
  `scaled_problem`'s design `scale` on both configurations' SAND arms:

  | configuration | SLSQP | design scale ratio | condition scale ratio |
  |---|---|---:|---:|
  | `stellarator_helias` | cap(500), 501 it | 2.18e+22 | 7.49e+19 |
  | `large_tokamak_nof` | converged, 13 it | **1.97e+23** | 3.51e+18 |

  The configuration that converges in thirteen iterations has a design-scale spread an
  order of magnitude **wider**. A large ratio is therefore not sufficient, and "SLSQP's
  single penalty parameter cannot serve conditions spread over twenty decades" -- stated
  in this file as the thing to test -- does not survive being tested.

**What the contrast with VMCON actually shows.** VMCON hits the *same* conditions, harder:
at iteration 2, after the largest step of its run (`||dx|| = 0.92`), `wp_width_r_min`
spikes to **52.46** and `c62` to 2.1-2.8. It then walks them down with further large,
well-aimed steps -- `wp_width_r_min`: 52.46, -0.47, 0.002, -0.028, -0.43, -0.003, 0.0005 --
and by iteration ~13 both are at 1e-4-1e-5 with `||dx||` damping geometrically (0.018,
0.012, 0.012, 0.0045, 0.0016, ... 2e-4 by iteration 24). SLSQP never takes that step. The
paths separate at iteration 1-2, and from there SLSQP makes small steps that settle into
the decaying cycle above.

**Hypothesis, labelled as one.** The mechanism consistent with all of the above is that
`wp_width_r_min` and `c62` are a genuinely *conflicting* pair -- improving one worsens the
other -- and that SLSQP's single scalar merit-function penalty cannot authorise the large
step that trades them off, where VMCON's per-constraint Lagrange multipliers can. That is
a statement about the pair, not about the magnitude of any scale factor, and it explains
why the scale-spread control comes out the way it does: `large_tokamak_nof` has worse
spread and no such pair. **It is not established.** The test that would establish it is to
re-run with `c62` dropped, or with the two conditions rescaled relative to each other, and
see whether the cycle disappears; that has not been done.
