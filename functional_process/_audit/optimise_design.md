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

> **WITHDRAWN the same day -- see §50.** The two classes below are an artefact of
> compile *order*, not a property of modules: RSS grows only when glibc maps new memory,
> so a program compiled into arena an earlier one freed reads as "cheap" while producing
> an equally large executable. Re-measured with the arena trimmed before each compile,
> every module lands in one band at 171-351 B/char. §31.16's ~200 B/char stands. The
> paragraphs below are kept because the retraction is the point.

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

## 48. `helias_5b`: which repair, measured -- and a false negative in §46's own diagnostic (2026-09-06)

§46 established that `helias_5b.IN.DAT` states `icc = 11` (`rbld == rmajor`) without
`ixc = 3` (`rmajor`), leaving an equality no design variable can move. Two repairs
suggest themselves and only one of them works. Both were built as scratchpad copies and
run on both arms under both drivers -- eight cells, none skipped.

- **Variant A -- add the variable.** `ixc = 3` inserted with `boundl(3) = 10.0`,
  `boundu(3) = 30.0` (`stellarator_helias.IN.DAT`'s own convention).
- **Variant B -- drop the equation.** The `icc = 11` line deleted and
  `n_equality_constraints` taken 3 -> 2.

| variant | arm | driver | status | it | `objf` (9 s.f.) |
|---|---|---|---|---:|---:|
| baseline | MDF | VMCON | converged | 4 | 0.764215516 |
| baseline | SAND | VMCON | converged | 7 | 0.764215517 |
| baseline | MDF | SLSQP | stopped | 2 | -- |
| baseline | SAND | SLSQP | stopped | 2 | -- |
| **A** | MDF | VMCON | converged | 11 | **0.721837749** |
| **A** | SAND | VMCON | converged | 12 | **0.721837749** |
| **A** | MDF | SLSQP | **stopped** | 2 | -- |
| **A** | SAND | SLSQP | **stopped** | 2 | -- |
| **B** | MDF | VMCON | converged | 4 | **0.764215516** |
| **B** | SAND | VMCON | converged | 7 | **0.764215517** |
| **B** | MDF | SLSQP | **converged** | 5 | **0.764215516** |
| **B** | SAND | SLSQP | **converged** | 8 | **0.764215517** |

(The `B` and baseline rows above were re-run independently before this was written; the
figures are that re-run's, and they reproduced the agent's to every digit shown.)

**Variant B is the repair.** SLSQP converges on both arms, to the same optimum VMCON
reaches, and VMCON's own answer and iteration count are *unchanged* -- 4 and 7, matching
`reference_warm_matrix.txt` exactly. Dropping an equality whose residual was already
1.1e-16 and whose row was already zero is a genuine no-op on the physics. That is the
recommendation to take upstream: `icc = 11` should not be listed on a configuration that
carries no design variable able to move it.

**Variant A fails on both counts, and the second one is the interesting one.**

1. **It does not fix SLSQP.** Still `stopped` at iteration 2 on both arms. With `rmajor`
   promoted to a design variable, `c11`'s row reads `[-1.6e-16, -0.0, -0.0, -0.0]` scaled
   -- sixteen orders below its siblings, and the equality block is still rank-deficient
   on that same row (smallest left singular vector still `u = [0, -1, 0]` in
   `[c2, c11, c16]` order). `rmajor` demonstrably *does* enter the differentiable path
   elsewhere: the objective, `c2` and `c16` carry `rmajor` columns of 1.07, 0.85 and
   -3.75. So promoting the variable does not connect it to *this* residual. Whatever
   `.Constraint11` reads `rbld` through is not reached by the promotion. **Not
   root-caused** -- reproducible, and left open.
2. **It changes the machine.** VMCON converges to `objf = 0.721837749` on both arms, a
   5.6 % drop, with `rmajor` moving 22.0 -> **20.911 m** (4.9 %, nowhere near its bounds)
   and pulling the rest with it (`temp` 7.15 -> 6.567 keV, `hfact` 1.219 -> 1.232). Even
   had it fixed SLSQP it would have been answering a different design problem.

### Correction: §46's diagnostic tested for exact zero, and variant A walked through it

`_name_singular_equalities`, landed the same day, listed a row only when
`not np.any(row != 0.0)`. On variant A it therefore reported *"no zero row, but rank 8 of
9"* -- technically true, and a **false negative** on the very case the function exists
for: a row at 1e-16 is inert in every sense that matters and the block was still singular
on it. AD roundoff is enough to defeat an exact test, and an exactness that was a genuine
finding in §46 (the *baseline* row really is bit-exact zero) does not generalise to a
test. Now compared against the block's largest entry at `1e-10` -- unit-free, and far
below the 1e-1 to 1e+1 the live rows carry. Re-checked: baseline still names `c11` on
both arms, variant B warns not at all, and variant A now names it.

## 49. The `c62` experiment: the pair is real, and rescaling does not fix it (2026-09-06)

§47 named one hypothesis and the experiment that would settle it: `wp_width_r_min` and
`c62` are a conflicting pair, and SLSQP's single merit penalty cannot authorise the
trade-off step VMCON takes. Run, on a scratchpad copy of `stellarator_helias.IN.DAT` with
the `icc = 62` line deleted, and with runtime overrides of
`sand.residual_condition_scales`.

**Dropping `c62` ends the cycle outright.** `wp_width_r_min` descends smoothly from -0.18
through -0.40 over ~9 iterations, then drops in **one step** to -1.7e-4 and decays to
-4.6e-10 -- the same shape VMCON takes on the intact problem, not the back-and-forth
SLSQP otherwise takes. 91 iterations, `nfev/nit` 6.54 -> 5.31, `max|eq| = 4.6e-10`. So
`c62` is confirmed as the antagonist: remove it and the zigzag is gone immediately.

**But it is not a fix, it is a different problem.** `objf = 1.22231038` against the intact
`1.21848284` -- **0.31 % away**, and scipy's own `success` flag never turns `True` even at
those residuals. §47 asked for exactly this check and this is the answer it warned about:
an intervention that converges by relaxing the problem has not fixed anything.

**And rescaling does not work either, which kills the cheap fix.** `wp_width_r_min` gets
1.591 from the natural `1/|u|` rule; `c62` is an ordinary PROCESS inequality with no entry
in `residual_condition_scales` at all, so it defaults to 1.0. Two targeted overrides, two
orders of magnitude, opposite targets:

| override | `nfev/nit` | status | tail |
|---|---:|---|---|
| `c62` 1.0 -> 50 | 6.38 | still caps | sharp excursions (-0.33, -0.069) mixed with small values |
| `wp_width_r_min` 1.591 -> 159.1 | 5.58 | still caps | large excursions (-7.0, -4.4, -3.8, -3.6) persist |

Two points is not a sweep and the agent said so. But if this were a pure row-scaling
artefact, a 100x push on either row should have found daylight, and neither did. **The
"just rescale it" reading is not supported.**

**What `c62` is, and what it shares with `wp_width_r_min`: nothing physical.**
`constraint_equation_62` (`process/core/solver/constraints.py:1393-1409`) is a lower limit
on `f_t_alpha_energy_confinement`, the ratio of alpha-particle to global energy
confinement time -- a helium-ash thermalisation floor. `wp_width_r_min` is a TF-coil
winding-pack geometric root find (`stellarator/namespace.py:246-249`, an `Intersect`).
They share no quantity; the interaction is through the design vector and through SAND's
formulation.

**Verdict, and it is architectural rather than numerical.** `wp_width_r_min` is a
**SAND-only** exposure of an inner root find that MDF solves to convergence on every
evaluation -- which is why the same file under MDF converges under SLSQP in 27 iterations
and only SAND caps. SAND hands that root find to the outer SQP as one more equality,
competing in the same merit function against `c62` at a point where both are near-active.
VMCON's per-constraint multipliers resolve the corner in one large step (`||dx|| = 0.92`
at iteration 2, §47); SLSQP's single scalar penalty cannot, and cycles. So §47's
hypothesis survives at the level of *which* pair and *why they fight*, and its suggested
remedy does not: this is not reachable by a condition-scale tweak, and the lever -- if one
is wanted -- is whether SAND should expose this particular coupling to the outer solver at
all. **Not attempted**, and worth noting that the current behaviour costs one row of
twenty-four under a driver that is not the production one.

## 50. What is in a 2.3-million-character module, and what comes out (2026-09-06)

> **Two claims here are CORRECTED by §57**: `serialize()` returns the optimised HLO,
> not machine code (so "the executable serialises to 11 MB" is measuring the wrong
> object), and the per-program resident figures are allocator-order artefacts. The
> op histogram and the optimised-HLO expansion stand.

§45 measured memory per compiled program without ever looking inside one. Two questions
follow directly and both have clean answers: what is the intermediate representation
*doing* with millions of characters, and how big is the thing it produces. Probe:
`compiler.backend_compile_and_load` hooked as in `_audit/rss_per_program.py`, plus an op
histogram of the module text and the executable's own accessors. `st_regression` SAND,
34 programs, largest module.

**The IR is 28 106 operations, spelled verbosely.** 2 282 899 characters / 29 128 lines /
28 106 ops -- about **81 characters per op**, one op per line. Nothing pathological: it is
MLIR's textual form, where every line carries full tensor types. A model with 28k
operations is simply a big model, and 2.3 M characters is what 28k typed lines costs.

**But 41.6 % of those ops are not arithmetic.** The mix:

| op | count | share |
|---|---:|---:|
| `broadcast_in_dim` | 7 843 | 27.9 % |
| `multiply` | 5 725 | 20.4 % |
| `constant` | 3 985 | 14.2 % |
| `add` | 2 382 | 8.5 % |
| `slice` | 1 684 | 6.0 % |
| `divide` | 1 392 | 5.0 % |
| `reshape` | 1 112 | 4.0 % |
| `subtract` / `convert` / `select` / `negate` / `compare` | 2 711 | 9.6 % |
| **pure data movement and shape** | **11 682** | **41.6 %** |

More than a quarter of the module is `broadcast_in_dim`. That is the signature of a
**scalar-valued graph expressed in an array language**: nearly every node computes a
scalar, and each one is wrapped into a rank-0 or rank-1 tensor to meet the next op's
type. Constants are another 14.2 % of ops and 11.8 % of the text. So the honest reading of
"3 million characters" is *not* "the model is enormous" -- it is **28k scalar operations,
two fifths of which are shape plumbing**, and that plumbing is the part a vectorisation
pass could actually remove (§31.2's 55 -> 52 StableHLO lines per node was a first bite).

**XLA's optimised HLO is bigger, not smaller: 9 648 248 characters, 422 % of the input.**
Fusion does not shrink the text; it expands it, because the optimised module prints each
fusion's sub-computation. Post-optimisation text size is therefore not a proxy for
anything, and the other four large programs land at a consistent 260 %.

**The executable serialises to 11.0 MB** -- all 34 programs together, 21.1 MB. XLA's
`size_of_generated_code_in_bytes()` returns **0** on the CPU backend, so it is not the
field to read here; `serialize()` is.

**And yet compiling it costs 760 MB, of which loading it back costs ~500.** The decisive
measurement: capture that 11.0 MB blob during a real solve, then deserialise it three
times in the same process, trimming between:

| | RSS | time |
|---|---:|---:|
| compile from StableHLO | 760.2 MB | 13.80 s |
| load #1 from the 11.0 MB blob | 476.9 MB | 1.86 s |
| load #2 | 538.9 MB | 1.65 s |
| load #3 | 557.2 MB | 1.67 s |

Each load costs about two thirds of a full compile and roughly **45x its own serialised
size**, and three loads cost three times over. So the resident cost is **the executable**,
not the compiler's workspace -- §31.16's original claim, and the one §45 muddied. (The
1.7 s and 500 MB per load say deserialisation is doing real codegen rather than mapping a
finished binary, which is consistent with the failure mode being LLVM's *section memory*.)

### Correction: §45's "two classes of module" was an allocator artefact, and is withdrawn

§45 reported that large modules split into an expensive class at ~100-440 bytes of RSS per
StableHLO character and a cheap one at ~5-46, and filed that as a correction to §31.16's
~200 B/char. **That was wrong, and the error was mine.** RSS grows only when glibc must
map *new* memory: a compile that fits in arena an earlier compile freed costs nothing
measurable while producing an equally large executable. The "cheap" programs were simply
the ones compiled **later**.

Re-measured with `malloc_trim(0)` before each compile, in compile order:

| order | chars | B/char as §45 measured it | B/char with the arena trimmed |
|---:|---:|---:|---:|
| 16 | 1 498 177 | 317 | 310 |
| 21 | 177 529 | -- | 171 |
| 27 | 2 282 899 | 242 | 351 |
| 32 | 1 055 811 | **7.3** | **223** |
| 33 | 499 628 | **4.7** | **184** |

One band, **171-351 B/char**, no classes. §31.16's ~200 B/char stands and
`tried_and_rejected.md` is restored to it. The lesson is narrower than the mistake: an RSS
delta measures what the *allocator* did, not what the *program* costs, and the two agree
only when the arena is empty first.

## 51. Can the bloat be fixed? The mechanism, priced exactly (2026-09-06)

§50 established *what* is in a 2.28 M-character module (28 106 ops, 41.6 % of them shape
plumbing) and what it costs (~500 MB resident per loaded executable). The follow-up
question is whether any of that is removable. Four candidate levers, all measured.

### The broadcasts are not folded away -- XLA's optimiser makes the program bigger

Comparing the op histogram before and after XLA's own passes, on the same module:

| | before | after |
|---|---:|---:|
| total ops | 28 101 | **70 065** |
| `broadcast_in_dim` / `broadcast` | 7 843 | **9 051** |
| `constant` | 3 985 | 6 217 |
| `parameter` | -- | 11 351 |
| `fusion` | -- | 1 648 |
| shape plumbing | 11 681 (41.6 %) | 16 565 (23.6 %) |

Optimisation **expands** the program 2.5x: 1 648 fusion kernels, each printing its own
parameters and its own copies of the constants it captures. The broadcasts survive and
multiply. So they are not free -- something downstream really does emit code for them,
and removing them at the source would remove real work rather than work XLA was going to
delete anyway.

### XLA's optimisation level is not the lever -- measured, ~4 %

`st_regression` SAND, whole solve, three settings; the answer must not move and does not:

| `XLA_FLAGS` | programs | compile | wall | biggest program | HWM | `objf` |
|---|---:|---:|---:|---:|---:|---|
| *(none)* | 34 | 25.30 s | 36.07 s | 747.8 MB | 1497.1 MB | -16.5885766807 |
| `--xla_backend_optimization_level=0` | 34 | 24.11 s | 35.81 s | **719.4 MB** | 1478.5 MB | -16.5885766807 |
| `--xla_llvm_disable_expensive_passes=true` | 34 | 31.72 s | 45.91 s | 749.8 MB | 1504.3 MB | -16.5885766807 |

Turning XLA's optimiser off entirely saves **3.8 %** of peak. The memory is not in the
passes; it is in the number of instructions LLVM must generate code for.

### The serialised IR is 4.6x smaller than its text, and that buys nothing

The module's MLIR **bytecode** -- what `jax.export` stores -- is **493 840 bytes against
2 282 899 characters of text, 21.6 %**. (`jax.export` itself is unavailable in this env:
`serialize()` raises `ImportError: Please install 'flatbuffers'`, so the bytecode was
taken directly from `module.operation.write_bytecode`, which is the same payload.) But
**serialised size is not what costs memory**: an `Exported` still holds StableHLO, so
loading one compiles it again and arrives at the same ~500 MB. This matches §31.20's
finding that the persistent compilation cache removes *all* compile time and 1.2 % of
peak RSS. A more compact IR is a distribution and caching win, not a memory one.

### Vectorising repeated structure is the lever, and it is worth about everything

The controlled version of the port's shape: `N = 600` nodes each computing the same
ten-op scalar recipe, written first as 600 separate scalar computations -- exactly how
the port emits them -- and then as one computation over a length-600 array.

| | StableHLO | bytecode | compile RSS | compile |
|---|---:|---:|---:|---:|
| 600 scalar nodes | **809 768 chars** | 127 169 B | **186.2 MB** | 2.39 s |
| the same, vectorised | **1 215 chars** | 1 228 B | **2.5 MB** | 0.03 s |

**666x less IR, 74x less memory, 80x faster to compile, and `max |diff| = 0.000e+00`** --
the identical answer, bit for bit. So the mechanism behind §50's 41.6 % is real and its
removal is worth roughly everything.

**The honest caveat, and it is a large one.** Those 600 nodes are *structurally
identical*, and `vmap` requires that. The port's ~500 nodes are mostly **different
formulas**, and no amount of vectorisation merges two different equations. This
experiment prices the mechanism; it does **not** price the port. The gain is bounded by
how much of the graph is genuinely repeated structure -- arrays of impurities, coil turns,
radial-build elements, per-element loops -- and **that fraction has not been measured**.
Measuring it is the next step, and it is a graph question rather than a compiler one:
count the nodes whose `fn` is the same callable applied to different `VarPath`s.

Until that number exists, the defensible claim is narrow: **the 41.6 % shape plumbing is
real work, XLA does not remove it, and where structure genuinely repeats it can be removed
for free.** Not "the port's memory can be cut 74x".

## 52. Constraint 11 is a tautology on the stellarator build path (2026-09-06)

§48 left one thing unexplained: with `.physics.rmajor` promoted to a design variable,
`c11`'s row stayed inert at 1.6e-16, even though `rmajor` demonstrably reaches the
objective, `c2` and `c16`. The hypothesis was that `rbld` is itself computed *from*
`rmajor` with derivative 1, so `c11 = rbld - rmajor` cancels identically. **Confirmed,
and it is exact.**

```
d(.build.rbld)/d(.physics.rmajor)  = 1.0                (bit-exact)
d(c11)/d(.physics.rmajor)          = -7.34e-18          (noise)
rbld - rmajor                      = -3.55e-15          (zero)
```

**The algebra, and it is two adjacent statements in one function**
(`functional_process/models/stellarator/build.py:143` and `:156`, and verbatim the same
two in PROCESS's own `process/models/stellarator/build.py:48` and `:62`):

```python
dr_bore = rmajor - (dr_cs + dr_cs_tf_gap + dr_tf_inboard + dr_shld_vv_gap_inboard
                    + dr_vv_inboard + dr_shld_inboard + dr_blkt_inboard
                    + dr_fw_inboard + dr_fw_plasma_gap_inboard + rminor)
rbld    = (dr_bore + dr_cs + dr_cs_tf_gap + dr_tf_inboard + dr_shld_vv_gap_inboard
           + dr_vv_inboard + dr_shld_inboard + dr_blkt_inboard
           + dr_fw_inboard + dr_fw_plasma_gap_inboard + rminor)
```

The ten terms are the same ten, in the same order. `rbld = (rmajor - S) + S = rmajor`
for **any** values of any of them. `dr_bore` is solved for as whatever is left of `rmajor`
after allocating every other radial element, and `rbld` then sums those same elements back
on. Checked by reading both sources, not only by differentiating. PROCESS's own comment
on the line reads `#  Radial build to centre of plasma (should be equal to
data.physics.rmajor)` -- and on this path it is, identically, which is precisely why the
equation asserting it can never bind.

**The tokamak is different, and constraint 11 is real there.** `process/models/build.py:1863`
builds `rbld` *outward* from `r_sh_inboard_out` through the shield/blanket/first-wall stack;
`rmajor` does not appear in it, and `dr_bore` is an independent field with its own
iteration variable (**ID 29**, `iteration_variables.py`). Measured on `large_tokamak_nof`,
which already carries `icc = 11`, `ixc = 3` **and** `ixc = 29` -- i.e. correctly specified:

```
d(.build.rbld)/d(.physics.rmajor)  = 0.3333333333333333    (not 1)
c11 residual                        = -1.186512e-02          (genuinely nonzero)
d(c11)/d(.physics.rmajor)           = +8.481647e-02          (live)
```

**This changes the upstream recommendation, and enlarges it.** §48 concluded "drop
`icc = 11` from `helias_5b`, which forgot `ixc = 3`". That is too small and, in its
reasoning, wrong: adding `ixc = 3` cannot fix it, because **no** choice of design
variables can give `c11` a nonzero row on this build formula. The correct statement is
that **`icc = 11` is structurally meaningless on the stellarator build path and should not
be listed on stellarator configurations at all**, while it remains correct and necessary
on tokamak ones. `stellarator_helias.IN.DAT` is, by that reading, already right -- it omits
`icc = 11` and frees `rmajor` -- and `helias_5b.IN.DAT` is the outlier.

**Scope, stated honestly.** The identity was verified in the unconditional body of
`calculate_build` / `st_build`, which every stellarator run in this port's MDF and SAND
graphs goes through; it is not gated by `blktmodel` or `ipowerflow`, the two switches that
routine does branch on. Other stellarator-mode branches were not exhaustively checked.

**What this is worth beyond the port.** A constraint has been evaluated on every
stellarator run, contributing a row to every Jacobian, and it cannot fail. Nothing in
PROCESS could have reported that, because a satisfied constraint looks exactly like a
working one -- it took a solver that factorises the equality block on its own (scipy) to
turn it into a visible error. That is the rewrite's case in miniature: the structure was
always there to be read, and nothing read it.

## 53. Should SAND expose the root find? The knob exists, and using it is worse (2026-09-06)

> **Largely SUPERSEDED by §56, same day.** The headline is wrong: nesting fails because of
> a seeding defect in `run_sand_harness._seed`, not because of the formulation, and with
> that patched it converges in 75 iterations at machine-precision residuals. The candidate
> rule below is also dead in both directions. Kept in full because the corrections are
> only legible against it.

§49 ended on a design question rather than a bug: `wp_width_r_min` is a SAND-only exposure
of an inner root find that MDF solves internally, so should SAND expose it at all?

**The knob already exists and was built for this question.** `sand.sand_graph(graph,
skip=(), keep=())` -- `keep` leaves a named problem un-lifted, staying a declared problem
driven by its own algorithm and invoked as an inner `Drive` once per outer condition
evaluation (paired with `sand_schedule(nest=True)`). Its docstring asks, in as many words,
whether *"SAND's trouble on `stellarator_helias` [is] the conditioning of one lifted row,
or the lift itself"*. Nobody had answered it.

**The rule SAND implements today** is: lift **every** declared problem into the combined
`Optimise`, with no discrimination by problem type -- a `FixedPoint` becomes a residual
equality via `Residualise` (`g(u) - u`), an `ImplicitFunction` is folded in via `Combine`.
Two exceptions are *detected* rather than chosen: `degenerate_fixed_points` and
`array_valued_problems` are dropped automatically.

**Nesting it does not work, and fails harder than the disease.** Structurally the variant
built exactly as documented -- unknowns 14 -> 13, equalities 8 -> 7, `wp_width_r_min` on
its own `SeededNewtonDriver`. Numerically:

- with optimistix's defaults, the **first** outer Jacobian evaluation raised *"The maximum
  number of steps was reached in the nonlinear solver"* -- the outer SQP never ran;
- patched permissive (`max_steps=4096`, `throw=False`), the outer solve managed **one**
  iteration and stopped at `objf = 1.03e23`, `max|eq| = 4055`, `min ie = -47870`.

So it is not a slow crawl but a blow-up, against an intact SAND arm that at least stays in
a bounded, physically sane region for its whole 500-iteration cap. **The "answer-preserving"
expectation this experiment was set up to check is refuted**, and there is no converged
nested answer to compare against VMCON's `1.21848284`.

**Why, as the best-supported explanation and not a verified one.** Under MDF the *whole*
coupled system is converged jointly at every trial point, so the inner root find only ever
sees a self-consistent state. Under SAND-with-`keep`, the other five couplings remain free
SQP variables and can be arbitrarily inconsistent at an early trial point, so the nested
`Intersect` is handed inputs MDF would never present -- landing it in the flat region
`SeededNewtonDriver`'s own docstring records (residual exactly flat below `x ~ 0.1`) where
Newton cannot move. Confirming this would mean nesting all six, which is MDF. Not done.

**`wp_width_r_min` is structurally the odd one out, and this part is solid.** Of the six
coupling equalities SAND introduces here, it is the **only one that is not a residualised
`FixedPoint`**: `Intersect` is an `ImplicitFunction` (`stellarator/coils/coils.py:229`),
while `IonVolAvgTemperature` (`physics/plasma_profiles.py:164`) and `DeltaEtaStep`
(`power/thermal_cryo.py:314`) are `FixedPointFunction`s, and `f_ster_div_single` plus the
two rate-density self-loops are `FixedPointCut`/self-loop residuals. Two independent
properties already on record attach to it and to nothing else here:
`VmconDriver.condition_scale`'s *"the only one whose units are genuinely not its
unknown's"* (a `FixedPoint` residual `g(u) - u` is definitionally in `u`'s units; an
`ImplicitFunction`'s condition need not be), and the flat region above. And empirically,
**none of the other five appeared among the oscillating conditions at all**.

**Candidate rule**: *expose `FixedPoint` residuals -- same units as their unknown,
generically full-rank once degenerate and array-valued cases are screened -- and keep
genuine `RootFind`/`ImplicitFunction` problems driven internally.* Consistent with
everything measured, and **it is one `RootFind` against five `FixedPoint`s on one
configuration.**

**A wrinkle the experiment did not cover.** This graph declares **two**
`ImplicitFunction`s, not one: `^problem.stellarator.coils.intersect` and
`^problem.vacuum.duct_diameter_root_find` (`vacuum/vacuum.py:123`). Only the first shows
up among SAND's six exposed couplings, so either the vacuum one is already being dropped
by the degenerate/array-valued screens or it is excluded some other way. **Unchecked** --
and it matters, because a rule of the form "keep every `ImplicitFunction` internal" would
touch both, and one of them is apparently already internal for reasons nobody has stated.

**`low_aspect_ratio_DEMO` was not reached** -- the intended cross-check. Its graph is 247
nodes against the stellarator's 154 and it did not finish compiling inside the time box.
Worth flagging from the reference table alone, and only as a question: that arm's
pathology is the *opposite* shape, with **VMCON** slow (79 iterations against SLSQP's 17)
rather than SLSQP. Whether that is a different mechanism or the same one striking the
other solver first is unmeasured speculation.

## 54. The size chain, end to end: 244 nodes to 500 MB (2026-09-06)

> **The last third is CORRECTED by §57.** The size chain down to the instruction
> count stands; "11 MB of machine code, ~250 MB resident" does not -- the machine
> code is 5.8 MB and the retained figure is not a per-program property.
> The Jacobian multiplier is also corrected there (§59): measured 2.4x, not 5.6x.

§50 and §51 measured pieces; this is the whole chain on one configuration, because
"70 000 instructions for 244 nodes" is the right thing to be suspicious about and the
answer is that it is not 244 nodes' worth of anything.

`st_regression` SAND assembles **244 nodes** and compiles 34 programs. The four largest,
by StableHLO characters:

| chars | ops | `broadcast_in_dim` | ops / node |
|---:|---:|---:|---:|
| 2 282 899 | 28 101 | 7 843 | **115.2** |
| 1 498 177 | 17 988 | 2 124 | 73.7 |
| 1 055 811 | 12 425 | 1 778 | 50.9 |
| 499 628 | 5 060 | 186 | **20.7** |

**The model itself is ~21 ops per node.** That is the bottom row, and it is an entirely
ordinary number for a physics formula -- a handful of multiplies, an add, a divide. The
chain to 500 MB is four multiplications on top of it, none of them the model getting
bigger:

1. **244 nodes x ~21 ops = ~5 000 ops** -- the value program. Reasonable.
2. **x ~5.6 -> 28 101 ops** -- the largest program carries *derivatives*. `jacfwd` is
   `vmap(jvp)`, so the program holds the primal plus its tangent, and the broadcast count
   goes 186 -> 7 843 along with it. This is the step people skip: the big program is not
   the model, it is the model **and its Jacobian**.
3. **x 2.5 -> 70 065 HLO instructions** -- XLA's own fusion pass (§51). It merges
   element-wise ops into 1 648 fusion kernels so intermediates stay in registers instead
   of round-tripping through memory; the printed instruction count *rises* because each
   fusion is emitted as its own sub-computation with its own parameter list (11 351
   `parameter` instructions) and its own copies of the constants it captures (3 985 ->
   6 217).
4. **-> 11 MB of machine code, ~250 MB resident.**

### Where the memory actually sits: retained against transient

Loading the same 11.0 MB serialized executable three times in one process, trimming the
allocator arena after each:

| | during load | **retained after trim** | time |
|---|---:|---:|---:|
| compile from StableHLO | 762.3 MB | -- | 14.14 s |
| load #1 | 476.9 MB | **178.1 MB** | 2.18 s |
| load #2 | 538.9 MB | **247.4 MB** | 1.99 s |
| load #3 | 557.4 MB | **248.6 MB** | 1.93 s |

So the ~500 MB splits roughly in half. About **250 MB is genuinely retained per loaded
executable** -- that is the in-memory program, ~20x its serialized size, and it is what
`clear_caches` + `malloc_trim` reclaims when the executable is dropped. The other ~250-300
MB is LLVM's transient JIT workspace, released immediately.

**Can it be turned off? No, not by any switch.** §51 measured
`--xla_backend_optimization_level=0` at **3.8 %** and
`--xla_llvm_disable_expensive_passes=true` at worse-than-nothing. A more compact IR does
not help either: MLIR bytecode is 21.6 % of the text, but an `Exported` still holds
StableHLO and compiles again on load. The only lever is step 1 and step 2 -- fewer ops in,
which is §51's vectorisation question, and it remains bounded by how much of the graph is
genuinely repeated structure. **Unmeasured, and still the number that decides this.**

## 55. jax-native optimisation is blocked by one `while_loop`, not by the optimisers (2026-09-06)

Both drivers reach a host NumPy library through `jax.pure_callback`, which makes a
converged solve opaque: `jax.grad` through one raises *"Pure callbacks do not support
JVP"* (`_sqp_callback`'s docstring). With the problem better conditioned than it was --
the Ward kink smoothed, `icc = 11` gone, all four `helias_5b` cells converging -- the
question was whether a jax-native optimiser can now do the job.

**It can, and the blocker turned out to be somewhere else entirely.**

- `optimistix 0.1.0` is installed (`Newton`, `BFGS`, `GaussNewton`, `LevenbergMarquardt`,
  `NonlinearCG`, `FixedPointIteration`); `optax` and `jaxopt` are **not**.
- `cottax`'s own `OptimiseDriver` refuses this problem by construction: its `check()`
  rejects any `Optimise` whose conditions carry an `Equality`/`Inequality` role, and
  `helias_5b` MDF has two of each.
- `optimistix.BFGS` on a quadratic-penalty objective **fails**, and the reason is the
  finding: `BFGS.step` takes its gradient by `jax.linear_transpose` -- reverse mode -- and
  this graph's `solve_duct_geometry` (`models/vacuum/vacuum.py`) is a `lax.while_loop`
  with dynamic bounds, which has a JVP and **no transpose rule**:
  `ValueError: Reverse-mode differentiation does not work for lax.while_loop`.
  `optimistix 0.1.0` exposes no `autodiff_mode` switch on its gradient-based solvers, so
  they all hit the same wall.
- Forward mode works. A hand-rolled Adam loop over a `jacfwd` gradient of the penalty
  objective, entirely in `jax.numpy` with no callback anywhere, converges `helias_5b` MDF
  to **`objf = 0.764211181`** against VMCON's `0.764215516` (4.3e-6, ~6 s.f.), in ~40-150
  ms warm. The whole 150-step loop wrapped in `lax.scan` **is `jax.jit`-able end to end**,
  and `jax.jacfwd` **through the entire solve succeeds** (31 s to compile
  forward-over-forward). `jax.grad` through it fails, same transpose gap.
- `stellarator_helias` MDF, the harder one: 500 Adam steps, 542 ms warm,
  `objf = 1.219117069` against `1.21848284` (~3 s.f.), largest equality residual 1.7e-3
  and one inequality still violated by 3.1e-4 -- a fixed-`rho` penalty leaves
  infeasibility, as it should.

**This promotes an item that has been sitting at the bottom of `next_steps.md`.**
`vacuum.py`'s `solve_duct_geometry` was filed as "the sole remaining reverse-mode AD
blocker" -- a curiosity about one node. It is not: it is **the reason no off-the-shelf
jax-native optimiser can be used at all**, because every gradient-based `optimistix`
solver is reverse-mode. Giving that loop a fixed or `scan`-shaped iteration count would
unlock the entire library with no custom optimiser code, and the `vmap`-over-64-candidates
conversion already sized in `next_steps.md` is exactly that shape. The other route --
building a real constrained forward-mode method (augmented Lagrangian, or KKT
least-squares through `GaussNewton`/`LevenbergMarquardt` on forward-mode Jacobians) --
is independent and would also work.

**Not a driver yet, and the numbers above are a scoping experiment**: fixed-penalty Adam
is not competitive with VMCON's 4 iterations and is not meant to be. What it establishes
is that the *formulation* is expressible in JAX and the barrier is one `while_loop`.

## 56. The exposure rule is dead, and §53's mechanism was wrong (2026-09-06)

§53 proposed a rule -- *expose `FixedPoint` residuals, keep genuine `RootFind`s internal*
-- on one sample against five, and explained the nesting blow-up by inconsistent free
couplings. Both are now refuted, and the second refutation found a real defect.

### The rule fails in both directions

The SAND coupling table across all five SAND-bearing configurations (the two
evaluation-mode files have no SAND arm):

| configuration | exposes an `ImplicitFunction`? | SAND SLSQP | SAND VMCON |
|---|---|---|---|
| `stellarator_helias` | **yes** -- `Intersect`/`wp_width_r_min` | **cap(500)** | 24 |
| `helias_5b` | **yes** -- the same `Intersect` | **converged, 8** | 7 |
| `large_tokamak_nof` | no (5 `FixedPoint`s) | converged, 13 | 10 |
| `low_aspect_ratio_DEMO` | no (5 `FixedPoint`s) | converged, 17 | **79** |
| `st_regression` | no (4 `FixedPoint`s) | converged, 10 | 10 |

- **Not necessary**: `low_aspect_ratio_DEMO` exposes none and is badly troubled anyway
  (79 VMCON iterations against its own MDF's 11).
- **Not sufficient**: `helias_5b` exposes the *same* `Intersect` as `stellarator_helias`,
  over 9 unknowns, and converges in **8**.

**That second row only became readable today.** Before `icc = 11` was removed (§52),
`helias_5b`'s SLSQP arms died at iteration 2 on the tautological constraint, which made it
look like a confirming case for the rule when it was really failing for an unrelated
reason. Removing the inert constraint **unconfounded the experiment**, and the case it
then provided killed the rule. Worth noting as a method point: the rule would have
survived on stale numbers.

### `DuctDiameterRootFind` is not evidence -- it is not in any graph

§53 flagged a second declared `ImplicitFunction` (`vacuum/vacuum.py:123`) that SAND does
not expose, and asked why. The answer is that it **never appears in any configuration's
`driven.declared`, or anywhere in any of these graphs**. Its own docstring says so: it is
registered in `total_process.py` but *"not wired to any other node ... every one of these
six `VarPath`s is minted and unique to this class, so it sits as its own disconnected
island"*, driving deferred. It is not screened out by `degenerate_fixed_points` or
`array_valued_problems` -- those only examine `FixedPoint`s. So it is **a separate,
pre-existing gap in the port** (a registered node that nothing reaches), unrelated to SAND
or to any switch, and it provides no evidence either way. `^problem.stellarator.coils.intersect`
is the only `ImplicitFunction` any configuration's graph actually contains.

### `low_aspect_ratio_DEMO` is a different mechanism

Its 79-iteration VMCON trajectory: `dx` runs 1.65, 0.078, 0.098, 0.012, 0.064 early and
then shrinks essentially monotonically to 1e-4-5e-4 over the last twenty; `max|eq|` and
`min ie` reach ~0 by **iteration 2** and stay there; the objective wobbles in the fifth
decimal for the whole tail. **Creeping, not oscillating** -- a slow, roughly linear-rate
asymptotic approach with no ramp-and-reset cycles and no conflicting pair. Structurally
unlike `stellarator_helias`, and consistent with the table above: SAND has at least one
second failure mode that has nothing to do with exposing a root find. Uncharacterised.

### §53's mechanism is refuted, and the real cause is a seeding defect

The ladder settles it: nesting 1, 2 or 4 of the five declared problems gives
**byte-identical** blow-ups at `objf = 1.03194311e+23`. Adding or removing other nested
couplings changes nothing, so "the other free couplings hand it an inconsistent state" is
wrong. (Nesting all five is structurally refused by `Combine` -- *"fewer than two problems,
so nothing to combine"* -- a real limit of `sand_graph` today, not a result.)

A direct `jax.debug.print` at the failure shows the nested Newton's input is **exactly
`y0 = [0.]`**, the documented flat plateau, on every rung. And the reason is one line in
`run_sand_harness._seed`:

```python
coupling = cut or (source in drive.unknowns and source not in design)
```

`_seed` decides "is this a coupling unknown, and therefore seeded from the converged MDA
fallback?" by asking whether it is in **`drive.unknowns`** -- the *outer* block's unknown
set. But `sand_graph(keep=...)` is defined to *remove* a kept problem's unknowns from
exactly that set and give them to an inner `Drive`. So the moment a coupling is nested it
stops being recognised as coupling, falls through to `ground_truth(base, source)`, and
gets the cold `DataStructure` default -- `0.0` -- while the correct value sat unused in
`fallback` all along (`fallback['.stellarator.wp_width_r_min'] = 0.6286`). An isolated 1-D
Newton started at a point of zero derivative cannot move.

**This is a latent defect in the port, not a property of the problem**: `_seed` conflates
*"is a coupling quantity"* with *"is exposed to the outer block"*, and those coincide only
while nothing is kept. It has never bitten production because `keep=` is a research knob.
The fix shape is to ask the question about the quantity rather than about which block
currently owns it. `mda.ROOT_FIND_SEEDS` also has no fallback entry for this path -- it
moved to `SUPPLIED_STARTS` and nothing replaced it -- so the later rescue path cannot save
it either.

**With both seeding gaps patched at runtime, nesting works.** Nested SLSQP converges in
**75 iterations** at `max|eq| = 2.49e-14`, `min ie = -2.23e-13` -- residuals ten orders
better than the exposed arm ever reached in its 500-iteration cap (3.95e-4). So §53's
headline, *"the knob exists and using it is worse"*, was **an artefact of the defect
above**, not a finding about SAND.

**It is still not answer-preserving, and the gap is small but real.** `objf = 1.21833891`
against VMCON's `1.21848284` -- 1.44e-4 absolute, **0.012 % relative**. That is ~25x
closer than the drop-`c62` experiment's 0.31 %, and both formulations are at machine
precision on their residuals, so this reads as convergence to a genuinely different nearby
point in a non-convex problem -- plausibly a different active-set path -- rather than an
unconverged one. **Not verified**, and it is the question a real fix would have to answer.

**Negative results, recorded so nobody repeats them**: rescaling `c62` or `wp_width_r_min`
(§49) does not work; nesting without the seed fix fails identically at every rung; nesting
every problem is not buildable through `sand_graph` today.

## 57. Correction: `serialize()` is HLO, the machine code is 6 MB, and "the memory is the executable" is wrong (2026-09-06)

§50 reported *"the executable serialises to 11.0 MB"* and §54 concluded *"the resident cost
is the executable, not the compiler's workspace"*. Both are wrong, and the question that
exposed them was the obvious one: **why would an 11 MB program need 250 MB of RAM?** It
does not. The two numbers were never measuring the same object.

**`LoadedExecutable.serialize()` returns the optimised HLO, not code.** The blob has no
object-file magic -- it opens `\x8fA\x08\x03\x12\tpjrt_ifrt`, protobuf wire format -- and
it contains, as plain text, **12 214** occurrences of `fusion`, **15 063** of `broadcast`,
**11 364** of `parameter` and **12 434** of `constant`: HLO instruction opcodes, in counts
matching the optimised module. Its size is **1.20x the optimised HLO text**, and that
ratio holds across every program measured (1.14-1.71). So the 11 MB is XLA's *input to
code generation*, serialised. This also explains the 1.9 s deserialise in §54: loading one
re-runs LLVM codegen, which is why it costs hundreds of megabytes rather than mapping a
finished binary.

**The actual machine code is 5.8 MB.** Measured by counting anonymous **executable**
(`r-x`) mappings in `/proc/self/maps` across the compile, which is where a JIT puts code:

| StableHLO chars | post-opt instrs | machine code | bytes / instruction |
|---:|---:|---:|---:|
| 2 282 899 | 70 065 | **5.8 MB** | 87 |
| 1 498 177 | 28 003 | 4.1 MB | 154 |
| 1 055 811 | 20 169 | 3.2 MB | 164 |
| 499 628 | 8 575 | 0.9 MB | 106 |
| 177 529 | 1 190 | 0.2 MB | 145 |

**87-165 bytes of machine code per post-optimisation HLO instruction** -- stable, and
entirely unremarkable for compiled scalar arithmetic. There is no 20x expansion anywhere;
that number came from dividing a serialised-HLO size by a figure that was mostly not code.

**And "retained per program" does not survive scrutiny either.** The same run's
retained-RSS column, after `malloc_trim(0)` on both sides of each compile:

| post-opt instrs | kept | kept / instruction |
|---:|---:|---:|
| 70 065 | 309.7 MB | 4.53 KB |
| 28 003 | 203.4 MB | 7.44 KB |
| 20 169 | 49.3 MB | 2.51 KB |
| 8 575 | 10.3 MB | 1.23 KB |
| 1 190 | **-11.9 MB** | -- |

A **negative** row, and a 6x spread. So retained RSS is *still* order-dependent even with
the arena trimmed on both sides -- the same class of artefact §50's withdrawn "two
classes" finding was, one level subtler: trimming before and after a compile does not stop
a later compile reusing what an earlier one left. **Per-program retained cost is not a
well-defined quantity here**, and §54's table of it should be read as what one particular
ordering did, not as a property.

### What actually stands, after three corrections to the same measurement

1. **Peak during compile is real and large**: ~750 MB for the largest program, ~15 s. This
   has been stable across every method used and is what the OOM was about.
2. **The pass-level floor plateaus**: §45's whole-configuration numbers (~850 MB, peak
   2.2 GB over seven configurations) were measured end-to-end rather than per program, and
   are unaffected by any of this.
3. **Machine code is small** -- 5.8 MB for the biggest program, ~87-165 B per instruction.
4. **What the other ~300 MB is, is not established.** It is not code, and it is not
   cleanly attributable per program. A plausible reading -- 70 065 instructions at ~4 KB
   apiece would be the in-memory `HloModule` XLA keeps alive to serve `hlo_modules()` --
   fits the largest row and **does not fit the others**, so it is a hypothesis with one
   supporting point and three against. **Not established, and stated here so nobody
   quotes it.**

**The lever is unchanged and the reason for it is now better founded**: instruction count
drives both the compile peak and the code size, XLA's fusion pass *raises* the instruction
count 2.5x, and the only way to emit fewer is §51's vectorisation question -- still bounded
by an unmeasured repeated-structure fraction.

**Method note, since this is the third correction to one measurement.** Every wrong answer
here came from the same mistake in a different costume: attributing a *process-level* RSS
delta to a *single program*. glibc's allocator does not work that way, and neither trimming
the arena first (§50) nor trimming it on both sides (here) makes it work that way. The
measurements that held up all avoided the attribution entirely -- end-to-end pass peaks
(§45), `/proc/self/maps` executable pages, and instruction counts.

## 58. A genuinely constrained forward-mode method works -- on the easy problem (2026-09-06)

§55's penalty-Adam result was not a constrained solve: a quadratic penalty is an
unconstrained approximation, which is why it left `max|eq| = 1.7e-3` and a violated
inequality on `stellarator_helias`. The proper test is an augmented Lagrangian with real
multiplier updates (`lambda += rho*c_eq`, `mu = max(0, mu + rho*c_ineq)`), inner
minimisation by a damped Newton step whose gradient **and Hessian** both come from
`jacfwd(jacfwd(...))` -- forward-over-forward, no transpose anywhere.

**`helias_5b` MDF: it works, and it is a real constrained solve.**

| | AL + forward-mode Newton | VMCON |
|---|---|---|
| `objf` | **0.7642155163** | 0.764215516 |
| difference | **2.97e-10** | -- |
| `max\|eq\|` | **2.4e-14 to 2.3e-13** | -- |
| `min ie` | -3.650 (feasible with margin) | -- |
| cost | 120 Newton steps, 98.0 s (untuned) | 4 iterations |

Converged to 9 s.f. by outer iteration 3-4. The equality residual is driven to machine
epsilon rather than traded against the objective -- which is exactly what separates this
from §55's penalty run, and it is the number to check first on any such claim.

A scaled variant (`1/x_start`, VMCON's own conditioning, plus a trust-region cap) reaches
the same precision in 60 steps / 83.3 s. The scaling is **not optional**: `flat_start`
spans ~7 to ~2.1e20 (density in m^-3), and on `stellarator_helias` an unscaled Newton step
drove the model's *own* inner MDA solve past `max_steps` -- a hard
`EquinoxRuntimeError` inside the model before any line search could reject the point.

**`stellarator_helias` MDF: it does not converge, and it stalls rather than crawls.**
`objf` improves once (0.732 -> 0.875 at outer 0->1) and is then **bit-identical for
thirteen further outer iterations** while `rho` grows 1 -> 1e8. Final `objf = 0.875` against
VMCON's `1.21848284` -- **0.343 away** -- with `max|eq| = 0.099` and `c24` violated by
`+0.0208`. A stuck outer loop: the backtracking line search rejects every trial step once
the AL landscape steepens at `trust = 0.3`, `damping = 1e-6`, 6 inner steps. Untried:
smaller trust region, more inner steps, better-conditioned damping, slacks or an active set
for the twelve inequalities.

**And the crude method beats the good one there.** Penalty-Adam reached `objf` within
6.3e-4 on this problem; AL+Newton is 0.343 away. That is a statement about tuning, not
about the methods -- but it is worth recording that the *genuinely constrained* method is
currently worse on the 8-variable/14-condition case while being nine orders better on the
3-variable/4-condition one.

**Not re-checked**: whether the AL version is still `jit`/`scan`-able and
`jacfwd`-differentiable end to end. Only the inner Newton solve is jitted; the outer loop
is a Python `for`. The Hessian adds a second `jacfwd` layer and differentiating the solve
would add a third, so the expectation is "yes but expensive" -- **inference, not
measurement**.

### The fact that decides the vacuum question

`optimistix`'s least-squares route was not attempted, but the question behind it has a
one-line answer, read directly from the installed source:
`optimistix/_solver/gauss_newton.py:176` computes its Jacobian with **`jax.jacrev`**, and
there is no user-facing `jac` argument on `GaussNewton`/`LevenbergMarquardt` to override
it. So the least-squares solvers are reverse-mode like the rest, and fixing the
`vacuum.py` `while_loop` is **necessary** for any off-the-shelf `optimistix` route, not
merely convenient.

## 59. The vacuum node: discrete, statically bounded, and cheaper to fix than to keep (2026-09-06)

`solve_duct_geometry` (`models/vacuum/vacuum.py:316-399`) is the port's last reverse-mode
blocker and, per §55, the reason no off-the-shelf jax-native optimiser can run on this
graph.

**What it does.** For one gas species it repeatedly re-solves `solve_duct_diameter` -- a
genuine Newton root find -- at a shrinking target conductance `ceff_i_init * 0.9**k`,
checks whether the resulting duct fits between adjacent TF coils (`a1_new < a1max`), and
**stops at the first `k` that fits**; it gives up with `nflag = 1` if shrinking would drop
`ceff` below `1.1 * s_i`. A faithful port of PROCESS's own
`Vacuum._newton_method_duct_diameter` outer `while True` (`process/models/vacuum.py:460`),
unbounded there and capped at 64 here.

**It is genuinely discrete, and that settles the goal.** The output is a *selection* among
candidates, so crossing a threshold changes which candidate's answer is returned -- a jump
in **value**, not merely in gradient. Its sibling `solve_duct_diameter` is the opposite: a
true root of a smooth equation, hence smooth by the implicit function theorem, which is why
the `stop_gradient`-plus-one-live-Newton-step treatment applies there and cannot transfer
here. So *"make it differentiable"* is the wrong goal; *"make it transposable, with the
honest a.e. derivative of a piecewise-smooth selection"* is right -- the same thing one
gets differentiating through any `argmax`.

**`max_outer = 64` is a plain Python default**, i.e. a compile-time constant, so the
`vmap`-all-candidates-then-`argmax` conversion is available.

**The prototype, measured on real inputs.** Six real per-species calls captured from
`helias_5b` and `stellarator_helias` at their converged points (**every one hits `k = 0`**
-- the first candidate fits), plus five synthetic edge cases (`k = 5`, `k = 63`,
never-fits, floor-triggered `nflag = 1`):

| | result |
|---|---|
| agreement with the `while_loop` | **max \|diff\| = 5.55e-17** -- float64 round-off |
| `jax.grad` downstream, current code | raises `Reverse-mode differentiation does not work for lax.while_loop` |
| `jax.grad` downstream, prototype | **succeeds**, matching `jacfwd` to the same precision |
| HLO text | 85 846 chars (loop) -> **81 201** (vmap) -- *smaller* |
| runtime at `k = 0` (the real case) | 45.1 us -> 55.0 us (**1.22x slower**) |
| runtime at `k = 63` (worst case) | 133-138 us -> **~52 us (2.6x faster)** |

**The feared ~20x arithmetic cost does not appear.** `vmap` batches the 64 candidate Newton
solves rather than duplicating them, and beats sequential `while_loop` dispatch outright
once the loop runs more than a few trips. The HLO gets *smaller*.

**And it unlocks `optimistix`, demonstrated end to end rather than argued.** On the real
assembled `helias_5b` MDF graph (159 nodes, 3 design variables, 5 conditions) with a
penalty objective built from its actual `condition_map`, `optimistix.minimise(BFGS(...))`:

- **current code**: raises the filed `ValueError` after 3.45 s of tracing;
- **prototype monkeypatched in, nothing else changed**: runs, and in 60 steps converges to
  `objf = 0.7642142560891302` against VMCON's `0.764215516` (1.6e-6), equalities ~1e-11,
  both inequalities feasible. **Off-the-shelf gradient-based solver, zero custom optimiser
  code.**

**Recommendation: make the change.** It reproduces PROCESS's answers to round-off on every
case tested, is transposable, costs the same or less at every trip count measured, and is
confirmed to unlock the library on a real graph rather than a toy. One caveat belongs in
the docstring if it lands, parallel to `solve_duct_diameter`'s own: the recovered gradient
is the honest a.e. derivative of a piecewise-smooth selection, **not** a smoothing of a
genuinely discontinuous function.

**Correction to §54's arithmetic.** That section put the Jacobian program at "~5.6x" the
value program, which was inferred from the *ordering* of `st_regression` SAND's compiled
programs, not measured. Measured directly through `host_cache.bind`'s three programs on
`stellarator_helias` MDF (8 unknowns, 15 conditions): **values 514 064 chars, jacobian
1 247 635, fused 1 252 549** -- so the Jacobian program is **2.4x** the value program, and
the fused pair costs **0.4 %** more than the Jacobian alone, which is why
`VmconDriver.fused` defaults to `True`. The direction §54 claimed is confirmed; the
multiplier was a guess and is now a measurement on a named configuration.

## 60. The missing memory, measured in isolation at last (2026-09-06)

Three attempts at per-program memory produced two retractions (§50, §57), all from the
same mistake: attributing a **process-level** RSS delta to a **single program** while
glibc's arena reuse makes that meaningless. §57 concluded that per-program retained cost
"is not a well-defined quantity here". It is -- but only if the process contains exactly
one program.

**Method, and it is the whole result.** Harvest each program's StableHLO **bytecode**
straight from the `backend_compile_and_load` hook
(`module.operation.write_bytecode`), save it, then in a **fresh interpreter per program**
-- nothing compiled before it, ever -- parse it back and call
`backend_compile_and_load` directly, with no cottax or session machinery involved.
Cross-checked against §57's independent `/proc/self/maps` route: 5 952 KB of executable
pages for the largest program against §57's 5.8 MB. Same number, different method.

### The honest per-program peak

| StableHLO chars | post-opt instrs | peak RSS | peak / instruction |
|---:|---:|---:|---:|
| 2 282 899 | 70 065 | **866.9 MB** (857-877 over 3 runs) | 12.7 KB |
| 1 514 799 | 28 003 | 531.6 MB | 19.4 KB |
| 1 055 811 | 20 169 | 393.7 MB | 20.0 KB |
| 516 021 | 8 575 | 245.2 MB | 29.3 KB |
| 177 529 | 1 190 | 110.7 MB | 95.2 KB |

`peak_MB ~ 157 + 0.0106 x instructions` -- **a ~157 MB fixed cost per compile plus ~11 KB
per post-optimisation HLO instruction**. The fit is loose (residuals to ~75 MB), so it is
scaling, not a model; a clean bytes-per-instruction constant does not exist because there
is a real fixed component. **These supersede every per-program figure in §45, §50 and
§54.** §45's *whole-configuration* numbers are a different measurement, end to end, and
stand untouched.

### Where the bytes are, split three ways

Largest program, mean of three runs:

| stage | RSS above baseline | share |
|---|---:|---:|
| peak, immediately after compile | 866.9 MB | 100 % |
| after `gc` + `malloc_trim`, **executable still alive** | 397.9 MB | 45.9 % |
| after dropping every reference + `clear_caches` + gc + trim | **142.5 MB** | 16.4 % |

- **~469 MB (54 %) is transient LLVM codegen workspace** -- reclaimed by `malloc_trim`
  while the executable is still live. §57's guess, now measured in isolation.
- **~255 MB (30 %) is genuinely attached to the live executable** -- it returns when the
  last Python reference is dropped. So something *is* retained per loaded executable,
  which partially confirms the hypothesis §57 filed. **Not** the "`HloModule` at 4 KB per
  instruction" story, though: that number still fails on the smaller programs.
- **~142 MB survives everything**, and is roughly **size-independent** (63-150 MB across a
  60x range of instruction counts) -- a fixed per-compile-event residue.

### The crux: that residue is almost all allocator free-list, not live data

For the largest program, after the full drop-and-clean, glibc's own `mallinfo2.uordblks`
-- bytes it considers **genuinely in use** -- is **~14 MB above baseline**, while process
RSS is **142.5 MB** above it. The ~128 MB gap is memory glibc has already marked free and
has not returned to the kernel, and an explicit `malloc_trim(0)` immediately before the
reading could not reclaim it -- almost certainly fragmentation holes below the arena's
high-water mark rather than top-of-heap space `brk` can shrink.

**So the answer to "why does compiling a few graphs need gigabytes" is: mostly it does
not.** Of a 867 MB peak, ~14 MB is live afterwards, ~255 MB is held by the live
executable and returns when it is dropped, and the rest is allocator behaviour. That makes
it a **tuning problem rather than a hard cost**, which is the best available outcome.

### Levers, measured

| lever | effect on peak | cost |
|---|---|---|
| **`MALLOC_ARENA_MAX=1`** | 866.9 -> **586.1 MB (-32 %)**, reproducible (586.3 / 586.0 / 580.8) | compile **14.7 s -> 23.0 s (+57 %)**; a single arena serialises malloc across XLA's compile threads |
| `MALLOC_TRIM_THRESHOLD_` / `MALLOC_MMAP_THRESHOLD_=65536` | ~7-13 % alone | adds nothing on top of `ARENA_MAX=1` |
| `--xla_cpu_parallel_codegen_split_count=1` | **none** (876.8 MB; 591.5 with ARENA_MAX) | -- |
| *(from §51)* `--xla_backend_optimization_level=0` | 3.8 % | -- |
| *(from §31.20)* persistent compile cache | 1.2 % | removes all compile *time* |

`MALLOC_ARENA_MAX=1` is a real 32 % lever on the peak -- the only one found -- and it
costs 57 % more compile time. It barely moves the retained residue. Whether that trade is
worth making depends on whether a pass is memory-bound or time-bound, and the matrix
runners are currently neither since `malloc_trim` between rows already bounds them (§45).

**Still not established**: which arena and which allocation class the ~128 MB of
unreturned-but-free memory belongs to. `mallinfo2` gives aggregates only; answering it
needs `malloc_info` XML or heap walking.

## 61. Reverse mode: what it costs, and why checkpointing is the wrong tool here (2026-09-06)

§59 made reverse-mode AD possible on the port's graph for the first time. What it costs
was unmeasured, and the natural worry -- reverse mode stores the forward pass, so does
this need `jax.checkpoint`? -- deserved a number rather than a guess.

Measured **one mode per process** (§60's discipline), on two real MDF blocks. Reverse and
forward agree: the Jacobian checksums match to 1 ulp
(`80.58773110605564` against `...571`).

### Compile cost -- this is where reverse mode is expensive

| config | mode | compile | peak RSS | HLO chars |
|---|---|---:|---:|---:|
| `helias_5b` (3 unknowns, 5 conditions) | values | 2.27 s | 177 MB | 632 782 |
| | `jacfwd` | 6.52 s | 179 MB | 1 063 494 |
| | `jacrev` | 7.59 s | **259 MB** | 1 172 115 |
| | `grad` (scalar) | 5.82 s | 203 MB | 1 043 833 |
| `stellarator_helias` (8 unknowns, 15 conditions) | values | 2.12 s | 162 MB | 678 267 |
| | `jacfwd` | 7.63 s | 324 MB | 1 417 483 |
| | `jacrev` | **21.95 s** | **1081 MB** | 1 621 536 |
| | `grad` (scalar) | 15.84 s | 694 MB | 1 388 303 |

On the larger block `jacrev` costs **2.9x the compile time and 3.3x the peak RSS** of
`jacfwd`, for only 14 % more HLO. So the expense is not the emitted program -- it is what
XLA does with it.

**And forward mode is the right choice for these Jacobians anyway**: a Jacobian of `n`
inputs to `m` outputs costs `n` forward passes or `m` reverse ones, and both blocks have
`m > n` (5 > 3, 15 > 8). Reverse mode's case is the **scalar objective**, `grad`, which is
one pass against `n` -- and that is what `optimistix.BFGS` uses, which is why §59's
unblocking matters even though `jacfwd` wins on the full Jacobian.

### Runtime cost -- reverse mode does store the forward pass, and it does not matter

`Compiled.memory_analysis()` reports what the **program** needs when it runs, as opposed
to what compiling it cost:

| config | mode | temp | peak |
|---|---|---:|---:|
| `stellarator_helias` | values | 105.3 KB | 111.3 KB |
| | `jacfwd` | 339.4 KB | 112.2 KB |
| | `jacrev` | **1758.7 KB** | 112.4 KB |
| | `grad` | 385.9 KB | 111.3 KB |
| | `grad` + `checkpoint` | 458.4 KB | **67.1 KB** |

Reverse mode's temp buffer is **5.2x forward's** -- exactly the "stores the forward pass"
cost, now measured rather than assumed. **And the absolute number is 1.7 MB.** Runtime
memory is four orders of magnitude below compile memory here, so it is not a constraint on
anything.

### `jax.checkpoint` works exactly as designed, and is the wrong tool

| | runtime peak | compile time | compile peak RSS |
|---|---|---|---|
| `grad` | 111.3 KB | 15.84 s | 694 MB |
| `grad` + `checkpoint` | **67.1 KB (-40 %)** | 17.78 s (+12 %) | **780 MB (+12 %)** |

It does what it promises: a **40 % cut in the program's runtime peak**. But it buys 44 KB
and costs 86 MB of compile memory and 2 s of compile time -- and on `helias_5b`
`jacrev_ckpt` is worse still, 259 -> 361 MB peak (+39 %) and 7.59 -> 9.61 s (+27 %).

**The reason is structural, not a tuning failure.** `checkpoint` trades memory for
recomputation *at runtime*, and expresses that trade by emitting the recomputation as
extra graph -- which is more for XLA to compile. This port's bottleneck is compilation of
a graph of ~28k scalar operations (§50, §54); its runtime buffers are ~100 KB. Optimising
the small side by enlarging the large one is a straight loss.

**So: no checkpointing.** The hypothesis was reasonable and the measurement says no. If a
future block ever has genuinely large intermediates -- an array-valued model rather than
this scalar one -- the calculus flips, and the numbers above are the baseline to
re-measure against.

## 62. `slsqp_jax` retried: one config fixed, one still diverging for a different reason (2026-09-06)

`slsqp_jax` (installed) was tried around 2026-09-02 as an in-graph SQP and abandoned: its
apparent 136 ms/iteration turned out to be a QP budget artefact (5 000 projected-CG
iterations per step, §31.11), and it reported `Nonlinear solve diverged` at **step 18** on
`stellarator_helias`. The hypothesis worth testing was that the divergence was **the
problem being chaotic**, not the package -- since then the Ward square-root kink was
smoothed and `icc = 11` was removed (§52).

Retried with `qp_max_iter = 20, qp_max_cg_iter = 20` (§31.11 found 20x20 bitwise-identical
to the 100x50 default at ~8x lower cost), VMCON's `1/x_start` design scaling, and
**`jax.jacfwd` only**. Worth noting: **no vacuum monkeypatch was needed** -- forward mode
was never blocked, which is consistent with §59's account that the `while_loop` blocked
only *reverse* mode.

| config | vars | steps | `objf` | vs VMCON | `max\|eq\|` | `min ie` |
|---|---:|---:|---|---:|---:|---:|
| `helias_5b` MDF | 3 | **2** | 0.764215525 | **9.3e-09** | 2.25e-07 | +0.070 feasible |
| `stellarator_helias` MDF | 8 | 19 | 1.222089621 | 3.6e-03 | 4.42e-03 | -1.2e-04 |

**`helias_5b` is a clean win** -- two steps, near-machine agreement, genuinely feasible.
Its baseline was re-derived rather than read from `reference_warm_matrix.txt`, whose row
predates the `icc = 11` removal.

**`stellarator_helias` still diverges, at essentially the same step (19 against 18), but
from a far better place.** `slsqp_jax` reports *"converged to a minimum-constraint-violation
infeasible stationary point"* with `last_step_size = 1.9e-6` -- stalled, not blown up.
`objf` is 3.6e-3 from the answer rather than 0.34, and the only violation is `c24` at
+1.2e-4, essentially on the boundary. **The Jacobian there is well conditioned**: stacking
the two equality rows with the three near-active inequalities (`c24`, `c83`, `c35`) gives
singular values `[5.59, 2.55, 0.72, 0.43, 0.035]`. **Not a rank collapse** -- so this is
not §46's failure mode recurring. It looks like the QP subproblem becoming locally
infeasible with five constraints simultaneously near-active in eight dimensions.

**Verdict: the hypothesis is half right.** The kink smoothing and the `icc = 11` removal
demonstrably helped -- the same run now stalls 3.6e-3 from the answer with a
well-conditioned Jacobian instead of landing on a trust-box corner 0.34 away. But it still
does not converge, at the same iteration count, and the remaining failure looks like this
QP handling near a corner of the feasible region rather than chaos in the model. Same
corner `pyvmcon`/OSQP has produced "non-convex KKT matrix" on elsewhere in this history.

**A genuine API trap, recorded because it produced a convincing fake failure.**
`slsqp_jax.compat.parse_constraints` infers a `NonlinearConstraint`'s component count from
the **shape of `lb`/`ub`**, not from `fun`'s output. Passing scalar `0.0, 0.0` for a
two-component equality silently truncates it to one row: `n_eq_constraints` came back `1`
where the block declares `2`. The first run looked exactly like a divergence -- hit a
bound, `max|eq| = 1.32` -- and was solving an under-constrained problem. Pass
`np.zeros(n_eq)` / `np.full(n_ineq, np.inf)` explicitly.

## 63. Codegen: what a plain C compiler does with the same arithmetic (2026-09-06)

The idea worth pricing: rather than lowering the graph to XLA -- which turns 28 101
StableHLO ops into 70 065 HLO instructions in 1 648 fusion kernels (§50) -- **emit the
block as one straight-line scalar function and hand it to an ordinary compiler.** The
graph is already an explicit dataflow IR, which is the thing you would need; that is the
rewrite's own thesis cashed out, since no such emission is possible from PROCESS's
imperative call order.

Before designing anything, the cheap bound: generate straight-line C with the measured op
mix (~50 % multiply/add, ~13 % divide, ~12 % transcendental, no shape plumbing) and time
`gcc`.

| ops | source | `-O0` | `-O1` | `-O2` | object (`-O2`) |
|---:|---:|---|---|---|---:|
| 7 000 | 235 KB | -- | -- | 3.0 s / 116 MB | 284 KB |
| 28 000 | 994 KB | **3.4 s / 242 MB** | 6.3 s / 260 MB | 10.0 s / 388 MB | 816 KB |
| 70 000 | 2.55 MB | -- | -- | 42.7 s / 1228 MB | 1.91 MB |

**Against XLA on the real block: 15 s and ~870 MB** (§54, §60).

Three things fall out, and the second is the one that stops this being a slam dunk.

1. **At `-O0`, 28 000 ops cost 3.4 s and 242 MB -- about 4.4x faster and 3.6x smaller than
   XLA.** And a scalar codegen would emit **fewer** than 28 000: §50 measured 41.6 % of the
   StableHLO as shape plumbing (`broadcast_in_dim` alone 27.9 %), which exists only because
   scalars are being wrapped into tensors. Emitting scalars directly deletes that category
   outright, leaving ~16 400 real arithmetic ops. Interpolating: **~2 s and ~150 MB.**
2. **This is not "compilers beat XLA".** At `-O2` and 70 000 ops, `gcc` costs **42.7 s and
   1.23 GB** -- considerably *worse* than XLA on the equivalent program. Straight-line code
   of this size is hard for any optimising compiler; several LLVM/GCC passes are
   superlinear in basic-block length. The win above comes from **emitting less** and
   **optimising less**, not from the compiler being better.
3. **`-O0` is defensible here in a way it usually is not.** The generated function is one
   flat basic block with no loops, so most of what `-O2` buys -- unrolling, licm,
   vectorisation -- has nothing to act on. What is lost is CSE and algebraic
   simplification, and both are *better done on the cottax graph itself*, where they are
   graph rewrites over named nodes rather than pattern matches over emitted text.

**What this does not price, and none of it is small**: derivatives (forward mode over ~8
design variables is 9x the arithmetic, inline and mechanical; reverse mode is a real
compiler feature), the non-traceable nodes (CoolProp), correctness plumbing against the
JAX path, and the GPU variant where 28k live values would spill registers badly. It also
says nothing about *runtime*, only compile.

**The honest summary**: the compile-side prize is real and roughly **an order of
magnitude**, it comes from the graph being explicit enough to emit narrowly, and the
first decisive experiment is small -- emit C for one block's **value** function, compile
`-O0`, and check bitwise agreement and timing against the JAX path. Everything after that
(derivatives, GPU, batching) is contingent on that number.

## 64. `slsqp_jax`'s divergence is the solver, not the problem (2026-09-06)

§62 read the `stellarator_helias` stall as "QP subproblem becoming locally infeasible with
five constraints simultaneously near-active in eight dimensions". **That reading is wrong
on both halves.** Three independent measurements:

1. **The linearised subproblem is feasible.** Built at the stall point directly from
   `jax.jacfwd` -- 2 equalities, 12 inequalities, box bounds, no trust region, no QP
   objective -- and solved with `scipy.optimize.linprog` (HiGHS). **A step exists with zero
   violation**: `A_eq d + c_eq = [0, -2.2e-16]`, all twelve inequalities satisfied, `d`
   inside its box.
2. **VMCON steps out of the identical point.** Seeded with `mdf.seed(..., design_values=stall_x)`,
   `VmconDriver` goes `objf` 1.2221 -> 1.2185 in three iterations with `max|eq|`
   4.4e-3 -> 1e-4 -> 8e-6, heading for its own 1.21848284. No local infeasibility.
3. **The failure is budget-invariant, and starts at step 1.** `slsqp_jax`'s verbose trace
   reports `QP ok: False` with `QPiter` pinned at the cap on **every step from the first**,
   plus `QPcyc: 2` (active-set cycling). Raising the budget **10x** (200x100 against 20x20)
   produces a **bitwise-identical trajectory** -- same `f`, same step sizes
   (9.537e-07, 4.883e-04, 1.221e-04, 3.052e-05). So it is not a budget shortfall; the
   active-set QP cycles regardless.

**And the arithmetic never supported §62's reading either.** At the stall: 2 equalities
plus inequalities within 1e-2 gives 3, and **0** within 1e-4 -- at most 5 against `n = 8`.
Not over-determined. The "10 constraints against 8 unknowns" in the verbose trace is
`slsqp_jax`'s *own* working-set bookkeeping during its failed cycling near step 1, not the
true active set.

So the residual failure is `slsqp_jax`'s projected-CG/active-set QP layer on an
8-variable, 14-condition problem -- **not** model chaos, **not** a rank collapse, and not a
property of the problem. §62's other half stands: the kink smoothing and the `icc = 11`
removal moved the stall from a trust-box corner 0.34 away to a near-feasible point 3.6e-3
away.

*(Not independently reproduced here, unlike most findings in this file: three converging
measurements on a third-party package's limitation, where the conclusion changes none of
this port's code.)*

## 65. `jaxipm` cannot be used, for two independent structural reasons (2026-09-06)

Evaluated after §58 established that a genuinely constrained forward-mode method works on
the easy problem and stalls on the hard one, and that `optimistix`'s solvers are all
reverse-mode. `jaxipm` (arXiv 2606.26341, *"Scaling Nonlinear Optimization: Many Problems
One GPU"*) is a GPU-batched IPOPT in JAX -- interior-point, so it does no active-set
identification and might have sailed past §64's failure mode entirely.

It cannot run here, for two reasons that are independent and neither of which is a fight
worth having:

1. **Its sparse KKT solve is cuDSS-only, with no CPU path in the source.** Every core
   module does an unguarded `from spineax import cudss`; the import fails with
   `ImportError: libcudss.so.0`. Its README requires **CUDA 13** (this machine has 12.6)
   plus the cuDSS/cuBLAS/cuSPARSE/NCCL stack.
2. **Its derivative front end cannot consume this graph, and this would remain fatal with
   a GPU.** `jaxipm` uses **no JAX autodiff at all** -- it goes `jax.make_jaxpr` ->
   `jax2sympy` -> symbolic `sympy.diff` -> back to JAX, in order to extract exact sparsity
   for the KKT system. Of **63 distinct primitives** in `helias_5b` MDF's objective jaxpr,
   **23 have no translation rule**, including `while`, `scan`, `cond`, `custom_jvp_call`,
   `linear_solve`, `stop_gradient` -- and cottax's own batching primitives `unvmap_any`,
   `unvmap_max`, `nonbatchable`, `select_if_vmap`. Those are structural to how cottax
   works, not incidental, and `helias_5b` is the *smallest* problem in the matrix.

**Env intact**: installs were `--no-deps` pure-Python (`jaxipm`, `spineax`, `jax2sympy`,
`sympy`, `sympy2jax`); `jax`/`jaxlib`/`numpy` unchanged, `cottax` still editable under
`~/jaxgraph/src`, `-k "driver or importer"` still 94 passed.

**Side finding worth keeping**: on the real `helias_5b` `condition_map`, `jax.jit`,
`jax.grad`, `jax.jacfwd` and `jax.jacrev` all work -- independent confirmation that §59's
reverse-mode fix holds on an assembled block.

## 66. The constraint census: why more conditions than variables is fine (2026-09-06)

The question worth asking of any of these blocks: *how can there be more constraints than
optimiser variables?* Answered by measurement, from the input files and from the assembled
graphs, which agree exactly.

**Only equalities consume degrees of freedom.** Per file, `ixc` / `icc` /
`n_equality_constraints`, and the graph-side split confirmed identical on all seven:

| configuration | unknowns | obj | eq | ineq | DOF left |
|---|---:|---:|---:|---:|---:|
| `helias_5b` | 3 | 1 | 2 | 2 | 1 |
| `stellarator_helias` | 8 | 1 | 2 | 12 | 6 |
| `st_regression` | 14 | 1 | 3 | 15 | 11 |
| `low_aspect_ratio_DEMO` | 19 | 1 | 4 | 21 | 15 |
| `large_tokamak_nof` | 20 | 1 | 3 | 23 | 17 |
| `large_tokamak_eval` | 2 | -- | 2 | 23 | **0** |
| `spherical_tokamak_eval` | 3 | -- | 3 | 15 | **0** |

The two zero-DOF rows are the two `i_process_run_mode = -2` files -- **root finds, not
optimisations** -- and they are exactly square by construction. SAND adds coupling unknowns
and an equal number of residual equalities, leaving the PROCESS-native inequality count
untouched.

**No arm is over-determined at its solution.** Equalities plus *active* inequalities
against unknowns, at the converged point:

| configuration | arm | active set | unknowns |
|---|---|---:|---:|
| `stellarator_helias` | MDF | 2 + 4 = **6** | 8 |
| `stellarator_helias` | SAND | 8 + 4 = 12 | 14 |
| `helias_5b` | MDF | 2 + 0 = 2 | 3 |
| `large_tokamak_nof` | MDF | 3 + 7 = 10 | 20 |
| `low_aspect_ratio_DEMO` | SAND | 11 + 10 = 21 | 26 |
| `st_regression` | MDF | 3 + 4 = 7 | 14 |

Worst margin in the matrix is 21 of 26. So an over-determined KKT system explains no
observed stall anywhere -- consistent with §64.

**`icc = 11` was the only inert constraint.** Zero rows fail the
`_name_singular_equalities` test on any arm, at cold start or converged point, and
`boundary --inert` independently reports zero structurally-inert driven conditions on all
seven files. No inequality is slack by a suspicious margin either -- the largest are O(1)
to O(14) in scaled units, nothing near the orders-of-magnitude gap that would suggest a
units or wiring defect. **Caveat, as inference**: a topological reachability test would not
catch an *algebraic-cancellation* tautology like `c11` -- that needed the numeric Jacobian
(§46, §52) -- so "zero" here is reassuring rather than a proof for switch branches outside
this matrix.

**One finding nobody was looking for**: the two root-find files evaluate their inequalities
and **violate most of them** -- 20 of 23 on `large_tokamak_eval`, 14 of 15 on
`spherical_tokamak_eval`. PROCESS's `fsolve` mode reports them and enforces none. That is
the source's design, not a port defect, but it is worth knowing before anyone reads an
evaluation run's constraint output as a feasibility statement.
