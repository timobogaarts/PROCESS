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
   plus `QPcyc: 2`. (**§69 corrects that gloss**: `QPcyc` is a bitmask and 2 is
   `reached_max_iter`, *not* the ping-pong "cycling" bit -- the loop exhausts its budget
   rather than tripping the dedicated cycling detector.) Raising the budget **10x**
   (200x100 against 20x20) produces a **bitwise-identical trajectory** -- same `f`, same step sizes
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

## 67. `jaxipm`, second pass: the sparsity blocker dissolves, the driver wall does not (2026-09-06)

§65 rejected `jaxipm` for two reasons. Both were assumptions worth testing rather than
re-asserting, and testing them changed one verdict completely.

### Blocker 2 is soluble, and the reason is the rewrite's own thesis

§65's reading was "its derivative front end is `jax2sympy`, and 23 of 63 primitives in our
jaxpr have no translation". True, but it frames the problem backwards. **`jaxipm` uses a
CAS to *recover* the KKT sparsity pattern — and recovery is only necessary because a jaxpr
has lost it.** Which condition reads which unknown is exactly what a cottax `Graph`
declares. We never lose it, so we never need to recover it.

Reading the source bears that out. `jaxipm.initialization.initialize_common_problem` calls
`jax2sympy` in two places: `get_sparsity_pattern` (lines 243-248, **unconditional**,
building the six COO arrays) and `sparse_jacobian_sym`/`sparse_hessian_sym` (lines 277-282,
the derivative *values*, already skippable through the existing `override_sparse_funcs`
argument). Both bottom out in `jaxpr_to_sympy_expressions`, which is where the 23
primitives bite — so `override_sparse_funcs` alone is not enough, as had been hoped.

**But `get_sparsity_pattern` is bound by a plain `from ... import` into
`jaxipm.initialization`'s namespace, so it is monkeypatchable from outside** — no fork.
Supply the pattern from the graph, the values through `override_sparse_funcs` with
`jax.jacfwd` and `jacfwd(jacfwd(...))`, and `jax2sympy` is bypassed entirely. A real,
near-supported path.

### The sparsity pattern check, which is worth keeping regardless

Structural pattern from `Graph.readers`/`.descendants` against the numerical pattern from
`jax.jacfwd` at four points (the primed cold start plus three ±2 % perturbations,
log-scaled to avoid a huge-magnitude variable reading as a false zero), on `helias_5b` MDF
(3 unknowns x 5 conditions):

**14 of 15 entries agree.** The one disagreement is the objective against `hfact`, where
the graph shows a reachable path and the analytic partial is exactly `0.0` at every
sampled point. **Structure over-declares and never under-declares** — the safe direction
for a KKT pattern, since a wrongly-dense entry costs a little work and a wrongly-sparse one
gives a wrong step. (The disagreement also reads as correct physics: `helias_5b` minimises
`i_figure_merit = 7`, capital cost, which this configuration computes from geometry and
masses; `hfact` is a confinement multiplier and enters through the power-balance and
net-electric constraints, not through cost.)

**So the port has a validated conservative sparsity pattern available to any sparse
solver, derived structurally and cross-checked numerically.** That outlives `jaxipm`.

### Blocker 1 is real, and is a driver-version wall

`nvidia-cudss-cu12` is a genuine CUDA-12 build — `readelf` shows it needs only
`libcublas.so.12` and no `libcudart` at all — so §65's "needs CUDA 13" was indeed a
packaging assumption about *that* library. It was also not the binding constraint.
**`spineax`'s own prebuilt native extension (`pbatch_solve.abi3.so`) is independently
linked against `libcudart.so.13` and `libcublas.so.13`**, whatever cuDSS is supplied, and
its README states CUDA 13 flatly with no CUDA-12 build path.

Closed empirically rather than by reading: staging `libcudss.so.0` (cu12) plus cu13
`libcudart`/`libcublas` on `LD_LIBRARY_PATH` makes `import spineax.cudss` **succeed**, and
the first real call then fails at runtime with

```
INTERNAL: spineax token: cudaGetDevice failed:
CUDA driver version is insufficient for CUDA runtime version
```

CUDA 13 requires driver >= 580.65.06; this machine has **560.35.03**. That is a
root-plus-reboot system upgrade, not a packaging question, so `helias_5b` and
`stellarator_helias` were never reached.

**Verdict: a second negative, but a better-grounded one.** The sparsity-injection idea
holds up under inspection and would likely have worked; the wall is the NVIDIA driver on
this machine. If that driver is ever upgraded, this is worth exactly one more attempt, and
the two hooks needed are now identified by line number.

**Env verified after**: CPU `process_port` untouched (no installs); GPU env's temporary
packages uninstalled, `jax.devices()` -> `[CudaDevice(id=0)]`, `cottax.__file__` under
`~/jaxgraph/src`, `tests/unit` -> **846 passed**; disk back to ~29 GB free after ~525 MB of
scratch downloads were deleted.

## 68. The matrix on a GPU: same answers, 2.4x slower, and one row that disagrees (2026-09-06)

Built `process_port_gpu` (`CLAUDE.md` § the environment) and ran the warm matrix on a
**Quadro T1000, 4 GB VRAM**, `XLA_PYTHON_CLIENT_PREALLOCATE=false`. The CPU side was
**re-run** rather than compared against the published file, which predated both today's
`icc = 11` removal and the `vacuum.py` conversion. Published as
`reference_warm_matrix.txt` (CPU) and `reference_warm_matrix_gpu.txt` (GPU).

**Caveat on control**: the CPU env is jax **0.11.0** and the GPU env is **0.11.1**. Prior
evidence says that drift is inert (both suites reproduce their counts across it), but this
is a backend comparison with a patch-version confound, not a clean one.

### 23 of 24 rows: identical iteration counts, identical `objf`

| configuration | arm | drv | it | CPU s | GPU s | GPU/CPU | CPU ms/call | GPU ms/call |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `stellarator_helias` | MDF | VMCON | 41 | 0.726 | 1.541 | 2.12x | 2.12 | 9.76 |
| `helias_5b` | MDF | VMCON | 4 | 0.135 | 0.241 | 1.79x | 1.97 | 10.04 |
| `large_tokamak_nof` | MDF | VMCON | 7 | 0.424 | 1.200 | 2.83x | 11.20 | 31.28 |
| `low_aspect_ratio_DEMO` | MDF | VMCON | 11 | 0.581 | 2.351 | **4.05x** | 15.30 | 84.60 |
| `low_aspect_ratio_DEMO` | SAND | VMCON | 79 | 2.594 | 15.067 | **5.81x** | 7.98 | 72.38 |
| `low_aspect_ratio_DEMO` | SAND | SLSQP | 17 | 0.381 | 3.444 | **9.04x** | 2.63 | 33.94 |
| `st_regression` | MDF | VMCON | 10 | 0.329 | 0.563 | 1.71x | 4.49 | 8.41 |

**Median 2.41x slower, min 0.81x, max 9.04x** over the 23 rows converging on both. The one
sub-1.0 row (`large_tokamak_nof` MDF SLSQP) is CPU-side noise -- its *per-call* cost still
goes 2.77 -> 12.60 ms.

**And the ratio grows with the graph.** `low_aspect_ratio_DEMO` is the biggest
configuration (247 nodes, 19 unknowns, 25 constraints) and the worst by a wide margin,
4.05-9.04x, while `st_regression` and `helias_5b` sit at 1.7-2.2x. That is the expected
signature: per-call cost is dominated by per-kernel fixed costs, so more graph means more
overhead and no more parallelism to recover it. (**§70 corrects the arithmetic here**: 1 648
was a *CPU* fusion count, where fusions are loop nests and not kernels at all; the GPU
compiles this block to **134** kernels, and at 57 us of gap apiece launch overhead is a
minority share -- FP64 running **16.4x** slower than FP32 on this card is the larger term.) Per-call goes from **0.8-15 ms on CPU
to 3.8-85 ms on GPU**.

**Nothing here is a GPU indictment** -- it is the workload being the wrong shape, as
predicted before the measurement: ~28k *scalar* ops, ~100 KB of runtime buffers, no
arithmetic intensity, and ~97 % of a cold row is compilation. The plausible win remains
**batching** (`vmap` over independent scan points), which this matrix does not test.

### The one row that disagrees, and what it says

```
stellarator_helias SAND SLSQP    CPU: 501 iters, cap(500), objf 1.21843409
                                 GPU: 129 iters, converged, objf 1.21890181
```

Nothing about that configuration changed today. What differs is the backend's floating
point -- different fusion, different FMA contraction, last-bit differences compounding
through the trajectory. On the GPU the arm escapes the period-2 cycle of §47 and converges
in 129 iterations, to a point **3.5e-4 from VMCON's `1.21848284`** rather than to VMCON's
answer.

**A second, independent sign of the same sensitivity**: the CPU value for that row moved
from the previously published `1.21846128` to `1.21843409` on this re-run, with the only
intervening change being the `vacuum.py` `while_loop` -> `vmap` conversion (§59) -- which
is bit-identical on seven of eight test cases and 1 ulp on the eighth. **Every converging
row's `objf` is unchanged by that conversion; only the capped, non-converged one drifts.**

So: §47 recorded that before the Ward-kink smoothing this arm swung between 87 and 333
iterations on a +-1 ulp draw. The smoothing removed the kink, and this says the underlying
sensitivity is still there -- a change of *backend*, or a 1-ulp change in one unrelated
node, is enough to move it between cycling at the cap and converging elsewhere.

**That corroborates §64 from a new direction.** If the failure were geometric -- a genuinely
infeasible corner -- a last-bit nudge would not rescue it. An active-set QP cycling near a
corner is exactly the kind of thing a tiny numerical difference knocks loose. Two
independent lines now say the same: the problem is fine, the QP layer is not.

## 69. `slsqp_jax` at source level: a known weak point, one real knob, and size is not it (2026-09-06)

§64 concluded the `stellarator_helias` stall is `slsqp_jax`'s QP layer rather than the
problem. Reading the source (v0.21.1, matching GitHub `lucianopaz/slsqp-jax`) confirms it,
sharpens it, and corrects one thing I wrote.

### The loose end is closed: `QP ok: False` is not over-reported

A fresh verbose run of the **converging** configuration, `helias_5b` MDF at the same
settings, reproduces the recorded answer exactly (2 steps, `objf = 0.7642155252631632`
against the recorded `0.764215525`) and reports **`QP ok: True, QPiter: 0, QPcyc: 0` on
every step**. The flag only fires on the run that is struggling, so §64 stands as read.

### What the QP actually is

A **primal active-set method** -- add most-violated, drop most-negative-multiplier --
in `slsqp_jax/qp/active_set.py::run_active_set_loop`, shared by all three QP strategies.
Both configurations (`m_eq = 2`) route through `solve_qp_proximal`, which absorbs
equalities into an augmented-Lagrangian penalty and active-set-manages only the
inequalities; the inner equality-constrained subproblem uses `ProjectedCGCholesky`.

**Correction to §64.** I glossed `QPcyc: 2` as "active-set cycling". It is a **bitmask**
(`slsqp/_step_body.py:863`): bit 0 is `ping_ponged`, bit 1 is `reached_max_iter`. A value
of 2 is **bit 1 only** -- the loop exhausting `qp_max_iter`, with the dedicated ping-pong
detector *not* firing. And `QP ok` is `final_converged = converged & ~reached_max_iter`
(`qp/proximal.py:185`), so hitting the cap forces `False` by construction. Combined with
the measured budget-invariance -- 10x the budget, bitwise-identical trajectory -- the
honest reading is that it **exhausts whatever budget it is given without converging**.
That is cycling in effect; it is not the flag named "cycling", and I should not have
quoted the flag as if it were.

### It is a documented weak point, and this package is end-of-line

- Its own README has a **"QP anti-cycling"** section describing this failure mode.
- `slsqp_jax/results.py` carries a dedicated result code **`infeasible_stationary`** for
  exactly this scenario, with a default-enabled `RestorationConfig` fallback.
- **PR #54, "Harden QP convergence" (2026-04-22), added a ping-pong anti-cycling
  short-circuit and was reverted four hours later** (`714ddf2`) because it leaked
  ill-conditioned multipliers into the outer merit penalty. The shipped default is
  `ping_pong_threshold = 2**31 - 1`, i.e. disabled.
- **v0.21.1 is the final release.** All 35 commits since (to 2026-08-07) build a
  replacement package, `sqpdax`, in the same repository, whose stated goal is to *"enable
  both active set and interior point based algorithms"*. The maintainer is replacing the
  active-set-only design rather than patching it.

### One knob is a real win, and two make it worse

`stellarator_helias` MDF, one flag changed at a time from baseline:

| variant | `objf` | vs VMCON `1.21848284` | `max\|eq\|` | `min ie` |
|---|---|---:|---:|---:|
| baseline | 1.222062667 | 3.58e-3 | 4.41e-3 | -1.79e-4 |
| `qp_ping_pong_threshold = 3` | 1.107193558 | 1.11e-1 **worse** | 7.74e-3 | -8.18e-2 |
| **`active_set_method = "lpeca_init"`** | **1.218082014** | **4.01e-4 (9x better)** | **7.67e-7** | -4.10e-4 |
| `proximal_tau = 0.0` | 1.114645897 | 1.04e-1 **worse** | 8.91e-4 | -7.48e-2 |

**LPEC-A active-set identification is an algorithmic win, not a budget one** -- 18 steps
against a 30 cap, and `max|eq|` improves four orders to 7.67e-7. The ping-pong
short-circuit making things worse is consistent with the maintainer's own revert rationale.

**A gap worth naming**: `expand_factor`, the EXPAND tolerance-ramp rate the anti-cycling
mechanism itself depends on, is a real parameter of `solve_qp` but is **never exposed**
through `QPConfig` or `compat.py`'s option mapping, and `SLSQP._solve_qp_subproblem` does
not pass it. It is silently pinned at 1.0 with no user escape hatch.

### Size is conclusively not the discriminator

| problem | vars / eq / ineq | `objf` | vs VMCON |
|---|---|---|---:|
| `helias_5b` | 3 / 2 / 2 | 0.764215525 | **9.3e-09** |
| `stellarator_helias` | 8 / 2 / 12 | 1.222062667 | 3.6e-03 |
| `st_regression` | 14 / 3 / 15 | -10.718742 | **5.87**, `max\|eq\| = 0.86` |
| `large_tokamak_nof` | 20 / 3 / 23 | **1.600000000** | **0.0**, `max\|eq\| = 1.7e-10` |

**The largest problem converges essentially exactly, and two smaller ones fail.** So the
discriminator is constraint geometry, not dimension. And `st_regression` fails far worse
than `stellarator_helias` -- badly infeasible rather than near-feasible -- which suggests a
distinct or more severe degeneracy there rather than the same mechanism scaled up.
**Uninvestigated**, and it is the obvious next thread if this solver is pursued.

**Verdict**: a genuine limitation of the active-set QP as shipped, on this constraint
geometry -- not model chaos (§64's LP found a feasible linearised step at the stall), not
size, and not our usage. It is a recognised structural weak point that the maintainer is
addressing by replacing the design. If `slsqp_jax` is used here, **use
`active_set_method="lpeca_init"`**.

## 70. What a "fusion" actually is on each backend, and a correction (2026-09-06)

§68 explained the GPU's 2.4x slowdown as *"launch overhead over ~1 648 tiny kernels"*.
**That sentence is wrong in two independent ways**, and the word "kernel" was doing work
it should not have been.

**On XLA CPU there are no kernels and no launches.** A `fusion` HLO instruction is a
sub-computation that LLVM turns into a **loop nest inside the single compiled function**.
Nothing is dispatched, nothing crosses a device boundary. So a fusion count on CPU says
nothing about launch cost, because there is none.

**On XLA GPU a fusion *is* a real CUDA kernel** with a real launch. But the counts differ
enormously between backends, and 1 648 was a *CPU* count -- from `st_regression` SAND --
which I then reused to explain GPU behaviour. Measured on the same block
(`stellarator_helias` MDF, the fused value+Jacobian program):

| backend | optimised instructions | **fusions** | parameters |
|---|---:|---:|---:|
| CPU | 36 619 | **816** | 6 539 |
| GPU | 14 776 | **134** | 2 362 |

**The GPU backend fuses roughly six times more aggressively** -- 134 kernels, not 1 648 --
and emits under half the instructions, because it has a much stronger incentive to avoid
memory round-trips than the CPU backend does.

### So what does explain the gap?

`stellarator_helias` MDF is 2.12 ms/call on CPU and 9.76 ms/call on GPU, a **7.64 ms**
difference over **134 kernels** -- **57 us per kernel**. CUDA launch overhead is a few
microseconds, so **launch overhead alone does not explain it**, and §68's account was too
glib even after the count is fixed.

What else is in that 57 us, in decreasing confidence:

- **FP64 at 1/16 rate on this card, measured**: a 1024x1024 matmul runs at **1254.7
  GFLOP/s in fp32 and 76.6 GFLOP/s in fp64** -- a **16.4x** penalty (consumer Turing
  specifies 1/32; the fp32 figure is presumably not at peak either). **PROCESS is float64
  throughout**, so every kernel pays this. This is the single most important number for any
  future GPU plan, and it is hardware-specific: a data-centre card at 1/2 rate is an
  entirely different proposition from this 35 W laptop part.
- **Global memory round-trips between kernels.** Values that stay in registers within a
  CPU loop nest must be written to and re-read from global memory between GPU kernels.
- Small grids: each kernel here has almost no parallel work, so per-kernel fixed costs
  dominate whatever they are.

**Not decomposed further, and it would need a profiler** (`nsys`) to attribute the 57 us
properly. What is established: the fusion counts above, the FP64 ratio, and that the
launch-overhead story is at best a minority share rather than the explanation.

**The conclusion of §68 is unchanged** -- same answers, median 2.41x slower, ratio growing
with graph size -- but its *reason* was partly wrong, and the corrected reason points
somewhere more actionable: on hardware with real FP64 throughput the picture could differ
substantially, and that is worth knowing before anyone writes off GPU work on the strength
of these numbers.

## 71. Batching: the GPU wins, and the crossover is at ~256 points (2026-09-06)

§68 and §70 measured single solves, where the GPU is a median 2.4x slower, and both said
the plausible win was **batching** -- `vmap` over many independent points, which is the
shape of `process/core/scan.py` and the premise of the `jaxipm` paper. That was asserted
and untested. Tested now.

`vmap` over the `stellarator_helias` MDF condition evaluation (8 unknowns, 15 conditions),
identical code both backends, warm, mean of 5 calls. **Milliseconds per point:**

| batch | CPU | GPU | winner |
|---:|---:|---:|---|
| 1 | 0.865 | 10.738 | CPU **12.4x** |
| 4 | 0.342 | 2.121 | CPU 6.2x |
| 16 | 0.196 | 0.557 | CPU 2.8x |
| 64 | 0.143 | 0.245 | CPU 1.7x |
| **256** | **0.150** | **0.140** | **GPU 1.07x -- crossover** |
| 1024 | 0.176 | 0.075 | **GPU 2.3x** |
| 4096 | 0.203 | **0.060** | **GPU 3.4x** |

**The two curves go in opposite directions.** The CPU bottoms out around 64 points at
0.143 ms and then gets *worse* -- 0.203 ms at 4096, a 42 % regression, which is cache
pressure as the working set outgrows L2/L3. The GPU improves monotonically the whole way,
0.865 -> 0.060 ms, and **had not stopped improving at 4096**.

Batching speedup from the same backend's own single-point cost: **CPU 6.1x at best, GPU
78.7x at 256 and still climbing.** That gap is the whole story -- the CPU has no
parallelism left to find and the GPU has barely started.

**Compile cost is essentially flat in batch size**: 4.3-5.1 s (CPU) and 5.7-6.2 s (GPU)
across a 64x range of batch sizes. So batching is nearly free at compile time -- you pay
one compile per *shape*, not per point, which is what makes a scan sweep viable at all.

**And this is despite the FP64 penalty.** §70 measured this card running float64 at
**1/16.4** of float32, and PROCESS is float64 throughout; the 3.4x above is what survives
that. On a card with 1/2 FP64 the crossover would move sharply left and the asymptote
sharply down.

### What this does and does not establish

**Does**: the model evaluation batches well, the GPU wins decisively past a few hundred
points, and the crossover is low enough (~256) to be reachable by a realistic scan.
`process/core/scan.py` sweeps one input over many points, each an independent solve --
exactly this shape.

**Does not**: this is the **evaluation**, not the solve. Both drivers are host-side
`jax.pure_callback`s and `_sqp_callback`'s docstring is explicit that *"`vmap` over this is
sequential"* -- neither library vectorises, so a batch of *solves* is a host-side loop
today. Realising the number above needs a jax-native solver, which is precisely what §55,
§58 and §67 have been circling. It also does not address that points in a batch converge at
different rates; the `jaxipm` paper's "iteration-level batching" (replace finished problems
mid-batch) is the published answer to that, and is the part of that work most worth
stealing even though the solver itself is unreachable here (§67).

**So the single-solve numbers were never the interesting question, and the batched ones
justify the direction**: the reason to want a jax-native constrained solver is not that it
beats VMCON on one problem -- it does not -- but that it is the only thing that can turn a
scan into one batched program.

## 72. The knife-edge hunt: `wp_width_r_min` is non-smooth by construction (2026-09-06)

Chasing knife-edge behaviour has been the highest-yield thread in this port -- it found the
Ward `sqrt` kink the optimiser sat 5.5e-08 from and crossed on 46 % of steps, and the
`x ** p`-at-zero defect that put 46 non-finite cells in a Jacobian. §68's finding that a
*backend change* flips `stellarator_helias` SAND between capping and converging said the
sensitivity had not gone away. So: hunt systematically.

### The numerical sweep found nothing at the converged MDF points

Using the harness's own detector -- `_harness/finite_difference.fd_gradient_with_error`
(Richardson-extrapolated FD with an honest per-point error bar) against `jax.jacfwd`, at
`Tier1Contract`'s `allowed = 25 x error_bar` -- at each configuration's **converged** point,
every condition against every design coordinate:

- `stellarator_helias` MDF, 8 x 15: **0 findings**
- `helias_5b` MDF, 3 x 5: **0 findings**
- `large_tokamak_nof`: solved, analysis incomplete at the time box; four configurations not
  reached.

A genuine negative on what was checked: **the Ward smoothing holds**, and the converged MDF
points are smooth to the harness's own error bars.

### The real finding is structural, and it explains six earlier sections

`wp_width_r_min` -- the SAND coupling unknown at the centre of §47's period-2 cycle -- **is
non-smooth by construction, in two independent ways**, both verified by reading the source:

1. **`coils.py:296-333`, `intersect()`**: the crossing of two `jnp.interp`
   **piecewise-linear** curves, found by bisection plus a Newton polish. The crossing point
   is continuous in its inputs but only **piecewise**-smooth -- there is a derivative kink
   whenever the crossing moves across one of the ~200 interpolation breakpoints. This is a
   structural analogue of an `argmax` kink, not a numerical artefact.
2. **`calculate.py:797`**: `wp_width_r_min = jnp.maximum(dx_tf_turn_general**2, wp_width_r_min)`
   -- a hard floor clamp applied immediately downstream of (1). **§82 measured this and it
   is NOT live**: a ~200x margin between the two arguments, winner never flips on either
   arm. Only mechanism (1) is real on this configuration.

**This completes a mechanism that has been half-explained since §47.** The chain:

- `wp_width_r_min` is piecewise-smooth with kinks, by construction.
- **MDF never exposes it**: it is an inner `RootFind` converged inside every evaluation, so
  the outer solver sees only its converged value -- which is why `stellarator_helias` MDF
  converges under SLSQP in 27 iterations.
- **SAND exposes it** as an unknown with a residual equality, so the outer SQP is handed a
  **non-smooth constraint** -- which is why the same file's SAND arm caps at 500.
- An active-set SQP assumes smoothness in exactly the place this violates it, and cycles
  (§64, §69); VMCON's per-constraint multipliers tolerate it (§47).
- And a 1-ulp change anywhere upstream can move the crossing across a breakpoint, which is
  why a **backend change** flips the outcome (§68).

**So §64 and §69 were right that the QP layer is where it manifests, and incomplete about
why.** It is not only that `slsqp_jax`'s active-set QP is weak -- it is that SAND hands it a
problem that is not differentiable at the points it cares about. Both are true and they
compose. §56's refutation of the "expose `FixedPoint`s, keep `RootFind`s internal" rule
stands on its own evidence, but this supplies the property that rule was groping for:
**not "is it a `RootFind`" but "is its residual smooth".**

**The obvious follow-up, not done**: probe the SAND arm at this specific site -- does the
trajectory actually cross interpolation breakpoints, and how often? That is the direct
analogue of the Ward measurement (5.5e-08 away, crossed on 46 % of steps) which turned that
kink from a suspicion into a fix.

### Static sweep: one confirmed-attainable kink and four candidates

Already known and excluded: the Ward kink (smoothed), `x ** p` at zero (fixed), and
`sqrt(jnp.maximum(0, x))` -- **zero remaining occurrences** in `models/`, docstrings only.

**Confirmed attainable, and sitting exactly on it**: the divertor tie at
`divertor.py:296-302`, `f_p_div_lower == 0.5`, deliberately documented and unsmoothed.
**`spherical_tokamak_eval.IN.DAT` and `st_regression.IN.DAT` both set `f_p_div_lower = 0.5`
exactly** (verified), so those runs sit precisely on the tie at *every* evaluation, not
merely near it at a solution.

Candidates, ranked by attainability, none yet shown to be reached:

| site | expression | why it could be hit |
|---|---|---|
| `build.py:916-940` | `maximum(r_tf_outboard_mid_unrippled, r_tf_outboard_midmin)` | PROCESS's "ripple too large, move the TF leg" switch; a ripple-constrained optimum sits *on* it |
| `pfcoil/superconductor.py:313,382` | `maximum(abs(b_pf_coil_peak), abs(bpf2))` | inner/outer coil-edge field selection feeding `j_crit`; PF optimisation plausibly balances the two |
| `physics.py`, `force_positive_separatrix_power` | `p /= 1 - exp(-p)` | reproduces PROCESS's own `0/0 -> nan` at `p == 0`, **unsmoothed** unlike the Ward kink |
| `confinement_time.py:1571-1580` | `minimum` of two confinement scalings | live only if `i_confinement_time` selects that model; which configurations do is unchecked |

Deprioritised with reasons: `pfcoil/geometry.py:339`'s `sqrt(r^2 - z^2)` (inherited PROCESS
guard; note `sqrt` of a negative gives `nan`, not `inf`, so the `isinf` kludge may not catch
every case -- inherited, not new), `stresses.py`'s elliptic integrals near `m -> 1` (real
singularity, attainability unclear), and several `maximum`/`minimum` clamps at fixed small
constants, which are numerical floors rather than physical crossings.

## 73. A real gradient anomaly on `large_tokamak_nof`, partially characterised (2026-09-06)

The §72 sweep, run to completion on more configurations, found **48 gradient-agreement
violations at `large_tokamak_nof` MDF's converged point** -- against the harness's own
Richardson-extrapolated FD at `Tier1Contract`'s `allowed = 25 x error_bar`. All 48 trace to
**2 of 20 design variables**, `.build.dr_cs` and `.build.dr_bore`, each disagreeing across
9-12 of 27 conditions, by 100-1500x the allowed tolerance.

That was reported as the session's strongest finding. **It is real but it is not yet
understood, and two checks were needed before it could be written down as anything.**

### It is not PROCESS's FD scheme

The obvious deflation -- `jacfwd` differentiates the block with its inner solve implicit
while PROCESS's FD re-runs a pipeline converged only to `rtol = 1e-6`, so the two measure
different things -- **does not hold.** Finite-differencing the **port's own** block
evaluation (central, through `drivers.scaled_problem`'s `evaluate`) reproduces the same
disagreement on the same variables and largely the same conditions:

| condition / variable | `jacfwd` | central FD (same function) | relative |
|---|---:|---:|---:|
| `c32` / `dr_bore` | -0.471039 | -0.455907 | **3.21e-02** |
| `c31` / `dr_bore` | -0.604695 | -0.587011 | 2.92e-02 |
| `c65` / `dr_cs` | -0.051089 | -0.049920 | 2.29e-02 |
| `c36` / `dr_bore` | -7.37663 | -7.36540 | 1.52e-03 |

So the anomaly is **internal to the port**, not an artefact of comparing against PROCESS.

### But the step-size sweep matches neither explanation

The decisive test between "kink" and "loose inner solve" is how the disagreement behaves as
the FD step shrinks: a kink straddled by large steps disappears once the step fits on one
side; an implicit-versus-re-solved derivative is roughly step-independent. Measured:

| `h` | `c31`/`dr_bore` | `c32`/`dr_bore` | `c65`/`dr_cs` | `c36`/`dr_bore` |
|---:|---:|---:|---:|---:|
| 1e-03 | 2.937e-02 | 3.226e-02 | 2.326e-02 | 1.530e-03 |
| 1e-04 | 2.935e-02 | 3.225e-02 | 2.323e-02 | 1.528e-03 |
| 1e-05 | 2.924e-02 | 3.212e-02 | 2.289e-02 | 1.522e-03 |
| 1e-06 | 2.873e-02 | 3.156e-02 | 2.000e-02 | 1.496e-03 |
| 1e-07 | 1.805e-02 | 1.983e-02 | 8.937e-04 | 9.399e-04 |
| 1e-08 | **1.128e-03** | **1.239e-03** | 8.962e-04 | 5.869e-05 |

**Flat across three decades, then a collapse between 1e-6 and 1e-8.** That is neither
signature. A central difference on a smooth function has `O(h^2)` truncation error, so
1e-3 -> 1e-6 should improve by 1e6 and does not. A straddled kink's secant slope varies
with `h` rather than sitting flat. And a purely implicit-versus-re-solved mismatch should
not collapse at all.

The shape is what a **switch at a fixed distance of order 1e-7 in scaled coordinates**
would produce -- every step larger than that crosses it, every smaller step does not -- but
that is a reading of the shape, **not a located cause**. It could equally be an inner
driver changing iteration count above some perturbation size.

### What is and is not established

**Established**: a real, reproducible, above-threshold disagreement between the port's
analytic gradient and a finite difference *of the port's own function*, at a converged point
of the production tokamak configuration, localised to two build variables; converging to
~1e-3 as `h -> 1e-8` rather than being an outright wrong derivative.

**Not established**: the cause, and whether it matters. The 48-violation count is measured
against a threshold calibrated for **unit** functions in the Tier-1 harness; applying it to
a whole converged MDF block with inner drivers is a different test whose expected behaviour
nobody has characterised, so "48 violations" may not carry its usual meaning here.

**The next step is a bisection, not more sweeping**: `dr_cs` and `dr_bore` both feed
`r_tf_inboard_*` (`build.py:538`, checked -- a clean division, no kink) and propagate through
the confirmed physics<->build<->TF SCC to `build.py:916-940`'s
`jnp.maximum(r_tf_outboard_mid_unrippled, r_tf_outboard_midmin)` -- the ripple switch
flagged statically in §72. Evaluating *that* expression's two arguments along the same
perturbation would confirm or eliminate it in one run. **Not done.**

**Also from the same sweep, for the record**: `large_tokamak_eval` (root find, 2 x 2) is
clean, joining `stellarator_helias` MDF (8 x 15) and `helias_5b` MDF (3 x 5) at zero
findings. Three configurations remain unmeasured.

### The ripple `jnp.maximum` is eliminated (2026-09-06)

§72 flagged `build.py:916`'s `jnp.maximum(r_tf_outboard_mid_unrippled, r_tf_outboard_midmin)`
as the leading structural suspect for §73's anomaly, on the reasoning that both implicated
variables feed it through the physics<->build<->TF SCC. **Measured, and it is not the
cause.**

Watched with a `jax.debug.callback` -- which fires at runtime with concrete values, unlike
reading the tracer, which raises `TracerArrayConversionError` here -- across the whole
solve and under perturbations of `.build.dr_cs` and `.build.dr_bore` from 0 to 1e-3:

```
last call: unrippled = 13.8128079   midmin = 14.9836619
winner    : MIDMIN (ripple active)
gap       : 1.17085   (8.477e-02 relative)

.build.dr_cs    0:M(-1.05)  1e-8:M(-1.05)  1e-7:M(-1.05) ... 1e-3:M(-1.05)
.build.dr_bore  0:M(-1.05)  1e-8:M(-1.05)  1e-7:M(-1.05) ... 1e-3:M(-1.05)
```

The ripple constraint **is** active on this configuration -- `midmin` wins, so the TF leg
really has been moved out -- but the two arguments differ by **1.17 m, 8.5 % relative**, and
the winner does not flip at any step size. The switch is nowhere near. A `jnp.maximum` this
far from its own tie cannot contribute a kink.

**So the anomaly's cause is still unlocated**, and the most attractive structural
explanation is gone. What remains: another switch in the inboard-build chain, or the
implicit-versus-re-solved question that the step-size sweep did not cleanly settle either.
The next probe should walk the chain from `dr_cs`/`dr_bore`/`dr_tf_inboard` forward,
watching *every* `maximum`/`minimum`/`where` the same way, rather than guessing another
single site.

### Reproducible across configurations

The same sweep on `low_aspect_ratio_DEMO` MDF finds **24 violations** in
`.build.dr_tf_inboard` and `.build.dr_cs` -- again inboard radial-build variables, and
again the **same constraint family** (c16, c31-c36, c65) that `large_tokamak_nof` showed on
`.build.dr_cs`/`.build.dr_bore`. Two independent tokamak configurations, overlapping
variable sets, overlapping constraint sets.

**Both root-find configurations are clean** (`large_tokamak_eval` 2x2 and
`spherical_tokamak_eval`, 0 findings each), as are `stellarator_helias` MDF (8x15) and
`helias_5b` MDF (3x5). So the pattern is specific to the **tokamak inboard radial build**
under an optimisation arm, and is not a property of the harness's threshold applied to any
converged block -- four other converged blocks pass it cleanly. That last point matters,
because §73 hedged on exactly it.

### The sweep completed: all seven configurations, and two findings that are probably artefacts

| configuration | arm | findings |
|---|---|---|
| `large_tokamak_nof` | MDF | **48** -- `dr_cs`, `dr_bore` |
| `low_aspect_ratio_DEMO` | MDF | **24** -- `dr_tf_inboard`, `dr_cs` |
| `st_regression` | MDF | 9 -- see below |
| `stellarator_helias` | MDF | 0 |
| `helias_5b` | MDF | 0 |
| `large_tokamak_eval` | root find | 0 |
| `spherical_tokamak_eval` | root find | 0 |

`st_regression`'s nine are **structurally unlike** the build-variable cluster, and both
groups are more likely measurement artefacts than defects. Recorded because the reasoning
is worth not re-deriving, not because they are findings:

- **`c11` disagrees by a near-identical ~3.1e-7 *absolute* across four unrelated design
  variables** (`dr_cs`, `dr_bore`, `dr_tf_nose_case`, `dr_tf_wp_with_insulation`) -- a
  direction-independent bias, not a derivative blow-up, and large only *relative* to a very
  tight local error bar. **Likely cancellation, and here the constraint's own history
  explains it**: `c11` is `rbld - rmajor`, a difference of two O(m) quantities that the
  solve has pinned to ~0 (`max_eq = 2.9e-15`). Finite-differencing such a difference carries
  absolute error ~`eps * |rbld| / h`, which a Richardson bar computed on the FD *of the
  difference* will underestimate. (§52 established this same constraint is a tautology on
  the stellarator build path; on a tokamak it is live, which is why it appears here and not
  on the stellarators.) **Hypothesis, untested.**
- **`c62` disagrees at `.physics.f_nd_plasma_separatrix_greenwald`, whose value is
  `0.0010000000507170005`** -- i.e. sitting exactly on its lower bound of 1e-3. PROCESS's
  FD perturbs `x * (1 +- epsfcn)` symmetrically, so the backward probe goes **below the
  bound**, into a region the model was never meant to evaluate. **Probably a probe artefact
  of extrapolating past a box constraint**, and a bound-respecting one-sided FD would settle
  it. Worth noting the coincidence, though: `c62` is the same constraint implicated in
  §47-49's `stellarator_helias` SAND cycle.

**Neither changes the headline.** The reproducible, cross-configuration, large-relative
disagreement on the tokamak **inboard radial-build variables** remains the one finding that
is not explained away, and the one whose cause is still unlocated after the ripple switch
was eliminated.

## 74. NVIDIA Warp as a codegen target -- a design note, nothing measured (2026-09-06)

**This section contains no measurements of its own** (§76 supplies them: 335 live leaf
functions of which 92 % are mechanical, and the CoolProp caveat below is measured away). It is a design argument that existed
only in a session transcript, written down because every number it leans on *is* measured
elsewhere in this file and the argument is the thing that would otherwise be lost.

§63 priced emitting the graph as straight-line C: at `-O0`, ~28 000 ops cost 3.4 s and
242 MB against XLA's 15 s and 870 MB, and a scalar emitter would emit fewer still since
41.6 % of the StableHLO is shape plumbing (§50). Its two largest unpriced costs were
**derivatives** and **the GPU variant**. [NVIDIA Warp](https://github.com/NVIDIA/warp)
addresses both, and reframes the codegen problem into a much smaller one.

### The reframing, which is the load-bearing idea

Do **not** emit 28 000 operations. Write each leaf model function once as a `@wp.func`, and
**codegen only the kernel that calls them in the graph's order** -- roughly **250 call
sites**, not 28 000 ops. That turns a compiler backend into a small, auditable emitter.

And it lands on a seam the port already has. PROCESS's `calculate_*` staticmethods are
already pure functions of explicit arguments -- which is exactly what a `@wp.func` is. So
this is re-annotation plus a graph-ordered stitcher, not a rewrite. **The graph supplies
the call order**, which is precisely what cannot be recovered from PROCESS's imperative
`Caller._call_models_once` (`CLAUDE.md` § the current architecture). Same thesis as §67's
sparsity result: the port has structural information other tools must reverse-engineer.

### What it would fix, each against a measured number here

| | why it matters |
|---|---|
| **Shape plumbing disappears** | Warp is scalar-native (built for particles/robotics), so §50's 41.6 % `broadcast_in_dim`/`reshape`/`slice` never exists. That category is an artefact of wrapping scalars into tensors. |
| **AD comes free** | Warp generates adjoint code source-to-source -- §63's largest unpriced cost. **But see §80: this is an advantage over a hand-written C codegen path, NOT over JAX, which has had adjoints all along.** |
| **Batching is native** | `wp.launch(kernel, dim=n_points)`, one thread per scan point -- exactly `process/core/scan.py`'s shape, and §71 measured the GPU crossover at **~256 points, 3.4x by 4096**. |
| **CPU backend too** | Warp compiles to C++ via clang, so the whole thing is testable without a GPU. |

### SAND is the formulation to target, not MDF

Under **MDF** the inner couplings are converged by a solve, so differentiating through it
needs the implicit function theorem -- machinery that would have to be hand-built. Under
**SAND** there is no inner solve to differentiate through: the couplings are unknowns of
the outer problem, so evaluation collapses to **one straight-line residual pass** with no
nested Newton and no `while_loop`.

That inverts where SAND has been costing us. §53 and §56 found SAND's exposure of couplings
makes the outer problem *harder* for some solvers, and §72 found why (`wp_width_r_min` is
non-smooth, and SAND hands that non-smoothness to the outer SQP). **For codegen the same
property is exactly what you want.**

### And the solver need not be in the kernel

The natural split is **kernel = evaluation, host = optimisation**: the kernel computes
residuals and Jacobian, VMCON stays where it is. An in-kernel solver is needed only for
*per-thread independent* solves across a scan sweep -- and writing a per-thread SQP means
hand-rolling dense linear algebra, which is real work at any size.

**That is precisely the problem `jaxipm`'s paper solves**, and its "iteration-level
batching" -- replace finished problems mid-batch rather than waiting for the slowest -- is
the trick that avoids per-thread solvers entirely. §67 established `jaxipm` itself is
unreachable here (a driver-version wall), but **that idea is the part worth stealing**, and
it is independent of the package.

### Caveats, the first of which is hardware-specific and measured

- **FP64.** PROCESS is float64 throughout, and §70 measured this card running float64 at
  **1/16.4** of float32 (76.6 against 1254.7 GFLOP/s). A Warp kernel doing float64 physics
  hits exactly that wall on a T1000 and would not on a data-centre card at 1/2. **Any
  Warp/GPU plan has to state which hardware it assumes.**
- Warp's Python subset is restrictive and typed; some model bodies will not port cleanly.
- Register pressure: ~28k live values in one kernel would spill badly, hurting occupancy.
  Whether that matters depends on the batch, since a scan sweep needs throughput rather
  than occupancy -- **unmeasured**.
- The non-traceable nodes (CoolProp, `CLAUDE.md` § Difficulties) still need an answer, the
  same one they need under JAX.

### Status: unstarted, and the cheap first experiment is known

Nothing has been installed, written or measured. The decisive first step is the same one
§63 named and is unchanged by any of the above: **emit one block's value function, compile
it, and check bitwise agreement and timing against the JAX path.** Warp changes what that
emitter is written in and hands back derivatives for free; it does not change that this
number gates everything after it.

## 75. §73's anomaly located: an active clamp, not a defect (2026-09-06)

> **§77 refines this.** The clamp is active, but the mechanism is a **ratchet** --
> the input arm reads the value it writes back, so once the clamp fires the two
> arguments are equal by construction. And there *is* a way out: move the clamp into
> the constraint set. The "optimiser drives it there" reading below is not what
> happens.

§73 recorded a reproducible gradient anomaly on the tokamak inboard radial-build variables
and could not attribute it; §72's leading suspect was measured and eliminated. Located now,
and the answer changes what it means.

**Method**: rather than guess the next site, patch `jnp.maximum`/`jnp.minimum` **globally**,
attribute every call to its first frame inside `functional_process/models/`, record both
arguments with a runtime `jax.debug.callback`, and find the sites whose **winner flips**
under perturbations of the implicated design variables. 48 distinct switch sites on
`large_tokamak_nof`.

**One site sits on its own tie on both configurations:**

| configuration | site | `a` | `b` | relative gap | flips at |
|---|---|---:|---:|---:|---:|
| `large_tokamak_nof` | `tfcoil/base.py:192` | 0.0714836 | 0.0714836 | **4.13e-10** | `h = 1e-6` |
| `low_aspect_ratio_DEMO` | `tfcoil/base.py:192` | 0.0728348 | 0.0728348 | **0.00e+00** (bit-identical) | `h = 1e-8` |

That site is `dr_tf_plasma_case_from_input`:

```python
return jnp.maximum(
    dr_tf_plasma_case,                                   # the input value
    dr_tf_plasma_case_minimum(r_tf_inboard_in, dr_tf_inboard, n_tf_coils),
)
```

with `dr_tf_plasma_case_minimum = (r_tf_inboard_in + dr_tf_inboard) * (1 - cos(pi/n_tf_coils))`.
**`dr_cs`, `dr_bore` and `dr_tf_inboard` all feed `r_tf_inboard_in`/`dr_tf_inboard`**, so
they move the clamp's second argument directly -- which is exactly why §73's anomaly is
localised to those three variables and no others, on both tokamaks and on neither
stellarator.

### And this is almost certainly not a defect

**Unlike the Ward kink, the clamp is *supposed* to be active here.** Ward was a
`sqrt(x - 0.65)` singularity -- a modelling artefact with no physical meaning at the
crossing. This is a geometric floor: the TF plasma-side case cannot be thinner than the
winding pack's corner geometry requires, and **a cost-minimising optimiser will drive it
onto that floor and sit there.** Converging exactly onto an active `maximum` is what a
correct solution looks like, not a symptom.

The consequence for the measurement is then straightforward and not a port bug: at an
active clamp the derivative is **one-sided**. `jacfwd` reports one side; a **symmetric**
finite difference straddles the tie and reports a blend of both. They disagree by
construction, and neither is wrong for its own definition. That is the whole of §73's
"48 violations".

**So the harness's AD-versus-FD comparison will flag every active clamp**, and reading its
output requires knowing that. §73 hedged that "48 violations against a threshold calibrated
for unit functions" might not carry its usual meaning; the hedge was right, for a sharper
reason than the one offered.

**Left open, and it is a real question rather than this one**: `dr_tf_plasma_case_from_input`
is a **`FixedPointFunction`** -- the entering `dr_tf_plasma_case` is the same variable the
result is written back to (its own docstring says so). A fixed point sitting exactly on a
clamp is a more delicate object than either alone, and nobody has checked whether that
composition is well-posed at the tie. That is worth a look; smoothing the clamp is not,
since it would change physics that is doing its job.

### The other sites, for the record

- **`pfcoil/fields.py:90`**, `jnp.minimum(4 r r' / d, _S_MAX)` -- flips on both
  configurations, with the raw argument at exactly `1` against `_S_MAX = 0.999999`. This is
  the elliptic-integral guard doing its job (the raw value reaches 1 when a test point
  coincides with a current loop, where `log(1/t)` would diverge). Active, expected, same
  one-sided-derivative story.
- **`pfcoil/currents.py:188`** in `_solv()` -- the most flips on `low_aspect_ratio_DEMO`
  (17), with `a = b = 0` exactly. A clamp against zero where the quantity *is* zero.
  Benign-looking, unexamined.
- **`blankets/hcpb.py:499`**, `a = b = 0.3` exactly on both -- a clamp against a literal
  constant, so it never moves. Harmless.

## 76. How much work is a Warp port? Counted (2026-09-06)

§74 argued the Warp reframing -- write each leaf as a `@wp.func`, codegen only the kernel
that calls them in graph order -- without sizing it. The observation that makes it sizeable
is that `functional_process/models/**` is **already separate** from `functional_process/cottax/**`,
so nothing is annotated in place and the wrapping is generated. The question is therefore
just: **how many leaf functions, and how convertible are their bodies?**

Counted by building all seven reference machines, resolving each `ImplementedFunction`
node's wrapper to the plain function it delegates to in `models/**`, AST-parsing that
function plus one level of same-module helpers, and pattern-matching for Warp-relevant
constructs. **No Warp installed, no Warp code written.**

### The denominator

| | count |
|---|---:|
| module-level functions in `models/**` | **850** |
| **reached by the seven configs' graphs** | **335** (332 in `models/**` + 3 inline in wrappers) |

**39 %.** The rest is either swept into a live function's own classification as a helper, or
genuinely unreached by any reference configuration today.

### The classification, and the surprise

| bucket | count | share |
|---|---:|---:|
| **Trivial** -- scalar arithmetic, `exp`/`log`/`sqrt`/`tanh`, comparisons | 237 | **71 %** |
| **Easy** -- `where`/`min`/`max`, plain `if`/`for` with static trip counts | 70 | 21 % |
| **Work** -- array-valued intermediates, `interp`, `scan`, `scipy` | 28 | 8 % |
| **Blocker** | **0** | **0 %** |

**Blocker = 0, and it is verified rather than inferred.** CoolProp is genuinely present in
the port (`tfcoil/quench.py`, `blankets/hcpb.py`, `fw.py`, `engineering/*`), and §74 carried
"the non-traceable nodes still need an answer" as an open caveat. Measured by counting
`PropsSI` calls per phase:

```
large_tokamak_nof      assembly 150   cold solve 0   warm solve 0
stellarator_helias     assembly   0   cold solve 0   warm solve 0
```

**150 calls at machine assembly, zero inside any solve.** `quench.py` evaluates its helium
table eagerly in NumPy at assembly (`indat.py`), memoised, and hands the result to the node
as a **static** field; `hcpb.py` and `fw.py` document their CoolProp branches as dead on the
live switch settings, which the stellarator's zero confirms. No `pure_callback` and no
`process` import reaches a live node either. **So the known blocker is already outside the
graph** -- for these seven machines. §74's caveat is answered, with that scope.

### The work is 10 functions, not 28

Of the 28 "Work" functions, weighted by how many configurations reach them:

| configs | count | what |
|---:|---:|---|
| **7/7** | **3** | the `vmap`-over-`interp` impurity/charge-state pair, `calculate_profile_grid` |
| **5/7** | **7** | tokamak-universal: PF equilibrium currents/waveform/placement/turn-currents, bootstrap fraction, pedestal profile (`scipy` Simpson), PF power supplies |
| 3/7 | 8 | large-tokamak family: CS flux swing, plasma-initiation currents, PF-CS inductance, TF D-shape inductance (a real `lax.scan`) |
| 2/7 | 10 | narrow variants: no-CS PF family, stellarator quench (`interp`, 2 tables) and neoclassics, picture-frame TF stress |

**Ten are load-bearing regardless of scope**; the other eighteen bite only when that
topology is targeted. The hardest single item is the impurity/charge-state pair -- `jax.vmap`
over a `jnp.interp`-based per-species function, live on all seven.

### Emitter size: §74's "~250 call sites" checks out, with a correction to why

| configuration | nodes | leaf nodes | distinct wrappers *in that graph* |
|---|---:|---:|---:|
| stellarators | 154 | 150 | 147 |
| tokamaks | 244-249 | 241-245 | 239-242 |

So the emitter is **~150 (stellarator) to ~245 (tokamak) call sites** -- §74's figure is
right. But **within one graph there is almost no deduplication** (150 -> 147): a
configuration rarely calls the same leaf twice. **The saving is portfolio-wide**: 1 513 leaf
occurrences summed across all seven graphs collapse to **335 distinct bodies**, because
tokamak and stellarator share costs, power, availability, vacuum, buildings and most of
physics -- exactly as `total_process.py` claims. §74 implied a within-graph effect; it is a
cross-configuration one. The 26 non-`ImplementedFunction` nodes are `RootFind`/`FixedPoint`
drivers that reference other nodes' residuals and own no body, so they add no conversion
burden.

### Honest answer, and the uncertainty

**92 % of the live layer is mechanical, and the real work is a named list of ten.** That is
a much smaller project than §74's prose implied.

**Biggest uncertainty, stated plainly**: this is a static AST census with one level of
helper expansion, **not** a Warp-compatibility check. Warp's typed subset is stricter than
"does it call `jnp.array`", so some Easy-bucket functions will need adjustment once someone
actually emits them, and two Work-bucket items (`tf_stress_*`,
`calculate_pf_coil_power_supplies`) have multi-helper chains not expanded past one level
that could hide further Work-bucket helpers. Treat 237/70/28 as the right order of
magnitude and the right *shape*, not as exact.

## 77. What the TF case clamp actually models, and the way out (2026-09-06)

§75 located §73's anomaly at `tfcoil/base.py:192` and read it as "a cost-minimising
optimiser drives the TF case onto its geometric floor and sits there". **That reading is
half right and it misses the mechanism.** Reading the model properly changes both the
diagnosis and what to do about it.

### The physics is a wedge, and the minimum is a sagitta

```python
dr_tf_plasma_case_minimum = (r_tf_inboard_in + dr_tf_inboard) * (1 - cos(pi / n_tf_coils))
```

A TF coil is one wedge of `n_tf_coils` around the torus. The winding pack is a **rectangle**
inscribed in that wedge, while the plasma-facing case follows the **arc**. `R (1 - cos(pi/n))`
is exactly the arc's **sagitta** -- how far the arc bulges past the chord at half-angle
`pi/n`. So the rule is: *the plasma-side case must be at least as thick as the arc bulges,
or the winding pack clips its corners.* Real geometry, correctly ported.

PROCESS offers two ways to set the thickness (`i_f_dr_tf_plasma_case`), and **both are
clamped by the same sagitta**: a direct input, or `f_dr_tf_plasma_case * dr_tf_inboard`.

### Why it sits exactly on the clamp: a ratchet, not an optimum

Both affected files set **`dr_tf_plasma_case = 0.06`**, and both converge to a *larger*
value -- 0.0714836 and 0.0728348. So the clamp fires because the geometry genuinely demands
more than the input. That much matches §75.

But the input arm **reads the value it writes back**. PROCESS's own source is
`dr_tf_plasma_case = data.tfcoil.dr_tf_plasma_case`, a `DataStructure` field the previous
pass **overwrote**, and `Caller.call_models` re-runs the pipeline up to ten times. So:

- pass 1 reads `0.06`, the clamp fires, writes `0.0714836`;
- pass 2 reads **`0.0714836`**, not `0.06`.

**It is a ratchet.** `x <- max(x, m)` never comes back down, the file's input is consulted
exactly once, and thereafter the "input" is whatever the last pass wrote. The port
reproduces this faithfully as a `FixedPointFunction` (its docstring says so), which is why
§75 found `a == b` **bit-identically** on `low_aspect_ratio_DEMO`: once latched, the two
arguments are the same number by construction, not by coincidence.

**And that fixed point is not unique.** `x = max(x, m)` is satisfied by *every* `x >= m`.
Which value you land on is a property of the iteration history, not of the equations.
`sand.degenerate_fixed_points` screens identity fixed points; this is a `maximum`, so it
passes the screen while being degenerate in the same way. PROCESS knows something is off
here -- its own next comment is *"Warn that the value has been forced to a minimum value at
some point"*.

### The way out, and it generalises

**Fixing the ratchet alone does not remove the kink.** Read the file's `0.06` instead of the
written-back value and you still get `max(0.06, minimum(design))`, still active, still
kinked -- because the geometry really does demand more than 0.06. The ratchet is a separate
defect worth fixing on its own terms (a non-unique, path-dependent fixed point), but it is
not the answer to the knife-edge.

**The answer is to move the clamp out of the model and into the constraint set.** Make
`dr_tf_plasma_case` a design variable and `dr_tf_plasma_case >= dr_tf_plasma_case_minimum`
an inequality constraint. Then:

- the model body becomes **smooth** -- the variable is just a variable, no `maximum`;
- the binding is handled by the solver's active set, with a **multiplier**, which is what
  SQP machinery is for;
- and the derivative is no longer ambiguous at the tie, because there is no tie in the
  model.

**The general principle, and it is the more valuable half**: *a `jnp.maximum(x, floor(design))`
inside a model body is an inequality constraint the solver cannot see.* The clamp still
binds, but it binds invisibly -- the optimiser gets a kinked residual instead of a
constraint with a multiplier, and no `icc` row records that the design is limited by coil
geometry. The switch census (§75) found **48** such sites on `large_tokamak_nof`; how many
are hidden constraints of this kind rather than genuine numerical floors is **unmeasured**,
and is the obvious sweep to run next.

**Cost, stated honestly**: promoting a clamp to a constraint adds an `icc`/`ixc` pair,
changes iteration counts, and changes what PROCESS's regression reference contains -- the
same conversation `icc = 11`'s removal needed (§52, where the answer turned out to be a
single metadata row). It is a change to the problem statement, not a refactor, and belongs
upstream rather than in the port alone.

## 78. De-ratcheting the TF case is free (2026-09-06)

§77 identified the input arm's self-read as a ratchet with a non-unique fixed point, and
predicted that fixing it alone would not remove the kink. **Both halves tested.**

Replaced `dr_tf_plasma_case_from_input`'s self-read with the **file's** stated value (0.06
on both affected configurations) -- turning `x <- max(x, m)` into the ordinary explicit
`max(0.06, m)` -- and re-solved:

| configuration | | status | it | `objf` |
|---|---|---|---:|---|
| `large_tokamak_nof` | as-is | converged | 7 | 1.60000000001 |
| | de-ratcheted | converged | 7 | 1.60000000001 |
| `low_aspect_ratio_DEMO` | as-is | converged | 11 | -0.40631273195 |
| | de-ratcheted | converged | 11 | -0.40631273195 |

**`objf` bit-identical on both** (compared as `float.hex()`), iteration counts identical,
`max_eq` and `min_ie` identical to every digit (2.396e-06 / -1.218e-06 and
7.327e-15 / 9.264e-12 respectively).

**So the ratchet is numerically inert**, which §75's measurement already implied without
anyone drawing the conclusion: the entering value was equal to the geometric minimum
bit-for-bit, so it had *tracked* the minimum rather than latching above it at some earlier,
larger iterate. The path-dependence is real and the fixed point is genuinely non-unique,
but on these two configurations the iteration lands on the same point either way.

**Which makes the fix free.** Removing the self-read costs nothing numerically, deletes a
degenerate fixed point (`x = max(x, m)` holds for every `x >= m`), and turns a
`FixedPointFunction` into an ordinary explicit function -- one fewer node in the driven set.
It is a strictly-simplifying change with a measured zero-difference receipt.

**And it does not remove the kink**, as §77 predicted: `max(0.06, m)` is still active,
because the geometry still demands more than 0.06. The knife-edge needs the constraint
promotion, not the de-ratchet. Two separate changes, and this is the cheap one.

**Not done here.** It is a change to a ported model body whose faithfulness to PROCESS is
the port's whole contract -- and PROCESS *does* ratchet, by virtue of re-running the
pipeline over a mutated `DataStructure`. Making the port stop ratcheting is a deliberate
divergence from the source, defensible on the evidence above but not something to slip in
as a refactor. It belongs in the same conversation as §77's constraint promotion.

## 79. A staged plan for the Warp path, with costs (2026-09-06)

§76 sized the conversion; this sizes the *project*. Two things were measured for it that
did not exist in the record, both PROCESS-free across all seven configurations:

**MDF nests 3-6 separately driven blocks per configuration** (35 across the seven), while
**SAND collapses to exactly one `Drive`, on all seven without exception.** That is not a
hopeful reading of §74's argument -- it is what the graphs do. It is also by construction:
`sand.sand_shape`'s docstring says *"The **one** `Drive`'s size"* and it takes
`next(step for step in schedule.steps if isinstance(step, Drive))`. The single `Drive`'s
interior is 95-184 nodes of pure dataflow, because `Residualise`+`Combine` turn every
`FixedPoint` into a residual condition of the outer problem before scheduling. **SAND is
the target; MDF is not.**

And: **`helias_5b`'s SAND `Drive` (95 nodes, the smallest across all seven) already contains
`impurity_radiation_totals`** -- §76's "hardest single item", a `vmap` over `jnp.interp`.
**No configuration, however small, lets the first milestone dodge a Work-bucket function.**

### The stages

| stage | what | effort | confidence |
|---|---|---:|---|
| **0** | micro-spike: ~10-15 Trivial `@wp.func`s, hand-stitched kernel, CPU-only, one input point | 2-3 d | high |
| **1** | **the decisive experiment**: emit `helias_5b`'s **SAND** `Drive` (95 nodes) as one straight-line kernel; bitwise + timing against the JAX `Drive` | **5-8 d** | **low** |
| **2** | the `@wp.func` layer: an AST transpiler for the 237 Trivial + 70 Easy, plus the **10 load-bearing** Work functions hand-ported, plus a drift test | 16-23 d | medium |
| **3** | the kernel emitter: `Blocking.ordered_graph()` and the `In`/`Out` ports already exist; a `VarPath -> identifier` mapper and declared port shapes/dtypes do not | 4-8 d | medium |
| **4** | derivatives | 4-9 d | medium |
| **5** | validation: a `WarpContract` tier reusing the Richardson-bar logic | 2-4 d | medium-high |
| | **total, one engineer, to all seven SAND kernels emitting, differentiating and validated** | **~33-55 d (7-11 weeks)** | |

**Stage 1 is the gate.** None of stages 2-5 (~26-44 days) should be spent before its number
exists. It is deliberately structured so the expensive part is *behind* the cheap decisive
one, and it must not be routed around if it fails -- an interp gradient coming back wrong,
or Warp's typed subset choking on the ~90 bodies in that block, is the answer.

**Note the brief's own naive reading was wrong and the plan corrects it**: `helias_5b` **MDF**
is smaller by variable count (3 unknowns, 5 conditions) but is exactly the nested-solve
shape §74 argues against. `helias_5b` **SAND** is larger by node count and is the actual
straight-line block the thesis rests on.

### Three things read from Warp's documentation rather than assumed

- **`wp.grad(func)` computes all partials of a `@wp.func` in one launch**, which the docs
  frame as the efficient alternative to one `wp.Tape` backward pass per Jacobian row. For
  SAND that is the difference between 1 launch and 11-34 (the condition counts). **This is
  the mechanism to build toward**, and it needs the residual expressed as `@wp.func` calls,
  which stages 2-3 already produce.
- **Dynamic loops are not replayed in the backward pass** -- a documented gotcha, and it
  would have bitten the interpolation functions. It does not, and §76's own CoolProp finding
  is why: the tables are evaluated at assembly and handed in as **static** fields, so the
  loop bound is a Python int at trace time and the loop can be written in Warp's *static*
  form, which **is** correctly differentiated. **That is an acceptance criterion for
  whoever ports the impurity function, not a hope.**
- **Warp ships `gradcheck()`/`jacobian_fd()`** -- the same AD-versus-FD idea the Tier-1
  harness implements. Reuse it and feed the result through the existing Richardson-bar
  reporting rather than building a second comparator.

**Do not try to get a Hessian from Warp** -- no second-order support was found documented.
That matches §74's "kernel = evaluation, host = optimisation": Warp supplies value and
Jacobian, VMCON keeps its quasi-Newton approximation on the host.

### What it pays, and when

**Two separable wins, and only one needs a GPU.**

1. **Compile-time, at N=1, CPU only.** §63's bound: ~4.4x faster and ~3.6x smaller than XLA
   at `-O0`, and §74's reframing means emitting ~150-245 call sites rather than 28k ops.
   Free adjoints replace what §63 called the largest unpriced cost. **This pays immediately
   and is what stages 0-5 are sized for.**
2. **GPU batching, gated at >=256 points, evaluation-only.** §71's crossover, 3.4x by 4096,
   *despite* §70's 1/16.4 FP64 penalty. But §71 is explicit that it measured *evaluation*,
   not a batched solve -- realising it needs either a shared/padded outer iteration count
   across the batch (**an untested assumption about convergence uniformity across a sweep**)
   or `jaxipm`'s iteration-level batching (§67: unreachable here). **A single solve gets
   nothing: it is 2.4-12x slower on this GPU, measured.** The payoff exists only for
   `scan.py`-shaped sweeps of hundreds to thousands of points, and any budget against it
   must state its hardware assumption.

### Confidence

Measured: the MDF-versus-SAND driver counts, and that the smallest SAND block contains a
Work function. Read from Warp's docs, not run: everything in the previous subsection.
**Guesses, flagged**: stage 1's day count (entirely friction nobody has hit), the distinct-
wrapper count inside that 95-node block (extrapolated from the 150->147 whole-graph pattern,
**counting it is the first 30 minutes of stage 1**), and whether padded batching realises
§71's crossover for real scans.

## 80. The Warp spike, run (2026-09-06)

§79 sized a 5-8 day stage 1. This is the *stage 0* micro-spike, done in an afternoon:
`warp-lang 1.17.0` installed into `process_port_gpu` (157 MB; the CPU audit env untouched),
two **real** port functions transpiled -- `pure_formulas.rether` and
`_fast_alpha_fraction_ward`, the latter including the smoothed Ward kink -- composed in one
`@wp.kernel`, and run against the identical JAX arithmetic.

### Two mechanical findings, and the second is a real addition to the cost

**1. It is a rewrite, not a wrap.** Warp compiles the **source** of the decorated function,
parsing its AST; `jnp.sqrt` must become `wp.sqrt`. You cannot decorate an existing jnp
function and have it work. §74's "write each leaf once as a `@wp.func`" is right, but the
word "wrap" would be wrong.

**2. Warp is strictly typed and does not promote.** `1.0` is a **float32** literal, so
`1.0 + alphan` against a `wp.float64` is a compile error:

```
RuntimeError: Input types must be the same, got ['float32', 'float64']
```

**Every numeric constant in a float64 kernel must be written `wp.float64(1.0)`.** PROCESS
is float64 throughout, so this is pervasive -- it applies to every literal in all 335
functions. §76's census classified bodies by their *jnp constructs* and did not consider
literal typing at all, so **this is work the census did not price.** It is mechanical and a
transpiler can do it, but it must be in the transpiler's spec.

### Performance: the scalar-native-CUDA hypothesis holds, on the GPU only

Per point, microseconds, same arithmetic, same inputs:

| batch | JAX CPU | JAX GPU | Warp CPU | **Warp GPU** |
|---:|---:|---:|---:|---:|
| 1 | **23.55** | 205.85 | 31.27 | 38.38 |
| 256 | **0.0961** | 0.5612 | 0.1415 | 0.1437 |
| 4 096 | 0.0358 | 0.0433 | 0.0338 | **0.0093** |
| 65 536 | 0.0109 | 0.0080 | 0.0276 | **0.0041** |

- **Warp GPU beats JAX GPU 4.7x at 4 096 and 1.95x at 65 536**, and beats the best JAX
  number of any backend by **3.9x** and **2.7x** respectively. The reasoning behind the
  hypothesis -- one kernel with everything in registers, against XLA's many kernels with
  global-memory round-trips between them -- is supported.
- **And it holds despite this card's 1/16.4 FP64 penalty (§70)**, which makes it more
  impressive rather than less.
- **But Warp CPU is *slower* than JAX CPU** -- 0.0276 against 0.0109 at 65 536, a 2.5x
  loss. XLA's CPU vectoriser beats Warp's CPU codegen on this arithmetic. **So the CPU
  runtime win that §63's compile-time numbers might have suggested does not appear.**
- At batch 1 everything is overhead and JAX CPU wins.

### Agreement

**max relative difference 3.27e-13**, roughly half the 4 096 points bit-identical, same on
both devices. Well inside any tolerance that matters (regression is 5 %, the harness's bars
~1e-6). **Caveat, and it is mine**: the transpile rewrote `te**1.5` as `te * sqrt(te)` and
`(1+alphan)**2` as `(1+alphan)*(1+alphan)` -- mathematically equal, differently rounded --
so part of that 3e-13 is transcription rather than Warp. A real transpiler should preserve
the expression form and this check should be redone against one.

### Compile time, and why it does not extrapolate

Warp took **2.54 s** to build the CPU module and **0.66 s** the CUDA one, against JAX's
0.05-0.14 s. Warp compiles a whole C++/CUDA module through clang/nvcc; JAX traced two
functions. **For two functions JAX wins and it means nothing** -- the interesting number is
the 95-node block, where §63's bound predicts the ordering reverses. Not measured.

### Correction to §74 and §79: "free adjoints" oversold it

Both sections list source-to-source adjoint generation as a Warp advantage. **Against a
hand-written C codegen path (§63) that is true and it was the largest unpriced cost there.
Against JAX it is not an advantage at all** -- JAX has had adjoints the whole time, and
§59's work made reverse mode work on this graph. The honest list of what Warp offers *over
the status quo* is: **scalar-native code generation with no shape plumbing, and native CUDA
kernels that keep intermediates in registers.** The GPU numbers above are that advantage.
Adjoints are table stakes, not a differentiator.

### What this changes about the plan

**Stage 0 is done and it passed.** §79's stage 1 -- the 95-node `helias_5b` SAND `Drive` --
remains the gate, and is now better specified: the transpiler must handle literal typing,
and the payoff to look for is **GPU throughput at batch**, not CPU runtime and not compile
time. Nothing here contradicts §79's 5-8 day estimate for that stage; the friction found
was in typing rather than in the arithmetic, which is the cheaper kind.

## 81. A prototype transpiler, and how much it covers (2026-09-06)

§80 established that a Warp port is a **rewrite, not a wrap** -- Warp parses the source of
the decorated function, so `jnp.sqrt` must become `wp.sqrt` and every literal must be
`wp.float64(...)`. The obvious follow-up: how much of that can be generated?

**A transpiler translates source to source.** It does not produce machine code -- it emits
*other Python*, which Warp then compiles. Parse each model function with `ast`, rewrite the
nodes that differ, unparse as a `@wp.func`. Prototype at
`_audit/warp_transpile_prototype.py`, ~100 lines, three rewrites:

| | |
|---|---|
| `jnp.X(...)` -> `wp.X(...)` | a direct table plus renames (`maximum`->`max`, `minimum`->`min`, ...) |
| numeric literal -> `wp.float64(n)` | §80's strict-typing finding, applied to every `ast.Constant` |
| bare signature -> annotated | every parameter and the return as `wp.float64` |

**Design rule, and it is the important one: it refuses rather than guesses.** An
unrecognised construct raises `Unsupported` and that function goes on the hand-port list. A
transpiler that silently mistranslates one formula is far worse than one that covers less
and says so.

### Coverage, swept over the whole model layer

**405 of 501 functions (80.8 %) transpile cleanly**, with no hand-tuning. The 96 refusals
are highly structured:

| reason | count | what it is |
|---|---:|---|
| `jnp.asarray` | 25 | mostly type coercion -- often a no-op once inside a typed kernel |
| `zeros` / `stack` / `sum` / `zeros_like` / `arange` / `array` / `take` / `repeat` | **52** | **genuinely array-valued** -- the real hand-port list |
| `clip` / `max` / `isinf` / `round` / `degrees` | 12 | **table gaps**, trivially addable (`clip`->`clamp`, and so on) |
| other | 7 | assorted |

So the picture is: **~81 % free today, ~87 % after an afternoon's more table entries, and an
irreducible core of ~50 genuinely array-shaped functions** that need hand-porting -- which
is the same population §76's census called the "Work" bucket, arrived at independently.

(This sweep covers all 501 functions discovered under `models/**`, not just the **335**
§76 measured as live on the seven configurations. The live subset should do at least as
well, since dead code skews odd; not separately measured.)

### The property that makes this unusually favourable

**Every generated function has a reference implementation sitting next to it.** The JAX
original is right there, so the generator can be validated per function, automatically, on
random inputs -- no golden files, no manual review of 400 translations. That is exactly the
shape of the existing Tier-1 harness (`Tier1Contract`, AD against Richardson FD), and §79's
stage 5 can reuse it rather than inventing a second comparator.

**What it does not solve.** The 52 array-valued functions still need a human, and §80's
transcription caveat stands: the prototype leaves `**` alone rather than rewriting `x ** 1.5`
into `x * sqrt(x)`, which is right -- the manual spike's rewrite of exactly that was part of
its 3.27e-13 disagreement. Preserving expression form is a feature, not an omission.

## 82. The stellarator knife-edge, measured -- and §72 was half wrong (2026-09-06)

§72 said `wp_width_r_min` is non-smooth **two ways**: `intersect()`'s piecewise-linear
crossing, and `calculate.py:797`'s `jnp.maximum` clamp. Instrumented both with runtime
`jax.debug.callback` hooks on `stellarator_helias`, SLSQP on both arms.

### The clamp is not live at all

`dx_tf_turn_general**2 = 0.003136` constant against `wp_width_r_min` in `[0.62, 0.74]` --
a **~200x margin**, relative gap pinned at 0.995-0.996, winner never flips on either arm.
**§72's second mechanism does not exist on this configuration.** The whole knife-edge here
is the interpolation kink.

### The Ward signature does *not* transfer, and the zigzag is real anyway

| | Ward (§47) | this site, SAND |
|---|---:|---:|
| crossing rate | 46 % of steps | **0.5 %** (21 of 4 011 calls) |
| closest approach | 5.5e-08 | **2.09 % of an interval width** (5.78e-04) |

So the headline numbers that made Ward actionable are absent -- **and the trajectory shape
still matches §47 exactly**: a decaying period-2 zigzag straddling the boundary between
breakpoint intervals **20 and 21** over calls 18-42 (`21, 20, 21, 20, ...`), which then
**locks onto interval 20 permanently**. The remaining ~3 970 calls -- 99 % of the run --
never cross again, converging toward a near-fixed `x = 0.7078943051`.

**That is why the aggregate rate is 0.5 %**: the crossing happens in a short early burst and
then stops. An average over the whole run hides it. Ward's 46 % was a *sustained* crossing;
this is a transient one that does its damage and then latches.

### And the clean confirmation: exposure, not the kink

**MDF crosses breakpoints on 59.2 % of calls -- over 100x SAND's rate -- and converges in
27 iterations to `objf = 1.21848284` exactly, `max|eq| = 4.9e-10`.**

MDF crosses this kink constantly and does not care, because it converges the root find
*inside* every evaluation and the outer solver only ever sees the converged value. SAND
crosses it a hundred times less often and caps at 500, because it hands the outer SQP a
residual that is non-differentiable there. **§72's conclusion is confirmed, by the arm that
crosses more and suffers less.**

### A fix that works, and a control that was not run

Raising the interpolation resolution (`_N_WINDING_PACK_SAMPLES`,
`models/stellarator/coils/calculate.py:530`, runtime patch only):

| samples | status | iterations | `objf` | vs VMCON `1.21848284` | `max\|eq\|` |
|---:|---|---:|---|---:|---:|
| 200 (baseline) | cap(500) | 501 | 1.218434089 | 4.9e-05 | 7.7e-05 |
| 500 | **converged** | 92 | 1.218368727 | 1.06e-03 | 7.3e-12 |
| 2 000 | **converged** | 105 | 1.218311345 | 1.71e-03 | 3.2e-12 |

**The 500-cap disappears outright**, and `max|eq|` improves seven orders to machine
precision. That is strong causal evidence that the piecewise-linear resolution *is* the
mechanism.

**But the comparison column is apples to oranges, and this matters.** Changing the sample
count changes the **discretisation**, so at N = 500 and N = 2 000 the solver is converging a
*different model*. Drifting 1.06e-03 from VMCON's N = 200 answer is not error -- it is a
different problem's answer, and a finer grid is presumably the *more* accurate one. **The
missing control is VMCON at the same N**, which would separate "SLSQP now converges" from
"the model moved". **Not run, and it is the first thing to do before anyone reads the table
above as a fix.**

Also untested: a monotone-cubic interpolant (C1, so no kinks at all -- but a divergence from
PROCESS's own piecewise-linear scheme), and whether the finer grid changes the `c62`
interaction §49 found.

### §81 continued: the registry, and coverage after one afternoon's table entries

The right architecture is **transpile by default, with a registry of hand-written
overrides** -- the same shape as the port's own unit registry, and the escape hatch is the
only place a human writes Warp:

```python
REGISTRY: dict[str, str] = {
    "safe_sqrt": '''@wp.func
def safe_sqrt(x: wp.float64) -> wp.float64:
    return wp.sqrt(wp.max(x, wp.float64(0.0)))''',
}
```

Anything the transpiler refuses either has a registry entry or goes on the to-do list. It
never guesses.

**Coverage after extending the table** with the §81 gaps -- `clip`->`clamp`, `isinf`,
`round`, `trunc`, `degrees` as a multiply, and `asarray`/`array` treated as **identity**
(a coercion a statically-typed kernel does not need):

| | before | after |
|---|---:|---:|
| transpiled cleanly | 405 / 501 (80.8 %) | **420 / 501 (83.8 %)** |
| refused | 96 | 81 |

And the residue is now almost pure signal: **74 of the 81 refusals are array constructors
or reductions** -- `stack` 15, `zeros` 14, `zeros_like` 11, `sum` 11, `arange` 6, `max` 4,
`take`/`repeat`/`interp`/`broadcast_to`/`atleast_1d` the rest. That is the genuinely
array-valued population, and it is what the registry is for.

**End to end, verified.** The *generated* source for `_fast_alpha_fraction_ward` compiles in
Warp unmodified and agrees with the JAX original:

```
warp : [0.19282124   0.00026853  ]
jax  : [0.19282124034257048, 0.0002685253535502736]
```

So the loop closes: JAX function -> transpiler -> Warp source -> compiled kernel ->
matching numbers, with no hand editing in between.

**One thing that did not work, recorded so nobody repeats it.** Scoping the emit to
`stellarator_helias`'s block specifically needs resolving each graph node to its leaf
function, and a quick AST walk over `ImplementedFunction.fn` resolved only **4 of 154**.
`g.nodes` is `NodePath`s and `g.definitions` maps them to `ImplementedFunction`s whose `.fn`
is the wrapper's callable -- but the wrappers are not uniformly the thin
`return calculate_x(...)` shape that walk assumes. **§76's census already solved this**
(AST-parsing the wrapper *class*'s `__call__` plus one level of helper expansion, 335
functions across seven configs); reuse that rather than the shortcut above. The coverage
numbers here are therefore over the **whole** model layer, not the stellarator block --
which is the more useful denominator anyway, since the registry is written once for all
configurations.

## 84. The whole model layer transpiled, compiled, and validated (2026-09-06)

§81 covered the transpiler; this ran it over everything and put the output through Warp.

### Two more "refuse rather than guess" rules, and what they cost

Emitting for real surfaced two ways a body can fail to be self-contained, both now refused:

- **Unresolved globals.** A transpiled body may reference module-level names. Each must
  resolve to a scalar (emitted as a `wp.constant`) or to another leaf in the same module;
  anything else -- an enum, a table, a class -- and the function is not self-contained.
- **Default arguments.** Warp has no equivalent, and silently dropping a default is exactly
  the guess this design forbids.

**They cost a lot of coverage, and that is the honest number**: 420 of 501 became **326 of
527** once self-containedness was actually demanded. The 81-refusal figure in §81 was
measuring "does the arithmetic translate", not "is this function emittable on its own".

### It compiles

**323 `@wp.func` emitted as one Warp module, 65 `wp.constant`s, 717 lines -- and the whole
module builds in 2.44 s.** Compiling is a real gate that transpiling does not clear on its
own: Warp's type checker only runs at module build.

### And it agrees -- the self-validating property, demonstrated

§81 claimed the generator can validate itself, because every generated function has its JAX
original next to it. Done: random positive inputs through both sides, compared where the
reference is finite.

| | |
|---|---:|
| exercised end to end | **115** |
| **agree to 1e-9** | **115** |
| disagree | **0** |
| **worst relative difference** | **0.000e+00 -- bit-identical** |
| not exercised (harness signature handling, arity) | 170 |
| non-finite reference (random inputs are unphysical) | 8 |

**Every function the harness could drive agreed bit-for-bit.** The 170 skips are my probe's
keyword-only-signature handling, not transpiler failures; a real validation stage would
drive them from the graph's own port declarations instead of guessing an arity.

### Answering "inlined, or called?" with the generated source

Warp caches what it emits (`~/.cache/warp/<version>/`), so this is readable rather than
assumed. A `@wp.func` becomes:

```c
static CUDA_CALLABLE wp::float64 _fast_alpha_fraction_ward_0(
    wp::float64 var_density_ratio_sq, wp::float64 var_temp_sum_20) { ... }
```

and the kernel body contains `var_3 = _fast_alpha_fraction_ward_0(var_4, var_5);` -- **a
real call to a `static` device function, not textual inlining.** Whether it is then inlined
is `nvcc`/`clang`'s decision, which for a small `static` function it almost always takes.

**That softens §74's register-pressure worry**: each `@wp.func` has its own scope rather
than 28k values being live at once, and the compiler chooses where to inline.

And the adjoint sits right beside it in the same file --
`adj__fast_alpha_fraction_ward_0(...)`, generated from the same source, with the kernel
emitted in both forward and backward forms. §80's correction stands (this is not an
advantage over JAX), but it is visible here rather than taken on trust.

### Less NVIDIA-centred alternatives, for the record

- **Taichi** -- the closest substitute: Python-embedded kernels, backends for CPU, CUDA,
  Vulkan and Metal, **and built-in autodiff**. The portable analogue if NVIDIA-only is the
  objection.
- **JAX Pallas** -- a kernel DSL *inside* JAX, so AD and the existing harness carry over
  unchanged; targets Triton on GPU. No second language, narrower control.
- **Enzyme + plain C** -- LLVM-level AD over emitted C, which is exactly §63's path plus
  the derivatives it lacked. The most portable (any LLVM target) and the heaviest to set up.
- **Numba** (CPU + CUDA) and **Triton** -- both viable as kernel targets, **neither has
  autodiff**, which for this port is disqualifying on its own.

**None of these has been tried.** Named so the choice is visible rather than defaulted into.

## 85. Correction to §84: "323 compiled" was measuring the wrong thing (2026-09-06)

§84 reported **"323 `@wp.func` emitted as one Warp module ... the whole module builds in
2.44 s"** and treated that as a gate cleared. **It was not.** Two independent problems, one
found by an agent working the array bucket and one by checking its report.

### `wp.load_module` on a file with no kernels compiles almost nothing

Warp codegens a `@wp.func` when something **reaches** it -- a `@wp.kernel` that calls it.
`warp_all.py` contained 383 `@wp.func` and **zero kernels**, so loading it exercised the
Python decorators and little else. The 2.44 s was module *loading*, not code generation, and
§84 should not have called it "it compiles".

**Measured properly** -- generate a one-line kernel per function and load *that*:

| | codegen |
|---|---:|
| single-return, sampled 17 | **13 ok** |
| multi-return, sampled 14 | **0 ok** |

### And 151 of 385 emitted functions were mis-annotated, invisibly

`transpile()` stamped `fdef.returns = wp.float64` unconditionally. **151 of 385 emitted
functions return a tuple**, so every one carried a return annotation contradicting its body:

```
WarpCodegenError: The function `_avail_from_blanket_lifetime` has its return type
annotated as `float64` but the code returns 5 values.
```

**The validator could not see it.** `validate.py` calls `float(jax_fn(*vals))` on the
**JAX reference first**, which raises on a tuple -- so every multi-return function landed in
"skipped" *before* Warp was ever asked to compile it. §84's headline "115 validated
bit-identical" is therefore true **only of single-return functions**, and it was never a
sample of the module.

The annotation is fixed (do not annotate the return when the body returns a tuple; Warp
infers multi-return itself), which also lifted 60 functions out of refusal -- 326 -> 386
transpiled, 201 -> 117 refused. **But multi-return functions still do not codegen in the
probe above**, and whether that is the emitted code or the probe's own kernel (which needs
tuple unpacking, `a, b = fn(...)`, and does not do it) is **not yet determined**. Stated as
unknown rather than resolved in either direction.

### What §84's numbers should have said

- **383 functions emit** as syntactically valid Warp source. True, and worth having.
- **13 of 17 sampled single-return functions codegen.** A real but much narrower claim.
- **151 multi-return functions are unverified**, and were unverified when §84 called them
  compiled.
- **115 validated bit-identical stands** -- that check did run end to end -- but it covers
  only single-return functions and is not representative of the module.

**The lesson is the validator's, not the transpiler's.** A harness that silently *skips*
what it cannot drive will report success on the subset it can reach and say nothing about
the rest. §84's skip count (170) was sitting there the whole time and I read it as harness
friction rather than as the thing hiding the failure.

## 86. Is the resolution fix causal or lucky? Both, in different ways (2026-09-06)

§82 raised the `intersect` sample count from 200 and the 500-iteration cap disappeared;
the doubt was whether that is causal or a lucky landing on a still-chaotic arm. Tested with
the measurement that settled Ward: perturb the cold start by whole ulps and look at the
spread.

**A harness bug had to be fixed first, and it would have inverted the answer.**
`native.NativeState` keeps **two independent stores** for a boundary value -- a flat
`(area, name)` dict and each `_Area`'s own `name` dict -- populated at construction and
never synced, while every real read goes through the `_Area` copy. A perturbation written
to the flat dict alone **never reaches the solve**, and the first attempt duly returned
bit-identical results across 0, +-1 and +-2 ulps -- which reads exactly like "not chaotic".
Caught, both stores written, and the fix verified to reach the solve before any number
below was trusted. **This is a live port defect and belongs in `next_steps.md`.**

### The perturbation test: the fix is real

Ten draws (+-1..+-5 ulp on `.physics.rmajor`), SLSQP, cap 500:

| N | outcome |
|---:|---|
| **200** | **8/10 converge** at 235, 267, 304, 254, 281, 308, 264, 429 iterations -- a wide, unpredictable spread -- and **2/10 hit the cap** |
| **500** | **10/10 converge**, 89-96 iterations (tight) |
| **2 000** | **10/10 converge**, 98-134 iterations |

N = 200 is a genuine Ward-style chaotic signature. N = 500 and 2 000 are robust under every
perturbation tried. **So it is not two lucky points -- the mechanism is causal.**

### But the sweep is not monotone, so the kink is narrowed rather than removed

N in {200 ... 3000}, single seed: **200 caps, 250-1000 all converge, 1500 caps** (badly,
`max|eq| = 1.6e-4`), **2000 and 3000 converge**. A follow-up at N = 1500 shows its natural
seed caps while +-1/+-2 ulp draws converge in 112-167 -- i.e. the same "unlucky exact seed"
event N = 200's baseline was. **Higher resolution shrinks the kinks; it does not eliminate
them, and specific points in resolution space still land on a crossing.**

### And the control §82 asked for does not vindicate the fix

SLSQP against **VMCON at the same N** -- the honest comparison, since changing N changes the
model:

| N | SLSQP `objf` | VMCON `objf` | relative |
|---:|---|---|---:|
| 200 | 1.218434089 (capped) | 1.218482842 (24 it) | 4.00e-05 |
| 500 | 1.218368727 (92 it) | 1.218447998 (26 it) | 6.51e-05 |
| 2 000 | 1.218311345 (105 it) | 1.218441862 (46 it) | 1.07e-04 |

**They agree to about four significant figures, not nine -- and the disagreement grows with
resolution.** Both satisfy their own residual criteria and land on *different points*. So
"SLSQP converges at N = 500" is true and is not the same as "the arm is solved": it settles
somewhere KKT-satisfying that is not where the production driver settles, at every
resolution tested.

**Verdict**: causal, incomplete, and not a fix. Raising the resolution turns a chaotic arm
into a reliably-convergent one, which is real; it neither removes the kink nor brings SLSQP
to VMCON's answer. **A C1 interpolant is the better-motivated intervention** -- it removes
derivative jumps outright rather than shrinking them -- and is being tested separately.

**Not reached**: crossing-burst counts per N, and the global `maximum`/`minimum` switch
sweep pointed at the stellarator rather than the tokamaks.

## 87. A C1 interpolant: smaller intervention, same deeper gap (2026-09-06)

§86 left the resolution increase as "causal, incomplete, and not a fix". The
better-motivated alternative -- **raising resolution shrinks the kinks, a C1 interpolant
removes them** -- was tested with a JAX-traceable Fritsch-Carlson monotone cubic (validated
against `scipy.interpolate.PchipInterpolator` to <1e-9 and against finite-difference
gradients), runtime-patched into `intersect_residual` at the **original N = 200**.

**Both drivers converge, no cap.**

| driver | iterations | `objf` | `max\|eq\|` | `min ie` |
|---|---:|---|---:|---:|
| VMCON | 43 | 1.21844143 | 1.10e-06 | -4.48e-10 |
| SLSQP | 99 | 1.21831738 | 6.44e-10 | -8.92e-12 |

**And it is much the smaller intervention**, which was the point:

| | crossing point moves | `objf` shift |
|---|---:|---:|
| C1 at N = 200 | 0.7169721730 -> 0.7169141045, **8.1e-05 relative** | -- |
| resolution N = 500 | -- | 1.06e-03 |
| resolution N = 2 000 | -- | 1.71e-03 |

An order of magnitude less perturbation to the model for the same removal of the cap.

**But it does not close the deeper gap.** SLSQP against VMCON **at the same C1 interpolant**
is `|1.21844143 - 1.21831738| / 1.21844143 = 1.02e-04` -- **not smaller** than
piecewise-linear's 4.0e-05, and comparable to N = 500's 6.5e-05. So C1 removes the kink and
lets SLSQP converge at the original resolution, and the two drivers still land on different
points. **§86's "not a fix" verdict survives**: whatever separates SLSQP from VMCON here is
not the interpolation kink.

**Two honesty notes, both the agent's own and both right.** The unpatched control run
converged in 140 iterations rather than capping -- which does not match §82's documented cap
for the same nominal seed, and is exactly what §86's chaos finding predicts: **a single
unperturbed draw is not evidence at N = 200.** By the same token, **the single converged C1
run could itself be a lucky draw.** The ulp-perturbation sweep §86 used has not been run on
the C1 arm, and until it has, this is one draw. That is the bounded next step.

### The transpiler tail: two of my own hypotheses refuted

- **`MISC-JNP` was mislabeled.** The recorded reason is only the *first* `Unsupported` the
  AST walk hits, so 8 of 9 were not "small table additions" at all -- underneath sit
  `lax.while_loop`/`scan`, or `jax.grad`/`jax.vmap` called **inside** the body
  (`calculate_n_cycle`, `solve_duct_diameter`, `solve_duct_geometry`,
  `_solve_vacuum_pumping_old`). Only `_divwade_hldiv_base` was the expected shape; it now
  transpiles, compiles and is **bit-identical** to JAX. **A refusal reason is a first
  symptom, not a diagnosis** -- worth remembering before reading any of these bucket counts
  as work estimates.
- **`DEFAULTS`: my hypothesis was wrong for all six.** I suggested the emitter supplies every
  argument explicitly so defaults never need to exist. It does not hold: the defaults are a
  `PFCoilTopology` **dataclass** (graph-assembly data, not a port), an Optional-array
  sentinel that a downstream branch tests for, a static precondition in a function that
  `raise`s and returns a 2-tuple, and a config reader that reads JSON at assembly time and
  is documented as "not a node, and not traced". Two are genuinely droppable (`steps=8`,
  `max_iter`/`tol`) and both of those functions are refused for unrelated reasons anyway.
  **Leave `DEFAULTS` refused.**

### And a fix of mine that is still unverified

§85 fixed the multi-return annotation -- do not stamp `-> wp.float64` on a function whose
body returns a tuple. **It has still never been exercised.** The tuple-aware validator
(written for exactly this) reports **138 single-return agreeing at worst 0.000e+00** and
**148 multi-return failing codegen**, with the error still reading *"annotated as `float64`
but the code returns 2 values"* -- i.e. the emitted module was regenerated through a
`transpile.py` that does not carry the fix, because three agents were editing that file
concurrently and the emitter's import path did not pick up the isolated copy. **The fix is
written in `_audit/warp_transpile_prototype.py` and remains untested.** Stated plainly
rather than counted as done -- which is the whole of §85's lesson, and I nearly repeated it.

## 88. The multi-return fix, verified (2026-09-06)

§85 wrote the fix -- do not stamp `-> wp.float64` on a function whose body returns a tuple
-- and §87 had to record it as **still unverified**, because the emitted module kept coming
back mis-annotated. **The cause was mine and mundane**: rebuilding in an "isolated"
directory, my path rewrite left `sys.path.insert(0, S)` -- a *variable* still pointing at
the shared directory -- so the emitter kept importing the unfixed transpiler while I read
the output as evidence about the fixed one. Three agents editing one file concurrently made
that easy to miss and hard to see.

Rebuilt from the **tracked** prototype in a directory nothing else touches:

```
emitted 326 @wp.func + 65 constants
multi-return: 117, mis-annotated: 0
```

### And they validate

Tuple-aware validation (the harness §85 called for, driving `r0, r1 = fn(...)` rather than
`float(fn(...))`):

| | count |
|---|---:|
| agree (single-return) | 124 |
| **agree (multi-return)** | **62** |
| codegen failed | 59 |
| skipped: arity / JAX raised / non-finite / array-valued | 47 / 17 / 16 / 3 |
| **worst agreeing relative difference** | **0.000e+00** |

**186 functions validated bit-identical against their JAX originals, 62 of them
multi-return** -- against §84's headline of 115, which covered only single-return functions
and was measured by a harness that could not see the other half.

**This closes §85's correction.** The chain was: an unconditional return annotation, a
validator that skipped what it could not drive, a "compiled" claim that was really module
loading, and finally a path rewrite that pointed at the wrong copy. Each hid the next.

### A loose end worth naming

The solo rebuild emits **326** where the shared copy emits **378**, because three agents
improved the *scratchpad* `transpile.py` -- `jnp.pi`/`math.*`/builtin `abs`, literal-indexed
tuples, resolving globals against the defining module rather than the discovering one, an
`enum.Enum` exclusion -- and **none of that is in the tracked
`_audit/warp_transpile_prototype.py`.** The tracked file has the multi-return fix; the
scratchpad copy has the coverage work. **Neither is complete, and merging them is the next
concrete task**, along with the `zeros_like`/`ones_like` scalar rule and the two-store
`NativeState` defect (§86) that is unrelated but equally unlanded.

## 89. C1 verified robust -- and then not landed, because of what it costs (2026-09-06)

§87 left the C1 interpolant on one draw. The ulp sweep §86 called for has now been run --
same protocol, both drivers, N = 200, ten draws (`ulps` in 0, +-1..+-4, +5):

| | piecewise-linear (§86) | **C1** |
|---|---|---|
| SLSQP | **8/10** converged, 235-429 iterations, **2 hard caps** | **10/10**, **83-101 iterations** |
| VMCON | -- | **10/10, exactly 43 every draw** |

**It clears the bar decisively.** Not a lucky draw: a tight spread where the baseline had a
2:1 range and two caps, and VMCON is provably untouched -- 43 iterations regardless of
perturbation, confirming the production driver never sees this kink.

The SLSQP/VMCON gap is also **stable rather than scattered**: 9.47e-05 to 1.27e-04 across
all ten draws, centred on §87's single-draw 1.02e-04. So that disagreement is a real,
reproducible property of the arm and not sampling noise -- and C1 does not touch it.

### And then it was applied, measured, and reverted

Applying it breaks `test_st_coil_matches_process_end_to_end` -- a tier-3 end-to-end
comparison against PROCESS at **`rtol = 1e-9`** -- by roughly **5e-04 relative, across the
whole TF coil chain**:

| quantity | port (C1) | PROCESS | relative |
|---|---|---|---:|
| `dr_tf_wp_with_insulation` | 0.52817703 | 0.528366452857 | 3.6e-04 |
| `a_tf_inboard_total` | 25.83482039 | 25.847257117856 | 4.8e-04 |
| `m_tf_coils_total` | 5 694 513.48 | 5 697 113.26 | 4.6e-04 |
| `m_tf_coil_superconductor` | 7 327.51 | 7 332.77 | 7.2e-04 |

**This is not a solver-only change.** §87 measured the crossing point moving 8.1e-05 and I
reported that as "a genuinely small perturbation"; it propagates to ~5e-04 on TF coil
masses, areas and current densities. Well inside the 5 % regression tolerance, and five
orders outside what the port's own faithfulness test demands.

**So the trade, stated plainly: ~5e-04 on TF coil physics outputs, to fix an arm that only
`SlsqpDriver` fails.** VMCON -- the production driver -- converges that arm in 24-43
iterations either way, and does so on all ten perturbed draws. SLSQP is a **second opinion**
kept precisely so that two solvers disagreeing localises a problem (`scaled_problem`'s
docstring). Degrading agreement with PROCESS across a whole subsystem to make the second
opinion converge on one configuration inverts what the second opinion is for.

**Reverted; the tree is green.** The patch is kept at
`scratchpad/mem/coils_C1.py.patch` and the measurements above are the argument for landing
it should the trade ever look right -- if SLSQP became a production driver, or if the
`intersect` table itself were revisited on physical grounds rather than numerical ones.

**What is genuinely established and outlives the decision:**

1. The kink mechanism is confirmed -- removing it converts 8/10-with-caps into 10/10.
2. **C1 is the cheaper intervention than resolution** by an order of magnitude
   (8.1e-05 against 1.06e-03 crossing/objf perturbation), and unlike resolution it removes
   the kinks rather than shrinking them (§86's sweep was non-monotone; N = 1500 still caps).
3. The **SLSQP/VMCON 1e-04 gap is reproducible and is not the interpolation** -- it survives
   both interventions and all ten draws. That is the open question, and it is a question
   about the two solvers' KKT points, not about the model.

## 90. Correction: four subpackages were invisible to every sweep (2026-09-06)

§81, §84 and §88 all reported transpiler coverage against a denominator of **501-527
functions**, swept with `pkgutil.walk_packages` over `functional_process.models`. **That
function does not descend into namespace packages** -- directories with no `__init__.py` --
and **four of them were therefore never seen at all**: `physics/`, `costs/`, `blankets/`,
`engineering/`.

The real denominator is **853**, which lands within 3 of §76's independently-derived 850 --
that census walked the assembled graphs rather than the filesystem and was right all along.
**Three published coverage figures were measuring a subset without saying so.**

### The corrected numbers, on a merged transpiler

The two diverged copies are merged (§88's loose end): the tracked prototype's multi-return
fix plus the scratchpad's coverage work -- `jnp.pi`/`math.*`/builtin `abs`, literal-indexed
tuples, globals resolved against the **defining** module, an `enum.Enum` exclusion so
`IntEnum` members are not silently inlined as ints, `zeros_like`/`ones_like` to scalars.

| | before | **corrected** |
|---|---:|---:|
| functions swept | 501-527 | **853** |
| transpile cleanly | 326-420 | **757 (88.7 %)** |
| emitted as one module | 323-378 | **686 `@wp.func` + 77 constants** |
| mis-annotated multi-return | 151 -> 0 | **0** |
| **validated bit-identical** | 186 (124 single + 62 multi) | **455 (323 single + 132 multi)** |
| worst relative difference | 0.000e+00 | **3.18e-15** |

**455 functions agreeing, zero disagreements**, at a worst relative difference of 3.18e-15 --
float round-off, not the exact 0.0 of the smaller sample, and three orders under the 1e-9
bar.

**That is the fourth correction to this measurement in one day**, and the pattern is now
unmistakable: an unconditional return annotation, a validator that skipped what it could not
drive, a "compiled" claim that was module loading, a path rewrite pointing at the wrong
copy, and now an enumerator that silently skipped a third of the codebase. **Every one
inflated confidence by hiding a subset rather than by being wrong about what it measured.**
The measurements that survived -- §76's census, this merged sweep -- both enumerate from the
*graph* or the *filesystem* directly rather than trusting a convenience API.

## 91. The SAND leaves resolve (2026-09-06)

§88's blocker -- a naive AST walk resolving 4 of 154 nodes -- is cleared. The resolver
generalises §76's census approach and had to see through **three wrapper shapes** beyond the
plain `return calculate_x(a, b)` it assumed: equinox's `BoundMethod` (not a real Python bound
method), `.fn`-field wrappers (`_NormalisedResidual`/`_Metric`, which every constraint and
objective node uses), and a leaf passed as a *value* into a helper rather than called.

Ordered by `sand_shape(...)["drive"].body.subgraph.topological_order` -- the block with its
`Optimise` node removed, confirmed acyclic per §79:

| | `helias_5b` | `stellarator_helias` |
|---|---:|---:|
| Drive interior nodes | 94 | 123 |
| resolved to a leaf | **78 (83 %)** | **106 (86 %)** |
| structural (comparison/sign, no leaf expected) | 4 | 4 |
| unresolved | 12 | 13 |
| **resolved leaves having a `@wp.func`** | **75/78 (96 %)** | **93/106 (88 %)** |

**Why the unresolved are unresolved**, each reported rather than guessed at:

- **Switch-parameterised constraints and the objective** -- the leaf takes a static switch
  (`ireactor`, `istell`) frozen at assembly with **no `VarPath`**, and the `Leaf` contract
  had no slot for a compile-time constant. **Escalated rather than fabricated**, which was
  the right call; the contract now carries a `statics` field for exactly this.
- **Genuinely array-valued** (`plasma_composition`, `winding_pack_intersect_inputs`) --
  §76's Work bucket, no positional `VarPath` list to state without re-deriving the array
  construction.
- **Own-field and literal arguments** (`self.imp_indices`, `self.sbar`, a literal `0.0`) --
  static configuration baked into the wrapper instance, not a graph read. Also `statics`.
- One honest miss needing a second level of helper expansion.

The few "resolved but no `@wp.func`" leaves are mostly `constraint_<id>`/`objective_metric_<id>`
living in `cottax/core/solver/`, **outside the transpiler's `models/**` sweep by design** --
a scope question, not a failure.

## 92. The emitter: mechanics proven, one configuration two nodes short (2026-09-06)

Stage 1 of §79's plan, in progress. Two halves built against a fixed interface -- a
`Leaf` contract of `(node, fn, inputs, outputs, module, statics, order)` -- so the resolver
and the emitter could be developed in parallel without the shared-file contention that cost
§88 hours.

### The mechanics are proven end to end

A hand-wired kernel over **real** port functions (`rether` -> stored thermal energy ->
`_fast_alpha_fraction_ward`) compiles and runs on **both `cpu` and `cuda:0`** and agrees
with JAX to **4.9e-16 / 1.6e-16** worst relative difference -- the residual being `**`
rounding, exactly the caveat §80 recorded.

That validates the whole chain at once: argument ordering, multi-return tuple unpacking,
`^`-minted path names, the boundary/unknown split, module-global constant resolution, and
helper stitching. **Two things needed no correction anywhere**: `Leaf.inputs` in signature
order across 64 real functions, and the `VarPath` -> Warp identifier mapper, with zero
collisions on dotted paths, `^`-minted names and bracket indices.

### The resolver, after the `statics` extension

The largest unresolved group was constraints and objectives taking a **static switch frozen
at assembly** -- `ireactor`, `istell`, `self.i_p_coolant_pumping` -- which have no `VarPath`
and which the contract had no slot for. Rather than fabricate a fake input, that was
escalated; the contract gained `statics` (name/frozen-scalar pairs, accepted **only** for a
bare literal or a name resolving to `int`/`float`/`bool`/`IntEnum`) and `order` (every
parameter in signature order, so interleaved calls reconstruct by name).

| | `helias_5b` (94 interior) | `stellarator_helias` (123) |
|---|---:|---:|
| resolved | 78 -> **88** | 106 -> **116** |
| structural (`Compare`, no leaf expected) | 4 | 4 |
| **unresolved** | 12 -> **2** | 13 -> **3** |
| resolved leaves with a `@wp.func` | 75 -> **79** | 93 -> **97** |

**A real bug fell out of wiring it**: `self.<field>` reads were being resolved against the
wrapper's *class* rather than its *instance*, silently failing every static-self-field case.

**What stays unresolved, and why it should**: `.physics.plasma_composition` and
`.stellarator.coils.winding_pack_intersect_inputs` are array-valued (the leaf is passed as a
value into a `jnp.stack`-assembling helper), and `.physics.impurity_radiation_totals` takes
`self.imp_indices`, a **14-element tuple** -- correctly refused rather than rendered as a
fake single literal. All three are §76 Work-bucket items.

### What is not done, stated plainly

**No real-configuration kernel compiles end to end yet, so there is no residual-vector
agreement measurement for any real SAND block** -- and therefore no timing, and no check of
whether the tokamaks come nearly free off the shared leaves. The blocker is a small set of
Work-bucket gaps, dominated by **list-literal table lookups** (`[a, b, c, d][idx]`,
`ast.List` unsupported in a kernel) in several cost functions. Per-function Warp codegen on
`helias_5b`'s needs: **64 of 85 clean**.

**An offer that was declined, and the reason.** The emitter proposed building a kernel from
only the clean functions with stubs for the rest, to obtain *a* compile and *a* timing
number. Refused: that measures a program which does not compute the residual, and this file
now carries **four** corrections (§85, §88, §90) whose common shape was a number that looked
good because it quietly covered a subset. **A missing number is better than one that needs a
footnote saying it is not the real block.**

The remaining arithmetic on `helias_5b` is **88 resolved + 4 structural = 92 of 94**, with
two understood gaps. That is what "nearly there" looks like when the gaps are named.

## 93. A Warp kernel for a real SAND sub-DAG: bit-identical, and faster (2026-09-06)

§79's stage 1, reached. Not the whole `helias_5b` Drive -- the **maximal prefix-closed,
fully-emittable sub-DAG** of it, compiled and compared against **JAX evaluating the
identical sub-DAG**. That distinction is the whole point: this is a smaller quantity
computed *completely and correctly*, not the residual with stubs in it (§92 records that
offer being refused and why).

**Coverage: 18 of 89 entries, 1 of 11 conditions** -- reachable in one topological pass from
the 9 unknowns and 300 boundary inputs. Four leaves excluded **by name**: `pchip_interp`
(`jnp.searchsorted`), `calculate_quench_protection_current_density` (`jnp.interp`),
`calculate_parabolic_profile_values` (unresolved `scipy.special.gamma`), and
`calculate_tf_magnet_cost_superconducting_per_kg` (`ucsc[i_tf_sc_mat - 1]`, a genuine
array-valued *parameter*, so the list-lookup rewrite correctly declines it).

### Agreement

```
warp = jax = -6.25000000e+00      relative difference 0.000e+00
```

**Bit-identical**, at the reference cold-start point rather than at random values --
several boundary inputs are switch-typed and PROCESS indexes literal tables with them, so
random draws would have been physically meaningless.

### Timing: Warp wins at every batch that fits

Microseconds per point, Warp against JAX on the **same sub-DAG**:

| batch | JAX CPU | JAX GPU | **Warp CPU** | **Warp GPU** |
|---:|---:|---:|---:|---:|
| 1 | 114.8 | 221.5 | **19.7** | 31.6 |
| 16 | 7.70 | 10.5 | **1.25** | 2.71 |
| 256 | 0.585 | 0.714 | **0.089** | 0.110 |
| 4 096 | 0.030 | 0.044 | 0.013 | **0.0066** |
| 65 536 | **0.0018** | 0.0027 | 0.0089 | OOM |

**Warp CPU is 5.8x JAX CPU at batch 1, 6.2x at 16, 6.6x at 256** -- and that is the regime
that matters for a scan, not the asymptote. At 4 096 Warp GPU is **4.5x** the best JAX
number. Only at 65 536 does JAX's asymptotic per-point cost drop below Warp's floor, which
is what §71 and §80 both predicted: **Warp's advantage is fixed overhead and register
residency, not raw throughput**, and an 18-node kernel runs out of work to amortise before
JAX runs out of vectorisation.

*The 65 536 Warp-GPU cell is an allocator OOM on a 4 GB card with ~1 GB already resident;
JAX succeeded at the same batch on the same device, so it reads as fragmentation rather
than a kernel defect. Not investigated.*

### Register pressure, measured -- and the tooling problem solved sideways

§(the inlining question) could not be answered because `cuobjdump`, `nvdisasm` and `nvcc`
are all absent from this env. They are not needed: the **CUDA driver API** answers it
directly, `cuFuncGetAttribute` on the cached `.cubin` through `ctypes`.

| kernel | registers/thread | local (spill) bytes |
|---|---:|---:|
| forward | **22** | **0** |
| backward (adjoint) | **244** | **0** |

**Zero spill either way.** The backward pass costs ~11x the registers of the forward one --
which is the real, measured shape of §74's register-pressure worry, and at this size it is
not a problem. **Worth re-checking as the covered subgraph grows**, because 244 registers is
already the point where occupancy starts to bite on `sm_75`.

### Also landed

`eq`/`leq`/`geq` went into the hand-written registry: they live outside
`functional_process.models`, so the closure scanner's module-prefix rule never saw them as
helpers. **A three-function registry entry, not a transpiler feature** -- exactly the shape
the registry exists for.

### What this establishes, and what it does not

**Establishes**: the generated pipeline produces a Warp kernel that is bit-identical to JAX
on a real SAND sub-DAG, faster than JAX at every batch size that fits, with zero register
spill, and with the adjoint generated automatically alongside.

**Does not**: cover the full 11-condition residual. Coverage grows as the resolver's
remaining unresolved nodes clear and the four named gaps close, and nothing here says the
ratios hold at 89 nodes rather than 18 -- a bigger kernel amortises its overhead better
*and* uses more registers, and those pull in opposite directions.

## 94. The arity invariant: three mis-resolutions, and a guarantee (2026-09-06)

§92 found the resolver picking the **first** `functional_process.models` call in a wrapper
rather than the one producing the node's declared outputs -- `.buildings.sizing` bound to
`calculate_shield_height` (1 return) against **12** declared outputs, and it failed
*silently*, caught only by Warp's `TypeError: object of type 'Var' has no len()`.

**The fix asked for was the invariant, not the case**: *a resolved leaf's return arity must
equal the node's declared output count.* Both numbers were already in hand.

### It caught three, not one

| | `helias_5b` (94) | `stellarator_helias` (123) |
|---|---:|---:|
| resolved before | 88 | 116 |
| **resolved after** | **85** | **113** |
| newly `Unresolved` | 2 | 2 |
| newly `Composition` | 1 | 1 |

- **`.buildings.sizing`** -- the known case. Now correctly refuses: `calculate_bldgs` *does*
  match arity, but its `shh` argument is a prelude-local rather than a wrapper parameter, so
  the node comes back `Unresolved` rather than as a wrong `Leaf`.
- **`.availability.electric_production`** -- **a second instance of the same bug**, which
  nobody had found and which would have failed the same silent way.
- **`.power.delta_eta_step`** -- a new **`Composition`** category. Its wrapper is a five-call
  pipeline whose real output comes from `_, _, _, _, x = calculate_delta_eta(...)`, while
  three *other* calls in the body each also happen to return exactly one value. Ambiguous by
  arity alone, so it is reported as a composition rather than picked between.

**Counts going down is the correct direction.** A smaller set that is right beats a larger
one containing silent errors, and the invariant now holds across **all 198 resolved leaves
on both configurations, zero violations**.

### What the arity check had to see through

Counting `len(elts)` on a literal tuple is not enough. `_return_arity` handles
`return other_leaf(...)` (recursive, cycle-guarded), `return jnp.where(...)` (confidently 1,
by the codebase's scalar-typed convention), `return f(...)[:12]` (a literal slice), and
`return (a, b, *other_leaf(...))` (starred unpack). **An unrecognised call stays `None` --
unknown, never guessed as 1** -- which is why `constraint_84`'s `return geq(...)`, really
four values, does not silently resolve.

### And the input question, answered

Selection being wrong did **not** mean arguments were mixed across calls: `_bind_args`
always read from whichever call was selected, self-consistently, so a wrong selection
carried a wrong-but-coherent argument list. Now that selection is arity-correct, arguments
follow by construction. §93's "argument ordering needed no correction across 64 compiled
functions" was real rather than luck -- **though those 64 are worth re-checking against
`Leaf.fn` now that the resolver has changed underneath them**, which is exactly what §93's
own measurement needs too.

**This is the fifth silent-subset failure in this file** (§85, §88, §90, §92, and now this),
and the first one caught by a *stated invariant* rather than by a downstream crash. That is
the difference worth generalising: `test_registry_coverage.py` exists because a registry
without a coverage check drifts; a resolver without an arity check does the same thing more
quietly.
