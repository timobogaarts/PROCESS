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
