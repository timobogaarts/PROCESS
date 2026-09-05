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
