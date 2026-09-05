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
