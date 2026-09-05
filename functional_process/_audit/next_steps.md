# Next steps

A current-state reference and priority-ordered punch list for the `functional_process`
port. `unit_registry.md` remains the authoritative per-unit status; this file is what's
still open at the driver/graph/solver level, above the level of any one unit.

This file used to carry a long, dated sequence of "state, YYYY-MM-DD" narrative sections
(§1 through §32) recording investigations as they happened. That narrative is no longer
here — most of what it found is either landed code, superseded by a later measurement in
the same sequence, or (where it's a refutation worth not re-deriving) moved to
`_audit/tried_and_rejected.md`. The full sequence remains in git history
(`git log -- functional_process/_audit/next_steps.md functional_process/_audit/
next_steps_archive.md`); section numbers cited elsewhere are unchanged, they just resolve
into git history rather than into this file now.

## Landed

SAND and MDF drivers both work end to end, on all seven tracked configurations, through
`jax.pure_callback` with exact forward-mode gradients (`core/solver/drivers.py`,
`sand.py`, `mdf.py`, `sand_harness.py`, `run_sand_harness.py`, `run_cold_matrix.py`).
Both tokamak and stellarator device classes are ported and registered. Reverse-mode AD
works on the whole tokamak graph's scalar objective except for one node (see
`tried_and_rejected.md`). The condition map is bound once per solve rather than rebuilt
per call; compiled-call caching lives in `core/solver/host_cache.py`; the boundary
provider distinguishes `input`/`guess`/`stated` boundary categories.

## Open

**Next session's three, agreed 2026-09-06** — all measured, none diagnosed:

1. **`helias_5b` fails under SLSQP on both arms at iteration 1**, scipy status 6,
   *"Singular matrix C in LSQ subproblem"*. That is a rank-deficient constraint Jacobian
   at that configuration's cold start, and VMCON's QP survives it. Nobody has looked at
   **which** constraints are dependent there, which is the whole question: it is evidence
   about the problem, not about scipy, and a degeneracy VMCON merely tolerates is still a
   degeneracy. Unchanged across every SLSQP run since the flag was built
   (`_audit/performance.md`, `optimise_design.md` §42).
2. **`stellarator_helias` SAND under SLSQP hits the 500-iteration cap**, 4019 block calls,
   where VMCON converges in 24. Its line search evaluates ~8 points per iteration there
   (`nfev/nit` 7.0 against 1.1-1.5 everywhere it converges), so the question is what the
   line search is failing to make progress on. The Ward kink was smoothed
   (`WARD_KINK_SMOOTHING`) and this arm is still the pathological one, so it is not that.
3. **A whole-matrix pass runs out of memory**, and the failure is odd enough to be worth
   understanding rather than working around. `run_warm_matrix` died on the fourth
   configuration with `LLVM ERROR: Unable to allocate section memory` — and the request it
   could not satisfy was **63 bytes**, so this is not one large program but XLA's CPU
   section allocator having nothing left to map. Both matrix runners now
   `jax.clear_caches()` between rows, which is a workaround; what is not known is the
   resident cost per compiled program, whether `clear_caches` actually returns it, and
   why ~44 programs a configuration exhausts a 15 GB machine when the emitted modules are
   tens of thousands of MLIR lines rather than millions. Measure RSS per program before
   theorising.


- **`vacuum.py:474` (`solve_duct_geometry`) is the sole remaining reverse-mode AD
  blocker** — a discrete first-fit search, not a root find, so it doesn't take the same
  `stop_gradient`-plus-Newton-step treatment already applied to its two sibling loops.
  Sizing for a `vmap`-over-64-candidates conversion exists; not done.
- **`sand_harness._SCHEDULE_WHOLE`/`_SCHEDULE_RUNNERS` are unbounded dicts** with the
  same trap `host_cache._BOUND` had before its removal. Not yet fixed.
  (`host_cache._BOUND` and `_flat_key` are **gone** as of 2026-09-05: the removal
  condition arrived in the better form — the graph is static outright, so there are no
  array leaves to precompute.)
- **[resolved 2026-09-05] The `stellarator_helias` SAND-arm instability.** Cause was a
  square-root singularity in `_fast_alpha_fraction_ward` (`sqrt(temp_sum_20 - 0.65)`:
  derivative zero below the threshold, unbounded above) that the optimiser sat 5.5e-08
  from and crossed on 46 % of steps. Smoothed at `WARD_KINK_SMOOTHING = 1e-3`; every
  ±1 ulp draw now takes 24 iterations and agrees on `objf` to fifteen digits, against
  87–333 and one catastrophic stop. Cost and the three rejected repairs are in
  `tried_and_rejected.md`. **Still open**: nine of twenty-six separation jumps are
  active-set changes rather than kink crossings, and are unexplained.
- **[resolved 2026-09-05] The SLSQP driver has now been run across the full
  seven-configuration matrix** (`run_cold_matrix.py --slsqp`, published as
  `reference_slsqp_matrix.txt`; `optimise_design.md` §42). 9 of 12 rows converge and
  agree with VMCON on the answer, several with better constraint residuals and one in
  17 iterations against 79. **Still open**: `helias_5b` fails on *both* arms at iteration
  1 with scipy's *"Singular matrix C in LSQ subproblem"* -- a rank-deficient constraint
  Jacobian at the cold start that VMCON's QP survives and scipy's does not. That is
  evidence about the problem, and nobody has looked at which constraints are dependent
  there. `stellarator_helias` SAND hits the 500 cap under SLSQP where VMCON takes 24.
- **Two unexplained boundary offsets, both on `.heat_transport.p_plant_electric_net_mw`**
  via constraint c16: `spherical_tokamak_eval` shows a `-16.35 MW` offset at PROCESS's own
  converged point, of the same shape as `mda_harness.EXPLAINED_DISAGREEMENTS`' documented
  stellarator mechanism (`Stellarator.run(output=True)` re-running build/coil in a
  different order than the solve pass) — but that mechanism is stellarator-specific and
  this is a tokamak. Not yet explained.
- **`^cond.constraints.c13` (burn-time lower limit) disagrees systematically** on two
  tokamak configurations, at a residual level well above every neighbouring constraint.
  Both configurations report `.times.t_plant_pulse_burn` as an array-valued fixed point
  dropped from the SAND block with its loop-carried value frozen at the seed — the likely
  cause, not yet confirmed by a direct diff against PROCESS's converged value.
- **`degenerate_fixed_points` asks a structural question with a pointwise measurement** —
  it decides "is this update the identity?" by evaluating at one point, so a live
  coupling that happens to look like the identity at that point is silently dropped.
  Should sample several points and refuse on disagreement, matching the rest of the
  port's "can't verify → refuse, don't report healthy" discipline.
- **[resolved 2026-09-05] `SlsqpDriver` derived at points nothing asked about.** Two
  defects, not one: `at` cached value and Jacobian together, and the per-iterate callback
  built a full `pyvmcon.Result` whose three derivative fields no callback in this tree
  reads. Jacobian programs on `stellarator_helias` went 3 557 -> 1 014 -> **527**, which
  is scipy's own `njev` exactly; the capped SAND arm went 22.2 s to 19.5 s and every row
  is unchanged (`optimise_design.md` §43).
- **The remaining warm-solve cost is host-side, not evaluation.** The stellarator arms
  already spend only 87-164 ms in XLA per solve; 425-552 ms goes to `pyvmcon` rebuilding
  its `cvxpy` QP every iteration and to `host_cache.call`/`__eq__` per-call overhead
  (§41, §43). A parametrised (DPP) `cvxpy` problem reused across iterations is the one
  large lever, and it is upstream of this port. The SLSQP arms' `sqp` phase reading 0.0 s
  on all twelve rows is the evidence that this is one library's allocation pattern and
  not anything structural.
- **`run_cold_matrix.main`'s docstring claims `--cache` gives "bitwise-identical
  rows", and that has never been tested as stated.** It survived the one direct check
  made of it (§42: a fresh cache reproduces the uncached numbers on `st_regression`),
  but the claim is about all twelve rows and the check was two.
- **Twenty declarations are unreferenced but only some are dead**, and the distinction is
  worth keeping: a declaration naming the same field as both an output and a read is
  refused by `to_graph` outright and can never be wired by anyone; one that is merely
  unreferenced is a valid arm nobody has wired *yet*. Three of the first kind were
  deleted 2026-09-05 (`PlantElectricProduction`, `PlantThermalEfficiency`,
  `PlantThermalEfficiency2`). `Cryo` is the fourth and is **kept**: it has a live test
  asserting `to_graph(Cryo(...))` raises, which is what stops someone "fixing" the split
  by re-merging it. The other seventeen are valid unwired arms -- the eight `Jcrit*` are
  `i_tf_sc_mat == 1..8`, `TfMagnetCostResistive` says in its own docstring it is ported
  but deliberately unregistered, `WessonInternalInductance` names
  `large_tokamak_eval.IN.DAT:311` as its arm -- and deleting them would remove
  capability, not garbage. The mechanical test is
  `_audit/declaration_census.py`'s sibling reasoning; re-derive it rather than trusting
  this list. **Open**: the other three ungraphable-by-construction classes have no test
  pinning why, where `Cryo` does. Giving them one would be better than either keeping
  them silently or deleting them.
- **`--provider` should be retired in favour of `--native` as the default** — understood,
  not done. The comparison axis (`--compare-process`) already exists independently.
- **The reduced-space MDF formulation is unbuilt**: let the graph close the equality
  constraints with a Newton solve and hand the optimiser only the design DOF, objective
  and inequalities — "multidisciplinary feasible" taken literally, a third point beyond
  today's MDF and SAND formulations.
- **`IFE.IN.DAT` is out of scope** — `.ife.*` has no unit in `unit_registry.md`.
