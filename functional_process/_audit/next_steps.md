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

**The three of 2026-09-06 -- diagnosed, then acted on** (`optimise_design.md` §45-§50).

1. **[closed, repair identified] `helias_5b`'s "Singular matrix C in LSQ subproblem" is
   one inert equality row** (§46, §48). PROCESS constraint 11, `rbld == rmajor`, has a
   bit-exact zero Jacobian row on both arms with a residual already at 1.1e-16; the
   equality block SLSQP factorises is rank 2 of 3 (MDF) and 8 of 9 (SAND). The cause is
   the input file, faithfully ported: `helias_5b.IN.DAT` states `icc = 11` without
   `ixc = 3`, and `stellarator_helias.IN.DAT` is the exact complement. **Both repairs were
   measured on all eight cells** and only one works: **dropping `icc = 11`** converges
   SLSQP on both arms (5 and 8 iterations) and leaves VMCON's answer and iteration count
   *identical* (`0.764215516`/`0.764215517`, 4 and 7). **Adding `ixc = 3` does neither
   job** -- SLSQP still stops at iteration 2 because `c11`'s row stays inert at 1.6e-16,
   and VMCON converges to a **different machine** (`objf` -5.6 %, `rmajor` 22.0 -> 20.911).
   **Landed**: `drivers._name_singular_equalities` names the offending rows on scipy
   status 6; its first spelling tested `!= 0.0` and was a false negative on exactly the
   variant-A case, now a relative `1e-10` test (§48). **Open**: (a) propose the
   `icc = 11` removal upstream -- it is PROCESS's input file, not the port's, and it
   changes iteration counts in the regression data; (b) **[answered -- §52]** `rbld` is
   computed *from* `rmajor` with derivative **exactly 1**: `dr_bore = rmajor - S` and
   `rbld = dr_bore + S` are two adjacent statements over the same ten terms, so
   `rbld == rmajor` identically for any values. **Constraint 11 is a tautology on the
   stellarator build path** and no choice of design variables can make it bind. It is
   genuine on the tokamak, where `rbld` is summed outward from `r_sh_inboard_out` with no
   `rmajor` in it (`d/d rmajor = 0.333`, residual -1.19e-02, live gradient). So the
   upstream recommendation is **larger** than §48's: `icc = 11` should not be listed on
   stellarator configurations at all -- not "`helias_5b` forgot `ixc = 3`", which §52
   shows could not have helped. `stellarator_helias.IN.DAT` already omits it;
   `helias_5b.IN.DAT` is the outlier.
2. **[closed as architectural] `stellarator_helias` SAND under SLSQP cycles on a
   conflicting pair, and it is not fixable by rescaling** (§47, §49). The cycle is a
   decaying period-2 zigzag on `^cond.stellarator.wp_width_r_min` against `c62`; three
   candidates were measured and rejected (not a bound, not the Jacobian, and **not the
   scale spread** -- `large_tokamak_nof` converges in 13 with a *wider* design-scale ratio,
   1.97e+23 against 2.18e+22). **Dropping `c62` ends the cycle outright** -- one large
   corrective step then near-quadratic decay, 91 iterations -- **but moves `objf` by
   0.31 %**, so it relaxes the problem rather than fixing it. **Two condition-rescalings,
   opposite targets, 100x each, both still cap**, which kills the cheap fix. The two share
   no physical quantity (`c62` is a helium-ash confinement-time floor; `wp_width_r_min` is
   a TF winding-pack geometric root find). **The verdict is architectural**:
   `wp_width_r_min` is a SAND-only exposure of an inner root find that MDF solves
   internally -- which is why the same file under MDF converges under SLSQP in 27 -- and
   SAND hands it to the outer SQP to compete in one scalar merit function against a
   near-active `c62`. **[investigated -- §53, then corrected by §56]** The knob already exists
   (`sand_graph(keep=...)` + `sand_schedule(nest=True)`, whose docstring asks this exact
   question) and **using it is worse**: nesting the `Intersect` blows up on the first
   outer Jacobian (`objf = 1.03e23`, `max|eq| = 4055`), because SAND's other five
   couplings stay free and can hand it a state MDF would never produce. `wp_width_r_min`
   is nonetheless structurally the odd one out -- the only one of the six that is an
   `ImplicitFunction` rather than a residualised `FixedPoint`. **Candidate rule**: expose
   `FixedPoint` residuals, keep genuine `RootFind`s internal -- on a sample of one against
   five, on one configuration. **§56 then demolished most of this.** The rule is dead
   both ways: `helias_5b` exposes the *same* `Intersect` and converges in 8 (readable only
   after `icc = 11` was removed, which unconfounded it), and `low_aspect_ratio_DEMO`
   exposes none and still takes 79 VMCON iterations. The nesting blow-up was a **seeding
   defect**, not the formulation: `_seed` decides "is this coupling?" by
   `source in drive.unknowns`, and `sand_graph(keep=...)` is *defined* to remove kept
   unknowns from that set -- so a nested coupling silently falls back to the cold
   `DataStructure` default `0.0`, landing Newton exactly on its flat plateau, while the
   right value sits unused in `fallback`. Patched at runtime, nesting converges in 75
   iterations at `max|eq| = 2.5e-14`. **Open**: (i) **fix `_seed` properly** -- it
   conflates "is a coupling quantity" with "is exposed to the outer block", which coincide
   only while nothing is kept; `mda.ROOT_FIND_SEEDS` also lost this path's fallback entry
   to `SUPPLIED_STARTS`. Latent, since `keep=` is a research knob. (ii) the nested answer
   is 0.012 % from VMCON's (`1.21833891` against `1.21848284`) with both at machine
   precision -- probably a different active-set path in a non-convex problem, **unverified**.
   (iii) `low_aspect_ratio_DEMO`'s slow VMCON arm is a **second, uncharacterised failure
   mode** -- creeping, feasible by iteration 2, no oscillation, nothing like the
   stellarator's. (iv) `^problem.vacuum.duct_diameter_root_find` turns out to be in **no**
   configuration's graph at all -- registered in `total_process.py` but wired to nothing,
   "driving deferred", a disconnected island. A separate pre-existing gap, and no evidence
   for any rule. Costs one row of twenty-four,
   under the non-production driver.
3. **[closed] The whole-matrix OOM is resident compiled executables; the existing trim is
   the right fix** (§45, §50). Post-trim the floor **saturates** rather than leaking
   (MDF 594 -> ... -> 851 MB; SAND 575 -> ... -> 766), and a seven-configuration pass peaks
   at 2.16 GB (MDF) / 2.23 GB (SAND). The original 63-byte failure was the trim's absence:
   four measurements a configuration, ~1 GB each left resident, reaching 15 GB in the
   fourth. **Anatomy** (§50, §54, corrected by §57): the largest module is
   2.28 M characters = 28 106 ops, of which **41.6 % is pure shape plumbing**
   (`broadcast_in_dim` alone 27.9 %) -- the signature of a scalar-valued graph in an array
   language. `jacfwd` then roughly sextuples it (the model alone is ~21 ops/node) and
   XLA's fusion pass takes it to 70 065 instructions. Its **machine code is 5.8 MB**,
   ~87-165 B per instruction; `serialize()`'s 11 MB is the optimised **HLO**, not code. **A §45
   correction was itself withdrawn**: the "expensive/cheap module classes" were compile
   *order*, not content -- with the arena trimmed first, every module is 171-351 B/char and
   §31.16's ~200 stands. **The levers, all measured** (§51): XLA's own
   optimiser *expands* the program 2.5x (28 101 -> 70 065 ops, 1 648 fusions) and the
   broadcasts survive rather than folding, so they are real work; turning that optimiser
   off saves **3.8 %** and is not a lever; MLIR bytecode is 21.6 % of the text but a more
   compact IR changes nothing resident, matching §31.20's compile-cache result.
   **Vectorising repeated structure is the lever**: 600 structurally identical scalar
   nodes cost 809 768 chars and 186.2 MB, the same arithmetic over a length-600 array
   costs **1 215 chars and 2.5 MB** with `max |diff| = 0.000e+00` -- 666x less IR, 74x
   less memory. **Open, and it is the number that decides whether any of this is worth
   doing**: `vmap` needs *identical* structure and the port's ~500 nodes are mostly
   different formulas, so the achievable gain is bounded by how much of the graph is
   genuinely repeated structure -- **unmeasured**. Count the nodes whose `fn` is the same
   callable applied to different `VarPath`s; that is a graph question, and
   `_audit/declaration_census.py` is the tool nearest to it.

   **[§60, 2026-09-06] Measured one program per process, which is the only way the
   question is well posed** -- harvest each program's StableHLO bytecode from the compile
   hook, recompile it in a fresh interpreter with nothing before it; cross-checks against
   the independent `/proc/self/maps` route to the same number. Peak is **~157 MB fixed +
   ~11 KB per post-optimisation HLO instruction** (867 MB for the largest). Split:
   **54 % transient LLVM codegen workspace** (reclaimed by `malloc_trim` with the
   executable still alive), **30 % held by the live executable** (returns when the last
   reference is dropped), and a **~142 MB size-independent residue** that survives
   everything -- of which `mallinfo2.uordblks` says only **~14 MB is genuinely live**.
   The rest is glibc free-list below the arena high-water mark that `malloc_trim` cannot
   return, so **it is a tuning problem, not a hard cost**. One real lever:
   **`MALLOC_ARENA_MAX=1`, peak -32 % (867 -> 586 MB, reproducible), compile time +57 %**
   because a single arena serialises malloc across XLA's compile threads;
   `--xla_cpu_parallel_codegen_split_count=1` does nothing. **Still open**: which arena and
   allocation class holds the unreturned ~128 MB (needs `malloc_info` XML or heap walking,
   beyond `mallinfo2`'s aggregates). Every per-program figure in §45/§50/§54 is superseded
   by §60's; §45's *whole-configuration* numbers are a different measurement and stand.
   Probes: `_audit/rss_per_program.py`, `_audit/hlo_anatomy.py`.

- **[DONE 2026-09-06] `vacuum.py` (`solve_duct_geometry`) is a `vmap` selection, and
  reverse-mode AD works on the port's graph.** `jax.grad` succeeds where it raised and
  agrees with `jacfwd`; `optimistix.BFGS` converges the real `helias_5b` MDF graph in
  9.4 s to `objf = 0.7642142560891302` against VMCON's `0.764215516`, equalities at 4e-11
  and 2.5e-05, both inequalities feasible, **with no custom optimiser code**. Seven of
  eight comparison cases bit-identical, `nflag` right on all eight, and the eighth's 1 ulp
  is the *old* loop's fusion context (see the code comment). **Now open instead**: (a)
  reverse-mode **compile time and memory** are unmeasured -- the tangent tower is
  replaced by a transpose and nobody knows what that costs here, nor whether
  `jax.checkpoint` is needed on the big blocks; (b) `slsqp_jax` and the rest of
  `optimistix` are now reachable and untried at scale; (c) `SlsqpDriver`/`VmconDriver`
  remain `pure_callback`-based, so a *driver* built on this is still unwritten.
  *(Superseded rationale, kept because it is the argument:)*
  **Necessary, not optional**: `optimistix/_solver/gauss_newton.py:176` uses `jax.jacrev`
  and offers no `jac` override, so *every* off-the-shelf `optimistix` solver is
  reverse-mode and none can run on this graph today. The node is a **discrete first-fit
  selection** (first `k` where `ceff_i_init * 0.9**k` fits between the TF coils), not a
  root find, so the goal is transposability rather than smoothness; `max_outer = 64` is a
  plain Python default and therefore a compile-time constant. A `vmap`-64-plus-`argmax`
  prototype agrees to **5.55e-17** on six real captured calls and five edge cases, is
  `jax.grad`-able where the loop raises, emits **smaller** HLO (81 201 against 85 846
  chars), costs 1.22x at `k = 0` and is **2.6x faster** at `k = 63` -- the feared 20x
  arithmetic penalty does not appear. Demonstrated end to end: `optimistix.BFGS` raises on
  the current code and, with the prototype swapped in and nothing else changed, converges
  the real `helias_5b` MDF graph to 1.6e-6 of VMCON with zero custom optimiser code.
  **To do**: write it, with a docstring caveat parallel to `solve_duct_diameter`'s -- the
  recovered gradient is the honest a.e. derivative of a piecewise-smooth selection, not a
  smoothing of a discontinuity.
  *(Superseded detail, kept for the reasoning:)* It was filed as
  "the sole remaining reverse-mode AD blocker", which undersold it by a lot: it is a
  `lax.while_loop` with dynamic bounds, so it has a JVP and **no transpose rule**, and
  every gradient-based `optimistix` solver takes its gradient by `jax.linear_transpose` --
  reverse mode. So `optimistix.BFGS` and its siblings fail on this graph with
  `ValueError: Reverse-mode differentiation does not work for lax.while_loop`, and
  `optimistix 0.1.0` offers no `autodiff_mode` switch to force forward mode. Giving the
  loop a fixed or `scan`-shaped iteration count would unlock the whole library **with no
  custom optimiser code**, and the `vmap`-over-64-candidates conversion already sized for
  it is exactly that shape. It is a discrete first-fit search, not a root find, so it does
  not take the `stop_gradient`-plus-Newton-step treatment its two sibling loops got.
  **Measured, so the prize is not hypothetical**: forward mode already works -- a
  hand-rolled Adam loop over a `jacfwd` penalty gradient, no `pure_callback` anywhere,
  reaches `objf = 0.764211181` on `helias_5b` MDF against VMCON's `0.764215516` (~6 s.f.),
  is `lax.scan`/`jit`-able end to end, and is `jacfwd`-differentiable *through the whole
  solve*. `jax.grad` through it still fails, on this same loop.
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
  17 iterations against 79. Its two failures -- `helias_5b`'s *"Singular matrix C in LSQ
  subproblem"* and `stellarator_helias` SAND's 500-iteration cap -- were **diagnosed
  2026-09-06**; see items 1 and 2 above and `optimise_design.md` §46-§47.
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
