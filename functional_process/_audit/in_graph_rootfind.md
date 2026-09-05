# The evaluation-mode root find, stated inside the graph

**Status: built.** `functional_process/mdf.py` (`in_graph_root_find`, `InGraphRootFind`,
`in_graph_solve`), `tests/functional_process/test_mdf.py`. The two `i_process_run_mode =
-2` files (`spherical_tokamak_eval`, `large_tokamak_eval`) state a square root find over
the equality constraints alone, no objective — PROCESS answers with `scipy.fsolve`. This
record is what happened when that root find was declared as a node of the graph, with
`Blocking.scc` deciding what it drives, instead of an outer driver whose residual re-ran
the whole MDA schedule per call.

## Result: the answer does not move, and the win is compile size, not wall clock

The in-graph and outer-driver formulations agree on both files to `1.4e-15` /
`8.6e-16` relative per design variable, identical Newton step counts, and reproduce
PROCESS's own `fsolve` x to the same digits recorded elsewhere (§ worst-dx ~1e-9/1e-12).
**Read the cost honestly: the win is in the program, not the clock.** Per residual, the
in-graph shape touches 6.8-8x fewer nodes and compiles 2.1-2.3x faster, but executes only
**1.2x** faster — at this size a residual is ~0.4ms and dominated by dispatch, so removing
straight-line arithmetic from one fused XLA program barely moves run time; it moves
compile time, which is what actually scales with program size (a graph an order of
magnitude larger, or a solve with hundreds of Newton steps, is where the execution column
would show it). End to end, 1.2-1.3x faster wall clock and 11→5 XLA compiles — and
walking the schedule step-by-step instead (856-993 compiles) is markedly worse, since a
`Schedule` re-enters Python once per node. **Neither number is what a rewrite should be
sold on**; the exact, configuration-independent structural claim (a residual touches 38
nodes instead of 259, derived from the graph rather than hand-maintained) is the real
result. `prune` for a caller who only wants `x` (dropping ~145 downstream blocks nothing
here drops) is still separate, unbuilt work.

## Two things worth not re-deriving

- **A stale claim in a docstring survived because the test that guarded it used a loose
  `pytest.raises(match=r"declares|problem")`.** cottax had already changed underneath the
  claim (nesting works today; `Schedule.steps` does read `blocking.inner`), but the
  regex matched *some* refusal for an unrelated reason, so the suite stayed green while
  the module's central claim became false — and a green test then reads as evidence for
  a claim that isn't true. **Rule adopted from this, applied everywhere in this file
  since**: match a `pytest.raises` message as a whole distinctive sentence (`re.escape`,
  or one `.*`-joined pattern for an interpolated one), not a disjunction of common words —
  a claim about what a pinned dependency can't do is a measurement with a date, not a
  fact, and the test must be able to tell *that* refusal from a different one or it pins
  nothing. (Two other places in this codebase caught the same failure shape independently
  — `next_steps.md`'s driver-refactor work, twice.)
- **`MdfConditionMap` (the old outer-driver residual) is deliberately still in the tree,
  not deleted**, as the independent cross-check the "answer does not move" comparison
  above depends on — remove it only after in-graph root-finding has been trusted for a
  while independently of this record.
