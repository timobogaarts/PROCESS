# The evaluation-mode root find, stated inside the graph — record

---
kind: design + measurement
status: **built**. `functional_process/mdf.py` (`in_graph_root_find` / `in_graph_inputs`
        / `in_graph_solve` / `in_graph_shape` / `root_find_node` / `InGraphRootFind`),
        `tests/functional_process/test_mdf.py`.
confidence: high — every number below is `[measured]` on the two configurations that
            actually state the problem, and §3 pins the answer against PROCESS's own
            `fsolve` x.
---

**Scope.** The two input files whose `i_process_run_mode` is `-2`
(`tests/regression/input_files/spherical_tokamak_eval.IN.DAT` and
`large_tokamak_eval.IN.DAT`) state a square root find over the equality constraints alone
— PROCESS answers them with `scipy.optimize.fsolve` over `evaluate_eq_cons`, forming no
objective and never examining the inequalities (`process/main.py:449-462`, `_Fsolve`;
`_audit/next_steps.md` §24.10). The port answered them with an **outer** driver whose
residual function (`mdf.MdfConditionMap`) re-ran the entire MDA schedule per call. This
record is what happened when the root find was declared as a **node of the graph**
instead, and `Blocking.scc` was allowed to decide what it drives.

**Provenance.** PROCESS `9335f784` with `mdf.py` / `test_mdf.py` as recorded here and
five other files dirty from two concurrent sessions (`sand_harness.py`,
`core/solver/drivers.py`, `run_sand_harness.py`, `run_cold_matrix.py`,
`render_xdsm.py`); `~/jaxgraph` at `33af0a5`, clean. Env `process_port`, CPU jax, x64 on.
Timings are one machine, one process per arm; they are reported to one significant
figure of confidence and the *ratios* are the claim, not the seconds.

---

## §1 The upstream claim `mdf.py` was built around was stale, and the test could not see it

`mdf.py`'s module docstring and `MdfConditionMap`'s said that `cottax`'s
`ConditionMap.__call__` runs its body with `_run_acyclic`, which requires an acyclic body,
that `Schedule.steps` "never reads `blocking.inner`", and that this is why
`MdfConditionMap` has to exist at all. At `~/jaxgraph` `33af0a5` none of that is true:

- `evaluate.py:16-18` — *"Nesting is `Blocking.inner`, consumed here: a driven block's
  body is a `Step` rather than a graph."*
- `Drive` carries `(subgraph, problem, body : Step)`; `Schedule.steps` builds
  `Call(sub.runnable)` when `blocking.inner[i] is None` and `Schedule(held)` when it is
  not.
- `ConditionMap.body : Step`, and `__call__` does `at = self.body(env)`.

**[measured] Exercised, not read.** `nested_blocking(...)` on the reference stellarator,
`assign_drivers(graph, default_drivers(graph))`, `Blocking.scc(...).nest(the Optimise)`,
`schedule_for(...)`: builds a **47-step** schedule whose step at the `Optimise`'s block is
a `Drive` over **123** nodes with a **112**-block `Schedule` for its body. So MDF —
`Optimise` outside, MDA inside — schedules and runs today.

**Why the old claim survived.** `nested_blocking` returns a graph with **no drivers
assigned at all**: `mda.cut_graph` is structure only, and `sand.optimise_graph` attaches
none by default (it must not, because `Combine` refuses to join two problems that already
carry an algorithm). `schedule_for` therefore refused it — with

> `NodePath(^problem.physics.profiles.ion_vol_avg_temperature)` is a FixedPoint and
> **carries no driver**, so nothing answers block […] — structure says it must be driven,
> and `Assign` is how the algorithm is said

which is a refusal about a missing `Assign` and says nothing about nesting.
`test_mdf.py::test_cottax_cannot_run_that_nesting` asserted
`pytest.raises(ValueError, match=r"declares|problem")`, and the word *problem* appears in
that message. **The test passed for the wrong reason**, so an upstream change that
falsified the module's central claim was invisible to the suite.

**As built.** That test is replaced by two:
`test_the_undriven_nesting_is_refused_for_want_of_an_assign` (matching
`r"carries no driver"` exactly) and `test_cottax_runs_that_nesting_once_the_drivers_are_assigned`
(asserting `isinstance(Drive.body, Schedule)` and that the body is the block minus the
problem). Together they say *why* it refuses, which is what the loose regex threw away.

**Generalisable, and this is the third time.** `_audit/next_steps.md` §24.11 caught
`mdf.traceable_drivers`' `TracerArrayConversionError` claim the same way, and §24.4 the
`check_antichain` one. A claim about a pinned dependency is a **measurement with a date**,
not a fact.

**The reusable half is about the test, not about the claim.** A `pytest.raises` whose
`match=` is a disjunction of common words (`r"declares|problem"`) does not pin a refusal;
it pins *that something refused*, and the graph refuses for a dozen reasons. So a stale
claim can go on passing while the thing it claimed becomes false — and worse, the test
reads as evidence for the claim in every later review. The rule this pass adopts, applied
to **every** `pytest.raises` added here: match the message the code actually writes, as a
whole distinctive sentence, via `re.escape` where the message is fixed and one
`.*`-joined pattern where it interpolates. `test_the_undriven_nesting_is_refused_for_want_of_an_assign`
now matches *"carries no driver, so nothing answers block … structure says it must be
driven, and `Assign` is how the algorithm is said"*, which no other refusal in cottax
produces. Where a docstring says "upstream cannot", the test under it must be able to tell
*that* refusal from a different one, or it is pinning nothing.

---

## §2 What `Blocking.scc` actually drives

The formulation: insert a `cottax.problem.RootFind` owning the `ixc` design variables and
reading the equality-constraint condition variables (`mdf.root_find_node`); `Assign` a
driver onto it and onto every problem already in the graph; `Blocking.scc`; `.nest(the
root find)`. Nothing is a new mechanism — every step is cottax's own, and the design
variables were boundary inputs of the MDA graph (`assemble` refuses an assembly where they
are not), so registering the problem turns free inputs into owned variables and changes
nothing else.

**[measured] The prediction and the result agree exactly.** The prediction is
`descendants(readers of the design) ∩ ancestors(owners of the conditions)` on the graph
*without* the problem — computed independently of `Blocking`, and pinned as
`test_the_driven_block_is_exactly_what_reachability_predicts`:

| | `spherical_tokamak_eval` | `large_tokamak_eval` |
|---|---|---|
| graph nodes (without / with the problem) | 259 / 260 | 271 / 272 |
| design variables / equalities | 3 / 3 | 2 / 2 |
| **predicted loop** | **38** | **34** |
| **driven block** (`Blocking.scc`) | **39** | **35** |
| `Drive.body` — the block, one problem in | 38 | 34 |
| blocks that run **before** it | 69 | 80 |
| blocks that run **after** it | 147 | 142 |
| total blocks in the outer schedule | 217 | 223 |

The block is the loop plus the problem, on both files. So ~85 % of each graph is outside
the loop: 69/80 blocks are upstream of the design variables and constant for the whole
solve, 147/142 are downstream of the constraints and have no vote in choosing `x`.

**[measured] The fraction is a property of the run, not of the formulation**, and the
test's bound is deliberately loose because of it. `test_mdf.py`'s `in_graph` fixture
root-finds the reference *stellarator*'s first two `ixc`, and its block is **61 of 169** —
36 %, against 15 % and 13 % on the two files that actually state this problem. Which `ixc`
are active is what decides how much of the graph the loop reaches; a tighter assertion
would be pinning one configuration's reachability and calling it a law.

### §2.1 The nesting

**[measured]** `Blocking.inner` at the root find's block — the block minus the root find,
blocked again by `Blocking.scc`:

| | `spherical_tokamak_eval` | `large_tokamak_eval` |
|---|---|---|
| interior blocks | 29 | 25 |
| of which driven | **2** | **2** |
| their sizes | 2, 9 | 2, 9 |
| driven blocks in the whole MDA (`mdf_shape`) | 4 | 6 |

Both interiors' driven blocks are the same two, and they are the same on both machines:

- `[2]` `FixedPoint` — `.physics.profiles.ion_vol_avg_temperature` and its
  `^problem.…` copy;
- `[9]` `FixedPoint` — `.physics.fusion_rates`, `.physics.fusion_power_totals_mw`,
  `.physics.fusion_totals_no_beam`, `.physics.plasma_composition`, the three
  `.physics.profiles.*` nodes, `.physics.profiles.parameterisation.*`, and
  `^problem.physics.proton_rate_density.cycle`.

**[measured] And the coupled blocks the loop does *not* meet.** Multi-node SCCs of the
graph without the problem: 4 on `spherical_tokamak_eval` (sizes 2, 2, 5, 9) and 6 on
`large_tokamak_eval` (2, 2, 2, 5, 9, 10). Exactly **2** of each fall inside the loop — the
2 and the 9 above; the rest are entirely upstream or downstream and are driven **once**
per solve rather than once per residual. That is the whole efficiency case, and it is a
statement about *this* graph's structure, not about MDF.

This is the first `Blocking.inner` built in this tree — `sand_harness._driven_runner`'s
nested-body branch says *"Nothing in this tree builds one today"*, and it is now reachable
(no change made to that file; the branch is simply live for these schedules).

---

## §3 The answer does not move

**[measured]** Both files, cold, from the input file's own values, primed once by
`mdf.prime`, one process per arm:

| file | shape | Newton steps | `max|eq|` | `min ie` | **worst dx vs PROCESS's `fsolve` x** | §24.10's |
|---|---|---|---|---|---|---|
| `spherical_tokamak_eval` | outer driver | 2 | `1.16e-09` | `-13.759` | `3.635e-09` (ixc 6) | `3.64e-09` |
| `spherical_tokamak_eval` | **in-graph** | 2 | `1.16e-09` | `-13.759` | **`3.635e-09`** (ixc 6) | — |
| `large_tokamak_eval` | outer driver | 3 | `3.50e-14` | `-1.9899` | `3.292e-12` (ixc 4) | `3.29e-12` |
| `large_tokamak_eval` | **in-graph** | 3 | `3.47e-14` | `-1.9899` | **`3.292e-12`** (ixc 4) | — |

Both reproduce §24.10's table to the digit it prints. The step counts are identical, which
matters: the two shapes are the same Newton on the same residual, so a different trajectory
would have shown as a different count long before it showed as a different `x`.

**[measured] The two formulations against *each other*** — the stronger check, since both
arms share every model, every inner driver, every seed, and the same primed env, and differ
only in where the problem is stated. Per design variable, `|in-graph − outer| / |outer|`:

| file | worst |
|---|---|
| `spherical_tokamak_eval` | `1.4e-15` |
| `large_tokamak_eval` | `8.6e-16` |

**[measured] And the whole output env**, key by key at the answer:

| file | shared keys | worst relative where the quantity has a scale (> `1e-8`) | max absolute anywhere |
|---|---|---|---|
| `spherical_tokamak_eval` | 1116 | `5.3e-13` (`^cond.constraints.c9`, abs `1.2e-15`) | `6.3e+06` |
| `large_tokamak_eval` | 1181 | `8.3e-14` (`.physics.j_plasma_bootstrap_sauter_profile`, abs `4.2e-08` of `5.1e+05`) | `7.9e+06` at `.physics.molflow_plasma_fuelling_required` — scale `4.6e+21`, **rel `1.7e-15`** |

A naive worst-relative over *all* keys reads `9.8e-02`, and that is §24.4's caveat and not
a finding: it lands on `^cond.constraints.c1`, a residual at the root, where a relative
measure has nothing to divide by. The largest absolute difference in either env is a
last-bit difference on a quantity of order `1e+21`.

**So this is XLA reassociation, not a moved answer.** `_audit/optimise_design.md` §19–§20's
finding — that a last-bit perturbation *can* move an iterative solve's trajectory — is
live and is the reason the check above is stated three ways rather than one; here it did
not fire, and the identical Newton step counts are the evidence that it did not.

---

## §4 What it costs, and where the win actually is

Two different measurements, and conflating them would overstate the result.

### §4.1 Per residual — the clean comparison

The outer-driver residual is `mdf.condition_map(...)`, whose body is the **whole** MDA
schedule. The in-graph residual is `Drive.condition_map(env)` at the driven block, whose
body is the interior's `Schedule`. Both jitted under `eqx.filter_jit`, timed as the median
of 25 calls after one warm-up, on a primed env:

| file | | nodes per residual | jaxpr equations | compile (s) | jitted (ms) |
|---|---|---|---|---|---|
| `spherical_tokamak_eval` | outer driver | 259 | 14 147 | 1.95 | 0.428 |
| `spherical_tokamak_eval` | **in-graph** | **38** | **3 403** | **0.92** | 0.360 |
| `large_tokamak_eval` | outer driver | 271 | 17 290 | 2.18 | 0.387 |
| `large_tokamak_eval` | **in-graph** | **34** | **3 358** | **0.94** | 0.336 |

and the Jacobian the outer Newton actually takes (`jax.jacfwd` of the same map, one XLA
program either way):

| file | | compile (s) | jitted (ms) |
|---|---|---|---|
| `spherical_tokamak_eval` | outer driver | 5.93 | 0.630 |
| `spherical_tokamak_eval` | **in-graph** | **3.22** | 0.579 |
| `large_tokamak_eval` | outer driver | 7.29 | 0.596 |
| `large_tokamak_eval` | **in-graph** | **3.58** | 0.557 |

**[measured] The residual values agree** between the two maps at the same design point to
`4.2e-16` and `5.6e-16` absolute.

**Read this honestly: the win is in the program, not in the clock.** 6.8× and 8.0× fewer
nodes, 4.2× and 5.1× fewer jaxpr equations, 2.1× and 2.3× faster to compile, 1.6× and
2.0× faster to differentiate — and only **1.2×** faster to *execute*. At this size a
residual is ~0.4 ms of CPU and is dominated by dispatch, so removing 220 nodes of
straight-line arithmetic from a single fused XLA program buys almost nothing at run time.
That is not an argument against the restructuring; it is the same measurement
`_audit/next_steps.md` §24.10 made from the other end — *the cold matrix's 120 s per solve
is compile* — and compile is exactly what scales with program size. On a graph an order of
magnitude larger, or a solve with hundreds of Newton steps rather than two or three, the
execution column is where this would show; here it is the compile column.

### §4.2 End to end — and the bigger lever is orthogonal to this change

From a primed env to the answer *and* the full output env, one arm per process so no warm
jit cache leaks between them (`prime` itself costs ~7–8 s and 1 compile in every arm and
is excluded):

| file | shape | wall (s) | XLA compiles | Newton steps |
|---|---|---|---|---|
| `spherical_tokamak_eval` | outer driver (`condition_map` + driver, then `run_schedule(eager)`) | 17.6 | 11 | 2 |
| `spherical_tokamak_eval` | **in-graph, `in_graph_solve`** | **14.2** | **5** | 2 |
| `spherical_tokamak_eval` | in-graph, plain `Schedule` step walk | 30.0 | 856 | 2 |
| `large_tokamak_eval` | outer driver | 20.3 | 11 | 3 |
| `large_tokamak_eval` | **in-graph, `in_graph_solve`** | **15.5** | **5** | 3 |
| `large_tokamak_eval` | in-graph, plain `Schedule` step walk | 35.4 | 993 | 3 |

All rows record the driver's step count; the in-graph rows are the current default shape
(§6 closed the two defects that used to make "5 compiles" and "a step count" alternatives,
and these numbers are re-measured after that).

1.2× and 1.3×, and 11 compiles against 5. **The third row is the point of the table**: run
step by step, a `Schedule` re-enters Python once per node and XLA compiles once per `jnp`
primitive (`_audit/optimise_design.md` §18.6), so the ~215 ordinary `Call` steps either
side of the driven block cost twice what the whole solve does. `sand_harness.run_schedule`
— one jit for the entire run, the outer Newton and both nested inner solves inside it — is
the fix, and it is **not this change's**; it is §24.11's. `in_graph_solve` takes it
whenever the schedule can be hashed, and §6 is why it always can be.

**Neither row is the number a rewrite should be sold on.** The in-graph shape's structural
claim — a residual touches 38 nodes instead of 259 — is exact and configuration-independent
in a way the seconds are not.

---

## §5 What "nodes per residual" does and does not mean

The outer-driver shape re-runs every node **because nothing told it not to**: `MdfConditionMap`
is handed `mdf.traceable`, the schedule over the whole graph, and its `__call__` writes the
design values into the env and calls it. There is no filter, no `prune`, and no way to
write one that is not a hand-maintained node list — which is exactly the thing the graph
exists to derive.

Stating the problem inside the graph does not make the solve *cheaper by fiat*. It makes
the question askable: `Blocking.scc` answers *what is coupled to this problem* from the
reads and owns that were already there, and the answer is a partition, so what is not in
the block is not merely skipped — it is **placed**, before or after, and run exactly once.
That is why the upstream/downstream split in §2 is a table of blocks and not a count of
"nodes we managed to avoid": the schedule is a total order over every node in the graph,
and the loop is one of its steps.

The corollary is the honest limit: **`prune` is still the other half.** Nothing here drops
the 147/142 downstream blocks for a caller who wants only `x`. They are cheap (they run
once) and they are what `min ie` and the output writer read, so nothing wants them dropped
today — but "the residual is 38 nodes" and "the run is 260" are different numbers and this
change moves only the first.

---

## §6 Asking for the step count used to cost the single jit — two defects, both closed

**This section previously recorded a trade-off and reported a seam in `sand_harness.py`.
Both were wrong, and in an instructive way: what looked like one problem in the *cache*
was two problems on the *driver*, and both are fixable where they are.**

`MdfNewtonDriver.outcome` is a mutable results sink the driver writes its step count and
optimistix verdict into (a field and not a return value, because
`AbstractDriver.__call__`'s contract is the answer and nothing else). Asking for one used
to cost `sand_harness.run_schedule`'s whole-schedule jit — **856** XLA compiles / 30.0 s on
`spherical_tokamak_eval` and **993** / 35.4 s on `large_tokamak_eval`, against **5** / 14.4 s
and **5** / 16.4 s without. Same answer to `1e-15`; only the cost differed.

### §6.1 The hash — closed by `core/solver/drivers.Outcome`

A driver is an `eqx.Module` and goes into the graph (`assign_drivers`), so
`Schedule.__hash__` reaches its fields through `Graph.__hash__`, and `run_schedule` hashes
the schedule on every call to key its whole-jit verdict and its runner groups. A plain
`dict` field therefore raised

> `TypeError: unhashable type: 'dict'` — at `sand_harness.py:485`, `_SCHEDULE_WHOLE.get(schedule)`

several frames from the driver that caused it, which reads as a cottax problem and is not
one. `core/solver/drivers.Outcome` (a `dict` subclass with `__hash__ = object.__hash__`,
added by the `SlsqpDriver` session for the identical failure) is the fix, **and it belongs
on the field rather than in the cache**: identity is the right hash for a mutable results
sink — two outcomes are the same outcome when they are the same object — and the values
are written after every hash that matters has been taken.

**Refused, not coerced.** `MdfNewtonDriver.__check_init__` raises on a bare `dict`, naming
`Outcome`. Wrapping a caller's `dict` would be worse than refusing it: the caller would go
on writing into *their* object while the driver wrote into a copy, so the step count would
read back empty with nothing raising. `mdf.Outcome` re-exports the class so a caller of
this module needs one import and not two.

### §6.2 The write — closed by `jax.debug.callback`, and it was mine

**[measured] `Outcome` alone did not close the trade**, which is why this section is
measured twice. With the hash fixed, `run_schedule` *attempted* the single jit and its
probe still failed:

> `TracerArrayConversionError: The numpy.ndarray conversion method __array__() was called
> on traced array with shape int64[]`

`MdfNewtonDriver.__call__` wrote `int(np.asarray(solution.stats["num_steps"]))`,
`bool(solution.result == optx.RESULTS.successful)` and `str(...result._value)` — three
concretisations of jax arrays, which inside a trace are tracers. So the outer driver
carried **exactly the defect `mdf.traceable_drivers` documents on the inner ones**
(`_audit/next_steps.md` §24.11's `_usable`), written by this module, three years of
docstring about the inner case notwithstanding. `run_schedule` fell back to its fused walk:

| `spherical_tokamak_eval`, outcome asked for | wall | XLA compiles |
|---|---|---|
| plain `Schedule` step walk (the old fallback) | 30.0 s | 856 |
| `Outcome` only — single jit refused, fused walk | 16.0 s | 26 |
| **`Outcome` + `jax.debug.callback`** | **14.2 s** | **5** |

| `large_tokamak_eval`, outcome asked for | wall | XLA compiles |
|---|---|---|
| plain `Schedule` step walk | 35.4 s | 993 |
| `Outcome` only — single jit refused, fused walk | 17.6 s | 34 |
| **`Outcome` + `jax.debug.callback`** | **15.5 s** | **5** |

`MdfNewtonDriver._record` now takes the three values through `jax.debug.callback`, which
runs host-side when the program runs — so one code path serves the eager call and the
compiled one, and the conversions always see concrete numpy. Verified both ways: the
eager `mdf.solve` path still reports 3 Newton steps on `large_tokamak_eval`, and the
compiled `in_graph_solve` path reports the same 3 with `whole=True` and 5 compiles.

**[measured] The answer is untouched by either fix**: `3.635e-09` and `3.292e-12` worst dx
against PROCESS, the same step counts, the same `max|eq|` and `min ie` as §3.

### §6.3 As built, and what it says about seams

`in_graph_root_find(outcome=None)` now **constructs** an `Outcome`, so the step count is
on by default and costs nothing; `InGraphRootFind.outcome` is always present, with `steps`
and `successful` reading off it. `in_graph_solve` chooses `run_schedule` on **whether the
schedule actually hashes** (`_hashable`, which asks by hashing) and not on what `outcome`
holds — the old branch inferred the answer from one particular field, which is exactly
what made two independent things look like one trade. That branch is now a guard against a
hand-built driver carrying some other unhashable field, and `test_mdf.py`'s
`tier4` value test asserts `schedule_verdict(...)` is `whole=True` **and** that the step
count came back, which is the only place both halves are checked together.

**No change to `sand_harness.py`.** The seam this section originally proposed — rekeying
`_SCHEDULE_WHOLE` by `id(schedule)` — would have hidden §6.1 rather than fixed it, and
would have done nothing at all about §6.2. The general lesson is the same one §1 records
from the other direction: *a failure that surfaces in the cache is not necessarily a
defect in the cache*, and the frame the `TypeError` is raised in is the frame that
happened to ask the question, not the one that answered it wrongly.

## §7 `MdfConditionMap` is removable, and is deliberately still here

Given §1, `MdfConditionMap` computes nothing `cottax.evaluate.ConditionMap` does not: its
one difference — a body run by a `Schedule` rather than by `_run_acyclic` — is what
`ConditionMap.body : Step` now is. The replacement `mdf.py`'s docstring named
(`nested_blocking()` → `schedule_for` → delete `MdfConditionMap`) is available.

**It is not removed in this pass, and that is the rule and not caution.** A restructuring
and a deletion landing together makes a moved answer unattributable: §3's whole claim is
that this change moved nothing, and it is only checkable because the outer-driver arm is
still there to compare against. `test_the_in_graph_root_find_gives_the_same_answer` runs
**both** arms in one process at one primed env for exactly that reason. The removal is a
second pass, and the thing to check on it is that `mdf.solve`'s `Optimise` arm — the five
non-evaluation configurations — still reaches the same iterates through a plain
`ConditionMap`.

**Also still open, and larger:** the `Optimise` arm can be stated in-graph too — §1
measures that it schedules — but its outer driver is a `VmconDriver` (a `cvxpy` QP, a
`pyvmcon` line search, a Python callback), which does not trace, so the whole-schedule jit
of §4.2 is not available there and the measurement would be a different one wearing the
same name. `mdf.root_find_node` refuses an `Optimise` with that message rather than
silently answering it with a `SeededNewtonDriver`.

---

## §8 What cottax could not express

Nothing, on this problem. Every step is a cottax primitive used as documented:
`Insert` a `DeclaredNode`, `Assign` a driver, `Blocking.scc`, `Blocking.nest`,
`schedule_for`. Three small frictions, none of them a limitation:

1. **A problem over several unknowns has no name to be minted from.**
   `rewrites`' `mint_key_problem` spells a problem `^problem.<var>`, which names it after
   the one variable it owns; a root find over two or three `ixc` entries has no such
   variable. `sand.optimise_graph` already met this and binds a plain `.Opt`;
   `mdf.IN_GRAPH_PLACE` is `.RootFind` for the same reason, and `place` is a parameter so
   two of them in one graph do not collide. Not a gap — `Combine`'s `place` argument is
   cottax's own answer to exactly this, and this is the same fact one level up.
2. **`Blocking.nest` needs the drivers attached first**, and the refusal it gives
   otherwise reads as a refusal of the nesting (§1). That is a message-quality issue in
   `Drive.__check_init__`, not a structural one, and `~/jaxgraph` is not this session's to
   change.
3. **The `Start` ports change hands.** A design variable is a boundary input before the
   problem is inserted and driver data (`^guess.<place>`) after it, so a seeding routine
   written for one shape does not fill the other. `mdf.in_graph_inputs` applies the rule
   `seed`/`prime` already apply to every inner unknown — fill a `^guess.*` port from the
   unknown it starts — and raises naming the port rather than defaulting to `0.0`. Getting
   this wrong is silent (the outer Newton would start from the cold point instead of the
   input file's `ixc`), which is why it is a test.

---

## §9 Files

| file | what changed |
|---|---|
| `functional_process/mdf.py` | `IN_GRAPH_PLACE`, `InGraphRootFind`, `root_find_node`, `in_graph_root_find`, `in_graph_inputs`, `in_graph_solve`, `in_graph_shape`, `_hashable`; `Outcome` imported and re-exported; `MdfNewtonDriver.outcome` retyped with a `__check_init__` refusal and its writes moved onto `MdfNewtonDriver._record` behind `jax.debug.callback`; the module docstring's "Evaluation: not expressible today" section corrected to §1; `MdfConditionMap` and `nested_blocking` docstrings corrected |
| `tests/functional_process/test_mdf.py` | `test_cottax_cannot_run_that_nesting` replaced by `test_the_undriven_nesting_is_refused_for_want_of_an_assign` + `test_cottax_runs_that_nesting_once_the_drivers_are_assigned`; eleven new tests for the in-graph shape, two of them `tier4` value checks over both `_eval` files; every new `pytest.raises` matched on the exact message (§1) |

`test_mdf.py`: **37 passed**, ~130 s (the two `tier4` rows need `sand_harness.reference_run`,
which is disk-cached; a cold cache costs one ~95 s PROCESS solve per file).

Nothing else was touched — `core/solver/drivers.py` supplied `Outcome` and was read, not
edited, and `sand_harness.py` needs no change at all (§6.3).

**The one line `run_cold_matrix.py` needs** (that file is not this session's). Its
`root_find` arm at line 437 builds a bare `dict`, which `MdfNewtonDriver` now refuses by
name:

```python
        outcome: dict = {}          # ->
        outcome = mdf.Outcome()
```

Nothing else on that arm moves: `Outcome` **is** a `dict`, so `outcome.get("steps")` at
lines 449/453/459 reads exactly as before, and the object the caller holds is still the
object the driver writes into (§6.1). Switching the same arm from `mdf.assemble` +
`mdf.solve` to `mdf.in_graph_root_find` + `mdf.in_graph_solve` is a separate and larger
change; §3's table is the evidence that the row's numbers would not move if it were made.
