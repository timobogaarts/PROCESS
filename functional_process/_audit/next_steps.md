# Next steps

**What this file is.** A current-state reference and a priority-ordered punch list for
the `functional_process` port. It is deliberately *not* a changelog: sections record what
is true now and what is open, not the order in which it became true.
`unit_registry.md` remains the authoritative per-unit status.

**Section numbers are frozen.** They are cited from other `_audit/*.md` records, from the
per-unit `.md` records, and from live `.py` docstrings (`total_process.py` cites §5,
`mda_harness.py` §8, `sand.py` §6, among others). A section whose material is closed is
emptied to a stub that says where the live material went; nothing is ever renumbered.

**Where to start:** §16 (the current state and priority order, 2026-08-29), then §5 (the
structural vocabulary — Shape A / Shape B — that the code itself cites). §13's priority
order is superseded by §16.6, and §11's by §13; the measurements in both stay live.

**The `Verified state` table below is stale in more than one place** — it says 159 nodes
/ 348 unowned inputs where the live graph is 156 / 320 (`model_tree_design.md` step 4c
removed three cost nodes for a subsystem a stellarator does not have), and its suite
count is behind by the whole 2026-08-27 wave day. **§16.1 carries the current
measurements**, taken against `HEAD`; re-measure rather than cite this table, and see
§13.1 before trusting any gate measured against `cottax`'s working tree.

## Sections 1–25 — archived

Their bodies moved verbatim to `next_steps_archive.md` on 2026-09-01, under the
rule this file states above: closed material is stubbed with a pointer, never
renumbered. Each section below keeps its number and heading so citations resolve;
read the body in the archive.

- **Verified state** — `next_steps_archive.md`
- **0. Closed since the last snapshot** — `next_steps_archive.md`
- **1. Variant dispatch — the mechanism holds up under a second real case** — `next_steps_archive.md`
- **1b. Harness: the gradient error bar fix — reconfirmed, not re-diagnosed** — `next_steps_archive.md`
- **1c. Harness: default-run compile time fixed** — `next_steps_archive.md`
- **2. Review pass (yours, not mechanical)** — `next_steps_archive.md`
- **3. Consolidation — `st_fwbs` synthesis done, one new re-chunking to act on** — `next_steps_archive.md`
- **4. Remaining audit dispatches (updated — §4b's and §4c's waves closed, see §0)** — `next_steps_archive.md`
- **4b. [CLOSED, consolidated — see § 0] Dispatch — 5 parallel agents, priority-ordered** — `next_steps_archive.md`
- **4c. [CLOSED, consolidated — see § 0] Proposed next dispatch (not yet launched)** — `next_steps_archive.md`
- **5. Structural work** — `next_steps_archive.md`
- **6. Constraints, objective, iteration variables — not a separate layer, just a thin** — `next_steps_archive.md`
- **7. A third pattern, distinct from Shape A/B — raised, then resolved by a sharper** — `next_steps_archive.md`
- **8. The MDA harness — built, run for the first time, and iterated on this session** — `next_steps_archive.md`
- **9. `.costs.coe` has a producer — the `Optimise` layer's stated blocker, closed** — `next_steps_archive.md`
- **10. The `Optimise` layer — built, and the ladder run end to end** — `next_steps_archive.md`
- **11. Session wrap — verified state, and what to pick up first** — `next_steps_archive.md`
- **12. `free`, alternatives, and what a design variable means — brainstormed, to crystallize** — `next_steps_archive.md`
- **13. Priority order, 2026-08-25 — what to pick up next** — `next_steps_archive.md`
- **14. State, 2026-08-26 — the switch conversion, the driver refactor, and the tokamak** — `next_steps_archive.md`
- **15. State, 2026-08-27 — waves 2/3 registered, the PF coil block driven** — `next_steps_archive.md`
- **16. State, 2026-08-29 — the wave day banked, and the tokamak SAND ladder runs** — `next_steps_archive.md`
- **17. State, 2026-08-30 — the iteration-count gap closed to 1.26x, and what it left open** — `next_steps_archive.md`
- **18. The spherical tokamaks' complete blocker list, 2026-08-30** — `next_steps_archive.md`
- **19. State, 2026-08-30 (evening) — the cold-start matrix, and the missing producers** — `next_steps_archive.md`
- **20. `helias_5b` closed — the reference stellarator's sibling, 2026-08-30** — `next_steps_archive.md`
- **20. State, 2026-08-30 (late) — the cold start is a stage now, and it found two things** — `next_steps_archive.md`
- **21. Session close, 2026-08-30 — what landed, what is open, and what not to re-derive** — `next_steps_archive.md`
- **20. The CroCo cluster is closed, and there was a ninth blocker behind it** — `next_steps_archive.md`
- **20. The PF cluster, closed — 2026-08-30 (evening)** — `next_steps_archive.md`
- **22. The boundary provider — cutting the PROCESS seed, 2026-08-31** — `next_steps_archive.md`
- **23. PROCESS-free at runtime — the target, measured, 2026-08-31** — `next_steps_archive.md`
- **24. The target architecture, and what an importer must carry, 2026-08-31** — `next_steps_archive.md`
- **25. Corrections measured 2026-08-31** — `next_steps_archive.md`

## 26. The objective gap is decomposed, and the `_eval` files stop being two rows — 2026-09-01

§24.10 item 3 named one open measurement — *"the port's objective at **PROCESS's own
converged `x`**, which decomposes `d objf` into 'the model evaluates it differently' and
'the optimiser landed somewhere else'"* — and said no row of the cold matrix should be
described as matching PROCESS's objective until it was taken. It is taken.

### 26.1 The measurement, and the flag that makes it one command

Stage A of `run_sand_harness.py` already *was* this measurement: it evaluates every
condition of the SAND block at PROCESS's converged point and prints the port's value
beside PROCESS's own, `^cond.numerics.objf` included. What it did not have was a way to
run it alone — Stage B's finite-difference Jacobian is `5 * len(ixc)` PROCESS pipeline
sweeps and Stage C is two full solves, so taking one number cost the whole ladder or a
scratch script restating `main`'s setup, which is the third harness
`run_cold_matrix.py`'s docstring exists to refuse.

`run_sand_harness.py --stages A` (`stages()`, default `ABC`, refuses a letter that is
not a stage rather than running the ones it recognised;
`tests/functional_process/test_run_sand_harness.py`, 5 cases). The setup ahead of Stage
A is deliberately **not** gated — the PROCESS run, the MDA env and the assembly are what
every stage is a measurement *of*. A run without Stage B takes the disk cache for
`reference_run` (Stage B is the only stage that needs the live model objects), so
`--stages A` on the stellarator costs the MDA and the assembly rather than PROCESS's own
93 s solve.

### 26.2 [measured] `stellarator_helias`: at a shared design point the gap **is** the
chain, to the digit

| | value |
|---|---|
| port's objective at **PROCESS's** converged `x` | `+1.235973781` |
| PROCESS's objective at its own `x` | `+1.214916785` |
| **relative gap at the shared point** | **`1.7332e-02`** |

`mda_harness.EXPLAINED_DISAGREEMENTS[".heat_transport.p_plant_electric_base_total_mw"]`
records the `+17.604 MW` chain reaching `.costs.coe` at `rel_diff = 1.73e-2`, and
`objective_metric_6` is `coe/100`. **That is the number, not a number of the same order.**

**A second, independent witness in the same table, and it is exact.** `^cond.constraints.c16`
reads `+1.760374481e-02` against PROCESS's `+3.779981883e-09`. c16 is the *net electric
power* lower limit — `geq(p_plant_electric_net_mw, p_plant_electric_net_required_mw)`,
`constraints.py:640` — and `stellarator_helias.IN.DAT:18` sets that requirement to
`1000` MW. So the port's normalised residual is `17.60374481 MW / 1000 MW`: the chain's
own `-17.604` on `p_plant_electric_net_mw`, read off a constraint rather than off the
cost accumulation, agreeing to eight significant digits. PROCESS sits on the bound
(`3.8e-09`); the port is short by exactly that offset.

**Everything else at PROCESS's point agrees.** 11 of 15 comparable conditions are exact
at `rel < 1e-9` and the rest are `1e-16`--`1e-14`, except two that are the
divide-by-nothing shape §24.11 met twice: `c2` at `6.02e-07` (both sides `-7.375e-10`)
and `c35` at `1.15e-09` (both `-1.94e-07`).

#### The decomposition, which is the point

| quantity | value |
|---|---|
| the model's own gap, at a shared `x` | **`+1.7332e-02`** |
| the port's optimiser moving off that point | **`-1.4739e-02`** |
| `d objf` the matrix reports at the port's optimum | `+2.3381e-03` |

**The matrix row's small `d objf` was two larger effects partially cancelling, not
agreement**, and this is the third form of §17.2's error rather than a fourth: the two
objectives being differenced were never at the same design point (`worst dx 1.08e-01` at
ixc 109), and the difference of two quantities measured at two points is not evidence
about either. §24.10 recorded the arithmetic conflict honestly — *"the chain predicts
`1.73e-2` where `2.34e-03` is observed — an order of magnitude apart. Both cannot be the
same number, and the difference is presumably the design point"* — and withheld the
`EXPLAINED` label on that reasoning. **The guess was right and the withholding was
right.** `EXPLAINED_OBJECTIVE_READS`'s `worst dx <= 1e-4` gate is what a label needs, and
this row still does not pass it.

**Not measured, and it is the obvious next question:** PROCESS's *own* objective at the
**port's** design point. If the `+17.604 MW` offset were constant in `x` — which nothing
here establishes — PROCESS's metric at the port's answer would be `1.196700`, i.e. **1.5 %
better than PROCESS's own converged `1.214917`**, and the port would be finding a better
design by PROCESS's own measure rather than a worse one by its own. That is a claim worth
either proving or killing, and proving it costs one PROCESS pipeline evaluation at the
port's `x`, not a solve. Recorded as an inference, flagged as an inference.

### 26.3 [measured] `large_tokamak_nof`: the model is **exact** at PROCESS's point, so
`worst dx 6.37e-01` is not a wrong model

| | value |
|---|---|
| port's objective at PROCESS's converged `x` | `+1.600000000` |
| PROCESS's objective at its own `x` | `+1.600000000` |
| relative gap | **`0.00e+00`** |

23 of 27 comparable conditions exact at `rel < 1e-9`. §24.10's "Not done" item said of
this configuration's `worst dx 6.37e-01` (SAND, ixc 135 `f_nd_impurity_electrons(13)`)
and `2.01e-02` (MDF, ixc 57 `dr_tf_nose_case`) at `d objf ~1e-11`: *"a flat optimum, a
genuinely different local solution, and a wrong model all produce that shape."*
**The third is now ruled out** — the model reproduces PROCESS's objective exactly and its
constraints to `1e-15` at PROCESS's own point. What remains is a flat optimum or a
genuinely different solution, and those two are not separated here.

**One lead this turned up, and it is not the objective.** `^cond.constraints.c13` is the
worst comparable row on this configuration: port `-6.516518645e-06` against PROCESS's
`-8.266498332e-06`, `2.12e-01` relative and `1.75e-06` absolute — three orders of
magnitude worse than the `1e-15`--`1e-09` every other constraint manages, on a residual
too far from zero for the divide-by-nothing reading to apply. c13 is the **burn time**
lower limit (`geq(times.t_plant_pulse_burn, constraints.t_burn_min)`,
`constraints.py:589`), and `.times.t_plant_pulse_burn` is exactly the value
`mda.CUTS` cuts (§24.4) and whose fixed point this configuration's SAND assembly reports
as an **array-valued problem dropped, loop-carried value frozen at the seed**
(`^problem.times.t_plant_pulse_burn.cycle`, printed by `sand_harness.assemble` on both
tokamaks). That is the one structural candidate in view. **Not measured**: whether the
port's converged `.times.t_plant_pulse_burn` differs from PROCESS's by the corresponding
amount. One diff of that path between the warm MDA env and `reference.data` settles it,
and `mda_harness`'s 472/34 census is where it should be reconciled — the path is not in
`EXPLAINED_DISAGREEMENTS` today. `c16` on the same file (`2.58e-03` relative,
`4.2e-08` absolute, both sides `-1.6e-05`) is small enough to be that same shape and is
named only so a later reader does not rediscover it as new.

### 26.4 [measured] `large_tokamak_eval`, taken en route — c72 is PROCESS's own value

`--machine` is `large_tokamak_eval`, not `large_tokamak_nof`, so this row was measured
before the intended one and is kept because it settles something. 23 of 26 exact, and
**`^cond.constraints.c72` reads `+5.529140796e-01` on both sides at `rel 1.20e-15`.**
§21.3 recorded that constraint as violated by `+5.53e-01` with an identically zero
gradient row, and §24.10 reframed it as *"a constraint PROCESS never looks at, in a
problem PROCESS never poses"*. It is now checked from the other side: PROCESS's own
converged `DataStructure` carries that same violation to fifteen digits. The port
reproduces PROCESS exactly; what differed was only which problem was being asked.

Its `^cond.numerics.objf` row (`1.33e-06`) compares two evaluations of
`objective_metric_7`, which **PROCESS never forms for this file** — it is a model
agreement check between the port and `objective_function`, not a comparison with an
answer PROCESS produced. It is not evidence about that file's solve.

### 26.5 The `_eval` files are one row, because the formulation split has no content for
them

§24.10 left *"SAND still states an `Optimise` on the two `_eval` files"* as the same fix
applied in a second place. **It is not the same fix, and applying it there would have
been wrong.** MDF and SAND are two ways of distributing an *optimisation*: MDF hands the
optimiser PROCESS's `ixc` and converges the MDA inside each evaluation, SAND hands it
design and coupling together and holds them with residual equalities. A file stating
`i_process_run_mode = -2` states no optimisation to distribute. MDF's design vector *is*
`ixc`, so the MDF root find poses **exactly** PROCESS's own square system; a SAND root
find would pose a larger square system over design *and* coupling that PROCESS never
writes down, and printing it under a "formulation" heading beside PROCESS's answer
implies a comparison with no content.

So `run_cold_matrix` runs **one arm** on a root-find file and prints **one line**
(`_solve_both`, `render`), and the notes block states the argument on every affected row
so the single line reads as a decision rather than as a measurement that failed. The
inequalities are still evaluated once at the answer, exactly as PROCESS evaluates them,
and reported through `Mdf.reported`.

**A test changed sides, and that is worth recording.**
`test_a_formulation_that_optimised_a_root_find_file_says_so` pinned the *label* on the
mismatch — a SAND `objf` beside a `PRO objf` of `none`. It was right about the mismatch
and wrong about the remedy. It is replaced by
`test_a_root_find_file_gets_one_row_and_the_table_says_why` (one MDF line, no SAND line,
the argument in the notes) and `test_an_objf_on_a_root_find_row_is_flagged_as_unexpected`
— which keeps the old guard alive for a state nothing produces today: `mdf_graph` mints
no objective node when the file names no figure of merit, so a number in that cell would
be a defect, and the note now says so instead of explaining a known gap.

`tests/functional_process/test_cold_matrix.py` 30 passed,
`test_run_sand_harness.py` 5 passed. **The matrix itself was not regenerated** — these
two rows change shape, and §24.10's table already carries a provenance header saying it
was measured on a dirty tree.

### 26.6 What this leaves open

1. **PROCESS's objective at the port's `x`** (§26.2). One pipeline evaluation. It decides
   whether the stellarator's port answer is better or worse than PROCESS's by PROCESS's
   own metric, and no claim about that row should be made until it is taken.
1b. **Where `spherical_tokamak_eval`'s `-16.35 MW` comes from** (§26.7). The documented
   chain's cause is stellarator-specific and this file is not a stellarator. The largest
   single unexplained thing this measurement found.
2. **`.times.t_plant_pulse_burn` against PROCESS's** (§26.3, §26.7). One diff, and now
   backed by two configurations rather than one.
3. **A flat optimum against a different local solution** on `large_tokamak_nof`
   (§26.3) — unchanged from §24.10 except that "a wrong model" is eliminated.
4. ~~**Regenerate the cold matrix on a clean tree.**~~ **Done 2026-09-01 — §26.8.** Every
   optimisation row byte-identical; the two `_eval` SAND rows gone as intended; and
   `st_regression`'s failures replaced by the two real ones behind the schedule blocker
   `9335f784` closed. §24.10's "90" is still unreconciled against §22.9's "83" and
   `optimise_design.md` §20.10's "90 eager / 108 fused" — see item 8.
5. **`st_regression`'s objective condition is unreachable from its SAND block** (§26.7).
   `KeyError: VarPath(^cond.numerics.objf)`, the same failure its SAND matrix row has, and
   in the same class as the two constraints that run reports as omitted. **Not** the
   schedule blocker `9335f784` closed, and now known to sit one layer up from it.
6. **Instrument `cvxpy`'s QP refusal** (`optimise_design.md` §20.11-20.12). Every
   one-ulp failure measured across both formulations ends in `QSPSolver` — the QP
   declining a subproblem — and that component has never been instrumented. It is also
   exactly where §17's own last confident diagnosis turned out to live (the OSQP/CLARABEL
   default). **This is the live lead, and it displaced three hypotheses that did not
   survive measurement**: the active set does not change at the fork and `c24` is not
   even active there, so the documented kink is not the trigger; `cond(J)` is `1.7e4`
   median for SAND and **42** for MDF, and MDF flips anyway, so conditioning is not the
   mechanism; and nothing is frozen or dropped on this configuration (`degenerate`,
   `array_valued`, `omitted`, `external` all `0`).
7. **SAND's conditioning is one row.** `^cond.stellarator.wp_width_r_min` has row norm
   **2892** against O(1) for every other; it *is* the top singular value, and dropping it
   takes `cond(J)` from `1.7e4` to **68**. The same row §19.2 flagged and the one
   `VmconDriver.condition_scale`'s docstring calls the coil-island residual. New here is
   the MDF comparison at 42, which shows a well-conditioned version of this same problem
   exists — so this is a scaling defect in one condition, not a property of the machine.
8. **Why SAND's §15 rows moved** (83/58 → 90/80, C2 `unbounded` → 231 converged),
   unidentified and predating the fusion work. §17's correction depends on it.

### 26.9 The Stage B Jacobian, 2026-09-01 — 111 of 120 cells inside PROCESS's own error bar

Taken with the new `--stages AB` (§26.1) on `stellarator_helias`. Port Jacobian `(21, 14)`,
**0 non-finite cells**, jitted median **0.757 ms** against PROCESS's 40 pipeline sweeps in
2.06 s for the same quantity. Nine of the 120 compared cells fall outside 4x the finite
difference's own Richardson error bar: **seven of the eight `objf` columns** (1.3 % to
15 %) and **`c16` at x3 and x59** (`1.6e-02`, `4.1e-03`). Every other constraint row agrees
at `1e-4` to `1e-16`.

**This is not an error in the port's descent direction, and §11 already measured that** --
*"against a central difference of the port's own condition map the `objf` row is correct to
`7.7e-10`; §10.5c's 18-34 % is a port-vs-PROCESS difference"*. It is the
`p_plant_electric_base_total_mw` chain, which §26.2 measures in *value* at `1.73e-02` on the
same file, now seen differentiated -- and `c16` being the only other starred row is
consistent rather than coincidental, since it reads that same field. Today's 1.3-15 % is
also smaller than the 18-34 % §10.5c recorded.

`c24` at x3 reads `1.35e+282` and is **not** starred, which is the whole story in one cell:
PROCESS's FD entry there is numerically zero and its own error bar swamps the difference.
That is the documented kink -- `fast_alpha_beta`'s clamped square root, every converged
point on `(Te + Ti)/20 == 0.65` to `1.9e-09`, `epsfcn = 0.01` seven orders wider than the
feature, so PROCESS returns a chord across it and `c24` alone drifts like `h^0.52`.

### 26.7 [measured] All seven configurations, and c16 is the most informative row in the table

§§26.2--26.4 report the three configurations the open items named. The remaining four were
taken the same way and the set is worth reading whole. **Six of seven**; `st_regression`
is the exception and its reason is known (below).

| configuration | port objf at PROCESS's `x` | PROCESS's own | rel | exact rows |
|---|---|---|---|---|
| `large_tokamak_nof` | `+1.600000000` | `+1.600000000` | **`0.00e+00`** | 23/27 |
| `large_tokamak_eval` † | `+0.9286072531` | `+0.9286060221` | `1.33e-06` | 23/26 |
| `low_aspect_ratio_DEMO` | `-0.4062952568` | `-0.4062962302` | `2.40e-06` | 13/26 |
| `helias_5b` | `+0.7636707575` | `+0.7635183720` | `2.00e-04` | 4/6 |
| `spherical_tokamak_eval` † | `+0.5926272532` | `+0.5886069969` | `6.83e-03` | 14/17 |
| `stellarator_helias` | `+1.235973781` | `+1.214916785` | **`1.73e-02`** | 11/15 |
| `st_regression` | — | — | — | Stage A does not run |

† These two state `i_process_run_mode = -2`, so PROCESS forms **no** objective for them
(§24.10). The cell compares two evaluations of `objective_metric_7` at
`numerics.py:154`'s dataclass default — a model-against-model check, not a comparison
with anything PROCESS produced, and not evidence about that file's solve.

**Read the `exact rows` column with care and never as a score.** Most non-exact rows are
residuals sitting at `1e-11`--`1e-16` where a relative measure has nothing to divide by —
`low_aspect_ratio_DEMO`'s 13 of 26 is almost entirely that, with `c1` at `3.78e-01`
comparing `+6.9e-15` against `+5.0e-15`. The rows that carry information are the ones with
a scale, and there are few of them.

**`st_regression` fails Stage A with `KeyError: VarPath(^cond.numerics.objf)`** — the
objective's own condition is absent from the block's evaluated map. That is the *same*
failure `run_cold_matrix.py`'s docstring already records for its SAND build, and it is
consistent with what the run prints just above it: constraints 56 and 67 are omitted for
*"reads nothing the SAND block produces ... unreachable by its condition map"*, and this
file's objective (`i_figure_merit = -5`) is evidently in that class too. **HEAD
`9335f784` fixed this configuration's MDA *schedule*, which is a different and earlier
blocker**; nothing about the objective condition moved with it, and this measurement is
the first thing to say so.

#### c16 disagrees on three configurations, and one of the three is not a stellarator

`^cond.constraints.c16` is the net electric power lower limit and it is normalised as
`1 - value/bound` (`constraints.py:194`), so the difference in normalised residual times
the file's own `p_plant_electric_net_required_mw` is a difference in **megawatts** of
`p_plant_electric_net_mw`. Every configuration that has c16 active, at PROCESS's own
converged point:

| configuration | bound (MW) | Δ normalised | **Δ `p_plant_electric_net_mw` (MW)** |
|---|---|---|---|
| `stellarator_helias` | 1000 | `+1.760374e-02` | **`-17.6037`** |
| `spherical_tokamak_eval` | 100 | `+1.635017e-01` | **`-16.3502`** |
| `helias_5b` | 1000 | `+1.082474e-02` | **`-10.8247`** |
| `large_tokamak_nof` | 400 | `+4.23e-08` | `-1.7e-05` |
| `low_aspect_ratio_DEMO` | 350 | `+5.97e-08` | `-2.1e-05` |
| `large_tokamak_eval` | 400 | `~0` (`rel 1.69e-13`) | `~0` |

Three consequences, in increasing order of how much they should worry a reader:

1. **`stellarator_helias`'s `-17.6037 MW` is `EXPLAINED_DISAGREEMENTS`' `+17.604 MW`
   chain, read off a constraint instead of off the cost accumulation**, to six digits.
   §26.2 uses it as the second witness for the objective gap. Nothing new.
2. **The chain's magnitude is per-machine, not a constant.** `helias_5b` runs the same
   documented mechanism at **`-10.8247 MW`**. The write-up in `mda_harness.py` explains
   the mechanism with the reference stellarator's number embedded in the prose; the
   mechanism generalises and the number does not, and a reader checking a second machine
   against `17.604` would conclude the explanation had failed.
3. **`spherical_tokamak_eval` is not a stellarator, and the documented cause is
   stellarator-specific.** `EXPLAINED_DISAGREEMENTS`' account is that
   `Stellarator.run(output=True)` reruns `st_build`/`st_coil` in the opposite order to
   the solve pass, so `.build.z_tf_inside_half` moves between passes, so
   `a_plant_floor_effective` moves while `p_plant_electric_base_total_mw` is never
   recomputed. **None of that applies to a spherical tokamak**, and this file
   nevertheless shows a `-16.3502 MW` offset in the same field. Either there is a second
   dual-write of the same shape on the tokamak path, or the documented cause is a special
   case of something more general. **This is a new finding and it is not explained here.**
   The measurement that would settle it is the one `mda_harness` already knows how to
   take: instrument the report pass on `spherical_tokamak_eval` and diff
   `a_plant_floor_effective` and `p_plant_electric_base_total_mw` between the solve pass
   and the report pass, exactly as the stellarator's account was built.

#### c13 is a second systematic row, and it points at a frozen loop-carried value

§26.3 raised `^cond.constraints.c13` — the burn time lower limit — on
`large_tokamak_nof` at `1.75e-06` absolute against surrounding constraints agreeing to
`1e-15`. It repeats on `low_aspect_ratio_DEMO`: `-6.251810273e-01` against
`-6.251849209e-01`, `6.23e-06` relative and **`3.89e-06` absolute**. Two configurations,
same constraint, same order of magnitude, and both are files whose SAND assembly reports
`^problem.times.t_plant_pulse_burn.cycle` as an **array-valued problem dropped, its
loop-carried value frozen at the seed**. `.times.t_plant_pulse_burn` is the value
`mda.CUTS` cuts (§24.4) and the only quantity c13 reads. **Still not measured**, and it is
now worth more than it was: one diff of that path between the warm MDA env and
`reference.data`, on either file. §24.4's own warning is the frame — *"in the target
state a redundant cut carrying a stale value is a wrong answer on some machine nobody has
run yet"* — except that this is not the redundant cut but the array-valued drop, and the
machines have now been run.

#### One item that reads as closed

`low_aspect_ratio_DEMO`'s `^cond.constraints.c90` reads `+2.159935106e-06` against
PROCESS's `-1.979780784e-10`. §19.1 item 2 (*"Port `cs_fatigue.ncycle`"*) recorded that
constraint as **violated at exactly `+1.000000` with an identically zero gradient row,
blocking that configuration from any start**. It is neither `+1.0` nor zero-rowed now, and
§24.10's matrix has that configuration solving. Stage A is at PROCESS's converged point
and the blocker was recorded at a cold one, so this is consistent with the item having
been closed since rather than proof of it; the residual `2.16e-06` absolute is small, real,
and unexplained.

### 26.8 [measured] The matrix regenerated, 2026-09-01 — one row set changed, and it is the intended one

§26.6 item 4 asked for this on three grounds. All seven configurations, `--provider`,
one pass, **383 s** against §24.10's 751 s — the jit work of §24.4/§24.11/§20.9 roughly
halving a full matrix run is the first end-to-end measurement of it.

**Every optimisation row is byte-identical to the tracked table of 2026-08-31 22:24.**
`stellarator_helias` MDF 67/SAND 108, `helias_5b` 4/7, `large_tokamak_nof` 7/10,
`low_aspect_ratio_DEMO` 10/500 — objectives, `d objf`, `worst dx`, `max|eq|` and `min ie`
all unchanged to every printed digit. **The fused default (`optimise_design.md` §20.9)
moved nothing here**, which is consistent with §20.11's account rather than lucky: fusion
only reaches an answer where it moves a value a host-side `Drive` reads, and on six of
seven configurations it does not.

**What changed, and both were intended:**

1. **The two `_eval` SAND rows are gone** (§26.5), taking with them
   `large_tokamak_eval`'s `no-step` — the last one on the table — and
   `spherical_tokamak_eval`'s `objf 0.594644641` beside a `PRO objf` of `none`. Their MDF
   rows are unchanged: `3.29e-12` and `3.64e-09` against PROCESS's own `fsolve` x.
2. **`st_regression`'s two failures are now different failures**, and this is HEAD
   `9335f784` showing up in the table for the first time. Both arms used to read *"coupled
   block [`dr_tf_inboard_winding_pack`, `tf_inboard_radii`, `dr_tf_plasma_case`] declares
   no problem"* — the schedule blocker that commit closed. Behind it:

   - **MDF: `ValueError: the SQP was handed a non-finite problem`**, with
     `non-finite derivative rows: ['^cond.constraints.c16']` and no non-finite condition
     values and no all-zero column. This is the lead `9335f784` recorded and did not
     chase: **unjitted `jax.jacfwd` gives a fully finite Jacobian at the same point where
     `jax.jit(jax.jacfwd(...))` does not.** A jit artefact in a derivative is a different
     and more serious class than anything else on this table.
   - **SAND: `KeyError: VarPath(^cond.numerics.objf)`** — §26.7's finding, reached from
     the matrix instead of from Stage A.

**`c16` is now implicated in three separate ways and they should not be conflated.** It
carries the `+17.604`/`10.82`/`16.35 MW` net-electric offset on three configurations
(§26.7); it is one of the two Jacobian rows outside PROCESS's own FD error bar on
`stellarator_helias` (§26.9, and that is the same chain differentiated); and its
derivative row goes non-finite under jit on `st_regression` alone. The first two are one
cause. The third is not obviously related to either, and assuming it is would be the
fourth repetition of §17.2's error.

## 27. State and triage, 2026-09-01 (evening)

**Read this first.** §26 is the morning's state; this supersedes its open list.

### 27.1 What landed today

Three commits are pushed (`cd01d16a`, `01994843`, `73efe2dc`); two more patches are
applied to the working tree and not yet committed.

| change | where | evidence |
|---|---|---|
| `custom_jvp` on `PFCoil._solv` — the SVD JVP divided by `sigma_i^2 - sigma_j^2` on a **structurally** repeated singular value, so the tangent was `nan` under jit and finite eager | `models/pfcoil/currents.py` | `optimise_design.md` §21.3; pfcoil 693 passed with `--fp-gradients` |
| SQP drivers through `jax.pure_callback`; `DriverOut` replaces the mutable `Outcome` | `core/solver/drivers.py`, `sand_harness.py` | §22; verdict `(False, TracerArrayConversionError)` -> `(True, None)`; every SAND row bitwise |
| DSM hover: node I/O on the diagonal, variables first off it | `visualization/grouping.py` + 4 published pages | 21 tests, JS evaluated in `quickjs` |
| SAND cut-set knobs (`keep=`, `nest=`, `inner_drivers=`), all defaulting to today's behaviour | `sand.py`, `sand_harness.py` | §23; 193 passed |
| compiled-call cache out of the driver into `core/solver/host_cache.py` | `drivers.py` + new module | §24; 2 -> 0 compiles on the second call |

**The last one broke its own bitwise gate**, deliberately and with consent: on
`stellarator_helias`, SAND goes `stopped` at 108 -> **converged at 257**
(`objf 1.22217408 -> 1.21775737`, `max|eq| 2.85e-02 -> 3.57e-05`) and MDF 67 -> 108
iterations. Cause is demonstrated, not suspected: moving the driver's arrays from a
closure to an argument moves the compile-time/run-time boundary, and optimised-HLO
constants go 4270 -> 3046. **`reference_cold_matrix.txt` is therefore stale and must be
regenerated** (see 27.3).

### 27.2 Two negative results worth not re-deriving

- **Neither the row's weight nor the structural lift explains the `stellarator_helias`
  fork** (§23). The sharpest number: the arm that removes the lift gets the *best*
  conditioning of any arm (`cond(J)` 81, from 25265, against MDF's 42) and the *worst*
  answer. Arm 2 fails with a Jacobian matching Arm 0's Schur complement to `2.242e-15`
  and an inner tolerance that changes nothing bitwise from `1e-4` to `1e-10`.
- **Bitwise fusion-invariance is not a health signal.** It was chosen as §23's deciding
  test and it was the wrong test: invariance appears at scale factors `1e-3` and
  `4.6e-4`, is absent at `3e-4` and `1e-4`, returns at `1e-5`, and appears at `5e-3` on a
  solve that stopped at 49 iterations with `min ie -0.41`. `low_aspect_ratio_DEMO` is
  bitwise invariant *while capping at 500*. Treat any agreement measurement on this
  problem as weak evidence.

**Correction to how §23 was reported**: "the structural hypothesis is refuted" is too
strong. Arm 2 removes *one* of ~6 couplings (14 unknowns -> 13; MDF has 8), and the
intermediate points of the MDF<->SAND spectrum have no reason to be monotonic. One arm
landing worse on a configuration where six levers flip the outcome is one sample of a
chaotic system. It establishes that the in-graph move is not a *guaranteed* fix; it does
not establish that the direction is wrong. Sweeping the cut set would.

### 27.3 Now — in flight or immediately next

1. **Regenerate `reference_cold_matrix.txt`.** Stale since 27.1's last row. Run it
   **alone** — a pass concurrent with two agents died on
   `LLVM ERROR: Unable to allocate section memory` (15 GB machine), and it had already
   checkpointed 4 of 7 rows over the published file, which had to be restored from
   `HEAD`. Use `$PY -m functional_process.run_cold_matrix` (see 27.5).
2. **The HLO constant census** (agent running, `optimise_design.md` §25). 3046 constants
   remain after 27.1's change and nobody knows what they are. The invariant wanted is
   *live data is an argument, only genuine constants are constants* — a folded
   subexpression is executed by the compiler on the host, which is the wrong machine for
   anything expensive and a place where a silently-baked value will not change when it
   should.
3. **The missing-producer census** (agent running, §26 of `optimise_design.md`). See 27.4
   — this is the highest-value open item.
4. **The SLSQP column.** `SlsqpDriver` shares `pure_callback` and `host_cache` now but
   has never been run across the matrix (§21.2 says SLSQP was not put through the
   jit/eager axis; §23 lists it untouched). It separates *"this problem is degenerate"*
   from *"VMCON handles degeneracy badly"* — and for SAND there is no PROCESS answer to
   compare against at all, so a second independent optimiser is the closest thing to an
   oracle available.

### 27.4 The missing-producer problem is the headline

`st_regression.IN.DAT` sets `i_process_run_mode = 1` — a real optimisation, which PROCESS
solves in 10 iterations to `objf -16.5885765` — and `i_figure_merit = -5`, maximising
`FUSION_GAIN_Q` by reading `.current_drive.big_q_plasma`. **The tokamak graph does not own
that path**; only `models/stellarator/heating.py` produces it. So it was a boundary input
frozen at cold `0.0`, the objective was identically zero with zero gradient, and VMCON
silently solved a **feasibility** problem while reporting `converged` in 4 iterations.
Nothing said the objective was inert. `d objf 1.00e+00` and `worst dx 9.90e+01` were the
only symptoms, and both read as ordinary disagreement.

That is the **second** missing producer to surface this way (SAND on the same file fails
with `KeyError: VarPath(^cond.numerics.objf)`, same root). The open question is not this
file; it is **how much of the instability currently blamed on the solver is actually a
missing producer**, and how many more there are. The mechanical discriminator: a path
this graph *reads* but does not *own*, which some other subsystem's node does own.

Related, and not to be conflated: `st_regression`'s `min ie -1.51e-03` is **not** what is
wrong with that row — PROCESS's own converged `x` scores `-1.06e-01` on the same
constraint, 70x worse.

### 27.5 The import trap — read before measuring anything in a copy

`$PY functional_process/<script>.py` puts the **script's own directory** on `sys.path[0]`,
so the package is not found there and resolution falls through to the editable install at
`/home/tbogaarts/PROCESS`. An agent measuring in a scratch copy silently measured the live
tree twice this way. The vicious part: the natural check,
`cd <copy> && $PY -c "import functional_process; print(...)"`, **passes**, because `-c`
puts the cwd on the path. Use `$PY -m functional_process.<module>` or export `PYTHONPATH`,
and verify **inside the run** by printing `functional_process.__file__` as its first line.

### 27.6 Later

- **Sweep the SAND cut set** rather than testing one point of it (27.2's correction).
  `sand.sand_graph(keep=...)` is the knob and it landed today.
- **Instrument the merit function and line search.** Everything structural is now ruled
  out for the `stellarator_helias` fork — the row's weight, the row's presence, the inner
  tolerance, the conditioning — and this is the only uninstrumented component left with
  evidence behind it. §21.4 points at it independently: the failing QP sits one line-search
  step past the state the published row reports (`max|eq| 6.44e-01` against `2.85e-02`).
- **`mdf.solve` costs 854-971 compiles** on its first call — ~969 one-primitive programs,
  the signature of an eager `Schedule.__call__`. One line: `mdf.solve` ends with
  `mdf.eager(...)`, bypassing `run_schedule`, where `mdf.prime` a few lines away goes
  through it and costs 1 (§24).
- **Per-iteration cost is ~29 ms**, split values 31.9% / Jacobian 32.7% / cvxpy
  canonicalisation 28.8% / CLARABEL 0.9%. `solve_qsp` rebuilds `cp.Variable`/`cp.Problem`
  every call — no `cp.Parameter`, no DPP — spending 8.40 ms canonicalising to hand CLARABEL
  0.27 ms. The 14-column Jacobian costs 2.5% more than a single value evaluation, so the
  JAX side is dispatch-bound, not FLOP-bound. Neither lever alone reaches the 0.5 ms/Jacobian
  target; both would be needed.
- **`low_aspect_ratio_DEMO` MDF** stops with `Status=2` (QP refused at iteration 11, cap is
  800) at a **feasible** point: equalities `<= 2.2e-14`, all 25 inequalities satisfied. The
  KKT-vs-feasibility explanation does **not** apply there — refuted. It does apply to
  `st_regression`, where `grad f == 0` deletes the test's first term exactly.
- **`IFE.IN.DAT`** remains out of scope: `.ife.*` has no unit in `unit_registry.md`.

## 28. Plan for the next session, 2026-09-01 (close)

**Read §27 first for the state as of midday; this section supersedes its §27.3/§27.6
lists.** Ten commits landed after it (`cd01d16a` … `3107bf49`).

### 28.1 The shape of what was learned today

Three things were believed and are now measured false. They are listed first because each
was acted on before it was checked, and the pattern is the lesson.

1. **"`--native` solves better than `--provider`, so something is seeding differently."**
   The starting states are **bit-identical** — 0 differing boundary places on six of seven
   configurations. What differed is that `--provider` read PROCESS's **converged**
   `DataStructure` to decide the SAND block's *shape* (`degenerate_fixed_points`) and the
   QP's *row weights* (`residual_condition_scales`). §17.2's error, relocated from the seed
   into the problem statement. One of the two "improvements" (`low_aspect_ratio_DEMO`) is
   causal; the other (`stellarator_helias` 257 → 169) is noise — one ulp in one scale row
   moves that count by 58.
2. **"c1 and c2 are cycles; c83 is the sole active inequality at the solution."** All
   three wrong (§30). c1 is an *assignment* — making it produce `beta_total_vol_avg`
   leaves the graph acyclic, and `Stellarator.run()` already inlines it saying so. c2 owns
   no unknown anywhere. c83 is one of **four** in the solution's active set, and §23(d)
   had already withdrawn the identity reading.
3. **"The compile is slow because XLA folds closure arithmetic."** Refuted (§25): the
   whole constant pool is 24.6 kB, everything folded is scalar, no reduction, matmul,
   decomposition or loop is folded. **Compilation dominates because the module is 38 635
   lines (132 125 for a tokamak)**, not because it computes anything.

The general lesson, already stated in §23 and re-earned twice: on this problem *any*
agreement measurement is weak evidence. Bitwise fusion-invariance is not a health signal —
it appears on a solve that stopped at 49 iterations with `min ie -0.41`.

### 28.2 The rule that organises the constraint work

An equality constraint is often a cycle wearing different clothes. Over a chain
`A → a → B → b → C → c`, the constraint `a − c = 0` is `a = C(B(a))` — a fixed point, and
it exists only because two things now produce `a`.

The resolution is **not** to demote `A` to a guess. The equation determines the nearest
**producerless** variable upstream, which may be far from the constraint — and that is
exactly what an `ixc` entry names. So:

> **an `ixc` entry + an equality = drop a producer + supply its replacing equation = one cycle**

Whence the decomposition of every configuration:

    eq  equalities        determine  eq  of the ixc      <- the implicit system
    ixc − eq                         the design freedom
    inequalities + objective         the actual optimisation

| configuration | mode | ixc | eq | DOF | ineq |
|---|---|---|---|---|---|
| helias_5b | OPT | 3 | 3 | **0** | 2 |
| large_tokamak_eval | EVAL | 2 | 2 | 0 (correct) | 23 |
| spherical_tokamak_eval | EVAL | 3 | 3 | 0 (correct) | 15 |
| stellarator_helias | OPT | 8 | 2 | 6 | 12 |
| st_regression | OPT | 14 | 3 | 11 | 15 |
| low_aspect_ratio_DEMO | OPT | 19 | 4 | 15 | 21 |
| large_tokamak_nof | OPT | 20 | 3 | 17 | 23 |

Both evaluation files are exactly square, which is the cycle reading confirmed: PROCESS
root-finds them with `fsolve` and never forms an objective.

**And `helias_5b` is only well-posed as an optimisation because one of its three
equalities is broken** — c11 is inert there, restoring a degree of freedom by failing. No
configuration is over-determined, but the balance holds *by authorship* and nothing checks
the counting half.

### 28.3 The queue, in the order to take it

1. **The `min ie` sign on the root-find arm** (§29). `^cond.*` carries PROCESS's
   *unnegated* normalised residual — **positive means violated** — and
   `run_cold_matrix.cold_mdf`'s SQP arm reads the VMCON-signed callback trace while its
   root-find arm reads the raw `^cond` values. So on the two `i_process_run_mode = -2`
   files the column prints the most comfortably *satisfied* constraint under a footer
   saying "NEGATIVE IS VIOLATED". `spherical_tokamak_eval`'s `-1.38e+01` and
   `large_tokamak_eval`'s `-1.99e+00` are both that artefact; the genuinely violated
   inequality is **c56 at `+7.04e-03`**. One-line fix, but it moves every root-find row of
   a published pin, so it needs its own commit and a regenerated matrix.
2. **Regenerate `reference_cold_matrix.txt`.** Stale since `b14da8c1`. Run it **alone**:
   two passes died on `LLVM ERROR: Unable to allocate section memory` before
   `jax.clear_caches()` between rows landed, and a configuration peaks at 4.2 GB on a
   15 GB machine. Use `$PY -m functional_process.run_cold_matrix` (see §27.5).
3. **Port `.current_drive.big_q_plasma`.** The last missing producer, and the reason
   `st_regression` cannot be measured end to end under any seeding mode. First establish
   whether it is a *registration* problem — the node exists in
   `models/stellarator/heating.py` and may simply be absent from the tokamak graph — or a
   real port.
4. **Retire `--provider` and make `--native` the default.** Understood (§27), not done.
   The axis is already split (`--compare-process`), so this is the flip plus deletion.
5. **The balance guard.** `refuse_inert_conditions` catches dead rows; the counting half
   would have flagged `helias_5b` on day one. Same static walk, no solve.
6. **`degenerate_fixed_points` needs a rule.** It asks a *structural* question ("is `g`
   the identity?") with a *pointwise* measurement, so `jnp.maximum(u, m)` reads as
   degenerate wherever it is evaluated on the flat side, and a live coupling is silently
   deleted. Its own docstring records the right long-term answer — split the switch and
   spell the identity arm as an empty slot. Until then it should **sample several points
   and refuse on disagreement**, matching its existing "a block that could not be measured
   raises rather than reporting healthy" discipline. Choosing the cold start instead of
   the warm one is just the other arbitrary single point.
7. **The SLSQP column.** Never run across the matrix. It separates *"this problem is
   degenerate"* from *"VMCON handles degeneracy badly"*, and for SAND there is no PROCESS
   answer to compare against at all, so a second independent optimiser is the closest
   thing to an oracle available.
8. **Instrument the merit function and line search.** Now the *sole* surviving candidate
   for the `stellarator_helias` fork: §26 cleared missing producers on that configuration
   (0 of 294 boundary inputs differ, 0 of 15 conditions unreachable), and §23 ruled out the
   row's weight, the row's presence, the inner tolerance and the conditioning. §21.4 points
   at it independently — the failing QP sits one line-search step past the state the
   published row reports.

### 28.4 Later, and deliberately not now

**Reproducing PROCESS comes first.** MDF is PROCESS's own formulation and the regression
suite is the only oracle for numerical behaviour; restructuring before matching would make
every remaining disagreement unattributable — port bug or formulation change. §30 is
therefore diagnostic, and its recommendations wait.

When they are taken, they are: c1 → a producer for `beta_total_vol_avg` where `ixc 5` is
active (no SCC forms; PROCESS already does this); c11 → a `RootFind` over `rmajor` on the
two files where `ixc 3` is active; the pure checks and the ~21 specifications left alone,
because a constraint over only-computed quantities **cannot** become a producer and cottax
refuses it as a type error (`output(s) already produced here`).

Beyond that sits the reduced-space formulation §28.2 implies and nobody has built: let the
graph close the `eq` equalities with a Newton and hand the optimiser only the `ixc − eq`
design variables, the objective and the inequalities. That is what "multidisciplinary
feasible" means taken literally, and it is a third point beyond today's MDF and SAND.

Also open: the nine (helias) / twelve (tokamak) node outputs baked as compile-time
constants because `models/initialisation.py` bodies return Python floats — a `Scan`, which
is PROCESS's core sweep workflow, therefore recompiles the whole module per point; and
`optimise_graph`'s docstring is stale about the producer-dropping policy, which
`indat.py:5055` already implements (`0 if 140 in ixc else 1`).

### 28.5 The phase-timing block's `model` column is unverified

Added at the close of 2026-09-01 and **not to be quoted until reconciled.** The per-arm
split in `reference_cold_matrix.txt`'s `PHASE TIMINGS` disagrees with a direct probe of the
same configuration on the same tree:

| | probe, whole row | published table, MDF + SAND |
|---|---|---|
| `model` | **4.14 s** | 6.6 + 4.4 = **11.0 s** |
| `compile` | 11.1 s | 14.3 + 11.0 = 25.3 s |

The probe wraps `host_cache.flat_conditions`/`flat_condition_jacobian`, calls
`run_one(..., NATIVE, True)`, and reproduces itself to three significant figures across two
runs (`model` 4.19 / 4.14, `sqp` 1.79 / 1.77, 552 / 552 calls). So the disagreement is not
run-to-run noise. Untested causes: the table resets `phase_timing` per *arm* where the probe
resets once per *row*; the arm span may include work the probe attributes elsewhere; and
`split`'s residual clamp may be absorbing something. **Resolve before the column is used
for anything.**

What the probe does establish, and what survives the disagreement:

- **552 calls each** of the value and Jacobian entry points for 108 + 169 = 277 SQP
  iterations -- about four model evaluations per iteration, i.e. the line-search trials
  §21.1 had to wrap `pyvmcon`'s problem object to see.
- **13.7 ms and 32.7 ms per call inclusive; ~3.75 ms exclusive** of the trace/lower/compile
  happening inside each. So 75-85 % of a "model evaluation" is compilation, and a few
  hundred iterations at a few milliseconds cannot and do not account for the wall clock.
- The shape of §28's conclusion is unaffected: **compilation dominates, arithmetic is
  seconds.** It is the per-arm attribution that is wrong, not the direction.

## 31. State, 2026-09-02 — the compile picture, and what it changed

**Read §28 first for the state at the close of 2026-09-01; this supersedes its §28.3
queue only where it says so.** The detailed measurements all live in
`optimise_design.md` §31 (§31.1--§31.10) — this is the state and the priority order, not
a second copy of the numbers.

Note on citations: §28.3's references to "§29" and "§30" are to `optimise_design.md`, not
to this file, which has no such sections. This one is numbered 31 to match its record.

### 31.1 What landed

One commit, branch `eager-dispatch-compiles`: **two eager schedule walks routed through a
jit** (`mdf.solve`'s final MDA re-run, which §24.8 measured and §24.9 deliberately left;
and `run_cold_matrix.cold_sand`'s pre-solve non-finite probe, which had the same defect
and nobody had looked). A full `--native` row goes **336 → 277 → 54** XLA compiles, of
which 39 are under 20 MLIR lines. Row unchanged to the table's precision; **not checked
bitwise** (`optimise_design.md` §31.8).

### 31.2 The four things that were believed and are now measured

Same pattern as §28.1, and the same lesson.

1. **"The 0.5 ms was async dispatch."** No. `block_until_ready` changes nothing.
   §18.3 folded the `ConditionMap` in as a *constant*; `host_cache` passes it as an
   *argument*, and `eqx.filter_jit` flattens 2 382 leaves per call. **§24.1's bill,
   arriving in time rather than in numerics** (§31.6).
2. **"The module is large because of the whole-program jit."** No — 1.6 % is
   `drivers.py` and **0 %** is cottax's schedule (§31.2). And "38 635 lines" is the SAND
   *Jacobian* program, not the schedule; three programs were being quoted as one number
   (§31.1).
3. **"`safe_math`'s guards might not earn their 13 % of the literals."** They cost 4 %
   of the equations and **no measurable time**, exactly two of 52 sites fire, and
   removing them puts **NaN in the objective's gradient** at the cold start (§31.4).
   Nothing to change.
4. **"An in-graph optimiser collapses the compile bill."** It collapses the *count* to 1
   and makes compilation **3x worse** — 15.4–17.6 s → 49.7 s, 52.8k → 207–225k HLO lines
   (§31.10).

### 31.3 Now — in priority order

1. **The six-fold staging in the in-graph probe.** The single cheapest experiment on this
   list: `slsqp-jax`'s API wants six callables, so the MDA is *inferred* to be staged six
   times. Hand it one fused evaluation and re-measure. It separates "in-graph is
   expensive" from "this API stages six times", needs no new optimiser, and decides
   whether the in-graph direction is alive at all.
2. **Three `lax.while_loop`s block reverse-mode AD everywhere** —
   `models/cs_fatigue.py:318`, `models/vacuum/vacuum.py:329` and `:474`. Not the drivers:
   `PicardDriver`/`SeededNewtonDriver` go through optimistix's `ImplicitAdjoint` and
   transpose fine, so `mdf.py`'s docstring is wrong about the mechanism and
   `vacuum.py:289`'s "costs nothing here" is falsified (§31.9). Reverse mode would not
   *win* on the Jacobian's shape, but it is the only way to a cheap objective-only
   gradient, and the doc defect should be corrected regardless.
3. **The ~1.75 s of first-call jit setup** that is neither trace, lower, compile, nor
   `filter_jit` — arm-independent, bounded, unexplained. A `cProfile` of the first two
   calls (§31.6).
4. **Bind the condition map once per solve.** 6.0x on the steady state, bitwise
   identical, belongs in `host_cache.py` beside the existing pair. The floor beneath it
   is ~0.50/0.76 ms, set by jax dispatch, and is not removable from a host loop (§31.6).
5. **The persistent compilation cache** — 2.57x, two env vars, bitwise identical, but
   **leave it off while the structural work is in flight**: a cache hit erases compile
   time from the phase table (§31.7).
6. Everything in §28.3 that this section does not touch, in its own order.

### 31.4 Two findings that belong to other work

- **`.physics.nu_star` is NaN in value at the cold point**, from `dlamie == 0.0` over
  `plasma_current == 0.0` — both boundary inputs frozen at cold zero, and `dlamie` should
  be ~17. Nothing consumes it, so it is a dead output, not a live contaminant. It has
  §27.4's missing-producer signature (§31.5).
- **`jax` is 0.11.0 on at least one machine**, not the 0.11.1 `CLAUDE.md` records.
  `CLAUDE.md` already says the conda root differs per machine and ties its suite counts to
  the version; the pin should say "0.11.0 or 0.11.1 depending on machine" rather than be
  corrected to either.
