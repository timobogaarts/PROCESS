# paper_tests

The uncertainty study of the PROCESS section of *Managing complexity for integrated
design in fusion using a graph-based functional framework* (`~/graph_paper`): what
happens to PROCESS's deterministic stellarator optimum when the beliefs behind it are
sampled. Three drivers, one shared module, one job script. Everything else this
directory once held (the census, timings, architectures, batching, close-conditions,
Sobol' and OUU studies and their outputs) was removed on 2026-09-20 and is in git
history at `eae0ae45`.

**The scripts are thin command-line drivers over the port.** The physics, the graph
operations and the batched programs live in `functional_process/cottax/architectures/`
(`session`, `closing`, `lift`, `stages`, `beliefs`, `ouu`) and
`functional_process/configurations/kinds.py` (the decision kinds and the belief
table); what lives here is the command line, the statistics, the figures, and the one
thing the port deliberately does not carry -- PROCESS itself in the loop
(`common.process_reference`: one converged VMCON run, cached under
`~/.cache/functional_process/`, which is the deterministic design every study starts
from and is compared against).

| file | what |
|---|---|
| `flexibility.py` | **the study.** Fix the machine, let the operator re-optimise per belief draw; the coe histogram against the deterministic optimum. Its docstring is the recipe and the answer table. `--pairing he --table flex` (the default) is the 2026-09-22 form: the pruned belief table and the helium fraction closed by `c62` inside the MDA rather than a free second knob; `--pairing one --table screening` reproduces the 2026-09-17 study. |
| `flexibility_dsm.py` | the graph that study solves, as two interactive DSM pages (`out/dsm/`); its docstring is the graph recipe from the uncut base graph, step by step with node counts, extended 2026-09-22 for the `c62` closure and the one-knob operator problem |
| `ga.py` | the converse question: a genetic algorithm over the machine, scored by the batched MDA across sampled worlds -- is there a machine that fares better? (still reads the wide `kinds.BELIEFS_SCREENING`-shaped table via its own call, not the pruned default) |
| `common.py` | the env pin, `process_reference`, `deterministic_values` |
| `decision_kinds.md`, `output_kinds.md` | the census behind `kinds.py`: every build decision (design variable or fixed), every belief, and the guarantee that no build decision is left in the graph |
| `cluster/` | `job_flex.sh` (the 2026-09-17 table/pairing) and `job_flex_he.sh` (the pruned table, `c62` closure; `TAG=he` output prefix), N = 262144 on one Snellius `rome` node in ~6-7 min each, and their environment |
| `out/` | the tracked runs: `flex_n262144_s{0.10,0.05,0.02}_hfact{,_req982.4}.npz` + `_summary.json` (2026-09-17 table, `--pairing one`), `flex_he_n262144_s{0.10,0.05,0.02}_hfact{,_req982.4}.npz` + `_summary.json` (2026-09-22 table, `--pairing he`, including the rated-power variant), `flex_coe.png`, `flex_histograms.png`, `flex_he_histograms.png`, `ga_run.json` (penalty 1000), `ga_run_penalty250.json`, `ga_best.json`, `ga.png`, `dsm/` (regenerated, not tracked) |

```bash
PY=~/miniconda3/envs/process_port/bin/python      # see ../CLAUDE.md for the env
export JAX_PLATFORMS=cpu PYTHONPATH=~/PROCESS:~/PROCESS/paper_tests
cd ~/PROCESS
$PY paper_tests/flexibility.py --n 1024 --workers 8 --order hfact --knobs te          # ~3 min
$PY paper_tests/flexibility.py --n 1024 --workers 8 --order hfact --require-net 982.4
$PY paper_tests/flexibility.py --figures --tag he
$PY paper_tests/flexibility_dsm.py
$PY paper_tests/ga.py --pop 32 --n 512 --gens 40 --certify 4096                       # ~10 min
```

**2026-09-22: do not prepend a `cottax_head` archive of `6b1d540` to `PYTHONPATH`.**
Every architecture module under `functional_process/cottax/architectures/` (`closing`,
`mdf`, `sand`, `lift`, `mda`, ...) now imports from `cottax.pytree.*`, and that
subpackage does not exist at `6b1d540` at all (`git -C ~/jaxgraph show
6b1d540:src/cottax` lists a flat package -- `cottax.graph`, `cottax.problem`, no
`pytree/`); it was reintroduced later in `~/jaxgraph`'s history. The commands above
resolve cottax from the ordinary editable install, which does have it. This supersedes
the older instruction (still correct as history, wrong as a recipe to run today) to
pin `PYTHONPATH` at `6b1d540` -- CLAUDE.md's own warning that "the port tracks a
cottax that moves" cuts both ways: the pin can go stale in either direction. **The
port now tracks `~/jaxgraph` at `6488a1c`** (measured: `git -C ~/jaxgraph rev-parse
HEAD`) -- write down the commit the next time this drifts, rather than a commit that
was already several re-ports old when this file first named it.

**The actual bug, found and fixed at the source, 2026-09-22 (two earlier notes here were
each superseded same day; both are kept below, struck through in spirit not in text,
because the trail is part of the record).** Every SLSQP-driven call in `flexibility.py`
(and `ouu.two_stage(...).make()` generally, once the belief table was narrow enough)
failed to lower with `jaxlib.mlir...MLIRError: "jit(branched_error_if_impl)" ...
operand type mismatch: expected 'tensor<1xf64>', got 'tensor<Nx1xf64>'`, from inside
`equinox`'s `EnumerationItem.error_if` under `jax.jacfwd`. **Root cause**:
`architectures.drivers.PicardDriver.__call__` (`mda.default_drivers`' driver for every
`FixedPoint`-shaped problem, which includes `^mda.fwbs.f_ster_div_single`, the divertor
wetted-fraction cycle) called `optx.fixed_point(...)` at optimistix's default
`throw=True` -- deliberately, on cottax's own reasoning that "a budget is only honest
if running out of it is loud" (`cottax.execution.drivers.optimistix.PicardDriver`'s
docstring). `throw=True` raises out of `error_if` when the step budget is exhausted,
and that `error_if` reproducibly fails to *lower at all* once composed with
`jax.jacfwd` through a `vmap`'d batch -- an equinox/optimistix batching-rule gap, not
this study's code, but *triggered* by narrowing the belief table (with the 26-row
screening table, enough beliefs reached every driven sub-problem that none of them hit
the code path that exposes it). **Fix**: `PicardDriver.__call__` now calls
`optx.fixed_point(..., throw=False)` and reads the verdict as data -- which this
port's driver already does elsewhere (`Steps`/`Converged` when `report_steps=True`; a
raised exception inside a batched trace was never visible to any caller anyway).
Confirmed end to end, real files, no workaround: PLASMA + LIMITS alone (nine rows, no
extra belief) traces and runs `flexibility.py --pairing he` cleanly through `jacfwd`,
`hoist=False` (what `flexibility.py` actually uses). The N = 256 local smoke test and
the N = 262144 x 3 sigma Snellius run both complete on this fix -- see below.

**A separate, narrower bug remains, not fixed here.** `test_ouu.py`'s own fixtures use
`ouu.make(model)` with `hoist=True` (the *first stage hoisted, second stage batched*
split flexibility.py does not use, `hoist=False` there). Under `hoist=True`, even with
`throw=False`, the same divertor `FixedPointDriver` (once nothing sampled reaches it)
returns a solution shaped `(N+1, 1)` where its own `Start` guess was shaped `(1,)` --
a `ValueError` at `unravel_pytree`, not an MLIR crash, so at least it surfaces cleanly
now. Four of `test_ouu.py`'s fourteen tests fail on this (`test_assembly_has_the_shape
_the_handoff_states`, `test_flattened_closure_answers_the_same_nominal_row`,
`test_value_jac_starts_is_finite_and_matches_a_central_difference`,
`test_one_boxed_slsqp_call_moves_the_design_and_reports`); the other ten pass. This is
a real bug in `ouu.py`'s hoist/stage-crossing machinery -- a driven sub-problem that
becomes a function of first-stage constants alone under the pruned table -- not
attempted here: it is not what `flexibility.py` needs (which never hoists), fixing it
blind risks the `hoist=True` path every other caller of `ouu.make` relies on, and it is
outside this task's remaining budget. Flagged as follow-up work.

*(The trail, in order: the first note here reported the MLIR error as a pre-existing
`jax==0.11.0`/`equinox==0.13.8` incompatibility, "reproduces on the unmodified
`test_ouu.py`" -- true only because `test_ouu.py`'s fixture builds its model with
`beliefs=` defaulting to `kinds.BELIEFS`, so testing against the "unmodified" test
after `kinds.py` was already edited in place was not a control against the original
table. The second note, after the coordinator's clean-worktree bisection caught that,
correctly narrowed the trigger to one belief row (`.divertor.tdiv`) and shipped
"keep it sampled" as a working but unexplained fix. This note replaces both: the
mechanism is now known and fixed at its actual source, and `tdiv` is back out of
`kinds.BELIEFS` (nine rows again, PLASMA + LIMITS only, no workaround row).)*

## The answer (N = 262144, `flexibility.py`, 2026-09-17 table, `--pairing one`)

With the build held -- five numbers, the winding-pack width among them once
`lift.lift_winding_pack` stops the coil from being re-sized per sample -- and the
operator free to set `T_e` and the helium fraction in each world, PROCESS's machine
can be run in 43 % of worlds (hfact sigma 0.10; 40 % / 38 % at 0.05 / 0.02), and in
those worlds its cost of electricity is centred **1.29x the deterministic value**
(median 159.5 against 123.6 $/MWh; 1.19x / 1.12x at the smaller sigmas). The
deterministic optimum sits at the 10th percentile of its own distribution: only one
operable world in ten is cheaper than the one it was designed in. Every failure is an
operating failure (the seven build constraints hold in every world), the binding set is
c83 / c35 / the pack rule at ~100 %, all fixed by the build, so the knobs buy almost
nothing. Rated power is a reference line, not a criterion: 10 % of operable worlds reach
it (`flex_coe.png`, lower panel).

## The pruned table / `c62` closure (2026-09-22, `--pairing he` the default)

The belief table was pruned from 26 rows (`kinds.BELIEFS_SCREENING`, still importable)
to nine -- PLASMA (six rows: confinement and five other 0-D-closure beliefs), LIMITS
(two rows: the constraint thresholds that are themselves physics beliefs,
`beta_vol_avg_max`, `f_t_alpha_energy_confinement_min`) -- and the helium fraction
stopped being a second free operator knob: `c62` (`tau_He*/tau_E >=
f_t_alpha_energy_confinement_min`) is now named as a closure beside `c2`, so particle
balance sets it (`kinds.PAIRINGS["he"]`, `closing.close`, extended to accept an
inequality as a closure). The operator's problem carries one unknown (`T_e`) and five
movable conditions instead of two and six; see `flexibility_dsm.py`'s docstring for the
measured node counts.

**The reference point is the operator's own optimum in the nominal world**, not
PROCESS's own converged point -- PROCESS's design was constrained by `c16` = 1000 MW
as an *equality* with the He fraction as its closure, and the operator's problem here
has neither, so comparing against PROCESS's 123.6 $/MWh conflates "no `c16`
requirement" with "uncertainty". The right reference is `flexibility.py`'s own
formulation with every belief at nominal (`hfact` = PROCESS's own 1.056, T_e optimised
under the same `he` closure): **coe = 125.17 $/MWh, net = 969.7 MW, T_e = 5.78 keV,
f_He = 0.0301** -- active at c24, c83, c35 and the pack rule (c24, the beta limit, is
*already active* at the nominal point, which is exactly why LIMITS sampling matters so
much at small sigma; see below). 123.6 is kept in the table as context only.

**N = 262144 x 3 sigma, Snellius, `job_flex_he.sh`, 2026-09-22** (after the
`PicardDriver` fix below; `flex_he_n262144_s{0.10,0.05,0.02}_hfact{,_req982.4}.npz` +
`_summary.json`, `flex_he_histograms.png`):

| hfact sigma | operable | operable at rated power (982.4 MW) | coe p5 / p50 / p95 ($/MWh) | coe p50 ratio to the nominal-world 125.17 | coe p50 ratio to PROCESS's 123.6 (context) | net MW p5 / p50 / p95 | T_e p50 (keV) | f_He p50 |
|---|---|---|---|---|---|---|---|---|
| 0.10 | 0.471 (123455/262144) | 0.298 (78096/262144) | 90.36 / 134.11 / 223.70 | **1.071x** | 1.085x | 510 / 896 / 1411 | 6.70 | 0.0366 |
| 0.05 | 0.462 (120979/262144) | 0.346 (90774/262144) | 88.43 / 121.69 / 185.16 | **0.972x** | 0.985x | 626 / 1000 / 1447 | 6.23 | 0.0344 |
| 0.02 | 0.458 (119949/262144) | 0.379 (99225/262144) | 88.68 / 116.38 / 161.84 | **0.930x** | 0.942x | 727 / 1052 / 1442 | 6.05 | 0.0335 |

Active set in every sigma: c83, c35 and the lifted pack rule at 100 % (fixed by the
build, same as the old table), c24 (beta) at 47-50 % (a genuine chance constraint now
that `beta_vol_avg_max` is sampled, where the old table held it fixed), c17 near 0 %.
`c62` is not in the active-set table at all -- it is closed, not read as a movable
condition (`closing.close` drops it from `report["inequalities"]`). At rated power the
operable fraction drops by a third to a fifth (0.471 -> 0.298 at sigma 0.10; the gap
narrows at smaller sigma, 0.458 -> 0.379 at 0.02) -- consistent with the old table's
own finding that rated power is a much harder bar than mere operability, now under the
new closure too. What drives operability, by correlation with the belief's quantile:
`hfact` dominates at sigma 0.10 (+0.66) but *not* at 0.02 (fourth of five, well behind
`f_temp_plasma_ion_electron`/`alphat`/`beta_vol_avg_max`) -- with `hfact`'s own spread
narrowed, the other beliefs' fixed spread starts to dominate which worlds are operable.
Timing: 268-311 s wall per task on one 96-worker `rome` node, 77-84 s of that
compiling once; workers peaked at ~2 GB RSS -- the `throw=False` fix costs nothing
measurable.

**Against the old table** (`flex_n262144_s*_hfact_summary.json`, `--pairing one`, two
operator knobs, 2026-09-17): operable fraction is **higher** under the new
table/pairing at every sigma (47.1/46.2/45.8 % against 43.1/39.9/37.7 %) -- closing the
He fraction by particle balance rather than leaving it free does not cost operability;
if anything the beta and He-exhaust limits becoming *sampled* widens the operable set
about as much as losing a free knob narrows it. Against the *correct* reference (the
nominal-world operator optimum, 125.17, not PROCESS's 123.6), coe moved from
consistently *above* the deterministic reference under the old table (1.29x/1.19x/1.12x
against 123.6) to **roughly at or below the nominal-world optimum** under the new one
(1.071x/0.972x/0.930x against 125.17) -- most of the old table's apparent coe premium
was an artefact of comparing against a reference the new formulation was never trying
to reproduce (PROCESS's `c16`-constrained point) plus the 26-row table's extra
cost-model and balance-of-plant beliefs (all held at nominal here); the physics-only
premium at the historical hfact sigma (0.10) is a modest 7 %, and at the smaller,
arguably more realistic sigmas the median operable world is *no more expensive* than
the nominal one.

**LIMITS sensitivity (task 4): a real share of sigma = 0.02's inoperable worlds are the
limit draws alone.** `beta_vol_avg_max` U[0.035, 0.05] is not centred on the file's
0.04, and `f_t_alpha_energy_confinement_min` U[3, 6] is not centred on the file's 4 --
and PROCESS's own optimum sits *on* c24 (confirmed above: c24 active at the nominal
point too), so holding `beta_vol_avg_max` fixed at 0.04 while everything else varies
biases operability low whenever a draw's other beliefs would have needed a bit more
beta headroom. Quantified locally (N = 256, same seed/order, LIMITS sampled vs. held
at nominal, otherwise identical): at sigma 0.10 the difference is small (operable
0.465 sampled vs. 0.441 nominal, +5 % relative); at **sigma 0.02 it is not** (0.465
sampled vs. 0.371 nominal, **+25 % relative**) -- as `hfact`'s own spread narrows, the
LIMITS draws stop being a minor correction and become a first-order share of why a
world is or is not operable. This is why the pruned table keeps LIMITS sampled rather
than fixed at the file's nominal, independent of the driver bug below.

**The `PicardDriver` fix (task 1) and the deterministic point.** Evaluated without
differentiating (`ouu.make(...)["run_batch"]` alone): at PROCESS's own converged
`hfact` = 1.056, the combined root find (density, two cut copies, thermal alpha
fraction) converges in 0 Newton steps from the primed start, coe = 127.20 $/MWh, net =
952.30 MW, helium fraction 0.0299 (PROCESS's own optimum: 0.0336, chosen to hit `c16`
= 1000 MW rather than the He-exhaust limit, which `c16` is not in this constraint
set), `c62`'s residual 9e-13 (at its zero, as closing it demands). Sweeping `hfact` in
[0.8, 1.2] from that same fixed start converges at 0.9-1.2 (3-9 Newton steps) and fails
to converge at 0.8 -- the closing Newton is always started from the nominal priming
point, not warm-started along a sweep, and a non-convergent closing Newton is treated
as an infeasible/inoperable draw by the existing machinery (`ouu.py`'s `FAILED_G`),
which is exactly what the N = 262144 run above shows happening at the low tail rather
than a silent wrong answer. Differentiating through this same point (`jax.jacfwd`, what
every SLSQP call in the study needs) used to fail to *lower at all* for the pruned
belief table -- see the driver bug and its fix, described in full where it was found
and fixed, `architectures.drivers.PicardDriver.__call__`'s docstring
(`functional_process/cottax/architectures/drivers.py`), summarised: cottax's own
`PicardDriver` calls `optx.fixed_point` at optimistix's default `throw=True`, which
raises out of `equinox`'s `EnumerationItem.error_if` on a step-budget exhaustion, and
that `error_if` cannot be lowered once composed with `jax.jacfwd` through a `vmap`.
The port's `PicardDriver.__call__` now passes `throw=False` and reads the verdict as
data instead. A separate, narrower bug in `ouu.py`'s `hoist=True` path (which
`flexibility.py` does not use) remains open -- see the same docstring and the note
further up this file.
