# paper_tests

Generators for the PROCESS section of *Managing complexity for integrated design in
fusion using a graph-based functional framework* (`~/graph_paper`). Each script measures
one thing over the seven reference input files and writes it to `out/` as a `.csv`
(numbers), a `.tex` fragment (a `tabular`, for `\input`) and, where the raw rows are
worth keeping, a `.json`. Every fragment's first line says which script, tree, cottax
commit and machine produced it.

**The scripts are thin command-line drivers over the port.** The physics, the graph
operations and the statistics live in `functional_process/cottax/architectures/`
(`session`, `mda`, `recipes`, `mdf`, `sand`, `evaluate`, `closing`, `stages`,
`beliefs`, `ouu`) and `functional_process/configurations/` (`kinds`: the decision kinds
and the belief table); what lives here is the command line, the rendering (tables,
figures, JSON), the measurement loops, and the one thing the port deliberately does
not carry -- PROCESS itself in the loop (`common.process_reference`: one converged
PROCESS run per input file, cached under `~/.cache/functional_process/`, for the
comparison columns and for the deterministic design the UQ and OUU studies start
from). `common.py` is the shared writer and provenance line.

```bash
PY=~/miniconda3/envs/process_port/bin/python      # see ../CLAUDE.md for the env
export JAX_PLATFORMS=cpu
$PY paper_tests/graph_census.py        # ~1 min   structure: nodes, SCCs, cuts, body depth
$PY paper_tests/timings.py --measure   # ~30 min  cold / warm per arm and optimiser; then
$PY paper_tests/timings.py             # instant  renders out/timings.json
$PY paper_tests/mda_convergence.py     # ~5 min   Picard steps per block under each cut
$PY paper_tests/architectures.py       # ~1 h     MDF/SAND x cut x optimiser, warm and cold
$PY paper_tests/stage_check.py         # instant  the two-stage split's violations
$PY paper_tests/close_conditions.py --table --solve   # ~7 min  equalities closed in the MDA
$PY paper_tests/uq.py --n-base 64 --n-mc 64 --chunk 64   # the smallest Sobol' study
$PY paper_tests/ouu.py --smoke --n 32 --alpha 0.75 --max-iter 3   # the smallest OUU run
```

`batching.py` is the exception to the env line above: it needs a CUDA jaxlib, so it runs
in `process_port_gpu` (see "Batching" below).

If `~/jaxgraph`'s working tree is mid-edit, pin cottax with
`PYTHONPATH=<a worktree of its committed HEAD>/src` (`../CLAUDE.md`, "The port tracks a
cottax that moves").

## What each table says

**`graph_census`** -- the raw graph per configuration (nodes, boundary inputs, the cyclic
components and their sizes, the problems the models declare themselves), then per cut
recipe the copies minted and the deepest coupled-block body. The body depth is the
structural parallelism number: a Jacobi body is one layer (every node of the component
evaluates from the previous iterate), a Gauss-Seidel body is as deep as its sweep.

**`timings`** -- one row per configuration and arm, under VMCON and SLSQP: cold total
(assembly, compile, first solve), SQP iterations, verdict, warm wall -- measured
through `architectures.session` (`--measure`, into `out/timings.json`) and rendered
from that. The `reference_*_matrix.txt` tables this used to read were deleted with
the harness (`f3015ede`), and with them the phase split (trace + lower / compile)
and the model ms per call: those columns are gone.

**`mda_convergence`** -- the inner analysis alone, from one cold state
(`evaluate.cold_state`: the graph after one pass in call order, which is where
PROCESS starts too): Picard steps per coupled block, their sum, the sum weighted by body
depth (sequential node evaluations), and the warm wall of one converged MDA.

**`architectures`** -- MDF and SAND assembled on each cut and answered by each optimiser
(`session.open_session(name, optimiser=..., cut=...)`, `solve(arm)` cold then warm):
iterations, verdict, objective, design entries (a Jacobi SAND carries whole profiles),
cold wall and warm wall. One `tabular` per optimiser. The XLA share of the warm wall
and the ms per model call are no longer measured (the harness's timing hooks went
with it).

**`stage_check`** -- the two-stage split (`architectures.stages`: `leaves`, `split`,
`violations`, `report`) on the three leaf sets of the 2026-09-17 handoff (`all`,
`sampled`, `build`), on the 156-node graph it measured and against the claims it had
(`kinds.CLAIMED_BUILD_OUTPUTS` sections 1a-1d): 27/156 & 13, 36/156 & 13, 60/156 & 4.
`--runnable` for the 154-node graph the port runs, `--lifetimes` for the two
section-4 claims added since; `tests/architectures/test_stages.py` pins all four.
Writes `out/stage_check.{md,json}`.

**`build_spread`** -- the four rule-closed build outputs that still vary per sample on
the `build` table, their spread over the belief samples at a design: `ouu.two_stage`
through `ouu.py`'s `Choice`, a local `vmap` of the closed MDA for the columns.

**`batching`** -- whether `jax.vmap` pays on the GPU (RTX 3080, 10 GB, FP64 at 1/64 of
FP32). Two shapes, each vmapped over N = 1, 4, ..., 65536 independent points and timed
per point on the CPU and on the GPU *in the same env* (`process_port_gpu`, jax 0.11.1
+ CUDA 12; `JAX_PLATFORMS=cpu` for the CPU rows): `mda`, the hand-cut MDA of `scan.py`'s
shape at N points on a grid of `.physics.rmajor` x `.physics.b_plasma_toroidal_on_axis`;
and `sand`, the SAND block's fused value+Jacobian at N perturbed designs. `plot_batching.py`
draws `out/batching.png` (us/point) and `out/batching_wall.png` (wall per call).
Findings: the stellarator MDA crosses over at N~2000 and is still halving per 4x at 16k
(8.7 us/pt); the tokamak never crosses because `pf_coil`'s SVD is padded past cuSOLVER's
batched limit; f32 is unusable (fusion rates overflow) and irrelevant (latency-bound).

**`close_conditions`** -- `stellarator_helias`'s two equalities closed as root finds
inside the MDA instead of handed to the optimiser -- `architectures.closing` (`close`,
`seed`, `nested_blocking`, `describe`, `cycle_of`; the `SafeguardedNewtonDriver` is
`architectures.drivers`'). `--table` ranks every pairing by the cycle it closes
(`closing.cycle_of`) and its sensitivity (`jacfwd` of `mdf.condition_map`); `--solve`
runs the 6-variable optimisation (`mdf.solve` on `Closed.problem`, VMCON and SLSQP,
cold then warm, the plain MDF and PROCESS's own answer beside it); `--batch` vmaps
the MDA. The default pairings are `kinds.HISTORIC_PAIRING` (`c2` by `hfact`, `c16` by
the alpha fraction); `--pairing one|two|te` are `kinds.PAIRINGS`. Four batch shapes: `plain` (the hand-cut MDA), `nested` (root finds with
the cycle's Picards nested inside, exact Newton), `closed` (the c16 root find `Residualise`d
and `Combine`d with those Picards into one 4-unknown square Newton with a Broyden update --
no loop inside the loop; the default), and `predicted` (`closed` started from a first-order
predictor across the batch). `out/batching_closed*.png` plot them. On the GPU at N=4096:
plain 83 ms, nested 196, closed 137, predicted 107 -- a converged, equality-feasible
evaluation for 1.3-1.7x a plain one. Two lessons that transfer: a `while_loop` nested in a
vmapped `while_loop` must mask its predicate by the outer activity, and on this card a
forward tangent through FP64 transcendentals costs a primal, so a Jacobian recomputed
every step loses to a secant update.

**`uq`** -- the deterministic PROCESS optimum under uncertainty. `stellarator_helias`'s
design is fixed at PROCESS's own converged `ixc` (`common.process_reference`, handed to
`ouu.two_stage` as `design_values` / `closing_values`), `hfact` -- PROCESS's closure
variable for the power balance -- becomes an *uncertain input*, and one closure stays
inside the MDA: the power balance `c2` closed by the plasma density
(`kinds.PAIRINGS["one"]`, `closing.close` flattened: the root find combined with the
fusion-rate Picard, 3 unknowns, Broyden). The net-electric equality `c16` is not
closed -- net power is an output and its residual against the 1000 MW target is
reported like an inequality (`with_c16=True`, so it is a column of the batched
program). The belief table is `kinds.BELIEFS` (26 boundary inputs plus a `dummy`
nothing reads; `--variant old` is the 2026-09-16 table the numbers below were measured
with: hfact lognormal(0.15), tungsten and ripple lognormal(ln 2 / 2)); the batched
program is `ouu.make(model)["run_batch"]` (the first stage hoisted, the recourse
vmapped), the predictor's Jacobian `ouu.sensitivity`, the row mask `ouu.valid_rows`.
What is no longer a column: the fusion power and the closed residual `c2` (the
batched program returns coe, net power, availability, concost, the constraints, the
closing steps and verdict, and the closing unknowns). Saltelli sampling (scrambled
Sobol', (k + 2) N = 29 x 8192 = 237 568 evaluations), Saltelli-2010 S1 and Jansen ST
with 200-replicate bootstrap CIs, and a plain 100 000-sample Monte Carlo, all
evaluated in chunks of 16 384 on the RTX 3080. Outputs:
`out/uq_inputs.tex`, `out/uq_sobol.{csv,tex}`, `out/uq_mc.{csv,tex}`, `out/uq.json`,
`out/uq_tornado_{net_power,cost,concost,c24}.png`, `out/uq_hist_{net_power,cost}.png`.

What it found (2026-09-16, with the `old` table and the harness's own batched
program; the nominal sanity check below reproduces on the port's units):

- **Cost and convergence.** 337 568 closed-MDA evaluations in 95 s wall end to end
  (build 17 s, the sensitivity `jacfwd` 21 s, three compiles of ~12 s each for the
  start-strategy calibration, then 0.496 s per 16 384-point call: **33 300 evaluations/s
  warm, 30 us per converged evaluation**). Every evaluation converged (0 of 337 568
  dropped; max |c2| <= 1e-10) over ranges that move the closed density by a factor
  0.2-10. The Newton is started from a log-linear predictor of all three closing
  unknowns (`u0 * prod (x_i/x_i0)^s_i`, `s_i` the scaled sensitivities from one `jacfwd`
  at the nominal): 3.5 steps mean / 8 max against 5.6 / 11 from the nominal root, 654 ->
  496 ms per call. A linear predictor is worse (4.6 / 8), and predicting only the density
  while leaving the Picard copies at their nominal values makes the Newton stall on a
  quarter to a third of the points -- the combined problem's unknowns have to move together.
- **Sanity.** At the nominal the closed MDA returns PROCESS's density to 3e-9 (relative)
  and its fusion power to 6e-9; net power is 982.4 against PROCESS's 1000.0 MW and coe
  123.6 against 121.5 $/MWh -- exactly the documented +17.6 MW base-load offset of
  PROCESS's own report pass (`functional_process/deliberate_divergences.md`); concost agrees to
  2.3e-4. The port's `c24` at PROCESS's point is +1.5e-3 (PROCESS: -4.8e-7) because it
  carries `beta_fast_alpha` = 5.9e-5 where PROCESS's stored value is 0, and `c35` is
  +1.6e-4 from a 1.6e-4 difference in `j_tf_wp`; so `P(g > g_nominal)` is reported next
  to `P(g > 0)`. The dummy's S1 and ST are exactly 0 (its AB block is bit-identical to A),
  and sum S1 <= 1 for every output (1.000 for the five constraints that read one input).
- **One input owns the answer.** `hfact` has ST = 0.90 [0.83, 0.96] on net power, 0.91 on
  coe (capped at 1000 $/MWh -- the raw coe is a 1e21 sentinel wherever net power <= 0,
  5.4 % of samples), 0.89 on concost, 0.86 on the beta limit; the closed density scales
  as hfact^-3 (scaled sensitivity -3.0) and fusion power as its square. Next, an order of
  magnitude down: the tungsten fraction (0.06), `alphat` (0.06), T_i/T_e (0.06), `fhole`
  (0.03), `f_p_alpha_plasma_deposited` and `eta_turbine` (0.02). Every plant and cost
  input -- availability, plant life, discount rate, superconductor unit cost, coil
  temperature, insulation -- is below 0.005 on net power and cost. `eta_ecrh_injector_
  wall_plug` has ST = 0 on every output (nothing reported reads it: no injected power in
  this stellarator's flat top); `bmn`, `f_asym`, `tdiv` reach only the divertor heat-load
  constraint `c18` (ST 0.01, 0.005, 0.01). Four TF-coil constraints (c32, c34, c35, c82)
  read only `f_j_tf_wp_critical_max` (plus the insulation for c82), and c65 only
  `dr_fw_wall`.
- **The optimum's feasibility is luck.** Net power: median 858 MW, mean 1553, 5-95 %
  -8 to 5382 MW, P(< 1000 MW) = 0.55, P(<= 0) = 0.054; fusion power 5-95 % 767-16 690 MW
  (nominal 2973); coe median 151 $/MWh (nominal 124). P(all twelve inequalities
  satisfied) = 0.015, P(all satisfied and net >= 1000 MW) = 6e-5 (6 samples of 100 000),
  P(no inequality worse than at the nominal) = 0.016, 2.4 violated inequalities per
  sample on average. The three constraints active at the optimum (c24 beta, c35 TF quench
  protection, c83 radial build) are each violated in 50-58 % of samples -- active means
  half of any perturbation crosses it -- and c8 (wall load) in 29 %, c67 (radiation wall
  load) 26 %, c62 (alpha confinement ratio) 16 %, c18 (divertor load) 11 %.

**`ouu`** -- optimisation under uncertainty on the closed MDA: `architectures.ouu`
end to end (`two_stage` -> `make` -> `outer` -> `solve` -> `report`, the evidence
through `evaluate` / `summarise` / `per_sample` / `design_table` / `fresh_sample`),
every run started from PROCESS's converged design. The command line maps onto the
unit: `--pairing` is `kinds.PAIRINGS`, `--table build` holds `kinds.BUILD_LEAVES`
and `--inputs physics` `kinds.ECONOMIC` (together `two_stage`'s `held=`), `--table
old` the 2026-09-16 distributions, `--hfact-sigma` replaces `hfact`'s row;
`--objective`, `--alpha`, `--with-c16 --alpha16`, `--te-recourse --te-range`, `--jac`,
`--chunk` go to `two_stage` / `make`; `--max-iter --move-limit --tol --ftol --gtol`
to the `BoxedSlsqpDriver`. `--no-warm` is gone (the driver warm-starts through the
batched program's memo). The run's JSON keeps the layout `ouu_summary.py` and the
cluster's `out_cluster/ouu2_*.json` have (`robust`, `final`, `deterministic`,
`scipy`, `outer`, `best_feasible`, `trace`, `timing`); `--evidence`, `--sweep-te`,
`--decompose`, `--scaling` / `--measure` and `--plot` as before. One difference to
the cluster runs: the `dummy` row is now drawn in every table (the port keeps it), so
a sample set at the same seed is not the old one.

```bash
G=~/miniconda3/envs/process_port_gpu/bin/python   # conda create -n process_port_gpu python=3.12;
                                                  # pip install -e ~/PROCESS[test] -e ~/jaxgraph[dev,viz];
                                                  # pip install --upgrade "jax[cuda12]"  (last)
export XLA_PYTHON_CLIENT_PREALLOCATE=false        # XLA takes 75 % of VRAM otherwise
$G paper_tests/batching.py --backend cpu          # ~20 min  both shapes, both configurations
$G paper_tests/batching.py --backend gpu          # ~20 min
$G paper_tests/batching.py --backend gpu --precision f32 --sizes 1024
$G paper_tests/batching.py --backend gpu --hardware
$G paper_tests/batching.py --render               # csv + tex from out/batching.json
JAX_PLATFORMS=cuda $G paper_tests/close_conditions.py --batch   # the closed MDA, all four shapes
JAX_PLATFORMS=cuda $G paper_tests/uq.py         # ~2 min  Sobol' + Monte Carlo on the closed MDA
$G paper_tests/uq.py --render                     # tables and figures from the saved samples
```

What it found (2026-09-16, RTX 3080 against the R7 3700X, jax 0.11.1, float64 unless
said):

- **The GPU pays only for batches of thousands, and only on the stellarator.** Per
  point, one MDA is 5.3 ms on the CPU and 43 ms on the GPU (N=1: the graph is ~28k
  scalar ops and the GPU is latency). The CPU's `vmap` saturates at 52-75 us/pt from
  N=1024; the GPU crosses it between N=1024 (0.67x) and 4096 (2.9x) and reaches 8.7 us/pt
  at N=16384 (8.7x), where it stops for memory: N=65536 needs more than the 7.5 GiB
  XLA may take of the card. The SAND block's fused value+Jacobian (14 design entries x
  21 conditions) crosses between N=64 (0.58x) and 256 (1.7x) and is 22.8 us/pt at N=4096
  (14x); the CPU is flat at ~330 us/pt from N=256.
- **The tokamak MDA never crosses: 1.9 ms/pt flat on the GPU against 0.39-0.45 ms/pt on
  the CPU.** The floor is two SVDs per point (`.tokamak.pf_coil.initiation_currents`
  and `.equilibrium_currents`, 1.27 s + 0.65 s of a 1.99 s batched call at N=1024): PROCESS
  pads `gmat` to `LROW1 = 74` rows and cuSOLVER's batched Jacobi SVD stops at 32 rows,
  so XLA calls `gesvd` once per matrix. Measured on the kernel alone: 1350 us/pt for a
  74x4 SVD, 5.7 us/pt for a 32x4 one (`--hardware`). Trim `gmat` to its live rows and
  the tokamak's floor is gone; that is a port change, not a hardware fact. The tokamak
  SAND block (27 x 34) sits on the same floor (2.0 ms/pt from N=256, 0.8x the CPU's
  1.5-1.7 ms/pt) and OOMs at N=4096 (a 5.3 GiB tangent intermediate).
- **The hand-cut SAND block under `vmap` is 2.5x its own unbatched call on the CPU**
  (4.2 ms -> 1.5-1.7 ms/pt on the tokamak) and `architectures.py`'s 5.5 ms/call for the
  same block is that 4.2 ms plus the host wrapper (`phase` timing, `pure_callback`,
  numpy). Its 123 ms/call is the *Jacobi* cut (1193 entries x 66 conditions, 1193
  forward tangents): here 130 ms unbatched, 78 ms/pt under `vmap` (`--recipe jacobi`),
  and on the GPU 25 ms unbatched and 7.2 ms/pt at N=64 -- the one block the GPU wins at
  N=1, because the tangents are already a batch of 1193.
- **GPU and CPU agree to float64 rounding**: at N=64 the max normwise difference is
  4e-16 to 1e-15 and the max elementwise 8e-15 (MDA) to 5e-12 (Jacobian entries). The
  CPU env `process_port` (jax 0.11.0) gives the same figures against the GPU, so the two
  CPU stacks agree with each other.
- **float32 is a hardware number only.** The card's f64/f32 is 26x on a dense matmul
  (0.5 against 13 TFLOP/s) and 3.7x on an elementwise chain; the port's SAND block gains
  1.2x (stellarator) to 1.6x (tokamak, SVD-bound) in f32 because it is not FLOP-bound.
  And the values are wrong: 21-38 % of the block's outputs and 5-6 % of the MDA's are
  inf/nan (a fusion reaction rate is a density squared, ~1e40, past float32's 3.4e38),
  the finite ones deviate by up to 85x elementwise. The f32 MDA rows ran their Picards to
  `max_steps` on nan and are not comparable with anything.
- **Compile** is 1.5-2.7x longer on the GPU for the same program (MDA: 5 vs 8-10 s
  stellarator, 16 vs 23 s tokamak; SAND tokamak 28 s vs 40-76 s, growing with N).

## The cuts

| name | what is cut | where |
|---|---|---|
| `hand` | nine variables measured so that one Picard iterate is one PROCESS pass | `functional_process/cottax/architectures/mda.py` (`CUTS`) |
| `jacobi` | every coupling variable of every component | `functional_process/cottax/architectures/recipes.py` |
| `gauss_seidel` | the backward reads in the graph's binding order (PROCESS's call order) | same |
| `gauss_seidel_minimal` | the backward reads in the order that cuts fewest variables (exact, subset DP) | same |

`recipes.py`'s docstring is the definition; `functional_process/tests/test_architectures.py`
pins every configuration under every arm from cold.
