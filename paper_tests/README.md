# paper_tests

Generators for the PROCESS section of *Managing complexity for integrated design in
fusion using a graph-based functional framework* (`~/graph_paper`). Each script measures
one thing over the seven reference input files and writes it to `out/` as a `.csv`
(numbers), a `.tex` fragment (a `tabular`, for `\input`) and, where the raw rows are
worth keeping, a `.json`. Every fragment's first line says which script, tree, cottax
commit and machine produced it.

```bash
PY=~/miniconda3/envs/process_port/bin/python      # see ../CLAUDE.md for the env
export JAX_PLATFORMS=cpu
$PY paper_tests/graph_census.py        # ~1 min   structure: nodes, SCCs, cuts, body depth
$PY paper_tests/timings.py             # instant  renders the three reference matrices
$PY paper_tests/mda_convergence.py     # ~5 min   Picard steps per block under each cut
$PY paper_tests/architectures.py       # ~1 h     MDF/SAND x cut x optimiser, warm and cold
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

**`timings`** -- one row per configuration and arm, under VMCON and SLSQP: trace+lower,
compile, cold total, SQP iterations, warm wall, model ms/call. Read straight off
`functional_process/cottax/reference_*_matrix.txt`; regenerate those first
(`functional_process/_audit/performance.md`).

**`mda_convergence`** -- the inner analysis alone, from one cold state
(`sand_harness.cold_state`: the graph after one pass in call order, which is where
PROCESS starts too): Picard steps per coupled block, their sum, the sum weighted by body
depth (sequential node evaluations), and the warm wall of one converged MDA.

**`architectures`** -- MDF and SAND assembled on each cut and answered by each optimiser:
iterations, verdict, objective, design entries (a Jacobi SAND carries whole profiles),
cold wall and warm wall. One `tabular` per optimiser.

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
inside the MDA instead of handed to the optimiser (`--table` ranks every pairing by the
cycle it closes and its sensitivity; `--solve` runs the 6-variable optimisation; `--batch`
vmaps the MDA). Four batch shapes: `plain` (the hand-cut MDA), `nested` (root finds with
the cycle's Picards nested inside, exact Newton), `closed` (the c16 root find `Residualise`d
and `Combine`d with those Picards into one 4-unknown square Newton with a Broyden update --
no loop inside the loop; the default), and `predicted` (`closed` started from a first-order
predictor across the batch). `out/batching_closed*.png` plot them. On the GPU at N=4096:
plain 83 ms, nested 196, closed 137, predicted 107 -- a converged, equality-feasible
evaluation for 1.3-1.7x a plain one. Two lessons that transfer: a `while_loop` nested in a
vmapped `while_loop` must mask its predicate by the outer activity, and on this card a
forward tangent through FP64 transcendentals costs a primal, so a Jacobian recomputed
every step loses to a secant update.

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
| `hand` | nine variables measured so that one Picard iterate is one PROCESS pass | `functional_process/cottax/mda.py` (`CUTS`) |
| `jacobi` | every coupling variable of every component | `functional_process/cottax/recipes.py` |
| `gauss_seidel` | the backward reads in the graph's binding order (PROCESS's call order) | same |
| `gauss_seidel_minimal` | the backward reads in the order that cuts fewest variables (exact, subset DP) | same |

`recipes.py`'s docstring is the definition; `functional_process/tests/test_recipes.py`
pins the census on the stellarator and that every recipe reaches the hand cut's fixed
point and optimum.
