# paper_tests/architectures

The architecture table of the PROCESS section: the seven reference machines under
MDF, IDF and SAND, built by `cottax.mdao_architectures` (`Plan(g) + GaussSeidelMinimal()
+ MDF()` and its siblings, through `functional_process.cottax.architectures.session`),
timed on one CPU, and PROCESS itself on the same files at the same tolerance.

| script | one csv in `out/` | what it measures |
|---|---|---|
| `structure.py` | `structure_<scheme>.csv`, `dsm/<machine>_<arm>.html` | nodes, cycles, cut variables, the optimiser's unknowns and conditions, nesting, steps; the DSM of every arm |
| `iteration.py` | `iteration_<scheme>.csv` | one evaluation and one Jacobian of the optimiser's problem, warm, and their compile time; the MDA alone for the `MDA` row |
| `solve.py` | `solve_<scheme>_<optimiser>.csv` | the full solve cold and warm, iterations, status, objective, residuals |
| `native.py` | `native/` | PROCESS: one pass, its idempotence loop, its finite-difference gradient, the full VMCON run at `epsvmc = TOLERANCE` |
| `batched.py` | `batched_<scheme>_<platform>/` | the optimiser's problem over N designs at once (`vmap`, `lax.map`ped over chunks past `--chunk`), per design, CPU or GPU |
| `om_mdf.py` | `openmdao_mdf/` | the same models under OpenMDAO as MDF: a jitted component per model with sparse coloured partials, NLBGS, DirectSolver, SLSQP |
| `table.py` | `table.tex`, `table_batched.tex`, `table_scaling.tex`, `table_openmdao.tex` | the paper's `tabular`s |
| `bench.py` | -- | what they share: the machines, the arms, the schemes, the optimisers, the timer, the writer |

```bash
PY=~/miniconda3/envs/process_port/bin/python
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=~/PROCESS:~/jaxgraph/src
cd ~/PROCESS
paper_tests/architectures/run_all.sh                     # the serial tables, ~1.5 h, one process per machine
paper_tests/architectures/run_batched.sh                 # batched, CPU then GPU (process_port_gpu), ~2 h
JAX_PLATFORMS=cuda ~/miniconda3/envs/process_port_gpu/bin/python paper_tests/architectures/batched.py \
    --configurations stellarator_helias --batches 1 4096 65536 262144 --chunk 4096
$PY paper_tests/architectures/solve.py --optimiser slsqp --configurations helias_5b
$PY paper_tests/architectures/table.py
```

`JAX_ENABLE_X64=1` is not optional: PROCESS is float64 and the Picards diverge in
float32 (the port's own tests get it from a `conftest` side effect). `--scheme` picks
how the cycles are cut (`minimal`, `binding`, `jacobi`); `--optimiser` the driver on
the optimiser (`vmcon`, `slsqp`). Every csv starts with a comment naming the commits
and the machine it was made on.

Two machines (`large_tokamak_eval`, `spherical_tokamak_eval`) state a root find, not
an optimisation: they have an `MDA` row and an `MDF` row (the root find in the graph)
and no IDF or SAND.

Every script is one process per machine (`run_all.sh` loops): XLA's CPU JIT on this box
runs out of section memory once enough programs have been compiled in one process. A
GPU compile of a block holding Newton solves takes minutes (XLA:GPU autotunes and
`ptxas`-compiles every kernel); `jax.config.update("jax_compilation_cache_dir", ...)`
makes it once per machine, and is left off so the `compile_s` column is the real one.
`XLA_PYTHON_CLIENT_PREALLOCATE=false` on the GPU, since the card is shared with the display.
