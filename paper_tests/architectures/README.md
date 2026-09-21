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
| `native.py` | `native.csv` | PROCESS: one pass, its idempotence loop, its finite-difference gradient, the full VMCON run at `epsvmc = TOLERANCE` |
| `table.py` | `table.tex` | the paper's `tabular` from the four |
| `bench.py` | -- | what they share: the machines, the arms, the schemes, the optimisers, the timer, the writer |

```bash
PY=~/miniconda3/envs/process_port/bin/python
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=~/PROCESS:~/jaxgraph/src
cd ~/PROCESS
paper_tests/architectures/run_all.sh                     # everything, ~1-2 h; log in out/run_all.log
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
