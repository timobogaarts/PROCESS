# paper_tests/distributed_architectures

Distributed architectures on `stellarator_helias`, each a recipe of primitive graph
operations on the same base as AAO: the file's problem as `problem_model` declares it.
They follow the three steps of jaxgraph's `examples/sellar_plans.py` (remove, assemble,
add). Inside each of their problems they do what AAO's SAND does to the whole problem:
the same cuts and closures, and the coil's implicit solve absorbed.

- **Sequential**: a magnet stage (`b_t`, `rmajor`, `tdmptf`, `fcutfsu`; the coil
  constraints; minimises `0.2 rmajor` with `b_t ≥ 5`), then a plasma stage (the other
  four variables and constraints; minimises COE). Not equivalent to the file's problem.
- **ALC**: augmented Lagrangian coordination (Tosserams, Etman & Rooda 2008) of build,
  operation and economics, split by design context. There are 43 consistency
  constraints, each discipline is an SLSQP solve, and the coordination is a fixed point
  over the three, run by ALC's own driver (`alc.Coordinator`). Equivalent at its fixed
  point. It departs from the paper's stated algorithm in three choices: the inner loop
  ends when the copies settle (fixed tolerance), the stopping test is stricter, and the
  weights are never raised (β = 1). What it took to converge, and how each choice stands
  against the paper, is in `alc_report.md`.

| file | writes to `out/` | what |
|---|---|---|
| `partition.py` | -- | the reach partition (sequential), the build/operation/economics assignment (ALC), the AAO base every recipe starts from (`base`), and several problems driven and valued as `study` drives and values one |
| `sequential.py` | -- | the sequential recipe; run alone, its result and the refusal of a split whose problems share a cycle |
| `alc.py` | -- | the ALC recipe, its coordination driver, and a run checked against AAO (AAO started at ALC's answer) |
| `benchmark.py` | `benchmark.{csv,md}`, `benchmark_problems.csv` | AAO, sequential and ALC side by side: optimum, constraint violation, overarching-loop and per-problem iterations, cold and warm time (mean of `--runs`) |
| `run_benchmark.sh` | `benchmark.log` | `benchmark.py` with the environment pinned (single-threaded BLAS, as the Snellius jobs) |
| `dsm.py` | `stellarator_helias_{sequential,alc}.html` | the XDSM and N2 of a case, cottax's interactive page |
| `alc_report.md` | -- | the ALC implementation, every change convergence needed, every run |

```bash
cd ~/projects/functional_PROCESS/PROCESS
paper_tests/distributed_architectures/run_benchmark.sh --runs 5     # ~30 min on one CPU, ALC most of it
# one script by hand:
JAX_PLATFORMS=cpu PYTHONPATH=~/projects/jaxgraph/src:.:paper_tests/distributed_architectures \
    ~/anaconda3/envs/func_PROCESS_env/bin/python paper_tests/distributed_architectures/alc.py
```

All three cases use SLSQP at the same settings (`ftol` 1e-8, `max_iter` 500, scaled
coordinates, coupling conditions scaled by the MDA) and start from the file's design.
VMCON is not used, because ALC's unbounded copies take it to a non-finite model.
