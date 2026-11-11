# paper_tests/architecture_variants

The architecture space between the presets: on each of the seven reference machines,
every combination of, per coupling block on the optimiser's cycle, an MDA of its own
(`F`, as MDF) or the optimiser (`O`, as IDF/SAND), and, per implicit model on that
cycle, a local solve (`L`, as MDF/IDF) or the optimiser (`O`, as SAND). 44 variants,
13 of them the presets the paper table already runs.

The variants are one cottax architecture, `variants.Hybrid`, a final of cottax's
`Monolithic` beside `MDF`, `IDF` and `SAND`: it shares all their stages and decides
per cycle and per implicit model what they decide for all at once, with nothing but
`Combine` and `Nest`. cottax itself is not changed. At the corners it produces the
presets' graphs exactly; `structure.py` checks that before anything is timed.

| script | writes to `out/` | what it measures |
|---|---|---|
| `structure.py` | `structure_minimal/<machine>.csv`, `variants.txt` | per variant: blocks, nodes, problems, nesting, the optimiser's unknowns and conditions, refusals, and at the corners whether the graph is the preset's |
| `iteration.py` | `iteration_minimal/<machine>.csv` | one evaluation and one Jacobian of the optimiser's problem, warm, and their compile time: `../architectures/iteration.py`'s `optimiser_row` (without the eager columns) |
| `solve.py` | `solve_minimal_slsqp/<machine>.csv`, `starts_minimal_slsqp/<machine>/<label>.csv` | the SLSQP solve from the file's start, cold and warm, with the optimiser's model calls (`../architectures/solve.py`'s `CALLS`); then 100 random starts and their success rate |
| `dsm.py` | `dsm/<machine>_<label>.html` | the XDSM and N2 of chosen variants, cottax's interactive page, solves ringed; built, not solved |
| `table.py` | `summary.csv`, `comparison_snellius.csv` | one row per variant; the corners beside the Snellius rows of `../architectures/out` |
| `variants.py` | -- | `Hybrid`, the switches of a machine, the variants, and `study.block_plan` / `study.assemble` for a variant |
| `common.py` | -- | `../architectures/bench.py` (as `bench`), provenance with dirty state, the parser, the atomic per-variant writer |
| `run_all.sh` | `run.log` | the sweep: structure, then per variant `iteration.py` and `solve.py`, one process each, resumable |

```bash
cd ~/projects/functional_PROCESS/PROCESS
paper_tests/architecture_variants/run_all.sh          # ~1.5-2.5 h on one CPU, resumes where it stopped
paper_tests/architecture_variants/table.py            # with the env below
# one variant by hand:
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=~/projects/jaxgraph/src \
    ~/anaconda3/envs/func_PROCESS_env/bin/python paper_tests/architecture_variants/solve.py \
    --configurations helias_5b --variants FO --starts 3
```

`VARIANTS_OUT=<dir>` writes everything there instead of `out/`: for a trial run that
must not leave rows the sweep would then skip.

## Labels

One letter per block in binding order, then `-` and one letter per switchable implicit
model. `structure_minimal/*.csv` names the blocks (`blocks`, the closure each MDA is
named after) and the implicit models (`implicit`) in that order.

| machine | blocks | implicit | variants | presets |
|---|---|---|---|---|
| `stellarator_helias` | 2 | 1 (coil intersect) | 8 | `FF-L` MDF, `OO-L` IDF, `OO-O` SAND |
| `helias_5b` | 2 | 0 (its implicit model is off the optimiser's cycle) | 4 | `FF` MDF, `OO` IDF = SAND |
| `low_aspect_ratio_DEMO` | 3 | 0 | 8 | `FFF` MDF, `OOO` IDF = SAND |
| `large_tokamak_nof` | 4 | 0 | 16 | `FFFF` MDF, `OOOO` IDF = SAND |
| `st_regression` | 2 | 0 | 4 | `FF` MDF, `OO` IDF = SAND |
| `large_tokamak_eval` | 1 on the top's cycle | 0 | 2 | `F` MDF; `O` combines the block into the top's Newton |
| `spherical_tokamak_eval` | 1 | 0 | 2 | `F` MDF; `O` likewise |

## The random starts

Per machine one generator, seeded by the machine's name (`zlib.crc32`), draws for each
of `--starts` starts a factor `U(1 - spread, 1 + spread)` per design variable
(`--spread 0.1`, so ±10%); the start is the file's value times the factor, clipped to the
variable's bounds (a root find's unknowns have none). Every variant of a machine gets
the same starts. The MDA is solved at each start (the same compiled MDA as the file's)
and the solve's inputs are built from it as `study.block_values` builds them for the
file's start: the coupling copies start consistent with the design. A start where the
MDA raises or leaves an input of the solve not finite is `invalid` and is in no rate.

The variant's graph, and its condition scaling, are the file start's: a start changes
array inputs only, so it reuses the compiled solve (each start costs about one warm
solve). `solve.py` first checks that the machinery at factor 1 gives back exactly the
file start's inputs.

`success` = SLSQP converged and the objective within `1e-3` (relative) of the paper
table's MDF SLSQP objective on that machine; for a root find, converged and the
unknowns within `1e-3` of the variant's own answer from the file's start. Every
start's objective, residuals, start and answer are in its csv.

## SLSQP's exit, and the inner work

`exit_code` / `exit_message` are scipy's SLSQP exit mode (`SlsqpDriver` reports
`result.status` as its `Status`; `study.answer` drops it, so `solve.py` reads it off the
run): e.g. 4 incompatible inequality constraints, 6 singular matrix C, 8 positive
directional derivative in the line search, 9 iteration limit. Per start, and tallied
over the starts in `starts_exit_codes`.

`values` / `jacobians` count the optimiser's calls into the model, one each however much
the nested solves inside a call iterate, so they favour `F` (a block's MDA runs inside
every call). `InnerCounter` counts that work afterwards, untimed: every point the
optimiser called the model at is recorded on the host (`RECORD`, outside `CALLS`' timing;
the compiled program is not touched), then re-evaluated on a copy of the variant's graph
whose nested problems also report `Steps` (optimistix's `num_steps`). The same program
returns the conditions, checked against the values the timed call returned
(`count_check`, the largest relative difference). Columns:

- `steps:<kind>:<problem>`, `inner_steps` (all calls), `inner_steps_values` (value calls
  only): nested solver steps summed over the solve's calls;
- `model_evals` (`_values`): per call, every model on the top's cycle once (`top_models`)
  and each nested problem's own models (`inner_sizes`) once per step instead, summed. A
  machine-independent measure of the work, Jacobian calls counted as one primal pass
  (their tangents are linear solves, not model evaluations);
- `starts_inner_steps_median`, `starts_model_evals_median`: over the successful starts.

A root find (the two evaluation machines) has no host-side calls: its Newton runs
inside the graph, so it has no inner columns. The counting adds about one eval-only
compile per variant and ~2 ms per distinct point (`count_s`).

## How this differs from the paper table's runs (`../architectures`, Snellius)

- **One process per variant**, for `iteration.py` and `solve.py` alike: every
  `compile_s`, `cold_s` and `assemble_s` here is cold. There it was one process per
  machine and script, all arms in it: SAND's `cold_s` was a compile-cache hit on IDF's
  (same graph), and `assemble_s` did not include the MDA's compile (the MDA arm ran
  first).
- **No eager columns** in `iteration.py` (`eager_walk_ms`, `eager_ms`).
- **Code**: whatever `HEAD` is (every row says which, `+dirty` if the tree was);
  the Snellius rows are cottax `8ba473d` / PROCESS `a69d6abd` (solve) and cottax
  `a30eb62` / PROCESS `49acbb58` (iteration).
- **The random starts and the success rate** are new. The file's start is the paper's.
- **BLAS threads**: `run_all.sh` sets `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1`, as the
  Snellius job does (`../cluster/job_tables.sh`). The first sweep (2026-11-10) ran without
  it; on every machine but `st_regression` that changed nothing, but scipy's OpenBLAS
  (which SLSQP calls) takes last-bit different paths at 1 and at 2+ threads.
- **`st_regression`'s counts are not reproducible numbers.** Its objective (fusion gain)
  has zero gradient in five inboard build variables, so its optimum is a flat face, not
  a point: IDF/SAND's SLSQP count spans ~70-170 under last-bit changes (ulp moves of the
  start, BLAS threads or kernel, the order of unknowns or constraints), MDF's ~10-15. The
  same code gives 11/81 on Snellius and 13/102 (multi-threaded BLAS) or 11/144 (one
  thread) here. A difference of that size between variants on this machine is not an
  architecture effect. (Investigation of 2026-11-10; the scripts and dumps were left in
  that session's scratchpad.)
- Unchanged: the `minimal` rule (`study.RULE`), `SlsqpDriver` with the condition scaling
  from the file start's MDA, `MAX_ITER = 500`, Picard/Newton by `study.driven`, five warm
  repeats, `JAX_ENABLE_X64`, VMCON as the (never run) top of `iteration.py`'s assembly.
  Where IDF and SAND coincide, the corner is valued as SAND (`nested_given` off), the
  row the paper prints; the two rows' iterations agree on every such machine there.
