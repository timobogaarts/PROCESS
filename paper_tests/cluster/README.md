# Running the batching studies on Snellius (H100)

`batching.py` and `close_conditions.py --batch` on one `gpu_h100` node, in a venv of
their own (`$HOME/cottaxvenv`, never `jaxvenv`). Everything here is self-contained;
nothing else in `~/PROCESS` or `~/jaxgraph` is touched.

| file | runs where | what |
|---|---|---|
| `sync_to_cluster.sh` | laptop | mirror `~/jaxgraph` + `~/PROCESS` to `snellius:/home/tbogaarts/` (y / d = dry-run) |
| `setup_venv.sh` | Snellius login node, once | modules, `$HOME/cottaxvenv`, pinned jax + CUDA 12, editable cottax`[solvers]` and PROCESS, smoke check |
| `env.sh` | sourced by the others | the three `module load`s, venv activation, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `FP_HARNESS_CACHE_DIR` |
| `job_batching.sh` | `sbatch` | CPU ladder (reference arrays) -> GPU ladder to N=262144 -> `--hardware` -> `--render` -> `plot_batching.py` |
| `job_closed.sh` | `sbatch` | `close_conditions.py --batch` on the GPU, sizes to 262144 |
| `run_batching.py` | inside the job | `batching.py` unmodified, plus `BATCHING_WALL_LIMIT` and a `host`/`cpu` stamp on every row |
| `sync_from_cluster.sh` | laptop | pull `out/` and the logs into `~/PROCESS/paper_tests/out_cluster/` (additive) |

## Sequence

```bash
# laptop
bash ~/PROCESS/paper_tests/cluster/sync_to_cluster.sh          # d first, then y

# Snellius (ssh snellius.surf.nl)
bash ~/PROCESS/paper_tests/cluster/setup_venv.sh                # once; ~10 min; ends with the smoke check
# optional: the GPU smoke check, 5 min of a shared H100
srun -p gpu_h100 --gpus=1 --cpus-per-task=4 -t 00:05:00 bash -c \
  'source ~/PROCESS/paper_tests/cluster/env.sh && python -c "import jax; print(jax.devices())"'
cd ~/PROCESS/paper_tests/cluster && mkdir -p logs
sbatch job_batching.sh                                          # 8 h limit; expect 1.5-3 h
sbatch job_closed.sh                                            # 4 h limit; expect < 1 h
squeue -u $USER;  tail -f logs/batching_h100.log                # one line per (shape, config, N)

# laptop, when both jobs are done
bash ~/PROCESS/paper_tests/cluster/sync_from_cluster.sh         # -> paper_tests/out_cluster/
```

Re-syncing code later: `sync_to_cluster.sh` again (the venv's installs are editable, no
reinstall); it never touches the remote `paper_tests/out/` or `cluster/logs/`.

Knobs, as environment on the `sbatch` line: `SIZES=1,4,16 sbatch job_batching.sh`
(GPU ladder; `mda` needs squares), `CPU_SIZES` (default the script's own ladder to
4096; `64,1024` is the minimum the diff cache needs), `REPEATS` (3), `BATCHING_WALL_LIMIT`
(900 s; `batching.py`'s constant is 60 s, which would end the tokamak MDA ladder near
N=65536 at 1.9 ms/pt -- the wrapper lifts it, the script is not edited), `SHAPES` /
`INPUTS` (passed through as `--shape` / `--input`). `job_closed.sh` takes `SIZES`.

## What comes back (`paper_tests/out_cluster/`)

| file | from | content |
|---|---|---|
| `batching.json` | `batching.py` | every row: shape, configuration, backend, precision, N, first/warm wall, us/pt, peak device MB and RSS, `*_vs_cpu_f64` diffs at N=64/1024; `device` (`NVIDIA H100 ...` / `cpu`), plus `host` and `cpu` from the wrapper |
| `batching.csv`, `batching.tex` | `--render` | the same as a table; the `.tex` header line names the machine |
| `batching.png`, `batching_wall.png` | `plot_batching.py` | us/pt and wall per call. **The legend labels are hard-coded** ("Ryzen 7 3700X", "RTX 3080") in `plot_batching.py`: change them before using the H100 figures |
| `close_conditions_batch_gpu.{csv,tex}` | `close_conditions.py --batch` | the four shapes x N; rows carry `backend` only, no device or host -- they are H100 rows only because they sit in `out_cluster/` |
| `logs/batching_h100.log`, `logs/closed_h100.log` | the jobs | python stdout/stderr, one `==` line per step with time and exit code |
| `logs/<job>-<id>.out` | slurm | the job's own stdout (normally empty) |

Keep the two `out/` directories apart. `batching.py` merges new rows into an existing
`out/batching.json` on `(shape, configuration, cut, backend, precision, tag, N, vmap)` --
no device, no host -- so H100 rows in the laptop's `out/` would silently replace the
RTX 3080 rows. That is why `sync_to_cluster.sh` excludes `paper_tests/out/` (the
cluster starts from an empty JSON) and `sync_from_cluster.sh` writes to `out_cluster/`.
Merging both into one table needs `device` (and a host for the CPU rows) in
`batching._key` -- a change to `batching.py`, not made here.

## Environment facts

- Python 3.13.5 (module) against 3.12 locally; cottax needs >= 3.12, PROCESS's test
  matrix includes 3.13. Pins: `jax[cuda12]==0.11.1` (jaxlib and the CUDA 12 plugin follow;
  the CUDA 12.9 module is loaded but the wheels bundle their own libraries), `equinox==0.13.8`,
  `optimistix==0.1.0`, `lineax==0.1.1` -- the `process_port_gpu` versions of 2026-09-17.
  Override with `JAX_VERSION=... bash setup_venv.sh`.
- `pip install -e ~/PROCESS` is a pure-Python hatchling install: no cmake, no gfortran, no
  `--no-build-isolation`, no extras. `hatch-vcs` reports version `0.0.0` without `.git`
  (excluded from the sync), and the `.tex` provenance line says `PROCESS unknown` for the
  same reason. The full `process` package is required: `functional_process` imports
  `process.core.model` and the model modules everywhere, and the harness runs PROCESS's
  own VMCON (`process.main.SingleRun`) once per input file to fill
  `~/.cache/functional_process/` -- that happens inside the first job (a minute or two per
  file, on the CPU). `pytest` is installed separately: `_harness/__init__.py` imports it.
- The remote `~/jaxvenv` resolves `cottax` editable from `~/jaxgraph`, so the mirror moves
  the cottax that venv sees too.
- Snellius shares H100 nodes: 16 cores, 1 GPU (80 GB) and 180 GB RAM per job here.
