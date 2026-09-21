# Running the flexibility analysis on Snellius (CPU)

`flexibility.py` at N = 262144, three hfact sigmas, on one exclusive `rome` node each:
96 workers, ~6 min a task. Everything here is self-contained; nothing in `~/PROCESS` or
`~/jaxgraph` is touched by the sync beyond mirroring it.

| file | runs where | what |
|---|---|---|
| `sync_to_cluster.sh` | laptop | mirror `~/jaxgraph` + `~/PROCESS` to `snellius:/home/tbogaarts/` (y / d = dry-run); never sends `paper_tests/out/` |
| `setup_venv.sh` | Snellius login node, once | modules, `$HOME/cottaxvenv`, editable cottax and PROCESS |
| `env.sh` | sourced by the job | the `module load`s, venv activation, the cache directory |
| `job_flex.sh` | `sbatch` | the array job: task i = sigma 0.10 / 0.05 / 0.02; `ORDER`, `REQUIRE_NET`, `N`, `WORKERS` as environment |
| `sync_from_cluster.sh` | laptop | pull the cluster's `paper_tests/out/` down (additive, into `out/`; the logs into `cluster/logs/`) |

```bash
# laptop
bash ~/PROCESS/paper_tests/cluster/sync_to_cluster.sh                # d first, then y

# Snellius (ssh snellius.surf.nl); the port tracks cottax 6b1d540 and the mirror's
# ~/jaxgraph has moved past it, so the job puts a read-only export first on PYTHONPATH
mkdir -p ~/cottax_head && git -C ~/jaxgraph archive 6b1d540 src | tar -x -C ~/cottax_head
bash ~/PROCESS/paper_tests/cluster/setup_venv.sh                     # once
cd ~/PROCESS/paper_tests/cluster && mkdir -p logs
sbatch --export=ALL,ORDER=hfact job_flex.sh                          # can it be run
sbatch --export=ALL,ORDER=hfact,REQUIRE_NET=982.4 job_flex.sh        # at rated power
squeue -u $USER; tail -f logs/flex_n262144_s0.10_hfact.log
```

Results land in the cluster's `paper_tests/out/` as `flex_n<N>_s<sigma>_hfact[_req982.4]
.json` (one row per draw, ~10 MB); `flexibility.py --figures` condenses them to the
`.npz` + `_summary.json` pairs that are tracked, and draws the figures.

## Environment facts

- Python 3.13.5 (module) against 3.12 locally; cottax needs >= 3.12. `pip install -e
  ~/PROCESS` is a pure-Python hatchling install (`hatch-vcs` reports `0.0.0` without
  `.git`, which is excluded from the sync). The full `process` package is required:
  `common.process_reference` runs PROCESS's own VMCON once to fill
  `~/.cache/functional_process/` (a minute or two, on the CPU, inside the first job).
- `~/PROCESS` on the cluster is an rsync mirror, not a checkout: edit locally, sync,
  never edit there.
- The job pins every thread pool to one thread (`OMP_NUM_THREADS=1` and friends): 96
  workers each spawning 128 OpenBLAS threads thrash the node, and OpenBLAS's reduction
  order depends on the thread count, which SLSQP's iterates can see.
- 96 workers rather than 128: a worker's peak RSS is 2.25 GB and the node has 224 GB.
