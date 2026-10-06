# The architecture tables on Snellius (CPU, and the batched table's H100)

The paper's tables (`paper_tests/architectures/`, `table.py`) measured on one kind of
core. The jobs run from a mirror of `~/PROCESS` at `~/arch_run/PROCESS`, on a cottax
exported at one commit (`pin.sh`), so a later sync of `~/jaxgraph` moves nothing they run.

| file | runs where | what |
|---|---|---|
| `sync_to_cluster.sh` | laptop | mirror `~/PROCESS` (and `~/jaxgraph`, unless `SYNC_SRC`) to the cluster (y / d = dry-run); never sends the cluster's own `out/` or logs |
| `pin.sh` | laptop | export `~/jaxgraph` at `COTTAX` (default HEAD, `src` clean) to `~/cottax_<commit>`, write `pins.env` naming both commits |
| `setup_venv.sh` | login node, once | modules, `$HOME/cottaxvenv`: jax, equinox, optimistix, openmdao pinned; cottax and PROCESS editable |
| `env.sh` | sourced by the jobs | the `module load`s, venv activation, the cache directory |
| `job_pin.sh` | `sbatch` | the port's pin (`test_architectures.py -m tier4`) on the pinned cottax |
| `job_tables.sh` | `sbatch` | the serial tables, one machine per array task: structure, iteration, solve (vmcon, slsqp), native, om_mdf; `SCRIPTS` reruns a subset |
| `job_batched.sh` | `sbatch -p genoa` / `-p gpu_h100` | the batched table |
| `sync_from_cluster.sh` | laptop | pull the cluster's csvs and logs down (additive) |

```bash
# laptop (Snellius takes ssh only from an allowed network: the VPN)
REMOTE_HOME=/home/tbogaarts/arch_run SYNC_SRC=PROCESS bash ~/PROCESS/paper_tests/cluster/sync_to_cluster.sh
REMOTE_HOME=/home/tbogaarts/arch_run bash ~/PROCESS/paper_tests/cluster/pin.sh

# Snellius
cd ~/arch_run/PROCESS/paper_tests/cluster && mkdir -p logs
sbatch job_pin.sh                       # first: the pin passes here as on the laptop
sbatch job_tables.sh
sbatch -p genoa --cpus-per-task=24 job_batched.sh

# laptop
REMOTE_HOME=/home/tbogaarts/arch_run bash ~/PROCESS/paper_tests/cluster/sync_from_cluster.sh
```

Every csv is stamped with the two commits in `pins.env`.

## Environment facts

- Python 3.13.5 (module) against 3.12 locally; cottax needs >= 3.12. `pip install -e
  ~/PROCESS` is a pure-Python hatchling install (`hatch-vcs` reports `0.0.0` without
  `.git`, which the sync excludes). The full `process` package is required: the harness
  runs PROCESS's own VMCON once per input file to fill `~/.cache/functional_process/`.
- The mirror is an rsync copy, not a checkout: edit locally, sync, never edit there.
- Every thread pool is pinned to one thread (`OMP_NUM_THREADS=1` and friends):
  OpenBLAS's reduction order depends on the thread count, and the SQPs see the last bits.
- An existing `cottaxvenv` without OpenMDAO takes `pip install openmdao==3.45.1`;
  `setup_venv.sh` refuses to overwrite a venv.
