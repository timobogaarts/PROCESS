"""What every script in this folder shares: the machines, the arms, the schemes, the
optimisers, one timer, one csv writer.

Run any script as a file, from the repo root, with the environment set (see README):

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=~/PROCESS:~/jaxgraph/src \\
        $PY paper_tests/architectures/solve.py --scheme minimal --optimiser vmcon

`JAX_PLATFORMS` is the caller's: `cpu` for every table but the batched one, which is
run twice, `cpu` and `cuda`.

Every script writes one csv to `out/`, first line a comment saying which commits and
machine produced it, so a table built from these files says where its numbers came from.
"""

from __future__ import annotations

import argparse
import csv
import functools
import os
import platform
import subprocess
import time
from datetime import date
from pathlib import Path

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)  # PROCESS is float64; the Picards NaN without it

from cottax.mdao_architectures import GaussSeidel, GaussSeidelMinimal, Jacobi  # noqa: E402

from functional_process import configurations  # noqa: E402
from functional_process.cottax.architectures import session  # noqa: E402
from functional_process.cottax.architectures.drivers import SlsqpDriver, VmconDriver  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
ROOT = HERE.parent.parent

NAMES = tuple(configurations.NAMES)
"""The seven reference machines: five optimisations and two evaluations (root finds)."""

ARMS = ("MDA", "MDF", "IDF", "SAND")

SCHEMES = {
    "minimal": GaussSeidelMinimal(),   # the default: fewest cut variables, exactly
    "binding": GaussSeidel(),          # PROCESS's own call order
    "jacobi": Jacobi(),                # every coupling
}

PROCESS_EPSFCN = 1.0e-3
"""PROCESS's own relative finite-difference perturbation (`data.numerics.epsfcn`,
`configurations/defaults.py`), the step `Evaluators.fcnvmc2` differentiates with."""

OPTIMISERS = {
    "vmcon": None,                     # the port's VMCON (pyvmcon), the session's default
    "slsqp": SlsqpDriver,              # scipy's SLSQP, exact jacobians
    # The same driver on the same block, differentiated the way PROCESS differentiates:
    # `x * (1 +/- epsfcn)` per coordinate, `2n` evaluations per iteration. The only
    # difference from the `vmcon` row is where the derivative comes from, so the two
    # together separate "a declared architecture" from "an autodiff-visible one".
    "vmcon-fd": functools.partial(VmconDriver, epsfcn=PROCESS_EPSFCN),
    "slsqp-fd": functools.partial(SlsqpDriver, epsfcn=PROCESS_EPSFCN),
}

TOLERANCE = 1.0e-8
"""VMCON's convergence parameter (`epsvmc` in PROCESS, `VmconDriver.tolerance` in the
port): the same number on both sides."""


def arguments(description: str, *, optimiser: bool = True, batches: bool = False,
              epsfcn: bool = False, paper: bool = False) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--configurations", nargs="*", default=list(NAMES), metavar="NAME")
    p.add_argument("--scheme", choices=sorted(SCHEMES), default="minimal")
    if optimiser:
        p.add_argument("--optimiser", choices=sorted(OPTIMISERS), default="vmcon")
    if epsfcn:
        p.add_argument("--epsfcn", nargs="?", type=float, const=PROCESS_EPSFCN, default=None,
                       metavar="STEP", help="also measure PROCESS's finite-difference "
                       f"Jacobian at this relative step (bare flag: {PROCESS_EPSFCN:g})")
    if batches:
        p.add_argument("--batches", nargs="*", type=int, default=[1, 16, 256, 4096], metavar="N")
        p.add_argument("--chunk", type=int, default=4096, help="designs per vmap; larger batches are lax.map'ed over chunks")
    if batches:
        p.add_argument("--arms", nargs="*", default=None, metavar="ARM",
                       help="only these arms (default: every optimising arm); the rows are "
                       "merged into the machine's csv, so one process per arm adds up")
    if paper:
        p.add_argument("--paper", action="store_true", help="also write the paper's copy of each table")
    p.add_argument("--once", action="store_true", help=argparse.SUPPRESS)  # om_mdf.py's
    p.add_argument("--repeats", type=int, default=5, help="warm repeats to take the median of")
    return p.parse_args()


def open_session(name: str, args) -> session.Session:
    optimiser = OPTIMISERS[getattr(args, "optimiser", "vmcon")]
    return session.open_session(name, optimiser=optimiser, scheme=SCHEMES[args.scheme])


def timed(fn, *a, **kw):
    """`(result, seconds)`, wall clock, jax results blocked on."""
    began = time.perf_counter()
    result = fn(*a, **kw)
    jax.block_until_ready(result) if _is_jax(result) else None
    return result, time.perf_counter() - began


def _is_jax(x) -> bool:
    try:
        jax.tree_util.tree_leaves(x)
        return True
    except Exception:  # noqa: BLE001
        return False


def median_seconds(fn, repeats: int) -> float:
    """The median of `repeats` warm calls of `fn()`."""
    times = sorted(timed(fn)[1] for _ in range(repeats))
    return times[len(times) // 2]


def head(repo: Path) -> str:
    """`repo`'s short commit; on the cluster, where the trees are rsync mirrors without
    their `.git`, the job states it instead (`COMMIT_PROCESS`, `COMMIT_COTTAX`, written
    by `cluster/pin.sh` from the laptop)."""
    stated = os.environ.get("COMMIT_COTTAX" if "jaxgraph" in str(repo) or "cottax" in str(repo) else "COMMIT_PROCESS")
    if stated:
        return stated
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "?"


def provenance(script: str) -> str:
    import cottax  # noqa: PLC0415

    return (f"# {script}, {date.today()}, {platform.node()} ({platform.processor() or 'cpu'}), "
            f"PROCESS {head(ROOT)}, cottax {head(Path(cottax.__file__).parents[2])}")


def write(script: str, rows: list[dict], name: str, merge_on: tuple[str, ...] = ()) -> None:
    """`out/<name>/<configuration>.csv`, one file per machine among `rows`: a
    provenance comment, then its rows. A rerun of one machine replaces one file, and a
    run is one process per machine (`run_all.sh`), since XLA's JIT on this box runs
    out of section memory once enough programs have been compiled in one process."""
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    for configuration in dict.fromkeys(r["configuration"] for r in rows):
        mine = [r for r in rows if r["configuration"] == configuration]
        path = folder / f"{configuration}.csv"
        if merge_on and path.exists():
            # Keep the rows this run did not produce (another arm's, from another
            # process), so a machine split over processes adds up to one file. The
            # provenance line is this run's.
            ran = {tuple(str(r[k]) for k in merge_on) for r in mine}
            with path.open() as f:
                kept = [r for r in csv.DictReader(line for line in f if not line.startswith("#"))
                        if tuple(r[k] for k in merge_on) not in ran]
            mine = kept + mine
        with path.open("w", newline="") as f:
            f.write(provenance(script) + "\n")
            writer = csv.DictWriter(f, fieldnames=list(mine[-1]))
            writer.writeheader()
            writer.writerows(mine)
        print(f"wrote {path.relative_to(ROOT)} ({len(mine)} rows)")


def read(name: str) -> list[dict]:
    """Every machine's rows of `out/<name>/`, in `NAMES` order."""
    rows = []
    for configuration in NAMES:
        path = OUT / name / f"{configuration}.csv"
        if not path.exists():
            continue
        with path.open() as f:
            lines = [line for line in f if not line.startswith("#")]
        rows += list(csv.DictReader(lines))
    return rows
