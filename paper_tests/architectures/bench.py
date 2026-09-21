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
from functional_process.cottax.architectures.drivers import SlsqpDriver  # noqa: E402

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

OPTIMISERS = {
    "vmcon": None,                     # the port's VMCON (pyvmcon), the session's default
    "slsqp": SlsqpDriver,              # scipy's SLSQP, exact jacobians
}

TOLERANCE = 1.0e-8
"""VMCON's convergence parameter (`epsvmc` in PROCESS, `VmconDriver.tolerance` in the
port): the same number on both sides."""


def arguments(description: str, *, optimiser: bool = True, batches: bool = False) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--configurations", nargs="*", default=list(NAMES), metavar="NAME")
    p.add_argument("--scheme", choices=sorted(SCHEMES), default="minimal")
    if optimiser:
        p.add_argument("--optimiser", choices=sorted(OPTIMISERS), default="vmcon")
    if batches:
        p.add_argument("--batches", nargs="*", type=int, default=[1, 16, 256, 4096], metavar="N")
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
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "?"


def provenance(script: str) -> str:
    import cottax  # noqa: PLC0415

    return (f"# {script}, {date.today()}, {platform.node()} ({platform.processor() or 'cpu'}), "
            f"PROCESS {head(ROOT)}, cottax {head(Path(cottax.__file__).parents[2])}")


def write(script: str, rows: list[dict], name: str) -> None:
    """`out/<name>/<configuration>.csv`, one file per machine among `rows`: a
    provenance comment, then its rows. A rerun of one machine replaces one file, and a
    run is one process per machine (`run_all.sh`), since XLA's JIT on this box runs
    out of section memory once enough programs have been compiled in one process."""
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    for configuration in dict.fromkeys(r["configuration"] for r in rows):
        mine = [r for r in rows if r["configuration"] == configuration]
        path = folder / f"{configuration}.csv"
        with path.open("w", newline="") as f:
            f.write(provenance(script) + "\n")
            writer = csv.DictWriter(f, fieldnames=list(mine[0]))
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
