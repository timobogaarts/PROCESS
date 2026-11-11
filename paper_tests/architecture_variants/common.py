"""What every script in this folder shares: `../architectures/bench.py` (loaded as
`bench`, so the `architectures` scripts reused here find the same module), the
folder's own `out/`, provenance with the trees' dirty state, the argument parser, and
a writer that keeps one row per variant and replaces its file atomically after every
variant, so a run stopped at any point loses at most the variant it was on.

Named `common`, not `bench`: `../architectures/iteration.py` and `solve.py` are
imported here for their measurement code, and they `import bench` -- theirs.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARCH = HERE.parent / "architectures"
OUT = Path(os.environ.get("VARIANTS_OUT", HERE / "out"))
"""`out/`, or `$VARIANTS_OUT` for a trial run that must not leave rows the sweep would skip."""
ROOT = HERE.parent.parent

SCHEME = "minimal"
"""The cut rule of every variant: `study.RULE`, as the paper table's `--scheme minimal`."""
OPTIMISER = "slsqp"
"""The solves' driver, `SlsqpDriver` (`bench.OPTIMISERS["slsqp"]`)."""


def load(name: str, path: Path):
    """`path` imported as module `name`, once."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(ARCH))           # `eager`, which `architectures/iteration.py` imports
bench = load("bench", ARCH / "bench.py")
sys.path.insert(0, str(HERE))


def _dirty(repo: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return "?"
    return "+dirty" if out.strip() else ""


def provenance(script: str) -> str:
    """`bench.provenance`, each tree marked `+dirty` where it has uncommitted changes."""
    import cottax  # noqa: PLC0415

    cottax_root = Path(cottax.__file__).parents[2]
    line = bench.provenance(script)
    return (line.replace(f"PROCESS {bench.head(ROOT)}", f"PROCESS {bench.head(ROOT)}{_dirty(ROOT)}")
                .replace(f"cottax {bench.head(cottax_root)}", f"cottax {bench.head(cottax_root)}{_dirty(cottax_root)}"))


def arguments(description: str, *, starts: bool = False) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configurations", nargs="*", default=list(bench.NAMES), metavar="NAME")
    p.add_argument("--variants", nargs="*", default=None, metavar="LABEL",
                   help="labels to run (default: every variant of each configuration)")
    p.add_argument("--repeats", type=int, default=5, help="warm repeats to take the median of")
    p.add_argument("--force", action="store_true", help="rerun variants that already have a row")
    if starts:
        p.add_argument("--starts", type=int, default=100, help="random starts per variant")
        p.add_argument("--spread", type=float, default=0.1,
                       help="each design variable started at x0 * U(1 - spread, 1 + spread), clipped to its bounds")
    return p.parse_args()


# ---------------------------------------------------------------- one row per variant


def shown(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def path_of(folder: str, configuration: str) -> Path:
    return OUT / folder / f"{configuration}.csv"


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def done(folder: str, configuration: str, label: str) -> bool:
    return any(r.get("label") == label for r in read_rows(path_of(folder, configuration)))


def _replace(path: Path, header: str, rows: list[dict]) -> None:
    """`rows` to `path` through a temporary file in the same folder and `os.replace`,
    flushed and synced: the file is the old one or the new one, never half of either."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(k for r in rows for k in r))
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", newline="") as f:
        f.write(header + "\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def upsert(script: str, folder: str, row: dict) -> Path:
    """`row` into `out/<folder>/<configuration>.csv`, replacing the row of the same
    variant if there is one; every row carries where and when it was made, since rows
    of one file can come from different runs."""
    stamp = {"made": f"{date.today()} {provenance(script).split(', ', 2)[2]}"}
    row = {**row, **stamp}
    path = path_of(folder, row["configuration"])
    kept = [r for r in read_rows(path) if r.get("label") != row["label"]]
    _replace(path, provenance(script), kept + [row])
    print(f"wrote {shown(path)} ({len(kept) + 1} rows)", flush=True)
    return path


def write_all(script: str, path: Path, rows: list[dict]) -> None:
    """A whole file of rows (one variant's starts), replaced atomically."""
    _replace(path, provenance(script), rows)
    print(f"wrote {shown(path)} ({len(rows)} rows)", flush=True)
