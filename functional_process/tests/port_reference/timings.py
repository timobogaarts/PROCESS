"""The MDA's timing per configuration, measured exactly as the paper's `iteration.py`
measures its `MDA` row (`bench.open_session`, `iteration.mda_row`: compile once, then the
median of `--repeats` warm runs), written here rather than into `paper_tests/.../out/`.

One process per configuration, as `run_all.sh` does, since XLA's CPU JIT runs out of
section memory once enough programs are compiled in one process:

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=<jaxgraph checkout>/src:~/PROCESS $PY \\
        -m functional_process.tests.port_reference.timings [--repeats 5]
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
PAPER = Path(__file__).resolve().parents[3] / "paper_tests" / "architectures"


def one(name: str, repeats: int) -> dict:
    sys.path.insert(0, str(PAPER))
    import bench  # noqa: PLC0415
    import iteration  # noqa: PLC0415

    args = argparse.Namespace(scheme="minimal", repeats=repeats, epsfcn=None)
    live = bench.open_session(name, args)
    return {"configuration": name, **iteration.mda_row(live, args)}, bench.provenance("timings.py")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--configuration", help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path, default=HERE / "timings")
    args = parser.parse_args()
    if args.configuration:                                   # the child: one machine
        row, provenance = one(args.configuration, args.repeats)
        args.out.mkdir(parents=True, exist_ok=True)
        with (args.out / f"{args.configuration}.csv").open("w", newline="") as f:
            f.write(provenance + "\n")
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        print(row)
        return
    from functional_process import configurations  # noqa: PLC0415

    for name in configurations.NAMES:
        subprocess.run([sys.executable, "-m", "functional_process.tests.port_reference.timings",
                        "--configuration", name, "--repeats", str(args.repeats),
                        "--out", str(args.out)], check=True)


if __name__ == "__main__":
    main()
