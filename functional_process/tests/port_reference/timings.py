"""The MDA's timing per configuration: the machine's own MDA (`study.mda_graph` of the
machine alone) at the file's values, compiled once, then the median of `--repeats` warm
runs -- what the paper's `iteration.mda_row` measured before the port moved, so the two
columns compare. `timings/` holds that reference; write a new run elsewhere (`--out`).

One process per configuration, since XLA's CPU JIT runs out of section memory once
enough programs are compiled in one process:

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=<jaxgraph checkout>/src:~/PROCESS-port $PY \\
        -m functional_process.tests.port_reference.timings --out <dir> [--repeats 5]
"""

from __future__ import annotations

import argparse
import csv
import datetime
import platform
import subprocess
import sys
import time
from pathlib import Path

import jax

HERE = Path(__file__).parent


def _timed(fn, *args):
    started = time.perf_counter()
    jax.block_until_ready(fn(*args))
    return time.perf_counter() - started


def one(name: str, repeats: int) -> tuple[dict, str]:
    from functional_process import configurations  # noqa: PLC0415
    from functional_process.cottax import study  # noqa: PLC0415
    from functional_process.cottax.input.native import state_of  # noqa: PLC0415

    configuration = configurations.load(name)
    model = study.cx.declare(configuration.machine)
    graph = study.mda_graph(model)
    values = study.values(graph, state_of(configuration), model.given)
    compile_seconds = _timed(study.solve, graph, values)
    warm = sorted(_timed(study.solve, graph, values) for _ in range(repeats))
    row = {
        "evaluate_ms": 1e3 * warm[len(warm) // 2],
        "serial_us": float("nan"),
        "jacobian_ms": float("nan"),
        "serial_jacobian_us": float("nan"),
        "compile_s": compile_seconds,
        "unknowns": 0,
        "block_nodes": len(graph.nodes),
    }
    return {"configuration": name, **row}, _provenance()


def _provenance() -> str:
    def head(path):
        return subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()

    import cottax  # noqa: PLC0415

    return (f"# timings.py, {datetime.date.today()}, {platform.node()} "
            f"({platform.machine()}), PROCESS {head(HERE)}, "
            f"cottax {head(Path(cottax.__file__).parent)}")


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
