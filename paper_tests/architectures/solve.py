"""Table 3 -- the full solve of every arm on every machine, from the file's own values:
wall time cold (assembly, compilation and the solve) and warm (the same solve again,
everything compiled), the optimiser's iterations, and where it ended.

    $PY paper_tests/architectures/solve.py [--scheme minimal] [--optimiser vmcon|slsqp]

`status` is the driver's own verdict (`converged`, `stopped`, ...); `objf`, `max_eq`,
`min_ie` are the objective, the largest equality residual and the least inequality
slack at the answer, as the driver reports them. An evaluation machine's `MDF` is its
root find; its `MDA` is the analysis at the file's design, never iterated.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402


def main():
    args = bench.arguments(__doc__)
    rows = []
    for name in args.configurations:
        live = bench.open_session(name, args)
        for arm in live.arms:
            _, assemble_s = bench.timed(live.assemble, arm)
            cold, cold_s = bench.timed(live.solve, arm)      # compiles, then solves
            warm, warm_s = bench.timed(live.solve, arm)      # the same solve, compiled
            rows.append({
                "configuration": name,
                "arm": arm,
                "optimiser": args.optimiser,
                "status": warm["status"],
                "iterations": warm["iterations"],
                "objf": warm["objf"],
                "max_eq": warm["max_eq"],
                "min_ie": warm["min_ie"],
                "assemble_s": assemble_s,
                "cold_s": cold_s,
                "warm_s": warm_s,
                "solve_s": warm["seconds"],           # the driver's own clock, warm
                "note": warm.get("note", ""),
            })
            print({k: v for k, v in rows[-1].items() if k != "note"})
    bench.write("solve.py", rows, f"solve_{args.scheme}_{args.optimiser}")


if __name__ == "__main__":
    main()
