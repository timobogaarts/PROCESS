"""Table 3 -- the full solve of every arm on every machine, from the file's own values:
wall time cold (assembly, compilation and the solve) and warm (the same solve again,
everything compiled), the optimiser's iterations, and where it ended.

    $PY paper_tests/architectures/solve.py [--scheme minimal] [--optimiser vmcon|slsqp]

`warm_s` is the median of `--repeats` warm solves. On the last of them every call the
optimiser makes into the model is counted and timed on the host (`model_calls`): the
values, the Jacobians and the fused value-and-Jacobian programs, and the seconds spent
inside them (`model_s`, dispatch included). What is left of the solve is the driver --
its Python loop, its QP, scipy's -- and with the in-program times of `iteration.py` the
counts say what the same solve would cost with the model alone.

`status` is the driver's own verdict (`converged`, `stopped`, ...); `objf`, `max_eq`,
`min_ie` are the objective, the largest equality residual and the least inequality
slack at the answer, as the driver reports them. An evaluation machine's `MDF` is its
root find; its `MDA` is the analysis at the file's design, never iterated.
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402

from functional_process.cottax.architectures import drivers  # noqa: E402

CALLS: Counter = Counter()
"""Model calls and seconds inside them, per kind, since the last `CALLS.clear()`."""


def _counting(scaled_problem):
    """`drivers.scaled_problem` with its value, Jacobian and fused programs counted and
    timed: both SQP drivers reach the model through it and through nothing else, so the
    counts are every model call the optimiser made."""
    def wrap(kind, fn):
        def call(x):
            began = time.perf_counter()
            out = fn(x)
            CALLS[f"{kind}_s"] += time.perf_counter() - began
            CALLS[kind] += 1
            return out
        return call

    def counted(*a, **kw):
        values, jacobian, both, *rest = scaled_problem(*a, **kw)
        return (wrap("values", values), wrap("jacobians", jacobian), wrap("fused", both), *rest)
    return counted


drivers.scaled_problem = _counting(drivers.scaled_problem)


def main():
    args = bench.arguments(__doc__)
    rows = []
    for name in args.configurations:
        live = bench.open_session(name, args)
        for arm in live.arms:
            _, assemble_s = bench.timed(live.assemble, arm)
            cold, cold_s = bench.timed(live.solve, arm)      # compiles, then solves
            warm_times = []
            for _ in range(args.repeats):                     # the same solve, compiled
                CALLS.clear()
                warm, seconds = bench.timed(live.solve, arm)
                warm_times.append(seconds)
            warm_s = sorted(warm_times)[len(warm_times) // 2]
            model_s = CALLS["values_s"] + CALLS["jacobians_s"] + CALLS["fused_s"]
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
                "values": CALLS["values"],            # model calls in the last warm solve
                "jacobians": CALLS["jacobians"],
                "fused": CALLS["fused"],
                "model_s": model_s,                   # the seconds inside them
                "note": warm.get("note", ""),
            })
            print({k: v for k, v in rows[-1].items() if k != "note"})
    bench.write("solve.py", rows, f"solve_{args.scheme}_{args.optimiser}")


if __name__ == "__main__":
    main()
