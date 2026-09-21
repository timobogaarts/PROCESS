"""The paper's table from the four csvs: one row per machine and arm, PROCESS's own
run beside them.

    $PY paper_tests/architectures/table.py [--scheme minimal]

Writes `out/table.tex` (a `tabular` for `\\input`) and prints it. Columns:

    machine | arm | unknowns / conditions | block nodes | eval ms | jac ms
            | VMCON: iterations, time | SLSQP: iterations, time | objective

with PROCESS's row per machine: its pass, its loop, its gradient, its iterations, its
time, its objective. Times are warm wall-clock seconds; a status other than converged
is written after the iteration count.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402

SHORT = {"stellarator_helias": "helias", "helias_5b": "helias-5b", "large_tokamak_nof": "tokamak",
         "large_tokamak_eval": "tokamak (eval)", "low_aspect_ratio_DEMO": "LAR demo",
         "spherical_tokamak_eval": "ST (eval)", "st_regression": "ST"}


def by(rows, *keys):
    return {tuple(r[k] for k in keys): r for r in rows}


def num(v, digits=3):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "--"
    if x != x:
        return "--"
    return f"{x:.{digits}g}"


def iterations(row):
    if row is None:
        return "--"
    it = row["iterations"]
    return it if row["status"] == "converged" else f"{it} ({row['status']})"


def main():
    args = bench.arguments(__doc__, optimiser=False)
    structure = by(bench.read(f"structure_{args.scheme}"), "configuration", "arm")
    iteration = by(bench.read(f"iteration_{args.scheme}"), "configuration", "arm")
    solves = {opt: by(bench.read(f"solve_{args.scheme}_{opt}"), "configuration", "arm") for opt in bench.OPTIMISERS}
    native = by(bench.read("native"), "configuration")

    lines = [
        r"\begin{tabular}{llrrrrrrrrr}",
        r"machine & arm & $n$ / $m$ & nodes & eval (ms) & jac (ms) & VMCON it & VMCON (s) & SLSQP it & SLSQP (s) & $f^*$ \\ \hline",
    ]
    for name in bench.NAMES:
        arms = [a for a in bench.ARMS if (name, a) in structure]
        for i, arm in enumerate(arms):
            st, it = structure[(name, arm)], iteration[(name, arm)]
            v, s = solves["vmcon"].get((name, arm)), solves["slsqp"].get((name, arm))
            lines.append(" & ".join([
                SHORT.get(name, name) if i == 0 else "",
                arm,
                f"{st['unknowns']} / {int(st['equalities']) + int(st['inequalities'])}",
                it["block_nodes"],
                num(it["evaluate_ms"]),
                num(it["jacobian_ms"]),
                str(iterations(v)), num(v["warm_s"]) if v else "--",
                str(iterations(s)), num(s["warm_s"]) if s else "--",
                num((v or s or {}).get("objf"), 6),
            ]) + r" \\")
        p = native.get((name,))
        if p is not None:
            lines.append(" & ".join([
                "", "PROCESS",
                f"{p['design']} / {p['constraints']}",
                "--",
                f"{num(p['pass_ms'])} / {num(p['loop_ms'])}",
                num(p["gradient_ms"]),
                p["iterations"], num(p["solve_s"]),
                "--", "--",
                num(p["objf"], 6),
            ]) + r" \\")
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    text = "\n".join(lines) + "\n"
    (bench.OUT / "table.tex").write_text(bench.provenance("table.py") .replace("#", "%") + "\n" + text)
    print(text)


if __name__ == "__main__":
    main()
