"""The paper's table from the four csvs: one row per machine and arm, PROCESS's own
run beside them.

    $PY paper_tests/architectures/table.py [--scheme minimal]

Writes `out/table.tex` (a `tabular` for `\\input`) and prints it. Columns:

    machine | arm | unknowns / conditions | block nodes | eval ms | jac ms
            | VMCON: iterations, time | SLSQP: iterations, time | objective

with PROCESS's row per machine: its pass / its idempotence loop under `eval`, its
finite-difference gradient under `jac`, its iterations, its full run under `cold`, its
objective. `cold` includes assembly and compilation, `warm` is the same solve again;
the `MDA` row's `eval` is one run of the analysis. A status other than converged is
written after the iteration count.
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
    if row is None or row["status"] == "evaluated":
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
        r"\begin{tabular}{llrrrrrrrrrr}",
        r"machine & arm & $n$ / $m$ & nodes & eval (ms) & jac (ms) & VMCON it & cold (s) & warm (s) & SLSQP it & warm (s) & $f^*$ \\ \hline",
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
                str(iterations(v)), num(v["cold_s"]) if v and arm != "MDA" else "--", num(v["warm_s"]) if v and arm != "MDA" else "--",
                str(iterations(s)), num(s["warm_s"]) if s and arm != "MDA" else "--",
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
                p["iterations"] if int(p["iterations"]) else "--", "--", num(p["solve_s"]),
                "--", "--",
                num(p["objf"], 6) if int(p["iterations"]) else "--",
            ]) + r" \\")
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    text = "\n".join(lines) + "\n"
    (bench.OUT / "table.tex").write_text(bench.provenance("table.py").replace("#", "%") + "\n" + text)
    print(text)
    batched(args)
    openmdao(args)


def openmdao(args):
    """`out/table_openmdao.tex`: the same MDF under OpenMDAO beside the port's, per
    machine -- one converged MDA and one total Jacobian, warm, and the SLSQP run."""
    om = by(bench.read("openmdao_mdf"), "configuration")
    if not om:
        return
    it = by(bench.read(f"iteration_{args.scheme}"), "configuration", "arm")
    sv = by(bench.read(f"solve_{args.scheme}_slsqp"), "configuration", "arm")
    lines = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"machine & components & \multicolumn{2}{c}{OpenMDAO: MDA / Jacobian (ms)} & SLSQP it & (s) & \multicolumn{2}{c}{port: eval / Jacobian (ms)} & SLSQP it (s) \\ \hline",
    ]
    for name in bench.NAMES:
        o = om.get((name,))
        if o is None:
            continue
        p_it, p_sv = it.get((name, "MDF")), sv.get((name, "MDF"))
        lines.append(" & ".join([
            SHORT.get(name, name), o["components"],
            num(o["model_ms"]), num(o["totals_ms"]), str(iterations(o)), num(o["driver_s"]),
            num(p_it["evaluate_ms"]) if p_it else "--", num(p_it["jacobian_ms"]) if p_it else "--",
            f"{iterations(p_sv)} ({num(p_sv['warm_s'])})" if p_sv else "--",
        ]) + r" \\")
    lines.append(r"\end{tabular}")
    text = "\n".join(lines) + "\n"
    (bench.OUT / "table_openmdao.tex").write_text(bench.provenance("table.py").replace("#", "%") + "\n" + text)
    print(text)


def batched(args):
    """`out/table_batched.tex`: the block over N designs, per design -- the CPU at
    N = 1 and at its largest batch, the GPU at 4096 and at the largest batch it took."""
    cpu = bench.read(f"batched_{args.scheme}_cpu")
    gpu = bench.read(f"batched_{args.scheme}_gpu")
    if not cpu and not gpu:
        return

    def at(rows, name, arm, n=None):
        mine = [r for r in rows if r["configuration"] == name and r["arm"] == arm]
        if not mine:
            return None
        if n is None:
            return max(mine, key=lambda r: int(r["N"]))
        return next((r for r in mine if int(r["N"]) == n), None)

    def cell(r):
        return "--" if r is None else f"{num(r['evaluate_us_per_design'])} / {num(r['jacobian_us_per_design'])}"

    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"machine & arm & $n$ & CPU $N{=}1$ & CPU $N{=}4096$ & GPU $N{=}4096$ & GPU largest & $N$ \\ \hline",
        r"& & & \multicolumn{4}{c}{$\mu$s per design, evaluation / Jacobian} & \\",
    ]
    for name in bench.NAMES:
        arms = [a for a in ("MDF", "IDF", "SAND") if at(cpu, name, a) or at(gpu, name, a)]
        for i, arm in enumerate(arms):
            top = at(gpu, name, arm)
            lines.append(" & ".join([
                SHORT.get(name, name) if i == 0 else "", arm,
                (at(cpu, name, arm) or at(gpu, name, arm))["unknowns"],
                cell(at(cpu, name, arm, 1)), cell(at(cpu, name, arm, 4096)),
                cell(at(gpu, name, arm, 4096)), cell(top), top["N"] if top else "--",
            ]) + r" \\")
        if arms:
            lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    text = "\n".join(lines) + "\n"
    (bench.OUT / "table_batched.tex").write_text(bench.provenance("table.py").replace("#", "%") + "\n" + text)
    print(text)


if __name__ == "__main__":
    main()
