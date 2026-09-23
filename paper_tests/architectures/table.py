"""The paper's table from the four csvs: one row per machine and optimising arm,
PROCESS's own run beside them.

    $PY paper_tests/architectures/table.py [--scheme minimal] [--paper]

Writes `out/table.tex` (a booktabs `tabular` for `\\input`) and prints it. Columns:

    machine | arm | n | equalities | inequalities | block nodes | eval ms | jac ms
            | VMCON: iterations, warm s | SLSQP: iterations, warm s | objective

`eval` and `jac` are in-program (`serial_us`, `serial_jacobian_us`: no dispatch in
them). PROCESS's row per machine: its idempotence loop under `eval` (what it pays per
optimiser call), its finite-difference gradient under `jac`, its iterations and its
full run, its objective; its equality/inequality split is the MDF arm's, the same
problem. A mark precedes each iteration count: a tick for a solve that returned
converged, a cross otherwise. Times carry two significant figures, the objective four,
none in exponent notation. The `MDA` rows and the cold solve (mostly compilation) are
not printed.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402

SHORT = {"stellarator_helias": "helias", "helias_5b": "helias-5b", "large_tokamak_nof": "tokamak",
         "large_tokamak_eval": "tokamak (eval)", "low_aspect_ratio_DEMO": "LAR demo",
         "spherical_tokamak_eval": "ST (eval)", "st_regression": "ST"}


PAPER = Path.home() / "graph_paper" / "listings" / "process_cases"
"""Where `--paper` puts the two tabulars the paper `\\input`s."""
PAPER_NAMES = {"table.tex": "architectures_table.tex", "table_batched.tex": "architectures_batched.tex"}

IDF_IS_SAND = {"large_tokamak_nof", "low_aspect_ratio_DEMO", "st_regression"}
"""Machines whose IDF and SAND assemble to the same problem (same unknowns, nodes and
schedule; checked 2026-09-23 on cottax 078fb9f): no implicit model, so IDF has no
discipline solve to keep local. The tables print the SAND row only."""


def shown(name, arm):
    return arm != "MDA" and not (arm == "IDF" and name in IDF_IS_SAND)


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


def fixed(v, digits=2):
    """`v` to `digits` significant figures, as a plain decimal: `1230 -> 1200`,
    `0.0702 -> 0.070`; `--` for a missing value."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "--"
    if x != x:
        return "--"
    if x == 0:
        return "0"
    x = float(f"{x:.{digits}g}")
    decimals = max(0, digits - 1 - math.floor(math.log10(abs(x))))
    return f"{x:.{decimals}f}"


def marked(row):
    """The iteration count behind a tick (converged) or a cross (anything else)."""
    if row is None or row["status"] == "evaluated":
        return "--"
    mark = r"\checkmark" if row["status"] == "converged" else r"$\times$"
    return f"{mark}\\ {row['iterations']}"


def main():
    args = bench.arguments(__doc__, optimiser=False, paper=True)
    structure = by(bench.read(f"structure_{args.scheme}"), "configuration", "arm")
    iteration = by(bench.read(f"iteration_{args.scheme}"), "configuration", "arm")
    solves = {opt: by(bench.read(f"solve_{args.scheme}_{opt}"), "configuration", "arm") for opt in bench.OPTIMISERS}
    native = by(bench.read("native"), "configuration")

    lines = [
        r"\begin{tabular}{llrrrrrrrrrrr}",
        r"\toprule",
        r"machine & arm & $n$ & $m_\mathrm{eq}$ & $m_\mathrm{ineq}$ & nodes & eval (ms) & jac (ms)"
        r" & VMCON it & VMCON (s) & SLSQP it & SLSQP (s) & $f^*$ \\",
    ]
    for name in bench.NAMES:
        arms = [a for a in bench.ARMS if shown(name, a) and (name, a) in structure]
        if not arms:
            continue
        lines.append(r"\midrule")
        for i, arm in enumerate(arms):
            st, it = structure[(name, arm)], iteration[(name, arm)]
            v, s = solves["vmcon"].get((name, arm)), solves["slsqp"].get((name, arm))
            lines.append(" & ".join([
                SHORT.get(name, name) if i == 0 else "",
                arm,
                st["unknowns"], st["equalities"], st["inequalities"],
                it["block_nodes"],
                fixed(1e-3 * float(it["serial_us"])),
                fixed(1e-3 * float(it.get("serial_jacobian_us", "nan"))),
                marked(v), fixed(v["warm_s"]) if v else "--",
                marked(s), fixed(s["warm_s"]) if s else "--",
                fixed((v or s or {}).get("objf"), 4),
            ]) + r" \\")
        p = native.get((name,))
        if p is not None:
            mdf = structure[(name, "MDF")]
            solved = int(p["iterations"]) > 0
            lines.append(" & ".join([
                "", "PROCESS",
                p["design"], mdf["equalities"], mdf["inequalities"],
                "--",
                fixed(p["loop_ms"]),
                fixed(p["gradient_ms"]),
                f"\\checkmark\\ {p['iterations']}" if solved else "--",
                fixed(p["solve_s"]),
                "--", "--",
                fixed(p["objf"], 4) if solved else "--",
            ]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    text = "\n".join(lines) + "\n"
    write_table("table.tex", text, args)
    print(text)
    batched(args)
    scaling(args)
    openmdao(args)


def scaling(args):
    """`out/table_scaling.tex`: wall time of one evaluation and one Jacobian of the
    MDF block over a batch of N designs, in ms, CPU and GPU, per machine -- the
    scaling study. `--` where the run refused (memory) or did not run."""
    rows = {"cpu": bench.read(f"batched_{args.scheme}_cpu"), "gpu": bench.read(f"batched_{args.scheme}_gpu")}
    ns = sorted({int(r["N"]) for rs in rows.values() for r in rs})
    if not ns:
        return
    lines = [
        r"\begin{tabular}{ll" + "r" * len(ns) + "}",
        "machine & & " + " & ".join(f"$N={n}$" for n in ns) + r" \\ \hline",
    ]
    for name in bench.NAMES:
        for platform in ("cpu", "gpu"):
            mine = {int(r["N"]): r for r in rows[platform] if r["configuration"] == name and r["arm"] == "MDF"}
            if not mine:
                continue
            for what, key in (("eval", "evaluate_ms"), ("jac", "jacobian_ms")):
                lines.append(" & ".join([
                    SHORT.get(name, name) if (platform, what) == ("cpu", "eval") or (platform == "gpu" and what == "eval" and not any(r["configuration"] == name for r in rows["cpu"])) else "",
                    f"{platform.upper()} {what}",
                    *[num(mine[n][key]) if n in mine else "--" for n in ns],
                ]) + r" \\")
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    text = "\n".join(lines) + "\n"
    (bench.OUT / "table_scaling.tex").write_text(bench.provenance("table.py").replace("#", "%") + "\n" + text)
    print(text)


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


BATCHED_COLUMNS = (("cpu", None), ("cpu", 4096), ("gpu", 4096), ("gpu", 262144))
"""The batched table's column pairs: the CPU one design at a time (in-program, from
`iteration.py`: the N = 1 reference with no dispatch in it), then per design at a
batch the CPU and the GPU both take, then at the largest batch the GPU takes."""


def batched(args):
    """`out/table_batched.tex`: the block per design, evaluation and Jacobian, in us --
    the serial CPU reference beside the batches `BATCHED_COLUMNS` names."""
    rows = {p: bench.read(f"batched_{args.scheme}_{p}") for p in ("cpu", "gpu")}
    iteration = by(bench.read(f"iteration_{args.scheme}"), "configuration", "arm")
    structure = by(bench.read(f"structure_{args.scheme}"), "configuration", "arm")
    if not rows["cpu"] and not rows["gpu"]:
        return

    def at(platform, name, arm, n):
        return next((r for r in rows[platform]
                     if r["configuration"] == name and r["arm"] == arm and int(r["N"]) == n), None)

    def pair(platform, name, arm, n):
        if n is None:
            it = iteration.get((name, arm))
            return [fixed(it["serial_us"]), fixed(it.get("serial_jacobian_us"))] if it else ["--", "--"]
        r = at(platform, name, arm, n)
        return [fixed(r["evaluate_us_per_design"]), fixed(r["jacobian_us_per_design"])] if r else ["--", "--"]

    # Short, so a pair of number columns is not stretched under its header: the batch
    # as a power of two, the serial reference as "serial".
    head = ["CPU, serial" if n is None else f"{platform.upper()}, $2^{{{n.bit_length() - 1}}}$"
            for platform, n in BATCHED_COLUMNS]
    k = len(BATCHED_COLUMNS)
    lines = [
        r"\begin{tabular}{llr" + "rr" * k + "}",
        r"\toprule",
        " & & & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{h}}}" for h in head) + r" \\",
        " ".join(rf"\cmidrule(lr){{{4 + 2 * i}-{5 + 2 * i}}}" for i in range(k)),
        r"machine & arm & $n$ & " + " & ".join(["eval", "jac"] * k) + r" \\",
    ]
    for name in bench.NAMES:
        arms = [a for a in ("MDF", "IDF", "SAND") if shown(name, a) and (name, a) in structure]
        if not arms:
            continue
        lines.append(r"\midrule")
        for i, arm in enumerate(arms):
            cells = [c for platform, n in BATCHED_COLUMNS for c in pair(platform, name, arm, n)]
            lines.append(" & ".join([SHORT.get(name, name) if i == 0 else "", arm,
                                     structure[(name, arm)]["unknowns"], *cells]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    text = "\n".join(lines) + "\n"
    write_table("table_batched.tex", text, args)
    print(text)


def write_table(filename, text, args):
    """`out/<filename>`, and with `--paper` the paper's copy (`PAPER`, which main.tex
    `\\input`s), each under the provenance of the rows it was made from."""
    stamped = bench.provenance("table.py").replace("#", "%") + "\n" + text
    (bench.OUT / filename).write_text(stamped)
    if getattr(args, "paper", False):
        (PAPER / PAPER_NAMES[filename]).write_text(stamped)


if __name__ == "__main__":
    main()
