"""The paper's table from the four csvs: one row per machine and optimising arm,
PROCESS's own run beside them.

    $PY paper_tests/architectures/table.py [--scheme minimal] [--paper]

Writes `out/table.tex` (a booktabs `tabular` for `\\input`) and prints it. Columns:

    arm | n, equalities, inequalities, block nodes | VMCON it, SLSQP it
        | eval ms, jac ms, VMCON warm s, SLSQP warm s | objective

under one row per machine naming it,

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
"""Short names for the side tables (scaling, OpenMDAO) that the paper does not print."""


def machine_row(name, columns, last_rule=True):
    """The machine's own name -- the configuration's, as the paper's figures print it --
    in the first column, over its arms: that column is as wide as the longest name, and
    the rules between the groups run through the row unbroken."""
    cell = rf"\rule{{0pt}}{{2.4ex}}\texttt{{{name.replace('_', chr(92) + '_')}}}"
    if last_rule:
        return cell + " &" * (columns - 1) + r" \\"
    # No rule before the last column (a root find has no objective): the column before
    # it takes plain space of the rule's width instead, as `main`'s `no_answer` does.
    return cell + " &" * (columns - 2) + r" \multicolumn{1}{r@{\hspace{8.2pt}}}{} & \\"


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


def ms(seconds):
    """A time in seconds, printed in ms beside the eval and jac columns."""
    return fixed(1e3 * float(seconds))


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

    # Four groups, ruled apart: the problem (sizes), the iterations each driver took,
    # the times, the answer. `booktabs` rules and `|` do not meet cleanly, so the
    # separator is a hairline with its own padding, and the rules lose their gaps.
    bar = r"@{\hspace{4pt}\vrule width 0.2pt\hspace{4pt}}"
    # The first rule, between the names and the numbers, heavier than the three
    # between the number groups.
    first = r"@{\hspace{6pt}\vrule width 0.8pt\hspace{6pt}}"
    def header(arm, iterations, times, answer=True):
        last = rf"{times[1]} (ms) & $f^*$" if answer else no_answer(f"{times[1]} (ms)")
        return (rf"\rule{{0pt}}{{2.4ex}}{arm} arm & $n$ & $m_\mathrm{{eq}}$ & $m_\mathrm{{ineq}}$ & nodes"
                rf" & {iterations[0]} it & {iterations[1]} & eval (ms) & jac (ms)"
                rf" & {times[0]} (ms) & {last} \\[0.3ex]")

    def no_answer(cell):
        """The last time column where there is no objective: the rule before `f^*` is
        that column's right edge, so it is replaced by plain space of the same width
        (4 + 0.2 + 4 pt) and the empty `f^*` cell stands unruled."""
        return rf"\multicolumn{{1}}{{r@{{\hspace{{8.2pt}}}}}}{{{cell}}} &"

    def root_find(name):
        return structure[(name, "MDF")].get("problem") == "root-find"

    lines = [
        r"\setlength{\aboverulesep}{0pt}\setlength{\belowrulesep}{0pt}",
        r"\begin{tabular}{l" + first + "rrrr" + bar + "rr" + bar + "rrrr" + bar + "r}",
        r"\toprule",
        header("Optimization", ("VMCON", "SLSQP it"), ("VMCON", "SLSQP")),
    ]
    for name in [n for n in bench.NAMES if (n, "MDF") in structure and not root_find(n)]:
        lines += [r"\midrule", machine_row(name, 12)]
        for arm in [a for a in bench.ARMS if shown(name, a) and (name, a) in structure]:
            st, it = structure[(name, arm)], iteration[(name, arm)]
            v, s = solves["vmcon"].get((name, arm)), solves["slsqp"].get((name, arm))
            lines.append(" & ".join([
                arm,
                st["unknowns"], st["equalities"], st["inequalities"], it["block_nodes"],
                marked(v), marked(s),
                fixed(1e-3 * float(it["serial_us"])),
                fixed(1e-3 * float(it.get("serial_jacobian_us", "nan"))),
                ms(v["warm_s"]) if v else "--", ms(s["warm_s"]) if s else "--",
                fixed((v or s or {}).get("objf"), 4),
            ]) + r" \\")
        p = native.get((name,))
        if p is not None:
            mdf = structure[(name, "MDF")]
            lines.append(" & ".join([
                "PROCESS",
                p["design"], mdf["equalities"], mdf["inequalities"], "--",
                f"\\checkmark\\ {p['iterations']}", "--",
                fixed(p["loop_ms"]), fixed(p["gradient_ms"]),
                ms(p["solve_s"]), "--",
                fixed(p["objf"], 4),
            ]) + r" \\[0.3ex]")

    # The root finds (`i_process_run_mode = -2`) under their own header: neither side
    # runs an optimiser. The port's MDF states the root find in the graph and a Newton
    # (optimistix) answers it, whichever optimiser the session holds; PROCESS hands it
    # to `fsolve`, MINPACK's hybrid Powell. Its column is headed "it" like the others,
    # but what it holds is function evaluations -- its difference Jacobian's columns
    # included -- since MINPACK reports no iterations: the caption has to say so.
    roots = [n for n in bench.NAMES if (n, "MDF") in structure and root_find(n)]
    if roots:
        lines += [r"\midrule\midrule", header("Evaluation", ("Newton", "Powell it"), ("Newton", "Powell"), answer=False)]
    for name in roots:
        lines += [r"\midrule", machine_row(name, 12, last_rule=False)]
        st, it = structure[(name, "MDF")], iteration[(name, "MDF")]
        v = solves["vmcon"].get((name, "MDF"))
        lines.append(" & ".join([
            "MDF",
            st["unknowns"], st["equalities"], st["inequalities"], it["block_nodes"],
            marked(v), "--",
            fixed(1e-3 * float(it["serial_us"])),
            fixed(1e-3 * float(it.get("serial_jacobian_us", "nan"))),
            ms(v["warm_s"]) if v else "--",
        ]) + " & " + no_answer("--") + r" \\")
        p = native.get((name,))
        if p is not None:
            evaluations = p.get("fsolve_evaluations")
            solved = evaluations and float(p["max_eq"]) < 1e-6
            lines.append(" & ".join([
                "PROCESS",
                p["design"], st["equalities"], st["inequalities"], "--",
                "--", f"\\checkmark\\ {evaluations}" if solved else (evaluations or "--"),
                fixed(p["loop_ms"]), fixed(p["gradient_ms"]),
                "--",
            ]) + " & " + no_answer(ms(p["solve_s"])) + r" \\[0.3ex]")
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


BATCHED_N = {"cpu": (1, 256, 4096, 16384), "gpu": (1, 256, 4096, 16384, 65536, 262144)}
"""The batches the batched table prints, per platform, as their actual N: wall time of
one call over the whole batch, so a flat row is designs for free and a row growing
with N is a saturated device."""


def batched(args):
    """`out/table_batched.tex`: wall time (ms) of one evaluation and one Jacobian of the
    arm's block over N designs at once, CPU and GPU, at the batches `BATCHED_N` names --
    one row per quantity, so the eye reads the scaling along the row."""
    rows = {p: bench.read(f"batched_{args.scheme}_{p}") for p in ("cpu", "gpu")}
    structure = by(bench.read(f"structure_{args.scheme}"), "configuration", "arm")
    if not rows["cpu"] and not rows["gpu"]:
        return

    def at(platform, name, arm, n):
        return next((r for r in rows[platform]
                     if r["configuration"] == name and r["arm"] == arm and int(r["N"]) == n), None)

    columns = [(p, n) for p in ("cpu", "gpu") for n in BATCHED_N[p]]
    k_cpu, k_gpu = len(BATCHED_N["cpu"]), len(BATCHED_N["gpu"])
    lines = [
        r"\begin{tabular}{ll" + "r" * len(columns) + "}",
        r"\toprule",
        rf" & & \multicolumn{{{k_cpu}}}{{c}}{{CPU, $N$}} & \multicolumn{{{k_gpu}}}{{c}}{{GPU, $N$}} \\",
        rf"\cmidrule(lr){{3-{2 + k_cpu}}} \cmidrule(lr){{{3 + k_cpu}-{2 + k_cpu + k_gpu}}}",
        r"arm & & " + " & ".join(str(n) for _, n in columns) + r" \\",
    ]
    for name in bench.NAMES:
        arms = [a for a in ("MDF", "IDF", "SAND") if shown(name, a) and (name, a) in structure]
        if not arms:
            continue
        lines += [r"\midrule", machine_row(name, 2 + len(columns))]
        for arm in arms:
            for j, (label, key) in enumerate((("eval", "evaluate_ms"), ("jac", "jacobian_ms"))):
                cells = []
                for platform, n in columns:
                    r = at(platform, name, arm, n)
                    cells.append(fixed(r[key]) if r else "--")
                lines.append(" & ".join([arm if j == 0 else "", label, *cells]) + r" \\")
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
