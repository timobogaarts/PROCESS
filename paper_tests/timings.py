"""The timings table: the three reference matrices, one `tabular`.

Reads `functional_process/cottax/reference_cold_matrix.txt` (VMCON, cold),
`reference_slsqp_matrix.txt` (SLSQP, cold) and `reference_warm_matrix.txt` (both,
warm) -- regenerate those first (`_audit/performance.md` says how) -- and writes one
row per configuration and arm: the cold phases (trace + lower, compile, total) and
the steady state (iterations, wall, per-call model time) under each optimiser. No
solve is run here; this is a renderer, so the table and the references cannot
disagree.

    $PY paper_tests/timings.py
"""

from __future__ import annotations

from pathlib import Path

from common import ROOT, fmt, tex_name, write_csv, write_tex

REF = ROOT / "functional_process" / "cottax"


def cold_rows(path: Path) -> tuple[dict, dict]:
    """`{(configuration, form): row}` for the solve table and the timing block."""
    solve, timing, mode = {}, {}, None
    for line in path.read_text().splitlines():
        if line.startswith("COLD MATRIX"):
            mode = "rows"
            continue
        if line.startswith("PHASE TIMINGS"):
            mode = "timing"
            continue
        if line.startswith(("BOUNDARY VALUES", "NOTES")):
            mode = None
            continue
        f = line.split()
        if mode == "rows" and len(f) >= 16 and f[1] in ("MDF", "SAND") and f[-1].isdigit():
            solve[(f[0], f[1])] = dict(
                sqp=int(f[9]), status=f[10], objf=f[11], pro_objf=f[12], d_objf=f[13],
                worst_dx=f[14], max_eq=f[15], min_ie=f[16],
            )
        if mode == "timing" and len(f) == 9 and f[1] in ("MDF", "SAND"):
            timing[(f[0], f[1])] = dict(zip(
                ("trace", "lower", "compile", "model", "sqp", "other", "total"),
                map(float, f[2:]), strict=True,
            ))
    return solve, timing


def warm_rows(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        f = line.split()
        if len(f) == 11 and f[1] in ("MDF", "SAND") and f[2] in ("VMCON", "SLSQP"):
            out[(f[0], f[1], f[2])] = dict(
                it=int(f[3]), status=f[4], objf=f[5], wall=float(f[6]), xla=float(f[7]),
                host=float(f[8]), calls=int(f[9]), ms=float(f[10]),
            )
    return out


def main() -> int:
    vm_solve, vm_time = cold_rows(REF / "reference_cold_matrix.txt")
    sl_solve, sl_time = cold_rows(REF / "reference_slsqp_matrix.txt")
    warm = warm_rows(REF / "reference_warm_matrix.txt")
    rows, tex = [], []
    for key in vm_solve:
        cfg, form = key
        row = {"configuration": cfg, "arm": form}
        cells = [tex_name(cfg), form]
        for driver, solve, timing in (("VMCON", vm_solve, vm_time), ("SLSQP", sl_solve, sl_time)):
            t = timing.get(key, {})
            w = warm.get((cfg, form, driver), {})
            s = solve.get(key, {})
            row.update({
                f"{driver}_trace_lower": (t.get("trace", 0) + t.get("lower", 0)) if t else None,
                f"{driver}_compile": t.get("compile"),
                f"{driver}_cold_total": t.get("total"),
                f"{driver}_it": w.get("it", s.get("sqp")),
                f"{driver}_status": w.get("status", s.get("status")),
                f"{driver}_warm_wall": w.get("wall"),
                f"{driver}_ms_per_call": w.get("ms"),
            })
            cells += [
                fmt(row[f"{driver}_trace_lower"], 1), fmt(row[f"{driver}_compile"], 1),
                fmt(row[f"{driver}_cold_total"], 1), fmt(row[f"{driver}_it"]),
                fmt(row[f"{driver}_warm_wall"], 3),
                "--" if not w or w.get("calls", 0) == 0 else fmt(w["ms"], 1),
            ]
        rows.append(row)
        tex.append(cells)
    write_csv("timings.py", rows)
    write_tex(
        "timings.py",
        ["configuration", "arm",
         r"\multicolumn{6}{c}{VMCON}", r"\multicolumn{6}{c}{SLSQP}"],
        tex,
        align="ll" + "rrrrrr" * 2,
        caption_note=("per optimiser: trace+lower [s], compile [s], cold total [s], SQP iterations, "
                      "warm wall [s], model ms/call. Read `_audit/performance.md` for the caveats."),
    )
    # A second header line the fragment cannot carry in `write_tex`'s one-row header:
    path = Path(__file__).resolve().parent / "out" / "timings.tex"
    text = path.read_text().replace(
        r"\midrule",
        r"\cmidrule(lr){3-8}\cmidrule(lr){9-14}" "\n"
        " & & tr+lo & compile & cold & it & warm & ms/call & tr+lo & compile & cold & it & warm & ms/call \\\\\n"
        r"\midrule", 1,
    )
    path.write_text(text)
    for row in tex:
        print("  ".join(str(c) for c in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
