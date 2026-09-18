"""The timings table: one row per configuration and arm (MDF, SAND), under VMCON and
SLSQP -- the cold wall (assembly, trace, compile, first solve), the SQP iterations,
the verdict and the warm wall of a second solve of the same assembled arm.

`--measure` takes the measurement through `architectures.session` (one `Session`
per configuration and optimiser, `solve(arm)` cold then `--repeats` more times) and
writes `out/timings.json`; without it this script is a renderer over that JSON, so
the table and the measurement cannot disagree. The three `reference_*_matrix.txt`
tables this used to read were deleted with the harness (`f3015ede`), and with them
the phase split (trace + lower / compile) and the model ms per call, which came from
the harness's own timing hooks: those columns are gone from this table.

    $PY paper_tests/timings.py --measure [--input <name>]... [--repeats 2]   # ~30 min
    $PY paper_tests/timings.py                                              # render
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import jax
from common import (
    CONFIGURATIONS,
    OUT,
    cottax_tree,
    fmt,
    machine,
    stem,
    tex_name,
    write_csv,
    write_json,
    write_tex,
)

JSON = OUT / "timings.json"
ARMS = ("MDF", "SAND")
DRIVERS = ("VMCON", "SLSQP")


def measure(chosen, repeats: int) -> dict:
    """`{configuration: {arm: {driver: row}}}`, checkpointed to `out/timings.json`
    after every row.
    """
    from functional_process.cottax.architectures import session  # noqa: PLC0415
    from functional_process.cottax.architectures.drivers import SlsqpDriver  # noqa: PLC0415

    optimisers = {"VMCON": None, "SLSQP": SlsqpDriver}
    rows: dict = {"machine": machine(), "cottax": cottax_tree(), "rows": {}}
    if JSON.exists():
        rows["rows"] = json.loads(JSON.read_text()).get("rows", {})
    began = time.perf_counter()
    for path in chosen:
        name = stem(str(path))
        for driver in DRIVERS:
            live = session.open_session(name, optimiser=optimisers[driver])
            for arm in ARMS:
                if arm not in live.arms:
                    continue
                cold_began = time.perf_counter()
                cold = live.solve(arm)
                cold_wall = time.perf_counter() - cold_began
                warm = [live.solve(arm)["seconds"] for _ in range(repeats)]
                row = {
                    "cold_total": cold_wall,
                    "cold_solve": cold["seconds"],
                    "it": cold["iterations"],
                    "status": cold["status"],
                    "objf": cold["objf"],
                    "max_eq": cold["max_eq"],
                    "min_ie": cold["min_ie"],
                    "warm_wall": min(warm) if warm else None,
                    "note": cold.get("note", ""),
                }
                rows["rows"].setdefault(name, {}).setdefault(arm, {})[driver] = row
                print(f"{name:22} {arm:4} {driver:5} it {row['it']!s:>4} {row['status']:>10} "
                      f"cold {cold_wall:6.1f} warm {fmt(row['warm_wall'], 3):>7}  "
                      f"[{time.perf_counter() - began:.0f} s]", flush=True)
                write_json("timings.py", rows)
            jax.clear_caches()
    return rows


def render(rows: dict) -> int:
    table, tex = [], []
    for cfg, arms in rows["rows"].items():
        for arm, drivers in arms.items():
            row = {"configuration": cfg, "arm": arm}
            cells = [tex_name(cfg), arm]
            for driver in DRIVERS:
                r = drivers.get(driver, {})
                row.update({
                    f"{driver}_cold_total": r.get("cold_total"),
                    f"{driver}_it": r.get("it"),
                    f"{driver}_status": r.get("status"),
                    f"{driver}_warm_wall": r.get("warm_wall"),
                })
                cells += [fmt(r.get("cold_total"), 1), fmt(r.get("it")), r.get("status", "--"),
                          fmt(r.get("warm_wall"), 3)]
            table.append(row)
            tex.append(cells)
    write_csv("timings.py", table)
    write_tex(
        "timings.py",
        ["configuration", "arm", r"\multicolumn{4}{c}{VMCON}", r"\multicolumn{4}{c}{SLSQP}"],
        tex,
        align="ll" + "rrlr" * 2,
        caption_note=(f"per optimiser: cold total [s] (assembly + compile + first solve), SQP "
                      f"iterations, verdict, warm wall [s]; measured on {rows.get('machine')}, "
                      f"cottax {str(rows.get('cottax', '')).split(' at ')[0]}"),
    )
    # A second header line the fragment cannot carry in `write_tex`'s one-row header:
    path = Path(__file__).resolve().parent / "out" / "timings.tex"
    text = path.read_text().replace(
        r"\midrule",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}" "\n"
        " & & cold & it & verdict & warm & cold & it & verdict & warm \\\\\n"
        r"\midrule", 1,
    )
    path.write_text(text)
    for cells in tex:
        print("  ".join(str(c) for c in cells))
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--help" in argv or "-h" in argv:
        print(__doc__)
        return 0
    if "--measure" in argv:
        chosen = [argv[i + 1] for i, a in enumerate(argv) if a == "--input"] or list(CONFIGURATIONS)
        repeats = int(argv[argv.index("--repeats") + 1]) if "--repeats" in argv else 2
        rows = measure(chosen, repeats)
        return render(rows)
    if not JSON.exists():
        raise SystemExit(
            f"no {JSON}: the reference_*_matrix.txt tables this rendered were deleted with "
            f"the harness (f3015ede); run `paper_tests/timings.py --measure` first"
        )
    return render(json.loads(JSON.read_text()))


if __name__ == "__main__":
    raise SystemExit(main())
