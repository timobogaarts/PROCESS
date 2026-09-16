"""The architectures table: MDF and SAND, under each cut, under each optimiser.

One row per (configuration, arm, optimiser, cut): the optimiser's iterations and
verdict, the objective, the problem's size in the optimiser (design entries and
condition entries -- a Jacobi SAND carries whole profiles), the cold wall (assembly,
trace, compile, first solve) and the warm wall with its XLA share, exactly as
`run_warm_matrix` measures them. Rows are checkpointed to the CSV as they land, so a
killed run keeps what it measured; `--input`, `--arm`, `--driver`, `--recipe` select.

    $PY paper_tests/architectures.py                     # everything (~1 h)
    $PY paper_tests/architectures.py --input tests/regression/input_files/helias_5b.IN.DAT
    $PY paper_tests/architectures.py --recipe jacobi --arm sand --driver SLSQP
"""

from __future__ import annotations

import sys
import time

import jax
from common import CONFIGURATIONS, LABEL, RECIPES, cut_for, fmt, stem, tex_name, write_csv, write_tex

from functional_process.cottax import session
from functional_process.cottax.core.solver import drivers as _drivers
from functional_process.cottax.importer import read_indat
from functional_process.cottax.run_cold_matrix import _resolve
from functional_process.cottax.run_warm_matrix import measure

DRIVERS = {"VMCON": None, "SLSQP": _drivers.SlsqpDriver}


def _sizes(live, arm: str) -> dict:
    """Design and condition entry counts of the optimiser's problem, after assembly."""
    if arm == "sand":
        drive = live.sand_build.drive
        env = None
        try:
            import numpy as np  # noqa: PLC0415

            from functional_process.cottax.sand_harness import mda_env  # noqa: PLC0415

            env = mda_env(live.reference, graph=live.machine_graph,
                          **({} if live.cut is None else {"cut": live.cut}))[1]
            design = sum(int(np.size(env[u])) if u in env else 1 for u in drive.unknowns)
        except Exception:  # noqa: BLE001 -- a size, not a result
            design = len(drive.unknowns)
        return {"unknowns": len(drive.unknowns), "design_entries": design,
                "conditions": len(drive.conditions)}
    problem = live.mdf_build.problem
    return {"unknowns": len(problem.design), "design_entries": len(problem.design),
            "conditions": len(problem.conditions)}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    chosen = [argv[i + 1] for i, a in enumerate(argv) if a == "--input"] or list(CONFIGURATIONS)
    arms = [argv[i + 1] for i, a in enumerate(argv) if a == "--arm"] or ["mdf", "sand"]
    which_drivers = [argv[i + 1] for i, a in enumerate(argv) if a == "--driver"] or list(DRIVERS)
    which = [argv[i + 1] for i, a in enumerate(argv) if a == "--recipe"] or list(RECIPES)
    repeats = int(argv[argv.index("--repeats") + 1]) if "--repeats" in argv else 2
    rows = []
    began = time.perf_counter()
    for path in chosen:
        path = _resolve(path)
        name = stem(str(path))
        root_find = read_indat(str(path)).problem.is_evaluation
        for arm in arms:
            if arm == "sand" and root_find:
                continue
            for driver in which_drivers:
                for recipe in which:
                    row = {"configuration": name, "arm": arm.upper(), "driver": driver, "recipe": recipe}
                    try:
                        live = session.open_session(str(path), optimiser=DRIVERS[driver], cut=cut_for(recipe))
                        r = measure(path, arm, DRIVERS[driver], repeats, cut=cut_for(recipe))
                        # `measure` opened its own session; sizes want the assembled build,
                        # so assemble once more here (cached programs make it cheap).
                        getattr(live, arm)()
                        row.update(_sizes(live, arm))
                        row.update(
                            iterations=r.get("iterations"), status=r.get("status"),
                            objf=r.get("objf"), note=(r.get("note") or "")[:120],
                            cold_wall=r["_cold_wall"], warm_wall=r["_wall"], warm_xla=r["_xla"],
                            calls=r["_calls"], ms_per_call=r["_median_call"] * 1000,
                        )
                    except Exception as failure:  # noqa: BLE001 -- a row, not an exit
                        row.update(status="FAILED", note=f"{type(failure).__name__}: {failure}"[:200])
                    rows.append(row)
                    print(f"{name:22} {arm.upper():4} {driver:5} {recipe:20} it {row.get('iterations')!s:>4} "
                          f"{row.get('status')!s:>10} objf {row.get('objf')!s:>14} "
                          f"design {row.get('design_entries')!s:>4} cond {row.get('conditions')!s:>3} "
                          f"cold {fmt(row.get('cold_wall'), 1):>6} warm {fmt(row.get('warm_wall'), 3):>7}  "
                          f"[{time.perf_counter() - began:.0f} s]", flush=True)
                    write_csv("architectures.py", rows)
                    jax.clear_caches()
    _render(rows)
    return 0


def _render(rows):
    """One `tabular` per driver: configuration x arm rows, one cell per cut."""
    for driver in sorted({r["driver"] for r in rows}):
        tex = []
        keys = list(dict.fromkeys((r["configuration"], r["arm"]) for r in rows if r["driver"] == driver))
        for cfg, arm in keys:
            cells = [tex_name(cfg), arm]
            for recipe in RECIPES:
                match = [r for r in rows if r["driver"] == driver and r["configuration"] == cfg
                         and r["arm"] == arm and r["recipe"] == recipe]
                if not match:
                    cells.append("--")
                    continue
                r = match[0]
                if r.get("status") not in ("converged",):
                    cells.append(f"{r.get('status')}")
                    continue
                cells.append(f"{r['iterations']} / {r['design_entries']} / {fmt(r['warm_wall'], 2)}")
            tex.append(cells)
        write_tex(
            "architectures.py", ["configuration", "arm", *(LABEL[r] for r in RECIPES)], tex,
            name=f"architectures_{driver.lower()}",
            caption_note=f"{driver}: SQP iterations / design entries / warm wall [s], per cut",
        )


if __name__ == "__main__":
    raise SystemExit(main())
