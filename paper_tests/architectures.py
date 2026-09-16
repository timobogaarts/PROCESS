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
    """Two `tabular`s -- MDF and SAND -- one row per configuration x cut, one cell per
    optimiser: `it / entries / warm wall (XLA share)`."""
    for arm in ("MDF", "SAND"):
        tex = []
        keys = list(dict.fromkeys((r["configuration"], r["recipe"]) for r in rows if r["arm"] == arm))
        for cfg, recipe in keys:
            cells = [tex_name(cfg), LABEL[recipe]]
            for driver in ("VMCON", "SLSQP"):
                match = [r for r in rows if r["arm"] == arm and r["configuration"] == cfg
                         and r["recipe"] == recipe and r["driver"] == driver]
                if not match:
                    cells.append("--")
                    continue
                r = match[0]
                if r.get("status") != "converged":
                    cells.append(f"{r.get('status')} ({r.get('iterations') or '--'})")
                    continue
                share = (r["warm_xla"] / r["warm_wall"] * 100) if r.get("warm_wall") else 0
                cells.append(f"{r['iterations']} / {r['design_entries']} / {fmt(r['warm_wall'], 2)} ({share:.0f}\\%)")
            tex.append(cells)
        write_tex(
            "architectures.py", ["configuration", "cut", "VMCON", "SLSQP"], tex,
            name=f"architectures_{arm.lower()}", align="llrr",
            caption_note=(f"{arm}: SQP iterations / design entries / warm wall [s] "
                          f"(share of it inside the compiled block programs); the rest is the "
                          f"optimiser's own cost"),
        )


def render_csv(path=None):
    """Re-render the tables from an existing CSV, e.g. after a partial run."""
    import csv  # noqa: PLC0415

    from common import OUT  # noqa: PLC0415

    with open(path or OUT / "architectures.csv") as handle:
        rows = []
        for r in csv.DictReader(handle):
            for k in ("warm_wall", "warm_xla", "cold_wall", "ms_per_call"):
                r[k] = float(r[k]) if r.get(k) else None
            rows.append(r)
    _render(rows)


if __name__ == "__main__":
    if "--render" in sys.argv:
        render_csv()
        raise SystemExit(0)
    raise SystemExit(main())
