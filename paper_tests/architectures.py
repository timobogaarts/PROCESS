"""The architectures table: MDF and SAND, under each cut, under each optimiser.

One row per (configuration, arm, optimiser, cut): the optimiser's iterations and
verdict, the objective, the problem's size in the optimiser (design entries and
condition entries -- a Jacobi SAND carries whole profiles), the cold wall (assembly,
trace, compile, first solve -- `session.Session.solve`'s first call) and the warm wall
(the least of `--repeats` further solves of the same assembled arm). Rows are
checkpointed to the CSV as they land, so a killed run keeps what it measured;
`--input`, `--arm`, `--driver`, `--recipe` select.

What the deleted `run_warm_matrix` also gave -- the XLA share of the warm wall and
the model ms per call, from its phase timing -- is not measured here: the port's
drivers carry no timing hooks.

    $PY paper_tests/architectures.py                     # everything (~1 h)
    $PY paper_tests/architectures.py --input helias_5b
    $PY paper_tests/architectures.py --recipe jacobi --arm sand --driver SLSQP
    $PY paper_tests/architectures.py --render            # the tables from out/architectures.csv
"""

from __future__ import annotations

import sys
import time

import jax
import numpy as np
from common import CONFIGURATIONS, LABEL, RECIPES, cut_for, fmt, stem, tex_name, write_csv, write_tex

from functional_process.cottax.architectures import session
from functional_process.cottax.architectures.drivers import SlsqpDriver
from functional_process.cottax.architectures.evaluate import mda_env

DRIVERS = {"VMCON": None, "SLSQP": SlsqpDriver}


def _sizes(live, arm: str) -> dict:
    """Design and condition entry counts of the optimiser's problem, after assembly."""
    if arm == "sand":
        drive = live.builds["SAND"].drive
        try:
            env = mda_env(live.reference, graph=live.machine_graph,
                          **({} if live.cut is None else {"cut": live.cut}))[1]
            design = sum(int(np.size(env[u])) if u in env else 1 for u in drive.unknowns)
        except Exception:  # noqa: BLE001 -- a size, not a result
            design = len(drive.unknowns)
        return {"unknowns": len(drive.unknowns), "design_entries": design,
                "conditions": len(drive.conditions)}
    problem = live.builds["MDF"].problem
    return {"unknowns": len(problem.design), "design_entries": len(problem.design),
            "conditions": len(problem.conditions)}


def measure(live, arm: str, repeats: int) -> dict:
    """Cold (assembly + compile + first solve) and warm (the least of `repeats` more)."""
    began = time.perf_counter()
    cold = live.solve(arm.upper())
    cold_wall = time.perf_counter() - began
    warm = [live.solve(arm.upper())["seconds"] for _ in range(repeats)]
    return dict(cold, _cold_wall=cold_wall, _wall=min(warm) if warm else None)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--help" in argv or "-h" in argv:
        print(__doc__)
        return 0
    chosen = [argv[i + 1] for i, a in enumerate(argv) if a == "--input"] or list(CONFIGURATIONS)
    arms = [argv[i + 1] for i, a in enumerate(argv) if a == "--arm"] or ["mdf", "sand"]
    which_drivers = [argv[i + 1] for i, a in enumerate(argv) if a == "--driver"] or list(DRIVERS)
    which = [argv[i + 1] for i, a in enumerate(argv) if a == "--recipe"] or list(RECIPES)
    repeats = int(argv[argv.index("--repeats") + 1]) if "--repeats" in argv else 2
    rows = []
    began = time.perf_counter()
    for path in chosen:
        name = stem(str(path))
        for arm in arms:
            for driver in which_drivers:
                for recipe in which:
                    row = {"configuration": name, "arm": arm.upper(), "driver": driver, "recipe": recipe}
                    try:
                        live = session.open_session(name, optimiser=DRIVERS[driver], cut=cut_for(recipe))
                        if arm == "sand" and live.root_find:
                            continue
                        r = measure(live, arm, repeats)
                        row.update(_sizes(live, arm))
                        row.update(
                            iterations=r.get("iterations"), status=r.get("status"),
                            objf=r.get("objf"), note=(r.get("note") or "")[:120],
                            cold_wall=r["_cold_wall"], warm_wall=r["_wall"],
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
    optimiser: `it / entries / warm wall`."""
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
                cells.append(f"{r['iterations']} / {r['design_entries']} / {fmt(r['warm_wall'], 2)}")
            tex.append(cells)
        write_tex(
            "architectures.py", ["configuration", "cut", "VMCON", "SLSQP"], tex,
            name=f"architectures_{arm.lower()}", align="llrr",
            caption_note=f"{arm}: SQP iterations / design entries / warm wall [s]",
        )


def render_csv(path=None):
    """Re-render the tables from an existing CSV, e.g. after a partial run."""
    import csv  # noqa: PLC0415

    from common import OUT  # noqa: PLC0415

    with open(path or OUT / "architectures.csv") as handle:
        rows = []
        for r in csv.DictReader(handle):
            for k in ("warm_wall", "cold_wall"):
                r[k] = float(r[k]) if r.get(k) else None
            rows.append(r)
    _render(rows)


if __name__ == "__main__":
    if "--render" in sys.argv:
        render_csv()
        raise SystemExit(0)
    raise SystemExit(main())
