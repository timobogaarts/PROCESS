"""The inner analysis under each cut: Picard steps per coupled block, from one cold
state, and the steady-state cost of one converged MDA.

Every recipe starts from the same point -- `evaluate.cold_state`, the graph after
one pass in call order, which is where PROCESS starts too -- so the step counts
compare the *iteration*, not the start. `steps` is `optimistix`'s count per block;
`total` sums them; `depth` is the body's longest chain (`graph_census.py`), so
`total x depth` is a proxy for sequential node evaluations and `total` alone for a
machine that ran a Jacobi body's layer in parallel.

    $PY paper_tests/mda_convergence.py [--input <IN.DAT>]... [--repeats N]
"""

from __future__ import annotations

import sys
import time

import networkx as nx
from common import CONFIGURATIONS, LABEL, RECIPES, cut_for, fmt, stem, tex_name, write_csv, write_json, write_tex
from cottax.answerable import AnswerableGraph, runnable
from cottax.evaluation.schedule import Schedule
from cottax.names import PathMap
from cottax.visualization.sequencing import Solve, entries

from functional_process.cottax.architectures import session
from functional_process.cottax.architectures.drivers import PicardDriver
from functional_process.cottax.architectures.evaluate import (
    cold_state,
    jit_schedule,
    seed_env,
    without_excluded,
)
from functional_process.cottax.architectures.mda import assign_drivers, cut_graph, default_drivers
from functional_process.cottax.input.indat import graph_for


def measure(live, raw, recipe: str, repeats: int) -> dict:
    cut = cut_for(recipe)
    graph = (cut_graph if cut is None else cut)(raw)
    drivers = default_drivers(graph)
    for problem, driver in drivers.items():
        if isinstance(driver, PicardDriver):
            drivers[problem] = PicardDriver(report_steps=True)
    runnable_graph = assign_drivers(graph, drivers)
    schedule = Schedule(AnswerableGraph(runnable_graph))
    env = PathMap(seed_env(live.reference.data, schedule, runnable_graph,
                           cold_state(live.reference.data, live.machine_graph)))
    run = jit_schedule(schedule)
    out = dict(run(env))  # cold: compiles
    wall = []
    for _ in range(repeats):
        began = time.perf_counter()
        out = dict(run(env))
        wall.append(time.perf_counter() - began)
    steps = {}
    for var in out:
        if var.spelling.startswith("^driver_out.steps"):
            block = var.spelling.split("^problem", 1)[-1]
            steps[block] = int(out[var])
    depths = {}
    for entry in entries(runnable_graph):
        if not isinstance(entry, Solve) or len(entry.nodes) <= 1:
            continue
        body = runnable(entry.subgraph)
        depths[entry.problem.spelling.split("^problem", 1)[-1]] = (
            nx.dag_longest_path_length(body.graph._nx_dependencies) + 1 if body.nodes else 0
        )
    return {
        "steps": steps,
        "total_steps": sum(steps.values()),
        "sequential": sum(steps.get(b, 0) * d for b, d in depths.items()),
        "max_depth": max(depths.values(), default=0),
        "depths": depths,
        "warm_wall": min(wall),
    }


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--help" in argv or "-h" in argv:
        print(__doc__)
        return 0
    chosen = [argv[i + 1] for i, a in enumerate(argv) if a == "--input"] or list(CONFIGURATIONS)
    repeats = int(argv[argv.index("--repeats") + 1]) if "--repeats" in argv else 3
    rows, tex, detail = [], [], {}
    for path in chosen:
        live = session.open_session(path)
        raw = without_excluded(live.machine_graph if live.machine_graph is not None else graph_for())
        name = stem(path)
        detail[name] = {}
        cells = [tex_name(name)]
        for recipe in RECIPES:
            m = measure(live, raw, recipe, repeats)
            detail[name][recipe] = m
            rows.append({"configuration": name, "recipe": recipe,
                         **{k: v for k, v in m.items() if k not in ("steps", "depths")}})
            cells.append(f"{m['total_steps']} / {m['sequential']} / {fmt(m['warm_wall'] * 1000, 1)}")
            print(f"{name:24} {recipe:22} steps {m['steps']}  depths {m['depths']}  "
                  f"warm {m['warm_wall'] * 1000:.1f} ms", flush=True)
        tex.append(cells)
        import jax  # noqa: PLC0415

        jax.clear_caches()
    write_csv("mda_convergence.py", rows)
    write_json("mda_convergence.py", detail)
    write_tex(
        "mda_convergence.py",
        ["configuration", *(LABEL[r] for r in RECIPES)],
        tex,
        caption_note="per cut: Picard steps summed over coupled blocks / steps x body depth / warm MDA wall [ms]",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
