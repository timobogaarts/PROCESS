"""Table 1 -- what each architecture makes of each machine: the graph's size, what the
scheme cut, and what the optimiser holds. Also the DSM of every arm, as the port's
interactive page (`out/dsm/<configuration>_<arm>.html`).

    $PY paper_tests/architectures/structure.py [--scheme minimal] [--configurations ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402
from cottax.interfaces import (  # noqa: E402
    RunnableGraph,
    Schedule,
    is_optimise,
    problems,
)

from functional_process.cottax.architectures.evaluate import (
    without_excluded,  # noqa: E402
)
from functional_process.cottax.architectures.mda import SCHEME, relation_counts  # noqa: E402
from functional_process.cottax.visualization.grouping import (  # noqa: E402
    render_grouped_dsm_html,
    structure_order,
)
from functional_process.cottax.visualization.render_xdsm import SPELLING  # noqa: E402


def graph_of(live, arm, build):
    """The driven graph the arm solves."""
    if arm == "MDA":
        return build.problem.eager.executable.graph
    if live.root_find:
        return build.in_graph.blocking.graph
    return build.solve_schedule.executable.graph


def schedule_of(live, arm, build):
    if arm == "MDA":
        return build.problem.eager
    if live.root_find:
        return Schedule(build.in_graph.blocking)
    return build.solve_schedule


def cuts_of(live, args) -> int:
    """How many variables the scheme cuts on this machine's raw graph."""
    raw = without_excluded(live.machine_graph)
    return sum(len(cl.cuts) for cl in bench.SCHEMES[args.scheme].composed(raw))


def top_of(graph) -> dict:
    """What the one outermost statement holds -- the optimiser, or an evaluation's
    root find: its unknowns, its objectives and its relations split by symbol. Zeros
    where there is no single top (the MDA: every solve stands on its own)."""
    top = [n for n in graph.outermost_problems if is_optimise(graph[n])] or list(graph.outermost_problems)
    if len(top) != 1:
        return {"unknowns": 0, "objectives": 0, "equalities": 0, "inequalities": 0}
    node = graph[top[0]]
    n_equality, n_inequality = relation_counts(node)
    return {
        "unknowns": len(node.unknowns),
        "objectives": len(node.objectives),
        "equalities": n_equality,
        "inequalities": n_inequality,
    }


def dsm(live, arm, graph):
    out = bench.OUT / "dsm"
    out.mkdir(parents=True, exist_ok=True)
    drawn = RunnableGraph(graph)
    render_grouped_dsm_html(
        drawn,
        order=structure_order(drawn),
        mode="structure",
        title=f"{live.name} -- {arm}",
        file_name=f"{live.name}_{arm}",
        outdir=str(out),
        write=True,
        formatter=SPELLING,
    )


def main():
    args = bench.arguments(__doc__, optimiser=False)
    rows = []
    for name in args.configurations:
        live = bench.open_session(name, args)
        raw = without_excluded(live.machine_graph)
        for arm in live.arms:
            build = live.assemble(arm)
            graph = graph_of(live, arm, build)
            top = (
                {"unknowns": len(build.in_graph.design), "objectives": 0,
                 "equalities": len(build.in_graph.conditions), "inequalities": len(build.in_graph.reported)}
                if live.root_find and arm == "MDF" else top_of(graph)
            )
            rows.append({
                "configuration": name,
                "problem": "root-find" if live.root_find else "optimise",
                "arm": arm,
                "nodes": len(graph.nodes),
                "cycles": len(SCHEME.composed(raw)),   # one closure per cycle the scheme opens
                "cut_variables": cuts_of(live, args),
                "problems": len(problems(graph.definitions)),
                "nested": len(graph.within),
                "steps": len(schedule_of(live, arm, build).steps),
                **top,
            })
            dsm(live, arm, graph)
            print(rows[-1])
    bench.write("structure.py", rows, f"structure_{args.scheme}")


if __name__ == "__main__":
    main()
