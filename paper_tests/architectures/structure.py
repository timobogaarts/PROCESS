"""Table 1 -- what each architecture makes of each machine: the graph's size, what the
scheme cut, and what the optimiser holds. Also the N2 of every arm, as cottax's own
interactive page (`out/dsm/<configuration>_<arm>.html`).

    $PY paper_tests/architectures/structure.py [--scheme minimal] [--configurations ...]

The graphs are `study`'s (`study.mda_graph`, `study.block_graph`), built without a
solve. `cycles` and `cut_variables` are the closures `cx.mda` binds on the machine alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402
import cottax.relational_pytree as cx  # noqa: E402
from cottax.relational_pytree import Eq, Le  # noqa: E402
from cottax.consumers.visualization import render_html  # noqa: E402

from functional_process import configurations  # noqa: E402
from functional_process.cottax import study  # noqa: E402


def graph_of(configuration, arm, rule):
    """The driven graph the arm solves: no MDA is solved, so the top's condition
    scaling is left at one (which moves no edge)."""
    if arm == "MDA":
        return study.mda_graph(study.machine_model(configuration), rule)
    return study.block_graph(configuration, arm, {}, [], rule=rule)[1]


def top_of(graph) -> dict:
    """What `study.TOP` holds -- the optimiser, or an evaluation's root find: its
    unknowns, its objectives and its relations split by symbol, a bound on an unknown
    aside. Zeros where there is no top (the MDA: every solve stands on its own)."""
    if study.TOP not in graph.nodes:
        return {"unknowns": 0, "objectives": 0, "equalities": 0, "inequalities": 0}
    node = graph[study.TOP].condition
    held = [r for r in node.relations if not study._is_bound(node, r)]
    return {
        "unknowns": len(node.unknowns),
        "objectives": len(node.objectives),
        "equalities": sum(r.op is Eq for r in held),
        "inequalities": sum(r.op is Le for r in held),
    }


def dsm(name, arm, graph):
    out = bench.OUT / "dsm"
    out.mkdir(parents=True, exist_ok=True)
    render_html(graph, views=("n2",), write=True, file_name=f"{name}_{arm}", outdir=str(out))


def main():
    args = bench.arguments(__doc__, optimiser=False)
    rule = bench.SCHEMES[args.scheme]
    rows = []
    for name in args.configurations:
        configuration = configurations.load(name)
        closures = cx.mda(rule).resolution(study.machine_model(configuration).graph).closures
        for arm in study.arms(configuration):
            graph = graph_of(configuration, arm, rule)
            rows.append({
                "configuration": name,
                "problem": "root-find" if bench.is_root_find(configuration) else "optimise",
                "arm": arm,
                "nodes": len(graph.nodes),
                "cycles": len(closures),             # one closure per cycle the scheme opens
                "cut_variables": sum(len(c.cuts) for c in closures),
                "problems": sum(study.is_problem(graph[n]) for n in graph.nodes),
                "nested": len(graph.within),
                "steps": len(study.proved(graph).steps),
                **top_of(graph),
            })
            dsm(name, arm, graph)
            print(rows[-1])
    bench.write("structure.py", rows, f"structure_{args.scheme}")


if __name__ == "__main__":
    main()
