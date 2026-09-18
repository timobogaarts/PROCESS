"""The graphs, counted: what the PROCESS section's structural claims rest on.

Per configuration -- nodes, boundary inputs, the cyclic components of the raw graph
and their sizes, the problems the models declare themselves (root finds and
fixed-point self-loops) -- and then, per cut recipe, what closing those components
costs: copies minted (`^hat.*`), reads rerouted, and the **depth** of each coupled
block's body, the longest chain of nodes one Picard iterate has to evaluate in
sequence. A Jacobi body is one layer deep by construction; a Gauss-Seidel body is
as deep as its sweep.

    $PY paper_tests/graph_census.py            # all seven configurations
    $PY paper_tests/graph_census.py --input tests/regression/input_files/helias_5b.IN.DAT
"""

from __future__ import annotations

import sys

import networkx as nx
from common import CONFIGURATIONS, LABEL, RECIPES, cut_for, fmt, stem, tex_name, write_csv, write_json, write_tex
from cottax.abstract import runnable
from cottax.blocking import Blocking
from cottax.problem import ConditionalNode, is_fixed_point, is_root_find

from functional_process.cottax import session
from functional_process.cottax.indat import graph_for
from functional_process.cottax.mda_harness import _without_excluded


def raw_graph(live):
    return _without_excluded(live.machine_graph if live.machine_graph is not None else graph_for())


def census_raw(graph) -> dict:
    problems = [n for n in graph.nodes if isinstance(graph[n], ConditionalNode)]
    return {
        "nodes": len(graph.nodes),
        "functions": len(graph.nodes) - len(problems),
        "boundary_inputs": len(graph.boundary_inputs),
        "variables": len(graph.owned_variables),
        "components": len(graph.cycles),
        "component_sizes": [len(c) for c in graph.cycles],
        "largest_component": max((len(c) for c in graph.cycles), default=0),
        "root_finds": sum(1 for p in problems if is_root_find(graph[p])),
        "fixed_points": sum(1 for p in problems if is_fixed_point(graph[p])),
    }


def census_cut(graph) -> dict:
    """What a cut graph carries: copies, rerouted reads, problems, and body depths."""
    hats = [v for v in graph.owned_variables if v.spelling.startswith("^hat.")]
    reads = sum(
        1 for n in graph.nodes if not isinstance(graph[n], ConditionalNode)
        for v in graph[n].reads if v.spelling.startswith("^hat.")
    )
    blocking = Blocking.scc(graph)
    depths = []
    for sub in blocking.subgraphs:
        if len(sub.nodes) <= 1:
            continue
        body = runnable(sub)
        deps = body._nx_dependencies
        depths.append(nx.dag_longest_path_length(deps) + 1 if body.nodes else 0)
    return {
        "copies": len(hats),
        "reads_rerouted": reads,
        "problems": sum(1 for n in graph.nodes if isinstance(graph[n], ConditionalNode)),
        "coupled_blocks": len(depths),
        "body_depths": depths,
        "max_body_depth": max(depths, default=0),
    }


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    chosen = [argv[i + 1] for i, a in enumerate(argv) if a == "--input"] or list(CONFIGURATIONS)
    rows, tex_rows, detail = [], [], {}
    for path in chosen:
        live = session.open_session(path)
        raw = raw_graph(live)
        base = census_raw(raw)
        name = stem(path)
        detail[name] = {"raw": base, "cuts": {}}
        per_cut = {}
        for recipe in RECIPES:
            cut = cut_for(recipe)
            from functional_process.cottax.mda import cut_graph  # noqa: PLC0415

            graph = (cut_graph if cut is None else cut)(raw)
            per_cut[recipe] = census_cut(graph)
            detail[name]["cuts"][recipe] = per_cut[recipe]
            rows.append({"configuration": name, "recipe": recipe, **base, **per_cut[recipe]})
        print(f"{name:24} nodes {base['nodes']:4} components {base['component_sizes']} "
              f"copies " + " ".join(f"{r}={per_cut[r]['copies']}/{per_cut[r]['max_body_depth']}" for r in RECIPES))
        tex_rows.append([
            tex_name(name), base["nodes"], base["boundary_inputs"],
            ", ".join(str(s) for s in base["component_sizes"]),
            base["root_finds"] + base["fixed_points"],
            *(f"{per_cut[r]['copies']} / {per_cut[r]['max_body_depth']}" for r in RECIPES),
        ])
    write_csv("graph_census.py", rows)
    write_json("graph_census.py", detail)
    write_tex(
        "graph_census.py",
        ["configuration", "nodes", "inputs", "SCC sizes", "declared", *(LABEL[r] for r in RECIPES)],
        tex_rows,
        align="lrrlr" + "r" * len(RECIPES),
        caption_note="per recipe: copies minted / deepest coupled-block body (nodes in sequence per iterate)",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
