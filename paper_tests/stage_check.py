"""Stage check for OUU: which build quantities actually vary per belief sample.

`architectures.stages` does the work -- `leaves` sorts the graph's boundary inputs by
`configurations.kinds.KINDS`, `split` finds every node downstream of a belief or an
operating leaf (`reach`), `violations` names every claimed rule-closed build output
(`kinds.CLAIMED_BUILD_OUTPUTS`) whose owner is second stage -- a non-anticipativity
violation -- and `report` is the table. This script runs the three leaf sets of the
2026-09-17 handoff (`all`: every belief and operating leaf; `sampled`: the rows
`kinds.BELIEFS` draws and the two recourse variables; `build`: the same with
`BUILD_LEAVES` held at nominal) on the graph the handoff measured -- the driven MDA
graph *with* `.vacuum.duct_diameter_root_find`, 156 nodes -- against the claims it
had then (sections 1a-1d of `output_kinds.md`: 27/156 & 13, 36/156 & 13, 60/156 & 4),
and writes `out/stage_check.{md,json}`. `--runnable` measures the 154-node graph the
port runs instead, `--lifetimes` adds the two section-4 lifetimes `kinds.py` claimed
afterwards (two more violations in every row, both `Decision.RECOURSE`);
`tests/architectures/test_stages.py` states all four combinations.

    $PY paper_tests/stage_check.py [--runnable] [--lifetimes] [--configuration stellarator_helias]
"""

from __future__ import annotations

import sys
import time

from common import OUT, option, write_json

from functional_process.configurations import kinds, load
from functional_process.cottax.architectures import mda, stages
from functional_process.cottax.architectures.evaluate import without_excluded
from functional_process.cottax.input.indat import graph_for

SETS = ("all", "sampled", "build")


def leaf_sets(graph) -> dict:
    """The three leaf sets of the handoff's table, on `graph`."""
    return {
        "all": stages.leaves(graph),
        "sampled": stages.leaves(
            graph, sampled=stages.sampled_paths(), varying=stages.RECOURSE_PATHS
        ),
        "build": stages.leaves(
            graph,
            sampled=stages.sampled_paths(held=kinds.BUILD_LEAVES),
            varying=stages.RECOURSE_PATHS,
        ),
    }


def truncate(items, n=8):
    items = list(items)
    if len(items) <= n:
        return ", ".join(items)
    return ", ".join(items[:n]) + f" +{len(items) - n} more"


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--help" in argv or "-h" in argv:
        print(__doc__)
        return 0
    began = time.perf_counter()
    name = option(argv, "--configuration", "stellarator_helias", str)
    runnable = "--runnable" in argv
    claims = {
        s: sec
        for s, sec in kinds.CLAIMED_BUILD_OUTPUTS.items()
        if "--lifetimes" in argv or sec != "4"
    }
    configuration = load(name)
    raw = graph_for(configuration.machine)
    g = mda.driven_graph(without_excluded(raw) if runnable else raw)
    print(
        f"graph: {len(g.nodes)} nodes, {len(g.graph.boundary_inputs)} boundary inputs "
        f"({'runnable' if runnable else 'as measured by the handoff'}; "
        f"{time.perf_counter() - began:.0f}s)"
    )
    sets = leaf_sets(g)
    print(
        "belief sets: "
        + ", ".join(f"{k}={len(v.belief)}+{len(v.operating)}" for k, v in sets.items())
    )
    print(f"recourse (per-sample operating point): {list(stages.RECOURSE_PATHS)}")

    summary_rows, violation_tables, clean_tables = [], {}, {}
    print("\nsummary:")
    for label in SETS:
        split = stages.split(g, sets[label])
        hits, clean = stages.violations(g, split, claims)
        violation_tables[label] = [
            {
                "place": v.place,
                "section": v.section,
                "producing_node": v.owner.spelling,
                "stage": v.stage.value,
                "n_responsible": v.n_responsible,
                "responsible": list(v.responsible),
            }
            for v in hits
        ]
        clean_tables[label] = [
            {"place": c.place, "section": c.section, "producing_node": c.owner.spelling}
            for c in clean
        ]
        counts = split.counts
        summary_rows.append({
            "belief_set": label,
            "n_leaves": len(sets[label].belief),
            "n_operating": len(sets[label].operating),
            "nodes_first": counts[stages.Stage.FIRST].nodes,
            "nodes_second": counts[stages.Stage.SECOND].nodes,
            "nodes_recourse": counts[stages.Stage.RECOURSE].nodes,
            "owned_first": counts[stages.Stage.FIRST].owned,
            "owned_second": counts[stages.Stage.SECOND].owned,
            "owned_recourse": counts[stages.Stage.RECOURSE].owned,
            "first_stage_fraction_nodes": split.first_stage_fraction,
            "n_violations": len(hits),
            "n_confirmed_first_stage": len(clean),
        })
        print(stages.report(split, hits, label).splitlines()[0])

    for label in ("sampled", "build"):
        print(f"\nviolations, belief set {label!r} ({len(violation_tables[label])}):")
        for v in violation_tables[label]:
            print(
                f"  {v['place']:45s} {v['section']:4s} <- {v['producing_node']:40s} "
                f"({v['n_responsible']} leaves: {truncate(v['responsible'])})"
            )

    payload = {
        "configuration": name,
        "graph": "runnable (154)" if runnable else "as measured (156, duct root find kept)",
        "n_nodes": len(g.nodes),
        "n_boundary_inputs": len(g.graph.boundary_inputs),
        "claims": "sections 1a-1d and 4" if "--lifetimes" in argv else "sections 1a-1d (the handoff's)",
        "n_claimed_build_outputs": len(claims),
        "recourse": list(stages.RECOURSE_PATHS),
        "leaves": {k: sorted(v.spelling for v in s.belief) for k, s in sets.items()},
        "summary": summary_rows,
        "violations": violation_tables,
        "confirmed_first_stage": clean_tables,
    }
    write_json("stage_check", payload)

    md = ["# Stage check: non-anticipativity of claimed build outputs\n"]
    md.append(f"Graph: {payload['graph']}, {len(g.nodes)} nodes; claims: "
              f"`kinds.CLAIMED_BUILD_OUTPUTS`, {payload['claims']} ({len(claims)}).\n")
    md.append("## Summary\n")
    md.append("| belief set | leaves | nodes 1st | nodes 2nd | nodes recourse | owned 1st | "
              "owned 2nd | 1st-stage frac | violations |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for row in summary_rows:
        md.append(f"| {row['belief_set']} | {row['n_leaves']} + {row['n_operating']} | "
                  f"{row['nodes_first']} | {row['nodes_second']} | {row['nodes_recourse']} | "
                  f"{row['owned_first']} | {row['owned_second']} | "
                  f"{row['first_stage_fraction_nodes']:.2f} | {row['n_violations']} |")
    for label in SETS:
        md.append(f"\n## Violations -- belief set '{label}' ({len(violation_tables[label])})\n")
        md.append("| place | section | producing node | stage | # leaves | leaves |")
        md.append("|---|---|---|---|---|---|")
        for v in violation_tables[label]:
            md.append(f"| `{v['place']}` | {v['section']} | `{v['producing_node']}` | "
                      f"{v['stage']} | {v['n_responsible']} | {truncate(v['responsible'])} |")
        md.append(f"\n## Confirmed first-stage -- belief set '{label}' "
                  f"({len(clean_tables[label])})\n")
        md.append("| place | section | producing node |")
        md.append("|---|---|---|")
        for c in clean_tables[label]:
            md.append(f"| `{c['place']}` | {c['section']} | `{c['producing_node']}` |")
    (OUT / "stage_check.md").write_text("\n".join(md) + "\n")
    print(f"\nwrote {OUT / 'stage_check.json'} and {OUT / 'stage_check.md'} "
          f"({time.perf_counter() - began:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
