"""What each variant makes of each machine, built and not solved: the graph's size, the
problems and their nesting, what the optimiser holds -- the columns of
`../architectures/structure.py`, per variant -- and, at the corners, whether the
variant's graph is the preset's (`cx.absolute.mdf` / `idf` / `sand`). A variant cottax
refuses (an op whose check fails) is a row too, its refusal in `refused`.

Writes `out/structure_minimal/<configuration>.csv` and `out/variants.txt` (one
`configuration label` per line, the sweep `run_all.sh` walks), and exits non-zero if a
corner is not its preset: nothing should be timed then.

    $PY paper_tests/architecture_variants/structure.py [--configurations ...]
"""

from __future__ import annotations

import sys

import common
from common import bench

import cottax.assemblies.jax as cx  # noqa: E402
from cottax.core import Plan, is_problem  # noqa: E402

from functional_process import configurations  # noqa: E402
from functional_process.cottax import study  # noqa: E402
from functional_process.cottax.problem import TOP, problem_model  # noqa: E402

import variants as V  # noqa: E402

arch_structure = common.load("architectures_structure", common.ARCH / "structure.py")

PRESETS = {"MDF": cx.absolute.mdf, "IDF": cx.absolute.idf, "SAND": cx.absolute.sand}


def corner_check(configuration, variant) -> str:
    """`equal` where the variant's graph is every preset it stands for, else which is
    not; empty for a hybrid."""
    if not variant.preset:
        return ""
    base = Plan(problem_model(configuration).graph)
    mine = (base + V.hybrid(variant)).graph
    unequal = [p for p in variant.preset.split("/") if p in study.arms(configuration)
               and (base + PRESETS[p](TOP, study.RULE)).graph != mine]
    return "equal" if not unequal else "differs from " + "/".join(unequal)


def main():
    args = common.arguments(__doc__)
    listing, failed = [], []
    for name in args.configurations:
        configuration = configurations.load(name)
        sw = V.switches(configuration)
        closures = cx.absolute.mda(study.RULE).resolution(study.machine_model(configuration).graph).closures
        rows = []
        for v in V.variants(configuration):
            row = {
                "configuration": name,
                "problem": "root-find" if bench.is_root_find(configuration) else "optimise",
                "label": v.label,
                "preset": v.preset,
                "blocks": "|".join(map(str, sw.blocks)),
                "implicit": "|".join(map(str, sw.implicit)),
                "cycles": len(closures),
                "cut_variables": sum(len(c.cuts) for c in closures),
            }
            try:
                graph = V.variant_plan(configuration, v, {}, []).graph
                row.update({
                    "refused": "",
                    "nodes": len(graph.nodes),
                    "problems": sum(is_problem(graph[n]) for n in graph.nodes),
                    "nested": len(graph.within),
                    "steps": len(study.proved(graph).steps),
                    **arch_structure.top_of(graph),
                    "corner": corner_check(configuration, v),
                })
            except Exception as e:  # noqa: BLE001 -- a refusal is a result here
                row.update({"refused": f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"})
            if row.get("corner", "").startswith("differs"):
                failed.append(f"{name} {v.label}: {row['corner']}")
            rows.append(row)
            listing.append(f"{name} {v.label}")
            print(row, flush=True)
        common.write_all("structure.py", common.path_of(f"structure_{common.SCHEME}", name), rows)
    if set(args.configurations) == set(bench.NAMES):
        (common.OUT / "variants.txt").write_text("\n".join(listing) + "\n")
        print(f"wrote {common.shown(common.OUT / 'variants.txt')} ({len(listing)} variants)")
    if failed:
        print("CORNER CHECK FAILED:\n  " + "\n  ".join(failed), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
