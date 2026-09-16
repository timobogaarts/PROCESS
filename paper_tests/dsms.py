"""The interactive DSM pages -- the GitHub Pages kind (`grouping.render_grouped_dsm_html`,
self-contained, no ragraph) -- for every configuration and every architecture.

Per configuration, under `out/dsm/<configuration>/`, each drawn twice -- ordered by
provenance (namespace bands, the top level's coupled blocks boxed) and by structure (the
run order at every nesting level, every solve boxed inside the solve it is nested in,
ringed by the kind of problem it answers and labelled with its driver):

    uncut               the graph as `indat` declares it: no cut, no optimiser -- what
                        the Pages site shows
    uncut_optimiser     the same plus the file's `Optimise` (or `RootFind`) node,
                        uncut: the optimiser's cycle is one SCC over most of the machine
    mdf_<cut>           MDF stated as structure -- the graph cut by <cut>, the optimiser
                        nested around everything it iterates (`mdf.nested_blocking`)
    sand_<cut>          SAND -- every problem residualised and combined into one,
                        as structure (no MDA is run; nothing is dropped)

for <cut> in hand, jacobi, gauss_seidel, gauss_seidel_minimal. A root-find file (the two
`*_eval`) has an `uncut_optimiser` with its `RootFind` and an `mdf_<cut>` stated in the
graph, and no SAND. An `index.html` per configuration and one on top.

    $PY paper_tests/dsms.py [--input <IN.DAT>]... [--only uncut,mdf,sand]
"""

from __future__ import annotations

import sys
import traceback

from common import CONFIGURATIONS, LABEL, OUT, RECIPES, cut_for, provenance, stem
import jax
from cottax.blocking import Blocking
from cottax.names import PathMap
from cottax.plan import Insert, Plan
from cottax.problem import RootFind

from functional_process.cottax import mdf, sand, session
from functional_process.cottax.indat import graph_for
from functional_process.cottax.mda import assign_drivers, cut_graph, default_drivers
from functional_process.cottax.mda_harness import _without_excluded
from functional_process.cottax.render_xdsm import SPELLING
from functional_process.cottax.run_cold_matrix import (
    _return_freed_memory_to_the_os,
    build_mdf,
)
from functional_process.cottax.visualization.grouping import (
    dependency_group_sequence,
    provenance_order,
    render_grouped_dsm_html,
    structure_order,
)


def draw(blocking: Blocking, outdir, name: str, title: str) -> list[str]:
    """Both orderings of one blocking; returns the two file names."""
    graph = blocking.graph
    axis = dependency_group_sequence(graph, depth=None)
    common = {"depth": None, "outdir": str(outdir), "write": True, "formatter": SPELLING}
    render_grouped_dsm_html(
        blocking,
        order=provenance_order(graph.nodes, depth=None, owners=graph.owners, groups=axis),
        title=f"{title} -- ordered by provenance",
        file_name=f"{name}_provenance",
        mode="provenance",
        **common,
    )
    render_grouped_dsm_html(
        blocking,
        order=structure_order(blocking),
        title=f"{title} -- ordered by structure (run order, solves nested)",
        file_name=f"{name}_scc",
        mode="structure",
        **common,
    )
    return [f"{name}_provenance.html", f"{name}_scc.html"]


def raw_graph(live):
    return _without_excluded(live.machine_graph if live.machine_graph is not None else graph_for())


def uncut_optimiser(live, raw):
    """The raw graph with the file's own problem node inserted, uncut -- a picture, so
    built without the executability checks an assembly runs (an uncut cycle has no
    problem to drive it, which is exactly what this page shows)."""
    ref = live.reference
    if live.root_find:
        graph, _conditions, _n, report = mdf.mdf_graph(
            raw, ref.icc, ref.n_equality, None, live.switch_values
        )
        node = RootFind(
            conditions=tuple(report["equalities"]),
            unknowns=tuple(sand.iteration_variable_path(i) for i in ref.ixc),
        )
        graph = (Plan(graph) + Insert(PathMap(((mdf.IN_GRAPH_PLACE, node),)))).graph
        return Blocking.scc(graph)
    with_problem, _name, _report = sand.optimise_graph(
        raw, ref.ixc, ref.icc, ref.n_equality, ref.i_figure_merit,
        switch_values=live.switch_values,
    )
    return Blocking.scc(with_problem)


def mdf_blocking(live, recipe: str):
    ref = live.reference
    cut = cut_for(recipe) or cut_graph
    if live.root_find:
        build = build_mdf(ref, live.machine_graph, live.switch_values, root_find=True, cut=cut)
        return build.in_graph.blocking
    blocking, _name, _report = mdf.nested_blocking(
        ref.ixc, ref.icc, ref.n_equality, ref.i_figure_merit,
        graph=live.machine_graph, cut=cut, switch_values=live.switch_values,
    )
    # `nested_blocking` states the structure and no algorithm; the page names the driver
    # of every solve, so the default one is `Assign`ed on -- the same choice `mdf.assemble`
    # makes for the inner problems, and the file's own `VmconDriver` for the `Optimise`.
    # `Assign` carries `within`, so the nesting survives and the blocking is re-read.
    graph = blocking.graph
    return Blocking.scc(assign_drivers(graph, default_drivers(graph)))


def sand_blocking(live, recipe: str):
    """SAND as structure: the cut graph, the optimiser inserted, every declared problem
    residualised and combined into one, drivers assigned -- no MDA run. `build_sand`
    runs one to find fixed points to *drop* (degenerate, array-valued), and a SAND that
    drops nothing is the one the recipes solve, so the picture is that graph."""
    ref = live.reference
    cut = cut_for(recipe) or cut_graph
    with_problem, _name, _report = sand.optimise_graph(
        cut(raw_graph(live)), ref.ixc, ref.icc, ref.n_equality, ref.i_figure_merit,
        switch_values=live.switch_values,
    )
    combined, _residualised = sand.sand_graph(with_problem)
    return Blocking.scc(assign_drivers(combined, default_drivers(combined)))


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    chosen = [argv[i + 1] for i, a in enumerate(argv) if a == "--input"] or list(CONFIGURATIONS)
    only = set((argv[argv.index("--only") + 1] if "--only" in argv else "uncut,optimiser,mdf,sand").split(","))
    top = OUT / "dsm"
    top.mkdir(exist_ok=True)
    top_links = []
    for path in chosen:
        name = stem(path)
        outdir = top / name
        outdir.mkdir(exist_ok=True)
        live = session.open_session(path)
        raw = raw_graph(live)
        pages: list[tuple[str, list[str]]] = []
        jobs = []
        if "uncut" in only:
            jobs.append(("uncut", "the declared graph, uncut", lambda: Blocking.scc(raw)))
        if "optimiser" in only:
            jobs.append(("uncut_optimiser", "uncut, with the file's problem inserted",
                         lambda: uncut_optimiser(live, raw)))
        for recipe in RECIPES:
            if "mdf" in only:
                jobs.append((f"mdf_{recipe}", f"MDF, cut: {LABEL[recipe]}",
                             lambda r=recipe: mdf_blocking(live, r)))
            if "sand" in only and not live.root_find:
                jobs.append((f"sand_{recipe}", f"SAND, cut: {LABEL[recipe]}",
                             lambda r=recipe: sand_blocking(live, r)))
        for key, label, build in jobs:
            try:
                blocking = build()
                files = draw(blocking, outdir, key, f"PROCESS port, {name} -- {label}")
                sizes = sorted((len(b) for b in blocking.blocks if len(b) > 1), reverse=True)
                pages.append((f"{label} (coupled blocks: {sizes or 'none'})", files))
                print(f"{name:24} {key:28} blocks>1 {sizes}", flush=True)
            except Exception as failure:  # noqa: BLE001 -- a page, not an exit
                pages.append((f"{label}: FAILED {type(failure).__name__}: {str(failure)[:160]}", []))
                print(f"{name:24} {key:28} FAILED {type(failure).__name__}: {failure}", flush=True)
                if "--trace" in argv:
                    traceback.print_exc()
        items = "\n".join(
            f"<li>{label}" + (
                f' -- <a href="{files[0]}">provenance</a> | <a href="{files[1]}">structure</a>'
                if files else "") + "</li>"
            for label, files in pages
        )
        (outdir / "index.html").write_text(
            f"<!doctype html><meta charset='utf-8'><title>{name} DSMs</title>"
            f"<!-- {provenance('dsms.py')} --><h1>{name}</h1><ul>{items}</ul>\n"
        )
        top_links.append(f'<li><a href="{name}/index.html">{name}</a></li>')
        # Every SAND assembly compiles an MDA; jax keeps each for the life of the
        # process and LLVM's section memory runs out on the seventh configuration.
        jax.clear_caches()
        _return_freed_memory_to_the_os()
    # The top index lists every configuration that has pages, not only this run's.
    top_links = [
        f'<li><a href="{d.name}/index.html">{d.name}</a></li>'
        for d in sorted(top.iterdir()) if d.is_dir() and (d / "index.html").exists()
    ]
    (top / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>PROCESS port DSMs</title>"
        f"<!-- {provenance('dsms.py')} --><h1>PROCESS port DSMs</h1>"
        "<p>Per configuration: the uncut graph, the uncut graph with its optimiser, and the "
        "MDF and SAND architectures under each cut recipe; each ordered by provenance and by "
        "structure. Hover a mark for the variables it carries.</p><ul>"
        + "\n".join(top_links) + "</ul>\n"
    )
    print(f"index: {top / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
