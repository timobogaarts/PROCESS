"""Render the port's graphs as self-contained, interactive XDSM/DSM HTML pages."""

import os
import re
import sys
from functools import partial
from pathlib import Path

from cottax.interfaces.pytree_namespace_module import xDSMFormatterFlat
from cottax.visualization import render_xdsm_html

# `render_dsm_html` used to be re-exported by `cottax.visualization` itself; jaxgraph's
# "theory 2 notebook" commit (b1a2bbc) dropped `ragraph_dsm`'s imports from that
# `__init__.py` as unrelated notebook cleanup, and the module still defines the function
# -- only the package-level re-export went missing. Importing the submodule directly
# survives that either way and needs no jaxgraph edit (out of scope here: this file's
# territory is `functional_process/`, not `~/jaxgraph`).
from cottax.visualization.ragraph_dsm import render_dsm_html

from functional_process.cottax.boundary import TOKAMAK_INPUT_FILE
from functional_process.cottax.indat import GRAPH, graph_for, machine_from_indat

OUTDIR = Path(__file__).parent

SPELLING = xDSMFormatterFlat()
"""How every diagram here writes a name."""


def machine_graph(input_file: str | None = None):
    """`(graph, file-name suffix)` for the machine asked for; the reference if none."""
    if input_file is None:
        return GRAPH, ""
    if os.path.normpath(input_file) == os.path.normpath(TOKAMAK_INPUT_FILE):
        suffix = "_tokamak"
    else:
        suffix = "_" + Path(input_file).name.removesuffix(".IN.DAT").lower()
    return graph_for(machine_from_indat(input_file)), suffix


def machine_label(suffix: str) -> str:
    """How a diagram's title names the device `suffix` belongs to."""
    return "reference stellarator" if not suffix else suffix[1:].replace("_", " ")


def main(input_file: str | None = None):
    """Write `xdsm.html`/`dsm.html` for the model graph; return the XDSM path."""
    from cottax import Blocking

    graph, suffix = machine_graph(input_file)
    render_xdsm_html(
        Blocking.fused(graph),
        file_name=f"xdsm{suffix}",
        outdir=str(OUTDIR),
        write=True,
        collapse_names=True,
        blocks=True,
        collapse_models=True,
        formatter=SPELLING,
    )
    render_dsm_html(
        Blocking.fused(graph),
        file_name=f"dsm{suffix}",
        outdir=str(OUTDIR),
        write=True,
        formatter=SPELLING,
    )
    return OUTDIR / f"xdsm{suffix}.html"


# ==================================================== provenance, until the tree lands
_MODELS = "functional_process.cottax."
_SPLIT_FILE = re.compile(r"_[A-Z](?:_.*)?$")


def grouped(depth: int | None = None, input_file: str | None = None):
    """Write `dsm_provenance.html`/`dsm_scc.html`: § 11's comparison, drawn."""
    from cottax import Blocking
    from functional_process.cottax.visualization.grouping import (
        dependency_group_sequence,
        group_label,
        grouping_report,
        provenance_order,
        render_grouped_dsm_html,
        structure_order,
    )

    from functional_process.cottax.mda import driven_graph

    # The model tree is real now, so node names already carry their own prefix --
    # `by_subsystem`'s module-derived relabelling was scaffolding for before it
    # landed and is gone. `group_of` reads the grouping straight off the name.
    declared, suffix = machine_graph(input_file)
    graph = driven_graph(declared)
    blocking = Blocking.scc(graph)
    report = grouping_report(blocking, depth=depth)
    print(f"grouped ({machine_label(suffix)}): {report.summary()}")

    # The group *axis* of the provenance picture, in dependency order rather than
    # declaration order -- see `dependency_group_sequence`. Declaration order put `costs`
    # first (it is the first slot written in the top-level namespace), which drew the one
    # subsystem nothing reads from at the top left of the matrix and read as "everything
    # depends on costs". Row order *within* a group is still declaration order, so the
    # provenance claim is unchanged where it is actually about how the file was written.
    axis = dependency_group_sequence(graph, depth=depth)
    print("  group axis: " + " -> ".join(group_label(g) for g in axis))
    # Both kinds are printed. `crossing` is now the strict question -- a block no
    # namespace contains -- so on the reference machine it is empty, and printing only
    # it would make the one genuinely multi-namespace loop invisible at exactly the
    # moment the measurement got sharper. `nesting` is that loop: several namespaces,
    # one subtree.
    for block in report.crossing:
        print(
            "  CROSSES: "
            + ", ".join(sorted(".".join(g) for g in block.named_groups))
            + " -- "
            + ", ".join(sorted(SPELLING.node((m, graph[m])) for m in block.members))
        )
    for block in report.nesting:
        print(
            f"  within {group_label(block.container)}: "
            + ", ".join(sorted(".".join(g) for g in block.named_groups))
            + " -- "
            + ", ".join(sorted(SPELLING.node((m, graph[m])) for m in block.members))
        )

    common = {
        "depth": depth,
        "outdir": str(OUTDIR),
        "write": True,
        "formatter": SPELLING,
    }
    render_grouped_dsm_html(
        blocking,
        order=provenance_order(
            graph.nodes, depth=depth, owners=graph.owners, groups=axis
        ),
        title=f"PROCESS port, {machine_label(suffix)} -- ordered by provenance",
        file_name=f"dsm_provenance{suffix}",
        **common,
    )
    render_grouped_dsm_html(
        blocking,
        order=structure_order(blocking),
        title=f"PROCESS port, {machine_label(suffix)} -- ordered by structure (SCC)",
        file_name=f"dsm_scc{suffix}",
        **common,
    )
    return OUTDIR / f"dsm_provenance{suffix}.html"


def grouped_uncut(depth: int | None = None, input_file: str | None = None):
    """Write `dsm_provenance_uncut.html`/`dsm_scc_uncut.html`: `grouped`'s comparison,
    but of the graph exactly as `machine_graph` returns it -- **no `driven_graph`, no
    `cut_graph`, no `assign_drivers`, no problem that `indat.py` did not already
    declare**.
    """
    from cottax import Blocking
    from functional_process.cottax.visualization.grouping import (
        dependency_group_sequence,
        group_label,
        grouping_report,
        provenance_order,
        render_grouped_dsm_html,
        structure_order,
    )

    declared, suffix = machine_graph(input_file)
    blocking = Blocking.scc(declared)
    report = grouping_report(blocking, depth=depth)
    print(f"grouped_uncut ({machine_label(suffix)}): {report.summary()}")

    axis = dependency_group_sequence(declared, depth=depth)
    print("  group axis: " + " -> ".join(group_label(g) for g in axis))
    # Every genuinely coupled block is printed here, not just the crossing/nesting
    # subsets `grouped` prints -- this function's whole point is the census of
    # coupling itself, so a block confined to one group (e.g. the stellarator's
    # divertor/fw_area pair) is exactly as much the answer as one that spans several.
    for block in report.coupled:
        kind = (
            "CROSSES"
            if block.crosses
            else f"within {group_label(block.container)}"
            if block.nests
            else "single-group"
        )
        print(
            f"  {kind}: "
            + ", ".join(sorted(".".join(g) for g in block.named_groups))
            + " -- "
            + ", ".join(sorted(SPELLING.node((m, declared[m])) for m in block.members))
        )

    common = {
        "depth": depth,
        "outdir": str(OUTDIR),
        "write": True,
        "formatter": SPELLING,
    }
    render_grouped_dsm_html(
        blocking,
        order=provenance_order(
            declared.nodes, depth=depth, owners=declared.owners, groups=axis
        ),
        title=f"PROCESS port, {machine_label(suffix)} "
        "-- UNCUT graph, ordered by provenance",
        file_name=f"dsm_provenance_uncut{suffix}",
        **common,
    )
    render_grouped_dsm_html(
        blocking,
        order=structure_order(blocking),
        title=f"PROCESS port, {machine_label(suffix)} "
        "-- UNCUT graph, SCC membership (mutual coupling, not a run order)",
        file_name=f"dsm_scc_uncut{suffix}",
        **common,
    )
    return OUTDIR / f"dsm_provenance_uncut{suffix}.html"


def cold_reference(input_file=None):
    """What the SAND assembly needs, from the input file alone -- **no solve**."""
    from types import SimpleNamespace

    from process.main import SingleRun

    from functional_process.cottax.sand_harness import (
        REFERENCE_INPUT_FILE,
        _scratch_copy,
    )

    data = SingleRun(_scratch_copy(input_file or REFERENCE_INPUT_FILE), "vmcon").data
    n = int(data.numerics.n_iteration_variables)
    m = int(data.numerics.n_equality_constraints) + int(
        data.numerics.n_inequality_constraints
    )
    return SimpleNamespace(
        data=data,
        ixc=[int(i) for i in data.numerics.ixc[:n]],
        icc=[int(i) for i in data.numerics.icc[:m]],
        n_equality=int(data.numerics.n_equality_constraints),
        i_figure_merit=int(data.numerics.i_figure_merit),
    )


def sand():
    """Write `xdsm_sand.html`/`dsm_sand.html` for the assembled SAND graph."""
    from cottax import Blocking

    from functional_process.cottax.sand_harness import assemble, mda_env

    reference = cold_reference()
    driven, env = mda_env(reference)
    combined, report = assemble(reference, driven, env)
    print(
        f"SAND graph: {len(combined.nodes)} nodes | "
        f"degenerate fixed points dropped: {len(report['degenerate'])} | "
        f"residualised: {len(report['residualised'])}"
    )
    if report["omitted"]:
        print(f"  CONSTRAINTS OMITTED: {report['omitted']}")

    # `Blocking.scc`, **not** `Blocking.fused` -- and the difference is the difference
    # between a diagram that tells the truth and one that libels the graph.
    #
    # Measured on this graph: `fused` gives **3 blocks** (130 + 21 + 20 nodes) and
    # `scc` gives **42**, of which exactly **one** has more than a single node. `fused`
    # lumps independent work together, so every edge inside a lump renders as a
    # coupling and the picture reads as though most of the design were mutually
    # entangled. It is not.
    #
    # What is genuinely cyclic here is one thing and one thing only: the `Optimise`
    # node. It owns the 23 unknowns and reads the 30 conditions, so it closes a loop
    # around the entire MDA -- hence a 130-node SCC. Drop that single node, which is
    # exactly what `ConditionMap.body` does, and the remaining 129 are **acyclic**
    # (`drive.body.is_acyclic is True`, measured): `_run_acyclic` topologically sorts
    # them and runs them in one pass per evaluation. So the loop in the diagram is the
    # optimiser's iteration, not feedback among the models.
    #
    # Both groupings are written, because they answer different questions and the
    # difference between them is itself informative:
    #   `xdsm_sand`       -- `scc`, the finest honest partition: 42 blocks, 41 of them
    #                        single nodes in dependency order, one genuinely coupled.
    #   `xdsm_sand_fused` -- `fused`, the execution shape: one box per solve and one per
    #                        run of ordinary nodes between them (here 20 / 130 / 21).
    # `fused` is the more readable framing and used to be the misleading one: a run
    # block was drawn as a lump with no internal order even though it is a totally
    # ordered chain with **zero** coupling (`Blocking.scc` on either gives 20 and 21
    # singletons). `cottax.visualization` sequences a problem-free block's interior
    # now, so both are truthful and the choice is about what you want to see.
    for name, blocking in (("xdsm_sand", Blocking.scc(combined)),):
        render_xdsm_html(
            blocking,
            file_name=name,
            outdir=str(OUTDIR),
            write=True,
            collapse_names=True,
            blocks=True,
            collapse_models=True,
            formatter=SPELLING,
        )
        render_dsm_html(
            blocking,
            file_name=f"dsm_{name[5:]}",
            outdir=str(OUTDIR),
            write=True,
            formatter=SPELLING,
        )
        print(f"  {name}: {len(blocking.blocks)} blocks")
    render_dsm_html(
        Blocking.scc(combined),
        file_name="dsm_sand",
        outdir=str(OUTDIR),
        write=True,
        formatter=SPELLING,
    )
    return OUTDIR / "xdsm_sand.html"


MODES = {"sand": sand, "grouped": grouped, "grouped_uncut": grouped_uncut}

USAGE = """usage: python -m functional_process.cottax.render_xdsm [sand|grouped|grouped_uncut]
                                                [--machine [IN.DAT]]

  (no argument)  the declared model graph  -> xdsm.html, dsm.html        (fast)
  sand           the assembled SAND graph  -> xdsm_sand.html,
                                              dsm_sand.html
  grouped        provenance vs. structure  -> dsm_provenance.html,
                                              dsm_scc.html               (fast)
  grouped_uncut  the same pair, undriven,  -> dsm_provenance_uncut.html,
                 uncut                        dsm_scc_uncut.html         (fast)

  --machine [IN.DAT]   draw that machine instead of the reference stellarator,
                       into `*_tokamak.html` (default: the conventional large
                       tokamak). Not available for `sand`.
"""

MACHINE_FLAG = "--machine"
"""Spelled as `boundary.py` spells it, and only in that one long form."""


def _machine_argument(argv):
    """`(input file or None, argv without the machine argument)`."""
    if MACHINE_FLAG not in argv:
        return None, argv

    at = argv.index(MACHINE_FLAG)
    rest = argv[at + 1 :]
    if rest and not rest[0].startswith("-") and rest[0].lstrip("-") not in MODES:
        return rest[0], argv[:at] + rest[1:]
    return TOKAMAK_INPUT_FILE, argv[:at] + rest


def mode(argv):
    """Which renderer `argv` asks for, as the function to call."""

    machine, argv = _machine_argument(list(argv))
    asked = [MODES[a.lstrip("-")] for a in argv if a.lstrip("-") in MODES]
    rest = [a for a in argv if a.lstrip("-") not in MODES]
    if rest:
        raise SystemExit(f"unrecognised argument(s): {' '.join(rest)}\n\n{USAGE}")
    if len(asked) > 1:
        raise SystemExit(f"one mode at a time, not {len(asked)}\n\n{USAGE}")
    chosen = asked[0] if asked else main
    if machine is None:
        return chosen
    if chosen is sand:
        raise SystemExit(
            f"`sand` draws the reference stellarator's solve only; "
            f"{MACHINE_FLAG} is not available for it\n\n{USAGE}"
        )
    return partial(chosen, input_file=machine)


def wants_sand(argv) -> bool:
    """Whether `argv` asks for the SAND graph. Kept for callers that predate `mode`."""
    chosen = mode(argv)
    return getattr(chosen, "func", chosen) is sand


if __name__ == "__main__":
    path = mode(sys.argv[1:])()
    print(f"wrote {path}")
