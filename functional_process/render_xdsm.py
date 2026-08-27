"""Render the port's graphs as self-contained, interactive XDSM/DSM HTML pages.

Run directly, from the repo root, in the `process_port` env:

    $PY -m functional_process.render_xdsm            # xdsm.html, dsm.html
    $PY -m functional_process.render_xdsm grouped    # dsm_provenance.html, dsm_scc.html
    $PY -m functional_process.render_xdsm sand       # xdsm_sand.html, dsm_sand.html

Either fast mode takes `--machine [<IN.DAT>]`, which draws a *different device* and
writes to its own suffixed files:

    $PY -m functional_process.render_xdsm --machine          # xdsm_tokamak.html,
                                                            # dsm_tokamak.html
    $PY -m functional_process.render_xdsm grouped --machine  # dsm_provenance_tokamak,
                                                            # dsm_scc_tokamak

`$PY` is deliberately not spelled out: **the conda root differs per machine**
(`~/miniconda3` on one, `~/miniconda` on another -- `CLAUDE.md` records both), and this
docstring used to hardcode the wrong one, so copy-pasting it gave
`No such file or directory` and read as "the renderer is broken". Set it once:

    PY=~/miniconda3/envs/process_port/bin/python   # or ~/miniconda/envs/...

or `conda activate process_port` and use plain `python`. `ls -d ~/miniconda*/envs/
process_port` answers which you have.

The bare form writes `xdsm.html`/`dsm.html` for `indat.GRAPH` -- the declared
model graph, before anything is cut, driven or optimised. Re-run after porting a new
unit to see it join the diagram.

`--machine` is spelled exactly as `boundary.py`'s: bare it means
`boundary.TOKAMAK_INPUT_FILE`, the conventional large tokamak, and it takes an `IN.DAT`
if some other machine is wanted. It needs no new machinery -- `indat.graph_for(
machine_from_indat(...))` has assembled a tokamak since "The tokamak assembles as
written", and every renderer here is a pure function of a `Graph` -- only somewhere to
put the answer, hence the file-name suffix (`machine_graph`). The stellarator's files
keep their unsuffixed names, so nothing that already links to `dsm.html` moves, and the
two devices' pictures sit side by side exactly as their two boundary pins do.

`grouped` writes `dsm_provenance.html`/`dsm_scc.html`: the same graph twice, once with
every subsystem's nodes adjacent and once in the order `Blocking` actually runs, rows
coloured by subsystem in both. That is `switch_elimination_design.md` § 11's
provenance-against-structure comparison as a picture instead of a table -- see `grouped`
below, and `visualization/grouping.py`'s module docstring for what the two of them
show and for why the grouping lives here rather than in `cottax.visualization`.

`sand` writes `xdsm_sand.html`/`dsm_sand.html` for **the graph the SAND solve actually
runs**, which is a different and much more informative object: `GRAPH` with its raw
cycles cut into declared `FixedPoint` problems (`mda.driven_graph`), the run's own
constraints and objective assembled onto it as condition nodes, the structural fixed
points residualised, and one `Optimise` node owning every unknown. That is the picture
of what the optimiser is handed -- which blocks it drives, which quantities are
unknowns, and which conditions it is trying to zero.

It does **not** need a PROCESS solve. `numerics.ixc`/`icc`/`n_equality_constraints`/
`i_figure_merit` are read straight from `IN.DAT` by `SingleRun.__init__`, so parsing the
input file is enough to know which constraints and iteration variables exist -- and the
graph's *structure* is a function of nothing else. The only thing that wants values is
`degenerate_fixed_points`, which asks whether a fixed point is an identity, and a cold
MDA run answers that as well as a converged one.

This used to call `sand_harness.reference_run()` and pay 95 s for a full solve. That was
copied from the harness, where the solve is needed because the harness *compares* against
PROCESS's answers -- a renderer compares against nothing.

`sand` is the reference stellarator's and refuses `--machine`: it assembles a solve, not
just a structure (`ixc`/`icc` off one `IN.DAT`, an MDA env, degeneracy measured on
values), and none of that has been validated for a second device. It is a scope
statement, not a limit of the graph.
"""

import os
import re
import sys
from functools import partial
from pathlib import Path

from cottax.interfaces.pytree_namespace_module import xDSMFormatterFlat
from cottax.visualization import render_dsm_html, render_xdsm_html

from functional_process.boundary import TOKAMAK_INPUT_FILE
from functional_process.indat import GRAPH, graph_for, machine_from_indat

OUTDIR = Path(__file__).parent

SPELLING = xDSMFormatterFlat()
"""How every diagram here writes a name.

`cottax`'s default `NoFormat` spells a `NodePath` the way jax does -- `['Build']`, and
`['physics']['profiles']['DensityProfile']` once names are hierarchical. Both surfaces
name a node with string `DictKey`s, so the brackets and the quotes say nothing a reader
of *this* port does not already know; `xDSMFormatterFlat` drops them and joins with dots.

Measured, not assumed. Over every string in the struct `xdsm_struct` ships for
`Blocking.fused(GRAPH)`: **910** bracket-spelled before, **0** after -- 159 step labels,
159 block-membership entries, 24 in the returned-variable lists and the rest in the
per-variable producer entries. § 13 of `switch_elimination_design.md` records this as
"136 to 0"; the shape of the claim holds and the figure does not, which is what
re-running a measurement is for -- that count was taken on a graph that has since grown
from 143 nodes to 161, and it does not match this count on any grouping of today's.
"""


def machine_graph(input_file: str | None = None):
    """`(graph, file-name suffix)` for the machine asked for; the reference if none.

    The suffix is the whole of what a second device costs the renderers: they are pure
    functions of a `Graph`, so drawing a tokamak is `graph_for(machine_from_indat(...))`
    and somewhere to put the files. Empty for `GRAPH`, so the stellarator's diagrams keep
    the names everything already refers to.

    `_tokamak` is spelled for `TOKAMAK_INPUT_FILE` by name rather than derived from it,
    because that file is *the* tokamak this port measures itself against (it owns a
    boundary pin under the same short name) and `dsm_large_tokamak_eval.html` would say
    less about which device it is. Any other `IN.DAT` labels itself by its own stem --
    unrecognised is not a reason to refuse to draw, only a reason not to invent a name.
    """
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
_MODELS = "functional_process.models."
_SPLIT_FILE = re.compile(r"_[A-Z](?:_.*)?$")


def grouped(depth: int | None = None, input_file: str | None = None):
    """Write `dsm_provenance.html`/`dsm_scc.html`: § 11's comparison, drawn.

    With `input_file` the pair is drawn for that machine instead, into
    `dsm_provenance_tokamak.html`/`dsm_scc_tokamak.html` -- see `machine_graph`. The
    comparison is per-device by construction: provenance is where a node was *written*
    and structure is the order *this* graph runs in, and a machine that occupies
    different slots gets a different answer to the second question with the same answer
    to the first.

    Both pictures are of **the driven graph** (`mda.driven_graph(GRAPH)`), not of `GRAPH`
    -- which is what § 11.1 measured, and the only one where "structure" means anything:
    a raw cycle has no order, so `Blocking.scc` on the undriven graph would be reporting
    on a schedule nothing can run. `driven_graph` is a pure structural transform and
    needs no PROCESS solve, exactly as `sand` does not.

    Everything but the row order is held fixed between the two files, because the row
    order is the entire comparison.

    **Neither page carries any prose**, so what they show is written down here instead:

    - *provenance* is the driven graph with every subsystem's nodes adjacent, subsystems
      ordered by dependency (each contracted to a vertex, that graph SCC'd and sorted,
      mutually coupled subsystems left adjacent) and rows within a subsystem left in
      declaration order. It is **not** a run order. A mark below the diagonal inside one
      colour band is a place where provenance and dependency disagree; one across bands
      is real feedback between subsystems.
    - *scc* is the same graph in `Blocking.scc`'s order -- the order it runs in. A
      subsystem that is also a schedulable unit shows as one unbroken colour band; one
      interleaved with others shows as stripes, and is a label rather than a module.

    The summary line both pages used to print is what this function prints to stdout.
    """
    from cottax import Blocking
    from functional_process.visualization.grouping import (
        dependency_group_sequence,
        group_label,
        grouping_report,
        provenance_order,
        render_grouped_dsm_html,
        structure_order,
    )

    from functional_process.mda import driven_graph

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


def cold_reference(input_file=None):
    """What the SAND assembly needs, from the input file alone -- **no solve**.

    `assemble` reads exactly four things off a reference: `ixc`, `icc`, `n_equality` and
    `i_figure_merit`. All four live in `numerics` and are parsed out of `IN.DAT` by
    `SingleRun.__init__`; none is produced by solving. Verified on
    `stellarator_helias.IN.DAT` without ever calling `.run()`: `ixc` `[2, 3, 4, 6, 10,
    56, 59, 109]`, `icc` `[2, 16, 24, 8, 17, 18, 67, 82, 83, 62, 32, 34, 35, 65]`,
    `n_equality` 2, `i_figure_merit` 6 -- the same values `reference_run()` reports after
    95 s of VMCON.

    Returned as a namespace rather than a `ReferenceRun` deliberately: a `ReferenceRun`
    carries converged values, timings and a live `Models`, and promising those here would
    invite somebody to read one that is not there. This has the four fields the assembly
    uses and `data`, and nothing else.
    """
    from types import SimpleNamespace

    from process.main import SingleRun

    from functional_process.sand_harness import REFERENCE_INPUT_FILE, _scratch_copy

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
    """Write `xdsm_sand.html`/`dsm_sand.html` for the assembled SAND graph.

    Built exactly as `run_sand_harness.main` builds it -- same `reference_run`, same
    `mda_env`, same `assemble` -- so the diagram is of the object that actually solves,
    not of a reconstruction that might have drifted from it.
    """
    from cottax import Blocking

    from functional_process.sand_harness import assemble, mda_env

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


MODES = {"sand": sand, "grouped": grouped}

USAGE = """usage: python -m functional_process.render_xdsm [sand|grouped]
                                                [--machine [IN.DAT]]

  (no argument)  the declared model graph  -> xdsm.html, dsm.html        (fast)
  sand           the assembled SAND graph  -> xdsm_sand.html,
                                              dsm_sand.html
  grouped        provenance vs. structure  -> dsm_provenance.html,
                                              dsm_scc.html               (fast)

  --machine [IN.DAT]   draw that machine instead of the reference stellarator,
                       into `*_tokamak.html` (default: the conventional large
                       tokamak). Not available for `sand`.
"""

MACHINE_FLAG = "--machine"
"""Spelled as `boundary.py` spells it, and only in that one long form.

`mode` accepts every other argument `-`- or `--`-prefixed or bare, but this one takes an
optional *value*, and `machine large_tokamak_eval.IN.DAT` with the flag bare would be
two positional words in a parser that otherwise treats an unknown word as an error.
One spelling keeps "is this a file name?" answerable by position alone.
"""


def _machine_argument(argv):
    """`(input file or None, argv without the machine argument)`.

    A bare `--machine` is `TOKAMAK_INPUT_FILE`; a following word that is not itself a
    flag is the `IN.DAT` to read.
    """
    if MACHINE_FLAG not in argv:
        return None, argv
    at = argv.index(MACHINE_FLAG)
    rest = argv[at + 1 :]
    if rest and not rest[0].startswith("-") and rest[0].lstrip("-") not in MODES:
        return rest[0], argv[:at] + rest[1:]
    return TOKAMAK_INPUT_FILE, argv[:at] + rest


def mode(argv):
    """Which renderer `argv` asks for, as the function to call.

    Each mode is accepted bare, `--`-prefixed and `-`-prefixed, because there is no
    reason to make somebody guess which one this module chose. Anything else is a usage
    error rather than a silent fall-through to another graph -- a mistyped argument that
    quietly renders a *different* graph and prints "wrote ..." is worse than one that
    says it does not understand. More than one mode at once is the same kind of mistake:
    only one path can be reported as the one written.

    `--machine` is bound here rather than left to the caller, so that the returned
    callable stays a nullary "render what was asked for". `sand` with a machine is
    refused for the reason this module's docstring gives -- and refused loudly, since a
    renderer that silently drew the *stellarator's* SAND graph under a tokamak's name
    would be the exact failure the paragraph above is about.
    """
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
