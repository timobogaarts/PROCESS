"""Render the port's graphs as self-contained, interactive XDSM/DSM HTML pages.

Run directly:

    ~/miniconda/envs/process_port/bin/python -m functional_process.render_xdsm
    ~/miniconda/envs/process_port/bin/python -m functional_process.render_xdsm sand

The bare form writes `xdsm.html`/`dsm.html` for `total_process.GRAPH` -- the declared
model graph, before anything is cut, driven or optimised. Re-run after porting a new
unit to see it join the diagram.

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
"""

import sys
from pathlib import Path

from cottax.visualization import render_dsm_html, render_xdsm_html

from functional_process.total_process import GRAPH

OUTDIR = Path(__file__).parent


def main():
    """Write `xdsm.html`/`dsm.html` for the declared model graph; return the XDSM path."""
    from cottax import Blocking

    render_xdsm_html(
        Blocking.fused(GRAPH),
        file_name="xdsm",
        outdir=str(OUTDIR),
        write=True,
        collapse_names=True,
        blocks=True,
        collapse_models=True,
    )
    render_dsm_html(Blocking.fused(GRAPH), file_name="dsm", outdir=str(OUTDIR), write=True)
    return OUTDIR / "xdsm.html"


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
    for name, blocking in (
        ("xdsm_sand", Blocking.scc(combined)),        
    ):
        render_xdsm_html(
            blocking,
            file_name=name,
            outdir=str(OUTDIR),
            write=True,
            collapse_names=True,
            blocks=True,
            collapse_models=True,
        )
        render_dsm_html(blocking, file_name=f"dsm_{name[5:]}", outdir=str(OUTDIR), write=True)
        print(f"  {name}: {len(blocking.blocks)} blocks")
    render_dsm_html(
        Blocking.scc(combined), file_name="dsm_sand", outdir=str(OUTDIR), write=True
    )
    return OUTDIR / "xdsm_sand.html"


USAGE = """usage: python -m functional_process.render_xdsm [sand]

  (no argument)  the declared model graph  -> xdsm.html, dsm.html        (fast)
  sand           the assembled SAND graph  -> xdsm_sand.html,
                                              xdsm_sand_fused.html,
                                              dsm_sand.html
"""


def wants_sand(argv) -> bool:
    """Whether `argv` asks for the SAND graph.

    Accepts `sand`, `--sand` and `-sand`, because there is no reason to make
    somebody guess which one this module chose. Anything else is a usage error
    rather than a silent fall-through to the other graph -- a mistyped argument
    that quietly renders a *different* graph and prints "wrote ..." is worse than
    one that says it does not understand.
    """
    rest = [a for a in argv if a not in ("sand", "--sand", "-sand")]
    if rest:
        raise SystemExit(f"unrecognised argument(s): {' '.join(rest)}\n\n{USAGE}")
    return len(argv) > 0


if __name__ == "__main__":
    path = sand() if wants_sand(sys.argv[1:]) else main()
    print(f"wrote {path}")
