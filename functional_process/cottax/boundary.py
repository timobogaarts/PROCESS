"""What a machine is allowed to read from outside itself, and a check that it grows no
more.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Iterable, Mapping

from cottax.graph import Graph
from cottax.spec import NodePath, VarPath
from cottax.tools.minting import MintKey, is_minted

STATED_MINT = MintKey("stated")
"""`models/stated.STATED`, restated for the same reason `GUESS` is: reading the pin
should not pull in the model layer.
"""

GUESS = MintKey("guess")
"""The mint `cottax.rewrites.Initialise` names a `Start` port with -- `^guess.<place>`.
"""

INPUT, GUESSED, STATED = "input", "guess", "stated"

PIN = os.path.join(os.path.dirname(__file__), "reference_boundary.txt")
"""The reference machine's audited boundary, one `category<space>path` per line."""

TOKAMAK_PIN = os.path.join(os.path.dirname(__file__), "reference_boundary_tokamak.txt")
"""The same, for the conventional tokamak `TOKAMAK_INPUT_FILE` describes."""

TOKAMAK_INPUT_FILE = "tests/regression/input_files/large_tokamak_eval.IN.DAT"

MISSING_PRODUCERS_PIN = os.path.join(
    os.path.dirname(__file__), "missing_producers_tokamak.txt"
)
"""Boundary `input` entries that PROCESS **computes** -- one written path per line."""

MISSING_PRODUCERS_INPUT_FILE = "tests/regression/input_files/large_tokamak_nof.IN.DAT"
"""Measured on the optimising tokamak, not `TOKAMAK_INPUT_FILE`."""

"""The conventional tokamak this port measures itself against, as `indat`'s
`REFERENCE_INPUT_FILE` is the stellarator. Named here because `TOKAMAK_PIN` is a pin
*of* it and the two must not drift; `--machine` on this module's command line defaults
to it."""


def category(var: VarPath) -> str:
    """Which kind of boundary entry `var` is -- see this module's docstring."""
    if is_minted(var):
        if var.keys[0] == GUESS:
            return GUESSED
        if var.keys[0] == STATED_MINT:
            return STATED
    return INPUT


def boundary(graph: Graph) -> tuple[tuple[str, VarPath], ...]:
    """`graph`'s unowned inputs, categorised, in a stable order."""
    return tuple(
        sorted(
            ((category(v), v) for v in graph.unowned_inputs),
            key=lambda row: (row[0], row[1].path_str()),
        )
    )


def readers_of(graph: Graph, var: VarPath) -> tuple[NodePath, ...]:
    """Every node that reads `var` -- what an orphan's error message needs."""
    return tuple(name for name in graph.nodes if var in graph[name].reads)


def check_boundary(graph: Graph, allowed: Iterable[VarPath], pin: str = PIN) -> None:
    """Raise unless every unowned input of `graph` is in `allowed`."""
    allowed = set(allowed)
    orphans = [(kind, var) for kind, var in boundary(graph) if var not in allowed]
    if not orphans:
        return
    lines = [
        f"  {kind:5} {var.path_str()}  <- read by "
        + ", ".join(name.path_str() for name in readers_of(graph, var))
        for kind, var in orphans
    ]
    inputs = sum(1 for kind, _ in orphans if kind == INPUT)
    raise ValueError(
        f"{len(orphans)} read(s) of this graph have no producer and are not on the "
        f"declared boundary ({inputs} input, {len(orphans) - inputs} guess).\n"
        + "\n".join(lines)
        + "\n\nAn `input` orphan is the defect this check exists for: its consumers "
        "would otherwise fall back to whatever the `DataStructure` holds, silently. "
        "A `guess` orphan is a newly driven problem -- expected, and fixed by "
        f"regenerating the pin ({pin})."
    )


def computed_by_process(input_file: str) -> frozenset[tuple[str, str]]:
    """`{(area, field)}` that PROCESS's own pipeline **writes** in one pass from cold.
    """
    # Imported here, not at module scope: this module is imported by graph assembly, and
    # `cold_start` reaches `process.main`, which pulls in the whole of PROCESS. Same
    # reason `_machine_graph` below defers its own imports.
    from functional_process.cottax.cold_start import cold_state  # noqa: PLC0415

    return cold_state(input_file).written


def unproduced_but_computed(
    graph: Graph, computed: Iterable[tuple[str, str]], design: Iterable[VarPath] = ()
) -> tuple[VarPath, ...]:
    """Boundary `input` entries that PROCESS **computes** -- i.e. missing producers."""
    computed = frozenset(computed)
    design = frozenset(design)
    found = []
    for kind, var in boundary(graph):
        if kind != INPUT or var in design:
            continue
        keys = var.path_str().lstrip(".").split(".")
        if len(keys) == 2 and (keys[0], keys[1]) in computed:
            found.append(var)
    return tuple(found)


@dataclasses.dataclass(frozen=True)
class Inert:
    """One condition of a stated problem that **no design variable reaches**."""

    condition: VarPath
    """The condition's own `VarPath` -- `^cond.numerics.objf`, `^cond.constraints.c56`.
    """

    node: NodePath
    """The node owning it -- what the diagnostic names, `.Objective`/`.Constraint56`."""

    frozen: tuple[VarPath, ...]
    """The condition node's **own** operands that are boundary inputs, name-sorted."""

    operands: int
    """How many variables the condition node reads in total."""

    cone: int
    """How many boundary inputs are in the whole ancestor cone."""


def _frozen(graph: Graph, design: Iterable[VarPath]) -> set[VarPath]:
    """The graph's boundary inputs **minus the design variables**."""
    return set(graph.unowned_inputs) - set(design)


def frozen_reads(
    graph: Graph, name: NodePath, design: Iterable[VarPath] = ()
) -> tuple[VarPath, ...]:
    """The graph's frozen boundary inputs among `name`'s **own** reads, name-sorted."""
    outside = _frozen(graph, design)
    return tuple(
        sorted(
            (v for v in graph[name].reads if v in outside), key=lambda v: v.path_str()
        )
    )


def frozen_cone(graph: Graph, name: NodePath, design: Iterable[VarPath] = ()) -> int:
    """How many frozen boundary inputs feed `name`, transitively."""
    outside = _frozen(graph, design)
    return len({
        var
        for above in graph.ancestors((name,))
        for var in graph[above].reads
        if var in outside
    })


def inert_conditions(
    graph: Graph, design: Iterable[VarPath], conditions: Iterable[VarPath]
) -> tuple[Inert, ...]:
    """The `conditions` of `graph` that no variable in `design` reaches."""
    design = tuple(design)
    known = set(graph.variables)
    reached = set(graph.reach([v for v in design if v in known]))
    owners = graph.owners
    found = []
    for condition in conditions:
        name = owners.get(condition)
        if name is None or name in reached:
            continue
        found.append(
            Inert(
                condition,
                name,
                frozen_reads(graph, name, design),
                len(graph[name].reads),
                frozen_cone(graph, name, design),
            )
        )
    return tuple(found)


def refuse_inert_conditions(
    graph: Graph, design: Iterable[VarPath], conditions: Iterable[VarPath]
) -> None:
    """Raise unless every one of `conditions` is reachable from `design`."""
    inert = inert_conditions(graph, design, conditions)
    if not inert:
        return
    lines = [
        f"  {row.node.path_str()}  ({row.condition.path_str()})  "
        f"{len(row.frozen)} of its {row.operands} operand(s) frozen"
        + (": " + ", ".join(v.path_str() for v in row.frozen) if row.frozen else "")
        + f"; {row.cone} boundary input(s) in its cone"
        for row in inert
    ]
    raise ValueError(
        f"{len(inert)} condition(s) of this problem are not reachable from any design "
        "variable, so their Jacobian rows are identically zero and the optimiser "
        "cannot steer them:\n"
        + "\n".join(lines)
        + "\n\nAn inert OBJECTIVE turns the optimisation into a feasibility problem "
        "while still reporting `converged` -- see `_audit/optimise_design.md` §26. A "
        "row that rests on a frozen boundary input is a missing producer: PROCESS "
        "computes that path and this graph does not own it, so the port evaluates the "
        "condition at a seed where PROCESS evaluates it at a live value."
    )


def owned_elsewhere(
    graph: Graph, others: Mapping[str, Graph]
) -> tuple[tuple[VarPath, tuple[str, ...]], ...]:
    """Boundary inputs of `graph` that some *other* configuration's graph **owns**."""
    mine = set(graph.owners)
    rows = []
    for kind, var in boundary(graph):
        if kind != INPUT or var in mine:
            continue
        owners = tuple(sorted(k for k, g in others.items() if var in g.owners))
        if owners:
            rows.append((var, owners))
    return tuple(rows)


def orphaned_by(base: Graph, swapped: Graph) -> tuple[VarPath, ...]:
    """Reads that `base` produced, `swapped` does not, and something still reads."""
    return tuple(
        sorted(
            (var for var in swapped.unowned_inputs if var in base.owners),
            key=lambda v: v.path_str(),
        )
    )


def write_pin(graph: Graph, path: str = PIN) -> tuple[tuple[str, VarPath], ...]:
    """Regenerate the pin from `graph`."""
    rows = boundary(graph)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            "# The reference machine's audited boundary: what it reads and does not\n"
            "# produce. Generated by `$PY -m functional_process.cottax.boundary --write`;\n"
            "# do not hand-edit. `input` is read from PROCESS's DataStructure and\n"
            "# growth in it is a lost producer; `guess` is a Start port for a driven\n"
            "# unknown and moves only when a Drive does. See "
            "functional_process/cottax/boundary.py.\n"
        )
        for kind, var in rows:
            handle.write(f"{kind} {var.path_str()}\n")
    return rows


def missing_producers(input_file: str = MISSING_PRODUCERS_INPUT_FILE) -> tuple[str, ...]:
    """The written paths `MISSING_PRODUCERS_PIN` holds, recomputed from `input_file`."""
    from functional_process.cottax.indat import graph_for, machine_from_indat
    from functional_process.cottax.mda import driven_graph
    from functional_process.cottax.sand import iteration_variable_path
    from functional_process.cottax.sand_harness import reference_run

    graph = driven_graph(graph_for(machine_from_indat(input_file)))
    design = {iteration_variable_path(i) for i in reference_run(input_file).ixc}
    computed = computed_by_process(input_file)
    return tuple(
        var.path_str() for var in unproduced_but_computed(graph, computed, design)
    )


def write_missing_producers_pin(
    rows: Iterable[str], path: str = MISSING_PRODUCERS_PIN
) -> None:
    """Regenerate `MISSING_PRODUCERS_PIN`."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.writelines(f"{name}\n" for name in rows)


def read_pin(path: str = PIN) -> tuple[tuple[str, str], ...]:
    """The pin as `(category, written path)` pairs, comments and blanks dropped."""
    with open(path, encoding="utf-8") as handle:
        lines = [line.strip() for line in handle]
    return tuple(
        (kind, name)
        for kind, _, name in (
            line.partition(" ") for line in lines if line and not line.startswith("#")
        )
    )


def counts(rows: Iterable[tuple[str, object]]) -> Mapping[str, int]:
    """How many of each category -- the two numbers worth watching."""
    out: dict[str, int] = {INPUT: 0, GUESSED: 0, STATED: 0}
    for kind, _ in rows:
        out[kind] = out.get(kind, 0) + 1
    return out


def _machine_graph(argv: list[str]):
    """`(graph, pin path)` for the machine this invocation is about."""
    from functional_process.cottax.indat import GRAPH, graph_for, machine_from_indat

    if "--machine" not in argv:
        return GRAPH, PIN
    index = argv.index("--machine") + 1
    name = argv[index] if index < len(argv) and not argv[index].startswith("-") else None
    return graph_for(machine_from_indat(name or TOKAMAK_INPUT_FILE)), TOKAMAK_PIN


def _main_missing(argv: list[str]) -> int:
    """`--missing`: report, or with `--write` regenerate, `MISSING_PRODUCERS_PIN`."""
    found = missing_producers(MISSING_PRODUCERS_INPUT_FILE)
    print(
        f"{len(found)} boundary input(s) of "
        f"{os.path.basename(MISSING_PRODUCERS_INPUT_FILE)} are fields PROCESS computes"
    )
    for name in found:
        print(f"  {name}")
    if "--write" not in argv:
        return 0
    write_missing_producers_pin(found)
    print(f"wrote {MISSING_PRODUCERS_PIN}")
    return 0


def problem_graph(input_file: str):
    """`(graph, design, driven, reported)` for one file -- assembly only, no PROCESS."""
    from functional_process.cottax import mdf  # noqa: PLC0415
    from functional_process.cottax.indat import (  # noqa: PLC0415
        graph_for,
        machine_from_indat,
        problem_from_indat,
        switch_values_from_indat,
    )
    from functional_process.cottax.sand import iteration_variable_path  # noqa: PLC0415

    problem = problem_from_indat(input_file)
    icc = tuple(problem.icc)
    n_equality = problem.n_equality_constraints
    if n_equality is None:  # the `-1` sentinel -- see `indat.problem_from_indat`
        n_equality = len(icc) - (problem.n_inequality_constraints or 0)
    graph, _, _, report = mdf.mdf_graph(
        graph_for(machine_from_indat(input_file)),
        icc,
        n_equality,
        None if problem.is_evaluation else problem.i_figure_merit,
        switch_values_from_indat(input_file),
    )
    conditions = tuple(report["equalities"])
    reported = ()
    if problem.is_evaluation:
        reported = tuple(report["inequalities"])
    else:
        conditions = (report["objective"], *conditions, *report["inequalities"])
    design = tuple(iteration_variable_path(i) for i in problem.ixc)
    return graph, design, conditions, reported


def _main_inert(argv: list[str]) -> int:
    """`--inert [<IN.DAT> ...]`: the inert-condition census. Assembly only, no solve."""
    from functional_process.cottax.run_cold_matrix import CONFIGURATIONS  # noqa: PLC0415

    files = [argv[i + 1] for i, a in enumerate(argv) if a == "--input"] or list(
        CONFIGURATIONS
    )
    total = 0
    for input_file in files:
        graph, design, driven, reported = problem_graph(input_file)
        rows = inert_conditions(graph, design, driven)
        loose = inert_conditions(graph, design, reported)
        total += len(rows)
        print(
            f"{os.path.basename(input_file):32} {len(design):3} design, "
            f"{len(driven):3} driven condition(s) -> {len(rows)} inert"
            + (f"  (+{len(loose)}/{len(reported)} reported-only)" if reported else "")
        )
        for label, found in (("", rows), ("reported-only ", loose)):
            for row in found:
                print(
                    f"    {label}{row.node.path_str():16} {len(row.frozen)}/"
                    f"{row.operands} operand(s) frozen, {row.cone} in cone: "
                    + ", ".join(v.path_str() for v in row.frozen)
                )
    return 1 if total else 0


def _main(argv: list[str]) -> int:
    from functional_process.cottax.mda import driven_graph

    if "--missing" in argv:
        return _main_missing(argv)
    if "--inert" in argv:
        return _main_inert(argv)
    graph, pin = _machine_graph(argv)
    driven = driven_graph(graph)
    rows = boundary(driven)
    have = counts(rows)
    print(f"declared graph: {len(graph.unowned_inputs)} unowned input(s)")
    print(
        f"driven graph:   {len(rows)} = {have[INPUT]} input + {have[GUESSED]} guess"
        f" + {have[STATED]} stated"
    )
    if "--write" in argv:
        write_pin(driven, pin)
        print(f"wrote {pin}")
        return 0
    pinned = read_pin(pin)
    was = counts(pinned)
    print(
        f"pin ({os.path.basename(pin)}): {len(pinned)} = {was[INPUT]} input + "
        f"{was[GUESSED]} guess + {was[STATED]} stated"
    )
    check_boundary(
        driven,
        {v for _, v in rows if v.path_str() in {name for _, name in pinned}},
        pin=pin,
    )
    gone = {name for _, name in pinned} - {v.path_str() for _, v in rows}
    if gone:
        print(
            f"{len(gone)} pinned read(s) no longer on the boundary -- a producer "
            f"landed. Regenerate:\n  " + "\n  ".join(sorted(gone))
        )
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv[1:]))
