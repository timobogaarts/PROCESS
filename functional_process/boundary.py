"""
What a machine is allowed to read from outside itself, and a check that it grows no more.

**The boundary is the distance from a purely functional graph.** Every read a node makes
is either produced by another node or taken from outside -- out of PROCESS's
`DataStructure`, which `init_process` filled from an `IN.DAT`. The second kind is this
module's subject. It is not an error: 109 float-valued inputs in the reference `IN.DAT`
are genuine inputs and always will be. It is a *number that should only ever go down*,
because everything else on it is a read whose producer is not ported yet, and the end
state -- every model ported, nothing overwriting the central data structure, only minted
namespaces -- is a boundary holding the real inputs and nothing else.

**Nothing checked it until this module.** A slot whose new occupant does not write what
the old one wrote does not fail: its consumers silently fall back to whatever value sits
in the `DataStructure`, and the run completes with a wrong number. That defect class has
**eight recorded instances and not one of them was found by a check** -- every one was
noticed downstream, by a consumer disagreeing (`next_steps.md` §12.1). `check_boundary`
is the check: it converts "silently reads a stale default" into "refuses to build".

**Two kinds of boundary entry, counted apart.** A physical input and a solver's starting
guess are both unowned, and lumping them together makes the number useless: landing a
producer takes one away and declaring a problem adds another, so a single total can sit
still while both halves move. `Initialise` mints one `^guess.<place>` port per driven
unknown (`mda.driven_graph`), so the split is mechanical and exact:

- `input` -- read from the `DataStructure`. **Growth here is the defect.**
- `guess`  -- a `Start` port for a driven unknown. Growth here is a new problem, which
  is structure, not a regression; it should move only when a `Drive` does.

The reference machine today: **297 inputs**, and 6 guesses on top of that once
`driven_graph` has assigned its problems' drivers. The conventional tokamak
(`TOKAMAK_INPUT_FILE`) is **356 inputs and 11 guesses**, pinned separately in
`reference_boundary_tokamak.txt` -- and the comparison between the two files is the point
of having a second one: a tokamak reads *more* than a stellarator, from a graph with more
nodes in it, which is the honest shape of a device whose device-specific namespace is
twenty-six slots filled and two still empty. (Waves 2/3's consolidation *grew* the
tokamak's input count, 328 -> 349, while closing ten rows: the eleven newly registered
slots' nodes declare thirty-one genuine inputs of their own -- `cboot`, `q95`, the PF
coil current-density settings and their kin -- each named in advance by its unit's
"boundary inputs this slot then needs" list. Growth from a *landed producer's own
declared reads* is the boundary doing its job; growth from a lost producer is the
defect.)

Regenerate a pin (never hand-edit one) with::

    $PY -m functional_process.boundary --write             # the stellarator
    $PY -m functional_process.boundary --machine --write   # the tokamak
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping

from cottax.graph import Graph
from cottax.spec import NodePath, VarPath
from cottax.tools.minting import MintKey, is_minted

GUESS = MintKey("guess")
"""The mint `cottax.rewrites.Initialise` names a `Start` port with -- `^guess.<place>`.

Held here rather than imported from the op so that reading the pin does not depend on
the driver layer, which is where the churn is.
"""

INPUT, GUESSED = "input", "guess"

PIN = os.path.join(os.path.dirname(__file__), "reference_boundary.txt")
"""The reference machine's audited boundary, one `category<space>path` per line."""

TOKAMAK_PIN = os.path.join(os.path.dirname(__file__), "reference_boundary_tokamak.txt")
"""The same, for the conventional tokamak `TOKAMAK_INPUT_FILE` describes.

**A second file rather than a second column**, because it is a second *machine*. A
boundary is a property of one assembled graph and the two graphs share only their
device-agnostic subsystems; merging them would need a per-row "which machines is this
on" column that nothing would ever read, and would hide the interesting comparison --
which is between the two files, not inside one.
"""

TOKAMAK_INPUT_FILE = "tests/regression/input_files/large_tokamak_eval.IN.DAT"

MISSING_PRODUCERS_PIN = os.path.join(
    os.path.dirname(__file__), "missing_producers_tokamak.txt"
)
"""Boundary `input` entries that PROCESS **computes** -- one written path per line.

The list `unproduced_but_computed` returns for `MISSING_PRODUCERS_INPUT_FILE`. Pinned
rather than asserted empty because it is not empty. **Twenty-two** when the measurement
was first taken on 2026-08-30 (`optimise_design.md` §16); then, in one day and three
independent waves, `.physics.beta_poloidal_vol_avg` (`.tokamak.plasma_beta.poloidal`),
four build rows (`.tokamak.build`'s `tf_top_height`/`blkt_upper_thickness`/
`tf_inner_bore`), four PF-power rows (`.power.pf_coil_power`, `Power.pfpwr`) and five
physics/divertor/cryostat rows landed. **This number may only go down.** A new entry
means a node stopped writing something PROCESS writes -- the silent-stale-read defect
this module exists to catch, and the one that has eight recorded instances none of which
a check found.

Regenerate with `$PY -m functional_process.boundary --missing --write`, which `_main`
grew on 2026-08-30 -- this docstring named the command a wave before the branch existed.
"""

MISSING_PRODUCERS_INPUT_FILE = "tests/regression/input_files/large_tokamak_nof.IN.DAT"
"""Measured on the optimising tokamak, not `TOKAMAK_INPUT_FILE`.

`large_tokamak_eval` is an evaluation-mode run (`fsolve` over the equalities alone), so
its pipeline exercises less; the optimising file is the one whose cold start these holes
actually broke.
"""

"""The conventional tokamak this port measures itself against, as `indat`'s
`REFERENCE_INPUT_FILE` is the stellarator. Named here because `TOKAMAK_PIN` is a pin
*of* it and the two must not drift; `--machine` on this module's command line defaults
to it."""


def category(var: VarPath) -> str:
    """Which kind of boundary entry `var` is -- see this module's docstring."""
    return GUESSED if is_minted(var) and var.keys[0] == GUESS else INPUT


def boundary(graph: Graph) -> tuple[tuple[str, VarPath], ...]:
    """`graph`'s unowned inputs, categorised, in a stable order.

    Sorted by written name rather than left in read order: this is a set that gets
    diffed against a file, and a diff that reordered when an unrelated node was added
    would report noise as change.
    """
    return tuple(
        sorted(
            ((category(v), v) for v in graph.unowned_inputs),
            key=lambda row: (row[0], row[1].path_str()),
        )
    )


def readers_of(graph: Graph, var: VarPath) -> tuple[NodePath, ...]:
    """Every node that reads `var` -- what an orphan's error message needs.

    The slot, in other words. `model_tree_design.md` §6 asks the check to name "the slot
    whose occupant lost it", and the honest form of that is the consumers: the producer
    is by definition absent, so there is nothing to name on that side, while the nodes
    left holding the read are exactly what has to be re-pointed or re-produced.
    """
    return tuple(name for name in graph.nodes if var in graph[name].reads)


def check_boundary(graph: Graph, allowed: Iterable[VarPath], pin: str = PIN) -> None:
    """Raise unless every unowned input of `graph` is in `allowed`.

    One-directional on purpose: a boundary that *shrank* is a producer landing, which is
    the point of the exercise and must not fail a build. The test alongside this asserts
    equality, so a shrink still shows up -- as a pin to regenerate, at the moment it is
    cheap, rather than as a build that breaks under someone else.

    `pin` is the pin file the error message tells the reader to regenerate --
    `TOKAMAK_PIN` for a `--machine` invocation. It defaults to the stellarator's
    because that is the machine every no-argument caller means; a message that always
    named `reference_boundary.txt` sent a tokamak orphan's reader to the wrong file.

    Raises
    ------
    ValueError
        Naming every orphan, its category, and the nodes still reading it.
    """
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

    Measured, not grepped. `tokamak_boundary.md` classified its boundary rows by
    searching `process/` for a writer and reasoning about whether that writer was
    dormant on the run -- correct there, but it cannot see a writer that fires for a
    reason the reader did not anticipate, and it has to be redone by hand per
    configuration. This runs the pipeline instead: snapshot every numeric field of every
    area of a `SingleRun`'s `DataStructure` before any model has run, evaluate once at
    the cold `x`, snapshot again, and return what moved.

    One `SingleRun` plus one `Evaluators.fcnvmc1` -- 0.3 s on a tokamak, 6 s on the
    stellarator, which re-reads its preset file inside every model call. Callers still
    pin the result rather than recomputing it per test, because the *graph* half of the
    comparison is the expensive part, not this.

    **The measurement itself moved to `cold_start.cold_state`** on 2026-08-30, and this
    is now a one-line delegation to it. Not a tidy-up: `cold_start` is the value-side
    twin of this function -- it asks whether the graph *computes* what PROCESS computes
    where this one asks whether it *owns* it -- and the two answers are only comparable
    if they come from the same pipeline evaluation at the same point. Two independent
    `SingleRun`s would have been two chances to drift. `cold_state` also caches and
    copies the input file to a scratch directory first, so a caller no longer leaves an
    `OUT.DAT`/`MFILE.DAT` beside the repository's own `IN.DAT`.
    """
    # Imported here, not at module scope: this module is imported by graph assembly, and
    # `cold_start` reaches `process.main`, which pulls in the whole of PROCESS. Same
    # reason `_machine_graph` below defers its own imports.
    from functional_process.cold_start import cold_state  # noqa: PLC0415

    return cold_state(input_file).written


def unproduced_but_computed(
    graph: Graph, computed: Iterable[tuple[str, str]], design: Iterable[VarPath] = ()
) -> tuple[VarPath, ...]:
    """Boundary `input` entries that PROCESS **computes** -- i.e. missing producers.

    This is the discrimination this module's docstring asks for and could not make.
    `boundary` counts 297 inputs on the stellarator and 349 on the tokamak without
    saying which of them are the "109 genuine inputs" and which are "a read whose
    producer is not ported yet". A field PROCESS writes every pipeline pass is, by
    construction, the second kind: nothing in the graph owns it, so it stays frozen at
    whatever the seed supplied while PROCESS recomputes it.

    **Why this is the check that was missing.** The eight recorded instances of the
    silent-stale-read defect were all found downstream, by a consumer disagreeing. So
    were the twenty-two found on `large_tokamak_nof` (`optimise_design.md` §16) -- and
    those were *invisible* at the one place the harness looks hardest, because Stage A
    and C2 seed boundary inputs from PROCESS's **converged** `DataStructure`, which
    hands every missing producer the right answer. Only a cold start exposes them, and
    only if somebody asks this question.

    `design` names the run's iteration variables, which are boundary inputs on purpose:
    the optimiser owns them, and PROCESS writes them for exactly that reason
    (`set_scaled_iteration_variable`). Excluded rather than reported.
    """
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


def orphaned_by(base: Graph, swapped: Graph) -> tuple[VarPath, ...]:
    """Reads that `base` produced, `swapped` does not, and something still reads.

    **The swap contract, and the check `next_steps.md` §12.2 designed and nothing
    implemented.** Alternatives are keyed on output *nearly*: colliding outputs prove
    exclusivity but do not define it, and equal output sets must not be required --
    `i_cost_model`'s arms genuinely compute different things. What is left is the
    partial-overlap hazard: A owns `{x, y}`, B owns `{x}`, so choosing B leaves `y` with
    no producer and every consumer of `y` silently reading the `DataStructure` instead.

    So the question is asked of **consumers, not producers**: after the swap, does every
    remaining read still have an owner? An unowned input is by definition read by
    someone, so intersecting `swapped`'s unowned inputs with what `base` owned is exactly
    "was produced, is not now, and somebody still wants it" -- no comparison of the two
    occupants' output sets is needed, which is what keeps this from re-introducing the
    equal-outputs rule §12.2 rejects.

    A variable that was *always* on the boundary is not orphaned by anything, and an
    occupant that legitimately needs a new input of its own is not either -- neither was
    ever owned by `base`.
    """
    return tuple(
        sorted(
            (var for var in swapped.unowned_inputs if var in base.owners),
            key=lambda v: v.path_str(),
        )
    )


def write_pin(graph: Graph, path: str = PIN) -> tuple[tuple[str, VarPath], ...]:
    """Regenerate the pin from `graph`. Generated, never typed -- 338 hand-copied paths
    is a transcription error waiting to be argued with.
    """
    rows = boundary(graph)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            "# The reference machine's audited boundary: what it reads and does not\n"
            "# produce. Generated by `$PY -m functional_process.boundary --write`;\n"
            "# do not hand-edit. `input` is read from PROCESS's DataStructure and\n"
            "# growth in it is a lost producer; `guess` is a Start port for a driven\n"
            "# unknown and moves only when a Drive does. See functional_process/boundary.py.\n"
        )
        for kind, var in rows:
            handle.write(f"{kind} {var.path_str()}\n")
    return rows


def missing_producers(input_file: str = MISSING_PRODUCERS_INPUT_FILE) -> tuple[str, ...]:
    """The written paths `MISSING_PRODUCERS_PIN` holds, recomputed from `input_file`.

    Exactly what `test_no_new_boundary_input_is_something_process_computes` asserts,
    lifted out of the test so `--missing --write` can regenerate the pin the same way
    the test reads it. Expensive (one `SingleRun`, one reference solve and one graph
    assembly, tens of seconds) and deliberately not cached: it is run twice a year, when
    a producer lands.
    """
    from functional_process.indat import graph_for, machine_from_indat
    from functional_process.mda import driven_graph
    from functional_process.sand import iteration_variable_path
    from functional_process.sand_harness import reference_run

    graph = driven_graph(graph_for(machine_from_indat(input_file)))
    design = {iteration_variable_path(i) for i in reference_run(input_file).ixc}
    computed = computed_by_process(input_file)
    return tuple(
        var.path_str() for var in unproduced_but_computed(graph, computed, design)
    )


def write_missing_producers_pin(
    rows: Iterable[str], path: str = MISSING_PRODUCERS_PIN
) -> None:
    """Regenerate `MISSING_PRODUCERS_PIN`. Generated, never typed, like every pin here.

    No comment header, unlike `write_pin`: the test that reads this file takes every
    non-blank line as a path, and the explanation belongs where the number is defended
    (`MISSING_PRODUCERS_PIN`'s own docstring and the test's).
    """
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
    out: dict[str, int] = {INPUT: 0, GUESSED: 0}
    for kind, _ in rows:
        out[kind] = out.get(kind, 0) + 1
    return out


def _machine_graph(argv: list[str]):
    """`(graph, pin path)` for the machine this invocation is about.

    No argument means the stellarator reference machine and `PIN`, exactly as before.
    `--machine [<IN.DAT>]` means the machine that file describes and `TOKAMAK_PIN`,
    defaulting to `TOKAMAK_INPUT_FILE`.

    **The smallest extension that gives a second device a pin**, and deliberately not a
    general one: a pin is only useful for a machine something else also names, and this
    port names exactly two. A third would want a table rather than a second branch.
    """
    from functional_process.indat import GRAPH, graph_for, machine_from_indat

    if "--machine" not in argv:
        return GRAPH, PIN
    index = argv.index("--machine") + 1
    name = argv[index] if index < len(argv) and not argv[index].startswith("-") else None
    return graph_for(machine_from_indat(name or TOKAMAK_INPUT_FILE)), TOKAMAK_PIN


def _main_missing(argv: list[str]) -> int:
    """`--missing`: report, or with `--write` regenerate, `MISSING_PRODUCERS_PIN`.

    Shares `missing_producers` with the test that checks the pin, deliberately: a pin
    generated differently from the way it is read is a pin that can pass while being
    wrong.
    """
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


def _main(argv: list[str]) -> int:
    from functional_process.mda import driven_graph

    if "--missing" in argv:
        return _main_missing(argv)
    graph, pin = _machine_graph(argv)
    driven = driven_graph(graph)
    rows = boundary(driven)
    have = counts(rows)
    print(f"declared graph: {len(graph.unowned_inputs)} unowned input(s)")
    print(f"driven graph:   {len(rows)} = {have[INPUT]} input + {have[GUESSED]} guess")
    if "--write" in argv:
        write_pin(driven, pin)
        print(f"wrote {pin}")
        return 0
    pinned = read_pin(pin)
    was = counts(pinned)
    print(
        f"pin ({os.path.basename(pin)}): {len(pinned)} = {was[INPUT]} input + "
        f"{was[GUESSED]} guess"
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
