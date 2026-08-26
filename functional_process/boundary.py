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

The reference machine today: **320 inputs**, and 18 guesses on top of that once
`driven_graph` has initialised its problems.

Regenerate the pin (never hand-edit it) with::

    $PY -m functional_process.boundary --write
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


def category(var: VarPath) -> str:
    """Which kind of boundary entry `var` is -- see this module's docstring."""
    return GUESSED if is_minted(var) and var.keys[0] == GUESS else INPUT


def boundary(graph: Graph) -> tuple[tuple[str, VarPath], ...]:
    """`graph`'s unowned inputs, categorised, in a stable order.

    Sorted by written name rather than left in read order: this is a set that gets
    diffed against a file, and a diff that reordered when an unrelated node was added
    would report noise as change.
    """
    return tuple(sorted(((category(v), v) for v in graph.unowned_inputs),
                        key=lambda row: (row[0], row[1].path_str())))


def readers_of(graph: Graph, var: VarPath) -> tuple[NodePath, ...]:
    """Every node that reads `var` -- what an orphan's error message needs.

    The slot, in other words. `model_tree_design.md` §6 asks the check to name "the slot
    whose occupant lost it", and the honest form of that is the consumers: the producer
    is by definition absent, so there is nothing to name on that side, while the nodes
    left holding the read are exactly what has to be re-pointed or re-produced.
    """
    return tuple(name for name in graph.nodes if var in graph[name].reads)


def check_boundary(graph: Graph, allowed: Iterable[VarPath]) -> None:
    """Raise unless every unowned input of `graph` is in `allowed`.

    One-directional on purpose: a boundary that *shrank* is a producer landing, which is
    the point of the exercise and must not fail a build. The test alongside this asserts
    equality, so a shrink still shows up -- as a pin to regenerate, at the moment it is
    cheap, rather than as a build that breaks under someone else.

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
        f"regenerating the pin ({PIN})."
    )


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
    return tuple(sorted((var for var in swapped.unowned_inputs if var in base.owners),
                        key=lambda v: v.path_str()))


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


def read_pin(path: str = PIN) -> tuple[tuple[str, str], ...]:
    """The pin as `(category, written path)` pairs, comments and blanks dropped."""
    with open(path, encoding="utf-8") as handle:
        lines = [line.strip() for line in handle]
    return tuple(
        (kind, name)
        for kind, _, name in (line.partition(" ") for line in lines
                              if line and not line.startswith("#"))
    )


def counts(rows: Iterable[tuple[str, object]]) -> Mapping[str, int]:
    """How many of each category -- the two numbers worth watching."""
    out: dict[str, int] = {INPUT: 0, GUESSED: 0}
    for kind, _ in rows:
        out[kind] = out.get(kind, 0) + 1
    return out


def _main(argv: list[str]) -> int:
    from functional_process.indat import GRAPH
    from functional_process.mda import driven_graph

    driven = driven_graph(GRAPH)
    rows = boundary(driven)
    have = counts(rows)
    print(f"declared graph: {len(GRAPH.unowned_inputs)} unowned input(s)")
    print(f"driven graph:   {len(rows)} = {have[INPUT]} input + {have[GUESSED]} guess")
    if "--write" in argv:
        write_pin(driven)
        print(f"wrote {PIN}")
        return 0
    pinned = read_pin()
    was = counts(pinned)
    print(f"pin:            {len(pinned)} = {was[INPUT]} input + {was[GUESSED]} guess")
    check_boundary(driven, {v for _, v in rows if v.path_str()
                            in {name for _, name in pinned}})
    gone = {name for _, name in pinned} - {v.path_str() for _, v in rows}
    if gone:
        print(f"{len(gone)} pinned read(s) no longer on the boundary -- a producer "
              f"landed. Regenerate:\n  " + "\n  ".join(sorted(gone)))
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv[1:]))
