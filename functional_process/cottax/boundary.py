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
- `stated` -- a `^stated.<place>` read of a value `models/stated.StatesValues` states at
  assembly (`indat.STATED_VALUES`). Counted apart for the same reason `guess` is, and
  the reason is sharper here: the *place* is still owned by a node, so no consumer fell
  back to the `DataStructure`. What is unowned is the statement itself, and it is
  unowned deliberately -- a value the graph carried would be one nothing could supply,
  sweep or differentiate, and one XLA would fold into the arithmetic that reads it
  (`_audit/optimise_design.md` §34). Growth here means a node stopped deriving something
  and started stating it, which is a modelling regression of a different kind; it is not
  a lost producer, and adding it to `input` would have moved the stellarator 289 -> 298
  and hidden that distinction.

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

    $PY -m functional_process.cottax.boundary --write             # the stellarator
    $PY -m functional_process.cottax.boundary --machine --write   # the tokamak

**The boundary of the model graph is not the boundary of the problem.** Everything
above -- the pins, `provider.py`'s classification, `unproduced_but_computed` -- is
measured on `driven_graph(graph_for(...))`, which carries the models and *not* the
objective or the constraint nodes. A path read only by a condition is therefore
invisible to all of it, and that is how `st_regression`'s objective came to read a
frozen `0.0` for a whole session (`_audit/optimise_design.md` §26). `inert_conditions`
and `refuse_inert_conditions` are the check for that half, over `mdf.mdf_graph`'s
graph, and they need no PROCESS run at all::

    $PY -m functional_process.cottax.boundary --inert             # all seven configurations
    $PY -m functional_process.cottax.boundary --inert --input tests/.../st_regression.IN.DAT
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
should not pull in the model layer."""

GUESS = MintKey("guess")
"""The mint `cottax.rewrites.Initialise` names a `Start` port with -- `^guess.<place>`.

Held here rather than imported from the op so that reading the pin does not depend on
the driver layer, which is where the churn is.
"""

INPUT, GUESSED, STATED = "input", "guess", "stated"

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

Regenerate with `$PY -m functional_process.cottax.boundary --missing --write`, which `_main`
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
    if is_minted(var):
        if var.keys[0] == GUESS:
            return GUESSED
        if var.keys[0] == STATED_MINT:
            return STATED
    return INPUT


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
    from functional_process.cottax.cold_start import cold_state  # noqa: PLC0415

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


@dataclasses.dataclass(frozen=True)
class Inert:
    """One condition of a stated problem that **no design variable reaches**.

    Its value is a constant over the whole design space, so its Jacobian row is
    identically zero and the optimiser cannot steer it. The row is still *evaluated*
    and still *reported*, which is what makes this defect quiet: an inert objective
    reads `-0` and an inert `leq` whose frozen operand is `0.0` reads as comfortably
    satisfied.
    """

    condition: VarPath
    """The condition's own `VarPath` -- `^cond.numerics.objf`, `^cond.constraints.c56`.
    """

    node: NodePath
    """The node owning it -- what the diagnostic names, `.Objective`/`.Constraint56`."""

    frozen: tuple[VarPath, ...]
    """The condition node's **own** operands that are boundary inputs, name-sorted.

    Its own reads, not its cone. **A predicted discriminator that measurement
    refuted, and the correction:** the first version of this field held every boundary
    input in the ancestor cone, on the reasoning that a condition inert because of the
    file's own problem statement would have an empty one and a condition inert because
    of a missing producer would not. It is never empty. Every chain ends at the
    boundary, so `helias_5b`'s `.Constraint11` -- which is inert purely because its
    three iteration variables are the temperature, the density and `hfact`, none of
    which moves a stellarator's radial build -- listed twenty-five perfectly ordinary
    inputs (`.build.dr_blkt_inboard`, `.tfcoil.tftmp`, ...) and looked exactly like a
    defect. The cone size is kept as `cone` for scale; the *direct* operands are what
    separates the two:

    - `len(frozen) == len(operands)` is a condition comparing two constants, and no
      configuration of a working problem should state one. Both `st_regression`
      rows are this shape -- `.Constraint56` is `leq(.physics.p_plasma_separatrix_
      rmajor_mw, .constraints.p_plasma_separatrix_rmajor_max_mw)` with *both* operands
      frozen, and `.Objective` is `objective_metric_5(.current_drive.big_q_plasma)`,
      one operand of one.
    - a condition with a live operand it still cannot steer is the second kind.
      `.Constraint11` is `eq(.build.rbld, .physics.rmajor)`: `rbld` is owned and
      genuinely computed, and the `ixc` simply does not reach it.

    Neither test is a verdict on its own. The verdict is the value side --
    `provider.answers_for`'s `computed` reason, or `unproduced_but_computed` -- which
    asks whether PROCESS's own pipeline writes the path this graph freezes.
    """

    operands: int
    """How many variables the condition node reads in total. `len(frozen)` out of this
    many is the ratio the field above is about."""

    cone: int
    """How many boundary inputs are in the whole ancestor cone. Scale, not evidence --
    see `frozen`. `st_regression`'s `.Objective` has 1; `helias_5b`'s `.Constraint11`
    has 25."""


def _frozen(graph: Graph, design: Iterable[VarPath]) -> set[VarPath]:
    """The graph's boundary inputs **minus the design variables**.

    The subtraction is not cosmetic. `mdf.mdf_graph` inserts the conditions and *not*
    the `Optimise`, so every active `ixc` entry is an unowned input of that graph --
    owned by the problem, which is not in it. Counting those as frozen would call a
    steered variable stuck and would put a design variable in a diagnostic that says
    "this is a missing producer".
    """
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
    """How many frozen boundary inputs feed `name`, transitively.

    `Graph.ancestors` is inclusive, so this counts `name`'s own reads too.
    """
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
    """The `conditions` of `graph` that no variable in `design` reaches. **The check.**

    `_audit/optimise_design.md` §26 is what this exists for. `st_regression.IN.DAT`
    states a real optimisation (`i_process_run_mode = 1`) maximising `FUSION_GAIN_Q`,
    which reads `.current_drive.big_q_plasma` -- a path only
    `models/stellarator/heating.py` owns, so on a *tokamak* graph it is a boundary
    input frozen at its cold `0.0`. VMCON was handed an objective that was identically
    zero with an identically zero gradient, solved the feasibility problem that leaves,
    and reported `converged` in 4 iterations. Nothing refused and nothing warned; the
    only symptoms in `reference_cold_matrix.txt` were `objf -0` and `d objf 1.00e+00`,
    both of which read as ordinary disagreement.

    **Structural, and that is the point.** `Graph.reach` is a walk over declared reads
    and owns -- no PROCESS run, no seed, no solve, no Jacobian. The whole census over
    the seven reference configurations costs seven graph assemblies. The numeric twin
    (`drivers._refuse_inert_objective`, on the first Jacobian the SQP forms) catches the
    same defect one step later and catches numeric coincidences this cannot see; this
    one catches it before anything is evaluated and can say *which boundary path* the
    row rests on, which a zero Jacobian row cannot.

    `design` entries the graph does not know are dropped rather than raising: an `ixc`
    the assembled graph does not carry as a variable is a different defect with its own
    report, and this check should not be the thing that fails on it.

    Parameters
    ----------
    graph :
        The assembled problem graph -- `mdf.mdf_graph`'s, i.e. models *plus* the
        objective and constraint nodes. The plain model graph carries none of the
        conditions and so has nothing for this to check, which is exactly why the
        `reference_provider_*.txt` pins never saw `big_q_plasma`.
    design :
        The design variables' `VarPath`s -- `sand.iteration_variable_path` over the
        run's `ixc`.
    conditions :
        The conditions to check, objective first if there is one.
    """
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
    """Raise unless every one of `conditions` is reachable from `design`.

    Same shape as `check_boundary` and `drivers._refuse_non_finite`: a refusal that
    names the offending rows and their cause, so that a caller which records refusals
    (`run_cold_matrix.py` turns any raise into a row with a reason) reports a
    *measurement* rather than a crash.

    Raises
    ------
    ValueError
        Naming every inert condition, how many of its operands are frozen, and which.
    """
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
    """Boundary inputs of `graph` that some *other* configuration's graph **owns**.

    **The cheap discriminator, and the one that would have caught `st_regression`.**
    A path this graph reads but does not produce, which a node elsewhere in the repo
    does produce, is the `big_q_plasma` shape exactly: not an input at all, but a
    quantity whose producer sits on a variant arm this configuration does not select.

    It is a *lead*, not a verdict, and the difference matters -- 24 to 31 rows per
    configuration answer it and only a handful are defects. `.physics.aspect` is owned
    by a stellarator's graph (`aspect = rmajor / rminor` falls out of the config file)
    and is a declared input on every tokamak, which is correct on both. The verdict
    needs the value side: `unproduced_but_computed`, or `provider.answers_for`'s
    `computed` reason. Use this to *rank* what to check, and note that it is not
    sufficient on its own either -- `.physics.p_plasma_separatrix_rmajor_mw` is owned
    by no graph in the repo at all (`models/physics/exhaust.py` leaves
    `calculate_psep_over_r_metric` unported) and is still a missing producer.

    Returns
    -------
    :
        `(path, (name, ...))` per row, the names those of `others` that own it, sorted
        by written path.
    """
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
    """The written paths `MISSING_PRODUCERS_PIN` holds, recomputed from `input_file`.

    Exactly what `test_no_new_boundary_input_is_something_process_computes` asserts,
    lifted out of the test so `--missing --write` can regenerate the pin the same way
    the test reads it. Expensive (one `SingleRun`, one reference solve and one graph
    assembly, tens of seconds) and deliberately not cached: it is run twice a year, when
    a producer lands.
    """
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
    out: dict[str, int] = {INPUT: 0, GUESSED: 0, STATED: 0}
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
    from functional_process.cottax.indat import GRAPH, graph_for, machine_from_indat

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


def problem_graph(input_file: str):
    """`(graph, design, driven, reported)` for one file -- assembly only, no PROCESS.

    What `inert_conditions` takes, built the way `run_cold_matrix.py`'s MDF arm builds
    it, so this check and that row are about the same problem. Kept here rather than in
    `mdf` because it is the *problem statement*, read off the file: `mdf.mdf_graph`
    already takes `icc`/`n_equality`/`i_figure_merit` as arguments precisely so that it
    does not do this.

    **`driven` and `reported` are separated because inertness means different things
    for them.** `i_figure_merit` is `None` for an evaluation-mode file
    (`i_process_run_mode = -2`), which mints no objective node, and PROCESS then
    root-finds the *equalities* with `scipy.optimize.fsolve`, forms no objective and
    never examines the inequalities. So on such a file only the equalities are `driven`
    and an inert inequality is not a defect at all -- eight of `large_tokamak_eval`'s 23
    are inert and nothing is driving them by design. They are still *evaluated at the
    answer and printed*, which is where an inert one can still mislead a reader, so
    they come back as `reported` rather than being dropped: `spherical_tokamak_eval`'s
    `.Constraint56` is inert, reads a frozen `0.0` against a bound of `40`, and PROCESS
    at its own answer reads `40.28` -- a **violated** constraint the port reports as
    comfortably satisfied. That belongs in the census and not in the refusal.
    """
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
