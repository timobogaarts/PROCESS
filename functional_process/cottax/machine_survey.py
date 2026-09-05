"""
What a given `IN.DAT` asks for that this port cannot yet give it.

`machine_from_indat` refuses at the *first* switch it has no occupant for, which is right
for building and useless for planning: a second device needs the whole list at once. This
module walks an input file's switches against the tree without building anything, and
says of each one whether the factory reads it, whether the tree pins it as a static
kwarg, or whether the port has never heard of it -- and, for the last kind, which
PROCESS modules read it and whether any of those is bound to a library JAX cannot trace.

**Every column is measured.** The factory's own fields are read out of `indat.py`'s
source (the literals `_slot_occupant`/`pick` are called with, and -- since 2026-08-29 --
the registry each one is looked up in, off the AST); the pinned switches are
introspected off the assembled graph's declaration instances, the same walk
`mda_harness.switch_audit` does; the readers come from `process/` itself. Nothing here
is a list somebody typed and nothing has to be kept in step by hand.

**Two blind spots, both found by the ST closing wave and both closed.** A planning tool
that under-reports is worse than none, and this one under-reported twice on the same two
files:

1. A value with **no occupant and no `UNPORTED` reason** -- `_slot_occupant`'s
   `ValueError` path rather than its `NotImplementedError` path -- printed as "the
   factory dispatches on it", which is true of the field and false of the value. That is
   how `i_beta_norm_max = 0` survived a re-survey whose whole purpose was to enumerate
   what both spherical tokamaks still needed. `slot_registries` now derives
   `field -> registry` from `indat.py`'s AST and the second failure mode is its own kind.
2. A slot dispatched on a **derived arm index** (`pf_coil_system_arm` and a dozen
   siblings) has no name in any `IN.DAT`, so no column of a switch table can ever reach
   it. No amount of care fixes that; `report` therefore ends with one real
   `machine_from_indat` attempt (`assembly_verdict`), which is exact and costs a second.

The pair is the reason both tracked spherical tokamaks were briefed as "exactly four
switch values away" when they were six -- and two of the six are unported model packages
rather than arms.

**Why the traceability column exists.** `next_steps.md` §5 records the CoolProp policy as
flagged and unresolved: `process/core/coolprop_interface.py` is an opaque external C call,
not a JAX primitive, so a model reached through it cannot be traced until it is wrapped.
A slot that is unported because nobody has written it and a slot that is unported because
it calls into CoolProp are different work, and an enumeration that does not separate them
will be re-derived by whoever picks it up.
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import os
import re
import subprocess

COOLPROP_MODULES = (
    "process/models/fw.py",
    "process/models/engineering/pumping.py",
    "process/models/stellarator/stellarator.py",
    "process/models/tfcoil/quench.py",
    "process/models/blankets/blanket_library.py",
    "process/models/blankets/hcpb.py",
)
"""Every module under `process/models/` that reaches CoolProp, directly or through
`blanket_library`. Measured (`grep -rln coolprop process/models/`), not curated."""

NOT_TOPOLOGY = {
    "icc": "an array line; the parser reads its first element, not a switch",
    "ixc": "an array line; the parser reads its first element, not a switch",
    "n_equality_constraints": "belongs to the *study* (which conditions are solved), "
    "not to the machine -- `next_steps.md` §13.8",
    "i_process_run_mode": "run control: solve, evaluate, or scan",
    "output_costs": "output control: whether the cost tables are printed",
    "p_fusion_total_max_mw": "a limit *value* that happens to be integral, not a choice "
    "between models",
}
"""Integers an `IN.DAT` carries that are not topology decisions, and why.

Stated as a table rather than filtered silently, because "23 new decisions" and "17 new
decisions" are different plans and the difference is entirely in this dict. Every
exclusion is a claim someone can check and argue with; `p_fusion_total_max_mw = 3000` is
the one that shows why the parser cannot decide this on shape alone.
"""

SHAPE = {
    "n_tf_coils": "a count -- it sizes arrays rather than selecting a model "
    "(`switch_elimination_design.md` §3 kind (b))",
    "n_pf_coil_groups": "a count, same kind",
}
"""Counts, which are real work and not *model* choices. Kept in the total and marked."""

FACTORY = re.compile(
    r'(?:_slot_occupant|pick)\(\s*"(\w+)"'
    r'|switches\.get\(\s*"(\w+)"'
    r'|numbers\.get\(\s*"(\w+)"'
)
"""How the fields the factory dispatches on are read back out of `indat.py`.

Source-derived rather than a second list to maintain: the alternative is a table that
silently goes stale the first time a registry is added, which is the failure mode
`model_tree_design.md` §6 exists to stop in the boundary and is no better here.

**`numbers.get` joined the two dispatch forms** when the tokamak wave gave the factory
two keys that are switch-shaped without being switches: `.tfcoil.n_tf_coils`, whose
*rounded* value selects one of four ripple fits, and `.build.dz_xpoint_divertor`, whose
being effectively zero decides whether a node owns it. Both are read through
`numbers_from_indat` rather than `switches_from_indat` because neither is an integer
switch, and without this alternation the survey would report `n_tf_coils` as *"the port
has never read it"* -- a false statement in the one tool whose job is to say what the
port has read.
"""


@dataclasses.dataclass(frozen=True)
class Row:
    """One switch of one input file, and what this port can do with it."""

    name: str
    value: int
    verdict: str
    """`factory` (a slot dispatches on it) / `pinned` (the tree hardcodes it as a static
    kwarg) / `unknown` (the port has never read it).

    **`pinned` stays an `elif` behind `factory`, and the contradiction check does not
    live here.** `_audit/switch_consultation_audit.md` §6 proposed lifting it -- `iohcl`
    was in `factory_fields()` *and* pinned on `PfMagnetCost`, so the `DISAGREES` branch
    never ran for it and §2's live wrong answer was invisible to this module. Lifting it
    was tried on 2026-08-31 and **reverted**, because this function's `graph` defaults to
    `indat.GRAPH` -- the *reference* machine, built from `stellarator_helias` -- so
    comparing any other file's switches against its pins is unsound in both directions.
    It immediately reported `large_tokamak_eval`'s `i_p_coolant_pumping = 3` as
    contradicted by a `1` that belongs to a stellarator this file has nothing to do with.
    A pinned-value check is only meaningful against the machine *that file* assembles,
    and that is what
    `functional_process/tests/test_switch_coverage.py::
    test_no_pinned_switch_contradicts_its_own_input_file` does for all seven
    configurations. The guard exists; it simply cannot be this function."""
    detail: str = ""
    readers: tuple[str, ...] = ()
    coolprop: bool = False
    """Whether *some* module reading this switch also reaches CoolProp.

    A hint, deliberately weak: it says the neighbourhood is untraceable, not that this
    switch's own branch is. `n_tf_coils` trips it only because `stellarator.py` reads
    both. Treat it as "check the wrapping policy before scheduling this", never as a
    verdict.
    """


def factory_fields(path: str | None = None) -> frozenset[str]:
    """The switch fields `machine_from_indat` dispatches on."""
    path = path or os.path.join(os.path.dirname(__file__), "indat.py")
    with open(path, encoding="utf-8") as handle:
        found = FACTORY.findall(handle.read())
    # Three alternations, so each match is a triple with two empty parts. All three
    # forms count:
    # `_slot_occupant("X", ...)` names the slot's switch directly, and
    # `switches.get("X", ...)` is how a switch that feeds a *joint* arm function is read
    # -- `inuclear` reaches `_slot_occupant` only as part of `"inuclear_i_tf_sup"`, so
    # matching the dispatch call alone reported it as a decision nobody had made.
    # `numbers.get("X", ...)` is the third: a float or a count that selects an occupant.
    return frozenset(name for pair in found for name in pair if name)


@functools.cache
def slot_registries(path: str | None = None) -> dict[str, tuple[str, ...]]:
    """`switch field -> the registry names `_slot_occupant` looks it up in`.

    Parsed out of `indat.py`'s AST rather than matched with a regex, because the call is
    routinely spread over five lines with a comment between the value and the registry
    and the value expression itself contains parentheses.

    **This exists because `UNPORTED` is only half of "no occupant".** `_slot_occupant`
    has two failure modes: a value that is in `UNPORTED` (a real PROCESS branch nobody
    has written, `NotImplementedError`) and a value that is in **neither** the registry
    nor `UNPORTED` (`ValueError`, "not a known value"). Until 2026-08-29 this module
    checked only the first, so a value of the second kind reported as *"the factory
    dispatches on it"* -- which is true of the field and false of the value, and is
    exactly how `i_beta_norm_max = 0` stayed invisible on both tracked spherical
    tokamaks through a re-survey that was explicitly counting the remaining blockers.
    A refusal the planning tool cannot see is worse than no tool, so the second failure
    mode is derived here from the same two sources `_slot_occupant` itself reads.

    A field may fill more than one slot (`i_tf_case_geom` fills two, `n_divertors`
    four), so the value is a tuple and a miss in *any* of them is a miss.
    """
    path = path or os.path.join(os.path.dirname(__file__), "indat.py")
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    found: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_slot_occupant"
            and len(node.args) >= 3
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and isinstance(node.args[2], ast.Name)
        ):
            registries = found.setdefault(node.args[0].value, [])
            if node.args[2].id not in registries:
                registries.append(node.args[2].id)
    return {field: tuple(names) for field, names in found.items()}


ROUTED_AWAY = {
    ("i_hcd_primary", 13): ("HCD_PRIMARY_EFFICIENCY", "HCD_PRIMARY_EFFICIENCY_FREETHY"),
}
"""`(field, value) -> (the registry it is absent from, the registry it is routed to)`.

The one thing `slot_registries`' static read cannot see: a *routing function* that
answers a switch value from a different registry before `_slot_occupant` is reached.
`_hcd_primary_efficiency` is the only one -- `i_hcd_primary == 13` is the single value
with a switch nested inside it (`i_ecrh_wave_mode`), so it consults
`HCD_PRIMARY_EFFICIENCY_FREETHY` and never the outer registry, which is why the outer
registry deliberately has no entry for it (`indat.py`'s own docstring says so).

**Each entry is earned, not asserted**: `test_machine_survey.py` fails an entry whose
value is not in fact absent from the first registry and present in the second, so a
routing that goes away cannot leave a stale exemption behind. Same discipline as
`_harness/boundary.py`'s register.
"""


def unoccupied_registries(name, value) -> tuple[str, ...]:
    """The registries dispatching on `name` that hold no occupant for `value`.

    Empty when every slot this field fills has an arm for the value -- including when
    the field fills no slot by this route at all (a joint arm function reads it, and the
    composite name is what `_slot_occupant` sees).
    """
    from functional_process.cottax import indat

    routed = ROUTED_AWAY.get((name, value))
    missing = []
    for registry_name in slot_registries().get(name, ()):
        if routed is not None and registry_name == routed[0]:
            continue
        registry = getattr(indat, registry_name, None)
        if isinstance(registry, dict) and value not in registry:
            missing.append(registry_name)
    return tuple(missing)


def pinned_switches(graph) -> dict[str, set[int]]:
    """Every static switch kwarg the assembled graph carries, and the values it holds.

    Introspection of the graph's own declaration instances -- `switch_audit`'s walk
    without its `DataStructure` argument, because what is wanted here is *which
    questions the tree has already answered*, not whether it answered them correctly.
    """
    from functional_process.cottax.mda_harness import (
        STATIC_KWARG_KINDS,
        SWITCH,
        _declaration_modules,
    )

    out: dict[str, set[int]] = {}
    for node in graph.definitions.values():
        for declaration in _declaration_modules(node, set()):
            for field in dataclasses.fields(declaration):
                if not field.metadata.get("static"):
                    continue
                if STATIC_KWARG_KINDS.get(field.name, SWITCH) is not SWITCH:
                    continue
                value = getattr(declaration, field.name, None)
                if isinstance(value, int):
                    out.setdefault(field.name, set()).add(int(value))
    return out


def readers_in_process(name: str, root: str = "process") -> tuple[str, ...]:
    """Which PROCESS modules read `name` -- where the work would have to be done."""
    found = subprocess.run(
        ["grep", "-rlw", "--include=*.py", name, root],
        capture_output=True,
        text=True,
        check=False,
    )
    return tuple(sorted(line for line in found.stdout.split() if line))


def _how_the_port_reads(name: str, graph) -> str:
    """Why a switch reached the `unknown` bucket -- three different reasons, and only
    one of them is "nothing in the port has ever looked at it".

    A row lands here when no slot of the machine tree dispatches on the switch and no
    node pins it as a static kwarg. That is the question `survey` is asking, and the
    verdict stays `unknown` for all three. What is *not* the same for all three is the
    sentence, and the old one ("the port has never read it") was false for every
    `unknown` row `large_tokamak_eval` produces:

    * **The constraint layer binds it.** `sand._bind` partials a constraint's or the
      objective's switch parameters at assembly time, so `i_beta_component` and
      `i_plant_availability` genuinely select a formula -- in a layer the machine tree
      does not contain. Nothing is owed for these; the tree simply is not where they
      live.
    * **A node declares it as an ordinary read.** `.heat_transport.i_shld_primary_heat`
      is an `In` on a real edge, which means a switch integer is travelling through a
      declared port. That is not coverage, it is
      `_audit/switch_kwarg_survey.md` §0's second defect ("10 declared ports carrying a
      switch integer") showing up in a second measurement, and the row should read as
      work rather than as absence.
    * **Genuinely unread**, which is what the sentence used to claim for all of them.

    Measured off the two live sources rather than listed here, so it cannot drift: the
    constraint layer's own `SWITCH_PARAMETER_NAMES`, and the assembled graph's
    variables.
    """
    from functional_process.cottax.sand import SWITCH_PARAMETER_NAMES

    if name in SWITCH_PARAMETER_NAMES:
        return (
            "no slot dispatches on it, but the constraint/objective layer binds it as "
            "a static kwarg (`sand.SWITCH_PARAMETER_NAMES`) -- read, outside the tree"
        )
    declared = {
        var.path_str()
        for var in graph.variables
        if var.keys and getattr(var.keys[-1], "name", None) == name
    }
    if declared:
        return (
            f"no slot dispatches on it; a node declares {min(declared)} as an "
            f"ordinary read -- a declared port carrying a switch integer, "
            f"`switch_kwarg_survey.md` §0"
        )
    return "the port has never read it"


def survey(input_file: str, graph=None) -> tuple[Row, ...]:
    """Every switch-shaped integer in `input_file`, classified against the tree."""
    from functional_process.cottax.indat import UNPORTED, switches_from_indat

    if graph is None:
        from functional_process.cottax.indat import GRAPH as graph

    fields = factory_fields()
    pinned = pinned_switches(graph)

    rows = []
    for name, value in sorted(switches_from_indat(input_file).items()):
        if name in fields:
            reason = UNPORTED.get((name, value))
            if reason:
                detail = "no occupant: " + reason.split(":")[0]
            elif unoccupied := unoccupied_registries(name, value):
                # Neither an occupant nor a recorded refusal: `_slot_occupant` raises
                # `ValueError` here, not `NotImplementedError`. Named as its own kind
                # because the fix is different -- either write the arm, or record why
                # PROCESS's arm cannot be one (an empty `None` occupant is the usual
                # answer for a value at which PROCESS computes nothing).
                detail = (
                    "no occupant AND no recorded reason: "
                    f"{', '.join(unoccupied)} has no entry for this value, so "
                    "`_slot_occupant` refuses it as a typo"
                )
            else:
                detail = "the factory dispatches on it"
            rows.append(Row(name, value, "factory", detail))
        elif name in pinned:
            held = sorted(pinned[name])
            agrees = "agrees" if value in held else f"DISAGREES, tree holds {held}"
            rows.append(
                Row(name, value, "pinned", f"hardcoded as a static kwarg; {agrees}")
            )
        else:
            if name in NOT_TOPOLOGY:
                rows.append(Row(name, value, "not-topology", NOT_TOPOLOGY[name]))
                continue
            readers = readers_in_process(name)
            detail = SHAPE.get(name) or _how_the_port_reads(name, graph)
            rows.append(
                Row(
                    name,
                    value,
                    "unknown",
                    detail,
                    readers,
                    any(r in COOLPROP_MODULES for r in readers),
                )
            )
    return tuple(rows)


def assembly_verdict(input_file: str) -> str:
    """What `machine_from_indat` actually does with this file, in one line.

    **The switch table cannot answer this and it took the ST wave to notice.** Every row
    above is keyed on a name that appears in the `IN.DAT`; a slot dispatched on a
    *derived* arm index -- `pf_coil_system_arm`, `fw_blkt_vv_shape_arm`,
    `centrepost_neutronics_arm`, a dozen more -- has no such name, so its refusal is
    invisible to the table however carefully the table is read. Both tracked spherical
    tokamaks were re-surveyed on 2026-08-29 as "exactly four switch values away" and
    were in fact also refused by `pf_coil_system_arm`, which no column could show.

    One assembly attempt costs a second and is exact, so the report ends with it.
    """
    from functional_process.cottax.indat import machine_from_indat

    try:
        machine_from_indat(input_file)
    except NotImplementedError as refusal:
        return f"ASSEMBLY REFUSED (first blocker only): {refusal}"
    except ValueError as refusal:
        return f"ASSEMBLY REFUSED (unknown value): {refusal}"
    return "ASSEMBLES."


def report(input_file: str) -> str:
    """`survey` as a table, with the counts that size the work."""
    rows = survey(input_file)
    kinds = {
        kind: [r for r in rows if r.verdict == kind]
        for kind in ("factory", "pinned", "unknown", "not-topology")
    }
    lines = [f"{input_file}: {len(rows)} switch-shaped integer(s)", ""]
    for kind, group in kinds.items():
        lines.append(f"{kind.upper()} ({len(group)})")
        for row in group:
            flag = "  [CoolProp]" if row.coolprop else ""
            lines.append(f"  {row.name:<32} = {row.value:<4} {row.detail}{flag}")
            if row.verdict == "unknown" and row.readers:
                lines.append(
                    f"{'':36}   read in "
                    + ", ".join(row.readers[:4])
                    + (f" (+{len(row.readers) - 4})" if len(row.readers) > 4 else "")
                )
        lines.append("")
    blocked = [r for r in kinds["unknown"] if r.coolprop]
    shapes = [r for r in kinds["unknown"] if r.name in SHAPE]
    disagree = [r for r in kinds["pinned"] if "DISAGREES" in r.detail]
    lines.append(
        f"{len(rows)} integer(s): {len(kinds['not-topology'])} not topology, "
        f"{len(kinds['factory'])} the factory already dispatches on, "
        f"{len(kinds['pinned'])} pinned in the tree ({len(disagree)} of which this file "
        f"contradicts), {len(kinds['unknown'])} new "
        f"({len(shapes)} of them counts rather than model choices)."
    )
    lines.append(
        f"{len(blocked)} new one(s) sit in a neighbourhood that reaches CoolProp and "
        f"want the wrapping policy (`next_steps.md` §5) checked before scheduling."
    )
    if disagree:
        lines.append("")
        lines.append(
            "The tree contradicts this file on: "
            + ", ".join(r.name for r in disagree)
            + " -- these are `switch_kwarg_survey.md` band (b) slots, and they "
            "are the first tokamak deliverable, not a prerequisite to it."
        )
    lines.append("")
    lines.append(assembly_verdict(input_file))
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    print(
        report(
            sys.argv[1]
            if len(sys.argv) > 1
            else "tests/regression/input_files/large_tokamak_eval.IN.DAT"
        )
    )
