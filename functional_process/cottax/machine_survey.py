"""What a given `IN.DAT` asks for that this port cannot yet give it."""

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
`blanket_library`.
"""

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
"""Integers an `IN.DAT` carries that are not topology decisions, and why."""

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
"""How the fields the factory dispatches on are read back out of `indat.py`."""


@dataclasses.dataclass(frozen=True)
class Row:
    """One switch of one input file, and what this port can do with it."""

    name: str
    value: int
    verdict: str
    """`factory` (a slot dispatches on it) / `pinned` (the tree hardcodes it as a static
    kwarg) / `unknown` (the port has never read it).
    """
    detail: str = ""
    readers: tuple[str, ...] = ()
    coolprop: bool = False
    """Whether *some* module reading this switch also reaches CoolProp."""


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
    """`switch field -> the registry names `_slot_occupant` looks it up in`."""
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
"""


def unoccupied_registries(name, value) -> tuple[str, ...]:
    """The registries dispatching on `name` that hold no occupant for `value`."""
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
    """What `machine_from_indat` actually does with this file, in one line."""
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
