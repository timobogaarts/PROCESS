"""Read an `IN.DAT`."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from functional_process.vocabulary.input_variables import INPUT_VARIABLES, InputDecl

_ASSIGNMENT = re.compile(
    r"([a-zA-Z0-9_]+)(?:\(([0-9]+)\))?[ ]*=[ ]*([ +\-a-zA-Z0-9.,]+).*"
)
"""`parse_input_file`'s own line grammar, character class included."""


# --------------------------------------------------------------------------- values


@dataclass(frozen=True, slots=True)
class ArrayInput:
    """What an `IN.DAT` said about one array field, sparsely."""

    elements: tuple[tuple[int, float], ...]
    zero_filled: bool = False

    def as_dict(self) -> dict[int, float]:
        return dict(self.elements)

    def dense(self, length: int, fill: float = 0.0) -> list[float]:
        """`length` elements, unset ones at `fill`. Only lossless when `zero_filled`."""
        out = [fill] * length
        for index, value in self.elements:
            out[index] = value
        return out


@dataclass(frozen=True, slots=True)
class Assignment:
    """One `name = value` line, as the file spelled it."""

    line: int
    name: str
    index: int | None
    """Fortran 1-based, or `None` for a whole-name assignment."""
    text: str


OPTIMISATION_RUN_MODE = 1
EVALUATION_RUN_MODE = -2
"""`process.data_structure.numerics.PROCESSRunMode`, vendored (§23.2)."""


@dataclass(frozen=True, slots=True)
class Problem:
    """§23.4's parts, as the file states them."""

    ixc: tuple[int, ...] = ()
    icc: tuple[int, ...] = ()
    i_figure_merit: int | None = None
    n_equality_constraints: int | None = None
    n_inequality_constraints: int | None = None
    i_process_run_mode: int | None = None
    """`None` when the file names none, which is `OPTIMISATION_RUN_MODE` -- unresolved
    here for the same reason `n_equality_constraints`' sentinel is.
    """

    @property
    def is_evaluation(self) -> bool:
        """Whether this file states a **root find** rather than an optimisation."""
        if self.i_process_run_mode is None:
            return False
        return int(self.i_process_run_mode) == EVALUATION_RUN_MODE


@dataclass(frozen=True, slots=True)
class Imported:
    """Everything one `IN.DAT` says, and nothing derived from it."""

    path: str
    values: dict[tuple[str, str], Any] = field(default_factory=dict)
    """`(area, field) -> scalar | str | ArrayInput`, for names that address a field."""
    present: frozenset[str] = frozenset()
    """Every input name the file mentions, as spelled (lowercased)."""
    assignments: tuple[Assignment, ...] = ()
    problem: Problem = Problem()
    unknown: tuple[Assignment, ...] = ()
    """Assignments whose name is not in PROCESS's registry."""
    errors: tuple[str, ...] = ()
    """Lines that did not parse or cast. Same reason: reported, never raised (§24.2)."""

    # -------------------------------------------------------------- presence (§24.2)

    def named(self, name: str) -> bool:
        """Did the file mention this `IN.DAT` name? The irreducible question."""
        return name.lower() in self.present

    @property
    def present_paths(self) -> frozenset[tuple[str, str]]:
        """The `(area, field)` pairs the file names."""
        return frozenset(self.values)

    # ----------------------------------------------------------------- values

    def get(self, area: str, name: str, default: Any = None) -> Any:
        return self.values.get((area, name), default)

    def scalars(self) -> dict[tuple[str, str], float | int | str]:
        """Just the non-array values, for a caller that wants the old shape."""
        return {k: v for k, v in self.values.items() if not isinstance(v, ArrayInput)}

    def raw_values(self) -> dict[str, Any]:
        """The same values under the `raw` root: `.raw.<area>.<field>` (§24.2 item 2).
        """
        return {
            f".raw.{area}.{name}": value for (area, name), value in self.values.items()
        }


# --------------------------------------------------------------------------- reading


def _cast(text: str, kind: type) -> Any:
    """`validate_variable`'s cast, minus every bound it checks."""
    if kind is str:
        return text
    return kind(text.lower().replace("d", "e"))


def read_indat(path: str | Path) -> Imported:
    """Parse an `IN.DAT`. Never raises on content -- see `Imported.errors`."""
    path = Path(path)
    values: dict[tuple[str, str], Any] = {}
    present: set[str] = set()
    assignments: list[Assignment] = []
    unknown: list[Assignment] = []
    errors: list[str] = []
    ixc: list[int] = []
    icc: list[int] = []
    arrays: dict[tuple[str, str], tuple[dict[int, float], bool]] = {}

    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped[0] == "*":
            continue
        match = _ASSIGNMENT.match(stripped)
        if match is None:
            errors.append(f"line {line_no}: unparsed ({stripped})")
            continue

        name, index_text, value_text = match.groups()
        name = name.lower()
        index = None if index_text is None else int(index_text)
        assignment = Assignment(line_no, name, index, value_text.strip())
        assignments.append(assignment)

        decl: InputDecl | None = INPUT_VARIABLES.get(name)
        if decl is None:
            unknown.append(assignment)
            continue
        present.add(name)

        if name == "ixc" or name == "icc":
            try:
                (ixc if name == "ixc" else icc).append(int(assignment.text))
            except ValueError:
                errors.append(
                    f"line {line_no}: {name} is not an integer ({assignment.text})"
                )
            continue

        place = (decl.module, name)
        try:
            if "," in assignment.text and decl.array:
                # `parse_input_file`'s whole-array form: zero the array, then fill 1..n.
                items = [v.strip() for v in assignment.text.split(",")]
                elements = {i: _cast(v, decl.type) for i, v in enumerate(items) if v}
                arrays[place] = (elements, True)
            elif "," in assignment.text:
                errors.append(
                    f"line {line_no}: '{name}' is not an array but lists values"
                )
            elif decl.array:
                if index is None:
                    errors.append(f"line {line_no}: '{name}' is an array with no index")
                    continue
                slot, was_zeroed = arrays.get(place, ({}, False))
                slot[index - 1] = _cast(assignment.text, decl.type)
                arrays[place] = (slot, was_zeroed)
            else:
                if index is not None:
                    errors.append(f"line {line_no}: '{name}' is not an array")
                    continue
                values[place] = _cast(assignment.text, decl.type)
        except ValueError:
            errors.append(f"line {line_no}: cannot cast '{name}' ({assignment.text})")

    for place, (elements, zero_filled) in arrays.items():
        values[place] = ArrayInput(tuple(sorted(elements.items())), zero_filled)

    return Imported(
        path=str(path),
        values=values,
        present=frozenset(present),
        assignments=tuple(assignments),
        problem=Problem(
            ixc=tuple(ixc),
            icc=tuple(icc),
            i_figure_merit=values.get(("numerics", "i_figure_merit")),
            n_equality_constraints=values.get(("numerics", "n_equality_constraints")),
            n_inequality_constraints=values.get((
                "numerics",
                "n_inequality_constraints",
            )),
            i_process_run_mode=values.get(("numerics", "i_process_run_mode")),
        ),
        unknown=tuple(unknown),
        errors=tuple(errors),
    )
