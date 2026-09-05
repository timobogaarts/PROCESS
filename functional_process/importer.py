"""Read an `IN.DAT`. The legacy-input arm of the target architecture (§24).

*models + default solvers / cycle cutters + **import legacy PROCESS input file***. Today
the port reads integer switches out of the file text (`indat.switches_from_indat`) and
PROCESS's `SingleRun` supplies every number. This module is the replacement for the
second half, and it emits four things -- three of which no node can (§24.2, §24.3):

1. **Values, scalars and arrays.** Only scalars were read from file text before, and the
   gap already produced a wrong answer: a last-wins name scan answered the ten-element
   `.pf_coil.zref` with the `1.0` from `zref(10) = 1.0` (§22.6). An indexed or
   comma-listed assignment is an `ArrayInput` here, not a scalar.
2. **Presence** -- the set of names the file *mentions*. Irreducible: it is a property of
   the text, not of any value, so no node recovers it (§24.2 item 1). `init.py` writes
   four fields from it, and the live defect at `indat.py:4420-4428` is exactly what its
   absence costs -- that code infers presence for `i_f_dr_tf_plasma_case` and
   `tfc_sidewall_is_fraction` by scanning `switches_from_indat` for names that **are not
   declared PROCESS inputs**, so the scan can only ever return `0`, and one of
   `st_regression`'s two missing producers follows. `named()` is the answer that code
   needs. Not fixed here -- `indat.py` is owned elsewhere.
3. **The `raw` namespace** (`raw_values()`, `.raw.<area>.<field>`). §24.2 item 2: the
   eight sentinel resolutions (`eff_tf_cryo = -1.0 -> 0.13`) read and write one path, so
   they cannot be nodes as stated. Under a separate root they can: raw -> resolved is an
   edge. It also disarms the two sentinels that *look* like answers (`eyoung_ins` at
   `1e8`, `eyoung_cond_axial` at `6.6e8`, both replaced by two orders of magnitude) --
   as a raw value they cannot be mistaken for a user's number the way a flat defaults
   table mistakes them.
4. **The problem statement** -- `ixc`, `icc`, `i_figure_merit`, and
   `i_process_run_mode` (§23.4, §24.3). `ixc`/`icc` are the only two names in PROCESS's
   registry that set no field at all; they reach the `DataStructure` through an
   `additional_actions` hook, which is a parser concern by construction.
   `i_process_run_mode` is what says *which kind of problem the file states* -- `-2`
   makes PROCESS root-find the equalities with `fsolve` and form no objective at all --
   and reading it here is what let the port stop building an `Optimise` for the two
   `_eval` files (`Problem.is_evaluation`, §24.10).

**Deliberately out of scope, and each is the next layer.** Sentinel resolution;
`init.py`/`st_init`/`initialise_imprad` derivations; validation raises (`init.py` holds
51, and a dozen encode physics-validity ranges the port does not enforce at all); and any
node. This is what those sit on, not those.

**No `process` import** (§23). The name -> field table is vendored at
`vocabulary/input_variables.py` and asserted equal to PROCESS's in
`functional_process/tests/test_importer.py`, per §23.2's standing rule: vendor for
runtime, assert equality in tests.

**Fidelity.** The line grammar, the Fortran `d`->`e` exponent fix, the "a comma list
zeroes the array first" rule and the 1-based array index are `parse_input_file`'s own,
transcribed rather than reinvented; the oracle test diffs every parsed value against a
real `SingleRun`'s `DataStructure` taken **before** `set_active_constraints`, because
`init.py` destroys three genuine inputs and `st_init` overwrites nine more
(`_audit/init_audit.md`), so a post-init comparison could not tell a parse bug from a
derivation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from functional_process.vocabulary.input_variables import INPUT_VARIABLES, InputDecl

_ASSIGNMENT = re.compile(
    r"([a-zA-Z0-9_]+)(?:\(([0-9]+)\))?[ ]*=[ ]*([ +\-a-zA-Z0-9.,]+).*"
)
"""`parse_input_file`'s own line grammar, character class included.

`*` is absent from the value class, so `icc = 1 * Beta` yields `"1 "` and the trailing
`.*` eats the comment. The index is Fortran 1-based.
"""


# --------------------------------------------------------------------------- values


@dataclass(frozen=True, slots=True)
class ArrayInput:
    """What an `IN.DAT` said about one array field, sparsely.

    `elements` is 0-based, so it indexes the `DataStructure` array directly.
    `zero_filled` records which of PROCESS's two array spellings was used, and the
    difference is not cosmetic: `a = 1,2,3` makes `parse_input_file` do `array[:] = 0.0`
    *before* filling, while `a(2) = 1.0` leaves every other element at its dataclass
    default. The importer holds no defaults and no shapes, so it cannot densify on its
    own -- `dense(length)` is offered for a caller that knows one, and only for the
    zero-filled form is that lossless.
    """

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
"""`process.data_structure.numerics.PROCESSRunMode`, vendored (§23.2).

`1` is `numerics.py:150`'s dataclass default, so a file that names no run mode optimises.
`-2` is the mode `main.py:452-456` answers by replacing VMCON with `scipy.optimize.fsolve`
over the **equalities alone** -- `_Fsolve.evaluate_eq_cons` calls
`fcnvmc1(n, self.meq, ...)`, and `_Fsolve.solve` ends with `self.objf = None`.
"""


@dataclass(frozen=True, slots=True)
class Problem:
    """§23.4's parts, as the file states them.

    Integer IDs, not paths: translating `ixc = [4, 6, 29]` into `VarPath`s is a table
    lookup (`vocabulary.iteration_variables`) and belongs to whoever states the problem,
    not to the reader of the text. `n_equality_constraints` is `None` when the file does
    not set it, which is the `-1` sentinel `init.py` resolves to `count - n_inequality`
    -- the resolution is not done here.

    **`i_process_run_mode` is part of the problem statement, not a value**, and it is the
    one field here that changes *which kind of problem* the file states. Carried since
    2026-08-31, when `_audit/next_steps.md` §24.10 measured that the port was building an
    `Optimise` for two files PROCESS root-finds.
    """

    ixc: tuple[int, ...] = ()
    icc: tuple[int, ...] = ()
    i_figure_merit: int | None = None
    n_equality_constraints: int | None = None
    n_inequality_constraints: int | None = None
    i_process_run_mode: int | None = None
    """`None` when the file names none, which is `OPTIMISATION_RUN_MODE` -- unresolved
    here for the same reason `n_equality_constraints`' sentinel is."""

    @property
    def is_evaluation(self) -> bool:
        """Whether this file states a **root find** rather than an optimisation.

        The discriminator is the file's own `i_process_run_mode = -2` and nothing
        inferred. Two weaker rules were considered and are wrong:

        - *"no `i_figure_merit`"* -- `large_tokamak_nof` names none either and optimises
          (its `i_figure_merit = 1` is stated, but a file may omit it and still optimise
          on `numerics.py:154`'s default of `7`).
        - *"square"* (`len(icc[:n_equality]) == len(ixc)`) -- `helias_5b` is square
          (3 equalities, 3 iteration variables) and PROCESS runs **VMCON** on it, in 3
          iterations, because it names `i_process_run_mode = 1`. Squareness is a
          *consequence* of the evaluation mode, not its cause, and a caller should check
          it as a consistency test rather than use it as the test.
        """
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
    """Assignments whose name is not in PROCESS's registry. `parse_input_file` raises on
    these; collected instead, because refusing to read a file is not this layer's call."""
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

        A namespacing, not new data. It exists so a sentinel resolution can be a node
        with a read and a distinct write instead of a self-loop.
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
