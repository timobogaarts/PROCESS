"""Answer the boundary a graph declares -- with a **reason**, not just a value.

`boundary(graph)` says what a machine reads and does not produce. Nothing said where
those values should come from, so they came wholesale from PROCESS's `DataStructure`
after `init_process`. `_audit/next_steps.md` §22.1 lists four bugs that seeding caused,
every one invisible *because the seed supplied the answer*.

This module is the provider. It resolves each boundary path to `(value, reason)`:

- `input`     -- a declared PROCESS input this `IN.DAT` names.
- `default`   -- a declared PROCESS input the file does not name.
- `derived`   -- not a declared input; `init_process` wrote it.
- `computed`  -- PROCESS's **models** write it every pass. A missing producer, i.e. a
  bug. Outranks `input` deliberately: `.buildings.dz_tf_cryostat` is a genuine
  `InputVariable` sitting at `2.5` that `cryostat.py:58-60` overwrites before its only
  live reader, so a defaults table would answer it confidently and be wrong (§22.3).
- `unwritten` -- not a declared input, and PROCESS's pipeline does not write it on this
  configuration, so the bare dataclass default stands. Two things live here and the pins
  tell them apart *between* files rather than a third reason doing it inside one: a
  hard-coded constant nothing anywhere writes (the `.costs.UC*` unit costs), and
  `.physics.dlamie` on a stellarator, whose only writer in `process/` is on the tokamak
  path -- so PROCESS computes with a value nothing ever wrote. That second kind shows up
  as `unwritten` on one machine's pin and `computed`/owned on another's. There is no
  right answer to supply for it; that is a reason, not a crash.
- `solver`    -- a path the problem owns (an active `ixc` entry). Not supplied -- which
  is why the signature takes an owned path set as well as the graph. §22.2 says "the
  problem"; **narrowed to the owned paths**, because of a problem's four parts only
  `ixc` reaches here -- `icc` and `i_figure_merit` change which nodes are live, and
  that is already in the assembled `graph`. Paths, not integer IDs: the IDs are
  throwaway indirection over what are structurally paths.
- `guess`     -- a `Start` port for a driven unknown; `boundary`'s own second category.
- `minted`    -- a path with no `.area.field` to resolve at all.

Where the value comes from is tracked apart from why, as `source`:

- `indat`    -- parsed out of the input file's own text.
- `defaults` -- a bare `DataStructure()` field.
- `process`  -- the seed. **This is the transitional half**: §22.3's rule is that the
  seed becomes the *oracle*, not the source, so a path answered from `indat`/`defaults`
  is answered independently and a path answered from `process` is not yet. "N of M
  answered independently" is the number that has to move.

The diff is the deliverable: every independently answered path is checked against the
seed, and a disagreement is a finding -- a path classified `input`/`default` that
`init.py` in fact moved. Regenerate the pins (never hand-edit) with::

    $PY -m functional_process.cottax.provider --write

Where this stops, and what carries on
-------------------------------------
Everything here needs the seed -- to classify against, and to answer the `derived`,
`computed` and array rows from. `install` therefore writes its answers *into* a copy of
a `DataStructure`, so even the 89-92 % it answers independently reaches `mdf.seed`
through a PROCESS object (§22.7). **`functional_process/cottax/native.py` is the module with no
`DataStructure` in it**: it answers `.<area>.<field>` out of `importer.read_indat` and a
vendored table of PROCESS's dataclass defaults, and it has no classification at all. The
two are complementary rather than alternatives -- this one says which of that one's
answers are wrong, which is what turns a native failure into a work list (§22.8).
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from dataclasses import dataclass

from cottax.spec import VarPath

from functional_process.cottax.boundary import GUESSED, boundary

CONFIGURATIONS = (
    "tests/regression/input_files/stellarator_helias.IN.DAT",
    "tests/regression/input_files/helias_5b.IN.DAT",
    "tests/regression/input_files/large_tokamak_nof.IN.DAT",
    "tests/regression/input_files/large_tokamak_eval.IN.DAT",
    "tests/regression/input_files/low_aspect_ratio_DEMO.IN.DAT",
    "tests/regression/input_files/spherical_tokamak_eval.IN.DAT",
    "tests/regression/input_files/st_regression.IN.DAT",
)
"""Every in-scope file, in `run_cold_matrix.CONFIGURATIONS`' order. Five assemble today;
the two spherical tokamaks are refused by `machine_from_indat` and get no pin, which
`--write` reports rather than raising -- a runner that dies on the sixth entry says
nothing about the seventh."""

PIN_DIR = os.path.dirname(__file__)

# ----------------------------------------------------------------- reasons and sources

INPUT, DEFAULT, DERIVED = "input", "default", "derived"
COMPUTED, UNWRITTEN = "computed", "unwritten"
SOLVER, GUESS, MINTED = "solver", "guess", "minted"


INDAT, DEFAULTS, PROCESS = "indat", "defaults", "process"

INDEPENDENT = frozenset({INDAT, DEFAULTS})
"""Sources that do not read PROCESS's seed. The independence count is over these."""

BUGS = frozenset({COMPUTED})
"""Reasons that name a defect rather than an answer -- `computed` is a missing
producer. `unwritten` is not here: it is PROCESS being incoherent, not the port."""


@dataclass(frozen=True)
class Answer:
    """One boundary path resolved. `value` is `None` only when nothing could read it."""

    var: VarPath
    reason: str
    source: str
    value: object = None
    seeded: object = None

    @property
    def independent(self) -> bool:
        return self.source in INDEPENDENT

    @property
    def place(self) -> tuple[str, str] | None:
        """`(area, field)` this path addresses, or `None` if it addresses no field."""
        located = _area_field(self.var)
        return None if located is None else located[0]

    @property
    def is_element(self) -> bool:
        """Does this path address one slot of an array field?"""
        located = _area_field(self.var)
        return bool(located and located[1])

    @property
    def agrees(self) -> bool:
        """Is the provider's value the seed's? Meaningless unless `independent`."""
        return _same(self.value, self.seeded)

    def path_str(self) -> str:
        return self.var.path_str()


# ----------------------------------------------------------------- what the file names

_ASSIGNMENT = re.compile(r"\s*([A-Za-z_]\w*)\s*(\(\s*\d+\s*\))?\s*=(.*)$")


def named_in(input_file: str) -> dict[str, str | None]:
    """`{lowercased name: the text right of the `=`}` for every assignment in the file.

    Deliberately a name scan, not a parser: `indat.numbers_from_indat` and its siblings
    already refuse to be one, for the reason recorded there. What this needs is the *set*
    of names a file sets -- the value is taken only when it is a bare scalar.

    **An indexed assignment maps to `None`, not to its text.** `zref(10) = 1.0` sets one
    element of a ten-element array, and a last-wins dict over the ten lines would answer
    the whole array with the last element -- measured, on `large_tokamak_eval`, before
    the index group was captured. `None` means "this file names it, and the text is not
    the field's value", which is exactly what the caller needs to fall back on the seed.
    """
    found: dict[str, str | None] = {}
    with open(input_file, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("*"):
                continue
            match = _ASSIGNMENT.match(stripped)
            if not match:
                continue
            name, index, text = match.group(1).lower(), match.group(2), match.group(3)
            found[name] = None if index else text.split("*")[0].strip()
    return found


def _scalar(text: str) -> float | None:
    """`text` as a float, or `None` if it is not one bare number."""
    if not text or "," in text:
        return None
    try:
        return float(text.replace("d", "e").replace("D", "e"))
    except ValueError:
        return None


def declared_inputs() -> dict[tuple[str, str], str]:
    """`{(area, field): the name an IN.DAT spells it with}` from PROCESS's own registry.

    `INPUT_VARIABLES[name].module` is the dotted attribute path `parse_input_file` walks
    on the `DataStructure`, and `target_name or name` is the field it sets -- so the
    registry already carries exactly the `(area, field)` pair a boundary `VarPath` is.
    Entries with a non-string module or `set_variable = False` (`ixc`, `icc`) address no
    field and are dropped.
    """
    from process.core.input import INPUT_VARIABLES  # noqa: PLC0415

    out: dict[tuple[str, str], str] = {}
    for name, config in INPUT_VARIABLES.items():
        if not isinstance(config.module, str) or not config.set_variable:
            continue
        out[config.module, config.target_name or name] = name
    return out


# ----------------------------------------------------------------- the resolution


def _area_field(var: VarPath) -> tuple[tuple[str, str], bool] | None:
    """`((area, field), is_element)` for `.area.field` or `.area.field[i]`, else `None`.

    **Index-addressed paths keep their base field**, unlike `cold_start._area_field`,
    which returns `None` for them. The reason is a detection hole: twelve of
    `large_tokamak_eval`'s boundary entries are
    `.impurity_radiation.f_nd_impurity_electron_array[i]`, and dropping them
    unclassified would hide a missing producer whose parent array PROCESS writes. The
    *value* still comes from the seed for an element -- resolving one array slot out of
    an `IN.DAT`'s `f_nd_impurity_electrons(03)` spelling is not done here -- but the
    reason is answered.
    """
    keys = var.path_str().lstrip(".").split(".")
    if len(keys) != 2:
        return None
    field, _, index = keys[1].partition("[")
    return (keys[0], field), bool(index)


def _get(data, area: str, field: str):
    try:
        return getattr(getattr(data, area), field)
    except AttributeError:
        return None


def _is_computed(var, place, is_element, computed, seed, final) -> bool:
    """Does PROCESS's own pipeline write this path?

    `computed` is measured per **field**, so an array whose first element PROCESS
    normalises marks every element written. Measured on both large tokamaks: PROCESS
    moves `.impurity_radiation.f_nd_impurity_electron_array[0]` (and `[1]` on
    `large_tokamak_eval`) and no other element, while the boundary holds `[2]`..`[13]` --
    so the field-level set would report eleven missing producers that are not. For an
    element path the two `cold_state` snapshots are compared **at that index** instead,
    which is exact. `boundary.unproduced_but_computed` sidesteps this by never
    classifying an indexed path at all.
    """
    if not is_element:
        return place in computed
    if final is None:
        return place in computed
    from cottax.tools.pytree import get_at  # noqa: PLC0415

    try:
        return not _same(get_at(seed, var.keys), get_at(final, var.keys))
    except (AttributeError, IndexError, KeyError, TypeError):
        return place in computed


def _same(a, b) -> bool:
    import numpy as np  # noqa: PLC0415

    if a is None or b is None:
        return a is b
    try:
        left, right = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    except (TypeError, ValueError):
        return bool(a == b)
    return left.shape == right.shape and bool(
        np.array_equal(np.nan_to_num(left), np.nan_to_num(right))
    )


def answer(
    var: VarPath,
    kind: str,
    *,
    design: frozenset[VarPath],
    computed: frozenset[tuple[str, str]],
    inputs: dict[tuple[str, str], str],
    named: dict[str, str | None],
    seed,
    final,
    defaults,
) -> Answer:
    """One path's `(value, reason)`. The ladder is ordered; see this module's docstring.

    `computed` outranks `input`, which is the whole point of the reason column.
    """
    if kind == GUESSED:
        return Answer(var, GUESS, PROCESS)
    located = _area_field(var)
    if located is None:
        return Answer(var, MINTED, PROCESS)
    place, is_element = located
    seeded = _get(seed, *place)
    if var in design:
        return Answer(var, SOLVER, PROCESS, seeded, seeded)
    if _is_computed(var, place, is_element, computed, seed, final):
        return Answer(var, COMPUTED, PROCESS, seeded, seeded)
    default = _get(defaults, *place)
    name = inputs.get(place)
    if name is not None and name in named:
        value = None if is_element else _scalar(named[name] or "")
        if value is None:  # an array, a list or a string -- not resolvable from text yet
            return Answer(var, INPUT, PROCESS, seeded, seeded)
        return Answer(var, INPUT, INDAT, value, seeded)
    if name is not None:
        source, value = (PROCESS, seeded) if is_element else (DEFAULTS, default)
        return Answer(var, DEFAULT, source, value, seeded)
    if _same(seeded, default):
        source, value = (PROCESS, seeded) if is_element else (DEFAULTS, default)
        return Answer(var, UNWRITTEN, source, value, seeded)
    return Answer(var, DERIVED, PROCESS, seeded, seeded)


def provide(
    graph,
    owned: Iterable[VarPath],
    input_file: str,
    seed=None,
    computed=None,
    final=None,
) -> tuple[Answer, ...]:
    """Answer every boundary entry of `graph`, in `boundary`'s own stable order.

    `owned` is **the set of paths the problem owns**, not the problem and not an `ixc`
    list. Of a PROCESS problem's four parts only `ixc` reaches here: `icc` and
    `i_figure_merit` change which nodes are live, and that is already in `graph`.
    Taking paths rather than integer IDs is deliberate -- the IDs are throwaway
    indirection over what are structurally paths, and a provider that speaks paths
    needs no change when the problem is stated structurally.

    `seed`/`computed` are `cold_state(input_file)`'s two halves; passed in when the
    caller already has them, since that measurement is the expensive part.
    """
    from process.core.model import DataStructure  # noqa: PLC0415

    if seed is None or computed is None:
        from functional_process.cottax.cold_start import cold_state  # noqa: PLC0415

        state = cold_state(input_file)
        seed = state.seed if seed is None else seed
        computed = state.written if computed is None else computed
        final = state.process if final is None else final
    context = {
        "design": frozenset(owned),
        "computed": frozenset(computed),
        "inputs": declared_inputs(),
        "named": named_in(input_file),
        "seed": seed,
        "final": final,
        "defaults": DataStructure(),
    }
    return tuple(answer(var, kind, **context) for kind, var in boundary(graph))


def disagreements(answers: Iterable[Answer]) -> tuple[Answer, ...]:
    """Independently answered paths whose value is not the seed's -- the findings.

    §22.3's move, for the input side: the seed is the oracle, so a `default` that
    `init.py` in fact moved, or an `input` PROCESS overrode after parsing it, shows up
    here instead of being silently believed.
    """
    return tuple(a for a in answers if a.independent and not _same(a.value, a.seeded))


def tally(answers: Iterable[Answer]) -> dict[str, int]:
    """`{reason: count}` plus `answered`/`from_process` -- §22.3's number that moves."""
    out: dict[str, int] = {}
    for a in answers:
        out[a.reason] = out.get(a.reason, 0) + 1
        out["answered" if a.independent else "from_process"] = (
            out.get("answered" if a.independent else "from_process", 0) + 1
        )
    return out


# ----------------------------------------------------------------- consuming it

NOT_SUPPLIED = frozenset({SOLVER, GUESS})
"""Reasons that are not boundary *values* at all and so do not belong in a ratio of
"answered independently". A `guess` is a `Drive`'s `Start` port and a `solver` row is
owned by the problem; neither is something a provider could supply, so counting them in
the denominator understates the ratio. `tally`'s `answered`/`from_process` pair is over
*every* row, which is why `installed` reports both denominators."""


def _like(value, seeded):
    """`value` in `seeded`'s type, so an `IN.DAT` float does not land in an int field.

    The file text is parsed as a float (`_scalar`), and a switch or a count read back as
    `1.0` instead of `1` compares unequal to its `IntEnum` and silently reroutes a
    branch. Only integral seeds are coerced; everything else is written as it stands.
    """
    import numpy as np  # noqa: PLC0415

    if isinstance(seeded, bool | np.bool_):
        return type(seeded)(value)
    if isinstance(seeded, int | np.integer) and not isinstance(value, bool):
        return type(seeded)(value)
    return value


def install(answers: Iterable[Answer], data, *, disagreeing: bool = False):
    """Write every independently answered value into `data`, in place. **The consumer.**

    `data` is a `DataStructure` -- a *copy* of the seed, which the caller owns -- and
    this writes the provider's answer over the seed's for each path the provider answers
    without reading PROCESS. It is the seeding machinery's input that moves, not the
    machinery: every downstream reader (`mdf.seed`, `sand_harness.mda_env`,
    `run_sand_harness._seed`) goes on reading a `DataStructure` exactly as before, and a
    path the provider does not answer keeps the seed's value. That is what makes "N of M
    boundary values came from the input file and the dataclass defaults" a property of a
    *solve* rather than of a classification.

    `disagreeing` decides what happens at the `off` rows -- the 5 to 8 paths per
    configuration where the provider's answer is not the seed's:

    - `False` (the default) holds them at the seed. The substitution is then inert **by
      construction** -- every written value is bit-identical to the one it replaces -- so
      a table that moves under it is a defect in this function, not a finding about
      `init.py`. This is the mode a comparable cold matrix runs in, and it is honest only
      because the disagreements are pinned and named elsewhere.
    - `True` takes the provider at its word everywhere, which is §22.3's real question:
      what does a solve do when `init.py`'s corrections are not applied? For the
      stellarators that is a machine with a central solenoid and a 1000 s burn instead
      of a steady-state one, so the answer is expected to be "a lot".

    A path whose answer is `None` is **skipped and counted** (`nothing`), not written:
    `.vacuum.l1` and four siblings default to `None` and the seed holds the same `None`,
    so there is nothing to supply and nothing to disagree about. Counting them keeps
    `written + held + nothing + from_process == supplied`, so a path cannot go missing
    between the classification and the solve without the arithmetic saying so.

    Returns `(counts, moved)`: `counts` as in `installed`, `moved` the `(path, provider,
    seed)` triples actually written over a different value.
    """
    import copy  # noqa: PLC0415

    answers = tuple(answers)
    moved, written, held, nothing = [], 0, 0, 0
    for a in answers:
        if not a.independent:
            continue
        if a.value is None:
            nothing += 1
            continue
        place = a.place
        if place is None or a.is_element:
            # An element is answered from the seed (`answer` never marks one
            # independent), so this is a guard, not a branch: writing `a.value` to
            # `place` would overwrite the whole array with one slot's value.
            continue
        area = getattr(data, place[0], None)
        if area is None or not hasattr(area, place[1]):
            continue
        if not a.agrees:
            if not disagreeing:
                held += 1
                continue
            moved.append((a.path_str(), a.value, a.seeded))
        value = _like(a.value, a.seeded)
        setattr(area, place[1], copy.deepcopy(value))
        written += 1
    return installed(answers, written, held, nothing), tuple(moved)


def installed(
    answers: Iterable[Answer], written: int, held: int, nothing: int = 0
) -> dict[str, int]:
    """The counts a run reports: how much of its boundary did not come from PROCESS.

    `supplied` is the denominator that means something -- every boundary path minus the
    `solver` and `guess` rows, which no provider could answer (see `NOT_SUPPLIED`).
    `paths` is the raw total, reported beside it because §22.6's published ratios are
    over that one and the two must not be confused.
    """
    answers = tuple(answers)
    supplied = [a for a in answers if a.reason not in NOT_SUPPLIED]
    return {
        "paths": len(answers),
        "supplied": len(supplied),
        "independent": sum(1 for a in supplied if a.independent),
        "written": written,
        "held": held,
        "nothing": nothing,
        "from_process": sum(1 for a in supplied if not a.independent),
    }


# ----------------------------------------------------------------- the pin


def stem(input_file: str) -> str:
    name = os.path.basename(input_file)
    return name.removesuffix(".IN.DAT")


def pin_path(input_file: str) -> str:
    """One pin per configuration. A boundary is a property of one assembled graph --
    `boundary.TOKAMAK_PIN`'s reason for a second file rather than a second column, and
    there are five graphs now, not two.
    """
    return os.path.join(PIN_DIR, f"reference_provider_{stem(input_file)}.txt")


def rows(answers: Iterable[Answer]) -> tuple[str, ...]:
    """`answers` as pin lines: `<reason> <source> <path>`, then one `off <path>` per
    disagreement. Values are **not** pinned, for `cold_start.rows`' reason -- they move
    in the last digits for reasons that are not regressions.
    """
    answers = tuple(answers)
    lines = [f"{a.reason} {a.source} {a.path_str()}" for a in answers]
    lines += [f"off {a.path_str()}" for a in disagreements(answers)]
    return tuple(lines)


def write_pin(lines: Iterable[str], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            "# What this machine's boundary is answered with, and why. Generated by\n"
            "# `$PY -m functional_process.cottax.provider --write`; do not hand-edit.\n"
            "# `<reason> <source> <path>`; `computed` is a missing producer and may\n"
            "# only go down; `off` is an independently answered path the seed\n"
            "# disagrees with.\n"
            "# See functional_process/cottax/provider.py.\n"
        )
        handle.writelines(f"{line}\n" for line in lines)


def read_pin(path: str) -> tuple[str, ...]:
    with open(path, encoding="utf-8") as handle:
        return tuple(
            line.strip() for line in handle if line.strip() and not line.startswith("#")
        )


def answers_for(input_file: str) -> tuple[Answer, ...]:
    """The provider's answer for one configuration, assembled from the file alone.

    The design is PROCESS's own active `ixc`, read off the seed rather than re-derived:
    `SingleRun.__init__` has already parsed it, and `sand.iteration_variable_path` is the
    same map `boundary.missing_producers` uses.
    """
    from functional_process.cottax.cold_start import _resolve, cold_state  # noqa: PLC0415
    from functional_process.cottax.indat import graph_for, machine_from_indat  # noqa: PLC0415
    from functional_process.cottax.mda import driven_graph  # noqa: PLC0415
    from functional_process.cottax.sand import iteration_variable_path  # noqa: PLC0415

    input_file = _resolve(input_file)
    # Assembly first: a configuration that refuses needs nothing from PROCESS, and a
    # 100 s solve for a row that will be one line of text is the difference between a
    # matrix that gets re-run and one that does not (`run_cold_matrix`'s own rule).
    graph = driven_graph(graph_for(machine_from_indat(input_file)))
    state = cold_state(input_file)
    n = int(state.seed.numerics.n_iteration_variables)
    design = {iteration_variable_path(int(i)) for i in state.seed.numerics.ixc[:n]}
    return provide(
        graph,
        design,
        input_file,
        seed=state.seed,
        computed=state.written,
        final=state.process,
    )


def summary(input_file: str, answers: Iterable[Answer]) -> str:
    answers = tuple(answers)
    have = tally(answers)
    order = (INPUT, DEFAULT, DERIVED, COMPUTED, UNWRITTEN, SOLVER, GUESS, MINTED)
    lines = [
        f"=== {stem(input_file)}: {len(answers)} boundary path(s), "
        f"{have.get('answered', 0)} answered independently, "
        f"{have.get('from_process', 0)} still from PROCESS",
        "  " + "  ".join(f"{k}={have.get(k, 0)}" for k in order),
    ]
    for a in answers:
        if a.reason in BUGS:
            lines.append(f"    MISSING PRODUCER  {a.path_str()}")
    for a in disagreements(answers):
        lines.append(
            f"    off ({a.reason}/{a.source})  {a.path_str()}  "
            f"provider={a.value!r} seed={a.seeded!r}"
        )
    return "\n".join(lines)


def _main(argv: list[str]) -> int:
    import jax  # noqa: PLC0415

    jax.config.update("jax_enable_x64", True)

    chosen = [argv[i + 1] for i, a in enumerate(argv) if a == "--input"]
    for input_file in chosen or CONFIGURATIONS:
        try:
            answers = answers_for(input_file)
        except (NotImplementedError, ValueError) as error:
            print(f"=== {stem(input_file)}: REFUSED -- {error}"[:400])
            continue
        print(summary(input_file, answers))
        if "--write" in argv:
            path = pin_path(input_file)
            write_pin(rows(answers), path)
            print(f"  wrote {os.path.basename(path)}")
    return 0


if __name__ == "__main__":
    import sys

    from functional_process.cottax.provider import _main as main

    raise SystemExit(main(sys.argv[1:]))
