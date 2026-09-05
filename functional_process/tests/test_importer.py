"""`functional_process.importer` -- the legacy `IN.DAT` reader (§24).

Three things are tested, and the middle one is the point.

1. **The vendored table equals PROCESS's** (§23.2: vendor for runtime, assert equality in
   tests). `functional_process/vocabulary/input_variables.py` carries 865 name -> field
   declarations that the runtime must not import `process` to get; drift in either the
   name set or any row fails here instead of silently changing an answer.
2. **The oracle.** Every value the importer parses is diffed against a real `SingleRun`'s
   `DataStructure` taken at the **pre-`init_process`** point -- precisely, immediately
   after `parse_input_file` returns and before `set_active_constraints`. That point and
   no other: `init.py` *destroys three genuine inputs* under its double-null branch and
   `st_init` overwrites nine values a stellarator `IN.DAT` set explicitly
   (`_audit/init_audit.md`), so a comparison after those stages cannot tell a parse bug
   from a derivation. The name set is compared against `parse_input_file`'s own return
   value, which is the only complete statement of what PROCESS read out of the file.
3. **The grammar**, on synthetic files: the array forms, presence, the Fortran exponent,
   and `ixc`/`icc` reaching the problem statement rather than any field.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import math
from pathlib import Path

import numpy as np
import pytest

from functional_process.importer import (
    EVALUATION_RUN_MODE,
    OPTIMISATION_RUN_MODE,
    ArrayInput,
    Problem,
    read_indat,
)
from functional_process.vocabulary.input_variables import INPUT_VARIABLES

ROOT = Path(__file__).resolve().parents[2]

CONFIGURATIONS = (
    "stellarator_helias",
    "helias_5b",
    "large_tokamak_nof",
    "large_tokamak_eval",
    "low_aspect_ratio_DEMO",
    "spherical_tokamak_eval",
    "st_regression",
)
"""`provider.CONFIGURATIONS`' seven, by stem. `IFE.IN.DAT` is out of scope everywhere in
this port and is not read here either."""


def _input_file(stem: str) -> str:
    return str(ROOT / "tests" / "regression" / "input_files" / f"{stem}.IN.DAT")


# ------------------------------------------------------------------ the vendored table


class TestVendoredTable:
    def test_name_set_equals_process(self):
        from process.core.input import INPUT_VARIABLES as REFERENCE

        assert set(INPUT_VARIABLES) == set(REFERENCE)

    def test_every_row_equals_process(self):
        from process.core.input import INPUT_VARIABLES as REFERENCE

        for name, reference in REFERENCE.items():
            ours = INPUT_VARIABLES[name]
            module = reference.module if isinstance(reference.module, str) else None
            assert ours.module == module, name
            assert ours.type is reference.type, name
            assert ours.array == reference.array, name

    def test_only_ixc_and_icc_address_no_field(self):
        """The two `set_variable=False` rows, vendored as `module=None`. If PROCESS grows
        a third, the importer's `values`/`problem` split needs a decision, not a default.
        """
        assert {n for n, d in INPUT_VARIABLES.items() if d.module is None} == {
            "ixc",
            "icc",
        }

    def test_target_name_is_still_unused_in_process(self):
        """`InputVariable.target_name` exists and no row sets it, so the vendored table
        drops it. This fails the day one does, which is the day the importer needs it."""
        from process.core.input import INPUT_VARIABLES as REFERENCE

        assert not [n for n, c in REFERENCE.items() if c.target_name]


# ------------------------------------------------------------------------- the oracle


class _StopAfterParse(Exception):  # noqa: N818
    """Abort the `SingleRun` the instant the parse is done -- see `parsed_state`."""


@functools.lru_cache(maxsize=None)
def parsed_state(input_file: str):
    """`(DataStructure, {name: ...})` as `parse_input_file` left them.

    `_audit/init_audit.md`'s method: wrap a stage of `init_process` inside a *real*
    `SingleRun` and `deepcopy` the data structure around it. Here the wrapper also
    raises, because everything after the parse -- `set_active_constraints`,
    `set_device_type`, `st_init`, `check_process` -- is the derivation layer this module
    deliberately does not implement, and running it would put derived values into the
    oracle.
    """
    import process.core.init as init_module
    from process.main import SingleRun

    from functional_process.cottax.cold_start import _resolve, _scratch_copy

    captured: dict = {}
    real = init_module.parse_input_file

    def spy(data):
        captured["inputs"] = real(data)
        captured["data"] = copy.deepcopy(data)
        raise _StopAfterParse

    init_module.parse_input_file = spy
    try:
        SingleRun(_scratch_copy(_resolve(input_file)), "vmcon")
    except _StopAfterParse:
        pass
    finally:
        init_module.parse_input_file = real
    return captured["data"], captured["inputs"]


NOT_WRITTEN_BY_PARSE = {
    # `SingleRun.set_filenames`, before `init_process` runs at all.
    ("globals", "fileprefix"),
    ("globals", "output_prefix"),
    # `initialise_iteration_variables`, `init_process`'s first line -- the iteration
    # variable labels and their default bounds. A file may then overwrite a bound.
    ("numerics", "lablxc"),
    ("numerics", "boundl"),
    ("numerics", "boundu"),
    # `initialise_imprad` (`main.py`), the *fifth* initialisation source `init_audit.md`
    # §5 named: it reads impurity data files off disk before `init_process`.
    ("impurity_radiation", "f_nd_impurity_electron_array"),
    ("impurity_radiation", "impurity_arr_label"),
    ("impurity_radiation", "impurity_arr_len_tab"),
    ("impurity_radiation", "impurity_arr_z"),
    ("impurity_radiation", "impurity_arr_zav"),
    ("impurity_radiation", "m_impurity_amu_array"),
    ("impurity_radiation", "pden_impurity_lz_nd_temp_array"),
    ("impurity_radiation", "temp_impurity_keV_array"),
}
"""Fields that differ from a bare `DataStructure()` at the pre-init point and that
`parse_input_file` did **not** write. Each is another initialisation source, and each is
the next layer's, not the importer's."""


def _equal(a, b) -> bool:
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    try:
        left, right = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    except (TypeError, ValueError):
        return bool(np.all(np.asarray(a, dtype=object) == np.asarray(b, dtype=object)))
    return left.shape == right.shape and bool(
        np.array_equal(left, right, equal_nan=True)
    )


def _flat(value):
    """PROCESS stores a >1-D input array column-major and `set_array_variable` ravels it
    that way before indexing, so a Fortran index maps onto `.T.ravel()`."""
    array = np.asarray(value)
    return array.T.ravel() if array.ndim > 1 else array


@pytest.mark.parametrize("stem", CONFIGURATIONS)
class TestOracle:
    def test_parses_without_complaint(self, stem):
        imported = read_indat(_input_file(stem))
        assert imported.errors == ()
        assert imported.unknown == ()

    def test_name_set_equals_process_parse(self, stem):
        """Every name PROCESS read, and no other. `parse_input_file`'s return value is
        the complete statement of that, so the two sets are compared directly."""
        _, inputs = parsed_state(_input_file(stem))
        assert read_indat(_input_file(stem)).present == set(inputs)

    def test_values_agree_with_process(self, stem):
        data, _ = parsed_state(_input_file(stem))
        wrong = []
        for (area, name), ours in sorted(read_indat(_input_file(stem)).values.items()):
            theirs = getattr(getattr(data, area), name)
            if not isinstance(ours, ArrayInput):
                if not _equal(ours, theirs):
                    wrong.append(f"{area}.{name}: ours={ours!r} process={theirs!r}")
                continue
            flat = _flat(theirs)
            for index, value in ours.elements:
                if not _equal(value, flat[index]):
                    wrong.append(f"{area}.{name}[{index}]: {value!r} vs {flat[index]!r}")
            if ours.zero_filled:
                # The `a = 1,2,3` form zeroes the whole array first.
                unset = set(range(len(flat))) - {i for i, _ in ours.elements}
                wrong += [
                    f"{area}.{name}[{i}]: zero-filled but process has {flat[i]!r}"
                    for i in sorted(unset)
                    if not _equal(0.0, flat[i])
                ]
        assert wrong == []

    def test_no_parsed_field_is_missed(self, stem):
        """The reverse direction: a field PROCESS's parse wrote that the importer has no
        entry for. Compared against a bare `DataStructure()`, with the other four
        initialisation sources named explicitly rather than tolerated silently."""
        data, _ = parsed_state(_input_file(stem))
        from process.core.model import DataStructure

        ours = set(read_indat(_input_file(stem)).values) | {
            # `ixc`/`icc` set no field of their own; they reach `numerics` through
            # `additional_actions`, and the importer routes them to `problem`.
            ("numerics", "ixc"),
            ("numerics", "icc"),
            ("numerics", "n_iteration_variables"),
            ("numerics", "n_constraints"),
        }
        blank = DataStructure()
        missed = []
        for area in dataclasses.fields(blank):
            left, right = getattr(blank, area.name), getattr(data, area.name)
            if not dataclasses.is_dataclass(left):
                continue
            for entry in dataclasses.fields(left):
                place = (area.name, entry.name)
                if place in ours or place in NOT_WRITTEN_BY_PARSE:
                    continue
                if not _equal(getattr(left, entry.name), getattr(right, entry.name)):
                    missed.append(place)
        assert missed == []

    def test_problem_statement_agrees(self, stem):
        data, _ = parsed_state(_input_file(stem))
        problem = read_indat(_input_file(stem)).problem
        n = int(data.numerics.n_iteration_variables)
        m = int(data.numerics.n_constraints)
        assert list(problem.ixc) == list(np.asarray(data.numerics.ixc)[:n].astype(int))
        assert list(problem.icc) == list(np.asarray(data.numerics.icc)[:m].astype(int))
        assert problem.i_figure_merit in (None, int(data.numerics.i_figure_merit))

    def test_raw_namespace_covers_every_value(self, stem):
        """§24.2 item 2: the same values under a `raw` root, so a sentinel resolution can
        be a node with a read and a distinct write instead of a self-loop."""
        imported = read_indat(_input_file(stem))
        raw = imported.raw_values()
        assert len(raw) == len(imported.values)
        for area, name in imported.values:
            assert f".raw.{area}.{name}" in raw


# ------------------------------------------------------------------------ the grammar


def _write(tmp_path: Path, body: str) -> str:
    path = tmp_path / "TEST.IN.DAT"
    path.write_text(body)
    return str(path)


class TestGrammar:
    def test_indexed_array_is_not_a_last_wins_scalar(self, tmp_path):
        """The defect this module exists to end: a last-wins name scan answered the
        ten-element `.pf_coil.zref` with the `1.0` of `zref(10) = 1.0` (§22.6)."""
        imported = read_indat(_write(tmp_path, "zref(1) = 3.6\nzref(10) = 1.0\n"))
        value = imported.get("pf_coil", "zref")
        assert isinstance(value, ArrayInput)
        assert value.as_dict() == {0: 3.6, 9: 1.0}
        assert not value.zero_filled

    def test_comma_list_is_zero_filled(self, tmp_path):
        """`parse_input_file` does `array[:] = 0.0` before filling a comma list, so an
        element the list does not reach is `0.0` and not its dataclass default."""
        imported = read_indat(_write(tmp_path, "i_pf_location = 2,2,3\n"))
        value = imported.get("pf_coil", "i_pf_location")
        assert value.zero_filled
        assert value.as_dict() == {0: 2, 1: 2, 2: 3}
        assert value.dense(5) == [2, 2, 3, 0.0, 0.0]

    def test_presence_is_recorded_for_a_default_valued_name(self, tmp_path):
        """§24.2 item 1. `dx_tf_side_case_min = 0.0` is indistinguishable from the
        dataclass default *by value*; only the text says the file named it. This is what
        `indat.py:4420-4428` cannot ask today, and why its scan can only return `0`."""
        imported = read_indat(_write(tmp_path, "dx_tf_side_case_min = 0.0\n"))
        assert imported.named("dx_tf_side_case_min")
        assert not imported.named("f_dr_tf_plasma_case")

    def test_fortran_exponent_and_comments(self, tmp_path):
        imported = read_indat(
            _write(tmp_path, "* a comment\nnd_plasma_electrons_vol_avg = 7.5D19 * n_e\n")
        )
        assert imported.get("physics", "nd_plasma_electrons_vol_avg") == 7.5e19

    def test_ixc_icc_are_the_problem_not_values(self, tmp_path):
        imported = read_indat(
            _write(tmp_path, "ixc = 4\nixc = 6\nicc = 1 * Beta\ni_figure_merit = -14\n")
        )
        assert imported.problem.ixc == (4, 6)
        assert imported.problem.icc == (1,)
        assert imported.problem.i_figure_merit == -14
        assert ("numerics", "ixc") not in imported.values
        assert imported.named("ixc")

    def test_unrecognised_name_is_reported_not_raised(self, tmp_path):
        """`parse_input_file` raises here. Refusing to read a file is a validation
        decision and validation is the next layer (§24.2 item 3), so it is collected."""
        imported = read_indat(_write(tmp_path, "not_a_process_input = 1.0\n"))
        assert [a.name for a in imported.unknown] == ["not_a_process_input"]
        assert imported.values == {}
        assert not imported.named("not_a_process_input")

    def test_int_type_is_preserved(self, tmp_path):
        imported = read_indat(_write(tmp_path, "i_tf_sup = 1\n"))
        value = imported.get("tfcoil", "i_tf_sup")
        assert isinstance(value, int) and not isinstance(value, bool)

    def test_no_process_import_at_runtime(self):
        """§23, checked rather than asserted: a subprocess with `process` blocked at
        `sys.meta_path` imports the module and reads a real file. In this env `process`
        is importable and already in `sys.modules`, so an in-process blocker would prove
        nothing -- hence the subprocess."""
        import subprocess
        import sys

        script = (
            "import sys\n"
            "class Block:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] == 'process':\n"
            "            raise ImportError(name)\n"
            "sys.meta_path.insert(0, Block())\n"
            "from functional_process.importer import read_indat\n"
            f"r = read_indat({_input_file('st_regression')!r})\n"
            "assert len(r.values) > 100 and 'process' not in sys.modules\n"
        )
        subprocess.run(  # noqa: S603
            [sys.executable, "-c", script], cwd=ROOT, check=True, capture_output=True
        )


def test_scalar_count_is_a_strict_subset_of_values():
    """Arrays are values too -- `scalars()` is the old shape, offered but not the whole."""
    imported = read_indat(_input_file("st_regression"))
    assert set(imported.scalars()) < set(imported.values)
    assert any(isinstance(v, ArrayInput) for v in imported.values.values())


def test_no_value_is_nan():
    """A parsed number is a number. Guards the `d`->`e` substitution, which turns a
    stray letter into a cast failure rather than a silent `nan`."""
    for stem in CONFIGURATIONS:
        for place, value in read_indat(_input_file(stem)).values.items():
            numbers = (
                [v for _, v in value.elements]
                if isinstance(value, ArrayInput)
                else [value]
            )
            for number in numbers:
                assert isinstance(number, str) or not math.isnan(number), (stem, place)


# --------------------------------------------------------- the run mode (§24.10)


def test_the_run_mode_is_read_off_the_file_and_says_which_problem_it_states():
    """`i_process_run_mode` is the discriminator PROCESS itself uses.

    `main.run_scan` reads it and nothing else: `1` keeps VMCON, `-2` replaces the solver
    with `scipy.optimize.fsolve` over the equalities alone. Reading it here is what let
    the port stop building an `Optimise` for the two `_eval` files (§24.10).
    """
    root = Path(__file__).resolve().parents[2] / "tests/regression/input_files"
    stated = {
        p.name[: -len(".IN.DAT")]: read_indat(p).problem.i_process_run_mode
        for p in sorted(root.glob("*.IN.DAT"))
        if p.name != "IFE.IN.DAT"
    }
    assert stated["large_tokamak_eval"] == EVALUATION_RUN_MODE
    assert stated["spherical_tokamak_eval"] == EVALUATION_RUN_MODE
    assert stated["stellarator_helias"] == OPTIMISATION_RUN_MODE
    assert stated["helias_5b"] == OPTIMISATION_RUN_MODE
    assert stated["low_aspect_ratio_DEMO"] == OPTIMISATION_RUN_MODE
    assert stated["st_regression"] == OPTIMISATION_RUN_MODE
    # `large_tokamak_nof` names no run mode at all, which is the `numerics.py:150`
    # default -- unresolved here, exactly as `n_equality_constraints`' sentinel is.
    assert stated["large_tokamak_nof"] is None


def test_a_file_naming_no_run_mode_optimises():
    """The unresolved `None` must not read as "not an optimisation"."""
    assert not Problem().is_evaluation
    assert not Problem(i_process_run_mode=OPTIMISATION_RUN_MODE).is_evaluation
    assert Problem(i_process_run_mode=EVALUATION_RUN_MODE).is_evaluation


def test_exactly_the_two_eval_files_state_a_root_find():
    """The premise §24.10 rests on, measured per file rather than assumed.

    Three properties, and only the **first** is the discriminator: PROCESS branches on
    `i_process_run_mode` alone (`main.py:449-462`). The other two are checked here
    because both were proposed as the test and both are wrong:

    - *"no `i_figure_merit`"* is **necessary but not sufficient** on these seven -- it
      does single out the same two files, but only because a file in evaluation mode has
      no reason to name one. Nothing stops it naming one anyway.
    - *"square"* is **not even necessary**: `helias_5b` states 3 equalities against 3
      iteration variables and PROCESS runs VMCON on it. Squareness is a consequence of
      the mode, which is why `mdf.assemble` checks it as a consistency test instead.
    """
    root = Path(__file__).resolve().parents[2] / "tests/regression/input_files"
    problems = {
        p.name[: -len(".IN.DAT")]: read_indat(p).problem
        for p in sorted(root.glob("*.IN.DAT"))
        if p.name != "IFE.IN.DAT"
    }
    root_finds = {n for n, q in problems.items() if q.is_evaluation}
    assert root_finds == {"large_tokamak_eval", "spherical_tokamak_eval"}

    # Each root find is square -- one equality per iteration variable -- which is what
    # makes `fsolve` over the equalities a well-posed problem at all.
    for name in root_finds:
        problem = problems[name]
        assert problem.n_equality_constraints == len(problem.ixc), name

    # And no root find names a figure of merit, so the `7` a `ReferenceRun` reports for
    # them comes from `numerics.py:154`'s default and from nowhere else.
    assert all(problems[n].i_figure_merit is None for n in root_finds)

    # The counterexample that rules out "square" as the discriminator.
    helias = problems["helias_5b"]
    assert helias.n_equality_constraints == len(helias.ixc) == 3
    assert not helias.is_evaluation
