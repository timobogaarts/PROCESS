"""The boundary provider, pinned per configuration.

What is tested is the *policy* -- that a boundary path is answered with a reason and not
a bare value, that `computed` outranks `input` so a genuine-looking `IN.DAT` number
PROCESS overwrites cannot be answered confidently and wrongly, and that a path the
problem owns is not answered at all. The lists live in
`functional_process/reference_provider_<stem>.txt` and are generated, never typed.
"""

from pathlib import Path

import pytest
from cottax.spec import VarPath

from functional_process.provider import (
    COMPUTED,
    DEFAULT,
    DEFAULTS,
    DERIVED,
    GUESS,
    INDAT,
    INPUT,
    PROCESS,
    SOLVER,
    UNWRITTEN,
    Answer,
    answer,
    answers_for,
    declared_inputs,
    disagreements,
    install,
    named_in,
    pin_path,
    read_pin,
    rows,
    tally,
)

CONFIGURATIONS = (
    "stellarator_helias",
    "helias_5b",
    "large_tokamak_nof",
    "large_tokamak_eval",
    "low_aspect_ratio_DEMO",
    "spherical_tokamak_eval",
    "st_regression",
)

INPUT_DIR = "tests/regression/input_files"


def input_file(stem: str) -> str:
    return f"{INPUT_DIR}/{stem}.IN.DAT"


def V(*keys) -> VarPath:
    from jax.tree_util import GetAttrKey

    return VarPath(tuple(GetAttrKey(k) for k in keys))


def _context(**over):
    from process.core.model import DataStructure

    base = {
        "design": frozenset(),
        "computed": frozenset(),
        "inputs": {},
        "named": {},
        "seed": DataStructure(),
        "final": None,
        "defaults": DataStructure(),
    }
    base.update(over)
    return base


# ============================================================== the ladder
def test_a_path_the_problem_owns_is_not_supplied():
    """§22.2's one reason the signature is wider than `(graph,)`. `.physics.aspect` is
    owned by its node only when it is not an active iteration variable; the same path is
    a supplied input on one run and the solver's unknown on the next, and the provider
    must not answer it in the second case.
    """
    var = V("physics", "aspect")
    got = answer(var, INPUT, **_context(design=frozenset({var})))
    assert got.reason == SOLVER
    assert not got.independent


def test_a_field_process_computes_outranks_its_being_a_declared_input():
    """The failure mode §22.3 names by path. `.buildings.dz_tf_cryostat` seeds at `2.5`,
    is a genuine `InputVariable`, and `cryostat.py:58-60` overwrites it before its only
    live reader -- so a defaults table would answer `2.5` confidently and be wrong. The
    reason column is what makes that visible, and it only does so if `computed` is asked
    first.
    """
    var = V("buildings", "dz_tf_cryostat")
    place = ("buildings", "dz_tf_cryostat")
    got = answer(
        var,
        INPUT,
        **_context(
            computed=frozenset({place}),
            inputs={place: "dz_tf_cryostat"},
            named={"dz_tf_cryostat": "2.5"},
        ),
    )
    assert got.reason == COMPUTED


def test_a_field_nothing_writes_is_a_reason_and_not_a_crash():
    """`.physics.dlamie`'s shape: not a declared input, not written on this
    configuration, so the bare dataclass default stands and there is no right answer to
    supply. Answered `unwritten`, never raised.
    """
    got = answer(V("physics", "dlamie"), INPUT, **_context())
    assert got.reason == UNWRITTEN


def test_init_process_writing_a_non_input_is_derived():
    from process.core.model import DataStructure

    seed = DataStructure()
    seed.physics.rmajor = 9.0
    got = answer(V("physics", "rmajor"), INPUT, **_context(seed=seed))
    assert got.reason == DERIVED
    assert got.source == PROCESS  # only the seed knows it; not answered independently


def test_a_named_scalar_input_is_answered_from_the_file_and_a_default_from_defaults():
    place = ("physics", "rmajor")
    named = answer(
        V(*place), INPUT, **_context(inputs={place: "rmajor"}, named={"rmajor": "8.5"})
    )
    assert (named.reason, named.source, named.value) == (INPUT, INDAT, 8.5)
    unnamed = answer(V(*place), INPUT, **_context(inputs={place: "rmajor"}))
    assert (unnamed.reason, unnamed.source) == (DEFAULT, DEFAULTS)


# ============================================================== the file scan
def test_an_indexed_assignment_does_not_answer_the_whole_array(tmp_path):
    """Measured before it was fixed: `zref(10) = 1.0` under a last-wins dict answered
    `.pf_coil.zref` -- a ten-element array -- with `1.0`.
    """
    path = tmp_path / "x.IN.DAT"
    path.write_text("zref(1) = 3.6\nzref(10) = 1.0\nrmajor = 8.5 * a comment\n")
    found = named_in(str(path))
    assert found["zref"] is None
    assert found["rmajor"] == "8.5"


def test_declared_inputs_comes_from_process_s_own_registry():
    """Not a table typed here: `INPUT_VARIABLES[name].module` is the dotted attribute
    path `parse_input_file` walks and `target_name or name` is the field it sets, which
    is exactly the `(area, field)` pair a boundary `VarPath` is.
    """
    inputs = declared_inputs()
    assert inputs["buildings", "dz_tf_cryostat"] == "dz_tf_cryostat"
    assert inputs["physics", "rmajor"] == "rmajor"
    assert ("numerics", "ixc") not in inputs  # `set_variable = False`


# ============================================================== the diff
def test_the_diff_reports_an_independently_answered_path_the_seed_contradicts():
    """§22.3's move for the input side: the seed is the oracle, not the source."""
    off = Answer(V("a", "b"), DEFAULT, DEFAULTS, value=1.0, seeded=2.0)
    same = Answer(V("a", "c"), DEFAULT, DEFAULTS, value=1.0, seeded=1.0)
    from_seed = Answer(V("a", "d"), DERIVED, PROCESS, value=1.0, seeded=2.0)
    assert disagreements([off, same, from_seed]) == (off,)


def test_the_tally_counts_independence_apart_from_reason():
    got = tally([
        Answer(V("a", "b"), INPUT, INDAT),
        Answer(V("a", "c"), DERIVED, PROCESS),
    ])
    assert (got["answered"], got["from_process"]) == (1, 1)


# ============================================================== consuming it
class _Area:
    """One `DataStructure` area, as much of one as `install` touches."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


class _Data:
    def __init__(self, **areas):
        self.__dict__.update(areas)


def _data():
    return _Data(a=_Area(b=2.0, c=1.0, d=2.0, e=3), f=_Area())


def test_the_provider_is_installed_over_the_seed_and_the_off_rows_are_held():
    """The consuming half. In the default mode the substitution is inert **by
    construction** -- every value written is the one it replaces -- so a solve that moves
    under it is a defect in `install`, not a finding about `init.py`. The `off` row is
    the one path that would move, and it is counted rather than dropped.
    """
    answers = [
        Answer(V("a", "b"), DEFAULT, DEFAULTS, value=1.0, seeded=2.0),  # off
        Answer(V("a", "c"), INPUT, INDAT, value=1.0, seeded=1.0),
        Answer(V("a", "d"), DERIVED, PROCESS, value=9.0, seeded=2.0),  # the seed's
    ]
    data = _data()
    counts, moved = install(answers, data)
    assert (data.a.b, data.a.c, data.a.d) == (2.0, 1.0, 2.0)
    assert moved == ()
    assert (counts["written"], counts["held"], counts["from_process"]) == (1, 1, 1)


def test_taking_the_provider_at_its_word_writes_the_disagreements_and_names_them():
    """`--provider-strict`: what a solve does when `init.py`'s corrections are absent.
    The paths it moved travel back with the counts, because a row that moves is only a
    finding if the reader can see which values moved it.
    """
    answers = [Answer(V("a", "b"), DEFAULT, DEFAULTS, value=1.0, seeded=2.0)]
    data = _data()
    counts, moved = install(answers, data, disagreeing=True)
    assert data.a.b == pytest.approx(1.0)
    assert moved == ((".a.b", 1.0, 2.0),)
    assert (counts["written"], counts["held"]) == (1, 0)


def test_an_int_seed_keeps_its_type_so_a_switch_cannot_become_a_float():
    """`_scalar` parses the file as a float, and a switch read back as `1.0` compares
    unequal to its `IntEnum` -- a silently rerouted branch, not a number that is off.
    """
    answers = [Answer(V("a", "e"), INPUT, INDAT, value=4.0, seeded=3)]
    data = _data()
    install(answers, data, disagreeing=True)
    assert data.a.e == 4
    assert isinstance(data.a.e, int)


def test_a_path_with_no_field_and_a_value_of_none_are_skipped_not_written():
    """`.vacuum.l1` and four siblings are `None` in both the defaults and the seed, so
    there is nothing to supply; a minted path addresses no field at all. Both are counted
    (`nothing`) rather than dropped, which is what keeps the arithmetic below closed.
    """
    answers = [
        Answer(V("a", "z"), DEFAULT, DEFAULTS, value=None, seeded=None),
        Answer(V("f", "missing"), DEFAULT, DEFAULTS, value=1.0, seeded=None),
    ]
    counts, _moved = install(answers, _data())
    assert (counts["written"], counts["nothing"]) == (0, 1)


def test_the_supplied_denominator_leaves_out_what_no_provider_could_answer():
    """A `guess` is a `Drive`'s `Start` port and a `solver` row is owned by the problem;
    counting either in the denominator understates the ratio. `paths` keeps the raw
    total beside it because §22.6's published numbers are over that one.
    """
    answers = [
        Answer(V("a", "c"), INPUT, INDAT, value=1.0, seeded=1.0),
        Answer(V("a", "d"), SOLVER, PROCESS, value=2.0, seeded=2.0),
        Answer(V("a", "g"), GUESS, PROCESS),
    ]
    counts, _moved = install(answers, _data())
    assert (counts["paths"], counts["supplied"]) == (3, 1)
    assert (
        counts["written"] + counts["held"] + counts["nothing"] + counts["from_process"]
        == counts["supplied"]
    )


# ============================================================== the configurations
@pytest.mark.parametrize("stem", CONFIGURATIONS)
def test_each_configuration_s_answer_is_its_pin(stem):
    """Equality, not containment, for `test_boundary`'s reason: a row that changed
    reason is as much a finding as a row that appeared. Regenerate with
    `$PY -m functional_process.provider --write`.

    Skipped rather than failed when a machine refuses to assemble -- the two spherical
    tokamaks were unblocked while this was being written and could regress; a runner
    that dies on one configuration says nothing about the next
    (`run_cold_matrix`'s rule).
    """
    from functional_process.indat import machine_from_indat

    try:
        machine_from_indat(input_file(stem))
    except (NotImplementedError, ValueError) as error:
        pytest.skip(f"{stem} does not assemble: {str(error)[:120]}")
    assert list(rows(answers_for(input_file(stem)))) == list(read_pin(pin_path(stem)))


def test_every_configuration_has_a_pin():
    for stem in CONFIGURATIONS:
        assert Path(pin_path(stem)).exists(), stem


def test_the_only_missing_producers_left_are_the_two_spherical_tokamak_rows():
    """`computed` is the missing-producer count and **may only go down**.

    Measured on 2026-08-31 over all seven configurations: zero on both stellarators,
    both large tokamaks and `low_aspect_ratio_DEMO` -- which reproduces
    `boundary.missing_producers`' own answer on `large_tokamak_nof`, the one file that
    check covers -- and **one on each spherical tokamak**, `.build.r_cp_top`.

    It was **two** when first measured; `.tfcoil.dx_tf_side_case_min` closed the same
    day. Its cause was not a missing model but a wrong presence inference: `indat.py`
    scanned `switches_from_indat` for `i_f_dr_tf_plasma_case`/`tfc_sidewall_is_fraction`,
    which are not declared PROCESS inputs, so the scan could only ever return `0` while
    `init.py:925-930` sets both `True` whenever the partner field is unset.
    `importer.Imported.named()` is the question it actually needed. See
    `next_steps.md` §23.7.

    Read off the pins rather than recomputed, so this costs nothing; the pins themselves
    are checked against a live run by the test above.
    """
    found = {
        stem: sorted(
            line.split(" ", 2)[2]
            for line in read_pin(pin_path(stem))
            if line.startswith(f"{COMPUTED} ")
        )
        for stem in CONFIGURATIONS
    }
    st = [".build.r_cp_top"]
    assert found == {
        "stellarator_helias": [],
        "helias_5b": [],
        "large_tokamak_nof": [],
        "large_tokamak_eval": [],
        "low_aspect_ratio_DEMO": [],
        "spherical_tokamak_eval": st,
        "st_regression": st,
    }


@pytest.mark.parametrize("stem", CONFIGURATIONS)
def test_no_configuration_disagrees_with_the_seed_any_more(stem):
    """The `off` rows are **gone**, and the mechanism is §24.1's, not a loosened check.

    They used to be 5-8 per configuration, and the two devices' lists were disjoint:
    every tokamak carried `.tfcoil.eff_tf_cryo` (a `-1.0` *sentinel* against `init.py`'s
    `0.13`), `.tfcoil.eyoung_ins`, `.tfcoil.eyoung_cond_axial`, `.pf_coil.rho_pf_coil`
    and `.physics.f_nd_beam_electron`; the stellarators carried `init.py`'s zeroed
    central solenoid (`.build.dr_cs`, `.build.dr_cs_tf_gap`) and all four rewritten pulse
    times, `.times.t_plant_pulse_burn` from `1000` s to `3.15576e7` s -- one year, i.e.
    steady state. Each was a path a defaults table would answer confidently and wrongly,
    which is §22.3's stated failure mode, measured.

    **A derivation ported as a node removes a boundary input** (§24.1), so those paths
    are not answered correctly now -- they are not *boundary* now, and the pins show them
    dropping out rather than changing reason. This test therefore asserts the end state
    the old one was measuring the distance to; the list above is kept because a pin says
    which paths exist and not which ones used to.
    """
    off = [
        line.split(" ", 1)[1]
        for line in read_pin(pin_path(stem))
        if line.startswith("off ")
    ]
    assert off == [], f"{stem} answers {len(off)} path(s) the seed contradicts: {off}"


# =============================================== the native state (`functional_process/native.py`)
#
# The provider installs its answers into a copy of PROCESS's `DataStructure`, so even the
# 89-92 % it answers independently reaches `mdf.seed` through a PROCESS object (§22.7).
# `native.py` is the replacement for that object: it answers `.<area>.<field>` out of
# `importer.read_indat` and a vendored table of PROCESS's dataclass defaults, and it has
# no `DataStructure` in it at all. The tests below are the two things that can silently
# go wrong -- a stale vendored default, and a miss answered rather than reported.


def test_the_vendored_defaults_are_process_s_own():
    """§23.2's standing rule: vendor for runtime, assert equality in tests.

    Every row of `DATACLASS_DEFAULTS` is read straight off a bare `DataStructure()`, so
    the table is a cache of PROCESS's dataclass defaults and nothing else. A PROCESS
    release that moves one of them must fail here rather than quietly change what every
    native solve starts from -- which is exactly the failure `_audit/next_steps.md`
    §22.3 says a defaults table invites.
    """
    from process.core.model import DataStructure

    from functional_process.native import DATACLASS_DEFAULTS, _expand

    data = DataStructure()
    wrong = []
    for (area, name), value in DATACLASS_DEFAULTS.items():
        theirs = getattr(getattr(data, area), name)
        ours = _expand(value)
        if isinstance(theirs, (list, tuple)) or hasattr(theirs, "shape"):
            import numpy as np

            if not np.array_equal(np.asarray(theirs, dtype=float), np.asarray(ours)):
                wrong.append(f".{area}.{name}")
        elif theirs != ours:
            wrong.append(f".{area}.{name}")
    assert not wrong, f"vendored default(s) no longer PROCESS's: {wrong}"


def test_the_defaults_table_holds_no_place_process_does_not_have():
    """A typo in a generated table is a field that silently never answers anything."""
    from process.core.model import DataStructure

    from functional_process.native import DATACLASS_DEFAULTS

    data = DataStructure()
    assert not [
        f".{area}.{name}"
        for area, name in DATACLASS_DEFAULTS
        if not hasattr(getattr(data, area, None), name)
    ]


def test_the_file_s_own_values_win_over_the_defaults_and_say_so():
    """Two sources, and which one answered is reported -- §22.6's `source` column with
    the `process` row gone, because there is no seed here to take one from.
    """
    from functional_process.native import native_state

    state = native_state("tests/regression/input_files/large_tokamak_nof.IN.DAT")
    assert state.sources["physics", "rmajor"] == "indat"
    assert state.sources["costs", "UCAD"] == "defaults"
    assert state.physics.rmajor == 8.0


def test_an_indexed_array_assignment_fills_its_default_rather_than_replacing_it():
    """`zref(4) = 2.8` moves one element; the other nine keep their dataclass default.

    This is the half the provider could not do at all -- it answered a scalar from file
    text and took every array from the seed (§22.6 "what was not done" (a)) -- and the
    two `IN.DAT` array spellings differ, so getting it wrong is a wrong *array*, not a
    missing one. `.pf_coil.zref`'s default begins `3.6, 1.2, 2.5, 1.0`, and
    `large_tokamak_nof` sets element 4 only.
    """
    import numpy as np

    from functional_process.native import DATACLASS_DEFAULTS, native_state

    state = native_state("tests/regression/input_files/large_tokamak_nof.IN.DAT")
    assert np.array_equal(
        state.pf_coil.zref, [3.6, 1.2, 1.0, 2.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )
    assert DATACLASS_DEFAULTS["pf_coil", "zref"][:2] == [3.6, 1.2]


def test_a_place_the_state_cannot_answer_is_recorded_and_not_invented():
    """The instrument. `mdf.seed`/`mda_env` turn the `AttributeError` into `0.0`, so
    without the record a hole in the table is indistinguishable from a real zero -- which
    is §22.1's whole body count, met from the other side.
    """
    from functional_process.native import native_state

    state = native_state("tests/regression/input_files/helias_5b.IN.DAT")
    with pytest.raises(AttributeError):
        state.physics.no_such_field
    assert state.missing == [("physics", "no_such_field")]


def test_the_native_state_is_built_without_importing_process():
    """§23's rule: independence is *checked*, in a subprocess with `process` blocked at
    `sys.meta_path`, not asserted by reading the source. Same shape as
    `test_process_free_import.py`'s and `test_sand.py`'s, deliberately copied.
    """
    import subprocess
    import sys

    script = (
        "import sys\n"
        "class Block:\n"
        "    def find_module(self, name, path=None):\n"
        "        if name == 'process' or name.startswith('process.'):\n"
        "            raise ImportError('process is blocked')\n"
        "        return None\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        return self.find_module(name, path)\n"
        "sys.meta_path.insert(0, Block())\n"
        "from functional_process.native import native_reference\n"
        "r = native_reference('tests/regression/input_files/large_tokamak_nof.IN.DAT')\n"
        "assert r.cold.physics.rmajor == 8.0\n"
        "assert len(r.ixc) == 20 and len(r.icc) == 26\n"
        "print('ok')\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert "ok" in done.stdout


# ------------------------------------------------------- the derivations (`DERIVATIONS`)
#
# The third source. Two of the three are vendored constants, and §23.2's standing rule
# says a vendored constant needs an equality test against PROCESS or it is a value that
# drifts silently -- which is the failure mode unit #8 was written about. The other is a
# rule, and its test is that it reproduces the rule's own branch.


def test_the_vendored_impurity_tables_are_initialise_imprad_s_own():
    """§23.2 again, for `initialise_imprad`'s three `(14, 200)` atomic-data tables.

    `functional_process/data/impurity_tables.npz` stands in for reading 28 `.dat` files
    out of `process/data/lz_non_corona_14_elements/` at startup. It is exactly as good as
    this assertion: a PROCESS release that reships one of those data files, or changes
    `read_impurity_file`'s parse, must fail here rather than quietly change the radiated
    power every native solve starts from. Element-for-element, no tolerance -- the port
    does not interpolate the vendored numbers, it holds them.
    """
    import numpy as np
    from process.core.model import DataStructure
    from process.models.physics.impurity_radiation import initialise_imprad

    from functional_process.models.physics.impurity_radiation import (
        impurity_tables,
    )

    data = DataStructure()
    initialise_imprad(data)
    for name, ours in impurity_tables().items():
        theirs = np.asarray(getattr(data.impurity_radiation, name), dtype=float)
        assert ours.shape == theirs.shape, name
        assert np.array_equal(ours, theirs), name


def test_the_vendored_impurity_masses_are_process_s_constants():
    """The fourth of `initialise_imprad`'s outputs, vendored as fourteen literals.

    Read back off a live `initialise_imprad` rather than off `process.core.constants`, so
    the test pins the *order* of the fourteen `init_imp_element` calls too, not just the
    fourteen numbers.
    """
    import numpy as np
    from process.core.model import DataStructure
    from process.models.physics.impurity_radiation import initialise_imprad

    from functional_process.models.physics.impurity_radiation import (
        IMPURITY_LABELS,
        M_IMPURITY_AMU_ARRAY,
    )

    data = DataStructure()
    initialise_imprad(data)
    assert np.array_equal(
        np.asarray(M_IMPURITY_AMU_ARRAY, dtype=float),
        np.asarray(data.impurity_radiation.m_impurity_amu_array, dtype=float),
    )
    assert list(IMPURITY_LABELS) == [
        str(label) for label in data.impurity_radiation.impurity_arr_label
    ]


def test_the_impurity_temperature_axis_is_one_row_repeated():
    """Why the `.npz` stores 200 temperatures and not 2800.

    `init_imp_element` reads a `Te[eV]` row per species; all fourteen files carry the
    same 200 temperatures, so PROCESS's `(14, 200)` array is one row repeated. Measured
    here rather than assumed, because if it ever stopped being true the vendored file
    would be silently wrong for thirteen species.
    """
    import numpy as np

    from functional_process.models.physics.impurity_radiation import impurity_tables

    temperature = impurity_tables()["temp_impurity_keV_array"]
    assert all(np.array_equal(temperature[0], temperature[i]) for i in range(14))


def test_the_alias_loop_moves_the_declared_impurity_fractions_into_the_array():
    """`init.py:381-384`. The declared input is `f_nd_impurity_electrons`; the array
    every model and twelve `ITERATION_VARIABLES` entries address is
    `f_nd_impurity_electron_array`, which nothing declares.

    `stellarator_helias` states element 14 (`= 1e-05`), so the copy is visible at an
    index no default and no other rule touches -- and `init_imp_element`'s own
    `f_nd_impurity_electron_array[0] = 1.0` survives only because the *declared* field's
    first element is `1.0` too, not because it was left alone.
    """
    import numpy as np

    from functional_process.native import native_state

    state = native_state("tests/regression/input_files/stellarator_helias.IN.DAT")
    assert np.array_equal(
        np.asarray(state.impurity_radiation.f_nd_impurity_electron_array),
        np.asarray(state.impurity_radiation.f_nd_impurity_electrons),
    )
    assert state.impurity_radiation.f_nd_impurity_electron_array[13] == pytest.approx(
        1e-05
    )
    assert state.sources["impurity_radiation", "f_nd_impurity_electron_array"] == (
        "derived"
    )


def test_n_divertors_is_derived_from_i_single_null_and_not_left_at_its_default():
    """`init.py:606-617`. **A structural count, not a rounding difference** -- divertor
    area, mass, heat load and cost all scale on it, and eight slots in this port are
    keyed on it.

    Its dataclass default is `2`, so the failure this pins is silent by construction: a
    single-null machine given the double-null arm still runs, it just runs a different
    machine. `stellarator_helias` is single-null (`i_single_null` defaults to `1`) and
    `large_tokamak_nof` states `i_single_null = 0`.
    """
    from functional_process.native import DATACLASS_DEFAULTS, native_state

    assert DATACLASS_DEFAULTS["divertor", "n_divertors"] == 2  # never what a run sees
    single = native_state("tests/regression/input_files/stellarator_helias.IN.DAT")
    assert single.physics.i_single_null == 1
    assert single.divertor.n_divertors == 1
    assert single.sources["divertor", "n_divertors"] == "derived"


def test_a_double_null_machine_forces_its_upper_build_to_match_its_lower():
    """The other half of `init.py:606-617`, and `init_audit.md` §5b's one row that
    **overrides an input the file states**: on a double-null machine `dz_shld_upper` is
    forced to `dz_shld_lower` whatever the `IN.DAT` says. Splitting one `if` across two
    sessions is how a branch gets half-ported, so both halves land together.

    `i_single_null = 0` *is* the double-null value (`DivertorNumberModels`), so the two
    spherical tokamaks are the double-null machines among the seven and the three large
    tokamaks are not -- the opposite of what the switch's name suggests.
    """
    from functional_process.native import native_state

    state = native_state("tests/regression/input_files/spherical_tokamak_eval.IN.DAT")
    assert state.physics.i_single_null == 0
    assert state.divertor.n_divertors == 2
    assert state.build.dz_shld_upper == state.build.dz_shld_lower
    assert state.build.dz_fw_plasma_gap == state.build.dz_xpoint_divertor
    assert state.build.dz_vv_upper == state.build.dz_vv_lower
    assert state.sources["build", "dz_shld_upper"] == "derived"

    single = native_state("tests/regression/input_files/large_tokamak_nof.IN.DAT")
    assert single.physics.i_single_null == 1
    assert single.divertor.n_divertors == 1
    assert single.build.dz_shld_upper != single.build.dz_shld_lower  # branch not taken


def test_tmargmin_wins_over_both_temperature_margin_fields_when_stated():
    """`init.py:1171-1190`, and **the one derivation the boundary diff could not see**.

    `.tfcoil.temp_tf_superconductor_margin_min` is what constraint 36 compares against.
    Its dataclass default is `0.0` and `large_tokamak_eval` states only the deprecated
    `tmargmin = 1.5`, so before this rule the native env compared a computed margin to
    `0.0` -- while a diff built on `provider.answers_for`'s boundary reported *zero*
    disagreements on all seven, because that path is not in the boundary the provider
    enumerates. A boundary-derived agreement count is a lower bound on the disagreement.
    """
    from functional_process.native import DATACLASS_DEFAULTS, native_state

    assert DATACLASS_DEFAULTS["tfcoil", "temp_tf_superconductor_margin_min"] == 0.0
    state = native_state("tests/regression/input_files/large_tokamak_eval.IN.DAT")
    assert state.tfcoil.tmargmin == pytest.approx(1.5)
    assert state.tfcoil.temp_tf_superconductor_margin_min == pytest.approx(1.5)
    assert state.tfcoil.temp_cs_superconductor_margin_min == pytest.approx(1.5)
    assert state.sources["tfcoil", "temp_tf_superconductor_margin_min"] == "derived"


def test_the_pedestal_raises_one_lower_bound_and_only_where_it_fires():
    """`init.py:444-459` -- **a bound, not a value** (§24.2 item 3), so it lives in
    `native_bounds` and not in `DERIVATIONS`.

    Measured to fire on 1 of 7 configurations. `large_tokamak_nof` has `4` in `ixc` and
    `temp_plasma_pedestal_kev = 5.5`, so its lower bound becomes `5.5055` where its
    `IN.DAT`'s own `boundl(4)` is `5.0`.

    Three of the four guards are exercised by the other configurations pinned below and
    the fourth is not, which is worth stating rather than leaving to be discovered:
    `spherical_tokamak_eval` has `4` in `ixc` and `i_plasma_pedestal = 1` and is stopped
    only by the `boundl[3] < teped * 1.001` comparison itself (`4.5045 < 5.0` is false);
    both stellarators are stopped by `i_plasma_pedestal`, but they *state* `0`, so the
    `st_init` override `_pedestal_temperature_bound` applies on top is **not** covered by
    any of the seven files -- it is there because `init_process` calls `st_init` at `:76`
    and reaches this branch at `:397`, not because a measurement forced it.
    """
    from functional_process.native import native_reference, native_state

    def bound(stem):
        reference = native_reference(f"tests/regression/input_files/{stem}.IN.DAT")
        return next(
            (low, high)
            for (_, low, high), i in zip(reference.bounds, reference.ixc, strict=True)
            if i == 4
        )

    assert bound("large_tokamak_nof") == pytest.approx((5.5055, 100.0))
    # `4` is in all four of these files' `ixc`, and none of them moves.
    assert bound("stellarator_helias") == pytest.approx((3.0, 15.0))
    assert bound("helias_5b") == pytest.approx((4.0, 25.0))
    assert bound("spherical_tokamak_eval") == pytest.approx((5.0, 25.0))
    stopped = native_state("tests/regression/input_files/spherical_tokamak_eval.IN.DAT")
    assert stopped.physics.i_plasma_pedestal == 1  # only the comparison stops this one
    assert stopped.physics.temp_plasma_pedestal_kev * 1.001 < 5.0


def test_the_problem_statement_is_sorted_the_way_single_run_sorts_it():
    """**An eighth initialisation source.** `SingleRun.init` sorts `ixc` at
    `process/main.py:434-438`, *after* `init_process` returns -- so it is outside every
    stage `_audit/init_audit.md` wrapped and outside its §5 list. Three of the seven
    tracked files state `ixc` out of order, and the order is the design vector's, i.e.
    VMCON's column order. §23.7's "byte-identical on all eight" missed it because
    `iteration_variables_from_indat` returns a `frozenset`.
    """
    from functional_process.importer import read_indat
    from functional_process.native import native_reference

    stated = read_indat(
        "tests/regression/input_files/stellarator_helias.IN.DAT"
    ).problem.ixc
    assert list(stated) == [2, 3, 4, 6, 10, 109, 59, 56]  # the file's own order
    assert native_reference(
        "tests/regression/input_files/stellarator_helias.IN.DAT"
    ).ixc == [2, 3, 4, 6, 10, 56, 59, 109]
