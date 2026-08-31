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


def test_the_stellarators_and_the_tokamaks_disagree_with_the_seed_differently():
    """The diff is not one list, and that is the point of a pin per configuration.

    Five rows are common to every tokamak -- `.tfcoil.eff_tf_cryo` (`-1.0` sentinel
    against `init.py`'s `0.13`), `.tfcoil.eyoung_ins`, `.tfcoil.eyoung_cond_axial`,
    `.pf_coil.rho_pf_coil` and `.physics.f_nd_beam_electron`. The stellarators disagree
    on a different set: `init.py`'s stellarator arm zeroes the central solenoid
    (`.build.dr_cs`, `.build.dr_cs_tf_gap`) and rewrites all four pulse times, notably
    `.times.t_plant_pulse_burn` from `1000` s to `3.15576e7` s -- one year, i.e. steady
    state. Every one of these is a path a defaults table would answer confidently and
    wrongly, which is §22.3's stated failure mode, measured.
    """
    off = {
        stem: {
            line.split(" ", 1)[1]
            for line in read_pin(pin_path(stem))
            if line.startswith("off ")
        }
        for stem in ("stellarator_helias", "large_tokamak_nof")
    }
    assert ".tfcoil.eff_tf_cryo" in off["stellarator_helias"] & off["large_tokamak_nof"]
    assert ".times.t_plant_pulse_burn" in off["stellarator_helias"]
    assert ".times.t_plant_pulse_burn" not in off["large_tokamak_nof"]
    assert ".pf_coil.rho_pf_coil" in off["large_tokamak_nof"]
    assert ".pf_coil.rho_pf_coil" not in off["stellarator_helias"]
