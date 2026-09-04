"""Tests for `functional_process.session` -- the build-once repeated-solve entry point.

These pin the **wiring**, not the physics: that a second solve reuses the first solve's
assembly, that the two arms are assembled lazily and independently, and that a root-find
file refuses SAND rather than inventing an optimisation for it. The numbers a session
produces are `run_cold_matrix.solve_mdf`/`solve_sand`'s own -- literally the same
functions a matrix row calls -- so `test_cold_matrix.py` and the matrix itself are where
values are checked, and duplicating a 25-second solve here would buy nothing.
"""

import pytest

from functional_process import run_cold_matrix, session


class _Reference:
    """The little of a `ReferenceRun`/`NativeReference` these tests reach."""

    ixc = (1, 2)
    icc = (1,)
    n_equality = 1
    i_figure_merit = 6
    bounds = ()
    cold = "COLD"


def _session(monkeypatch, root_find=False):
    """A `Session` whose two builders and two solvers are counted, not run."""
    built = {"mdf": 0, "sand": 0}
    solved = []
    monkeypatch.setattr(
        session,
        "build_mdf",
        lambda *a, **k: built.__setitem__("mdf", built["mdf"] + 1) or "MDF_BUILD",
    )
    monkeypatch.setattr(
        session,
        "build_sand",
        lambda *a, **k: built.__setitem__("sand", built["sand"] + 1) or "SAND_BUILD",
    )
    monkeypatch.setattr(
        session,
        "solve_mdf",
        lambda build, reference, cold: solved.append(("mdf", build, cold)) or {},
    )
    monkeypatch.setattr(
        session,
        "solve_sand",
        lambda build, reference, graph, cold: solved.append(("sand", build, cold)) or {},
    )
    live = session.Session(
        name="fake",
        path="fake.IN.DAT",
        reference=_Reference(),
        cold="COLD",
        root_find=root_find,
    )
    return live, built, solved


def test_a_second_solve_reuses_the_first_solve_s_assembly(monkeypatch):
    """The whole point of the module: a second solve does not re-assemble.

    It used to be that every memo below the assembly -- `host_cache._BOUND`,
    `sand_harness._SCHEDULE_WHOLE`, jax's executable cache -- was keyed on the *object*
    that was built, so a rebuild was a re-trace and a re-compile
    (`_audit/optimise_design.md` §32.2). `_BOUND` is gone and jax's cache now hits on a
    re-assembled block (§37), so the *penalty* is smaller than that section measured.
    What this test asserts is unchanged and is the thing the module promises: the second
    solve reuses the first solve's assembly rather than building another one.
    """
    live, built, solved = _session(monkeypatch)
    live.mdf()
    live.mdf()
    live.mdf()
    assert built["mdf"] == 1
    assert [arm for arm, _build, _cold in solved] == ["mdf", "mdf", "mdf"]
    assert {build for _arm, build, _cold in solved} == {"MDF_BUILD"}


def test_each_arm_is_assembled_only_when_it_is_first_asked_for(monkeypatch):
    """A caller who only wants MDF should not pay SAND's assembly, which on
    `large_tokamak_nof` is ~90 s and ~0.8 GiB of resident executables.
    """
    live, built, _solved = _session(monkeypatch)
    live.mdf()
    assert built == {"mdf": 1, "sand": 0}
    live.sand()
    assert built == {"mdf": 1, "sand": 1}


def test_a_different_cold_state_is_a_re_seed_and_not_a_rebuild(monkeypatch):
    """A scan point. Measured warm, a full re-seed plus re-prime from a different start
    is 25-42 ms with zero compiles (§32.2), so this is the supported way to walk one.
    """
    live, built, solved = _session(monkeypatch)
    live.mdf()
    live.mdf(cold="OTHER")
    assert built["mdf"] == 1
    assert [cold for _arm, _build, cold in solved] == ["COLD", "OTHER"]


def test_a_root_find_file_has_no_sand_arm_and_the_refusal_says_why(monkeypatch):
    """`run_cold_matrix._solve_both` prints one row and not two for such a file; a
    session has to refuse for the same reason rather than solve a larger square system
    PROCESS never writes down.
    """
    live, built, _solved = _session(monkeypatch, root_find=True)
    with pytest.raises(ValueError, match="no SAND arm"):
        live.sand()
    assert built["sand"] == 0


def test_a_session_solves_through_the_matrix_s_own_functions():
    """Not a re-implementation. `session` imports `run_cold_matrix`'s halves, so a
    session's answer and a matrix row's are the same computation by construction --
    `run_cold_matrix.py`'s own rule for why it is not a third harness.
    """
    assert session.build_mdf is run_cold_matrix.build_mdf
    assert session.solve_mdf is run_cold_matrix.solve_mdf
    assert session.build_sand is run_cold_matrix.build_sand
    assert session.solve_sand is run_cold_matrix.solve_sand


def test_the_matrix_row_is_still_the_two_halves_run_back_to_back(monkeypatch):
    """`cold_mdf` must stay exactly `build_mdf` then `solve_mdf`: if the row grew a step
    the session skips, the two would drift and only a full matrix would notice.
    """
    calls = []
    monkeypatch.setattr(
        run_cold_matrix,
        "build_mdf",
        lambda *a, **k: calls.append("build") or "BUILD",
    )
    monkeypatch.setattr(
        run_cold_matrix,
        "solve_mdf",
        lambda build, reference, cold: calls.append(("solve", build)) or {},
    )
    run_cold_matrix.cold_mdf(_Reference(), None, None, "COLD")
    assert calls == ["build", ("solve", "BUILD")]


def test_the_series_reports_one_row_per_solve(monkeypatch):
    """`series` is the instrument the regime is measured with, so it must not silently
    drop or merge a repeat.
    """
    live, _built, _solved = _session(monkeypatch)
    monkeypatch.setattr(
        session,
        "solve_mdf",
        lambda build, reference, cold: {
            "iterations": 3,
            "objf": 1.0,
            "max_eq": 0.0,
            "min_ie": 0.0,
            "_x": (1.0,),
        },
    )
    measured = session.series(live, 4, "mdf")
    assert len(measured) == 4
    assert all(row["compiles"] == 0 for row in measured)
    assert "ANSWER STABLE" in session.render(live, measured, "mdf")


def test_a_moved_answer_is_reported_as_a_defect(monkeypatch):
    """Every change this regime was made fast by is a caching change, so an answer that
    moves between repeats of one configuration is a defect and the table must say so
    rather than print two numbers side by side.
    """
    live, _built, _solved = _session(monkeypatch)
    answers = iter([(1.0,), (2.0,)])
    monkeypatch.setattr(
        session,
        "solve_mdf",
        lambda build, reference, cold: {
            "iterations": 3,
            "objf": 1.0,
            "max_eq": 0.0,
            "min_ie": 0.0,
            "_x": next(answers),
        },
    )
    rendered = session.render(live, session.series(live, 2, "mdf"), "mdf")
    assert "ANSWER MOVED at solve(s) [1]" in rendered
    assert "MOVED" in rendered
