"""The cold matrix runner's reporting, tested without running anything.

`run_cold_matrix.py`'s expensive half -- PROCESS, the MDA primes, the two solves -- is
imported from the two ladder harnesses and is tested where those live. What is *new*
here is the reporting: which of four outcomes a trace represents, what a cell shows when
a measurement was not taken, and that a configuration which refuses still produces a row.
Those are exactly the parts that decide whether a fifteen-minute run is legible
afterwards, and they cost nothing to check, so they are checked.

The distinction the tests below are really defending is `no-step` against `converged`.
A cold SAND solve on `large_tokamak_eval` records **zero** SQP iterations because
`pyvmcon`'s first QP has no feasible point (`_audit/next_steps.md` §21.3: c72 is violated
with an identically zero gradient row and no producer changes that), and a table that
rendered that the same way as a solve which converged immediately would be actively
misleading about the one row the records most want read carefully.
"""

from pathlib import Path

import jax
import pytest

jax.config.update("jax_enable_x64", True)

from functional_process.mda_harness import EXPLAINED_DISAGREEMENTS  # noqa: E402
from functional_process.run_cold_matrix import (  # noqa: E402
    CONFIGURATIONS,
    EXPLAINED_OBJECTIVE_READS,
    NATIVE,
    PORT_FILES,
    PROVIDER,
    PROVIDER_STRICT,
    SEED_ONLY,
    Row,
    _against_process,
    _blank,
    _boundary_seed,
    _cell,
    _compare,
    _headline,
    _mode,
    _process_objective,
    _status,
    _trace_tail,
    checkpoint,
    compares_by_default,
    provenance,
    render,
)


def _trace(*entries):
    """A callback trace: `(i, convergence, objf, max|eq|, min ie)` per SQP iteration."""
    return [
        (i, convergence, objf, 0.0, 0.0) for i, (convergence, objf) in enumerate(entries)
    ]


def test_the_configuration_list_is_every_reference_file_but_ife():
    """The list is the point of the runner, so it is pinned rather than trusted.

    `IFE.IN.DAT` is excluded on purpose (a whole unported device, `_audit/next_steps.md`
    §20.4); everything else in `tests/regression/input_files` is a row, including the two
    that are expected to refuse today.
    """
    root = Path(__file__).resolve().parents[2] / "tests/regression/input_files"
    on_disk = {p.name for p in root.glob("*.IN.DAT")}
    listed = {Path(c).name for c in CONFIGURATIONS}
    assert on_disk - listed == {"IFE.IN.DAT"}
    assert listed <= on_disk


def test_a_solve_that_converged_and_one_that_took_no_step_are_not_the_same_word():
    """The four outcomes, and the two that both show a small iteration count."""
    assert _status(_trace((1e-2, 1.0), (1e-12, 1.0)), 1e-8, 500) == "converged"
    assert _status([], 1e-8, 500) == "no-step"
    assert _status(_trace(*[(1e-2, 1.0)] * 500), 1e-8, 500) == "cap(500)"
    assert _status(_trace((1e-2, 1.0), (1e-3, 1.0)), 1e-8, 500) == "stopped"


def test_a_none_tolerance_is_read_against_the_driver_s_own_default():
    """`SAND_TOLERANCE` is deliberately `None` -- `VmconDriver`'s default -- so the
    status has to have a number to compare against or every SAND row reads `stopped`.
    """
    assert _status(_trace((1e-7, 1.0)), None, 500) == "converged"
    assert _status(_trace((1e-7, 1.0)), 1e-8, 500) == "stopped"


def test_an_empty_trace_reports_no_measurements_rather_than_zeros():
    """Zero iterations means the solve never evaluated an answer worth reporting; the
    objective and the feasibility columns must be *absent*, not `0.0`, or the row reads
    as a feasible design at `objf = 0`.
    """
    iterations, objf, max_eq, min_ie = _trace_tail([])
    assert iterations == 0
    assert (objf, max_eq, min_ie) == (None, None, None)


def test_the_tail_of_a_trace_is_the_last_iteration():
    iterations, objf, max_eq, min_ie = _trace_tail([
        (0, 1e-2, 5.0, 1e-3, -1.0),
        (1, 1e-9, 7.0, 1e-8, 2.0),
    ])
    assert (iterations, objf, max_eq, min_ie) == (2, 7.0, 1e-8, 2.0)


def test_a_cell_with_no_measurement_is_a_dash():
    assert _cell(None, 6).strip() == "-"
    assert _cell(1.2345678901, 15, "g").strip() == "1.23456789"
    assert _cell(1.5e-9, 10, "e").strip() == "1.50e-09"


def test_a_refusal_is_shortened_but_never_emptied():
    """`indat`'s refusals are paragraphs. The table keeps the head of one and says where
    the rest is; what it must not do is drop the reason.
    """
    short = _headline(ValueError("tf_stress_arm == (0, 1, 0) is not ported"))
    assert short == "tf_stress_arm == (0, 1, 0) is not ported"
    long = _headline(ValueError("x" * 4000))
    assert len(long) < 400
    assert long.startswith("x" * 100)
    assert "machine_from_indat" in long


def test_a_refused_configuration_still_renders_a_row_and_keeps_its_reason():
    """The whole point of catching the refusal is that the table still has the row."""
    table = render([Row(name="spherical_tokamak_eval", assembles=False, note="because")])
    assert "spherical_tokamak_eval" in table
    assert "REFUSED" in table
    assert "ASSEMBLY REFUSED -- because" in table


def test_both_formulations_render_even_when_one_of_them_failed_to_build():
    """A formulation that raised must not take the other's measurements off the table:
    the two failures are independent evidence, which is why `run_one` catches each.
    """
    mdf = _blank()
    mdf.update(built=False, status="FAILED", note="ValueError: nope")
    sand = _blank()
    sand.update(
        built=True,
        design=14,
        conditions=21,
        iterations=83,
        status="converged",
        objf=1.2177573,
        max_eq=2.8e-11,
        min_ie=5.5e-12,
    )
    table = render([
        Row(
            name="stellarator_helias",
            assembles=True,
            graph_nodes=150,
            process_iterations=46,
            mdf=mdf,
            sand=sand,
        )
    ])
    assert "FAILED" in table
    assert "ValueError: nope" in table
    assert "1.2177573" in table
    assert "83" in table


def test_the_checkpoint_is_written_after_every_row(tmp_path):
    """The reason `main` writes per row rather than at the end: a partial table on disk
    beats a complete one that an interrupted run never printed.
    """
    out = tmp_path / "matrix.txt"
    checkpoint([Row(name="helias_5b", assembles=False, note="why")], str(out))
    assert "helias_5b" in out.read_text()
    checkpoint([], str(out))
    assert "COLD MATRIX" in out.read_text()


def test_a_checkpoint_that_cannot_be_written_is_reported_not_raised(tmp_path, capsys):
    """Losing the file must never lose the row that was just computed."""
    checkpoint([Row(name="x", assembles=False)], str(tmp_path / "no" / "such" / "d.txt"))
    assert "could not checkpoint" in capsys.readouterr().out


@pytest.mark.parametrize("form", ["MDF", "SAND"])
def test_every_configuration_gets_one_line_per_formulation(form):
    table = render([
        Row(name="large_tokamak_nof", assembles=True, graph_nodes=238, mdf={}, sand={})
    ])
    rows = [
        line
        for line in table.splitlines()
        if line.lstrip().startswith("large_tokamak_nof")
    ]
    assert len(rows) == 2
    assert sum(form in line.split() for line in rows) == 1


# ========================================================== where the values come from
def test_the_default_mode_is_the_provider_and_a_flag_overrides_it():
    """`--seed` is the pre-2026-08-31 path exactly, kept so the two can be diffed; it
    wins over the others so that "just give me the old numbers" cannot be half-applied.
    """
    assert _mode([]) == PROVIDER
    assert _mode(["--provider-strict"]) == PROVIDER_STRICT
    assert _mode(["--seed"]) == SEED_ONLY
    assert _mode(["--seed", "--provider-strict"]) == SEED_ONLY


def test_the_seed_mode_hands_back_the_run_s_own_cold_structure_untouched():
    """Not a copy and not a provider answer: `--seed` must be the same object the old
    code path passed, or the mode that exists to reproduce previous rows would be a
    fourth thing nobody has measured.
    """

    class _Reference:
        cold = object()

    reference = _Reference()
    cold, counts, moved = _boundary_seed(reference, "unread.IN.DAT", SEED_ONLY)
    assert cold is reference.cold
    assert (counts, moved) == ({}, ())


def test_the_boundary_block_is_absent_until_something_measured_one():
    """A `--seed` run, or one whose provider failed, must not print an empty block: a
    heading with no rows under it reads as a measurement that came back zero.
    """
    assert "BOUNDARY VALUES" not in render([Row(name="x", assembles=True)])


def test_the_boundary_block_s_columns_close():
    """`provider + seed + held + none == supplied` is the arithmetic that says no path
    went missing between the classification and the solve, so the table prints all five.
    """
    row = Row(
        name="x",
        assembles=True,
        boundary={
            "mode": PROVIDER,
            "paths": 303,
            "supplied": 289,
            "written": 258,
            "held": 8,
            "nothing": 5,
            "from_process": 18,
        },
    )
    block = render([row])
    assert "BOUNDARY VALUES -- mode provider" in block
    assert "258" in block
    assert "289" in block
    assert "303" in block


# ======================================================================== `--native`
def test_native_is_a_fourth_mode_and_the_seed_still_wins_over_it():
    """`--seed` stays the mode that cannot be half-applied, for the reason above; the new
    flag sits between it and the two provider modes.
    """
    from functional_process.run_cold_matrix import NATIVE

    assert _mode(["--native"]) == NATIVE
    assert _mode(["--seed", "--native"]) == SEED_ONLY
    assert _mode(["--native", "--provider-strict"]) == NATIVE


def test_a_native_row_reports_no_value_from_process_at_all():
    """The one number a native row exists to state. `written == supplied` and
    `from_process == 0` are true **by construction** -- there is no seed in a native run
    to fall back to -- so a non-zero `from_process` here would mean the mode had grown a
    PROCESS dependency without anyone noticing.
    """
    from functional_process.run_cold_matrix import NATIVE, _native_counts

    class _State:
        sources = {("a", "b"): "indat", ("a", "c"): "defaults", ("d", "e"): "defaults"}

    counts = _native_counts(_State(), NATIVE)
    assert counts["from_process"] == counts["held"] == counts["nothing"] == 0
    assert counts["written"] == counts["supplied"] == counts["paths"] == 3
    assert (counts["indat"], counts["defaults"]) == (1, 2)


def test_a_place_the_native_state_could_not_answer_reaches_the_notes():
    """A miss is seeded `0.0` by `mdf.seed`'s own fallback, so the table has to say which
    places those were or a hole is indistinguishable from a real zero. Measured: **zero**
    on all seven configurations, which is a claim that only means something if the
    reporting works when it is not zero.
    """
    row = Row(
        name="x",
        assembles=True,
        omitted_paths=(".physics.made_up", ".build.also_made_up"),
    )
    text = render([row])
    assert "UNANSWERED NATIVELY" in text
    assert ".physics.made_up" in text and ".build.also_made_up" in text


# ------------------------------- the "matches PROCESS" columns (§24.10)
#
# `reference_cold_matrix.txt` carried PROCESS's iteration count and never PROCESS's
# *answer*, so every "matches" claim made off it compared the port against itself under
# another seeding mode -- `_audit/next_steps.md` §17.2's error, repeated twice after
# §17.2 named it. These check the three columns that close it, and in particular that a
# documented disagreement is labelled rather than reported as a failure.


class _Reference:
    """The four fields `_against_process` reads off a `ReferenceRun`."""

    def __init__(self, ixc, converged):
        self.ixc = ixc
        self.converged = converged


def test_the_worst_deviation_is_over_the_whole_design_vector():
    """One number and the iteration variable it happened at -- `run_sand_harness.py`'s
    own `rel` column, worst-cased, computed the same way.
    """
    store = _blank()
    store.update(_x=(1.0, 2.2, 3.0))
    _against_process(store, _Reference([4, 6, 29], {4: 1.0, 6: 2.0, 29: 3.0}), None)
    assert store["dx"] == pytest.approx(0.1)
    assert store["dx_at"] == 6
    assert store["dobjf"] is None


def test_a_native_row_compares_against_nothing_because_there_was_no_process_run():
    """A `NativeReference` has no `converged` and no objective, and the columns must
    stay empty rather than fall back to something -- `PRO` is `-` on those rows for the
    same reason.
    """
    store = _blank()
    store.update(_x=(1.0,), objf=5.0)
    _against_process(store, _Reference([4], {}), None)
    assert store["dx"] is None
    assert store["dobjf"] is None
    assert store["explained"] == ""


def test_a_documented_disagreement_is_labelled_not_failed():
    """The stellarator's 0.23 %: `objective_metric_6` is `coe/100`, and `.costs.coe` is
    the tail of the `+17.604 MW` chain `EXPLAINED_DISAGREEMENTS` documents, where
    PROCESS's converged `DataStructure` is internally inconsistent and the port is the
    self-consistent side. A column that called that a regression would be worse than no
    column.
    """
    store = _blank()
    store.update(_x=(1.0,), objf=1.21775735)
    _against_process(
        store,
        _Reference([4], {4: 1.0}),
        1.2149167845171462,
        (".heat_transport.p_plant_electric_base_total_mw", ".costs.coe"),
    )
    assert store["dobjf"] == pytest.approx(2.3e-3, rel=0.1)
    assert store["explained"] == ".heat_transport.p_plant_electric_base_total_mw"
    table = render([
        Row(name="stellarator_helias", assembles=True, mdf=store, process_objf=1.2149)
    ])
    assert "EXPLAINED GAP, not a regression" in table


def test_an_objective_gap_at_a_moved_design_is_not_explained():
    """The second half of the rule, and the part that keeps the label honest: the chain
    is a difference in *evaluating* the objective at a shared point. A row whose design
    vector has moved is a different answer and gets no label.
    """
    store = _blank()
    store.update(_x=(1.5,), objf=1.21775735)
    _against_process(
        store,
        _Reference([4], {4: 1.0}),
        1.2149167845171462,
        (".heat_transport.p_plant_electric_base_total_mw", ".costs.coe"),
    )
    assert store["dx"] == pytest.approx(0.5)
    assert store["explained"] == ""
    # ...and the withholding is reported rather than silent, because "a documented chain
    # is in play but this comparison cannot attribute the gap to it" is a third state,
    # not the absence of the first two.
    assert store["withheld"] == ".heat_transport.p_plant_electric_base_total_mw"
    table = render([Row(name="stellarator_helias", assembles=True, mdf=store)])
    assert "NOT labelled explained" in table
    assert "two DIFFERENT points" in table


def test_every_explanation_cites_a_record_that_still_exists():
    """A cell reading EXPLAINED claims somebody already chased this and wrote down why.
    If the write-up is deleted and the label outlives it, the cell asserts something with
    nothing behind it -- strictly worse than the unlabelled number it replaced. Checked
    at import too, so the failure lands on whoever moved the record.
    """
    assert EXPLAINED_OBJECTIVE_READS
    for read, key in EXPLAINED_OBJECTIVE_READS.items():
        assert key in EXPLAINED_DISAGREEMENTS, read


def test_a_root_find_row_says_process_formed_no_objective():
    """`none`, not a number and not a blank.

    `reference.i_figure_merit` is `7` on both `_eval` files **because
    `numerics.py:154` defaults it there** -- PROCESS's `_Fsolve.solve` ends
    `self.objf = None` and never evaluates a metric. Printing `objective_function(7,
    data)` in a column headed `PRO objf` would be inventing PROCESS's answer.
    """
    store = _blank()
    store.update(built=True, iterations=3, status="converged", max_eq=3.6e-14)
    table = render([
        Row(name="large_tokamak_eval", assembles=True, root_find=True, mdf=store)
    ])
    assert "none" in table
    # And the short circuit is in `_process_objective` itself, not only in the renderer.
    assert _process_objective(_Reference([], {}), root_find=True) is None


def test_the_table_carries_its_own_provenance_header(tmp_path):
    """The header is emitted by `checkpoint`, not hand-written on top afterwards.

    That is the whole change: the previous table's provenance was a hand-added block, so
    the first re-run silently deleted it and a reader of the new file had no way to know
    what tree the rows came from. A provenance line a re-run destroys is worse than none,
    because its absence is invisible.
    """
    out = tmp_path / "matrix.txt"
    checkpoint(
        [Row(name="helias_5b", assembles=True)],
        out,
        PROVIDER,
        ["--provider"],
        compare=True,
    )
    text = out.read_text()
    assert text.startswith("# Generated by")
    assert "MEASURED" in text
    assert "TREE: HEAD" in text
    assert "SEEDED    `--provider`" in text
    assert "COLD MATRIX" in text
    assert "helias_5b" in text


def test_the_header_states_seeding_and_scoring_as_two_separate_lines(tmp_path):
    """A reader must never take a filled `PRO objf` for "seeded from PROCESS".

    The two axes were one until 2026-09-01 -- a `--native` row had `process_objf = None`
    by construction -- and that conflation was the only surviving reason to run
    `--provider` at all (`_audit/optimise_design.md` §27). The header says both things
    and says they are two, and the `seed` column repeats the first one on every line, so
    the mistake needs three separate misreadings rather than one.
    """
    out = tmp_path / "matrix.txt"
    checkpoint(
        [Row(name="helias_5b", seed_mode=NATIVE, compared=True, assembles=True)],
        out,
        NATIVE,
        ["--native", "--compare-process"],
        compare=True,
    )
    text = out.read_text()
    assert "SEEDED    `--native`" in text
    assert "SCORED    against PROCESS's converged answer" in text
    assert "TWO AXES" in text
    # and the row itself carries the seeding mode, not the scoring one
    assert " nat " in text

    off = tmp_path / "off.txt"
    checkpoint(
        [Row(name="helias_5b", seed_mode=NATIVE, assembles=True)],
        off,
        NATIVE,
        ["--native"],
        compare=False,
    )
    assert "SCORED    NOT against PROCESS" in off.read_text()


def test_a_native_row_can_be_scored_against_process_without_being_seeded_from_it():
    """`_solve_both`'s `oracle` is the scoring side and `reference` the seeding side.

    They are the same object under `--provider` and two different objects under
    `--native --compare-process`, which is the whole point of the split: `dx` and
    `d objf` come off PROCESS's converged run while every value the solve started from
    came off `native.native_state`.
    """
    store = _blank()
    store.update(objf=2.0, _x=(11.0,))
    _against_process(store, _Reference([1], {1: 10.0}), 1.0, ixc=[1])
    assert store["dx"] == pytest.approx(0.1)
    assert store["dx_at"] == 1
    assert store["dobjf"] == pytest.approx(1.0)


def test_a_permuted_ixc_blanks_the_dx_column_rather_than_comparing_the_wrong_variables():
    """`native.native_reference` sorts `ixc` and `SingleRun.init` sorts PROCESS's, so the
    two agree on all seven files today -- but a `dx` computed by zipping two differently
    ordered ID lists would report a permutation as a per-variable disagreement, silently,
    in the one column that exists to catch silent disagreement.
    """
    store = _blank()
    store.update(objf=2.0, _x=(11.0, 21.0))
    _against_process(store, _Reference([1, 2], {1: 10.0, 2: 20.0}), 1.0, ixc=[2, 1])
    assert store["dx"] is None
    assert store["dobjf"] == pytest.approx(1.0)


def test_comparing_is_free_where_process_ran_and_opt_in_where_it_did_not():
    """The default is not a preference, it is a cost: the three seeding modes that build
    their start out of `reference_run` have PROCESS's answer in hand already, and
    `--native` is the mode whose claim is that PROCESS is not in the path.
    """
    assert compares_by_default(PROVIDER)
    assert compares_by_default(PROVIDER_STRICT)
    assert compares_by_default(SEED_ONLY)
    assert not compares_by_default(NATIVE)
    # the flags override in both directions, on every mode
    assert _compare(["--native", "--compare-process"], NATIVE)
    assert not _compare(["--provider", "--no-compare-process"], PROVIDER)
    assert _compare(["--provider"], PROVIDER)
    assert not _compare(["--native"], NATIVE)


def test_the_header_names_uncommitted_edits_and_does_not_claim_the_commit():
    """`git rev-parse HEAD` alone is a lie on a dirty tree, and every table this port has
    produced was produced on one. So the header names the commit *and* every file of
    `PORT_FILES` that differs from it -- a row measured against `mdf.py` plus two hundred
    uncommitted lines is not a row measured against that commit.
    """
    lines = provenance()
    assert any(line.startswith("# TREE: HEAD") for line in lines)
    body = "\n".join(lines)
    # Exactly one of the two claims is made, never both and never neither.
    dirty = "UNCOMMITTED edits" in body
    clean = "is clean, so these rows are that commit's" in body
    assert dirty != clean
    # And `PORT_FILES` is the path a row's numbers actually travel, not the package.
    assert {"mdf.py", "sand.py", "run_cold_matrix.py", "native.py"} <= set(PORT_FILES)


def test_a_root_find_file_gets_one_row_and_the_table_says_why():
    """An evaluation-mode file has no MDF/SAND split, and the single line must read as a
    decision rather than as a missing measurement.

    This test previously pinned the opposite: SAND assembled an `Optimise` on the
    `_eval` files, reported an `objf` beside a `PRO objf` of `none`, and the table
    labelled that mismatch. The label was right about the mismatch and wrong about the
    remedy -- MDF-against-SAND is a split between two ways of distributing an
    *optimisation*, and a file stating `i_process_run_mode = -2` states none, so the
    second row was never a second reading of the same problem. It is gone, and what
    replaces it is the argument for its absence.
    """
    store = _blank()
    store.update(built=True, iterations=3, status="conv", dx=3.6e-09)
    table = render([
        Row(name="spherical_tokamak_eval", assembles=True, root_find=True, mdf=store)
    ])
    assert "states a ROOT FIND" in table
    assert "ONE row and not two" in table
    # The single line is MDF's, and no SAND line is printed at all. Counted above the
    # NOTES block, since the note that explains the absence necessarily says "SAND".
    grid = table.split("\nNOTES")[0].split("\n")
    body = [line for line in grid if "spherical_tokamak_eval" in line]
    assert len(body) == 1
    assert " MDF " in body[0]
    assert " SAND " not in body[0]


def test_an_objf_on_a_root_find_row_is_flagged_as_unexpected():
    """Nothing produces one today -- `mdf_graph` mints no objective node when the file
    names no figure of merit -- but a number in that cell would be `objective_metric_7`
    at `numerics.py:154`'s default, which is not a metric this file or PROCESS ever
    chose, sitting beside a `PRO objf` of `none` and looking comparable to it.
    """
    store = _blank()
    store.update(built=True, iterations=3, status="conv", objf=0.594644641)
    table = render([
        Row(name="spherical_tokamak_eval", assembles=True, root_find=True, mdf=store)
    ])
    assert "UNEXPECTED `objf`" in table
    assert "compares to nothing" in table
