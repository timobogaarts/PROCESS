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

from functional_process.run_cold_matrix import (  # noqa: E402
    CONFIGURATIONS,
    PROVIDER,
    PROVIDER_STRICT,
    SEED_ONLY,
    Row,
    _blank,
    _boundary_seed,
    _cell,
    _headline,
    _mode,
    _status,
    _trace_tail,
    checkpoint,
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
