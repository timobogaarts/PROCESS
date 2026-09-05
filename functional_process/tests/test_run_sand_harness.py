"""Argument parsing for the `Optimise` ladder's runner.

The ladder itself is not tested here and cannot be: every stage needs a live PROCESS
solve (~95 s for the stellarator) and the numbers it produces are recorded in
`_audit/optimise_design.md`, not pinned in a test. What *is* testable, and what a stage
selector makes worth testing, is which stages an invocation asks for -- because getting
that wrong records a number against the wrong measurement, silently.
"""

import pytest

from functional_process.cottax.run_sand_harness import stages


def test_no_flag_runs_all_three_stages():
    """The default is the whole ladder, exactly as it was before a selector existed."""
    assert stages([]) == "ABC"
    assert stages(["--machine"]) == "ABC"
    assert stages(["--input", "some.IN.DAT"]) == "ABC"


def test_a_subset_is_taken_verbatim_and_case_insensitively():
    assert stages(["--stages", "A"]) == "A"
    assert stages(["--stages", "a"]) == "A"
    assert stages(["--stages", "AC"]) == "AC"
    assert stages(["--stages", "ABC"]) == "ABC"


def test_stage_a_alone_is_the_case_this_exists_for():
    """`_audit/next_steps.md` §24.10 item 3 asks for the port's objective at PROCESS's
    own converged x, which is Stage A's `^cond.numerics.objf` row and nothing else.
    Before the flag, taking it meant sitting through Stage B's finite-difference
    Jacobian and Stage C's two solves, or writing a scratch script restating `main`'s
    setup -- the third harness `run_cold_matrix.py` exists to refuse.
    """
    assert stages(["--input", "x.IN.DAT", "--stages", "A"]) == "A"


def test_an_unknown_stage_is_refused_rather_than_partially_honoured():
    """`--stages D` must not quietly mean "A, B and C" -- nor may `--stages AD` quietly
    mean `A`. A selector that runs the stages it recognised and says nothing about the
    one it did not is how a measurement gets filed under the wrong name.
    """
    with pytest.raises(SystemExit, match="not a stage"):
        stages(["--stages", "D"])
    with pytest.raises(SystemExit, match="not a stage"):
        stages(["--stages", "AD"])


def test_the_flag_needs_a_value():
    """Same rule, and the same reason, as `run_mda_harness.input_file`'s `--input`: a
    harness that silently fell back to the default here would print a full report for a
    different measurement than the one asked for.
    """
    with pytest.raises(SystemExit, match="subset of ABC"):
        stages(["--stages"])
    with pytest.raises(SystemExit, match="subset of ABC"):
        stages(["--stages", "--machine"])
