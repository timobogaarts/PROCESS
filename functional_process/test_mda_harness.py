"""`mda_harness.compare`'s tolerance policy and its agreement classification.

Not the harness run itself -- that needs a converged PROCESS `DataStructure` and is
`run_mda_harness.py`'s job, treated as a standing regression check
(`_audit/next_steps.md`, "Verified state"). What is pinned here is the comparison
*policy*, which is what `next_steps.md` §11.6 item 6 was about and what measurement
settled: the absolute floor is gone -- justified against a known defect reintroduced on
purpose, not against the clean run -- and the *second* vacuous category, which nobody
had named, is reported instead of folded into the headline count.
"""

import inspect

import numpy as np

from functional_process.mda_harness import _is_trivially_zero, compare


def test_compare_defaults_to_a_purely_relative_tolerance():
    """`atol` is `0.0`, and that is a decision, not an omission.

    An absolute floor cannot serve a `DataStructure` whose fields span 1e-15 to 1e+20:
    it exempts whole subsystems by their choice of units, which is not a property
    anyone chose as a tolerance policy.
    """
    # RUF069 does not apply: this compares a *declared literal default* against the
    # literal it is meant to be, not two computed floats.
    assert inspect.signature(compare).parameters["atol"].default == 0.0  # noqa: RUF069


def test_the_old_floor_hid_the_rho_bug_and_the_new_one_does_not():
    """The measurement that justifies the change: a **known** defect, reintroduced.

    Setting `ProfileValues.rho` back to `0.0` (the old `FromExactly(r_eff)` binding's
    value, `neoclassics.py`'s own docstring) and re-running the MDA harness gives the
    four values below. The two Joule-valued fields are 71 % and 100 % wrong and the old
    `atol=1e-9` reports **both as agreements**; at `atol=0.0` both are disagreements,
    and the harness sees 4 of that bug's fields instead of 2.

    This is the whole argument, and it is not the one the removal was first made on.
    On the code as it stands `atol` is inert -- 0 of 499 agreements depend on it -- and
    that reads as "the trap was never sprung". It is inert only because this bug was
    fixed by another route. **A guard that currently catches nothing has not thereby
    been shown to be unnecessary**; the measurement that settles its worth is taken
    with the defect present (`_audit/next_steps.md` §11.7).
    """
    hidden_by_the_old_floor = [
        (1.999686e-15, 1.170517e-15),  # .neoclassics.temperatures[0], 71 % off
        (-0.0, -1.215030e-15),  # .neoclassics.dr_temperatures[0], 100 % off
    ]
    for got, expected in hidden_by_the_old_floor:
        assert np.isclose(got, expected, rtol=1e-6, atol=1e-9), "the floor hid it"
        assert not np.isclose(got, expected, rtol=1e-6, atol=0.0), "and now it does not"

    # The same bug's two large-magnitude fields were never hidden -- which is exactly
    # why the defect was found at all, and why the small ones stayed invisible.
    caught_either_way = [
        (2.357046e20, 2.016188e20),  # .neoclassics.densities[0]
        (-0.0, -6.104176e19),  # .neoclassics.dr_densities[0]
    ]
    for got, expected in caught_either_way:
        assert not np.isclose(got, expected, rtol=1e-6, atol=1e-9)
        assert not np.isclose(got, expected, rtol=1e-6, atol=0.0)


def test_the_three_sub_floor_fields_are_checked_now_that_rho_is_right():
    """With the bug fixed these agree **bit-exactly**, so they need no floor.

    §11.6 item 6 called them "not actually checked by anything". That was true while
    `rho` was wrong and is not true now -- the values are the reference run's own, port
    and `data` alike, identical to every digit. Pinned so the claim is a number rather
    than a recollection.
    """
    for got, expected in (
        (6.426511641e-23, 6.426511641e-23),  # .physics.sigmav_dt_average
        (1.170517108e-15, 1.170517108e-15),  # .neoclassics.temperatures[0], Joules
        (-1.215029950e-15, -1.215029950e-15),  # .neoclassics.dr_temperatures[0]
    ):
        assert np.isclose(got, expected, rtol=1e-6, atol=0.0)


def test_a_pair_that_is_zero_on_both_sides_still_agrees_with_no_floor():
    """`|0 - 0| <= rtol * 0` holds, so removing the floor does not turn the 73
    both-sides-zero pairs into disagreements -- they stay agreements, and get counted
    separately instead.
    """
    assert np.isclose(0.0, 0.0, rtol=1e-6, atol=0.0)
    assert np.isclose(np.zeros(4), np.zeros(4), rtol=1e-6, atol=0.0).all()


def test_trivially_zero_needs_both_sides_zero_everywhere():
    """The classification is "no arithmetic was exercised", not "the expected value is
    zero" -- a port that produced a nonzero answer where PROCESS has zero is a
    disagreement and never reaches this test, and one nonzero element is enough to
    make an array's agreement a real one.
    """
    assert _is_trivially_zero(0.0, 0.0)
    assert _is_trivially_zero(np.zeros(4), np.zeros(4))
    assert not _is_trivially_zero(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]))
    assert not _is_trivially_zero(1e-30, 0.0)
    assert not _is_trivially_zero(0.0, 1e-30)


def test_a_nan_pair_is_not_trivially_zero():
    """`nan` is not zero, and an agreeing `nan`/`nan` pair (`equal_nan=True` in
    `_diff`) is a real fact about the port worth leaving in the headline count.
    """
    assert not _is_trivially_zero(np.nan, np.nan)
