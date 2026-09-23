"""`paper_tests/one_dial.py`: the recipe, and the one number that has to be PROCESS's.

Two checks, and the second is the load-bearing one. The recipe is four named steps and
each records the operations it applied, so the shape of the recipe is testable as data
(`Step.ops`) rather than by reading the printout. And at PROCESS's own `hfact`, with
the build frozen at PROCESS's own design, the variable that absorbs the power balance
must land on PROCESS's own value for it: the density PROCESS iterated to on the
stellarator, and on the driven machine the 10 MW of heating PROCESS holds as an input
and never questions. If that number moves, the closure is not asking PROCESS's
question.

The two machines also differ in **which step draws the refusal**, and that is pinned
here too: `stellarator_helias` has `hfact` as an active `ixc`, so giving it away is
what over-determines the problem; `low_aspect_ratio_DEMO` fixes `hfact` in the file, so
step 1 is a no-op and the count breaks at `freeze_the_build` instead.
"""

from __future__ import annotations

import pytest

from paper_tests import one_dial
from paper_tests.common import process_reference


@pytest.fixture(scope="module")
def ignited():
    """The stellarator's recipe, built once: the density absorbs `c2`."""
    return one_dial.recipe("stellarator_helias")


@pytest.fixture(scope="module")
def driven():
    """`low_aspect_ratio_DEMO`'s recipe, built once: the heating power absorbs `c2`."""
    return one_dial.recipe("low_aspect_ratio_DEMO")


def nominal_point(plan):
    """The first row of the sweep: PROCESS's own `hfact`, the build held."""
    (row,) = one_dial.sweep(plan, one_dial.DIAL, [plan.values[one_dial.DIAL]])
    return row


# ---------------------------------------------------------------- the recipe


@pytest.mark.parametrize("which", ["ignited", "driven"])
def test_the_recipe_is_four_steps(which, request):
    """Read the file, give the dial away, name what absorbs the equation, freeze."""
    plan = request.getfixturevalue(which)
    assert [step.name for step in plan.steps] == [
        "read the input file",
        "make hfact a given",
        plan.steps[2].name,  # the absorbing variable names this one
        "freeze the build",
    ]
    assert plan.chosen == ()  # the machine is built: nothing is left to choose


@pytest.mark.parametrize("which", ["ignited", "driven"])
def test_exactly_one_step_draws_the_refusal(which, request):
    """The graph refuses once, and the recipe records which step drew it."""
    plan = request.getfixturevalue(which)
    drew = [n for n, step in enumerate(plan.steps) if one_dial.REFUSAL in step.note]
    assert len(drew) == 1
    quoted = plan.steps[drew[0]].note
    assert "against nothing: it has no unknowns" in quoted
    assert ".constraints.c2" in quoted
    assert "REFUSED" in one_dial.report(plan)


def test_the_stellarator_refuses_when_the_dial_is_given_away(ignited):
    """`hfact` is `ixc = 10` here, so step 1 is what over-determines the problem."""
    assert one_dial.DIAL in one_dial.read_the_input_file(ignited.machine).chosen
    assert one_dial.REFUSAL in ignited.steps[1].note
    assert ignited.steps[1].ops == ()


def test_the_driven_machine_refuses_only_when_the_build_is_frozen(driven):
    """`hfact` is fixed in this file, so step 1 is a no-op and step 3 breaks the count."""
    assert one_dial.DIAL not in one_dial.read_the_input_file(driven.machine).chosen
    assert one_dial.REFUSAL not in driven.steps[1].note
    assert "no-op" in driven.steps[1].note
    assert one_dial.REFUSAL in driven.steps[3].note


def test_the_density_closure_is_one_determine(ignited):
    """Step 2, ignited: one requirement `Determine`d and nested, the density given up."""
    step = ignited.steps[2]
    assert step.ops == (
        "Determine(Requirement(.constraints.c2), "
        "(.physics.nd_plasma_electrons_vol_avg,)) at .Close.c2",
        "nested_inside(.Close.c2)",
    )
    assert ignited.graph.report["closing"] == {
        one_dial.BALANCE: one_dial.DENSITY
    }


def test_the_heating_closure_is_one_determine_over_a_plain_input(driven):
    """Step 2, driven: the same one operation, over a variable PROCESS never iterates."""
    step = driven.steps[2]
    assert step.ops == (
        "Determine(Requirement(.constraints.c2), "
        "(.current_drive.p_hcd_primary_extra_heat_mw,)) at .Close.c2",
        "nested_inside(.Close.c2)",
    )
    assert one_dial.HEATING not in one_dial.read_the_input_file(driven.machine).chosen


@pytest.mark.parametrize("which", ["ignited", "driven"])
def test_freezing_the_build_is_a_cut_and_a_check_per_re_sized_output(which, request):
    """Step 3: every build output the dial still reaches is `Cut` and checked, so the
    recipe's last step is two operations per frozen place and nothing else.
    """
    step = request.getfixturevalue(which).steps[3]
    assert step.ops, "the dial reaches at least one claimed build output"
    assert len(step.ops) % 2 == 0
    assert all(op.startswith("Cut(") for op in step.ops[::2])
    assert all(op.startswith("Insert(.Built.") for op in step.ops[1::2])


# ---------------------------------------------------------------- the number


def test_the_ignited_nominal_point_is_process_s_own_density(ignited):
    """At PROCESS's `hfact`, with its design held, the closure finds its density."""
    row = nominal_point(ignited)
    assert row["converged"]
    assert row[one_dial.DENSITY] == pytest.approx(
        process_reference("stellarator_helias")["x"][one_dial.DENSITY], rel=1e-6
    )
    assert abs(row["conditions"][".constraints.c24"]) < 1e-2  # on its beta limit


def test_the_driven_nominal_point_is_process_s_own_heating_power(driven):
    """At PROCESS's `hfact`, the power balance asks for exactly the heating the file
    installs -- 10 MW, which PROCESS states as an input and never iterates.
    """
    row = nominal_point(driven)
    assert row["converged"]
    assert row[one_dial.HEATING] == pytest.approx(driven.values[one_dial.HEATING], rel=1e-6)
    assert driven.values[one_dial.HEATING] == pytest.approx(10.0)


def test_a_frozen_capacity_check_is_not_counted_as_a_tight_condition(ignited):
    """A `^cond.built.*` check is zero the day it is frozen, so it is reported apart."""
    (row,) = one_dial.sweep(ignited, one_dial.DIAL, [ignited.values[one_dial.DIAL]])
    assert row["frozen"], "the dial reaches at least one claimed build output"
    assert all(s.startswith("^cond.built.") for s in row["frozen"])
    assert not any(s.startswith("^cond.built.") for s in row["tight"])
