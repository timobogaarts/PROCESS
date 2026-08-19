"""Assembly-level tests for `functional_process.mdf`, plus one end-to-end cold run.

Same division as `test_sand.py`: everything checkable without a 95-second PROCESS solve
is checked here, and the ladder against PROCESS's own answer lives in
`run_mdf_harness.py`. Two things make this module's split slightly different from that
one:

- **What MDF claims is structural**, so most of it *is* checkable here -- that the
  optimiser's problem is exactly PROCESS's (eight design variables, fifteen conditions,
  no coupling anywhere in either), and that cottax states the nesting and refuses to run
  it are both assertions about objects, not about numbers.
- **The one claim that is not** -- that `jax.jacfwd` through thirteen driven blocks is
  the right derivative -- cannot be made without running the thing, so
  `test_the_gradient_through_the_inner_solve_is_correct` runs a real cold MDA. It is the
  crux of the whole architecture (a converging solve with a wrong derivative is worthless
  and looks fine), so it is a test and not only a harness line. It costs ~20 s and is
  marked `tier4`.
"""

import dataclasses
import operator

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cottax.evaluate import ConditionMap, schedule_for
from cottax.problem import Optimise

from functional_process import mdf, sand
from functional_process.core.solver.drivers import SeededNewtonDriver
from functional_process.mda import default_drivers
from functional_process.sand_harness import REFERENCE_INPUT_FILE, _scratch_copy
from functional_process.test_sand import (
    REFERENCE_FIGURE_OF_MERIT,
    REFERENCE_ICC,
    REFERENCE_IXC,
    REFERENCE_N_EQUALITY,
)
from process.main import SingleRun

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def problem():
    """The reference run's MDF assembly. Structural only -- no PROCESS run, no values."""
    return mdf.assemble(
        REFERENCE_IXC, REFERENCE_ICC, REFERENCE_N_EQUALITY, REFERENCE_FIGURE_OF_MERIT
    )


def test_the_optimisers_problem_is_exactly_the_one_process_states(problem):
    """Eight design variables and fifteen conditions -- `ixc` and `icc` and nothing else.

    This is the definition of MDF as against SAND, stated as an assertion: SAND's design
    vector is `ixc` **plus** every coupling variable and its equalities are PROCESS's
    **plus** one residual per coupling variable, which is why its Jacobian has to be
    Schur-reduced before it can be compared with PROCESS's. If this test ever passes with
    more than `len(ixc)` design variables, the module has stopped being MDF.
    """
    assert [v.path_str() for v in problem.design] == [
        sand.iteration_variable_path(i).path_str() for i in REFERENCE_IXC
    ]
    assert len(problem.conditions) == 1 + len(REFERENCE_ICC)
    assert problem.conditions[0].path_str() == "^cond.numerics.objf"
    assert [c.path_str() for c in problem.conditions[1:]] == [
        f"^cond.constraints.c{cid}" for cid in REFERENCE_ICC
    ]
    assert problem.n_equality == REFERENCE_N_EQUALITY
    assert problem.n_inequality == len(REFERENCE_ICC) - REFERENCE_N_EQUALITY


def test_no_coupling_variable_reaches_the_optimiser(problem):
    """Every condition is one of PROCESS's own, and every unknown of the MDA stays inside
    the MDA.

    The positive half of the previous test: the coupling variables (`mdf.eager.unknowns`
    -- the fixed points and root finds SAND turns into design variables and equalities)
    must appear in *neither* `design` nor `conditions`. That is what makes
    `VmconDriver.condition_scale` unnecessary here (nothing carries physical units into
    the constraint vector) and what makes the design bounds PROCESS's own without
    invention.
    """
    inner = set(problem.eager.unknowns)
    assert inner, "the MDA has no driven blocks -- this graph is not the reference one"
    assert not inner & set(problem.design)
    assert not inner & set(problem.conditions)
    for condition in problem.conditions:
        assert condition.path_str().startswith((
            "^cond.constraints.c",
            "^cond.numerics.objf",
        ))


def test_the_design_variables_are_boundary_inputs_of_the_mda(problem):
    """The eight `ixc` are inputs of the schedule, not outputs of any node.

    `assemble` raises if they are not, and this pins that the reference run is in fact in
    that state -- 7 of the 83-entry `ITERATION_VARIABLES` table *are* produced by a node
    (`sand.optimise_graph` records the same fact and the same policy), so this is a
    property of this run, not of the formulation.
    """
    assert set(problem.design) <= set(problem.eager.inputs)


def test_cottax_states_mdf_structurally(problem):
    """`Blocking.nest` records MDF exactly: the `Optimise` outside, the MDA inside.

    The first half of this module's central claim about cottax. `nested_blocking`
    inserts the `Optimise` -- which fuses the optimiser and the whole MDA into one SCC --
    and then nests it, cottax's own documented answer to `Graph.problem_type`'s refusal
    of a block with two problems.
    """
    nested, name, _report = mdf.nested_blocking(
        REFERENCE_IXC, REFERENCE_ICC, REFERENCE_N_EQUALITY, REFERENCE_FIGURE_OF_MERIT
    )
    index = nested.index[name]
    assert issubclass(nested.problem_types[index], Optimise)
    interior = nested.inner[index]
    assert interior is not None
    # The interior is the block minus the node solved at this level, and it holds the
    # MDA's own driven blocks -- the ones `mdf` runs as the inner solve.
    assert name not in interior.index
    assert len(interior.blocks) > 1
    assert sum(1 for t in interior.problem_types if t is not None) > 1


def test_cottax_cannot_run_that_nesting(problem):
    """...and the second half: `schedule_for` ignores `Blocking.inner`, so it refuses.

    Pinned as a test rather than left as a docstring claim because this is the single
    upstream gap the whole module is built around, and the day it stops being true is the
    day `mdf.MdfConditionMap` should be deleted in favour of a real nested `Drive`. A
    failure here is good news, not a regression.
    """
    nested, name, _report = mdf.nested_blocking(
        REFERENCE_IXC, REFERENCE_ICC, REFERENCE_N_EQUALITY, REFERENCE_FIGURE_OF_MERIT
    )
    drivers = dict(default_drivers(problem.eager.blocking))
    drivers[name] = mdf.driver(problem)
    with pytest.raises(ValueError, match=r"declares|problem"):
        schedule_for(nested, drivers)


def test_the_condition_map_is_a_condition_map(problem):
    """`MdfConditionMap` is substitutable for cottax's own, which is what lets an
    unmodified `VmconDriver` answer it.

    The whole local workaround is "a `ConditionMap` whose body is run by a `Schedule`",
    and if it stopped being a `ConditionMap` the claim that only one field's *type* is
    missing upstream would be an overstatement.
    """
    empty = mdf.condition_map(problem, dict.fromkeys(problem.design, jnp.asarray(1.0)))
    assert isinstance(empty, ConditionMap)
    assert empty.unknowns == problem.design
    assert empty.conditions == problem.conditions
    with pytest.raises(TypeError, match="design variable"):
        empty(jnp.asarray(1.0))


def test_traceable_drivers_only_clears_the_seed(problem):
    """The traced schedule differs from the eager one in exactly one field.

    `traceable_drivers` exists to work around an untraceable `np.asarray` in
    `SeededNewtonDriver._usable`, and a workaround that quietly changed a tolerance or a
    driver type would make every traced number incomparable with the eager ones it is
    supposed to reproduce.
    """
    drivers = default_drivers(problem.eager.blocking)
    traceable = mdf.traceable_drivers(drivers)
    assert set(traceable) == set(drivers)
    for name, driver in drivers.items():
        other = traceable[name]
        assert type(other) is type(driver)
        if isinstance(driver, SeededNewtonDriver):
            assert driver.seed is not None
            assert other.seed is None
            assert dataclasses.replace(other, seed=driver.seed) == driver
        else:
            assert other == driver


def test_the_driver_reads_its_split_off_the_assembly(problem):
    """`mdf.driver`'s equality/inequality counts come from the assembly, never from a
    caller counting conditions -- `mda.default_drivers`' own rule.
    """
    driver = mdf.driver(problem)
    assert driver.n_equality == problem.n_equality
    assert driver.n_inequality == problem.n_inequality
    assert 1 + driver.n_equality + driver.n_inequality == len(problem.conditions)
    # MDF has no residual conditions, so nothing is rescaled -- see `mdf.py`.
    assert driver.condition_scale == ()


@pytest.fixture(scope="module")
def cold_run(problem):
    """A cold `DataStructure`, the primed env, and the condition map at the cold point.

    `SingleRun.__init__` runs `init_process` and stops, so this is the input file's own
    starting state -- no solve, ~2 s. Priming runs the MDA once from it.
    """
    data = SingleRun(_scratch_copy(REFERENCE_INPUT_FILE), "vmcon").data
    env, _out = mdf.prime(problem, mdf.seed(problem, data))
    start = tuple(jnp.asarray(env[v]) for v in problem.design)
    return mdf.condition_map(problem, env), start


@pytest.mark.tier4
def test_the_conditions_are_finite_at_a_cold_start(cold_run):
    """Every condition is a real number at the input file's own values.

    The property that costs SAND a seeding rule and MDF nothing. A SAND condition map
    holds the coupling variables at whatever guess it was given, and seeded from a cold
    `DataStructure`'s `0.0`s that made 20 of its 24 conditions `nan`
    (`_audit/optimise_design.md` §10.6); it now cold-starts because
    `run_sand_harness._seed` hands those unknowns a completed MDA run instead. An MDF map
    has no coupling unknowns to seed -- it computes them -- so there is nothing to get
    right here, which is what this test pins.
    """
    conditions, start = cold_run
    values = np.asarray([float(np.asarray(v)) for v in conditions(*start)])
    assert np.all(np.isfinite(values)), dict(
        zip(
            [c.path_str() for c in conditions.conditions],
            values.tolist(),
            strict=True,
        )
    )


@pytest.mark.tier4
def test_the_gradient_through_the_inner_solve_is_correct(cold_run):
    """`jax.jacfwd` through thirteen driven blocks equals a central difference of the
    same map.

    **The crux of the architecture.** MDF differentiates through a converged inner solve:
    twelve `jax.lax.while_loop`s (`PicardDriver`) and an `optimistix` root find. Any of
    the standard failure modes there -- a JVP that does not match the loop that ran, an
    implicit-function derivative on a root find that did not converge, a `stop_gradient`
    hiding in a model -- produces a solve that still converges, to the wrong place, with
    every value test passing. Only this comparison sees it.

    The control is a central difference *of the port's own condition map*, so both sides
    share every model and every solver and differ only in autodiff; agreement therefore
    isolates the derivative. Agreement with PROCESS's own finite differences is a
    different and weaker claim and lives in `run_mdf_harness.py`'s Stage B.

    Cells the difference cannot resolve (below `1e-8` of the largest entry, mostly
    structural zeros) are excluded -- comparing `1e-21` against `1e-30` measures nothing.
    """
    conditions, start = cold_run
    full, _compile_seconds, _ms = mdf.jacobian(conditions, start, repeats=1)
    assert np.all(np.isfinite(full))
    finite = mdf.central_difference(conditions, start)
    alive = np.abs(finite) > 1e-8 * np.abs(finite).max()
    relative = np.abs(full - finite) / np.maximum(np.abs(finite), 1e-300)
    worst = float(relative[alive].max())
    off = [
        (conditions.conditions[i].path_str(), int(j))
        for i, j in zip(*np.where(alive & (relative > 1e-4)), strict=True)
    ]
    assert worst < 1e-4, (
        f"autodiff and a central difference of the same map disagree by "
        f"{worst:.2e} at {off}"
    )


@pytest.mark.tier4
def test_the_inner_solve_is_actually_converged_at_that_point(problem, cold_run):
    """Every driven block's own residual is small where the gradient was taken.

    `PicardDriver` runs `lax.while_loop` to `max_iter` and cannot raise, so "the MDA
    converged" is an assumption unless something checks it. It is also the assumption the
    whole formulation rests on: MDF's conditions are only PROCESS's constraints if the
    inner solve reached the same fixed point PROCESS's ten Gauss-Seidel passes do.
    """
    conditions, start = cold_run
    env = dict(conditions.context)
    env.update(zip(problem.design, start, strict=True))
    rows = mdf.inner_residuals(problem.eager, problem.eager(env))
    assert rows
    worst = max(rows, key=operator.itemgetter(3))
    assert worst[3] < 1e-6, (
        f"{worst[1].path_str()} is only converged to {worst[3]:.2e} relative"
    )


@pytest.mark.tier4
def test_one_pass_of_the_schedule_is_idempotent(problem, cold_run):
    """PROCESS's own convergence test, applied to **one** pass of the port's schedule.

    This is the question "does MDF need an outer fixed-point iteration on top of the
    per-block drivers?", answered by measurement rather than by argument.
    `Caller.call_models` runs the whole pipeline at least twice and up to ten times and
    stops when the objective and the constraints agree to `rtol = 1e-6`
    (`process/core/caller.py:96-126`); here the same comparison is made between one pass
    and two, with `np.allclose`'s own `rtol = 1e-6` -- `check_agreement`'s exact rule.

    The structural answer is that it must pass: `Blocking.__check_init__` refuses any
    block that reads what a later one owns, so the block order *is* a topological order
    over the condensation, and every cyclic block has been driven to its own problem's
    answer by the time the pass leaves it. PROCESS re-runs everything ten times because
    it has no condensation to order, not because the system needs it. But "must" is what
    a docstring says and this is what a test says.
    """
    conditions, start = cold_run
    env = dict(conditions.context)
    env.update(zip(problem.design, start, strict=True))
    once = problem.eager(dict(env))
    twice = problem.eager(dict(once))
    for condition in problem.conditions:
        first = np.asarray(once[condition], dtype=float)
        second = np.asarray(twice[condition], dtype=float)
        assert np.allclose(first, second, rtol=1e-6, equal_nan=True), (
            f"{condition.path_str()} moved from {first} to {second} on a second pass, "
            f"so one pass of the schedule is not the converged MDA"
        )


def test_x64_is_on():
    """Everything above is float64. A harness that silently ran in float32 would compare
    a 1e-7 derivative disagreement against a 1e-7 noise floor.
    """
    assert jax.config.jax_enable_x64
