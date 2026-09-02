"""Assembly-level tests for `functional_process.mdf`, plus one end-to-end cold run.

Same division as `test_sand.py`: everything checkable without a 95-second PROCESS solve
is checked here, and the ladder against PROCESS's own answer lives in
`run_mdf_harness.py`. Two things make this module's split slightly different from that
one:

- **What MDF claims is structural**, so most of it *is* checkable here -- that the
  optimiser's problem is exactly PROCESS's (eight design variables, fifteen conditions,
  no coupling anywhere in either), and that cottax both states *and runs* the nesting are
  assertions about objects, not about numbers. (It used to be "states and refuses to
  run": the refusal was about a missing `Assign` and the regex pinning it was loose
  enough not to notice -- `_audit/in_graph_rootfind.md` §1.)
- **The one claim that is not** -- that `jax.jacfwd` through thirteen driven blocks is
  the right derivative -- cannot be made without running the thing, so
  `test_the_gradient_through_the_inner_solve_is_correct` runs a real cold MDA. It is the
  crux of the whole architecture (a converging solve with a wrong derivative is worthless
  and looks fine), so it is a test and not only a harness line. It costs ~20 s and is
  marked `tier4`.
"""

import dataclasses
import operator
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cottax.blocking import Blocking
from cottax.evaluate import ConditionMap, Drive, Schedule, schedule_for
from cottax.graph import Graph
from cottax.problem import (
    Converged,
    FixedPoint,
    Optimise,
    Residual,
    RootFind,
    Start,
    Steps,
    conditions_of,
    driver_vars,
    unknowns_of,
)

from functional_process import mdf, sand
from functional_process.core.solver.drivers import SeededNewtonDriver
from functional_process.mda import assign_drivers, default_drivers
from functional_process.run_mdf_harness import MAX_ITER, _why_it_stopped
from functional_process.sand_harness import REFERENCE_INPUT_FILE, _scratch_copy
from process.main import SingleRun
from tests.functional_process.test_sand import (
    REFERENCE_FIGURE_OF_MERIT,
    REFERENCE_ICC,
    REFERENCE_IXC,
    REFERENCE_N_EQUALITY,
)

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


def test_the_undriven_nesting_is_refused_for_want_of_an_assign():
    """`nested_blocking` attaches no drivers, so `schedule_for` refuses it -- and the
    refusal names the missing `Assign`, not the nesting.

    **This test used to claim the opposite** (`test_cottax_cannot_run_that_nesting`:
    *"`schedule_for` ignores `Blocking.inner`, so it refuses"*), and it passed for years
    of the port's life because its regex was `r"declares|problem"`, which the driver
    message matches on the word "problem". The claim was falsified upstream and the test
    could not see it. So the message is matched exactly here, and
    `test_cottax_runs_that_nesting_once_the_drivers_are_assigned` is the other half:
    together they say *why* it refuses, which is the thing a loose regex threw away.
    `_audit/in_graph_rootfind.md` §1.
    """
    nested, _name, _report = mdf.nested_blocking(
        REFERENCE_IXC, REFERENCE_ICC, REFERENCE_N_EQUALITY, REFERENCE_FIGURE_OF_MERIT
    )
    # Matched as the whole distinctive sentence, not a fragment. A `match=` loose enough
    # to catch a different failure is how the claim this test replaces survived being
    # falsified upstream -- so the assertion says which refusal, in cottax's own words.
    with pytest.raises(
        ValueError,
        match=(
            r"carries no driver, so nothing answers block .* structure says it must be "
            r"driven, and `Assign` is how the algorithm is said"
        ),
    ):
        schedule_for(nested)


def test_cottax_runs_that_nesting_once_the_drivers_are_assigned():
    """...and with `Assign` done, MDF schedules: a `Drive` whose body is a `Schedule`.

    The upstream gap this module was built around is closed (`~/jaxgraph` `33af0a5`):
    `Drive.body` is a `Step`, `ConditionMap.body` is a `Step`, and `Schedule.steps`
    descends into `blocking.inner`. Exercised rather than read -- the previous claim in
    this file was read off a docstring and was stale by the time anyone looked.

    A failure here is not a regression in `mdf.py`; it is cottax having moved.
    """
    nested, name, _report = mdf.nested_blocking(
        REFERENCE_IXC, REFERENCE_ICC, REFERENCE_N_EQUALITY, REFERENCE_FIGURE_OF_MERIT
    )
    graph = nested.graph
    blocking = Blocking.scc(assign_drivers(graph, default_drivers(graph))).nest(name)
    schedule = schedule_for(blocking)
    outer = schedule.steps[blocking.index[name]]
    assert isinstance(outer, Drive)
    assert outer.problem == name
    # The body is the interior's own `Schedule`, not a `Call` over an acyclic graph --
    # which is exactly what "one iteration of the outer driver is a whole run of the
    # inner one" means, and what `MdfConditionMap` was written to fake.
    assert isinstance(outer.body, Schedule)
    assert set(outer.body.nodes) == set(blocking.blocks[blocking.index[name]]) - {name}


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
    drivers = default_drivers(problem.eager.blocking.graph)
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
    # A schedule refuses values at owned names now, so its own output cannot be fed
    # back wholesale: `restart` keeps the inputs and re-seeds every `Start` port from
    # the unknown its driver converged -- the second pass starts where the first ended.
    twice = problem.eager(mdf.restart(problem, once))
    for condition in problem.conditions:
        first = np.asarray(once[condition], dtype=float)
        second = np.asarray(twice[condition], dtype=float)
        assert np.allclose(first, second, rtol=1e-6, equal_nan=True), (
            f"{condition.path_str()} moved from {first} to {second} on a second pass, "
            f"so one pass of the schedule is not the converged MDA"
        )


def test_the_harness_distinguishes_a_cap_from_a_driver_that_gave_up():
    """`run_mdf_harness._why_it_stopped` separates the two ways "not converged" happens.

    Both were live at once and were read as one thing for three days
    (`_audit/optimise_design.md` §14): C2 was a solve stopped two-thirds of the way
    through by `MAX_ITER`, C3 is `pyvmcon` raising `QSPSolverException` at 60 and
    `VmconDriver` keeping the point. Raising the cap fixes exactly one of them, so the
    report has to name which.
    """
    # 523 measured for C2; the cap must clear it with margin, as `SAND_MAX_ITER`'s does.
    assert MAX_ITER > 523

    assert "convergence test" in _why_it_stopped(True, 523)
    assert "cap" in _why_it_stopped(False, MAX_ITER)
    assert "cap" in _why_it_stopped(False, MAX_ITER + 1)
    short = _why_it_stopped(False, 60)
    assert "stopped short" in short
    assert "NOT help" in short


class _Affine:
    """`g(x) = rate * x + offset`, element-wise on an array. A class, not a lambda, so
    the node definition stays a stable jit cache key -- `test_sand._Objective`'s reason.
    """

    def __init__(self, rate, offset):
        self.rate = tuple(rate)
        self.offset = tuple(offset)

    def __call__(self, x):
        return jnp.asarray(self.rate) * x + jnp.asarray(self.offset)

    def __eq__(self, other):
        return (
            isinstance(other, _Affine)
            and other.rate == self.rate
            and other.offset == self.offset
        )

    def __hash__(self):
        return hash((type(self), self.rate, self.offset))


def _array_fixed_point(max_iter):
    """A `FixedPoint` owning a length-three **array**, driven by Picard.

    The shape `inner_residuals` could not report on: the tokamak PF-coil ring is cut at
    `.pf_coil.n_pf_coil_turns` (a per-coil vector) and `.pf_coil.ind_pf_cs_plasma_mutual`
    (a matrix), so `^problem.times.t_plant_pulse_burn.cycle` owns arrays on every tokamak
    configuration. The three elements converge at three different rates, so which element
    is the worst is a fact and not a coincidence.
    """
    from cottax.rewrites import Assign
    from cottax.spec import CallableNode, In, NodePath, Out, VarPath
    from cottax.tools.path import path_map
    from jax.tree_util import GetAttrKey

    from functional_process.core.solver.drivers import PicardDriver

    u = VarPath((GetAttrKey("toy"), GetAttrKey("w")))
    hat = VarPath((GetAttrKey("^hat"), GetAttrKey("toy"), GetAttrKey("w")))
    problem = NodePath((GetAttrKey("P"),))
    rate, offset = (0.1, 0.5, 0.95), (1.0, 1.0, 1.0)
    graph = Graph(
        path_map([
            (
                NodePath((GetAttrKey("A"),)),
                CallableNode(
                    inputs=(In(u),), outputs=(Out(hat),), fn=_Affine(rate, offset)
                ),
            ),
            (problem, FixedPoint(inputs=(In(hat),), outputs=(Out(u),))),
        ])
    )
    graph = Assign(problem, PicardDriver(max_steps=max_iter)).apply(graph)
    (start,) = driver_vars(graph[problem], Start)
    schedule = schedule_for(graph)
    out = schedule({start: jnp.zeros(3)})
    return schedule, out, u, np.asarray(rate), np.asarray(offset)


def test_inner_residuals_reports_an_array_valued_inner_unknown():
    """One row, reduced to the worst element by relative gap -- not a `TypeError`.

    The regression guard for `_audit/optimise_design.md` §16.7's third defect:
    `float(np.asarray(env[unknown]))` raises `TypeError: only 0-dimensional arrays can be
    converted to Python scalars` on any array unknown, so the one instrument for "did the
    MDA converge" had **never run on a tokamak**. Measured against the closed-form
    residual, so a reduction that merely returned *some* element -- or the mean, or the
    norm -- fails this too.

    **The off-fixed-point env is written by hand, not manufactured by under-budgeting the
    driver.** It used to be the latter, and that stopped being possible when
    `PicardDriver` became a subclass of `cottax.drivers.PicardDriver`: that driver
    **raises** rather than returning an unconverged iterate (§31.27), which is the very
    silence this instrument was built to detect. Setting the unknown directly tests the
    same reduction against the same closed form, and is honest about the fact that the
    driver can no longer produce this state.
    """
    schedule, out, u, rate, offset = _array_fixed_point(max_iter=500)
    env = dict(out)
    # Deliberately off the fixed point, and by a *different* relative amount per element
    # so that "which element is worst" has one right answer.
    fixed_point = offset / (1.0 - rate)
    env[u] = jnp.asarray(fixed_point * np.array([1.001, 1.02, 1.5]))
    current = np.asarray(env[u], dtype=float)
    gap = rate * current + offset - current
    relative = np.abs(gap) / np.maximum(np.abs(current), 1e-30)

    (row,) = mdf.inner_residuals(schedule, env)
    _problem, unknown, residual, reported = row
    assert unknown == u
    assert isinstance(residual, float)
    assert reported == pytest.approx(relative.max())
    assert residual == pytest.approx(gap[int(np.argmax(relative))])
    assert int(np.argmax(relative)) == 2
    assert reported > 1e-6


def test_picard_refuses_to_return_an_unconverged_block():
    """The silence the test above used to have to detect, closed at the source.

    `PicardDriver` is `cottax.drivers.PicardDriver`, which runs `optx.fixed_point` with
    `throw=True`: a block that exhausts its budget **raises** instead of handing back the
    last iterate as if it had converged. The old hand-written `lax.while_loop` returned
    silently at `max_iter`, which `mdf.py`'s own docstring called out and which
    `inner_residuals` existed to catch after the fact.
    """
    import equinox

    with pytest.raises(equinox.EquinoxRuntimeError, match="maximum number of steps"):
        _array_fixed_point(max_iter=4)


def test_inner_residuals_reports_a_converged_array_block_as_converged():
    """The same block, given iterations enough -- the other half of the guard, so a
    reduction that always reports the worst *possible* number cannot pass.
    """
    schedule, out, u, rate, offset = _array_fixed_point(max_iter=500)
    (row,) = mdf.inner_residuals(schedule, out)
    assert row[3] < 1e-6
    # `PicardDriver` stops on `rtol = 1e-6` between successive *iterates*, which for a
    # contraction of rate `0.95` leaves ~`rtol / (1 - rate)` of distance to the true
    # fixed point still on the table -- the reason a converged block is checked against
    # its own residual and not against the closed form.
    assert np.asarray(out[u]) == pytest.approx(offset / (1.0 - rate), rel=1e-4)


def test_x64_is_on():
    """Everything above is float64. A harness that silently ran in float32 would compare
    a 1e-7 derivative disagreement against a 1e-7 noise floor.
    """
    assert jax.config.jax_enable_x64


# ------------------------------------------------- the root-find arm (§24.10)
#
# The two `_eval` files state `i_process_run_mode = -2`, and PROCESS answers that by
# root-finding the equalities with `scipy.optimize.fsolve` -- no objective, the
# inequalities evaluated once at the answer. These check that `root_find=True` states
# *that* problem and not an `Optimise` with the objective quietly defaulted to `7`.
# The value check -- that both files then reproduce PROCESS's own `fsolve` answer -- is a
# `run_cold_matrix.py` row, because it needs a real PROCESS run.


@pytest.fixture(scope="module")
def square_problem():
    """A `RootFind` assembly: the reference `icc` truncated to a square subproblem.

    The reference file is not itself an evaluation-mode run, so this states a square
    problem out of it -- two equalities against two iteration variables -- rather than
    reading one of the `_eval` files, which would need `machine_from_indat` and a
    machine graph for a purely structural check.
    """
    ixc = REFERENCE_IXC[:REFERENCE_N_EQUALITY]
    return mdf.assemble(
        ixc,
        REFERENCE_ICC,
        REFERENCE_N_EQUALITY,
        REFERENCE_FIGURE_OF_MERIT,
        root_find=True,
    )


def test_a_root_find_drives_the_equalities_and_nothing_else(square_problem):
    """One condition per design variable, and every one of them an equality.

    This is the whole difference from the `Optimise` arm and it is what PROCESS's
    `_Fsolve.evaluate_eq_cons` does: `fcnvmc1(n, self.meq, x, 0)` returns the first
    `meq` constraints and stops.
    """
    assert issubclass(square_problem.problem_type, RootFind)
    assert len(square_problem.conditions) == len(square_problem.design)
    assert square_problem.conditions == tuple(square_problem.report["equalities"])
    assert square_problem.n_inequality == 0


def test_a_root_find_forms_no_objective_at_all(square_problem):
    """Not "an objective nobody reads" -- no objective node in the graph.

    `_Fsolve.solve` ends `self.objf = None` and the output writer omits the
    figure-of-merit line entirely. A node computing `numerics.py:154`'s default of `7`
    would be inventing a quantity PROCESS never forms, which is exactly the paper-over
    §24.10 was opened to remove.
    """
    assert square_problem.report["objective"] is None
    assert not any(
        v.path_str() == "^cond.numerics.objf" for v in square_problem.conditions
    )
    assert "^cond.numerics.objf" not in {
        v.path_str() for v in square_problem.graph.variables
    }


def test_the_inequalities_survive_for_reporting_but_are_not_driven(square_problem):
    """PROCESS evaluates all `m` constraints once at the root, so the port must be able
    to as well -- `min ie` on the cold matrix is that evaluation.

    They are in `reported` and in the graph, and in neither `conditions` nor the count
    the driver reads its split from.
    """
    assert square_problem.reported == tuple(square_problem.report["inequalities"])
    assert square_problem.reported
    assert not set(square_problem.reported) & set(square_problem.conditions)
    assert set(square_problem.reported) <= set(square_problem.graph.variables)


def test_the_condition_roles_are_residuals_not_an_objective_and_bounds(square_problem):
    """`RootFind.condition_roles` is `(Residual,) * n`, and the map must say so.

    The roles travel on the driver seam (`_audit/optimise_design.md` §8), so a map that
    still claimed `(Objective, Equality, ...)` would hand a root find to a driver as an
    optimisation with a constraint standing in for the objective.
    """
    env = dict.fromkeys(square_problem.eager.inputs, jnp.asarray(1.0))
    conditions = mdf.condition_map(square_problem, env)
    assert conditions.roles == (Residual,) * len(square_problem.conditions)


def test_a_non_square_problem_is_refused_rather_than_root_found():
    """Squareness is a *consequence* of PROCESS's evaluation mode, not the test for it,
    so a caller that asks for a root find over a non-square problem is told so.

    `fsolve` on `n` unknowns and `meq != n` residuals is not a problem PROCESS could
    answer either; failing at assembly beats failing inside optimistix's linear solve.
    """
    with pytest.raises(ValueError, match="one equality per iteration variable"):
        mdf.assemble(
            REFERENCE_IXC,
            REFERENCE_ICC,
            REFERENCE_N_EQUALITY,
            REFERENCE_FIGURE_OF_MERIT,
            root_find=True,
        )


def test_the_optimise_arm_is_untouched(problem, square_problem):
    """The two arms differ in the problem, not in the graph underneath it.

    Everything structural about the MDA -- how many blocks, how many of them driven --
    is the same object built the same way; only the outer problem changed. If this ever
    fails, `root_find` has started changing the model rather than the question asked of
    it.
    """
    assert issubclass(problem.problem_type, Optimise)
    assert problem.reported == ()
    assert problem.report["objective"] is not None
    # Exactly one block apart, and the one is the objective node: it is an ordinary
    # `CallableNode` in the graph, so an assembly that mints none has one block fewer.
    # Nothing about the *MDA* moved -- the driven-block count is identical.
    assert problem.report["blocks"] == square_problem.report["blocks"] + 1
    assert problem.report["driven_blocks"] == square_problem.report["driven_blocks"]


def test_the_root_find_driver_refuses_an_optimise_and_needs_a_start(
    problem, square_problem
):
    """`mdf.solve` picks the driver off `problem_type`, so the mismatch is caught at the
    one place a caller could get it wrong by hand.
    """
    with pytest.raises(TypeError, match="states an Optimise"):
        mdf.root_find_driver(problem)
    driver = mdf.root_find_driver(square_problem)
    assert driver.drives is RootFind
    with pytest.raises(ValueError, match="needs a starting value"):
        driver(mdf.condition_map(square_problem, {}), {})


# ------------------------------------- the root find stated *inside* the graph
#
# `assemble` + `solve` keeps the problem outside the graph: the outer driver is called by
# hand and `MdfConditionMap` re-runs the whole MDA per residual. `in_graph_root_find`
# states it as a node instead and lets `Blocking.scc` decide what it drives. These check
# the structural half -- what lands in the block, and that the nesting is real. The value
# half (both `_eval` files still reproduce PROCESS's own `fsolve` x) is the `tier4` test
# at the bottom, because it needs a PROCESS run.


@pytest.fixture(scope="module")
def in_graph(square_problem):
    """`square_problem` with its `RootFind` inserted, driven, blocked and nested."""
    return mdf.in_graph_root_find(square_problem)


def test_the_problem_is_a_node_of_the_graph_and_owns_the_design(in_graph):
    """The design variables stop being boundary inputs and become owned.

    That is the whole formulation in one assertion: an `ixc` entry is no longer something
    a caller supplies per residual evaluation, it is something a node of the graph
    produces -- so `Blocking` can see the cycle it closes, and `Graph.owners` answers
    "who determines this" for the design vector the way it does for everything else.
    """
    node = in_graph.graph[in_graph.problem]
    # `unknowns_of`, not `owns`: the node also owns what its driver reports
    # (`^driver_out.*.RootFind`), which is not something the root find solves for.
    assert set(unknowns_of(node)) == set(in_graph.design)
    assert set(node.owns) & set(in_graph.mdf.eager.inputs) == set(in_graph.design)
    assert set(in_graph.design) & set(in_graph.schedule.inputs) == set()
    # Its conditions are PROCESS's equalities and nothing else -- driver data
    # (`^guess.*`, minted by `Assign`) is a read of the graph, not a condition.
    assert conditions_of(node) == in_graph.conditions


def test_the_blocking_drives_the_cycle_and_not_the_whole_graph(in_graph):
    """The driven block is a minority of the graph, and the rest is ordinary steps.

    The measured claim this formulation exists for. With the problem outside the graph a
    residual evaluation runs every node; with it inside, `Blocking.scc` puts in the block
    exactly what is coupled to it, everything upstream runs once before and everything
    downstream once after.

    **The bound is loose on purpose, because the fraction is a property of the run and
    not of the formulation** [measured, `_audit/in_graph_rootfind.md` §2]: 61 of 169 here
    (this fixture root-finds the reference stellarator's first two `ixc`, which reach
    deep into the coil set), against 39 of 260 on `spherical_tokamak_eval` and 35 of 272
    on `large_tokamak_eval` -- the two files that actually state this problem. A tighter
    assertion would be pinning one configuration's reachability and calling it a law.
    """
    assert len(in_graph.block) < len(in_graph.graph.nodes) / 2
    assert in_graph.index > 0, "nothing upstream of the loop -- wrong graph"
    assert in_graph.index < len(in_graph.blocking.blocks) - 1, "nothing downstream"


def test_the_driven_block_is_exactly_what_reachability_predicts(in_graph):
    """`block == descendants(readers of design) & ancestors(owners of conditions)`, plus
    the problem itself.

    Not a restatement of `Blocking.scc`: this is the *independent* prediction, computed
    from `Graph.descendants`/`ancestors` on the graph **without** the problem in it, and
    it is what says the SCC the problem closes is the loop and nothing more. If the block
    is ever bigger than this, some node reads a design variable that nothing predicted --
    a more interesting result than agreement, so the assertion is two-sided.
    """
    graph = in_graph.mdf.graph
    entry = {n for d in in_graph.design for n in graph.readers.get(d, ())}
    exit_ = {graph.owners[c] for c in in_graph.conditions}
    loop = set(graph.descendants(entry)) & set(graph.ancestors(exit_))
    assert set(in_graph.block) == loop | {in_graph.problem}


def test_the_interior_is_the_block_one_problem_in_and_holds_its_coupled_blocks(in_graph):
    """`Blocking.inner` at the root find is the block minus the root find, blocked again.

    This is the part that exercises `Blocking.inner` for the first time in this tree. The
    interior is not a partition of the block -- the outer problem is in none of its
    sub-blocks -- and stating it is what says which of the block's problems is outer.
    """
    interior = in_graph.interior
    assert interior is not None
    assert in_graph.problem not in interior.index
    assert set(interior.graph.nodes) == set(in_graph.block) - {in_graph.problem}
    # The coupled blocks that fall inside the loop are driven at the inner level, so one
    # outer Newton step is one run of the interior's schedule.
    assert sum(1 for t in interior.problem_types if t is not None) >= 1
    assert len(interior.blocks) > 1


def test_the_outer_drive_body_is_the_interiors_own_schedule(in_graph):
    """`Drive.body` is a `Schedule`, not a `Call` -- which is what nesting *is*.

    `Call(sub.runnable)` is the flat case and would be a silent wrong answer here: the
    block holds driven sub-blocks, and running them as one acyclic body would evaluate a
    fixed point once instead of converging it.
    """
    outer = in_graph.drive
    assert isinstance(outer, Drive)
    assert isinstance(outer.body, Schedule)
    assert issubclass(outer.problem_type, RootFind)
    assert outer.unknowns == in_graph.design
    assert outer.conditions == in_graph.conditions
    assert set(outer.body.nodes) == set(in_graph.block) - {in_graph.problem}


def test_the_design_values_arrive_at_the_start_ports(in_graph):
    """A design variable is owned now, so its value is driver data, not an input.

    `Assign` mints one `^guess.<place>` per unknown and `in_graph_inputs` fills it from
    the unknown's own name in the seed env -- the same rule `seed`/`prime` already apply
    to every inner unknown. Getting this wrong is silent: the ports fall to `0.0` and the
    outer Newton starts from the cold point rather than from the input file's `ixc`.
    """
    node = in_graph.graph[in_graph.problem]
    guesses = driver_vars(node, Start)
    assert len(guesses) == len(in_graph.design)
    assert set(guesses) <= set(in_graph.schedule.inputs)
    env = {v: jnp.asarray(3.0) for v in in_graph.mdf.eager.inputs}
    env |= {v: jnp.asarray(0.0) for v in in_graph.mdf.eager.unknowns}
    inputs = mdf.in_graph_inputs(in_graph, env)
    assert all(float(inputs[g]) == pytest.approx(3.0) for g in guesses)


def test_an_optimise_is_not_stated_in_graph_here(problem):
    """The `Optimise` arm nests just as well but its driver does not trace, so it is
    refused rather than silently answered by a `SeededNewtonDriver`.

    `default_drivers` would happily put a `VmconDriver` on an `Optimise`; what it cannot
    do is make the resulting schedule traceable, and `in_graph_solve` would then be a
    different measurement wearing the same name.
    """
    with pytest.raises(
        TypeError,
        match=re.escape(
            "this MDF states an Optimise, and only the `RootFind` arm is stated in-graph"
        ),
    ):
        mdf.in_graph_root_find(problem)


def test_a_place_already_bound_is_refused(square_problem):
    """A node name is a binding, so binding twice is a caller error and says so."""
    taken = next(iter(square_problem.graph.nodes))
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"{taken!r} is already a node of this graph -- pass `place` to bind the "
            f"root find somewhere else"
        ),
    ):
        mdf.in_graph_root_find(square_problem, place=taken)


def test_the_verdict_and_the_whole_schedule_jit_are_not_alternatives(
    square_problem,
):
    """The verdict is reported **and** the schedule still hashes -- by construction now.

    This used to be a trade: `MdfNewtonDriver.outcome` was a plain `dict`, a driver is a
    field of the graph, `Schedule.__hash__` reaches it through `Graph.__hash__`, and
    `sand_harness.run_schedule` keys its whole-jit verdict on the schedule -- so asking
    for the step count cost 856/993 XLA compiles instead of 5, for the same answer
    (`_audit/in_graph_rootfind.md` §6). `Outcome` -- a `dict` hashed by identity --
    closed it by making the sink hashable; `DriverOut` closes it by there being no sink:
    the verdict is a value the driver returns and a name the node owns, so nothing
    mutable is a driver field at all.
    """
    built = mdf.in_graph_root_find(square_problem)
    assert hash(built.schedule)
    assert built.steps({}) is None, "nothing has run yet, so the env holds no verdict"


def test_the_driver_reports_its_verdict_through_ports_the_node_owns(square_problem):
    """`Assign` mints one name per reported kind, and the problem node owns it.

    The structural half of the change: a diagnostic is a name in the graph, so `owners`
    reports it, `prune` could ask for it, and a `Schedule` refuses an incoming value at
    it -- none of which a mutable field on a frozen driver could offer.
    """
    built = mdf.in_graph_root_find(square_problem)
    node = built.graph[built.problem]
    assert [p.var.path_str() for p in node.driver_out] == [
        "^driver_out.steps.RootFind",
        "^driver_out.converged.RootFind",
        "^driver_out.status.RootFind",
    ]
    for kind in (Steps, Converged, mdf.Status):
        assert built.graph.owners[kind.name_for(built.problem)] == built.problem


def test_the_unknowns_are_still_only_the_design(square_problem):
    """`owns` grew by the reports; what the root find *solves for* did not.

    The one assertion the design predicted would move, and the reason `mda.starts_for`
    had to stop asking `owns`: a `Start` pairs with the unknowns, and a step count is
    not one.
    """
    node = built = mdf.in_graph_root_find(square_problem)
    node = built.graph[built.problem]
    assert unknowns_of(node) == built.design
    assert set(node.owns) - set(unknowns_of(node)) == {
        Steps.name_for(built.problem),
        Converged.name_for(built.problem),
        mdf.Status.name_for(built.problem),
    }


def test_two_runs_of_one_assembly_report_their_own_verdicts(square_problem):
    """A verdict is a question about a *run*, not a field on the assembly.

    Under `Outcome` the sink lived on the driver, so a second run of one assembly
    overwrote the first run's step count and there was no way to hold both. Reading the
    env makes that unrepresentable.
    """
    built = mdf.in_graph_root_find(square_problem)
    first = {Steps.name_for(built.problem): 3}
    second = {Steps.name_for(built.problem): 7}
    assert built.steps(first) == 3
    assert built.steps(second) == 7


# ------------------------------------------------- the value check, on the real files


EVALUATION_FILES = ["large_tokamak_eval", "spherical_tokamak_eval"]
"""The two configurations whose `i_process_run_mode` is `-2`. `run_cold_matrix.py` finds
them by reading the mode; they are named here so this test is one parametrisation and not
a survey."""

WORST_DX = 1e-8
"""What §24.10's table measured for these two files (`3.29e-12` and `3.64e-09`), rounded
up to one bound. A restructuring that moves the answer past this is a bug until shown
otherwise -- and `_audit/optimise_design.md` §19--§20's finding, that a last-bit
perturbation can move an iterative solve's trajectory, is why the *tighter* check in this
test is the one against the outer-driver answer rather than against PROCESS."""


@pytest.mark.tier4
@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            n,
            marks=pytest.mark.xfail(
                n == "large_tokamak_eval",
                strict=True,
                reason=(
                    "Newly SURFACED, not newly caused (`_audit/optimise_design.md` "
                    "§31.27): the `PicardDriver` over "
                    "(ind_pf_cs_plasma_mutual, n_pf_coil_turns, t_plant_pulse_burn) "
                    "raises `Nonfinite (inf or nan) values detected during solve`. The "
                    "old hand-written `lax.while_loop` returned that iterate silently -- "
                    "its convergence test is `all(|next - cur| <= tol)`, which a `nan` "
                    "fails, so it simply ran to `max_iter` and handed the `nan` back. "
                    "Whether the block was always non-finite or optimistix visits a "
                    "point the hand-rolled loop never did is UNRESOLVED and is the same "
                    "question as the three tokamak `no-step` arms."
                ),
            ),
        )
        for n in EVALUATION_FILES
    ],
)
def test_the_in_graph_root_find_gives_the_same_answer(name):
    """Both `_eval` files reproduce PROCESS's own `fsolve` x, and the two formulations
    agree with each other to roundoff.

    Two comparisons, and they answer different questions:

    - **against PROCESS** (`reference.converged`) says the port still solves the problem
      PROCESS states, to §24.10's measured `3.29e-12` / `3.64e-09`.
    - **against the outer driver**, at the same primed env, says the *restructuring*
      moved nothing -- which is the stronger claim, since both arms share every model,
      every inner driver and every seed and differ only in where the problem is stated.
      Measured at `~1e-15`, i.e. XLA reassociation and nothing else.

    Not marked `tier4` for the MDA's sake but for PROCESS's: `reference_run` is disk
    cached, and a cold cache costs one ~95 s solve per file.
    """
    from functional_process.indat import (  # noqa: PLC0415 -- module import is ~20 s
        graph_for,
        machine_from_indat,
        switch_values_from_indat,
    )
    from functional_process.sand_harness import (  # noqa: PLC0415
        reference_run,
        schedule_verdict,
    )

    path = f"tests/regression/input_files/{name}.IN.DAT"
    reference = reference_run(path)
    assembled = mdf.assemble(
        reference.ixc,
        reference.icc,
        reference.n_equality,
        reference.i_figure_merit,
        graph=graph_for(machine_from_indat(path)),
        switch_values=switch_values_from_indat(path),
        root_find=True,
    )
    env, _out = mdf.prime(assembled, mdf.seed(assembled, reference.cold))

    built = mdf.in_graph_root_find(assembled)
    x, _at, _seconds = mdf.in_graph_solve(built, env)
    got = [float(np.asarray(v)) for v in x]
    # The closed trade, end to end: the whole run is **one** XLA program *and* the
    # driver's verdict came back -- out of the env, as data, with no callback and no
    # sink. Either alone used to be available and not both
    # (`_audit/in_graph_rootfind.md` §6), and the step count here is a traced `int64[]`
    # that survived the compile.
    whole, reason = schedule_verdict(built.schedule)
    assert whole, f"the single jit was refused: {reason}"
    assert built.steps(_at) is not None
    assert int(np.asarray(built.steps(_at))) > 0
    assert bool(np.asarray(built.successful(_at)))
    against_process = [
        abs(v - reference.converged[i]) / abs(reference.converged[i])
        for i, v in zip(reference.ixc, got, strict=True)
    ]
    assert max(against_process) < WORST_DX, dict(
        zip(reference.ixc, against_process, strict=True)
    )

    outer, _out, _seconds = mdf.solve(
        assembled, env, optimiser=mdf.root_find_driver(assembled)
    )
    against_outer = [
        abs(v - float(np.asarray(w))) / abs(float(np.asarray(w)))
        for v, w in zip(got, outer, strict=True)
    ]
    assert max(against_outer) < 1e-12, dict(
        zip(reference.ixc, against_outer, strict=True)
    )
