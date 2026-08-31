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
from cottax.graph import Graph
from cottax.problem import (
    FixedPoint,
    Optimise,
    Residual,
    RootFind,
    Start,
    driver_vars,
)

from functional_process import mdf, sand
from functional_process.core.solver.drivers import SeededNewtonDriver
from functional_process.mda import default_drivers
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
    drivers = dict(default_drivers(problem.eager.blocking.graph))
    drivers[name] = mdf.driver(problem)
    with pytest.raises(ValueError, match=r"declares|problem"):
        schedule_for(nested)


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
    graph = Assign(problem, PicardDriver(max_iter=max_iter)).apply(graph)
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
    residual of a deliberately unconverged Picard, so a reduction that merely returned
    *some* element -- or the mean, or the norm -- fails this too.
    """
    schedule, out, u, rate, offset = _array_fixed_point(max_iter=4)
    current = np.asarray(out[u], dtype=float)
    gap = rate * current + offset - current
    relative = np.abs(gap) / np.maximum(np.abs(current), 1e-30)

    (row,) = mdf.inner_residuals(schedule, out)
    _problem, unknown, residual, reported = row
    assert unknown == u
    assert isinstance(residual, float)
    assert reported == pytest.approx(relative.max())
    assert residual == pytest.approx(gap[int(np.argmax(relative))])
    # The slowest element (rate 0.95) is the one still moving after four iterations, and
    # the check has to be able to say so: an unconverged block reported as converged is
    # exactly the silence `PicardDriver`'s `lax.while_loop` cannot break.
    assert int(np.argmax(relative)) == 2
    assert reported > 1e-6


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
