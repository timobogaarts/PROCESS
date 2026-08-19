"""Assembly-level tests for `functional_process.sand` and `VmconDriver`.

Deliberately **not** a value comparison against PROCESS: that is `sand_harness.py`'s
ladder, which needs a live 95-second PROCESS run and belongs in `run_sand_harness.py`.
What is checked here is everything that can be checked without one -- that the problem
assembled is the problem PROCESS states (which constraints, which are equalities, which
variables are the unknowns), that the driver's contract holds on a problem with a known
answer, and that the two places this layer can silently be wrong (a dropped constraint,
a mislabelled switch argument) fail loudly instead.
"""

import inspect
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cottax.evaluate import Drive
from cottax.graph import Graph
from cottax.problem import Optimise
from cottax.spec import CallableNode, In, NodePath, Out, VarPath
from cottax.tools.path import path_map
from jax.tree_util import GetAttrKey

from functional_process.core.solver import constraints as ported_constraints
from functional_process.core.solver.drivers import VmconDriver
from functional_process.sand import (
    REFERENCE_SWITCH_VALUES,
    constraint_nodes,
    design_bounds,
    iteration_variable_path,
    objective_node,
    optimise_graph,
    sand_graph,
    sand_shape,
    sand_schedule,
)
from functional_process.total_process import GRAPH
from process.core.solver.iteration_variables import ITERATION_VARIABLES

REFERENCE_IXC = [2, 3, 4, 6, 10, 56, 59, 109]
REFERENCE_ICC = [2, 16, 24, 8, 17, 18, 67, 82, 83, 62, 32, 34, 35, 65]
REFERENCE_N_EQUALITY = 2
REFERENCE_FIGURE_OF_MERIT = 6
"""`stellarator_helias.IN.DAT`'s own problem. Transcribed rather than read off a live
run so these tests need no PROCESS execution; `run_sand_harness.py` reads the real
`numerics` and would disagree loudly if the file ever changed."""


def test_reference_problem_matches_the_input_file():
    """The four constants above really are what the reference `IN.DAT` asks for.

    Same discipline as `test_configuration.py::
    test_reference_configuration_matches_the_input_file`: a transcription that nothing
    checks is a transcription that drifts. `ixc`/`icc` are not parsed (they are
    multi-line lists with per-entry comments); the three scalars that decide the
    *shape* of the problem are.
    """
    text = (
        Path(__file__).resolve().parent.parent
        / "tests/regression/input_files/stellarator_helias.IN.DAT"
    ).read_text()
    settings = {
        m.group(1): int(m.group(2))
        for line in text.splitlines()
        if (m := re.match(r"\s*([A-Za-z_]\w*)\s*=\s*(-?\d+)\s*(\*.*)?$", line))
    }
    assert settings["n_equality_constraints"] == REFERENCE_N_EQUALITY
    assert settings["i_figure_merit"] == REFERENCE_FIGURE_OF_MERIT
    # `n_inequality_constraints` is *derived*, not set: `init.set_active_constraints`
    # takes it as `len(icc) - n_equality_constraints`
    # (`process/core/init.py:1277-1294`), which is why the split is positional in the
    # first place.
    assert len(REFERENCE_ICC) > REFERENCE_N_EQUALITY


def _contract_static_argnames():
    """`{constraint id: static_argnames}` read out of `test_constraints.py`'s own
    `Tier1Contract` classes -- the maintained source of truth for which parameter of a
    ported constraint is a switch."""
    source = (
        Path(__file__).resolve().parent / "core/solver/test_constraints.py"
    ).read_text()
    out = {}
    for block in source.split("\nclass "):
        ported = re.search(r"ported = constraint_(\d+)", block)
        if not ported:
            continue
        declared = re.search(r"static_argnames = \(([^)]*)\)", block)
        names = (
            tuple(
                part.strip().strip("\"'")
                for part in declared.group(1).split(",")
                if part.strip()
            )
            if declared
            else ()
        )
        out[int(ported.group(1))] = names
    return out


def test_switch_values_match_the_contracts_static_argnames():
    """`REFERENCE_SWITCH_VALUES` decides, mechanically, which parameters are frozen at
    assembly time. Both directions matter and both are checked.

    A switch that leaked onto a port would be traced into an `if` (a `TracerBoolConversion`
    at best, a silently-wrong branch at worst); a real variable frozen into the partial
    would be a constant where a derivative belongs, which is exactly the defect class
    Stage B exists to find. Cross-checking against the `Tier1Contract`s means the two
    declarations cannot drift apart silently.
    """
    contracts = _contract_static_argnames()
    for cid in REFERENCE_ICC:
        fn = getattr(ported_constraints, f"constraint_{cid}")
        parameters = set(inspect.signature(fn).parameters)
        assert parameters & set(REFERENCE_SWITCH_VALUES) == set(contracts[cid]), (
            f"constraint {cid}: sand would bind "
            f"{sorted(parameters & set(REFERENCE_SWITCH_VALUES))} statically, but its "
            f"Tier1Contract declares static_argnames={contracts[cid]}"
        )


def test_iteration_variable_path_is_processs_own_accessor():
    """`ITERATION_VARIABLES[id]`'s `(module, target_name or name)` is the `VarPath`."""
    for i in REFERENCE_IXC:
        iteration_variable = ITERATION_VARIABLES[i]
        assert iteration_variable_path(i) == VarPath((
            GetAttrKey(iteration_variable.module),
            GetAttrKey(iteration_variable.target_name or iteration_variable.name),
        ))


def test_array_indexed_iteration_variables_are_refused_not_approximated():
    """IDs 125-136 address one element of `f_nd_impurity_electron_array`.

    A `SequenceKey` resolves for a *read*, so this could easily be made to look like it
    works. It does not: `Graph.__check_init__` refuses an output lying inside a variable
    another node reads whole, and every consumer of impurity fractions reads the whole
    array. Refusing here keeps that fact visible instead of deferring it to a confusing
    graph-construction error. See `_audit/optimise_design.md` §1.3.
    """
    with pytest.raises(NotImplementedError, match="array"):
        iteration_variable_path(125)


def test_all_eight_design_variables_are_boundary_inputs():
    """An `Optimise` owning them therefore collides with nothing.

    This is what makes the whole layer cheap: registering the problem turns eight free
    inputs into eight owned variables and changes nothing else. Seven of the 83-entry
    table *are* produced by a node (ID 1 `.physics.aspect` among them) and would be a
    duplicate-ownership conflict; none of those is active in this run.
    """
    owned = set(GRAPH.owners)
    for i in REFERENCE_IXC:
        assert iteration_variable_path(i) not in owned


def test_constraint_16_is_an_equality_despite_its_geq_body():
    """The equality/inequality split is **positional and user-chosen**, not a property
    of the constraint function.

    `stellarator_helias.IN.DAT:12` sets `n_equality_constraints = 2`, so `icc[:2]` --
    constraints 2 and 16 -- are the equalities. Constraint 16's ported body is a `geq`
    ("net electric power lower limit") and PROCESS still drives it to equality. Anyone
    "fixing" `constraint_nodes` to read the body would silently change the problem.
    """
    nodes, equalities, inequalities, omitted = constraint_nodes(
        GRAPH, REFERENCE_ICC, REFERENCE_N_EQUALITY
    )
    assert omitted == {}
    assert len(nodes) == len(REFERENCE_ICC)
    assert [v.path_str() for v in equalities] == [
        "^cond.constraints.c2",
        "^cond.constraints.c16",
    ]
    assert len(inequalities) == len(REFERENCE_ICC) - REFERENCE_N_EQUALITY


def test_an_unassemblable_constraint_raises_rather_than_being_dropped():
    """Silently solving 13 of PROCESS's 14 constraints is a different problem.

    Constraint 76 is the standing instance: its `f_nd_impurity_electrons` argument is
    an *array element* (`_audit/optimise_design.md` §3), which name-based resolution
    cannot reach. It is asserted to be the only unassemblable one of the ~82 ported
    constraints, so this test also records that fact rather than merely using it.
    """
    unassemblable = sorted(
        cid
        for name in dir(ported_constraints)
        if name.startswith("constraint_")
        and (cid := int(name.removeprefix("constraint_")))
        and _cannot_resolve(cid)
    )
    assert unassemblable == [76]
    unresolvable = unassemblable[0]
    with pytest.raises(ValueError, match="cannot be assembled"):
        constraint_nodes(GRAPH, [*REFERENCE_ICC, unresolvable], REFERENCE_N_EQUALITY)
    _nodes, _eq, _ineq, omitted = constraint_nodes(
        GRAPH,
        [*REFERENCE_ICC, unresolvable],
        REFERENCE_N_EQUALITY,
        omit={unresolvable},
    )
    assert unresolvable in omitted


def _cannot_resolve(cid):
    try:
        constraint_nodes(GRAPH, [cid], 0)
    except ValueError:
        return True
    return False


def test_objective_node_folds_in_the_sign_and_owns_one_minted_variable():
    """`Optimise` minimises, so a negative `i_figure_merit` (PROCESS's "maximise") has
    to become a negated body -- not a flag on the driver, which would make its `drives`
    claim a lie, and not a field on `Optimise`, which has none."""
    _name, node, var = objective_node(GRAPH, REFERENCE_FIGURE_OF_MERIT)
    assert var.path_str() == "^cond.numerics.objf"
    assert len(node.outputs) == 1
    _name, negated, _var = objective_node(GRAPH, -REFERENCE_FIGURE_OF_MERIT)
    coe = 121.5
    assert float(node.fn(coe)) == pytest.approx(-float(negated.fn(coe)))


def test_sand_assembles_and_orders_its_conditions():
    """The whole assembly, and the ordering `VmconDriver`'s count-based split rests on.

    `Drive.conditions` is the problem node's `reads` and `Optimise.inputs` is
    `(objective, *equalities, *inequalities)`, so the objective is first and the
    equalities precede the inequalities. That is *checked* here rather than assumed,
    because it is the one thing standing between a correct solve and one that quietly
    treats an inequality as the objective.
    """
    from functional_process.mda import driven_graph
    from functional_process.mda_harness import _without_excluded

    graph = driven_graph(_without_excluded(GRAPH))
    with_problem, _name, report = optimise_graph(
        graph,
        REFERENCE_IXC,
        REFERENCE_ICC,
        REFERENCE_N_EQUALITY,
        REFERENCE_FIGURE_OF_MERIT,
    )
    combined, _residualised = sand_graph(with_problem)
    schedule = sand_schedule(combined, None)
    shape = sand_shape(schedule)
    drive = shape["drive"]

    names = [c.path_str() for c in drive.conditions]
    assert names[0] == "^cond.numerics.objf"
    definition = drive.subgraph[drive.problem]
    assert names[1 : 1 + len(definition.equalities)] == [
        c.var.path_str() for c in definition.equalities
    ]
    assert names[1 + len(definition.equalities) :] == [
        c.var.path_str() for c in definition.inequalities
    ]
    # The eight design variables come first among the unknowns, so the Schur reduction
    # in `sand_harness` can index them positionally.
    assert [v.path_str() for v in drive.unknowns[: len(REFERENCE_IXC)]] == [
        iteration_variable_path(i).path_str() for i in REFERENCE_IXC
    ]
    # Not the whole graph in one block: the acyclic remainder still runs as `Call` steps.
    assert shape["drive_nodes"] < len(combined.nodes)
    assert shape["schedule_steps"] > 1


def test_default_drivers_reads_the_split_off_the_problem_node():
    """`mda.default_drivers` never counts conditions -- it asks the `Optimise`."""
    from cottax.blocking import Blocking

    from functional_process.mda import default_drivers, driven_graph
    from functional_process.mda_harness import _without_excluded

    graph = driven_graph(_without_excluded(GRAPH))
    with_problem, _name, _report = optimise_graph(
        graph,
        REFERENCE_IXC,
        REFERENCE_ICC,
        REFERENCE_N_EQUALITY,
        REFERENCE_FIGURE_OF_MERIT,
    )
    combined, _residualised = sand_graph(with_problem)
    blocking = Blocking.scc(combined)
    drivers = default_drivers(blocking)
    optimise = [d for d in drivers.values() if isinstance(d, VmconDriver)]
    assert len(optimise) == 1
    driver = optimise[0]
    problem = next(
        p
        for p, t in zip(blocking.problems, blocking.problem_types, strict=True)
        if t is not None and issubclass(t, Optimise)
    )
    definition = blocking.graph[problem]
    assert driver.n_equality == len(definition.equalities)
    assert driver.n_inequality == len(definition.inequalities)


def test_design_bounds_are_processs_own_table():
    for var, lower, upper in design_bounds(REFERENCE_IXC):
        assert lower < upper
        assert var in {iteration_variable_path(i) for i in REFERENCE_IXC}


class _Conditions:
    """The two attributes `residual_condition_scales` reads off a `Drive`.

    A stub rather than a real `Drive`: the function is pure name arithmetic over
    `unknowns`/`conditions` plus a lookup in `env`, so assembling a 130-node graph to
    exercise it would test the assembly, not the rule.
    """

    def __init__(self, unknowns, conditions):
        self.unknowns = unknowns
        self.conditions = conditions


def test_a_residual_whose_unknown_has_no_scale_keeps_a_factor_of_one():
    """`1/|u|` degrades to **`1.0`**, never to `1/floor`.

    Regression pin for a measured defect, not a hypothetical: `.power.qac` is
    identically zero on the reference run (no PF energy swing to dissipate), so
    `1 / max(|u|, 1e-12)` weighted its residual row -- the trivial equality `u = 0`,
    whose Jacobian row is exactly `-1` in one column -- by `1e12`, nine orders of
    magnitude above every other row in the problem. The condition number of the Jacobian
    `VmconDriver` hands VMCON was `6.7e12` against `2.1e4` without it, and Stage C2 went
    from 62 SQP iterations to 73 -- and, on the tree state where this was first seen, to
    `max_iter` without converging at all. `1.0` is what `VmconDriver.scaled` already
    degrades to for a design variable starting at `0.0`, so the two scalings agree.
    """
    from cottax.tools.minting import prefix_path

    from functional_process.sand import COND, residual_condition_scales

    zero = VarPath((GetAttrKey("power"), GetAttrKey("qac")))
    real = VarPath((GetAttrKey("power"), GetAttrKey("qss")))
    constraint = prefix_path(
        VarPath((GetAttrKey("constraints"), GetAttrKey("c2"))), COND
    )
    # A residualised `FixedPoint`'s condition is `^cond^cond.X` against the unknown `.X`
    # -- two mints, which `residual_condition_scales` strips to pair them.
    residual = {v: prefix_path(prefix_path(v, COND), COND) for v in (zero, real)}
    drive = _Conditions(
        unknowns=(zero, real),
        conditions=(constraint, residual[zero], residual[real]),
    )
    scales = dict(residual_condition_scales(drive, {zero: 0.0, real: 4.0}, floor=1e-12))

    # Exact comparison is the point: both factors are computed by a single division of
    # exactly representable values, and the whole defect was a factor being *silently*
    # something other than the one the rule names.
    assert scales[residual[zero]] == 1.0  # noqa: RUF069
    assert scales[residual[real]] == 0.25  # noqa: RUF069
    # PROCESS's own constraints are never touched -- that is what keeps the iterates
    # comparable with PROCESS's own.
    assert constraint not in scales


# ---------------------------------------------------------------- the driver


def _toy_problem():
    """`min (x-3)^2 + (y-2)^2` subject to `x + y - 4 <= 0`, as a cottax graph.

    Written out rather than reusing the real graph so the answer is known in closed
    form: the unconstrained minimum `(3, 2)` violates the constraint, so it is active and
    the solution is the projection `(2.5, 1.5)` with objective `0.5`. That makes this a
    real test of the **sign convention** -- with the inequality passed unflipped, VMCON
    would read the feasible set as `x + y >= 4` and stop at `(3, 2)`.
    """
    x = VarPath((GetAttrKey("d"), GetAttrKey("x")))
    y = VarPath((GetAttrKey("d"), GetAttrKey("y")))
    f = VarPath((GetAttrKey("c"), GetAttrKey("f")))
    g = VarPath((GetAttrKey("c"), GetAttrKey("g")))
    graph = Graph(
        path_map([
            (
                NodePath((GetAttrKey("F"),)),
                CallableNode(
                    inputs=(In(x), In(y)),
                    outputs=(Out(f),),
                    fn=_Objective(),
                ),
            ),
            (
                NodePath((GetAttrKey("G"),)),
                CallableNode(
                    inputs=(In(x), In(y)),
                    outputs=(Out(g),),
                    fn=_Constraint(),
                ),
            ),
            (
                NodePath((GetAttrKey("Opt"),)),
                Optimise(
                    objective=In(f),
                    design=(Out(x), Out(y)),
                    inequalities=(In(g),),
                ),
            ),
        ])
    )
    return graph, x, y


class _Objective(jax.tree_util.register_static and object):
    """`(x-3)^2 + (y-2)^2`. A class, not a lambda, so the node definition stays a stable
    jit cache key -- the same reason `cottax.rewrites.Compare` uses `Pairwise`."""

    def __call__(self, x, y):
        return (x - 3.0) ** 2 + (y - 2.0) ** 2

    def __eq__(self, other):
        return isinstance(other, _Objective)

    def __hash__(self):
        return hash(type(self))


class _Constraint:
    """`x + y - 4`, in cottax's `g <= 0` convention."""

    def __call__(self, x, y):
        return x + y - 4.0

    def __eq__(self, other):
        return isinstance(other, _Constraint)

    def __hash__(self):
        return hash(type(self))


def test_vmcon_driver_reaches_a_known_constrained_optimum():
    graph, x, y = _toy_problem()
    drive = Drive(
        subgraph=graph, driver=VmconDriver(n_equality=0, n_inequality=1, scaled=False)
    )
    out = drive({x: jnp.asarray(0.0), y: jnp.asarray(0.0)})
    assert float(out[x]) == pytest.approx(2.5, abs=1e-6)
    assert float(out[y]) == pytest.approx(1.5, abs=1e-6)


def test_vmcon_driver_honours_bounds_as_bounds():
    """A box bound is handed to VMCON as a bound, not re-expressed as two inequality
    constraints -- which would be a different QP subproblem and different iterates."""
    graph, x, y = _toy_problem()
    drive = Drive(
        subgraph=graph,
        driver=VmconDriver(
            n_equality=0,
            n_inequality=1,
            scaled=False,
            bounds=((x, -np.inf, 2.0),),
        ),
    )
    out = drive({x: jnp.asarray(0.0), y: jnp.asarray(0.0)})
    assert float(out[x]) <= 2.0 + 1e-9


def test_vmcon_driver_scaling_does_not_move_the_answer():
    """PROCESS's `x * (1/x_start)` conditioning changes the path, never the optimum."""
    graph, x, y = _toy_problem()
    out = Drive(
        subgraph=graph, driver=VmconDriver(n_equality=0, n_inequality=1, scaled=True)
    )({x: jnp.asarray(1.0), y: jnp.asarray(1.0)})
    assert float(out[x]) == pytest.approx(2.5, abs=1e-6)
    assert float(out[y]) == pytest.approx(1.5, abs=1e-6)


def test_vmcon_driver_refuses_a_wrong_condition_count():
    """`ConditionMap` cannot say which condition is which, so the counts are a contract
    the driver is given -- and a stale pair must fail loudly, not mislabel a constraint.
    """
    graph, x, y = _toy_problem()
    drive = Drive(
        subgraph=graph, driver=VmconDriver(n_equality=1, n_inequality=1, scaled=False)
    )
    with pytest.raises(ValueError, match="equalities"):
        drive({x: jnp.asarray(0.0), y: jnp.asarray(0.0)})


def test_vmcon_driver_needs_a_start():
    graph, x, y = _toy_problem()
    driver = VmconDriver(n_equality=0, n_inequality=1)
    subgraph = graph
    with pytest.raises(ValueError, match="starting value"):
        driver(
            Drive(subgraph=subgraph, driver=driver).condition_map({}),
            None,
        )
