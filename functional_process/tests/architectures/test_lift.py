"""`architectures.lift` on `stellarator_helias`: the winding pack lifted out of the
coil model.

Four things, all deterministic. On the closed graph the lift takes the intersect
root find apart (the problem gone, `wp_width_r_min` a boundary input, the inequality's
node there); at the root the un-lifted run found the lifted inequality is active, a
wider pack satisfies it and a narrower one violates it -- the sign convention, pinned
numerically; every other output at that point is the un-lifted run's; and
`ouu.two_stage(lifts=(lift_winding_pack,))` assembles with one more design variable
and one more CVaR constraint, whose fused Jacobian column is a central difference.
Plus the stage split before and after the lift on the MDA graph `test_stages.py`
measures, so the handoff's coil finding has its "after" row.
"""

from __future__ import annotations

import time

import jax
import numpy as np
import pytest
from cottax.interfaces import (
    ExecutableGraph,
    Le,
    RunnableGraph,
    Schedule,
    is_root_find,
)
from cottax.interfaces import violations as executable_violations
from cottax.pytree.mint import prefix_path

from functional_process.configurations import load
from functional_process.configurations.kinds import BUILD_LEAVES, KINDS, Kind
from functional_process.cottax.architectures import (
    closing,
    lift,
    mda,
    mdf,
    ouu,
    session,
    stages,
)
from functional_process.cottax.architectures.evaluate import (
    run_schedule,
    without_excluded,
)
from functional_process.cottax.architectures.lift import (
    CURRENT_DENSITY,
    INTERSECT,
    J_TF_SC_WP,
    J_TF_SC_WP_MAX,
    WP_WIDTH_R_MIN,
    lift_winding_pack,
)
from functional_process.cottax.architectures.stages import (
    RECOURSE_PATHS,
    Stage,
    leaves,
    sampled_paths,
    split,
    violations,
)
from functional_process.cottax.input.indat import graph_for
from functional_process.cottax.queries import declared

jax.config.update("jax_enable_x64", True)

NAME = "stellarator_helias"
N = 16
ALPHA = 0.75
COIL = (".tfcoil.j_tf_wp", ".tfcoil.dr_tf_wp_with_insulation")
"""The two owned coil places the decision names beside the unknown."""
INSULATION = ".tfcoil.dx_tf_wp_insulation"


def _rel(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-300)


def _same(a, b, rtol=1e-10):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return False
    both_nan = np.isnan(a) & np.isnan(b)
    return bool(np.all(both_nan | (_rel(a, b) <= rtol)))


@pytest.fixture(scope="module")
def live():
    """The configuration, opened once for the module."""
    return session.open_session(NAME)


@pytest.fixture(scope="module")
def closed(live):
    """The closed MDA (`c2` by the density), un-lifted."""
    return closing.close(live)


@pytest.fixture(scope="module")
def primed(closed, live):
    """`(env, out)`: the un-lifted problem seeded at the file's cold design and primed
    once -- `env` with every start port at its root, `out` the run.
    """
    env = closing.seed(closed, live.reference.cold)
    return mdf.prime(closed.problem, env)


@pytest.fixture(scope="module")
def lifted(closed):
    """`lift_winding_pack` on the driven closed graph, the relaxed requirement dropped
    (`keep=False`): the optimiser of a closed problem is outside this graph, so nothing
    in it could absorb one. `(graph, condition, report)`.
    """
    return lift_winding_pack(closed.graph, keep=False)


@pytest.fixture(scope="module")
def lifted_run(lifted, primed):
    """The lifted schedule and its run at the primed env with the pack width at the
    root the un-lifted run found: `(schedule, env, out)`.
    """
    graph, _condition, _report = lifted
    env, out = primed
    schedule = Schedule(RunnableGraph(graph))
    inputs = set(schedule.inputs)
    at_root = {v: value for v, value in env.items() if v in inputs}
    at_root[WP_WIDTH_R_MIN] = out[WP_WIDTH_R_MIN]
    return schedule, at_root, run_schedule(schedule, at_root)


# ---------------------------------------------------------------- the rewrite


def test_lift_takes_the_root_find_apart(closed, lifted):
    """The intersect problem is gone, its unknown is a boundary input, the
    current-density node is there and the inequality is a relation between its two
    outputs -- nothing computes a signed residual.
    """
    before = closed.graph
    graph, condition, report = lifted
    problem = next(p for p in declared(before) if p.spelling == INTERSECT)
    assert is_root_find(before[problem])
    assert WP_WIDTH_R_MIN not in before.graph.boundary_inputs
    assert WP_WIDTH_R_MIN in before.graph.owners

    assert INTERSECT not in {p.spelling for p in declared(graph)}
    assert problem not in graph.nodes
    assert WP_WIDTH_R_MIN in graph.graph.boundary_inputs
    assert WP_WIDTH_R_MIN not in graph.graph.owners
    assert len(graph.nodes) == len(before.nodes)  # - problem + density
    assert condition == J_TF_SC_WP
    assert tuple(graph[CURRENT_DENSITY].owns) == (J_TF_SC_WP, J_TF_SC_WP_MAX)
    assert WP_WIDTH_R_MIN in graph[CURRENT_DENSITY].reads
    assert graph.graph.owners[condition] == CURRENT_DENSITY
    # The residual's producer still runs -- the lift keeps it, as a cut keeps its.
    residual = prefix_path(WP_WIDTH_R_MIN, lift.COND)
    assert residual.spelling == "^cond.stellarator.wp_width_r_min"
    assert graph.graph.owners[residual] == before.graph.owners[residual]
    # The report: cottax's ops, in order, and nothing composed by hand.
    assert report["ops"] == (
        f"undrive({INTERSECT})",
        f"undetermine({INTERSECT})",
        f"delete({INTERSECT})",
    )
    assert report["driver"] == "SeededNewtonDriver"
    assert report["unnested"] == ()
    assert (report["safe"], report["sign"]) == ("above", -1.0)
    assert report["kept"] is False
    assert report["relation"].op is Le
    assert report["relation"].lhs == J_TF_SC_WP
    assert report["relation"].rhs == J_TF_SC_WP_MAX
    assert report["icc"] == 33
    assert report["design"] == {
        "unknown": WP_WIDTH_R_MIN.spelling,
        "bounds": (0.001, 2.0),
        "bounds_from": "ITERATION_VARIABLES[140], the place's own unit (m)",
        "ixc": 140,
        "kind": Kind.BUILD,
    }
    # Every other problem and its driver is untouched.
    for p in declared(graph):
        assert graph[p] == before[p]
    assert not graph.within


def test_the_kept_requirement_is_the_inequality_and_nothing_answers_it(closed):
    """`keep=True` leaves the relaxed requirement where the root find was: an
    inequality between the two current densities, determining nothing -- so the graph
    is drawn and rewritten but not executable until an architecture absorbs it.
    """
    graph, _condition, report = lift_winding_pack(closed.graph, keep=True)
    assert report["kept"] is True
    assert report["node"] == INTERSECT
    place = next(n for n in graph.nodes if n.spelling == INTERSECT)
    node = graph[place]
    assert node.unknowns == ()
    assert node.relations == (report["relation"],)
    # Every case the proof reports, as data rather than as the first refusal: the one
    # thing wrong with this graph is the requirement nothing answers.
    cases = list(executable_violations(ExecutableGraph, graph))
    assert [type(c).__name__ for c in cases] == ["BareCondition"]
    assert cases[0].node == place
    with pytest.raises(ValueError, match="against nothing"):
        ExecutableGraph(graph)


def test_lift_refuses_what_it_cannot_take_apart(closed, lifted):
    """A second lift finds no problem; a fixed point, a wrong unknown and a wrong
    `safe` are refused where the lift begins.
    """
    graph = closed.graph
    with pytest.raises(KeyError, match="not a problem"):
        lift_winding_pack(lifted[0])
    problem = next(p for p in declared(graph) if p.spelling == INTERSECT)
    # A genuine fixed point: the one `closing.close` makes by cutting the
    # `f_ster_div_single` cycle. Was `^problem.power.delta_eta_step`, until that node
    # stopped declaring a problem at all -- its "self-loop" read PROCESS's incoming
    # field rather than its own earlier output, and it now reads `.power.delta_eta_in`
    # (see `DeltaEtaStep`). Every fixed point in this port is cut-made now, so a cut
    # one is what this test has to take.
    fixed_point = next(
        p for p in declared(graph) if p.spelling == "^mda.fwbs.f_ster_div_single"
    )
    with pytest.raises(ValueError, match="one of"):
        lift.lift(graph, problem, unknown=WP_WIDTH_R_MIN, safe="wide")
    with pytest.raises(KeyError, match="not an unknown"):
        lift.lift(graph, problem, unknown=J_TF_SC_WP, safe="above")
    with pytest.raises(ValueError, match="not a root find"):
        lift.lift(
            graph,
            fixed_point,
            unknown=next(iter(graph[fixed_point].owns)),
            safe="above",
        )
    with pytest.raises(TypeError, match="determines nothing"):
        lift.lift(lifted[0], CURRENT_DENSITY, unknown=WP_WIDTH_R_MIN, safe="above")


# ---------------------------------------------------------------- the sign


def test_lifted_inequality_is_active_at_the_root_and_a_wider_pack_satisfies(
    lifted, lifted_run
):
    """At the root the lifted inequality is active (|g| < 1e-8, dimensionless);
    5 % wider it is satisfied (negative), 5 % narrower violated (positive). That is
    `safe="above"`: the residual `f j_c - j` is positive on the wide side.
    """
    _graph, _condition, report = lifted
    schedule, at_root, out = lifted_run
    gap = lambda env: float(
        np.asarray(env[report["relation"].lhs]) - np.asarray(env[report["relation"].rhs])
    )
    # The statement is PROCESS's icc 33 itself, `j <= f j_c`, between the two current
    # densities read off the coil's curves at the chosen width.
    j, j_max = float(np.asarray(out[J_TF_SC_WP])), float(np.asarray(out[J_TF_SC_WP_MAX]))
    assert j_max > 0
    assert j > 0
    g0 = gap(out)
    assert abs(g0) < 1e-8 * j_max, g0
    root = float(np.asarray(at_root[WP_WIDTH_R_MIN]))
    assert 0.1 < root < 2.0
    moved = {}
    for factor in (1.05, 0.95):
        env = dict(at_root)
        env[WP_WIDTH_R_MIN] = at_root[WP_WIDTH_R_MIN] * factor
        moved[factor] = gap(run_schedule(schedule, env)) / j_max
    assert moved[1.05] < -1e-3, moved
    assert moved[0.95] > 1e-3, moved
    # And the lifted unknown reaches the coil: the pack thickness *is* the width.
    thickness = next(v for v in out if v.spelling == ".tfcoil.dr_tf_wp_with_insulation")
    assert float(np.asarray(out[thickness])) == pytest.approx(root, rel=1e-14)


def test_every_other_output_is_the_unlifted_ones(closed, primed, lifted_run):
    """The un-lifted schedule run from the primed env and the lifted one at the root
    agree on every output to 1e-10 relative -- the driver verdicts and the start
    ports aside, which are the runs' own -- and the lifted run has every place the
    un-lifted one had.
    """
    env, _out = primed
    reference = run_schedule(closed.problem.eager, mdf._inputs_only(closed.problem, env))
    _schedule, _at_root, out = lifted_run
    shared = [v for v in reference if v in out]
    assert len(shared) > 500
    assert not {v.spelling for v in reference if v not in out}
    differing = [
        v.spelling
        for v in shared
        if not v.spelling.startswith(("^steps.", "^converged.", "^status.", "^guess."))
        and not _same(reference[v], out[v])
    ]
    assert not differing, differing[:20]
    assert _same(reference[WP_WIDTH_R_MIN], out[WP_WIDTH_R_MIN], rtol=0.0)


# ---------------------------------------------------------------- the stage split


@pytest.fixture(scope="module")
def runnable():
    """The driven MDA graph `test_stages.py` measures (152 nodes), and it lifted."""
    graph = mda.driven_graph(without_excluded(graph_for(load(NAME).machine)))
    return graph, lift_winding_pack(graph, keep=False)[0]


def _rows(graph, table):
    return {
        "all": leaves(graph, table),
        "sampled": leaves(graph, table, sampled=sampled_paths(), varying=RECOURSE_PATHS),
        "sampled-insulation": leaves(
            graph,
            table,
            sampled=sampled_paths(held=(INSULATION,)),
            varying=RECOURSE_PATHS,
        ),
        "build": leaves(
            graph,
            table,
            sampled=sampled_paths(held=BUILD_LEAVES),
            varying=RECOURSE_PATHS,
        ),
    }


@pytest.mark.parametrize(
    ("label", "before", "after"),
    [
        ("all", (25, 15), (37, 6)),
        ("sampled", (34, 15), (34, 14)),
        ("sampled-insulation", (34, 15), (46, 10)),
        ("build", (58, 6), (58, 6)),
    ],
)
def test_stage_split_before_and_after_the_lift(runnable, label, before, after):
    """`(first-stage nodes, claimed-build violations)` per leaf set, un-lifted
    (`test_stages.py`'s numbers) and lifted. On `all` -- every belief and operating
    leaf, `tftmp` among them -- the nine coil and radial-build violations the handoff
    traced to the coil temperature are gone and the six accepted ones remain; on
    `sampled` the pack width is a leaf and the lifted inequality is what
    `f_j_tf_wp_critical_max` reaches, while `j_tf_wp`'s owner is still reached by the
    insulation scatter, so holding that one row makes the whole coil first stage.
    """
    plain, lifted_graph = runnable
    table = {**KINDS, WP_WIDTH_R_MIN.spelling: Kind.BUILD}
    assert (len(plain.nodes), len(lifted_graph.nodes)) == (152, 152)
    stage_before = split(plain, _rows(plain, KINDS)[label])
    stage_after = split(lifted_graph, _rows(lifted_graph, table)[label])
    hits_before, _ = violations(plain, stage_before)
    hits_after, _ = violations(lifted_graph, stage_after)
    assert (len(stage_before.first), len(hits_before)) == before
    assert (len(stage_after.first), len(hits_after)) == after, stages.report(
        stage_after, hits_after, label
    )
    assert WP_WIDTH_R_MIN.spelling not in {v.place for v in hits_after}
    assert WP_WIDTH_R_MIN in stage_after.leaves.build
    # The lift computes no condition of its own: what the inequality is a function of
    # is the current-density node, so that is what has a stage.
    node = CURRENT_DENSITY
    if label == "all":
        assert stage_after.stage[node] is Stage.RECOURSE
        assert stage_after.responsible(node) == (".tfcoil.tftmp",)
        assert not {v.place for v in hits_after} & set(COIL)
    if label == "sampled":
        by_place = {v.place: v.responsible for v in hits_before}
        assert by_place[".tfcoil.j_tf_wp"] == (
            ".constraints.f_j_tf_wp_critical_max",
            INSULATION,
        )
        by_place = {v.place: v.responsible for v in hits_after}
        assert by_place[".tfcoil.j_tf_wp"] == (INSULATION,)
        assert stage_after.responsible(node) == (".constraints.f_j_tf_wp_critical_max",)
    if label == "sampled-insulation":
        assert not {v.place for v in hits_after} & set(COIL)
        assert stage_after.stage[node] is Stage.SECOND
    if label == "build":
        assert stage_after.stage[node] is Stage.FIRST


# ---------------------------------------------------------------- the two-stage hook


@pytest.fixture(scope="module")
def model(live):
    """The two-stage problem with the pack lifted: build table, N = 16, alpha 0.75."""
    return ouu.two_stage(live, n=N, alpha=ALPHA, seed=0, lifts=(lift_winding_pack,))


def test_two_stage_assembles_with_the_pack_lifted(model, primed, lifted):
    """Seven design places (the six plus the pack width, `ixc` 140, bounds 0.001 ..
    2.0 m, started at the root), thirteen CVaR constraints (the twelve plus the
    lifted inequality), and the coil first stage on the build table.
    """
    _env, out = primed
    _graph, condition, report = lifted
    gap = ouu.Gap.of(report["relation"])
    assert len(model.design) == 7
    assert model.design[-1] == WP_WIDTH_R_MIN
    assert model.ixc == (2, 3, 4, 56, 59, 109, 140)
    assert model.x0.shape == model.lower.shape == model.upper.shape == (7,)
    assert (model.lower[-1], model.upper[-1]) == (0.001, 2.0)
    assert model.x0[-1] == pytest.approx(float(np.asarray(out[WP_WIDTH_R_MIN])), rel=0)
    assert np.all(model.lower < model.x0)
    assert np.all(model.x0 < model.upper)
    assert model.n_g == 13
    # The lift leaves a relation, not a place: the column is the gap between the two
    # current densities, which is what the statement holds at `<= 0`.
    assert model.constraints[-1] == gap
    assert len(model.columns) == 4 + 13 + 2 + 1  # the bracketed closure: one unknown
    assert model.layout["c_u"] == (19, 20)
    assert model.closed.report["lifts"][condition.spelling]["design"]["ixc"] == 140
    assert model.closed.report["lift_relations"] == (report["relation"],)
    assert set(model.closed.report["inequalities"]) == set(model.constraints) - {gap}
    assert model.closed.problem.n_inequality == 13
    assert model.closed.design == model.closed.problem.design
    assert INTERSECT not in {p.spelling for p in declared(model.closed.graph)}
    assert INTERSECT not in {p.spelling for p in declared(model.closed.problem.graph)}
    # At the nominal the lifted inequality is active, in the un-batched run too.
    assert abs(float(np.asarray(gap.at(model.nominal_out)))) < 1e-8 * float(
        np.asarray(model.nominal_out[J_TF_SC_WP_MAX])
    )
    # The split on the build table: the current-density node first stage in place of
    # the root find, so the coil group and the lifted inequality are the same in every
    # sample until a coil row is sampled (`held=ECONOMIC` makes
    # `f_j_tf_wp_critical_max` reach it).
    counts = {s.value: c.nodes for s, c in model.stages.counts.items()}
    assert counts == {"first": 66, "second": 102, "recourse": 0}
    assert model.stages.stage[CURRENT_DENSITY] is Stage.FIRST
    owned = stages.owned_by_spelling(model.closed.graph)
    for place in COIL:
        assert model.stages.stage[owned[place][1]] is Stage.FIRST
    assert WP_WIDTH_R_MIN in model.stages.leaves.build
    assert WP_WIDTH_R_MIN in set(model.first.inputs)
    assert WP_WIDTH_R_MIN not in set(model.recourse.inputs)


def test_value_jac_starts_column_for_the_pack_width(model):
    """One fused call is finite, and `d/d wp_width_r_min` -- the new column -- is a
    central difference of the forward program to 1e-4 relative on every row it
    moves; the lifted CVaR's own entry is the largest, since the inequality is
    active and the width is what it is a function of.
    """
    fns = ouu.make(model)
    program = ouu.Program(fns, model.theta, model.starts0)
    began = time.perf_counter()
    evaluation = program.at(model.x0)
    fused_s = time.perf_counter() - began
    assert fused_s < 150, fused_s
    values, jacobian = evaluation.values, evaluation.jacobian
    assert values.shape == (1 + 13 + 1,)
    assert jacobian.shape == (1 + 13 + 1, 7)
    assert np.all(np.isfinite(values))
    assert np.all(np.isfinite(jacobian))
    assert evaluation.next_starts.shape == (N + 1, 1)
    assert 0.0 <= values[-1] <= 0.2
    # The lifted CVaR: active at the nominal design, the same in every sample.
    assert abs(values[13]) < 1e-8, values[13]

    j = 6  # the pack width
    h = 1e-5 * abs(model.x0[j])
    up, down = model.x0.copy(), model.x0.copy()
    up[j] += h
    down[j] -= h
    difference = (program.forward(up) - program.forward(down)) / (2.0 * h)
    column = jacobian[:, j]
    moving = np.abs(column) > 1e-8 * np.max(np.abs(column))
    assert moving.sum() >= 4, column
    assert moving[0]
    assert moving[13]
    assert column[13] < 0.0  # wider is safer
    assert np.max(_rel(difference[moving], column[moving])) < 1e-4
    assert column[-1] == pytest.approx(0.0, abs=0.0)  # the failed fraction is a count
