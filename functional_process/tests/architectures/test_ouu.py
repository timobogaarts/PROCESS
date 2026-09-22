"""`architectures.ouu` on `stellarator_helias`: the two-stage problem assembled on the
build table at N = 32, alpha 0.75, from the file's own cold design, under the default
`bracketed` closure (the density's root find alone, the Picards nested inside it) --
and once under the `flattened` one, the handoff's form, so both stay covered.

Three things are checked, all deterministic: the batched program's nominal row is the
un-batched closed MDA (and the hoisted program is the un-hoisted one, every row); the
fused value + Jacobian is finite and its Jacobian is a central difference of the
forward program; and one boxed SLSQP call through the outer graph moves the design
and reports. Nothing is pinned to a number a sample set could move -- the only
numbers here are the nominal design's, and those are the closed MDA's own.

Beside them, the columns a caller adds (`extra_columns`), the other drivers' verdicts
in a row's validity, and `outer`'s hooks as keywords.

Plus the `BoxedSlsqpDriver` on a toy `Optimise` it can be checked against by hand.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cottax.pytree.executable import ExecutableGraph
from cottax.execution import RunnableGraph
from cottax.execution.schedule import Schedule
from cottax.pytree.graph import Graph
from cottax.interfaces.pytree_namespace_module import area, resolve
from cottax.pytree.names import PathMap
from cottax.pytree.nodes import ImplementedFunction
from cottax.pytree.problem import Converged, Optimise, Steps
from cottax.pytree.rewrites import Assign
from cottax.pytree.spec import NodePath, VarPath
from jax.tree_util import GetAttrKey

from functional_process.configurations import kinds
from functional_process.cottax.architectures import beliefs, ouu, session
from functional_process.cottax.architectures.drivers import (
    BOXED_CONVERGED,
    BoxedSlsqpDriver,
    Status,
)
from functional_process.cottax.architectures.mda import guess_sources

jax.config.update("jax_enable_x64", True)

NAME = "stellarator_helias"
N = 32
ALPHA = 0.75

DESIGN = (
    ".physics.b_plasma_toroidal_on_axis",
    ".physics.rmajor",
    ".physics.temp_plasma_electron_vol_avg_kev",
    ".tfcoil.t_tf_superconductor_quench",
    ".tfcoil.f_a_tf_turn_cable_copper",
    ".physics.f_nd_alpha_thermal_electron",
)
"""The file's `ixc` minus the density (closed) and `hfact` (a belief)."""


def _rel(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-300)


@pytest.fixture(scope="module")
def live():
    """The configuration, opened once for the module."""
    return session.open_session(NAME)


@pytest.fixture(scope="module")
def model(live):
    """The two-stage problem: build table, N = 32, alpha 0.75, `levelised`."""
    return ouu.two_stage(live, n=N, alpha=ALPHA, seed=0)


@pytest.fixture(scope="module")
def fns(model):
    """The hoisted batched program."""
    return ouu.make(model)


@pytest.fixture(scope="module")
def program(model, fns):
    """The program bound to the model's sample set."""
    return ouu.Program(fns, model.theta, model.starts0)


# ---------------------------------------------------------------- assembly


def test_assembly_has_the_shape_the_handoff_states(model):
    """Six design places, eighteen sampled beliefs (plus the dummy), twelve CVaR
    constraints, the closing problem of three unknowns, a first stage worth hoisting.
    """
    assert [v.spelling for v in model.design] == list(DESIGN)
    assert len(model.ixc) == 6
    assert model.x0.shape == model.lower.shape == model.upper.shape == (6,)
    assert np.all(model.lower < model.x0)
    assert np.all(model.x0 < model.upper)
    sampled = [b.path for b in model.beliefs if b.path != "dummy"]
    assert len(sampled) == 18
    assert not set(sampled) & set(ouu.BUILD_TABLE)
    assert model.dropped == ()
    assert model.n_g == 12
    assert model.m == 8  # ceil(0.25 * 32)
    assert model.unknowns[0].spelling == kinds.PAIRINGS["one"]["^cond.constraints.c2"]
    # The default closure: the root find over the density alone, nested.
    assert model.closure == "bracketed"
    assert model.closed.flat is False
    assert len(model.unknowns) == len(model.guesses) == 1
    assert model.closed.report["closing_problems"] == {
        ".Close.c2": (".physics.nd_plasma_electrons_vol_avg",)
    }
    assert model.te_grid is None
    # The sample set: N + 1 rows, the nominal last, every batched value N + 1 long.
    k = len(model.beliefs)
    assert model.Theta.shape == (N + 1, k)
    assert model.starts0.shape == (N + 1, 1)
    assert all(np.asarray(v).shape[0] == N + 1 for v in model.theta.values())
    nominal = beliefs.nominal_coordinates(model.beliefs, model.nominal)
    assert np.array_equal(model.Theta[-1], nominal)
    assert np.all((model.Theta[:-1] != nominal).any(axis=1))
    # The split: the whole coil and radial build hoisted, the plasma per sample.
    counts = {s.value: c.nodes for s, c in model.stages.counts.items()}
    assert counts["first"] == 66
    assert counts["second"] == 102
    assert len(model.columns) == 4 + 12 + 2 + 1
    assert model.layout["c_u"] == (18, 19)
    # No other driver in the recourse schedule reports `Converged` here (the four
    # Picards report nothing -- the density cycle's own is nested inside the root
    # find, where the flattened closure folds it in), so the verdict block is empty,
    # and so is the extra one.
    assert model.verdicts == ()
    assert model.extra == ()
    assert model.layout["c_verdicts"] == (19, 19)
    assert model.layout["c_extra"] == (19, 19)
    assert model.layout["verdicts"] == model.layout["extra"] == ()
    others = [
        step.problem.spelling
        for step in ouu.driven_problems(model.recourse)
        if step.problem != model.place
    ]
    assert len(others) == 2
    # The density cycle's statement. It is `nd_plasma_fuel_ions_vol_avg`, not
    # `fusden_alpha_total`, because `acb7f34d` replaced `mda.CUTS` -- a fixed table of
    # nine hand-picked loop-carried variables -- with `GaussSeidelMinimal`, which
    # derives the minimal cut per cycle. Both open the same cycle; the algorithm
    # chooses a different variable on it. The `Tabled` scheme that wrapped the hand
    # table was kept at `acb7f34d` "for the OUU line", never wired to it, and deleted
    # at `48cc2c14`; this literal outlived it.
    assert "^mda.physics.nd_plasma_fuel_ions_vol_avg.mda" in others
    assert all(
        not step.reports
        for step in ouu.driven_problems(model.recourse)
        if step.problem != model.place
    )


def test_extra_columns_are_carried_through_and_ignored_by_the_measures(live):
    """`extra_columns`: a recourse output, a closed residual, a first-stage output
    and an input, resolved by spelling, appended last; the nominal row reads the
    un-batched closed MDA's value at each; `per_sample` / `summarise` carry them by
    name; a non-finite extra does not invalidate a row; a misspelling is refused
    with the spellings near it.
    """
    asked = (
        ".physics.p_fusion_total_mw",
        "^cond.constraints.c2",
        ".stellarator.wp_width_r_min",  # first stage: read as a constant per sample
        ".physics.hfact",  # an input, sampled: the belief's own value
    )
    model = ouu.two_stage(live, n=4, alpha=ALPHA, seed=0, extra_columns=asked)
    layout = model.layout
    assert [v.spelling for v in model.extra] == list(asked)
    assert layout["extra"] == asked
    assert layout["c_extra"] == (19, 23)
    assert len(model.columns) == 23
    assert model.columns[19:] == model.extra
    fns = ouu.make(model)
    rows = np.asarray(
        jax.block_until_ready(
            ouu.jitted(fns, "run_batch")(
                jnp.asarray(model.x0), model.theta, jnp.asarray(model.starts0)
            )
        )
    )
    assert rows.shape == (5, 23)
    e0, _e1 = layout["c_extra"]
    for j, var in enumerate(model.extra):
        reference = float(np.asarray(model.nominal_out[var]))
        # `c2` is round-off zero at the root, so it is compared absolutely: the
        # bracketed driver re-evaluates the residual there and gets another zero.
        assert _rel(rows[-1, e0 + j], reference) < 1e-10 or (
            abs(rows[-1, e0 + j] - reference) < 1e-12
        ), var.spelling
    hfact = rows[:, e0 + 3]
    assert np.allclose(hfact, np.asarray(model.theta[model.var_of[".physics.hfact"]]))
    valid = ouu.valid_rows(layout, rows)
    assert np.all(np.abs(rows[valid, e0 + 1]) < 1e-8)  # c2 closed where converged
    by_name = ouu.extra_columns(layout, rows)
    assert list(by_name) == list(asked)
    assert np.array_equal(by_name[asked[0]], rows[:, e0])
    columns = ouu.per_sample(layout, rows)
    assert set(columns["extra"]) == set(asked)
    assert columns["extra"][asked[0]].shape == (4,)
    summary = ouu.summarise(model, layout, rows)
    assert set(summary["extra"]) == set(asked)
    assert summary["extra"][asked[0]]["nominal"] == pytest.approx(rows[-1, e0])
    # Not measured: a NaN in an extra column leaves the row valid.
    spoiled = rows.copy()
    spoiled[:, e0] = np.nan
    assert np.array_equal(ouu.valid_rows(layout, spoiled), ouu.valid_rows(layout, rows))
    with pytest.raises(KeyError, match="p_fusion_total_mw"):
        ouu.two_stage(live, n=4, extra_columns=(".physics.p_fusion_total_mv",))


def test_te_recourse_takes_the_temperature_off_the_design(live):
    """`te_recourse = 3`: five design places and a three-point grid over T_e's bounds."""
    model = ouu.two_stage(live, n=4, alpha=ALPHA, te_recourse=3)
    assert [v.spelling for v in model.design] == [d for d in DESIGN if d != kinds.TE]
    assert model.te_grid.shape == (3,)
    assert model.te_grid[0] == pytest.approx(3.0)
    assert model.te_grid[-1] == pytest.approx(15.0)
    assert kinds.TE in {v.spelling for v in model.recourse.inputs}
    assert kinds.TE not in {v.spelling for v in model.first.inputs}


def test_refusals(live):
    """An unknown objective, an unknown closure, and `with_c16` on a pairing that
    closes c16.
    """
    with pytest.raises(ValueError, match="objective"):
        ouu.two_stage(live, n=4, objective="worst")
    with pytest.raises(ValueError, match="closure"):
        ouu.two_stage(live, n=4, closure="nested")
    with pytest.raises(ValueError, match="closure"):
        ouu.close(live, kinds.PAIRINGS["one"], "newton")
    with pytest.raises(ValueError, match="closes"):
        ouu.two_stage(live, n=4, pairing="two", with_c16=True)


# ---------------------------------------------------------------- the batch


def test_nominal_row_is_the_closed_mda_and_the_hoist_is_exact(model, fns):
    """The batched program's last row at the deterministic start is the un-batched
    closed MDA's output (coe, net, every g, the closing unknowns) to 1e-10, and the
    hoisted program equals the whole-graph program on every row to 1e-10.
    """
    began = time.perf_counter()
    rows = np.asarray(
        jax.block_until_ready(
            ouu.jitted(fns, "run_batch")(
                jnp.asarray(model.x0), model.theta, jnp.asarray(model.starts0)
            )
        )
    )
    hoisted_s = time.perf_counter() - began
    assert rows.shape == (N + 1, len(model.columns))
    layout = model.layout
    nominal = rows[-1]
    assert nominal[layout["c_conv"]] > 0.5
    # Started from its own root: the bracketed driver spends one evaluation seeing
    # that the residual is already zero there (`test_bracketed`).
    assert nominal[layout["c_steps"]] <= 1
    primed = model.nominal_out
    for i, c in enumerate(model.columns):
        if i == layout["c_steps"]:
            continue
        reference = float(np.asarray(primed[c]))
        assert _rel(nominal[i], reference) < 1e-10, (c.spelling, nominal[i], reference)
    coe = float(np.asarray(primed[model.var_of[ouu.COE]]))
    net = float(np.asarray(primed[model.var_of[ouu.NET]]))
    assert 10.0 < coe < 1000.0
    assert net > 0.0
    # Every sample converged or not is a row; its columns are finite where it did.
    conv = ouu.valid_rows(layout, rows[:-1])
    assert conv.mean() > 0.9
    assert np.all(np.isfinite(rows[:-1][conv]))

    whole = ouu.make(model, hoist=False)
    began = time.perf_counter()
    rows_whole = np.asarray(
        jax.block_until_ready(
            ouu.jitted(whole, "run_batch")(
                jnp.asarray(model.x0), model.theta, jnp.asarray(model.starts0)
            )
        )
    )
    whole_s = time.perf_counter() - began
    both_nan = np.isnan(rows) & np.isnan(rows_whole)
    assert np.all(both_nan | (_rel(rows, rows_whole) < 1e-10))
    assert hoisted_s < 120, hoisted_s
    assert whole_s < 120, whole_s


def test_flattened_closure_answers_the_same_nominal_row(live, model, fns):
    """`closure="flattened"`, the handoff's form: the closing problem is the Newton
    over the density and the two cut copies, the un-batched closed MDA it is seeded
    at is the bracketed one's to 1e-10 at every measured column, and the batched
    program's nominal row is that MDA's to 1e-10 -- the same equalities the default
    closure holds. What the two do not share is reported, not pinned: the failed
    fraction and `f` at the cold design (the bracketed closure converges from any
    start, the flattened Newton stalls in some samples).
    """
    flat = ouu.two_stage(live, n=N, alpha=ALPHA, seed=0, closure="flattened")
    assert flat.closure == "flattened"
    assert flat.closed.flat is True
    assert [v.spelling for v in flat.design] == list(DESIGN)
    assert len(flat.unknowns) == len(flat.guesses) == 3
    assert flat.unknowns[0] == model.unknowns[0]
    assert flat.starts0.shape == (N + 1, 3)
    assert len(flat.columns) == 4 + 12 + 2 + 3
    assert flat.layout["c_u"] == (18, 21)
    assert np.array_equal(flat.Theta, model.Theta)  # the same Sobol' set
    assert np.array_equal(flat.x0, model.x0)
    layout = flat.layout
    for c in flat.columns[: layout["c_steps"]]:
        reference = float(np.asarray(model.nominal_out[c]))
        assert _rel(np.asarray(flat.nominal_out[c]), reference) < 1e-10, c.spelling
    flat_fns = ouu.make(flat)
    rows = np.asarray(
        jax.block_until_ready(
            ouu.jitted(flat_fns, "run_batch")(
                jnp.asarray(flat.x0), flat.theta, jnp.asarray(flat.starts0)
            )
        )
    )
    assert rows.shape == (N + 1, len(flat.columns))
    nominal = rows[-1]
    assert nominal[layout["c_conv"]] > 0.5
    assert nominal[layout["c_steps"]] == 0  # the Newton at its own root
    for i, c in enumerate(flat.columns):
        if i == layout["c_steps"]:
            continue
        reference = float(np.asarray(flat.nominal_out[c]))
        assert _rel(nominal[i], reference) < 1e-10, (c.spelling, nominal[i], reference)
    conv = ouu.valid_rows(layout, rows[:-1])
    assert conv.mean() > 0.9
    # The two closures agree where both converged, on every measured column.
    bracketed = np.asarray(
        jax.block_until_ready(
            ouu.jitted(fns, "run_batch")(
                jnp.asarray(model.x0), model.theta, jnp.asarray(model.starts0)
            )
        )
    )
    both = conv & ouu.valid_rows(model.layout, bracketed[:-1])
    assert both.sum() >= 0.9 * N
    measured = slice(0, layout["c_steps"])
    assert np.max(_rel(rows[:-1][both, measured], bracketed[:-1][both, measured])) < 1e-8
    program = ouu.Program(flat_fns, flat.theta, flat.starts0)
    values = program.at(flat.x0).values
    print(
        f"\nflattened closure at the cold design, N = {N}: failed "
        f"{round(values[-1] * N)}/{N}, f {values[0]:.4f}"
    )


def test_value_jac_starts_is_finite_and_matches_a_central_difference(model, program):
    """One fused call: `[f, *cvar, failed]` and its Jacobian finite, the next starts
    the converged roots, and `d/d rmajor` a central difference of the forward program
    to 1e-4 relative on every row it moves.
    """
    evaluation = program.at(model.x0)
    values, jacobian = evaluation.values, evaluation.jacobian
    assert values.shape == (1 + model.n_g + 1,)
    assert jacobian.shape == (1 + model.n_g + 1, 6)
    assert np.all(np.isfinite(values))
    assert np.all(np.isfinite(jacobian))
    assert evaluation.next_starts.shape == (N + 1, 1)
    failed = values[-1]
    assert 0.0 <= failed <= 0.2
    # `f` is the levelised coe plus the failed penalty; the nominal row is in it.
    assert values[0] == pytest.approx(
        evaluation.diagnostics["levelised"] + ouu.FAILED_F * failed, rel=1e-12
    )
    assert evaluation.diagnostics["coe_nominal"] == pytest.approx(
        float(np.asarray(model.nominal_out[model.var_of[ouu.COE]])), rel=1e-10
    )
    # The next starts: the converged samples' roots, the start where one failed.
    rows = program.rows(model.x0)
    layout = model.layout
    conv = ouu.valid_rows(layout, rows)
    u0, u1 = layout["c_u"]
    nxt = np.asarray(evaluation.next_starts)
    assert np.allclose(nxt[conv], rows[conv, u0:u1], rtol=1e-12)
    assert np.array_equal(nxt[~conv], model.starts0[~conv])
    # The memo: a second call at the same point is free.
    assert program.at(model.x0) is evaluation
    assert program.calls == 1

    j = 1  # rmajor
    h = 1e-5 * abs(model.x0[j])
    up, down = model.x0.copy(), model.x0.copy()
    up[j] += h
    down[j] -= h
    difference = (program.forward(up) - program.forward(down)) / (2.0 * h)
    column = jacobian[:, j]
    moving = np.abs(column) > 1e-8 * np.max(np.abs(column))
    assert moving.sum() >= 10
    assert np.max(_rel(difference[moving], column[moving])) < 1e-4
    assert column[-1] == pytest.approx(0.0, abs=0.0)  # the failed fraction is a count


# ---------------------------------------------------------------- the outer graph


def test_outer_graph_is_the_statistics_node_and_the_optimise(model, fns, program):
    """The architecture as structure: one node owning the conditions, one `Optimise`
    reading them, one driven block.
    """
    built = ouu.outer(model, fns, max_iter=1, max_outer=1)
    assert set(built.graph.nodes) == {ouu.STATISTICS, ouu.OPTIMISE}
    node = built.graph[ouu.STATISTICS]
    assert tuple(node.reads) == model.design
    assert tuple(node.owns) == (built.objective, *built.cvars, built.failed)
    assert [c.spelling for c in built.cvars] == [
        f"^cond.ouu.cvar.{c.spelling.rsplit('.', 1)[-1]}" for c in model.constraints
    ]
    (step,) = built.schedule.steps
    assert step.problem == ouu.OPTIMISE
    assert tuple(step.unknowns) == model.design
    assert tuple(step.conditions) == (built.objective, *built.cvars, built.failed)
    assert {v.spelling for v in built.starts.values()} == set(DESIGN)
    assert isinstance(built.driver, BoxedSlsqpDriver)
    assert built.driver.n_inequality == model.n_g + 1
    assert built.driver.jacobian == built.program.jacobian
    assert built.driver.warm_start == built.program.adopt


def test_outer_takes_its_hooks_as_keywords(model, fns):
    """`warm_start` / `jacobian`: `DEFAULT` is the program's own, `None` is the
    driver's own meaning (no warm start; `jacfwd` through the node), and any other
    callable passes through.
    """
    built = ouu.outer(model, fns, warm_start=None, jacobian=None, max_iter=1)
    assert built.driver.warm_start is None
    assert built.driver.jacobian is None

    def rows(flat_x):
        return np.zeros((1 + model.n_g + 1, len(flat_x)))

    def adopt(flat_x):
        return None

    built = ouu.outer(model, fns, warm_start=adopt, jacobian=rows, max_iter=1)
    assert built.driver.warm_start is adopt
    assert built.driver.jacobian is rows
    assert repr(ouu.DEFAULT) == "DEFAULT"
    built = ouu.outer(model, fns, warm_start=ouu.DEFAULT, max_iter=1)
    assert built.driver.warm_start == built.program.adopt


def test_valid_rows_needs_every_verdict():
    """`valid_rows` on a hand-made layout: the closing verdict, every other driver's
    verdict, and finiteness up to `c_steps` all have to hold.
    """
    layout = {
        "c_coe": 0,
        "c_net": 1,
        "c_avail": 2,
        "c_concost": 3,
        "c_g": (4, 5),
        "c_steps": 5,
        "c_conv": 6,
        "c_u": (7, 8),
        "c_verdicts": (8, 10),
        "verdicts": ("^problem.a", "^problem.b"),
        "c_extra": (10, 11),
        "extra": (".x",),
        "tiny": ouu.TINY,
    }
    good = [1.0, 2.0, 0.5, 3.0, -0.1, 4.0, 1.0, 0.7, 1.0, 1.0, np.nan]
    rows = np.array([
        good,
        [*good[:6], 0.0, *good[7:]],  # the closing problem failed
        [*good[:8], 0.0, 1.0, np.nan],  # driver a failed
        [*good[:8], 1.0, 0.0, np.nan],  # driver b failed
        [np.nan, *good[1:]],  # a measured column non-finite
    ])
    assert ouu.valid_rows(layout, rows).tolist() == [True, False, False, False, False]


def test_one_boxed_slsqp_call_moves_the_design_and_reports(model, fns):
    """One SLSQP call boxed at +-25 %, three major iterations: the call moves the
    design inside its box and the run reports `Steps`, `Converged`, `Status` without
    raising. From the file's cold design the CVaRs are far from feasible, so the call
    ends on the box, less infeasible than it started, and the driver re-centres there
    and hands that point back when its one outer call is spent.
    """
    entries = []
    built = ouu.outer(model, fns, max_iter=3, max_outer=1, callback=entries.append)
    began = time.perf_counter()
    x, out, _elapsed = ouu.solve(built)
    wall = time.perf_counter() - began
    assert wall < 150, wall
    assert x.shape == (6,)
    (entry,) = entries
    moved = np.asarray(entry["x"]) - model.x0
    assert entry["nit"] >= 1
    assert entry["moved"] > 0.0
    assert np.max(np.abs(moved)) > 0.0
    assert np.all(np.abs(moved) <= 0.25 * np.abs(model.x0) * (1 + 1e-9))
    assert np.max(np.abs(x - model.x0)) > 0.0
    assert np.allclose(x, entry["x"])
    steps = int(np.asarray(built.verdict(out, Steps)))
    assert steps == entry["nit"]
    assert bool(np.asarray(built.verdict(out, Converged))) is False
    assert int(np.asarray(built.verdict(out, Status))) != BOXED_CONVERGED
    # The statistics at the answer are in the env, at the node's names.
    values = built.program.at(x).values
    assert float(np.asarray(out[built.objective])) == pytest.approx(values[0])
    assert float(np.asarray(out[built.failed])) == pytest.approx(
        values[-1] - ouu.EPS_FAILED
    )
    summary = ouu.report(built, out)
    assert set(summary["x"]) == set(DESIGN)
    assert summary["model_calls"] == built.program.calls >= 2
    # The warm-start hook adopted a point: the program's starts moved off the cold
    # ones where the incumbent's samples converged.
    assert built.program.adopted
    # Evidence functions run on what the program returns.
    y_cold, y_warm, timing = ouu.evaluate(
        fns, x, model.theta, model.starts0, x_from=model.x0, steps=2
    )
    assert y_cold.shape == y_warm.shape == (N + 1, len(model.columns))
    assert timing["calls"] >= 2
    summarised = ouu.summarise(model, fns["layout"], y_warm)
    assert set(summarised["p_violated"]) == set(model.names)
    assert 0.0 <= summarised["feasible_fraction"] <= 1.0
    columns = ouu.per_sample(fns["layout"], y_warm)
    assert columns["coe"].shape == (N,)
    table = ouu.design_table(model, model.x0, x)
    assert [row["place"] for row in table] == list(DESIGN)


# ---------------------------------------------------------------- the driver alone

toy = area("toy")


def _var(where) -> VarPath:
    return resolve(where, VarPath)


class _Quadratic:
    """`f = (x - 1)^2 + (y - 2)^2`, `g = x + y - 2 <= 0`: the answer is (0.5, 1.5)."""

    def __call__(self, x, y):
        return (x - 1.0) ** 2 + (y - 2.0) ** 2, x + y - 2.0


def _toy_schedule(driver) -> tuple[Schedule, dict]:
    node = ImplementedFunction(
        reads=(_var(toy.x), _var(toy.y)),
        owns=(_var(toy.f), _var(toy.g)),
        fn=_Quadratic(),
    )
    problem = Optimise(
        objective=_var(toy.f),
        unknowns=(_var(toy.x), _var(toy.y)),
        inequalities=(_var(toy.g),),
    )
    place = NodePath((GetAttrKey("Opt"),))
    graph = Graph.of({NodePath((GetAttrKey("Toy"),)): node, place: problem})
    driven = Assign(place, driver).apply(graph)
    schedule = Schedule(RunnableGraph(driven))
    starts = dict(guess_sources(driven).items())
    return schedule, starts


def test_other_verdicts_lists_every_other_reporting_driver():
    """Two independent toy problems in one schedule, both driven by a driver that
    reports `Converged`: `other_verdicts` for one names the other's verdict, and only
    that; `driven_problems` sees both.
    """
    driver = BoxedSlsqpDriver(
        n_inequality=1,
        bounds=((_var(toy.x), -10.0, 10.0), (_var(toy.y), -10.0, 10.0)),
    )
    node = ImplementedFunction(
        reads=(_var(toy.x), _var(toy.y)),
        owns=(_var(toy.f), _var(toy.g)),
        fn=_Quadratic(),
    )
    problem = Optimise(
        objective=_var(toy.f),
        unknowns=(_var(toy.x), _var(toy.y)),
        inequalities=(_var(toy.g),),
    )
    other = area("other")
    node2 = ImplementedFunction(
        reads=(_var(other.x), _var(other.y)),
        owns=(_var(other.f), _var(other.g)),
        fn=_Quadratic(),
    )
    problem2 = Optimise(
        objective=_var(other.f),
        unknowns=(_var(other.x), _var(other.y)),
        inequalities=(_var(other.g),),
    )
    driver2 = BoxedSlsqpDriver(
        n_inequality=1,
        bounds=((_var(other.x), -10.0, 10.0), (_var(other.y), -10.0, 10.0)),
    )
    place, place2 = NodePath((GetAttrKey("Opt"),)), NodePath((GetAttrKey("Opt2"),))
    graph = Graph.of({
        NodePath((GetAttrKey("Toy"),)): node,
        place: problem,
        NodePath((GetAttrKey("Other"),)): node2,
        place2: problem2,
    })
    driven = Assign(place2, driver2).apply(Assign(place, driver).apply(graph))
    schedule = Schedule(RunnableGraph(driven))
    assert {s.problem for s in ouu.driven_problems(schedule)} == {place, place2}
    assert ouu.other_verdicts(schedule, place) == (Converged.name_for(place2),)
    assert ouu.other_verdicts(schedule, place2) == (Converged.name_for(place),)
    assert set(ouu.other_verdicts(schedule, NodePath((GetAttrKey("None"),)))) == {
        Converged.name_for(place),
        Converged.name_for(place2),
    }


def test_boxed_slsqp_driver_on_a_toy_converges_through_re_centring():
    """From (3, 3) with a +-25 % box, SLSQP has to re-centre several times before it
    succeeds strictly inside a box at (0.5, 1.5).
    """
    entries = []
    driver = BoxedSlsqpDriver(
        n_inequality=1,
        bounds=((_var(toy.x), -10.0, 10.0), (_var(toy.y), -10.0, 10.0)),
        delta=0.25,
        max_iter=200,
        callback=entries.append,
    )
    schedule, starts = _toy_schedule(driver)
    env = {port: jnp.asarray(3.0) for port in starts}
    out = dict(schedule.run(PathMap(env)))
    x, y = float(out[_var(toy.x)]), float(out[_var(toy.y)])
    assert (x, y) == pytest.approx((0.5, 1.5), abs=1e-5)
    place = NodePath((GetAttrKey("Opt"),))
    assert bool(out[Converged.name_for(place)])
    assert int(out[Status.name_for(place)]) == BOXED_CONVERGED
    assert int(out[Steps.name_for(place)]) == sum(e["nit"] for e in entries)
    assert len(entries) >= 3
    assert entries[0]["on_move_limit"]
    assert not entries[-1]["on_move_limit"]
    assert all(e["move_limit"] == pytest.approx(0.25) for e in entries)


def test_boxed_slsqp_driver_takes_its_jacobian_and_warm_starts_from_the_body():
    """`jacobian=` rows replace `jacfwd`, `warm_start=` sees every accepted iterate."""
    adopted = []

    def rows(flat_x):
        x, y = flat_x
        return np.array([[2.0 * (x - 1.0), 2.0 * (y - 2.0)], [1.0, 1.0]])

    driver = BoxedSlsqpDriver(
        n_inequality=1,
        bounds=((_var(toy.x), -10.0, 10.0), (_var(toy.y), -10.0, 10.0)),
        delta=0.5,
        jacobian=rows,
        warm_start=lambda flat_x: adopted.append(np.array(flat_x)),
    )
    schedule, starts = _toy_schedule(driver)
    env = {port: jnp.asarray(3.0) for port in starts}
    out = dict(schedule.run(PathMap(env)))
    assert (float(out[_var(toy.x)]), float(out[_var(toy.y)])) == pytest.approx(
        (0.5, 1.5), abs=1e-5
    )
    assert adopted
    assert np.allclose(adopted[-1], [0.5, 1.5], atol=1e-5)
