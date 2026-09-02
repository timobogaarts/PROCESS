"""Harness cases for the ported subset of `process/models/vacuum.py` (registry #16).

Audit record: `functional_process/_audit/units/models/vacuum.md`. Three units:

- `TestVacuumPumpingSimple` -- `Vacuum.vacuum_simple`, tier-1.
- `TestSolveDuctDiameter` -- `Vacuum._newton_method_duct_diameter`'s inner Newton loop
  (the duct-diameter root-find, isolated), tier-2, same shape as `coils.py`'s
  `TestIntersect`.
- `TestVacuumPumpingOld` -- `Vacuum.vacuum` (the full `"old"` duct-sizing model) plus
  `Vacuum.run()`'s rounding step, tier-2.

`VacuumVessel` is out of scope on the stellarator (unreached from `Stellarator.run()`)
but reached on the tokamak (`caller.py:331`) -- wave-1 tokamak dispatch adds two more
units here:

- `TestCalculateVesselHalfHeight` -- `VacuumVessel.calculate_vessel_half_height`,
  `n_divertors == 1` baked, tier-1.
- `TestCalculateEllipticalVesselVolumes` -- `VacuumVessel.
  calculate_elliptical_vessel_volumes`, tier-1.

See `vacuum.md`'s tokamak-scope addendum.
"""

from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pytest
from cottax import (
    AbstractDriver,
    CallableNode,
    Feasibility,
    Graph,
    RootFind,
    Start,
)
from cottax.blocking import Blocking
from cottax.evaluate import Schedule
from cottax.interfaces.pytree_namespace_module import to_graph
from cottax.rewrites import Assign
from cottax.spec import NodePath
from cottax.tools.path import path_map
from jax.tree_util import DictKey

from functional_process._harness import (
    Sample,
    Tier1Contract,
    Tier2Contract,
    fuzz_samples,
    legacy_sample,
)
from functional_process._harness import path as vpath
from functional_process.models.vacuum.vacuum import (
    XMULT,
    DuctDiameterRootFind,
    DuctFeasibility,
    DuctFeasibilityConditions,
    _solve_vacuum_pumping_old,
    _solve_vacuum_pumping_old_from_fields,
    calculate_dshaped_vessel_volumes,
    calculate_elliptical_vessel_volumes,
    calculate_vacuum_pumping_simple,
    calculate_vacuum_vessel_outputs,
    calculate_vacuum_vessel_outputs_double_null,
    calculate_vacuum_vessel_outputs_dshaped_double_null,
    calculate_vessel_half_height,
    calculate_vessel_half_height_double_null,
    duct_diameter_residual,
    duct_fits_residual,
    pumping_speed_floor_residual,
    solve_duct_diameter,
)
from process.core import constants
from process.core.model import DataStructure
from process.models.vacuum import Vacuum, VacuumVessel


def _reference_vacuum_pumping_simple(
    molflow_plasma_fuelling_required,
    molflow_vac_pumps,
    volflow_vac_pumps_max,
    f_a_vac_pump_port_plasma_surface,
    f_volflow_vac_pumps_impedance,
    a_plasma_surface,
    n_tf_coils,
    outgasfactor,
    pres_vv_chamber_base,
    outgasindex,
    t_plant_pulse_dwell,
):
    """Call PROCESS's `Vacuum.vacuum_simple` through the port's signature."""
    data = DataStructure()
    data.physics.molflow_plasma_fuelling_required = molflow_plasma_fuelling_required
    data.vacuum.molflow_vac_pumps = molflow_vac_pumps
    data.vacuum.volflow_vac_pumps_max = volflow_vac_pumps_max
    data.vacuum.f_a_vac_pump_port_plasma_surface = f_a_vac_pump_port_plasma_surface
    data.vacuum.f_volflow_vac_pumps_impedance = f_volflow_vac_pumps_impedance
    data.physics.a_plasma_surface = a_plasma_surface
    data.tfcoil.n_tf_coils = n_tf_coils
    data.vacuum.outgasfactor = outgasfactor
    data.vacuum.pres_vv_chamber_base = pres_vv_chamber_base
    data.vacuum.outgasindex = outgasindex
    data.times.t_plant_pulse_dwell = t_plant_pulse_dwell

    v = Vacuum()
    v.data = data
    return v.vacuum_simple(output=False)


class TestVacuumPumpingSimple(Tier1Contract):
    """`Vacuum.vacuum_simple` -> `calculate_vacuum_pumping_simple`."""

    audit_record = "models/vacuum.md"
    reference = _reference_vacuum_pumping_simple
    ported = calculate_vacuum_pumping_simple

    # tests/unit/models/test_vacuum.py::TestVacuum::test_simple_model.
    samples = [
        legacy_sample(
            "simple-model",
            molflow_plasma_fuelling_required=7.5745668997694112e22,
            molflow_vac_pumps=1.2155e22,
            volflow_vac_pumps_max=27.3,
            f_a_vac_pump_port_plasma_surface=0.0203,
            f_volflow_vac_pumps_impedance=0.4,
            a_plasma_surface=1500.3146527709359,
            n_tf_coils=18,
            outgasfactor=0.0235,
            pres_vv_chamber_base=0.0005,
            outgasindex=1.0,
            t_plant_pulse_dwell=500.0,
        ),
    ]

    fuzz_bounds = {
        "molflow_plasma_fuelling_required": (1.0e21, 1.0e23),
        "molflow_vac_pumps": (1.0e21, 1.0e23),
        "volflow_vac_pumps_max": (5.0, 50.0),
        "f_a_vac_pump_port_plasma_surface": (0.005, 0.05),
        "f_volflow_vac_pumps_impedance": (0.05, 0.5),
        "a_plasma_surface": (200.0, 3000.0),
        "n_tf_coils": (10.0, 60.0),
        "outgasfactor": (0.005, 0.05),
        "pres_vv_chamber_base": (1.0e-5, 1.0e-3),
        "outgasindex": (0.5, 2.0),
        "t_plant_pulse_dwell": (10.0, 2000.0),
    }


def _reference_solve_duct_diameter(l1, l2, l3, xmult_i, ceff_i, max_iter=100, tol=0.01):
    """PROCESS's own duct-diameter Newton loop, calling its `_newton_function` directly.

    `Vacuum._newton_method_duct_diameter`'s inner loop
    (`process/models/vacuum.py:469-484`) is not exposed as a standalone function -- it's
    interleaved with the outer area-fit loop `solve_duct_geometry` ports separately. This
    adapter is a thin, faithful re-orchestration of just that inner loop (fixed `d=1.0`
    start, up to `max_iter` steps, early stop at relative step `<= tol`), but the actual
    arithmetic at every step comes from PROCESS's own `Vacuum._newton_function`
    (`@staticmethod`, called directly, not reimplemented) -- same division of labour as
    `divertor.py`'s reference adapters, which construct a `DataStructure` and call the
    real PROCESS method rather than recomputing its formula.
    """
    d = 1.0
    for _ in range(max_iter):
        d_new, _a1 = Vacuum._newton_function(d, l1, l2, l3, xmult_i, ceff_i)
        step = abs((d - d_new) / d)
        d = d_new
        if step <= tol:
            break
    return d


def _duct_diameter_residual_for_contract(solution, l1, l2, l3, xmult_i, ceff_i):
    """`Tier2Contract.residual`'s `(solution, **kwargs) -> array` shape."""
    return duct_diameter_residual(solution, l1, l2, l3, xmult_i, ceff_i)


def _duct_diameter_samples():
    """Fuzzed duct geometries, plus the one lifted from `test_old_model`'s solve.

    `tests/unit/models/test_vacuum.py::TestVacuum::test_old_model`'s helium-species
    (`i=2`) Newton solve -- extracted by instrumenting
    `Vacuum._newton_method_duct_diameter` directly, see `vacuum.md`'s worked example
    for the full derivation and why PROCESS's own reported diameter at that point does
    *not* zero this residual (its `0.01` step tolerance stops one iteration before the
    true root -- exactly the discrepancy this port's tighter default `tol` closes).
    """
    rng = np.random.default_rng(20260818)
    n = 24
    l1 = rng.uniform(0.5, 3.0, size=n)
    l2 = rng.uniform(0.5, 6.0, size=n)
    l3 = np.full(n, 2.0)
    xmult_i = rng.choice(np.asarray(XMULT), size=n)
    ceff_i = 10 ** rng.uniform(-1.0, 3.0, size=n)
    samples = [
        Sample(
            MappingProxyType({
                "l1": float(l1[i]),
                "l2": float(l2[i]),
                "l3": float(l3[i]),
                "xmult_i": float(xmult_i[i]),
                "ceff_i": float(ceff_i[i]),
            }),
            "synthetic",
            f"duct-{i}",
        )
        for i in range(n)
    ]
    samples.append(
        legacy_sample(
            "helium-duct-old-model",
            l1=0.4 + 0.63812,
            l2=0.4 + 4.0,
            l3=2.0,
            xmult_i=0.378,
            ceff_i=3.7718486393739226,
        )
    )
    return samples


class TestSolveDuctDiameter(Tier2Contract):
    """`solve_duct_diameter` -> the isolated duct-diameter Newton solve.

    No value-agreement test by construction (`Tier2Contract`) -- PROCESS's own
    `0.01`-relative-step stopping criterion is not a considered accuracy target (see
    `solve_duct_diameter`'s docstring and `vacuum.md`'s worked example), so its answer
    is not ground truth here any more than `intersect`'s 100-iteration fixed-Newton
    answer was for `coils.py`.
    """

    audit_record = "models/vacuum.md"
    reference = staticmethod(_reference_solve_duct_diameter)
    ported = solve_duct_diameter
    residual = staticmethod(_duct_diameter_residual_for_contract)

    samples = _duct_diameter_samples()


class _NewtonRootFindDriver(AbstractDriver):
    """Test-only `AbstractDriver` for `RootFind`, wrapping the exact algorithm
    `solve_duct_diameter` already uses: `jax.grad`-based Newton inside a
    `jax.lax.while_loop`, same default `max_iter=100`/`tol=1e-10`, same single fixed
    `d = 1.0` start when no guess is supplied.

    Exists purely so `TestDuctDiameterRootFind` below gets a real converged number out
    of `DuctDiameterRootFind`'s structural, undriven `RootFind` declaration -- per
    `evaluate.py`'s own docstring for `AbstractDriver`, this is "a concrete
    `AbstractDriver` wrapping the exact Newton scheme this codebase already uses,
    purely for test purposes". Only handles a single scalar unknown, same shape as
    `solve_duct_diameter` itself -- not a general-purpose Newton driver.
    """

    drives = RootFind

    max_iter: int = 100
    tol: float = 1e-10

    def __call__(self, conditions, data):
        start = data.get(Start)
        d0 = jnp.asarray(start[0]) if start is not None else jnp.asarray(1.0)

        def residual_fn(d):
            (r,) = conditions(d)
            return r

        def cond(carry):
            _d, step, it = carry
            return jnp.logical_and(it < self.max_iter, step > self.tol)

        def body(carry):
            d, _step, it = carry
            f = residual_fn(d)
            df = jax.grad(residual_fn)(d)
            d_new = d - f / df
            step = jnp.abs((d - d_new) / d)
            return (d_new, step, it + 1)

        init = (d0, jnp.asarray(jnp.inf), jnp.asarray(0))
        d, _step, _it = jax.lax.while_loop(cond, body, init)
        return (d,)


def _duct_diameter_env(kw):
    """`sample.kwargs` for a `_duct_diameter_samples()` point, as a
    `DuctDiameterRootFind` env (every minted `VarPath` but the unknown itself).
    """
    return {
        vpath(".vacuum.l1"): jnp.asarray(kw["l1"]),
        vpath(".vacuum.l2"): jnp.asarray(kw["l2"]),
        vpath(".vacuum.l3"): jnp.asarray(kw["l3"]),
        vpath(".vacuum.xmult_i"): jnp.asarray(kw["xmult_i"]),
        vpath(".vacuum.ceff_i"): jnp.asarray(kw["ceff_i"]),
    }


# `vacuum.DuctDiameterRootFind`: the structural `ImplicitFunction` counterpart to
# `solve_duct_diameter`, per `vacuum.md`'s discussion. Not a `Tier1Contract`/
# `Tier2Contract` case -- there is no PROCESS reference for a node PROCESS itself
# doesn't have -- so three narrower checks instead: the graph it declares assembles
# (`to_graph`), driving it with `_NewtonRootFindDriver` (the same algorithm
# `solve_duct_diameter` uses) reaches the same answer `solve_duct_diameter` does on
# every sample `TestSolveDuctDiameter` already exercises, and that converged answer
# actually zeroes the defining equation.


def test_duct_diameter_root_find_builds_cleanly():
    """The two minted nodes -- the residual body and the `RootFind` problem -- build
    into one `Graph` with no errors, and are wired to each other as
    `ImplicitFunction`'s docstring promises: the problem owns the unknown and reads
    exactly what the body owns (its residual).
    """
    d = DuctDiameterRootFind()
    g = to_graph(DuctDiameterRootFind)

    assert set(g.nodes) == {d.name, d.problem_name}
    body, problem = g[d.name], g[d.problem_name]
    assert isinstance(body, CallableNode)
    assert isinstance(problem, RootFind)
    assert problem.owns == (vpath(".vacuum.d_duct"),)
    assert problem.reads == body.owns
    assert body.reads == (
        vpath(".vacuum.d_duct"),
        vpath(".vacuum.l1"),
        vpath(".vacuum.l2"),
        vpath(".vacuum.l3"),
        vpath(".vacuum.xmult_i"),
        vpath(".vacuum.ceff_i"),
    )


def test_duct_diameter_root_find_drive_matches_solve_duct_diameter():
    """Driving the declared block with the same Newton scheme `solve_duct_diameter`
    runs internally reaches the same converged diameter, for every sample
    `TestSolveDuctDiameter` exercises (the 24 fuzzed geometries plus the
    `test_old_model` legacy point).
    """
    d = DuctDiameterRootFind()
    schedule = Schedule(
        Blocking.scc(Assign(d.problem_name, _NewtonRootFindDriver()).apply(to_graph(d)))
    )

    for sample in _duct_diameter_samples():
        kw = sample.kwargs
        out = schedule.run(path_map(_duct_diameter_env(kw)))
        expected = solve_duct_diameter(
            kw["l1"], kw["l2"], kw["l3"], kw["xmult_i"], kw["ceff_i"]
        )
        assert float(out[vpath(".vacuum.d_duct")]) == pytest.approx(
            float(expected), rel=1e-9, abs=1e-12
        ), sample.id


def test_duct_diameter_root_find_drive_zeroes_the_residual():
    """The converged diameter actually satisfies the defining equation -- the same
    residual-based pass criterion `Tier2Contract` uses, checked directly here since
    this is not a contract case.
    """
    d = DuctDiameterRootFind()
    schedule = Schedule(
        Blocking.scc(Assign(d.problem_name, _NewtonRootFindDriver()).apply(to_graph(d)))
    )

    sample = _duct_diameter_samples()[-1]  # the test_old_model legacy point
    kw = sample.kwargs
    out = schedule.run(path_map(_duct_diameter_env(kw)))
    residual = duct_diameter_residual(out[vpath(".vacuum.d_duct")], **kw)
    assert abs(float(residual)) < 1e-8


# `vacuum.DuctFeasibility`: "find a feasible `ceff_i`", `solve_duct_geometry`'s outer
# 10%-shrink loop's real problem shape -- see that function's own module-level comment
# and `DuctFeasibility`'s own docstring. Joins with `DuctDiameterRootFind`'s `RootFind`
# problem into one combined block; these checks prove that join is real (the two
# assemble into one cycle, `+` produces the algebraically-joined `Feasibility`, and the
# joined block is actually drivable to a feasible point) -- not just that `Feasibility`
# the type exists.


def _duct_feasibility_graph():
    """`DuctFeasibility` (a bare `problem.py` `DeclaredNode`, not a `NodalDeclaration`,
    so it carries no class-derived name the way `DuctFeasibilityConditions`/
    `DuctDiameterRootFind` do) assembled together with `DuctFeasibilityConditions` and
    `DuctDiameterRootFind` via `to_graph`'s `{name: NodeDefinition}` mapping form --
    added upstream in `cottax` specifically for this shape, see `DuctFeasibility`'s own
    docstring.
    """
    return to_graph(
        DuctFeasibilityConditions(),
        DuctDiameterRootFind(),
        {"DuctFeasibility": DuctFeasibility},
    )


def test_duct_feasibility_forms_one_combined_cycle_with_the_root_find():
    """`DuctFeasibility` (owns `.vacuum.ceff_i`), `DuctFeasibilityConditions` (reads
    `.vacuum.d_duct`/`ceff_i`, owns the two inequality residuals `DuctFeasibility`
    reads), and `DuctDiameterRootFind` (owns `.vacuum.d_duct`, reads `.vacuum.ceff_i`
    indirectly through `duct_diameter_residual`'s own `ceff_i` argument) close a single
    4-node cycle -- the same "Shape A" cross-node-cycle shape
    `WindingPackIntersectInputs`/`Intersect`/`WindingPackTotalSizePost` already
    established in `coils/calculate.py`, not a self-loop on any one node.
    """
    graph = _duct_feasibility_graph()
    assert len(graph.definitions) == 4
    assert not graph.is_acyclic
    (cycle,) = graph.cycles
    assert {n.path_str() for n in cycle} == {
        "['DuctFeasibility']",
        "['DuctFeasibilityConditions']",
        "['DuctDiameterRootFind']",
        "^problem['DuctDiameterRootFind']",
    }


def test_duct_feasibility_joins_algebraically_with_the_root_find_problem():
    """`DuctFeasibility + DuctDiameterRootFind`'s own `RootFind` problem node produces
    exactly the `Feasibility` `problem.py`'s join rule promises: the `RootFind`'s one
    unknown (`d_duct`) joins `design`, its one residual joins `equalities`, and
    `DuctFeasibility`'s own two inequalities are untouched -- `Feasibility.__add__`'s
    `RootFind` branch, checked directly rather than trusted from `~/jaxgraph`'s own
    tests alone, since this is the first real (non-toy) instance of that join in this
    codebase.
    """
    root_find_problem = _duct_feasibility_graph()[DuctDiameterRootFind().problem_name]
    assert isinstance(root_find_problem, RootFind)

    joined = DuctFeasibility + root_find_problem
    assert isinstance(joined, Feasibility)
    assert joined.design == DuctFeasibility.design + root_find_problem.outputs
    assert joined.equalities == root_find_problem.inputs
    assert joined.inequalities == DuctFeasibility.inequalities


class _MeritFunctionFeasibilityDriver(AbstractDriver):
    """Test-only `AbstractDriver` answering `Feasibility` by the reduction its own
    docstring names as the standard move: stack the equality residual with `relu` of
    the two inequality residuals, and drive the resulting 3-vector to zero as an
    ordinary least-squares problem (`optx.LevenbergMarquardt`) over the 2 unknowns
    (`ceff_i`, `d_duct`) -- a `RootFind`-shaped reduction, not a new algorithm family,
    exactly as `Residualise` does the analogous reduction for `FixedPoint`.

    Assumes exactly one equality followed by two inequalities, in that order (this
    driver's own `drives = Feasibility` only promises the *type* is answered correctly,
    not that it is generic over every possible `Feasibility` shape -- same scoping
    `IntersectBisectionNewtonPolish`/`_GenericBisectionRootFind` accept in
    `coils/test_calculate.py`).
    """

    drives = Feasibility

    def __call__(self, conditions, data):
        start = data.get(Start)
        x0 = (
            jnp.asarray(start, dtype=float)
            if start is not None
            else jnp.asarray([1.0, 1.0])
        )

        def merit(x, _):
            equality, fits, floor = conditions(x[0], x[1])
            return jnp.stack([equality, jax.nn.relu(fits), jax.nn.relu(floor)])

        solution = optx.least_squares(
            merit,
            optx.LevenbergMarquardt(rtol=1e-10, atol=1e-10),
            x0,
            throw=False,
        )
        return tuple(solution.value)


def test_duct_feasibility_drives_to_a_point_that_satisfies_every_condition():
    """Driving the joined block reaches a point where the equality residual is ~0 and
    both inequality residuals are `<= 0` -- feasible, not merely converged. Uses the
    `test_old_model` legacy geometry (same point `test_duct_diameter_root_find_drive_
    zeroes_the_residual` above already verifies `DuctDiameterRootFind` on) with
    `a1max`/`s_i` chosen so the unconstrained root of `duct_diameter_residual` at the
    sample's own `ceff_i` already satisfies both inequalities (`a1max` double the
    resulting duct's cross-section; `s_i` half `ceff_i`).

    The equality alone (one equation, two unknowns `ceff_i`/`d_duct`) is underdetermined
    -- an entire curve of `(ceff_i, d_duct)` pairs satisfies it -- and the merit-function
    reduction has no gradient pressure to prefer any particular point on that curve while
    both inequalities stay inactive (`relu` of a negative residual contributes nothing to
    the loss), so the least-squares solve is free to land anywhere feasible along it,
    including exactly on an inequality's boundary (confirmed empirically: it lands with
    `pumping_speed_floor_residual == 0.0` here, not strictly interior) -- feasible either
    way, so the check below is `<=`, not `<`. Discovering the specific point an *outer*
    preference (e.g. "smallest `ceff_i`") would pick is a harder problem this test does
    not attempt -- that is exactly the gap a real objective (`Optimise`, not
    `Feasibility`) would close, not a limitation of this reduction.
    """
    sample = _duct_diameter_samples()[-1]
    kw = sample.kwargs
    d_at_sample_ceff = solve_duct_diameter(
        kw["l1"], kw["l2"], kw["l3"], kw["xmult_i"], kw["ceff_i"]
    )
    a1max = 2.0 * 0.25 * jnp.pi * d_at_sample_ceff * d_at_sample_ceff
    s_i = 0.5 * kw["ceff_i"]

    graph = _duct_feasibility_graph()
    name = NodePath((DictKey("DuctFeasibility"),))
    root_find_problem = graph[DuctDiameterRootFind().problem_name]
    joined = DuctFeasibility + root_find_problem

    body = graph.runnable  # every plain node, problem nodes dropped
    merged = Graph(path_map({**dict(body.definitions), name: joined}))

    schedule = Schedule(
        Blocking.scc(Assign(name, _MeritFunctionFeasibilityDriver()).apply(merged))
    )
    env = {
        vpath(".vacuum.l1"): jnp.asarray(kw["l1"]),
        vpath(".vacuum.l2"): jnp.asarray(kw["l2"]),
        vpath(".vacuum.l3"): jnp.asarray(kw["l3"]),
        vpath(".vacuum.xmult_i"): jnp.asarray(kw["xmult_i"]),
        vpath(".vacuum.a1max"): a1max,
        vpath(".vacuum.s_i"): s_i,
    }
    out = schedule.run(path_map(env))

    d_duct = out[vpath(".vacuum.d_duct")]
    ceff_i = out[vpath(".vacuum.ceff_i")]
    # `d_duct` itself is not checked against `d_at_sample_ceff` -- the underdetermined
    # equality has no reason to keep `ceff_i` near the sample's own value (see docstring
    # above), so the converged `d_duct` legitimately differs. `a1max` was built from
    # `d_at_sample_ceff` only to guarantee a feasible region exists at all.
    assert float(d_duct) > 0.0
    assert float(
        duct_diameter_residual(
            d_duct, kw["l1"], kw["l2"], kw["l3"], kw["xmult_i"], ceff_i
        )
    ) == pytest.approx(0.0, abs=1e-6)
    assert float(duct_fits_residual(d_duct, a1max)) <= 1e-8
    assert float(pumping_speed_floor_residual(ceff_i, s_i)) <= 1e-8


def _reference_vacuum_pumping_old(
    p_fusion_total_mw,
    rmajor,
    rminor,
    dsol,
    a_plasma_surface,
    vol_plasma,
    dr_shld_outboard,
    dr_shld_inboard,
    dr_tf_inboard,
    ritf,
    n_tf_coils,
    t_plant_pulse_dwell,
    n_divertors,
    qtorus,
    gasld,
    i_vac_pump_dwell,
    i_vacuum_pump_type,
    pres_vv_chamber_base,
    pres_div_chamber_burn,
    outgrat_fw,
    t_plant_pulse_coil_precharge,
):
    """Call PROCESS's `Vacuum.vacuum` through the port's (diagnostic) signature.

    Returns the same 7-tuple `_solve_vacuum_pumping_old` does: PROCESS's own five
    outputs (`pumpn` *not* rounded, matching that function), plus `imax`/`ceff_used` --
    the species that ended up governing the design, and the target conductance its
    reported diameter was actually solved for. Neither is a `data` field PROCESS
    writes, so they're recovered by instrumenting
    `Vacuum._newton_method_duct_diameter` (bound on this one instance, restored
    implicitly when the instance is discarded) to record its `(i, ceff[i])` on every
    call -- the last call before `vacuum()` returns is exactly the one that produced
    the final `dimax`/`imax`, same reasoning as `vacuum.md`'s worked example.

    `nplasma`/`temp_vv_chamber_gas_burn_end` are fixed, arbitrary values here (`1e20`
    K, `300` K) -- proven not to affect any of `vacuum()`'s five outputs, see
    `calculate_vacuum_pumping_old`'s docstring and `vacuum.md`.
    `temp_plasma_electron_vol_avg_kev` is set but never actually read on this path
    (only reachable through a non-convergence log message this instrumentation never
    triggers in-sample).
    """
    data = DataStructure()
    data.vacuum.i_vac_pump_dwell = i_vac_pump_dwell
    data.vacuum.i_vacuum_pump_type = i_vacuum_pump_type
    data.vacuum.pres_vv_chamber_base = pres_vv_chamber_base
    data.vacuum.pres_div_chamber_burn = pres_div_chamber_burn
    data.vacuum.outgrat_fw = outgrat_fw
    data.vacuum.temp_vv_chamber_gas_burn_end = 300.0
    data.times.t_plant_pulse_coil_precharge = t_plant_pulse_coil_precharge
    data.physics.p_fusion_total_mw = p_fusion_total_mw
    data.physics.temp_plasma_electron_vol_avg_kev = 15.0

    v = Vacuum()
    v.data = data
    captured = {}
    orig = v._newton_method_duct_diameter

    def _capture(d, i, s, xmult, l1, l2, l3, ntf, r0, aw, ritf_, thcsh, ceff):
        captured["imax"] = i
        captured["ceff_used"] = ceff[i]
        return orig(d, i, s, xmult, l1, l2, l3, ntf, r0, aw, ritf_, thcsh, ceff)

    v._newton_method_duct_diameter = _capture

    pumpn, nduct, dlscalc, mvdsh, dimax = v.vacuum(
        pfusmw=p_fusion_total_mw,
        r0=rmajor,
        aw=rminor,
        dsol=dsol,
        plasma_sarea=a_plasma_surface,
        plasma_vol=vol_plasma,
        thshldo=dr_shld_outboard,
        thshldi=dr_shld_inboard,
        thtf=dr_tf_inboard,
        ritf=ritf,
        n_tf_coils=n_tf_coils,
        t_plant_pulse_dwell=t_plant_pulse_dwell,
        nplasma=1.0e20,
        ndiv=n_divertors,
        qtorus=qtorus,
        gasld=gasld,
        output=False,
    )
    return pumpn, nduct, dlscalc, mvdsh, dimax, captured["imax"], captured["ceff_used"]


def _vacuum_pumping_old_residual(solution, **kwargs):
    """`Tier2Contract.residual`'s `(solution, **kwargs) -> array` shape.

    The defining equation of the whole design: the winning species' duct conductance,
    at the diameter the design actually reports, should equal the target conductance
    that diameter was solved for (`duct_diameter_residual`, reused from
    `solve_duct_diameter`'s own unit -- same equation, same residual). `l1`/`l2`/`l3`
    are recomputed from the same inputs `_solve_vacuum_pumping_old` itself derives them
    from (see that function's body) rather than threaded through as extra outputs.
    """
    _pumpn, _nduct, _dlscalc, _mvdsh, dimax, imax, ceff_used = solution
    l1 = kwargs["dr_shld_outboard"] + kwargs["dr_tf_inboard"]
    l2 = kwargs["dr_shld_outboard"] + 4.0
    l3 = 2.0
    xmult_i = jnp.asarray(XMULT)[jnp.asarray(imax).astype(int)]
    return duct_diameter_residual(dimax, l1, l2, l3, xmult_i, ceff_used)


def _vacuum_pumping_old_samples():
    """The `test_old_model` legacy point, plus fuzzed geometries at each switch combo.

    `i_vac_pump_dwell` (0/1/2) and `i_vacuum_pump_type` (0/1) are genuine switches
    (`_audit/naming_convention.md`) -- fuzzing them as continuous values between their
    bounds would draw meaningless non-integer switch settings, so each combination gets
    its own small fixed-switch fuzz batch instead (`fuzz_samples(..., fixed=...)`,
    same pattern `coils.py`'s `_intersect_samples` uses for a curated, non-CLI-driven
    sample set). Geometry bounds are centred on `test_old_model`'s own scale
    (`rmajor~8`, `rminor~3.3`, `n_tf_coils~18`) with enough spread to exercise more
    than one governing species (`imax`), verified empirically not to hit the
    "space limited" (`nflag = 1`) regime PROCESS's own duct-sizing model can enter --
    see `vacuum.md`'s open questions for why that regime is excluded here rather than
    exercised.
    """
    bounds = {
        "p_fusion_total_mw": (500.0, 4000.0),
        "rmajor": (6.0, 20.0),
        "rminor": (1.5, 5.0),
        "dsol": (0.05, 0.5),
        "a_plasma_surface": (400.0, 2500.0),
        "vol_plasma": (400.0, 4500.0),
        "dr_shld_outboard": (0.2, 0.8),
        "dr_shld_inboard": (0.05, 0.4),
        "dr_tf_inboard": (0.3, 1.2),
        "ritf": (3.0, 12.0),
        "n_tf_coils": (12.0, 40.0),
        "t_plant_pulse_dwell": (100.0, 1800.0),
        "n_divertors": (1.0, 2.0),
        "qtorus": (0.0, 0.0),
        "gasld": (1.0e-6, 1.0e-4),
        "pres_vv_chamber_base": (1.0e-5, 1.0e-3),
        "pres_div_chamber_burn": (0.1, 0.8),
        "outgrat_fw": (1.0e-9, 1.0e-7),
        "t_plant_pulse_coil_precharge": (10.0, 60.0),
    }
    samples = []
    for seed, dwell, pump_type in ((1, 0, 0), (2, 1, 1), (3, 2, 0)):
        samples += fuzz_samples(
            bounds,
            4,
            seed,
            fixed={"i_vac_pump_dwell": dwell, "i_vacuum_pump_type": pump_type},
        )
    samples.append(
        legacy_sample(
            "old-model-g-l-nb-ti",
            p_fusion_total_mw=2115.3899563651776,
            rmajor=8.1386000000000003,
            rminor=3.2664151549205331,
            dsol=0.22500000000000003,
            a_plasma_surface=1468.3151179059994,
            vol_plasma=2907.2299918381777,
            dr_shld_outboard=0.40000000000000002,
            dr_shld_inboard=0.12000000000000001,
            dr_tf_inboard=0.63812000000000002,
            ritf=3.6371848450794664,
            n_tf_coils=18,
            t_plant_pulse_dwell=1800.0,
            n_divertors=1,
            qtorus=0.0,
            gasld=2.7947500651998464e-05,
            i_vac_pump_dwell=0,
            i_vacuum_pump_type=1,
            pres_vv_chamber_base=0.00050000000000000001,
            pres_div_chamber_burn=0.35999999999999999,
            outgrat_fw=1.3000000000000001e-08,
            t_plant_pulse_coil_precharge=30.0,
        )
    )
    return samples


class TestVacuumPumpingOld(Tier2Contract):
    """`Vacuum.vacuum` (+ `Vacuum.run()`'s rounding) -> `_solve_vacuum_pumping_old`.

    No value-agreement test by construction. `vacuum.md`'s worked example shows
    PROCESS's own reported `dimax`, on the exact `test_old_model` legacy point, does
    not itself zero this unit's defining equation -- `duct_conductance(dimax, ...)`
    comes out ~0.16% away from the `ceff` it was meant to solve for, purely because
    PROCESS's own `0.01` relative-step stopping criterion exits one Newton step early.
    `calculate_vacuum_pumping_old` (the audited, five-output public function this
    file's node would wrap) is not tested directly -- it is a thin, trivially-correct
    slice-and-round wrapper over `_solve_vacuum_pumping_old`, tested here.
    """

    audit_record = "models/vacuum.md"
    reference = staticmethod(_reference_vacuum_pumping_old)
    ported = _solve_vacuum_pumping_old
    residual = staticmethod(_vacuum_pumping_old_residual)

    samples = _vacuum_pumping_old_samples()


def _reference_vacuum_pumping_old_from_fields(
    p_fusion_total_mw,
    rmajor,
    rminor,
    dr_fw_plasma_gap_inboard,
    dr_fw_plasma_gap_outboard,
    a_plasma_surface,
    vol_plasma,
    dr_shld_outboard,
    dr_shld_inboard,
    dr_tf_inboard,
    r_shld_inboard_inner,
    dr_shld_vv_gap_inboard,
    dr_vv_inboard,
    n_tf_coils,
    t_plant_pulse_dwell,
    n_divertors,
    qtorus,
    molflow_plasma_fuelling_required,
    m_fuel_amu,
    i_vac_pump_dwell,
    i_vacuum_pump_type,
    pres_vv_chamber_base,
    pres_div_chamber_burn,
    outgrat_fw,
    t_plant_pulse_coil_precharge,
):
    """`_reference_vacuum_pumping_old`, with `dsol`/`ritf`/`gasld` derived from the raw
    fields exactly as `Vacuum.run()` does -- what confirms `vacuum.py`'s
    `_derive_vacuum_pumping_old_locals` inlining is exact, not merely plausible.
    """
    dsol = 0.5 * (dr_fw_plasma_gap_inboard + dr_fw_plasma_gap_outboard)
    ritf = r_shld_inboard_inner - dr_shld_vv_gap_inboard - dr_vv_inboard
    gasld = 2.0 * molflow_plasma_fuelling_required * m_fuel_amu * constants.UMASS
    return _reference_vacuum_pumping_old(
        p_fusion_total_mw,
        rmajor,
        rminor,
        dsol,
        a_plasma_surface,
        vol_plasma,
        dr_shld_outboard,
        dr_shld_inboard,
        dr_tf_inboard,
        ritf,
        n_tf_coils,
        t_plant_pulse_dwell,
        n_divertors,
        qtorus,
        gasld,
        i_vac_pump_dwell,
        i_vacuum_pump_type,
        pres_vv_chamber_base,
        pres_div_chamber_burn,
        outgrat_fw,
        t_plant_pulse_coil_precharge,
    )


def _vacuum_pumping_old_from_fields_samples():
    """Field-level cousin of `_vacuum_pumping_old_samples`: same geometry/switch scale,
    `dsol`/`ritf`/`gasld` replaced by the raw fields they're derived from. The legacy
    point's raw fields are chosen so the derived locals land close to
    `_vacuum_pumping_old_samples`'s own legacy point (not required to match exactly --
    the reference is computed fresh from these fields either way).
    """
    bounds = {
        "p_fusion_total_mw": (500.0, 4000.0),
        "rmajor": (6.0, 20.0),
        "rminor": (1.5, 5.0),
        "dr_fw_plasma_gap_inboard": (0.02, 0.5),
        "dr_fw_plasma_gap_outboard": (0.02, 0.5),
        "a_plasma_surface": (400.0, 2500.0),
        "vol_plasma": (400.0, 4500.0),
        "dr_shld_outboard": (0.2, 0.8),
        "dr_shld_inboard": (0.05, 0.4),
        "dr_tf_inboard": (0.3, 1.2),
        "r_shld_inboard_inner": (3.0, 12.0),
        "dr_shld_vv_gap_inboard": (0.0, 0.0),
        "dr_vv_inboard": (0.0, 0.0),
        "n_tf_coils": (12.0, 40.0),
        "t_plant_pulse_dwell": (100.0, 1800.0),
        "n_divertors": (1.0, 2.0),
        "qtorus": (0.0, 0.0),
        "molflow_plasma_fuelling_required": (1.0e21, 5.0e22),
        "m_fuel_amu": (2.0, 3.0),
        "pres_vv_chamber_base": (1.0e-5, 1.0e-3),
        "pres_div_chamber_burn": (0.1, 0.8),
        "outgrat_fw": (1.0e-9, 1.0e-7),
        "t_plant_pulse_coil_precharge": (10.0, 60.0),
    }
    samples = []
    for seed, dwell, pump_type in ((11, 0, 0), (12, 1, 1)):
        samples += fuzz_samples(
            bounds,
            4,
            seed,
            fixed={"i_vac_pump_dwell": dwell, "i_vacuum_pump_type": pump_type},
        )
    samples.append(
        legacy_sample(
            "old-model-g-l-nb-ti-from-fields",
            p_fusion_total_mw=2115.3899563651776,
            rmajor=8.1386000000000003,
            rminor=3.2664151549205331,
            dr_fw_plasma_gap_inboard=0.22500000000000003,
            dr_fw_plasma_gap_outboard=0.22500000000000003,
            a_plasma_surface=1468.3151179059994,
            vol_plasma=2907.2299918381777,
            dr_shld_outboard=0.40000000000000002,
            dr_shld_inboard=0.12000000000000001,
            dr_tf_inboard=0.63812000000000002,
            r_shld_inboard_inner=3.8621848450794664,
            dr_shld_vv_gap_inboard=0.155,
            dr_vv_inboard=0.07,
            n_tf_coils=18,
            t_plant_pulse_dwell=1800.0,
            n_divertors=1,
            qtorus=0.0,
            molflow_plasma_fuelling_required=3.3658206e21,
            m_fuel_amu=2.5,
            i_vac_pump_dwell=0,
            i_vacuum_pump_type=1,
            pres_vv_chamber_base=0.00050000000000000001,
            pres_div_chamber_burn=0.35999999999999999,
            outgrat_fw=1.3000000000000001e-08,
            t_plant_pulse_coil_precharge=30.0,
        )
    )
    return samples


class TestVacuumPumpingOldFromFields(Tier2Contract):
    """`_solve_vacuum_pumping_old_from_fields`: the same algorithm as
    `TestVacuumPumpingOld`, exercised through the raw `.build.*`/`.physics.*` signature
    `VacuumOld`'s node actually uses, instead of PROCESS's own `dsol`/`ritf`/`gasld`-
    taking `vacuum()` boundary. Same residual, same tolerance -- this class exists to
    confirm `_derive_vacuum_pumping_old_locals`'s inlining is exact, not to re-test the
    duct-sizing algorithm itself (that's `TestVacuumPumpingOld`'s job).
    """

    audit_record = "models/vacuum.md"
    reference = staticmethod(_reference_vacuum_pumping_old_from_fields)
    ported = _solve_vacuum_pumping_old_from_fields
    residual = staticmethod(_vacuum_pumping_old_residual)

    samples = _vacuum_pumping_old_from_fields_samples()


# ---------------------------------------------------------------------------
# `VacuumVessel` -- reached on the tokamak, not the stellarator (see module docstring
# and `vacuum.md`'s tokamak-scope addendum). Both are already real PROCESS
# `@staticmethod`s, so no `DataStructure` adapter is needed -- they are diffed against
# `VacuumVessel`'s own methods directly.
# ---------------------------------------------------------------------------


class TestCalculateVesselHalfHeight(Tier1Contract):
    """`calculate_vessel_half_height` -> `VacuumVessel.calculate_vessel_half_height`
    at `n_divertors == 1` (single null).
    """

    audit_record = "models/vacuum.md"
    reference = staticmethod(
        lambda **kw: VacuumVessel.calculate_vessel_half_height(n_divertors=1, **kw)
    )
    ported = calculate_vessel_half_height

    fuzz_bounds = {
        "z_tf_inside_half": (2.0, 15.0),
        "dz_shld_vv_gap": (0.05, 0.5),
        "dz_vv_lower": (0.1, 1.0),
        "dz_blkt_upper": (0.1, 1.5),
        "dz_shld_upper": (0.1, 1.0),
        "z_plasma_xpoint_upper": (1.0, 10.0),
        "dr_fw_plasma_gap_inboard": (0.05, 0.5),
        "dr_fw_plasma_gap_outboard": (0.05, 0.5),
        "dr_fw_inboard": (0.01, 0.1),
        "dr_fw_outboard": (0.01, 0.1),
    }


class TestCalculateVesselHalfHeightDoubleNull(Tier1Contract):
    """`calculate_vessel_half_height_double_null` ->
    `VacuumVessel.calculate_vessel_half_height` at `n_divertors == 2` (double null).

    The seven parameters this arm does not read are passed as `nan`, not `0.0`, so
    "PROCESS does not look at these" is executed rather than asserted: were the branch
    not taken, the reference would return `nan` and the comparison would fail instead of
    agreeing on a zero.
    """

    audit_record = "models/vacuum.md"
    reference = staticmethod(
        lambda **kw: VacuumVessel.calculate_vessel_half_height(
            n_divertors=2,
            dz_blkt_upper=np.nan,
            dz_shld_upper=np.nan,
            z_plasma_xpoint_upper=np.nan,
            dr_fw_plasma_gap_inboard=np.nan,
            dr_fw_plasma_gap_outboard=np.nan,
            dr_fw_inboard=np.nan,
            dr_fw_outboard=np.nan,
            **kw,
        )
    )
    ported = calculate_vessel_half_height_double_null

    fuzz_bounds = {
        "z_tf_inside_half": (2.0, 15.0),
        "dz_shld_vv_gap": (0.05, 0.5),
        "dz_vv_lower": (0.1, 1.0),
    }


class TestCalculateEllipticalVesselVolumes(Tier1Contract):
    """`calculate_elliptical_vessel_volumes` -> `VacuumVessel.
    calculate_elliptical_vessel_volumes`, unchanged signature.
    """

    audit_record = "models/vacuum.md"
    reference = staticmethod(VacuumVessel.calculate_elliptical_vessel_volumes)
    ported = calculate_elliptical_vessel_volumes

    # tests/unit/models/test_vacuum.py::test_elliptical_vessel_volumes, verbatim.
    samples = [
        legacy_sample(
            "elliptical-vessel-legacy",
            rmajor=8.0,
            rminor=2.6666666666666665,
            triang=0.5,
            r_shld_inboard_inner=4.083333333333334,
            r_shld_outboard_outer=12.716666666666667,
            dz_vv_half=7.5032752487304135,
            dr_vv_inboard=0.30000000000000004,
            dr_vv_outboard=0.30000000000000004,
            dz_vv_upper=0.30000000000000004,
            dz_vv_lower=0.30000000000000004,
        ),
    ]

    fuzz_bounds = {
        "rmajor": (2.0, 20.0),
        "rminor": (0.5, 5.0),
        "triang": (0.0, 0.8),
        "r_shld_inboard_inner": (0.5, 8.0),
        "r_shld_outboard_outer": (5.0, 20.0),
        "dz_vv_half": (1.0, 15.0),
        "dr_vv_inboard": (0.05, 1.0),
        "dr_vv_outboard": (0.05, 1.0),
        "dz_vv_upper": (0.05, 1.0),
        "dz_vv_lower": (0.05, 1.0),
    }


def _reference_vacuum_vessel_outputs(
    z_tf_inside_half,
    dz_shld_vv_gap,
    dz_vv_lower,
    dz_blkt_upper,
    dz_shld_upper,
    z_plasma_xpoint_upper,
    dr_fw_plasma_gap_inboard,
    dr_fw_plasma_gap_outboard,
    dr_fw_inboard,
    dr_fw_outboard,
    rmajor,
    rminor,
    triang,
    r_shld_inboard_inner,
    r_shld_outboard_outer,
    dr_vv_inboard,
    dr_vv_outboard,
    dz_vv_upper,
    fvoldw,
    den_steel,
):
    """Call PROCESS's real `VacuumVessel.run()` through the port's signature, at the
    one switch combination the port bakes in (`itart=0`, `i_fw_blkt_vv_shape=2` --
    both already PROCESS defaults; `n_divertors=1`).
    """
    data = DataStructure()
    data.build.z_tf_inside_half = z_tf_inside_half
    data.build.dz_shld_vv_gap = dz_shld_vv_gap
    data.build.dz_vv_lower = dz_vv_lower
    data.divertor.n_divertors = 1
    data.build.dz_blkt_upper = dz_blkt_upper
    data.build.dz_shld_upper = dz_shld_upper
    data.build.z_plasma_xpoint_upper = z_plasma_xpoint_upper
    data.build.dr_fw_plasma_gap_inboard = dr_fw_plasma_gap_inboard
    data.build.dr_fw_plasma_gap_outboard = dr_fw_plasma_gap_outboard
    data.build.dr_fw_inboard = dr_fw_inboard
    data.build.dr_fw_outboard = dr_fw_outboard
    data.physics.itart = 0
    data.fwbs.i_fw_blkt_vv_shape = 2
    data.physics.rmajor = rmajor
    data.physics.rminor = rminor
    data.physics.triang = triang
    data.build.r_shld_inboard_inner = r_shld_inboard_inner
    data.build.r_shld_outboard_outer = r_shld_outboard_outer
    data.build.dr_vv_inboard = dr_vv_inboard
    data.build.dr_vv_outboard = dr_vv_outboard
    data.build.dz_vv_upper = dz_vv_upper
    data.fwbs.fvoldw = fvoldw
    data.fwbs.den_steel = den_steel

    vv = VacuumVessel()
    vv.data = data
    vv.run()

    return (
        vv.data.blanket.dz_vv_half,
        vv.data.blanket.vol_vv_inboard,
        vv.data.blanket.vol_vv_outboard,
        vv.data.fwbs.vol_vv,
        vv.data.fwbs.m_vv,
    )


class TestCalculateVacuumVesselOutputs(Tier1Contract):
    """`calculate_vacuum_vessel_outputs` -> real `VacuumVessel.run()`, the whole live
    pipeline (own contract for `calculate_vacuum_vessel_mass` too -- it has no
    isolated PROCESS function, see that function's docstring).
    """

    audit_record = "models/vacuum.md"
    reference = _reference_vacuum_vessel_outputs
    ported = calculate_vacuum_vessel_outputs

    fuzz_bounds = {
        "z_tf_inside_half": (2.0, 15.0),
        "dz_shld_vv_gap": (0.05, 0.5),
        "dz_vv_lower": (0.1, 1.0),
        "dz_blkt_upper": (0.1, 1.5),
        "dz_shld_upper": (0.1, 1.0),
        "z_plasma_xpoint_upper": (1.0, 10.0),
        "dr_fw_plasma_gap_inboard": (0.05, 0.5),
        "dr_fw_plasma_gap_outboard": (0.05, 0.5),
        "dr_fw_inboard": (0.01, 0.1),
        "dr_fw_outboard": (0.01, 0.1),
        "rmajor": (2.0, 20.0),
        "rminor": (0.5, 5.0),
        "triang": (0.0, 0.8),
        "r_shld_inboard_inner": (0.5, 8.0),
        "r_shld_outboard_outer": (5.0, 20.0),
        "dr_vv_inboard": (0.05, 1.0),
        "dr_vv_outboard": (0.05, 1.0),
        "dz_vv_upper": (0.05, 1.0),
        "fvoldw": (0.5, 1.5),
        "den_steel": (6000.0, 9000.0),
    }


def _reference_vacuum_vessel_outputs_double_null(
    z_tf_inside_half,
    dz_shld_vv_gap,
    dz_vv_lower,
    rmajor,
    rminor,
    triang,
    r_shld_inboard_inner,
    r_shld_outboard_outer,
    dr_vv_inboard,
    dr_vv_outboard,
    dz_vv_upper,
    fvoldw,
    den_steel,
):
    """Real `VacuumVessel.run()` at `n_divertors = 2`, otherwise the same configuration
    as `_reference_vacuum_vessel_outputs`.

    The seven `.build` fields this arm does not read are seeded with `nan`.
    `process/models/vacuum.py` reads every one of them in exactly one place --
    `:744-756`, the arguments of the half-height call -- so on this arm nothing may touch
    them, and a `nan` proves it rather than a zero hiding it.
    """
    data = DataStructure()
    data.build.z_tf_inside_half = z_tf_inside_half
    data.build.dz_shld_vv_gap = dz_shld_vv_gap
    data.build.dz_vv_lower = dz_vv_lower
    data.divertor.n_divertors = 2
    data.build.dz_blkt_upper = np.nan
    data.build.dz_shld_upper = np.nan
    data.build.z_plasma_xpoint_upper = np.nan
    data.build.dr_fw_plasma_gap_inboard = np.nan
    data.build.dr_fw_plasma_gap_outboard = np.nan
    data.build.dr_fw_inboard = np.nan
    data.build.dr_fw_outboard = np.nan
    data.physics.itart = 0
    data.fwbs.i_fw_blkt_vv_shape = 2
    data.physics.rmajor = rmajor
    data.physics.rminor = rminor
    data.physics.triang = triang
    data.build.r_shld_inboard_inner = r_shld_inboard_inner
    data.build.r_shld_outboard_outer = r_shld_outboard_outer
    data.build.dr_vv_inboard = dr_vv_inboard
    data.build.dr_vv_outboard = dr_vv_outboard
    data.build.dz_vv_upper = dz_vv_upper
    data.fwbs.fvoldw = fvoldw
    data.fwbs.den_steel = den_steel

    vv = VacuumVessel()
    vv.data = data
    vv.run()

    return (
        vv.data.blanket.dz_vv_half,
        vv.data.blanket.vol_vv_inboard,
        vv.data.blanket.vol_vv_outboard,
        vv.data.fwbs.vol_vv,
        vv.data.fwbs.m_vv,
    )


class TestCalculateVacuumVesselOutputsDoubleNull(Tier1Contract):
    """`calculate_vacuum_vessel_outputs_double_null` -> real `VacuumVessel.run()` at
    `n_divertors == 2`.

    The single-null contract's box minus the seven inputs this arm does not read, so the
    two composites are exercised over the same geometry and the difference between them
    is the branch.
    """

    audit_record = "models/vacuum.md"
    reference = _reference_vacuum_vessel_outputs_double_null
    ported = calculate_vacuum_vessel_outputs_double_null

    fuzz_bounds = {
        "z_tf_inside_half": (2.0, 15.0),
        "dz_shld_vv_gap": (0.05, 0.5),
        "dz_vv_lower": (0.1, 1.0),
        "rmajor": (2.0, 20.0),
        "rminor": (0.5, 5.0),
        "triang": (0.0, 0.8),
        "r_shld_inboard_inner": (0.5, 8.0),
        "r_shld_outboard_outer": (5.0, 20.0),
        "dr_vv_inboard": (0.05, 1.0),
        "dr_vv_outboard": (0.05, 1.0),
        "dz_vv_upper": (0.05, 1.0),
        "fvoldw": (0.5, 1.5),
        "den_steel": (6000.0, 9000.0),
    }


def _reference_dshaped_vessel_volumes(**kwargs):
    """`VacuumVessel.calculate_dshaped_vessel_volumes`, already a bare
    `@staticmethod` -- no adapter, and no `nan` poisoning possible or needed: PROCESS's
    own D-shaped signature simply has no `rmajor`/`rminor`/`triang` parameters.
    """
    return VacuumVessel.calculate_dshaped_vessel_volumes(**kwargs)


class TestCalculateDshapedVesselVolumes(Tier1Contract):
    """`calculate_dshaped_vessel_volumes` -> the same, unchanged."""

    audit_record = "models/vacuum.md"
    reference = _reference_dshaped_vessel_volumes
    ported = calculate_dshaped_vessel_volumes

    fuzz_bounds = {
        "r_shld_inboard_inner": (0.5, 8.0),
        "r_shld_outboard_outer": (9.0, 20.0),
        "dz_vv_half": (1.0, 15.0),
        "dr_vv_inboard": (0.05, 0.4),
        "dr_vv_outboard": (0.05, 1.0),
        "dz_vv_upper": (0.05, 1.0),
        "dz_vv_lower": (0.05, 1.0),
    }
    """`dr_vv_inboard` is capped below the smallest `r_shld_inboard_inner`: it is
    `dshellvol`'s `drin`, whose inboard term `rmajor**2 - (rmajor - drin)**2` needs
    `drin < rmajor` to stay a volume. PROCESS has no guard and both sides would agree on
    the nonsense, so this keeps the draws physical rather than hiding a disagreement."""


def _reference_vacuum_vessel_outputs_dshaped_double_null(
    z_tf_inside_half,
    dz_shld_vv_gap,
    dz_vv_lower,
    r_shld_inboard_inner,
    r_shld_outboard_outer,
    dr_vv_inboard,
    dr_vv_outboard,
    dz_vv_upper,
    fvoldw,
    den_steel,
):
    """Real `VacuumVessel.run()` at `n_divertors = 2` **and** the D-shaped shape arm --
    `spherical_tokamak_eval.IN.DAT`/`st_regression.IN.DAT`'s own configuration.

    **Ten fields are poisoned with `nan`** -- the seven the double-null half-height does
    not read, plus `.physics.rmajor`, `.physics.rminor` and `.physics.triang`, which
    `vacuum.py` reads at `:781-783` only, as arguments of the *elliptical* volume call.
    This adapter therefore executes the claim that the occupant it backs has no
    `.physics` edge at all.

    `itart = 1` **and** `i_fw_blkt_vv_shape = D_SHAPED` are both set, as both ST files
    set both.
    """
    data = DataStructure()
    data.build.z_tf_inside_half = z_tf_inside_half
    data.build.dz_shld_vv_gap = dz_shld_vv_gap
    data.build.dz_vv_lower = dz_vv_lower
    data.divertor.n_divertors = 2
    data.build.dz_blkt_upper = np.nan
    data.build.dz_shld_upper = np.nan
    data.build.z_plasma_xpoint_upper = np.nan
    data.build.dr_fw_plasma_gap_inboard = np.nan
    data.build.dr_fw_plasma_gap_outboard = np.nan
    data.build.dr_fw_inboard = np.nan
    data.build.dr_fw_outboard = np.nan
    data.physics.itart = 1
    data.fwbs.i_fw_blkt_vv_shape = 1
    data.physics.rmajor = np.nan
    data.physics.rminor = np.nan
    data.physics.triang = np.nan
    data.build.r_shld_inboard_inner = r_shld_inboard_inner
    data.build.r_shld_outboard_outer = r_shld_outboard_outer
    data.build.dr_vv_inboard = dr_vv_inboard
    data.build.dr_vv_outboard = dr_vv_outboard
    data.build.dz_vv_upper = dz_vv_upper
    data.fwbs.fvoldw = fvoldw
    data.fwbs.den_steel = den_steel

    vv = VacuumVessel()
    vv.data = data
    vv.run()

    return (
        vv.data.blanket.dz_vv_half,
        vv.data.blanket.vol_vv_inboard,
        vv.data.blanket.vol_vv_outboard,
        vv.data.fwbs.vol_vv,
        vv.data.fwbs.m_vv,
    )


class TestCalculateVacuumVesselOutputsDshapedDoubleNull(Tier1Contract):
    """`calculate_vacuum_vessel_outputs_dshaped_double_null` -> real
    `VacuumVessel.run()` at the D-shaped double-null cell -- what
    `VacuumVesselDShapedDoubleNull` wraps, and the configuration both spherical-tokamak
    input files select.

    Ten inputs against the elliptical single-null contract's twenty, and the box is the
    elliptical double-null one with the three `.physics` entries removed -- so the two
    composites are exercised over the same build geometry and the difference between them
    is the shape branch.
    """

    audit_record = "models/vacuum.md"
    reference = _reference_vacuum_vessel_outputs_dshaped_double_null
    ported = calculate_vacuum_vessel_outputs_dshaped_double_null

    fuzz_bounds = {
        "z_tf_inside_half": (2.0, 15.0),
        "dz_shld_vv_gap": (0.05, 0.5),
        "dz_vv_lower": (0.1, 1.0),
        "r_shld_inboard_inner": (1.5, 8.0),
        "r_shld_outboard_outer": (9.0, 20.0),
        "dr_vv_inboard": (0.05, 1.0),
        "dr_vv_outboard": (0.05, 1.0),
        "dz_vv_upper": (0.05, 1.0),
        "fvoldw": (0.5, 1.5),
        "den_steel": (6000.0, 9000.0),
    }
    """`r_shld_inboard_inner` starts at `1.5` rather than the sibling's `0.5`: on this
    arm it *is* `dshellvol`'s `rmajor`, and `dr_vv_inboard` runs up to `1.0`, so the
    lower bound keeps `drin < rmajor` over the whole box (see
    `TestCalculateDshapedVesselVolumes`). On the elliptical arm the same field is only
    one term of a difference and no such constraint applies."""
