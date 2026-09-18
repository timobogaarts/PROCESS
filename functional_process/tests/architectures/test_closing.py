"""`architectures.closing` on `stellarator_helias`: the power balance closed by the
density inside the MDA (`kinds.PAIRINGS["one"]`), flattened and nested, at PROCESS's
own converged design.

PROCESS's answer for this file (its VMCON over all eight design variables) is the
oracle: at that design the port's root find over the density must land on PROCESS's
density, since the power balance is one equation in one unknown there. The cost of
electricity is pinned to the port's own nominal, which sits 1.7 % above PROCESS's --
`deliberate_divergences.md` on the net electric power, which is `c16`'s 0.018 at this
design and not the closure's doing.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from functional_process.configurations import kinds
from functional_process.cottax.architectures import closing, mdf, session

NAME = "stellarator_helias"

PROCESS_DESIGN = {
    ".physics.b_plasma_toroidal_on_axis": 4.703724863747863,
    ".physics.rmajor": 26.694370149249416,
    ".physics.temp_plasma_electron_vol_avg_kev": 5.673208206328849,
    ".physics.hfact": 1.0555868996252038,
    ".tfcoil.t_tf_superconductor_quench": 35.319929504548796,
    ".tfcoil.f_a_tf_turn_cable_copper": 0.7380005121197526,
    ".physics.f_nd_alpha_thermal_electron": 0.03359040614117263,
}
"""PROCESS's converged `x` on this file, minus the density the closure owns."""

PROCESS_DENSITY = 1.7459597848266706e20
"""PROCESS's converged density: the closing variable's starting guess."""

DENSITY = 1.745959789987205e20
"""The density the closure finds at that design -- PROCESS's to 3e-9, the residual
`c2` of PROCESS's own answer evaluated by the port."""

PROCESS_COE = 121.49167845171463
PORT_COE = 123.59794465431297
"""The port's cost of electricity at that design, and PROCESS's."""


def by_spelling(env):
    """`{spelling: VarPath}` over an env's keys."""
    return {v.spelling: v for v in env}


@pytest.fixture(scope="module")
def live():
    """The configuration, opened once for the module."""
    return session.open_session(NAME)


def primed_at_process_design(built):
    """`closing.seed` at PROCESS's design and density, then `mdf.prime`: the env and
    the run's output.
    """
    env = closing.seed(
        built,
        built.session.reference.cold,
        design_values=[PROCESS_DESIGN[v.spelling] for v in built.design],
        closing_values={".physics.nd_plasma_electrons_vol_avg": PROCESS_DENSITY},
    )
    return mdf.prime(built.problem, env)


def assert_closed_at_process_design(built, out, steps_at_most):
    """The closure converged, on PROCESS's density, at the port's own cost."""
    reports = built.root_find_reports(out)
    (cond,) = reports
    steps, converged, residual, density = reports[cond]
    assert converged
    assert steps <= steps_at_most
    assert abs(residual) < 1e-8
    assert density == pytest.approx(DENSITY, rel=1e-6)
    at = by_spelling(out)
    coe = float(np.asarray(out[at[".costs.coe"]]))
    assert coe == pytest.approx(PORT_COE, rel=1e-6)
    assert coe == pytest.approx(PROCESS_COE, rel=2e-2)


def test_flattened_one_closes_the_density_at_process_design(live):
    """`PAIRINGS["one"]`, flattened: one Newton over the density and two copies."""
    began = time.perf_counter()
    built = closing.close(live, kinds.PAIRINGS["one"])
    _env, out = primed_at_process_design(built)
    elapsed = time.perf_counter() - began
    assert_closed_at_process_design(built, out, steps_at_most=10)
    assert elapsed < 120, f"build + seed + prime took {elapsed:.0f} s"


def test_flattened_one_closes_from_the_cold_start(live):
    """What `mdf.solve` starts from: the file's own design, the copies from an MDA."""
    built = closing.close(live, kinds.PAIRINGS["one"])
    env = closing.seed(built, live.reference.cold)
    _env, out = mdf.prime(built.problem, env)
    (report,) = built.root_find_reports(out).values()
    steps, converged, residual, density = report
    assert converged
    assert steps <= built.graph[built.places[next(iter(built.places))]].driver.max_steps
    assert abs(residual) < 1e-8
    assert 1e19 < density < 1e21


def test_flattened_one_has_the_shape_the_plan_states(live):
    """What closed what, in one combined problem of three unknowns; seven kept."""
    built = closing.close(live, kinds.PAIRINGS["one"])
    report = built.report
    assert report["closing"] == kinds.PAIRINGS["one"]
    assert report["flattened"] is True
    # One combined problem: the density plus the two cut copies of its cycle.
    (place,) = report["closing_problems"]
    assert place == "^problem.Close.c2"
    unknowns = report["closing_problems"][place]
    assert len(unknowns) == 3
    assert unknowns[0] == ".physics.nd_plasma_electrons_vol_avg"
    assert all(u.startswith("^hat.") for u in unknowns[1:])
    # The outer problem keeps the other seven, in `ixc` order, and no equality.
    assert [v.spelling for v in built.design] == list(PROCESS_DESIGN)
    assert built.problem.n_equality == 0
    assert built.problem.conditions[0] == report["objective"]
    assert len(built.problem.conditions) == 1 + built.problem.n_inequality
    assert built.problem.eager is built.problem.traceable
    assert built.closing == tuple(built.pairings.values())


def test_nested_one_closes_to_the_same_density(live):
    """`flatten=False`: the cycle's Picards nested inside the density's root find."""
    built = closing.close(live, kinds.PAIRINGS["one"], flatten=False)
    assert built.report["flattened"] is False
    assert built.report["closing_problems"] == {
        ".Close.c2": (".physics.nd_plasma_electrons_vol_avg",)
    }
    _env, out = primed_at_process_design(built)
    assert_closed_at_process_design(built, out, steps_at_most=10)


def test_two_pairings_on_one_cycle_are_flattened_together_or_refused(live):
    """`PAIRINGS["two"]`: c2 and c16 share a cycle, so they become one problem."""
    built = closing.close(live, kinds.PAIRINGS["two"])
    (place,) = built.report["closing_problems"]
    unknowns = built.report["closing_problems"][place]
    assert set(kinds.PAIRINGS["two"].values()) <= set(unknowns)
    assert len(set(built.places.values())) == 1
    assert len(built.design) == 6
    with pytest.raises(ValueError, match="shares a cycle"):
        closing.close(live, kinds.PAIRINGS["two"], flatten=False)


def test_nested_blocking_states_the_optimise_around_the_closure(live):
    """The structural picture: the `Optimise` outermost, the closure inside it."""
    built = closing.close(live, kinds.PAIRINGS["one"])
    lines = closing.describe(closing.nested_blocking(built))
    assert "optimise .Opt" in lines[0]
    assert any("root-find ^problem.Close.c2" in line for line in lines[1:])
