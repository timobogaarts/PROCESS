"""`architectures.driven` on `large_tokamak_nof`: the study's rewrites are the ops they
say they are, the three closures converge at PROCESS's design and reproduce its power
balance, the freezes put the PF coil set back in the first stage, and the two-stage
problem assembles with the shape `decision_kinds_tokamak.md` states.

Every number here is the port's own at PROCESS's converged design (`common.
process_reference`, cached) -- nothing a sample set could move. One model build per
module (~1 min, ~2 GB): the fixtures are module-scoped.
"""

from __future__ import annotations

import numpy as np
import pytest

from functional_process.configurations import kinds_tokamak as kinds
from functional_process.cottax.architectures import (
    driven,
    mdf,
    sand,
    session,
    stages,
)
from functional_process.cottax.architectures.evaluate import without_excluded
from functional_process.cottax.architectures.mda import SCHEME, cut_graph

NAME = "large_tokamak_nof"


@pytest.fixture(scope="module")
def live():
    return session.open_session(NAME)


@pytest.fixture(scope="module")
def process(live):
    """PROCESS's converged design, from the cache `paper_tests/common` fills."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "paper_tests"))
    from common import process_reference  # noqa: PLC0415

    return process_reference(NAME)


@pytest.fixture(scope="module")
def problem_graph(live):
    ref = live.reference
    graph, _c, _n, _r = mdf.mdf_graph(
        cut_graph(without_excluded(live.machine_graph), SCHEME),
        ref.icc,
        ref.n_equality,
        ref.i_figure_merit,
        live.switch_values,
    )
    return graph


def _values(live, process, pairings):
    design = [sand.iteration_variable_path(i).spelling for i in live.reference.ixc]
    closed = set(pairings.values())
    return (
        [process["x"][v] for v in design if v not in closed],
        {v: process["x"].get(v, 75.0) for v in closed},
    )


# ---------------------------------------------------------------- the rewrites


def test_installed_power_rewires_the_cost_node_and_c30(problem_graph):
    g = driven.installed_power(problem_graph)
    cost, c30 = g[driven.COST_NODE], g[driven.CONSTRAINT30]
    assert driven.INSTALLED in cost.reads
    assert driven.USED_ECRH not in cost.reads
    assert driven.INSTALLED in c30.reads
    assert driven.CAP not in c30.reads
    inputs = {v.spelling for v in g.graph.boundary_inputs}
    assert driven.INSTALLED.spelling in inputs
    assert driven.CAP.spelling not in inputs
    # The used power is still what everything else reads.
    readers = [n for n, d in g.definitions.items() if driven.USED_TOTAL in d.reads]
    assert len(readers) >= 5
    assert len(g.nodes) == len(problem_graph.nodes)


def test_beta_limit_factor_redefines_the_wesson_node(problem_graph):
    g = driven.beta_limit_factor(problem_graph)
    node = g[driven.NORM_MAX]
    assert tuple(node.reads) == (driven.LI, driven.F_BETA_NORM_MAX)
    assert tuple(node.owns) == (driven.BETA_NORM_MAX,)
    assert node.fn(0.9, 1.0) == pytest.approx(3.6)  # Wesson's 4 l_i, the file's l_i
    assert node.fn(0.9, 1.2) == pytest.approx(4.32)


# ---------------------------------------------------------------- the closures


def test_process_point_reproduces_the_power_balance(live, process):
    """`c2` closed by the heating power at PROCESS's design asks for the file's 75 MW."""
    dv, cv = _values(live, process, driven.PROCESS_PAIRINGS)
    at = driven.process_point(live, dv, cv)
    assert at["heating"] == pytest.approx(75.0, abs=1e-6)
    assert at["installed"] == pytest.approx(75.0 + at["cd"])
    assert 5.0 < at["cd"] < 20.0
    for cond, (steps, converged, residual, _u) in at["reports"].items():
        assert converged and abs(residual) < 1e-9, cond
    assert float(at["out"][".costs.coe"]) == pytest.approx(
        process["outputs"][".costs.coe"], rel=3e-3
    )
    assert float(at["out"][".heat_transport.p_plant_electric_net_mw"]) == pytest.approx(
        400.0, abs=0.5
    )
    # c62 is inactive at PROCESS's optimum: rho* above the 5 the file requires.
    assert at["conditions"]["^cond.constraints.c62"] < -0.3


def test_three_closures_are_one_square_problem_of_five(live, process):
    built = driven.close(live)
    assert built.report["closed_inequalities"] == ("^cond.constraints.c62",)
    assert len(built.report["inequalities"]) == 22
    (unknowns,) = built.report["closing_problems"].values()
    assert set(unknowns) == {
        driven.HEATING.spelling,
        "^hat.physics.nd_plasma_fuel_ions_vol_avg",
        "^hat.physics.nd_plasma_ions_total_vol_avg",
        ".physics.f_nd_alpha_thermal_electron",
        ".physics.beta_total_vol_avg",
    }
    assert driven.HEATING not in built.design
    dv, cv = _values(live, process, driven.PAIRINGS)
    env = driven.seed(built, dv, cv, installed=84.8)
    _primed, out = mdf.prime(built.problem, env)
    for cond, (steps, converged, residual, value) in built.root_find_reports(
        out
    ).items():
        assert converged and abs(residual) < 1e-9 and steps <= 20, cond
    reports = {c.spelling: r for c, r in built.root_find_reports(out).items()}
    # Held with equality, the helium fraction falls from PROCESS's 0.0857.
    assert 0.05 < reports["^cond.constraints.c62"][3] < 0.08
    assert 0.0 < reports["^cond.constraints.c2"][3] < 75.0


# ---------------------------------------------------------------- the split


@pytest.fixture(scope="module")
def model(live, process):
    dv, cv = _values(live, process, driven.PAIRINGS)
    return driven.two_stage(
        live,
        n=4,
        design_values=dv,
        closing_values=cv,
        installed=84.8,
        freezes=driven.FREEZES,
    )


def test_two_stage_has_the_shape_the_census_states(model):
    assert [v.spelling for v in model.design][-1] == driven.INSTALLED.spelling
    assert len(model.design) == 18  # 20 ixc - 3 closing + hfact sampled ... + installed
    assert ".physics.hfact" not in [v.spelling for v in model.design]
    assert all(k in [v.spelling for v in model.design] for k in kinds.KNOBS)
    sampled = [b.path for b in model.beliefs if b.path != "dummy"]
    assert len(sampled) == 8
    assert model.dropped == ()
    assert len(model.constraints) == 24  # 22 + the coil capacity + the heating bound
    assert len(model.unknowns) == 5
    counts = model.stages.counts
    assert counts[stages.Stage.FIRST].nodes == 135
    assert model.stages.n_nodes == 277


def test_freezes_put_the_coil_set_in_the_first_stage(model):
    first = {n.spelling for n in model.stages.first}
    for node in (
        ".tokamak.pf_coil.sizes",
        ".tokamak.pf_coil.masses",
        ".tokamak.cryostat",
        ".tokamak.structure",
        ".buildings.sizing",
        ".costs.pf_magnet_cost",
        ".tokamak.ccfe_hcpb.component_masses",
        ".tokamak.cs_coil.turn_geometry",
    ):
        assert node in first, node
    varying = {n.spelling for n in model.stages.varying}
    for node in (
        ".tokamak.pf_coil.equilibrium_currents",
        ".tokamak.pulse.burn_time",
        ".tokamak.cs_coil.stresses",
        ".Built.c_pf_cs_coils_peak_ma",
        ".Constraint13",
    ):
        assert node in varying, node
    frozen = model.closed.report["freezes"]
    assert len(frozen) == 24
    assert (
        frozen[".pf_coil.c_pf_cs_coils_peak_ma"]["condition"]
        == "^cond.built.pf_coil.c_pf_cs_coils_peak_ma"
    )


def test_nominal_row_sits_on_the_built_coil_currents(model):
    """At the nominal, the world's currents are the built ones: the capacity is 0."""
    from functional_process.cottax.architectures import ouu  # noqa: PLC0415

    fns = ouu.make(model, hoist=False)
    y = np.asarray(fns["run_batch"](model.x0, model.theta, model.starts0))
    L = model.layout
    names = [c.spelling for c in model.constraints]
    k = names.index("^cond.built.pf_coil.c_pf_cs_coils_peak_ma")
    assert abs(y[-1, L["c_g"][0] + k]) < 1e-9
    assert y[-1, L["c_conv"]] > 0.5
    assert 300 < y[-1, L["c_coe"]] < 600
