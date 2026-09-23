"""The architecture-example notebooks run, and what they conclude has not moved.

Each notebook in `functional_process/architecture_examples/<case>/` is run in a fresh
kernel and its final `RESULT` dict is read back. The checks are the notebook's headline
claims (the solve converged, the objective is the reference one, the structure is the
one the text describes), not its every number. Outputs are not written back, so a test
run never rewrites the checked-in notebooks.

`tier4`: a full assembly and solve per notebook, about two minutes in all:

    $PY -m pytest functional_process/tests/examples -m tier4
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from functional_process.architecture_examples import notebook_tools

CASES = Path(notebook_tools.__file__).resolve().parent

REFERENCE_COE = 1.2184414
"""COE at the joint optimum of `stellarator_helias` -- SAND and IDF reach it; the
two-driver case is a few percent above it by construction."""

pytestmark = pytest.mark.tier4


def _run(case: str) -> dict:
    nb = CASES / case / f"{case}.ipynb"
    _, answers = notebook_tools.execute_notebook(nb, output=None, user_expressions={"result": "RESULT"})
    return ast.literal_eval(answers["result"])


@pytest.fixture(scope="module")
def mda():
    return _run("mda_gauss_seidel")


@pytest.fixture(scope="module")
def mdf():
    return _run("mdf")


@pytest.fixture(scope="module")
def idf():
    return _run("idf")


@pytest.fixture(scope="module")
def sand():
    return _run("sand")


@pytest.fixture(scope="module")
def two_opt():
    return _run("two_opt_driver")


def test_gauss_seidel_mda_reaches_the_same_fixed_point(mda):
    assert mda["compared"] > 500          # every scalar both MDAs produce
    assert mda["worst_relative_difference"] < 1e-5
    # Every Picard-driven block reported its steps, each within the budget.
    assert len(mda["steps"]) >= 2
    assert all(1 <= n <= 256 for n in mda["steps"].values())
    # The three schemes all ran, and Jacobi needs at least as many sweeps as
    # Gauss-Seidel in the same order.
    assert set(mda["recipes"]) == {"jacobi", "gauss_seidel", "gauss_seidel_minimal"}
    assert mda["recipes"]["jacobi"]["total"] >= mda["recipes"]["gauss_seidel"]["total"]


def test_mdf_converges(mdf):
    assert mdf["converged"], mdf
    assert mdf["max_eq"] < 1e-6
    assert mdf["min_ie"] > -1e-6
    assert mdf["objf"] == pytest.approx(REFERENCE_COE, rel=2e-3)


def test_sand_converges_to_the_reference_optimum(sand):
    assert sand["status"] == 0, sand       # VMCON's own convergence test
    assert sand["max_eq"] < 1e-5
    assert sand["objf"] == pytest.approx(REFERENCE_COE, rel=1e-6)
    # Eight design variables plus the unknowns of every absorbed solve.
    assert sand["unknowns"] > 8
    assert sand["conditions"] > 15


def test_idf_converges_to_the_same_optimum(idf, sand):
    assert idf["status"] == 0, idf
    assert idf["max_eq"] < 1e-6
    assert idf["objf"] == pytest.approx(sand["objf"], rel=1e-6)
    # Only the coupling is lifted: fewer unknowns than SAND, more than the design.
    assert 8 < idf["unknowns"] < sand["unknowns"]
    # The models' own solves are nested inside the optimiser, not absorbed into it.
    assert len(idf["nested"]) >= 1
    for i, x in idf["design"].items():
        assert x == pytest.approx(sand["design"][i], rel=1e-4), i


def test_two_sequential_optimisers_converge_above_the_joint_optimum(two_opt):
    assert two_opt["sand"]["status"] == "converged"
    assert set(two_opt["iterations"]) == {"OptMagnet", "OptPlasma"}
    assert all(n > 0 for n in two_opt["iterations"].values())
    # Feasible but not optimal: the decoupling costs a few percent of COE.
    assert two_opt["sand"]["coe"] == pytest.approx(REFERENCE_COE, rel=1e-6)
    assert 1.0 < two_opt["coe"] / two_opt["sand"]["coe"] < 1.15
    assert two_opt["design"][2] == pytest.approx(5.0, abs=1e-6)   # the b_t >= 5 T bound is active
