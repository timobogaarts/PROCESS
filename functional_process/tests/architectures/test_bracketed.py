"""`closing.bracketed()` on `stellarator_helias`: the power balance closed by the
density inside the MDA (`kinds.PAIRINGS["one"]`, `flatten=False`, the cycle's Picard
nested inside the root find) by `drivers.BracketedRootDriver`, which converges from
any start.

What the flattened safeguarded Newton (`test_closing`) cannot do, and this driver is
for: the handoff (`plans/handoff_2026-09-17.md`, action 3) found that evaluating a far
design from cold Newton starts fails 35-42 % of belief samples, a solver artefact,
and asked for a bracketed 1-D root with the Picards nested so any design can be
evaluated from cold. Measured here: at the nominal design from 0.3x and 3x the root,
at the handoff's far design (variant B2's alpha-0.9 design), and under `jax.vmap`
over perturbed designs, against the flattened Newton from the same starts.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cottax.execution.drivers.kinds import Converged, Steps
from cottax.pytree.path import PathMap

from functional_process.configurations import kinds
from functional_process.cottax.architectures import closing, mdf, session
from functional_process.cottax.architectures.drivers import (
    BRACKET_CONVERGED,
    BracketedRootDriver,
    Status,
)
from functional_process.tests.architectures.test_closing import (
    DENSITY,
    NAME,
    assert_closed_at_process_design,
    primed_at_process_design,
)

DENSITY_PATH = ".physics.nd_plasma_electrons_vol_avg"
RMAJOR = ".physics.rmajor"
BFIELD = ".physics.b_plasma_toroidal_on_axis"

FAR_DESIGN = {
    RMAJOR: 30.0,
    BFIELD: 8.62,
    ".physics.temp_plasma_electron_vol_avg_kev": 12.9,
    ".tfcoil.t_tf_superconductor_quench": 50.0,
    ".tfcoil.f_a_tf_turn_cable_copper": 0.67,
    ".physics.f_nd_alpha_thermal_electron": 0.122,
}
"""The handoff's far design: variant B2 at alpha 0.9 -- R at its 30 m bound, B 8.6 T,
T_e 12.9 keV, the quench time at its 50 s bound, copper 0.67, alpha fraction 0.12.
`hfact` stays at the file's own value."""


@pytest.fixture(scope="module")
def live():
    """The configuration, opened once for the module."""
    return session.open_session(NAME)


@pytest.fixture(scope="module")
def bracketed(live):
    """The nested closure under the bracketed driver -- built once, so every test
    shares its compiled schedule.
    """
    return closing.close(
        live, kinds.PAIRINGS["one"], flatten=False, driver=closing.bracketed()
    )


@pytest.fixture(scope="module")
def flattened(live):
    """`test_closing`'s form: the flattened safeguarded Newton over three unknowns."""
    return closing.close(live, kinds.PAIRINGS["one"])


def place_of(built):
    """The one closing problem's path."""
    (place,) = set(built.places.values())
    return place


def verdict(built, out) -> tuple[int, bool, int, float, float]:
    """`(steps, converged, status, residual, density)` of the closing problem."""
    node = built.graph[place_of(built)]
    (cond,) = built.places
    return (
        int(np.asarray(mdf.verdict(out, Steps, node))),
        bool(np.asarray(mdf.verdict(out, Converged, node))),
        int(np.asarray(mdf.verdict(out, Status, node))),
        float(np.asarray(out[cond])),
        float(np.asarray(out[built.pairings[cond]])),
    )


def solve_from(built, design: dict, density: float):
    """Seed at `design` (spellings over the file's cold values) with the density's
    start at `density`, prime, and return the verdict.
    """
    cold = built.session.reference.cold
    env = closing.seed(built, cold)
    values = dict(design)
    design_values = [
        values.get(v.spelling, float(np.asarray(env[v]))) for v in built.design
    ]
    env = closing.seed(
        built, cold, design_values=design_values, closing_values={DENSITY_PATH: density}
    )
    _env, out = mdf.prime(built.problem, env)
    return verdict(built, out)


# ---------------------------------------------------------------- (a) the same root


def test_driver_is_the_bracketed_one_with_the_files_bounds(bracketed):
    """`close` fills the closing variable's bounds from the reference, as the first
    bracket; the problem keeps its one unknown, the Picard nested inside.
    """
    driver = bracketed.graph[place_of(bracketed)].driver
    assert isinstance(driver, BracketedRootDriver)
    assert driver.rtol == closing.NEWTON_TOL
    (var,) = bracketed.closing
    assert driver.bracket_for(var) == (3e19, 3e20)
    assert bracketed.report["flattened"] is False
    assert bracketed.report["closing_problems"] == {".Close.c2": (DENSITY_PATH,)}
    assert driver.reports == (Steps, Converged, Status)
    # `mdf.traceable_drivers` clears a `SeededNewtonDriver`'s `seed`; this driver has
    # none and passes through unchanged.
    assert mdf.traceable_drivers({place_of(bracketed): driver}) == {
        place_of(bracketed): driver
    }


def test_bracketed_closes_at_process_design_to_the_flattened_density(bracketed):
    """At PROCESS's converged design, from PROCESS's density: the same root as the
    flattened Newton (`test_closing.DENSITY`), the bounds bracketing at once.
    """
    _env, out = primed_at_process_design(bracketed)
    assert_closed_at_process_design(bracketed, out, steps_at_most=6)
    _steps, converged, status, residual, density = verdict(bracketed, out)
    assert converged
    assert status == BRACKET_CONVERGED
    assert density == pytest.approx(DENSITY, rel=1e-8)
    assert abs(residual) < 2 * closing.NEWTON_TOL


def test_bracketed_closes_from_the_cold_start_to_the_flattened_root(
    bracketed, flattened, live
):
    """From the file's own design and density (what `mdf.solve` starts from), where
    the root lies **outside** the file's bounds and the bracket must widen.
    """
    cold = live.reference.cold
    _env, out = mdf.prime(bracketed.problem, closing.seed(bracketed, cold))
    steps, converged, status, residual, density = verdict(bracketed, out)
    assert converged
    assert status == BRACKET_CONVERGED
    assert abs(residual) < 2 * closing.NEWTON_TOL
    _env, reference = mdf.prime(flattened.problem, closing.seed(flattened, cold))
    _n_steps, n_converged, _s, _r, n_density = verdict(flattened, reference)
    assert n_converged
    assert density == pytest.approx(n_density, rel=1e-8)
    assert not (3e19 <= density <= 3e20), "the root is outside the file's bounds here"
    assert steps <= 12


def test_warm_start_at_the_root_takes_one_evaluation(bracketed, live):
    """A start already within tolerance is neither bracketed nor stepped."""
    cold = live.reference.cold
    _env, out = mdf.prime(bracketed.problem, closing.seed(bracketed, cold))
    _s, _c, _st, _r, root = verdict(bracketed, out)
    steps, converged, status, _residual, density = solve_from(bracketed, {}, root)
    assert converged
    assert status == BRACKET_CONVERGED
    assert steps == 1
    assert density == root


# ---------------------------------------------------------------- (b) far starts


@pytest.fixture(scope="module")
def nominal_root(bracketed, live):
    """The density at the file's own design, from the bracketed closure."""
    _env, out = mdf.prime(
        bracketed.problem, closing.seed(bracketed, live.reference.cold)
    )
    return verdict(bracketed, out)[-1]


FAR_STARTS = [
    ("nominal", 0.3),
    ("nominal", 3.0),
    ("far", 1.0),
    ("far", 0.3),
    ("far", 3.0),
]
"""`(design, start as a multiple of the nominal root)`: the far design from the
nominal root is the handoff's case -- a belief sample's design evaluated from the
deterministic root."""


@pytest.mark.parametrize(("design", "factor"), FAR_STARTS)
def test_bracketed_converges_from_far_starts(
    bracketed, flattened, nominal_root, design, factor
):
    """The bracketed closure converges from every start, and to one root per design;
    the flattened Newton's verdict from the same start is reported, not asserted.
    """
    values = FAR_DESIGN if design == "far" else {}
    start = factor * nominal_root
    b_steps, b_converged, b_status, b_residual, b_density = solve_from(
        bracketed, values, start
    )
    n_steps, n_converged, n_status, n_residual, n_density = solve_from(
        flattened, values, start
    )
    print(
        f"\n{design:8s} x{factor:<4g} bracketed: steps {b_steps:3d} converged "
        f"{b_converged!s:5s} status {b_status} density {b_density:.6e} |r| "
        f"{abs(b_residual):.1e} | flattened Newton: steps {n_steps:3d} converged "
        f"{n_converged!s:5s} status {n_status} density {n_density:.6e} |r| "
        f"{abs(n_residual):.1e}"
    )
    assert b_converged
    assert b_status == BRACKET_CONVERGED
    assert abs(b_residual) < 2 * closing.NEWTON_TOL
    # One root per design: the nominal's is `nominal_root`; the far design's is the
    # bracketed answer from its nominal-root start, which the other starts must share.
    reference = (
        nominal_root
        if design == "nominal"
        else solve_from(bracketed, FAR_DESIGN, nominal_root)[-1]
    )
    assert b_density == pytest.approx(reference, rel=1e-8)
    if n_converged:
        assert n_density == pytest.approx(reference, rel=1e-6)


# ---------------------------------------------------------------- (c) vmap, jacfwd


def density_of(built, live):
    """`f(rmajor, b) -> (density, steps, converged, status)`, the whole schedule run
    at the file's cold values otherwise -- traceable, so `vmap` and `jacfwd` apply.
    """
    schedule = built.problem.eager
    env = mdf._inputs_only(built.problem, closing.seed(built, live.reference.cold))
    at = {v.spelling: v for v in [*env, *schedule.unknowns]}
    node = built.graph[place_of(built)]

    def one(rmajor, b_field):
        values = dict(env)
        values[at[RMAJOR]] = rmajor
        values[at[BFIELD]] = b_field
        out = schedule.run(PathMap(values))
        return (
            out[at[DENSITY_PATH]],
            mdf.verdict(out, Steps, node),
            mdf.verdict(out, Converged, node),
            mdf.verdict(out, Status, node),
        )

    return one, float(env[at[RMAJOR]]), float(env[at[BFIELD]])


def perturbed(r0, b0, n=16):
    """`n` designs at +-20 % on R and B, one fixed draw."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    return (
        r0 * (1 + 0.2 * jax.random.uniform(k1, (n,), minval=-1, maxval=1)),
        b0 * (1 + 0.2 * jax.random.uniform(k2, (n,), minval=-1, maxval=1)),
    )


def test_vmap_over_perturbed_designs_converges_every_row(bracketed, flattened, live):
    """16 designs at +-20 % on R and B from the file's density: every row of the
    bracketed closure converges; the flattened Newton's rows are reported.
    """
    one, r0, b0 = density_of(bracketed, live)
    rs, bs = perturbed(r0, b0)
    began = time.perf_counter()
    density, steps, converged, status = jax.jit(jax.vmap(one))(rs, bs)
    elapsed = time.perf_counter() - began
    print(
        f"\nvmap x16 bracketed: {int(np.sum(converged))}/16 converged, steps "
        f"{np.asarray(steps).tolist()}, compile+run {elapsed:.1f} s"
    )
    assert np.all(np.isfinite(np.asarray(density)))
    assert np.all(np.asarray(converged))
    assert np.all(np.asarray(status) == BRACKET_CONVERGED)
    newton, _r, _b = density_of(flattened, live)
    n_density, _n_steps, n_converged, n_status = jax.jit(jax.vmap(newton))(rs, bs)
    print(
        f"vmap x16 flattened Newton: {int(np.sum(n_converged))}/16 converged, status "
        f"{np.asarray(n_status).tolist()}"
    )
    # Where the Newton did converge, it found the same root.
    agree = np.asarray(n_converged)
    assert np.allclose(
        np.asarray(density)[agree], np.asarray(n_density)[agree], rtol=1e-8
    )


def test_jacfwd_of_the_density_through_the_driver_matches_a_central_difference(
    bracketed, live
):
    """`jax.jacfwd` through `lax.custom_root` and the nested Picard's implicit
    adjoint: finite, and the central difference's value to 1e-4.
    """
    one, r0, b0 = density_of(bracketed, live)
    b = jnp.asarray(b0)

    def density(r):
        return one(r, b)[0]

    slope = float(jax.jit(jax.jacfwd(density))(jnp.asarray(r0)))
    assert np.isfinite(slope)
    h = 1e-4 * r0
    at = jax.jit(density)
    central = (float(at(jnp.asarray(r0 + h))) - float(at(jnp.asarray(r0 - h)))) / (2 * h)
    print(f"\nd density / d rmajor: jacfwd {slope:.9e}, central {central:.9e}")
    assert slope == pytest.approx(central, rel=1e-4)


# ---------------------------------------------------------------- (d) wall time


def test_wall_time_against_the_flattened_newton(bracketed, flattened, live):
    """Steps and ms per call of the whole schedule at the nominal, warm, single and
    batched by 16: measured and printed, not asserted -- the handoff expected the
    nested form to be slower.
    """
    rows = []
    for label, built in [
        ("bracketed nested", bracketed),
        ("flattened Newton", flattened),
    ]:
        one, r0, b0 = density_of(built, live)
        single = jax.jit(one)
        r, b = jnp.asarray(r0), jnp.asarray(b0)
        _d, steps, _c, _s = single(r, b)
        times = []
        for _ in range(20):
            began = time.perf_counter()
            single(r, b)[0].block_until_ready()
            times.append(time.perf_counter() - began)
        batched = jax.jit(jax.vmap(one))
        rs, bs = perturbed(r0, b0)
        batched(rs, bs)[0].block_until_ready()
        began = time.perf_counter()
        for _ in range(5):
            batched(rs, bs)[0].block_until_ready()
        per_batch = (time.perf_counter() - began) / 5
        rows.append((label, int(steps), np.median(times) * 1e3, per_batch * 1e3))
    print()
    for label, steps, ms, batch_ms in rows:
        print(
            f"{label:18s} steps {steps:3d}  {ms:6.2f} ms/call single  "
            f"{batch_ms:6.1f} ms per 16-row batch"
        )
    assert all(np.isfinite(ms) for _l, _s, ms, _b in rows)
