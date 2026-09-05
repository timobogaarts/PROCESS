"""Harness cases for `functional_process/cottax/cs_fatigue.py` -- two tier-1 contracts.

`surface_stress_intensity_factor` is diffed directly against `CsFatigue`'s own
`@staticmethod` (itself `@njit`-compiled, but numba and JAX both just run it as plain
Python/numpy-shaped arithmetic here since neither `self` nor `self.data` is touched) --
no adapter needed.

Its legacy sample is `tests/unit/models/test_cs_fatigue.py::test_surface_stress_
intensity_factor`'s point, verbatim -- genuinely legacy, and it exercises the `a <= c`
formula (`a=0.00089`, `c=0.00267`). `fuzz_bounds` deliberately straddles `a == c` (both
`a` and `c` drawn from the same range) so `test_gradient_agreement` also exercises the
`a > c` formula and the `jnp.where` switch-over itself -- see the port module's docstring
for why this is safe (the discarded branch's own non-finite sub-expressions do not leak
into the selected branch's value or gradient).

`calculate_n_cycle` (added 2026-08-30) does need an adapter, and `_reference_n_cycle`
below is where the "close the `data` back-door" claim stops being an assertion:
`ncycle`'s seven material and safety-factor coefficients are `self.data.cs_fatigue.*`
reads on PROCESS's side and arguments on the port's, so the adapter writes each one onto
a fresh `DataStructure` from the sample. Leaving them at their defaults instead would
make seven of the port's twelve arguments untested.
"""

from functional_process._harness import Tier1Contract, legacy_sample
from functional_process.cottax.cs_fatigue import (
    calculate_n_cycle,
    surface_stress_intensity_factor,
)
from process.core.model import DataStructure
from process.models.cs_fatigue import CsFatigue


def _reference_n_cycle(
    max_hoop_stress,
    residual_stress,
    t_crack_vertical,
    dz_cs_turn_conduit,
    dr_cs_turn_conduit,
    paris_coefficient,
    paris_power_law,
    walker_coefficient,
    sf_vertical_crack,
    sf_radial_crack,
    fracture_toughness,
    sf_fast_fracture,
):
    """`CsFatigue.ncycle`, with its seven coefficient reads bound onto a `DataStructure`.

    **This adapter is where the port's "close the `data` back-door" claim is tested
    rather than asserted** (`_harness.contracts.PortContract.reference`). `ncycle` reads
    all seven off `self.data.cs_fatigue`, so they are *written there* per call from the
    sample's own kwargs -- not left at their dataclass defaults, which would let the
    port disagree on any of them without a single test noticing.

    Only the first return value is compared. PROCESS's second, `t_crack_radial`, is
    `3 * t_crack_vertical` computed before the loop, and the port deliberately does not
    produce it -- see `calculate_n_cycle`'s docstring for why owning an `IN.DAT` input
    to report a derived value is the one thing this node must not do.
    """
    model = CsFatigue()
    model.data = DataStructure()
    model.data.cs_fatigue.paris_coefficient = paris_coefficient
    model.data.cs_fatigue.paris_power_law = paris_power_law
    model.data.cs_fatigue.walker_coefficient = walker_coefficient
    model.data.cs_fatigue.sf_vertical_crack = sf_vertical_crack
    model.data.cs_fatigue.sf_radial_crack = sf_radial_crack
    model.data.cs_fatigue.fracture_toughness = fracture_toughness
    model.data.cs_fatigue.sf_fast_fracture = sf_fast_fracture
    n_cycle, _t_crack_radial = model.ncycle(
        max_hoop_stress=max_hoop_stress,
        residual_stress=residual_stress,
        t_crack_vertical=t_crack_vertical,
        dz_cs_turn_conduit=dz_cs_turn_conduit,
        dr_cs_turn_conduit=dr_cs_turn_conduit,
    )
    return n_cycle


class TestSurfaceStressIntensityFactor(Tier1Contract):
    """`surface_stress_intensity_factor` -> the same, unchanged (branch on `a <= c`
    replaced by `jnp.where`).
    """

    audit_record = "models/cs_fatigue.md"
    reference = staticmethod(CsFatigue.surface_stress_intensity_factor)
    ported = surface_stress_intensity_factor

    samples = [
        legacy_sample(
            "surface_stress_intensity_factor-baseline_2018",
            hoop_stress=659.99351867335338,
            t=0.0063104538380405924,
            w=0.0063104538380405924,
            a=0.00088999999999999995,
            c=0.0026699999999999996,
            phi=1.5707963267948966,
        ),
    ]

    fuzz_bounds = {
        "hoop_stress": (100.0, 1000.0),
        "t": (0.004, 0.01),
        "w": (0.004, 0.01),
        "a": (0.0005, 0.0015),
        "c": (0.0005, 0.0015),
        "phi": (0.1, 1.5),
    }
    """`t`/`w`/`a`/`c` are kept in a range where `sqrt(a/t) * c/w < 1` always (worst case
    ~0.23 here) -- otherwise `sqrt(a_t) * pi * c / (2 * w)` in the port's (and PROCESS's
    own) `cos(...)` argument can exceed `pi/2`, making `cos` negative and the enclosing
    `sqrt(1/cos(...))` genuinely `nan` **on both sides** (confirmed: `CsFatigue.
    surface_stress_intensity_factor` itself returns `nan` there, not just the port) --
    a real PROCESS domain gap, not a porting defect, and not one
    `reference_domain_errors` can flag since PROCESS's own `numpy` `sqrt` of a negative
    number warns rather than raises. Same class of domain guard as
    `plasma_geometry.md`'s D1: documented and avoided by sampling, not fixed."""


class TestNCycle(Tier1Contract):
    """`calculate_n_cycle` -> `CsFatigue.ncycle`, `lax.while_loop` for the Python one.

    **Tier 1, with the gradient checks structurally excused** -- the disposition
    `cs_fatigue.md`'s open question 1 settled on 2026-08-27 and this pass carries out.
    Tier 1 is right for the *value*: `delta` is a fixed module constant, so the loop's
    termination is deterministic and PROCESS's answer is exact for that discretisation
    rather than an approximation to something else, which is the one thing Tier 2's
    "PROCESS may not have converged" contract exists to handle and there is nothing here
    for it to handle.

    The excuse is `static_argnames` covering **every** argument, so `diff_argnames` is
    empty and `_jacobians` returns without tracing. That is a heavier hammer than
    `inductance.md`'s `noh`, where three arguments are excused because they jointly
    determine one structural integer; here the trip count is a step function of all
    twelve, so there is no subset to excuse.

    **The excuse is a precaution, and it is worth recording that it is only that.**
    Measured at `low_aspect_ratio_DEMO`'s own operating point (`max_hoop_stress =
    2.94e8`, `dz_cs_turn_conduit = 0.010042`), `jacfwd` through the `while_loop` agrees
    with a central finite difference to **2.4e-8 relative** in `max_hoop_stress` and
    1.2e-8 in `dz_cs_turn_conduit` -- so the derivative is not merely finite, it is the
    right one, and `lax.while_loop`'s forward-mode rule is doing what it should. What
    the excuse protects against is the *riser*: `n_pulse` is a sum over a data-dependent
    number of steps, so at a fuzz point where PROCESS's `epsfcn` perturbation crosses a
    trip-count boundary the two sides differ by one whole `delta_n`, for a reason that
    belongs to PROCESS's discretisation and not to this port's arithmetic. Revisiting it
    would mean measuring how often that happens over the fuzz distribution, which is a
    piece of work and not a tolerance to tune.
    """

    audit_record = "models/cs_fatigue.md"
    reference = staticmethod(_reference_n_cycle)
    ported = calculate_n_cycle

    static_argnames = (
        "max_hoop_stress",
        "residual_stress",
        "t_crack_vertical",
        "dz_cs_turn_conduit",
        "dr_cs_turn_conduit",
        "paris_coefficient",
        "paris_power_law",
        "walker_coefficient",
        "sf_vertical_crack",
        "sf_radial_crack",
        "fracture_toughness",
        "sf_fast_fracture",
    )

    samples = [
        legacy_sample(
            "ncycle-baseline_2018",
            max_hoop_stress=659999225.25370133,
            residual_stress=240000000.0,
            t_crack_vertical=0.00088999999999999995,
            dz_cs_turn_conduit=0.0063104538380405924,
            dr_cs_turn_conduit=0.0063104538380405924,
            paris_coefficient=65.0e-14,
            paris_power_law=3.5,
            walker_coefficient=0.436,
            sf_vertical_crack=2.0,
            sf_radial_crack=2.0,
            fracture_toughness=2.0e2,
            sf_fast_fracture=1.5,
        ),
    ]
    """`tests/unit/models/test_cs_fatigue.py::test_ncycle`'s point, verbatim, with the
    seven coefficients spelled out at the `cs_fatigue_variables.py` defaults that
    fixture leaves them at. Genuinely legacy -- its own docstring says it came from
    `baseline_2018_IN.DAT`, a file no longer in the repository."""

    fuzz_bounds = {
        "max_hoop_stress": (2.0e8, 7.0e8),
        "residual_stress": (1.5e8, 3.0e8),
        "t_crack_vertical": (0.0005, 0.0012),
        "dz_cs_turn_conduit": (0.006, 0.025),
        "dr_cs_turn_conduit": (0.006, 0.025),
        "paris_coefficient": (4.0e-13, 9.0e-13),
        "paris_power_law": (3.0, 4.0),
        "walker_coefficient": (0.3, 0.6),
        "sf_vertical_crack": (1.5, 2.5),
        "sf_radial_crack": (1.5, 2.5),
        "fracture_toughness": (150.0, 250.0),
        "sf_fast_fracture": (1.2, 1.8),
    }
    """Straddling both tracked operating points -- `low_aspect_ratio_DEMO`'s computed
    conduit thicknesses (~0.0099 m) and the legacy sample's 0.0063 -- and staying inside
    `surface_stress_intensity_factor`'s own domain gap, which is the binding constraint
    here: `t` and `w` are the two conduit thicknesses and they are drawn from one range,
    so `sqrt(a/t) * pi * c / (2 * w)` stays well under `pi/2` for every crack size the
    loop walks through. See `TestSurfaceStressIntensityFactor.fuzz_bounds` for what
    happens on the other side of it (`nan` on both sides, PROCESS's gap and not the
    port's).

    The upper end of `max_hoop_stress` is the legacy point's own 6.6e8, not higher: the
    trip count falls as the stress rises, and a fuzz point that terminated on the very
    first pass would compare two numbers that exercise none of the loop."""
