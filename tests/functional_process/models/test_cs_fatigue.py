"""Harness case for the ported CS fatigue stress-intensity factor
(`functional_process/models/cs_fatigue.py`).

`surface_stress_intensity_factor` is diffed directly against `CsFatigue`'s own
`@staticmethod` (itself `@njit`-compiled, but numba and JAX both just run it as plain
Python/numpy-shaped arithmetic here since neither `self` nor `self.data` is touched) --
no adapter needed.

The legacy sample is `tests/unit/models/test_cs_fatigue.py::test_surface_stress_
intensity_factor`'s point, verbatim -- genuinely legacy, and it exercises the `a <= c`
formula (`a=0.00089`, `c=0.00267`). `fuzz_bounds` deliberately straddles `a == c` (both
`a` and `c` drawn from the same range) so `test_gradient_agreement` also exercises the
`a > c` formula and the `jnp.where` switch-over itself -- see the port module's docstring
for why this is safe (the discarded branch's own non-finite sub-expressions do not leak
into the selected branch's value or gradient).
"""

from functional_process._harness import Tier1Contract, legacy_sample
from functional_process.models.cs_fatigue import surface_stress_intensity_factor
from process.models.cs_fatigue import CsFatigue


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
