"""Harness cases for the coolant-hydraulics leaf formulas ported from
`process/models/engineering/pumping.py` -- see
`functional_process/_audit/units/models/engineering/pumping.md`.

All three are already pure in `process/` (module-level `def`s, no `self.data`), so the
PROCESS reference is called directly with no `DataStructure` adapter needed. Every
legacy sample below is lifted verbatim from `tests/unit/models/engineering/
test_pumping.py`, which is the only free oracle any of the three has.

These validate arithmetic behind `.fwbs.i_p_coolant_pumping == 2`, an arm no tracked
regression input selects and which `indat.py` still refuses on CoolProp grounds. Nothing
here changes that -- see the port module's docstring.
"""

from functional_process.cottax._harness import Tier1Contract, legacy_sample
from functional_process.models.engineering.pumping import (
    calculate_reynolds_number,
    darcy_friction_haaland,
    gnielinski_heat_transfer_coefficient,
)
from process.models.engineering.pumping import (
    calculate_reynolds_number as _reference_calculate_reynolds_number,
)
from process.models.engineering.pumping import (
    darcy_friction_haaland as _reference_darcy_friction_haaland,
)
from process.models.engineering.pumping import (
    gnielinski_heat_transfer_coefficient as _reference_gnielinski,
)


class TestCalculateReynoldsNumber(Tier1Contract):
    """`calculate_reynolds_number` -> the same, unchanged.

    Fuzz box spans both coolants PROCESS's `CoolantType` knows about: helium at blanket
    pressure (density order 10 kg/m^3, viscosity order 4e-5 Pa.s) at the low end of
    `den_coolant` and pressurised water (order 700 kg/m^3, 1e-4 Pa.s) at the high end.
    """

    audit_record = "models/engineering/pumping.md"
    reference = staticmethod(_reference_calculate_reynolds_number)
    ported = calculate_reynolds_number

    samples = [
        legacy_sample(
            "reynolds-tests-unit-test_pumping",
            den_coolant=8.8673250601290707,
            vel_coolant=12.649110769896881,
            radius_channel=0.0060000000000000001,
            visc_coolant=4.0416219836935569e-05,
        ),
    ]

    fuzz_bounds = {
        "den_coolant": (1.0, 1000.0),
        "vel_coolant": (0.1, 30.0),
        "radius_channel": (0.001, 0.1),
        "visc_coolant": (1.0e-5, 1.0e-3),
    }


class TestDarcyFrictionHaaland(Tier1Contract):
    """`darcy_friction_haaland` -> the same, unchanged.

    `reynolds` is fuzzed over the turbulent range the Haaland approximation is written
    for; `roughness_channel` spans PROCESS's own two defaults (`6e-8` in the unit test's
    Gnielinski point, `1e-6` in its Haaland point) and two decades either side.

    No `safe_pow` on either exponent, and the port's docstring records why: `1.11` and
    `-2` are both outside `safe_math`'s `0 < p < 1` window, and the smooth-pipe limit
    `roughness_channel == 0` already has a finite derivative. The gradient check under
    `--fp-gradients` is what holds that claim to account.
    """

    audit_record = "models/engineering/pumping.md"
    reference = staticmethod(_reference_darcy_friction_haaland)
    ported = darcy_friction_haaland

    samples = [
        legacy_sample(
            "haaland-tests-unit-test_pumping",
            reynolds=5500.0,
            roughness_channel=1.0e-6,
            radius_channel=0.1,
        ),
    ]

    fuzz_bounds = {
        "reynolds": (3.0e3, 5.0e6),
        "roughness_channel": (1.0e-9, 1.0e-4),
        "radius_channel": (0.001, 0.1),
    }


class TestGnielinskiHeatTransferCoefficient(Tier1Contract):
    """`gnielinski_heat_transfer_coefficient` -> the same, unchanged.

    The correlation is valid for `3000 < Re < 5e6` and `0.5 < Pr < 2000`, and PROCESS
    checks both with a `logger.error` that changes no value (the port drops all three
    checks; see its docstring). The box below is drawn so that the *typical* draw sits
    inside those bounds -- `Re = mflux * 2 * radius / visc` runs about `5e3` to `1e6`
    across it, `Pr = heatcap * visc / thermcond` about `0.4` to `7` -- rather than to
    guarantee it at every corner, which no product of four independently sampled
    factors can do. A corner that dips just under `Pr = 0.5` is not a test failure and
    is not being hidden: both sides evaluate the same closed form there, so they agree
    exactly, and only the physical *interpretation* of the number degrades.

    The one narrow bound is `roughness_channel`, kept below `radius_channel` by three
    decades. `roughness / radius` above about `0.05` puts the Haaland bracket near `1`,
    where the `** (-2)` is singular; PROCESS has no guard, so both sides would return
    the same infinity and the contract would still pass. The bound is here to keep the
    sampled points physical, not to hide a disagreement -- the same reasoning
    `test_ivc_functions.py::TestDshellvol` records for its `drin`.
    """

    audit_record = "models/engineering/pumping.md"
    reference = staticmethod(_reference_gnielinski)
    ported = gnielinski_heat_transfer_coefficient

    samples = [
        legacy_sample(
            "gnielinski-tests-unit-test_pumping",
            mflux_coolant=112.19853108876258,
            den_coolant=8.8673250601290707,
            radius_channel=0.0060000000000000001,
            heatcap_coolant=5184.9330299967578,
            visc_coolant=4.0416219836935569e-05,
            thermcond_coolant=0.3211653052986152,
            roughness_channel=6e-8,
        ),
    ]

    fuzz_bounds = {
        "mflux_coolant": (50.0, 500.0),
        "den_coolant": (1.0, 1000.0),
        "radius_channel": (0.003, 0.02),
        "heatcap_coolant": (4000.0, 6000.0),
        "visc_coolant": (2.0e-5, 6.0e-5),
        "thermcond_coolant": (0.05, 0.2),
        "roughness_channel": (1.0e-9, 1.0e-6),
    }
