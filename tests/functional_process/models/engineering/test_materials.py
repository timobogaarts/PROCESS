"""Harness case for `eurofer97_thermal_conductivity`, ported from
`process/models/engineering/materials.py` -- see
`functional_process/_audit/units/models/engineering/materials.md`.

One class, not three: `calculate_tresca_stress` and `calculate_von_mises_stress` live in
the same PROCESS module but were ported into the stress packages
(`functional_process/models/pfcoil/stresses.py`, `models/tfcoil/stress.py`) and are
covered by those units' cases. Porting them a second time here would give one formula
two homes.
"""

from functional_process._harness import Tier1Contract, legacy_sample
from functional_process.models.engineering.materials import (
    eurofer97_thermal_conductivity,
)
from process.models.engineering.materials import (
    eurofer97_thermal_conductivity as _reference_eurofer97_thermal_conductivity,
)


class TestEurofer97ThermalConductivity(Tier1Contract):
    """`eurofer97_thermal_conductivity` -> the same, unchanged.

    The legacy sample is `tests/unit/models/engineering/test_materials.py::
    test_eurofer97_thermal_conductivity`, verbatim -- including its `temp = 1900.0`,
    which is well past the ~800 K the fit is documented for. That is PROCESS's own
    choice of test point and it is kept rather than tidied: the cubic extrapolates
    without complaint on both sides, and a port that agreed only inside the documented
    range would not be the same function.

    `fw_th_conductivity` is fuzzed around `28.34`, the cubic's own value at 293 K and
    hence the normalising constant -- the argument scales the whole curve linearly, so
    a wide box would test multiplication rather than the fit. `temp` runs from cryogenic
    to the extrapolated end of PROCESS's own test.
    """

    audit_record = "models/engineering/materials.md"
    reference = staticmethod(_reference_eurofer97_thermal_conductivity)
    ported = eurofer97_thermal_conductivity

    samples = [
        legacy_sample(
            "eurofer97-tests-unit-test_materials",
            temp=1900.0,
            fw_th_conductivity=28.9,
        ),
    ]

    fuzz_bounds = {
        "temp": (77.0, 1900.0),
        "fw_th_conductivity": (20.0, 40.0),
    }
