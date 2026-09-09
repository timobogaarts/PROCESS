"""Harness cases for the ported stellarator divertor model (registry unit #4).

No PROCESS unit test covers `st_div` directly (`tests/unit/models/stellarator/` has no
`test_divertor.py` and `test_stellarator.py` never calls it) -- there is no `legacy_sample`
to reuse here, only `fuzz_bounds`, so this unit's coverage is entirely against the real
`st_div` reference at random points rather than a human-checked operating point. Flagged
in the audit record.
"""

from functional_process.cottax._harness import Tier1Contract
from functional_process.cottax._harness.process_reference import data_reference
from functional_process.cottax.stellarator.divertor import calculate_divertor
from process.models.stellarator.divertor import st_div


def _call_divertor(data):
    st_div(stellarator=None, f_output=False, data=data)
    return (
        data.divertor.pflux_div_heat_load_mw,
        data.divertor.a_div_surface_total,
        data.fwbs.f_ster_div_single,
    )


_reference_divertor = data_reference(_call_divertor)


class TestDivertor(Tier1Contract):
    """`st_div` -> `calculate_divertor`."""

    audit_record = "models/stellarator/divertor.md"
    reference = _reference_divertor
    ported = calculate_divertor

    # No PROCESS unit test exercises `st_div` -- see module docstring, so there is no
    # `legacy_sample` to add here. Coverage is fuzz-only, at stellarator-scale bounds.
    fuzz_bounds = {
        "flpitch": (0.01, 0.5),
        "rmajor": (5.0, 30.0),
        "p_plasma_separatrix_mw": (10.0, 500.0),
        "anginc": (0.01, 0.5),
        "xpertin": (0.1, 5.0),
        "tdiv": (5.0, 100.0),
        "m_fuel_amu": (2.0, 3.0),
        "bmn": (1.0e-4, 1.0e-2),
        "shear": (0.1, 10.0),
        "n_res": (1.0, 20.0),
        "f_w": (0.1, 1.0),
        "m_res": (1.0, 20.0),
        "fdivwet": (0.1, 1.0),
        "f_asym": (1.0, 3.0),
        "a_fw_total": (100.0, 2000.0),
    }
