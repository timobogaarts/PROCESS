"""Harness cases for the ported stellarator divertor model (registry unit #4).

No PROCESS unit test covers `st_div` directly (`tests/unit/models/stellarator/` has no
`test_divertor.py` and `test_stellarator.py` never calls it) -- there is no `legacy_sample`
to reuse here, only `fuzz_bounds`, so this unit's coverage is entirely against the real
`st_div` reference at random points rather than a human-checked operating point. Flagged
in the audit record.
"""

from functional_process.cottax._harness import Tier1Contract
from functional_process.cottax.stellarator.divertor import calculate_divertor
from process.core.model import DataStructure
from process.models.stellarator.divertor import st_div


def _reference_divertor(
    flpitch,
    rmajor,
    p_plasma_separatrix_mw,
    anginc,
    xpertin,
    tdiv,
    m_fuel_amu,
    bmn,
    shear,
    n_res,
    f_w,
    m_res,
    fdivwet,
    f_asym,
    a_fw_total,
):
    """Call PROCESS's `st_div` through the port's signature."""
    data = DataStructure()
    data.stellarator.flpitch = flpitch
    data.physics.rmajor = rmajor
    data.physics.p_plasma_separatrix_mw = p_plasma_separatrix_mw
    data.divertor.anginc = anginc
    data.divertor.xpertin = xpertin
    data.divertor.tdiv = tdiv
    data.physics.m_fuel_amu = m_fuel_amu
    data.stellarator.bmn = bmn
    data.stellarator.shear = shear
    data.stellarator.n_res = n_res
    data.stellarator.f_w = f_w
    data.stellarator.m_res = m_res
    data.stellarator.fdivwet = fdivwet
    data.stellarator.f_asym = f_asym
    data.first_wall.a_fw_total = a_fw_total

    st_div(stellarator=None, f_output=False, data=data)
    return (
        data.divertor.pflux_div_heat_load_mw,
        data.divertor.a_div_surface_total,
        data.fwbs.f_ster_div_single,
    )


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
