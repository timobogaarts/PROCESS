"""Pure physics functions extracted from
`functional_process.models.stellarator.divertor`, per `_audit/formulas_split.md`
step 2 phase A. The graph declarations that wrap these stay in that sibling
module.
"""

import jax.numpy as jnp

from functional_process.models.safe_math import safe_sqrt

_ELECTRON_CHARGE = 1.602176634e-19
"""Coulombs. `process/core/constants.py::ELECTRON_CHARGE`."""


_ATOMIC_MASS_UNIT = 1.660538921e-27
"""kg. `process/core/constants.py::UMASS`."""


def calculate_divertor(
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
    """Stellarator divertor heat load and wetted-area geometry.

    Ports `st_div`'s three real outputs (drops the reporting-only intermediates that
    `output()` prints but nothing downstream reads: wetted area, plate length/width,
    channel-broadening factor, power decay width, island width, X-point-to-plate
    distance -- see the audit record's data-footprint table).

    Reference: Stellarator Divertor Model for the Systems Code PROCESS, F. Warmer,
    21/06/2013.

    Parameters
    ----------
    flpitch :
        Field line pitch (rad). `.stellarator.flpitch`.
    rmajor :
        Plasma major radius (m). `.physics.rmajor`.
    p_plasma_separatrix_mw :
        Power to the separatrix (MW). `.physics.p_plasma_separatrix_mw`.
    anginc :
        Angle of incidence (rad). `.divertor.anginc`.
    xpertin :
        Perpendicular heat transport coefficient (m2/s). `.divertor.xpertin`.
    tdiv :
        Divertor plasma (scrape-off) temperature (eV). `.divertor.tdiv`.
    m_fuel_amu :
        Fuel ion mass (amu). `.physics.m_fuel_amu`.
    bmn :
        Relative radial field perturbation. `.stellarator.bmn`.
    shear :
        Magnetic shear (/m). `.stellarator.shear`.
    n_res :
        Toroidal resonance number. `.stellarator.n_res`.
    f_w :
        Island size fraction factor. `.stellarator.f_w`.
    m_res :
        Poloidal resonance number. `.stellarator.m_res`.
    fdivwet :
        Wetted-area fraction of total plate area. `.stellarator.fdivwet`.
    f_asym :
        Heat load peaking factor. `.stellarator.f_asym`.
    a_fw_total :
        Total first-wall area (m2). `.first_wall.a_fw_total`.

    Returns
    -------
    :
        `(pflux_div_heat_load_mw, a_div_surface_total, f_ster_div_single)` -- peak
        divertor heat load (MW/m2), total divertor plate area (m2), and the divertor's
        area fraction of the first wall.
    """
    e = tdiv * _ELECTRON_CHARGE
    c_s = safe_sqrt(e / (m_fuel_amu * _ATOMIC_MASS_UNIT))

    w_r = 4.0 * safe_sqrt(bmn * rmajor / (shear * n_res))
    delta = f_w * w_r
    l_p = 2.0 * jnp.pi * rmajor * m_res / n_res
    l_x_t = delta / flpitch
    l_q = safe_sqrt(xpertin * (l_x_t / c_s))
    l_b = safe_sqrt(xpertin * l_p / c_s)
    f_x = 1.0 + (l_b / (l_p * flpitch))
    l_d = f_x * l_p * (flpitch / anginc)
    l_t = 2.0 * n_res * l_d
    a_eff = l_t * l_q

    darea = a_eff / fdivwet

    pflux_div_heat_load_mw = f_asym * (p_plasma_separatrix_mw / a_eff)
    a_div_surface_total = darea
    f_ster_div_single = darea / a_fw_total

    return pflux_div_heat_load_mw, a_div_surface_total, f_ster_div_single
