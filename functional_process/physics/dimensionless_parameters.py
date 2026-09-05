"""Pure physics functions extracted from `models/physics/dimensionless_parameters.py`.

See that file for the declarations wiring these into the graph, and
`_audit/units/models/physics/dimensionless_parameters.md` for the audit record. No
graph-framework import belongs in this module -- see `_audit/formulas_split.md`.
"""

import jax.numpy as jnp

from functional_process.models.safe_math import safe_sqrt
from functional_process.vocabulary import constants


def calculate_dimensionless_plasma_parameters(
    dlamie,
    vol_plasma,
    rmajor,
    b_plasma_toroidal_on_axis,
    eps,
    nd_plasma_electron_line,
    kappa,
    e_plasma_beta,
    plasma_current,
    m_ions_total_amu,
):
    """Three dimensionless plasma parameters: `nu_star`, `rho_star`, `beta_mcdonald`.

    Ports the only computation in `Physics.outplas` (`physics.py:1790-1822`) --
    straight-line arithmetic, no branches, no calls. Everything else in the 1095-line
    source method is either a call into another model's own `.output()` (out of scope,
    see the audit record) or a `process_output` write with no `self.data` write at all.

    Parameters
    ----------
    dlamie :
        Ion-electron Coulomb logarithm.
    vol_plasma :
        Plasma volume (m^3).
    rmajor :
        Plasma major radius (m).
    b_plasma_toroidal_on_axis :
        Toroidal field on axis (T).
    eps :
        Inverse aspect ratio.
    nd_plasma_electron_line :
        Line-averaged electron density (m^-3).
    kappa :
        Plasma elongation.
    e_plasma_beta :
        Total plasma stored energy from beta (J).
    plasma_current :
        Plasma current (A).
    m_ions_total_amu :
        Average mass of all ions (amu).

    Returns
    -------
    tuple
        `(nu_star, rho_star, beta_mcdonald)`.
    """
    nu_star = (
        1.0
        / constants.RMU0
        * (15.0 * constants.ELECTRON_CHARGE**4 * dlamie)
        / (4.0 * jnp.pi**1.5 * constants.EPSILON0**2)
        * vol_plasma**2
        * rmajor**2
        * b_plasma_toroidal_on_axis
        * safe_sqrt(eps)
        * nd_plasma_electron_line**3
        * kappa
        / (e_plasma_beta**2 * plasma_current)
    )

    rho_star = safe_sqrt(
        2.0
        * constants.PROTON_MASS
        * m_ions_total_amu
        * e_plasma_beta
        / (3.0 * vol_plasma * nd_plasma_electron_line)
    ) / (constants.ELECTRON_CHARGE * b_plasma_toroidal_on_axis * eps * rmajor)

    beta_mcdonald = (
        4.0
        / 3.0
        * constants.RMU0
        * e_plasma_beta
        / (vol_plasma * b_plasma_toroidal_on_axis**2)
    )

    return nu_star, rho_star, beta_mcdonald
