"""Pure-functional port of the one real computation inside
`Physics.outplas` (`process/models/physics/physics.py`).

Registry unit #9, chunk C. Audit record:
`functional_process/models/physics/physics_C_outplas.md` -- read it first for why
`outplas` (1095 source lines) reduces to a single 3-output, 9-input pure function.
"""

import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    FromExactly,
    Output,
)

from functional_process.models.safe_math import safe_sqrt
from process.core import constants


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

    rho_star = safe_sqrt(2.0
        * constants.PROTON_MASS
        * m_ions_total_amu
        * e_plasma_beta
        / (3.0 * vol_plasma * nd_plasma_electron_line)) / (constants.ELECTRON_CHARGE * b_plasma_toroidal_on_axis * eps * rmajor)

    beta_mcdonald = (
        4.0
        / 3.0
        * constants.RMU0
        * e_plasma_beta
        / (vol_plasma * b_plasma_toroidal_on_axis**2)
    )

    return nu_star, rho_star, beta_mcdonald


class DimensionlessPlasmaParameters(ExplicitFunction):
    """cottax node: `calculate_dimensionless_plasma_parameters`, ports declared."""

    nu_star = Output(lambda s: s.physics.nu_star)
    rho_star = Output(lambda s: s.physics.rho_star)
    beta_mcdonald = Output(lambda s: s.physics.beta_mcdonald)

    def __call__(
        self,
        dlamie=FromExactly(lambda s: s.physics.dlamie),
        vol_plasma=FromExactly(lambda s: s.physics.vol_plasma),
        rmajor=FromExactly(lambda s: s.physics.rmajor),
        b_plasma_toroidal_on_axis=FromExactly(lambda s: s.physics.b_plasma_toroidal_on_axis),
        eps=FromExactly(lambda s: s.physics.eps),
        nd_plasma_electron_line=FromExactly(lambda s: s.physics.nd_plasma_electron_line),
        kappa=FromExactly(lambda s: s.physics.kappa),
        e_plasma_beta=FromExactly(lambda s: s.physics.e_plasma_beta),
        plasma_current=FromExactly(lambda s: s.physics.plasma_current),
        m_ions_total_amu=FromExactly(lambda s: s.physics.m_ions_total_amu),
    ):
        return calculate_dimensionless_plasma_parameters(
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
        )
