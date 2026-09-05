"""Pure physics functions extracted from `models/physics/l_h_transition.py`.

See that file for the declarations wiring these into the graph, and
`_audit/units/models/physics/l_h_transition.md` for the audit record. No
graph-framework import belongs in this module -- see `_audit/formulas_split.md`.
"""

import jax.numpy as jnp

from functional_process.models.safe_math import safe_pow


def calculate_iter1996_nominal(dnla20, b_plasma_toroidal_on_axis, rmajor):
    """Nominal ITER-1996 L-H threshold. `PlasmaConfinementTransitionModel.
    ITER1996_NOMINAL` (1). `process/models/physics/l_h_transition.py:541-570`.
    """
    return 0.45 * safe_pow(dnla20, 0.75) * b_plasma_toroidal_on_axis * rmajor**2


def calculate_iter1996_upper(dnla20, b_plasma_toroidal_on_axis, rmajor):
    """Upper ITER-1996 L-H threshold. `PlasmaConfinementTransitionModel.
    ITER1996_UPPER` (2). `process/models/physics/l_h_transition.py:572-602`.
    """
    return 0.3960502816 * dnla20 * b_plasma_toroidal_on_axis * rmajor**2.5


def calculate_iter1996_lower(dnla20, b_plasma_toroidal_on_axis, rmajor):
    """Lower ITER-1996 L-H threshold. `PlasmaConfinementTransitionModel.
    ITER1996_LOWER` (3). `process/models/physics/l_h_transition.py:604-634`.
    """
    return 0.5112987149 * safe_pow(dnla20, 0.5) * b_plasma_toroidal_on_axis * rmajor**1.5


def calculate_snipes1997_iter(dnla20, b_plasma_toroidal_on_axis, rmajor):
    """Snipes 1997 ITER L-H threshold. `PlasmaConfinementTransitionModel.
    SNIPES1997_ITER` (4). `process/models/physics/l_h_transition.py:637-665`.
    """
    return (
        0.65
        * safe_pow(dnla20, 0.93)
        * safe_pow(b_plasma_toroidal_on_axis, 0.86)
        * rmajor**2.15
    )


def calculate_snipes1997_kappa(dnla20, b_plasma_toroidal_on_axis, rmajor, kappa):
    """Snipes 1997 ITER L-H threshold with a kappa factor.
    `PlasmaConfinementTransitionModel.SNIPES1997_KAPPA` (5).
    `process/models/physics/l_h_transition.py:667-705`.
    """
    return (
        0.42
        * safe_pow(dnla20, 0.80)
        * safe_pow(b_plasma_toroidal_on_axis, 0.90)
        * rmajor**1.99
        * safe_pow(kappa, 0.76)
    )


def calculate_martin08_nominal(
    dnla20, b_plasma_toroidal_on_axis, a_plasma_surface, m_ions_total_amu
):
    """Nominal Martin 2008 L-H threshold. `PlasmaConfinementTransitionModel.
    MARTIN08_NOMINAL` (6). `process/models/physics/l_h_transition.py:707-755`.
    """
    return (
        0.0488
        * safe_pow(dnla20, 0.717)
        * safe_pow(b_plasma_toroidal_on_axis, 0.803)
        * safe_pow(a_plasma_surface, 0.941)
        * (2.0 / m_ions_total_amu)
    )


def calculate_martin08_upper(
    dnla20, b_plasma_toroidal_on_axis, a_plasma_surface, m_ions_total_amu
):
    """Upper Martin 2008 L-H threshold. `PlasmaConfinementTransitionModel.
    MARTIN08_UPPER` (7). `process/models/physics/l_h_transition.py:757-804`.
    """
    return (
        0.05166240355
        * safe_pow(dnla20, 0.752)
        * safe_pow(b_plasma_toroidal_on_axis, 0.835)
        * safe_pow(a_plasma_surface, 0.96)
        * (2.0 / m_ions_total_amu)
    )


def calculate_martin08_lower(
    dnla20, b_plasma_toroidal_on_axis, a_plasma_surface, m_ions_total_amu
):
    """Lower Martin 2008 L-H threshold. `PlasmaConfinementTransitionModel.
    MARTIN08_LOWER` (8). `process/models/physics/l_h_transition.py:806-853`.
    """
    return (
        0.04609619059
        * safe_pow(dnla20, 0.682)
        * safe_pow(b_plasma_toroidal_on_axis, 0.771)
        * safe_pow(a_plasma_surface, 0.922)
        * (2.0 / m_ions_total_amu)
    )


def calculate_snipes2000_nominal(
    dnla20, b_plasma_toroidal_on_axis, rmajor, rminor, m_ions_total_amu
):
    """Nominal Snipes 2000 L-H threshold. `PlasmaConfinementTransitionModel.
    SNIPES2000_NOMINAL` (9). `process/models/physics/l_h_transition.py:855-905`.
    """
    return (
        1.42
        * safe_pow(dnla20, 0.58)
        * safe_pow(b_plasma_toroidal_on_axis, 0.82)
        * rmajor
        * safe_pow(rminor, 0.81)
        * (2.0 / m_ions_total_amu)
    )


def calculate_snipes2000_upper(
    dnla20, b_plasma_toroidal_on_axis, rmajor, rminor, m_ions_total_amu
):
    """Upper Snipes 2000 L-H threshold. `PlasmaConfinementTransitionModel.
    SNIPES2000_UPPER` (10). `process/models/physics/l_h_transition.py:907-958`.
    """
    return (
        1.547
        * safe_pow(dnla20, 0.615)
        * safe_pow(b_plasma_toroidal_on_axis, 0.851)
        * rmajor**1.089
        * safe_pow(rminor, 0.876)
        * (2.0 / m_ions_total_amu)
    )


def calculate_snipes2000_lower(
    dnla20, b_plasma_toroidal_on_axis, rmajor, rminor, m_ions_total_amu
):
    """Lower Snipes 2000 L-H threshold. `PlasmaConfinementTransitionModel.
    SNIPES2000_LOWER` (11). `process/models/physics/l_h_transition.py:960-1011`.
    """
    return (
        1.293
        * safe_pow(dnla20, 0.545)
        * safe_pow(b_plasma_toroidal_on_axis, 0.789)
        * safe_pow(rmajor, 0.911)
        * safe_pow(rminor, 0.744)
        * (2.0 / m_ions_total_amu)
    )


def calculate_snipes2000_closed_divertor_nominal(
    dnla20, b_plasma_toroidal_on_axis, rmajor, m_ions_total_amu
):
    """Nominal Snipes 2000 (closed divertor) L-H threshold.
    `PlasmaConfinementTransitionModel.SNIPES2000_CLOSED_DIVERTOR_NOMINAL` (12).
    `process/models/physics/l_h_transition.py:1013-1061`.
    """
    return (
        0.8
        * safe_pow(dnla20, 0.5)
        * safe_pow(b_plasma_toroidal_on_axis, 0.53)
        * rmajor**1.51
        * (2.0 / m_ions_total_amu)
    )


def calculate_snipes2000_closed_divertor_upper(
    dnla20, b_plasma_toroidal_on_axis, rmajor, m_ions_total_amu
):
    """Upper Snipes 2000 (closed divertor) L-H threshold.
    `PlasmaConfinementTransitionModel.SNIPES2000_CLOSED_DIVERTOR_UPPER` (13).
    `process/models/physics/l_h_transition.py:1063-1111`.
    """
    return (
        0.867
        * safe_pow(dnla20, 0.561)
        * safe_pow(b_plasma_toroidal_on_axis, 0.588)
        * rmajor**1.587
        * (2.0 / m_ions_total_amu)
    )


def calculate_snipes2000_closed_divertor_lower(
    dnla20, b_plasma_toroidal_on_axis, rmajor, m_ions_total_amu
):
    """Lower Snipes 2000 (closed divertor) L-H threshold.
    `PlasmaConfinementTransitionModel.SNIPES2000_CLOSED_DIVERTOR_LOWER` (14).
    `process/models/physics/l_h_transition.py:1113-1161`.
    """
    return (
        0.733
        * safe_pow(dnla20, 0.439)
        * safe_pow(b_plasma_toroidal_on_axis, 0.472)
        * rmajor**1.433
        * (2.0 / m_ions_total_amu)
    )


def calculate_hubbard2012_nominal(plasma_current, dnla20):
    """Nominal Hubbard 2012 L-I threshold. `PlasmaConfinementTransitionModel.
    HUBBARD2012_NOMINAL` (15). `process/models/physics/l_h_transition.py:1163-1187`.
    """
    return 2.11 * safe_pow(plasma_current / 1.0e6, 0.94) * safe_pow(dnla20, 0.65)


def calculate_hubbard2012_upper(plasma_current, dnla20):
    """Upper Hubbard 2012 L-I threshold. `PlasmaConfinementTransitionModel.
    HUBBARD2012_UPPER` (17). `process/models/physics/l_h_transition.py:1189-1213`.
    `plasma_current`'s exponent (`1.18`) is `> 1`, so it keeps the bare `**`.
    """
    return 2.11 * (plasma_current / 1.0e6) ** 1.18 * safe_pow(dnla20, 0.83)


def calculate_hubbard2012_lower(plasma_current, dnla20):
    """Lower Hubbard 2012 L-I threshold. `PlasmaConfinementTransitionModel.
    HUBBARD2012_LOWER` (16). `process/models/physics/l_h_transition.py:1216-1239`.
    """
    return 2.11 * safe_pow(plasma_current / 1.0e6, 0.7) * safe_pow(dnla20, 0.47)


def calculate_hubbard2017(dnla20, a_plasma_surface, b_plasma_toroidal_on_axis):
    """Hubbard 2017 L-I threshold. `PlasmaConfinementTransitionModel.
    HUBBARD2017_I_MODE` (18). `process/models/physics/l_h_transition.py:1241-1272`.
    """
    return 0.162 * dnla20 * a_plasma_surface * safe_pow(b_plasma_toroidal_on_axis, 0.26)


def _martin08_aspect_correction(aspect):
    """The aspect-ratio correction shared by the three `martin08_aspect_*` laws.

    `process/models/physics/l_h_transition.py:1328-1331` (and the two siblings' identical
    copies): `if aspect <= 2.7: ... else: 1.0` on `aspect`, a plain differentiable
    argument here, not a switch -- `needs-lax-cond-or-where` (see the audit record's JAX
    flags). `jnp.where` evaluates both branches; the "then" branch's own denominator,
    `1.0 - safe_pow(2.0 / (1.0 + aspect), 0.5)`, is singular only at `aspect == 1.0`,
    outside every physically meaningful aspect ratio and outside
    `ITERATION_VARIABLES[1]`'s declared bounds `(1.1, 10.0)` -- documented, not guarded,
    matching `plasma_geometry.md`'s convention for a singularity outside the sampled
    domain.
    """
    correction = 0.098 * aspect / (1.0 - safe_pow(2.0 / (1.0 + aspect), 0.5))
    return jnp.where(aspect <= 2.7, correction, 1.0)


def calculate_martin08_aspect_nominal(
    dnla20, b_plasma_toroidal_on_axis, a_plasma_surface, m_ions_total_amu, aspect
):
    """Nominal Martin 2008 L-H threshold, aspect-ratio corrected.
    `PlasmaConfinementTransitionModel.MARTIN08_ASPECT_NOMINAL` (19) -- PROCESS's own
    default (`physics_variables.py:1234`) and the live arm on
    `large_tokamak_eval.IN.DAT`, which never sets `i_l_h_threshold`.
    `process/models/physics/l_h_transition.py:1274-1340`.
    """
    return (
        0.0488
        * safe_pow(dnla20, 0.717)
        * safe_pow(b_plasma_toroidal_on_axis, 0.803)
        * safe_pow(a_plasma_surface, 0.941)
        * (2.0 / m_ions_total_amu)
        * _martin08_aspect_correction(aspect)
    )


def calculate_martin08_aspect_upper(
    dnla20, b_plasma_toroidal_on_axis, a_plasma_surface, m_ions_total_amu, aspect
):
    """Upper Martin 2008 L-H threshold, aspect-ratio corrected.
    `PlasmaConfinementTransitionModel.MARTIN08_ASPECT_UPPER` (20).
    `process/models/physics/l_h_transition.py:1342-1408`.
    """
    return (
        0.05166240355
        * safe_pow(dnla20, 0.752)
        * safe_pow(b_plasma_toroidal_on_axis, 0.835)
        * safe_pow(a_plasma_surface, 0.96)
        * (2.0 / m_ions_total_amu)
        * _martin08_aspect_correction(aspect)
    )


def calculate_martin08_aspect_lower(
    dnla20, b_plasma_toroidal_on_axis, a_plasma_surface, m_ions_total_amu, aspect
):
    """Lower Martin 2008 L-H threshold, aspect-ratio corrected.
    `PlasmaConfinementTransitionModel.MARTIN08_ASPECT_LOWER` (21).
    `process/models/physics/l_h_transition.py:1410-1476`.
    """
    return (
        0.04609619059
        * safe_pow(dnla20, 0.682)
        * safe_pow(b_plasma_toroidal_on_axis, 0.771)
        * safe_pow(a_plasma_surface, 0.922)
        * (2.0 / m_ions_total_amu)
        * _martin08_aspect_correction(aspect)
    )


def calculate_martin08_nominal_threshold_power(
    nd_plasma_electron_line,
    b_plasma_toroidal_on_axis,
    a_plasma_surface,
    m_ions_total_amu,
):
    """`i_l_h_threshold == 6`. No `aspect` -- that correction is not part of this arm."""
    return calculate_martin08_nominal(
        1.0e-20 * nd_plasma_electron_line,
        b_plasma_toroidal_on_axis,
        a_plasma_surface,
        m_ions_total_amu,
    )


def calculate_martin08_upper_threshold_power(
    nd_plasma_electron_line,
    b_plasma_toroidal_on_axis,
    a_plasma_surface,
    m_ions_total_amu,
):
    """`i_l_h_threshold == 7`."""
    return calculate_martin08_upper(
        1.0e-20 * nd_plasma_electron_line,
        b_plasma_toroidal_on_axis,
        a_plasma_surface,
        m_ions_total_amu,
    )


def calculate_martin08_lower_threshold_power(
    nd_plasma_electron_line,
    b_plasma_toroidal_on_axis,
    a_plasma_surface,
    m_ions_total_amu,
):
    """`i_l_h_threshold == 8`."""
    return calculate_martin08_lower(
        1.0e-20 * nd_plasma_electron_line,
        b_plasma_toroidal_on_axis,
        a_plasma_surface,
        m_ions_total_amu,
    )


def calculate_martin08_aspect_nominal_threshold_power(
    nd_plasma_electron_line,
    b_plasma_toroidal_on_axis,
    a_plasma_surface,
    m_ions_total_amu,
    aspect,
):
    """`i_l_h_threshold == 19` -- the reference arm on `large_tokamak_eval.IN.DAT`."""
    return calculate_martin08_aspect_nominal(
        1.0e-20 * nd_plasma_electron_line,
        b_plasma_toroidal_on_axis,
        a_plasma_surface,
        m_ions_total_amu,
        aspect,
    )


def calculate_martin08_aspect_upper_threshold_power(
    nd_plasma_electron_line,
    b_plasma_toroidal_on_axis,
    a_plasma_surface,
    m_ions_total_amu,
    aspect,
):
    """`i_l_h_threshold == 20`."""
    return calculate_martin08_aspect_upper(
        1.0e-20 * nd_plasma_electron_line,
        b_plasma_toroidal_on_axis,
        a_plasma_surface,
        m_ions_total_amu,
        aspect,
    )


def calculate_martin08_aspect_lower_threshold_power(
    nd_plasma_electron_line,
    b_plasma_toroidal_on_axis,
    a_plasma_surface,
    m_ions_total_amu,
    aspect,
):
    """`i_l_h_threshold == 21`."""
    return calculate_martin08_aspect_lower(
        1.0e-20 * nd_plasma_electron_line,
        b_plasma_toroidal_on_axis,
        a_plasma_surface,
        m_ions_total_amu,
        aspect,
    )
