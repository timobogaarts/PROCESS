"""Pure-functional port of `process/models/physics/confinement_time.py`.

Registry unit #10. Audit record: `functional_process/models/physics/confinement_time.md`.
Read it first, especially "A latent PROCESS bug, ported faithfully" (the
`KAYE_GOLDSTON` branch) and "A dead branch" (`PAZ_SOLDAN_NT`) before trusting any single
scaling law's numbers against `calculate_confinement_time`'s dispatch.

In scope: `calculate_confinement_time` and `calculate_double_and_triple_product`
(registry's stated method list), plus everything they call transitively within this same
file -- 48 individual `<name>_confinement_time` scaling-law statics, all already pure
(no `self.data` access of their own). Also ported here, out of nominal file scope but
needed for closure: `calculate_iter_physics_basis_elongation`, a one-line pure formula
`calculate_confinement_time` calls into `process/models/physics/plasma_geometry.py`
for -- see the audit record's "calls into other models".

Every scaling law keeps its PROCESS parameter names and formula verbatim, translated
`np.` -> `jnp.`, `min`/`max` -> `jnp.minimum`/`jnp.maximum` (JAX cannot trace a Python
`min`/`max` over a differentiable argument -- see the audit record's JAX-difficulty
flags). `menard_nstx_petty08_hybrid_confinement_time`'s three-way `if`/`elif`/`else` on
`1/aspect` is replaced with the equivalent clipped linear blend (verified continuous:
the "else" branch already reduces to the two boundary values exactly at the two
thresholds), since `aspect` is a differentiable argument here, not a switch.

`i_confinement_time` and `i_rad_loss` are switches (`_audit/naming_convention.md` §
"switches are not ports"): plain Python ints used for ordinary branching in
`calculate_confinement_time`, never traced. The harness marks them
`static_argnames` so `jacfwd` never differentiates through the dispatch itself.
"""

import inspect

import equinox as eqx
import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    FromExactly,
    OutputInto,
)

from functional_process.models.safe_math import safe_pow, safe_sqrt
from functional_process.paths import current_drive, physics, stellarator
from process.data_structure.physics_variables import (
    ConfinementRadiationLossModel,
    ConfinementTimeModel,
    PlasmaIgnitionModel,
)

# ---------------------------------------------------------------------------
# `plasma_geometry.py::PlasmaGeom.calculate_iter_physics_basis_elongation` --
# not this file's own method, but called unconditionally from
# `calculate_confinement_time`'s body (`self.data.physics.kappa_ipb = PlasmaGeom.
# calculate_iter_physics_basis_elongation(...)`). One line, no further dependencies;
# ported here rather than deferred, same call as `radiation_power.md`/`fusion_
# reactions.md` made for their own out-of-registry one-line dependencies.
# ---------------------------------------------------------------------------


def calculate_iter_physics_basis_elongation(vol_plasma, rmajor, rminor):
    """ITER physics basis elongation (kappa_ipb). Ports `PlasmaGeom.calculate_iter_
    physics_basis_elongation`, `process/models/physics/plasma_geometry.py:1003-1032`,
    unchanged.
    """
    return (vol_plasma / (2.0 * jnp.pi * rmajor)) / (jnp.pi * rminor**2)


# ---------------------------------------------------------------------------
# The 48 scaling laws. Direct ports, source order preserved (matches
# `ConfinementTimeModel`'s declaration order, and `tests/unit/models/physics/
# test_confinement_time.py`'s parametrisation, which this unit's legacy samples
# are lifted from verbatim).
# ---------------------------------------------------------------------------


def neo_alcator_confinement_time(dene20, rminor, rmajor, qstar):
    """Neo-Alcator (NA) OH scaling. `ConfinementTimeModel.NEO_ALCATOR` (1)."""
    return 0.07 * dene20 * rminor * rmajor * rmajor * qstar


def mirnov_confinement_time(rminor, kappa95, cur_plasma_ma):
    """Mirnov-like (H-mode) scaling. `ConfinementTimeModel.MIRNOV` (2)."""
    return 0.2 * rminor * safe_sqrt(kappa95) * cur_plasma_ma


def merezhkin_muhkovatov_confinement_time(
    rmajor, rminor, kappa95, qstar, nd_plasma_electron_line_20, afuel, ten
):
    """Merezhkin-Mukhovatov (MM) OH/L-mode scaling.
    `ConfinementTimeModel.MEREZHKIN_MUHKOVATOV` (3).
    """
    return (
        3.5e-3
        * rmajor**2.75
        * safe_pow(rminor, 0.25)
        * safe_pow(kappa95, 0.125)
        * qstar
        * nd_plasma_electron_line_20
        * safe_sqrt(afuel)
        / safe_sqrt(ten / 10.0)
    )


def shimomura_confinement_time(
    rmajor, rminor, b_plasma_toroidal_on_axis, kappa95, afuel
):
    """Shimomura (S) optimized H-mode scaling. `ConfinementTimeModel.SHIMOMURA` (4)."""
    return (
        0.045
        * rmajor
        * rminor
        * b_plasma_toroidal_on_axis
        * safe_sqrt(kappa95)
        * safe_sqrt(afuel)
    )


def kaye_goldston_confinement_time(
    kappa95,
    cur_plasma_ma,
    n20,
    rmajor,
    afuel,
    b_plasma_toroidal_on_axis,
    rminor,
    p_plasma_loss_mw,
):
    """Kaye-Goldston (KG) L-mode scaling. `ConfinementTimeModel.KAYE_GOLDSTON` (5).

    **`calculate_confinement_time`'s own call site passes its positional arguments in
    the wrong order for this signature** -- see the audit record's "A latent PROCESS
    bug, ported faithfully". This function itself is correct against its own parameter
    names; the bug is entirely in how the composite dispatcher below calls it, and is
    reproduced there exactly, not fixed.
    """
    return (
        0.055
        * safe_pow(kappa95, 0.28)
        * cur_plasma_ma**1.24
        * safe_pow(n20, 0.26)
        * rmajor**1.65
        * safe_sqrt(afuel / 1.5)
        / (
            safe_pow(b_plasma_toroidal_on_axis, 0.09)
            * safe_pow(rminor, 0.49)
            * safe_pow(p_plasma_loss_mw, 0.58)
        )
    )


def iter_89p_confinement_time(
    cur_plasma_ma,
    rmajor,
    rminor,
    kappa,
    nd_plasma_electron_line_20,
    b_plasma_toroidal_on_axis,
    afuel,
    p_plasma_loss_mw,
):
    """ITER Power scaling - ITER 89-P (L-mode). `ConfinementTimeModel.ITER_89P` (6)."""
    return (
        0.048
        * safe_pow(cur_plasma_ma, 0.85)
        * rmajor**1.2
        * safe_pow(rminor, 0.3)
        * safe_sqrt(kappa)
        * safe_pow(nd_plasma_electron_line_20, 0.1)
        * safe_pow(b_plasma_toroidal_on_axis, 0.2)
        * safe_sqrt(afuel)
        / safe_sqrt(p_plasma_loss_mw)
    )


def iter_89_0_confinement_time(
    cur_plasma_ma,
    rmajor,
    rminor,
    kappa,
    nd_plasma_electron_line_20,
    b_plasma_toroidal_on_axis,
    afuel,
    p_plasma_loss_mw,
):
    """ITER Offset linear scaling - ITER 89-O (L-mode). `ConfinementTimeModel.ITER_89_0`
    (7).
    """
    term1 = (
        0.04
        * safe_pow(cur_plasma_ma, 0.5)
        * safe_pow(rmajor, 0.3)
        * safe_pow(rminor, 0.8)
        * safe_pow(kappa, 0.6)
        * safe_pow(afuel, 0.5)
    )
    term2 = (
        0.064
        * safe_pow(cur_plasma_ma, 0.8)
        * rmajor**1.6
        * safe_pow(rminor, 0.6)
        * safe_pow(kappa, 0.5)
        * safe_pow(nd_plasma_electron_line_20, 0.6)
        * safe_pow(b_plasma_toroidal_on_axis, 0.35)
        * safe_pow(afuel, 0.2)
        / p_plasma_loss_mw
    )
    return term1 + term2


def rebut_lallia_confinement_time(
    rminor,
    rmajor,
    kappa,
    afuel,
    cur_plasma_ma,
    zeff,
    nd_plasma_electron_line_20,
    b_plasma_toroidal_on_axis,
    p_plasma_loss_mw,
):
    """Rebut-Lallia offset linear scaling (L-mode). `ConfinementTimeModel.REBUT_LALLIA`
    (8).
    """
    rll = safe_pow(rminor**2 * rmajor * kappa, 1.0 / 3.0)
    term1 = 1.2e-2 * cur_plasma_ma * rll**1.5 / safe_sqrt(zeff)
    term2 = (
        0.146
        * safe_pow(nd_plasma_electron_line_20, 0.75)
        * safe_sqrt(cur_plasma_ma)
        * safe_sqrt(b_plasma_toroidal_on_axis)
        * rll**2.75
        * safe_pow(zeff, 0.25)
        / p_plasma_loss_mw
    )
    return 1.65 * safe_sqrt(afuel / 2.0) * (term1 + term2)


def goldston_confinement_time(
    cur_plasma_ma, rmajor, rminor, kappa95, afuel, p_plasma_loss_mw
):
    """Goldston scaling (L-mode). `ConfinementTimeModel.GOLDSTON` (9)."""
    return (
        0.037
        * cur_plasma_ma
        * rmajor**1.75
        * rminor ** (-0.37)
        * safe_sqrt(kappa95)
        * safe_sqrt(afuel / 1.5)
        / safe_sqrt(p_plasma_loss_mw)
    )


def t10_confinement_time(
    nd_plasma_electron_line_20,
    rmajor,
    qstar,
    b_plasma_toroidal_on_axis,
    rminor,
    kappa95,
    p_plasma_loss_mw,
    zeff,
    cur_plasma_ma,
):
    """T-10 scaling (L-mode). `ConfinementTimeModel.T_10` (10)."""
    denfac = (
        nd_plasma_electron_line_20 * rmajor * qstar / (1.3 * b_plasma_toroidal_on_axis)
    )
    denfac = jnp.minimum(1.0, denfac)
    return (
        0.095
        * rmajor
        * rminor
        * b_plasma_toroidal_on_axis
        * safe_sqrt(kappa95)
        * denfac
        / safe_pow(p_plasma_loss_mw, 0.4)
        * safe_pow(
            zeff**2 * cur_plasma_ma**4 / (rmajor * rminor * qstar**3 * kappa95**1.5),
            0.08,
        )
    )


def jaeri_confinement_time(
    kappa95,
    rminor,
    afuel,
    n20,
    cur_plasma_ma,
    b_plasma_toroidal_on_axis,
    rmajor,
    qstar,
    zeff,
    p_plasma_loss_mw,
):
    """JAERI / Odajima-Shimomura L-mode scaling. `ConfinementTimeModel.JAERI` (11)."""
    gjaeri = (
        safe_pow(zeff, 0.4)
        * safe_pow((15.0 - zeff) / 20.0, 0.6)
        * safe_pow(3.0 * qstar * (qstar + 5.0) / ((qstar + 2.0) * (qstar + 7.0)), 0.6)
    )
    return (
        0.085 * kappa95 * rminor**2 * safe_sqrt(afuel)
        + 0.069
        * safe_pow(n20, 0.6)
        * cur_plasma_ma
        * safe_pow(b_plasma_toroidal_on_axis, 0.2)
        * safe_pow(rminor, 0.4)
        * rmajor**1.6
        * safe_sqrt(afuel)
        * gjaeri
        * safe_pow(kappa95, 0.2)
        / p_plasma_loss_mw
    )


def kaye_big_confinement_time(
    rmajor,
    rminor,
    b_plasma_toroidal_on_axis,
    kappa95,
    cur_plasma_ma,
    n20,
    afuel,
    p_plasma_loss_mw,
):
    """Kaye-Big scaling (based only on big tokamak data). `ConfinementTimeModel.
    KAYE_BIG` (12).
    """
    return (
        0.105
        * safe_sqrt(rmajor)
        * safe_pow(rminor, 0.8)
        * safe_pow(b_plasma_toroidal_on_axis, 0.3)
        * safe_pow(kappa95, 0.25)
        * safe_pow(cur_plasma_ma, 0.85)
        * safe_pow(n20, 0.1)
        * safe_sqrt(afuel)
        / safe_sqrt(p_plasma_loss_mw)
    )


def iter_h90_p_confinement_time(
    cur_plasma_ma,
    rmajor,
    rminor,
    kappa,
    nd_plasma_electron_line_20,
    b_plasma_toroidal_on_axis,
    afuel,
    p_plasma_loss_mw,
):
    """ITER H-mode scaling - ITER H90-P. `ConfinementTimeModel.ITER_H90_P` (13)."""
    return (
        0.064
        * safe_pow(cur_plasma_ma, 0.87)
        * rmajor**1.82
        * rminor ** (-0.12)
        * safe_pow(kappa, 0.35)
        * safe_pow(nd_plasma_electron_line_20, 0.09)
        * safe_pow(b_plasma_toroidal_on_axis, 0.15)
        * safe_sqrt(afuel)
        / safe_sqrt(p_plasma_loss_mw)
    )


def riedel_l_confinement_time(
    cur_plasma_ma,
    rmajor,
    rminor,
    kappa95,
    nd_plasma_electron_line_20,
    b_plasma_toroidal_on_axis,
    p_plasma_loss_mw,
):
    """Riedel scaling (L-mode). `ConfinementTimeModel.RIEDEL_L` (15)."""
    return (
        0.044
        * safe_pow(cur_plasma_ma, 0.93)
        * rmajor**1.37
        * rminor ** (-0.049)
        * safe_pow(kappa95, 0.588)
        * safe_pow(nd_plasma_electron_line_20, 0.078)
        * safe_pow(b_plasma_toroidal_on_axis, 0.152)
        / safe_pow(p_plasma_loss_mw, 0.537)
    )


def christiansen_confinement_time(
    cur_plasma_ma,
    rmajor,
    rminor,
    kappa95,
    nd_plasma_electron_line_20,
    b_plasma_toroidal_on_axis,
    p_plasma_loss_mw,
    afuel,
):
    """Christiansen et al scaling (L-mode). `ConfinementTimeModel.CHRISTIANSEN` (16)."""
    return (
        0.24
        * safe_pow(cur_plasma_ma, 0.79)
        * safe_pow(rmajor, 0.56)
        * rminor**1.46
        * safe_pow(kappa95, 0.73)
        * safe_pow(nd_plasma_electron_line_20, 0.41)
        * safe_pow(b_plasma_toroidal_on_axis, 0.29)
        / (safe_pow(p_plasma_loss_mw, 0.79) * safe_pow(afuel, 0.02))
    )


def lackner_gottardi_confinement_time(
    cur_plasma_ma,
    rmajor,
    rminor,
    kappa95,
    nd_plasma_electron_line_20,
    b_plasma_toroidal_on_axis,
    p_plasma_loss_mw,
):
    """Lackner-Gottardi scaling (L-mode). `ConfinementTimeModel.LACKNER_GOTTARDI`
    (17).
    """
    qhat = (
        (1.0 + kappa95**2)
        * rminor**2
        * b_plasma_toroidal_on_axis
        / (0.4 * cur_plasma_ma * rmajor)
    )
    return (
        0.12
        * safe_pow(cur_plasma_ma, 0.8)
        * rmajor**1.8
        * safe_pow(rminor, 0.4)
        * kappa95
        * (1.0 + kappa95) ** (-0.8)
        * safe_pow(nd_plasma_electron_line_20, 0.6)
        * safe_pow(qhat, 0.4)
        / safe_pow(p_plasma_loss_mw, 0.6)
    )


def neo_kaye_confinement_time(
    cur_plasma_ma,
    rmajor,
    rminor,
    kappa95,
    nd_plasma_electron_line_20,
    b_plasma_toroidal_on_axis,
    p_plasma_loss_mw,
):
    """Neo-Kaye scaling (L-mode). `ConfinementTimeModel.NEO_KAYE` (18)."""
    return (
        0.063
        * cur_plasma_ma**1.12
        * rmajor**1.3
        * rminor ** (-0.04)
        * safe_pow(kappa95, 0.28)
        * safe_pow(nd_plasma_electron_line_20, 0.14)
        * safe_pow(b_plasma_toroidal_on_axis, 0.04)
        / safe_pow(p_plasma_loss_mw, 0.59)
    )


def riedel_h_confinement_time(
    cur_plasma_ma,
    rmajor,
    rminor,
    kappa95,
    nd_plasma_electron_line_20,
    b_plasma_toroidal_on_axis,
    afuel,
    p_plasma_loss_mw,
):
    """Riedel scaling (H-mode). `ConfinementTimeModel.RIEDEL_H` (19)."""
    return (
        0.1
        * safe_sqrt(afuel)
        * safe_pow(cur_plasma_ma, 0.884)
        * rmajor**1.24
        * rminor ** (-0.23)
        * safe_pow(kappa95, 0.317)
        * safe_pow(b_plasma_toroidal_on_axis, 0.207)
        * safe_pow(nd_plasma_electron_line_20, 0.105)
        / safe_pow(p_plasma_loss_mw, 0.486)
    )


def iter_h90_p_amended_confinement_time(
    cur_plasma_ma, b_plasma_toroidal_on_axis, afuel, rmajor, p_plasma_loss_mw, kappa
):
    """Amended ITER H90-P law. `ConfinementTimeModel.ITER_H90_P_AMENDED` (20)."""
    return (
        0.082
        * cur_plasma_ma**1.02
        * safe_pow(b_plasma_toroidal_on_axis, 0.15)
        * safe_sqrt(afuel)
        * rmajor**1.60
        / (safe_pow(p_plasma_loss_mw, 0.47) * safe_pow(kappa, 0.19))
    )


def sudo_et_al_confinement_time(
    rmajor,
    rminor,
    nd_plasma_electron_line_20,
    b_plasma_toroidal_on_axis,
    p_plasma_loss_mw,
):
    """Sudo et al. scaling (stellarators/heliotron). `ConfinementTimeModel.SUDO_ET_AL`
    (21).
    """
    return (
        0.17
        * safe_pow(rmajor, 0.75)
        * rminor**2
        * safe_pow(nd_plasma_electron_line_20, 0.69)
        * safe_pow(b_plasma_toroidal_on_axis, 0.84)
        * p_plasma_loss_mw ** (-0.58)
    )


def gyro_reduced_bohm_confinement_time(
    b_plasma_toroidal_on_axis,
    nd_plasma_electron_line_20,
    p_plasma_loss_mw,
    rminor,
    rmajor,
):
    """Gyro-reduced Bohm scaling. `ConfinementTimeModel.GYRO_REDUCED_BOHM` (22)."""
    return (
        0.25
        * safe_pow(b_plasma_toroidal_on_axis, 0.8)
        * safe_pow(nd_plasma_electron_line_20, 0.6)
        * p_plasma_loss_mw ** (-0.6)
        * rminor**2.4
        * safe_pow(rmajor, 0.6)
    )


def lackner_gottardi_stellarator_confinement_time(
    rmajor,
    rminor,
    nd_plasma_electron_line_20,
    b_plasma_toroidal_on_axis,
    p_plasma_loss_mw,
    q,
):
    """Lackner-Gottardi stellarator scaling. `ConfinementTimeModel.
    LACKNER_GOTTARDI_STELLARATOR` (23).
    """
    return (
        0.17
        * rmajor
        * rminor**2
        * safe_pow(nd_plasma_electron_line_20, 0.6)
        * safe_pow(b_plasma_toroidal_on_axis, 0.8)
        * p_plasma_loss_mw ** (-0.6)
        * safe_pow(q, 0.4)
    )


def iter_93h_confinement_time(
    cur_plasma_ma,
    b_plasma_toroidal_on_axis,
    p_plasma_loss_mw,
    afuel,
    rmajor,
    nd_plasma_electron_line_20,
    aspect,
    kappa,
):
    """ITER-93H scaling, ELM-free. `ConfinementTimeModel.ITER_93H` (24)."""
    return (
        0.036
        * cur_plasma_ma**1.06
        * safe_pow(b_plasma_toroidal_on_axis, 0.32)
        * p_plasma_loss_mw ** (-0.67)
        * safe_pow(afuel, 0.41)
        * rmajor**1.79
        * safe_pow(nd_plasma_electron_line_20, 0.17)
        * safe_pow(aspect, 0.11)
        * safe_pow(kappa, 0.66)
    )


def iter_h97p_confinement_time(
    cur_plasma_ma,
    b_plasma_toroidal_on_axis,
    p_plasma_loss_mw,
    nd_plasma_electron_line_19,
    rmajor,
    aspect,
    kappa,
    afuel,
):
    """ELM-free ITER H-mode scaling - ITER H97-P. `ConfinementTimeModel.ITER_H97P`
    (26).
    """
    return (
        0.031
        * safe_pow(cur_plasma_ma, 0.95)
        * safe_pow(b_plasma_toroidal_on_axis, 0.25)
        * p_plasma_loss_mw ** (-0.67)
        * safe_pow(nd_plasma_electron_line_19, 0.35)
        * rmajor**1.92
        * aspect ** (-0.08)
        * safe_pow(kappa, 0.63)
        * safe_pow(afuel, 0.42)
    )


def iter_h97p_elmy_confinement_time(
    cur_plasma_ma,
    b_plasma_toroidal_on_axis,
    p_plasma_loss_mw,
    nd_plasma_electron_line_19,
    rmajor,
    aspect,
    kappa,
    afuel,
):
    """ELMy ITER H-mode scaling - ITER H97-P(y). `ConfinementTimeModel.
    ITER_H97P_ELMY` (27).
    """
    return (
        0.029
        * safe_pow(cur_plasma_ma, 0.90)
        * safe_pow(b_plasma_toroidal_on_axis, 0.20)
        * p_plasma_loss_mw ** (-0.66)
        * safe_pow(nd_plasma_electron_line_19, 0.40)
        * rmajor**2.03
        * aspect ** (-0.19)
        * safe_pow(kappa, 0.92)
        * safe_pow(afuel, 0.2)
    )


def iter_96p_confinement_time(
    cur_plasma_ma,
    b_plasma_toroidal_on_axis,
    kappa95,
    rmajor,
    aspect,
    nd_plasma_electron_line_19,
    afuel,
    p_plasma_loss_mw,
):
    """ITER-96P (= ITER-97L) L-mode scaling. `ConfinementTimeModel.ITER_96P` (28)."""
    return (
        0.023
        * safe_pow(cur_plasma_ma, 0.96)
        * safe_pow(b_plasma_toroidal_on_axis, 0.03)
        * safe_pow(kappa95, 0.64)
        * rmajor**1.83
        * safe_pow(aspect, 0.06)
        * safe_pow(nd_plasma_electron_line_19, 0.40)
        * safe_pow(afuel, 0.20)
        * p_plasma_loss_mw ** (-0.73)
    )


def valovic_elmy_confinement_time(
    cur_plasma_ma,
    b_plasma_toroidal_on_axis,
    nd_plasma_electron_line_19,
    afuel,
    rmajor,
    rminor,
    kappa,
    p_plasma_loss_mw,
):
    """Valovic modified ELMy-H mode scaling. `ConfinementTimeModel.VALOVIC_ELMY`
    (29).
    """
    return (
        0.067
        * safe_pow(cur_plasma_ma, 0.9)
        * safe_pow(b_plasma_toroidal_on_axis, 0.17)
        * safe_pow(nd_plasma_electron_line_19, 0.45)
        * safe_pow(afuel, 0.05)
        * rmajor**1.316
        * safe_pow(rminor, 0.79)
        * safe_pow(kappa, 0.56)
        * p_plasma_loss_mw ** (-0.68)
    )


def kaye_confinement_time(
    cur_plasma_ma,
    b_plasma_toroidal_on_axis,
    kappa,
    rmajor,
    aspect,
    nd_plasma_electron_line_19,
    afuel,
    p_plasma_loss_mw,
):
    """Kaye PPPL Workshop April 1998 L-mode scaling. `ConfinementTimeModel.KAYE`
    (30).
    """
    return (
        0.021
        * safe_pow(cur_plasma_ma, 0.81)
        * safe_pow(b_plasma_toroidal_on_axis, 0.14)
        * safe_pow(kappa, 0.7)
        * rmajor**2.01
        * aspect ** (-0.18)
        * safe_pow(nd_plasma_electron_line_19, 0.47)
        * safe_pow(afuel, 0.25)
        * p_plasma_loss_mw ** (-0.73)
    )


def iter_pb98py_confinement_time(
    cur_plasma_ma,
    b_plasma_toroidal_on_axis,
    nd_plasma_electron_line_19,
    p_plasma_loss_mw,
    rmajor,
    kappa,
    aspect,
    afuel,
):
    """ITERH-PB98P(y) ELMy H-mode scaling. `ConfinementTimeModel.ITER_PB98P_Y` (31).

    Source's `kappa` parameter (its own docstring: "Plasma separatrix elongation") is,
    at `calculate_confinement_time`'s one call site, always fed `.physics.kappa_ipb`
    -- the same IPB-corrected elongation every sibling IPB98-family scaling below uses
    under a parameter actually named `kappa_ipb`. Almost certainly a naming slip in the
    source (this scaling's parameter should have been called `kappa_ipb` too, not a
    value bug), since the value supplied is consistent with the whole family -- see the
    audit record's "A minor naming inconsistency". Kept as `kappa` here, unchanged from
    source, so the port's signature matches PROCESS's own.
    """
    return (
        0.0615
        * safe_pow(cur_plasma_ma, 0.9)
        * safe_pow(b_plasma_toroidal_on_axis, 0.1)
        * safe_pow(nd_plasma_electron_line_19, 0.4)
        * p_plasma_loss_mw ** (-0.66)
        * rmajor**2
        * safe_pow(kappa, 0.75)
        * aspect ** (-0.66)
        * safe_pow(afuel, 0.2)
    )


def iter_ipb98y_confinement_time(
    cur_plasma_ma,
    b_plasma_toroidal_on_axis,
    nd_plasma_electron_line_19,
    p_plasma_loss_mw,
    rmajor,
    kappa,
    aspect,
    afuel,
):
    """IPB98(y) ELMy H-mode scaling -- uses the *true* separatrix elongation, unlike
    the other IPB98 laws (source's own note). `ConfinementTimeModel.IPB98_Y` (32).
    """
    return (
        0.0365
        * safe_pow(cur_plasma_ma, 0.97)
        * safe_pow(b_plasma_toroidal_on_axis, 0.08)
        * safe_pow(nd_plasma_electron_line_19, 0.41)
        * p_plasma_loss_mw ** (-0.63)
        * rmajor**1.93
        * safe_pow(kappa, 0.67)
        * aspect ** (-0.23)
        * safe_pow(afuel, 0.2)
    )


def iter_ipb98y1_confinement_time(
    cur_plasma_ma,
    b_plasma_toroidal_on_axis,
    nd_plasma_electron_line_19,
    p_plasma_loss_mw,
    rmajor,
    kappa_ipb,
    aspect,
    afuel,
):
    """IPB98(y,1) ELMy H-mode scaling. `ConfinementTimeModel.ITER_IPB98Y1` (33)."""
    return (
        0.0503
        * safe_pow(cur_plasma_ma, 0.91)
        * safe_pow(b_plasma_toroidal_on_axis, 0.15)
        * safe_pow(nd_plasma_electron_line_19, 0.44)
        * p_plasma_loss_mw ** (-0.65)
        * rmajor**2.05
        * safe_pow(kappa_ipb, 0.72)
        * aspect ** (-0.57)
        * safe_pow(afuel, 0.13)
    )


def iter_ipb98y2_confinement_time(
    cur_plasma_ma,
    b_plasma_toroidal_on_axis,
    nd_plasma_electron_line_19,
    p_plasma_loss_mw,
    rmajor,
    kappa_ipb,
    aspect,
    afuel,
):
    """IPB98(y,2) ELMy H-mode scaling. `ConfinementTimeModel.ITER_IPB98Y2` (34)."""
    return (
        0.0562
        * safe_pow(cur_plasma_ma, 0.93)
        * safe_pow(b_plasma_toroidal_on_axis, 0.15)
        * safe_pow(nd_plasma_electron_line_19, 0.41)
        * p_plasma_loss_mw ** (-0.69)
        * rmajor**1.97
        * safe_pow(kappa_ipb, 0.78)
        * aspect ** (-0.58)
        * safe_pow(afuel, 0.19)
    )


def iter_ipb98y3_confinement_time(
    cur_plasma_ma,
    b_plasma_toroidal_on_axis,
    nd_plasma_electron_line_19,
    p_plasma_loss_mw,
    rmajor,
    kappa_ipb,
    aspect,
    afuel,
):
    """IPB98(y,3) ELMy H-mode scaling. `ConfinementTimeModel.ITER_IPB98Y3` (35)."""
    return (
        0.0564
        * safe_pow(cur_plasma_ma, 0.88)
        * safe_pow(b_plasma_toroidal_on_axis, 0.07)
        * safe_pow(nd_plasma_electron_line_19, 0.40)
        * p_plasma_loss_mw ** (-0.69)
        * rmajor**2.15
        * safe_pow(kappa_ipb, 0.78)
        * aspect ** (-0.64)
        * safe_pow(afuel, 0.20)
    )


def iter_ipb98y4_confinement_time(
    cur_plasma_ma,
    b_plasma_toroidal_on_axis,
    nd_plasma_electron_line_19,
    p_plasma_loss_mw,
    rmajor,
    kappa_ipb,
    aspect,
    afuel,
):
    """IPB98(y,4) ELMy H-mode scaling. `ConfinementTimeModel.ITER_IPB98Y4` (36)."""
    return (
        0.0587
        * safe_pow(cur_plasma_ma, 0.85)
        * safe_pow(b_plasma_toroidal_on_axis, 0.29)
        * safe_pow(nd_plasma_electron_line_19, 0.39)
        * p_plasma_loss_mw ** (-0.70)
        * rmajor**2.08
        * safe_pow(kappa_ipb, 0.76)
        * aspect ** (-0.69)
        * safe_pow(afuel, 0.17)
    )


def iss95_stellarator_confinement_time(
    rminor,
    rmajor,
    nd_plasma_electron_line_19,
    b_plasma_toroidal_on_axis,
    p_plasma_loss_mw,
    iotabar,
):
    """ISS95 stellarator scaling. `ConfinementTimeModel.ISS95_STELLARATOR` (37)."""
    return (
        0.079
        * rminor**2.21
        * safe_pow(rmajor, 0.65)
        * safe_pow(nd_plasma_electron_line_19, 0.51)
        * safe_pow(b_plasma_toroidal_on_axis, 0.83)
        * p_plasma_loss_mw ** (-0.59)
        * safe_pow(iotabar, 0.4)
    )


def iss04_stellarator_confinement_time(
    rminor,
    rmajor,
    nd_plasma_electron_line_19,
    b_plasma_toroidal_on_axis,
    p_plasma_loss_mw,
    iotabar,
):
    """ISS04 stellarator scaling. `ConfinementTimeModel.ISS04_STELLARATOR` (38)."""
    return (
        0.134
        * rminor**2.28
        * safe_pow(rmajor, 0.64)
        * safe_pow(nd_plasma_electron_line_19, 0.54)
        * safe_pow(b_plasma_toroidal_on_axis, 0.84)
        * p_plasma_loss_mw ** (-0.61)
        * safe_pow(iotabar, 0.41)
    )


def ds03_confinement_time(
    cur_plasma_ma,
    b_plasma_toroidal_on_axis,
    nd_plasma_electron_line_19,
    p_plasma_loss_mw,
    rmajor,
    kappa95,
    aspect,
    afuel,
):
    """DS03 beta-independent H-mode scaling. `ConfinementTimeModel.DS03` (39)."""
    return (
        0.028
        * safe_pow(cur_plasma_ma, 0.83)
        * safe_pow(b_plasma_toroidal_on_axis, 0.07)
        * safe_pow(nd_plasma_electron_line_19, 0.49)
        * p_plasma_loss_mw ** (-0.55)
        * rmajor**2.11
        * safe_pow(kappa95, 0.75)
        * aspect ** (-0.3)
        * safe_pow(afuel, 0.14)
    )


def murari_confinement_time(
    cur_plasma_ma,
    rmajor,
    kappa_ipb,
    nd_plasma_electron_line_19,
    b_plasma_toroidal_on_axis,
    p_plasma_loss_mw,
):
    """Murari "Non-power law" H-mode scaling. `ConfinementTimeModel.MURARI` (40)."""
    return (
        0.0367
        * cur_plasma_ma**1.006
        * rmajor**1.731
        * kappa_ipb**1.450
        * p_plasma_loss_mw ** (-0.735)
        * (
            safe_pow(nd_plasma_electron_line_19, 0.448)
            / (
                1.0
                + jnp.exp(
                    -9.403
                    * (nd_plasma_electron_line_19 / b_plasma_toroidal_on_axis) ** -1.365
                )
            )
        )
    )


def petty08_confinement_time(
    cur_plasma_ma,
    b_plasma_toroidal_on_axis,
    nd_plasma_electron_line_19,
    p_plasma_loss_mw,
    rmajor,
    kappa_ipb,
    aspect,
):
    """Beta independent dimensionless Petty08 scaling. `ConfinementTimeModel.PETTY08`
    (41).
    """
    return (
        0.052
        * safe_pow(cur_plasma_ma, 0.75)
        * safe_pow(b_plasma_toroidal_on_axis, 0.3)
        * safe_pow(nd_plasma_electron_line_19, 0.32)
        * p_plasma_loss_mw ** (-0.47)
        * rmajor**2.09
        * safe_pow(kappa_ipb, 0.88)
        * aspect ** (-0.84)
    )


def lang_high_density_confinement_time(
    plasma_current,
    b_plasma_toroidal_on_axis,
    nd_plasma_electron_line,
    p_plasma_loss_mw,
    rmajor,
    rminor,
    q,
    qstar,
    aspect,
    afuel,
    kappa_ipb,
):
    """Lang high density relevant confinement scaling. `ConfinementTimeModel.
    LANG_HIGH_DENSITY` (42).
    """
    qratio = q / qstar
    n_gw = 1.0e14 * plasma_current / (jnp.pi * rminor * rminor)
    nratio = nd_plasma_electron_line / n_gw
    return (
        6.94e-7
        * plasma_current**1.3678
        * safe_pow(b_plasma_toroidal_on_axis, 0.12)
        * safe_pow(nd_plasma_electron_line, 0.032236)
        * (p_plasma_loss_mw * 1.0e6) ** (-0.74)
        * rmajor**1.2345
        * safe_pow(kappa_ipb, 0.37)
        * aspect**2.48205
        * safe_pow(afuel, 0.2)
        * safe_pow(qratio, 0.77)
        * safe_pow(aspect, -0.9 * jnp.log(aspect))
        * safe_pow(nratio, -0.22 * jnp.log(nratio))
    )


def hubbard_nominal_confinement_time(
    cur_plasma_ma,
    b_plasma_toroidal_on_axis,
    nd_plasma_electron_line_20,
    p_plasma_loss_mw,
):
    """Hubbard 2017 I-mode scaling - nominal. `ConfinementTimeModel.HUBBARD_NOMINAL`
    (43).
    """
    return (
        0.014
        * safe_pow(cur_plasma_ma, 0.68)
        * safe_pow(b_plasma_toroidal_on_axis, 0.77)
        * safe_pow(nd_plasma_electron_line_20, 0.02)
        * p_plasma_loss_mw ** (-0.29)
    )


def hubbard_lower_confinement_time(
    cur_plasma_ma,
    b_plasma_toroidal_on_axis,
    nd_plasma_electron_line_20,
    p_plasma_loss_mw,
):
    """Hubbard 2017 I-mode scaling - lower. `ConfinementTimeModel.HUBBARD_LOWER`
    (44).
    """
    return (
        0.014
        * safe_pow(cur_plasma_ma, 0.60)
        * safe_pow(b_plasma_toroidal_on_axis, 0.70)
        * nd_plasma_electron_line_20 ** (-0.03)
        * p_plasma_loss_mw ** (-0.33)
    )


def hubbard_upper_confinement_time(
    cur_plasma_ma,
    b_plasma_toroidal_on_axis,
    nd_plasma_electron_line_20,
    p_plasma_loss_mw,
):
    """Hubbard 2017 I-mode scaling - upper. `ConfinementTimeModel.HUBBARD_UPPER`
    (45).
    """
    return (
        0.014
        * safe_pow(cur_plasma_ma, 0.76)
        * safe_pow(b_plasma_toroidal_on_axis, 0.84)
        * safe_pow(nd_plasma_electron_line_20, 0.07)
        * p_plasma_loss_mw ** (-0.25)
    )


def menard_nstx_confinement_time(
    cur_plasma_ma,
    b_plasma_toroidal_on_axis,
    nd_plasma_electron_line_19,
    p_plasma_loss_mw,
    rmajor,
    kappa_ipb,
    aspect,
    afuel,
):
    """Menard NSTX ELMy H-mode scaling. `ConfinementTimeModel.MENARD_NSTX` (46)."""
    return (
        0.095
        * safe_pow(cur_plasma_ma, 0.57)
        * b_plasma_toroidal_on_axis**1.08
        * safe_pow(nd_plasma_electron_line_19, 0.44)
        * p_plasma_loss_mw ** (-0.73)
        * rmajor**1.97
        * safe_pow(kappa_ipb, 0.78)
        * aspect ** (-0.58)
        * safe_pow(afuel, 0.19)
    )


def menard_nstx_petty08_hybrid_confinement_time(
    cur_plasma_ma,
    b_plasma_toroidal_on_axis,
    nd_plasma_electron_line_19,
    p_plasma_loss_mw,
    rmajor,
    kappa_ipb,
    aspect,
    afuel,
):
    """Menard NSTX-Petty08 hybrid. `ConfinementTimeModel.MENARD_NSTX_PETTY08_HYBRID`
    (47).

    Source is a three-way `if`/`elif`/`else` on `1/aspect` against 0.4/0.6: below 0.4
    use Petty08, above 0.6 use NSTX, in between a linear interpolation of the two. That
    interpolation already reduces to Petty08 exactly at `1/aspect = 0.4` and to NSTX
    exactly at `1/aspect = 0.6` (verified algebraically), so the whole thing is one
    continuous, everywhere-differentiable function of `aspect`: clip the interpolation
    fraction to `[0, 1]` and blend. `aspect` is a plain differentiable argument here,
    not a switch (`needs-lax-cond-or-where`, `workaround-known` -- see the audit
    record).
    """
    petty = petty08_confinement_time(
        cur_plasma_ma,
        b_plasma_toroidal_on_axis,
        nd_plasma_electron_line_19,
        p_plasma_loss_mw,
        rmajor,
        kappa_ipb,
        aspect,
    )
    nstx = menard_nstx_confinement_time(
        cur_plasma_ma,
        b_plasma_toroidal_on_axis,
        nd_plasma_electron_line_19,
        p_plasma_loss_mw,
        rmajor,
        kappa_ipb,
        aspect,
        afuel,
    )
    frac = jnp.clip(((1.0 / aspect) - 0.4) / (0.6 - 0.4), 0.0, 1.0)
    return frac * nstx + (1.0 - frac) * petty


def nstx_gyro_bohm_confinement_time(
    cur_plasma_ma,
    b_plasma_toroidal_on_axis,
    p_plasma_loss_mw,
    rmajor,
    nd_plasma_electron_line_20,
):
    """NSTX gyro-Bohm (Buxton) scaling. `ConfinementTimeModel.NSTX_GYRO_BOHM` (48)."""
    return (
        0.21
        * safe_pow(cur_plasma_ma, 0.54)
        * safe_pow(b_plasma_toroidal_on_axis, 0.91)
        * p_plasma_loss_mw ** (-0.38)
        * rmajor**2.14
        * nd_plasma_electron_line_20 ** (-0.05)
    )


def itpa20_confinement_time(
    cur_plasma_ma,
    b_plasma_toroidal_on_axis,
    nd_plasma_electron_line_19,
    p_plasma_loss_mw,
    rmajor,
    triang,
    kappa_ipb,
    eps,
    aion,
):
    """ITPA20 (Issue #3164) scaling. `ConfinementTimeModel.ITPA20` (49)."""
    return (
        0.0534
        * safe_pow(cur_plasma_ma, 0.976)
        * safe_pow(b_plasma_toroidal_on_axis, 0.218)
        * safe_pow(nd_plasma_electron_line_19, 0.2442)
        * p_plasma_loss_mw ** (-0.6687)
        * rmajor**1.710
        * safe_pow(1 + triang, 0.362)
        * safe_pow(kappa_ipb, 0.799)
        * safe_pow(eps, 0.354)
        * safe_pow(aion, 0.195)
    )


def itpa20_il_confinement_time(
    cur_plasma_ma,
    b_plasma_toroidal_on_axis,
    p_plasma_loss_mw,
    nd_plasma_electron_line_19,
    aion,
    rmajor,
    triang,
    kappa_ipb,
):
    """ITPA20-IL (Issue #1852) scaling. `ConfinementTimeModel.ITPA20_IL` (50)."""
    return (
        0.0670
        * cur_plasma_ma**1.291
        * b_plasma_toroidal_on_axis**-0.134
        * safe_pow(nd_plasma_electron_line_19, 0.1473)
        * p_plasma_loss_mw ** (-0.6442)
        * rmajor**1.194
        * safe_pow(1 + triang, 0.560)
        * safe_pow(kappa_ipb, 0.673)
        * safe_pow(aion, 0.302)
    )


def ncst_confinement_time(
    cur_plasma_ma,
    b_plasma_toroidal_on_axis,
    p_plasma_loss_mw,
    nd_plasma_electron_line_19,
):
    """NCST spherical tokamak L-mode scaling. `ConfinementTimeModel.NCST` (51)."""
    return (
        0.11
        * safe_pow(cur_plasma_ma, 0.33)
        * b_plasma_toroidal_on_axis**1.03
        * p_plasma_loss_mw ** (-0.07)
        * nd_plasma_electron_line_19 ** (-0.01)
    )


def paz_soldan_nt_confinement_time(
    cur_plasma_ma,
    b_plasma_toroidal_on_axis,
    p_plasma_loss_mw,
    nd_plasma_electron_line_19,
):
    """Paz-Soldan negative triangularity scaling.

    `ConfinementTimeModel.PAZ_SOLDAN_NT` shares `NCST`'s enum value (51) in the source
    -- see the audit record's "A dead branch": `calculate_confinement_time` can never
    actually reach this formula through `i_confinement_time` (the `NCST` branch, tried
    first, already matches every value 51 request). Ported and tested standalone since
    the formula itself is a legitimate, independent scaling; **not** wired into
    `calculate_confinement_time`'s dispatch below, matching PROCESS's real behaviour
    exactly rather than the behaviour its source appears to have intended.
    """
    return (
        0.0821
        * cur_plasma_ma**1.02
        * safe_pow(b_plasma_toroidal_on_axis, 0.11)
        * safe_pow(nd_plasma_electron_line_19, 0.51)
        * p_plasma_loss_mw ** (-0.91)
    )


# ---------------------------------------------------------------------------
# `calculate_double_and_triple_product` -- already a clean, self-contained pure
# function in the source (no `self.data` access). Unchanged.
# ---------------------------------------------------------------------------


def calculate_double_and_triple_product(
    nd_plasma_electrons_vol_avg, temp_plasma_electrons_vol_avg_kev, t_energy_confinement
):
    """Plasma double (n*tau_E) and triple (n*T*tau_E) product. Ports `PlasmaConfinement
    Time.calculate_double_and_triple_product`, unchanged.

    Returns
    -------
    tuple
        `(ntau, nTtau)`.
    """
    ntau = t_energy_confinement * nd_plasma_electrons_vol_avg
    n_t_tau = ntau * temp_plasma_electrons_vol_avg_kev
    return ntau, n_t_tau


# ---------------------------------------------------------------------------
# `calculate_confinement_time` -- the composite dispatcher. `i_confinement_time` and
# `i_rad_loss` are switches: plain ints, ordinary Python branching, never traced (see
# module docstring). Every other implicit `self.data.physics.*` read the source method
# performs mid-body is promoted to an explicit argument here -- see the audit record's
# data footprint table for the full read/write list.
# ---------------------------------------------------------------------------


def calculate_confinement_time(
    m_fuel_amu,
    p_alpha_total_mw,
    aspect,
    b_plasma_toroidal_on_axis,
    nd_plasma_electrons_vol_avg,
    nd_plasma_electron_line,
    eps,
    hfact,
    i_confinement_time,
    i_plasma_ignited,
    kappa,
    kappa95,
    p_non_alpha_charged_mw,
    p_hcd_injected_total_mw,
    plasma_current,
    pden_plasma_core_rad_mw,
    rmajor,
    rminor,
    temp_plasma_electron_density_weighted_kev,
    q95,
    qstar,
    vol_plasma,
    zeff,
    eden_plasma_electrons_thermal_vol_avg,
    eden_plasma_ions_thermal_vol_avg,
    f_p_alpha_plasma_deposited,
    p_plasma_ohmic_mw,
    i_rad_loss,
    pden_plasma_rad_mw,
    pden_plasma_sync_mw,
    p_plasma_inner_rad_mw,
    triang,
    m_ions_total_amu,
    e_plasma_beta,
    tauee_in,
):
    """Confinement times and transport power loss terms. Ports `PlasmaConfinementTime.
    calculate_confinement_time`, `process/models/physics/confinement_time.py:58-1035`.

    The first 25 arguments (through `eden_plasma_ions_thermal_vol_avg`) are the
    source's own explicit signature, unchanged. The remaining 9 promote implicit
    `self.data.physics.*` reads the source method performs mid-body into explicit
    arguments -- none of them appear in the source's own parameter list, but all are
    read (and, for `kappa_ipb`/`t_energy_confinement_beta`, two are written) somewhere
    inside its body. See the audit record's data footprint table.

    Returns
    -------
    tuple
        `(pden_electron_transport_loss_mw, pden_ion_transport_loss_mw,
        t_electron_energy_confinement, t_ion_energy_confinement,
        t_plasma_energy_confinement, p_plasma_loss_mw, hstar, kappa_ipb,
        t_energy_confinement_beta)` -- the source's `ConfinementTimeData` field order,
        with the two extra `self.data.physics` writes (`kappa_ipb`,
        `t_energy_confinement_beta`) appended.

    Raises
    ------
    ValueError
        `i_confinement_time == 0` (`USER_INPUT`) or an unrecognised value -- both
        switch-driven, never traced. `USER_INPUT` is a confirmed dead branch in
        PROCESS itself; see the audit record's "A dead branch: `USER_INPUT`".
    """
    p_plasma_loss_mw = (
        f_p_alpha_plasma_deposited * p_alpha_total_mw
        + p_non_alpha_charged_mw
        + p_plasma_ohmic_mw
    )
    if PlasmaIgnitionModel(i_plasma_ignited) == PlasmaIgnitionModel.NON_IGNITED:
        p_plasma_loss_mw += p_hcd_injected_total_mw

    rad_loss_model = ConfinementRadiationLossModel(int(i_rad_loss))
    if rad_loss_model == ConfinementRadiationLossModel.FULL_RADIATION:
        p_plasma_loss_mw -= pden_plasma_rad_mw * vol_plasma
    elif rad_loss_model == ConfinementRadiationLossModel.CORE_ONLY:
        p_plasma_loss_mw -= pden_plasma_core_rad_mw * vol_plasma
    # NO_RADIATION: no adjustment.

    p_plasma_loss_mw = jnp.maximum(p_plasma_loss_mw, 1.0e-3)

    nd_plasma_electron_line_20 = nd_plasma_electron_line * 1.0e-20
    nd_plasma_electron_line_19 = nd_plasma_electron_line * 1.0e-19
    n20 = nd_plasma_electrons_vol_avg / 1.0e20
    cur_plasma_ma = plasma_current / 1.0e6

    kappa_ipb = calculate_iter_physics_basis_elongation(vol_plasma, rmajor, rminor)

    try:
        model = ConfinementTimeModel(i_confinement_time)
    except ValueError as exc:
        raise ValueError(
            f"Illegal value for i_confinement_time: {i_confinement_time}"
        ) from exc

    if model == ConfinementTimeModel.USER_INPUT:
        # Faithful reproduction of a confirmed real PROCESS bug, not a porting error --
        # see the audit record's "A dead branch: USER_INPUT". Source has this arm as a
        # standalone `if` (reading `tauee_in`), immediately followed by a *second*,
        # independent `if model == NEO_ALCATOR:` rather than `elif` -- so the
        # USER_INPUT arm's computed value is always discarded and execution falls
        # through the entire `elif` chain into the final `else: raise`. Verified
        # empirically against the live PROCESS reference (`i_confinement_time=0`
        # always raises `ProcessValueError`), not just read off the source. Reproduced
        # here exactly: this value is unreachable.
        raise ValueError(
            "Illegal value for i_confinement_time: 0 (USER_INPUT is dead code in "
            "PROCESS itself -- see confinement_time.md)"
        )
    if model == ConfinementTimeModel.NEO_ALCATOR:
        t_electron_confinement = neo_alcator_confinement_time(n20, rminor, rmajor, qstar)
    elif model == ConfinementTimeModel.MIRNOV:
        t_electron_confinement = mirnov_confinement_time(rminor, kappa95, cur_plasma_ma)
    elif model == ConfinementTimeModel.MEREZHKIN_MUHKOVATOV:
        t_electron_confinement = merezhkin_muhkovatov_confinement_time(
            rmajor,
            rminor,
            kappa95,
            qstar,
            nd_plasma_electron_line_20,
            m_fuel_amu,
            temp_plasma_electron_density_weighted_kev,
        )
    elif model == ConfinementTimeModel.SHIMOMURA:
        t_electron_confinement = shimomura_confinement_time(
            rmajor, rminor, b_plasma_toroidal_on_axis, kappa95, m_fuel_amu
        )
    elif model == ConfinementTimeModel.KAYE_GOLDSTON:
        # Faithful reproduction of a real PROCESS bug -- see `kaye_goldston_
        # confinement_time`'s docstring and the audit record. This positional call
        # does NOT match that function's own parameter order; kept exactly as
        # PROCESS calls it, not corrected.
        t_electron_confinement = kaye_goldston_confinement_time(
            cur_plasma_ma,
            rmajor,
            rminor,
            kappa,
            nd_plasma_electron_line_20,
            b_plasma_toroidal_on_axis,
            m_fuel_amu,
            p_plasma_loss_mw,
        )
    elif model == ConfinementTimeModel.ITER_89P:
        t_electron_confinement = iter_89p_confinement_time(
            cur_plasma_ma,
            rmajor,
            rminor,
            kappa,
            nd_plasma_electron_line_20,
            b_plasma_toroidal_on_axis,
            m_fuel_amu,
            p_plasma_loss_mw,
        )
    elif model == ConfinementTimeModel.ITER_89_0:
        t_electron_confinement = iter_89_0_confinement_time(
            cur_plasma_ma,
            rmajor,
            rminor,
            kappa,
            nd_plasma_electron_line_20,
            b_plasma_toroidal_on_axis,
            m_fuel_amu,
            p_plasma_loss_mw,
        )
    elif model == ConfinementTimeModel.REBUT_LALLIA:
        t_electron_confinement = rebut_lallia_confinement_time(
            rminor,
            rmajor,
            kappa,
            m_fuel_amu,
            cur_plasma_ma,
            zeff,
            nd_plasma_electron_line_20,
            b_plasma_toroidal_on_axis,
            p_plasma_loss_mw,
        )
    elif model == ConfinementTimeModel.GOLDSTON:
        t_electron_confinement = goldston_confinement_time(
            cur_plasma_ma, rmajor, rminor, kappa95, m_fuel_amu, p_plasma_loss_mw
        )
    elif model == ConfinementTimeModel.T_10:
        t_electron_confinement = t10_confinement_time(
            nd_plasma_electron_line_20,
            rmajor,
            qstar,
            b_plasma_toroidal_on_axis,
            rminor,
            kappa95,
            p_plasma_loss_mw,
            zeff,
            cur_plasma_ma,
        )
    elif model == ConfinementTimeModel.JAERI:
        t_electron_confinement = jaeri_confinement_time(
            kappa95,
            rminor,
            m_fuel_amu,
            n20,
            cur_plasma_ma,
            b_plasma_toroidal_on_axis,
            rmajor,
            qstar,
            zeff,
            p_plasma_loss_mw,
        )
    elif model == ConfinementTimeModel.KAYE_BIG:
        t_electron_confinement = kaye_big_confinement_time(
            rmajor,
            rminor,
            b_plasma_toroidal_on_axis,
            kappa95,
            cur_plasma_ma,
            n20,
            m_fuel_amu,
            p_plasma_loss_mw,
        )
    elif model == ConfinementTimeModel.ITER_H90_P:
        t_electron_confinement = iter_h90_p_confinement_time(
            cur_plasma_ma,
            rmajor,
            rminor,
            kappa,
            nd_plasma_electron_line_20,
            b_plasma_toroidal_on_axis,
            m_fuel_amu,
            p_plasma_loss_mw,
        )
    elif model == ConfinementTimeModel.MINIMUM_OF_ITER_89P_AND_ITER_89_0:
        t_electron_confinement = jnp.minimum(
            iter_89p_confinement_time(
                cur_plasma_ma,
                rmajor,
                rminor,
                kappa,
                nd_plasma_electron_line_20,
                b_plasma_toroidal_on_axis,
                m_fuel_amu,
                p_plasma_loss_mw,
            ),
            iter_89_0_confinement_time(
                cur_plasma_ma,
                rmajor,
                rminor,
                kappa,
                nd_plasma_electron_line_20,
                b_plasma_toroidal_on_axis,
                m_fuel_amu,
                p_plasma_loss_mw,
            ),
        )
    elif model == ConfinementTimeModel.RIEDEL_L:
        t_electron_confinement = riedel_l_confinement_time(
            cur_plasma_ma,
            rmajor,
            rminor,
            kappa95,
            nd_plasma_electron_line_20,
            b_plasma_toroidal_on_axis,
            p_plasma_loss_mw,
        )
    elif model == ConfinementTimeModel.CHRISTIANSEN:
        t_electron_confinement = christiansen_confinement_time(
            cur_plasma_ma,
            rmajor,
            rminor,
            kappa95,
            nd_plasma_electron_line_20,
            b_plasma_toroidal_on_axis,
            p_plasma_loss_mw,
            m_fuel_amu,
        )
    elif model == ConfinementTimeModel.LACKNER_GOTTARDI:
        t_electron_confinement = lackner_gottardi_confinement_time(
            cur_plasma_ma,
            rmajor,
            rminor,
            kappa95,
            nd_plasma_electron_line_20,
            b_plasma_toroidal_on_axis,
            p_plasma_loss_mw,
        )
    elif model == ConfinementTimeModel.NEO_KAYE:
        t_electron_confinement = neo_kaye_confinement_time(
            cur_plasma_ma,
            rmajor,
            rminor,
            kappa95,
            nd_plasma_electron_line_20,
            b_plasma_toroidal_on_axis,
            p_plasma_loss_mw,
        )
    elif model == ConfinementTimeModel.RIEDEL_H:
        t_electron_confinement = riedel_h_confinement_time(
            cur_plasma_ma,
            rmajor,
            rminor,
            kappa95,
            nd_plasma_electron_line_20,
            b_plasma_toroidal_on_axis,
            m_fuel_amu,
            p_plasma_loss_mw,
        )
    elif model == ConfinementTimeModel.ITER_H90_P_AMENDED:
        t_electron_confinement = iter_h90_p_amended_confinement_time(
            cur_plasma_ma,
            b_plasma_toroidal_on_axis,
            m_fuel_amu,
            rmajor,
            p_plasma_loss_mw,
            kappa,
        )
    elif model == ConfinementTimeModel.SUDO_ET_AL:
        t_electron_confinement = sudo_et_al_confinement_time(
            rmajor,
            rminor,
            nd_plasma_electron_line_20,
            b_plasma_toroidal_on_axis,
            p_plasma_loss_mw,
        )
    elif model == ConfinementTimeModel.GYRO_REDUCED_BOHM:
        t_electron_confinement = gyro_reduced_bohm_confinement_time(
            b_plasma_toroidal_on_axis,
            nd_plasma_electron_line_20,
            p_plasma_loss_mw,
            rminor,
            rmajor,
        )
    elif model == ConfinementTimeModel.LACKNER_GOTTARDI_STELLARATOR:
        t_electron_confinement = lackner_gottardi_stellarator_confinement_time(
            rmajor,
            rminor,
            nd_plasma_electron_line_20,
            b_plasma_toroidal_on_axis,
            p_plasma_loss_mw,
            q95,
        )
    elif model == ConfinementTimeModel.ITER_93H:
        t_electron_confinement = iter_93h_confinement_time(
            cur_plasma_ma,
            b_plasma_toroidal_on_axis,
            p_plasma_loss_mw,
            m_fuel_amu,
            rmajor,
            nd_plasma_electron_line_20,
            aspect,
            kappa,
        )
    elif model == ConfinementTimeModel.TITAN_REMOVED:
        raise ValueError("Scaling removed")
    elif model == ConfinementTimeModel.ITER_H97P:
        t_electron_confinement = iter_h97p_confinement_time(
            cur_plasma_ma,
            b_plasma_toroidal_on_axis,
            p_plasma_loss_mw,
            nd_plasma_electron_line_19,
            rmajor,
            aspect,
            kappa,
            m_fuel_amu,
        )
    elif model == ConfinementTimeModel.ITER_H97P_ELMY:
        t_electron_confinement = iter_h97p_elmy_confinement_time(
            cur_plasma_ma,
            b_plasma_toroidal_on_axis,
            p_plasma_loss_mw,
            nd_plasma_electron_line_19,
            rmajor,
            aspect,
            kappa,
            m_fuel_amu,
        )
    elif model == ConfinementTimeModel.ITER_96P:
        t_electron_confinement = iter_96p_confinement_time(
            cur_plasma_ma,
            b_plasma_toroidal_on_axis,
            kappa95,
            rmajor,
            aspect,
            nd_plasma_electron_line_19,
            m_fuel_amu,
            p_plasma_loss_mw,
        )
    elif model == ConfinementTimeModel.VALOVIC_ELMY:
        t_electron_confinement = valovic_elmy_confinement_time(
            cur_plasma_ma,
            b_plasma_toroidal_on_axis,
            nd_plasma_electron_line_19,
            m_fuel_amu,
            rmajor,
            rminor,
            kappa,
            p_plasma_loss_mw,
        )
    elif model == ConfinementTimeModel.KAYE:
        t_electron_confinement = kaye_confinement_time(
            cur_plasma_ma,
            b_plasma_toroidal_on_axis,
            kappa,
            rmajor,
            aspect,
            nd_plasma_electron_line_19,
            m_fuel_amu,
            p_plasma_loss_mw,
        )
    elif model == ConfinementTimeModel.ITER_PB98P_Y:
        t_electron_confinement = iter_pb98py_confinement_time(
            cur_plasma_ma,
            b_plasma_toroidal_on_axis,
            nd_plasma_electron_line_19,
            p_plasma_loss_mw,
            rmajor,
            kappa_ipb,
            aspect,
            m_fuel_amu,
        )
    elif model == ConfinementTimeModel.IPB98_Y:
        t_electron_confinement = iter_ipb98y_confinement_time(
            cur_plasma_ma,
            b_plasma_toroidal_on_axis,
            nd_plasma_electron_line_19,
            p_plasma_loss_mw,
            rmajor,
            kappa,
            aspect,
            m_fuel_amu,
        )
    elif model == ConfinementTimeModel.ITER_IPB98Y1:
        t_electron_confinement = iter_ipb98y1_confinement_time(
            cur_plasma_ma,
            b_plasma_toroidal_on_axis,
            nd_plasma_electron_line_19,
            p_plasma_loss_mw,
            rmajor,
            kappa_ipb,
            aspect,
            m_fuel_amu,
        )
    elif model == ConfinementTimeModel.ITER_IPB98Y2:
        t_electron_confinement = iter_ipb98y2_confinement_time(
            cur_plasma_ma,
            b_plasma_toroidal_on_axis,
            nd_plasma_electron_line_19,
            p_plasma_loss_mw,
            rmajor,
            kappa_ipb,
            aspect,
            m_fuel_amu,
        )
    elif model == ConfinementTimeModel.ITER_IPB98Y3:
        t_electron_confinement = iter_ipb98y3_confinement_time(
            cur_plasma_ma,
            b_plasma_toroidal_on_axis,
            nd_plasma_electron_line_19,
            p_plasma_loss_mw,
            rmajor,
            kappa_ipb,
            aspect,
            m_fuel_amu,
        )
    elif model == ConfinementTimeModel.ITER_IPB98Y4:
        t_electron_confinement = iter_ipb98y4_confinement_time(
            cur_plasma_ma,
            b_plasma_toroidal_on_axis,
            nd_plasma_electron_line_19,
            p_plasma_loss_mw,
            rmajor,
            kappa_ipb,
            aspect,
            m_fuel_amu,
        )
    elif model == ConfinementTimeModel.ISS95_STELLARATOR:
        t_electron_confinement = iss95_stellarator_confinement_time(
            rminor,
            rmajor,
            nd_plasma_electron_line_19,
            b_plasma_toroidal_on_axis,
            p_plasma_loss_mw,
            q95,
        )
    elif model == ConfinementTimeModel.ISS04_STELLARATOR:
        t_electron_confinement = iss04_stellarator_confinement_time(
            rminor,
            rmajor,
            nd_plasma_electron_line_19,
            b_plasma_toroidal_on_axis,
            p_plasma_loss_mw,
            q95,
        )
    elif model == ConfinementTimeModel.DS03:
        t_electron_confinement = ds03_confinement_time(
            cur_plasma_ma,
            b_plasma_toroidal_on_axis,
            nd_plasma_electron_line_19,
            p_plasma_loss_mw,
            rmajor,
            kappa95,
            aspect,
            m_fuel_amu,
        )
    elif model == ConfinementTimeModel.MURARI:
        t_electron_confinement = murari_confinement_time(
            cur_plasma_ma,
            rmajor,
            kappa_ipb,
            nd_plasma_electron_line_19,
            b_plasma_toroidal_on_axis,
            p_plasma_loss_mw,
        )
    elif model == ConfinementTimeModel.PETTY08:
        t_electron_confinement = petty08_confinement_time(
            cur_plasma_ma,
            b_plasma_toroidal_on_axis,
            nd_plasma_electron_line_19,
            p_plasma_loss_mw,
            rmajor,
            kappa_ipb,
            aspect,
        )
    elif model == ConfinementTimeModel.LANG_HIGH_DENSITY:
        t_electron_confinement = lang_high_density_confinement_time(
            plasma_current,
            b_plasma_toroidal_on_axis,
            nd_plasma_electron_line,
            p_plasma_loss_mw,
            rmajor,
            rminor,
            q95,
            qstar,
            aspect,
            m_fuel_amu,
            kappa_ipb,
        )
    elif model == ConfinementTimeModel.HUBBARD_NOMINAL:
        t_electron_confinement = hubbard_nominal_confinement_time(
            cur_plasma_ma,
            b_plasma_toroidal_on_axis,
            nd_plasma_electron_line_20,
            p_plasma_loss_mw,
        )
    elif model == ConfinementTimeModel.HUBBARD_LOWER:
        t_electron_confinement = hubbard_lower_confinement_time(
            cur_plasma_ma,
            b_plasma_toroidal_on_axis,
            nd_plasma_electron_line_20,
            p_plasma_loss_mw,
        )
    elif model == ConfinementTimeModel.HUBBARD_UPPER:
        t_electron_confinement = hubbard_upper_confinement_time(
            cur_plasma_ma,
            b_plasma_toroidal_on_axis,
            nd_plasma_electron_line_20,
            p_plasma_loss_mw,
        )
    elif model == ConfinementTimeModel.MENARD_NSTX:
        t_electron_confinement = menard_nstx_confinement_time(
            cur_plasma_ma,
            b_plasma_toroidal_on_axis,
            nd_plasma_electron_line_19,
            p_plasma_loss_mw,
            rmajor,
            kappa_ipb,
            aspect,
            m_fuel_amu,
        )
    elif model == ConfinementTimeModel.MENARD_NSTX_PETTY08_HYBRID:
        t_electron_confinement = menard_nstx_petty08_hybrid_confinement_time(
            cur_plasma_ma,
            b_plasma_toroidal_on_axis,
            nd_plasma_electron_line_19,
            p_plasma_loss_mw,
            rmajor,
            kappa_ipb,
            aspect,
            m_fuel_amu,
        )
    elif model == ConfinementTimeModel.NSTX_GYRO_BOHM:
        t_electron_confinement = nstx_gyro_bohm_confinement_time(
            cur_plasma_ma,
            b_plasma_toroidal_on_axis,
            p_plasma_loss_mw,
            rmajor,
            nd_plasma_electron_line_20,
        )
    elif model == ConfinementTimeModel.ITPA20:
        t_electron_confinement = itpa20_confinement_time(
            cur_plasma_ma,
            b_plasma_toroidal_on_axis,
            nd_plasma_electron_line_19,
            p_plasma_loss_mw,
            rmajor,
            triang,
            kappa_ipb,
            eps,
            m_ions_total_amu,
        )
    elif model == ConfinementTimeModel.ITPA20_IL:
        t_electron_confinement = itpa20_il_confinement_time(
            cur_plasma_ma,
            b_plasma_toroidal_on_axis,
            p_plasma_loss_mw,
            nd_plasma_electron_line_19,
            m_ions_total_amu,
            rmajor,
            triang,
            kappa_ipb,
        )
    elif model == ConfinementTimeModel.NCST:
        # `ConfinementTimeModel.PAZ_SOLDAN_NT` shares this same enum value (51) in the
        # source and is therefore unreachable here -- see `paz_soldan_nt_confinement_
        # time`'s docstring and the audit record's "A dead branch". Faithful to
        # PROCESS's actual behaviour: this arm is the only one ever taken for
        # `i_confinement_time == 51`.
        t_electron_confinement = ncst_confinement_time(
            cur_plasma_ma,
            b_plasma_toroidal_on_axis,
            p_plasma_loss_mw,
            nd_plasma_electron_line_19,
        )
    else:
        raise ValueError(f"Illegal value for i_confinement_time: {i_confinement_time}")

    t_electron_energy_confinement = hfact * t_electron_confinement
    t_ion_energy_confinement = t_electron_energy_confinement

    if rad_loss_model == ConfinementRadiationLossModel.CORE_ONLY:
        hstar = hfact * safe_pow(
            p_plasma_loss_mw
            / (
                p_plasma_loss_mw
                + pden_plasma_sync_mw * vol_plasma
                + p_plasma_inner_rad_mw
            ),
            0.31,
        )
    elif rad_loss_model == ConfinementRadiationLossModel.FULL_RADIATION:
        hstar = hfact * safe_pow(
            p_plasma_loss_mw / (p_plasma_loss_mw + pden_plasma_rad_mw * vol_plasma),
            0.31,
        )
    else:  # NO_RADIATION
        hstar = hfact

    pden_ion_transport_loss_mw = (
        eden_plasma_ions_thermal_vol_avg / t_ion_energy_confinement
    ) / 1e6
    pden_electron_transport_loss_mw = (
        eden_plasma_electrons_thermal_vol_avg / t_electron_energy_confinement
    ) / 1e6

    ratio = eden_plasma_ions_thermal_vol_avg / eden_plasma_electrons_thermal_vol_avg

    t_energy_confinement = (ratio + 1.0) / (
        ratio / t_ion_energy_confinement + 1.0 / t_electron_energy_confinement
    )

    t_energy_confinement_beta = (e_plasma_beta / 1e6) / p_plasma_loss_mw

    return (
        pden_electron_transport_loss_mw,
        pden_ion_transport_loss_mw,
        t_electron_energy_confinement,
        t_ion_energy_confinement,
        t_energy_confinement,
        p_plasma_loss_mw,
        hstar,
        kappa_ipb,
        t_energy_confinement_beta,
    )


# ---------------------------------------------------------------------------
# cottax nodes
# ---------------------------------------------------------------------------


class IterPhysicsBasisElongation(ExplicitFunction):
    """cottax node: `calculate_iter_physics_basis_elongation`, ports declared."""

    kappa_ipb = OutputInto(physics)

    def __call__(
        self,
        vol_plasma=From(physics),
        rmajor=From(physics),
        rminor=From(physics),
    ):
        return calculate_iter_physics_basis_elongation(vol_plasma, rmajor, rminor)


class ConfinementTime(ExplicitFunction):
    """cottax node: `calculate_confinement_time`, ports declared.

    `i_confinement_time`, `i_rad_loss` and `i_plasma_ignited` are switches resolved once
    at node-instantiation (graph-assembly) time -- `eqx.field(static=True)`, not
    reads, per `_audit/naming_convention.md` § "switches are not ports" (same move as
    `EcrhDensityLimit(i_plasma_pedestal=...)` in `models/stellarator/density_
    limits.py`). Which concrete value(s) belong in `total_process.py`'s default graph
    is a registration decision left to the consolidation pass, not decided here -- all
    three fields are required, with no default, so an un-instantiated choice fails
    loudly rather than silently picking one scaling law.

    **Consolidation-pass fix**: the audit record's own "switches touched" section
    already concludes `i_plasma_ignited` needs the same static treatment as the other
    two (its reads-set differs by one term, same shape as `i_rad_loss`/
    `i_confinement_time`) -- this class originally bound it as an ordinary read
    instead, and all three fields were missing `eqx.field(static=True)` entirely (plain
    `int`-annotated fields on an `eqx.Module` are still dynamic pytree leaves). Both are
    fixed here: `calculate_confinement_time`'s internal `if`/`elif` dispatch on these
    three values requires them to be concrete Python ints, not JAX tracers, under any
    `jax.jit`/`jacfwd` a driver might wrap this node in.

    `q95` is bound to `.physics.q95` here, the tokamak-mode producer -- `stellarator.py`
    instead feeds this same parameter from `.stellarator.iotabar` at its call site (see
    the audit record's data footprint table). Which producer is correct is a
    device-mode (`istell`) topology question, out of scope for this node declaration;
    a stellarator-mode instantiation of this class would rebind the `q95` input.
    """

    i_confinement_time: ConfinementTimeModel = eqx.field(static=True)
    i_rad_loss: ConfinementRadiationLossModel = eqx.field(static=True)
    i_plasma_ignited: PlasmaIgnitionModel = eqx.field(static=True)

    pden_electron_transport_loss_mw = OutputInto(physics)
    pden_ion_transport_loss_mw = OutputInto(physics)
    t_electron_energy_confinement = OutputInto(physics)
    t_ion_energy_confinement = OutputInto(physics)
    t_energy_confinement = OutputInto(physics)
    p_plasma_loss_mw = OutputInto(physics)
    hstar = OutputInto(physics)
    kappa_ipb = OutputInto(physics)
    t_energy_confinement_beta = OutputInto(physics)

    def __call__(
        self,
        m_fuel_amu=From(physics),
        p_alpha_total_mw=From(physics),
        aspect=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
        nd_plasma_electrons_vol_avg=From(physics),
        nd_plasma_electron_line=From(physics),
        eps=From(physics),
        hfact=From(physics),
        kappa=From(physics),
        kappa95=From(physics),
        p_non_alpha_charged_mw=From(physics),
        p_hcd_injected_total_mw=From(current_drive),
        plasma_current=From(physics),
        pden_plasma_core_rad_mw=From(physics),
        rmajor=From(physics),
        rminor=From(physics),
        temp_plasma_electron_density_weighted_kev=From(physics),
        q95=From(physics),
        qstar=From(physics),
        vol_plasma=From(physics),
        n_charge_plasma_effective_vol_avg=From(physics),
        eden_plasma_electrons_thermal_vol_avg=From(physics),
        eden_plasma_ions_thermal_vol_avg=From(physics),
        f_p_alpha_plasma_deposited=From(physics),
        p_plasma_ohmic_mw=From(physics),
        pden_plasma_rad_mw=From(physics),
        pden_plasma_sync_mw=From(physics),
        p_plasma_inner_rad_mw=From(physics),
        triang=From(physics),
        m_ions_total_amu=From(physics),
        e_plasma_beta=From(physics),
        tauee_in=From(physics),
    ):
        return calculate_confinement_time(
            m_fuel_amu,
            p_alpha_total_mw,
            aspect,
            b_plasma_toroidal_on_axis,
            nd_plasma_electrons_vol_avg,
            nd_plasma_electron_line,
            eps,
            hfact,
            self.i_confinement_time,
            self.i_plasma_ignited,
            kappa,
            kappa95,
            p_non_alpha_charged_mw,
            p_hcd_injected_total_mw,
            plasma_current,
            pden_plasma_core_rad_mw,
            rmajor,
            rminor,
            temp_plasma_electron_density_weighted_kev,
            q95,
            qstar,
            vol_plasma,
            n_charge_plasma_effective_vol_avg,
            eden_plasma_electrons_thermal_vol_avg,
            eden_plasma_ions_thermal_vol_avg,
            f_p_alpha_plasma_deposited,
            p_plasma_ohmic_mw,
            self.i_rad_loss,
            pden_plasma_rad_mw,
            pden_plasma_sync_mw,
            p_plasma_inner_rad_mw,
            triang,
            m_ions_total_amu,
            e_plasma_beta,
            tauee_in,
        )


class DoubleAndTripleProduct(ExplicitFunction):
    """cottax node: `calculate_double_and_triple_product`, ports declared."""

    ntau = OutputInto(physics)
    nTtau = OutputInto(physics)

    def __call__(
        self,
        nd_plasma_electrons_vol_avg=From(physics),
        temp_plasma_electron_vol_avg_kev=From(physics),
        t_energy_confinement=From(physics),
    ):
        return calculate_double_and_triple_product(
            nd_plasma_electrons_vol_avg,
            temp_plasma_electron_vol_avg_kev,
            t_energy_confinement,
        )


def _rebound_signature(call, **replacements):
    """`call`'s signature with the named parameters' `FromExactly` defaults replaced.

    The mechanism a device-mode subclass of a `NodalDeclaration` uses to rebind one
    read without restating the other 35. `FromExactly` is a *parameter default* on
    `__call__` (`cottax/interfaces/pytree_namespace_module.py:113-126`), and
    `ExplicitFunction._params` resolves the reads from `inspect.signature(
    self.__call__)` (`ibid.:213-215`) -- which honours a `__signature__` attribute --
    so replacing one default in a derived signature is all a rebinding takes.

    Deriving the signature rather than copying it is the point: a subclass that
    restated all 36 parameters would silently keep reading the old `VarPath` if the
    base ever gained, lost or rebound one. Here anything but the named replacement
    tracks the base automatically, and a `replacements` key that is *not* a parameter
    of `call` raises immediately -- so a renamed parameter fails loudly at import
    instead of quietly rebinding nothing.

    Safe to delegate positionally (`Base.__call__(self, *args, **kwargs)`) because
    `CallableNode`'s body is invoked positionally, in `inputs` order
    (`cottax/evaluate.py:76`: `node.fn(*_args(env, name, node))`), and `inputs` is
    built from these same parameters in order (`ibid.:183-185`).

    Not put in `~/jaxgraph` (yet): it is a *caller's* convenience for declaring one
    node in terms of another, not graph machinery, the same reasoning
    `_audit/next_steps.md` §8 records for keeping `PicardDriver` in this repo.

    Parameters
    ----------
    call :
        The base declaration's unbound `__call__`.
    **replacements :
        `parameter_name=FromExactly(...)` for each read to rebind.

    Returns
    -------
    :
        An `inspect.Signature` to assign to the overriding `__call__`'s
        `__signature__`.

    Raises
    ------
    ValueError
        If a `replacements` key is not a parameter of `call`.
    """
    signature = inspect.signature(call)
    unseen = dict(replacements)
    parameters = [
        parameter.replace(default=unseen.pop(name)) if name in unseen else parameter
        for name, parameter in signature.parameters.items()
    ]
    if unseen:
        raise ValueError(
            f"{call.__qualname__} has no parameter(s) {sorted(unseen)} to rebind; "
            f"its parameters are {sorted(signature.parameters)}"
        )
    return signature.replace(parameters=parameters)


class StellaratorConfinementTime(ConfinementTime):
    """`ConfinementTime` with its 20th read bound to `.stellarator.iotabar`.

    **Not a formula change -- the same `calculate_confinement_time` body, reading a
    different producer.** PROCESS's `calculate_confinement_time` names that parameter
    `q95` (`process/models/physics/confinement_time.py:79`, the 20th positional), and
    the *tokamak* caller does pass `.physics.q95`. The stellarator caller passes
    `self.data.stellarator.iotabar` into the very same slot
    (`process/models/stellarator/stellarator.py:2312`) -- the parameter is named for
    the tokamak quantity, but in stellarator mode it carries the rotational
    transform. `iss04_stellarator_confinement_time` even names its own parameter
    `iotabar` and raises it to `0.41` (this module, `iss04_stellarator_confinement_
    time`), so the base class's `.physics.q95` binding fed the ISS04 law a
    safety factor where PROCESS feeds it a rotational transform.

    **Found by the MDA-vs-PROCESS harness, and confirmed arithmetically, not
    inferred**: on `stellarator_helias.IN.DAT`'s converged run `q95 = 1.03` and
    `iotabar = 1.0`, so the base binding overstates `t_energy_confinement` by
    `1.03**0.41 = 1.0121928428817748` -- against the harness's reported `rel_diff` of
    `1.219e-02` on `.physics.t_energy_confinement`, and `1.205e-02` on
    `.physics.f_t_alpha_energy_confinement` downstream (it scales as `1/tau`).
    `ConfinementTime`'s own docstring had already predicted this exact fix ("a
    stellarator-mode instantiation of this class would rebind the `q95` input"); it
    was simply never built, and `total_process.py` registered the base class.

    Why a subclass rather than editing `ConfinementTime`: this module ports ~40
    scaling laws, most of them tokamak ones for which `.physics.q95` is right, so the
    binding is genuinely device-dependent and cannot be resolved in one place. Why a
    subclass rather than a static field on one class: reads are class-level
    parameter defaults, so an instance's `eqx.field(static=True)` cannot vary them --
    the declaring class *is* the unit of rebinding, and `NodalDeclaration.name` is
    `type(self).__name__` (`pytree_namespace_module.py:190-194`), so this arm gets its
    own node name in the graph for free.

    Registered under `total_process.TOPOLOGY_SWITCHES`'s `.stellarator.istell` switch,
    not unconditionally -- changing which node produces a read changes an edge, which
    is `configuration.py`'s own criterion for a topology switch. This class and
    `ConfinementTime` own byte-identical `Output` sets, so they are genuine mutually
    exclusive arms and pass `Switch.check_arms_are_exclusive`.
    """

    def __call__(self, *args, **kwargs):
        """`ConfinementTime.__call__`'s body, under the rebound signature below."""
        return ConfinementTime.__call__(self, *args, **kwargs)

    __call__.__signature__ = _rebound_signature(
        ConfinementTime.__call__,
        q95=FromExactly(stellarator.iotabar),
    )
