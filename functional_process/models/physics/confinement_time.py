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

import equinox as eqx
import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, Input, Output

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
    return 0.2 * rminor * jnp.sqrt(kappa95) * cur_plasma_ma


def merezhkin_muhkovatov_confinement_time(
    rmajor, rminor, kappa95, qstar, nd_plasma_electron_line_20, afuel, ten
):
    """Merezhkin-Mukhovatov (MM) OH/L-mode scaling.
    `ConfinementTimeModel.MEREZHKIN_MUHKOVATOV` (3).
    """
    return (
        3.5e-3
        * rmajor**2.75
        * rminor**0.25
        * kappa95**0.125
        * qstar
        * nd_plasma_electron_line_20
        * jnp.sqrt(afuel)
        / jnp.sqrt(ten / 10.0)
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
        * jnp.sqrt(kappa95)
        * jnp.sqrt(afuel)
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
        * kappa95**0.28
        * cur_plasma_ma**1.24
        * n20**0.26
        * rmajor**1.65
        * jnp.sqrt(afuel / 1.5)
        / (b_plasma_toroidal_on_axis**0.09 * rminor**0.49 * p_plasma_loss_mw**0.58)
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
        * cur_plasma_ma**0.85
        * rmajor**1.2
        * rminor**0.3
        * jnp.sqrt(kappa)
        * nd_plasma_electron_line_20**0.1
        * b_plasma_toroidal_on_axis**0.2
        * jnp.sqrt(afuel)
        / jnp.sqrt(p_plasma_loss_mw)
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
        0.04 * cur_plasma_ma**0.5 * rmajor**0.3 * rminor**0.8 * kappa**0.6 * afuel**0.5
    )
    term2 = (
        0.064
        * cur_plasma_ma**0.8
        * rmajor**1.6
        * rminor**0.6
        * kappa**0.5
        * nd_plasma_electron_line_20**0.6
        * b_plasma_toroidal_on_axis**0.35
        * afuel**0.2
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
    rll = (rminor**2 * rmajor * kappa) ** (1.0 / 3.0)
    term1 = 1.2e-2 * cur_plasma_ma * rll**1.5 / jnp.sqrt(zeff)
    term2 = (
        0.146
        * nd_plasma_electron_line_20**0.75
        * jnp.sqrt(cur_plasma_ma)
        * jnp.sqrt(b_plasma_toroidal_on_axis)
        * rll**2.75
        * zeff**0.25
        / p_plasma_loss_mw
    )
    return 1.65 * jnp.sqrt(afuel / 2.0) * (term1 + term2)


def goldston_confinement_time(
    cur_plasma_ma, rmajor, rminor, kappa95, afuel, p_plasma_loss_mw
):
    """Goldston scaling (L-mode). `ConfinementTimeModel.GOLDSTON` (9)."""
    return (
        0.037
        * cur_plasma_ma
        * rmajor**1.75
        * rminor ** (-0.37)
        * jnp.sqrt(kappa95)
        * jnp.sqrt(afuel / 1.5)
        / jnp.sqrt(p_plasma_loss_mw)
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
        * jnp.sqrt(kappa95)
        * denfac
        / p_plasma_loss_mw**0.4
        * (zeff**2 * cur_plasma_ma**4 / (rmajor * rminor * qstar**3 * kappa95**1.5))
        ** 0.08
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
        zeff**0.4
        * ((15.0 - zeff) / 20.0) ** 0.6
        * (3.0 * qstar * (qstar + 5.0) / ((qstar + 2.0) * (qstar + 7.0))) ** 0.6
    )
    return (
        0.085 * kappa95 * rminor**2 * jnp.sqrt(afuel)
        + 0.069
        * n20**0.6
        * cur_plasma_ma
        * b_plasma_toroidal_on_axis**0.2
        * rminor**0.4
        * rmajor**1.6
        * jnp.sqrt(afuel)
        * gjaeri
        * kappa95**0.2
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
        * jnp.sqrt(rmajor)
        * rminor**0.8
        * b_plasma_toroidal_on_axis**0.3
        * kappa95**0.25
        * cur_plasma_ma**0.85
        * n20**0.1
        * jnp.sqrt(afuel)
        / jnp.sqrt(p_plasma_loss_mw)
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
        * cur_plasma_ma**0.87
        * rmajor**1.82
        * rminor ** (-0.12)
        * kappa**0.35
        * nd_plasma_electron_line_20**0.09
        * b_plasma_toroidal_on_axis**0.15
        * jnp.sqrt(afuel)
        / jnp.sqrt(p_plasma_loss_mw)
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
        * cur_plasma_ma**0.93
        * rmajor**1.37
        * rminor ** (-0.049)
        * kappa95**0.588
        * nd_plasma_electron_line_20**0.078
        * b_plasma_toroidal_on_axis**0.152
        / p_plasma_loss_mw**0.537
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
        * cur_plasma_ma**0.79
        * rmajor**0.56
        * rminor**1.46
        * kappa95**0.73
        * nd_plasma_electron_line_20**0.41
        * b_plasma_toroidal_on_axis**0.29
        / (p_plasma_loss_mw**0.79 * afuel**0.02)
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
        * cur_plasma_ma**0.8
        * rmajor**1.8
        * rminor**0.4
        * kappa95
        * (1.0 + kappa95) ** (-0.8)
        * nd_plasma_electron_line_20**0.6
        * qhat**0.4
        / p_plasma_loss_mw**0.6
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
        * kappa95**0.28
        * nd_plasma_electron_line_20**0.14
        * b_plasma_toroidal_on_axis**0.04
        / p_plasma_loss_mw**0.59
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
        * jnp.sqrt(afuel)
        * cur_plasma_ma**0.884
        * rmajor**1.24
        * rminor ** (-0.23)
        * kappa95**0.317
        * b_plasma_toroidal_on_axis**0.207
        * nd_plasma_electron_line_20**0.105
        / p_plasma_loss_mw**0.486
    )


def iter_h90_p_amended_confinement_time(
    cur_plasma_ma, b_plasma_toroidal_on_axis, afuel, rmajor, p_plasma_loss_mw, kappa
):
    """Amended ITER H90-P law. `ConfinementTimeModel.ITER_H90_P_AMENDED` (20)."""
    return (
        0.082
        * cur_plasma_ma**1.02
        * b_plasma_toroidal_on_axis**0.15
        * jnp.sqrt(afuel)
        * rmajor**1.60
        / (p_plasma_loss_mw**0.47 * kappa**0.19)
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
        * rmajor**0.75
        * rminor**2
        * nd_plasma_electron_line_20**0.69
        * b_plasma_toroidal_on_axis**0.84
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
        * b_plasma_toroidal_on_axis**0.8
        * nd_plasma_electron_line_20**0.6
        * p_plasma_loss_mw ** (-0.6)
        * rminor**2.4
        * rmajor**0.6
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
        * nd_plasma_electron_line_20**0.6
        * b_plasma_toroidal_on_axis**0.8
        * p_plasma_loss_mw ** (-0.6)
        * q**0.4
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
        * b_plasma_toroidal_on_axis**0.32
        * p_plasma_loss_mw ** (-0.67)
        * afuel**0.41
        * rmajor**1.79
        * nd_plasma_electron_line_20**0.17
        * aspect**0.11
        * kappa**0.66
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
        * cur_plasma_ma**0.95
        * b_plasma_toroidal_on_axis**0.25
        * p_plasma_loss_mw ** (-0.67)
        * nd_plasma_electron_line_19**0.35
        * rmajor**1.92
        * aspect ** (-0.08)
        * kappa**0.63
        * afuel**0.42
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
        * cur_plasma_ma**0.90
        * b_plasma_toroidal_on_axis**0.20
        * p_plasma_loss_mw ** (-0.66)
        * nd_plasma_electron_line_19**0.40
        * rmajor**2.03
        * aspect ** (-0.19)
        * kappa**0.92
        * afuel**0.2
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
        * cur_plasma_ma**0.96
        * b_plasma_toroidal_on_axis**0.03
        * kappa95**0.64
        * rmajor**1.83
        * aspect**0.06
        * nd_plasma_electron_line_19**0.40
        * afuel**0.20
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
        * cur_plasma_ma**0.9
        * b_plasma_toroidal_on_axis**0.17
        * nd_plasma_electron_line_19**0.45
        * afuel**0.05
        * rmajor**1.316
        * rminor**0.79
        * kappa**0.56
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
        * cur_plasma_ma**0.81
        * b_plasma_toroidal_on_axis**0.14
        * kappa**0.7
        * rmajor**2.01
        * aspect ** (-0.18)
        * nd_plasma_electron_line_19**0.47
        * afuel**0.25
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
        * cur_plasma_ma**0.9
        * b_plasma_toroidal_on_axis**0.1
        * nd_plasma_electron_line_19**0.4
        * p_plasma_loss_mw ** (-0.66)
        * rmajor**2
        * kappa**0.75
        * aspect ** (-0.66)
        * afuel**0.2
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
        * cur_plasma_ma**0.97
        * b_plasma_toroidal_on_axis**0.08
        * nd_plasma_electron_line_19**0.41
        * p_plasma_loss_mw ** (-0.63)
        * rmajor**1.93
        * kappa**0.67
        * aspect ** (-0.23)
        * afuel**0.2
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
        * cur_plasma_ma**0.91
        * b_plasma_toroidal_on_axis**0.15
        * nd_plasma_electron_line_19**0.44
        * p_plasma_loss_mw ** (-0.65)
        * rmajor**2.05
        * kappa_ipb**0.72
        * aspect ** (-0.57)
        * afuel**0.13
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
        * cur_plasma_ma**0.93
        * b_plasma_toroidal_on_axis**0.15
        * nd_plasma_electron_line_19**0.41
        * p_plasma_loss_mw ** (-0.69)
        * rmajor**1.97
        * kappa_ipb**0.78
        * aspect ** (-0.58)
        * afuel**0.19
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
        * cur_plasma_ma**0.88
        * b_plasma_toroidal_on_axis**0.07
        * nd_plasma_electron_line_19**0.40
        * p_plasma_loss_mw ** (-0.69)
        * rmajor**2.15
        * kappa_ipb**0.78
        * aspect ** (-0.64)
        * afuel**0.20
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
        * cur_plasma_ma**0.85
        * b_plasma_toroidal_on_axis**0.29
        * nd_plasma_electron_line_19**0.39
        * p_plasma_loss_mw ** (-0.70)
        * rmajor**2.08
        * kappa_ipb**0.76
        * aspect ** (-0.69)
        * afuel**0.17
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
        * rmajor**0.65
        * nd_plasma_electron_line_19**0.51
        * b_plasma_toroidal_on_axis**0.83
        * p_plasma_loss_mw ** (-0.59)
        * iotabar**0.4
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
        * rmajor**0.64
        * nd_plasma_electron_line_19**0.54
        * b_plasma_toroidal_on_axis**0.84
        * p_plasma_loss_mw ** (-0.61)
        * iotabar**0.41
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
        * cur_plasma_ma**0.83
        * b_plasma_toroidal_on_axis**0.07
        * nd_plasma_electron_line_19**0.49
        * p_plasma_loss_mw ** (-0.55)
        * rmajor**2.11
        * kappa95**0.75
        * aspect ** (-0.3)
        * afuel**0.14
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
            nd_plasma_electron_line_19**0.448
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
        * cur_plasma_ma**0.75
        * b_plasma_toroidal_on_axis**0.3
        * nd_plasma_electron_line_19**0.32
        * p_plasma_loss_mw ** (-0.47)
        * rmajor**2.09
        * kappa_ipb**0.88
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
        * b_plasma_toroidal_on_axis**0.12
        * nd_plasma_electron_line**0.032236
        * (p_plasma_loss_mw * 1.0e6) ** (-0.74)
        * rmajor**1.2345
        * kappa_ipb**0.37
        * aspect**2.48205
        * afuel**0.2
        * qratio**0.77
        * aspect ** (-0.9 * jnp.log(aspect))
        * nratio ** (-0.22 * jnp.log(nratio))
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
        * cur_plasma_ma**0.68
        * b_plasma_toroidal_on_axis**0.77
        * nd_plasma_electron_line_20**0.02
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
        * cur_plasma_ma**0.60
        * b_plasma_toroidal_on_axis**0.70
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
        * cur_plasma_ma**0.76
        * b_plasma_toroidal_on_axis**0.84
        * nd_plasma_electron_line_20**0.07
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
        * cur_plasma_ma**0.57
        * b_plasma_toroidal_on_axis**1.08
        * nd_plasma_electron_line_19**0.44
        * p_plasma_loss_mw ** (-0.73)
        * rmajor**1.97
        * kappa_ipb**0.78
        * aspect ** (-0.58)
        * afuel**0.19
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
        * cur_plasma_ma**0.54
        * b_plasma_toroidal_on_axis**0.91
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
        * cur_plasma_ma**0.976
        * b_plasma_toroidal_on_axis**0.218
        * nd_plasma_electron_line_19**0.2442
        * p_plasma_loss_mw ** (-0.6687)
        * rmajor**1.710
        * (1 + triang) ** 0.362
        * kappa_ipb**0.799
        * eps**0.354
        * aion**0.195
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
        * nd_plasma_electron_line_19**0.1473
        * p_plasma_loss_mw ** (-0.6442)
        * rmajor**1.194
        * (1 + triang) ** 0.560
        * kappa_ipb**0.673
        * aion**0.302
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
        * cur_plasma_ma**0.33
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
        * b_plasma_toroidal_on_axis**0.11
        * nd_plasma_electron_line_19**0.51
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
        hstar = (
            hfact
            * (
                p_plasma_loss_mw
                / (
                    p_plasma_loss_mw
                    + pden_plasma_sync_mw * vol_plasma
                    + p_plasma_inner_rad_mw
                )
            )
            ** 0.31
        )
    elif rad_loss_model == ConfinementRadiationLossModel.FULL_RADIATION:
        hstar = (
            hfact
            * (p_plasma_loss_mw / (p_plasma_loss_mw + pden_plasma_rad_mw * vol_plasma))
            ** 0.31
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

    kappa_ipb = Output(lambda s: s.physics.kappa_ipb)

    def __call__(
        self,
        vol_plasma=Input(lambda s: s.physics.vol_plasma),
        rmajor=Input(lambda s: s.physics.rmajor),
        rminor=Input(lambda s: s.physics.rminor),
    ):
        return calculate_iter_physics_basis_elongation(vol_plasma, rmajor, rminor)


class ConfinementTime(ExplicitFunction):
    """cottax node: `calculate_confinement_time`, ports declared.

    `i_confinement_time`, `i_rad_loss` and `i_plasma_ignited` are switches resolved once
    at node-instantiation (graph-assembly) time -- `eqx.field(static=True)`, not
    `Input`s, per `_audit/naming_convention.md` § "switches are not ports" (same move as
    `EcrhDensityLimit(i_plasma_pedestal=...)` in `models/stellarator/density_
    limits.py`). Which concrete value(s) belong in `total_process.py`'s default graph
    is a registration decision left to the consolidation pass, not decided here -- all
    three fields are required, with no default, so an un-instantiated choice fails
    loudly rather than silently picking one scaling law.

    **Consolidation-pass fix**: the audit record's own "switches touched" section
    already concludes `i_plasma_ignited` needs the same static treatment as the other
    two (its reads-set differs by one term, same shape as `i_rad_loss`/
    `i_confinement_time`) -- this class originally bound it as an ordinary `Input`
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

    i_confinement_time: int = eqx.field(static=True)
    i_rad_loss: int = eqx.field(static=True)
    i_plasma_ignited: int = eqx.field(static=True)

    pden_electron_transport_loss_mw = Output(
        lambda s: s.physics.pden_electron_transport_loss_mw
    )
    pden_ion_transport_loss_mw = Output(lambda s: s.physics.pden_ion_transport_loss_mw)
    t_electron_energy_confinement = Output(
        lambda s: s.physics.t_electron_energy_confinement
    )
    t_ion_energy_confinement = Output(lambda s: s.physics.t_ion_energy_confinement)
    t_energy_confinement = Output(lambda s: s.physics.t_energy_confinement)
    p_plasma_loss_mw = Output(lambda s: s.physics.p_plasma_loss_mw)
    hstar = Output(lambda s: s.physics.hstar)
    kappa_ipb = Output(lambda s: s.physics.kappa_ipb)
    t_energy_confinement_beta = Output(lambda s: s.physics.t_energy_confinement_beta)

    def __call__(
        self,
        m_fuel_amu=Input(lambda s: s.physics.m_fuel_amu),
        p_alpha_total_mw=Input(lambda s: s.physics.p_alpha_total_mw),
        aspect=Input(lambda s: s.physics.aspect),
        b_plasma_toroidal_on_axis=Input(lambda s: s.physics.b_plasma_toroidal_on_axis),
        nd_plasma_electrons_vol_avg=Input(
            lambda s: s.physics.nd_plasma_electrons_vol_avg
        ),
        nd_plasma_electron_line=Input(lambda s: s.physics.nd_plasma_electron_line),
        eps=Input(lambda s: s.physics.eps),
        hfact=Input(lambda s: s.physics.hfact),
        kappa=Input(lambda s: s.physics.kappa),
        kappa95=Input(lambda s: s.physics.kappa95),
        p_non_alpha_charged_mw=Input(lambda s: s.physics.p_non_alpha_charged_mw),
        p_hcd_injected_total_mw=Input(lambda s: s.current_drive.p_hcd_injected_total_mw),
        plasma_current=Input(lambda s: s.physics.plasma_current),
        pden_plasma_core_rad_mw=Input(lambda s: s.physics.pden_plasma_core_rad_mw),
        rmajor=Input(lambda s: s.physics.rmajor),
        rminor=Input(lambda s: s.physics.rminor),
        temp_plasma_electron_density_weighted_kev=Input(
            lambda s: s.physics.temp_plasma_electron_density_weighted_kev
        ),
        q95=Input(lambda s: s.physics.q95),
        qstar=Input(lambda s: s.physics.qstar),
        vol_plasma=Input(lambda s: s.physics.vol_plasma),
        zeff=Input(lambda s: s.physics.n_charge_plasma_effective_vol_avg),
        eden_plasma_electrons_thermal_vol_avg=Input(
            lambda s: s.physics.eden_plasma_electrons_thermal_vol_avg
        ),
        eden_plasma_ions_thermal_vol_avg=Input(
            lambda s: s.physics.eden_plasma_ions_thermal_vol_avg
        ),
        f_p_alpha_plasma_deposited=Input(lambda s: s.physics.f_p_alpha_plasma_deposited),
        p_plasma_ohmic_mw=Input(lambda s: s.physics.p_plasma_ohmic_mw),
        pden_plasma_rad_mw=Input(lambda s: s.physics.pden_plasma_rad_mw),
        pden_plasma_sync_mw=Input(lambda s: s.physics.pden_plasma_sync_mw),
        p_plasma_inner_rad_mw=Input(lambda s: s.physics.p_plasma_inner_rad_mw),
        triang=Input(lambda s: s.physics.triang),
        m_ions_total_amu=Input(lambda s: s.physics.m_ions_total_amu),
        e_plasma_beta=Input(lambda s: s.physics.e_plasma_beta),
        tauee_in=Input(lambda s: s.physics.tauee_in),
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
            zeff,
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

    ntau = Output(lambda s: s.physics.ntau)
    nTtau = Output(lambda s: s.physics.nTtau)

    def __call__(
        self,
        nd_plasma_electrons_vol_avg=Input(
            lambda s: s.physics.nd_plasma_electrons_vol_avg
        ),
        temp_plasma_electrons_vol_avg_kev=Input(
            lambda s: s.physics.temp_plasma_electron_vol_avg_kev
        ),
        t_energy_confinement=Input(lambda s: s.physics.t_energy_confinement),
    ):
        return calculate_double_and_triple_product(
            nd_plasma_electrons_vol_avg,
            temp_plasma_electrons_vol_avg_kev,
            t_energy_confinement,
        )
