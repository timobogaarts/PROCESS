"""Pure physics functions extracted from `models/physics/plasma_current.py`.

See that file for the declarations wiring these into the graph, and
`_audit/units/models/physics/plasma_current.md` for the audit record. No
graph-framework import belongs in this module -- see `_audit/formulas_split.md`.
"""

import jax.numpy as jnp

from functional_process.models.safe_math import safe_pow
from functional_process.vocabulary import constants

RMU0 = constants.RMU0
"""Vacuum permeability (H/m), `process/core/constants.py:277`. Imported rather than
inlined, the dominant convention in this port (`models/tfcoil/base.py:79`), so the two
sides of every harness diff cannot drift apart in the last digit."""


def calculate_cyclindrical_plasma_current(
    rminor, rmajor, q95, b_plasma_toroidal_on_axis
):
    """Plasma current of the equivalent circular cylindrical plasma (A).

    Ports `PlasmaCurrent.calculate_cyclindrical_plasma_current`,
    `process/models/physics/plasma_current.py:595-623`, unchanged (spelling of the name
    included -- it is misspelled in PROCESS and kept so, per
    `_audit/naming_convention.md`).

    This is the shared factor of every `i_plasma_current` arm **except** `2`, which
    builds its current directly (`plasma_current.py:322-330`) and never multiplies this.
    """
    return (2.0 * jnp.pi / RMU0) * rminor**2 / (rmajor * q95) * b_plasma_toroidal_on_axis


def calculate_current_coefficient_ipdg89(eps, kappa95, triang95):
    """The `fq` shaping coefficient of the IPDG89 plasma current scaling.

    Ports `PlasmaCurrent.calculate_current_coefficient_ipdg89`,
    `process/models/physics/plasma_current.py:782-815`, unchanged.

    References
    ----------
    [1] N.A. Uckan and ITER Physics Group, 'ITER Physics Design Guidelines: 1989'
    """
    return (
        0.5
        * (1.17 - 0.65 * eps)
        / ((1.0 - eps * eps) ** 2)
        * (1.0 + kappa95**2 * (1.0 + 2.0 * triang95**2 - 1.2 * triang95**3))
    )


def calculate_plasma_current_ipdg89(
    eps, kappa95, triang95, rminor, rmajor, q95, b_plasma_toroidal_on_axis
):
    """Plasma current (A) under `i_plasma_current == IPDG89_SCALING` (4).

    The `i_plasma_current == 4` path through `PlasmaCurrent.calculate_plasma_current`
    (`process/models/physics/plasma_current.py:337-340` for the coefficient,
    `:392-401` for the product), with the eight other arms and the enum dispatch
    removed -- they are other occupants' bodies, not branches of this one.

    **The two guards in the PROCESS body are not carried, and neither can fire here.**
    `plasma_current.py:305-309` raises when `triang < 0` and `i_plasma_current != 8`;
    `:385-389` raises on an out-of-range `i_plasma_current`. Both are switch-domain
    checks, answered by *which occupant exists* rather than at call time --
    `_audit/naming_convention.md` § "switches are not ports". The negative-triangularity
    guard is a genuine precondition of this arm (the caller's to hold, as in PROCESS)
    and this unit's test module keeps every sample and fuzz bound at `triang95 >= 0`.
    """
    return calculate_cyclindrical_plasma_current(
        rminor=rminor,
        rmajor=rmajor,
        q95=q95,
        b_plasma_toroidal_on_axis=b_plasma_toroidal_on_axis,
    ) * calculate_current_coefficient_ipdg89(eps=eps, kappa95=kappa95, triang95=triang95)


def calculate_current_coefficient_fiesta(eps, kappa, triang):
    """The `fq` shaping coefficient of the FIESTA ST plasma current scaling.

    Ports `PlasmaCurrent.calculate_current_coefficient_fiesta`,
    `process/models/physics/plasma_current.py:990-1016`, value-identical. Selected by
    `i_plasma_current == 9` at `plasma_current.py:380-383`.

    **Reads the separatrix shaping pair, not the 95% one.** `kappa`/`triang`, where the
    IPDG89 arm above reads `kappa95`/`triang95` — the two arms have genuinely different
    read sets, which is why they are two occupants and not one node with a switch.

    **`triang ** 0.060` is the one deviation, and it is `safe_pow`, not a repair.**
    PROCESS's expression is `triang**0.060`; at `triang == 0` its value is `0.0` and its
    derivative is `0.06 * 0.0**-0.94 == +inf`, the exact `x ** p, 0 < p < 1` shape
    `models/safe_math.py` exists for (`_audit/next_steps.md` §9). `safe_pow` is
    bit-identical for every `triang != 0` — including the `nan` PROCESS returns for a
    negative base — and differs only in giving the derivative *at* zero the value `0`
    instead of `inf`. The value defect is therefore reproduced, not repaired; only the
    Jacobian poisoning is removed, on the same terms as every other `safe_pow` site in
    this port.

    **The negative-triangularity `nan` is real and is left alone.** `triang < 0` makes
    `triang ** 0.060` `nan` in value, and PROCESS never reaches it: the caller raises
    first (`plasma_current.py:305-309` — `triang < 0` is refused for every
    `i_plasma_current` except `8`). Same treatment as the IPDG89 arm's identical guard:
    a precondition of this arm, the caller's to hold as in PROCESS, and this unit's test
    module keeps every sample and fuzz bound at `triang > 0`. See the audit record's
    **D7**.

    References
    ----------
    [1] S. Muldrew et al., '"PROCESS": Systems studies of spherical tokamaks',
    Fusion Engineering and Design 154 (2020) 111530.
    https://doi.org/10.1016/j.fusengdes.2020.111530
    """
    return 0.538 * (1.0 + 2.440 * eps**2.736) * kappa**2.154 * safe_pow(triang, 0.060)


def calculate_plasma_current_fiesta(
    eps, kappa, triang, rminor, rmajor, q95, b_plasma_toroidal_on_axis
):
    """Plasma current (A) under `i_plasma_current == FIESTA_ST_SCALING` (9).

    The `i_plasma_current == 9` path through `PlasmaCurrent.calculate_plasma_current`
    (`process/models/physics/plasma_current.py:380-383` for the coefficient,
    `:392-401` for the product), with the eight other arms and the enum dispatch removed.
    Structurally the IPDG89 arm's shape — `fq * calculate_cyclindrical_plasma_current`
    — over a different coefficient and a different shaping pair.

    The two guards in the PROCESS body are not carried, for the reason
    `calculate_plasma_current_ipdg89` gives: both are switch-domain checks answered by
    *which occupant exists*. The negative-triangularity one is a genuine precondition
    here, and this arm is the one PROCESS's own guard singles out — `i_plasma_current`
    `8` (Sauter) is the only value it exempts, so FIESTA is refused for `triang < 0` by
    PROCESS just as every other non-Sauter arm is.
    """
    return calculate_cyclindrical_plasma_current(
        rminor=rminor,
        rmajor=rmajor,
        q95=q95,
        b_plasma_toroidal_on_axis=b_plasma_toroidal_on_axis,
    ) * calculate_current_coefficient_fiesta(eps=eps, kappa=kappa, triang=triang)


def calculate_cylindrical_safety_factor(
    rmajor, rminor, plasma_current, b_plasma_toroidal_on_axis, kappa95, triang95
):
    """Cylindrical safety factor `qstar` (dimensionless), IPDG89 guidelines.

    Ports the module-level `calculate_cylindrical_safety_factor`,
    **`process/models/physics/physics.py:54-99`** (`@nb.jit`ted there; the `numba`
    decorator is the only thing dropped), unchanged. Called once, unconditionally, at
    `physics.py:303-310`.

    Not gated by `i_plasma_current`: PROCESS computes `qstar` from whichever current the
    selected scaling produced, so this is one node for every arm.

    **PROCESS's division *by a quotient* is kept verbatim, and it is a registered
    boundary defect.** `rminor**2 / (rmajor * plasma_current /
    b_plasma_toroidal_on_axis)` is non-differentiable at
    `b_plasma_toroidal_on_axis == 0`: the inner quotient goes to
    `+inf`, the outer division pulls the value back to a finite `0.0`, and the tangent
    stays `nan`. That is the unguarded-division class
    `_harness/boundary.py`'s register exists for, and this site is listed there as
    `("TestCalculateCylindricalSafetyFactor", "b_plasma_toroidal_on_axis")`.

    Reassociating to `rminor**2 * b / (rmajor * plasma_current)` would remove the `nan`
    and give the same `0.0` at the boundary, but it is not bit-identical (measured over
    20 000 fuzz-domain points: 32% differ, worst relative difference `5.8e-16`, ~2.6 ulp)
    -- and **faithfulness won**: a ported body spells PROCESS's expression, and a
    derivative defect PROCESS itself carries is recorded rather than quietly repaired.
    See the audit record's "## suspected defects in PROCESS" **D6**.
    """
    return (
        ((2 * jnp.pi) / RMU0)
        * rminor**2
        / (rmajor * plasma_current / b_plasma_toroidal_on_axis)
        * 0.5
        * (1.0 + kappa95**2 * (1.0 + 2.0 * triang95**2 - 1.2 * triang95**3))
    )


def calculate_current_profile_index_wesson(qstar, q0):
    """Wesson current-profile index `alphaj`.

    Ports `Physics.calculate_current_profile_index_wesson`,
    **`process/models/physics/physics.py:1136-1164`**, unchanged. PROCESS stores the
    result in `.physics.alphaj_wesson` (`physics.py:330`) and then copies it into
    `.physics.alphaj` when `i_alphaj == 1` (`physics.py:343`); this port writes
    `.physics.alphaj` directly, since `alphaj_wesson` has no non-reporting reader.

    References
    ----------
    [1] Wesson, J. (2011) Tokamaks. 4th Edition, Oxford Science Publications, Vol. 149.
    """
    return qstar / q0 - 1.0


def calculate_internal_inductance_wesson(alphaj):
    """Normalised plasma internal inductance `li` from the Wesson scaling.

    Ports `PlasmaInductance.calculate_internal_inductance_wesson`,
    **`process/models/physics/physics.py:4977-5005`**, unchanged (`np.log` ->
    `jnp.log`). Selected by `i_ind_plasma_internal_norm == 1` at `physics.py:4743-4745`
    via the `model_map` lookup at `:4759-4764`.
    """
    return jnp.log(1.65 + 0.89 * alphaj)
