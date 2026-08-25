"""Pure-functional port of `process/models/superconductors.py` (registry unit #22).

Audit record: `functional_process/_audit/units/models/physics/superconductors.md`. Read
it first, especially "tier signal" for why all 7 in-scope functions land tier-1 (not the
tier-2 the registry anticipated -- the file's one `scipy.optimize` call, in
`current_sharing_rebco`, is out of scope and calls `jcrit_rebco`, not the other way
round) and "JAX-difficulty flags" for the domain guards each `jnp.where` needed to keep
its untaken branch finite under `jax.jacfwd`.

`bottura_scaling` is ported alongside the 7 named units as a shared, transitively-called
helper (`itersc` and `western_superconducting_nb3sn` both call it) -- same precedent as
`fusion_reactions.py`'s `bosch_hale_reactivity`.

No `cottax` node is written for any of the 8 functions here -- see the audit record's
"cottax node" section: every real call site's arguments are locals inside
`jcrit_from_material` (`process/models/stellarator/coils/coils.py`, unit #10, unported),
not established `.area.field` `VarPath`s this audit can verify yet. Same situation, same
reasoning, as `coils.py`'s own three unwrapped functions (`j_crit_cable_from_fraction`,
`bmax_from_awp`, `intersect`).
"""

import jax.numpy as jnp

from functional_process.models.safe_math import safe_pow, safe_sqrt


def jcrit_rebco(temp_conductor, b_conductor):
    """Critical current density for a "REBCO" 2nd-generation HTS superconductor.

    Direct port of the module-level function of the same name. Validity range:
    `4.2 K <= temp_conductor <= 72.0 K`; field range depends on temperature
    (`<= 15.0 T` below 65 K, `<= 11.5 T` at or above). Source: `logger.error(...)`
    when outside this range -- a diagnostic side effect with no return-value
    consequence, dropped here (see the audit record's JAX-difficulty flags).
    PROCESS does not raise on this condition (unlike `bi2212` below), it only logs,
    so the port needs no `reference_domain_errors` -- the formula itself is
    evaluated and returned regardless of `validity`.

    Both data-dependent branches (`temp_conductor < temp_c0max` for `birr`,
    `b_conductor < birr` for `j_critical`) have their untaken arm's fractional-power base
    guarded against going negative -- verified by `jax.jacfwd` at both branch boundaries
    while writing this port (see the audit record's "tier signal").

    Parameters
    ----------
    temp_conductor :
        Superconductor temperature (K).
    b_conductor :
        Magnetic field at the superconductor (T).

    Returns
    -------
    tuple
        `(j_critical, validity, b_c20max, temp_c0max)` -- critical current density
        (A/m^2), whether the inputs are within the fit's validity range, upper critical
        field (T) at zero temperature/strain, critical temperature (K) at zero
        field/strain.
    """
    temp_c0max = 90.0
    b_c20max = 132.5

    c = 1.82962e8
    p = 0.5875
    q = 1.7
    alpha = 1.54121
    beta = 1.96679
    one_over_alpha = 1.0 / alpha

    in_temperature_range = (temp_conductor >= 4.2) & (temp_conductor <= 72.0)
    low_temperature = temp_conductor < 65.0
    in_field_range = jnp.where(
        low_temperature,
        (b_conductor >= 0.0) & (b_conductor <= 15.0),
        (b_conductor >= 0.0) & (b_conductor <= 11.5),
    )
    validity = in_temperature_range & in_field_range

    below_curie = temp_conductor < temp_c0max
    safe_ratio = jnp.where(below_curie, 1.0 - temp_conductor / temp_c0max, 1.0)
    birr = jnp.where(
        below_curie,
        b_c20max * safe_ratio**alpha,
        b_c20max * (1.0 - temp_conductor / temp_c0max),
    )

    below_birr = b_conductor < birr
    safe_b_ratio = jnp.where(below_birr, b_conductor / birr, 0.5)
    factor = safe_b_ratio**p * (1.0 - safe_b_ratio) ** q
    # tcb: critical temperature at field b, used only on the `b_conductor >= birr` arm.
    tcb = temp_c0max * (1.0 - (b_conductor / b_c20max) ** one_over_alpha)
    j_critical = jnp.where(
        below_birr,
        (c / b_conductor) * (birr**beta) * factor,
        -(temp_conductor - tcb),
    )

    return j_critical, validity, b_c20max, temp_c0max


def bottura_scaling(
    csc,
    p,
    q,
    c_a1,
    c_a2,
    epsilon_0a,
    temp_conductor,
    b_conductor,
    epsilon,
    b_c20max,
    temp_c0max,
):
    """`Jc(B, T, epsilon)` scaling shared by `itersc`/`western_superconducting_nb3sn`.

    Direct port of the module-level function of the same name -- the ITER-2008 Nb3Sn
    strain scaling, parameterized by `(csc, p, q, c_a1, c_a2, epsilon_0a)` so the two
    callers below share this one implementation with different constants.

    Two data-dependent branches, both `jnp.where`-guarded against a negative
    fractional-power base on the untaken arm (see the audit record's JAX-difficulty
    flags): the `temp_critical` normal/abnormal split (`f_b_conductor_critical_no_temp <
    1.0`), and the "inside/outside the critical surface" split that selects `j_scaling`'s
    formula. Source: two `logger.error(...)` diagnostics ("artificially lowered") with no
    return-value consequence, dropped.

    Parameters
    ----------
    csc, p, q, c_a1, c_a2, epsilon_0a :
        Scaling constants (caller-supplied, e.g. `itersc`'s ITER values or
        `western_superconducting_nb3sn`'s WST values).
    temp_conductor :
        Superconductor temperature (K).
    b_conductor :
        Magnetic field at the conductor (T).
    epsilon :
        Strain in the superconductor.
    b_c20max :
        Upper critical field (T) at zero temperature and strain.
    temp_c0max :
        Critical temperature (K) at zero field and strain.

    Returns
    -------
    tuple
        `(j_scaling, b_critical, temp_critical)`.
    """
    epsilon_sh = (c_a2 * epsilon_0a) / safe_sqrt(c_a1**2 - c_a2**2)

    strain_func = safe_sqrt(epsilon_sh**2 + epsilon_0a**2) - safe_sqrt(
        (epsilon - epsilon_sh) ** 2 + epsilon_0a**2
    )
    strain_func = strain_func * c_a1 - (c_a2 * epsilon)
    strain_func = 1.0 + (1.0 / (1.0 - c_a1 * epsilon_0a)) * strain_func

    b_c20_eps = b_c20max * strain_func
    temp_c0_eps = temp_c0max * safe_pow(strain_func, 1.0 / 3.0)

    f_temp_conductor_critical_no_field = temp_conductor / temp_c0_eps
    f_b_conductor_critical_no_temp = b_conductor / b_c20_eps

    normal_field = f_b_conductor_critical_no_temp < 1.0
    safe_complement = jnp.where(normal_field, 1.0 - f_b_conductor_critical_no_temp, 1.0)
    temp_critical = jnp.where(
        normal_field,
        temp_c0_eps * safe_pow(safe_complement, 1.0 / 1.52),
        -temp_c0_eps
        * safe_pow(jnp.abs(1.0 - f_b_conductor_critical_no_temp), 1.0 / 1.52),
    )

    b_critical = b_c20_eps * (1.0 - f_temp_conductor_critical_no_field**1.52)

    jc1 = (csc / b_conductor) * strain_func

    inside_critical_surface = (
        (f_temp_conductor_critical_no_field > 0.0)
        & (f_temp_conductor_critical_no_field < 1.0)
        & (b_conductor > 0.0)
        & (b_conductor < b_critical)
        & (b_critical > 0.0)
    )
    # Guard the "inside" arm's divisor: b_critical may be <= 0 on the untaken branch.
    safe_b_critical = jnp.where(inside_critical_surface, b_critical, 1.0)
    b_reduced = b_conductor / safe_b_critical

    jc2_inside = (1.0 - f_temp_conductor_critical_no_field**1.52) * (
        1.0 - f_temp_conductor_critical_no_field**2
    )
    jc3_inside = b_reduced**p * (1.0 - b_reduced) ** q
    j_scaling_inside = jc1 * jc2_inside * jc3_inside

    jc2_outside = f_temp_conductor_critical_no_field
    jc3_outside = b_conductor / jnp.maximum(b_critical, 1.0e-8)
    j_scaling_outside = -jnp.abs(jc1 * jc2_outside * jc3_outside)

    j_scaling = jnp.where(inside_critical_surface, j_scaling_inside, j_scaling_outside)

    return j_scaling, b_critical, temp_critical


def itersc(temp_conductor, b_conductor, strain, b_c20max, temp_c0max):
    """Critical current density/field/temperature for an ITER Nb3Sn superconductor.

    Direct port of the module-level function of the same name: `bottura_scaling` with
    ITER's fitting constants, then a per-strand-to-per-area unit conversion.

    Parameters
    ----------
    temp_conductor :
        Superconductor temperature (K).
    b_conductor :
        Magnetic field at the conductor (T).
    strain :
        Strain in the superconductor.
    b_c20max :
        Upper critical field (T) at zero temperature and strain.
    temp_c0max :
        Critical temperature (K) at zero field and strain.

    Returns
    -------
    tuple
        `(j_critical, b_critical, temp_critical)`.
    """
    csc = 19922.0
    p = 0.63
    q = 2.1
    ca1 = 44.48
    ca2 = 0.0
    epsilon_0a = 0.00256

    diter = 0.82
    f_a_strand_copper = 0.5

    j_scaling, b_critical, temp_critical = bottura_scaling(
        csc=csc,
        p=p,
        q=q,
        c_a1=ca1,
        c_a2=ca2,
        epsilon_0a=epsilon_0a,
        temp_conductor=temp_conductor,
        b_conductor=b_conductor,
        epsilon=strain,
        b_c20max=b_c20max,
        temp_c0max=temp_c0max,
    )

    scalefac = jnp.pi * (0.5 * diter) ** 2 * (1.0 - f_a_strand_copper) / 1.0e6
    j_critical = j_scaling / scalefac

    return j_critical, b_critical, temp_critical


def jcrit_nbti(temp_conductor, b_conductor, c0, b_c20max, temp_c0max):
    """Critical current density/temperature for a NbTi superconductor (Lubell scaling).

    Direct port of the module-level function of the same name. The one data-dependent
    branch (`bratio < 1`) is `jnp.where`-guarded against a negative fractional-power base
    on the untaken arm.

    Parameters
    ----------
    temp_conductor :
        Superconductor temperature (K).
    b_conductor :
        Magnetic field at the conductor (T).
    c0 :
        Scaling constant (A/m^2).
    b_c20max :
        Upper critical field (T) at zero temperature and strain.
    temp_c0max :
        Critical temperature (K) at zero field and strain.

    Returns
    -------
    tuple
        `(j_critical, temp_critical)`.
    """
    bratio = b_conductor / b_c20max
    below_critical_field = bratio < 1.0

    safe_complement = jnp.where(below_critical_field, 1.0 - bratio, 0.0)
    temp_critical = jnp.where(
        below_critical_field,
        temp_c0max * safe_pow(safe_complement, 0.59),
        temp_c0max * (1.0 - bratio),
    )

    tbar = 1.0 - temp_conductor / temp_critical
    j_critical = c0 * (1.0 - bratio) * tbar

    return j_critical, temp_critical


def bi2212(b_conductor, jstrand, temp_conductor, f_strain):
    """Fitted Bi-2212 superconductor critical current density and temperature margin.

    Direct port of the module-level function of the same name. Source: `raise
    ProcessValueError(...)` outside the fit's validity range (`temp_conductor > 20.0`
    or `b_conductor < 6.0` or `b > 104.0`, computed *after* both return values -- so
    the port computes the identical arithmetic either way). A traced function cannot
    raise on a data-dependent condition, so the port masks both outputs to `jnp.nan`
    there instead; the harness case declares
    `reference_domain_errors = (ProcessValueError,)` and asserts exactly that (same
    idiom as `plasma_profiles.py`'s `_gradient_length`).

    Parameters
    ----------
    b_conductor :
        Magnetic field at the conductor (T).
    jstrand :
        Current density in strand (A/m^2).
    temp_conductor :
        Superconductor temperature (K).
    f_strain :
        Strain/radiation/fatigue/AC-loss adjustment factor (<= 1).

    Returns
    -------
    tuple
        `(j_critical, temp_margin)` -- critical current density in strand (A/m^2),
        temperature margin (K). Both `nan` outside the fit's validity range.
    """
    b = b_conductor / jnp.exp(-0.168 * (temp_conductor - 4.2))

    j_critical = f_strain * (1.175e9 * jnp.exp(-0.02115 * b) - 1.288e8)

    temp_margin = (
        (1.0 / 0.168)
        * jnp.log(
            jnp.log(1.175e9 / (jstrand / f_strain + 1.288e8)) / (0.02115 * b_conductor)
        )
        + 4.2
        - temp_conductor
    )

    out_of_range = (temp_conductor > 20.0) | (b_conductor < 6.0) | (b > 104.0)
    j_critical = jnp.where(out_of_range, jnp.nan, j_critical)
    temp_margin = jnp.where(out_of_range, jnp.nan, temp_margin)

    return j_critical, temp_margin


def gl_nbti(temp_conductor, b_conductor, strain, b_c20max, t_c0):
    """Critical current density/field/temperature for Nb-Ti (Ginzburg-Landau theory).

    Direct port of the module-level function of the same name. The one data-dependent
    branch (`b_reduced <= 1.0`, selecting the exponent applied to `(1 - b_reduced)`) is
    `jnp.where`-guarded against a negative fractional-power base on the untaken arm --
    the `> 1.0` arm's own exponent is exactly `1.0`, so it needs no guard itself.

    Parameters
    ----------
    temp_conductor :
        Superconductor temperature (K).
    b_conductor :
        Magnetic field at the conductor (T).
    strain :
        Intrinsic strain in the superconductor (%).
    b_c20max :
        Strain-dependent upper critical field at zero temperature (T).
    t_c0 :
        Strain-dependent critical temperature at zero strain (K).

    Returns
    -------
    tuple
        `(j_critical, b_critical, t_critical)`.
    """
    a_0 = 1102e6
    p = 0.49
    q = 0.56
    n = 1.83
    v = 1.42
    u = 0.0
    w = 2.2

    c2 = -0.0025
    c3 = -0.0003
    c4 = -0.0001
    epsilon_m = -0.002e-2

    epsilon_i = strain - epsilon_m
    strain_func = 1.0 + c2 * epsilon_i**2 + c3 * epsilon_i**3 + c4 * epsilon_i**4

    t_e = t_c0 * strain_func ** (1.0 / w)
    t_reduced = temp_conductor / t_e
    a_e = a_0 * strain_func ** (u / w)

    b_critical = b_c20max * (1.0 - t_reduced**v) * strain_func
    b_reduced = b_conductor / b_critical
    t_critical = t_e

    below_bc2 = b_reduced <= 1.0
    safe_complement = jnp.where(below_bc2, 1.0 - b_reduced, 1.0)
    field_factor = jnp.where(below_bc2, safe_complement**q, (1.0 - b_reduced) ** 1.0)

    j_critical = (
        a_e
        * (t_e * (1.0 - t_reduced**2)) ** 2
        * b_critical ** (n - 3.0)
        * b_reduced ** (p - 1.0)
        * field_factor
    )

    return j_critical, b_critical, t_critical


def gl_rebco(temp_conductor, b_conductor, strain, b_c20max, t_c0):
    """Critical current density/field/temperature for SuperPower REBCO tape (GL theory).

    Direct port of the module-level function of the same name -- already unbranched
    in the source (unlike `gl_nbti`, it applies the fractional exponent `q`
    unconditionally). See the audit record's JAX-difficulty flags for the resulting
    corner (`b_reduced > 1`, unphysical, not exercised by the harness's fuzz range).

    Parameters
    ----------
    temp_conductor :
        Coolant/superconductor temperature (K).
    b_conductor :
        Magnetic field at conductor (T).
    strain :
        Intrinsic strain in superconductor (%).
    b_c20max :
        Strain-dependent upper critical field at zero temperature (T).
    t_c0 :
        Strain-dependent critical temperature at zero strain (K).

    Returns
    -------
    tuple
        `(j_critical, b_critical, temp_critical)`.
    """
    a_0 = 2.95e2

    p = 0.32
    q = 2.50
    n = 3.33
    s = 5.27

    c2 = -0.0191
    c3 = 0.0039
    c4 = 0.00103
    epsilon_m = 0.058

    u = 0.0
    w = 2.2

    epsilon_i = strain - epsilon_m
    strain_func = 1.0 + c2 * epsilon_i**2 + c3 * epsilon_i**3 + c4 * epsilon_i**4

    t_e = t_c0 * strain_func ** (1.0 / w)
    t_reduced = temp_conductor / t_e
    a_e = a_0 * strain_func ** (u / w)

    b_critical = b_c20max * (1.0 - t_reduced) ** s * strain_func
    b_reduced = b_conductor / b_critical
    temp_critical = t_e

    j_critical = (
        a_e
        * (t_e * (1.0 - t_reduced**2)) ** 2
        * b_critical ** (n - 3.0)
        * b_reduced ** (p - 1.0)
        * (1.0 - b_reduced) ** q
    )

    return j_critical, b_critical, temp_critical


def western_superconducting_nb3sn(
    temp_conductor, b_conductor, strain, b_c20max, temp_c0max
):
    """Critical current density/field/temperature for WST Nb3Sn (ITER surface model).

    Direct port of the module-level function of the same name: `bottura_scaling` with
    WST's fitting constants, then a strand-area unit conversion.

    Parameters
    ----------
    temp_conductor :
        Superconductor temperature (K).
    b_conductor :
        Magnetic field at the superconductor (T).
    strain :
        Strain in the superconductor.
    b_c20max :
        Upper critical field (T) at zero temperature and strain.
    temp_c0max :
        Critical temperature (K) at zero field and strain.

    Returns
    -------
    tuple
        `(j_critical, b_critical, t_critical)`.
    """
    csc = 83075.0
    p = 0.593
    q = 2.156
    c_a1 = 50.06
    c_a2 = 0.0
    epsilon_0a = 0.00312

    j_scaling, b_critical, t_critical = bottura_scaling(
        csc=csc,
        p=p,
        q=q,
        c_a1=c_a1,
        c_a2=c_a2,
        epsilon_0a=epsilon_0a,
        temp_conductor=temp_conductor,
        b_conductor=b_conductor,
        epsilon=strain,
        b_c20max=b_c20max,
        temp_c0max=temp_c0max,
    )

    scalefac = 1.0e6
    j_critical = j_scaling * scalefac

    return j_critical, b_critical, t_critical


# No `cottax` node for any of the 8 functions above -- see the audit record's "cottax
# node" section. Every real call site's arguments (`b_max`, `t_helium`, and the
# per-branch `bc20m`/`tc0m`) are locals inside `jcrit_from_material`'s unported switch
# (`process/models/stellarator/coils/coils.py`, unit #10), not established
# `.area.field` paths this audit has independently verified -- wrapping any of these
# as an `ExplicitFunction` now would assert a wiring this pass has no basis for. The
# natural home, once unit #10 mints real `VarPath`s for its locals, is one node per
# `i_tf_sc_mat` branch (per `traceability_policy.md`'s split-by-default and the
# reads-set evidence in the audit record's "switches touched" section), e.g.:
#
#   class ItersMaterial(ExplicitFunction):
#       j_crit_sc = Output(tfcoil.j_crit_sc)  # sketch only -- VarPath TBD
#
#       def __call__(
#           self,
#           t_helium=FromExactly(tfcoil.t_helium),  # sketch only -- VarPath TBD
#           b_max=FromExactly(tfcoil.b_max),  # sketch only -- VarPath TBD
#           strain=-0.005,  # static, per jcrit_from_material
#           b_c20max=32.97,  # static literal, i_tf_sc_mat == 1
#           temp_c0max=16.06,  # static literal, i_tf_sc_mat == 1
#       ):
#           j_critical, _b_critical, _temp_critical = itersc(
#               t_helium, b_max, strain, b_c20max, temp_c0max
#           )
#           return j_critical
#
# -- one such node per branch, chosen at graph-build time by `i_tf_sc_mat`'s value,
# not a single node with an internal switch. Design step for whoever ports
# `jcrit_from_material`, not this unit.
