"""Pure physics functions extracted from `models/physics/plasma_profiles.py`.

See that file for the declarations wiring these into the graph, and
`_audit/units/models/physics/plasma_profiles.md` for the audit record. No
graph-framework import belongs in this module -- see `_audit/formulas_split.md`.
"""

import jax.numpy as jnp
from jax.scipy.special import gamma, gammaln

from functional_process.models.safe_math import safe_sqrt

# `process/core/constants.py`'s KILOELECTRON_VOLT -- the J-per-keV conversion.
KILOELECTRON_VOLT = 1.602176634e-16

L_MODE_PROFILE_VALUES = (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0)
"""`parabolic_parameterisation`'s own literals, in this module's argument order."""


def _beta(a, b):
    """`scipy.special.beta`, which `jax.scipy.special` does not provide.

    Via `gammaln` rather than `gamma(a) * gamma(b) / gamma(a + b)`: the log form is what
    keeps the intermediate in range, and it is what `scipy` does internally. Agrees with
    `scipy.special.beta` to ~2e-15 relative -- three orders inside tier 1's `rtol=1e-12`
    -- and differentiates.
    """
    return jnp.exp(gammaln(a) + gammaln(b) - gammaln(a + b))


def _simpson(y, x):
    """`scipy.integrate.simpson`'s **general, non-uniform** composite Simpson's rule.

    PROCESS integrates the profile arrays with `sp.integrate.simpson(arg, x=rho, ...)`.
    A fixed quadrature rule, not a solver -- hence tier 1 -- but with two traps, both of
    which this port hit:

    - **It switches formula when the number of intervals is odd.**
      `n_plasma_profile_elements` defaults to 201, i.e. 200 intervals, so the composite
      rule applies. Asserted below rather than assumed, since an even-point grid would
      disagree with PROCESS here and nowhere else.
    - **Passing `x` selects the non-uniform formula, even when `x` happens to be
      uniform.** The uniform shortcut `h = (x[-1] - x[0]) / n` with `1,4,2,...,4,1`
      weights gives the *same value* on a uniform grid but a *different derivative with
      respect to each `x[i]`*, because it only sees the two endpoints. That is a silent
      difference no value comparison can detect: it was caught by
      `Tier1Contract.test_gradient_agreement`, which found `d(output)/d(profile_x[i])`
      wrong by factors of 2-30 while every value agreed to machine precision. Exactly the
      failure mode the gradient check was built for.

    So this implements the general form (Cartwright's, as `scipy` uses): over each pair
    of intervals with widths `h0`, `h1`,

        (h0 + h1)/6 * [(2 - h1/h0) y0 + (h0 + h1)^2/(h0 h1) y1 + (2 - h0/h1) y2]

    which reduces to `h/3 * (y0 + 4 y1 + y2)` when `h0 == h1`. Verified against `scipy`
    to 0 relative error on uniform 11- and non-uniform 11-point grids, and 4e-16 on the
    201-point grid PROCESS actually uses.
    """
    n_intervals = y.shape[0] - 1
    if n_intervals % 2:
        raise ValueError(
            f"{y.shape[0]} profile points means {n_intervals} intervals (odd); "
            f"`scipy.integrate.simpson` falls back to a different rule there and this "
            f"port would silently disagree. PROCESS's default is 201 points"
        )
    h = x[1:] - x[:-1]
    h0, h1 = h[0::2], h[1::2]
    y0, y1, y2 = y[0:-1:2], y[1::2], y[2::2]
    hsum = h0 + h1
    return jnp.sum(
        hsum
        / 6.0
        * ((2.0 - h1 / h0) * y0 + hsum**2 / (h0 * h1) * y1 + (2.0 - h0 / h1) * y2)
    )


def calculate_ion_vol_avg_temperature(
    f_temp_plasma_ion_electron,
    temp_plasma_electron_vol_avg_kev,
    temp_plasma_ion_vol_avg_kev,
):
    """Volume-averaged ion temperature (keV), `parameterise_plasma` L64-68.

    PROCESS writes this field **only** when `f_temp_plasma_ion_electron > 0`, and
    otherwise uses whatever the input supplied. A traced port cannot conditionally skip a
    write, so the incumbent value is taken as an argument and selected between -- which
    makes `.physics.temp_plasma_ion_vol_avg_kev` both a read and a write of this unit.
    That is faithful, not a workaround: PROCESS's field really does serve both roles.

    Parameters
    ----------
    f_temp_plasma_ion_electron :
        Ion-to-electron temperature ratio; `<= 0` means "use the input directly".
    temp_plasma_electron_vol_avg_kev :
        Volume-averaged electron temperature (keV).
    temp_plasma_ion_vol_avg_kev :
        The incumbent value, used unchanged when the ratio is not positive.

    Returns
    -------
    temp_plasma_ion_vol_avg_kev :
        Volume-averaged ion temperature (keV).
    """
    return jnp.where(
        f_temp_plasma_ion_electron > 0.0,
        f_temp_plasma_ion_electron * temp_plasma_electron_vol_avg_kev,
        temp_plasma_ion_vol_avg_kev,
    )


def calculate_parabolic_profile_values(
    alphan,
    alphat,
    nd_plasma_electrons_vol_avg,
    temp_plasma_electron_vol_avg_kev,
    temp_plasma_ion_vol_avg_kev,
):
    """Line-averaged and density-weighted profile values for `i_plasma_pedestal == 0`.

    Ports `parabolic_parameterisation`'s arithmetic tail (L126-161). Two things in the
    source method are deliberately **not** here, both per the audit record:

    - The seven-field L-mode reset with its `logger.error` (L92-117) is input validation,
      not physics, and is not traceable (data-dependent logging).
    - The four on-axis writes (L165-181) are `redundant-duplicate-write`s of values
      `TeProfile`/`NeProfile.set_physics_variables` already wrote, algebraically
      identically, forty lines earlier. Keeping PROCESS's rewrite would put a second
      producer for one field in the graph.

    The line-average factors are `<n> (1 + alphan) * (gamma(1/2)/2) * gamma(alphan + 1)
    / gamma(alphan + 3/2)`, i.e. the parabolic profile integrated over rho in [0, 1].

    Parameters
    ----------
    alphan, alphat :
        Density and temperature profile indices.
    nd_plasma_electrons_vol_avg :
        Volume-averaged electron density (m^-3).
    temp_plasma_electron_vol_avg_kev, temp_plasma_ion_vol_avg_kev :
        Volume-averaged electron and ion temperatures (keV).

    Returns
    -------
    tuple
        `(f_temp_plasma_electron_density_vol_avg, nd_plasma_electron_line,
        temp_plasma_electron_line_avg_kev, temp_plasma_electron_density_weighted_kev,
        temp_plasma_ion_density_weighted_kev)`.
    """
    f_temp_plasma_electron_density_vol_avg = (
        (1.0 + alphan) * (1.0 + alphat) / (1.0 + alphan + alphat)
    )

    nd_plasma_electron_line = (
        nd_plasma_electrons_vol_avg
        * (1.0 + alphan)
        * (gamma(0.5) / 2.0)
        * gamma(alphan + 1.0)
        / gamma(alphan + 1.5)
    )
    temp_plasma_electron_line_avg_kev = (
        temp_plasma_electron_vol_avg_kev
        * (1.0 + alphat)
        * (gamma(0.5) / 2.0)
        * gamma(alphat + 1.0)
        / gamma(alphat + 1.5)
    )

    temp_plasma_electron_density_weighted_kev = (
        temp_plasma_electron_vol_avg_kev * f_temp_plasma_electron_density_vol_avg
    )
    temp_plasma_ion_density_weighted_kev = (
        temp_plasma_ion_vol_avg_kev * f_temp_plasma_electron_density_vol_avg
    )

    return (
        f_temp_plasma_electron_density_vol_avg,
        nd_plasma_electron_line,
        temp_plasma_electron_line_avg_kev,
        temp_plasma_electron_density_weighted_kev,
        temp_plasma_ion_density_weighted_kev,
    )


def lmode_profile_reset(
    radius_plasma_pedestal_temp_norm=1.0,
    radius_plasma_pedestal_density_norm=1.0,
    temp_plasma_pedestal_kev=0.0,
    temp_plasma_separatrix_kev=0.0,
    nd_plasma_pedestal_electron=0.0,
    nd_plasma_separatrix_electron=0.0,
    tbeta=2.0,
):
    """The seven L-mode pedestal fields `parabolic_parameterisation` resets, as a total
    function of the incoming values (source L92-117).

    **The result does not depend on the arguments, and that is the whole content of the
    function.** PROCESS's block is `if <any of the seven differs from its L-mode value>:
    <set all seven to their L-mode values>`; when the guard is false every field already
    holds the value the body would assign. So the post-condition is unconditional -- the
    seven fields hold `(1, 1, 0, 0, 0, 0, 2)` on exit whatever they held on entry -- and
    the guard exists only to decide whether `logger.error` fires. That is a diagnostic,
    is data-dependent, and is not ported; the coercion it guards is.

    The arguments are kept, defaulted to the L-mode values, for two reasons. They make
    the independence a *testable* claim rather than a comment
    (`test_plasma_profiles.TestLModeProfileReset` fuzzes all seven against PROCESS and
    the reference returns the same constants every time), and they let `LModeProfileReset`
    call the function with no arguments at all, so the node declares no read and no
    self-loop.

    `plasma_profiles.md` classified these seven as `input-validation-reset` and its open
    question 2 asked whether they belonged in `configuration.py` as graph-assembly-time
    coercion instead. They do not: PROCESS applies them inside the pipeline, on the
    parabolic arm only, and a node under that arm says exactly that with no new
    machinery. `profiles.DensityProfile`'s docstring records why it matters -- its single
    formula is the pedestal one and only degenerates to `n0 * (1 - rho**2) ** alphan`
    once `radius_plasma_pedestal_density_norm == 1` and the two pedestal densities are
    zero.

    Parameters
    ----------
    radius_plasma_pedestal_temp_norm, radius_plasma_pedestal_density_norm :
        Incoming normalised pedestal radii for temperature and density.
    temp_plasma_pedestal_kev, temp_plasma_separatrix_kev :
        Incoming pedestal and separatrix electron temperatures (keV).
    nd_plasma_pedestal_electron, nd_plasma_separatrix_electron :
        Incoming pedestal and separatrix electron densities (m^-3).
    tbeta :
        Incoming density-profile core exponent.

    Returns
    -------
    tuple
        The same seven quantities at their L-mode values, in the same order.
    """
    del (
        radius_plasma_pedestal_temp_norm,
        radius_plasma_pedestal_density_norm,
        temp_plasma_pedestal_kev,
        temp_plasma_separatrix_kev,
        nd_plasma_pedestal_electron,
        nd_plasma_separatrix_electron,
        tbeta,
    )
    return tuple(jnp.asarray(value) for value in L_MODE_PROFILE_VALUES)


def calculate_pedestal_profile_values(
    profile_x,
    ne_profile_y,
    te_profile_y,
    ne_profile_integ,
    te_profile_integ,
    temp_plasma_ion_vol_avg_kev,
    temp_plasma_electron_vol_avg_kev,
    nd_plasma_separatrix_electron,
    nd_plasma_electrons_vol_avg,
):
    """Density-weighted profile values for `i_plasma_pedestal != 0`.

    Ports `pedestal_parameterisation` (L205-247). Where the parabolic arm has closed-form
    profile factors, this one integrates the actual profile arrays:
    `T_n = integral(rho n T drho) / integral(rho n drho)`.

    `profile_dx` is not an argument although the source passes it to `simpson`: `scipy`
    ignores `dx` whenever `x` is given, so passing it here would imply a dependence the
    computation does not have.

    Parameters
    ----------
    profile_x :
        Normalised radius grid, shared by both profiles.
    ne_profile_y, te_profile_y :
        Density (m^-3) and temperature (keV) profiles on `profile_x`.
    ne_profile_integ, te_profile_integ :
        Pre-integrated line averages, computed by the profile objects.
    temp_plasma_ion_vol_avg_kev, temp_plasma_electron_vol_avg_kev :
        Volume-averaged ion and electron temperatures (keV).
    nd_plasma_separatrix_electron, nd_plasma_electrons_vol_avg :
        Separatrix and volume-averaged electron densities (m^-3).

    Returns
    -------
    tuple
        `(temp_plasma_electron_density_weighted_kev,
        temp_plasma_ion_density_weighted_kev,
        f_temp_plasma_electron_density_vol_avg, nd_plasma_electron_line,
        temp_plasma_electron_line_avg_kev, prn1)`. The last is `.divertor.prn1`, this
        file's only cross-area write.
    """
    integ1 = _simpson(profile_x * ne_profile_y * te_profile_y, profile_x)
    integ2 = _simpson(profile_x * ne_profile_y, profile_x)

    temp_plasma_electron_density_weighted_kev = integ1 / integ2
    temp_plasma_ion_density_weighted_kev = (
        temp_plasma_ion_vol_avg_kev
        / temp_plasma_electron_vol_avg_kev
        * temp_plasma_electron_density_weighted_kev
    )
    f_temp_plasma_electron_density_vol_avg = (
        temp_plasma_electron_density_weighted_kev / temp_plasma_electron_vol_avg_kev
    )

    # Scrape-off density / volume-averaged density, floored to prevent a later
    # division by zero (the source's own comment, L247).
    prn1 = jnp.maximum(0.01, nd_plasma_separatrix_electron / nd_plasma_electrons_vol_avg)

    return (
        temp_plasma_electron_density_weighted_kev,
        temp_plasma_ion_density_weighted_kev,
        f_temp_plasma_electron_density_vol_avg,
        ne_profile_integ,
        te_profile_integ,
        prn1,
    )


def calculate_profile_factors(
    ne_profile_y,
    te_profile_y,
    nd_plasma_electron_on_axis,
    temp_plasma_electron_on_axis_kev,
    nd_plasma_ions_on_axis,
    temp_plasma_ion_on_axis_kev,
    nd_plasma_ions_total_vol_avg,
    nd_plasma_electrons_vol_avg,
    nd_plasma_fuel_ions_vol_avg,
    f_temp_plasma_ion_electron,
    temp_plasma_electron_density_weighted_kev,
    temp_plasma_ion_density_weighted_kev,
    alphan,
    alphat,
    alphaj,
    plasma_current,
    a_plasma_poloidal,
):
    """Plasma pressure profiles and central current density; runs in **both** branches.

    Ports `calculate_profile_factors` (L259-324). Four of the eight returns are arrays of
    `n_plasma_profile_elements`.

    The on-axis inputs are produced by `NeProfile`/`TeProfile.set_physics_variables` in
    both branches (with different formulas per branch) -- not by
    `parabolic_parameterisation`, whose writes of them are the redundant duplicates
    noted above. So this function is branch-agnostic, which is why it is `COMMON` rather
    than an arm of the `i_plasma_pedestal` switch.

    `pres_plasma_thermal_vol_avg` uses the **density-weighted** temperatures, not the
    volume-averaged ones, since `<nT> != <n><T>` (the source's comment, L306-307).

    Returns
    -------
    tuple
        `(pres_plasma_thermal_on_axis, pres_plasma_electron_profile,
        pres_plasma_ion_total_profile, pres_plasma_thermal_total_profile,
        pres_plasma_fuel_profile, alphap, pres_plasma_thermal_vol_avg,
        j_plasma_on_axis)`.
    """
    # Central pressure (Pa), from the ideal gas law p = nkT.
    pres_plasma_thermal_on_axis = (
        nd_plasma_electron_on_axis * temp_plasma_electron_on_axis_kev
        + nd_plasma_ions_on_axis * temp_plasma_ion_on_axis_kev
    ) * KILOELECTRON_VOLT

    pres_plasma_electron_profile = ne_profile_y * (te_profile_y * KILOELECTRON_VOLT)

    pres_plasma_ion_total_profile = (
        nd_plasma_ions_total_vol_avg * (ne_profile_y / nd_plasma_electrons_vol_avg)
    ) * (te_profile_y * KILOELECTRON_VOLT * f_temp_plasma_ion_electron)

    pres_plasma_thermal_total_profile = (
        pres_plasma_electron_profile + pres_plasma_ion_total_profile
    )

    pres_plasma_fuel_profile = (
        nd_plasma_fuel_ions_vol_avg * (ne_profile_y / nd_plasma_electrons_vol_avg)
    ) * (te_profile_y * KILOELECTRON_VOLT * f_temp_plasma_ion_electron)

    # Pressure profile index -- only true for a parabolic profile (source's note).
    alphap = alphan + alphat

    pres_plasma_thermal_vol_avg = (
        nd_plasma_electrons_vol_avg * temp_plasma_electron_density_weighted_kev
        + nd_plasma_ions_total_vol_avg * temp_plasma_ion_density_weighted_kev
    ) * KILOELECTRON_VOLT

    # Central plasma current density (A/m^2), assuming a parabolic current profile.
    j_plasma_on_axis = plasma_current * 2 / (_beta(0.5, alphaj + 1) * a_plasma_poloidal)

    return (
        pres_plasma_thermal_on_axis,
        pres_plasma_electron_profile,
        pres_plasma_ion_total_profile,
        pres_plasma_thermal_total_profile,
        pres_plasma_fuel_profile,
        alphap,
        pres_plasma_thermal_vol_avg,
        j_plasma_on_axis,
    )


def calculate_parabolic_gradient_lengths(
    alphat,
    alphan,
    temp_plasma_electron_on_axis_kev,
    nd_plasma_electron_on_axis,
    rminor,
):
    """Normalised gradient lengths at the steepest point; `i_plasma_pedestal == 0` only.

    Ports `calculate_parabolic_profile_factors` (L342-430), minus its `i_plasma_pedestal`
    guard -- `grep -rn calculate_parabolic_profile_factors process tests` finds exactly
    one caller, inside the parabolic branch, so the guard is dead.

    Each of the two profile indices takes one of three arms:

    - `alpha > 1`: `rho_max = 1/sqrt(2 alpha - 1)`, the analytic point where the second
      derivative vanishes.
    - `0 < alpha <= 1`: the gradient diverges at the edge, so the source pins
      `rho_max = 0.9` as a deliberate approximation (its comment calls the value
      "wrong").
    - `alpha <= 0`: PROCESS raises `ProcessValueError`. A traced port cannot raise on a
      data-dependent condition, so it returns non-finite here instead; the contract
      declares `reference_domain_errors` and asserts exactly that.

    The `alpha > 1` arm is evaluated at `alpha <= 1` too (both arms of a `jnp.where` are
    traced), where `(-1 + alpha) ** (-1 + alpha)` has a negative base and would be NaN.
    The base is therefore clamped away from the invalid region before the power, so the
    untaken branch stays finite -- the exact failure `Tier1Contract.test_outputs_finite`
    exists to catch.

    Returns
    -------
    tuple
        `(gradient_length_te, gradient_length_ne)`.
    """
    gradient_length_te = _gradient_length(
        alphat, temp_plasma_electron_on_axis_kev, rminor
    )
    gradient_length_ne = _gradient_length(alphan, nd_plasma_electron_on_axis, rminor)
    return gradient_length_te, gradient_length_ne


def _gradient_length(alpha, on_axis_value, rminor):
    """One profile's normalised gradient length; shared by the te and ne cases.

    The source spells the temperature and density blocks out separately (L346-420) with
    identical structure and different variable names. They are one function.
    """
    steep = alpha > 1.0

    # Guard the `alpha > 1` arm's base so the untaken branch cannot produce a NaN that
    # `jnp.where` would then propagate through the gradient.
    safe_alpha = jnp.where(steep, alpha, 2.0)

    rho_steep = 1.0 / safe_sqrt(-1.0 + 2.0 * safe_alpha)
    dvdrho_steep = (
        -(2.0**safe_alpha)
        * (-1.0 + safe_alpha) ** (-1.0 + safe_alpha)
        * safe_alpha
        * (-1.0 + 2.0 * safe_alpha) ** (0.5 - safe_alpha)
        * on_axis_value
    )

    # The `0 < alpha <= 1` arm: rho pinned at 0.9, gradient evaluated directly.
    rho_boxy = 0.9
    dvdrho_boxy = (
        -2.0 * alpha * rho_boxy * (1.0 - rho_boxy**2) ** (-1.0 + alpha) * on_axis_value
    )

    rho_max = jnp.where(steep, rho_steep, rho_boxy)
    dvdrho_max = jnp.where(steep, dvdrho_steep, dvdrho_boxy)
    v_max = on_axis_value * (1.0 - rho_max**2) ** alpha

    gradient_length = -dvdrho_max * rminor * rho_max / v_max

    # PROCESS raises on a non-positive index; a traced port returns non-finite instead.
    return jnp.where(alpha > 0.0, gradient_length, jnp.nan)
