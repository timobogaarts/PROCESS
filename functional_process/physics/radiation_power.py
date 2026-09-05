"""Pure physics functions extracted from `models/physics/radiation_power.py`.

See that file for the declarations wiring these into the graph, and
`_audit/units/models/physics/radiation_power.md` for the audit record. No
graph-framework import belongs in this module -- see `_audit/formulas_split.md`.
"""

import jax
import jax.numpy as jnp

from functional_process.models.safe_math import safe_pow
from functional_process.physics.plasma_profiles import _simpson

W_TO_MW = 1.0e-6
"""`integrate_radiation_loss_profiles`'s W/m^3 -> MW/m^3 factor."""

PROFILE_INTEGRAL_JACOBIAN = 2.0
"""The factor 2 in `integrate_radiation_loss_profiles`.

`2 * integral_0^1 rho f(rho) drho` is the volume average of `f` over a circular
cross-section. PROCESS's own comment there points at
`github.com/ukaea/PROCESS/issues/3968` for why the `rho` weight and the 2 are correct
and not a leftover; carried through unchanged.
"""


def calculate_impurity_radiation_power_density(
    nd_electron_profile,
    temp_electron_profile_kev,
    f_nd_impurity_electron,
    temp_impurity_kev,
    pden_impurity_lz_nd_temp,
):
    """One impurity species' radiated power density profile (W/m^3).

    Ports `impurity_radiation.calculate_impurity_radiation_power_density`
    (`impurity_radiation.py:513-602`) for a single species, with the two table rows and
    the species' relative density passed in rather than indexed out of
    `data.impurity_radiation`.

    `n_i n_e L(Z, T_e)` with `n_i = f_nd n_e`, and `L` interpolated **log-log** in
    temperature. Outside the table's temperature range the interpolation clamps to the
    end value, which is deliberate (line radiation dominates below, bremsstrahlung above)
    -- but see the bug note.

    **Two things in the source are dead code**, and are not ported:

    - the `np.digitize` block (L541-544) computes `indices` and never uses it. It also
      indexes `indices` as an array, so it would raise on a scalar `temp_electron_*`
      despite the type hint allowing one.
    - `impurity_arr_len_tab[i] - 1` is used as the top-of-table index, but the
      interpolation itself always uses the *full* 200-wide row. PROCESS therefore already
      requires `len_tab == 200` (a shorter table would leave trailing zeros and
      `np.log(0)` would blow up the interpolation), so this port uses `[-1]` and drops
      `len_tab` from the signature.

    Bug (reproduced, not fixed)
    ---------------------------
    The two clamps write `L` itself into the result:

        pden_impurity_profile[temp <= table_min] = pden_impurity_lz_nd_temp_array[i, 0]
        pden_impurity_profile[temp >= table_max] = pden_impurity_lz_nd_temp_array[i, -1]

    `L` is ~1e-33 W m^3; the surrounding quantity is `f_nd * n_e^2 * L` ~ 1e6 W/m^3. The
    clamp is thus not "use the end-of-table loss function" but "set the radiated power to
    zero", off by `f_nd * n_e^2` ~ 4e38. Measured at `T_e = 50 keV` (above the 40 keV
    table top) with `n_e = 8e19`, `f_nd = 0.05`: PROCESS returns `1.685e-33` where the
    intended clamp gives `5.4e5`. Since the table spans 0.001-40 keV and on-axis
    temperatures above 40 keV are reachable in high-performance designs, this is not
    unreachable. Left exactly as PROCESS has it.

    Parameters
    ----------
    nd_electron_profile :
        Electron density profile (m^-3), on the profile grid.
    temp_electron_profile_kev :
        Electron temperature profile (keV), same grid.
    f_nd_impurity_electron :
        This species' density relative to the electron density (`n_imp / n_e`).
    temp_impurity_kev :
        The species' L(Z, Te) table temperatures (keV), ascending.
    pden_impurity_lz_nd_temp :
        The species' L(Z, Te) values (W m^3), same length.

    Returns
    -------
    :
        Radiated power density profile (W/m^3), same shape as the input profiles.
    """
    # Linear interpolation in log-log space, i.e. a piecewise power law in T_e.
    power_loss_function = jnp.exp(
        jnp.interp(
            jnp.log(temp_electron_profile_kev),
            jnp.log(temp_impurity_kev),
            jnp.log(pden_impurity_lz_nd_temp),
        )
    )

    pden_impurity_profile = (
        f_nd_impurity_electron
        * nd_electron_profile
        * nd_electron_profile
        * power_loss_function
    )

    # Both clamps as PROCESS writes them -- see the bug note above.
    pden_impurity_profile = jnp.where(
        temp_electron_profile_kev <= temp_impurity_kev[0],
        pden_impurity_lz_nd_temp[0],
        pden_impurity_profile,
    )
    return jnp.where(
        temp_electron_profile_kev >= temp_impurity_kev[-1],
        pden_impurity_lz_nd_temp[-1],
        pden_impurity_profile,
    )


def create_f_rad_core_profile(
    profile_x, radius_plasma_core_norm, f_p_plasma_core_rad_reduction
):
    """The core-region weight profile: `f` inside `radius_plasma_core_norm`, 0 outside.

    Ports `impurity_radiation.create_f_rad_core_profile` (`impurity_radiation.py:379-405`)
    -- a boolean mask assignment there, a `jnp.where` here. The comparison is strict
    (`<`), matching the source.

    This is a **step function of `radius_plasma_core_norm` and of each `profile_x[i]`**.
    Its derivative is zero almost everywhere and undefined at the grid point that
    straddles the boundary, which is faithful but means a finite-difference comparison is
    meaningless whenever a grid point sits exactly on `radius_plasma_core_norm`. The
    harness cases keep the boundary off-grid for that reason.
    """
    return jnp.where(
        profile_x < radius_plasma_core_norm, f_p_plasma_core_rad_reduction, 0.0
    )


def calculate_impurity_radiation_totals(
    profile_x,
    nd_electron_profile,
    temp_electron_profile_kev,
    f_nd_impurity_electron_array,
    temp_impurity_kev_array,
    pden_impurity_lz_nd_temp_array,
    radius_plasma_core_norm,
    f_p_plasma_core_rad_reduction,
):
    """Volume-averaged total and core impurity radiation power densities (MW/m^3).

    Ports the whole of `ImpurityRadiation.calculate_imprad()` -- `map_imprad_profile`,
    `calculate_radiation_loss_profiles` and `integrate_radiation_loss_profiles`
    (`impurity_radiation.py:677-755`) -- as one function. The class's four zero-initialised
    accumulator arrays and three zero-initialised scalars exist only because PROCESS
    accumulates species by mutating `self`; a sum over the species axis is the same thing.

    **The species axis is already the selected subset.** PROCESS computes
    `self.imp = np.nonzero(f_nd_impurity_electron_array > 1e-30)[0]` in
    `ImpurityRadiation.__init__` and maps only over those indices. That selection is a
    data-dependent gather and cannot be traced, so it is resolved at graph-assembly time
    instead (see `ImpurityRadiationTotals.imp_indices`) and this function receives the
    three per-species arguments already stacked over the selected species, in ascending
    index order.

    That is not a cosmetic difference from "sum over all 14 species". Where the profile
    temperature is *inside* the table range the two agree exactly, because an absent
    species contributes `f_nd * n_e^2 * L = 0`. Where it is outside, the clamp documented
    in `calculate_impurity_radiation_power_density` overwrites that zero with `L` itself,
    so a species with `f_nd == 0` would contribute a spurious `~1e-33` per grid point per
    species. Small, but nonzero and not what PROCESS computes -- hence the faithful
    subset.

    Parameters
    ----------
    profile_x :
        Normalised radius grid (`neprofile.profile_x`), 0 to 1, uniform.
    nd_electron_profile, temp_electron_profile_kev :
        Electron density (m^-3) and temperature (keV) profiles on that grid.
    f_nd_impurity_electron_array :
        `(n_species,)` relative densities of the selected species.
    temp_impurity_kev_array, pden_impurity_lz_nd_temp_array :
        `(n_species, n_table)` L(Z, Te) tables for the selected species.
    radius_plasma_core_norm, f_p_plasma_core_rad_reduction :
        Normalised radius bounding the 'core' region, and the fraction of core radiation
        counted towards the core total.

    Returns
    -------
    tuple
        `(pden_impurity_rad_total_mw, pden_impurity_core_rad_total_mw)`, both MW/m^3.
    """
    per_species = jax.vmap(
        calculate_impurity_radiation_power_density, in_axes=(None, None, 0, 0, 0)
    )(
        nd_electron_profile,
        temp_electron_profile_kev,
        f_nd_impurity_electron_array,
        temp_impurity_kev_array,
        pden_impurity_lz_nd_temp_array,
    )
    pden_impurity_radiation_profile = jnp.sum(per_species, axis=0)

    f_rad_core_profile = create_f_rad_core_profile(
        profile_x, radius_plasma_core_norm, f_p_plasma_core_rad_reduction
    )

    pden_impurity_rad_profile = pden_impurity_radiation_profile * profile_x
    pden_impurity_core_rad_profile = pden_impurity_radiation_profile * (
        profile_x * f_rad_core_profile
    )

    scale = PROFILE_INTEGRAL_JACOBIAN * W_TO_MW
    return (
        scale * _simpson(pden_impurity_rad_profile, profile_x),
        scale * _simpson(pden_impurity_core_rad_profile, profile_x),
    )


def psync_albajar_fidone(
    nd_plasma_electron_on_axis,
    rminor,
    b_plasma_toroidal_on_axis,
    aspect,
    alphan,
    alphat,
    tbeta,
    temp_plasma_electron_on_axis_kev,
    f_sync_reflect,
    rmajor,
    kappa,
    vol_plasma,
):
    """Synchrotron radiation power density (MW/m^3), Albajar-Fidone fit.

    Ports `radiation_power.psync_albajar_fidone` (`radiation_power.py:142-243`)
    unchanged -- straight-line arithmetic over twelve scalars, no `data` access, no
    branch. `np.exp` becomes `jnp.exp`; nothing else moves.

    Domain note: the fit's `k_function` raises `(tbeta**1.53 + 1.87 alphat - 0.16)` to a
    negative power, so it is non-finite where that base is <= 0 (e.g. `tbeta` small and
    `alphat` small). PROCESS does not guard this either -- it returns `nan`/`inf` in
    float64 rather than raising -- so the port needs no `reference_domain_errors` and the
    harness's finiteness check simply requires the sample points to stay in the physical
    region. Same for `p_a0**0.41` at zero density.

    Parameters
    ----------
    nd_plasma_electron_on_axis :
        Central electron density (m^-3).
    rminor, rmajor :
        Plasma minor and major radii (m).
    b_plasma_toroidal_on_axis :
        Toroidal field on axis (T).
    aspect :
        Aspect ratio.
    alphan, alphat, tbeta :
        Density and temperature profile indices, and the temperature profile exponent.
    temp_plasma_electron_on_axis_kev :
        Central electron temperature (keV).
    f_sync_reflect :
        Wall reflectivity for synchrotron radiation.
    kappa :
        Elongation.
    vol_plasma :
        Plasma volume (m^3); the fit gives a total power, which is divided out here.

    Returns
    -------
    :
        Synchrotron radiation power per unit volume (MW/m^3).
    """
    # Variable names follow the reference papers, as PROCESS's own comment says.
    ne0_20 = 1.0e-20 * nd_plasma_electron_on_axis

    p_a0 = 6.04e3 * (rminor * ne0_20) / b_plasma_toroidal_on_axis

    g_function = 0.93 * (1.0 + 0.85 * jnp.exp(-0.82 * aspect))

    k_function = (
        (alphan + 3.87 * alphat + 1.46) ** -0.79
        * (1.98 + alphat) ** 1.36
        * tbeta**2.14
        * (tbeta**1.53 + 1.87 * alphat - 0.16) ** -1.33
    )

    dum = (
        1.0
        + 0.12
        * (temp_plasma_electron_on_axis_kev / safe_pow(p_a0, 0.41))
        * safe_pow(1.0 - f_sync_reflect, 0.41)
    ) ** -1.51

    p_sync_mw = (
        3.84e-8
        * safe_pow(1.0 - f_sync_reflect, 0.62)
        * rmajor
        * rminor**1.38
        * safe_pow(kappa, 0.79)
        * b_plasma_toroidal_on_axis**2.62
        * safe_pow(ne0_20, 0.38)
        * temp_plasma_electron_on_axis_kev
        * (16.0 + temp_plasma_electron_on_axis_kev) ** 2.61
        * dum
        * g_function
        * k_function
    )

    # Albajar gives a total; PROCESS reports a density.
    return p_sync_mw / vol_plasma


def combine_radiation_powers(
    pden_impurity_rad_total_mw,
    pden_impurity_core_rad_total_mw,
    pden_plasma_sync_mw,
):
    """The three derived radiation power densities (MW/m^3).

    Ports `calculate_radiation_powers`'s own arithmetic (`radiation_power.py:106-139`),
    which is the only thing in that function that is neither the impurity model nor the
    synchrotron fit. Synchrotron radiation is booked entirely to the core.

    `pden_plasma_sync_mw` is an argument rather than a return because
    `SynchrotronRadiationPower` already owns that `VarPath`; two nodes cannot own one
    variable. The four-value `RadpwrData` PROCESS returns is reassembled by
    `calculate_radiation_powers` below.

    Returns
    -------
    tuple
        `(pden_plasma_core_rad_mw, pden_plasma_outer_rad_mw, pden_plasma_rad_mw)`.
    """
    pden_plasma_outer_rad_mw = (
        pden_impurity_rad_total_mw - pden_impurity_core_rad_total_mw
    )
    pden_plasma_core_rad_mw = pden_impurity_core_rad_total_mw + pden_plasma_sync_mw
    pden_plasma_rad_mw = pden_impurity_rad_total_mw + pden_plasma_sync_mw
    return pden_plasma_core_rad_mw, pden_plasma_outer_rad_mw, pden_plasma_rad_mw


def calculate_radiation_powers(
    profile_x,
    nd_electron_profile,
    temp_electron_profile_kev,
    f_nd_impurity_electron_array,
    temp_impurity_kev_array,
    pden_impurity_lz_nd_temp_array,
    radius_plasma_core_norm,
    f_p_plasma_core_rad_reduction,
    nd_plasma_electron_on_axis,
    rminor,
    b_plasma_toroidal_on_axis,
    aspect,
    alphan,
    alphat,
    tbeta,
    temp_plasma_electron_on_axis_kev,
    f_sync_reflect,
    rmajor,
    kappa,
    vol_plasma,
):
    """`radiation_power.calculate_radiation_powers`, with both back doors closed.

    The whole point of the unit: PROCESS's signature is thirteen scalars plus a
    `PlasmaProfile` object plus the entire `DataStructure`; this one is the same thirteen
    scalars plus the eight values that were actually being read off those two objects.
    Nothing else is reachable from the source. Its `RadpwrData` return is a plain tuple
    here, in the dataclass's own field order.

    This composite has **no graph node of its own**. The graph expresses it as the
    three nodes below (`ImpurityRadiationTotals` -> `SynchrotronRadiationPower` ->
    `PlasmaRadiationPowers`), which is strictly more informative: the synchrotron fit and
    the impurity model share no input and their results only meet in three additions. The
    composite exists so a harness case can diff the *whole* of PROCESS's entry point,
    object arguments and all, against the port in one call -- which is the check that
    proves the read set above is complete.

    Returns
    -------
    tuple
        `(pden_plasma_sync_mw, pden_plasma_core_rad_mw, pden_plasma_outer_rad_mw,
        pden_plasma_rad_mw)`, all MW/m^3 -- `RadpwrData`'s field order.
    """
    pden_impurity_rad_total_mw, pden_impurity_core_rad_total_mw = (
        calculate_impurity_radiation_totals(
            profile_x,
            nd_electron_profile,
            temp_electron_profile_kev,
            f_nd_impurity_electron_array,
            temp_impurity_kev_array,
            pden_impurity_lz_nd_temp_array,
            radius_plasma_core_norm,
            f_p_plasma_core_rad_reduction,
        )
    )

    pden_plasma_sync_mw = psync_albajar_fidone(
        nd_plasma_electron_on_axis,
        rminor,
        b_plasma_toroidal_on_axis,
        aspect,
        alphan,
        alphat,
        tbeta,
        temp_plasma_electron_on_axis_kev,
        f_sync_reflect,
        rmajor,
        kappa,
        vol_plasma,
    )

    pden_plasma_core_rad_mw, pden_plasma_outer_rad_mw, pden_plasma_rad_mw = (
        combine_radiation_powers(
            pden_impurity_rad_total_mw,
            pden_impurity_core_rad_total_mw,
            pden_plasma_sync_mw,
        )
    )

    return (
        pden_plasma_sync_mw,
        pden_plasma_core_rad_mw,
        pden_plasma_outer_rad_mw,
        pden_plasma_rad_mw,
    )


def impurity_radiation_totals_from_indexed_impurities(
    imp_indices,
    radius_plasma_profile_norm,
    nd_plasma_electron_profile,
    temp_plasma_electron_profile_kev,
    f_nd_impurity_electron_array_0,
    f_nd_impurity_electron_array_1,
    f_nd_impurity_electron_array_2,
    f_nd_impurity_electron_array_3,
    f_nd_impurity_electron_array_4,
    f_nd_impurity_electron_array_5,
    f_nd_impurity_electron_array_6,
    f_nd_impurity_electron_array_7,
    f_nd_impurity_electron_array_8,
    f_nd_impurity_electron_array_9,
    f_nd_impurity_electron_array_10,
    f_nd_impurity_electron_array_11,
    f_nd_impurity_electron_array_12,
    f_nd_impurity_electron_array_13,
    temp_impurity_keV_array,
    pden_impurity_lz_nd_temp_array,
    radius_plasma_core_norm,
    f_p_plasma_core_rad_reduction,
):
    """Reassembles the fourteen individually-addressed fractions, gathers the
    `imp_indices` species subset, then delegates.

    `imp_indices` is a graph-assembly-time (static) fact -- see
    `ImpurityRadiationTotals.imp_indices`'s own docstring -- so it arrives here as a
    plain value, not a port.
    """
    f_nd_impurity_electron_array = jnp.stack([
        f_nd_impurity_electron_array_0,
        f_nd_impurity_electron_array_1,
        f_nd_impurity_electron_array_2,
        f_nd_impurity_electron_array_3,
        f_nd_impurity_electron_array_4,
        f_nd_impurity_electron_array_5,
        f_nd_impurity_electron_array_6,
        f_nd_impurity_electron_array_7,
        f_nd_impurity_electron_array_8,
        f_nd_impurity_electron_array_9,
        f_nd_impurity_electron_array_10,
        f_nd_impurity_electron_array_11,
        f_nd_impurity_electron_array_12,
        f_nd_impurity_electron_array_13,
    ])
    # A jax array (which `jnp.stack` above always produces) refuses list-shaped fancy
    # indexing outright (`arr[[0, 8]]` raises `TypeError`, not merely a deprecation
    # warning -- jax's own message: "use `arr[array(seq)]`"), unlike a bare numpy array.
    # `jnp.array(...)` once, reused for all three gathers, is the fix and matches every
    # other gather in this codebase.
    selected = jnp.array(imp_indices)
    return calculate_impurity_radiation_totals(
        radius_plasma_profile_norm,
        nd_plasma_electron_profile,
        temp_plasma_electron_profile_kev,
        f_nd_impurity_electron_array[selected],
        temp_impurity_keV_array[selected],
        pden_impurity_lz_nd_temp_array[selected],
        radius_plasma_core_norm,
        f_p_plasma_core_rad_reduction,
    )
