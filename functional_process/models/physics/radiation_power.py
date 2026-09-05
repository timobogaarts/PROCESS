"""Pure-functional port of `process/models/physics/radiation_power.py`.

Registry unit #20. Audit record:
`functional_process/_audit/units/models/physics/radiation_power.md`.

PROCESS's entry point is

    calculate_radiation_powers(plasma_profile, <13 physics scalars>, data_structure)

-- two opaque object arguments, one of which is the whole `DataStructure`. Closing those
two back doors is the whole point of this unit, and the answer turns out to be small:

- **off `plasma_profile`**: three arrays and nothing else --
  `neprofile.profile_y` (electron density profile), `teprofile.profile_y` (electron
  temperature profile) and `neprofile.profile_x` (the normalised radius grid). A fourth,
  `neprofile.profile_dx`, is passed to `scipy.integrate.simpson` but **ignored**, because
  `x=` is given alongside it -- so it is not a dependency at all.
- **off `data_structure`**: six `.impurity_radiation` fields and one `.physics` one, all
  reads. `calculate_radiation_powers` writes **nothing** to `data_structure` --
  `ImpurityRadiation` accumulates only onto `self`, and the caller assigns the returned
  `RadpwrData` itself. The `data` argument is a pure read-through.

### Scope finding

Following that read set means this unit's real source is `radiation_power.py` (243 LOC)
**plus the `ImpurityRadiation` half of `process/models/physics/impurity_radiation.py`**
(756 LOC), which is **not in `unit_registry.md`**. See the record's § Scope correction.
Everything on the impurity path is ported here rather than deferred, because
`calculate_radiation_powers` is 6 lines of arithmetic wrapped around it -- porting the
wrapper alone would have ported nothing.

### What is deliberately not ported

`initialise_imprad` / `init_imp_element` / `read_impurity_file` -- the file readers that
populate the L(Z,Te) tables from `process/data/lz_non_corona_14_elements/*.dat`. They are
I/O run once at startup; their product is a compile-time constant of the graph, not a
value flowing along an edge. `calculate_average_charge_at_temp` is not on this path at
all (nothing in `calculate_radiation_powers` reaches it).

### A latent PROCESS bug, ported faithfully

`calculate_impurity_radiation_power_density`'s two out-of-table clamps assign the *loss
function* `L(Z, Te)` (units W m^3, ~1e-33) directly into the *power density* array (units
W/m^3, ~1e6) instead of clamping `L` and then multiplying by `f_nd * ne^2`. Out of range
the answer is therefore ~38 orders of magnitude too small -- effectively zero. See
`calculate_impurity_radiation_power_density`'s docstring and the record's § open
questions. Reproduced exactly here; **not** silently fixed.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    FromExactly,
    OutputInto,
)

from functional_process.models.physics.plasma_profiles import _simpson
from functional_process.models.safe_math import safe_pow
from functional_process.paths import impurity_radiation, physics

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

    This composite has **no cottax node**. The graph expresses it as the three nodes
    below (`ImpurityRadiationTotals` -> `SynchrotronRadiationPower` ->
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


class SynchrotronRadiationPower(ExplicitFunction):
    """cottax node: `psync_albajar_fidone`, unchanged, ports declared.

    Every input is an ordinary `.physics` field and the single output is
    `.physics.pden_plasma_sync_mw`, a real `physics_variables.py` field. Nothing minted,
    no switch, no alternative arm -- this one is ready for `total_process.COMMON`.

    Note the output is *only* the synchrotron term. `stellarator.py:2147` and
    `physics.py` both assign it to `.physics.pden_plasma_sync_mw` verbatim, so the node's
    output and PROCESS's field are the same quantity with no post-processing.
    """

    pden_plasma_sync_mw = OutputInto(physics)

    def __call__(
        self,
        nd_plasma_electron_on_axis=From(physics),
        rminor=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
        aspect=From(physics),
        alphan=From(physics),
        alphat=From(physics),
        tbeta=From(physics),
        temp_plasma_electron_on_axis_kev=From(physics),
        f_sync_reflect=From(physics),
        rmajor=From(physics),
        kappa=From(physics),
        vol_plasma=From(physics),
    ):
        return psync_albajar_fidone(
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


class ImpurityRadiationTotals(ExplicitFunction):
    """cottax node: `calculate_impurity_radiation_totals`, ports declared.

    **Blocked from `total_process.py` on one thing** -- see `imp_indices` below and the
    record's § open questions 2. Everything else about the node is settled.

    **`f_nd_impurity_electron_array` is read as fourteen individual, index-addressed
    `FromExactly`s**, not one whole-array `FromExactly` -- the same per-index treatment
    `composition.PlasmaComposition`/`CalculateEffectiveChargeIonisationProfiles`
    now give the identical field (`SequenceKey`-addressed, matching the real
    `DataStructure` field's own `list[float]` storage, per `naming_convention.md` §
    "Array elements"). This is what this class's own `__call__` docstring used to flag
    as the open reason the array stayed whole: "the number of parameters would then vary
    with the configuration." **That is resolved, not just worked around**: the signature
    now declares all fourteen indices unconditionally -- the array's length is fixed at
    14 for every real `DataStructure` (`initialise_imprad`'s hardcoded per-species
    arguments never vary the count, only which entries are near-zero), so the *port
    count* no longer depends on `imp_indices` at all. `imp_indices` still selects which
    of the fourteen feed `calculate_impurity_radiation_totals`, exactly as before -- only
    now that selection happens on fourteen individually-named `VarPath`s instead of one
    array `VarPath`, via `jnp.stack` + a static gather inside `__call__`.

    This narrows, but does not close, § open questions 2: the residual blocker there is
    whether `imp_indices` can silently disagree with which fractions are actually
    near-zero *during a solve* (the `f_plasma_fuel_helium3 == 0 and
    f_nd_alpha_thermal_electron == 0` case) -- a data-dependent shape-stability question,
    not a signature-shape one. Per-index addressing does not touch that; see the record's
    updated § open questions 2 for exactly what remains.

    Three minted `VarPath`s, same justification as `ProfileFactors`' two in
    `plasma_profiles.py`:

    - `.physics.nd_plasma_electron_profile`, `.physics.temp_plasma_electron_profile_kev`
      -- **reused, not re-minted**: `plasma_profiles.ProfileFactors` already mints exactly
      these two for exactly these two arrays (`neprofile.profile_y`,
      `teprofile.profile_y`). This unit is the second consumer, which is the evidence
      those spellings were the right call.
    - `.physics.radius_plasma_profile_norm` -- **new here.** The normalised radius grid
      (`neprofile.profile_x`) likewise has no PROCESS storage; it is built by
      `Profile.initialise_profile_x`/`normalise_profile_x` (`profiles.py:60-84`) as
      `arange(n) / (n - 1)` and read straight off the object. Named as a sibling of the
      existing `.impurity_radiation.radius_plasma_core_norm`, which it is compared
      against. Its producer is unit #21 (`profiles.py`), still unported.

    `neprofile.profile_dx` is **not** an input, although PROCESS passes it to both
    `integrate.simpson` calls: `scipy` ignores `dx` whenever `x` is given. Declaring it
    would assert a dependency the computation does not have.

    The two outputs are minted too. `pden_impurity_rad_total_mw` /
    `pden_impurity_core_rad_total_mw` exist only as attributes of the `ImpurityRadiation`
    instance that `calculate_radiation_powers` constructs and discards; they are never
    stored on `data`. Minted into `.impurity_radiation` because that is the area whose
    tables and fractions produce them.
    """

    imp_indices: tuple[int, ...] = eqx.field(static=True)
    """Which of the 14 species are present -- a graph-assembly-time fact.

    PROCESS derives this per evaluation as
    `np.nonzero(f_nd_impurity_electron_array > 1e-30)[0]`
    (`impurity_radiation.py:650-652`). That is a data-dependent gather: neither the
    result's *shape* nor its contents are known to a tracer, so it cannot stay inside the
    node. Hoisting it to a static field is the same move `naming_convention.md` § "switches
    are not ports" prescribes for a topology-changing switch, and it is honest -- which
    impurity species a machine has is a design decision, not a state variable.

    **This is what blocks registration**, though for a narrower reason than it looks.
    Iteration variables 125-136 cover array indices 2-13 with bounds `(1e-8, 0.01)`, 22
    orders above the threshold, so the optimiser cannot deselect a species. But
    `physics.plasma_composition()` recomputes indices 0 (`H_`) and 1 (`He`) every
    evaluation -- on the stellarator path too, `stellarator.py:1910` -- and `He` is
    exactly `0.0` when `f_plasma_fuel_helium3` and `f_nd_alpha_thermal_electron` both are.
    That is a per-run configuration fact, so `imp_indices` *can* be resolved at assembly
    time; nothing in `configuration.py` does it or checks it yet. See the record's § open
    questions 2.

    **Not a signature-shape blocker any more.** Before per-index reads, this field's
    docstring on `__call__` also worried that addressing individual species one at a
    time would make the node's own *parameter count* vary with `imp_indices` -- it does
    not: `__call__` now always declares fourteen (one per index, unconditionally),
    and `imp_indices` only selects a static gather over them, exactly as it selected a
    gather over the one whole-array `FromExactly` before. Only the shape-*during-a-solve*
    question above remains open.
    """

    pden_impurity_rad_total_mw = OutputInto(impurity_radiation)
    pden_impurity_core_rad_total_mw = OutputInto(impurity_radiation)

    def __call__(
        self,
        radius_plasma_profile_norm=From(physics),
        nd_plasma_electron_profile=From(physics),
        temp_plasma_electron_profile_kev=From(physics),
        f_nd_impurity_electron_array_0=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[0]
        ),
        f_nd_impurity_electron_array_1=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[1]
        ),
        f_nd_impurity_electron_array_2=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[2]
        ),
        f_nd_impurity_electron_array_3=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[3]
        ),
        f_nd_impurity_electron_array_4=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[4]
        ),
        f_nd_impurity_electron_array_5=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[5]
        ),
        f_nd_impurity_electron_array_6=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[6]
        ),
        f_nd_impurity_electron_array_7=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[7]
        ),
        f_nd_impurity_electron_array_8=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[8]
        ),
        f_nd_impurity_electron_array_9=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[9]
        ),
        f_nd_impurity_electron_array_10=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[10]
        ),
        f_nd_impurity_electron_array_11=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[11]
        ),
        f_nd_impurity_electron_array_12=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[12]
        ),
        f_nd_impurity_electron_array_13=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[13]
        ),
        temp_impurity_keV_array=From(impurity_radiation),
        pden_impurity_lz_nd_temp_array=From(impurity_radiation),
        radius_plasma_core_norm=From(impurity_radiation),
        f_p_plasma_core_rad_reduction=From(impurity_radiation),
    ):
        """Reassembles the fourteen individually-addressed fractions and selects
        `imp_indices` before forwarding.

        Each species' fraction is its own read (`SequenceKey`-addressed `VarPath`, one
        per index) rather than one whole-array read -- fourteen ports, always,
        regardless of `imp_indices`, so the signature stays fixed as
        `NodalDeclaration` requires. `imp_indices` only changes which of the fourteen
        are gathered out before being handed to `calculate_impurity_radiation_totals`
        -- the selection is a gather on the (now reassembled) read, not a second home
        for the arithmetic, same as before this change. The two 200-wide table
        arguments stay whole-array `FromExactly`s: they are compile-time constants, not
        per-species runtime values a caller would ever address individually.
        """
        return impurity_radiation_totals_from_indexed_impurities(
            self.imp_indices,
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
        )


class PlasmaRadiationPowers(ExplicitFunction):
    """cottax node: `combine_radiation_powers`, unchanged, ports declared.

    Three real `.physics` fields out, one real `.physics` field plus the two
    `ImpurityRadiationTotals` mints in. Structurally ready; it can only be registered
    once `ImpurityRadiationTotals` is, since without that node its two impurity inputs
    have no producer.

    Note what this node does **not** do: `stellarator.py:2152-2159` clips
    `pden_plasma_core_rad_mw` and `pden_plasma_outer_rad_mw` at zero *after* assigning
    them. That `max(..., 0)` is in `st_phys` (unit #1, chunk 1B), not in
    `calculate_radiation_powers`, so it belongs to the caller's node; ported here it would
    double-own the fields. **The other caller,
    `physics.PhysicsCalculations.physics()` (`physics.py:750-753`), does not clip at
    all** -- the two call sites disagree, which is the whole reason the clip cannot live
    in this node: it is not a property of the radiation model, it is a property of one
    of its two callers.

    **So this node's two clipped outputs are minted `_unclipped` names**, and
    `plasma_physics.py`'s `ClippedRadiationPowers` owns the real
    `.physics.pden_plasma_core_rad_mw`/`pden_plasma_outer_rad_mw` fields. Before that
    split this node claimed the real fields while computing the *pre*-clip value -- a
    latent divergence from PROCESS, invisible only because the clip happens to be
    inactive on this run (measured: core `0.0575`, outer `0.0553`, both comfortably
    positive). `pden_plasma_rad_mw` keeps its real name: PROCESS never clips it
    (`stellarator.py:2151`), which is also what makes `KNOWN_MINT_VALUES`'s
    `pden_impurity_rad_total_mw` inversion sound where the core one is not.
    """

    pden_plasma_core_rad_mw_unclipped = OutputInto(physics)
    pden_plasma_outer_rad_mw_unclipped = OutputInto(physics)
    pden_plasma_rad_mw = OutputInto(physics)

    def __call__(
        self,
        pden_impurity_rad_total_mw=From(impurity_radiation),
        pden_impurity_core_rad_total_mw=From(impurity_radiation),
        pden_plasma_sync_mw=From(physics),
    ):
        return combine_radiation_powers(
            pden_impurity_rad_total_mw,
            pden_impurity_core_rad_total_mw,
            pden_plasma_sync_mw,
        )
