"""Pure-functional port of `process/models/stellarator/neoclassics.py` (registry unit #7).

Audit record: `functional_process/_audit/units/models/stellarator/neoclassics.md`.

Only two functions are ported *and* validated by the harness here:
`calculate_profile_values` (`init_profile_values_from_PROCESS`) and
`calculate_effective_thermal_diffusivity` (`st_calc_eff_chi`) — both take only scalar
`data.physics.*`/`data.stellarator*.*` arguments, so `Tier1Contract`'s per-argument
`jax.jacfwd`-vs-finite-difference check (which differentiates one named kwarg at a time
via `float(sample.kwargs[name])`) applies to them unchanged.

The rest of the file's pure functions (`calculate_kt` through `calculate_q_flux`) are
also ported below -- faithful, tier-1, no internal solve -- but are **not** wrapped in a
`cottax` node and have no test file yet. Every one of them takes at least one
species-array argument (`densities`/`temperatures`/etc., always length 4: e, D, T, alpha)
rather than a scalar, and the harness's `Tier1Contract` has no scheme for differentiating
an array-valued argument -- `_jacobian`/`_reference_along` call `float(sample.kwargs[name])`,
which raises on anything but a 0-d/1-element value. This is a harness gap, not a property
of these functions (they are exactly as pure and exactly as tier-1 as the two that are
tested) -- see `neoclassics.md`'s open questions for the finding and what closing it
would need (a per-component fuzz+differentiate scheme, most likely). Do not add these to
`total_process.py` until that lands: an untested node in the graph would misrepresent
"ported" as "validated."
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.models.safe_math import safe_sqrt
from functional_process.paths import (
    impurity_radiation,
    neoclassics,
    physics,
    stellarator,
    stellarator_config,
)
from functional_process.vocabulary import constants

KEV = 1e3 * constants.ELECTRON_CHARGE

# 30-point Gauss-Laguerre quadrature nodes/weights (`Neoclassics.init_neoclassics`).
# Fixed numerical-method constants, not physics inputs -- module-level, not a port
# argument, same treatment as `tf_nuclear_heating.py`'s `_FACT`/`_COEF`/`_DECAY`.
ROOTS = np.array([
    4.740718054080526184e-2,
    2.499239167531593919e-1,
    6.148334543927683749e-1,
    1.143195825666101451,
    1.836454554622572344,
    2.696521874557216147,
    3.725814507779509288,
    4.927293765849881879,
    6.304515590965073635,
    7.861693293370260349,
    9.603775985479263255,
    1.153654659795613924e1,
    1.366674469306423489e1,
    1.600222118898106771e1,
    1.855213484014315029e1,
    2.132720432178312819e1,
    2.434003576453269346e1,
    2.760555479678096091e1,
    3.114158670111123683e1,
    3.496965200824907072e1,
    3.911608494906788991e1,
    4.361365290848483056e1,
    4.850398616380419980e1,
    5.384138540650750571e1,
    5.969912185923549686e1,
    6.618061779443848991e1,
    7.344123859555988076e1,
    8.173681050672767867e1,
    9.155646652253683726e1,
    1.041575244310588886e2,
])
WEIGHTS = np.array([
    1.160440860204388913e-1,
    2.208511247506771413e-1,
    2.413998275878537214e-1,
    1.946367684464170855e-1,
    1.237284159668764899e-1,
    6.367878036898660943e-2,
    2.686047527337972682e-2,
    9.338070881603925677e-3,
    2.680696891336819664e-3,
    6.351291219408556439e-4,
    1.239074599068830081e-4,
    1.982878843895233056e-5,
    2.589350929131392509e-6,
    2.740942840536013206e-7,
    2.332831165025738197e-8,
    1.580745574778327984e-9,
    8.427479123056716393e-11,
    3.485161234907855443e-12,
    1.099018059753451500e-13,
    2.588312664959080167e-15,
    4.437838059840028968e-17,
    5.365918308212045344e-19,
    4.393946892291604451e-21,
    2.311409794388543236e-23,
    7.274588498292248063e-26,
    1.239149701448267877e-28,
    9.832375083105887477e-32,
    2.842323553402700938e-35,
    1.878608031749515392e-39,
    8.745980440465011553e-45,
])
NO_ROOTS = len(ROOTS)

# Species order throughout: (electron, deuterium, tritium, alpha).
_SPECIES_MASS = np.array([
    constants.ELECTRON_MASS,
    constants.PROTON_MASS * 2.0,
    constants.PROTON_MASS * 3.0,
    constants.PROTON_MASS * 4.0,
])
_SPECIES_CHARGE = np.array([-1.0, 1.0, 1.0, 2.0]) * constants.ELECTRON_CHARGE


def calculate_profile_values(
    rho,
    temp_plasma_electron_on_axis_kev,
    temp_plasma_ion_on_axis_kev,
    alphat,
    nd_plasma_electron_on_axis,
    f_plasma_fuel_deuterium,
    nd_plasma_ions_on_axis,
    nd_plasma_alphas_thermal_vol_avg,
    alphan,
    rminor,
):
    """Species density/temperature profiles (and their radial derivatives) at `rho`.

    Ports `Neoclassics.init_profile_values_from_PROCESS`. PROCESS's parabolic profile
    shape, evaluated at one normalised radius `rho` for all four species.

    Parameters
    ----------
    rho :
        Normalised minor radius, `r_eff / rminor` (dimensionless, 0 at the core).
    temp_plasma_electron_on_axis_kev :
        On-axis electron temperature (keV). `.physics.temp_plasma_electron_on_axis_kev`.
    temp_plasma_ion_on_axis_kev :
        On-axis ion temperature (keV), shared by D/T/alpha in the source.
        `.physics.temp_plasma_ion_on_axis_kev`.
    alphat :
        Temperature profile peaking exponent. `.physics.alphat`.
    nd_plasma_electron_on_axis :
        On-axis electron density (/m3). `.physics.nd_plasma_electron_on_axis`.
    f_plasma_fuel_deuterium :
        Deuterium fraction of the fuel ions. `.physics.f_plasma_fuel_deuterium`.
    nd_plasma_ions_on_axis :
        On-axis fuel ion density (/m3). `.physics.nd_plasma_ions_on_axis`.
    nd_plasma_alphas_thermal_vol_avg :
        Volume-averaged thermal alpha density (/m3).
        `.physics.nd_plasma_alphas_thermal_vol_avg`.
    alphan :
        Density profile peaking exponent. `.physics.alphan`.
    rminor :
        Plasma minor radius (m). `.physics.rminor`.

    Returns
    -------
    :
        `(densities, temperatures, dr_densities, dr_temperatures)`, each a
        `(4,)` array ordered (electron, deuterium, tritium, alpha); temperatures in J
        (PROCESS's `KEV` conversion applied), densities in /m3, derivatives with
        respect to physical radius (1/m and J/m respectively).
    """
    one_minus_rho2 = 1.0 - rho**2

    temp_species = jnp.array([
        temp_plasma_electron_on_axis_kev,
        temp_plasma_ion_on_axis_kev,
        temp_plasma_ion_on_axis_kev,
        temp_plasma_ion_on_axis_kev,
    ])
    temperatures = temp_species * one_minus_rho2**alphat * KEV

    densities = jnp.array([
        nd_plasma_electron_on_axis * one_minus_rho2**alphan,
        f_plasma_fuel_deuterium * nd_plasma_ions_on_axis * one_minus_rho2**alphan,
        (1.0 - f_plasma_fuel_deuterium)
        * nd_plasma_ions_on_axis
        * one_minus_rho2**alphan,
        nd_plasma_alphas_thermal_vol_avg * (1.0 + alphan) * one_minus_rho2**alphan,
    ])

    dr_common = -2.0 / rminor * rho * one_minus_rho2 ** (alphat - 1.0) * alphat
    dr_temperatures = temp_species * dr_common * KEV

    dr_common_n = -2.0 / rminor * rho * one_minus_rho2 ** (alphan - 1.0) * alphan
    dr_densities = jnp.array([
        nd_plasma_electron_on_axis * dr_common_n,
        f_plasma_fuel_deuterium * nd_plasma_ions_on_axis * dr_common_n,
        (1.0 - f_plasma_fuel_deuterium) * nd_plasma_ions_on_axis * dr_common_n,
        nd_plasma_alphas_thermal_vol_avg * (1.0 + alphan) * dr_common_n,
    ])

    return densities, temperatures, dr_densities, dr_temperatures


def calculate_effective_thermal_diffusivity(
    vol_plasma,
    f_st_rmajor,
    radius_plasma_core_norm,
    rminor,
    stella_config_rminor_ref,
    a_plasma_surface,
    f_p_alpha_plasma_deposited,
    pden_alpha_total_mw,
    pden_plasma_core_rad_mw,
    nd_plasma_electron_on_axis,
    temp_plasma_electron_on_axis_kev,
    alphat,
    alphan,
):
    """Effective thermal diffusivity from core alpha heating to the boundary.

    Ports `Neoclassics.st_calc_eff_chi`. Independent of the rest of this file's
    neoclassics pipeline -- reads only `.physics.*`/`.impurity_radiation.*`/
    `.stellarator.*`/`.stellarator_config.*`, no `.neoclassics.*` field.

    Parameters
    ----------
    vol_plasma :
        Plasma volume (m3). `.physics.vol_plasma`.
    f_st_rmajor :
        Major-radius scaling factor. `.stellarator.f_st_rmajor`.
    radius_plasma_core_norm :
        Normalised core radius. `.impurity_radiation.radius_plasma_core_norm`.
    rminor :
        Plasma minor radius (m). `.physics.rminor`.
    stella_config_rminor_ref :
        Reference minor radius (m). `.stellarator_config.stella_config_rminor_ref`.
    a_plasma_surface :
        Plasma surface area (m2). `.physics.a_plasma_surface`.
    f_p_alpha_plasma_deposited :
        Fraction of alpha power deposited in plasma. `.physics.f_p_alpha_plasma_deposited`.
    pden_alpha_total_mw :
        Alpha power density (MW/m3). `.physics.pden_alpha_total_mw`.
    pden_plasma_core_rad_mw :
        Core radiated power density (MW/m3). `.physics.pden_plasma_core_rad_mw`.
    nd_plasma_electron_on_axis :
        On-axis electron density (/m3). `.physics.nd_plasma_electron_on_axis`.
    temp_plasma_electron_on_axis_kev :
        On-axis electron temperature (keV). `.physics.temp_plasma_electron_on_axis_kev`.
    alphat :
        Temperature profile peaking exponent. `.physics.alphat`.
    alphan :
        Density profile peaking exponent. `.physics.alphan`.

    Returns
    -------
    :
        `chi_PROCESS_e`, the effective electron thermal diffusivity.
    """
    radius_scaling = radius_plasma_core_norm * rminor / stella_config_rminor_ref

    volscaling = vol_plasma * f_st_rmajor * radius_scaling**2
    surfacescaling = a_plasma_surface * f_st_rmajor * radius_scaling

    nominator = (
        f_p_alpha_plasma_deposited * pden_alpha_total_mw - pden_plasma_core_rad_mw
    ) * volscaling

    # Source comment: a `0 * alphan` term present in the original Fortran was dropped
    # here "for obvious reasons" -- ported as PROCESS's Python already reads, not
    # re-derived.
    denominator = (
        (
            3.0
            * nd_plasma_electron_on_axis
            * constants.ELECTRON_CHARGE
            * temp_plasma_electron_on_axis_kev
            * 1e3
            * alphat
            * radius_plasma_core_norm
            * (1.0 - radius_plasma_core_norm**2) ** (alphan + alphat - 1.0)
        )
        * surfacescaling
        * 1e-6
    )

    return nominator / denominator


def calculate_kt(temperatures):
    """Energy grid from the Gauss-Laguerre roots, per species.

    Ports `Neoclassics.neoclassics_calc_KT`. **Not harness-tested** -- see module
    docstring (array-valued argument, harness gap).

    Parameters
    ----------
    temperatures :
        `(4,)` species temperatures (J). `.neoclassics.temperatures`.

    Returns
    -------
    :
        `(4, NO_ROOTS)` energy grid (J).
    """
    k = jnp.repeat((ROOTS / KEV)[:, None], 4, axis=1)
    return (k * temperatures).T


def _pitch_angle_factor(xk):
    """Chandrasekhar-style erf-based collision factor, shared by `nu`/`nu_star_fromT`."""
    expxk = jnp.exp(-xk)
    t = 1.0 / (1.0 + 0.3275911 * safe_sqrt(xk))
    erfn = (
        1.0
        - t
        * (
            0.254829592
            + t
            * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))
        )
        * expxk
    )
    return (1.0 - 0.5 / xk) * erfn + expxk / safe_sqrt(jnp.pi * xk)


def calculate_collision_frequency(densities, temperatures):
    """Collision frequency on the Gauss-Laguerre energy grid, per species pair.

    Ports `Neoclassics.neoclassics_calc_nu`. **Not harness-tested** -- see module
    docstring.

    Parameters
    ----------
    densities :
        `(4,)` species densities (/m3). `.neoclassics.densities`.
    temperatures :
        `(4,)` species temperatures (J). `.neoclassics.temperatures`.

    Returns
    -------
    :
        `(4, NO_ROOTS)` collision frequency.
    """
    lnlambda = (
        32.2
        - 1.15 * jnp.log10(densities[0])
        + 2.3 * jnp.log10(temperatures[0] / constants.ELECTRON_CHARGE)
    )

    out = jnp.zeros((4, NO_ROOTS))
    for j in range(4):
        for k in range(4):
            xk = (
                (_SPECIES_MASS[k] / _SPECIES_MASS[j])
                * (temperatures[j] / temperatures[k])
                * ROOTS
            )
            phixmgx = _pitch_angle_factor(xk)
            v = safe_sqrt(2.0 * ROOTS * temperatures[j] / _SPECIES_MASS[j])
            out = out.at[j, :].add(
                densities[k]
                * (_SPECIES_CHARGE[j] * _SPECIES_CHARGE[k]) ** 2
                * lnlambda
                * phixmgx
                / (4.0 * jnp.pi * constants.EPSILON0**2 * _SPECIES_MASS[j] ** 2 * v**3)
            )
    return out


def calculate_normalized_collision_frequency(temperatures, nu, iota, rmajor):
    """Collision frequency normalised by the (relativistic) transit frequency.

    Ports `Neoclassics.neoclassics_calc_nu_star`. **Not harness-tested** -- see module
    docstring.

    Parameters
    ----------
    temperatures :
        `(4,)` species temperatures (J). `.neoclassics.temperatures`.
    nu :
        `(4, NO_ROOTS)` collision frequency. `.neoclassics.nu`.
    iota :
        Rotational transform. `.neoclassics.iota` -- **read but never written anywhere
        in this file**; see `neoclassics.md`'s open questions.
    rmajor :
        Plasma major radius (m). `.physics.rmajor`.

    Returns
    -------
    :
        `(4, NO_ROOTS)` normalised collision frequency.
    """
    k = jnp.repeat(ROOTS[:, None], 4, axis=1)
    kk = (k * temperatures).T

    v = constants.SPEED_LIGHT * safe_sqrt(
        1.0 - (kk / (_SPECIES_MASS[:, None] * constants.SPEED_LIGHT**2) + 1.0) ** (-1)
    )
    return rmajor * nu / (iota * v)


def calculate_normalized_collision_frequency_from_temperature(
    iota,
    temp_plasma_electron_vol_avg_kev,
    temp_plasma_ion_vol_avg_kev,
    nd_plasma_electrons_vol_avg,
    nd_plasma_fuel_ions_vol_avg,
    f_plasma_fuel_deuterium,
    nd_plasma_alphas_thermal_vol_avg,
    rmajor,
):
    """Collision frequency normalised by transit frequency, from volume-averaged T/n.

    Ports `Neoclassics.neoclassics_calc_nu_star_fromT`. **Not harness-tested** -- see
    module docstring (also: an internal `for` loop over species, not a solve).

    Parameters
    ----------
    iota :
        Rotational transform, passed as an argument here (unlike `calculate_
        normalized_collision_frequency`, where the same-named field is an unexplained
        implicit read -- see `neoclassics.md`).
    temp_plasma_electron_vol_avg_kev, temp_plasma_ion_vol_avg_kev :
        Volume-averaged electron/ion temperatures (keV).
    nd_plasma_electrons_vol_avg, nd_plasma_fuel_ions_vol_avg :
        Volume-averaged electron/fuel-ion densities (/m3).
    f_plasma_fuel_deuterium :
        Deuterium fraction of the fuel ions.
    nd_plasma_alphas_thermal_vol_avg :
        Volume-averaged thermal alpha density (/m3).
    rmajor :
        Plasma major radius (m).

    Returns
    -------
    :
        `(4,)` normalised collision frequency, per species.
    """
    temp = (
        jnp.array([
            temp_plasma_electron_vol_avg_kev,
            temp_plasma_ion_vol_avg_kev,
            temp_plasma_ion_vol_avg_kev,
            temp_plasma_ion_vol_avg_kev,
        ])
        * KEV
    )
    density = jnp.array([
        nd_plasma_electrons_vol_avg,
        nd_plasma_fuel_ions_vol_avg * f_plasma_fuel_deuterium,
        nd_plasma_fuel_ions_vol_avg * (1.0 - f_plasma_fuel_deuterium),
        nd_plasma_alphas_thermal_vol_avg,
    ])

    lnlambda = (
        32.2
        - 1.15 * jnp.log10(density[0])
        + 2.3 * jnp.log10(temp[0] / constants.ELECTRON_CHARGE)
    )

    out = jnp.zeros((4,))
    for j in range(4):
        v = safe_sqrt(2.0 * temp[j] / _SPECIES_MASS[j])
        for k in range(4):
            xk = (_SPECIES_MASS[k] / _SPECIES_MASS[j]) * (temp[j] / temp[k])
            # Source guards `exp(-xk)` with `if xk < 200.0 else 0.0` to avoid
            # underflow noise; `jnp.where` is the traced equivalent.
            expxk = jnp.where(xk < 200.0, jnp.exp(-xk), 0.0)
            t = 1.0 / (1.0 + 0.3275911 * safe_sqrt(xk))
            erfn = (
                1.0
                - t
                * (
                    0.254829592
                    + t
                    * (
                        -0.284496736
                        + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))
                    )
                )
                * expxk
            )
            phixmgx = (1.0 - 0.5 / xk) * erfn + expxk / safe_sqrt(jnp.pi * xk)
            out = out.at[j].add(
                density[k]
                * (_SPECIES_CHARGE[j] * _SPECIES_CHARGE[k]) ** 2
                * lnlambda
                * phixmgx
                / (4.0 * jnp.pi * constants.EPSILON0**2 * _SPECIES_MASS[j] ** 2 * v**4)
                * rmajor
                / iota
            )
    return out


def calculate_drift_velocity(temperatures, rmajor, b_plasma_toroidal_on_axis):
    """Radial drift velocity on the Gauss-Laguerre grid, per species.

    Ports `Neoclassics.neoclassics_calc_vd`. **Not harness-tested** -- see module
    docstring.

    Parameters
    ----------
    temperatures :
        `(4,)` species temperatures (J). `.neoclassics.temperatures`.
    rmajor :
        Plasma major radius (m). `.physics.rmajor`.
    b_plasma_toroidal_on_axis :
        Toroidal field on axis (T). `.physics.b_plasma_toroidal_on_axis`.

    Returns
    -------
    :
        `(4, NO_ROOTS)` drift velocity.
    """
    # Species 3 (alpha) alone has an extra factor of 2 in the denominator (charge 2).
    charge_factor = jnp.array([1.0, 1.0, 1.0, 2.0])
    return jnp.outer(temperatures, ROOTS) / (
        constants.ELECTRON_CHARGE
        * rmajor
        * b_plasma_toroidal_on_axis
        * charge_factor[:, None]
    )


def calculate_plateau_transport_coefficient(kt, vd, rmajor, iota):
    """Plateau-regime radial transport coefficient (`D11_star`).

    Ports `Neoclassics.neoclassics_calc_D11_plateau`. **Not harness-tested** -- see
    module docstring.

    Parameters
    ----------
    kt :
        `(4, NO_ROOTS)` energy grid (J). `.neoclassics.kt`.
    vd :
        `(4, NO_ROOTS)` drift velocity. `.neoclassics.vd`.
    rmajor :
        Plasma major radius (m). `.physics.rmajor`.
    iota :
        Rotational transform. `.neoclassics.iota` -- see `calculate_normalized_
        collision_frequency`'s docstring on this same unexplained implicit read.

    Returns
    -------
    :
        `(4, NO_ROOTS)` plateau transport coefficient.
    """
    v = constants.SPEED_LIGHT * safe_sqrt(
        1.0 - (kt / (_SPECIES_MASS[:, None] * constants.SPEED_LIGHT**2) + 1.0) ** (-1)
    )
    return jnp.pi / 4.0 * vd**2 * rmajor / iota / v


def calculate_monoenergetic_transport_coefficient(eps_eff, vd, nu):
    """Monoenergetic radial transport coefficient from effective epsilon.

    Ports `Neoclassics.neoclassics_calc_d11_mono`.

    Parameters
    ----------
    eps_eff :
        Effective helical ripple. `.stellarator_config.stella_config_epseff`.
    vd :
        `(4, NO_ROOTS)` drift velocity. `.neoclassics.vd`.
    nu :
        `(4, NO_ROOTS)` collision frequency. `.neoclassics.nu`.

    Returns
    -------
    :
        `(4, NO_ROOTS)` monoenergetic transport coefficient.
    """
    return 4.0 / (9.0 * jnp.pi) * (2.0 * eps_eff) ** 1.5 * vd**2 / nu


def calculate_integrated_radial_transport_coefficient(d11_mono, index):
    """Gauss-Laguerre-integrated radial transport coefficient (`d111`/`d112`/`d113`).

    Ports `Neoclassics.calc_integrated_radial_transport_coeffs`. `index` selects which
    of the three moments (1/2/3) is integrated -- a precondition/static argument (which
    physical quantity you get back), not a `VarPath`-backed value: the source calls this
    once per index (1, 2, 3) to build three distinct `data.neoclassics.*` fields, so
    `index` plays the same "static, not traced" role as `EcrhDensityLimit.i_plasma_pedestal`.

    Parameters
    ----------
    d11_mono :
        `(4, NO_ROOTS)` monoenergetic transport coefficient. `.neoclassics.d11_mono`.
    index :
        Which moment (1, 2, or 3).

    Returns
    -------
    :
        `(4,)` integrated transport coefficient.
    """
    return jnp.sum(
        2.0 / safe_sqrt(jnp.pi) * d11_mono * ROOTS ** (index - 0.5) * WEIGHTS, axis=1
    )


def calculate_gamma_flux(
    densities, temperatures, dr_densities, dr_temperatures, d111, d112, er
):
    """Neoclassical particle flux, per species.

    Ports `Neoclassics.neoclassics_calc_gamma_flux`. **Not harness-tested** -- see
    module docstring.

    Parameters
    ----------
    densities, temperatures, dr_densities, dr_temperatures :
        `(4,)` species profile values and radial derivatives. `.neoclassics.*`.
    d111, d112 :
        `(4,)` integrated transport coefficients. `.neoclassics.d111`/`d112`.
    er :
        Radial electric field. `.neoclassics.er` -- **read but never written anywhere
        in this file**; see `neoclassics.md`'s open questions.

    Returns
    -------
    :
        `(4,)` particle flux, per species.
    """
    z = jnp.array([-1.0, 1.0, 1.0, 2.0])
    return (
        -densities
        * d111
        * (
            (dr_densities / densities - z * er / temperatures)
            + (d112 / d111 - 1.5) * dr_temperatures / temperatures
        )
    )


def calculate_q_flux(
    densities, temperatures, dr_densities, dr_temperatures, d112, d113, er
):
    """Neoclassical energy flux, per species.

    Ports `Neoclassics.neoclassics_calc_q_flux`. **Not harness-tested** -- see module
    docstring.

    Parameters
    ----------
    densities, temperatures, dr_densities, dr_temperatures :
        `(4,)` species profile values and radial derivatives. `.neoclassics.*`.
    d112, d113 :
        `(4,)` integrated transport coefficients. `.neoclassics.d112`/`d113`.
    er :
        Radial electric field. `.neoclassics.er` -- see `calculate_gamma_flux`'s
        docstring on this same unexplained implicit read.

    Returns
    -------
    :
        `(4,)` energy flux, per species.
    """
    z = jnp.array([-1.0, 1.0, 1.0, 2.0])
    return (
        -densities
        * temperatures
        * d112
        * (
            (dr_densities / densities - z * er / temperatures)
            + (d113 / d112 - 1.5) * dr_temperatures / temperatures
        )
    )


class ProfileValues(ExplicitFunction):
    """cottax node: `calculate_profile_values`, unchanged, ports declared.

    Mints under `.neoclassics.*` -- the source stores this call's four outputs there
    (`init_neoclassics`), even though the `rho=0.6` argument used at that one call site
    is itself a literal, not read from `data` (see `neoclassics.md`).

    That literal is `rho`, below. It was previously bound as
    `FromExactly(neoclassics.r_eff)`, which was a **wrong answer, not a coverage
    gap**: `.neoclassics.r_eff` is declared `= 0.0` in
    `process/data_structure/neoclassics_variables.py:87` and PROCESS never assigns it
    anywhere -- the real argument is `init_neoclassics`'s local parameter `r_effin`,
    passed the literal `0.6` at `process/models/stellarator/neoclassics.py:290`. The
    port therefore evaluated every profile on axis instead of at mid-radius:
    `dr_densities` came out identically `-0.0` against PROCESS's `-6.1e19`. Found by
    `_audit/boundary_inputs_audit.md` §6.1 and invisible to the MDA harness until its
    §6.2 array-comparison hole was closed, because all four outputs are arrays.
    """

    rho: float = eqx.field(static=True, default=0.6)
    """Normalised radius the neoclassical profiles are evaluated at -- PROCESS's own
    literal at its one call site, hoisted to a graph-assembly-time fact.

    Static rather than an `FromExactly` because there is no field to read it from: it is a
    modelling choice about where to sample, and the only `DataStructure` field with the
    right name (`.neoclassics.r_eff`) is a permanently-zero placeholder. Same move as
    `ImpurityRadiationTotals.imp_indices`, and declared in
    `mda_harness.STATIC_KWARGS_WITHOUT_BACKING_FIELD` for the same reason.
    """

    densities = OutputInto(neoclassics)
    temperatures = OutputInto(neoclassics)
    dr_densities = OutputInto(neoclassics)
    dr_temperatures = OutputInto(neoclassics)

    def __call__(
        self,
        temp_plasma_electron_on_axis_kev=From(physics),
        temp_plasma_ion_on_axis_kev=From(physics),
        alphat=From(physics),
        nd_plasma_electron_on_axis=From(physics),
        f_plasma_fuel_deuterium=From(physics),
        nd_plasma_ions_on_axis=From(physics),
        nd_plasma_alphas_thermal_vol_avg=From(physics),
        alphan=From(physics),
        rminor=From(physics),
    ):
        return calculate_profile_values(
            self.rho,
            temp_plasma_electron_on_axis_kev,
            temp_plasma_ion_on_axis_kev,
            alphat,
            nd_plasma_electron_on_axis,
            f_plasma_fuel_deuterium,
            nd_plasma_ions_on_axis,
            nd_plasma_alphas_thermal_vol_avg,
            alphan,
            rminor,
        )


class EffectiveThermalDiffusivity(ExplicitFunction):
    """cottax node: `calculate_effective_thermal_diffusivity`, unchanged, ports declared.

    `.neoclassics.chi_process_e` is an invented `VarPath`: `st_calc_eff_chi`'s return
    value is a local in `calc_neoclassics` (`chi_PROCESS_e`), never stored to `data` --
    same situation as `EcrhDensityLimit`'s outputs, see that module's docstring.
    """

    chi_process_e = OutputInto(neoclassics)

    def __call__(
        self,
        vol_plasma=From(physics),
        f_st_rmajor=From(stellarator),
        radius_plasma_core_norm=From(impurity_radiation),
        rminor=From(physics),
        stella_config_rminor_ref=From(stellarator_config),
        a_plasma_surface=From(physics),
        f_p_alpha_plasma_deposited=From(physics),
        pden_alpha_total_mw=From(physics),
        pden_plasma_core_rad_mw=From(physics),
        nd_plasma_electron_on_axis=From(physics),
        temp_plasma_electron_on_axis_kev=From(physics),
        alphat=From(physics),
        alphan=From(physics),
    ):
        return calculate_effective_thermal_diffusivity(
            vol_plasma,
            f_st_rmajor,
            radius_plasma_core_norm,
            rminor,
            stella_config_rminor_ref,
            a_plasma_surface,
            f_p_alpha_plasma_deposited,
            pden_alpha_total_mw,
            pden_plasma_core_rad_mw,
            nd_plasma_electron_on_axis,
            temp_plasma_electron_on_axis_kev,
            alphat,
            alphan,
        )
