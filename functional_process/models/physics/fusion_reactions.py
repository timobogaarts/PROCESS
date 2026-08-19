"""Pure-functional port of `process/models/physics/fusion_reactions.py`.

Registry unit #19.

Audit record: `functional_process/models/physics/fusion_reactions.md`. Read it first,
especially "cottax node" for why `.deuterium_branching()` gets no node of its own
(its only externally-visible effect has no `VarPath` until `.set_physics_variables()`
runs) and "tier signal" for why `beam_fusion()`/`beam_reaction_rate_coefficient()` are
**not** ported: PROCESS's own `scipy.integrate.quad` answer there is bounded to ~1e-6
relative accuracy (measured, not assumed -- replacing it with fixed-order Gauss-Legendre
quadrature at up to 256 nodes plateaus at the same disagreement, the signature of the
integrand's own kinks rather than of quadrature error), four orders outside this
harness's tier-1 `rtol=1e-12` value bar, and it is not JAX-traceable as written
regardless.

Everything else in `beam_fusion`'s dependency chain -- everything the `quad` call does
not touch -- is ported below as plain functions with no cottax node, ready for whenever
that blocker is resolved.
"""

import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, Input, Output

from functional_process.models.physics.plasma_profiles import _simpson
from process.core import constants
from functional_process.models.safe_math import safe_pow, safe_sqrt

REACTION_CONSTANTS_DT = {
    "bg": 34.3827,
    "mrc2": 1.124656e6,
    "cc1": 1.17302e-9,
    "cc2": 1.51361e-2,
    "cc3": 7.51886e-2,
    "cc4": 4.60643e-3,
    "cc5": 1.35000e-2,
    "cc6": -1.06750e-4,
    "cc7": 1.36600e-5,
}

REACTION_CONSTANTS_DHE3 = {
    "bg": 68.7508,
    "mrc2": 1.124572e6,
    "cc1": 5.51036e-10,
    "cc2": 6.41918e-3,
    "cc3": -2.02896e-3,
    "cc4": -1.91080e-5,
    "cc5": 1.35776e-4,
    "cc6": 0.0,
    "cc7": 0.0,
}

REACTION_CONSTANTS_DD1 = {
    "bg": 31.3970,
    "mrc2": 0.937814e6,
    "cc1": 5.43360e-12,
    "cc2": 5.85778e-3,
    "cc3": 7.68222e-3,
    "cc4": 0.0,
    "cc5": -2.96400e-6,
    "cc6": 0.0,
    "cc7": 0.0,
}

REACTION_CONSTANTS_DD2 = {
    "bg": 31.3970,
    "mrc2": 0.937814e6,
    "cc1": 5.65718e-12,
    "cc2": 3.41267e-3,
    "cc3": 1.99167e-3,
    "cc4": 0.0,
    "cc5": 1.05060e-5,
    "cc6": 0.0,
    "cc7": 0.0,
}


def bosch_hale_reactivity(ion_temperature_profile, reaction_constants):
    """Bosch-Hale volumetric fusion reaction rate <sigma v> (m^3/s).

    Direct port of the module-level function of the same name -- already pure, no `self`
    access. `reaction_constants` is a plain mapping (one of the `REACTION_CONSTANTS_*`
    dicts above) rather than the source's `BoschHaleConstants` dataclass, so the four
    reaction functions below can pass the module-level dict straight through without an
    extra wrapping step; the field names are identical either way.

    Parameters
    ----------
    ion_temperature_profile :
        Ion temperature profile (keV), any shape.
    reaction_constants :
        One of `REACTION_CONSTANTS_DT`/`_DHE3`/`_DD1`/`_DD2`.

    Returns
    -------
    :
        <sigma v> (m^3/s), same shape as `ion_temperature_profile`.
    """
    t = ion_temperature_profile
    rc = reaction_constants

    theta1 = (t * (rc["cc2"] + t * (rc["cc4"] + t * rc["cc6"]))) / (
        1.0 + t * (rc["cc3"] + t * (rc["cc5"] + t * rc["cc7"]))
    )
    theta = t / (1.0 - theta1)

    xi = safe_pow((rc["bg"] ** 2) / (4.0 * theta), 1 / 3)

    sigmav = (
        1.0e-6
        * rc["cc1"]
        * theta
        * safe_sqrt(xi / (rc["mrc2"] * t**3))
        * jnp.exp(-3.0 * xi)
    )

    # Source: `sigmav[t_mask] = 0.0` in-place mask assignment. A traced port cannot
    # mutate in place, so this is the `jnp.where` equivalent.
    return jnp.where(t == 0.0, 0.0, sigmav)  # noqa: RUF069


def calculate_deuterium_branching_trit(ion_temperature):
    """Relative rate of tritium-producing D-D reactions to 3He-producing ones.

    Ports `.deuterium_branching()`. No cottax node of its own -- see the audit record's
    "cottax node" section: its only consumer within scope is `FusionRates`, which calls
    this internally rather than taking the result as a separate `Input`, since PROCESS's
    own `.f_dd_branching_trit` never has a `VarPath` until `.set_physics_variables()`
    writes it.

    Parameters
    ----------
    ion_temperature :
        Volume-averaged ion temperature (keV). `.physics.temp_plasma_ion_vol_avg_kev`.

    Returns
    -------
    :
        Branching ratio, dimensionless. Valid for 0.5 keV < T < 200 keV (source's note).
    """
    t = ion_temperature
    return (
        1.02934 - 8.3264e-3 * t + 1.7631e-4 * t**2 - 1.8201e-6 * t**3 + 6.9855e-9 * t**4
    ) / 2.0


def _reaction_rate_integral(
    profile_x,
    ion_temperature_profile,
    ne_profile_y,
    nd_plasma_electrons_vol_avg,
    sigv_profile,
):
    """`fusion_rate_integral`'s integrand, folded into the caller.

    Ports the arithmetic of `fusion_rate_integral` (source lines 666-718), taking the
    already-evaluated `bosch_hale_reactivity` array rather than recomputing it -- the
    source calls `bosch_hale_reactivity` a second time here with the identical
    `ion_temperature_profile` input already used for `fusrat_plasma_*_profile`; the audit
    record flags this as redundant computation, not a redundant *write* (nothing is
    written twice), and this port removes the duplicate call rather than carrying it
    forward.
    """
    density_profile_normalised = (1.0 / nd_plasma_electrons_vol_avg) * ne_profile_y
    return 2.0 * profile_x * sigv_profile * density_profile_normalised**2


def _dt_reaction(
    profile_x,
    te_profile_y,
    ne_profile_y,
    temp_plasma_ion_vol_avg_kev,
    temp_plasma_electron_vol_avg_kev,
    f_plasma_fuel_deuterium,
    f_plasma_fuel_tritium,
    nd_plasma_fuel_ions_vol_avg,
    nd_plasma_electrons_vol_avg,
):
    """D + T -> 4He + n. Ports `dt_reaction`."""
    ion_temperature_profile = (
        temp_plasma_ion_vol_avg_kev / temp_plasma_electron_vol_avg_kev
    ) * te_profile_y
    sigv_profile = bosch_hale_reactivity(ion_temperature_profile, REACTION_CONSTANTS_DT)

    fusrat_plasma_dt_profile = (
        sigv_profile
        * f_plasma_fuel_deuterium
        * f_plasma_fuel_tritium
        * (ne_profile_y * (nd_plasma_fuel_ions_vol_avg / nd_plasma_electrons_vol_avg))
        ** 2
    )

    sigmav = _simpson(
        _reaction_rate_integral(
            profile_x,
            ion_temperature_profile,
            ne_profile_y,
            nd_plasma_electrons_vol_avg,
            sigv_profile,
        ),
        profile_x,
    )

    reaction_energy = constants.D_T_ENERGY / 1.0e6
    fusion_power_density = (
        sigmav
        * reaction_energy
        * (f_plasma_fuel_deuterium * nd_plasma_fuel_ions_vol_avg)
        * (f_plasma_fuel_tritium * nd_plasma_fuel_ions_vol_avg)
    )

    alpha_power_density = (
        1.0 - constants.DT_NEUTRON_ENERGY_FRACTION
    ) * fusion_power_density
    pden_non_alpha_charged_mw = 0.0 * fusion_power_density
    neutron_power_density = constants.DT_NEUTRON_ENERGY_FRACTION * fusion_power_density
    fusion_rate_density = fusion_power_density / reaction_energy
    alpha_rate_density = fusion_rate_density
    proton_rate_density = 0.0 * fusion_power_density

    return (
        fusion_power_density,
        sigmav,
        fusrat_plasma_dt_profile,
        alpha_power_density,
        pden_non_alpha_charged_mw,
        neutron_power_density,
        fusion_rate_density,
        alpha_rate_density,
        proton_rate_density,
    )


def _dhe3_reaction(
    profile_x,
    te_profile_y,
    ne_profile_y,
    temp_plasma_ion_vol_avg_kev,
    temp_plasma_electron_vol_avg_kev,
    f_plasma_fuel_deuterium,
    f_plasma_fuel_helium3,
    nd_plasma_fuel_ions_vol_avg,
    nd_plasma_electrons_vol_avg,
):
    """D + 3He -> 4He + p. Ports `dhe3_reaction`."""
    ion_temperature_profile = (
        temp_plasma_ion_vol_avg_kev / temp_plasma_electron_vol_avg_kev
    ) * te_profile_y
    sigv_profile = bosch_hale_reactivity(
        ion_temperature_profile, REACTION_CONSTANTS_DHE3
    )

    fusrat_plasma_dhe3_profile = (
        sigv_profile
        * f_plasma_fuel_deuterium
        * f_plasma_fuel_helium3
        * (ne_profile_y * (nd_plasma_fuel_ions_vol_avg / nd_plasma_electrons_vol_avg))
        ** 2
    )

    sigmav = _simpson(
        _reaction_rate_integral(
            profile_x,
            ion_temperature_profile,
            ne_profile_y,
            nd_plasma_electrons_vol_avg,
            sigv_profile,
        ),
        profile_x,
    )

    reaction_energy = constants.D_HELIUM_ENERGY / 1.0e6
    fusion_power_density = (
        sigmav
        * reaction_energy
        * (f_plasma_fuel_deuterium * nd_plasma_fuel_ions_vol_avg)
        * (f_plasma_fuel_helium3 * nd_plasma_fuel_ions_vol_avg)
    )

    alpha_power_density = (
        1.0 - constants.DHELIUM_PROTON_ENERGY_FRACTION
    ) * fusion_power_density
    pden_non_alpha_charged_mw = (
        constants.DHELIUM_PROTON_ENERGY_FRACTION * fusion_power_density
    )
    neutron_power_density = 0.0 * fusion_power_density
    fusion_rate_density = fusion_power_density / reaction_energy
    alpha_rate_density = fusion_rate_density
    proton_rate_density = fusion_rate_density

    return (
        fusion_power_density,
        fusrat_plasma_dhe3_profile,
        alpha_power_density,
        pden_non_alpha_charged_mw,
        neutron_power_density,
        fusion_rate_density,
        alpha_rate_density,
        proton_rate_density,
    )


def _dd_helion_reaction(
    profile_x,
    te_profile_y,
    ne_profile_y,
    temp_plasma_ion_vol_avg_kev,
    temp_plasma_electron_vol_avg_kev,
    f_plasma_fuel_deuterium,
    nd_plasma_fuel_ions_vol_avg,
    nd_plasma_electrons_vol_avg,
    f_dd_branching_trit,
):
    """D + D -> 3He + n. Ports `dd_helion_reaction`."""
    ion_temperature_profile = (
        temp_plasma_ion_vol_avg_kev / temp_plasma_electron_vol_avg_kev
    ) * te_profile_y
    sigv_profile = bosch_hale_reactivity(ion_temperature_profile, REACTION_CONSTANTS_DD1)

    fusrat_plasma_dd_helion_profile = (
        sigv_profile
        * f_plasma_fuel_deuterium
        * f_plasma_fuel_deuterium
        * (ne_profile_y * (nd_plasma_fuel_ions_vol_avg / nd_plasma_electrons_vol_avg))
        ** 2
    )

    sigmav = _simpson(
        _reaction_rate_integral(
            profile_x,
            ion_temperature_profile,
            ne_profile_y,
            nd_plasma_electrons_vol_avg,
            sigv_profile,
        ),
        profile_x,
    )

    reaction_energy = constants.DD_HELIUM_ENERGY / 1.0e6
    fusion_power_density = (
        sigmav
        * reaction_energy
        * (1.0 - f_dd_branching_trit)
        * (f_plasma_fuel_deuterium * nd_plasma_fuel_ions_vol_avg)
        * (f_plasma_fuel_deuterium * nd_plasma_fuel_ions_vol_avg)
    )

    alpha_power_density = 0.0 * fusion_power_density
    pden_non_alpha_charged_mw = (
        1.0 - constants.DD_NEUTRON_ENERGY_FRACTION
    ) * fusion_power_density
    neutron_power_density = constants.DD_NEUTRON_ENERGY_FRACTION * fusion_power_density
    fusion_rate_density = fusion_power_density / reaction_energy
    alpha_rate_density = 0.0 * fusion_power_density
    proton_rate_density = 0.0 * fusion_power_density

    return (
        fusion_power_density,
        fusrat_plasma_dd_helion_profile,
        alpha_power_density,
        pden_non_alpha_charged_mw,
        neutron_power_density,
        fusion_rate_density,
        alpha_rate_density,
        proton_rate_density,
    )


def _dd_triton_reaction(
    profile_x,
    te_profile_y,
    ne_profile_y,
    temp_plasma_ion_vol_avg_kev,
    temp_plasma_electron_vol_avg_kev,
    f_plasma_fuel_deuterium,
    nd_plasma_fuel_ions_vol_avg,
    nd_plasma_electrons_vol_avg,
    f_dd_branching_trit,
):
    """D + D -> T + p. Ports `dd_triton_reaction`."""
    ion_temperature_profile = (
        temp_plasma_ion_vol_avg_kev / temp_plasma_electron_vol_avg_kev
    ) * te_profile_y
    sigv_profile = bosch_hale_reactivity(ion_temperature_profile, REACTION_CONSTANTS_DD2)

    fusrat_plasma_dd_triton_profile = (
        sigv_profile
        * f_plasma_fuel_deuterium
        * f_plasma_fuel_deuterium
        * (ne_profile_y * (nd_plasma_fuel_ions_vol_avg / nd_plasma_electrons_vol_avg))
        ** 2
    )

    sigmav = _simpson(
        _reaction_rate_integral(
            profile_x,
            ion_temperature_profile,
            ne_profile_y,
            nd_plasma_electrons_vol_avg,
            sigv_profile,
        ),
        profile_x,
    )

    reaction_energy = constants.DD_TRITON_ENERGY / 1.0e6
    fusion_power_density = (
        sigmav
        * reaction_energy
        * f_dd_branching_trit
        * (f_plasma_fuel_deuterium * nd_plasma_fuel_ions_vol_avg)
        * (f_plasma_fuel_deuterium * nd_plasma_fuel_ions_vol_avg)
    )

    alpha_power_density = 0.0 * fusion_power_density
    pden_non_alpha_charged_mw = fusion_power_density
    neutron_power_density = 0.0 * fusion_power_density
    fusion_rate_density = fusion_power_density / reaction_energy
    alpha_rate_density = 0.0 * fusion_power_density
    proton_rate_density = fusion_rate_density

    return (
        fusion_power_density,
        fusrat_plasma_dd_triton_profile,
        alpha_power_density,
        pden_non_alpha_charged_mw,
        neutron_power_density,
        fusion_rate_density,
        alpha_rate_density,
        proton_rate_density,
    )


def calculate_fusion_rates(
    profile_x,
    te_profile_y,
    ne_profile_y,
    temp_plasma_ion_vol_avg_kev,
    temp_plasma_electron_vol_avg_kev,
    f_plasma_fuel_deuterium,
    f_plasma_fuel_tritium,
    f_plasma_fuel_helium3,
    nd_plasma_fuel_ions_vol_avg,
    nd_plasma_electrons_vol_avg,
    f_dd_branching_trit,
):
    """All four fusion reactions, summed. Ports `.calculate_fusion_rates()` **and**
    `.set_physics_variables()` -- see the audit record's "cottax node" section for why
    these two collapse into one pure function: `set_physics_variables()` is a straight
    copy of this function's own accumulated totals onto `data.physics.*`, the same
    redundant-internal-write pattern `density_limits.md` already established should
    collapse to a single return value rather than two write sites.

    Parameters
    ----------
    profile_x :
        Normalised radius grid (`n_plasma_profile_elements` points), shared by
        `teprofile`/`neprofile` -- see the audit record's minted-`VarPath` note for why
        one shared array is correct here.
    te_profile_y :
        Electron temperature profile (keV). `teprofile.profile_y`.
    ne_profile_y :
        Electron density profile (m^-3). `neprofile.profile_y`.
    temp_plasma_ion_vol_avg_kev, temp_plasma_electron_vol_avg_kev :
        Volume-averaged ion/electron temperatures (keV).
    f_plasma_fuel_deuterium, f_plasma_fuel_tritium, f_plasma_fuel_helium3 :
        Fuel species fractions.
    nd_plasma_fuel_ions_vol_avg, nd_plasma_electrons_vol_avg :
        Volume-averaged fuel-ion and electron densities (m^-3).
    f_dd_branching_trit :
        D-D tritium-branching ratio. `calculate_deuterium_branching_trit`'s output.

    Returns
    -------
    tuple
        `(pden_plasma_alpha_mw, pden_non_alpha_charged_mw, pden_plasma_neutron_mw,
        fusden_plasma, fusden_plasma_alpha, proton_rate_density, sigmav_dt_average,
        dt_power_density_plasma, dhe3_power_density, dd_power_density,
        f_dd_branching_trit, fusrat_plasma_dt_profile, fusrat_plasma_dhe3_profile,
        fusrat_plasma_dd_helion_profile, fusrat_plasma_dd_triton_profile)` -- the first
        eleven match `.set_physics_variables()`'s copy list exactly; the last four are
        written directly inside `.calculate_fusion_rates()` itself, not through
        `set_physics_variables` (see the audit record's data-footprint table).
    """
    dt_power, sigmav_dt_average, fusrat_dt, a1, c1, n1, r1, ar1, pr1 = _dt_reaction(
        profile_x,
        te_profile_y,
        ne_profile_y,
        temp_plasma_ion_vol_avg_kev,
        temp_plasma_electron_vol_avg_kev,
        f_plasma_fuel_deuterium,
        f_plasma_fuel_tritium,
        nd_plasma_fuel_ions_vol_avg,
        nd_plasma_electrons_vol_avg,
    )
    dhe3_power, fusrat_dhe3, a2, c2, n2, r2, ar2, pr2 = _dhe3_reaction(
        profile_x,
        te_profile_y,
        ne_profile_y,
        temp_plasma_ion_vol_avg_kev,
        temp_plasma_electron_vol_avg_kev,
        f_plasma_fuel_deuterium,
        f_plasma_fuel_helium3,
        nd_plasma_fuel_ions_vol_avg,
        nd_plasma_electrons_vol_avg,
    )
    dd_helion_power, fusrat_dd_helion, a3, c3, n3, r3, ar3, pr3 = _dd_helion_reaction(
        profile_x,
        te_profile_y,
        ne_profile_y,
        temp_plasma_ion_vol_avg_kev,
        temp_plasma_electron_vol_avg_kev,
        f_plasma_fuel_deuterium,
        nd_plasma_fuel_ions_vol_avg,
        nd_plasma_electrons_vol_avg,
        f_dd_branching_trit,
    )
    dd_triton_power, fusrat_dd_triton, a4, c4, n4, r4, ar4, pr4 = _dd_triton_reaction(
        profile_x,
        te_profile_y,
        ne_profile_y,
        temp_plasma_ion_vol_avg_kev,
        temp_plasma_electron_vol_avg_kev,
        f_plasma_fuel_deuterium,
        nd_plasma_fuel_ions_vol_avg,
        nd_plasma_electrons_vol_avg,
        f_dd_branching_trit,
    )

    return (
        a1 + a2 + a3 + a4,  # pden_plasma_alpha_mw
        c1 + c2 + c3 + c4,  # pden_non_alpha_charged_mw
        n1 + n2 + n3 + n4,  # pden_plasma_neutron_mw
        r1 + r2 + r3 + r4,  # fusden_plasma
        ar1 + ar2 + ar3 + ar4,  # fusden_plasma_alpha
        pr1 + pr2 + pr3 + pr4,  # proton_rate_density
        sigmav_dt_average,
        dt_power,  # dt_power_density_plasma
        dhe3_power,  # dhe3_power_density
        dd_helion_power + dd_triton_power,  # dd_power_density
        f_dd_branching_trit,
        fusrat_dt,
        fusrat_dhe3,
        fusrat_dd_helion,
        fusrat_dd_triton,
    )


def set_fusion_powers(
    f_alpha_electron,
    f_alpha_ion,
    p_beam_alpha_mw,
    pden_non_alpha_charged_mw,
    pden_plasma_neutron_mw,
    vol_plasma,
    pden_plasma_alpha_mw,
    f_p_alpha_plasma_deposited,
):
    """Fusion power metrics from plasma + beam contributions. Direct port of the
    module-level function of the same name -- already pure, unchanged in shape.

    Returns
    -------
    tuple
        `(pden_neutron_total_mw, p_plasma_alpha_mw, p_alpha_total_mw,
        p_plasma_neutron_mw, p_neutron_total_mw, p_non_alpha_charged_mw,
        pden_alpha_total_mw, f_pden_alpha_electron_mw, f_pden_alpha_ions_mw,
        p_charged_particle_mw, p_fusion_total_mw)`.
    """
    p_plasma_alpha_mw = pden_plasma_alpha_mw * vol_plasma
    pden_alpha_total_mw = pden_plasma_alpha_mw + (p_beam_alpha_mw / vol_plasma)
    p_alpha_total_mw = pden_alpha_total_mw * vol_plasma

    p_plasma_neutron_mw = pden_plasma_neutron_mw * vol_plasma
    pden_neutron_total_mw = pden_plasma_neutron_mw + (
        (
            (
                constants.DT_NEUTRON_ENERGY_FRACTION
                / (1.0 - constants.DT_NEUTRON_ENERGY_FRACTION)
            )
            * p_beam_alpha_mw
        )
        / vol_plasma
    )
    p_neutron_total_mw = pden_neutron_total_mw * vol_plasma

    p_non_alpha_charged_mw = pden_non_alpha_charged_mw * vol_plasma
    p_charged_particle_mw = p_alpha_total_mw + p_non_alpha_charged_mw
    p_fusion_total_mw = p_alpha_total_mw + p_neutron_total_mw + p_non_alpha_charged_mw

    f_pden_alpha_ions_mw = f_p_alpha_plasma_deposited * pden_alpha_total_mw * f_alpha_ion
    f_pden_alpha_electron_mw = (
        f_p_alpha_plasma_deposited * pden_alpha_total_mw * f_alpha_electron
    )

    return (
        pden_neutron_total_mw,
        p_plasma_alpha_mw,
        p_alpha_total_mw,
        p_plasma_neutron_mw,
        p_neutron_total_mw,
        p_non_alpha_charged_mw,
        pden_alpha_total_mw,
        f_pden_alpha_electron_mw,
        f_pden_alpha_ions_mw,
        p_charged_particle_mw,
        p_fusion_total_mw,
    )


# ---------------------------------------------------------------------------
# `beam_fusion`'s dependency chain, minus `beam_reaction_rate_coefficient` -- see the
# audit record's "tier signal"/"JAX-difficulty flags" for why that one function (and
# therefore `beam_fusion` itself) is not ported. Everything below is pure,
# self-contained, and verified against PROCESS's own values, but gets no cottax node:
# nothing in scope calls these directly (only `beam_fusion` does), so there is no
# `VarPath` boundary for a node to sit at until `beam_fusion` itself can be assembled.
# ---------------------------------------------------------------------------


def beam_fusion_cross_section(vrelsq):
    """Beam fusion cross-section (cm^2) from beam kinetic energy.

    Ports `_beam_fusion_cross_section`, renamed (no longer a private implementation
    detail of the one blocked function -- see module docstring). The three-way clamp
    (source: two early `return`s) becomes nested `jnp.where`, a `needs-lax-cond-or-where`
    case the audit record already flags as ordinary, not a blocker.

    Parameters
    ----------
    vrelsq :
        Square of the beam ion speed (keV/amu).

    Returns
    -------
    :
        Fusion reaction cross-section (cm^2).
    """
    a1 = 45.95
    a2 = 5.02e4
    a3 = 1.368e-2
    a4 = 1.076
    a5 = 4.09e2

    e_beam_kev = 0.5 * constants.M_DEUTERON_AMU * vrelsq

    t1 = a2 / (1.0 + (a3 * e_beam_kev - a4) ** 2) + a5
    t2 = e_beam_kev * (jnp.exp(a1 / safe_sqrt(e_beam_kev)) - 1.0)
    mid = 1.0e-24 * t1 / t2

    low = jnp.asarray(1.0e-27) + 0.0 * e_beam_kev
    high = jnp.asarray(8.0e-26) + 0.0 * e_beam_kev

    return jnp.where(e_beam_kev < 10.0, low, jnp.where(e_beam_kev > 1.0e4, high, mid))


def hot_beam_fusion_reaction_rate_integrand(velocity_ratio, critical_velocity):
    """Integrand for the hot-beam fusion rate coefficient.

    Ports `_hot_beam_fusion_reaction_rate_integrand`, renamed to match
    `beam_fusion_cross_section` above.

    Parameters
    ----------
    velocity_ratio :
        Beam velocity normalised to the critical velocity.
    critical_velocity :
        Critical velocity for electron/ion slowing down of the beam ion (m/s).

    Returns
    -------
    :
        Integrand value.
    """
    integral_term = (velocity_ratio**3) / (1.0 + velocity_ratio**3)
    beam_velocity = critical_velocity * velocity_ratio
    xvcs = beam_velocity**2 * constants.ATOMIC_MASS_UNIT / constants.KILOELECTRON_VOLT
    return integral_term * beam_fusion_cross_section(xvcs)


def fast_ion_pressure_integral(e_beam_kev, critical_energy):
    """Dimensionless fast-ion pressure integral. Direct port of the source function --
    already pure, unchanged in shape.

    Parameters
    ----------
    e_beam_kev :
        Beam birth energy (keV).
    critical_energy :
        Critical energy for electron/ion slowing down (keV).

    Returns
    -------
    :
        Dimensionless pressure integral factor.
    """
    xcs = e_beam_kev / critical_energy
    xc = safe_sqrt(xcs)

    t1 = xcs / 2.0
    t2 = jnp.log((xcs + 2.0 * xc + 1.0) / (xcs - xc + 1.0)) / 6.0

    xarg = (2.0 * xc - 1.0) / jnp.sqrt(3.0)
    t3 = jnp.arctan(xarg) / jnp.sqrt(3.0)
    t4 = (1 / jnp.sqrt(3.0)) * jnp.arctan(1 / jnp.sqrt(3.0))

    return t1 + t2 - t3 - t4


def beam_slowing_down_state(
    e_beam_kev,
    critical_energy_deuterium,
    critical_energy_tritium,
    t_beam_slow,
    f_beam_tritium,
    c_beam_total,
    vol_plasma,
):
    """Beam slowing-down state and hot-ion properties. Direct port of the source
    function -- already pure. Returns a plain tuple rather than the source's
    `BeamSlowingDownState` dataclass, since a JAX-traced function's return needs to be a
    pytree `_as_array` can flatten, not a hand-rolled frozen dataclass.

    The three `x > 0.0 else 0.0` conditionals (source: Python ternaries on
    `deuterium_beam_density`/`tritium_beam_density`/`nd_beam_hot`) become `jnp.where`,
    with the divisor guarded so the untaken branch cannot produce a NaN that would leak
    through the gradient (`test_outputs_finite`'s exact failure mode).

    Returns
    -------
    tuple
        `(deuterium_beam_density, tritium_beam_density, deuterium_critical_energy_speed,
        tritium_critical_energy_speed, nd_beam_hot, e_beam_deposited_kev)`.
    """
    beam_current_deuterium = c_beam_total * (1.0 - f_beam_tritium)
    beam_current_tritium = c_beam_total * f_beam_tritium

    beam_energy_ratio_deuterium = e_beam_kev / critical_energy_deuterium
    characteristic_deuterium_t_beam_slow = (
        t_beam_slow / 3.0 * jnp.log(1.0 + (beam_energy_ratio_deuterium) ** 1.5)
    )
    deuterium_beam_density = (
        beam_current_deuterium
        * characteristic_deuterium_t_beam_slow
        / (constants.ELECTRON_CHARGE * vol_plasma)
    )

    beam_energy_ratio_tritium = e_beam_kev / critical_energy_tritium
    characteristic_tritium_t_beam_slow = (
        t_beam_slow / 3.0 * jnp.log(1.0 + (beam_energy_ratio_tritium) ** 1.5)
    )
    tritium_beam_density = (
        beam_current_tritium
        * characteristic_tritium_t_beam_slow
        / (constants.ELECTRON_CHARGE * vol_plasma)
    )

    nd_beam_hot = deuterium_beam_density + tritium_beam_density

    deuterium_critical_energy_speed = safe_sqrt(2.0
        * constants.KILOELECTRON_VOLT
        * critical_energy_deuterium
        / (constants.ATOMIC_MASS_UNIT * constants.M_DEUTERON_AMU))
    tritium_critical_energy_speed = safe_sqrt(2.0
        * constants.KILOELECTRON_VOLT
        * critical_energy_tritium
        / (constants.ATOMIC_MASS_UNIT * constants.M_TRITON_AMU))

    source_deuterium = beam_current_deuterium / (constants.ELECTRON_CHARGE * vol_plasma)
    source_tritium = beam_current_tritium / (constants.ELECTRON_CHARGE * vol_plasma)

    pressure_coeff_deuterium = (
        constants.M_DEUTERON_AMU
        * constants.ATOMIC_MASS_UNIT
        * t_beam_slow
        * deuterium_critical_energy_speed**2
        * source_deuterium
        / (constants.KILOELECTRON_VOLT * 3.0)
    )
    pressure_coeff_tritium = (
        constants.M_TRITON_AMU
        * constants.ATOMIC_MASS_UNIT
        * t_beam_slow
        * tritium_critical_energy_speed**2
        * source_tritium
        / (constants.KILOELECTRON_VOLT * 3.0)
    )

    deuterium_pressure = pressure_coeff_deuterium * fast_ion_pressure_integral(
        e_beam_kev, critical_energy_deuterium
    )
    tritium_pressure = pressure_coeff_tritium * fast_ion_pressure_integral(
        e_beam_kev, critical_energy_tritium
    )

    deuterium_positive = deuterium_beam_density > 0.0
    deuterium_deposited_energy = jnp.where(
        deuterium_positive,
        1.5
        * deuterium_pressure
        / jnp.where(deuterium_positive, deuterium_beam_density, 1.0),
        0.0,
    )
    tritium_positive = tritium_beam_density > 0.0
    tritium_deposited_energy = jnp.where(
        tritium_positive,
        1.5 * tritium_pressure / jnp.where(tritium_positive, tritium_beam_density, 1.0),
        0.0,
    )

    hot_positive = nd_beam_hot > 0.0
    e_beam_deposited_kev = jnp.where(
        hot_positive,
        (
            deuterium_beam_density * deuterium_deposited_energy
            + tritium_beam_density * tritium_deposited_energy
        )
        / jnp.where(hot_positive, nd_beam_hot, 1.0),
        0.0,
    )

    return (
        deuterium_beam_density,
        tritium_beam_density,
        deuterium_critical_energy_speed,
        tritium_critical_energy_speed,
        nd_beam_hot,
        e_beam_deposited_kev,
    )


def beam_target_reaction_rate(nd_beam_ion, nd_target_ion, sigv_beam, vol_plasma):
    """Total beam-target fusion reaction rate (s^-1). Direct port -- already pure."""
    return nd_beam_ion * nd_target_ion * sigv_beam * vol_plasma


def alpha_power_beam(beam_target_reaction_rate_value):
    """Alpha power from beam-target fusion (MW). Direct port -- already pure."""
    return beam_target_reaction_rate_value * constants.DT_ALPHA_ENERGY / 1.0e6


class FusionRates(ExplicitFunction):
    """cottax node: `calculate_fusion_rates`, fusing all three in-scope
    `FusionReactionRate` methods (`.deuterium_branching()`, `.calculate_fusion_rates()`,
    `.set_physics_variables()`) -- see the audit record's "cottax node" section for why.

    **Two reused minted `VarPath`s**: `temp_plasma_electron_profile_kev`/
    `nd_plasma_electron_profile` are the same array objects
    `functional_process.models.physics.plasma_profiles.ProfileFactors` already minted
    those names for (`teprofile.profile_y`/`neprofile.profile_y` off the same
    `PlasmaProfile` instance) -- not a new mint.

    **A third reused minted `VarPath`, not a new mint.** This class's own earlier
    draft minted a fresh `.physics.profile_x` here on the reasoning that neither
    `teprofile.profile_x` nor `neprofile.profile_x` had an existing `VarPath` -- true
    at the time, but `profiles.py`'s `ProfileGrid` (a source node, no inputs) already
    mints exactly this grid as `.physics.radius_plasma_profile_norm`, and
    `radiation_power.py`'s own node already reads it under that name. The two are the
    same array (`np.arange(n_plasma_profile_elements)`, normalised by
    `Profile.normalise_profile_x()`, verified in `plasma_profiles.py`'s own test
    stub) -- confirmed directly, not assumed, by the block-by-block MDA-vs-PROCESS
    comparison harness surfacing `.physics.profile_x` as an ungrounded boundary input
    duplicating an already-real one. Fixed by reading `radius_plasma_profile_norm`
    here too, same as `radiation_power.py` -- three consumers of one mint now, not
    two plus a stray duplicate. See the audit record's data-footprint table.
    """

    pden_plasma_alpha_mw = Output(lambda s: s.physics.pden_plasma_alpha_mw)
    pden_non_alpha_charged_mw = Output(lambda s: s.physics.pden_non_alpha_charged_mw)
    pden_plasma_neutron_mw = Output(lambda s: s.physics.pden_plasma_neutron_mw)
    fusden_plasma = Output(lambda s: s.physics.fusden_plasma)
    fusden_plasma_alpha = Output(lambda s: s.physics.fusden_plasma_alpha)
    proton_rate_density = Output(lambda s: s.physics.proton_rate_density)
    sigmav_dt_average = Output(lambda s: s.physics.sigmav_dt_average)
    dt_power_density_plasma = Output(lambda s: s.physics.dt_power_density_plasma)
    dhe3_power_density = Output(lambda s: s.physics.dhe3_power_density)
    dd_power_density = Output(lambda s: s.physics.dd_power_density)
    f_dd_branching_trit = Output(lambda s: s.physics.f_dd_branching_trit)
    fusrat_plasma_dt_profile = Output(lambda s: s.physics.fusrat_plasma_dt_profile)
    fusrat_plasma_dhe3_profile = Output(lambda s: s.physics.fusrat_plasma_dhe3_profile)
    fusrat_plasma_dd_helion_profile = Output(
        lambda s: s.physics.fusrat_plasma_dd_helion_profile
    )
    fusrat_plasma_dd_triton_profile = Output(
        lambda s: s.physics.fusrat_plasma_dd_triton_profile
    )

    def __call__(
        self,
        profile_x=Input(lambda s: s.physics.radius_plasma_profile_norm),
        te_profile_y=Input(lambda s: s.physics.temp_plasma_electron_profile_kev),
        ne_profile_y=Input(lambda s: s.physics.nd_plasma_electron_profile),
        temp_plasma_ion_vol_avg_kev=Input(
            lambda s: s.physics.temp_plasma_ion_vol_avg_kev
        ),
        temp_plasma_electron_vol_avg_kev=Input(
            lambda s: s.physics.temp_plasma_electron_vol_avg_kev
        ),
        f_plasma_fuel_deuterium=Input(lambda s: s.physics.f_plasma_fuel_deuterium),
        f_plasma_fuel_tritium=Input(lambda s: s.physics.f_plasma_fuel_tritium),
        f_plasma_fuel_helium3=Input(lambda s: s.physics.f_plasma_fuel_helium3),
        nd_plasma_fuel_ions_vol_avg=Input(
            lambda s: s.physics.nd_plasma_fuel_ions_vol_avg
        ),
        nd_plasma_electrons_vol_avg=Input(
            lambda s: s.physics.nd_plasma_electrons_vol_avg
        ),
    ):
        f_dd_branching_trit = calculate_deuterium_branching_trit(
            temp_plasma_ion_vol_avg_kev
        )
        return calculate_fusion_rates(
            profile_x,
            te_profile_y,
            ne_profile_y,
            temp_plasma_ion_vol_avg_kev,
            temp_plasma_electron_vol_avg_kev,
            f_plasma_fuel_deuterium,
            f_plasma_fuel_tritium,
            f_plasma_fuel_helium3,
            nd_plasma_fuel_ions_vol_avg,
            nd_plasma_electrons_vol_avg,
            f_dd_branching_trit,
        )


class SetFusionPowers(ExplicitFunction):
    """cottax node: `set_fusion_powers`, unchanged, ports declared.

    `.physics.p_beam_alpha_mw` currently has no producer node in the graph -- it is
    written by `beam_fusion`, which is not ported (see module docstring). See the audit
    record's data-footprint table; not a reason to withhold this node, which only needs a
    value to arrive at that `VarPath`, not a specific producer.
    """

    pden_neutron_total_mw = Output(lambda s: s.physics.pden_neutron_total_mw)
    p_plasma_alpha_mw = Output(lambda s: s.physics.p_plasma_alpha_mw)
    p_alpha_total_mw = Output(lambda s: s.physics.p_alpha_total_mw)
    p_plasma_neutron_mw = Output(lambda s: s.physics.p_plasma_neutron_mw)
    p_neutron_total_mw = Output(lambda s: s.physics.p_neutron_total_mw)
    p_non_alpha_charged_mw = Output(lambda s: s.physics.p_non_alpha_charged_mw)
    pden_alpha_total_mw = Output(lambda s: s.physics.pden_alpha_total_mw)
    f_pden_alpha_electron_mw = Output(lambda s: s.physics.f_pden_alpha_electron_mw)
    f_pden_alpha_ions_mw = Output(lambda s: s.physics.f_pden_alpha_ions_mw)
    p_charged_particle_mw = Output(lambda s: s.physics.p_charged_particle_mw)
    p_fusion_total_mw = Output(lambda s: s.physics.p_fusion_total_mw)

    def __call__(
        self,
        f_alpha_electron=Input(lambda s: s.physics.f_alpha_electron),
        f_alpha_ion=Input(lambda s: s.physics.f_alpha_ion),
        p_beam_alpha_mw=Input(lambda s: s.physics.p_beam_alpha_mw),
        pden_non_alpha_charged_mw=Input(lambda s: s.physics.pden_non_alpha_charged_mw),
        pden_plasma_neutron_mw=Input(lambda s: s.physics.pden_plasma_neutron_mw),
        vol_plasma=Input(lambda s: s.physics.vol_plasma),
        pden_plasma_alpha_mw=Input(lambda s: s.physics.pden_plasma_alpha_mw),
        f_p_alpha_plasma_deposited=Input(lambda s: s.physics.f_p_alpha_plasma_deposited),
    ):
        return set_fusion_powers(
            f_alpha_electron,
            f_alpha_ion,
            p_beam_alpha_mw,
            pden_non_alpha_charged_mw,
            pden_plasma_neutron_mw,
            vol_plasma,
            pden_plasma_alpha_mw,
            f_p_alpha_plasma_deposited,
        )
