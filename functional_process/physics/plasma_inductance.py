"""Pure physics functions extracted from `models/physics/plasma_inductance.py`.

See that file for the declarations wiring these into the graph, and
`_audit/units/models/physics/plasma_inductance.md` for the audit record. No
graph-framework import belongs in this module -- see `_audit/formulas_split.md`.
"""

import jax.numpy as jnp

from functional_process.vocabulary import constants

RMU0 = constants.RMU0
"""Vacuum permeability (H/m), `process/core/constants.py:277`."""


def calculate_internal_inductance_wesson(alphaj):
    """Wesson's normalised internal inductance from the current profile index.

    Ports `PlasmaInductance.calculate_internal_inductance_wesson`,
    `process/models/physics/physics.py:4978-5005`, unchanged (`np.log` -> `jnp.log`).

    Parameters
    ----------
    alphaj :
        Current profile index. `.physics.alphaj`.

    Returns
    -------
    :
        `l_i`, the plasma normalised internal inductance.
    """
    return jnp.log(1.65 + 0.89 * alphaj)


def calculate_internal_inductance_menard(kappa):
    """Menard's normalised internal inductance for a spherical tokamak.

    Ports `PlasmaInductance.calculate_internal_inductance_menard`,
    `process/models/physics/physics.py:4949-4975`, unchanged. Fitted to NSTX data over
    `l_i` in 0.4-0.85 and recommended only for `kappa > 2.5`; the reference run's
    `kappa = 1.85` is outside that, which is one reason it selects Wesson instead.

    Parameters
    ----------
    kappa :
        Plasma separatrix elongation. `.physics.kappa`.

    Returns
    -------
    :
        `l_i`.
    """
    return 3.4 - kappa


def calculate_normalised_internal_inductance_iter_3(
    b_plasma_poloidal_vol_avg, c_plasma, vol_plasma, rmajor
):
    """The `l_i(3)` normalised internal inductance.

    Ports `PlasmaInductance.calculate_normalised_internal_inductance_iter_3`,
    `process/models/physics/physics.py:4902-4945`, unchanged.

    Parameters
    ----------
    b_plasma_poloidal_vol_avg :
        Volume-averaged poloidal field (T). PROCESS passes
        `.physics.b_plasma_surface_poloidal_average` here (`physics.py:4732`), whose name
        says *surface* -- the argument name and the field name disagree, and the field is
        what is read. Recorded in `plasma_inductance.md`, not corrected.
    c_plasma :
        Plasma current (A). `.physics.plasma_current`.
    vol_plasma :
        Plasma volume (m^3). `.physics.vol_plasma`.
    rmajor :
        Plasma major radius (m). `.physics.rmajor`.

    Returns
    -------
    :
        `l_i(3)`.
    """
    return (
        2 * vol_plasma * b_plasma_poloidal_vol_avg**2 / (RMU0**2 * c_plasma**2 * rmajor)
    )


def calculate_volt_second_requirements(
    csawth,
    eps,
    f_c_plasma_inductive,
    ejima_coeff,
    kappa,
    rmajor,
    res_plasma,
    plasma_current,
    t_plant_pulse_fusion_ramp,
    t_plant_pulse_burn,
    ind_plasma_internal_norm,
):
    """Volt-seconds the plasma needs to ramp up and to burn, and its total inductance.

    Ports `PlasmaInductance.calculate_volt_second_requirements`,
    `process/models/physics/physics.py:4766-4900`, arithmetic unchanged (`np.sqrt` and
    `np.log` -> `jnp`). No switch, no iteration, no branch of any kind.

    The source's docstring says it returns six values; it returns eight. Left as is --
    the tuple below is in the source's own return order, and the record notes the
    discrepancy.

    PROCESS's own comment at `:4882-4884` warns that `t_plant_pulse_burn` is wrong on the
    first iteration of a pulsed run and right on subsequent ones. That is the pulse
    timing loop, not this function: `t_plant_pulse_burn` is an ordinary declared input
    here and nothing in this module is loop-carried. See `plasma_inductance.md`
    § "Statefulness".

    Parameters
    ----------
    csawth :
        Enhancement factor on the flat-top volt-second requirement, for sawtooth
        effects. `.physics.csawth`.
    eps :
        Inverse aspect ratio. `.physics.eps`.
    f_c_plasma_inductive :
        Fraction of the plasma current driven inductively.
        `.physics.f_c_plasma_inductive`.
    ejima_coeff :
        Ejima coefficient for the resistive start-up volt-second component.
        `.physics.ejima_coeff`.
    kappa :
        Plasma elongation. `.physics.kappa`.
    rmajor :
        Plasma major radius (m). `.physics.rmajor`.
    res_plasma :
        Plasma resistance (ohm). `.physics.res_plasma`.
    plasma_current :
        Plasma current (A). `.physics.plasma_current`.
    t_plant_pulse_fusion_ramp, t_plant_pulse_burn :
        Fusion ramp and burn durations (s). `.times.*`.
    ind_plasma_internal_norm :
        Plasma normalised internal inductance. `.physics.ind_plasma_internal_norm`.

    Returns
    -------
    tuple
        `(vs_plasma_internal, ind_plasma_total, vs_plasma_burn_required,
        vs_plasma_ramp_required, vs_self_ind_ramp, vs_res_ramp,
        vs_plasma_total_required, v_plasma_loop_burn)` -- Wb, H, Wb, Wb, Wb, Wb, Wb, V.
        The call site (`physics.py:929-950`) binds the second to `.physics.ind_plasma`,
        the fifth to `.physics.vs_plasma_ind_ramp` and the sixth to
        `.physics.vs_plasma_res_ramp`.
    """
    # Plasma internal inductance and its flux.
    ind_plasma_internal = RMU0 * rmajor * ind_plasma_internal_norm / 2.0
    vs_plasma_internal = ind_plasma_internal * plasma_current

    # Start-up resistive component -- ITER formula without the 10 V-s add-on.
    vs_res_ramp = ejima_coeff * RMU0 * plasma_current * rmajor

    # Hirshman and Neilson fit for the external inductance.
    aeps = (1.0 + 1.81 * jnp.sqrt(eps) + 2.05 * eps) * jnp.log(8.0 / eps) - (
        2.0 + 9.25 * jnp.sqrt(eps) - 1.21 * eps
    )
    beps = 0.73 * jnp.sqrt(eps) * (1.0 + 2.0 * eps**4 - 6.0 * eps**5 + 3.7 * eps**6)
    ind_plasma_external = rmajor * RMU0 * aeps * (1.0 - eps) / (1.0 - eps + beps * kappa)

    ind_plasma_total = ind_plasma_external + ind_plasma_internal

    vs_self_ind_ramp = ind_plasma_total * plasma_current
    vs_plasma_ramp_required = vs_res_ramp + vs_self_ind_ramp

    # Flat-top loop voltage, enhanced for MHD sawtooth effects.
    v_plasma_loop_burn = plasma_current * res_plasma * f_c_plasma_inductive
    v_burn_resistive = v_plasma_loop_burn * csawth

    vs_plasma_burn_required = v_burn_resistive * (
        t_plant_pulse_fusion_ramp + t_plant_pulse_burn
    )
    vs_plasma_total_required = vs_plasma_ramp_required + vs_plasma_burn_required

    return (
        vs_plasma_internal,
        ind_plasma_total,
        vs_plasma_burn_required,
        vs_plasma_ramp_required,
        vs_self_ind_ramp,
        vs_res_ramp,
        vs_plasma_total_required,
        v_plasma_loop_burn,
    )


def internal_inductance_norm_scalings(
    alphaj,
    kappa,
    b_plasma_surface_poloidal_average,
    plasma_current,
    vol_plasma,
    rmajor,
):
    """All three normalised-internal-inductance scalings, evaluated unconditionally."""
    return (
        calculate_internal_inductance_wesson(alphaj),
        calculate_internal_inductance_menard(kappa),
        calculate_normalised_internal_inductance_iter_3(
            b_plasma_poloidal_vol_avg=b_plasma_surface_poloidal_average,
            c_plasma=plasma_current,
            vol_plasma=vol_plasma,
            rmajor=rmajor,
        ),
    )


def internal_inductance_norm_wesson(ind_plasma_internal_norm_wesson):
    """The Wesson scaling's value, unchanged -- this occupant's whole job."""
    return ind_plasma_internal_norm_wesson
