"""Pure physics functions extracted from `models/physics/physics.py`.

See that file for the declarations wiring these into the graph, and
`_audit/units/models/physics/physics.md` for the audit record. No
graph-framework import belongs in this module -- see `_audit/formulas_split.md`.
"""

import jax.numpy as jnp

from functional_process.vocabulary import constants


def calculate_surface_averaged_poloidal_field_amperes(cur_plasma, len_plasma_poloidal):
    """Surface-averaged poloidal field from Ampere's law, <Bp(a)> [T].

    Ports `PlasmaFields.calculate_surface_averaged_poloidal_field`'s
    `i_plasma_current != PENG_DIVERTOR_SCALING` arm, `plasma_fields.py:83-84`, unchanged.

    The whole arm is one line, and the point of declaring it separately is what it does
    **not** read: the source's signature takes eight arguments and this branch uses two
    of them. `q95`, `aspect`, `b_plasma_toroidal_on_axis`, `kappa` and `triang` are
    read only by the `PENG_DIVERTOR_SCALING` arm (`plasma_fields.py:86-93`), so a node
    declaring the union would claim five edges this configuration does not have.

    Parameters
    ----------
    cur_plasma :
        Plasma current (A). `.physics.plasma_current`.
    len_plasma_poloidal :
        Plasma poloidal perimeter (m). `.physics.len_plasma_poloidal`.

    Returns
    -------
    :
        Surface-averaged poloidal field (T).
    """
    return constants.RMU0 * cur_plasma / len_plasma_poloidal


def calculate_unclipped_radiation_powers(
    pden_plasma_core_rad_mw_unclipped,
    pden_plasma_outer_rad_mw_unclipped,
    vol_plasma,
):
    """The tokamak's core/outer radiation densities and their volume integrals.

    Ports `physics.py:751-752` (two bare assignments off `calculate_radiation_powers`'s
    `RadpwrData`) and `physics.py:758-763` (the two products).

    **This is the tokamak half of a divergence between `calculate_radiation_powers`'s
    two callers, and the reason `PlasmaRadiationPowers` mints `_unclipped` names.**
    `stellarator.py:2153-2158` clips both densities at zero before forming the products;
    `physics.py:751-752` does not clip at all. The clip is therefore a property of one
    caller, not of the radiation model, and this function is the other caller: it is
    `models/stellarator/plasma_physics.py::calculate_clipped_radiation_powers` with the
    two `max(..., 0.0)` removed and nothing else changed. Confirmed by reading
    `physics.py:750-766` in full -- there is no `max`, no `jnp.maximum`, and no guard of
    any kind between the assignment and the products.

    A negative `pden_plasma_core_rad_mw` therefore propagates on a tokamak where the
    stellarator would have floored it, including into `.physics.p_plasma_inner_rad_mw`
    and (through `confinement_time.py`'s `power_loss`) into `.physics.p_plasma_loss_mw`.
    Ported faithfully; flagged in the record as a suspected PROCESS defect (**D1**),
    not fixed.

    Parameters
    ----------
    pden_plasma_core_rad_mw_unclipped :
        Core radiation power density (MW/m^3), as `PlasmaRadiationPowers` produces it.
        `.physics.pden_plasma_core_rad_mw_unclipped`.
    pden_plasma_outer_rad_mw_unclipped :
        Edge radiation power density (MW/m^3).
        `.physics.pden_plasma_outer_rad_mw_unclipped`.
    vol_plasma :
        Plasma volume (m^3). `.physics.vol_plasma`.

    Returns
    -------
    :
        `(pden_plasma_core_rad_mw, pden_plasma_outer_rad_mw, p_plasma_inner_rad_mw,
        p_plasma_outer_rad_mw)`, the first two in MW/m^3 and the last two in MW.
    """
    pden_plasma_core_rad_mw = pden_plasma_core_rad_mw_unclipped
    pden_plasma_outer_rad_mw = pden_plasma_outer_rad_mw_unclipped
    return (
        pden_plasma_core_rad_mw,
        pden_plasma_outer_rad_mw,
        pden_plasma_core_rad_mw * vol_plasma,
        pden_plasma_outer_rad_mw * vol_plasma,
    )


def calculate_total_radiation_power(pden_plasma_rad_mw, vol_plasma):
    """Total radiated power from the plasma (MW). Ports `physics.py:764-766`.

    A separate node from `calculate_unclipped_radiation_powers` above even though
    PROCESS writes all four products in one straight-line block, because its input is a
    different variable: `.physics.pden_plasma_rad_mw` is a **real** `DataStructure`
    field that `PlasmaRadiationPowers` owns directly (PROCESS never clips it -- see that
    node's docstring), where the core/outer pair arrive through the two `_unclipped`
    mints. Bundling them would give every consumer of `p_plasma_rad_mw` an edge from the
    mints it does not depend on.

    Parameters
    ----------
    pden_plasma_rad_mw :
        Total radiation power density (MW/m^3). `.physics.pden_plasma_rad_mw`.
    vol_plasma :
        Plasma volume (m^3). `.physics.vol_plasma`.

    Returns
    -------
    :
        Total radiated power (MW).
    """
    return pden_plasma_rad_mw * vol_plasma


def calculate_separatrix_power(
    f_p_alpha_plasma_deposited,
    p_alpha_total_mw,
    p_non_alpha_charged_mw,
    p_hcd_injected_total_mw,
    p_plasma_ohmic_mw,
    p_plasma_rad_mw,
):
    """Power crossing the separatrix, before the positivity transform (MW).

    Ports `PlasmaExhaust.calculate_separatrix_power`,
    `process/models/physics/exhaust.py:88-127`, unchanged. Out of `physics.py`'s own
    file scope and out of `exhaust.py`'s ported scope (`exhaust.md` records the other
    three `PlasmaExhaust` statics as deliberately not ported there), but it is the sole
    producer of one of this slot's eight outputs, so it is ported here. If `exhaust.py`'s
    scope is ever widened to cover it, one of the two copies must go -- there is no
    reason for both.

    Parameters
    ----------
    f_p_alpha_plasma_deposited :
        Fraction of alpha power deposited in the plasma.
        `.physics.f_p_alpha_plasma_deposited`.
    p_alpha_total_mw :
        Total alpha power (MW). `.physics.p_alpha_total_mw`.
    p_non_alpha_charged_mw :
        Non-alpha charged-particle power (MW). `.physics.p_non_alpha_charged_mw`.
    p_hcd_injected_total_mw :
        Injected heating and current-drive power (MW).
        `.current_drive.p_hcd_injected_total_mw`, or `0.0` on the ignited arm.
    p_plasma_ohmic_mw :
        Ohmic heating power (MW). `.physics.p_plasma_ohmic_mw`.
    p_plasma_rad_mw :
        Total radiated power (MW). `.physics.p_plasma_rad_mw`.

    Returns
    -------
    :
        Power crossing the separatrix (MW), which may be negative.
    """
    return (
        f_p_alpha_plasma_deposited * p_alpha_total_mw
        + p_non_alpha_charged_mw
        + p_hcd_injected_total_mw
        + p_plasma_ohmic_mw
        - p_plasma_rad_mw
    )


def force_positive_separatrix_power(p_plasma_separatrix_mw_raw):
    """PROCESS's positivity transform on the separatrix power (MW).

    Ports `physics.py:839-845` verbatim -- the source's own label is *"KLUDGE: Ensure
    p_plasma_separatrix_mw is continuously positive (physical, rather than negative
    potential power), as required by other models"*:

        p_plasma_separatrix_mw /= 1 - exp(-p_plasma_separatrix_mw)

    It is a smooth `softplus`-like map, not a clip: it is the identity to within
    `exp(-x)` for `x` of order a few, and maps negative `x` to a small positive number.

    **This is the second write to `.physics.p_plasma_separatrix_mw` in a single pass**,
    and it is why the pre-transform value is minted as
    `.physics.p_plasma_separatrix_mw_raw` rather than the two writes sharing one name.
    Three PROCESS call sites read the *pre*-transform value between the two writes --
    `calculate_psep_over_r_metric` (`physics.py:811-816`),
    `calculate_eu_demo_re_attachment_metric` (`:818-826`) and `ScrapeOffLayer.run`
    (`:832`) -- and every consumer after line 845 reads the post-transform one. A single
    node owning `.physics.p_plasma_separatrix_mw` and applying the transform inside would
    be correct for its own output and would silently hand the wrong one of the two values
    to those three. Same precedent, same shape, and the same reason as
    `radiation_power.py`'s `pden_plasma_core_rad_mw_unclipped` mint.

    Not a `jnp.where`-guarded domain: at exactly `x == 0` the source evaluates `0.0/0.0`
    and returns `nan` (a `RuntimeWarning`, not a raise), and the port reproduces that
    rather than inventing a limit PROCESS does not take.

    Parameters
    ----------
    p_plasma_separatrix_mw_raw :
        Separatrix power before the transform (MW).
        `.physics.p_plasma_separatrix_mw_raw`.

    Returns
    -------
    :
        Separatrix power after the transform (MW).
    """
    return p_plasma_separatrix_mw_raw / (1 - jnp.exp(-p_plasma_separatrix_mw_raw))


def calculate_pulsed_plant_ramp_times(plasma_current):
    """Plasma-current ramp-up and ramp-down times for a pulsed plant (s).

    Ports `physics.py:476-483`, the `i_pulsed_plant == 1 and pulsetimings == 0` arm:

        t_plant_pulse_plasma_current_ramp_up   = plasma_current / 1.0e5
        t_plant_pulse_plasma_current_ramp_down = t_plant_pulse_plasma_current_ramp_up

    `.times.t_plant_pulse_coil_precharge` is **not** written on this arm -- the source's
    own comment at `:477` says it is an input -- which is exactly what distinguishes it
    from the other two arms and why the switch is a split rather than a static kwarg.

    Parameters
    ----------
    plasma_current :
        Plasma current (A). `.physics.plasma_current`.

    Returns
    -------
    :
        `(t_plant_pulse_plasma_current_ramp_up,
        t_plant_pulse_plasma_current_ramp_down)`, both in seconds.
    """
    t_plant_pulse_plasma_current_ramp_up = plasma_current / 1.0e5
    return (
        t_plant_pulse_plasma_current_ramp_up,
        t_plant_pulse_plasma_current_ramp_up,
    )


def calculate_continuous_plant_ramp_times(plasma_current):
    """Plasma-current ramp times for a continuous (non-pulsed) plant (s).

    Ports `physics.py:465-474`, the `i_pulsed_plant != 1 and i_t_current_ramp_up == 0`
    arm, unchanged:

        t_plant_pulse_plasma_current_ramp_up   = plasma_current / 5.0e5
        t_plant_pulse_coil_precharge           = t_plant_pulse_plasma_current_ramp_up
        t_plant_pulse_plasma_current_ramp_down = t_plant_pulse_plasma_current_ramp_up

    Unlike the pulsed-default arm (`calculate_pulsed_plant_ramp_times`, `:476-483`),
    this arm *does* write `.times.t_plant_pulse_coil_precharge` -- the third output --
    which is exactly why the two are separate occupants rather than one function with a
    literal swapped (`5e5` vs `1e5`): the write-sets differ, not just a constant.

    Parameters
    ----------
    plasma_current :
        Plasma current (A). `.physics.plasma_current`.

    Returns
    -------
    :
        `(t_plant_pulse_plasma_current_ramp_up, t_plant_pulse_coil_precharge,
        t_plant_pulse_plasma_current_ramp_down)`, all in seconds, all equal --
        PROCESS's write order at `physics.py:466-474`.
    """
    t_plant_pulse_plasma_current_ramp_up = plasma_current / 5.0e5
    return (
        t_plant_pulse_plasma_current_ramp_up,
        t_plant_pulse_plasma_current_ramp_up,
        t_plant_pulse_plasma_current_ramp_up,
    )


def calculate_plasma_energy_from_beta(beta, b_field, vol_plasma):
    """Plasma stored energy derived from beta (J).

    Ports `PlasmaBeta.calculate_plasma_energy_from_beta`, `physics.py:4153-4176`,
    unchanged:

        E = 1.5 * beta * B^2 / (2 * mu_0) * V

    Parameters
    ----------
    beta :
        Plasma beta (dimensionless).
    b_field :
        Magnetic field (T).
    vol_plasma :
        Plasma volume (m^3). `.physics.vol_plasma`.

    Returns
    -------
    :
        Plasma energy (J).
    """
    return (1.5e0 * beta * b_field**2) / (2.0e0 * constants.RMU0) * vol_plasma


def plasma_ohmic_heating(
    f_c_plasma_inductive,
    kappa95,
    plasma_current,
    rmajor,
    rminor,
    temp_plasma_electron_density_weighted_kev,
    vol_plasma,
    zeff,
    plasma_res_factor,
):
    """Ohmic heating power and plasma resistance (IPDG89).

    Ports `Physics.plasma_ohmic_heating`, `physics.py:1605-1697`, term for term --
    **including its live defect**. PROCESS's neo-classical enhancement guard is the
    chained comparison `1.0 if 2.5 >= rmajor / rminor <= 4.0 else 4.3 - 0.6 * rmajor /
    rminor` (`physics.py:1675`), which Python reads as `(2.5 >= A) and (A <= 4.0)`,
    i.e. `A <= 2.5` -- **not** the documented "aspect ratios in the range 2.5 to 4.0".
    Reproduced exactly as `jnp.where(A <= 2.5, ...)`; on `large_tokamak_eval`
    (`A = 3.1`) the enhancement arm is taken either way.

    Two shells are dropped, neither of them arithmetic: the `aspect` parameter (read
    only by the negative-resistance `logger.error`, `physics.py:1682-1685` -- a traced
    function cannot log on a data-dependent condition, and the message's own value is
    unused), and that logger itself.

    Parameters
    ----------
    f_c_plasma_inductive :
        Fraction of plasma current driven inductively.
        `.physics.f_c_plasma_inductive`.
    kappa95 :
        Plasma elongation at the 95% surface. `.physics.kappa95`.
    plasma_current :
        Plasma current (A). `.physics.plasma_current`.
    rmajor, rminor :
        Plasma major/minor radius (m). `.physics.rmajor`, `.physics.rminor`.
    temp_plasma_electron_density_weighted_kev :
        Density-weighted average electron temperature (keV).
        `.physics.temp_plasma_electron_density_weighted_kev`.
    vol_plasma :
        Plasma volume (m^3). `.physics.vol_plasma`.
    zeff :
        Plasma effective charge (the staticmethod's own spelling; the storage field is
        `.physics.n_charge_plasma_effective_vol_avg`, `Physics.run` `:782`, and the
        node port uses that name per the declaration-surface rule).
    plasma_res_factor :
        Plasma resistivity pre-factor. `.physics.plasma_res_factor`.

    Returns
    -------
    tuple
        `(pden_plasma_ohmic_mw, p_plasma_ohmic_mw, f_res_plasma_neo, res_plasma)` --
        MW/m^3, MW, dimensionless, ohm; `.physics.` all four (`physics.py:768-773`).
    """
    t10 = temp_plasma_electron_density_weighted_kev / 10.0

    res_plasma = (
        plasma_res_factor * 2.15e-9 * zeff * rmajor / (kappa95 * rminor**2 * t10**1.5)
    )

    # PROCESS's chained comparison, reproduced -- see the docstring.
    f_res_plasma_neo = jnp.where(
        rmajor / rminor <= 2.5, 1.0, 4.3 - 0.6 * rmajor / rminor
    )

    res_plasma = res_plasma * f_res_plasma_neo

    pden_plasma_ohmic_mw = (
        f_c_plasma_inductive * plasma_current**2 * res_plasma * 1.0e-6 / vol_plasma
    )

    p_plasma_ohmic_mw = pden_plasma_ohmic_mw * vol_plasma

    return pden_plasma_ohmic_mw, p_plasma_ohmic_mw, f_res_plasma_neo, res_plasma


def calculate_coulomb_logarithm_ion_electron(
    nd_plasma_electrons_vol_avg, temp_plasma_electron_vol_avg_kev
):
    """Coulomb logarithm for ion-electron collisions.

    Ports `Physics.run`, `process/models/physics/physics.py:279-283`, unchanged:

        ln(Lambda)_ie = 31.3 - ln(n_e)/2 + ln(T_e [eV])

    **This is a tokamak-only producer, and that is not obvious from a grep.**
    `.physics.dlamie` is read on both devices -- `stellarator.py:2021` and `:2125` pass
    it into the stellarator's own physics -- but every one of those is a *read*. The
    only write in `process/` is this line, inside `Physics.run`, and
    `caller.py:272-275` returns from the stellarator arm before `caller.py:290` ever
    calls `Physics.run`. So a stellarator run reads a `dlamie` nothing computed: it is
    a genuine boundary input there (`reference_boundary.txt`) and a missing producer
    here, and the same `VarPath` is honestly both. Ported into `.tokamak.physics` for
    that reason and not into the shared `.physics` subsystem.

    **The electron-electron sibling `.physics.dlamee` (`:274-278`, identical but for
    `31.0`) is UNPORTED**: nothing in this graph reads it, so owning it would add an
    output with no consumer instead of closing a hole. Its one PROCESS reader is
    `current_drive.py:1720`, inside the `i_hcd_primary == 3` arm that `indat.py` refuses
    for a second reason as well (`ElectronCyclotron.electron_cyclotron_fenstermacher` is
    not written either) -- so this is two lines of work the day that arm is wanted, not
    a hole.

    Parameters
    ----------
    nd_plasma_electrons_vol_avg :
        Volume-averaged electron density (m^-3).
        `.physics.nd_plasma_electrons_vol_avg`.
    temp_plasma_electron_vol_avg_kev :
        Volume-averaged electron temperature (keV) -- converted to eV inside, as
        PROCESS does. `.physics.temp_plasma_electron_vol_avg_kev`.

    Returns
    -------
    :
        Coulomb logarithm, ion-electron (dimensionless). `.physics.dlamie`.
    """
    return (
        31.3
        - (jnp.log(nd_plasma_electrons_vol_avg) / 2.0)
        + jnp.log(temp_plasma_electron_vol_avg_kev * 1000.0)
    )


def calculate_pflux_plasma_surface_neutron_avg_mw(p_neutron_total_mw, a_plasma_surface):
    """Average neutron flux through the plasma surface (MW/m^2).

    Ports `Physics.run`, `process/models/physics/physics.py:835-837`, unchanged: the
    total neutron power divided by the plasma surface area.

    **`p_neutron_total_mw`, not `p_plasma_neutron_mw`.** PROCESS divides the *total*
    (`.physics.p_neutron_total_mw`, which `.physics.set_fusion_powers` owns and which
    includes the beam-target contribution), while the divertor's own split three lines
    away uses the plasma-only figure. Two different fields with confusable names, and
    the port takes the one the source line takes.

    A missing producer, ported 2026-08-30: `.tokamak.first_wall` reads this field --
    `calculate_pflux_fw_neutron_mw_ffwal` is `ffwal` times it and nothing else -- so with
    no owner the whole first-wall neutron flux was `0.0` against PROCESS's `0.71479842`
    on `large_tokamak_nof`.

    Parameters
    ----------
    p_neutron_total_mw :
        Total neutron power, plasma plus beam-target (MW).
        `.physics.p_neutron_total_mw`.
    a_plasma_surface :
        Plasma surface area (m^2). `.physics.a_plasma_surface`.

    Returns
    -------
    :
        Average neutron flux through the plasma surface (MW/m^2).
        `.physics.pflux_plasma_surface_neutron_avg_mw`.
    """
    return p_neutron_total_mw / a_plasma_surface


def calculate_beta_norm_max_wesson(ind_plasma_internal_norm):
    """Wesson's normalised beta upper limit, beta_N_max.

    Ports `PlasmaBeta.calculate_beta_norm_max_wesson`, `physics.py:3941-3974`,
    unchanged -- the whole body is `4 * l_i`.

    Parameters
    ----------
    ind_plasma_internal_norm :
        Plasma normalised internal inductance. `.physics.ind_plasma_internal_norm`.

    Returns
    -------
    :
        Wesson normalised beta upper limit.
    """
    return 4 * ind_plasma_internal_norm


def calculate_beta_limit_from_norm(
    b_plasma_toroidal_on_axis,
    beta_norm_max,
    plasma_current,
    rminor,
):
    """Maximum allowed volume-averaged beta, from the normalised limit.

    Ports `PlasmaBeta.calculate_beta_limit_from_norm`, `physics.py:4180-4235`,
    unchanged (AEA FUS 172). The `0.01` converts the Troyon coefficient from per-cent
    to a fraction.

    **This node owns `.physics.beta_vol_avg_max` and nothing selects among components
    here.** `.physics.i_beta_component` chooses which beta the *constraint* compares
    against the limit (`constraint_24`), not which limit is computed -- PROCESS
    computes exactly this one limit whatever the switch says, and the switch is already
    a static kwarg of the ported `constraint_24`.

    Parameters
    ----------
    b_plasma_toroidal_on_axis :
        Toroidal field on the plasma axis (T).
    beta_norm_max :
        Troyon-like g coefficient. `.physics.beta_norm_max`.
    plasma_current :
        Plasma current (A).
    rminor :
        Plasma minor radius (m).

    Returns
    -------
    :
        Volume-averaged beta limit (dimensionless).
    """
    return (
        0.01
        * beta_norm_max
        * (plasma_current / 1.0e6)
        / (rminor * b_plasma_toroidal_on_axis)
    )


def calculate_toroidal_beta(
    beta_total_vol_avg,
    b_plasma_total,
    b_plasma_toroidal_on_axis,
):
    """Volume-averaged beta referred to the toroidal field alone.

    Ports `physics.py:3818-3822` -- an inline assignment in `PlasmaBeta.run` with no
    `@staticmethod` of its own, transcribed term for term.

    Parameters
    ----------
    beta_total_vol_avg :
        Volume-averaged total beta, referred to the total field.
    b_plasma_total :
        Total field on axis (T).
    b_plasma_toroidal_on_axis :
        Toroidal field on axis (T).

    Returns
    -------
    :
        Toroidal beta (dimensionless).
    """
    return beta_total_vol_avg * b_plasma_total**2 / b_plasma_toroidal_on_axis**2


def calculate_poloidal_beta(b_plasma_total, b_plasma_poloidal_average, beta):
    """Volume-averaged beta referred to the poloidal field alone.

    Ports `Physics.calculate_poloidal_beta` (`physics.py:4239-4263`), called from
    `physics.py:3825` -- the one line of `PlasmaBeta.run`'s 3818-3835 block this slot
    skipped, sitting between `ToroidalBeta` (3818-3822) and `ThermalBeta` (3831-3835).

    **This was a known hole and it was load-bearing.** `constraint_48`'s docstring has
    recorded since `batch5.md` that "`beta_poloidal_vol_avg`'s real producer
    (`Physics.calculate_poloidal_beta`, `physics.py:3825`) is not yet ported anywhere in
    `functional_process`", and ported the constraint over the unproduced read anyway.
    The read is not only constraint 48's: `models/pfcoil/currents.py::
    calculate_equilibrium_currents` puts it inside `log(8*aspect) + beta_poloidal_vol_avg
    + l_i/2 - 1.5`, the bracket that sets the **equilibrium PF coil currents**. With no
    producer the term was `0.0` against PROCESS's `1.0874` on `large_tokamak_nof` -- an
    O(1) error in an O(1) bracket, propagating through the coil flux to the volt-second
    balance, the burn time (55x), the CS field and finally `stress_shear_cs_peak` (708x),
    which is constraint 72 and which is *active* at PROCESS's optimum. See
    `_audit/optimise_design.md` §16.

    References
    ----------
    - J.P. Freidberg, "Plasma physics and fusion energy", Cambridge University Press
      (2007) Page 270 ISBN 0521851076

    Parameters
    ----------
    b_plasma_total :
        Total field on axis (T).
    b_plasma_poloidal_average :
        Surface-averaged poloidal field (T).
    beta :
        Volume-averaged total beta, referred to the total field.

    Returns
    -------
    :
        Poloidal beta (dimensionless).
    """
    return beta * (b_plasma_total / b_plasma_poloidal_average) ** 2


def calculate_thermal_beta(beta_total_vol_avg, beta_fast_alpha, beta_beam):
    """Volume-averaged thermal beta: the total less both fast-ion contributions.

    Ports `physics.py:3831-3835`, an inline assignment in `PlasmaBeta.run`.

    Parameters
    ----------
    beta_total_vol_avg :
        Volume-averaged total beta.
    beta_fast_alpha :
        Fast-alpha beta contribution. `.physics.beta_fast_alpha`.
    beta_beam :
        Neutral-beam fast-ion beta contribution. `.physics.beta_beam`.

    Returns
    -------
    :
        Thermal beta (dimensionless).
    """
    return beta_total_vol_avg - beta_fast_alpha - beta_beam
