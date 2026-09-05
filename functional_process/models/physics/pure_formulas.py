"""Pure physics functions extracted from `models/physics/pure_formulas.py`.

See that file for the declarations wiring these into the graph, and
`_audit/units/models/physics/pure_formulas.md` for the audit record. No
graph-framework import belongs in this module -- see `_audit/formulas_split.md`.
"""

import jax.numpy as jnp

from functional_process.models.safe_math import safe_sqrt
from functional_process.vocabulary import constants

WARD_KINK_SMOOTHING = 1.0e-3
"""Width of the regularisation applied to `_fast_alpha_fraction_ward`'s threshold.

**Its two costs are exactly conjugate.** At the threshold the value shift against
PROCESS's expression is `0.26 r^2 sqrt(eps/2)` and the derivative cap is
`0.065 r^2 / sqrt(eps/2)`, so their product is `0.0169 r^4` **whatever `eps` is** -- an
order of magnitude off the derivative costs an order of magnitude of value error, and no
retuning improves both (`_audit/optimise_design.md` §31.37.4).

**This cannot simply be reduced.** `eps` in roughly `1e-5 .. 3e-4` lands in a measured
failure basin: the regularisation leaves a sub-threshold tail whose gradient is linear in
`eps` (`~0.065 r^2 eps / |a|^{3/2}`, where PROCESS's is exactly zero), and in the band it
is strong enough to attract the optimiser below the threshold but too weak to guide it
there -- `stellarator_helias` wanders to `temp_sum_20 - 0.65 ~ -0.3` and 4 of 8 sampled
`eps` in the band stop (§31.40). `5e-4 .. 5e-3` all converge in 19-24 iterations.

**`5e-4` was measured and deliberately not taken** (§31.41): it clears `test_mdf.py`'s
`WORST_DX` where this does not (`9.60e-09` and `9.29e-09` against `1e-8`) and is equally
deterministic (23 iterations, 13 digits), but by a 4 % margin on a tripwire and sitting at
the basin's edge. A recorded deviation was preferred to an unrecorded fragility.
"""


def rether(
    alphan,
    alphat,
    nd_plasma_electrons_vol_avg,
    dlamie,
    te,
    temp_plasma_ion_vol_avg_kev,
    n_charge_plasma_effective_mass_weighted_vol_avg,
):
    """Ion/electron equilibration power density (MW/m^3).

    Ports `physics.rether` (`physics.py:103-152`) verbatim -- straight-line arithmetic,
    no branches, no domain guard needed (`(1.0 + alphat)` and `te**1.5` are always
    evaluated at physically positive profile indices and temperatures in PROCESS's own
    usage).

    Parameters
    ----------
    alphan :
        Density profile index.
    alphat :
        Temperature profile index.
    nd_plasma_electrons_vol_avg :
        Electron density (m^-3).
    dlamie :
        Ion-electron Coulomb logarithm.
    te :
        Electron temperature (keV).
    temp_plasma_ion_vol_avg_kev :
        Ion temperature (keV).
    n_charge_plasma_effective_mass_weighted_vol_avg :
        Mass-weighted plasma effective charge.

    Returns
    -------
    :
        Ion/electron equilibration power density (MW/m^3).
    """
    profie = (1.0 + alphan) ** 2 / (
        (2.0 * alphan - 0.5 * alphat + 1.0) * safe_sqrt(1.0 + alphat)
    )
    conie = (
        2.42165e-41
        * dlamie
        * nd_plasma_electrons_vol_avg**2
        * n_charge_plasma_effective_mass_weighted_vol_avg
        * profie
    )
    return conie * (temp_plasma_ion_vol_avg_kev - te) / (te**1.5)


def phyaux(
    aspect,
    nd_plasma_fuel_ions_vol_avg,
    fusden_total,
    fusden_alpha_total,
    plasma_current,
    sbar,
    nd_plasma_alphas_thermal_vol_avg,
    t_energy_confinement,
    vol_plasma,
    burnup_in,
    tauratio,
):
    """Auxiliary physics quantities.

    Ports `Physics.phyaux` (`physics.py:1493-1602`, itself already an
    `@staticmethod`/`@nb.njit`). Two of PROCESS's `if`/`else` selections become
    `jnp.where`, each with its *unselected* branch's denominator substituted rather than
    left to divide by the value that made it unselected in the first place:

    - `t_alpha_confinement = 0.0 if fusden_alpha_total == 0.0 else nd_alphas /
      fusden_alpha_total` -- the reference implementation short-circuits and never
      evaluates the division when the guard fires; a traced `jnp.where` evaluates both
      arms, so the substitution keeps the *unselected* arm's division away from `0/0`.
      Both a genuine `0/0` (`NaN`) and any exact-zero float would otherwise poison the
      gradient through the selected `0.0` branch, exactly the "value looks right,
      gradient is `NaN`" failure `_audit/test_harness.md`'s pilot retrospective
      describes.
    - `burnup = burnup_in if burnup_in > 1e-9 else <computed via tauratio>` -- same
      guard, this time on `tauratio` (only ever a physical, positive ratio in PROCESS's
      own usage, but the *unselected* branch of the `jnp.where` still traces it).

    Parameters
    ----------
    aspect :
        Plasma aspect ratio.
    nd_plasma_fuel_ions_vol_avg :
        Fuel ion density (/m3).
    fusden_total :
        Fusion reaction rate from plasma and beams (/m3/s).
    fusden_alpha_total :
        Alpha particle production rate (/m3/s).
    plasma_current :
        Plasma current (A).
    sbar :
        Exponent for aspect ratio (PROCESS's stellarator caller always passes the
        literal `1.0`; see the audit record's data-footprint table).
    nd_plasma_alphas_thermal_vol_avg :
        Alpha ash density (/m3).
    t_energy_confinement :
        Global energy confinement time (s).
    vol_plasma :
        Plasma volume (m3).
    burnup_in :
        User-input fractional plasma burnup (`<= 1e-9` means "not specified, compute
        it").
    tauratio :
        Ratio of He and pellet particle confinement times.

    Returns
    -------
    tuple
        `(burnup, figmer, fusrat, molflow_plasma_fuelling_required, rndfuel,
        t_alpha_confinement, f_t_alpha_energy_confinement)`, matching
        `Physics.phyaux`'s return order exactly (including `fusrat`, which PROCESS's
        stellarator caller discards -- see the audit record).
    """
    figmer = 1.0e-6 * plasma_current * aspect**sbar
    fusrat = fusden_total * vol_plasma

    no_alphas = fusden_alpha_total == 0.0
    safe_fusden_alpha_total = jnp.where(no_alphas, 1.0, fusden_alpha_total)
    t_alpha_confinement = jnp.where(
        no_alphas, 0.0, nd_plasma_alphas_thermal_vol_avg / safe_fusden_alpha_total
    )

    burnup_is_computed = burnup_in <= 1.0e-9
    safe_tauratio = jnp.where(burnup_is_computed, tauratio, 1.0)
    burnup_computed = (
        nd_plasma_alphas_thermal_vol_avg
        / (nd_plasma_alphas_thermal_vol_avg + 0.5 * nd_plasma_fuel_ions_vol_avg)
        / safe_tauratio
    )
    burnup = jnp.where(burnup_is_computed, burnup_computed, burnup_in)

    rndfuel = fusrat
    molflow_plasma_fuelling_required = rndfuel / burnup
    f_t_alpha_energy_confinement = t_alpha_confinement / t_energy_confinement

    return (
        burnup,
        figmer,
        fusrat,
        molflow_plasma_fuelling_required,
        rndfuel,
        t_alpha_confinement,
        f_t_alpha_energy_confinement,
    )


def calculate_total_plasma_heating_power(
    f_p_alpha_plasma_deposited,
    p_alpha_total_mw,
    p_non_alpha_charged_mw,
    p_plasma_ohmic_mw,
    p_hcd_injected_total_mw,
):
    """Total plasma heating power (MW).

    Ports `Physics.calculate_total_plasma_heating_power` (`physics.py:3538-3568`)
    verbatim -- one linear combination, no branches.
    """
    return (
        f_p_alpha_plasma_deposited * p_alpha_total_mw
        + p_non_alpha_charged_mw
        + p_plasma_ohmic_mw
        + p_hcd_injected_total_mw
    )


def calaculate_stored_thermal_energy(
    vol_plasma,
    nd_plasma_vol_avg,
    temp_plasma_density_weighted_vol_avg_kev,
):
    """Stored thermal energy (density and total) for one species.

    Ports `Physics.calaculate_stored_thermal_energy` (`physics.py:3573-3609`) verbatim,
    under PROCESS's own misspelling ("calaculate") -- see
    `_audit/naming_convention.md`'s "port the existing name where one already exists".
    Species-agnostic: PROCESS's stellarator caller (`stellarator.py:2267-2283`) calls
    this twice, once for electrons and once for ions, with different `VarPath`
    bindings -- see the two node classes below.

    Parameters
    ----------
    vol_plasma :
        Plasma volume (m^3).
    nd_plasma_vol_avg :
        Volume-averaged density of the species (/m^3).
    temp_plasma_density_weighted_vol_avg_kev :
        Volume-averaged, density-weighted temperature of the species (keV).

    Returns
    -------
    tuple
        `(eden_plasma_thermal_vol_avg, e_plasma_thermal)`: energy density (J/m^3) and
        total stored energy (J).
    """
    eden_plasma_thermal_vol_avg = (
        1.5
        * constants.KILOELECTRON_VOLT
        * nd_plasma_vol_avg
        * temp_plasma_density_weighted_vol_avg_kev
    )
    e_plasma_thermal = eden_plasma_thermal_vol_avg * vol_plasma
    return eden_plasma_thermal_vol_avg, e_plasma_thermal


def fast_alpha_beta(
    b_plasma_poloidal_average,
    b_plasma_toroidal_on_axis,
    nd_plasma_electrons_vol_avg,
    nd_plasma_fuel_ions_vol_avg,
    nd_plasma_ions_total_vol_avg,
    temp_plasma_electron_density_weighted_kev,
    temp_plasma_ion_density_weighted_kev,
    pden_alpha_total_mw,
    pden_plasma_alpha_mw,
    i_beta_fast_alpha,
    f_plasma_fuel_deuterium,
):
    """Fast alpha beta (beta_fast_alpha) component.

    Ports `PlasmaBeta.fast_alpha_beta` (`physics.py:4265-4392`, itself already an
    `@staticmethod`/`@nb.njit`). `f_plasma_fuel_deuterium < 1.0` is a genuine
    data-dependent (traced) branch -- unlike `i_beta_fast_alpha`, it is a continuous
    fuel-mix fraction, not a configuration switch -- so it becomes `jnp.where`, guarded
    the same way as `phyaux`'s: `pden_plasma_alpha_mw` is `0.0` in PROCESS's own comment
    on the untaken (`>= 1.0`, "negligible alpha production") branch, and the reference
    divides by it only because Python short-circuits past the `else`. `i_beta_fast_alpha`
    keeps Python control flow -- see `FastAlphaBeta` below for why that is safe under
    tracing.

    Parameters
    ----------
    b_plasma_poloidal_average :
        Poloidal field (T).
    b_plasma_toroidal_on_axis :
        Toroidal field on axis (T).
    nd_plasma_electrons_vol_avg :
        Electron density (m^-3).
    nd_plasma_fuel_ions_vol_avg :
        Fuel ion density (m^-3).
    nd_plasma_ions_total_vol_avg :
        Total ion density (m^-3).
    temp_plasma_electron_density_weighted_kev :
        Density-weighted electron temperature (keV).
    temp_plasma_ion_density_weighted_kev :
        Density-weighted ion temperature (keV).
    pden_alpha_total_mw :
        Alpha power per unit volume, from beams and plasma (MW/m^3).
    pden_plasma_alpha_mw :
        Alpha power per unit volume just from plasma (MW/m^3).
    i_beta_fast_alpha :
        Switch for fast alpha pressure method (`0` = IPDG89, else Ward). A **static**
        Python `int`, not a traced value -- see `FastAlphaBeta`.
    f_plasma_fuel_deuterium :
        Plasma deuterium fuel fraction.

    Returns
    -------
    :
        Fast alpha beta component.
    """
    arm = (
        fast_alpha_beta_iter_physics_rules
        if i_beta_fast_alpha == 0
        else fast_alpha_beta_ward
    )
    return arm(
        b_plasma_poloidal_average,
        b_plasma_toroidal_on_axis,
        nd_plasma_electrons_vol_avg,
        nd_plasma_fuel_ions_vol_avg,
        nd_plasma_ions_total_vol_avg,
        temp_plasma_electron_density_weighted_kev,
        temp_plasma_ion_density_weighted_kev,
        pden_alpha_total_mw,
        pden_plasma_alpha_mw,
        f_plasma_fuel_deuterium,
    )


def _fast_alpha_fraction_iter_physics_rules(density_ratio_sq, temp_sum_20):
    """`i_beta_fast_alpha == ITER_PHYSICS_RULES` (0): the fast-alpha pressure fraction
    of the ITER Physics Design Guidelines, `physics.py:2043`.

    Split out of `fast_alpha_beta` because a switch value selects an occupant, not a
    coefficient (`_audit/next_steps.md` §14.2). The two arms read *identical* fields --
    `switch_kwarg_survey.md` band (c) -- so nothing structural changes; what changes is
    that the two published formulas are two named objects the tree selects between,
    rather than two branches of an `if` on an integer a node carries.
    """
    return jnp.minimum(0.3, 0.29 * density_ratio_sq * (temp_sum_20 - 0.37))


def _fast_alpha_fraction_ward(density_ratio_sq, temp_sum_20):
    """`i_beta_fast_alpha == WARD` (1): Ward's fast-alpha pressure fraction,
    `physics.py:2049` -- PROCESS's own default and the reference run's.

    **The `sqrt(temp_sum_20 - 0.65)` threshold is smoothed, and that is a deliberate
    departure from PROCESS** -- declared as such by `TestFastAlphaBetaWard`'s
    `declared_deviation`, not absorbed into a tolerance. PROCESS's expression has an
    unbounded derivative at the threshold and `stellarator_helias` runs *on* it (within
    `5.5e-08`, crossing on 46 % of SQP steps, each crossing moving the `c24` Jacobian row
    by `339x`), which made that configuration's converged/stopped outcome turn on the
    last bit of a Jacobian cell.

    **How it works, which is not what it looks like.** `0.5 * (a + sqrt(a**2 + eps**2))`
    in place of `max(a, 0)` does not stabilise the same answer -- it **displaces the
    optimum off the singularity**. The converged `a` tracks `eps` at 2-3x it
    (`2.40e-03` here, against the shipped `5.5e-08`), and the arm stops crossing the kink
    because its optimum is no longer on it. Every `+-1` ulp Jacobian draw then takes the
    same 24 iterations and agrees on `objf` to fifteen digits, against 87-333 iterations
    and one catastrophic stop before. Defensible -- this is a fitted correlation's
    threshold and the optimiser was parked `5.5e-08` from it -- but it is **not** "the
    same answer computed more stably", and should not be read as such.

    **The cost, in one place, accepted deliberately for that determinism**: `objf` moves
    `6.0e-04` relative on `stellarator_helias`; its agreement with PROCESS degrades
    (`d objf 2.34e-03 -> 2.94e-03`, `ixc 109` `1.08e-01 -> 1.09e-01`); SAND's largest
    equality residual goes `2.88e-06 -> 7.13e-06`; three other configurations move 3-4
    orders further from PROCESS on rows that had agreed to twelve digits; and it takes one
    tier-1 declared deviation, four `cold_start.ACCEPTED` entries, and a per-configuration
    `WORST_DX` deviation on two tokamak root-finds. Below the threshold PROCESS returns
    **exactly zero** and this returns at most `4.6e-05` on a quantity ranging to
    `1.5e-02` -- the qualitative infidelity, unavoidable for this family.
    `_audit/optimise_design.md` §31.36-§31.41 has the measurements and the argument.

    The `safe_sqrt`/double-`jnp.where` this used to need is gone with the clamp: the
    argument to `sqrt` is now strictly positive, so there is no `inf * 0` to guard.
    """
    above = temp_sum_20 - 0.65
    soft = 0.5 * (
        above + jnp.sqrt(above * above + WARD_KINK_SMOOTHING * WARD_KINK_SMOOTHING)
    )
    return jnp.minimum(0.30, 0.26 * density_ratio_sq * jnp.sqrt(soft))


def fast_alpha_beta_iter_physics_rules(
    b_plasma_poloidal_average,
    b_plasma_toroidal_on_axis,
    nd_plasma_electrons_vol_avg,
    nd_plasma_fuel_ions_vol_avg,
    nd_plasma_ions_total_vol_avg,
    temp_plasma_electron_density_weighted_kev,
    temp_plasma_ion_density_weighted_kev,
    pden_alpha_total_mw,
    pden_plasma_alpha_mw,
    f_plasma_fuel_deuterium,
):
    """`fast_alpha_beta` at `i_beta_fast_alpha == ITER_PHYSICS_RULES` (0).

    Parameters and return are the composite's, less `i_beta_fast_alpha`.
    """
    return _fast_alpha_beta(
        _fast_alpha_fraction_iter_physics_rules,
        b_plasma_poloidal_average,
        b_plasma_toroidal_on_axis,
        nd_plasma_electrons_vol_avg,
        nd_plasma_fuel_ions_vol_avg,
        nd_plasma_ions_total_vol_avg,
        temp_plasma_electron_density_weighted_kev,
        temp_plasma_ion_density_weighted_kev,
        pden_alpha_total_mw,
        pden_plasma_alpha_mw,
        f_plasma_fuel_deuterium,
    )


def fast_alpha_beta_ward(
    b_plasma_poloidal_average,
    b_plasma_toroidal_on_axis,
    nd_plasma_electrons_vol_avg,
    nd_plasma_fuel_ions_vol_avg,
    nd_plasma_ions_total_vol_avg,
    temp_plasma_electron_density_weighted_kev,
    temp_plasma_ion_density_weighted_kev,
    pden_alpha_total_mw,
    pden_plasma_alpha_mw,
    f_plasma_fuel_deuterium,
):
    """`fast_alpha_beta` at `i_beta_fast_alpha == WARD` (1) -- PROCESS's own default.

    Parameters and return are the composite's, less `i_beta_fast_alpha`.
    """
    return _fast_alpha_beta(
        _fast_alpha_fraction_ward,
        b_plasma_poloidal_average,
        b_plasma_toroidal_on_axis,
        nd_plasma_electrons_vol_avg,
        nd_plasma_fuel_ions_vol_avg,
        nd_plasma_ions_total_vol_avg,
        temp_plasma_electron_density_weighted_kev,
        temp_plasma_ion_density_weighted_kev,
        pden_alpha_total_mw,
        pden_plasma_alpha_mw,
        f_plasma_fuel_deuterium,
    )


def _fast_alpha_beta(
    fast_alpha_fraction,
    b_plasma_poloidal_average,
    b_plasma_toroidal_on_axis,
    nd_plasma_electrons_vol_avg,
    nd_plasma_fuel_ions_vol_avg,
    nd_plasma_ions_total_vol_avg,
    temp_plasma_electron_density_weighted_kev,
    temp_plasma_ion_density_weighted_kev,
    pden_alpha_total_mw,
    pden_plasma_alpha_mw,
    f_plasma_fuel_deuterium,
):
    """Everything both `i_beta_fast_alpha` arms share, given the arm's own fast-alpha
    pressure fraction.

    `fast_alpha_fraction` is the arm's own function, not a switch: which of the two a
    node gets follows from which arm function it called.
    """
    is_dt_like = f_plasma_fuel_deuterium < 1.0

    beta_thermal = (
        2.0
        * constants.RMU0
        * constants.KILOELECTRON_VOLT
        * (
            nd_plasma_electrons_vol_avg * temp_plasma_electron_density_weighted_kev
            + nd_plasma_ions_total_vol_avg * temp_plasma_ion_density_weighted_kev
        )
        / (b_plasma_toroidal_on_axis**2 + b_plasma_poloidal_average**2)
    )

    temp_sum_20 = (
        temp_plasma_electron_density_weighted_kev + temp_plasma_ion_density_weighted_kev
    ) / 20.0
    density_ratio_sq = (nd_plasma_fuel_ions_vol_avg / nd_plasma_electrons_vol_avg) ** 2

    fact = jnp.maximum(fast_alpha_fraction(density_ratio_sq, temp_sum_20), 0.0)

    safe_pden_plasma_alpha_mw = jnp.where(is_dt_like, pden_plasma_alpha_mw, 1.0)
    fact2 = pden_alpha_total_mw / safe_pden_plasma_alpha_mw

    return jnp.where(is_dt_like, beta_thermal * fact * fact2, 0.0)
