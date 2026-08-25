"""Pure-functional port of five already-pure formulas from
`process/models/physics/physics.py`.

Registry unit #9, chunk A. Audit record:
`functional_process/_audit/units/models/physics/pure_formulas.md`.

All five source functions are, in PROCESS's own code, already `@staticmethod`s (or, for
`rether`, a bare module-level function) with an explicit signature and no `self.data`
access at all -- the extraction seam `../../CLAUDE.md` describes as already present in
this codebase, found intact here. Nothing about their *shape* changes in this port; only
Python control flow that cannot survive tracing (`if`/`min`/`max` on a traced value)
becomes `jnp.where`/`jnp.minimum`/`jnp.maximum`.

- `rether` -- ion/electron equilibration power density. No branches at all.
- `phyaux` -- auxiliary physics quantities (burnup, figmer, fuelling rate, ...). Two
  `if`/`else` selections become `jnp.where`, both guarded against a zero denominator on
  the untaken branch (see the function's docstring) -- this is exactly the class of bug
  `_audit/test_harness.md`'s pilot retrospective flags: a `jnp.where` is only as safe as
  its *unselected* branch, because JAX still evaluates it, and an unguarded one produces
  a finite value but a `NaN` gradient.
- `calculate_total_plasma_heating_power`, `calaculate_stored_thermal_energy` -- pure
  arithmetic, no branches. The second is ported under its source spelling
  (`calaculate_*`, PROCESS's own typo) per `_audit/naming_convention.md` ("port the
  existing name where one already exists"); see the audit record for where the typo is
  first flagged.
- `fast_alpha_beta` (on `PlasmaBeta`, not `Physics` -- see the audit record for why it
  is grouped here anyway) -- one traced branch (`f_plasma_fuel_deuterium < 1.0`,
  `jnp.where`, again denominator-guarded: `pden_plasma_alpha_mw` is `0` on the untaken
  branch) and one static branch (`i_beta_fast_alpha`, kept as a static field on the
  node -- see the audit record's switches section for the reads-set justification).
"""

import equinox as eqx
import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    OutputInto,
)

from functional_process.models.safe_math import safe_sqrt
from functional_process.models.switch_enums import FastAlphaPressureModel
from functional_process.paths import current_drive, physics
from process.core import constants


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

    if i_beta_fast_alpha == 0:
        fact = jnp.minimum(0.3, 0.29 * density_ratio_sq * (temp_sum_20 - 0.37))
    else:
        # `jnp.sqrt(jnp.maximum(0.0, x))` is value-correct and returns `nan` from
        # `jacfwd` on the clamped branch, because `sqrt` has an infinite derivative at
        # zero and `inf * 0` is `nan`. The standard **double `jnp.where`** avoids it:
        # the inner one keeps a finite argument out of `sqrt`'s reverse/forward rule, the
        # outer one selects the value. Same defect and same fix as
        # `costs.py:2874-2888`'s clamped net-electric-power square root
        # (`_audit/next_steps.md` §9, "a JAX trap worth carrying forward").
        #
        # **Live on the reference run, not hypothetical**: the clamp is *active* there
        # (`temp_sum_20 = 0.6449` against the 0.65 threshold), and this row was the only
        # non-finite row of the SAND Jacobian once `.physics.beta_total_vol_avg` gained a
        # producer and constraint 24 started reading it. It was invisible before because
        # nothing downstream of `beta_fast_alpha` fed a condition, so `jacfwd` never
        # traced it.
        above = temp_sum_20 - 0.65
        positive = above > 0.0
        fact = jnp.minimum(
            0.30,
            0.26
            * density_ratio_sq
            * jnp.where(positive, safe_sqrt(jnp.where(positive, above, 1.0)), 0.0),
        )
    fact = jnp.maximum(fact, 0.0)

    safe_pden_plasma_alpha_mw = jnp.where(is_dt_like, pden_plasma_alpha_mw, 1.0)
    fact2 = pden_alpha_total_mw / safe_pden_plasma_alpha_mw

    return jnp.where(is_dt_like, beta_thermal * fact * fact2, 0.0)


class IonElectronEquilibration(ExplicitFunction):
    """cottax node: `rether`, unchanged, ports declared."""

    pden_ion_electron_equilibration_mw = OutputInto(physics)

    def __call__(
        self,
        alphan=From(physics),
        alphat=From(physics),
        nd_plasma_electrons_vol_avg=From(physics),
        dlamie=From(physics),
        temp_plasma_electron_vol_avg_kev=From(physics),
        temp_plasma_ion_vol_avg_kev=From(physics),
        n_charge_plasma_effective_mass_weighted_vol_avg=From(physics),
    ):
        return rether(
            alphan,
            alphat,
            nd_plasma_electrons_vol_avg,
            dlamie,
            temp_plasma_electron_vol_avg_kev,
            temp_plasma_ion_vol_avg_kev,
            n_charge_plasma_effective_mass_weighted_vol_avg,
        )


class AuxiliaryPhysicsQuantities(ExplicitFunction):
    """cottax node: `phyaux`, ports declared.

    `sbar` is not a `VarPath`: PROCESS's stellarator caller (`stellarator.py:2378`)
    always passes the Python literal `1.0`, never a `data` field -- kept as a plain
    (non-read) field here rather than hardcoded into the body, so it stays visible and
    overridable rather than silently baked in. `fusrat`'s output is minted to the real
    `.physics.fusrat` field even though the stellarator call site discards it (assigns to
    `_fusrat`) -- see the audit record; `physics.py`'s own (tokamak, unit #22) caller
    does store it there.
    """

    sbar: float = 1.0

    burnup = OutputInto(physics)
    figmer = OutputInto(physics)
    fusrat = OutputInto(physics)
    molflow_plasma_fuelling_required = OutputInto(physics)
    rndfuel = OutputInto(physics)
    t_alpha_confinement = OutputInto(physics)
    f_t_alpha_energy_confinement = OutputInto(physics)

    def __call__(
        self,
        aspect=From(physics),
        nd_plasma_fuel_ions_vol_avg=From(physics),
        fusden_total=From(physics),
        fusden_alpha_total=From(physics),
        plasma_current=From(physics),
        nd_plasma_alphas_thermal_vol_avg=From(physics),
        t_energy_confinement=From(physics),
        vol_plasma=From(physics),
        burnup_in=From(physics),
        tauratio=From(physics),
    ):
        return phyaux(
            aspect,
            nd_plasma_fuel_ions_vol_avg,
            fusden_total,
            fusden_alpha_total,
            plasma_current,
            self.sbar,
            nd_plasma_alphas_thermal_vol_avg,
            t_energy_confinement,
            vol_plasma,
            burnup_in,
            tauratio,
        )


class TotalPlasmaHeatingPower(ExplicitFunction):
    """cottax node: `calculate_total_plasma_heating_power`, unchanged, ports declared."""

    p_plasma_heating_total_mw = OutputInto(physics)

    def __call__(
        self,
        f_p_alpha_plasma_deposited=From(physics),
        p_alpha_total_mw=From(physics),
        p_non_alpha_charged_mw=From(physics),
        p_plasma_ohmic_mw=From(physics),
        p_hcd_injected_total_mw=From(current_drive),
    ):
        return calculate_total_plasma_heating_power(
            f_p_alpha_plasma_deposited,
            p_alpha_total_mw,
            p_non_alpha_charged_mw,
            p_plasma_ohmic_mw,
            p_hcd_injected_total_mw,
        )


class ElectronThermalEnergy(ExplicitFunction):
    """cottax node: `calaculate_stored_thermal_energy`, electron binding.

    One of two node classes wrapping the same species-agnostic function -- see the
    function's own docstring and the audit record's "cottax node" section for why a
    single node cannot cover both of PROCESS's call sites.
    """

    eden_plasma_electrons_thermal_vol_avg = OutputInto(physics)
    e_plasma_electrons_thermal = OutputInto(physics)

    def __call__(
        self,
        vol_plasma=From(physics),
        nd_plasma_electrons_vol_avg=From(physics),
        temp_plasma_electron_density_weighted_kev=From(physics),
    ):
        return calaculate_stored_thermal_energy(
            vol_plasma,
            nd_plasma_electrons_vol_avg,
            temp_plasma_electron_density_weighted_kev,
        )


class IonThermalEnergy(ExplicitFunction):
    """cottax node: `calaculate_stored_thermal_energy`, ion binding. See
    `ElectronThermalEnergy`.
    """

    eden_plasma_ions_thermal_vol_avg = OutputInto(physics)
    e_plasma_ions_thermal = OutputInto(physics)

    def __call__(
        self,
        vol_plasma=From(physics),
        nd_plasma_ions_total_vol_avg=From(physics),
        temp_plasma_ion_density_weighted_kev=From(physics),
    ):
        return calaculate_stored_thermal_energy(
            vol_plasma,
            nd_plasma_ions_total_vol_avg,
            temp_plasma_ion_density_weighted_kev,
        )


class FastAlphaBeta(ExplicitFunction):
    """cottax node: `fast_alpha_beta`, ports declared.

    `i_beta_fast_alpha` is a **static** field, not a read. Per
    `_audit/traceability_policy.md`'s switch-split default, this is the "exception:
    static kwarg" case, not the default "split": both branches read exactly the same
    variables (`nd_plasma_fuel_ions_vol_avg`, `nd_plasma_electrons_vol_avg`,
    `temp_plasma_electron_density_weighted_kev`, `temp_plasma_ion_density_weighted_kev`)
    and differ only in two numeric coefficients and whether a `sqrt` guard is applied --
    a genuine solver-method-choice shape, not alternate physics with a different input
    set. Same precedent as `EcrhDensityLimit.i_plasma_pedestal`
    (`models/stellarator/density_limits.py`).
    """

    i_beta_fast_alpha: FastAlphaPressureModel = eqx.field(static=True)

    beta_fast_alpha = OutputInto(physics)

    def __call__(
        self,
        b_plasma_surface_poloidal_average=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
        nd_plasma_electrons_vol_avg=From(physics),
        nd_plasma_fuel_ions_vol_avg=From(physics),
        nd_plasma_ions_total_vol_avg=From(physics),
        temp_plasma_electron_density_weighted_kev=From(physics),
        temp_plasma_ion_density_weighted_kev=From(physics),
        pden_alpha_total_mw=From(physics),
        pden_plasma_alpha_mw=From(physics),
        f_plasma_fuel_deuterium=From(physics),
    ):
        return fast_alpha_beta(
            b_plasma_surface_poloidal_average,
            b_plasma_toroidal_on_axis,
            nd_plasma_electrons_vol_avg,
            nd_plasma_fuel_ions_vol_avg,
            nd_plasma_ions_total_vol_avg,
            temp_plasma_electron_density_weighted_kev,
            temp_plasma_ion_density_weighted_kev,
            pden_alpha_total_mw,
            pden_plasma_alpha_mw,
            self.i_beta_fast_alpha,
            f_plasma_fuel_deuterium,
        )
