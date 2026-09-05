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

from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    OutputInto,
)

from functional_process.paths import current_drive, physics
from functional_process.physics.pure_formulas import (
    calaculate_stored_thermal_energy,
    calculate_total_plasma_heating_power,
    fast_alpha_beta,
    fast_alpha_beta_iter_physics_rules,
    fast_alpha_beta_ward,
    phyaux,
    rether,
)

__all__ = [
    "fast_alpha_beta",
]


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
    """The `fast_alpha_beta` family -- one occupant per `.physics.i_beta_fast_alpha`
    value.

    **`i_beta_fast_alpha` was an `eqx.field(static=True)` here and is gone**
    (`_audit/next_steps.md` §14.2). This is the case that policy was hardest on: the two
    arms read *exactly* the same ten fields and differ only in two coefficients and a
    `sqrt` guard, so `switch_kwarg_survey.md` band (c) recorded it as inventing no edge,
    and `traceability_policy.md`'s "exception: static kwarg" was written with it in
    mind. Split anyway, and the argument is `model_tree_design.md` §4's rather than the
    survey's: an enum family has no cheap escape when a third published formula needs a
    read the family cannot express, and the two occupants are two named models rather
    than two arms of an integer.
    """

    beta_fast_alpha = OutputInto(physics)


class FastAlphaBetaIterPhysicsRules(FastAlphaBeta):
    """`i_beta_fast_alpha == ITER_PHYSICS_RULES` (0) -- the ITER Physics Design
    Guidelines fraction (`physics.py:2043`).
    """

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
        return fast_alpha_beta_iter_physics_rules(
            b_plasma_surface_poloidal_average,
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


class FastAlphaBetaWard(FastAlphaBeta):
    """`i_beta_fast_alpha == WARD` (1) -- PROCESS's own default
    (`physics_variables.py:238`) and the reference run's.
    """

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
        return fast_alpha_beta_ward(
            b_plasma_surface_poloidal_average,
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
