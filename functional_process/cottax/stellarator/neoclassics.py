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
import jax.numpy as jnp  # noqa: F401
import numpy as np  # noqa: F401
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    OutputInto,
)

from functional_process.models.safe_math import (
    safe_sqrt,  # noqa: F401
)
from functional_process.cottax.paths import (
    impurity_radiation,
    neoclassics,
    physics,
    stellarator,
    stellarator_config,
)
from functional_process.models.stellarator.neoclassics import (
    KEV,  # noqa: F401
    NO_ROOTS,  # noqa: F401
    ROOTS,  # noqa: F401
    WEIGHTS,  # noqa: F401
    calculate_collision_frequency,  # noqa: F401
    calculate_drift_velocity,  # noqa: F401
    calculate_effective_thermal_diffusivity,
    calculate_gamma_flux,  # noqa: F401
    calculate_integrated_radial_transport_coefficient,  # noqa: F401
    calculate_kt,  # noqa: F401
    calculate_monoenergetic_transport_coefficient,  # noqa: F401
    calculate_normalized_collision_frequency,  # noqa: F401
    calculate_normalized_collision_frequency_from_temperature,  # noqa: F401
    calculate_plateau_transport_coefficient,  # noqa: F401
    calculate_profile_values,
    calculate_q_flux,  # noqa: F401
)
from functional_process.vocabulary import (
    constants,  # noqa: F401
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
