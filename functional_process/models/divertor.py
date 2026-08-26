"""Pure-functional port of `process/models/divertor.py` (`Divertor`,
`.tokamak.divertor`) -- **not** `process/models/stellarator/divertor.py`, which is a
different model of a different device's divertor, already ported separately and half of
the stellarator graph's one non-`problem` cycle. Recording the distinction explicitly,
per the wave-1 brief.

Audit record: `functional_process/_audit/units/models/divertor.md`. Entry point is
`Divertor.run()`, which (a) computes an unconditional heat-flux split, then (b)
dispatches on `.divertor.i_div_heat_load` to one of three heat-load models. Only the
`WADE` arm (`i_div_heat_load == 2`) is live on
`tests/regression/input_files/large_tokamak_eval.IN.DAT:139`; `USER_INPUT` and
`PENG_CHAMBER` (`divtart`, the tight-aspect-ratio ST model) are UNPORTED.

`divwade` also branches internally on `.divertor.n_divertors` (`1` for single-null, `2`
for double-null) -- read to *branch*, not arithmetically, so per the wave-1 policy it
selects a formula, not a parameter. `large_tokamak_eval.IN.DAT:307` sets
`i_single_null = 1`, and `process/core/init.py:606-616` derives `n_divertors = 1` from
that (`DivertorNumberModels.SINGLE_NULL`) -- **not** the `DataStructure` field's own
default of `2`, which only applies before `init.py` runs. `calculate_divertor_heat_load_
wade` below therefore bakes in the single-null formula (`pflux_div_heat_load_mw =
hldiv_base` directly, no `f_p_div_lower` read at all); the double-null formula is
UNPORTED.
"""

import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.models.safe_math import safe_pow
from functional_process.paths import blanket, divertor, fwbs, physics


def calculate_divertor_heat_flux_split(
    deg_blkt_inboard_poloidal_plasma,
    p_plasma_neutron_mw,
    n_divertors,
):
    """`Divertor.run()`'s unconditional preamble: the single-divertor subtended angle,
    the solid-angle fraction it corresponds to, and the neutron power it intercepts.

    Ports `Divertor.single_divertor_angle` (`process/models/divertor.py:106-113`) and
    the `f_ster_div_single`/`incident_neutron_power` lines of `run()` (`:41-50`),
    unchanged. `n_divertors` is read here purely arithmetically (a multiplier), not to
    branch -- an ordinary input, unlike its role inside `divwade` (see module
    docstring).

    `.fwbs.p_div_rad_total_mw` (`run()`'s `incident_radiation_power` sibling call,
    `:52-56`) is not ported: not on `.tokamak.divertor`'s declared boundary and not
    needed to produce what is (`divertor.md` § scope discipline).

    Parameters
    ----------
    deg_blkt_inboard_poloidal_plasma :
        Inboard blanket poloidal angle subtended by the plasma (degrees).
        `.blanket.deg_blkt_inboard_poloidal_plasma`.
    p_plasma_neutron_mw :
        Total neutron power from the plasma (MW). `.physics.p_plasma_neutron_mw`.
    n_divertors :
        Number of divertors (1 or 2). `.divertor.n_divertors`.

    Returns
    -------
    tuple
        `(deg_div_poloidal_plasma, f_ster_div_single, p_div_nuclear_heat_total_mw)`.
    """
    deg_div_poloidal_plasma = (180.0 - deg_blkt_inboard_poloidal_plasma) / 2.0
    f_ster_div_single = deg_div_poloidal_plasma / 360.0
    p_div_nuclear_heat_total_mw = p_plasma_neutron_mw * f_ster_div_single * n_divertors
    return deg_div_poloidal_plasma, f_ster_div_single, p_div_nuclear_heat_total_mw


def calculate_divertor_heat_load_wade(
    rmajor,
    rminor,
    aspect,
    b_plasma_toroidal_on_axis,
    b_plasma_poloidal_average,
    p_plasma_separatrix_mw,
    f_div_flux_expansion,
    nd_plasma_separatrix_electron,
    deg_div_field_plate,
    rad_fraction_sol,
):
    """Divertor heat load, Wade (2020) scaling, `n_divertors == 1` (single null) --
    the combination live on `large_tokamak_eval.IN.DAT`. Ports `Divertor.divwade`,
    `process/models/divertor.py:272-406`; the `n_divertors == 2` lower/upper split
    (which reads `.physics.f_p_div_lower`) is UNPORTED, and this occupant never reads
    `f_p_div_lower` at all -- see module docstring. Every fractional power is wrapped
    in `safe_pow` (`_audit/next_steps.md` §9's zero-derivative trap -- found by
    `--fp-gradients`, see `divertor.md`'s JAX-difficulty flags);
    `b_plasma_toroidal_on_axis == 0` is a separate, unfixable division singularity
    inside `atan(bp_omp/bt_omp)`, registered in `_harness/boundary.py` rather than
    worked around.

    Parameters
    ----------
    rmajor :
        Plasma major radius (m). `.physics.rmajor`.
    rminor :
        Plasma minor radius (m). `.physics.rminor`.
    aspect :
        Tokamak aspect ratio. `.physics.aspect`.
    b_plasma_toroidal_on_axis :
        Toroidal field (T). `.physics.b_plasma_toroidal_on_axis`.
    b_plasma_poloidal_average :
        Poloidal field (T). `.physics.b_plasma_poloidal_average`.
    p_plasma_separatrix_mw :
        Power to the divertor (MW). `.physics.p_plasma_separatrix_mw`.
    f_div_flux_expansion :
        Plasma flux expansion in the divertor. `.divertor.f_div_flux_expansion`.
    nd_plasma_separatrix_electron :
        Electron density at the separatrix (m^-3).
        `.physics.nd_plasma_separatrix_electron`.
    deg_div_field_plate :
        Field line angle w.r.t. the divertor target plate (degrees).
        `.divertor.deg_div_field_plate`.
    rad_fraction_sol :
        SOL radiation fraction. `.physics.rad_fraction_sol`.

    Returns
    -------
    :
        Divertor heat load (MW/m^2). `.divertor.pflux_div_heat_load_mw`.
    """
    r_omp = rmajor + rminor
    bp_omp = -b_plasma_poloidal_average * rmajor / r_omp
    bt_omp = -b_plasma_toroidal_on_axis * rmajor / r_omp

    lambda_eich = (
        1.35
        * safe_pow(p_plasma_separatrix_mw, -0.02)
        * rmajor**0.04
        * safe_pow(b_plasma_poloidal_average, -0.92)
        * safe_pow(aspect, 0.42)
    )

    spread_fact = (
        0.12
        * safe_pow(nd_plasma_separatrix_electron / 1e19, -0.02)
        * safe_pow(p_plasma_separatrix_mw, -0.21)
        * rmajor**0.71
        * safe_pow(b_plasma_poloidal_average, -0.82)
    )

    lambda_int = lambda_eich + 1.64 * spread_fact

    alpha_mid = jnp.degrees(jnp.arctan(bp_omp / bt_omp))
    alpha_div = f_div_flux_expansion * alpha_mid

    theta_div = jnp.arcsin(
        (1 + 1 / alpha_div**2) * jnp.sin(jnp.radians(deg_div_field_plate))
    )

    area_wetted = (
        2 * jnp.pi * rmajor * lambda_int * f_div_flux_expansion * jnp.sin(theta_div)
    )

    return p_plasma_separatrix_mw * (1 - rad_fraction_sol) / area_wetted


class DivertorHeatFluxSplit(ExplicitFunction):
    """cottax node: `.tokamak.divertor`'s unconditional heat-flux split.

    Not switch-gated -- runs regardless of `i_div_heat_load`.
    """

    deg_div_poloidal_plasma = OutputInto(divertor)
    f_ster_div_single = OutputInto(fwbs)
    p_div_nuclear_heat_total_mw = OutputInto(fwbs)

    def __call__(
        self,
        deg_blkt_inboard_poloidal_plasma=From(blanket),
        p_plasma_neutron_mw=From(physics),
        n_divertors=From(divertor),
    ):
        return calculate_divertor_heat_flux_split(
            deg_blkt_inboard_poloidal_plasma, p_plasma_neutron_mw, n_divertors
        )


class DivertorHeatLoadWade(ExplicitFunction):
    """cottax node: `.tokamak.divertor`'s heat-load occupant for `i_div_heat_load ==
    DivertorHeatLoadModel.WADE` (2) -- live on `large_tokamak_eval.IN.DAT`, and for
    `n_divertors == 1` (single null, this run's derived value -- see module
    docstring). `USER_INPUT` and `PENG_CHAMBER` (`divtart`) are UNPORTED; the
    `n_divertors == 2` arm of this same switch value is also UNPORTED.
    """

    pflux_div_heat_load_mw = OutputInto(divertor)

    def __call__(
        self,
        rmajor=From(physics),
        rminor=From(physics),
        aspect=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
        b_plasma_poloidal_average=From(physics),
        p_plasma_separatrix_mw=From(physics),
        f_div_flux_expansion=From(divertor),
        nd_plasma_separatrix_electron=From(physics),
        deg_div_field_plate=From(divertor),
        rad_fraction_sol=From(physics),
    ):
        return calculate_divertor_heat_load_wade(
            rmajor,
            rminor,
            aspect,
            b_plasma_toroidal_on_axis,
            b_plasma_poloidal_average,
            p_plasma_separatrix_mw,
            f_div_flux_expansion,
            nd_plasma_separatrix_electron,
            deg_div_field_plate,
            rad_fraction_sol,
        )
