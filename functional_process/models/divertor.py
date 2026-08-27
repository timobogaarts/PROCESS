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
hldiv_base` directly, no `f_p_div_lower` read at all).

2026-08-27 (the double-null wave): the `n_divertors == 2` formula is written too, as
`calculate_divertor_heat_load_wade_double_null`, for the two spherical-tokamak input
files that set `i_single_null = 0`. It is the one arm in that wave that adds a *read*
rather than only changing a constant: `.physics.f_p_div_lower`.

**`.physics.f_p_div_lower` is a declared boundary input and nothing in this port
produces it.** Measured, not assumed: `grep -rn f_p_div_lower process/` finds the field
declared at `physics_variables.py:740` with a default of `1.0`, registered as a
user-settable `InputVariable` at `core/input.py:189` and as scan variable 51
(`core/scan.py:194`), read at `models/divertor.py:101`/`:378-379` and at
`models/physics/physics.py:852`/`:1008-1052` -- and **written nowhere** outside the
input parser. It is an input in PROCESS and it is an input here; the double-null
occupant declares the read and the boundary carries it, which is the honest answer.
Stubbing it to `1.0` would silently pick the lower divertor and hide the `max`. Both ST
files set it explicitly (`spherical_tokamak_eval.IN.DAT:266`, `st_regression.IN.DAT:634`,
both `0.5`).
"""

import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    FromExactly,
    OutputInto,
)

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


def _divwade_hldiv_base(
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
    """`hldiv_base` -- everything in `Divertor.divwade` above its `n_divertors` branch.

    Ports `process/models/divertor.py:322-374`, which both arms share verbatim; only
    what is done with the result differs. Private, and not a node: it owns no `VarPath`
    and exists so the two occupants below hold one copy of the Wade scaling between
    them rather than two that can drift.

    Every fractional power is wrapped in `safe_pow` (`_audit/next_steps.md` §9's
    zero-derivative trap -- found by `--fp-gradients`, see `divertor.md`'s
    JAX-difficulty flags); `b_plasma_toroidal_on_axis == 0` is a separate, unfixable
    division singularity inside `atan(bp_omp/bt_omp)`, registered in
    `_harness/boundary.py` rather than worked around.

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
        `hldiv_base` (MW/m^2) -- the heat load before the divertor count is applied.
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
    `process/models/divertor.py:272-406`, `else` arm (`:382`): the base heat load is
    the answer, and this occupant never reads `f_p_div_lower` at all -- see module
    docstring. The Wade scaling itself is `_divwade_hldiv_base` above; the parameters
    are its parameters.

    Returns
    -------
    :
        Divertor heat load (MW/m^2). `.divertor.pflux_div_heat_load_mw`.
    """
    return _divwade_hldiv_base(
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


def calculate_divertor_heat_load_wade_double_null(
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
    f_p_div_lower,
):
    """Divertor heat load, Wade (2020) scaling, `n_divertors == 2` (double null). Ports
    the `if n_divertors == 2` arm of `Divertor.divwade`,
    `process/models/divertor.py:377-380`.

    The separatrix power splits between the two divertors in the ratio
    `f_p_div_lower : 1 - f_p_div_lower`, and PROCESS sizes the machine on whichever of
    the two sees more: `max(f * base, (1 - f) * base)`. `max` -> `jnp.maximum`, which is
    the same function and the same value.

    **One extra read, `f_p_div_lower`, and nothing in this port produces it** -- it is a
    PROCESS input with a default of `1.0` and no writer anywhere in `process/models/`.
    See the module docstring for the measurement.

    **`max` is not differentiable where the two arms meet**, which is at
    `f_p_div_lower == 0.5` -- and both spherical-tokamak input files set exactly that.
    The value is unambiguous there (`0.5 * base` either way); the *derivative* is not,
    and `jnp.maximum`'s JVP splits the tangent evenly between the arms while a
    one-sided finite difference picks one of them. This is a property of the model
    PROCESS wrote, not of the port -- a double-null machine balanced exactly between its
    divertors is at a kink of its own heat-load definition -- so it is recorded rather
    than smoothed. The harness case keeps its sample points off the tie for that
    reason.

    Parameters
    ----------
    rmajor, rminor, aspect, b_plasma_toroidal_on_axis, b_plasma_poloidal_average,
    p_plasma_separatrix_mw, f_div_flux_expansion, nd_plasma_separatrix_electron,
    deg_div_field_plate, rad_fraction_sol :
        As `_divwade_hldiv_base` above.
    f_p_div_lower :
        Fraction of the separatrix power reaching the **lower** divertor.
        `.physics.f_p_div_lower` -- a boundary input, see the module docstring.

    Returns
    -------
    :
        Divertor heat load (MW/m^2). `.divertor.pflux_div_heat_load_mw`.
    """
    hldiv_base = _divwade_hldiv_base(
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

    hldiv_lower = f_p_div_lower * hldiv_base
    hldiv_upper = (1.0 - f_p_div_lower) * hldiv_base

    return jnp.maximum(hldiv_lower, hldiv_upper)


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
    """The family that owns `.divertor.pflux_div_heat_load_mw` at `i_div_heat_load ==
    DivertorHeatLoadModel.WADE` (2): one occupant per `n_divertors` arm of
    `Divertor.divwade`'s own internal branch.

    Both arms are written (2026-08-27). `USER_INPUT` and `PENG_CHAMBER` (`divtart`) are
    different models of the same quantity and remain UNPORTED -- they are values of
    `i_div_heat_load`, not members of this family, which is why `indat.py` answers the
    two switches with one joint `_divertor_heat_load_arm`.
    """


class DivertorHeatLoadWadeSingleNull(DivertorHeatLoadWade):
    """cottax node: `.tokamak.divertor`'s heat-load occupant at `n_divertors == 1`
    (single null) -- this run's derived value on `large_tokamak_eval.IN.DAT`, see module
    docstring.
    """

    pflux_div_heat_load_mw = OutputInto(divertor)

    def __call__(
        self,
        rmajor=From(physics),
        rminor=From(physics),
        aspect=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
        # **Not `From(physics)`.** `divwade`'s *parameter* is called
        # `b_plasma_poloidal_average` and the *field* PROCESS passes into it is
        # `.physics.b_plasma_surface_poloidal_average` (`process/models/
        # divertor.py:90-95`, positional) -- the two names differ, so resolving the
        # parameter name against the namespace named a field that does not exist on
        # `DataStructure` at all. That is not a read of the wrong number, it is a read
        # of *no* number: the port kept the parameter's spelling as a boundary input
        # nothing produces, and the MDA harness's first tokamak run reported it as its
        # only ungrounded input, with 16 outputs downstream of it unverifiable.
        # `physics.py:707`/`:3827` pass the same field into the same parameter name
        # elsewhere, and `pure_formulas.py:527` already binds it the right way round.
        b_plasma_poloidal_average=FromExactly(physics.b_plasma_surface_poloidal_average),
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


class DivertorHeatLoadWadeDoubleNull(DivertorHeatLoadWade):
    """cottax node: `.tokamak.divertor`'s heat-load occupant at `n_divertors == 2`
    (double null) -- the value `spherical_tokamak_eval.IN.DAT` and
    `st_regression.IN.DAT` derive from `i_single_null = 0`.

    Reads one field its single-null sibling does not: `.physics.f_p_div_lower`, a
    boundary input with no producer anywhere in this port (see the module docstring for
    the measurement). It is declared rather than stubbed, so the boundary census counts
    it and a machine assembled from this occupant says out loud that it needs a number
    the graph cannot compute.
    """

    pflux_div_heat_load_mw = OutputInto(divertor)

    def __call__(
        self,
        rmajor=From(physics),
        rminor=From(physics),
        aspect=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
        # Same name mismatch as the single-null sibling, and the same fix -- see the
        # comment there.
        b_plasma_poloidal_average=FromExactly(physics.b_plasma_surface_poloidal_average),
        p_plasma_separatrix_mw=From(physics),
        f_div_flux_expansion=From(divertor),
        nd_plasma_separatrix_electron=From(physics),
        deg_div_field_plate=From(divertor),
        rad_fraction_sol=From(physics),
        f_p_div_lower=From(physics),
    ):
        return calculate_divertor_heat_load_wade_double_null(
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
            f_p_div_lower,
        )
