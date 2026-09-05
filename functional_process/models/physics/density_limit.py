"""Pure physics functions extracted from `models/physics/density_limit.py`.

See that file for the declarations wiring these into the graph, and
`_audit/units/models/physics/density_limit.md` for the audit record. No
graph-framework import belongs in this module -- see `_audit/formulas_split.md`.
"""

import jax.numpy as jnp

from functional_process.models.safe_math import safe_pow, safe_sqrt


def calculate_asdex_density_limit(p_perp, b_plasma_toroidal_on_axis, q95, rmajor, prn1):
    """Old ASDEX density limit. `DensityLimitModel.ASDEX` (1). Ports
    `PlasmaDensityLimit.calculate_asdex_density_limit`,
    `process/models/physics/density_limit.py:144-182`, unchanged.

    Not wired as an occupant -- dead work at `i_density_limit != 1` on the reference
    arm, see the module docstring. Ported and tested for the free oracle it costs
    nothing to keep.
    """
    return (
        1.54e20
        * safe_pow(p_perp, 0.43)
        * safe_pow(b_plasma_toroidal_on_axis, 0.31)
        / safe_pow(q95 * rmajor, 0.45)
    ) / prn1


def calculate_borrass_iter_i_density_limit(
    p_perp, b_plasma_toroidal_on_axis, q95, rmajor, prn1
):
    """Borrass ITER I density limit. `DensityLimitModel.BORRASS_ITER_I` (2). Ports
    `PlasmaDensityLimit.calculate_borrass_iter_i_density_limit`,
    `process/models/physics/density_limit.py:185-223`, unchanged. Not wired (see
    module docstring).
    """
    return (
        1.8e20
        * safe_pow(p_perp, 0.53)
        * safe_pow(b_plasma_toroidal_on_axis, 0.31)
        / safe_pow(q95 * rmajor, 0.22)
    ) / prn1


def calculate_borrass_iter_ii_density_limit(
    p_perp, b_plasma_toroidal_on_axis, q95, rmajor, prn1
):
    """Borrass ITER II density limit. `DensityLimitModel.BORRASS_ITER_II` (3). Ports
    `PlasmaDensityLimit.calculate_borrass_iter_ii_density_limit`,
    `process/models/physics/density_limit.py:226-264`, unchanged. Not wired (see
    module docstring).
    """
    return (
        0.5e20
        * safe_pow(p_perp, 0.57)
        * safe_pow(b_plasma_toroidal_on_axis, 0.31)
        / safe_pow(q95 * rmajor, 0.09)
    ) / prn1


def calculate_jet_edge_radiation_density_limit(
    zeff, p_hcd_injected_total_mw, prn1, qcyl
):
    """JET edge radiation density limit. `DensityLimitModel.JET_EDGE_RADIATION` (4).
    Ports `PlasmaDensityLimit.calculate_jet_edge_radiation_density_limit`,
    `process/models/physics/density_limit.py:267-297`, unchanged in value. Not wired
    (see module docstring).

    Source returns a bare Python `0.0` when `denom <= 0.0` -- a real, PROCESS-intended
    domain branch (not "PROCESS signals invalid input by raising"), so ported as a
    `jnp.where` rather than `reference_domain_errors`, with the classic safe-denominator
    guard (`_audit/test_harness.md`'s `test_gradient_finite`) so the untaken branch's
    `sqrt` argument never sees the true (possibly non-positive) `denom`.
    """
    denom = (zeff - 1.0) * (1.0 - 4.0 / (3.0 * qcyl))
    positive = denom > 0.0
    safe_denom = jnp.where(positive, denom, 1.0)
    return jnp.where(
        positive,
        (1.0e20 * safe_sqrt(p_hcd_injected_total_mw / safe_denom)) / prn1,
        0.0,
    )


def calculate_jet_simple_density_limit(
    b_plasma_toroidal_on_axis, p_plasma_separatrix_mw, rmajor, prn1
):
    """JET simplified density limit. `DensityLimitModel.JET_SIMPLE` (5). Ports
    `PlasmaDensityLimit.calculate_jet_simple_density_limit`,
    `process/models/physics/density_limit.py:300-335`, unchanged. Not wired (see
    module docstring).
    """
    return (
        0.237e20 * b_plasma_toroidal_on_axis * safe_sqrt(p_plasma_separatrix_mw) / rmajor
    ) / prn1


def calculate_hugill_murakami_density_limit(b_plasma_toroidal_on_axis, rmajor, qcyl):
    """Hugill-Murakami density limit. `DensityLimitModel.HUGILL_MURAKAMI` (6). Ports
    `PlasmaDensityLimit.calculate_hugill_murakami_density_limit`,
    `process/models/physics/density_limit.py:338-362`, unchanged. Not wired (see
    module docstring).
    """
    return 3.0e20 * b_plasma_toroidal_on_axis / (rmajor * qcyl)


def calculate_greenwald_density_limit(c_plasma, rminor):
    """Greenwald density limit (n_GW). `DensityLimitModel.GREENWALD` (7). Ports
    `PlasmaDensityLimit.calculate_greenwald_density_limit`,
    `process/models/physics/density_limit.py:365-395`, unchanged.

    **The one arm wired as an occupant** (`GreenwaldDensityLimit` below) -- the
    `large_tokamak_eval.IN.DAT` reference arm's `i_density_limit` value.
    """
    return 1.0e14 * c_plasma / (jnp.pi * rminor**2)


def calculate_asdex_new_density_limit(p_hcd_injected_total_mw, c_plasma, q95, prn1):
    """ASDEX Upgrade new density limit. `DensityLimitModel.ASDEX_NEW` (8). Ports
    `PlasmaDensityLimit.calculate_asdex_new_density_limit`,
    `process/models/physics/density_limit.py:398-440`, unchanged. Not wired (see
    module docstring) -- also PROCESS's own bare `physics_variables.py` default for
    `i_density_limit`, which `large_tokamak_eval.IN.DAT` overrides to `7`.
    """
    return (
        1.0e20
        * 0.506
        * (safe_pow(p_hcd_injected_total_mw, 0.396) * safe_pow(c_plasma / 1.0e6, 0.265))
        / safe_pow(q95, 0.323)
    ) / prn1


def select_enforced_density_limit_greenwald(nd_plasma_electron_max_array_7):
    """The `i_density_limit == 7` (GREENWALD) arm of `get_density_limit_value`.

    Trivial by construction: ports the one entry of `get_density_limit_value`'s
    `model_map` (`process/models/physics/density_limit.py:131-141`) that this occupant
    answers -- `model_map[DensityLimitModel.GREENWALD]`, a bare index into the array
    element `GreenwaldDensityLimit` produces. Not itself a standalone PROCESS function;
    an extraction of one dispatch arm, same shape as
    `bootstrap_current.py`'s `get_bootstrap_current_fraction_value` arms.
    """
    return nd_plasma_electron_max_array_7


def calculate_greenwald_fraction(
    nd_plasma_electron_line, nd_plasma_electron_max_array_7
):
    """Greenwald fraction. Ports the assignment in `PlasmaDensityLimit.run`,
    `process/models/physics/density_limit.py:106-109` -- unconditional, not gated on
    `i_density_limit` (see module docstring).
    """
    return nd_plasma_electron_line / nd_plasma_electron_max_array_7
