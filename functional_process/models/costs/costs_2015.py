"""Pure-functional port of a self-contained subset of `process/models/costs/costs_2015.py`
(the 2015/Kovari cost model, registry unit #18, the other half of `i_cost_model`).

**Scope note**: registry row 18 nominally scopes this unit to the whole of
`Costs2015.run()` (13 methods, ~1227 lines). This file ports 2 of the 13 `calc_*`
methods -- `calc_building_costs` and `calc_land_costs` -- chosen as the cleanest,
fully self-contained representatives of the file's dominant pattern (see `costs_2015.md`
for the full per-method audit and the remaining 11). See `costs.md` (this directory,
sibling file) for the `i_cost_model` topology-switch finding, which is this dispatch's
main deliverable and applies to both cost models equally -- not repeated here.

**No cottax node is written for either function** -- see `costs_2015.md`'s "cottax node"
section. Both functions write into *slices* of a shared 100-element array
(`.costs_2015.s_cost`/`s_cref`/`s_k`/`s_kref`/`s_cost_factor`), one slice per one of 13
sibling methods in the source; cottax's `VarPath`-per-output model has no established
convention yet for several nodes co-owning disjoint slices of one array field (the
schema's existing per-species-index precedent, `ImpurityRadiationTotals`'s `imp_indices`,
solves a related but different problem -- *which* indices exist, not *several nodes*
owning different ones). Flagged as an open question, not resolved here. The functions
themselves are fully ported and harness-tested regardless -- Tier 1 needs no node, per
`schema.md`'s "skip this section... not a second design pass" and the existing
`superconductors.py`/`impurity_radiation.py` precedent (functions ported, zero nodes,
because every real call site's argument is a local inside an unwired unit -- same shape
here, just for a different reason).
"""

import jax.numpy as jnp
from functional_process.models.safe_math import safe_pow


def calculate_building_costs(
    cost_factor_buildings,
    light_build_cost_per_vol,
    tok_build_cost_per_vol,
    r_cryostat_inboard,
    z_cryostat_half_inside,
    pwpnb,
    helpow,
    r_pf_coil_outer_max,
    c_tf_total,
    n_tf_coils,
    e_tf_magnetic_stored_total_gj,
    p_plant_primary_heat_mw,
    p_plant_secondary_heat_mw,
):
    """Cost of all buildings (`.costs_2015.s_cost[0:9]`). Ports
    `Costs2015.calc_building_costs` verbatim -- a single fixed-length (9-item),
    branch-free sum-of-scaled-terms; the `for i in range(9)` in the source is a
    compile-time-constant-length loop (9 building categories are hardcoded, not
    data-dependent), unrolled here rather than looped.

    `s_label` (a string per item, reporting-only) is dropped -- not differentiable, not
    read by any other computation, see `costs_2015.md`'s JAX-difficulty flags.

    Parameters
    ----------
    cost_factor_buildings :
        Overall buildings cost scaling factor. `.costs.cost_factor_buildings`.
    light_build_cost_per_vol, tok_build_cost_per_vol :
        Reference unit costs ($/m^3) for light/shielded buildings.
        `.costs.light_build_cost_per_vol`, `.costs.tok_build_cost_per_vol`.
    r_cryostat_inboard, z_cryostat_half_inside :
        Cryostat inboard radius, half-height (m). `.fwbs.r_cryostat_inboard`,
        `.fwbs.z_cryostat_half_inside`.
    pwpnb :
        Neutral beam wall-plug power (MW). `.current_drive.pwpnb`.
    helpow :
        Cryoplant heat load at ~4.5K (W). `.heat_transport.helpow`.
    r_pf_coil_outer_max :
        Largest PF coil outer radius (m). `.pf_coil.r_pf_coil_outer_max`.
    c_tf_total, n_tf_coils :
        Total TF coil current (A), number of TF coils. `.tfcoil.c_tf_total`,
        `.tfcoil.n_tf_coils`.
    e_tf_magnetic_stored_total_gj :
        Total TF coil stored magnetic energy (GJ).
        `.tfcoil.e_tf_magnetic_stored_total_gj`.
    p_plant_primary_heat_mw, p_plant_secondary_heat_mw :
        Primary/secondary heat removed from the core (MW).
        `.heat_transport.p_plant_primary_heat_mw`, `.heat_transport.p_plant_secondary_heat_mw`.

    Returns
    -------
    tuple
        `(s_cost_factor, s_cref, s_k, s_kref, s_cost)`, each a length-9 `jnp` array
        indexed `[0:9]` -- items 0-7 are the individual buildings, item 8 is their total.
    """
    f = cost_factor_buildings * jnp.ones(9)

    s_cref = jnp.zeros(9)
    s_cref = s_cref.at[0].set(129000.0e0 * light_build_cost_per_vol)
    s_cref = s_cref.at[1].set(1100000.0e0 * tok_build_cost_per_vol)
    s_cref = s_cref.at[2].set(28000.0e0 * light_build_cost_per_vol)
    s_cref = s_cref.at[3].set(130000.0e0 * light_build_cost_per_vol)
    s_cref = s_cref.at[4].set(190000.0e0 * light_build_cost_per_vol)
    s_cref = s_cref.at[5].set(110000.0e0 * light_build_cost_per_vol)
    s_cref = s_cref.at[6].set(35000.0e0 * light_build_cost_per_vol)
    s_cref = s_cref.at[7].set(51000.0e0 * light_build_cost_per_vol)

    s_k = jnp.zeros(9)
    s_k = s_k.at[1].set((jnp.pi * r_cryostat_inboard**2) * 2.0e0 * z_cryostat_half_inside)
    s_k = s_k.at[2].set(pwpnb)
    s_k = s_k.at[3].set(helpow / 1.0e3)
    s_k = s_k.at[4].set(r_pf_coil_outer_max**2)
    s_k = s_k.at[5].set((c_tf_total / n_tf_coils) / 1.0e6)
    s_k = s_k.at[6].set(e_tf_magnetic_stored_total_gj)
    s_k = s_k.at[7].set(p_plant_primary_heat_mw + p_plant_secondary_heat_mw)

    s_kref = jnp.zeros(9)
    s_kref = s_kref.at[1].set(18712.0e0)
    s_kref = s_kref.at[2].set(120.0e0)
    s_kref = s_kref.at[3].set(61.0e0)
    s_kref = s_kref.at[4].set(12.4e0**2)
    s_kref = s_kref.at[5].set(9.1e0)
    s_kref = s_kref.at[6].set(41.0e0)
    s_kref = s_kref.at[7].set(880.0e0)

    safe_kref = jnp.where(s_kref == 0.0, 1.0, s_kref)  # noqa: RUF069
    ratio = jnp.where(s_kref == 0.0, 1.0, s_k / safe_kref)  # noqa: RUF069
    s_cost = jnp.where(
        (jnp.arange(9) == 0) | (jnp.arange(9) == 8),
        f * s_cref,
        f * s_cref * ratio,
    )
    s_cost = s_cost.at[8].set(jnp.sum(s_cost[:8]))

    return f, s_cref, s_k, s_kref, s_cost


def calculate_land_costs(
    cost_factor_land,
    r_cryostat_inboard,
    costexp,
    dh_tf_inner_bore,
    dr_tf_inner_bore,
    dr_tf_inboard,
):
    """Cost of land (`.costs_2015.s_cost[9:13]`). Ports `Costs2015.calc_land_costs`
    verbatim -- fixed-length (4-item), branch-free.

    Parameters
    ----------
    cost_factor_land :
        Overall land cost scaling factor. `.costs.cost_factor_land`.
    r_cryostat_inboard :
        Cryostat inboard radius (m). `.fwbs.r_cryostat_inboard`.
    costexp :
        Cost scaling exponent. `.costs.costexp`.
    dh_tf_inner_bore, dr_tf_inner_bore, dr_tf_inboard :
        TF coil bore height/radius, inboard leg thickness (m). `.build.dh_tf_inner_bore`,
        `.build.dr_tf_inner_bore`, `.build.dr_tf_inboard`.

    Returns
    -------
    tuple
        `(s_cost_factor, s_cref, s_k, s_kref, s_cost)`, each a length-4 `jnp` array
        indexed `[9:13]` -- items 9-11 are land purchasing/improvement/roads, item 12 is
        their total (`range(9, 12)` in the source is `[9, 10, 11]`, i.e. all three).
    """
    iter_total_land_area = 180.0e0
    iter_key_buildings_land_area = 42.0e0
    iter_buffer_land_area = iter_total_land_area - iter_key_buildings_land_area

    f = cost_factor_land * jnp.ones(4)  # indices 9, 10, 11, 12 -> local 0..3

    area_cryostat = jnp.pi * r_cryostat_inboard**2

    s_cref = jnp.array([
        318000.0e0,  # 9: land purchasing, $/hectare
        214.0e6,  # 10: land improvement
        150.0e6,  # 11: road improvements
        0.0e0,  # 12: total, unused
    ])
    s_k = jnp.array([
        area_cryostat,  # 9
        area_cryostat,  # 10
        jnp.maximum(dh_tf_inner_bore, dr_tf_inner_bore) + 2.0e0 * dr_tf_inboard,  # 11
        0.0e0,
    ])
    s_kref = jnp.array([638.0e0, 638.0e0, 14.0e0, 0.0e0])

    s_cost_9 = f[0] * s_cref[0] * (
        iter_key_buildings_land_area * safe_pow(s_k[0] / s_kref[0], costexp)
        + iter_buffer_land_area
    )
    s_cost_10 = f[1] * safe_pow(s_k[1] / s_kref[1], costexp) * s_cref[1]
    s_cost_11 = f[2] * s_cref[2] * (s_k[2] / s_kref[2]) ** costexp
    s_cost_12 = s_cost_9 + s_cost_10 + s_cost_11  # range(9, 12) == [9, 10, 11], all three

    s_cost = jnp.stack([s_cost_9, s_cost_10, s_cost_11, s_cost_12])

    return f, s_cref, s_k, s_kref, s_cost
