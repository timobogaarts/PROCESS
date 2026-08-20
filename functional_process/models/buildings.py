"""Pure-functional port of `process/models/buildings.py`'s `Buildings.run()` (unit #15).

Audit record: `functional_process/models/buildings.md`. `run()` is a short preamble
(TF coil envelope geometry, unconditional) followed by a dispatch, keyed by
`.buildings.i_bldgs_size`, to exactly one of `bldgs` (`BuildingsModel.ITER_1992`, the
legacy model) or `bldgs_sizes` (`BuildingsModel.CHAPMAN_2024`). All three pieces are
tier-1: no internal solve, no calls into another `Model`, no loops.

- `calculate_tf_coil_envelope` -- `run()`'s own unconditional body. Feeds both branches
  below (`tfro`/`tfri`/`tf_vertical_dim`/`tfmtn` into `bldgs`, `tf_radial_dim`/
  `tf_vertical_dim` into `bldgs_sizes`).
- `calculate_bldgs` -- the `ITER_1992` branch. Returns both what `bldgs()` writes onto
  `self.data.buildings.*` directly (`wrbi`, `a_plant_floor_effective`, `admvol`,
  `shovol`, `convol`, `volnucb`) and its own return tuple (`cryv`, `vrci`, `rbv`,
  `rmbv`, `wsv`, `elev`) -- which `run()` (not `bldgs()` itself) stores onto
  `.buildings.cryvol`/`.volrci`/`.rbvol`/`.rmbvol`/`.wsvol`/`.elevol`. See the audit
  record's data-footprint table for why those two naming layers differ.
- `calculate_bldgs_sizes` -- the `CHAPMAN_2024` branch. Only 5 of its many locals are
  real `data` writes (`reactor_hall_l`/`_w`/`_h`, `a_plant_floor_effective`,
  `volnucb`); everything else accumulated along the way (footprints/volumes of ~25
  individual buildings) is consumed only by the source's `if output:` reporting block
  and is not returned here -- same "local-intermediate, not a port" reasoning as
  `build.py`'s `awall`.

Two switches, both kept static/traced rather than split (deliberate policy deviations,
documented in the audit record):
- `is_neutral_beam` (`bldgs_sizes` only) -- derived outside the traced function from
  `.current_drive.i_hcd_primary` via `CurrentDriveModel(...).method ==
  CurrentDriveMethodType.NEUTRAL_BEAM`, since that enum lookup cannot itself be traced.
  A static field on `BldgsSizes`, same shape as `density_limits.py`'s
  `EcrhDensityLimit.i_plasma_pedestal`.
- `i_tf_sup` (`bldgs_sizes`'s centre-post branch only) -- an ordinary traced argument
  selected with `jnp.where`, matching `unit_registry.md`'s `.physics.itart` precedent.

One real PROCESS bug found, not fixed (see the audit record's "open questions"):
`calculate_bldgs_sizes`'s inboard/outboard shield-blanket-first-wall hot-cell storage
divides `life_plant` by itself (`hcomp_req_supply`), always yielding exactly `1.0`
rather than scaling by a per-component replacement lifetime the way the divertor/
centre-post calculations two sections later correctly do.
"""

import equinox as eqx
import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    FromExactly,
    Output,
)

from functional_process.models.safe_math import safe_pow
from process.models.physics.current_drive import (
    CurrentDriveMethodType,
    CurrentDriveModel,
)


def _safe_ratio(numerator, denominator):
    """`numerator / denominator`, with the denominator floored to keep the gradient of
    the *unselected* branch of a caller's `jnp.where` finite when `denominator == 0`.

    Same pattern as `physics_A_pure_formulas.md`'s `phyaux`/`fast_alpha_beta` guards:
    the value is only ever used where `denominator != 0` is true (a caller wraps this
    in an outer `jnp.where`), but `jax.jacfwd` differentiates both branches of a
    `jnp.where`, so a literal `0/0` here would leak a NaN gradient into the selected
    branch even though the *value* is masked out correctly.
    """
    safe_denominator = jnp.where(denominator != 0.0, denominator, 1.0)
    return numerator / safe_denominator


def calculate_tf_coil_envelope(
    r_tf_outboard_mid,
    dr_tf_outboard,
    r_tf_inboard_mid,
    dr_tf_inboard,
    z_tf_inside_half,
    m_tf_coils_total,
    n_tf_coils,
):
    """TF coil envelope geometry -- `Buildings.run()`'s unconditional preamble.

    Feeds both `calculate_bldgs` (`tfro`, `tfri`, `tf_vertical_dim`, `tfmtn`) and
    `calculate_bldgs_sizes` (`tf_radial_dim`, `tf_vertical_dim`). None of its outputs
    are ever stored to `data` -- they are locals in `run()`, passed straight to
    whichever building-size model is selected.

    Parameters
    ----------
    r_tf_outboard_mid, dr_tf_outboard :
        Outboard TF coil mid-leg radial position and half-thickness (m).
        `.build.r_tf_outboard_mid`, `.build.dr_tf_outboard`.
    r_tf_inboard_mid, dr_tf_inboard :
        Inboard counterparts (m). `.build.r_tf_inboard_mid`, `.build.dr_tf_inboard`.
    z_tf_inside_half :
        Half-height inside the TF coil (m). `.build.z_tf_inside_half`.
    m_tf_coils_total, n_tf_coils :
        Total TF coil set mass (kg), number of TF coils. `.tfcoil.m_tf_coils_total`,
        `.tfcoil.n_tf_coils`.

    Returns
    -------
    :
        `(tfro, tfri, tf_radial_dim, tf_vertical_dim, tfmtn)` -- outer/inner TF coil
        radius (m), TF coil radial width (m), full TF coil height (m), mass of one TF
        coil (tonne).
    """
    tfro = r_tf_outboard_mid + dr_tf_outboard * 0.5
    tfri = r_tf_inboard_mid - dr_tf_inboard * 0.5
    tf_radial_dim = tfro - tfri
    tf_vertical_dim = 2.0 * (z_tf_inside_half + dr_tf_outboard)
    tfmtn = 1.0e-3 * m_tf_coils_total / n_tf_coils
    return tfro, tfri, tf_radial_dim, tf_vertical_dim, tfmtn


def calculate_shield_height(z_tf_inside_half, dz_shld_vv_gap, dz_vv_upper, dz_vv_lower):
    """Attached-shield height passed to `bldgs` -- an inline expression at `run()`'s
    call site, not a stored `data` field (see `run()`: the `shh` positional argument to
    `self.bldgs(...)` is `2.0*(z_tf_inside_half - dz_shld_vv_gap) - dz_vv_upper -
    dz_vv_lower`, computed at the call site rather than assigned anywhere first).

    Parameters
    ----------
    z_tf_inside_half :
        Half-height inside the TF coil (m). `.build.z_tf_inside_half`.
    dz_shld_vv_gap :
        Shield-to-vacuum-vessel gap (m). `.build.dz_shld_vv_gap`.
    dz_vv_upper, dz_vv_lower :
        Upper/lower vacuum vessel thickness (m). `.build.dz_vv_upper`,
        `.build.dz_vv_lower`.

    Returns
    -------
    :
        Attached shield height (m), `shh` in `bldgs`'s own vocabulary.
    """
    return 2.0 * (z_tf_inside_half - dz_shld_vv_gap) - dz_vv_upper - dz_vv_lower


def calculate_bldgs(
    pfr,
    pfm,
    tfro,
    tfri,
    tfh,
    tfm,
    n_tf_coils,
    shro,
    shri,
    shh,
    shm,
    crr,
    helpow,
    rxcl,
    trcl,
    row,
    wgt,
    shmf,
    clh2,
    dz_tf_cryostat,
    stcl,
    rbvfac,
    rbwt,
    rbrt,
    fndt,
    hcwt,
    hccl,
    wgt2,
    mbvfac,
    wsvfac,
    tfcbv,
    pfbldgm3,
    esbldgm3,
    pibv,
    triv,
    conv,
    admv,
    shov,
):
    """Plant building sizes, `BuildingsModel.ITER_1992` branch.

    Ports `Buildings.bldgs`. See module docstring for why the return tuple's mapping
    onto `data` fields is split across two naming layers.

    Parameters
    ----------
    pfr, pfm :
        Largest PF coil outer radius (m), largest PF coil mass (tonne).
        `.pf_coil.r_pf_coil_outer_max`, `.pf_coil.m_pf_coil_max`.
    tfro, tfri, tfh, tfm :
        TF coil outer/inner radius (m), full height (m), one-coil mass (tonne) --
        `calculate_tf_coil_envelope`'s outputs.
    n_tf_coils :
        Number of TF coils. `.tfcoil.n_tf_coils`.
    shro, shri, shh, shm :
        Attached shield outer/inner radius (m), height (m), total mass (kg).
        `.build.r_shld_outboard_outer`, `.build.r_shld_inboard_inner`, a derived
        shield-height expression, `.fwbs.whtshld`.
    crr :
        Outer radius of the common cryostat (m). `.fwbs.r_cryostat_inboard`.
    helpow :
        Total cryogenic load (W). `.heat_transport.helpow`.
    rxcl, trcl, row :
        Clearance around reactor, transportation clearance, crane-operation clearance
        (m). `.buildings.rxcl`, `.buildings.trcl`, `.buildings.row`.
    wgt, shmf :
        Reactor-building crane capacity (kg, 0 = calculated), shield-mass-per-coil lift
        fraction. `.buildings.wgt`, `.buildings.shmf`.
    clh2, dz_tf_cryostat, stcl :
        Clearance beneath TF coil to foundation (m), TF-coil-to-cryostat clearance (m),
        crane-to-roof clearance (m). `.buildings.clh2`, `.buildings.dz_tf_cryostat`,
        `.buildings.stcl`.
    rbvfac, rbwt, rbrt, fndt :
        Reactor building volume factor, wall thickness (m), roof thickness (m),
        foundation thickness (m). `.buildings.rbvfac`, `.rbwt`, `.rbrt`, `.fndt`.
    hcwt, hccl :
        Hot cell wall thickness (m), hot cell component clearance (m).
        `.buildings.hcwt`, `.buildings.hccl`.
    wgt2, mbvfac, wsvfac :
        Hot cell crane capacity (kg, 0 = calculated), maintenance building volume
        factor, warm shop volume factor. `.buildings.wgt2`, `.mbvfac`, `.wsvfac`.
    tfcbv, pfbldgm3, esbldgm3, pibv :
        TF coil PSU, PF coil PSU, energy storage, power injection building volumes
        (m3). `.buildings.tfcbv`, `.pfbldgm3`, `.esbldgm3`, `.pibv`.
    triv, conv, admv, shov :
        Tritium, control, administration, shops building volumes (m3).
        `.buildings.triv`, `.conv`, `.admv`, `.shov`.

    Returns
    -------
    :
        `(cryv, vrci, rbv, rmbv, wsv, elev, wrbi, a_plant_floor_effective, admvol,
        shovol, convol, volnucb)`.
    """
    bmr = jnp.maximum(jnp.maximum(crr, pfr), tfro)

    sectl = shro - shri
    coill = tfro - tfri
    sectl = jnp.maximum(coill, sectl)

    wrbi = bmr + rxcl + sectl + trcl + row

    layl = jnp.maximum(crr, pfr)
    hy = bmr + rxcl + sectl + trcl + layl
    ang = (wrbi - trcl - layl) / hy
    ang = jnp.clip(ang, -1.0, 1.0)

    drbi = trcl + layl + hy * jnp.sin(jnp.arccos(ang)) + wrbi

    wt_default = shmf * shm / n_tf_coils
    wt_default = jnp.maximum(wt_default, jnp.maximum(1.0e3 * pfm, 1.0e3 * tfm))
    wt = jnp.where(wgt > 1.0, wgt, wt_default)

    crcl = 9.41e-6 * wt + 5.1

    hrbi = clh2 + 2.0 * tfh + dz_tf_cryostat + trcl + crcl + stcl

    vrci = rbvfac * 2.0 * wrbi * drbi * hrbi
    vrci = jnp.where(jnp.isinf(vrci), 1.0e10, vrci)

    rbw = 2.0 * wrbi + 2.0 * rbwt
    rbl = drbi + 2.0 * rbwt
    rbh = hrbi + rbrt + fndt
    rbv = rbvfac * rbw * rbl * rbh

    tcw = shro - shri + 4.0 * trcl
    tcl = 5.0 * tcw + 2.0 * hcwt

    dcw = 2.0 * tcw + 1.0

    hcw = shro - shri + 3.0 * hccl + 2.0
    hcl = 3.0 * (shro - shri) + 4.0 * hccl + tcw

    rmbw = hcw + dcw + 3.0 * hcwt
    rmbl = hcl + 2.0 * hcwt

    wgts_default = shmf * shm / n_tf_coils
    wgts = jnp.where(wgt2 > 1.0, wgt2, wgts_default)

    cran = 9.41e-6 * wgts + 5.1
    rmbh = 10.0 + shh + trcl + cran + stcl + fndt
    tch = shh + stcl + fndt

    rmbv = mbvfac * rmbw * rmbl * rmbh + tcw * tcl * tch

    wsa = (rmbw + 7.0) * 20.0 + rmbl * 7.0
    wsv = wsvfac * wsa * rmbh

    cryv = 55.0 * safe_pow(helpow, 0.5)

    elev = tfcbv + pfbldgm3 + esbldgm3 + pibv

    a_plant_floor_effective = (
        rbv + rmbv + wsv + triv + elev + conv + cryv + admv + shov
    ) / 6.0

    admvol = admv
    shovol = shov
    convol = conv

    volnucb = vrci + rmbv + wsv + triv + cryv

    return (
        cryv,
        vrci,
        rbv,
        rmbv,
        wsv,
        elev,
        wrbi,
        a_plant_floor_effective,
        admvol,
        shovol,
        convol,
        volnucb,
    )


def calculate_bldgs_sizes(
    r_pf_coil_outer_max,
    r_cryostat_inboard,
    tf_radial_dim,
    bioshld_thk,
    reactor_clrnc,
    transp_clrnc,
    crane_clrnc_h,
    cryostat_clrnc,
    ground_clrnc,
    crane_arm_h,
    tf_vertical_dim,
    is_neutral_beam,
    nbi_sys_l,
    nbi_sys_w,
    hcd_building_l,
    hcd_building_w,
    hcd_building_h,
    fc_building_l,
    fc_building_w,
    reactor_wall_thk,
    reactor_roof_thk,
    reactor_fndtn_thk,
    life_plant,
    z_tf_inside_half,
    dr_tf_inboard,
    dr_tf_shld_gap,
    dz_shld_thermal,
    dz_shld_vv_gap,
    dr_shld_inboard,
    dr_blkt_inboard,
    dr_fw_inboard,
    rmajor,
    rminor,
    dr_fw_plasma_gap_inboard,
    n_tf_coils,
    hot_sepdist,
    qnty_sfty_fac,
    dr_fw_outboard,
    dr_blkt_outboard,
    dr_shld_outboard,
    dr_fw_plasma_gap_outboard,
    life_div_fpy,
    dz_divertor,
    cplife,
    i_tf_sup,
    r_cp_top,
    hotcell_h,
    chemlab_l,
    chemlab_w,
    chemlab_h,
    heat_sink_l,
    heat_sink_w,
    heat_sink_h,
    aux_build_l,
    aux_build_w,
    aux_build_h,
    magnet_trains_l,
    magnet_trains_w,
    magnet_trains_h,
    magnet_pulse_l,
    magnet_pulse_w,
    magnet_pulse_h,
    control_buildings_l,
    control_buildings_w,
    control_buildings_h,
    warm_shop_l,
    warm_shop_w,
    warm_shop_h,
    workshop_l,
    workshop_w,
    workshop_h,
    robotics_l,
    robotics_w,
    robotics_h,
    maint_cont_l,
    maint_cont_w,
    maint_cont_h,
    cryomag_l,
    cryomag_w,
    cryomag_h,
    cryostore_l,
    cryostore_w,
    cryostore_h,
    auxcool_l,
    auxcool_w,
    auxcool_h,
    elecdist_l,
    elecdist_w,
    elecdist_h,
    elecload_l,
    elecload_w,
    elecload_h,
    elecstore_l,
    elecstore_w,
    elecstore_h,
    turbine_hall_l,
    turbine_hall_w,
    turbine_hall_h,
    ilw_smelter_l,
    ilw_smelter_w,
    ilw_smelter_h,
    ilw_storage_l,
    ilw_storage_w,
    ilw_storage_h,
    llw_storage_l,
    llw_storage_w,
    llw_storage_h,
    hw_storage_l,
    hw_storage_w,
    hw_storage_h,
    tw_storage_l,
    tw_storage_w,
    tw_storage_h,
    gas_buildings_l,
    gas_buildings_w,
    gas_buildings_h,
    water_buildings_l,
    water_buildings_w,
    water_buildings_h,
    sec_buildings_l,
    sec_buildings_w,
    sec_buildings_h,
    staff_buildings_area,
    staff_buildings_h,
):
    """Plant building sizes, `BuildingsModel.CHAPMAN_2024` branch.

    Ports `Buildings.bldgs_sizes`. Only 5 of the many locals below are real `data`
    writes (see the `Returns` section) -- the rest (each building's own footprint/
    volume) exist only to accumulate into `buildings_total_vol`/`hotcell_vol_ext` and
    to feed the source's reporting block, which is out of port scope. Parameters are
    grouped by the same sections the source uses; see `buildings.md`'s data-footprint
    table for the exhaustive per-field `VarPath` mapping.

    A genuine PROCESS bug is reproduced faithfully here, not fixed: the inboard/
    outboard shield-blanket-first-wall `hcomp_req_supply` divides `life_plant` by
    itself (always `1.0` whenever `life_plant != 0`) rather than by a per-component
    replacement lifetime, unlike the divertor/centre-post calculations below it. See
    `buildings.md`'s open questions.

    Parameters
    ----------
    is_neutral_beam :
        Whether the primary HCD method is neutral-beam-shaped (`ITER_NEUTRAL_BEAM`/
        `CULHAM_NEUTRAL_BEAM`). Derived outside this function from
        `.current_drive.i_hcd_primary` -- see module docstring. Static/non-differentiated.
    i_tf_sup :
        TF coil superconductor switch. `.tfcoil.i_tf_sup`. Only used to select between
        `r_cp_top`/`dr_tf_inboard` in the centre-post branch (`jnp.where`, ordinary
        traced argument, not held static -- see `buildings.md`'s switches section).
    life_plant, life_div_fpy, cplife :
        Plant lifetime (full-power years), divertor lifetime, centre-post lifetime.
        `.costs.life_plant`, `.costs.life_div_fpy`, `.costs.cplife`. Each `== 0.0`
        disables its associated hot-cell storage sub-computation (guarded with a
        safe-divide, see `_safe_ratio`).
    Everything else :
        A plain `.buildings.*`/`.build.*`/`.physics.*`/`.tfcoil.*`/`.divertor.*` scalar
        read, named identically to its PROCESS field -- see `buildings.md`.

    Returns
    -------
    :
        `(reactor_hall_l, reactor_hall_w, reactor_hall_h, a_plant_floor_effective,
        volnucb)` -- the only 5 real `.buildings.*` writes this method makes.
    """
    # Reactor building.
    width_reactor_piece = (
        jnp.maximum(jnp.maximum(r_pf_coil_outer_max, r_cryostat_inboard), tf_radial_dim)
        + bioshld_thk
    )
    key_width = (
        2.0 * width_reactor_piece + reactor_clrnc + transp_clrnc + crane_clrnc_h
    )
    reactor_hall_w = 3.0 * key_width
    reactor_hall_l = 3.0 * key_width

    height_clrnc = (
        reactor_clrnc
        + transp_clrnc
        + cryostat_clrnc
        + ground_clrnc
        + crane_clrnc_h
        + crane_arm_h
    )
    reactor_hall_h = 2.0 * tf_vertical_dim + height_clrnc

    # Heating and Current Drive facility.
    if is_neutral_beam:
        reactor_hall_l = reactor_hall_l + nbi_sys_l + reactor_clrnc + transp_clrnc
        reactor_hall_w = reactor_hall_w + nbi_sys_w + reactor_clrnc + transp_clrnc
        hcd_building_area = 0.0
        hcd_building_vol = 0.0
    else:
        hcd_building_area = hcd_building_l * hcd_building_w
        hcd_building_vol = hcd_building_area * hcd_building_h

    # Fuel Cycle facilities.
    reactor_hall_l = reactor_hall_l + fc_building_l
    reactor_hall_w = reactor_hall_w + fc_building_w

    reactor_hall_area = reactor_hall_l * reactor_hall_w
    reactor_hall_vol = reactor_hall_area * reactor_hall_h

    reactor_building_l = reactor_hall_l + 2.0 * reactor_wall_thk
    reactor_building_w = reactor_hall_w + 2.0 * reactor_wall_thk
    reactor_building_h = reactor_hall_h + reactor_roof_thk + reactor_fndtn_thk
    reactor_building_area = reactor_building_l * reactor_building_w
    reactor_building_vol = reactor_building_area * reactor_building_h

    # Reactor maintenance basement and tunnel.
    reactor_basement_l = reactor_hall_w
    reactor_basement_w = reactor_hall_w
    reactor_basement_area = reactor_basement_l * reactor_basement_w
    reactor_basement_h = (
        tf_vertical_dim + transp_clrnc + crane_clrnc_h + crane_arm_h
    )
    reactor_basement_vol = reactor_basement_area * reactor_basement_h

    reactor_build_totvol = reactor_building_vol + reactor_basement_vol
    buildings_total_vol = reactor_hall_vol + reactor_basement_vol

    # Hot Cell Facility -- inboard shield/blanket/first-wall.
    life_plant_nonzero = life_plant != 0.0

    hcomp_height_shield = 2.0 * (
        z_tf_inside_half - (dr_tf_inboard + dr_tf_shld_gap + dz_shld_thermal + dz_shld_vv_gap)
    )

    hcomp_rad_thk_ib = dr_shld_inboard + dr_blkt_inboard + dr_fw_inboard
    hcomp_tor_thk_ib = (
        2.0
        * jnp.pi
        * (
            rmajor
            - (
                rminor
                + dr_fw_plasma_gap_inboard
                + dr_fw_inboard
                + dr_blkt_inboard
                + dr_shld_inboard
            )
        )
    ) / n_tf_coils
    hcomp_footprint_ib = (hcomp_height_shield + hot_sepdist) * (
        jnp.maximum(hcomp_rad_thk_ib, hcomp_tor_thk_ib) + hot_sepdist
    )
    hcomp_vol_ib = hcomp_footprint_ib * (
        jnp.minimum(hcomp_rad_thk_ib, hcomp_tor_thk_ib) + hot_sepdist
    )
    # Bug reproduced faithfully: divides life_plant by itself -- see docstring.
    hcomp_req_supply_ib = (
        n_tf_coils * _safe_ratio(life_plant, life_plant)
    ) * qnty_sfty_fac
    ib_hotcell_vol = jnp.where(
        life_plant_nonzero, hcomp_req_supply_ib * hcomp_vol_ib, 0.0
    )

    # Outboard first-wall/blanket/shield (same height expression as inboard -- matches
    # source, which repeats it verbatim rather than reusing a shared local).
    hcomp_rad_thk_ob = dr_fw_outboard + dr_blkt_outboard + dr_shld_outboard
    hcomp_tor_thk_ob = (
        2.0
        * jnp.pi
        * (
            rmajor
            + rminor
            + dr_fw_plasma_gap_outboard
            + dr_fw_outboard
            + dr_blkt_outboard
            + dr_shld_outboard
        )
    ) / n_tf_coils
    hcomp_footprint_ob = (hcomp_height_shield + hot_sepdist) * (
        jnp.maximum(hcomp_rad_thk_ob, hcomp_tor_thk_ob) + hot_sepdist
    )
    hcomp_vol_ob = hcomp_footprint_ob * (
        jnp.minimum(hcomp_rad_thk_ob, hcomp_tor_thk_ob) + hot_sepdist
    )
    hcomp_req_supply_ob = (
        n_tf_coils * _safe_ratio(life_plant, life_plant)
    ) * qnty_sfty_fac
    ob_hotcell_vol = jnp.where(
        life_plant_nonzero, hcomp_req_supply_ob * hcomp_vol_ob, 0.0
    )

    # Divertor.
    life_div_nonzero = life_div_fpy != 0.0
    hcomp_height_div = dz_divertor
    hcomp_rad_thk_div = 2.0 * rminor
    hcomp_tor_thk_div = rmajor + rminor
    hcomp_footprint_div = (hcomp_height_div + hot_sepdist) * (
        jnp.maximum(hcomp_rad_thk_div, hcomp_tor_thk_div) + hot_sepdist
    )
    hcomp_vol_div = hcomp_footprint_div * (
        jnp.minimum(hcomp_rad_thk_div, hcomp_tor_thk_div) + hot_sepdist
    )
    hcomp_req_supply_div = (
        n_tf_coils * _safe_ratio(life_plant, life_div_fpy)
    ) * qnty_sfty_fac
    div_hotcell_vol = jnp.where(
        life_div_nonzero, hcomp_req_supply_div * hcomp_vol_div, 0.0
    )

    # Centre post.
    cplife_nonzero = cplife != 0.0
    hcomp_height_cp = 2.0 * z_tf_inside_half
    hcomp_rad_thk_cp = jnp.where(i_tf_sup != 1, r_cp_top, dr_tf_inboard)
    hcomp_footprint_cp = (hcomp_height_cp + hot_sepdist) * (hcomp_rad_thk_cp + hot_sepdist)
    hcomp_vol_cp = hcomp_footprint_cp * (hcomp_rad_thk_cp + hot_sepdist)
    hcomp_req_supply_cp = _safe_ratio(life_plant, cplife) * qnty_sfty_fac
    cp_hotcell_vol = jnp.where(
        cplife_nonzero, hcomp_req_supply_cp * hcomp_vol_cp, 0.0
    )

    hotcell_vol = ib_hotcell_vol + ob_hotcell_vol + div_hotcell_vol + cp_hotcell_vol
    hotcell_area = hotcell_vol / hotcell_h
    hotcell_l = safe_pow(hotcell_area, 0.5)
    hotcell_w = hotcell_l
    hotcell_area_ext = (hotcell_l + 2.0 * reactor_wall_thk) * (
        hotcell_w + 2.0 * reactor_wall_thk
    )
    hotcell_vol_ext = hotcell_area_ext * (
        hotcell_h + reactor_roof_thk + reactor_fndtn_thk
    )

    buildings_total_vol = buildings_total_vol + hotcell_vol

    # Reactor Auxiliary Buildings.
    chemlab_area = chemlab_l * chemlab_w
    chemlab_vol = chemlab_area * chemlab_h
    heat_sink_area = heat_sink_l * heat_sink_w
    heat_sink_vol = heat_sink_area * heat_sink_h
    aux_build_area = aux_build_l * aux_build_w
    aux_build_vol = aux_build_area * aux_build_h
    reactor_aux_vol = chemlab_vol + heat_sink_vol + aux_build_vol
    buildings_total_vol = buildings_total_vol + reactor_aux_vol

    # Magnet power facilities.
    magnet_trains_area = magnet_trains_l * magnet_trains_w
    magnet_trains_vol = magnet_trains_area * magnet_trains_h
    magnet_pulse_area = magnet_pulse_l * magnet_pulse_w
    magnet_pulse_vol = magnet_pulse_area * magnet_pulse_h
    power_buildings_vol = hcd_building_vol + magnet_trains_vol + magnet_pulse_vol
    buildings_total_vol = buildings_total_vol + power_buildings_vol

    # Control.
    control_buildings_area = control_buildings_l * control_buildings_w
    control_buildings_vol = control_buildings_area * control_buildings_h
    buildings_total_vol = buildings_total_vol + control_buildings_vol

    # Warm Shop.
    warm_shop_area = warm_shop_l * warm_shop_w
    warm_shop_vol = warm_shop_area * warm_shop_h
    buildings_total_vol = buildings_total_vol + warm_shop_vol

    # Maintenance.
    workshop_area = workshop_l * workshop_w
    workshop_vol = workshop_area * workshop_h
    robotics_area = robotics_l * robotics_w
    robotics_vol = robotics_area * robotics_h
    maint_cont_area = maint_cont_l * maint_cont_w
    maint_cont_vol = maint_cont_area * maint_cont_h
    maintenance_vol = workshop_vol + robotics_vol + maint_cont_vol
    buildings_total_vol = buildings_total_vol + maintenance_vol

    # Cryogenic & cooling facilities.
    cryomag_area = cryomag_l * cryomag_w
    cryomag_vol = cryomag_area * cryomag_h
    cryostore_area = cryostore_l * cryostore_w
    cryostore_vol = cryostore_area * cryostore_h
    auxcool_area = auxcool_l * auxcool_w
    auxcool_vol = auxcool_area * auxcool_h
    cryocool_vol = cryomag_vol + cryostore_vol + auxcool_vol
    buildings_total_vol = buildings_total_vol + cryocool_vol

    # Electrical.
    elecdist_area = elecdist_l * elecdist_w
    elecdist_vol = elecdist_area * elecdist_h
    elecload_area = elecload_l * elecload_w
    elecload_vol = elecload_area * elecload_h
    elecstore_area = elecstore_l * elecstore_w
    elecstore_vol = elecstore_area * elecstore_h
    elec_buildings_vol = elecdist_vol + elecload_vol + elecstore_vol
    buildings_total_vol = buildings_total_vol + elec_buildings_vol

    # Turbine Hall.
    turbine_hall_area = turbine_hall_l * turbine_hall_w
    turbine_hall_vol = turbine_hall_area * turbine_hall_h
    buildings_total_vol = buildings_total_vol + turbine_hall_vol

    # Waste.
    ilw_smelter_area = ilw_smelter_l * ilw_smelter_w
    ilw_smelter_vol = ilw_smelter_area * ilw_smelter_h
    ilw_storage_area = ilw_storage_l * ilw_storage_w
    ilw_storage_vol = ilw_storage_area * ilw_storage_h
    llw_storage_area = llw_storage_l * llw_storage_w
    llw_storage_vol = llw_storage_area * llw_storage_h
    hw_storage_area = hw_storage_l * hw_storage_w
    hw_storage_vol = hw_storage_area * hw_storage_h
    tw_storage_area = tw_storage_l * tw_storage_w
    tw_storage_vol = tw_storage_area * tw_storage_h
    waste_buildings_vol = (
        ilw_smelter_vol + ilw_storage_vol + llw_storage_vol + hw_storage_vol + tw_storage_vol
    )
    buildings_total_vol = buildings_total_vol + waste_buildings_vol

    # Site Services.
    gas_buildings_area = gas_buildings_l * gas_buildings_w
    gas_buildings_vol = gas_buildings_area * gas_buildings_h
    water_buildings_area = water_buildings_l * water_buildings_w
    water_buildings_vol = water_buildings_area * water_buildings_h
    sec_buildings_area = sec_buildings_l * sec_buildings_w
    sec_buildings_vol = sec_buildings_area * sec_buildings_h
    buildings_total_vol = (
        buildings_total_vol + gas_buildings_vol + water_buildings_vol + sec_buildings_vol
    )

    # Staff Services.
    staff_buildings_vol = staff_buildings_area * staff_buildings_h
    buildings_total_vol = buildings_total_vol + staff_buildings_vol

    a_plant_floor_effective = buildings_total_vol / 6.0
    volnucb = reactor_build_totvol + hotcell_vol_ext

    return reactor_hall_l, reactor_hall_w, reactor_hall_h, a_plant_floor_effective, volnucb


class TfCoilEnvelope(ExplicitFunction):
    """cottax node: `calculate_tf_coil_envelope`, ports declared.

    No real `data` writes -- see module docstring. `tfro`/`tfri`/`tf_radial_dim`/
    `tf_vertical_dim`/`tfmtn` are minted `VarPath`s under `.buildings.*` (invented,
    same reasoning as `build.py`'s `a_fw_total_unadjusted`: PROCESS never stores these,
    they are locals in `run()`), so that this node has somewhere to write its outputs
    for `Bldgs`/`BldgsSizes` to read from.
    """

    tfro = Output(lambda s: s.buildings.tfro)
    tfri = Output(lambda s: s.buildings.tfri)
    tf_radial_dim = Output(lambda s: s.buildings.tf_radial_dim)
    tf_vertical_dim = Output(lambda s: s.buildings.tf_vertical_dim)
    tfmtn = Output(lambda s: s.buildings.tfmtn)

    def __call__(
        self,
        r_tf_outboard_mid=FromExactly(lambda s: s.build.r_tf_outboard_mid),
        dr_tf_outboard=FromExactly(lambda s: s.build.dr_tf_outboard),
        r_tf_inboard_mid=FromExactly(lambda s: s.build.r_tf_inboard_mid),
        dr_tf_inboard=FromExactly(lambda s: s.build.dr_tf_inboard),
        z_tf_inside_half=FromExactly(lambda s: s.build.z_tf_inside_half),
        m_tf_coils_total=FromExactly(lambda s: s.tfcoil.m_tf_coils_total),
        n_tf_coils=FromExactly(lambda s: s.tfcoil.n_tf_coils),
    ):
        return calculate_tf_coil_envelope(
            r_tf_outboard_mid,
            dr_tf_outboard,
            r_tf_inboard_mid,
            dr_tf_inboard,
            z_tf_inside_half,
            m_tf_coils_total,
            n_tf_coils,
        )


class Bldgs(ExplicitFunction):
    """cottax node: `calculate_bldgs`. Instantiate iff `i_bldgs_size == ITER_1992`."""

    cryvol = Output(lambda s: s.buildings.cryvol)
    volrci = Output(lambda s: s.buildings.volrci)
    rbvol = Output(lambda s: s.buildings.rbvol)
    rmbvol = Output(lambda s: s.buildings.rmbvol)
    wsvol = Output(lambda s: s.buildings.wsvol)
    elevol = Output(lambda s: s.buildings.elevol)
    wrbi = Output(lambda s: s.buildings.wrbi)
    a_plant_floor_effective = Output(lambda s: s.buildings.a_plant_floor_effective)
    admvol = Output(lambda s: s.buildings.admvol)
    shovol = Output(lambda s: s.buildings.shovol)
    convol = Output(lambda s: s.buildings.convol)
    volnucb = Output(lambda s: s.buildings.volnucb)

    def __call__(
        self,
        pfr=FromExactly(lambda s: s.pf_coil.r_pf_coil_outer_max),
        pfm=FromExactly(lambda s: s.pf_coil.m_pf_coil_max),
        tfro=FromExactly(lambda s: s.buildings.tfro),
        tfri=FromExactly(lambda s: s.buildings.tfri),
        tfh=FromExactly(lambda s: s.buildings.tf_vertical_dim),
        tfm=FromExactly(lambda s: s.buildings.tfmtn),
        n_tf_coils=FromExactly(lambda s: s.tfcoil.n_tf_coils),
        shro=FromExactly(lambda s: s.build.r_shld_outboard_outer),
        shri=FromExactly(lambda s: s.build.r_shld_inboard_inner),
        z_tf_inside_half=FromExactly(lambda s: s.build.z_tf_inside_half),
        dz_shld_vv_gap=FromExactly(lambda s: s.build.dz_shld_vv_gap),
        dz_vv_upper=FromExactly(lambda s: s.build.dz_vv_upper),
        dz_vv_lower=FromExactly(lambda s: s.build.dz_vv_lower),
        shm=FromExactly(lambda s: s.fwbs.whtshld),
        crr=FromExactly(lambda s: s.fwbs.r_cryostat_inboard),
        helpow=FromExactly(lambda s: s.heat_transport.helpow),
        rxcl=FromExactly(lambda s: s.buildings.rxcl),
        trcl=FromExactly(lambda s: s.buildings.trcl),
        row=FromExactly(lambda s: s.buildings.row),
        wgt=FromExactly(lambda s: s.buildings.wgt),
        shmf=FromExactly(lambda s: s.buildings.shmf),
        clh2=FromExactly(lambda s: s.buildings.clh2),
        dz_tf_cryostat=FromExactly(lambda s: s.buildings.dz_tf_cryostat),
        stcl=FromExactly(lambda s: s.buildings.stcl),
        rbvfac=FromExactly(lambda s: s.buildings.rbvfac),
        rbwt=FromExactly(lambda s: s.buildings.rbwt),
        rbrt=FromExactly(lambda s: s.buildings.rbrt),
        fndt=FromExactly(lambda s: s.buildings.fndt),
        hcwt=FromExactly(lambda s: s.buildings.hcwt),
        hccl=FromExactly(lambda s: s.buildings.hccl),
        wgt2=FromExactly(lambda s: s.buildings.wgt2),
        mbvfac=FromExactly(lambda s: s.buildings.mbvfac),
        wsvfac=FromExactly(lambda s: s.buildings.wsvfac),
        tfcbv=FromExactly(lambda s: s.buildings.tfcbv),
        pfbldgm3=FromExactly(lambda s: s.buildings.pfbldgm3),
        esbldgm3=FromExactly(lambda s: s.buildings.esbldgm3),
        pibv=FromExactly(lambda s: s.buildings.pibv),
        triv=FromExactly(lambda s: s.buildings.triv),
        conv=FromExactly(lambda s: s.buildings.conv),
        admv=FromExactly(lambda s: s.buildings.admv),
        shov=FromExactly(lambda s: s.buildings.shov),
    ):
        shh = calculate_shield_height(
            z_tf_inside_half, dz_shld_vv_gap, dz_vv_upper, dz_vv_lower
        )
        return calculate_bldgs(
            pfr,
            pfm,
            tfro,
            tfri,
            tfh,
            tfm,
            n_tf_coils,
            shro,
            shri,
            shh,
            shm,
            crr,
            helpow,
            rxcl,
            trcl,
            row,
            wgt,
            shmf,
            clh2,
            dz_tf_cryostat,
            stcl,
            rbvfac,
            rbwt,
            rbrt,
            fndt,
            hcwt,
            hccl,
            wgt2,
            mbvfac,
            wsvfac,
            tfcbv,
            pfbldgm3,
            esbldgm3,
            pibv,
            triv,
            conv,
            admv,
            shov,
        )


class BldgsSizes(ExplicitFunction):
    """cottax node: `calculate_bldgs_sizes`. Instantiate iff `i_bldgs_size ==
    CHAPMAN_2024`.

    `i_hcd_primary` is a static field (not an `FromExactly`) -- see module docstring.
    """

    i_hcd_primary: CurrentDriveModel = eqx.field(static=True)

    reactor_hall_l = Output(lambda s: s.buildings.reactor_hall_l)
    reactor_hall_w = Output(lambda s: s.buildings.reactor_hall_w)
    reactor_hall_h = Output(lambda s: s.buildings.reactor_hall_h)
    a_plant_floor_effective = Output(lambda s: s.buildings.a_plant_floor_effective)
    volnucb = Output(lambda s: s.buildings.volnucb)

    def __call__(
        self,
        r_pf_coil_outer_max=FromExactly(lambda s: s.pf_coil.r_pf_coil_outer_max),
        r_cryostat_inboard=FromExactly(lambda s: s.fwbs.r_cryostat_inboard),
        tf_radial_dim=FromExactly(lambda s: s.buildings.tf_radial_dim),
        bioshld_thk=FromExactly(lambda s: s.buildings.bioshld_thk),
        reactor_clrnc=FromExactly(lambda s: s.buildings.reactor_clrnc),
        transp_clrnc=FromExactly(lambda s: s.buildings.transp_clrnc),
        crane_clrnc_h=FromExactly(lambda s: s.buildings.crane_clrnc_h),
        cryostat_clrnc=FromExactly(lambda s: s.buildings.cryostat_clrnc),
        ground_clrnc=FromExactly(lambda s: s.buildings.ground_clrnc),
        crane_arm_h=FromExactly(lambda s: s.buildings.crane_arm_h),
        tf_vertical_dim=FromExactly(lambda s: s.buildings.tf_vertical_dim),
        nbi_sys_l=FromExactly(lambda s: s.buildings.nbi_sys_l),
        nbi_sys_w=FromExactly(lambda s: s.buildings.nbi_sys_w),
        hcd_building_l=FromExactly(lambda s: s.buildings.hcd_building_l),
        hcd_building_w=FromExactly(lambda s: s.buildings.hcd_building_w),
        hcd_building_h=FromExactly(lambda s: s.buildings.hcd_building_h),
        fc_building_l=FromExactly(lambda s: s.buildings.fc_building_l),
        fc_building_w=FromExactly(lambda s: s.buildings.fc_building_w),
        reactor_wall_thk=FromExactly(lambda s: s.buildings.reactor_wall_thk),
        reactor_roof_thk=FromExactly(lambda s: s.buildings.reactor_roof_thk),
        reactor_fndtn_thk=FromExactly(lambda s: s.buildings.reactor_fndtn_thk),
        life_plant=FromExactly(lambda s: s.costs.life_plant),
        z_tf_inside_half=FromExactly(lambda s: s.build.z_tf_inside_half),
        dr_tf_inboard=FromExactly(lambda s: s.build.dr_tf_inboard),
        dr_tf_shld_gap=FromExactly(lambda s: s.build.dr_tf_shld_gap),
        dz_shld_thermal=FromExactly(lambda s: s.build.dz_shld_thermal),
        dz_shld_vv_gap=FromExactly(lambda s: s.build.dz_shld_vv_gap),
        dr_shld_inboard=FromExactly(lambda s: s.build.dr_shld_inboard),
        dr_blkt_inboard=FromExactly(lambda s: s.build.dr_blkt_inboard),
        dr_fw_inboard=FromExactly(lambda s: s.build.dr_fw_inboard),
        rmajor=FromExactly(lambda s: s.physics.rmajor),
        rminor=FromExactly(lambda s: s.physics.rminor),
        dr_fw_plasma_gap_inboard=FromExactly(lambda s: s.build.dr_fw_plasma_gap_inboard),
        n_tf_coils=FromExactly(lambda s: s.tfcoil.n_tf_coils),
        hot_sepdist=FromExactly(lambda s: s.buildings.hot_sepdist),
        qnty_sfty_fac=FromExactly(lambda s: s.buildings.qnty_sfty_fac),
        dr_fw_outboard=FromExactly(lambda s: s.build.dr_fw_outboard),
        dr_blkt_outboard=FromExactly(lambda s: s.build.dr_blkt_outboard),
        dr_shld_outboard=FromExactly(lambda s: s.build.dr_shld_outboard),
        dr_fw_plasma_gap_outboard=FromExactly(lambda s: s.build.dr_fw_plasma_gap_outboard),
        life_div_fpy=FromExactly(lambda s: s.costs.life_div_fpy),
        dz_divertor=FromExactly(lambda s: s.divertor.dz_divertor),
        cplife=FromExactly(lambda s: s.costs.cplife),
        i_tf_sup=FromExactly(lambda s: s.tfcoil.i_tf_sup),
        r_cp_top=FromExactly(lambda s: s.build.r_cp_top),
        hotcell_h=FromExactly(lambda s: s.buildings.hotcell_h),
        chemlab_l=FromExactly(lambda s: s.buildings.chemlab_l),
        chemlab_w=FromExactly(lambda s: s.buildings.chemlab_w),
        chemlab_h=FromExactly(lambda s: s.buildings.chemlab_h),
        heat_sink_l=FromExactly(lambda s: s.buildings.heat_sink_l),
        heat_sink_w=FromExactly(lambda s: s.buildings.heat_sink_w),
        heat_sink_h=FromExactly(lambda s: s.buildings.heat_sink_h),
        aux_build_l=FromExactly(lambda s: s.buildings.aux_build_l),
        aux_build_w=FromExactly(lambda s: s.buildings.aux_build_w),
        aux_build_h=FromExactly(lambda s: s.buildings.aux_build_h),
        magnet_trains_l=FromExactly(lambda s: s.buildings.magnet_trains_l),
        magnet_trains_w=FromExactly(lambda s: s.buildings.magnet_trains_w),
        magnet_trains_h=FromExactly(lambda s: s.buildings.magnet_trains_h),
        magnet_pulse_l=FromExactly(lambda s: s.buildings.magnet_pulse_l),
        magnet_pulse_w=FromExactly(lambda s: s.buildings.magnet_pulse_w),
        magnet_pulse_h=FromExactly(lambda s: s.buildings.magnet_pulse_h),
        control_buildings_l=FromExactly(lambda s: s.buildings.control_buildings_l),
        control_buildings_w=FromExactly(lambda s: s.buildings.control_buildings_w),
        control_buildings_h=FromExactly(lambda s: s.buildings.control_buildings_h),
        warm_shop_l=FromExactly(lambda s: s.buildings.warm_shop_l),
        warm_shop_w=FromExactly(lambda s: s.buildings.warm_shop_w),
        warm_shop_h=FromExactly(lambda s: s.buildings.warm_shop_h),
        workshop_l=FromExactly(lambda s: s.buildings.workshop_l),
        workshop_w=FromExactly(lambda s: s.buildings.workshop_w),
        workshop_h=FromExactly(lambda s: s.buildings.workshop_h),
        robotics_l=FromExactly(lambda s: s.buildings.robotics_l),
        robotics_w=FromExactly(lambda s: s.buildings.robotics_w),
        robotics_h=FromExactly(lambda s: s.buildings.robotics_h),
        maint_cont_l=FromExactly(lambda s: s.buildings.maint_cont_l),
        maint_cont_w=FromExactly(lambda s: s.buildings.maint_cont_w),
        maint_cont_h=FromExactly(lambda s: s.buildings.maint_cont_h),
        cryomag_l=FromExactly(lambda s: s.buildings.cryomag_l),
        cryomag_w=FromExactly(lambda s: s.buildings.cryomag_w),
        cryomag_h=FromExactly(lambda s: s.buildings.cryomag_h),
        cryostore_l=FromExactly(lambda s: s.buildings.cryostore_l),
        cryostore_w=FromExactly(lambda s: s.buildings.cryostore_w),
        cryostore_h=FromExactly(lambda s: s.buildings.cryostore_h),
        auxcool_l=FromExactly(lambda s: s.buildings.auxcool_l),
        auxcool_w=FromExactly(lambda s: s.buildings.auxcool_w),
        auxcool_h=FromExactly(lambda s: s.buildings.auxcool_h),
        elecdist_l=FromExactly(lambda s: s.buildings.elecdist_l),
        elecdist_w=FromExactly(lambda s: s.buildings.elecdist_w),
        elecdist_h=FromExactly(lambda s: s.buildings.elecdist_h),
        elecload_l=FromExactly(lambda s: s.buildings.elecload_l),
        elecload_w=FromExactly(lambda s: s.buildings.elecload_w),
        elecload_h=FromExactly(lambda s: s.buildings.elecload_h),
        elecstore_l=FromExactly(lambda s: s.buildings.elecstore_l),
        elecstore_w=FromExactly(lambda s: s.buildings.elecstore_w),
        elecstore_h=FromExactly(lambda s: s.buildings.elecstore_h),
        turbine_hall_l=FromExactly(lambda s: s.buildings.turbine_hall_l),
        turbine_hall_w=FromExactly(lambda s: s.buildings.turbine_hall_w),
        turbine_hall_h=FromExactly(lambda s: s.buildings.turbine_hall_h),
        ilw_smelter_l=FromExactly(lambda s: s.buildings.ilw_smelter_l),
        ilw_smelter_w=FromExactly(lambda s: s.buildings.ilw_smelter_w),
        ilw_smelter_h=FromExactly(lambda s: s.buildings.ilw_smelter_h),
        ilw_storage_l=FromExactly(lambda s: s.buildings.ilw_storage_l),
        ilw_storage_w=FromExactly(lambda s: s.buildings.ilw_storage_w),
        ilw_storage_h=FromExactly(lambda s: s.buildings.ilw_storage_h),
        llw_storage_l=FromExactly(lambda s: s.buildings.llw_storage_l),
        llw_storage_w=FromExactly(lambda s: s.buildings.llw_storage_w),
        llw_storage_h=FromExactly(lambda s: s.buildings.llw_storage_h),
        hw_storage_l=FromExactly(lambda s: s.buildings.hw_storage_l),
        hw_storage_w=FromExactly(lambda s: s.buildings.hw_storage_w),
        hw_storage_h=FromExactly(lambda s: s.buildings.hw_storage_h),
        tw_storage_l=FromExactly(lambda s: s.buildings.tw_storage_l),
        tw_storage_w=FromExactly(lambda s: s.buildings.tw_storage_w),
        tw_storage_h=FromExactly(lambda s: s.buildings.tw_storage_h),
        gas_buildings_l=FromExactly(lambda s: s.buildings.gas_buildings_l),
        gas_buildings_w=FromExactly(lambda s: s.buildings.gas_buildings_w),
        gas_buildings_h=FromExactly(lambda s: s.buildings.gas_buildings_h),
        water_buildings_l=FromExactly(lambda s: s.buildings.water_buildings_l),
        water_buildings_w=FromExactly(lambda s: s.buildings.water_buildings_w),
        water_buildings_h=FromExactly(lambda s: s.buildings.water_buildings_h),
        sec_buildings_l=FromExactly(lambda s: s.buildings.sec_buildings_l),
        sec_buildings_w=FromExactly(lambda s: s.buildings.sec_buildings_w),
        sec_buildings_h=FromExactly(lambda s: s.buildings.sec_buildings_h),
        staff_buildings_area=FromExactly(lambda s: s.buildings.staff_buildings_area),
        staff_buildings_h=FromExactly(lambda s: s.buildings.staff_buildings_h),
    ):
        is_neutral_beam = (
            CurrentDriveModel(self.i_hcd_primary).method
            == CurrentDriveMethodType.NEUTRAL_BEAM
        )

        return calculate_bldgs_sizes(
            r_pf_coil_outer_max,
            r_cryostat_inboard,
            tf_radial_dim,
            bioshld_thk,
            reactor_clrnc,
            transp_clrnc,
            crane_clrnc_h,
            cryostat_clrnc,
            ground_clrnc,
            crane_arm_h,
            tf_vertical_dim,
            is_neutral_beam,
            nbi_sys_l,
            nbi_sys_w,
            hcd_building_l,
            hcd_building_w,
            hcd_building_h,
            fc_building_l,
            fc_building_w,
            reactor_wall_thk,
            reactor_roof_thk,
            reactor_fndtn_thk,
            life_plant,
            z_tf_inside_half,
            dr_tf_inboard,
            dr_tf_shld_gap,
            dz_shld_thermal,
            dz_shld_vv_gap,
            dr_shld_inboard,
            dr_blkt_inboard,
            dr_fw_inboard,
            rmajor,
            rminor,
            dr_fw_plasma_gap_inboard,
            n_tf_coils,
            hot_sepdist,
            qnty_sfty_fac,
            dr_fw_outboard,
            dr_blkt_outboard,
            dr_shld_outboard,
            dr_fw_plasma_gap_outboard,
            life_div_fpy,
            dz_divertor,
            cplife,
            i_tf_sup,
            r_cp_top,
            hotcell_h,
            chemlab_l,
            chemlab_w,
            chemlab_h,
            heat_sink_l,
            heat_sink_w,
            heat_sink_h,
            aux_build_l,
            aux_build_w,
            aux_build_h,
            magnet_trains_l,
            magnet_trains_w,
            magnet_trains_h,
            magnet_pulse_l,
            magnet_pulse_w,
            magnet_pulse_h,
            control_buildings_l,
            control_buildings_w,
            control_buildings_h,
            warm_shop_l,
            warm_shop_w,
            warm_shop_h,
            workshop_l,
            workshop_w,
            workshop_h,
            robotics_l,
            robotics_w,
            robotics_h,
            maint_cont_l,
            maint_cont_w,
            maint_cont_h,
            cryomag_l,
            cryomag_w,
            cryomag_h,
            cryostore_l,
            cryostore_w,
            cryostore_h,
            auxcool_l,
            auxcool_w,
            auxcool_h,
            elecdist_l,
            elecdist_w,
            elecdist_h,
            elecload_l,
            elecload_w,
            elecload_h,
            elecstore_l,
            elecstore_w,
            elecstore_h,
            turbine_hall_l,
            turbine_hall_w,
            turbine_hall_h,
            ilw_smelter_l,
            ilw_smelter_w,
            ilw_smelter_h,
            ilw_storage_l,
            ilw_storage_w,
            ilw_storage_h,
            llw_storage_l,
            llw_storage_w,
            llw_storage_h,
            hw_storage_l,
            hw_storage_w,
            hw_storage_h,
            tw_storage_l,
            tw_storage_w,
            tw_storage_h,
            gas_buildings_l,
            gas_buildings_w,
            gas_buildings_h,
            water_buildings_l,
            water_buildings_w,
            water_buildings_h,
            sec_buildings_l,
            sec_buildings_w,
            sec_buildings_h,
            staff_buildings_area,
            staff_buildings_h,
        )
