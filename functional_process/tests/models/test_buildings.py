"""Harness cases for the ported plant buildings sizing (unit #15).

Follows `test_density_limits.py`'s shape: no test functions here, just reference
adapters, the ports, and the sample points, subclassing the tier each function's audit
record (`buildings.md`) assigns -- all tier 1.

Legacy points for `calculate_bldgs`/`calculate_bldgs_sizes` are lifted from PROCESS's
own `tests/unit/models/test_buildings.py` (`test_bldgs`/`test_bldgs_sizes`), already-
validated input points. Rather than duplicating those tests' hardcoded expected output
values, the reference adapters below call the real `Buildings.bldgs`/`bldgs_sizes`
methods directly (same approach `test_build.py` takes for `st_build`) -- so the
comparison is always against a live PROCESS evaluation, not a second copy of numbers
that could drift from the source.

`calculate_tf_coil_envelope`/`calculate_shield_height` have no covering PROCESS unit
test (`Buildings.run()`'s preamble is not itself under test anywhere in
`tests/unit`), so their samples are fuzz-only, same situation `test_build.py`
documents for `st_build`.
"""

from functional_process.cottax._harness import Tier1Contract
from functional_process.cottax._harness.sample_store import FROM_FILE
from functional_process.cottax.buildings.buildings import (
    calculate_bldgs,
    calculate_bldgs_sizes,
    calculate_shield_height,
    calculate_tf_coil_envelope,
)
from process.core.model import DataStructure
from process.models.buildings import Buildings
from process.models.physics.current_drive import (
    CurrentDriveMethodType,
    CurrentDriveModel,
)


def _buildings():
    """A `Buildings` instance with a bare `DataStructure` attached."""
    buildings = Buildings()
    buildings.data = DataStructure()
    return buildings


def _reference_tf_coil_envelope(
    r_tf_outboard_mid,
    dr_tf_outboard,
    r_tf_inboard_mid,
    dr_tf_inboard,
    z_tf_inside_half,
    m_tf_coils_total,
    n_tf_coils,
):
    """Reproduce `Buildings.run()`'s unconditional preamble directly.

    There is no PROCESS entry point taking these as arguments on their own (`run()`
    computes them inline before dispatching) -- reproduced here rather than invented,
    same reasoning `build.py`'s `_reference_a_fw_total_no_powerflow` gives for a value
    PROCESS never stores.
    """
    tfro = r_tf_outboard_mid + dr_tf_outboard * 0.5
    tfri = r_tf_inboard_mid - dr_tf_inboard * 0.5
    tf_radial_dim = tfro - tfri
    tf_vertical_dim = 2.0 * (z_tf_inside_half + dr_tf_outboard)
    tfmtn = 1.0e-3 * m_tf_coils_total / n_tf_coils
    return tfro, tfri, tf_radial_dim, tf_vertical_dim, tfmtn


def _reference_shield_height(z_tf_inside_half, dz_shld_vv_gap, dz_vv_upper, dz_vv_lower):
    """Reproduce `run()`'s `shh` call-site expression directly -- see `buildings.md`."""
    return 2.0 * (z_tf_inside_half - dz_shld_vv_gap) - dz_vv_upper - dz_vv_lower


def _reference_bldgs(
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
    """Call PROCESS's `Buildings.bldgs` through the port's signature."""
    buildings = _buildings()
    data = buildings.data

    data.buildings.rxcl = rxcl
    data.buildings.trcl = trcl
    data.buildings.row = row
    data.buildings.wgt = wgt
    data.buildings.shmf = shmf
    data.buildings.clh2 = clh2
    data.buildings.dz_tf_cryostat = dz_tf_cryostat
    data.buildings.stcl = stcl
    data.buildings.rbvfac = rbvfac
    data.buildings.rbwt = rbwt
    data.buildings.rbrt = rbrt
    data.buildings.fndt = fndt
    data.buildings.hcwt = hcwt
    data.buildings.hccl = hccl
    data.buildings.wgt2 = wgt2
    data.buildings.mbvfac = mbvfac
    data.buildings.wsvfac = wsvfac
    data.buildings.tfcbv = tfcbv
    data.buildings.pfbldgm3 = pfbldgm3
    data.buildings.esbldgm3 = esbldgm3
    data.buildings.pibv = pibv
    data.buildings.triv = triv
    data.buildings.conv = conv
    data.buildings.admv = admv
    data.buildings.shov = shov

    cryv, vrci, rbv, rmbv, wsv, elev = buildings.bldgs(
        output=False,
        pfr=pfr,
        pfm=pfm,
        tfro=tfro,
        tfri=tfri,
        tfh=tfh,
        tfm=tfm,
        n_tf_coils=n_tf_coils,
        shro=shro,
        shri=shri,
        shh=shh,
        shm=shm,
        crr=crr,
        helpow=helpow,
    )

    return (
        cryv,
        vrci,
        rbv,
        rmbv,
        wsv,
        elev,
        data.buildings.wrbi,
        data.buildings.a_plant_floor_effective,
        data.buildings.admvol,
        data.buildings.shovol,
        data.buildings.convol,
        data.buildings.volnucb,
    )


def _is_neutral_beam(i_hcd_primary):
    return CurrentDriveModel(i_hcd_primary).method == CurrentDriveMethodType.NEUTRAL_BEAM


def _reference_bldgs_sizes(
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
    """Call PROCESS's `Buildings.bldgs_sizes` through the port's signature.

    `is_neutral_beam` stands in for `.current_drive.i_hcd_primary` -- the reference
    adapter picks a representative concrete `i_hcd_primary` for each side of the switch
    (10 = `USER_INPUT_ELECTRON_CYCLOTRON`, not NBI; 5 = `ITER_NEUTRAL_BEAM`) since the
    port itself takes the already-resolved boolean, per `buildings.md`.
    """
    buildings = _buildings()
    data = buildings.data

    data.current_drive.i_hcd_primary = 5 if is_neutral_beam else 10
    data.buildings.bioshld_thk = bioshld_thk
    data.buildings.reactor_clrnc = reactor_clrnc
    data.buildings.transp_clrnc = transp_clrnc
    data.buildings.crane_clrnc_h = crane_clrnc_h
    data.buildings.cryostat_clrnc = cryostat_clrnc
    data.buildings.ground_clrnc = ground_clrnc
    data.buildings.crane_arm_h = crane_arm_h
    data.buildings.nbi_sys_l = nbi_sys_l
    data.buildings.nbi_sys_w = nbi_sys_w
    data.buildings.hcd_building_l = hcd_building_l
    data.buildings.hcd_building_w = hcd_building_w
    data.buildings.hcd_building_h = hcd_building_h
    data.buildings.fc_building_l = fc_building_l
    data.buildings.fc_building_w = fc_building_w
    data.buildings.reactor_wall_thk = reactor_wall_thk
    data.buildings.reactor_roof_thk = reactor_roof_thk
    data.buildings.reactor_fndtn_thk = reactor_fndtn_thk
    data.buildings.hot_sepdist = hot_sepdist
    data.buildings.qnty_sfty_fac = qnty_sfty_fac
    data.buildings.hotcell_h = hotcell_h
    data.buildings.chemlab_l = chemlab_l
    data.buildings.chemlab_w = chemlab_w
    data.buildings.chemlab_h = chemlab_h
    data.buildings.heat_sink_l = heat_sink_l
    data.buildings.heat_sink_w = heat_sink_w
    data.buildings.heat_sink_h = heat_sink_h
    data.buildings.aux_build_l = aux_build_l
    data.buildings.aux_build_w = aux_build_w
    data.buildings.aux_build_h = aux_build_h
    data.buildings.magnet_trains_l = magnet_trains_l
    data.buildings.magnet_trains_w = magnet_trains_w
    data.buildings.magnet_trains_h = magnet_trains_h
    data.buildings.magnet_pulse_l = magnet_pulse_l
    data.buildings.magnet_pulse_w = magnet_pulse_w
    data.buildings.magnet_pulse_h = magnet_pulse_h
    data.buildings.control_buildings_l = control_buildings_l
    data.buildings.control_buildings_w = control_buildings_w
    data.buildings.control_buildings_h = control_buildings_h
    data.buildings.warm_shop_l = warm_shop_l
    data.buildings.warm_shop_w = warm_shop_w
    data.buildings.warm_shop_h = warm_shop_h
    data.buildings.workshop_l = workshop_l
    data.buildings.workshop_w = workshop_w
    data.buildings.workshop_h = workshop_h
    data.buildings.robotics_l = robotics_l
    data.buildings.robotics_w = robotics_w
    data.buildings.robotics_h = robotics_h
    data.buildings.maint_cont_l = maint_cont_l
    data.buildings.maint_cont_w = maint_cont_w
    data.buildings.maint_cont_h = maint_cont_h
    data.buildings.turbine_hall_l = turbine_hall_l
    data.buildings.turbine_hall_w = turbine_hall_w
    data.buildings.turbine_hall_h = turbine_hall_h
    data.buildings.gas_buildings_l = gas_buildings_l
    data.buildings.gas_buildings_w = gas_buildings_w
    data.buildings.gas_buildings_h = gas_buildings_h
    data.buildings.water_buildings_l = water_buildings_l
    data.buildings.water_buildings_w = water_buildings_w
    data.buildings.water_buildings_h = water_buildings_h
    data.buildings.sec_buildings_l = sec_buildings_l
    data.buildings.sec_buildings_w = sec_buildings_w
    data.buildings.sec_buildings_h = sec_buildings_h
    data.buildings.staff_buildings_area = staff_buildings_area
    data.buildings.staff_buildings_h = staff_buildings_h
    data.buildings.ilw_smelter_l = ilw_smelter_l
    data.buildings.ilw_smelter_w = ilw_smelter_w
    data.buildings.ilw_smelter_h = ilw_smelter_h
    data.buildings.ilw_storage_l = ilw_storage_l
    data.buildings.ilw_storage_w = ilw_storage_w
    data.buildings.ilw_storage_h = ilw_storage_h
    data.buildings.llw_storage_l = llw_storage_l
    data.buildings.llw_storage_w = llw_storage_w
    data.buildings.llw_storage_h = llw_storage_h
    data.buildings.hw_storage_l = hw_storage_l
    data.buildings.hw_storage_w = hw_storage_w
    data.buildings.hw_storage_h = hw_storage_h
    data.buildings.tw_storage_l = tw_storage_l
    data.buildings.tw_storage_w = tw_storage_w
    data.buildings.tw_storage_h = tw_storage_h
    data.buildings.auxcool_l = auxcool_l
    data.buildings.auxcool_w = auxcool_w
    data.buildings.auxcool_h = auxcool_h
    data.buildings.cryomag_l = cryomag_l
    data.buildings.cryomag_w = cryomag_w
    data.buildings.cryomag_h = cryomag_h
    data.buildings.cryostore_l = cryostore_l
    data.buildings.cryostore_w = cryostore_w
    data.buildings.cryostore_h = cryostore_h
    data.buildings.elecdist_l = elecdist_l
    data.buildings.elecdist_w = elecdist_w
    data.buildings.elecdist_h = elecdist_h
    data.buildings.elecstore_l = elecstore_l
    data.buildings.elecstore_w = elecstore_w
    data.buildings.elecstore_h = elecstore_h
    data.buildings.elecload_l = elecload_l
    data.buildings.elecload_w = elecload_w
    data.buildings.elecload_h = elecload_h

    data.tfcoil.n_tf_coils = n_tf_coils
    data.tfcoil.i_tf_sup = i_tf_sup
    data.pf_coil.r_pf_coil_outer_max = r_pf_coil_outer_max
    data.costs.life_plant = life_plant
    data.costs.cplife = cplife
    data.costs.life_div_fpy = life_div_fpy
    data.fwbs.r_cryostat_inboard = r_cryostat_inboard

    data.build.z_tf_inside_half = z_tf_inside_half
    data.build.dr_tf_inboard = dr_tf_inboard
    data.build.dr_tf_shld_gap = dr_tf_shld_gap
    data.build.dz_shld_thermal = dz_shld_thermal
    data.build.dz_shld_vv_gap = dz_shld_vv_gap
    data.build.dr_shld_inboard = dr_shld_inboard
    data.build.dr_shld_outboard = dr_shld_outboard
    data.build.dr_fw_plasma_gap_inboard = dr_fw_plasma_gap_inboard
    data.build.dr_fw_plasma_gap_outboard = dr_fw_plasma_gap_outboard
    data.build.dr_fw_inboard = dr_fw_inboard
    data.build.dr_fw_outboard = dr_fw_outboard
    data.build.dr_blkt_inboard = dr_blkt_inboard
    data.build.dr_blkt_outboard = dr_blkt_outboard
    data.build.r_cp_top = r_cp_top

    data.divertor.dz_divertor = dz_divertor
    data.physics.rmajor = rmajor
    data.physics.rminor = rminor

    buildings.bldgs_sizes(
        output=False,
        tf_radial_dim=tf_radial_dim,
        tf_vertical_dim=tf_vertical_dim,
    )

    return (
        data.buildings.reactor_hall_l,
        data.buildings.reactor_hall_w,
        data.buildings.reactor_hall_h,
        data.buildings.a_plant_floor_effective,
        data.buildings.volnucb,
    )


class TestTfCoilEnvelope(Tier1Contract):
    """`Buildings.run()`'s preamble -> `calculate_tf_coil_envelope`."""

    audit_record = "models/buildings.md"
    reference = _reference_tf_coil_envelope
    ported = calculate_tf_coil_envelope

    fuzz_bounds = {
        "r_tf_outboard_mid": (5.0, 30.0),
        "dr_tf_outboard": (0.1, 2.0),
        "r_tf_inboard_mid": (1.0, 10.0),
        "dr_tf_inboard": (0.1, 2.0),
        "z_tf_inside_half": (1.0, 20.0),
        "m_tf_coils_total": (1.0e5, 1.0e8),
        "n_tf_coils": (10.0, 20.0),
    }


class TestShieldHeight(Tier1Contract):
    """`run()`'s `shh` call-site expression -> `calculate_shield_height`."""

    audit_record = "models/buildings.md"
    reference = _reference_shield_height
    ported = calculate_shield_height

    fuzz_bounds = {
        "z_tf_inside_half": (1.0, 20.0),
        "dz_shld_vv_gap": (0.0, 1.0),
        "dz_vv_upper": (0.0, 2.0),
        "dz_vv_lower": (0.0, 2.0),
    }


class TestBldgs(Tier1Contract):
    """`Buildings.bldgs` -> `calculate_bldgs` (`BuildingsModel.ITER_1992`)."""

    audit_record = "models/buildings.md"
    reference = _reference_bldgs
    ported = calculate_bldgs

    # tests/unit/models/test_buildings.py::test_bldgs, both parametrized cases.
    samples = FROM_FILE

    fuzz = True


class TestBldgsSizes(Tier1Contract):
    """`Buildings.bldgs_sizes` -> `calculate_bldgs_sizes`.

    `BuildingsModel.CHAPMAN_2024` branch.
    """

    audit_record = "models/buildings.md"
    reference = _reference_bldgs_sizes
    ported = calculate_bldgs_sizes

    static_argnames = ("is_neutral_beam", "i_tf_sup")

    # tests/unit/models/test_buildings.py::test_bldgs_sizes, both parametrized cases
    # (i_hcd_primary=10 in both -> is_neutral_beam=False).
    samples = FROM_FILE

    fuzz = True
    fuzz_fixed = {"is_neutral_beam": False, "i_tf_sup": 1}
