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

from functional_process._harness import Tier1Contract, legacy_sample
from functional_process.models.buildings.buildings import (
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
    return (
        CurrentDriveModel(i_hcd_primary).method == CurrentDriveMethodType.NEUTRAL_BEAM
    )


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
    samples = [
        legacy_sample(
            "test-bldgs-0",
            pfr=18.98258241468535,
            pfm=1071.5897090529959,
            tfro=17.123405859443331,
            tfri=2.9939411851091102,
            tfh=20.562180043124066,
            tfm=1327.1818597762153,
            n_tf_coils=16,
            shro=13.764874193548387,
            shri=4.7423258064516141,
            shh=17.446180043124063,
            shm=2294873.8131476026,
            crr=19.48258241468535,
            helpow=77840.021662652987,
            rxcl=4,
            trcl=1,
            row=4,
            wgt=500000,
            shmf=0.5,
            clh2=15,
            dz_tf_cryostat=5.7514039424138126,
            stcl=3,
            rbvfac=1.6000000000000001,
            rbwt=2,
            rbrt=1,
            fndt=2,
            hcwt=1.5,
            hccl=5,
            wgt2=100000,
            mbvfac=2.7999999999999998,
            wsvfac=1.8999999999999999,
            tfcbv=10601.097615432001,
            pfbldgm3=20000,
            esbldgm3=1000,
            pibv=20000,
            triv=40000,
            conv=60000,
            admv=100000,
            shov=100000,
        ),
        legacy_sample(
            "test-bldgs-1",
            pfr=18.982980877139834,
            pfm=1073.3372194668184,
            tfro=17.123405859443331,
            tfri=2.9939411851091102,
            tfh=20.562180043124066,
            tfm=1327.9750836697808,
            n_tf_coils=16,
            shro=13.782874193548388,
            shri=4.7243258064516143,
            shh=17.446180043124063,
            shm=2297808.3935174868,
            crr=19.482980877139834,
            helpow=221493.99746816326,
            rxcl=4,
            trcl=1,
            row=4,
            wgt=500000,
            shmf=0.5,
            clh2=15,
            dz_tf_cryostat=5.8405005070918357,
            stcl=3,
            rbvfac=1.6000000000000001,
            rbwt=2,
            rbrt=1,
            fndt=2,
            hcwt=1.5,
            hccl=5,
            wgt2=100000,
            mbvfac=2.7999999999999998,
            wsvfac=1.8999999999999999,
            tfcbv=10609.268177478583,
            pfbldgm3=20000,
            esbldgm3=1000,
            pibv=20000,
            triv=40000,
            conv=60000,
            admv=100000,
            shov=100000,
        ),
    ]

    fuzz_bounds = {
        "pfr": (5.0, 30.0),
        "pfm": (100.0, 5000.0),
        "tfro": (10.0, 30.0),
        "tfri": (1.0, 8.0),
        "tfh": (5.0, 30.0),
        "tfm": (100.0, 5000.0),
        "n_tf_coils": (10.0, 20.0),
        "shro": (5.0, 20.0),
        "shri": (1.0, 10.0),
        "shh": (5.0, 25.0),
        "shm": (1.0e5, 5.0e6),
        "crr": (5.0, 30.0),
        "helpow": (1.0e3, 3.0e5),
        "rxcl": (1.0, 10.0),
        "trcl": (0.5, 5.0),
        "row": (1.0, 10.0),
        "wgt": (0.0, 1.0e6),
        "shmf": (0.1, 1.0),
        "clh2": (5.0, 30.0),
        "dz_tf_cryostat": (1.0, 10.0),
        "stcl": (1.0, 10.0),
        "rbvfac": (1.0, 3.0),
        "rbwt": (0.5, 5.0),
        "rbrt": (0.5, 5.0),
        "fndt": (0.5, 5.0),
        "hcwt": (0.5, 5.0),
        "hccl": (1.0, 10.0),
        "wgt2": (0.0, 5.0e5),
        "mbvfac": (1.0, 5.0),
        "wsvfac": (1.0, 5.0),
        "tfcbv": (1.0e3, 5.0e4),
        "pfbldgm3": (1.0e3, 5.0e4),
        "esbldgm3": (0.0, 5.0e3),
        "pibv": (1.0e3, 5.0e4),
        "triv": (1.0e3, 1.0e5),
        "conv": (1.0e3, 1.0e5),
        "admv": (1.0e3, 5.0e5),
        "shov": (1.0e3, 5.0e5),
    }


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
    samples = [
        legacy_sample(
            "test-bldgs-sizes-0",
            r_pf_coil_outer_max=18.98258241468535,
            r_cryostat_inboard=19.48258241468535,
            tf_radial_dim=14.129464674334221,
            bioshld_thk=2.5,
            reactor_clrnc=4,
            transp_clrnc=1,
            crane_clrnc_h=4,
            cryostat_clrnc=2.5,
            ground_clrnc=5,
            crane_arm_h=10,
            tf_vertical_dim=20.562180043124066,
            is_neutral_beam=False,
            nbi_sys_l=225,
            nbi_sys_w=185,
            hcd_building_l=70,
            hcd_building_w=40,
            hcd_building_h=25,
            fc_building_l=60,
            fc_building_w=60,
            reactor_wall_thk=2,
            reactor_roof_thk=1,
            reactor_fndtn_thk=2,
            life_plant=40,
            z_tf_inside_half=9.0730900215620327,
            dr_tf_inboard=1.208,
            dr_tf_shld_gap=0.05000000000000001,
            dz_shld_thermal=0.050000000000000003,
            dz_shld_vv_gap=0.05,
            dr_shld_inboard=0.30000000000000004,
            dr_blkt_inboard=0.75500000000000012,
            dr_fw_inboard=0.018000000000000002,
            rmajor=8.8901000000000003,
            rminor=2.8677741935483869,
            dr_fw_plasma_gap_inboard=0.22500000000000003,
            n_tf_coils=16,
            hot_sepdist=2,
            qnty_sfty_fac=2,
            dr_fw_outboard=0.018000000000000002,
            dr_blkt_outboard=0.98199999999999998,
            dr_shld_outboard=0.80000000000000004,
            dr_fw_plasma_gap_outboard=0.22500000000000003,
            life_div_fpy=0,
            dz_divertor=0.62100000000000011,
            cplife=0,
            i_tf_sup=1,
            r_cp_top=4.20194118510911,
            hotcell_h=12,
            chemlab_l=50,
            chemlab_w=30,
            chemlab_h=6,
            heat_sink_l=160,
            heat_sink_w=80,
            heat_sink_h=12,
            aux_build_l=60,
            aux_build_w=30,
            aux_build_h=5,
            magnet_trains_l=120,
            magnet_trains_w=90,
            magnet_trains_h=5,
            magnet_pulse_l=105,
            magnet_pulse_w=40,
            magnet_pulse_h=5,
            control_buildings_l=80,
            control_buildings_w=60,
            control_buildings_h=6,
            warm_shop_l=100,
            warm_shop_w=50,
            warm_shop_h=10,
            workshop_l=150,
            workshop_w=125,
            workshop_h=10,
            robotics_l=50,
            robotics_w=30,
            robotics_h=30,
            maint_cont_l=125,
            maint_cont_w=100,
            maint_cont_h=6,
            cryomag_l=120,
            cryomag_w=90,
            cryomag_h=5,
            cryostore_l=160,
            cryostore_w=30,
            cryostore_h=20,
            auxcool_l=20,
            auxcool_w=20,
            auxcool_h=5,
            elecdist_l=380,
            elecdist_w=350,
            elecdist_h=5,
            elecload_l=100,
            elecload_w=90,
            elecload_h=3,
            elecstore_l=100,
            elecstore_w=60,
            elecstore_h=12,
            turbine_hall_l=109,
            turbine_hall_w=62,
            turbine_hall_h=15,
            ilw_smelter_l=50,
            ilw_smelter_w=30,
            ilw_smelter_h=30,
            ilw_storage_l=120,
            ilw_storage_w=100,
            ilw_storage_h=8,
            llw_storage_l=45,
            llw_storage_w=20,
            llw_storage_h=5,
            hw_storage_l=20,
            hw_storage_w=10,
            hw_storage_h=5,
            tw_storage_l=90,
            tw_storage_w=30,
            tw_storage_h=5,
            gas_buildings_l=25,
            gas_buildings_w=15,
            gas_buildings_h=5,
            water_buildings_l=110,
            water_buildings_w=10,
            water_buildings_h=5,
            sec_buildings_l=30,
            sec_buildings_w=25,
            sec_buildings_h=6,
            staff_buildings_area=480000,
            staff_buildings_h=5,
        ),
        legacy_sample(
            "test-bldgs-sizes-1",
            r_pf_coil_outer_max=18.982980877139834,
            r_cryostat_inboard=19.482980877139834,
            tf_radial_dim=14.129464674334221,
            bioshld_thk=2.5,
            reactor_clrnc=4,
            transp_clrnc=1,
            crane_clrnc_h=4,
            cryostat_clrnc=2.5,
            ground_clrnc=5,
            crane_arm_h=10,
            tf_vertical_dim=20.562180043124066,
            is_neutral_beam=False,
            nbi_sys_l=225,
            nbi_sys_w=185,
            hcd_building_l=70,
            hcd_building_w=40,
            hcd_building_h=25,
            fc_building_l=60,
            fc_building_w=60,
            reactor_wall_thk=2,
            reactor_roof_thk=1,
            reactor_fndtn_thk=2,
            life_plant=40,
            z_tf_inside_half=9.0730900215620327,
            dr_tf_inboard=1.208,
            dr_tf_shld_gap=0.05000000000000001,
            dz_shld_thermal=0.050000000000000003,
            dz_shld_vv_gap=0.05,
            dr_shld_inboard=0.30000000000000004,
            dr_blkt_inboard=0.75500000000000012,
            dr_fw_inboard=0.018000000000000002,
            rmajor=8.8901000000000003,
            rminor=2.8677741935483869,
            dr_fw_plasma_gap_inboard=0.22500000000000003,
            n_tf_coils=16,
            hot_sepdist=2,
            qnty_sfty_fac=2,
            dr_fw_outboard=0.018000000000000002,
            dr_blkt_outboard=0.98199999999999998,
            dr_shld_outboard=0.80000000000000004,
            dr_fw_plasma_gap_outboard=0.22500000000000003,
            life_div_fpy=6.1337250397740126,
            dz_divertor=0.62100000000000011,
            cplife=0,
            i_tf_sup=1,
            r_cp_top=4.20194118510911,
            hotcell_h=12,
            chemlab_l=50,
            chemlab_w=30,
            chemlab_h=6,
            heat_sink_l=160,
            heat_sink_w=80,
            heat_sink_h=12,
            aux_build_l=60,
            aux_build_w=30,
            aux_build_h=5,
            magnet_trains_l=120,
            magnet_trains_w=90,
            magnet_trains_h=5,
            magnet_pulse_l=105,
            magnet_pulse_w=40,
            magnet_pulse_h=5,
            control_buildings_l=80,
            control_buildings_w=60,
            control_buildings_h=6,
            warm_shop_l=100,
            warm_shop_w=50,
            warm_shop_h=10,
            workshop_l=150,
            workshop_w=125,
            workshop_h=10,
            robotics_l=50,
            robotics_w=30,
            robotics_h=30,
            maint_cont_l=125,
            maint_cont_w=100,
            maint_cont_h=6,
            cryomag_l=120,
            cryomag_w=90,
            cryomag_h=5,
            cryostore_l=160,
            cryostore_w=30,
            cryostore_h=20,
            auxcool_l=20,
            auxcool_w=20,
            auxcool_h=5,
            elecdist_l=380,
            elecdist_w=350,
            elecdist_h=5,
            elecload_l=100,
            elecload_w=90,
            elecload_h=3,
            elecstore_l=100,
            elecstore_w=60,
            elecstore_h=12,
            turbine_hall_l=109,
            turbine_hall_w=62,
            turbine_hall_h=15,
            ilw_smelter_l=50,
            ilw_smelter_w=30,
            ilw_smelter_h=30,
            ilw_storage_l=120,
            ilw_storage_w=100,
            ilw_storage_h=8,
            llw_storage_l=45,
            llw_storage_w=20,
            llw_storage_h=5,
            hw_storage_l=20,
            hw_storage_w=10,
            hw_storage_h=5,
            tw_storage_l=90,
            tw_storage_w=30,
            tw_storage_h=5,
            gas_buildings_l=25,
            gas_buildings_w=15,
            gas_buildings_h=5,
            water_buildings_l=110,
            water_buildings_w=10,
            water_buildings_h=5,
            sec_buildings_l=30,
            sec_buildings_w=25,
            sec_buildings_h=6,
            staff_buildings_area=480000,
            staff_buildings_h=5,
        ),
        # Same as case 0, but the NBI branch of `.current_drive.i_hcd_primary`, which
        # neither PROCESS test file above exercises for `bldgs_sizes` (both use
        # i_hcd_primary=10). Included so this switch's other arm is checked at all,
        # not just fuzzed.
        legacy_sample(
            "nbi-branch",
            r_pf_coil_outer_max=18.98258241468535,
            r_cryostat_inboard=19.48258241468535,
            tf_radial_dim=14.129464674334221,
            bioshld_thk=2.5,
            reactor_clrnc=4,
            transp_clrnc=1,
            crane_clrnc_h=4,
            cryostat_clrnc=2.5,
            ground_clrnc=5,
            crane_arm_h=10,
            tf_vertical_dim=20.562180043124066,
            is_neutral_beam=True,
            nbi_sys_l=225,
            nbi_sys_w=185,
            hcd_building_l=70,
            hcd_building_w=40,
            hcd_building_h=25,
            fc_building_l=60,
            fc_building_w=60,
            reactor_wall_thk=2,
            reactor_roof_thk=1,
            reactor_fndtn_thk=2,
            life_plant=40,
            z_tf_inside_half=9.0730900215620327,
            dr_tf_inboard=1.208,
            dr_tf_shld_gap=0.05000000000000001,
            dz_shld_thermal=0.050000000000000003,
            dz_shld_vv_gap=0.05,
            dr_shld_inboard=0.30000000000000004,
            dr_blkt_inboard=0.75500000000000012,
            dr_fw_inboard=0.018000000000000002,
            rmajor=8.8901000000000003,
            rminor=2.8677741935483869,
            dr_fw_plasma_gap_inboard=0.22500000000000003,
            n_tf_coils=16,
            hot_sepdist=2,
            qnty_sfty_fac=2,
            dr_fw_outboard=0.018000000000000002,
            dr_blkt_outboard=0.98199999999999998,
            dr_shld_outboard=0.80000000000000004,
            dr_fw_plasma_gap_outboard=0.22500000000000003,
            life_div_fpy=0,
            dz_divertor=0.62100000000000011,
            cplife=3.5,
            i_tf_sup=0,
            r_cp_top=4.20194118510911,
            hotcell_h=12,
            chemlab_l=50,
            chemlab_w=30,
            chemlab_h=6,
            heat_sink_l=160,
            heat_sink_w=80,
            heat_sink_h=12,
            aux_build_l=60,
            aux_build_w=30,
            aux_build_h=5,
            magnet_trains_l=120,
            magnet_trains_w=90,
            magnet_trains_h=5,
            magnet_pulse_l=105,
            magnet_pulse_w=40,
            magnet_pulse_h=5,
            control_buildings_l=80,
            control_buildings_w=60,
            control_buildings_h=6,
            warm_shop_l=100,
            warm_shop_w=50,
            warm_shop_h=10,
            workshop_l=150,
            workshop_w=125,
            workshop_h=10,
            robotics_l=50,
            robotics_w=30,
            robotics_h=30,
            maint_cont_l=125,
            maint_cont_w=100,
            maint_cont_h=6,
            cryomag_l=120,
            cryomag_w=90,
            cryomag_h=5,
            cryostore_l=160,
            cryostore_w=30,
            cryostore_h=20,
            auxcool_l=20,
            auxcool_w=20,
            auxcool_h=5,
            elecdist_l=380,
            elecdist_w=350,
            elecdist_h=5,
            elecload_l=100,
            elecload_w=90,
            elecload_h=3,
            elecstore_l=100,
            elecstore_w=60,
            elecstore_h=12,
            turbine_hall_l=109,
            turbine_hall_w=62,
            turbine_hall_h=15,
            ilw_smelter_l=50,
            ilw_smelter_w=30,
            ilw_smelter_h=30,
            ilw_storage_l=120,
            ilw_storage_w=100,
            ilw_storage_h=8,
            llw_storage_l=45,
            llw_storage_w=20,
            llw_storage_h=5,
            hw_storage_l=20,
            hw_storage_w=10,
            hw_storage_h=5,
            tw_storage_l=90,
            tw_storage_w=30,
            tw_storage_h=5,
            gas_buildings_l=25,
            gas_buildings_w=15,
            gas_buildings_h=5,
            water_buildings_l=110,
            water_buildings_w=10,
            water_buildings_h=5,
            sec_buildings_l=30,
            sec_buildings_w=25,
            sec_buildings_h=6,
            staff_buildings_area=480000,
            staff_buildings_h=5,
        ),
    ]

    fuzz_bounds = {
        "r_pf_coil_outer_max": (5.0, 30.0),
        "r_cryostat_inboard": (5.0, 30.0),
        "tf_radial_dim": (5.0, 20.0),
        "bioshld_thk": (0.5, 5.0),
        "reactor_clrnc": (1.0, 10.0),
        "transp_clrnc": (0.5, 5.0),
        "crane_clrnc_h": (1.0, 10.0),
        "cryostat_clrnc": (0.5, 5.0),
        "ground_clrnc": (1.0, 10.0),
        "crane_arm_h": (1.0, 20.0),
        "tf_vertical_dim": (5.0, 30.0),
        "nbi_sys_l": (50.0, 300.0),
        "nbi_sys_w": (50.0, 300.0),
        "hcd_building_l": (20.0, 150.0),
        "hcd_building_w": (10.0, 100.0),
        "hcd_building_h": (5.0, 50.0),
        "fc_building_l": (10.0, 150.0),
        "fc_building_w": (10.0, 150.0),
        "reactor_wall_thk": (0.5, 5.0),
        "reactor_roof_thk": (0.5, 5.0),
        "reactor_fndtn_thk": (0.5, 5.0),
        "life_plant": (1.0, 60.0),
        "z_tf_inside_half": (1.0, 20.0),
        "dr_tf_inboard": (0.1, 2.0),
        "dr_tf_shld_gap": (0.01, 0.5),
        "dz_shld_thermal": (0.01, 0.5),
        "dz_shld_vv_gap": (0.01, 0.5),
        "dr_shld_inboard": (0.05, 2.0),
        "dr_blkt_inboard": (0.1, 2.0),
        "dr_fw_inboard": (0.005, 0.1),
        "rmajor": (5.0, 30.0),
        "rminor": (0.5, 5.0),
        "dr_fw_plasma_gap_inboard": (0.05, 1.0),
        "n_tf_coils": (10.0, 20.0),
        "hot_sepdist": (0.5, 5.0),
        "qnty_sfty_fac": (1.0, 5.0),
        "dr_fw_outboard": (0.005, 0.1),
        "dr_blkt_outboard": (0.1, 2.0),
        "dr_shld_outboard": (0.05, 2.0),
        "dr_fw_plasma_gap_outboard": (0.05, 1.0),
        "life_div_fpy": (0.5, 20.0),
        "dz_divertor": (0.1, 2.0),
        "cplife": (0.5, 20.0),
        "r_cp_top": (0.5, 10.0),
        "hotcell_h": (5.0, 30.0),
        "chemlab_l": (10.0, 100.0),
        "chemlab_w": (10.0, 100.0),
        "chemlab_h": (2.0, 20.0),
        "heat_sink_l": (10.0, 300.0),
        "heat_sink_w": (10.0, 200.0),
        "heat_sink_h": (2.0, 30.0),
        "aux_build_l": (10.0, 200.0),
        "aux_build_w": (10.0, 100.0),
        "aux_build_h": (2.0, 20.0),
        "magnet_trains_l": (10.0, 300.0),
        "magnet_trains_w": (10.0, 200.0),
        "magnet_trains_h": (2.0, 20.0),
        "magnet_pulse_l": (10.0, 300.0),
        "magnet_pulse_w": (10.0, 200.0),
        "magnet_pulse_h": (2.0, 20.0),
        "control_buildings_l": (10.0, 200.0),
        "control_buildings_w": (10.0, 200.0),
        "control_buildings_h": (2.0, 20.0),
        "warm_shop_l": (10.0, 300.0),
        "warm_shop_w": (10.0, 200.0),
        "warm_shop_h": (2.0, 20.0),
        "workshop_l": (10.0, 300.0),
        "workshop_w": (10.0, 200.0),
        "workshop_h": (2.0, 20.0),
        "robotics_l": (10.0, 200.0),
        "robotics_w": (10.0, 100.0),
        "robotics_h": (2.0, 50.0),
        "maint_cont_l": (10.0, 300.0),
        "maint_cont_w": (10.0, 200.0),
        "maint_cont_h": (2.0, 20.0),
        "cryomag_l": (10.0, 300.0),
        "cryomag_w": (10.0, 200.0),
        "cryomag_h": (2.0, 20.0),
        "cryostore_l": (10.0, 300.0),
        "cryostore_w": (10.0, 100.0),
        "cryostore_h": (2.0, 50.0),
        "auxcool_l": (5.0, 100.0),
        "auxcool_w": (5.0, 100.0),
        "auxcool_h": (2.0, 20.0),
        "elecdist_l": (10.0, 600.0),
        "elecdist_w": (10.0, 600.0),
        "elecdist_h": (2.0, 20.0),
        "elecload_l": (10.0, 200.0),
        "elecload_w": (10.0, 200.0),
        "elecload_h": (2.0, 20.0),
        "elecstore_l": (10.0, 200.0),
        "elecstore_w": (10.0, 200.0),
        "elecstore_h": (2.0, 30.0),
        "turbine_hall_l": (20.0, 300.0),
        "turbine_hall_w": (10.0, 150.0),
        "turbine_hall_h": (5.0, 40.0),
        "ilw_smelter_l": (10.0, 100.0),
        "ilw_smelter_w": (10.0, 100.0),
        "ilw_smelter_h": (2.0, 50.0),
        "ilw_storage_l": (10.0, 300.0),
        "ilw_storage_w": (10.0, 200.0),
        "ilw_storage_h": (2.0, 30.0),
        "llw_storage_l": (10.0, 100.0),
        "llw_storage_w": (10.0, 100.0),
        "llw_storage_h": (2.0, 20.0),
        "hw_storage_l": (5.0, 50.0),
        "hw_storage_w": (5.0, 50.0),
        "hw_storage_h": (2.0, 20.0),
        "tw_storage_l": (10.0, 200.0),
        "tw_storage_w": (10.0, 100.0),
        "tw_storage_h": (2.0, 20.0),
        "gas_buildings_l": (5.0, 50.0),
        "gas_buildings_w": (5.0, 50.0),
        "gas_buildings_h": (2.0, 20.0),
        "water_buildings_l": (10.0, 300.0),
        "water_buildings_w": (5.0, 50.0),
        "water_buildings_h": (2.0, 20.0),
        "sec_buildings_l": (5.0, 100.0),
        "sec_buildings_w": (5.0, 100.0),
        "sec_buildings_h": (2.0, 20.0),
        "staff_buildings_area": (1.0e4, 1.0e6),
        "staff_buildings_h": (2.0, 20.0),
    }
    fuzz_fixed = {"is_neutral_beam": False, "i_tf_sup": 1}
