"""Pure-functional port of `process/models/buildings.py`'s `Buildings.run()`.

Unit #15.
"""

import jax.numpy as jnp  # noqa: F401
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.cottax.paths import (
    build,
    buildings,
    costs,
    divertor,
    fwbs,
    heat_transport,
    pf_coil,
    physics,
    tfcoil,
)
from functional_process.cottax.wraps import WrapsFunction
from functional_process.models.buildings.buildings import (
    calculate_bldgs,  # noqa: F401 -- re-exported for tests
    calculate_bldgs_from_elements,
    calculate_bldgs_sizes,  # noqa: F401 -- re-exported for tests
    calculate_bldgs_sizes_neutral_beam,
    calculate_bldgs_sizes_other_hcd,
    calculate_shield_height,  # noqa: F401 -- re-exported for tests
    calculate_tf_coil_envelope,
)
from functional_process.models.safe_math import safe_pow  # noqa: F401


class TfCoilEnvelope(WrapsFunction):
    """cottax node: `calculate_tf_coil_envelope`, ports declared."""

    fn = calculate_tf_coil_envelope

    r_tf_outboard_mid = From(build)
    dr_tf_outboard = From(build)
    r_tf_inboard_mid = From(build)
    dr_tf_inboard = From(build)
    z_tf_inside_half = From(build)
    m_tf_coils_total = From(tfcoil)
    n_tf_coils = From(tfcoil)

    tfro = OutputInto(buildings)
    tfri = OutputInto(buildings)
    tf_radial_dim = OutputInto(buildings)
    tf_vertical_dim = OutputInto(buildings)
    tfmtn = OutputInto(buildings)


class Bldgs(WrapsFunction):
    """cottax node: `calculate_bldgs_from_elements` --
    `calculate_bldgs`. Instantiate iff `i_bldgs_size == ITER_1992`.
    """

    fn = calculate_bldgs_from_elements

    r_pf_coil_outer_max = From(pf_coil)
    m_pf_coil_max = From(pf_coil)
    tfro = From(buildings)
    tfri = From(buildings)
    tf_vertical_dim = From(buildings)
    tfmtn = From(buildings)
    n_tf_coils = From(tfcoil)
    r_shld_outboard_outer = From(build)
    r_shld_inboard_inner = From(build)
    z_tf_inside_half = From(build)
    dz_shld_vv_gap = From(build)
    dz_vv_upper = From(build)
    dz_vv_lower = From(build)
    whtshld = From(fwbs)
    r_cryostat_inboard = From(fwbs)
    helpow = From(heat_transport)
    rxcl = From(buildings)
    trcl = From(buildings)
    row = From(buildings)
    wgt = From(buildings)
    shmf = From(buildings)
    clh2 = From(buildings)
    dz_tf_cryostat = From(buildings)
    stcl = From(buildings)
    rbvfac = From(buildings)
    rbwt = From(buildings)
    rbrt = From(buildings)
    fndt = From(buildings)
    hcwt = From(buildings)
    hccl = From(buildings)
    wgt2 = From(buildings)
    mbvfac = From(buildings)
    wsvfac = From(buildings)
    tfcbv = From(buildings)
    pfbldgm3 = From(buildings)
    esbldgm3 = From(buildings)
    pibv = From(buildings)
    triv = From(buildings)
    conv = From(buildings)
    admv = From(buildings)
    shov = From(buildings)

    cryvol = OutputInto(buildings)
    volrci = OutputInto(buildings)
    rbvol = OutputInto(buildings)
    rmbvol = OutputInto(buildings)
    wsvol = OutputInto(buildings)
    elevol = OutputInto(buildings)
    wrbi = OutputInto(buildings)
    a_plant_floor_effective = OutputInto(buildings)
    admvol = OutputInto(buildings)
    shovol = OutputInto(buildings)
    convol = OutputInto(buildings)
    volnucb = OutputInto(buildings)


class BldgsSizesBase(ExplicitFunction):
    """cottax node family: `calculate_bldgs_sizes` -- bodiless base. Two occupants,
    keyed on `.current_drive.i_hcd_primary`'s method: `BldgsSizesNeutralBeam`/
    `BldgsSizesOtherHcd`. `i_hcd_primary` has 12 values but `bldgs_sizes` reads it to
    decide one bit (whether the method is neutral-beam-shaped) -- see this unit's
    `models/` module docstring. Reads are declared once, here, and shared by both arms;
    each arm names only its `fn`.
    """

    r_pf_coil_outer_max = From(pf_coil)
    r_cryostat_inboard = From(fwbs)
    tf_radial_dim = From(buildings)
    bioshld_thk = From(buildings)
    reactor_clrnc = From(buildings)
    transp_clrnc = From(buildings)
    crane_clrnc_h = From(buildings)
    cryostat_clrnc = From(buildings)
    ground_clrnc = From(buildings)
    crane_arm_h = From(buildings)
    tf_vertical_dim = From(buildings)
    nbi_sys_l = From(buildings)
    nbi_sys_w = From(buildings)
    hcd_building_l = From(buildings)
    hcd_building_w = From(buildings)
    hcd_building_h = From(buildings)
    fc_building_l = From(buildings)
    fc_building_w = From(buildings)
    reactor_wall_thk = From(buildings)
    reactor_roof_thk = From(buildings)
    reactor_fndtn_thk = From(buildings)
    life_plant = From(costs)
    z_tf_inside_half = From(build)
    dr_tf_inboard = From(build)
    dr_tf_shld_gap = From(build)
    dz_shld_thermal = From(build)
    dz_shld_vv_gap = From(build)
    dr_shld_inboard = From(build)
    dr_blkt_inboard = From(build)
    dr_fw_inboard = From(build)
    rmajor = From(physics)
    rminor = From(physics)
    dr_fw_plasma_gap_inboard = From(build)
    n_tf_coils = From(tfcoil)
    hot_sepdist = From(buildings)
    qnty_sfty_fac = From(buildings)
    dr_fw_outboard = From(build)
    dr_blkt_outboard = From(build)
    dr_shld_outboard = From(build)
    dr_fw_plasma_gap_outboard = From(build)
    life_div_fpy = From(costs)
    dz_divertor = From(divertor)
    cplife = From(costs)
    i_tf_sup = From(tfcoil)
    r_cp_top = From(build)
    hotcell_h = From(buildings)
    chemlab_l = From(buildings)
    chemlab_w = From(buildings)
    chemlab_h = From(buildings)
    heat_sink_l = From(buildings)
    heat_sink_w = From(buildings)
    heat_sink_h = From(buildings)
    aux_build_l = From(buildings)
    aux_build_w = From(buildings)
    aux_build_h = From(buildings)
    magnet_trains_l = From(buildings)
    magnet_trains_w = From(buildings)
    magnet_trains_h = From(buildings)
    magnet_pulse_l = From(buildings)
    magnet_pulse_w = From(buildings)
    magnet_pulse_h = From(buildings)
    control_buildings_l = From(buildings)
    control_buildings_w = From(buildings)
    control_buildings_h = From(buildings)
    warm_shop_l = From(buildings)
    warm_shop_w = From(buildings)
    warm_shop_h = From(buildings)
    workshop_l = From(buildings)
    workshop_w = From(buildings)
    workshop_h = From(buildings)
    robotics_l = From(buildings)
    robotics_w = From(buildings)
    robotics_h = From(buildings)
    maint_cont_l = From(buildings)
    maint_cont_w = From(buildings)
    maint_cont_h = From(buildings)
    cryomag_l = From(buildings)
    cryomag_w = From(buildings)
    cryomag_h = From(buildings)
    cryostore_l = From(buildings)
    cryostore_w = From(buildings)
    cryostore_h = From(buildings)
    auxcool_l = From(buildings)
    auxcool_w = From(buildings)
    auxcool_h = From(buildings)
    elecdist_l = From(buildings)
    elecdist_w = From(buildings)
    elecdist_h = From(buildings)
    elecload_l = From(buildings)
    elecload_w = From(buildings)
    elecload_h = From(buildings)
    elecstore_l = From(buildings)
    elecstore_w = From(buildings)
    elecstore_h = From(buildings)
    turbine_hall_l = From(buildings)
    turbine_hall_w = From(buildings)
    turbine_hall_h = From(buildings)
    ilw_smelter_l = From(buildings)
    ilw_smelter_w = From(buildings)
    ilw_smelter_h = From(buildings)
    ilw_storage_l = From(buildings)
    ilw_storage_w = From(buildings)
    ilw_storage_h = From(buildings)
    llw_storage_l = From(buildings)
    llw_storage_w = From(buildings)
    llw_storage_h = From(buildings)
    hw_storage_l = From(buildings)
    hw_storage_w = From(buildings)
    hw_storage_h = From(buildings)
    tw_storage_l = From(buildings)
    tw_storage_w = From(buildings)
    tw_storage_h = From(buildings)
    gas_buildings_l = From(buildings)
    gas_buildings_w = From(buildings)
    gas_buildings_h = From(buildings)
    water_buildings_l = From(buildings)
    water_buildings_w = From(buildings)
    water_buildings_h = From(buildings)
    sec_buildings_l = From(buildings)
    sec_buildings_w = From(buildings)
    sec_buildings_h = From(buildings)
    staff_buildings_area = From(buildings)
    staff_buildings_h = From(buildings)

    reactor_hall_l = OutputInto(buildings)
    reactor_hall_w = OutputInto(buildings)
    reactor_hall_h = OutputInto(buildings)
    a_plant_floor_effective = OutputInto(buildings)
    volnucb = OutputInto(buildings)


class BldgsSizesNeutralBeam(BldgsSizesBase, WrapsFunction):
    """`i_hcd_primary`'s method is `NEUTRAL_BEAM` (`ITER_NEUTRAL_BEAM`/
    `CULHAM_NEUTRAL_BEAM`) -- `calculate_bldgs_sizes_neutral_beam`.
    """

    fn = calculate_bldgs_sizes_neutral_beam


class BldgsSizesOtherHcd(BldgsSizesBase, WrapsFunction):
    """Every other `i_hcd_primary` value -- `calculate_bldgs_sizes_other_hcd`."""

    fn = calculate_bldgs_sizes_other_hcd
