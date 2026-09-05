"""Pure-functional port of `process/models/buildings.py`'s `Buildings.run()` (unit #15).

Audit record: `functional_process/_audit/units/models/buildings.md`. `run()` is a short
preamble (TF coil envelope geometry, unconditional) followed by a dispatch, keyed by
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
import jax.numpy as jnp  # noqa: F401
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.models.buildings.buildings import (
    calculate_bldgs,
    calculate_bldgs_sizes,
    calculate_shield_height,
    calculate_tf_coil_envelope,
)
from functional_process.models.safe_math import safe_pow  # noqa: F401
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
from functional_process.vocabulary import (
    CurrentDriveMethodType,
    CurrentDriveModel,
)


class TfCoilEnvelope(ExplicitFunction):
    """cottax node: `calculate_tf_coil_envelope`, ports declared.

    No real `data` writes -- see module docstring. `tfro`/`tfri`/`tf_radial_dim`/
    `tf_vertical_dim`/`tfmtn` are minted `VarPath`s under `.buildings.*` (invented,
    same reasoning as `build.py`'s `a_fw_total_unadjusted`: PROCESS never stores these,
    they are locals in `run()`), so that this node has somewhere to write its outputs
    for `Bldgs`/`BldgsSizes` to read from.
    """

    tfro = OutputInto(buildings)
    tfri = OutputInto(buildings)
    tf_radial_dim = OutputInto(buildings)
    tf_vertical_dim = OutputInto(buildings)
    tfmtn = OutputInto(buildings)

    def __call__(
        self,
        r_tf_outboard_mid=From(build),
        dr_tf_outboard=From(build),
        r_tf_inboard_mid=From(build),
        dr_tf_inboard=From(build),
        z_tf_inside_half=From(build),
        m_tf_coils_total=From(tfcoil),
        n_tf_coils=From(tfcoil),
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

    def __call__(
        self,
        r_pf_coil_outer_max=From(pf_coil),
        m_pf_coil_max=From(pf_coil),
        tfro=From(buildings),
        tfri=From(buildings),
        tf_vertical_dim=From(buildings),
        tfmtn=From(buildings),
        n_tf_coils=From(tfcoil),
        r_shld_outboard_outer=From(build),
        r_shld_inboard_inner=From(build),
        z_tf_inside_half=From(build),
        dz_shld_vv_gap=From(build),
        dz_vv_upper=From(build),
        dz_vv_lower=From(build),
        whtshld=From(fwbs),
        r_cryostat_inboard=From(fwbs),
        helpow=From(heat_transport),
        rxcl=From(buildings),
        trcl=From(buildings),
        row=From(buildings),
        wgt=From(buildings),
        shmf=From(buildings),
        clh2=From(buildings),
        dz_tf_cryostat=From(buildings),
        stcl=From(buildings),
        rbvfac=From(buildings),
        rbwt=From(buildings),
        rbrt=From(buildings),
        fndt=From(buildings),
        hcwt=From(buildings),
        hccl=From(buildings),
        wgt2=From(buildings),
        mbvfac=From(buildings),
        wsvfac=From(buildings),
        tfcbv=From(buildings),
        pfbldgm3=From(buildings),
        esbldgm3=From(buildings),
        pibv=From(buildings),
        triv=From(buildings),
        conv=From(buildings),
        admv=From(buildings),
        shov=From(buildings),
    ):
        shh = calculate_shield_height(
            z_tf_inside_half, dz_shld_vv_gap, dz_vv_upper, dz_vv_lower
        )
        return calculate_bldgs(
            r_pf_coil_outer_max,
            m_pf_coil_max,
            tfro,
            tfri,
            tf_vertical_dim,
            tfmtn,
            n_tf_coils,
            r_shld_outboard_outer,
            r_shld_inboard_inner,
            shh,
            whtshld,
            r_cryostat_inboard,
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

    reactor_hall_l = OutputInto(buildings)
    reactor_hall_w = OutputInto(buildings)
    reactor_hall_h = OutputInto(buildings)
    a_plant_floor_effective = OutputInto(buildings)
    volnucb = OutputInto(buildings)

    def __call__(
        self,
        r_pf_coil_outer_max=From(pf_coil),
        r_cryostat_inboard=From(fwbs),
        tf_radial_dim=From(buildings),
        bioshld_thk=From(buildings),
        reactor_clrnc=From(buildings),
        transp_clrnc=From(buildings),
        crane_clrnc_h=From(buildings),
        cryostat_clrnc=From(buildings),
        ground_clrnc=From(buildings),
        crane_arm_h=From(buildings),
        tf_vertical_dim=From(buildings),
        nbi_sys_l=From(buildings),
        nbi_sys_w=From(buildings),
        hcd_building_l=From(buildings),
        hcd_building_w=From(buildings),
        hcd_building_h=From(buildings),
        fc_building_l=From(buildings),
        fc_building_w=From(buildings),
        reactor_wall_thk=From(buildings),
        reactor_roof_thk=From(buildings),
        reactor_fndtn_thk=From(buildings),
        life_plant=From(costs),
        z_tf_inside_half=From(build),
        dr_tf_inboard=From(build),
        dr_tf_shld_gap=From(build),
        dz_shld_thermal=From(build),
        dz_shld_vv_gap=From(build),
        dr_shld_inboard=From(build),
        dr_blkt_inboard=From(build),
        dr_fw_inboard=From(build),
        rmajor=From(physics),
        rminor=From(physics),
        dr_fw_plasma_gap_inboard=From(build),
        n_tf_coils=From(tfcoil),
        hot_sepdist=From(buildings),
        qnty_sfty_fac=From(buildings),
        dr_fw_outboard=From(build),
        dr_blkt_outboard=From(build),
        dr_shld_outboard=From(build),
        dr_fw_plasma_gap_outboard=From(build),
        life_div_fpy=From(costs),
        dz_divertor=From(divertor),
        cplife=From(costs),
        i_tf_sup=From(tfcoil),
        r_cp_top=From(build),
        hotcell_h=From(buildings),
        chemlab_l=From(buildings),
        chemlab_w=From(buildings),
        chemlab_h=From(buildings),
        heat_sink_l=From(buildings),
        heat_sink_w=From(buildings),
        heat_sink_h=From(buildings),
        aux_build_l=From(buildings),
        aux_build_w=From(buildings),
        aux_build_h=From(buildings),
        magnet_trains_l=From(buildings),
        magnet_trains_w=From(buildings),
        magnet_trains_h=From(buildings),
        magnet_pulse_l=From(buildings),
        magnet_pulse_w=From(buildings),
        magnet_pulse_h=From(buildings),
        control_buildings_l=From(buildings),
        control_buildings_w=From(buildings),
        control_buildings_h=From(buildings),
        warm_shop_l=From(buildings),
        warm_shop_w=From(buildings),
        warm_shop_h=From(buildings),
        workshop_l=From(buildings),
        workshop_w=From(buildings),
        workshop_h=From(buildings),
        robotics_l=From(buildings),
        robotics_w=From(buildings),
        robotics_h=From(buildings),
        maint_cont_l=From(buildings),
        maint_cont_w=From(buildings),
        maint_cont_h=From(buildings),
        cryomag_l=From(buildings),
        cryomag_w=From(buildings),
        cryomag_h=From(buildings),
        cryostore_l=From(buildings),
        cryostore_w=From(buildings),
        cryostore_h=From(buildings),
        auxcool_l=From(buildings),
        auxcool_w=From(buildings),
        auxcool_h=From(buildings),
        elecdist_l=From(buildings),
        elecdist_w=From(buildings),
        elecdist_h=From(buildings),
        elecload_l=From(buildings),
        elecload_w=From(buildings),
        elecload_h=From(buildings),
        elecstore_l=From(buildings),
        elecstore_w=From(buildings),
        elecstore_h=From(buildings),
        turbine_hall_l=From(buildings),
        turbine_hall_w=From(buildings),
        turbine_hall_h=From(buildings),
        ilw_smelter_l=From(buildings),
        ilw_smelter_w=From(buildings),
        ilw_smelter_h=From(buildings),
        ilw_storage_l=From(buildings),
        ilw_storage_w=From(buildings),
        ilw_storage_h=From(buildings),
        llw_storage_l=From(buildings),
        llw_storage_w=From(buildings),
        llw_storage_h=From(buildings),
        hw_storage_l=From(buildings),
        hw_storage_w=From(buildings),
        hw_storage_h=From(buildings),
        tw_storage_l=From(buildings),
        tw_storage_w=From(buildings),
        tw_storage_h=From(buildings),
        gas_buildings_l=From(buildings),
        gas_buildings_w=From(buildings),
        gas_buildings_h=From(buildings),
        water_buildings_l=From(buildings),
        water_buildings_w=From(buildings),
        water_buildings_h=From(buildings),
        sec_buildings_l=From(buildings),
        sec_buildings_w=From(buildings),
        sec_buildings_h=From(buildings),
        staff_buildings_area=From(buildings),
        staff_buildings_h=From(buildings),
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
