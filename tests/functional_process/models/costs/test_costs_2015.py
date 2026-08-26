"""Harness cases for the ported subset of `costs/costs_2015.py` (registry unit #18, the
`i_cost_model` counterpart to `costs.py`).

Both ported functions (`calculate_building_costs`, `calculate_land_costs`) are tier-1,
branch-free, fixed-length -- see `costs_2015.md`. Legacy samples are lifted from
`tests/unit/models/test_costs_2015.py`'s own fixtures (`calcbuildingcostsparam`/
`calclandcostsparam`), the same free-oracle reuse `test_harness.md` describes.
"""

from functional_process._harness import Tier1Contract, legacy_sample
from functional_process.models.costs.costs_2015 import (
    calculate_building_costs,
    calculate_land_costs,
)
from process.core.model import DataStructure
from process.models.costs.costs_2015 import Costs2015


def _make_costs2015():
    costs = Costs2015()
    costs.data = DataStructure()
    return costs


def _reference_building_costs(
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
    costs = _make_costs2015()
    costs.data.costs.cost_factor_buildings = cost_factor_buildings
    costs.data.costs.light_build_cost_per_vol = light_build_cost_per_vol
    costs.data.costs.tok_build_cost_per_vol = tok_build_cost_per_vol
    costs.data.fwbs.r_cryostat_inboard = r_cryostat_inboard
    costs.data.fwbs.z_cryostat_half_inside = z_cryostat_half_inside
    costs.data.current_drive.pwpnb = pwpnb
    costs.data.heat_transport.helpow = helpow
    costs.data.pf_coil.r_pf_coil_outer_max = r_pf_coil_outer_max
    costs.data.tfcoil.c_tf_total = c_tf_total
    costs.data.tfcoil.n_tf_coils = n_tf_coils
    costs.data.tfcoil.e_tf_magnetic_stored_total_gj = e_tf_magnetic_stored_total_gj
    costs.data.heat_transport.p_plant_primary_heat_mw = p_plant_primary_heat_mw
    costs.data.heat_transport.p_plant_secondary_heat_mw = p_plant_secondary_heat_mw
    costs.calc_building_costs()
    c = costs.data.costs_2015
    return (
        tuple(c.s_cost_factor[0:9]),
        tuple(c.s_cref[0:9]),
        tuple(c.s_k[0:9]),
        tuple(c.s_kref[0:9]),
        tuple(c.s_cost[0:9]),
    )


def _reference_land_costs(
    cost_factor_land,
    r_cryostat_inboard,
    costexp,
    dh_tf_inner_bore,
    dr_tf_inner_bore,
    dr_tf_inboard,
):
    costs = _make_costs2015()
    costs.data.costs.cost_factor_land = cost_factor_land
    costs.data.fwbs.r_cryostat_inboard = r_cryostat_inboard
    costs.data.costs.costexp = costexp
    costs.data.build.dh_tf_inner_bore = dh_tf_inner_bore
    costs.data.build.dr_tf_inner_bore = dr_tf_inner_bore
    costs.data.build.dr_tf_inboard = dr_tf_inboard
    costs.calc_land_costs()
    c = costs.data.costs_2015
    return (
        tuple(c.s_cost_factor[9:13]),
        tuple(c.s_cref[9:13]),
        tuple(c.s_k[9:13]),
        tuple(c.s_kref[9:13]),
        tuple(c.s_cost[9:13]),
    )


class TestBuildingCosts(Tier1Contract):
    audit_record = "models/costs/costs_2015.md"
    reference = _reference_building_costs
    ported = calculate_building_costs

    samples = [
        legacy_sample(
            "baseline",
            cost_factor_buildings=1.0,
            light_build_cost_per_vol=270.0,
            tok_build_cost_per_vol=1283.0,
            r_cryostat_inboard=8.4,
            z_cryostat_half_inside=8.75,
            pwpnb=125.0,
            helpow=88000.0,
            r_pf_coil_outer_max=6.5,
            c_tf_total=1.6e8,
            n_tf_coils=16.0,
            e_tf_magnetic_stored_total_gj=140.0,
            p_plant_primary_heat_mw=2200.0,
            p_plant_secondary_heat_mw=300.0,
        ),
    ]
    fuzz_bounds = {
        "light_build_cost_per_vol": (100.0, 500.0),
        "tok_build_cost_per_vol": (500.0, 2000.0),
        "r_cryostat_inboard": (3.0, 15.0),
        "z_cryostat_half_inside": (3.0, 15.0),
        "pwpnb": (0.0, 300.0),
        "helpow": (1.0e4, 2.0e5),
        "r_pf_coil_outer_max": (2.0, 12.0),
        "c_tf_total": (1.0e7, 3.0e8),
        "n_tf_coils": (10.0, 24.0),
        "e_tf_magnetic_stored_total_gj": (10.0, 300.0),
        "p_plant_primary_heat_mw": (100.0, 4000.0),
        "p_plant_secondary_heat_mw": (10.0, 1000.0),
    }
    fuzz_fixed = {"cost_factor_buildings": 1.0}


class TestLandCosts(Tier1Contract):
    audit_record = "models/costs/costs_2015.md"
    reference = _reference_land_costs
    ported = calculate_land_costs

    samples = [
        legacy_sample(
            "baseline",
            cost_factor_land=1.0,
            r_cryostat_inboard=8.4,
            costexp=0.8,
            dh_tf_inner_bore=14.0,
            dr_tf_inner_bore=14.0,
            dr_tf_inboard=1.0,
        ),
    ]
    fuzz_bounds = {
        "r_cryostat_inboard": (3.0, 15.0),
        "dh_tf_inner_bore": (5.0, 25.0),
        "dr_tf_inner_bore": (5.0, 25.0),
        "dr_tf_inboard": (0.2, 3.0),
    }
    fuzz_fixed = {"cost_factor_land": 1.0, "costexp": 0.8}
