"""Harness cases for the ported TF coil nuclear heating (chunk 1F).

Only the SUPERCONDUCTING branch is ported (see the port's module docstring), so the
reference adapter pins `i_tf_sup` to that value rather than exposing it as a sample
argument -- consistent with `switches.md`'s `i_tf_sup` split decision.
"""

from functional_process._harness import Tier1Contract, legacy_sample
from functional_process.models.stellarator.stellarator_F_tf_nuclear_heating import (
    calculate_sc_tf_coil_nuclear_heating,
)
from process.core.model import DataStructure
from process.models.stellarator.stellarator import Stellarator
from process.models.tfcoil.base import TFConductorModel


def _reference_sc_tf_coil_nuclear_heating(
    dr_shld_inboard,
    dr_fw_inboard,
    dr_blkt_inboard,
    dr_shld_outboard,
    dr_fw_outboard,
    dr_blkt_outboard,
    dr_tf_wp_with_insulation,
    dx_tf_wp_insulation,
    pflux_fw_neutron_mw,
    tfsai,
    tfsao,
    dr_tf_plasma_case,
    f_t_plant_available,
    life_plant,
):
    """Call PROCESS's `Stellarator.sc_tf_coil_nuclear_heating_iter90` (SC branch)."""
    stellarator = Stellarator(*([None] * 12))
    stellarator.data = DataStructure()
    data = stellarator.data

    data.tfcoil.i_tf_sup = TFConductorModel.SUPERCONDUCTING
    data.build.dr_shld_inboard = dr_shld_inboard
    data.build.dr_fw_inboard = dr_fw_inboard
    data.build.dr_blkt_inboard = dr_blkt_inboard
    data.build.dr_shld_outboard = dr_shld_outboard
    data.build.dr_fw_outboard = dr_fw_outboard
    data.build.dr_blkt_outboard = dr_blkt_outboard
    data.tfcoil.dr_tf_wp_with_insulation = dr_tf_wp_with_insulation
    data.tfcoil.dx_tf_wp_insulation = dx_tf_wp_insulation
    data.physics.pflux_fw_neutron_mw = pflux_fw_neutron_mw
    data.tfcoil.tfsai = tfsai
    data.tfcoil.tfsao = tfsao
    data.tfcoil.dr_tf_plasma_case = dr_tf_plasma_case
    data.costs.f_t_plant_available = f_t_plant_available
    data.costs.life_plant = life_plant

    (
        coilhtmx,
        dpacop,
        htheci,
        flu_tf_neutron_fast_peak,
        pheci,
        pheco,
        ptfiwp,
        ptfowp,
        raddose,
        p_tf_nuclear_heat_mw,
    ) = stellarator.sc_tf_coil_nuclear_heating_iter90()

    return (
        coilhtmx,
        dpacop,
        htheci,
        flu_tf_neutron_fast_peak,
        pheci,
        pheco,
        ptfiwp,
        ptfowp,
        raddose,
        p_tf_nuclear_heat_mw,
    )


class TestScTfCoilNuclearHeating(Tier1Contract):
    """`sc_tf_coil_nuclear_heating_iter90` (SC branch) ->
    `calculate_sc_tf_coil_nuclear_heating`.
    """

    audit_record = "models/stellarator/stellarator_F_tf_nuclear_heating.md"
    reference = _reference_sc_tf_coil_nuclear_heating
    ported = calculate_sc_tf_coil_nuclear_heating

    # tests/unit/models/stellarator/test_stellarator.py::test_sctfcoil_nuclear_heating_iter90,
    # generated from a modified stellarator_helias.IN.DAT. tfsai/tfsao are both 0 in this
    # sample, which zeroes pheci/pheco/ptfiwp/ptfowp/p_tf_nuclear_heat_mw -- the fuzz
    # bounds below exercise the nonzero regime the legacy point can't.
    samples = [
        legacy_sample(
            "sctfcoil-heating-helias",
            dr_blkt_inboard=0.83499999999999996,
            dr_blkt_outboard=1.085,
            dr_fw_inboard=0.018000000000000002,
            dr_fw_outboard=0.018000000000000002,
            dr_shld_inboard=0.20000000000000001,
            dr_shld_outboard=0.20000000000000001,
            f_t_plant_available=0.75000000000000011,
            life_plant=40,
            pflux_fw_neutron_mw=0.61095969282042206,
            dr_tf_plasma_case=0.050000000000000003,
            tfsai=0,
            tfsao=0,
            dr_tf_wp_with_insulation=0.73180646211514355,
            dx_tf_wp_insulation=0.01,
        ),
    ]

    fuzz_bounds = {
        "dr_shld_inboard": (0.05, 1.0),
        "dr_fw_inboard": (0.005, 0.1),
        "dr_blkt_inboard": (0.1, 2.0),
        "dr_shld_outboard": (0.05, 1.0),
        "dr_fw_outboard": (0.005, 0.1),
        "dr_blkt_outboard": (0.1, 2.0),
        "dr_tf_wp_with_insulation": (0.1, 2.0),
        "dx_tf_wp_insulation": (0.001, 0.1),
        "pflux_fw_neutron_mw": (0.01, 5.0),
        "tfsai": (1.0, 1.0e3),
        "tfsao": (1.0, 1.0e3),
        "dr_tf_plasma_case": (0.01, 0.5),
        "f_t_plant_available": (0.3, 0.95),
        "life_plant": (10.0, 60.0),
    }
