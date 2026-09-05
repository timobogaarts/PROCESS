"""Pure-functional port of the AC/electric-production sub-unit of
`process/models/power.py` (registry unit #14, chunk C).

Audit record: `functional_process/_audit/units/models/power/electric_production.md`.
Covers `Power.acpow` (696-813), `Power.power_profiles_over_time` (2632-2825) and
`Power.plant_electric_production` (1631-1772) -- see the audit record's data-footprint
table for the full trace.

All three are tier-1: no internal iteration, no calls into any other model. The time
axis `power_profiles_over_time` builds is always exactly 7 points (`PulseTimings.
total_pulse_cumulative` is `len()` of a fixed 7-tuple of cumulative sums over the six
phase durations, never data-dependent in length) -- so every array in this chunk has
a static, compile-time-known shape; there is no dynamic-shape difficulty here at all.
"""

import equinox as eqx
import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.models.switch_enums import (
    BlanketDualCoolantModel,
    CostOfElectricityModel,
    SphericalTokamakModel,
)
from functional_process.paths import (
    buildings,
    fwbs,
    heat_transport,
    pf_coil,
    pf_power,
    physics,
    power,
    tfcoil,
    times,
)
from functional_process.models.power.electric_production import (
    calculate_acpow,
    calculate_acpow_line,
    calculate_acpow_motor_generator_flywheel,
    calculate_plant_electric_production,
    calculate_plant_electric_production_reactor,
    calculate_plant_electric_production_resistive_centrepost_liquid_breeder,
    centrepost_coolant_pump_power_absent,
    centrepost_coolant_pump_power_resistive,
    gross_electric_power_liquid_breeder,
    gross_electric_power_single_coolant,
    power_profiles_over_time,
)
from functional_process.vocabulary import PumpingPowerModelTypes, TFConductorModel

# ruff's docstring rules treat `__all__` membership as the definition of "public" once
# one is present, so this lists every public name this module resolved before step 2 of
# `_audit/formulas_split.md` moved the pure functions out -- not just the few that turn
# out to be unused here (see this file's own first-pass commit for why a partial list
# is the wrong move: it silently drops the D101/D102 checks on the classes below).
__all__ = [
    "Acpow",
    "AcpowLine",
    "AcpowMotorGeneratorFlywheel",
    "BlanketDualCoolantModel",
    "CostOfElectricityModel",
    "ExplicitFunction",
    "From",
    "OutputInto",
    "PlantElectricProductionLiquidBreeder",
    "PlantElectricProductionReactor",
    "PlantElectricProductionResistiveCentrepostLiquidBreeder",
    "PlantElectricProductionResistiveCentrepostSingleCoolant",
    "PlantElectricProductionSingleCoolant",
    "PowerProfilesOverTime",
    "PumpingPowerModelTypes",
    "SphericalTokamakModel",
    "TFConductorModel",
    "buildings",
    "calculate_acpow",
    "calculate_acpow_line",
    "calculate_acpow_motor_generator_flywheel",
    "calculate_plant_electric_production",
    "calculate_plant_electric_production_reactor",
    "calculate_plant_electric_production_resistive_centrepost_liquid_breeder",
    "centrepost_coolant_pump_power_absent",
    "centrepost_coolant_pump_power_resistive",
    "eqx",
    "fwbs",
    "gross_electric_power_liquid_breeder",
    "gross_electric_power_single_coolant",
    "heat_transport",
    "jnp",
    "pf_coil",
    "pf_power",
    "physics",
    "power",
    "power_profiles_over_time",
    "tfcoil",
    "times",
]


class Acpow(ExplicitFunction):
    """The `calculate_acpow` family -- one occupant per
    `.pf_power.i_pf_energy_storage_source` value.

    **The switch was an `eqx.field(static=True)` here and is gone**
    (`_audit/next_steps.md` §14.2). The two arms' reads are **complementary**: the line
    arm reads `.heat_transport.peakmva` and not `fmgdmw`, the flywheel arm the reverse.
    One node carrying the switch declared both, so exactly one edge was invented either
    way -- the smallest and cleanest case in `switch_kwarg_survey.md` band (b3).
    """

    pacpmw = OutputInto(heat_transport)
    tlvpmw = OutputInto(heat_transport)


class AcpowLine(Acpow):
    """`i_pf_energy_storage_source == LINE` (2) -- the reference run's.

    **One read leaves with this occupant**: `.heat_transport.fmgdmw`.
    """

    def __call__(
        self,
        p_tf_electric_supplies_mw=From(heat_transport),
        srcktpm=From(pf_power),
        peakmva=From(heat_transport),
        p_hcd_electric_total_mw=From(heat_transport),
        p_cryo_plant_electric_mw=From(heat_transport),
        vachtmw=From(heat_transport),
        p_coolant_pump_elec_total_mw=From(heat_transport),
        p_tritium_plant_electric_mw=From(heat_transport),
        p_plant_electric_base_total_mw=From(heat_transport),
    ):
        return calculate_acpow_line(
            p_tf_electric_supplies_mw,
            srcktpm,
            peakmva,
            p_hcd_electric_total_mw,
            p_cryo_plant_electric_mw,
            vachtmw,
            p_coolant_pump_elec_total_mw,
            p_tritium_plant_electric_mw,
            p_plant_electric_base_total_mw,
        )


class AcpowMotorGeneratorFlywheel(Acpow):
    """`i_pf_energy_storage_source == MGF` (1) -- all power from motor-generator
    flywheel units, PROCESS's own default (`pf_power_variables.py:18`).

    **Reads `.heat_transport.fmgdmw` and not `.heat_transport.peakmva`.**
    """

    def __call__(
        self,
        p_tf_electric_supplies_mw=From(heat_transport),
        srcktpm=From(pf_power),
        p_hcd_electric_total_mw=From(heat_transport),
        p_cryo_plant_electric_mw=From(heat_transport),
        vachtmw=From(heat_transport),
        p_coolant_pump_elec_total_mw=From(heat_transport),
        p_tritium_plant_electric_mw=From(heat_transport),
        p_plant_electric_base_total_mw=From(heat_transport),
        fmgdmw=From(heat_transport),
    ):
        return calculate_acpow_motor_generator_flywheel(
            p_tf_electric_supplies_mw,
            srcktpm,
            p_hcd_electric_total_mw,
            p_cryo_plant_electric_mw,
            vachtmw,
            p_coolant_pump_elec_total_mw,
            p_tritium_plant_electric_mw,
            p_plant_electric_base_total_mw,
            fmgdmw,
        )


class PowerProfilesOverTime(ExplicitFunction):
    """cottax node: `power_profiles_over_time`."""

    e_plant_net_electric_pulse_kwh = OutputInto(power)
    e_plant_net_electric_pulse_mj = OutputInto(power)
    p_plant_electric_base_total_profile_mw = OutputInto(power)
    p_plant_electric_gross_profile_mw = OutputInto(power)
    p_plant_electric_net_profile_mw = OutputInto(power)
    p_hcd_electric_total_profile_mw = OutputInto(power)
    p_coolant_pump_elec_total_profile_mw = OutputInto(power)
    p_tf_electric_supplies_profile_mw = OutputInto(power)
    p_pf_electric_supplies_profile_mw = OutputInto(power)
    vachtmw_profile_mw = OutputInto(power)
    p_tritium_plant_electric_profile_mw = OutputInto(power)
    p_cryo_plant_electric_profile_mw = OutputInto(power)
    p_fusion_total_profile_mw = OutputInto(power)

    def __call__(
        self,
        p_plant_electric_base_total_mw=From(heat_transport),
        p_cryo_plant_electric_mw=From(heat_transport),
        p_tritium_plant_electric_mw=From(heat_transport),
        vachtmw=From(heat_transport),
        p_tf_electric_supplies_mw=From(heat_transport),
        p_pf_electric_supplies_mw=From(pf_coil),
        p_coolant_pump_elec_total_mw=From(heat_transport),
        p_hcd_electric_total_mw=From(heat_transport),
        p_fusion_total_mw=From(physics),
        p_plant_electric_gross_mw=From(heat_transport),
        p_plant_electric_net_mw=From(heat_transport),
        t_plant_pulse_coil_precharge=From(times),
        t_plant_pulse_plasma_current_ramp_up=From(times),
        t_plant_pulse_fusion_ramp=From(times),
        t_plant_pulse_burn=From(times),
        t_plant_pulse_plasma_current_ramp_down=From(times),
        t_plant_pulse_dwell=From(times),
    ):
        return power_profiles_over_time(
            p_plant_electric_base_total_mw,
            p_cryo_plant_electric_mw,
            p_tritium_plant_electric_mw,
            vachtmw,
            p_tf_electric_supplies_mw,
            p_pf_electric_supplies_mw,
            p_coolant_pump_elec_total_mw,
            p_hcd_electric_total_mw,
            p_fusion_total_mw,
            p_plant_electric_gross_mw,
            p_plant_electric_net_mw,
            t_plant_pulse_coil_precharge,
            t_plant_pulse_plasma_current_ramp_up,
            t_plant_pulse_fusion_ramp,
            t_plant_pulse_burn,
            t_plant_pulse_plasma_current_ramp_down,
            t_plant_pulse_dwell,
        )


class PlantElectricProductionReactor(ExplicitFunction):
    """The `calculate_plant_electric_production` family at `ireactor == 1` -- one
    occupant per `(itart, i_tf_sup)` x `(i_blkt_dual_coolant,
    i_p_coolant_pumping)` arm pair.

    **Why this exists as a separate class, and why `PlantElectricProduction` above is
    not registerable.** `PlantElectricProduction` declares
    `p_plant_electric_gross_mw` / `p_turbine_loss_mw` / `p_plant_electric_recirc_mw` /
    `p_plant_electric_net_mw` / `f_p_plant_electric_recirc` as both `Output`s and
    `FromExactly`s, so `to_graph` refuses it outright (*"reads [...], which it also owns"*,
    confirmed directly -- `total_process.py`'s own comment records the same refusal).
    That is not a modelling cycle: it is PROCESS's conditional-ownership
    pass-through, and it exists **only on the `ireactor == 0` arm**. Read
    `calculate_plant_electric_production`'s body: all five are assigned inside
    `if ireactor == 1:`, before `power_profiles_over_time` consumes
    `p_plant_electric_gross_mw`/`p_plant_electric_net_mw`, so on that arm not one of
    the five entering values is ever read. `.costs.ireactor` is a static switch
    (`process/main.py` resolves the cost model once per run; it is neither an
    iteration variable nor a scan variable), so which arm is live is a
    graph-assembly-time fact -- exactly `machine_from_indat`'s category. This class is
    the `ireactor == 1` arm with the five dead reads simply not declared, which makes
    it an ordinary acyclic node owning all 23 fields.

    **The five dead parameters are passed as `jnp.nan`, deliberately.** They are
    provably overwritten before use on this arm, so any value would do; `nan` is the
    one that makes a future edit which *starts* reading them fail loudly (a `nan`
    reaching `.heat_transport.p_plant_electric_net_mw` is caught by the very first
    comparison in `mda_harness.compare`) instead of silently substituting a zero.

    `ireactor` is therefore **not** a static field here -- it is structural, spent by
    picking this class over the `ireactor == 0` arm, which is
    `PowerProfilesOverTime` (whose 13 outputs are a strict subset of this node's, and
    which reads the two carried-over values as boundary inputs, exactly as PROCESS's
    `ireactor == 0` run does).

    Registering this closes `.heat_transport.p_plant_electric_net_mw`'s producer gap,
    which matters beyond its own value: `CostOfElectricity` reads that field, so
    without a producer `.costs.coe` -- this run's own objective -- was a function of a
    *boundary input* rather than of the design variables along that whole path, and
    constraint 16 (net electric power, an equality in
    `stellarator_helias.IN.DAT`) had no live argument at all.

    **All four switches were `eqx.field(static=True)`s here and none is now**
    (`_audit/next_steps.md` §14.2). They gate two things and nothing else: whether the
    centrepost coolant pump draws electric power (`itart == 1 and i_tf_sup == 0`) and
    whether the liquid breeder has its own turbine efficiency
    (`i_blkt_dual_coolant > 0 and i_p_coolant_pumping == MECHANICAL`). Both conditions
    are joint, so this is two arm indices rather than four switches, and the four
    occupants below are their product.

    **Three reads leave with the conventional/single-coolant occupant**:
    `.tfcoil.p_cp_coolant_pump_elec`, `.heat_transport.etath_liq` and
    `.power.p_blkt_liquid_breeder_heat_deposited_mw`. The first is the `.tfcoil ->
    .power` edge `switch_kwarg_survey.md` §3 records as `live (1)` for `itart`; the
    other two are its `live (2)` for `i_blkt_dual_coolant`. Both were reported "(joint)"
    there because neither switch decides them alone -- which is exactly why the two arm
    indices are joint here too.
    """

    p_cp_coolant_pump_elec_mw = OutputInto(power)
    p_plant_electric_base_total_mw = OutputInto(heat_transport)
    fachtmw = OutputInto(heat_transport)
    p_plant_core_systems_elec_mw = OutputInto(power)
    p_plant_secondary_heat_mw = OutputInto(heat_transport)
    p_plant_electric_gross_mw = OutputInto(heat_transport)
    p_turbine_loss_mw = OutputInto(power)
    p_plant_electric_recirc_mw = OutputInto(heat_transport)
    p_plant_electric_net_mw = OutputInto(heat_transport)
    f_p_plant_electric_recirc = OutputInto(heat_transport)
    e_plant_net_electric_pulse_kwh = OutputInto(power)
    e_plant_net_electric_pulse_mj = OutputInto(power)
    p_plant_electric_base_total_profile_mw = OutputInto(power)
    p_plant_electric_gross_profile_mw = OutputInto(power)
    p_plant_electric_net_profile_mw = OutputInto(power)
    p_hcd_electric_total_profile_mw = OutputInto(power)
    p_coolant_pump_elec_total_profile_mw = OutputInto(power)
    p_tf_electric_supplies_profile_mw = OutputInto(power)
    p_pf_electric_supplies_profile_mw = OutputInto(power)
    vachtmw_profile_mw = OutputInto(power)
    p_tritium_plant_electric_profile_mw = OutputInto(power)
    p_cryo_plant_electric_profile_mw = OutputInto(power)
    p_fusion_total_profile_mw = OutputInto(power)

    def _production(
        self,
        p_cp_coolant_pump_elec_mw,
        p_plant_electric_gross_mw,
        p_plant_electric_base,
        a_plant_floor_effective,
        pflux_plant_floor_electric,
        p_cryo_plant_electric_mw,
        p_tf_electric_supplies_mw,
        p_tritium_plant_electric_mw,
        vachtmw,
        p_pf_electric_supplies_mw,
        p_hcd_electric_loss_mw,
        p_coolant_pump_loss_total_mw,
        p_div_secondary_heat_mw,
        p_shld_secondary_heat_mw,
        p_hcd_secondary_heat_mw,
        p_tf_nuclear_heat_mw,
        p_plant_primary_heat_mw,
        eta_turbine,
        p_hcd_electric_total_mw,
        p_coolant_pump_elec_total_mw,
        p_fusion_total_mw,
        t_plant_pulse_coil_precharge,
        t_plant_pulse_plasma_current_ramp_up,
        t_plant_pulse_fusion_ramp,
        t_plant_pulse_burn,
        t_plant_pulse_plasma_current_ramp_down,
        t_plant_pulse_dwell,
    ):
        """The twenty-three outputs, given the two quantities the four arms
        disagree about.

        Not a port surface: `_params` reads `__call__`'s signature only
        (`ExplicitFunction._signature_of`), so what each occupant declares is
        still its own parameter list.
        """
        return calculate_plant_electric_production_reactor(
            p_cp_coolant_pump_elec_mw,
            p_plant_electric_gross_mw,
            p_plant_electric_base,
            a_plant_floor_effective,
            pflux_plant_floor_electric,
            p_cryo_plant_electric_mw,
            p_tf_electric_supplies_mw,
            p_tritium_plant_electric_mw,
            vachtmw,
            p_pf_electric_supplies_mw,
            p_hcd_electric_loss_mw,
            p_coolant_pump_loss_total_mw,
            p_div_secondary_heat_mw,
            p_shld_secondary_heat_mw,
            p_hcd_secondary_heat_mw,
            p_tf_nuclear_heat_mw,
            p_plant_primary_heat_mw,
            eta_turbine,
            p_hcd_electric_total_mw,
            p_coolant_pump_elec_total_mw,
            p_fusion_total_mw,
            t_plant_pulse_coil_precharge,
            t_plant_pulse_plasma_current_ramp_up,
            t_plant_pulse_fusion_ramp,
            t_plant_pulse_burn,
            t_plant_pulse_plasma_current_ramp_down,
            t_plant_pulse_dwell,
        )


class PlantElectricProductionSingleCoolant(PlantElectricProductionReactor):
    """No resistive centrepost, one coolant -- the reference run's and the
    conventional tokamak's (`itart = 0`, `i_blkt_dual_coolant = 0`).

    Declares **none** of `.tfcoil.p_cp_coolant_pump_elec`,
    `.power.p_blkt_liquid_breeder_heat_deposited_mw`,
    `.heat_transport.etath_liq`.
    """

    def __call__(
        self,
        p_plant_electric_base=From(heat_transport),
        a_plant_floor_effective=From(buildings),
        pflux_plant_floor_electric=From(heat_transport),
        p_cryo_plant_electric_mw=From(heat_transport),
        p_tf_electric_supplies_mw=From(heat_transport),
        p_tritium_plant_electric_mw=From(heat_transport),
        vachtmw=From(heat_transport),
        p_pf_electric_supplies_mw=From(pf_coil),
        p_hcd_electric_loss_mw=From(heat_transport),
        p_coolant_pump_loss_total_mw=From(heat_transport),
        p_div_secondary_heat_mw=From(heat_transport),
        p_shld_secondary_heat_mw=From(heat_transport),
        p_hcd_secondary_heat_mw=From(heat_transport),
        p_tf_nuclear_heat_mw=From(fwbs),
        p_plant_primary_heat_mw=From(heat_transport),
        eta_turbine=From(heat_transport),
        p_hcd_electric_total_mw=From(heat_transport),
        p_coolant_pump_elec_total_mw=From(heat_transport),
        p_fusion_total_mw=From(physics),
        t_plant_pulse_coil_precharge=From(times),
        t_plant_pulse_plasma_current_ramp_up=From(times),
        t_plant_pulse_fusion_ramp=From(times),
        t_plant_pulse_burn=From(times),
        t_plant_pulse_plasma_current_ramp_down=From(times),
        t_plant_pulse_dwell=From(times),
    ):
        return self._production(
            centrepost_coolant_pump_power_absent(),
            gross_electric_power_single_coolant(p_plant_primary_heat_mw, eta_turbine),
            p_plant_electric_base,
            a_plant_floor_effective,
            pflux_plant_floor_electric,
            p_cryo_plant_electric_mw,
            p_tf_electric_supplies_mw,
            p_tritium_plant_electric_mw,
            vachtmw,
            p_pf_electric_supplies_mw,
            p_hcd_electric_loss_mw,
            p_coolant_pump_loss_total_mw,
            p_div_secondary_heat_mw,
            p_shld_secondary_heat_mw,
            p_hcd_secondary_heat_mw,
            p_tf_nuclear_heat_mw,
            p_plant_primary_heat_mw,
            eta_turbine,
            p_hcd_electric_total_mw,
            p_coolant_pump_elec_total_mw,
            p_fusion_total_mw,
            t_plant_pulse_coil_precharge,
            t_plant_pulse_plasma_current_ramp_up,
            t_plant_pulse_fusion_ramp,
            t_plant_pulse_burn,
            t_plant_pulse_plasma_current_ramp_down,
            t_plant_pulse_dwell,
        )


class PlantElectricProductionLiquidBreeder(PlantElectricProductionReactor):
    """No resistive centrepost; liquid breeder with its own turbine efficiency
    (`i_blkt_dual_coolant > 0` and `i_p_coolant_pumping == MECHANICAL`).
    """

    def __call__(
        self,
        p_blkt_liquid_breeder_heat_deposited_mw=From(power),
        etath_liq=From(heat_transport),
        p_plant_electric_base=From(heat_transport),
        a_plant_floor_effective=From(buildings),
        pflux_plant_floor_electric=From(heat_transport),
        p_cryo_plant_electric_mw=From(heat_transport),
        p_tf_electric_supplies_mw=From(heat_transport),
        p_tritium_plant_electric_mw=From(heat_transport),
        vachtmw=From(heat_transport),
        p_pf_electric_supplies_mw=From(pf_coil),
        p_hcd_electric_loss_mw=From(heat_transport),
        p_coolant_pump_loss_total_mw=From(heat_transport),
        p_div_secondary_heat_mw=From(heat_transport),
        p_shld_secondary_heat_mw=From(heat_transport),
        p_hcd_secondary_heat_mw=From(heat_transport),
        p_tf_nuclear_heat_mw=From(fwbs),
        p_plant_primary_heat_mw=From(heat_transport),
        eta_turbine=From(heat_transport),
        p_hcd_electric_total_mw=From(heat_transport),
        p_coolant_pump_elec_total_mw=From(heat_transport),
        p_fusion_total_mw=From(physics),
        t_plant_pulse_coil_precharge=From(times),
        t_plant_pulse_plasma_current_ramp_up=From(times),
        t_plant_pulse_fusion_ramp=From(times),
        t_plant_pulse_burn=From(times),
        t_plant_pulse_plasma_current_ramp_down=From(times),
        t_plant_pulse_dwell=From(times),
    ):
        return self._production(
            centrepost_coolant_pump_power_absent(),
            gross_electric_power_liquid_breeder(
                p_plant_primary_heat_mw,
                eta_turbine,
                p_blkt_liquid_breeder_heat_deposited_mw,
                etath_liq,
            ),
            p_plant_electric_base,
            a_plant_floor_effective,
            pflux_plant_floor_electric,
            p_cryo_plant_electric_mw,
            p_tf_electric_supplies_mw,
            p_tritium_plant_electric_mw,
            vachtmw,
            p_pf_electric_supplies_mw,
            p_hcd_electric_loss_mw,
            p_coolant_pump_loss_total_mw,
            p_div_secondary_heat_mw,
            p_shld_secondary_heat_mw,
            p_hcd_secondary_heat_mw,
            p_tf_nuclear_heat_mw,
            p_plant_primary_heat_mw,
            eta_turbine,
            p_hcd_electric_total_mw,
            p_coolant_pump_elec_total_mw,
            p_fusion_total_mw,
            t_plant_pulse_coil_precharge,
            t_plant_pulse_plasma_current_ramp_up,
            t_plant_pulse_fusion_ramp,
            t_plant_pulse_burn,
            t_plant_pulse_plasma_current_ramp_down,
            t_plant_pulse_dwell,
        )


class PlantElectricProductionResistiveCentrepostSingleCoolant(
    PlantElectricProductionReactor
):
    """Resistive centrepost (`itart == 1` and `i_tf_sup == 0`), one coolant.

    Reads `.tfcoil.p_cp_coolant_pump_elec`, which the two conventional
    occupants do not.
    """

    def __call__(
        self,
        p_cp_coolant_pump_elec=From(tfcoil),
        p_plant_electric_base=From(heat_transport),
        a_plant_floor_effective=From(buildings),
        pflux_plant_floor_electric=From(heat_transport),
        p_cryo_plant_electric_mw=From(heat_transport),
        p_tf_electric_supplies_mw=From(heat_transport),
        p_tritium_plant_electric_mw=From(heat_transport),
        vachtmw=From(heat_transport),
        p_pf_electric_supplies_mw=From(pf_coil),
        p_hcd_electric_loss_mw=From(heat_transport),
        p_coolant_pump_loss_total_mw=From(heat_transport),
        p_div_secondary_heat_mw=From(heat_transport),
        p_shld_secondary_heat_mw=From(heat_transport),
        p_hcd_secondary_heat_mw=From(heat_transport),
        p_tf_nuclear_heat_mw=From(fwbs),
        p_plant_primary_heat_mw=From(heat_transport),
        eta_turbine=From(heat_transport),
        p_hcd_electric_total_mw=From(heat_transport),
        p_coolant_pump_elec_total_mw=From(heat_transport),
        p_fusion_total_mw=From(physics),
        t_plant_pulse_coil_precharge=From(times),
        t_plant_pulse_plasma_current_ramp_up=From(times),
        t_plant_pulse_fusion_ramp=From(times),
        t_plant_pulse_burn=From(times),
        t_plant_pulse_plasma_current_ramp_down=From(times),
        t_plant_pulse_dwell=From(times),
    ):
        return self._production(
            centrepost_coolant_pump_power_resistive(p_cp_coolant_pump_elec),
            gross_electric_power_single_coolant(p_plant_primary_heat_mw, eta_turbine),
            p_plant_electric_base,
            a_plant_floor_effective,
            pflux_plant_floor_electric,
            p_cryo_plant_electric_mw,
            p_tf_electric_supplies_mw,
            p_tritium_plant_electric_mw,
            vachtmw,
            p_pf_electric_supplies_mw,
            p_hcd_electric_loss_mw,
            p_coolant_pump_loss_total_mw,
            p_div_secondary_heat_mw,
            p_shld_secondary_heat_mw,
            p_hcd_secondary_heat_mw,
            p_tf_nuclear_heat_mw,
            p_plant_primary_heat_mw,
            eta_turbine,
            p_hcd_electric_total_mw,
            p_coolant_pump_elec_total_mw,
            p_fusion_total_mw,
            t_plant_pulse_coil_precharge,
            t_plant_pulse_plasma_current_ramp_up,
            t_plant_pulse_fusion_ramp,
            t_plant_pulse_burn,
            t_plant_pulse_plasma_current_ramp_down,
            t_plant_pulse_dwell,
        )


class PlantElectricProductionResistiveCentrepostLiquidBreeder(
    PlantElectricProductionReactor
):
    """Resistive centrepost and liquid breeder -- both extra reads at once."""

    def __call__(
        self,
        p_cp_coolant_pump_elec=From(tfcoil),
        p_blkt_liquid_breeder_heat_deposited_mw=From(power),
        etath_liq=From(heat_transport),
        p_plant_electric_base=From(heat_transport),
        a_plant_floor_effective=From(buildings),
        pflux_plant_floor_electric=From(heat_transport),
        p_cryo_plant_electric_mw=From(heat_transport),
        p_tf_electric_supplies_mw=From(heat_transport),
        p_tritium_plant_electric_mw=From(heat_transport),
        vachtmw=From(heat_transport),
        p_pf_electric_supplies_mw=From(pf_coil),
        p_hcd_electric_loss_mw=From(heat_transport),
        p_coolant_pump_loss_total_mw=From(heat_transport),
        p_div_secondary_heat_mw=From(heat_transport),
        p_shld_secondary_heat_mw=From(heat_transport),
        p_hcd_secondary_heat_mw=From(heat_transport),
        p_tf_nuclear_heat_mw=From(fwbs),
        p_plant_primary_heat_mw=From(heat_transport),
        eta_turbine=From(heat_transport),
        p_hcd_electric_total_mw=From(heat_transport),
        p_coolant_pump_elec_total_mw=From(heat_transport),
        p_fusion_total_mw=From(physics),
        t_plant_pulse_coil_precharge=From(times),
        t_plant_pulse_plasma_current_ramp_up=From(times),
        t_plant_pulse_fusion_ramp=From(times),
        t_plant_pulse_burn=From(times),
        t_plant_pulse_plasma_current_ramp_down=From(times),
        t_plant_pulse_dwell=From(times),
    ):
        return calculate_plant_electric_production_resistive_centrepost_liquid_breeder(
            p_cp_coolant_pump_elec,
            p_blkt_liquid_breeder_heat_deposited_mw,
            etath_liq,
            p_plant_electric_base,
            a_plant_floor_effective,
            pflux_plant_floor_electric,
            p_cryo_plant_electric_mw,
            p_tf_electric_supplies_mw,
            p_tritium_plant_electric_mw,
            vachtmw,
            p_pf_electric_supplies_mw,
            p_hcd_electric_loss_mw,
            p_coolant_pump_loss_total_mw,
            p_div_secondary_heat_mw,
            p_shld_secondary_heat_mw,
            p_hcd_secondary_heat_mw,
            p_tf_nuclear_heat_mw,
            p_plant_primary_heat_mw,
            eta_turbine,
            p_hcd_electric_total_mw,
            p_coolant_pump_elec_total_mw,
            p_fusion_total_mw,
            t_plant_pulse_coil_precharge,
            t_plant_pulse_plasma_current_ramp_up,
            t_plant_pulse_fusion_ramp,
            t_plant_pulse_burn,
            t_plant_pulse_plasma_current_ramp_down,
            t_plant_pulse_dwell,
        )
