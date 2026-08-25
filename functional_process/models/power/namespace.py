"""The power subsystem's namespace -- thermal and electric flows, and cryogenics.

Beside the nodes it names (`model_tree_design.md` §11).
"""

import dataclasses

from cottax.interfaces.pytree_namespace_module import ModelNamespace

from functional_process.models.power.electric_production import (
    Acpow,
)
from functional_process.models.power.tf_coil_power import (
    TfPowerResistive,
    TfPowerSuperconducting,
)
from functional_process.models.power.thermal_cryo import (
    ComponentThermalPowers,
    CryoLoads,
    CryoQLoadsStep,
    CryoQNucStep,
    DeltaEtaStep,
    EtathLiqStep,
    EtaTurbineStep,
    PFwBlktCoolantPumpMwStep,
    PFwDivHeatDepositedMwStep,
    TempTurbineCoolantInStep,
)
from functional_process.models.switch_enums import (
    BlanketDualCoolantModel,
    PFEnergyStorageSource,
)
from process.data_structure.blanket_variables import BlktModelTypes
from process.models.power import ElectricConversionModelTypes, PumpingPowerModelTypes


class Power(ModelNamespace):
    """Thermal and electric power flows, cryogenics, and the plant's own consumption."""

    tf_power: TfPowerResistive | TfPowerSuperconducting = dataclasses.field(
        kw_only=True
    )
    """TF-coil power supplies (`.tfcoil.i_tf_sup`, default 1 = superconducting).

    The aluminium arm (`i_tf_sup == 2`) is refused rather than aliased onto the resistive
    occupant: it runs the identical branch in PROCESS, but saying so by registry absence
    keeps the claim visible instead of burying it in a shared occupant.
    """

    # `tf_coil_power.py` (unit #14 chunk A). `TfPowerResistive`/
    # `TfPowerSuperconducting` are registered under `TOPOLOGY_SWITCHES`'s new
    # `.tfcoil.i_tf_sup` switch instead of here -- see that switch's own comment.
    # `thermal_cryo.py` (unit #14 chunk B). Six of `calculate_
    # component_thermal_powers`'s outputs are genuine single-node self-loops (each
    # field's *entering* value is read, then a freshly-computed value is written back to
    # the same `VarPath` later in the same PROCESS call) -- already split this session
    # into their own `FixedPointFunction`s, same "Shape B" treatment as
    # `plasma_composition`'s `first_call`/`Avail`'s `cplife` above.
    # `i_blkt_dual_coolant=0`/`i_blanket_type=1`/`secondary_cycle_liq=4` match
    # `fwbs_variables.py`'s own defaults (lines 526, 70, 273) and agree with this run,
    # confirmed by `mda_harness.py`'s `switch_audit`.
    #
    # `i_p_coolant_pumping=1` (`PumpingPowerModelTypes.FRACTION_OF_HEAT`,
    # `power.py:23`) -- **not** `fwbs_variables.py:249`'s bare default `2`
    # (`MECHANICAL`), which all four registrations below used to carry.
    # `stellarator_helias.IN.DAT:198` sets `1`. Flagged but not fixed by the pass that
    # corrected `i_thermal_electric_conversion`; fixed here, and now checked
    # automatically rather than by luck (`switch_audit`). Checked before flipping,
    # same discipline: the two switch-dependent bodies are conditional-ownership
    # pass-throughs, and value `1` selects the *recompute* side of both, out of
    # arguments these nodes already take --
    # `calculate_p_fw_blkt_coolant_pump_mw` (`thermal_cryo.py:206-211`)
    # returns `p_fw_coolant_pump_mw + p_blkt_coolant_pump_mw` for `1 not in
    # {MECHANICAL, MECHANICAL_WITH_PRESSURE_DROP}`, and
    # `calculate_p_fw_div_heat_deposited_mw` (`thermal_cryo.py:308-310`)
    # returns `p_fw_heat_deposited_mw + p_div_heat_deposited_mw` for
    # `1 != MECHANICAL_WITH_PRESSURE_DROP`. Both operands are already `FromExactly`s (or
    # rebuilt from `FromExactly`s) on every node below, so no arm has a hole in it.
    #
    # `i_thermal_electric_conversion=2` (`ElectricConversionModelTypes.USER_INPUT`) --
    # **not** `0` (`CCFE_HCPB_VALUE`, `fwbs_variables.py:264`'s bare default). Found and
    # corrected via the block-by-block MDA-vs-PROCESS comparison harness
    # (`mda_harness.py`): `stellarator_helias.IN.DAT` sets this explicitly (line 203),
    # and the wrong hardcoded `0` fed a completely different branch of
    # `calculate_plant_thermal_efficiency`/`calculate_component_thermal_powers`/
    # `calculate_delta_eta` than PROCESS's own real run took -- confirmed as the exact,
    # sole cause of `ComponentThermalPowers`/`EtaTurbineStep`/`DeltaEtaStep`'s
    # disagreements (bit-for-bit match once corrected, `PicardDriver` itself was never
    # at fault). Every branch these four nodes' `step`/`__call__` bodies already read
    # this switch through -- `USER_INPUT` needs no input this port doesn't already wire
    # (it is a pure identity pass-through for `eta_turbine`, confirmed against
    # `power.py:1992-1994`), so this is a like-for-like default correction, not a new
    # port. Duplicated identically across these four registrations rather than one
    # shared source of truth -- same caution `i_confinement_time` (just above) already
    # flags for itself: a real `Switch`/`Alternative` covering
    # `ElectricConversionModelTypes`'s 5 values is a separate, larger follow-up, not
    # done here.
    component_thermal_powers: ComponentThermalPowers = ComponentThermalPowers(
        i_p_coolant_pumping=PumpingPowerModelTypes.FRACTION_OF_HEAT,
        i_blkt_dual_coolant=BlanketDualCoolantModel.SINGLE_COOLANT_SOLID_BREEDER,
        i_thermal_electric_conversion=ElectricConversionModelTypes.USER_INPUT,
        i_blanket_type=BlktModelTypes.CCFE_HCPB,
        secondary_cycle_liq=(
            ElectricConversionModelTypes.SUPERCRITICAL_CO2_BRAYTON_CYCLE
        ),
    )
    delta_eta_step: DeltaEtaStep = DeltaEtaStep(
        i_p_coolant_pumping=PumpingPowerModelTypes.FRACTION_OF_HEAT,
        i_blkt_dual_coolant=BlanketDualCoolantModel.SINGLE_COOLANT_SOLID_BREEDER,
        i_thermal_electric_conversion=ElectricConversionModelTypes.USER_INPUT,
    )
    eta_turbine_step: EtaTurbineStep = EtaTurbineStep(
        i_thermal_electric_conversion=ElectricConversionModelTypes.USER_INPUT,
        i_blanket_type=BlktModelTypes.CCFE_HCPB,
    )
    etath_liq_step: EtathLiqStep = EtathLiqStep(
        secondary_cycle_liq=(
            ElectricConversionModelTypes.SUPERCRITICAL_CO2_BRAYTON_CYCLE
        )
    )
    temp_turbine_coolant_in_step: TempTurbineCoolantInStep = TempTurbineCoolantInStep(
        i_thermal_electric_conversion=ElectricConversionModelTypes.USER_INPUT,
        i_blanket_type=BlktModelTypes.CCFE_HCPB,
        secondary_cycle_liq=(
            ElectricConversionModelTypes.SUPERCRITICAL_CO2_BRAYTON_CYCLE
        ),
    )
    p_fw_div_heat_deposited_mw_step: PFwDivHeatDepositedMwStep = (
        PFwDivHeatDepositedMwStep(
            i_p_coolant_pumping=PumpingPowerModelTypes.FRACTION_OF_HEAT
        )
    )
    p_fw_blkt_coolant_pump_mw_step: PFwBlktCoolantPumpMwStep = PFwBlktCoolantPumpMwStep(
        i_p_coolant_pumping=PumpingPowerModelTypes.FRACTION_OF_HEAT
    )
    # `PlantThermalEfficiency`/`PlantThermalEfficiency2` (the raw, un-split
    # `ExplicitFunction`s `EtaTurbineStep`/`EtathLiqStep`/`TempTurbineCoolantInStep` are
    # extracted from) are NOT registered: each is *itself* a genuine, still-unresolved
    # self-loop on its own -- `to_graph(PlantThermalEfficiency(...))` raises
    # `ValueError: reads [...], which it also owns` directly (confirmed this pass, not
    # merely asserted), since both own and read `eta_turbine`/`temp_turbine_coolant_in`
    # (`etath_liq`/`temp_turbine_coolant_in` for the second). They are superseded by the
    # three `*Step` `FixedPointFunction`s above for graph purposes, not usable
    # standalone.
    #
    # `Power.calculate_cryo_loads` (`_audit/boundary_inputs_audit.md` §7 item 7) is the
    # second wave of exactly that Shape-B gap, and it is now split the same way. Its
    # raw node `Cryo` stays NOT registered for the same reason
    # `PlantThermalEfficiency` does -- `to_graph(Cryo(...))` raises `ValueError: reads
    # ['.fwbs.qnuc'], which it also owns` -- and the three nodes below replace it:
    #   * `CryoQNucStep` owns `.fwbs.qnuc`, conditionally written by PROCESS under
    #     `inuclear == 0 and i_tf_sup == 1` ("Issue #511: if inuclear = 1: qnuc is
    #     input", `power.py:1825`);
    #   * `CryoQLoadsStep` owns `.power.qss`/`qac`/`qcl`/`qmisc`, conditionally written
    #     under the *other* guard, `i_tf_sup == 1 or i_pf_conductor == SUPERCONDUCTING`
    #     (`power.py:1054-1057`), which is why the five fields are two nodes and not
    #     one -- see `CryoQNucStep`'s docstring for the degeneracy argument;
    #   * `CryoLoads` owns the four fields written on every path
    #     (`.heat_transport.helpow`, `.p_cryo_plant_electric_mw`, `.helpow_cryal`,
    #     `.tfcoil.cryo_cool_req`) and reads the five `q*` as plain `FromExactly`s.
    # This closes `.heat_transport.helpow` (read by `Bldgs`, `CryogenicSystemCost`) and
    # `.heat_transport.p_cryo_plant_electric_mw` (read by `Acpow`,
    # `PlantElectricProductionReactor`, `AuxiliaryComponentCoolingCost`) as boundary
    # inputs. `inuclear=0`/`i_pf_conductor=0` are `fwbs_variables.py:81`/
    # `pfcoil_variables.py:230`'s defaults, neither set by `REFERENCE_INPUT_FILE`;
    # `i_tf_sup=1` is `tfcoil_variables.py:261`'s, likewise unset -- the same value the
    # rest of this file's TF-coil registrations already carry.
    cryo_q_nuc_step: CryoQNucStep = dataclasses.field(kw_only=True)
    """`.fwbs.qnuc`'s fixed point, carrying `.tfcoil.i_tf_sup` (default 1).

    **Threaded from `machine_from_indat`, not hardcoded.** `i_tf_sup` already decides
    the `tf_power` slot above; writing it again here let the two disagree, and they did
    -- an `i_tf_sup = 0` machine assembled `TfPowerResistive` next to five nodes still
    saying `SUPERCONDUCTING`. `inuclear` stays a kwarg: it decides nothing else.
    """

    cryo_q_loads_step: CryoQLoadsStep = dataclasses.field(kw_only=True)
    """`.power.qss`/`qac`/`qcl`/`qmisc`'s fixed point -- `i_tf_sup` threaded, for the
    reason `cryo_q_nuc_step` gives. `i_pf_conductor` stays a kwarg."""

    cryo_loads: CryoLoads = dataclasses.field(kw_only=True)
    """The unconditionally-owned cryogenic loads -- `i_tf_sup` threaded, same reason."""
    # `electric_production.py` (unit #14 chunk C). `i_pf_energy_storage_source=2`
    # matches `pf_power_variables.py:18`'s default.
    acpow: Acpow = Acpow(i_pf_energy_storage_source=PFEnergyStorageSource.LINE)
