"""The power subsystem's namespace -- thermal and electric flows, and cryogenics.

Beside the nodes it names (`model_tree_design.md` §11).
"""

import dataclasses

from cottax.interfaces.pytree_namespace_module import ModelNamespace

from functional_process.models.power.electric_production import (
    Acpow,
)
from functional_process.models.power.pf_coil_power import PfCoilPowerSupplies
from functional_process.models.power.tf_coil_power import (
    TfPowerResistive,
    TfPowerSuperconducting,
)
from functional_process.models.power.thermal_cryo import (
    ComponentThermalPowers,
    CryoLoads,
    CryoQLoads,
    CryoQNuc,
    DeltaEtaStep,
    EtathLiq,
    EtaTurbine,
    PFwBlktCoolantPumpMw,
    PFwDivHeatDepositedMw,
    TempTurbineCoolantIn,
)


class Power(ModelNamespace):
    """Thermal and electric power flows, cryogenics, and the plant's own consumption."""

    pf_coil_power: PfCoilPowerSupplies | None = dataclasses.field(kw_only=True)
    """`Power.pfpwr` -- **the one subsystem of `power.py` a tokamak has and a
    stellarator does not**, so the one slot in this namespace that can be honestly
    absent.

    `total_process.TokamakProcess.power` measured the asymmetry before there was
    anything to put here: "Shared, 11 functions / 1522 lines; tokamak-new, `Power.pfpwr`
    and its four `_pf_loss_*` helpers -- the PF-coil power supply, which a stellarator
    has no PF coils to need." `stellarator.py:114-186` never calls `Power.run`, so it
    never reaches `pfpwr`; `indat` fills this slot on a tokamak and leaves it `None` on a
    stellarator, which cottax reads as absence rather than as a zero.

    Not switched -- `pfpwr` has no dispatch of its own, and every topology switch inside
    it is already part of `indat._pf_coil_system_arm`'s joint predicate. Added
    2026-08-30 to close four missing producers; see `pf_coil_power.py`."""

    tf_power: TfPowerResistive | TfPowerSuperconducting = dataclasses.field(kw_only=True)
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
    #
    # **`i_p_coolant_pumping` is threaded by the factory now, not written here.** It was
    # the same hardcoded `FRACTION_OF_HEAT` on all four nodes below, correct for the
    # Helias run (`stellarator_helias.IN.DAT:198` sets `1`) and wrong for the first
    # tokamak, which sets `3` (`large_tokamak_eval.IN.DAT:172`). Nothing caught it,
    # because `switch_audit` compares a registration against *the reference run's*
    # converged state and the reference run was a stellarator. The tokamak caught it by
    # refusing to assemble -- see `p_fw_blkt_coolant_pump_mw_step` below.
    component_thermal_powers: ComponentThermalPowers = dataclasses.field(kw_only=True)
    delta_eta_step: DeltaEtaStep = dataclasses.field(kw_only=True)
    eta_turbine: EtaTurbine | None = dataclasses.field(kw_only=True)
    """`.heat_transport.eta_turbine` -- **or nothing, which is what the reference run
    gets.**

    This was `eta_turbine_step`, a `FixedPointFunction` carrying
    `i_thermal_electric_conversion` and `i_blanket_type`. Four of that dispatch's arms
    are `return eta_turbine` -- the field is a user input -- so the self-loop existed
    only where there was nothing to solve. `switch_kwarg_survey.md` §4.7 measured it as
    the identity map on this machine and flagged that `.costs.coe`, this run's
    objective, depends on it: *"a quantity the objective depends on is nominally driven
    while being, on this configuration, an unowned boundary input in disguise."* It is
    an unowned boundary input, said outright (`_audit/next_steps.md` §14.2)."""
    etath_liq: EtathLiq | None = dataclasses.field(kw_only=True)
    """`.heat_transport.etath_liq` -- the same shape as `eta_turbine`, on
    `.fwbs.secondary_cycle_liq`. `== 2` is "the efficiency is an input" and is spelled
    `None`; `== 4` computes it from `.fwbs.outlet_temp_liq` alone."""
    temp_turbine_coolant_in: TempTurbineCoolantIn | None = dataclasses.field(
        kw_only=True
    )
    """`.heat_transport.temp_turbine_coolant_in` -- three arms of
    `i_thermal_electric_conversion` x `i_blanket_type` x `secondary_cycle_liq`, one of
    them absent. Two stages write this field in order and the self-loop existed only
    where both passed the entering value through; see `TempTurbineCoolantIn`."""
    p_fw_div_heat_deposited_mw: PFwDivHeatDepositedMw | None = dataclasses.field(
        kw_only=True
    )
    """`.heat_transport.p_fw_div_heat_deposited_mw` -- owned on every
    `i_p_coolant_pumping` value except `MECHANICAL_WITH_PRESSURE_DROP`, where PROCESS
    passes the entering value through and the field's only other producer is
    `models/ife.py`, out of scope. Absent there, and no longer a `FixedPointFunction`
    anywhere."""
    p_fw_blkt_coolant_pump_mw: PFwBlktCoolantPumpMw | None = dataclasses.field(
        kw_only=True
    )
    """`.primary_pumping.p_fw_blkt_coolant_pump_mw` -- **owned here on two of
    `i_p_coolant_pumping`'s four values and by the blanket on the other two.**

    This node's own docstring already called the field *"a conditional-ownership
    pass-through"* and named the other producer -- `process/models/blankets/hcpb.py`,
    *"not yet ported, registry unit #13"*. It is ported now, and the two producers met:
    assembling a tokamak with `i_p_coolant_pumping = 3` raised cottax's duplicate-owner
    error outright, naming this node and `.tokamak.ccfe_hcpb.pumping_power`.

    The resolution is the one the tree already has a spelling for: on `MECHANICAL` and
    `MECHANICAL_WITH_PRESSURE_DROP` **PROCESS does not compute this field here at all**
    (`power.py:815-820` only overwrites it on the other two values), so the slot is
    `None` and the blanket owns it. Absence, spelled as absence -- the same answer
    `costs.cost_of_electricity`, `power.cryo_q_nuc` and
    `cicc_superconducting_tf_coil.dx_tf_side_case_min` give.

    `hcpb.md` open question 1 asked whether the fix was *"a change to `power.py`'s
    existing nodes or four new occupants"*. It is neither: it is a registration, because
    the node that already exists is correct on exactly the arms it is now given. What is
    **not** resolved here is that open question's other half -- at
    `i_p_coolant_pumping == 3` nothing produces
    `.heat_transport.p_fw_coolant_pump_mw`/`p_blkt_coolant_pump_mw` either, and this
    node still reads both. They surface as boundary inputs on a tokamak. That is
    `next_steps.md` §14.9 item 2's call and is left to it.
    """
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
    cryo_q_nuc: CryoQNuc | None = dataclasses.field(kw_only=True)
    """`.fwbs.qnuc` -- **and the slot is empty when PROCESS takes it as an input.**

    `inuclear` was a static kwarg on a `FixedPointFunction` here. Its two arms read
    disjoint variables (the computed arm reads `p_tf_nuclear_heat_mw` and not the
    incumbent; the input arm reads only the incumbent), which is
    `traceability_policy.md`'s split-by-default case, and splitting collapsed the fixed
    point entirely -- see `CryoQNuc`. `None` is the `inuclear == 1` arm: no node owns
    `.fwbs.qnuc`, so it is an ordinary boundary input, which is exactly what PROCESS
    means by *"if inuclear = 1: qnuc is input"*. Stated by the tree instead of recovered
    at runtime by differentiating a degenerate residual.

    `i_tf_sup` gates it jointly, and is threaded from `machine_from_indat` rather than
    written twice: it already decides the `tf_power` slot above, and writing it again
    here let the two disagree -- an `i_tf_sup = 0` machine once assembled
    `TfPowerResistive` next to five nodes still saying `SUPERCONDUCTING`.
    """

    cryo_q_loads: CryoQLoads | None = dataclasses.field(kw_only=True)
    """`.power.qss`/`qac`/`qcl`/`qmisc` -- **or nothing**, when PROCESS never calls
    `Power.cryo`.

    This was `cryo_q_loads_step`, a `FixedPointFunction` carrying `i_tf_sup` and
    `i_pf_conductor`. The self-read existed only outside the guard, where the body is a
    four-way pass-through -- which is "these are inputs", i.e. an empty slot
    (`_audit/next_steps.md` §14.2). Inside the guard the two occupants differ by three
    reads, all TF-coil fields the resistive arm does not touch."""

    cryo_loads: CryoLoads = dataclasses.field(kw_only=True)
    """The unconditionally-owned cryogenic loads -- one occupant per arm of the same
    two switches. The aluminium arm has no occupant (`('i_tf_sup', 2)` is `UNPORTED`),
    which is what lets both live occupants drop `.tfcoil.p_cp_resistive`,
    `.tfcoil.p_tf_leg_resistive`, `.tfcoil.p_tf_joints_resistive` and
    `.fwbs.pnuc_cp_tf`."""
    # `electric_production.py` (unit #14 chunk C). `i_pf_energy_storage_source=2`
    # matches `pf_power_variables.py:18`'s default.
    acpow: Acpow = dataclasses.field(kw_only=True)
    """Plant AC power requirement -- one occupant per
    `.pf_power.i_pf_energy_storage_source` value.

    **The switch was a static kwarg here and is a slot now** (`_audit/next_steps.md`
    §14.2). The two arms read complementary fields -- `.heat_transport.peakmva` on the
    line arm, `.heat_transport.fmgdmw` on the flywheel arm -- so one node declared
    exactly one edge no run makes."""
