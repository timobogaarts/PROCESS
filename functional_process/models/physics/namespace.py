"""Plasma physics' namespaces -- the two profile-shape arms, the `profiles` and
`confinement_time` sub-namespaces, and `Physics` itself.

Beside the nodes they name (`model_tree_design.md` §11).
`ProfileParameterisationParabolic`/`Pedestal` are *occupants of a switched slot*;
which one a machine gets is `indat.py`'s answer, and nothing here reads a switch.
"""

import dataclasses

from cottax.interfaces.pytree_namespace_module import ModelNamespace

from functional_process.models.physics.composition import (
    CalculateEffectiveChargeIonisationProfiles,
    PlasmaComposition,
)
from functional_process.models.physics.confinement_time import (
    ConfinementTime,
    DoubleAndTripleProduct,
)
from functional_process.models.physics.dimensionless_parameters import (
    DimensionlessPlasmaParameters,
)
from functional_process.models.physics.exhaust import RadiationFraction
from functional_process.models.physics.fusion_reactions import (
    FusionRates,
    SetFusionPowers,
)
from functional_process.models.physics.plasma_profiles import (
    IonVolAvgTemperature,
    LModeProfileReset,
    ParabolicGradientLengths,
    ParabolicProfileValues,
    ProfileFactors,
)
from functional_process.models.physics.profiles import (
    DensityProfile,
    NeProfileIntegral,
    ParabolicOnAxisDensities,
    ParabolicOnAxisTemperatures,
    ParabolicTemperatureProfile,
    PedestalOnAxisDensities,
    PedestalOnAxisTemperatures,
    PedestalTemperatureProfile,
    ProfileGrid,
    TeProfileIntegral,
)
from functional_process.models.physics.pure_formulas import (
    AuxiliaryPhysicsQuantities,
    ElectronThermalEnergy,
    FastAlphaBeta,
    IonElectronEquilibration,
    IonThermalEnergy,
    TotalPlasmaHeatingPower,
)
from functional_process.models.physics.radiation_power import (
    ImpurityRadiationTotals,
    PlasmaRadiationPowers,
    SynchrotronRadiationPower,
)
from functional_process.models.stellarator.density_limits import (
    EcrhDensityLimit,
)
from functional_process.models.stellarator.plasma_physics import (
    FusionPowerTotalsMw,
    FusionTotalsNoBeam,
)
from functional_process.models.switch_enums import (
    FastAlphaPressureModel,
)
from process.data_structure.physics_variables import (
    PlasmaIgnitionModel,
)
from process.models.physics.profiles import PlasmaProfileShapeType


class ProfileParameterisationParabolic(ModelNamespace):
    """Parabolic profiles: the `.physics.i_plasma_pedestal == 0` occupant, 7 nodes."""

    # Not just a formula switch: `density_limits.py:146-150` shows
    # PROCESS itself has no formula for `st_d_limit_ecrh` outside
    # `i_plasma_pedestal == 0` -- the `else` arm only logs an error and
    # produces no defined `dlimit_ecrh`/`bt_max_ecrh`. So the value-0
    # requirement `EcrhDensityLimit` already enforces internally
    # (`density_limits.py`'s `calculate_ecrh_density_limit` raises
    # otherwise) is not a stricter precondition than this arm's own --
    # co-locating the static kwarg next to the switch value that selects
    # it is what next_steps.md's "one source of truth" proposal reduces
    # to for this single instance, with no extra machinery needed: there
    # is only one place `i_plasma_pedestal=0` is written now.
    ecrh_density_limit: EcrhDensityLimit = EcrhDensityLimit(
        i_plasma_pedestal=PlasmaProfileShapeType.PARABOLIC_PROFILE
    )
    parabolic_temperature_profile: ParabolicTemperatureProfile = (
        ParabolicTemperatureProfile()
    )
    parabolic_on_axis_densities: ParabolicOnAxisDensities = ParabolicOnAxisDensities()
    parabolic_on_axis_temperatures: ParabolicOnAxisTemperatures = (
        ParabolicOnAxisTemperatures()
    )
    parabolic_gradient_lengths: ParabolicGradientLengths = ParabolicGradientLengths()
    # `ParabolicProfileValues` -- the line-average/density-weighted tail
    # of `parabolic_parameterisation`. `plasma_profiles.md`'s "cottax
    # node" section deferred it pending that record's open question 1
    # ("`i_plasma_pedestal` holds two different switch roles"); the
    # comment above on `EcrhDensityLimit(i_plasma_pedestal=0)` is that
    # question's answer for this switch's only instance, so the blocker
    # is gone. Without it `.physics.temp_plasma_electron_density_
    # weighted_kev`/`temp_plasma_ion_density_weighted_kev` were boundary
    # inputs and iteration variable 4 had no path into fusion
    # reactivity or beta -- see that class's docstring.
    parabolic_profile_values: ParabolicProfileValues = ParabolicProfileValues()
    # The seven-field L-mode reset (`plasma_profiles.py:92-117`), the
    # producer these fields never had. Belongs on this arm and only this
    # arm: PROCESS applies it in `parabolic_parameterisation`, i.e.
    # exactly when `i_plasma_pedestal == 0`. Without it a cold run
    # carries the input file's `nd_plasma_pedestal_electron`/
    # `nd_plasma_separatrix_electron` into `DensityProfile`, whose one
    # formula is the pedestal one and only degenerates to the parabolic
    # profile once they are zero -- so a parabolic run silently got a
    # pedestal density profile. See `LModeProfileReset`'s docstring for
    # the measured effect on the SAND cold solve.
    l_mode_profile_reset: LModeProfileReset = LModeProfileReset()


class ProfileParameterisationPedestal(ModelNamespace):
    """Pedestal profiles: the `.physics.i_plasma_pedestal == 1` occupant, 3 nodes."""

    pedestal_temperature_profile: PedestalTemperatureProfile = (
        PedestalTemperatureProfile()
    )
    pedestal_on_axis_densities: PedestalOnAxisDensities = PedestalOnAxisDensities()
    pedestal_on_axis_temperatures: PedestalOnAxisTemperatures = (
        PedestalOnAxisTemperatures()
    )
    # No pedestal-arm counterpart to `EcrhDensityLimit`: PROCESS's own
    # default configuration (`i_plasma_pedestal == 1`) never computes a
    # real ECRH density limit at all -- see the note on the value-0 arm.
    # `dlimit_ecrh`/`bt_max_ecrh` are therefore genuinely unproduced in
    # this arm, not merely unported.


class PhysicsProfiles(ModelNamespace):
    """Plasma profile shapes and the volume averages taken over them.

    A third level for the same reason `StellaratorCoils` is one: the density/temperature
    profile nodes and the fusion rates that read them form an SCC.
    """

    parameterisation: (
        ProfileParameterisationParabolic | ProfileParameterisationPedestal
    ) = dataclasses.field(kw_only=True)
    """How the plasma profiles are shaped (`.physics.i_plasma_pedestal`, default 1).

    Not a formula switch: the parabolic occupant carries `EcrhDensityLimit` as well, a
    node the pedestal occupant has no counterpart for at all -- under
    `i_plasma_pedestal == 1` PROCESS never computes a real ECRH density limit, so
    `.stellarator.dlimit_ecrh`/`bt_max_ecrh` are genuinely unproduced there. Ragged arms
    again, and the reason this could not be a static kwarg on one node.
    """

    # unit #12, physics/plasma_profiles.py
    profile_factors: ProfileFactors = ProfileFactors()
    # unit #21, physics/profiles.py -- arms not gated by `i_plasma_pedestal` only
    profile_grid: ProfileGrid = ProfileGrid(
        n_plasma_profile_elements=201
    )  # `physics_variables.py:1054` default
    ne_profile_integral: NeProfileIntegral = NeProfileIntegral()
    te_profile_integral: TeProfileIntegral = TeProfileIntegral()
    density_profile: DensityProfile = DensityProfile()
    # `plasma_profiles.py`. Unswitched, not under `.physics.i_plasma_pedestal`:
    # PROCESS writes `.physics.temp_plasma_ion_vol_avg_kev` in `parameterise_plasma`
    # *before* the branch, so it runs in both arms. A `FixedPointFunction` rather than an
    # `ExplicitFunction` because the field is conditionally owned by *data*
    # (`f_temp_plasma_ion_electron > 0`) -- see the class's own docstring for why that is
    # the honest shape and not a workaround.
    ion_vol_avg_temperature: IonVolAvgTemperature = IonVolAvgTemperature()


class PhysicsConfinementTime(ModelNamespace):
    """Energy confinement, and what is derived from it."""

    model: ConfinementTime = dataclasses.field(kw_only=True)
    """Which confinement-time scaling runs (`.stellarator.istell == 6`, Helias).

    **The one slot whose family already existed**: `StellaratorConfinementTime`
    subclasses
    `ConfinementTime`, so the annotation is the base and the occupant satisfies it --
    `model_tree_design.md` §4 case 1, the shared-body case. Every other switched slot in
    this tree is annotated with a union of its occupants, because no base was ever
    written for them; see this module's own note above `machine_from_indat`. The bare
    `ConfinementTime` was this slot's `istell == 0` occupant until that arm was deleted;
    a one-member family is what remains, not a spurious one.

    **The three settings live in the factory, and this slot has no default, because of
    them.** They are the reference run's, not `physics_variables.py`'s bare defaults, and
    each was a registration bug once: `i_confinement_time` was `34` (IPB98(y,2), a
    tokamak H-mode law) against ISS04's `38`, and `i_plasma_ignited` was `0` against the
    input file's `1`, which alone was the residual ~1.2 % `t_energy_confinement`
    disagreement. Both found by the MDA harness, never by a check -- and a slot default
    is exactly where a check could not see them, since the defaults test compared
    occupant *classes* only.
    """

    # unit #10, physics/confinement_time.py. `IterPhysicsBasisElongation` (the standalone
    # wrap of `calculate_iter_physics_basis_elongation`) is deliberately NOT registered:
    # `ConfinementTime` already computes and owns `.physics.kappa_ipb` itself (it calls
    # the same underlying function internally, then returns the value as its own 8th
    # output) -- registering both would be a duplicate-ownership conflict on one VarPath,
    # not two independent nodes. `i_confinement_time`/`i_rad_loss`/`i_plasma_ignited` are
    # kept as static kwargs (`ConfinementTime.__call__`'s dispatch needs concrete Python
    # ints, not traced values -- see confinement_time.py's own docstring, fixed during
    # this consolidation pass, for why all three needed `eqx.field(static=True)`).
    # `i_confinement_time=38` (ISS04, stellarator) -- **not** `physics_variables.py`'s own
    # bare default (`34`, IPB98(y,2), a tokamak H-mode scaling law): that default is
    # PROCESS's own tokamak-first choice, not one this project's stellarator scope should
    # inherit uncritically. Found and corrected via the block-by-block MDA-vs-PROCESS
    # comparison harness (`mda_harness.py`): registering `34` fed a tokamak confinement
    # formula stellarator inputs, producing a degenerate `t_energy_confinement` that
    # cascaded into `DoubleAndTripleProduct.ntau == 0.0` and several `inf` values
    # downstream (reciprocals of ~0). The ISS04 branch was already fully ported
    # (`confinement_time.py:1743`) -- this was a wrong static default at registration,
    # not a missing-physics gap. `i_confinement_time` genuinely has ~40 possible values
    # (tokamak and stellarator scaling laws both); a real `Switch`/`Alternative` covering
    # more of them is a separate, larger follow-up, not done here -- `38` is the
    # pragmatic, scope-appropriate single default for now, same discipline as
    # `i_rad_loss=1` alongside it.
    #
    # `i_plasma_ignited=1` (IGNITED) -- **not** `physics_variables.py:881`'s bare
    # default `0`, which is what this registration originally carried. Found by
    # `mda_harness.py`'s `switch_audit`, the systemic check added for exactly this
    # defect class; `stellarator_helias.IN.DAT:126` sets `i_plasma_ignited = 1`.
    # This was the sole cause of the residual ~1.2% `t_energy_confinement`/`ntau`
    # disagreement `next_steps.md` §8 previously listed as open and undiagnosed:
    # `confinement_time.py:1333-1334` adds `p_hcd_injected_total_mw` into
    # `p_plasma_loss_mw` only under NON_IGNITED, so registering `0` inflated the loss
    # power PROCESS's real ignited run never adds, and `t_energy_confinement` (and
    # everything scaled by it) came out correspondingly off.
    # Checked before flipping, same discipline as `i_thermal_electric_conversion`:
    # the IGNITED arm simply *omits* that one addition -- it reads a strict subset of
    # the NON_IGNITED arm's inputs, so it needs nothing this port does not wire.
    # **Not registered here**: `ConfinementTime`/`StellaratorConfinementTime` are
    # arms of `TOPOLOGY_SWITCHES`'s `.stellarator.istell` switch -- see that switch
    # for why the device mode decides which node produces this block's 20th read.
    double_and_triple_product: DoubleAndTripleProduct = DoubleAndTripleProduct()


class Physics(ModelNamespace):
    """Plasma physics: composition, profiles, fusion rates, beta, exhaust."""

    # **Filed here on physical grounds, against PROCESS's own filing.** Both live in
    # `stellarator.py`'s `st_phys` and every earlier grouping put them under
    # `stellarator` because of it -- but they own **only** `.physics.*` fields, exactly
    # as `FusionRates`/`SetFusionPowers` beside them do, and nothing about them is
    # stellarator-specific. Leaving them under `stellarator` was measured to be the
    # sole reason the density/fusion/pedestal cycle crossed a subsystem boundary, which
    # `switch_elimination_design.md` §11.1 had recorded as never happening. §11.3 says
    # the stronger claim is a grouping "declared on physical grounds" rather than
    # mirrored from PROCESS's file layout, and this is the first place the two
    # disagree -- so the declaration wins, and §11.1's containment result survives.
    fusion_power_totals_mw: FusionPowerTotalsMw = FusionPowerTotalsMw()
    # The `else` arm of `stellarator.py:2002-2054`, three identities -- and the only
    # producer of `.physics.fusden_total`/`.fusden_alpha_total`/`.p_dt_total_mw`,
    # which were boundary inputs until it landed. Unconditional because the arm is
    # selected by `i_plasma_ignited == IGNITED` on this run, not merely by the absence
    # of a beam, and because the beam arm calls the unportable `reactions.beam_fusion`
    # (unit #19) -- there is no second arm to switch between. See
    # `_audit/boundary_inputs_audit.md` §4c (b7)/(b8) and the class's own docstring.
    fusion_totals_no_beam: FusionTotalsNoBeam = FusionTotalsNoBeam()
    profiles: PhysicsProfiles = dataclasses.field(kw_only=True)

    confinement_time: PhysicsConfinementTime = dataclasses.field(kw_only=True)

    # unit #19, physics/fusion_reactions.py
    fusion_rates: FusionRates = FusionRates()
    set_fusion_powers: SetFusionPowers = SetFusionPowers()
    # unit #20, physics/radiation_power.py
    synchrotron_radiation_power: SynchrotronRadiationPower = SynchrotronRadiationPower()
    # `imp_indices` is a graph-assembly-time fact (which impurity species this machine
    # has), not a per-evaluation switch -- see `ImpurityRadiationTotals`'s docstring.
    # All 14 species: H/He are always recomputed by `plasma_composition()`, and species
    # 2-13 are held non-zero by iteration variables 125-136's lower bound (1e-8, 22
    # orders above the 1e-30 selection threshold) in the reference configuration this
    # scope targets. A run without those iteration variables active could legitimately
    # need a narrower tuple; nothing here checks that yet (radiation_power.md § open
    # questions 2).
    impurity_radiation_totals: ImpurityRadiationTotals = ImpurityRadiationTotals(
        imp_indices=tuple(range(14))
    )
    plasma_radiation_powers: PlasmaRadiationPowers = PlasmaRadiationPowers()
    # unit #9 chunk A, physics/pure_formulas.py -- five already-pure formulas
    # lifted verbatim, no entanglement, no switch-driven topology split.
    ion_electron_equilibration: IonElectronEquilibration = IonElectronEquilibration()
    auxiliary_physics_quantities: AuxiliaryPhysicsQuantities = (
        AuxiliaryPhysicsQuantities()
    )
    total_plasma_heating_power: TotalPlasmaHeatingPower = TotalPlasmaHeatingPower()
    electron_thermal_energy: ElectronThermalEnergy = ElectronThermalEnergy()
    ion_thermal_energy: IonThermalEnergy = IonThermalEnergy()
    # `i_beta_fast_alpha` kept as a static kwarg, not a Switch -- both branches read the
    # same six variables (pure_formulas.md's "switches touched"), same
    # shape as `EcrhDensityLimit`'s `i_plasma_pedestal`. Default `1` (WARD),
    # `physics_variables.py:875`.
    fast_alpha_beta: FastAlphaBeta = FastAlphaBeta(
        i_beta_fast_alpha=FastAlphaPressureModel.WARD
    )
    # unit #9 chunk B, physics/composition.py. `plasma_composition`'s
    # `.physics.first_call` turned out not to be a genuine cycle at all -- its real
    # referent (`f_temp_plasma_electron_density_vol_avg`, from `plasma_profiles.py`) has
    # no dependency back on this node, so `first_call` is an ordering artifact of
    # PROCESS's imperative call sequence (`next_steps.md` §5), not ported. An earlier
    # draft represented it as a `NextFirstCall`/`FixedPointFunction` self-loop; removed.
    # The second Shape-B self-loop this chunk's own audit found this session
    # (`.impurity_radiation.f_nd_impurity_electron_array`, read at indices 2-13, written
    # at 0/1) is resolved without any `Cut`/`FixedPoint` machinery at all -- per-index
    # `VarPath`s (`s.impurity_radiation.f_nd_impurity_electron_array[i]`) make the read
    # and write ranges genuinely disjoint `VarPath`s, not one whole-array self-reference.
    # `i_plasma_ignited=IGNITED` -- **not** `physics_variables.py:881`'s bare
    # default (`i_plasma_ignited = 0`, NON_IGNITED). `stellarator_helias.IN.DAT:126`
    # sets `i_plasma_ignited = 1` (IGNITED), and the converged run confirms it: the
    # `switch_audit` check `mda_harness.py` now runs over every registered static kwarg
    # reported `registered=False but .physics.i_plasma_ignited == True`. Same defect
    # class as `i_confinement_time`/`i_thermal_electric_conversion` below -- a bare
    # `*_variables.py` default copied uncritically into a registration.
    # This node used to spell the same fact as a `bool` named `is_ignited`, which
    # `switch_elimination_design.md` §3 classifies as kind (d) alias/noise and which
    # cost `mda_harness.STATIC_KWARG_ALIASES` a hand-written `bool(v == 1)` entry
    # just so `switch_audit` could resolve it. It is now spelled and typed as
    # PROCESS spells it, `PlasmaIgnitionModel` under PROCESS's own field name, and
    # resolves by name like every other static kwarg.
    # Checked before flipping, same discipline as `i_thermal_electric_conversion`
    # below: the IGNITED arm needs no input this port does not already wire --
    # `composition.py:219-222` is `nd_beam_ions = 0` under IGNITED
    # versus `nd_plasma_electrons_vol_avg * f_nd_beam_electron` otherwise, so the
    # ignited arm reads a strict *subset* of the non-ignited arm's inputs.
    plasma_composition: PlasmaComposition = PlasmaComposition(
        i_plasma_ignited=PlasmaIgnitionModel.IGNITED
    )
    # Over the line length and left that way: slot name and occupant class are both
    # this long, and the annotation cannot be wrapped -- `ruff format` strips
    # parentheses from around an annotation. An import alias would hide the class name.
    calculate_effective_charge_ionisation_profiles: CalculateEffectiveChargeIonisationProfiles = CalculateEffectiveChargeIonisationProfiles()  # noqa: E501
    # unit #9 chunk C, physics/dimensionless_parameters.py -- the one real
    # computation inside the 1095-line `outplas` reporting method.
    dimensionless_plasma_parameters: DimensionlessPlasmaParameters = (
        DimensionlessPlasmaParameters()
    )
    # unit #11, physics/exhaust.py
    radiation_fraction: RadiationFraction = RadiationFraction()
