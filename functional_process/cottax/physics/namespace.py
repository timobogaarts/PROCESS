"""Plasma physics' namespaces -- the two profile-shape arms, the `profiles` and
`confinement_time` sub-namespaces, and `Physics` itself.
"""

import dataclasses

from cottax.interfaces.pytree_namespace_module import ModelNamespace

from functional_process.cottax.physics.composition import (
    CalculateEffectiveChargeIonisationProfiles,
    PlasmaComposition,
)
from functional_process.cottax.physics.confinement_time import (
    ConfinementScalingInputs,
    ConfinementTail,
    ConfinementTimeScaling,
    DoubleAndTripleProduct,
    IterPhysicsBasisElongation,
    PlasmaPowerLoss,
)
from functional_process.cottax.physics.dimensionless_parameters import (
    DimensionlessPlasmaParameters,
)
from functional_process.cottax.physics.exhaust import RadiationFraction
from functional_process.cottax.physics.fusion_reactions import (
    FusionRates,
    SetFusionPowers,
)
from functional_process.cottax.physics.plasma_profiles import (
    IonVolAvgTemperature,
    LModeProfileReset,
    ParabolicGradientLengths,
    ParabolicProfileValues,
    PedestalProfileValues,
    ProfileFactors,
)
from functional_process.cottax.physics.profiles import (
    DensityProfile,
    GreenwaldDensityFractions,
    NeProfileIntegral,
    ParabolicOnAxisDensities,
    ParabolicOnAxisTemperatures,
    ParabolicTemperatureProfile,
    PedestalOnAxisDensities,
    PedestalOnAxisTemperatures,
    PedestalSeparatrixDensities,
    PedestalTemperatureProfile,
    ProfileGrid,
    TeProfileIntegral,
)
from functional_process.cottax.physics.pure_formulas import (
    AuxiliaryPhysicsQuantities,
    ElectronThermalEnergy,
    FastAlphaBeta,
    IonElectronEquilibration,
    IonThermalEnergy,
    TotalPlasmaHeatingPower,
)
from functional_process.cottax.physics.radiation_power import (
    ImpurityRadiationTotals,
    PlasmaRadiationPowers,
    SynchrotronRadiationPower,
)
from functional_process.cottax.stellarator.density_limits import (
    EcrhDensityLimit,
)
from functional_process.cottax.stellarator.plasma_physics import (
    FusionPowerTotalsMw,
    FusionTotalsNoBeam,
)


class ProfileParameterisationParabolic(ModelNamespace):
    """Parabolic profiles: the `.physics.i_plasma_pedestal == 0` occupant, 6 or 7 nodes.
    """

    ecrh_density_limit: EcrhDensityLimit | None = dataclasses.field(kw_only=True)
    """The ECRH density limit -- **present on a stellarator, absent on a tokamak.** Not
    just a formula switch: `density_limits.py:146-150` shows PROCESS itself has no
    formula for `st_d_limit_ecrh` outside `i_plasma_pedestal == 0` -- the `else` arm
    only logs an error and produces no defined `dlimit_ecrh`/`bt_max_ecrh`.
    """

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
    """Pedestal profiles: the `.physics.i_plasma_pedestal == 1` occupant, 5 nodes."""

    pedestal_separatrix: PedestalSeparatrixDensities | GreenwaldDensityFractions = (
        dataclasses.field(kw_only=True)
    )
    """`.physics.i_nd_plasma_pedestal_separatrix` -- **the two arms are inverses of one
    another, and that is why neither could be defaulted.** `GREENWALD_FRACTION` (`1`,
    PROCESS's default and `large_tokamak_eval`'s) reads the two Greenwald *fractions*
    and owns the two *densities*; `USER_INPUT` (`0`) reads the two densities and owns
    the two fractions.
    """

    pedestal_temperature_profile: PedestalTemperatureProfile = (
        PedestalTemperatureProfile()
    )
    pedestal_on_axis_densities: PedestalOnAxisDensities = PedestalOnAxisDensities()
    pedestal_on_axis_temperatures: PedestalOnAxisTemperatures = (
        PedestalOnAxisTemperatures()
    )
    # `PedestalProfileValues` -- the line-average/density-weighted tail
    # of `pedestal_parameterisation`, and the mirror of the parabolic
    # arm's `parabolic_profile_values` below. **Found by assembling a
    # tokamak, not by a test**: this is the first machine in the port to
    # select `i_plasma_pedestal == 1`
    # (`large_tokamak_eval.IN.DAT:291`), and until it existed the
    # pedestal arm produced four fewer variables than the parabolic one
    # -- `.physics.f_temp_plasma_electron_density_vol_avg`,
    # `nd_plasma_electron_line`,
    # `temp_plasma_electron_density_weighted_kev` and
    # `temp_plasma_ion_density_weighted_kev` all surfaced as boundary
    # inputs on a tokamak and on no stellarator. Every *value* test for
    # `calculate_pedestal_profile_values` already passed; what was
    # missing was the binding, which is exactly the class of defect
    # `_audit/tokamak_boundary.md` § "The four that are a shared
    # subsystem's gap" exists to catch. **Ragged arms are legitimate;
    # ragged arms nobody declared are not** -- a slot with an occupant
    # on both arms can still have one arm produce less than the other,
    # and only the boundary sees it.
    pedestal_profile_values: PedestalProfileValues = PedestalProfileValues()
    # No pedestal-arm counterpart to `EcrhDensityLimit`: PROCESS's own
    # default configuration (`i_plasma_pedestal == 1`) never computes a
    # real ECRH density limit at all -- see the note on the value-0 arm.
    # `dlimit_ecrh`/`bt_max_ecrh` are therefore genuinely unproduced in
    # this arm, not merely unported.


class PhysicsProfiles(ModelNamespace):
    """Plasma profile shapes and the volume averages taken over them."""

    parameterisation: (
        ProfileParameterisationParabolic | ProfileParameterisationPedestal
    ) = dataclasses.field(kw_only=True)
    """How the plasma profiles are shaped (`.physics.i_plasma_pedestal`, default 1)."""

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
    """Energy confinement: the head, the scaling law, and the tail, as separate slots.
    """

    inputs: ConfinementScalingInputs = ConfinementScalingInputs()
    """The unit conversions the laws take as arguments (`nd_plasma_electron_line_19`,
    `cur_plasma_ma`).
    """

    elongation: IterPhysicsBasisElongation = IterPhysicsBasisElongation()
    """`.physics.kappa_ipb`, and **now registered.** It was deliberately left out while
    `ConfinementTime` computed and owned `kappa_ipb` itself -- registering both would
    have been a duplicate-ownership conflict on one `VarPath`.
    """

    power_loss: PlasmaPowerLoss = dataclasses.field(kw_only=True)
    """The head: `.physics.p_plasma_loss_mw`, decided by `i_plasma_ignited` **and**
    `i_rad_loss` together, since one adds injected heating and the other subtracts a
    radiation term.
    """

    scaling: ConfinementTimeScaling = dataclasses.field(kw_only=True)
    """Which law runs (`i_confinement_time`)."""

    double_and_triple_product: DoubleAndTripleProduct = DoubleAndTripleProduct()
    """`.physics.ntau`/`.physics.nTtau`, downstream of `t_energy_confinement`."""

    tail: ConfinementTail = dataclasses.field(kw_only=True)
    """Everything downstream of the law, identical for all of them, and the reason the
    split does not duplicate anything.
    """


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
    # **`i_beta_fast_alpha` was a static kwarg here and is a slot now**
    # (`_audit/next_steps.md` §14.2). The note it replaces -- "kept as a static kwarg,
    # not a Switch, both branches read the same six variables" -- was correct about the
    # reads and is no longer the rule: a switch value selects an occupant whatever its
    # reads. Nothing structural moves; what moves is that the two published formulas are
    # two named occupants. Default `1` (WARD), `physics_variables.py:875`.
    fast_alpha_beta: FastAlphaBeta = dataclasses.field(kw_only=True)
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
    plasma_composition: PlasmaComposition = dataclasses.field(kw_only=True)
    """**`i_plasma_ignited` was a static kwarg here and is a slot now**
    (`_audit/next_steps.md` §14.2).
    """
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
