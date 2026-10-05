"""The device's own namespaces -- its coils, its FWBS, and `Stellarator` itself."""

import dataclasses

import equinox as eqx
from cottax.relational_pytree import Explicit, Implicit

from functional_process.cottax.models.stellarator.build import (
    Build,
)
from functional_process.cottax.models.stellarator.coils.calculate import (
    CoilCasing,
    CoilCoilToroidalGap,
    CoilCrossSectionalArea,
    CoilCurrent,
    CoilHalfWidths,
    CoilRadialThickness,
    CoilsSummaryVariables,
    CoilToroidalThickness,
    HorizontalPorts,
    LenTfCoil,
    PlasmaFacingCoilArea,
    StoredMagneticEnergy,
    TfCryoArea,
    VerticalPorts,
    WindingPackGeometry,
    WindingPackTotalSizePost,
    ZTfInsideHalf,
)
from functional_process.cottax.models.stellarator.coils.coils import Intersect
from functional_process.cottax.models.stellarator.coils.forces import (
    MaxForceDensity,
    MaximumStress,
)
from functional_process.cottax.models.stellarator.coils.mass import CoilsMass
from functional_process.cottax.models.stellarator.coils.quench import QuenchProtection
from functional_process.cottax.models.stellarator.density_limits import (
    SudoDensityLimit,
)
from functional_process.cottax.models.stellarator.divertor import Divertor
from functional_process.cottax.models.stellarator.geometry import (
    DefaultAspectRatio,
    StellaratorPlasmaGeometry,
    StellaratorScalingFactors,
)
from functional_process.cottax.models.stellarator.heating import (
    BeamCurrent,
    FusionGain,
    InjectedPowerTotal,
)
from functional_process.cottax.models.stellarator.initialization import PulseDurations
from functional_process.cottax.models.stellarator.neoclassics import (
    EffectiveThermalDiffusivity,
    ProfileValues,
)
from functional_process.cottax.models.stellarator.plasma_physics import (
    ClippedRadiationPowers,
    PoloidalFieldFromRotationalTransform,
    StellaratorBetaAndStoredEnergy,
    ThermalEnergyTotals,
    TotalField,
)
from functional_process.cottax.models.stellarator.stellarator_fwbs_s1_s5 import (
    CryostatAndVvGeometry,
)
from functional_process.cottax.models.stellarator.stellarator_fwbs_s2 import (
    ExponentialAttenuationBlanketShieldPower,
)
from functional_process.cottax.models.stellarator.stellarator_fwbs_s3 import (
    DivertorPlateMass,
)
from functional_process.cottax.models.stellarator.stellarator_fwbs_s4 import (
    ShieldMass,
)
from functional_process.cottax.models.stellarator.structure import (
    StructureMasses,
)
from functional_process.cottax.models.stellarator.tf_nuclear_heating import (
    ScTfCoilNuclearHeating,
)


class BlanketShieldPowerExponential(eqx.Module):
    """Exponential-attenuation blanket/shield power: `blktmodel == 0 & ipowerflow == 0`."""

    # Over the line length and left that way -- see `Physics`'s own note: the slot
    # name and the occupant class are both this long and `ruff format` strips
    # parentheses from around an annotation.
    exponential_attenuation_blanket_shield_power: Explicit = (
        ExponentialAttenuationBlanketShieldPower  # noqa: E501
    )
    # `ScTfCoilNuclearHeating` moved here from the unswitched part -- this arm is
    # its one genuine caller that keeps its `p_tf_nuclear_heat_mw`
    # output (`stellarator.py:727-728`); the `blktmodel == 1` arm calls
    # it too but discards that particular output (`stellarator.py:465-476`
    # unpacks nine `_`s and keeps only `flu_tf_neutron_fast_peak`), and arm 2
    # computes its own, different `p_tf_nuclear_heat_mw` formula.
    # Leaving it unconditional was a real bug, the same
    # class already fixed once for `EcrhDensityLimit`: PROCESS's actual
    # default configuration lands in arm 2, not this one, so the
    # old unconditional placement was computing SC-coil TF nuclear
    # heating via the wrong formula for the default `GRAPH`.
    sc_tf_coil_nuclear_heating: Explicit = ScTfCoilNuclearHeating


class StellaratorCoils(eqx.Module):
    """The modular-coil set: geometry, current, casing, ports, structure, cryogenics."""

    # unit #9, coils/calculate.py
    coil_toroidal_thickness: Explicit = CoilToroidalThickness
    coil_radial_thickness: Explicit = CoilRadialThickness
    coil_cross_sectional_area: Explicit = CoilCrossSectionalArea
    coil_half_widths: Explicit = CoilHalfWidths
    plasma_facing_coil_area: Explicit = PlasmaFacingCoilArea
    coil_coil_toroidal_gap: Explicit = CoilCoilToroidalGap
    coils_summary_variables: Explicit = CoilsSummaryVariables
    stored_magnetic_energy: Explicit = StoredMagneticEnergy
    winding_pack_geometry: Explicit = WindingPackGeometry
    coil_current: Explicit = CoilCurrent
    coil_casing: Explicit = CoilCasing
    vertical_ports: Explicit = VerticalPorts
    horizontal_ports: Explicit = HorizontalPorts
    # `st_coil`'s formula for `.build.z_tf_inside_half` -- see `Build`'s own comment
    # above (unit #2, build.py) for why this one, not `Build`'s, owns the field.
    z_tf_inside_half: Explicit = ZTfInsideHalf
    # `.tfcoil.tfcryoarea`, carved out of the same inline `st_coil` geometry block as
    # `ZTfInsideHalf` and for the same reason (the eager `st_coil` orchestrator is not
    # registered, so anything only it computes has no owner). Prerequisite for
    # `CryoQLoadsStep` below: without it, registering the cryo nodes would have traded
    # two boundary inputs for one new one (`_audit/boundary_inputs_audit.md` §4c (c1)'s
    # sibling gap, §7 items 4 and 7). Of its two neighbours in that block,
    # `min_bending_radius` still stays unported for want of any reader.
    tf_cryo_area: Explicit = TfCryoArea
    # `.tfcoil.len_tf_coil`, the third formula from that same block, and the one held
    # back longest: four registered nodes read it (`StructureMasses`,
    # `PlasmaFacingCoilArea`, `CoilsMass`, `TfMagnetCostSuperconducting`) while it had
    # no producer at all. **The stale-vs-fresh decision it was waiting on is resolved,
    # in favour of binding fresh** -- there is no feedback path back into it (measured:
    # its producer's owned input comes from `StellaratorScalingFactors`, unreachable
    # from all four readers), so a `FixedPointFunction` self-loop modelling PROCESS's
    # read-before-write would be a degenerate fixed point that
    # `degenerate_fixed_points` deletes on sight. See `LenTfCoil`'s own docstring for
    # the full argument and the one honest caveat. This also closes the cold-start
    # `nan` in `.costs.c22211`/`.c2221` (`costs.md`).
    len_tf_coil: Explicit = LenTfCoil
    # unit #12, coils/mass.py
    coils_mass: Explicit = CoilsMass
    """The TF coil masses, at the material's density `.tfcoil.den_tf_sc_material`."""
    # unit #11, coils/forces.py
    max_force_density: Explicit = MaxForceDensity
    maximum_stress: Explicit = MaximumStress
    # unit #14, coils/quench.py
    quench_protection: Explicit = QuenchProtection
    # `coils/calculate.py`'s `winding_pack_total_size` (unit #9's remaining tier-2 gap),
    # now the full three-piece split: `winding_pack_intersect_inputs` (pre-`intersect`,
    # a **slot**, mints `.stellarator.wp_width_r`/`.lhs`/`.rhs` and
    # `.wp_width_r_min_guess`), `coils.py`'s `Intersect`
    # (`ImplicitFunction`/`RootFind`, owns `.stellarator.wp_width_r_min`),
    # `WindingPackTotalSizePost` (post-`intersect`, owns `.tfcoil.j_tf_wp` along with
    # everything else `winding_pack_post_intersect` computes).
    #
    # **An earlier pass registered `WindingPackJTfWp` here instead** -- a
    # `FixedPointFunction` that isolated just `.tfcoil.j_tf_wp` by re-running the whole
    # (unchanged) `winding_pack_total_size` pure function internally, duplicating the
    # 200-point sampling and eager `intersect` call the three-piece split next to it
    # already did, unregistered. That duplication is why it is gone: once
    # `WindingPackTotalSizePost` owns `.tfcoil.j_tf_wp` (it used to discard the fresh
    # value, deferring ownership to `WindingPackJTfWp`) and `WindingPackIntersectInputs`
    # reads it (already did), the self-reference closes through the three real nodes
    # plus `Intersect`'s own `RootFind` problem -- one merged 4-node SCC, the same
    # "Shape A" cross-node-cycle shape as `Divertor`/`AFwTotalWithPowerflow` below, not a
    # self-loop on any single node. `Blocking`/`to_graph()` finds it with no
    # `FixedPointFunction`/`Cut` wrapper at all (confirmed:
    # `test_winding_pack_intersect_split_forms_one_combined_cycle` in
    # `test_calculate.py`).
    # `Intersect` (like every other undriven declared node in this graph) needs no
    # production driver to be registered here -- structural admission only, driving
    # deferred, per `_audit/next_steps.md` §5.
    winding_pack_intersect_inputs: Explicit = dataclasses.field(kw_only=True)
    """The pre-`intersect` curves, on `.tfcoil.i_tf_sc_mat` -- one occupant per
    superconductor, `WINDING_PACK_MATERIAL` in `indat.py`.
    """
    intersect: Implicit = Intersect
    winding_pack_total_size_post: Explicit = WindingPackTotalSizePost


class StellaratorFwbs(eqx.Module):
    """First wall, blanket and shield -- the `st_fwbs` chunk's registered nodes."""

    blanket_shield_power: Explicit = dataclasses.field(kw_only=True)
    """Blanket/shield power deposition, on `.fwbs.blktmodel` x `.heat_transport.
    ipowerflow` x `.fwbs.i_p_coolant_pumping` jointly -- one slot, three integers,
    resolved by `_blanket_shield_power_arm` in `machine_from_indat`.
    """

    blanket_masses: Explicit = dataclasses.field(kw_only=True)
    """Blanket component masses, on `.fwbs.blktmodel` x `.fwbs.blkttype` jointly,
    resolved by `_blanket_mass_arm`.
    """

    # `st_fwbs` S1/S5 (`stellarator_E_fwbs_synthesis.md`). S1's occupant is chosen by
    # `.heat_transport.ipowerflow` (`FwBlanketShieldGeometry*`).
    fw_blanket_shield_geometry: Explicit = dataclasses.field(kw_only=True)
    cryostat_and_vv_geometry: Explicit = CryostatAndVvGeometry
    # `st_fwbs` S3 (`stellarator_fwbs_s3.md`). Reads `.divertor.a_div_surface_total`,
    # which `Divertor` owns -- an ordinary acyclic edge, not a cycle: `Divertor`'s own
    # inputs have no dependency back on anything `st_fwbs`/`DivertorPlateMass` produces
    # (verified directly against `divertor.py`'s `FromExactly`s), so PROCESS's own staleness
    # here (`st_fwbs` runs before `st_div`, so it reads the *previous* `run()`'s value)
    # is a call-order artifact of its imperative code, not a genuine two-way dependency.
    # Registering this the ordinary way (`Divertor` before `DivertorPlateMass` in
    # topological order) is strictly more self-consistent than PROCESS's own lagged
    # read -- confirmed by the build below staying at the same SCC count.
    divertor_plate_mass: Explicit = DivertorPlateMass
    # `st_fwbs` S4's shield-mass block (`stellarator_fwbs_s4.md`). Unswitched, not
    # behind a `Switch` because `stellarator.py:1195-1206` is outside every branch in
    # `st_fwbs` -- no `blktmodel`, `blkttype` or `ipowerflow` guard -- so both outputs
    # exist in every configuration. Its sibling `BlanketComponentMasses` *is* switched,
    # see `TOPOLOGY_SWITCHES`'s `.fwbs.blktmodel,.fwbs.blkttype` entry. Closes
    # `_audit/boundary_inputs_audit.md` § 4c (b5)/(b6): `Bldgs` and `ShieldCost` were
    # reading `.fwbs.whtshld`, and `ShieldCost` `.fwbs.wpenshld`, as boundary inputs.
    shield_mass: Explicit = ShieldMass


class Stellarator(eqx.Module):
    """Everything device-specific: the machine's own geometry, coils, and FWBS."""

    machine_config: Explicit = dataclasses.field(kw_only=True)
    """The 34 `.stellarator_config.stella_config_*` fields, for whichever machine."""

    heating: Explicit = dataclasses.field(kw_only=True)
    """Which auxiliary heating model runs (`.stellarator.isthtr`, default 1 = ECRH)."""

    fw_area: Explicit = dataclasses.field(kw_only=True)
    """First-wall area (`.heat_transport.ipowerflow`, default 1)."""

    coils: StellaratorCoils = dataclasses.field(kw_only=True)
    """The coil sub-namespace."""

    fwbs: StellaratorFwbs = dataclasses.field(kw_only=True)

    # unit #1 chunks
    sudo_density_limit: Explicit = SudoDensityLimit
    # EcrhDensityLimit moved to TOPOLOGY_SWITCHES's `i_plasma_pedestal` switch -- its
    # static kwarg is no longer independent of that switch's value, see there.
    structure_masses: Explicit = StructureMasses
    # ScTfCoilNuclearHeating moved to TOPOLOGY_SWITCHES's new joint `blktmodel`/
    # `ipowerflow` switch (value=1 arm) -- see that switch's own comment. Unconditional
    # placement here was a real bug: PROCESS's own default configuration lands in the
    # switch's value=2 arm, which computes `.fwbs.p_tf_nuclear_heat_mw` via a different
    # formula (`DetailedPowerflowBlanketShieldPower`), not this one.
    # unit #2, build.py -- `BlktmodelBlanketThickness` deliberately NOT here (see the
    # import comment above): PROCESS's own default `blktmodel = 0` means this node must
    # not be instantiated at all (`conditional-ownership-by-run-config`), a bug fixed
    # this pass, same class as `ScTfCoilNuclearHeating`/`EcrhDensityLimit` above.
    # `Build` no longer owns `.build.z_tf_inside_half` -- moved to `coils/calculate.py`'s
    # `ZTfInsideHalf` (registered below, near the rest of unit #9/#10's coil nodes), a
    # real ordering-artifact bug found via the block-by-block MDA-vs-PROCESS comparison
    # harness: two independent PROCESS writers of this field, `Build`'s formula was the
    # transient one, not the one that survives a real run's final report pass. See
    # `build.py`'s `calculate_build` docstring and `ZTfInsideHalf`'s own for the full
    # account.
    build: Explicit = Build
    # unit #4, divertor.py
    divertor: Explicit = Divertor
    # unit #5, heating.py
    injected_power_total: Explicit = InjectedPowerTotal
    beam_current: Explicit = BeamCurrent
    fusion_gain: Explicit = FusionGain
    # unit #6, initialization.py
    pulse_durations: Explicit = PulseDurations
    # unit #7, neoclassics.py (scalar-argument functions only, see module docstring)
    profile_values: Explicit = ProfileValues
    effective_thermal_diffusivity: Explicit = EffectiveThermalDiffusivity
    # `plasma_physics.py` (chunk 1B of unit #1). `StellaratorBetaAndRhoStar` is
    # still NOT registered: its `.physics.rho_star` output is algebraically identical to
    # `DimensionlessPlasmaParameters`'s own `rho_star` formula above (same inputs, same
    # expression, confirmed by direct comparison) -- a genuine redundant-duplicate-write
    # in PROCESS itself (`st_phys` and `outplas` both compute it), not a porting choice.
    # Registering both would be a duplicate-ownership conflict, the same shape as
    # `IterPhysicsBasisElongation`/`ConfinementTime`'s `kappa_ipb` above.
    #
    # **What is new: dropping that node used to cost `.physics.beta_total_vol_avg` and
    # `.physics.e_plasma_beta` their only producer as collateral.** They are not in
    # conflict with anything -- only `rho_star` was. `StellaratorBetaAndStoredEnergy`
    # (same pure function, same 13 inputs, `rho_star`'s return value discarded) owns
    # exactly those two and is registered instead. `.physics.beta_total_vol_avg` is
    # constraint 24's only argument, one of the 14 active constraints of the reference
    # run, so this is what makes that constraint assemblable at all -- see that class's
    # own docstring.
    stellarator_beta_and_stored_energy: Explicit = StellaratorBetaAndStoredEnergy
    poloidal_field_from_rotational_transform: Explicit = (
        PoloidalFieldFromRotationalTransform
    )
    total_field: Explicit = TotalField
    # `stellarator.py:2152-2166`: `st_phys`'s two zero-clips on the radiation power
    # densities and the two total powers formed from them. Owns the real
    # `.physics.pden_plasma_*_rad_mw` fields, which `PlasmaRadiationPowers` now mints
    # as `*_unclipped` -- the clip has two disagreeing call sites in PROCESS, so it
    # belongs to this caller, not to `calculate_radiation_powers`. Also gives
    # `.physics.p_plasma_inner_rad_mw` (read by `StellaratorConfinementTime`) its first
    # producer -- `_audit/boundary_inputs_audit.md` §7 item 6.
    clipped_radiation_powers: Explicit = ClippedRadiationPowers
    neutron_wall_load: Explicit = dataclasses.field(kw_only=True)
    """Neutron wall load -- one occupant per arm of `.physics.i_pflux_fw_neutron` x
    `.heat_transport.ipowerflow` (`indat.py`'s `_wall_load_arm`).
    """
    heating_and_radiation_power: Explicit = dataclasses.field(kw_only=True)
    """Heating power, SOL radiation split and alpha power to the wall -- one occupant
    per `.physics.i_plasma_ignited` value.
    """
    radiated_wall_load_and_fraction: Explicit = dataclasses.field(kw_only=True)
    """Radiated wall load and radiation fraction -- the same three arms as
    `neutron_wall_load` above, from the same `_wall_load_arm` dispatch.
    """
    thermal_energy_totals: Explicit = ThermalEnergyTotals
    # `geometry.py` (chunk 1C of unit #1). `DefaultAspectRatio` is the
    # `1 not in data.numerics.ixc` conditional-ownership case (module docstring): the
    # bare `NumericsData` dataclass default (`ixc = [0, 0, ...]`, no real iteration-
    # variable ID ever present) makes `1 not in ixc` true, so this node is instantiated
    # unconditionally here, matching PROCESS's own bare-default configuration -- the
    # same convention every topology switch's own `default` already follows.
    # `StellaratorScalingFactors` takes `aspect` as a plain `FromExactly` regardless of source
    # (this node's own output, when active, or an external iteration-variable value
    # otherwise), so no further wiring decision is needed here.
    default_aspect_ratio: Explicit = DefaultAspectRatio
    stellarator_scaling_factors: Explicit = StellaratorScalingFactors
    stellarator_plasma_geometry: Explicit = StellaratorPlasmaGeometry
