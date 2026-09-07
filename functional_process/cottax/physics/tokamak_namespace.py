"""The tokamak-only namespaces whose nodes live under `models/physics/`."""

import dataclasses

from cottax.interfaces.pytree_namespace_module import ModelNamespace

from functional_process.cottax.physics.current_drive import (
    FusionGain,
    HcdElectricTotal,
    HcdInjectedPowerTotal,
    HcdPrimaryEfficiency,
    HcdPrimaryInjectedPower,
    HcdPrimaryPowers,
    HcdSecondaryDrivenCurrent,
    HcdSecondaryHeating,
)
from functional_process.cottax.physics.exhaust import (
    EuDemoReAttachmentMetric,
    PsepOverRMetric,
)
from functional_process.cottax.physics.physics import (
    BetaLimitFromNorm,
    BetaNormMaxWesson,
    CoulombLogarithmIonElectron,
    PlasmaEnergyFromBeta,
    PlasmaOhmicHeating,
    PlasmaSurfaceNeutronFlux,
    PoloidalBeta,
    PositiveSeparatrixPower,
    PulseRampTimes,
    SeparatrixPower,
    ThermalBeta,
    ToroidalBeta,
    TotalRadiationPower,
    UnclippedRadiationPowers,
)
from functional_process.cottax.physics.plasma_geometry import (
    PlasmaGeometryArm,
    PlasmaMinorRadius,
    PlasmaShapeKappa95Triang95,
)
from functional_process.cottax.pulse import PulseBurnTime
from functional_process.cottax.stellarator.initialization import PulseDurations


class TokamakPhysics(ModelNamespace):
    """`.tokamak.physics` -- the radiation, separatrix and ohmic blocks of `physics.py`.
    """

    unclipped_radiation_powers: UnclippedRadiationPowers = UnclippedRadiationPowers()
    """The tokamak's counterpart to `stellarator.py`'s clipped arm, and the reason
    `PlasmaRadiationPowers` owns `_unclipped` mints rather than the real fields: one
    PROCESS function feeds both devices and only one caller clips.
    """

    total_radiation_power: TotalRadiationPower = TotalRadiationPower()

    separatrix_power: SeparatrixPower = dataclasses.field(kw_only=True)
    """`.physics.i_plasma_ignited` -- the two arms differ by exactly one read,
    `.current_drive.p_hcd_injected_total_mw`, and declaring the split is what keeps that
    `.current_drive -> .physics` edge out of the arm that does not make it.
    """

    ohmic_heating: PlasmaOhmicHeating = PlasmaOhmicHeating()
    """`Physics.plasma_ohmic_heating` (`physics.py:1605-1697`, written back `:768-778`).
    """

    positive_separatrix_power: PositiveSeparatrixPower = PositiveSeparatrixPower()
    """Owns the real `.physics.p_plasma_separatrix_mw`, downstream of the mint
    `.physics.p_plasma_separatrix_mw_raw`.
    """

    re_attachment_metric: EuDemoReAttachmentMetric = EuDemoReAttachmentMetric()
    """`.physics.p_div_bt_q_aspect_rmajor_mw` (`physics.py:818-826`)."""

    psep_over_r_metric: PsepOverRMetric = PsepOverRMetric()
    """`.physics.p_plasma_separatrix_rmajor_mw` (`physics.py:811-816`)."""

    coulomb_logarithm: CoulombLogarithmIonElectron = CoulombLogarithmIonElectron()
    """`.physics.dlamie` (`physics.py:279-283`)."""

    plasma_surface_neutron_flux: PlasmaSurfaceNeutronFlux = PlasmaSurfaceNeutronFlux()
    """`.physics.pflux_plasma_surface_neutron_avg_mw` (`physics.py:835-837`)."""


class TokamakPlasmaBeta(ModelNamespace):
    """`.tokamak.plasma_beta` -- `physics/physics.py::PlasmaBeta.run`, six slots."""

    energy_from_beta: PlasmaEnergyFromBeta = PlasmaEnergyFromBeta()
    """`.physics.e_plasma_beta` (`physics.py:3912-3916`) -- the original occupant of
    this slot, unchanged but for its name.
    """

    norm_max: BetaNormMaxWesson | None = dataclasses.field(kw_only=True)
    """`.physics.i_beta_norm_max` -- six values, one node and one empty arm."""

    limit: BetaLimitFromNorm = BetaLimitFromNorm()
    """`.physics.beta_vol_avg_max` (`physics.py:3810-3816`), constraint 24's bound."""

    toroidal: ToroidalBeta = ToroidalBeta()
    """`.physics.beta_toroidal_vol_avg` (`physics.py:3818-3822`)."""

    poloidal: PoloidalBeta = PoloidalBeta()
    """`.physics.beta_poloidal_vol_avg` (`physics.py:3825`)."""

    thermal: ThermalBeta = ThermalBeta()
    """`.physics.beta_thermal_vol_avg` (`physics.py:3831-3835`)."""


class TokamakPulse(ModelNamespace):
    """`.tokamak.pulse` -- the plasma-current ramp times and the pulse-duration sums."""

    ramp_times: PulseRampTimes = dataclasses.field(kw_only=True)
    """`(.pulse.i_pulsed_plant, .times.pulsetimings, .times.i_t_current_ramp_up)` --
    four arms, one written, and a joint arm index because no one of the three decides
    it.
    """

    durations: PulseDurations = PulseDurations()
    """`.times.t_plant_pulse_plasma_present`, `_no_burn` and `_total` -- and the class
    is `models/stellarator/initialization.py`'s, unchanged.
    """

    burn_time: PulseBurnTime | None = dataclasses.field(kw_only=True)
    """`Pulse.calculate_burn_time` (`pulse.py:275-316`)."""


class TokamakCurrentDrive(ModelNamespace):
    """`.tokamak.current_drive` -- heating and current drive, eight slots."""

    primary_efficiency: HcdPrimaryEfficiency = dataclasses.field(kw_only=True)
    """`.current_drive.i_hcd_primary` -- thirteen values, one occupant."""

    secondary_heating: HcdSecondaryHeating = dataclasses.field(kw_only=True)
    """`.current_drive.i_hcd_secondary` -- `0` (`NO_CURRENT_DRIVE`, PROCESS's default,
    `current_drive_variables.py:206`) is written.
    """

    secondary_driven_current: HcdSecondaryDrivenCurrent = HcdSecondaryDrivenCurrent()
    primary_injected_power: HcdPrimaryInjectedPower = HcdPrimaryInjectedPower()
    """Unswitched, and outside every `if` in PROCESS's own method."""

    primary_powers: HcdPrimaryPowers = dataclasses.field(kw_only=True)
    """`(.current_drive.i_hcd_primary, .current_drive.i_hcd_secondary)` -- **a joint
    key, and the only genuinely combinatorial one in this port.** The coupling is an
    accumulator: the primary block's `+=` at `current_drive.py:2147` starts from
    whatever the secondary block left at `:1955`, so the primary technology and the
    secondary technology together decide the arm.
    """

    injected_power_total: HcdInjectedPowerTotal = HcdInjectedPowerTotal()

    electric_total: HcdElectricTotal = dataclasses.field(kw_only=True)
    """`.physics.i_plasma_ignited` -- **both arms are written**, which is unusual enough
    in this wave to name: an ignited plasma draws no wall-plug power for heating, so the
    `IGNITED` occupant reads nothing and returns zero, and the `NON_IGNITED` one sums
    the primary and secondary electric powers.
    """

    fusion_gain: FusionGain = FusionGain()
    """Unswitched, and PROCESS's own last statement in this method
    (`current_drive.py:2301-2308`).
    """


class TokamakPlasmaGeom(ModelNamespace):
    """`.tokamak.plasma_geom` -- plasma cross-sectional shape, three slots."""

    minor_radius: PlasmaMinorRadius = PlasmaMinorRadius()
    """`.physics.rminor` and `.physics.eps` from `rmajor`/`aspect`."""

    shape: PlasmaShapeKappa95Triang95 = dataclasses.field(kw_only=True)
    """`.physics.i_plasma_geometry` -- thirteen values, one occupant."""

    geometry: PlasmaGeometryArm = dataclasses.field(kw_only=True)
    """The compound predicate `.physics.i_plasma_current == 8 or .physics.i_plasma_shape
    == SAUTER` (`plasma_geometry.py:467-470`) -- **one boolean, evaluated once by the
    factory**, and the cleanest result in this record: a compound switch does not have
    to become a compound node.
    """
