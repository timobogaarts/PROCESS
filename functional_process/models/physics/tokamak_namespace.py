"""The tokamak-only namespaces whose nodes live under `models/physics/`.

`models/physics/namespace.py` beside this one holds `Physics` -- the **shared**
plasma-physics subsystem, 31 of whose 33 nodes both devices enter. What is here is the
other kind: the four `Tokamak` slots whose occupants happen to be filed under
`models/physics/` because PROCESS files them under `process/models/physics/`, and which
a stellarator never reaches at all.

Two namespaces rather than one file each, for the reason `model_tree_design.md` §11
gives: a namespace lives beside the nodes it names, and these name nodes in
`physics/physics.py`, `physics/current_drive.py` and `physics/plasma_geometry.py`.
`.tokamak.plasma_fields` is the exception and has its own module
(`physics/plasma_fields.py`), because that slot needed a node written for it.

**The name collision with `TokamakProcess.physics` is deliberate**, and
`models/tokamak/namespace.py` already argues it: `.physics` is the shared subsystem,
`.tokamak.physics` is the part of the same 6931-line PROCESS file only a tokamak enters.
Naming the second one something else would hide that they are one source file.
"""

import dataclasses

from cottax.interfaces.pytree_namespace_module import ModelNamespace

from functional_process.models.physics.current_drive import (
    HcdElectricTotal,
    HcdInjectedPowerTotal,
    HcdPrimaryEfficiency,
    HcdPrimaryInjectedPower,
    HcdPrimaryPowers,
    HcdSecondaryDrivenCurrent,
    HcdSecondaryHeating,
)
from functional_process.models.physics.physics import (
    PlasmaOhmicHeating,
    PositiveSeparatrixPower,
    PulseRampTimes,
    SeparatrixPower,
    TotalRadiationPower,
    UnclippedRadiationPowers,
)
from functional_process.models.physics.plasma_geometry import (
    PlasmaGeometryArm,
    PlasmaMinorRadius,
    PlasmaShapeKappa95Triang95,
)
from functional_process.models.pulse import PulseBurnTime
from functional_process.models.stellarator.initialization import PulseDurations


class TokamakPhysics(ModelNamespace):
    """`.tokamak.physics` -- the radiation, separatrix and ohmic blocks of
    `physics.py`.

    Five slots, one of them switched. What is *not* here and might be expected:

    * `SurfaceAveragedPoloidalField` is `.tokamak.plasma_fields`', because PROCESS puts
      it in `plasma_fields.py` and injects that model into `Physics`.
    * `PlasmaEnergyFromBeta` is `.tokamak.plasma_beta`', for the same reason -- it is
      `PlasmaBeta.run`'s, `physics.py:3912-3916`.
    * the pulse ramp times are `.tokamak.pulse`'s. `physics.md` OQ4 left that a slot
      choice rather than a re-port; `pulse.py::Pulse` is where every other pulse-timing
      concern lives and the `Tokamak.pulse` slot's own docstring already claims decision
      15 (`pulsetimings`), so the code's file is the weaker argument here.
    """

    unclipped_radiation_powers: UnclippedRadiationPowers = UnclippedRadiationPowers()
    """The tokamak's counterpart to `stellarator.py`'s clipped arm, and the reason
    `PlasmaRadiationPowers` owns `_unclipped` mints rather than the real fields: one
    PROCESS function feeds both devices and only one caller clips."""

    total_radiation_power: TotalRadiationPower = TotalRadiationPower()

    separatrix_power: SeparatrixPower = dataclasses.field(kw_only=True)
    """`.physics.i_plasma_ignited` -- the two arms differ by exactly one read,
    `.current_drive.p_hcd_injected_total_mw`, and declaring the split is what keeps that
    `.current_drive -> .physics` edge out of the arm that does not make it.

    The live arm here is `NON_IGNITED`, the **opposite** of
    `PhysicsConfinementTime.power_loss`'s live arm on the stellarator runs. Two nodes,
    two configurations, one switch, no conflict -- and a good reminder that "the live
    arm" is a property of a machine and not of a switch."""

    ohmic_heating: PlasmaOhmicHeating = PlasmaOhmicHeating()
    """`Physics.plasma_ohmic_heating` (`physics.py:1605-1697`, written back
    `:768-778`). Unswitched -- `run` computes it unconditionally -- so a default, per
    the `plasma_beta` rule. Added 2026-08-27, `cold_boundary.md` producer 3: its
    `.physics.res_plasma` was the boundary zero that (jointly with
    `.pf_coil.vs_cs_pf_total_burn`) made the cold burn time nan. Ports PROCESS's
    chained-comparison defect in the neo-classical factor as-is; see
    `plasma_ohmic_heating`'s docstring."""

    positive_separatrix_power: PositiveSeparatrixPower = PositiveSeparatrixPower()
    """Owns the real `.physics.p_plasma_separatrix_mw`, downstream of the mint
    `.physics.p_plasma_separatrix_mw_raw`.

    **The mint is what resolves a field PROCESS writes twice in one pass.** Three call
    sites read the first value (`physics.py:811-816`, `:818-826`, `:832`) and everything
    after `:845` sees the second, so one node applying the transform internally would
    hand those three the post-transform number -- a silent behaviour change. Same shape
    and same resolution as `radiation_power.py`'s `_unclipped` mints and
    `build.py`'s `r_tf_outboard_mid_unrippled`; `physics.md` OQ5 asked for this call and
    the mint is what is kept."""


class TokamakPulse(ModelNamespace):
    """`.tokamak.pulse` -- the plasma-current ramp times and the pulse-duration sums.

    `process/models/pulse.py::Pulse`, `caller.py:322`. Three slots, and the second of
    them is **a registration rather than a port**.
    """

    ramp_times: PulseRampTimes = dataclasses.field(kw_only=True)
    """`(.pulse.i_pulsed_plant, .times.pulsetimings, .times.i_t_current_ramp_up)` --
    four arms, one written, and a joint arm index because no one of the three decides it.

    `pulsetimings` is read at `physics.py:476` and **nowhere else in all of
    `process/models/**`**, which is `tokamak_call_surface.md` §E's sharpest single-site
    result; this slot is the whole of that decision.

    Arm 3 is refused for a stronger reason than "not written": `physics.py:489-492`
    reads `.times.t_plant_pulse_coil_precharge` and writes it back, so its occupant
    would read what it owns. That needs a `FixedPointFunction` or a producer split, and
    `physics.md` OQ2 declined to improvise one."""

    durations: PulseDurations = PulseDurations()
    """`.times.t_plant_pulse_plasma_present`, `_no_burn` and `_total` -- and the class is
    `models/stellarator/initialization.py`'s, unchanged.

    **Two of `.tokamak.physics`'s eight boundary outputs are closed by a registration
    rather than by a port.** `calculate_pulse_durations` is `pulse.py:71-95` term for
    term, checked; the stellarator reaches the same three sums through `st_init` and the
    tokamak through `Pulse.run`, and there is no formula difference to port.

    **It should move to a shared module**, and does not here. A node is named by the slot
    that holds it, so the file it is declared in changes no name and moving it is pure
    churn on a wave that is already large; `physics.md`'s "already-ported sub-calls"
    section says the same. Recorded so that a reader who finds a stellarator import in
    the tokamak's pulse namespace sees a decision rather than an accident."""

    burn_time: PulseBurnTime = PulseBurnTime()
    """`Pulse.calculate_burn_time` (`pulse.py:275-316`). No switch of its own --
    `i_pulsed_plant` decides whether this node's *machine* is pulsed, not which formula
    it uses -- so no `kw_only` factory, per `pulse.md`'s registration instructions.
    Reads `.pf_coil.vs_cs_pf_total_burn`, produced since 2026-08-27 by
    `.tokamak.pf_coil.volt_seconds` (`pfcoil.py::vsec`, `cold_boundary.md` producer 4)
    -- the registration that merged this node's two-node ring with the PF coil ring
    into one nine-node SCC; see `models/pfcoil/volt_seconds.py`."""


class TokamakCurrentDrive(ModelNamespace):
    """`.tokamak.current_drive` -- heating and current drive, seven slots.

    `process/models/physics/current_drive.py::CurrentDrive` and its four injected
    sources, run from `physics.py:593` when `.current_drive.i_hcd_calculations != 0`.
    That switch is **topology, not an occupant**: `1` means these nodes exist and `0`
    means none of them does, so it is answered by whether the slot is filled at all.

    Four of the seven slots are switched. Three are not, and that is a **result** rather
    than a convenience: PROCESS computes those three lines outside every `if` in the
    method (`current_drive.py:1821-1855`, `:2265-2270`), so the split found real
    unswitched work inside a method that looks switched throughout.
    """

    primary_efficiency: HcdPrimaryEfficiency = dataclasses.field(kw_only=True)
    """`.current_drive.i_hcd_primary` -- thirteen values, one occupant.

    `10` (`USER_INPUT_ELECTRON_CYCLOTRON`, `large_tokamak_eval.IN.DAT:124`) is written.
    Two of the refused values are refused for a reason no port can fix: `6` and `7`
    (`CULHAM_LOWER_HYBRID`, `CULHAM_ELECTRON_CYCLOTRON`) **PROCESS itself cannot
    execute** -- `calculate_profile_y` returns `None` and the arms raise `TypeError` at
    `current_drive.py:1498` and `:815`. Two live PROCESS defects, found by this port and
    recorded in `current_drive.md`."""

    secondary_heating: HcdSecondaryHeating = dataclasses.field(kw_only=True)
    """`.current_drive.i_hcd_secondary` -- `0` (`NO_CURRENT_DRIVE`, PROCESS's default,
    `current_drive_variables.py:206`) is written. A node that reads nothing and returns
    three zeros, which is legitimate for the same reason `CentrepostNeutronicsAbsent` is:
    PROCESS's own source on this arm assigns literals."""

    secondary_driven_current: HcdSecondaryDrivenCurrent = HcdSecondaryDrivenCurrent()
    primary_injected_power: HcdPrimaryInjectedPower = HcdPrimaryInjectedPower()
    """Unswitched, and outside every `if` in PROCESS's own method."""

    primary_powers: HcdPrimaryPowers = dataclasses.field(kw_only=True)
    """`(.current_drive.i_hcd_primary, .current_drive.i_hcd_secondary)` -- **a joint
    key, and the only genuinely combinatorial one in this port.**

    The coupling is an accumulator: the primary block's `+=` at
    `current_drive.py:2147` starts from whatever the secondary block left at `:1955`, so
    the primary technology and the secondary technology together decide the arm. Five
    primary methods times six secondary methods in principle; one cell written, the one
    `large_tokamak_eval` runs (electron cyclotron, no secondary).

    `current_drive.md` flags the shape as the thing to fix upstream rather than to scale:
    a per-technology "secondary contribution" field would make this two slots instead of
    a product. That needs a minted name PROCESS does not have, so it is recorded, not
    done."""

    injected_power_total: HcdInjectedPowerTotal = HcdInjectedPowerTotal()

    electric_total: HcdElectricTotal = dataclasses.field(kw_only=True)
    """`.physics.i_plasma_ignited` -- **both arms are written**, which is unusual enough
    in this wave to name: an ignited plasma draws no wall-plug power for heating, so the
    `IGNITED` occupant reads nothing and returns zero, and the `NON_IGNITED` one sums the
    primary and secondary electric powers.

    Owns `.heat_transport.p_hcd_electric_total_mw`, and the reason that is worth
    checking rather than assuming: no stellarator node owns it -- the stellarator's
    heating slot (`EcrhHeating`/`LowhybHeating`) owns
    `.heat_transport.p_hcd_electric_loss_mw` and `.current_drive.p_hcd_injected_total_mw`
    and leaves this field a boundary input. So the two devices do not collide, and they
    could not: ownership is per-graph and the two graphs are never assembled together."""


class TokamakPlasmaGeom(ModelNamespace):
    """`.tokamak.plasma_geom` -- plasma cross-sectional shape, three slots.

    `process/models/physics/plasma_geometry.py::PlasmaGeom`, `caller.py`'s first physics
    model. One of the three slots is unswitched, one answers a single switch, and one
    answers a **compound** predicate over two.
    """

    minor_radius: PlasmaMinorRadius = PlasmaMinorRadius()
    """`.physics.rminor` and `.physics.eps` from `rmajor`/`aspect`. Unconditional --
    PROCESS computes it before the dispatch."""

    shape: PlasmaShapeKappa95Triang95 = dataclasses.field(kw_only=True)
    """`.physics.i_plasma_geometry` -- thirteen values, one occupant.

    `0` (`IPDG89_X_POINT`, `large_tokamak_eval.IN.DAT:301`) is written; the other twelve
    read genuinely different fields and each needs its own class when someone needs it.
    `plasma_geometry.md`'s open question "eight occupants or thirteen?" is **superseded**
    by this wave's binding policy rather than answered by it: one occupant class per
    value ever supported, no grouping by reads-identical sets."""

    geometry: PlasmaGeometryArm = dataclasses.field(kw_only=True)
    """The compound predicate `.physics.i_plasma_current == 8 or .physics.i_plasma_shape
    == SAUTER` (`plasma_geometry.py:467-470`) -- **one boolean, evaluated once by the
    factory**, and the cleanest result in this record: a compound switch does not have to
    become a compound node.

    The `False` arm (double-arc) is written and live. The `True` arm's pure functions are
    ported (`sauter_geometry`, `calculate_geometry_sauter`) and deliberately not wired:
    no tracked input reaches it, so it has no regression oracle at all.

    **The predicate is owned in one place on purpose.** `indat.py`'s
    `_plasma_geometry_arm` is it, and whoever ports `plasma_current.py`'s own
    `i_plasma_current` topology split must call that function rather than re-derive the
    disjunction -- two independent derivations of one predicate is how the two halves
    drift apart."""
