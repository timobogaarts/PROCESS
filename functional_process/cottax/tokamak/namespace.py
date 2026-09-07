"""The tokamak's own subsystems -- twenty-eight filled, one still empty."""

import dataclasses

from cottax.interfaces.pytree_namespace_module import ModelNamespace

from functional_process.cottax.blankets.namespace import CcfeHcpb
from functional_process.cottax.cryostat import Cryostat
from functional_process.cottax.cs_fatigue import CsFatigue
from functional_process.cottax.fw import (
    FirstWall,
    FirstWallGeometry,
    RadiatedWallLoad,
)
from functional_process.cottax.namespace import Build, Divertor
from functional_process.cottax.pfcoil.namespace import CSCoil, PFCoil
from functional_process.cottax.physics.bootstrap_current import (
    BootstrapCurrentFractionScaling,
    PlasmaCurrentFractions,
    PlasmaDiamagneticCurrentFraction,
    PlasmaPfirschSchluterCurrentFraction,
)
from functional_process.cottax.physics.density_limit import TokamakDensityLimit
from functional_process.cottax.physics.l_h_transition import LHThresholdPower
from functional_process.cottax.physics.plasma_current import TokamakPlasmaCurrent
from functional_process.cottax.physics.plasma_fields import PlasmaFields
from functional_process.cottax.physics.plasma_inductance import TokamakPlasmaInductance
from functional_process.cottax.physics.scrape_off_layer import TokamakScrapeOffLayer
from functional_process.cottax.physics.tokamak_namespace import (
    TokamakCurrentDrive,
    TokamakPhysics,
    TokamakPlasmaBeta,
    TokamakPlasmaGeom,
    TokamakPulse,
)
from functional_process.cottax.shield import TokamakShield
from functional_process.cottax.structure import Structure
from functional_process.cottax.tfcoil.namespace import CiccSuperconductingTfCoil
from functional_process.cottax.vacuum.vacuum import VacuumVesselElliptical


class Tokamak(ModelNamespace):
    """Everything a conventional tokamak has and a stellarator does not."""

    # ---- plasma geometry and the tokamak arm of the shared physics body -------------

    plasma_geom: TokamakPlasmaGeom = dataclasses.field(kw_only=True)
    """`physics/plasma_geometry.py::PlasmaGeom`, §A row 0 -- 7 entered functions, 549
    entered LOC, 1 of them shared with the stellarator
    (`calculate_iter_physics_basis_elongation`, already ported).
    """

    physics: TokamakPhysics = dataclasses.field(kw_only=True)
    """`physics/physics.py::Physics`, §A row 2 -- the tokamak arm of the one 6931-line
    file, 11 entered functions of which 3 are shared and already in `.physics`.
    """

    plasma_inductance: TokamakPlasmaInductance = dataclasses.field(kw_only=True)
    """`physics/physics.py::PlasmaInductance`, §A row 2.1 -- injected into `Physics` and
    run at `physics.py:356`.
    """

    plasma_beta: TokamakPlasmaBeta = dataclasses.field(kw_only=True)
    """`physics/physics.py::PlasmaBeta`, §A row 2.3 (`physics.py:429`)."""

    plasma_current: TokamakPlasmaCurrent = dataclasses.field(kw_only=True)
    """`physics/plasma_current.py::PlasmaCurrent`, §A row 2.4 (`physics.py:527`) -- 4
    entered functions, 296 entered LOC, none shared.
    """

    bootstrap_current: BootstrapCurrentFractionScaling | None = dataclasses.field(
        kw_only=True
    )
    """`physics/bootstrap_current.py::PlasmaBootstrapCurrent`, §A row 2.5
    (`physics.py:543`) -- 14 entered functions, 1228 entered LOC.
    """

    diamagnetic_current: PlasmaDiamagneticCurrentFraction = dataclasses.field(
        kw_only=True
    )
    """`physics/plasma_current.py::PlasmaDiamagneticCurrent` (`physics.py:527`) -- **a
    new slot, not one of the traced twenty-five**: `tokamak_boundary.md` folded it into
    `plasma_current`, whose record then declared it out of scope, so
    `bootstrap_current.md`'s registration instructions gave it a home beside its only
    consumer (`current_fractions` below).
    """

    pfirsch_schluter_current: PlasmaPfirschSchluterCurrentFraction = dataclasses.field(
        kw_only=True
    )
    """`physics.py:534-541`'s Pfirsch-Schluter fraction -- a new slot on the same
    grounds as `diamagnetic_current`.
    """

    current_fractions: PlasmaCurrentFractions = PlasmaCurrentFractions()
    """`Physics.calculate_plasma_current_fractions` (`physics.py:558-591`) -- sums the
    bootstrap, diamagnetic and Pfirsch-Schluter fractions into the inductive and
    auxiliary ones.
    """

    l_h_transition: LHThresholdPower = dataclasses.field(kw_only=True)
    """`physics/l_h_transition.py::PlasmaConfinementTransition`, §A row 2.6
    (`physics.py:788`) -- 23 entered functions, 1124 entered LOC.
    """

    scrape_off_layer: TokamakScrapeOffLayer = dataclasses.field(kw_only=True)
    """`physics/scrape_off_layer.py::ScrapeOffLayer`, §A row 2.7 (`physics.py:832`) -- 5
    entered functions, 226 entered LOC.
    """

    density_limit: TokamakDensityLimit = dataclasses.field(kw_only=True)
    """`physics/density_limit.py::PlasmaDensityLimit`, §A row 2.8 (`physics.py:870`) --
    11 entered functions, 531 entered LOC, 0 shared.
    """

    current_drive: TokamakCurrentDrive = dataclasses.field(kw_only=True)
    """`physics/current_drive.py::CurrentDrive` and its four injected sources
    (`NeutralBeam`, `ElectronCyclotron`, `LowerHybrid`, `ElectronBernstein`), run from
    `physics.py:593` when `i_hcd_calculations != 0` -- 5 entered functions, 737 entered
    LOC, unported.
    """

    plasma_fields: PlasmaFields = dataclasses.field(kw_only=True)
    """`physics/plasma_fields.py::PlasmaFields`, injected into `Physics`
    (`physics.py:197`) -- 1 entered function, 67 entered LOC.
    """

    # ---- the machine ---------------------------------------------------------------

    build: Build = dataclasses.field(kw_only=True)
    """`build.py::Build`, §A row 1 (`caller.py:288`) -- 6 entered functions but 2306 of
    the file's 2360 LOC entered, unported, nothing shared.
    """

    cicc_superconducting_tf_coil: CiccSuperconductingTfCoil = dataclasses.field(
        kw_only=True
    )
    """`tfcoil/superconducting.py::CICCSuperconductingTFCoil`, §A row 3
    (`caller.py:306`) -- 19 entered functions, 2457 entered LOC, unported, nothing
    shared, and with `tfcoil/base.py::TFCoil` (8 functions, 753 LOC) reached through it
    by inheritance rather than by any call in `caller.py`.
    """

    pf_coil: PFCoil = dataclasses.field(kw_only=True)
    """`pfcoil.py::PFCoil`, §A row 4 (`caller.py:319`) -- twelve slots including the
    inductance matrix.
    """

    cs_coil: CSCoil | None = dataclasses.field(kw_only=True)
    """`pfcoil.py::CSCoil`, injected at `main.py:652` -- seven slots."""

    cs_fatigue: CsFatigue = CsFatigue()
    """`cs_fatigue.py::CsFatigue`, injected at `main.py:652` and reached through
    `pfcoil.py:3492` -- 1 entered function (`ncycle`), 93 entered LOC.
    """

    pulse: TokamakPulse = dataclasses.field(kw_only=True)
    """`pulse.py::Pulse`, §A row 5 (`caller.py:322`) -- 12 entered functions, 236
    entered LOC, 3 shared.
    """

    divertor: Divertor = dataclasses.field(kw_only=True)
    """`divertor.py::Divertor`, §A row 6 (`caller.py:324`) -- 5 entered functions, 262
    entered LOC, unported.
    """

    first_wall: FirstWall = dataclasses.field(kw_only=True)
    """`fw.py::FirstWall`, §A row 7 (`caller.py:327`) -- 6 entered functions, 299
    entered LOC, unported.
    """

    first_wall_geometry: FirstWallGeometry = FirstWallGeometry()
    """`fw.py::FirstWall.set_fw_geometry` (`fw.py:347-352`) -- **a new slot, not one of
    the traced twenty-five**, on the same grounds as `diamagnetic_current`:
    `tokamak_boundary.md` folded it into `first_wall`, whose record then declared it out
    of that slot's boundary, and `cold_boundary.md` (2026-08-27) measured it as producer
    1 of the cold MDA's missing four -- `.build.dr_fw_inboard`/`.build.dr_fw_outboard`,
    the boundary zeros behind 7 of the 11 non-finite roots.
    """

    radiated_wall_load: RadiatedWallLoad = RadiatedWallLoad()
    """`fw.py:130-144` -- `.physics.pflux_fw_rad_mw` and
    `.constraints.pflux_fw_rad_max_mw`.
    """

    shield: TokamakShield = dataclasses.field(kw_only=True)
    """`shield.py::Shield`, §A row 8 (`caller.py:329`) -- 4 entered functions, 270
    entered LOC.
    """

    vacuum_vessel: VacuumVesselElliptical = dataclasses.field(kw_only=True)
    """`vacuum.py::VacuumVessel`, §A row 9 (`caller.py:331`) -- 3 entered functions."""

    ccfe_hcpb: CcfeHcpb = dataclasses.field(kw_only=True)
    """`blankets/hcpb.py::CCFE_HCPB`, §A row 10 (`caller.py:345`) -- 7 entered
    functions, 956 entered LOC, 0 shared, and `blankets/blanket_library.py` (14
    functions, 822 LOC) reached through it by inheritance (`hcpb.py:25`), never by a
    call in `caller.py`.
    """

    cryostat: Cryostat = Cryostat()
    """`cryostat.py::Cryostat`, §A row 11 (`caller.py:351`) -- 2 entered functions, 69
    entered LOC).
    """

    structure: Structure = dataclasses.field(kw_only=True)
    """`structure.py::Structure`, §A row 12 (`caller.py:354`) -- 2 entered functions,
    200 entered LOC.
    """

    water_use: ModelNamespace | None = None
    """`water_use.py::WaterUse`, §A row 19 (`caller.py:385`) -- 7 entered functions, 265
    entered LOC, unported and not reached by the stellarator at all.
    """
