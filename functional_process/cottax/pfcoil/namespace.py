"""The PF coil system's two namespaces: `.tokamak.cs_coil` and `.tokamak.pf_coil`."""

import dataclasses

from cottax.interfaces.pytree_namespace_module import ModelNamespace

from functional_process.cottax.pfcoil import SPHERICAL_TOKAMAK_TOPOLOGY
from functional_process.cottax.pfcoil.currents import (
    CSCurrentDensityPulseStart,
    CSFluxSwing,
    PFCoilEquilibriumCurrents,
    PFCoilInitiationCurrents,
    PFCoilInitiationCurrentsNoCentralSolenoid,
    PFCoilTimePointCurrents,
    PFCoilTimePointCurrentsNoCentralSolenoid,
)
from functional_process.cottax.pfcoil.fields import (
    CSCoilPeakField,
    PFCoilCurrentWaveform,
    PFCoilPeakField,
    PFCoilPeakFieldNoCentralSolenoid,
)
from functional_process.cottax.pfcoil.geometry import (
    CSCoilGeometry,
    CSCoilTurnGeometry,
    PFCoilPlacement,
    PFCoilPlacementSphericalTokamak,
    PFCoilPositions,
    PFCoilPositionsNoCentralSolenoid,
)
from functional_process.cottax.pfcoil.inductance import (
    PFCoilInductance,
    PFCoilInductanceNoCentralSolenoid,
)
from functional_process.cottax.pfcoil.masses import (
    PFCoilMasses,
    PFCoilMassesCsWstNb3Sn,
    PFCoilMassesNoCentralSolenoid,
    PFCoilSizes,
    PFCoilSizesNoCentralSolenoid,
)
from functional_process.cottax.pfcoil.stresses import CSCoilStresses
from functional_process.cottax.pfcoil.superconductor import (
    CSCriticalCurrentDensitiesIterNb3Sn,
    CSTemperatureMarginIterNb3Sn,
    PFStrandCriticalCurrentDensity,
    PFStrandCriticalCurrentDensityHazeltonZhaiRebco,
)
from functional_process.cottax.pfcoil.volt_seconds import (
    PFCoilTurnCurrents,
    PFCoilVoltSeconds,
    PFCoilVoltSecondsNoCentralSolenoid,
)


class CSCoil(ModelNamespace):
    """`.tokamak.cs_coil` -- the central solenoid, seven slots."""

    geometry: CSCoilGeometry = CSCoilGeometry()
    """The CS cross-section and its fourteen scaling filaments (`pfcoil.py:120-158`,
    `:202-234`).
    """

    turn_geometry: CSCoilTurnGeometry = CSCoilTurnGeometry()
    """The EU DEMO stadium-shaped CS turn and its steel conduit (`ohcalc`,
    `pfcoil.py:3296-3319`, via `calculate_cs_turn_geometry_eu_demo`).
    """

    current_density_pulse_start: CSCurrentDensityPulseStart = (
        CSCurrentDensityPulseStart()
    )
    """`j_cs_pulse_start = j_cs_flat_top_end * f_j_cs_start_pulse_end_flat_top`
    (`pfcoil.py:161-164`).
    """

    flux_swing: CSFluxSwing = CSFluxSwing()
    """The CS flux-swing balance (`pfcoil.py:600-657`) -- owns
    `.pf_coil.f_j_cs_start_end_flat_top`, and one of the two nodes whose reads make the
    package's four-node cycle (`.pf_coil.n_pf_coil_turns`,
    `.pf_coil.ind_pf_cs_plasma_mutual`).
    """

    peak_field: CSCoilPeakField = CSCoilPeakField()
    """The CS's own peak field at end-of-flat-top and beginning-of-pulse (`ohcalc`'s
    field block, `pfcoil.py:3327-3396`), plus index `[6]` of the two whole-array peak
    fields.
    """

    critical_current: CSCriticalCurrentDensitiesIterNb3Sn = dataclasses.field(
        kw_only=True
    )
    """`.pf_coil.i_cs_superconductor` -- two reachable values, both written."""

    temperature_margin: CSTemperatureMarginIterNb3Sn = dataclasses.field(kw_only=True)
    """`.pf_coil.i_cs_superconductor` again -- constraint 60's
    `.pf_coil.temp_cs_superconductor_margin`.
    """

    stresses: CSCoilStresses = CSCoilStresses()
    """The CS's hoop/axial/radial stress state and its Tresca and von Mises combinations
    (`ohcalc`'s superconducting arm, `pfcoil.py:3398-3521`).
    """


class PFCoil(ModelNamespace):
    """`.tokamak.pf_coil` -- the PF coil set, thirteen slots including the inductance
    and the volt-second accounting.
    """

    placement: PFCoilPlacement = PFCoilPlacement()
    """The four groups' `(r, z)` positions (`pfcoil.py:236-364`) -- the occupant for the
    `(2, 2, 3, 3)` location pattern, which also fixes every array index in the package.
    """

    positions: PFCoilPositions = PFCoilPositions()
    """The group positions flattened to per-coil arrays (`pfcoil.py:1183-1234`)."""

    initiation_currents: PFCoilInitiationCurrents = PFCoilInitiationCurrents()
    """The plasma-initiation SVD solve (`pfcoil.py:366-405`)."""

    equilibrium_currents: PFCoilEquilibriumCurrents = PFCoilEquilibriumCurrents()
    """The equilibrium SVD solve and the required vertical field (`pfcoil.py:456-598`).
    """

    time_point_currents: PFCoilTimePointCurrents = PFCoilTimePointCurrents()
    """Per-coil currents at the pulse's time points (`pfcoil.py:663-728`)."""

    waveform: PFCoilCurrentWaveform = PFCoilCurrentWaveform()
    """Peak currents and the waveform fraction array (`pfcoil.py:1741-1748`,
    `waveform()`).
    """

    peak_field: PFCoilPeakField = PFCoilPeakField()
    """Peak field at each PF coil, per index `[0..5]` (`pfcoil.py:4444-4646`,
    `peak_b_field_at_pf_coil`).
    """

    sizes: PFCoilSizes = PFCoilSizes()
    """Coil cross-sections, turns and the whole-array coil extents (`pfcoil.py:730-849`)
    -- owns `.pf_coil.n_pf_coil_turns`, the cycle's other loop-carried variable.
    """

    masses: PFCoilMasses = PFCoilMasses()
    """Conductor and structure masses (`pfcoil.py:851-1023`)."""

    strand_critical_current: PFStrandCriticalCurrentDensity = (
        PFStrandCriticalCurrentDensity()
    )
    """`.pf_coil.j_crit_str_pf` -- the last PF coil's strand critical current density
    (`pfcoil.py:871-904`, `superconpf`'s NbTi arm).
    """

    inductance: PFCoilInductance = PFCoilInductance()
    """The 22x22 mutual-inductance matrix (`pfcoil.py:1750-2010`, `induct`), owned whole
    -- the node that enlarged the cycle from three to four (`inductance.md` § "The
    cycle, one node larger").
    """

    turn_currents: PFCoilTurnCurrents = PFCoilTurnCurrents()
    """Per-turn circuit currents at the six waveform time points (`pfcoil.py:1082-1111`)
    -- `.pf_coil.c_pf_coil_turn`, owned whole.
    """

    volt_seconds: PFCoilVoltSeconds = PFCoilVoltSeconds()
    """`PFCoil.vsec` (`pfcoil.py:1615-1720`), `iohcl = 1` arm -- the volt-second
    capability, `.pf_coil.vs_cs_pf_total_burn`/`.vs_cs_pf_total_pulse`.
    """


class PFCoilCsWstNb3Sn(PFCoil):
    """`.tokamak.pf_coil` for the `(i_pf_superconductor, i_cs_superconductor) = (3, 5)`
    pair -- NbTi PF coils, WST Nb3Sn CS, `low_aspect_ratio_DEMO.IN.DAT`'s pair (`:806`,
    `:845`), `indat._pf_coil_system_arm` arm `1`.
    """

    masses: PFCoilMassesCsWstNb3Sn = PFCoilMassesCsWstNb3Sn()
    """Conductor and structure masses (`pfcoil.py:851-1023`), CS conductor density from
    `.tfcoil.dcond[4]` (WST Nb3Sn) instead of `.tfcoil.dcond[0]`.
    """


class PFCoilSphericalTokamak(PFCoil):
    """`.tokamak.pf_coil` for a machine with **no central solenoid** --
    `indat._pf_coil_system_arm` arm 2, `spherical_tokamak_eval.IN.DAT` and
    `st_regression.IN.DAT`.
    """

    placement: PFCoilPlacementSphericalTokamak = PFCoilPlacementSphericalTokamak()
    """`i_pf_location = (2, 3, 3, 4)`, picture-frame TF, `rref` read
    (`pfcoil.py:245-352`).
    """

    positions: PFCoilPositionsNoCentralSolenoid = PFCoilPositionsNoCentralSolenoid()
    """The group positions flattened, with no CS slot to append (`:663-672`)."""

    initiation_currents: PFCoilInitiationCurrentsNoCentralSolenoid = (
        PFCoilInitiationCurrentsNoCentralSolenoid()
    )
    """The plasma-initiation SVD solve with `nfxf = 0` (`:366-405`, `:202-204`)."""

    equilibrium_currents: PFCoilEquilibriumCurrents = PFCoilEquilibriumCurrents(
        topology=SPHERICAL_TOKAMAK_TOPOLOGY
    )
    """The equilibrium SVD solve (`:456-598`)."""

    time_point_currents: PFCoilTimePointCurrentsNoCentralSolenoid = (
        PFCoilTimePointCurrentsNoCentralSolenoid()
    )
    """Per-coil currents at the pulse's time points (`:663-728`), and
    `.pf_coil.f_j_cs_start_end_flat_top = 1.0` (`:660`).
    """

    waveform: PFCoilCurrentWaveform = PFCoilCurrentWaveform(
        topology=SPHERICAL_TOKAMAK_TOPOLOGY
    )
    """Peak currents and the waveform fraction array over eight circuits plus the plasma
    (`waveform()`, `:2869-2940`).
    """

    peak_field: PFCoilPeakFieldNoCentralSolenoid = PFCoilPeakFieldNoCentralSolenoid()
    """Peak field at each of the eight PF coils, owned as two whole arrays
    (`peak_b_field_at_pf_coil` with `kk = 0`, `:4487-4489`).
    """

    sizes: PFCoilSizesNoCentralSolenoid = PFCoilSizesNoCentralSolenoid()
    """Coil cross-sections, turns and edges (`:730-849`), with the plasma one index
    further along and no CS slot between.
    """

    masses: PFCoilMassesNoCentralSolenoid = PFCoilMassesNoCentralSolenoid()
    """Conductor and structure masses (`:849-1026`), REBCO tape conductor density
    (`.tfcoil.dcond[8]`) and no CS steel or cable space.
    """

    strand_critical_current: PFStrandCriticalCurrentDensityHazeltonZhaiRebco = (
        PFStrandCriticalCurrentDensityHazeltonZhaiRebco()
    )
    """`.pf_coil.j_crit_str_pf` from `superconpf`'s `HAZELTON_ZHAI_REBCO` arm
    (`:4851-4866`), evaluated at the last of eight coils rather than the last of six.
    """

    inductance: PFCoilInductanceNoCentralSolenoid = PFCoilInductanceNoCentralSolenoid()
    """The mutual-inductance matrix (`induct`, `:1721-1984`) with its two CS blocks
    guarded out and `nef = n_cs_pf_coils`.
    """

    turn_currents: PFCoilTurnCurrents = PFCoilTurnCurrents(
        topology=SPHERICAL_TOKAMAK_TOPOLOGY
    )
    """Per-turn circuit currents at the six waveform time points (`:1082-1111`)."""

    volt_seconds: PFCoilVoltSecondsNoCentralSolenoid = (
        PFCoilVoltSecondsNoCentralSolenoid()
    )
    """`PFCoil.vsec` (`:1615-1720`), `iohcl = 0` arm -- the PF sums alone, with
    `vs_cs_ramp`/`vs_cs_burn` never assigned.
    """
