"""The PF coil system's two namespaces: `.tokamak.cs_coil` and `.tokamak.pf_coil`."""

import dataclasses

import equinox as eqx
from cottax.interfaces import Explicit

from functional_process.cottax.models.pfcoil.currents import (
    CSCurrentDensityPulseStart,
    CSFluxSwing,
    PFCoilEquilibriumCurrents,
    PFCoilEquilibriumCurrentsNoCentralSolenoid,
    PFCoilInitiationCurrents,
    PFCoilInitiationCurrentsNoCentralSolenoid,
    PFCoilTimePointCurrents,
    PFCoilTimePointCurrentsNoCentralSolenoid,
)
from functional_process.cottax.models.pfcoil.fields import (
    CSCoilPeakField,
    PFCoilCurrentWaveform,
    PFCoilCurrentWaveformNoCentralSolenoid,
    PFCoilPeakField,
    PFCoilPeakFieldNoCentralSolenoid,
)
from functional_process.cottax.models.pfcoil.geometry import (
    CSCoilGeometry,
    CSCoilTurnGeometry,
    PFCoilPlacement,
    PFCoilPlacementSphericalTokamak,
    PFCoilPositions,
    PFCoilPositionsNoCentralSolenoid,
)
from functional_process.cottax.models.pfcoil.inductance import (
    PFCoilInductance,
    PFCoilInductanceNoCentralSolenoid,
)
from functional_process.cottax.models.pfcoil.masses import (
    PFCoilMasses,
    PFCoilMassesNoCentralSolenoid,
    PFCoilSizes,
    PFCoilSizesNoCentralSolenoid,
)
from functional_process.cottax.models.pfcoil.stresses import CSCoilStresses
from functional_process.cottax.models.pfcoil.superconductor import (
    PFStrandCriticalCurrentDensity,
    PFStrandCriticalCurrentDensityHazeltonZhaiRebco,
)
from functional_process.cottax.models.pfcoil.volt_seconds import (
    PFCoilTurnCurrents,
    PFCoilTurnCurrentsNoCentralSolenoid,
    PFCoilVoltSeconds,
    PFCoilVoltSecondsNoCentralSolenoid,
)


class CSCoil(eqx.Module):
    """`.tokamak.cs_coil` -- the central solenoid, seven slots."""

    geometry: Explicit = CSCoilGeometry
    """The CS cross-section and its fourteen scaling filaments (`pfcoil.py:120-158`,
    `:202-234`).
    """

    turn_geometry: Explicit = CSCoilTurnGeometry
    """The EU DEMO stadium-shaped CS turn and its steel conduit (`ohcalc`,
    `pfcoil.py:3296-3319`, via `calculate_cs_turn_geometry_eu_demo`).
    """

    current_density_pulse_start: Explicit = CSCurrentDensityPulseStart
    """`j_cs_pulse_start = j_cs_flat_top_end * f_j_cs_start_pulse_end_flat_top`
    (`pfcoil.py:161-164`).
    """

    flux_swing: Explicit = CSFluxSwing
    """The CS flux-swing balance (`pfcoil.py:600-657`) -- owns
    `.pf_coil.f_j_cs_start_end_flat_top`, and one of the two nodes whose reads make the
    package's four-node cycle (`.pf_coil.n_pf_coil_turns`,
    `.pf_coil.ind_pf_cs_plasma_mutual`).
    """

    peak_field: Explicit = CSCoilPeakField
    """The CS's own peak field at end-of-flat-top and beginning-of-pulse (`ohcalc`'s
    field block, `pfcoil.py:3327-3396`), plus index `[6]` of the two whole-array peak
    fields.
    """

    critical_current: Explicit = dataclasses.field(kw_only=True)
    """`.pf_coil.i_cs_superconductor` -- two reachable values, both written."""

    temperature_margin: Explicit = dataclasses.field(kw_only=True)
    """`.pf_coil.i_cs_superconductor` again -- constraint 60's
    `.pf_coil.temp_cs_superconductor_margin`.
    """

    stresses: Explicit = CSCoilStresses
    """The CS's hoop/axial/radial stress state and its Tresca and von Mises combinations
    (`ohcalc`'s superconducting arm, `pfcoil.py:3398-3521`).
    """


class PFCoil(eqx.Module):
    """`.tokamak.pf_coil` -- the PF coil set, thirteen slots including the inductance
    and the volt-second accounting.
    """

    placement: Explicit = PFCoilPlacement
    """The four groups' `(r, z)` positions (`pfcoil.py:236-364`) -- the occupant for the
    `(2, 2, 3, 3)` location pattern, which also fixes every array index in the package.
    """

    positions: Explicit = PFCoilPositions
    """The group positions flattened to per-coil arrays (`pfcoil.py:1183-1234`)."""

    initiation_currents: Explicit = PFCoilInitiationCurrents
    """The plasma-initiation SVD solve (`pfcoil.py:366-405`)."""

    equilibrium_currents: Explicit = PFCoilEquilibriumCurrents
    """The equilibrium SVD solve and the required vertical field (`pfcoil.py:456-598`).
    """

    time_point_currents: Explicit = PFCoilTimePointCurrents
    """Per-coil currents at the pulse's time points (`pfcoil.py:663-728`)."""

    waveform: Explicit = PFCoilCurrentWaveform
    """Peak currents and the waveform fraction array (`pfcoil.py:1741-1748`,
    `waveform()`).
    """

    peak_field: Explicit = PFCoilPeakField
    """Peak field at each PF coil, per index `[0..5]` (`pfcoil.py:4444-4646`,
    `peak_b_field_at_pf_coil`).
    """

    sizes: Explicit = PFCoilSizes
    """Coil cross-sections, turns and the whole-array coil extents (`pfcoil.py:730-849`)
    -- owns `.pf_coil.n_pf_coil_turns`, the cycle's other loop-carried variable.
    """

    masses: Explicit = PFCoilMasses
    """Conductor and structure masses (`pfcoil.py:851-1023`)."""

    strand_critical_current: Explicit = PFStrandCriticalCurrentDensity
    """`.pf_coil.j_crit_str_pf` -- the last PF coil's strand critical current density
    (`pfcoil.py:871-904`, `superconpf`'s NbTi arm).
    """

    inductance: Explicit = PFCoilInductance
    """The 22x22 mutual-inductance matrix (`pfcoil.py:1750-2010`, `induct`), owned whole
    -- the node that enlarged the cycle from three to four (`inductance.md` § "The
    cycle, one node larger").
    """

    turn_currents: Explicit = PFCoilTurnCurrents
    """Per-turn circuit currents at the six waveform time points (`pfcoil.py:1082-1111`)
    -- `.pf_coil.c_pf_coil_turn`, owned whole.
    """

    volt_seconds: Explicit = PFCoilVoltSeconds
    """`PFCoil.vsec` (`pfcoil.py:1615-1720`), `iohcl = 1` arm -- the volt-second
    capability, `.pf_coil.vs_cs_pf_total_burn`/`.vs_cs_pf_total_pulse`.
    """


class PFCoilSphericalTokamak(PFCoil):
    """`.tokamak.pf_coil` for a machine with **no central solenoid** --
    `indat._pf_coil_system_arm` arm 2, `spherical_tokamak_eval.IN.DAT` and
    `st_regression.IN.DAT`.
    """

    placement: Explicit = PFCoilPlacementSphericalTokamak
    """`i_pf_location = (2, 3, 3, 4)`, picture-frame TF, `rref` read
    (`pfcoil.py:245-352`).
    """

    positions: Explicit = PFCoilPositionsNoCentralSolenoid
    """The group positions flattened, with no CS slot to append (`:663-672`)."""

    initiation_currents: Explicit = PFCoilInitiationCurrentsNoCentralSolenoid
    """The plasma-initiation SVD solve with `nfxf = 0` (`:366-405`, `:202-204`)."""

    equilibrium_currents: Explicit = PFCoilEquilibriumCurrentsNoCentralSolenoid
    """The equilibrium SVD solve (`:456-598`)."""

    time_point_currents: Explicit = PFCoilTimePointCurrentsNoCentralSolenoid
    """Per-coil currents at the pulse's time points (`:663-728`), and
    `.pf_coil.f_j_cs_start_end_flat_top = 1.0` (`:660`).
    """

    waveform: Explicit = PFCoilCurrentWaveformNoCentralSolenoid
    """Peak currents and the waveform fraction array over eight circuits plus the plasma
    (`waveform()`, `:2869-2940`).
    """

    peak_field: Explicit = PFCoilPeakFieldNoCentralSolenoid
    """Peak field at each of the eight PF coils, owned as two whole arrays
    (`peak_b_field_at_pf_coil` with `kk = 0`, `:4487-4489`).
    """

    sizes: Explicit = PFCoilSizesNoCentralSolenoid
    """Coil cross-sections, turns and edges (`:730-849`), with the plasma one index
    further along and no CS slot between.
    """

    masses: Explicit = PFCoilMassesNoCentralSolenoid
    """Conductor and structure masses (`:849-1026`), with no CS steel or cable space."""

    strand_critical_current: Explicit = PFStrandCriticalCurrentDensityHazeltonZhaiRebco
    """`.pf_coil.j_crit_str_pf` from `superconpf`'s `HAZELTON_ZHAI_REBCO` arm
    (`:4851-4866`), evaluated at the last of eight coils rather than the last of six.
    """

    inductance: Explicit = PFCoilInductanceNoCentralSolenoid
    """The mutual-inductance matrix (`induct`, `:1721-1984`) with its two CS blocks
    guarded out and `nef = n_cs_pf_coils`.
    """

    turn_currents: Explicit = PFCoilTurnCurrentsNoCentralSolenoid
    """Per-turn circuit currents at the six waveform time points (`:1082-1111`)."""

    volt_seconds: Explicit = PFCoilVoltSecondsNoCentralSolenoid
    """`PFCoil.vsec` (`:1615-1720`), `iohcl = 0` arm -- the PF sums alone, with
    `vs_cs_ramp`/`vs_cs_burn` never assigned.
    """
