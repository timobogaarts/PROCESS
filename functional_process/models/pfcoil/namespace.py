"""The PF coil system's two namespaces: `.tokamak.cs_coil` and `.tokamak.pf_coil`.

Beside the nodes they name (`model_tree_design.md` §11): the fifteen node classes live
in this package's six modules (`geometry`, `currents`, `fields`, `masses`,
`inductance`, `volt_seconds`), and each class's own docstring already names the slot
it occupies here -- the split below is a transcription of those claims, not a new
decision.

**Two namespaces because PROCESS injects two models.** `Models.__init__`
(`process/main.py:652`) constructs `CSCoil` as its own `Model` with its own switch
(`i_cs_superconductor`) and injects it into `PFCoil`; `models/tokamak/namespace.py` gave
each a slot of its own on the same evidence, and a stellarator has neither
(`st_init` sets `.build.iohcl = 0` unconditionally).

**Every slot is an instance default, and that is not an exception to "a slot the
factory fills has no default".** The whole package is one occupant set for one joint
configuration -- `iohcl = 1`, `i_pf_location = (2, 2, 3, 3)`, `itart = itartpf = 0`,
`i_pf_current = 1`, superconducting conductors with the file's two materials, D-shaped
TF -- and `indat._pf_coil_system_arm` resolves that predicate **once**, refusing any
deviation before either namespace is constructed. One predicate, fifteen slots, the
`_fw_blkt_vv_shape_arm` shape ("one predicate, four slots") scaled up: inside a
namespace the factory only ever builds for the single supported arm, there is nothing
left for a slot to decide, and a per-slot `kw_only` factory would be fifteen
transcriptions of one answer. `noh = 30` (`inductance.NOH`) is the one occupant-fixing
integer the factory cannot see -- it is a step function of the *converged* CS geometry,
not of any input (`inductance.md` § "noh is a step function of the CS geometry") -- and
it stays a module constant on the occupant.

**The cycle is expected, and it grew on 2026-08-27.** `PFCoilTimePointCurrents ->
PFCoilCurrentWaveform -> PFCoilSizes -> PFCoilInductance -> CSFluxSwing ->
PFCoilTimePointCurrents` is Shape A (no node reads a `VarPath` it owns), left for
`Blocking` to find and `mda.CUTS` to cut -- see `currents.md` § "The cycle" and
`mda.py`. Registering `volt_seconds`/`turn_currents` merged that ring with the
volt-second/burn-time ring into one nine-node SCC (`volt_seconds.py`'s module
docstring carries the walk); the existing cuts still break it, measured in
`test_mda.py`.
"""

from cottax.interfaces.pytree_namespace_module import ModelNamespace

from functional_process.models.pfcoil.currents import (
    CSCurrentDensityPulseStart,
    CSFluxSwing,
    PFCoilEquilibriumCurrents,
    PFCoilInitiationCurrents,
    PFCoilTimePointCurrents,
)
from functional_process.models.pfcoil.fields import (
    PFCoilCurrentWaveform,
    PFCoilPeakField,
)
from functional_process.models.pfcoil.geometry import (
    CSCoilGeometry,
    PFCoilPlacement,
    PFCoilPositions,
)
from functional_process.models.pfcoil.inductance import PFCoilInductance
from functional_process.models.pfcoil.masses import PFCoilMasses, PFCoilSizes
from functional_process.models.pfcoil.volt_seconds import (
    PFCoilTurnCurrents,
    PFCoilVoltSeconds,
)


class CSCoil(ModelNamespace):
    """`.tokamak.cs_coil` -- the central solenoid, three slots.

    What is *not* here: the CS stress/fatigue chain (`ohcalc`'s `scipy.special`
    ellipk/ellipe calls and `cs_fatigue.ncycle`) is UNPORTED and `.tokamak.cs_fatigue`
    stays an empty slot; the CS's own self-field (`b_pf_coil_peak[6]`) is part of that
    chain, which is why `PFCoilPeakField` owns its arrays per index `[0..5]` only.
    """

    geometry: CSCoilGeometry = CSCoilGeometry()
    """The CS cross-section and its fourteen scaling filaments (`pfcoil.py:120-158`,
    `:202-234`)."""

    current_density_pulse_start: CSCurrentDensityPulseStart = (
        CSCurrentDensityPulseStart()
    )
    """`j_cs_pulse_start = j_cs_flat_top_end * f_j_cs_start_pulse_end_flat_top`
    (`pfcoil.py:161-164`)."""

    flux_swing: CSFluxSwing = CSFluxSwing()
    """The CS flux-swing balance (`pfcoil.py:600-657`) -- owns
    `.pf_coil.f_j_cs_start_end_flat_top`, and one of the two nodes whose reads make the
    package's four-node cycle (`.pf_coil.n_pf_coil_turns`,
    `.pf_coil.ind_pf_cs_plasma_mutual`)."""


class PFCoil(ModelNamespace):
    """`.tokamak.pf_coil` -- the PF coil set, twelve slots including the inductance
    and the volt-second accounting.

    Closes the three boundary reads the first tokamak wave left on `Structure`
    (`.pf_coil.m_pf_coil_conductor_total`, `.m_pf_coil_structure_total`) and `Cryostat`
    (`.pf_coil.r_pf_coil_outer_max`) -- `tokamak_boundary.md`'s pf_coil rows, now
    stale, said zero reads; `masses.md`'s open questions record the correction.
    """

    placement: PFCoilPlacement = PFCoilPlacement()
    """The four groups' `(r, z)` positions (`pfcoil.py:236-364`) -- the occupant for
    the `(2, 2, 3, 3)` location pattern, which also fixes every array index in the
    package."""

    positions: PFCoilPositions = PFCoilPositions()
    """The group positions flattened to per-coil arrays (`pfcoil.py:1183-1234`)."""

    initiation_currents: PFCoilInitiationCurrents = PFCoilInitiationCurrents()
    """The plasma-initiation SVD solve (`pfcoil.py:366-405`)."""

    equilibrium_currents: PFCoilEquilibriumCurrents = PFCoilEquilibriumCurrents()
    """The equilibrium SVD solve and the required vertical field
    (`pfcoil.py:456-598`)."""

    time_point_currents: PFCoilTimePointCurrents = PFCoilTimePointCurrents()
    """Per-coil currents at the pulse's time points (`pfcoil.py:663-728`)."""

    waveform: PFCoilCurrentWaveform = PFCoilCurrentWaveform()
    """Peak currents and the waveform fraction array (`pfcoil.py:1741-1748`,
    `waveform()`)."""

    peak_field: PFCoilPeakField = PFCoilPeakField()
    """Peak field at each PF coil, per index `[0..5]`
    (`pfcoil.py:4444-4646`, `peak_b_field_at_pf_coil`)."""

    sizes: PFCoilSizes = PFCoilSizes()
    """Coil cross-sections, turns and the whole-array coil extents
    (`pfcoil.py:730-849`) -- owns `.pf_coil.n_pf_coil_turns`, the cycle's other
    loop-carried variable."""

    masses: PFCoilMasses = PFCoilMasses()
    """Conductor and structure masses (`pfcoil.py:851-1023`)."""

    inductance: PFCoilInductance = PFCoilInductance()
    """The 22x22 mutual-inductance matrix (`pfcoil.py:1750-2010`, `induct`), owned
    whole -- the node that enlarged the cycle from three to four
    (`inductance.md` § "The cycle, one node larger")."""

    turn_currents: PFCoilTurnCurrents = PFCoilTurnCurrents()
    """Per-turn circuit currents at the six waveform time points
    (`pfcoil.py:1082-1111`) -- `.pf_coil.c_pf_coil_turn`, owned whole. Added
    2026-08-27, half of `cold_boundary.md` producer 4 (`volt_seconds.py`'s module
    docstring carries the scope and cycle argument)."""

    volt_seconds: PFCoilVoltSeconds = PFCoilVoltSeconds()
    """`PFCoil.vsec` (`pfcoil.py:1615-1720`), `iohcl = 1` arm -- the volt-second
    capability, `.pf_coil.vs_cs_pf_total_burn`/`.vs_cs_pf_total_pulse`. Added
    2026-08-27 (`cold_boundary.md` producer 4): the burn value was the boundary zero
    that, with `.physics.res_plasma`, made the cold burn time nan. Registering it
    merges the PF coil ring with the volt-second/burn-time ring into one nine-node
    SCC; `mda.CUTS`'s existing three entries still cut it (measured,
    `test_mda.py`)."""
