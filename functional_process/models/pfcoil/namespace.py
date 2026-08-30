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

**Every slot but one is an instance default, and that is not an exception to "a slot the
factory fills has no default".** The whole package is one occupant set per joint
configuration -- `iohcl = 1`, `i_pf_location = (2, 2, 3, 3)`, `itart = itartpf = 0`,
`i_pf_current = 1`, superconducting conductors, D-shaped TF -- and
`indat._pf_coil_system_arm` resolves that predicate **once**, refusing any deviation
before either namespace is constructed. Two superconductor pairs are ported, and the
second (`PFCoilCsWstNb3Sn` below) differs from the first in exactly one slot occupant;
everything else in both namespaces is the single supported arm's answer. One
predicate, eighteen slots, the `_fw_blkt_vv_shape_arm` shape ("one predicate, four
slots") scaled up: inside a namespace the factory only ever builds for one resolved
arm, there is nothing left for a slot to decide, and a per-slot `kw_only` factory
would be eighteen transcriptions of one answer.

The one exception is `CSCoil.critical_current`, added 2026-08-27, and it is worth its
own sentence because it looks like a contradiction and is not: that slot's switch
(`.pf_coil.i_cs_superconductor`) *is* part of the joint predicate, but the predicate's
arm `1` accepts two of its values (`1` and `5`) and those two need different
critical-surface fits. One predicate answers "which `.tfcoil.dcond` element"; it does
not answer "which critical surface". See that slot and `indat.CS_SUPERCONDUCTOR`.

`noh = 30` (`inductance.NOH`) is the one occupant-fixing integer the factory cannot
see -- a step function of the *converged* CS geometry, not of any input
(`inductance.md` § "noh is a step function of the CS geometry") -- and it stays a
module constant on the occupant.

**The cycle is expected, and it grew on 2026-08-27.** `PFCoilTimePointCurrents ->
PFCoilCurrentWaveform -> PFCoilSizes -> PFCoilInductance -> CSFluxSwing ->
PFCoilTimePointCurrents` is Shape A (no node reads a `VarPath` it owns), left for
`Blocking` to find and `mda.CUTS` to cut -- see `currents.md` § "The cycle" and
`mda.py`. Registering `volt_seconds`/`turn_currents` merged that ring with the
volt-second/burn-time ring into one nine-node SCC (`volt_seconds.py`'s module
docstring carries the walk); the existing cuts still break it, measured in
`test_mda.py`.
"""

import dataclasses

from cottax.interfaces.pytree_namespace_module import ModelNamespace

from functional_process.models.pfcoil import SPHERICAL_TOKAMAK_TOPOLOGY
from functional_process.models.pfcoil.currents import (
    CSCurrentDensityPulseStart,
    CSFluxSwing,
    PFCoilEquilibriumCurrents,
    PFCoilInitiationCurrents,
    PFCoilInitiationCurrentsNoCentralSolenoid,
    PFCoilTimePointCurrents,
    PFCoilTimePointCurrentsNoCentralSolenoid,
)
from functional_process.models.pfcoil.fields import (
    CSCoilPeakField,
    PFCoilCurrentWaveform,
    PFCoilPeakField,
    PFCoilPeakFieldNoCentralSolenoid,
)
from functional_process.models.pfcoil.geometry import (
    CSCoilGeometry,
    CSCoilTurnGeometry,
    PFCoilPlacement,
    PFCoilPlacementSphericalTokamak,
    PFCoilPositions,
    PFCoilPositionsNoCentralSolenoid,
)
from functional_process.models.pfcoil.inductance import (
    PFCoilInductance,
    PFCoilInductanceNoCentralSolenoid,
)
from functional_process.models.pfcoil.masses import (
    PFCoilMasses,
    PFCoilMassesCsWstNb3Sn,
    PFCoilMassesNoCentralSolenoid,
    PFCoilSizes,
    PFCoilSizesNoCentralSolenoid,
)
from functional_process.models.pfcoil.stresses import CSCoilStresses
from functional_process.models.pfcoil.superconductor import (
    CSCriticalCurrentDensitiesIterNb3Sn,
    CSTemperatureMarginIterNb3Sn,
    PFStrandCriticalCurrentDensity,
    PFStrandCriticalCurrentDensityHazeltonZhaiRebco,
)
from functional_process.models.pfcoil.volt_seconds import (
    PFCoilTurnCurrents,
    PFCoilVoltSeconds,
    PFCoilVoltSecondsNoCentralSolenoid,
)


class CSCoil(ModelNamespace):
    """`.tokamak.cs_coil` -- the central solenoid, seven slots.

    Three until 2026-08-27. `optimise_design.md` §11.5 found four constraints (26, 27,
    60, 72) reading fields whose only producer is `ohcalc`, and the three slots added
    then -- `peak_field`, `critical_current`, `stresses` -- were most of that closure.
    `temperature_margin` (2026-08-30) is the rest of it.

    **This paragraph used to say the CS chain was UNPORTED, and named two reasons.** One
    is gone: `stresses.py` replaces `ohcalc`'s `scipy.special` ellipk/ellipe with the
    AGM, which is traceable, differentiable and agrees with scipy to 1-2 ulp. The second
    is gone too, as of 2026-08-30: `.pf_coil.temp_cs_superconductor_margin` (constraint
    60) was a boundary zero against PROCESS's `3.4208032` K and now has a producer --
    `superconpf`'s `scipy.optimize.newton` secant solve, driven by the TF coil's own
    replica of it (`solve_current_sharing_temperature`), which is the shared driver
    `superconductor.py` said the deferral was waiting for. What stands is
    `cs_fatigue.ncycle`: still unported, `.tokamak.cs_fatigue` still an empty slot, no
    active constraint reading it.

    The CS's own self-field is no longer part of any of that: `peak_field` owns
    `b_pf_coil_peak[6]`/`bpf2[6]`, the two slots `PFCoilPeakField` leaves alone.
    """

    geometry: CSCoilGeometry = CSCoilGeometry()
    """The CS cross-section and its fourteen scaling filaments (`pfcoil.py:120-158`,
    `:202-234`)."""

    turn_geometry: CSCoilTurnGeometry = CSCoilTurnGeometry()
    """The EU DEMO stadium-shaped CS turn and its steel conduit
    (`ohcalc`, `pfcoil.py:3296-3319`, via `calculate_cs_turn_geometry_eu_demo`).

    Added 2026-08-30, with `.tokamak.cs_fatigue`. It owns
    `.cs_fatigue.dr_cs_turn_conduit`/`.dz_cs_turn_conduit`, which are `ncycle`'s two
    crack-size limits -- `pfcoil_variables.py` gives them input defaults of `0.07`/
    `0.022` and PROCESS computes `0.0099` for both on `low_aspect_ratio_DEMO`, so
    landing `CsFatigue` without this slot would have swapped a wrong `n_cycle` of zero
    for a wrong `n_cycle` that looked plausible. Unswitched: `calculate_cs_turn_
    geometry_eu_demo` has no branch but its own 1 mm floor, and PROCESS has no second
    turn model to choose between."""

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

    peak_field: CSCoilPeakField = CSCoilPeakField()
    """The CS's own peak field at end-of-flat-top and beginning-of-pulse
    (`ohcalc`'s field block, `pfcoil.py:3327-3396`), plus index `[6]` of the two
    whole-array peak fields.

    Added 2026-08-27 (`optimise_design.md` §11.5). This is the head of the CS chain
    this namespace's docstring recorded as UNPORTED: every one of constraints 26, 27,
    60 and 72 reads something downstream of these two fields, and both were boundary
    zeros against PROCESS's converged 14.041 / 13.978 T."""

    critical_current: CSCriticalCurrentDensitiesIterNb3Sn = dataclasses.field(
        kw_only=True
    )
    """`.pf_coil.i_cs_superconductor` -- two reachable values, both written.

    **The one factory-filled slot in this package, and the exception proves this
    namespace's own rule.** Everything else here is an instance default because
    `indat._pf_coil_system_arm` resolves the whole joint configuration once, before
    either namespace is built -- and `i_cs_superconductor` is *part of* that predicate,
    which is how `PFCoilCsWstNb3Sn` exists at all. But that predicate answers a
    different question (which `.tfcoil.dcond` element the masses read) and its arm `1`
    covers `i_cs_superconductor = 5` as well as `1`. Those two values need different
    critical-surface fits, so this slot has to ask the switch again, on its own.

    `indat.CS_SUPERCONDUCTOR` holds the registry and is **total** -- the six other values
    `superconpf` dispatches on are refused by `_pf_coil_system_arm` before this slot
    exists, so it carries no `UNPORTED` entries at all.

    Added 2026-08-27 (`optimise_design.md` §11.5), constraints 26 and 27."""

    temperature_margin: CSTemperatureMarginIterNb3Sn = dataclasses.field(kw_only=True)
    """`.pf_coil.i_cs_superconductor` again -- constraint 60's
    `.pf_coil.temp_cs_superconductor_margin`.

    The second factory-filled slot, filled from the same registry
    (`indat.CS_TEMPERATURE_MARGIN`, keyed on the same two reachable values) and for the
    same reason as `critical_current`: `_pf_coil_system_arm`'s arm `1` spans both, so
    the fit has to be chosen by asking the switch again.

    Separate from `critical_current` rather than a fifth output of it: the two come out
    of the same `superconpf` calls but not the same read set -- only this one needs
    `.pf_coil.c_pf_cs_coils_peak_ma`. The TF coil splits its own two the same way.

    Added 2026-08-30 as a missing producer measured on `large_tokamak_nof` (`0.0`
    against PROCESS's `3.4208032` K)."""

    stresses: CSCoilStresses = CSCoilStresses()
    """The CS's hoop/axial/radial stress state and its Tresca and von Mises
    combinations (`ohcalc`'s superconducting arm, `pfcoil.py:3398-3521`).

    Unswitched here, so a default: `i_pf_conductor` is already part of
    `_pf_coil_system_arm`'s conjunction (arm `-5` refuses anything but
    `SUPERCONDUCTING`), and PROCESS's resistive `else` computes no stresses at all, so
    there is no second occupant for a factory to choose between.

    Added 2026-08-27, constraint 72's `.pf_coil.stress_shear_cs_peak` -- and this is the
    node that closes the `scipy.special` ellipk/ellipe blocker this namespace's own
    docstring named. See `stresses.py` for why its elliptic integrals are the AGM while
    `fields.py`'s are Abramowitz & Stegun fits."""


class PFCoil(ModelNamespace):
    """`.tokamak.pf_coil` -- the PF coil set, thirteen slots including the inductance
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

    strand_critical_current: PFStrandCriticalCurrentDensity = (
        PFStrandCriticalCurrentDensity()
    )
    """`.pf_coil.j_crit_str_pf` -- the last PF coil's strand critical current density
    (`pfcoil.py:871-904`, `superconpf`'s NbTi arm).

    Added 2026-08-30, and the prerequisite for `.costs.pf_magnet_cost`: Account 222.2's
    `PER_KAM` arm reads this field, PROCESS computes `1.1018e9` A/m^2 for it on
    `large_tokamak_nof`, and nothing owned it -- so registering the account without this
    slot would have moved a missing-producer row rather than closed one
    (`_audit/cost_boundary_inputs.md` §13.2).

    An instance default rather than a factory-filled slot, unlike its two `CSCoil`
    namesakes: `.pf_coil.i_pf_superconductor` is pinned to `3` by
    `_pf_coil_system_deviations`' `-6` on **both** positive arms, so there is one
    critical surface and nothing to choose. `PFCoilCsWstNb3Sn` inherits it unchanged for
    the same reason."""

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


class PFCoilCsWstNb3Sn(PFCoil):
    """`.tokamak.pf_coil` for the `(i_pf_superconductor, i_cs_superconductor) = (3, 5)`
    pair -- NbTi PF coils, WST Nb3Sn CS, `low_aspect_ratio_DEMO.IN.DAT`'s pair
    (`:806`, `:845`), `indat._pf_coil_system_arm` arm `1`.

    One slot re-occupied, twelve inherited: the superconductor switches' only effect in
    the ported closure is which `.tfcoil.dcond` element `masses` reads (`masses.md`
    § switches touched), so the masses occupant is the entire difference. The slot is a
    **place** -- the node bound at `.tokamak.pf_coil.masses` keeps its name whichever
    occupant fills it (`ModelNamespace`'s own contract).
    """

    masses: PFCoilMassesCsWstNb3Sn = PFCoilMassesCsWstNb3Sn()
    """Conductor and structure masses (`pfcoil.py:851-1023`), CS conductor density
    from `.tfcoil.dcond[4]` (WST Nb3Sn) instead of `.tfcoil.dcond[0]`."""


class PFCoilSphericalTokamak(PFCoil):
    """`.tokamak.pf_coil` for a machine with **no central solenoid** --
    `indat._pf_coil_system_arm` arm 2, `spherical_tokamak_eval.IN.DAT` and
    `st_regression.IN.DAT`.

    Thirteen slots re-occupied and none inherited, which is the honest count of what
    `iohcl = 0` and `i_pf_location = (2, 3, 3, 4)` change together. They split into two
    kinds, and the distinction is the one this port keeps making:

    **Eight occupants declare different reads.** `placement` gains `.pf_coil.rref` (the
    `i_pf_location = 4` group); `positions`, `initiation_currents`, `sizes`, `masses`,
    `inductance` and `time_point_currents` each *drop* the CS fields they would
    otherwise read, because with `ohcalc` never entered (`pfcoil.py:1048-1050`) those
    fields have no producer at all; `peak_field` drops five and owns two whole arrays
    instead of twelve slots of them; `strand_critical_current` swaps NbTi for REBCO tape
    and gains the tape's three dimensions. A different read set is a different node --
    a node that read `.pf_coil.a_cs_poloidal` on a machine with no solenoid would be
    reading a boundary zero and calling it a cross-section.

    **Five carry the same body with the eight-coil `PFCoilTopology`.**
    `equilibrium_currents`, `waveform`, `turn_currents` and `volt_seconds` read the same
    fields on both machines -- `pfcoil()` reads those arrays whole and branches on
    nothing -- so what changes is only which slots the loops cover, which is static
    graph-assembly data (`pfcoil/__init__.py`).

    **`flux_swing` has no counterpart here**, and it is not a slot this class leaves
    empty: it lives in `.tokamak.cs_coil`, which arm 2 makes `None` outright. Its one
    output, `.pf_coil.f_j_cs_start_end_flat_top`, is produced by
    `time_point_currents` instead -- `pfcoil.py:658-661` assigns the constant `1.0` on
    this arm, over a storage default of `0.0`, so it is a value that must be produced
    and not one absence can supply. That also means the package's four-node cycle
    (`currents.py`'s module docstring) is **broken on this machine**: with `flux_swing`
    gone the ring `sizes -> flux_swing -> time_point_currents` has no middle. Whether
    the remaining edges still close one is `Blocking`'s question.
    """

    placement: PFCoilPlacementSphericalTokamak = PFCoilPlacementSphericalTokamak()
    """`i_pf_location = (2, 3, 3, 4)`, picture-frame TF, `rref` read
    (`pfcoil.py:245-352`)."""

    positions: PFCoilPositionsNoCentralSolenoid = PFCoilPositionsNoCentralSolenoid()
    """The group positions flattened, with no CS slot to append (`:663-672`)."""

    initiation_currents: PFCoilInitiationCurrentsNoCentralSolenoid = (
        PFCoilInitiationCurrentsNoCentralSolenoid()
    )
    """The plasma-initiation SVD solve with `nfxf = 0` (`:366-405`, `:202-204`)."""

    equilibrium_currents: PFCoilEquilibriumCurrents = PFCoilEquilibriumCurrents(
        topology=SPHERICAL_TOKAMAK_TOPOLOGY
    )
    """The equilibrium SVD solve (`:456-598`). One fixed-current group (the pair above
    the TF) and three solved-for (the two outside it and the generally-placed pair),
    against the conventional arm's two and two."""

    time_point_currents: PFCoilTimePointCurrentsNoCentralSolenoid = (
        PFCoilTimePointCurrentsNoCentralSolenoid()
    )
    """Per-coil currents at the pulse's time points (`:663-728`), and
    `.pf_coil.f_j_cs_start_end_flat_top = 1.0` (`:660`)."""

    waveform: PFCoilCurrentWaveform = PFCoilCurrentWaveform(
        topology=SPHERICAL_TOKAMAK_TOPOLOGY
    )
    """Peak currents and the waveform fraction array over eight circuits plus the
    plasma (`waveform()`, `:2869-2940`)."""

    peak_field: PFCoilPeakFieldNoCentralSolenoid = PFCoilPeakFieldNoCentralSolenoid()
    """Peak field at each of the eight PF coils, owned as two whole arrays
    (`peak_b_field_at_pf_coil` with `kk = 0`, `:4487-4489`)."""

    sizes: PFCoilSizesNoCentralSolenoid = PFCoilSizesNoCentralSolenoid()
    """Coil cross-sections, turns and edges (`:730-849`), with the plasma one index
    further along and no CS slot between."""

    masses: PFCoilMassesNoCentralSolenoid = PFCoilMassesNoCentralSolenoid()
    """Conductor and structure masses (`:849-1026`), REBCO tape conductor density
    (`.tfcoil.dcond[8]`) and no CS steel or cable space."""

    strand_critical_current: PFStrandCriticalCurrentDensityHazeltonZhaiRebco = (
        PFStrandCriticalCurrentDensityHazeltonZhaiRebco()
    )
    """`.pf_coil.j_crit_str_pf` from `superconpf`'s `HAZELTON_ZHAI_REBCO` arm
    (`:4851-4866`), evaluated at the last of eight coils rather than the last of six."""

    inductance: PFCoilInductanceNoCentralSolenoid = PFCoilInductanceNoCentralSolenoid()
    """The mutual-inductance matrix (`induct`, `:1721-1984`) with its two CS blocks
    guarded out and `nef = n_cs_pf_coils`."""

    turn_currents: PFCoilTurnCurrents = PFCoilTurnCurrents(
        topology=SPHERICAL_TOKAMAK_TOPOLOGY
    )
    """Per-turn circuit currents at the six waveform time points (`:1082-1111`)."""

    volt_seconds: PFCoilVoltSecondsNoCentralSolenoid = (
        PFCoilVoltSecondsNoCentralSolenoid()
    )
    """`PFCoil.vsec` (`:1615-1720`), `iohcl = 0` arm -- the PF sums alone, with
    `vs_cs_ramp`/`vs_cs_burn` never assigned."""
