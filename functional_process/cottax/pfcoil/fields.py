"""Poloidal field from a set of circular current loops, and the peak field at each PF
coil's inner/outer edge.

Audit record: `functional_process/_audit/units/models/pfcoil/fields.md`.

Two units:

- `calculate_b_field_at_point` -- a direct port of PROCESS's `@numba.njit` kernel
  (`process/models/pfcoil.py:4926-5060`), vectorised over the loops instead of looped.
  It is the whole numerical content of `efc`'s matrix assembly *and* of the peak-field
  calculation, so it is ported once and reused by `currents.py`.
- `calculate_pf_coil_peak_fields` -- `peak_b_field_at_pf_coil`
  (`pfcoil.py:4414-4638`) specialised to the reference run's coil topology, for the four
  PF groups. It **folds in** `PFCoil.waveform` (`:2869-2940`); see below.

**Why `waveform` is folded in.** `peak_b_field_at_pf_coil` reads
`.pf_coil.c_pf_cs_coils_peak_ma` and `.pf_coil.f_c_pf_cs_peak_time_array` and then
*re-derives* which of the three time points the peak came from, by asking which of
`c_pf_cs_coil_pulse_start_ma`/`_flat_top_ma`/`_pulse_end_ma` equals it to within
`1e-12` -- raising `ProcessValueError` if none does (`pfcoil.py:4484-4487`). Both of
those fields are pure functions of the same three arrays (that is all `waveform` is), so
taking the three arrays as this unit's inputs and deriving the peak internally
**narrows** the declared read set rather than inventing an edge, and it makes the
precondition structurally unreachable instead of a data-dependent raise: `c_peak` is a
bitwise copy of one of the three, so exactly one of the three comparisons is an exact
equality. A port that took `c_peak` as a free input would be outside PROCESS's domain at
essentially every fuzz point, which is a validated *nothing*, not a validated port.

**A PROCESS defect ported faithfully.** CS current filaments 0 and 1 do not hold CS
positions on this arm. `pfcoil()`'s equilibrium branch overwrites
`r/z_pf_cs_current_filaments[0:2]` with the group-0 and group-1 PF coil positions
(`pfcoil.py:474-479`, `nocoil` starting from 0) *after* `place_cs_filaments` has filled
all 14, and `peak_b_field_at_pf_coil` then rewrites only the 14 *currents*
(`:4500-4509`), never the positions. Two of the fourteen CS filaments therefore sit at
the PF coils' radii and heights while carrying a CS filament's current. Verified against
a live traced run, not inferred: the filament arrays handed to
`calculate_b_field_at_point` on `large_tokamak_eval.IN.DAT` begin
`r = [5.5667, 5.5667, 2.2773, ...]`, `z = [9.6443, -10.8782, 2.8344, ...]`. Reproduced
here exactly; see `fields.md` § "A PROCESS defect ported faithfully".
"""

import equinox as eqx
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    Output,
    OutputInto,
)

from functional_process.cottax.pfcoil import (
    CS_INDEX,
    REFERENCE_TOPOLOGY,
    SPHERICAL_TOKAMAK_TOPOLOGY,
    PFCoilTopology,
)
from functional_process.cottax.paths import pf_coil, physics
from functional_process.models.pfcoil.fields import (
    calculate_b_field_at_point,  # noqa: F401 -- re-exported for inductance.py / tests
    calculate_coil_current_waveform,  # noqa: F401 -- re-exported for tests
    calculate_coil_current_waveform_for_topology,
    calculate_cs_bore_magnetic_field,  # noqa: F401 -- re-exported for tests
    calculate_cs_peak_fields,  # noqa: F401 -- re-exported for tests
    calculate_cs_peak_fields_reference_widths,
    calculate_cs_self_peak_magnetic_field,  # noqa: F401 -- re-exported for tests
    calculate_pf_coil_peak_fields,  # noqa: F401 -- re-exported for tests
    calculate_pf_coil_peak_fields_no_central_solenoid,  # noqa: F401 -- re-exported for tests
    calculate_pf_coil_peak_fields_no_central_solenoid_for_topology,
    calculate_pf_coil_peak_fields_reference_arm,
)


class PFCoilPeakField(ExplicitFunction):
    """cottax node: `.tokamak.pf_coil.peak_field`.

    Owns the peak field at the inner and outer edge of each of the six PF coils, as six
    per-index `Output`s each -- **not** the whole `NGC2`-wide array. Index 6 (the CS) is
    written by `CSCoil.ohcalc` from the CS's own self-field
    (`calculate_cs_self_peak_magnetic_field`, `pfcoil.py:3331-3342`), which is UNPORTED
    on this pass because nothing in the mass closure reads it; owning the whole array
    would claim a value for index 6 that this node does not compute. Per-index
    addressing is `_audit/naming_convention.md` § "Array elements", and follows
    `models/physics/composition.py`'s precedent.

    Occupant for `iohcl = 1`, `i_pf_current = 1`, `itart = 0` and
    `i_pf_location = (2, 2, 3, 3)` -- the reference arm. Other arms are UNPORTED; see
    `fields.md`.
    """

    b_pf_coil_peak_0 = Output(pf_coil.b_pf_coil_peak[0])
    b_pf_coil_peak_1 = Output(pf_coil.b_pf_coil_peak[1])
    b_pf_coil_peak_2 = Output(pf_coil.b_pf_coil_peak[2])
    b_pf_coil_peak_3 = Output(pf_coil.b_pf_coil_peak[3])
    b_pf_coil_peak_4 = Output(pf_coil.b_pf_coil_peak[4])
    b_pf_coil_peak_5 = Output(pf_coil.b_pf_coil_peak[5])
    bpf2_0 = Output(pf_coil.bpf2[0])
    bpf2_1 = Output(pf_coil.bpf2[1])
    bpf2_2 = Output(pf_coil.bpf2[2])
    bpf2_3 = Output(pf_coil.bpf2[3])
    bpf2_4 = Output(pf_coil.bpf2[4])
    bpf2_5 = Output(pf_coil.bpf2[5])

    def __call__(
        self,
        c_pf_cs_coil_pulse_start_ma=From(pf_coil),
        c_pf_cs_coil_flat_top_ma=From(pf_coil),
        c_pf_cs_coil_pulse_end_ma=From(pf_coil),
        r_pf_coil_middle=From(pf_coil),
        z_pf_coil_middle=From(pf_coil),
        r_pf_coil_inner=From(pf_coil),
        r_pf_coil_outer=From(pf_coil),
        z_pf_coil_upper=From(pf_coil),
        z_pf_coil_lower=From(pf_coil),
        r_pf_coil_middle_group_array=From(pf_coil),
        z_pf_coil_middle_group_array=From(pf_coil),
        r_cs_middle=From(pf_coil),
        dz_cs_full=From(pf_coil),
        a_cs_poloidal=From(pf_coil),
        j_cs_pulse_start=From(pf_coil),
        j_cs_flat_top_end=From(pf_coil),
        rmajor=From(physics),
        plasma_current=From(physics),
    ):
        return calculate_pf_coil_peak_fields_reference_arm(
            c_pf_cs_coil_pulse_start_ma=c_pf_cs_coil_pulse_start_ma,
            c_pf_cs_coil_flat_top_ma=c_pf_cs_coil_flat_top_ma,
            c_pf_cs_coil_pulse_end_ma=c_pf_cs_coil_pulse_end_ma,
            r_pf_coil_middle=r_pf_coil_middle,
            z_pf_coil_middle=z_pf_coil_middle,
            r_pf_coil_inner=r_pf_coil_inner,
            r_pf_coil_outer=r_pf_coil_outer,
            z_pf_coil_upper=z_pf_coil_upper,
            z_pf_coil_lower=z_pf_coil_lower,
            r_pf_coil_middle_group_array=r_pf_coil_middle_group_array,
            z_pf_coil_middle_group_array=z_pf_coil_middle_group_array,
            r_cs_middle=r_cs_middle,
            dz_cs_full=dz_cs_full,
            a_cs_poloidal=a_cs_poloidal,
            j_cs_pulse_start=j_cs_pulse_start,
            j_cs_flat_top_end=j_cs_flat_top_end,
            rmajor=rmajor,
            plasma_current=plasma_current,
        )


class PFCoilPeakFieldNoCentralSolenoid(ExplicitFunction):
    """cottax node: `.tokamak.pf_coil.peak_field`, the `iohcl = 0` occupant.

    **Owns `.pf_coil.b_pf_coil_peak` and `.pf_coil.bpf2` whole**, where
    `PFCoilPeakField` owns six slots of each -- and that difference is the point rather
    than a convenience. Per-index ownership exists on the conventional arm because index
    6 belongs to `CSCoil.ohcalc`'s own self-field node (`CSCoilPeakField`). With
    `iohcl = 0` that node does not exist, `ohcalc` is never entered
    (`pfcoil.py:1048-1050`), and the group loop writes every slot PROCESS writes -- so
    there is no slice of either array this node does not compute, which is exactly
    `_audit/naming_convention.md` § "Array elements"' test for owning one whole.

    Occupant for `i_pf_current = 1`, `not (itart == 1 and itartpf == 0)` and
    `i_pf_location = (2, 3, 3, 4)`.
    """

    topology: PFCoilTopology = eqx.field(static=True, default=SPHERICAL_TOKAMAK_TOPOLOGY)

    b_pf_coil_peak = OutputInto(pf_coil)
    bpf2 = OutputInto(pf_coil)

    def __call__(
        self,
        c_pf_cs_coil_pulse_start_ma=From(pf_coil),
        c_pf_cs_coil_flat_top_ma=From(pf_coil),
        c_pf_cs_coil_pulse_end_ma=From(pf_coil),
        r_pf_coil_middle=From(pf_coil),
        z_pf_coil_middle=From(pf_coil),
        r_pf_coil_inner=From(pf_coil),
        r_pf_coil_outer=From(pf_coil),
        z_pf_coil_upper=From(pf_coil),
        z_pf_coil_lower=From(pf_coil),
        rmajor=From(physics),
        plasma_current=From(physics),
    ):
        return calculate_pf_coil_peak_fields_no_central_solenoid_for_topology(
            c_pf_cs_coil_pulse_start_ma=c_pf_cs_coil_pulse_start_ma,
            c_pf_cs_coil_flat_top_ma=c_pf_cs_coil_flat_top_ma,
            c_pf_cs_coil_pulse_end_ma=c_pf_cs_coil_pulse_end_ma,
            r_pf_coil_middle=r_pf_coil_middle,
            z_pf_coil_middle=z_pf_coil_middle,
            r_pf_coil_inner=r_pf_coil_inner,
            r_pf_coil_outer=r_pf_coil_outer,
            z_pf_coil_upper=z_pf_coil_upper,
            z_pf_coil_lower=z_pf_coil_lower,
            rmajor=rmajor,
            plasma_current=plasma_current,
            topology=self.topology,
        )


class PFCoilCurrentWaveform(ExplicitFunction):
    """cottax node: `.tokamak.pf_coil.waveform`. Ports `PFCoil.waveform`
    (`process/models/pfcoil.py:2869-2940`) as a node in its own right.

    `PFCoilPeakField` above derives the same two quantities internally rather than
    reading them (module docstring), so this node exists for the *other* readers of
    `.pf_coil.c_pf_cs_coils_peak_ma` -- `masses.py`'s sizing and mass chains, `outpf`,
    `vsec`, `induct` -- not for that one. The duplicated arithmetic is three lines and
    the alternative is a data-dependent precondition no fuzz point can satisfy.

    Owns the two arrays at their full `NGC2` width. The CS entry (index 6) is written
    twice by PROCESS -- here, and again by `ohcalc` (`:3264-3281`) -- with values that
    are algebraically identical (`c_cs_flat_top_end = -a_cs_poloidal *
    j_cs_flat_top_end` makes `waveform`'s end-of-flat-top pick equal `ohcalc`'s
    `sgn * 1e-6 * j_cs_flat_top_end * a_cs_poloidal` whenever that is the largest of the
    three, which it is on this arm), so there is no ownership question to resolve.
    The plasma row (index 7) is set to all ones by `waveform` (`:2877-2879`); reproduced
    here so the owned array matches PROCESS's state.
    """

    topology: PFCoilTopology = eqx.field(static=True, default=REFERENCE_TOPOLOGY)
    """Static. How many circuits `waveform`'s loop covers, and which row is the plasma's.
    The same node serves both topologies: `waveform` reads the three current arrays
    whole and branches on nothing, so the read set does not move with the topology --
    contrast `peak_field`, whose `iohcl = 0` arm drops five reads."""

    c_pf_cs_coils_peak_ma = OutputInto(pf_coil)
    f_c_pf_cs_peak_time_array = OutputInto(pf_coil)

    def __call__(
        self,
        c_pf_cs_coil_pulse_start_ma=From(pf_coil),
        c_pf_cs_coil_flat_top_ma=From(pf_coil),
        c_pf_cs_coil_pulse_end_ma=From(pf_coil),
    ):
        return calculate_coil_current_waveform_for_topology(
            c_pf_cs_coil_pulse_start_ma=c_pf_cs_coil_pulse_start_ma,
            c_pf_cs_coil_flat_top_ma=c_pf_cs_coil_flat_top_ma,
            c_pf_cs_coil_pulse_end_ma=c_pf_cs_coil_pulse_end_ma,
            topology=self.topology,
        )


# ---------------------------------------------------------------------------
# The Central Solenoid's own peak field -- `ohcalc`'s field block
# (`process/models/pfcoil.py:3327-3396`).
#
# Added 2026-08-27 for `optimise_design.md` §11.5: `.pf_coil.b_cs_peak_flat_top_end`
# and `.b_cs_peak_pulse_start` are what the CS critical-current and stress chains read,
# and both were boundary zeros against PROCESS's converged 14.041 / 13.978 T. This is
# the "CS's own self-field" `namespace.CSCoil` recorded as UNPORTED and the reason
# `PFCoilPeakField` owns its arrays per index `[0..5]` only.
# ---------------------------------------------------------------------------


class CSCoilPeakField(ExplicitFunction):
    """cottax node: `.tokamak.cs_coil.peak_field`.

    Owns `.pf_coil.b_cs_peak_flat_top_end`, `.b_cs_peak_pulse_start`,
    `.b_cs_self_outer_midplane` and index `[6]` of the two whole-array peak fields --
    the two slots `PFCoilPeakField` deliberately leaves alone (see its docstring).
    Per-index `Output`s for the same reason it uses them.

    **`.pf_coil.b_cs_self_outer_midplane` is owned even though it is a literal zero.**
    `ohcalc:3360` assigns `0.0` unconditionally ("self-field is assumed to be zero --
    long solenoid approximation"), and owning it says that the port computes PROCESS's
    answer rather than leaving a field to a `DataStructure` default that happens to
    match. Same call `CentrepostNeutronicsAbsent` and `HcdSecondaryHeating` make; the
    `0.0 * b_cs_peak_flat_top_end` in the pure function is what keeps the output a
    traced array of the right shape rather than a Python float.

    Occupant for `iohcl = 1` with the package's single supported topology, exactly as
    every other node here.
    """

    b_cs_peak_flat_top_end = OutputInto(pf_coil)
    b_cs_peak_pulse_start = OutputInto(pf_coil)
    b_cs_self_outer_midplane = OutputInto(pf_coil)
    b_pf_coil_peak_cs = Output(pf_coil.b_pf_coil_peak[CS_INDEX])
    bpf2_cs = Output(pf_coil.bpf2[CS_INDEX])

    def __call__(
        self,
        c_pf_cs_coil_pulse_start_ma=From(pf_coil),
        c_pf_cs_coil_flat_top_ma=From(pf_coil),
        c_pf_cs_coil_pulse_end_ma=From(pf_coil),
        r_pf_coil_middle=From(pf_coil),
        z_pf_coil_middle=From(pf_coil),
        r_cs_inner=From(pf_coil),
        r_cs_outer=From(pf_coil),
        z_cs_middle=From(pf_coil),
        z_cs_upper=From(pf_coil),
        j_cs_flat_top_end=From(pf_coil),
        j_cs_pulse_start=From(pf_coil),
        rmajor=From(physics),
        plasma_current=From(physics),
    ):
        return calculate_cs_peak_fields_reference_widths(
            c_pf_cs_coil_pulse_start_ma=c_pf_cs_coil_pulse_start_ma,
            c_pf_cs_coil_flat_top_ma=c_pf_cs_coil_flat_top_ma,
            c_pf_cs_coil_pulse_end_ma=c_pf_cs_coil_pulse_end_ma,
            r_pf_coil_middle=r_pf_coil_middle,
            z_pf_coil_middle=z_pf_coil_middle,
            r_cs_inner=r_cs_inner,
            r_cs_outer=r_cs_outer,
            z_cs_middle=z_cs_middle,
            z_cs_upper=z_cs_upper,
            j_cs_flat_top_end=j_cs_flat_top_end,
            j_cs_pulse_start=j_cs_pulse_start,
            rmajor=rmajor,
            plasma_current=plasma_current,
        )
