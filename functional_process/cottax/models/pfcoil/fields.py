"""Poloidal field from a set of circular current loops, and the peak field at each PF
coil's inner/outer edge.
"""

from cottax.cx import V

from functional_process.cottax.paths import pf_coil
from functional_process.models.pfcoil.fields import (
    calculate_b_field_at_point,  # noqa: F401 -- re-exported for inductance.py / tests
    calculate_coil_current_waveform,  # noqa: F401 -- re-exported for tests
    calculate_coil_current_waveform_for_topology,  # noqa: F401 -- re-exported for tests
    calculate_coil_current_waveform_no_central_solenoid,
    calculate_coil_current_waveform_reference,
    calculate_cs_bore_magnetic_field,  # noqa: F401 -- re-exported for tests
    calculate_cs_peak_fields,  # noqa: F401 -- re-exported for tests
    calculate_cs_peak_fields_reference_widths,
    calculate_cs_self_peak_magnetic_field,  # noqa: F401 -- re-exported for tests
    calculate_pf_coil_peak_fields,  # noqa: F401 -- re-exported for tests
    calculate_pf_coil_peak_fields_no_central_solenoid,  # noqa: F401 -- re-exported for tests
    calculate_pf_coil_peak_fields_no_central_solenoid_bound,
    calculate_pf_coil_peak_fields_no_central_solenoid_for_topology,  # noqa: F401 -- re-exported for tests
    calculate_pf_coil_peak_fields_reference_arm,
)

# cottax node: `.tokamak.pf_coil.peak_field`.
PFCoilPeakField = pf_coil.function(
    calculate_pf_coil_peak_fields_reference_arm,
    V.pf_coil.b_pf_coil_peak[0],
    V.pf_coil.b_pf_coil_peak[1],
    V.pf_coil.b_pf_coil_peak[2],
    V.pf_coil.b_pf_coil_peak[3],
    V.pf_coil.b_pf_coil_peak[4],
    V.pf_coil.b_pf_coil_peak[5],
    V.pf_coil.bpf2[0],
    V.pf_coil.bpf2[1],
    V.pf_coil.bpf2[2],
    V.pf_coil.bpf2[3],
    V.pf_coil.bpf2[4],
    V.pf_coil.bpf2[5],
)


# cottax node: `.tokamak.pf_coil.peak_field`, the `iohcl = 0` occupant.
PFCoilPeakFieldNoCentralSolenoid = pf_coil.function(
    calculate_pf_coil_peak_fields_no_central_solenoid_bound,
    "b_pf_coil_peak",
    "bpf2",
)


# cottax node: `.tokamak.pf_coil.waveform`.
PFCoilCurrentWaveform = pf_coil.function(
    calculate_coil_current_waveform_reference,
    "c_pf_cs_coils_peak_ma",
    "f_c_pf_cs_peak_time_array",
)


# cottax node: `.tokamak.pf_coil.waveform`, the `iohcl = 0` occupant.
#
# Same reads and outputs as `PFCoilCurrentWaveform` -- only `fn`'s baked topology
# differs.
PFCoilCurrentWaveformNoCentralSolenoid = pf_coil.function(
    calculate_coil_current_waveform_no_central_solenoid,
    "c_pf_cs_coils_peak_ma",
    "f_c_pf_cs_peak_time_array",
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


# cottax node: `.tokamak.cs_coil.peak_field`.
CSCoilPeakField = pf_coil.function(
    calculate_cs_peak_fields_reference_widths,
    "b_cs_peak_flat_top_end",
    "b_cs_peak_pulse_start",
    "b_cs_self_outer_midplane",
    V.pf_coil.b_pf_coil_peak[6],
    V.pf_coil.bpf2[6],
)
