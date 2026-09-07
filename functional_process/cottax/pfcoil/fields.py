"""Poloidal field from a set of circular current loops, and the peak field at each PF
coil's inner/outer edge.
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
    """cottax node: `.tokamak.pf_coil.peak_field`."""

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
    """cottax node: `.tokamak.pf_coil.peak_field`, the `iohcl = 0` occupant."""

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
    """cottax node: `.tokamak.pf_coil.waveform`."""

    topology: PFCoilTopology = eqx.field(static=True, default=REFERENCE_TOPOLOGY)
    """Static."""

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
    """cottax node: `.tokamak.cs_coil.peak_field`."""

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
