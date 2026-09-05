"""Harness cases for the ported pulse-duration sums (registry unit #6).

`st_init` is a bare function of `data: DataStructure` (no `Stellarator` instance
needed) -- the reference adapter only has to set `istell` nonzero (to pass the
whole-function gate) and the six duration fields the port reads.
"""

from functional_process._harness import Tier1Contract, legacy_sample
from functional_process.cottax.stellarator.initialization import (
    calculate_pulse_durations,
)
from process.core.model import DataStructure
from process.models.stellarator.initialization import st_init


def _reference_pulse_durations(
    t_plant_pulse_coil_precharge,
    t_plant_pulse_plasma_current_ramp_up,
    t_plant_pulse_burn,
    t_plant_pulse_plasma_current_ramp_down,
    t_plant_pulse_fusion_ramp,
    t_plant_pulse_dwell,
):
    """Call PROCESS's `st_init` and read back its three summed duration writes.

    `st_init` overwrites `t_plant_pulse_coil_precharge`/`_ramp_up`/`_burn`/`_ramp_down`
    itself (device-preset literals, see the audit record) rather than reading them, so
    those four sample arguments are set on `data` only for documentation of the port's
    signature -- they are not actually consulted by `st_init`. `t_plant_pulse_fusion_ramp`
    and `t_plant_pulse_dwell` genuinely are read, so those two are load-bearing.
    """
    data = DataStructure()
    data.stellarator.istell = 1
    data.times.t_plant_pulse_fusion_ramp = t_plant_pulse_fusion_ramp
    data.times.t_plant_pulse_dwell = t_plant_pulse_dwell

    st_init(data)

    return (
        data.times.t_plant_pulse_plasma_present,
        data.times.t_plant_pulse_no_burn,
        data.times.t_plant_pulse_total,
    )


class TestPulseDurations(Tier1Contract):
    """`st_init`'s pulse-duration sums -> `calculate_pulse_durations`."""

    audit_record = "models/stellarator/initialization.md"
    reference = _reference_pulse_durations
    ported = calculate_pulse_durations

    # `st_init` hardcodes these four unconditionally (see the port's module docstring)
    # -- the *reference* adapter's output cannot respond to them at all, so a gradient
    # check against them would compare the port's real (nonzero) derivative to the
    # reference's structural zero. Excluded from differentiation for that reason, not
    # because they're switches; value agreement still holds (both sides use the same
    # literal, see `fuzz_fixed` below).
    static_argnames = (
        "t_plant_pulse_coil_precharge",
        "t_plant_pulse_plasma_current_ramp_up",
        "t_plant_pulse_burn",
        "t_plant_pulse_plasma_current_ramp_down",
    )

    # PROCESS's own defaults (times_variables.py) and st_init's own literals
    # (initialization.py) -- a real stellarator-mode operating point.
    samples = [
        legacy_sample(
            "st-init-defaults",
            t_plant_pulse_coil_precharge=0.0,
            t_plant_pulse_plasma_current_ramp_up=0.0,
            t_plant_pulse_burn=3.15576e7,
            t_plant_pulse_plasma_current_ramp_down=0.0,
            t_plant_pulse_fusion_ramp=10.0,
            t_plant_pulse_dwell=1800.0,
        ),
    ]

    # Only these two are actually read by `st_init` -- the other four are overwritten
    # unconditionally by its own literals regardless of what the reference adapter sets
    # on `data` beforehand (see the adapter's docstring), so fuzzing them would compare
    # the port against a reference that silently ignores the fuzzed value. `fuzz_fixed`
    # pins the other four to `st_init`'s own literals so the comparison stays honest.
    fuzz_bounds = {
        "t_plant_pulse_fusion_ramp": (0.0, 1.0e3),
        "t_plant_pulse_dwell": (0.0, 1.0e4),
    }
    fuzz_fixed = {
        "t_plant_pulse_coil_precharge": 0.0,
        "t_plant_pulse_plasma_current_ramp_up": 0.0,
        "t_plant_pulse_burn": 3.15576e7,
        "t_plant_pulse_plasma_current_ramp_down": 0.0,
    }
