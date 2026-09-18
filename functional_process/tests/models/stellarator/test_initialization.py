"""Harness cases for the ported pulse-duration sums (registry unit #6).

`st_init` is a bare function of `data: DataStructure` (no `Stellarator` instance
needed) -- the reference adapter only has to set `istell` nonzero (to pass the
whole-function gate) and the six duration fields the port reads.
"""

from functional_process.tests._harness import Tier1Contract
from functional_process.tests._harness.process_reference import data_reference
from functional_process.tests._harness.sample_store import FROM_FILE
from functional_process.cottax.models.stellarator.initialization import (
    calculate_pulse_durations,
)
from process.models.stellarator.initialization import st_init


def _call_pulse_durations(data):
    """Call PROCESS's `st_init` and read back its three summed duration writes.

    `st_init` overwrites `t_plant_pulse_coil_precharge`/`_ramp_up`/`_burn`/`_ramp_down`
    itself (device-preset literals, see the audit record) regardless of what is poked
    onto `data` beforehand, so those four sample arguments have no effect on the
    output -- only `t_plant_pulse_fusion_ramp` and `t_plant_pulse_dwell` are genuinely
    read and load-bearing.
    """
    data.stellarator.istell = 1
    st_init(data)
    return (
        data.times.t_plant_pulse_plasma_present,
        data.times.t_plant_pulse_no_burn,
        data.times.t_plant_pulse_total,
    )


_reference_pulse_durations = data_reference(_call_pulse_durations)


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
    samples = FROM_FILE

    # Only these two are actually read by `st_init` -- the other four are overwritten
    # unconditionally by its own literals regardless of what the reference adapter sets
    # on `data` beforehand (see the adapter's docstring), so fuzzing them would compare
    # the port against a reference that silently ignores the fuzzed value. `fuzz_fixed`
    # pins the other four to `st_init`'s own literals so the comparison stays honest.
    fuzz = True
    fuzz_fixed = {
        "t_plant_pulse_coil_precharge": 0.0,
        "t_plant_pulse_plasma_current_ramp_up": 0.0,
        "t_plant_pulse_burn": 3.15576e7,
        "t_plant_pulse_plasma_current_ramp_down": 0.0,
    }
