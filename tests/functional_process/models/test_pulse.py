"""Harness case for the ported subset of `process/models/pulse.py` (`.tokamak.pulse`).

Audit record: `functional_process/_audit/units/models/pulse.md`. One unit in scope:
`calculate_burn_time` (`Pulse.calculate_burn_time`, `process/models/pulse.py:275-316`),
tier-1, the sole occupant of `PulseBurnTime`. `tohswg`
(`.constraints.t_current_ramp_up_min`) is deliberately not ported -- see the port's
module docstring and the audit record's "Not ported" section.

`calculate_burn_time` is already a bare `@staticmethod` with no `self.data` access
(same shape as `confinement_time.py`'s 48 scaling laws), so it is called directly as
the reference -- no `DataStructure` adapter needed.
"""

from functional_process._harness import Tier1Contract, legacy_sample
from functional_process.models.pulse import calculate_burn_time
from process.models.pulse import Pulse


class TestCalculateBurnTime(Tier1Contract):
    """`calculate_burn_time` -> `Pulse.calculate_burn_time`, unchanged (module
    docstring's dropped `logger.error` call aside -- a diagnostic with no effect on
    the returned value).
    """

    audit_record = "models/pulse.md"
    reference = staticmethod(Pulse.calculate_burn_time)
    ported = calculate_burn_time

    # tests/unit/models/test_pulse.py::test_calculate_burn_time_valid, verbatim --
    # already-validated input points from PROCESS's own unit test, including the
    # negative-vs_cs_pf_total_burn case exercising `abs()`.
    samples = [
        legacy_sample(
            "unit-test-nominal",
            vs_cs_pf_total_burn=100.0,
            v_plasma_loop_burn=10.0,
            t_plant_pulse_fusion_ramp=2.0,
        ),
        legacy_sample(
            "unit-test-negative-vs",
            vs_cs_pf_total_burn=-100.0,
            v_plasma_loop_burn=10.0,
            t_plant_pulse_fusion_ramp=2.0,
        ),
        legacy_sample(
            "unit-test-zero-fusion-ramp",
            vs_cs_pf_total_burn=50.0,
            v_plasma_loop_burn=5.0,
            t_plant_pulse_fusion_ramp=0.0,
        ),
    ]

    # Physically reasonable domains, not PROCESS-declared iteration-variable bounds --
    # none of the three arguments is an iteration variable
    # (`bounds_from_iteration_variables` has nothing to look up for them).
    # `v_plasma_loop_burn` order-of-magnitude from `tests/unit/models/physics/
    # test_physics.py`'s `voltsecondreqparam` (~0.04 V); kept away from exactly 0.0,
    # where the function has a genuine, PROCESS-shared division-by-zero (not a porting
    # defect -- `test_outputs_finite`/`test_gradient_finite_at_zero` treat a value that
    # goes non-finite at the boundary as out of scope, not a failure, per
    # `_harness/contracts.py`).
    fuzz_bounds = {
        "vs_cs_pf_total_burn": (-200.0, 200.0),
        "v_plasma_loop_burn": (0.01, 0.5),
        "t_plant_pulse_fusion_ramp": (0.0, 10000.0),  # process/core/input.py:788
    }
