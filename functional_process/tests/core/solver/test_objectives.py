"""Harness cases for the ported objective-function metrics
(`FiguresOfMerit` ids 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19).

`_reference_*` adapters bind a bare `DataStructure` with only the fields each metric's
audited data footprint says it reads, then call PROCESS's own `objective_function`
with `i_figure_merit` set to the *positive* id --
`objective_sign = np.sign(i_figure_merit)` is then `+1`, so PROCESS's return value is
exactly the unsigned `objective_metric` this
port's own `objective_metric_<id>` functions compute (see `objectives.py`'s module
docstring: sign is applied by the caller, not folded into these functions).
"""

import pytest

from functional_process.cottax._harness import Tier1Contract, legacy_sample
from functional_process.cottax.core.solver.objectives import (
    objective_metric_1,
    objective_metric_3,
    objective_metric_4,
    objective_metric_5,
    objective_metric_6,
    objective_metric_7,
    objective_metric_8,
    objective_metric_9,
    objective_metric_10,
    objective_metric_11,
    objective_metric_14,
    objective_metric_15,
    objective_metric_16,
    objective_metric_17,
    objective_metric_18,
    objective_metric_19,
)
from process.core.model import DataStructure
from process.core.solver.objectives import objective_function
from process.models.availability import AvailabilityModel


def _reference_1(rmajor):
    data = DataStructure()
    data.physics.rmajor = rmajor
    return objective_function(1, data)


def _reference_3(pflux_fw_neutron_mw):
    data = DataStructure()
    data.physics.pflux_fw_neutron_mw = pflux_fw_neutron_mw
    return objective_function(3, data)


def _reference_4(tfcmw, srcktpm):
    data = DataStructure()
    data.tfcoil.tfcmw = tfcmw
    data.pf_power.srcktpm = srcktpm
    return objective_function(4, data)


def _reference_5(big_q_plasma):
    data = DataStructure()
    data.current_drive.big_q_plasma = big_q_plasma
    return objective_function(5, data)


def _reference_6(coe):
    data = DataStructure()
    data.costs.coe = coe
    return objective_function(6, data)


def _reference_7(cdirt, concost, ireactor):
    data = DataStructure()
    data.costs.cdirt = cdirt
    data.costs.concost = concost
    data.costs.ireactor = ireactor
    return objective_function(7, data)


def _reference_8(aspect):
    data = DataStructure()
    data.physics.aspect = aspect
    return objective_function(8, data)


def _reference_9(pflux_div_heat_load_mw):
    data = DataStructure()
    data.divertor.pflux_div_heat_load_mw = pflux_div_heat_load_mw
    return objective_function(9, data)


def _reference_10(b_plasma_toroidal_on_axis):
    data = DataStructure()
    data.physics.b_plasma_toroidal_on_axis = b_plasma_toroidal_on_axis
    return objective_function(10, data)


def _reference_11(p_hcd_injected_total_mw):
    data = DataStructure()
    data.current_drive.p_hcd_injected_total_mw = p_hcd_injected_total_mw
    return objective_function(11, data)


def _reference_14(t_plant_pulse_burn):
    data = DataStructure()
    data.times.t_plant_pulse_burn = t_plant_pulse_burn
    return objective_function(14, data)


def _reference_15(i_plant_availability, f_t_plant_available):
    data = DataStructure()
    data.costs.i_plant_availability = i_plant_availability
    data.costs.f_t_plant_available = f_t_plant_available
    return objective_function(15, data)


def _reference_16(rmajor, t_plant_pulse_burn):
    data = DataStructure()
    data.physics.rmajor = rmajor
    data.times.t_plant_pulse_burn = t_plant_pulse_burn
    return objective_function(16, data)


def _reference_17(p_plant_electric_net_mw):
    data = DataStructure()
    data.heat_transport.p_plant_electric_net_mw = p_plant_electric_net_mw
    return objective_function(17, data)


def _reference_18():
    data = DataStructure()
    return objective_function(18, data)


def _reference_19(big_q_plasma, t_plant_pulse_burn):
    data = DataStructure()
    data.current_drive.big_q_plasma = big_q_plasma
    data.times.t_plant_pulse_burn = t_plant_pulse_burn
    return objective_function(19, data)


class TestObjectiveMetric1(Tier1Contract):
    """`objective_function(1, ...)` -> `objective_metric_1`. `MAJOR_RADIUS`."""

    audit_record = "core/solver/objectives.md"
    reference = _reference_1
    ported = objective_metric_1

    samples = [legacy_sample("nominal", rmajor=9.0)]
    fuzz_bounds = {"rmajor": (1.0, 20.0)}


class TestObjectiveMetric3(Tier1Contract):
    """`objective_function(3, ...)` -> `objective_metric_3`. `NEUTRON_WALL_LOAD`."""

    audit_record = "core/solver/objectives.md"
    reference = _reference_3
    ported = objective_metric_3

    samples = [legacy_sample("nominal", pflux_fw_neutron_mw=1.5)]
    fuzz_bounds = {"pflux_fw_neutron_mw": (0.0, 10.0)}


class TestObjectiveMetric4(Tier1Contract):
    """`objective_function(4, ...)` -> `objective_metric_4`. `P_TF_PLUS_P_PF`."""

    audit_record = "core/solver/objectives.md"
    reference = _reference_4
    ported = objective_metric_4

    samples = [legacy_sample("nominal", tfcmw=50.0, srcktpm=2000.0)]
    fuzz_bounds = {"tfcmw": (0.0, 200.0), "srcktpm": (0.0, 10000.0)}


class TestObjectiveMetric5(Tier1Contract):
    """`objective_function(5, ...)` -> `objective_metric_5`. `FUSION_GAIN_Q`."""

    audit_record = "core/solver/objectives.md"
    reference = _reference_5
    ported = objective_metric_5

    samples = [legacy_sample("nominal", big_q_plasma=15.0)]
    fuzz_bounds = {"big_q_plasma": (0.1, 50.0)}


class TestObjectiveMetric6(Tier1Contract):
    """`objective_function(6, ...)` -> `objective_metric_6`. `COST_OF_ELECTRICITY`."""

    audit_record = "core/solver/objectives.md"
    reference = _reference_6
    ported = objective_metric_6

    samples = [legacy_sample("nominal", coe=80.0)]
    fuzz_bounds = {"coe": (10.0, 500.0)}


class TestObjectiveMetric7(Tier1Contract):
    """`objective_function(7, ...)` -> `objective_metric_7`. `CAPITAL_COST`."""

    audit_record = "core/solver/objectives.md"
    reference = _reference_7
    ported = objective_metric_7

    static_argnames = ("ireactor",)

    samples = [
        legacy_sample("direct-cost", cdirt=5000.0, concost=8000.0, ireactor=0),
        legacy_sample("construction-cost", cdirt=5000.0, concost=8000.0, ireactor=1),
    ]
    fuzz_bounds = {"cdirt": (100.0, 20000.0), "concost": (100.0, 30000.0)}
    fuzz_fixed = {"ireactor": 0}


class TestObjectiveMetric8(Tier1Contract):
    """`objective_function(8, ...)` -> `objective_metric_8`. `ASPECT_RATIO`."""

    audit_record = "core/solver/objectives.md"
    reference = _reference_8
    ported = objective_metric_8

    samples = [legacy_sample("nominal", aspect=10.0)]
    fuzz_bounds = {"aspect": (2.0, 20.0)}


class TestObjectiveMetric9(Tier1Contract):
    """`objective_function(9, ...)` -> `objective_metric_9`. `DIVERTOR_HEAT_LOAD`."""

    audit_record = "core/solver/objectives.md"
    reference = _reference_9
    ported = objective_metric_9

    samples = [legacy_sample("nominal", pflux_div_heat_load_mw=5.0)]
    fuzz_bounds = {"pflux_div_heat_load_mw": (0.0, 20.0)}


class TestObjectiveMetric10(Tier1Contract):
    """`objective_function(10, ...)` -> `objective_metric_10`. `TOROIDAL_FIELD`."""

    audit_record = "core/solver/objectives.md"
    reference = _reference_10
    ported = objective_metric_10

    samples = [legacy_sample("nominal", b_plasma_toroidal_on_axis=5.0)]
    fuzz_bounds = {"b_plasma_toroidal_on_axis": (1.0, 15.0)}


class TestObjectiveMetric11(Tier1Contract):
    """`objective_function(11, ...)` -> `objective_metric_11`. `TOTAL_INJECTED_POWER`."""

    audit_record = "core/solver/objectives.md"
    reference = _reference_11
    ported = objective_metric_11

    samples = [legacy_sample("nominal", p_hcd_injected_total_mw=50.0)]
    fuzz_bounds = {"p_hcd_injected_total_mw": (0.0, 200.0)}


class TestObjectiveMetric14(Tier1Contract):
    """`objective_function(14, ...)` -> `objective_metric_14`. `PULSE_LENGTH`."""

    audit_record = "core/solver/objectives.md"
    reference = _reference_14
    ported = objective_metric_14

    samples = [legacy_sample("nominal", t_plant_pulse_burn=7200.0)]
    fuzz_bounds = {"t_plant_pulse_burn": (0.0, 50000.0)}


class TestObjectiveMetric15(Tier1Contract):
    """`objective_function(15, ...)` -> `objective_metric_15`.
    `PLANT_AVAILABILITY_FACTOR`.

    Only non-`USER_INPUT` `i_plant_availability` values are sampled -- the `USER_INPUT`
    case is a precondition failure, not a value/gradient-agreement case, see
    `test_objective_metric_15_raises_on_user_input_availability` below.
    """

    audit_record = "core/solver/objectives.md"
    reference = _reference_15
    ported = objective_metric_15

    static_argnames = ("i_plant_availability",)

    samples = [
        legacy_sample(
            "ward-taylor",
            i_plant_availability=int(AvailabilityModel.WARD_TAYLOR),
            f_t_plant_available=0.8,
        ),
        legacy_sample(
            "morris",
            i_plant_availability=int(AvailabilityModel.MORRIS),
            f_t_plant_available=0.85,
        ),
    ]
    fuzz_bounds = {"f_t_plant_available": (0.5, 1.0)}
    fuzz_fixed = {"i_plant_availability": int(AvailabilityModel.WARD_TAYLOR)}


def test_objective_metric_15_raises_on_user_input_availability():
    """`i_plant_availability == AvailabilityModel.USER_INPUT` has no model output to
    optimise against -- both PROCESS and this port raise rather than silently
    returning the raw user input. Confirms the port's `ValueError` fires on exactly
    the same condition as PROCESS's real `ProcessValueError`.
    """
    data = DataStructure()
    data.costs.i_plant_availability = int(AvailabilityModel.USER_INPUT)
    data.costs.f_t_plant_available = 0.75
    with pytest.raises(Exception, match="user input"):
        objective_function(15, data)

    with pytest.raises(ValueError, match="USER_INPUT"):
        objective_metric_15(int(AvailabilityModel.USER_INPUT), 0.75)


class TestObjectiveMetric16(Tier1Contract):
    """`objective_function(16, ...)` -> `objective_metric_16`. `MIN_R0_MAX_TAU_BURN`."""

    audit_record = "core/solver/objectives.md"
    reference = _reference_16
    ported = objective_metric_16

    samples = [legacy_sample("nominal", rmajor=9.0, t_plant_pulse_burn=7200.0)]
    fuzz_bounds = {"rmajor": (1.0, 20.0), "t_plant_pulse_burn": (0.0, 50000.0)}


class TestObjectiveMetric17(Tier1Contract):
    """`objective_function(17, ...)` -> `objective_metric_17`.
    `NET_ELECTRICAL_OUTPUT`.
    """

    audit_record = "core/solver/objectives.md"
    reference = _reference_17
    ported = objective_metric_17

    samples = [legacy_sample("nominal", p_plant_electric_net_mw=500.0)]
    fuzz_bounds = {"p_plant_electric_net_mw": (0.0, 2000.0)}


class TestObjectiveMetric18(Tier1Contract):
    """`objective_function(18, ...)` -> `objective_metric_18`. `NULL_FIGURE_OF_MERIT`.

    No arguments -- a single sample is the entire domain.
    """

    audit_record = "core/solver/objectives.md"
    reference = _reference_18
    ported = objective_metric_18

    samples = [legacy_sample("nominal")]


class TestObjectiveMetric19(Tier1Contract):
    """`objective_function(19, ...)` -> `objective_metric_19`.
    `MAX_Q_MAX_T_PLANT_PULSE_BURN` -- **not** the same formula as id 16, despite
    `objective_function`'s own inline docstring listing both as "major radius/burn
    time" (see `objectives.py`'s module docstring for the discrepancy).
    """

    audit_record = "core/solver/objectives.md"
    reference = _reference_19
    ported = objective_metric_19

    samples = [legacy_sample("nominal", big_q_plasma=15.0, t_plant_pulse_burn=7200.0)]
    fuzz_bounds = {"big_q_plasma": (0.1, 50.0), "t_plant_pulse_burn": (0.0, 50000.0)}


def test_objective_metrics_16_and_19_are_not_the_same_formula():
    """Direct counter-check against `objective_function`'s own misleading inline
    docstring (both id 16 and id 19 listed as "Major radius/burn time"): confirms the
    two ported functions read different fields and disagree numerically at a point
    where both are defined.
    """
    value_16 = objective_metric_16(rmajor=9.0, t_plant_pulse_burn=7200.0)
    value_19 = objective_metric_19(big_q_plasma=9.0, t_plant_pulse_burn=7200.0)
    assert value_16 != value_19
