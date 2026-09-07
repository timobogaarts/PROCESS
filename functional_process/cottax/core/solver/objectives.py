"""Pure-functional port of `process/core/solver/objective_function`
(`process/core/solver/objectives.py`).
"""

from functional_process.vocabulary import FiguresOfMerit
from functional_process.vocabulary import AvailabilityModel


def objective_metric_1(rmajor):
    """`FiguresOfMerit.MAJOR_RADIUS`. Ports `objectives.py:57`."""
    return 0.2 * rmajor


def objective_metric_3(pflux_fw_neutron_mw):
    """`FiguresOfMerit.NEUTRON_WALL_LOAD`. Ports `objectives.py:59`."""
    return pflux_fw_neutron_mw


def objective_metric_4(tfcmw, srcktpm):
    """`FiguresOfMerit.P_TF_PLUS_P_PF`."""
    return (tfcmw + 1.0e-3 * srcktpm) / 10.0


def objective_metric_5(big_q_plasma):
    """`FiguresOfMerit.FUSION_GAIN_Q`. Ports `objectives.py:63`."""
    return big_q_plasma


def objective_metric_6(coe):
    """`FiguresOfMerit.COST_OF_ELECTRICITY`. Ports `objectives.py:65`."""
    return coe / 100.0


def objective_metric_7(cdirt, concost, ireactor):
    """`FiguresOfMerit.CAPITAL_COST`."""
    return cdirt / 1.0e3 if ireactor == 0 else concost / 1.0e4


def objective_metric_8(aspect):
    """`FiguresOfMerit.ASPECT_RATIO`. Ports `objectives.py:73`."""
    return aspect


def objective_metric_9(pflux_div_heat_load_mw):
    """`FiguresOfMerit.DIVERTOR_HEAT_LOAD`. Ports `objectives.py:75`."""
    return pflux_div_heat_load_mw


def objective_metric_10(b_plasma_toroidal_on_axis):
    """`FiguresOfMerit.TOROIDAL_FIELD`. Ports `objectives.py:77`."""
    return b_plasma_toroidal_on_axis


def objective_metric_11(p_hcd_injected_total_mw):
    """`FiguresOfMerit.TOTAL_INJECTED_POWER`. Ports `objectives.py:79`."""
    return p_hcd_injected_total_mw


def objective_metric_14(t_plant_pulse_burn):
    """`FiguresOfMerit.PULSE_LENGTH`. Ports `objectives.py:81`."""
    return t_plant_pulse_burn / 2.0e4


def objective_metric_15(i_plant_availability, f_t_plant_available):
    """`FiguresOfMerit.PLANT_AVAILABILITY_FACTOR`."""
    if AvailabilityModel(i_plant_availability) == AvailabilityModel.USER_INPUT:
        raise ValueError(
            "objective_metric_15 (plant availability factor): "
            "`i_plant_availability` is AvailabilityModel.USER_INPUT -- "
            "`f_t_plant_available` is not a model output in this mode, "
            "so there is nothing to optimise against."
        )
    return f_t_plant_available


def objective_metric_16(rmajor, t_plant_pulse_burn):
    """`FiguresOfMerit.MIN_R0_MAX_TAU_BURN`."""
    return 0.95 * (rmajor / 9.0) - 0.05 * (t_plant_pulse_burn / 7200.0)


def objective_metric_17(p_plant_electric_net_mw):
    """`FiguresOfMerit.NET_ELECTRICAL_OUTPUT`. Ports `objectives.py:97`."""
    return p_plant_electric_net_mw / 500.0


def objective_metric_18():
    """`FiguresOfMerit.NULL_FIGURE_OF_MERIT`."""
    return 1.0


def objective_metric_19(big_q_plasma, t_plant_pulse_burn):
    """`FiguresOfMerit.MAX_Q_MAX_T_PLANT_PULSE_BURN`."""
    return -0.5 * (big_q_plasma / 20.0) - 0.5 * (t_plant_pulse_burn / 7200.0)


OBJECTIVE_METRICS = {
    FiguresOfMerit.MAJOR_RADIUS: objective_metric_1,
    FiguresOfMerit.NEUTRON_WALL_LOAD: objective_metric_3,
    FiguresOfMerit.P_TF_PLUS_P_PF: objective_metric_4,
    FiguresOfMerit.FUSION_GAIN_Q: objective_metric_5,
    FiguresOfMerit.COST_OF_ELECTRICITY: objective_metric_6,
    FiguresOfMerit.CAPITAL_COST: objective_metric_7,
    FiguresOfMerit.ASPECT_RATIO: objective_metric_8,
    FiguresOfMerit.DIVERTOR_HEAT_LOAD: objective_metric_9,
    FiguresOfMerit.TOROIDAL_FIELD: objective_metric_10,
    FiguresOfMerit.TOTAL_INJECTED_POWER: objective_metric_11,
    FiguresOfMerit.PULSE_LENGTH: objective_metric_14,
    FiguresOfMerit.PLANT_AVAILABILITY_FACTOR: objective_metric_15,
    FiguresOfMerit.MIN_R0_MAX_TAU_BURN: objective_metric_16,
    FiguresOfMerit.NET_ELECTRICAL_OUTPUT: objective_metric_17,
    FiguresOfMerit.NULL_FIGURE_OF_MERIT: objective_metric_18,
    FiguresOfMerit.MAX_Q_MAX_T_PLANT_PULSE_BURN: objective_metric_19,
}
"""`FiguresOfMerit` member -> the `objective_metric_<id>` function above that ports it.
"""
