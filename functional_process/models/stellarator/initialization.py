"""Pure physics functions extracted from
`functional_process.cottax.stellarator.initialization`, per `_audit/formulas_split.md`
step 2 phase A. The graph declarations that wrap these stay in that sibling
module.
"""


def calculate_pulse_durations(
    t_plant_pulse_coil_precharge,
    t_plant_pulse_plasma_current_ramp_up,
    t_plant_pulse_burn,
    t_plant_pulse_plasma_current_ramp_down,
    t_plant_pulse_fusion_ramp,
    t_plant_pulse_dwell,
):
    """Total, no-burn, and plasma-present durations of one pulse cycle.

    Ports the three summed `data.times.*` writes at the end of `st_init`.

    Parameters
    ----------
    t_plant_pulse_coil_precharge :
        Coil precharge time (s). `.times.t_plant_pulse_coil_precharge`.
    t_plant_pulse_plasma_current_ramp_up :
        Plasma current ramp-up time (s). `.times.t_plant_pulse_plasma_current_ramp_up`.
    t_plant_pulse_burn :
        Burn time (s). `.times.t_plant_pulse_burn`.
    t_plant_pulse_plasma_current_ramp_down :
        Plasma current ramp-down time (s). `.times.t_plant_pulse_plasma_current_ramp_down`.
    t_plant_pulse_fusion_ramp :
        Fusion power ramp time (s). `.times.t_plant_pulse_fusion_ramp`.
    t_plant_pulse_dwell :
        Dwell time between pulses (s). `.times.t_plant_pulse_dwell`.

    Returns
    -------
    :
        `(t_plant_pulse_plasma_present, t_plant_pulse_no_burn, t_plant_pulse_total)`,
        all in seconds.
    """
    t_plant_pulse_plasma_present = (
        t_plant_pulse_plasma_current_ramp_up
        + t_plant_pulse_fusion_ramp
        + t_plant_pulse_burn
        + t_plant_pulse_plasma_current_ramp_down
    )
    t_plant_pulse_no_burn = (
        t_plant_pulse_coil_precharge
        + t_plant_pulse_plasma_current_ramp_up
        + t_plant_pulse_plasma_current_ramp_down
        + t_plant_pulse_dwell
        + t_plant_pulse_fusion_ramp
    )
    t_plant_pulse_total = (
        t_plant_pulse_coil_precharge
        + t_plant_pulse_plasma_current_ramp_up
        + t_plant_pulse_fusion_ramp
        + t_plant_pulse_burn
        + t_plant_pulse_plasma_current_ramp_down
        + t_plant_pulse_dwell
    )
    return t_plant_pulse_plasma_present, t_plant_pulse_no_burn, t_plant_pulse_total
