"""Pure-functional port of `process/models/physics/l_h_transition.py`."""

from functional_process.cottax.paths import physics
from functional_process.models.physics.l_h_transition import (
    calculate_hubbard2012_lower,
    calculate_hubbard2012_nominal,
    calculate_hubbard2012_upper,
    calculate_hubbard2017,
    calculate_iter1996_lower,
    calculate_iter1996_nominal,
    calculate_iter1996_upper,
    calculate_martin08_aspect_lower,
    calculate_martin08_aspect_lower_threshold_power,
    calculate_martin08_aspect_nominal,
    calculate_martin08_aspect_nominal_threshold_power,
    calculate_martin08_aspect_upper,
    calculate_martin08_aspect_upper_threshold_power,
    calculate_martin08_lower,
    calculate_martin08_lower_threshold_power,
    calculate_martin08_nominal,
    calculate_martin08_nominal_threshold_power,
    calculate_martin08_upper,
    calculate_martin08_upper_threshold_power,
    calculate_snipes1997_iter,
    calculate_snipes1997_kappa,
    calculate_snipes2000_closed_divertor_lower,
    calculate_snipes2000_closed_divertor_nominal,
    calculate_snipes2000_closed_divertor_upper,
    calculate_snipes2000_lower,
    calculate_snipes2000_nominal,
    calculate_snipes2000_upper,
)

__all__ = [
    "calculate_hubbard2012_lower",
    "calculate_hubbard2012_nominal",
    "calculate_hubbard2012_upper",
    "calculate_hubbard2017",
    "calculate_iter1996_lower",
    "calculate_iter1996_nominal",
    "calculate_iter1996_upper",
    "calculate_martin08_aspect_lower",
    "calculate_martin08_aspect_nominal",
    "calculate_martin08_aspect_upper",
    "calculate_martin08_lower",
    "calculate_martin08_nominal",
    "calculate_martin08_upper",
    "calculate_snipes1997_iter",
    "calculate_snipes1997_kappa",
    "calculate_snipes2000_closed_divertor_lower",
    "calculate_snipes2000_closed_divertor_nominal",
    "calculate_snipes2000_closed_divertor_upper",
    "calculate_snipes2000_lower",
    "calculate_snipes2000_nominal",
    "calculate_snipes2000_upper",
]


# The LHThresholdPower family:
# The family that owns `.physics.p_l_h_threshold_mw`: one occupant per arm.


# `i_l_h_threshold == 6`.
Martin08NominalLHThresholdPower = physics.function(
    calculate_martin08_nominal_threshold_power, "p_l_h_threshold_mw"
)


# `i_l_h_threshold == 7`.
Martin08UpperLHThresholdPower = physics.function(
    calculate_martin08_upper_threshold_power, "p_l_h_threshold_mw"
)


# `i_l_h_threshold == 8`.
Martin08LowerLHThresholdPower = physics.function(
    calculate_martin08_lower_threshold_power, "p_l_h_threshold_mw"
)


# `i_l_h_threshold == 19` -- the reference arm on `large_tokamak_eval.IN.DAT`.
Martin08AspectNominalLHThresholdPower = physics.function(
    calculate_martin08_aspect_nominal_threshold_power, "p_l_h_threshold_mw"
)


# `i_l_h_threshold == 20`.
Martin08AspectUpperLHThresholdPower = physics.function(
    calculate_martin08_aspect_upper_threshold_power, "p_l_h_threshold_mw"
)


# `i_l_h_threshold == 21`.
Martin08AspectLowerLHThresholdPower = physics.function(
    calculate_martin08_aspect_lower_threshold_power, "p_l_h_threshold_mw"
)
