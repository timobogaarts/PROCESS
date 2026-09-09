"""Pure-functional port of `process/models/physics/l_h_transition.py`."""

from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.cottax.paths import physics
from functional_process.cottax.wraps import WrapsFunction
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


class LHThresholdPower(ExplicitFunction):
    """The family that owns `.physics.p_l_h_threshold_mw`: one occupant per arm."""


class Martin08NominalLHThresholdPower(LHThresholdPower, WrapsFunction):
    """`i_l_h_threshold == 6`."""

    fn = calculate_martin08_nominal_threshold_power

    nd_plasma_electron_line = From(physics)
    b_plasma_toroidal_on_axis = From(physics)
    a_plasma_surface = From(physics)
    m_ions_total_amu = From(physics)

    p_l_h_threshold_mw = OutputInto(physics)


class Martin08UpperLHThresholdPower(LHThresholdPower, WrapsFunction):
    """`i_l_h_threshold == 7`."""

    fn = calculate_martin08_upper_threshold_power

    nd_plasma_electron_line = From(physics)
    b_plasma_toroidal_on_axis = From(physics)
    a_plasma_surface = From(physics)
    m_ions_total_amu = From(physics)

    p_l_h_threshold_mw = OutputInto(physics)


class Martin08LowerLHThresholdPower(LHThresholdPower, WrapsFunction):
    """`i_l_h_threshold == 8`."""

    fn = calculate_martin08_lower_threshold_power

    nd_plasma_electron_line = From(physics)
    b_plasma_toroidal_on_axis = From(physics)
    a_plasma_surface = From(physics)
    m_ions_total_amu = From(physics)

    p_l_h_threshold_mw = OutputInto(physics)


class Martin08AspectNominalLHThresholdPower(LHThresholdPower, WrapsFunction):
    """`i_l_h_threshold == 19` -- the reference arm on `large_tokamak_eval.IN.DAT`."""

    fn = calculate_martin08_aspect_nominal_threshold_power

    nd_plasma_electron_line = From(physics)
    b_plasma_toroidal_on_axis = From(physics)
    a_plasma_surface = From(physics)
    m_ions_total_amu = From(physics)
    aspect = From(physics)

    p_l_h_threshold_mw = OutputInto(physics)


class Martin08AspectUpperLHThresholdPower(LHThresholdPower, WrapsFunction):
    """`i_l_h_threshold == 20`."""

    fn = calculate_martin08_aspect_upper_threshold_power

    nd_plasma_electron_line = From(physics)
    b_plasma_toroidal_on_axis = From(physics)
    a_plasma_surface = From(physics)
    m_ions_total_amu = From(physics)
    aspect = From(physics)

    p_l_h_threshold_mw = OutputInto(physics)


class Martin08AspectLowerLHThresholdPower(LHThresholdPower, WrapsFunction):
    """`i_l_h_threshold == 21`."""

    fn = calculate_martin08_aspect_lower_threshold_power

    nd_plasma_electron_line = From(physics)
    b_plasma_toroidal_on_axis = From(physics)
    a_plasma_surface = From(physics)
    m_ions_total_amu = From(physics)
    aspect = From(physics)

    p_l_h_threshold_mw = OutputInto(physics)
