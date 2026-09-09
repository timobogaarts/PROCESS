"""Pure-functional port of `process/models/physics/exhaust.py`."""

from cottax.interfaces.pytree_namespace_module import (
    From,
    FromExactly,
    OutputInto,
)

from functional_process.cottax.paths import physics
from functional_process.cottax.wraps import WrapsFunction
from functional_process.models.physics.exhaust import (
    calculate_eu_demo_re_attachment_metric,
    calculate_psep_over_r_metric,
    calculate_radiation_fraction,
)


class RadiationFraction(WrapsFunction):
    """cottax node: `calculate_radiation_fraction`, ports declared."""

    fn = calculate_radiation_fraction

    p_plasma_rad_mw = From(physics)
    p_plasma_heating_mw = FromExactly(physics.p_plasma_heating_total_mw)

    f_p_plasma_separatrix_rad = OutputInto(physics)


class EuDemoReAttachmentMetric(WrapsFunction):
    """cottax node: `calculate_eu_demo_re_attachment_metric`, ports declared."""

    fn = calculate_eu_demo_re_attachment_metric

    p_plasma_separatrix_mw = FromExactly(physics.p_plasma_separatrix_mw_raw)
    b_plasma_toroidal_on_axis = From(physics)
    q95 = From(physics)
    aspect = From(physics)
    rmajor = From(physics)

    p_div_bt_q_aspect_rmajor_mw = OutputInto(physics)


class PsepOverRMetric(WrapsFunction):
    """cottax node: `calculate_psep_over_r_metric`, ports declared."""

    fn = calculate_psep_over_r_metric

    p_plasma_separatrix_mw = FromExactly(physics.p_plasma_separatrix_mw_raw)
    rmajor = From(physics)

    p_plasma_separatrix_rmajor_mw = OutputInto(physics)
