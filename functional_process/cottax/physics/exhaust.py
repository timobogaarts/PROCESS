"""Pure-functional port of `process/models/physics/exhaust.py`."""

from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    OutputInto,
)

from functional_process.cottax.paths import physics
from functional_process.models.physics.exhaust import (
    calculate_eu_demo_re_attachment_metric,
    calculate_psep_over_r_metric,
    calculate_radiation_fraction,
)


class RadiationFraction(ExplicitFunction):
    """cottax node: `calculate_radiation_fraction`, ports declared."""

    f_p_plasma_separatrix_rad = OutputInto(physics)

    def __call__(
        self,
        p_plasma_rad_mw=From(physics),
        p_plasma_heating_total_mw=From(physics),
    ):
        return calculate_radiation_fraction(p_plasma_rad_mw, p_plasma_heating_total_mw)


class EuDemoReAttachmentMetric(ExplicitFunction):
    """cottax node: `calculate_eu_demo_re_attachment_metric`, ports declared."""

    p_div_bt_q_aspect_rmajor_mw = OutputInto(physics)

    def __call__(
        self,
        p_plasma_separatrix_mw_raw=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
        q95=From(physics),
        aspect=From(physics),
        rmajor=From(physics),
    ):
        return calculate_eu_demo_re_attachment_metric(
            p_plasma_separatrix_mw_raw,
            b_plasma_toroidal_on_axis,
            q95,
            aspect,
            rmajor,
        )


class PsepOverRMetric(ExplicitFunction):
    """cottax node: `calculate_psep_over_r_metric`, ports declared."""

    p_plasma_separatrix_rmajor_mw = OutputInto(physics)

    def __call__(
        self,
        p_plasma_separatrix_mw_raw=From(physics),
        rmajor=From(physics),
    ):
        return calculate_psep_over_r_metric(p_plasma_separatrix_mw_raw, rmajor)
