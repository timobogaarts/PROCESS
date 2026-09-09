"""Pure-functional port of `st_fwbs`'s S4 component-mass block
(`process/models/stellarator/stellarator.py:1045-1274`, excluding S3's 1030-1043).
"""

from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    OutputInto,
)

from functional_process.cottax.paths import (
    fwbs,
)
from functional_process.models.stellarator.stellarator_fwbs_s4 import (
    calculate_blanket_component_masses,
    calculate_shield_mass,
)


class BlanketComponentMasses(ExplicitFunction):
    """cottax node: `calculate_blanket_component_masses`, unchanged, ports declared."""

    m_blkt_li2o = OutputInto(fwbs)
    m_blkt_beryllium = OutputInto(fwbs)
    m_blkt_steel_total = OutputInto(fwbs)
    m_blkt_vanadium = OutputInto(fwbs)
    m_blkt_total = OutputInto(fwbs)

    def __call__(
        self,
        vol_blkt_total=From(fwbs),
        fblli2o=From(fwbs),
        fblbe=From(fwbs),
        den_steel=From(fwbs),
        fblss=From(fwbs),
        fblvd=From(fwbs),
    ):
        return calculate_blanket_component_masses(
            vol_blkt_total, fblli2o, fblbe, den_steel, fblss, fblvd
        )


class ShieldMass(ExplicitFunction):
    """cottax node: `calculate_shield_mass`, unchanged, ports declared."""

    whtshld = OutputInto(fwbs)
    wpenshld = OutputInto(fwbs)

    def __call__(
        self,
        vol_shld_total=From(fwbs),
        den_steel=From(fwbs),
        vfshld=From(fwbs),
    ):
        return calculate_shield_mass(vol_shld_total, den_steel, vfshld)
