"""Pure-functional port of `st_fwbs`'s S4 component-mass block
(`process/models/stellarator/stellarator.py:1045-1274`, excluding S3's 1030-1043).
"""

from functional_process.cottax.paths import (
    fwbs,
)
from functional_process.models.stellarator.stellarator_fwbs_s4 import (
    calculate_blanket_component_masses,
    calculate_shield_mass,
)

# cottax node: `calculate_blanket_component_masses`, unchanged, ports declared.
BlanketComponentMasses = fwbs.function(
    calculate_blanket_component_masses,
    "m_blkt_li2o",
    "m_blkt_beryllium",
    "m_blkt_steel_total",
    "m_blkt_vanadium",
    "m_blkt_total",
)


# cottax node: `calculate_shield_mass`, unchanged, ports declared.
ShieldMass = fwbs.function(calculate_shield_mass, "whtshld", "wpenshld")
