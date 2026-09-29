"""Pure-functional port of `st_fwbs`'s S3 fragment (`stellarator.py:1030-1043`)."""

from functional_process.cottax.paths import (
    divertor,
)
from functional_process.models.stellarator.stellarator_fwbs_s3 import (
    calculate_divertor_plate_mass,
)

# cottax node: `calculate_divertor_plate_mass`, unchanged, ports declared.
DivertorPlateMass = divertor.function(calculate_divertor_plate_mass, "m_div_plate")
