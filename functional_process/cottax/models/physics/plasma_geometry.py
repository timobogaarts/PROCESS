"""Pure-functional port of `process/models/physics/plasma_geometry.py`."""

from functional_process.cottax.paths import physics
from functional_process.models.physics.plasma_geometry import (
    calculate_geometry_double_arc,
    calculate_geometry_sauter,
    calculate_minor_radius,
    calculate_shape_create_data_eu_demo_x_point,
    calculate_shape_ipdg89_x_point,
    plasma_angles_arcs,
    plasma_cross_section,
    plasma_poloidal_perimeter,
    plasma_surface_area,
    plasma_volume,
    sauter_geometry,
)

__all__ = [
    "calculate_geometry_sauter",
    "plasma_angles_arcs",
    "plasma_cross_section",
    "plasma_poloidal_perimeter",
    "plasma_surface_area",
    "plasma_volume",
    "sauter_geometry",
]


# cottax node: `calculate_minor_radius`, ports declared.
PlasmaMinorRadius = physics.function(calculate_minor_radius, "rminor", "eps")


# The PlasmaShapeKappa95Triang95 family:
# The family that owns `.physics.kappa95`/`.physics.triang95` under
# `i_plasma_geometry`: one occupant per value, per `_audit/traceability_policy.md`'s
# split default -- **this pass ports only `IPDG89_X_POINT` (0)**, the value
# `large_tokamak_eval.IN.DAT` uses.


# `i_plasma_geometry == IPDG89_X_POINT` (0).
Ipdg89XPointPlasmaShape = physics.function(
    calculate_shape_ipdg89_x_point, "kappa95", "triang95"
)


# `i_plasma_geometry == CREATE_DATA_EU_DEMO_X_POINT` (10).
CreateDataEuDemoXPointPlasmaShape = physics.function(
    calculate_shape_create_data_eu_demo_x_point, "kappa95", "kappa", "triang95"
)


# The PlasmaGeometryArm family:
# The family that owns `.physics.len_plasma_poloidal`, `.vol_plasma`,
# `.a_plasma_poloidal`, `.a_plasma_surface`: one occupant per arm of the compound
# switch `i_plasma_current == 8 or i_plasma_shape == SAUTER`
# (`process/models/physics/plasma_geometry.py:467-469`).


# `i_plasma_current != 8 and i_plasma_shape != SAUTER` -- the arm
# `large_tokamak_eval.IN.DAT` takes.
DoubleArcPlasmaGeometry = physics.function(
    calculate_geometry_double_arc,
    "len_plasma_poloidal",
    "vol_plasma",
    "a_plasma_poloidal",
    "a_plasma_surface",
)
