"""Pure-functional port of `process/models/physics/plasma_geometry.py`."""

from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

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


class PlasmaMinorRadius(ExplicitFunction):
    """cottax node: `calculate_minor_radius`, ports declared."""

    rminor = OutputInto(physics)
    eps = OutputInto(physics)

    def __call__(self, rmajor=From(physics), aspect=From(physics)):
        return calculate_minor_radius(rmajor, aspect)


class PlasmaShapeKappa95Triang95(ExplicitFunction):
    """The family that owns `.physics.kappa95`/`.physics.triang95` under
    `i_plasma_geometry`: one occupant per value, per `_audit/traceability_policy.md`'s
    split default -- **this pass ports only `IPDG89_X_POINT` (0)**, the value
    `large_tokamak_eval.IN.DAT` uses.
    """


class Ipdg89XPointPlasmaShape(PlasmaShapeKappa95Triang95):
    """`i_plasma_geometry == IPDG89_X_POINT` (0)."""

    kappa95 = OutputInto(physics)
    triang95 = OutputInto(physics)

    def __call__(self, kappa=From(physics), triang=From(physics)):
        return calculate_shape_ipdg89_x_point(kappa, triang)


class CreateDataEuDemoXPointPlasmaShape(PlasmaShapeKappa95Triang95):
    """`i_plasma_geometry == CREATE_DATA_EU_DEMO_X_POINT` (10)."""

    kappa95 = OutputInto(physics)
    kappa = OutputInto(physics)
    triang95 = OutputInto(physics)

    def __call__(
        self,
        aspect=From(physics),
        m_s_limit=From(physics),
        triang=From(physics),
    ):
        return calculate_shape_create_data_eu_demo_x_point(aspect, m_s_limit, triang)


class PlasmaGeometryArm(ExplicitFunction):
    """The family that owns `.physics.len_plasma_poloidal`, `.vol_plasma`,
    `.a_plasma_poloidal`, `.a_plasma_surface`: one occupant per arm of the compound
    switch `i_plasma_current == 8 or i_plasma_shape == SAUTER`
    (`process/models/physics/plasma_geometry.py:467-469`).
    """


class DoubleArcPlasmaGeometry(PlasmaGeometryArm):
    """`i_plasma_current != 8 and i_plasma_shape != SAUTER` -- the arm
    `large_tokamak_eval.IN.DAT` takes.
    """

    len_plasma_poloidal = OutputInto(physics)
    vol_plasma = OutputInto(physics)
    a_plasma_poloidal = OutputInto(physics)
    a_plasma_surface = OutputInto(physics)

    def __call__(
        self,
        rmajor=From(physics),
        rminor=From(physics),
        kappa=From(physics),
        triang=From(physics),
        f_vol_plasma=From(physics),
    ):
        return calculate_geometry_double_arc(rmajor, rminor, kappa, triang, f_vol_plasma)
