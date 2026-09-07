"""Pure-functional port of the tokamak plasma-current chain."""

import dataclasses

from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    ModelNamespace,
    OutputInto,
)

from functional_process.cottax.paths import physics
from functional_process.models.physics.plasma_current import (
    calculate_current_coefficient_fiesta,
    calculate_current_coefficient_ipdg89,
    calculate_current_profile_index_wesson,
    calculate_cyclindrical_plasma_current,
    calculate_cylindrical_safety_factor,
    calculate_internal_inductance_wesson,
    calculate_plasma_current_fiesta,
    calculate_plasma_current_ipdg89,
)

__all__ = [
    "calculate_current_coefficient_fiesta",
    "calculate_current_coefficient_ipdg89",
    "calculate_cyclindrical_plasma_current",
]


class PlasmaCurrentScaling(ExplicitFunction):
    """The family that owns `.physics.plasma_current` under `i_plasma_current`."""


class Ipdg89PlasmaCurrent(PlasmaCurrentScaling):
    """`i_plasma_current == IPDG89_SCALING` (4) -- the arm `large_tokamak_eval` takes.
    """

    plasma_current = OutputInto(physics)

    def __call__(
        self,
        eps=From(physics),
        kappa95=From(physics),
        triang95=From(physics),
        rminor=From(physics),
        rmajor=From(physics),
        q95=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
    ):
        return calculate_plasma_current_ipdg89(
            eps=eps,
            kappa95=kappa95,
            triang95=triang95,
            rminor=rminor,
            rmajor=rmajor,
            q95=q95,
            b_plasma_toroidal_on_axis=b_plasma_toroidal_on_axis,
        )


class FiestaStPlasmaCurrent(PlasmaCurrentScaling):
    """`i_plasma_current == FIESTA_ST_SCALING` (9) -- the arm both tracked spherical
    tokamaks take (`spherical_tokamak_eval.IN.DAT:288`, `st_regression.IN.DAT`).
    """

    plasma_current = OutputInto(physics)

    def __call__(
        self,
        eps=From(physics),
        kappa=From(physics),
        triang=From(physics),
        rminor=From(physics),
        rmajor=From(physics),
        q95=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
    ):
        return calculate_plasma_current_fiesta(
            eps=eps,
            kappa=kappa,
            triang=triang,
            rminor=rminor,
            rmajor=rmajor,
            q95=q95,
            b_plasma_toroidal_on_axis=b_plasma_toroidal_on_axis,
        )


class PlasmaCylindricalSafetyFactor(ExplicitFunction):
    """cottax node: `qstar`, ports declared."""

    qstar = OutputInto(physics)

    def __call__(
        self,
        rmajor=From(physics),
        rminor=From(physics),
        plasma_current=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
        kappa95=From(physics),
        triang95=From(physics),
    ):
        return calculate_cylindrical_safety_factor(
            rmajor=rmajor,
            rminor=rminor,
            plasma_current=plasma_current,
            b_plasma_toroidal_on_axis=b_plasma_toroidal_on_axis,
            kappa95=kappa95,
            triang95=triang95,
        )


class CurrentProfileIndexScaling(ExplicitFunction):
    """The family that owns `.physics.alphaj` under `i_alphaj` (`physics.py:334-348`).
    """


class WessonCurrentProfileIndex(CurrentProfileIndexScaling):
    """`i_alphaj == WESSON` (1) -- `large_tokamak_eval.IN.DAT:275`."""

    alphaj = OutputInto(physics)

    def __call__(self, qstar=From(physics), q0=From(physics)):
        return calculate_current_profile_index_wesson(qstar, q0)


class NormalisedInternalInductanceScaling(ExplicitFunction):
    """The family that owns `.physics.ind_plasma_internal_norm` under
    `i_ind_plasma_internal_norm` (`physics.py:4738-4764`).
    """


class WessonInternalInductance(NormalisedInternalInductanceScaling):
    """`i_ind_plasma_internal_norm == WESSON` (1) -- `large_tokamak_eval.IN.DAT:311`."""

    ind_plasma_internal_norm = OutputInto(physics)

    def __call__(self, alphaj=From(physics)):
        return calculate_internal_inductance_wesson(alphaj)


class TokamakPlasmaCurrent(ModelNamespace):
    """`.tokamak.plasma_current` -- the plasma-current chain, three slots."""

    plasma_current: PlasmaCurrentScaling = dataclasses.field(kw_only=True)
    """`.physics.i_plasma_current` -- nine values, one occupant."""

    cylindrical_safety_factor: PlasmaCylindricalSafetyFactor = (
        PlasmaCylindricalSafetyFactor()
    )
    """Unconditional -- PROCESS computes `qstar` outside every switch
    (`physics.py:303`).
    """

    current_profile_index: CurrentProfileIndexScaling | None = dataclasses.field(
        kw_only=True
    )
    """`.physics.i_alphaj` -- `1` (Wesson) is written; `0` (USER_INPUT) is an **empty
    slot** (`physics.py:338` assigns the field to itself), under which `.physics.alphaj`
    is a boundary input with no producer.
    """
