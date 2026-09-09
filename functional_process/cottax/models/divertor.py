"""Pure-functional port of `process/models/divertor.py` (`Divertor`,
`.tokamak.divertor`) -- **not** `process/models/stellarator/divertor.py`, which is a
different model of a different device's divertor, already ported separately and half of
the stellarator graph's one non-`problem` cycle.
"""

import jax.numpy as jnp  # noqa: F401
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    FromExactly,
    OutputInto,
)

from functional_process.models.divertor import (
    calculate_divertor_heat_flux_split,
    calculate_divertor_heat_load_wade,
    calculate_divertor_heat_load_wade_double_null,
)
from functional_process.models.safe_math import safe_pow  # noqa: F401
from functional_process.cottax.paths import blanket, divertor, fwbs, physics


class DivertorHeatFluxSplit(ExplicitFunction):
    """cottax node: `.tokamak.divertor`'s unconditional heat-flux split."""

    deg_div_poloidal_plasma = OutputInto(divertor)
    f_ster_div_single = OutputInto(fwbs)
    p_div_nuclear_heat_total_mw = OutputInto(fwbs)
    p_div_rad_total_mw = OutputInto(fwbs)

    def __call__(
        self,
        deg_blkt_inboard_poloidal_plasma=From(blanket),
        p_plasma_neutron_mw=From(physics),
        p_plasma_rad_mw=From(physics),
        n_divertors=From(divertor),
    ):
        return calculate_divertor_heat_flux_split(
            deg_blkt_inboard_poloidal_plasma,
            p_plasma_neutron_mw,
            p_plasma_rad_mw,
            n_divertors,
        )


class DivertorHeatLoadWade(ExplicitFunction):
    """The family that owns `.divertor.pflux_div_heat_load_mw` at `i_div_heat_load ==
    DivertorHeatLoadModel.WADE` (2): one occupant per `n_divertors` arm of
    `Divertor.divwade`'s own internal branch.
    """


class DivertorHeatLoadWadeSingleNull(DivertorHeatLoadWade):
    """cottax node: `.tokamak.divertor`'s heat-load occupant at `n_divertors == 1`
    (single null) -- this run's derived value on `large_tokamak_eval.IN.DAT`, see module
    docstring.
    """

    pflux_div_heat_load_mw = OutputInto(divertor)

    def __call__(
        self,
        rmajor=From(physics),
        rminor=From(physics),
        aspect=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
        # **Not `From(physics)`.** `divwade`'s *parameter* is called
        # `b_plasma_poloidal_average` and the *field* PROCESS passes into it is
        # `.physics.b_plasma_surface_poloidal_average` (`process/models/
        # divertor.py:90-95`, positional) -- the two names differ, so resolving the
        # parameter name against the namespace named a field that does not exist on
        # `DataStructure` at all. That is not a read of the wrong number, it is a read
        # of *no* number: the port kept the parameter's spelling as a boundary input
        # nothing produces, and the MDA harness's first tokamak run reported it as its
        # only ungrounded input, with 16 outputs downstream of it unverifiable.
        # `physics.py:707`/`:3827` pass the same field into the same parameter name
        # elsewhere, and `pure_formulas.py:527` already binds it the right way round.
        b_plasma_poloidal_average=FromExactly(physics.b_plasma_surface_poloidal_average),
        p_plasma_separatrix_mw=From(physics),
        f_div_flux_expansion=From(divertor),
        nd_plasma_separatrix_electron=From(physics),
        deg_div_field_plate=From(divertor),
        rad_fraction_sol=From(physics),
    ):
        return calculate_divertor_heat_load_wade(
            rmajor,
            rminor,
            aspect,
            b_plasma_toroidal_on_axis,
            b_plasma_poloidal_average,
            p_plasma_separatrix_mw,
            f_div_flux_expansion,
            nd_plasma_separatrix_electron,
            deg_div_field_plate,
            rad_fraction_sol,
        )


class DivertorHeatLoadWadeDoubleNull(DivertorHeatLoadWade):
    """cottax node: `.tokamak.divertor`'s heat-load occupant at `n_divertors == 2`
    (double null) -- the value `spherical_tokamak_eval.IN.DAT` and
    `st_regression.IN.DAT` derive from `i_single_null = 0`.
    """

    pflux_div_heat_load_mw = OutputInto(divertor)

    def __call__(
        self,
        rmajor=From(physics),
        rminor=From(physics),
        aspect=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
        # Same name mismatch as the single-null sibling, and the same fix -- see the
        # comment there.
        b_plasma_poloidal_average=FromExactly(physics.b_plasma_surface_poloidal_average),
        p_plasma_separatrix_mw=From(physics),
        f_div_flux_expansion=From(divertor),
        nd_plasma_separatrix_electron=From(physics),
        deg_div_field_plate=From(divertor),
        rad_fraction_sol=From(physics),
        f_p_div_lower=From(physics),
    ):
        return calculate_divertor_heat_load_wade_double_null(
            rmajor,
            rminor,
            aspect,
            b_plasma_toroidal_on_axis,
            b_plasma_poloidal_average,
            p_plasma_separatrix_mw,
            f_div_flux_expansion,
            nd_plasma_separatrix_electron,
            deg_div_field_plate,
            rad_fraction_sol,
            f_p_div_lower,
        )
