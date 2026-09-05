"""Pure-functional port of `process/models/divertor.py` (`Divertor`,
`.tokamak.divertor`) -- **not** `process/models/stellarator/divertor.py`, which is a
different model of a different device's divertor, already ported separately and half of
the stellarator graph's one non-`problem` cycle. Recording the distinction explicitly,
per the wave-1 brief.

Audit record: `functional_process/_audit/units/models/divertor.md`. Entry point is
`Divertor.run()`, which (a) computes an unconditional heat-flux split, then (b)
dispatches on `.divertor.i_div_heat_load` to one of three heat-load models. Only the
`WADE` arm (`i_div_heat_load == 2`) is live on
`tests/regression/input_files/large_tokamak_eval.IN.DAT:139`; `USER_INPUT` and
`PENG_CHAMBER` (`divtart`, the tight-aspect-ratio ST model) are UNPORTED.

`divwade` also branches internally on `.divertor.n_divertors` (`1` for single-null, `2`
for double-null) -- read to *branch*, not arithmetically, so per the wave-1 policy it
selects a formula, not a parameter. `large_tokamak_eval.IN.DAT:307` sets
`i_single_null = 1`, and `process/core/init.py:606-616` derives `n_divertors = 1` from
that (`DivertorNumberModels.SINGLE_NULL`) -- **not** the `DataStructure` field's own
default of `2`, which only applies before `init.py` runs. `calculate_divertor_heat_load_
wade` below therefore bakes in the single-null formula (`pflux_div_heat_load_mw =
hldiv_base` directly, no `f_p_div_lower` read at all).

2026-08-27 (the double-null wave): the `n_divertors == 2` formula is written too, as
`calculate_divertor_heat_load_wade_double_null`, for the two spherical-tokamak input
files that set `i_single_null = 0`. It is the one arm in that wave that adds a *read*
rather than only changing a constant: `.physics.f_p_div_lower`.

**`.physics.f_p_div_lower` is a declared boundary input and nothing in this port
produces it.** Measured, not assumed: `grep -rn f_p_div_lower process/` finds the field
declared at `physics_variables.py:740` with a default of `1.0`, registered as a
user-settable `InputVariable` at `core/input.py:189` and as scan variable 51
(`core/scan.py:194`), read at `models/divertor.py:101`/`:378-379` and at
`models/physics/physics.py:852`/`:1008-1052` -- and **written nowhere** outside the
input parser. It is an input in PROCESS and it is an input here; the double-null
occupant declares the read and the boundary carries it, which is the honest answer.
Stubbing it to `1.0` would silently pick the lower divertor and hide the `max`. Both ST
files set it explicitly (`spherical_tokamak_eval.IN.DAT:266`, `st_regression.IN.DAT:634`,
both `0.5`).
"""

import jax.numpy as jnp  # noqa: F401
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    FromExactly,
    OutputInto,
)

from functional_process.divertor import (
    calculate_divertor_heat_flux_split,
    calculate_divertor_heat_load_wade,
    calculate_divertor_heat_load_wade_double_null,
)
from functional_process.models.safe_math import safe_pow  # noqa: F401
from functional_process.paths import blanket, divertor, fwbs, physics


class DivertorHeatFluxSplit(ExplicitFunction):
    """cottax node: `.tokamak.divertor`'s unconditional heat-flux split.

    Not switch-gated -- runs regardless of `i_div_heat_load`.
    """

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

    Both arms are written (2026-08-27). `USER_INPUT` and `PENG_CHAMBER` (`divtart`) are
    different models of the same quantity and remain UNPORTED -- they are values of
    `i_div_heat_load`, not members of this family, which is why `indat.py` answers the
    two switches with one joint `_divertor_heat_load_arm`.
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

    Reads one field its single-null sibling does not: `.physics.f_p_div_lower`, a
    boundary input with no producer anywhere in this port (see the module docstring for
    the measurement). It is declared rather than stubbed, so the boundary census counts
    it and a machine assembled from this occupant says out loud that it needs a number
    the graph cannot compute.
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
