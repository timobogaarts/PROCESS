"""Pure-functional port of `process/models/physics/scrape_off_layer.py`
(`ScrapeOffLayer`, would-be `.tokamak.scrape_off_layer`).
"""

import dataclasses

from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    FromExactly,
    ModelNamespace,
    OutputInto,
)

from functional_process.cottax.paths import physics
from functional_process.cottax.wraps import WrapsFunction
from functional_process.models.physics.scrape_off_layer import (
    calculate_eich2013_sol_power_decay_length,
    calculate_mast2014_sol_power_decay_length_1,
    calculate_mast2014_sol_power_decay_length_2,  # noqa: F401 -- re-exported for tests
    calculate_mast2014_sol_power_decay_length_2_from_plasma_current,
    calculate_scrape_off_layer,
    calculate_upstream_sol_outboard_parallel_area,
    outboard_sol_eich13_parallel_power_flux,
    outboard_sol_parallel_power_flux,
    outboard_sol_power_decay_length_eich2013,
)

__all__ = [
    "calculate_scrape_off_layer",
]


class Eich2013SOLPowerDecayLength(WrapsFunction):
    """cottax node: `calculate_eich2013_sol_power_decay_length`, unconditional."""

    fn = calculate_eich2013_sol_power_decay_length

    p_plasma_separatrix_mw = FromExactly(physics.p_plasma_separatrix_mw_raw)
    rmajor = From(physics)
    b_plasma_surface_poloidal_average = From(physics)
    aspect = From(physics)

    len_plasma_sol_eich13_power_decay = OutputInto(physics)


class Mast2014SOLPowerDecayLength1(WrapsFunction):
    """cottax node: `calculate_mast2014_sol_power_decay_length_1`, unconditional."""

    fn = calculate_mast2014_sol_power_decay_length_1

    p_plasma_separatrix_mw = FromExactly(physics.p_plasma_separatrix_mw_raw)
    b_plasma_surface_poloidal_average = From(physics)

    len_plasma_sol_mast14_power_decay_1 = OutputInto(physics)


class Mast2014SOLPowerDecayLength2(WrapsFunction):
    """cottax node: `calculate_mast2014_sol_power_decay_length_2_from_plasma_current` --
    `calculate_mast2014_sol_power_decay_length_2`, unconditional.
    """

    fn = calculate_mast2014_sol_power_decay_length_2_from_plasma_current

    p_plasma_separatrix_mw_raw = From(physics)
    plasma_current = From(physics)

    len_plasma_sol_mast14_power_decay_2 = OutputInto(physics)


class OutboardSOLPowerDecayLength(ExplicitFunction):
    """The family that owns `.physics.len_sol_outboard_power_decay`: a
    computes-then-selects switch, `.physics.i_len_sol_outboard_power_decay`.
    """


class OutboardSOLPowerDecayLengthEich2013(OutboardSOLPowerDecayLength, WrapsFunction):
    """`i_len_sol_outboard_power_decay == EICH_2013` (1) -- PROCESS's own default
    (`physics_variables.py:1718`) and the value live on `large_tokamak_eval.IN.DAT`,
    which never sets this switch.
    """

    fn = outboard_sol_power_decay_length_eich2013

    len_plasma_sol_eich13_power_decay = From(physics)

    len_sol_outboard_power_decay = OutputInto(physics)


class UpstreamSOLOutboardParallelArea(WrapsFunction):
    """cottax node: `calculate_upstream_sol_outboard_parallel_area` at the
    switch-selected length -> `.physics.a_plasma_outboard_sol_parallel`.
    """

    fn = calculate_upstream_sol_outboard_parallel_area

    rmajor = From(physics)
    rminor = From(physics)
    len_plasma_sol_power_decay = FromExactly(physics.len_sol_outboard_power_decay)
    b_plasma_outboard_total = From(physics)
    b_plasma_surface_poloidal_average = From(physics)

    a_plasma_outboard_sol_parallel = OutputInto(physics)


class UpstreamSOLOutboardEich13ParallelArea(WrapsFunction):
    """cottax node: `calculate_upstream_sol_outboard_parallel_area` at the Eich 2013
    length specifically -> `.physics.a_plasma_outboard_sol_eich13_parallel`.
    """

    fn = calculate_upstream_sol_outboard_parallel_area

    rmajor = From(physics)
    rminor = From(physics)
    len_plasma_sol_power_decay = FromExactly(physics.len_plasma_sol_eich13_power_decay)
    b_plasma_outboard_total = From(physics)
    b_plasma_surface_poloidal_average = From(physics)

    a_plasma_outboard_sol_eich13_parallel = OutputInto(physics)


class OutboardSOLParallelPowerFlux(WrapsFunction):
    """cottax node: `.physics.pflux_plasma_outboard_sol_parallel_mw`, the switch-
    selected power flux.
    """

    fn = outboard_sol_parallel_power_flux

    p_plasma_separatrix_mw_raw = From(physics)
    a_plasma_outboard_sol_parallel = From(physics)

    pflux_plasma_outboard_sol_parallel_mw = OutputInto(physics)


class OutboardSOLEich13ParallelPowerFlux(WrapsFunction):
    """cottax node: `.physics.pflux_plasma_outboard_sol_eich13_parallel_mw`, the Eich
    2013 power flux, unconditional.
    """

    fn = outboard_sol_eich13_parallel_power_flux

    p_plasma_separatrix_mw_raw = From(physics)
    a_plasma_outboard_sol_eich13_parallel = From(physics)

    pflux_plasma_outboard_sol_eich13_parallel_mw = OutputInto(physics)


class TokamakScrapeOffLayer(ModelNamespace):
    """`.tokamak.scrape_off_layer` -- nine node classes, eight slots, one switched."""

    eich2013_sol_power_decay_length: Eich2013SOLPowerDecayLength = (
        Eich2013SOLPowerDecayLength()
    )
    mast2014_sol_power_decay_length_1: Mast2014SOLPowerDecayLength1 = (
        Mast2014SOLPowerDecayLength1()
    )
    mast2014_sol_power_decay_length_2: Mast2014SOLPowerDecayLength2 = (
        Mast2014SOLPowerDecayLength2()
    )

    outboard_power_decay_length: OutboardSOLPowerDecayLength | None = dataclasses.field(
        kw_only=True
    )
    """`.physics.i_len_sol_outboard_power_decay` -- `1` (EICH_2013, PROCESS's default,
    `physics_variables.py:1718`) is written; `0` (USER_INPUT) is an **empty slot**
    (PROCESS has no `else` arm: the field keeps whatever it already was, so it is a run
    input with no producer); `2`/`3` (MAST) are UNPORTED one-line siblings.
    """

    upstream_sol_outboard_parallel_area: UpstreamSOLOutboardParallelArea = (
        UpstreamSOLOutboardParallelArea()
    )
    upstream_sol_outboard_eich13_parallel_area: UpstreamSOLOutboardEich13ParallelArea = (
        UpstreamSOLOutboardEich13ParallelArea()
    )
    outboard_sol_parallel_power_flux: OutboardSOLParallelPowerFlux = (
        OutboardSOLParallelPowerFlux()
    )
    outboard_sol_eich13_parallel_power_flux: OutboardSOLEich13ParallelPowerFlux = (
        OutboardSOLEich13ParallelPowerFlux()
    )
