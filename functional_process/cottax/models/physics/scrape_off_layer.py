"""Pure-functional port of `process/models/physics/scrape_off_layer.py`
(`ScrapeOffLayer`, would-be `.tokamak.scrape_off_layer`).
"""

import dataclasses

import equinox as eqx
from cottax.interfaces import Explicit, V

from functional_process.cottax.paths import physics
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


# cottax node: `calculate_eich2013_sol_power_decay_length`, unconditional.
Eich2013SOLPowerDecayLength = physics.function(
    calculate_eich2013_sol_power_decay_length,
    "len_plasma_sol_eich13_power_decay",
    reads={
        "p_plasma_separatrix_mw": V.physics.p_plasma_separatrix_mw_raw,
    },
)


# cottax node: `calculate_mast2014_sol_power_decay_length_1`, unconditional.
Mast2014SOLPowerDecayLength1 = physics.function(
    calculate_mast2014_sol_power_decay_length_1,
    "len_plasma_sol_mast14_power_decay_1",
    reads={
        "p_plasma_separatrix_mw": V.physics.p_plasma_separatrix_mw_raw,
    },
)


# cottax node: `calculate_mast2014_sol_power_decay_length_2_from_plasma_current` --
# `calculate_mast2014_sol_power_decay_length_2`, unconditional.
Mast2014SOLPowerDecayLength2 = physics.function(
    calculate_mast2014_sol_power_decay_length_2_from_plasma_current,
    "len_plasma_sol_mast14_power_decay_2",
)


# The OutboardSOLPowerDecayLength family:
# The family that owns `.physics.len_sol_outboard_power_decay`: a
# computes-then-selects switch, `.physics.i_len_sol_outboard_power_decay`.


# `i_len_sol_outboard_power_decay == EICH_2013` (1) -- PROCESS's own default
# (`physics_variables.py:1718`) and the value live on `large_tokamak_eval.IN.DAT`,
# which never sets this switch.
OutboardSOLPowerDecayLengthEich2013 = physics.function(
    outboard_sol_power_decay_length_eich2013, "len_sol_outboard_power_decay"
)


# cottax node: `calculate_upstream_sol_outboard_parallel_area` at the
# switch-selected length -> `.physics.a_plasma_outboard_sol_parallel`.
UpstreamSOLOutboardParallelArea = physics.function(
    calculate_upstream_sol_outboard_parallel_area,
    "a_plasma_outboard_sol_parallel",
    reads={
        "len_plasma_sol_power_decay": V.physics.len_sol_outboard_power_decay,
    },
)


# cottax node: `calculate_upstream_sol_outboard_parallel_area` at the Eich 2013
# length specifically -> `.physics.a_plasma_outboard_sol_eich13_parallel`.
UpstreamSOLOutboardEich13ParallelArea = physics.function(
    calculate_upstream_sol_outboard_parallel_area,
    "a_plasma_outboard_sol_eich13_parallel",
    reads={
        "len_plasma_sol_power_decay": V.physics.len_plasma_sol_eich13_power_decay,
    },
)


# cottax node: `.physics.pflux_plasma_outboard_sol_parallel_mw`, the switch-
# selected power flux.
OutboardSOLParallelPowerFlux = physics.function(
    outboard_sol_parallel_power_flux, "pflux_plasma_outboard_sol_parallel_mw"
)


# cottax node: `.physics.pflux_plasma_outboard_sol_eich13_parallel_mw`, the Eich
# 2013 power flux, unconditional.
OutboardSOLEich13ParallelPowerFlux = physics.function(
    outboard_sol_eich13_parallel_power_flux,
    "pflux_plasma_outboard_sol_eich13_parallel_mw",
)


class TokamakScrapeOffLayer(eqx.Module):
    """`.tokamak.scrape_off_layer` -- nine node classes, eight slots, one switched."""

    eich2013_sol_power_decay_length: Explicit = Eich2013SOLPowerDecayLength
    mast2014_sol_power_decay_length_1: Explicit = Mast2014SOLPowerDecayLength1
    mast2014_sol_power_decay_length_2: Explicit = Mast2014SOLPowerDecayLength2

    outboard_power_decay_length: Explicit | None = dataclasses.field(kw_only=True)
    """`.physics.i_len_sol_outboard_power_decay` -- `1` (EICH_2013, PROCESS's default,
    `physics_variables.py:1718`) is written; `0` (USER_INPUT) is an **empty slot**
    (PROCESS has no `else` arm: the field keeps whatever it already was, so it is a run
    input with no producer); `2`/`3` (MAST) are UNPORTED one-line siblings.
    """

    upstream_sol_outboard_parallel_area: Explicit = UpstreamSOLOutboardParallelArea
    upstream_sol_outboard_eich13_parallel_area: Explicit = (
        UpstreamSOLOutboardEich13ParallelArea
    )
    outboard_sol_parallel_power_flux: Explicit = OutboardSOLParallelPowerFlux
    outboard_sol_eich13_parallel_power_flux: Explicit = (
        OutboardSOLEich13ParallelPowerFlux
    )
