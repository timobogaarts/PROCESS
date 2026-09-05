"""Pure physics functions extracted from `models/physics/scrape_off_layer.py`.

See that file for the declarations wiring these into the graph, and
`_audit/units/models/physics/scrape_off_layer.md` for the audit record. No
graph-framework import belongs in this module -- see `_audit/formulas_split.md`.
"""

import jax.numpy as jnp

from functional_process.models.safe_math import safe_pow
from functional_process.vocabulary import OutbordSOLPowerDecayLengthModel


def calculate_eich2013_sol_power_decay_length(
    p_plasma_separatrix_mw: float,
    rmajor: float,
    b_plasma_surface_poloidal_average: float,
    aspect: float,
) -> float:
    """Eich 2013 SOL power decay length (lambda_q). Ports `ScrapeOffLayer.calculate_
    eich2013_sol_power_decay_length`, `process/models/physics/scrape_off_layer.py:
    173-216`, unchanged but for `safe_pow`.
    """
    return (
        1.35e-3
        * safe_pow(p_plasma_separatrix_mw, -0.02)
        * safe_pow(rmajor, 0.04)
        * safe_pow(b_plasma_surface_poloidal_average, -0.92)
        * safe_pow(aspect, -0.42)
    )


def calculate_mast2014_sol_power_decay_length_1(
    p_plasma_separatrix_mw: float,
    b_plasma_surface_poloidal_average: float,
) -> float:
    """MAST 2014 SOL power decay length (lambda_q), fit 1. Ports `ScrapeOffLayer.
    calculate_mast2014_sol_power_decay_length_1`, `scrape_off_layer.py:218-254`,
    unchanged but for `safe_pow`.
    """
    return (
        1.84e-3
        * safe_pow(p_plasma_separatrix_mw, 0.18)
        * safe_pow(b_plasma_surface_poloidal_average, -0.68)
    )


def calculate_mast2014_sol_power_decay_length_2(
    p_plasma_separatrix_mw: float,
    cur_plasma_ma: float,
) -> float:
    """MAST 2014 SOL power decay length (lambda_q), fit 2. Ports `ScrapeOffLayer.
    calculate_mast2014_sol_power_decay_length_2`, `scrape_off_layer.py:256-284`,
    unchanged but for `safe_pow`. `cur_plasma_ma` is `plasma_current / 1e6` at every
    call site (source: `run()`'s own inline conversion, `:36-42`) -- kept out of this
    signature, matching PROCESS's own `@staticmethod`, which also takes the already-
    converted value.
    """
    return (
        4.57e-3 * safe_pow(p_plasma_separatrix_mw, 0.22) * safe_pow(cur_plasma_ma, -0.64)
    )


def calculate_upstream_sol_outboard_parallel_area(
    rmajor: float,
    rminor: float,
    len_plasma_sol_power_decay: float,
    b_plasma_outboard_total: float,
    b_plasma_surface_poloidal_average: float,
) -> float:
    """Outboard SOL upstream parallel area (A_||,u). Ports `ScrapeOffLayer.calculate_
    upstream_sol_outboard_parallel_area`, `scrape_off_layer.py:286-328`, unchanged.

    No fractional powers, so no `safe_pow` site. `b_plasma_outboard_total == 0` is an
    unguarded division that makes the *value* itself non-finite (`b_plasma_surface_
    poloidal_average / b_plasma_outboard_total` -> inf) -- `test_outputs_finite` and
    `test_gradient_finite_at_zero`'s own "skip if the value is already non-finite"
    branch both own that case; nothing to register in `_harness/boundary.py`, since that
    register is only for a value that stays *finite* while its gradient does not.
    """
    return (
        (2 * jnp.pi * (rmajor + rminor))
        * len_plasma_sol_power_decay
        * (b_plasma_surface_poloidal_average / b_plasma_outboard_total)
    )


def calculate_scrape_off_layer(
    p_plasma_separatrix_mw_raw: float,
    rmajor: float,
    rminor: float,
    b_plasma_surface_poloidal_average: float,
    b_plasma_outboard_total: float,
    aspect: float,
    plasma_current: float,
    i_len_sol_outboard_power_decay: int,
) -> tuple[float, float, float, float, float, float, float, float]:
    """`ScrapeOffLayer.run()`, composed from the four `@staticmethod`s above plus the
    switch selection and the two divisions `run()` does inline.

    Argument named `p_plasma_separatrix_mw_raw`, not `p_plasma_separatrix_mw` --
    deliberately, per the module docstring's RAW mint caution: this is the value
    `run()` actually reads at its one call site, before `physics.py`'s positivity
    kludge.

    Only `EICH_2013`/`MAST_2014_1`/`MAST_2014_2` are supported -- the three values
    PROCESS's `if`/`elif`/`elif` actually computes something for. `USER_INPUT` (0)
    is not a computation in PROCESS (the field is left at whatever it already was),
    so there is no "answer" this composite could reproduce; passing it raises
    `ValueError` rather than silently returning a wrong `len_sol_outboard_power_decay`.
    This is a **declared divergence** from PROCESS, which does not raise for `0` --
    see the audit record's "switches touched".

    Returns
    -------
    tuple
        `(len_plasma_sol_eich13_power_decay, len_plasma_sol_mast14_power_decay_1,
        len_plasma_sol_mast14_power_decay_2, len_sol_outboard_power_decay,
        a_plasma_outboard_sol_parallel, a_plasma_outboard_sol_eich13_parallel,
        pflux_plasma_outboard_sol_parallel_mw,
        pflux_plasma_outboard_sol_eich13_parallel_mw)`.
    """
    len_eich13 = calculate_eich2013_sol_power_decay_length(
        p_plasma_separatrix_mw_raw, rmajor, b_plasma_surface_poloidal_average, aspect
    )
    len_mast14_1 = calculate_mast2014_sol_power_decay_length_1(
        p_plasma_separatrix_mw_raw, b_plasma_surface_poloidal_average
    )
    len_mast14_2 = calculate_mast2014_sol_power_decay_length_2(
        p_plasma_separatrix_mw_raw, plasma_current / 1.0e6
    )

    model = OutbordSOLPowerDecayLengthModel(int(i_len_sol_outboard_power_decay))
    if model == OutbordSOLPowerDecayLengthModel.EICH_2013:
        len_selected = len_eich13
    elif model == OutbordSOLPowerDecayLengthModel.MAST_2014_1:
        len_selected = len_mast14_1
    elif model == OutbordSOLPowerDecayLengthModel.MAST_2014_2:
        len_selected = len_mast14_2
    else:
        raise ValueError(
            f"i_len_sol_outboard_power_decay == {model!r} is not ported -- USER_INPUT "
            "is not a computation in PROCESS (the field is left as-is), so there is no "
            "answer for this composite to reproduce"
        )

    area_selected = calculate_upstream_sol_outboard_parallel_area(
        rmajor,
        rminor,
        len_selected,
        b_plasma_outboard_total,
        b_plasma_surface_poloidal_average,
    )
    area_eich13 = calculate_upstream_sol_outboard_parallel_area(
        rmajor,
        rminor,
        len_eich13,
        b_plasma_outboard_total,
        b_plasma_surface_poloidal_average,
    )

    pflux_selected = p_plasma_separatrix_mw_raw / area_selected
    pflux_eich13 = p_plasma_separatrix_mw_raw / area_eich13

    return (
        len_eich13,
        len_mast14_1,
        len_mast14_2,
        len_selected,
        area_selected,
        area_eich13,
        pflux_selected,
        pflux_eich13,
    )


def outboard_sol_power_decay_length_eich2013(len_plasma_sol_eich13_power_decay):
    """The switch's `EICH_2013` arm: pass the already-computed Eich length through."""
    return len_plasma_sol_eich13_power_decay


def outboard_sol_parallel_power_flux(
    p_plasma_separatrix_mw_raw, a_plasma_outboard_sol_parallel
):
    """The switch-selected outboard SOL parallel power flux (MW/m^2).

    No PROCESS `calculate_*` counterpart -- `run()`'s own inline division
    (`scrape_off_layer.py:90-93`), reproduced as-is.
    """
    return p_plasma_separatrix_mw_raw / a_plasma_outboard_sol_parallel


def outboard_sol_eich13_parallel_power_flux(
    p_plasma_separatrix_mw_raw, a_plasma_outboard_sol_eich13_parallel
):
    """The unconditional Eich 2013 outboard SOL parallel power flux (MW/m^2).

    No PROCESS `calculate_*` counterpart -- `run()`'s own inline division
    (`scrape_off_layer.py:95-98`), reproduced as-is.
    """
    return p_plasma_separatrix_mw_raw / a_plasma_outboard_sol_eich13_parallel
