"""Pure-functional port of `process/models/physics/scrape_off_layer.py`
(`ScrapeOffLayer`, would-be `.tokamak.scrape_off_layer`).

Audit record: `functional_process/_audit/units/models/physics/scrape_off_layer.md`. Read
it first, especially "the RAW mint caution" and "who reads this, and who does not"
before wiring any node here to a graph.

**In scope: all five entered functions** (`tokamak_call_surface.md` §A row 2.7:
"5 entered functions, 226 entered LOC, unported" --
`models/tokamak/namespace.py:157-159`).
`ScrapeOffLayer.run()` is a thin, fully-unconditional-except-one-selection shell over its
own four `@staticmethod`s:

- `calculate_eich2013_sol_power_decay_length` -- always computed.
- `calculate_mast2014_sol_power_decay_length_1` -- always computed.
- `calculate_mast2014_sol_power_decay_length_2` -- always computed (source converts
  `plasma_current` A -> MA at the call site, `scrape_off_layer.py:36-42`; reproduced the
  same way here, inside the node/composite that calls the pure function, not inside the
  pure function itself, so its signature stays `cur_plasma_ma` exactly as PROCESS's own
  `@staticmethod` spells it).
- A **computes-then-selects** switch, `.physics.i_len_sol_outboard_power_decay`
  (`OutbordSOLPowerDecayLengthModel`): all three lengths above are computed regardless,
  and the switch only decides which one becomes `.physics.len_sol_outboard_power_decay`.
  Per the wave-1 settled policy ("computes-then-selects families -> one occupant class
  per switch value, each declaring only its own arm's reads"), the live value
  (`EICH_2013`, PROCESS's own default -- `physics_variables.py:1718`, unset on
  `large_tokamak_eval.IN.DAT`) gets one occupant that reads only the already-computed
  Eich value; `MAST_2014_1`/`MAST_2014_2` are the same one-line shape and are UNPORTED
  (not needed by the reference arm); `USER_INPUT` (0) is not a computation at all --
  PROCESS's `if`/`elif`/`elif` has no `else`, so that arm leaves the field at whatever
  the input file (or its own default) already set. See the audit record's "switches
  touched" for the full table.
- `calculate_upstream_sol_outboard_parallel_area` -- called **twice**, unconditionally:
  once with the switch-selected length (-> `.physics.a_plasma_outboard_sol_parallel`) and
  once always with the Eich length specifically, regardless of the switch (->
  `.physics.a_plasma_outboard_sol_eich13_parallel`).
- Two plain divisions with no PROCESS `calculate_*` counterpart --
  `pflux_plasma_outboard_sol_{parallel,eich13_parallel}_mw = p_plasma_separatrix_mw /
  a_plasma_outboard_sol_{parallel,eich13_parallel}` (`scrape_off_layer.py:90-98`) --
  each given its own module-level function since 2026-09-05
  (`_audit/formulas_split.md`, step 1) so that a declaration's `__call__` is a name and
  not a body; there is still no PROCESS `calculate_*` counterpart to port them from.

**The RAW mint caution.** `ScrapeOffLayer.run()` is called from
`Physics.run()` at `process/models/physics/physics.py:832`, which reads
`self.data.physics.p_plasma_separatrix_mw` at that point -- **before** the "KLUDGE:
Ensure p_plasma_separatrix_mw is continuously positive" transform at `physics.py:839-845`
runs. Every one of this file's occupants therefore reads the **raw** mint,
`.physics.p_plasma_separatrix_mw_raw` (owned by `SeparatrixPowerNonIgnited` in
`functional_process/models/physics/physics.py`, another agent's file in this wave), never
the final `.physics.p_plasma_separatrix_mw` that `PositiveSeparatrixPower` produces one
node later. Verified by reading `physics.py:800-845` directly: the raw write is at
`:800`,
`scrape_off_layer.run()` at `:832`, the kludge divide at `:843` -- the read is
textually and temporally between the two.

**Who reads this, and who does not.** `_audit/tokamak_boundary.md`'s "The 58 that are
the work list" table attributes **zero** boundary reads to `.tokamak.scrape_off_layer` --
recorded there as a real result, not a gap: nothing else on the traced graph reads what
this file produces. Confirmed independently here: the only read of any of this unit's
outputs anywhere in `process/` outside this file and its own `output()` is
`physics.py:1033` (`fio`'s `drsep / len_sol_outboard_power_decay`), which feeds
`fio`/`fli`/`flo`/`plimw`/`plomw` -- a chain that is itself consumed **only** by
`Physics.output()`'s reporting (`physics.py:2230-2253`), not by any constraint
(`process/core/solver/constraints.py` has no reference to any name this file writes) or
any other model's `run()`. So this unit's outputs are real PROCESS outputs with no
current consumer in this port's traced graph -- ported for completeness and because a
future consumer (or `output()` reporting parity) may need them, not because something
today is blocked on them.
"""

import dataclasses

import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    ModelNamespace,
    OutputInto,
)

from functional_process.models.safe_math import safe_pow
from functional_process.paths import physics
from functional_process.vocabulary import OutbordSOLPowerDecayLengthModel

# ---------------------------------------------------------------------------
# `ScrapeOffLayer`'s four `@staticmethod`s -- direct ports, source order preserved.
# Every fractional power is wrapped in `safe_pow` (`_audit/next_steps.md` §9's
# zero-derivative trap): unlike `divertor.py`'s `divwade`, none of these functions has a
# coincidental second singularity (e.g. a `0/0` division) that would make the *value*
# itself non-finite when an argument is zeroed, so nothing here gets the "wasn't flagged,
# confirmed by measurement" exemption `divertor.md` records for `rmajor**0.04` --
# `rmajor**0.04` is wrapped here for exactly that reason: zeroing `rmajor` alone leaves
# every other factor finite, so an unwrapped power would fail
# `Tier1Contract.test_gradient_finite_at_zero`.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# `calculate_scrape_off_layer` -- the composite `ScrapeOffLayer.run()` itself, all
# eight outputs. PROCESS has no bare-function equivalent (`run()` is a `self.data`
# shell), so this is the boundary the test file diffs against a `DataStructure`
# adapter, same role `calculate_confinement_time` plays for `confinement_time.py`.
# `i_len_sol_outboard_power_decay` is a switch: plain Python int, ordinary branching,
# never traced -- the harness marks it `static_argnames`.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# cottax nodes
# ---------------------------------------------------------------------------


class Eich2013SOLPowerDecayLength(ExplicitFunction):
    """cottax node: `calculate_eich2013_sol_power_decay_length`, unconditional.

    Reads the RAW separatrix-power mint -- see module docstring.
    """

    len_plasma_sol_eich13_power_decay = OutputInto(physics)

    def __call__(
        self,
        p_plasma_separatrix_mw_raw=From(physics),
        rmajor=From(physics),
        b_plasma_surface_poloidal_average=From(physics),
        aspect=From(physics),
    ):
        return calculate_eich2013_sol_power_decay_length(
            p_plasma_separatrix_mw_raw, rmajor, b_plasma_surface_poloidal_average, aspect
        )


class Mast2014SOLPowerDecayLength1(ExplicitFunction):
    """cottax node: `calculate_mast2014_sol_power_decay_length_1`, unconditional."""

    len_plasma_sol_mast14_power_decay_1 = OutputInto(physics)

    def __call__(
        self,
        p_plasma_separatrix_mw_raw=From(physics),
        b_plasma_surface_poloidal_average=From(physics),
    ):
        return calculate_mast2014_sol_power_decay_length_1(
            p_plasma_separatrix_mw_raw, b_plasma_surface_poloidal_average
        )


class Mast2014SOLPowerDecayLength2(ExplicitFunction):
    """cottax node: `calculate_mast2014_sol_power_decay_length_2`, unconditional.

    Does the A -> MA conversion PROCESS's own `run()` does at its call site
    (`scrape_off_layer.py:36-42`), so the pure function's signature stays exactly
    PROCESS's own (`cur_plasma_ma`).
    """

    len_plasma_sol_mast14_power_decay_2 = OutputInto(physics)

    def __call__(
        self,
        p_plasma_separatrix_mw_raw=From(physics),
        plasma_current=From(physics),
    ):
        return calculate_mast2014_sol_power_decay_length_2(
            p_plasma_separatrix_mw_raw, plasma_current / 1.0e6
        )


class OutboardSOLPowerDecayLength(ExplicitFunction):
    """The family that owns `.physics.len_sol_outboard_power_decay`: a
    computes-then-selects switch, `.physics.i_len_sol_outboard_power_decay`.

    All three candidate lengths above are computed unconditionally by PROCESS
    regardless of this switch's value -- the switch only decides which one this node
    passes through. Per the wave-1 settled policy each value is still its own occupant
    (a union node reading all three candidates would be over-connected the same way a
    13-branch `i_plasma_geometry` union would be, per `plasma_geometry.md`), even
    though every occupant's body is a bare passthrough of one already-computed value.
    """


def outboard_sol_power_decay_length_eich2013(len_plasma_sol_eich13_power_decay):
    """The switch's `EICH_2013` arm: pass the already-computed Eich length through."""
    return len_plasma_sol_eich13_power_decay


class OutboardSOLPowerDecayLengthEich2013(OutboardSOLPowerDecayLength):
    """`i_len_sol_outboard_power_decay == EICH_2013` (1) -- PROCESS's own default
    (`physics_variables.py:1718`) and the value live on `large_tokamak_eval.IN.DAT`,
    which never sets this switch. `MAST_2014_1`/`MAST_2014_2` are the same one-line
    passthrough shape and are UNPORTED (not needed by the reference arm); `USER_INPUT`
    is not a computation at all (see module docstring) and has no occupant to write.
    """

    len_sol_outboard_power_decay = OutputInto(physics)

    def __call__(self, len_plasma_sol_eich13_power_decay=From(physics)):
        return outboard_sol_power_decay_length_eich2013(
            len_plasma_sol_eich13_power_decay
        )


class UpstreamSOLOutboardParallelArea(ExplicitFunction):
    """cottax node: `calculate_upstream_sol_outboard_parallel_area` at the
    switch-selected length -> `.physics.a_plasma_outboard_sol_parallel`.

    Unconditional in the sense that it always runs; *which* length it reads is decided
    by `OutboardSOLPowerDecayLength`'s occupant, not by this node.
    """

    a_plasma_outboard_sol_parallel = OutputInto(physics)

    def __call__(
        self,
        rmajor=From(physics),
        rminor=From(physics),
        len_sol_outboard_power_decay=From(physics),
        b_plasma_outboard_total=From(physics),
        b_plasma_surface_poloidal_average=From(physics),
    ):
        return calculate_upstream_sol_outboard_parallel_area(
            rmajor,
            rminor,
            len_sol_outboard_power_decay,
            b_plasma_outboard_total,
            b_plasma_surface_poloidal_average,
        )


class UpstreamSOLOutboardEich13ParallelArea(ExplicitFunction):
    """cottax node: `calculate_upstream_sol_outboard_parallel_area` at the Eich 2013
    length specifically -> `.physics.a_plasma_outboard_sol_eich13_parallel`.

    Reads `len_plasma_sol_eich13_power_decay` directly, **not**
    `len_sol_outboard_power_decay` -- PROCESS computes this area for the Eich length
    unconditionally, regardless of `i_len_sol_outboard_power_decay`
    (`scrape_off_layer.py:82-88`). Declaring the direct read (rather than reusing
    `UpstreamSOLOutboardParallelArea`'s own output, which happens to equal this one
    only when the switch selects `EICH_2013`) is what keeps this node correct under
    every switch value, not just the live one.
    """

    a_plasma_outboard_sol_eich13_parallel = OutputInto(physics)

    def __call__(
        self,
        rmajor=From(physics),
        rminor=From(physics),
        len_plasma_sol_eich13_power_decay=From(physics),
        b_plasma_outboard_total=From(physics),
        b_plasma_surface_poloidal_average=From(physics),
    ):
        return calculate_upstream_sol_outboard_parallel_area(
            rmajor,
            rminor,
            len_plasma_sol_eich13_power_decay,
            b_plasma_outboard_total,
            b_plasma_surface_poloidal_average,
        )


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


class OutboardSOLParallelPowerFlux(ExplicitFunction):
    """cottax node: `.physics.pflux_plasma_outboard_sol_parallel_mw`, the switch-
    selected power flux. No PROCESS `calculate_*` counterpart -- `run()`'s own inline
    division (`scrape_off_layer.py:90-93`), reproduced as-is.
    """

    pflux_plasma_outboard_sol_parallel_mw = OutputInto(physics)

    def __call__(
        self,
        p_plasma_separatrix_mw_raw=From(physics),
        a_plasma_outboard_sol_parallel=From(physics),
    ):
        return outboard_sol_parallel_power_flux(
            p_plasma_separatrix_mw_raw, a_plasma_outboard_sol_parallel
        )


class OutboardSOLEich13ParallelPowerFlux(ExplicitFunction):
    """cottax node: `.physics.pflux_plasma_outboard_sol_eich13_parallel_mw`, the Eich
    2013 power flux, unconditional. Ports `run()`'s inline division
    (`scrape_off_layer.py:95-98`).
    """

    pflux_plasma_outboard_sol_eich13_parallel_mw = OutputInto(physics)

    def __call__(
        self,
        p_plasma_separatrix_mw_raw=From(physics),
        a_plasma_outboard_sol_eich13_parallel=From(physics),
    ):
        return outboard_sol_eich13_parallel_power_flux(
            p_plasma_separatrix_mw_raw, a_plasma_outboard_sol_eich13_parallel
        )


class TokamakScrapeOffLayer(ModelNamespace):
    """`.tokamak.scrape_off_layer` -- nine node classes, eight slots, one switched.

    **Flat, not grouped** -- `scrape_off_layer.md` open question 2 left the shape to
    the consolidation pass, and flat is what the tree's existing sub-namespace
    precedents do (`TokamakCurrentDrive`, seven flat slots): the family structure here
    is one abstract base over three one-line passthroughs, which is not the "real
    thing" (`total_process.py`'s grain rule) an intermediate namespace has to be.

    PROCESS computes all three candidate decay lengths unconditionally and the switch
    only selects which one feeds `len_sol_outboard_power_decay` (`run()`'s three length
    calls precede the `if`/`elif`/`elif`), so the three producers are unswitched
    defaults and only the selector slot answers `i_len_sol_outboard_power_decay`.
    """

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
    input with no producer); `2`/`3` (MAST) are UNPORTED one-line siblings."""

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
