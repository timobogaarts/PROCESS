"""Pure-functional port of `process/models/shield.py` (partial -- see "not ported").

Audit record: `functional_process/_audit/units/models/shield.md`. Read it first,
especially "reachability" (why `Shield` is ported at all: `.fwbs.vol_shld_total` is read
by `blankets/hcpb.py`'s shield-mass calculation, which is on the tokamak call surface for
`large_tokamak_eval.IN.DAT`) and "suspected defect" (the raw `vol_shld_total` from the
staticmethod is computed then immediately discarded by `Shield.run()`'s own
coverage-factor overwrite -- ported faithfully, not fixed).

**Scope of this pass.** The minimal closure that produces `.fwbs.vol_shld_total` on
`large_tokamak_eval.IN.DAT`'s own configuration: `itart=0`, `i_fw_blkt_vv_shape=2`
(`ELLIPTICAL_SHAPED`, the dataclass default, unset in the IN.DAT), `n_divertors=1`
(`i_single_null=1` resolved by `process/core/init.py:617` before any model runs). Areas
(`.build.a_shld_*`, `.build.a_shld_total_surface`) are **not** ported: grep confirms
their only readers outside `shield.py` are `models/stellarator/stellarator.py`, which is
out of tokamak scope -- porting them would be dead-node work per the wave's scope
discipline.

**Not ported, and why:**

- `n_divertors == 2` (double-null) half-height branch -- ported as a function
  (`calculate_shield_half_height_double_null`, trivial, and PROCESS's own
  `calculate_shield_half_height` already unifies both branches under one signature so
  porting it cost nothing extra) and wired as a full occupant, since both values of this
  binary switch are cheap to support. Not live on `large_tokamak_eval` (`n_divertors=1`).
- The D-shaped shield-volume arm (`itart == 1 or i_fw_blkt_vv_shape == D_SHAPED`) --
  `calculate_dshaped_shield_volumes` is ported as a function, verbatim, for
  completeness (same reasoning `plasma_geometry.md` used for `sauter_geometry`), but
  **not wired to an occupant class**: not live here (`itart=0`,
  `i_fw_blkt_vv_shape=ELLIPTICAL_SHAPED`), and the audit record notes this switch is the
  same `itart` x `i_fw_blkt_vv_shape` joint key `unit_registry.md` says
  `indat.py::_fw_blkt_vv_shape_arm` already owns for four other slots
  (`blanket_library.md`, `fw.md`, `vacuum.md`) -- wiring it here independently would
  risk drifting from that joint key rather than reusing it.
- Shield areas (`a_shld_inboard_surface`, `a_shld_outboard_surface`,
  `a_shld_total_surface`) -- no reader on the tokamak path (see module docstring above);
  out of scope per this wave's scope discipline.
- `Shield.output()` -- reporting-only, no `data` writes.

Every switch touched by this file (`n_divertors`, the `itart`/`i_fw_blkt_vv_shape`
compound) is consumed only by *which occupant class exists*, never inside a function
body -- `_audit/naming_convention.md` § "switches are not ports".
"""

import dataclasses

import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    ModelNamespace,
    OutputInto,
)

from functional_process.paths import blanket, build, divertor, fwbs, physics

# ---------------------------------------------------------------------------
# `calculate_shield_half_height` -- ports `Shield.calculate_shield_half_height`,
# `process/models/shield.py:143-197`, split per the `n_divertors` branch (same
# occupant-per-branch treatment `divertor.md` gave `divwade`'s own `n_divertors` read:
# "a switch read to branch selects an occupant").
# ---------------------------------------------------------------------------


def calculate_shield_half_height_double_null(
    z_plasma_xpoint_lower, dz_xpoint_divertor, dz_divertor
):
    """Shield half-height, `n_divertors == 2` (double-null) branch.

    Ports `Shield.calculate_shield_half_height`'s `if n_divertors == 2` arm,
    `process/models/shield.py:184-185`: `z_top = z_bottom`, so the returned
    `0.5 * (z_top + z_bottom)` reduces to `z_bottom` itself. Not live on
    `large_tokamak_eval.IN.DAT` (`n_divertors=1`); ported since PROCESS's own
    `calculate_shield_half_height` already unifies both branches in one signature.
    """
    return z_plasma_xpoint_lower + dz_xpoint_divertor + dz_divertor


def calculate_shield_half_height_single_null(
    z_plasma_xpoint_lower,
    dz_xpoint_divertor,
    dz_divertor,
    z_plasma_xpoint_upper,
    dr_fw_plasma_gap_inboard,
    dr_fw_plasma_gap_outboard,
    dr_fw_inboard,
    dr_fw_outboard,
    dz_blkt_upper,
):
    """Shield half-height, `n_divertors != 2` branch -- live on
    `large_tokamak_eval.IN.DAT` (`n_divertors=1`, resolved from `i_single_null=1` at
    `process/core/init.py:617`, before any model in `_call_models_once` runs).

    Ports `Shield.calculate_shield_half_height`'s `else` arm,
    `process/models/shield.py:186-194`, unchanged.
    """
    z_bottom = z_plasma_xpoint_lower + dz_xpoint_divertor + dz_divertor
    z_top = z_plasma_xpoint_upper + 0.5 * (
        dr_fw_plasma_gap_inboard
        + dr_fw_plasma_gap_outboard
        + dr_fw_inboard
        + dr_fw_outboard
    )
    z_top += dz_blkt_upper
    return 0.5 * (z_top + z_bottom)


# ---------------------------------------------------------------------------
# `eshellvol`/`dshellvol` -- ported inline from
# `process/models/engineering/ivc_functions.py:170-306`, needed for closure (that file
# is otherwise out of this unit's scope -- no registry row, reached by `fw.py`/
# `vacuum.py` too). Private: not nodes, not claiming the rest of that file.
# ---------------------------------------------------------------------------


def _eshellvol(rshell, rmini, rmino, zminor, drin, drout, dz):
    """Ports `engineering/ivc_functions.py::eshellvol`, unchanged."""
    a = rmini
    b = zminor
    elong = b / a
    v1 = 2.0 * jnp.pi * elong * (0.5 * jnp.pi * rshell * a**2 - (2.0 / 3.0) * a**3)

    a = rmini + drin
    b = zminor + dz
    elong = b / a
    v2 = 2.0 * jnp.pi * elong * (0.5 * jnp.pi * rshell * a**2 - (2.0 / 3.0) * a**3)

    vin = v2 - v1

    a = rmino
    b = zminor
    elong = b / a
    v1 = 2.0 * jnp.pi * elong * (0.5 * jnp.pi * rshell * a**2 + (2.0 / 3.0) * a**3)

    a = rmino + drout
    b = zminor + dz
    elong = b / a
    v2 = 2.0 * jnp.pi * elong * (0.5 * jnp.pi * rshell * a**2 + (2.0 / 3.0) * a**3)

    vout = v2 - v1

    return vin, vout, vin + vout


def _dshellvol(rmajor, rminor, zminor, drin, drout, dz):
    """Ports `engineering/ivc_functions.py::dshellvol`, unchanged."""
    vin = 2.0 * (zminor + dz) * jnp.pi * (rmajor**2 - (rmajor - drin) ** 2)

    a = rminor
    b = zminor
    elong = b / a
    v1 = 2.0 * jnp.pi * elong * (0.5 * jnp.pi * rmajor * a**2 + (2.0 / 3.0) * a**3)

    a = rminor + drout
    b = zminor + dz
    elong = b / a
    v2 = 2.0 * jnp.pi * elong * (0.5 * jnp.pi * rmajor * a**2 + (2.0 / 3.0) * a**3)

    vout = v2 - v1

    return vin, vout, vin + vout


# ---------------------------------------------------------------------------
# `calculate_elliptical_shield_volumes`/`calculate_dshaped_shield_volumes` -- already
# `@staticmethod`s in `process/`, ported verbatim (`np.` -> `jnp.`). Zero `self.data`
# access in the source.
# ---------------------------------------------------------------------------


def calculate_elliptical_shield_volumes(
    r_shld_inboard_inner,
    r_shld_outboard_outer,
    rmajor,
    triang,
    dr_shld_inboard,
    rminor,
    dz_shld_half,
    dr_shld_outboard,
    dz_shld_upper,
):
    """Raw (pre-coverage-factor) elliptical shield volumes. Ports
    `Shield.calculate_elliptical_shield_volumes`, `process/models/shield.py:327-385`,
    unchanged. Live on `large_tokamak_eval.IN.DAT` (`itart=0`,
    `i_fw_blkt_vv_shape=ELLIPTICAL_SHAPED`).

    Returns
    -------
    tuple
        `(vol_shld_inboard, vol_shld_outboard, vol_shld_total)`, all pre-coverage-factor
        -- see `apply_shield_volume_coverage_factors`.
    """
    r_1 = rmajor - rminor * triang
    r_2 = r_1 - r_shld_inboard_inner - dr_shld_inboard
    r_3 = r_shld_outboard_outer - r_1 - dr_shld_outboard

    return _eshellvol(
        rshell=r_1,
        rmini=r_2,
        rmino=r_3,
        zminor=dz_shld_half,
        drin=dr_shld_inboard,
        drout=dr_shld_outboard,
        dz=dz_shld_upper,
    )


def calculate_dshaped_shield_volumes(
    r_shld_inboard_inner,
    dr_shld_inboard,
    dr_fw_inboard,
    dr_fw_plasma_gap_inboard,
    rminor,
    dr_fw_plasma_gap_outboard,
    dr_fw_outboard,
    dr_blkt_inboard,
    dr_blkt_outboard,
    dz_shld_half,
    dr_shld_outboard,
    dz_shld_upper,
):
    """Raw (pre-coverage-factor) D-shaped shield volumes. Ports `Shield.
    calculate_dshaped_shield_volumes`, `process/models/shield.py:199-267`, unchanged.

    Ported for completeness (`plasma_geometry.md`'s reasoning for `sauter_geometry`);
    **not wired to an occupant class in this pass** -- not live on
    `large_tokamak_eval.IN.DAT`.

    Returns
    -------
    tuple
        `(vol_shld_inboard, vol_shld_outboard, vol_shld_total)`, pre-coverage-factor.
    """
    r_1 = r_shld_inboard_inner + dr_shld_inboard
    r_2 = (
        dr_fw_inboard
        + dr_fw_plasma_gap_inboard
        + 2.0 * rminor
        + dr_fw_plasma_gap_outboard
        + dr_fw_outboard
    )
    r_2 = dr_blkt_inboard + r_2 + dr_blkt_outboard

    return _dshellvol(
        rmajor=r_1,
        rminor=r_2,
        zminor=dz_shld_half,
        drin=dr_shld_inboard,
        drout=dr_shld_outboard,
        dz=dz_shld_upper,
    )


# ---------------------------------------------------------------------------
# New: extracted from `Shield.run()`'s coverage-factor block,
# `process/models/shield.py:132-140`. No PROCESS function of this shape -- inline code
# in `run()`, shared by both the elliptical and D-shaped arms.
# ---------------------------------------------------------------------------


def apply_shield_volume_coverage_factors(
    vol_shld_inboard, vol_shld_outboard, fvolsi, fvolso
):
    """Apply shield coverage factors and recompute the total.

    Ports `Shield.run()`'s coverage-factor block, `process/models/shield.py:132-139`
    (volume half only -- the area half, `:121-130`, is out of this unit's scope, no
    reader on the tokamak path).

    **PROCESS writes `.fwbs.vol_shld_total` twice per `run()` call**: once from
    `calculate_{elliptical,dshaped}_shield_volumes`'s own raw sum (the third element of
    its returned tuple, assigned at `shield.py:107`/`:72` then **immediately
    discarded**), and once here, from the coverage-corrected `vol_shld_inboard +
    vol_shld_outboard`. This function ports the second (surviving) write; the raw sum is
    not carried forward at all, faithfully reproducing the discard.
    """
    vol_shld_inboard = fvolsi * vol_shld_inboard
    vol_shld_outboard = fvolso * vol_shld_outboard
    vol_shld_total = vol_shld_inboard + vol_shld_outboard
    return vol_shld_inboard, vol_shld_outboard, vol_shld_total


def calculate_shield_volumes_elliptical(
    r_shld_inboard_inner,
    r_shld_outboard_outer,
    rmajor,
    triang,
    dr_shld_inboard,
    rminor,
    dz_shld_half,
    dr_shld_outboard,
    dz_shld_upper,
    fvolsi,
    fvolso,
):
    """Elliptical shield volumes, coverage-factor-adjusted -- `Shield.run()`'s own
    `.blanket.vol_shld_inboard`/`.blanket.vol_shld_outboard`/`.fwbs.vol_shld_total`.

    Composes `calculate_elliptical_shield_volumes` and
    `apply_shield_volume_coverage_factors`; introduces no new arithmetic of its own.
    Live on `large_tokamak_eval.IN.DAT`.

    Returns
    -------
    tuple
        `(vol_shld_inboard, vol_shld_outboard, vol_shld_total)`, coverage-adjusted.
    """
    vin_raw, vout_raw, _ = calculate_elliptical_shield_volumes(
        r_shld_inboard_inner=r_shld_inboard_inner,
        r_shld_outboard_outer=r_shld_outboard_outer,
        rmajor=rmajor,
        triang=triang,
        dr_shld_inboard=dr_shld_inboard,
        rminor=rminor,
        dz_shld_half=dz_shld_half,
        dr_shld_outboard=dr_shld_outboard,
        dz_shld_upper=dz_shld_upper,
    )
    return apply_shield_volume_coverage_factors(vin_raw, vout_raw, fvolsi, fvolso)


# ---------------------------------------------------------------------------
# cottax nodes
# ---------------------------------------------------------------------------


class ShieldHalfHeight(ExplicitFunction):
    """The family that owns `.blanket.dz_shld_half`: one occupant per `n_divertors`
    branch of `Shield.calculate_shield_half_height`. Both values are cheap enough to
    support fully (2-line trivial branch vs. the general one), so unlike most switches
    in this wave neither is left UNPORTED.
    """


class DoubleNullShieldHalfHeight(ShieldHalfHeight):
    """`n_divertors == 2`. Not live on `large_tokamak_eval.IN.DAT` (`n_divertors=1`)."""

    dz_shld_half = OutputInto(blanket)

    def __call__(
        self,
        z_plasma_xpoint_lower=From(build),
        dz_xpoint_divertor=From(build),
        dz_divertor=From(divertor),
    ):
        return calculate_shield_half_height_double_null(
            z_plasma_xpoint_lower, dz_xpoint_divertor, dz_divertor
        )


class SingleNullShieldHalfHeight(ShieldHalfHeight):
    """`n_divertors != 2` -- the arm `large_tokamak_eval.IN.DAT` takes
    (`n_divertors=1`).
    """

    dz_shld_half = OutputInto(blanket)

    def __call__(
        self,
        z_plasma_xpoint_lower=From(build),
        dz_xpoint_divertor=From(build),
        dz_divertor=From(divertor),
        z_plasma_xpoint_upper=From(build),
        dr_fw_plasma_gap_inboard=From(build),
        dr_fw_plasma_gap_outboard=From(build),
        dr_fw_inboard=From(build),
        dr_fw_outboard=From(build),
        dz_blkt_upper=From(build),
    ):
        return calculate_shield_half_height_single_null(
            z_plasma_xpoint_lower,
            dz_xpoint_divertor,
            dz_divertor,
            z_plasma_xpoint_upper,
            dr_fw_plasma_gap_inboard,
            dr_fw_plasma_gap_outboard,
            dr_fw_inboard,
            dr_fw_outboard,
            dz_blkt_upper,
        )


class ShieldVolumes(ExplicitFunction):
    """The family that owns `.blanket.vol_shld_inboard`/`.blanket.vol_shld_outboard`/
    `.fwbs.vol_shld_total`: one occupant per arm of the compound switch
    `itart == 1 or i_fw_blkt_vv_shape == D_SHAPED` (`process/models/shield.py:48-51`).

    **This pass ports only the elliptical (`False`) arm** -- `large_tokamak_eval.IN.DAT`
    has `itart=0` and leaves `i_fw_blkt_vv_shape` at its default (`2`,
    `ELLIPTICAL_SHAPED`), so neither half of the disjunction is true. The D-shaped arm
    (`True`) is UNPORTED as an occupant (the function itself is ported, see
    `calculate_dshaped_shield_volumes`): not live here, and per the audit record this is
    the same `itart` x `i_fw_blkt_vv_shape` joint key `indat.py::_fw_blkt_vv_shape_arm`
    already owns for four other slots -- wiring it independently here risks drifting from
    that key rather than reusing it. Flagged in the wave-1 report, not decided here.
    """


class EllipticalShieldVolumes(ShieldVolumes):
    """`itart != 1 and i_fw_blkt_vv_shape != D_SHAPED` -- the arm
    `large_tokamak_eval.IN.DAT` takes.
    """

    vol_shld_inboard = OutputInto(blanket)
    vol_shld_outboard = OutputInto(blanket)
    vol_shld_total = OutputInto(fwbs)

    def __call__(
        self,
        r_shld_inboard_inner=From(build),
        r_shld_outboard_outer=From(build),
        rmajor=From(physics),
        triang=From(physics),
        dr_shld_inboard=From(build),
        rminor=From(physics),
        dz_shld_half=From(blanket),
        dr_shld_outboard=From(build),
        dz_shld_upper=From(build),
        fvolsi=From(fwbs),
        fvolso=From(fwbs),
    ):
        return calculate_shield_volumes_elliptical(
            r_shld_inboard_inner,
            r_shld_outboard_outer,
            rmajor,
            triang,
            dr_shld_inboard,
            rminor,
            dz_shld_half,
            dr_shld_outboard,
            dz_shld_upper,
            fvolsi,
            fvolso,
        )


class TokamakShield(ModelNamespace):
    """`.tokamak.shield` -- two slots, both switched.

    The volumes slot's D-shaped arm is keyed on the **existing**
    `indat.py::_fw_blkt_vv_shape_arm` joint predicate (`itart == 1 or
    i_fw_blkt_vv_shape == D_SHAPED`), the same key `blanket_library.md`/`fw.md`/
    `vacuum.md` already share -- `shield.md`'s "switches touched" table says this
    file's split *"should join that key at consolidation, not mint an independent
    one"*, and it does.
    """

    half_height: ShieldHalfHeight = dataclasses.field(kw_only=True)
    """`.divertor.n_divertors` (derived from `i_single_null` by `indat._n_divertors`)
    -- both values of this binary switch have written occupants, `1` (single null,
    live) and `2` (double null)."""

    volumes: ShieldVolumes = dataclasses.field(kw_only=True)
    """`_fw_blkt_vv_shape_arm` -- the elliptical arm (`1`) is written; the D-shaped
    arm (`0`) is UNPORTED (`calculate_dshaped_shield_volumes` is ported as a pure
    function, no occupant -- `shield.md` "ported" table)."""
