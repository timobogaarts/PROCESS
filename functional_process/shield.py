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
- Shield areas (`a_shld_inboard_surface`, `a_shld_outboard_surface`,
  `a_shld_total_surface`) -- no reader on the tokamak path (see module docstring above);
  out of scope per this wave's scope discipline.
- `Shield.output()` -- reporting-only, no `data` writes.

Every switch touched by this file (`n_divertors`, the `itart`/`i_fw_blkt_vv_shape`
compound) is consumed only by *which occupant class exists*, never inside a function
body -- `_audit/naming_convention.md` § "switches are not ports".

**2026-08-27 (the D-shaped wave): both slots of this file are now total.**
`calculate_dshaped_shield_volumes` had been ported as a bare function since wave 1, with
its occupant deliberately withheld pending a decision on which key to hang it on. That
decision was already recorded -- the *existing* `indat.py::_fw_blkt_vv_shape_arm`, not a
new key -- so this wave supplies `DShapedShieldVolumes` on that key and nothing else
changes. `spherical_tokamak_eval.IN.DAT` and `st_regression.IN.DAT` (both
`i_fw_blkt_vv_shape = 1` **and** `itart = 1`) are what needed it.

The same wave lifted this file's two **private** shell helpers into the shared
`models/engineering/ivc_functions.py`. They were filed here in wave 1 with an explicit
note that they belonged there; `eshellvol` was moved to that file by a later pass and
`dshellvol` by this one, so keeping local copies beside a public pair of the same
formulas would have been the worse of the two debts. This file now imports both.
"""

from functional_process.models.engineering.ivc_functions import dshellvol, eshellvol

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

    return eshellvol(
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

    Ported for completeness in wave 1 (`plasma_geometry.md`'s reasoning for
    `sauter_geometry`) and wired to `DShapedShieldVolumes` on 2026-08-27, on the existing
    `_fw_blkt_vv_shape_arm` key. Live on `spherical_tokamak_eval.IN.DAT` and
    `st_regression.IN.DAT`; not on `large_tokamak_eval.IN.DAT`.

    **Reads-set differences from `calculate_elliptical_shield_volumes`**: gone are
    `.physics.rmajor`, `.physics.triang` and `.build.r_shld_outboard_outer`; new are
    `.build.dr_fw_inboard`/`_outboard`, `.build.dr_fw_plasma_gap_inboard`/`_outboard` and
    `.build.dr_blkt_inboard`/`_outboard`. Nine reads against the elliptical arm's seven,
    with four in common -- the same anchored-on-the-inboard-build shape
    `blanket_library.py`'s D-shaped pair has.

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

    return dshellvol(
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


def calculate_shield_volumes_dshaped(
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
    fvolsi,
    fvolso,
):
    """D-shaped shield volumes, coverage-factor-adjusted -- `Shield.run()`'s own
    `.blanket.vol_shld_inboard`/`.blanket.vol_shld_outboard`/`.fwbs.vol_shld_total` on
    the D-shaped arm.

    Composes `calculate_dshaped_shield_volumes` and
    `apply_shield_volume_coverage_factors`; introduces no new arithmetic of its own, and
    reuses the *same* coverage-factor function as the elliptical composite because
    `Shield.run()`'s coverage block (`process/models/shield.py:132-139`) sits below the
    shape branch and is shared by both arms.

    Live on `spherical_tokamak_eval.IN.DAT` and `st_regression.IN.DAT`.

    Returns
    -------
    tuple
        `(vol_shld_inboard, vol_shld_outboard, vol_shld_total)`, coverage-adjusted.
    """
    vin_raw, vout_raw, _ = calculate_dshaped_shield_volumes(
        r_shld_inboard_inner=r_shld_inboard_inner,
        dr_shld_inboard=dr_shld_inboard,
        dr_fw_inboard=dr_fw_inboard,
        dr_fw_plasma_gap_inboard=dr_fw_plasma_gap_inboard,
        rminor=rminor,
        dr_fw_plasma_gap_outboard=dr_fw_plasma_gap_outboard,
        dr_fw_outboard=dr_fw_outboard,
        dr_blkt_inboard=dr_blkt_inboard,
        dr_blkt_outboard=dr_blkt_outboard,
        dz_shld_half=dz_shld_half,
        dr_shld_outboard=dr_shld_outboard,
        dz_shld_upper=dz_shld_upper,
    )
    return apply_shield_volume_coverage_factors(vin_raw, vout_raw, fvolsi, fvolso)
