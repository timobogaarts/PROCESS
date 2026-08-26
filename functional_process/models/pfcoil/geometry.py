"""Where the CS and the PF coils are: cross-sections, filament placement, coil centres.

Audit record: `functional_process/_audit/units/models/pfcoil/geometry.md`.

Four units, all straight-line algebra with no iteration and no call into another model:

- `calculate_cs_geometry` -- `CSCoil.calculate_cs_geometry`
  (`process/models/pfcoil.py:3005-3072`), already a `@staticmethod` with explicit
  arguments; the port is a signature promotion plus `np.` -> `jnp.`.
- `place_cs_filaments` -- `CSCoil.place_cs_filaments` (`:3151-3226`), likewise.
- `calculate_pf_coil_group_positions` -- the `i_pf_location` dispatch loop of
  `pfcoil()` (`:247-354`) collapsed to this run's four groups, calling
  `place_pf_above_tf` (`:1178-1263`) for groups 0-1 and `place_pf_outside_tf`
  (`:1265-1343`) for groups 2-3.
- `calculate_pf_coil_positions` -- `pfcoil()`'s group-array flattening (`:663-672`)
  together with the CS's own slot (`:176-194`).

**`top_bottom` is structural, not an input.** `pfcoil()` initialises `top_bottom = 1`
(`:127`) and `place_pf_above_tf` flips it every time it places a coil (`:1254-1261`), so
which side of the midplane an `i_pf_location = 2` group lands on is decided by that
group's *position in the group ordering*, not by any field. On this run the two such
groups hold one coil each, so group 0 goes above and group 1 below. That is a fact about
the assembled graph, not a value flowing along an edge, and it is baked into
`calculate_pf_coil_group_positions` below rather than passed in -- a third
`i_pf_location = 2` group would be a different occupant, not a different argument.

**Switches this file bakes in**, all at their `large_tokamak_eval.IN.DAT` value; every
other value is UNPORTED with its reason in `geometry.md`:

| switch | value here |
|---|---|
| `.pf_coil.i_pf_location` | `(2, 2, 3, 3)` -- above TF x2, then outside TF x2 |
| `.physics.itart` / `.physics.itartpf` | `0` / `0` (not a spherical tokamak) |
| `.tfcoil.i_tf_shape` | `1` = `D_SHAPE` (not `PICTURE_FRAME`) |
| `.pf_coil.i_r_pf_outside_tf_placement` | `0` -- radius follows the TF curve |
"""

import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.models.pfcoil import (
    CS_INDEX,
    N_COILS_IN_GROUP,
    N_CS_FILAMENTS,
    N_PF_GROUPS,
    N_PF_GROUPS_MAX,
    NFXF,
    NGC2,
)
from functional_process.paths import build, pf_coil, physics, superconducting_tfcoil

N_PF_COILS_IN_GROUP_MAX = 2
"""`pfcoil_variables.N_PF_COILS_IN_GROUP_MAX` -- the second axis of every
`*_group_array`. A group holding more than two coils is rejected by PROCESS itself
(`pfcoil.py:142-148`)."""

_PF_ABOVE_TF_Z_CLEARANCE = 0.86
"""The bare `0.86` m in `place_pf_above_tf` (`pfcoil.py:1255`, `:1259`) -- clearance
between the top of the TF coil and the PF coil centre. Unexplained in the source; named
here rather than repeated twice."""


def calculate_cs_geometry(z_tf_inside_half, f_z_cs_tf_internal, dr_cs, dr_cs_bore):
    """Central Solenoid cross-section and edge coordinates.

    Ports `CSCoil.calculate_cs_geometry`, `process/models/pfcoil.py:3005-3072`,
    arithmetic unchanged; PROCESS's `CSGeometry` dataclass return becomes a tuple in the
    source's own field order, since a `CallableNode` binds returns positionally.

    `dr_cs_full` is `2 * r_cs_coil_outer` in the source (`:3052`) -- twice the *outer
    radius*, not the coil's radial thickness, despite the `dr_` prefix. Ported verbatim;
    flagged in `geometry.md` rather than corrected, since it is stored and reported.

    Parameters
    ----------
    z_tf_inside_half :
        Half-height of the TF coil bore (m). `.build.z_tf_inside_half`.
    f_z_cs_tf_internal :
        CS height as a fraction of the TF bore height. `.pf_coil.f_z_cs_tf_internal`.
    dr_cs :
        Radial thickness of the CS (m). `.build.dr_cs`.
    dr_cs_bore :
        Radius of the CS bore (m). `.build.dr_cs_bore`.

    Returns
    -------
    tuple
        `(z_cs_coil_upper, z_cs_coil_lower, r_cs_coil_middle, r_cs_middle,
        z_cs_coil_middle, r_cs_coil_outer, r_cs_coil_inner, a_cs_poloidal,
        a_cs_toroidal, dz_cs_full, dr_cs_full)`.
    """
    r_cs_middle = dr_cs_bore + (0.5 * dr_cs)
    z_cs_half = z_tf_inside_half * f_z_cs_tf_internal
    dz_cs_full = 2.0 * z_cs_half

    z_cs_coil_upper = z_cs_half
    z_cs_coil_lower = -z_cs_coil_upper

    r_cs_coil_middle = r_cs_middle
    z_cs_coil_middle = jnp.zeros_like(r_cs_middle)

    r_cs_coil_outer = r_cs_middle + 0.5 * dr_cs
    r_cs_coil_inner = r_cs_coil_outer - dr_cs

    dr_cs_full = 2.0 * r_cs_coil_outer

    a_cs_poloidal = dz_cs_full * dr_cs
    a_cs_toroidal = jnp.pi * (r_cs_coil_outer**2 - r_cs_coil_inner**2)

    return (
        z_cs_coil_upper,
        z_cs_coil_lower,
        r_cs_coil_middle,
        r_cs_middle,
        z_cs_coil_middle,
        r_cs_coil_outer,
        r_cs_coil_inner,
        a_cs_poloidal,
        a_cs_toroidal,
        dz_cs_full,
        dr_cs_full,
    )


def place_cs_filaments(
    r_cs_middle, z_cs_inside_half, c_cs_flat_top_end, f_j_cs_start_pulse_end_flat_top
):
    """The CS split into `2 * N_CS_FILAMENTS` current filaments, symmetric about z = 0.

    Ports `CSCoil.place_cs_filaments`, `process/models/pfcoil.py:3151-3226`, arithmetic
    unchanged. Two shape deviations, both recorded in `geometry.md`:

    - `n_cs_current_filaments` and `nfxf` are not arguments. They are `7` and `14` here,
      module constants: a filament count is a discretisation choice read once to fix the
      array layout, which `_audit/naming_convention.md` § "Switches are not ports" puts
      on the graph assembler rather than on an edge.
    - The returned arrays are `NFXF` long, not `NFIXMX = 64`. PROCESS allocates the
      padded array and fills the first `nfxf` entries; the remaining 50 are structural
      zeros that every consumer slices away (`peak_b_field_at_pf_coil` and `fixb` both
      take `[:nfxf]`/`[:kk]`), so carrying them would only cost fifty extra fuzz columns.

    Parameters
    ----------
    r_cs_middle :
        CS mean radius (m) -- every filament sits here.
    z_cs_inside_half :
        Half-height of the CS (m), i.e. `dz_cs_full / 2`.
    c_cs_flat_top_end :
        Total CS current at end of flat-top (A), signed.
    f_j_cs_start_pulse_end_flat_top :
        Ratio of CS current density at beginning of pulse to end of flat-top.

    Returns
    -------
    tuple
        `(r_pf_cs_current_filaments, z_pf_cs_current_filaments,
        c_pf_cs_current_filaments)`, each `NFXF` long: upper half first, then its mirror.
    """
    upper_z = z_cs_inside_half / N_CS_FILAMENTS * (jnp.arange(N_CS_FILAMENTS) + 0.5)
    z_filaments = jnp.concatenate([upper_z, -upper_z])

    r_filaments = jnp.full(NFXF, r_cs_middle)

    current = -c_cs_flat_top_end / NFXF * f_j_cs_start_pulse_end_flat_top
    c_filaments = jnp.full(NFXF, current)

    return r_filaments, z_filaments, c_filaments


def calculate_pf_coil_group_positions(
    rmajor,
    rminor,
    triang,
    rpf2,
    z_tf_top,
    dz_tf_upper_lower_midplane,
    zref,
    r_pf_outside_tf_midplane,
):
    """Coil centres by `(group, coil)` for `i_pf_location = (2, 2, 3, 3)`.

    Ports `pfcoil()`'s placement loop (`process/models/pfcoil.py:247-354`) for this run's
    four groups, i.e. `place_pf_above_tf` (`:1178-1263`) for groups 0 and 1 and
    `place_pf_outside_tf` (`:1265-1343`) for groups 2 and 3, at `itart = 0`,
    `i_tf_shape != PICTURE_FRAME` and `i_r_pf_outside_tf_placement = 0`. The
    `top_bottom` toggle is resolved at trace time -- see the module docstring.

    `place_pf_outside_tf`'s `np.isinf` kludge (`:1334-1339`, log-and-set-`1e10`) is kept
    as a `jnp.where`; the `logger.error` beside it is pure reporting and is dropped, the
    same treatment `models/structure.py` gives the identical kludge in `coldmass`.

    Parameters
    ----------
    rmajor, rminor, triang :
        Plasma major radius (m), minor radius (m) and triangularity. `.physics.*`.
    rpf2 :
        Radial offset of an `i_pf_location = 2` group, in units of `triang * rminor`.
        `.pf_coil.rpf2`.
    z_tf_top :
        Height of the top of the TF coil (m). `.build.z_tf_top`.
    dz_tf_upper_lower_midplane :
        Up/down asymmetry of the TF coil about the midplane (m).
        `.build.dz_tf_upper_lower_midplane`.
    zref :
        Per-group vertical placement ratio, four entries. For an
        `i_pf_location = 3` group this is the coil height in units of `rminor`.
        `.pf_coil.zref[:4]`.
    r_pf_outside_tf_midplane :
        Radius at which an `i_pf_location = 3` coil would sit on the midplane (m).
        `.pf_coil.r_pf_outside_tf_midplane`.

    Returns
    -------
    tuple
        `(r_pf_coil_middle_group_array, z_pf_coil_middle_group_array)`, each
        `(N_PF_GROUPS, N_PF_COILS_IN_GROUP_MAX)`. A group holding one coil leaves its
        second column at zero, exactly as PROCESS's pre-zeroed array does.
    """
    r_group = jnp.zeros((N_PF_GROUPS, N_PF_COILS_IN_GROUP_MAX))
    z_group = jnp.zeros((N_PF_GROUPS, N_PF_COILS_IN_GROUP_MAX))

    # Groups 0 and 1: `i_pf_location = 2`, stacked above/below the TF coil.
    r_above_tf = rmajor + rpf2 * triang * rminor
    z_above_tf_top = z_tf_top + _PF_ABOVE_TF_Z_CLEARANCE
    z_above_tf_bottom = -1.0 * (
        z_tf_top - dz_tf_upper_lower_midplane + _PF_ABOVE_TF_Z_CLEARANCE
    )
    r_group = r_group.at[0, 0].set(r_above_tf).at[1, 0].set(r_above_tf)
    z_group = z_group.at[0, 0].set(z_above_tf_top).at[1, 0].set(z_above_tf_bottom)

    # Groups 2 and 3: `i_pf_location = 3`, radially outside the TF coil, one coil above
    # and one below the midplane, radius following the D-shaped TF curve.
    for group in (2, 3):
        for coil in range(N_COILS_IN_GROUP[group]):
            sign = 1.0 if coil == 0 else -1.0
            z = rminor * zref[group] * sign
            r_raw = jnp.sqrt(r_pf_outside_tf_midplane**2 - z**2)
            r = jnp.where(jnp.isinf(r_raw), 1e10, r_raw)
            z_group = z_group.at[group, coil].set(z)
            r_group = r_group.at[group, coil].set(r)

    return r_group, z_group


def calculate_pf_coil_positions(
    r_pf_coil_middle_group_array, z_pf_coil_middle_group_array, r_cs_middle
):
    """Coil centres flattened out of the group arrays, with the CS in its own slot.

    Ports `pfcoil()`'s `ncl` loop (`process/models/pfcoil.py:663-672`) together with the
    CS's writes into the same two arrays (`:182`, `:186-188`). Group-then-coil order,
    which is what fixes every per-coil index in this package.

    Returns
    -------
    tuple
        `(r_pf_coil_middle, z_pf_coil_middle)`, seven entries each -- six PF coils then
        the CS.
    """
    r_flat = [
        r_pf_coil_middle_group_array[group, coil]
        for group in range(N_PF_GROUPS)
        for coil in range(N_COILS_IN_GROUP[group])
    ]
    z_flat = [
        z_pf_coil_middle_group_array[group, coil]
        for group in range(N_PF_GROUPS)
        for coil in range(N_COILS_IN_GROUP[group])
    ]
    r_flat.append(jnp.asarray(r_cs_middle))
    z_flat.append(jnp.zeros_like(jnp.asarray(r_cs_middle)))
    return jnp.stack(r_flat), jnp.stack(z_flat)


class CSCoilGeometry(ExplicitFunction):
    """cottax node: `.tokamak.cs_coil.geometry`. Owns the CS's own cross-section and
    edge coordinates -- the ten scalar `.pf_coil.*cs*` fields `calculate_cs_geometry`
    produces.

    PROCESS writes these twice per pass, once at the head of `pfcoil()`
    (`process/models/pfcoil.py:169-198`) and once at the head of `ohcalc()`
    (`:3230-3259`), from the same four inputs with the same expression. One node, not
    two: the second write is idempotent, so there is nothing for a second occupant to
    do that this one does not.

    The four `.pf_coil.{r,z}_pf_coil_*[6]` array slots those same two blocks fill are
    **not** owned here; `masses.py`'s `PFCoilSizes` owns those arrays whole, reading the
    scalars this node produces. Splitting them any other way would leave one array with
    two owners.
    """

    z_cs_upper = OutputInto(pf_coil)
    z_cs_lower = OutputInto(pf_coil)
    r_cs_middle = OutputInto(pf_coil)
    z_cs_middle = OutputInto(pf_coil)
    r_cs_outer = OutputInto(pf_coil)
    r_cs_inner = OutputInto(pf_coil)
    a_cs_poloidal = OutputInto(pf_coil)
    a_cs_toroidal = OutputInto(pf_coil)
    dz_cs_full = OutputInto(pf_coil)
    dr_cs_full = OutputInto(pf_coil)

    def __call__(
        self,
        z_tf_inside_half=From(build),
        f_z_cs_tf_internal=From(pf_coil),
        dr_cs=From(build),
        dr_cs_bore=From(build),
    ):
        (
            z_cs_upper,
            z_cs_lower,
            _r_cs_coil_middle,
            r_cs_middle,
            z_cs_middle,
            r_cs_outer,
            r_cs_inner,
            a_cs_poloidal,
            a_cs_toroidal,
            dz_cs_full,
            dr_cs_full,
        ) = calculate_cs_geometry(
            z_tf_inside_half=z_tf_inside_half,
            f_z_cs_tf_internal=f_z_cs_tf_internal,
            dr_cs=dr_cs,
            dr_cs_bore=dr_cs_bore,
        )
        # `CSGeometry.r_cs_coil_middle` is dropped: it is bit-for-bit `r_cs_middle`
        # (`pfcoil.py:3030`, `:3042`) and `DataStructure`'s `PfCoilVariables` has no
        # field of that name -- PROCESS stores it only into
        # `r_pf_coil_middle[n_cs_pf_coils - 1]`, which `PFCoilPositions` owns. Owning it
        # here would mint a `VarPath` that names no place.
        return (
            z_cs_upper,
            z_cs_lower,
            r_cs_middle,
            z_cs_middle,
            r_cs_outer,
            r_cs_inner,
            a_cs_poloidal,
            a_cs_toroidal,
            dz_cs_full,
            dr_cs_full,
        )


class PFCoilPlacement(ExplicitFunction):
    """cottax node: `.tokamak.pf_coil.placement`.

    Occupant for `i_pf_location = (2, 2, 3, 3)` with `itart = 0`,
    `i_tf_shape = D_SHAPE` and `i_r_pf_outside_tf_placement = 0`. Owns the two
    `(N_PF_GROUPS_MAX, 2)` group arrays and `.pf_coil.r_pf_outside_tf_midplane`
    (`pfcoil.py:239-242`, one line, folded in here because it is this placement's own
    input and has no other producer or consumer).

    `r_cs_middle` is *not* read: no group on this arm has `i_pf_location = 1`, and it is
    only `place_pf_above_cs` that needs it. Declaring it would be exactly the
    union-of-arms invented edge the occupant split exists to remove.
    """

    r_pf_outside_tf_midplane = OutputInto(pf_coil)
    r_pf_coil_middle_group_array = OutputInto(pf_coil)
    z_pf_coil_middle_group_array = OutputInto(pf_coil)

    def __call__(
        self,
        r_tf_outboard_out=From(superconducting_tfcoil),
        dr_pf_tf_outboard_out_offset=From(pf_coil),
        rmajor=From(physics),
        rminor=From(physics),
        triang=From(physics),
        rpf2=From(pf_coil),
        z_tf_top=From(build),
        dz_tf_upper_lower_midplane=From(build),
        zref=From(pf_coil),
    ):
        r_pf_outside_tf_midplane = r_tf_outboard_out + dr_pf_tf_outboard_out_offset

        r_group, z_group = calculate_pf_coil_group_positions(
            rmajor=rmajor,
            rminor=rminor,
            triang=triang,
            rpf2=rpf2,
            z_tf_top=z_tf_top,
            dz_tf_upper_lower_midplane=dz_tf_upper_lower_midplane,
            zref=zref[:N_PF_GROUPS],
            r_pf_outside_tf_midplane=r_pf_outside_tf_midplane,
        )
        pad = jnp.zeros((N_PF_GROUPS_MAX, N_PF_COILS_IN_GROUP_MAX))
        return (
            r_pf_outside_tf_midplane,
            pad.at[:N_PF_GROUPS].set(r_group),
            pad.at[:N_PF_GROUPS].set(z_group),
        )


class PFCoilPositions(ExplicitFunction):
    """cottax node: `.tokamak.pf_coil.positions`. Owns `.pf_coil.r_pf_coil_middle` and
    `.pf_coil.z_pf_coil_middle` at their full `NGC2` width -- six PF coils flattened out
    of the group arrays, then the CS, then structural zeros.

    Index 7 (the plasma) is *not* written by PROCESS in these two arrays -- `pfcoil()`'s
    "Plasma size and shape" block (`:1067-1079`) sets the plasma's inner/outer radius and
    upper/lower height but never its centre -- so it stays zero here too.
    """

    r_pf_coil_middle = OutputInto(pf_coil)
    z_pf_coil_middle = OutputInto(pf_coil)

    def __call__(
        self,
        r_pf_coil_middle_group_array=From(pf_coil),
        z_pf_coil_middle_group_array=From(pf_coil),
        r_cs_middle=From(pf_coil),
    ):
        r_flat, z_flat = calculate_pf_coil_positions(
            r_pf_coil_middle_group_array=r_pf_coil_middle_group_array[:N_PF_GROUPS],
            z_pf_coil_middle_group_array=z_pf_coil_middle_group_array[:N_PF_GROUPS],
            r_cs_middle=r_cs_middle,
        )
        pad = jnp.zeros(NGC2)
        return (
            pad.at[: CS_INDEX + 1].set(r_flat),
            pad.at[: CS_INDEX + 1].set(z_flat),
        )
