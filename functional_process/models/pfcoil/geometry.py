"""Pure functions for the PF/CS coil geometry: CS geometry and filament placement,
and PF coil group/individual placement, extracted from
`functional_process/cottax/pfcoil/geometry.py`.

That module still holds the graph declarations (`ExplicitFunction` occupants) that wire
these functions to `VarPath`s; read its module docstring (and
`functional_process/cottax/pfcoil/__init__.py`'s package docstring) for scope and the
reference-arm switch table. The audit record is
`functional_process/_audit/units/models/pfcoil/geometry.md` and mirrors these functions,
not the declarations that call them.
"""

import jax.numpy as jnp

from functional_process.models.pfcoil import (
    CS_INDEX,
    N_CS_FILAMENTS,
    N_PF_GROUPS_MAX,
    NFXF,
    REFERENCE_TOPOLOGY,
    PFLocation,
)
from functional_process.models.safe_math import safe_sqrt

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
    source's own field order, since a `ImplementedFunction` binds returns positionally.

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


DR_CS_TURN_CONDUIT_MIN = 1.0e-3
"""The 1 mm floor `calculate_cs_turn_geometry_eu_demo` clamps the radial conduit
thickness to (`process/models/pfcoil.py:3138-3141`).

PROCESS calls this a "kludge" in its own `logger.error` and applies it to `dr` only,
leaving `dz` -- which is the same number one line earlier -- unclamped. Reproduced as
written, asymmetry included: the two fields feed `ncycle`'s two crack-size limits and
which one is floored changes the answer.
"""


def calculate_cs_turn_geometry_eu_demo(
    a_cs_poloidal,
    n_pf_coil_turns_cs,
    f_dr_dz_cs_turn,
    radius_cs_turn_corners,
    f_a_cs_turn_steel,
):
    """The EU DEMO stadium-shaped CS turn: its dimensions and its steel conduit.

    Ports `CSCoil.calculate_cs_turn_geometry_eu_demo`
    (`process/models/pfcoil.py:3074-3149`) together with the one line of `ohcalc` that
    feeds it, `a_cs_turn = a_cs_poloidal / n_pf_coil_turns[cs]` (`:3297-3300`). Those
    two are one node because `.pf_coil.a_cs_turn` has no other reader: keeping them
    apart would mint a node for a single division.

    Arithmetic unchanged (`np.`/`math.` -> `jnp.`); the `< 1 mm` clamp becomes
    `jnp.maximum`, the `logger.error` beside it is dropped, and `** 0.5` on the turn
    height becomes `safe_sqrt` -- the fractional-power-at-zero trap
    (`_audit/next_steps.md` §9), reachable here because `a_cs_turn` is proportional to
    the CS cross-section, which the solver is free to drive towards zero.

    **Why this is ported at all**, given that none of its six outputs had a reader in
    this graph until 2026-08-30: `ncycle` reads two of them, and reads them as the two
    crack-size limits that decide when its integration stops. On
    `low_aspect_ratio_DEMO` PROCESS computes `0.00990` for both conduit thicknesses
    against `pfcoil_variables.py`'s input defaults of `0.07` and `0.022` -- a factor of
    two on the vertical limit and seven on the radial -- so a `CsFatigue` node reading
    the defaults would have produced a confidently wrong `n_cycle` for constraint 90
    rather than an obviously wrong zero. `boundary.unproduced_but_computed` named both
    fields the moment that node landed, which is the whole point of that measure.

    Parameters
    ----------
    a_cs_poloidal :
        CS poloidal cross-sectional area (m^2). `.pf_coil.a_cs_poloidal`.
    n_pf_coil_turns_cs :
        Turns in the CS -- element `CS_INDEX` of `.pf_coil.n_pf_coil_turns`.
    f_dr_dz_cs_turn :
        Ratio of a CS turn's radial width to its vertical height.
        `.pf_coil.f_dr_dz_cs_turn`.
    radius_cs_turn_corners :
        Radius of a turn's curved outer corner (m). `.pf_coil.radius_cs_turn_corners`.
    f_a_cs_turn_steel :
        Steel area fraction of a CS turn. `.pf_coil.f_a_cs_turn_steel`.

    Returns
    -------
    :
        `(a_cs_turn, dz_cs_turn, dr_cs_turn, radius_cs_turn_cable_space,
        dr_cs_turn_conduit, dz_cs_turn_conduit)`.
    """
    a_cs_turn = a_cs_poloidal / n_pf_coil_turns_cs

    dz_cs_turn = safe_sqrt(a_cs_turn / f_dr_dz_cs_turn)
    dr_cs_turn = f_dr_dz_cs_turn * dz_cs_turn

    offset = (dr_cs_turn - dz_cs_turn) / jnp.pi
    radius_cs_turn_cable_space = -offset + jnp.sqrt(
        offset**2
        + (
            (dr_cs_turn * dz_cs_turn)
            - (4 - jnp.pi) * radius_cs_turn_corners**2
            - (a_cs_turn * f_a_cs_turn_steel)
        )
        / jnp.pi
    )

    dz_cs_turn_conduit = (dz_cs_turn / 2) - radius_cs_turn_cable_space
    # In this model the vertical and radial thicknesses are the same -- except that
    # PROCESS floors only the radial one. See `DR_CS_TURN_CONDUIT_MIN`.
    dr_cs_turn_conduit = jnp.maximum(dz_cs_turn_conduit, DR_CS_TURN_CONDUIT_MIN)

    return (
        a_cs_turn,
        dz_cs_turn,
        dr_cs_turn,
        radius_cs_turn_cable_space,
        dr_cs_turn_conduit,
        dz_cs_turn_conduit,
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
    rref=None,
    *,
    topology=REFERENCE_TOPOLOGY,
    r_pf_outside_tf_is_constant=False,
):
    """Coil centres by `(group, coil)`, for one topology's `i_pf_location` pattern.

    Ports `pfcoil()`'s placement loop (`process/models/pfcoil.py:245-352`), i.e.
    `place_pf_above_tf` (`:1178-1263`) for an `i_pf_location = 2` group,
    `place_pf_outside_tf` (`:1265-1343`) for a `3` and `place_pf_generally`
    (`:1345-1401`) for a `4`, at `not (itart == 1 and itartpf == 0)`. The `top_bottom`
    toggle is carried here exactly as `pfcoil()` carries it, and resolved at trace time
    -- see the module docstring.

    `i_pf_location = 1` (`place_pf_above_cs`, `:1115-1176`) is **UNPORTED**: it is the
    only arm that also needs `r_cs_middle`/`dr_pf_cs_middle_offset`, so it is a
    different read set rather than a different branch, and no tracked file uses it.
    A topology carrying it raises here rather than falling through.

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
        Per-group vertical placement ratio, one entry per group. For an
        `i_pf_location = 3` or `4` group this is the coil height in units of `rminor`.
        `.pf_coil.zref[:n_pf_coil_groups]`.
    r_pf_outside_tf_midplane :
        Radius at which an `i_pf_location = 3` coil would sit on the midplane (m).
        `.pf_coil.r_pf_outside_tf_midplane`.
    rref :
        Per-group radial placement ratio, in units of `rminor` from the plasma centre.
        `.pf_coil.rref[:n_pf_coil_groups]`. Read **only** by an `i_pf_location = 4`
        group, so it stays `None` on a topology that has none -- declaring it there
        would be the union-of-arms invented edge the occupant split exists to remove.
    topology :
        Static. Which groups exist, how many coils each holds and where each sits.
    r_pf_outside_tf_is_constant :
        Static. `i_tf_shape == PICTURE_FRAME or i_r_pf_outside_tf_placement == 1`
        (`pfcoil.py:1322-1326`) -- the two switches enter `place_pf_outside_tf` only
        through this disjunction, so the occupant resolves it once. `True` stacks an
        `i_pf_location = 3` group at the midplane radius; `False` follows the D-shaped
        TF curve.

    Returns
    -------
    tuple
        `(r_pf_coil_middle_group_array, z_pf_coil_middle_group_array)`, each
        `(topology.n_pf_coil_groups, N_PF_COILS_IN_GROUP_MAX)`. A group holding one coil
        leaves its second column at zero, exactly as PROCESS's pre-zeroed array does.
    """
    n_groups = topology.n_pf_coil_groups
    r_group = jnp.zeros((n_groups, N_PF_COILS_IN_GROUP_MAX))
    z_group = jnp.zeros((n_groups, N_PF_COILS_IN_GROUP_MAX))

    # `pfcoil()` initialises the toggle once, before the group loop (`:127`), and
    # `place_pf_above_tf` flips it per coil placed (`:1252-1261`) -- so it is carried
    # across groups as well as within one.
    top_bottom = 1

    r_above_tf = rmajor + rpf2 * triang * rminor
    z_above_tf_top = z_tf_top + _PF_ABOVE_TF_Z_CLEARANCE
    z_above_tf_bottom = -1.0 * (
        z_tf_top - dz_tf_upper_lower_midplane + _PF_ABOVE_TF_Z_CLEARANCE
    )

    for group in range(n_groups):
        location = topology.i_pf_location[group]
        for coil in range(topology.n_pf_coils_in_group[group]):
            sign = 1.0 if coil == 0 else -1.0

            if location is PFLocation.ABOVE_TF:
                r = r_above_tf
                z = z_above_tf_top if top_bottom == 1 else z_above_tf_bottom
                top_bottom = -top_bottom

            elif location is PFLocation.OUTSIDE_TF:
                z = rminor * zref[group] * sign
                if r_pf_outside_tf_is_constant:
                    r = jnp.asarray(r_pf_outside_tf_midplane)
                else:
                    r_raw = jnp.sqrt(r_pf_outside_tf_midplane**2 - z**2)
                    r = jnp.where(jnp.isinf(r_raw), 1e10, r_raw)

            elif location is PFLocation.GENERALLY_PLACED:
                z = rminor * zref[group] * sign
                r = rminor * rref[group] + rmajor

            else:
                raise NotImplementedError(
                    f"i_pf_location = {int(location)} on group {group} is a real "
                    "PROCESS branch (place_pf_above_cs, pfcoil.py:1115) but is not "
                    "ported: it is the only placement that reads r_cs_middle and "
                    "dr_pf_cs_middle_offset, so it is a different read set rather "
                    "than a different arm of this one, and no tracked input file "
                    "uses it"
                )

            r_group = r_group.at[group, coil].set(r)
            z_group = z_group.at[group, coil].set(z)

    return r_group, z_group


def calculate_pf_coil_positions(
    r_pf_coil_middle_group_array,
    z_pf_coil_middle_group_array,
    r_cs_middle=None,
    *,
    topology=REFERENCE_TOPOLOGY,
):
    """Coil centres flattened out of the group arrays, with the CS in its own slot.

    Ports `pfcoil()`'s `ncl` loop (`process/models/pfcoil.py:663-672`) together with the
    CS's writes into the same two arrays (`:182`, `:186-188`). Group-then-coil order,
    which is what fixes every per-coil index in this package.

    `r_cs_middle` is read only when the topology has a central solenoid; with
    `iohcl = 0` there is no slot for it and PROCESS's own CS write at `:182` lands on
    the *last PF coil's* index, where this same loop overwrites it one block later.

    Returns
    -------
    tuple
        `(r_pf_coil_middle, z_pf_coil_middle)`, `topology.n_cs_pf_coils` entries each --
        the PF coils in group-then-coil order, then the CS if there is one.
    """
    r_flat = [
        r_pf_coil_middle_group_array[group, coil]
        for group in range(topology.n_pf_coil_groups)
        for coil in range(topology.n_pf_coils_in_group[group])
    ]
    z_flat = [
        z_pf_coil_middle_group_array[group, coil]
        for group in range(topology.n_pf_coil_groups)
        for coil in range(topology.n_pf_coils_in_group[group])
    ]
    if topology.has_central_solenoid:
        r_flat.append(jnp.asarray(r_cs_middle))
        z_flat.append(jnp.zeros_like(jnp.asarray(r_cs_middle)))
    return jnp.stack(r_flat), jnp.stack(z_flat)


def calculate_cs_geometry_ports(z_tf_inside_half, f_z_cs_tf_internal, dr_cs, dr_cs_bore):
    """`CSCoilGeometry`: drops `r_cs_coil_middle` from `calculate_cs_geometry`'s tuple.

    It is bit-for-bit `r_cs_middle` (`pfcoil.py:3030`, `:3042`) and `DataStructure` has
    no field of that name -- see the class docstring.
    """
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


def calculate_cs_turn_geometry_eu_demo_from_turns(
    a_cs_poloidal,
    n_pf_coil_turns,
    f_dr_dz_cs_turn,
    radius_cs_turn_corners,
    f_a_cs_turn_steel,
):
    """`CSCoilTurnGeometry`: picks the CS's own turn count out of the full-width
    array.
    """
    return calculate_cs_turn_geometry_eu_demo(
        a_cs_poloidal,
        n_pf_coil_turns[CS_INDEX],
        f_dr_dz_cs_turn,
        radius_cs_turn_corners,
        f_a_cs_turn_steel,
    )


def calculate_pf_coil_placement(
    r_pf_outside_tf_midplane,
    rmajor,
    rminor,
    triang,
    rpf2,
    z_tf_top,
    dz_tf_upper_lower_midplane,
    zref,
    rref,
    *,
    topology,
    r_pf_outside_tf_is_constant,
):
    """`PFCoilPlacement`/`PFCoilPlacementSphericalTokamak`: trims `zref`/`rref` to
    `topology`'s group count and pads the two group arrays back out to
    `N_PF_GROUPS_MAX`.
    """
    n_groups = topology.n_pf_coil_groups
    r_group, z_group = calculate_pf_coil_group_positions(
        rmajor=rmajor,
        rminor=rminor,
        triang=triang,
        rpf2=rpf2,
        z_tf_top=z_tf_top,
        dz_tf_upper_lower_midplane=dz_tf_upper_lower_midplane,
        zref=zref[:n_groups],
        r_pf_outside_tf_midplane=r_pf_outside_tf_midplane,
        rref=None if rref is None else rref[:n_groups],
        topology=topology,
        r_pf_outside_tf_is_constant=r_pf_outside_tf_is_constant,
    )
    pad = jnp.zeros((N_PF_GROUPS_MAX, N_PF_COILS_IN_GROUP_MAX))
    return (
        r_pf_outside_tf_midplane,
        pad.at[:n_groups].set(r_group),
        pad.at[:n_groups].set(z_group),
    )


def calculate_pf_coil_placement_for_topology(
    r_tf_outboard_out,
    dr_pf_tf_outboard_out_offset,
    rmajor,
    rminor,
    triang,
    rpf2,
    z_tf_top,
    dz_tf_upper_lower_midplane,
    zref,
    rref=None,
    *,
    topology,
    r_pf_outside_tf_is_constant,
):
    """`PFCoilPlacement`/`PFCoilPlacementSphericalTokamak`: derives
    `r_pf_outside_tf_midplane` before delegating to `calculate_pf_coil_placement`.
    """
    r_pf_outside_tf_midplane = r_tf_outboard_out + dr_pf_tf_outboard_out_offset
    return calculate_pf_coil_placement(
        r_pf_outside_tf_midplane=r_pf_outside_tf_midplane,
        rmajor=rmajor,
        rminor=rminor,
        triang=triang,
        rpf2=rpf2,
        z_tf_top=z_tf_top,
        dz_tf_upper_lower_midplane=dz_tf_upper_lower_midplane,
        zref=zref,
        rref=rref,
        topology=topology,
        r_pf_outside_tf_is_constant=r_pf_outside_tf_is_constant,
    )
