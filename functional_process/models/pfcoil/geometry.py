"""Where the CS and the PF coils are: cross-sections, filament placement, coil centres.

Audit record: `functional_process/_audit/units/models/pfcoil/geometry.md`.

Five units, all straight-line algebra with no iteration and no call into another model:

- `calculate_cs_geometry` -- `CSCoil.calculate_cs_geometry`
  (`process/models/pfcoil.py:3005-3072`), already a `@staticmethod` with explicit
  arguments; the port is a signature promotion plus `np.` -> `jnp.`.
- `calculate_cs_turn_geometry_eu_demo` -- `CSCoil.calculate_cs_turn_geometry_eu_demo`
  (`:3074-3149`), likewise, with `ohcalc`'s own `a_cs_turn` division (`:3297-3300`)
  folded into it. Added 2026-08-30 for `.tokamak.cs_fatigue`, which reads two of its
  outputs.
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

import equinox as eqx
import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.models.pfcoil import (
    CS_INDEX,
    N_CS_FILAMENTS,
    N_PF_GROUPS_MAX,
    NFXF,
    NGC2,
    REFERENCE_TOPOLOGY,
    SPHERICAL_TOKAMAK_TOPOLOGY,
    PFCoilTopology,
    PFLocation,
)
from functional_process.models.safe_math import safe_sqrt
from functional_process.paths import (
    build,
    cs_fatigue,
    pf_coil,
    physics,
    superconducting_tfcoil,
)

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


class CSCoilTurnGeometry(ExplicitFunction):
    """cottax node: `.tokamak.cs_coil.turn_geometry`. Owns the CS turn's dimensions and
    the two conduit thicknesses `ncycle` reads. No switch.

    **The two conduit thicknesses land in `.cs_fatigue`, not `.pf_coil`**, which is
    PROCESS's own placement (`pfcoil.py:3314-3319`) and not a choice made here: they are
    written by `ohcalc` and read only by `CsFatigue.ncycle`, and the area they live in
    follows the reader rather than the writer. That is the one cross-area edge this node
    makes, and it is why `.tokamak.cs_coil` gains a slot for a calculation whose other
    four outputs nothing in this graph reads.
    """

    a_cs_turn = OutputInto(pf_coil)
    dz_cs_turn = OutputInto(pf_coil)
    dr_cs_turn = OutputInto(pf_coil)
    radius_cs_turn_cable_space = OutputInto(pf_coil)
    dr_cs_turn_conduit = OutputInto(cs_fatigue)
    dz_cs_turn_conduit = OutputInto(cs_fatigue)

    def __call__(
        self,
        a_cs_poloidal=From(pf_coil),
        n_pf_coil_turns=From(pf_coil),
        f_dr_dz_cs_turn=From(pf_coil),
        radius_cs_turn_corners=From(pf_coil),
        f_a_cs_turn_steel=From(pf_coil),
    ):
        return calculate_cs_turn_geometry_eu_demo(
            a_cs_poloidal,
            n_pf_coil_turns[CS_INDEX],
            f_dr_dz_cs_turn,
            radius_cs_turn_corners,
            f_a_cs_turn_steel,
        )


class PFCoilPlacement(ExplicitFunction):
    """cottax node: `.tokamak.pf_coil.placement`.

    Occupant for `i_pf_location = (2, 2, 3, 3)` with `not (itart == 1 and itartpf == 0)`,
    `i_tf_shape = D_SHAPE` and `i_r_pf_outside_tf_placement = 0`. Owns the two
    `(N_PF_GROUPS_MAX, 2)` group arrays and `.pf_coil.r_pf_outside_tf_midplane`
    (`pfcoil.py:239-242`, one line, folded in here because it is this placement's own
    input and has no other producer or consumer).

    `r_cs_middle` is *not* read: no group on this arm has `i_pf_location = 1`, and it is
    only `place_pf_above_cs` that needs it. Declaring it would be exactly the
    union-of-arms invented edge the occupant split exists to remove. `rref` is not read
    for the same reason -- no group here has `i_pf_location = 4`.
    """

    topology: PFCoilTopology = eqx.field(static=True, default=REFERENCE_TOPOLOGY)
    """Static, and the reference topology by construction: this occupant's whole
    identity is that pattern. `PFCoilPlacementSphericalTokamak` carries the other one."""

    r_pf_outside_tf_is_constant: bool = eqx.field(static=True, default=False)
    """`i_tf_shape == PICTURE_FRAME or i_r_pf_outside_tf_placement == 1`
    (`pfcoil.py:1322-1326`), resolved once. `False` here -- a D-shaped TF with the
    default placement, so an outside-TF coil's radius follows the TF curve."""

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
        return self._placed(
            r_pf_outside_tf_midplane=r_pf_outside_tf_midplane,
            rmajor=rmajor,
            rminor=rminor,
            triang=triang,
            rpf2=rpf2,
            z_tf_top=z_tf_top,
            dz_tf_upper_lower_midplane=dz_tf_upper_lower_midplane,
            zref=zref,
            rref=None,
        )

    def _placed(
        self,
        r_pf_outside_tf_midplane,
        rmajor,
        rminor,
        triang,
        rpf2,
        z_tf_top,
        dz_tf_upper_lower_midplane,
        zref,
        rref,
    ):
        """The placement and its `N_PF_GROUPS_MAX` padding, given this arm's reads.

        Not a port surface: `_params` reads `__call__`'s signature only, so each
        occupant still declares its own ports -- the `CoilsMass` shape `masses.py`
        already uses. The only entry that differs between the two occupants is whether
        `rref` is a read or a `None`.
        """
        n_groups = self.topology.n_pf_coil_groups
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
            topology=self.topology,
            r_pf_outside_tf_is_constant=self.r_pf_outside_tf_is_constant,
        )
        pad = jnp.zeros((N_PF_GROUPS_MAX, N_PF_COILS_IN_GROUP_MAX))
        return (
            r_pf_outside_tf_midplane,
            pad.at[:n_groups].set(r_group),
            pad.at[:n_groups].set(z_group),
        )


class PFCoilPlacementSphericalTokamak(PFCoilPlacement):
    """cottax node: `.tokamak.pf_coil.placement`, the spherical tokamaks' occupant.

    Occupant for `i_pf_location = (2, 3, 3, 4)` with
    `n_pf_coils_in_group = (2, 2, 2, 2)`, `i_tf_shape = PICTURE_FRAME` and
    `i_r_pf_outside_tf_placement = 1` --
    `spherical_tokamak_eval.IN.DAT` (`:233`, `:236-237`, `:357`) and
    `st_regression.IN.DAT` (`:1755`, `:1764`, `:1788`, `:803`), which set the same four.

    **Three differences from `PFCoilPlacement`, all of them structural**, and none of
    them a value:

    1. The topology. Group 0 holds *two* `i_pf_location = 2` coils, so `pfcoil()`'s
       `top_bottom` toggle flips inside one group instead of between two; groups 1 and 2
       are the outside-TF pairs; group 3 is the `i_pf_location = 4` pair, which
       `place_pf_generally` (`pfcoil.py:1345-1401`) places from `rref`/`zref` about the
       plasma centre.
    2. `r_pf_outside_tf_is_constant`. `i_tf_shape = 2` and
       `i_r_pf_outside_tf_placement = 1` are the two halves of one disjunction
       (`:1322-1326`); either alone stacks the outside-TF coils at the midplane radius
       instead of following the TF curve, and both are set on both files.
    3. **`rref` is a read here and is not one on the conventional arm.** That is the
       whole reason this is a second occupant rather than a second static field: the
       `i_pf_location = 4` group is the only thing in `pfcoil()` that touches
       `.pf_coil.rref`, so declaring it on an arm with no such group would be an
       invented edge.

    `itart`/`itartpf` do **not** enter: both files set `itartpf = 1`, and
    `place_pf_above_tf`'s spherical-tokamak arm is guarded by `itart == 1 and
    itartpf == 0` (`:1250`). See `indat._pf_coil_system_deviations`' `-3`.
    """

    topology: PFCoilTopology = eqx.field(static=True, default=SPHERICAL_TOKAMAK_TOPOLOGY)
    r_pf_outside_tf_is_constant: bool = eqx.field(static=True, default=True)

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
        rref=From(pf_coil),
    ):
        r_pf_outside_tf_midplane = r_tf_outboard_out + dr_pf_tf_outboard_out_offset
        return self._placed(
            r_pf_outside_tf_midplane=r_pf_outside_tf_midplane,
            rmajor=rmajor,
            rminor=rminor,
            triang=triang,
            rpf2=rpf2,
            z_tf_top=z_tf_top,
            dz_tf_upper_lower_midplane=dz_tf_upper_lower_midplane,
            zref=zref,
            rref=rref,
        )


class PFCoilPositions(ExplicitFunction):
    """cottax node: `.tokamak.pf_coil.positions`. Owns `.pf_coil.r_pf_coil_middle` and
    `.pf_coil.z_pf_coil_middle` at their full `NGC2` width -- the PF coils flattened out
    of the group arrays, then the CS, then structural zeros.

    The plasma's index is *not* written by PROCESS in these two arrays -- `pfcoil()`'s
    "Plasma size and shape" block (`:1067-1079`) sets the plasma's inner/outer radius and
    upper/lower height but never its centre -- so it stays zero here too.
    """

    topology: PFCoilTopology = eqx.field(static=True, default=REFERENCE_TOPOLOGY)
    """Static. Which slot each coil occupies, and whether there is a CS slot at all."""

    r_pf_coil_middle = OutputInto(pf_coil)
    z_pf_coil_middle = OutputInto(pf_coil)

    def __call__(
        self,
        r_pf_coil_middle_group_array=From(pf_coil),
        z_pf_coil_middle_group_array=From(pf_coil),
        r_cs_middle=From(pf_coil),
    ):
        return self._flattened(
            r_pf_coil_middle_group_array,
            z_pf_coil_middle_group_array,
            r_cs_middle,
        )

    def _flattened(
        self,
        r_pf_coil_middle_group_array,
        z_pf_coil_middle_group_array,
        r_cs_middle,
    ):
        """The flattening and its `NGC2` padding, given this arm's reads."""
        n_groups = self.topology.n_pf_coil_groups
        r_flat, z_flat = calculate_pf_coil_positions(
            r_pf_coil_middle_group_array=r_pf_coil_middle_group_array[:n_groups],
            z_pf_coil_middle_group_array=z_pf_coil_middle_group_array[:n_groups],
            r_cs_middle=r_cs_middle,
            topology=self.topology,
        )
        pad = jnp.zeros(NGC2)
        filled = self.topology.n_cs_pf_coils
        return (
            pad.at[:filled].set(r_flat),
            pad.at[:filled].set(z_flat),
        )


class PFCoilPositionsNoCentralSolenoid(PFCoilPositions):
    """cottax node: `.tokamak.pf_coil.positions`, the `iohcl = 0` occupant.

    **`r_cs_middle` is not read**, and that is the whole difference. With no central
    solenoid there is no slot for it in either array -- `pfcoil()`'s CS write at `:182`
    lands on `n_cs_pf_coils - 1`, which with `iohcl = 0` is the *last PF coil's* index,
    and the `ncl` loop three hundred lines later (`:663-672`) overwrites it with that
    coil's own centre. So the CS geometry never survives into these arrays on this arm,
    and reading it would be an edge to a namespace this machine does not have.
    """

    topology: PFCoilTopology = eqx.field(static=True, default=SPHERICAL_TOKAMAK_TOPOLOGY)

    def __call__(
        self,
        r_pf_coil_middle_group_array=From(pf_coil),
        z_pf_coil_middle_group_array=From(pf_coil),
    ):
        return self._flattened(
            r_pf_coil_middle_group_array,
            z_pf_coil_middle_group_array,
            r_cs_middle=None,
        )
