"""Pure-functional port of `process/models/cryostat.py` (`Cryostat`,
`.tokamak.cryostat`) -- partial, the minimal closure for `.fwbs.r_cryostat_inboard`.

Audit record: `functional_process/_audit/units/models/cryostat.md`. **Not** the
stellarator's cryostat, which is `process/models/stellarator/stellarator.py:1282-1330`
(unit #1 chunk S5, already ported, already a slot of `.stellarator.fwbs`) -- two
different models of two different devices' cryostats.

`Cryostat.external_cryo_geometry` (`process/models/cryostat.py:25-85`) computes **seven**
fields in one straight-line sequence (`r_cryostat_inboard` -> `dz_pf_cryostat` ->
`z_cryostat_half_inside` -> `dz_tf_cryostat`/`vol_cryostat_internal` -> `vol_cryostat`
-> `dewmkg`). **All seven are ported now; only the first was until 2026-08-30.** (Six is
what this docstring and `cryostat.md` both said, counting the arrow *stages* above --
`dz_tf_cryostat` and `vol_cryostat_internal` share one. The field count is seven, and it
is the field count that a node's outputs have to match.)

The original scope was correct on its own terms and wrong on the measure that did not
exist yet. `tokamak_boundary.md` gave this slot exactly one listed output --
`.fwbs.r_cryostat_inboard`, read by `.buildings.sizing` -- and the wave-1 discipline
("port the minimal closure of functions that produces your slot's listed output
variables") stops there, because nothing downstream of that first line is needed to
produce it. What that walk could not see is a field with a reader *outside* this
graph: `.fwbs.dewmkg` is read by `structure.py`'s `StructureMasses`
(`.structure.coldmass`, and it is 14.4 million kg of it) and `.buildings.dz_tf_cryostat`
by the site-buildings sizing, and both sat frozen at `0.0` while PROCESS recomputed
them every pass. `boundary.unproduced_but_computed` is what named them, and this module
is two of its rows.

The whole method is one node, as it is one method: every field is a plain function of
the ones above it and there is no branch anywhere in it, so splitting it would mint
intermediate places for nothing.
"""

import jax.numpy as jnp


def calculate_r_cryostat_inboard(r_pf_coil_outer, dr_pf_cryostat):
    """Cryostat inboard radius (m): furthest PF coil's outer radius plus clearance.

    Ports `Cryostat.external_cryo_geometry`, `process/models/cryostat.py:39-41`,
    unchanged (`np.max` -> `jnp.max`).

    Parameters
    ----------
    r_pf_coil_outer :
        Outer radius of each PF coil (m), fixed-size array.
        `.pf_coil.r_pf_coil_outer`.
    dr_pf_cryostat :
        Clearance between the outermost PF coil and the cryostat (m).
        `.fwbs.dr_pf_cryostat`.

    Returns
    -------
    :
        Cryostat inboard radius (m).
    """
    return jnp.max(r_pf_coil_outer) + dr_pf_cryostat


def calculate_external_cryo_geometry(
    r_pf_coil_outer,
    dr_pf_cryostat,
    f_z_cryostat,
    z_pf_coil_upper,
    z_tf_inside_half,
    dr_tf_inboard,
    dr_cryostat,
    vol_vv,
    den_steel,
):
    """The cryostat's geometry and the cold mass it encloses.

    Ports `Cryostat.external_cryo_geometry` (`process/models/cryostat.py:25-85`) whole,
    unchanged (`np.max` -> `jnp.max`, `np.pi` -> `jnp.pi`). Six values, computed in the
    source's own order, each a plain function of the ones before it.

    `dr_tf_inboard` enters only through the TF-to-cryostat clearance and is *subtracted*
    there, which is worth stating because it is the one read of this function that is a
    thickness rather than a height: `:58-59` measures from the top of the TF coil, and
    the top of the TF coil is `z_tf_inside_half + dr_tf_inboard`.

    Parameters
    ----------
    r_pf_coil_outer :
        Outer radius of each PF coil (m), fixed-size array. `.pf_coil.r_pf_coil_outer`.
    dr_pf_cryostat :
        Clearance between the outermost PF coil and the cryostat (m).
        `.fwbs.dr_pf_cryostat`.
    f_z_cryostat :
        Cryostat lid clearance scaling factor, ITER-derived. `.build.f_z_cryostat`.
    z_pf_coil_upper :
        Upper edge height of each PF coil (m), fixed-size array.
        `.pf_coil.z_pf_coil_upper`.
    z_tf_inside_half :
        Half-height of the TF coil inside edge (m). `.build.z_tf_inside_half`.
    dr_tf_inboard :
        Inboard TF coil leg thickness (m). `.build.dr_tf_inboard`.
    dr_cryostat :
        Cryostat wall thickness (m). `.build.dr_cryostat`.
    vol_vv :
        Vacuum vessel structure volume (m^3). `.fwbs.vol_vv`.
    den_steel :
        Steel density (kg/m^3). `.fwbs.den_steel`.

    Returns
    -------
    :
        `(r_cryostat_inboard, dz_pf_cryostat, z_cryostat_half_inside, dz_tf_cryostat,
        vol_cryostat_internal, vol_cryostat, dewmkg)` -- seven, in the source's order.
    """
    r_cryostat_inboard = calculate_r_cryostat_inboard(r_pf_coil_outer, dr_pf_cryostat)

    # Clearance between the uppermost PF coil and the cryostat lid (m), scaled from
    # ITER by M. Kovari.
    dz_pf_cryostat = f_z_cryostat * (2.0 * r_cryostat_inboard) / 28.440

    z_cryostat_half_inside = jnp.max(z_pf_coil_upper) + dz_pf_cryostat
    dz_tf_cryostat = z_cryostat_half_inside - (z_tf_inside_half + dr_tf_inboard)

    vol_cryostat_internal = jnp.pi * r_cryostat_inboard**2 * 2 * z_cryostat_half_inside
    # The outer cylinder less the inner one.
    vol_cryostat = (
        jnp.pi
        * (r_cryostat_inboard + dr_cryostat) ** 2
        * 2
        * (dr_cryostat + z_cryostat_half_inside)
    ) - vol_cryostat_internal

    dewmkg = (vol_vv + vol_cryostat) * den_steel
    return (
        r_cryostat_inboard,
        dz_pf_cryostat,
        z_cryostat_half_inside,
        dz_tf_cryostat,
        vol_cryostat_internal,
        vol_cryostat,
        dewmkg,
    )
