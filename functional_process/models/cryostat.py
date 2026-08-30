"""Pure-functional port of `process/models/cryostat.py` (`Cryostat`,
`.tokamak.cryostat`) -- partial: the minimal closure for `.fwbs.r_cryostat_inboard` and,
since 2026-08-30, for `.buildings.dz_tf_cryostat`.

Audit record: `functional_process/_audit/units/models/cryostat.md`. **Not** the
stellarator's cryostat, which is `process/models/stellarator/stellarator.py:1282-1330`
(unit #1 chunk S5, already ported, already a slot of `.stellarator.fwbs`) -- two
different models of two different devices' cryostats.

`Cryostat.external_cryo_geometry` (`process/models/cryostat.py:25-85`) computes six
fields in a straight-line sequence (`r_cryostat_inboard` -> `dz_pf_cryostat` ->
`z_cryostat_half_inside` -> `dz_tf_cryostat`/`vol_cryostat_internal` -> `vol_cryostat`
-> `dewmkg`). Wave 1 ported the **first**: `tokamak_boundary.md`'s `.tokamak.cryostat`
row named `.fwbs.r_cryostat_inboard` and nothing else, and nothing downstream of it is
needed to produce it, so per that wave's scope discipline the other five were UNPORTED.

**Four of the six are ported now**, because `.buildings.dz_tf_cryostat` -- the fourth --
turned out to be a missing producer. `.buildings.sizing` reads it; nothing owned it;
`.buildings.dz_tf_cryostat` is one of PROCESS's own `InputVariable`s
(`process/core/input.py:377`, default `2.5`, `buildings_variables.py:56`), so the port
was not reading a cold `0.0` but a **live-looking input**, `2.5` against PROCESS's
`5.5730055` on `large_tokamak_nof`. That is the harder half of the missing-producer
class: an input PROCESS unconditionally overwrites before anything reads it.

The overwriting is unconditional and the overwrite always wins, measured rather than
assumed. `external_cryo_geometry` writes the field with no branch above it;
`caller.py:351` runs the cryostat and `caller.py:370` runs the buildings, in that order;
and the only other reads of the field in `process/` are inside `if output:` reporting
blocks in `models/build.py` (`:191-219`, `:456-512`, `:769-776`). So no live read ever
sees the `IN.DAT` value -- and `large_tokamak_nof.IN.DAT` does not set it anyway. It is
a genuine producer, not a genuine input, and the seed it was frozen at is dead.

`.fwbs.vol_cryostat_internal`, `.fwbs.vol_cryostat` and `.fwbs.dewmkg` remain UNPORTED.
`.fwbs.dewmkg` is *also* on `missing_producers_tokamak.txt` and is not ported here: it
needs `vol_cryostat`, which needs `.build.dr_cryostat`, and that is its own closure --
left for the pass that takes it, rather than folded in unaudited. See `cryostat.md`.
"""

import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.paths import blanket, build, buildings, fwbs, pf_coil


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


def calculate_cryostat_vertical_clearances(
    f_z_cryostat,
    r_cryostat_inboard,
    z_pf_coil_upper,
    z_tf_inside_half,
    dr_tf_inboard,
):
    """`external_cryo_geometry`'s vertical chain: the PF-coil-to-lid clearance, the
    cryostat's inside half-height, and the TF-coil-to-cryostat clearance.

    Ports `Cryostat.external_cryo_geometry`, `process/models/cryostat.py:43-60`,
    unchanged (`np.max` -> `jnp.max`). Three PROCESS statements in one function rather
    than three, for the same reason `calculate_divertor_heat_flux_split` holds three of
    `Divertor.run()`'s: they are consecutive, unconditional, and each reads the one
    before, so there is nothing between them for a caller to choose. The `28.440` in the
    first is ITER's cryostat diameter in metres -- PROCESS calls the line a "scaling
    from ITER by M. Kovari" and it carries no port-side interpretation.

    **`.buildings.dz_tf_cryostat` is why this function exists**, and it is the awkward
    kind of missing producer: not a cold `0.0` but a PROCESS `InputVariable`
    (`core/input.py:377`, default `2.5`) that this method unconditionally overwrites --
    `5.5730055` on `large_tokamak_nof` -- before its only live reader
    (`.buildings.sizing`, `process/models/buildings.py:245`) runs. See the module
    docstring for the ordering measurement that establishes the seed is dead.

    Parameters
    ----------
    f_z_cryostat :
        Cryostat lid clearance scaling factor. `.build.f_z_cryostat` -- a genuine
        PROCESS input (`core/input.py:443`, default `4.268`), written nowhere in
        `process/models/`, and a **new boundary input** of this port as of the day this
        function landed. That growth is the boundary doing its job: it is a landed
        producer's own declared read, not a lost producer.
    r_cryostat_inboard :
        Cryostat inboard radius (m). `.fwbs.r_cryostat_inboard`, from
        `calculate_r_cryostat_inboard` above.
    z_pf_coil_upper :
        Upper vertical extent of each PF coil (m), fixed-size array.
        `.pf_coil.z_pf_coil_upper`.
    z_tf_inside_half :
        Half-height of the TF coil's inside edge (m). `.build.z_tf_inside_half`.
    dr_tf_inboard :
        Inboard TF coil radial thickness (m). `.build.dr_tf_inboard`.

    Returns
    -------
    tuple
        `(dz_pf_cryostat, z_cryostat_half_inside, dz_tf_cryostat)` --
        `.blanket.dz_pf_cryostat`, `.fwbs.z_cryostat_half_inside`,
        `.buildings.dz_tf_cryostat`.
    """
    dz_pf_cryostat = f_z_cryostat * (2.0 * r_cryostat_inboard) / 28.440
    z_cryostat_half_inside = jnp.max(z_pf_coil_upper) + dz_pf_cryostat
    dz_tf_cryostat = z_cryostat_half_inside - (z_tf_inside_half + dr_tf_inboard)
    return dz_pf_cryostat, z_cryostat_half_inside, dz_tf_cryostat


class Cryostat(ExplicitFunction):
    """cottax node: `.tokamak.cryostat`. Owns the **first four** of the six fields
    `external_cryo_geometry` computes; `.fwbs.vol_cryostat_internal`,
    `.fwbs.vol_cryostat` and `.fwbs.dewmkg` remain UNPORTED (module docstring).

    **One node rather than four**, unlike `.tokamak.plasma_beta`'s split into a
    namespace: PROCESS's source here is one straight-line method with no branch, and the
    two intermediates (`.blanket.dz_pf_cryostat`, `.fwbs.z_cryostat_half_inside`) are
    written once and read back immediately -- `schema.md`'s `local-intermediate`, which
    is the classification for a value routed through `self.data` because that is this
    codebase's idiom for every value and not because anything shares it. Owning them is
    still right: PROCESS writes them, so `boundary.computed_by_process` counts them, and
    a consumer appearing later (`costs_2015.py` already reads
    `.fwbs.z_cryostat_half_inside`, on a cost model this machine does not select) must
    find a producer rather than a seed.
    """

    r_cryostat_inboard = OutputInto(fwbs)
    dz_pf_cryostat = OutputInto(blanket)
    z_cryostat_half_inside = OutputInto(fwbs)
    dz_tf_cryostat = OutputInto(buildings)

    def __call__(
        self,
        r_pf_coil_outer=From(pf_coil),
        dr_pf_cryostat=From(fwbs),
        f_z_cryostat=From(build),
        z_pf_coil_upper=From(pf_coil),
        z_tf_inside_half=From(build),
        dr_tf_inboard=From(build),
    ):
        r_cryostat_inboard = calculate_r_cryostat_inboard(
            r_pf_coil_outer, dr_pf_cryostat
        )
        return (
            r_cryostat_inboard,
            *calculate_cryostat_vertical_clearances(
                f_z_cryostat,
                r_cryostat_inboard,
                z_pf_coil_upper,
                z_tf_inside_half,
                dr_tf_inboard,
            ),
        )
