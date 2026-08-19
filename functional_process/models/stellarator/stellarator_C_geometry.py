"""Pure-functional port of `Stellarator.st_new_config`/`st_geom` (chunk 1C of unit #1).

Audit record: `functional_process/models/stellarator/stellarator_C_geometry.md`. Source:
`process/models/stellarator/stellarator.py`, lines 191-319.

Three tier-1 functions result:

- `calculate_default_aspect_ratio` -- the `1 not in ixc` preamble of `st_new_config`.
  This is the record's main finding: whether `.physics.aspect` is an *output* of this
  chunk at all is a per-run configuration fact (is `aspect` an active iteration
  variable?), not a static property of the code -- the same
  `conditional-ownership-by-run-config` shape unit #2 (`build.py`) already found for
  `.build.dr_blkt_inboard`/`dr_blkt_outboard`, and resolved the same way here: split the
  maybe-owned field into its own tiny node, only instantiated when `1 not in
  data.numerics.ixc`, and have the downstream function take `aspect` as a plain input
  regardless of where it came from. The body is a trivial passthrough of
  `stella_config_aspect_ref` -- ported as its own function anyway (rather than inlined
  as a bare graph edge) because a `Graph` node is what makes the conditional-instantiation
  decision structural and inspectable, not because the arithmetic needs a function.
- `calculate_stellarator_scaling_factors` -- everything `st_new_config` computes
  unconditionally after the `aspect` decision. Takes `aspect` as an ordinary explicit
  arg; never re-derives or re-decides it. Device-config values (`stella_config_*`) are
  taken as plain scalar inputs too -- device-config *loading*
  (`load_stellarator_config`/`preset_config.py`, registry unit #8) is deliberately left
  out of this traced function: it's `istell`-keyed table selection plus, for `istell==6`,
  file I/O, i.e. graph-assembly-time setup, not a computation. See
  `preset_config.md` (unit #8, already audited and found "not portable as a node") and
  this chunk's audit record for the reasoning; not re-litigated here.
- `calculate_stellarator_plasma_geometry` -- `st_geom`, clean and switch-free.

**A real finding, documented but not fixed** (see the audit record): under the current
PROCESS control flow, `f_st_n_coils` is *always* exactly `1.0`. `st_new_config`
unconditionally overwrites `.tfcoil.n_tf_coils` to
`stella_config_coilspermodule * stella_config_symmetry` a few lines above computing
`f_st_n_coils = n_tf_coils / (stella_config_coilspermodule * stella_config_symmetry)` --
the same straight-line function, no intervening branch or call that could make the two
diverge, so the division is algebraically forced to 1. Ported faithfully (as a genuine
division of two arguments, not collapsed to a literal `1.0`), because collapsing it would
be "fixing" behaviour this audit is only chartered to document, and because a future,
non-PROCESS caller of this node could legitimately pass a `n_tf_coils` that was *not*
just derived from the same two config fields (e.g. once `n_tf_coils`'s
conditional-ownership, if any, is untangled elsewhere).
"""

import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    FromExactly,
    Output,
)


def calculate_default_aspect_ratio(stella_config_aspect_ref):
    """`.physics.aspect`'s default value, when `aspect` is not an active iteration
    variable.

    Ports the `if 1 not in self.data.numerics.ixc: self.data.physics.aspect = ...`
    branch of `st_new_config`. Only instantiate this node when `1 not in
    data.numerics.ixc` for the run being built -- when `aspect` *is* an active iteration
    variable, this node must not exist at all (the value is a boundary input owned by the
    optimiser, with no forward producer here). See module docstring.

    Parameters
    ----------
    stella_config_aspect_ref :
        Reference-configuration aspect ratio. `.stellarator_config.stella_config_aspect_ref`.

    Returns
    -------
    :
        `.physics.aspect` (dimensionless).
    """
    return stella_config_aspect_ref


def calculate_stellarator_scaling_factors(
    rmajor,
    aspect,
    b_plasma_toroidal_on_axis,
    f_st_coil_aspect,
    stella_config_coilspermodule,
    stella_config_symmetry,
    stella_config_rmajor_ref,
    stella_config_rminor_ref,
    stella_config_aspect_ref,
    stella_config_bt_ref,
    stella_config_coil_rmajor,
    stella_config_coil_rminor,
    stella_config_min_plasma_coil_distance,
):
    """The unconditional body of `st_new_config`, after the `aspect` decision.

    Ports everything `st_new_config` computes once `.physics.aspect` is settled: plasma
    minor radius/inverse aspect ratio, the coil count (unconditionally overwritten --
    PROCESS's own comment: "This overwrites n_tf_coils in input file"), the
    reference-configuration scaling factors, and the coil major/minor radius and shape
    factor. See module docstring for the `f_st_n_coils` finding and for why device-config
    loading is not part of this function.

    Parameters
    ----------
    rmajor :
        Plasma major radius (m). `.physics.rmajor`.
    aspect :
        Plasma aspect ratio. `.physics.aspect` -- from `calculate_default_aspect_ratio`
        or an external (iteration-variable) input, depending on run config; see module
        docstring.
    b_plasma_toroidal_on_axis :
        Toroidal field on axis (T). `.physics.b_plasma_toroidal_on_axis`.
    f_st_coil_aspect :
        Coil aspect ratio factor. `.stellarator.f_st_coil_aspect`.
    stella_config_coilspermodule, stella_config_symmetry :
        Coils per module, number of field periods. `.stellarator_config.*`.
    stella_config_rmajor_ref, stella_config_rminor_ref, stella_config_aspect_ref,
    stella_config_bt_ref :
        Reference-configuration plasma major/minor radius (m), aspect ratio, toroidal
        field (T). `.stellarator_config.*`.
    stella_config_coil_rmajor, stella_config_coil_rminor :
        Reference-configuration coil major/minor radius (m). `.stellarator_config.*`.
    stella_config_min_plasma_coil_distance :
        Reference-configuration minimum plasma-coil distance (m).
        `.stellarator_config.stella_config_min_plasma_coil_distance`.

    Returns
    -------
    :
        `(rminor, eps, n_tf_coils, f_st_rmajor, f_st_rminor, f_st_aspect, f_st_n_coils,
        f_st_b, r_coil_major, r_coil_minor, f_coil_shape)`.
    """
    rminor = rmajor / aspect
    eps = 1.0e0 / aspect

    n_tf_coils = stella_config_coilspermodule * stella_config_symmetry

    f_st_rmajor = rmajor / stella_config_rmajor_ref
    f_st_rminor = rminor / stella_config_rminor_ref
    f_st_aspect = aspect / stella_config_aspect_ref
    # Algebraically forced to 1.0 given the `n_tf_coils` assignment above -- see module
    # docstring's "real finding".
    f_st_n_coils = n_tf_coils / (
        stella_config_coilspermodule * stella_config_symmetry
    )
    f_st_b = b_plasma_toroidal_on_axis / stella_config_bt_ref

    f_coil_aspect = f_st_coil_aspect

    r_coil_major = stella_config_coil_rmajor * f_st_rmajor
    r_coil_minor = stella_config_coil_rminor * f_st_rmajor / f_coil_aspect

    f_coil_shape = (
        stella_config_min_plasma_coil_distance + stella_config_rminor_ref
    ) / stella_config_coil_rminor

    return (
        rminor,
        eps,
        n_tf_coils,
        f_st_rmajor,
        f_st_rminor,
        f_st_aspect,
        f_st_n_coils,
        f_st_b,
        r_coil_major,
        r_coil_minor,
        f_coil_shape,
    )


def calculate_stellarator_plasma_geometry(
    f_st_rmajor,
    f_st_rminor,
    rminor,
    stella_config_vol_plasma,
    stella_config_plasma_surface,
):
    """`st_geom`: plasma volume and surface area, scaled from the reference config.

    Switch-free, calls nothing else. Ports the whole of `st_geom`.

    Parameters
    ----------
    f_st_rmajor, f_st_rminor :
        Major/minor-radius scaling factors. `.stellarator.f_st_rmajor`,
        `.stellarator.f_st_rminor` -- from `calculate_stellarator_scaling_factors`.
    rminor :
        Plasma minor radius (m). `.physics.rminor` -- from
        `calculate_stellarator_scaling_factors`.
    stella_config_vol_plasma, stella_config_plasma_surface :
        Reference-configuration plasma volume (m3) and surface area (m2).
        `.stellarator_config.*`.

    Returns
    -------
    :
        `(vol_plasma, a_plasma_surface, a_plasma_poloidal, a_plasma_surface_outboard)`.
        `a_plasma_surface_outboard` retained only for the "obsolescent fispact
        calculation" per PROCESS's own comment; approximated as half the total surface,
        same as for tokamaks.
    """
    vol_plasma = f_st_rmajor * f_st_rminor**2 * stella_config_vol_plasma

    a_plasma_surface = f_st_rmajor * f_st_rminor * stella_config_plasma_surface

    a_plasma_poloidal = jnp.pi * rminor * rminor

    a_plasma_surface_outboard = 0.5e0 * a_plasma_surface

    return vol_plasma, a_plasma_surface, a_plasma_poloidal, a_plasma_surface_outboard


class DefaultAspectRatio(ExplicitFunction):
    """cottax node: `calculate_default_aspect_ratio`, ports declared.

    Only instantiate this node when `1 not in data.numerics.ixc` -- see module
    docstring. When `aspect` *is* an active iteration variable, no node in this chunk
    owns `.physics.aspect` at all.
    """

    aspect = Output(lambda s: s.physics.aspect)

    def __call__(
        self,
        stella_config_aspect_ref=FromExactly(
            lambda s: s.stellarator_config.stella_config_aspect_ref
        ),
    ):
        return calculate_default_aspect_ratio(stella_config_aspect_ref)


class StellaratorScalingFactors(ExplicitFunction):
    """cottax node: `calculate_stellarator_scaling_factors`, ports declared.

    `aspect` read from wherever the `1 in ixc` graph-assembly choice puts it --
    `DefaultAspectRatio`'s output, or an external (iteration-variable) input, per module
    docstring.
    """

    rminor = Output(lambda s: s.physics.rminor)
    eps = Output(lambda s: s.physics.eps)
    n_tf_coils = Output(lambda s: s.tfcoil.n_tf_coils)
    f_st_rmajor = Output(lambda s: s.stellarator.f_st_rmajor)
    f_st_rminor = Output(lambda s: s.stellarator.f_st_rminor)
    f_st_aspect = Output(lambda s: s.stellarator.f_st_aspect)
    f_st_n_coils = Output(lambda s: s.stellarator.f_st_n_coils)
    f_st_b = Output(lambda s: s.stellarator.f_st_b)
    r_coil_major = Output(lambda s: s.stellarator.r_coil_major)
    r_coil_minor = Output(lambda s: s.stellarator.r_coil_minor)
    f_coil_shape = Output(lambda s: s.stellarator.f_coil_shape)

    def __call__(
        self,
        rmajor=FromExactly(lambda s: s.physics.rmajor),
        aspect=FromExactly(lambda s: s.physics.aspect),
        b_plasma_toroidal_on_axis=FromExactly(lambda s: s.physics.b_plasma_toroidal_on_axis),
        f_st_coil_aspect=FromExactly(lambda s: s.stellarator.f_st_coil_aspect),
        stella_config_coilspermodule=FromExactly(
            lambda s: s.stellarator_config.stella_config_coilspermodule
        ),
        stella_config_symmetry=FromExactly(
            lambda s: s.stellarator_config.stella_config_symmetry
        ),
        stella_config_rmajor_ref=FromExactly(
            lambda s: s.stellarator_config.stella_config_rmajor_ref
        ),
        stella_config_rminor_ref=FromExactly(
            lambda s: s.stellarator_config.stella_config_rminor_ref
        ),
        stella_config_aspect_ref=FromExactly(
            lambda s: s.stellarator_config.stella_config_aspect_ref
        ),
        stella_config_bt_ref=FromExactly(lambda s: s.stellarator_config.stella_config_bt_ref),
        stella_config_coil_rmajor=FromExactly(
            lambda s: s.stellarator_config.stella_config_coil_rmajor
        ),
        stella_config_coil_rminor=FromExactly(
            lambda s: s.stellarator_config.stella_config_coil_rminor
        ),
        stella_config_min_plasma_coil_distance=FromExactly(
            lambda s: s.stellarator_config.stella_config_min_plasma_coil_distance
        ),
    ):
        return calculate_stellarator_scaling_factors(
            rmajor,
            aspect,
            b_plasma_toroidal_on_axis,
            f_st_coil_aspect,
            stella_config_coilspermodule,
            stella_config_symmetry,
            stella_config_rmajor_ref,
            stella_config_rminor_ref,
            stella_config_aspect_ref,
            stella_config_bt_ref,
            stella_config_coil_rmajor,
            stella_config_coil_rminor,
            stella_config_min_plasma_coil_distance,
        )


class StellaratorPlasmaGeometry(ExplicitFunction):
    """cottax node: `calculate_stellarator_plasma_geometry`, ports declared."""

    vol_plasma = Output(lambda s: s.physics.vol_plasma)
    a_plasma_surface = Output(lambda s: s.physics.a_plasma_surface)
    a_plasma_poloidal = Output(lambda s: s.physics.a_plasma_poloidal)
    a_plasma_surface_outboard = Output(lambda s: s.physics.a_plasma_surface_outboard)

    def __call__(
        self,
        f_st_rmajor=FromExactly(lambda s: s.stellarator.f_st_rmajor),
        f_st_rminor=FromExactly(lambda s: s.stellarator.f_st_rminor),
        rminor=FromExactly(lambda s: s.physics.rminor),
        stella_config_vol_plasma=FromExactly(
            lambda s: s.stellarator_config.stella_config_vol_plasma
        ),
        stella_config_plasma_surface=FromExactly(
            lambda s: s.stellarator_config.stella_config_plasma_surface
        ),
    ):
        return calculate_stellarator_plasma_geometry(
            f_st_rmajor,
            f_st_rminor,
            rminor,
            stella_config_vol_plasma,
            stella_config_plasma_surface,
        )
