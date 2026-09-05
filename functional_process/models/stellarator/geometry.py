"""Pure physics functions extracted from
`functional_process.cottax.stellarator.geometry`, per `_audit/formulas_split.md`
step 2 phase A. The graph declarations that wrap these stay in that sibling
module.
"""

import jax.numpy as jnp


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
    f_st_n_coils = n_tf_coils / (stella_config_coilspermodule * stella_config_symmetry)
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
