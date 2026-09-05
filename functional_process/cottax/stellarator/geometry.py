"""Pure-functional port of `Stellarator.st_new_config`/`st_geom` (chunk 1C of unit #1).

Audit record:
`functional_process/_audit/units/models/stellarator/geometry.md`. Source:
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

import jax.numpy as jnp  # noqa: F401
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    OutputInto,
)

from functional_process.cottax.paths import (
    physics,
    stellarator,
    stellarator_config,
    tfcoil,
)
from functional_process.models.stellarator.geometry import (
    calculate_default_aspect_ratio,
    calculate_stellarator_plasma_geometry,
    calculate_stellarator_scaling_factors,
)


class DefaultAspectRatio(ExplicitFunction):
    """cottax node: `calculate_default_aspect_ratio`, ports declared.

    Only instantiate this node when `1 not in data.numerics.ixc` -- see module
    docstring. When `aspect` *is* an active iteration variable, no node in this chunk
    owns `.physics.aspect` at all.
    """

    aspect = OutputInto(physics)

    def __call__(
        self,
        stella_config_aspect_ref=From(stellarator_config),
    ):
        return calculate_default_aspect_ratio(stella_config_aspect_ref)


class StellaratorScalingFactors(ExplicitFunction):
    """cottax node: `calculate_stellarator_scaling_factors`, ports declared.

    `aspect` read from wherever the `1 in ixc` graph-assembly choice puts it --
    `DefaultAspectRatio`'s output, or an external (iteration-variable) input, per module
    docstring.
    """

    rminor = OutputInto(physics)
    eps = OutputInto(physics)
    n_tf_coils = OutputInto(tfcoil)
    f_st_rmajor = OutputInto(stellarator)
    f_st_rminor = OutputInto(stellarator)
    f_st_aspect = OutputInto(stellarator)
    f_st_n_coils = OutputInto(stellarator)
    f_st_b = OutputInto(stellarator)
    r_coil_major = OutputInto(stellarator)
    r_coil_minor = OutputInto(stellarator)
    f_coil_shape = OutputInto(stellarator)

    def __call__(
        self,
        rmajor=From(physics),
        aspect=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
        f_st_coil_aspect=From(stellarator),
        stella_config_coilspermodule=From(stellarator_config),
        stella_config_symmetry=From(stellarator_config),
        stella_config_rmajor_ref=From(stellarator_config),
        stella_config_rminor_ref=From(stellarator_config),
        stella_config_aspect_ref=From(stellarator_config),
        stella_config_bt_ref=From(stellarator_config),
        stella_config_coil_rmajor=From(stellarator_config),
        stella_config_coil_rminor=From(stellarator_config),
        stella_config_min_plasma_coil_distance=From(stellarator_config),
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

    vol_plasma = OutputInto(physics)
    a_plasma_surface = OutputInto(physics)
    a_plasma_poloidal = OutputInto(physics)
    a_plasma_surface_outboard = OutputInto(physics)

    def __call__(
        self,
        f_st_rmajor=From(stellarator),
        f_st_rminor=From(stellarator),
        rminor=From(physics),
        stella_config_vol_plasma=From(stellarator_config),
        stella_config_plasma_surface=From(stellarator_config),
    ):
        return calculate_stellarator_plasma_geometry(
            f_st_rmajor,
            f_st_rminor,
            rminor,
            stella_config_vol_plasma,
            stella_config_plasma_surface,
        )
