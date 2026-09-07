"""Pure-functional port of `process/models/shield.py` (partial -- see "not ported")."""

import dataclasses

from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    ModelNamespace,
    OutputInto,
)

from functional_process.models.engineering.ivc_functions import (
    dshellvol,  # noqa: F401
    eshellvol,  # noqa: F401
)
from functional_process.cottax.paths import blanket, build, divertor, fwbs, physics
from functional_process.models.shield import (
    apply_shield_volume_coverage_factors,  # noqa: F401
    calculate_dshaped_shield_volumes,  # noqa: F401
    calculate_elliptical_shield_volumes,  # noqa: F401
    calculate_shield_half_height_double_null,
    calculate_shield_half_height_single_null,
    calculate_shield_volumes_dshaped,
    calculate_shield_volumes_elliptical,
)

# ---------------------------------------------------------------------------
# cottax nodes
# ---------------------------------------------------------------------------


class ShieldHalfHeight(ExplicitFunction):
    """The family that owns `.blanket.dz_shld_half`: one occupant per `n_divertors`
    branch of `Shield.calculate_shield_half_height`.
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
    `.fwbs.vol_shld_total`: one occupant per arm of the compound switch `itart == 1 or
    i_fw_blkt_vv_shape == D_SHAPED` (`process/models/shield.py:48-51`).
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


class DShapedShieldVolumes(ShieldVolumes):
    """`itart == 1 or i_fw_blkt_vv_shape == D_SHAPED` -- the arm
    `spherical_tokamak_eval.IN.DAT` and `st_regression.IN.DAT` take, both of them twice
    over (`itart = 1` **and** `i_fw_blkt_vv_shape = 1`).
    """

    vol_shld_inboard = OutputInto(blanket)
    vol_shld_outboard = OutputInto(blanket)
    vol_shld_total = OutputInto(fwbs)

    def __call__(
        self,
        r_shld_inboard_inner=From(build),
        dr_shld_inboard=From(build),
        dr_fw_inboard=From(build),
        dr_fw_plasma_gap_inboard=From(build),
        rminor=From(physics),
        dr_fw_plasma_gap_outboard=From(build),
        dr_fw_outboard=From(build),
        dr_blkt_inboard=From(build),
        dr_blkt_outboard=From(build),
        dz_shld_half=From(blanket),
        dr_shld_outboard=From(build),
        dz_shld_upper=From(build),
        fvolsi=From(fwbs),
        fvolso=From(fwbs),
    ):
        return calculate_shield_volumes_dshaped(
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
        )


class TokamakShield(ModelNamespace):
    """`.tokamak.shield` -- two slots, both switched."""

    half_height: ShieldHalfHeight = dataclasses.field(kw_only=True)
    """`.divertor.n_divertors` (derived from `i_single_null` by `indat._n_divertors`) --
    both values of this binary switch have written occupants, `1` (single null, live)
    and `2` (double null).
    """

    volumes: ShieldVolumes = dataclasses.field(kw_only=True)
    """`_fw_blkt_vv_shape_arm` -- **both** arms written (2026-08-27): the elliptical arm
    (`1`) since wave 1, the D-shaped arm (`0`) since the D-shaped wave.
    """
