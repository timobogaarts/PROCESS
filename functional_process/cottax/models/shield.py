"""Pure-functional port of `process/models/shield.py` (partial -- see "not ported")."""

import dataclasses

import equinox as eqx
from cottax.cx import Explicit, V

from functional_process.cottax.paths import blanket
from functional_process.models.engineering.ivc_functions import (
    dshellvol,  # noqa: F401
    eshellvol,  # noqa: F401
)
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


# The ShieldHalfHeight family:
# The family that owns `.blanket.dz_shld_half`: one occupant per `n_divertors`
# branch of `Shield.calculate_shield_half_height`.


# `n_divertors == 2`. Not live on `large_tokamak_eval.IN.DAT` (`n_divertors=1`).
DoubleNullShieldHalfHeight = blanket.function(
    calculate_shield_half_height_double_null,
    "dz_shld_half",
)


# `n_divertors != 2` -- the arm `large_tokamak_eval.IN.DAT` takes
# (`n_divertors=1`).
SingleNullShieldHalfHeight = blanket.function(
    calculate_shield_half_height_single_null,
    "dz_shld_half",
)


# The ShieldVolumes family:
# The family that owns `.blanket.vol_shld_inboard`/`.blanket.vol_shld_outboard`/
# `.fwbs.vol_shld_total`: one occupant per arm of the compound switch `itart == 1 or
# i_fw_blkt_vv_shape == D_SHAPED` (`process/models/shield.py:48-51`).


# `itart != 1 and i_fw_blkt_vv_shape != D_SHAPED` -- the arm
# `large_tokamak_eval.IN.DAT` takes.
EllipticalShieldVolumes = blanket.function(
    calculate_shield_volumes_elliptical,
    "vol_shld_inboard",
    "vol_shld_outboard",
    V.fwbs.vol_shld_total,
)


# `itart == 1 or i_fw_blkt_vv_shape == D_SHAPED` -- the arm
# `spherical_tokamak_eval.IN.DAT` and `st_regression.IN.DAT` take, both of them twice
# over (`itart = 1` **and** `i_fw_blkt_vv_shape = 1`).
DShapedShieldVolumes = blanket.function(
    calculate_shield_volumes_dshaped,
    "vol_shld_inboard",
    "vol_shld_outboard",
    V.fwbs.vol_shld_total,
)


class TokamakShield(eqx.Module):
    """`.tokamak.shield` -- two slots, both switched."""

    half_height: Explicit = dataclasses.field(kw_only=True)
    """`.divertor.n_divertors` (derived from `i_single_null` by `indat._n_divertors`) --
    both values of this binary switch have written occupants, `1` (single null, live)
    and `2` (double null).
    """

    volumes: Explicit = dataclasses.field(kw_only=True)
    """`_fw_blkt_vv_shape_arm` -- **both** arms written (2026-08-27): the elliptical arm
    (`1`) since wave 1, the D-shaped arm (`0`) since the D-shaped wave.
    """
