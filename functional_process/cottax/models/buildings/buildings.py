"""Pure-functional port of `process/models/buildings.py`'s `Buildings.run()`.

Unit #15.
"""

import jax.numpy as jnp  # noqa: F401

from functional_process.cottax.paths import (
    build,
    buildings,
    costs,
    divertor,
    fwbs,
    heat_transport,
    pf_coil,
    physics,
    tfcoil,
)
from functional_process.models.buildings.buildings import (
    calculate_bldgs,  # noqa: F401 -- re-exported for tests
    calculate_bldgs_from_elements,
    calculate_bldgs_sizes,  # noqa: F401 -- re-exported for tests
    calculate_bldgs_sizes_neutral_beam,
    calculate_bldgs_sizes_other_hcd,
    calculate_shield_height,  # noqa: F401 -- re-exported for tests
    calculate_tf_coil_envelope,
)
from functional_process.models.safe_math import safe_pow  # noqa: F401

# cottax node: `calculate_tf_coil_envelope`, ports declared.
TfCoilEnvelope = buildings.function(
    calculate_tf_coil_envelope,
    "tfro",
    "tfri",
    "tf_radial_dim",
    "tf_vertical_dim",
    "tfmtn",
    reads={
        "r_tf_outboard_mid": build,
        "dr_tf_outboard": build,
        "r_tf_inboard_mid": build,
        "dr_tf_inboard": build,
        "z_tf_inside_half": build,
        "m_tf_coils_total": tfcoil,
        "n_tf_coils": tfcoil,
    },
)


# cottax node: `calculate_bldgs_from_elements` --
# `calculate_bldgs`. Instantiate iff `i_bldgs_size == ITER_1992`.
Bldgs = buildings.function(
    calculate_bldgs_from_elements,
    "cryvol",
    "volrci",
    "rbvol",
    "rmbvol",
    "wsvol",
    "elevol",
    "wrbi",
    "a_plant_floor_effective",
    "admvol",
    "shovol",
    "convol",
    "volnucb",
    reads={
        "r_pf_coil_outer_max": pf_coil,
        "m_pf_coil_max": pf_coil,
        "n_tf_coils": tfcoil,
        "r_shld_outboard_outer": build,
        "r_shld_inboard_inner": build,
        "z_tf_inside_half": build,
        "dz_shld_vv_gap": build,
        "dz_vv_upper": build,
        "dz_vv_lower": build,
        "whtshld": fwbs,
        "r_cryostat_inboard": fwbs,
        "helpow": heat_transport,
    },
)


# The BldgsSizesBase family:
# cottax node family: `calculate_bldgs_sizes` -- bodiless base. Two occupants,
# keyed on `.current_drive.i_hcd_primary`'s method: `BldgsSizesNeutralBeam`/
# `BldgsSizesOtherHcd`. `i_hcd_primary` has 12 values but `bldgs_sizes` reads it to
# decide one bit (whether the method is neutral-beam-shaped) -- see this unit's
# `models/` module docstring. Reads are declared once, here, and shared by both arms;
# each arm names only its `fn`.


# `i_hcd_primary`'s method is `NEUTRAL_BEAM` (`ITER_NEUTRAL_BEAM`/
# `CULHAM_NEUTRAL_BEAM`) -- `calculate_bldgs_sizes_neutral_beam`.
BldgsSizesNeutralBeam = buildings.function(
    calculate_bldgs_sizes_neutral_beam,
    "reactor_hall_l",
    "reactor_hall_w",
    "reactor_hall_h",
    "a_plant_floor_effective",
    "volnucb",
    reads={
        "r_pf_coil_outer_max": pf_coil,
        "r_cryostat_inboard": fwbs,
        "life_plant": costs,
        "z_tf_inside_half": build,
        "dr_tf_inboard": build,
        "dr_tf_shld_gap": build,
        "dz_shld_thermal": build,
        "dz_shld_vv_gap": build,
        "dr_shld_inboard": build,
        "dr_blkt_inboard": build,
        "dr_fw_inboard": build,
        "rmajor": physics,
        "rminor": physics,
        "dr_fw_plasma_gap_inboard": build,
        "n_tf_coils": tfcoil,
        "dr_fw_outboard": build,
        "dr_blkt_outboard": build,
        "dr_shld_outboard": build,
        "dr_fw_plasma_gap_outboard": build,
        "life_div_fpy": costs,
        "dz_divertor": divertor,
        "cplife": costs,
        "i_tf_sup": tfcoil,
        "r_cp_top": build,
    },
)


# Every other `i_hcd_primary` value -- `calculate_bldgs_sizes_other_hcd`.
BldgsSizesOtherHcd = buildings.function(
    calculate_bldgs_sizes_other_hcd,
    "reactor_hall_l",
    "reactor_hall_w",
    "reactor_hall_h",
    "a_plant_floor_effective",
    "volnucb",
    reads={
        "r_pf_coil_outer_max": pf_coil,
        "r_cryostat_inboard": fwbs,
        "life_plant": costs,
        "z_tf_inside_half": build,
        "dr_tf_inboard": build,
        "dr_tf_shld_gap": build,
        "dz_shld_thermal": build,
        "dz_shld_vv_gap": build,
        "dr_shld_inboard": build,
        "dr_blkt_inboard": build,
        "dr_fw_inboard": build,
        "rmajor": physics,
        "rminor": physics,
        "dr_fw_plasma_gap_inboard": build,
        "n_tf_coils": tfcoil,
        "dr_fw_outboard": build,
        "dr_blkt_outboard": build,
        "dr_shld_outboard": build,
        "dr_fw_plasma_gap_outboard": build,
        "life_div_fpy": costs,
        "dz_divertor": divertor,
        "cplife": costs,
        "i_tf_sup": tfcoil,
        "r_cp_top": build,
    },
)
