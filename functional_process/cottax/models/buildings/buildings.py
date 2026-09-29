"""Pure-functional port of `process/models/buildings.py`'s `Buildings.run()`.

Unit #15.
"""

import jax.numpy as jnp  # noqa: F401

from functional_process.cottax.paths import (
    buildings,
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
)


# Every other `i_hcd_primary` value -- `calculate_bldgs_sizes_other_hcd`.
BldgsSizesOtherHcd = buildings.function(
    calculate_bldgs_sizes_other_hcd,
    "reactor_hall_l",
    "reactor_hall_w",
    "reactor_hall_h",
    "a_plant_floor_effective",
    "volnucb",
)
