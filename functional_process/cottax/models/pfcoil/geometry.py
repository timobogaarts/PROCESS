"""Where the CS and the PF coils are: cross-sections, filament placement, coil centres."""

from cottax.interfaces import V

from functional_process.cottax.paths import (
    build,
    pf_coil,
    physics,
    superconducting_tfcoil,
)
from functional_process.models.pfcoil.geometry import (
    calculate_cs_geometry,  # noqa: F401 -- re-exported for tests
    calculate_cs_geometry_ports,
    calculate_cs_turn_geometry_eu_demo,  # noqa: F401 -- re-exported for tests
    calculate_cs_turn_geometry_eu_demo_from_turns,
    calculate_pf_coil_group_positions,  # noqa: F401 -- re-exported for tests
    calculate_pf_coil_placement_for_topology,  # noqa: F401 -- re-exported for tests
    calculate_pf_coil_placement_no_central_solenoid,
    calculate_pf_coil_placement_reference,
    calculate_pf_coil_positions,  # noqa: F401 -- re-exported for tests
    calculate_pf_coil_positions_for_topology,  # noqa: F401 -- re-exported for tests
    calculate_pf_coil_positions_from_elements,
    calculate_pf_coil_positions_no_central_solenoid,
    place_cs_filaments,  # noqa: F401 -- re-exported for currents.py / tests
)

# cottax node: `.tokamak.cs_coil.geometry`.
#
# `CSGeometry.r_cs_coil_middle` is dropped: it is bit-for-bit `r_cs_middle`
# (`pfcoil.py:3030`, `:3042`) and `DataStructure`'s `PfCoilVariables` has no field of
# that name -- PROCESS stores it only into `r_pf_coil_middle[n_cs_pf_coils - 1]`,
# which `PFCoilPositions` owns. Owning it here would mint a `VarPath` that names no
# place.
CSCoilGeometry = pf_coil.function(
    calculate_cs_geometry_ports,
    "z_cs_upper",
    "z_cs_lower",
    "r_cs_middle",
    "z_cs_middle",
    "r_cs_outer",
    "r_cs_inner",
    "a_cs_poloidal",
    "a_cs_toroidal",
    "dz_cs_full",
    "dr_cs_full",
    reads={
        "z_tf_inside_half": build,
        "dr_cs": build,
        "dr_cs_bore": build,
    },
)


# cottax node: `.tokamak.cs_coil.turn_geometry`.
CSCoilTurnGeometry = pf_coil.function(
    calculate_cs_turn_geometry_eu_demo_from_turns,
    "a_cs_turn",
    "dz_cs_turn",
    "dr_cs_turn",
    "radius_cs_turn_cable_space",
    V.cs_fatigue.dr_cs_turn_conduit,
    V.cs_fatigue.dz_cs_turn_conduit,
)


# cottax node: `.tokamak.pf_coil.placement`.
PFCoilPlacement = pf_coil.function(
    calculate_pf_coil_placement_reference,
    "r_pf_outside_tf_midplane",
    "r_pf_coil_middle_group_array",
    "z_pf_coil_middle_group_array",
    reads={
        "r_tf_outboard_out": superconducting_tfcoil,
        "rmajor": physics,
        "rminor": physics,
        "triang": physics,
        "z_tf_top": build,
        "dz_tf_upper_lower_midplane": build,
    },
)


# cottax node: `.tokamak.pf_coil.placement`, the spherical tokamaks' occupant.
#
# Reads everything `PFCoilPlacement` does, plus `rref` -- the `GENERALLY_PLACED`
# group's radial placement ratio, which `REFERENCE_TOPOLOGY` never has and this
# topology does. A strict superset, so this stays the inheriting arm: it adds the one
# extra read and overrides `fn` to the topology/`r_pf_outside_tf_is_constant`-bound
# counterpart.
PFCoilPlacementSphericalTokamak = pf_coil.function(
    calculate_pf_coil_placement_no_central_solenoid,
    "r_pf_outside_tf_midplane",
    "r_pf_coil_middle_group_array",
    "z_pf_coil_middle_group_array",
    reads={
        "r_tf_outboard_out": superconducting_tfcoil,
        "rmajor": physics,
        "rminor": physics,
        "triang": physics,
        "z_tf_top": build,
        "dz_tf_upper_lower_midplane": build,
    },
)


# cottax node: `.tokamak.pf_coil.positions`.
#
# Which slot each coil occupies, and whether there is a CS slot at all, is fixed by
# `fn` (`calculate_pf_coil_positions_from_elements`, for `REFERENCE_TOPOLOGY`) rather
# than carried as a static field here -- see `PFCoilPositionsNoCentralSolenoid` for
# the other arm.
PFCoilPositions = pf_coil.function(
    calculate_pf_coil_positions_from_elements, "r_pf_coil_middle", "z_pf_coil_middle"
)


# cottax node: `.tokamak.pf_coil.positions`, the `iohcl = 0` occupant.
PFCoilPositionsNoCentralSolenoid = pf_coil.function(
    calculate_pf_coil_positions_no_central_solenoid,
    "r_pf_coil_middle",
    "z_pf_coil_middle",
)
