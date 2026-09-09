"""Where the CS and the PF coils are: cross-sections, filament placement, coil centres."""

import equinox as eqx
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.cottax.models.pfcoil import (
    REFERENCE_TOPOLOGY,
    SPHERICAL_TOKAMAK_TOPOLOGY,
    PFCoilTopology,
)
from functional_process.cottax.paths import (
    build,
    cs_fatigue,
    pf_coil,
    physics,
    superconducting_tfcoil,
)
from functional_process.cottax.wraps import WrapsFunction
from functional_process.models.pfcoil.geometry import (
    calculate_cs_geometry,  # noqa: F401 -- re-exported for tests
    calculate_cs_geometry_ports,
    calculate_cs_turn_geometry_eu_demo,  # noqa: F401 -- re-exported for tests
    calculate_cs_turn_geometry_eu_demo_from_turns,
    calculate_pf_coil_group_positions,  # noqa: F401 -- re-exported for tests
    calculate_pf_coil_placement_for_topology,
    calculate_pf_coil_positions,  # noqa: F401 -- re-exported for tests
    calculate_pf_coil_positions_for_topology,
    calculate_pf_coil_positions_from_elements,
    place_cs_filaments,  # noqa: F401 -- re-exported for currents.py / tests
)


class CSCoilGeometry(WrapsFunction):
    """cottax node: `.tokamak.cs_coil.geometry`.

    `CSGeometry.r_cs_coil_middle` is dropped: it is bit-for-bit `r_cs_middle`
    (`pfcoil.py:3030`, `:3042`) and `DataStructure`'s `PfCoilVariables` has no field of
    that name -- PROCESS stores it only into `r_pf_coil_middle[n_cs_pf_coils - 1]`,
    which `PFCoilPositions` owns. Owning it here would mint a `VarPath` that names no
    place.
    """

    fn = calculate_cs_geometry_ports

    z_tf_inside_half = From(build)
    f_z_cs_tf_internal = From(pf_coil)
    dr_cs = From(build)
    dr_cs_bore = From(build)

    z_cs_upper = OutputInto(pf_coil)
    z_cs_lower = OutputInto(pf_coil)
    r_cs_middle = OutputInto(pf_coil)
    z_cs_middle = OutputInto(pf_coil)
    r_cs_outer = OutputInto(pf_coil)
    r_cs_inner = OutputInto(pf_coil)
    a_cs_poloidal = OutputInto(pf_coil)
    a_cs_toroidal = OutputInto(pf_coil)
    dz_cs_full = OutputInto(pf_coil)
    dr_cs_full = OutputInto(pf_coil)


class CSCoilTurnGeometry(WrapsFunction):
    """cottax node: `.tokamak.cs_coil.turn_geometry`."""

    fn = calculate_cs_turn_geometry_eu_demo_from_turns

    a_cs_poloidal = From(pf_coil)
    n_pf_coil_turns = From(pf_coil)
    f_dr_dz_cs_turn = From(pf_coil)
    radius_cs_turn_corners = From(pf_coil)
    f_a_cs_turn_steel = From(pf_coil)

    a_cs_turn = OutputInto(pf_coil)
    dz_cs_turn = OutputInto(pf_coil)
    dr_cs_turn = OutputInto(pf_coil)
    radius_cs_turn_cable_space = OutputInto(pf_coil)
    dr_cs_turn_conduit = OutputInto(cs_fatigue)
    dz_cs_turn_conduit = OutputInto(cs_fatigue)


class PFCoilPlacement(ExplicitFunction):
    """cottax node: `.tokamak.pf_coil.placement`."""

    topology: PFCoilTopology = eqx.field(static=True, default=REFERENCE_TOPOLOGY)
    """Static, and the reference topology by construction: this occupant's whole
    identity is that pattern.
    """

    r_pf_outside_tf_is_constant: bool = eqx.field(static=True, default=False)
    """`i_tf_shape == PICTURE_FRAME or i_r_pf_outside_tf_placement == 1`
    (`pfcoil.py:1322-1326`), resolved once.
    """

    r_pf_outside_tf_midplane = OutputInto(pf_coil)
    r_pf_coil_middle_group_array = OutputInto(pf_coil)
    z_pf_coil_middle_group_array = OutputInto(pf_coil)

    def __call__(
        self,
        r_tf_outboard_out=From(superconducting_tfcoil),
        dr_pf_tf_outboard_out_offset=From(pf_coil),
        rmajor=From(physics),
        rminor=From(physics),
        triang=From(physics),
        rpf2=From(pf_coil),
        z_tf_top=From(build),
        dz_tf_upper_lower_midplane=From(build),
        zref=From(pf_coil),
    ):
        return calculate_pf_coil_placement_for_topology(
            r_tf_outboard_out=r_tf_outboard_out,
            dr_pf_tf_outboard_out_offset=dr_pf_tf_outboard_out_offset,
            rmajor=rmajor,
            rminor=rminor,
            triang=triang,
            rpf2=rpf2,
            z_tf_top=z_tf_top,
            dz_tf_upper_lower_midplane=dz_tf_upper_lower_midplane,
            zref=zref,
            rref=None,
            topology=self.topology,
            r_pf_outside_tf_is_constant=self.r_pf_outside_tf_is_constant,
        )


class PFCoilPlacementSphericalTokamak(PFCoilPlacement):
    """cottax node: `.tokamak.pf_coil.placement`, the spherical tokamaks' occupant."""

    topology: PFCoilTopology = eqx.field(static=True, default=SPHERICAL_TOKAMAK_TOPOLOGY)
    r_pf_outside_tf_is_constant: bool = eqx.field(static=True, default=True)

    def __call__(
        self,
        r_tf_outboard_out=From(superconducting_tfcoil),
        dr_pf_tf_outboard_out_offset=From(pf_coil),
        rmajor=From(physics),
        rminor=From(physics),
        triang=From(physics),
        rpf2=From(pf_coil),
        z_tf_top=From(build),
        dz_tf_upper_lower_midplane=From(build),
        zref=From(pf_coil),
        rref=From(pf_coil),
    ):
        return calculate_pf_coil_placement_for_topology(
            r_tf_outboard_out=r_tf_outboard_out,
            dr_pf_tf_outboard_out_offset=dr_pf_tf_outboard_out_offset,
            rmajor=rmajor,
            rminor=rminor,
            triang=triang,
            rpf2=rpf2,
            z_tf_top=z_tf_top,
            dz_tf_upper_lower_midplane=dz_tf_upper_lower_midplane,
            zref=zref,
            rref=rref,
            topology=self.topology,
            r_pf_outside_tf_is_constant=self.r_pf_outside_tf_is_constant,
        )


class PFCoilPositions(WrapsFunction):
    """cottax node: `.tokamak.pf_coil.positions`.

    Which slot each coil occupies, and whether there is a CS slot at all, is fixed by
    `fn` (`calculate_pf_coil_positions_from_elements`, for `REFERENCE_TOPOLOGY`) rather
    than carried as a static field here -- see `PFCoilPositionsNoCentralSolenoid` for
    the other arm.
    """

    fn = calculate_pf_coil_positions_from_elements

    r_pf_coil_middle_group_array = From(pf_coil)
    z_pf_coil_middle_group_array = From(pf_coil)
    r_cs_middle = From(pf_coil)

    r_pf_coil_middle = OutputInto(pf_coil)
    z_pf_coil_middle = OutputInto(pf_coil)


class PFCoilPositionsNoCentralSolenoid(ExplicitFunction):
    """cottax node: `.tokamak.pf_coil.positions`, the `iohcl = 0` occupant.

    Not `WrapsFunction`: `calculate_pf_coil_positions_for_topology` takes a `topology`
    keyword this occupant must thread through explicitly (`self.topology`).
    """

    topology: PFCoilTopology = eqx.field(static=True, default=SPHERICAL_TOKAMAK_TOPOLOGY)

    r_pf_coil_middle = OutputInto(pf_coil)
    z_pf_coil_middle = OutputInto(pf_coil)

    def __call__(
        self,
        r_pf_coil_middle_group_array=From(pf_coil),
        z_pf_coil_middle_group_array=From(pf_coil),
    ):
        return calculate_pf_coil_positions_for_topology(
            r_pf_coil_middle_group_array=r_pf_coil_middle_group_array,
            z_pf_coil_middle_group_array=z_pf_coil_middle_group_array,
            r_cs_middle=None,
            topology=self.topology,
        )
