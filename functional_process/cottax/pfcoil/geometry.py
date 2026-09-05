"""Where the CS and the PF coils are: cross-sections, filament placement, coil centres.

Audit record: `functional_process/_audit/units/models/pfcoil/geometry.md`.

Five units, all straight-line algebra with no iteration and no call into another model:

- `calculate_cs_geometry` -- `CSCoil.calculate_cs_geometry`
  (`process/models/pfcoil.py:3005-3072`), already a `@staticmethod` with explicit
  arguments; the port is a signature promotion plus `np.` -> `jnp.`.
- `calculate_cs_turn_geometry_eu_demo` -- `CSCoil.calculate_cs_turn_geometry_eu_demo`
  (`:3074-3149`), likewise, with `ohcalc`'s own `a_cs_turn` division (`:3297-3300`)
  folded into it. Added 2026-08-30 for `.tokamak.cs_fatigue`, which reads two of its
  outputs.
- `place_cs_filaments` -- `CSCoil.place_cs_filaments` (`:3151-3226`), likewise.
- `calculate_pf_coil_group_positions` -- the `i_pf_location` dispatch loop of
  `pfcoil()` (`:247-354`) collapsed to this run's four groups, calling
  `place_pf_above_tf` (`:1178-1263`) for groups 0-1 and `place_pf_outside_tf`
  (`:1265-1343`) for groups 2-3.
- `calculate_pf_coil_positions` -- `pfcoil()`'s group-array flattening (`:663-672`)
  together with the CS's own slot (`:176-194`).

**`top_bottom` is structural, not an input.** `pfcoil()` initialises `top_bottom = 1`
(`:127`) and `place_pf_above_tf` flips it every time it places a coil (`:1254-1261`), so
which side of the midplane an `i_pf_location = 2` group lands on is decided by that
group's *position in the group ordering*, not by any field. On this run the two such
groups hold one coil each, so group 0 goes above and group 1 below. That is a fact about
the assembled graph, not a value flowing along an edge, and it is baked into
`calculate_pf_coil_group_positions` below rather than passed in -- a third
`i_pf_location = 2` group would be a different occupant, not a different argument.

**Switches this file bakes in**, all at their `large_tokamak_eval.IN.DAT` value; every
other value is UNPORTED with its reason in `geometry.md`:

| switch | value here |
|---|---|
| `.pf_coil.i_pf_location` | `(2, 2, 3, 3)` -- above TF x2, then outside TF x2 |
| `.physics.itart` / `.physics.itartpf` | `0` / `0` (not a spherical tokamak) |
| `.tfcoil.i_tf_shape` | `1` = `D_SHAPE` (not `PICTURE_FRAME`) |
| `.pf_coil.i_r_pf_outside_tf_placement` | `0` -- radius follows the TF curve |
"""

import equinox as eqx
import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.cottax.pfcoil import (
    NGC2,
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
from functional_process.models.pfcoil.geometry import (
    calculate_cs_geometry,  # noqa: F401 -- re-exported for tests
    calculate_cs_geometry_ports,
    calculate_cs_turn_geometry_eu_demo,  # noqa: F401 -- re-exported for tests
    calculate_cs_turn_geometry_eu_demo_from_turns,
    calculate_pf_coil_group_positions,  # noqa: F401 -- re-exported for tests
    calculate_pf_coil_placement_for_topology,
    calculate_pf_coil_positions,
    place_cs_filaments,  # noqa: F401 -- re-exported for currents.py / tests
)


class CSCoilGeometry(ExplicitFunction):
    """cottax node: `.tokamak.cs_coil.geometry`. Owns the CS's own cross-section and
    edge coordinates -- the ten scalar `.pf_coil.*cs*` fields `calculate_cs_geometry`
    produces.

    PROCESS writes these twice per pass, once at the head of `pfcoil()`
    (`process/models/pfcoil.py:169-198`) and once at the head of `ohcalc()`
    (`:3230-3259`), from the same four inputs with the same expression. One node, not
    two: the second write is idempotent, so there is nothing for a second occupant to
    do that this one does not.

    The four `.pf_coil.{r,z}_pf_coil_*[6]` array slots those same two blocks fill are
    **not** owned here; `masses.py`'s `PFCoilSizes` owns those arrays whole, reading the
    scalars this node produces. Splitting them any other way would leave one array with
    two owners.
    """

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

    def __call__(
        self,
        z_tf_inside_half=From(build),
        f_z_cs_tf_internal=From(pf_coil),
        dr_cs=From(build),
        dr_cs_bore=From(build),
    ):
        # `CSGeometry.r_cs_coil_middle` is dropped: it is bit-for-bit `r_cs_middle`
        # (`pfcoil.py:3030`, `:3042`) and `DataStructure`'s `PfCoilVariables` has no
        # field of that name -- PROCESS stores it only into
        # `r_pf_coil_middle[n_cs_pf_coils - 1]`, which `PFCoilPositions` owns. Owning it
        # here would mint a `VarPath` that names no place.
        return calculate_cs_geometry_ports(
            z_tf_inside_half=z_tf_inside_half,
            f_z_cs_tf_internal=f_z_cs_tf_internal,
            dr_cs=dr_cs,
            dr_cs_bore=dr_cs_bore,
        )


class CSCoilTurnGeometry(ExplicitFunction):
    """cottax node: `.tokamak.cs_coil.turn_geometry`. Owns the CS turn's dimensions and
    the two conduit thicknesses `ncycle` reads. No switch.

    **The two conduit thicknesses land in `.cs_fatigue`, not `.pf_coil`**, which is
    PROCESS's own placement (`pfcoil.py:3314-3319`) and not a choice made here: they are
    written by `ohcalc` and read only by `CsFatigue.ncycle`, and the area they live in
    follows the reader rather than the writer. That is the one cross-area edge this node
    makes, and it is why `.tokamak.cs_coil` gains a slot for a calculation whose other
    four outputs nothing in this graph reads.
    """

    a_cs_turn = OutputInto(pf_coil)
    dz_cs_turn = OutputInto(pf_coil)
    dr_cs_turn = OutputInto(pf_coil)
    radius_cs_turn_cable_space = OutputInto(pf_coil)
    dr_cs_turn_conduit = OutputInto(cs_fatigue)
    dz_cs_turn_conduit = OutputInto(cs_fatigue)

    def __call__(
        self,
        a_cs_poloidal=From(pf_coil),
        n_pf_coil_turns=From(pf_coil),
        f_dr_dz_cs_turn=From(pf_coil),
        radius_cs_turn_corners=From(pf_coil),
        f_a_cs_turn_steel=From(pf_coil),
    ):
        return calculate_cs_turn_geometry_eu_demo_from_turns(
            a_cs_poloidal=a_cs_poloidal,
            n_pf_coil_turns=n_pf_coil_turns,
            f_dr_dz_cs_turn=f_dr_dz_cs_turn,
            radius_cs_turn_corners=radius_cs_turn_corners,
            f_a_cs_turn_steel=f_a_cs_turn_steel,
        )


class PFCoilPlacement(ExplicitFunction):
    """cottax node: `.tokamak.pf_coil.placement`.

    Occupant for `i_pf_location = (2, 2, 3, 3)` with `not (itart == 1 and itartpf == 0)`,
    `i_tf_shape = D_SHAPE` and `i_r_pf_outside_tf_placement = 0`. Owns the two
    `(N_PF_GROUPS_MAX, 2)` group arrays and `.pf_coil.r_pf_outside_tf_midplane`
    (`pfcoil.py:239-242`, one line, folded in here because it is this placement's own
    input and has no other producer or consumer).

    `r_cs_middle` is *not* read: no group on this arm has `i_pf_location = 1`, and it is
    only `place_pf_above_cs` that needs it. Declaring it would be exactly the
    union-of-arms invented edge the occupant split exists to remove. `rref` is not read
    for the same reason -- no group here has `i_pf_location = 4`.
    """

    topology: PFCoilTopology = eqx.field(static=True, default=REFERENCE_TOPOLOGY)
    """Static, and the reference topology by construction: this occupant's whole
    identity is that pattern. `PFCoilPlacementSphericalTokamak` carries the other one."""

    r_pf_outside_tf_is_constant: bool = eqx.field(static=True, default=False)
    """`i_tf_shape == PICTURE_FRAME or i_r_pf_outside_tf_placement == 1`
    (`pfcoil.py:1322-1326`), resolved once. `False` here -- a D-shaped TF with the
    default placement, so an outside-TF coil's radius follows the TF curve."""

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
    """cottax node: `.tokamak.pf_coil.placement`, the spherical tokamaks' occupant.

    Occupant for `i_pf_location = (2, 3, 3, 4)` with
    `n_pf_coils_in_group = (2, 2, 2, 2)`, `i_tf_shape = PICTURE_FRAME` and
    `i_r_pf_outside_tf_placement = 1` --
    `spherical_tokamak_eval.IN.DAT` (`:233`, `:236-237`, `:357`) and
    `st_regression.IN.DAT` (`:1755`, `:1764`, `:1788`, `:803`), which set the same four.

    **Three differences from `PFCoilPlacement`, all of them structural**, and none of
    them a value:

    1. The topology. Group 0 holds *two* `i_pf_location = 2` coils, so `pfcoil()`'s
       `top_bottom` toggle flips inside one group instead of between two; groups 1 and 2
       are the outside-TF pairs; group 3 is the `i_pf_location = 4` pair, which
       `place_pf_generally` (`pfcoil.py:1345-1401`) places from `rref`/`zref` about the
       plasma centre.
    2. `r_pf_outside_tf_is_constant`. `i_tf_shape = 2` and
       `i_r_pf_outside_tf_placement = 1` are the two halves of one disjunction
       (`:1322-1326`); either alone stacks the outside-TF coils at the midplane radius
       instead of following the TF curve, and both are set on both files.
    3. **`rref` is a read here and is not one on the conventional arm.** That is the
       whole reason this is a second occupant rather than a second static field: the
       `i_pf_location = 4` group is the only thing in `pfcoil()` that touches
       `.pf_coil.rref`, so declaring it on an arm with no such group would be an
       invented edge.

    `itart`/`itartpf` do **not** enter: both files set `itartpf = 1`, and
    `place_pf_above_tf`'s spherical-tokamak arm is guarded by `itart == 1 and
    itartpf == 0` (`:1250`). See `indat._pf_coil_system_deviations`' `-3`.
    """

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


class PFCoilPositions(ExplicitFunction):
    """cottax node: `.tokamak.pf_coil.positions`. Owns `.pf_coil.r_pf_coil_middle` and
    `.pf_coil.z_pf_coil_middle` at their full `NGC2` width -- the PF coils flattened out
    of the group arrays, then the CS, then structural zeros.

    The plasma's index is *not* written by PROCESS in these two arrays -- `pfcoil()`'s
    "Plasma size and shape" block (`:1067-1079`) sets the plasma's inner/outer radius and
    upper/lower height but never its centre -- so it stays zero here too.
    """

    topology: PFCoilTopology = eqx.field(static=True, default=REFERENCE_TOPOLOGY)
    """Static. Which slot each coil occupies, and whether there is a CS slot at all."""

    r_pf_coil_middle = OutputInto(pf_coil)
    z_pf_coil_middle = OutputInto(pf_coil)

    def __call__(
        self,
        r_pf_coil_middle_group_array=From(pf_coil),
        z_pf_coil_middle_group_array=From(pf_coil),
        r_cs_middle=From(pf_coil),
    ):
        return self._flattened(
            r_pf_coil_middle_group_array,
            z_pf_coil_middle_group_array,
            r_cs_middle,
        )

    def _flattened(
        self,
        r_pf_coil_middle_group_array,
        z_pf_coil_middle_group_array,
        r_cs_middle,
    ):
        """The flattening and its `NGC2` padding, given this arm's reads."""
        n_groups = self.topology.n_pf_coil_groups
        r_flat, z_flat = calculate_pf_coil_positions(
            r_pf_coil_middle_group_array=r_pf_coil_middle_group_array[:n_groups],
            z_pf_coil_middle_group_array=z_pf_coil_middle_group_array[:n_groups],
            r_cs_middle=r_cs_middle,
            topology=self.topology,
        )
        pad = jnp.zeros(NGC2)
        filled = self.topology.n_cs_pf_coils
        return (
            pad.at[:filled].set(r_flat),
            pad.at[:filled].set(z_flat),
        )


class PFCoilPositionsNoCentralSolenoid(PFCoilPositions):
    """cottax node: `.tokamak.pf_coil.positions`, the `iohcl = 0` occupant.

    **`r_cs_middle` is not read**, and that is the whole difference. With no central
    solenoid there is no slot for it in either array -- `pfcoil()`'s CS write at `:182`
    lands on `n_cs_pf_coils - 1`, which with `iohcl = 0` is the *last PF coil's* index,
    and the `ncl` loop three hundred lines later (`:663-672`) overwrites it with that
    coil's own centre. So the CS geometry never survives into these arrays on this arm,
    and reading it would be an edge to a namespace this machine does not have.
    """

    topology: PFCoilTopology = eqx.field(static=True, default=SPHERICAL_TOKAMAK_TOPOLOGY)

    def __call__(
        self,
        r_pf_coil_middle_group_array=From(pf_coil),
        z_pf_coil_middle_group_array=From(pf_coil),
    ):
        return self._flattened(
            r_pf_coil_middle_group_array,
            z_pf_coil_middle_group_array,
            r_cs_middle=None,
        )
