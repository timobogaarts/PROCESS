"""Pure-functional port of `process/models/vacuum.py` (registry unit #16).

Audit record: `functional_process/_audit/units/models/vacuum.md`. Entry point is
`Vacuum.run()`, which dispatches on the topology-changing switch
`.vacuum.i_vacuum_pumping` (`"old"`/`"simple"`) to one of two, essentially disjoint,
computations:

- **`"simple"`** -- `vacuum_simple`: straight-line algebra, no iteration.
  `calculate_vacuum_pumping_simple` below, tier-1.
- **`"old"`** -- `vacuum`: the ETR-derived detailed model. Straight-line algebra to
  build four required pumping speeds, then a genuine internal solve (Newton's method
  for a duct diameter, wrapped in an outer loop that shrinks the target conductance
  until the duct physically fits between TF coils) to size the pumping ducts. Tier-2.
  `calculate_vacuum_pumping_old` below.

`VacuumVessel` (the second class in the source file) is **out of scope on the
stellarator**: it is not reached from `Stellarator.run()` at all. `Stellarator.__init__`
(`process/models/stellarator/stellarator.py`) is injected a `vacuum: Vacuum` but no
`vacuum_vessel` -- confirmed by `process/main.py:668-669,729,783-784`
(`Models.__init__` constructs both `self.vacuum`/`self.vacuum_vessel` and calls
`self.vacuum_vessel.output()` only from the tokamak/general `main.py` output path,
never from `stellarator.py`). The
stellarator pipeline computes its own vacuum-vessel geometry inline
(`Stellarator.st_fwbs`'s "S5 cryostat_and_vv_geometry" chunk, see
`stellarator_E_fwbs_synthesis.md`) instead of calling `VacuumVessel`.

**`VacuumVessel` IS reached on the tokamak path** -- `caller.py:331`, confirming unit
#16's own prediction ("confirmed unreachable on the stellarator pipeline, no action
needed"). Ported below (wave-1 tokamak dispatch, `.tokamak.vacuum_vessel`): the minimal
closure for `.fwbs.m_vv`, the one variable `tokamak_boundary.md` lists on this slot.
See `vacuum.md`'s tokamak-scope addendum for the full trace, and this module's own
`VacuumVesselElliptical` docstring below for the switches baked in.
"""

import jax  # noqa: F401
import jax.numpy as jnp  # noqa: F401
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    ImplicitFunction,
    OutputInto,
    resolve,
)
from cottax.problem import Feasibility
from cottax.spec import In, Out, VarPath

from functional_process.models.engineering.ivc_functions import (
    dshellvol,  # noqa: F401
    eshellvol,  # noqa: F401
)
from functional_process.paths import (
    blanket,
    build,
    divertor,
    fwbs,
    physics,
    tfcoil,
    times,
    vacuum,
)
from functional_process.models.vacuum.vacuum import (
    XMULT,  # noqa: F401
    _solve_vacuum_pumping_old,  # noqa: F401
    _solve_vacuum_pumping_old_from_fields,  # noqa: F401
    calculate_dshaped_vessel_volumes,  # noqa: F401
    calculate_duct_feasibility_conditions,
    calculate_elliptical_vessel_volumes,  # noqa: F401
    calculate_vacuum_pumping_old,
    calculate_vacuum_pumping_simple,
    calculate_vacuum_vessel_mass,  # noqa: F401
    calculate_vacuum_vessel_outputs,
    calculate_vacuum_vessel_outputs_double_null,
    calculate_vacuum_vessel_outputs_dshaped_double_null,
    calculate_vessel_half_height,  # noqa: F401
    calculate_vessel_half_height_double_null,  # noqa: F401
    duct_conductance,  # noqa: F401
    duct_diameter_residual,
    duct_fits_residual,
    pumping_speed_floor_residual,
    solve_duct_diameter,  # noqa: F401
    solve_duct_geometry,  # noqa: F401
)


class VacuumPumpingSimple(ExplicitFunction):
    """cottax node: `calculate_vacuum_pumping_simple`'s combined pump count.

    Written back to `data` as `.vacuum.n_iter_vacuum_pumps`, per `Vacuum.run()`'s
    `"simple"` branch (`vp.n_iter_vacuum_pumps = self.vacuum_simple(output=output)`).
    """

    n_iter_vacuum_pumps = OutputInto(vacuum)

    def __call__(
        self,
        molflow_plasma_fuelling_required=From(physics),
        molflow_vac_pumps=From(vacuum),
        volflow_vac_pumps_max=From(vacuum),
        f_a_vac_pump_port_plasma_surface=From(vacuum),
        f_volflow_vac_pumps_impedance=From(vacuum),
        a_plasma_surface=From(physics),
        n_tf_coils=From(tfcoil),
        outgasfactor=From(vacuum),
        pres_vv_chamber_base=From(vacuum),
        outgasindex=From(vacuum),
        t_plant_pulse_dwell=From(times),
    ):
        return calculate_vacuum_pumping_simple(
            molflow_plasma_fuelling_required,
            molflow_vac_pumps,
            volflow_vac_pumps_max,
            f_a_vac_pump_port_plasma_surface,
            f_volflow_vac_pumps_impedance,
            a_plasma_surface,
            n_tf_coils,
            outgasfactor,
            pres_vv_chamber_base,
            outgasindex,
            t_plant_pulse_dwell,
        )


class DuctDiameterRootFind(ImplicitFunction):
    """cottax node: `duct_diameter_residual` as a genuine `RootFind` implicit model.

    Structural counterpart to `solve_duct_diameter` above -- same defining equation
    (`duct_diameter_residual`), declared rather than solved eagerly. `next_steps.md`
    §7 had earlier concluded `solve_duct_diameter` didn't need this treatment (its
    unknown is fully encapsulated inside `VacuumOld`'s own computation, so no other
    node reads it) -- that finding is **superseded for this unit by explicit
    instruction**, not re-derived here; see `vacuum.md` for the fuller discussion.
    `solve_duct_diameter` itself is kept unchanged and is still what any plain caller
    (including `solve_duct_geometry` below) should call -- this class exists
    alongside it, not instead of it, exactly as `duct_conductance` already sits
    alongside `_newton_function`'s closed-form half.

    Every `VarPath` here is **minted**, not an established `data` field: neither the
    duct diameter unknown nor `l1`/`l2`/`l3`/`xmult_i`/`ceff_i` has a `data`-reachable
    home today (all five are locals of `_solve_vacuum_pumping_old`'s per-species loop,
    see `vacuum.md`'s data footprint) -- same minting precedent as `coils.py`'s
    `JcritIterNb3sn` (`t_helium`/`b_max`) and the `Intersect` sketch at the bottom of
    that file. `.vacuum.d_duct` is a fresh name, chosen to avoid colliding with the
    already-established `.vacuum.dia_vv_vacuum_ducts` (the *final*, post-outer-loop
    winning diameter `VacuumOld` writes) -- this node's unknown is the per-species,
    per-outer-iteration Newton unknown, a different quantity at a different point in
    the computation. `l1`/`l2`/`l3`/`xmult_i`/`ceff_i` keep the plain parameter names
    `duct_diameter_residual` already uses.

    **Updated, later consolidation pass: registered in `total_process.py`.** Still not
    wired to any other node registered there -- every one of these six `VarPath`s is
    minted and unique to this class, so it sits as its own disconnected island in the
    default graph, same caution `coils.py`'s unregistered `Jcrit*` nodes are flagged
    with (see `total_process.py`'s own module docstring) -- registered anyway, on
    explicit instruction, as a perfectly valid undriven `RootFind` problem
    (`Graph.declared`, same as every other undriven declared node here). It does gain a
    real neighbour outside `total_process.py`, though: this file's own `DuctFeasibility`
    (below) reads `.vacuum.d_duct` as an ordinary cross-node `From`, forming a combined
    4-node cycle when the two are assembled together (see `DuctFeasibility`'s own
    docstring and `test_vacuum.py`).

    `functional_process/cottax/test_vacuum.py`'s
    `TestDuctDiameterRootFind` builds `to_graph(DuctDiameterRootFind)` directly and
    drives it with a test-only `AbstractDriver` (see that file) to confirm the two
    minted nodes (this body, and the `RootFind` problem `ImplicitFunction` also
    mints) assemble and converge to the same answer `solve_duct_diameter` does.
    """

    d_duct = OutputInto(vacuum)

    def residual(
        self,
        d_duct=From(vacuum),
        l1=From(vacuum),
        l2=From(vacuum),
        l3=From(vacuum),
        xmult_i=From(vacuum),
        ceff_i=From(vacuum),
    ):
        return duct_diameter_residual(d_duct, l1, l2, l3, xmult_i, ceff_i)


class DuctFeasibilityConditions(ExplicitFunction):
    """cottax node: the two inequality residuals `DuctFeasibility` (below) reads.

    A `ProblemNode` like `Feasibility` is bodyless -- it owns/reads pre-existing
    `VarPath`s, it does not compute them -- so the residuals themselves need an ordinary
    node to produce them, the same role `Intersect.residual`/
    `DuctDiameterRootFind.residual` play for their own `RootFind` problems. `d_duct` is
    read as a plain, non-owning
    `From` -- `DuctDiameterRootFind`'s `RootFind` problem owns it, an ordinary
    cross-node edge, not a second self-loop (same shape `WindingPackTotalSizePost`'s read
    of `.stellarator.wp_width_r_min` already established). `ceff_i` is read the same way
    -- `DuctFeasibility` (below) owns it as its one `design` unknown.
    """

    duct_fits_residual = OutputInto(vacuum)
    pumping_speed_floor_residual = OutputInto(vacuum)

    def __call__(
        self,
        d_duct=From(vacuum),
        a1max=From(vacuum),
        ceff_i=From(vacuum),
        s_i=From(vacuum),
    ):
        return calculate_duct_feasibility_conditions(d_duct, a1max, ceff_i, s_i)


DuctFeasibility = Feasibility(
    design=(Out(resolve(vacuum.ceff_i, VarPath)),),
    inequalities=(
        In(resolve(vacuum.duct_fits_residual, VarPath)),
        In(resolve(vacuum.pumping_speed_floor_residual, VarPath)),
    ),
)
"""The declared problem itself: "find a feasible `ceff_i`", no objective.

A bare `problem.py` `ProblemNode` instance like this one is not a `NodalDeclaration`
(`pytree_namespace_module.py`'s own class-based protocol, which `ExplicitFunction`/
`ImplicitFunction` implement) and, unlike those, carries no class-derived name of its
own -- `to_graph(DuctFeasibility)` alone raises `TypeError`. `to_graph` itself now
accepts a `{name: NodeDefinition}` mapping for exactly this case (fixed upstream in
`cottax.interfaces.{flat,pytree}_namespace_module.node_and_names`, since the same gap
applied to any bare `RootFind`/`Optimise`/`Feasibility` built directly, not just this
one): `to_graph(DuctFeasibilityConditions(), DuctDiameterRootFind(),
{"DuctFeasibility": DuctFeasibility})` assembles the full 4-node block in one call and
finds the combined cycle (`test_vacuum.py`'s own test does exactly this) -- no manual
`Graph(path_map(...))` construction needed any more.

Structurally this is `DuctFeasibility + DuctDiameterRootFind's RootFind` --
`Feasibility.__add__`'s `RootFind` branch (`design`/`equalities`/`inequalities`
concatenate) -- though the join itself is never invoked directly here: placing both
problem nodes and `DuctFeasibilityConditions`' residuals in one `Graph` lets
`Blocking`/`.cycles` find the same combined block structurally, the same way
`WindingPackIntersectInputs`/`Intersect`/`WindingPackTotalSizePost` never call
`Feasibility.__add__`/`Optimise.__add__` either -- the algebra states what a rewrite
*could* fold into one node; a plain shared-`VarPath` cycle across separately-registered
nodes already gets the same graph-level effect without invoking it.

Not registered in `total_process.py` (same as `Intersect`/`DuctDiameterRootFind` --
structural admission only, driving deferred) and not itself wired to
`DuctDiameterRootFind` there either, since `DuctDiameterRootFind` alone is what gets
registered (see that class's own docstring on why it is presently an island): joining
the two into one block is demonstrated in `test_vacuum.py`, not asserted by
registration."""


class VacuumOld(ExplicitFunction):
    """cottax node: `calculate_vacuum_pumping_old`'s five real outputs.

    Every read below is a genuine, already-existing `VarPath` -- no minting needed.
    `qtorus` is hardcoded `0.0` (not `From`-wrapped) since it is always `0.0` at
    `Vacuum.run()`'s only call site (see `vacuum.md`), a static default rather than a
    place in `data`.
    """

    n_vac_pumps_high = OutputInto(vacuum)
    n_vv_vacuum_ducts = OutputInto(vacuum)
    dlscal = OutputInto(vacuum)
    m_vv_vacuum_duct_shield = OutputInto(vacuum)
    dia_vv_vacuum_ducts = OutputInto(vacuum)

    def __call__(
        self,
        p_fusion_total_mw=From(physics),
        rmajor=From(physics),
        rminor=From(physics),
        dr_fw_plasma_gap_inboard=From(build),
        dr_fw_plasma_gap_outboard=From(build),
        a_plasma_surface=From(physics),
        vol_plasma=From(physics),
        dr_shld_outboard=From(build),
        dr_shld_inboard=From(build),
        dr_tf_inboard=From(build),
        r_shld_inboard_inner=From(build),
        dr_shld_vv_gap_inboard=From(build),
        dr_vv_inboard=From(build),
        n_tf_coils=From(tfcoil),
        t_plant_pulse_dwell=From(times),
        n_divertors=From(divertor),
        molflow_plasma_fuelling_required=From(physics),
        m_fuel_amu=From(physics),
        i_vac_pump_dwell=From(vacuum),
        i_vacuum_pump_type=From(vacuum),
        pres_vv_chamber_base=From(vacuum),
        pres_div_chamber_burn=From(vacuum),
        outgrat_fw=From(vacuum),
        t_plant_pulse_coil_precharge=From(times),
    ):
        return calculate_vacuum_pumping_old(
            p_fusion_total_mw,
            rmajor,
            rminor,
            dr_fw_plasma_gap_inboard,
            dr_fw_plasma_gap_outboard,
            a_plasma_surface,
            vol_plasma,
            dr_shld_outboard,
            dr_shld_inboard,
            dr_tf_inboard,
            r_shld_inboard_inner,
            dr_shld_vv_gap_inboard,
            dr_vv_inboard,
            n_tf_coils,
            t_plant_pulse_dwell,
            n_divertors,
            0.0,
            molflow_plasma_fuelling_required,
            m_fuel_amu,
            i_vac_pump_dwell,
            i_vacuum_pump_type,
            pres_vv_chamber_base,
            pres_div_chamber_burn,
            outgrat_fw,
            t_plant_pulse_coil_precharge,
        )


class VacuumVesselElliptical(ExplicitFunction):
    """The family that occupies `.tokamak.vacuum_vessel`: one occupant per cell of the
    shape x divertor-count grid (see the module comment above for the grid).

    Each occupant owns `.fwbs.m_vv` (`tokamak_boundary.md`'s one declared read of this
    slot) plus `dz_vv_half`, `vol_vv_inboard`, `vol_vv_outboard` and `vol_vv`, all
    produced by the same straight-line chain in `VacuumVessel.run()`.

    The name records the family's *original* single arm; since 2026-08-27 the shape is a
    family axis too and `VacuumVesselDShapedDoubleNull` is a member. Left as it is
    because `indat.py` and `vacuum.md` name it, and a rename would touch neither
    behaviour nor structure.
    """


class VacuumVesselEllipticalSingleNull(VacuumVesselElliptical):
    """cottax node: `.tokamak.vacuum_vessel` at `.divertor.n_divertors == 1` -- the
    combination live on `large_tokamak_eval.IN.DAT` (see module comment above). Thin
    wrap of `calculate_vacuum_vessel_outputs`, no arithmetic of its own.
    """

    dz_vv_half = OutputInto(blanket)
    vol_vv_inboard = OutputInto(blanket)
    vol_vv_outboard = OutputInto(blanket)
    vol_vv = OutputInto(fwbs)
    m_vv = OutputInto(fwbs)

    def __call__(
        self,
        z_tf_inside_half=From(build),
        dz_shld_vv_gap=From(build),
        dz_vv_lower=From(build),
        dz_blkt_upper=From(build),
        dz_shld_upper=From(build),
        z_plasma_xpoint_upper=From(build),
        dr_fw_plasma_gap_inboard=From(build),
        dr_fw_plasma_gap_outboard=From(build),
        dr_fw_inboard=From(build),
        dr_fw_outboard=From(build),
        rmajor=From(physics),
        rminor=From(physics),
        triang=From(physics),
        r_shld_inboard_inner=From(build),
        r_shld_outboard_outer=From(build),
        dr_vv_inboard=From(build),
        dr_vv_outboard=From(build),
        dz_vv_upper=From(build),
        fvoldw=From(fwbs),
        den_steel=From(fwbs),
    ):
        return calculate_vacuum_vessel_outputs(
            z_tf_inside_half=z_tf_inside_half,
            dz_shld_vv_gap=dz_shld_vv_gap,
            dz_vv_lower=dz_vv_lower,
            dz_blkt_upper=dz_blkt_upper,
            dz_shld_upper=dz_shld_upper,
            z_plasma_xpoint_upper=z_plasma_xpoint_upper,
            dr_fw_plasma_gap_inboard=dr_fw_plasma_gap_inboard,
            dr_fw_plasma_gap_outboard=dr_fw_plasma_gap_outboard,
            dr_fw_inboard=dr_fw_inboard,
            dr_fw_outboard=dr_fw_outboard,
            rmajor=rmajor,
            rminor=rminor,
            triang=triang,
            r_shld_inboard_inner=r_shld_inboard_inner,
            r_shld_outboard_outer=r_shld_outboard_outer,
            dr_vv_inboard=dr_vv_inboard,
            dr_vv_outboard=dr_vv_outboard,
            dz_vv_upper=dz_vv_upper,
            fvoldw=fvoldw,
            den_steel=den_steel,
        )


class VacuumVesselEllipticalDoubleNull(VacuumVesselElliptical):
    """cottax node: `.tokamak.vacuum_vessel` at `.divertor.n_divertors == 2` -- the
    value `spherical_tokamak_eval.IN.DAT` and `st_regression.IN.DAT` derive from
    `i_single_null = 0`. Thin wrap of
    `calculate_vacuum_vessel_outputs_double_null`.

    Owns the same five fields as its single-null sibling and reads seven fewer: the
    signature below has no `dz_blkt_upper`, `dz_shld_upper`, `z_plasma_xpoint_upper`,
    `dr_fw_plasma_gap_inboard`/`_outboard` or `dr_fw_inboard`/`_outboard`.
    """

    dz_vv_half = OutputInto(blanket)
    vol_vv_inboard = OutputInto(blanket)
    vol_vv_outboard = OutputInto(blanket)
    vol_vv = OutputInto(fwbs)
    m_vv = OutputInto(fwbs)

    def __call__(
        self,
        z_tf_inside_half=From(build),
        dz_shld_vv_gap=From(build),
        dz_vv_lower=From(build),
        rmajor=From(physics),
        rminor=From(physics),
        triang=From(physics),
        r_shld_inboard_inner=From(build),
        r_shld_outboard_outer=From(build),
        dr_vv_inboard=From(build),
        dr_vv_outboard=From(build),
        dz_vv_upper=From(build),
        fvoldw=From(fwbs),
        den_steel=From(fwbs),
    ):
        return calculate_vacuum_vessel_outputs_double_null(
            z_tf_inside_half=z_tf_inside_half,
            dz_shld_vv_gap=dz_shld_vv_gap,
            dz_vv_lower=dz_vv_lower,
            rmajor=rmajor,
            rminor=rminor,
            triang=triang,
            r_shld_inboard_inner=r_shld_inboard_inner,
            r_shld_outboard_outer=r_shld_outboard_outer,
            dr_vv_inboard=dr_vv_inboard,
            dr_vv_outboard=dr_vv_outboard,
            dz_vv_upper=dz_vv_upper,
            fvoldw=fvoldw,
            den_steel=den_steel,
        )


class VacuumVesselDShapedDoubleNull(VacuumVesselElliptical):
    """cottax node: `.tokamak.vacuum_vessel` at `.divertor.n_divertors == 2` **and** the
    D-shaped shape arm -- the configuration live on `spherical_tokamak_eval.IN.DAT` and
    `st_regression.IN.DAT` (`i_single_null = 0`; `itart = 1` and
    `i_fw_blkt_vv_shape = 1`, either of which alone selects the D-shaped arm). Thin wrap
    of `calculate_vacuum_vessel_outputs_dshaped_double_null`.

    Owns the same five fields as the other two occupants. Its signature has **no
    `From(physics)` port at all**: `rmajor`, `rminor` and `triang` are absent on top of
    the seven the double-null half-height already drops.
    """

    dz_vv_half = OutputInto(blanket)
    vol_vv_inboard = OutputInto(blanket)
    vol_vv_outboard = OutputInto(blanket)
    vol_vv = OutputInto(fwbs)
    m_vv = OutputInto(fwbs)

    def __call__(
        self,
        z_tf_inside_half=From(build),
        dz_shld_vv_gap=From(build),
        dz_vv_lower=From(build),
        r_shld_inboard_inner=From(build),
        r_shld_outboard_outer=From(build),
        dr_vv_inboard=From(build),
        dr_vv_outboard=From(build),
        dz_vv_upper=From(build),
        fvoldw=From(fwbs),
        den_steel=From(fwbs),
    ):
        return calculate_vacuum_vessel_outputs_dshaped_double_null(
            z_tf_inside_half=z_tf_inside_half,
            dz_shld_vv_gap=dz_shld_vv_gap,
            dz_vv_lower=dz_vv_lower,
            r_shld_inboard_inner=r_shld_inboard_inner,
            r_shld_outboard_outer=r_shld_outboard_outer,
            dr_vv_inboard=dr_vv_inboard,
            dr_vv_outboard=dr_vv_outboard,
            dz_vv_upper=dz_vv_upper,
            fvoldw=fvoldw,
            den_steel=den_steel,
        )
