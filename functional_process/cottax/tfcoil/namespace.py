"""The tokamak TF coil's namespaces -- what fills
`.tokamak.cicc_superconducting_tf_coil`, whichever turn the machine is wound with.
"""

import dataclasses

from cottax.interfaces.pytree_namespace_module import ModelNamespace

from functional_process.cottax.tfcoil.base import (
    DrTfPlasmaCaseFromFraction,
    DrTfPlasmaCaseFromInput,
    DxTfSideCaseMinFromFraction,
    GenericTfCoilAreaAndMasses,
    RBTfInboardPeak,
    TfCoilSelfInductance,
    TfCoilShape,
    TfCurrent,
    TfGlobalGeometry,
    TfStoredMagneticEnergy,
)
from functional_process.cottax.tfcoil.croco import (
    CrocoCableGeometry,
    CrocoCableSpaceProperties,
    CrocoInboardAreasAndFractions,
    CrocoSuperconductorProperties,
    CrocoTurnCableSpaceCoolingFraction,
    CrocoTurnCableSpaceExtraVoid,
    CrocoTurnGeometry,
)
from functional_process.cottax.tfcoil.quench import (
    TfCoilDumpQuenchVoltage,
    TfCoilQuenchHeatCurrentDensity,
)
from functional_process.cottax.tfcoil.stress import (
    TfFieldAndForce,
    TfStress,
)
from functional_process.cottax.tfcoil.superconducting import (
    CiccInboardAreasAndFractions,
    CiccSuperconductorProperties,
    CiccTurnGeometry,
    DxTfSideCase,
    PeakBTfInboardWithRipple,
    SuperconductingTfCoilAreasAndMasses,
    SuperconductingTfWpGeometry,
    TfCaseAreas,
    TfSuperconductorTemperatureMargin,
    TfTurnArea,
    TfWpCurrents,
    VvStressOnQuench,
)


class SuperconductingTfCoil(ModelNamespace):
    """What every superconducting TF coil has, whichever turn it is wound with."""

    # ---- base.py: the geometry every TF coil model shares --------------------------

    tf_global_geometry: TfGlobalGeometry = dataclasses.field(kw_only=True)
    """`.tfcoil.i_tf_case_geom` -- circular or straight front case, two occupants whose
    reads-sets are identical and which are separate classes anyway (`next_steps.md`
    §14.2).
    """

    dr_tf_plasma_case: DrTfPlasmaCaseFromInput | DrTfPlasmaCaseFromFraction = (
        dataclasses.field(kw_only=True)
    )
    """`.tfcoil.i_f_dr_tf_plasma_case` -- and the one slot in this port whose two arms
    are *different kinds of node*, which is why the annotation is a union and not a
    family base class: there is no class both arms could inherit from, because
    `FixedPointFunction` and `ExplicitFunction` are the two kinds.
    """

    dx_tf_side_case_min: DxTfSideCaseMinFromFraction | None = dataclasses.field(
        kw_only=True
    )
    """`.tfcoil.tfc_sidewall_is_fraction` -- **`None` on the reference run.** PROCESS
    computes `.tfcoil.dx_tf_side_case_min` only when the sidewall thickness is given as
    a fraction; at the default `False` (`tfcoil_variables.py:95`) it is an input and
    there is no arm at all.
    """

    r_b_tf_inboard_peak: RBTfInboardPeak = RBTfInboardPeak()
    """`.tfcoil.r_b_tf_inboard_peak`, the radius the peak inboard field is quoted at."""

    tf_current: TfCurrent = TfCurrent()
    """`.tfcoil.c_tf_total` and the peak symmetric field. Unswitched."""

    tf_coil_shape: TfCoilShape = dataclasses.field(kw_only=True)
    """`.tfcoil.i_tf_shape` x `.physics.itart` x `.physics.i_single_null` -- and the
    producer of `.tfcoil.len_tf_coil`, one of the two `VarPath`s this device shares with
    the stellarator from a completely different formula (`base.md` §"Shared with the
    stellarator").
    """

    tf_coil_self_inductance: TfCoilSelfInductance = dataclasses.field(kw_only=True)
    """`(.physics.itart, .tfcoil.i_tf_shape)` -- the D-shape arm integrates the arcs,
    the picture-frame arm is a closed form reading four entirely different fields.
    """

    tf_stored_magnetic_energy: TfStoredMagneticEnergy = TfStoredMagneticEnergy()
    """`.tfcoil.e_tf_coil_magnetic_stored` and its two totals."""

    generic_tf_coil_area_and_masses: GenericTfCoilAreaAndMasses = (
        GenericTfCoilAreaAndMasses()
    )
    """`.tfcoil.tfcryoarea`, `.tfcoil.tfocrn` and `.tfcoil.tficrn`."""

    # ---- superconducting.py: the winding pack, the turns and the masses -------------

    superconducting_tf_wp_geometry: SuperconductingTfWpGeometry = dataclasses.field(
        kw_only=True
    )
    """`.tfcoil.i_tf_wp_geom` -- rectangular, double-rectangular or trapezoidal."""

    tf_case_areas: TfCaseAreas = dataclasses.field(kw_only=True)
    """`.tfcoil.i_tf_case_geom`, the same switch `tf_global_geometry` answers -- one
    input value filling two slots, the shape `PhysicsConfinementTime.tail` already
    records for `i_rad_loss`.
    """

    dx_tf_side_case: DxTfSideCase = dataclasses.field(kw_only=True)
    """`.tfcoil.i_tf_wp_geom` again; the trapezoidal arm reads one field where the other
    two read three.
    """

    tf_wp_currents: TfWpCurrents = TfWpCurrents()
    """`.tfcoil.j_tf_wp`, and **not** a `FixedPointFunction` here."""

    peak_b_tf_inboard_with_ripple: PeakBTfInboardWithRipple = dataclasses.field(
        kw_only=True
    )
    """`round(.tfcoil.n_tf_coils)` -- treated as a switch, because the arms select
    different fit coefficients **and** own different numbers of outputs: the
    flat-allowance fallback returns before three of the four are assigned.
    """

    tf_turn_area: TfTurnArea = TfTurnArea()
    """`.tfcoil.a_tf_turn` -- one division written inline in `run` rather than in any
    function (`superconducting.py:2700-2704`), which is why it is a node of its own.
    """

    superconducting_tf_coil_areas_and_masses: SuperconductingTfCoilAreasAndMasses = (
        dataclasses.field(kw_only=True)
    )
    """`.physics.itart` -- the conventional arm owns ten fields, the spherical arm two
    more (`whtcp`, `whttflgs`).
    """

    # ---- stress.py: the vertical tension and the peak stresses ----------------------

    tf_field_and_force: TfFieldAndForce = dataclasses.field(kw_only=True)
    """`(.physics.itart, .tfcoil.i_cp_joints)` -- `.tfcoil.vforce` and its three
    siblings.
    """

    tf_stress: TfStress = dataclasses.field(kw_only=True)
    """`(.tfcoil.i_tf_stress_model, .tfcoil.i_tf_bucking, .tfcoil.i_tf_turns_integer)`
    -- constraint 31's `.tfcoil.sig_tf_case`, constraint 32's `.tfcoil.sig_tf_wp`, and
    the `.tfcoil.str_wp` the two slots below read.
    """

    # ---- the temperature margin and the two quench limits ---------------------------
    #
    # The critical-current slot itself is turn-type-specific and lives in each subclass;
    # the margin below is not, because both turn types call the *same* PROCESS function
    # (`calculate_superconductor_temperature_margin`) and it owns the same two fields
    # whichever material selects the residual.

    tf_superconductor_temperature_margin: TfSuperconductorTemperatureMargin = (
        dataclasses.field(kw_only=True)
    )
    """`.tfcoil.i_tf_sc_mat` x `.tfcoil.i_str_wp` -- constraint 36's
    `.tfcoil.temp_tf_superconductor_margin`, and the port's second internal solve
    (`scipy.optimize.newton`'s secant branch, replicated).
    """

    vv_stress_on_quench: VvStressOnQuench = VvStressOnQuench()
    """`.superconducting_tfcoil.vv_stress_quench`, constraint 65's read."""

    # ---- quench.py -----------------------------------------------------------------

    tf_coil_dump_quench_voltage: TfCoilDumpQuenchVoltage = TfCoilDumpQuenchVoltage()
    """`.tfcoil.v_tf_coil_dump_quench_kv`, three reads and no switch."""

    tf_coil_quench_heat_current_density: TfCoilQuenchHeatCurrentDensity = (
        dataclasses.field(kw_only=True)
    )
    """`.tfcoil.j_tf_wp_quench_heat_max`, constraint 35's read -- **the one CoolProp
    dependency in the whole tokamak scope, and now inside the graph rather than outside
    it.** Not switched: the field is `dataclasses.field(kw_only=True)` because the
    occupant carries four static values, not because a switch selects it.
    """


class CiccSuperconductingTfCoil(SuperconductingTfCoil):
    """A cable-in-conduit superconducting TF coil: the tokamak's magnet system."""

    cicc_turn_geometry: CiccTurnGeometry = dataclasses.field(kw_only=True)
    """`.tfcoil.i_tf_turns_integer` first, then (on the averaged arm)
    `.tfcoil.i_dx_tf_turn_general_input` together with
    `.tfcoil.i_dx_tf_turn_cable_space_general_input`.
    """

    cicc_inboard_areas_and_fractions: CiccInboardAreasAndFractions = (
        CiccInboardAreasAndFractions()
    )
    """The nine inboard areas and steel/insulation fractions. Unswitched."""

    cicc_superconductor_properties: CiccSuperconductorProperties = dataclasses.field(
        kw_only=True
    )
    """`.tfcoil.i_tf_sc_mat` x `.tfcoil.i_str_wp` -- the critical-current surface, and
    constraint 33's `.tfcoil.j_tf_wp_critical`.
    """


class CrocoSuperconductingTfCoil(SuperconductingTfCoil):
    """A CroCo (cross-conductor) REBCO-tape superconducting TF coil."""

    croco_turn_geometry: CrocoTurnGeometry = dataclasses.field(kw_only=True)
    """`.tfcoil.i_tf_turns_integer`, then the two turn-dimension input flags."""

    croco_cable_space_properties: CrocoCableSpaceProperties = CrocoCableSpaceProperties()
    """The cable space of one turn: seven circles of diameter `d` in a `3d x 3d` square,
    six of them CroCo strands and the seventh the central copper bar.
    """

    croco_cable_geometry: CrocoCableGeometry = CrocoCableGeometry()
    """One CroCo strand -- a copper tube around a soldered stack of REBCO tapes."""

    croco_turn_cable_space_extra_void: CrocoTurnCableSpaceExtraVoid = (
        CrocoTurnCableSpaceExtraVoid()
    )
    """`.tfcoil.f_a_tf_turn_cable_space_extra_void = 0.0`, one literal assignment in
    `run` (`superconducting.py:3894`).
    """

    croco_inboard_areas_and_fractions: CrocoInboardAreasAndFractions = (
        CrocoInboardAreasAndFractions()
    )
    """The same nine inboard areas and fractions the CICC slot owns, two of them from a
    different formula: no coolant channel, and a conductor area counted as strands
    rather than as cable space less voids.
    """

    croco_turn_cable_space_cooling_fraction: CrocoTurnCableSpaceCoolingFraction = (
        CrocoTurnCableSpaceCoolingFraction()
    )
    """`.superconducting_tfcoil.f_a_tf_turn_cable_space_cooling`, the one line of
    `run`'s inline copper block that any computation reads
    (`quench_heat_protection_current_ density` takes it).
    """

    croco_superconductor_properties: CrocoSuperconductorProperties = dataclasses.field(
        kw_only=True
    )
    """`.tfcoil.i_tf_sc_mat` x `.tfcoil.i_str_wp` -- the critical-current surface, and
    constraint 33's `.tfcoil.j_tf_wp_critical`.
    """
