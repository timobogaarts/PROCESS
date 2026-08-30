"""The tokamak TF coil's namespaces -- what fills
`.tokamak.cicc_superconducting_tf_coil`, whichever turn the machine is wound with.

**Three classes, and the middle one is the point.** `SuperconductingTfCoil` holds the
twenty-two slots every superconducting TF coil has; `CiccSuperconductingTfCoil` and
`CrocoSuperconductingTfCoil` add the three and six that their turn actually differs in.
That mirrors PROCESS, where `CROCOSuperconductingTFCoil` *subclasses*
`CICCSuperconductingTFCoil`'s own parent and both `run`s open with the same
`run_base_superconducting_tf` call; and it is what `caller.py:298-313` decides, above
every model, from `.superconducting_tfcoil.i_tf_turn_type`.

Beside the nodes they name (`model_tree_design.md` §11), and spanning four modules
because PROCESS's own classes do: `base.py` is reached through them by *inheritance*
rather than by any call in `caller.py` (`tokamak_call_surface.md` §A row 3), `quench.py`
through one call inside them, `croco.py` only from the second. That is why this is a
`namespace.py` in the package rather than a class at the foot of one module -- there is
no single module the slots are all beside.

**Every switch is answered by `indat.py`**, never here. The families and the switch each
answers are in the class docstrings below; the reference-run arm of every cable-in-
conduit one is `tests/regression/input_files/large_tokamak_eval.IN.DAT`'s and of every
CroCo one is `spherical_tokamak_eval.IN.DAT`'s, and `tfcoil/base.md`,
`tfcoil/superconducting.md`, `tfcoil/croco.md` and `tfcoil/quench.md` carry the per-arm
evidence.

**The slot's name still says `cicc`, and that is now one turn type's name on a place
both fill.** Renaming it would move every node under it -- `.tokamak.cicc_
superconducting_tf_coil.tf_stress` and its twenty-odd siblings -- across the boundary
pins, the DSM exports and four test modules, for no change in structure. Left as it is
deliberately, recorded here so the mismatch is a known one; `croco.md` §open questions
carries it.

**One slot is legitimately empty on the reference run.** `dx_tf_side_case_min` has a
node only when `.tfcoil.tfc_sidewall_is_fraction` is `True`; PROCESS's default is
`False` (`tfcoil_variables.py:95`), on which `.tfcoil.dx_tf_side_case_min` is simply an
input and no code computes it. `None` is the honest occupant, the same spelling
`costs.cost_of_electricity` and `power.cryo_q_nuc` already use -- an unowned read is a
correct answer for that field.

**`.tfcoil.c_tf_turn` is a boundary input on the reference configuration, by
measurement.** `superconducting.md` finding 1: at `i_dx_tf_turn_general_input == False`
and `i_dx_tf_turn_cable_space_general_input == False` nothing under `process/models/`
produces it -- it is iteration variable 60 and enters from the input file. So this slot
produces **nine** of the ten variables `_audit/tokamak_boundary.md` lists against it,
and the tenth is an unknown rather than a missing node. On `i_tf_turns_integer == 1`
(`low_aspect_ratio_DEMO`) the ownership flips: `CiccIntegerTurnGeometry` *produces*
`c_tf_turn` from the coil current and the fixed turn count, and the field is not a
boundary input of that machine.
"""

import dataclasses

from cottax.interfaces.pytree_namespace_module import ModelNamespace

from functional_process.models.tfcoil.base import (
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
from functional_process.models.tfcoil.croco import (
    CrocoCableGeometry,
    CrocoCableSpaceProperties,
    CrocoInboardAreasAndFractions,
    CrocoSuperconductorProperties,
    CrocoTurnCableSpaceCoolingFraction,
    CrocoTurnCableSpaceExtraVoid,
    CrocoTurnGeometry,
)
from functional_process.models.tfcoil.quench import (
    TfCoilDumpQuenchVoltage,
    TfCoilQuenchHeatCurrentDensity,
)
from functional_process.models.tfcoil.stress import (
    TfFieldAndForce,
    TfStress,
)
from functional_process.models.tfcoil.superconducting import (
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
    """What every superconducting TF coil has, whichever turn it is wound with.

    The twenty-two slots `CICCSuperconductingTFCoil` and `CROCOSuperconductingTFCoil`
    share -- which is to say `run_base_superconducting_tf` and everything it reaches
    (`process/models/tfcoil/superconducting.py:161-285`), plus the areas, masses,
    stresses and quench limits both `run`s call unchanged afterwards. The two subclasses
    below add only what their turn actually differs in.

    **A base class, and not a fourteenth copy of the same fourteen docstrings.** The two
    `Model` classes in PROCESS are related by inheritance and the port says so the same
    way; before this class existed the CroCo namespace would have had to restate every
    shared slot's argument, and two statements of one argument are two things that can
    drift.

    Written in dependency order rather than in file order, because the chain is long and
    legible: global geometry and case thickness, then the current, then the winding
    pack, then the turns, then the areas and masses, then the two quantities the quench
    and energy limits read. Each subclass's own slots keep their place in that order by
    being written where they belong in its body.
    """

    # ---- base.py: the geometry every TF coil model shares --------------------------

    tf_global_geometry: TfGlobalGeometry = dataclasses.field(kw_only=True)
    """`.tfcoil.i_tf_case_geom` -- circular or straight front case, two occupants whose
    reads-sets are identical and which are separate classes anyway (`next_steps.md`
    §14.2). Owns nine unswitched outputs including `.tfcoil.a_tf_inboard_total`, which
    everything below depends on."""

    dr_tf_plasma_case: DrTfPlasmaCaseFromInput | DrTfPlasmaCaseFromFraction = (
        dataclasses.field(kw_only=True)
    )
    """`.tfcoil.i_f_dr_tf_plasma_case` -- and the one slot in this port whose two arms
    are *different kinds of node*, which is why the annotation is a union and not a
    family base class: there is no class both arms could inherit from, because
    `FixedPointFunction` and `ExplicitFunction` are the two kinds. The `False` arm is a
    `FixedPointFunction` because PROCESS clamps `dr_tf_plasma_case` in place; the `True`
    arm computes it from a fraction and never reads the entering value, so it is an
    `ExplicitFunction`. The
    loop is a property of the arm, not of the quantity (`base.md` OQ1)."""

    dx_tf_side_case_min: DxTfSideCaseMinFromFraction | None = dataclasses.field(
        kw_only=True
    )
    """`.tfcoil.tfc_sidewall_is_fraction` -- **`None` on the reference run.**

    PROCESS computes `.tfcoil.dx_tf_side_case_min` only when the sidewall thickness is
    given as a fraction; at the default `False` (`tfcoil_variables.py:95`) it is an
    input and there is no arm at all. Absence, spelled as absence -- not a refusal,
    because PROCESS itself computes nothing here."""

    r_b_tf_inboard_peak: RBTfInboardPeak = RBTfInboardPeak()
    """`.tfcoil.r_b_tf_inboard_peak`, the radius the peak inboard field is quoted at.
    Unswitched."""

    tf_current: TfCurrent = TfCurrent()
    """`.tfcoil.c_tf_total` and the peak symmetric field. Unswitched."""

    tf_coil_shape: TfCoilShape = dataclasses.field(kw_only=True)
    """`.tfcoil.i_tf_shape` x `.physics.itart` x `.physics.i_single_null` -- and the
    producer of `.tfcoil.len_tf_coil`, one of the two `VarPath`s this device shares with
    the stellarator from a completely different formula (`base.md` §"Shared with the
    stellarator"). Two of the arms are written, both D-shaped and conventional-aspect."""

    tf_coil_self_inductance: TfCoilSelfInductance = dataclasses.field(kw_only=True)
    """`(.physics.itart, .tfcoil.i_tf_shape)` -- the D-shape arm integrates the arcs,
    the picture-frame arm is a closed form reading four entirely different fields."""

    tf_stored_magnetic_energy: TfStoredMagneticEnergy = TfStoredMagneticEnergy()
    """`.tfcoil.e_tf_coil_magnetic_stored` and its two totals. Unswitched, and read by
    `tf_coil_dump_quench_voltage` below."""

    generic_tf_coil_area_and_masses: GenericTfCoilAreaAndMasses = (
        GenericTfCoilAreaAndMasses()
    )
    """`.tfcoil.tfcryoarea`, `.tfcoil.tfocrn` and `.tfcoil.tficrn`. Unswitched.

    `.tfcoil.tfcryoarea` is the second of the two `VarPath`s this device shares with the
    stellarator from an entirely different formula (`.tfcoil.len_tf_coil` is the other,
    from `tf_coil_shape` above): a stellarator's is a modular-coil scaling, a tokamak's
    is this coil's surface area. One variable, two device-specific producers, two
    slots -- which is what a slot is for, and never both in one graph."""

    # ---- superconducting.py: the winding pack, the turns and the masses -------------

    superconducting_tf_wp_geometry: SuperconductingTfWpGeometry = dataclasses.field(
        kw_only=True
    )
    """`.tfcoil.i_tf_wp_geom` -- rectangular, double-rectangular or trapezoidal. All
    three arms read the same seven fields; one class per value regardless, per the
    binding policy. Note the value is **not** read from the input file as written:
    `init.py:977-989` resolves the `-1` `UNSET` default from `i_tf_turns_integer`, and
    `machine_from_indat` reproduces that resolution rather than the raw default."""

    tf_case_areas: TfCaseAreas = dataclasses.field(kw_only=True)
    """`.tfcoil.i_tf_case_geom`, the same switch `tf_global_geometry` answers -- one
    input value filling two slots, the shape `PhysicsConfinementTime.tail` already
    records for `i_rad_loss`."""

    dx_tf_side_case: DxTfSideCase = dataclasses.field(kw_only=True)
    """`.tfcoil.i_tf_wp_geom` again; the trapezoidal arm reads one field where the other
    two read three."""

    tf_wp_currents: TfWpCurrents = TfWpCurrents()
    """`.tfcoil.j_tf_wp`, and **not** a `FixedPointFunction` here.

    `models/stellarator/namespace.py`'s counterpart carries a long argument about
    whether this field needs one. On the tokamak path it does not: PROCESS's body
    (`superconducting.py:1963-1970`) never consults the entering value, so there is no
    self-reference to cut."""

    peak_b_tf_inboard_with_ripple: PeakBTfInboardWithRipple = dataclasses.field(
        kw_only=True
    )
    """`round(.tfcoil.n_tf_coils)` -- treated as a switch, because the arms select
    different fit coefficients **and** own different numbers of outputs: the
    flat-allowance fallback returns before three of the four are assigned. A build-time
    branch is legitimate here only because `n_tf_coils` is not an iteration variable
    (`superconducting.md` OQ2)."""

    tf_turn_area: TfTurnArea = TfTurnArea()
    """`.tfcoil.a_tf_turn` -- one division written inline in `run` rather than in any
    function (`superconducting.py:2700-2704`), which is why it is a node of its own."""

    superconducting_tf_coil_areas_and_masses: SuperconductingTfCoilAreasAndMasses = (
        dataclasses.field(kw_only=True)
    )
    """`.physics.itart` -- the conventional arm owns ten fields, the spherical arm two
    more (`whtcp`, `whttflgs`). Conditional ownership, so occupants and not a kwarg."""

    # ---- stress.py: the vertical tension and the peak stresses ----------------------

    tf_field_and_force: TfFieldAndForce = dataclasses.field(kw_only=True)
    """`(.physics.itart, .tfcoil.i_cp_joints)` -- `.tfcoil.vforce` and its three
    siblings. Only the clamped-joint arm is written; the sliding-joint one additionally
    **owns** `.tfcoil.f_vforce_inboard`, which this arm reads and returns unchanged, so
    the two arms are separate classes rather than one node with a kwarg."""

    tf_stress: TfStress = dataclasses.field(kw_only=True)
    """`(.tfcoil.i_tf_stress_model, .tfcoil.i_tf_bucking, .tfcoil.i_tf_turns_integer)`
    -- constraint 31's `.tfcoil.sig_tf_case`, constraint 32's `.tfcoil.sig_tf_wp`, and
    the `.tfcoil.str_wp` the two slots below read.

    **The slot whose absence dropped two active constraints.** Before it existed the
    port had no producer for either stress and constraint 31/32 evaluated `0 <= max` on
    `large_tokamak_nof`, which activates both -- not a wrong number but a silently
    satisfied condition (`stress.md`, 2026-08-30). Two of the three switches have one
    written arm each and the reasons are in `stress.py`'s module docstring; the third,
    `i_tf_turns_integer`, has both, because it decides which cable-space field the
    transverse smearing reads."""

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
    (`scipy.optimize.newton`'s secant branch, replicated). Its registry is **not** the
    critical-current slot's with the names changed: on the cable-in-conduit side two of
    the five ported materials are refused here for reasons specific to *this* function,
    and on the CroCo side the same value `9` selects a residual with three tape
    dimensions and no strain. See `CiccSuperconductorProperties` and
    `HazeltonZhaiRebcoCrocoTemperatureMargin`."""

    vv_stress_on_quench: VvStressOnQuench = VvStressOnQuench()
    """`.superconducting_tfcoil.vv_stress_quench`, constraint 65's read. Unswitched, and
    by some distance the widest-reading node in this slot -- seventeen `.build` fields,
    because the Itoh surrogate needs the vacuum vessel's own current centre line as well
    as the coil's."""

    # ---- quench.py -----------------------------------------------------------------

    tf_coil_dump_quench_voltage: TfCoilDumpQuenchVoltage = TfCoilDumpQuenchVoltage()
    """`.tfcoil.v_tf_coil_dump_quench_kv`, three reads and no switch."""

    tf_coil_quench_heat_current_density: TfCoilQuenchHeatCurrentDensity = (
        dataclasses.field(kw_only=True)
    )
    """`.tfcoil.j_tf_wp_quench_heat_max`, constraint 35's read -- **the one CoolProp
    dependency in the whole tokamak scope, and now inside the graph rather than outside
    it.**

    Not switched: the field is `dataclasses.field(kw_only=True)` because the occupant
    carries four static values, not because a switch selects it. `indat.py` fills them
    from the machine's own `tftmp`/`temp_tf_conductor_quench_max` with one CoolProp
    round-trip at assembly time, and refuses the machine outright if either is an
    iteration variable. The reasoning is in `TfCoilQuenchHeatCurrentDensity`'s
    docstring; it resolves `quench.md` OQ1 in favour of its option (a)."""


class CiccSuperconductingTfCoil(SuperconductingTfCoil):
    """A cable-in-conduit superconducting TF coil: the tokamak's magnet system.

    Named for `process/models/tfcoil/superconducting.py::CICCSuperconductingTFCoil`,
    which is what `caller.py:306` runs at `i_tf_sup == 1` and
    `.superconducting_tfcoil.i_tf_turn_type == 1`. Both of those switches are resolved
    **above every model**, in `caller.py:295-316`, which is why neither appears as a slot
    here: a machine whose `i_tf_turn_type` is `2` fills the same
    `.tokamak.cicc_superconducting_tf_coil` slot with `CrocoSuperconductingTfCoil`
    below, and there is no slot inside this one that decides it.

    Three slots of its own on top of the twenty-two above -- the turn, the inboard areas
    and the critical-current chain, which is exactly what PROCESS's two `run`s differ in
    on the cable side.
    """

    cicc_turn_geometry: CiccTurnGeometry = dataclasses.field(kw_only=True)
    """`.tfcoil.i_tf_turns_integer` first, then (on the averaged arm)
    `.tfcoil.i_dx_tf_turn_general_input` together with
    `.tfcoil.i_dx_tf_turn_cable_space_general_input`. Four arms, two written: the
    reference averaged arm (both booleans `False`), which reads `.tfcoil.c_tf_turn`
    and does not own it -- that field is this slot's one boundary input on that
    configuration -- and the integer arm (`i_tf_turns_integer == 1`,
    `low_aspect_ratio_DEMO`'s), which **owns** `c_tf_turn` because the turn count is
    fixed by `n_tf_wp_layers * n_tf_wp_pancakes`. Conditional ownership across arms of
    one slot, the same shape `models/power/thermal_cryo.py` records."""

    cicc_inboard_areas_and_fractions: CiccInboardAreasAndFractions = (
        CiccInboardAreasAndFractions()
    )
    """The nine inboard areas and steel/insulation fractions. Unswitched."""

    cicc_superconductor_properties: CiccSuperconductorProperties = dataclasses.field(
        kw_only=True
    )
    """`.tfcoil.i_tf_sc_mat` x `.tfcoil.i_str_wp` -- the critical-current surface, and
    constraint 33's `.tfcoil.j_tf_wp_critical`. Five of the nine materials are written
    and four are refused with a measured reason each; the strain switch has only its
    default arm. See the class docstring."""


class CrocoSuperconductingTfCoil(SuperconductingTfCoil):
    """A CroCo (cross-conductor) REBCO-tape superconducting TF coil.

    Named for `process/models/tfcoil/superconducting.py::CROCOSuperconductingTFCoil`,
    which `caller.py:307-313` runs at `i_tf_sup == 1` and
    `.superconducting_tfcoil.i_tf_turn_type == 2` -- **both tracked spherical tokamaks**
    (`spherical_tokamak_eval.IN.DAT:72`, `st_regression.IN.DAT:800`). It fills the same
    `.tokamak.cicc_superconducting_tf_coil` slot as its sibling, because PROCESS
    resolves the switch above every model and the *place* in the machine is the same
    magnet system either way.

    Six slots of its own on top of the twenty-two shared, three more than the
    cable-in-conduit sibling -- and the three extra are the reason the two are different
    `Model` classes rather than one with a branch: a CroCo turn's conductor is a cable
    of six soldered REBCO-tape strands, so there is a strand geometry
    (`croco_cable_geometry`), a cable space wrapped around it
    (`croco_cable_space_properties`) and a void fraction that is identically zero
    (`croco_turn_cable_space_extra_void`), none of which a cable-in-conduit turn has any
    counterpart for.

    Written in the same dependency order the base class is; `models/tfcoil/croco.py`'s
    module docstring carries the per-node evidence and, in particular, the five writes
    of `run` this namespace deliberately has no slot for.
    """

    croco_turn_geometry: CrocoTurnGeometry = dataclasses.field(kw_only=True)
    """`.tfcoil.i_tf_turns_integer`, then the two turn-dimension input flags. **One arm
    exists in PROCESS at all**: `run` raises `ProcessValueError` on integer turns
    (`superconducting.py:3834-3840`), so the refusal in `indat.UNPORTED` quotes PROCESS
    rather than the port.

    Reads `.tfcoil.c_tf_turn` and does not own it, exactly as the cable-in-conduit
    averaged arm does -- and owns **neither** `.tfcoil.a_tf_turn_cable_space_no_void`
    nor `.tfcoil.a_tf_turn_steel`, which `croco_cable_space_properties` overwrites three
    statements later before anything reads them."""

    croco_cable_space_properties: CrocoCableSpaceProperties = CrocoCableSpaceProperties()
    """The cable space of one turn: seven circles of diameter `d` in a `3d x 3d` square,
    six of them CroCo strands and the seventh the central copper bar. Unswitched, and
    the owner of the two fields the slot above hands it."""

    croco_cable_geometry: CrocoCableGeometry = CrocoCableGeometry()
    """One CroCo strand -- a copper tube around a soldered stack of REBCO tapes. Ten
    outputs, all in `.superconducting_tfcoil`, and the only node in the tokamak whose
    every read is a tape or tube thickness the input file sets directly. Unswitched."""

    croco_turn_cable_space_extra_void: CrocoTurnCableSpaceExtraVoid = (
        CrocoTurnCableSpaceExtraVoid()
    )
    """`.tfcoil.f_a_tf_turn_cable_space_extra_void = 0.0`, one literal assignment in
    `run` (`superconducting.py:3894`). A node with no reads, and the slot that keeps a
    field PROCESS *computes* on this device from re-entering the graph as a boundary
    input -- the missing-producer class of `_audit/optimise_design.md` §16. It has no
    counterpart in the cable-in-conduit namespace because there the field really is an
    input."""

    croco_inboard_areas_and_fractions: CrocoInboardAreasAndFractions = (
        CrocoInboardAreasAndFractions()
    )
    """The same nine inboard areas and fractions the CICC slot owns, two of them from a
    different formula: no coolant channel, and a conductor area counted as strands
    rather than as cable space less voids. Unswitched."""

    croco_turn_cable_space_cooling_fraction: CrocoTurnCableSpaceCoolingFraction = (
        CrocoTurnCableSpaceCoolingFraction()
    )
    """`.superconducting_tfcoil.f_a_tf_turn_cable_space_cooling`, the one line of `run`'s
    inline copper block that any computation reads (`quench_heat_protection_current_
    density` takes it). Unswitched, and a slot of its own for the same reason
    `tf_turn_area` is one: it is a statement in `run`, not a function."""

    croco_superconductor_properties: CrocoSuperconductorProperties = dataclasses.field(
        kw_only=True
    )
    """`.tfcoil.i_tf_sc_mat` x `.tfcoil.i_str_wp` -- the critical-current surface, and
    constraint 33's `.tfcoil.j_tf_wp_critical`. The switch has **three** reachable values
    here rather than nine: the function refuses any non-`TAPE` shape in its first four
    lines (`superconducting.py:4435-4441`), leaving `6`, `8` and `9` -- the exact
    complement of the cable-in-conduit slot's five. One is written, `9`, which is what
    both tracked ST files set."""
