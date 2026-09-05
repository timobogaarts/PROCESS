"""Pure-functional port of `process/models/tfcoil/superconducting.py` --
`CROCOSuperconductingTFCoil`, the cross-conductor (CroCo) REBCO-tape TF coil.

Audit record: `functional_process/_audit/units/models/tfcoil/croco.md`.

**A sibling of `superconducting.py`, not a layer on top of it.** PROCESS resolves
`.superconducting_tfcoil.i_tf_turn_type` in `core/caller.py:298-313`, *above every
model*, and runs `CROCOSuperconductingTFCoil` (`:3773-4865`) instead of
`CICCSuperconductingTFCoil` at value `2`. The two classes share their whole base --
`run_base_superconducting_tf` and everything under it, already ported in `base.py` and
`superconducting.py` -- and differ only in the winding-pack turn, the cable, the
inboard areas and the critical-current chain. This module is exactly that difference:
seven pure functions and eight nodes, gathered into
`namespace.CrocoSuperconductingTfCoil`, which fills the same
`.tokamak.cicc_superconducting_tf_coil` slot the cable-in-conduit namespace does.

Both tracked spherical tokamaks are CroCo machines --
`tests/regression/input_files/spherical_tokamak_eval.IN.DAT:72` and
`st_regression.IN.DAT:800` set `i_tf_turn_type = 2` -- with `i_tf_sc_mat = 9`
(`HAZELTON_ZHAI_REBCO`), `i_tf_turns_integer = 0` and `i_tf_wp_geom = 2`.

## What is *not* ported, and the measurement for each

**Five of `run`'s writes are dead**, in the strict sense that a later statement in the
same `run` overwrites them before anything reads them. They are not ported, and each
absence is a measurement rather than a judgement:

| write | `superconducting.py` | overwritten by | reader in between |
|---|---|---|---|
| `a_tf_turn_cable_space_no_void` from the turn geometry | `:3813` (returned unchanged from `data`, `:4379`) | `tf_turn_croco_cable_space_properties`, `:3849` | none |
| `a_tf_turn_steel` from the turn geometry | `:3816` | the same, `:3855` | none |
| `f_a_tf_turn_cable_space_cooling` from the cable space | `:3856` | the inline block, `:3948` | none |
| `a_tf_wp_conductor` from the inboard areas | `:3912` | the inline block, `:3938` -- **with the identical expression** | none |
| `v_tf_coil_dump_quench_kv` from `croco_voltage()` | `:4020-4021` | `quench_heat_protection_current_density`'s second return, `:4258-4259` | none |

The first two matter most: `tf_croco_averaged_turn_geometry` computes its
`a_tf_turn_steel` from `self.data.tfcoil.a_tf_turn_cable_space_no_void` as it stands on
entry (`:4375`), i.e. from the *previous* pipeline pass -- a genuine implicit-io read of
a stale value. Because the cable-space node recomputes both fields from scratch three
statements later, the port owns neither at the turn-geometry node and the stale read
disappears with them. That is the honest resolution: there is no ordering to reproduce
because there is no live value.

`croco_voltage` (`:4677-4706`) is therefore **not ported at all**. Its return feeds only
the overwritten `v_tf_coil_dump_quench_kv`, and its two side-effect writes
(`.superconducting_tfcoil.time2`/`tau2`) are read nowhere outside its own body --
verified by grep over `process/`. `.tfcoil.quench_model` is a *string* switch
(`core/input.py:1102`, choices `"linear"`/`"exponential"`) with no default in either
tracked file, on which the function returns `0.0`; none of that reaches an output.

**The inline copper block (`:3930-3959`) is output-only except one line.**
`a_tf_turn_croco_copper_bar`, `a_tf_turn_croco_cable_space_copper`,
`a_tf_turn_copper_total`, `f_a_tf_turn_copper` and `a_tf_turn_croco_hastelloy` are read
only by `output_croco_info` (`:4843-4858`) -- again by grep over `process/`. Only
`f_a_tf_turn_cable_space_cooling` (`:3948-3955`) survives into a computation, being read
by `quench_heat_protection_current_density`, so that one line is a node
(`CrocoTurnCableSpaceCoolingFraction`) and the rest is dropped as reporting.

Dropping `f_a_tf_turn_copper` also removes the module's one true ordering hazard:
PROCESS divides by `self.data.tfcoil.a_tf_turn` at `:3944-3946`, **before** `run`
recomputes that field at `:3961-3965`, so it uses the previous pass's turn area. A node reading
`.tfcoil.a_tf_turn` would get the current one and disagree with PROCESS by construction.
Recorded as defect **D1** in the audit record.

**`tf_croco_superconductor_properties`' temperature-margin tail is dead too**
(`:4540-4547`): it calls `superconductors.current_sharing_rebco` -- a second
`scipy.optimize.newton` solve -- and writes `.tfcoil.temp_margin`, which
`calculate_superconductor_temperature_margin` overwrites at `:1278` on every arm this
namespace can reach. So the port does not need `current_sharing_rebco` and does not
have it; `.tfcoil.temp_margin` is owned by the margin node alone, exactly as on the
cable-in-conduit side.

**Integer turns are refused by PROCESS itself** (`:3838-3840`,
`ProcessValueError("Integer turn geometry not implemented for CroCo conductor.")`), so
`CROCO_TURN_GEOMETRY` has one arm and `indat.UNPORTED` carries that sentence rather than
a port's excuse. Both tracked files set `i_tf_turns_integer = 0`.

## Switch splits in this file

| PROCESS function | switch | occupants written | refused |
|---|---|---|---|
| `run`'s turn-geometry choice | `i_tf_turns_integer` | `0` (averaged) | `1` -- PROCESS raises |
| `tf_croco_averaged_turn_geometry` | `i_dx_tf_turn_general_input`, `i_dx_tf_turn_cable_space_general_input` | the both-`False` arm | the other two |
| `tf_croco_superconductor_properties` | `i_tf_sc_mat` x `i_str_wp` | `(1, 9)` | `(*, 6)`, `(*, 8)`, `(0, 9)` |
| `calculate_superconductor_temperature_margin` | `i_tf_sc_mat` x `i_str_wp` | `(1, 9)` | the same three |

The `i_tf_sc_mat` axis has only three values here at all: the function's own first guard
(`:4435-4441`) refuses any `SuperconductorShape` but `TAPE`, which leaves `6`
(`CROCO_REBCO`), `8` (`DURHAM_REBCO`) and `9` (`HAZELTON_ZHAI_REBCO`) -- the exact
complement of the cable-in-conduit slot's five. Value `6` is refused with a measured
reason of its own: its properties arm runs, but
`calculate_superconductor_temperature_margin` handles `{1, 3, 4, 5, 7, 8, 9}` and
nothing else, so PROCESS raises `ProcessValueError` one call later (`:1290-1292`).

Neither arm reads a strain, and that is not an omission. `run` chooses one at
`:4001-4004` and `tf_croco_superconductor_properties` chooses one again at `:4443-4446`,
but the value reaches nothing on the ported arm: `hijc_rebco` takes no strain argument,
and the `abs(strain) > 0.7e-2` clip at `:4486-4492` therefore clips a number that is
never used. `i_str_wp` is still a key of both registries, because it is the switch that
decides *which field* a strain would be read from and arm `8` does use one.
"""

from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    OutputInto,
)

from functional_process.models.stated import StatesValues
from functional_process.models.tfcoil.superconducting import (
    TfSuperconductorTemperatureMargin,
)
from functional_process.paths import superconducting_tfcoil, tfcoil
from functional_process.tfcoil.croco import (
    calculate_hazelton_zhai_rebco_croco_temperature_margin,
    croco_averaged_turn_geometry_from_current_per_turn,
    croco_cable_geometry,
    croco_cable_space_properties,
    croco_inboard_areas_and_fractions,
    croco_superconductor_properties_hijc_rebco,
    croco_turn_cable_space_cooling_fraction,
    croco_turn_cable_space_extra_void,  # noqa: F401 -- re-exported for indat.py / tests
    temperature_margin_hijc_rebco,  # noqa: F401 -- re-exported for tests/.../test_croco.py
)

# ---------------------------------------------------------------------------
# cottax nodes
# ---------------------------------------------------------------------------


class CrocoTurnGeometry(ExplicitFunction):
    """The family that owns the CroCo winding-pack turn geometry.

    One occupant, because PROCESS has one: `run` raises outright on
    `i_tf_turns_integer == 1` (`superconducting.py:3834-3840`). The family class exists
    anyway, so the slot's annotation names a family rather than a single class and the
    integer arm has somewhere to land if PROCESS ever writes it.
    """


class CrocoAveragedTurnGeometryFromCurrentPerTurn(CrocoTurnGeometry):
    """Both turn-dimension input flags `False` -- PROCESS's default and both ST files'.

    **Reads `.tfcoil.c_tf_turn`; does not own it**, and owns neither
    `.tfcoil.a_tf_turn_cable_space_no_void` nor `.tfcoil.a_tf_turn_steel`, which
    `CrocoCableSpaceProperties` overwrites before any reader. See the module docstring.
    """

    a_tf_turn_insulation = OutputInto(tfcoil)
    n_tf_coil_turns = OutputInto(tfcoil)
    dx_tf_turn_general = OutputInto(tfcoil)
    dr_tf_turn = OutputInto(superconducting_tfcoil)
    dx_tf_turn = OutputInto(superconducting_tfcoil)
    dx_tf_turn_conduit_full_average = OutputInto(tfcoil)
    dx_tf_turn_cable_space_average = OutputInto(superconducting_tfcoil)

    def __call__(
        self,
        j_tf_wp=From(tfcoil),
        c_tf_turn=From(tfcoil),
        dx_tf_turn_steel=From(tfcoil),
        dx_tf_turn_insulation=From(tfcoil),
        layer_ins=From(tfcoil),
        a_tf_wp_no_insulation=From(superconducting_tfcoil),
    ):
        return croco_averaged_turn_geometry_from_current_per_turn(
            j_tf_wp=j_tf_wp,
            c_tf_turn=c_tf_turn,
            dx_tf_turn_steel=dx_tf_turn_steel,
            dx_tf_turn_insulation=dx_tf_turn_insulation,
            layer_ins=layer_ins,
            a_tf_wp_no_insulation=a_tf_wp_no_insulation,
        )


class CrocoCableSpaceProperties(ExplicitFunction):
    """cottax node: `tf_turn_croco_cable_space_properties`. No switch.

    Owns the two fields the turn-geometry node above deliberately does not, which is
    what makes that node's stale `a_tf_turn_cable_space_no_void` read disappear rather
    than be reproduced.
    """

    dia_tf_turn_croco_cable = OutputInto(superconducting_tfcoil)
    a_tf_turn_cable_space_no_void = OutputInto(tfcoil)
    a_tf_turn_cable_space_effective = OutputInto(superconducting_tfcoil)
    a_tf_turn_steel = OutputInto(tfcoil)

    def __call__(
        self,
        dx_tf_turn_conduit_full_average=From(tfcoil),
        dx_tf_turn_steel=From(tfcoil),
    ):
        return croco_cable_space_properties(
            dx_tf_turn_conduit_full_average=dx_tf_turn_conduit_full_average,
            dx_tf_turn_steel=dx_tf_turn_steel,
        )


class CrocoCableGeometry(ExplicitFunction):
    """cottax node: `superconductors.calculate_croco_cable_geometry`. No switch.

    The only node in this port whose reads are **all** in
    `.superconducting_tfcoil` -- the five tape and copper-tube thicknesses, every one of
    them a genuine input that both ST files set explicitly
    (`spherical_tokamak_eval.IN.DAT:73-76`).
    """

    dia_tf_croco_strand_tape_region = OutputInto(superconducting_tfcoil)
    n_tf_croco_strand_hts_tapes = OutputInto(superconducting_tfcoil)
    a_tf_croco_strand_copper_total = OutputInto(superconducting_tfcoil)
    a_tf_croco_strand_hastelloy = OutputInto(superconducting_tfcoil)
    a_tf_croco_strand_solder = OutputInto(superconducting_tfcoil)
    a_tf_croco_strand_rebco = OutputInto(superconducting_tfcoil)
    a_tf_croco_strand = OutputInto(superconducting_tfcoil)
    dr_tf_hts_tape = OutputInto(superconducting_tfcoil)
    dx_tf_hts_tape_total = OutputInto(superconducting_tfcoil)
    dx_tf_croco_strand_tape_stack = OutputInto(superconducting_tfcoil)

    def __call__(
        self,
        dia_tf_turn_croco_cable=From(superconducting_tfcoil),
        dx_tf_croco_strand_copper=From(superconducting_tfcoil),
        dx_tf_hts_tape_rebco=From(superconducting_tfcoil),
        dx_tf_hts_tape_copper=From(superconducting_tfcoil),
        dx_tf_hts_tape_hastelloy=From(superconducting_tfcoil),
    ):
        return croco_cable_geometry(
            dia_croco_strand=dia_tf_turn_croco_cable,
            dx_croco_strand_copper=dx_tf_croco_strand_copper,
            dx_hts_tape_rebco=dx_tf_hts_tape_rebco,
            dx_hts_tape_copper=dx_tf_hts_tape_copper,
            dx_hts_tape_hastelloy=dx_tf_hts_tape_hastelloy,
        )


class CrocoTurnCableSpaceExtraVoid(StatesValues):
    """cottax node: `run`'s literal `f_a_tf_turn_cable_space_extra_void = 0.0`
    (`superconducting.py:3894`). Computes nothing -- it states its one output.

    **Conditional ownership across two `Model` classes.** On the cable-in-conduit path
    the same `VarPath` is a plain input that the run file sets; here PROCESS overwrites
    it unconditionally, so the field has a producer on a CroCo machine and none on a
    cable-in-conduit one. Both tracked ST files leave the input unset, so nothing would
    disagree numerically today -- but a graph that read it would be reading a coincidence
    (`_audit/optimise_design.md` §16, the missing-producer class).
    """

    f_a_tf_turn_cable_space_extra_void = OutputInto(tfcoil)
    """The ported literal, *stated* at `^stated.tfcoil.f_a_tf_turn_cable_space_extra_void`
    rather than produced inside the body -- a value built during the trace is a constant
    exactly as the literal was, and one held on the declaration is an array the graph may
    not carry (`models/stated.py`, `_audit/optimise_design.md` §28, §34). The unit
    (`croco_turn_cable_space_extra_void`) still supplies it, through
    `indat.STATED_VALUES`."""


class CrocoInboardAreasAndFractions(ExplicitFunction):
    """cottax node: `tf_croco_inboard_areas_and_fractions`. No switch."""

    a_tf_wp_coolant_channels = OutputInto(tfcoil)
    a_tf_wp_conductor = OutputInto(tfcoil)
    a_tf_wp_extra_void = OutputInto(tfcoil)
    a_tf_coil_wp_turn_insulation = OutputInto(tfcoil)
    a_tf_wp_steel = OutputInto(tfcoil)
    a_tf_coil_inboard_steel = OutputInto(superconducting_tfcoil)
    f_a_tf_coil_inboard_steel = OutputInto(superconducting_tfcoil)
    a_tf_coil_inboard_insulation = OutputInto(superconducting_tfcoil)
    f_a_tf_coil_inboard_insulation = OutputInto(superconducting_tfcoil)

    def __call__(
        self,
        a_tf_turn_cable_space_no_void=From(tfcoil),
        n_tf_coil_turns=From(tfcoil),
        f_a_tf_turn_cable_space_extra_void=From(tfcoil),
        a_tf_turn_insulation=From(tfcoil),
        a_tf_turn_steel=From(tfcoil),
        a_tf_coil_inboard_case=From(tfcoil),
        n_tf_coils=From(tfcoil),
        a_tf_inboard_total=From(tfcoil),
        a_tf_wp_ground_insulation=From(superconducting_tfcoil),
        a_tf_croco_strand=From(superconducting_tfcoil),
    ):
        return croco_inboard_areas_and_fractions(
            a_tf_turn_cable_space_no_void=a_tf_turn_cable_space_no_void,
            n_tf_coil_turns=n_tf_coil_turns,
            f_a_tf_turn_cable_space_extra_void=f_a_tf_turn_cable_space_extra_void,
            a_tf_turn_insulation=a_tf_turn_insulation,
            a_tf_turn_steel=a_tf_turn_steel,
            a_tf_coil_inboard_case=a_tf_coil_inboard_case,
            n_tf_coils=n_tf_coils,
            a_tf_inboard_total=a_tf_inboard_total,
            a_tf_wp_ground_insulation=a_tf_wp_ground_insulation,
            a_tf_croco_strand=a_tf_croco_strand,
        )


class CrocoTurnCableSpaceCoolingFraction(ExplicitFunction):
    """cottax node: the one live line of `run`'s inline copper block
    (`superconducting.py:3947-3955`). No switch.

    A node of its own for the same reason `TfTurnArea` is one: it is a statement in
    `run` rather than a function, and its single output crosses into
    `quench_heat_protection_current_density`.
    """

    f_a_tf_turn_cable_space_cooling = OutputInto(superconducting_tfcoil)

    def __call__(
        self,
        a_tf_turn_cable_space_no_void=From(tfcoil),
        a_tf_croco_strand=From(superconducting_tfcoil),
    ):
        return croco_turn_cable_space_cooling_fraction(
            a_tf_turn_cable_space_no_void=a_tf_turn_cable_space_no_void,
            a_tf_croco_strand=a_tf_croco_strand,
        )


class CrocoSuperconductorProperties(ExplicitFunction):
    """The family that owns the CroCo critical-current chain -- constraint 33's read.

    `i_tf_sc_mat` decides it, over the three `SuperconductorShape.TAPE` values the
    function's own guard leaves standing (`superconducting.py:4435-4441`). One is
    written: `9`, both tracked ST files'. See the module docstring for `6` and `8`.

    **One output more than the cable-in-conduit family**: PROCESS writes the strand
    critical current to two fields in one chained assignment
    (`.superconducting_tfcoil.cur_tf_turn_croco_strand_critical` and
    `.c_tf_turn_cables_critical`, `:3997-3999`), where the CICC path writes only the
    second. Two `VarPath`s, one number -- transcribed rather than deduplicated, because
    that is what PROCESS writes; neither is read by any computation in `process/`
    (both go to a report), so nothing downstream can tell them apart.
    """

    j_tf_wp_critical = OutputInto(tfcoil)
    j_crit_str_tf = OutputInto(tfcoil)
    f_c_tf_turn_operating_critical = OutputInto(superconducting_tfcoil)
    j_tf_coil_turn = OutputInto(superconducting_tfcoil)
    j_tf_superconductor = OutputInto(superconducting_tfcoil)
    cur_tf_turn_croco_strand_critical = OutputInto(superconducting_tfcoil)
    c_tf_turn_cables_critical = OutputInto(superconducting_tfcoil)
    j_tf_superconductor_critical = OutputInto(superconducting_tfcoil)
    b_tf_superconductor_critical_zero_temp_strain = OutputInto(superconducting_tfcoil)
    temp_tf_superconductor_critical_zero_field_strain = OutputInto(
        superconducting_tfcoil
    )


class HazeltonZhaiRebcoCrocoSuperconductorProperties(CrocoSuperconductorProperties):
    """`i_tf_sc_mat == 9` *(live on both tracked ST files)*.

    `superconducting.py:4482-4538`. Reads no strain -- see
    `croco_superconductor_properties_hijc_rebco`.

    Owns both spellings of the strand critical current, because `run` assigns both in
    one chained statement (`:3997-3999`) and one number reaching two `VarPath`s is what
    the port is asked to reproduce.
    """

    def __call__(
        self,
        a_tf_turn=From(tfcoil),
        b_tf_inboard_peak_with_ripple=From(tfcoil),
        c_tf_turn=From(tfcoil),
        tftmp=From(tfcoil),
        dr_tf_hts_tape=From(superconducting_tfcoil),
        dx_tf_hts_tape_rebco=From(superconducting_tfcoil),
        dx_tf_hts_tape_total=From(superconducting_tfcoil),
        a_tf_croco_strand=From(superconducting_tfcoil),
    ):
        return croco_superconductor_properties_hijc_rebco(
            a_tf_turn=a_tf_turn,
            b_tf_inboard_peak=b_tf_inboard_peak_with_ripple,
            cur_tf_turn=c_tf_turn,
            temp_tf_peak=tftmp,
            dr_tf_hts_tape=dr_tf_hts_tape,
            dx_tf_hts_tape_rebco=dx_tf_hts_tape_rebco,
            dx_tf_hts_tape_total=dx_tf_hts_tape_total,
            a_tf_croco_strand=a_tf_croco_strand,
        )


class HazeltonZhaiRebcoCrocoTemperatureMargin(TfSuperconductorTemperatureMargin):
    """`i_tf_sc_mat == 9` -- constraint 36's read on a CroCo machine.

    Subclasses the cable-in-conduit family base because the *slot* is the same one and
    owns the same two fields, `.tfcoil.temp_tf_superconductor_margin` and
    `.tfcoil.temp_margin` (`superconducting.py:4006`, `:1278`), holding the same number.
    What differs is the residual: `hijc_rebco` takes three tape dimensions and no
    strain, so this occupant reads `.superconducting_tfcoil.dr_tf_hts_tape` and its two
    siblings where `_TemperatureMarginWithStrain` reads `.tfcoil.str_wp`.

    `.tfcoil.str_wp` is therefore **not** a boundary input of a CroCo machine's margin
    node -- but it still is of the machine, being read by `TfStress`'s consumers.
    """

    def __call__(
        self,
        j_tf_superconductor=From(superconducting_tfcoil),
        b_tf_inboard_peak_with_ripple=From(tfcoil),
        b_tf_superconductor_critical_zero_temp_strain=From(superconducting_tfcoil),
        temp_tf_superconductor_critical_zero_field_strain=From(superconducting_tfcoil),
        dr_tf_hts_tape=From(superconducting_tfcoil),
        dx_tf_hts_tape_rebco=From(superconducting_tfcoil),
        dx_tf_hts_tape_total=From(superconducting_tfcoil),
        tftmp=From(tfcoil),
    ):
        return calculate_hazelton_zhai_rebco_croco_temperature_margin(
            j_tf_superconductor,
            b_tf_inboard_peak_with_ripple,
            b_tf_superconductor_critical_zero_temp_strain,
            temp_tf_superconductor_critical_zero_field_strain,
            dr_tf_hts_tape,
            dx_tf_hts_tape_rebco,
            dx_tf_hts_tape_total,
            tftmp,
        )
