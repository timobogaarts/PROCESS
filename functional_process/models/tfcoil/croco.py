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

import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    OutputInto,
)

from functional_process.models.stated import StatesValues
from functional_process.models.physics.superconductors import hijc_rebco
from functional_process.models.safe_math import safe_sqrt
from functional_process.models.tfcoil.superconducting import (
    TfSuperconductorTemperatureMargin,
    _temperature_margin,
)
from functional_process.paths import superconducting_tfcoil, tfcoil

N_CROCO_STRANDS_TURN = 6
"""CroCo strands per TF turn. `process/models/superconductors.py:15`, a module constant
in PROCESS and a module constant here -- it is a count, not a switch."""


# ---------------------------------------------------------------------------
# `tf_croco_averaged_turn_geometry` -- `superconducting.py:4276-4389`
# ---------------------------------------------------------------------------


def croco_averaged_turn_geometry_from_current_per_turn(
    *,
    j_tf_wp,
    c_tf_turn,
    dx_tf_turn_steel,
    dx_tf_turn_insulation,
    layer_ins,
    a_tf_wp_no_insulation,
):
    """`i_dx_tf_turn_general_input == False` and
    `i_dx_tf_turn_cable_space_general_input == False` -- both tracked ST files' arm.

    Ports `process/models/tfcoil/superconducting.py:4340-4389`, the third branch plus
    the shared tail, minus the two dead outputs the module docstring tabulates.

    `c_tf_turn` is an **input** here and is not returned: PROCESS returns the parameter
    unchanged (`:4379-4389`) and `run` writes it back to the field it came from (`:3820`),
    which is an identity rather than a production -- the same shape
    `cicc_averaged_turn_geometry_from_current_per_turn` records as finding 1. Neither
    tracked ST file sets `c_tf_turn`, so it enters at
    `tfcoil_variables.py`'s default.

    Returns
    -------
    :
        `(a_tf_turn_insulation, n_tf_coil_turns, dx_tf_turn_general, dr_tf_turn,
        dx_tf_turn, dx_tf_turn_conduit_full_average, dx_tf_turn_cable_space_average)` --
        `run`'s own write order (`superconducting.py:3813-3833`), with
        `a_tf_turn_cable_space_no_void` and `a_tf_turn_steel` removed.
    """
    # Turn area, including conduit and inter-layer insulation [m^2].
    a_tf_turn = c_tf_turn / j_tf_wp

    # Side of the square cross-section of each turn, insulation included [m].
    dx_tf_turn_general = safe_sqrt(a_tf_turn)

    # Square-turn assumption: both extents are that side.
    dr_tf_turn = dx_tf_turn_general
    dx_tf_turn = dx_tf_turn_general

    # The conduit's outer side, after taking the inter-layer insulation off the turn.
    dx_tf_turn_conduit_full_average = (
        -layer_ins + safe_sqrt(layer_ins**2 + 4.0 * a_tf_turn)
    ) / 2 - 2.0 * dx_tf_turn_insulation

    # Turns per TF coil; not required to be an integer on this arm.
    n_tf_coil_turns = a_tf_wp_no_insulation / a_tf_turn

    # Inter-turn insulation area, one turn [m^2].
    a_tf_turn_insulation = a_tf_turn - dx_tf_turn_conduit_full_average**2

    # Diameter of the circular cable space inside the conduit [m].
    dx_tf_turn_cable_space_average = (
        dx_tf_turn_conduit_full_average - 2.0 * dx_tf_turn_steel
    )

    return (
        a_tf_turn_insulation,
        n_tf_coil_turns,
        dx_tf_turn_general,
        dr_tf_turn,
        dx_tf_turn,
        dx_tf_turn_conduit_full_average,
        dx_tf_turn_cable_space_average,
    )


# ---------------------------------------------------------------------------
# `tf_turn_croco_cable_space_properties` -- `superconducting.py:4560-4607`
# ---------------------------------------------------------------------------


def croco_cable_space_properties(*, dx_tf_turn_conduit_full_average, dx_tf_turn_steel):
    """The cable space of one CroCo turn: seven strands in a `3 x 3`-diameter square.

    Ports `process/models/tfcoil/superconducting.py:4581-4607`. The geometry is a
    conduit of side `dx_tf_turn_conduit_full_average` holding a `3 x 3` block of circles
    of diameter `dia_tf_turn_croco_cable`, of which six are CroCo strands and the
    seventh is the central copper bar -- which is where the `9/4 * pi * d^2` (nine
    quarter-circles' worth of area) and the `N + 1` in the cooling term come from.

    PROCESS's fifth return, `f_a_tf_turn_cable_space_cooling` (`:4597-4599`), is **not**
    computed here: `run` overwrites it before any reader (module docstring), and it is
    dimensionally an area on this line and a fraction on the one that replaces it. The
    harness's reference adapter slices PROCESS's dataclass to match.

    Returns
    -------
    :
        `(dia_tf_turn_croco_cable, a_tf_turn_cable_space_no_void,
        a_tf_turn_cable_space_effective, a_tf_turn_steel)` -- `run`'s write order
        (`superconducting.py:3845-3857`) minus the dead fifth.
    """
    # One strand's diameter: a third of the conduit bore, the steel walls removed.
    dia_tf_turn_croco_cable = (
        dx_tf_turn_conduit_full_average / 3.0 - dx_tf_turn_steel * (2.0 / 3.0)
    )

    # Area of the full cable circle in the turn [m^2].
    a_tf_turn_cable_space_no_void = 9.0 / 4.0 * jnp.pi * dia_tf_turn_croco_cable**2

    # The CroCo strands alone, voids excluded [m^2].
    a_tf_turn_cable_space_effective = (
        N_CROCO_STRANDS_TURN * jnp.pi * (dia_tf_turn_croco_cable / 2.0) ** 2
    )

    # Conduit jacket per turn: what the square has left over the cable circle [m^2].
    a_tf_turn_steel = dx_tf_turn_conduit_full_average**2 - a_tf_turn_cable_space_no_void

    return (
        dia_tf_turn_croco_cable,
        a_tf_turn_cable_space_no_void,
        a_tf_turn_cable_space_effective,
        a_tf_turn_steel,
    )


# ---------------------------------------------------------------------------
# `calculate_croco_cable_geometry` -- `process/models/superconductors.py:1117-1196`
# ---------------------------------------------------------------------------


def croco_cable_geometry(
    *,
    dia_croco_strand,
    dx_croco_strand_copper,
    dx_hts_tape_rebco,
    dx_hts_tape_copper,
    dx_hts_tape_hastelloy,
):
    """One CroCo strand: a copper tube around a soldered stack of REBCO tapes.

    Ports the module-level `calculate_croco_cable_geometry`,
    `process/models/superconductors.py:1117-1196`. PROCESS's `logger.error` on a
    non-positive tape-region diameter (`:1147-1148`) has no value effect and is dropped.

    **`n_croco_strand_hts_tapes` is a floor, and stays one.** PROCESS writes
    `np.floor(...).astype(int)`; here it is `jnp.floor`, i.e. the same number carried as
    a float. That keeps the four areas that multiply by it bit-identical, and it makes
    the node's derivative with respect to every tape thickness **zero almost
    everywhere** -- which is what the quantity's derivative is. PROCESS's own one-sided
    finite difference agrees except across a step, so the harness's gradient check is
    run at points away from one. Flagged as `piecewise-constant-integer-count` in the
    audit record.

    Returns
    -------
    :
        `(dia_croco_strand_tape_region, n_croco_strand_hts_tapes,
        a_croco_strand_copper_total, a_croco_strand_hastelloy, a_croco_strand_solder,
        a_croco_strand_rebco, a_croco_strand, dr_hts_tape, dx_hts_tape_total,
        dx_croco_strand_tape_stack)` -- `run`'s write order
        (`superconducting.py:3866-3888`), which is also the dataclass's field order.
    """
    # Inside the copper tube [m].
    dia_croco_strand_tape_region = dia_croco_strand - 2.0 * dx_croco_strand_copper

    # One tape, all layers [m].
    dx_hts_tape_total = dx_hts_tape_rebco + dx_hts_tape_copper + dx_hts_tape_hastelloy

    # Tape width, scaled off the reference 3.75 mm tape in a 5.4 mm bore [m].
    scaling = dia_croco_strand_tape_region / 5.4e-3
    dr_hts_tape = scaling * 3.75e-3

    # The chord the stack occupies across the bore [m].
    dx_croco_strand_tape_stack = safe_sqrt(
        dia_croco_strand_tape_region**2 - dr_hts_tape**2
    )

    # How many whole tapes fit in that chord.
    n_croco_strand_hts_tapes = jnp.floor(dx_croco_strand_tape_stack / dx_hts_tape_total)

    # Copper: the tube's annulus plus the copper layer of every tape [m^2].
    a_croco_strand_copper_total = (
        jnp.pi * dx_croco_strand_copper * dia_croco_strand
        - jnp.pi * dx_croco_strand_copper**2
        + dx_hts_tape_copper * dr_hts_tape * n_croco_strand_hts_tapes
    )

    # Hastelloy substrate, all tapes [m^2].
    a_croco_strand_hastelloy = (
        dx_hts_tape_hastelloy * dr_hts_tape * n_croco_strand_hts_tapes
    )

    # Solder: the bore minus the rectangle the stack fills [m^2].
    a_croco_strand_solder = (
        jnp.pi / 4.0 * dia_croco_strand_tape_region**2
        - dx_croco_strand_tape_stack * dr_hts_tape
    )

    # REBCO, all tapes [m^2].
    a_croco_strand_rebco = dx_hts_tape_rebco * dr_hts_tape * n_croco_strand_hts_tapes

    # The whole strand [m^2].
    a_croco_strand = jnp.pi / 4.0 * dia_croco_strand**2

    return (
        dia_croco_strand_tape_region,
        n_croco_strand_hts_tapes,
        a_croco_strand_copper_total,
        a_croco_strand_hastelloy,
        a_croco_strand_solder,
        a_croco_strand_rebco,
        a_croco_strand,
        dr_hts_tape,
        dx_hts_tape_total,
        dx_croco_strand_tape_stack,
    )


# ---------------------------------------------------------------------------
# `run`'s literal `f_a_tf_turn_cable_space_extra_void = 0.0`
# -- `superconducting.py:3894`
# ---------------------------------------------------------------------------


def croco_turn_cable_space_extra_void():
    """The extra void fraction of a CroCo cable space: exactly zero.

    `process/models/tfcoil/superconducting.py:3895`, one literal assignment in `run`,
    placed there because a CroCo turn has no central helium channel and no allowance
    beside its strands -- the void is the interstitial space between the seven circles
    and it is accounted for in `f_a_tf_turn_cable_space_cooling` instead.

    A node with no reads, the same shape as
    `models/blankets/hcpb.py::CentrepostNeutronicsAbsent`, and for the same reason: on
    this arm PROCESS's source *is* a literal. Owning it is what stops
    `.tfcoil.f_a_tf_turn_cable_space_extra_void` from re-entering the graph as a
    boundary input that the two nodes below would then read from the input file --
    `_audit/optimise_design.md` §16's defect class exactly.
    """
    return 0.0


# ---------------------------------------------------------------------------
# `tf_croco_inboard_areas_and_fractions` -- `superconducting.py:4609-4672`
# ---------------------------------------------------------------------------


def croco_inboard_areas_and_fractions(
    *,
    a_tf_turn_cable_space_no_void,
    n_tf_coil_turns,
    f_a_tf_turn_cable_space_extra_void,
    a_tf_turn_insulation,
    a_tf_turn_steel,
    a_tf_coil_inboard_case,
    n_tf_coils,
    a_tf_inboard_total,
    a_tf_wp_ground_insulation,
    a_tf_croco_strand,
):
    """Inboard winding-pack areas and steel/insulation fractions, CroCo.

    Ports `process/models/tfcoil/superconducting.py:4615-4672`. Two lines differ from
    the cable-in-conduit twin and both follow from the conductor:
    `a_tf_wp_coolant_channels` is a literal `0.0` (no central channel in a CroCo turn)
    and `a_tf_wp_conductor` counts strands rather than subtracting voids from the cable
    space. The remaining seven are the same arithmetic on the same fields.

    `run` recomputes `a_tf_wp_conductor` immediately afterwards (`:3938-3941`) with the
    identical expression -- a `redundant-duplicate-write`, kept once here.

    PROCESS logs an error when any of eight outputs comes out non-positive
    (`:4025-4050`); logging only, dropped.

    Returns
    -------
    :
        `(a_tf_wp_coolant_channels, a_tf_wp_conductor, a_tf_wp_extra_void,
        a_tf_coil_wp_turn_insulation, a_tf_wp_steel, a_tf_coil_inboard_steel,
        f_a_tf_coil_inboard_steel, a_tf_coil_inboard_insulation,
        f_a_tf_coil_inboard_insulation)`.
    """
    # No central helium channel in a CroCo conductor [m^2].
    a_tf_wp_coolant_channels = 0.0

    # Conductor is the strands themselves [m^2].
    a_tf_wp_conductor = n_tf_coil_turns * a_tf_croco_strand * N_CROCO_STRANDS_TURN

    # Void in the cable space, central channel excluded [m^2].
    a_tf_wp_extra_void = (
        a_tf_turn_cable_space_no_void
        * n_tf_coil_turns
        * f_a_tf_turn_cable_space_extra_void
    )

    # Inter-turn insulation, whole winding pack [m^2].
    a_tf_coil_wp_turn_insulation = n_tf_coil_turns * a_tf_turn_insulation

    # Steel conduit, whole winding pack [m^2].
    a_tf_wp_steel = n_tf_coil_turns * a_tf_turn_steel

    # Inboard steel: case plus conduit [m^2], and its share of the inboard leg.
    a_tf_coil_inboard_steel = a_tf_coil_inboard_case + a_tf_wp_steel
    f_a_tf_coil_inboard_steel = n_tf_coils * a_tf_coil_inboard_steel / a_tf_inboard_total

    # Inboard insulation: turn plus ground [m^2], and its share.
    a_tf_coil_inboard_insulation = (
        a_tf_coil_wp_turn_insulation + a_tf_wp_ground_insulation
    )
    f_a_tf_coil_inboard_insulation = (
        n_tf_coils * a_tf_coil_inboard_insulation / a_tf_inboard_total
    )

    return (
        a_tf_wp_coolant_channels,
        a_tf_wp_conductor,
        a_tf_wp_extra_void,
        a_tf_coil_wp_turn_insulation,
        a_tf_wp_steel,
        a_tf_coil_inboard_steel,
        f_a_tf_coil_inboard_steel,
        a_tf_coil_inboard_insulation,
        f_a_tf_coil_inboard_insulation,
    )


# ---------------------------------------------------------------------------
# The one live line of `run`'s inline copper block -- `superconducting.py:3947-3955`
# ---------------------------------------------------------------------------


def croco_turn_cable_space_cooling_fraction(
    *, a_tf_turn_cable_space_no_void, a_tf_croco_strand
):
    """The fraction of the cable space left for coolant.

    `process/models/tfcoil/superconducting.py:3948-3955`, transcribed as written: the
    cable space less "six strands minus the copper bar", over the cable space. The bar
    is `a_tf_croco_strand` (`:3930`), so the subtracted term is five strands -- but the
    expression is kept in PROCESS's own shape, because the two occurrences of
    `a_tf_croco_strand` mean different things (six cables, one bar) and collapsing them
    would hide that.

    The one line of the inline copper block that any computation reads:
    `quench_heat_protection_current_density` takes it (`:4246`). Everything else in that
    block is `output_croco_info`'s -- see the module docstring.
    """
    return (
        a_tf_turn_cable_space_no_void
        - (
            (N_CROCO_STRANDS_TURN * a_tf_croco_strand)
            - a_tf_croco_strand  # the central copper bar, `:3930`
        )
    ) / a_tf_turn_cable_space_no_void


# ---------------------------------------------------------------------------
# `tf_croco_superconductor_properties` -- `superconducting.py:4391-4558`
# ---------------------------------------------------------------------------


def _croco_superconductor_properties(
    *, j_superconductor_critical, a_tf_croco_strand, a_tf_turn, cur_tf_turn
):
    """The shared tail of `tf_croco_superconductor_properties`, `:4511-4538`.

    Every tape arm reaches it with its own `j_superconductor_critical`; nothing after
    the fit depends on which fit produced it.

    Returns
    -------
    :
        `(j_tf_wp_critical, j_crit_str_tf, f_c_tf_turn_operating_critical,
        j_tf_coil_turn, j_superconductor, cur_tf_turn_croco_strand_critical,
        c_turn_cables_critical)` -- `cicc_superconductor_properties`' shape plus the
        second spelling of the strand critical current, which `run` assigns in the same
        chained statement (`superconducting.py:3997-3999`).
    """
    # Strand critical current for costing in $/kAm. The tape fit already includes the
    # buffer and support layers, so -- unlike the cable-in-conduit twin -- there is no
    # copper fraction to divide out (`:4508-4513`).
    j_crit_str_tf = j_superconductor_critical

    # One strand's critical current [A].
    cur_tf_turn_croco_strand_critical = j_superconductor_critical * a_tf_croco_strand

    # The turn's, over its six strands [A].
    cur_tf_turn_critical = cur_tf_turn_croco_strand_critical * N_CROCO_STRANDS_TURN

    # Critical current density in the winding pack; `a_tf_turn` is the whole jacketed
    # conductor with insulation.
    j_tf_wp_critical = cur_tf_turn_critical / a_tf_turn

    # Ratio of operating to critical current.
    f_c_tf_turn_operating_critical = cur_tf_turn / cur_tf_turn_critical

    # Operating current density over the turn.
    j_tf_coil_turn = cur_tf_turn / a_tf_turn

    # Actual current density in the superconductor.
    j_superconductor = f_c_tf_turn_operating_critical * j_superconductor_critical

    return (
        j_tf_wp_critical,
        j_crit_str_tf,
        f_c_tf_turn_operating_critical,
        j_tf_coil_turn,
        j_superconductor,
        cur_tf_turn_croco_strand_critical,
        cur_tf_turn_croco_strand_critical,
    )


def croco_superconductor_properties_hijc_rebco(
    *,
    a_tf_turn,
    b_tf_inboard_peak,
    cur_tf_turn,
    temp_tf_peak,
    dr_tf_hts_tape,
    dx_tf_hts_tape_rebco,
    dx_tf_hts_tape_total,
    a_tf_croco_strand,
):
    """`i_tf_sc_mat == 9` -- Hazelton/Zhai high-current-density REBCO, both ST files'.

    Ports `process/models/tfcoil/superconducting.py:4482-4538`. `(bc20m, tc0m)` are the
    literals `(138, 92)` on this branch, so they are literals here and not reads.

    **The strain this branch clips is never used.** `:4486-4492` warns and clamps
    `|strain|` to `0.7e-2`, then calls `superconductors.hijc_rebco`, whose signature has
    no strain argument (`process/models/superconductors.py:728-736`). So the arm reads
    no strain field at all, and `i_str_wp` decides nothing here -- it stays a registry
    key because it decides something on arm `8`. The `logger.error` is dropped with the
    clip, having nothing left to guard.

    PROCESS's `logger.error` for a field above 14 T (`:4515-4521`, outside the fit's
    interpolation range) is likewise diagnostic only.

    Returns
    -------
    :
        `(j_tf_wp_critical, j_crit_str_tf, f_c_tf_turn_operating_critical,
        j_tf_coil_turn, j_superconductor, cur_tf_turn_croco_strand_critical,
        c_turn_cables_critical, j_superconductor_critical, bc20m, tc0m)`.
    """
    bc20m = 138.0
    tc0m = 92.0

    j_superconductor_critical, _, _ = hijc_rebco(
        temp_tf_peak,
        b_tf_inboard_peak,
        bc20m,
        tc0m,
        dr_tf_hts_tape,
        dx_tf_hts_tape_rebco,
        dx_tf_hts_tape_total,
    )

    return (
        *_croco_superconductor_properties(
            j_superconductor_critical=j_superconductor_critical,
            a_tf_croco_strand=a_tf_croco_strand,
            a_tf_turn=a_tf_turn,
            cur_tf_turn=cur_tf_turn,
        ),
        j_superconductor_critical,
        bc20m,
        tc0m,
    )


# ---------------------------------------------------------------------------
# `calculate_superconductor_temperature_margin`, tape arms
# -- `superconducting.py:1174-1291`
# ---------------------------------------------------------------------------


def temperature_margin_hijc_rebco(
    *,
    j_superconductor,
    b_tf_inboard_peak,
    b_c20max,
    temp_c0max,
    dr_hts_tape,
    dx_hts_tape_rebco,
    dx_hts_tape_total,
    temp_tf_coolant_peak_field,
):
    """Temperature margin on `i_tf_sc_mat == 9`.

    `superconductor_current_density_margin`'s branch 9
    (`process/models/superconductors.py:1272-1280`) driven by the same secant search
    every other arm uses -- `solve_current_sharing_temperature` in
    `models/tfcoil/superconducting.py`, which reproduces `scipy.optimize.newton`'s
    stopping rule step for step. Read that function's docstring for why.

    Unlike the Nb3Sn arms this residual takes no strain and three tape dimensions
    instead, which is the whole difference between the two nodes.
    """

    def margin_fn(temperature):
        j_critical, _, _ = hijc_rebco(
            temperature,
            b_tf_inboard_peak,
            b_c20max,
            temp_c0max,
            dr_hts_tape,
            dx_hts_tape_rebco,
            dx_hts_tape_total,
        )
        return j_critical - j_superconductor

    return _temperature_margin(
        margin_fn=margin_fn, temp_tf_coolant_peak_field=temp_tf_coolant_peak_field
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
    (`superconducting.py:3894`). Reads nothing.

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
        margin = temperature_margin_hijc_rebco(
            j_superconductor=j_tf_superconductor,
            b_tf_inboard_peak=b_tf_inboard_peak_with_ripple,
            b_c20max=b_tf_superconductor_critical_zero_temp_strain,
            temp_c0max=temp_tf_superconductor_critical_zero_field_strain,
            dr_hts_tape=dr_tf_hts_tape,
            dx_hts_tape_rebco=dx_tf_hts_tape_rebco,
            dx_hts_tape_total=dx_tf_hts_tape_total,
            temp_tf_coolant_peak_field=tftmp,
        )
        return margin, margin
