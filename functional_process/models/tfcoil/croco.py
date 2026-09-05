"""Pure functions for the CroCo (cross-conductor) REBCO-tape TF coil, extracted from
`functional_process/cottax/tfcoil/croco.py`.

That module still holds the graph declarations (`ExplicitFunction` occupants, gathered
into `namespace.CrocoSuperconductingTfCoil`) that wire these functions to `VarPath`s;
read its module docstring for the switch splits, what is deliberately unported, and how
this sibling relates to `superconducting.py`. The audit record is
`functional_process/_audit/units/models/tfcoil/croco.md` and mirrors these functions,
not the declarations that call them.
"""

import jax.numpy as jnp

from functional_process.models.physics.superconductors import hijc_rebco
from functional_process.models.safe_math import safe_sqrt
from functional_process.models.tfcoil.superconducting import _temperature_margin

N_CROCO_STRANDS_TURN = 6
"""CroCo strands per TF turn. `process/models/superconductors.py:15`, a module constant
in PROCESS and a module constant here -- it is a count, not a switch."""


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


def croco_turn_cable_space_extra_void():
    """The extra void fraction of a CroCo cable space: exactly zero.

    `process/models/tfcoil/superconducting.py:3895`, one literal assignment in `run`,
    placed there because a CroCo turn has no central helium channel and no allowance
    beside its strands -- the void is the interstitial space between the seven circles
    and it is accounted for in `f_a_tf_turn_cable_space_cooling` instead.

    A node that computes nothing, the same shape as
    `models/blankets/hcpb.py::CentrepostNeutronicsAbsent`, and for the same reason: on
    this arm PROCESS's source *is* a literal. (Both state their outputs rather than
    returning them -- `models/stated.py` -- so each reads exactly its own statement.) Owning it is what stops
    `.tfcoil.f_a_tf_turn_cable_space_extra_void` from re-entering the graph as a
    boundary input that the two nodes below would then read from the input file --
    `_audit/optimise_design.md` §16's defect class exactly.
    """
    return 0.0


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


def calculate_hazelton_zhai_rebco_croco_temperature_margin(
    j_tf_superconductor,
    b_tf_inboard_peak_with_ripple,
    b_tf_superconductor_critical_zero_temp_strain,
    temp_tf_superconductor_critical_zero_field_strain,
    dr_tf_hts_tape,
    dx_tf_hts_tape_rebco,
    dx_tf_hts_tape_total,
    tftmp,
):
    """`i_tf_sc_mat == 9`'s temperature margin, written to both
    `.tfcoil.temp_tf_superconductor_margin` and `.tfcoil.temp_margin` -- one number to
    two `VarPath`s, as PROCESS's own chained assignment does (see
    `HazeltonZhaiRebcoCrocoTemperatureMargin`'s docstring).
    """
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
