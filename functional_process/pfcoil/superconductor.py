"""Pure functions for the PF/CS superconductor critical current density and
temperature margin, extracted from `functional_process/models/pfcoil/superconductor.py`.

That module still holds the graph declarations (`ExplicitFunction` occupants) that wire
these functions to `VarPath`s; read its module docstring (and
`functional_process/models/pfcoil/__init__.py`'s package docstring) for scope. The
audit record is `functional_process/_audit/units/models/pfcoil/superconductor.md` and
mirrors these functions, not the declarations that call them.
"""

import jax.numpy as jnp

from functional_process.models.pfcoil import CS_INDEX
from functional_process.models.physics.superconductors import (
    hijc_rebco,
    itersc,
    jcrit_nbti,
    western_superconducting_nb3sn,
)
from functional_process.tfcoil.superconducting import solve_current_sharing_temperature

ITER_NB3SN_B_C20MAX = 32.97
"""`bc20m` for the ITER Nb3Sn arm (T) -- critical field at 0 K and 0 strain,
`process/models/pfcoil.py:4733`. A literal inside the arm, not a `DataStructure` field:
`.tfcoil.bcritsc` is read only by the `USER_DEFINED_NB3SN` arm (`:4787-4788`), which is
why this occupant does not declare it."""


ITER_NB3SN_TEMP_C0MAX = 16.06
"""`tc0m` for the ITER Nb3Sn arm (K) -- critical temperature at 0 T and 0 strain,
`pfcoil.py:4734`. Same argument as `ITER_NB3SN_B_C20MAX` for `.tfcoil.tcritsc`."""


WST_NB3SN_B_C20MAX = 32.97
"""`bc20m` for the WST Nb3Sn arm (T), `pfcoil.py:4795`. **The same literal as the ITER
arm's**, which is worth naming rather than deduplicating: the two arms happen to share
both constants and differ only in the critical-surface function
(`western_superconducting_nb3sn` carries its own `csc`/`p`/`q`/`c_a1`/`epsilon_0a`).
Kept as two named pairs so a change to one of PROCESS's arms cannot silently move the
other."""


WST_NB3SN_TEMP_C0MAX = 16.06
"""`tc0m` for the WST Nb3Sn arm (K), `pfcoil.py:4796`. See `WST_NB3SN_B_C20MAX`."""


OLD_LUBELL_NBTI_B_C20MAX = 15.0
"""`bc20m` for the NbTi arm (T) -- upper critical field at 0 K and 0 strain,
`process/models/pfcoil.py:4775`. The PF coils' conductor on **both** ported arms
(`.pf_coil.i_pf_superconductor == 3`, `large_tokamak_eval.IN.DAT:246` and
`low_aspect_ratio_DEMO.IN.DAT:806`) -- `indat._pf_coil_system_deviations` refuses every
other value of that switch, which is why `PFStrandCriticalCurrentDensity` is one class
and not a family the way its two CS neighbours are."""


OLD_LUBELL_NBTI_TEMP_C0MAX = 9.3
"""`tc0m` for the NbTi arm (K) -- critical temperature at 0 T and 0 strain,
`pfcoil.py:4776`."""


OLD_LUBELL_NBTI_C0 = 1.0e10
"""`c0` for the NbTi arm (A/m^2) -- the Lubell scaling constant, `pfcoil.py:4777`. A
literal inside the arm like its two siblings, and the same three literals
`models/tfcoil/superconducting.py:1798-1800` names for `supercon`'s NbTi arm; kept as a
third named constant here rather than imported from there, for the reason
`WST_NB3SN_B_C20MAX` gives -- a change to one of PROCESS's arms must not silently move
another."""


def calculate_cs_critical_current_density(
    j_crit_sc,
    f_a_cs_void,
    fcuohsu,
    a_cs_cable_space,
    a_cs_poloidal,
):
    """Filament critical current density scaled to the whole CS cross-section (A/m^2).

    The half of `superconpf` that every arm shares: `j_crit_cable_frac`
    (`process/models/pfcoil.py:4724-4729`) and `jcritwp = j_crit_cable` (`:4870`),
    followed by the two `ohcalc` lines that scale cable space to the full poloidal
    cross-section (`:3630-3634` at end of flat-top, `:3670-3679` at beginning of pulse
    -- the same two lines twice).

    Parameters
    ----------
    j_crit_sc :
        Critical current density in the superconductor filaments alone (A/m^2).
    f_a_cs_void :
        Fraction of the cable space taken by helium coolant (`fhe`).
    fcuohsu :
        Copper fraction of the CS cable conductor (`fcu`).
    a_cs_cable_space :
        CS cable-space cross-sectional area (m^2).
    a_cs_poloidal :
        CS total poloidal cross-sectional area (m^2).

    Returns
    -------
    :
        Critical current density referred to the whole CS cross-section (A/m^2).
    """
    j_crit_cable = j_crit_sc * (1.0 - fcuohsu) * (1.0 - f_a_cs_void)
    return j_crit_cable * a_cs_cable_space / a_cs_poloidal


def calculate_cs_critical_current_density_iter_nb3sn(
    b_cs_peak,
    f_a_cs_void,
    fcuohsu,
    strain,
    temp_cs_superconductor_operating,
    a_cs_cable_space,
    a_cs_poloidal,
):
    """CS winding-pack and conductor critical current densities, ITER Nb3Sn (A/m^2).

    Ports `superconpf`'s `ITER_NB3SN` arm (`process/models/pfcoil.py:4731-4746`,
    `:4870-4872`) plus the shared scaling above.

    The chain, in PROCESS's own terms:

    - `j_crit_sc` is the critical current density **in the superconductor filaments
      alone**, from `superconductors.itersc` at the ITER fitting constants.
    - `j_crit_cable = j_crit_sc * (1 - fcu) * (1 - fhe)` -- filaments to cable, taking
      out the copper in the strand and the helium in the cable space
      (`j_crit_cable_frac`, `:4724-4729`).
    - `jcritwp = j_crit_cable` (`:4870`), and `ohcalc` then scales cable space to the
      full poloidal cross-section: `j_cs_critical = jcritwp * a_cs_cable_space /
      a_cs_poloidal`.

    **`j_pf_wp` is not read on this arm.** `superconpf` takes the operating winding-pack
    current density as an argument and the ITER Nb3Sn arm never uses it for the critical
    current -- only `BI2212` does (`:4762`), and only the temperature margin needs it
    otherwise. `ohcalc` computes it anyway at both call sites
    (`abs(c_pf_cs_coils_peak_ma[6]) / a_cs_cable_space * 1e6`, `:3597-3606`), which is
    dead work on this arm; declaring the read would invent an edge from
    `.pf_coil.c_pf_cs_coils_peak_ma` that this configuration does not make.

    Parameters
    ----------
    b_cs_peak :
        Peak field at the CS conductor (T) -- `.pf_coil.b_cs_peak_flat_top_end` or
        `.b_cs_peak_pulse_start`.
    f_a_cs_void :
        Fraction of the cable space taken by helium coolant (`fhe`).
    fcuohsu :
        Copper fraction of the CS cable conductor (`fcu`).
    strain :
        Strain on the CS superconductor at operating conditions.
        `.tfcoil.str_cs_con_res`.
    temp_cs_superconductor_operating :
        Helium temperature at the peak-field point (K).
    a_cs_cable_space :
        CS cable-space cross-sectional area (m^2).
    a_cs_poloidal :
        CS total poloidal cross-sectional area (m^2).

    Returns
    -------
    tuple
        `(j_cs_critical, j_cs_conductor_critical)` -- the critical current density
        referred to the whole CS cross-section and the critical current density in the
        superconductor filaments (both A/m^2).
    """
    j_crit_sc, _b_critical, _temp_critical = itersc(
        temp_conductor=temp_cs_superconductor_operating,
        b_conductor=b_cs_peak,
        strain=strain,
        b_c20max=ITER_NB3SN_B_C20MAX,
        temp_c0max=ITER_NB3SN_TEMP_C0MAX,
    )
    return (
        calculate_cs_critical_current_density(
            j_crit_sc, f_a_cs_void, fcuohsu, a_cs_cable_space, a_cs_poloidal
        ),
        j_crit_sc,
    )


def calculate_cs_critical_current_density_wst_nb3sn(
    b_cs_peak,
    f_a_cs_void,
    fcuohsu,
    strain,
    temp_cs_superconductor_operating,
    a_cs_cable_space,
    a_cs_poloidal,
):
    """CS winding-pack and conductor critical current densities, WST Nb3Sn (A/m^2).

    Ports `superconpf`'s `WST_NB3SN` arm (`process/models/pfcoil.py:4793-4808`) plus the
    shared scaling. Identical to the ITER Nb3Sn arm above but for the critical-surface
    function and its two constants -- and even the constants happen to match; see
    `WST_NB3SN_B_C20MAX`.

    Parameters
    ----------
    b_cs_peak :
        Peak field at the CS conductor (T).
    f_a_cs_void :
        Fraction of the cable space taken by helium coolant.
    fcuohsu :
        Copper fraction of the CS cable conductor.
    strain :
        Strain on the CS superconductor. `.tfcoil.str_cs_con_res`.
    temp_cs_superconductor_operating :
        Helium temperature at the peak-field point (K).
    a_cs_cable_space, a_cs_poloidal :
        CS cable-space and total poloidal cross-sectional areas (m^2).

    Returns
    -------
    tuple
        `(j_cs_critical, j_cs_conductor_critical)`, both A/m^2.
    """
    j_crit_sc, _b_critical, _temp_critical = western_superconducting_nb3sn(
        temp_conductor=temp_cs_superconductor_operating,
        b_conductor=b_cs_peak,
        strain=strain,
        b_c20max=WST_NB3SN_B_C20MAX,
        temp_c0max=WST_NB3SN_TEMP_C0MAX,
    )
    return (
        calculate_cs_critical_current_density(
            j_crit_sc, f_a_cs_void, fcuohsu, a_cs_cable_space, a_cs_poloidal
        ),
        j_crit_sc,
    )


def calculate_cs_strand_critical_current_density(
    j_cs_conductor_critical_flat_top_end, fcuohsu
):
    """CS strand critical current density for costing, `$/kA m` (A/m^2).

    Ports `ohcalc`'s `:3619-3628`. The `i_cs_superconductor in {2, 6, 8}` branch (Bi-2212
    and the two REBCO parameterisations, whose fits already return a whole-strand
    density) returns the conductor value unscaled; every other value takes the copper
    fraction out -- and both values that reach this package (`1` and `5`) are in that
    "every other".

    Written as the `else` arm only, because `.pf_coil.i_cs_superconductor` is answered by
    this module's occupants and not by a traced branch -- the `{2, 6, 8}` arm belongs to
    occupants that do not exist and that `indat._pf_coil_system_arm` refuses first.

    Parameters
    ----------
    j_cs_conductor_critical_flat_top_end :
        Critical current density in the CS superconductor filaments at the end of
        flat-top (A/m^2).
    fcuohsu :
        Copper fraction of the CS cable conductor.

    Returns
    -------
    :
        Strand critical current density (A/m^2).
    """
    return j_cs_conductor_critical_flat_top_end * (1.0 - fcuohsu)


def calculate_pf_strand_critical_current_density(
    b_pf_coil_peak, bpf2, temp_pf_peak_field, fcupfsu
):
    """PF strand critical current density for costing, `$/kA m` (A/m^2).

    Ports `pfcoil()`'s `:871-904`: the peak field at one PF coil, `superconpf`'s
    `OLD_LUBELL_NBTI` arm (`:4773-4784`), and the `else` half of the strand branch at
    `:898-904`.

    **PROCESS keeps one scalar for six coils, and the last coil wins.** The block sits
    inside `pfcoil()`'s group-then-coil loop and assigns `.pf_coil.j_crit_str_pf`
    unconditionally on every pass (`:900`, `:902`), with no index -- so the value that
    survives the loop is the last PF coil's, index `N_PF_COILS - 1`. That is reproduced
    rather than averaged or re-indexed: Account 222.2's `PER_KAM` arm reads the scalar,
    and the scalar is the last coil's. The five overwritten values are not computed here
    at all, which is the same "dropped rather than computed and discarded" rule
    `masses.calculate_pf_coil_masses` applies to the rest of this block.

    **The critical-surface arm reads neither `fhe` nor `fcu`.** `superconpf` scales
    filaments to cable with both (`j_crit_cable_frac`, `:4724-4729`) but that product is
    `jcritwp`, its *first* return; the third, `j_crit_sc`, comes straight off the fit.
    So `.pf_coil.f_a_pf_coil_void` is not a read of this unit even though it is a read of
    `superconpf`, and `fcupfsu` enters only through the strand branch below.

    **The strand branch is written as its `else` arm only.** `:898` tests
    `i_cs_superconductor in {2, 6, 8}` -- the **CS** switch, deciding the **PF** value,
    which is PROCESS as written and is reproduced as written. Both values that reach this
    package (`1` and `5`, `indat.CS_SUPERCONDUCTOR`) are outside that set, so the `if`
    arm belongs to occupants that do not exist and that `_pf_coil_system_arm` refuses
    first -- the same argument `calculate_cs_strand_critical_current_density` makes for
    its own `else`.

    Parameters
    ----------
    b_pf_coil_peak, bpf2 :
        Field at the inner and outer edge of the last PF coil (T) --
        `.pf_coil.b_pf_coil_peak[5]` and `.pf_coil.bpf2[5]`. `superconpf`'s `b_pf_peak`
        is `max(|b_pf_coil_peak|, |bpf2|)` (`:871-874`).
    temp_pf_peak_field :
        Helium temperature at the peak-field point (K). `.tfcoil.tftmp`.
    fcupfsu :
        Copper fraction of the PF superconducting strand. `.pf_coil.fcupfsu`.

    Returns
    -------
    :
        Strand critical current density (A/m^2).
    """
    b_pf_peak = jnp.maximum(jnp.abs(b_pf_coil_peak), jnp.abs(bpf2))
    j_crit_sc, _temp_critical = jcrit_nbti(
        temp_conductor=temp_pf_peak_field,
        b_conductor=b_pf_peak,
        c0=OLD_LUBELL_NBTI_C0,
        b_c20max=OLD_LUBELL_NBTI_B_C20MAX,
        temp_c0max=OLD_LUBELL_NBTI_TEMP_C0MAX,
    )
    return j_crit_sc * (1.0 - fcupfsu)


HAZELTON_ZHAI_REBCO_B_C20MAX = 138.0
"""`bc20m` in `superconpf`'s `HAZELTON_ZHAI_REBCO` arm (`pfcoil.py:4853`) -- upper
critical field at zero temperature and strain (T)."""


HAZELTON_ZHAI_REBCO_T_C0 = 92.0
"""`tc0m` in the same arm (`pfcoil.py:4854`) -- critical temperature at zero field and
strain (K)."""


def calculate_pf_strand_critical_current_density_hazelton_zhai_rebco(
    b_pf_coil_peak,
    bpf2,
    temp_pf_peak_field,
    fcupfsu,
    dr_hts_tape,
    dx_hts_tape_rebco,
    dx_hts_tape_total,
):
    """`calculate_pf_strand_critical_current_density` for REBCO tape PF coils.

    The same block of `pfcoil()` (`:871-904`) with `superconpf`'s
    `HAZELTON_ZHAI_REBCO` arm (`:4851-4866`) in place of the `OLD_LUBELL_NBTI` one --
    `i_pf_superconductor = 9`, both spherical tokamaks'
    (`spherical_tokamak_eval.IN.DAT:235`, `st_regression.IN.DAT:1670`).

    Everything the NbTi sibling's docstring argues carries over unchanged: the scalar is
    the *last* PF coil's because the assignment has no index, `fhe`/`fcu` do not enter
    the critical surface (they scale `superconpf`'s first return, not its third), and
    the strand branch at `:898` is written as its `else` arm because both reachable
    values of the **CS** switch it tests are outside `{2, 6, 8}` -- including on a
    machine with no CS at all, where `.pf_coil.i_cs_superconductor` keeps
    `pfcoil_variables.py`'s default `1`.

    **Three reads the NbTi arm does not have**: the tape's width and its REBCO/total
    thicknesses, which live in `.superconducting_tfcoil` and are the TF coil's tape
    geometry. That is PROCESS's own wiring (`pfcoil.py:892-894` passes exactly those
    three fields into the PF call), not a shortcut -- there is no PF-side tape geometry
    in `pfcoil_variables.py` to read instead.

    Parameters
    ----------
    b_pf_coil_peak, bpf2 :
        Field at the inner and outer edge of the last PF coil (T).
    temp_pf_peak_field :
        Helium temperature at the peak-field point (K). `.tfcoil.tftmp`.
    fcupfsu :
        Copper fraction of the PF superconducting strand. `.pf_coil.fcupfsu`.
    dr_hts_tape, dx_hts_tape_rebco, dx_hts_tape_total :
        Tape width, REBCO layer thickness and total tape thickness (m).
        `.superconducting_tfcoil.dr_tf_hts_tape`, `.dx_tf_hts_tape_rebco`,
        `.dx_tf_hts_tape_total`.

    Returns
    -------
    :
        Strand critical current density (A/m^2).
    """
    b_pf_peak = jnp.maximum(jnp.abs(b_pf_coil_peak), jnp.abs(bpf2))
    j_crit_sc, _b_critical, _temp_critical = hijc_rebco(
        temp_conductor=temp_pf_peak_field,
        b_conductor=b_pf_peak,
        b_c20max=HAZELTON_ZHAI_REBCO_B_C20MAX,
        t_c0=HAZELTON_ZHAI_REBCO_T_C0,
        dr_hts_tape=dr_hts_tape,
        dx_hts_tape_rebco=dx_hts_tape_rebco,
        dx_hts_tape_total=dx_hts_tape_total,
    )
    return j_crit_sc * (1.0 - fcupfsu)


def calculate_cs_superconductor_current_density(
    c_pf_cs_coils_peak_ma, a_cs_cable_space, f_a_cs_void, fcuohsu
):
    """The CS's *operating* current density in the filaments alone (A/m^2).

    `superconpf`'s three lines between the critical-surface dispatch and the root find
    (`process/models/pfcoil.py:4874-4878`), with `ohcalc`'s own spelling of `j_pf_wp`
    (`pfcoil.py:3597-3606`, the same expression at both call sites) folded in:

        j_pf_wp  = |c_pf_cs_coils_peak_ma[CS]| / a_cs_cable_space * 1e6
        jstrand  = j_pf_wp / (1 - fhe)      # cable space to conductor
        jsc      = jstrand / (1 - fcu)      # conductor to non-copper filaments

    **This is the read `CSCriticalCurrentDensitiesIterNb3Sn` declines to declare, and it
    stops being dead work here.** That node's docstring records that `ohcalc` computes
    `j_pf_wp` at both call sites and that the ITER/WST arms never use it -- true of the
    *critical* current density, and the reason the edge from
    `.pf_coil.c_pf_cs_coils_peak_ma` was not invented there. The temperature margin is
    the one consumer, so the edge is declared by this node instead of that one.

    Parameters
    ----------
    c_pf_cs_coils_peak_ma :
        Peak current in the CS (MA) -- `.pf_coil.c_pf_cs_coils_peak_ma[CS_INDEX]`.
        Despite the name PROCESS stores MA here and multiplies by `1e6` at every use.
    a_cs_cable_space :
        CS cable-space cross-sectional area (m^2).
    f_a_cs_void :
        Fraction of the cable space taken by helium coolant (`fhe`).
    fcuohsu :
        Copper fraction of the CS cable conductor (`fcu`).

    Returns
    -------
    :
        Operating current density in the superconducting filaments (A/m^2).
    """
    j_pf_wp = jnp.abs(c_pf_cs_coils_peak_ma) / a_cs_cable_space * 1.0e6
    return j_pf_wp / (1.0 - f_a_cs_void) / (1.0 - fcuohsu)


def _cs_temperature_margin(
    *,
    critical_surface,
    b_c20max,
    temp_c0max,
    b_cs_peak,
    strain,
    temp_cs_superconductor_operating,
    j_superconductor,
):
    """`t_zero_margin - temp_pf_peak_field` for one peak field (`pfcoil.py:4906-4922`).

    The residual is `superconductors.superconductor_current_density_margin` with
    everything but the temperature bound -- and for the two arms that reach this package
    that is the arm's own critical-surface fit minus `jsc`
    (`process/models/superconductors.py:1259` and `:1262-1264`).

    `solve_current_sharing_temperature` is the TF coil's replica of
    `scipy.optimize.newton`'s secant branch, imported rather than repeated: PROCESS
    passes the same `tol`/`rtol`/`maxiter`/`x1 = 2 * T_op` here as it does there
    (`pfcoil.py:4908-4920` against `tfcoil/superconducting.py:1265-1281`), so a second
    copy could only drift.
    """

    def margin_fn(temperature):
        return (
            critical_surface(
                temp_conductor=temperature,
                b_conductor=b_cs_peak,
                strain=strain,
                b_c20max=b_c20max,
                temp_c0max=temp_c0max,
            )[0]
            - j_superconductor
        )

    return (
        solve_current_sharing_temperature(
            margin_fn=margin_fn, temp_start=temp_cs_superconductor_operating
        )
        - temp_cs_superconductor_operating
    )


def calculate_cs_temperature_margin_iter_nb3sn(
    b_cs_peak_flat_top_end,
    b_cs_peak_pulse_start,
    c_pf_cs_coils_peak_ma,
    a_cs_cable_space,
    f_a_cs_void,
    fcuohsu,
    strain,
    temp_cs_superconductor_operating,
):
    """CS temperature margin, ITER Nb3Sn (K) -- `min` of the two operating points.

    Ports `ohcalc`'s `temp_cs_superconductor_margin = min(tmarg1, tmarg2)`
    (`process/models/pfcoil.py:3679`), where `tmarg1` is `superconpf`'s fourth return at
    the end-of-flat-top peak field (`:3586-3618`) and `tmarg2` the same at the
    beginning-of-pulse peak field (`:3636-3665`). Constraint 60 compares it against
    `.tfcoil.temp_cs_superconductor_margin_min`.

    **The operating current density is the same at both points and the peak field is
    not.** `superconpf` is handed the identical `j_pf_wp` twice (`ohcalc` recomputes the
    same expression), so the two solves differ only in `b_pf_peak` -- which is why
    `min` is not symmetric in anything but the field.

    Parameters
    ----------
    b_cs_peak_flat_top_end, b_cs_peak_pulse_start :
        Peak field at the CS conductor at the two operating points (T).
    c_pf_cs_coils_peak_ma :
        Peak current in the CS (MA), `.pf_coil.c_pf_cs_coils_peak_ma[CS_INDEX]`.
    a_cs_cable_space, f_a_cs_void, fcuohsu :
        Cable-space area (m^2), helium fraction of it, copper fraction of the conductor.
    strain :
        Strain on the CS superconductor. `.tfcoil.str_cs_con_res`.
    temp_cs_superconductor_operating :
        Helium temperature at the peak-field point (K) -- both the solve's `x0` and the
        temperature the margin is measured from.

    Returns
    -------
    :
        Temperature margin (K).
    """
    return _cs_temperature_margin_pair(
        critical_surface=itersc,
        b_c20max=ITER_NB3SN_B_C20MAX,
        temp_c0max=ITER_NB3SN_TEMP_C0MAX,
        b_cs_peak_flat_top_end=b_cs_peak_flat_top_end,
        b_cs_peak_pulse_start=b_cs_peak_pulse_start,
        c_pf_cs_coils_peak_ma=c_pf_cs_coils_peak_ma,
        a_cs_cable_space=a_cs_cable_space,
        f_a_cs_void=f_a_cs_void,
        fcuohsu=fcuohsu,
        strain=strain,
        temp_cs_superconductor_operating=temp_cs_superconductor_operating,
    )


def calculate_cs_temperature_margin_wst_nb3sn(
    b_cs_peak_flat_top_end,
    b_cs_peak_pulse_start,
    c_pf_cs_coils_peak_ma,
    a_cs_cable_space,
    f_a_cs_void,
    fcuohsu,
    strain,
    temp_cs_superconductor_operating,
):
    """CS temperature margin, WST Nb3Sn (K). Same shape as the ITER arm above.

    `superconpf`'s `WST_NB3SN` arm (`pfcoil.py:4793-4808`) feeding the same root find;
    `low_aspect_ratio_DEMO.IN.DAT:845`'s conductor. See
    `calculate_cs_temperature_margin_iter_nb3sn` for the parameters.
    """
    return _cs_temperature_margin_pair(
        critical_surface=western_superconducting_nb3sn,
        b_c20max=WST_NB3SN_B_C20MAX,
        temp_c0max=WST_NB3SN_TEMP_C0MAX,
        b_cs_peak_flat_top_end=b_cs_peak_flat_top_end,
        b_cs_peak_pulse_start=b_cs_peak_pulse_start,
        c_pf_cs_coils_peak_ma=c_pf_cs_coils_peak_ma,
        a_cs_cable_space=a_cs_cable_space,
        f_a_cs_void=f_a_cs_void,
        fcuohsu=fcuohsu,
        strain=strain,
        temp_cs_superconductor_operating=temp_cs_superconductor_operating,
    )


def _cs_temperature_margin_pair(
    *,
    critical_surface,
    b_c20max,
    temp_c0max,
    b_cs_peak_flat_top_end,
    b_cs_peak_pulse_start,
    c_pf_cs_coils_peak_ma,
    a_cs_cable_space,
    f_a_cs_void,
    fcuohsu,
    strain,
    temp_cs_superconductor_operating,
):
    """`min(tmarg1, tmarg2)` for one critical surface -- the two public arms' body."""
    j_superconductor = calculate_cs_superconductor_current_density(
        c_pf_cs_coils_peak_ma, a_cs_cable_space, f_a_cs_void, fcuohsu
    )
    shared = {
        "critical_surface": critical_surface,
        "b_c20max": b_c20max,
        "temp_c0max": temp_c0max,
        "strain": strain,
        "temp_cs_superconductor_operating": temp_cs_superconductor_operating,
        "j_superconductor": j_superconductor,
    }
    return jnp.minimum(
        _cs_temperature_margin(b_cs_peak=b_cs_peak_flat_top_end, **shared),
        _cs_temperature_margin(b_cs_peak=b_cs_peak_pulse_start, **shared),
    )


def calculate_cs_critical_current_densities(
    critical_surface,
    b_cs_peak_flat_top_end,
    b_cs_peak_pulse_start,
    f_a_cs_void,
    fcuohsu,
    str_cs_con_res,
    temp_cs_superconductor_operating,
    a_cs_cable_space,
    a_cs_poloidal,
):
    """`CSCriticalCurrentDensitiesIterNb3Sn`/`CSCriticalCurrentDensitiesWstNb3Sn`:
    evaluates the arm's critical surface at both time points and derives the strand
    density from the flat-top-end conductor density.
    """
    shared = {
        "f_a_cs_void": f_a_cs_void,
        "fcuohsu": fcuohsu,
        "strain": str_cs_con_res,
        "temp_cs_superconductor_operating": temp_cs_superconductor_operating,
        "a_cs_cable_space": a_cs_cable_space,
        "a_cs_poloidal": a_cs_poloidal,
    }
    j_flat_top_end, j_cond_flat_top_end = critical_surface(
        b_cs_peak=b_cs_peak_flat_top_end, **shared
    )
    j_pulse_start, j_cond_pulse_start = critical_surface(
        b_cs_peak=b_cs_peak_pulse_start, **shared
    )
    return (
        j_flat_top_end,
        j_pulse_start,
        j_cond_flat_top_end,
        j_cond_pulse_start,
        calculate_cs_strand_critical_current_density(j_cond_flat_top_end, fcuohsu),
    )


def calculate_cs_temperature_margin_from_full_width_current(
    critical_surface,
    b_cs_peak_flat_top_end,
    b_cs_peak_pulse_start,
    c_pf_cs_coils_peak_ma,
    a_cs_cable_space,
    f_a_cs_void,
    fcuohsu,
    str_cs_con_res,
    temp_cs_superconductor_operating,
):
    """`CSTemperatureMarginIterNb3Sn`/`CSTemperatureMarginWstNb3Sn`: picks the CS's own
    peak current out of the full-width array before calling the arm's temperature-margin
    fit.
    """
    return critical_surface(
        b_cs_peak_flat_top_end=b_cs_peak_flat_top_end,
        b_cs_peak_pulse_start=b_cs_peak_pulse_start,
        c_pf_cs_coils_peak_ma=c_pf_cs_coils_peak_ma[CS_INDEX],
        a_cs_cable_space=a_cs_cable_space,
        f_a_cs_void=f_a_cs_void,
        fcuohsu=fcuohsu,
        strain=str_cs_con_res,
        temp_cs_superconductor_operating=temp_cs_superconductor_operating,
    )


def calculate_pf_strand_critical_current_density_hazelton_zhai_rebco_topology(
    b_pf_coil_peak,
    bpf2,
    tftmp,
    fcupfsu,
    dr_tf_hts_tape,
    dx_tf_hts_tape_rebco,
    dx_tf_hts_tape_total,
    *,
    topology,
):
    """`PFStrandCriticalCurrentDensityHazeltonZhaiRebco`: picks the last PF coil's peak
    field out of the whole-array reads.
    """
    last = topology.n_pf_coils - 1
    return calculate_pf_strand_critical_current_density_hazelton_zhai_rebco(
        b_pf_coil_peak=b_pf_coil_peak[last],
        bpf2=bpf2[last],
        temp_pf_peak_field=tftmp,
        fcupfsu=fcupfsu,
        dr_hts_tape=dr_tf_hts_tape,
        dx_hts_tape_rebco=dx_tf_hts_tape_rebco,
        dx_hts_tape_total=dx_tf_hts_tape_total,
    )
