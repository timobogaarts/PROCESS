"""The Central Solenoid's superconductor properties -- `superconpf`'s ITER Nb3Sn arm.

Audit record: `functional_process/_audit/units/models/pfcoil/superconductor.md`, which
the wave that wrote this module could not create -- `unit_registry.md` was held open by
two sibling agents, so the material went into `pfcoil/fields.md` § "the CS chain" and
this docstring said a row was owed. Both were done on 2026-08-29 and the material moved
unchanged.

Ports the `ohcalc` block at `process/models/pfcoil.py:3577-3684` -- two `superconpf`
calls, at the end of flat-top and at the beginning of pulse, and the four
`.pf_coil.*` fields they produce that a constraint reads.

**Two arms of `superconpf`, not `superconpf`, and two is the whole of what can be
reached.** `.pf_coil.i_cs_superconductor` selects among eight critical-surface
parameterisations (`superconpf`, `pfcoil.py:4641-4924`), but `indat._pf_coil_system_arm`
has already refused every value but `1` and `5` before this module's slot is built --
they are the two halves of the `(i_pf_superconductor, i_cs_superconductor)` pair the
package's occupant sets are keyed on. Both are written here: `1` (`ITER_NB3SN`,
`large_tokamak_eval.IN.DAT:245`) and `5` (`WST_NB3SN`,
`low_aspect_ratio_DEMO.IN.DAT:845`). `indat.CS_SUPERCONDUCTOR` is therefore **total**
over the values that reach it, and has no `UNPORTED` entries at all.

The other six read genuinely different constants and, in three cases, different
variables -- `BI2212` computes its own temperature margin and never calls the root find
below, `HAZELTON_ZHAI_REBCO` reads the three `.superconducting_tfcoil.d*_hts_tape*`
fields nothing else touches -- so each would need its own occupant if the pair predicate
were ever widened. The critical-surface fits themselves are all already ported in
`functional_process/models/physics/superconductors.py` and are not re-ported here.

**The two occupants differ in one call and nothing else**, which is why they share a
base class the way `masses.PFCoilMassesCsWstNb3Sn` shares `PFCoilMasses`: identical
reads, identical outputs, one substituted critical-surface function and its two
constants.

**The temperature margin is a root find, and it is `superconpf`'s only non-arithmetic
step.** PROCESS solves `j_crit_sc(T) - j_sc = 0` for `T` with `scipy.optimize.newton`'s
secant iteration (`pfcoil.py:4906-4921`; `fprime=None`, `x1 = 2*T_op`, `tol = 1e-6`,
`rtol = 1e-6`, `maxiter = 50`, `disp=False`).

**Ported 2026-08-30, and the shared driver this module was waiting for is the TF coil's**
(`models/tfcoil/superconducting.py::solve_current_sharing_temperature`). That is why the
margin was deferred rather than written twice: `.tfcoil.temp_tf_superconductor_margin`
(constraint 36) is the same `scipy.optimize.newton` call on the same critical-surface
functions, it landed first, and it is imported here rather than re-derived. So the pair
is a `jax.lax.fori_loop` inside an `ExplicitFunction`, exactly as the TF coil's is, and
**not** the declared `ImplicitFunction`/`RootFind` pair this docstring predicted -- the
same decision `models/vacuum/vacuum.py::solve_duct_diameter` and the TF margin both
already made, for the reason `solve_current_sharing_temperature`'s own docstring gives:
the endpoint *is* the quantity constraint 60 compares against PROCESS's, so reproducing
scipy's stopping rule is what makes that a value test.
"""

import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    OutputInto,
)

from functional_process.models.pfcoil import CS_INDEX
from functional_process.models.physics.superconductors import (
    itersc,
    western_superconducting_nb3sn,
)
from functional_process.models.tfcoil.superconducting import (
    solve_current_sharing_temperature,
)
from functional_process.paths import pf_coil, tfcoil

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


class CSCriticalCurrentDensitiesIterNb3Sn(ExplicitFunction):
    """cottax node: `.tokamak.cs_coil.critical_current`, `i_cs_superconductor == 1`.

    Owns the two critical current densities constraints 26 and 27 compare against the
    operating ones, the two conductor-level densities behind them, and the strand
    density the costing model reads.

    Added 2026-08-27 for `optimise_design.md` §11.5:
    `.pf_coil.j_cs_critical_flat_top_end` and `.j_cs_critical_pulse_start` were
    boundary zeros against PROCESS's converged `3.780e7` / `3.841e7` A/m^2, so both
    constraints compared a real operating current density against a frozen zero.

    **`.pf_coil.temp_cs_superconductor_margin` is still not owned here**, and that is
    now a placement decision rather than a deferral: it is
    `CSTemperatureMarginIterNb3Sn` below, `.tokamak.cs_coil.temperature_margin`, landed
    2026-08-30. Two nodes rather than one for the same reason the TF coil has
    `cicc_superconductor_properties` and `tf_superconductor_temperature_margin` as two
    slots -- and because the margin declares a read this node argues it must not:
    `.pf_coil.c_pf_cs_coils_peak_ma`, dead work for the critical current density and the
    only input the root find needs beyond what is already here.

    The paragraph this replaces recorded the margin as owed to a shared driver with the
    TF coil's, "one commit rather than two". That is what happened: the driver is
    `models/tfcoil/superconducting.py::solve_current_sharing_temperature` and the CS arm
    imports it.

    `_critical_surface` is the one thing the WST Nb3Sn sibling below overrides.
    """

    j_cs_critical_flat_top_end = OutputInto(pf_coil)
    j_cs_critical_pulse_start = OutputInto(pf_coil)
    j_cs_conductor_critical_flat_top_end = OutputInto(pf_coil)
    j_cs_conductor_critical_pulse_start = OutputInto(pf_coil)
    j_crit_str_cs = OutputInto(pf_coil)

    _critical_surface = staticmethod(calculate_cs_critical_current_density_iter_nb3sn)
    """The arm's critical-current function. A `staticmethod` rather than an
    `eqx.field(static=True)` integer: the switch is answered by *which class is bound*,
    and the class is where the answer belongs (`_audit/next_steps.md` §14.2)."""

    def __call__(
        self,
        b_cs_peak_flat_top_end=From(pf_coil),
        b_cs_peak_pulse_start=From(pf_coil),
        f_a_cs_void=From(pf_coil),
        fcuohsu=From(pf_coil),
        str_cs_con_res=From(tfcoil),
        temp_cs_superconductor_operating=From(pf_coil),
        a_cs_cable_space=From(pf_coil),
        a_cs_poloidal=From(pf_coil),
    ):
        shared = {
            "f_a_cs_void": f_a_cs_void,
            "fcuohsu": fcuohsu,
            "strain": str_cs_con_res,
            "temp_cs_superconductor_operating": temp_cs_superconductor_operating,
            "a_cs_cable_space": a_cs_cable_space,
            "a_cs_poloidal": a_cs_poloidal,
        }
        j_flat_top_end, j_cond_flat_top_end = self._critical_surface(
            b_cs_peak=b_cs_peak_flat_top_end, **shared
        )
        j_pulse_start, j_cond_pulse_start = self._critical_surface(
            b_cs_peak=b_cs_peak_pulse_start, **shared
        )
        return (
            j_flat_top_end,
            j_pulse_start,
            j_cond_flat_top_end,
            j_cond_pulse_start,
            calculate_cs_strand_critical_current_density(j_cond_flat_top_end, fcuohsu),
        )


class CSCriticalCurrentDensitiesWstNb3Sn(CSCriticalCurrentDensitiesIterNb3Sn):
    """cottax node: `.tokamak.cs_coil.critical_current`, `i_cs_superconductor == 5`.

    `low_aspect_ratio_DEMO.IN.DAT:845`'s value. One overridden `staticmethod` and
    nothing else -- identical reads, identical outputs, `superconpf`'s WST Nb3Sn
    critical surface in place of the ITER one. Subclassing rather than a second full
    class for the same reason `masses.PFCoilMassesCsWstNb3Sn` does it: the slot is a
    **place**, and the node bound at `.tokamak.cs_coil.critical_current` keeps its name
    whichever occupant fills it.

    **Written because refusing it would have broken a file that already assembled.**
    `indat._pf_coil_system_arm`'s arm `1` exists precisely for
    `low_aspect_ratio_DEMO`'s `(3, 5)` superconductor pair, so a registry with only the
    ITER occupant would have turned an assembling machine into a `NotImplementedError`
    -- caught by `test_machine.test_a_switch_that_decides_two_slots_decides_both`, which
    builds that machine. A new slot may not narrow the set of files the port accepts.
    """

    _critical_surface = staticmethod(calculate_cs_critical_current_density_wst_nb3sn)


class CSTemperatureMarginIterNb3Sn(ExplicitFunction):
    """cottax node: `.tokamak.cs_coil.temperature_margin`, `i_cs_superconductor == 1`.

    Owns `.pf_coil.temp_cs_superconductor_margin`, constraint 60's read and the last of
    the four constraints `optimise_design.md` §11.5 found reading an `ohcalc` field with
    no producer. Landed 2026-08-30 as a missing producer measured on
    `large_tokamak_nof`: the port had `0.0` against PROCESS's `3.4208032` K, and
    constraint 60 (`temp_cs_superconductor_margin >= temp_cs_superconductor_margin_min`)
    was therefore comparing a frozen zero against a real bound.

    **A slot of its own, not a fifth output of `critical_current`.** The two nodes come
    out of the same PROCESS call -- `ohcalc` invokes `superconpf` twice and takes four of
    its returns -- but they have different read sets: only the margin needs
    `.pf_coil.c_pf_cs_coils_peak_ma`, whose edge `CSCriticalCurrentDensitiesIterNb3Sn`
    explicitly declines to invent because the critical-surface arms never use it. Same
    shape as the TF coil's two slots (`cicc_superconductor_properties` and
    `tf_superconductor_temperature_margin`), which split the same way for the same
    reason.

    `_critical_surface` is the one thing the WST Nb3Sn sibling below overrides -- and
    unlike `CSCriticalCurrentDensitiesIterNb3Sn`'s, it is the *arm function*, not an
    arm-specific wrapper, because the margin's two arms differ only in which fit and
    which pair of constants `_cs_temperature_margin_pair` is handed.
    """

    temp_cs_superconductor_margin = OutputInto(pf_coil)

    _critical_surface = staticmethod(calculate_cs_temperature_margin_iter_nb3sn)

    def __call__(
        self,
        b_cs_peak_flat_top_end=From(pf_coil),
        b_cs_peak_pulse_start=From(pf_coil),
        c_pf_cs_coils_peak_ma=From(pf_coil),
        a_cs_cable_space=From(pf_coil),
        f_a_cs_void=From(pf_coil),
        fcuohsu=From(pf_coil),
        str_cs_con_res=From(tfcoil),
        temp_cs_superconductor_operating=From(pf_coil),
    ):
        return self._critical_surface(
            b_cs_peak_flat_top_end=b_cs_peak_flat_top_end,
            b_cs_peak_pulse_start=b_cs_peak_pulse_start,
            c_pf_cs_coils_peak_ma=c_pf_cs_coils_peak_ma[CS_INDEX],
            a_cs_cable_space=a_cs_cable_space,
            f_a_cs_void=f_a_cs_void,
            fcuohsu=fcuohsu,
            strain=str_cs_con_res,
            temp_cs_superconductor_operating=temp_cs_superconductor_operating,
        )


class CSTemperatureMarginWstNb3Sn(CSTemperatureMarginIterNb3Sn):
    """cottax node: `.tokamak.cs_coil.temperature_margin`, `i_cs_superconductor == 5`.

    `low_aspect_ratio_DEMO.IN.DAT:845`'s value, and written for the same reason its
    `critical_current` sibling was: `indat._pf_coil_system_arm`'s arm `1` exists for that
    file's `(3, 5)` superconductor pair, so a registry with only the ITER occupant would
    turn a machine that assembles today into a `NotImplementedError`.
    """

    _critical_surface = staticmethod(calculate_cs_temperature_margin_wst_nb3sn)


# `.pf_coil.j_pf_wp_critical` -- left unowned, and named here so a reader who greps for
# it finds the reason rather than an omission. `ohcalc:3670-3675` writes the
# beginning-of-pulse winding-pack critical density into the CS slot and then copies it
# straight back out into `.pf_coil.j_cs_critical_pulse_start` (`:3677-3679`), so the CS
# slot is the same number under two names and the second name is the one every consumer
# reads. The six PF slots are `pfcoil()`'s (`:877`), whose own `superconpf` calls
# `masses.md` records as computed-and-discarded and which this pass does not port; owning
# one slot of a seven-slot array whose other six have no producer would claim more than
# is here.
