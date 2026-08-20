"""Pure-functional port of `process/models/availability.py` (registry unit #17).

Audit record: `functional_process/models/availability.md`. `Availability.run()` dispatches
on `.costs.i_plant_availability` (`AvailabilityModel`) to one of three whole-branch
alternatives -- `avail()` (USER_INPUT/WARD_TAYLOR, 0/1), `avail_2()` (MORRIS, 2), `avail_st()`
(ST, 3). All three are self-contained (no calls into other, unported `Model`s) and are
ported here as tier-1 pure functions, composed from a shared set of leaf helpers used by
two or three of the branches at once (`calculate_divertor_lifetime`,
`calculate_u_unplanned_*`, the two `calculate_cp_lifetime_*` alternatives).

Two switches are split into **separate node alternatives** rather than kept as a static
branch inside one function, matching `i_tf_sup`'s precedent in
`stellarator_F_tf_nuclear_heating.py`:

- `.tfcoil.i_tf_sup` selects between `calculate_cp_lifetime_superconducting` and
  `calculate_cp_lifetime_resistive` -- both branches of the source's `cp_lifetime` are
  non-trivial (unlike the TF-coil precedent's all-zero resistive branch), so this is two
  real alternative producers of one slot (`.costs.cplife`), not "the absence of a node".
- `.costs.i_plant_availability`'s USER_INPUT/WARD_TAYLOR split (0 vs 1, both reachable
  inside `avail()`) is **not a formula switch at all** once separated out: for
  USER_INPUT (0), `f_t_plant_available` is never computed by `avail()` -- the source
  simply never touches it, leaving the input value in place. That is exactly cottax's "no
  `InputNode`": `.costs.f_t_plant_available` has *no producer* on that branch, it is a
  boundary input. `calculate_ward_taylor_availability` is therefore the WARD_TAYLOR-only
  producer of that slot; `calculate_avail` (the rest of `avail()`, common to both) takes
  `f_t_plant_available` as a plain input regardless of which branch supplied it.

Every other switch touched here (`.costs.ibkt_life`, `.physics.itart`) is kept as a
static `eqx.field` per `naming_convention.md`'s "switches are not ports" -- see the audit
record's "switches touched" section for why these were not also split.

`.physics.itart` gates whether `.costs.cplife` is *computed* by `avail()`/`avail_2()`'s
`calc_u_planned` at all (a `conditional-ownership-by-run-config` case, same shape as
`stellarator_C_geometry.md`'s `.physics.aspect` finding) -- ported by threading a
`cplife_in` passthrough argument rather than resolving the ownership question here; see
the record. `avail_st()` differs: it computes `.costs.cplife` **unconditionally**, and
only the later *lifetime-adjustment* step is `itart`-gated -- the two `itart` gates are
not the same gate reused, see the record's data-footprint table.

At the **node** level, this conditional/unconditional read-then-write of `.costs.cplife`
within one function body is a genuine Shape B self-loop (`next_steps.md` §5): a node
whose own `Output` and `FromExactly` name the identical `VarPath`, which `cottax.spec`'s
`__check_init__` refuses outright (`reads [...], which it also owns`). `CplifeAvail`
(shared by `Avail`/`Avail2`) and `CplifeAvailSt` isolate exactly that self-reference as
`FixedPointFunction` declarations -- `Avail`/`Avail2`/`AvailSt` themselves are now
ordinary `ExplicitFunction`s over the *rest* of each branch's outputs, reading
`.costs.cplife` (or, for `AvailSt`, the same recompute inputs `CplifeAvailSt` uses) as a
plain value rather than also owning it. See the "cottax node" section below and
`availability.md`'s "cottax node" section for the split's exact shape and why `AvailSt`'s
`ExplicitFunction` half cannot simply read `.costs.cplife` back (the mod-adjusted value
`CplifeAvailSt` owns is not the same number `avail_st()`'s own `shortest_lifetime`
needs).

`.vacuum.n_vac_pumps_high` and `.costs.redun_vac` feed a Python `range()` inside
`calculate_u_unplanned_vacuum` (the source's cryopump-redundancy sum) -- both are
genuinely `int`-typed PROCESS fields, so they are ordinary (non-`jnp`) Python arguments,
declared `static` in the harness (`static_argnames`) and as `eqx.field(static=True)` on
the node. `calculate_redun_vac` itself is plain Python (`math.floor`, not `jnp`): it must
be resolved to a concrete int *before* tracing, since its result becomes another node's
loop bound -- see the record's JAX-difficulty flags.
"""

import math

import equinox as eqx
import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    FixedPointFunction,
    FromExactly,
    Output,
)

from functional_process.models.switch_enums import (
    BlanketLifetimeModel,
    SphericalTokamakModel,
)
from process.models.tfcoil.base import TFConductorModel

DAY_SECONDS = 60 * 60 * 24
DAYS_IN_YEAR = 365.25
YEAR_SECONDS = DAY_SECONDS * DAYS_IN_YEAR

_REF_FUSION_POWER_MW = 2.0e3
_REF_DPA_FPY = 10.0


# ---------------------------------------------------------------------------
# Leaf helpers shared by two or three of the three top-level branches
# ---------------------------------------------------------------------------


def calculate_dpa_per_fpy(p_fusion_total_mw):
    """DPA/FPY scaled from EU-DEMO's reference point (T. Franke 2020).

    Ports the identical three-line block repeated in `avail`, `calc_u_planned` and
    `avail_st`.

    Parameters
    ----------
    p_fusion_total_mw :
        Total fusion power (MW). `.physics.p_fusion_total_mw`.

    Returns
    -------
    :
        `dpa_fpy`.
    """
    f_scale = p_fusion_total_mw / _REF_FUSION_POWER_MW
    return f_scale * _REF_DPA_FPY


def calculate_divertor_lifetime(adivflnc, pflux_div_heat_load_mw, life_plant):
    """Divertor lifetime from allowable heat fluence.

    Ports `Availability.divertor_lifetime`.

    Parameters
    ----------
    adivflnc :
        Allowable divertor heat fluence (MW-yr/m2). `.costs.adivflnc`.
    pflux_div_heat_load_mw :
        Divertor heat load (MW/m2). `.divertor.pflux_div_heat_load_mw`.
    life_plant :
        Total plant lifetime (years). `.costs.life_plant`.

    Returns
    -------
    :
        `life_div_fpy`.
    """
    return jnp.maximum(0.0, jnp.minimum(adivflnc / pflux_div_heat_load_mw, life_plant))


def calculate_cp_lifetime_superconducting(neut_flux_cp, flu_tf_neutron_fast_max, life_plant):
    """Centrepost lifetime, `.tfcoil.i_tf_sup == SUPERCONDUCTING` branch.

    Ports `Availability.cp_lifetime`'s SC branch. Mutually exclusive alternative to
    `calculate_cp_lifetime_resistive` -- see module docstring.

    Parameters
    ----------
    neut_flux_cp :
        Centrepost TF fast neutron flux (m^-2 s^-1). `.fwbs.neut_flux_cp`.
    flu_tf_neutron_fast_max :
        Max allowed fast neutron fluence on the TF coil (n/m2).
        `.constraints.flu_tf_neutron_fast_max`.
    life_plant :
        Total plant lifetime (years). `.costs.life_plant`.

    Returns
    -------
    :
        `cplife`.
    """
    no_flux = neut_flux_cp <= 0.0
    safe_flux = jnp.where(no_flux, 1.0, neut_flux_cp)
    limited = jnp.minimum(flu_tf_neutron_fast_max / (safe_flux * YEAR_SECONDS), life_plant)
    return jnp.where(no_flux, life_plant, limited)


def calculate_cp_lifetime_resistive(cpstflnc, pflux_fw_neutron_mw, life_plant):
    """Centrepost lifetime, `.tfcoil.i_tf_sup != SUPERCONDUCTING` branch.

    Ports `Availability.cp_lifetime`'s aluminium/copper branch. Mutually exclusive
    alternative to `calculate_cp_lifetime_superconducting`.

    Parameters
    ----------
    cpstflnc :
        Allowable ST centrepost neutron fluence (MW-yr/m2). `.costs.cpstflnc`.
    pflux_fw_neutron_mw :
        Average neutron wall load (MW/m2). `.physics.pflux_fw_neutron_mw`.
    life_plant :
        Total plant lifetime (years). `.costs.life_plant`.

    Returns
    -------
    :
        `cplife`.
    """
    return jnp.minimum(cpstflnc / pflux_fw_neutron_mw, life_plant)


def calculate_u_unplanned_magnets(
    temp_tf_superconductor_margin_min,
    temp_cs_superconductor_margin_min,
    t_plant_operational_total_yrs,
    conf_mag,
    temp_margin,
):
    """Unplanned unavailability of the magnets.

    Ports `Availability.calc_u_unplanned_magnets`.

    Parameters
    ----------
    temp_tf_superconductor_margin_min, temp_cs_superconductor_margin_min :
        Minimum TF/CS superconductor temperature margins (K). `.tfcoil.*`.
    t_plant_operational_total_yrs :
        Total DT operational time (years). `.costs.t_plant_operational_total_yrs`.
    conf_mag :
        Temperature-margin risk-onset factor. `.costs.conf_mag`.
    temp_margin :
        Actual TF coil temperature margin (K). `.tfcoil.temp_margin`.

    Returns
    -------
    :
        `u_unplanned_magnets`.
    """
    tmargmin = jnp.minimum(
        temp_tf_superconductor_margin_min, temp_cs_superconductor_margin_min
    )
    mag_main_time = 0.5
    mag_min_u_unplanned = mag_main_time / (t_plant_operational_total_yrs + mag_main_time)

    start_of_risk = tmargmin / conf_mag
    below_risk = temp_margin >= start_of_risk

    # Guarded: `start_of_risk - tmargmin` is exactly the boundary `below_risk` tests, so
    # an unguarded division would leak a 0/0 NaN gradient through the unselected branch.
    safe_denom = jnp.where(below_risk, 1.0, start_of_risk - tmargmin)
    t_life = jnp.maximum(
        0.0,
        (t_plant_operational_total_yrs / safe_denom) * (temp_margin - tmargmin),
    )
    at_risk = mag_main_time / (t_life + mag_main_time)

    return jnp.where(below_risk, mag_min_u_unplanned, at_risk)


def calculate_u_unplanned_divertor(
    life_div_fpy, t_plant_pulse_total, div_prob_fail, div_umain_time, div_nu, div_nref
):
    """Unplanned unavailability of the divertor.

    Ports `Availability.calc_u_unplanned_divertor`.

    Parameters
    ----------
    life_div_fpy :
        Divertor lifetime (FPY). `.costs.life_div_fpy`.
    t_plant_pulse_total :
        Total pulse length (s). `.times.t_plant_pulse_total`.
    div_prob_fail :
        Divertor failure probability per operational day. `.costs.div_prob_fail`.
    div_umain_time :
        Divertor repair time (years). `.costs.div_umain_time`.
    div_nu :
        Cycle at which failure is 100% certain. `.costs.div_nu`.
    div_nref :
        Reference cycle life. `.costs.div_nref`.

    Returns
    -------
    :
        `u_unplanned_div`.
    """
    n = life_div_fpy * YEAR_SECONDS / t_plant_pulse_total
    pf = (div_prob_fail / DAY_SECONDS) * t_plant_pulse_total
    a0 = 1.0 - pf * div_umain_time * YEAR_SECONDS / t_plant_pulse_total

    below = n <= div_nref
    above = n >= div_nu
    edge = below | above
    safe_n = jnp.where(edge, 1.0, n)
    between = (a0 / (div_nu - div_nref)) * (div_nu - 0.5 * div_nref**2.0 / safe_n - 0.5 * safe_n)

    div_avail = jnp.where(below, a0, jnp.where(above, 0.0, between))
    return 1.0 - div_avail


def calculate_u_unplanned_fwbs(
    life_blkt_fpy, t_plant_pulse_total, fwbs_prob_fail, fwbs_umain_time, fwbs_nu, fwbs_nref
):
    """Unplanned unavailability of the first wall / blanket.

    Ports `Availability.calc_u_unplanned_fwbs` -- same shape as
    `calculate_u_unplanned_divertor`, different field names.

    Parameters
    ----------
    life_blkt_fpy :
        Blanket lifetime (FPY). `.fwbs.life_blkt_fpy`.
    t_plant_pulse_total :
        Total pulse length (s). `.times.t_plant_pulse_total`.
    fwbs_prob_fail, fwbs_umain_time, fwbs_nu, fwbs_nref :
        First wall/blanket failure probability, repair time, and cycle-life reference
        points. `.costs.fwbs_*`.

    Returns
    -------
    :
        `u_unplanned_fwbs`.
    """
    n = life_blkt_fpy * YEAR_SECONDS / t_plant_pulse_total
    pf = (fwbs_prob_fail / DAY_SECONDS) * t_plant_pulse_total
    a0 = 1.0 - pf * fwbs_umain_time * YEAR_SECONDS / t_plant_pulse_total

    below = n <= fwbs_nref
    above = n >= fwbs_nu
    edge = below | above
    safe_n = jnp.where(edge, 1.0, n)
    between = (a0 / (fwbs_nu - fwbs_nref)) * (
        fwbs_nu - 0.5 * fwbs_nref**2.0 / safe_n - 0.5 * safe_n
    )

    fwbs_avail = jnp.where(below, a0, jnp.where(above, 0.0, between))
    return 1.0 - fwbs_avail


def calculate_u_unplanned_bop(t_plant_operational_total_yrs):
    """Unplanned unavailability of the balance of plant.

    Ports `Availability.calc_u_unplanned_bop`.

    Parameters
    ----------
    t_plant_operational_total_yrs :
        Total DT operational time (years). `.costs.t_plant_operational_total_yrs`.

    Returns
    -------
    :
        `u_unplanned_bop`.
    """
    bop_fail_rate = 9.39e-5
    bop_num_failures = jnp.ceil(
        bop_fail_rate * DAYS_IN_YEAR * 24.0 * t_plant_operational_total_yrs
    )
    bop_mttr = 96.0 / (24.0 * DAYS_IN_YEAR)
    return (bop_mttr * bop_num_failures) / t_plant_operational_total_yrs


def calculate_u_unplanned_hcd():
    """Unplanned unavailability of the heating/current-drive system.

    Ports `Availability.calc_u_unplanned_hcd` -- a fixed placeholder value, no inputs.

    Returns
    -------
    :
        `u_unplanned_hcd` (`0.02`).
    """
    return 0.02


def calculate_redun_vac(n_vac_pumps_high, redun_vacp):
    """Number of redundant vacuum pumps.

    Ports `avail_2`/`avail_st`'s `redun_vac` line. **Plain Python, not `jnp`**: its
    result becomes `calculate_u_unplanned_vacuum`'s Python `range()` bound, so it must be
    a concrete int resolved before any trace -- see module docstring and the audit
    record's JAX-difficulty flags.

    Parameters
    ----------
    n_vac_pumps_high :
        Number of high-vacuum pumps. `.vacuum.n_vac_pumps_high`.
    redun_vacp :
        Redundant-pump percentage. `.costs.redun_vacp`.

    Returns
    -------
    :
        `redun_vac`, as a Python `int`.
    """
    return math.floor(n_vac_pumps_high * redun_vacp / 100.0 + 0.5)


def calculate_u_unplanned_vacuum(
    t_plant_operational_total_yrs,
    life_plant,
    num_rh_systems,
    n_vac_pumps_high,
    redun_vac,
):
    """Unplanned unavailability of the vacuum system.

    Ports `Availability.calc_u_unplanned_vacuum`. `n_vac_pumps_high`/`redun_vac` are
    static (see module docstring): they set the Python `range()`/`total_pumps` bound of
    the source's cryopump-redundancy sum, which cannot be a traced value.

    Parameters
    ----------
    t_plant_operational_total_yrs :
        Total DT operational time (years). `.costs.t_plant_operational_total_yrs`.
    life_plant :
        Total plant lifetime (years). `.costs.life_plant`.
    num_rh_systems :
        Number of remote-handling systems. `.costs.num_rh_systems`.
    n_vac_pumps_high :
        Number of high-vacuum pumps, **static**. `.vacuum.n_vac_pumps_high`.
    redun_vac :
        Number of redundant pumps, **static**. `.costs.redun_vac`.

    Returns
    -------
    :
        `u_unplanned_vacuum`.
    """
    n_shutdown = jnp.round(
        (life_plant - t_plant_operational_total_yrs)
        / ((21.0 * num_rh_systems ** (-0.9) + 2.0) / 12.0)
    )
    t_op_bt = t_plant_operational_total_yrs / (n_shutdown + 1.0)

    cryo_main_time = 1.0 / 6.0
    total_pumps = n_vac_pumps_high + redun_vac

    cryo_failure_rate = 2.0e-6 * DAYS_IN_YEAR * 24.0 * t_op_bt
    cryo_nfailure_rate = 1.0 - cryo_failure_rate

    sum_prob = 0.0
    for n in range(redun_vac + 1, total_pumps + 1):
        sum_prob = sum_prob + (
            math.comb(total_pumps, n)
            * (cryo_nfailure_rate ** (total_pumps - n))
            * (cryo_failure_rate**n)
            * (n - redun_vac)
        )

    t_down = (n_shutdown + 1.0) * cryo_main_time * sum_prob
    return jnp.maximum(0.005, t_down / (t_plant_operational_total_yrs + t_down))


# ---------------------------------------------------------------------------
# `avail()` (USER_INPUT / WARD_TAYLOR, `i_plant_availability` in {0, 1})
# ---------------------------------------------------------------------------


def calculate_blanket_lifetime_fpy_avail(
    life_fw_fpy, ibkt_life, abktflnc, pflux_fw_neutron_mw, life_dpa, dpa_fpy, life_plant
):
    """Blanket lifetime, `avail()`'s own four-way branch.

    Ports `avail`'s blanket-lifetime block (source lines 161-191). `ibkt_life` is static
    (`switches are not ports`). Unlike `calculate_blanket_lifetime_fpy_simple` below, this
    guards `pflux_fw_neutron_mw == 0.0` in the `ibkt_life == 0` sub-branch -- the source
    only does so here, not in `calc_u_planned`/`avail_st`'s copies of the `ibkt_life == 0`
    formula; see the audit record's PROCESS-bug note.

    Parameters
    ----------
    life_fw_fpy :
        Pre-computed first-wall lifetime (FPY), if already known. `.fwbs.life_fw_fpy`.
    ibkt_life :
        Blanket-lifetime model switch, **static**. `.costs.ibkt_life`.
    abktflnc :
        Allowable blanket neutron fluence (MW-yr/m2). `.costs.abktflnc`.
    pflux_fw_neutron_mw :
        Average neutron wall load (MW/m2). `.physics.pflux_fw_neutron_mw`.
    life_dpa :
        Allowable blanket DPA. `.costs.life_dpa`.
    dpa_fpy :
        DPA per FPY. `calculate_dpa_per_fpy`'s output.
    life_plant :
        Total plant lifetime (years). `.costs.life_plant`.

    Returns
    -------
    :
        `life_blkt_fpy`.
    """
    # `life_fw_fpy < 0.0001` is a genuine data-dependent condition (not a switch), so it
    # is `jnp.where`, not a Python `if` -- `ibkt_life` *is* static, so it is a Python
    # `if`, picked first so only the formula it actually needs is ever built (avoids an
    # unguarded division in the unused branch poisoning value/gradient of the used one,
    # e.g. `life_dpa / dpa_fpy` when `dpa_fpy == 0` and `ibkt_life == 0` is selected).
    unset = life_fw_fpy < 0.0001

    if ibkt_life == 0:
        # Only the "unset" sub-branch guards `pflux_fw_neutron_mw == 0.0` in the
        # source -- the "not unset" (`elif`) sub-branch does not. Kept unguarded here
        # to match; see the audit record's PROCESS-bug note.
        no_flux = pflux_fw_neutron_mw == 0.0
        safe_flux = jnp.where(no_flux, 1.0, pflux_fw_neutron_mw)
        life_if_unset = jnp.where(
            no_flux, life_plant, jnp.minimum(abktflnc / safe_flux, life_plant)
        )
        life_if_set = jnp.minimum(
            jnp.minimum(life_fw_fpy, abktflnc / pflux_fw_neutron_mw), life_plant
        )
        return jnp.where(unset, life_if_unset, life_if_set)

    demo_life = jnp.minimum(life_dpa / dpa_fpy, life_plant)
    return jnp.where(unset, demo_life, jnp.minimum(life_fw_fpy, demo_life))


def calculate_ward_taylor_availability(
    life_div_fpy,
    life_blkt_fpy,
    t_div_replace_yrs,
    t_blkt_replace_yrs,
    tcomrepl,
    uubop,
    uucd,
    uudiv,
    uufuel,
    uufw,
    uumag,
    uuves,
):
    """Total plant availability, WARD_TAYLOR model (`i_plant_availability == 1`).

    Ports `avail`'s WARD_TAYLOR block (source lines 219-252). Exists as a producer of
    `.costs.f_t_plant_available` **only** when `i_plant_availability == 1` -- for
    USER_INPUT (0), that field has no producer at all (see module docstring).

    Parameters
    ----------
    life_div_fpy, life_blkt_fpy :
        Divertor/blanket lifetimes (FPY). `.costs.life_div_fpy`, `.fwbs.life_blkt_fpy`.
    t_div_replace_yrs, t_blkt_replace_yrs :
        Divertor/blanket replacement times (years). `.costs.*`.
    tcomrepl :
        Combined blanket+divertor replacement time (years). `.costs.tcomrepl`.
    uubop, uucd, uudiv, uufuel, uufw, uumag, uuves :
        Per-subsystem unplanned unavailabilities. `.costs.*`.

    Returns
    -------
    :
        `f_t_plant_available`.
    """
    shorter_is_div = life_div_fpy < life_blkt_fpy
    ld = jnp.where(shorter_is_div, life_div_fpy, life_blkt_fpy)
    lb = jnp.where(shorter_is_div, life_blkt_fpy, life_div_fpy)
    td = jnp.where(shorter_is_div, t_div_replace_yrs, t_blkt_replace_yrs)

    n = jnp.ceil(lb / ld) - 1.0
    uplanned = (n * td + tcomrepl) / ((n + 1.0) * ld + (n * td + tcomrepl))

    uutot = uubop
    uutot = uutot + (1.0 - uutot) * uucd
    uutot = uutot + (1.0 - uutot) * uudiv
    uutot = uutot + (1.0 - uutot) * uufuel
    uutot = uutot + (1.0 - uutot) * uufw
    uutot = uutot + (1.0 - uutot) * uumag
    uutot = uutot + (1.0 - uutot) * uuves

    return 1.0 - (uplanned + uutot - uplanned * uutot)


def calculate_avail(
    p_fusion_total_mw,
    life_fw_fpy,
    abktflnc,
    pflux_fw_neutron_mw,
    life_dpa,
    life_plant,
    pflux_div_heat_load_mw,
    adivflnc,
    t_plant_pulse_total,
    t_plant_pulse_burn,
    f_t_plant_available,
    cplife,
    cplife_in,
    *,
    ibkt_life,
    itart,
):
    """`avail()`'s common tail: lifetimes, `bktcycles`, `cpfact`, lifetime adjustment.

    Ports `avail` (source lines 132-288) excluding the WARD_TAYLOR block
    (`calculate_ward_taylor_availability`, a separate node) and the `ife.ife == 1` early
    exit (out of scope for a stellarator port -- `ife.ife` is never touched by
    `Stellarator.run()`; see the audit record).

    Parameters
    ----------
    p_fusion_total_mw :
        Total fusion power (MW). `.physics.p_fusion_total_mw`.
    life_fw_fpy :
        Pre-computed first-wall lifetime (FPY), if known. `.fwbs.life_fw_fpy`.
    abktflnc, pflux_fw_neutron_mw, life_dpa, life_plant :
        See `calculate_blanket_lifetime_fpy_avail`.
    pflux_div_heat_load_mw, adivflnc :
        See `calculate_divertor_lifetime` (the source clamps
        `pflux_div_heat_load_mw` to a `1e-10` floor first).
    t_plant_pulse_total, t_plant_pulse_burn :
        Pulse length / burn time (s). `.times.*`.
    f_t_plant_available :
        Plant availability fraction -- either the USER_INPUT input value or
        `calculate_ward_taylor_availability`'s output, supplied by the caller.
        `.costs.f_t_plant_available`.
    cplife :
        Centrepost lifetime (FPY), from whichever `calculate_cp_lifetime_*` alternative
        applies. Only used when `itart == 1`. `.costs.cplife`.
    cplife_in :
        The `.costs.cplife` value already on `data`, used unchanged when `itart != 1`
        (the source never computes it in that case -- `conditional-ownership-by-run-config`).
    ibkt_life, itart :
        Static switches. `.costs.ibkt_life`, `.physics.itart`.

    Returns
    -------
    :
        `(life_blkt_fpy, life_div_fpy, cplife, bktcycles, cpfact, life_hcd_fpy)`.
    """
    dpa_fpy = calculate_dpa_per_fpy(p_fusion_total_mw)
    life_blkt_fpy = calculate_blanket_lifetime_fpy_avail(
        life_fw_fpy, ibkt_life, abktflnc, pflux_fw_neutron_mw, life_dpa, dpa_fpy, life_plant
    )
    pflux_div_heat_load_mw_clamped = jnp.maximum(pflux_div_heat_load_mw, 1.0e-10)
    life_div_fpy = calculate_divertor_lifetime(
        adivflnc, pflux_div_heat_load_mw_clamped, life_plant
    )
    cplife_selected = cplife if itart == 1 else cplife_in

    pulse_fpy = t_plant_pulse_total / YEAR_SECONDS
    bktcycles = (life_blkt_fpy / pulse_fpy) + 1.0

    cpfact = f_t_plant_available * (t_plant_pulse_burn / t_plant_pulse_total)

    life_blkt_fpy_mod = jnp.where(
        life_blkt_fpy < life_plant,
        jnp.minimum(life_blkt_fpy / f_t_plant_available, life_plant),
        life_blkt_fpy,
    )
    life_div_fpy_mod = jnp.where(
        life_div_fpy < life_plant,
        jnp.minimum(life_div_fpy / f_t_plant_available, life_plant),
        life_div_fpy,
    )
    if itart == 1:
        cplife_mod = jnp.where(
            cplife_selected < life_plant,
            jnp.minimum(cplife_selected / f_t_plant_available, life_plant),
            cplife_selected,
        )
    else:
        cplife_mod = cplife_selected

    life_hcd_fpy = life_blkt_fpy_mod

    return life_blkt_fpy_mod, life_div_fpy_mod, cplife_mod, bktcycles, cpfact, life_hcd_fpy


# ---------------------------------------------------------------------------
# `avail_2()` (MORRIS, `i_plant_availability == 2`)
# ---------------------------------------------------------------------------


def calculate_blanket_lifetime_fpy_simple(
    ibkt_life, abktflnc, pflux_fw_neutron_mw, life_dpa, dpa_fpy, life_plant
):
    """Blanket lifetime, the two-way branch shared by `calc_u_planned` and `avail_st`.

    Ports `calc_u_planned`'s/`avail_st`'s identical blanket-lifetime block (source lines
    624-632 / 1275-1283). **Unlike** `calculate_blanket_lifetime_fpy_avail`, this does
    *not* guard `pflux_fw_neutron_mw == 0.0` -- neither source copy does either; see the
    audit record's PROCESS-bug note. `ibkt_life` is static.

    Parameters
    ----------
    ibkt_life :
        Blanket-lifetime model switch, **static**. `.costs.ibkt_life`.
    abktflnc, pflux_fw_neutron_mw, life_dpa, life_plant :
        See `calculate_blanket_lifetime_fpy_avail`.
    dpa_fpy :
        DPA per FPY. `calculate_dpa_per_fpy`'s output.

    Returns
    -------
    :
        `life_blkt_fpy`.
    """
    if ibkt_life == 0:
        return jnp.minimum(abktflnc / pflux_fw_neutron_mw, life_plant)
    return jnp.minimum(life_dpa / dpa_fpy, life_plant)


def calculate_u_planned(
    p_fusion_total_mw,
    abktflnc,
    pflux_fw_neutron_mw,
    life_dpa,
    adivflnc,
    pflux_div_heat_load_mw,
    life_plant,
    num_rh_systems,
    *,
    ibkt_life,
):
    """Planned unavailability, `avail_2`'s model (`calc_u_planned`).

    Ports `Availability.calc_u_planned` excluding its `itart`-gated `cplife` write (see
    module docstring -- `calculate_cp_lifetime_*` is wired separately, gated the same way
    as in `calculate_avail`).

    Parameters
    ----------
    p_fusion_total_mw, abktflnc, pflux_fw_neutron_mw, life_dpa, life_plant :
        See `calculate_blanket_lifetime_fpy_simple`.
    adivflnc, pflux_div_heat_load_mw :
        See `calculate_divertor_lifetime`.
    num_rh_systems :
        Number of remote-handling systems. `.costs.num_rh_systems`.
    ibkt_life :
        Static switch. `.costs.ibkt_life`.

    Returns
    -------
    :
        `(u_planned, life_blkt_fpy, life_div_fpy, life_hcd_fpy)`.
    """
    dpa_fpy = calculate_dpa_per_fpy(p_fusion_total_mw)
    life_blkt_fpy = calculate_blanket_lifetime_fpy_simple(
        ibkt_life, abktflnc, pflux_fw_neutron_mw, life_dpa, dpa_fpy, life_plant
    )
    life_div_fpy = calculate_divertor_lifetime(adivflnc, pflux_div_heat_load_mw, life_plant)
    life_hcd_fpy = life_blkt_fpy

    mttr_blanket = (21.0 * num_rh_systems ** (-0.9) + 2.0) / 12.0
    mttr_divertor = 0.7 * mttr_blanket

    shorter_is_div = life_div_fpy < life_blkt_fpy
    lifetime_shortest = jnp.where(shorter_is_div, life_div_fpy, life_blkt_fpy)
    lifetime_longest = jnp.where(shorter_is_div, life_blkt_fpy, life_div_fpy)
    mttr_shortest = jnp.where(shorter_is_div, mttr_divertor, mttr_blanket)

    n = jnp.ceil(lifetime_longest / lifetime_shortest) - 1.0
    u_planned = (n * mttr_shortest + mttr_blanket) / (
        (n + 1.0) * lifetime_shortest + (n * mttr_shortest + mttr_blanket)
    )
    return u_planned, life_blkt_fpy, life_div_fpy, life_hcd_fpy


def calculate_avail_2(
    p_fusion_total_mw,
    abktflnc,
    pflux_fw_neutron_mw,
    life_dpa,
    adivflnc,
    pflux_div_heat_load_mw,
    life_plant,
    num_rh_systems,
    temp_tf_superconductor_margin_min,
    temp_cs_superconductor_margin_min,
    conf_mag,
    temp_margin,
    div_prob_fail,
    div_umain_time,
    div_nu,
    div_nref,
    fwbs_prob_fail,
    fwbs_umain_time,
    fwbs_nu,
    fwbs_nref,
    n_vac_pumps_high,
    redun_vac,
    t_plant_pulse_burn,
    t_plant_pulse_total,
    cplife,
    cplife_in,
    *,
    ibkt_life,
    itart,
):
    """Total plant availability, MORRIS model (`avail_2`, `i_plant_availability == 2`).

    Ports `Availability.avail_2` in full, composed from `calculate_u_planned` and the
    `calculate_u_unplanned_*` helpers. `n_vac_pumps_high`/`redun_vac` are static (see
    `calculate_u_unplanned_vacuum`); `ibkt_life`/`itart` are static switches.

    Parameters
    ----------
    (see `calculate_u_planned`, `calculate_u_unplanned_magnets`,
    `calculate_u_unplanned_divertor`, `calculate_u_unplanned_fwbs`,
    `calculate_u_unplanned_vacuum` for the individual fields)
    t_plant_pulse_burn, t_plant_pulse_total :
        Pulse burn time / total length (s). `.times.*`.
    cplife, cplife_in :
        See `calculate_avail`.

    Returns
    -------
    :
        `(life_blkt_fpy, life_div_fpy, life_hcd_fpy, cplife, t_plant_operational_total_yrs,
        u_planned, u_unplanned, f_t_plant_available, cpfact)`.
    """
    u_planned, life_blkt_fpy, life_div_fpy, life_hcd_fpy = calculate_u_planned(
        p_fusion_total_mw,
        abktflnc,
        pflux_fw_neutron_mw,
        life_dpa,
        adivflnc,
        pflux_div_heat_load_mw,
        life_plant,
        num_rh_systems,
        ibkt_life=ibkt_life,
    )
    t_plant_operational_total_yrs = life_plant * (1.0 - u_planned)

    u_unplanned_magnets = calculate_u_unplanned_magnets(
        temp_tf_superconductor_margin_min,
        temp_cs_superconductor_margin_min,
        t_plant_operational_total_yrs,
        conf_mag,
        temp_margin,
    )
    u_unplanned_div = calculate_u_unplanned_divertor(
        life_div_fpy, t_plant_pulse_total, div_prob_fail, div_umain_time, div_nu, div_nref
    )
    u_unplanned_fwbs = calculate_u_unplanned_fwbs(
        life_blkt_fpy, t_plant_pulse_total, fwbs_prob_fail, fwbs_umain_time, fwbs_nu, fwbs_nref
    )
    u_unplanned_bop = calculate_u_unplanned_bop(t_plant_operational_total_yrs)
    u_unplanned_hcd = calculate_u_unplanned_hcd()
    u_unplanned_vacuum = calculate_u_unplanned_vacuum(
        t_plant_operational_total_yrs, life_plant, num_rh_systems, n_vac_pumps_high, redun_vac
    )

    u_unplanned = jnp.minimum(
        1.0,
        u_unplanned_magnets
        + u_unplanned_div
        + u_unplanned_fwbs
        + u_unplanned_bop
        + u_unplanned_hcd
        + u_unplanned_vacuum,
    )

    f_t_plant_available = jnp.maximum(
        1.0 - (u_planned + u_unplanned + u_planned * u_unplanned), 0.0
    )

    life_blkt_fpy_mod = jnp.where(
        life_blkt_fpy < life_plant,
        jnp.minimum(life_blkt_fpy / f_t_plant_available, life_plant),
        life_blkt_fpy,
    )
    life_hcd_fpy_mod = jnp.where(life_blkt_fpy < life_plant, life_blkt_fpy_mod, life_hcd_fpy)
    life_div_fpy_mod = jnp.where(
        life_div_fpy < life_plant,
        jnp.minimum(life_div_fpy / f_t_plant_available, life_plant),
        life_div_fpy,
    )
    if itart == 1:
        cplife_mod = jnp.where(
            cplife < life_plant,
            jnp.minimum(cplife / f_t_plant_available, life_plant),
            cplife,
        )
    else:
        cplife_mod = cplife_in

    cpfact = f_t_plant_available * (t_plant_pulse_burn / t_plant_pulse_total)

    return (
        life_blkt_fpy_mod,
        life_div_fpy_mod,
        life_hcd_fpy_mod,
        cplife_mod,
        t_plant_operational_total_yrs,
        u_planned,
        u_unplanned,
        f_t_plant_available,
        cpfact,
    )


# ---------------------------------------------------------------------------
# `avail_st()` (ST, `i_plant_availability == 3`)
# ---------------------------------------------------------------------------


def calculate_avail_st(
    abktflnc,
    pflux_fw_neutron_mw,
    life_dpa,
    p_fusion_total_mw,
    adivflnc,
    pflux_div_heat_load_mw,
    life_plant,
    cplife,
    tmain,
    temp_tf_superconductor_margin_min,
    temp_cs_superconductor_margin_min,
    conf_mag,
    temp_margin,
    div_prob_fail,
    div_umain_time,
    div_nu,
    div_nref,
    fwbs_prob_fail,
    fwbs_umain_time,
    fwbs_nu,
    fwbs_nref,
    num_rh_systems,
    n_vac_pumps_high,
    redun_vac,
    u_unplanned_cp,
    t_plant_pulse_burn,
    t_plant_pulse_total,
    *,
    ibkt_life,
    itart,
):
    """Total plant availability, ST model (`avail_st`, `i_plant_availability == 3`).

    Ports `Availability.avail_st` in full. Reachable on the stellarator pipeline only via
    `Stellarator.output()`'s final report-writing call (`Availability.run(output=True)`)
    -- never during the solve loop, which always calls `avail()` directly. Requires
    `.physics.itart == 1` (checked by `Availability.run`, not by this function) but
    nothing in PROCESS's own input validation forbids a stellarator `IN.DAT` from setting
    `itart = 1`; see the audit record's "open questions" for the full trace.

    Unlike `avail`/`avail_2`, `.costs.cplife` is computed **unconditionally** here (the
    source's own `itart` gate only applies to the later lifetime-*adjustment* step, not
    to the initial `cp_lifetime()` call) -- `cplife` is supplied by the caller (from
    whichever `calculate_cp_lifetime_*` alternative applies) rather than gated by `itart`
    the way `calculate_avail`/`calculate_avail_2` gate it.

    Parameters
    ----------
    (see `calculate_blanket_lifetime_fpy_simple`, `calculate_divertor_lifetime`,
    `calculate_u_unplanned_*` for the shared fields)
    cplife :
        Centrepost lifetime (FPY), from `calculate_cp_lifetime_superconducting`/
        `_resistive`. `.costs.cplife`.
    tmain :
        Maintenance time for replacing the centrepost (years). `.costs.tmain`.
    u_unplanned_cp :
        Centrepost unplanned unavailability -- read only, not computed by this unit
        (produced elsewhere; out of scope). `.costs.u_unplanned_cp`.
    itart :
        Static switch, gates only the lifetime-adjustment step (see above).
        `.physics.itart`.

    Returns
    -------
    :
        `(life_blkt_fpy, life_div_fpy, life_hcd_fpy, cplife, maint_cycle, n_cycles_main,
        n_centre_cols, u_planned, t_plant_operational_total_yrs, u_unplanned,
        f_t_plant_available, cpfact)`.
    """
    dpa_fpy = calculate_dpa_per_fpy(p_fusion_total_mw)
    life_blkt_fpy = calculate_blanket_lifetime_fpy_simple(
        ibkt_life, abktflnc, pflux_fw_neutron_mw, life_dpa, dpa_fpy, life_plant
    )
    life_div_fpy = calculate_divertor_lifetime(adivflnc, pflux_div_heat_load_mw, life_plant)
    life_hcd_fpy = life_blkt_fpy

    shortest_lifetime = jnp.minimum(
        jnp.minimum(jnp.minimum(life_blkt_fpy, life_div_fpy), cplife),
        jnp.minimum(life_hcd_fpy, life_plant),
    )
    maint_cycle = shortest_lifetime + tmain
    n_cycles_main = life_plant / maint_cycle
    n_centre_cols = jnp.ceil(n_cycles_main)

    u_planned = tmain / maint_cycle
    t_plant_operational_total_yrs = life_plant * (1.0 - u_planned)

    u_unplanned_magnets = calculate_u_unplanned_magnets(
        temp_tf_superconductor_margin_min,
        temp_cs_superconductor_margin_min,
        t_plant_operational_total_yrs,
        conf_mag,
        temp_margin,
    )
    u_unplanned_div = calculate_u_unplanned_divertor(
        life_div_fpy, t_plant_pulse_total, div_prob_fail, div_umain_time, div_nu, div_nref
    )
    u_unplanned_fwbs = calculate_u_unplanned_fwbs(
        life_blkt_fpy, t_plant_pulse_total, fwbs_prob_fail, fwbs_umain_time, fwbs_nu, fwbs_nref
    )
    u_unplanned_bop = calculate_u_unplanned_bop(t_plant_operational_total_yrs)
    u_unplanned_hcd = calculate_u_unplanned_hcd()
    u_unplanned_vacuum = calculate_u_unplanned_vacuum(
        t_plant_operational_total_yrs, life_plant, num_rh_systems, n_vac_pumps_high, redun_vac
    )

    u_unplanned = jnp.minimum(
        1.0,
        u_unplanned_magnets
        + u_unplanned_div
        + u_unplanned_fwbs
        + u_unplanned_bop
        + u_unplanned_hcd
        + u_unplanned_vacuum
        + u_unplanned_cp,
    )

    f_t_plant_available = jnp.maximum(
        1.0 - (u_planned + u_unplanned + u_planned * u_unplanned), 0.0
    )

    life_blkt_fpy_mod = jnp.where(
        life_blkt_fpy < life_plant,
        jnp.minimum(life_blkt_fpy / f_t_plant_available, life_plant),
        life_blkt_fpy,
    )
    life_hcd_fpy_mod = jnp.where(life_blkt_fpy < life_plant, life_blkt_fpy_mod, life_hcd_fpy)
    life_div_fpy_mod = jnp.where(
        life_div_fpy < life_plant,
        jnp.minimum(life_div_fpy / f_t_plant_available, life_plant),
        life_div_fpy,
    )
    if itart == 1:
        cplife_mod = jnp.where(
            cplife < life_plant,
            jnp.minimum(cplife / f_t_plant_available, life_plant),
            cplife,
        )
    else:
        cplife_mod = cplife

    cpfact = f_t_plant_available * (t_plant_pulse_burn / t_plant_pulse_total)

    return (
        life_blkt_fpy_mod,
        life_div_fpy_mod,
        life_hcd_fpy_mod,
        cplife_mod,
        maint_cycle,
        n_cycles_main,
        n_centre_cols,
        u_planned,
        t_plant_operational_total_yrs,
        u_unplanned,
        f_t_plant_available,
        cpfact,
    )


# ---------------------------------------------------------------------------
# cottax nodes
#
# Only the leaf/composite functions whose *entire* return tuple maps onto real PROCESS
# storage get a node here -- a `NodeDefinition` must own at least one variable
# (`~/jaxgraph/CLAUDE.md`: "a node is a thing that mints variables"), and several of the
# functions above return one or more values with no `VarPath` at all (`u_planned`,
# `u_unplanned`, `n_cycles_main`, `n_centre_cols`, `maint_cycle` -- the source keeps these
# as local variables, never writing them to `data`; see the audit record). Those stay
# plain composable Python functions, used internally by `Avail`/`Avail2`/`AvailSt`'s
# `__call__` and independently tier-1-tested, but are not wrapped as standalone nodes.
#
# `Avail`/`Avail2`/`AvailSt` are **one node per branch**, matching PROCESS's own
# granularity (one `Model` method call producing every output at once) rather than
# atomising further -- nothing outside `Availability` ever calls `divertor_lifetime`,
# `calc_u_planned` etc. independently, so a graph with one node per branch is the
# faithful shape, not an arbitrary choice. `CpLifetimeSuperconducting`/
# `CpLifetimeResistive` are a *different* kind of exception: `.costs.cplife` genuinely
# has two independent producers selected by `.tfcoil.i_tf_sup`, exactly the `i_tf_sup`
# shape already used in `stellarator_F_tf_nuclear_heating.py`.
#
# `.costs.cplife` **also** self-references within `avail`/`avail_2`/`avail_st` themselves
# (Shape B, `next_steps.md` §5: a node whose own `Output` and `FromExactly` name the identical
# `VarPath`) -- `to_graph(Avail(...))` raised `ValueError: reads ['.costs.cplife'], which
# it also owns` directly from `cottax.spec`'s `__check_init__` before this was split.
# `CplifeAvail` (shared by `Avail`/`Avail2` -- their `itart == 1` cplife-adjustment
# formula is identical, see the audit record) and `CplifeAvailSt` isolate exactly that
# self-reference as `FixedPointFunction` declarations, per `next_steps.md` §5's Action.
# `Avail`/`Avail2`/`AvailSt` themselves are now ordinary `ExplicitFunction`s over each
# branch's *other* outputs only -- `.costs.cplife` is no longer one of their declared
# `Output`s. `CpLifetimeSuperconducting`/`CpLifetimeResistive` are left unconsumed by
# this split (their `i_tf_sup` branch is duplicated inline inside `CplifeAvail`/
# `CplifeAvailSt` instead, as a static Python `if` -- see those classes' docstrings for
# why): both those nodes and the new `FixedPoint` problem nodes independently want to
# own `.costs.cplife`, and only one may in any graph that actually registers them
# together -- an open question left to whoever designs `total_process.py`'s wiring, not
# resolved here (registration is explicitly out of this split's scope).
# ---------------------------------------------------------------------------


class CpLifetimeSuperconducting(ExplicitFunction):
    """cottax node: `calculate_cp_lifetime_superconducting`, unchanged, ports declared.

    Mutually exclusive alternative to `CpLifetimeResistive` -- `.tfcoil.i_tf_sup` selects
    at most one at graph-assembly time (same shape as `i_tf_sup` in
    `stellarator_F_tf_nuclear_heating.py`).
    """

    cplife = Output(lambda s: s.costs.cplife)

    def __call__(
        self,
        neut_flux_cp=FromExactly(lambda s: s.fwbs.neut_flux_cp),
        flu_tf_neutron_fast_max=FromExactly(lambda s: s.constraints.flu_tf_neutron_fast_max),
        life_plant=FromExactly(lambda s: s.costs.life_plant),
    ):
        return calculate_cp_lifetime_superconducting(
            neut_flux_cp, flu_tf_neutron_fast_max, life_plant
        )


class CpLifetimeResistive(ExplicitFunction):
    """cottax node: `calculate_cp_lifetime_resistive`, unchanged, ports declared.

    Mutually exclusive alternative to `CpLifetimeSuperconducting`.
    """

    cplife = Output(lambda s: s.costs.cplife)

    def __call__(
        self,
        cpstflnc=FromExactly(lambda s: s.costs.cpstflnc),
        pflux_fw_neutron_mw=FromExactly(lambda s: s.physics.pflux_fw_neutron_mw),
        life_plant=FromExactly(lambda s: s.costs.life_plant),
    ):
        return calculate_cp_lifetime_resistive(cpstflnc, pflux_fw_neutron_mw, life_plant)


class WardTaylorAvailability(ExplicitFunction):
    """cottax node: `calculate_ward_taylor_availability`, unchanged, ports declared.

    Exists **only** when `.costs.i_plant_availability == 1` -- for USER_INPUT (0),
    `.costs.f_t_plant_available` has no producer at all (an ordinary unowned boundary
    input); see module docstring.
    """

    f_t_plant_available = Output(lambda s: s.costs.f_t_plant_available)

    def __call__(
        self,
        life_div_fpy=FromExactly(lambda s: s.costs.life_div_fpy),
        life_blkt_fpy=FromExactly(lambda s: s.fwbs.life_blkt_fpy),
        t_div_replace_yrs=FromExactly(lambda s: s.costs.t_div_replace_yrs),
        t_blkt_replace_yrs=FromExactly(lambda s: s.costs.t_blkt_replace_yrs),
        tcomrepl=FromExactly(lambda s: s.costs.tcomrepl),
        uubop=FromExactly(lambda s: s.costs.uubop),
        uucd=FromExactly(lambda s: s.costs.uucd),
        uudiv=FromExactly(lambda s: s.costs.uudiv),
        uufuel=FromExactly(lambda s: s.costs.uufuel),
        uufw=FromExactly(lambda s: s.costs.uufw),
        uumag=FromExactly(lambda s: s.costs.uumag),
        uuves=FromExactly(lambda s: s.costs.uuves),
    ):
        return calculate_ward_taylor_availability(
            life_div_fpy,
            life_blkt_fpy,
            t_div_replace_yrs,
            t_blkt_replace_yrs,
            tcomrepl,
            uubop,
            uucd,
            uudiv,
            uufuel,
            uufw,
            uumag,
            uuves,
        )


# ---------------------------------------------------------------------------
# `.costs.cplife`'s Shape B self-reference, isolated (`next_steps.md` §5)
#
# Pure step functions first, `FixedPointFunction` wrappers below them -- same pairing as
# every other `calculate_*` -> node in this file. Neither reads `.costs.cplife` for its
# *value* except as the pass-through branch's own current value (`calculate_cplife_next`,
# `itart != 1`); the `itart == 1` recompute never depends on the old value at all (real
# PROCESS's `cp_lifetime()` doesn't read `.costs.cplife` either -- it only writes it).
# ---------------------------------------------------------------------------


def calculate_cplife_lifetime_adjustment(cplife, life_plant, f_t_plant_available):
    """The `itart == 1` lifetime-adjustment step applied to a freshly computed `cplife`.

    Reproduces `avail`'s/`avail_2`'s/`avail_st`'s identical
    `if cplife < life_plant: cplife = min(cplife / f_t_plant_available, life_plant)`
    block. Kept as a free-standing duplicate of the formula already inlined
    (independently, three times) inside `calculate_avail`/`calculate_avail_2`/
    `calculate_avail_st`, rather than factored out and shared with them -- so those three
    functions' bodies (and `test_availability.py`'s existing contracts, which call them
    directly) stay byte-for-byte untouched by this split; see the audit record's "cottax
    node" section.

    Parameters
    ----------
    cplife :
        Centrepost lifetime (FPY) before adjustment.
    life_plant :
        Total plant lifetime (years). `.costs.life_plant`.
    f_t_plant_available :
        Plant availability fraction. `.costs.f_t_plant_available`.

    Returns
    -------
    :
        The adjusted `cplife`.
    """
    return jnp.where(
        cplife < life_plant,
        jnp.minimum(cplife / f_t_plant_available, life_plant),
        cplife,
    )


def calculate_cplife_next(
    cplife,
    neut_flux_cp,
    flu_tf_neutron_fast_max,
    cpstflnc,
    pflux_fw_neutron_mw,
    life_plant,
    f_t_plant_available,
    *,
    i_tf_sup,
    itart,
):
    """`.costs.cplife`'s next value across one `avail()`/`avail_2()` call.

    `CplifeAvail`'s `FixedPointFunction` step. `itart != 1`: real PROCESS never touches
    `.costs.cplife` inside `avail()`/`avail_2()` at all -- a trivial identity fixed point
    on `cplife`, the value already there (`cplife_in`, in `calculate_avail`'s/
    `calculate_avail_2`'s own naming). `itart == 1`: `.costs.cplife` is recomputed from
    scratch via whichever `calculate_cp_lifetime_*` alternative `i_tf_sup` selects (the
    previous value is never read on this branch), then adjusted by
    `calculate_cplife_lifetime_adjustment`. `i_tf_sup`/`itart` are static (switches are
    not ports) -- see `CplifeAvail`'s docstring for why `i_tf_sup`'s branch is duplicated
    here rather than sourced from `CpLifetimeSuperconducting`/`CpLifetimeResistive`.

    Parameters
    ----------
    cplife :
        The current value of `.costs.cplife` (used only when `itart != 1`).
    neut_flux_cp, flu_tf_neutron_fast_max :
        See `calculate_cp_lifetime_superconducting` (used only when `i_tf_sup == 1` and
        `itart == 1`).
    cpstflnc, pflux_fw_neutron_mw :
        See `calculate_cp_lifetime_resistive` (used only when `i_tf_sup != 1` and
        `itart == 1`).
    life_plant :
        Total plant lifetime (years). `.costs.life_plant`.
    f_t_plant_available :
        Plant availability fraction. `.costs.f_t_plant_available`.
    i_tf_sup, itart :
        Static switches. `.tfcoil.i_tf_sup`, `.physics.itart`.

    Returns
    -------
    :
        `.costs.cplife`'s next value.
    """
    if itart != 1:
        return cplife
    if i_tf_sup == 1:
        fresh = calculate_cp_lifetime_superconducting(
            neut_flux_cp, flu_tf_neutron_fast_max, life_plant
        )
    else:
        fresh = calculate_cp_lifetime_resistive(
            cpstflnc, pflux_fw_neutron_mw, life_plant
        )
    return calculate_cplife_lifetime_adjustment(fresh, life_plant, f_t_plant_available)


def calculate_cplife_avail_st_next(
    neut_flux_cp,
    flu_tf_neutron_fast_max,
    cpstflnc,
    pflux_fw_neutron_mw,
    life_plant,
    f_t_plant_available,
    *,
    i_tf_sup,
    itart,
):
    """`.costs.cplife`'s next value across one `avail_st()` call.

    `CplifeAvailSt`'s `FixedPointFunction` step. Unlike `calculate_cplife_next`,
    `avail_st()` recomputes `.costs.cplife` **unconditionally** (the module docstring's
    "the two `itart` gates are not the same gate reused" finding) -- so this function
    never reads a previous `.costs.cplife` value at all; the `itart` gate here only
    selects whether `calculate_cplife_lifetime_adjustment` is applied afterward. The
    `FixedPoint` this feeds converges in exactly one iteration regardless of its starting
    guess -- a degenerate but structurally honest use of the same primitive (see
    `CplifeAvailSt`'s docstring).

    Parameters
    ----------
    (see `calculate_cplife_next` for every parameter except `cplife`, which this function
    has no equivalent of)

    Returns
    -------
    :
        `.costs.cplife`'s next value.
    """
    if i_tf_sup == 1:
        fresh = calculate_cp_lifetime_superconducting(
            neut_flux_cp, flu_tf_neutron_fast_max, life_plant
        )
    else:
        fresh = calculate_cp_lifetime_resistive(
            cpstflnc, pflux_fw_neutron_mw, life_plant
        )
    if itart != 1:
        return fresh
    return calculate_cplife_lifetime_adjustment(fresh, life_plant, f_t_plant_available)


class CplifeAvail(FixedPointFunction):
    """cottax node: `.costs.cplife`'s Shape B self-reference in `Avail`/`Avail2`
    (`next_steps.md` §5), split out as a `FixedPointFunction`. `step` ->
    `calculate_cplife_next`.

    Shared by `Avail` and `Avail2`: both branches' `itart == 1` cplife-adjustment formula
    is identical once `cplife`/`life_plant`/`f_t_plant_available`/`itart` are given
    (confirmed by direct comparison of `calculate_avail`'s and `calculate_avail_2`'s
    `itart == 1` blocks -- see the audit record) -- one shared node, not a duplicate per
    branch, per the task's own preference for this case.

    `i_tf_sup`/`itart` are static (switches are not ports). Note this duplicates
    `i_tf_sup`'s SC/resistive branch *inline* as a Python `if`, rather than consuming
    `CpLifetimeSuperconducting`/`CpLifetimeResistive`'s own node outputs -- unlike that
    pair's top-level split (justified there because `.costs.cplife` had no other owner to
    conflict with), this node's `FixedPoint` problem *also* wants to own `.costs.cplife`,
    and only one producer of one `VarPath` may exist in any graph that registers both
    together. Duplicating the two-line `calculate_cp_lifetime_*` dispatch here avoids
    that conflict entirely rather than resolving it; `CpLifetimeSuperconducting`/
    `CpLifetimeResistive` remain valid, independently useful standalone nodes.
    """

    i_tf_sup: TFConductorModel = eqx.field(static=True)
    itart: SphericalTokamakModel = eqx.field(static=True)

    cplife = Output(lambda s: s.costs.cplife)

    def step(
        self,
        cplife=FromExactly(lambda s: s.costs.cplife),
        neut_flux_cp=FromExactly(lambda s: s.fwbs.neut_flux_cp),
        flu_tf_neutron_fast_max=FromExactly(lambda s: s.constraints.flu_tf_neutron_fast_max),
        cpstflnc=FromExactly(lambda s: s.costs.cpstflnc),
        pflux_fw_neutron_mw=FromExactly(lambda s: s.physics.pflux_fw_neutron_mw),
        life_plant=FromExactly(lambda s: s.costs.life_plant),
        f_t_plant_available=FromExactly(lambda s: s.costs.f_t_plant_available),
    ):
        return calculate_cplife_next(
            cplife,
            neut_flux_cp,
            flu_tf_neutron_fast_max,
            cpstflnc,
            pflux_fw_neutron_mw,
            life_plant,
            f_t_plant_available,
            i_tf_sup=self.i_tf_sup,
            itart=self.itart,
        )


class CplifeAvailSt(FixedPointFunction):
    """cottax node: `.costs.cplife`'s Shape B self-reference in `AvailSt`
    (`next_steps.md` §5), split out as a `FixedPointFunction`. `step` ->
    `calculate_cplife_avail_st_next`.

    `avail_st()` computes `.costs.cplife` **unconditionally** -- no `cplife_in`
    pass-through branch exists here, unlike `CplifeAvail` -- so this node's `step`
    ignores whatever the graph currently holds at `.costs.cplife` entirely; its output
    depends only on the genuine recompute inputs below. Still declared as a
    `FixedPointFunction` (not a plain `ExplicitFunction`) for the same structural reason
    as `CplifeAvail`: `AvailSt`'s *other* outputs (`shortest_lifetime` and everything
    downstream of it) need to read `.costs.cplife` too, so whichever node owns it must
    not be the same node -- see `AvailSt`'s docstring for why that read cannot simply be
    `.costs.cplife` fed back in (the value this node owns is the *adjusted* one;
    `avail_st()`'s `shortest_lifetime` needs the pre-adjustment one).

    `i_tf_sup`/`itart` are static -- see `CplifeAvail`'s docstring for why `i_tf_sup`'s
    branch is duplicated here rather than sourced from `CpLifetimeSuperconducting`/
    `CpLifetimeResistive`.
    """

    i_tf_sup: TFConductorModel = eqx.field(static=True)
    itart: SphericalTokamakModel = eqx.field(static=True)

    cplife = Output(lambda s: s.costs.cplife)

    def step(
        self,
        neut_flux_cp=FromExactly(lambda s: s.fwbs.neut_flux_cp),
        flu_tf_neutron_fast_max=FromExactly(lambda s: s.constraints.flu_tf_neutron_fast_max),
        cpstflnc=FromExactly(lambda s: s.costs.cpstflnc),
        pflux_fw_neutron_mw=FromExactly(lambda s: s.physics.pflux_fw_neutron_mw),
        life_plant=FromExactly(lambda s: s.costs.life_plant),
        f_t_plant_available=FromExactly(lambda s: s.costs.f_t_plant_available),
    ):
        return calculate_cplife_avail_st_next(
            neut_flux_cp,
            flu_tf_neutron_fast_max,
            cpstflnc,
            pflux_fw_neutron_mw,
            life_plant,
            f_t_plant_available,
            i_tf_sup=self.i_tf_sup,
            itart=self.itart,
        )


class Avail(ExplicitFunction):
    """cottax node: `calculate_avail`'s outputs *other* than `.costs.cplife`, unchanged,
    ports declared. `.costs.cplife` itself is `CplifeAvail`'s (see that class and the
    module docstring's "cottax nodes" section for why this needed splitting at all --
    Shape B, `next_steps.md` §5).

    `ibkt_life`/`itart` are static -- see module docstring. Mutually exclusive
    alternative to `Avail2`/`AvailSt`: `.costs.i_plant_availability` selects at most one
    of the three branch nodes at graph-assembly time.

    `cplife` is read here as a plain current-value `FromExactly` (`.costs.cplife`, i.e.
    `CplifeAvail`'s output once both are registered together) and passed to
    `calculate_avail` unchanged -- but its value is **provably inert** for every output
    this node declares: inspecting `calculate_avail`'s body shows `cplife`/`cplife_in`
    feed *only* the `cplife_mod` return slot, which this node discards. Kept as a real
    `FromExactly` anyway (matching what a full port of `avail()`'s real read/write order would
    show, and the task's own recipe) rather than a magic constant, even though any value
    would do here -- unlike `AvailSt` below, where the same-looking read is load-bearing.
    """

    ibkt_life: BlanketLifetimeModel = eqx.field(static=True)
    itart: SphericalTokamakModel = eqx.field(static=True)

    life_blkt_fpy = Output(lambda s: s.fwbs.life_blkt_fpy)
    life_div_fpy = Output(lambda s: s.costs.life_div_fpy)
    bktcycles = Output(lambda s: s.costs.bktcycles)
    cpfact = Output(lambda s: s.costs.cpfact)
    life_hcd_fpy = Output(lambda s: s.costs.life_hcd_fpy)

    def __call__(
        self,
        p_fusion_total_mw=FromExactly(lambda s: s.physics.p_fusion_total_mw),
        life_fw_fpy=FromExactly(lambda s: s.fwbs.life_fw_fpy),
        abktflnc=FromExactly(lambda s: s.costs.abktflnc),
        pflux_fw_neutron_mw=FromExactly(lambda s: s.physics.pflux_fw_neutron_mw),
        life_dpa=FromExactly(lambda s: s.costs.life_dpa),
        life_plant=FromExactly(lambda s: s.costs.life_plant),
        pflux_div_heat_load_mw=FromExactly(lambda s: s.divertor.pflux_div_heat_load_mw),
        adivflnc=FromExactly(lambda s: s.costs.adivflnc),
        t_plant_pulse_total=FromExactly(lambda s: s.times.t_plant_pulse_total),
        t_plant_pulse_burn=FromExactly(lambda s: s.times.t_plant_pulse_burn),
        f_t_plant_available=FromExactly(lambda s: s.costs.f_t_plant_available),
        cplife=FromExactly(lambda s: s.costs.cplife),
    ):
        (
            life_blkt_fpy,
            life_div_fpy,
            _cplife_mod,
            bktcycles,
            cpfact,
            life_hcd_fpy,
        ) = calculate_avail(
            p_fusion_total_mw,
            life_fw_fpy,
            abktflnc,
            pflux_fw_neutron_mw,
            life_dpa,
            life_plant,
            pflux_div_heat_load_mw,
            adivflnc,
            t_plant_pulse_total,
            t_plant_pulse_burn,
            f_t_plant_available,
            cplife,
            cplife,
            ibkt_life=self.ibkt_life,
            itart=self.itart,
        )
        return life_blkt_fpy, life_div_fpy, bktcycles, cpfact, life_hcd_fpy


class Avail2(ExplicitFunction):
    """cottax node: `calculate_avail_2`'s outputs *other* than `.costs.cplife`,
    unchanged, ports declared, `u_planned`/`u_unplanned` dropped (no `VarPath` -- see the
    module-level note above). `.costs.cplife` itself is `CplifeAvail`'s -- see that
    class's docstring; `Avail`/`Avail2` share it since their cplife-adjustment formula is
    identical.

    `ibkt_life`/`itart`/`n_vac_pumps_high`/`redun_vac` are static (the last two because
    they set a Python `range()` bound inside `calculate_u_unplanned_vacuum` -- see
    `calculate_redun_vac`'s docstring). Mutually exclusive alternative to `Avail`/
    `AvailSt`.

    `cplife` is read here as a plain current-value `FromExactly`, same provably-inert role as
    in `Avail` -- see that class's docstring (`calculate_avail_2`'s `cplife`/`cplife_in`
    also feed only the discarded `cplife_mod` slot; verified the same way).
    """

    ibkt_life: BlanketLifetimeModel = eqx.field(static=True)
    itart: SphericalTokamakModel = eqx.field(static=True)
    n_vac_pumps_high: int = eqx.field(static=True)
    redun_vac: int = eqx.field(static=True)

    life_blkt_fpy = Output(lambda s: s.fwbs.life_blkt_fpy)
    life_div_fpy = Output(lambda s: s.costs.life_div_fpy)
    life_hcd_fpy = Output(lambda s: s.costs.life_hcd_fpy)
    t_plant_operational_total_yrs = Output(
        lambda s: s.costs.t_plant_operational_total_yrs
    )
    f_t_plant_available = Output(lambda s: s.costs.f_t_plant_available)
    cpfact = Output(lambda s: s.costs.cpfact)

    def __call__(
        self,
        p_fusion_total_mw=FromExactly(lambda s: s.physics.p_fusion_total_mw),
        abktflnc=FromExactly(lambda s: s.costs.abktflnc),
        pflux_fw_neutron_mw=FromExactly(lambda s: s.physics.pflux_fw_neutron_mw),
        life_dpa=FromExactly(lambda s: s.costs.life_dpa),
        adivflnc=FromExactly(lambda s: s.costs.adivflnc),
        pflux_div_heat_load_mw=FromExactly(lambda s: s.divertor.pflux_div_heat_load_mw),
        life_plant=FromExactly(lambda s: s.costs.life_plant),
        num_rh_systems=FromExactly(lambda s: s.costs.num_rh_systems),
        temp_tf_superconductor_margin_min=FromExactly(
            lambda s: s.tfcoil.temp_tf_superconductor_margin_min
        ),
        temp_cs_superconductor_margin_min=FromExactly(
            lambda s: s.tfcoil.temp_cs_superconductor_margin_min
        ),
        conf_mag=FromExactly(lambda s: s.costs.conf_mag),
        temp_margin=FromExactly(lambda s: s.tfcoil.temp_margin),
        div_prob_fail=FromExactly(lambda s: s.costs.div_prob_fail),
        div_umain_time=FromExactly(lambda s: s.costs.div_umain_time),
        div_nu=FromExactly(lambda s: s.costs.div_nu),
        div_nref=FromExactly(lambda s: s.costs.div_nref),
        fwbs_prob_fail=FromExactly(lambda s: s.costs.fwbs_prob_fail),
        fwbs_umain_time=FromExactly(lambda s: s.costs.fwbs_umain_time),
        fwbs_nu=FromExactly(lambda s: s.costs.fwbs_nu),
        fwbs_nref=FromExactly(lambda s: s.costs.fwbs_nref),
        t_plant_pulse_burn=FromExactly(lambda s: s.times.t_plant_pulse_burn),
        t_plant_pulse_total=FromExactly(lambda s: s.times.t_plant_pulse_total),
        cplife=FromExactly(lambda s: s.costs.cplife),
    ):
        (
            life_blkt_fpy,
            life_div_fpy,
            life_hcd_fpy,
            _cplife_mod,
            t_plant_operational_total_yrs,
            _u_planned,
            _u_unplanned,
            f_t_plant_available,
            cpfact,
        ) = calculate_avail_2(
            p_fusion_total_mw,
            abktflnc,
            pflux_fw_neutron_mw,
            life_dpa,
            adivflnc,
            pflux_div_heat_load_mw,
            life_plant,
            num_rh_systems,
            temp_tf_superconductor_margin_min,
            temp_cs_superconductor_margin_min,
            conf_mag,
            temp_margin,
            div_prob_fail,
            div_umain_time,
            div_nu,
            div_nref,
            fwbs_prob_fail,
            fwbs_umain_time,
            fwbs_nu,
            fwbs_nref,
            self.n_vac_pumps_high,
            self.redun_vac,
            t_plant_pulse_burn,
            t_plant_pulse_total,
            cplife,
            cplife,
            ibkt_life=self.ibkt_life,
            itart=self.itart,
        )
        return (
            life_blkt_fpy,
            life_div_fpy,
            life_hcd_fpy,
            t_plant_operational_total_yrs,
            f_t_plant_available,
            cpfact,
        )


class AvailSt(ExplicitFunction):
    """cottax node: `calculate_avail_st`'s outputs *other* than `.costs.cplife`,
    unchanged, ports declared, `maint_cycle`/`n_cycles_main`/`n_centre_cols`/
    `u_planned`/`u_unplanned` dropped (no `VarPath`). `.costs.cplife` itself is
    `CplifeAvailSt`'s -- see that class's docstring.

    `ibkt_life`/`itart`/`n_vac_pumps_high`/`redun_vac` are static -- see `Avail2`.
    `i_tf_sup` is a **new** static field this split needed (see below). Reachable on the
    stellarator pipeline only via `Stellarator.output()`'s final report-writing call,
    never during the solve loop; see the audit record.

    **Does not read `.costs.cplife` at all -- deliberately, unlike `Avail`/`Avail2`
    above.** `calculate_avail_st`'s `cplife` parameter is the *pre-adjustment* value
    (used for `shortest_lifetime`, hence `maint_cycle`/`u_planned`/
    `t_plant_operational_total_yrs`/every unplanned-unavailability term/
    `f_t_plant_available`/every `*_mod` output this node declares -- genuinely
    load-bearing here, unlike `Avail`/`Avail2`'s provably-inert `cplife`), while
    `.costs.cplife`'s real, persistent value (what `CplifeAvailSt` owns) is the
    *post*-adjustment one -- a different number whenever `itart == 1` and the adjustment
    actually applies (`cplife / f_t_plant_available != cplife` in general). Feeding
    `.costs.cplife` back into this node's own `cplife` argument would silently double
    only *some* of the intended dependency and corrupt every output that flows through
    `shortest_lifetime`. So this node recomputes the same pre-adjustment value
    `CplifeAvailSt` computes, from the same genuine inputs (`neut_flux_cp`/
    `flu_tf_neutron_fast_max`/`cpstflnc`/`pflux_fw_neutron_mw`, `i_tf_sup`-gated) --
    matching `test_availability.py::TestAvailSt`'s own `ported` adapter, which already
    does exactly this (calls `calculate_cp_lifetime_resistive` before
    `calculate_avail_st`). The duplicate recompute is the same trade-off `CplifeAvail`'s
    docstring documents for `i_tf_sup`, not a new one.
    """

    ibkt_life: BlanketLifetimeModel = eqx.field(static=True)
    itart: SphericalTokamakModel = eqx.field(static=True)
    n_vac_pumps_high: int = eqx.field(static=True)
    redun_vac: int = eqx.field(static=True)
    i_tf_sup: TFConductorModel = eqx.field(static=True)

    life_blkt_fpy = Output(lambda s: s.fwbs.life_blkt_fpy)
    life_div_fpy = Output(lambda s: s.costs.life_div_fpy)
    life_hcd_fpy = Output(lambda s: s.costs.life_hcd_fpy)
    t_plant_operational_total_yrs = Output(
        lambda s: s.costs.t_plant_operational_total_yrs
    )
    f_t_plant_available = Output(lambda s: s.costs.f_t_plant_available)
    cpfact = Output(lambda s: s.costs.cpfact)

    def __call__(
        self,
        abktflnc=FromExactly(lambda s: s.costs.abktflnc),
        pflux_fw_neutron_mw=FromExactly(lambda s: s.physics.pflux_fw_neutron_mw),
        life_dpa=FromExactly(lambda s: s.costs.life_dpa),
        p_fusion_total_mw=FromExactly(lambda s: s.physics.p_fusion_total_mw),
        adivflnc=FromExactly(lambda s: s.costs.adivflnc),
        pflux_div_heat_load_mw=FromExactly(lambda s: s.divertor.pflux_div_heat_load_mw),
        life_plant=FromExactly(lambda s: s.costs.life_plant),
        neut_flux_cp=FromExactly(lambda s: s.fwbs.neut_flux_cp),
        flu_tf_neutron_fast_max=FromExactly(lambda s: s.constraints.flu_tf_neutron_fast_max),
        cpstflnc=FromExactly(lambda s: s.costs.cpstflnc),
        tmain=FromExactly(lambda s: s.costs.tmain),
        temp_tf_superconductor_margin_min=FromExactly(
            lambda s: s.tfcoil.temp_tf_superconductor_margin_min
        ),
        temp_cs_superconductor_margin_min=FromExactly(
            lambda s: s.tfcoil.temp_cs_superconductor_margin_min
        ),
        conf_mag=FromExactly(lambda s: s.costs.conf_mag),
        temp_margin=FromExactly(lambda s: s.tfcoil.temp_margin),
        div_prob_fail=FromExactly(lambda s: s.costs.div_prob_fail),
        div_umain_time=FromExactly(lambda s: s.costs.div_umain_time),
        div_nu=FromExactly(lambda s: s.costs.div_nu),
        div_nref=FromExactly(lambda s: s.costs.div_nref),
        fwbs_prob_fail=FromExactly(lambda s: s.costs.fwbs_prob_fail),
        fwbs_umain_time=FromExactly(lambda s: s.costs.fwbs_umain_time),
        fwbs_nu=FromExactly(lambda s: s.costs.fwbs_nu),
        fwbs_nref=FromExactly(lambda s: s.costs.fwbs_nref),
        num_rh_systems=FromExactly(lambda s: s.costs.num_rh_systems),
        u_unplanned_cp=FromExactly(lambda s: s.costs.u_unplanned_cp),
        t_plant_pulse_burn=FromExactly(lambda s: s.times.t_plant_pulse_burn),
        t_plant_pulse_total=FromExactly(lambda s: s.times.t_plant_pulse_total),
    ):
        if self.i_tf_sup == 1:
            cplife = calculate_cp_lifetime_superconducting(
                neut_flux_cp, flu_tf_neutron_fast_max, life_plant
            )
        else:
            cplife = calculate_cp_lifetime_resistive(
                cpstflnc, pflux_fw_neutron_mw, life_plant
            )
        (
            life_blkt_fpy,
            life_div_fpy,
            life_hcd_fpy,
            _cplife_mod,
            _maint_cycle,
            _n_cycles_main,
            _n_centre_cols,
            _u_planned,
            t_plant_operational_total_yrs,
            _u_unplanned,
            f_t_plant_available,
            cpfact,
        ) = calculate_avail_st(
            abktflnc,
            pflux_fw_neutron_mw,
            life_dpa,
            p_fusion_total_mw,
            adivflnc,
            pflux_div_heat_load_mw,
            life_plant,
            cplife,
            tmain,
            temp_tf_superconductor_margin_min,
            temp_cs_superconductor_margin_min,
            conf_mag,
            temp_margin,
            div_prob_fail,
            div_umain_time,
            div_nu,
            div_nref,
            fwbs_prob_fail,
            fwbs_umain_time,
            fwbs_nu,
            fwbs_nref,
            num_rh_systems,
            self.n_vac_pumps_high,
            self.redun_vac,
            u_unplanned_cp,
            t_plant_pulse_burn,
            t_plant_pulse_total,
            ibkt_life=self.ibkt_life,
            itart=self.itart,
        )
        return (
            life_blkt_fpy,
            life_div_fpy,
            life_hcd_fpy,
            t_plant_operational_total_yrs,
            f_t_plant_available,
            cpfact,
        )
