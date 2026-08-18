"""Pure-functional port of `process/models/vacuum.py` (registry unit #16).

Audit record: `functional_process/models/vacuum.md`. Entry point is `Vacuum.run()`,
which dispatches on the topology-changing switch `.vacuum.i_vacuum_pumping`
(`"old"`/`"simple"`) to one of two, essentially disjoint, computations:

- **`"simple"`** -- `vacuum_simple`: straight-line algebra, no iteration.
  `calculate_vacuum_pumping_simple` below, tier-1.
- **`"old"`** -- `vacuum`: the ETR-derived detailed model. Straight-line algebra to
  build four required pumping speeds, then a genuine internal solve (Newton's method
  for a duct diameter, wrapped in an outer loop that shrinks the target conductance
  until the duct physically fits between TF coils) to size the pumping ducts. Tier-2.
  `calculate_vacuum_pumping_old` below.

`VacuumVessel` (the second class in the source file) is **out of scope**: it is not
reached from `Stellarator.run()` at all. `Stellarator.__init__`
(`process/models/stellarator/stellarator.py`) is injected a `vacuum: Vacuum` but no
`vacuum_vessel` -- confirmed by `process/main.py:668-669,729,783-784`
(`Models.__init__` constructs both `self.vacuum`/`self.vacuum_vessel` and calls
`self.vacuum_vessel.output()` only from the tokamak/general `main.py` output path,
never from `stellarator.py`). The
stellarator pipeline computes its own vacuum-vessel geometry inline
(`Stellarator.st_fwbs`'s "S5 cryostat_and_vv_geometry" chunk, see
`stellarator_E_fwbs_synthesis.md`) instead of calling `VacuumVessel`. See `vacuum.md` for
the full trace.
"""

import jax
import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, Input, Output

# Gas-species ordering used throughout `vacuum()`: (N2, D-T, He, D-T again).
# `process/models/vacuum.py`'s module-level `xmult` list -- conductance-to-nitrogen
# multiplier per species, a fixed physical-constant table, not a `data` field.
XMULT = (1.0, 0.423, 0.378, 0.423)

# Per-pump nominal speed (m^3/s) for the two `i_vacuum_pump_type` values, same ordering
# as `XMULT`. `process/models/vacuum.py::vacuum`'s `sp` literal.
_SP_TURBOMOLECULAR = (1.95, 1.8, 1.8, 1.8)
_SP_COMPOUND_CRYOPUMP = (9.0, 25.0, 5.0, 25.0)

_BOLTZMANN_CONSTANT = 1.38e-23  # J/K, `vacuum()`'s local `k`.
_DUCT_SHIELD_DENSITY = 7900.0  # kg/m^3, `densh`.
_DUCT_SHIELD_SOLID_FRACTION = 0.9  # `fsolid`.
_UMASS = 1.660538921e-27  # kg, `process.core.constants.UMASS` (atomic mass unit).


def calculate_vacuum_pumping_simple(
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
):
    """`Vacuum.vacuum_simple`: combined pump count for the simple steady-state model.

    PROCESS combines an ITER-cryopump-throughput estimate (`n_iter_vacuum_pumps`, a
    local) with a pump-down-time estimate (`npumpdown`, also a local) by taking the max
    ("Combine the two (somewhat inconsistent) models" per the source comment) -- this
    port returns only that combined value, dropping the two intermediates from the
    return the same way `divertor.py` drops its own reporting-only locals: neither
    `n_iter_vacuum_pumps` nor `npumpdown` is written to `data` anywhere (both exist
    only to be displayed by `_vacuum_simple_output`, out of this port's scope, and as
    inputs to this one `max`) -- see `vacuum.md`.

    The returned value is what `Vacuum.run()` writes to `.vacuum.n_iter_vacuum_pumps` --
    note the name collision: the *field* `n_iter_vacuum_pumps` ends up holding this
    combined value, not PROCESS's same-named intermediate *local*; not fixed, just
    flagged (see `vacuum.md`).

    Parameters
    ----------
    molflow_plasma_fuelling_required :
        Plasma fuelling rate (nucleus-pairs/s).
        `.physics.molflow_plasma_fuelling_required`.
    molflow_vac_pumps :
        Pump throughput (molecules/s). `.vacuum.molflow_vac_pumps`.
    volflow_vac_pumps_max :
        Maximum pumping speed per unit area for D-T, molecular flow.
        `.vacuum.volflow_vac_pumps_max`.
    f_a_vac_pump_port_plasma_surface :
        Pumping-port area as a fraction of plasma surface area.
        `.vacuum.f_a_vac_pump_port_plasma_surface`.
    f_volflow_vac_pumps_impedance :
        Effective pumping-speed reduction factor due to duct impedance.
        `.vacuum.f_volflow_vac_pumps_impedance`.
    a_plasma_surface :
        Plasma surface area (m2). `.physics.a_plasma_surface`.
    n_tf_coils :
        Number of TF coils. `.tfcoil.n_tf_coils`.
    outgasfactor :
        Outgassing prefactor (Pa m s-1). `.vacuum.outgasfactor`.
    pres_vv_chamber_base :
        Base pressure during dwell (Pa). `.vacuum.pres_vv_chamber_base`.
    outgasindex :
        Outgassing decay index. `.vacuum.outgasindex`.
    t_plant_pulse_dwell :
        Dwell time between pulses (s). `.times.t_plant_pulse_dwell`.

    Returns
    -------
    :
        `npump`, the combined pump count (`.vacuum.n_iter_vacuum_pumps` once written).
    """
    n_iter_vacuum_pumps = molflow_plasma_fuelling_required / molflow_vac_pumps

    pumpspeed = (
        volflow_vac_pumps_max
        * f_a_vac_pump_port_plasma_surface
        * f_volflow_vac_pumps_impedance
        * a_plasma_surface
        / n_tf_coils
    )

    wallarea = (a_plasma_surface / 1084.0) * 2000.0
    pumpdownspeed = (
        outgasfactor * wallarea / pres_vv_chamber_base
    ) * t_plant_pulse_dwell ** (-outgasindex)
    npumpdown = pumpdownspeed / pumpspeed

    return jnp.maximum(n_iter_vacuum_pumps, npumpdown)


class VacuumPumpingSimple(ExplicitFunction):
    """cottax node: `calculate_vacuum_pumping_simple`'s combined pump count.

    Written back to `data` as `.vacuum.n_iter_vacuum_pumps`, per `Vacuum.run()`'s
    `"simple"` branch (`vp.n_iter_vacuum_pumps = self.vacuum_simple(output=output)`).
    """

    n_iter_vacuum_pumps = Output(lambda s: s.vacuum.n_iter_vacuum_pumps)

    def __call__(
        self,
        molflow_plasma_fuelling_required=Input(
            lambda s: s.physics.molflow_plasma_fuelling_required
        ),
        molflow_vac_pumps=Input(lambda s: s.vacuum.molflow_vac_pumps),
        volflow_vac_pumps_max=Input(lambda s: s.vacuum.volflow_vac_pumps_max),
        f_a_vac_pump_port_plasma_surface=Input(
            lambda s: s.vacuum.f_a_vac_pump_port_plasma_surface
        ),
        f_volflow_vac_pumps_impedance=Input(
            lambda s: s.vacuum.f_volflow_vac_pumps_impedance
        ),
        a_plasma_surface=Input(lambda s: s.physics.a_plasma_surface),
        n_tf_coils=Input(lambda s: s.tfcoil.n_tf_coils),
        outgasfactor=Input(lambda s: s.vacuum.outgasfactor),
        pres_vv_chamber_base=Input(lambda s: s.vacuum.pres_vv_chamber_base),
        outgasindex=Input(lambda s: s.vacuum.outgasindex),
        t_plant_pulse_dwell=Input(lambda s: s.times.t_plant_pulse_dwell),
    ):
        npump = calculate_vacuum_pumping_simple(
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
        return (npump,)


# ---------------------------------------------------------------------------
# `"old"` branch: `Vacuum.vacuum` (ETR-derived detailed duct-sizing model).
# ---------------------------------------------------------------------------
#
# Genuine internal solve, tier-2. PROCESS's own version is a Newton-Raphson duct-
# diameter search (fixed 100-iteration cap, early `break` on convergence), itself
# wrapped in an outer loop that shrinks the required conductance by 10% steps whenever
# the resulting duct doesn't physically fit in the space between adjacent TF coils
# (`a1 < a1max`), until it fits or the shrunk conductance drops below the actual
# required speed (`nflag = 1`, "space limited", logged but not raised). Both loops are
# data-dependent in their iteration count -- no faithful `jax`-traceable translation as
# fixed-count Python loops -- so this port uses `jax.lax.while_loop` throughout instead
# (see `vacuum.md`'s JAX-difficulty flags for why that's fine here: `Tier2Contract`
# never differentiates the port, so `while_loop`'s lack of autodiff support is not a
# constraint).


def duct_conductance(diameter, l1, l2, l3, xmult_i):
    """Effective conductance (m^3/s) of a duct of the given diameter.

    Three conductance elements in series -- aperture, first duct run (length `l1`,
    diameter `diameter`), and two further runs (`l2`, `l3`, diameter `1.2*diameter`,
    matching `vacuum()`'s `dout = 1.2*d` upsizing) -- each a Clausing-type formula
    (`119 * area * transmission_factor / xmult_i`). Ports the closed-form half of
    `Vacuum._newton_function` (`process/models/vacuum.py:504-526`): the `cap`/`c1`/`c2`/
    `c3`/`cnew` expressions, without the hand-differentiated `dy`/Newton-step half --
    `solve_duct_diameter` below drives this with `jax.grad` instead (see its
    docstring).

    Parameters
    ----------
    diameter :
        Duct diameter, `d` (m).
    l1, l2, l3 :
        Duct segment lengths (m) -- divertor-to-duct passage, duct-to-elbow,
        elbow-to-pumps.
    xmult_i :
        This gas species' conductance-to-nitrogen multiplier (`XMULT[i]`).

    Returns
    -------
    :
        Effective conductance (m^3/s).
    """
    a1 = 0.25 * jnp.pi * diameter * diameter
    a2 = 1.44 * a1
    a3 = a2
    k1 = (4.0 / 3.0) * diameter / (l1 + (4.0 / 3.0) * diameter)
    k2 = (4.0 / 3.0) * diameter * 1.2 / (l2 + (4.0 / 3.0) * diameter * 1.2)
    k3 = (4.0 / 3.0) * diameter * 1.2 / (l3 + (4.0 / 3.0) * diameter * 1.2)
    cap = 119.0 * a1 / xmult_i
    c1 = 119.0 * a1 * k1 / xmult_i
    c2 = 119.0 * a2 * k2 / xmult_i
    c3 = 119.0 * a3 * k3 / xmult_i
    return 1.0 / (1.0 / cap + 1.0 / c1 + 1.0 / c2 + 1.0 / c3)


def duct_diameter_residual(diameter, l1, l2, l3, xmult_i, ceff_i):
    """`duct_conductance(diameter, ...) - ceff_i` -- vanishes at the target diameter.

    The defining equation `solve_duct_diameter` solves and
    `TestSolveDuctDiameter.residual` (`test_vacuum.py`) checks, same shape as
    `coils.py`'s `intersect_residual`.
    """
    return duct_conductance(diameter, l1, l2, l3, xmult_i) - ceff_i


def solve_duct_diameter(l1, l2, l3, xmult_i, ceff_i, max_iter=100, tol=1e-10):
    """Duct diameter whose conductance equals `ceff_i`, found by Newton's method.

    Ports `Vacuum._newton_method_duct_diameter`'s inner loop
    (`process/models/vacuum.py:469-484`): fixed start `d = 1.0`, up to `max_iter`
    Newton steps, early stop once the relative step `|d - d_new| / d <= tol`. PROCESS
    hand-derives the Newton step's denominator (`dy` in `_newton_function`); this port
    uses `jax.grad(duct_diameter_residual)` for the same derivative instead -- exact
    for the same reason `coils.py`'s `_intersect_newton_polish` uses `jax.grad` rather
    than a by-hand formula: it is differentiating the *same* closed-form expression, so
    the two agree to float64 round-off, not merely approximately.

    **`tol` defaults to `1e-10`, not PROCESS's own `0.01`.** `test_harness.md`'s tier-2
    section expects a "properly convergent driver landing somewhere numerically
    different from PROCESS's heuristic endpoint" -- PROCESS's `0.01` is a coarse 1%
    relative-step cutoff, not a considered accuracy target, and Newton's quadratic
    convergence means tightening to `1e-10` is nearly free once already within 1%: a
    300-point fuzz sweep across stellarator-scale `(l1, l2, l3, xmult_i, ceff_i)` never
    needed more than 10 of the 100 available iterations to reach an absolute residual
    below `5e-13` (see `vacuum.md`). Keeping PROCESS's own `0.01` is still available by
    passing `tol=0.01` explicitly, if a value-level comparison against PROCESS's own
    endpoint is ever wanted.

    Implemented as `jax.lax.while_loop` rather than a Python `for`/`break`: the
    iteration count is data-dependent (JAX has no early-exit `break`), and
    `Tier2Contract` never differentiates `ported` (see `_harness/contracts.py`), so
    `while_loop`'s lack of autodiff support costs nothing here.

    Does not reproduce PROCESS's `logger.error` on non-convergence (100 iterations
    exhausted without reaching `tol`) -- a traced function cannot log conditionally;
    the loop simply stops at whatever `d` it reached, same as PROCESS's own math does
    (the log call has no effect on `d` either).

    Parameters
    ----------
    l1, l2, l3 :
        Duct segment lengths (m).
    xmult_i :
        This gas species' conductance-to-nitrogen multiplier.
    ceff_i :
        Target effective conductance (m^3/s).
    max_iter :
        Newton-step cap, matching PROCESS's hardcoded 100.
    tol :
        Relative-step convergence tolerance, matching PROCESS's hardcoded 0.01.

    Returns
    -------
    :
        Converged duct diameter `d` (m).
    """

    def cond(carry):
        _d, step, it = carry
        return jnp.logical_and(it < max_iter, step > tol)

    def body(carry):
        d, _step, it = carry
        residual_fn = lambda x: duct_diameter_residual(x, l1, l2, l3, xmult_i, ceff_i)  # noqa: E731
        f = residual_fn(d)
        df = jax.grad(residual_fn)(d)
        d_new = d - f / df
        step = jnp.abs((d - d_new) / d)
        return (d_new, step, it + 1)

    init = (jnp.asarray(1.0), jnp.asarray(jnp.inf), jnp.asarray(0))
    d, _step, _it = jax.lax.while_loop(cond, body, init)
    return d


def solve_duct_geometry(l1, l2, l3, xmult_i, ceff_i_init, a1max, s_i, max_outer=64):
    """Fit a duct to the available inter-coil area, shrinking `ceff_i` if it doesn't.

    Ports `Vacuum._newton_method_duct_diameter`'s outer loop
    (`process/models/vacuum.py:460-501`): re-solve `solve_duct_diameter` (always from a
    fresh `d = 1.0` guess, matching PROCESS's `d[i] = 1.0` reset every outer pass) at
    the current target conductance; if the resulting duct's cross-sectional area
    (`0.25*pi*d**2`) fits in the space between adjacent TF coils (`< a1max`), stop.
    Otherwise shrink the target conductance by 10% and try again, unless it has already
    dropped to within 10% of the actually-required speed `s_i`, in which case give up
    ("space limited", `nflag = 1`) and stop anyway.

    **One deliberate deviation from a line-for-line translation**, flagged in
    `vacuum.md`: PROCESS's own `a1` (used for the fits-or-not test) is computed by
    `_newton_function` from the diameter *before* that Newton step's update, not from
    the diameter the loop actually keeps (`d[i] = dnew` happens after `a1` is read) --
    a harmless staleness bounded by the loop's own 1% convergence tolerance, not
    reproduced here; this port computes `a1` from the diameter `solve_duct_diameter`
    actually returns.

    **Returns `ceff_used` alongside `ceff_final`, unlike PROCESS**, which only keeps
    one array (`ceff[i]`, mutated in place). PROCESS's `ceff[i]` sometimes stops
    meaning "the conductance `d[i]` was solved for": if the loop exits via the
    "space limited" branch, `ceff[i]` has already been shrunk by 10% *after* `d[i]` was
    computed with the pre-shrink value -- so PROCESS's own `ceff[imax]` (as read
    downstream, e.g. for `cmax`/`snet`) can be inconsistent with `d[imax]` by that same
    step, a real quirk of the source, not fixed here (see `vacuum.md`). `ceff_used` is
    the value `d` was *actually* solved for, always satisfying
    `duct_diameter_residual(d, ..., ceff_used) ~= 0` -- this is what
    `TestVacuumPumpingOld`'s residual check in `test_vacuum.py` uses, so the check
    stays meaningful in both the ordinary and "space limited" cases. `ceff_final` is
    kept too, since it's what a caller mimicking PROCESS's `cmax = ceff[i]` needs.

    Parameters
    ----------
    l1, l2, l3 :
        Duct segment lengths (m).
    xmult_i :
        This gas species' conductance-to-nitrogen multiplier.
    ceff_i_init :
        Initial target conductance (m^3/s), before any shrinking.
    a1max :
        Maximum aperture cross-sectional area available between adjacent TF coils (m2).
    s_i :
        This gas species' actually-required pump speed (m^3/s) -- the floor `ceff_i`
        must stay above.
    max_outer :
        Cap on the number of 10%-shrink steps. PROCESS's own outer loop has no cap at
        all (a genuine unbounded `while True`); `ceff_i *= 0.9` shrinks geometrically,
        so any physically plausible input resolves in far fewer than 64 steps (`0.9**64
        ~ 1.2e-3`) -- see `vacuum.md`'s open questions for the case this cap could,
        in principle, truncate early where PROCESS would not.

    Returns
    -------
    :
        `(d, ceff_used, ceff_final, nflag)`.
    """

    def cond(carry):
        *_, done = carry
        return jnp.logical_not(done)

    def body(carry):
        _d, _ceff_used, ceff_cur, _nflag, it, _done = carry
        d_new = solve_duct_diameter(l1, l2, l3, xmult_i, ceff_cur)
        a1_new = 0.25 * jnp.pi * d_new * d_new
        fits = a1_new < a1max
        ceff_next = jnp.where(fits, ceff_cur, ceff_cur * 0.9)
        too_small = jnp.logical_and(jnp.logical_not(fits), ceff_next <= 1.1 * s_i)
        done_next = jnp.logical_or(fits, jnp.logical_or(too_small, it + 1 >= max_outer))
        nflag_next = jnp.where(too_small, 1, 0)
        return (d_new, ceff_cur, ceff_next, nflag_next, it + 1, done_next)

    init = (
        jnp.asarray(1.0),
        ceff_i_init,
        ceff_i_init,
        jnp.asarray(0),
        jnp.asarray(0),
        jnp.asarray(False),
    )
    d, ceff_used, ceff_final, nflag, _it, _done = jax.lax.while_loop(cond, body, init)
    return d, ceff_used, ceff_final, nflag


def _solve_vacuum_pumping_old(
    p_fusion_total_mw,
    rmajor,
    rminor,
    dsol,
    a_plasma_surface,
    vol_plasma,
    dr_shld_outboard,
    dr_shld_inboard,
    dr_tf_inboard,
    ritf,
    n_tf_coils,
    t_plant_pulse_dwell,
    n_divertors,
    qtorus,
    gasld,
    i_vac_pump_dwell,
    i_vacuum_pump_type,
    pres_vv_chamber_base,
    pres_div_chamber_burn,
    outgrat_fw,
    t_plant_pulse_coil_precharge,
):
    """`Vacuum.vacuum` (the `"old"` ETR duct-sizing model), full diagnostic form.

    Not the audited unit's public signature -- see `calculate_vacuum_pumping_old`,
    which slices this down to PROCESS's actual five (well, six, see below) real
    outputs. This diagnostic form additionally returns `imax` (which gas species ended
    up governing the design) and `ceff_used` (the target conductance `d[imax]` was
    actually solved for, per `solve_duct_geometry`'s docstring) -- both needed by
    `test_vacuum.py`'s residual check, neither a `VarPath` PROCESS itself writes.

    **`nplasma`/`temp_vv_chamber_gas_burn_end` dropped from the signature**, a real
    finding (see `vacuum.md`): PROCESS's `vacuum()` reads both
    (`nplasma`/`.vacuum.temp_vv_chamber_gas_burn_end`) to compute
    `pend = 0.5*nplasma*k*temp_vv_chamber_gas_burn_end` and `pstart = 0.01*pend`, then
    uses only `log(pend/pstart)` -- which is `log(100)` *identically*, for any `pend
    != 0`, since `pstart` is defined as a fixed fraction of `pend`. Neither input can
    change any of `vacuum()`'s five returned values through this path (they still
    affect the `pend`/`pstart` numbers `_write_to_outfile` prints, which is out of this
    port's scope). This port computes `math.log(100.0)` directly instead of taking
    inputs that provably cancel.

    Parameters
    ----------
    p_fusion_total_mw :
        Fusion power (MW). `.physics.p_fusion_total_mw`.
    rmajor :
        Plasma major radius (m). `.physics.rmajor`. (`r0` in the source.)
    rminor :
        Plasma minor radius (m). `.physics.rminor`. (`aw` in the source.)
    dsol :
        Scrape-off layer average width (m), `0.5 * (dr_fw_plasma_gap_inboard +
        dr_fw_plasma_gap_outboard)`. Computed by the caller (`Vacuum.run`), not this
        function, in PROCESS -- kept as a single explicit argument here too, matching
        `vacuum()`'s own signature.
    a_plasma_surface :
        Plasma surface area (m2). `.physics.a_plasma_surface`. (`plasma_sarea`.)
    vol_plasma :
        Plasma volume (m3). `.physics.vol_plasma`. (`plasma_vol`.)
    dr_shld_outboard, dr_shld_inboard :
        Outboard/inboard shield thickness (m). `.build.dr_shld_outboard`/
        `dr_shld_inboard`. (`thshldo`/`thshldi`.)
    dr_tf_inboard :
        TF coil radial thickness (m). `.build.dr_tf_inboard`. (`thtf`.)
    ritf :
        Radius of the inboard TF leg point nearest the plasma (m), `r_shld_inboard_inner
        - dr_shld_vv_gap_inboard - dr_vv_inboard`. Computed by the caller in PROCESS,
        kept as a single explicit argument here.
    n_tf_coils :
        Number of TF coils. `.tfcoil.n_tf_coils`.
    t_plant_pulse_dwell :
        Dwell time between pulses (s). `.times.t_plant_pulse_dwell`.
    n_divertors :
        Number of divertors with pumping (1 or 2). `.divertor.n_divertors`. (`ndiv`.)
    qtorus :
        NBI gas load (deuterons/second). Always `0.0` at `Vacuum.run()`'s only call
        site (see `vacuum.md`) -- kept as an explicit argument for fidelity to
        `vacuum()`'s own signature.
    gasld :
        Total D-T gas load (kg/s), `2 * molflow_plasma_fuelling_required * m_fuel_amu *
        UMASS`. Computed by the caller in PROCESS, kept as a single explicit argument.
    i_vac_pump_dwell :
        Switch for dwell pumping options (0/1/2). `.vacuum.i_vac_pump_dwell`.
    i_vacuum_pump_type :
        Switch for pump type (`VacuumPumpType`, 0 turbomolecular / 1 compound
        cryopump). `.vacuum.i_vacuum_pump_type`.
    pres_vv_chamber_base :
        Base pressure required (Pa). `.vacuum.pres_vv_chamber_base`.
    pres_div_chamber_burn :
        Divertor chamber pressure during burn (Pa). `.vacuum.pres_div_chamber_burn`.
    outgrat_fw :
        First wall outgassing rate (Pa-m/s). `.vacuum.outgrat_fw`.
    t_plant_pulse_coil_precharge :
        CS ramp-up time (s). `.times.t_plant_pulse_coil_precharge`.

    Returns
    -------
    :
        `(pumpn, n_vv_vacuum_ducts, dlscal, m_vv_vacuum_duct_shield,
        dia_vv_vacuum_ducts, imax, ceff_used)`. `pumpn` here is the *raw* pump count,
        before `Vacuum.run()`'s `math.floor(pumpn + 0.5)` rounding to
        `.vacuum.n_vac_pumps_high` -- see `calculate_vacuum_pumping_old`, which applies
        that rounding.
    """
    xmult = jnp.asarray(XMULT)
    sp_turbo = jnp.asarray(_SP_TURBOMOLECULAR)
    sp_cryo = jnp.asarray(_SP_COMPOUND_CRYOPUMP)
    is_cryo = i_vacuum_pump_type == 1
    sp = jnp.where(is_cryo, sp_cryo, sp_turbo)

    ntf = jnp.floor(n_tf_coils)
    ndiv = n_divertors
    nduct = ntf * ndiv

    frate = gasld + qtorus * 6.64e-27
    thcsh = dr_shld_inboard / 3.0

    area = a_plasma_surface * (rminor + dsol) / rminor
    ogas = outgrat_fw * area * 10.0
    s0 = ogas / pres_vv_chamber_base

    volume = vol_plasma * (rminor + dsol) * (rminor + dsol) / (rminor * rminor)
    cond1 = jnp.logical_or(i_vac_pump_dwell == 1, t_plant_pulse_dwell == 0)
    cond2 = i_vac_pump_dwell == 2
    tpump = jnp.where(
        cond1,
        t_plant_pulse_coil_precharge,
        jnp.where(
            cond2,
            t_plant_pulse_dwell + t_plant_pulse_coil_precharge,
            t_plant_pulse_dwell,
        ),
    )
    # log(pend / pstart) == log(100) identically -- see the docstring above.
    s1 = volume / tpump * jnp.log(100.0)

    source = (p_fusion_total_mw * 1.0e6) * 1.47e-09
    fhe = source / (frate * 4.985e5)
    s2 = source / pres_div_chamber_burn / fhe
    s3 = (frate * 4.985e5 - source) / (pres_div_chamber_burn * (1.0 - fhe))
    s = jnp.stack([s0, s1, s2, s3])

    l1 = dr_shld_outboard + dr_tf_inboard
    l2 = dr_shld_outboard + 4.0
    l3 = 2.0
    ltot = l1 + l2 + l3

    theta = jnp.pi / ntf
    a1max = (rmajor + rminor - ritf - thcsh / jnp.tan(theta)) ** 2 * jnp.tan(theta)

    def outer_body(i, carry):
        imax, cmax, pumpn, ceff, d, ceff_used, nflag = carry
        sp_i = sp[i]
        xmult_i = xmult[i]
        xmult_imax = xmult[imax]
        s_i = s[i]

        sss = nduct / (1.0 / sp_i / pumpn + (1.0 / cmax) * xmult_i / xmult_imax)

        def do_process(_):
            ccc = 2.0 * s_i / nduct
            pumpn1 = 1.0 / (sp_i * (nduct / s_i - 1.0 / ccc))
            pumpn2 = 1.01 * s_i / (sp_i * nduct)
            pumpn_new = jnp.maximum(pumpn, jnp.maximum(pumpn1, pumpn2))
            ceff_i_val = 1.0 / (nduct / s_i - 1.0 / (sp_i * pumpn_new))
            d_i, ceff_used_i, ceff_final_i, nflag_i = solve_duct_geometry(
                l1, l2, l3, xmult_i, ceff_i_val, a1max, s_i
            )
            ceff_new = ceff.at[i].set(ceff_final_i)
            d_new = d.at[i].set(d_i)
            return (i, ceff_final_i, pumpn_new, ceff_new, d_new, ceff_used_i, nflag_i)

        def do_skip(_):
            return (imax, cmax, pumpn, ceff, d, ceff_used, nflag)

        return jax.lax.cond(sss > s_i, do_skip, do_process, operand=None)

    imax0 = jnp.asarray(1)
    cmax0 = jnp.asarray(0.01)
    pumpn0 = jnp.asarray(1.0)
    ceff0 = jnp.full((4,), 1e-6)
    d0 = jnp.full((4,), 1e-6)
    ceff_used0 = jnp.asarray(1e-6)
    nflag0 = jnp.asarray(0)

    imax, _cmax, pumpn, _ceff, d, ceff_used, nflag = jax.lax.fori_loop(
        0, 4, outer_body, (imax0, cmax0, pumpn0, ceff0, d0, ceff_used0, nflag0)
    )
    del nflag  # diagnostic only, not part of the audited output; see `vacuum.md`.

    pumpn *= nduct
    pumpn = jnp.where(is_cryo, pumpn * 2.0, pumpn)

    dimax = d[imax]
    dlscalc = l1 * dimax**1.4 + (ltot - l1) * (dimax * 1.2) ** 1.4

    thdsh = 0.0  # Hardcoded in the source -- "no biological shielding". See vacuum.md.
    arsh = 0.25 * jnp.pi * ((dimax * 1.2 + thdsh) ** 2 - (dimax * 1.2) ** 2)
    mvdsh = arsh * (ltot - l1) * _DUCT_SHIELD_DENSITY * _DUCT_SHIELD_SOLID_FRACTION

    return pumpn, nduct, dlscalc, mvdsh, dimax, imax, ceff_used


def _derive_vacuum_pumping_old_locals(
    dr_fw_plasma_gap_inboard,
    dr_fw_plasma_gap_outboard,
    r_shld_inboard_inner,
    dr_shld_vv_gap_inboard,
    dr_vv_inboard,
    molflow_plasma_fuelling_required,
    m_fuel_amu,
):
    """`dsol`/`ritf`/`gasld`, exactly as `Vacuum.run()` computes them (lines 50-88)
    immediately before calling `vacuum()`.

    Pulled out into its own function so `calculate_vacuum_pumping_old`'s real inputs
    are the raw `.build.*`/`.physics.*` fields PROCESS actually stores, not the three
    composite locals `run()` derives from them -- `Vacuum.run()`'s `vacuum()` argument
    itself takes `dsol`/`ritf`/`gasld` as plain parameters (it doesn't compute them
    either), which is a faithful match to *that* function's own boundary, but one frame
    short of where the real data dependencies live. None of the three has any other
    reader anywhere in the stellarator pipeline (grepped: each is a `run()`-local, used
    only at this one call site), so there is no case for minting a `VarPath` for any of
    them -- inlining the arithmetic here instead is the same move `naming_convention.md`
    calls `local-intermediate`.

    Returns
    -------
    :
        `(dsol, ritf, gasld)`.
    """
    dsol = 0.5 * (dr_fw_plasma_gap_inboard + dr_fw_plasma_gap_outboard)
    ritf = r_shld_inboard_inner - dr_shld_vv_gap_inboard - dr_vv_inboard
    gasld = 2.0 * molflow_plasma_fuelling_required * m_fuel_amu * _UMASS
    return dsol, ritf, gasld


def _solve_vacuum_pumping_old_from_fields(
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
    qtorus,
    molflow_plasma_fuelling_required,
    m_fuel_amu,
    i_vac_pump_dwell,
    i_vacuum_pump_type,
    pres_vv_chamber_base,
    pres_div_chamber_burn,
    outgrat_fw,
    t_plant_pulse_coil_precharge,
):
    """`_solve_vacuum_pumping_old`, with `dsol`/`ritf`/`gasld` derived internally.

    Same diagnostic 7-tuple return (`pumpn` unrounded, plus `imax`/`ceff_used`) -- see
    `_solve_vacuum_pumping_old` for the algorithm and full parameter documentation of
    everything below except the seven listed here, which replace `dsol`/`ritf`/`gasld`
    -- see `_derive_vacuum_pumping_old_locals`.

    Parameters
    ----------
    dr_fw_plasma_gap_inboard, dr_fw_plasma_gap_outboard :
        First-wall/plasma gap, inboard/outboard (m). `.build.dr_fw_plasma_gap_inboard`/
        `dr_fw_plasma_gap_outboard`.
    r_shld_inboard_inner :
        Inner radius of the inboard shield (m). `.build.r_shld_inboard_inner`.
    dr_shld_vv_gap_inboard :
        Inboard shield/VV gap (m). `.build.dr_shld_vv_gap_inboard`.
    dr_vv_inboard :
        Inboard VV thickness (m). `.build.dr_vv_inboard`.
    molflow_plasma_fuelling_required :
        Plasma fuelling rate (nucleus-pairs/s). `.physics.molflow_plasma_fuelling_required`.
    m_fuel_amu :
        Average fuel ion mass (amu). `.physics.m_fuel_amu`.
    """
    dsol, ritf, gasld = _derive_vacuum_pumping_old_locals(
        dr_fw_plasma_gap_inboard,
        dr_fw_plasma_gap_outboard,
        r_shld_inboard_inner,
        dr_shld_vv_gap_inboard,
        dr_vv_inboard,
        molflow_plasma_fuelling_required,
        m_fuel_amu,
    )
    return _solve_vacuum_pumping_old(
        p_fusion_total_mw,
        rmajor,
        rminor,
        dsol,
        a_plasma_surface,
        vol_plasma,
        dr_shld_outboard,
        dr_shld_inboard,
        dr_tf_inboard,
        ritf,
        n_tf_coils,
        t_plant_pulse_dwell,
        n_divertors,
        qtorus,
        gasld,
        i_vac_pump_dwell,
        i_vacuum_pump_type,
        pres_vv_chamber_base,
        pres_div_chamber_burn,
        outgrat_fw,
        t_plant_pulse_coil_precharge,
    )


def calculate_vacuum_pumping_old(
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
    qtorus,
    molflow_plasma_fuelling_required,
    m_fuel_amu,
    i_vac_pump_dwell,
    i_vacuum_pump_type,
    pres_vv_chamber_base,
    pres_div_chamber_burn,
    outgrat_fw,
    t_plant_pulse_coil_precharge,
):
    """`Vacuum.run()`'s `"old"` branch in full: `vacuum()`'s own pre-computation of
    `dsol`/`ritf`/`gasld`, the model itself, and `run()`'s rounding step -- every
    `data`-reachable part of `run()`'s `"old"`-branch computation, in one function with
    only real `VarPath`s as arguments.

    Returns exactly the five fields `Vacuum.run()`'s `"old"` branch writes to `data` --
    see `VacuumOld`'s `Output`s below. The first, `n_vac_pumps_high`, is `math.floor(
    pumpn + 0.5)` applied to the raw pump count `vacuum()` returns (that rounding
    happens in `Vacuum.run()`, not `Vacuum.vacuum`, but both are within `run()`'s
    reachable `"old"`-branch computation, the audited unit here) -- the un-rounded
    `pumpn` itself is never written to `data` anywhere, so it has no `VarPath` and is
    not part of this function's return.

    See `_solve_vacuum_pumping_old_from_fields`/`_solve_vacuum_pumping_old` for the
    algorithm and full parameter documentation; this is a thin wrapper applying the
    rounding and dropping the two diagnostic-only values (`imax`, `ceff_used`).

    Returns
    -------
    :
        `(n_vac_pumps_high, n_vv_vacuum_ducts, dlscal, m_vv_vacuum_duct_shield,
        dia_vv_vacuum_ducts)`.
    """
    pumpn, nduct, dlscalc, mvdsh, dimax, _imax, _ceff_used = (
        _solve_vacuum_pumping_old_from_fields(
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
            qtorus,
            molflow_plasma_fuelling_required,
            m_fuel_amu,
            i_vac_pump_dwell,
            i_vacuum_pump_type,
            pres_vv_chamber_base,
            pres_div_chamber_burn,
            outgrat_fw,
            t_plant_pulse_coil_precharge,
        )
    )
    n_vac_pumps_high = jnp.floor(pumpn + 0.5)
    return n_vac_pumps_high, nduct, dlscalc, mvdsh, dimax


class VacuumOld(ExplicitFunction):
    """cottax node: `calculate_vacuum_pumping_old`'s five real outputs.

    Every `Input` below is a genuine, already-existing `VarPath` -- no minting needed.
    `qtorus` is hardcoded `0.0` (not `Input`-wrapped) since it is always `0.0` at
    `Vacuum.run()`'s only call site (see `vacuum.md`), a static default rather than a
    place in `data`.
    """

    n_vac_pumps_high = Output(lambda s: s.vacuum.n_vac_pumps_high)
    n_vv_vacuum_ducts = Output(lambda s: s.vacuum.n_vv_vacuum_ducts)
    dlscal = Output(lambda s: s.vacuum.dlscal)
    m_vv_vacuum_duct_shield = Output(lambda s: s.vacuum.m_vv_vacuum_duct_shield)
    dia_vv_vacuum_ducts = Output(lambda s: s.vacuum.dia_vv_vacuum_ducts)

    def __call__(
        self,
        p_fusion_total_mw=Input(lambda s: s.physics.p_fusion_total_mw),
        rmajor=Input(lambda s: s.physics.rmajor),
        rminor=Input(lambda s: s.physics.rminor),
        dr_fw_plasma_gap_inboard=Input(lambda s: s.build.dr_fw_plasma_gap_inboard),
        dr_fw_plasma_gap_outboard=Input(lambda s: s.build.dr_fw_plasma_gap_outboard),
        a_plasma_surface=Input(lambda s: s.physics.a_plasma_surface),
        vol_plasma=Input(lambda s: s.physics.vol_plasma),
        dr_shld_outboard=Input(lambda s: s.build.dr_shld_outboard),
        dr_shld_inboard=Input(lambda s: s.build.dr_shld_inboard),
        dr_tf_inboard=Input(lambda s: s.build.dr_tf_inboard),
        r_shld_inboard_inner=Input(lambda s: s.build.r_shld_inboard_inner),
        dr_shld_vv_gap_inboard=Input(lambda s: s.build.dr_shld_vv_gap_inboard),
        dr_vv_inboard=Input(lambda s: s.build.dr_vv_inboard),
        n_tf_coils=Input(lambda s: s.tfcoil.n_tf_coils),
        t_plant_pulse_dwell=Input(lambda s: s.times.t_plant_pulse_dwell),
        n_divertors=Input(lambda s: s.divertor.n_divertors),
        molflow_plasma_fuelling_required=Input(
            lambda s: s.physics.molflow_plasma_fuelling_required
        ),
        m_fuel_amu=Input(lambda s: s.physics.m_fuel_amu),
        i_vac_pump_dwell=Input(lambda s: s.vacuum.i_vac_pump_dwell),
        i_vacuum_pump_type=Input(lambda s: s.vacuum.i_vacuum_pump_type),
        pres_vv_chamber_base=Input(lambda s: s.vacuum.pres_vv_chamber_base),
        pres_div_chamber_burn=Input(lambda s: s.vacuum.pres_div_chamber_burn),
        outgrat_fw=Input(lambda s: s.vacuum.outgrat_fw),
        t_plant_pulse_coil_precharge=Input(
            lambda s: s.times.t_plant_pulse_coil_precharge
        ),
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
