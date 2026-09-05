"""Pure-functional port of `Stellarator.st_phys`'s genuinely-new sub-computations
(chunk 1B of unit #1). Audit record: `plasma_physics.md` -- read it first.

`st_phys` (`process/models/stellarator/stellarator.py:1886-2456`) is, per the audit
record, an acyclic composition of ~13 sub-computations. Most of those sub-computations
turn out to already have ported nodes elsewhere in `functional_process/` -- see the
record's "already-ported sub-calls" section for the full cross-reference. What is ported
*here* is only the arithmetic that lives directly in `st_phys`'s own body (never
delegated to another `Model`/function), and is genuinely new:

1. `TotalField` / `PoloidalFieldFromRotationalTransform` -- the field/beta block
   (lines 1915-1976). Deliberately **two separate nodes**, not one: the source reads
   `.physics.b_plasma_surface_poloidal_average` (line 1918) *before* overwriting it
   (lines 1971-1976) -- exactly the read-before-write the audit record identifies as
   the Picard-iteration mechanism behind `power_at_ignition_point`'s hardcoded "call
   twice". A single node may not both read and own one `VarPath` (`~/jaxgraph/CLAUDE.md`
   "The graph": "A node may not read what it owns"), so this chunk is split along
   exactly that seam -- one node reads the (possibly stale) value, a different node
   produces the next one. Which is upstream of which in the assembled graph is a wiring
   decision for a later pass, not resolved here (see "Framing" in the task that
   commissioned this audit).
2. `StellaratorBetaAndRhoStar` -- beta_total_vol_avg / e_plasma_beta / rho_star
   (lines 1930-1968). Reads `.physics.beta_fast_alpha`/`.physics.beta_beam` as plain
   `VarPath` inputs; a follow-up check found this is an ordinary acyclic edge (Shape A,
   not a self-loop) for both quantities -- see this node's own docstring for the
   dependency check that confirms it, and `plasma_physics.md`'s "New findings"
   section for the fuller writeup. No `FixedPointFunction` needed here.
3. `FusionPowerTotalsMw` -- p_plasma_dt_mw / p_dhe3_total_mw / p_dd_total_mw
   (lines 1991-2001), from the already-ported `FusionRates` node's outputs.
4. `NeutronWallLoad` -- pflux_fw_neutron_mw (lines 2093-2117), a 3-way switch on
   `i_pflux_fw_neutron` / `heat_transport.ipowerflow`.
5. `HeatingAndRadiationPower` -- powht / p_plasma_rad_mw / psolradmw /
   p_plasma_separatrix_mw / p_fw_alpha_mw (lines 2175-2220).
6. `RadiatedWallLoadAndFraction` -- pflux_fw_rad_mw / pflux_fw_rad_max_mw /
   rad_fraction_total (lines 2222-2257), the same 3-way switch as (4), applied to the
   radiated power instead of the neutron power.
7. `ThermalEnergyTotals` -- eden_plasma_thermal_vol_avg / e_plasma_thermal_total
   (lines 2282-2290), trivial sums of the already-ported `ElectronThermalEnergy` /
   `IonThermalEnergy` outputs.

Switches (`i_pflux_fw_neutron`, `heat_transport.ipowerflow`, `i_plasma_ignited`) are
kept as **static** Python `int`s (plain `if`/`elif` inside the pure function, `eqx.
field(static=True)` on the node), the same convention already established by
`ConfinementTime.i_rad_loss` (`models/physics/confinement_time.py`) and
`FastAlphaBeta.i_beta_fast_alpha` (`models/physics/pure_formulas.py`) --
not the naming_convention.md default "split" recommendation, since the precedent in
this codebase already treats a differing-but-small reads-set delta as the "exception:
static kwarg" case rather than forcing per-value node classes. See the audit record's
"switches touched" section for the reasoning applied here specifically.

Everything st_phys calls into another model/function for (plasma_composition,
plasma_profile.run(), st_heat, the fusion-rate/beam-fusion machinery, fast_alpha_beta,
rether, calculate_radiation_powers, calaculate_stored_thermal_energy,
calculate_confinement_time, calculate_double_and_triple_product,
calculate_total_plasma_heating_power, calculate_radiation_fraction, phyaux,
calc_neoclassics) stays out of this file -- either it already has a node elsewhere
(cross-referenced in the record) or it is still entangled with an unported unit
(beam_fusion, plasma_composition's self-loop, plasma_profile.run()'s scope gap,
calc_neoclassics' incomplete orchestrator -- all audit-only, see the record).
"""

import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.models.safe_math import safe_sqrt
from functional_process.paths import (
    constraints,
    current_drive,
    first_wall,
    fwbs,
    physics,
    stellarator,
)
from functional_process.vocabulary import constants


def calculate_total_field(b_plasma_toroidal_on_axis, b_plasma_surface_poloidal_average):
    """Total field magnitude from toroidal and poloidal components.

    Ports `stellarator.py:1916-1919`. `b_plasma_surface_poloidal_average` here is
    whatever value the field currently holds -- see this module's docstring on why that
    is a separate node from `calculate_poloidal_field_from_rotational_transform`, which
    produces the *next* value of the same `VarPath`.
    """
    return safe_sqrt(b_plasma_toroidal_on_axis**2 + b_plasma_surface_poloidal_average**2)


def calculate_poloidal_field_from_rotational_transform(
    rminor, b_plasma_toroidal_on_axis, rmajor, iotabar
):
    """Poloidal field from the stellarator rotational transform.

    Ports `stellarator.py:1971-1976`. Purely geometric -- no dependency on anything
    `st_phys` itself computes (`rminor`/`rmajor` are build/geometry outputs, `iotabar`
    is a stellarator-config input), which is worth noting for the later wiring pass:
    since this node's inputs never depend on `calculate_total_field`'s outputs, wiring
    this node as the producer and `calculate_total_field` as a consumer of
    `.physics.b_plasma_surface_poloidal_average` gives a single acyclic pass with no
    iteration needed at all -- consistent with `../../CLAUDE.md`'s "most of PROCESS is
    probably not actually cyclic once dependencies are made explicit". Confirming that
    requires the full graph (this chunk alone cannot rule out a channel through
    `plasma_composition`/`plasma_profile.run()`, both still pending), so it is reported
    here, not assumed.
    """
    return rminor * b_plasma_toroidal_on_axis / rmajor * iotabar


def calculate_stellarator_beta_and_rho_star(
    beta_fast_alpha,
    beta_beam,
    nd_plasma_electrons_vol_avg,
    temp_plasma_electron_density_weighted_kev,
    nd_plasma_ions_total_vol_avg,
    temp_plasma_ion_density_weighted_kev,
    b_plasma_total,
    vol_plasma,
    m_ions_total_amu,
    nd_plasma_electron_line,
    b_plasma_toroidal_on_axis,
    eps,
    rmajor,
):
    """Total beta, stored thermal energy and normalised gyroradius.

    Ports `stellarator.py:1930-1968` (`beta_total_vol_avg`, `e_plasma_beta`,
    `rho_star`).

    `beta_fast_alpha` and `beta_beam` are read here **textually before** either is
    (re)computed later in the same `st_phys` source method: `beta_fast_alpha` by
    `self.beta.fast_alpha_beta(...)` at line 2079, `beta_beam` by `reactions.
    beam_fusion(...)` at line 2011-2017 (only when beams are active). A follow-up check
    (per `_audit/next_steps.md` §5's Shape A/B distinction, using the same
    dependency-check that resolved `Divertor`/`DivertorPlateMass` as *not* a real cycle)
    found this is **not** the same shape as the `b_plasma_surface_poloidal_average`
    self-loop above -- it is an ordinary acyclic cross-node edge (Shape A), for both
    quantities, not a genuine self-loop (Shape B) needing a `FixedPointFunction`:

    - `beta_fast_alpha`'s sole owner is `FastAlphaBeta`
      (`models/physics/pure_formulas.py`, already registered in
      `total_process.py`). `FastAlphaBeta`'s own inputs (`b_plasma_surface_poloidal_
      average`, `b_plasma_toroidal_on_axis`, density/temperature averages,
      `pden_alpha_total_mw`, `pden_plasma_alpha_mw`, `f_plasma_fuel_deuterium`) do not
      include `beta_total_vol_avg`/`e_plasma_beta`/`rho_star` (this node's own outputs)
      or `beta_beam` -- so there is no path back from this node to `FastAlphaBeta`.
      Assembling `to_graph([FastAlphaBeta(i_beta_fast_alpha=1),
      StellaratorBetaAndRhoStar()])` succeeds (two ordinary nodes, no
      "reads ... which it also owns" error), confirming this directly.
    - `beta_beam`'s sole owner in the PROCESS source is `reactions.beam_fusion(...)`
      (`process/models/physics/fusion_reactions.py`), which is not yet ported (blocked
      on a non-JAX-traceable `scipy.integrate.quad` call -- audit-only, see
      `fusion_reactions.md`). Its inputs (`beamfus0`, `betbm0`, `b_plasma_total`,
      `c_beam_total`, `nd_plasma_electrons_vol_avg`, `nd_plasma_fuel_ions_vol_avg`,
      `dlamie`, `e_beam_kev`, `f_plasma_fuel_deuterium`, `f_plasma_fuel_tritium`,
      `f_beam_tritium`, `temp_plasma_electron_density_weighted_kev`, `vol_plasma`,
      `n_charge_plasma_effective_mass_weighted_vol_avg`) likewise never include
      `beta_total_vol_avg`/`e_plasma_beta`/`rho_star`/`beta_beam` itself -- so once
      `beam_fusion` is ported as a node, it too wires as an ordinary upstream producer
      of `.physics.beta_beam`, ready-made, no fixed point required. This can't be
      verified by an actual `to_graph()` call yet (no node exists to assemble), only by
      reading `beam_fusion`'s signature -- flagged here rather than assumed.

    Both remain ordinary `VarPath` inputs on this node; no code change was needed here
    -- see the module/record docstrings for the fuller writeup of this finding.
    """
    beta_total_vol_avg = (
        beta_fast_alpha
        + beta_beam
        + 2.0e3
        * constants.RMU0
        * constants.ELECTRON_CHARGE
        * (
            nd_plasma_electrons_vol_avg * temp_plasma_electron_density_weighted_kev
            + nd_plasma_ions_total_vol_avg * temp_plasma_ion_density_weighted_kev
        )
        / b_plasma_total**2
    )
    e_plasma_beta = (
        1.5e0
        * beta_total_vol_avg
        * b_plasma_total
        * b_plasma_total
        / (2.0e0 * constants.RMU0)
        * vol_plasma
    )
    rho_star = safe_sqrt(
        2.0e0
        * constants.PROTON_MASS
        * m_ions_total_amu
        * e_plasma_beta
        / (3.0e0 * vol_plasma * nd_plasma_electron_line)
    ) / (constants.ELECTRON_CHARGE * b_plasma_toroidal_on_axis * eps * rmajor)

    return beta_total_vol_avg, e_plasma_beta, rho_star


def calculate_fusion_power_totals_mw(
    dt_power_density_plasma, dhe3_power_density, dd_power_density, vol_plasma
):
    """D-T/D-He3/D-D total fusion power, from power densities already computed by the
    already-ported `FusionRates` node.

    Ports `stellarator.py:1991-2001`.
    """
    p_plasma_dt_mw = dt_power_density_plasma * vol_plasma
    p_dhe3_total_mw = dhe3_power_density * vol_plasma
    p_dd_total_mw = dd_power_density * vol_plasma
    return p_plasma_dt_mw, p_dhe3_total_mw, p_dd_total_mw


def calculate_fusion_totals_no_beam(fusden_plasma, fusden_plasma_alpha, p_plasma_dt_mw):
    """Total fusion rates and D-T power with **no neutral beam** -- three identities.

    Ports the `else` arm of `stellarator.py:2002-2054`. PROCESS adds a beam-driven
    contribution to each of these three totals when
    `p_hcd_beam_injected_total_mw != 0` *and* the plasma is not ignited; otherwise
    "the total alpha rates and power are the same as the plasma values"
    (`stellarator.py:2048-2049`, PROCESS's own comment), which is this arm.

    Arithmetically trivial, structurally not: without it `.physics.fusden_total`,
    `.physics.fusden_alpha_total` and `.physics.p_dt_total_mw` are **boundary inputs**
    of the port's graph -- frozen constants seeded from the converged run -- while
    `FusionRates` computes `fusden_plasma`/`fusden_plasma_alpha` right next to them and
    nothing reads the result. Every value was therefore correct at the reference point
    and every *derivative* through them was zero, which is exactly the defect class
    `_audit/optimise_design.md` §10.5a records for iteration variable 4.

    The beam arm is deliberately not ported: it calls `reactions.beam_fusion`, which
    unit #19 records as audit-only (a `scipy.integrate.quad` call that is both
    non-traceable and accurate only to ~1e-6 in PROCESS's own hands -- four orders
    outside tier-1's tolerance). See `_audit/boundary_inputs_audit.md` §4c (b7)/(b8).

    Parameters
    ----------
    fusden_plasma :
        Fusion reaction rate from the plasma alone (/m3/s). `.physics.fusden_plasma`.
    fusden_plasma_alpha :
        Alpha production rate from the plasma alone (/m3/s).
        `.physics.fusden_plasma_alpha`.
    p_plasma_dt_mw :
        D-T fusion power from the plasma alone (MW). `.physics.p_plasma_dt_mw`.

    Returns
    -------
    tuple
        `(fusden_total, fusden_alpha_total, p_dt_total_mw)`.
    """
    return fusden_plasma, fusden_plasma_alpha, p_plasma_dt_mw


def calculate_clipped_radiation_powers(
    pden_plasma_core_rad_mw_unclipped, pden_plasma_outer_rad_mw_unclipped, vol_plasma
):
    """`st_phys`'s two zero-clips on the radiation power densities, and the two total
    powers it forms from them.

    Ports `stellarator.py:2152-2166`. Four writes, one node, because they are one
    straight-line block with no branch between them: the clips and the products PROCESS
    computes *from the clipped values*.

    **The clips belong here rather than in `radiation_power.py`** and that is a
    modelling fact, not a filing choice: `calculate_radiation_powers` has two callers
    and only this one clips (`physics.PhysicsCalculations.physics()`,
    `physics.py:750-753`, does not). A clip inside the callee would be wrong for the
    other caller. `PlasmaRadiationPowers` therefore mints
    `pden_plasma_core_rad_mw_unclipped`/`pden_plasma_outer_rad_mw_unclipped` and this
    function owns the real fields.

    `jnp.maximum(x, 0.0)` rather than `jnp.clip`: PROCESS's own expression is
    `max(x, 0.0)`, one-sided, and there is no upper bound to state. Its derivative is
    the step function -- well defined either side and conventionally `0` at the join.
    This is **not** the `jnp.sqrt(jnp.maximum(0.0, x))` trap that has bitten this port
    twice: there is no infinite-derivative outer function here, so no `inf * 0`.

    Measured at this run's converged point, both clips are **inactive** -- core
    `0.0575`, outer `0.0553`, against a threshold of `0.0` -- so registering this
    changes no value here. What it changes is that `.physics.p_plasma_inner_rad_mw`
    acquires a producer at all: `StellaratorConfinementTime` reads it, and until now it
    read a frozen boundary input. That is the defect class this project has now found
    five times, every one invisible to a value comparison
    (`_audit/boundary_inputs_audit.md` §7 item 6).

    `p_plasma_rad_mw` (`stellarator.py:2168-2170`) is deliberately not here -- it is
    already owned by `HeatingAndRadiationPower`, and it is formed from the *unclipped*
    `pden_plasma_rad_mw`, so it is not part of this block's arithmetic.

    Parameters
    ----------
    pden_plasma_core_rad_mw_unclipped :
        Core radiation power density before the clip (MW/m3).
        `.physics.pden_plasma_core_rad_mw_unclipped`, minted by `PlasmaRadiationPowers`.
    pden_plasma_outer_rad_mw_unclipped :
        Edge radiation power density before the clip (MW/m3).
        `.physics.pden_plasma_outer_rad_mw_unclipped`, same origin.
    vol_plasma :
        Plasma volume (m3). `.physics.vol_plasma`.

    Returns
    -------
    tuple
        `(pden_plasma_core_rad_mw, pden_plasma_outer_rad_mw, p_plasma_inner_rad_mw,
        p_plasma_outer_rad_mw)` -- the two clipped densities and the two total powers,
        in PROCESS's own write order.
    """
    pden_plasma_core_rad_mw = jnp.maximum(pden_plasma_core_rad_mw_unclipped, 0.0)
    pden_plasma_outer_rad_mw = jnp.maximum(pden_plasma_outer_rad_mw_unclipped, 0.0)
    # PROCESS's own comment on the first of these: "Should probably be vol_core".
    # Reproduced as written.
    return (
        pden_plasma_core_rad_mw,
        pden_plasma_outer_rad_mw,
        pden_plasma_core_rad_mw * vol_plasma,
        pden_plasma_outer_rad_mw * vol_plasma,
    )


def calculate_neutron_wall_load(
    i_pflux_fw_neutron,
    ipowerflow,
    ffwal,
    p_neutron_total_mw,
    a_plasma_surface,
    fhole,
    a_fw_total,
    f_a_fw_outboard_hcd,
    f_ster_div_single,
):
    """Nominal mean neutron wall load, 3-way switch on `i_pflux_fw_neutron` /
    `heat_transport.ipowerflow`.

    Ports `stellarator.py:2095-2117`. `i_pflux_fw_neutron` and `ipowerflow` are **static**
    Python `int`s (see this module's docstring) -- an ordinary `if`/`elif` is safe under
    `jax.jacfwd` as long as the harness contract excludes them from differentiation
    (`static_argnames`), the same convention `calculate_confinement_time`/
    `fast_alpha_beta` already use.
    """
    if i_pflux_fw_neutron == 1:
        return calculate_neutron_wall_load_scaled_plasma_surface(
            ffwal, p_neutron_total_mw, a_plasma_surface
        )
    if ipowerflow == 0:
        return calculate_neutron_wall_load_first_wall_area_pre_2014(
            p_neutron_total_mw, fhole, a_fw_total
        )
    return calculate_neutron_wall_load_first_wall_area_comprehensive_2014(
        p_neutron_total_mw, fhole, a_fw_total, f_a_fw_outboard_hcd, f_ster_div_single
    )


def _wall_load_scaled_plasma_surface(ffwal, power_mw, a_plasma_surface):
    """`i_pflux_fw_neutron == SCALED_PLASMA_SURFACE_AREA` (1): the flux is the power
    over the *plasma* surface, scaled by `ffwal` (`stellarator.py:2100`).

    **Reads no first-wall field at all** -- not `.fwbs.fhole`, not
    `.first_wall.a_fw_total`, not `f_a_fw_outboard_hcd`/`f_ster_div_single`. That is
    the four-edge difference the `i_pflux_fw_neutron` static kwarg invented on each of
    the two wall-load nodes, and `.first_wall.a_fw_total` is `stellarator.fw_area`'s own
    output -- so the invented edge crossed a slot the factory already resolves
    (`_audit/switch_kwarg_survey.md` band (b2)).
    """
    return ffwal * power_mw / a_plasma_surface


def _wall_load_first_wall_area_pre_2014(power_mw, fhole, a_fw_total):
    """`i_pflux_fw_neutron != 1` with `ipowerflow == PRE_2014` (0): the power over the
    first-wall area, less the hole fraction (`stellarator.py:2107`).
    """
    return (1.0e0 - fhole) * power_mw / a_fw_total


def _wall_load_first_wall_area_comprehensive_2014(
    power_mw, fhole, a_fw_total, f_a_fw_outboard_hcd, f_ster_div_single
):
    """`i_pflux_fw_neutron != 1` with `ipowerflow == COMPREHENSIVE_2014` (1): as
    `_wall_load_first_wall_area_pre_2014`, with the HCD-port and divertor solid angles
    also removed (`stellarator.py:2112-2116`).
    """
    return (
        (1.0e0 - fhole - f_a_fw_outboard_hcd - f_ster_div_single) * power_mw / a_fw_total
    )


def calculate_neutron_wall_load_scaled_plasma_surface(
    ffwal, p_neutron_total_mw, a_plasma_surface
):
    """`calculate_neutron_wall_load`'s `SCALED_PLASMA_SURFACE_AREA` arm -- the
    reference run's."""
    return _wall_load_scaled_plasma_surface(ffwal, p_neutron_total_mw, a_plasma_surface)


def calculate_neutron_wall_load_first_wall_area_pre_2014(
    p_neutron_total_mw, fhole, a_fw_total
):
    """`calculate_neutron_wall_load`'s `(FIRST_WALL_AREA, PRE_2014)` arm."""
    return _wall_load_first_wall_area_pre_2014(p_neutron_total_mw, fhole, a_fw_total)


def calculate_neutron_wall_load_first_wall_area_comprehensive_2014(
    p_neutron_total_mw, fhole, a_fw_total, f_a_fw_outboard_hcd, f_ster_div_single
):
    """`calculate_neutron_wall_load`'s `(FIRST_WALL_AREA, COMPREHENSIVE_2014)` arm."""
    return _wall_load_first_wall_area_comprehensive_2014(
        p_neutron_total_mw, fhole, a_fw_total, f_a_fw_outboard_hcd, f_ster_div_single
    )


def calculate_heating_and_radiation_power(
    f_p_alpha_plasma_deposited,
    p_alpha_total_mw,
    p_non_alpha_charged_mw,
    p_plasma_ohmic_mw,
    pden_plasma_rad_mw,
    vol_plasma,
    i_plasma_ignited,
    p_hcd_injected_total_mw,
    f_rad,
):
    """Heating power to the plasma, SOL radiation split, and alpha power to the wall.

    Ports `stellarator.py:2175-2220` (`powht` is a plain Python local in the source,
    never stored to `data` -- folded in here rather than given its own node/`VarPath`,
    per `_audit/schema.md`'s `local-intermediate` classification). `i_plasma_ignited` is
    a **static** Python `int` (see module docstring); ordinary Python `if` is safe under
    tracing once the harness contract excludes it from differentiation.

    `pden_plasma_rad_mw * vol_plasma` is computed twice in the PROCESS source (once
    inside `powht`'s subtraction, once as `.physics.p_plasma_rad_mw`'s first write,
    line 2167) -- both folded into one `jnp` expression here (`p_plasma_rad_mw_raw`)
    rather than repeated; `minor` JAX-difficulty note, not a correctness issue (both
    reads are the same `VarPath` at the same point in the call).

    Returns
    -------
    :
        `(p_plasma_rad_mw, psolradmw, p_plasma_separatrix_mw, p_fw_alpha_mw)`.
    """
    if i_plasma_ignited == 0:  # PlasmaIgnitionModel.NON_IGNITED
        return calculate_heating_and_radiation_power_non_ignited(
            f_p_alpha_plasma_deposited,
            p_alpha_total_mw,
            p_non_alpha_charged_mw,
            p_plasma_ohmic_mw,
            pden_plasma_rad_mw,
            vol_plasma,
            f_rad,
            p_hcd_injected_total_mw,
        )
    return calculate_heating_and_radiation_power_ignited(
        f_p_alpha_plasma_deposited,
        p_alpha_total_mw,
        p_non_alpha_charged_mw,
        p_plasma_ohmic_mw,
        pden_plasma_rad_mw,
        vol_plasma,
        f_rad,
    )


def calculate_heating_and_radiation_power_ignited(
    f_p_alpha_plasma_deposited,
    p_alpha_total_mw,
    p_non_alpha_charged_mw,
    p_plasma_ohmic_mw,
    pden_plasma_rad_mw,
    vol_plasma,
    f_rad,
):
    """`i_plasma_ignited == IGNITED` (1) -- the reference run's.

    **`.current_drive.p_hcd_injected_total_mw` is not read at all**: an ignited plasma
    adds no injected heating to `powht` (`stellarator.py:2183`). That single read is a
    `.current_drive -> .physics` edge no ignited run makes, and it is the one the
    `i_plasma_ignited` static kwarg invented here.
    """
    return _heating_and_radiation_power(
        0.0,
        f_p_alpha_plasma_deposited,
        p_alpha_total_mw,
        p_non_alpha_charged_mw,
        p_plasma_ohmic_mw,
        pden_plasma_rad_mw,
        vol_plasma,
        f_rad,
    )


def calculate_heating_and_radiation_power_non_ignited(
    f_p_alpha_plasma_deposited,
    p_alpha_total_mw,
    p_non_alpha_charged_mw,
    p_plasma_ohmic_mw,
    pden_plasma_rad_mw,
    vol_plasma,
    f_rad,
    p_hcd_injected_total_mw,
):
    """`i_plasma_ignited == NON_IGNITED` (0) -- PROCESS's own default.

    The injected heating joins `powht` after the `1e-5` clamp, which is why it is added
    to the *clamped* value rather than folded into the sum.
    """
    return _heating_and_radiation_power(
        p_hcd_injected_total_mw,
        f_p_alpha_plasma_deposited,
        p_alpha_total_mw,
        p_non_alpha_charged_mw,
        p_plasma_ohmic_mw,
        pden_plasma_rad_mw,
        vol_plasma,
        f_rad,
    )


def _heating_and_radiation_power(
    p_hcd_added_mw,
    f_p_alpha_plasma_deposited,
    p_alpha_total_mw,
    p_non_alpha_charged_mw,
    p_plasma_ohmic_mw,
    pden_plasma_rad_mw,
    vol_plasma,
    f_rad,
):
    """The body both `i_plasma_ignited` arms share, given the injected-heating term the
    arm contributes to `powht` -- a literal `0.0` when ignited.

    Written as an added term rather than as a branch so the two arms differ by data and
    not by an integer, and so `powht + 0.0` is the identical floating-point expression
    the ignited branch had (the clamp is applied first either way).
    """
    p_plasma_rad_mw_raw = pden_plasma_rad_mw * vol_plasma

    powht = (
        f_p_alpha_plasma_deposited * p_alpha_total_mw
        + p_non_alpha_charged_mw
        + p_plasma_ohmic_mw
        - p_plasma_rad_mw_raw
    )
    powht = jnp.maximum(0.00001e0, powht)

    powht = powht + p_hcd_added_mw

    psolradmw = f_rad * powht
    p_plasma_separatrix_mw = powht - psolradmw
    p_plasma_separatrix_mw = jnp.maximum(0.001e0, p_plasma_separatrix_mw)

    p_plasma_rad_mw = jnp.maximum(0.0e0, p_plasma_rad_mw_raw) + psolradmw

    p_fw_alpha_mw = p_alpha_total_mw * (1.0e0 - f_p_alpha_plasma_deposited)

    return p_plasma_rad_mw, psolradmw, p_plasma_separatrix_mw, p_fw_alpha_mw


def calculate_radiated_wall_load_and_fraction(
    i_pflux_fw_neutron,
    ipowerflow,
    ffwal,
    p_plasma_rad_mw,
    a_plasma_surface,
    fhole,
    a_fw_total,
    f_a_fw_outboard_hcd,
    f_ster_div_single,
    f_fw_rad_max,
    f_p_alpha_plasma_deposited,
    p_alpha_total_mw,
    p_non_alpha_charged_mw,
    p_plasma_ohmic_mw,
    p_hcd_injected_total_mw,
):
    """Nominal mean photon wall load, its constraint bound, and the total radiated
    fraction. Same 3-way switch as `calculate_neutron_wall_load`, applied to
    `p_plasma_rad_mw` instead of `p_neutron_total_mw`.

    Ports `stellarator.py:2223-2257`. `rad_fraction_total`'s denominator is not
    clamped in the source (a genuine `ZeroDivisionError` domain edge in plain Python,
    `nan`/`inf` under `jnp` -- not guarded here since it is not physically reachable at
    any sampled operating point, flagged as an open question rather than silently
    patched).
    """
    shared = (
        p_plasma_rad_mw,
        f_fw_rad_max,
        f_p_alpha_plasma_deposited,
        p_alpha_total_mw,
        p_non_alpha_charged_mw,
        p_plasma_ohmic_mw,
        p_hcd_injected_total_mw,
    )
    if i_pflux_fw_neutron == 1:
        return calculate_radiated_wall_load_scaled_plasma_surface(
            ffwal, a_plasma_surface, *shared
        )
    if ipowerflow == 0:
        return calculate_radiated_wall_load_first_wall_area_pre_2014(
            fhole, a_fw_total, *shared
        )
    return calculate_radiated_wall_load_first_wall_area_comprehensive_2014(
        fhole, a_fw_total, f_a_fw_outboard_hcd, f_ster_div_single, *shared
    )


def calculate_radiated_wall_load_scaled_plasma_surface(
    ffwal,
    a_plasma_surface,
    p_plasma_rad_mw,
    f_fw_rad_max,
    f_p_alpha_plasma_deposited,
    p_alpha_total_mw,
    p_non_alpha_charged_mw,
    p_plasma_ohmic_mw,
    p_hcd_injected_total_mw,
):
    """`calculate_radiated_wall_load_and_fraction`'s `SCALED_PLASMA_SURFACE_AREA` arm --
    the reference run's. Reads no first-wall field; see
    `_wall_load_scaled_plasma_surface`."""
    return _radiated_wall_load(
        _wall_load_scaled_plasma_surface(ffwal, p_plasma_rad_mw, a_plasma_surface),
        p_plasma_rad_mw,
        f_fw_rad_max,
        f_p_alpha_plasma_deposited,
        p_alpha_total_mw,
        p_non_alpha_charged_mw,
        p_plasma_ohmic_mw,
        p_hcd_injected_total_mw,
    )


def calculate_radiated_wall_load_first_wall_area_pre_2014(
    fhole,
    a_fw_total,
    p_plasma_rad_mw,
    f_fw_rad_max,
    f_p_alpha_plasma_deposited,
    p_alpha_total_mw,
    p_non_alpha_charged_mw,
    p_plasma_ohmic_mw,
    p_hcd_injected_total_mw,
):
    """`calculate_radiated_wall_load_and_fraction`'s `(FIRST_WALL_AREA, PRE_2014)`
    arm."""
    return _radiated_wall_load(
        _wall_load_first_wall_area_pre_2014(p_plasma_rad_mw, fhole, a_fw_total),
        p_plasma_rad_mw,
        f_fw_rad_max,
        f_p_alpha_plasma_deposited,
        p_alpha_total_mw,
        p_non_alpha_charged_mw,
        p_plasma_ohmic_mw,
        p_hcd_injected_total_mw,
    )


def calculate_radiated_wall_load_first_wall_area_comprehensive_2014(
    fhole,
    a_fw_total,
    f_a_fw_outboard_hcd,
    f_ster_div_single,
    p_plasma_rad_mw,
    f_fw_rad_max,
    f_p_alpha_plasma_deposited,
    p_alpha_total_mw,
    p_non_alpha_charged_mw,
    p_plasma_ohmic_mw,
    p_hcd_injected_total_mw,
):
    """`calculate_radiated_wall_load_and_fraction`'s
    `(FIRST_WALL_AREA, COMPREHENSIVE_2014)` arm."""
    return _radiated_wall_load(
        _wall_load_first_wall_area_comprehensive_2014(
            p_plasma_rad_mw, fhole, a_fw_total, f_a_fw_outboard_hcd, f_ster_div_single
        ),
        p_plasma_rad_mw,
        f_fw_rad_max,
        f_p_alpha_plasma_deposited,
        p_alpha_total_mw,
        p_non_alpha_charged_mw,
        p_plasma_ohmic_mw,
        p_hcd_injected_total_mw,
    )


def _radiated_wall_load(
    pflux_fw_rad_mw,
    p_plasma_rad_mw,
    f_fw_rad_max,
    f_p_alpha_plasma_deposited,
    p_alpha_total_mw,
    p_non_alpha_charged_mw,
    p_plasma_ohmic_mw,
    p_hcd_injected_total_mw,
):
    """The constraint bound and the total radiated fraction, which every arm shares
    given its own photon flux.
    """
    pflux_fw_rad_max_mw = pflux_fw_rad_mw * f_fw_rad_max

    rad_fraction_total = p_plasma_rad_mw / (
        f_p_alpha_plasma_deposited * p_alpha_total_mw
        + p_non_alpha_charged_mw
        + p_plasma_ohmic_mw
        + p_hcd_injected_total_mw
    )

    return pflux_fw_rad_mw, pflux_fw_rad_max_mw, rad_fraction_total


def calculate_thermal_energy_totals(
    eden_plasma_electrons_thermal_vol_avg,
    eden_plasma_ions_thermal_vol_avg,
    e_plasma_electrons_thermal,
    e_plasma_ions_thermal,
):
    """Combined electron+ion stored thermal energy density and total.

    Ports `stellarator.py:2282-2290`. Trivial sums of the already-ported
    `ElectronThermalEnergy`/`IonThermalEnergy` (`models/physics/
    pure_formulas.py`) outputs.
    """
    eden_plasma_thermal_vol_avg = (
        eden_plasma_electrons_thermal_vol_avg + eden_plasma_ions_thermal_vol_avg
    )
    e_plasma_thermal_total = e_plasma_electrons_thermal + e_plasma_ions_thermal
    return eden_plasma_thermal_vol_avg, e_plasma_thermal_total


class TotalField(ExplicitFunction):
    """cottax node: `calculate_total_field`, ports declared."""

    b_plasma_total = OutputInto(physics)

    def __call__(
        self,
        b_plasma_toroidal_on_axis=From(physics),
        b_plasma_surface_poloidal_average=From(physics),
    ):
        return calculate_total_field(
            b_plasma_toroidal_on_axis, b_plasma_surface_poloidal_average
        )


class PoloidalFieldFromRotationalTransform(ExplicitFunction):
    """cottax node: `calculate_poloidal_field_from_rotational_transform`, ports
    declared.
    """

    b_plasma_surface_poloidal_average = OutputInto(physics)

    def __call__(
        self,
        rminor=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
        rmajor=From(physics),
        iotabar=From(stellarator),
    ):
        return calculate_poloidal_field_from_rotational_transform(
            rminor, b_plasma_toroidal_on_axis, rmajor, iotabar
        )


class StellaratorBetaAndRhoStar(ExplicitFunction):
    """cottax node: `calculate_stellarator_beta_and_rho_star`, ports declared."""

    beta_total_vol_avg = OutputInto(physics)
    e_plasma_beta = OutputInto(physics)
    rho_star = OutputInto(physics)

    def __call__(
        self,
        beta_fast_alpha=From(physics),
        beta_beam=From(physics),
        nd_plasma_electrons_vol_avg=From(physics),
        temp_plasma_electron_density_weighted_kev=From(physics),
        nd_plasma_ions_total_vol_avg=From(physics),
        temp_plasma_ion_density_weighted_kev=From(physics),
        b_plasma_total=From(physics),
        vol_plasma=From(physics),
        m_ions_total_amu=From(physics),
        nd_plasma_electron_line=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
        eps=From(physics),
        rmajor=From(physics),
    ):
        return calculate_stellarator_beta_and_rho_star(
            beta_fast_alpha,
            beta_beam,
            nd_plasma_electrons_vol_avg,
            temp_plasma_electron_density_weighted_kev,
            nd_plasma_ions_total_vol_avg,
            temp_plasma_ion_density_weighted_kev,
            b_plasma_total,
            vol_plasma,
            m_ions_total_amu,
            nd_plasma_electron_line,
            b_plasma_toroidal_on_axis,
            eps,
            rmajor,
        )


class FusionPowerTotalsMw(ExplicitFunction):
    """cottax node: `calculate_fusion_power_totals_mw`, ports declared."""

    p_plasma_dt_mw = OutputInto(physics)
    p_dhe3_total_mw = OutputInto(physics)
    p_dd_total_mw = OutputInto(physics)

    def __call__(
        self,
        dt_power_density_plasma=From(physics),
        dhe3_power_density=From(physics),
        dd_power_density=From(physics),
        vol_plasma=From(physics),
    ):
        return calculate_fusion_power_totals_mw(
            dt_power_density_plasma, dhe3_power_density, dd_power_density, vol_plasma
        )


class FusionTotalsNoBeam(ExplicitFunction):
    """cottax node: `calculate_fusion_totals_no_beam`, ports declared.

    Registered unconditionally, and the reason is stronger than "this run happens to
    have no beam": PROCESS's gate is `p_hcd_beam_injected_total_mw != 0` **and**
    `i_plasma_ignited == NON_IGNITED` (`stellarator.py:2005-2009`), and this run is
    IGNITED (`stellarator_helias.IN.DAT:126`, the same value
    `HeatingAndRadiationPower(i_plasma_ignited=1)` is registered with). An ignited
    plasma takes this arm whatever the beam power is. The other arm cannot be written
    at all (see the function's docstring), so a `Switch` -- which needs two arms --
    is not available even in principle here.
    """

    fusden_total = OutputInto(physics)
    fusden_alpha_total = OutputInto(physics)
    p_dt_total_mw = OutputInto(physics)

    def __call__(
        self,
        fusden_plasma=From(physics),
        fusden_plasma_alpha=From(physics),
        p_plasma_dt_mw=From(physics),
    ):
        return calculate_fusion_totals_no_beam(
            fusden_plasma, fusden_plasma_alpha, p_plasma_dt_mw
        )


class ClippedRadiationPowers(ExplicitFunction):
    """cottax node: `calculate_clipped_radiation_powers`, ports declared.

    Owns the two real clipped `.physics.pden_plasma_*_rad_mw` fields (which
    `PlasmaRadiationPowers` used to claim while computing the pre-clip value) and gives
    `.physics.p_plasma_inner_rad_mw` its first producer. See the function's docstring.
    """

    pden_plasma_core_rad_mw = OutputInto(physics)
    pden_plasma_outer_rad_mw = OutputInto(physics)
    p_plasma_inner_rad_mw = OutputInto(physics)
    p_plasma_outer_rad_mw = OutputInto(physics)

    def __call__(
        self,
        pden_plasma_core_rad_mw_unclipped=From(physics),
        pden_plasma_outer_rad_mw_unclipped=From(physics),
        vol_plasma=From(physics),
    ):
        return calculate_clipped_radiation_powers(
            pden_plasma_core_rad_mw_unclipped,
            pden_plasma_outer_rad_mw_unclipped,
            vol_plasma,
        )


class NeutronWallLoad(ExplicitFunction):
    """The `calculate_neutron_wall_load` family -- one occupant per arm of the
    `.physics.i_pflux_fw_neutron` x `.heat_transport.ipowerflow` dispatch.

    **Both switches were `eqx.field(static=True)` here and both are gone**
    (`_audit/next_steps.md` §14.2). The three arms read genuinely different fields, so
    the single node declared **four dead reads** at this machine's own values --
    `.fwbs.fhole`, `.first_wall.a_fw_total`, `.fwbs.f_a_fw_outboard_hcd` and
    `.fwbs.f_ster_div_single`. `.first_wall.a_fw_total` is `stellarator.fw_area`'s own
    output, so the invented edge crossed a slot the factory already resolves: this is
    the only place in `switch_kwarg_survey.md` band (b) where that happens.
    """

    pflux_fw_neutron_mw = OutputInto(physics)


class NeutronWallLoadScaledPlasmaSurface(NeutronWallLoad):
    """`i_pflux_fw_neutron == SCALED_PLASMA_SURFACE_AREA` (1) -- PROCESS's own default
    (`physics_variables.py:1006`) and the reference run's. Reads no first-wall field.
    """

    def __call__(
        self,
        ffwal=From(physics),
        p_neutron_total_mw=From(physics),
        a_plasma_surface=From(physics),
    ):
        return calculate_neutron_wall_load_scaled_plasma_surface(
            ffwal, p_neutron_total_mw, a_plasma_surface
        )


class NeutronWallLoadFirstWallAreaPre2014(NeutronWallLoad):
    """`i_pflux_fw_neutron == FIRST_WALL_AREA` (2) with `ipowerflow == PRE_2014` (0).

    Reads `.fwbs.fhole` and `.first_wall.a_fw_total`, and neither `.physics.ffwal` nor
    `.physics.a_plasma_surface`.
    """

    def __call__(
        self,
        p_neutron_total_mw=From(physics),
        fhole=From(fwbs),
        a_fw_total=From(first_wall),
    ):
        return calculate_neutron_wall_load_first_wall_area_pre_2014(
            p_neutron_total_mw, fhole, a_fw_total
        )


class NeutronWallLoadFirstWallAreaComprehensive2014(NeutronWallLoad):
    """`i_pflux_fw_neutron == FIRST_WALL_AREA` (2) with
    `ipowerflow == COMPREHENSIVE_2014` (1) -- PROCESS's own `ipowerflow` default.

    Its sibling's two reads plus `.fwbs.f_a_fw_outboard_hcd` and
    `.fwbs.f_ster_div_single`.
    """

    def __call__(
        self,
        p_neutron_total_mw=From(physics),
        fhole=From(fwbs),
        a_fw_total=From(first_wall),
        f_a_fw_outboard_hcd=From(fwbs),
        f_ster_div_single=From(fwbs),
    ):
        return calculate_neutron_wall_load_first_wall_area_comprehensive_2014(
            p_neutron_total_mw,
            fhole,
            a_fw_total,
            f_a_fw_outboard_hcd,
            f_ster_div_single,
        )


class HeatingAndRadiationPower(ExplicitFunction):
    """The `calculate_heating_and_radiation_power` family -- one occupant per
    `.physics.i_plasma_ignited` value.

    **`i_plasma_ignited` was an `eqx.field(static=True)` here and is gone**
    (`_audit/next_steps.md` §14.2). The ignited arm adds no injected heating, so the
    single node declared `.current_drive.p_hcd_injected_total_mw` -- a
    `.current_drive -> .physics` edge the reference run does not make. It is the same
    read, for the same reason, that the confinement split removed from its own head
    (§14.3).
    """

    p_plasma_rad_mw = OutputInto(physics)
    psolradmw = OutputInto(physics)
    p_plasma_separatrix_mw = OutputInto(physics)
    p_fw_alpha_mw = OutputInto(physics)


class HeatingAndRadiationPowerIgnited(HeatingAndRadiationPower):
    """`i_plasma_ignited == IGNITED` (1) -- the reference run's.

    **One read leaves with this occupant**: `.current_drive.p_hcd_injected_total_mw`.
    """

    def __call__(
        self,
        f_p_alpha_plasma_deposited=From(physics),
        p_alpha_total_mw=From(physics),
        p_non_alpha_charged_mw=From(physics),
        p_plasma_ohmic_mw=From(physics),
        pden_plasma_rad_mw=From(physics),
        vol_plasma=From(physics),
        f_rad=From(stellarator),
    ):
        return calculate_heating_and_radiation_power_ignited(
            f_p_alpha_plasma_deposited,
            p_alpha_total_mw,
            p_non_alpha_charged_mw,
            p_plasma_ohmic_mw,
            pden_plasma_rad_mw,
            vol_plasma,
            f_rad,
        )


class HeatingAndRadiationPowerNonIgnited(HeatingAndRadiationPower):
    """`i_plasma_ignited == NON_IGNITED` (0) -- PROCESS's own default
    (`physics_variables.py:881`).
    """

    def __call__(
        self,
        f_p_alpha_plasma_deposited=From(physics),
        p_alpha_total_mw=From(physics),
        p_non_alpha_charged_mw=From(physics),
        p_plasma_ohmic_mw=From(physics),
        pden_plasma_rad_mw=From(physics),
        vol_plasma=From(physics),
        f_rad=From(stellarator),
        p_hcd_injected_total_mw=From(current_drive),
    ):
        return calculate_heating_and_radiation_power_non_ignited(
            f_p_alpha_plasma_deposited,
            p_alpha_total_mw,
            p_non_alpha_charged_mw,
            p_plasma_ohmic_mw,
            pden_plasma_rad_mw,
            vol_plasma,
            f_rad,
            p_hcd_injected_total_mw,
        )


class RadiatedWallLoadAndFraction(ExplicitFunction):
    """The `calculate_radiated_wall_load_and_fraction` family -- the same three arms as
    `NeutronWallLoad`, applied to `.physics.p_plasma_rad_mw`.

    **Both switches were `eqx.field(static=True)` here and both are gone**
    (`_audit/next_steps.md` §14.2), and the same four reads are dead at this machine's
    values; see `NeutronWallLoad`'s docstring. One dispatch serves both slots -- they
    are one family read twice, not two -- which is `indat.py`'s `_wall_load_arm`.
    """

    pflux_fw_rad_mw = OutputInto(physics)
    pflux_fw_rad_max_mw = OutputInto(constraints)
    rad_fraction_total = OutputInto(physics)


class RadiatedWallLoadScaledPlasmaSurface(RadiatedWallLoadAndFraction):
    """`i_pflux_fw_neutron == SCALED_PLASMA_SURFACE_AREA` (1) -- the reference run's.
    Reads no first-wall field.
    """

    def __call__(
        self,
        ffwal=From(physics),
        a_plasma_surface=From(physics),
        p_plasma_rad_mw=From(physics),
        f_fw_rad_max=From(constraints),
        f_p_alpha_plasma_deposited=From(physics),
        p_alpha_total_mw=From(physics),
        p_non_alpha_charged_mw=From(physics),
        p_plasma_ohmic_mw=From(physics),
        p_hcd_injected_total_mw=From(current_drive),
    ):
        return calculate_radiated_wall_load_scaled_plasma_surface(
            ffwal,
            a_plasma_surface,
            p_plasma_rad_mw,
            f_fw_rad_max,
            f_p_alpha_plasma_deposited,
            p_alpha_total_mw,
            p_non_alpha_charged_mw,
            p_plasma_ohmic_mw,
            p_hcd_injected_total_mw,
        )


class RadiatedWallLoadFirstWallAreaPre2014(RadiatedWallLoadAndFraction):
    """`i_pflux_fw_neutron == FIRST_WALL_AREA` (2) with `ipowerflow == PRE_2014` (0)."""

    def __call__(
        self,
        fhole=From(fwbs),
        a_fw_total=From(first_wall),
        p_plasma_rad_mw=From(physics),
        f_fw_rad_max=From(constraints),
        f_p_alpha_plasma_deposited=From(physics),
        p_alpha_total_mw=From(physics),
        p_non_alpha_charged_mw=From(physics),
        p_plasma_ohmic_mw=From(physics),
        p_hcd_injected_total_mw=From(current_drive),
    ):
        return calculate_radiated_wall_load_first_wall_area_pre_2014(
            fhole,
            a_fw_total,
            p_plasma_rad_mw,
            f_fw_rad_max,
            f_p_alpha_plasma_deposited,
            p_alpha_total_mw,
            p_non_alpha_charged_mw,
            p_plasma_ohmic_mw,
            p_hcd_injected_total_mw,
        )


class RadiatedWallLoadFirstWallAreaComprehensive2014(RadiatedWallLoadAndFraction):
    """`i_pflux_fw_neutron == FIRST_WALL_AREA` (2) with
    `ipowerflow == COMPREHENSIVE_2014` (1).
    """

    def __call__(
        self,
        fhole=From(fwbs),
        a_fw_total=From(first_wall),
        f_a_fw_outboard_hcd=From(fwbs),
        f_ster_div_single=From(fwbs),
        p_plasma_rad_mw=From(physics),
        f_fw_rad_max=From(constraints),
        f_p_alpha_plasma_deposited=From(physics),
        p_alpha_total_mw=From(physics),
        p_non_alpha_charged_mw=From(physics),
        p_plasma_ohmic_mw=From(physics),
        p_hcd_injected_total_mw=From(current_drive),
    ):
        return calculate_radiated_wall_load_first_wall_area_comprehensive_2014(
            fhole,
            a_fw_total,
            f_a_fw_outboard_hcd,
            f_ster_div_single,
            p_plasma_rad_mw,
            f_fw_rad_max,
            f_p_alpha_plasma_deposited,
            p_alpha_total_mw,
            p_non_alpha_charged_mw,
            p_plasma_ohmic_mw,
            p_hcd_injected_total_mw,
        )


class ThermalEnergyTotals(ExplicitFunction):
    """cottax node: `calculate_thermal_energy_totals`, ports declared."""

    eden_plasma_thermal_vol_avg = OutputInto(physics)
    e_plasma_thermal_total = OutputInto(physics)

    def __call__(
        self,
        eden_plasma_electrons_thermal_vol_avg=From(physics),
        eden_plasma_ions_thermal_vol_avg=From(physics),
        e_plasma_electrons_thermal=From(physics),
        e_plasma_ions_thermal=From(physics),
    ):
        return calculate_thermal_energy_totals(
            eden_plasma_electrons_thermal_vol_avg,
            eden_plasma_ions_thermal_vol_avg,
            e_plasma_electrons_thermal,
            e_plasma_ions_thermal,
        )


def select_stellarator_beta_and_stored_energy(
    beta_fast_alpha,
    beta_beam,
    nd_plasma_electrons_vol_avg,
    temp_plasma_electron_density_weighted_kev,
    nd_plasma_ions_total_vol_avg,
    temp_plasma_ion_density_weighted_kev,
    b_plasma_total,
    vol_plasma,
    m_ions_total_amu,
    nd_plasma_electron_line,
    b_plasma_toroidal_on_axis,
    eps,
    rmajor,
):
    """`(beta_total_vol_avg, e_plasma_beta)` half of
    `calculate_stellarator_beta_and_rho_star` -- `rho_star` is `DimensionlessPlasma
    Parameters`' output (see `StellaratorBetaAndStoredEnergy`'s own docstring for why
    this class must not also produce it), so it is discarded here exactly as the
    declaration already discarded it.
    """
    beta_total_vol_avg, e_plasma_beta, _rho_star = (
        calculate_stellarator_beta_and_rho_star(
            beta_fast_alpha,
            beta_beam,
            nd_plasma_electrons_vol_avg,
            temp_plasma_electron_density_weighted_kev,
            nd_plasma_ions_total_vol_avg,
            temp_plasma_ion_density_weighted_kev,
            b_plasma_total,
            vol_plasma,
            m_ions_total_amu,
            nd_plasma_electron_line,
            b_plasma_toroidal_on_axis,
            eps,
            rmajor,
        )
    )
    return beta_total_vol_avg, e_plasma_beta


class StellaratorBetaAndStoredEnergy(ExplicitFunction):
    """cottax node: `calculate_stellarator_beta_and_rho_star` minus its `rho_star`
    output -- the registerable form of `StellaratorBetaAndRhoStar` above.

    `StellaratorBetaAndRhoStar` cannot be registered alongside
    `DimensionlessPlasmaParameters`: both own `.physics.rho_star`, and `Graph`
    refuses two producers for one variable. That is a **redundant duplicate write in
    PROCESS itself** -- `st_phys` and `outplas` compute `rho_star` from the same
    inputs by the same expression -- not a modelling disagreement, so dropping one of
    the two writes loses nothing. `total_process.py` used to drop the whole node,
    which also cost `.physics.beta_total_vol_avg` and `.physics.e_plasma_beta` their
    only producer; this class drops only the redundant output.

    The two surviving outputs are computed by the same pure function, whose third
    return value is discarded here (XLA eliminates the dead `sqrt` -- nothing else in
    the body feeds `beta_total_vol_avg`/`e_plasma_beta` from it). Splitting the pure
    function in two was rejected: `rho_star`'s formula shares no sub-expression with
    the other two, so a split would buy nothing the discard does not, and would
    invalidate `plasma_physics.md`'s data-footprint row for a function PROCESS
    genuinely computes as one block.

    `.physics.beta_total_vol_avg` is constraint 24's only argument
    (`core/solver/constraints.py`'s `constraint_24`), one of the 14 active
    constraints of `stellarator_helias.IN.DAT`; without this node that constraint had
    no live argument and could not be assembled into an `Optimise` at all.
    """

    beta_total_vol_avg = OutputInto(physics)
    e_plasma_beta = OutputInto(physics)

    def __call__(
        self,
        beta_fast_alpha=From(physics),
        beta_beam=From(physics),
        nd_plasma_electrons_vol_avg=From(physics),
        temp_plasma_electron_density_weighted_kev=From(physics),
        nd_plasma_ions_total_vol_avg=From(physics),
        temp_plasma_ion_density_weighted_kev=From(physics),
        b_plasma_total=From(physics),
        vol_plasma=From(physics),
        m_ions_total_amu=From(physics),
        nd_plasma_electron_line=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
        eps=From(physics),
        rmajor=From(physics),
    ):
        return select_stellarator_beta_and_stored_energy(
            beta_fast_alpha,
            beta_beam,
            nd_plasma_electrons_vol_avg,
            temp_plasma_electron_density_weighted_kev,
            nd_plasma_ions_total_vol_avg,
            temp_plasma_ion_density_weighted_kev,
            b_plasma_total,
            vol_plasma,
            m_ions_total_amu,
            nd_plasma_electron_line,
            b_plasma_toroidal_on_axis,
            eps,
            rmajor,
        )
