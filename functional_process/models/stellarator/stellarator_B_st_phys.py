"""Pure-functional port of `Stellarator.st_phys`'s genuinely-new sub-computations
(chunk 1B of unit #1). Audit record: `stellarator_B_st_phys.md` -- read it first.

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
   dependency check that confirms it, and `stellarator_B_st_phys.md`'s "New findings"
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
`FastAlphaBeta.i_beta_fast_alpha` (`models/physics/physics_A_pure_formulas.py`) --
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

import equinox as eqx
import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, Input, Output

from process.core import constants


def calculate_total_field(b_plasma_toroidal_on_axis, b_plasma_surface_poloidal_average):
    """Total field magnitude from toroidal and poloidal components.

    Ports `stellarator.py:1916-1919`. `b_plasma_surface_poloidal_average` here is
    whatever value the field currently holds -- see this module's docstring on why that
    is a separate node from `calculate_poloidal_field_from_rotational_transform`, which
    produces the *next* value of the same `VarPath`.
    """
    return jnp.sqrt(b_plasma_toroidal_on_axis**2 + b_plasma_surface_poloidal_average**2)


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
      (`models/physics/physics_A_pure_formulas.py`, already registered in
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
    rho_star = jnp.sqrt(
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
        return ffwal * p_neutron_total_mw / a_plasma_surface
    if ipowerflow == 0:
        return (1.0e0 - fhole) * p_neutron_total_mw / a_fw_total
    return (
        (1.0e0 - fhole - f_a_fw_outboard_hcd - f_ster_div_single)
        * p_neutron_total_mw
        / a_fw_total
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
    p_plasma_rad_mw_raw = pden_plasma_rad_mw * vol_plasma

    powht = (
        f_p_alpha_plasma_deposited * p_alpha_total_mw
        + p_non_alpha_charged_mw
        + p_plasma_ohmic_mw
        - p_plasma_rad_mw_raw
    )
    powht = jnp.maximum(0.00001e0, powht)  # noqa: PLR2004 -- ported literal, see source

    if i_plasma_ignited == 0:  # PlasmaIgnitionModel.NON_IGNITED
        powht = powht + p_hcd_injected_total_mw

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
    if i_pflux_fw_neutron == 1:
        pflux_fw_rad_mw = ffwal * p_plasma_rad_mw / a_plasma_surface
    elif ipowerflow == 0:
        pflux_fw_rad_mw = (1.0e0 - fhole) * p_plasma_rad_mw / a_fw_total
    else:
        pflux_fw_rad_mw = (
            (1.0e0 - fhole - f_a_fw_outboard_hcd - f_ster_div_single)
            * p_plasma_rad_mw
            / a_fw_total
        )

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
    physics_A_pure_formulas.py`) outputs.
    """
    eden_plasma_thermal_vol_avg = (
        eden_plasma_electrons_thermal_vol_avg + eden_plasma_ions_thermal_vol_avg
    )
    e_plasma_thermal_total = e_plasma_electrons_thermal + e_plasma_ions_thermal
    return eden_plasma_thermal_vol_avg, e_plasma_thermal_total


class TotalField(ExplicitFunction):
    """cottax node: `calculate_total_field`, ports declared."""

    b_plasma_total = Output(lambda s: s.physics.b_plasma_total)

    def __call__(
        self,
        b_plasma_toroidal_on_axis=Input(lambda s: s.physics.b_plasma_toroidal_on_axis),
        b_plasma_surface_poloidal_average=Input(
            lambda s: s.physics.b_plasma_surface_poloidal_average
        ),
    ):
        return calculate_total_field(
            b_plasma_toroidal_on_axis, b_plasma_surface_poloidal_average
        )


class PoloidalFieldFromRotationalTransform(ExplicitFunction):
    """cottax node: `calculate_poloidal_field_from_rotational_transform`, ports
    declared."""

    b_plasma_surface_poloidal_average = Output(
        lambda s: s.physics.b_plasma_surface_poloidal_average
    )

    def __call__(
        self,
        rminor=Input(lambda s: s.physics.rminor),
        b_plasma_toroidal_on_axis=Input(lambda s: s.physics.b_plasma_toroidal_on_axis),
        rmajor=Input(lambda s: s.physics.rmajor),
        iotabar=Input(lambda s: s.stellarator.iotabar),
    ):
        return calculate_poloidal_field_from_rotational_transform(
            rminor, b_plasma_toroidal_on_axis, rmajor, iotabar
        )


class StellaratorBetaAndRhoStar(ExplicitFunction):
    """cottax node: `calculate_stellarator_beta_and_rho_star`, ports declared."""

    beta_total_vol_avg = Output(lambda s: s.physics.beta_total_vol_avg)
    e_plasma_beta = Output(lambda s: s.physics.e_plasma_beta)
    rho_star = Output(lambda s: s.physics.rho_star)

    def __call__(
        self,
        beta_fast_alpha=Input(lambda s: s.physics.beta_fast_alpha),
        beta_beam=Input(lambda s: s.physics.beta_beam),
        nd_plasma_electrons_vol_avg=Input(
            lambda s: s.physics.nd_plasma_electrons_vol_avg
        ),
        temp_plasma_electron_density_weighted_kev=Input(
            lambda s: s.physics.temp_plasma_electron_density_weighted_kev
        ),
        nd_plasma_ions_total_vol_avg=Input(
            lambda s: s.physics.nd_plasma_ions_total_vol_avg
        ),
        temp_plasma_ion_density_weighted_kev=Input(
            lambda s: s.physics.temp_plasma_ion_density_weighted_kev
        ),
        b_plasma_total=Input(lambda s: s.physics.b_plasma_total),
        vol_plasma=Input(lambda s: s.physics.vol_plasma),
        m_ions_total_amu=Input(lambda s: s.physics.m_ions_total_amu),
        nd_plasma_electron_line=Input(lambda s: s.physics.nd_plasma_electron_line),
        b_plasma_toroidal_on_axis=Input(lambda s: s.physics.b_plasma_toroidal_on_axis),
        eps=Input(lambda s: s.physics.eps),
        rmajor=Input(lambda s: s.physics.rmajor),
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

    p_plasma_dt_mw = Output(lambda s: s.physics.p_plasma_dt_mw)
    p_dhe3_total_mw = Output(lambda s: s.physics.p_dhe3_total_mw)
    p_dd_total_mw = Output(lambda s: s.physics.p_dd_total_mw)

    def __call__(
        self,
        dt_power_density_plasma=Input(lambda s: s.physics.dt_power_density_plasma),
        dhe3_power_density=Input(lambda s: s.physics.dhe3_power_density),
        dd_power_density=Input(lambda s: s.physics.dd_power_density),
        vol_plasma=Input(lambda s: s.physics.vol_plasma),
    ):
        return calculate_fusion_power_totals_mw(
            dt_power_density_plasma, dhe3_power_density, dd_power_density, vol_plasma
        )


class NeutronWallLoad(ExplicitFunction):
    """cottax node: `calculate_neutron_wall_load`, ports declared.

    `i_pflux_fw_neutron`/`ipowerflow` static, per this module's docstring.
    """

    i_pflux_fw_neutron: int = eqx.field(static=True)
    ipowerflow: int = eqx.field(static=True)

    pflux_fw_neutron_mw = Output(lambda s: s.physics.pflux_fw_neutron_mw)

    def __call__(
        self,
        ffwal=Input(lambda s: s.physics.ffwal),
        p_neutron_total_mw=Input(lambda s: s.physics.p_neutron_total_mw),
        a_plasma_surface=Input(lambda s: s.physics.a_plasma_surface),
        fhole=Input(lambda s: s.fwbs.fhole),
        a_fw_total=Input(lambda s: s.first_wall.a_fw_total),
        f_a_fw_outboard_hcd=Input(lambda s: s.fwbs.f_a_fw_outboard_hcd),
        f_ster_div_single=Input(lambda s: s.fwbs.f_ster_div_single),
    ):
        return calculate_neutron_wall_load(
            self.i_pflux_fw_neutron,
            self.ipowerflow,
            ffwal,
            p_neutron_total_mw,
            a_plasma_surface,
            fhole,
            a_fw_total,
            f_a_fw_outboard_hcd,
            f_ster_div_single,
        )


class HeatingAndRadiationPower(ExplicitFunction):
    """cottax node: `calculate_heating_and_radiation_power`, ports declared.

    `i_plasma_ignited` static, per this module's docstring.
    """

    i_plasma_ignited: int = eqx.field(static=True)

    p_plasma_rad_mw = Output(lambda s: s.physics.p_plasma_rad_mw)
    psolradmw = Output(lambda s: s.physics.psolradmw)
    p_plasma_separatrix_mw = Output(lambda s: s.physics.p_plasma_separatrix_mw)
    p_fw_alpha_mw = Output(lambda s: s.physics.p_fw_alpha_mw)

    def __call__(
        self,
        f_p_alpha_plasma_deposited=Input(lambda s: s.physics.f_p_alpha_plasma_deposited),
        p_alpha_total_mw=Input(lambda s: s.physics.p_alpha_total_mw),
        p_non_alpha_charged_mw=Input(lambda s: s.physics.p_non_alpha_charged_mw),
        p_plasma_ohmic_mw=Input(lambda s: s.physics.p_plasma_ohmic_mw),
        pden_plasma_rad_mw=Input(lambda s: s.physics.pden_plasma_rad_mw),
        vol_plasma=Input(lambda s: s.physics.vol_plasma),
        p_hcd_injected_total_mw=Input(lambda s: s.current_drive.p_hcd_injected_total_mw),
        f_rad=Input(lambda s: s.stellarator.f_rad),
    ):
        return calculate_heating_and_radiation_power(
            f_p_alpha_plasma_deposited,
            p_alpha_total_mw,
            p_non_alpha_charged_mw,
            p_plasma_ohmic_mw,
            pden_plasma_rad_mw,
            vol_plasma,
            self.i_plasma_ignited,
            p_hcd_injected_total_mw,
            f_rad,
        )


class RadiatedWallLoadAndFraction(ExplicitFunction):
    """cottax node: `calculate_radiated_wall_load_and_fraction`, ports declared.

    `i_pflux_fw_neutron`/`ipowerflow` static, per this module's docstring.
    """

    i_pflux_fw_neutron: int = eqx.field(static=True)
    ipowerflow: int = eqx.field(static=True)

    pflux_fw_rad_mw = Output(lambda s: s.physics.pflux_fw_rad_mw)
    pflux_fw_rad_max_mw = Output(lambda s: s.constraints.pflux_fw_rad_max_mw)
    rad_fraction_total = Output(lambda s: s.physics.rad_fraction_total)

    def __call__(
        self,
        ffwal=Input(lambda s: s.physics.ffwal),
        p_plasma_rad_mw=Input(lambda s: s.physics.p_plasma_rad_mw),
        a_plasma_surface=Input(lambda s: s.physics.a_plasma_surface),
        fhole=Input(lambda s: s.fwbs.fhole),
        a_fw_total=Input(lambda s: s.first_wall.a_fw_total),
        f_a_fw_outboard_hcd=Input(lambda s: s.fwbs.f_a_fw_outboard_hcd),
        f_ster_div_single=Input(lambda s: s.fwbs.f_ster_div_single),
        f_fw_rad_max=Input(lambda s: s.constraints.f_fw_rad_max),
        f_p_alpha_plasma_deposited=Input(lambda s: s.physics.f_p_alpha_plasma_deposited),
        p_alpha_total_mw=Input(lambda s: s.physics.p_alpha_total_mw),
        p_non_alpha_charged_mw=Input(lambda s: s.physics.p_non_alpha_charged_mw),
        p_plasma_ohmic_mw=Input(lambda s: s.physics.p_plasma_ohmic_mw),
        p_hcd_injected_total_mw=Input(lambda s: s.current_drive.p_hcd_injected_total_mw),
    ):
        return calculate_radiated_wall_load_and_fraction(
            self.i_pflux_fw_neutron,
            self.ipowerflow,
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
        )


class ThermalEnergyTotals(ExplicitFunction):
    """cottax node: `calculate_thermal_energy_totals`, ports declared."""

    eden_plasma_thermal_vol_avg = Output(lambda s: s.physics.eden_plasma_thermal_vol_avg)
    e_plasma_thermal_total = Output(lambda s: s.physics.e_plasma_thermal_total)

    def __call__(
        self,
        eden_plasma_electrons_thermal_vol_avg=Input(
            lambda s: s.physics.eden_plasma_electrons_thermal_vol_avg
        ),
        eden_plasma_ions_thermal_vol_avg=Input(
            lambda s: s.physics.eden_plasma_ions_thermal_vol_avg
        ),
        e_plasma_electrons_thermal=Input(lambda s: s.physics.e_plasma_electrons_thermal),
        e_plasma_ions_thermal=Input(lambda s: s.physics.e_plasma_ions_thermal),
    ):
        return calculate_thermal_energy_totals(
            eden_plasma_electrons_thermal_vol_avg,
            eden_plasma_ions_thermal_vol_avg,
            e_plasma_electrons_thermal,
            e_plasma_ions_thermal,
        )


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
    invalidate `stellarator_B_st_phys.md`'s data-footprint row for a function PROCESS
    genuinely computes as one block.

    `.physics.beta_total_vol_avg` is constraint 24's only argument
    (`core/solver/constraints.py`'s `constraint_24`), one of the 14 active
    constraints of `stellarator_helias.IN.DAT`; without this node that constraint had
    no live argument and could not be assembled into an `Optimise` at all.
    """

    beta_total_vol_avg = Output(lambda s: s.physics.beta_total_vol_avg)
    e_plasma_beta = Output(lambda s: s.physics.e_plasma_beta)

    def __call__(
        self,
        beta_fast_alpha=Input(lambda s: s.physics.beta_fast_alpha),
        beta_beam=Input(lambda s: s.physics.beta_beam),
        nd_plasma_electrons_vol_avg=Input(
            lambda s: s.physics.nd_plasma_electrons_vol_avg
        ),
        temp_plasma_electron_density_weighted_kev=Input(
            lambda s: s.physics.temp_plasma_electron_density_weighted_kev
        ),
        nd_plasma_ions_total_vol_avg=Input(
            lambda s: s.physics.nd_plasma_ions_total_vol_avg
        ),
        temp_plasma_ion_density_weighted_kev=Input(
            lambda s: s.physics.temp_plasma_ion_density_weighted_kev
        ),
        b_plasma_total=Input(lambda s: s.physics.b_plasma_total),
        vol_plasma=Input(lambda s: s.physics.vol_plasma),
        m_ions_total_amu=Input(lambda s: s.physics.m_ions_total_amu),
        nd_plasma_electron_line=Input(lambda s: s.physics.nd_plasma_electron_line),
        b_plasma_toroidal_on_axis=Input(lambda s: s.physics.b_plasma_toroidal_on_axis),
        eps=Input(lambda s: s.physics.eps),
        rmajor=Input(lambda s: s.physics.rmajor),
    ):
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
