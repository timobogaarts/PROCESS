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

import jax.numpy as jnp  # noqa: F401
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    OutputInto,
)

from functional_process.models.safe_math import (
    safe_sqrt,  # noqa: F401
)
from functional_process.cottax.paths import (
    constraints,
    current_drive,
    first_wall,
    fwbs,
    physics,
    stellarator,
)
from functional_process.models.stellarator.plasma_physics import (
    calculate_clipped_radiation_powers,
    calculate_fusion_power_totals_mw,
    calculate_fusion_totals_no_beam,
    calculate_heating_and_radiation_power,  # noqa: F401
    calculate_heating_and_radiation_power_ignited,
    calculate_heating_and_radiation_power_non_ignited,
    calculate_neutron_wall_load,  # noqa: F401
    calculate_neutron_wall_load_first_wall_area_comprehensive_2014,
    calculate_neutron_wall_load_first_wall_area_pre_2014,
    calculate_neutron_wall_load_scaled_plasma_surface,
    calculate_poloidal_field_from_rotational_transform,
    calculate_radiated_wall_load_and_fraction,  # noqa: F401
    calculate_radiated_wall_load_first_wall_area_comprehensive_2014,
    calculate_radiated_wall_load_first_wall_area_pre_2014,
    calculate_radiated_wall_load_scaled_plasma_surface,
    calculate_stellarator_beta_and_rho_star,
    calculate_thermal_energy_totals,
    calculate_total_field,
    select_stellarator_beta_and_stored_energy,
)
from functional_process.vocabulary import (
    constants,  # noqa: F401
)


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
