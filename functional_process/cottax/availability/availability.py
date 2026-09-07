"""Pure-functional port of `process/models/availability.py` (registry unit #17)."""

import math  # noqa: F401

import equinox as eqx
import jax.numpy as jnp  # noqa: F401
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    FixedPointFunction,
    From,
    OutputInto,
)

from functional_process.models.availability.availability import (
    DAY_SECONDS,  # noqa: F401
    DAYS_IN_YEAR,  # noqa: F401
    YEAR_SECONDS,  # noqa: F401
    blanket_lifetime_fpy_displacements_per_atom,  # noqa: F401
    blanket_lifetime_fpy_neutron_fluence,  # noqa: F401
    calculate_avail,  # noqa: F401
    calculate_avail_2,
    calculate_avail_displacements_per_atom,
    calculate_avail_neutron_fluence,
    calculate_avail_st,
    calculate_blanket_lifetime_fpy_avail,  # noqa: F401
    calculate_blanket_lifetime_fpy_simple,  # noqa: F401
    calculate_cp_lifetime_resistive,
    calculate_cp_lifetime_superconducting,
    calculate_cplife_avail_st_next,
    calculate_cplife_lifetime_adjustment,  # noqa: F401
    calculate_cplife_next,  # noqa: F401
    calculate_cplife_resistive,
    calculate_cplife_superconducting,
    calculate_divertor_lifetime,  # noqa: F401
    calculate_dpa_per_fpy,  # noqa: F401
    calculate_redun_vac,  # noqa: F401
    calculate_u_planned,  # noqa: F401
    calculate_u_unplanned_bop,  # noqa: F401
    calculate_u_unplanned_divertor,  # noqa: F401
    calculate_u_unplanned_fwbs,  # noqa: F401
    calculate_u_unplanned_hcd,  # noqa: F401
    calculate_u_unplanned_magnets,  # noqa: F401
    calculate_u_unplanned_vacuum,  # noqa: F401
    calculate_ward_taylor_availability,
    unset_life,  # noqa: F401
)
from functional_process.models.switch_enums import (
    BlanketLifetimeModel,
    SphericalTokamakModel,
)
from functional_process.cottax.paths import (
    constraints,
    costs,
    divertor,
    fwbs,
    physics,
    tfcoil,
    times,
)
from functional_process.vocabulary import TFConductorModel

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
# shape already used in `tf_nuclear_heating.py`.
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
    """

    cplife = OutputInto(costs)

    def __call__(
        self,
        neut_flux_cp=From(fwbs),
        flu_tf_neutron_fast_max=From(constraints),
        life_plant=From(costs),
    ):
        return calculate_cp_lifetime_superconducting(
            neut_flux_cp, flu_tf_neutron_fast_max, life_plant
        )


class CpLifetimeResistive(ExplicitFunction):
    """cottax node: `calculate_cp_lifetime_resistive`, unchanged, ports declared."""

    cplife = OutputInto(costs)

    def __call__(
        self,
        cpstflnc=From(costs),
        pflux_fw_neutron_mw=From(physics),
        life_plant=From(costs),
    ):
        return calculate_cp_lifetime_resistive(cpstflnc, pflux_fw_neutron_mw, life_plant)


class WardTaylorAvailability(ExplicitFunction):
    """cottax node: `calculate_ward_taylor_availability`, unchanged, ports declared."""

    f_t_plant_available = OutputInto(costs)

    def __call__(
        self,
        life_div_fpy=From(costs),
        life_blkt_fpy=From(fwbs),
        t_div_replace_yrs=From(costs),
        t_blkt_replace_yrs=From(costs),
        tcomrepl=From(costs),
        uubop=From(costs),
        uucd=From(costs),
        uudiv=From(costs),
        uufuel=From(costs),
        uufw=From(costs),
        uumag=From(costs),
        uuves=From(costs),
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


class CplifeAvail(ExplicitFunction):
    """The `.costs.cplife` family for `Avail`/`Avail2` -- one occupant per arm of
    `.physics.itart` x `.tfcoil.i_tf_sup`.
    """

    cplife = OutputInto(costs)


class CplifeAvailSuperconducting(CplifeAvail):
    """`itart == 1` with `i_tf_sup == SUPERCONDUCTING` (1): the centrepost lasts until
    its fast-neutron fluence limit, then adjusted for plant availability.
    """

    def __call__(
        self,
        neut_flux_cp=From(fwbs),
        flu_tf_neutron_fast_max=From(constraints),
        life_plant=From(costs),
        f_t_plant_available=From(costs),
    ):
        return calculate_cplife_superconducting(
            neut_flux_cp, flu_tf_neutron_fast_max, life_plant, f_t_plant_available
        )


class CplifeAvailResistive(CplifeAvail):
    """`itart == 1` with `i_tf_sup != SUPERCONDUCTING`: the centrepost lasts until its
    allowable stress fluence is spent, then adjusted for plant availability.
    """

    def __call__(
        self,
        cpstflnc=From(costs),
        pflux_fw_neutron_mw=From(physics),
        life_plant=From(costs),
        f_t_plant_available=From(costs),
    ):
        return calculate_cplife_resistive(
            cpstflnc, pflux_fw_neutron_mw, life_plant, f_t_plant_available
        )


class CplifeAvailSt(FixedPointFunction):
    """cottax node: `.costs.cplife`'s Shape B self-reference in `AvailSt`
    (`next_steps.md` §5), split out as a `FixedPointFunction`.
    """

    i_tf_sup: TFConductorModel = eqx.field(static=True)
    itart: SphericalTokamakModel = eqx.field(static=True)

    cplife = OutputInto(costs)

    def step(
        self,
        neut_flux_cp=From(fwbs),
        flu_tf_neutron_fast_max=From(constraints),
        cpstflnc=From(costs),
        pflux_fw_neutron_mw=From(physics),
        life_plant=From(costs),
        f_t_plant_available=From(costs),
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
    """The `calculate_avail` family -- `calculate_avail`'s outputs *other* than
    `.costs.cplife`, one occupant per `.costs.ibkt_life` value.
    """

    life_blkt_fpy = OutputInto(fwbs)
    life_div_fpy = OutputInto(costs)
    bktcycles = OutputInto(costs)
    cpfact = OutputInto(costs)
    life_hcd_fpy = OutputInto(costs)


class AvailNeutronFluence(Avail):
    """`ibkt_life == NEUTRON_FLUENCE` (0) -- PROCESS's own default
    (`cost_variables.py:416`) and the reference run's.
    """

    def __call__(
        self,
        life_fw_fpy=From(fwbs),
        abktflnc=From(costs),
        pflux_fw_neutron_mw=From(physics),
        life_plant=From(costs),
        pflux_div_heat_load_mw=From(divertor),
        adivflnc=From(costs),
        t_plant_pulse_total=From(times),
        t_plant_pulse_burn=From(times),
        f_t_plant_available=From(costs),
    ):
        return calculate_avail_neutron_fluence(
            life_fw_fpy,
            abktflnc,
            pflux_fw_neutron_mw,
            life_plant,
            pflux_div_heat_load_mw,
            adivflnc,
            t_plant_pulse_total,
            t_plant_pulse_burn,
            f_t_plant_available,
        )


class AvailDisplacementsPerAtom(Avail):
    """`ibkt_life == FUSION_POWER` (1) -- the blanket lifetime set by displacement
    damage per full-power year.
    """

    def __call__(
        self,
        life_fw_fpy=From(fwbs),
        p_fusion_total_mw=From(physics),
        life_dpa=From(costs),
        life_plant=From(costs),
        pflux_div_heat_load_mw=From(divertor),
        adivflnc=From(costs),
        t_plant_pulse_total=From(times),
        t_plant_pulse_burn=From(times),
        f_t_plant_available=From(costs),
    ):
        return calculate_avail_displacements_per_atom(
            life_fw_fpy,
            p_fusion_total_mw,
            life_dpa,
            life_plant,
            pflux_div_heat_load_mw,
            adivflnc,
            t_plant_pulse_total,
            t_plant_pulse_burn,
            f_t_plant_available,
        )


class Avail2(ExplicitFunction):
    """cottax node: `calculate_avail_2`'s outputs *other* than `.costs.cplife`,
    unchanged, ports declared, `u_planned`/`u_unplanned` dropped (no `VarPath` -- see
    the module-level note above).
    """

    ibkt_life: BlanketLifetimeModel = eqx.field(static=True)
    itart: SphericalTokamakModel = eqx.field(static=True)
    n_vac_pumps_high: int = eqx.field(static=True)
    redun_vac: int = eqx.field(static=True)

    life_blkt_fpy = OutputInto(fwbs)
    life_div_fpy = OutputInto(costs)
    life_hcd_fpy = OutputInto(costs)
    t_plant_operational_total_yrs = OutputInto(costs)
    f_t_plant_available = OutputInto(costs)
    cpfact = OutputInto(costs)

    def __call__(
        self,
        p_fusion_total_mw=From(physics),
        abktflnc=From(costs),
        pflux_fw_neutron_mw=From(physics),
        life_dpa=From(costs),
        adivflnc=From(costs),
        pflux_div_heat_load_mw=From(divertor),
        life_plant=From(costs),
        num_rh_systems=From(costs),
        temp_tf_superconductor_margin_min=From(tfcoil),
        temp_cs_superconductor_margin_min=From(tfcoil),
        conf_mag=From(costs),
        temp_margin=From(tfcoil),
        div_prob_fail=From(costs),
        div_umain_time=From(costs),
        div_nu=From(costs),
        div_nref=From(costs),
        fwbs_prob_fail=From(costs),
        fwbs_umain_time=From(costs),
        fwbs_nu=From(costs),
        fwbs_nref=From(costs),
        t_plant_pulse_burn=From(times),
        t_plant_pulse_total=From(times),
        cplife=From(costs),
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
    `u_planned`/`u_unplanned` dropped (no `VarPath`).
    """

    ibkt_life: BlanketLifetimeModel = eqx.field(static=True)
    itart: SphericalTokamakModel = eqx.field(static=True)
    n_vac_pumps_high: int = eqx.field(static=True)
    redun_vac: int = eqx.field(static=True)
    i_tf_sup: TFConductorModel = eqx.field(static=True)

    life_blkt_fpy = OutputInto(fwbs)
    life_div_fpy = OutputInto(costs)
    life_hcd_fpy = OutputInto(costs)
    t_plant_operational_total_yrs = OutputInto(costs)
    f_t_plant_available = OutputInto(costs)
    cpfact = OutputInto(costs)

    def __call__(
        self,
        abktflnc=From(costs),
        pflux_fw_neutron_mw=From(physics),
        life_dpa=From(costs),
        p_fusion_total_mw=From(physics),
        adivflnc=From(costs),
        pflux_div_heat_load_mw=From(divertor),
        life_plant=From(costs),
        neut_flux_cp=From(fwbs),
        flu_tf_neutron_fast_max=From(constraints),
        cpstflnc=From(costs),
        tmain=From(costs),
        temp_tf_superconductor_margin_min=From(tfcoil),
        temp_cs_superconductor_margin_min=From(tfcoil),
        conf_mag=From(costs),
        temp_margin=From(tfcoil),
        div_prob_fail=From(costs),
        div_umain_time=From(costs),
        div_nu=From(costs),
        div_nref=From(costs),
        fwbs_prob_fail=From(costs),
        fwbs_umain_time=From(costs),
        fwbs_nu=From(costs),
        fwbs_nref=From(costs),
        num_rh_systems=From(costs),
        u_unplanned_cp=From(costs),
        t_plant_pulse_burn=From(times),
        t_plant_pulse_total=From(times),
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
