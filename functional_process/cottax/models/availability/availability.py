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

from functional_process.cottax.paths import (
    constraints,
    costs,
    divertor,
    fwbs,
    physics,
    tfcoil,
    times,
)
from functional_process.cottax.wraps import WrapsFunction
from functional_process.models.availability.availability import (
    DAY_SECONDS,  # noqa: F401
    DAYS_IN_YEAR,  # noqa: F401
    YEAR_SECONDS,  # noqa: F401
    blanket_lifetime_fpy_displacements_per_atom,  # noqa: F401
    blanket_lifetime_fpy_neutron_fluence,  # noqa: F401
    calculate_avail,  # noqa: F401
    calculate_avail_2,  # noqa: F401 -- re-exported for tests
    calculate_avail_2_dropping_extras,
    calculate_avail_displacements_per_atom,
    calculate_avail_neutron_fluence,
    calculate_avail_st,  # noqa: F401 -- re-exported for tests
    calculate_avail_st_dropping_extras,
    calculate_blanket_lifetime_fpy_avail,  # noqa: F401
    calculate_blanket_lifetime_fpy_simple,  # noqa: F401
    calculate_cp_lifetime_resistive,
    calculate_cp_lifetime_superconducting,
    calculate_cplife_avail_st_next,  # noqa: F401 -- re-exported for tests
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
# `Avail`/`Avail2`/`AvailSt` match PROCESS's own granularity (one `Model` method call
# producing every output at once) rather than atomising further -- nothing outside
# `Availability` ever calls `divertor_lifetime`, `calc_u_planned` etc. independently.
# Each is now a bodiless family base plus one occupant per combination of the switches
# its branch touches (`.costs.ibkt_life`, `.physics.itart`, and, for `AvailSt` only,
# `.tfcoil.i_tf_sup`) -- no occupant holds any of them as a static field (switches are
# not ports); each bakes its own combination to a literal when it calls its
# `functional_process/models/` helper. `CpLifetimeSuperconducting`/`CpLifetimeResistive`
# are a *different* kind of exception: `.costs.cplife` genuinely has two independent
# producers selected by `.tfcoil.i_tf_sup`, exactly the `i_tf_sup` shape already used in
# `tf_nuclear_heating.py`.
#
# `.costs.cplife` **also** self-references within `avail`/`avail_2`/`avail_st` themselves
# (Shape B, `next_steps.md` §5: a node whose own `Output` and `FromExactly` name the identical
# `VarPath`) -- `to_graph(Avail(...))` raised `ValueError: reads ['.costs.cplife'], which
# it also owns` directly from `cottax.pytree.spec`'s `__check_init__` before this was split.
# `CplifeAvail` (shared by `Avail`/`Avail2` -- their `itart == 1` cplife-adjustment
# formula is identical, see the audit record) and `CplifeAvailSt` isolate exactly that
# self-reference; both are now bodiless family bases too, each with one occupant per
# `i_tf_sup` value (`CplifeAvail`) or per `i_tf_sup` x `itart` pair (`CplifeAvailSt` --
# `itart` alone decides whether the `avail_st()` lifetime-adjustment step applies).
# `CplifeAvail`'s occupants are ordinary `ExplicitFunction`s (no self-read survives
# either `i_tf_sup` arm, so neither is a `FixedPoint` any more); `CplifeAvailSt` stays a
# `FixedPointFunction` family for every occupant, even though none of them reads a
# previous `.costs.cplife` either -- see its own docstring for why that shape is kept.
# `Avail`/`Avail2`/`AvailSt` themselves are ordinary `ExplicitFunction`s over each
# branch's *other* outputs only -- `.costs.cplife` is not one of their declared
# `Output`s. `CpLifetimeSuperconducting`/`CpLifetimeResistive` are left unconsumed by
# this split: both those nodes and the `CplifeAvail`/`CplifeAvailSt` families
# independently want to own `.costs.cplife`, and only one occupant of one family may in
# any graph that actually registers them together -- an open question left to whoever
# designs `total_process.py`'s wiring, not resolved here (registration is explicitly out
# of this split's scope).
# ---------------------------------------------------------------------------


class CpLifetimeSuperconducting(WrapsFunction):
    """cottax node: `calculate_cp_lifetime_superconducting`, unchanged, ports declared."""

    fn = calculate_cp_lifetime_superconducting

    neut_flux_cp = From(fwbs)
    flu_tf_neutron_fast_max = From(constraints)
    life_plant = From(costs)

    cplife = OutputInto(costs)


class CpLifetimeResistive(WrapsFunction):
    """cottax node: `calculate_cp_lifetime_resistive`, unchanged, ports declared."""

    fn = calculate_cp_lifetime_resistive

    cpstflnc = From(costs)
    pflux_fw_neutron_mw = From(physics)
    life_plant = From(costs)

    cplife = OutputInto(costs)


class WardTaylorAvailability(WrapsFunction):
    """cottax node: `calculate_ward_taylor_availability`, unchanged, ports declared."""

    fn = calculate_ward_taylor_availability

    life_div_fpy = From(costs)
    life_blkt_fpy = From(fwbs)
    t_div_replace_yrs = From(costs)
    t_blkt_replace_yrs = From(costs)
    tcomrepl = From(costs)
    uubop = From(costs)
    uucd = From(costs)
    uudiv = From(costs)
    uufuel = From(costs)
    uufw = From(costs)
    uumag = From(costs)
    uuves = From(costs)

    f_t_plant_available = OutputInto(costs)


class CplifeAvail(ExplicitFunction):
    """The `.costs.cplife` family for `Avail`/`Avail2` -- one occupant per arm of
    `.physics.itart` x `.tfcoil.i_tf_sup`.
    """

    cplife = OutputInto(costs)


class CplifeAvailSuperconducting(CplifeAvail, WrapsFunction):
    """`itart == 1` with `i_tf_sup == SUPERCONDUCTING` (1): the centrepost lasts until
    its fast-neutron fluence limit, then adjusted for plant availability.
    """

    fn = calculate_cplife_superconducting

    neut_flux_cp = From(fwbs)
    flu_tf_neutron_fast_max = From(constraints)
    life_plant = From(costs)
    f_t_plant_available = From(costs)


class CplifeAvailResistive(CplifeAvail, WrapsFunction):
    """`itart == 1` with `i_tf_sup != SUPERCONDUCTING`: the centrepost lasts until its
    allowable stress fluence is spent, then adjusted for plant availability.
    """

    fn = calculate_cplife_resistive

    cpstflnc = From(costs)
    pflux_fw_neutron_mw = From(physics)
    life_plant = From(costs)
    f_t_plant_available = From(costs)


class CplifeAvailSt(FixedPointFunction):
    """cottax node: `.costs.cplife`'s Shape B self-reference in `AvailSt`
    (`next_steps.md` §5) -- bodiless family base. Four occupants,
    `.tfcoil.i_tf_sup` x `.physics.itart`: which formula computes the fresh centrepost
    lifetime, and whether the `itart == SPHERICAL_TOKAMAK` lifetime-adjustment step is
    then applied.

    Kept a `FixedPointFunction` for every occupant even though none of the four reads a
    previous `.costs.cplife` value at all -- `avail_st()` recomputes `cplife` from
    scratch every call (see `calculate_cplife_avail_st_next`'s docstring, unchanged by
    this split); this was already true of the single pre-split node, whose own docstring
    called it "a degenerate but structurally honest use of the same primitive", and that
    reasoning now applies uniformly across all four occupants rather than to one
    switch-selected body.

    Each occupant reuses an already-existing, already-tested function rather than
    calling `calculate_cplife_avail_st_next` with baked switch values: the
    `itart == SPHERICAL_TOKAMAK` pair is formula-identical to `CplifeAvail`'s own arms
    (`CplifeAvailSuperconducting`/`CplifeAvailResistive`), and the
    `itart != SPHERICAL_TOKAMAK` pair to `CpLifetimeSuperconducting`/
    `CpLifetimeResistive` -- a `TFConductorModel`/`SphericalTokamakModel` case analysis
    of `calculate_cplife_avail_st_next`'s own body confirms this (see each occupant's
    docstring). Reusing them, rather than reintroducing `calculate_cplife_avail_st_next`
    per occupant, also drops the dead `cpstflnc`/`pflux_fw_neutron_mw` (superconducting
    arms) or `neut_flux_cp`/`flu_tf_neutron_fast_max` (resistive arms) reads that keeping
    it would otherwise force.
    """

    cplife = OutputInto(costs)


class CplifeAvailStSuperconductingAdjusted(CplifeAvailSt):
    """`i_tf_sup == SUPERCONDUCTING`, `itart == SPHERICAL_TOKAMAK`: the fresh
    superconducting centrepost lifetime, availability-adjusted -- the same formula as
    `CplifeAvailSuperconducting` (`.costs.cplife`'s `avail()`/`avail_2()` producer),
    same `fn`, different node shape (`FixedPointFunction`, not `ExplicitFunction`).
    """

    def step(
        self,
        neut_flux_cp=From(fwbs),
        flu_tf_neutron_fast_max=From(constraints),
        life_plant=From(costs),
        f_t_plant_available=From(costs),
    ):
        """`calculate_cplife_superconducting`, ports declared above."""
        return calculate_cplife_superconducting(
            neut_flux_cp, flu_tf_neutron_fast_max, life_plant, f_t_plant_available
        )


class CplifeAvailStSuperconductingUnadjusted(CplifeAvailSt):
    """`i_tf_sup == SUPERCONDUCTING`, `itart != SPHERICAL_TOKAMAK`: the fresh
    superconducting centrepost lifetime, unadjusted -- the same formula as
    `CpLifetimeSuperconducting`.
    """

    def step(
        self,
        neut_flux_cp=From(fwbs),
        flu_tf_neutron_fast_max=From(constraints),
        life_plant=From(costs),
    ):
        """`calculate_cp_lifetime_superconducting`, ports declared above."""
        return calculate_cp_lifetime_superconducting(
            neut_flux_cp, flu_tf_neutron_fast_max, life_plant
        )


class CplifeAvailStResistiveAdjusted(CplifeAvailSt):
    """`i_tf_sup != SUPERCONDUCTING`, `itart == SPHERICAL_TOKAMAK`: the fresh resistive
    centrepost lifetime, availability-adjusted -- the same formula as
    `CplifeAvailResistive`.
    """

    def step(
        self,
        cpstflnc=From(costs),
        pflux_fw_neutron_mw=From(physics),
        life_plant=From(costs),
        f_t_plant_available=From(costs),
    ):
        """`calculate_cplife_resistive`, ports declared above."""
        return calculate_cplife_resistive(
            cpstflnc, pflux_fw_neutron_mw, life_plant, f_t_plant_available
        )


class CplifeAvailStResistiveUnadjusted(CplifeAvailSt):
    """`i_tf_sup != SUPERCONDUCTING`, `itart != SPHERICAL_TOKAMAK`: the fresh resistive
    centrepost lifetime, unadjusted -- the same formula as `CpLifetimeResistive`.
    """

    def step(
        self,
        cpstflnc=From(costs),
        pflux_fw_neutron_mw=From(physics),
        life_plant=From(costs),
    ):
        """`calculate_cp_lifetime_resistive`, ports declared above."""
        return calculate_cp_lifetime_resistive(cpstflnc, pflux_fw_neutron_mw, life_plant)


class Avail(ExplicitFunction):
    """The `calculate_avail` family -- `calculate_avail`'s outputs *other* than
    `.costs.cplife`, one occupant per `.costs.ibkt_life` value.
    """

    life_blkt_fpy = OutputInto(fwbs)
    life_div_fpy = OutputInto(costs)
    bktcycles = OutputInto(costs)
    cpfact = OutputInto(costs)
    life_hcd_fpy = OutputInto(costs)


class AvailNeutronFluence(Avail, WrapsFunction):
    """`ibkt_life == NEUTRON_FLUENCE` (0) -- PROCESS's own default
    (`cost_variables.py:416`) and the reference run's.
    """

    fn = calculate_avail_neutron_fluence

    life_fw_fpy = From(fwbs)
    abktflnc = From(costs)
    pflux_fw_neutron_mw = From(physics)
    life_plant = From(costs)
    pflux_div_heat_load_mw = From(divertor)
    adivflnc = From(costs)
    t_plant_pulse_total = From(times)
    t_plant_pulse_burn = From(times)
    f_t_plant_available = From(costs)


class AvailDisplacementsPerAtom(Avail, WrapsFunction):
    """`ibkt_life == FUSION_POWER` (1) -- the blanket lifetime set by displacement
    damage per full-power year.
    """

    fn = calculate_avail_displacements_per_atom

    life_fw_fpy = From(fwbs)
    p_fusion_total_mw = From(physics)
    life_dpa = From(costs)
    life_plant = From(costs)
    pflux_div_heat_load_mw = From(divertor)
    adivflnc = From(costs)
    t_plant_pulse_total = From(times)
    t_plant_pulse_burn = From(times)
    f_t_plant_available = From(costs)


class Avail2(ExplicitFunction):
    """cottax node: `calculate_avail_2`'s outputs *other* than `.costs.cplife`, ports
    declared, `u_planned`/`u_unplanned` dropped (no `VarPath` -- see the module-level
    note above) -- bodiless family base. Four occupants, `.costs.ibkt_life` x
    `.physics.itart`.

    `n_vac_pumps_high`/`redun_vac` stay `eqx.field(static=True)`: genuine IN.DAT input
    values (`.vacuum.n_vac_pumps_high`, a `calculate_redun_vac`-derived
    `.costs.redun_vac`), not a model choice, so they are read via `self` in every
    occupant rather than baked -- unlike `ibkt_life`/`itart`, which each occupant bakes
    to a literal and never holds as a field at all.
    """

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
        return self._compute(
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
            t_plant_pulse_burn,
            t_plant_pulse_total,
            cplife,
        )


class Avail2NeutronFluenceConventional(Avail2):
    """`.costs.ibkt_life == NEUTRON_FLUENCE`, `.physics.itart ==
    CONVENTIONAL_ASPECT_RATIO`.
    """

    def _compute(self, *args):
        return calculate_avail_2_dropping_extras(
            *args,
            self.n_vac_pumps_high,
            self.redun_vac,
            ibkt_life=BlanketLifetimeModel.NEUTRON_FLUENCE,
            itart=SphericalTokamakModel.CONVENTIONAL_ASPECT_RATIO,
        )


class Avail2NeutronFluenceSphericalTokamak(Avail2):
    """`.costs.ibkt_life == NEUTRON_FLUENCE`, `.physics.itart ==
    SPHERICAL_TOKAMAK`.
    """

    def _compute(self, *args):
        return calculate_avail_2_dropping_extras(
            *args,
            self.n_vac_pumps_high,
            self.redun_vac,
            ibkt_life=BlanketLifetimeModel.NEUTRON_FLUENCE,
            itart=SphericalTokamakModel.SPHERICAL_TOKAMAK,
        )


class Avail2FusionPowerConventional(Avail2):
    """`.costs.ibkt_life == FUSION_POWER`, `.physics.itart ==
    CONVENTIONAL_ASPECT_RATIO`.
    """

    def _compute(self, *args):
        return calculate_avail_2_dropping_extras(
            *args,
            self.n_vac_pumps_high,
            self.redun_vac,
            ibkt_life=BlanketLifetimeModel.FUSION_POWER,
            itart=SphericalTokamakModel.CONVENTIONAL_ASPECT_RATIO,
        )


class Avail2FusionPowerSphericalTokamak(Avail2):
    """`.costs.ibkt_life == FUSION_POWER`, `.physics.itart == SPHERICAL_TOKAMAK`."""

    def _compute(self, *args):
        return calculate_avail_2_dropping_extras(
            *args,
            self.n_vac_pumps_high,
            self.redun_vac,
            ibkt_life=BlanketLifetimeModel.FUSION_POWER,
            itart=SphericalTokamakModel.SPHERICAL_TOKAMAK,
        )


class AvailSt(ExplicitFunction):
    """cottax node: `calculate_avail_st`'s outputs *other* than `.costs.cplife`, ports
    declared, `maint_cycle`/`n_cycles_main`/`n_centre_cols`/`u_planned`/`u_unplanned`
    dropped (no `VarPath`) -- bodiless family base. Eight occupants,
    `.costs.ibkt_life` x `.physics.itart` x `.tfcoil.i_tf_sup`: the third axis used to
    be a Python `if` inside this class's own `__call__` (selecting the fresh centrepost
    lifetime formula) -- the only control flow anywhere in this file's node bodies --
    and splitting the switch dissolves it into which occupant is built, exactly like
    `ibkt_life`/`itart` below. `calculate_avail_st_dropping_extras` (in
    `functional_process/models/`) still carries an `if i_tf_sup == 1` internally, per
    the pushdown convention (`ede83e67`): each occupant calls it with `i_tf_sup` already
    a literal, never from `self`, so the branch is inert once an occupant is chosen.

    `n_vac_pumps_high`/`redun_vac` stay static -- see `Avail2`'s docstring, same
    reasoning.
    """

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
        return self._compute(
            abktflnc,
            pflux_fw_neutron_mw,
            life_dpa,
            p_fusion_total_mw,
            adivflnc,
            pflux_div_heat_load_mw,
            life_plant,
            neut_flux_cp,
            flu_tf_neutron_fast_max,
            cpstflnc,
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
            u_unplanned_cp,
            t_plant_pulse_burn,
            t_plant_pulse_total,
        )


class AvailStNeutronFluenceConventionalSuperconducting(AvailSt):
    """`ibkt_life == NEUTRON_FLUENCE`, `itart == CONVENTIONAL_ASPECT_RATIO`,
    `i_tf_sup == SUPERCONDUCTING`.
    """

    def _compute(self, *args):
        return calculate_avail_st_dropping_extras(
            *args,
            self.n_vac_pumps_high,
            self.redun_vac,
            ibkt_life=BlanketLifetimeModel.NEUTRON_FLUENCE,
            itart=SphericalTokamakModel.CONVENTIONAL_ASPECT_RATIO,
            i_tf_sup=TFConductorModel.SUPERCONDUCTING,
        )


class AvailStNeutronFluenceConventionalResistive(AvailSt):
    """`ibkt_life == NEUTRON_FLUENCE`, `itart == CONVENTIONAL_ASPECT_RATIO`,
    `i_tf_sup != SUPERCONDUCTING`.
    """

    def _compute(self, *args):
        return calculate_avail_st_dropping_extras(
            *args,
            self.n_vac_pumps_high,
            self.redun_vac,
            ibkt_life=BlanketLifetimeModel.NEUTRON_FLUENCE,
            itart=SphericalTokamakModel.CONVENTIONAL_ASPECT_RATIO,
            i_tf_sup=TFConductorModel.WATER_COOLED_COPPER,
        )


class AvailStNeutronFluenceSphericalTokamakSuperconducting(AvailSt):
    """`ibkt_life == NEUTRON_FLUENCE`, `itart == SPHERICAL_TOKAMAK`,
    `i_tf_sup == SUPERCONDUCTING`.
    """

    def _compute(self, *args):
        return calculate_avail_st_dropping_extras(
            *args,
            self.n_vac_pumps_high,
            self.redun_vac,
            ibkt_life=BlanketLifetimeModel.NEUTRON_FLUENCE,
            itart=SphericalTokamakModel.SPHERICAL_TOKAMAK,
            i_tf_sup=TFConductorModel.SUPERCONDUCTING,
        )


class AvailStNeutronFluenceSphericalTokamakResistive(AvailSt):
    """`ibkt_life == NEUTRON_FLUENCE`, `itart == SPHERICAL_TOKAMAK`,
    `i_tf_sup != SUPERCONDUCTING`.
    """

    def _compute(self, *args):
        return calculate_avail_st_dropping_extras(
            *args,
            self.n_vac_pumps_high,
            self.redun_vac,
            ibkt_life=BlanketLifetimeModel.NEUTRON_FLUENCE,
            itart=SphericalTokamakModel.SPHERICAL_TOKAMAK,
            i_tf_sup=TFConductorModel.WATER_COOLED_COPPER,
        )


class AvailStFusionPowerConventionalSuperconducting(AvailSt):
    """`ibkt_life == FUSION_POWER`, `itart == CONVENTIONAL_ASPECT_RATIO`,
    `i_tf_sup == SUPERCONDUCTING`.
    """

    def _compute(self, *args):
        return calculate_avail_st_dropping_extras(
            *args,
            self.n_vac_pumps_high,
            self.redun_vac,
            ibkt_life=BlanketLifetimeModel.FUSION_POWER,
            itart=SphericalTokamakModel.CONVENTIONAL_ASPECT_RATIO,
            i_tf_sup=TFConductorModel.SUPERCONDUCTING,
        )


class AvailStFusionPowerConventionalResistive(AvailSt):
    """`ibkt_life == FUSION_POWER`, `itart == CONVENTIONAL_ASPECT_RATIO`,
    `i_tf_sup != SUPERCONDUCTING`.
    """

    def _compute(self, *args):
        return calculate_avail_st_dropping_extras(
            *args,
            self.n_vac_pumps_high,
            self.redun_vac,
            ibkt_life=BlanketLifetimeModel.FUSION_POWER,
            itart=SphericalTokamakModel.CONVENTIONAL_ASPECT_RATIO,
            i_tf_sup=TFConductorModel.WATER_COOLED_COPPER,
        )


class AvailStFusionPowerSphericalTokamakSuperconducting(AvailSt):
    """`ibkt_life == FUSION_POWER`, `itart == SPHERICAL_TOKAMAK`,
    `i_tf_sup == SUPERCONDUCTING`.
    """

    def _compute(self, *args):
        return calculate_avail_st_dropping_extras(
            *args,
            self.n_vac_pumps_high,
            self.redun_vac,
            ibkt_life=BlanketLifetimeModel.FUSION_POWER,
            itart=SphericalTokamakModel.SPHERICAL_TOKAMAK,
            i_tf_sup=TFConductorModel.SUPERCONDUCTING,
        )


class AvailStFusionPowerSphericalTokamakResistive(AvailSt):
    """`ibkt_life == FUSION_POWER`, `itart == SPHERICAL_TOKAMAK`,
    `i_tf_sup != SUPERCONDUCTING`.
    """

    def _compute(self, *args):
        return calculate_avail_st_dropping_extras(
            *args,
            self.n_vac_pumps_high,
            self.redun_vac,
            ibkt_life=BlanketLifetimeModel.FUSION_POWER,
            itart=SphericalTokamakModel.SPHERICAL_TOKAMAK,
            i_tf_sup=TFConductorModel.WATER_COOLED_COPPER,
        )
