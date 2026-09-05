"""Pure-functional port of `process/models/availability.py` (registry unit #17).

Audit record: `functional_process/_audit/units/models/availability.md`.
`Availability.run()` dispatches on `.costs.i_plant_availability` (`AvailabilityModel`)
to one of three whole-branch alternatives -- `avail()` (USER_INPUT/WARD_TAYLOR, 0/1),
`avail_2()` (MORRIS, 2), `avail_st()` (ST, 3). All three are self-contained (no calls
into other, unported `Model`s) and are ported here as tier-1 pure functions, composed
from a shared set of leaf helpers used by two or three of the branches at once
(`calculate_divertor_lifetime`, `calculate_u_unplanned_*`, the two
`calculate_cp_lifetime_*` alternatives).

Two switches are split into **separate node alternatives** rather than kept as a static
branch inside one function, matching `i_tf_sup`'s precedent in
`tf_nuclear_heating.py`:

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
`geometry.md`'s `.physics.aspect` finding) -- ported by threading a
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
from functional_process.paths import (
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

    Mutually exclusive alternative to `CpLifetimeResistive` -- `.tfcoil.i_tf_sup` selects
    at most one at graph-assembly time (same shape as `i_tf_sup` in
    `tf_nuclear_heating.py`).
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
    """cottax node: `calculate_cp_lifetime_resistive`, unchanged, ports declared.

    Mutually exclusive alternative to `CpLifetimeSuperconducting`.
    """

    cplife = OutputInto(costs)

    def __call__(
        self,
        cpstflnc=From(costs),
        pflux_fw_neutron_mw=From(physics),
        life_plant=From(costs),
    ):
        return calculate_cp_lifetime_resistive(cpstflnc, pflux_fw_neutron_mw, life_plant)


class WardTaylorAvailability(ExplicitFunction):
    """cottax node: `calculate_ward_taylor_availability`, unchanged, ports declared.

    Exists **only** when `.costs.i_plant_availability == 1` -- for USER_INPUT (0),
    `.costs.f_t_plant_available` has no producer at all (an ordinary unowned boundary
    input); see module docstring.
    """

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

    **This was a `FixedPointFunction`, and splitting the switches deleted the fixed
    point.** `calculate_cplife_next` opens `if itart != 1: return cplife` -- so on a
    conventional machine the step is the *identity map*, six of its seven declared reads
    are dead, and the `FixedPoint` problem that owned `^cond.costs.cplife` determined
    nothing (`_audit/switch_kwarg_survey.md` §4.7). On a spherical machine neither
    remaining arm reads `.costs.cplife` at all: the centrepost lifetime is computed
    fresh and then availability-adjusted. So the self-reference existed **only** at the
    value where the body is `return cplife`, which is not a fixed point but an input.

    That makes this the second instance of `inuclear`'s shape (`_audit/next_steps.md`
    §14.4): the conventional arm is an **empty slot** (`CplifeAvail | None`), and the
    two spherical arms are ordinary `ExplicitFunction`s. What `sand.
    degenerate_fixed_points` used to recover at runtime by differentiating a residual,
    the tree now states.

    Shared by `Avail` and `Avail2`: both branches' `itart == 1` cplife-adjustment
    formula is identical once `cplife`/`life_plant`/`f_t_plant_available` are given
    (confirmed by direct comparison of `calculate_avail`'s and `calculate_avail_2`'s
    `itart == 1` blocks -- see the audit record).

    **`total_process.py`'s recorded reason for not registering
    `CpLifetime{Superconducting,Resistive}` here has expired, and a second reason
    stands.** The expired one is ownership: occupants of one slot never coexist, so two
    candidate owners of `.costs.cplife` in the same slot are not a conflict. The
    standing one is that those two nodes return the *fresh* lifetime, where these arms
    return the availability-adjusted one -- a different quantity, so they cannot simply
    be dropped in. The two-line `calculate_cp_lifetime_*` dispatch is *consumed* by the
    occupants below rather than duplicated, which is the half of the old note that could
    be fixed.
    """

    cplife = OutputInto(costs)


class CplifeAvailSuperconducting(CplifeAvail):
    """`itart == 1` with `i_tf_sup == SUPERCONDUCTING` (1): the centrepost lasts until
    its fast-neutron fluence limit, then adjusted for plant availability.

    Reads `.fwbs.neut_flux_cp` and `.constraints.flu_tf_neutron_fast_max`, and neither
    `.costs.cpstflnc` nor `.physics.pflux_fw_neutron_mw` -- **and not `.costs.cplife`**,
    which is what stops this being a fixed point.
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

    Reads `.costs.cpstflnc` and `.physics.pflux_fw_neutron_mw`, and neither
    `.fwbs.neut_flux_cp` nor `.constraints.flu_tf_neutron_fast_max`.
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

    `.costs.cplife` itself is `CplifeAvail`'s (see that class and the module docstring's
    "cottax nodes" section for why this needed splitting at all -- Shape B,
    `next_steps.md` §5). Mutually exclusive alternative to `Avail2`/`AvailSt`:
    `.costs.i_plant_availability` selects at most one of the three branch nodes at
    graph-assembly time.

    **`ibkt_life` was an `eqx.field(static=True)` here and is a slot now; `itart` was
    one and is simply gone** (`_audit/next_steps.md` §14.2). The two are different
    cases, and the difference is worth stating:

    * `ibkt_life` is a real family. Its arms read disjoint fields -- `.costs.abktflnc`
      + `.physics.pflux_fw_neutron_mw` against `.costs.life_dpa` +
      `.physics.p_fusion_total_mw` -- so the one node declared two edges no run makes.
      `switch_kwarg_survey.md` §3 measured only one of the two (`live (1)`), because
      `p_fusion_total_mw` reaches `calculate_dpa_per_fpy` unconditionally and its jaxpr
      method counts a computed-then-discarded value as live. Splitting drops both.
    * `itart` decided **nothing this node computes**. `calculate_avail`'s only
      `itart`-gated output is `cplife_mod` (`availability.py:654-661`), which this node
      discards, so both arms have identical ports *and identical behaviour*. A switch
      that selects nothing is not a family, and the honest conversion is deletion --
      of the field **and** of the `.costs.cplife` read it existed to gate.

    That second read is the one that mattered. The previous docstring said it outright
    -- *"its value is provably inert for every output this node declares ... kept as a
    real `FromExactly` anyway ... even though any value would do here"* -- and keeping
    it made `Avail` a consumer of `CplifeAvail`'s `FixedPoint`, which is the identity
    map on this machine. It is not a consumer, and now does not say it is.
    """

    life_blkt_fpy = OutputInto(fwbs)
    life_div_fpy = OutputInto(costs)
    bktcycles = OutputInto(costs)
    cpfact = OutputInto(costs)
    life_hcd_fpy = OutputInto(costs)


class AvailNeutronFluence(Avail):
    """`ibkt_life == NEUTRON_FLUENCE` (0) -- PROCESS's own default
    (`cost_variables.py:416`) and the reference run's.

    **Two reads leave with this occupant**: `.costs.life_dpa` and
    `.physics.p_fusion_total_mw`.
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

    Reads `.costs.life_dpa` and `.physics.p_fusion_total_mw`, and neither
    `.costs.abktflnc` nor `.physics.pflux_fw_neutron_mw`.
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
