"""PROCESS's constraints and figures of merit as model bodies.

A `constraint_<id>` or `objective_metric_<id>` in `functional_process.models` is a
ported function like any other; this is the thin cottax layer over it. A **switch** is
never read: it picks which function is the model (`VARIANTS`, the switched functions
split into one per branch), and where a switch only chose which quantity a formula
reads, it picks the place instead. Every other parameter is read at the place its name
lives (`area_of`: PROCESS's declared inputs, and `NON_INPUT_FIELDS` for what a model
computes).

A constraint's body owns its four values, `c<id>_residual`, `c<id>` (the normalised
residual, the one held), `c<id>_value` and `c<id>_bound`. Whether `c<id>` is held `= 0`
or `<= 0` is not the constraint's: PROCESS holds the first `neqns` of `icc` as
equalities, whatever the function's own sense, so `problem.py` decides.
"""

import inspect

from cottax.interfaces import Scope, V
from jax.tree_util import GetAttrKey

from functional_process.models import constraints as ported_constraints
from functional_process.models import objectives as ported_objectives
from functional_process.vocabulary.enums import FiguresOfMerit
from functional_process.vocabulary import (
    AvailabilityModel,
    BetaComponentLimits,
    DensityLimitModel,
    TFConductorModel,
    TFCSRadialConfiguration,
)
from functional_process.vocabulary.input_variables import INPUT_VARIABLES

CONSTRAINTS = Scope(V.constraints, GetAttrKey)
"""Where a constraint's value is computed: `.constraints.c<id>`, beside the limits
PROCESS already keeps in that area."""

NUMERICS = Scope(V.numerics, GetAttrKey)
"""Where the figure of merit is computed: `.numerics.objf`."""


SWITCH_PARAMETER_NAMES = (
    "bkt_life_csf",
    "i_beta_component",
    "i_cp_lifetime",
    "i_density_limit",
    "i_plant_availability",
    "i_plasma_ignited",
    "i_q95_fixed",
    "i_rad_loss",
    "i_tf_bucking",
    "i_tf_inside_cs",
    "i_tf_sup",
    "ibkt_life",
    "ireactor",
    "istell",
    "itart",
)
"""Every parameter name that is a **switch** anywhere in the ported constraint/objective
surface -- the union of the `static_argnames` the `Tier1Contract`s in
`functional_process/tests/cottax/core/solver/test_constraints.py` and
`test_objectives.py` declare.
"""


NON_INPUT_FIELDS = {
    "available_radial_space": "build",
    "b_cs_peak_flat_top_end": "pf_coil",
    "b_cs_peak_pulse_start": "pf_coil",
    "b_plasma_total": "physics",
    "b_tf_inboard_peak_with_ripple": "tfcoil",
    "beta_beam": "physics",
    "beta_fast_alpha": "physics",
    "beta_poloidal_eps": "physics",
    "beta_poloidal_vol_avg": "physics",
    "beta_thermal_vol_avg": "physics",
    "beta_toroidal_vol_avg": "physics",
    "big_q_plasma": "current_drive",
    "bktcycles": "costs",
    "c_tf_total": "tfcoil",
    "cdirt": "costs",
    "coe": "costs",
    "concost": "costs",
    "coppera_m2": "rebco",
    "cplife": "costs",
    "dr_fw_outboard": "build",
    "dx_tf_inboard_out_toroidal": "tfcoil",
    "eps": "physics",
    "eta_cd_norm_hcd_primary": "current_drive",
    "f_p_beam_shine_through": "current_drive",
    "f_p_plasma_separatrix_rad": "physics",
    "f_pden_alpha_electron_mw": "physics",
    "f_pden_alpha_ions_mw": "physics",
    "f_t_alpha_energy_confinement": "physics",
    "flu_tf_neutron_fast_peak": "fwbs",
    "fzmin": "reinke",
    "j_cs_critical_flat_top_end": "pf_coil",
    "j_cs_critical_pulse_start": "pf_coil",
    "j_cs_pulse_start": "pf_coil",
    "j_tf_wp": "tfcoil",
    "j_tf_wp_critical": "tfcoil",
    "j_tf_wp_quench_heat_max": "tfcoil",
    "life_blkt_fpy": "fwbs",
    "life_div_fpy": "costs",
    "n_beam_decay_lengths_core": "current_drive",
    "n_charge_plasma_effective_vol_avg": "physics",
    "n_cycle": "cs_fatigue",
    "n_iter_vacuum_pumps": "vacuum",
    "nd_beam_ions": "physics",
    "nd_beam_ions_out": "physics",
    "nd_plasma_electron_line": "physics",
    "nd_plasma_electron_on_axis": "physics",
    "nd_plasma_electrons_max": "physics",
    "nd_plasma_ions_total_vol_avg": "physics",
    "p_cp_resistive_mw": "tfcoil",
    "p_cryo_plant_electric_mw": "heat_transport",
    "p_div_bt_q_aspect_rmajor_mw": "physics",
    "p_fusion_total_mw": "physics",
    "p_hcd_injected_electrons_mw": "current_drive",
    "p_hcd_injected_ions_mw": "current_drive",
    "p_hcd_injected_total_mw": "current_drive",
    "p_l_h_threshold_mw": "physics",
    "p_plant_electric_net_mw": "heat_transport",
    "p_plasma_heating_total_mw": "physics",
    "p_plasma_separatrix_mw": "physics",
    "p_plasma_separatrix_rmajor_mw": "physics",
    "p_tf_leg_resistive_mw": "tfcoil",
    "pden_alpha_total_mw": "physics",
    "pden_electron_transport_loss_mw": "physics",
    "pden_ion_electron_equilibration_mw": "physics",
    "pden_ion_transport_loss_mw": "physics",
    "pden_non_alpha_charged_mw": "physics",
    "pden_plasma_core_rad_mw": "physics",
    "pden_plasma_ohmic_mw": "physics",
    "pden_plasma_rad_mw": "physics",
    "peakpoloidalpower": "pf_power",
    "pflux_fw_neutron_mw": "physics",
    "pflux_fw_rad_max_mw": "constraints",
    "plasma_current": "physics",
    "powerht_constraint": "stellarator",
    "powerscaling_constraint": "stellarator",
    "psolradmw": "physics",
    "ptfnucpm3": "fwbs",
    "q95_min": "physics",
    "radius_beam_tangency": "current_drive",
    "radius_beam_tangency_max": "current_drive",
    "rbld": "build",
    "required_radial_space": "build",
    "rminor": "physics",
    "sig_tf_case": "tfcoil",
    "sig_tf_cs_bucked": "tfcoil",
    "sig_tf_wp": "tfcoil",
    "srcktpm": "pf_power",
    "str_wp": "tfcoil",
    "stress_shear_cs_peak": "pf_coil",
    "t_current_ramp_up_min": "constraints",
    "t_plant_pulse_total": "times",
    "tcpav2": "tfcoil",
    "temp_cp_peak": "tfcoil",
    "temp_croco_quench": "tfcoil",
    "temp_cs_superconductor_margin": "pf_coil",
    "temp_fw_peak": "fwbs",
    "temp_plasma_electron_density_weighted_kev": "physics",
    "temp_plasma_ion_density_weighted_kev": "physics",
    "temp_tf_superconductor_margin": "tfcoil",
    "tfcmw": "tfcoil",
    "toroidalgap": "tfcoil",
    "v_tf_coil_dump_quench_kv": "tfcoil",
    "vol_plasma": "physics",
    "vs_cs_pf_total_pulse": "pf_coil",
    "vs_cs_pf_total_ramp": "pf_coil",
    "vs_plasma_ramp_required": "physics",
    "vs_plasma_total_required": "physics",
    "vv_stress_quench": "superconducting_tfcoil",
}
"""`field name -> DataStructure area` for every name the ported constraint/objective
layer can ask for that is **not** a declared PROCESS input.
"""


def area_of(name: str) -> Scope:
    """The area parameter `name` is read in.

    Raises
    ------
    ValueError
        If `name` is neither a declared PROCESS input nor in `NON_INPUT_FIELDS`.
    """
    area = NON_INPUT_FIELDS.get(name)
    if area is None:
        # `ixc`/`icc` carry `module=None`: the problem statement, not a field.
        area = getattr(INPUT_VARIABLES.get(name), "module", None)
    if area is None:
        raise ValueError(
            f"{name!r} is neither a declared PROCESS input nor in `NON_INPUT_FIELDS`: "
            f"a typo, or a quantity an unported model computes, which belongs in "
            f"`NON_INPUT_FIELDS` with its area"
        )
    return Scope(getattr(V, area), GetAttrKey)


class NoVariant(ValueError):
    """A switch value no ported branch answers."""


def _at(name: str):
    """Where `name` lives: the place a variant reads in its parameter's stead."""
    return area_of(name).place(name)


def _ignited(switches) -> bool:
    return switches["i_plasma_ignited"] != 0


def _radiated(switches):
    """The radiated power `i_rad_loss` counts: a place, as a read of the variant."""
    i_rad_loss = switches["i_rad_loss"]
    if i_rad_loss == 0:
        return {"pden_plasma_radiated_mw": _at("pden_plasma_rad_mw")}
    if i_rad_loss == 1:
        return {"pden_plasma_radiated_mw": _at("pden_plasma_core_rad_mw")}
    raise NoVariant(f"i_rad_loss={i_rad_loss} (no radiated loss) has no ported variant")


def _only(fn, condition: bool, why: str, reads=None):
    if not condition:
        raise NoVariant(why)
    return fn, reads or {}


C, O = ported_constraints, ported_objectives

VARIANTS = {
    "constraint_2": lambda s: (
        C.constraint_2_ignited if _ignited(s) else C.constraint_2_non_ignited,
        _radiated(s),
    ),
    "constraint_3": lambda s: (
        C.constraint_3_ignited if _ignited(s) else C.constraint_3_non_ignited,
        {},
    ),
    "constraint_4": lambda s: (
        C.constraint_4_ignited if _ignited(s) else C.constraint_4_non_ignited,
        _radiated(s),
    ),
    "constraint_5": lambda s: (
        C.constraint_5_density_limit,
        {
            "nd_plasma_electron_limited": _at(
                "nd_plasma_electron_line"
                if s["i_density_limit"] == DensityLimitModel.GREENWALD
                else "nd_plasma_electrons_vol_avg",
            )
        },
    ),
    "constraint_7": lambda s: _only(
        C.constraint_7_non_ignited, not _ignited(s), "constraint 7: ignited"
    ),
    "constraint_17": lambda s: (
        C.constraint_17_stellarator if s["istell"] != 0 else C.constraint_17_tokamak,
        {},
    ),
    "constraint_24": lambda s: _beta_limit(s),
    "constraint_28": lambda s: _only(
        C.constraint_28_non_ignited, not _ignited(s), "constraint 28: ignited"
    ),
    "constraint_43": lambda s: (
        C.constraint_43_copper
        if s["i_tf_sup"] == TFConductorModel.WATER_COOLED_COPPER
        else C.constraint_43_other,
        {},
    ),
    "constraint_44": lambda s: (
        C.constraint_44_copper
        if s["i_tf_sup"] == TFConductorModel.WATER_COOLED_COPPER
        else C.constraint_44_other,
        {},
    ),
    "constraint_45": lambda s: _only(
        C.constraint_45_tart, s["itart"] != 0, "constraint 45: not a TART"
    ),
    "constraint_46": lambda s: _only(
        C.constraint_46_tart, s["itart"] != 0, "constraint 46: not a TART"
    ),
    "constraint_68": lambda s: (
        C.constraint_68_fixed_q95
        if s["i_q95_fixed"] == 1
        else C.constraint_68_computed_q95,
        {},
    ),
    "constraint_72": lambda s: (
        C.constraint_72_bucked
        if s["i_tf_bucking"] >= 2
        and s["i_tf_inside_cs"] == TFCSRadialConfiguration.TF_OUTSIDE_CS
        else C.constraint_72_free,
        {},
    ),
    "constraint_85": lambda s: _cp_lifetime(s),
    "constraint_90": lambda s: (
        C.constraint_90_cycles,
        {
            "n_cycle_bound": _at("bktcycles")
            if s["ibkt_life"] == 1 and s["bkt_life_csf"] == 1
            else _at("n_cycle_min")
        },
    ),
    "constraint_91": lambda s: (
        C.constraint_91_ignited if _ignited(s) else C.constraint_91_non_ignited,
        {},
    ),
    "objective_metric_7": lambda s: (
        O.objective_metric_7_direct
        if s["ireactor"] == 0
        else O.objective_metric_7_reactor,
        {},
    ),
    "objective_metric_15": lambda s: _only(
        O.objective_metric_15_modelled,
        s["i_plant_availability"] != AvailabilityModel.USER_INPUT,
        "objective 15: the availability is an input",
    ),
}
"""A switched function's name -> `switches -> (the variant, {parameter: place})`:
which function is the model, and the places it reads that are not read by name. These
run once, where the problem is stated; the graph holds only the variant."""


def _beta_limit(s):
    if (
        s["i_beta_component"] == BetaComponentLimits.THERMAL_AND_BEAM
        and s["istell"] == 0
    ):
        return C.constraint_24_thermal_and_beam, {}
    limited = {
        BetaComponentLimits.TOTAL: "beta_total_vol_avg",
        BetaComponentLimits.THERMAL: "beta_thermal_vol_avg",
        BetaComponentLimits.TOROIDAL: "beta_toroidal_vol_avg",
    }
    component = (
        BetaComponentLimits.TOTAL
        if s["istell"] != 0
        else BetaComponentLimits(s["i_beta_component"])
    )
    return C.constraint_24_beta_limit, {"beta_limited": _at(limited[component])}


def _cp_lifetime(s):
    bound = {0: "cplife_input", 1: "life_div_fpy", 2: "life_blkt_fpy", 3: "life_plant"}
    if s["i_cp_lifetime"] not in bound:
        raise NoVariant(f"i_cp_lifetime={s['i_cp_lifetime']} is not in {{0, 1, 2, 3}}")
    return C.constraint_85_lifetime, {"cplife_bound": _at(bound[s["i_cp_lifetime"]])}


def variant(fn, switches):
    """`(function, reads)` for ported `fn` under `switches`: `fn` itself where it takes
    no switch."""
    choose = VARIANTS.get(fn.__name__)
    if choose is None:
        return fn, {}
    return choose(switches)


def _reads(fn, chosen) -> dict:
    return {
        p: chosen[p] if p in chosen else area_of(p)
        for p in inspect.signature(fn).parameters
    }


def constraint_places(cid: int) -> tuple:
    """Constraint `cid`'s four values, in the ported function's order: the residual,
    the normalised residual (the one held), the value and the bound."""
    return tuple(
        CONSTRAINTS.place(f"c{cid}{suffix}")
        for suffix in ("_residual", "", "_value", "_bound")
    )


def constraint_place(cid: int):
    """Where constraint `cid`'s normalised residual is computed: `.constraints.c<id>`."""
    return constraint_places(cid)[1]


def constraint(cid: int, switches):
    """The body of PROCESS constraint `cid`, the variant `switches` pick, owning its
    four values (`constraint_places`).

    Raises
    ------
    ValueError
        If no `constraint_<id>` is ported, no variant answers `switches`, or a
        parameter has no place.
    """
    fn = getattr(ported_constraints, f"constraint_{cid}", None)
    if fn is None:
        raise ValueError(f"constraint {cid}: `models/constraints.py` ports no such one")
    fn, chosen = variant(fn, switches)
    return CONSTRAINTS.function(fn, *constraint_places(cid), reads=_reads(fn, chosen))


MERIT = NUMERICS.place("merit")
"""Where the figure of merit is computed, before its direction is applied."""


def merit(i_figure_merit: int, switches):
    """The body of PROCESS's figure of merit `abs(i_figure_merit)`, the variant
    `switches` pick, owning `MERIT`."""
    fn = ported_objectives.OBJECTIVE_METRICS[FiguresOfMerit(abs(i_figure_merit))]
    fn, chosen = variant(fn, switches)
    return NUMERICS.function(fn, MERIT, reads=_reads(fn, chosen))
