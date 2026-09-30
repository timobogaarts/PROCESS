"""PROCESS's constraints and figures of merit, declared.

A `constraint_<id>` or `objective_metric_<id>` in `functional_process.models` is a
ported function like any other; each is declared here once, by name. A function
PROCESS switches is split into one per branch (`constraint_17_tokamak`, ...), and where
a switch only chose which quantity a formula reads, each choice is its own declaration
reading its own place (`Constraint5Greenwald` reads the line-averaged density,
`Constraint5VolumeAverage` the volume-averaged one). No switch is read: the IN.DAT
converter picks the declarations, and a configuration names them.

Every other parameter is read at the place its name lives (`area_of`: PROCESS's
declared inputs, and `NON_INPUT_FIELDS` for what a model computes).

A constraint's body owns its four values, `c<id>_residual`, `c<id>` (the normalised
residual, `held`), `c<id>_value` and `c<id>_bound`; whether `c<id>` is held `= 0` or
`<= 0` is the problem's. A figure of merit owns `MERIT`; its direction is the problem's.
Constraint 76 is not declared: it reads `nd_plasma_electron_max_array_7`, which no
model computes and no input states.
"""

import inspect

from cottax.interfaces import Scope, V
from jax.tree_util import GetAttrKey

from functional_process.models import constraints as C
from functional_process.models import objectives as O
from functional_process.vocabulary.input_variables import INPUT_VARIABLES

CONSTRAINTS = Scope(V.constraints, GetAttrKey)
"""Where a constraint's values are computed: `.constraints.c<id>...`, beside the
limits PROCESS already keeps in that area."""

NUMERICS = Scope(V.numerics, GetAttrKey)

MERIT = NUMERICS.place("merit")
"""Where the figure of merit is computed."""


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


def _reads(fn, chosen) -> dict:
    """Each parameter of `fn` at the place its name lives, or at the place `chosen`
    names for it."""
    return {
        p: area_of(chosen[p]).place(chosen[p]) if p in chosen else area_of(p).place(p)
        for p in inspect.signature(fn).parameters
    }


def constraint_places(cid: int) -> tuple:
    """Constraint `cid`'s four values, in the ported function's order: the residual,
    the normalised residual (the one held), the value and the bound."""
    return tuple(
        CONSTRAINTS.place(f"c{cid}{suffix}")
        for suffix in ("_residual", "", "_value", "_bound")
    )


def held(constraint):
    """The value a constraint declaration's requirement holds: its normalised
    residual."""
    return constraint.function_outputs[1]


def _constraint(cid: int, fn, **chosen):
    return CONSTRAINTS.function(fn, *constraint_places(cid), reads=_reads(fn, chosen))


def _merit(fn):
    return NUMERICS.function(fn, MERIT, reads=_reads(fn, {}))


# ---------------------------------------------------------------- the constraints

Constraint1 = _constraint(1, C.constraint_1)
Constraint6 = _constraint(6, C.constraint_6)
Constraint8 = _constraint(8, C.constraint_8)
Constraint9 = _constraint(9, C.constraint_9)
Constraint11 = _constraint(11, C.constraint_11)
Constraint12 = _constraint(12, C.constraint_12)
Constraint13 = _constraint(13, C.constraint_13)
Constraint14 = _constraint(14, C.constraint_14)
Constraint15 = _constraint(15, C.constraint_15)
Constraint16 = _constraint(16, C.constraint_16)
Constraint18 = _constraint(18, C.constraint_18)
Constraint19 = _constraint(19, C.constraint_19)
Constraint20 = _constraint(20, C.constraint_20)
Constraint21 = _constraint(21, C.constraint_21)
Constraint22 = _constraint(22, C.constraint_22)
Constraint23 = _constraint(23, C.constraint_23)
Constraint25 = _constraint(25, C.constraint_25)
Constraint26 = _constraint(26, C.constraint_26)
Constraint27 = _constraint(27, C.constraint_27)
Constraint29 = _constraint(29, C.constraint_29)
Constraint30 = _constraint(30, C.constraint_30)
Constraint31 = _constraint(31, C.constraint_31)
Constraint32 = _constraint(32, C.constraint_32)
Constraint33 = _constraint(33, C.constraint_33)
Constraint34 = _constraint(34, C.constraint_34)
Constraint35 = _constraint(35, C.constraint_35)
Constraint36 = _constraint(36, C.constraint_36)
Constraint37 = _constraint(37, C.constraint_37)
Constraint39 = _constraint(39, C.constraint_39)
Constraint40 = _constraint(40, C.constraint_40)
Constraint41 = _constraint(41, C.constraint_41)
Constraint42 = _constraint(42, C.constraint_42)
Constraint48 = _constraint(48, C.constraint_48)
Constraint51 = _constraint(51, C.constraint_51)
Constraint53 = _constraint(53, C.constraint_53)
Constraint54 = _constraint(54, C.constraint_54)
Constraint56 = _constraint(56, C.constraint_56)
Constraint59 = _constraint(59, C.constraint_59)
Constraint60 = _constraint(60, C.constraint_60)
Constraint61 = _constraint(61, C.constraint_61)
Constraint62 = _constraint(62, C.constraint_62)
Constraint63 = _constraint(63, C.constraint_63)
Constraint64 = _constraint(64, C.constraint_64)
Constraint65 = _constraint(65, C.constraint_65)
Constraint66 = _constraint(66, C.constraint_66)
Constraint67 = _constraint(67, C.constraint_67)
Constraint73 = _constraint(73, C.constraint_73)
Constraint74 = _constraint(74, C.constraint_74)
Constraint75 = _constraint(75, C.constraint_75)
Constraint77 = _constraint(77, C.constraint_77)
Constraint78 = _constraint(78, C.constraint_78)
Constraint79 = _constraint(79, C.constraint_79)
Constraint80 = _constraint(80, C.constraint_80)
Constraint81 = _constraint(81, C.constraint_81)
Constraint82 = _constraint(82, C.constraint_82)
Constraint83 = _constraint(83, C.constraint_83)
Constraint84 = _constraint(84, C.constraint_84)
Constraint86 = _constraint(86, C.constraint_86)
Constraint87 = _constraint(87, C.constraint_87)
Constraint88 = _constraint(88, C.constraint_88)
Constraint89 = _constraint(89, C.constraint_89)
Constraint92 = _constraint(92, C.constraint_92)

Constraint2HeatedTotalRadiation = _constraint(
    2, C.constraint_2_non_ignited, pden_plasma_radiated_mw="pden_plasma_rad_mw"
)
Constraint2HeatedCoreRadiation = _constraint(
    2, C.constraint_2_non_ignited, pden_plasma_radiated_mw="pden_plasma_core_rad_mw"
)
Constraint2IgnitedTotalRadiation = _constraint(
    2, C.constraint_2_ignited, pden_plasma_radiated_mw="pden_plasma_rad_mw"
)
Constraint2IgnitedCoreRadiation = _constraint(
    2, C.constraint_2_ignited, pden_plasma_radiated_mw="pden_plasma_core_rad_mw"
)
Constraint3Heated = _constraint(3, C.constraint_3_non_ignited)
Constraint3Ignited = _constraint(3, C.constraint_3_ignited)
Constraint4HeatedTotalRadiation = _constraint(
    4, C.constraint_4_non_ignited, pden_plasma_radiated_mw="pden_plasma_rad_mw"
)
Constraint4HeatedCoreRadiation = _constraint(
    4, C.constraint_4_non_ignited, pden_plasma_radiated_mw="pden_plasma_core_rad_mw"
)
Constraint4IgnitedTotalRadiation = _constraint(
    4, C.constraint_4_ignited, pden_plasma_radiated_mw="pden_plasma_rad_mw"
)
Constraint4IgnitedCoreRadiation = _constraint(
    4, C.constraint_4_ignited, pden_plasma_radiated_mw="pden_plasma_core_rad_mw"
)
Constraint5Greenwald = _constraint(
    5, C.constraint_5_density_limit, nd_plasma_electron_limited="nd_plasma_electron_line"
)
Constraint5VolumeAverage = _constraint(
    5,
    C.constraint_5_density_limit,
    nd_plasma_electron_limited="nd_plasma_electrons_vol_avg",
)
Constraint7 = _constraint(7, C.constraint_7_non_ignited)
Constraint17Tokamak = _constraint(17, C.constraint_17_tokamak)
Constraint17Stellarator = _constraint(17, C.constraint_17_stellarator)
Constraint24TotalBeta = _constraint(
    24, C.constraint_24_beta_limit, beta_limited="beta_total_vol_avg"
)
Constraint24ThermalBeta = _constraint(
    24, C.constraint_24_beta_limit, beta_limited="beta_thermal_vol_avg"
)
Constraint24ToroidalBeta = _constraint(
    24, C.constraint_24_beta_limit, beta_limited="beta_toroidal_vol_avg"
)
Constraint24ThermalAndBeamBeta = _constraint(24, C.constraint_24_thermal_and_beam)
Constraint28 = _constraint(28, C.constraint_28_non_ignited)
Constraint43Copper = _constraint(43, C.constraint_43_copper)
Constraint43Other = _constraint(43, C.constraint_43_other)
Constraint44Copper = _constraint(44, C.constraint_44_copper)
Constraint44Other = _constraint(44, C.constraint_44_other)
Constraint45 = _constraint(45, C.constraint_45_tart)
Constraint46 = _constraint(46, C.constraint_46_tart)
Constraint68FixedQ95 = _constraint(68, C.constraint_68_fixed_q95)
Constraint68ComputedQ95 = _constraint(68, C.constraint_68_computed_q95)
Constraint72Bucked = _constraint(72, C.constraint_72_bucked)
Constraint72Free = _constraint(72, C.constraint_72_free)
Constraint85InputLifetime = _constraint(
    85, C.constraint_85_lifetime, cplife_bound="cplife_input"
)
Constraint85DivertorLifetime = _constraint(
    85, C.constraint_85_lifetime, cplife_bound="life_div_fpy"
)
Constraint85BlanketLifetime = _constraint(
    85, C.constraint_85_lifetime, cplife_bound="life_blkt_fpy"
)
Constraint85PlantLifetime = _constraint(
    85, C.constraint_85_lifetime, cplife_bound="life_plant"
)
Constraint90BlanketCycles = _constraint(
    90, C.constraint_90_cycles, n_cycle_bound="bktcycles"
)
Constraint90MinimumCycles = _constraint(
    90, C.constraint_90_cycles, n_cycle_bound="n_cycle_min"
)
Constraint91Heated = _constraint(91, C.constraint_91_non_ignited)
Constraint91Ignited = _constraint(91, C.constraint_91_ignited)


# ---------------------------------------------------------------- the figures of merit

MajorRadius = _merit(O.objective_metric_1)
NeutronWallLoad = _merit(O.objective_metric_3)
PTfPlusPPf = _merit(O.objective_metric_4)
FusionGainQ = _merit(O.objective_metric_5)
CostOfElectricity = _merit(O.objective_metric_6)
CapitalCostDirect = _merit(O.objective_metric_7_direct)
CapitalCostReactor = _merit(O.objective_metric_7_reactor)
AspectRatio = _merit(O.objective_metric_8)
DivertorHeatLoad = _merit(O.objective_metric_9)
ToroidalField = _merit(O.objective_metric_10)
TotalInjectedPower = _merit(O.objective_metric_11)
PulseLength = _merit(O.objective_metric_14)
PlantAvailabilityFactor = _merit(O.objective_metric_15_modelled)
MinR0MaxTauBurn = _merit(O.objective_metric_16)
NetElectricalOutput = _merit(O.objective_metric_17)
NullFigureOfMerit = _merit(O.objective_metric_18)
MaxQMaxTPlantPulseBurn = _merit(O.objective_metric_19)
