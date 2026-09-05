"""Pure-functional port of `process/core/solver/objective_function`
(`process/core/solver/objectives.py`).

Audit record: `functional_process/_audit/units/core/solver/objectives.md`.

**No stellarator/`istell` special-casing anywhere in the source** (verified:
`grep -n "istell\\|stellarator" process/core/solver/objectives.py` returns nothing) --
every `i_figure_merit` branch is a plain, device-agnostic read/scaling of one or two
already-computed `data` fields. This is the concrete case
`functional_process/_audit/next_steps.md` §6 already argued in the abstract: *"The
objective ... is not a node at all -- a per-run selection of which existing output is
'wanted' ... Selecting it is a `Graph.prune`-style query run when assembling an
`Optimise` problem, not something `total_process.py`/`configuration.py` needs to
represent at assembly time the way topology switches do."* Consistent with that, this
file has **no single `objective_function` dispatcher** -- PROCESS's own `if`/`elif`
chain on `figure_of_merit` selects *once per run*, from `numerics.i_figure_merit`, which
formula applies; reproducing that as one traced function with sixteen branches would
just be the same kind of switch-inside-one-function this codebase splits by default
everywhere else (`_audit/naming_convention.md`; e.g. `coils.py`'s
`jcrit_from_material_*`, one function per `i_tf_sc_mat` value). Instead, every
`FiguresOfMerit` value gets its own `objective_metric_<id>` function, named after
PROCESS's own external ID the same way `constraints.py` names `constraint_<id>` after
PROCESS's own constraint ID -- a reader who already knows `i_figure_merit=6` means "cost
of electricity" finds `objective_metric_6` without translation.

**Sign is not folded in.** PROCESS applies `objective_sign = np.sign(i_figure_merit)`
*outside* the branch (`objectives.py:54,105`) -- negative selects maximise, positive
minimise, applied to whichever `objective_metric` the branch picked. Every function here
returns the raw, unsigned metric (matching the source's own `objective_metric` local);
multiplying by `sign(i_figure_merit)` is the caller's job at `Optimise`-assembly time,
same as `constraints.py` leaves `leq`/`geq`'s bound comparisons ungated by
`numerics.icc`'s active-constraint selection -- selection and sign are both
assembly-time decisions, not physics this port needs to encode.

**Two real PROCESS docstring inaccuracies found while auditing, neither a code bug**:
`objective_function`'s own docstring lists both id 16 and id 19 as *"Major radius/burn
time"* -- copy-pasted, and wrong for 19. The `FiguresOfMerit` enum itself
(`process/data_structure/numerics.py:105-113`) has the real, distinct descriptions: 16
is *"Linear combination of major radius (minimised) and pulse length (maximised)"*
(`MIN_R0_MAX_TAU_BURN`, reads `rmajor`), 19 is *"Linear combination of big Q and pulse
length (maximised)"* (`MAX_Q_MAX_T_PLANT_PULSE_BURN`, reads `big_q_plasma` -- a
different field entirely). Ported as the two genuinely different formulas the code
actually computes, not as the stale docstring's implied duplicate.

**`objective_metric_15` (plant availability factor) keeps a real `raise`, not a
NaN/`jnp.where`.** Ports the source's `AvailabilityModel(i_plant_availability) ==
AvailabilityModel.USER_INPUT` precondition check exactly: `i_plant_availability` is a
static switch (which availability *model* computed `.costs.f_t_plant_available` --
Ward-Taylor/Morris/ST -- or whether it's just raw user input), so this is the same shape
`constraints.py`'s `constraint_24` already established a precedent for (a static/switch
value outside its valid domain for *this* call site raises a real `ValueError`/
`ProcessValueError`, since nothing is traced at that branch point) -- not the continuous-
domain-guard shape (`sqrt`/`log` on a bad continuous input) that becomes a silent NaN
elsewhere in this codebase. See `objectives.md`'s hole-in-MDA note on this one: a plant
run in `USER_INPUT` mode has no *model* for `f_t_plant_available` to differentiate
through at all, so "maximise it" is not a well-posed optimisation regardless of how this
port represents the precondition.
"""

from functional_process.vocabulary import FiguresOfMerit
from functional_process.vocabulary import AvailabilityModel


def objective_metric_1(rmajor):
    """`FiguresOfMerit.MAJOR_RADIUS`. Ports `objectives.py:57`."""
    return 0.2 * rmajor


def objective_metric_3(pflux_fw_neutron_mw):
    """`FiguresOfMerit.NEUTRON_WALL_LOAD`. Ports `objectives.py:59`."""
    return pflux_fw_neutron_mw


def objective_metric_4(tfcmw, srcktpm):
    """`FiguresOfMerit.P_TF_PLUS_P_PF`. Ports `objectives.py:61`.

    `srcktpm` is PF coil circuit power (kW, per its PROCESS name); the `1e-3` converts
    it to MW before summing with `tfcmw` (TF coil resistive power, MW already).
    """
    return (tfcmw + 1.0e-3 * srcktpm) / 10.0


def objective_metric_5(big_q_plasma):
    """`FiguresOfMerit.FUSION_GAIN_Q`. Ports `objectives.py:63`."""
    return big_q_plasma


def objective_metric_6(coe):
    """`FiguresOfMerit.COST_OF_ELECTRICITY`. Ports `objectives.py:65`."""
    return coe / 100.0


def objective_metric_7(cdirt, concost, ireactor):
    """`FiguresOfMerit.CAPITAL_COST`. Ports `objectives.py:67-71`.

    `ireactor` (0: cost model reports `cdirt`, direct/constructed cost; nonzero:
    `concost`, total construction cost including indirect costs) is a static switch --
    it picks which of two already-produced cost totals to read, not a continuous input.
    """
    return cdirt / 1.0e3 if ireactor == 0 else concost / 1.0e4


def objective_metric_8(aspect):
    """`FiguresOfMerit.ASPECT_RATIO`. Ports `objectives.py:73`."""
    return aspect


def objective_metric_9(pflux_div_heat_load_mw):
    """`FiguresOfMerit.DIVERTOR_HEAT_LOAD`. Ports `objectives.py:75`."""
    return pflux_div_heat_load_mw


def objective_metric_10(b_plasma_toroidal_on_axis):
    """`FiguresOfMerit.TOROIDAL_FIELD`. Ports `objectives.py:77`."""
    return b_plasma_toroidal_on_axis


def objective_metric_11(p_hcd_injected_total_mw):
    """`FiguresOfMerit.TOTAL_INJECTED_POWER`. Ports `objectives.py:79`."""
    return p_hcd_injected_total_mw


def objective_metric_14(t_plant_pulse_burn):
    """`FiguresOfMerit.PULSE_LENGTH`. Ports `objectives.py:81`."""
    return t_plant_pulse_burn / 2.0e4


def objective_metric_15(i_plant_availability, f_t_plant_available):
    """`FiguresOfMerit.PLANT_AVAILABILITY_FACTOR`. Ports `objectives.py:82-91`.

    Parameters
    ----------
    i_plant_availability :
        `AvailabilityModel` switch. Static -- see the module docstring's precondition
        note. Must not be `AvailabilityModel.USER_INPUT` (0).
    f_t_plant_available :
        Plant availability factor, as computed by whichever non-user-input
        `AvailabilityModel` is selected.

    Raises
    ------
    ValueError
        If `i_plant_availability == AvailabilityModel.USER_INPUT` -- there is no model
        output to optimise against in that mode, matching the source's own
        `ProcessValueError`.
    """
    if AvailabilityModel(i_plant_availability) == AvailabilityModel.USER_INPUT:
        raise ValueError(
            "objective_metric_15 (plant availability factor): "
            "`i_plant_availability` is AvailabilityModel.USER_INPUT -- "
            "`f_t_plant_available` is not a model output in this mode, "
            "so there is nothing to optimise against."
        )
    return f_t_plant_available


def objective_metric_16(rmajor, t_plant_pulse_burn):
    """`FiguresOfMerit.MIN_R0_MAX_TAU_BURN`. Ports `objectives.py:93-95`.

    "Linear combination of major radius (minimised) and pulse length (maximised)"
    (`FiguresOfMerit` enum's own description -- **not** "major radius/burn time", the
    stale duplicate text in `objective_function`'s inline docstring list; see this
    module's own docstring).
    """
    return 0.95 * (rmajor / 9.0) - 0.05 * (t_plant_pulse_burn / 7200.0)


def objective_metric_17(p_plant_electric_net_mw):
    """`FiguresOfMerit.NET_ELECTRICAL_OUTPUT`. Ports `objectives.py:97`."""
    return p_plant_electric_net_mw / 500.0


def objective_metric_18():
    """`FiguresOfMerit.NULL_FIGURE_OF_MERIT`. Ports `objectives.py:99`.

    `f(x) = 1` identically -- no arguments, no `data` read at all.
    """
    return 1.0


def objective_metric_19(big_q_plasma, t_plant_pulse_burn):
    """`FiguresOfMerit.MAX_Q_MAX_T_PLANT_PULSE_BURN`. Ports `objectives.py:101-103`.

    "Linear combination of big Q and pulse length (maximised)" (`FiguresOfMerit` enum's
    own description) -- a genuinely different formula from id 16 despite
    `objective_function`'s inline docstring listing both as "major radius/burn time";
    see this module's own docstring.
    """
    return -0.5 * (big_q_plasma / 20.0) - 0.5 * (t_plant_pulse_burn / 7200.0)


OBJECTIVE_METRICS = {
    FiguresOfMerit.MAJOR_RADIUS: objective_metric_1,
    FiguresOfMerit.NEUTRON_WALL_LOAD: objective_metric_3,
    FiguresOfMerit.P_TF_PLUS_P_PF: objective_metric_4,
    FiguresOfMerit.FUSION_GAIN_Q: objective_metric_5,
    FiguresOfMerit.COST_OF_ELECTRICITY: objective_metric_6,
    FiguresOfMerit.CAPITAL_COST: objective_metric_7,
    FiguresOfMerit.ASPECT_RATIO: objective_metric_8,
    FiguresOfMerit.DIVERTOR_HEAT_LOAD: objective_metric_9,
    FiguresOfMerit.TOROIDAL_FIELD: objective_metric_10,
    FiguresOfMerit.TOTAL_INJECTED_POWER: objective_metric_11,
    FiguresOfMerit.PULSE_LENGTH: objective_metric_14,
    FiguresOfMerit.PLANT_AVAILABILITY_FACTOR: objective_metric_15,
    FiguresOfMerit.MIN_R0_MAX_TAU_BURN: objective_metric_16,
    FiguresOfMerit.NET_ELECTRICAL_OUTPUT: objective_metric_17,
    FiguresOfMerit.NULL_FIGURE_OF_MERIT: objective_metric_18,
    FiguresOfMerit.MAX_Q_MAX_T_PLANT_PULSE_BURN: objective_metric_19,
}
"""`FiguresOfMerit` member -> the `objective_metric_<id>` function above that ports it.

A graph-assembly-time lookup (mirrors `element2index` in `impurity_radiation.py`: keyed
by a Python enum, resolved once when building an `Optimise` problem, never traced) --
not itself a port of `objective_function`'s `if`/`elif` chain, since that chain *is*
this dict plus "apply `np.sign(i_figure_merit)` to the result", both assembly-time
concerns per the module docstring above.
"""
