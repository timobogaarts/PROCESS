"""The zero-boundary gradient check, and the register of what it may still find.

**What this closes.** `_audit/next_steps.md` §9 and §10 record a defect class this
project hit four times: a ported body that is **value-correct everywhere and non-finite
in derivative at one point**. The canonical shape is `x ** p` with `0 < p < 1` (or
`jnp.sqrt`) evaluated at exactly `x == 0`: the value is `0`, which is right, while the
JVP is `p * x ** (p - 1) * dx = inf * dx` -- `+inf` along the direction that perturbs
`x` and `nan` (`inf * 0`) along every other. Every value test passes. Only a gradient
sees it, and the last instance was diagnosed by the SQP solver as "the problem seems to
be non-convex" after hours of looking. `models/safe_math.py` is the fix; this module is
the check that stops the fifth instance from costing the same.

**The check.** For one point per contract, set each differentiable argument component to
`0.0` in turn and require: *if the value at that point is finite, the Jacobian there
must be finite too.* That is exactly the class -- a point PROCESS itself evaluates
without complaint, where the port answers correctly and differentiates to `nan`. Where
zeroing the argument makes the value itself non-finite the point is outside the
function's domain, `test_outputs_finite`/`test_value_agreement` already own it, and this
check steps aside.

**What it is not.** It is a *structural* probe, not a physical one: nothing claims a
device can run with zero toroidal field. The claim is narrower and is the one that
matters for a solver -- a cold `DataStructure` has `0.0` in every model-computed field
(`next_steps.md` §11.6's cold-start gap), and one `nan` cell anywhere in the Jacobian
stops VMCON at zero iterations regardless of how unphysical the point that produced it
was.

**The register below is a second, different defect class, deliberately not fixed here.**
Every remaining entry is an *unguarded division* `a / x` at `x == 0`: the primal goes to
`inf`, and a downstream `jnp.minimum`/`jnp.maximum`/`jnp.exp`/negative power pulls it
back to a finite number while the tangent stays `inf`/`nan`. The double-`jnp.where`
idiom does not apply -- the repair is a guarded reciprocal at each site, and deciding
what the value *should* be where PROCESS divides by zero is a per-site modelling
question, not a mechanical one. They are registered rather than suppressed, in the same
spirit as `mda_harness.EXPLAINED_DISAGREEMENTS`: recorded, reported, and not subtracted
from anything.

**A register entry must be earned.** `Tier1Contract.test_gradient_finite_at_zero` fails
a contract whose registered site *stopped* failing, so a fixed site cannot leave a stale
excuse behind, and it fails on any site not registered at all -- which is what makes a
future `x ** 0.5` a test failure rather than another investigation.
"""

DIVISION_BY_ZERO_AT_BOUNDARY = {
    # ---- availability.py: lifetime = fluence_limit / flux, capped by the plant life.
    # `jnp.minimum(a / x, life_plant)` is finite at `x == 0` (the cap wins) but the
    # tangent through `a / x` is not. PROCESS returns `inf` here and numpy's own
    # `min` then picks `life_plant`, so the port's value is right.
    ("TestAvail", "f_t_plant_available"): (
        "availability.py:613, life_blkt_fpy / f_t_plant_available"
    ),
    ("TestAvail2", "pflux_div_heat_load_mw"): (
        "availability.py:126, adivflnc / pflux_div_heat_load_mw"
    ),
    ("TestAvail2", "pflux_fw_neutron_mw"): (
        "availability.py:665, abktflnc / pflux_fw_neutron_mw"
    ),
    ("TestAvailSt", "pflux_div_heat_load_mw"): (
        "availability.py:126, adivflnc / pflux_div_heat_load_mw"
    ),
    ("TestAvailSt", "pflux_fw_neutron_mw"): (
        "availability.py:176, cpstflnc / pflux_fw_neutron_mw"
    ),
    ("TestBlanketLifetimeFpySimple", "pflux_fw_neutron_mw"): (
        "availability.py:665, abktflnc / pflux_fw_neutron_mw"
    ),
    ("TestCpLifetimeResistive", "pflux_fw_neutron_mw"): (
        "availability.py:176, cpstflnc / pflux_fw_neutron_mw"
    ),
    ("TestCplifeLifetimeAdjustment", "f_t_plant_available"): (
        "availability.py:1182, cplife / f_t_plant_available"
    ),
    ("TestDivertorLifetime", "pflux_div_heat_load_mw"): (
        "availability.py:126, adivflnc / pflux_div_heat_load_mw"
    ),
    ("TestFwBlanketShieldGeometry", "pflux_fw_neutron_mw"): (
        "stellarator_fwbs_s1_s5.py:112, abktflnc / pflux_fw_neutron_mw"
    ),
    ("TestUPlanned", "pflux_div_heat_load_mw"): (
        "availability.py:126, adivflnc / pflux_div_heat_load_mw"
    ),
    ("TestUPlanned", "pflux_fw_neutron_mw"): (
        "availability.py:665, abktflnc / pflux_fw_neutron_mw"
    ),
    # `tmargmin / conf_mag` and the `safe_denom` reciprocal beside it: the existing
    # single `jnp.where` guards the *denominator's* zero, not the numerator's.
    ("TestAvail2", "conf_mag"): "availability.py:212, tmargmin / conf_mag",
    ("TestAvail2", "temp_margin"): (
        "availability.py:220, t_plant_operational_total_yrs / safe_denom"
    ),
    ("TestAvailSt", "conf_mag"): "availability.py:212, tmargmin / conf_mag",
    ("TestAvailSt", "temp_margin"): (
        "availability.py:220, t_plant_operational_total_yrs / safe_denom"
    ),
    ("TestUUnplannedMagnets", "conf_mag"): "availability.py:212, tmargmin / conf_mag",
    ("TestUUnplannedMagnets", "temp_margin"): (
        "availability.py:220, t_plant_operational_total_yrs / safe_denom"
    ),
    # ---- costs.py: coeoam = 1e9 * annoam / kwhpy, and kwhpy is linear in the pulse
    # length, so a zero-length pulse divides by zero.
    ("TestCostOfElectricity", "t_plant_pulse_total"): (
        "costs.py:2242, 1e9 * annoam / kwhpy; kwhpy is linear in t_plant_pulse_total"
    ),
    # ---- stellarator_fwbs_s2.py: exp(-thickness / decay_length) with a zero decay
    # length. The exponential saturates to a finite 1 - 0; its tangent does not.
    ("TestDetailedPowerflowBlanketShieldPower", "declblkt"): (
        "stellarator_fwbs_s2.py:307, exp(-dr_blkt_inboard / decaybzi)"
    ),
    ("TestDetailedPowerflowBlanketShieldPower", "declfw"): (
        "stellarator_fwbs_s2.py:298, exp(-2 * bfwi / decayfwi)"
    ),
    ("TestDetailedPowerflowBlanketShieldPower", "declshld"): (
        "stellarator_fwbs_s2.py:340, exp(-dr_shld_inboard / decayshldi)"
    ),
    # ---- physics: a density/geometry ratio whose denominator is the zeroed argument.
    ("TestFastAlphaBetaIpdg89", "nd_plasma_electrons_vol_avg"): (
        "pure_formulas.py:323, (nd_fuel_ions / nd_electrons) ** 2"
    ),
    ("TestFastAlphaBetaWard", "nd_plasma_electrons_vol_avg"): (
        "pure_formulas.py:323, same ratio"
    ),
    ("TestGreenwaldDensityFractions", "rminor"): (
        "profiles.py:88, plasma_current / (pi * rminor ** 2)"
    ),
    ("TestRadiationPowers", "b_plasma_toroidal_on_axis"): (
        "radiation_power.py:318, p_a0 = 6.04e3 * rminor * ne0_20 / b_toroidal"
    ),
    ("TestRadiationPowers", "nd_plasma_electron_on_axis"): (
        "radiation_power.py:332, temp_electron_on_axis_kev / p_a0 ** 0.41, p_a0 zero"
    ),
    ("TestRadiationPowers", "rminor"): (
        "radiation_power.py:332, same reciprocal of p_a0"
    ),
    ("TestSynchrotronRadiationPower", "b_plasma_toroidal_on_axis"): (
        "radiation_power.py:318, p_a0 reciprocal"
    ),
    ("TestSynchrotronRadiationPower", "nd_plasma_electron_on_axis"): (
        "radiation_power.py:332, p_a0 reciprocal"
    ),
    ("TestSynchrotronRadiationPower", "rminor"): (
        "radiation_power.py:332, p_a0 reciprocal"
    ),
    ("TestT10ConfinementTime", "b_plasma_toroidal_on_axis"): (
        "confinement_time.py:259, denfac numerator / (1.3 * b_plasma_toroidal_on_axis)"
    ),
    ("TestMurariConfinementTime", "b_plasma_toroidal_on_axis"): (
        "confinement_time.py:951, (nd_electron_line_19 / b_toroidal) ** -1.365"
    ),
    ("TestMurariConfinementTime", "nd_plasma_electron_line_19"): (
        "confinement_time.py:949, same ratio, zero base under a negative exponent"
    ),
    ("TestLangHighDensityConfinementTime", "plasma_current"): (
        "confinement_time.py:1000, nd_electron_line / n_gw; n_gw ~ plasma_current"
    ),
    ("TestLangHighDensityConfinementTime", "rminor"): (
        "confinement_time.py:999, n_gw = 1e14 * plasma_current / (pi * rminor ** 2)"
    ),
    # ---- vacuum.py / quench.py: reciprocals of a coil count and of a field/current.
    ("TestVacuumPumpingSimple", "n_tf_coils"): (
        "vacuum.py:122, pumpspeed numerator / n_tf_coils"
    ),
    ("TestQuenchProtection", "b_plasma_toroidal_on_axis"): (
        "quench.py:44, (_B_REF_T / b_plasma_toroidal_on_axis * ...) ** -1"
    ),
    ("TestQuenchProtection", "c_tf_total"): (
        "quench.py:44, (... _I_TOTAL_REF_A / c_tf_total ...) ** -1"
    ),
    ("TestQuenchProtection", "rminor"): (
        "quench.py:44, (... _RMINOR_REF_M ** 2 / rminor ** 2) ** -1"
    ),
    # ---- a different singularity again: x ** (c * log(x)). The exponent itself goes
    # to -inf at x == 0, so no `safe_pow` helps; the limit has to be taken by hand.
    ("TestLangHighDensityConfinementTime", "aspect"): (
        "confinement_time.py:1012, aspect ** (-0.9 * log(aspect)) -- log singularity"
    ),
    ("TestLangHighDensityConfinementTime", "nd_plasma_electron_line"): (
        "confinement_time.py:1013, nratio ** (-0.22 * log(nratio)) -- same"
    ),
    # ---- and a third class: `arcsin` at exactly its endpoint. No division and no
    # fractional power is involved; `arcsin` is simply not differentiable at +-1, and
    # the argument lands there *exactly* when the elongation is zero. At `kappa == 0`
    # the plasma arc radii collapse to `rco = 0.5 * rminor * (1 + triang)` and
    # `rci = 0.5 * rminor * |triang - 1|` (the `kappa ** 2` terms vanish), so both
    # `1 - rminor * (1 -+ triang) / rc*` reduce to exactly `-1`. The value is a perfectly
    # good `-pi/2` -- PROCESS itself computes it without complaint -- and the tangent is
    # `inf`. There is no guarded form to write: the derivative genuinely diverges, and
    # any repair would be a modelling choice about a zero-elongation plasma.
    ("TestDivertorGeometryConventional", "kappa"): (
        "build.py:895-899, arcsin(1 - rminor * (1 -+ triang) / rc*) == arcsin(-1) "
        "at kappa == 0"
    ),
    # ---- an ordinary unguarded division, same class as the availability.py entries
    # above: `bp_omp / bt_omp` where `bt_omp = -b_plasma_toroidal_on_axis * rmajor /
    # r_omp`. At `b_plasma_toroidal_on_axis == 0`, `bt_omp == 0` and the ratio is
    # `inf`/`-inf`; `arctan` saturates it to a finite `+-pi/2`, so the value this
    # test probes is finite (a real, if unphysical, angle) while the tangent through
    # the division is not. No `safe_pow` site is involved -- the singularity is the
    # division itself, and a guarded reciprocal would be a modelling choice about
    # what "the flux-expansion angle at zero toroidal field" should mean, not a
    # mechanical fix (per this module's own "what it is not").
    ("TestCalculateDivertorHeatLoadWade", "b_plasma_toroidal_on_axis"): (
        "divertor.py:331,333, bp_omp/bt_omp inside atan(...) -- "
        "bt_omp = -b_plasma_toroidal_on_axis * rmajor / r_omp"
    ),
    # The same site reached through the other `n_divertors` arm (2026-08-27): both
    # occupants call one `_divwade_hldiv_base`, so the singularity is one singularity
    # with two entry points, not two.
    ("TestCalculateDivertorHeatLoadWadeDoubleNull", "b_plasma_toroidal_on_axis"): (
        "divertor.py:331,333, bp_omp/bt_omp inside atan(...) -- "
        "bt_omp = -b_plasma_toroidal_on_axis * rmajor / r_omp"
    ),
    # ---- division *by a quotient*, the same class once unfolded: PROCESS spells the
    # cylindrical safety factor `rminor**2 / (rmajor * plasma_current / b)`. At
    # `b_plasma_toroidal_on_axis == 0` the inner quotient is `inf`, the outer division
    # pulls the value back to a finite `0.0` -- the correct limit, and what PROCESS's own
    # numpy evaluation returns -- while the tangent through `a / b` stays `nan`.
    # Reassociating to `rminor**2 * b / (rmajor * plasma_current)` removes it and is
    # value-equal to ~2.6 ulp (32% of 20 000 fuzz-domain points differ, worst 5.8e-16
    # relative), but that is a rewrite of PROCESS's expression rather than of its model,
    # and this unit's port keeps PROCESS's spelling. Registered, not repaired.
    ("TestCalculateCylindricalSafetyFactor", "b_plasma_toroidal_on_axis"): (
        "physics.py:93-99, rminor**2 / (rmajor * plasma_current / "
        "b_plasma_toroidal_on_axis)"
    ),
    # ---- the Sauter bootstrap scaling's three transport coefficients, all through one
    # division: `_electron_collisionality_sauter` divides by
    # `|inverse_q[j] * sqeps[j]**3 * sqrt(tempe[j]) * 1.875e7|`
    # (`bootstrap_current.py:1728-1733`). Zeroing either `inverse_q[j]` or `sqeps[j]`
    # makes that denominator zero and the collisionality `+inf`; every use of it is of
    # the form `f_trapped / (a + b * nu)`, which saturates to a finite `0` while the
    # tangent through `a / b` stays `nan`. `rmajor == 0` is the same site from the other
    # side -- it zeroes the collisionality's *numerator*, and `_calculate_l31_
    # coefficient`'s trailing `_beta_poloidal_total_sauter` keeps `(rmajor / ...) ** 2`
    # finite there, so again a finite value with a non-finite tangent. (It does not
    # survive into `_calculate_l31_32_coefficient`, where the extra
    # `_beta_poloidal_sauter / _beta_poloidal_total_sauter` quotient cancels the factor
    # that made it singular; hence no `rmajor` entry for that contract.)
    #
    # Registered rather than repaired, per this module's own "what it is not": there is
    # no guarded reciprocal to write without first deciding what the electron
    # collisionality *means* at infinite safety factor or zero inverse aspect ratio,
    # which is a modelling question about an unphysical point. The three functions are
    # additionally covered in value and gradient by `TestBootstrapFractionSauter`, whose
    # own zero-boundary probe is clean -- the scalars that would zero `inverse_q` or
    # `sqeps` there (`q0`, `q95`, `rminor`) make the whole value non-finite, which this
    # check steps aside for.
    ("TestCalculateL31Coefficient", "inverse_q"): (
        "bootstrap_current.py:1728-1733, "
        "670 * ... / |inverse_q * sqeps**3 * sqrt(tempe) * 1.875e7|"
    ),
    ("TestCalculateL31Coefficient", "sqeps"): (
        "bootstrap_current.py:1728-1733, same denominator, sqeps**3 factor"
    ),
    ("TestCalculateL31Coefficient", "rmajor"): (
        "bootstrap_current.py:1723-1727, rmajor zeroes the collisionality numerator "
        "while _beta_poloidal_total_sauter's (rmajor / ...)**2 stays finite"
    ),
    ("TestCalculateL3132Coefficient", "inverse_q"): (
        "bootstrap_current.py:1728-1733, same denominator, via f32ee/f32ei_teff"
    ),
    ("TestCalculateL3132Coefficient", "sqeps"): (
        "bootstrap_current.py:1728-1733, same denominator, sqeps**3 factor"
    ),
    ("TestCalculateL34Alpha31Coefficient", "inverse_q"): (
        "bootstrap_current.py:1728-1733, same denominator, via f34_teff"
    ),
}
"""`(contract class name, argument name)` -> where the `inf` primal is produced.

Keyed by contract rather than by argument name because the same PROCESS variable is
benign in one unit and singular in another -- `pflux_fw_neutron_mw` divides in
`availability.py` and merely multiplies elsewhere. A global by-name list would exempt
both.
"""


def registered_reason(contract_name, argument):
    """The recorded reason a site is allowed to be non-finite, or `None`.

    Parameters
    ----------
    contract_name :
        `type(contract).__name__`.
    argument :
        The differentiable argument's name, without any array index.

    Returns
    -------
    :
        The reason string, or `None` if the site is not registered.
    """
    return DIVISION_BY_ZERO_AT_BOUNDARY.get((contract_name, argument))
