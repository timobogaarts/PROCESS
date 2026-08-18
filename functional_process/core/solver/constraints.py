"""Pure-functional port of PROCESS's constraint set
(`process/core/solver/constraints.py`).

Audit record: `functional_process/core/solver/constraints.md`.

Covers every constraint PROCESS registers (`@ConstraintManager.register_constraint`,
~82 of them) except 50 and 52, which are IFE-only (`data.ife.ife`-gated) -- the entire
`.ife.*` subsystem has zero producers anywhere in this codebase, a genuine, documented
exclusion (see `constraints.md`'s "Constraints considered and excluded" section), not a
porting gap. Originally scoped to five stellarator-specific constraints only (17, 24, 82,
83, 91); broadened to the full general-constraint set in a later pass, ported without a
stellarator-relevance filter this time -- most of what's here is ordinary tokamak/general
physics-and-engineering bookkeeping, not stellarator-specific at all.

**Shape**: the overwhelming majority are the "bare residual read" case
(`_audit/naming_convention.md`, `CLAUDE.md`'s logical-mapping table row on
`ConstraintManager`), not `cottax.rewrites.Compare` -- the bound/limit operand is a
plain, already-produced `data` field, never the result of a `calculate_*` re-derivation
happening *inside* the constraint body itself. Constraint 1 is the one exception found so
far: it calls a re-derived `calculate_plasma_beta` and compares it to a stored field --
that pattern is `Compare`-shaped (`CLAUDE.md`'s own cited example), ported alongside it.

A constraint residual is also not stored anywhere in `DataStructure` --
`ConstraintManager.evaluate_constraint` returns it straight to the solver -- so there is
no `VarPath` for a node to *own*, and therefore no `ExplicitFunction` wrapper to write
here (see `naming_convention.md`; `_audit/schema.md`'s "cottax node" section is a
model-unit thing). What is ported is a plain pure function per constraint, wired into an
`Optimise` problem's conditions at graph-assembly time -- later work, not this task (see
`_audit/next_steps.md` §6) -- and tested the same way any other tier-1 unit is
(`_audit/test_harness.md` § Tier 1: "Constraints ... belong here too for testing
purposes, even though structurally they're Compare/condition shapes").

`leq`/`geq`/`eq` below port `process/core/solver/constraints.py`'s closure helpers of the
same name verbatim (same formulas), minus the `ConstraintRegistration` bookkeeping
(name/units/symbol -- output-formatting metadata, not physics). Each returns
`(residual, normalised_residual, constraint_value, constraint_bound)`, the same four
numeric fields `ConstraintResult` carries. `eq` was added once the first equality
constraint (1) needed it -- several later batches independently re-derived it in
isolation while porting in parallel; this is the single consolidated copy.

Switch arguments (`i_plasma_ignited`, `istell`, `i_beta_component`, `i_rad_loss`,
`i_density_limit`, and others introduced by individual constraints below) are static
Python ints, branched on with plain `if`/`elif` -- per `naming_convention.md` §
"switches are not ports", they select a formula rather than flowing along a graph edge,
and a caller differentiating one of these functions must exclude them
(`static_argnames` in the harness contracts below).
"""

from process.core import constants
from process.data_structure.build_variables import TFCSRadialConfiguration
from process.data_structure.physics_variables import PlasmaIgnitionModel
from process.models.physics.density_limit import DensityLimitModel
from process.models.physics.physics import BetaComponentLimits
from process.models.tfcoil.base import TFConductorModel


def leq(value, bound):
    """`value <= bound`. Ports `process.core.solver.constraints.leq`.

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    residual = value - bound
    normalised_residual = (value / bound) - 1.0
    return residual, normalised_residual, value, bound


def geq(value, bound):
    """`value >= bound`. Ports `process.core.solver.constraints.geq`.

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    residual = bound - value
    normalised_residual = 1.0 - (value / bound)
    return residual, normalised_residual, value, bound


def eq(value, bound):
    """`value == bound`. Ports `process.core.solver.constraints.eq`.

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    residual = value - bound
    normalised_residual = 1.0 - (value / bound)
    return residual, normalised_residual, value, bound


def calculate_plasma_beta(pres_plasma, b_field):
    """Plasma beta from pressure and field. Ports `PlasmaBeta.calculate_plasma_beta`
    (`process/models/physics/physics.py:3920`) verbatim -- already pure in the source,
    no `self.data` access, same lift-as-is treatment `physics_A_pure_formulas.py`
    already gives five other already-pure formulas from this same class of source.

    `beta = 2 * mu0 * pressure / B**2`.
    """
    return 2.0 * constants.RMU0 * pres_plasma / (b_field**2)


def constraint_1(
    beta_fast_alpha,
    beta_beam,
    nd_plasma_electrons_vol_avg,
    temp_plasma_electron_density_weighted_kev,
    nd_plasma_ions_total_vol_avg,
    temp_plasma_ion_density_weighted_kev,
    b_plasma_total,
    beta_total_vol_avg,
):
    """Relationship between beta, temperature and density. Ports
    `constraint_equation_1`.

    `Compare`-shaped: the LHS's thermal term is a re-derivation
    (`calculate_plasma_beta`, above) from the density-weighted pressure, not a plain
    already-produced field -- the canonical example `CLAUDE.md`'s own mapping table
    cites for this shape.

    **Real PROCESS finding, stellarator-specific**: `Stellarator.run()`
    (`process/models/stellarator/stellarator.py:1917-1930`) raises `ProcessValueError`
    if `beta_total_vol_avg` (iteration variable 5) is in `numerics.ixc` when `istell >
    0` ("Beta should not be in ixc if istell>0. Use Constraints 24 and 84 instead"),
    and then **directly overwrites** `.physics.beta_total_vol_avg` with this exact
    formula's RHS, with the comment "This replaces constraint equation 1 as it is just
    an equality." So on every real stellarator run, constraint 1 is never active --
    PROCESS's own stellarator code path inlines it as a direct assignment instead of an
    equality constraint tied to a free iteration variable. This port is still faithful
    to the general (tokamak-shaped) constraint as written; the finding is recorded here
    because it explains why this constraint should not be expected to appear active in
    any stellarator-scoped test/graph configuration downstream.

    Parameters
    ----------
    beta_fast_alpha, beta_beam :
        Fast-alpha and neutral-beam beta components.
    nd_plasma_electrons_vol_avg, nd_plasma_ions_total_vol_avg :
        Electron and total ion density (m^-3).
    temp_plasma_electron_density_weighted_kev, temp_plasma_ion_density_weighted_kev :
        Density-weighted electron/ion temperature (keV) -- distinct from the
        volume-averaged temperature used elsewhere, since <nT> != <n>*<T>.
    b_plasma_total :
        Total (toroidal + poloidal) magnetic field (T).
    beta_total_vol_avg :
        Total plasma beta (bound/RHS).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    beta_thermal_total_vol_avg = calculate_plasma_beta(
        pres_plasma=(
            constants.KILOELECTRON_VOLT
            * (
                nd_plasma_electrons_vol_avg * temp_plasma_electron_density_weighted_kev
                + nd_plasma_ions_total_vol_avg * temp_plasma_ion_density_weighted_kev
            )
        ),
        b_field=b_plasma_total,
    )
    return eq(
        beta_fast_alpha + beta_beam + beta_thermal_total_vol_avg,
        beta_total_vol_avg,
    )


def constraint_2(
    i_rad_loss,
    i_plasma_ignited,
    pden_electron_transport_loss_mw,
    pden_ion_transport_loss_mw,
    pden_plasma_rad_mw,
    pden_plasma_core_rad_mw,
    f_p_alpha_plasma_deposited,
    pden_alpha_total_mw,
    pden_non_alpha_charged_mw,
    pden_plasma_ohmic_mw,
    p_hcd_injected_total_mw,
    vol_plasma,
):
    """Global power balance equation (total). Ports `constraint_equation_2`.

    Bare residual read: every operand is an already-produced `data` field (physics
    power densities, or `p_hcd_injected_total_mw` from `current_drive.py`), none of them
    re-derived inside the constraint body -- `i_rad_loss`/`i_plasma_ignited` select
    which fields feed the numerator/denominator, they don't change the shape.

    Parameters
    ----------
    i_rad_loss :
        Radiation-loss-term switch (0/1/2). Static -- selects the numerator's
        radiation term, not a differentiable input.
    i_plasma_ignited :
        `PlasmaIgnitionModel` switch. Static -- selects whether injected power is
        included in the denominator.
    pden_electron_transport_loss_mw, pden_ion_transport_loss_mw :
        Electron/ion transport power density (MW/m3).
    pden_plasma_rad_mw, pden_plasma_core_rad_mw :
        Total/core radiation power density (MW/m3).
    f_p_alpha_plasma_deposited :
        Fraction of alpha power deposited in the plasma (input constant, default 0.95).
    pden_alpha_total_mw :
        Alpha power density (MW/m3).
    pden_non_alpha_charged_mw :
        Non-alpha charged-particle fusion power density (MW/m3).
    pden_plasma_ohmic_mw :
        Ohmic heating power density (MW/m3).
    p_hcd_injected_total_mw :
        Total auxiliary injected power (MW). Only used when not ignited.
    vol_plasma :
        Plasma volume (m3).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    pscaling = pden_electron_transport_loss_mw + pden_ion_transport_loss_mw
    if i_rad_loss == 0:
        pnumerator = pscaling + pden_plasma_rad_mw
    elif i_rad_loss == 1:
        pnumerator = pscaling + pden_plasma_core_rad_mw
    else:
        pnumerator = pscaling

    if PlasmaIgnitionModel(i_plasma_ignited) == PlasmaIgnitionModel.NON_IGNITED:
        pdenom = (
            f_p_alpha_plasma_deposited * pden_alpha_total_mw
            + pden_non_alpha_charged_mw
            + pden_plasma_ohmic_mw
            + p_hcd_injected_total_mw / vol_plasma
        )
    else:
        pdenom = (
            f_p_alpha_plasma_deposited * pden_alpha_total_mw
            + pden_non_alpha_charged_mw
            + pden_plasma_ohmic_mw
        )

    return eq(pnumerator, pdenom)


def constraint_3(
    i_plasma_ignited,
    pden_ion_transport_loss_mw,
    pden_ion_electron_equilibration_mw,
    f_p_alpha_plasma_deposited,
    f_pden_alpha_ions_mw,
    p_hcd_injected_ions_mw,
    vol_plasma,
):
    """Global power balance equation for ions. Ports `constraint_equation_3`.

    Bare residual read, same shape as constraint 2 (no re-derivation inside the body).

    Parameters
    ----------
    i_plasma_ignited :
        `PlasmaIgnitionModel` switch. Static.
    pden_ion_transport_loss_mw :
        Ion transport power density (MW/m3).
    pden_ion_electron_equilibration_mw :
        Ion/electron equilibration power density (MW/m3).
    f_p_alpha_plasma_deposited :
        Fraction of alpha power deposited in the plasma.
    f_pden_alpha_ions_mw :
        Alpha power density to ions (MW/m3).
    p_hcd_injected_ions_mw :
        Auxiliary injected power to ions (MW). Only used when not ignited.
    vol_plasma :
        Plasma volume (m3).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    lhs = pden_ion_transport_loss_mw + pden_ion_electron_equilibration_mw

    if PlasmaIgnitionModel(i_plasma_ignited) == PlasmaIgnitionModel.NON_IGNITED:
        rhs = (
            f_p_alpha_plasma_deposited * f_pden_alpha_ions_mw
            + p_hcd_injected_ions_mw / vol_plasma
        )
    else:
        rhs = f_p_alpha_plasma_deposited * f_pden_alpha_ions_mw

    return eq(lhs, rhs)


def constraint_4(
    i_rad_loss,
    i_plasma_ignited,
    pden_electron_transport_loss_mw,
    pden_plasma_rad_mw,
    pden_plasma_core_rad_mw,
    f_p_alpha_plasma_deposited,
    f_pden_alpha_electron_mw,
    pden_ion_electron_equilibration_mw,
    p_hcd_injected_electrons_mw,
    vol_plasma,
):
    """Global power balance equation for electrons. Ports `constraint_equation_4`.

    Bare residual read, same shape as constraints 2/3.

    Parameters
    ----------
    i_rad_loss :
        Radiation-loss-term switch. Static.
    i_plasma_ignited :
        `PlasmaIgnitionModel` switch. Static.
    pden_electron_transport_loss_mw :
        Electron transport power density (MW/m3).
    pden_plasma_rad_mw, pden_plasma_core_rad_mw :
        Total/core radiation power density (MW/m3).
    f_p_alpha_plasma_deposited :
        Fraction of alpha power deposited in the plasma.
    f_pden_alpha_electron_mw :
        Alpha power density to electrons (MW/m3).
    pden_ion_electron_equilibration_mw :
        Ion/electron equilibration power density (MW/m3).
    p_hcd_injected_electrons_mw :
        Auxiliary injected power to electrons (MW). Only used when not ignited.
    vol_plasma :
        Plasma volume (m3).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    pscaling = pden_electron_transport_loss_mw
    if i_rad_loss == 0:
        pnumerator = pscaling + pden_plasma_rad_mw
    elif i_rad_loss == 1:
        pnumerator = pscaling + pden_plasma_core_rad_mw
    else:
        pnumerator = pscaling

    if PlasmaIgnitionModel(i_plasma_ignited) == PlasmaIgnitionModel.NON_IGNITED:
        pdenom = (
            f_p_alpha_plasma_deposited * f_pden_alpha_electron_mw
            + pden_ion_electron_equilibration_mw
            + p_hcd_injected_electrons_mw / vol_plasma
        )
    else:
        pdenom = (
            f_p_alpha_plasma_deposited * f_pden_alpha_electron_mw
            + pden_ion_electron_equilibration_mw
        )

    return eq(pnumerator, pdenom)


def constraint_5(
    i_density_limit,
    nd_plasma_electron_line,
    nd_plasma_electrons_vol_avg,
    nd_plasma_electrons_max,
    f_nd_plasma_electron_limit_max,
):
    """Electron density upper limit. Ports `constraint_equation_5`.

    Bare residual read. `i_density_limit == GREENWALD` selects the line-averaged
    density (`nd_plasma_electron_line`) instead of the volume-averaged one -- the only
    branch, per the source's own comment ("Except when i_density_limit=7... line is
    used, not vol_avg").

    Parameters
    ----------
    i_density_limit :
        `DensityLimitModel` switch (7 = Greenwald). Static.
    nd_plasma_electron_line :
        Line-averaged electron density (m^-3). Only used for the Greenwald branch.
    nd_plasma_electrons_vol_avg :
        Volume-averaged electron density (m^-3). Used for every other branch.
    nd_plasma_electrons_max :
        Density limit from the selected model (m^-3).
    f_nd_plasma_electron_limit_max :
        Scale factor on the limit (input constant, default 1.0).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    bound = nd_plasma_electrons_max * f_nd_plasma_electron_limit_max
    if i_density_limit == DensityLimitModel.GREENWALD:
        return leq(nd_plasma_electron_line, bound)
    return leq(nd_plasma_electrons_vol_avg, bound)


def constraint_6(beta_poloidal_eps, beta_poloidal_eps_max):
    """Epsilon beta-poloidal upper limit. Ports `constraint_equation_6`.

    Bare residual read, both operands plain already-produced/input fields, no switch.

    Parameters
    ----------
    beta_poloidal_eps :
        `eps * beta_poloidal` (inverse aspect ratio times poloidal beta).
    beta_poloidal_eps_max :
        Maximum allowed value (input constant, default 1.38).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return leq(beta_poloidal_eps, beta_poloidal_eps_max)


def constraint_7(i_plasma_ignited, nd_beam_ions_out, nd_beam_ions):
    """Hot beam ion density consistency. Ports `constraint_equation_7`.

    Bare residual read. PROCESS raises if `i_plasma_ignited == IGNITED` (auxiliary
    power, hence beam ions, is excluded from steady-state balance when ignited) --
    reproduced here as a plain `ValueError`, same treatment `constraint_24` already
    gives its own domain-invalid switch value.

    **Real PROCESS finding**: `nd_beam_ions_out`'s only real producer is
    `reactions.beam_fusion` (`process/models/physics/physics.py:617-628`), called
    *conditionally* -- only when `current_drive.c_beam_total != 0` **and**
    `i_plasma_ignited == NON_IGNITED`. If beam current is zero (no neutral beam) on a
    non-ignited run, `nd_beam_ions_out` silently stays at its dataclass default (`0.0`,
    `physics_variables.py:666`) rather than being computed -- constraint 7 only checks
    the ignition switch, not the beam-current one, so it is possible to activate this
    constraint on a configuration where its LHS was never actually computed. Same
    unported-producer status as `beta_beam` (already flagged this session:
    "`beta_beam`'s sole owner is `beam_fusion`, unported"), and the same conditional-
    computation shape as that finding -- not fixed here, this constraint's own math is
    a faithful, complete port regardless of whether `beam_fusion` itself is ported.

    Parameters
    ----------
    i_plasma_ignited :
        `PlasmaIgnitionModel` switch. Static; raises if `IGNITED`.
    nd_beam_ions_out :
        Hot beam ion density, from `beam_fusion` (m^-3) -- see finding above.
    nd_beam_ions :
        Hot beam ion density, from `plasma_composition` (m^-3) -- already ported,
        `PlasmaComposition.nd_beam_ions`, this session.

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.

    Raises
    ------
    ValueError
        If `i_plasma_ignited == PlasmaIgnitionModel.IGNITED`. Static, so nothing here
        is traced.
    """
    if PlasmaIgnitionModel(i_plasma_ignited) == PlasmaIgnitionModel.IGNITED:
        raise ValueError("constraint_7: do not use if i_plasma_ignited=IGNITED")

    return eq(nd_beam_ions_out, nd_beam_ions)


def constraint_8(pflux_fw_neutron_mw, pflux_fw_neutron_max_mw):
    """Neutron wall load upper limit. Ports `constraint_equation_8`.

    Bare residual read, no switch. `pflux_fw_neutron_mw`'s real producer differs by
    device type (`fw.py`/`ife.py`/`stellarator.py`, all unconditional within their own
    branch) -- confirmed present on the stellarator path
    (`stellarator.py:2096-2108`).

    Parameters
    ----------
    pflux_fw_neutron_mw :
        Average neutron wall load (MW/m2).
    pflux_fw_neutron_max_mw :
        Allowable wall load (input constant, default 1.0).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return leq(pflux_fw_neutron_mw, pflux_fw_neutron_max_mw)


def constraint_9(p_fusion_total_mw, p_fusion_total_max_mw):
    """Fusion power upper limit. Ports `constraint_equation_9`.

    Bare residual read, no switch, no hole-in-MDA: `p_fusion_total_mw` is produced by
    `functional_process/models/physics/fusion_reactions.py` (already ported);
    `p_fusion_total_max_mw` is a plain input constant
    (`data_structure/constraint_variables.py`, never computed by any model).

    Parameters
    ----------
    p_fusion_total_mw :
        Fusion power (MW).
    p_fusion_total_max_mw :
        Maximum allowed fusion power (MW) (bound).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return leq(p_fusion_total_mw, p_fusion_total_max_mw)


def constraint_11(rbld, rmajor):
    """Radial build consistency (equality). Ports `constraint_equation_11`.

    Bare residual read, no switch, no hole-in-MDA: `rbld` is produced by
    `functional_process/models/stellarator/build.py`'s `Build` node (already ported,
    same producer constraint 83 already relies on for `required_radial_space`);
    `rmajor` is produced everywhere in this codebase's already-ported physics units.

    Parameters
    ----------
    rbld :
        Sum of build thicknesses to the major radius (m).
    rmajor :
        Plasma major radius (m) (bound).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return eq(rbld, rmajor)


def constraint_12(vs_cs_pf_total_pulse, vs_plasma_total_required):
    """Volt-second capability lower limit. Ports `constraint_equation_12`.

    Bare residual read, no switch. **Hole-in-MDA: yes, both operands.**
    `vs_cs_pf_total_pulse` is produced by `process/models/pfcoil.py:1710` — the PF coil
    subsystem is entirely unported in this codebase (no `functional_process/models/
    pfcoil*` file exists at all). `vs_plasma_total_required` is produced by
    `process/models/physics/physics.py:4889` — inside `physics.py`'s plasma-current/
    inductance section, also not yet ported. Ported anyway per this pass's instruction
    (port unless there's a reason not to — an unported producer is a wiring gap for
    later, not a reason to skip porting the constraint's own arithmetic, same precedent
    constraint 91 already set for `powerht_constraint`/`powerscaling_constraint`).

    PROCESS's own source negates `vs_cs_pf_total_pulse` before comparing (source
    comment: "vs_cs_pf_total_pulse is negative, requires sign change") — reproduced
    here as the caller's responsibility (this function takes the already-negated value
    as its first argument, matching `geq`'s signature; see the harness reference
    adapter for how the sign flip is applied).

    Parameters
    ----------
    vs_cs_pf_total_pulse :
        **Already sign-flipped** (`-data.pf_coil.vs_cs_pf_total_pulse` in the source) —
        total flux swing available for the pulse (Wb), positive.
    vs_plasma_total_required :
        Total volt-seconds needed (Wb) (bound).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return geq(vs_cs_pf_total_pulse, vs_plasma_total_required)


def constraint_13(t_plant_pulse_burn, t_burn_min):
    """Burn time lower limit. Ports `constraint_equation_13`.

    Bare residual read, no switch. **Not a hole-in-MDA for the stellarator
    configuration this codebase currently scopes**: `t_plant_pulse_burn` is not
    produced by any model on a stellarator run — `process/models/stellarator/
    initialization.py:45` sets it to a fixed constant (`3.15576e7`, one year,
    continuous/non-pulsed operation), not a computed value. It only has a real *model*
    producer (`pulse.py`'s `calculate_burn_time`) on the pulsed-tokamak path, out of
    this codebase's current scope. Treated as an ordinary explicit-arg here, matching
    how `PulseDurations` (`stellarator/initialization.py`, already ported) reads it as
    a plain `Input` rather than expecting an upstream producer. `t_burn_min` is a plain
    input constant.

    Parameters
    ----------
    t_plant_pulse_burn :
        Burn time (s).
    t_burn_min :
        Minimum burn time (s) (bound).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return geq(t_plant_pulse_burn, t_burn_min)


def constraint_14(n_beam_decay_lengths_core, n_beam_decay_lengths_core_required):
    """Neutral beam e-decay lengths to plasma centre (equality). Ports
    `constraint_equation_14`.

    Bare residual read, no switch. **Hole-in-MDA: yes.** `n_beam_decay_lengths_core` is
    produced by `process/models/physics/current_drive.py:201,294` — the current-drive
    subsystem (NBI physics) is entirely unported in this codebase. Ported anyway, same
    reasoning as constraint 12. `n_beam_decay_lengths_core_required` is a plain input
    constant.

    Parameters
    ----------
    n_beam_decay_lengths_core :
        Neutral beam e-decay lengths to plasma centre.
    n_beam_decay_lengths_core_required :
        Permitted neutral beam e-decay lengths to plasma centre (bound).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return eq(n_beam_decay_lengths_core, n_beam_decay_lengths_core_required)


def constraint_15(p_plasma_separatrix_mw, p_l_h_threshold_mw, f_h_mode_margin):
    """L-H power threshold limit (H-mode enforcement). Ports
    `constraint_equation_15`.

    Bare residual read, no switch. **Hole-in-MDA: partial.**
    `p_plasma_separatrix_mw` is produced by `functional_process/models/stellarator/
    stellarator_B_st_phys.py` (already ported). `p_l_h_threshold_mw` is produced by
    `process/models/physics/l_h_transition.py:86` — unported in this codebase. Ported
    anyway, same reasoning as constraint 12. `f_h_mode_margin` is a plain input
    constant (margin multiplier, default `1.0`).

    Parameters
    ----------
    p_plasma_separatrix_mw :
        Power conducted to the divertor region (MW).
    p_l_h_threshold_mw :
        L-H mode power threshold (MW).
    f_h_mode_margin :
        Margin multiplier on the threshold (dimensionless).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return geq(p_plasma_separatrix_mw, p_l_h_threshold_mw * f_h_mode_margin)


def constraint_16(p_plant_electric_net_mw, p_plant_electric_net_required_mw):
    """Net electric power lower limit. Ports `constraint_equation_16`.

    Bare residual read, no switch, no hole-in-MDA: `p_plant_electric_net_mw` is
    produced by `functional_process/models/power_C_electric_production.py`'s
    `PlantElectricProduction` node — built and harness-tested, though **not yet
    registered** in `total_process.py` (a separate, already-known gap, see
    `_audit/next_steps.md`'s alternates-audit backlog, not this constraint's concern).
    `p_plant_electric_net_required_mw` is a plain input constant.

    Parameters
    ----------
    p_plant_electric_net_mw :
        Net electric power (MW).
    p_plant_electric_net_required_mw :
        Required net electric power (MW) (bound).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return geq(p_plant_electric_net_mw, p_plant_electric_net_required_mw)


def constraint_17(
    istell,
    f_p_plasma_separatrix_rad,
    f_p_plasma_separatrix_rad_max,
    psolradmw,
    p_plasma_heating_total_mw,
):
    """Plasma radiation fraction upper limit. Ports `constraint_equation_17`.

    General constraint (not stellarator-specific), with an `istell`-gated adjustment:
    when `istell != 0`, a SOL-radiation contribution is subtracted from the fraction
    before comparing against the bound. See the audit record's "real PROCESS finding"
    note — this branch carries its own unresolved upstream `# TODO` in the PROCESS
    source, reproduced here as-is rather than fixed.

    Parameters
    ----------
    istell :
        Stellarator switch (0: tokamak model, nonzero: stellarator model). Static —
        selects the formula, not a differentiable input.
    f_p_plasma_separatrix_rad :
        Plasma radiation fraction at the separatrix.
    f_p_plasma_separatrix_rad_max :
        Maximum allowed plasma radiation fraction at the separatrix (bound).
    psolradmw :
        SOL radiation power (MW). Only used when `istell != 0`.
    p_plasma_heating_total_mw :
        Total plasma heating power (MW). Only used when `istell != 0`.

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    if istell != 0:
        f_rad_sol = psolradmw / p_plasma_heating_total_mw
        value = f_p_plasma_separatrix_rad - f_rad_sol
    else:
        value = f_p_plasma_separatrix_rad

    return leq(value, f_p_plasma_separatrix_rad_max)


def constraint_18(pflux_div_heat_load_mw, pflux_div_heat_load_max_mw):
    """Divertor heat load upper limit. Ports `constraint_equation_18`.

    Bare residual read, no switch, no hole-in-MDA: `pflux_div_heat_load_mw` is produced
    by `functional_process/models/stellarator/divertor.py`'s already-ported `Divertor`
    node. `pflux_div_heat_load_max_mw` is a plain input constant.

    Parameters
    ----------
    pflux_div_heat_load_mw :
        Divertor heat load (MW/m²).
    pflux_div_heat_load_max_mw :
        Heat load limit (MW/m²) (bound).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return leq(pflux_div_heat_load_mw, pflux_div_heat_load_max_mw)


def constraint_19(p_cp_resistive_mw, p_tf_leg_resistive_mw, mvalim):
    """MVA (power) upper limit: resistive TF coil set. Ports `constraint_equation_19`.

    Parameters
    ----------
    p_cp_resistive_mw :
        Peak resistive TF coil inboard leg power, total (MW).
    p_tf_leg_resistive_mw :
        TF coil outboard leg resistive power, total (MW).
    mvalim :
        MVA limit for resistive TF coil set, total (MW) (bound).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    totmva = p_cp_resistive_mw + p_tf_leg_resistive_mw
    return leq(totmva, mvalim)


def constraint_20(radius_beam_tangency, radius_beam_tangency_max):
    """Neutral beam tangency radius upper limit. Ports `constraint_equation_20`.

    Parameters
    ----------
    radius_beam_tangency :
        Neutral beam centreline tangency radius (m).
    radius_beam_tangency_max :
        Maximum tangency radius for centreline of beam (m) (bound).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return leq(radius_beam_tangency, radius_beam_tangency_max)


def constraint_21(rminor, rminor_min):
    """Minor radius lower limit. Ports `constraint_equation_21`.

    Parameters
    ----------
    rminor :
        Plasma minor radius (m).
    rminor_min :
        Minimum minor radius (m) (bound).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return geq(rminor, rminor_min)


def constraint_22(p_l_h_threshold_mw, f_l_mode_margin, p_plasma_separatrix_mw):
    """L-H power threshold limit, to enforce L-mode. Ports `constraint_equation_22`.

    `p_l_h_threshold_mw >= f_l_mode_margin * p_plasma_separatrix_mw`.

    Parameters
    ----------
    p_l_h_threshold_mw :
        L-H mode power threshold (MW).
    f_l_mode_margin :
        Margin factor on the constraint (1.0 = no margin).
    p_plasma_separatrix_mw :
        Power conducted to the divertor region (MW).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return geq(p_l_h_threshold_mw, f_l_mode_margin * p_plasma_separatrix_mw)


def constraint_23(
    rminor,
    dr_fw_plasma_gap_outboard,
    dr_fw_outboard,
    dr_blkt_outboard,
    f_r_conducting_wall,
):
    """Conducting shell radius / rminor upper limit. Ports `constraint_equation_23`.

    Parameters
    ----------
    rminor :
        Plasma minor radius (m).
    dr_fw_plasma_gap_outboard :
        Gap between plasma and first wall, outboard side (m).
    dr_fw_outboard :
        Outboard first wall thickness, initial estimate (m).
    dr_blkt_outboard :
        Outboard blanket thickness (m).
    f_r_conducting_wall :
        Maximum ratio of conducting wall distance to plasma minor radius, for
        vertical stability (bound multiplier).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    rcw = rminor + dr_fw_plasma_gap_outboard + dr_fw_outboard + dr_blkt_outboard
    return leq(rcw, f_r_conducting_wall * rminor)


def constraint_24(
    i_beta_component,
    istell,
    beta_total_vol_avg,
    beta_thermal_vol_avg,
    beta_beam,
    beta_toroidal_vol_avg,
    beta_vol_avg_max,
):
    """Beta upper limit. Ports `constraint_equation_24`.

    `value` is one of four already-computed beta fields, selected by
    `i_beta_component` -- except `istell != 0` overrides that selection and always
    picks the total-beta field. See the audit record's "real PROCESS finding" note:
    this override silently ignores `i_beta_component` on every stellarator run.

    Parameters
    ----------
    i_beta_component :
        `BetaComponentLimits` switch selecting which beta component the limit applies
        to. Static, ignored when `istell != 0`.
    istell :
        Stellarator switch. Static; nonzero forces the total-beta branch.
    beta_total_vol_avg :
        Total plasma beta.
    beta_thermal_vol_avg :
        Thermal beta component.
    beta_beam :
        Neutral beam beta component.
    beta_toroidal_vol_avg :
        Toroidal beta component.
    beta_vol_avg_max :
        Allowable beta (bound).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.

    Raises
    ------
    ValueError
        If `i_beta_component` is not a member of `BetaComponentLimits`. Legitimate:
        `i_beta_component` is a static argument, so nothing here is traced.
    """
    if i_beta_component == BetaComponentLimits.TOTAL or istell != 0:
        value = beta_total_vol_avg
    elif i_beta_component == BetaComponentLimits.THERMAL:
        value = beta_thermal_vol_avg
    elif i_beta_component == BetaComponentLimits.THERMAL_AND_BEAM:
        value = beta_thermal_vol_avg + beta_beam
    elif i_beta_component == BetaComponentLimits.TOROIDAL:
        value = beta_toroidal_vol_avg
    else:
        raise ValueError(
            f"constraint_24: i_beta_component={i_beta_component!r} is not a member of "
            f"BetaComponentLimits"
        )

    return leq(value, beta_vol_avg_max)


def constraint_25(b_tf_inboard_peak_with_ripple, b_tf_inboard_max):
    """Peak toroidal field upper limit. Ports `constraint_equation_25`.

    Parameters
    ----------
    b_tf_inboard_peak_with_ripple :
        Mean peak field at TF coil, with ripple (T).
    b_tf_inboard_max :
        Maximum peak toroidal field (T) (bound).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return leq(b_tf_inboard_peak_with_ripple, b_tf_inboard_max)


def constraint_26(j_cs_flat_top_end, j_cs_critical_flat_top_end, fjohc):
    """Central Solenoid current density upper limit at end-of-flattop.

    Ports `constraint_equation_26`.

    Parameters
    ----------
    j_cs_flat_top_end :
        Central Solenoid overall current density at end of flat-top (A/m2).
    j_cs_critical_flat_top_end :
        Allowable Central Solenoid current density at end of flat-top (A/m2).
    fjohc :
        Margin for Central Solenoid current at end-of-flattop (bound).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return leq(j_cs_flat_top_end / j_cs_critical_flat_top_end, fjohc)


def constraint_27(j_cs_pulse_start, j_cs_critical_pulse_start, fjohc0):
    """Central Solenoid current density upper limit at beginning-of-pulse.

    Ports `constraint_equation_27`.

    Parameters
    ----------
    j_cs_pulse_start :
        Central Solenoid overall current density at beginning of pulse (A/m2).
    j_cs_critical_pulse_start :
        Allowable Central Solenoid current density at beginning of pulse (A/m2).
    fjohc0 :
        Margin for Central Solenoid current at beginning of pulse (bound).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return leq(j_cs_pulse_start / j_cs_critical_pulse_start, fjohc0)


def constraint_28(i_plasma_ignited, big_q_plasma, big_q_plasma_min):
    """Fusion gain (big Q) lower limit. Ports `constraint_equation_28`.

    PROCESS raises if this constraint is used with an ignited plasma
    (`i_plasma_ignited != NON_IGNITED`) -- "Obviously, ignite must be zero if current
    drive is required." `i_plasma_ignited` is static (a graph-assembly-time switch, per
    `naming_convention.md`), so the raise happens at trace time on an invalid
    configuration, exactly like constraint 24's `i_beta_component` raise already in the
    canonical module -- same convention, plain `ValueError`, not
    `process.core.exceptions.ProcessValueError` (no reason to pull that dependency in
    for a port that never actually executes inside PROCESS's own exception-handling
    context).

    Parameters
    ----------
    i_plasma_ignited :
        `PlasmaIgnitionModel` switch. Static; must be `NON_IGNITED` for this constraint
        to be valid at all.
    big_q_plasma :
        Fusion gain, P_fusion / (P_injection + P_ohmic).
    big_q_plasma_min :
        Minimum allowed fusion gain (bound).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.

    Raises
    ------
    ValueError
        If `i_plasma_ignited != PlasmaIgnitionModel.NON_IGNITED`.
    """
    if PlasmaIgnitionModel(i_plasma_ignited) != PlasmaIgnitionModel.NON_IGNITED:
        raise ValueError("constraint_28: not valid if i_plasma_ignited != NON_IGNITED")

    return geq(big_q_plasma, big_q_plasma_min)


def constraint_29(rmajor, rminor, rinboard):
    """Inboard major radius consistency. Ports `constraint_equation_29`.

    An equality constraint: `rmajor - rminor == rinboard`.

    Parameters
    ----------
    rmajor, rminor :
        Plasma major/minor radius (m).
    rinboard :
        Plasma inboard radius (m) (bound).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return eq(rmajor - rminor, rinboard)


def constraint_30(p_hcd_injected_total_mw, p_hcd_injected_max):
    """Injection power upper limit. Ports `constraint_equation_30`.

    Parameters
    ----------
    p_hcd_injected_total_mw :
        Total auxiliary injected power (MW).
    p_hcd_injected_max :
        Maximum allowable injected power (MW) (bound).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return leq(p_hcd_injected_total_mw, p_hcd_injected_max)


def constraint_31(sig_tf_case, sig_tf_case_max):
    """TF coil case stress upper limit (SCTF). Ports `constraint_equation_31`.

    **Hole-in-MDA: yes, structurally, not just a porting gap.** `.tfcoil.sig_tf_case`
    is written only by `process/models/tfcoil/superconducting.py` (the general/tokamak
    TF coil model) -- confirmed by grepping every writer of `.tfcoil.sig_tf_case` across
    `process/models`. `process/core/caller.py:272-275` shows `_call_models_once` never
    calls `self.tfcoil.run()` for a stellarator configuration at all
    (`if self.data.stellarator.istell != 0: self.models.stellarator.run(); return` --
    an early return, before the tokamak TF coil model would otherwise run). So on any
    real stellarator run, `.tfcoil.sig_tf_case` never leaves its `DataStructure` default
    (`0.0`, `tfcoil_variables.py:477`) -- this constraint is vacuously satisfied
    (`0.0 <= 6e8`) on every stellarator run, not because the physics is trivially fine,
    but because PROCESS itself never computes the quantity for this device type. See
    `batch3.md` for the full writeup. Ported here anyway (the arithmetic itself is
    trivial and faithful to the source), but flagged clearly rather than silently
    treated as a normal, meaningful constraint.

    Parameters
    ----------
    sig_tf_case :
        TF coil case stress (Pa, Tresca criterion).
    sig_tf_case_max :
        Allowable maximum (Pa) (bound).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return leq(sig_tf_case, sig_tf_case_max)


def constraint_32(sig_tf_wp, sig_tf_wp_max):
    """TF coil conduit stress upper limit (SCTF). Ports `constraint_equation_32`.

    Clean, unlike its neighbour 31: `.tfcoil.sig_tf_wp` *is* computed for stellarators,
    by the stellarator-specific `coils/forces.py` (`MaxForceDensity`, already ported).
    `sig_tf_wp_max` is a plain input constant (`tfcoil_variables.py:48`, default `6e8`,
    never written by any model). No hole.

    Parameters
    ----------
    sig_tf_wp :
        TF conductor conduit stress (Pa, Tresca criterion).
    sig_tf_wp_max :
        Allowable maximum (Pa) (bound).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return leq(sig_tf_wp, sig_tf_wp_max)


def constraint_33(j_tf_wp, j_tf_wp_critical, f_j_tf_wp_critical_max):
    """TF coil operating/critical current density upper limit (SCTF). Ports
    `constraint_equation_33`.

    **Hole-in-MDA: yes, same shape as constraint 31.** `.tfcoil.j_tf_wp_critical` is
    written only by `process/models/tfcoil/superconducting.py`, which
    `_call_models_once` never calls for a stellarator run (see constraint 31's
    docstring for the full trace). Stellarator's own `j_tf_wp` (the numerator) *is*
    computed (`coils/calculate.py`), but the critical-current bound it is compared
    against never leaves its `DataStructure` default (`0.0`,
    `tfcoil_variables.py:370`) -- so `j_tf_wp / 0.0` is a genuine division-by-default-
    zero, not a meaningful comparison, on every real stellarator run. Ported anyway,
    flagged the same way as 31.

    Parameters
    ----------
    j_tf_wp :
        TF coil winding pack current density (A/m2).
    j_tf_wp_critical :
        Critical current density for the winding pack (A/m2).
    f_j_tf_wp_critical_max :
        Margin factor (dimensionless) -- the actual bound is
        `j_tf_wp_critical * f_j_tf_wp_critical_max`.

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return leq(j_tf_wp, j_tf_wp_critical * f_j_tf_wp_critical_max)


def constraint_34(v_tf_coil_dump_quench_kv, v_tf_coil_dump_quench_max_kv):
    """TF coil dump voltage upper limit (SCTF). Ports `constraint_equation_34`.

    Clean: both operands are real for stellarators. `.tfcoil.v_tf_coil_dump_quench_kv`
    is computed by the stellarator-specific `coils/quench.py` (`QuenchProtection`,
    already ported); `_max_kv` is a plain input constant, never written by any model.

    Parameters
    ----------
    v_tf_coil_dump_quench_kv :
        Voltage across a TF coil during quench (kV).
    v_tf_coil_dump_quench_max_kv :
        Maximum allowed voltage (kV) (bound).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return leq(v_tf_coil_dump_quench_kv, v_tf_coil_dump_quench_max_kv)


def constraint_35(j_tf_wp, j_tf_wp_quench_heat_max):
    """TF coil J_wp upper limit for quench protection. Ports
    `constraint_equation_35`.

    Clean: both operands are real for stellarators. `.tfcoil.j_tf_wp` is computed by
    the stellarator-specific `coils/calculate.py`; `j_tf_wp_quench_heat_max` is *also*
    computed (by `coils/quench.py`'s `QuenchProtection`, already ported) rather than
    being a plain input constant -- confirmed by grepping its writer, unlike most of
    this batch's other `_max` bounds.

    Parameters
    ----------
    j_tf_wp :
        TF coil winding pack current density (A/m2).
    j_tf_wp_quench_heat_max :
        Allowable current density for dump-temperature-rise protection (A/m2) (bound).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return leq(j_tf_wp, j_tf_wp_quench_heat_max)


def constraint_36(
    temp_tf_superconductor_margin,
    temp_tf_superconductor_margin_min,
):
    """TF coil superconductor temperature margin lower limit. Ports
    `constraint_equation_36`.

    Parameters
    ----------
    temp_tf_superconductor_margin :
        TF coil temperature margin (K).
    temp_tf_superconductor_margin_min :
        Minimum allowable temperature margin for TF coils (K) (bound).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return geq(temp_tf_superconductor_margin, temp_tf_superconductor_margin_min)


def constraint_37(
    eta_cd_norm_hcd_primary,
    eta_cd_norm_hcd_primary_max,
):
    """Current drive gamma upper limit. Ports `constraint_equation_37`.

    Parameters
    ----------
    eta_cd_norm_hcd_primary :
        Normalised current drive efficiency (1e20 A/W-m^2).
    eta_cd_norm_hcd_primary_max :
        Maximum current drive gamma (bound).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return leq(eta_cd_norm_hcd_primary, eta_cd_norm_hcd_primary_max)


def constraint_39(
    temp_fw_peak,
    temp_fw_max,
):
    """First wall temperature upper limit. Ports `constraint_equation_39`.

    PROCESS raises `ProcessValueError` if `temp_fw_peak < 1.0` (a proxy for
    `i_pulsed_plant == 0`, under which this constraint's bound is meaningless) -- not
    reproduced, see this module's own docstring.

    Parameters
    ----------
    temp_fw_peak :
        Peak first wall temperature (K).
    temp_fw_max :
        Maximum temperature of first wall material (K) (bound;
        `i_thermal_electric_conversion > 1` only, per the source docstring).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return leq(temp_fw_peak, temp_fw_max)


def constraint_40(
    p_hcd_injected_total_mw,
    p_hcd_injected_min_mw,
):
    """Auxiliary power lower limit. Ports `constraint_equation_40`.

    Parameters
    ----------
    p_hcd_injected_total_mw :
        Total auxiliary injected power (MW).
    p_hcd_injected_min_mw :
        Minimum auxiliary power (MW) (bound).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return geq(p_hcd_injected_total_mw, p_hcd_injected_min_mw)


def constraint_41(
    t_plant_pulse_plasma_current_ramp_up,
    t_current_ramp_up_min,
):
    """Plasma current ramp-up time lower limit. Ports `constraint_equation_41`.

    Parameters
    ----------
    t_plant_pulse_plasma_current_ramp_up :
        Plasma current ramp-up time for current initiation (s).
    t_current_ramp_up_min :
        Minimum plasma current ramp-up time (s) (bound).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return geq(t_plant_pulse_plasma_current_ramp_up, t_current_ramp_up_min)


def constraint_42(
    t_plant_pulse_total,
    t_cycle_min,
):
    """Cycle time lower limit. Ports `constraint_equation_42`.

    PROCESS raises `ProcessValueError` if `t_cycle_min < 1.0` (a proxy for
    `i_pulsed_plant == 0`) -- not reproduced, see this module's own docstring.

    Parameters
    ----------
    t_plant_pulse_total :
        Full cycle time (s).
    t_cycle_min :
        Minimum cycle time (s) (bound).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return geq(t_plant_pulse_total, t_cycle_min)


def constraint_43(
    i_tf_sup,
    temp_cp_average,
    tcpav2,
):
    """Average centrepost temperature consistency equation (TART). Ports
    `constraint_equation_43`.

    PROCESS raises `ProcessValueError` if `itart == 0` (this constraint is meaningless
    outside spherical-tokamak/TART configurations) -- not reproduced, see this module's
    own docstring; `itart` itself is therefore not a parameter here (it gates nothing
    inside the function body, only the raise).

    `i_tf_sup == WATER_COOLED_COPPER` subtracts room temperature from both operands
    before comparing -- kept, this is real formula-selecting logic, not a misuse guard.

    Parameters
    ----------
    i_tf_sup :
        `TFConductorModel` switch. Static -- selects whether both operands are
        room-temperature-referenced before comparison.
    temp_cp_average :
        Average temperature of TF coil inboard leg conductor (C, or C above room
        temperature when `i_tf_sup == WATER_COOLED_COPPER`).
    tcpav2 :
        Centrepost average temperature (C, same referencing) -- the consistency target.

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    if i_tf_sup == TFConductorModel.WATER_COOLED_COPPER:
        temp_cp_average -= constants.TEMP_ROOM
        tcpav2 -= constants.TEMP_ROOM

    return eq(temp_cp_average, tcpav2)


def constraint_44(
    i_tf_sup,
    temp_cp_max,
    temp_cp_peak,
):
    """Centrepost temperature upper limit (TART). Ports `constraint_equation_44`.

    PROCESS raises `ProcessValueError` if `itart == 0` -- not reproduced, same reasoning
    as `constraint_43`; `itart` is not a parameter here for the same reason.

    Parameters
    ----------
    i_tf_sup :
        `TFConductorModel` switch. Static -- selects whether both operands are
        room-temperature-referenced before comparison.
    temp_cp_max :
        Maximum peak centrepost temperature (bound).
    temp_cp_peak :
        Peak centrepost temperature.

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    if i_tf_sup == TFConductorModel.WATER_COOLED_COPPER:
        temp_cp_max -= constants.TEMP_ROOM
        temp_cp_peak -= constants.TEMP_ROOM

    return leq(temp_cp_peak, temp_cp_max)


def constraint_45(itart, q95, q95_min):
    """Edge safety factor lower limit (TART). Ports `constraint_manager_45`.

    TART-only: PROCESS raises if this constraint is invoked with `itart == 0`
    (conventional aspect ratio model, no spherical-tokamak `q95` definition to bound).
    `itart` is a static switch (`naming_convention.md` § "switches are not ports"), so
    the raise fires at trace-assembly time, never inside traced code -- same legitimacy
    as `constraint_24`'s own `ValueError` on an invalid `i_beta_component`. Plain
    `ValueError`, not `process.core.exceptions.ProcessValueError`, matching this port's
    established convention of not importing PROCESS's own exception hierarchy for
    switch-validation raises (see `constraint_24`).

    Parameters
    ----------
    itart :
        Spherical tokamak (ST) switch (0: conventional aspect ratio, 1: ST). Static.
    q95 :
        Safety factor 'near' plasma edge (or mean edge safety factor `qbar` when
        `i_plasma_current == 2`, per the source docstring -- that selection happens
        upstream of this constraint, not here).
    q95_min :
        Lower limit for edge safety factor (bound).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.

    Raises
    ------
    ValueError
        If `itart == 0`. Legitimate: `itart` is a static argument, nothing here is
        traced.
    """
    if itart == 0:
        raise ValueError("constraint_45: itart=0 -- constraint 45 requires itart=1")

    return geq(q95, q95_min)


def constraint_46(itart, eps, plasma_current, c_tf_total):
    """I_p / I_rod upper limit (TART). Ports `constraint_equation_46`.

    TART-only, same raise-on-`itart == 0` shape as `constraint_45` above -- see that
    function's docstring for the reasoning (static switch, legitimate trace-time raise).

    Parameters
    ----------
    itart :
        Spherical tokamak (ST) switch. Static.
    eps :
        Inverse aspect ratio.
    plasma_current :
        Plasma current (A).
    c_tf_total :
        Total (summed) current in TF coils (A).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.

    Raises
    ------
    ValueError
        If `itart == 0`.
    """
    if itart == 0:
        raise ValueError("constraint_46: itart=0 -- constraint 46 requires itart=1")

    cratmx = 1.0 + 4.91 * (eps - 0.62)
    return leq(plasma_current / c_tf_total, cratmx)


def constraint_48(beta_poloidal_vol_avg, beta_poloidal_max):
    """Poloidal beta upper limit. Ports `constraint_equation_48`.

    Bare residual read, no switch. See `batch5.md` for the hole-in-MDA note:
    `beta_poloidal_vol_avg`'s real producer (`Physics.calculate_poloidal_beta`,
    `physics.py:3825`) is not yet ported anywhere in `functional_process` -- ported here
    regardless, same convention `constraint_91` already established for its own
    then-unported `powerht_constraint`/`powerscaling_constraint` operands.

    Parameters
    ----------
    beta_poloidal_vol_avg :
        Volume-averaged poloidal beta.
    beta_poloidal_max :
        Maximum allowed poloidal beta (bound).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return leq(beta_poloidal_vol_avg, beta_poloidal_max)


def constraint_51(vs_plasma_ramp_required, vs_cs_pf_total_ramp):
    """Startup flux = available startup flux (equality). Ports
    `constraint_equation_51`.

    First equality constraint ported in this file -- see this module's own docstring
    for why `eq` is defined here rather than imported.

    Parameters
    ----------
    vs_plasma_ramp_required :
        Required flux swing for startup (Wb). PROCESS takes `abs()` of this operand
        before comparing (source: `abs(data.physics.vs_plasma_ramp_required)`),
        reproduced here via plain Python `abs()` (works elementwise/tracer-safe via
        `__abs__`, matching the source's own use of the builtin).
    vs_cs_pf_total_ramp :
        Total flux swing for startup from the CS/PF coil set (Wb).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return eq(abs(vs_plasma_ramp_required), vs_cs_pf_total_ramp)


def constraint_53(flu_tf_neutron_fast_peak, flu_tf_neutron_fast_max):
    """Fast neutron fluence on TF coil, upper limit. Ports `constraint_equation_53`.

    Bare residual read, no switch, **no hole-in-MDA**: `flu_tf_neutron_fast_peak`'s
    real producer is already ported (`stellarator_F_tf_nuclear_heating.py`'s
    `.fwbs.flu_tf_neutron_fast_peak` `Output`); `flu_tf_neutron_fast_max` is a plain
    input constant (`_audit` convention: bounds with no producer are ordinary inputs,
    not holes).

    Parameters
    ----------
    flu_tf_neutron_fast_peak :
        Peak fast neutron fluence on TF coil superconductor (n/m²).
    flu_tf_neutron_fast_max :
        Maximum fast neutron fluence on TF coil (n/m², bound).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return leq(flu_tf_neutron_fast_peak, flu_tf_neutron_fast_max)


def constraint_54(ptfnucpm3, ptfnucmax):
    """Peak TF coil nuclear heating, upper limit. Ports `constraint_equation_54`.

    Bare residual read, no switch. **Hole-in-MDA**: `ptfnucpm3`'s real producer is
    inline arithmetic inside `Stellarator.st_fwbs` (`stellarator.py:455`,
    `p_tf_nuclear_heat_mw / tf_volume`) that has not been extracted into any ported
    node -- distinct from (and not to be confused with) constraint 53's
    `flu_tf_neutron_fast_peak`, which *is* already ported despite living in a
    similarly-named nuclear-heating source region. `p_tf_nuclear_heat_mw` itself comes
    from `hcpb.py`'s `nuclear_heating_magnets`, whose cottax node
    (`NuclearHeatingMagnets`) is built but unregistered (see `_audit/next_steps.md`'s
    consolidation-gap audit). Ported here regardless, same as `constraint_48` above.

    Parameters
    ----------
    ptfnucpm3 :
        Nuclear heating in the TF coil (MW/m³, `blktmodel > 0`).
    ptfnucmax :
        Maximum nuclear heating in TF coil (MW/m³, bound).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return leq(ptfnucpm3, ptfnucmax)


def constraint_56(p_plasma_separatrix_rmajor_mw, p_plasma_separatrix_rmajor_max_mw):
    """P_sep / R0 upper limit. Ports `constraint_equation_56`.

    Parameters
    ----------
    p_plasma_separatrix_rmajor_mw :
        Ratio of power crossing the separatrix to plasma major radius (MW/m).
    p_plasma_separatrix_rmajor_max_mw :
        Maximum allowed value of the same ratio (bound, MW/m).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return leq(p_plasma_separatrix_rmajor_mw, p_plasma_separatrix_rmajor_max_mw)


def constraint_59(f_p_beam_shine_through, f_p_beam_shine_through_max):
    """Neutral beam shine-through fraction upper limit. Ports
    `constraint_equation_59`.

    Parameters
    ----------
    f_p_beam_shine_through :
        Neutral beam shine-through fraction.
    f_p_beam_shine_through_max :
        Maximum allowed shine-through fraction (bound).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return leq(f_p_beam_shine_through, f_p_beam_shine_through_max)


def constraint_60(temp_cs_superconductor_margin, temp_cs_superconductor_margin_min):
    """Central Solenoid s/c temperature margin lower limit. Ports
    `constraint_equation_60`.

    Parameters
    ----------
    temp_cs_superconductor_margin :
        Central solenoid temperature margin (K).
    temp_cs_superconductor_margin_min :
        Minimum allowable temperature margin (bound, K). Note: this is a genuinely
        different `VarPath` from the *value* operand's own producer default -- PROCESS
        itself sometimes initialises `temp_cs_superconductor_margin_min` from
        `.tfcoil.tmargmin` (`core/init.py:1190`), but that is an initial-value
        assignment, not a computation this constraint's body performs or depends on;
        the constraint just reads whatever the field currently holds.

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return geq(temp_cs_superconductor_margin, temp_cs_superconductor_margin_min)


def constraint_61(f_t_plant_available, f_t_plant_available_min):
    """Plant availability lower limit. Ports `constraint_equation_61`.

    Parameters
    ----------
    f_t_plant_available :
        Total plant availability fraction.
    f_t_plant_available_min :
        Minimum allowed availability (bound).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return geq(f_t_plant_available, f_t_plant_available_min)


def constraint_62(f_t_alpha_energy_confinement, f_t_alpha_energy_confinement_min):
    """Lower limit on the ratio of alpha-particle to energy confinement times. Ports
    `constraint_equation_62`.

    `f_t_alpha_energy_confinement` is also iteration variable 110
    (`data_structure/numerics.py:361`) -- a strong candidate for the same
    iteration-variable-pairing note `constraints.md`'s constraint-91 entry already
    makes for `te0_ecrh_achievable`/ID 169. Not resolved here (out of this port's
    scope), just flagged for whoever does the iteration-variable pass.

    Parameters
    ----------
    f_t_alpha_energy_confinement :
        Ratio of alpha-particle confinement time to energy confinement time.
    f_t_alpha_energy_confinement_min :
        Minimum allowed ratio (bound).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return geq(f_t_alpha_energy_confinement, f_t_alpha_energy_confinement_min)


def constraint_63(n_iter_vacuum_pumps, n_tf_coils):
    """Upper limit on the number of high-vacuum pumps (`i_vacuum_pumping = simple`).
    Ports `constraint_equation_63`.

    General constraint (not stellarator-specific): no `.stellarator.*` read, no
    switch. Considered and explicitly *excluded* from the earlier stellarator-only
    audit pass for exactly that reason (see `constraints.md`) -- that pass's
    "stellarator-specific logic only" scope does not apply here; this pass ports every
    constraint by default.

    Parameters
    ----------
    n_iter_vacuum_pumps :
        Number of high-vacuum pumps (real-valued count).
    n_tf_coils :
        Number of TF coils (bound -- one pump per coil, at most; default 50 for
        stellarators per the source docstring, but this function takes whatever value
        the field holds).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return leq(n_iter_vacuum_pumps, n_tf_coils)


def constraint_64(
    n_charge_plasma_effective_vol_avg, n_charge_plasma_effective_vol_avg_max
):
    """Upper limit on volume-averaged plasma effective charge (Zeff). Ports
    `constraint_equation_64`.

    Parameters
    ----------
    n_charge_plasma_effective_vol_avg :
        Volume-averaged plasma effective charge.
    n_charge_plasma_effective_vol_avg_max :
        Maximum allowed value (bound).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return leq(n_charge_plasma_effective_vol_avg, n_charge_plasma_effective_vol_avg_max)


def constraint_65(vv_stress_quench, max_vv_stress):
    """Upper limit on vacuum vessel stress during a TF coil quench. Ports
    `constraint_equation_65`.

    Parameters
    ----------
    vv_stress_quench :
        Vacuum vessel stress on TF coil quench (Pa).
    max_vv_stress :
        Maximum permitted VV stress (bound, Pa).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return leq(vv_stress_quench, max_vv_stress)


def constraint_66(peakpoloidalpower, maxpoloidalpower):
    """Upper limit on rate of change of energy in the poloidal field.
    Ports `constrain_equation_66`.

    Parameters
    ----------
    peakpoloidalpower :
        Peak absolute rate of change of stored energy in poloidal field (MW).
    maxpoloidalpower :
        Maximum permitted absolute rate of change of stored energy in poloidal field
        (MW) (bound).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return leq(peakpoloidalpower, maxpoloidalpower)


def constraint_67(pflux_fw_rad_max_mw, pflux_fw_rad_max):
    """Simple upper limit on radiation wall load. Ports `constraint_equation_67`.

    Both operands live under `.constraints.*` in PROCESS's `DataStructure` -- unusual
    (most constraints compare a `.physics.*`/model output against a `.constraints.*`
    input constant), but confirmed directly from the source: `pflux_fw_rad_max_mw` (the
    peak wall load actually reached, despite the confusing "_max" in its name -- see the
    source docstring, which even calls it "Peak radiation wall load") is itself
    calculated and stored under `.constraints.*` elsewhere in PROCESS, not `.physics.*`.

    Parameters
    ----------
    pflux_fw_rad_max_mw :
        Peak radiation wall load reached (MW/m2), despite the "_max" in its name.
    pflux_fw_rad_max :
        Maximum permitted radiation wall load (MW/m2) (bound).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return leq(pflux_fw_rad_max_mw, pflux_fw_rad_max)


def constraint_68(
    i_q95_fixed,
    p_plasma_separatrix_mw,
    b_plasma_toroidal_on_axis,
    q95,
    q95_fixed,
    aspect,
    rmajor,
    p_div_bt_q_aspect_rmajor_mw,
    p_div_bt_q_aspect_rmajor_max_mw,
):
    """Upper limit on Psep scaling (PsepBt / q95*A*R0). Ports `constraint_equation_68`.

    `i_q95_fixed == 1` re-derives the metric from a *fixed* `q95_fixed` value via
    `PlasmaExhaust.calculate_eu_demo_re_attachment_metric` (`process/models/physics/
    exhaust.py:149-183`) -- a trivial closed-form arithmetic staticmethod
    (`(p_plasma_separatrix_mw * b_plasma_toroidal_on_axis) / (q95 * aspect * rmajor)`,
    no `data` access, no branching), inlined here rather than imported since it is not
    yet ported as its own node in `functional_process/models/physics/exhaust.py`. This
    makes the `i_q95_fixed == 1` branch `Compare`-shaped in principle (a `calculate_*`
    re-derivation compared to a stored bound) once/if that function is ever registered
    as its own node elsewhere -- flagged for whoever does that, not resolved here.
    `i_q95_fixed == 0` (PROCESS's default) is a bare residual read of the
    already-computed `.physics.p_div_bt_q_aspect_rmajor_mw`.

    Parameters
    ----------
    i_q95_fixed :
        Switch: 1 fixes `q95` at `q95_fixed` for this constraint only; 0 (default) uses
        the already-computed `.physics.p_div_bt_q_aspect_rmajor_mw` directly. Static.
    p_plasma_separatrix_mw, b_plasma_toroidal_on_axis, q95, aspect, rmajor :
        Only used when `i_q95_fixed == 1` (re-derivation inputs).
    q95_fixed :
        Fixed safety factor, used in place of `q95` when `i_q95_fixed == 1`.
    p_div_bt_q_aspect_rmajor_mw :
        Already-computed metric, used directly when `i_q95_fixed == 0`.
    p_div_bt_q_aspect_rmajor_max_mw :
        Maximum permitted value of the metric (bound), either branch.

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    if i_q95_fixed == 1:
        value = (p_plasma_separatrix_mw * b_plasma_toroidal_on_axis) / (
            q95_fixed * aspect * rmajor
        )
    else:
        value = p_div_bt_q_aspect_rmajor_mw

    return leq(value, p_div_bt_q_aspect_rmajor_max_mw)


def constraint_72(
    i_tf_bucking,
    i_tf_inside_cs,
    stress_shear_cs_peak,
    sig_tf_cs_bucked,
    stress_cs_steel_max,
):
    """Upper limit on Central Solenoid Tresca yield stress.
    Ports `constraint_equation_72`.

    Bucked-and-wedged design (`i_tf_bucking >= 2` and `i_tf_inside_cs ==
    TFCSRadialConfiguration.TF_OUTSIDE_CS`) takes the larger of two stress scenarios
    (max current; flux swing with TF inward pressure); free-standing CS uses only the
    max-current stress. Both branches are bare residual reads -- no `calculate_*`
    re-derivation, `data` access, or branching inside either operand.

    Parameters
    ----------
    i_tf_bucking :
        TF structure design switch. Static.
    i_tf_inside_cs :
        `TFCSRadialConfiguration` switch (build radial configuration). Static.
    stress_shear_cs_peak :
        Maximum shear stress in coils/central solenoid at max current (Pa).
    sig_tf_cs_bucked :
        Maximum shear stress in CS case at flux swing, no current in CS (Pa). Only used
        in the bucked-and-wedged branch.
    stress_cs_steel_max :
        Allowable stress in Central Solenoid structural material (Pa) (bound).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    if i_tf_bucking >= 2 and i_tf_inside_cs == TFCSRadialConfiguration.TF_OUTSIDE_CS:
        value = max(stress_shear_cs_peak, sig_tf_cs_bucked)
    else:
        value = stress_shear_cs_peak

    return leq(value, stress_cs_steel_max)


def constraint_73(p_plasma_separatrix_mw, p_l_h_threshold_mw, p_hcd_injected_total_mw):
    """Lower limit: separatrix power >= L-H threshold power + auxiliary power.
    Ports `constraint_equation_73`. Related to constraint 15 (not audited here).

    Parameters
    ----------
    p_plasma_separatrix_mw :
        Power conducted to the divertor region (MW).
    p_l_h_threshold_mw :
        L-H mode power threshold (MW).
    p_hcd_injected_total_mw :
        Total auxiliary injected power (MW).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return geq(p_plasma_separatrix_mw, p_l_h_threshold_mw + p_hcd_injected_total_mw)


def constraint_74(temp_croco_quench, temp_croco_quench_max):
    """Upper limit on TF coil quench temperature. Ports `constraint_equation_74`.

    Only meaningful for CroCo HTS coils per the source docstring ("ONLY used for croco
    HTS coil") -- not itself a switch-gated branch in the constraint body, just a
    documented usage precondition; ported as an unconditional bare residual, same as the
    source.

    Parameters
    ----------
    temp_croco_quench :
        Actual temperature reached during a quench (K).
    temp_croco_quench_max :
        Maximum permitted temperature during a quench (K) (bound).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return leq(temp_croco_quench, temp_croco_quench_max)


def constraint_75(coppera_m2, tf_coppera_m2_max):
    """Upper limit on TF coil current / copper area. Ports `constraint_equation_75`.

    Only meaningful for CroCo HTS coils per the source docstring, same usage-precondition
    shape as constraint 74 -- ported unconditionally.

    Parameters
    ----------
    coppera_m2 :
        TF coil current / copper area (A/m2).
    tf_coppera_m2_max :
        Maximum permitted TF coil current / copper area (A/m2) (bound).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return leq(coppera_m2, tf_coppera_m2_max)


def constraint_76(
    kappa,
    triang,
    aspect,
    p_plasma_separatrix_mw,
    nd_plasma_electron_max_array_7,
    nd_plasma_separatrix_electron,
):
    """Upper limit for the Eich critical separatrix density model.
    Ports `constraint_equation_76`.

    **Real PROCESS quirk, reproduced as-is (not fixed)**: the source function writes
    two intermediates (`data.physics.alpha_crit`, `data.physics.
    nd_plasma_separatrix_electron_eich_max`) directly onto `DataStructure` from *inside*
    the constraint body, and the source itself carries a `# TODO: why on earth are these
    variables being set here!? Should they be local?` comment questioning exactly this.
    Ported here as ordinary local intermediates (not "written" anywhere), matching this
    codebase's standard `local-intermediate` classification for exactly this shape (a
    value computed and used once, never itself part of the constraint's port
    signature) -- the two `data` writes are a PROCESS-internal side effect this port
    does not reproduce, same policy `physics_B_composition.py`'s docstring states for
    unported side effects elsewhere in this codebase.

    `nd_plasma_electron_max_array(7)` in the source docstring is 1-indexed Fortran-style
    notation for `nd_plasma_electron_max_array[6]` (0-indexed) -- confirmed directly
    against the source body's own `[6]` index, not assumed from the docstring alone;
    passed here as a single scalar (`nd_plasma_electron_max_array_7`) rather than the
    whole 7+-element array, since this constraint only ever reads that one element.

    Parameters
    ----------
    kappa :
        Plasma separatrix elongation.
    triang :
        Plasma separatrix triangularity.
    aspect :
        Aspect ratio.
    p_plasma_separatrix_mw :
        Power conducted to the divertor region (MW).
    nd_plasma_electron_max_array_7 :
        Element index 6 (Fortran `(7)`) of the density-limit array (/m3) -- the "Eich"
        entry among PROCESS's several parallel density-limit models.
    nd_plasma_separatrix_electron :
        Actual electron density at the separatrix (/m3).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    alpha_crit = (kappa**1.2) * (1.0 + 1.5 * triang)
    nd_plasma_separatrix_electron_eich_max = (
        5.9
        * alpha_crit
        * (aspect ** (-2.0 / 7.0))
        * (((1.0 + (kappa**2.0)) / 2.0) ** (-6.0 / 7.0))
        * ((p_plasma_separatrix_mw * 1.0e6) ** (-11.0 / 70.0))
        * nd_plasma_electron_max_array_7
    )

    return leq(nd_plasma_separatrix_electron, nd_plasma_separatrix_electron_eich_max)


def constraint_77(c_tf_turn, c_tf_turn_max):
    """Maximum TF coil current per turn upper limit. Ports `constraint_equation_77`.

    Parameters
    ----------
    c_tf_turn :
        TF coil current per turn (A/turn).
    c_tf_turn_max :
        Allowable TF coil current per turn (A/turn, bound). Plain input constant
        (`tfcoil_variables.py:149`, default `9.0e4`) -- no producer, never computed.

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return leq(c_tf_turn, c_tf_turn_max)


def constraint_78(fzactual, fzmin):
    """Reinke criterion, divertor impurity fraction lower limit. Ports
    `constraint_equation_78`.

    **Hole-in-MDA**: both operands are `.reinke.*` fields with no producer anywhere in
    this port -- the Reinke divertor-impurity model (`process/models/reinke.py` or
    equivalent) is entirely unported. See `batch8.md` for the full note. Ported here as
    a bare arithmetic function regardless, since the comparison itself needs no
    unported code to compute once its two operands exist.

    Parameters
    ----------
    fzactual :
        Actual impurity fraction.
    fzmin :
        Minimum impurity fraction required by the Reinke model (bound).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return geq(fzactual, fzmin)


def constraint_79(b_cs_peak_flat_top_end, b_cs_peak_pulse_start, b_cs_limit_max):
    """Maximum central solenoid (CS) field. Ports `constraint_equation_79`.

    Source docstring's own unit tag ("A/turn") is a copy-paste leftover from the
    neighbouring constraints (77/79 share a source-file typo) -- the actual registered
    unit is field, not current (`ConstraintManager.register_constraint(79, "A/turn",
    "<=")` in the source is itself the same mislabel; not corrected here, reproduced
    faithfully as a real quirk of the PROCESS source, not fixed).

    **Hole-in-MDA**: all three operands are `.pf_coil.*` fields with no producer
    anywhere in this port -- the PF coil / central solenoid system is entirely
    unported. See `batch8.md`. Ported here as a bare arithmetic function regardless.

    Parameters
    ----------
    b_cs_peak_flat_top_end :
        Peak CS field at end of flat-top (T).
    b_cs_peak_pulse_start :
        Peak CS field at beginning of pulse (T).
    b_cs_limit_max :
        Central solenoid max field limit (T, bound).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    peak = max(b_cs_peak_flat_top_end, b_cs_peak_pulse_start)
    return leq(peak, b_cs_limit_max)


def constraint_80(p_plasma_separatrix_mw, p_plasma_separatrix_min_mw):
    """Lower limit on power crossing the separatrix. Ports `constraint_equation_80`.

    Parameters
    ----------
    p_plasma_separatrix_mw :
        Power crossing the separatrix (MW).
    p_plasma_separatrix_min_mw :
        Minimum power crossing the separatrix (MW, bound). Plain input constant
        (`constraint_variables.py:73`, default `150.0`).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return geq(p_plasma_separatrix_mw, p_plasma_separatrix_min_mw)


def constraint_81(nd_plasma_electron_on_axis, nd_plasma_pedestal_electron):
    """Lower limit ensuring central density exceeds the pedestal density. Ports
    `constraint_equation_81`.

    Parameters
    ----------
    nd_plasma_electron_on_axis :
        Central electron density (/m3).
    nd_plasma_pedestal_electron :
        Electron density at the pedestal (/m3, bound).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return geq(nd_plasma_electron_on_axis, nd_plasma_pedestal_electron)


def constraint_82(toroidalgap, dx_tf_inboard_out_toroidal):
    """Toroidal consistency of the stellarator build. Ports `constraint_equation_82`.

    No switches, no internal branching -- a bare `geq` of two already-produced
    fields. `toroidalgap` (minimal gap between two stellarator coils) and
    `dx_tf_inboard_out_toroidal` (total toroidal width of a TF coil) are both minted by
    `functional_process/models/stellarator/coils/calculate.py`'s already-ported
    `CoilCoilToroidalGap`/`CoilToroidalThickness` nodes.

    Parameters
    ----------
    toroidalgap :
        Minimal gap between two stellarator coils (m).
    dx_tf_inboard_out_toroidal :
        Total toroidal width of a TF coil (m).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return geq(toroidalgap, dx_tf_inboard_out_toroidal)


def constraint_83(available_radial_space, required_radial_space):
    """Radial consistency of the stellarator build. Ports `constraint_equation_83`.

    No switches, no internal branching -- a bare `geq` of two already-produced fields,
    both minted by `functional_process/models/stellarator/build.py`'s already-ported
    `Build` node.

    Parameters
    ----------
    available_radial_space :
        Available space in the radial direction, as given by the stellarator
        configuration (m).
    required_radial_space :
        Required space in the radial direction (m).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return geq(available_radial_space, required_radial_space)


def constraint_84(beta_total_vol_avg, beta_vol_avg_min):
    """Lower limit of plasma beta. Ports `constraint_equation_84`.

    Parameters
    ----------
    beta_total_vol_avg :
        Total plasma beta.
    beta_vol_avg_min :
        Lower limit for beta (bound). Plain input constant
        (`physics_variables.py:530`, default `0.0`).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return geq(beta_total_vol_avg, beta_vol_avg_min)


def constraint_85(
    i_cp_lifetime, cplife, cplife_input, life_div_fpy, life_blkt_fpy, life_plant
):
    """Equality constraint for the centrepost (CP) lifetime. Ports
    `constraint_equation_85`.

    `i_cp_lifetime` selects which of four already-produced/input fields `cplife` must
    equal -- a formula-changing switch, kept as a static Python `int` and branched with
    plain `if`/`elif` per this codebase's established convention (switches select a
    formula, not a differentiable input).

    Parameters
    ----------
    i_cp_lifetime :
        Switch selecting which plant element's lifetime the CP lifetime must equal
        (0: user input, 1: divertor, 2: breeding blanket, 3: whole plant). Static.
    cplife :
        Calculated CP full-power-year lifetime (years).
    cplife_input :
        User-specified CP lifetime (years), used when `i_cp_lifetime == 0`. Plain
        input constant (`cost_variables.py:345`, default `2.0`).
    life_div_fpy :
        Divertor full-power-year lifetime (years), used when `i_cp_lifetime == 1`.
    life_blkt_fpy :
        Breeding-blanket full-power-year lifetime (years), used when
        `i_cp_lifetime == 2`.
    life_plant :
        Whole-plant lifetime (years), used when `i_cp_lifetime == 3`. Plain input
        constant (`cost_variables.py:557`, default `30.0`).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.

    Raises
    ------
    ValueError
        If `i_cp_lifetime` is not one of `{0, 1, 2, 3}`. Static, so nothing here is
        traced.
    """
    if i_cp_lifetime == 0:
        bound = cplife_input
    elif i_cp_lifetime == 1:
        bound = life_div_fpy
    elif i_cp_lifetime == 2:
        bound = life_blkt_fpy
    elif i_cp_lifetime == 3:
        bound = life_plant
    else:
        raise ValueError(
            f"constraint_85: i_cp_lifetime={i_cp_lifetime!r} is not in {{0, 1, 2, 3}}"
        )

    return eq(cplife, bound)


def constraint_86(dx_tf_turn_general, t_turn_tf_max):
    """Upper limit on TF winding-pack turn edge length. Ports
    `constraint_equation_86`.

    Parameters
    ----------
    dx_tf_turn_general :
        TF coil turn edge length including turn insulation (m). Already an established
        `.tfcoil.*` field this codebase reads elsewhere (`coils/calculate.py`'s
        `WindingPackIntersectInputs`/`WindingPackTotalSizePost`), not re-derived here.
    t_turn_tf_max :
        TF turn edge length upper limit (m, bound). Plain input constant
        (`tfcoil_variables.py:113`, default `0.05`).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return leq(dx_tf_turn_general, t_turn_tf_max)


def constraint_87(p_cryo_plant_electric_mw, p_cryo_plant_electric_max_mw):
    """TF coil cryogenic power upper limit. Ports `constraint_equation_87`.

    General constraint, no switch. Both operands are plain already-produced fields --
    see the audit record for `p_cryo_plant_electric_mw`'s real producer
    (`power_B_thermal_cryo.py`'s `CryoLoads`, already ported, not yet registered in
    `total_process.py`).

    Parameters
    ----------
    p_cryo_plant_electric_mw :
        Cryogenic plant electric power (MW).
    p_cryo_plant_electric_max_mw :
        Maximum cryogenic plant electric power (MW) (bound).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return leq(p_cryo_plant_electric_mw, p_cryo_plant_electric_max_mw)


def constraint_88(str_wp, str_wp_max):
    """TF coil vertical strain upper limit (absolute value). Ports
    `constraint_equation_88`.

    General constraint, no switch. `abs(str_wp)` is taken because the source compares
    an unsigned strain magnitude to a positive bound; `str_wp` itself may be negative
    (compressive). Both operands' real producer (`process/models/tfcoil/
    superconducting.py`'s self-consistent winding-pack strain calculation) is not yet
    ported anywhere in this codebase -- see the audit record's hole-in-MDA note; this
    does not block porting the constraint itself, same precedent as constraint 91's
    `.stellarator.powerht_constraint` (from an unported unit).

    Parameters
    ----------
    str_wp :
        TF coil winding-pack vertical strain (dimensionless), signed.
    str_wp_max :
        Allowable maximum TF coil vertical strain (bound, positive).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return leq(abs(str_wp), str_wp_max)


def constraint_89(copperaoh_m2, copperaoh_m2_max):
    """Central Solenoid (OH) coil current / copper area upper limit. Ports
    `constraint_equation_89`.

    General constraint, no switch. Both operands' real producer (`process/models/
    pfcoil.py`'s REBCO CS current-density calculation) is not yet ported -- same
    "constraint portable, producer not yet in this codebase" situation as 88, see the
    audit record.

    Parameters
    ----------
    copperaoh_m2 :
        CS coil current at end-of-flattop / copper area (A/m²).
    copperaoh_m2_max :
        Maximum allowed coil current / copper area (A/m²) (bound).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return leq(copperaoh_m2, copperaoh_m2_max)


def constraint_90(n_cycle, n_cycle_min, ibkt_life, bkt_life_csf, bktcycles):
    """Lower limit for CS coil stress load cycles. Ports `constraint_equation_90`.

    **Real PROCESS finding**: the source function has a side effect --
    `if data.costs.ibkt_life == 1 and data.cs_fatigue.bkt_life_csf == 1:
    data.cs_fatigue.n_cycle_min = data.costs.bktcycles` -- it *writes* `n_cycle_min`
    into `data` before reading it back for the comparison, so evaluating this
    constraint can silently mutate shared state read by anything else that later reads
    `.cs_fatigue.n_cycle_min`. A pure port has no `data` object to mutate, so this
    function instead applies the override directly to the *value used in its own
    comparison*: when both switches are true, the effective `n_cycle_min` is
    `bktcycles`, not the passed-in `n_cycle_min` argument -- numerically identical to
    what PROCESS's own constraint evaluation call sees, but does not (and structurally
    cannot) reproduce the global-state write itself. See the audit record for the
    broader implication (other readers of `.cs_fatigue.n_cycle_min` after this
    constraint runs would see PROCESS's mutated value, a real ordering dependency this
    port does not model).

    `ibkt_life` (`cost_variables.py:416`, default `0`) and `bkt_life_csf`
    (`cs_fatigue_variables.py:31`, default `0.0`, "Switch to pass bkt_life cycles to
    n_cycle_min") are both switches -- static, per `naming_convention.md` -- compared
    with `== 1`, so kept as plain Python values, not traced.

    Parameters
    ----------
    n_cycle :
        Allowable number of cycles for CS.
    n_cycle_min :
        Minimum required cycles for CS, before any override.
    ibkt_life :
        Blanket-life-model switch (static). `1` together with `bkt_life_csf == 1`
        triggers the override below.
    bkt_life_csf :
        "Pass bkt_life cycles to n_cycle_min" switch (static).
    bktcycles :
        Blanket cycle count -- replaces `n_cycle_min` in the comparison when both
        switches above are `1`.

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    if ibkt_life == 1 and bkt_life_csf == 1:
        n_cycle_min = bktcycles

    return geq(n_cycle, n_cycle_min)


def constraint_91(
    i_plasma_ignited,
    p_hcd_primary_extra_heat_mw,
    powerht_constraint,
    powerscaling_constraint,
):
    """ECRH ignition heating-power lower limit. Ports `constraint_equation_91`.

    "Stellarators only (but in principle usable also for tokamaks)" per the original
    docstring. `powerht_constraint`/`powerscaling_constraint` are taken as plain float
    arguments, per the audit record: their own producer, `power_at_ignition_point`, is
    a separate (tier-2, unported) unit, not this constraint's concern.

    Parameters
    ----------
    i_plasma_ignited :
        `PlasmaIgnitionModel` switch. Static — selects which of two `value`
        expressions applies.
    p_hcd_primary_extra_heat_mw :
        Extra heating power for the non-ignited case (MW). Only used when
        `i_plasma_ignited == PlasmaIgnitionModel.NON_IGNITED`.
    powerht_constraint :
        Achievable ECRH heating power at the ignition point (MW).
    powerscaling_constraint :
        Required scaling/loss power at the ignition point (MW) (bound).

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    if PlasmaIgnitionModel(i_plasma_ignited) == PlasmaIgnitionModel.NON_IGNITED:
        value = powerht_constraint + p_hcd_primary_extra_heat_mw
    else:
        value = powerht_constraint

    return geq(value, powerscaling_constraint)


def constraint_92(f_plasma_fuel_deuterium, f_plasma_fuel_tritium, f_plasma_fuel_helium3):
    """D/T/He3 fuel fraction consistency (must sum to 1). Ports
    `constraint_equation_92`.

    Equality constraint -- the first one audited in this codebase, hence `eq` above.
    All three operands are plain user-input fractions (`physics_variables.py`), not
    computed outputs of any model -- `plasma_composition`
    (`functional_process/models/physics/physics_B_composition.py`) already reads all
    three as ordinary boundary `Input`s, confirming they are leaf inputs, not a
    hole-in-MDA candidate at all (nothing produces them; nothing needs to).

    Parameters
    ----------
    f_plasma_fuel_deuterium, f_plasma_fuel_tritium, f_plasma_fuel_helium3 :
        Fuel mix fractions (dimensionless), user input.

    Returns
    -------
    :
        `(residual, normalised_residual, constraint_value, constraint_bound)`.
    """
    return eq(
        f_plasma_fuel_deuterium + f_plasma_fuel_tritium + f_plasma_fuel_helium3,
        1.0,
    )
