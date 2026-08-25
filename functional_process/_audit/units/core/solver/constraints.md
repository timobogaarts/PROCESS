# Constraint audit records

One section per constraint, schema in `../../_audit/schema.md` § Constraint record.

Originally scoped to the five stellarator-specific constraints only (17, 24, 82, 83,
91) -- broadened to cover PROCESS's full general-constraint set in a later pass (see
"Port status" below), ported without a stellarator-relevance filter this time. Most of
what follows is ordinary tokamak/general physics-and-engineering bookkeeping, not
stellarator-specific at all; the five originally-scoped ones remain flagged as such in
their own entries below.

---
kind: constraint
status: reviewed
confidence: medium-high (structural classification), medium (physics-correctness judgment)
---

### Constraint 1: relationship between beta, temperature and density

**source**: `process/core/solver/constraints.py:217-251`.

**calls**: `PlasmaBeta.calculate_plasma_beta(pres_plasma, b_field)` -- a genuine
re-derivation inside the constraint body, ported alongside it in `batch0_constraints.py`
(already pure in the source, no `self.data` access, two-line formula). **This is the
first `Compare`-shaped constraint to land in this codebase** -- `CLAUDE.md`'s own mapping
table cites this exact constraint (constraint 1) as the canonical example of the shape,
and it is exactly as described: a pure `calculate_*` result compared to a stored field
via `eq(...)`.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.beta_fast_alpha` | read | explicit-arg | `physics.py:706` |
| `.physics.beta_beam` | read | explicit-arg | `fusion_reactions.py:1079`, see constraint 7's own finding for this field's conditional-computation caveat |
| `.physics.nd_plasma_electrons_vol_avg` | read | explicit-arg | iteration variable 6 (`iteration_variables.py:50`) -- a free unknown, not derived |
| `.physics.temp_plasma_electron_density_weighted_kev` | read | explicit-arg | `plasma_profiles.py:154/217` |
| `.physics.nd_plasma_ions_total_vol_avg` | read | explicit-arg | `physics.py:1323` |
| `.physics.temp_plasma_ion_density_weighted_kev` | read | explicit-arg | `plasma_profiles.py:158/218` |
| `.physics.b_plasma_total` | read | explicit-arg | `physics.py:373` |
| `.physics.beta_total_vol_avg` | read (bound) | explicit-arg | iteration variable 5 (`iteration_variables.py:49`) -- see finding below |

**proposed signature**: see `batch0_constraints.py::constraint_1`'s docstring.

**hole-in-MDA**: **No** for the tokamak path -- every operand is either an
unconditionally-computed field or a legitimate free iteration variable feeding an
equality constraint (the standard SAND shape this whole architecture is built around).

**Real PROCESS finding, stellarator-specific**: `Stellarator.run()`
(`process/models/stellarator/stellarator.py:1917-1930`) explicitly **raises**
`ProcessValueError` if `beta_total_vol_avg` (iteration variable 5) is in `numerics.ixc`
when `istell > 0` -- "Beta should not be in ixc if istell>0. Use Constraints 24 and 84
instead" -- and then directly overwrites `.physics.beta_total_vol_avg` with this exact
formula's RHS, commented "This replaces constraint equation 1 as it is just an
equality." **Constraint 1 is therefore never genuinely active on a stellarator run** --
PROCESS's own stellarator code path inlines it as a direct assignment rather than an
equality constraint. The port itself is a faithful general-purpose port (constraint 1 is
real PROCESS code, evaluable regardless); this finding just explains why it should never
be expected to appear in `numerics.icc` for any stellarator-scoped configuration
downstream, mirroring constraints 24/84's role there instead.

**switches touched**: none directly (no branch in the constraint body itself) -- the
`istell`-driven exclusivity with constraint 1 is enforced by `Stellarator.run()`, not by
this constraint.

---

### Constraint 2: global power balance (total)

**source**: `process/core/solver/constraints.py:254-322`.

**calls**: none -- bare residual read.

**data footprint**: all ten operands (`pden_electron_transport_loss_mw`,
`pden_ion_transport_loss_mw` -- `physics.py:901/904`; `pden_plasma_rad_mw`,
`pden_plasma_core_rad_mw` -- `physics.py:753/751`; `f_p_alpha_plasma_deposited` -- input
constant, `physics_variables.py:731`, default 0.95; `pden_alpha_total_mw`,
`pden_non_alpha_charged_mw` -- `fusion_reactions.py`'s `set_fusion_powers`, stored via
tuple-unpack at `physics.py:690/699`, already ported this session (`SetFusionPowers`);
`pden_plasma_ohmic_mw` -- `physics.py:770`; `p_hcd_injected_total_mw` --
`current_drive.py:2265`; `vol_plasma` -- `plasma_geometry.py:491`) are unconditionally
computed before constraints run.

**hole-in-MDA**: No.

**switches touched**: `.physics.i_rad_loss` (0/1/2, selects the radiation term),
`.physics.i_plasma_ignited` (`PlasmaIgnitionModel`, selects whether injected power is
in the denominator) -- both static, reads-set differs slightly by branch but the
difference is one term each, kept as plain `if`/`elif` in one function (same
proportionality judgement `physics_B_composition.md`'s `i_plasma_ignited` entry (formerly
spelled `is_ignited`) already makes
for a two-line differing branch inside an otherwise-shared body).

---

### Constraint 3: global power balance for ions

**source**: `process/core/solver/constraints.py:325-371`.

**calls**: none -- bare residual read.

**data footprint**: `pden_ion_transport_loss_mw` (`physics.py:904`),
`pden_ion_electron_equilibration_mw` (`physics.py:722`, `rether(...)`),
`f_p_alpha_plasma_deposited` (input constant), `f_pden_alpha_ions_mw`
(`fusion_reactions.py`'s `set_fusion_powers`, tuple-unpacked at `physics.py:692/2044`),
`p_hcd_injected_ions_mw` (`current_drive.py:2284`), `vol_plasma`
(`plasma_geometry.py:491`) -- all unconditionally computed.

**hole-in-MDA**: No.

**switches touched**: `.physics.i_plasma_ignited` -- selects whether injected power is
added to the RHS.

---

### Constraint 4: global power balance for electrons

**source**: `process/core/solver/constraints.py:374-435`.

**calls**: none -- bare residual read. Structurally identical to constraint 2
(same `i_rad_loss`/`i_plasma_ignited` shape), electron-only quantities.

**data footprint**: `pden_electron_transport_loss_mw` (`physics.py:901`),
`pden_plasma_rad_mw`/`pden_plasma_core_rad_mw` (as constraint 2),
`f_p_alpha_plasma_deposited` (input constant), `f_pden_alpha_electron_mw`
(`fusion_reactions.py`'s `set_fusion_powers`, tuple-unpacked at `physics.py:691/2037`),
`pden_ion_electron_equilibration_mw` (`physics.py:722`), `p_hcd_injected_electrons_mw`
(`current_drive.py:2279`), `vol_plasma` -- all unconditionally computed.

**hole-in-MDA**: No.

**switches touched**: `.physics.i_rad_loss`, `.physics.i_plasma_ignited` -- same as
constraint 2.

---

### Constraint 5: electron density upper limit

**source**: `process/core/solver/constraints.py:438-480`.

**calls**: none -- bare residual read.

**data footprint**: `nd_plasma_electron_line` (`plasma_profiles.py:136/234`, only used
on the Greenwald branch), `nd_plasma_electrons_vol_avg` (iteration variable 6, used on
every other branch), `nd_plasma_electrons_max` (`density_limit.py:95`,
`get_density_limit_value`, already the subject of this codebase's density-limit unit --
`SudoDensityLimit` is registered in `total_process.py`, though that's the Sudo model
specifically; the general `get_density_limit_value` dispatch across all seven
`i_density_limit` models is a separate, larger unit not audited here), and
`f_nd_plasma_electron_limit_max` (input constant, `constraint_variables.py:22`, default
1.0) -- all unconditionally available given whichever density-limit model actually ran.

**hole-in-MDA**: No, conditional on the assumption that whichever `i_density_limit`
model is selected has itself run before constraints are evaluated (true for the ordinary
PROCESS call sequence; not independently re-verified for all seven models here, only
that `nd_plasma_electrons_max` is written by the dispatch entry point regardless of which
model it selects).

**switches touched**: `.physics.i_density_limit` (`DensityLimitModel`, selects which of
two source fields is compared, Greenwald only).

---

### Constraint 6: epsilon beta-poloidal upper limit

**source**: `process/core/solver/constraints.py:483-495`.

**calls**: none -- bare residual read, no switch.

**data footprint**: `beta_poloidal_eps` (`physics.py:3837`), `beta_poloidal_eps_max`
(input constant, `physics_variables.py:714`, default 1.38) -- both unconditionally
available.

**hole-in-MDA**: No.

**switches touched**: none.

---

### Constraint 7: hot beam ion density consistency

**source**: `process/core/solver/constraints.py:498-524`.

**calls**: none -- bare residual read, plus the source's own domain-invalid raise
(`i_plasma_ignited == IGNITED`), reproduced as a plain `ValueError`.

**data footprint**: `nd_beam_ions_out`, `nd_beam_ions` -- see finding below.

**Real PROCESS finding**: `nd_beam_ions_out`'s only real producer anywhere in the
codebase is `reactions.beam_fusion` (`process/models/physics/physics.py:617-628`,
tuple-unpacked alongside `beta_beam`/`p_beam_alpha_mw`), called **only when**
`current_drive.c_beam_total != 0` **and** `i_plasma_ignited == NON_IGNITED`. Constraint
7's own guard only checks the ignition switch, not the beam-current one -- so a
non-ignited, zero-beam-current configuration can activate this constraint while
`nd_beam_ions_out` has silently never left its dataclass default (`0.0`,
`physics_variables.py:666`). This is the *same* conditional-producer shape already
flagged this session for `beta_beam` ("`beta_beam`'s sole owner is `beam_fusion`,
unported") -- `nd_beam_ions_out` shares that exact producer and that exact gating
condition, just a second field out of the same tuple-unpack. Not a hole in the port
(the constraint's own math is fully and faithfully ported either way), but a real
upstream-coverage gap worth tracking alongside `beta_beam`'s existing entry rather than
as a new, separate issue.

`nd_beam_ions` (the RHS) is unconditionally computed --
`.physics.nd_beam_ions`, `plasma_composition`'s own output (`PlasmaComposition` node,
already ported and registered this session).

**hole-in-MDA**: **Conditional hole** -- see finding above. Not fixed here (matches
`beta_beam`'s already-documented status, same unported producer).

**switches touched**: `.physics.i_plasma_ignited` -- raises if `IGNITED`.

---

### Constraint 8: neutron wall load upper limit

**source**: `process/core/solver/constraints.py:527-538`.

**calls**: none -- bare residual read, no switch.

**data footprint**: `pflux_fw_neutron_mw` (device-type-dependent producer --
`fw.py:122/127`, `ife.py:1404-1427`, or `stellarator.py:2096-2108`, each unconditional
within its own device-type branch; confirmed present on the stellarator path),
`pflux_fw_neutron_max_mw` (input constant, `constraint_variables.py:111`, default 1.0).

**hole-in-MDA**: No (on the stellarator path specifically verified).

**switches touched**: none directly (device-type selection happens upstream of this
constraint, not inside it).

---

### Constraint 9: fusion power upper limit

**source**: `process/core/solver/constraints.py:541-551`, `constraint_equation_9`.
`leq(p_fusion_total_mw, p_fusion_total_max_mw)`.

**calls**: none — bare residual read.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.p_fusion_total_mw` | read | explicit-arg | produced by `functional_process/models/physics/fusion_reactions.py` (already ported) |
| `.constraints.p_fusion_total_max_mw` | read | explicit-arg | plain input constant, never computed by any model |

**hole-in-MDA**: **No.** Both operands already have real producers in this codebase.

**switches touched**: none.

**proposed signature**: `constraint_9(p_fusion_total_mw, p_fusion_total_max_mw)` →
`leq(...)`. Implemented in `batch1_constraints.py`.

---

### Constraint 11: radial build consistency (equality)

**source**: `process/core/solver/constraints.py:555-565`, `constraint_equation_11`.
`eq(data.build.rbld, data.physics.rmajor)`.

**calls**: none — bare residual read.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.build.rbld` | read | explicit-arg | produced by `functional_process/models/stellarator/build.py`'s `Build` node (already ported) |
| `.physics.rmajor` | read | explicit-arg | produced throughout this codebase's already-ported physics units |

**hole-in-MDA**: **No.**

**switches touched**: none.

**proposed signature**: `constraint_11(rbld, rmajor)` → `eq(...)`. First equality
constraint audited in this codebase — needed `eq` ported alongside `leq`/`geq` (see
`batch1_constraints.py`'s module docstring for the consolidation note).

---

### Constraint 12: volt-second capability lower limit

**source**: `process/core/solver/constraints.py:569-584`, `constraint_equation_12`.
`geq(-data.pf_coil.vs_cs_pf_total_pulse, data.physics.vs_plasma_total_required)` — note
the sign flip on the first operand, done at the call site in the source (comment:
"vs_cs_pf_total_pulse is negative, requires sign change").

**calls**: none — bare residual read.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.pf_coil.vs_cs_pf_total_pulse` | read | explicit-arg | produced by `process/models/pfcoil.py:1710` — **PF coil subsystem entirely unported** in this codebase (no `functional_process/models/pfcoil*` exists) |
| `.physics.vs_plasma_total_required` | read | explicit-arg | produced by `process/models/physics/physics.py:4889` (plasma-current/inductance section) — **not yet ported** |

**hole-in-MDA**: **Yes, both operands.** Ported anyway per this pass's instruction —
an unported producer is a wiring gap for whoever ports the PF coil/inductance
subsystems later, not a reason to withhold the constraint's own (trivial) arithmetic.
Same precedent constraint 91 already set for `powerht_constraint`/
`powerscaling_constraint`.

**switches touched**: none.

**proposed signature**: `constraint_12(vs_cs_pf_total_pulse, vs_plasma_total_required)`
→ `geq(...)`. The port takes `vs_cs_pf_total_pulse` **already sign-flipped** by the
caller (harness reference adapter applies `-data.pf_coil.vs_cs_pf_total_pulse` before
calling PROCESS's real constraint, and passes the same flipped value to the port) — the
sign flip is source-level caller bookkeeping, not part of the constraint's own
comparison logic.

---

### Constraint 13: burn time lower limit

**source**: `process/core/solver/constraints.py:588-598`, `constraint_equation_13`.
`geq(data.times.t_plant_pulse_burn, data.constraints.t_burn_min)`.

**calls**: none — bare residual read.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.times.t_plant_pulse_burn` | read | explicit-arg | see hole-in-MDA note — not a real hole for this codebase's current (stellarator) scope |
| `.constraints.t_burn_min` | read | explicit-arg | plain input constant |

**hole-in-MDA**: **No, for this codebase's current scope.** `t_plant_pulse_burn` has a
real *model* producer only on the pulsed-tokamak path (`process/models/pulse.py`'s
`calculate_burn_time`), out of scope here. On the stellarator path,
`process/models/stellarator/initialization.py:45` sets it to a fixed constant
(`3.15576e7`, one year — continuous, non-pulsed operation), not a computed value.
`functional_process/models/stellarator/initialization.py`'s already-ported
`PulseDurations` node already reads `t_plant_pulse_burn` as a plain `Input` rather than
expecting an upstream producer, confirming this classification is already this
codebase's established convention, not a new one invented here.

**switches touched**: none (the pulsed/continuous choice is `i_pulsed_plant`, out of
scope for the same reason `pulse.py` itself is).

**proposed signature**: `constraint_13(t_plant_pulse_burn, t_burn_min)` → `geq(...)`.

---

### Constraint 14: NBI e-decay lengths to plasma centre (equality)

**source**: `process/core/solver/constraints.py:602-614`, `constraint_equation_14`.
`eq(data.current_drive.n_beam_decay_lengths_core,
data.current_drive.n_beam_decay_lengths_core_required)`.

**calls**: none — bare residual read.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.current_drive.n_beam_decay_lengths_core` | read | explicit-arg | produced by `process/models/physics/current_drive.py:201,294` — **current-drive (NBI) subsystem entirely unported** |
| `.current_drive.n_beam_decay_lengths_core_required` | read | explicit-arg | plain input constant |

**hole-in-MDA**: **Yes.** Same treatment as constraint 12 — ported anyway.

**switches touched**: none.

**proposed signature**:
`constraint_14(n_beam_decay_lengths_core, n_beam_decay_lengths_core_required)` →
`eq(...)`.

---

### Constraint 15: L-H power threshold limit (H-mode enforcement)

**source**: `process/core/solver/constraints.py:618-635`, `constraint_equation_15`.
`geq(data.physics.p_plasma_separatrix_mw, data.physics.p_l_h_threshold_mw *
data.constraints.f_h_mode_margin)`.

**calls**: none — bare residual read (the multiplication by `f_h_mode_margin` is
inline arithmetic, not a `calculate_*` re-derivation, so this stays "bare residual",
not `Compare`-shaped).

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.p_plasma_separatrix_mw` | read | explicit-arg | produced by `functional_process/models/stellarator/stellarator_B_st_phys.py` (already ported) |
| `.physics.p_l_h_threshold_mw` | read | explicit-arg | produced by `process/models/physics/l_h_transition.py:86` — **unported** |
| `.constraints.f_h_mode_margin` | read | explicit-arg | plain input constant, default `1.0` |

**hole-in-MDA**: **Partial** — one of two physics operands (`p_l_h_threshold_mw`) has
no producer yet. Ported anyway, same reasoning as 12/14.

**switches touched**: none.

**proposed signature**:
`constraint_15(p_plasma_separatrix_mw, p_l_h_threshold_mw, f_h_mode_margin)` →
`geq(p_plasma_separatrix_mw, p_l_h_threshold_mw * f_h_mode_margin)`.

---

### Constraint 16: net electric power lower limit

**source**: `process/core/solver/constraints.py:639-649`, `constraint_equation_16`.
`geq(data.heat_transport.p_plant_electric_net_mw,
data.constraints.p_plant_electric_net_required_mw)`.

**calls**: none — bare residual read.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.heat_transport.p_plant_electric_net_mw` | read | explicit-arg | produced by `functional_process/models/power_C_electric_production.py`'s `PlantElectricProduction` node — built and harness-tested, **not yet registered** in `total_process.py` (a known, separate gap, not this constraint's concern) |
| `.constraints.p_plant_electric_net_required_mw` | read | explicit-arg | plain input constant |

**hole-in-MDA**: **No** — the producer exists and is tested, just not yet wired into
the assembled graph (a registration gap, not a missing port).

**switches touched**: none.

**proposed signature**:
`constraint_16(p_plant_electric_net_mw, p_plant_electric_net_required_mw)` →
`geq(...)`.

---

### Constraint 17: plasma radiation fraction upper limit
**source**: `process/core/solver/constraints.py:653-674` (line numbers verified against
current source — unchanged from the earlier brief note). General constraint, not
stellarator-specific, but the `istell`-gated adjustment branch is squarely in this
audit's scope (see the correction note above). Docstring/logic:
`f_p_plasma_separatrix_rad <= f_p_plasma_separatrix_rad_max`, with the LHS adjusted when
`istell != 0`.

**calls**: no function call inside `constraint_equation_17` — reads `data` fields
directly and calls `eq_geq`'s `leq(...)`. Same shape as constraint 91.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.stellarator.istell` | read | explicit-arg (switch) | selects whether the SOL-radiation correction applies. The reads-set is the same either way (`psolradmw`/`p_plasma_heating_total_mw` are always plain arguments, just unused when `istell == 0`) — a formula-changing switch with a stable reads-set, so per `naming_convention.md` it becomes a static kwarg, not a `VarPath` |
| `.physics.f_p_plasma_separatrix_rad` | read | explicit-arg | unconditionally computed by `physics.py:1080` (`self.exhaust.calculate_radiation_fraction(...)`), before constraints are ever evaluated |
| `.constraints.f_p_plasma_separatrix_rad_max` | read | explicit-arg | plain input constant (`data_structure/constraint_variables.py:25`, default `1.0`); grepped for any producer — none found, it is never computed by any model |
| `.physics.psolradmw` | read | explicit-arg | only used in the `istell != 0` branch; written unconditionally by `Stellarator.run()` (`process/models/stellarator/stellarator.py:2203`, `= .stellarator.f_rad * powht`) whenever the stellarator model runs at all, so it is available by the time constraints are evaluated on a stellarator run |
| `.physics.p_plasma_heating_total_mw` | read | explicit-arg | denominator of the SOL correction; unconditionally computed by `physics.py:1071` before constraints run |

**proposed signature**:
```python
def constraint_17(
    istell: int,  # static — 0: tokamak model, nonzero: stellarator model
    f_p_plasma_separatrix_rad: float,
    f_p_plasma_separatrix_rad_max: float,
    psolradmw: float,
    p_plasma_heating_total_mw: float,
) -> tuple[float, float, float, float]:  # (residual, normalised_residual, value, bound)
    if istell != 0:
        f_rad_sol = psolradmw / p_plasma_heating_total_mw
        value = f_p_plasma_separatrix_rad - f_rad_sol
    else:
        value = f_p_plasma_separatrix_rad
    return leq(value, f_p_plasma_separatrix_rad_max)
```

**hole-in-MDA**: **No.** Traced both producers directly (`physics.py` for the base
fraction and heating power, `stellarator.py` for `psolradmw`, both unconditional):
every operand is already fully forward-computed before constraints are evaluated,
in both the `istell == 0` and `istell != 0` cases. No missing producer, no free
iteration variable standing in for an unwired relationship. Confidence: **high** on the
structural classification.

**Real PROCESS finding, documented not fixed**: the source itself carries an open
`# TODO` doubting the `istell != 0` branch: *"this is replicating behaviour before
#4299 / is this really what should happen?"* (`constraints.py:663-664`). This is an
upstream, unresolved question about the physics/history of the formula, not a porting
concern — the port reproduces the formula exactly as written, TODO and all, per this
session's "document, don't fix" policy.

**current closure mechanism**: VMCON-joint. No local solver — the function does nothing
but read `data` and call `leq(...)`, same shape as every other `ConstraintManager`
registration.

**candidate iteration variable(s)**: no direct name match (unlike 91's
`te0_ecrh_achievable` or 24's `beta_total_vol_avg` below). Best-effort, and weaker than
those two: the `f_nd_impurity_electrons(03)`..`(14)` array (IDs 125-136, module
`impurity_radiation`, `iteration_variables.py:125-136`) controls impurity-driven
radiated power, which several models upstream feeds into
`.physics.f_p_plasma_separatrix_rad` — a plausible but indirect chain, not a
one-hop argument relationship like 91's or 24's pairing. Flagged as weak on purpose
rather than overstated.

**confidence**: high (structural classification) / low (whether the `istell` TODO's
adjustment is physically correct — outside what a code read can settle).

**open questions**: whether the SOL-radiation subtraction under `istell != 0` is the
physically intended behaviour (PROCESS's own unresolved TODO, not this audit's to
answer).

**cottax shape**: **bare residual read**, not `Compare` — `f_p_plasma_separatrix_rad_max`
is a plain input constant (never another node's output), and no `calculate_*` runs
inside the constraint body itself (contrast constraint 1's `PlasmaBeta.calculate_plasma_beta`
pattern, which would be `Compare`-shaped). `istell` is a static kwarg selecting the
(reads-set-stable) formula, not a `VarPath`.

---

### Constraint 18: divertor heat load upper limit

**source**: `process/core/solver/constraints.py:677-687`, `constraint_equation_18`.
`leq(data.divertor.pflux_div_heat_load_mw, data.divertor.pflux_div_heat_load_max_mw)`.

**calls**: none — bare residual read.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.divertor.pflux_div_heat_load_mw` | read | explicit-arg | produced by `functional_process/models/stellarator/divertor.py`'s already-ported `Divertor` node |
| `.divertor.pflux_div_heat_load_max_mw` | read | explicit-arg | plain input constant |

**hole-in-MDA**: **No.**

**switches touched**: none.

**proposed signature**:
`constraint_18(pflux_div_heat_load_mw, pflux_div_heat_load_max_mw)` → `leq(...)`.

---

### Constraint 19: MVA (power) upper limit, resistive TF coil set

**source**: `process/core/solver/constraints.py:691-701`.
`totmva = p_cp_resistive_mw + p_tf_leg_resistive_mw`, `leq(totmva, mvalim)`.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.tfcoil.p_cp_resistive_mw`, `.tfcoil.p_tf_leg_resistive_mw` | read | explicit-arg | both produced by `power_A_tf_coil_power.py` (already ported, confirmed via `grep`) |
| `.constraints.mvalim` | read | explicit-arg | input constant, default `40.0` (`constraint_variables.py:60`), no producer |

**proposed signature**: `constraint_19(p_cp_resistive_mw, p_tf_leg_resistive_mw, mvalim)`.

**hole-in-MDA**: **No.** Both physics operands already have a ported producer.

---

### Constraint 20: neutral beam tangency radius upper limit

**source**: `process/core/solver/constraints.py:704-715`.
`leq(radius_beam_tangency, radius_beam_tangency_max)`.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.current_drive.radius_beam_tangency` | read | explicit-arg | **no ported producer** — `current_drive.py` (registry unit, whichever number covers it) is not ported in this codebase at all; no `functional_process/models/**` directory for it exists |
| `.current_drive.radius_beam_tangency_max` | read | explicit-arg | input constant, default `0.0` (`current_drive_variables.py:295`) — a `0.0` default with no visible producer is a strong hint the real value is set by input file, not computed, but not independently confirmed here |

**proposed signature**: `constraint_20(radius_beam_tangency, radius_beam_tangency_max)`.

**hole-in-MDA**: **Yes.** `radius_beam_tangency` has no producer anywhere in this
codebase — the entire neutral-beam current-drive subsystem is unported. The constraint's
*pure function* is still ported here (it's a two-line `leq`, no reason to withhold it),
but wiring it into any real `Optimise` assembly against a stellarator run would need
`radius_beam_tangency` supplied as a free/boundary input until that subsystem lands.

---

### Constraint 21: minor radius lower limit

**source**: `process/core/solver/constraints.py:718-729`. `geq(rminor, rminor_min)`.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.rminor` | read | explicit-arg | produced by `stellarator_C_geometry.py`'s `StellaratorGeometry`-family node (confirmed: `Output(lambda s: s.physics.rminor)`) |
| `.build.rminor_min` | read | explicit-arg | input constant, default `0.25` (`build_variables.py:58`), no producer |

**proposed signature**: `constraint_21(rminor, rminor_min)`.

**hole-in-MDA**: **No.**

---

### Constraint 22: L-H power threshold limit (enforce L-mode)

**source**: `process/core/solver/constraints.py:732-750`.
`geq(p_l_h_threshold_mw, f_l_mode_margin * p_plasma_separatrix_mw)`.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.p_l_h_threshold_mw` | read | explicit-arg | **no ported producer found** — default `0.0` (`physics_variables.py:1237`); the L-H threshold scaling model that would compute this is not ported |
| `.constraints.f_l_mode_margin` | read | explicit-arg | input constant, default `1.0` (`constraint_variables.py:128`) |
| `.physics.p_plasma_separatrix_mw` | read | explicit-arg | produced by `stellarator_B_st_phys.py` (confirmed: `Output(lambda s: s.physics.p_plasma_separatrix_mw)`) |

**proposed signature**: `constraint_22(p_l_h_threshold_mw, f_l_mode_margin, p_plasma_separatrix_mw)`.

**hole-in-MDA**: **Yes**, for `p_l_h_threshold_mw` — the L-H threshold scaling model is
unported. Function still ported (bare `leq`/`geq`-shape, no blocker to writing it).

---

### Constraint 23: conducting shell radius / rminor upper limit

**source**: `process/core/solver/constraints.py:753-780`.
`rcw = rminor + dr_fw_plasma_gap_outboard + dr_fw_outboard + dr_blkt_outboard`,
`leq(rcw, f_r_conducting_wall * rminor)`.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.rminor` | read | explicit-arg | ported, see constraint 21 |
| `.build.dr_fw_plasma_gap_outboard` | read | explicit-arg | input constant — read as `Input` by several ported nodes (`build.py`, `stellarator_fwbs_s1_s5.py`) but never written by any of them; a plain design parameter |
| `.build.dr_fw_outboard` | read | explicit-arg | produced by `build.py`'s `Build` node (confirmed: `Output(lambda s: s.build.dr_fw_outboard)`) |
| `.build.dr_blkt_outboard` | read | explicit-arg | produced by `build.py`'s `Build` node (confirmed: `Output(lambda s: s.build.dr_blkt_outboard)`) |
| `.physics.f_r_conducting_wall` | read | explicit-arg | input constant, default `1.35` (`physics_variables.py:638`) |

**proposed signature**:
`constraint_23(rminor, dr_fw_plasma_gap_outboard, dr_fw_outboard, dr_blkt_outboard, f_r_conducting_wall)`.

**hole-in-MDA**: **No.** Every operand is either an input constant or already produced.

---

### Constraint 24: beta upper limit
**source**: `process/core/solver/constraints.py:783-824` (decorator at 783, function body
784-824). **Line numbers corrected** — the earlier brief note in this file cited
`803-806`, which had already drifted; the function is materially longer than that range
suggested and was not previously read in full.

**calls**: no function call inside `constraint_equation_24` — reads `data` fields
directly (four candidate `value` sources, switch-selected) and calls `eq_geq`'s
`leq(...)`. Same shape as 17 and 91: no internal solve, no callee.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.i_beta_component` | read | explicit-arg (switch) | selects which of four already-computed beta fields becomes `value`. Unlike 17's `istell` branch, the **reads-set genuinely differs per branch** (each arm reads a different subset of the four beta fields) — not the "provably identical reads-set" case `naming_convention.md` expects to be the common one. Still resolved as a static kwarg rather than a `VarPath`: it is a run-configuration enum choice, not a differentiable physical quantity, and every branch's *possible* reads are declared as plain arguments regardless of which one fires (same treatment as `i_plasma_ignited` in constraint 91) |
| `.stellarator.istell` | read | explicit-arg (switch) | when nonzero, **forces the `TOTAL` branch regardless of `i_beta_component`** — see the real finding below |
| `.physics.beta_total_vol_avg` | read | explicit-arg | used whenever `i_beta_component == TOTAL` or `istell != 0`; also iteration variable ID 5 (`iteration_variables.py:49`) — see candidate below |
| `.physics.beta_thermal_vol_avg` | read | explicit-arg | used in the `THERMAL`/`THERMAL_AND_BEAM` branches only; computed unconditionally by `physics.py:3831` |
| `.physics.beta_beam` | read | explicit-arg | used in the `THERMAL_AND_BEAM` branch only; computed by `process/models/physics/fusion_reactions.py:1079` |
| `.physics.beta_toroidal_vol_avg` | read | explicit-arg | used in the `TOROIDAL` branch only; computed unconditionally by `physics.py:3818` |
| `.physics.beta_vol_avg_max` | read | explicit-arg | the bound. **Itself a computed physics output** (`physics.py:3811`,`self.calculate_beta_limit_from_norm(...)`, a Troyon-style scaling), not a raw user input — but `constraint_equation_24` only *reads* it, it does not recompute it, so this is still the bare-residual shape rather than `Compare` (see cottax-shape note below) |

**proposed signature**:
```python
def constraint_24(
    i_beta_component: int,  # BetaComponentLimits, static
    istell: int,  # static
    beta_total_vol_avg: float,
    beta_thermal_vol_avg: float,
    beta_beam: float,
    beta_toroidal_vol_avg: float,
    beta_vol_avg_max: float,
) -> tuple[float, float, float, float]:  # (residual, normalised_residual, value, bound)
    if i_beta_component == BetaComponentLimits.TOTAL or istell != 0:
        value = beta_total_vol_avg
    elif i_beta_component == BetaComponentLimits.THERMAL:
        value = beta_thermal_vol_avg
    elif i_beta_component == BetaComponentLimits.THERMAL_AND_BEAM:
        value = beta_thermal_vol_avg + beta_beam
    elif i_beta_component == BetaComponentLimits.TOROIDAL:
        value = beta_toroidal_vol_avg
    else:
        raise ValueError(...)  # BetaComponentLimits has no other member
    return leq(value, beta_vol_avg_max)
```

**hole-in-MDA**: **No.** Every operand, including the bound `beta_vol_avg_max`, is
unconditionally produced by `physics.py` (plus `fusion_reactions.py` for `beta_beam`)
before constraints are evaluated — traced each producer directly, same method as 17 and
91. No missing producer, no free iteration variable standing in for an unwired
relationship. Confidence: **high**.

**Real PROCESS finding, documented not fixed**: when `istell != 0`, the branch selection
**ignores `i_beta_component` entirely** and always uses `beta_total_vol_avg` — a
stellarator run with `i_beta_component` set to `THERMAL`/`THERMAL_AND_BEAM`/`TOROIDAL`
silently gets the total-beta limit instead, with no warning anywhere in the constraint
function. Whether that reflects "stellarators only ever get a total-beta limit,
intentionally" or is an oversight mirroring 17's `istell` TODO is outside what a code
read alone can settle — flagged, not fixed, per this session's policy.

**current closure mechanism**: VMCON-joint. No local solver, same as 17/91.

**candidate iteration variable(s)**: **`beta_total_vol_avg`** (ID 5, module `physics`,
bounds `0.001`–`1.0`, `iteration_variables.py:49`) — direct, strong evidence, stronger
than the usual best-effort pairing: it is not merely correlated with this constraint, it
**is** the `value` operand in the `TOTAL`/`istell != 0` branches, i.e. the majority of
cases and *every* stellarator run (since `istell != 0` forces that branch — see the
finding above). Not investigated further: whether `beta_thermal_vol_avg`/`beta_beam`/
`beta_toroidal_vol_avg` deserve their own candidate pairings for the non-stellarator,
non-`TOTAL` branches.

**confidence**: high (structural classification) / medium (whether the `istell` override
is intentional).

**open questions**: whether the `istell`-forces-`TOTAL` override is deliberate stellarator
policy or an oversight; no attempt made to find candidate iteration variables for the
`THERMAL`/`THERMAL_AND_BEAM`/`TOROIDAL` branches beyond the direct `TOTAL` match above.

**cottax shape**: **bare residual read**, same reasoning as 17 — `beta_vol_avg_max` is
read as a plain already-produced field, not recomputed inside the constraint body (no
`Compare`-shaped "call a `calculate_*`, compare to a stored field" pattern here, unlike
constraint 1). `i_beta_component` and `istell` are both static kwargs.

---

### Constraint 25: peak toroidal field upper limit

**source**: `process/core/solver/constraints.py:827-838`.
`leq(b_tf_inboard_peak_with_ripple, b_tf_inboard_max)`.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.tfcoil.b_tf_inboard_peak_with_ripple` | read | explicit-arg | **no ported producer found** — default `0.0` (`tfcoil_variables.py:71`); the TF ripple-field calculation is not ported in this codebase |
| `.constraints.b_tf_inboard_max` | read | explicit-arg | input constant, default `12.0` (`constraint_variables.py:19`) |

**proposed signature**: `constraint_25(b_tf_inboard_peak_with_ripple, b_tf_inboard_max)`.

**hole-in-MDA**: **Yes**, for `b_tf_inboard_peak_with_ripple`. Function still ported.

---

### Constraint 26: Central Solenoid current density upper limit at end-of-flattop

**source**: `process/core/solver/constraints.py:841-855`.
`leq(j_cs_flat_top_end / j_cs_critical_flat_top_end, fjohc)`.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.pf_coil.j_cs_flat_top_end`, `.pf_coil.j_cs_critical_flat_top_end` | read | explicit-arg | **no ported producer** — no PF coil / Central Solenoid model is ported anywhere in `functional_process/models/**` (confirmed: no `pf_coil.py`, only unrelated cost-model references to the `pf_coil` area name) |
| `.constraints.fjohc` | read | explicit-arg | input constant, default `0.7` (`constraint_variables.py:39`) |

**proposed signature**: `constraint_26(j_cs_flat_top_end, j_cs_critical_flat_top_end, fjohc)`.

**hole-in-MDA**: **Yes**, both physics operands. Worth flagging beyond the individual
hole: PF coil / Central Solenoid is not a stellarator subsystem at all in the physical
sense (stellarators are net-current-free, no CS needed for plasma current drive) — this
constraint is realistically **tokamak-only in practice**, even though nothing in its
own body or `numerics.icc` selection gates it by `istell`. Not excluded here (per this
pass's "port unless genuinely blocked" instruction, and the function itself has no real
blocker — it's a two-line ratio-vs-margin check), but flagged so a later pass doesn't
spend effort chasing a stellarator producer for `j_cs_flat_top_end` that will likely
never exist in this codebase's stellarator scope.

---

### Constraint 27: Central Solenoid current density upper limit at beginning-of-pulse

**source**: `process/core/solver/constraints.py:858-872`.
`leq(j_cs_pulse_start / j_cs_critical_pulse_start, fjohc0)`.

**data footprint**: identical shape to constraint 26, beginning-of-pulse instead of
end-of-flattop. Same hole, same tokamak-only-in-practice caveat.

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.pf_coil.j_cs_pulse_start`, `.pf_coil.j_cs_critical_pulse_start` | read | explicit-arg | no ported producer — see constraint 26 |
| `.constraints.fjohc0` | read | explicit-arg | input constant, default `0.7` (`constraint_variables.py:44`) |

**proposed signature**: `constraint_27(j_cs_pulse_start, j_cs_critical_pulse_start, fjohc0)`.

**hole-in-MDA**: **Yes**, both physics operands. See constraint 26's caveat.

---

### Constraint 28: fusion gain (big Q) lower limit

**source**: `process/core/solver/constraints.py:875-904`. `geq(big_q_plasma,
big_q_plasma_min)`, with a real `ProcessValueError` raised if
`i_plasma_ignited != NON_IGNITED` ("Obviously, ignite must be zero if current drive is
required").

**calls**: none inside the constraint body -- bare residual read plus the switch-gated
raise.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.i_plasma_ignited` | read | explicit-arg (switch, static) | selects whether the constraint is even valid; not NON_IGNITED raises |
| `.current_drive.big_q_plasma` | read | explicit-arg | produced by `stellarator/heating.py` (already ported) |
| `.constraints.big_q_plasma_min` | read | explicit-arg (constant) | plain input, default 10.0, never written by any model |

**proposed signature**: see `batch3_constraints.py::constraint_28`.

**hole-in-MDA**: no -- `big_q_plasma` is already produced by an already-ported unit.

**cottax shape**: bare residual read, not `Compare`-shaped (no `calculate_*`
re-derivation inside the body).

**switches touched**: `.physics.i_plasma_ignited` (`PlasmaIgnitionModel`) -- gates
validity, not just formula selection; the invalid branch is a hard error, ported as a
plain Python `ValueError` (not `process.core.exceptions.ProcessValueError`, to avoid an
unnecessary dependency -- same choice constraint 24's existing `ValueError` already made
for its own invalid-switch raise).

---

### Constraint 29: inboard major radius consistency

**source**: `process/core/solver/constraints.py:907-919`. Equality constraint:
`eq(rmajor - rminor, rinboard)`.

**calls**: none.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.rmajor`, `.physics.rminor` | read | explicit-arg | core physics outputs, ported long since |
| `.build.rinboard` | read | explicit-arg (constant) | plain input, default 0.651, never written by any model -- confirmed by grepping every writer under `process/models` |

**proposed signature**: see `batch3_constraints.py::constraint_29`.

**hole-in-MDA**: no.

**cottax shape**: bare residual read.

**switches touched**: none.

**Note**: this is the first equality constraint ported in this whole audit (17/24/82/83/91
are all inequalities). PROCESS's `eq` helper was not yet in the canonical
`constraints.py` (its own docstring says so) -- ported here (`batch3_constraints.py::eq`)
and flagged for the merge to promote into the canonical module rather than staying
duplicated per batch.

---

### Constraint 30: injection power upper limit

**source**: `process/core/solver/constraints.py:922-933`.
`leq(p_hcd_injected_total_mw, p_hcd_injected_max)`.

**calls**: none.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.current_drive.p_hcd_injected_total_mw` | read | explicit-arg | produced by `stellarator/heating.py` (already ported) |
| `.current_drive.p_hcd_injected_max` | read | explicit-arg (constant) | plain input, default 150.0, never written |

**proposed signature**: see `batch3_constraints.py::constraint_30`.

**hole-in-MDA**: no.

**cottax shape**: bare residual read.

**switches touched**: none.

---

### Constraints 31, 32, 33: TF coil stress/current-density limits (SCTF) -- a real finding

All three share `process/models/tfcoil/superconducting.py` as the *intended* producer of
one operand each (`sig_tf_case`, `j_tf_wp_critical`) or read a stellarator-real field
(`sig_tf_wp`, `j_tf_wp`). **Real finding, verified directly against
`process/core/caller.py:272-275`**: `_call_models_once` special-cases
`if self.data.stellarator.istell != 0: self.models.stellarator.run(); return` -- an
*early return* before the tokamak/general `self.tfcoil.run()` (built from
`process/models/tfcoil/superconducting.py`/`resistive.py`) would otherwise be reached.
**`process/models/tfcoil/superconducting.py` is never called on a real stellarator run,
full stop** -- confirmed by reading `_call_models_once`'s control flow directly, not
inferred from absence.

Consequence, checked per-field by grepping every writer under `process/models`:

| field | writer | called for `istell != 0`? |
|---|---|---|
| `.tfcoil.sig_tf_case` | `tfcoil/superconducting.py`, `tfcoil/resistive.py` only | **no** |
| `.tfcoil.sig_tf_wp` | `stellarator/coils/forces.py` (`MaxForceDensity`, ported) | **yes** |
| `.tfcoil.j_tf_wp_critical` | `tfcoil/superconducting.py` only | **no** |
| `.tfcoil.j_tf_wp` | `stellarator/coils/calculate.py` (ported) | **yes** |
| `.tfcoil.j_tf_wp_quench_heat_max` | `stellarator/coils/quench.py` (`QuenchProtection`, ported) | **yes** |

So **constraint 31** (`sig_tf_case <= sig_tf_case_max`) and **constraint 33**
(`j_tf_wp <= j_tf_wp_critical * f_j_tf_wp_critical_max`) each compare a real
stellarator-computed or plain-constant quantity against a field
(`sig_tf_case`/`j_tf_wp_critical`) that **never leaves its `DataStructure` default on
any real stellarator run** (`sig_tf_case = 0.0`, `j_tf_wp_critical = 0.0`). Constraint 31
is then vacuously satisfied (`0.0 <= 6e8`) always; constraint 33 compares against an
always-zero bound (`0.0 * f_j_tf_wp_critical_max = 0.0`), which is either trivially
infeasible or degenerate depending on how a solver treats a zero bound in
`normalised_residual = (value/bound) - 1.0` (division by zero). **This is not a porting
gap** -- the pure arithmetic is trivial and ported below regardless -- **it is a
structural fact about PROCESS itself**: these two constraints have no real physics
content on a stellarator configuration, because their governing model
(`process/models/tfcoil/superconducting.py`) is a tokamak-only code path PROCESS itself
never reaches when `istell != 0`. Whoever eventually assembles an `Optimise` problem
for a stellarator run should exclude 31/33 from `numerics.icc`-equivalent active-
constraint selection, or accept they are permanently vacuous/degenerate for this device
type -- flagged here, not resolved (resolving it would mean either porting
`superconducting.py` for stellarators, which is out of scope, or accepting the
degeneracy).

**Constraint 32** (`sig_tf_wp <= sig_tf_wp_max`) has no such hole: both operands are
real for stellarators (`sig_tf_wp` from the already-ported `MaxForceDensity`;
`sig_tf_wp_max` a plain, never-written input constant).

**cottax shape**: all three bare residual reads, not `Compare`-shaped.

**switches touched**: none of the three read a switch directly (the `istell` dependency
here is entirely at the level of "which model runs at all," not a branch inside the
constraint body itself -- worth distinguishing from 17/24's `istell`-branched formulas,
a different shape again).

**proposed signatures**: see `batch3_constraints.py::constraint_31`/`32`/`33`.

---

### Constraint 34: TF coil dump voltage upper limit

**source**: `process/core/solver/constraints.py:988-999`.
`leq(v_tf_coil_dump_quench_kv, v_tf_coil_dump_quench_max_kv)`.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.tfcoil.v_tf_coil_dump_quench_kv` | read | explicit-arg | produced by `stellarator/coils/quench.py` (`QuenchProtection`, already ported) |
| `.tfcoil.v_tf_coil_dump_quench_max_kv` | read | explicit-arg (constant) | plain input, never written |

**hole-in-MDA**: no.

**cottax shape**: bare residual read. **switches touched**: none.

---

### Constraint 35: TF coil J_wp upper limit for quench protection

**source**: `process/core/solver/constraints.py:1002-1016`.
`leq(j_tf_wp, j_tf_wp_quench_heat_max)`.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.tfcoil.j_tf_wp` | read | explicit-arg | produced by `stellarator/coils/calculate.py` (already ported) |
| `.tfcoil.j_tf_wp_quench_heat_max` | read | explicit-arg | **not a plain constant** -- produced by `stellarator/coils/quench.py`'s `QuenchProtection` (already ported), unlike most other `_max` bounds in this batch |

**hole-in-MDA**: no -- unlike 31/33, both operands here are genuinely real for a
stellarator run.

**cottax shape**: bare residual read. **switches touched**: none.

---

### Constraint 36: TF coil superconductor temperature margin lower limit
**source**: `process/core/solver/constraints.py:1019-1032`.

**calls**: none — reads `data` directly, calls `geq(...)`.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.tfcoil.temp_tf_superconductor_margin` | read | explicit-arg | computed unconditionally by `process/models/tfcoil/superconducting.py:2749`/`4006` (two call sites, one per conductor-turn model, both always run for a superconducting TF coil) |
| `.tfcoil.temp_tf_superconductor_margin_min` | read | explicit-arg | the bound — plain input constant, set once at init (`process/core/init.py:1189`, `= data.tfcoil.tmargmin`, itself a user input), never recomputed |

**proposed signature**:
```python
def constraint_36(temp_tf_superconductor_margin, temp_tf_superconductor_margin_min):
    return geq(temp_tf_superconductor_margin, temp_tf_superconductor_margin_min)
```

**hole-in-MDA**: **No.** Both operands traced to unconditional producers (a superconducting-TF-coil model always computes the margin; the bound is a plain input). Confidence: **high**.

**current closure mechanism**: VMCON-joint, same as every constraint in this file.

**candidate iteration variable(s)**: none found by direct name match in `iteration_variables.py`.

**confidence**: high.

**open questions**: none.

**cottax shape**: bare residual read — the bound is a plain input constant, no `calculate_*` re-derivation inside the constraint body.

---

### Constraint 37: current drive gamma upper limit
**source**: `process/core/solver/constraints.py:1035-1046`.

**calls**: none — reads `data` directly, calls `leq(...)`.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.current_drive.eta_cd_norm_hcd_primary` | read | explicit-arg | computed by `process/models/physics/current_drive.py:1807` (`self.calculate_normalised_current_drive_efficiency(...)`), inside the branch that runs whenever `i_hcd_primary` selects a recognised heating/CD model — i.e. whenever current drive is configured at all, which is this constraint's own precondition |
| `.constraints.eta_cd_norm_hcd_primary_max` | read | explicit-arg | the bound — plain input constant, never computed |

**proposed signature**:
```python
def constraint_37(eta_cd_norm_hcd_primary, eta_cd_norm_hcd_primary_max):
    return leq(eta_cd_norm_hcd_primary, eta_cd_norm_hcd_primary_max)
```

**hole-in-MDA**: **No.** `eta_cd_norm_hcd_primary`'s producer runs whenever a current-drive method is selected, which any run enabling this constraint necessarily has. Confidence: **high**.

**current closure mechanism**: VMCON-joint.

**candidate iteration variable(s)**: none found by direct name match.

**confidence**: high.

**open questions**: none.

**cottax shape**: bare residual read.

---

### Constraint 39: first wall temperature upper limit
**source**: `process/core/solver/constraints.py:1049-1073`.

**calls**: none besides the guard — reads `data` directly, calls `leq(...)`.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.fwbs.temp_fw_peak` | read | explicit-arg | computed by `process/models/blankets/blanket_library.py:2333` |
| `.fwbs.temp_fw_max` | read | explicit-arg | the bound — plain input constant, applicable per the source docstring only when `i_thermal_electric_conversion > 1` |

**PROCESS-side misuse guard, not reproduced**: `if data.fwbs.temp_fw_peak < 1.0: raise
ProcessValueError(...)` — a proxy check for "this constraint was enabled without a
pulsed plant" (the docstring: "do not use constraint 39 if i_pulsed_plant=0"), not a
domain-validity check on a computed value. `temp_fw_peak` itself is a well-defined
number regardless; the raise protects against a `numerics.icc` misconfiguration this
pure function receives no information about (it only sees its own two operands, not
which constraints are active or `i_pulsed_plant`'s value). Not reproduced, per the
same policy this codebase already applies to `physics_B_composition.py`'s `znfuel`
raise — flagged here, not silently dropped.

**proposed signature**:
```python
def constraint_39(temp_fw_peak, temp_fw_max):
    return leq(temp_fw_peak, temp_fw_max)
```

**hole-in-MDA**: **No.** `temp_fw_peak`'s producer is unconditional within `st_fwbs`/blanket-library runs. Confidence: **high**.

**current closure mechanism**: VMCON-joint.

**candidate iteration variable(s)**: none found by direct name match.

**confidence**: high.

**open questions**: whether the un-reproduced misuse guard should surface anywhere in
the eventual graph (e.g. a `Configuration`-time check that `i_pulsed_plant == 1` before
constraint 39 is even offered) — same open question this codebase already carries for
other un-reproduced raises, not resolved here.

**cottax shape**: bare residual read.

---

### Constraint 40: auxiliary power lower limit
**source**: `process/core/solver/constraints.py:1076-1087`.

**calls**: none — reads `data` directly, calls `geq(...)`.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.current_drive.p_hcd_injected_total_mw` | read | explicit-arg | computed by `process/models/physics/current_drive.py:2265` (tokamak path) and `process/models/stellarator/heating.py:105` (stellarator path) — both unconditional within their respective device-type call sequences |
| `.constraints.p_hcd_injected_min_mw` | read | explicit-arg | the bound — plain input constant, never computed |

**proposed signature**:
```python
def constraint_40(p_hcd_injected_total_mw, p_hcd_injected_min_mw):
    return geq(p_hcd_injected_total_mw, p_hcd_injected_min_mw)
```

**hole-in-MDA**: **No.** Two independent unconditional producers (tokamak and stellarator paths), both already forward-computed before constraints run. Confidence: **high**.

**current closure mechanism**: VMCON-joint.

**candidate iteration variable(s)**: none found by direct name match (the injected-power *fractions*/switches that drive this total are iteration variables individually, but not this aggregate itself).

**confidence**: high.

**open questions**: none.

**cottax shape**: bare residual read.

---

### Constraint 41: plasma current ramp-up time lower limit
**source**: `process/core/solver/constraints.py:1090-1103`.

**calls**: none — reads `data` directly, calls `geq(...)`.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.times.t_plant_pulse_plasma_current_ramp_up` | read | explicit-arg | switch-gated formula (`i_pulsed_plant`/`i_t_current_ramp_up`/`pulsetimings`, `physics.py:463-485`), but every branch of that gate unconditionally sets this field — no branch leaves it unset. Also `.stellarator.initialization.py:44` sets it to `0.0` on the stellarator init path before physics runs, consistent with the same field |
| `.constraints.t_current_ramp_up_min` | read | explicit-arg | the bound — computed once by `process/models/pulse.py:247`, not a raw input, but only *read* here, not re-derived — still bare-residual shape (same reasoning constraint 24's `beta_vol_avg_max` note already establishes for a computed-but-not-recomputed bound) |

**proposed signature**:
```python
def constraint_41(t_plant_pulse_plasma_current_ramp_up, t_current_ramp_up_min):
    return geq(t_plant_pulse_plasma_current_ramp_up, t_current_ramp_up_min)
```

**hole-in-MDA**: **No.** Both operands unconditionally forward-computed (one by branch-complete physics logic, one by `pulse.py`). Confidence: **high**.

**current closure mechanism**: VMCON-joint.

**candidate iteration variable(s)**: **`t_plant_pulse_plasma_current_ramp_up`** itself (ID 65, module `times`, bounds `0.1`-`1.0e3`, `iteration_variables.py:80-84`) — direct, strong match, the constraint's own `value` operand is a free unknown.

**confidence**: high.

**open questions**: none.

**cottax shape**: bare residual read.

---

### Constraint 42: cycle time lower limit
**source**: `process/core/solver/constraints.py:1106-1128`.

**calls**: none besides the guard — reads `data` directly, calls `geq(...)`.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.times.t_plant_pulse_total` | read | explicit-arg | computed unconditionally by `process/models/physics/physics.py:521` (tokamak) and `stellarator/initialization.py:60` (stellarator) |
| `.constraints.t_cycle_min` | read | explicit-arg | the bound — plain input constant, applicable per the source docstring only when `i_pulsed_plant != 0` |

**PROCESS-side misuse guard, not reproduced**: same shape as constraint 39's — `if
data.constraints.t_cycle_min < 1.0: raise ProcessValueError(...)`, a proxy for
`i_pulsed_plant == 0`. Not reproduced, same reasoning as 39 (see that entry).

**proposed signature**:
```python
def constraint_42(t_plant_pulse_total, t_cycle_min):
    return geq(t_plant_pulse_total, t_cycle_min)
```

**hole-in-MDA**: **No.** Both device-type paths unconditionally compute `t_plant_pulse_total`. Confidence: **high**.

**current closure mechanism**: VMCON-joint.

**candidate iteration variable(s)**: none found by direct name match.

**confidence**: high.

**open questions**: same un-reproduced-guard open question as constraint 39.

**cottax shape**: bare residual read.

---

### Constraint 43: average centrepost temperature consistency (TART)
**source**: `process/core/solver/constraints.py:1131-1156`.

**calls**: none — reads `data` directly, calls `eq(...)` (the first equality constraint
ported in this codebase beyond the pilot `eq`-shaped candidates — `eq` itself is added
to this batch's port, not previously needed by 17/24/82/83/91, all of which are
inequalities).

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.itart` | read | **not ported as a parameter** | gates only the `ProcessValueError` misuse guard (see below), not any formula branch — so it carries no information the pure function needs |
| `.tfcoil.i_tf_sup` | read | explicit-arg (switch) | `TFConductorModel`; `WATER_COOLED_COPPER` subtracts `constants.TEMP_ROOM` from both operands before comparing — genuine formula-selecting logic, kept as a static kwarg (reads-set is identical either way, only the additive offset differs, same shape as constraint 17's `istell`) |
| `.tfcoil.temp_cp_average` | read | explicit-arg | computed by `process/core/init.py:711` initially (`= temp_cp_coolant_inlet`) and refined by the TF-coil centrepost model during a TART (`itart != 0`) run — this constraint's own precondition |
| `.tfcoil.tcpav2` | read | explicit-arg | computed by `process/models/tfcoil/base.py:1427`, inside the same centrepost-sizing code path, gated on `itart` the same way |

**PROCESS-side misuse guard, not reproduced**: `if data.physics.itart == 0: raise
ProcessValueError("Do not use constraint 43 if itart=0")`. Same shape and same
"misconfiguration, not invalid value" reasoning as constraints 39/42's guards — not
reproduced. `itart` is consequently **not a parameter** of the ported function at all
(unlike `i_tf_sup`, which the *formula* itself branches on): the guard is the only
place `itart` appears in the source, so once the guard is dropped, `itart` has nothing
left to do here.

**proposed signature**:
```python
def constraint_43(i_tf_sup, temp_cp_average, tcpav2):
    if i_tf_sup == TFConductorModel.WATER_COOLED_COPPER:
        temp_cp_average = temp_cp_average - constants.TEMP_ROOM
        tcpav2 = tcpav2 - constants.TEMP_ROOM
    return eq(temp_cp_average, tcpav2)
```

**hole-in-MDA**: **No, conditional on `itart != 0`** — both operands' producers are
gated inside TART-specific TF-coil sizing code, but that is exactly this constraint's
own precondition (its docstring: "This is a consistency equation (TART)"), not a
missing/unwired relationship. Confidence: **high** on the structural classification,
**not independently verified** that every `itart != 0` configuration reaches both
producer call sites (base.py's centrepost sizing is itself gated by further TF-coil
model switches — not traced exhaustively here, flagged as a slightly lower-confidence
corner of an otherwise-high-confidence entry).

**current closure mechanism**: VMCON-joint.

**candidate iteration variable(s)**: **`temp_cp_average`** (ID 20, module `tfcoil`,
bounds `40.0`-`573.0`, `iteration_variables.py:65`) — direct, strong match, the
constraint's own `value` operand is a free unknown.

**confidence**: high (structural), medium (exhaustiveness of the `itart != 0` producer trace).

**open questions**: whether every `itart != 0` + TF-coil-model combination reaches both
producers, or whether some sub-combination leaves `tcpav2` at a stale/default value —
not exhaustively traced.

**cottax shape**: bare residual read — no `calculate_*` re-derivation inside the
constraint body, `i_tf_sup`'s offset is a static formula selection, not a recomputation
of either operand.

---

### Constraint 44: centrepost temperature upper limit (TART)
**source**: `process/core/solver/constraints.py:1159-1184`.

**calls**: none besides the guard — reads `data` directly, calls `leq(...)`.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.itart` | read | **not ported as a parameter** | same as constraint 43 — gates only the misuse guard |
| `.tfcoil.i_tf_sup` | read | explicit-arg (switch) | same room-temperature-offset logic as constraint 43 |
| `.tfcoil.temp_cp_max` | read | explicit-arg | the bound — plain input constant |
| `.tfcoil.temp_cp_peak` | read | explicit-arg | computed by `process/models/tfcoil/base.py:1435`, same TART-gated code path as constraint 43's operands |

**PROCESS-side misuse guard, not reproduced**: identical shape to constraint 43's
`itart == 0` guard. `itart` not ported as a parameter, same reasoning.

**proposed signature**:
```python
def constraint_44(i_tf_sup, temp_cp_max, temp_cp_peak):
    if i_tf_sup == TFConductorModel.WATER_COOLED_COPPER:
        temp_cp_max = temp_cp_max - constants.TEMP_ROOM
        temp_cp_peak = temp_cp_peak - constants.TEMP_ROOM
    return leq(temp_cp_peak, temp_cp_max)
```

**hole-in-MDA**: **No, conditional on `itart != 0`**, same reasoning and same
not-exhaustively-traced caveat as constraint 43.

**current closure mechanism**: VMCON-joint.

**candidate iteration variable(s)**: none found by direct name match for `temp_cp_peak`
itself (`temp_cp_average`, constraint 43's operand, is the named iteration variable in
this immediate neighbourhood; `temp_cp_peak` is not separately listed).

**confidence**: high (structural), medium (exhaustiveness of the `itart != 0` producer trace, same caveat as 43).

**open questions**: same as constraint 43's.

**cottax shape**: bare residual read.

---

### Constraint 45: edge safety factor lower limit (TART)

**source**: `process/core/solver/constraints.py:1187-1214` (`constraint_manager_45`).

**calls**: no function call inside the constraint body — reads `data` fields directly,
raises `ProcessValueError` if `itart == 0`, then calls `geq(...)`.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.itart` | read | explicit-arg (switch) | spherical tokamak (ST) switch; constraint is only valid when nonzero — static, selects a raise vs. the ordinary formula, per `naming_convention.md` |
| `.physics.q95` | read | explicit-arg | producer ported: `confinement_time.py`'s tokamak-mode `q95` `Output` |
| `.physics.q95_min` | read | explicit-arg | plain input constant, no producer found (bound) |

**proposed signature**: see `batch5_constraints.py`'s `constraint_45`.

**hole-in-MDA**: **No.** Both operands are already available (`q95` ported, `q95_min` a
plain constant). The `itart == 0` raise is reproduced faithfully as a plain `ValueError`
(not `ProcessValueError` — this port's constraints do not import PROCESS's exception
hierarchy, see `constraint_24`'s own precedent), legitimate because `itart` is a static
switch resolved before tracing.

**cottax shape**: bare residual read (`geq`), not `Compare` — neither operand is a
`calculate_*` re-derivation inside the constraint body.

**switches touched**: `.physics.itart` — static, gates a raise, not a formula branch
(unlike `istell` in constraints 17/24, which selects between two formulas).

---

### Constraint 46: I_p / I_rod upper limit (TART)

**source**: `process/core/solver/constraints.py:1217-1243` (`constraint_equation_46`).

**calls**: none — computes `cratmx` inline, calls `leq(...)`.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.itart` | read | explicit-arg (switch) | same TART gate as constraint 45 |
| `.physics.eps` | read | explicit-arg | producer ported: `stellarator_C_geometry.py`/`physics_C_outplas.py` both mint `.physics.eps` as `Output` |
| `.physics.plasma_current` | read | explicit-arg | producer ported: `physics_A_pure_formulas.py`/`physics_C_outplas.py`/`profiles.py` all read it as an established field with real producers upstream |
| `.tfcoil.c_tf_total` | read | explicit-arg | producer ported: `coils/calculate.py`'s `Output` |

**proposed signature**: see `batch5_constraints.py`'s `constraint_46`.

**hole-in-MDA**: **No.** Every operand has a real, already-ported producer.

**cottax shape**: bare residual read (`leq`); `cratmx` is a local scalar formula, not a
node-worthy re-derivation of a separately-tracked quantity.

**switches touched**: `.physics.itart`, same TART gate/raise shape as constraint 45.

---

### Constraint 48: poloidal beta upper limit

**source**: `process/core/solver/constraints.py:1246-1257` (`constraint_equation_48`).

**calls**: none.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.beta_poloidal_vol_avg` | read | explicit-arg | **hole-in-MDA** — real producer is `Physics.calculate_poloidal_beta` (`physics.py:3825`), not yet ported anywhere in `functional_process` |
| `.constraints.beta_poloidal_max` | read | explicit-arg | plain input constant, default `0.19` (`constraint_variables.py:13`) |

**proposed signature**: see `batch5_constraints.py`'s `constraint_48`.

**hole-in-MDA**: **Yes**, on `beta_poloidal_vol_avg`. Ported regardless — same
convention `constraint_91` already established (port the constraint even when an
operand's producer is itself unported/tier-2, the constraint's own correctness doesn't
depend on the producer existing yet).

**cottax shape**: bare residual read (`leq`).

**switches touched**: none.

---

### Constraint 50: IFE repetition rate upper limit — **not ported**

**source**: `process/core/solver/constraints.py:1260-1267` (`constraint_equation_50`).

**data footprint**: `.ife.reprat` (read), `.ife.rrmax` (read, bound). Both are IFE
("inertial fusion energy")-only fields; the docstring itself says "IFE option".

**Excluded, not audited further.** The entire `.ife.*` subsystem has no producer or
consumer anywhere in `functional_process` (`grep -rn "s\.ife\." functional_process`
finds exactly one hit, an unrelated switch read in `costs.py`, not a real IFE port).
This whole port's scope (`CLAUDE.md`) is tokamak/stellarator; IFE is PROCESS's third
whole-device mode and is not addressed anywhere in this codebase — porting one IFE-only
constraint in isolation, with no IFE model behind it, would be untestable (no reference
producer to build a meaningful sample against) and would assert a scope this session was
not asked to open. Matches the "genuinely blocked subsystem" exception in this batch's
own dispatch instructions.

---

### Constraint 51: startup flux equality

**source**: `process/core/solver/constraints.py:1270-1281` (`constraint_equation_51`).

**calls**: none — `abs()` on one operand, then `eq(...)`. First equality constraint
ported in this codebase; see `batch5_constraints.py`'s module docstring for why `eq`
is defined there rather than imported from the (not-yet-updated) canonical module.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.vs_plasma_ramp_required` | read | explicit-arg | **hole-in-MDA** — real producer `Physics.<...>` at `physics.py:4872` (`vs_res_ramp + vs_self_ind_ramp`), not yet ported |
| `.pf_coil.vs_cs_pf_total_ramp` | read | explicit-arg | **hole-in-MDA** — real producer is `pfcoil.py` (`process/models/pfcoil.py:1673`), an entire subsystem with no port anywhere in `functional_process` |

**proposed signature**: see `batch5_constraints.py`'s `constraint_51`.

**hole-in-MDA**: **Yes, on both operands.** Ported regardless (same convention as
constraint 48/91) — the pure function's correctness is independent of whether its
producers exist yet, and this is the harness's own stated purpose (audit-and-port
ahead of full producer coverage, catch the shape early).

**cottax shape**: bare residual read (`eq`), not `Compare`.

**switches touched**: none.

---

### Constraint 52: IFE tritium breeding ratio lower limit — **not ported**

**source**: `process/core/solver/constraints.py:1284-1307` (`constraint_equation_52`).

Same exclusion as constraint 50 — IFE-only (`if data.ife.ife != 1: raise ...`), no IFE
subsystem anywhere in this codebase. Not audited further; see constraint 50's entry
above for the full reasoning, which applies identically here.

---

### Constraint 53: fast neutron fluence on TF coil, upper limit

**source**: `process/core/solver/constraints.py:1310-1321` (`constraint_equation_53`).

**calls**: none.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.fwbs.flu_tf_neutron_fast_peak` | read | explicit-arg | producer **ported**: `stellarator_F_tf_nuclear_heating.py`'s `Output` |
| `.constraints.flu_tf_neutron_fast_max` | read | explicit-arg | plain input constant; already read elsewhere as an `Input` by several `availability.py` nodes, confirming it is a stable, unproduced bound field |

**proposed signature**: see `batch5_constraints.py`'s `constraint_53`.

**hole-in-MDA**: **No.** Both operands available.

**cottax shape**: bare residual read (`leq`).

**switches touched**: none.

---

### Constraint 54: peak TF coil nuclear heating, upper limit

**source**: `process/core/solver/constraints.py:1324-1335` (`constraint_equation_54`).

**calls**: none.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.fwbs.ptfnucpm3` | read | explicit-arg | **hole-in-MDA** — real producer is inline arithmetic in `Stellarator.st_fwbs` (`stellarator.py:455`), not extracted into any ported node. Its own input, `p_tf_nuclear_heat_mw`, comes from `hcpb.py`'s `nuclear_heating_magnets`; that node (`NuclearHeatingMagnets`) is built but **unregistered** (flagged in the earlier session's consolidation-gap audit, not this batch's finding) |
| `.constraints.ptfnucmax` | read | explicit-arg | plain input constant, default `1e-3` (`constraint_variables.py:96`) |

**proposed signature**: see `batch5_constraints.py`'s `constraint_54`.

**hole-in-MDA**: **Yes**, on `ptfnucpm3`. Ported regardless, same convention as 48/51.
Worth flagging for whoever next works the consolidation-gap list: `ptfnucpm3` needs a
small new node (the `p_tf_nuclear_heat_mw / tf_volume` division, `stellarator.py:455`)
in addition to `NuclearHeatingMagnets`'s registration — not just a registration gap,
also a genuine extraction gap, distinct from constraint 53's field which needed neither.

**cottax shape**: bare residual read (`leq`).

**switches touched**: none.

---

### Constraint 56: Pₛₑₚ / R₀ upper limit
**source**: `process/core/solver/constraints.py:1338-1351`.
**calls**: none -- `leq(data.physics.p_plasma_separatrix_rmajor_mw,
data.constraints.p_plasma_separatrix_rmajor_max_mw)` directly.
**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.p_plasma_separatrix_rmajor_mw` | read | explicit-arg | producer: `physics.py:811`, unconditional |
| `.constraints.p_plasma_separatrix_rmajor_max_mw` | read | explicit-arg | plain input constant, no computed producer |
**switches**: none.
**hole-in-MDA**: No. Value operand unconditionally computed by `physics.py:811` before
constraints run; bound is a plain input.

---

### Constraint 59: neutral beam shine-through fraction upper limit
**source**: `process/core/solver/constraints.py:1354-1364`.
**calls**: none.
**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.current_drive.f_p_beam_shine_through` | read | explicit-arg | producer: `current_drive.py:1781`, inside the neutral-beam current-drive branch |
| `.constraints.f_p_beam_shine_through_max` | read | explicit-arg | plain input constant |
**switches**: none in this constraint's own body (the *producer*'s own beam-model
branch is a separate unit's concern, not this constraint's).
**hole-in-MDA**: No, conditional on the neutral-beam current-drive model having run
(same caveat every constraint reading a beam-model output carries; not specific to this
one). No free iteration-variable substitute found.

---

### Constraint 60: Central Solenoid s/c temperature margin lower limit
**source**: `process/core/solver/constraints.py:1368-1378`.
**calls**: none.
**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.pf_coil.temp_cs_superconductor_margin` | read | explicit-arg | producer: `pfcoil.py:3679`, `min(tmarg1, tmarg2)` |
| `.tfcoil.temp_cs_superconductor_margin_min` | read | explicit-arg | plain input constant; `core/init.py:1190` seeds it from `.tfcoil.tmargmin` at initialisation, not recomputed by this constraint's body |
**switches**: none.
**hole-in-MDA**: No. `pfcoil.py`'s margin computation is unconditional whenever the PF
coil model runs.

---

### Constraint 61: plant availability lower limit
**source**: `process/core/solver/constraints.py:1382-1391`.
**calls**: none.
**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.costs.f_t_plant_available` | read | explicit-arg | producer: `availability.py`, three call sites (250/474/1362) depending on `i_plant_availability`/`ibkt_life` branch -- see `availability.md` for that switch's own audit, out of scope here |
| `.costs.f_t_plant_available_min` | read | explicit-arg | plain input constant |
**switches**: none in this constraint's own body (the producer's branch selection is
`availability.py`'s concern).
**hole-in-MDA**: No, conditional on the availability model having run (whichever
branch); no free iteration-variable substitute found.

---

### Constraint 62: alpha-particle / energy confinement time ratio lower limit
**source**: `process/core/solver/constraints.py:1396-1412`.
**calls**: none.
**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.f_t_alpha_energy_confinement` | read | explicit-arg | producer: `physics.py:1592`, `t_alpha_confinement / t_energy_confinement`, unconditional |
| `.constraints.f_t_alpha_energy_confinement_min` | read | explicit-arg | plain input constant |
**switches**: none.
**hole-in-MDA**: No. **Worth flagging**: `.physics.f_t_alpha_energy_confinement` is
also iteration variable 110 (`data_structure/numerics.py:361`, "(62)
f_t_alpha_energy_confinement the ratio of particle to energy confinement times (itv
110)") -- same shape as constraint 91's `te0_ecrh_achievable`/ID 169 pairing already
noted in `constraints.md`. Not resolved here (iteration-variable wiring is a separate
pass per `_audit/next_steps.md` §6), flagged for that pass.

---

### Constraint 63: high-vacuum pump count upper limit (`i_vacuum_pumping = simple`)
**source**: `process/core/solver/constraints.py:1416-1428`.
**calls**: none.
**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.vacuum.n_iter_vacuum_pumps` | read | explicit-arg | producer: `vacuum.py:92`, `vp.n_iter_vacuum_pumps = self.vacuum_simple(...)` -- only under the "simple" vacuum-pumping arm |
| `.tfcoil.n_tf_coils` | read | explicit-arg | plain input/derived field, default 50 for stellarators per docstring |
**switches**: none in this constraint's own body. General constraint (no
`.stellarator.*` read) -- explicitly *not* stellarator-specific, previously excluded
from the stellarator-only audit pass for that reason (see `constraints.md`'s "real
PROCESS finding" note on 63); ported here under this pass's broader default.
**hole-in-MDA**: No in the sense that matters (both operands are real, already-produced
fields when the "simple" vacuum model runs), but note the value operand is only
produced under one vacuum-pumping arm (`i_vacuum_pumping == "simple"`) -- same
conditional-producer caveat as constraint 59/61 above, not a missing-producer hole.

---

### Constraint 64: plasma effective charge (Zeff) upper limit
**source**: `process/core/solver/constraints.py:1432-1443`.
**calls**: none.
**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.n_charge_plasma_effective_vol_avg` | read | explicit-arg | producer: `physics.py:1359` sets a `0.0` default, then `physics_B_composition.py`'s ported `plasma_composition`/`PlasmaComposition` node computes the real value (registry unit #9 chunk B, already ported this session) |
| `.constraints.n_charge_plasma_effective_vol_avg_max` | read | explicit-arg | plain input constant |
**switches**: none.
**hole-in-MDA**: No. Real producer (`plasma_composition`) is unconditional and already
ported.

---

### Constraint 65: vacuum vessel stress on TF coil quench upper limit
**source**: `process/core/solver/constraints.py:1447-1457`.
**calls**: none.
**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.superconducting_tfcoil.vv_stress_quench` | read | explicit-arg | two producers depending on device type: `models/tfcoil/superconducting.py:1424` (tokamak) and `models/stellarator/coils/quench.py:55` (stellarator, `QuenchProtection` -- already ported and registered this session) |
| `.tfcoil.max_vv_stress` | read | explicit-arg | plain input constant |
**switches**: none in this constraint's own body (device-type dispatch happens at the
producer, not here).
**hole-in-MDA**: No. For the stellarator path specifically, the producer
(`QuenchProtection`) is already ported and registered in `total_process.py`.

---

### Constraint 66: upper limit on rate of change of poloidal field energy

**source**: `process/core/solver/constraints.py:1461-1474` (`constrain_equation_66` --
note the source function's own name is missing the "t" in "constraint", not a typo
introduced here).

**calls**: none -- bare residual, both operands plain `.pf_power.*` fields.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.pf_power.peakpoloidalpower` | read | explicit-arg | producer is PF-power-system modelling, not yet audited/ported anywhere in this codebase (`unit_registry.md` has no `pf_power` entry) |
| `.pf_power.maxpoloidalpower` | read | explicit-arg | input constant (bound) |

**proposed signature**: `constraint_66(peakpoloidalpower, maxpoloidalpower)` ->
`leq(peakpoloidalpower, maxpoloidalpower)`.

**hole-in-MDA**: not resolvable either way from this constraint's own audit -- its
producer (`pf_power`) is entirely unaudited elsewhere in this codebase. Not a blocker
for porting the constraint's own pure function (same precedent as constraint 91's
`powerht_constraint`/`powerscaling_constraint`, whose producer was tier-2/unported and
that was explicitly "not this constraint's concern" in the canonical record) -- flagged
here so whoever wires this into a live `Optimise` knows the producer side is still open.

**switches touched**: none.

**cottax node**: bare residual read, no `Compare` needed (see canonical `constraints.py`
module docstring for why).

---

### Constraint 67: simple upper limit on radiation wall load

**source**: `process/core/solver/constraints.py:1477-1487`.

**calls**: none -- bare residual.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.constraints.pflux_fw_rad_max_mw` | read | explicit-arg | **misleadingly named** -- despite "_max" in the name, this is the *peak wall load actually reached*, not a bound; confirmed against the source docstring ("Peak radiation wall load") and the `leq(pflux_fw_rad_max_mw, pflux_fw_rad_max, ...)` call order (first arg is `value`, second is `bound` in `leq`'s own signature). Lives under `.constraints.*`, not `.physics.*` -- unusual, but that is where PROCESS stores it; not investigated further here. |
| `.constraints.pflux_fw_rad_max` | read | explicit-arg | the actual bound (input constant) despite the *lack* of "_max" position confusion — i.e. the names are swapped relative to what a reader would guess from "_max" alone. Worth flagging for anyone wiring this constraint: `pflux_fw_rad_max_mw` is the *value*, `pflux_fw_rad_max` is the *bound*. |

**proposed signature**: `constraint_67(pflux_fw_rad_max_mw, pflux_fw_rad_max)` ->
`leq(pflux_fw_rad_max_mw, pflux_fw_rad_max)`.

**hole-in-MDA**: producer of `.constraints.pflux_fw_rad_max_mw` not traced in this pass
(out of budget for a full upstream trace across ~80 constraints) -- flagged, not
resolved.

**switches touched**: none.

**cottax node**: bare residual read.

---

### Constraint 68: upper limit on Psep scaling (PsepBt / q95*A*R0)

**source**: `process/core/solver/constraints.py:1491-1526`.

**calls**: `PlasmaExhaust.calculate_eu_demo_re_attachment_metric`
(`process/models/physics/exhaust.py:149-183`), only inside the `i_q95_fixed == 1`
branch -- a trivial closed-form arithmetic staticmethod, no `data` access, no
sub-branching. **Not yet ported as its own node** in
`functional_process/models/physics/exhaust.py` (that file ports other `PlasmaExhaust`
methods, not this one) -- inlined directly into `constraint_68`'s pure function rather
than adding a new node this pass, since it is a two-line formula, not a computation
worth its own port. Flagged: if `calculate_eu_demo_re_attachment_metric` is ever
ported as its own node elsewhere, this branch becomes `Compare`-shaped (re-derivation
compared to a stored bound) and should be revisited to call that node's output instead
of re-deriving the formula locally.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.constraints.i_q95_fixed` | read | explicit-arg, **switch** | static -- selects re-derivation vs. direct read, per `naming_convention.md` |
| `.physics.p_plasma_separatrix_mw`, `.physics.b_plasma_toroidal_on_axis`, `.physics.q95`, `.physics.aspect`, `.physics.rmajor` | read | explicit-arg | only used when `i_q95_fixed == 1` |
| `.constraints.q95_fixed` | read | explicit-arg | fixed `q95` substitute, only used when `i_q95_fixed == 1` |
| `.physics.p_div_bt_q_aspect_rmajor_mw` | read | explicit-arg | already-computed metric, used directly when `i_q95_fixed == 0` (PROCESS default); all five inputs above are already-ported `.physics.*` fields per this session's broader physics port |
| `.constraints.p_div_bt_q_aspect_rmajor_max_mw` | read | explicit-arg | bound, either branch |

**proposed signature**: see `_wip/batch7_constraints.py::constraint_68`.

**hole-in-MDA**: **no**, for the `i_q95_fixed == 0` default branch -- every operand is
an already-computed `.physics.*` field or `.constraints.*` input constant. The
`i_q95_fixed == 1` branch's five re-derivation inputs are likewise already-computed
`.physics.*` fields (`rmajor`/`aspect`/`q95` etc. are core physics outputs, ported
elsewhere in this session's broader physics work) -- no free/unwired iteration-variable
stand-in either way.

**switches touched**: `.constraints.i_q95_fixed` (static, formula-selecting, stable
reads-set is *not* stable across the branch -- the two branches genuinely read different
fields, `i_q95_fixed==1` needs five extra `.physics.*` reads the `==0` branch does not
touch at all).

**cottax node**: bare residual read (`i_q95_fixed == 0` branch); `Compare`-shaped in
principle for the `i_q95_fixed == 1` branch once/if `calculate_eu_demo_re_attachment_metric`
is independently ported (see "calls" above) -- not resolved as such this pass.

---

### Constraint 72: upper limit on Central Solenoid Tresca yield stress

**source**: `process/core/solver/constraints.py:1535-1567`.

**calls**: none -- bare residual, `max()` of two already-computed fields in the
bucked-and-wedged branch.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.tfcoil.i_tf_bucking` | read | explicit-arg, **switch** | static, selects branch |
| `.build.i_tf_inside_cs` | read | explicit-arg, **switch** | static (`TFCSRadialConfiguration` enum), selects branch jointly with `i_tf_bucking` |
| `.pf_coil.stress_shear_cs_peak` | read | explicit-arg | CS stress at max current; used in both branches |
| `.tfcoil.sig_tf_cs_bucked` | read | explicit-arg | CS stress at flux swing from TF inward pressure; only used in the bucked-and-wedged branch |
| `.pf_coil.stress_cs_steel_max` | read | explicit-arg | bound, either branch |

**proposed signature**: see `_wip/batch7_constraints.py::constraint_72`.

**hole-in-MDA**: producer of `.pf_coil.*`/`.tfcoil.sig_tf_cs_bucked` not traced this
pass (`pf_coil` module unaudited elsewhere in this codebase, same caveat as
constraint 66's `pf_power`) -- flagged, not resolved.

**switches touched**: `.tfcoil.i_tf_bucking` and `.build.i_tf_inside_cs` jointly (a
compound condition, both static).

**cottax node**: bare residual read.

---

### Constraint 73: lower limit, separatrix power >= L-H threshold + auxiliary power

**source**: `process/core/solver/constraints.py:1578-1592`. Related to constraint 15
(not audited this pass).

**calls**: none -- bare residual, sum of two fields compared to a third.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.p_plasma_separatrix_mw` | read | explicit-arg | already-computed core physics output |
| `.physics.p_l_h_threshold_mw` | read | explicit-arg | L-H threshold power, already-computed physics output |
| `.current_drive.p_hcd_injected_total_mw` | read | explicit-arg | producer (`current_drive` module) not audited/ported elsewhere in this codebase -- flagged, not resolved, same caveat pattern as constraints 66/72 |

**proposed signature**: `constraint_73(p_plasma_separatrix_mw, p_l_h_threshold_mw,
p_hcd_injected_total_mw)` -> `geq(p_plasma_separatrix_mw, p_l_h_threshold_mw +
p_hcd_injected_total_mw)`.

**hole-in-MDA**: partially open -- two of three operands are already-ported `.physics.*`
outputs; the third (`current_drive.p_hcd_injected_total_mw`) has an unaudited producer.
Not a blocker for porting the constraint itself.

**switches touched**: none.

**cottax node**: bare residual read.

---

### Constraint 74: upper limit on TF coil quench temperature (CroCo HTS only)

**source**: `process/core/solver/constraints.py:1595-1605`. Source docstring: "ONLY used
for croco HTS coil" -- a documented usage precondition, not a switch branch in the
constraint body itself; ported unconditionally, matching the source.

**calls**: none -- bare residual.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.tfcoil.temp_croco_quench` | read | explicit-arg | producer not traced this pass |
| `.tfcoil.temp_croco_quench_max` | read | explicit-arg | input constant (bound) |

**proposed signature**: `constraint_74(temp_croco_quench, temp_croco_quench_max)` ->
`leq(temp_croco_quench, temp_croco_quench_max)`.

**hole-in-MDA**: not resolved -- producer of `.tfcoil.temp_croco_quench` (CroCo-specific
TF coil modelling) not audited elsewhere in this codebase yet.

**switches touched**: none (usage precondition only, not a branch).

**cottax node**: bare residual read.

---

### Constraint 75: upper limit on TF coil current / copper area (CroCo HTS only)

**source**: `process/core/solver/constraints.py:1610-1621`. Same "CroCo HTS only" usage
precondition shape as constraint 74.

**calls**: none -- bare residual.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.rebco.coppera_m2` | read | explicit-arg | producer not traced this pass (`rebco` module unaudited elsewhere) |
| `.superconducting_tfcoil.tf_coppera_m2_max` | read | explicit-arg | input constant (bound); `superconducting_tfcoil` module also unaudited elsewhere |

**proposed signature**: `constraint_75(coppera_m2, tf_coppera_m2_max)` ->
`leq(coppera_m2, tf_coppera_m2_max)`.

**hole-in-MDA**: not resolved -- both producers unaudited.

**switches touched**: none.

**cottax node**: bare residual read.

---

### Constraint 76: upper limit, Eich critical separatrix density model

**source**: `process/core/solver/constraints.py:1625-1656`.

**calls**: none as an external call, but **the source function computes and writes two
intermediates directly onto `DataStructure` from inside the constraint body**
(`data.physics.alpha_crit`, `data.physics.nd_plasma_separatrix_electron_eich_max`) --
and the source itself carries a `# TODO: why on earth are these variables being set
here!? Should they be local?` comment, i.e. this is a PROCESS-acknowledged quirk, not
a hidden one. Ported as ordinary local intermediates (computed, used once, not part of
the port's own return/side effects) -- the `data` writes are not reproduced, same policy
this codebase applies to other unported side effects (see `physics_B_composition.py`'s
precedent).

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.kappa` | read | explicit-arg | already-computed core physics output |
| `.physics.triang` | read | explicit-arg | already-computed core physics output |
| `.physics.aspect` | read | explicit-arg | already-computed core physics output |
| `.physics.p_plasma_separatrix_mw` | read | explicit-arg | already-computed core physics output |
| `.physics.nd_plasma_electron_max_array[6]` | read | explicit-arg | single element (0-indexed `[6]` == Fortran-notation `(7)` in the source docstring, confirmed against the source body's own `[6]` index, not the docstring text alone); producer of this specific density-limit-array entry not traced this pass |
| `.physics.alpha_crit` | write (source only) | **not reproduced -- local-intermediate here** | source writes this to `data`; port computes it as a plain local, per the "real PROCESS quirk" note above |
| `.physics.nd_plasma_separatrix_electron_eich_max` | write (source only) | **not reproduced -- local-intermediate here** | ditto |
| `.physics.nd_plasma_separatrix_electron` | read | explicit-arg | already-computed core physics output (constraint value) |

**proposed signature**: see `_wip/batch7_constraints.py::constraint_76`.

**hole-in-MDA**: mostly no -- five of six real inputs are already-ported `.physics.*`
core outputs; the density-limit array element's specific producer not traced this pass.
The two `data`-write side effects are a real, source-acknowledged quirk (see above), not
a hole in this port's own MDA representation, since nothing downstream of this
constraint is shown to depend on those two written fields being present in `data` (not
independently verified against every other PROCESS call site, out of this audit's
budget).

**switches touched**: none.

**cottax node**: bare residual read (two intermediates computed inline, not owned by
any node -- see the writes note above).

---

### Constraint 77: maximum TF coil current per turn upper limit

**source**: `process/core/solver/constraints.py:1667-1678`. `leq(c_tf_turn,
c_tf_turn_max)`.

**calls**: none -- bare `data` reads, `leq(...)`. Bare-residual-read shape, not
`Compare`.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.tfcoil.c_tf_turn` | read | explicit-arg | produced by `coils/calculate.py`'s ported `WindingPackTotalSizePost` (`c_tf_turn = Output(lambda s: s.tfcoil.c_tf_turn)`, calculate.py:1128) |
| `.tfcoil.c_tf_turn_max` | read | explicit-arg | plain input constant (`tfcoil_variables.py:149`, default `9.0e4`); no producer anywhere in PROCESS |

**proposed signature**:
```python
def constraint_77(c_tf_turn, c_tf_turn_max):
    return leq(c_tf_turn, c_tf_turn_max)
```

**hole-in-MDA**: **No.** `c_tf_turn`'s real producer is already ported (though not yet
registered into `total_process.py` as of this writing, per the in-flight winding-pack
consolidation elsewhere this session -- registration status, not portability, is the
only open item). `c_tf_turn_max` is a genuine input constant.

**switches touched**: none.

---

### Constraint 78: Reinke criterion, divertor impurity fraction lower limit

**source**: `process/core/solver/constraints.py:1681-1691`. `geq(fzactual, fzmin)`.

**calls**: none -- bare `data` reads, `geq(...)`.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.reinke.fzactual` | read | explicit-arg | **no producer anywhere in this port** -- the Reinke divertor-impurity model has not been audited or ported at all |
| `.reinke.fzmin` | read | explicit-arg | same -- no producer |

**proposed signature**:
```python
def constraint_78(fzactual, fzmin):
    return geq(fzactual, fzmin)
```

**hole-in-MDA**: **Yes -- real hole.** Both operands come from the Reinke model
(`process/models/reinke.py` or equivalent -- not located/audited as part of this batch),
which is entirely unported. The constraint arithmetic itself is trivial and ported here
regardless (same "port the residual now, wire the edge once the producer exists" stance
this codebase takes elsewhere), but this constraint cannot be evaluated end-to-end in
the current graph until the Reinke model is audited and ported as its own unit. Flagging
for `unit_registry.md`: the Reinke model is not currently tracked as a registry row at
all, as far as this batch's audit could determine -- worth adding one.

**switches touched**: none.

---

### Constraint 79: maximum central solenoid (CS) field

**source**: `process/core/solver/constraints.py:1695-1715`. `leq(max(
b_cs_peak_flat_top_end, b_cs_peak_pulse_start), b_cs_limit_max)`.

**calls**: none -- bare `data` reads, a plain Python `max(...)` of two already-produced
fields (not a re-derivation, no `calculate_*` call), then `leq(...)`.

**Real PROCESS quirk found**: the constraint's own docstring and its
`@ConstraintManager.register_constraint(79, "A/turn", "<=")` registration both tag the
unit as `"A/turn"` (current per turn) -- copied verbatim from the neighbouring
constraint 77's real unit. The actual quantity being compared is a magnetic field (T),
confirmed by both operand names (`b_cs_*`) and the docstring's own body text ("Central
solenoid max field limit [T]"). Reproduced faithfully as a real PROCESS source quirk
(a stale copy-paste unit tag), not corrected here.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.pf_coil.b_cs_peak_flat_top_end` | read | explicit-arg | **no producer anywhere in this port** -- the PF coil / central solenoid system is entirely unported |
| `.pf_coil.b_cs_peak_pulse_start` | read | explicit-arg | same -- no producer |
| `.pf_coil.b_cs_limit_max` | read | explicit-arg | same -- no producer (also unclear whether this is meant as an input constant or a computed limit in the unported PF coil system; not resolved by this batch) |

**proposed signature**:
```python
def constraint_79(b_cs_peak_flat_top_end, b_cs_peak_pulse_start, b_cs_limit_max):
    return leq(max(b_cs_peak_flat_top_end, b_cs_peak_pulse_start), b_cs_limit_max)
```

**hole-in-MDA**: **Yes -- real hole.** All three operands are `.pf_coil.*` fields with
no producer in this port. Same "port the arithmetic now, wire it once the PF coil
system is ported" stance as constraint 78. `unit_registry.md` should track the PF coil /
central solenoid system as its own (currently missing) registry row.

**switches touched**: none.

---

### Constraint 80: lower limit on power crossing the separatrix

**source**: `process/core/solver/constraints.py:1719-1739`. `geq(
p_plasma_separatrix_mw, p_plasma_separatrix_min_mw)`.

**calls**: none -- bare `data` reads, `geq(...)`.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.p_plasma_separatrix_mw` | read | explicit-arg | produced by `stellarator_B_st_phys.py`'s ported node (`p_plasma_separatrix_mw = Output(...)`, stellarator_B_st_phys.py:502) |
| `.constraints.p_plasma_separatrix_min_mw` | read | explicit-arg | plain input constant (`constraint_variables.py:73`, default `150.0`) |

**proposed signature**:
```python
def constraint_80(p_plasma_separatrix_mw, p_plasma_separatrix_min_mw):
    return geq(p_plasma_separatrix_mw, p_plasma_separatrix_min_mw)
```

**hole-in-MDA**: **No.** Both operands have real producers or are genuine input
constants.

**switches touched**: none.

---

### Constraint 81: lower limit ensuring central density exceeds pedestal density

**source**: `process/core/solver/constraints.py:1743-1758`. `geq(
nd_plasma_electron_on_axis, nd_plasma_pedestal_electron)`.

**calls**: none -- bare `data` reads, `geq(...)`.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.nd_plasma_electron_on_axis` | read | explicit-arg | produced by `physics/profiles.py`'s ported nodes (two producers found, `profiles.py:697` and `:762` -- both port different profile-shape branches of the same field, consistent with this codebase's established convention of one node per switch arm; not a duplicate-ownership conflict since only one arm is ever registered per configuration) |
| `.physics.nd_plasma_pedestal_electron` | read | explicit-arg | produced by `physics/profiles.py`'s ported `PedestalOnAxisDensities`-family node (`profiles.py:1065`) -- this is one of the fields in this session's confirmed genuine cross-node cycle (`DensityProfile → FusionRates → PlasmaComposition → PedestalOnAxisDensities → DensityProfile`, see `_audit/next_steps.md` §5) |

**proposed signature**:
```python
def constraint_81(nd_plasma_electron_on_axis, nd_plasma_pedestal_electron):
    return geq(nd_plasma_electron_on_axis, nd_plasma_pedestal_electron)
```

**hole-in-MDA**: **No.** Both operands have real producers, though `nd_plasma_
pedestal_electron`'s producer sits inside an as-yet-undriven genuine SCC -- that is a
driving concern for the graph as a whole, not a hole specific to this constraint.

**switches touched**: none.

---

### Constraint 82: toroidal consistency of the stellarator build
**source**: `process/core/solver/constraints.py:1761-1771`. Genuinely stellarator-specific
(docstring: "Equation for toroidal consistency of stellarator build") — unlike 17/24,
not a general constraint with an embedded `istell` branch; this one has no switch at all.

**calls**: no function call inside `constraint_equation_82` — reads two `data` fields
directly and calls `geq(...)`. Same no-internal-solve shape as every constraint audited
in this file.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.tfcoil.toroidalgap` | read | explicit-arg | minimal gap between two stellarator coils. Minted by `functional_process/models/stellarator/coils/calculate.py`'s already-ported `CoilCoilToroidalGap` (`calculate_coil_coil_toroidal_gap`, `calculate.py:241-262`), which computes it unconditionally from `stella_config_dmin`/`r_coil_major`/`r_coil_minor`/`stella_config_coil_rmajor`/`stella_config_coil_rminor` — no switch gates its production |
| `.tfcoil.dx_tf_inboard_out_toroidal` | read | explicit-arg | total toroidal width of a TF coil. Minted by the same file's `CoilToroidalThickness` node (`calculate.py:88`), also unconditional |

**proposed signature**:
```python
def constraint_82(
    toroidalgap: float,
    dx_tf_inboard_out_toroidal: float,
) -> tuple[float, float, float, float]:  # (residual, normalised_residual, value, bound)
    return geq(toroidalgap, dx_tf_inboard_out_toroidal)
```

**hole-in-MDA**: **No.** Both operands are unconditionally produced by already-ported
nodes in `coils/calculate.py` (`CoilCoilToroidalGap`, `CoilToroidalThickness`) — traced
both producers directly, neither is gated behind a switch or an unported unit. No missing
producer, no free iteration variable standing in for an unwired relationship. Confidence:
**high**.

**Real PROCESS finding**: none found — this constraint is a plain geometric feasibility
check with no embedded switch, no TODO, no cross-branch override, unlike 17/24.

**current closure mechanism**: VMCON-joint. No local solver, same as every other
constraint in this file.

**candidate iteration variable(s)**: no direct name match in
`iteration_variables.py` for either `toroidalgap` or `dx_tf_inboard_out_toroidal`. Not
investigated further — no strong best-effort candidate found on a quick pass, unlike
91/24's direct matches.

**confidence**: high (structural classification; no physics-correctness ambiguity found,
unlike 17/24's open `istell` questions).

**open questions**: none.

**cottax shape**: **bare residual read** — both operands are plain already-produced node
outputs, neither is recomputed inside the constraint body itself, so there is nothing for
`Compare` to wrap (same reasoning as 17/24/91).

---

### Constraint 83: radial consistency of the stellarator build
**source**: `process/core/solver/constraints.py:1775-1789`. Genuinely stellarator-specific
(docstring: "Equation for radial consistency of stellarator build"; both operand
docstring lines explicitly reference "s.-configuration"). No switch.

**calls**: no function call inside `constraint_equation_83` — reads two `data` fields
directly and calls `geq(...)`.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.build.available_radial_space` | read | explicit-arg | minted by `functional_process/models/stellarator/build.py`'s already-ported `Build` node (`calculate_build`, `build.py:182-186` in the PROCESS source), computed unconditionally from `.stellarator.r_coil_minor`/`f_coil_shape`/`.physics.rminor`/`.stellarator.f_st_rmajor`/`stellarator_config` fields — no switch gates it |
| `.build.required_radial_space` | read | explicit-arg | minted by the same `Build` node (`build.py:172-179`), also unconditional — a sum of several `.build.dr_*` thicknesses |

**proposed signature**:
```python
def constraint_83(
    available_radial_space: float,
    required_radial_space: float,
) -> tuple[float, float, float, float]:  # (residual, normalised_residual, value, bound)
    return geq(available_radial_space, required_radial_space)
```

**hole-in-MDA**: **No.** Both operands are unconditionally produced by the already-ported
`Build` node in `functional_process/models/stellarator/build.py` — traced both producers
directly. No missing producer, no free iteration variable standing in for an unwired
relationship. Confidence: **high**.

**Real PROCESS finding**: none found.

**current closure mechanism**: VMCON-joint. No local solver.

**candidate iteration variable(s)**: no direct name match in `iteration_variables.py`
for either operand. Not investigated further, same as 82.

**confidence**: high (structural classification; no physics-correctness ambiguity found).

**open questions**: none.

**cottax shape**: **bare residual read** — same reasoning as 82/17/24/91, no
`calculate_*` re-derivation inside the constraint body.

---

### Constraint 84: lower limit of plasma beta

**source**: `process/core/solver/constraints.py:1791-1801`. `geq(beta_total_vol_avg,
beta_vol_avg_min)`.

**calls**: none -- bare `data` reads, `geq(...)`.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.beta_total_vol_avg` | read | explicit-arg | produced by `stellarator_B_st_phys.py`'s ported node (same field already used by canonical constraint 24) |
| `.physics.beta_vol_avg_min` | read | explicit-arg | plain input constant (`physics_variables.py:530`, default `0.0`) |

**proposed signature**:
```python
def constraint_84(beta_total_vol_avg, beta_vol_avg_min):
    return geq(beta_total_vol_avg, beta_vol_avg_min)
```

**hole-in-MDA**: **No.**

**switches touched**: none.

---

### Constraint 85: equality constraint for centrepost (CP) lifetime

**source**: `process/core/solver/constraints.py:1805-1832`. `eq(cplife, bound)`, `bound`
selected by `i_cp_lifetime` from one of four fields.

**calls**: none -- bare `data` reads and an `if`/`elif` selection, then `eq(...)`.
Bare-residual-read shape (the selected `bound` is itself a plain already-produced/input
field in every branch, never a `calculate_*` re-derivation).

**First equality constraint ported in this codebase** -- `eq` did not exist in the
canonical `constraints.py` before this batch (only `leq`/`geq`, since none of 17/24/82/
83/91 is an equality constraint). Ported here (`_wip/batch8_constraints.py`'s `eq`),
matching `process/core/solver/constraints.py:204-213`'s formula exactly (`residual =
value - bound`, `normalised_residual = 1.0 - (value / bound)` -- note this is `leq`'s
residual formula paired with `geq`'s normalised-residual formula, not a new pattern,
confirmed by reading the source directly rather than assumed).

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.costs.i_cp_lifetime` | read | explicit-arg (switch) | selects the bound; formula-changing, so static per `naming_convention.md`, not a `VarPath` |
| `.costs.cplife` | read | explicit-arg | produced by `availability.py`'s ported `CplifeAvail`/`Avail`-family node (`availability.py:1072`) |
| `.costs.cplife_input` | read | explicit-arg | plain input constant (`cost_variables.py:345`, default `2.0`); used only when `i_cp_lifetime == 0` |
| `.costs.life_div_fpy` | read | explicit-arg | produced by `availability.py`'s ported node (`availability.py:1412`/`:1482`); used only when `i_cp_lifetime == 1` |
| `.fwbs.life_blkt_fpy` | read | explicit-arg | produced by `availability.py`'s ported node (`availability.py:1411`/`:1481`); used only when `i_cp_lifetime == 2` |
| `.costs.life_plant` | read | explicit-arg | plain input constant (`cost_variables.py:557`, default `30.0`); used only when `i_cp_lifetime == 3` |

**proposed signature**:
```python
def constraint_85(
    i_cp_lifetime, cplife, cplife_input, life_div_fpy, life_blkt_fpy, life_plant
):
    if i_cp_lifetime == 0:
        bound = cplife_input
    elif i_cp_lifetime == 1:
        bound = life_div_fpy
    elif i_cp_lifetime == 2:
        bound = life_blkt_fpy
    elif i_cp_lifetime == 3:
        bound = life_plant
    return eq(cplife, bound)
```

**hole-in-MDA**: **No.** Every branch's bound is either a real producer (already
ported) or a genuine input constant.

**switches touched**: `.costs.i_cp_lifetime` (4-way, static, selects the bound field --
same "formula-changing switch with per-branch reads-set differing only in which field is
read" shape as constraint 24's `i_beta_component`).

---

### Constraint 86: upper limit on TF winding-pack turn edge length

**source**: `process/core/solver/constraints.py:1838-1848`. `leq(dx_tf_turn_general,
t_turn_tf_max)`.

**calls**: none -- bare `data` reads, `leq(...)`.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.tfcoil.dx_tf_turn_general` | read | explicit-arg | genuine input field (`tfcoil_variables.py:101`, default `0.0`), already read as a plain `Input` by `coils/calculate.py`'s ported `WindingPackIntersectInputs`/`WindingPackTotalSizePost` -- not itself computed by any model in this port |
| `.tfcoil.t_turn_tf_max` | read | explicit-arg | plain input constant (`tfcoil_variables.py:113`, default `0.05`) |

**proposed signature**:
```python
def constraint_86(dx_tf_turn_general, t_turn_tf_max):
    return leq(dx_tf_turn_general, t_turn_tf_max)
```

**hole-in-MDA**: **No.** Both operands are genuine input fields/constants, already
established as such by this codebase's existing winding-pack port.

**switches touched**: none.

---

### Constraint 87: TF coil cryogenic power upper limit
**source**: `process/core/solver/constraints.py:1852-1863`. General constraint, no
switch.

**calls**: no function call inside `constraint_equation_87` -- reads two `data` fields
directly and calls `leq(...)`.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.heat_transport.p_cryo_plant_electric_mw` | read | explicit-arg | cryogenic plant electric power. Real producer: `functional_process/models/power_B_thermal_cryo.py`'s `CryoLoads` node (already ported, harness-tested; **not yet registered** in `total_process.py` -- confirmed by cross-checking the current registration audit, same "ported but unregistered" status as several other units this session found). PROCESS's own source (`process/models/power.py:1050,1076`) initialises it unconditionally to `0.0` then overwrites it -- always available by the time constraints run. |
| `.heat_transport.p_cryo_plant_electric_max_mw` | read | explicit-arg | plain input constant (`heat_transport_variables.py:18`, default `50.0`); grepped for any producer -- none found (only a `process/core/scan.py` scan-variable entry, which is user-input addressing, not a model output) |

**proposed signature**:
```python
def constraint_87(
    p_cryo_plant_electric_mw: float,
    p_cryo_plant_electric_max_mw: float,
) -> tuple[float, float, float, float]:
    return leq(p_cryo_plant_electric_mw, p_cryo_plant_electric_max_mw)
```

**hole-in-MDA**: **No.** `p_cryo_plant_electric_mw`'s producer is ported (`CryoLoads`,
unregistered but real); the bound is a plain input constant, not computed at all. No
free iteration variable, no unwired relationship. Confidence: **high**.

**Real PROCESS finding**: none -- plain threshold check, no embedded switch, no TODO.

**current closure mechanism**: VMCON-joint, same as every other constraint here.

**candidate iteration variable(s)**: no direct name match found in
`iteration_variables.py` for either operand. Not investigated further.

**confidence**: high.

**open questions**: none.

**cottax shape**: **bare residual read** -- both operands are already-produced (or
plain-input) values, neither is recomputed inside the constraint body.

---

### Constraint 88: TF coil vertical strain upper limit
**source**: `process/core/solver/constraints.py:1866-1877`. General constraint, no
switch. Compares `abs(str_wp)` (source takes the absolute value explicitly -- `str_wp`
is signed, compressive strain is negative) against a positive bound.

**calls**: no function call inside `constraint_equation_88` -- reads two `data` fields,
takes `abs()`, calls `leq(...)`.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.tfcoil.str_wp` | read | explicit-arg | TF coil winding-pack vertical strain, signed. Real producer: `process/models/tfcoil/superconducting.py`'s self-consistent winding-pack strain calculation -- **not ported anywhere in `functional_process` at present** (grepped the whole tree, zero hits; the entire `tfcoil`/`pfcoil` engineering module is out of scope so far). Does not block porting this constraint itself -- same precedent constraint 91 already set (`.stellarator.powerht_constraint`, from an unported unit) |
| `.tfcoil.str_wp_max` | read | explicit-arg | plain input constant (`tfcoil_variables.py:497`, default). Source comment there: *"You can't have constraint 88 and i_str_wp = 0 at the same time"* -- a real input-validity precondition on `.tfcoil.i_str_wp`, not something this constraint function itself checks or needs to; noted for whoever eventually wires the TF coil switch layer |

**proposed signature**:
```python
def constraint_88(
    str_wp: float,
    str_wp_max: float,
) -> tuple[float, float, float, float]:
    return leq(abs(str_wp), str_wp_max)
```

**hole-in-MDA**: **Yes, but not blocking.** Neither operand has a producer ported in
this codebase yet (the entire TF coil engineering module is unported) -- there is
currently no way to *drive* this constraint end to end in the assembled graph. The
constraint function itself is still fully portable and independently testable directly
against PROCESS's own `constraint_equation_88` (which is exactly what
`TestConstraint88` below does), so it is ported now rather than deferred -- matches
`next_steps.md` §6's framing that constraint bodies and their producers are separate
porting concerns. Confidence: **high** that the hole exists; revisit once TF coil
engineering is in scope.

**Real PROCESS finding**: none beyond the `i_str_wp`/constraint-88 precondition comment
noted above (an input-validity note, not a constraint-body bug).

**current closure mechanism**: VMCON-joint.

**candidate iteration variable(s)**: not investigated -- `str_wp` is not obviously one
of `ITERATION_VARIABLES`'s entries on a quick name-match pass; would need a deeper check
once the TF coil module is audited.

**confidence**: high (structural classification straightforward; the hole-in-MDA is a
scope statement, not an ambiguity).

**open questions**: revisit once `tfcoil/superconducting.py` is audited, to confirm
`str_wp`'s real producer and whether it is gated by any switch this constraint should
also know about (e.g. `i_str_wp`).

**cottax shape**: **bare residual read** -- `str_wp`/`str_wp_max` are plain fields, no
`calculate_*` re-derivation inside the constraint body itself.

---

### Constraint 89: CS coil current / copper area upper limit
**source**: `process/core/solver/constraints.py:1880-1892`. General constraint, no
switch.

**calls**: no function call inside `constraint_equation_89` -- reads two `data` fields,
calls `leq(...)`.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.rebco.copperaoh_m2` | read | explicit-arg | CS coil current at end-of-flattop / copper area. Real producer: `process/models/pfcoil.py`'s REBCO CS current-density calculation -- **not ported** in `functional_process` (grepped, zero hits; PF coil module entirely out of scope so far). Same "portable now, producer later" situation as constraint 88 |
| `.rebco.copperaoh_m2_max` | read | explicit-arg | plain input constant (`rebco_variables.py:43`, default `1.0e8`) |

**proposed signature**:
```python
def constraint_89(
    copperaoh_m2: float,
    copperaoh_m2_max: float,
) -> tuple[float, float, float, float]:
    return leq(copperaoh_m2, copperaoh_m2_max)
```

**hole-in-MDA**: **Yes, but not blocking** -- identical reasoning to constraint 88: PF
coil / REBCO CS module is entirely unported, so this constraint can't be *driven* in the
assembled graph yet, but it is fully portable and testable against PROCESS directly.
Confidence: **high**.

**Real PROCESS finding**: none.

**current closure mechanism**: VMCON-joint.

**candidate iteration variable(s)**: not investigated.

**confidence**: high.

**open questions**: revisit once `process/models/pfcoil.py`'s REBCO CS module is
audited.

**cottax shape**: **bare residual read**.

---

### Constraint 90: CS coil stress load cycles lower limit
**source**: `process/core/solver/constraints.py:1895-1909`. General constraint. **Not**
an `istell`-style formula switch -- a genuine data-mutating side effect inside the
constraint evaluation, a new pattern not seen in 17/24/82/83/91.

**calls**: no function call inside `constraint_equation_90` -- but the function body
**writes to `data` before reading it back**:
```python
if data.costs.ibkt_life == 1 and data.cs_fatigue.bkt_life_csf == 1:
    data.cs_fatigue.n_cycle_min = data.costs.bktcycles
return geq(data.cs_fatigue.n_cycle, data.cs_fatigue.n_cycle_min, constraint_registration)
```

**Real PROCESS finding**: `ConstraintManager.evaluate_constraint` (and hence every
solver iteration that evaluates constraint 90) has a documented, real side effect --
evaluating this constraint can overwrite `.cs_fatigue.n_cycle_min` in shared `data`,
which any other reader of that field later in the same pass would then see. Constraint
*evaluation* mutating model state that outlives the call is a genuinely different shape
from anything else audited in this file so far (17/24's `istell` branches only change
*which formula* runs, never write back to `data`). This port cannot and does not
reproduce the global write (a pure function has nothing to mutate) -- it instead applies
the override to the *local value used in its own comparison* only, which reproduces
this constraint's own residual exactly, but does **not** reproduce the fact that
`.cs_fatigue.n_cycle_min` would be left mutated for any later reader in a real PROCESS
run. Flagged here rather than silently narrowed; see open questions.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.cs_fatigue.n_cycle` | read | explicit-arg | producer: `process/models/pfcoil.py` (unported, same module as constraint 89) |
| `.cs_fatigue.n_cycle_min` | read, conditionally overwritten in `data` by this constraint's own evaluation | explicit-arg (port), **real side effect not reproducible in a pure function** | default `2.0e4` (`cs_fatigue_variables.py:16`) if the override switches are off |
| `.costs.ibkt_life` | read | explicit-arg (static switch) | `cost_variables.py:416`, default `0`. Also referenced by `total_process.py`'s `Avail(ibkt_life=0, ...)` registration -- default `0` means the override branch is off by default on the currently-registered graph |
| `.cs_fatigue.bkt_life_csf` | read | explicit-arg (static switch) | `cs_fatigue_variables.py:31`, "Switch to pass bkt_life cycles to n_cycle_min", default `0.0` |
| `.costs.bktcycles` | read | explicit-arg | `cost_variables.py:424`, default `1.0e3`. Producer: `functional_process/models/availability.py` (already ported; need to confirm it mints `.costs.bktcycles` specifically -- not independently re-verified this pass, flagged for the consolidation reviewer) |

**proposed signature**:
```python
def constraint_90(
    n_cycle: float,
    n_cycle_min: float,
    ibkt_life: int,  # static
    bkt_life_csf: float,  # static (used as a 0/1 switch)
    bktcycles: float,
) -> tuple[float, float, float, float]:
    if ibkt_life == 1 and bkt_life_csf == 1:
        n_cycle_min = bktcycles
    return geq(n_cycle, n_cycle_min)
```

**hole-in-MDA**: **Yes, on two counts.** (1) `n_cycle`/`n_cycle_min`'s real producer
(`pfcoil.py`/CS fatigue module) is unported, same as 88/89. (2) The side-effect write
means this constraint's true behaviour in a multi-constraint PROCESS evaluation pass
depends on evaluation *order* relative to whatever else reads `.cs_fatigue.n_cycle_min`
-- an ordering dependency this port cannot model at the single-constraint level. Neither
blocks porting the constraint's own residual computation, which is what's done here; (2)
should be revisited once/if `n_cycle_min` gets a real producer node of its own in this
codebase, since at that point "who owns `.cs_fatigue.n_cycle_min`" becomes a genuine
architectural question (this constraint conditionally *writes* it in PROCESS, which
`cottax` would need to represent explicitly -- e.g. this constraint might need to be a
`Compare`-adjacent node that also *owns* the overridden value under some switch state,
not a bare residual read at all, once a producer exists to conflict with).

**Real PROCESS finding**: see above -- constraint evaluation mutating shared state.

**current closure mechanism**: VMCON-joint (for the `geq` residual); the `data` mutation
is outside VMCON's own model, an ordinary Python side effect of calling the constraint
function.

**candidate iteration variable(s)**: not investigated.

**confidence**: medium -- the residual-value reproduction is high confidence (verified
against PROCESS's own function via the harness test below), but the *architectural*
question raised by the side effect (open questions) is genuinely unresolved, not just
unexplored.

**open questions**:
1. If/when `.cs_fatigue.n_cycle_min` gets a real ported producer, does this constraint
   need to become something that can *override* that producer's output under
   `ibkt_life == 1 and bkt_life_csf == 1`, rather than merely reading it? That's a
   `cottax` node-ownership question this record flags but does not answer.
2. Is `.costs.bktcycles`'s producer in `availability.py` confirmed unconditional? Not
   independently re-verified this pass.

**cottax shape**: **bare residual read**, with the caveat in open question 1 above --
this classification may not survive once `n_cycle_min` has a real producer to conflict
with.

---

### Constraint 91: ECRH ignition heating-power lower limit
**source**: `process/core/solver/constraints.py:1912-1938`, docstring: "Lower limit to
ensure ECRH te is greater than required te for ignition at lower values for n and B... 
stellarators only (but in principle usable also for tokamaks)."

**calls**: no direct function call inside `constraint_equation_91` itself — it only
reads `data` fields and calls `eq_geq`'s `geq(...)`. But its two operands
(`data.stellarator.powerht_constraint`, `data.stellarator.powerscaling_constraint`) are
**not free values** — they are written by exactly one producer:
`power_at_ignition_point(stellarator, max_gyrotron_frequency, te0_ecrh_achievable)`
(`process/models/stellarator/density_limits.py:155-217`), called unconditionally from
`Stellarator.run()` at `process/models/stellarator/stellarator.py:178-187`. **That
producer's own footprint is out of scope for this record** — it's model-unit #3
(`density_limits.py`), audited in parallel; see
`functional_process/_audit/units/models/stellarator/density_limits.md`. Flagging here
because it
matters for the hole-in-MDA judgment below: **worth noting now, before that record
lands**, `power_at_ignition_point` deep-copies the entire `stellarator` model object
(`copy.deepcopy(stellarator)`, `density_limits.py:185`) and calls `st_phys()` on the copy
**twice** ("The second call seems to be necessary for all values to 'converge' (and is
sufficient)" — an unverified, hard-coded fixed-point-by-two-iterations, no convergence
check) to compute a counterfactual operating point. This is a `blocker`-severity
JAX-difficulty pattern (whole-state deepcopy + repeated full physics re-evaluation as a
counterfactual sub-computation) that the density_limits.py audit needs to address
directly; recorded here too since it's load-bearing for this constraint's true reads-set.

**data footprint** (direct reads of `constraint_equation_91` only):
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.i_plasma_ignited` | read | explicit-arg | selects which of two `value` expressions to use — see traceability_policy.md, this is a formula-shaped branch on an enum switch, not an implicit-io concern |
| `.current_drive.p_hcd_primary_extra_heat_mw` | read | explicit-arg | only used in the `NON_IGNITED` branch |
| `.stellarator.powerht_constraint` | read | explicit-arg (at this call site) | **but see note above — this value's own production is implicit/heavy**; treat this constraint's signature as taking `powerht_constraint`/`powerscaling_constraint` as plain float args, and treat the deepcopy/double-`st_phys` machinery as a separate node's problem, not this constraint's |
| `.stellarator.powerscaling_constraint` | read | explicit-arg (at this call site) | same as above |

**proposed signature**:
```python
def constraint_91(
    i_plasma_ignited: int,  # PlasmaIgnitionModel, static
    p_hcd_primary_extra_heat_mw: float,
    powerht_constraint: float,
    powerscaling_constraint: float,
) -> ConstraintResult:
    value = powerht_constraint + (
        p_hcd_primary_extra_heat_mw
        if PlasmaIgnitionModel(i_plasma_ignited) == PlasmaIgnitionModel.NON_IGNITED
        else 0.0
    )
    return geq(value, powerscaling_constraint, ...)
```

**hole-in-MDA**: **No — not a hole.** Both operands are already fully forward-computed
(by `power_at_ignition_point`, called unconditionally every `Stellarator.run()`, not
gated behind whether constraint 91 is even active in `icc` — the code comment at
`stellarator.py:179` ("If the respective constraint equation is not called, do not set
the values") suggests someone *intended* to skip this when the constraint isn't active,
but the call at line 183 is unconditional, so the comment doesn't match current
behaviour; not filing this as a MDA-hole, filing it as a separate note below). There is
no missing producer and no free iteration variable standing in for an unwired
relationship — this is an ordinary physical feasibility inequality (is the ECRH-driven
heating power at a hypothetical operating point at least the required scaling/loss
power?), structurally a plain `Compare`/condition node reading two already-produced
quantities. Confidence: **medium-high** on the structural classification (I traced the
producer definitively); **medium** on whether this is the physically "right" constraint
formulation, which is beyond what a code read alone can confirm.

**Separate note (not a hole-in-MDA finding, but worth carrying forward)**: the mismatch
between the comment at `stellarator.py:179` ("if the respective constraint equation is
not called, do not set the values") and the unconditional call at line 183 means
`power_at_ignition_point` — with its expensive deepcopy + double-`st_phys` — runs on
**every** stellarator evaluation regardless of whether constraint 91 is active. Whether
that's a genuine bug or the comment is just stale is outside this record's scope, but a
pure-functional port has an opportunity here: since the port makes `Graph.prune(wanted)`
explicit, this node would naturally only be included when constraint 91's condition is
actually wanted — cottax's own `prune` was designed to sidestep exactly this "always
compute in case it's needed" problem.

**current closure mechanism**: VMCON-joint. No local solver — confirmed by reading the
constraint function; it does nothing but read `data` and return a residual, same as every
other `ConstraintManager`-registered function. (The apparent "internal solve" is inside
`power_at_ignition_point`'s producer, not this constraint — see note above; that's a
different question, "how is `powerht_constraint` itself computed," not "how is this
constraint closed.")

**candidate iteration variable(s)**: **`te0_ecrh_achievable`** (ID 169, module
`stellarator`, bounds 1.0–1.0e3, `process/core/solver/iteration_variables.py:233`) —
stronger than usual best-effort evidence for this one: it's literally the
`te0_available` argument passed into `power_at_ignition_point`, which is the sole
producer of `powerht_constraint`, one of this constraint's two operands. Raising
`te0_ecrh_achievable` raises `powerht_constraint` (higher achievable temperature → more
achievable heating power), which is exactly the direction that satisfies this `>=`
constraint — a plausible, evidenced pairing, though still not authoritative (VMCON solves
jointly; nothing in the code declares this pairing explicitly). Iteration variable 176
(`f_st_coil_aspect`) is unrelated to this constraint.

**cottax shape**: **bare residual read**, not `Compare` — both operands are taken as
plain float arguments (per the note above, their own production is out of scope for this
record), and the "comparison" *is* the whole function body: a `value` formula (a static,
`i_plasma_ignited`-gated sum) fed into `geq(...)`. `.stellarator.powerht_constraint`/
`.stellarator.powerscaling_constraint` are not the output of a `calculate_*` re-derivation
happening inside the constraint itself (contrast constraint 1), so there is nothing here
for `Compare` to wrap — same shape as constraints 17 and 24 below.

---

### Constraint 92: D/T/He3 fuel fraction consistency
**source**: `process/core/solver/constraints.py:1941-1955`. General constraint, no
switch. **First equality constraint audited in this file** -- `eq` (not `leq`/`geq`) is
ported for the first time in `batch9_constraints.py`, to be merged into the canonical
`constraints.py` alongside `leq`/`geq` during consolidation.

**calls**: no function call inside `constraint_equation_92` -- sums three `data` fields
and calls `eq(...)`.

**data footprint**:
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.f_plasma_fuel_deuterium` | read | explicit-arg | plain user-input fuel fraction (`physics_variables.py`) -- already an ordinary boundary `Input` of `functional_process/models/physics/physics_B_composition.py`'s `PlasmaComposition` node, confirming it is a leaf input, not computed by any model |
| `.physics.f_plasma_fuel_tritium` | read | explicit-arg | same as above |
| `.physics.f_plasma_fuel_helium3` | read | explicit-arg | same as above |

**proposed signature**:
```python
def constraint_92(
    f_plasma_fuel_deuterium: float,
    f_plasma_fuel_tritium: float,
    f_plasma_fuel_helium3: float,
) -> tuple[float, float, float, float]:
    return eq(
        f_plasma_fuel_deuterium + f_plasma_fuel_tritium + f_plasma_fuel_helium3,
        1.0,
    )
```

**hole-in-MDA**: **No.** All three operands are plain user inputs, not computed by any
model -- there is nothing to trace a producer for; `PlasmaComposition` already reads all
three as ordinary `Input`s, confirming their leaf status independently. Confidence:
**high**.

**Real PROCESS finding**: none -- a straightforward input-consistency check.

**current closure mechanism**: VMCON-joint (equality residual driven to zero).

**candidate iteration variable(s)**: `f_plasma_fuel_helium3` is a plausible candidate --
not independently confirmed against `iteration_variables.py` this pass, flagged for a
follow-up check.

**confidence**: high.

**open questions**: none.

**cottax shape**: **bare residual read** -- an equality over three leaf inputs, no
`calculate_*` re-derivation.

---

## Constraints considered and excluded from this file's scope

**Constraints 50 and 52** (IFE repetition rate, IFE tritium breeding ratio -- both
gated on `data.ife.ife`, `constraints.py:1260-1307`): **not ported.** The entire
`.ife.*` subsystem has no producer or consumer anywhere in `functional_process`
(`grep -rn "s\.ife\." functional_process` finds exactly one hit, an unrelated switch
read in `costs.py`, not a real IFE port). This whole port's scope (`CLAUDE.md`) is
tokamak/stellarator; IFE is PROCESS's third whole-device mode and is not addressed
anywhere in this codebase -- porting either constraint in isolation, with no IFE model
behind it, would be untestable (no reference producer to build a meaningful sample
against) and would assert a scope this session was not asked to open.

**Constraint 63** (`n_iter_vacuum_pumps <= n_tf_coils`, gated on
`i_vacuum_pumping == "simple"`, `constraints.py:1416-1429`): **update -- now ported**,
superseding the note this section originally carried. The original note recorded that
its docstring mentions `n_tf_coils` "default = 50 for stellarators" while the
constraint body itself reads no `.stellarator.*` field and has no `istell` branch --
a general constraint that merely happens to be stellarator-relevant in practice, not
stellarator-specific in the sense 17/24/82/83/91 are. That structural classification
is still accurate and is why it was excluded under the earlier stellarator-only scope;
it just no longer means "not ported" now that this file's scope has broadened to the
general constraint set. See its entry above for the full port.

## Port status

**Updated -- broadened from five stellarator-specific constraints to
80 constraints, the full general set.** Covers every constraint PROCESS
registers (`@ConstraintManager.register_constraint`, ~82 of them) except 50 and 52
(IFE-only, see "Constraints considered and excluded" above) -- 80 ported,
2 excluded, matching PROCESS's total exactly.

**Shape**: the overwhelming majority are **bare residual read**: a plain pure function
per constraint, tested tier-1, no `ExplicitFunction`/`Compare` node written -- none has
a `VarPath` to own (a constraint residual is not stored anywhere in `DataStructure`;
`evaluate_constraint` returns it straight to the solver) and none recomputes a value via
a `calculate_*` call that a stored field is then compared against (the pattern that
would earn `Compare`). **Constraint 1 is the sole exception found**: it calls a
re-derived `calculate_plasma_beta` and compares it to a stored field -- genuinely
`Compare`-shaped, the first (and so far only) instance of that pattern in this file.
Ported in `constraints.py` alongside this record, with `leq`/`geq`/`eq` (PROCESS's own
closure helpers, minus the `ConstraintRegistration` metadata) shared between all of
them -- `eq` was added once constraint 1 needed it. Harness cases in
`test_constraints.py`, all `Tier1Contract` (explicit pure functions, no internal
iteration, no callee besides `calculate_plasma_beta` for constraint 1).

**Sweep method** (stellarator-relevance pass, earlier this session): every one of the
~82 `@ConstraintManager.register_constraint`-decorated functions in
`process/core/solver/constraints.py` was checked two ways -- docstring text for
"stellarator" mentions, and constraint *body* for `data.stellarator.*` field reads (the
second catches what the first would miss: 82/83 never say "istell" and 17/24's
docstrings don't flag their `istell` branch either, so neither check alone is
reliable). Result: exactly five constraints are stellarator-*specific* by either test
(17, 24, 82, 83, 91) -- still true, unaffected by the later broadening.

**General-constraint pass** (this update): every remaining registered constraint not
already covered was audited and ported, following the same discipline (source location,
data footprint with hole-in-MDA tracing, proposed signature, cottax shape
classification) -- see each constraint's own entry above for real findings. Notable
ones, not exhaustive (see individual entries for the rest):

- Constraint 1: never active on any real stellarator run -- `Stellarator.run()`
  overwrites `.physics.beta_total_vol_avg` directly and raises if it is used as an
  iteration variable when `istell > 0`, with a source comment saying this literally
  replaces constraint 1.
- Constraint 7: shares the same conditional-producer gap already flagged for
  `beta_beam` -- `beam_fusion` is unported, and is only called when beam current is
  nonzero and non-ignited.
- Constraints 31 and 33: structurally vacuous for stellarators -- `caller.py:272-275`
  never calls the tokamak TF-coil model when `istell != 0`, so their operands are
  permanently stuck at `DataStructure` defaults on any real stellarator run. Not a
  porting gap; a structural fact about PROCESS's own call graph.
- Constraint 54: real extraction gap, not just a registration gap -- its producer is
  inline arithmetic inside `Stellarator.st_fwbs`, never pulled into its own node.
- Constraint 67: a real naming trap -- `.constraints.pflux_fw_rad_max_mw` (with `_mw`)
  is the *value*, `.constraints.pflux_fw_rad_max` (without `_mw`) is the *bound*,
  inverted from what the names alone suggest.
- Constraint 76: PROCESS's own source carries a `# TODO: why on earth are these
  variables being set here!?` comment about a stray write; not reproduced as a write
  here, kept as ordinary local intermediates.
- Constraint 79: a real stale-copy-paste unit-tag bug (`"A/turn"`, copied from
  neighbouring constraint 77) -- the actual quantity is a magnetic field (T).
  Reproduced faithfully, not corrected.
- Constraint 90: a real PROCESS side-effect write (`.cs_fatigue.n_cycle_min`, under
  certain switch states) that a pure function cannot reproduce -- handled by applying
  the override to the local comparison value only; flagged as an open node-ownership
  question if that field ever gets a real producer.

`_audit/unit_registry.md`'s Constraints table lists every constraint with accurate
status.

Not done here: registering any of these into `total_process.py` or deciding which
constraints are graph-`Optimise`-active (explicitly out of this task's scope, a later
assembly-time decision per `_audit/next_steps.md` §6) -- and several individual
constraints' own hole-in-MDA producers remain unported (see each entry's own note),
same "ported ahead of full producer coverage" convention constraint 91 already
established.
