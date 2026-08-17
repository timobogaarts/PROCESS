# Switch audit records

One section per switch, schema in `../../_audit/schema.md` § Switch record. Split
decision default is "split" per `_audit/traceability_policy.md` — only mark
`keep-static` with an explicit reads-set proof.

Scope: the 10 switches read directly in `process/models/stellarator/stellarator.py`,
per `_audit/unit_registry.md`.

**Site-list method note**: `grep -rn '\b<switch>\b' process --include="*.py"` gives an
exact file:line list. For switches with a handful of sites the full list is given below;
for the two with 100+ sites (`i_tf_sup`, and large counts generally) only the file-level
breakdown is given plus the total count — reproducing 140 individual lines here isn't
useful, the grep is trivially reproducible. `process/core/io/obsolete_vars.py` and
`process/core/input.py` appear in nearly every switch's file list — these are the
obsolete-variable rename table and the IN.DAT parser respectively, i.e. declaration/parse
sites, not computational branches. Listed for completeness, excluded from reads-set
reasoning.

---
kind: switch
status: draft
confidence: see per-switch note (varies — some got a real reads-set diff from a direct
read of stellarator.py, others only forward the switch to a function outside current
scope and are marked pending on that unit's audit)
---

### `data.stellarator.istell`
**sites** (44 total): `process/main.py`, `process/models/stellarator/stellarator.py`,
`process/models/stellarator/initialization.py`, `process/models/stellarator/preset_config.py`,
`process/core/init.py`, `process/models/physics/confinement_time.py`,
`process/models/physics/plasma_geometry.py`, `process/models/physics/plasma_current.py`,
`process/core/caller.py`, `process/core/input.py`, `process/core/solver/constraints.py`,
`process/models/physics/physics.py`.
**per-value reads-set**: this is the master pipeline selector (`Caller._call_models_once`
branches `if istell != 0: models.stellarator.run(); return` vs the tokamak path) — it
doesn't pick between two formulas with a shared signature, it picks between two nearly
disjoint call graphs. Doesn't fit the split/keep-static framing built for formula
switches: it is already, structurally, the top-level pipeline split. Within
`stellarator.py` itself it's only read once (line 211, passed through) and referenced in
two docstrings (1899, 1924) about an iteration-variable constraint (beta must not be an
`ixc` variable when `istell>0`) — that constraint interaction is worth a note for whoever
audits the objective/iteration-variable crosswalk, not resolved here.
**Correction/addition from the constraint-91 pilot (`constraints.md`)**: not every
`istell` site is "whole different constraint" or "whole different model" — constraints 17
and 24 are *general* constraints (used by both pipelines) with an `istell != 0`
special-case branch embedded inside them, a third shape distinct from a stellarator-only
constraint like 91. So `istell`'s true fan-out includes these embedded-branch sites too,
not just the top-level pipeline dispatch and the two stellarator-only files/constraint —
worth carrying forward when the general (non-stellarator) constraints eventually get
audited.
**split decision**: N/A (already split at the pipeline level — see reasoning above).
**confidence**: high (this one is unambiguous).

### `data.tfcoil.i_tf_sup`
**sites** (140 total, 18 files): `main.py`, `structure.py`, `availability.py`,
`costs/costs.py`, `buildings.py`, `build.py`, `power.py`,
`stellarator/stellarator.py`, `core/init.py`, `core/input.py`,
`tfcoil/superconducting.py`, `core/io/obsolete_vars.py`, `core/caller.py`,
`tfcoil/resistive.py`, `core/solver/constraints.py`, `tfcoil/base.py`,
`blankets/hcpb.py`, `core/io/plot/summary.py`.
**per-value reads-set**: directly readable in `stellarator.py` at two sites — line 1022
(`p_tf_nuclear_heat_mw`: SUPERCONDUCTING branch reads `pnucsi, pnucso, pnucshldi,
pnucshldo`; the else/resistive branch reads nothing, sets a constant `0.0`) and line 1724
(guards a further TF-coil-specific block, not fully read here — out of directive scope).
Reads-sets differ (4 quantities vs. none) even in this small local sample.
**split decision**: split. Confirmed by the local evidence and consistent with the
190-site tokamak-wide fan-out already noted in `../../../CLAUDE.md` — this is the
paradigm case for "must split," not a borderline one.
**confidence**: high for the local evidence; the full picture needs the TF-coil files'
own model-unit audits (not yet scheduled — those are tokamak-side and largely out of
current stellarator scope except where stellarator.py reads their outputs).

### `data.fwbs.i_blkt_coolant_type`
**sites** (22 total): `stellarator/stellarator.py`, `costs/costs.py`, `blankets/hcpb.py`,
`core/io/obsolete_vars.py`, `blankets/blanket_library.py`, `core/input.py`.
**per-value reads-set**: read directly in `stellarator.py` at two computational sites.
Line 803: WATER branch computes `temp_blkt_coolant_out` either via a CoolProp call
(`FluidProperties.of(...)`, gated by a *separate* switch `irefprop`) or a closed-form
polynomial — both paths read only `self.data.fwbs.coolp`; the non-WATER case skips this
block entirely (implicit "else": no assignment shown in the read range). Line 1269-1274:
WATER (or `blktmodel > 0`) multiplies `coolvol` by `806.719`, else by `1.517` — **same
read (`coolvol`), only the constant differs**.
**split decision**: at the second site, this is the closest thing found so far to a
genuine `keep-static` candidate — same input, different constant. But the first site's
CoolProp-vs-polynomial branching (nested under `irefprop`, not this switch) is a real
traceability concern independent of the split question — flag `irefprop` as a switch
worth its own audit row later (out of the current 10, found incidentally). Net: **unsure,
leaning keep-static**, pending the non-stellarator sites (`hcpb.py`, `blanket_library.py`)
which likely carry more divergent logic (this switch's value set is water/helium/two
water sub-types per `CoolantType`, not a plain boolean, so the two-sites-read-here may not
be representative of the whole switch).
**confidence**: medium.

### `data.fwbs.i_thermal_electric_conversion`
**sites** (24 total): `stellarator/stellarator.py`, `blankets/hcpb.py`, `power.py`,
`blankets/dcll.py`, `core/io/obsolete_vars.py`, `blankets/blanket_library.py`,
`core/solver/constraints.py`, `core/input.py`.
**per-value reads-set**: the only site inside `stellarator.py` (line 1591-1601) is
**output-only** — `po.ovarre`/`po.ocmmnt` calls printing a report label, gated further by
`ipowerflow==1 and blktmodel==0`, not a computational branch. No computational impact
found within current scope.
**split decision**: unsure — pending audit of `hcpb.py`/`power.py`/`blanket_library.py`
(not yet scheduled; those sites likely carry the real computational branches this switch
exists for). Do not conclude "keep-static" from an output-only sample.
**confidence**: low (sample is not representative of the switch's real effect).

### `data.physics.i_plasma_ignited`
**sites** (42 total): `stellarator/heating.py`, `power.py`, `stellarator/stellarator.py`,
`physics/confinement_time.py`, `core/io/obsolete_vars.py`, `physics/current_drive.py`,
`core/input.py`, `core/solver/constraints.py`, `physics/physics.py`.
**per-value reads-set**: two shapes found in `stellarator.py`. (a) Line 2006-2009: guards
(together with `p_hcd_beam_injected_total_mw != 0.0`, not a switch) a `beam_fusion(...)`
call reading `beamfus0, betbm0, b_plasma_total, c_beam_total,
nd_plasma_electrons_vol_avg, ...` (further args beyond the read window) when
NON_IGNITED; the IGNITED case skips the block (no beam-fusion reads at all). Clear
reads-set difference. (b) Line 2302: **forwarded as a plain argument** into
`self.physics.confinement.calculate_confinement_time(...)` (not branched here — the real
branch, if any, is inside that function, in unit #10 `physics/confinement_time.py`, not
yet audited).
**split decision**: split, on the strength of (a) alone.
**confidence**: medium — solid for site (a); site (b) needs `confinement_time.py`'s audit
to know whether the forwarded switch changes that function's own reads-set too, so the
overall picture may still deepen.
**Addendum from `heating.md`'s audit**: the `heating.py` site (line 154, inside
`output()`) is reporting-only — selects between two comment strings, no computational
branch. Confirms this file as a genuine site (as already listed above) but adds no new
reads-set evidence; doesn't change the split decision or its confidence.

### `data.physics.i_beta_fast_alpha`
**sites** (7 total): `stellarator/stellarator.py`, `core/io/obsolete_vars.py`,
`core/input.py`, `physics/physics.py`.
**per-value reads-set**: the one site in `stellarator.py` (line 2089) **forwards** the
switch as a plain positional argument into `self.beta.fast_alpha_beta(...)` alongside ten
other physics quantities — not locally branched at all. The actual formula selection (if
any) happens inside `fast_alpha_beta`, in `physics/physics.py` (unit #9, in scope but not
yet audited).
**split decision**: unsure — cannot be determined from this file; entirely pending
`physics.py`'s audit of `PlasmaBeta.fast_alpha_beta`.
**confidence**: low (no local branch to evaluate).

### `data.physics.i_pflux_fw_neutron`
**sites** (6 total): `stellarator/stellarator.py`, `models/fw.py`, `core/input.py`,
`core/io/obsolete_vars.py`.
**per-value reads-set**: two symmetric sites (2095, 2223 — neutron wall load and photon
wall load respectively), each an `if this_switch == 1: ... elif
data.heat_transport.ipowerflow == 0: ...` chain. The `==1` branch reads `ffwal,
p_neutron_total_mw / p_plasma_rad_mw, a_plasma_surface`; the `elif` branch (a *different*
switch, `ipowerflow`, not one of the 10 in scope) reads `fhole` instead of `ffwal`, plus
`a_fw_total` instead of `a_plasma_surface` in the second occurrence — genuinely different
inputs.
**split decision**: split. Note this switch is compound with `ipowerflow` in practice —
the three-way outcome (`==1` / `ipowerflow==0` / neither) means a full split needs to
account for both switches together, not `i_pflux_fw_neutron` alone. Flag `ipowerflow` as
a second switch worth its own row (found incidentally, not audited here).
**confidence**: high for "reads differ"; medium for "this switch alone fully explains the
topology" given the compound condition.

### `data.physics.i_confinement_time`
**sites** (31 total): `stellarator/stellarator.py`, `core/init.py`,
`physics/confinement_time.py`, `core/io/obsolete_vars.py`, `core/input.py`,
`physics/physics.py`.
**per-value reads-set**: the one site in `stellarator.py` (line 2301) **forwards** the
switch as an argument into `calculate_confinement_time(...)` — not locally branched.
Given the name and PROCESS's well-known large family of energy confinement scaling laws,
this is very likely a genuine multi-formula, differing-reads-set switch, but that has to
be confirmed by reading `confinement_time.py` itself (unit #10).
**split decision**: unsure — pending unit #10's audit. Strong prior toward split given
what confinement scaling switches typically look like in PROCESS (cf. `i_density_limit`'s
8 formulas discussed in `../../../CLAUDE.md`), but not asserting split without reading
the actual function.
**confidence**: low (no local branch to evaluate; prior is not evidence).

### `data.fwbs.i_p_coolant_pumping`
**sites** (44 total): `stellarator/stellarator.py`, `power.py`, `blankets/dcll.py`,
`blankets/hcpb.py`, `core/io/obsolete_vars.py`, `blankets/blanket_library.py`,
`core/input.py`.
**per-value reads-set**: fully local branch in `stellarator.py` (901-928, enum
`PumpingPowerModelTypes`). `USER_INPUT` branch: `pass` — reads nothing, the pumping power
fields keep whatever was directly input. `FRACTION_OF_HEAT` branch: reads
`f_p_fw_coolant_pump_total_heat`, nuclear-heat quantities computed earlier in the same
call, `current_drive.p_beam_orbit_loss_mw`, `f_p_blkt_coolant_pump_total_heat`,
`f_p_blkt_multiplication`. A third value raises `ProcessValueError` (stellarator only
supports 0/1, per the docstring at line 510-511 and the code's own error message) — worth
noting for the naming-convention/traceability-policy docs: not every switch value seen
tokamak-side is valid stellarator-side, so a stellarator-scoped node must not silently
accept the full tokamak value range.
**split decision**: split — clean case, empty vs. non-trivial reads-set, plus the
value-range restriction itself is topology-relevant (fewer valid branches in this scope
than in general).
**confidence**: high.

### `data.costs.i_cost_model`
**sites** (5 total): `process/main.py`, `core/io/obsolete_vars.py`, `core/input.py`.
**per-value reads-set**: **not branched inside `stellarator.py` at all** (0 matches) —
`Stellarator.run()` calls `self.costs.run()`/`.output()` generically. The switch is
consumed entirely in `Models.costs` (a property in `main.py`) which resolves `self.costs`
to one of `Costs()` / `Costs2015()` / a user-supplied custom model *before* the pipeline
runs — i.e. PROCESS is already doing the "pick the concrete implementation once, from
outside, before the call graph executes" pattern this whole audit wants generalised. This
is direct supporting evidence for the switch/model-swap design in `../../../CLAUDE.md`.
**split decision**: split (already effectively split by construction — unit #18 in the
registry already lists `costs.py`/`costs_2015.py` as two candidate units for this reason).
**confidence**: high.
