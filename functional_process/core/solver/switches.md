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

---

## Second wave — switches found while auditing, owed since the `build.py`/`st_fwbs` wave

The 10 sections above are the ones scoped up front (read directly in `stellarator.py`,
found by grepping `self.data.<area>.i_*`). The five below were each found independently
by two or more unit audits (1E1, 1E2, 1E3, `build.py`, `preset_config.py`) and recorded
here rather than left in those records, per `_audit/next_steps.md` § 3.

### One result that applies to every switch on this page

**No switch in PROCESS is ever an iteration variable or a scan variable.**

    grep -n "\"i_\|'i_" process/core/solver/iteration_variables.py   -> no matches
    grep -n "\"i_\|'i_\|istell" process/core/scan.py                 -> no matches

Neither `ITERATION_VARIABLES` nor `ScanVariable` contains a single switch, under either
the `i_*` convention or the legacy names (`istell`, `blktmodel`, `blkttype`, `ipowerflow`,
`irefprop` — checked individually). So a switch is constant across every evaluation of one
assembled graph, carries no derivative, and can never appear on a graph edge. That is what
licenses `functional_process/configuration.py` to resolve **split** switches at
graph-assembly time and treat them as absent from the graph entirely, rather than as an
integer port some node reads. The finding is general, not per-switch: it is the reason the
"split decision" above has a mechanism to be implemented by at all.

### `data.fwbs.blktmodel`
**sites** (35 total, 11 files): `data_structure/{fwbs,constraint,cost,build}_variables.py`,
`models/build.py`, `models/stellarator/build.py`, `models/availability.py`,
`models/stellarator/stellarator.py`, `models/costs/costs.py`, `core/input.py`,
`core/solver/constraints.py`. **Default `0`** (`fwbs_variables.py:479`).
**per-value reads-set**: 13 branch sites in stellarator scope. The decisive one is
`stellarator/build.py:25` (`blktmodel > 0`), already ported as
`BlktmodelBlanketThickness`: the `> 0` branch **computes** `.build.dr_blkt_inboard`/
`dr_blkt_outboard` from six `blb*` layer thicknesses, while the `== 0` branch does not
write them at all — they stay whatever IN.DAT supplied. Reads-sets are therefore
`{blbuith, blbmith, blbpith, blbuoth, blbmoth, blbpoth, dr_shld_*}` vs. `{}`. Also
selects between water and helium coolant-mass constants (`stellarator.py:1269`, jointly
with `i_blkt_coolant_type`) and gates most of `st_fwbs`'s neutronics.
**split decision**: **split.** Empty vs. six-field reads-set is the clean case, and it is
the same *conditional ownership* shape `build.md` recorded — a field the node owns only
in one arm. Not yet in `TOPOLOGY_SWITCHES`: only the `> 0` arm is ported, so there is no
second arm to choose between yet (the `== 0` arm is "no node", which is representable but
pointless to declare until something depends on it).
**confidence**: high for `build.py`'s site; medium for `st_fwbs`'s, which is entangled in
the unresolved 1E1/1E2/1E3 boundary.

### `data.fwbs.blkttype`
**sites** (4 total): `data_structure/fwbs_variables.py`,
`models/stellarator/stellarator.py`, `core/input.py`. **Default `3`**
(`fwbs_variables.py:494`).
**per-value reads-set**: exactly **one** computational site in the whole codebase —
`stellarator.py:1057`, `if self.data.fwbs.blkttype in {1, 2}:` (liquid breeder, WCLL or
HCLL) with an `else` for solid breeder (HCPB), and that site is itself nested under
`blktmodel == 0`. Values 1/2 are one arm; 3 (the default) takes the `else`.
**split decision**: **split**, but note this is a *three-values-two-arms* switch, the
first on this page — the node-selection key is `blkttype in {1, 2}`, not `blkttype`
itself. `configuration.py`'s `Alternative.value` is one integer per arm, so representing
this needs either three alternatives with two of them sharing declarations, or a
predicate-keyed arm. Not yet forced: the site falls inside the unresolved `st_fwbs`
boundary, so nothing is ported from it. **Flagged as the first known case that the current
`Alternative` shape does not express cleanly.**
**confidence**: high (a 4-site switch with one branch is fully readable).

### `data.heat_transport.ipowerflow`
**sites** (14 total, 5 files): `data_structure/{heat_transport,fwbs}_variables.py`,
`models/stellarator/build.py`, `models/stellarator/stellarator.py`, `core/input.py`.
**Default `1`** (`heat_transport_variables.py:94`).
**per-value reads-set**: 11 branch sites in stellarator scope. The ported one is
`stellarator/build.py:171`: `ipowerflow == 0` computes `.first_wall.a_fw_total` from
`{a_fw_total_unadjusted, fhole}`, while `ipowerflow == 1` computes the same field from
`{a_fw_total_unadjusted, fhole, f_ster_div_single, f_a_fw_outboard_hcd}` — **two extra
reads**, not merely a different constant.
**split decision**: **split**, and this is the strongest-evidenced split on the page.
`.fwbs.f_ster_div_single` is owned by `Divertor`, which itself reads
`.first_wall.a_fw_total` — so the extra read *closes a cycle*: the `ipowerflow == 1`
graph has a genuine two-node SCC (`Divertor` ↔ `AFwTotalWithPowerflow`) and the
`ipowerflow == 0` graph is acyclic. A switch here decides not just which formula runs but
**whether the graph is a DAG**, which no single fused node branching internally could
express. Asserted in `functional_process/test_configuration.py`.
**confidence**: high for `build.py`'s site (ported, both arms tested, both assembled);
medium for the 10 `st_fwbs`/output sites, which are inside the unresolved chunk boundary.
**Live in `TOPOLOGY_SWITCHES`** — both arms ported and selectable.

### `data.fwbs.irefprop`
**sites** (3 total): `data_structure/fwbs_variables.py`,
`models/stellarator/stellarator.py`, `core/input.py`. **Default `1`**
(`fwbs_variables.py:449`).
**per-value reads-set**: one site, `stellarator.py:804`, nested under
`i_blkt_coolant_type == CoolantType.WATER` (this is the `irefprop` that
`i_blkt_coolant_type`'s section above flagged incidentally and deferred). Truthy: sets
`temp_blkt_coolant_out` from `FluidProperties.of("Water", pressure=coolp,
vapor_quality=0) - 20`, a **CoolProp call**. Falsy: the same field from a closed-form
polynomial. **Both arms read only `.fwbs.coolp`** — identical reads-set, differing only
in how the number is produced.
**split decision**: **keep-static is not the point here.** By the reads-set test this is
the cleanest `keep-static` candidate on the page (identical reads, one output), but the
truthy arm is not JAX-traceable at all — it is the CoolProp boundary `CLAUDE.md` lists
under "Not everything is JAX-traceable", reached through a two-level nesting
(`i_blkt_coolant_type` then `irefprop`) that no earlier record connected. So the split
decision is subordinate to the traceability decision: **`irefprop` is the switch that
selects whether this node can be differentiated at all**, and it should be resolved
together with `density_limits.py`'s CoolProp branch and `st_geom`'s `istell == 6` file
I/O, not on its own. Recorded here; still open in `_audit/next_steps.md` § 5.
**confidence**: high on the facts, deliberately undecided on the disposition.

### `data.stellarator.istell` — second role (device-config table selection)
The section above covers `istell` as the master pipeline switch
(tokamak/stellarator/IFE). Chunk 1C and `preset_config.py`'s audit independently found a
**second, structurally different role**: within stellarator mode, `istell` also indexes a
table of five hardcoded machine presets (Helias 5/4/3, W7-X 30/50) copied onto
`StellaratorConfigData` by a reflective `hasattr`/`setattr` loop, with `istell == 6`
reading the config from **file** instead.
**Why this is not the same kind of switch**: the first role is topology-changing (it picks
which pipeline runs). The second is **data-table-shaped** — every value produces the same
computation over a different set of constants, so there is no reads-set difference to
diff and nothing for a split decision to decide. It is not a switch in this page's sense
at all; it is a lookup key that happens to share a variable with one.
**disposition**: neither split nor keep-static. `preset_config.md` recommends replacing
the reflective copy with static, fully-enumerated per-machine config records selected at
graph-assembly time — the same policy question as `initialization.py`'s device-preset
literals and chunk 1D's `fncmass`/`gsmass` constants. Three instances, one decision,
still open (`_audit/next_steps.md` § 2).
**confidence**: high (two independent audits agree on the mechanism).
