---
kind: model-unit
status: reviewed
confidence: high
---

**Ported.** `buildings.py` / `test_buildings.py`, three tier-1 functions (the whole
numerical content of `Buildings.run()`), `Tier1Contract` tests passing (legacy points
lifted from `tests/unit/models/test_buildings.py`, plus fuzz).

## source

`process/models/buildings.py` (whole file, 1537 lines, registry unit #15). `Buildings`
has three methods: `run` (entry point, dispatches on a switch), `bldgs` (the
`ITER_1992`/legacy building-size model), `bldgs_sizes` (the `CHAPMAN_2024` model).
`output()` just calls `run(output=True)` — no separate code path.

`run()`'s own body (lines 36-95) is a short unconditional preamble (TF coil envelope
geometry) followed by an `if BuildingsModel(i_bldgs_size) == CHAPMAN_2024: ... else:
...` dispatch to exactly one of the two methods below. Both methods are single
straight-line functions (no loops, no calls into any other `Model`), each ending in an
`if output:` block that only formats and prints already-computed locals — no further
computation happens there, confirmed by an exhaustive read (same "reporting shell"
shape as `stellarator_G_output.md`/`build.md`'s `output()`).

Split into three port units below (`TfCoilEnvelope`, `Bldgs`, `BldgsSizes`) rather than
one record per PROCESS-level method boundary issue — unlike `stellarator.py`, `run()`'s
own boundaries line up exactly with clean pure-function boundaries, so no line-range
chunking was needed.

## data footprint

### `run()`'s preamble → `calculate_tf_coil_envelope`

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.build.r_tf_outboard_mid`, `.build.dr_tf_outboard` | read | explicit-arg | outboard TF edge |
| `.build.r_tf_inboard_mid`, `.build.dr_tf_inboard` | read | explicit-arg | inboard TF edge |
| `.build.z_tf_inside_half` | read | explicit-arg | feeds `tf_vertical_dim` |
| `.tfcoil.m_tf_coils_total`, `.tfcoil.n_tf_coils` | read | explicit-arg | feeds `tfmtn` (per-coil mass), `bldgs`-only |

No writes — `tfro`/`tfri`/`tf_radial_dim`/`tf_vertical_dim`/`tfmtn` are all locals in
`run()`, never stored to `data`; they are simply arguments to whichever of `bldgs`/
`bldgs_sizes` gets called next. Ported anyway as its own node (mechanical, feeds both
switch arms, no reason to inline it twice).

### `bldgs()` (ITER_1992 branch)

| VarPath | read/write | classification | note |
|---|---|---|---|
| `pfr`, `pfm`, `tfro`, `tfri`, `tfh`, `tfm`, `n_tf_coils`, `shro`, `shri`, `shh`, `shm`, `crr`, `helpow` | read (plain args) | explicit-arg | passed in by `run()` from `.pf_coil.r_pf_coil_outer_max`/`.pf_coil.m_pf_coil_max`/`TfCoilEnvelope`'s outputs/`.tfcoil.n_tf_coils`/`.build.r_shld_outboard_outer`/`.build.r_shld_inboard_inner`/`shh` (see next row)/`.fwbs.whtshld`/`.fwbs.r_cryostat_inboard`/`.heat_transport.helpow` — see `run()`'s call site; those upstream `VarPath`s are noted here for completeness even though `bldgs` itself only ever sees the already-evaluated plain floats |
| `.build.z_tf_inside_half`, `.build.dz_shld_vv_gap`, `.build.dz_vv_upper`, `.build.dz_vv_lower` | read | explicit-arg | feed `shh` (attached shield height), a genuine *inline expression at the `run()` call site* — `2.0*(z_tf_inside_half - dz_shld_vv_gap) - dz_vv_upper - dz_vv_lower` — never stored to any field. Ported as its own function, `calculate_shield_height`, rather than folded silently into `Bldgs`'s `__call__` |
| `.buildings.rxcl`, `.trcl`, `.row`, `.wgt`, `.shmf`, `.clh2`, `.dz_tf_cryostat`, `.stcl`, `.rbvfac`, `.rbwt`, `.rbrt`, `.fndt`, `.hcwt`, `.hccl`, `.wgt2`, `.mbvfac`, `.wsvfac`, `.tfcbv`, `.pfbldgm3`, `.esbldgm3`, `.pibv`, `.triv`, `.conv`, `.admv`, `.shov` | read | explicit-arg | plant-building input parameters, all straight reads |
| `.buildings.wrbi` | write | explicit-arg | reactor-building half-width, written directly inside `bldgs()` |
| `.buildings.a_plant_floor_effective` | write | explicit-arg | |
| `.buildings.admvol` | write | explicit-arg | `= admv`, a straight copy |
| `.buildings.shovol` | write | explicit-arg | `= shov`, straight copy |
| `.buildings.convol` | write | explicit-arg | `= conv`, straight copy |
| `.buildings.volnucb` | write | explicit-arg | |
| `.buildings.cryvol`, `.volrci`, `.rbvol`, `.rmbvol`, `.wsvol`, `.elevol` | write | explicit-arg | **not written inside `bldgs()` itself** — `bldgs()` *returns* `(cryv, vrci, rbv, rmbv, wsv, elev)` as a plain tuple, and it is `run()` (the caller) that unpacks the return value straight onto these six differently-named fields. Recorded here since `run()` is this unit's own scope too (registry: "`Buildings.run()`"). |

`local-intermediate`, not tabulated: `bmr`, `sectl`, `coill`, `layl`, `hy`, `ang`,
`drbi`, `wt`, `crcl`, `hrbi`, `rbw`, `rbl`, `rbh`, `tcw`, `tcl`, `dcw`, `hcw`, `hcl`,
`rmbw`, `rmbl`, `wgts`, `cran`, `rmbh`, `tch`, `wsa` — all written once, unconditionally,
and read back later in the same straight-line function.

### `bldgs_sizes()` (CHAPMAN_2024 branch)

Real `self.data.buildings.*` writes (everything else below is `local-intermediate`,
consumed only by the `if output:` reporting block, never stored):

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.buildings.reactor_hall_w` | write | explicit-arg | written 2x (initial + NBI-branch addition), then `.reactor_hall_l` gets a 3rd, fc_building addition too — all local-intermediate re-reads of the same field within one straight-line function |
| `.buildings.reactor_hall_l` | write | explicit-arg | see above |
| `.buildings.reactor_hall_h` | write | explicit-arg | written once |
| `.buildings.a_plant_floor_effective` | write | explicit-arg | |
| `.buildings.volnucb` | write | explicit-arg | |

Reads (all `explicit-arg`, all plain floats off `data.buildings.*` unless noted):
`.pf_coil.r_pf_coil_outer_max`, `.fwbs.r_cryostat_inboard`, `tf_radial_dim` (arg from
`run()`, = `TfCoilEnvelope` output), `.buildings.bioshld_thk`, `.reactor_clrnc`,
`.transp_clrnc`, `.crane_clrnc_h`, `.cryostat_clrnc`, `.ground_clrnc`, `.crane_arm_h`,
`tf_vertical_dim` (same), `.current_drive.i_hcd_primary` (**switch**, see below),
`.buildings.nbi_sys_l`, `.nbi_sys_w`, `.hcd_building_l`, `.hcd_building_w`,
`.hcd_building_h`, `.fc_building_l`, `.fc_building_w`, `.reactor_wall_thk`,
`.reactor_roof_thk`, `.reactor_fndtn_thk`, `.costs.life_plant` (**guard**, see below),
`.build.z_tf_inside_half`, `.dr_tf_inboard`, `.dr_tf_shld_gap`, `.dz_shld_thermal`,
`.dz_shld_vv_gap`, `.dr_shld_inboard`, `.dr_blkt_inboard`, `.dr_fw_inboard`,
`.physics.rmajor`, `.rminor`, `.build.dr_fw_plasma_gap_inboard`, `.tfcoil.n_tf_coils`,
`.buildings.hot_sepdist`, `.qnty_sfty_fac`, `.build.dr_fw_outboard`, `.dr_blkt_outboard`,
`.dr_shld_outboard`, `.dr_fw_plasma_gap_outboard`, `.costs.life_div_fpy` (**guard**),
`.divertor.dz_divertor`, `.costs.cplife` (**guard**), `.tfcoil.i_tf_sup` (**switch**,
see below), `.build.r_cp_top`, `.buildings.hotcell_h`, and 24 length/width/height
triples read straight through with no branching: `.chemlab_{l,w,h}`,
`.heat_sink_{l,w,h}`, `.aux_build_{l,w,h}`, `.magnet_trains_{l,w,h}`,
`.magnet_pulse_{l,w,h}`, `.control_buildings_{l,w,h}`, `.warm_shop_{l,w,h}`,
`.workshop_{l,w,h}`, `.robotics_{l,w,h}`, `.maint_cont_{l,w,h}`, `.cryomag_{l,w,h}`,
`.cryostore_{l,w,h}`, `.auxcool_{l,w,h}`, `.elecdist_{l,w,h}`, `.elecload_{l,w,h}`,
`.elecstore_{l,w,h}`, `.turbine_hall_{l,w,h}`, `.ilw_smelter_{l,w,h}`,
`.ilw_storage_{l,w,h}`, `.llw_storage_{l,w,h}`, `.hw_storage_{l,w,h}`,
`.tw_storage_{l,w,h}`, `.gas_buildings_{l,w,h}`, `.water_buildings_{l,w,h}`,
`.sec_buildings_{l,w,h}`, plus `.staff_buildings_area`, `.staff_buildings_h`.

## proposed signature(s)

```python
def calculate_tf_coil_envelope(r_tf_outboard_mid, dr_tf_outboard, r_tf_inboard_mid,
    dr_tf_inboard, z_tf_inside_half, m_tf_coils_total, n_tf_coils) -> (tfro, tfri,
    tf_radial_dim, tf_vertical_dim, tfmtn)

def calculate_shield_height(z_tf_inside_half, dz_shld_vv_gap, dz_vv_upper,
    dz_vv_lower) -> shh

def calculate_bldgs(pfr, pfm, tfro, tfri, tfh, tfm, n_tf_coils, shro, shri, shh, shm,
    crr, helpow, rxcl, trcl, row, wgt, shmf, clh2, dz_tf_cryostat, stcl, rbvfac, rbwt,
    rbrt, fndt, hcwt, hccl, wgt2, mbvfac, wsvfac, tfcbv, pfbldgm3, esbldgm3, pibv, triv,
    conv, admv, shov) -> (cryv, vrci, rbv, rmbv, wsv, elev, wrbi,
    a_plant_floor_effective, admvol, shovol, convol, volnucb)

def calculate_bldgs_sizes(<~90 named args, grouped by section — see `.py` docstring>,
    is_neutral_beam, i_tf_sup) -> (reactor_hall_l, reactor_hall_w, reactor_hall_h,
    a_plant_floor_effective, volnucb)
```

Actual, authoritative signatures are in `buildings.py` (the port is more current than a
duplicate written back into this record, per `build.md`'s precedent).

## cottax node

**Actually written**, in `buildings.py` (`TfCoilEnvelope`, `Bldgs`, `BldgsSizes`, all
`ExplicitFunction`s). Not yet registered in `functional_process/total_process.py` —
registration/switch wiring is reserved for the coordinating session per this dispatch's
instructions.

`TfCoilEnvelope` feeds both `Bldgs` and `BldgsSizes`; which of the latter two is
instantiated is the `i_bldgs_size` topology switch (see below) — a graph-assembly-time
choice, same resolution mechanism as `build.md`'s `blktmodel`/`ipowerflow`.

`calculate_shield_height` gets no node of its own — it is called inline inside
`Bldgs.__call__` (four raw `From` reads in, one local `shh` out, immediately forwarded to
`calculate_bldgs`), the same treatment `build.py` gives `awall`. Unlike `awall` it is
still a separately-defined, separately-testable pure function (its own `Tier1Contract`
in `test_buildings.py`), since — unlike `awall` — it is a nameable, non-trivial
quantity (`shh`, PROCESS's own vocabulary at the call site) rather than truly anonymous
arithmetic.

`Bldgs`'s six declared outputs corresponding to `bldgs()`'s *return tuple* are minted
onto the field names `run()` itself assigns them to at the call site (`.buildings.cryvol`,
`.volrci`, `.rbvol`, `.rmbvol`, `.wsvol`, `.elevol`) rather than `bldgs()`'s own local
names (`cryv`, `vrci`, ...) — those two naming layers genuinely differ in PROCESS itself
(see data-footprint table), and the node's job is to reproduce what actually lands in
`data`, not `bldgs()`'s internal vocabulary.

`BldgsSizes.i_hcd_primary` is a static field (not a `From` read), matching
`naming_convention.md`'s "switches are not ports": the enum lookup
`CurrentDriveModel(i_hcd_primary).method == CurrentDriveMethodType.NEUTRAL_BEAM` cannot
be traced, so it is resolved once, in Python, at node-construction/call time, and the
resulting Python `bool` (`is_neutral_beam`) is what the underlying pure function takes —
same shape as `density_limits.md`'s `EcrhDensityLimit.i_plasma_pedestal`.

## tier signal

**Tier 1** for all three functions — no internal solve, no calls into another `Model`,
no `scipy`, no data-dependent Python control flow that survives into the traced
function except the two deliberately-kept-static/traced switches noted below.

## switches touched

- `.buildings.i_bldgs_size` (`BuildingsModel` enum) — **topology-changing, split.**
  Selects `bldgs` vs. `bldgs_sizes` at graph-assembly time; the two methods have
  almost entirely disjoint reads-sets and disjoint output-value formulas (they agree
  only on writing `.buildings.a_plant_floor_effective`/`.volnucb`, with completely
  different formulas for each). No `switches.md` entry exists yet for this switch
  (out of this unit's editing boundary; noting here for whoever consolidates).
- `.current_drive.i_hcd_primary` (via `CurrentDriveModel(...).method ==
  CurrentDriveMethodType.NEUTRAL_BEAM`, inside `bldgs_sizes` only) — reads-sets
  genuinely differ (NBI branch reads `nbi_sys_l`/`_w`; non-NBI branch reads
  `hcd_building_l`/`_w`/`_h`), which by `naming_convention.md`'s default calls for a
  split. **Not split here** — kept as a static `bool` field on `BldgsSizes`
  (`is_neutral_beam`, derived from `i_hcd_primary` outside the traced function), a
  deliberate policy deviation with the same shape as `i_plasma_ignited`/
  `i_confinement_time` elsewhere in the registry (a 2-branch difference inside an
  otherwise single ~580-line straight-line function; two full node variants for this
  one branch was judged not worth it, matching prior precedent rather than deciding
  it fresh). Flagged, not resolved.
- `.tfcoil.i_tf_sup` (inside `bldgs_sizes`'s centre-post hot-cell branch only) — a
  trivial 2-way scalar selection (`r_cp_top` vs. `dr_tf_inboard`), reads-sets differ by
  exactly one field each way. **Kept as an ordinary traced argument** (`jnp.where`,
  reading both `r_cp_top` and `dr_tf_inboard` unconditionally — `dr_tf_inboard` is
  already read elsewhere in this same function for the shield/blanket geometry, so this
  costs nothing extra), same treatment `unit_registry.md` already gives `.physics.itart`
  ("keep-static, as an ordinary argument, not even switch-shaped" — here it isn't even
  held static, since `i_tf_sup` is a genuine `InputVariable`, not fixed by device mode).
- `.buildings.i_bldgs_v` (verbose-output flag) — only gates which lines the `if output:`
  block prints; never affects a computed value. Not a computation switch, not ported
  (out of scope, same as `output()` everywhere else in this project).

## calls into other models

None. All three functions read only `data.*`/plain arguments; no call to another
`Model`'s method anywhere in this file.

## JAX-difficulty flags

- **`bldgs`'s `if abs(ang) > 1.0: ang = abs(ang)/ang` (severity: minor, translation
  supplied).** This is exactly `jnp.clip(ang, -1.0, 1.0)` — for `abs(ang) <= 1` the
  clamp is a no-op (matches "leave `ang` unchanged"), for `abs(ang) > 1` it produces
  `sign(ang)` (matches `abs(ang)/ang`). No division needed in the port, so this
  translation is *safer* than the original (no risk of a `0/0` the original's guard
  never actually reaches either, since `abs(0) > 1` is false).
- **`bldgs`'s `if np.isinf(vrci): vrci = 1e10` kludge (severity: minor,
  `needs-lax-cond-or-where`).** Ported as `jnp.where(jnp.isinf(vrci), 1e10, vrci)`.
  Only reachable at extreme/degenerate inputs (double-precision overflow of a product
  of several ~10-1000 m-scale lengths), not expected to matter for realistic operating
  points, but reproduced faithfully rather than dropped.
- **`bldgs`'s `wgt > 1.0`/`wgt2 > 1.0` threshold branches (severity: minor,
  `needs-lax-cond-or-where`).** Plain `jnp.where`, no domain risk (both branches are
  ordinary finite arithmetic, no division introduced).
- **`bldgs_sizes`'s three `life_plant != 0.0` / `life_div_fpy != 0.0` / `cplife != 0.0`
  guards (severity: workaround-known, `needs-lax-cond-or-where` +
  denominator-guard).** Each wraps a division by the same guarded quantity
  (`life_plant/life_plant`, `life_plant/life_div_fpy`, `life_plant/cplife`) — ported
  with the project's now-standard safe-divide pattern (denominator replaced by `1.0`
  wherever it is zero, *then* the outer `jnp.where` zeroes the whole sub-result), same
  shape as `pure_formulas.md`'s `phyaux`/`fast_alpha_beta` guards — needed so
  `jax.jacfwd` does not see a `0/0` on the untaken branch and leak a NaN gradient
  through the selected one.
- **`bldgs_sizes`'s `i_tf_sup != 1` centre-post branch.** `jnp.where`, no domain risk.

## open questions

None outstanding for the port itself. One genuine PROCESS bug found, not fixed (per
this project's standing policy — see `radiation_power.md`'s precedent):

**`bldgs_sizes`'s inboard/outboard shield-blanket-first-wall hot-cell storage volume
divides `life_plant` by itself.** Lines 618-621 and 658-661 of `buildings.py`:

```python
hcomp_req_supply = (
    self.data.tfcoil.n_tf_coils
    * (self.data.costs.life_plant / self.data.costs.life_plant)
) * self.data.buildings.qnty_sfty_fac
```

`life_plant / life_plant` is always exactly `1.0` whenever `life_plant != 0` (the only
case this line executes at all, guarded by the enclosing `if`). By contrast, the
divertor and centre-post storage calculations two sections later correctly divide
`life_plant` by that *component's own* replacement lifetime (`life_div_fpy`, `cplife`
respectively) to compute how many replacement units must be stored over the plant's
life. There is no per-component lifetime field for the inboard/outboard shield-blanket-
first-wall assembly anywhere in `cost_variables.py` (checked: no `life_fw`/`life_blkt`/
similar exists), so this reads as a genuine leftover/copy-paste bug rather than a
deliberate simplification with a variable that just hasn't been wired in yet — as
written, PROCESS's inboard/outboard shield-blanket-wall hot-cell storage requirement is
always exactly `n_tf_coils * qnty_sfty_fac` (times the per-piece volume), never actually
scaled by how often that component needs replacing, unlike every other component this
routine sizes storage for. Reproduced faithfully in the port (`calculate_bldgs_sizes`
divides `life_plant` by a safe-guarded copy of itself, not by a distinct quantity) —
flagged here, not fixed.

## Derivative-safe power laws (`safe_pow` / `safe_sqrt`)

2 fractional power laws in this file have been rewritten from `x ** p` / `jnp.sqrt(x)` to
`models/safe_math.py`'s `safe_pow(x, p)` / `safe_sqrt(x)`.

**Why.** For `0 < p < 1` the function is continuous at `x == 0` and its derivative is
not: `d/dx x**p = p * x**(p-1) -> +inf`. JAX's JVP then returns `inf` along the
direction that perturbs `x` and `nan` (`inf * 0`) along every other, so the *value* is
right everywhere and the *Jacobian row* is poisoned. That is the defect class
`_audit/next_steps.md` §9 records; the most recent instance produced 46 non-finite
Jacobian cells and stalled a cold optimiser start at zero SQP steps, reported by the
solver as "the problem seems to be non-convex".

**Value identity, checked not asserted.** `safe_pow`/`safe_sqrt` dispatch on `x == 0`
and evaluate the identical expression otherwise, so every `x != 0` result is bit-for-bit
what it was, and the `x == 0` result is `0.0 ** p` / `sqrt(0.0)` -- again exactly what
the bare expression returns. Verified two ways: a hex-exact diff of every Tier-1
contract's output over every declared sample plus eight fresh fuzz draws (3655 points,
zero differing bits), and `run_mda_harness.py` unchanged at 492 agreements / 34
disagreements. PROCESS itself does not raise at `x == 0` here -- it is plain Python
`float.__pow__` / `numpy.sqrt`, both of which return `0.0` -- and the reference was
re-evaluated at each boundary point to confirm it returns the port's number.

**What changed is only the derivative at exactly `x == 0`**, which becomes `0` instead
of `inf`/`nan` -- the same convention JAX already uses at `jnp.maximum`'s kink.

`Tier1Contract.test_gradient_finite_at_zero` (`--fp-gradients`) now checks the whole
class automatically: it zeroes each differentiable argument in turn and requires a
finite Jacobian wherever the value is finite.
