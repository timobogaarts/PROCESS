---
kind: model-unit
status: draft
confidence: high
---

**Ported (partial, by design).** `functional_process/models/build.py` /
`tests/functional_process/models/test_build.py`. Thirteen tier-1 pure functions and
thirteen cottax nodes covering the **minimal closure that produces the six `.tokamak.build`
boundary variables** of `_audit/tokamak_boundary.md`, and nothing else. Not registered —
registration is the consolidation pass's job; see § registration below for exactly what is
asked for.

## source

`process/models/build.py`, 2360 lines, `2306` of them entered on `large_tokamak_eval`
(`_audit/tokamak_call_surface.md`) across **6 entered functions** — the widest
entered-LOC-to-function ratio in the tokamak scope. That ratio is the finding, not a
warning: the six functions are

| function | lines | what it is |
|---|---|---|
| `Build.run` | 48–65 | shell; calls the three below |
| `Build.output` | 40–46 | shell; re-runs both builds with `output=True` |
| `Build.calculate_beam_port_size` | 68–150 | `@staticmethod`, pure — **out of this unit's closure** |
| `Build.calculate_vertical_build` | 152–842 | ~30 lines of arithmetic, ~660 of `po.obuild`/`po.ovarre` |
| `Build.divgeom` | 844–1476 | ~80 lines of arithmetic (862–943), ~530 of reporting |
| `Build.plasma_outboard_edge_toroidal_ripple` | 1477–1636 | `@staticmethod`, pure, ~50 lines |
| `Build.calculate_radial_build` | 1638–2360 | ~330 lines of arithmetic, the rest reporting |

So the 2306 entered lines are overwhelmingly **reporting**, and the closure this unit
needs is a few hundred lines inside four of them. The `if output:` blocks are not ported
and are not intended to be: they write `OUT.DAT`/`MFILE.DAT` and, apart from
`dh_tf_inner_bore` and `dz_tf_plasma_centre_offset` (`:452`, `:765`, `:780`), assign
nothing.

## the extraction seam

Poor, and worth naming as the contrast case to `plasma_geometry.md`'s "unusually clean".
Two of the six are already `@staticmethod`s with plain arguments
(`calculate_beam_port_size`, `plasma_outboard_edge_toroidal_ripple`) and port directly.
The other two are `self.data`-mutating methods that interleave reads, writes,
switch branches and `po.*` calls throughout, with **no `calculate_*` pure core anywhere**.
There is therefore no seam to cut along; every ported function below was cut out by hand,
one assignment or short run of assignments at a time, and each one's `file:line` is in its
docstring.

That also means **no ported function has a same-shaped PROCESS counterpart to diff
against**, except the ripple fit. The test module solves this the way the rest of this
harness does — build a real `DataStructure`, set the fields the sample names, call
PROCESS's real method through it, read the answer back off `data`. Every reference in
`test_build.py` is a real PROCESS call.

## scope: what is ported and what is not

**In scope — the six boundary variables and their closure.**
`_audit/tokamak_boundary.md` § `.tokamak.build` lists

    .build.dr_tf_inboard  .build.dr_tf_outboard  .build.r_shld_inboard_inner
    .build.r_shld_outboard_outer  .build.r_tf_outboard_mid  .build.z_tf_inside_half

read by `.buildings.sizing`, `.buildings.tf_coil_envelope` and `.vacuum.vacuum_old`.

**Out of scope, and why:**

| not ported | reason |
|---|---|
| `calculate_beam_port_size` (`:68-150`) | produces `.current_drive.radius_beam_tangency*`, in `.tokamak.current_drive`'s closure, not this one. Also branches on `g > c` with a hard `0.0` fallback and three `np.isinf` kludges. |
| the central-solenoid radial chain (`:1691-1860`) — `r_tf_inboard_in`, `dr_cs_bore`, `dr_cs_precomp`, `r_tf_inboard_mid/out`, `r_cp_top`, `f_r_cp`, `r_vv_inboard_out`, `r_sh_inboard_*`, `rbld` | none of the six reads any of it. `r_shld_inboard_inner` is built *inwards from the plasma* (`:1873`), not accumulated outwards, so the whole `i_tf_inside_cs` / `i_cs_precomp` / `i_r_cp_top` / `itart` sub-tree is outside the closure. |
| `dz_blkt_upper` (`:1665`), `dz_fw_plasma_gap` (`:1672`) | not read by any of the six. `dz_fw_plasma_gap` is additionally a **self-read** (`max(..., dz_fw_plasma_gap)` under `i_single_null == SINGLE_NULL`) and would need the Shape-B treatment. |
| `dr_tf_inner_bore` (`:1912`, `:1949`) | reads `r_tf_inboard_mid`, i.e. the CS chain above. |
| `z_tf_top`, `dz_tf_upper_lower_midplane` (`:819-842`) | the `i_single_null` arms; not read by any of the six. |
| `.build.ripflag` (`:1919`, `:1961`) | see § what is deliberately not returned. |
| every `if output:` block | reporting. |

## data footprint

Reference run: `tests/regression/input_files/large_tokamak_eval.IN.DAT`. The values in
this section are its converged state, read off a live `process.main.SingleRun` in the
`process_port` env; the same numbers are the `BASELINE` dict at the head of
`test_build.py`, so the record and the tests cannot drift.

### `calculate_vertical_build` → `.build.z_plasma_xpoint_upper/lower` (`:167-172`)

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.rminor` | read | explicit-arg | `2.6666666666666665` |
| `.physics.kappa` | read | explicit-arg | `1.85` |
| `.build.z_plasma_xpoint_upper` | **write** | explicit-arg | `4.933333333333334` |
| `.build.z_plasma_xpoint_lower` | **write** | explicit-arg | same value; the source assumes top-down symmetry and writes the product twice |

### `divgeom` → `.build.dz_xpoint_divertor`, `.build.rspo` (`:862-943`, `:798-801`)

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.itart` | read | explicit-arg (switch) | `0`; `itart == 1` returns `1.75 * rminor` at `:863` and never reaches `rspo` |
| `.physics.rmajor`, `.rminor`, `.kappa`, `.triang` | read | explicit-arg | `triang` is read as both `triu` and `tril` (`:868-869`); only `tril` enters the arithmetic |
| `.build.plsepi`, `.plsepo` | read | explicit-arg | `1.0`, `1.5` |
| `.build.plleni`, `.plleno` | read | explicit-arg | `1.0`, `1.0` |
| `.divertor.betai`, `.betao` | read | explicit-arg | `1.0`, `1.0` |
| `.build.rspo` | **write** | explicit-arg | `7.422282732017246` (`:912`) |
| `.build.dz_xpoint_divertor` | **write** *(input `< 1e-5`)* **or** read as a plain input | **conditional-ownership-by-run-config** | `2.001883830794158`. `:800-801` assigns `divht` only when the input is effectively zero; `large_tokamak_eval` does not set it, so it takes `build_variables.py:326`'s default `0.0` and the assignment fires. Same shape as `models/stellarator/build.md`'s `.build.dr_blkt_inboard`. |

`rplti`/`rplbi`/`rplto`/`rplbo` (`:916-940`) are the plate ends' *radial* coordinates.
They are computed and then read only by the reporting block; `divht` is built from the
four `z*` coordinates. Not computed in the port. **Dead-in-the-run, cosmetic.**

### `calculate_vertical_build` → `.build.z_tf_inside_half` (`:807-816`)

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.build.z_plasma_xpoint_upper` | read | explicit-arg | from the node above |
| `.build.dz_xpoint_divertor` | read | explicit-arg | from `divgeom`, or an input |
| `.divertor.dz_divertor` | read | explicit-arg | `0.62` |
| `.build.dz_shld_lower`, `.dz_vv_lower` | read | explicit-arg | `0.7`, `0.3` |
| `.build.dz_shld_vv_gap`, `.dz_shld_thermal`, `.dr_tf_shld_gap` | read | explicit-arg | `0.163`, `0.05`, `0.05` |
| `.build.z_tf_inside_half` | **write** | explicit-arg | `8.818217164127494` |

The source's own comment at `:805-806` — *"TF coils are assumed to be symmetrical.
Therefore this applies to single and double null cases"* — is why this assignment carries
no `i_single_null` arm even though the two hundred lines above it are nothing but
`i_single_null` arms.

### `calculate_radial_build` → the inboard TF pair (`:1684-1689`, `:1739-1747`)

| VarPath | read/write | classification | note |
|---|---|---|---|
| `140 in .numerics.ixc[0:n_iteration_variables]` | read | explicit-arg (switch) | **false** on the reference run: `ixc = [4, 6]` (`large_tokamak_eval.IN.DAT:39,41`) |
| `.tfcoil.dr_tf_plasma_case`, `.dr_tf_nose_case` | read | explicit-arg | `0.07491064938739048`, `0.2816873221155309` |
| `.build.dr_tf_inboard` | **write** (`140 in ixc`) **or** read as a plain input | **conditional-ownership-by-run-config** | `1.2`, the input at `large_tokamak_eval.IN.DAT:74` |
| `.tfcoil.dr_tf_wp_with_insulation` | read (`140 in ixc`) **or** **write** | **conditional-ownership-by-run-config** | `0.8434020284970785`. The two assignments are exact inverses of one another and exactly one runs per run. |

`process/models/build.py:1743` is the **sole writer of `.tfcoil.dr_tf_wp_with_insulation`
on the tokamak path** — the only other writer anywhere is
`models/stellarator/coils/calculate.py:489`, a different device. `models/tfcoil/**` reads
it in eight places and writes it nowhere (checked by grep over
`resistive.py`/`superconducting.py`/`base.py`), so owning it here creates no dual
ownership with the `.tokamak.cicc_superconducting_tf_coil` slot.

### `calculate_radial_build` → the shield radii (`:1873-1890`)

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.rmajor`, `.rminor` | read | explicit-arg | `8.0`, `2.6666666666666665` |
| `.build.dr_fw_plasma_gap_inboard`, `.dr_fw_plasma_gap_outboard` | read | explicit-arg | `0.25`, `0.25` |
| `.build.dr_fw_inboard`, `.dr_fw_outboard` | read | explicit-arg | `0.018` each; produced by `.tokamak.first_wall` (`fw.py`), not here |
| `.build.dr_blkt_inboard`, `.dr_blkt_outboard` | read | explicit-arg | `0.7`, `1.0`; **inputs** on this run, see § blktmodel below |
| `.build.dr_shld_inboard`, `.dr_shld_outboard` | read | explicit-arg | `0.3`, `0.8` |
| `.build.r_shld_inboard_inner` | **write** | explicit-arg | `4.065333333333334` |
| `.build.r_shld_outboard_outer` | **write** | explicit-arg | `12.734666666666667` |

### `calculate_radial_build` → the outboard leg (`:1893-1977`)

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.tfcoil.i_tf_sup` | read | explicit-arg (switch) | `1` (superconducting) |
| `.build.f_dr_tf_outboard_inboard` | read | explicit-arg | `1.19` — **read only on the `i_tf_sup != 1` arm**, which this run does not take |
| `.build.dr_tf_outboard` | **write** | explicit-arg | `1.2` = `dr_tf_inboard` |
| `.build.dr_shld_blkt_gap`, `.dr_vv_outboard`, `.gapomin` | read | explicit-arg | `0.02`, `0.3`, `0.234` |
| `.build.dr_shld_thermal_outboard`, `.dr_tf_shld_gap` | read | explicit-arg | `0.05`, `0.05` |
| `.build.r_tf_outboard_mid` | **write**, twice (`:1901`, `:1939`) | **redundant-duplicate-write** | `14.978406000060053`. The first value (`13.988666666666669`) is the stacked-up build; the second is the ripple-constrained one. |
| `.build.dr_shld_vv_gap_outboard` | **write** | explicit-arg | `1.223739333393385` |
| `.tfcoil.ripple_b_tf_plasma_edge` | **write**, twice (`:1917`, `:1959`) | **redundant-duplicate-write** | `0.6`. The first write is unconditionally overwritten by the second; only the *second* call's value survives. |
| `.build.ripflag` | **write**, twice | **redundant-duplicate-write** | `0`; not ported, see below |
| `.tfcoil.ripple_b_tf_plasma_edge_max`, `.n_tf_coils` | read | explicit-arg | `0.6`, `16.0` |
| `.tfcoil.dx_tf_wp_primary_toroidal`, `.dx_tf_wp_insulation`, `.dx_tf_wp_insertion_gap` | read | explicit-arg | `1.2533980800120443`, `0.008`, `0.01` |
| `.tfcoil.i_tf_shape` | read | explicit-arg (switch) | `1` (`D_SHAPE`) |
| `.tfcoil.i_tf_wp_geom`, `.superconducting_tfcoil.r_tf_wp_inboard_inner/centre/outer` | read | **dead read on this arm** | see § the four reads that are not edges |

## switches touched

Every one is a topology switch (`_audit/naming_convention.md` § "switches are not
ports"): read once at graph-build time to choose an occupant, never a `VarPath` on a node.

| switch | values | live | decision |
|---|---|---|---|
| `140 in .numerics.ixc` | present / absent | **absent** | split. `DrTfInboardFromWindingPack` / `DrTfWpWithInsulationFromInboardBuild`. Disjoint write-sets, exact inverses. |
| `.tfcoil.i_tf_sup` | `0` copper / `1` SC / `2` aluminium | **1** | split. `DrTfOutboardSuperconducting` and `WpConductorMaxWidthSuperconducting` answer `1`. `0`/`2` UNPORTED. Reads-sets are disjoint on both: the non-SC leg arm reads `.build.f_dr_tf_outboard_inboard`, and the non-SC ripple arm reads `.superconducting_tfcoil.r_tf_wp_inboard_outer` and `.tfcoil.n_tf_coils` where the SC arm reads three `.tfcoil.dx_tf_wp_*` fields. |
| `.tfcoil.i_tf_shape` | `0` auto / `1` D-shape / `2` picture frame | **1** | split. `TfOutboardMidDShape` / `TfOutboardEdgeRipple` answer `1`. `0` and `2` UNPORTED. `2` is a different closed-form formula (`:1585-1590`) reading neither `c1`/`c2` nor the winding pack. `0` takes the same PROCESS branch as `1` but is a distinct switch value and gets its own occupant when someone needs it — not folded in, per the binding policy. |
| `.physics.itart` | `0` / `1` | **0** | split. `DivertorGeometryConventional` answers `0`. `1` is `DivertorGeometrySphericalTokamak` (2026-08-27): `divgeom` returns `1.75 * rminor` and **never writes `.build.rspo`**, so it is a different write-set, not just a different formula. |
| input `.build.dz_xpoint_divertor < 1e-5` | true / false | **true** (`0.0`) | conditional ownership, on *both* `itart` arms. When true, the arm's occupant owns the field; when false it is a plain run input — under `itart == 0` `divgeom` still runs for `.build.rspo` alone (that `rspo`-only arm is UNPORTED), and under `itart == 1` `divgeom` owns nothing at all and the slot's occupant is `None` (2026-08-27). |
| `.fwbs.blktmodel` | `0` / `> 0` | **0** | conditional ownership. Under `> 0`, `:1650-1662` produces `dr_blkt_inboard`/`dr_blkt_outboard`/`dz_shld_upper`; `models/stellarator/build.py::BlktmodelBlanketThickness` already ports that block verbatim (the two source files are line-for-line identical here — `process/models/build.py:1649-1662` vs `process/models/stellarator/build.py:25-33`). Under `0`, all three are run inputs. No new occupant needed. |
| `.physics.i_single_null` | `1` / `2` | **1** | **not a switch of this unit.** It gates `dz_fw_plasma_gap` (`:1670-1679`), `z_tf_top` (`:819-842`) and the reporting tables — none in the closure. `z_tf_inside_half` is explicitly outside it by the source's own comment. |
| `.build.i_tf_inside_cs`, `.i_cs_precomp`, `.i_r_cp_top` | | `0`, `1`, `0` | **not switches of this unit.** They gate the central-solenoid chain, which nothing in the closure reads. |
| `.build.iohcl` | `0` / `1` | **1** | **not read by `build.py` at all.** Recorded because the dispatch asked: `build_variables.py:177` defaults it to `1`, `large_tokamak_eval.IN.DAT` never sets it, and the only thing that sets it to `0` is `models/stellarator/initialization.py:24`. So the tokamak run holds `1`, confirmed by reading `.build.iohcl` off the converged `DataStructure`. Its readers are `power.py:346` and `pfcoil.py:153`, both other slots. |

## the four reads that are not edges

`plasma_outboard_edge_toroidal_ripple`'s `i_tf_sup == 1` arm (`:1551-1572`) computes

```python
r_wp_min = r_tf_wp_inboard_inner
i_tf_wp_geom = SuperconductingTFWPShapeType(i_tf_wp_geom)
if i_tf_wp_geom == RECTANGULAR:
    r_wp_max = r_wp_min
elif i_tf_wp_geom == DOUBLE_RECTANGULAR:
    r_wp_max = r_tf_wp_inboard_centre
elif i_tf_wp_geom == TRAPEZOIDAL:
    r_wp_max = r_tf_wp_inboard_outer
dx_tf_wp_conductor_max = dx_tf_wp_primary_toroidal - 2.0 * (...)
```

and then **never uses `r_wp_min` or `r_wp_max` again**. Only the resistive arm (`:1577`)
consumes `r_wp_max`. So on a superconducting run, `.tfcoil.i_tf_wp_geom` and all three
`.superconducting_tfcoil.r_tf_wp_inboard_*` radii are dead reads: declaring them would be
four invented edges from `.tokamak.cicc_superconducting_tf_coil` into `.tokamak.build`
that the run does not make. The port declares none of them. (They are still set in
`test_build.py`'s `BASELINE`, so the reference executes the real code path.)

This is not a bug in PROCESS — the two arms share a variable name and the dead branch is
harmless there — but it is exactly the kind of thing the union-of-arms reads-set would
have manufactured, and it is the third recorded instance in this port.

## what is deliberately not returned

**`.build.ripflag`.** `plasma_outboard_edge_toroidal_ripple` returns a third value, a
fitted-range applicability flag (`:1626-1634`), which PROCESS turns into a log warning at
`:1984-2021` and nothing else. It is not read by any node in this graph. It is also a step
function of `n_tf_coils` with thresholds at exactly `16` and `20`
(`if (n_tf_coils < 16) or (n_tf_coils > 20): flag = 2`), and `large_tokamak_eval` runs at
**exactly 16** — so PROCESS's own central difference with `epsfcn = 1e-3` perturbs
`n_tf_coils` to `15.984`, flips the flag, and reports a gradient of ~`1250` where a port
that treats the flag as piecewise-constant reports `0`. Returning it would make the
gradient contract fail on a correct port. Excluded, and the reference adapter in
`test_build.py` drops it explicitly rather than silently.

## contradiction with `tokamak_boundary.md`

`_audit/tokamak_boundary.md` § `.tokamak.build` attributes `.build.dr_tf_inboard` to this
slot. **On the run it measures, this model does not produce that variable.** The
assignment at `:1685` is guarded by `140 in ixc`, `large_tokamak_eval.IN.DAT` sets
`ixc = 4` and `ixc = 6` and nothing else, and the converged `DataStructure` holds
`dr_tf_inboard = 1.2`, which is the input at `large_tokamak_eval.IN.DAT:74` unchanged.
What build.py runs instead is the *inverse* assignment at `:1743`, producing
`.tfcoil.dr_tf_wp_with_insulation`.

The boundary file says its attribution is *"mechanical, not curated"* — an `ast` walk over
`Assign` targets — and an `ast` walk cannot see an `ixc` guard, so this is a known
limitation of that method rather than a mistake in it. Consequences for the consolidation
pass:

- `.build.dr_tf_inboard` stays a **boundary input** on `large_tokamak_eval`. The slot's
  count drops from 6 to 5 produced.
- `.tfcoil.dr_tf_wp_with_insulation` becomes a **new** `.tokamak.build` product, read by
  `models/tfcoil/**`. It is not in the boundary file at all, because on the measured graph
  nothing yet reads it.
- `DrTfInboardFromWindingPack` is written anyway, so the `140 in ixc` configuration is
  covered when someone runs one.

Nothing was smoothed over: both arms are ported, both are tested, and the record says
which is live.

## resolves `next_steps.md` §2: `dz_shld_upper` under `blktmodel <= 0`

`models/stellarator/build.md`'s open question 2 asks what `.build.dz_shld_upper` is
"supposed to be, structurally" when `blktmodel <= 0`, and notes that unlike
`dr_blkt_inboard`/`dr_blkt_outboard` there is no symmetric "or it is an external input"
story, because `st_build` never reads it back.

**The tokamak supplies the missing half of the story: it is a plain run input.** On
`large_tokamak_eval` (`blktmodel = 0`) `.build.dz_shld_upper` is never written by any
model and holds `build_variables.py:288`'s default `0.6` at convergence; and unlike on the
stellarator it *is* read back within the same file, at `process/models/build.py:832`
(`z_tf_top`, the `i_single_null == SINGLE_NULL` arm) and at `:276`/`:568` in the reporting
tables. So the shape is identical to `dr_blkt_inboard`'s after all —
`conditional-ownership-by-run-config`, produced by the `blktmodel > 0` block or read as an
input — and the asymmetry the stellarator record saw was an artefact of `st_build` having
no consumer of its own, not of the field being structurally different.

No occupant is needed here: `dz_shld_upper` is not in this unit's closure (nothing among
the six reads it), and under `blktmodel > 0` the block that produces it is already ported
as `models/stellarator/build.py::BlktmodelBlanketThickness`, whose formula is identical.

Recommend `next_steps.md` §2's `build.py` bullet be closed with this evidence.

## deviations from PROCESS

1. **`dr_shld_vv_gap_outboard` is written unconditionally where PROCESS branches**
   (`:1940-1956`). PROCESS assigns the long subtraction when the leg was moved out and the
   literal `gapomin` when it was not. Substituting
   `r_tf_outboard_mid = r_shld_outboard_outer + dr_shld_blkt_gap + dr_vv_outboard +
   gapomin + dr_shld_thermal_outboard + dr_tf_shld_gap + 0.5 dr_tf_outboard` (`:1901-1909`,
   the un-moved value) into the subtraction cancels every term but `gapomin`, so the two
   arms are one expression. Written as one; the only difference is float64 reassociation,
   far inside the tier-1 `rtol` of `1e-12`. Checked against PROCESS at the converged point
   (`1.223739333393385`) and under fuzz by `TestOutboardBuildChain`.

2. **`.build.dz_xpoint_divertor` is recomputed every evaluation; PROCESS latches it.**
   `:800-801` assigns `divht` only while the field is below `1e-5`, so after the first
   pipeline pass PROCESS keeps the *first* pass's value forever, and the converged answer
   is a stale one. `DivertorGeometryConventional` recomputes it. **On this run the two are
   identical**: every input to `calculate_divertor_geometry_conventional` is a run
   constant (`ixc = [4, 6]` is temperature and density; `rmajor`, `kappa`, `triang`,
   `plsep*`, `pllen*`, `beta*` are all inputs), and calling `Build.divgeom` on the
   converged `DataStructure` returns `2.001883830794158`, bit-identical to the stored
   value. On a run where the plasma shape is an iteration variable they would differ and
   **the port would be the correct one**. This is a fourth instance of the
   report-pass/solve-pass family `_audit/test_harness.md` § "A fourth category is owed"
   names, and the first where the stale value is a *latch* rather than a re-run.

3. **Four dead reads dropped** — see § the four reads that are not edges.

4. **`.build.ripflag` not produced** — see § what is deliberately not returned.

5. **The two `numpy` kludges inside the ripple fit are kept, as `jnp` selections.**
   `base <= 1e-6 → 1e-6` (`:1611-1613`) becomes `jnp.maximum(..., 1e-6)`; the
   `np.isinf(r_tf_outboard_midmin)` fallback to `3 (R + a)` (`:1619-1623`) becomes a
   `jnp.where`. Both test traced quantities, so neither can stay a Python branch. Neither
   is reached at any sample point; they are ported for faithfulness, not because they fire.

6. **`rplti`/`rplbi`/`rplto`/`rplbo` not computed** — dead outside the reporting block.

## tier signal

**Tier 1 throughout.** No internal iteration anywhere in the closure: the radial and
vertical builds are straight-line arithmetic, `divgeom` is closed-form trigonometry, and
the ripple fit is a closed-form correlation. The only iterative thing in the neighbourhood
is PROCESS's *outer* Gauss–Seidel over the whole pipeline, which is tier 4's problem.

**No cycle.** Worth stating because `:1916-1956` looks like one: PROCESS calls the ripple
fit, possibly moves `r_tf_outboard_mid`, then calls the fit again. But
`r_tf_outboard_midmin` is a function of `rmajor`, `rminor`, `n_tf_coils` and the winding
pack only — it does **not** depend on `r_tf_outboard_mid`, which enters only `ripple` and
`flag`. So the second call cannot change the leg radius, `r_tf_outboard_mid` is a plain
`maximum` of two independent quantities, and the whole stretch is a DAG. Verified
numerically: the fit called at the converged radius returns
`r_tf_outboard_midmin = 14.978406000060053`, exactly the stored `r_tf_outboard_mid`.

## JAX-difficulty flags

| flag | where |
|---|---|
| `needs-lax-cond-or-where`, `workaround-known` | `divgeom`'s `max(zplti, zplto) - min(zplbo, zplbi)` (`:943`) → `jnp.maximum`/`jnp.minimum`; the leg-move `if` (`:1938`) → `jnp.maximum`; the two ripple kludges (`:1611`, `:1619`) |
| `non-differentiable-output` | `.build.ripflag` — a step function of `n_tf_coils` with a threshold at the value the run uses. Not ported; see above. |
| domain restriction | `divgeom`'s `rci` divides by `(triang - 1)**2` and `rco` by `(triang + 1)**2`, and both feed `arcsin`. `triang = ±1` is a pole and `arcsin`'s derivative diverges at its endpoints. Fuzz bounds keep `triang` in `[0.2, 0.7]`. |
| kink | `calculate_r_tf_outboard_mid`'s `maximum`. At the reference point the two arguments differ by ~1 m (`13.99` vs `14.98`), so no finite-difference step straddles it; fuzz bounds are chosen to keep that true. |

## validation

`tests/functional_process/models/test_build.py`, eleven `Tier1Contract`s (eight at first
writing; `TestTfInboardRadii` added by the 2026-08-27 CS-slice wave,
`TestDivertorGeometrySphericalTokamak` by the 2026-08-27 spherical-tokamak wave, and
`TestTfInboardRadiiNoCsPrecomp` by the 2026-08-27 no-precompression wave), all
references real PROCESS calls:

| contract | reference | covers |
|---|---|---|
| `TestZPlasmaXpoint` | `Build.calculate_vertical_build` | `calculate_z_plasma_xpoint` |
| `TestDivertorGeometryConventional` | `Build.divgeom` | `calculate_divertor_geometry_conventional` |
| `TestDivertorGeometrySphericalTokamak` | `Build.divgeom`, `itart = 1` | `calculate_divertor_geometry_spherical_tokamak` |
| `TestZTfInsideHalf` | `Build.calculate_vertical_build` | `calculate_z_tf_inside_half` |
| `TestDrTfWpWithInsulation` | `Build.calculate_radial_build` | `calculate_dr_tf_wp_with_insulation` |
| `TestDrTfInboardFromWindingPack` | `Build.calculate_radial_build`, `ixc[0] = 140` | `calculate_dr_tf_inboard` |
| `TestTfInboardRadii` | `Build.calculate_radial_build` | `calculate_r_tf_inboard_radii_tf_outside_cs` |
| `TestTfInboardRadiiNoCsPrecomp` | `Build.calculate_radial_build`, `i_cs_precomp = 0` | `calculate_r_tf_inboard_radii_no_cs_precomp` |
| `TestRShldInboardInner` | `Build.calculate_radial_build` | `calculate_r_shld_inboard_inner` |
| `TestOutboardBuildChain` | `Build.calculate_radial_build` | the other six functions, composed |
| `TestRippleSuperconducting` | `Build.plasma_outboard_edge_toroidal_ripple` | the ripple fit and the conductor-width mint |

Two contracts are **composites** rather than one-function diffs, because PROCESS draws no
boundary where the graph does: `dx_tf_wp_conductor_max` is a local inside the ripple fit,
and the outboard leg radius is produced by a stretch of `calculate_radial_build` that
calls the fit twice. Testing those stretches end to end against PROCESS's own boundary is
the only 1:1 comparison available, and it is what licenses the finer node split — the same
trade `models/physics/confinement_time.py::plasma_power_loss_mw` records.

Two reference adapters need a note:

- `_reference_z_tf_inside_half` cannot set `z_plasma_xpoint_upper` on `data` directly
  (`:167` overwrites it), so it feeds it as `rminor = z_plasma_xpoint_upper`,
  `kappa = 1.0`; and it keeps every sample's `dz_xpoint_divertor` above `1e-5` so `:801`
  does not overwrite that one either. Both are the real method's own behaviour, not a
  bypass.
- `_reference_dr_tf_inboard` sets `numerics.ixc[0] = 140` and
  `n_iteration_variables = 1`, which is the only way to reach `:1685`.

`BASELINE` in `test_build.py` is `large_tokamak_eval.IN.DAT` at convergence, written down
as literals rather than re-solved, so the suite costs milliseconds. It is also the
evidence for every "the live value is X" claim above.

**Counts.** `84 passed` with `--fp-gradients` at the default one fuzz sample;
`308 passed` at `--fp-fuzz 8`; `244 passed` at `--fp-fuzz 6 --fp-fuzz-seed 7`. No skips,
no xfails.

**One entry added to `_harness/boundary.py`'s register**, and it is a *third* defect class
for that file rather than another instance of the two it already holds.
`test_gradient_finite_at_zero` zeroes each argument in turn; at `kappa == 0` the two arc
radii in `calculate_divertor_geometry_conventional` lose their `kappa ** 2` terms and
collapse to `rco = 0.5 rminor (1 + triang)`, `rci = 0.5 rminor |triang - 1|`, at which
point both `arcsin` arguments reduce to **exactly `-1`**. The value is a perfectly good
`-pi/2` — PROCESS computes it without complaint — and the tangent is `inf`, because
`arcsin` is genuinely not differentiable at its endpoints. No division and no fractional
power is involved, so neither `safe_pow` nor `safe_sqrt` applies, and there is no guarded
form to write: any repair would be a modelling decision about a zero-elongation plasma.
Registered with that reason, per the assertion's own instruction. The register already
holds one non-division class (`x ** (c log x)`), so this is the second such.

## a latent bug in `mda_harness._cache_key`, found while gathering the baseline

Not this unit's file and not fixed here, but it will bite the next agent who needs a
second machine's converged state, so it is recorded rather than worked around silently.

`converged_data(input_file)` keys its on-disk cache with `_cache_key`, which hashes
**every `*IN.DAT` and `*.json` in the input file's directory**, not the file it was asked
for. `tests/regression/input_files/` holds nine input files including both
`stellarator_helias.IN.DAT` and `large_tokamak_eval.IN.DAT`, so every one of them shares a
single cache key. The first call to `converged_data("…/large_tokamak_eval.IN.DAT")` in
this repo returned the **stellarator's** converged `DataStructure` — `rmajor = 26.69`,
`n_tf_coils = 50`, `ixc = [2, 3, 4, 6, 10, 56, 59, 109]`, and `r_tf_inboard_in = 0.0`
because `calculate_radial_build` had never run — with no error and no warning.

It has never bitten because `converged_data` has only ever been called with
`stellarator_helias.IN.DAT` (grep: `mda_constraint_harness.py:211`, `run_mda_harness.py`,
`sand_harness.py`). It bites the moment a second device is measured, which is now.

The docstring explains the directory-wide hash as a deliberate fix for a different
problem — `SingleRun.__init__` creates `OUT.DAT`/`MFILE.DAT` beside the input before the
key is taken, so hashing the directory wholesale made the key depend on whether anything
had run there. The fix for both is to hash *the named input file* (plus its
`.stella_conf.json` companion, which is named after it), not every file in the directory.
The baseline in `test_build.py` was gathered with `FP_HARNESS_CACHE_DIR` pointed at a
scratch directory, which sidesteps the collision only because one file was measured.

## registration (for the consolidation pass)

Thirteen nodes, all in the `.tokamak.build` slot unless noted:

| node | switch value it answers | owns |
|---|---|---|
| `PlasmaXpointHeights` | — | `.build.z_plasma_xpoint_upper`, `.build.z_plasma_xpoint_lower` |
| `DivertorGeometryConventional` | `itart == 0` **and** input `dz_xpoint_divertor < 1e-5` | `.build.dz_xpoint_divertor`, `.build.rspo` |
| `DivertorGeometrySphericalTokamak` | `itart == 1` **and** input `dz_xpoint_divertor < 1e-5` (2026-08-27) | `.build.dz_xpoint_divertor` only — the early return never reaches the `rspo` write |
| `ZTfInsideHalf` | — | `.build.z_tf_inside_half` |
| `DrTfWpWithInsulationFromInboardBuild` | `140 not in ixc` (**live**) | `.tfcoil.dr_tf_wp_with_insulation` |
| `DrTfInboardFromWindingPack` | `140 in ixc` (not live) | `.build.dr_tf_inboard` |
| `ShldInboardInnerRadius` | — | `.build.r_shld_inboard_inner` |
| `ShldOutboardOuterRadius` | — | `.build.r_shld_outboard_outer` |
| `DrTfOutboardSuperconducting` | `i_tf_sup == 1` | `.build.dr_tf_outboard` |
| `WpConductorMaxWidthSuperconducting` | `i_tf_sup == 1` | `.tfcoil.dx_tf_wp_conductor_max` *(mint)* |
| `TfOutboardMidUnrippled` | — | `.build.r_tf_outboard_mid_unrippled` *(mint)* |
| `TfOutboardMidDShape` | `i_tf_shape == 1` | `.build.r_tf_outboard_mid` |
| `TfOutboardEdgeRipple` | `i_tf_shape == 1` | `.tfcoil.ripple_b_tf_plasma_edge` |
| `ShldVvGapOutboard` | — | `.build.dr_shld_vv_gap_outboard` |

**UNPORTED, for `indat.py`'s `UNPORTED`:**

- `.tfcoil.i_tf_sup` `0` (copper) and `2` (aluminium) — the outboard leg scales by
  `.build.f_dr_tf_outboard_inboard`, and the ripple fit's conductor width comes from
  `.superconducting_tfcoil.r_tf_wp_inboard_outer` and `.tfcoil.n_tf_coils` instead of the
  three `dx_tf_wp_*` fields; two disjoint reads-sets, neither written.
- `.tfcoil.i_tf_shape` `2` (picture frame) — a different closed-form ripple formula
  (`:1585-1590`) reading neither the winding pack nor `c1`/`c2`; not written.
- `.tfcoil.i_tf_shape` `0` (auto-select) — takes the same PROCESS branch as `1` but is a
  distinct switch value with no occupant of its own; not folded into `TfOutboardMidDShape`.
- `.physics.itart` `1` (spherical tokamak) — *written 2026-08-27* as
  `DivertorGeometrySphericalTokamak` (`dz_xpoint_divertor` unset) and the slot's `None`
  arm (`dz_xpoint_divertor` set — `divgeom` owns nothing); no longer UNPORTED.
- `.physics.itart` `0` with input `.build.dz_xpoint_divertor >= 1e-5` — `divgeom` still
  runs for `.build.rspo` alone while `dz_xpoint_divertor` stays an input; that
  `rspo`-only occupant is not written.

**Two mints for `KNOWN_MINT_VALUES`**, both reconstructible from stored PROCESS fields by
an identity read off its own source:

- `.tfcoil.dx_tf_wp_conductor_max` =
  `dx_tf_wp_primary_toroidal - 2 (dx_tf_wp_insulation + dx_tf_wp_insertion_gap)`
  (`:1570-1572`); `1.2173980800120443` on the reference run.
- `.build.r_tf_outboard_mid_unrippled` = the `:1901-1909` stack;
  `13.988666666666669` on the reference run. Exists because PROCESS overwrites
  `.build.r_tf_outboard_mid` in place, so a node that owns the final value cannot also
  read the provisional one.

**Registry row** — this unit has no row in `_audit/unit_registry.md` yet, and
`test_registry_coverage.py`'s "no record the registry does not name" meta-test will fail
until one is added. Not added here; registry edits are the consolidation pass's.

## a note on `.build.z_tf_inside_half`'s two occupants

`models/stellarator/coils/calculate.py::ZTfInsideHalf` and this module's `ZTfInsideHalf`
own the same `VarPath` from different formulas on different devices. That is not dual
ownership — they are never in one graph — but it is worth recording as a *third* instance
of the same field having more than one writer, on top of the two the stellarator record
already names (`st_build` vs. `st_coil`, resolved by call order until it was made
structural). The tokamak's writer is `build.py:807`; on the stellarator side `st_build`
also has a formula for it, which its `Build` node deliberately does **not** own. So across
both devices this one field has three source-level writers and the winner is decided
differently in each case. A field with that many producers is a good candidate for the
"which writer wins" check `_audit/test_harness.md` says is owed.

## open questions

1. **Should `i_tf_shape == 0` get its own occupant class?** It takes the same PROCESS
   branch as `1` (the source tests `== PICTURE_FRAME`, not a three-way dispatch), so an
   occupant for it would be `TfOutboardMidDShape` under a second name. Left UNPORTED
   rather than folded in, because the binding policy says one class per value and
   `traceability_policy.md` § "The rule is stated unconditionally" is explicit that this
   question is unsettled. Flagging rather than deciding, since it recurs for every
   `!= <one value>` test in PROCESS and deserves one answer, not one per unit.

2. **Should `TfOutboardMidDShape` and `TfOutboardEdgeRipple` be one node?** They are
   PROCESS's two calls to the same fit. Two nodes is what keeps `.build.r_tf_outboard_mid`
   from being read and owned by the same node; one node would have to own both the radius
   and the ripple and read the radius it owns. Two, therefore — but the second node
   recomputes `r_tf_outboard_midmin` and discards it, which is a small duplicated
   computation the graph cannot see is duplicated.

3. **Does the `.tokamak.build` slot want `.build.rspo`?** `divgeom` writes it and it is
   not on the boundary list, because nothing in the current graph reads it. Owned here on
   the grounds that `build.py:912` is its sole tokamak writer; if the `.tokamak.divertor`
   slot's port claims it too, this one should yield.


## 2026-08-27 — the CS-to-TF radial slice ported (cold-boundary wave)

`cold_boundary.md` producer 2. `.build.r_tf_inboard_in` and `.build.r_tf_inboard_out`
were two of the six cold boundary zeros: `tf_global_geometry`'s
`a_tf_inboard_total = pi*(out^2 - in^2)` came out `0` and fed three of the 11
non-finite roots (`j_tf_coil_full_area = c/0`, both CICC inboard fractions). The
"whole central-solenoid radial chain is outside this closure" scoping above is hereby
narrowed: `calculate_r_tf_inboard_radii_tf_outside_cs` + `TfInboardRadiiTfOutsideCs`
port `process/models/build.py:1691-1735` as one contiguous slice — `dr_cs_bore =
dr_bore` (else-arm, `:1698-1699`), `dr_cs_precomp` (`:1702-1713`), and the three
inboard TF radii (`:1717-1735`).

The slice is taken at `:1691` rather than at the cold-boundary record's `:1720`
because stopping at `:1720` leaves `dr_cs_precomp` a fresh boundary input with a
degenerate cold `0.0` (in = 2.6306 for a converged 2.6986 — the silent-stale class
the record's other 23 names) and leaves `dr_cs_bore` the standing stale input it
already was (cold `1.42` for a converged `2.00384`, read by
`pfcoil/currents.py::CSFluxSwing`). Nothing in the wider port produces either field,
so no producer is duplicated.

Switches: a joint key `(.build.i_tf_inside_cs, .build.i_cs_precomp)`, resolved by
`indat._tf_inboard_radii_arm` — `(0, 1)` (both defaults, both live) is written;
`TF_INSIDE_CS` (different reads-set for the inner radius, `:1692-1698`) and the
no-precompression arm (`dr_cs_precomp = 0.0` literal, `fseppc`/`fcspc`/`sigallpc`
unread, `:1714`) are UNPORTED with recorded reasons. Data footprint: reads
`.build.dr_bore`, `.build.dr_cs`, `.build.fseppc`, `.build.fcspc`,
`.build.sigallpc`, `.build.dr_cs_tf_gap`, `.build.dr_tf_inboard` (all explicit-arg;
all file literals or defaults on the reference run — first-pass non-degeneracy per
`cold_boundary.md` Task A); writes `.build.dr_cs_bore`, `.build.dr_cs_precomp`,
`.build.r_tf_inboard_in`, `.build.r_tf_inboard_mid`, `.build.r_tf_inboard_out`.

Tier 1, no iteration; `test_build.py::TestTfInboardRadii` diffs the slice against the
real `calculate_radial_build` through the existing `_radial` adapter (legacy point =
the converged file literals; fuzz over plausible machine scales). No cycle created
(`Blocking.scc` on both reference machines, measured this wave — the slice reads only
run inputs).


## 2026-08-27 — the spherical-tokamak divertor geometry ported (ST frontier wave)

`divertor_geometry_arm == -1` was the standing refusal on **both**
`spherical_tokamak_eval.IN.DAT` and `st_regression.IN.DAT`. Written this wave:
`calculate_divertor_geometry_spherical_tokamak` + `DivertorGeometrySphericalTokamak`,
porting `process/models/build.py:862-863` — `divgeom`'s opening
`if itart == 1: return 1.75e0 * rminor` ("TART option: Peng SOFT paper"). The early
return is the entire arm: none of the arc geometry runs and control never reaches the
`.build.rspo` write at `:912`, so the occupant owns `.build.dz_xpoint_divertor` alone —
the different write-set the refusal named, now structural. The test's reference asserts
on every sample that the real `divgeom` left `rspo` untouched, so the write-set claim is
pinned, not narrated.

**The split turned out to be three-way, and the live disposition on both ST files is
`None`.** The `:800-801` latch (`dz_xpoint_divertor = divht` only when the entering
value is `< 1e-5`) gates the spherical arm exactly as it gates the conventional one, and
*both* tracked spherical-tokamak inputs set `dz_xpoint_divertor = 0.75`
(`spherical_tokamak_eval.IN.DAT:91`, `st_regression.IN.DAT:1989`). On them the
`1.75 * rminor` is computed and discarded and `divgeom` writes nothing at all — under
`itart == 0` that configuration leaves the `rspo`-only remainder (arm `-2`, still
UNPORTED), but under `itart == 1` there is no remainder, so the slot's occupant is
`None`: absence, not refusal, the `DX_TF_SIDE_CASE_MIN` shape, arm `-3` in
`indat._divertor_geometry_arm`. `DivertorGeometrySphericalTokamak` itself is therefore
live only on a spherical-tokamak run that leaves the field at its `0.0` default — no
tracked input does, which is also why no converged reference value exists to write down
(the legacy sample uses the input geometry, `rminor = 4.5 / 1.8 = 2.5`).

Nothing else `itart == 1` gates inside `divgeom`'s chain: the method's only `itart`
test is the `:862` early return, and `calculate_vertical_build`'s remainder
(`:797-842`) branches on `i_single_null` only. The other `itart` gates in `build.py`
(the centrepost/CS radial sub-tree, `r_cp_top`) belong to slots already keyed on it or
already scoped out (§ "the central-solenoid radial chain"). `UNPORTED`'s arm `-2` entry
was sharpened to say it is now the `itart == 0` configuration specifically.

Faithfulness note, inherited: the `:800` latch means a spherical-tokamak run that
*leaves* the field unset would see PROCESS keep the first pass's `1.75 * rminor`
forever while the node recomputes it — the same latch-vs-recompute deviation already
recorded for `DivertorGeometryConventional` (§ deviations, item 2), identical here
because `rminor` alone decides the value.

Harness: `TestDivertorGeometrySphericalTokamak`, tier 1 (one legacy + fuzz over
`rminor ∈ [0.8, 3.0]`); 10 cases green plain and under `--fp-gradients`. Frontier
probe after this wave (`machine_from_indat` + `graph_for`): **both** ST files advance
past `divertor_geometry_arm` and now refuse at `tf_inboard_radii_arm == -2` —
`i_cs_precomp == 0`, the no-CS-pre-compression arm of the CS-to-TF radial slice
(`dr_cs_precomp` is the literal `0.0` at `build.py:1714`; a strict-subset reads-set,
recorded UNPORTED by the CS-slice wave above). That is the next blocker for both
files.


## 2026-08-27 — the no-precompression CS slice ported (ST frontier wave 2)

`tf_inboard_radii_arm == -2` was the standing refusal on **both**
`spherical_tokamak_eval.IN.DAT` and `st_regression.IN.DAT` after the divertor-geometry
wave above. Written this wave: `calculate_r_tf_inboard_radii_no_cs_precomp` +
`TfInboardRadiiNoCsPrecomp`, the `(i_tf_inside_cs, i_cs_precomp) == (0, 0)` cell of
`process/models/build.py:1691-1735` — the exact cell both files select
(`spherical_tokamak_eval.IN.DAT:70-71` sets `i_cs_precomp = 0`, `i_tf_inside_cs = 0`;
`st_regression.IN.DAT:1845`/`:1811` the same), verified against the files rather than
assumed from the refusal text.

The arm is the sibling slice with the `i_cs_precomp` else-branch taken:
`dr_cs_bore = dr_bore` unchanged, `dr_cs_precomp` the literal `0.0e0` (`:1714`) instead
of the `fseppc`-formula, `r_tf_inboard_in`'s `+ dr_cs_precomp` term absorbed as the
exact zero it is, `r_tf_inboard_mid`/`r_tf_inboard_out` unchanged. Reads-set is a
strict subset of the sibling's (`fseppc`/`fcspc`/`sigallpc` never read — the reason it
is a different occupant and not a kwarg); write-set is **identical**, `dr_cs_precomp`
kept as a produced output so downstream readers see the arm's zero rather than a stale
boundary value. The port emits it as `jnp.zeros_like(dr_bore)` so the tuple stays
array-valued under `jacfwd`; its gradient row is identically zero on both sides of the
harness.

Registration: `indat.TF_INBOARD_RADII[-2] = TfInboardRadiiNoCsPrecomp`; the
`("tf_inboard_radii_arm", -2)` `UNPORTED` entry removed; `_tf_inboard_radii_arm`
unchanged (its `-2` return already existed — only the registry gained the occupant).
Arm `-1` (`TF_INSIDE_CS`) remains the slot's one refused value.

Harness: `TestTfInboardRadiiNoCsPrecomp`, tier 1, through the existing `_radial`
adapter with `build__i_cs_precomp=0` — the reference is the real
`calculate_radial_build` on the flipped switch, and `fseppc`/`fcspc`/`sigallpc` stay at
their nonzero `BASELINE` values deliberately, so a wrong-arm reference would fail by
value, not by division error. Legacy point = `spherical_tokamak_eval.IN.DAT`'s input
radial build (`dr_bore = 0.23375250334739459`, `:61`; `dr_cs = 0.20016400484967947`,
`:77`; `dr_cs_tf_gap = 0.0`, `:67` — the file's true zero-gap edge; `dr_tf_inboard =
0.9`, `:345`); input values, not converged ones — no converged reference for this cell
has been solved yet. Green plain and under `--fp-gradients` (test file: 57 passed
plain, 114 with gradients).

Frontier probe after this wave (`machine_from_indat` + `graph_for`): **both** ST files
advance past `tf_inboard_radii_arm` and now refuse at `i_tf_shape_build == 2` —
picture-frame TF coil (`i_tf_shape == 2`), whose closed-form ripple formula
(`process/models/build.py:1585-1590`) reads neither the winding pack nor the `c1`/`c2`
fit coefficients; recorded UNPORTED since first writing. That is the next blocker for
both files.
