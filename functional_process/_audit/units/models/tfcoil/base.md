---
kind: model-unit
status: draft
confidence: high
---

**Not registered.** No `unit_registry.md` row, no `total_process.py` binding, no
`indat.py` entry — registration is the consolidation pass's job (`next_steps.md` §4b),
and the wave-1 dispatch brief forbids editing those files from here. The rows the
registry needs are named in the porting agent's report; until they exist,
`tests/functional_process/test_registry_coverage.py::test_every_record_file_is_in_the_registry`
will fail on this file, and that failure is the reminder rather than a defect.

## source

`process/models/tfcoil/base.py`, **partial**. Scope is the minimal closure of
`.tokamak.cicc_superconducting_tf_coil`'s ten boundary reads
(`_audit/tokamak_boundary.md`), reached through
`SuperconductingTFCoil.run_base_superconducting_tf` -> `TFCoil.run_base_tf`
(`base.py:124-207`), not the whole 4670-line file.

| in scope | lines | shape |
|---|---|---|
| `tf_global_geometry` | 213–372 | `@staticmethod`, **but takes `data`** — three switches, two back-door reads |
| `tf_current` | 374–426 | `@staticmethod`, pure |
| `tf_coil_shape_inner` | 428–580 | instance method (calls `self.circumference`); three switches |
| `circumference` | 1183–1209 | `@staticmethod`, pure |
| `tf_coil_self_inductance` | 2065–2191 | `@staticmethod` + `numba.njit`; one switch pair, a 100-step loop |
| `tf_stored_magnetic_energy` | 582–634 | `@staticmethod`, pure |
| `generic_tf_coil_area_and_masses` | 2193–2218 | instance method, `data` in / `data` out, no switch |
| `run_base_tf`'s inline `r_b_tf_inboard_peak` | 166–171 | four-term subtraction written between two model calls |

Out of scope, with the reason each was excluded, is listed in the port module's
docstring (`functional_process/models/tfcoil/base.py`) rather than duplicated here.
The short version: `tf_field_and_force` and `stresscl` feed only stresses, which no
boundary read depends on; `cntrpst` and the `he_*`/`al_th_cond` property lookups it
calls are `itart == 1` centrepost code.

**`he_density`/`he_cp`/`he_visco`/`he_th_cond` (`base.py:1827-2017`) are CoolProp-backed
but are *not* on this slot's CoolProp path.** They are reached only from `cntrpst`. The
one CoolProp dependency that does touch this slot is in
`_audit/units/models/tfcoil/quench.md`.

## data footprint

Reference run: `tests/regression/input_files/large_tokamak_eval.IN.DAT`. Live switch
values on that run: `i_tf_sup = 1` (default, `tfcoil_variables.py:261`), `i_tf_case_geom
= 0` (default, `:234`), `i_f_dr_tf_plasma_case = False` (default, `:83`),
`tfc_sidewall_is_fraction = False` (default, `:95`), `itart = 0` (default),
`i_single_null = 1` (`IN.DAT:307`), `i_tf_shape = 1` — promoted from the `0` default by
`process/core/init.py:775-776` because `itart == 0`.

### `tf_global_geometry` (`base.py:213-372`)

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.tfcoil.n_tf_coils` | read | explicit-arg | *(live)* the parameter; **also read off `data` at `:334,339`** — see D4 |
| `.build.r_tf_inboard_out` | read | explicit-arg | *(live)* |
| `.build.r_tf_inboard_in` | read | explicit-arg | *(live)* |
| `.build.r_tf_outboard_mid` | read | explicit-arg | *(live)* |
| `.build.dr_tf_outboard` | read | explicit-arg | *(live)* |
| `.build.dr_tf_inboard` | read | explicit-arg | *(live)* only reaches `dr_tf_plasma_case` |
| `.tfcoil.dr_tf_nose_case` | read | explicit-arg | only reaches `dx_tf_side_case_min`, and only on the fraction arm |
| `.tfcoil.f_dr_tf_plasma_case` | read | explicit-arg | `i_f_dr_tf_plasma_case == True` arm only |
| `.tfcoil.casths_fraction` | read | explicit-arg | `tfc_sidewall_is_fraction == True` arm only |
| `.tfcoil.dr_tf_plasma_case` | **read** | implicit-io | *(live)* `:328`, the entering value on the `False` arm — and the same field the result is written to. The self-loop; see below |
| `.tfcoil.dx_tf_side_case_min` | **read** | implicit-io | *(live)* `:358`, the entering value on the `False` arm, returned unchanged — an identity |
| `.superconducting_tfcoil.rad_tf_coil_inboard_toroidal_half` | write | explicit-arg | *(live)* |
| `.superconducting_tfcoil.tan_theta_coil` | write | explicit-arg | *(live)* |
| `.tfcoil.a_tf_inboard_total` | write | explicit-arg | *(live)* the one output `i_tf_case_geom` branches |
| `.superconducting_tfcoil.r_tf_outboard_in` | write | explicit-arg | *(live)* |
| `.superconducting_tfcoil.r_tf_outboard_out` | write | explicit-arg | *(live)* |
| `.tfcoil.dx_tf_inboard_out_toroidal` | write | explicit-arg | *(live)* |
| `.tfcoil.a_tf_leg_outboard` | write | explicit-arg | *(live)* |
| `.tfcoil.dr_tf_full_midplane` | write | explicit-arg | *(live)* |
| `.tfcoil.dr_tf_internal_midplane` | write | explicit-arg | *(live)* |
| `.tfcoil.dr_tf_plasma_case` | **write** | implicit-io | *(live)* clamped, `:325-340` |
| `.tfcoil.dx_tf_side_case_min` | **write** | conditional-ownership-by-run-config | *(live)* on the `False` arm the write is an identity, so the port declares no owner there |

**The two self-writes are not the same shape and are ported differently.**

- `.tfcoil.dx_tf_side_case_min` on `tfc_sidewall_is_fraction == False` is
  `dx = data.tfcoil.dx_tf_side_case_min` (`:358`) — a verbatim read-back. The port
  writes **no node** for that arm and the field stays a run input. Conditional
  ownership, the shape `models/power/thermal_cryo.py` already records.
- `.tfcoil.dr_tf_plasma_case` on `i_f_dr_tf_plasma_case == False` reads the entering
  value (`:328`) and then raises it to a geometric minimum (`:333-340`), so the write is
  **not** an identity and the field cannot simply stay an input. `DrTfPlasmaCaseFromInput`
  is a `FixedPointFunction`: `step` reads `.tfcoil.dr_tf_plasma_case` and writes the
  minted `^cond.tfcoil.dr_tf_plasma_case`, and the paired `FixedPoint` owns the real
  path. The fixed point closes in one iteration because `jnp.maximum(x, m)` is idempotent
  in `x` and `m` does not depend on `x`. On the reference run
  `dr_tf_plasma_case` is unset (`tfcoil_variables.py:77` default `0.0`) and the clamp
  binds, so `d(out)/d(in) == 0` there exactly.
- The `i_f_dr_tf_plasma_case == True` sibling has **no** self-read at all — the clearest
  available demonstration that the loop belongs to one arm and not to the quantity.
  `DrTfPlasmaCaseFromFraction` is a plain `ExplicitFunction`.

**The fixed point is not unique, and a driver needs to know that.** `x = max(x, m)`
holds for *every* `x >= m`, so the solution set is a half-line, not a point, and which
member the drive lands on is decided entirely by the initial iterate — which is exactly
PROCESS's semantics (*keep the input thickness unless it is too thin*). A `FixedPoint`
seeded from the entering `DataStructure` value reproduces that in one step. A
`RootFind`/Newton drive over the same residual would be ill-posed on the `x > m` branch,
where the residual is identically zero and its Jacobian singular. Recorded here rather
than left for the driver-selection pass to discover.

### `tf_current` (`base.py:374-426`)

Reads `.tfcoil.n_tf_coils`, `.physics.b_plasma_toroidal_on_axis`, `.physics.rmajor`,
`.tfcoil.r_b_tf_inboard_peak`, `.tfcoil.a_tf_inboard_total`; writes
`.tfcoil.b_tf_inboard_peak_symmetric`, `.tfcoil.c_tf_total`,
`.superconducting_tfcoil.c_tf_coil`, `.tfcoil.j_tf_coil_full_area`. All `explicit-arg`,
all live, no switch.

### `tf_coil_shape_inner` (`base.py:428-580`)

Writes `.tfcoil.len_tf_coil` (**boundary read #3**), `.tfcoil.tfa`, `.tfcoil.tfb`,
`.tfcoil.r_tf_arc`, `.tfcoil.z_tf_arc` — all `explicit-arg`.

Reads, per arm — this is the table the split exists to produce:

| read | D-shape / itart 0 / single null *(live)* | D-shape / itart 0 / double null | D-shape / itart 1 | picture frame |
|---|:-:|:-:|:-:|:-:|
| `.build.r_tf_inboard_out` | yes | yes | – | itart 0 only |
| `.physics.rmajor`, `.physics.rminor` | yes | yes | yes | – |
| `.superconducting_tfcoil.r_tf_outboard_in` | yes | yes | yes | yes |
| `.build.z_tf_inside_half` | yes | yes | yes | yes |
| `.build.z_tf_top` | **yes** | **no** | yes | yes |
| `.build.dr_tf_inboard` | yes | yes | yes | yes |
| `.build.r_cp_top` | – | – | yes | itart 1 only |
| `.build.dr_tf_outboard` | – | – | yes | – |
| `.build.r_tf_outboard_mid`, `.build.r_tf_inboard_mid` | – | – | – | yes |

Two occupants are written (both `itart == 0`, `i_tf_shape == D_SHAPE`, one per
`i_single_null`); the other two arms are UNPORTED. `.build.z_tf_top` is the measured
invented edge: a single node carrying `i_single_null` as a static kwarg would declare it
on the double-null arm, where PROCESS never reads it.

### `tf_coil_self_inductance` (`base.py:2065-2191`)

Writes `.tfcoil.ind_tf_coil`. The `itart == 0 and i_tf_shape == 1` arm reads
`.build.dr_tf_inboard`, `.tfcoil.r_tf_arc`, `.tfcoil.z_tf_arc` and **nothing else**; the
`else` arm reads `.build.z_tf_inside_half`, `.build.dr_tf_outboard`,
`.build.r_tf_outboard_mid`, `.build.r_tf_inboard_mid` and nothing else. Disjoint, six
reads apart, and PROCESS's own unit test states the same thing in a comment
(`tests/unit/models/tfcoil/test_tfcoil.py:597-599`: *"the following 4 params are not used
by tf_coil_self_inductance because this tests the D-shaped coil branch"*). Both arms are
ported.

### `generic_tf_coil_area_and_masses` (`base.py:2193-2218`)

| VarPath | read/write | classification |
|---|---|---|
| `.build.r_tf_inboard_out`, `.build.r_tf_inboard_in`, `.build.r_tf_inboard_mid`, `.build.r_tf_outboard_mid` | read | explicit-arg |
| `.superconducting_tfcoil.rad_tf_coil_inboard_toroidal_half`, `.superconducting_tfcoil.tan_theta_coil` | read | explicit-arg |
| `.tfcoil.len_tf_coil` | read | explicit-arg |
| `.tfcoil.tfocrn` | write | explicit-arg |
| `.tfcoil.tficrn` | write | local-intermediate → the source writes `tfocrn` then reads it back one line later to form `tficrn` (`:2202-2206`); a Python local in the port, still a declared output |
| `.tfcoil.tfcryoarea` | write | explicit-arg — **boundary read #9** |

`wbtf` is already a source-level local. Despite the name, the function computes no mass
on this path.

## proposed signature(s)

Written and shipped in `functional_process/models/tfcoil/base.py`; the module docstring
carries the split rationale. Summary:

```python
circumference(aaa, bbb)
calculate_tf_global_geometry_circular_case(*, n_tf_coils, r_tf_inboard_out,
        r_tf_inboard_in, r_tf_outboard_mid, dr_tf_outboard)
calculate_tf_global_geometry_straight_case(...)          # same five reads
dr_tf_plasma_case_from_input(*, dr_tf_plasma_case, r_tf_inboard_in, dr_tf_inboard,
        n_tf_coils)
dr_tf_plasma_case_from_fraction(*, f_dr_tf_plasma_case, dr_tf_inboard, r_tf_inboard_in,
        n_tf_coils)
dx_tf_side_case_min_from_fraction(*, casths_fraction, r_tf_inboard_in, dr_tf_nose_case,
        n_tf_coils)
calculate_r_b_tf_inboard_peak(*, r_tf_inboard_out, dr_tf_plasma_case,
        dx_tf_wp_insulation, dx_tf_wp_insertion_gap)
tf_current(*, n_tf_coils, b_plasma_toroidal_on_axis, rmajor, r_b_tf_inboard_peak,
        a_tf_inboard_total)
tf_coil_shape_inner_d_shape_single_null(*, r_tf_inboard_out, rmajor, rminor,
        r_tf_outboard_in, z_tf_inside_half, z_tf_top, dr_tf_inboard)
tf_coil_shape_inner_d_shape_double_null(...)             # without z_tf_top
tf_coil_self_inductance_d_shape(*, dr_tf_inboard, r_tf_arc, z_tf_arc)
tf_coil_self_inductance_picture_frame(*, z_tf_inside_half, dr_tf_outboard,
        r_tf_outboard_mid, r_tf_inboard_mid)
tf_stored_magnetic_energy(*, ind_tf_coil, c_tf_total, n_tf_coils)
generic_tf_coil_area_and_masses(*, r_tf_inboard_out, r_tf_inboard_in,
        rad_tf_coil_inboard_toroidal_half, tan_theta_coil, len_tf_coil,
        r_tf_inboard_mid, r_tf_outboard_mid)
```

## cottax node

Thirteen classes, in the same file, in four families. See the port module. In summary:
`TfGlobalGeometry{CircularCase,StraightCase}`, `DrTfPlasmaCase{FromInput,FromFraction}`
(the first a `FixedPointFunction`), `DxTfSideCaseMinFromFraction`, `RBTfInboardPeak`,
`TfCurrent`, `TfCoilShapeDShape{SingleNull,DoubleNull}`,
`TfCoilSelfInductance{DShape,PictureFrame}`, `TfStoredMagneticEnergy`,
`GenericTfCoilAreaAndMasses`.

## tier signal

**Tier 1** for every function here. None calls `scipy.optimize`, none calls another
`Model`. `tf_coil_self_inductance`'s 100-step loop is a **fixed-count quadrature**, not
a solve — no convergence test, no stopping rule — so it is tier 1 by
`test_harness.md`'s own criterion (tier 2 is *"an internal iterative loop closing over
state"*). It is ported as a `lax.scan` that reproduces PROCESS's `r -= dr` recurrence
term by term, so the branch `x0 - r < ai` flips at exactly the same interval index.

## switches touched

| switch | values seen | note |
|---|---|---|
| `i_tf_case_geom` | 0 *(live)*, 1 | `tfcoil_variables.py:234`. Both ported. Reads-sets **identical**; split anyway, per the binding policy |
| `i_f_dr_tf_plasma_case` | False *(live)*, True | `:83`. Both ported. Reads differ (`dr_tf_plasma_case` vs `f_dr_tf_plasma_case`) **and** so does the node kind |
| `tfc_sidewall_is_fraction` | False *(live)*, True | `:95`. `True` ported; `False` is a no-node arm |
| `i_tf_shape` | 1 *(live, promoted)*, 2 | `:268`; `0` never survives `init.py` |
| `itart` | 0 *(live)*, 1 | `physics_variables`; `1` UNPORTED |
| `i_single_null` | 1 *(live)*, 0 | `IN.DAT:307`. Both ported |

`i_tf_sup` is **not** touched by anything in scope here: it appears in `run_base_tf`'s
callees only via `tf_field_and_force`, which is out of scope. The device-level dispatch
on it lives in `process/core/caller.py:295-316`, one layer above every model, which is
`schema.md`'s "resolved above this file" pattern.

## calls into other models

None. `tf_coil_shape_inner` calls `self.circumference`, which is in the same class and
in scope.

## JAX-difficulty flags

- `needs-lax-cond-or-where` — **minor**. `tf_global_geometry`'s minimum-thickness clamp
  (`:333`) and `tf_coil_self_inductance`'s `x0 - r < ai` (`:2145,2171`) branch on traced
  values. Ported as `jnp.maximum` and `jnp.where`; the inner square root is guarded by
  substitution (the double-`where` idiom) because `1 - ((r-x0)/ai)**2` is negative
  exactly where the branch is not taken.
- `in-place sequential mutation` — **workaround-known**. `tf_coil_shape_inner` fills
  `r_tf_arc`/`z_tf_arc`/`tfa`/`tfb` element by element (`:500-523`); ported as
  `jnp.stack` plus a vectorised difference. `tf_coil_self_inductance` accumulates
  `ind_tf_coil` in a Python loop; ported as `lax.scan` **deliberately, not as a
  vectorised sum** — see the tier-signal note.
- `numba.njit` on the reference — **minor**. `tf_coil_self_inductance` is compiled, so
  the harness adapter must pass `float`/`np.ndarray`, never `None` and never a JAX array.

## defects found

- **D1.** `base.py:344-346` logs *"dr_tf_plasma_case too small to accommodate the WP,
  forced to minimum value"* **unconditionally** — the call sits outside the `if` at
  `:333` whose body it describes, so every single evaluation emits it (visible in this
  audit's own probe output). Logging only, no value effect. Dropped in the port, since a
  pure function does not log.
- **D4.** `tf_global_geometry` takes `n_tf_coils` as a parameter but reads
  `data.tfcoil.n_tf_coils` at `:334` and `:339`, inside the clamp. In the pipeline
  `run_base_tf` passes `self.data.tfcoil.n_tf_coils` (`:132`), so the two always agree and
  the port is right to carry one read — but **PROCESS's own unit test drives them apart**:
  `test_tf_global_geometry`'s second case passes `12` while `data.tfcoil.n_tf_coils` keeps
  its `16` default, so the published expected value `0.04` is the un-clamped input for a
  16-coil machine. The harness adapter sets the field, which is the "close the `data`
  back-door" check working exactly as intended. Same shape, less consequential, for
  `data.tfcoil.dr_tf_plasma_case` (`:328`) and `data.tfcoil.dx_tf_side_case_min` (`:358`).

## Shared with the stellarator

`.tfcoil.len_tf_coil` and `.tfcoil.tfcryoarea` are the *same* `VarPath`s
`models/stellarator/coils/calculate.py`'s `LenTfCoil` and `TfCryoArea` produce
(`boundary_inputs_audit.md` §7 items 4 and 7). Here they come from
`tf_coil_shape_inner_d_shape_single_null` and `generic_tf_coil_area_and_masses`, from
entirely different formulas — a stellarator coil length is a modular-coil circumference
scaling, a tokamak's is four elliptical arcs plus a straight inboard section. **One
variable, two device-specific producers, two slots**, which is what a slot is for. Any
assembled machine must bind exactly one of each; `TokamakProcess` binds these two.

## open questions

1. **`.tfcoil.dr_tf_plasma_case`'s `FixedPointFunction` may be more machinery than the
   graph needs.** Its fixed point is reached in one step and, at the reference point,
   its self-derivative is exactly zero, so a driver has nothing to do. It is written as a
   loop because the *structure* is a loop — cottax refuses a node that reads what it
   owns, and the read is real. Whether the orchestrator would rather see `dr_tf_plasma_case`
   split into an input `dr_tf_plasma_case_input` and an owned clamped output under a new
   name is a naming-policy call this record does not take (`naming_convention.md`: "do
   not invent new names").
2. **`tf_field_and_force` is the one remaining piece of `run_base_tf`'s chain not ported
   here.** It is out of the boundary closure, but it is the last gap between this port
   and a complete `run_base_superconducting_tf`. It carries `i_tf_sup`, `itart` and
   `i_cp_joints`, so it is three more switch families rather than one function.
