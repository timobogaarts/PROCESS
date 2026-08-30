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

Two occupants were written first (both `itart == 0`, `i_tf_shape == D_SHAPE`, one per
`i_single_null`); a third, the picture frame at `itart == 1`, joined on 2026-08-27 — see
the dated section at the foot of this record, which also splits the last column of the
table above in two, because the picture frame's two `itart` sub-branches have different
reads-sets. `.build.z_tf_top` is the measured invented edge: a single node carrying
`i_single_null` as a static kwarg would declare it on the double-null arm, where PROCESS
never reads it.

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
- **D5.** `tf_coil_shape_inner`'s picture-frame branch (`:551-578`) never assigns `tfa` or
  `tfb`, so both leave the function as the `np.zeros(4)` they were allocated at
  `:495-496` — and are written to `.tfcoil.tfa`/`.tfcoil.tfb` anyway (`:186-190`). On
  that arm the two fields silently mean "unset" rather than being absent, and no reader
  can tell which. Ported faithfully (the occupant owns them and produces the zeros);
  found 2026-08-27, see the dated section.

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

## 2026-08-27 — the picture-frame TF coil shape at `itart == 1` ported (ST frontier wave 4)

Both tracked spherical-tokamak files were refused at `tf_coil_shape_arm == -1`, whose
`UNPORTED` reason read *"`.physics.itart == 1`: the TART arms of `tf_coil_shape_inner`
read `.build.r_cp_top`, which the conventional arms never touch."* **The refusal named an
arm neither file reaches.** `_tf_coil_shape_arm` tested `itart` first and returned `-1`
for every spherical tokamak; `tf_coil_shape_inner`'s own dispatch
(`process/models/tfcoil/base.py:498`, `:528`, `:551`) tests `i_tf_shape` first, and its
`itart == 1` clause at `:528` is guarded by `i_tf_shape == D_SHAPE`. Both files set
`i_tf_shape = 2` (`spherical_tokamak_eval.IN.DAT:357`, `st_regression.IN.DAT:803`)
together with `itart = 1` (`:283`, `:66`), so PROCESS takes the **picture-frame** branch
at `:551` and its `itart == 1` sub-branches at `:555-556` and `:575-578`. The
"centrepost D-shape" arm at `:528-549` is unreachable on both files.

So arms `-1` and `-2` were **not** two blockers in sequence: they were one dispatch
written down wrongly. Reordering `_tf_coil_shape_arm` to test `i_tf_shape` before `itart`
gives four cells where there were three, and only one of them needed writing:

| cell | arm | status |
|---|:-:|---|
| `PICTURE_FRAME`, `itart == 1` | `2` | **written this wave** — both ST files |
| `PICTURE_FRAME`, `itart == 0` | `-2` | UNPORTED, reason rewritten |
| `D_SHAPE`, `itart == 1` | `-1` | UNPORTED, reason rewritten |
| `D_SHAPE`, `itart == 0` | `0` / `1` | written, unchanged (`i_single_null`) |

The reads column of the table in §`tf_coil_shape_inner` splits accordingly: the written
picture-frame arm reads `.build.r_cp_top`, `.superconducting_tfcoil.r_tf_outboard_in`,
`.build.z_tf_inside_half`, `.build.z_tf_top`, `.build.dr_tf_inboard` and
`.build.r_tf_outboard_mid` — **six**, and it reads neither `.physics.rmajor` nor
`.physics.rminor`, which both D-shape siblings do, nor `.build.r_tf_inboard_out` /
`.build.r_tf_inboard_mid`, which its own `itart == 0` sibling does. A single fused node
carrying `i_tf_shape`/`itart` as static kwargs would have declared ten edges on every
arm; four of the ten would have been invented on this one. The reference adapter pins
all five unread arguments at `0.0`, so a port that secretly read any of them fails by
value.

**`.tfcoil.tfa` and `.tfcoil.tfb` are exact zeros on this arm, and that is PROCESS's
answer.** `:495-496` allocates them `np.zeros(4)`; the picture-frame branch never assigns
an element, because a picture frame has no elliptical arcs to take semi-axes of. The two
fields are still written to `data` at `:186-190`, so the occupant owns them and produces
the zeros rather than declining to own them. Recorded below as defect **D5**: on this arm
the two fields silently mean "unset" rather than being absent, and any downstream reader
that does not know which shape it is looking at cannot tell the difference.
`tf_coil_self_inductance` is the only in-scope reader and it takes the picture-frame arm
here (`_tf_self_inductance_arm` already sends any `itart == 1` to
`TfCoilSelfInductancePictureFrame`), which reads neither — so nothing in this port
consumes the zeros.

### `.build.r_cp_top` has no producer: a lost producer, declared, not stubbed

The new occupant's `.build.r_cp_top` read has nothing on the other end of it in this
port. PROCESS writes the field in `Build.calculate_radial_build`
(`process/models/build.py:1750-1813`), a slice that is not ported and belongs to
`models/build.md`, not here, so **the read enters both ST graphs as a boundary input**
— the same disposition `.build.z_tf_top` already has on the tokamak reference
(`reference_boundary_tokamak.txt:52`). No stub, no invented default: the boundary file's
own header calls growth in the `input` list "a lost producer", and this is one.

Worth recording what that producer *would* be, because it is a one-liner on these runs
and it is not the one the input files appear to ask for. Both files set `i_r_cp_top = 2`
(`spherical_tokamak_eval.IN.DAT:78`, `st_regression.IN.DAT:2029`), i.e. *"`r_cp_top` from
the top/midplane radius ratio"*, and `f_r_cp` is `1.4`. But the whole `i_r_cp_top` ladder
sits under `if itart == 1 and i_tf_sup != 1` (`:1750`), and both files set
`i_tf_sup = 1` (`:356`, `:820`) — superconducting. So the live line is the `else` at
`:1813`, `r_cp_top = r_tf_inboard_out`, and `f_r_cp = 1.4` is dead on both runs.
Confirmed by running PROCESS itself: `init_process` + `PlasmaGeometry.run()` +
`Build.run()` on `spherical_tokamak_eval.IN.DAT` gives
`r_cp_top = r_tf_inboard_out = 1.333916508197074` m. Whoever ports that slice should
expect a slot keyed on `(itart, i_tf_sup)` with three `i_r_cp_top` arms inside the
`itart == 1, i_tf_sup != 1` cell, not a bare `i_r_cp_top` registry.

### validation

`TestTfCoilShapePictureFrameTart`, tier 1, in the siblings' file. The legacy point is
the same PROCESS one-pass run: `r_cp_top = 1.333916508197074`,
`r_tf_outboard_in = 9.824594873354488` (= `r_tf_outboard_mid - 0.5 * dr_tf_outboard`
with `dr_tf_outboard = 0.9`, `f_dr_tf_outboard_inboard = 1.0` at `:85`),
`z_tf_inside_half = 11.735`, `z_tf_top = 12.635`, `dr_tf_inboard = 0.9` (the file's
literal, `:345`), `r_tf_outboard_mid = 10.274594873354488`. That last number is the
ripple-limited radius the picture-frame ripple wave (`models/build.md`, wave 3) derived
algebraically from the 1 % limit; arriving at it independently from a PROCESS run is a
free cross-check that both waves are describing the same machine. One pass, not a
converged solve — no converged reference for either ST file exists yet.

`tests/functional_process/models/tfcoil/test_base.py`: **83 passed / 83 skipped** plain
(was 79/79), **166 passed** with `--fp-gradients` (was 158), **502 passed** with
`--fp-fuzz 8 --fp-gradients`. `test_machine.py`, `test_switch_coverage.py`,
`test_registry_coverage.py`, `test_boundary.py`, `test_machine_survey.py`: 289 passed,
44 skipped, unchanged.

Registration: `indat.TF_COIL_SHAPE[2] = TfCoilShapePictureFrameTart`;
`_tf_coil_shape_arm` reordered as above; the `("tf_coil_shape_arm", -1)` and
`(..., -2)` `UNPORTED` reasons rewritten to name the arms they now actually key.

### frontier probe

`machine_from_indat` + `graph_for` on both ST files, after this wave: **neither
assembles yet**, and both advance past the TF coil shape to the same next refusal,
verbatim —

> `itart_sc_tf_masses == 1` is a real PROCESS branch but is not ported: the
> spherical-tokamak TF mass arm additionally owns `.tfcoil.whtcp` and
> `.tfcoil.whttflgs` (`superconducting.py:2086-2093`), which the conventional arm never
> writes -- conditional ownership again. Not written

Arm `-2` (`PICTURE_FRAME`, `itart == 0`) is therefore **not** the next blocker, which is
the direct answer to the question this wave was asked to check: the two `UNPORTED`
entries were one dispatch, not two waves of work. `machine_survey` was not run, since
neither file assembles and `survey` needs an assembled graph.

## 2026-08-30 — `stresscl` REFUSED, measured: `sig_tf_case`, `sig_tf_wp`, `str_wp`

> **Superseded the same day.** Everything below stands as measurement; only the verdict
> moved. `stresscl`, `tf_field_and_force`, `plane_stress` and the four smearing helpers
> are ported in `functional_process/models/tfcoil/stress.py` under registry row **55**,
> with `_audit/units/models/tfcoil/stress.md` as their record — scoped exactly as the
> "recommended next step" at the foot of this section asked, to the
> `(i_tf_stress_model, i_tf_bucking) == (1, 1)` cell, with `extended_plane_strain` and
> the bucked-and-wedged stack refused in `indat.py`. Two things this section got wrong
> in the small: `i_tf_tresca` turns out **not to be read at all** on that cell (both its
> branches are gated on `ii >= i_tf_bucking + 1` and the two reported layers are
> `n_tf_bucking` and `n_tf_bucking - 1`), and `.tfcoil.vforce` had to be ported with it,
> because `stresscl`'s only load input had no producer either. The cost estimate was
> right about the line count and wrong about what it buys: constraints 31 and 32 were
> not merely reading zero, they were evaluating to a **constant** `-1.0` with a zero
> Jacobian row, which `stress.md` records.

A missing-producer wave (`_audit/units/models/build.md`, same date) asked for
`.tfcoil.sig_tf_case`, `.tfcoil.sig_tf_wp` and `.tfcoil.str_wp` as producers. **All
three are outputs of `stresscl` and none of them is portable as a wiring change.** This
section records the measurement so the next wave does not re-derive it.

### the three fields have exactly one writer each, and it is the same one

| field | written at | PROCESS on `large_tokamak_nof` | port |
|---|---|---|---|
| `.tfcoil.sig_tf_wp` | `base.py:3232` (`s_shear_tf_peak[n_tf_bucking]`) | `4.9699609e+08` | `0.0` |
| `.tfcoil.sig_tf_case` | `base.py:3233` (`s_shear_tf_peak[n_tf_bucking - 1]`) | `5.9391981e+08` | `0.0` |
| `.tfcoil.str_wp` | `base.py:2988` or `:3035`, per `i_tf_stress_model` | `0.0017456388` | `0.0` |

The surrounding `superconducting.py`/`resistive.py` sites (`:2208`, `:2220`, `:2631`,
`:4161`, `:2211`…) are not second writers: each is
`data.tfcoil.X = data.tfcoil.X if data.tfcoil.X is None else X`, storing `stresscl`'s
return value.

### `.tfcoil.sig_tf_wp` already has an owner, and it is the wrong device

`models/stellarator/coils/forces.py::MaximumStress` owns `.tfcoil.sig_tf_wp` and is
registered on the stellarator. It is **not** a wiring candidate for a tokamak:
`calculate_maximum_stress` is `max_force_density * dr_tf_wp_with_insulation * 1e6`, a
one-line scaling whose input comes from `.stellarator_config.stella_config_max_force_
density` and four other `stella_config_*` fields that do not exist on a tokamak graph at
all. Same `VarPath`, same physical quantity, entirely different model — the
`ZTfInsideHalf` situation (`models/build.md` § "a note on `.build.z_tf_inside_half`'s
two occupants") rather than a lost registration. Checked, not assumed: registering the
stellarator node on a tokamak would fail on its own declared reads before it produced a
wrong number.

### what porting `stresscl` actually costs

`stresscl` is `base.py:2222-3274` — **1053 lines**, 65 parameters, four internal
switches (`i_tf_stress_model`, `i_tf_bucking`, `i_tf_tresca`, `i_pf_conductor`), and its
body is not arithmetic on scalars: it builds nine `n_tf_layer`-length and seven
`n_tf_layer * n_radial_array`-length arrays, calls out to a layer solver, then reduces
them with a **Python `argmax` loop over the radial array** (`:3199-3226`) whose index
`ii_max` selects which radial station every reported stress is read from. The two fields
this wave wanted are the last four lines of that reduction.

The solver it calls is `plane_stress` (`:4236-4459`, **224 lines**) on
`i_tf_stress_model == 1` — the **live arm on `large_tokamak_nof`**, which sets neither
`i_tf_stress_model` (default `1`, `tfcoil_variables.py:211`) nor `i_tf_bucking` (default
`-1`, resolved to `1` for a superconducting coil at `init.py:891-895`) — or
`extended_plane_strain` (`:3719-4235`, 517 lines) on `0`/`2`, plus four smearing helpers
(`eyoung_parallel`, `eyoung_parallel_array`, `eyoung_t_nested_squares`, `eyoung_series`,
`:3660-4670`). Even scoped to the live arm alone that is **~1300 lines of one unit**,
against the 30–150 a producer in this port has cost so far. It is a unit of its own with
its own registry row, not a slot to fill; the refusal is one of scope, not of
traceability (nothing in it is CoolProp-backed or otherwise untraceable — the `argmax`
loop and the `n_tf_layer` assembly would need `jnp` rewrites, which is work, not a
blocker).

### the cost of the absence, measured rather than assumed

`base.md`'s scope section says `stresscl` "feeds only stresses, which no boundary read
depends on". That was true of `_audit/tokamak_boundary.md`'s **ten** reads and is not
true of the constraint surface, which was added later:

- **constraint 31** (`constraints.py:936`) is `sig_tf_case <= sig_tf_case_max`, and
  **constraint 32** (`:952`) is `sig_tf_wp <= sig_tf_wp_max`. Both are on
  `large_tokamak_nof.IN.DAT:146-147`. With both operands' producers absent the port
  evaluates `0 <= max`, so **two active constraints are silently satisfied** — not a
  wrong number, a dropped constraint, which is worse because no residual reports it.
- **`.tfcoil.str_wp` feeds the superconductor fits**, not just constraint 88 (which this
  file does not activate). `i_str_wp` defaults to `1` (`tfcoil_variables.py:508`) and is
  unset in this input, so `superconducting.py:2902-2905` / `:4001-4004` read the
  *strain* from it for both the critical-current surface (constraint 33) and the
  temperature margin (constraint 36), **both active here**. Zero strain is the peak of
  the Nb3Sn strain fit, so the absence is optimistic rather than neutral.

`.tfcoil.str_wp` therefore stays on `missing_producers_tokamak.txt`;
`.tfcoil.sig_tf_case` and `.tfcoil.sig_tf_wp` are not on it only because they are read
by the constraint surface and not by any MDA node, so they appear on the MDF graph's
boundary instead (`tests/functional_process/test_boundary.py::
test_no_new_boundary_input_is_something_process_computes`, which names all three).

**Recommended next step, not taken here:** `stresscl` as its own registry row and its
own record, scoped to the `(i_tf_stress_model, i_tf_bucking) == (1, 1)` cell the tracked
tokamaks take, with `plane_stress` as a tier-2 sub-unit of it and the other stress-model
arms `UNPORTED`.
