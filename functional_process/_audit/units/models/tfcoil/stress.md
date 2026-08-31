---
kind: model-unit
status: draft
confidence: high
---

**Ported and registered.** `functional_process/models/tfcoil/stress.py`,
`tests/functional_process/models/tfcoil/test_stress.py`, registry row 55, two slots on
`.tokamak.cicc_superconducting_tf_coil`. This record supersedes
`_audit/units/models/tfcoil/base.md`'s 2026-08-30 section, which measured the cost of
this unit and refused it as a slot fill; its "recommended next step, not taken here" is
what landed.

**2026-08-31: `extended_plane_strain` landed too, and both tracked spherical tokamaks
assemble.** The section "the second solver" at the end of this record is that wave;
everything before it describes the plane-stress arm and is unchanged except where a
sentence has been overtaken (marked in place). `spherical_tokamak_eval` and
`st_regression` both build a `TokamakProcess` and a 234-node graph, and nothing else on
either file refuses.

## source

`process/models/tfcoil/base.py`, **partial**, four disjoint ranges:

| in scope | lines | shape |
|---|---|---|
| `tf_field_and_force` | 1623–1821 | `@staticmethod`, pure, three switches |
| `stresscl` | 2222–3274 | `@staticmethod` + `numba.njit`, 65 parameters, four switches |
| `eyoung_parallel` | 3660–3716 | module function, `numba.njit`, pure |
| `plane_stress` | 4236–4458 | module function, `numba.njit`, one `np.linalg.solve` |
| `eyoung_parallel_array` | 4460–4522 | module function, loops `eyoung_parallel` |
| `eyoung_t_nested_squares` | 4524–4600 | module function, loops both of the others |
| `eyoung_series` | 4602–4670 | module function, one branch on a zero modulus |

**The scope is one cell of a switch space, not a range of lines**, and the cell is
`(i_tf_sup, i_tf_stress_model, i_tf_bucking) == (1, 1, 1)` with both values of
`i_tf_turns_integer`. `stress.py`'s module docstring holds the per-switch table with the
reason each unported arm is unported; `indat.py`'s `_TF_STRESS_MODEL_REASON`,
`_TF_BUCKING_REASON` and the `tf_field_and_force_arm` entry hold the same reasons in the
form the factory raises them. Not duplicated here.

**`extended_plane_strain` (`base.py:3719-4234`, 517 lines) was the single largest
exclusion and is now ported** -- see "the second solver" below. This paragraph's
prediction held exactly: it was the live solver on both tracked *spherical* tokamaks
(`spherical_tokamak_eval.IN.DAT:350`, `st_regression.IN.DAT:1223`), neither of which
assembled at the time for an unrelated reason (`i_tf_turn_type == 2`, the CroCo turn),
so the refusal cost nothing measurable *then* and became the only thing costing anything
the day the CroCo unit landed.

## why this unit exists at all: two dropped constraints

Measured on `tests/regression/input_files/large_tokamak_nof.IN.DAT`, which activates
constraints 31 and 32 (`IN.DAT:146-147`).

`leq(value, bound)` (`core/solver/constraints.py:59-69`) returns a normalised residual of
`value / bound - 1`. With no producer for `.tfcoil.sig_tf_case` the port read the cold
`DataStructure`'s `0.0`, so constraint 31's condition was **exactly `-1.0`** — satisfied
with a fixed margin, *constant in every design variable*, and therefore contributing a
row of zeros to the Jacobian. Constraint 32 the same. Nothing anywhere reported it: a
constraint that evaluates is not distinguishable from a constraint that means something
unless somebody asks what it read.

After: at PROCESS's converged point, with the port's own producers,

| condition | before | after (port) | PROCESS | agreement |
|---|---|---|---|---|
| `^cond.constraints.c31` | `-1.0`, constant | `-2.2364977268e-05` | `-2.2364977268e-05` | 5.6e-16 |
| `^cond.constraints.c32` | `-1.0`, constant | `-1.3474702228e-01` | `-1.3474702228e-01` | 1.7e-15 |
| `^cond.constraints.c33` | (read `str_wp = 0`) | `-2.5955134073e-01` | `-2.5955134073e-01` | 0 |
| `^cond.constraints.c36` | (read `str_wp = 0`) | `-4.5556950689e-04` | `-4.5556950688e-04` | 6.7e-16 |

Constraint 31 is **nearly binding** at PROCESS's optimum — 2.2e-5 of relative margin
against `sig_tf_case_max`. The port had it at a comfortable `-1.0`, which is the
difference between a design that is at its case-stress limit and one that believes it
has 100 % of it spare.

33 and 36 are in the table because `.tfcoil.str_wp` feeds both through
`i_str_wp == 1` (`tfcoil_variables.py:508`, unset in this file):
`CiccSuperconductorProperties` and `TfSuperconductorTemperatureMargin` were reading
`0.0`, and **zero strain is the peak of the Nb3Sn strain fit**, so the absence was
optimistic rather than neutral. Both already agreed with PROCESS at the *seeded* point,
because the harness's Stage A seeds boundary inputs from PROCESS's converged
`DataStructure` — which is exactly the blindness `boundary.unproduced_but_computed`
exists to remove.

## data footprint

Reference run: `tests/regression/input_files/large_tokamak_nof.IN.DAT`, at PROCESS's
converged point. Live switch values there: `i_tf_sup = 1`, `i_tf_stress_model = 1`
(default, `tfcoil_variables.py:211`), `i_tf_bucking = 1` (`-1` default resolved at
`init.py:891-895`), `i_tf_tresca = 0`, `i_tf_turns_integer = 0`, `n_tf_graded_layers = 1`,
`n_rad_per_layer = 500` (overwritten by `superconducting.py:2100`), `itart = 0`,
`i_cp_joints = -1`.

### `tf_field_and_force` (`base.py:1623-1821`), clamped-joint superconducting arm

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.superconducting_tfcoil.r_tf_wp_inboard_outer` | read | explicit-arg | *(live)* |
| `.superconducting_tfcoil.r_tf_wp_inboard_inner` | read | explicit-arg | *(live)* |
| `.superconducting_tfcoil.r_tf_outboard_in` | read | explicit-arg | *(live)* |
| `.tfcoil.dx_tf_wp_insulation` | read | explicit-arg | *(live)* |
| `.tfcoil.dx_tf_wp_insertion_gap` | read | explicit-arg | *(live)* `i_tf_sup == 1` only |
| `.tfcoil.b_tf_inboard_peak_symmetric` | read | explicit-arg | *(live)* |
| `.tfcoil.c_tf_total` | read | explicit-arg | *(live)* |
| `.tfcoil.n_tf_coils` | read | explicit-arg | *(live)* |
| `.tfcoil.dr_tf_plasma_case` | read | explicit-arg | *(live)* |
| `.physics.rmajor` | read | explicit-arg | *(live)* |
| `.physics.b_plasma_toroidal_on_axis` | read | explicit-arg | *(live)* |
| `.tfcoil.f_vforce_inboard` | read | conditional-ownership-by-run-config | *(live)* an input on this arm; the sliding-joint arm **writes** it (`base.py:1801`) |
| `.build.r_cp_top` | read | — | sliding-joint arm only; **not a port here** |
| `.tfcoil.cforce` | write | explicit-arg | *(live)* |
| `.tfcoil.vforce` | write | explicit-arg | *(live)* |
| `.tfcoil.vforce_outboard` | write | explicit-arg | *(live)* |
| `.superconducting_tfcoil.vforce_inboard_tot` | write | explicit-arg | *(live)* |

### `stresscl` (`base.py:2222-3274`), the `(1, 1, 1)` cell

Thirty-seven reads, every one `explicit-arg`. Listed by group rather than one row each,
because none of them carries a classification worth a sentence — `stresscl` takes
everything as a parameter and touches no `self.data` at all, which is the one thing that
makes a 1053-line function cheap to port once the switch space is cut down.

| group | VarPaths |
|---|---|
| layer boundaries | `.build.r_tf_inboard_in`, `.superconducting_tfcoil.r_tf_wp_inboard_inner`, `.superconducting_tfcoil.r_tf_wp_inboard_outer`, `.tfcoil.dr_tf_plasma_case` |
| coil section | `.superconducting_tfcoil.tan_theta_coil`, `.superconducting_tfcoil.rad_tf_coil_inboard_toroidal_half` |
| case and steel areas | `.superconducting_tfcoil.a_tf_coil_inboard_steel`, `.superconducting_tfcoil.a_tf_plasma_case`, `.superconducting_tfcoil.a_tf_coil_nose_case`, `.tfcoil.a_tf_coil_inboard_case` |
| elastic constants | `.tfcoil.eyoung_steel`, `.poisson_steel`, `.eyoung_cond_axial`, `.poisson_cond_axial`, `.eyoung_cond_trans`, `.poisson_cond_trans`, `.eyoung_ins`, `.poisson_ins`, `.eyoung_copper`, `.poisson_copper` |
| turn geometry | `.tfcoil.dx_tf_turn_insulation`, `.dx_tf_wp_insertion_gap`, `.dx_tf_wp_insulation`, `.n_tf_coil_turns`, `.dia_tf_turn_coolant_channel`, `.f_a_tf_turn_cable_copper`, `.dx_tf_turn_steel`, `.a_tf_turn_steel`, and **one of** `.superconducting_tfcoil.dx_tf_turn_cable_space_average` / `.dr_tf_turn_cable_space` per `i_tf_turns_integer` |
| winding pack | `.superconducting_tfcoil.dx_tf_side_case_average`, `.dx_tf_wp_toroidal_average`, `.a_tf_coil_inboard_insulation`, `.a_tf_wp_with_insulation`, `.tfcoil.a_tf_wp_steel`, `.a_tf_wp_conductor` |
| load | `.tfcoil.c_tf_total`, `.tfcoil.vforce` |

Writes: `.tfcoil.sig_tf_wp`, `.tfcoil.sig_tf_case`, `.tfcoil.str_wp`, `.tfcoil.casestr`,
`.tfcoil.insstrain` — all five `explicit-arg`, all five stored by
`superconducting.py:2205-2231` under the `X = X if X is None else <computed>` idiom that
is **not** a second writer.

`.tfcoil.sig_tf_cs_bucked` is a sixth output of `stresscl` and this node does not own it:
`stresscl` leaves it `None` unless `i_tf_bucking >= 2` (`base.py:3230-3231`).
Conditional ownership, and this is the arm that does not own it — the same shape
`models/power/thermal_cryo.py` records.

### the twenty-eight returns that are not ports

`stresscl` returns 34 values; PROCESS stores six and the other 28 are locals of
`SuperconductingTFCoil.tf_stress` passed straight to `out_stress`
(`superconducting.py:2105-2138`, `:2232`). They have no `VarPath`, so there is nothing
to own, and the port does not compute the ones nothing else needs — specifically
`sig_tf_vmises`, `s_shear_cea_tf_cond` and the four `sig_tf_*_max` reductions, whose
only consumers are the printer and the `i_tf_tresca == 1` branch this cell cannot reach.

## proposed signature(s)

Seven functions, all in `functional_process/models/tfcoil/stress.py`:

```python
def eyoung_parallel(eyoung_j_1, a_1, poisson_j_perp_1, eyoung_j_2, a_2, poisson_j_perp_2)
def eyoung_series(eyoung_j_1, l_1, poisson_j_perp_1, eyoung_j_2, l_2, poisson_j_perp_2)
def eyoung_parallel_array(eyoung_j_in, a_in, poisson_j_perp_in)
def eyoung_t_nested_squares(eyoung_j_in, l_in, poisson_j_perp_in)
def plane_stress(*, nu, rad, ey, j, n_radial_array=N_RADIAL_ARRAY)
def tf_field_and_force_clamped_joints(*, ...)      -> (cforce, vforce, vforce_outboard,
                                                       vforce_inboard_tot)
def tf_stress_plane_stress_bucked_case(*, ...)     -> (sig_tf_wp, sig_tf_case, str_wp,
                                                       casestr, insstrain)
```

**Three signature changes from PROCESS's, each deliberate:**

1. `eyoung_parallel_array` and `eyoung_t_nested_squares` drop PROCESS's leading `n`. It
   is a member count that `range(n)` never exceeds, so the sequence length carries it —
   and PROCESS's own unit test passes `n = 4` with five-element arrays, which is exactly
   the mismatch a length can no longer have. The test module slices at the call site so
   the reference sees the same members.
2. `plane_stress` drops `nlayers` for the same reason.
3. `tf_stress_plane_stress_bucked_case` takes `dx_tf_turn_cable_space_eyoung` rather
   than either of the two fields `i_tf_turns_integer` chooses between
   (`base.py:2745-2749`). The two node arms spell the read differently; the arithmetic
   is shared, so the shared function names the *role*.

`n_radial_array` and `n_tf_graded_layers` are counts, not switches (kind (b) in
`_audit/switch_elimination_design.md` §3): the first is a module constant the node never
overrides (`superconducting.py:2100` fixes it at `500` regardless of the input file's
`n_rad_per_layer`), the second a static field on the occupant.

## cottax node

Four classes in two families, both slots on `.tokamak.cicc_superconducting_tf_coil`:

| family | arms written | switch |
|---|---|---|
| `TfFieldAndForce` | `TfFieldAndForceClampedJoints` | `(itart, i_cp_joints)` |
| `TfStress` | `TfStressPlaneStressBuckedCaseAveragedTurn`, `...IntegerTurn` | `(i_tf_stress_model, i_tf_bucking, i_tf_turns_integer)` |

Both families exist as base classes with no `__call__` because their unwritten arms own
*different* fields — `f_vforce_inboard` and `sig_tf_cs_bucked` respectively — which is
the test `next_steps.md` §14.2 asks a family to pass rather than a bare class.

`TfStressPlaneStressBuckedCase` is an intermediate class holding the five outputs and
the `n_tf_graded_layers` static field; the two turn arms differ in one `From` and
nothing else. Written out twice rather than parameterised, because a read is a
declaration: on the integer arm `.superconducting_tfcoil.dx_tf_turn_cable_space_average`
has **no producer at all** (`CiccIntegerTurnGeometry` does not own it), so declaring it
there would put a stale `DataStructure` value into the answer.

## tier signal

**Tier 1** for all seven. No `scipy.optimize`, no iteration, no other model's method.
`plane_stress` solves a linear system, which is a direct factorisation and not an
internal solve — `Tier2Contract` is for a loop whose convergence PROCESS does not check,
and there is none here.

## switches touched

| switch | values seen | note |
|---|---|---|
| `.tfcoil.i_tf_sup` | `1` | resolved above every model, `caller.py:295-316` — not a slot here |
| `.tfcoil.i_tf_stress_model` | `1` written; `0`, `2` refused | `plane_stress` vs `extended_plane_strain` |
| `.tfcoil.i_tf_bucking` | `1` written; `0`, `2`, `3` refused | `-1` resolved at `init.py:891-895` |
| `.tfcoil.i_tf_turns_integer` | `0` and `1`, **both written** | picks which cable-space field the transverse smearing reads |
| `.tfcoil.i_tf_tresca` | **not read** | both its branches are gated on `ii >= i_tf_bucking + 1` and this node reports layers `n_tf_bucking` and `n_tf_bucking - 1` |
| `.physics.itart` x `.tfcoil.i_cp_joints` | clamped written; sliding refused | `i_cp_joints == -1` resolves to `0` for a superconducting coil |
| `.tfcoil.n_tf_graded_layers` | `1` | a count, static field |

## calls into other models

None. `stresscl` and `tf_field_and_force` are `@staticmethod`s taking every input as a
parameter; the only things they call are the five module-level helpers in the same file
and `calculate_tresca_stress` / `calculate_von_mises_stress` from
`process/models/engineering/materials.py`. The Tresca one is transcribed into
`stress.py` (three lines); the von Mises one is not needed on this cell, because
`sig_tf_vmises` reaches no output this node owns.

## JAX-difficulty flags

- `np.linalg.solve` inside `numba.objmode` -> `jnp.linalg.solve` — **minor**. Same
  algorithm class; agreement is to rounding, not to the bit, and PROCESS's own comment
  (`base.py:4404-4412`) says its answer already varies with the LAPACK build.
- Python `argmax` loop over 500 radial stations per layer -> `jnp.argmax` on a reshape —
  **workaround-known**, applied. The degenerate case (`ii_max` left at the *global* zero
  when a layer's shear never exceeds `0.0`) is reproduced rather than normalised; see
  `_layer_peak_indices`.
- `eyoung_series`'s zero-modulus branch -> double `jnp.where` — **workaround-known**,
  applied, and load-bearing: `eyoung_cond_axial` and `eyoung_cond_trans` are `0.0` on
  the reference run, so this is the live branch and an unguarded `l / eyoung` would put
  `inf` into the primal and `nan` into every tangent (`models/safe_math.py`'s defect
  class, fifth instance).
- `max()` over a Python list of traced scalars in `eyoung_t_nested_squares` ->
  `jnp.max(jnp.stack(...))` — **minor**.
- Nothing CoolProp-backed and nothing otherwise untraceable.

## validation

`tests/functional_process/models/tfcoil/test_stress.py`, six `Tier1Contract`s.

Against PROCESS at `large_tokamak_nof`'s converged point, calling
`TFCoil.stresscl`/`TFCoil.tf_field_and_force` directly with the converged
`DataStructure`'s values:

| output | PROCESS | port | relative |
|---|---|---|---|
| `cforce` | `6.956085039419e+07` | same | `0` |
| `vforce` | `1.827090329732e+08` | same | `0` |
| `vforce_outboard` | `1.827090329732e+08` | same | `0` |
| `vforce_inboard_tot` | `2.923344527572e+09` | same | `0` |
| `sig_tf_wp` | `6.48939733290875e+08` | `6.48939733290873e+08` | `1.9e-15` |
| `sig_tf_case` | `7.49983226267049e+08` | `7.49983226267049e+08` | `6.7e-16` |
| `str_wp` | `2.12583514161912e-03` | same | `0` |
| `casestr` | `1.31603668237741e-03` | same | `0` |
| `insstrain` | `-5.91260699244997e-03` | `-5.91260699244996e-03` | `1.2e-15` |

All five also equal the converged `DataStructure`'s own fields to the same precision,
which is the check that the adapter passes what the pipeline passes.

**Two of PROCESS's own parametrisations are unusable and it is worth recording why.**
`test_tf_field_and_force`'s two samples are both `i_tf_sup = 0, itart = 1,
i_cp_joints = 1` — the resistive sliding-joint arm, the one this port refuses — so the
clamped arm has no legacy oracle and is sampled at the reference point plus fuzz.
`test_plane_stress`'s third sample has `rad[2] < rad[1]`, and PROCESS's own test guards
it with `skip_if_incompatible_system` because the matrix is ill-conditioned enough for
the answer to depend on the LAPACK build; comparing two solvers there measures the
condition number, not the port.

`test_stresscl`'s single sample **is** on this cell (`i_tf_sup = 1`,
`i_tf_stress_model = 1`, `i_tf_bucking = 1`, `i_tf_tresca = 0`, with
`i_tf_turns_integer = INTEGER`), so it is used verbatim — which is why
`tf_stress_plane_stress_bucked_case` keeps `n_radial_array` as a parameter at all: that
sample is taken at `100` and the pipeline runs at `500`.

## boundary effect

`functional_process/reference_boundary_tokamak.txt`: **369 -> 378 inputs**, 11 guesses
unchanged. `.tfcoil.str_wp` left (it has a producer now); ten arrived, and every one is a
material constant PROCESS itself takes from the input file — `eyoung_steel`,
`eyoung_ins`, `eyoung_copper`, `eyoung_cond_axial`, `eyoung_cond_trans`, the four
matching `poisson_*`, and `f_vforce_inboard`. Growth from a landed producer's own
declared reads, which is the boundary doing its job; `--missing` confirms none of the ten
is a field PROCESS computes.

`functional_process/missing_producers_tokamak.txt`: **2 -> 1**. `.costs.c2222` is all
that is left of the twenty-two found on 2026-08-30.

**The cold MDF solve still converges, and takes one more step.** `large_tokamak_nof`
from the `IN.DAT` values, CLARABEL, `tolerance = 1e-8`: **7 SQP iterations** against
PROCESS's 8, reproducing all twenty iteration variables to 1e-3 relative or better bar
one — `.tfcoil.dr_tf_nose_case` (ixc 57) at 2.0e-2, which is the variable constraint 31
now pushes on; `.physics.rmajor`, the figure of merit, agrees to every printed digit.
The wave before this one measured 6 on the same file, so the stress chain costs
one step — which is the expected direction and not a regression: constraint 31 is
*nearly binding* at the optimum, so the port is now solving a problem with two active
inequalities it previously ignored entirely.

## open questions

**OQ1. `n_radial_array` is a resolution the answer plausibly depends on, and nothing
here measures how much.** The five outputs are read off an `argmax` over a grid of
`n_radial_array` stations per layer, and the grid is *open* at the outer end — it starts
at each layer's inner radius and stops one step short of its outer one
(`base.py:4429-4433`) — so the reported peak is the largest sampled value, not the
largest value. PROCESS's pipeline fixes the grid at `500` and its own unit test at
`100`; the port reproduces whichever it is given and both are checked, so this is not a
disagreement. What is not measured is the difference between the two, i.e. how much of
the reported stress is a quadrature artefact. A rewrite that wanted the true peak would
maximise the Tresca stress over each layer analytically instead. Not attempted; the
port's job here is to agree.

**OQ2. The `argmax` makes the outputs piecewise-differentiable, and the pieces are
500-wide.** `jax.jacfwd` through `_layer_peak_indices` gives the derivative *at fixed
peak location*, which is correct almost everywhere and wrong exactly where the peak
station changes — where the true derivative has a kink and PROCESS's own finite
difference would straddle it. The gradient tests (`--fp-gradients`) sample away from
those points by construction. Whether an optimiser can walk into one has not been
measured.

**OQ3. `f_tf_stress_front_case` is applied and then un-applied.** `base.py:2960` scales
the front case's axial modulus by the area ratio and `:3123-3131` divides the resulting
vertical stress back by it — but the modulus also entered the *transverse* solve through
`eyoung_axial`, which `plane_stress` does not read. On this arm the round trip is
therefore exactly neutral for the three outputs that matter, and the port keeps it only
because it is not neutral on the `extended_plane_strain` arms, which do read
`eyoung_axial`. **Resolved for the extended arm, 2026-08-31**: it is indeed not neutral
there. The `ey_z[2]` reaching `extended_plane_strain` is
`205e9 * f_tf_stress_front_case = 1.3636e11` at `spherical_tokamak_eval`'s converged
point -- a 33 % cut in the front case's axial stiffness, which changes the whole stack's
`eps_z`, and only the resulting *stress* is divided back out. Reproduced, not repaired;
the port agrees with `stresscl` to the last bit on all three outputs it owns, which is
the only claim being made.


## the second solver: `extended_plane_strain`, 2026-08-31

`process/models/tfcoil/base.py:3719-4234`, ported as `extended_plane_strain` in
`stress.py`, wrapped as `tf_stress_extended_plane_strain_bucked_case`, occupied by
`TfStressExtendedPlaneStrainBuckedCaseAveragedTurn`, registered as
`TF_STRESS[(0, 1, 0)]`. Both tracked spherical tokamaks assemble: 234 nodes each, and
this was the last blocker on either.

### what was measured before the port was estimated

The CroCo wave's discipline -- check every write against its next use *before*
estimating -- applied to `stresscl`'s five outputs on the `i_tf_stress_model == 0`
branch. **Two of them are dead there, and dead in a way a value test cannot see:**

| output | at `i_tf_stress_model == 1` | at `== 0` | only reader |
|---|---|---|---|
| `sig_tf_wp` | `s_shear_tf_peak[n_tf_bucking]` | same | constraint 32 |
| `sig_tf_case` | `s_shear_tf_peak[n_tf_bucking - 1]` | same | constraint 31 |
| `str_wp` | `sig_tf_z[n_tf_bucking] / eyoung_wp_axial_eff` (`:2988`) | `str_tf_z[n_tf_bucking * n_radial_array]` (`:3034`) | constraints 33, 36 via `i_str_wp == 1` |
| `casestr` | `sig_tf_z[n_tf_bucking - 1] / eyoung_steel` (`:2991`) | **`None`** | `out_stress` only (`base.py:3646`) |
| `insstrain` | `sig_tf_r[...] * ... / eyoung_ins` (`:2994`) | **`None`** | `out_stress` only (`base.py:3653`) |

`casestr` and `insstrain` are initialised to `None` at `:2520-2521` and assigned only
inside the `i_tf_stress_model == 1` branch. `superconducting.py:2224-2231` then writes
that `None` over the `DataStructure`'s `0.0` default (`tfcoil_variables.py:74`, `:208`),
so a converged `SingleRun` on `spherical_tokamak_eval.IN.DAT` really does end with
`.tfcoil.casestr is None` and `.tfcoil.insstrain is None` -- measured on the live
pipeline, not inferred from the source. Grep confirms the only reader of either field
anywhere in `process/` is the printer. **So this occupant owns three fields where its
sibling owns five**, and a port that returned a number for the other two would have
invented one.
`tests/functional_process/models/tfcoil/test_stress.py::test_casestr_and_insstrain_are_none_on_this_arm`
keeps that measurement executable.

Three more reads drop out for a related reason. The vertical stress is *solved for* on
this arm rather than being `vforce / (a_tf_coil_inboard_case + a_tf_turn_steel *
n_tf_coil_turns)` broadcast over the whole leg, so **`.tfcoil.vforce`,
`.tfcoil.a_tf_coil_inboard_case` and `.tfcoil.a_tf_turn_steel` are not read at all**;
`.superconducting_tfcoil.vforce_inboard_tot` -- the whole *set's* inboard tension, not
one coil's -- arrives instead (`base.py:3021`). Net: 37 reads on the plane-stress arm,
35 here, and one of the 35 is a field the other never touches.

### why a second occupant and not a switch

`next_steps.md` §14.2's rule, and a textbook instance of it. An `i_tf_stress_model`
kwarg on `tf_stress_plane_stress_bucked_case` would have to declare `vforce`,
`a_tf_coil_inboard_case`, `a_tf_turn_steel` **and** `vforce_inboard_tot` as reads, three
of which are dead on whichever arm runs -- the `NeutronWallLoad` defect exactly -- and
would have to own `casestr`/`insstrain` on an arm where PROCESS's answer for them is
`None`. Two occupants, one switch key, no dead read.

### the solver itself

The docstring's promise holds: the linear solve is `4 x 4` regardless of layer count.
Each layer contributes a `5 x 5` transfer matrix on the vector
`(A, B, eps_z, 1, eps_z_slip)`, the stack collapses onto the outermost layer's, and four
scalar conditions -- zero radial stress outside, zero radial stress (or zero
displacement at a zero inner radius) inside, prescribed total axial force, zero axial
force on the slip layers -- become four rows. 517 lines of PROCESS became roughly 200 of
port, and the numerical core is one `jnp.linalg.solve` on a `(4, 4)`.

Everything that varies with `nlayers` or `i_tf_bucking` is a **static** Python loop, as
in `plane_stress`: `nlayers` is `len(d_curr)` and `i_tf_bucking` is a static argument,
because it decides how many rows of the transfer-matrix construction exist rather than a
value inside them.

Two grids differ between the solvers and it is worth naming, because OQ1 turns on it.
`plane_stress`'s radial grid is **open** -- step `(rad[ii+1] - rad[ii]) / n_radial_array`
-- so it never samples a layer's outer face. `extended_plane_strain`'s is **closed**:
step `/ (n_radial_array - 1)` (`base.py:4160`), last point exactly on the outer radius.
At `spherical_tokamak_eval`'s converged point the peak Tresca stress in each of the
three layers sits *on* a boundary, so `n_radial_array = 100` and `= 500` give
bit-identical answers -- measured, both are samples -- and OQ1's quadrature artefact is
smaller on this arm than on the other.

### JAX-difficulty flags

- `np.linalg.solve` on the 4x4 -> `jnp.linalg.solve` -- **minor**, but the matrix is
  worse conditioned than `plane_stress`'s and **`extended_plane_strain` does no row
  equilibration**, where `plane_stress` does (`base.py:4404-4419`). Measured condition
  numbers: `3.4e13` on PROCESS's FNSF sample, `4.8e12` at `spherical_tokamak_eval`'s
  converged point, so `cond * eps` is `7.5e-3` and `1.1e-3`. The *achievable* accuracy
  of the raw solution vector is therefore far worse than what either implementation
  delivers, because the ill-conditioned direction is one the outputs barely use -- which
  is what the validation section's scale-relative criterion is about.
- The `f_int_a[0]` division-by-zero guard -> double `jnp.where` -- **workaround-known**,
  applied. PROCESS branches on `f_rec_fac[0] == 0`, which is *exactly* the condition
  under which `log(rad[1] / rad[0])` is infinite: `currents_enclosed[0]` is identically
  zero, so `f_rec_fac[0] = -RMU0/2 * d_curr[0]^2 * rad[0]^2`, which vanishes whenever
  `rad[0]` does. The guard covers precisely the `nan` case and nothing else, so the
  double-`where` is exact rather than approximate.
- The `rad[kk] > 0` guard on `m_ext[0, 1, kk]` -> double `jnp.where` --
  **workaround-known**, applied, same idiom.
- The layer loops (`m_int`, `m_ext`, `m_tot`, and the reverse pass over the radial
  distributions) are all over `range(nlayers)` with `nlayers` static -- unrolled, no
  `lax.scan`. The reverse pass additionally has a closed form: PROCESS recomputes
  `a_vec_layer` from `a_vec_solution` at the end of every iteration rather than
  accumulating, so layer `ii`'s vector is `m_ext[ii+1] @ m_tot[ii+1] @ a_vec_solution`
  and the outermost is the solution itself. Ported as that, not as the loop.
- `sum()` over a numpy array -> `jnp.sum` for the two axial-stiffness areas -- **minor**,
  a reassociation of a sum of at most `nlayers` terms.
- Nothing CoolProp-backed and nothing otherwise untraceable.

### the PROCESS defect this port reproduces

**At `rad[0] == 0` the solver returns `nan`, and it is reproduced rather than repaired.**
The inner boundary condition forces `B = 0` at a zero inner radius, so `b_plot / r` is
`0 / 0` at the innermost test point, and `f_int_a_plot`'s `f_rec_fac * log(rad[1] / 0)`
is `0 * inf` for the same reason. PROCESS's own unit test records exactly this: its
first `ExtendedPlaneStrainParam` carries
`nan_init = ["sigr", "sigt", "sigz", "str_r", "str_t", "r_deflect"]`, six of the eight
returned arrays. The port produces `nan` in those six first elements and nowhere else --
verified against PROCESS on that sample, identical `nan` positions and `<= 1e-12`
relative agreement everywhere else.

**It is unreachable from the ported node.** `stresscl` raises `ProcessValueError` 245
whenever `r_tf_inboard_in` is zero and `i_tf_stress_model != 2` (`base.py:2524-2527`),
and `2` is refused in `indat.py`. The only way to see it is to call the solver directly,
which is why the refusal reason for `i_tf_stress_model == 2` now says what `2` actually
buys -- the removal of that guard and the `radtf[0] = 1e-9` patch at `:2963-2965` --
instead of saying the solver is unported.

### validation

`tests/functional_process/models/tfcoil/test_stress.py`, two new `Tier1Contract`s and
two hand-written tests. Local suite `tests/functional_process/models/tfcoil`:
**366 -> 378 passed** (361 skipped, gradient and fuzz cases opt-in as usual).

Against a real converged spherical tokamak -- a `SingleRun` on
`spherical_tokamak_eval.IN.DAT`, calling the port with that run's own `DataStructure`
values:

| output | PROCESS pipeline | port | relative |
|---|---|---|---|
| `sig_tf_wp` | `302391298.07593733` | same | `0` |
| `sig_tf_case` | `390959416.2857399` | same | `0` |
| `str_wp` | `0.0010498113096371342` | same | `0` |

Against `stresscl` called directly at `large_tokamak_nof`'s converged geometry with
`i_tf_stress_model = 0`: `7.7e-16`, `5.3e-16`, `0`. That call exists to test the
*inert-argument* claim -- `vforce = 0` and both steel areas `1.0`, absurd values against
a real machine's `4.6e7` and `1.6e-1`, so a read that leaked through would show as an
enormous diff rather than a marginal one.

**One contract carries `gradient_floor = 1e-8`, the second in the port to need one**,
and the five components that need it are all boundary conditions rather than model
disagreements. Three are on `sigr[1499]`, which *is* the outer boundary condition (zero
radial stress): the port's `jacfwd` returns an exact `0.0` there and PROCESS's finite
difference, straddling the same cancellation at `epsfcn = 1e-3`, reports `-5.7e-5` with
a Richardson error bar of `8.7e-18` -- an error bar that is small because the perturbed
evaluations agree with each other, not because the derivative is known. The other two
miss by `1.7e-8` and `1.2e-7` relative. Same size and same reason as `TestPlaneStress`'s.

**PROCESS's second unit-test sample is carried by a hand-written test rather than by the
contract, and the reason is a real limitation of an elementwise tolerance.** The FNSF
five-layer stack at `i_tf_bucking = 3` is the *only* oracle anywhere for the
`nonslip_layer > 1` half of this function -- the slip interface row of `m_ext`, the slip
axial-force row, the `eps_z_slip` inner boundary condition, the
`str_z = a_vec_solution[4]` branch. It agrees, but at `1e-13` of each output array's
*scale*:

| array | max abs difference | array scale | ratio |
|---|---|---|---|
| `sigr` | `2.2e-06` | `5.7e+07` | `3.8e-14` |
| `sigt` | `6.8e-06` | `2.2e+08` | `3.1e-14` |
| `sigz` | `1.7e-06` | `4.2e+08` | `4.1e-15` |
| `str_r` | `1.8e-16` | `7.8e-03` | `2.3e-14` |
| `str_t` | `5.1e-17` | `1.5e-03` | `3.5e-14` |
| `str_z` | `5.4e-18` | `1.2e-03` | `4.6e-15` |
| `r_deflect` | `1.2e-16` | `4.3e-03` | `2.8e-14` |

Elementwise, two components are unbounded in relative terms and neither is a
disagreement about the model: `sigr[0]` **is** the inner boundary condition, so PROCESS
returns a hard `0.0` where the port returns `-2.1e-6` from the same cancelling
expression; and the worst relative miss on `sigz`, `1.25e-12`, is on an element of
`1.3e6` inside an array whose scale is `4.2e8`. A per-element `rtol` loose enough to
pass both would be `1e-11` on 12000 numbers that mostly agree to `1e-15` -- a worse test
than the one that is there. So the criterion used is the one that matches the claim,
`1e-13` of each array's own scale, written out in a test function rather than hidden
behind a widened `Tolerance` on the contract.

### what did not change, and one test that had to

The boundary and missing-producer counts above are for the *tokamak reference* machine,
which is `large_tokamak_nof` and takes the plane-stress arm; this wave does not move
them. What it moves is which files assemble at all.

`test_croco.py::test_the_two_tracked_spherical_tokamaks_no_longer_refuse_the_croco_cluster`
had to be rewritten, and the reason is the one this project keeps rediscovering: it
asserted that both ST files still *raised*, and checked only which switch names had left
the refusal message, because when it was written five PF dimensions and
`i_tf_stress_model` still blocked assembly. Every one of those closed, so the test
started failing for the best possible reason. It is now
`test_the_two_tracked_spherical_tokamaks_assemble`, which asserts the stronger claim and
cannot rot in that direction.
