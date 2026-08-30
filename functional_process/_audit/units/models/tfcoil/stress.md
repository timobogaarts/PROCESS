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

**`extended_plane_strain` (`base.py:3719-4234`, 517 lines) is not ported and is the
single largest exclusion.** It is the live solver on both tracked *spherical* tokamaks
(`spherical_tokamak_eval.IN.DAT:350`, `st_regression.IN.DAT:1223`), neither of which
assembles today for an unrelated reason (`i_tf_turn_type == 2`, the CroCo turn). So the
refusal costs nothing measurable now and will be the next thing to cost something the
day the CroCo unit lands.

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
`eyoung_axial`. Worth re-checking if that arm is ever ported.
