---
kind: model-unit
status: draft
confidence: medium
---

**Ported (partial by switch, full by function).** `pfcoil/fields.py` /
`test_fields.py`: `calculate_b_field_at_point`, `calculate_coil_current_waveform`,
`calculate_pf_coil_peak_fields` — tier-1. Two cottax nodes: `PFCoilPeakField`
(`.tokamak.pf_coil.peak_field`) and `PFCoilCurrentWaveform`
(`.tokamak.pf_coil.waveform`).

## source

`process/models/pfcoil.py`:

| lines | what |
|---|---|
| `4926-5060` | `calculate_b_field_at_point`, `@numba.njit`, module-level, pure |
| `2869-2940` | `PFCoil.waveform`, instance method, `self.data` only |
| `4414-4638` | `peak_b_field_at_pf_coil`, module-level, takes and **mutates** a `DataStructure` |

UNPORTED in this file:

| lines | what | why not |
|---|---|---|
| `4452-4456`, `4618-4625` | `peak_b_field_at_pf_coil`'s CS arm (`n_coil == n_cs_pf_coils`) | it returns the four raw components rather than writing `b_pf_coil_peak`, and `ohcalc` combines them with the CS's own self-field (`calculate_cs_self_peak_magnetic_field`, `:3750-3823`). Nothing in this pass's mass closure reads `b_pf_coil_peak[6]`/`bpf2[6]` — see `masses.md` |
| `3707-3823` | `calculate_cs_bore_magnetic_field`, `calculate_cs_self_peak_magnetic_field` | only reachable from the CS arm above |

## data footprint

`calculate_b_field_at_point` reads and writes nothing: every argument is explicit and it
returns its four results. No `VarPath` at all — it is a kernel, reused by
`currents.py`'s `fixb`/`mtrx`.

`PFCoilCurrentWaveform`:

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.pf_coil.c_pf_cs_coil_pulse_start_ma` | read | explicit-arg | `pfcoil.py:2885` |
| `.pf_coil.c_pf_cs_coil_flat_top_ma` | read | explicit-arg | `:2889` |
| `.pf_coil.c_pf_cs_coil_pulse_end_ma` | read | explicit-arg | `:2886` |
| `.pf_coil.n_cs_pf_coils` | read | **topology** | `:2877`, `:2881` — a loop bound, consumed at graph-assembly time (`geometry.md` § switches) |
| `.pf_coil.c_pf_cs_coils_peak_ma` | write | explicit-arg | `:2891`, `:2904`, `:2918` |
| `.pf_coil.f_c_pf_cs_peak_time_array` | write | explicit-arg | `:2879`, `:2923-2940` |

`PFCoilPeakField` (per group; the union over the four groups is what the node declares):

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.pf_coil.c_pf_cs_coil_pulse_start_ma` | read | explicit-arg | see § "waveform folded in" |
| `.pf_coil.c_pf_cs_coil_flat_top_ma` | read | explicit-arg | idem |
| `.pf_coil.c_pf_cs_coil_pulse_end_ma` | read | explicit-arg | idem |
| `.pf_coil.r_pf_coil_middle` | read | explicit-arg | `:4528`, `:4579` |
| `.pf_coil.z_pf_coil_middle` | read | explicit-arg | `:4531`, `:4582`, `:4605` |
| `.pf_coil.r_pf_coil_inner` | read | explicit-arg | `:4604` |
| `.pf_coil.r_pf_coil_outer` | read | explicit-arg | `:4613` |
| `.pf_coil.z_pf_coil_upper` | read | explicit-arg | `:4524` |
| `.pf_coil.z_pf_coil_lower` | read | explicit-arg | `:4525` |
| `.pf_coil.r_pf_coil_middle_group_array` | read | explicit-arg | the two clobbered filament slots — see below |
| `.pf_coil.z_pf_coil_middle_group_array` | read | explicit-arg | idem |
| `.pf_coil.r_cs_middle` | read | explicit-arg | filaments 2-13, via `place_cs_filaments` |
| `.pf_coil.dz_cs_full` | read | explicit-arg | idem |
| `.pf_coil.a_cs_poloidal` | read | explicit-arg | `:4507` |
| `.pf_coil.j_cs_pulse_start` | read | explicit-arg | `:4495`, sign only |
| `.pf_coil.j_cs_flat_top_end` | read | explicit-arg | `:4495`, `:4505` |
| `.physics.rmajor` | read | explicit-arg | `:4593` |
| `.physics.plasma_current` | read | explicit-arg | `:4595` |
| `.pf_coil.b_pf_coil_peak[0..5]` | write | explicit-arg, **per index** | `:4630` |
| `.pf_coil.bpf2[0..5]` | write | explicit-arg, **per index** | `:4631` |
| `.pf_coil.r/z/c_pf_cs_current_filaments` | write | **refused** | `:4501-4595` mutates all three as scratch; see § "Storage this package refuses to own" |
| `.pf_coil.xind` | write | **refused** | `:4599`, `:4608` — a scratch slice of `calculate_b_field_at_point`'s mutual inductances, overwritten by the second call before anything can read it |
| `.pf_coil.nfxf`, `.pf_coil.n_cs_pf_coils`, `.pf_coil.n_pf_coil_groups`, `.pf_coil.n_pf_coils_in_group` | read | **topology** | loop bounds |
| `.build.iohcl` | read | switch | `:4452`, `:4489`, `:4619` |

## waveform folded in

`peak_b_field_at_pf_coil` reads `c_pf_cs_coils_peak_ma` and
`f_c_pf_cs_peak_time_array` and then re-derives which of the three time points the peak
came from, raising `ProcessValueError` if none matches within `1e-12` (`:4459-4487`).
Both fields are pure functions of the three `c_pf_cs_coil_*_ma` arrays — that is the
entirety of `waveform` — so `calculate_pf_coil_peak_fields` takes the three arrays and
derives them internally.

This **narrows** the declared read set (three arrays instead of two derived from those
three), so it invents no edge. What it buys is that the `1e-12` precondition becomes
structurally unreachable: the peak is a bitwise copy of one of the three, so exactly one
comparison is an exact equality. The alternative — taking `c_peak` as a free input —
would put essentially every fuzz draw outside PROCESS's domain, which is a validated
nothing rather than a validated port.

The cost is that `PFCoilCurrentWaveform` computes the same three lines again for its own
readers (`masses.py`'s sizing and mass chains, and the unported `outpf`/`vsec`/`induct`).
Recorded rather than hidden.

## A PROCESS defect ported faithfully

**CS current filaments 0 and 1 do not hold CS positions.** The sequence inside one
`pfcoil()` pass is:

1. `:223-234` — `place_cs_filaments` fills `r/z/c_pf_cs_current_filaments[0:14]` with the
   CS's 14 filaments.
2. `:474-479` — the `i_pf_current = 1` equilibrium branch overwrites entries `0` and `1`
   (`nocoil` starts at 0) with the *PF coil* positions of groups 0 and 1, and their
   currents, so those coils can act as fixed-current filaments in the equilibrium solve.
3. `:4500-4509` — `peak_b_field_at_pf_coil` rewrites the first `nfxf = 14` **currents**
   with the CS filament current, and never touches the positions.

The result is that two of the fourteen "CS filaments" used for every PF coil's peak field
sit at `r = 5.567 m, z = 9.644 m` and `r = 5.567 m, z = -10.878 m` — the PF coils — while
carrying `-CS current / 14`. **Verified against a live traced run**, not inferred from
the source: instrumenting `calculate_b_field_at_point` during
`PFCoil.pfcoil()` on `large_tokamak_eval.IN.DAT` shows the first filament array as
`r = [5.5667, 5.5667, 2.2773, 2.2773, ...]`, `z = [9.6443, -10.8782, 2.8344, 3.9682,
...]` where an unclobbered CS would give `r = [2.2773] * 14` and
`z = [0.5669, 1.7007, ...]`.

Reproduced exactly by `fields._cs_filament_positions`. The test's reference adapter
reproduces it by calling PROCESS's own `place_cs_filaments` and then applying PROCESS's
own two-entry overwrite, so the oracle is not a restatement of the port's assumption.

## Storage this package refuses to own

`.pf_coil.r_pf_cs_current_filaments`, `.pf_coil.z_pf_cs_current_filaments`,
`.pf_coil.c_pf_cs_current_filaments` and `.pf_coil.xind` are written by at least three
different places within one `pfcoil()` pass, each overwriting part of what the last
wrote, and are read only by the writer that comes next. They are scratch, not state:
giving them an owner would either need four nodes to own overlapping slices of the same
array or one node to own a value it does not compute. **No node in this package owns
them.** The functions that need them build them as locals from the `VarPath`s they are
derived from. Flagged for the tier-3/4 comparison, where they will read as
PROCESS-writes-nothing-here.

## proposed signature(s)

```python
def calculate_b_field_at_point(r_current_loop, z_current_loop, c_current_loop,
                               r_test_point, z_test_point) -> tuple  # (ind, br, bz, psi)
def calculate_coil_current_waveform(c_pf_cs_coil_pulse_start_ma,
                                    c_pf_cs_coil_flat_top_ma,
                                    c_pf_cs_coil_pulse_end_ma) -> tuple  # peak (7,), f_c (7, 6)
def calculate_pf_coil_peak_fields(...18 args...) -> tuple  # b_pf_coil_peak (6,), bpf2 (6,)
```

## cottax node

`PFCoilPeakField(ExplicitFunction)` — twelve **per-index** `Output`s,
`.pf_coil.b_pf_coil_peak[0..5]` and `.pf_coil.bpf2[0..5]`, per
`_audit/naming_convention.md` § "Array elements" and following
`models/physics/composition.py`'s precedent. Index 6 belongs to the CS's own self-field
(UNPORTED), so owning the whole array would claim a value this node does not compute.

`PFCoilCurrentWaveform(ExplicitFunction)` — owns `.pf_coil.c_pf_cs_coils_peak_ma` and
`.pf_coil.f_c_pf_cs_peak_time_array` whole. Index 6 of the first is written twice by
PROCESS (here and at `ohcalc`, `:3264-3281`) with algebraically identical values, since
`c_cs_flat_top_end = -a_cs_poloidal * j_cs_flat_top_end` makes `waveform`'s
end-of-flat-top pick equal `ohcalc`'s expression whenever that is the largest of the
three, which it is on this arm. Confirmed numerically on the reference run
(`-186.11980298805437` both ways).

## tier signal

**Tier 1.** No iteration on either side. `peak_b_field_at_pf_coil` mutates a
`DataStructure`, but only as scratch; the adapter binds one and reads back the two
arrays.

**Sample provenance.** Legacy points read off a converged in-process `SingleRun` of
`large_tokamak_eval.IN.DAT` (there is no `tests/unit` case for any of these three).
Fuzzing for `calculate_pf_coil_peak_fields` is `±5-15 %` around that point rather than
over an absolute range: the calculation has no domain guard to trip, but a filament set
drawn from unrelated decades measures nothing about whether the port sums PROCESS's
terms in PROCESS's order.

## switches touched

| switch | reachable values | live on `large_tokamak_eval` | decision | evidence |
|---|---|---|---|---|
| `.build.iohcl` | `0`, `1` | `1` | **split** | `:4489-4511` — `iohcl = 0` drops all 14 CS filaments and the whole `sgn`/`a_cs_poloidal` read |
| `.pf_coil.i_pf_current` | `0`, `1` | `1` | **split** (indirectly) | the filament clobber above exists only on the `i_pf_current = 1` path; a port for `i_pf_current = 0` would see unclobbered CS filaments |
| target coil is the CS | — | never, here | **split** | `:4452`, `:4619` — the CS arm returns different quantities; UNPORTED |

## calls into other models

None.

## JAX-difficulty flags

- **`min(s, 0.999999)`** (`:5002`) → `jnp.minimum`. A kink; the gradient is zero above
  the clamp, which is also what PROCESS's finite difference sees.
- **`if dr == 0.0: dr = 1e-6`** (`:5012-5014`) → `jnp.where(dr == 0.0, 1e-6, dr)`. Exact
  float equality on a traced value is a legitimate predicate; the branch is reachable
  only when a test point sits at exactly a loop's radius.
- **The plasma filament's conditional presence** (`:4591`, `t_b_field_peak > 2`) would
  make the traced filament *count* data-dependent. Replaced by an always-present loop
  whose current is masked to zero, which is exactly equivalent because `brx`/`bzx` are
  linear in the current. Recorded as a deviation in form, not in value.
- **The time-point index** is a traced integer used as an array index
  (`jnp.take(waveform, column, axis=1)`). Integers carry no tangent, so this is a
  piecewise-constant selection with zero derivative — the same thing PROCESS's finite
  difference sees, since a small perturbation does not move which time point is the peak
  except exactly at a tie.
- No CoolProp, no `scipy.special`. The elliptic integrals here are A&S polynomial fits
  (`:4969-4986`), which is why this kernel traces at all — unlike the CS *stress*
  functions (`:4102-4206`), which call `scipy.special.ellipk`/`ellipe` and would need a
  custom JAX primitive. Those are outside this pass's closure.

## open questions

- **Should `PFCoilCurrentWaveform` exist at all, given `PFCoilPeakField` re-derives it?**
  Kept because `masses.py`'s two nodes read `.pf_coil.c_pf_cs_coils_peak_ma` and would
  otherwise have to re-derive it a third time. The alternative — one node owning the
  waveform *and* the peak fields — would merge two computations with quite different
  read sets. Flagged, not decided.
- **`peak_b_field_at_pf_coil`'s `ProcessValueError`** (`:4484-4487`) has no non-finite
  counterpart in the port, because the folded-in waveform makes it unreachable. If a
  future occupant takes `c_peak` as a free input, that branch needs a NaN path and a
  `reference_domain_errors` declaration.
