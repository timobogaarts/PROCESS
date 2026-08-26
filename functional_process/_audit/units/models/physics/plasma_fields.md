---
kind: model-unit
status: draft
confidence: medium
---

**First written at the consolidation pass (one node, `TotalMagneticField`), extended
here (2026-08-26) to the rest of `PlasmaFields`' tokamak-relevant closure.** No prior
record existed for this file; this is the first full audit of
`process/models/physics/plasma_fields.py::PlasmaFields`, written by the wave-1 agent
this file was explicitly assigned to.

## source

`process/models/physics/plasma_fields.py`, 268 lines, full file read. **Six `def`s**:

| # | function | lines | shape |
|---|---|---|---|
| 1 | `PlasmaFields.__init__` | 19-22 | sets `self.outfile`/`self.mfile`/`self.current` only |
| 2 | `PlasmaFields.run` | 24-25 | empty stub -- docstring: *"This model cannot yet be 'run'."* |
| 3 | `PlasmaFields.calculate_surface_averaged_poloidal_field` | 27-93 | not `@staticmethod` (calls `self.current.plascar_bpol`); one switch (`i_plasma_current`) |
| 4 | `PlasmaFields.calculate_plasma_inboard_toroidal_field` | 95-118 | `@staticmethod`, pure |
| 5 | `PlasmaFields.calculate_plasma_outboard_toroidal_field` | 120-143 | `@staticmethod`, pure |
| 6 | `PlasmaFields.calculate_toroidal_field_profile` | 145-177 | `@staticmethod`, pure, array-valued |
| 7 | `PlasmaFields.calculate_total_magnetic_field` | 179-198 | `@staticmethod`, pure |
| 8 | `PlasmaFields.output` | 200-268 | pure reporting shell, no computation |

**Unusual shape for this project: there is no stateful `run()` shell at all.** Every
call site of every method above lives in `process/models/physics/physics.py`'s
`Physics.run()` (`self.fields.calculate_*`, `physics.py:313-421`), not in
`PlasmaFields.run()` itself, which the source docstring says outright cannot yet be
run. `PlasmaFields` is a bag of methods `Physics` calls into via composition
(`self.fields = PlasmaFields()`, `physics.py:197`), not a `Model` this project's usual
"stateful shell around a pure core" pattern applies to unmodified -- there is no shell,
only the pure cores plus one `Model` whose own `run()` is unreachable.

## the extraction seam

**As clean as `plasma_geometry.md`'s file, for the same reason: every computational
method is already `@staticmethod` with `self.data` access nowhere in scope.** Functions
4-7 are additionally `@nb.njit(cache=True)`, which changes nothing about their
tracing -- they take plain floats/int, return a plain float/array, `np.` -> `jnp.` is
the entire port. Function 3 (`calculate_surface_averaged_poloidal_field`) is the one
exception: it is a bound method (not `@staticmethod`) because its Peng arm calls
`self.current.plascar_bpol` (`process/models/physics/plasma_current.py`), a different
model instance -- but it still performs no `self.data` access of its own; every input
is an explicit parameter.

This pass ports functions 4-6 (5 and 7 were already reused/ported at consolidation, see
"## already ported" below). Function 8 (`output()`) is out of scope: pure reporting,
one incidental `data` write (`.physics.itart`-style copy does **not** occur here --
checked, `output()` only reads, it never writes a `data` field, unlike
`plasma_geometry.md`'s `itart_r`).

## already ported (not this pass, not re-derived)

- **Function 7, `calculate_total_magnetic_field`** — not re-ported. The consolidation
  pass recognised it is bit-for-bit `models/stellarator/plasma_physics.py::
  calculate_total_field` (already ported, already harness-tested there) and reused that
  function rather than transcribing the formula a second time, adding one node
  (`TotalMagneticField`, owning `.physics.b_plasma_total`) in *this* file. This pass's
  three new total-field nodes (`TotalMagneticFieldInboard`/`Outboard`) reuse the same
  imported function, for the same reason.
- **Function 3, `calculate_surface_averaged_poloidal_field`** — not touched by this
  pass, and not owned by this file's port at all: `functional_process/models/physics/
  physics.py`'s `SurfaceAveragedPoloidalField`/`SurfaceAveragedPoloidalFieldAmperes`
  family owns `.physics.b_plasma_surface_poloidal_average` for `i_plasma_current !=
  PENG_DIVERTOR_SCALING` (the Ampere arm, live on `large_tokamak_eval.IN.DAT`,
  `i_plasma_current = 4`). **The Peng arm (`i_plasma_current == 2`,
  `plasma_fields.py:83-93`) remains UNPORTED** — it calls
  `PlasmaCurrent.plascar_bpol`, and `plasma_current.py` (the model that owns that
  method and the rest of the `i_plasma_current` topology switch) is not ported. This
  record does not re-decide that split; it is `physics.py`'s scope, already made, and
  is repeated here only so this file's own closure is stated completely.

## what this pass ports

Functions 4-6, verbatim `np.` -> `jnp.` translations, no `safe_pow`/`safe_sqrt` site
in any of them (see § JAX-difficulty flags):

```python
def calculate_plasma_inboard_toroidal_field(b_plasma_toroidal_on_axis, rmajor, rminor) -> float
def calculate_plasma_outboard_toroidal_field(b_plasma_toroidal_on_axis, rmajor, rminor) -> float
def calculate_toroidal_field_profile(b_plasma_toroidal_on_axis, rmajor, rminor, n_plasma_profile_elements) -> jnp.ndarray
```

Plus four new cottax nodes, all unswitched, all in
`functional_process/models/physics/plasma_fields.py`:

| class | owns | reads |
|---|---|---|
| `PlasmaInboardToroidalField` | `.physics.b_plasma_inboard_toroidal` | `.physics.b_plasma_toroidal_on_axis`, `.physics.rmajor`, `.physics.rminor` |
| `PlasmaOutboardToroidalField` | `.physics.b_plasma_outboard_toroidal` | `.physics.b_plasma_toroidal_on_axis`, `.physics.rmajor`, `.physics.rminor` |
| `TotalMagneticFieldInboard` | `.physics.b_plasma_inboard_total` | `.physics.b_plasma_inboard_toroidal`, `.physics.b_plasma_surface_poloidal_average` |
| `TotalMagneticFieldOutboard` | `.physics.b_plasma_outboard_total` | `.physics.b_plasma_outboard_toroidal`, `.physics.b_plasma_surface_poloidal_average` |

`calculate_toroidal_field_profile` is ported as a pure function only, **not wired to a
node** — see "## who reads this, and who does not" below.

## who reads this, and who does not

**`.physics.b_plasma_outboard_total` is the reason this pass exists.**
`functional_process/models/physics/scrape_off_layer.py`'s
`UpstreamSOLOutboardParallelArea` (`:364-374`) and
`UpstreamSOLOutboardEich13ParallelArea` (`:396-406`) both declare
`b_plasma_outboard_total=From(physics)` — a read with **no producer anywhere in the
ported graph** before this pass. `TotalMagneticFieldOutboard` (this pass) closes that
gap. Confirmed by grep across `functional_process/` before this pass's edit: zero
matches for `b_plasma_outboard_total\s*=\s*OutputInto`.

`_audit/tokamak_boundary.md`'s "The 58 that are the work list" table (taken before
`scrape_off_layer.py` was ported) attributes **zero** boundary reads to
`.tokamak.plasma_fields`. That is now stale, not wrong at the time it was measured —
`scrape_off_layer.py` did not exist yet when that count was taken (its own record is
also dated 2026-08-26). Flagging the discrepancy per the wave-1 brief's evidence
discipline rather than silently treating this file as still unconsumed. Not correcting
`tokamak_boundary.md` itself — out of this unit's scope (another file, and regenerating
it is a `mda.py`/`boundary.py` invocation, not an edit).

**`.physics.b_plasma_inboard_total` and `.physics.b_plasma_toroidal_profile` still have
no reader anywhere in `functional_process/`.** Grepped for both across the ported tree:
zero matches outside this file and the PROCESS source itself. `b_plasma_inboard_total`
is wired anyway (one line, symmetric with the outboard node that *is* needed, and the
consolidation-pass module's own docstring already flagged both siblings as "one line
either way"). `b_plasma_toroidal_profile` is *not* wired — see below.

**`b_plasma_toroidal_profile`'s real consumers are inside `Physics.run()` itself**,
none ported:

- `physics.py:3860-3872` — `beta_thermal_toroidal_profile`, an array comprehension over
  `calculate_plasma_beta` at each profile point.
- `physics.py:5284-5330` — six calls to `calculate_larmor_frequency` (electron/deuteron/
  triton, on-axis and profile variants).

Both belong to `physics.py`'s own ~7000-line scope (`CLAUDE.md`'s difficulty list names
this file explicitly as an example of an uneven pure/impure split), not to this unit,
and neither is ported. The pure function is ported anyway (cheap, same file, same
closure) but left unwired — a node with no consumer is not wrong to declare, but there
is nothing today to bind it to, and inventing a placeholder reader would be exactly the
kind of unearned edge this project's own conventions warn against.

**No constraint reads anything this file's functions produce.** Grepped
`process/core/solver/constraints.py` for `b_plasma_inboard_toroidal`,
`b_plasma_outboard_toroidal`, `b_plasma_inboard_total`, `b_plasma_outboard_total`,
`b_plasma_toroidal_profile`: zero matches. (`b_plasma_total` and
`b_plasma_toroidal_on_axis` — both outside this pass's scope — do appear there;
unaffected by this pass.)

## data footprint

Reference run: `tests/regression/input_files/large_tokamak_eval.IN.DAT`,
`i_plasma_current = 4` (`IPDG89_SCALING`, Ampere arm).

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.b_plasma_toroidal_on_axis` | read | explicit-arg | `physics.py:397,405,416` — call-site argument, not read inside `plasma_fields.py` itself |
| `.physics.rmajor` | read | explicit-arg | `physics.py:398,406,417` |
| `.physics.rminor` | read | explicit-arg | `physics.py:399,407,418` |
| `.physics.n_plasma_profile_elements` | read | explicit-arg (static shape) | `physics.py:419`, default `201` (`physics_variables.py:1054`), unset on the reference input |
| `.physics.b_plasma_surface_poloidal_average` | read | explicit-arg | `physics.py:375,382,390` — produced by `SurfaceAveragedPoloidalFieldAmperes` (another file) |
| `.physics.b_plasma_inboard_toroidal` | **write** | explicit-arg | `physics.py:395-401`; also read at `physics.py:389` — see § ordering bug |
| `.physics.b_plasma_outboard_toroidal` | **write** | explicit-arg | `physics.py:403-409`; also read at `physics.py:381` — see § ordering bug |
| `.physics.b_plasma_toroidal_profile` | **write** | explicit-arg | `physics.py:414-421`; ported, not wired (see above) |
| `.physics.b_plasma_total` | **write** | explicit-arg | `physics.py:373-376`; already ported (`TotalMagneticField`) |
| `.physics.b_plasma_inboard_total` | **write** | explicit-arg | `physics.py:387-392`; this pass (`TotalMagneticFieldInboard`) |
| `.physics.b_plasma_outboard_total` | **write** | explicit-arg | `physics.py:379-384`; this pass (`TotalMagneticFieldOutboard`) |
| `.physics.b_plasma_vertical_required` | read (`output()` only) | — | produced by `process/models/pfcoil.py:444,575` — a *different*, fenced-off agent's file this wave; out of scope, read-only reporting in `plasma_fields.py`, not touched by this port at all |

## a genuine PROCESS ordering bug — implicit-io, confirmed by reading and by the
## converged reference point

**`physics.py:378-392` reads `b_plasma_inboard_toroidal`/`b_plasma_outboard_toroidal`
before `physics.py:394-409` writes them, within the same `Physics.run()` call.** Read
directly from source (`process/models/physics/physics.py:372-409`, quoted in the port
module's docstring):

```
372  b_plasma_total          = calculate_total_magnetic_field(b_plasma_toroidal_on_axis, b_plasma_surface_poloidal_average)
378  b_plasma_outboard_total = calculate_total_magnetic_field(b_plasma_outboard_toroidal, b_plasma_surface_poloidal_average)  # <- reads b_plasma_outboard_toroidal BEFORE it is (re)computed below
386  b_plasma_inboard_total  = calculate_total_magnetic_field(b_plasma_inboard_toroidal,  b_plasma_surface_poloidal_average)  # <- same, for the inboard field
394  b_plasma_inboard_toroidal  = calculate_plasma_inboard_toroidal_field(...)   # <- computed here, one call too late for line 386-392 above
403  b_plasma_outboard_toroidal = calculate_plasma_outboard_toroidal_field(...)  # <- computed here, one call too late for line 378-384 above
```

So on any single pass through `Physics.run()`, the two total-field calls at 378 and 386
read whatever `b_plasma_outboard_toroidal`/`b_plasma_inboard_toroidal` already held
*before this call started* — `DataStructure`'s `0.0` default on the very first pass,
or the previous `Caller.call_models` Gauss-Seidel iteration's value on every pass after
that (`CLAUDE.md`'s "Implicit cycles are hidden, not declared" describes exactly this
class of bug, generically; this is a concrete instance of it, self-contained within one
file rather than spanning models).

**Why this does not show up as a wrong converged answer.**
`b_plasma_inboard_toroidal`/`b_plasma_outboard_toroidal` depend only on
`b_plasma_toroidal_on_axis`, `rmajor`, `rminor` — none of which the stale read can
perturb (there is no feedback path from `b_plasma_inboard_total`/`b_plasma_outboard_
total` back to any of those three). So the stale value and the current value converge
to the same fixed point once `Caller.call_models`'s outer loop stabilises, and the bug
is invisible at convergence — it costs nothing but a few of the (already-allotted) up
to 10 Gauss-Seidel passes settling to what a single correctly-ordered pass would have
given immediately. **Verified numerically** against
`tests/regression/input_files/large_tokamak_eval.MFILE.DAT`'s converged operating
point: `b_plasma_outboard_toroidal = 3.98874163098367829`,
`b_plasma_surface_poloidal_average = 0.839681017309652056`,
`sqrt(3.98874163098367829**2 + 0.839681017309652056**2) = 4.076165356...`, matching
the reported `b_plasma_outboard_total = 4.07616535601446195` to float64 round-off; the
symmetric check for the inboard pair (`7.97748326196735569`, same poloidal average) →
`8.021552431...`, matching `b_plasma_inboard_total = 8.02155243115691974`.

**Not reproduced in the port, and not fixable in `process/` from this file alone
either** (fixing it would mean reordering `Physics.run()`, out of this unit's scope
and another agent's file). This port declares `TotalMagneticFieldInboard`/`Outboard` as
reading `PlasmaInboardToroidalField`/`PlasmaOutboardToroidalField`'s own current-call
output, which is the *only* shape a cottax DAG can express — there is no way to declare
"read the value this same evaluation is about to overwrite" as an edge; that idea only
exists because PROCESS's imperative call order lets a field be read before its own
write within one function body. The port is therefore a **declared divergence from
PROCESS's literal per-call behaviour, convergent with PROCESS's converged answer** —
exactly the shape `CLAUDE.md`'s SCC discussion predicts a real graph declaration would
produce once an implicit ordering dependency becomes explicit.

## tier signal

**Tier 1, all three new functions and all four new nodes.** No `scipy.optimize`, no
`fsolve`, no ad hoc fixed-iteration loop, no CoolProp. `calculate_toroidal_field_
profile` is array-valued but still a closed-form expression, no internal iteration.

**Sample provenance is the weak point.** No `tests/unit/models/physics/
test_plasma_fields.py` exists in `process/` at all (checked: the directory has no such
file), so — unlike `confinement_time.md`/`scrape_off_layer.md` — there is **no free
legacy sample to lift**. Legacy samples for this unit are instead read directly off
`tests/regression/input_files/large_tokamak_eval.MFILE.DAT`'s converged operating
point (`rmajor = 8.0`, `rminor = 2.66666666666666652`,
`b_plasma_toroidal_on_axis = 5.31832217464490409`,
`b_plasma_surface_poloidal_average = 0.839681017309652056`) — genuinely legacy in the
sense the harness doc means (a real, already-validated PROCESS operating point), just
sourced from a regression MFILE rather than a `tests/unit` parametrisation. PROCESS's
own reference functions (`PlasmaFields.calculate_plasma_inboard_toroidal_field` etc.,
called in-process) are the `reference=` callables, per this wave's environment
(`process` and `cottax` importable together).

## switches touched

None, in the functions this pass ports. `i_plasma_current` (function 3, out of scope)
is the file's only switch and is untouched here; see "## already ported" above for its
disposition (owned by `physics.py`, Peng arm UNPORTED there).

## calls into other models

None, in functions 4-6. Function 3 (out of scope) calls `self.current.plascar_bpol`
(`plasma_current.py`) on its Peng arm — noted for completeness, not re-derived.

## JAX-difficulty flags

- **No fractional powers, no `safe_pow`/`safe_sqrt` site** in any of the three newly
  ported functions — each is a plain division or a `sqrt` of a sum of squares (the
  latter via the already-`safe_sqrt`-wrapped `calculate_total_field`). Checked by
  inspection, confirmed by `--fp-gradients` (see § test results).
- **`calculate_plasma_inboard_toroidal_field` is singular at `rmajor == rminor`**
  (aspect ratio 1). Not reachable from `ITERATION_VARIABLES`' own bounds for
  `rmajor`/`rminor`/`aspect` at the same time (`aspect = rmajor/rminor`, and no
  combination of the three iteration variables' declared bounds drives `aspect` to
  exactly `1`), and not live on any tracked regression input. Matches PROCESS's own
  unguarded division exactly; not a `safe_*` candidate since the singularity is a
  genuine physical boundary (an aspect-ratio-1 tokamak), not a removable `0**p`
  artefact.
- **`calculate_toroidal_field_profile`'s `n_plasma_profile_elements` is a dynamic-shape
  hazard**, per `_audit/traceability_policy.md` § "Dynamic shape / mutation idioms":
  `2 * n_plasma_profile_elements` sizes a `jnp.linspace` call and must stay a concrete
  Python `int` under `jax.jit`/`jacfwd`, not a traced value. Not a switch (it does not
  select a model variant, it sets a profile *resolution*), but it does need
  `static_argnames` treatment wherever it is eventually wired into a node or a `jacfwd`
  trace. Flagged, not resolved, since the function is not wired to a node in this pass
  (no consumer — see § who reads this).
- **`jnp.where(rho == 0, 1e-10, rho)`** (the magnetic-axis guard) is unchanged from
  PROCESS's own `np.where`; no new JAX hazard, since the profile array's endpoints are
  `rmajor ∓ rminor`, both strictly positive on the physical domain and equal to zero
  only when `rminor == rmajor` — the same aspect-ratio-1 boundary as the inboard
  toroidal field's own singularity, not an independent hazard.

## open questions

1. **Should `TotalMagneticFieldInboard` be wired at all, given it has no reader?**
   Wired here for symmetry with `TotalMagneticFieldOutboard` and because the
   consolidation-pass module's own docstring already anticipated both siblings as
   "one line either way" — but it is a judgement call, not a requirement; the
   consolidation pass may prefer to leave it unwired until a real reader appears,
   matching the choice made for `calculate_toroidal_field_profile`. Flagging rather
   than deciding unilaterally which of the two "no reader yet" cases gets a node.
2. **The ordering bug (§ above) is documented, not raised as a `process/` issue.**
   Per this project's standing convention (`plasma_geometry.md`'s D1-D11), nothing in
   `process/` was touched or filed as a bug report; recording it here is the intended
   remedy. Whether it is worth a `process/` fix independent of this port is outside
   this unit's scope to decide.
3. **`tokamak_boundary.md`'s "zero boundary reads for `.tokamak.plasma_fields`" is now
   stale**, per § who reads this. Not corrected here (regenerating that file is a
   `boundary.py` invocation over the whole assembled graph, not a per-unit edit); flagged
   so the next regeneration is not surprised by the discrepancy.

## deviations from PROCESS

- **The ordering bug is not reproduced** (§ above) — `TotalMagneticFieldInboard`/
  `Outboard` read this same evaluation's toroidal-field output, not PROCESS's
  one-call-stale value. Convergent with PROCESS's own converged answer; divergent from
  PROCESS's literal per-call value on any run that has not yet converged. This is the
  only deviation; every ported pure function is otherwise bit-identical to PROCESS's
  own `@staticmethod`.

## test results

`$PY -m pytest tests/functional_process/models/physics/test_plasma_fields.py`:
**19 passed, 15 skipped** on a plain run (gradient checks skip by default). **34
passed** with `--fp-gradients` — no `_harness/boundary.py` registration needed, no
`safe_pow`/`safe_sqrt` site in any of the three ported functions (§ JAX-difficulty
flags). **118 passed** with `--fp-gradients --fp-fuzz 8`.
