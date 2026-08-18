---
kind: model-unit
status: draft
confidence: medium
---

## source
`process/models/physics/physics.py`, registry unit #9, chunk B:
- `Physics.plasma_composition`, lines 1166-1491.
- `Physics.calculate_effective_charge_ionisation_profiles`, lines 1749-1781.

## scope note: why these two are one chunk, separate from chunk A
Both call into `process/models/physics/impurity_radiation.py`
(`calculate_average_charge_at_temp`/`element2index`, registry unit #23) and were
therefore flagged **blocked** in `unit_registry.md` row #9 pending that unit's audit,
which — per the task brief for this dispatch — was being done by a parallel agent at
the same time as this one. Unlike chunk A's five functions (already pure in the
source), both of these read and write `self.data` extensively and needed a real port,
not a translation, so they are grouped separately.

## scope correction: unit #23 unblocked mid-session

Unit #23's audit record and port (`functional_process/models/physics/impurity_radiation.md`/`.py`)
landed in this same directory before this chunk was written. Its port supplies exactly
the two functions this chunk needs, already pure and JAX-traceable:
`calculate_average_charge_at_temp(temp_electron_kev, temp_impurity_kev,
impurity_arr_zav)` and `element2index(element, impurity_arr_label)` (the latter
explicitly documented there as a graph-assembly-time, non-traced lookup — see that
file's docstring). This chunk imports and uses `calculate_average_charge_at_temp`
directly; `element2index` is not called at all (see below) — its *result* is used,
resolved to a Python literal, because the label→index mapping is a compile-time
constant for every real PROCESS `DataStructure` (see the header note in
`physics_B_composition.py` and the index table below). Both functions are ported and
harness-tested there with `status: draft` — the same status this record carries, so if
that record moves, so must the constants documented below.

**This changes the task brief's expectation.** The brief anticipated this chunk would
stay audit-only ("if you hit one of those, note the dependency but do not wait...
that's a fine, expected outcome"); instead both functions are ported below, because the
blocking dependency resolved during this same session rather than staying open.

## the `first_call` self-loop

`plasma_composition` reads `.physics.first_call` once (to select between two formulas
for an intermediate, `pc`) and, **only inside the `first_call == 1` branch**, writes it
back to `0` (`physics.py:1381,1387`) — the other branch leaves the field alone, so
`first_call_next` is not the constant `0` a first read of the source suggests: it is `0`
on the bootstrap branch and an **identity pass-through** of the input on the other
(`d(first_call_next)/d(first_call) == 1` there). Caught by this record's own
`test_gradient_agreement`, not by inspection — an earlier draft of the port wrote an
unconditional `0` and passed every value check while failing exactly this one gradient,
the same class of bug `_audit/test_harness.md`'s pilot retrospective describes. This is
structurally a **self-loop** regardless: one `VarPath`, read and written by the same
node. `~/jaxgraph/CLAUDE.md` ("The graph"): *"A node may not read what it owns... so a
fixed point is always written with a minted copy, and a cycle is always at least two
nodes."* A single `cottax` node cannot own `.physics.first_call` and also read it — the
port (`plasma_composition` the function) exposes `first_call` as an ordinary input and
`first_call_next` (the pass-through/reset described above) as an
ordinary output, but **no node class is written for this function in this pass**,
because wiring that pair back onto one `VarPath` needs the same `Cut` treatment
`~/jaxgraph/CLAUDE.md`'s mapping table describes for PROCESS's other cross-call state
flags — a graph-assembly-time decision, not a per-node one, and out of this record's
authority (it touches `total_process.py`, off-limits per this dispatch's boundary).

**Third instance of this exact pattern**, not the first:
`functional_process/models/stellarator/stellarator_A_orchestration.md` already flagged
two — `data.stellarator.first_call` (recommending a new schema category,
`implicit-io-across-calls`) and the separate `self.first_call_stfwbs` instance attribute
— and `stellarator_E_fwbs_synthesis.md`'s S3 confirmed the second is a genuine two-node
SCC needing `Blocking`+`FixedPoint`. `.physics.first_call` is a **third, independent**
data field with the identical shape (read once, unconditionally overwritten with a
constant, used only to distinguish "this is the very first evaluation of this run from
every subsequent one"). Recommend the coordinating session add the
`implicit-io-across-calls` classification to `schema.md` now that three real, unrelated
instances exist — this record is further evidence, not a fourth ad hoc workaround.

Sketch of what the node would look like once `Cut` is available (**not implemented**):
```python
# plasma_composition reads .physics.first_call and would need to write the *next*
# value onto a minted copy, e.g. ^hat.physics.first_call, with the boundary/graph-
# assembly step responsible for feeding next run's `first_call` from this run's
# ^hat.physics.first_call -- exactly the Cut(var=.physics.first_call, readers=...)
# shape, not resolvable inside one node's __call__.
```

## the `znfuel` raise

`physics.py:1270-1271`: `if znfuel < 0.0: raise ProcessValueError(...)`. Unlike the
`sqrt`/`log`-domain guards elsewhere in this codebase (where an invalid input produces a
NaN/Inf *automatically*, and the port simply lets that happen instead of raising),
`znfuel` and everything downstream of it (`nd_plasma_fuel_ions_vol_avg`, the `H_`/`He_`
array writes, `nd_plasma_ions_total_vol_avg`, ...) remain **well-defined finite
numbers** when this fires — just physically nonsensical ones (a negative fuel ion
density). `test_harness.md`'s domain-guard convention ("the port must return non-finite
instead of raising") has no natural target here: forcing a NaN would mean adding a
`jnp.where` whose sole purpose is manufacturing a NaN where the arithmetic itself
produces none, which does not obviously belong in a *pure* port of the arithmetic
itself versus in a downstream validity check. **Not ported.** Flagged, not resolved —
see open questions.

## data footprint

### `plasma_composition`

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.nd_plasma_electrons_vol_avg` | read | explicit-arg | read repeatedly, value never changes within the function |
| `.physics.f_nd_alpha_thermal_electron` | read | explicit-arg | |
| `.physics.nd_plasma_alphas_thermal_vol_avg` | write then read | local-intermediate | written at the very top, unconditionally, then read several more times straight-line |
| `.physics.fusden_alpha_total` | read | explicit-arg | branch selector (`< 1e-6`) and, on the other branch, a divisor |
| `.physics.f_nd_protium_electrons` | read | explicit-arg | |
| `.physics.nd_plasma_protons_vol_avg` | write then read | local-intermediate | |
| `.physics.i_plasma_ignited` | read | explicit-arg, **switch** | see switches section — kept as static `is_ignited` bool |
| `.physics.f_nd_beam_electron` | read | explicit-arg | non-ignited branch only |
| `.physics.nd_beam_ions` | write then read | local-intermediate | |
| `.impurity_radiation.impurity_arr_z` | read | **not a port argument** | used only to build the `z > 2` mask, which is provably constant across every real configuration — see `IMPURITY_SLICE` in the port; not threaded through as a value |
| `.impurity_radiation.f_nd_impurity_electron_array` | read, then written at indices 0/1, then read again (all 14, and slice 2:14) | **mixed** — see note | read for the initial `znimp` sum (indices 2:14, pre-update — no overlap with the 0/1 write), written at indices 0 (`H_`) and 1 (`He`) via a `local-intermediate` computation, then re-read including the *updated* 0/1 entries for `n_charge_plasma_effective_vol_avg` (the one accumulator not gated by `z > 2`) — order-sensitive, confirmed by reading the source top-to-bottom, not merely assumed |
| `.physics.temp_plasma_electron_vol_avg_kev` | read | explicit-arg | single scalar, evaluated against every species' table |
| `.impurity_radiation.temp_impurity_keV_array`, `.impurity_radiation.impurity_arr_zav` | read | explicit-arg (constant) | `(14, 200)` each, compile-time |
| `.physics.f_plasma_fuel_deuterium`, `.f_plasma_fuel_tritium`, `.f_plasma_fuel_helium3` | read | explicit-arg | |
| `.physics.first_call` | read, then written | **self-loop** — see dedicated section above | not classifiable under the five existing schema categories; evidence for a new `implicit-io-across-calls` category |
| `.physics.alphan`, `.physics.alphat` | read | explicit-arg | `first_call`-branch only |
| `.physics.f_temp_plasma_electron_density_vol_avg` | read | explicit-arg | non-`first_call`-branch only; producer is `physics/plasma_profiles.py` (registry unit #12, already ported) |
| `.current_drive.f_beam_tritium` | read | explicit-arg | cross-area read |
| `.impurity_radiation.m_impurity_amu_array` | read | explicit-arg (constant) | `(14,)` |
| `.physics.n_charge_plasma_effective_vol_avg`, `.nd_plasma_impurities_vol_avg`, `.nd_plasma_ions_total_vol_avg`, `.f_nd_plasma_carbon_electron`, `.f_nd_plasma_oxygen_electron`, `.f_nd_plasma_iron_argon_electron`, `.f_alpha_electron`, `.f_alpha_ion`, `.m_fuel_amu`, `.m_beam_amu`, `.m_ions_total_amu`, `.n_charge_plasma_effective_mass_weighted_vol_avg` | write | explicit-arg | ordinary outputs, no further read within this function |
| `.physics.nd_plasma_fuel_ions_vol_avg` | write then read | local-intermediate | see `znfuel` note above re: the un-ported domain check |

### `calculate_effective_charge_ionisation_profiles`

| VarPath | read/write | classification | note |
|---|---|---|---|
| `plasma_profile.teprofile.profile_y` (object attribute, no `VarPath`) | read | explicit-arg | reused as `.physics.temp_plasma_electron_profile_kev` — already minted by unit #20/#21, see "cottax node" |
| `.impurity_radiation.f_nd_impurity_electron_array` | read | explicit-arg | whole `(14,)` array |
| `.impurity_radiation.temp_impurity_keV_array`, `.impurity_arr_zav` | read | explicit-arg (constant) | as above |
| `.physics.n_charge_plasma_effective_profile` | write | explicit-arg | `(n_points,)` |
| `.impurity_radiation.n_charge_impurity_profile` | write | explicit-arg | `(14, n_points)` |

## proposed signature(s)
```python
def plasma_composition(
    nd_plasma_electrons_vol_avg, f_nd_alpha_thermal_electron, fusden_alpha_total,
    f_nd_protium_electrons, proton_rate_density, f_nd_beam_electron,
    f_nd_impurity_electron_array, temp_plasma_electron_vol_avg_kev,
    temp_impurity_keV_array, impurity_arr_zav, f_plasma_fuel_deuterium,
    f_plasma_fuel_tritium, f_plasma_fuel_helium3, first_call, alphan, alphat,
    f_temp_plasma_electron_density_vol_avg, f_beam_tritium, m_impurity_amu_array,
    is_ignited,
) -> tuple[...]: ...

def calculate_effective_charge_ionisation_profiles(
    temp_electron_profile_kev, f_nd_impurity_electron_array,
    temp_impurity_keV_array, impurity_arr_zav,
) -> tuple[jax.Array, jax.Array]: ...
```
Implemented in `physics_B_composition.py`.

## cottax node
`CalculateEffectiveChargeIonisationProfiles` (`ExplicitFunction`) only — see "the
`first_call` self-loop" above for why `plasma_composition` gets no node class in this
pass, only the ported function.

## tier signal
1 (explicit pure function) for both, **with two caveats that keep them from being a
"clean" tier-1 in the sense chunk A's five functions are**: `plasma_composition`'s
`znfuel` domain check is unported (see above) and its node is deferred pending a `Cut`.
Neither caveat makes either function iterative or dependent on an unconverged internal
solve — no `scipy.optimize`, no fixed-iteration loop — so tier 1 remains the correct
classification, not tier 2.

## switches touched
- `.physics.i_plasma_ignited` (`PlasmaIgnitionModel`, values `NON_IGNITED`/`IGNITED`):
  the two branches read genuinely different variables (`f_nd_beam_electron` +
  `nd_plasma_electrons_vol_avg` vs. nothing), so `traceability_policy.md`'s default
  ("split") technically applies — **not followed here**, and flagged as a deliberate
  policy deviation rather than silently applied: the differing branch is two lines deep
  inside an otherwise-identical 328-line function, and splitting would duplicate the
  other ~95% of the body across two top-level functions for a two-line difference.
  Kept as a static `is_ignited: bool` parameter instead. Recommend the coordinating
  session treat this as a second data point (alongside `i_beta_fast_alpha` in chunk A,
  a case where the policy's "identical reads" exception cleanly applied) for whether
  `traceability_policy.md`'s switch-split default needs a size/entanglement-aware
  exception, not just a reads-set-equality one.

## calls into other models
`impurity_radiation.calculate_average_charge_at_temp` (registry unit #23, now ported —
see above). No other calls.

## JAX-difficulty flags
- **`needs-lax-cond-or-where`**: `protons_not_yet_calculated`
  (`fusden_alpha_total < 1e-6`) and `first_call_is_bootstrap` (`first_call == 1`), both
  `jnp.where`. Severity workaround-known.
- **Denominator-guard-required** (same category flagged in `physics_A_pure_formulas.md`
  — recommend the shared name land in `traceability_policy.md`): the
  `protons_not_yet_calculated` branch divides by `fusden_alpha_total`, which is
  exactly what selects the *other* branch when it is small/zero — guarded the same way
  as chunk A's instances. Severity workaround-known.
- **`needs-dynamic-array-scatter`** (new flag): `f_nd_impurity_electron_array.at[0/1].set(...)`
  replaces PROCESS's in-place index assignment via a string-label lookup
  (`element2index`). Resolved by hoisting the two indices to Python `int` literals
  (`H_INDEX`/`HE_INDEX`), justified in the module docstring — not a `.at[]` write at a
  *traced* index, so no real JAX difficulty remains once the indices are known. Severity
  minor (resolved).
- **Un-ported domain check** (see dedicated section above): `znfuel < 0` is not
  ported. Severity **blocker** for full raise-parity with PROCESS; **not a blocker** for
  the port's own correctness on any input where the check would not have fired (the
  arithmetic is otherwise identical) — flagged, not resolved, see open questions.
- **First-call self-loop, no node minted** (see dedicated section above). Severity
  **blocker** for graph registration of `plasma_composition` specifically; does not
  block the ported function itself, which is fully tested.
- **Step-function-at-its-own-jump** (new flag): `first_call`'s `jnp.where(first_call ==
  1, 0, first_call)` is a discrete `{0, 1}` flag evaluated exactly at its own decision
  boundary in the `large_tokamak_nof-first_call` sample (`first_call=1`). PROCESS's own
  central finite difference there estimates the slope of the *neighbouring* branch, not
  this branch's true (zero) analytic derivative — the two are not comparable at that
  point, through no fault of the port (confirmed: every value/finiteness check passes;
  only `test_gradient_agreement` disagrees, and only in the direction the discontinuity
  predicts). Resolved by declaring `first_call` a `static_argnames` entry — it is
  bookkeeping state, not a continuous design quantity, so excluding it from
  differentiation is consistent with `i_beta_fast_alpha`/`is_ignited`'s treatment
  elsewhere in this dispatch, not a special case invented for this failure. Severity
  minor (resolved); worth recording as a second, independent motivation (beyond "it's a
  switch") for excluding an argument from `test_gradient_agreement` — the existing
  `static_argnames` docstring only names the switch case.

## open questions
1. Should `znfuel < 0` become a real assertion somewhere in the eventual graph (e.g. a
   `Compare`-shaped sink a solver's constraints can see), rather than silently dropped?
   PROCESS treats it as fatal; a pure port that lets it flow through silently changes
   behaviour at exactly the input PROCESS considered invalid. Not resolved here —
   needs a decision about where domain validity checks live in the eventual graph, not
   a per-unit call.
2. `.physics.first_call`'s `Cut` treatment — same open question `stellarator_A_orchestration.md`
   already raised for its own two instances, now with a third data point. Whether all
   three should share one graph-assembly convention (a `first_call` root namespace?) or
   be resolved independently is the coordinating session's call.
3. Confirm with unit #23's own maintainers (if that record changes) that
   `element2index`'s label order stays `H_, He, Be, C_, N_, O_, Ne, Si, Ar, Fe, Ni, Kr,
   Xe, W_` — this record's hardcoded indices (`H_INDEX=0`, `HE_INDEX=1`,
   `CARBON_INDEX=3`, `OXYGEN_INDEX=5`, `ARGON_INDEX=8`, `IRON_INDEX=9`) depend on it,
   traced against `initialise_imprad`'s literal call order in
   `impurity_radiation.py:27-376`, not merely the default array literal (the two agree,
   confirmed by reading both).
