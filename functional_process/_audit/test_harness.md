# Test harness design

How the pure-functional port gets validated against PROCESS. Four tiers, increasing in
both scope and in how hard "agreement" is to even define. Revise as real tier-2/4 units
surface cases the design doesn't cover yet (the worked example below is the first one).

**All four tiers are built.** Tiers 1 and 2 are `functional_process/_harness/`; tiers 3
and 4 are the graph and harness layer — `mda.py`, `mda_harness.py`,
`mda_constraint_harness.py`, `sand.py`, `sand_harness.py`, `total_process.py`,
`configuration.py` — all run by `pytest functional_process` and by the two runnable entry
points `run_mda_harness.py` / `run_sand_harness.py`. That layer has caught bug classes
tiers 1 and 2 structurally cannot; see § As built, which is the current record of what
exists and where the tier descriptions below were refined by building them.

**Precondition, not a parallel activity**: a unit's Phase-0 audit record
(`schema.md`'s model-unit/constraint records) has to exist and its `implicit-io` /
`implicit-io-via-callee` / `redundant-duplicate-write` classifications resolved *before*
a tier-1/2 test can be written for it — the true signature isn't known until then. The
harness consumes audit records, it doesn't precede them.


## Declaration surface — `From` / `OutputInto`

Reads and writes are declared by **area plus the name being declared**:

    helpow = OutputInto(heat_transport)      # -> .heat_transport.helpow

    def __call__(self, tfcryoarea=From(tfcoil)): ...   # -> .tfcoil.tfcryoarea

The parameter name **is** the field name and there is no second spelling — `From` takes an
area and refuses a path, so nothing can be misread. A body wanting a shorter local name
renames in the body. `Input(lambda s: ...)` / `Output(...)` remain for places no name can
spell (elements of `f_nd_impurity_electron_array`), so sugar and escape hatch are told
apart by which function you call.

Areas come from `functional_process/paths.py`, generated from
`dataclasses.fields(DataStructure)` so they cannot drift; `data` is the whole namespace and
refuses a misspelled area at declaration time with a suggestion.

**This changes nothing a case declares.** `audit_record`/`reference`/`ported`/`samples` and
the tier base classes are untouched, and the conversion is provably inert: the pilot
module's 158 ports were compared against the pre-conversion module loaded side by side and
are identical. Converted so far: `power_B_thermal_cryo.py`. The rest is censused in
`next_steps.md`'s Verified-state table.

## Tier 1 — explicit pure functions

Functions with no internal iteration and (once the audit's promotion work is applied) an
explicit signature: no `self.data` access, plain arguments in, plain values out.
Constraints and the objective function belong here too for testing purposes, even though
structurally they're `Compare`/condition shapes rather than `CallableNode`s that own an
output (see `naming_convention.md`) — they're just as pure, just as easy to diff.

- **Value comparison**: call the PROCESS reference and the ported JAX function with the
  same inputs, assert agreement near machine precision. No solver is involved, so there's
  no excuse for a loose tolerance here.
- **Gradient agreement**: assert `jax.jacfwd(ported_fn)` agrees with PROCESS's own
  finite-difference gradient (`Evaluators.fcnvmc2`'s `epsfcn` perturbation, same `epsfcn`
  value) at the same point. This is the actual payoff being purchased by the rewrite —
  a function can match in value and still be wrong in derivative (e.g. a `jnp.where`
  hiding a NaN on the untaken branch), so this check catches what value-only diffing
  can't.
- **`jax_enable_x64` is mandatory**, not a nice-to-have. PROCESS is float64 throughout;
  running the harness under JAX's float32 default produces mismatches that look like
  porting bugs but are precision loss. Set it once, in the harness, not per test.
- **Sampling lives inside the harness**, not as an external manual step: primarily from
  converged operating points in `tests/regression`'s tracked input files (realistic
  domains — many of these functions have sqrt/log domain restrictions that only stay sane
  near actual converged points), secondarily from randomized fuzzing within the bounds
  already declared in `process/core/solver/iteration_variables.py`
  (`ITERATION_VARIABLES[id].lower_bound/upper_bound`).

## Tier 2 — internal solvers over one model

Units whose audit flagged tier 2: an internal iterative loop (`scipy.optimize`/`fsolve`,
or an ad hoc fixed-iteration-count pattern) closing over state local to one model.

**PROCESS's own answer is not ground truth here — this is the tier where value-equality
testing breaks down**, because the "convergence" being reproduced is often not real
convergence. A second, distinct reason has since been found and is worth naming beside it:
**PROCESS's own answer can have a bounded accuracy ceiling independent of approximation
method.** `beam_fusion` is the recorded instance — the reference number itself is only good
to a certain number of digits, so no porting strategy, however faithful, can be validated
past that ceiling. That is a third category alongside "opaque external call" and "PROCESS's
answer isn't ground truth for a non-converged tier-2 loop", and a unit that hits it should
say so in its record rather than loosening a tolerance until it passes.

Two concrete strategies, not mutually exclusive:

1. **Residual-based pass criterion.** Plug both PROCESS's answer and the ported driver's
   answer back into the unit's own defining equations; assert the ported driver's residual
   is small in an absolute/physical sense, and `<=` PROCESS's own residual at its
   stopping point. This sidesteps "whose stopping point is right" entirely — a properly
   convergent driver landing somewhere numerically different from PROCESS's heuristic
   endpoint is expected, not a bug.
2. **Tighten PROCESS's own convergence criteria to build a better reference**, where a
   real value-level comparison is wanted anyway (raise the iteration cap / tighten the
   `rtol`/`xtol` PROCESS itself uses), and compare the port against *that*, not against
   PROCESS's out-of-the-box default.

### Worked example, from the Phase-0 pilot: `power_at_ignition_point`

`process/models/stellarator/density_limits.py:155-217` is the first real tier-2 case
found, and it's a sharper version of the problem above than "PROCESS's loop might not be
fully converged." It deep-copies the entire stellarator model, mutates the copy, and
calls `st_phys()` on it **exactly twice, hardcoded** — the source comment: *"The second
call seems to be necessary for all values to 'converge' (and is sufficient)."* There is
**no convergence check at all**, not even a loose one — two calls, unconditionally,
treated as the answer. (Full audit: `functional_process/models/stellarator/density_limits.md`;
also load-bearing for constraint 91's operands, see `functional_process/core/solver/constraints.md`.)

This is not an edge case to special-case around — value-diffing a real convergence-checked
driver against "whatever two hardcoded calls happened to produce" would almost certainly
fail even for a *correct* port, for reasons that have nothing to do with the port being
wrong. Strategy (1) applies directly: build `st_phys`'s residual (once unit #1's audit
defines it), evaluate it at PROCESS's two-call answer and at the ported driver's
converged answer, and assert the latter is no worse — ideally demonstrably better, as a
positive signal the port is doing real work here rather than just reproducing a shortcut.

## Tier 3 — models calling other models (acyclic)

Composition of already-validated tier-1/2 nodes with no internal solver introduced at the
composition level. The new risk here is wiring, not numerics.

- Mostly **structural assertions**: every declared `In` is bound to something, no
  accidental cross-talk between ports.
- End-to-end value comparison over the subgraph's exposed inputs/outputs, near machine
  precision (no new solver, so tier 1's tolerance logic applies) — reuse existing
  `tests/unit/models/*.py` PROCESS tests as the oracle where one already exists rather
  than writing a new one from scratch.

## Tier 4 — full MDA

The whole coupled graph, cross-model SCCs, driven by `Blocking`+`Schedule`+`Drive`
against PROCESS's ad hoc "call the whole pipeline up to 10x, stop when objective and
constraints agree to `rtol=1e-6`" loop (`Caller.call_models`).

**Tolerance strategy explicitly deferred** — not a Phase-0 or Phase-1 concern. The
solver architectures are different enough (a real driver vs. an unbounded-order
Gauss-Seidel heuristic checked only on two scalar outputs) that a value-level tolerance
decided now would likely be wrong once real tier-4 units exist; this is exactly why the
tier system isolates this problem to one tier instead of letting it contaminate tiers
1-3's much simpler pass criteria. Revisit once tiers 1-3 give confidence in the
individual pieces.

**How it was actually built, which is not what this section predicted.** The plan was to
reuse `tests/regression`'s tracked-reference-data machinery once cottax's `Schedule` sat
where VMCON sits. Instead `mda_harness.converged_data` runs PROCESS's own `SingleRun`
**in-process** and reads `VarPath`s straight off the live `DataStructure` — no MFile
round-trip, no golden file, no tolerance file. That is only possible because `process` and
`cottax` are importable in the same interpreter (`CLAUDE.md`, the `process_port` env), and
it is strictly better: the comparison is per *variable* rather than per reported output, and
a disagreement names the node that produced it.

## Scope note

Stellarator only, per `unit_registry.md`. Tokamak and IFE are separate efforts sharing
only the tier-1 pure-function layer — not parameterized variants of one harness.

## As built

Run with `pytest functional_process` in the `process_port` env (see `../../CLAUDE.md`).

```
functional_process/
  conftest.py                    markers, --fp-fuzz options, sample parametrization
  _harness/
    __init__.py                  enables x64 on import — see below
    contracts.py                 Tier1Contract / Tier2Contract
    finite_difference.py         PROCESS's own FD scheme + its error bar
    sampling.py                  legacy / fuzz / (converged, not yet)
    tolerance.py                 Tolerance objects
    registry.py                  parses unit_registry.md
  models/stellarator/
    density_limits.md            audit record  \
    density_limits.py            the port       > one unit, one stem, three files
    test_density_limits.py       the case      /
  test_registry_coverage.py      meta-tests against the registry
```

**A unit is a stem, not a file.** The record, the port and the case share a name and sit
together, so a stale pair is visible in one `ls` and the record is what you read to write
the other two.

**Tier is a base class.** A unit declares `audit_record`, `reference`, `ported`,
`samples`, and inherits its tier's checks; there are no test functions in a unit's test
module. Tier therefore comes straight from the record's `## tier signal` with no second
decision to keep in sync, and `-k <UnitName>` selects everything about one unit. The
tiers differ in *which tests exist* — `Tier2Contract` has no value-agreement test at
all, so the tier-2 reasoning above is enforced structurally rather than by convention.

Two refinements to what the tier descriptions above say, both found while building it:

- **The tier-1 gradient tolerance is derived per point, not fixed.** `epsfcn` defaults to
  `1e-3`, a *relative* step, so PROCESS's own central difference carries a truncation
  error around `1e-6` relative — any fixed `rtol` is a coin flip on the local curvature.
  `fd_gradient_with_error` estimates the reference's own error by Richardson
  extrapolation (`(4/3)|D(h) - D(h/2)|`, plus a round-off floor) and the assertion is
  "agrees to within the reference's error bar × a safety factor". This self-calibrates:
  on the pilot it tightens to `~1e-13` in directions where the function is linear and
  loosens automatically at a `min()` kink.
- **Domain guards are a declared expectation, not an escape.** Where PROCESS raises on
  an out-of-domain input, the port must return non-finite instead — a traced function
  cannot raise on a data-dependent condition. A contract declares
  `reference_domain_errors`, and the value test then *asserts* the port is non-finite
  wherever PROCESS raises, rather than skipping. The other two checks skip such points.

**x64 is enabled at import of `_harness`, not in a fixture.** A session fixture runs
after collection has already imported the port modules, by which point an array built
during import would be float32 for good.

**Sample provenance is part of the test id** (`legacy-…`, `fuzz-seed0-003`), so `-k
legacy` and `-k fuzz` select between them and a fuzz failure is reproducible from its id.
`legacy` points are lifted from PROCESS's own `tests/unit`, which are already validated
input/output pairs generated from real stellarator input files — a free oracle rather
than a corpus to re-derive. `converged` is not implemented and raises rather than
skipping quietly.

**Validated by mutation, not by passing.** Two deliberate bugs were injected into the
pilot port to confirm the checks bite: perturbing a coefficient by `4e-7` relative failed
20 tests, and wrapping one input in `jax.lax.stop_gradient` — a bug with *no* effect on
any value — failed 10 tests, all of them `test_gradient_agreement`, with every value test
still passing. That second one is the case for the gradient tier existing at all.

### The tier-3/4 layer, and what it catches that tiers 1-2 cannot

Tier 3's structural assertions ("every `In` is bound", no cross-talk) turned out to be
cottax's job, as predicted — `Graph` construction refuses an unbound `In`, so there is no
test to write here. What the tier-3/4 layer actually buys is a different, and larger, list:
**a binding can be wrong while every pure function it binds is right**, and only running the
assembled graph sees it. Real instances, each recorded in `next_steps.md` §8: a 1-tuple
return from a single-`Output` node (eight instances); a static switch kwarg copied from a
`*_variables.py` default rather than from the run being modelled (four); a correct function
bound to the wrong `VarPath` (`q95` vs `iotabar`); and **a missing producer, where every
value is right and only a derivative is wrong** (three).

**Three declared policies carry what cannot be compared**, and all three are deliberately
documentation rather than suppression — a per-field tolerance would mask the next real
regression:

- `KNOWN_UNVERIFIABLE_OUTPUTS` — a `VarPath` PROCESS's own pipeline never writes on this
  device path, so the "expected" value is a dataclass default, not an answer.
- `EXPLAINED_DISAGREEMENTS` — a disagreement with a named, understood cause (the
  `VacuumOld` duct-diameter tolerance; PROCESS's `+17.604 MW` report-pass offset). Recorded,
  reported, and **not** subtracted from the comparison.
- `KNOWN_MINT_VALUES` — a mint reconstructed from real stored fields by an identity read off
  PROCESS's own source, which turns an unscorable node into a scored one.

**A fourth category is owed and does not exist yet: PROCESS's report-pass/solve-pass
inconsistency.** For any field PROCESS's `output=True` pass recomputes, the harness's
`expected` column is PROCESS's *reported* state, which is not the state its solver used —
`.build.z_tf_inside_half` and `.buildings.a_plant_floor_effective` are the measured
instance. Nothing detects this in general; a field is noticed only when a consumer of it
disagrees. It belongs beside the three policies above, as a check rather than a list.

**The accounting invariant.** `ComparisonReport` carries `owned_total` and `unaccounted`
and enforces that **every owned variable lands in exactly one bucket** (agreement /
disagreement / unverifiable / error). This exists because it did not: array-valued outputs
used to be dropped inside a bare `except: continue` *before* any bookkeeping, and the hole
was 25 variables — 21 silently agreeing and 4 silently disagreeing. Arrays are now compared
elementwise and reported as one disagreement carrying the worst element
(`shape`/`index`/`n_off`), and a non-comparable pair becomes an explicit error. **Any future
comparison must be accounted for the same way**: a bucket that can silently drop a case is
worse than no check, because it reads as a pass.

**A blind spot of the same family, still open.** `compare`'s `atol=1e-9` makes any field
whose natural magnitude is below that **vacuously agree**.
`.neoclassics.temperatures`/`dr_temperatures` are stored in Joules (~1e-15), so they agreed
both before and after a fix that materially changed them, and are **not actually checked by
anything**. A per-field relative floor, or a unit-aware scale, is what it needs.

### Not built

`converged` sampling (it raises rather than skipping quietly), and a cold-start path: every
tier-3/4 harness seeds from a converged PROCESS run, so nothing yet validates the graph
from a cold input file (`optimise_design.md` §10.6).
