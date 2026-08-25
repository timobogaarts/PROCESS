# JAX-traceability policy (draft)

Standing decisions so per-unit audits don't each re-litigate them. These are Phase-0
*defaults for flagging*, not implementation commitments — nothing here is coded yet.

## Precision

`x64` on, always. PROCESS is float64 throughout; a diff test between a ported function
and PROCESS run under default JAX float32 will show spurious mismatches that look like
porting bugs but are precision loss. Every pure-function audit record's proposed
signature assumes `jax.config.update("jax_enable_x64", True)`.

## Switches — the split default

**"Switch" means any parameter that changes which nodes run, not only a
`data.<area>.i_*` field.** Found via `stellarator.py`'s `run(output: bool)`: a plain
Python bool, not a `DataStructure` field, but it selects between two call sequences with
entirely different reads/writes (see
`functional_process/_audit/units/models/stellarator/stellarator_A_orchestration.md` —
`output=True` and `output=False` don't even call the same set of stellarator
submodules). It fails the split-default's own criterion exactly as a `data.*` switch
would, so it gets the same treatment: the pure port should have two separate top-level
functions (e.g. `compute` and `report`), not one function with an `if output:` branch
threaded through it. Anything whose value is read once to decide *which nodes exist* (a
Python kwarg, a config value, a `data.*` switch — the mechanism doesn't matter) is in
scope for this section.

**Default: split.** A switch whose branches read different `VarPath` sets (the common
case — PROCESS's alternate formulas are almost always alternate physics with different
physical dependencies, e.g. `i_density_limit`'s 8 formulas each use a different subset of
inputs) becomes separate ported functions/nodes, one per value, chosen at graph-build
time (see `naming_convention.md`'s "switches are not ports").

**Exception: static kwarg**, only when the per-branch reads-sets are **provably
identical** (not just "conceptually the same role") — e.g. a genuine solver-method
choice or output-formatting toggle with no physical-input difference. The audit record
must show the reads-set comparison, not just assert equivalence.

Every switch touched by an audited unit gets a row in the registry with this decision
and the reads-set evidence, even if the decision is provisional pending other units that
touch the same switch.

### The rule is stated unconditionally; six recorded instances deviate from it deliberately

This is not drift and it is not six oversights — each was justified in its own record, none
was reversed, and together they are evidence that the rule as written is missing a clause.
The deviations are: `i_confinement_time` and `i_rad_loss` (`confinement_time.md`),
`i_plasma_ignited` (`confinement_time.md` and `composition.md`, independently),
`supercond_cost_model` (two nodes), `i_pf_conductor`, and `itart` on `CostOfElectricity`
(all `costs.md`). In every one the branches' reads-sets **do** differ — so the split default
applies on its face — but the differing part of the body is a handful of lines inside a
large shared one (2–6 lines inside a 48-branch dispatcher; 15 lines of a 290-line function),
and splitting would duplicate the shared remainder.

**The question is now well-posed**, which it was not when this file was written, because the
contrast cases exist and two of them are in the same file, ported the same day, by the same
reasoning:

- `i_tf_sc_mat` (`superconductors.md`) — **split**, 8 genuinely different reads-sets and no
  shared body to speak of: one function per branch.
- `costs.py`'s `acc2221` — **split** (two arms, no shared body, disjoint reads) while the
  same file's `coelc` kept `itart` **static** (15 lines of 290, ~275 shared).

So the missing clause is about **the size or entanglement of the shared remainder**, not
about whether the reads-sets differ — something of the shape "split only when the differing
body exceeds N lines or M% of the function, or when the branches share no body at all".
Deciding it is a policy call, not another per-unit judgement; until it is made, each new unit
re-derives it independently. `next_steps.md` §1 and §9 track the same question from the
`Switch`/`Alternative` side, and `switch_elimination_design.md` §2 reads the deviation count
as the enforcement gap it is.

## Non-traceable external calls

CoolProp calls (`process/core/coolprop_interface.py`) and anything else that reaches an
opaque external library are flagged `non-traceable-external-call` in the audit record.
**Not resolved in Phase 0 or Phase 1** — no custom JAX primitive, no `pure_callback`
wrapper decided yet. Just flag precisely: which function, which inputs/outputs cross the
boundary, so the eventual resolution (primitive wrapper vs. re-fit vs. keep as a
non-differentiated island) has an accurate list to work from.

## Implicit vs. explicit reads

Every `self.data.<area>.<field>` access inside an audited unit is classified as one of:

- **`explicit-arg`** — read once, used as an ordinary parameter, no dependency on
  execution order within the call. Promotes directly to a function argument.
- **`implicit-io`** — read mid-loop, depends on state written earlier in the *same*
  call (by this unit or one it calls), or otherwise not resolvable to "read once at the
  top." These are the ones that actually need a careful read, not a grep — see
  `unit_registry.md` for who does this (agents draft with a confidence field; low-
  confidence or physics-heavy ones get flagged for review) and why (this *is* the
  mechanism that surfaces hidden non-local dependencies, per the original difficulty
  list in `../CLAUDE.md`).

## Dynamic shape / mutation idioms

- **Fixed-size arrays with an "active count."** PROCESS already Fortran-fixed-sizes most
  arrays (e.g. `f_nd_impurity_electron_array`) with a separate count field rather than
  variable-length lists — keep this shape. Pass the full fixed-size array plus an
  explicit count (static or traced as appropriate), don't dynamically slice.
- **In-place sequential mutation** (`array[i] = ...` inside a loop) must become
  `.at[i].set(...)` or a vectorised form when ported, and is worth flagging with extra
  scrutiny in the audit record even at Phase 0 — a loop like this often signals either a
  `lax.scan`-shaped recurrence (worth calling out explicitly) or a reduction that a
  vectorised rewrite would simplify, and knowing which one it is now saves a wrong first
  attempt at porting.
- **Python control flow branching on a traced value** (`if <computed from data> > x:` on
  a non-switch, non-static quantity) is flagged `needs-lax-cond-or-where`; distinguish
  this from a switch branch, which is a *static*-argument branch, not this.
