# Constraint audit records

One section per constraint, schema in `../../_audit/schema.md` § Constraint record.
Only stellarator-relevant constraints are in scope for now (see `_audit/unit_registry.md`).

---
kind: constraint
status: draft
confidence: medium-high (structural classification), medium (physics-correctness judgment)
---

## Correction to unit_registry.md's pilot row

The registry's "Constraint 17" row was mislabeled. **Constraint 17 is a general
constraint** (plasma radiation fraction upper limit, used for both tokamak and
stellarator) that merely has an `istell != 0` special-case branch — it does **not**
reference `powerht_constraint`/`powerscaling_constraint`. **Constraint 91 is the actual
stellarator-specific constraint** referencing those fields, and its docstring says so
explicitly ("stellarators only (but in principle usable also for tokamaks)"). Both are
recorded below; the full audit (data footprint, hole-in-MDA, etc.) is for **Constraint
91**, matching what the registry's pilot description actually intended. Please update
`_audit/unit_registry.md`'s Constraints table to reference 91, not 17, as the pilot unit
— or add 91 as a second row if you want 17 kept as a lightweight example of the same
`istell`-branch pattern.

Also worth noting for `switches.md`'s `istell` entry: constraint 17
(`constraints.py:661`) and constraint 24 (beta limit, `constraints.py:803-806`, not
otherwise in scope here) both special-case on `istell != 0` inside an otherwise-general
constraint, rather than being separate stellarator-only constraints. That's a third shape
beyond "shared switch site" and "stellarator-only constraint": a general constraint with
an embedded stellarator branch. Worth its own note in the switch record for `istell`
rather than assuming all `istell` sites are either "whole different constraint" or
"whole different model."

### Constraint 17 (general, not stellarator-specific — recorded briefly for context)
source: `process/core/solver/constraints.py:653-674`
Plasma radiation fraction upper limit (`f_p_plasma_separatrix_rad <=
f_p_plasma_separatrix_rad_max`). When `istell != 0`, subtracts a SOL radiation
contribution (`psolradmw / p_plasma_heating_total_mw`) from the fraction before
comparing — a documented-as-uncertain adjustment (`# TODO: this is replicating behaviour
before #4299 / is this really what should happen?` in the source). Not audited further
here — out of this unit's scope; flagged for whoever picks up general (non-stellarator)
constraint 17 later, since the `istell` branch itself IS in scope for the stellarator
audit and should be captured wherever `switches.md`'s `istell` entry lands.

---

### Constraint 91: ECRH ignition heating-power lower limit
**source**: `process/core/solver/constraints.py:1912-1938`, docstring: "Lower limit to
ensure ECRH te is greater than required te for ignition at lower values for n and B... 
stellarators only (but in principle usable also for tokamaks)."

**calls**: no direct function call inside `constraint_equation_91` itself — it only
reads `data` fields and calls `eq_geq`'s `geq(...)`. But its two operands
(`data.stellarator.powerht_constraint`, `data.stellarator.powerscaling_constraint`) are
**not free values** — they are written by exactly one producer:
`power_at_ignition_point(stellarator, max_gyrotron_frequency, te0_ecrh_achievable)`
(`process/models/stellarator/density_limits.py:155-217`), called unconditionally from
`Stellarator.run()` at `process/models/stellarator/stellarator.py:178-187`. **That
producer's own footprint is out of scope for this record** — it's model-unit #3
(`density_limits.py`), audited in parallel; see
`functional_process/models/stellarator/density_limits.md`. Flagging here because it
matters for the hole-in-MDA judgment below: **worth noting now, before that record
lands**, `power_at_ignition_point` deep-copies the entire `stellarator` model object
(`copy.deepcopy(stellarator)`, `density_limits.py:185`) and calls `st_phys()` on the copy
**twice** ("The second call seems to be necessary for all values to 'converge' (and is
sufficient)" — an unverified, hard-coded fixed-point-by-two-iterations, no convergence
check) to compute a counterfactual operating point. This is a `blocker`-severity
JAX-difficulty pattern (whole-state deepcopy + repeated full physics re-evaluation as a
counterfactual sub-computation) that the density_limits.py audit needs to address
directly; recorded here too since it's load-bearing for this constraint's true reads-set.

**data footprint** (direct reads of `constraint_equation_91` only):
| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.i_plasma_ignited` | read | explicit-arg | selects which of two `value` expressions to use — see traceability_policy.md, this is a formula-shaped branch on an enum switch, not an implicit-io concern |
| `.current_drive.p_hcd_primary_extra_heat_mw` | read | explicit-arg | only used in the `NON_IGNITED` branch |
| `.stellarator.powerht_constraint` | read | explicit-arg (at this call site) | **but see note above — this value's own production is implicit/heavy**; treat this constraint's signature as taking `powerht_constraint`/`powerscaling_constraint` as plain float args, and treat the deepcopy/double-`st_phys` machinery as a separate node's problem, not this constraint's |
| `.stellarator.powerscaling_constraint` | read | explicit-arg (at this call site) | same as above |

**proposed signature**:
```python
def constraint_91(
    i_plasma_ignited: int,  # PlasmaIgnitionModel, static
    p_hcd_primary_extra_heat_mw: float,
    powerht_constraint: float,
    powerscaling_constraint: float,
) -> ConstraintResult:
    value = powerht_constraint + (
        p_hcd_primary_extra_heat_mw
        if PlasmaIgnitionModel(i_plasma_ignited) == PlasmaIgnitionModel.NON_IGNITED
        else 0.0
    )
    return geq(value, powerscaling_constraint, ...)
```

**hole-in-MDA**: **No — not a hole.** Both operands are already fully forward-computed
(by `power_at_ignition_point`, called unconditionally every `Stellarator.run()`, not
gated behind whether constraint 91 is even active in `icc` — the code comment at
`stellarator.py:179` ("If the respective constraint equation is not called, do not set
the values") suggests someone *intended* to skip this when the constraint isn't active,
but the call at line 183 is unconditional, so the comment doesn't match current
behaviour; not filing this as a MDA-hole, filing it as a separate note below). There is
no missing producer and no free iteration variable standing in for an unwired
relationship — this is an ordinary physical feasibility inequality (is the ECRH-driven
heating power at a hypothetical operating point at least the required scaling/loss
power?), structurally a plain `Compare`/condition node reading two already-produced
quantities. Confidence: **medium-high** on the structural classification (I traced the
producer definitively); **medium** on whether this is the physically "right" constraint
formulation, which is beyond what a code read alone can confirm.

**Separate note (not a hole-in-MDA finding, but worth carrying forward)**: the mismatch
between the comment at `stellarator.py:179` ("if the respective constraint equation is
not called, do not set the values") and the unconditional call at line 183 means
`power_at_ignition_point` — with its expensive deepcopy + double-`st_phys` — runs on
**every** stellarator evaluation regardless of whether constraint 91 is active. Whether
that's a genuine bug or the comment is just stale is outside this record's scope, but a
pure-functional port has an opportunity here: since the port makes `Graph.prune(wanted)`
explicit, this node would naturally only be included when constraint 91's condition is
actually wanted — cottax's own `prune` was designed to sidestep exactly this "always
compute in case it's needed" problem.

**current closure mechanism**: VMCON-joint. No local solver — confirmed by reading the
constraint function; it does nothing but read `data` and return a residual, same as every
other `ConstraintManager`-registered function. (The apparent "internal solve" is inside
`power_at_ignition_point`'s producer, not this constraint — see note above; that's a
different question, "how is `powerht_constraint` itself computed," not "how is this
constraint closed.")

**candidate iteration variable(s)**: **`te0_ecrh_achievable`** (ID 169, module
`stellarator`, bounds 1.0–1.0e3, `process/core/solver/iteration_variables.py:233`) —
stronger than usual best-effort evidence for this one: it's literally the
`te0_available` argument passed into `power_at_ignition_point`, which is the sole
producer of `powerht_constraint`, one of this constraint's two operands. Raising
`te0_ecrh_achievable` raises `powerht_constraint` (higher achievable temperature → more
achievable heating power), which is exactly the direction that satisfies this `>=`
constraint — a plausible, evidenced pairing, though still not authoritative (VMCON solves
jointly; nothing in the code declares this pairing explicitly). Iteration variable 176
(`f_st_coil_aspect`) is unrelated to this constraint.
