# Boundary (unowned) inputs of the driven graph — are they all genuinely inputs?

**Question asked**: of the reference stellarator graph's 379 unowned inputs, 300 aren't
set by the input file. How many of those are actually genuine PROCESS inputs versus
missing producers, and does closing any create a cycle?

**Short answer: 12 out of 300, not 300 and not the 93 a cruder filter finds.** The other
288 are either never assigned anywhere in `process/` (207, genuine inputs), assigned only
to a constant at initialisation (35), or assigned only on a code path this configuration
never takes (46) — narrowed by an AST walk over every assignment target in `process/`,
not a regex (a regex both misses tuple-unpack writes and false-counts f-string
interpolations of a field name as a write). **None of the 12 hides a cross-subsystem
cycle** — checked by node-level reachability, not inspection; the one that touches a
cycle at all lands inside an SCC `mda.CUTS` already cuts.

## Two things found on the way, both worse than a missing producer, both since fixed

- **A boundary input bound to the wrong field.** `ProfileValues.rho` read
  `.neoclassics.r_eff`, a field PROCESS writes **nowhere** in the whole codebase — its one
  call site passes the literal `0.6` directly, never storing it, so the port's binding
  resolved at run time to the untouched dataclass default `0.0` instead. Not cosmetic: the
  profile's radial derivatives are identically zero at `rho=0` and nonzero at `0.6`, so
  everything downstream of `.neoclassics.dr_densities`/`dr_temperatures` was computed at
  the wrong point in the profile. Same bug class as an earlier `q95`/`iotabar` defect: a
  literal PROCESS passes directly to a function, misread as if it were a stored field.
  Fixed by making `rho` a static `0.6` on the node instead of an `Input`.
- **`mda_harness.compare` silently dropped every array-valued output from its own
  bookkeeping.** A non-scalar comparison hit `except (TypeError, ValueError): continue`
  *before* any counting, so it was recorded as neither agreement, disagreement,
  unverifiable, nor error — it simply vanished, which is why the `rho` bug above never
  showed up as a harness disagreement despite being wrong by construction. 29 of 487 owned
  variables were affected. Same "tautological agreement" blind spot the switch-consultation
  audit found independently, in array form. Fixed.

Full per-candidate classification of the 12 (and the ranking/closing-plan detail for each)
is in git history; none of it is needed once the two structural bugs above are fixed and
the headline count is known.
