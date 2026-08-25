# `_audit/closed/` — investigations whose question is answered

**What this directory is for.** This project keeps its reasons verbatim rather than
deleting them: a measurement that cost three PROCESS reference runs is worth more than the
one-line conclusion it produced, and the conclusion is only checkable if the working is
still there. But a finished investigation left beside the live design records competes with
them for attention, and the live set is the one that has to stay readable. So finished
records move here instead of being deleted or summarised away.

**The rule.** A record belongs here when its question is settled *and* nothing live —
`next_steps.md`, `model_tree_design.md`, `unit_registry.md`, `test_harness.md`, or a code
comment — defers to it for a decision still being made. Being *cited* is not enough; the
live documents cite these files as the evidence for findings they have already absorbed.
Anything that is still the authority for an open choice stays in the live set, however
finished it looks.

**Moving a record here does not weaken it.** The findings below are load-bearing and are
cited from `next_steps.md` at their new paths.

## What is in here

- **`x109_pinning_verification.md`** — settled whether `next_steps.md` §11.11's pinned
  `x109` point was genuinely better or a feasibility-tolerance artefact. It is genuinely
  feasible (`max|eq| 2.1e-12`, no inequality violated), and the multiplier hypothesis is
  refuted by five orders of magnitude (`Σ|λ_eq| = 1.22`, not `6.3e+04`).
- **`x109_hypotheses.md`** — settled *why*, testing four hypotheses. The cause is a kink in
  PROCESS's model, not a port defect and not premature termination: every converged point
  sits on `(Te + Ti)/20 == 0.65`, the threshold of `fast_alpha_beta`'s clamped square root,
  where two of 690 Jacobian cells — both in the `c24` row — disagree with a central
  difference of the port's own condition map. Its §8 records what is *not* settled
  (what to do about the kink, how wide the ridge is); those are open questions the
  investigation deliberately left open, not open questions about its own conclusion.
