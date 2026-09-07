# `_audit/` — what is here, and what used to be

These are the **working documents** of the port: the ones consulted while doing the work,
not the record of having done it.

| file | what it is |
|---|---|
| `unit_registry.md` | authoritative per-unit status — the one place to look |
| `next_steps.md` | priority-ordered punch list, plus what has landed |
| `traceability_policy.md` | the rules a port must follow (precision, switches, domain errors) |
| `test_harness.md` | how the validation harness is built and why |
| `tried_and_rejected.md` | measured refutations — read before re-deriving one |
| `deliberate_divergences.md` | every place the port knowingly differs from PROCESS |
| `naming_convention.md` | how minted names are spelled |
| `optimise_design.md` | the driver/optimiser design record |

**The 88 per-unit records under `_audit/units/` were deleted on 2026-09-07** — 32,075
lines. They were the derivation of each port: which PROCESS lines were read, what the
reads-set turned out to be, which branch was dead. That reasoning mattered while the port
was being made and is worth exactly nothing to a reader of the finished thing, who wants
`unit_registry.md`'s row and the port's own docstring. It is not lost, only moved off the
critical path: `git log --diff-filter=D -- functional_process/_audit/units/` finds the
deleting commit, and `git show <commit>^:functional_process/_audit/units/<path>.md` prints
any record verbatim. Citations elsewhere in the tree that name a record by path (e.g.
`density_limit.md "## UNPORTED"`) resolve that way and are deliberately left alone.

`schema.md`, which specified the record format, went with them. This is the same move
`next_steps_archive.md` records for the numbered state snapshots, and for the same reason.
