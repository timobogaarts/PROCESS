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

## Also deleted on 2026-09-07

- **`cottax/warp/`** (11 modules, 8,174 lines) and `warp_stellarator_helias.ipynb`. The
  experiment was archived in `next_steps.md` § Landed — its findings, its two refuted
  hypotheses and the resumption plan are all there — and the code went with the
  narrative rather than sitting in the tree as a second, unexercised backend. Nothing
  tracked imported it. `git log --diff-filter=D -- functional_process/cottax/warp/`.
- **The per-unit records' prose in `cottax/`'s node files.** The node layer is a wrapper:
  every docstring in it is now its own first sentence. What a quantity means and why a
  formula is what it is belongs in `models/`, which the node file imports from; why a
  port was written the way it was belongs in its commit. 32,815 → 21,775 lines.
- **`provider.py`, `test_provider.py` and the seven `reference_provider_*.txt` pins**
  (~4,000 lines), and with them the `--provider`, `--provider-strict` and `--seed`
  modes of `run_cold_matrix` and `session`. The provider existed to move one number --
  how much of the boundary need not come from PROCESS's seed -- and `--native` is where
  that number arrived. Keeping three ways to be *partly* seeded, one of them the
  default, meant every row carried a `seed` column asking which of them it was.
  `CONFIGURATIONS` and `stem` moved to `native.py`, which is where they belonged;
  `run_cold_matrix.CONFIGURATIONS` is a re-export for its four existing callers.

  **`reference_cold_matrix.txt` is now stale in two ways** and wants a re-run: it was
  measured at `a8f98e35`, several declaration-changing commits ago, and it still has the
  `seed` column that no longer exists. Its rows differ from a fresh `helias_5b` by one
  node / one condition / one equality, which is `ba84ce1d`'s recovered declarations and
  not this cleanup -- the assembled graph is byte-identical, node for node, to the
  session's starting commit.
