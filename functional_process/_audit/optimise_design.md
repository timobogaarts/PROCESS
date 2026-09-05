# The `Optimise` layer — design

**Status: built.** This was the design-and-measurement record for wiring PROCESS's
constraints, objective and iteration variables into a real `cottax.problem.Optimise` over
the driven graph, and for choosing/benchmarking the SQP driver around it
(`functional_process/sand.py`, `core/solver/drivers.py`, `sand_harness.py`,
`run_sand_harness.py`, `test_sand.py`, `mdf.py`). All of it landed. The port's SQP driver
reproduces PROCESS's answers on the tracked configurations (see `unit_registry.md` /
`next_steps.md` for current per-configuration status) and is measured at roughly 9-12x
PROCESS's wall clock end-to-end and ~180x per iteration, with exact forward-mode
gradients replacing PROCESS's finite differences.

This file grew to 12,000+ lines of investigation across dozens of numbered sections while
that was being worked out — SAND vs. MDF, driver choice, jit/compile strategy, a long
chase of one stellarator configuration's solve instability, and a kink-smoothing retune.
None of that narrative is needed to use or extend the driver layer today. What is worth
keeping — the specific things tried and rejected, with the numbers that killed them, so
nobody re-spends the time re-deriving them — has been extracted to
`_audit/tried_and_rejected.md`. The full investigation, section by section, remains in
git history (`git show be7756fc:functional_process/_audit/optimise_design.md` and its
ancestors, or `git log -- functional_process/_audit/optimise_design.md`); this file's
sections are cited by number from many code comments and other audit records, and none of
that numbering has been changed by this cut — it is simply no longer reproduced here.

## 2026-09-05: `VmconDriver.fused` landed on by default

`host_cache.bind`'s fused `values_and_jacobian` program (§31.30) was held off since
2026-09-03 because turning it on flipped `stellarator_helias` cold SAND from converged
to stopped — a real row, but §31.32 showed the arm was unstable to *any* last-bit
Jacobian change, fused or not, and `WARD_KINK_SMOOTHING = 1e-3` (landed since, see
`tried_and_rejected.md`'s Ward kink entry) closed off the mechanism: the same arm now
takes 24 iterations on every `+-1` ulp draw. That removed the one reason `fused` was
held back, so it was re-measured rather than re-argued.

**Re-measured [2026-09-05, measured].** Full seven-configuration cold matrix
(`--native --compare-process`, twelve MDF/SAND rows) with `fused` on is **bit-for-bit
identical** to the tracked `reference_cold_matrix.txt` — every status stays `converged`,
every printed digit of `objf`/`max|eq|`/`min ie`/`worst dx` is unchanged. No row was
re-baselined because no row moved. The fused-vs-split Jacobian still differs at the last
bit or two (unchanged from §31.30.4/§31.32.1's finding — not a bitwise spelling, and
never claimed to be), but nothing downstream of that difference is visible on any row
today.

**What it buys, SAND arm only, `--native` seeding, isolated via `host_cache`'s three
programs directly [measured 2026-09-05]:**

| | `values` + `jacobian` | fused | removed | per-call split | per-call fused |
|---|---|---|---|---|---|
| `stellarator_helias` SAND | 5 648 + 11 480 = 17 128 lines | 11 595 lines | 5 533 (32%) | 2.560 ms | 1.572 ms (-39%) |
| `large_tokamak_nof` SAND | 15 810 + 40 301 = 56 111 lines | 40 555 lines | 15 556 (28%) | 9.459 ms | 8.237 ms (-13%) |

Line counts are `lower(...).as_text()` StableHLO line counts — a property of the program,
not of machine load. Per-call times are medians over every call one solve makes (47 and
19 calls respectively), each set collected within one process. Whole-row wall clock is
not quoted; §31.30.1 already found it unusable under concurrent load.

`VmconDriver.fused` now defaults to `True`. `tests/functional_process tests/unit` still
pass (see the commit that landed this). `tried_and_rejected.md`'s fused entry records the
held-back-then-shipped history in full.
