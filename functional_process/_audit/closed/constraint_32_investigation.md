# The non-finite `^cond.constraints.c32` Jacobian row — traced, classified, closed

---
kind: investigation
status: closed; merged into `_audit/optimise_design.md` §5.2
confidence: high — every claim was run in `process_port`, not inferred
---

**This file is a stub.** Its content has been merged into
**`_audit/optimise_design.md` §5.2**, under "The non-finite `c32` row — traced and
closed", which is where the live account now lives. This stub remains only because
`functional_process/mda_harness.py` and three per-unit records
(`models/stellarator/coils/{coils,calculate,forces}.md`) cite this path.

**The cause, in one line.** The first non-finite value was
`.tfcoil.max_force_density = inf`, produced by `MaxForceDensity`
(`models/stellarator/coils/forces.py`), because its `a_tf_wp_no_insulation` input was the
harness's `0.0` placeholder — an artifact of `mda_harness.EXCLUDED_NODE_NAMES` excluding
the winding-pack coil island, **not** a port defect, and **not** (as first guessed)
`dr_tf_wp_with_insulation`, which was correct all along.

**The fix and its result.** Fixed at the cause rather than patched: the coil island is no
longer excluded, grounded by three `KNOWN_MINT_VALUES` reconstructions off PROCESS's own
stored fields (`wp_width_r_min`, `a_tf_wp_no_insulation`, `a_tf_wp_with_insulation` — the
derivations and their independent cross-checks are in `optimise_design.md` §5.2 and in
`models/stellarator/coils/calculate.md`). Measured: **17 → 0 non-finite cells**, every
Jacobian row finite. Two side effects worth knowing: `Intersect` gained the first
PROCESS-comparable value check it has ever had — its `Tier2Contract` has none by
construction — and its `RootFind` joined the SAND problem as a further design variable and
equality.

The general point, which is the durable part: **an excluded island does not merely zero
some Jacobian columns, it can make a whole constraint row non-differentiable**, which no
SQP accepts.

**One item is still open.** The x2/x59 columns of c82/c83/c32/c35 were measured as
spuriously zero while the island was cut out. They now have a live path, but **have not
been re-checked against `fcnvmc2` cell by cell** — see `optimise_design.md` §5.2's second
structured-disagreement bullet and §10.5 for the current per-cell record.
