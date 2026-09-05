# Next steps

A current-state reference and priority-ordered punch list for the `functional_process`
port. `unit_registry.md` remains the authoritative per-unit status; this file is what's
still open at the driver/graph/solver level, above the level of any one unit.

This file used to carry a long, dated sequence of "state, YYYY-MM-DD" narrative sections
(§1 through §32) recording investigations as they happened. That narrative is no longer
here — most of what it found is either landed code, superseded by a later measurement in
the same sequence, or (where it's a refutation worth not re-deriving) moved to
`_audit/tried_and_rejected.md`. The full sequence remains in git history
(`git log -- functional_process/_audit/next_steps.md functional_process/_audit/
next_steps_archive.md`); section numbers cited elsewhere are unchanged, they just resolve
into git history rather than into this file now.

## Landed

SAND and MDF drivers both work end to end, on all seven tracked configurations, through
`jax.pure_callback` with exact forward-mode gradients (`core/solver/drivers.py`,
`sand.py`, `mdf.py`, `sand_harness.py`, `run_sand_harness.py`, `run_cold_matrix.py`).
Both tokamak and stellarator device classes are ported and registered. Reverse-mode AD
works on the whole tokamak graph's scalar objective except for one node (see
`tried_and_rejected.md`). The condition map is bound once per solve rather than rebuilt
per call; compiled-call caching lives in `core/solver/host_cache.py`; the boundary
provider distinguishes `input`/`guess`/`stated` boundary categories.

## Open

- **`vacuum.py:474` (`solve_duct_geometry`) is the sole remaining reverse-mode AD
  blocker** — a discrete first-fit search, not a root find, so it doesn't take the same
  `stop_gradient`-plus-Newton-step treatment already applied to its two sibling loops.
  Sizing for a `vmap`-over-64-candidates conversion exists; not done.
- **`sand_harness._SCHEDULE_WHOLE`/`_SCHEDULE_RUNNERS` are unbounded dicts** with the
  same trap `host_cache._BOUND` had before its removal. Not yet fixed.
  (`host_cache._BOUND` and `_flat_key` are **gone** as of 2026-09-05: the removal
  condition arrived in the better form — the graph is static outright, so there are no
  array leaves to precompute.)
- **[resolved 2026-09-05] The `stellarator_helias` SAND-arm instability.** Cause was a
  square-root singularity in `_fast_alpha_fraction_ward` (`sqrt(temp_sum_20 - 0.65)`:
  derivative zero below the threshold, unbounded above) that the optimiser sat 5.5e-08
  from and crossed on 46 % of steps. Smoothed at `WARD_KINK_SMOOTHING = 1e-3`; every
  ±1 ulp draw now takes 24 iterations and agrees on `objf` to fifteen digits, against
  87–333 and one catastrophic stop. Cost and the three rejected repairs are in
  `tried_and_rejected.md`. **Still open**: nine of twenty-six separation jumps are
  active-set changes rather than kink crossings, and are unexplained.
- **The SLSQP driver has never been run across the full seven-configuration matrix.** It
  would separate "this problem is degenerate" from "VMCON handles degeneracy badly", and
  for SAND there is no PROCESS answer to compare against at all, so a second independent
  optimiser is the closest thing to an oracle available.
- **Two unexplained boundary offsets, both on `.heat_transport.p_plant_electric_net_mw`**
  via constraint c16: `spherical_tokamak_eval` shows a `-16.35 MW` offset at PROCESS's own
  converged point, of the same shape as `mda_harness.EXPLAINED_DISAGREEMENTS`' documented
  stellarator mechanism (`Stellarator.run(output=True)` re-running build/coil in a
  different order than the solve pass) — but that mechanism is stellarator-specific and
  this is a tokamak. Not yet explained.
- **`^cond.constraints.c13` (burn-time lower limit) disagrees systematically** on two
  tokamak configurations, at a residual level well above every neighbouring constraint.
  Both configurations report `.times.t_plant_pulse_burn` as an array-valued fixed point
  dropped from the SAND block with its loop-carried value frozen at the seed — the likely
  cause, not yet confirmed by a direct diff against PROCESS's converged value.
- **`degenerate_fixed_points` asks a structural question with a pointwise measurement** —
  it decides "is this update the identity?" by evaluating at one point, so a live
  coupling that happens to look like the identity at that point is silently dropped.
  Should sample several points and refuse on disagreement, matching the rest of the
  port's "can't verify → refuse, don't report healthy" discipline.
- **`--provider` should be retired in favour of `--native` as the default** — understood,
  not done. The comparison axis (`--compare-process`) already exists independently.
- **The reduced-space MDF formulation is unbuilt**: let the graph close the equality
  constraints with a Newton solve and hand the optimiser only the design DOF, objective
  and inequalities — "multidisciplinary feasible" taken literally, a third point beyond
  today's MDF and SAND formulations.
- **`IFE.IN.DAT` is out of scope** — `.ife.*` has no unit in `unit_registry.md`.
