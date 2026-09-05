# Tried and rejected

An index, not a record. One or two lines per entry: what was tried, the number that
killed it, where the detail lived. The detail itself is not here — it is in git history
(`git show be7756fc:functional_process/_audit/optimise_design.md` and its ancestors, or
`git log -- functional_process/_audit/`) at the section cited. Read this before re-trying
any of these; do not re-derive them.

## The Ward kink smoothing (`fast_alpha_beta`, `stellarator_helias`)

- **`eps = 5e-4` measured and rejected.** Clears `test_mdf.WORST_DX` (`9.60e-09` against
  the `1e-8` threshold) but sits at the edge of the failure band below with only a ~4%
  margin — rejected as a tripwire that ages badly, not as broken. (optimise_design.md
  §31.41.2)
- **Failure band `eps ~ 1e-5 .. 3e-4`.** Inside it the smoothing's sub-threshold tail is
  strong enough to attract the optimiser into a region PROCESS's model says is flat and
  empty, but too weak to define a descent there, and the solve stalls. Safe below
  (`<=1e-6`) and safe well above (`>=5e-4`, a full decade to `5e-3`). (§31.40)
- **The `eps²/√eps/eps` scaling.** Collateral damage away from the threshold scales
  `eps²/(8a^1.5)`; the benefit at the threshold scales `0.26 r² √(eps/2)`; the spurious
  sub-threshold tail scales `0.13 r² eps/√|a|`. On the plateau smaller `eps` always wins
  (collateral falls quadratically, benefit only as a square root) — the only reason to
  take a larger `eps` is margin. (§31.41.1)
- **Shipped:** `WARD_KINK_SMOOTHING = 1e-3`, mid-plateau, a factor of 2-3 above the band's
  upper edge with another factor of 5 of headroom. (§31.41)

## The vacuum pump-count staircase (`jnp.floor(pumpn + 0.5)`, `stellarator_helias`)

Three repairs tried, all worse than the discreteness they replace. Shipped baseline: 94
iterations, converged, residual `2.88e-06`.

- **Straight-through estimator** (`dn/dpumpn = 1`, value unchanged). Worst of the three
  despite looking most principled: 50 iterations, **stopped**, residual `8.3e+01`. An
  exact envelope derivative on a piecewise-constant value promises the QP a decrease the
  line search then can't deliver, every iteration. (§31.33.5)
- **Drop the rounding entirely.** Residual improves four orders (`2.4e-10`) but it's a
  different machine (166.118 pumps, not an integer count), moves `objf` by `1.7e-04`
  relative, and — decisively — makes the arm *less* robust to a last-bit Jacobian nudge
  (1 of 4 draws converge, against 3 of 4 shipped). The "staircase causes the fragility"
  hypothesis is refuted by its own control. (§31.33.5)
- **Node-local finite-difference tangent on the rounding**, seven step sizes. None
  converge. At every "principled" step size (a whole tread or multiple) it is bitwise
  identical to the already-refuted straight-through repair, by an exact identity; below
  that it introduces a derivative discontinuity of 3-30, an order of magnitude worse than
  the defect it targets. (§31.34.4)
- **Verdict: `jnp.floor(pumpn + 0.5)` stays exactly as PROCESS has it.** Real
  discreteness, PROCESS has the same staircase, PROCESS's own finite-difference
  instrument is blind to it, and the optimiser must cope — coping is not the hard part
  (a second configuration carries the same staircase and isn't chaotic at all). (§31.33.7)

## `jacfwd`/`jit` composition — two correctness traps, not fixes

- **A jitted and an unjitted `jax.jacfwd` of the same map at the same point disagree**,
  by about 1000x more than the single-ulp perturbation that flips a real solve between
  converged and stopped. Both sides are individually deterministic; they just aren't the
  same function. (§21.2)
- **Closing an array over a jit vs. passing it as a traced argument changes what XLA can
  constant-fold** — 17 of 21 conditions and 100 of 294 Jacobian cells move by 1-2 ulp
  between the two forms, and a 1-2 ulp move is enough to flip convergence on this
  problem. Passing arrays as arguments is required to cache the jit across solves; giving
  up the folding is the price, not a bug. "Keep the 2 compiles per solve" and "keep this
  row bitwise" are the same choice — you cannot have both. (§24.2)

## Five QP-conditioning nulls (why one SQP arm is unstable and another isn't)

Three arms compared (`stellarator_helias` SAND — unstable; `stellarator_helias` MDF —
unstable but finishes in time; `low_aspect_ratio_DEMO` SAND — flat, and larger). Every
proposed discrete mechanism inside the optimiser came out null:

1. **BFGS Hessian `cond(B)`** — degrades 8-12 orders on all three arms including the flat
   one; correlation with per-iteration instability is `-0.07` to `-0.15` (wrong sign).
2. **Constraint-Jacobian conditioning** — flat along every trajectory (growth
   `<5e-4` dec/it); the cross-arm ordering is backwards from the instability ordering.
3. **`design_scale`'s `1e-12` floor** — never reached; the closest coordinate on any arm
   is 8+ orders away.
4. **Line-search step length `alpha`** — exactly `1.0` on 100% of iterations, all three
   arms, both perturbed and unperturbed. Never shortens a step.
5. **Active-set churn** — anti-correlates with instability (the flat arm churns on 78% of
   iterations, the chaotic one on 24%).

Verdict: no discrete optimiser mechanism is doing it. The SQP map itself is expansive on
one problem's nonlinearity and contractive on the other's; the QP has almost no free
directions (2 of 8-14, 3 of 26), so the active constraints — not the objective's
curvature — set the step. (optimise_design.md §31.35)

## Three mechanisms shown to be *consequences* of the instability, not causes

Measured three separate times, same shape each time: the mechanism tracks the
already-diverged trajectories rather than driving the divergence.

- **The fused Jacobian** (§31.32.4).
- **The pump-count staircase** — both runs cross every tread in lockstep and only diverge
  afterward (§31.33.6).
- **Active-set changes** — first differ at iterate 38 (SAND) / 60 (MDF), by which point
  the trajectories have already separated by `~1e-5` / `~1e-6` (§31.35.6).

## Reverse-mode AD

The only remaining blocker for a whole-graph reverse-mode gradient is
**`models/vacuum/vacuum.py:474`** (`solve_duct_geometry`, a `lax.while_loop` search, not
a solve — genuinely hard to convert, not merely unconverted). With it stood down behind a
`stop_gradient` stand-in (not a landed fix), `jax.grad` of the scalar objective works
across the whole tokamak graph and agrees with `jax.jacfwd` to `2.0e-14`. (§33.7; the
other two `while_loop`s that used to block this — `cs_fatigue` and
`solve_duct_diameter` — are already converted/repaired, see `optimise_design.md`
history.)

## Measured constants worth not re-deriving

- **Driver benchmark**: the port is **9.1x PROCESS end-to-end** (12.4x with SLSQP instead
  of pyvmcon), **181x per SQP iteration** (2371 ms → 13.1 ms), while taking 7.1x more
  iterations; the per-evaluation cost alone is **~3200x** (one value+Jacobian pair costs
  0.74 ms against PROCESS's 2371 ms/iteration). (§13)
- **jax dispatch floor**: ~0.26 ms for a bare `jax.jit(x * 2.0)` call — the floor beneath
  any per-call timing measured in this file. (§31.14)
- **HLO cost**: ~52 StableHLO lines per assembled node (55/node before a vectorisation
  pass on the tokamak); ~200 bytes of peak RSS per character of pre-optimisation
  StableHLO a row lowers. (§31.2, §31.16)
- **Compile cache, cold vs. warm**: stellarator row 60s → 27s, tokamak row 146s → 50s —
  but only ~1.2% of peak RSS, so the persistent compilation cache is a speed lever, not a
  memory one (the OOM risk is unaffected). (§31.16)
- **Scan-point and re-assembly compile counts**: making stated boundary values traced
  program arguments (rather than baked into the graph) takes a second scan point from 1
  compile/9.25s to **0 compiles/0.40s (23x)**, and makes `graph_for(f) == graph_for(f)`
  true by value rather than merely by re-tracing. (§34.6)
