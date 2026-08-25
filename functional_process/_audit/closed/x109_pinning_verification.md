# Verifying §11.11's `x109` pinning claim

**What this file is.** `next_steps.md` §11.11 rests one load-bearing claim — *"the port's
converged answer is not the global optimum of the port's own problem"* — on a single
measured row: pinning `x109` (`.physics.f_nd_alpha_thermal_electron`) at PROCESS's own
`0.033590406` and re-solving the other seven gives `objf 1.217549493`, **below** the free
optimum's `1.217757336`. The row was recorded with `max|eq| 3.3e-09` against the free
point's `1.2e-13`, which left an alternative reading open: at a Lagrange multiplier of
~`6.3e+04` on the binding equalities, that residual violation alone buys the whole
`2.08e-04` objective gain, and the "better point" would be a feasibility-tolerance
artefact rather than a second, lower point.

This file settles it. Two independent lines of evidence, both run.

**Verdict: the claim stands, and the reason it stands is not the one §11.11 gives.**
The pinned point is genuinely feasible — to `max|eq| 2.1e-12` and *zero* inequality
violation — with `objf 1.21754949`, and the same design point evaluated through the
**free** eight-variable problem's own condition map is a feasible point of that problem
with a strictly lower objective. The multiplier hypothesis is refuted by five orders of
magnitude: `sum|lambda_eq| = 1.22`, not `6.3e+04`. But `x109` is **not a flat direction**
either: the reduced objective has a real, ~`2.2e-04` deep well whose minimum sits at
`x109 ~ 0.034`, i.e. essentially at PROCESS's own value, and the port's free solve
terminates on the shoulder of it. `x109` is a **premature-termination** artefact, not a
model difference and not a flat direction.

## What was measured, and where

- Env `process_port`, `~/miniconda/envs/process_port/bin/python`, `jax_enable_x64` on.
- `process` and `functional_process` both resolve to the worktree
  `/home/tbogaarts/PROCESS/.claude/worktrees/agent-a2c20704f5613cf65`, so the tree under
  test is this one, at `c1968f52` plus the one change below.
- One PROCESS reference run per script (`46 VMCON iterations, 92.0 s, conv 2.396e-07`),
  `tests/regression/input_files/stellarator_helias.IN.DAT`, `ixc [2,3,4,6,10,56,59,109]`,
  `icc [2,16,24,8,17,18,67,82,83,62,32,34,35,65]`, first 2 equalities.
- Every solve is SAND, `VmconDriver`, PROCESS's own per-run bounds, `1/x_start` design
  scaling, `residual_condition_scales` on the residual equalities only. Solves cost
  8–12 s each including compilation; the reference run dominates.

**The one code change made for this work.** `VmconDriver.__call__` computed its own
`1/x_start` scale inline with the **unfloored** guard `np.divide(..., where=flat_start
!= 0.0)`. §11.11 records that defect as *"Open"* and `next_steps.md`'s closed list
records it as *"Now floored at PROCESS's own threshold ... Seven tests pin it"* — both are
true of `design_scale`, which is used **only by `scaled_problem`, i.e. the SLSQP path**.
The VMCON path never got the fix. It does in this worktree (`VmconDriver.__call__` now
calls `design_scale`), and the change is **inert for every run reported below except the
continuation ones**: warm-from-PROCESS starts have `.power.qac` at *exactly* `0.0`, where
the old and new rules both give `scale = 1.0`.

## How "pinned" is expressed

`optimise_graph` is called with `ixc` minus `109`, so `.physics.f_nd_alpha_thermal_electron`
stops being an `Optimise`-owned unknown and reverts to an ordinary graph input, seeded
from the `DataStructure` the run is seeded from (`_seed`'s existing rule). Nothing else
changes: same nodes, same fourteen constraints, same objective, same residual equalities,
same scaling. The design vector goes 8 → 7 and the pinned value enters as data.

**The control that validates the method**: pinned at the port's *own free answer*
(`0.029951833`) the pinned problem returns `objf 1.21775733634` / `1.21775733700` in two
separate runs, against the free solve's `1.21775733558`. Agreement to `~1e-9`. So the
pinned problem is the free problem restricted, not a different problem.

## (b) The multipliers — the `6.3e+04` figure is wrong by five orders of magnitude

`pyvmcon.solve` returns `(x, lamda_equality, lamda_inequality, result)` and `VmconDriver`
discards all but the first. Captured by wrapping `pyvmcon.solve` (and, per iteration, by
passing `additional_convergence`, which `pyvmcon` calls with the multipliers). Values are
for the **scaled** problem VMCON actually solves, which is the right frame: the reported
`max|eq|` is scaled too, so `lambda * eq` is directly comparable to the objective gap.

At the free optimum (`objf 1.21775733561`):

| condition | `lambda` | value | `lambda * value` |
|---|---|---|---|
| eq `^cond.constraints.c2` | `1.0512e-02` | `-6.40e-13` | `-6.7e-15` |
| eq `^cond.constraints.c16` | `9.9456e-01` | `-1.23e-14` | `-1.2e-14` |
| eq `^cond.stellarator.wp_width_r_min` | `1.7122e-04` | `-3.37e-09` | `-5.8e-13` |
| eq `^cond^cond.physics.temp_plasma_ion_vol_avg_kev` | `1.4742e-01` | `-8.84e-17` | `-1.3e-17` |
| eq `^cond^cond.fwbs.qnuc` | `-2.5917e-02` | `-2.12e-14` | `5.5e-16` |
| eq `^cond^cond.power.qmisc` | `-1.7735e-02` | `-6.93e-14` | `1.2e-15` |
| eq `^cond^cond.power.qcl` | `-1.3766e-02` | `-2.35e-13` | `3.2e-15` |
| eq `^cond^cond.power.qss` | `-7.7707e-03` | `6.51e-14` | `-5.1e-16` |
| eq `^cond^cond.heat_transport.p_fw_div_heat_deposited_mw` | `-2.6142e-03` | `2.80e-15` | `-7.3e-18` |
| eq `^cond.fwbs.f_ster_div_single` | `6.7752e-04` | `7.32e-16` | `5.0e-19` |
| eq `^cond.physics.proton_rate_density` | `-1.7647e-05` | `9.07e-16` | `-1.6e-20` |
| eq `^cond.physics.fusden_alpha_total` | `1.7612e-05` | `1.35e-15` | `2.4e-20` |
| eq `^cond^cond.power.qac` | `-2.1082e-07` | `3.92e-25` | `-8.3e-32` |
| eq `^cond^cond.power.delta_eta` | `-2.0918e-11` | `-3.55e-14` | `7.4e-25` |
| eq `^cond^cond.primary_pumping.p_fw_blkt_coolant_pump_mw` | `-2.8424e-11` | `4.40e-15` | `-1.3e-25` |
| eq `^cond^cond.heat_transport.etath_liq` | `0` | `0` | `0` |
| eq `^cond^cond.heat_transport.temp_turbine_coolant_in` | `0` | `0` | `0` |
| **ie `^cond.constraints.c83`** (place for blanket) | **`1.4321e+00`** | `2.22e-16` | `3.2e-16` |
| ie `^cond.constraints.c24` (beta limit) | `2.1699e-01` | `-3.19e-12` | `-6.9e-13` |
| ie `^cond.constraints.c35` (quench) | `9.4946e-03` | `4.64e-12` | `4.4e-14` |
| ie `^cond.constraints.c62` (thermal He) | `7.9872e-03` | `7.97e-13` | `6.4e-15` |
| ie `c8`, `c17`, `c18`, `c67`, `c82`, `c32`, `c34`, `c65` | **exactly `0`** | `0.48`–`0.97` | `0` |

- **`sum|lambda_eq| = 1.221188`.** The hypothesis needs `6.3e+04`.
- `sum|lambda_eq| * 3.3e-09 = 4.03e-09`, against the `2.078e-04` gain to explain —
  **short by a factor of `5.2e+04`**.
- The realised `|lambda_eq . eq| = 5.9e-13` at the free optimum, and `1.3e-14` at the
  pinned one. The largest multiplier anywhere in the problem is `c83`'s `1.4321`.
- §11.11's active set is **confirmed exactly**: the two PROCESS equalities, the fifteen
  SAND residual equalities, and precisely four inequalities `c24`/`c83`/`c62`/`c35` with
  non-zero multipliers; the other eight inequalities have `lambda == 0` and slack
  `0.48`–`0.97`.

There is also an a-priori argument, which the numbers then confirm: `pyvmcon`'s own
convergence value **is** `|df.delta| + |sum(lambda_eq * eq)| + |sum(lambda_ie * ie)|`
(`vmcon.py:385-387`). A run reporting `conv 9.4e-12` has already bounded the
multiplier-weighted violation at `9.4e-12`, which cannot be `2.08e-04`.

## (a) Re-solving the pinned problem to the free point's own feasibility

**Feasibility has two halves and §11.11's table records only one.** The pinned solve's
objective gain does *not* come from the equalities — but at loose tolerance it does come
from an **inequality violation**, which the recorded row does not show. At
`tolerance = 1e-11` the pinned solve never gets there at all (200 iterations, ends at
`objf 1.2500`, `max|eq| 8.7e-02`); at `tolerance = 1e-13` it reaches `objf 1.21754947`
with `max|eq| 1.04e-12` but **`min ie = -1.14e-04`** — a genuine constraint violation, and
at `c83`'s multiplier of `1.43` that violation is worth `1.6e-04`, which is most of the
gain. So the sceptical reading was right in *kind* and wrong in *which constraint*.

It does not survive contact with the fully feasible iterates. Judging every iterate by
`max|eq| <= 1e-11` **and** `min ie >= -1e-11` — both properties of the point, never `conv`
— the pinned solve visits fully feasible points and they carry the gain:

| run | iterate | `objf` | `max|eq|` | `min ie` |
|---|---|---|---|---|
| pinned, `tol 1e-13`, `max_iter 300` | 287 | `1.21754975161` | `2.57e-12` | `+2.2e-16` |
| | 294 | `1.21754950431` | `6.74e-13` | `-2.2e-16` |
| | 295 | `1.21754950427` | `3.17e-13` | `-2.2e-16` |
| | 296 | `1.21754950403` | `3.17e-13` | `-2.2e-16` |
| | 297 | `1.21754950285` | `1.59e-13` | `-0.0` |
| | 298 | `1.21754949694` | `1.19e-13` | `-0.0` |
| pinned, `tol 1e-13`, `max_iter 400` | 332 | `1.21754949255` | `2.10e-12` | `-3.7e-16` |
| pinned, `tol 1e-13`, `max_iter 400` (3rd run) | 345 | `1.21754949237` | `9.43e-12` | `-2.2e-16` |
| **free**, `tol 1e-11` | **85** | **`1.21775733558`** | **`5.55e-13`** | **`-3.3e-16`** |
| free, `tol 1e-13` | 85 | `1.21775733558` | `5.55e-13` | `-3.3e-16` |

**This is the objf-vs-feasibility trajectory, and its shape is the answer.** Over the
pinned run's last five iterates `max|eq|` falls `6.7e-13 -> 1.2e-13` while `objf` moves
`1.21754950431 -> 1.21754949694` — a change of `7.4e-09`, four and a half orders of
magnitude smaller than the `2.078e-04` gap it would have to close. Extrapolated to zero
violation the pinned objective is `1.2175495`, flat to nine digits. The gain does not
decay with feasibility because it was never paid for out of feasibility.

Three independent pinned runs land on `1.2175494924 +/- 4e-10`. §11.11's recorded
`1.217549493` is reproduced to nine digits; only its `max|eq|` annotation and its
"feasible" verdict were wrong (and, as the table shows, the *free* run's own stopping
iterate is the one that sits at `max|eq| 3.4e-09` — the largest equality residual at both
points is the same one, `^cond.stellarator.wp_width_r_min`, the coil-island `Intersect`
residual, and where in `1e-13`–`1e-9` a run happens to stop is an accident of the iterate,
not a property of the answer).

### The independent existence proof

The strongest form needs no multiplier reasoning and no comparison of two solves. Take
the pinned run's best fully feasible design point, put `x109` back to PROCESS's
`0.033590406`, and evaluate it through the **free** eight-design problem's own
`ConditionMap` (one VMCON iteration with an unreachable tolerance, so no step is taken):

| | `objf` | `max|eq|` | worst equality | `min ie` | inequalities violated beyond `1e-11` |
|---|---|---|---|---|---|
| the pinned point | **`1.21754949255`** | `2.1010e-12` | `^cond.stellarator.wp_width_r_min` | `-3.68e-16` | **none** |
| the free optimum (control) | `1.21775733561` | `3.3688e-09` | `^cond.stellarator.wp_width_r_min` | `-3.19e-12` | none |

**A feasible point of the port's own free problem with an objective `2.078e-04` (`1.71e-04`
relative) below where the free solve converged.** That is the claim, demonstrated
directly.

### And the free solver walks away from it

Restarting the **free** (all eight design variables) solve *at* that point, `tol 1e-13`:
it leaves immediately (iterate 1: `objf 1.21752`, `max|eq| 6.2e-04`, `min ie -1.9e-02`),
wanders, and after **178 iterations reports `converged = True` at `objf 1.21775733536`**
with `x109` back at `0.02995183`. An SQP that starts at a feasible point and declares
convergence at a *worse* one is not at a minimum; the free answer is a point VMCON's
convergence test accepts, not a point the problem selects.

## The reduced objective is a well, not a flat direction

`phi(v)` = the best fully feasible objective the pinned solve attains with `x109 = v`
(an upper bound on the true reduced objective; `tol 1e-13`, `max_iter 400`, warm from
PROCESS's `DataStructure` in every row, so nothing but `v` differs):

| `x109` | `phi` | `phi - free` | relative | `max|eq|` | `min ie` | iterations |
|---|---|---|---|---|---|---|
| `0.029851833` | `1.21868904` | `+9.32e-04` | `+7.7e-04` | `6.3e-13` | `-4.4e-16` | 400 (weak bound) |
| **`0.029951833`** (port's free answer) | **`1.21775734`** | `+7e-10` | `+6e-10` | `7.9e-14` | `-0.0` | 141 |
| `0.030251833` | `1.21770331` | `-5.40e-05` | `-4.4e-05` | `2.5e-12` | `+2.2e-16` | 165 |
| `0.030551833` | `1.21771573` | `-4.16e-05` | `-3.4e-05` | `2.1e-12` | `-0.0` | 172 |
| `0.030861476` | `1.21766531` | `-9.20e-05` | `-7.6e-05` | `9.0e-12` | `-5.4e-16` | 153 |
| `0.032680763` | `1.21756164` | `-1.957e-04` | `-1.61e-04` | `2.4e-13` | `-2.1e-17` | 391 |
| **`0.033590406`** (PROCESS's own) | **`1.21754949`** | `-2.078e-04` | `-1.71e-04` | `9.4e-12` | `-2.2e-16` | 370 |
| `0.034000000` | `1.21753231` | `-2.250e-04` | `-1.85e-04` | `7.0e-12` | `-0.0` | 292 |
| `0.037228979` | `1.21767284` | `-8.45e-05` | `-6.9e-05` | `5.2e-13` | `-3.3e-16` | 151 |
| `0.031771120`, `0.034500000`, `0.035000000`, `0.036000000`, `0.044506126` | no fully feasible iterate inside 400 SQP iterations | | | | | |

Read it as three statements of decreasing confidence.

1. **Robust.** `phi` falls monotonically by `~2.2e-04` from the port's free answer to
   `x109 ~ 0.034`, and is back up at `0.0372`. There is a genuine interior minimum, it is
   `~2.2e-04` deep, and it sits **at or just past PROCESS's own `x109`**. Those
   differences (`1e-04`–`2e-04`) are four orders of magnitude above the run-to-run scatter
   of the pinned solve (`~4e-10`).
2. **Solid.** The free answer is therefore **not** a local minimum of `phi` — `phi` is
   still descending there, over a `12 %` stretch of `x109`. This is *not* what
   §11.11 calls a flat direction: `x56` costs `+39 ppm` to move `5.6 %` (flat), while
   moving `x109` `12 %` *pays* `171 ppm` (downhill).
3. **Not established.** The local slope of `phi` at the free answer itself. At `1e-04`
   spacing the signal (`~1e-05` in `phi`) is comparable to the pinned solve's own
   attainment noise: three of the eleven fine-scan pins found no fully feasible iterate in
   400 iterations, the `0.029851833` row's bound is visibly weak, and `0.030251833` /
   `0.030551833` are not monotone with respect to each other. **Deriving `d(phi)/d(x109)`
   from this scan would be over-reading it.** What settles it, if it ever matters, is the
   reduced gradient from the Jacobian at the free point (a Schur complement the harness
   already computes for Stage B), not more pinned solves.

## What was tried and did not work, with the reason

- **Continuation from the free optimum** — §11.11's own protocol for the pinned solve.
  With the `design_scale` floor now on the VMCON path it does **not** reproduce §11.11's
  row: it diverges, at both `1e-11` and `1e-13`, to `objf 2.3175`, `max|eq| 3.46`,
  `min ie -1.11` in 84 iterations. The cause is not the scaling any more — the only
  unknown below the floor at the free optimum is `.power.qac` (`-3.9e-25`), and it is now
  scaled `1.0`. It is that the free optimum's *coupling* unknowns are consistent with
  `x109 = 0.029952` and the pinned problem immediately imposes `0.033590`, so the solve
  starts from a badly inconsistent state. **Warm-from-PROCESS is the start that works**,
  and it reproduces §11.11's pinned value anyway, so nothing is lost.
- **`tolerance = 1e-11` on the pinned problem** — 200 iterations, ends at `objf 1.2500`,
  `max|eq| 8.7e-02`, nowhere near feasible; no fully feasible iterate at all. The pinned
  problem genuinely needs `1e-13` and a few hundred iterations. (`1e-13` does not
  "converge" either — `pyvmcon` raises and the best point is returned — but it *visits*
  fully feasible points, which is what the question needs.)
- **`tolerance = 1e-13` on the free problem** — stops after 96 iterations without a
  certificate, at the same point (`best feasible 1.21775733558`, identical to the
  `1e-11` run's). Tightening the tolerance does not move the free answer.

## What this changes

1. **§11.11's conclusion survives** and is now supported by a direct existence proof
   rather than by one row's arithmetic.
2. **§11.11's stated reason does not.** "`x109` is a flat direction" is wrong. `x56` is
   flat (`+39 ppm` for `5.6 %`); `x109` is a `~2.2e-04`-deep well whose minimum is
   essentially at PROCESS's own value, and the port's free solve stops on its shoulder.
   The right one-line reading of `x109` is: **the port's model, optimised properly, puts
   `x109` where PROCESS puts it; the `10.8 %` gap is the port's SQP terminating early in
   a shallow direction.** That is a strictly better outcome for the port's *model* than
   §11.11 claims, and a strictly worse one for its *solver*.
3. **`VmconDriver`'s `1/x_start` floor was never applied to the VMCON path** — the fix
   documented as closed lives in `design_scale`, which only `scaled_problem`/`SlsqpDriver`
   call. Fixed here; it is a one-line change and it is inert on every start whose
   near-zero unknowns are *exactly* zero.
4. **The harness should judge feasibility on both halves.** Stage C's trace already
   records `min ie` beside `max|eq|`; §11.11's table dropped it, and that is the whole
   reason the pinned row looked mysterious — its loose-tolerance version really was
   buying its objective with an inequality violation of `1.1e-04`. A "best fully feasible
   iterate" column (`max|eq| <= tol` **and** `min ie >= -tol`) is the statistic worth
   reporting for any solve that does not converge cleanly.
5. **Open, and now sharper than "is the port's answer optimal":** *why* does VMCON stop
   `12 %` short along `x109`, and walk away from a better feasible point when started
   there? The suspicion is the BFGS-scaled convergence test — `conv` contains
   `|df . delta|` with `delta` from the accumulated `B`, so a large `B` in the flat
   coordinate makes `conv` small without the reduced gradient being small. The cheap
   check is the reduced gradient at the free point, from the Stage B Jacobian.

## Reproducing

Scripts (scratch, not in the tree):
`scratchpad/x109_lib.py` (assembly for an arbitrary `ixc`, solve with multiplier capture,
`best_feasible`), `x109_run.py` (free + pinned + multipliers), `x109_verify.py` (the
existence proof, the free restart, the coarse scan), `x109_fine.py` (the fine scan).
Raw output: `x109_run.log`, `x109_verify.log`, `x109_fine.log`, and the `*.json` beside
them. Total cost: three PROCESS reference runs and ~25 SAND solves, under 20 minutes.
