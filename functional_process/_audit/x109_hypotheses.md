# `x109`: four hypotheses, tested

**One-line answer: H2 survives, relocated. The defect both solvers inherit is not in
`scaled_problem` — it is one row of the shared `jax.jacfwd` Jacobian, `c24`, and it is
wrong at every iterate the solvers visit, because every converged point of this problem
sits on a non-differentiable ridge of PROCESS's own model: `fast_alpha_beta`'s clamped
square root, whose kink is at `(Te + Ti)/20 == 0.65`.**

Measured at five independently converged points (four of them feasible to `<= 4.4e-16` in
the worst inequality), `temp_sum_20 - 0.65` is `+1.9e-09`, `-1.3e-10`, `-1.3e-10`,
`+1.1e-10`, `-2.6e-13`. The whole solution family lives on the kink. PROCESS's own converged design sits `-5.8e-03` away from it and its
1 % finite differences could not resolve it in any case.

H4 is excluded outright (MDF lands where SAND does, to nine digits). H1's *multi-modality*
turns out to be real — there is a barrier and a better region beyond it — but H1's
*account*, "both solvers behaving correctly and simply starting in the wrong basin", is
refuted: started **inside** the better region, at a feasible point `1.8e-04` better in
objective, both VMCON and SLSQP walk back out of it, uphill. H3 predicts nothing testable
about `x109`, for a reason that is available without measuring anything.

Everything below was run in the worktree
`/home/tbogaarts/PROCESS/.claude/worktrees/agent-a1d3cf17ec9d75d1f` against
`tests/regression/input_files/stellarator_helias.IN.DAT`, off **one** cached PROCESS run
(101.7 s) reused by every experiment.

| | |
|---|---|
| `process.__file__` | `<worktree>/process/__init__.py`, `0.0.1.dev1184+gc0ae5b286` |
| `functional_process.__file__` | `<worktree>/functional_process/__init__.py` |
| `cottax.__file__` | `/home/tbogaarts/jaxgraph/src/cottax/__init__.py`, `0.1.0` (jax 0.11.0) |
| `x64` | on before any array is created |

`~/jaxgraph`'s **uncommitted working tree** renames `Input` → `FromExactly`; this worktree
still spells `Input`, so every `functional_process` import fails against it. Worked around
with a one-line runtime alias (`pnm.Input = pnm.FromExactly`) set before importing
`functional_process` in the scratch scripts, after checking against `git diff` that the
change is a pure rename. **No file in either tree was touched.**

---

## The verdict

| | verdict | the measurement that decides it |
|---|---|---|
| **H1** — genuinely multi-modal, both solvers correct, wrong basin | **the multi-modality is real; the account is refuted** | the barrier exists and is now measured (`phi` rises `+1.24e-05` from `x109 = 0.0299518` to `0.0300`, then falls `-1.83e-04` by `0.0320`, all three converged to `<= 3.2e-13` feasibility). But released from the feasible `x109 = 0.0320` point with `x109 >= 0.031` imposed, **both** VMCON and SLSQP converge **on the bound** at `0.0310` with `objf 1.21763602` — uphill from the `1.21757404` they started at. A correct local method does not leave a feasible point for a worse feasible point. And H1's stated prediction fails: `d(phi)/d(x109)` is `-3.0e-02` on the differentiable version of the problem, not 0. |
| **H2** — a defect in the shared problem construction | **survives, relocated** | not `scaled_problem`: measured, `0 of 23` design-scale entries differ between `VmconDriver`'s inline rule and `design_scale` at the seeded start. The shared object that is wrong is the **`c24` row of the Jacobian**. Of 690 cells, exactly 2 disagree with a central difference of the port's own condition map and both are in that row; `c24` alone drifts like `h^0.52` along the null direction where every other condition drifts like `h^2.00`. |
| **H3** — the objective gradient is wrong | **not the mechanism** | the port's `objf` row agrees with a central difference of the port's own map to `7.7e-10`. The 18–34 % of §10.5c is a *port-vs-PROCESS* difference, not an error in the port's own descent direction, and no account of the form "stops at a different objective's stationary point" can explain a solver failing to reach a feasible point of **its own** objective `1.8e-04` lower. |
| **H4** — SAND exposes a harder problem than MDF | **excluded** | MDF warm converges (`conv 7.155e-12`, `max\|eq\| 1.5e-15`, `min ie +2.2e-16`) at `x109 = 0.0299518328` against SAND's `0.0299518330` — **nine digits**. |

---

## 1. H4, and the MDF landing values

`mdf.py`, PROCESS's own architecture: 8 design variables, 15 conditions, the coupling
converged inside every evaluation. `tolerance = 1e-11`, `max_iter = 400`.

| | iterations | `conv` | `max\|eq\|` | `min ie` | `objf` | `x109` | vs PROCESS |
|---|---|---|---|---|---|---|---|
| **MDF C2 warm** | 168 | **7.155e-12** | 1.5e-15 | +2.2e-16 | 1.217757336 | **0.0299518328** | 1.08e-01 |
| **MDF C3 cold** | 400 | 5.5e-02 (not converged) | 2.5e-02 | -5.6e-03 | 1.247 | **0.0297241657** | 1.15e-01 |
| SAND C2 warm (VMCON) | 96 | 1.8e-10 | 9.2e-08 | -2.2e-16 | 1.217757335 | 0.0299518330 | 1.08e-01 |
| SAND C2 warm (SLSQP) | — | — | 2.8e-13 | +7.8e-16 | 1.217844675 | 0.0299518055 | 1.08e-01 |

**MDF warm and SAND warm agree on `x109` to nine digits**, and MDF's point is the cleanly
feasible one. H4 is dead: exposing the 15 coupling residuals to the QP is not what puts
`x109` at `0.02995`, because hiding them inside an MDA puts it in exactly the same place.

Two side notes from the same run, both correcting §11.10:

- `x56`: MDF warm `31.7606`, SAND warm `31.7570` — `0.011 %` apart. §11.10's "MDF C3
  landing at 36.20 where SAND lands at 31.76, on *opposite sides* of PROCESS's 35.32" was
  measured before `LModeProfileReset` and is superseded; warm, the two architectures agree.
- `x59`: MDF `0.7177490`, SAND `0.7177269`.

---

## 2. The reduced gradient, its derivation, and its error bar

### Construction

`u` is the 23 SAND unknowns; `f(u)` the objective condition; `h(u) = 0` the 17 equalities
(PROCESS's `c2`, `c16` plus the 15 coupling residuals); `g(u) <= 0` the 12 inequalities. At
the free optimum exactly four inequalities are active — `c24` (beta limit), `c83` (place
for blanket), `c62` (thermal He), `c35` (quench) — with `|g| <= 1e-05` against a
next-nearest inactive `|g| = 0.476`, so the active set needs no tuned threshold. Every
design variable is interior to its bounds.

Everything is in **column-equilibrated coordinates** `v = D^-1 u`, `D = diag(|u|)`, rows
equilibrated by their largest entry — an exact change of variables (the raw Jacobian spans
~30 orders of magnitude), undone by one division by `D_109`.

- **A, null space.** `N` an orthonormal basis of `null(A_active)`; `r = N^T (D grad f)`,
  `s = N^T e_109`; `d(phi)/d(x109) = (r.s)/(s.s)/D_109`, and `||r - (r.s/s.s)s||` is the
  descent still available at *fixed* `x109`.
- **B, the pin multiplier.** Fit `grad f + A_h^T lam + A_g^T mu + nu e_109 = 0` with
  `mu >= 0` (`lsq_linear`, one-sided bounds); `d(phi)/d(x109) = -nu`. The dual of A; its
  residual says whether the point is a KKT point of the *pinned* problem.
- **C, against the function.** Walk the null direction that moves `x109` by `h` and diff
  the objective the port actually returns — no linear algebra trusted.

### Numbers

At the tightest feasible iterate of the warm solve (`tolerance = 1e-13`, 96 iterations;
that iterate has scaled `max|eq| = 1.08e-15`, worst `g = +2.1e-17`):

```
A is (21, 23), rank 21, singular values 4.216 ... 0.129    -> LICQ holds, well conditioned
||D grad f||  = 5.4005e+00
||r||         = 2.6259e-02      ||r_perp|| = 2.4515e-06   (4.54e-07 of ||D grad f||)
A: d(phi)/d(x109) = -7.720953e+00
B: d(phi)/d(x109) = -7.720953e+00
   KKT relative residual, no pin   4.8624e-03      with pin 4.5394e-07
   multipliers c24 +4.462e-01, c83 +1.431e+00, c62 +2.674e-01, c35 +1.898e-02  (all >= 0)
C: predicted/actual df   1.0000 (h=1e-9), 0.9999 (1e-8), 0.9989 (1e-7), 0.9892 (1e-6)
```

That number uses AD's `c24` row, and `c24` is **not differentiable at that point** (§3), so
the row is the clamped-branch derivative rather than the derivative. Redone on a problem
that *is* differentiable at its own answer — the same model with the clamped square root
softened by `eps = 1e-4` — the same three routes give:

```
275 iterations, conv 9.11e-14; objf 1.217914834478, x109 2.994552943306e-02
same four active inequalities, |g| <= 3.3e-12
||r|| 1.0465e-04   ||r_perp|| 2.1838e-06
A and B: d(phi)/d(x109) = -3.016745e-02
   KKT relative residual, no pin   1.9371e-05      with pin 4.0423e-07
```

### The error bar, and the thing that matters more than it

**`d(phi)/d(x109) = -3.02e-02` is the number to quote** (the differentiable problem);
`-7.72` is the same construction on the raw problem and the factor of 256 between them is
the cost of the `c24` kink, not numerical error. Within each problem:

- routes A and B agree to every printed digit in both (they are duals — this checks the
  arithmetic and the active-set signs, not the model);
- route C reproduces route A against the function to `1e-04` relative at `h = 1e-09`;
- across the last twelve SQP iterates of the raw solve, every iterate carrying the full
  four-inequality active set gives `-7.7209` to `-7.7210`, a spread below `1e-04` relative;
- the KKT residual with the pin bounds the ambiguity at `4.5e-07` / `4.0e-07` of
  `||grad f||`, against a pin contribution of `4.9e-03` / `1.9e-05`.

**The first-order model has a range of `|Δx109| <= 1e-08`, and that is the load-bearing
caveat.** On the smoothed problem the directional check degrades from ratio `0.973` at
`h = 1e-08` to `0.729` at `1e-07` and `-1.71` at `1e-06`; the active manifold's curvature
is ~`3e+07`. So:

- `d(phi)/d(x109) = -3.0e-02` (the derivative) and
- `+0.2577` (the **secant** to the right, measured directly from four converged feasible
  pinned solves at `δ = +5e-06 … +1e-04` on two versions of the problem, slopes `+0.2552`,
  `+0.2565`, `+0.2578`, `+0.2581`)

are both correct and are different quantities. `phi` turns around within `1e-08` of the
free optimum. **Quoting any reduced gradient as an estimate of what moving to PROCESS's
`x109` costs would be wrong by orders of magnitude**; the number is good for its sign and
for the KKT statement, and for nothing else.

### What it settles

`d(phi)/d(x109)` is **not** zero at the free optimum, on either version of the problem, by
three to seven orders of magnitude relative to the numerical floor. H1 required it to
vanish; it does not. But `-3.0e-02` against a KKT residual of `1.9e-05` is a *weak*
violation, and on its own it would not have been decisive — §4's release experiments are
what settle H1, not this.

---

## 3. The mechanism: a square-root kink of PROCESS's own model, sat on to 1.9e-09

### How it was found — two signals, neither looked for

**(a) AD against a central difference of the port's own SAND condition map.** MDF's ladder
has this as Stage B0; SAND had none, and this is it. At the free optimum, over relative
steps `1e-05`, `1e-06`, `1e-07`, exactly **2 of 690 cells** exceed `1e-06` and both are in
the `c24` row:

```
^cond.constraints.c24  d/d .physics.temp_plasma_electron_vol_avg_kev
    AD +5.204885e-01   FD +2.137806e+01 (1e-5) / +6.374040e+01 (1e-6) / +6.841075e+01 (1e-7)
```

The finite difference *grows* as the step shrinks. Every other cell agrees to `<= 5e-11` at
step `1e-05` — including the whole `objf` row (worst `7.65e-10`) and the whole `x109`
column (worst `5.46e-11`).

**(b) Drift along the null direction, per condition, as a log-log slope in the step.**
Every condition drifts like `h^2.00` — ordinary curvature, the linear term being zero by
construction. One does not:

```
^cond.constraints.c24   5.25e-05  2.04e-04  6.57e-04  2.08e-03  6.58e-03    slope +0.52
(next largest)          7.62e-14  7.60e-12  7.60e-10  7.60e-08  7.60e-06    slope +2.00
```

`h^0.52` is `sqrt(h)`.

### The site

`functional_process/models/physics/physics_A_pure_formulas.py:342-348`, in
`fast_alpha_beta` — a faithful port of `process/models/physics/physics.py:4265-4392`:

```python
above = temp_sum_20 - 0.65
positive = above > 0.0
fact = jnp.minimum(
    0.30,
    0.26 * density_ratio_sq
    * jnp.where(positive, safe_sqrt(jnp.where(positive, above, 1.0)), 0.0),
)
```

`safe_sqrt` is not at fault: it changes the derivative only *at* exactly zero. This is the
ordinary mathematics of `sqrt(max(0, x))` at `x = 0` — finite (zero) slope on one side,
unbounded on the other — and it is PROCESS's own formula, ported correctly.

### Every converged point of this problem sits on it

| point | `objf` | `x109` | `temp_sum_20 - 0.65` |
|---|---|---|---|
| free optimum, VMCON | 1.21775734 | 0.0299518330 | **+1.91e-09** |
| pinned `x109 = 0.0300` | 1.21776977 | 0.0300000000 | **-1.27e-10** |
| pinned `x109 = 0.0320` | 1.21757404 | 0.0320000000 | **-1.26e-10** |
| free with `x109 >= 0.031`, VMCON | 1.21763602 | 0.0310000000 | **+1.08e-10** |
| free with `x109 >= 0.031`, SLSQP | 1.21763602 | 0.0310000000 | **-2.55e-13** |
| **PROCESS's own converged point** | — | 0.0335904061 | **-5.757e-03** |

At the free optimum
`temp_plasma_electron_density_weighted_kev = 6.6666666862`,
`temp_plasma_ion_density_weighted_kev = 6.3333333519`,
`temp_sum_20 = 0.650000001909`.

This is not a coincidence about one answer. **The port's whole feasible solution family
lies on the ridge**, and every QP the solvers assemble is linearised on it.

### What it does to the QP

Walking the null direction (which by construction holds every *linearised* active
constraint fixed), `c24` — the beta limit — behaves like this:

```
    dx109            c24         d c24    (d c24)/h            objf
 -1.0e-05  -2.029857e-06   -2.0299e-06   +2.030e-01    1.2178429052
 -1.0e-06  -2.030443e-08   -2.0304e-08   +2.030e-02    1.2177651427
 -1.0e-07  -2.030501e-10   -2.0305e-10   +2.030e-03    1.2177581111
 -1.0e-08  -2.031167e-12   -2.0307e-12   +2.031e-04    1.2177574154
  1.0e-08  +2.043070e-04   +2.0431e-04   +2.043e+04    1.2177572609
  1.0e-07  +6.569293e-04   +6.5693e-04   +6.569e+03    1.2177565669
  1.0e-06  +2.080767e-03   +2.0808e-03   +2.081e+03    1.2177497008
  1.0e-05  +6.578802e-03   +6.5788e-03   +6.579e+02    1.2176884827
```

Downward: linear, `c24` going comfortably feasible. Upward: `2.0*sqrt(h)`, slope
**unbounded as `h -> 0`**. The objective genuinely falls upward (`-6.9e-05` at
`h = 1e-05`) and `c24` is violated by `+6.6e-03` at the same point, where the QP was told
it would not move at all. Every step towards PROCESS's `x109` is predicted feasible and is
actually infeasible by `O(sqrt(step))`.

### Why PROCESS is unaffected

Two reasons, both structural:

- PROCESS's own converged design sits `5.8e-03` *below* the switch-on temperature — it
  never touches the ridge;
- PROCESS differentiates by finite differences at `epsfcn = 0.01`, a **1 % relative**
  perturbation (`3.4e-04` in `x109` on this run), 10^5 times wider than the feature. Its
  gradient is a secant straight across it.

That is the whole rewrite's trade running in the unflattering direction for once: exact
derivatives resolve a non-smoothness that a 1 % finite difference smooths away, and the
non-smoothness is real, in PROCESS's own model.

### Smoothing it does *not* move `x109`, and why that is not a refutation

Replacing the clamp with `sqrt(0.5*(above + sqrt(above^2 + eps^2)))` at `eps = 1e-4`
(agreeing to `O(sqrt(eps))`, bounded derivative) gives a clean solve — 275 iterations,
`conv 9.11e-14`, feasible — at `x109 = 0.0299455280`, `0.02 %` from the unsmoothed answer,
with `above = -5.512e-04`.

The attractor moved by **exactly the smoothing width**, not away from it. Comparing the
two optima gives `d(temp_sum_20)/d(x109) ~ 87` along the direction the answer moved (a
secant across two problems, so an order of magnitude, not a derivative), so
`|above| <= eps = 1e-4`
is a window of `|Δx109| <= 1.1e-06` — five orders of magnitude narrower than the `3.6e-03`
gap. `eps = 1e-4` therefore tests only whether the *exact* kink point is essential; it is
not, and the ridge is. Removing the barrier needs an `eps` large enough to change the
model materially, which is a different experiment and not a controlled one.

---

## 4. The experiments that killed H1's account

`phi(x109)` from converged, feasible pinned re-solves (`tolerance = 1e-13`, coarse
continuation chain, accepting only `max|eq| < 1e-08` and worst `g < 1e-08`):

| `x109` | `objf` | vs free | `max\|eq\|` | worst `g` |
|---|---|---|---|---|
| **0.0299518330** (free) | 1.21775734 | — | 2.1e-07 | +1.0e-05 (`c24`) |
| 0.0300000000 | 1.21776977 | **+1.24e-05** | 3.2e-13 | +4.5e-14 |
| 0.0310000000 | 1.21763602 | **−1.21e-04** | 4.0e-14 | +4.4e-16 |
| 0.0320000000 | 1.21757404 | **−1.83e-04** | 4.0e-14 | +1.3e-13 |
| 0.0330000000 | 1.21758366 | **−1.74e-04** | 4.2e-11 | +7.2e-13 |
| 0.0335904061 (PROCESS's) | 1.21754949 | **−2.08e-04** | 2.7e-09 | +1.2e-06 |

The `0.0310` row is not a pin: it is the answer of the *free* problem with `x109 >= 0.031`
imposed, which converged **on** that bound, so it is the pinned optimum there by
construction — and a better-conditioned one than a pin, since the bound was allowed to be
inactive and was not.

So H1's landscape claim is **confirmed**: a barrier at `x109 ≈ 0.0300` (height `1.2e-05`)
and a strictly better region beyond `0.031` (depth `2.1e-04`). Then:

**Release the pin from `x109 = 0.0330` (`objf 1.21758366`, feasible):**

```
free from far side, VMCON   objf 1.21775507  x109 0.0299518246  max|eq| 4.2e-08  worst g +1.0e-05
free from far side, SLSQP   objf 1.21775734  x109 0.0299518329  max|eq| 2.0e-13  worst g +5.8e-15
```

**Free again from `x109 = 0.0320` (`objf 1.21757404`, feasible) with `x109 >= 0.031`:**

```
VMCON   objf 1.21763602  x109 0.0310000000 (ON THE BOUND)  max|eq| 2.0e-13  worst g +2.4e-06
SLSQP   objf 1.21763602  x109 0.0310000000 (ON THE BOUND)  max|eq| 4.0e-14  worst g +4.4e-16
```

Both experiments say the same thing twice, with two independently written SQPs, and SLSQP
reaches full feasibility both times: **started at a feasible point, the free solvers
converge to a feasible point with a higher objective, always by decreasing `x109`.** A
correct local method does not do that. The solvers are not falling into a basin at
`0.0299518` — they are being driven towards smaller `x109` by a QP whose model of `c24` is
wrong on every iterate, and they stop wherever that drive runs out (the ridge's end, or a
box bound).

That is the discriminator the whole investigation turned on, and it costs one solve.

---

## 5. Two claims of §11.10 checked rather than trusted

**"Both build the problem through the shared `scaled_problem`" is false as code and true in
effect.** `VmconDriver.__call__` (`drivers.py:661-668`) keeps its own inline
`np.divide(1.0, flat_start, out=scale, where=flat_start != 0.0)` — the exact-zero guard —
while `design_scale` (`drivers.py:43-69`) has the real `UNSCALABLE_BELOW = 1e-12` floor,
and only `scaled_problem`/`SlsqpDriver` call it. Confirmed in this worktree. **Measured at
the warm seed: `0 of 23` scale entries differ.** The one unknown below `1e-12` is
`.power.qac`, and on a seeded env it is *exactly* `0.0`, where both rules return `1.0`. The
duplication is real, it is a live trap on any restart from a solved point
(`qac = -3.8e-27` there), and it is **inert for the starts that produced the eight-digit
VMCON/SLSQP agreement**. §11.10's inference is sound even though its premise is not.

**SLSQP re-measured on SAND, warm, `tolerance = 1e-12`:** `x109 = 0.0299518055` against
VMCON's `0.0299518330` — seven digits — and SLSQP's point is fully feasible
(`max|eq| 2.8e-13`, worst `g 7.8e-16`) where VMCON's is not (`worst g +1.0e-05` on `c24`,
which is the ridge showing through).

**§11.11's `x109` pinning row is reproduced exactly.** Pinned at PROCESS's own
`0.033590406` at `tolerance = 1e-13` by an independent continuation chain:
`objf 1.21754949327` against §11.11's `1.217549493` — nine digits. The residual inequality
violation there is `+1.22e-06`; buying `1.7e-04` of objective with it would need a
multiplier of ~140 and the largest `|lambda|` anywhere in the problem is `1.43`, so **the
improvement is real and is not bought by the violation.** §11.11's central claim stands,
and it is now corroborated by two cleanly feasible points (`0.0320` and `0.0330`,
`max|eq| <= 4.2e-11`) that need no such caveat.

---

## 6. What H3 actually predicts, worked out

H3 is the subtlest of the four and the honest answer is that it predicts nothing about
`x109` **once the port's own objective is the yardstick** — and that is available without
measuring anything. §11.11 already established a feasible point with the *port's own*
objective `1.7e-04` lower than the port's answer. An account of the form "the port descends
a subtly different objective and stops at that objective's stationary point" cannot explain
a solver failing to reach a better point of the objective it is actually descending. H3
could only bite if the port's answer were a genuine stationary point of the port's problem
— which §4 rules out.

Tested anyway, in the one form that is falsifiable: **is the port's `objf` gradient the
true gradient of the port's own objective?** Yes — worst cell `7.65e-10` at relative step
`1e-05`, `2.06e-09` at `1e-07`, against a central difference of the same condition map. The
18–34 % disagreement §10.5c records is entirely *port-vs-PROCESS* (the `z_tf_inside_half`
report-pass/solve-pass inconsistency) and leaves the port's own descent direction correct.

H3 therefore survives as a true statement — the port and PROCESS solve different problems,
in the objective *and* in `c16`, which is an equality, so the two feasible sets differ —
and is excluded as the explanation for `x109`.

---

## 7. Proposed corrections to §11.11

Proposals, not edits — §11.11 is being handled in the main tree.

**(a) The one-line answer.** `x56` is a flat direction; `x109` is not, and treating them as
one thing is what sent two investigations down the same wrong road.

> old: **The one-line answer: both are flat directions of the port's own constrained
> problem, and the eighth missing producer that everyone was looking for is real but is the
> cause of something else entirely.**

> new: **The one-line answer: `x56` is a flat direction of the port's own constrained
> problem; `x109` is not. The port's answer is the point where the design first reaches
> `(Te + Ti)/20 == 0.65` — the kink in `fast_alpha_beta`'s clamped square root, which the
> port stops on to 1.9e-09 — and the solvers are held there by a `c24` Jacobian row that is
> not the derivative of the `c24` they evaluate. The eighth missing producer that everyone
> was looking for is real but is the cause of something else entirely.**

**(b) The `x109` bullet.** Its landscape claim is right; its implication that the port's
answer is one of two stationary points is not.

> old: … the port's converged answer is **not** the global optimum of the port's own
> problem — the landscape has at least two stationary points 12 % apart in `x109` and
> 1.7e-04 apart in objective.

> new: … the port's converged answer is **not** the global optimum of the port's own
> problem. The landscape is confirmed multi-modal — a barrier of height `1.2e-05` at
> `x109 ≈ 0.0300` and a strictly better feasible region beyond `0.031`, reaching
> `objf 1.21757404` at `0.0320` with `max|eq| 4.0e-14` — but the port's answer is not the
> other minimum of it. Released *inside* the better region, both VMCON and SLSQP walk back
> out of it uphill; with `x109 >= 0.031` imposed they converge **on that bound**. See
> `x109_hypotheses.md`.

**(c) "What actually determines the landing point".** Keep the active-set paragraph — it is
right and was re-measured (same four inequalities, `A` is 21×23 of rank 21, LICQ holds).
Replace the gradient-error account:

> old: … the objective's own gradient in those directions is what fixes them. That gradient
> is the port's weakest quantity: the `objf` row of the Stage B Jacobian disagrees with
> PROCESS's by 18–34 % in *every* column (§10.5c) … A 20–30 % error in the gradient along a
> valley this flat moves the landing point by ~10 % in the flat coordinates while barely
> moving anything else — which is exactly the observed shape … That is a *consistent*
> account rather than a demonstrated one.

> new: … the objective's own gradient in those directions is what would fix them, and it is
> not what does. **Measured against a central difference of the port's own condition map,
> the `objf` row is correct to `7.7e-10`** — the 18–34 % of §10.5c is a *port-vs-PROCESS*
> difference, not an error in the port's own descent direction. What fixes the landing
> point is one row of the *constraint* Jacobian: `c24`, at a point where its function is
> not differentiable. Of 690 cells, exactly the two `c24` cells disagree with the same
> central difference, and `c24` alone drifts like `h^0.52` along the null direction where
> every other condition drifts like `h^2.00`. §10.9 item 3 (pinning
> `.heat_transport.p_plant_electric_base_total_mw`) is no longer the controlled test for
> `x109`; it tests the `objf`/`c16` difference, which is real and is a separate question.

**(d) Keep unchanged**, all independently reproduced here: the pinning table
(`objf 1.217549493` at PROCESS's `x109`, reproduced to nine digits), the `x56` reading, the
two driver defects, and the `c16`-dropping warning.

**(e) Do not apply, as written, any correction that says the port's SQP terminates
prematurely.** It does not. On the smoothed problem it converges to `conv 9.11e-14` with a
KKT relative residual of `1.94e-05`; on the raw one, three solver/architecture combinations
reach the same point to seven-to-nine digits from four different starts, and SLSQP reaches
full feasibility there. The stopping is a *correct* response to an *incorrect* linear model
of one constraint. That is a different defect with a different fix, and the premature-
termination framing would send the next reader into `pyvmcon`'s convergence test, which is
the one place the fault is not.

---

## 8. What is left open

- **A fix.** Not attempted. The options are visibly different in kind: (i) treat the
  `temp_sum_20 > 0.65` switch as a declared branch (an `Alternative`) rather than a clamped
  square root inside one node, so the two arms are two smooth problems; (ii) give the SQP a
  second-order or trust-region safeguard that detects the `sqrt` violation; (iii) leave it
  and record that the port's answer is the ridge. Choosing needs a view on whether the
  *model* is meant to be differentiable there, which is a PROCESS question, not a port one.
- **How wide the ridge is, and where `phi` really turns.** `0.0305`, `0.0315`, `0.0325`,
  `0.0334` and `0.0336` would not converge from continuation. Whether `phi` is genuinely
  non-monotone between `0.0300` and `0.0310` or the SQP simply fails there is **not**
  settled by anything here.
- **The pinned-solve failure mode, recorded rather than solved.** Of 14 fine pins around
  the free optimum, 1 converged to full feasibility; the failures are catastrophic
  (`objf 3.29` with `max|eq| 1.79`; one pin returning `objf 1.02e+23`) rather than a stall
  near the answer, and they happen from a converged neighbour one step away. That is worth
  a line in any future pinning study: continuation does not rescue this problem, coarse
  steps sometimes work where fine ones do not, and success is not monotone in step size.
- **How many other clamped roots are being sat on.** The audited class
  (`next_steps.md` §9, §10.5b, §11.8) is about `nan` derivatives *at exactly zero*; this is
  the neighbouring failure — a finite derivative on one side and an unbounded one on the
  other, `1.9e-09` from the switch — and nothing in this repo looks for it. **SAND has no
  Stage B0**, and adding one (AD against a central difference of the SAND condition map at
  the answer, which is what found this in `2 of 690` cells) is cheap and would.
- **`x56`.** Untouched; §11.11's reading of it as a flat direction is not disturbed by
  anything above.

## 9. Scripts

`/tmp/claude-1000/-home-tbogaarts-PROCESS/fcc41886-e8dd-4982-b914-019a965e8013/scratchpad/x109/`
— `cache_ref.py` (one PROCESS run, pickled), `common.py`, `e2`/`e3`/`e6` (reduced
gradient), `e4`/`e6` (pinned scans), `e5` (MDF), `e7`/`e12`/`e13` (release and bounds),
`e9` (AD-vs-FD and drift scaling), `e10`/`e11` (the kink, and the smoothed problem).
Scratch, not committed.
