# ALC on Helias: implementation and convergence log

Augmented Lagrangian coordination (Tosserams, Etman & Rooda 2008, non-hierarchical form)
of the Helias stellarator problem split into **build**, **operation** and **economics**.
The code is in `alc.py`; the partition, the AAO base and the shared driving code are in
`partition.py` (renamed from `common.py`); `benchmark.py` compares AAO, the sequential
split and ALC. This folder moved from `PROCESS/distributed_architectures/` (gitignored
scratch) to `paper_tests/distributed_architectures/` on 2026-11-11. This file records what
the implementation is, every change convergence needed and why, and every run with its
result. Keep it current.

Reference: the file's own problem under SAND (AAO) converges from the file's start to
**COE 1.2184414**, in 23 iterations with VMCON and 16 with SLSQP.

## Departures from Tosserams, Etman & Rooda (2008)

The decomposition, the relaxation, the multiplier update (Eq. 24), the block-coordinate
descent over the disciplines and the equivalence argument are the paper's. Three choices
in how the coordination is run are not. Each was needed to reach AAO's optimum here
(section 3), and each is a parameter or a test in `alc.Coordinator`:

1. **The inner loop ends when the copies settle, with a fixed tolerance.** The paper
   either runs block-coordinate descent until the relaxed objective `F` changes by less
   than ε/100 (Eq. 28), or uses an inexact loop whose tolerances fall to zero
   (Section 5.5). Here the multiplier update is taken once no copy moves by more than
   10⁻³ (relative) over a sweep, or after 25 sweeps. Near the answer every sweep
   qualifies, so the tail runs as alternating direction (2025 updates in 2336 sweeps).
   The paper says that variant is outside its convergence proof and "terminates
   prematurely at non-optimal designs" on some problems.
2. **The stopping test is stricter than the paper's.** The paper stops when
   `‖q^k − q^{k−1}‖∞ < ε` and `‖q^k‖∞ < ε` (Eqs. 26–27). Here it also requires the
   multiplier step `max|2w²c|` ≤ 10⁻³, the copies' movement ≤ 10⁻⁶ and COE's movement ≤
   10⁻⁷ over a sweep. The paper's two tests were met at R3's and R4's non-optimal points,
   which AAO, started there, left for its own optimum.
3. **β = 1: the weights are never raised.** The paper's Eq. 25 has β > 1, with β = 2.2,
   γ = 0.4 for its exact inner loop and β = 2.0, γ = 0.5 for the inexact one. With
   β = 2.2 the weights rose to 23 in the transient, and the design then crept for
   thousands of sweeps (R6). β = 1 is still the method of multipliers, which converges at
   a fixed weight large enough to make the relaxed Lagrangian's Hessian positive
   definite, the paper's own condition (Section 5.4).

The other changes are tuning within the paper's algorithm: SLSQP as the subproblem solver,
each subproblem's objective scaled by its penalty curvature, no weight raised below
tolerance (moot at β = 1), and w₀ = 1 instead of Eq. 29 (which would give about 0.2–0.5
here). Section 3 has the evidence for every change, and section 3's table classifies them
all.

## 1. The formulation

| Discipline | Models | Own design variables | Constraints | Copies held | Objective |
|---|---|---|---|---|---|
| Build | 29 | `b_t`, `rmajor`, `tdmptf`, `fcutfsu` (plus `wp_width_r_min`, absorbed by SAND) | c32, c34, c35, c65, c82, c83 | 0 | none (penalty only) |
| Operation | 40 | `te`, `dene`, `hfact`, `f_alpha` (plus 3 SAND states) | c2, c8, c17, c18, c24, c62, c67 | 6 of build's outputs | none (penalty only) |
| Economics | 50 | none | c16 | 25 of build's outputs, 12 of operation's | COE |

- The assignment is `partition.disciplines`. Models are assigned by design context:
  costing, plant power, buildings, availability, vacuum, blanket and shield go to
  economics; the vessel and cryostat geometry and the coil cryostat area go to build. It
  refuses any back flow and any cycle that spans two disciplines.
- 21 design-dependent models feed neither a constraint nor COE; they are evaluated after.
- Five crossing places never move with the design. They were checked by moving each
  design variable by 1 % (`partition.constant_links`): `dr_fw_inboard`, `dr_fw_outboard`,
  `eps`, `m_fuel_amu` and `n_tf_coils`. These are read directly, with no copy.
- Each copy has one consistency constraint, `c = (copy − value) / |reference|`, where the
  reference is the MDA value at the file's start. That gives 43 constraints. None is
  redundant: a quantity that both operation and economics read is checked against build
  twice, never operation against economics.
- Each constraint is relaxed by `φ(c) = vᵀc + ‖w∘c‖²`. Each discipline minimises its share
  of the objective plus the `φ` terms of the constraints that touch its own variables,
  subject to its own constraints.
- **Equivalence.** At `c = 0` the three subproblems' KKT conditions combine into AAO's,
  with `v` as the multipliers of the consistency constraints. This needs no convexity;
  convergence is only proven for convex problems, so here it is empirical.
- No derivative is taken through any optimiser. Each subproblem differentiates its own
  models (`jax.jacfwd`); what it reads from the others is fixed during its solve.

## 2. The graph

Every recipe starts from the **AAO base** (`partition.base`): `problem_model`, the
machine, every constraint body, the merit, and `Opt` over all eight variables, which is
what AAO's `SAND` is applied to. Recipes use primitive ops only, one at a time, in the
three steps of jaxgraph's `examples/sellar_plans.py`. Where the Sellar base holds each
bound and constraint as its own requirement node, this base holds all 30 inside `Opt`,
and no op splits a node. So a recipe undetermines and deletes `Opt` ("what the
architecture replaces"), then inserts each of its problems with its share of `Opt`'s
relations, unchanged. Inside each problem it does what AAO's SAND does to the whole,
taken from AAO's own recipe on the same base: the same two cuts and closures (both in
operation) and the coil's implicit solve (build) combined in.

The ALC recipe (`alc.recipe`, 111 ops):

1. **Remove.**
   - Every cut first, while `Opt` still determines the design: the 43 link cuts onto
     `OperationCopy` / `EconomicsCopy`, then AAO's three cuts of operation's cycles.
   - `Opt` and the coil solve undetermined, and `Opt` deleted.
2. **Assemble.**
   - Per discipline, in binding order: its problem (a condition node determining
     nothing, with `Opt`'s bounds of its design and its constraints) and its objective
     function.
   - `consistency` (`c` at the end of a sweep) and AAO's two closures.
   - The closures combined into operation, the coil solve into build.
3. **Add.**
   - Each discipline determined: its design, the unknowns of what was combined into it,
     and its copies.
   - The upstream objectives' reads of downstream copies cut, giving the Gauss–Seidel
     order build → operation → economics.
   - The coordination problem, `Coordination`, inserted and determined. It owns the
     43 swept copies, the last COE, `v` and `w`, and holds `copy = value`,
     `merit_seen = merit` and the consistency `c = 0`.
   - The three problems nested in it.

The sequential recipe (`sequential.recipe`, 15 ops) is the same without copies or
coordination: AAO's cuts; `Opt` and the coil solve undetermined, `Opt` deleted; the
magnet stage (its bounds with `b_t ≥ 5`, the coil constraints, `0.2 rmajor`) and the
plasma stage (its bounds, the other constraints, COE) inserted, the closures and the solve
combined in; each determined. Both stages sit at the root, and the plasma runs after the
magnet because it reads the magnet's answer.

Before 2026-11-11 both recipes declared their problems on the machine afresh and
applied a per-problem SAND (`SandEach`, a composed op). The rebuilt sequential split
gives the same COE (1.2721691). ALC on the rebuilt recipe (R9) reproduces R8 in every count.

**Drivers:**
- Each discipline: `SlsqpDriver`.
- `Coordination`: `alc.Coordinator`, ALC's own driver, for its own contract
  (`alc.AugmentedLagrangian`, accepting a coordination: unknowns moved to their
  right-hand sides, the others on no relation, and equalities held at zero). Neither is
  part of cottax's or the port's standard set.
  - Each iteration it reads the model's outputs at the current point (one sweep of the
    disciplines: the copies' next values, and `c`). From them it decides the next point,
    as any driver does: the copies move to what they copy; `v` and `w` follow the method
    of multipliers on `c` once the copies settle.
  - It is traced with `lax.while_loop`, so it runs under `study.solve`'s `jit`; the
    nested SLSQP solves are host callbacks inside it.
  - Until 2026-11-11 the multiplier rule was a function node in the graph
    (`multiplier_update`, owning `v_next` and `w_next`, with the previous `c` an unknown
    of the coordination). It was tagged as a FR-12 workaround, but this is not FR-12.
    FR-12 is a driver stepping a fixed schedule (a weight ladder, a DOE). ALC's driver
    decides its next point from the model's outputs, as every optimiser and root finder
    does; it just needs implementing. The rule moved into the driver unchanged (R10
    reproduces R9).

**Workarounds (tagged in the code):**
- `WORKAROUND(FR-5)`: `OperationCopy` and `EconomicsCopy` mints, written by hand.
- `WORKAROUND(FR-10)`: plain jax bodies for the inserted functions.
- FR-1 is not worked around. Every sweep starts each subproblem from the same start,
  which costs time, not correctness.

## 3. What convergence needed

Each change, the evidence that forced it, and why it leaves the problem unchanged. Run
numbers refer to the table in section 4.

1. **SLSQP instead of VMCON for the disciplines** (R0, R0b).
   - *Evidence.* VMCON refused operation and economics in the first sweep (status 4,
     non-finite). With the scaling of change 2, VMCON's iterates took operation's copies
     of the plasma surface and volume to −332 and −56 by iteration 4, and the physics went
     NaN. With the same start, SLSQP converges all three subproblems in one sweep (9, 14
     and 12 iterations).
   - *Why it is safe.* A different driver for the same subproblems; no bounds are added to
     the copies.
2. **Each discipline's objective divided by `2·mean(w²)` over its terms** (`penalty_body`).
   - *Evidence.* Unscaled, the penalty's curvature (`2w²` ≈ 60 at w = 5.6) is about 60×
     the identity that an SQP's first step assumes in its scaled coordinates. The first
     step is then about 100 start-magnitudes long and ends on a NaN model.
   - *Why it is safe.* `w` is fixed during a subproblem solve, so no minimiser changes.
3. **The inexact inner loop instead of pure alternating direction** (R1 → R2).
   - *Evidence.* With an update every sweep, Tosserams' rule ("raise `w` where `|c|` fell
     by less than γ = 0.4") fired almost every sweep: one sweep cuts `|c|` by only about
     0.8. The weights reached 7·10¹⁶ by sweep 60 and `c` oscillated between 0.13 and 0.72.
   - *Change.* `Coordinator` keeps the multiplier update only once the copies settle
     (relative change ≤ 10⁻³ over a sweep) or after 25 sweeps at fixed multipliers.
     `--inner-tolerance inf` restores alternating direction.
4. **No weight raised on a constraint already within tolerance** (R2 → R3).
   - *Evidence.* R2 reached `max|c|` = 4.7·10⁻⁵ by sweep 86. The weights then kept rising
     ×2.2 per update at the noise floor (`c` ≈ 10⁻⁴), reached 10⁹ and the run broke at
     sweep 112 (`c` back to 0.84; weights 10²⁶ by the end).
   - *Change.* `update_body` raises `w_j` only where `|c_j| > tolerance` as well.
   - *Why it is safe.* It only spares the conditioning; a constraint already met needs no
     stiffer penalty.
5. **A stopping test that sees the design creep** (R3 → R4 → R6).
   - *Stopping on `max|c|` alone* (R3) stopped at consistent but unpriced designs, +1.7 %
     on COE.
   - *Adding the multiplier step `max|2w²c|` ≤ 10⁻³* (R4) still stopped early: +1.7 % from
     w₀ = 5.6, +0.39 % from w₀ ≈ 1. AAO (SAND, VMCON) started at either point moved to
     its own optimum, 1.2184414, so neither was a minimum.
   - *Cause.* Under stiff penalties each discipline moves only as far as the others' last
     values let it. The design creeps toward the optimum while `c` and the multiplier step
     already look converged.
   - *Change.* The coordination carries COE (`.alc.merit`). The run stops only when all of
     these hold at an update:
     - `max|c|` ≤ 10⁻⁴;
     - the multiplier step ≤ 10⁻³;
     - the copies moved ≤ 10⁻⁶ over the sweep;
     - COE moved ≤ 10⁻⁷ over the sweep.
6. **Initial weights 1 on the scaled inconsistencies**, instead of 5.6.
   - *Evidence.* w₀ = 5.6 was chosen so that a 1 % inconsistency on every constraint costs
     10 % of COE. Penalties that stiff settle the copies before the prices: R4 stopped at
     +1.7 %, against +0.39 % from w₀ ≈ 1.
   - *Correction.* An earlier version of this report called w = 1 "Tosserams' usual
     start". The paper does not say that. Its rule (Section 5.3, Eq. 29) is
     `w = √(α|f̂| / q̂ᵀq̂)` with α = 0.1, where `q̂` comes from one solve at small weights.
     The first sweep here has `|c|` of about 0.1–0.5 on 43 constraints, which would put
     that rule's w at roughly 0.2–0.5. So w = 1 is near what the rule would give, but it
     is a choice made here.
7. **Weights held at their start: β = 1 instead of Tosserams' 2.2** (R6 → R7).
   - *Evidence.* With β = 2.2 the weights rose from 1 to 23.4 in the first 300 sweeps,
     during the transient, and then stayed (change 4). From there each sweep moved the
     design along the prices by about `1/(2w²)`, so it crept: +1.7·10⁻⁵ on COE and
     `f_alpha` 31 % off after 3000 sweeps, still moving. With β = 1 and w = 1 the
     multipliers alone close the gaps. It stopped by its own test at sweep 2336 on AAO's
     optimum, design included.
   - *Why it is safe.* The method of multipliers converges at a fixed weight, locally for
     a nonconvex problem if the weight is large enough. Raising it is a safeguard, not
     part of the equivalence.
8. **Ruled out: SLSQP's `ftol`** (R5).
   - 10⁻¹³ instead of 10⁻⁸ moved COE in the fifth digit only (1.238874 against 1.238880),
     so it was reverted.

**Not changed:**
- No bounds and no constraints were added to the copies or the disciplines; adding any is
  for you to approve first.
- γ = 0.4 is unused at β = 1 (no weight is raised); kept for `--beta` above 1.

### Architecture or tuning? Against Tosserams, Etman & Rooda (2008)

None of the changes alters the architecture. These are all as the paper has them:
- the decomposition: disciplines, copies, one consistency constraint per copy, the
  non-hierarchical coupling (Section 4);
- the relaxation `φ = vᵀc + ‖w∘c‖²`;
- the method-of-multipliers update `v ← v + 2w∘w∘c` (Eq. 24);
- block-coordinate descent over the disciplines in a fixed order (Section 5.2);
- the equivalence argument (Section 5.4): KKT points of the relaxed problem at `c = 0`
  are KKT points of the original.

The changes are choices within the method. Several depart from the paper's stated
algorithm or parameters:

| Change | The paper | Here | Kind |
|---|---|---|---|
| 1. SLSQP | no subproblem solver prescribed | SLSQP for every subproblem | tuning (solver) |
| 2. Objective divided by `2·mean(w²)` | not discussed | each discipline's objective scaled | tuning (conditioning); no minimiser changes |
| 3. Inner loop | exact BCD to `\|ΔF\|/(1+\|F\|) < ε/100` (Eq. 28), or inexact with tolerances non-increasing to 0 (5.5); alternating direction mentioned, "not within the assumptions of the convergence proof" and sometimes "terminates prematurely at non-optimal designs" | update once the copies move < 10⁻³ per sweep, or after 25 sweeps; in R7 2025 of 2336 sweeps were updates, so the tail runs as alternating direction | **departs**: the test is on the copies, not on `F`; its tolerance is fixed, not → 0; the 25-sweep cap forces updates. The paper's warning (premature stops) is exactly what R3/R4 met |
| 4. No raise below tolerance | Eq. 25 raises `w_i` whenever `\|q_i\|` fell by less than γ | also requires `\|c_i\|` > tolerance | tuning (a safeguard); moot at β = 1 |
| 5. Stopping test | `‖q^k − q^{k−1}‖∞ < ε` and `‖q^k‖∞ < ε` (Eqs. 26–27) | `max\|c\|` ≤ 10⁻⁴, multiplier step ≤ 10⁻³, copies moved ≤ 10⁻⁶, COE moved ≤ 10⁻⁷ per sweep | **departs**: stricter; added because the paper's two tests were met at R3/R4's non-optimal points |
| 6. w₀ = 1 | Eq. 29, α = 0.1, `q̂` from a small-weight solve | fixed 1 | tuning; near the rule's value (see the correction above) |
| 7. β = 1 | β = 2.2, γ = 0.4 with BCD; β = 2.0, γ = 0.5 with an inexact inner loop; β > 1 in Eq. 25 | β = 1: weights never raised | **departs** from the stated range; still the method of multipliers, which converges at a fixed `w` large enough for the relaxed Lagrangian's Hessian to be positive definite (the paper's own convergence condition, 5.4) |

Not tried: the paper-faithful variant, with an exact BCD inner loop on `F` (ε/100),
Eqs. 26–27, Eq. 29 initial weights and β = 2.2, γ = 0.4. It would show whether the
departures are needed or only faster. It needs the relaxed objective `F` carried by the
coordination (COE plus every `φ`), a small change.

## 4. Runs

AAO reference: 1.2184414. Each sweep takes about 0.1–0.15 s (all three disciplines); compilation is about 15 s.

| Run | Setting | Sweeps | Updates | max\|c\| | COE | vs AAO | Note |
|---|---|---|---|---|---|---|---|
| R0 | VMCON, alternating direction, w₀ 5.6, unscaled objectives | 2 | 2 | 13 | — | — | operation and economics refused at the start (status 4) |
| R0b | VMCON, one sweep, scaled objectives | 1 | — | — | — | — | operation NaN at iteration 4: copies of surface and volume negative |
| R1 | SLSQP, scaled, alternating direction | 60 (cap) | 60 | 0.62 | 1.106 | not consistent | weights 7·10¹⁶, `c` oscillating |
| R2 | R1 plus inexact inner loop (10⁻³, 25) | 400 (cap) | 104 | 1.0 | 1.058 | not consistent | `c` 4.7·10⁻⁵ at sweep 86, then broke; weights 10²⁶ |
| R3 | R2 plus weight guard, tolerance 10⁻⁴ | 94 | 8 | 1.8·10⁻⁵ | 1.239094 | +1.70 % | stopped on `c` alone |
| R4 | R3 plus multiplier-step stop, w₀ 5.6 | 100 | 14 | 4.3·10⁻⁷ | 1.238880 | +1.68 % | AAO from here: 22 iterations to 1.2184414 |
| R4b | R4 with w₀ ≈ 1.0 | 164 | 13 | 5.0·10⁻⁶ | 1.223238 | +0.39 % | AAO from here: 18 iterations to 1.2184414 |
| R5 | R4 and R4b with `ftol` 10⁻¹³ | 100 / 163 | 14 / 12 | ~2·10⁻⁶ | 1.238874 / 1.223270 | — | `ftol` ruled out |
| R6 | R4 plus COE and settled-copies stop, w₀ 1 | 3000 (cap) | 2831 | 8·10⁻⁹ | 1.2184626 | +1.7·10⁻⁵ | still creeping, about 2·10⁻⁷ per sweep; 392 s |
| R7 | R6 with β = 1 (weights fixed at 1) | 2336 | 2025 | 6.1·10⁻⁷ | 1.2184414 | ~0 | stopped by its own test; 290 s; design matches AAO |
| R8 | the defaults (= R7), with the AAO checks | 2336 | 2025 | 6.1e-07 | 1.2184418 | +2.7·10⁻⁷ | 238 s; AAO from ALC's answer: 3 iterations to 1.2184414, so the same minimum |
| R9 | R8 on the rebuilt recipe (AAO base, primitive ops), in `benchmark.py` | 2336 | 2025 | — | 1.2184418 | +2.7·10⁻⁷ | identical to R8 in every count; 251 ± 10 s warm over 5 runs |
| R10 | R9 with ALC's own driver (`Coordinator` updates `v`, `w`; no `multiplier_update` node) | 2336 | 2025 | — | 1.2184418 | +2.7·10⁻⁷ | identical to R9 in every count and in the design; 253 ± 24 s warm over 5 runs (laptop in use) |

R6 trajectory:

| Sweep | 150 | 300 | 750 | 1500 | 2250 | 3000 |
|---|---|---|---|---|---|---|
| COE | 1.22324 | 1.22052 | 1.21949 | 1.21884 | 1.21855 | 1.21846 |

The weights rose from 1 to 23.4 by sweep 300 and then stayed (the guard). The R6 design at
sweep 3000 against AAO's:

| | `b_t` | `rmajor` | `te` | `dene` | `hfact` | `tdmptf` | `fcutfsu` | `f_alpha` |
|---|---|---|---|---|---|---|---|---|
| R6 | 4.748 | 26.678 | 5.704 | 1.771e20 | 1.053 | 32.56 | 0.7201 | 0.0393 |
| AAO | 4.716 | 26.645 | 5.703 | 1.739e20 | 1.048 | 31.81 | 0.7177 | 0.0299 |

R7 trajectory:

| Sweep | 150 | 300 | 600 | 1200 | 1950 | 2250 |
|---|---|---|---|---|---|---|
| COE | 1.24637 | 1.21438 | 1.21864 | 1.21827 | 1.21845 | 1.2184414 |
| max\|c\| | 4.3·10⁻² | 1.1·10⁻² | 9.1·10⁻⁵ | 2.3·10⁻⁵ | 6.2·10⁻⁶ | 5.7·10⁻⁸ |

COE dips below the optimum (1.21438 at sweep 300) while the copies are still
inconsistent: the relaxed problem lets economics price a plant that build and operation
do not yet deliver. The R7 design against AAO's:

| | `b_t` | `rmajor` | `te` | `dene` | `hfact` | `tdmptf` | `fcutfsu` | `f_alpha` |
|---|---|---|---|---|---|---|---|---|
| R7 | 4.7164 | 26.6446 | 5.7027 | 1.73916e20 | 1.0480 | 31.822 | 0.7178 | 0.0299 |
| AAO | 4.7165 | 26.6445 | 5.7027 | 1.73918e20 | 1.0480 | 31.810 | 0.7177 | 0.0299 |

## 5. Comparison: AAO, sequential, ALC

`benchmark.py`, 2026-11-11, one CPU (laptop), single-threaded BLAS. Every case: SLSQP at
`ftol` 1e-8 and `max_iter` 500, scaled, from the file's design; every recipe on the AAO
base. Warm: mean ± standard deviation of 5 runs of the compiled graph. Every count was
identical in all runs. Snellius timings to follow.

| Case | COE | vs AAO | Max violation | Overarching loop | SLSQP iterations (solves) | Cold (s) | Warm (s) |
|---|---|---|---|---|---|---|---|
| AAO (SAND) | 1.2184414 | — | 2.2·10⁻¹³ | — | Opt 16 (1) | 10.8 | 0.09 ± 0.01 |
| Sequential | 1.2721691 | +4.4 % | 4.2·10⁻⁹ | — | magnet 11 (1); plasma 122 (1) | 10.1 | 0.69 ± 0.06 |
| ALC | 1.2184418 | +2.7·10⁻⁷ | 1.3·10⁻¹⁰ | 2336 sweeps, 2025 multiplier updates | build 31 345 (2337); operation 44 168 (2337); economics 11 210 (2337) | 296 | 251 ± 10 |

Per solve, ALC's disciplines take 13.4 (build), 18.9 (operation) and 4.8 (economics)
iterations. No solve failed in any case. The table is R9. R10, after the multiplier rule
moved into ALC's own driver, reproduces every count and digit; its times were 13.1 s /
0.12 s (AAO), 11.5 s / 0.80 s (sequential) and 249 s / 253 ± 24 s (ALC), cold / warm, on
the same laptop under more load.

- **Optima.** ALC and AAO agree to 2.7·10⁻⁷ in COE and to 3–5 digits in the design
  (`tdmptf` 31.82 against 31.81 is the largest difference). The sequential split is a
  different, worse design (+4.4 % COE: `b_t` held at 5 T, `tdmptf` 4.1 s).
- **Cost.** About 2800× AAO's warm time: 86 723 SLSQP iterations over 7011 solves,
  against 16 in one, about 2.9 ms per iteration. The sweep count drives it: the tail is
  linear and slow because every discipline is pinned to the others' last values.
- **What implicit differentiation would save: nothing as ALC stands.** There is no
  finite difference to replace. Every SLSQP gets exact `jax.jacfwd` derivatives of its
  own discipline's models (the driver's `epsfcn` is off), and nothing differentiates
  through a nested solve: no level optimises over another's answer. That need arises in
  CO and IPD/EPD (FR-6), not in ALC. What would cut the cost:
  - *Warm starts* (FR-1). Near the answer each discipline is re-solved from the file's
    start every sweep, at 13–19 iterations, where its last answer is within 10⁻⁶. That
    would cut iterations per solve, not the 2336 sweeps.
  - *An accelerated or Newton coordination.* The tail converges linearly (`max|c|` falls
    about 3.2 decades over sweeps 600–2250, a factor of about 0.995 per sweep). A
    Newton-type step on the sweep map would need each discipline's post-optimal
    sensitivities, the derivatives of its answer with respect to the copies and `v`,
    which implicit differentiation of its KKT conditions gives. That is where implicit
    differentiation could pay, but it is a different coordination algorithm from
    Tosserams' ALC. Anderson acceleration of the sweep needs no derivatives at all.
    Neither has been tried, so the saving is unmeasured.

## 6. Open points

- **The equivalence check** (AAO started at ALC's answer) is part of
  `python paper_tests/distributed_architectures/alc.py`. R8 passes it: AAO takes 3 iterations from
  ALC's answer and stays at its own optimum.
- **Not FR-12: a custom driver, now implemented.** An earlier version of this report
  asked for FR-12 to be extended. It does not need to be. FR-12 is a driver that steps
  a fixed schedule (a weight ladder, a DOE), fixed before the run. ALC's driver decides
  its next point from the model's outputs (`c` after a sweep), as every optimiser and
  root finder does. So it is an ordinary driver of a new kind, ALC's own, not part of
  the standard set: `alc.AugmentedLagrangian` and `alc.Coordinator` (section 2).
- **`nu_star` is NaN, by PROCESS's own formula, in every case.** Investigated, not changed.
  - `nu_star = … dlamie … / (e_plasma_beta² · plasma_current)`
    (`models/physics/dimensionless_parameters.py`, ported from `Physics.outplas`).
  - On Helias both `dlamie` and `plasma_current` are exactly 0, so it is 0/0. `dlamie` is
    set only in PROCESS's tokamak physics routine (`Physics.run`), which a stellarator
    never calls; a stellarator has no plasma current.
  - It is NaN identically in the MDA at the file's start and at AAO's optimum, so not an
    ALC effect. It is a reporting output: no constraint and not COE reads it.
  - PROCESS's own `outplas` is called by `Stellarator.run(output=True)` and computes the
    same expression; its `np.sqrt` makes it a NumPy float, so 0/0 gives NaN rather than
    raising. This is by reading the code; PROCESS was not run.
- The ALC XDSM and N2: `python paper_tests/distributed_architectures/dsm.py alc` writes
  `out/stellarator_helias_alc.html`.
