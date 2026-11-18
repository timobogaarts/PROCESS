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
sweep over the disciplines and the equivalence argument are the paper's. How the
coordination is run is not.

**The paper baseline** (from 2026-11-16; section 7) departs in three choices. Its driver
is `anderson.AndersonCoordinator` at scipy's defaults, with every weight 0.5.

1. **No inner loop, and Anderson mixing of the sweeps.**
   - *What happens.* The multipliers are updated after every sweep. This is the paper's
     alternating-direction variant, which it notes is outside its convergence proof.
   - *What is added.* The next point is not the sweep's own answer: Anderson mixing of
     the last five sweeps picks it.
   - *What stays.* The fixed point, and with it the equivalence with AAO.
   - *Effect.* 262 sweeps, against 948 for the plain alternating direction at the same
     weight (T13) and 2336 for the earlier configuration below.
2. **β = 1: the weights are never raised**, as in the earlier configuration (item 3 below).
3. **The stopping test is the paper's in kind, on the copies.**
   - *The paper* (Eqs. 26–27) stops once `c` is small and no longer changing.
   - *Here* it stops once `max|c|` ≤ 10⁻⁶ and no copy moved by more than 10⁻⁶ (relative)
     over the sweep: the copies' movement stands in for `c`'s.
   - *Why it suffices.* At this ε the paper's kind of test did not stop early. R3 and R4
     did, at 10⁻⁴ and with the inexact loop.

Tuning within the paper's algorithm:
- SLSQP as the subproblem solver;
- each subproblem's objective scaled by its penalty curvature;
- w₀ = 0.5, within the range Eq. 29 would give here (0.2–0.5).

**The earlier configuration** (`alc.Coordinator`; R7–R10 and the 2026-11-11 benchmark;
T1 in section 6) departed in three other choices. Each was needed to reach AAO's
optimum with plain sweeps (section 3):

1. **The inner loop ends when the copies settle, with a fixed tolerance.** The paper
   either runs block-coordinate descent until the relaxed objective `F` changes by less
   than ε/100 (Eq. 28), or uses an inexact loop whose tolerances fall to zero
   (Section 5.5). Here the multiplier update is taken once no copy moves by more than
   10⁻³ (relative) over a sweep, or after 25 sweeps. Near the answer every sweep
   qualifies, so the tail runs as alternating direction (2025 updates in 2336 sweeps).
   The paper says that variant is outside its convergence proof and "terminates
   prematurely at non-optimal designs" on some problems. At β = 1, pure alternating
   direction does as well as this inner loop (section 6, T7).
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
   definite, the paper's own condition (Section 5.4). β = 1.2 is slower too (section 6,
   T5), and so are fixed weights of 2, 3 and 5. Softer weights, down to 0.4, are faster.

The earlier configuration's other changes were tuning within the paper's algorithm: SLSQP
as the subproblem solver, each subproblem's objective scaled by its penalty curvature, no
weight raised below tolerance (moot at β = 1), and w₀ = 1 instead of Eq. 29 (which would
give about 0.2–0.5 here). Section 3 has the evidence for every change, and section 3's table classifies them
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
| R11 | **the paper baseline** (Anderson at scipy's defaults, w₀ 0.5; section 7), `alc.py` with the AAO checks | 262 | 262 | 5.1·10⁻⁷ | 1.2184412 | −2.3·10⁻⁷ | as T20; AAO from ALC's answer: 2 iterations to 1.2184414, so the same minimum |
| R12 | R11 in `benchmark.py`, 2026-11-16 | 262 | 262 | — | 1.2184412 | −2.3·10⁻⁷ | identical to R11 in every count; 22.8 ± 0.9 s warm over 5 runs (section 5) |

The tuning runs (T1–T14, 2026-11-16) are in section 6.

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

**The paper baseline (R12).** `benchmark.py`, 2026-11-16, one CPU (laptop),
single-threaded BLAS.
- *Every case.* SLSQP at `ftol` 1e-8 and `max_iter` 500, scaled, from the file's
  design; every recipe on the AAO base.
- *ALC's coordination.* `anderson.py` at scipy's defaults, every weight 0.5.
- *Timing.* Warm: mean ± standard deviation of 5 runs of the compiled graph. Every count
  was identical in all runs.
- *cottax.* A `git archive` snapshot of the old `distributed_architectures` branch at
  `42fd44e`.
- Snellius timings to follow.

| Case | COE | vs AAO | Max violation | Overarching loop | SLSQP iterations (solves) | Cold (s) | Warm (s) |
|---|---|---|---|---|---|---|---|
| AAO (SAND) | 1.2184414 | — | 2.2·10⁻¹³ | — | Opt 16 (1) | 9.7 | 0.07 ± 0.00 |
| Sequential | 1.2721691 | +4.4 % | 4.2·10⁻⁹ | — | magnet 11 (1); plasma 122 (1) | 7.9 | 0.54 ± 0.02 |
| ALC | 1.2184412 | −2.3·10⁻⁷ | 9.1·10⁻¹⁰ | 262 sweeps, 262 multiplier updates | build 2 919 (263); operation 5 221 (263); economics 3 443 (263) | 35.8 | 22.8 ± 0.9 |

Per solve, ALC's disciplines take 11.1 (build), 19.9 (operation) and 13.1 (economics)
iterations. No solve failed in any case.

- **Optima.**
  - *ALC against AAO.* They agree to 2.3·10⁻⁷ in COE and to 4–5 digits in the design.
    The largest difference is `tdmptf`, 31.812 against 31.811. AAO started at ALC's
    answer takes 2 iterations and stays at its own optimum (R11).
  - *The sequential split* is a different, worse design: +4.4 % COE, with `b_t` held at
    5 T and `tdmptf` at 4.1 s.
- **Cost.**
  - *Against AAO.* About 330× AAO's warm time: 11 583 SLSQP iterations over 789 solves,
    against 16 in one, about 2.0 ms per iteration.
  - *Against the 2026-11-11 benchmark* (below): 9× fewer sweeps, 7.5× fewer
    iterations and 11× less time.
  - *Where it goes.* The sweeps still drive it: 262 sweeps, each re-solving every
    discipline from the file's start (FR-1).

**The 2026-11-11 benchmark (R9),** with `alc.Coordinator` (the earlier configuration:
inexact inner loop, w₀ 1), under the same settings otherwise:

| Case | COE | vs AAO | Max violation | Overarching loop | SLSQP iterations (solves) | Cold (s) | Warm (s) |
|---|---|---|---|---|---|---|---|
| AAO (SAND) | 1.2184414 | — | 2.2·10⁻¹³ | — | Opt 16 (1) | 10.8 | 0.09 ± 0.01 |
| Sequential | 1.2721691 | +4.4 % | 4.2·10⁻⁹ | — | magnet 11 (1); plasma 122 (1) | 10.1 | 0.69 ± 0.06 |
| ALC | 1.2184418 | +2.7·10⁻⁷ | 1.3·10⁻¹⁰ | 2336 sweeps, 2025 multiplier updates | build 31 345 (2337); operation 44 168 (2337); economics 11 210 (2337) | 296 | 251 ± 10 |

- *Per solve* there, the disciplines took 13.4 (build), 18.9 (operation) and 4.8
  (economics) iterations.
- *R10*, after the multiplier rule moved into ALC's own driver, reproduced every count
  and digit. Its times, cold / warm, on the same laptop under more load: 13.1 s / 0.12 s
  (AAO), 11.5 s / 0.80 s (sequential), 249 s / 253 ± 24 s (ALC).
- *Optima.* ALC and AAO agreed to 2.7·10⁻⁷ in COE and to 3–5 digits in the design
  (`tdmptf` 31.82 against 31.81).
- *Cost.* About 2800× AAO's warm time: 86 723 SLSQP iterations over 7011 solves, about
  2.9 ms per iteration. The slow linear tail drove it.

The rest of this section was written for R9; the change since R12 is marked.
- **What implicit differentiation would save: nothing as ALC stands.** There is no
  finite difference to replace. Every SLSQP gets exact `jax.jacfwd` derivatives of its
  own discipline's models (the driver's `epsfcn` is off), and nothing differentiates
  through a nested solve: no level optimises over another's answer. That need arises in
  CO and IPD/EPD (FR-6), not in ALC. What would cut the cost:
  - *Warm starts* (FR-1). Near the answer each discipline is re-solved from the file's
    start every sweep, at 11–20 iterations, where its last answer is within 10⁻⁶. That
    would cut iterations per solve, not the sweeps. A graph-level workaround exists
    (section 7). It was not tried, by decision (2026-11-16).
  - *An accelerated or Newton coordination.* Under R9 the tail converged linearly
    (`max|c|` fell about 3.2 decades over sweeps 600–2250, a factor of about 0.995 per
    sweep).
    - *Newton.* A Newton-type step on the sweep map would need each discipline's
      post-optimal sensitivities: the derivatives of its answer with respect to the
      copies and `v`, which implicit differentiation of its KKT conditions gives. That is
      where implicit differentiation could pay, but it is a different coordination
      algorithm from Tosserams' ALC. Not tried.
    - *Anderson* needs no derivatives at all. **Since R12 it is the baseline**: 262
      sweeps against R9's 2336.

## 6. Tuning the coordination (2026-11-16)

The question: does a smaller β or a larger initial weight converge faster? A one-off study:
one setting after another in one process, each run's trajectory logged per sweep. Its
script was not kept.

**Setup.**
- **cottax.** The `~/projects/jaxgraph` checkout no longer exists. Its replacement,
  `~/projects/cottax`, is on another branch with a different history. These runs use a
  snapshot of the old `distributed_architectures` branch at `42fd44e`, which differs from
  the benchmark's `0ffbd84` only in visualisation code. The defaults (T1) reproduce R9
  exactly: 2336 sweeps, 2025 updates, the same SLSQP iterations, the same COE.
- **Everything else** is as in the benchmark: SLSQP at its settings, from the file's start.
  The stopping test is the same, except where `tol` is given.
- **What is compared.** Settings are compared on the sweep from which COE stays within a
  given distance of AAO's, not on when they stop. The stopping test was set at w = 1. Its
  multiplier-step test, `2w²c ≤ 10⁻³`, is a weaker test of `c` at smaller weights. At
  w₀ = 0.5 with the default `tol` 10⁻⁴ (T8), it stopped at 788 sweeps with COE 1.8·10⁻⁵
  off and `max|c|` 4·10⁻⁵. With `tol` 10⁻⁶, every run stops at `max|c|` ≤ 10⁻⁶, below the
  defaults' own stop (6.5·10⁻⁷ at sweep 2336).
- **Cost.** SLSQP iterations are deterministic. The times are not: the same work (T1, T7)
  ran at 0.074 and 0.093 s per sweep in different batches on the laptop.

| Run | Setting (else the defaults: w₀ 1, β 1, inner loop 10⁻³ / 25, `tol` 10⁻⁴) | Stop | COE within 10⁻⁴ / 10⁻⁵ / 10⁻⁶ of AAO from sweep | `max\|c\|` ≤ 10⁻⁶ from | COE vs AAO at stop | Design vs AAO | SLSQP iterations | Loop (s) |
|---|---|---|---|---|---|---|---|---|
| T1 | the defaults (= R9) | 2336 | 1578 / 1852 / 2142 | 2271 | +2.7·10⁻⁷ | 3.5·10⁻⁴ | 86 723 | 174 |
| T2 | w₀ 2 | 3000 (cap) | 2598 / – / – | – | −2.0·10⁻⁵ | 0.28 | 103 389 | 238 |
| T3 | w₀ 3 | 3000 (cap) | – / – / – | 2993 | +1.0·10⁻³ | 0.59 | 100 804 | 243 |
| T4 | w₀ 5 | 3000 (cap) | – / – / – | 2992 | +4.1·10⁻³ | 0.98 | 101 919 | 253 |
| T5 | β 1.2 (weights 6.2 by sweep 300) | 3000 (cap) | 2183 / 2561 / 2575 | 2581 | +1.8·10⁻¹⁰ | 2.1·10⁻⁴ | 122 390 | 280 |
| T6 | inner loop capped at 5 sweeps | 2324 | 1571 / 1845 / 2134 | 2263 | +3.3·10⁻⁷ | 3.4·10⁻⁴ | 85 628 | 207 |
| T7 | alternating direction (update after every sweep) | 2318 | 1569 / 1843 / 2131 | 2259 | +3.2·10⁻⁷ | 3.3·10⁻⁴ | 84 894 | 216 |
| T8 | w₀ 0.5 | 788 | 630 / – / – | – | −1.8·10⁻⁵ | 5.0·10⁻⁵ | 38 445 | 75 |
| T9 | w₀ 0.7 | 1248 | 807 / 1031 / – | – | −2.5·10⁻⁶ | 1.8·10⁻⁴ | 48 635 | 108 |
| T10 | w₀ 0.7, `tol` 10⁻⁶ | 1366 | 807 / 1031 / 1353 | 1359 | −5.2·10⁻⁷ | 8.1·10⁻⁵ | 53 211 | 114 |
| T11 | w₀ 0.5, `tol` 10⁻⁶ | 1060 | 630 / 828 / 1024 | 1060 | −2.8·10⁻⁷ | 6.8·10⁻⁶ | 50 957 | 108 |
| T12 | w₀ 0.4, `tol` 10⁻⁶ | 918 | 609 / 769 / 906 | 911 | +5.1·10⁻⁷ | 1.9·10⁻⁶ | 44 882 | 111 |
| **T13** | **w₀ 0.5, alternating direction, `tol` 10⁻⁶** | **948** | **522 / 743 / 941** | **948** | **+2.8·10⁻⁷** | **9.1·10⁻⁶** | **42 865** | **92** |
| T14 | w₀ 0.3, `tol` 10⁻⁶ | failed in sweep 1 | | | | | | |

"Design vs AAO" is the largest relative difference over the eight iteration variables.

**Findings.**

1. **Stiffer penalties are slower: a larger w₀ (T2–T4) or β > 1 (T5).**
   - *Copies first.* At w₀ = 5 every copy was consistent to 3·10⁻⁷ by sweep 200, with COE
     still 3.5 % high.
   - *Then a creep.* From there COE fell only to +0.4 % by sweep 3000. Each discipline is
     pinned to the others' last answers, so the design can only creep along the
     consistent set. This is R4 and R6 again.
   - *β = 1.2.* The weights rose to 6.2 in the transient. It reached the optimum 20 %
     later than the defaults, at 3.4× the economics iterations.
   - On this problem, raising the weights is the wrong direction.
2. **Softer penalties are faster, down to w₀ = 0.4.**
   - *The tail.* COE goes from 10⁻³ to 10⁻⁶ in 614 sweeps (T13), against 1648 at the
     defaults: about 0.989 per sweep against 0.996.
   - *The price.* More SLSQP iterations per sweep, 45 against 37, mostly in economics,
     which moves further each sweep. Iterations in all fall by 2×, sweeps by 2.5×.
   - *The lower edge.* At w₀ = 0.3 the first sweep takes a subproblem's copies where its
     model is non-finite, and the driver's diagnostic SVD fails (T14). That is change 1's
     failure. w₀ = 0.5 keeps a margin.
3. **At β = 1 the inner loop does not matter.**
   - Alternating direction (T7), a 5-sweep cap (T6) and the inexact loop (T1) take 2318,
     2324 and 2336 sweeps.
   - The inexact loop (change 3) was needed because, under β = 2.2, the weights grew at
     every update (R1). At β = 1 nothing grows.
   - Departure 1 can therefore be replaced by the paper's own alternating-direction
     variant. That variant is still outside the paper's convergence proof.
4. **Softer weights need `tol` tightened**, because the multiplier-step test is
   `2w²c`. At w = 0.5 it tests `c` 4× more loosely than at w = 1 (T8 against T11).
5. **The paper's initial-weight rule points the right way.** Change 6 estimated that
   Eq. 29 would give w ≈ 0.2–0.5 here. The fastest stable start found, 0.4–0.5, is at
   the top of that range.

**Recommendation: T13** (w₀ = 0.5, alternating direction, `tol` 10⁻⁶).
- *Gain.* 948 sweeps against 2336, 42 865 SLSQP iterations against 86 723, and about
  half the loop time. It reaches the same optimum: COE +2.8·10⁻⁷, design within 10⁻⁵ of
  AAO's, closer than the defaults' 3.5·10⁻⁴.
- *Effect on the departures.*
  - Departure 1 becomes the paper's alternating-direction variant.
  - Departure 2 stays, with `tol` 10⁻⁶.
  - Departure 3 (β = 1) stays, now with β = 1.2 measured as slower too.
  - w₀ moves into Eq. 29's range.
- **Not yet adopted.** `alc.py`'s defaults, the benchmark (section 5) and the README are
  still T1/R9. Adopting T13 means changing three constants (`INITIAL_WEIGHT`,
  `INNER_TOLERANCE`, `TOLERANCE`), rerunning the benchmark and the AAO-from-ALC check, and
  updating the numbers.

**What tuning cannot fix.** The tail is still linear, at about 0.989 per sweep. A further
large gain needs a different coordination step, not other constants: Anderson or Newton
acceleration of the sweep, or warm starts per discipline (section 5).

## 7. Sweep order and Anderson acceleration (2026-11-16)

Two further cases, compared with T13 (w₀ 0.5, alternating direction, `tol` 10⁻⁶: 948
sweeps). Same setup and measures as section 6.

**Why so many sweeps (a hypothesis, untested).** The partition is feedforward, but the
optimisation is coupled backward: COE's preferences must reach build and operation, and
ALC carries them only through the copies and the multipliers.
- *The toy model.* Build and operation have no objective of their own, so wherever no
  active constraint pins them their problems are flat. Along such a direction build moves
  its output onto economics' copy. Economics then sets that copy as a weighted average:
  COE's preferred value with weight h, build's output with weight 2w².
- *The rate it predicts.* The design moves by `h/(h + 2w²)` of the remaining distance
  per sweep, so the error shrinks by `2w²/(h + 2w²)`, and the inconsistency stays as small
  as the step. Here h is COE's curvature along the direction.
- *What it explains.* Stiffer is slower; softer is faster; there are consistent but
  unconverged stops; the last variables to converge (`f_alpha`, `tdmptf`) are ones COE
  hardly feels.
- *Where it falls short.* It predicts a 4× faster tail from w = 1 to 0.5. The runs show
  1.4–2.7×, so it is at most part of the story.
- *A test.* Log the design per sweep and compare the tail's direction and rate with COE's
  curvature at AAO's optimum.

**Sweep order (T15–T19).**
- *What was run.* Gauss–Seidel in each of the five other orders of build, operation and
  economics, at T13's settings.
- *How another order is built.* Wherever a copy's holder solves before the copy's source,
  the holder reads the source's value from the last sweep. That is a lag cut, whose
  lagged value the coordination moves each sweep. The five constant crossings are lagged
  the same way.

| Run | Order | Sweeps | COE within 10⁻² / 10⁻⁴ / 10⁻⁶ from sweep | SLSQP iterations | COE vs AAO | Design vs AAO |
|---|---|---|---|---|---|---|
| T13 | build, operation, economics | 948 | 130 / 522 / 941 | 42 865 | +2.8·10⁻⁷ | 9.1·10⁻⁶ |
| T15 | economics, operation, build | 947 | 130 / 522 / 941 | 42 785 | +3.3·10⁻⁷ | 9.3·10⁻⁶ |
| T16 | economics, build, operation | 948 | 130 / 522 / 941 | 42 848 | +2.7·10⁻⁷ | 9.0·10⁻⁶ |
| T17 | operation, economics, build | 947 | 130 / 522 / 941 | 42 778 | +3.4·10⁻⁷ | 9.4·10⁻⁶ |
| T18 | operation, build, economics | 947 | 130 / 522 / 941 | 42 855 | +3.4·10⁻⁷ | 9.3·10⁻⁶ |
| T19 | build, economics, operation | 948 | 130 / 522 / 941 | 42 856 | +2.7·10⁻⁷ | 9.0·10⁻⁶ |

The order makes no difference, to within one sweep. That fits the hypothesis: the slow
mode is a round trip, whatever the order. The machinery that built other orders has been
removed, and only the data's order is built. It consisted of:
- an `order` argument;
- the lag cuts;
- a `Lagged` mint (FR-5), because the divertor solid-angle fraction is already cut with
  `cx.Copy()` by AAO's closure inside operation, and a second cut needs a second name.

**Anderson acceleration (T20–T34).** `anderson.py`'s `AndersonCoordinator` hands
`F(z) = G(z) − z` to `scipy.optimize.anderson`.
- *The map.* `G` is one sweep plus the multiplier update (Eq. 24).
- *The coordinates.* `z = (copy / |copy at start|, v / 2w²)`, so `F` is each copy's
  relative movement and `c` itself. The test is `max|F|` ≤ `f_tol` = 10⁻⁶, which matches
  T13's tests on `c` and on the copies; COE's movement is not tested.
- *The wrapping.* A host callback, as `SlsqpDriver` wraps scipy's SLSQP. The sweep reading
  is split into arrays and the rest, and one jitted walk of the interior is `F`.
- *The safeguard.* A sweep that fails or returns a non-finite value is rejected, and
  Anderson restarts from the last good point. It never fired.
- *Defaults.* scipy's: history `M` 5, regularisation `w0` 0.01, `alpha` 1, no line search.

| Run | Setting (else the defaults, w₀ 0.5) | Sweeps | COE within 10⁻² / 10⁻⁴ / 10⁻⁶ from sweep | SLSQP iterations | Loop (s) | COE vs AAO | Design vs AAO |
|---|---|---|---|---|---|---|---|
| T13 | no Anderson (reference) | 948 | 130 / 522 / 941 | 42 865 | 92 | +2.8·10⁻⁷ | 9.1·10⁻⁶ |
| T20 | the defaults | 262 | 40 / 158 / 240 | 11 583 | 24 | −2.3·10⁻⁷ | 5.2·10⁻⁵ |
| T21 | history 10 | 1202 | 1144 / 1178 / 1193 | 57 634 | 171 | +7.0·10⁻⁸ | 2.5·10⁻⁵ |
| T22 | 200 plain sweeps first | 397 | 130 / 260 / 359 | 17 577 | 52 | −3.0·10⁻⁷ | 4.9·10⁻⁵ |
| T23 | w₀ 1 | 994 | 351 / 842 / 906 | 36 617 | 122 | −1.1·10⁻⁷ | 1.5·10⁻⁴ |
| T24 | Armijo line search | 438 | 70 / 332 / 415 | 19 385 | 63 | −4.2·10⁻⁷ | 1.4·10⁻⁵ |
| T25 | history 3 | 438 | 54 / 229 / 382 | 19 662 | 60 | +2.4·10⁻⁷ | 4.3·10⁻⁵ |
| T26 | history 2 | 648 | 101 / 356 / 616 | 29 042 | 70 | −3.3·10⁻⁷ | 1.8·10⁻⁵ |
| T27 | history 4 | 406 | 39 / 245 / 352 | 18 289 | 47 | +2.1·10⁻⁷ | 4.4·10⁻⁵ |
| **T28** | **history 7** | **199** | **110 / 146 / 191** | **8 936** | **22** | **+5.5·10⁻⁸** | **1.5·10⁻⁵** |
| T29 | regularisation 10⁻³ | 416 | 45 / 299 / 408 | 18 979 | 51 | −3.9·10⁻⁷ | 3.5·10⁻⁵ |
| T30 | regularisation 0.1 | 378 | 46 / 247 / 346 | 17 025 | 48 | +2.4·10⁻⁷ | 4.6·10⁻⁵ |
| T31 | `alpha` 0.5 | 316 | 45 / 195 / 290 | 14 180 | 42 | −3.8·10⁻⁷ | 1.6·10⁻⁵ |
| T32 | `alpha` 1.5 | 377 | 31 / 215 / 324 | 16 993 | 57 | +2.7·10⁻⁷ | 4.7·10⁻⁵ |
| T33 | w₀ 0.4 | 238 | 25 / 122 / 204 | 10 610 | 35 | +3.0·10⁻⁷ | 2.4·10⁻⁵ |
| T34 | w₀ 0.7 | 525 | 38 / 250 / 453 | 19 046 | 67 | +2.5·10⁻⁷ | 6.3·10⁻⁵ |

**Findings.**
1. **2–5× fewer sweeps than T13** around scipy's defaults, at the same minimum.
   - *The range.* 199–438 sweeps over histories 3–7, regularisation 10⁻³–0.1, `alpha`
     0.5–1.5 and w₀ 0.4–0.5.
   - *The best.* History 7 (T28): 199 sweeps, 8 936 SLSQP iterations and 22 s of loop,
     against T13's 948, 42 865 and 92 s, and the original defaults' 2336, 86 723 and
     174 s.
   - *Its optimum.* COE within 5.5·10⁻⁸ of AAO's and the design within 1.5·10⁻⁵, the
     closest of any run.
2. **Only the number of sweeps falls.** A sweep costs the same, about 44 SLSQP iterations.
   Both the transient and the tail speed up: from COE 10⁻³ to 10⁻⁶ takes about 150 sweeps,
   against T13's 614.
3. **The dependence on the settings is erratic.** Histories 2, 3, 4, 5, 7 and 10 took 648,
   438, 406, 262, 199 and 1202 sweeps. A setting should be judged by its neighbours, not by
   the single fastest run.
4. **What hurts.**
   - *A long history (T21).* History 10 thrashed through the transient for 1100 sweeps:
     old pairs from far-off points spoil the least-squares fit.
   - *A stiffer penalty (T23).* w₀ = 1 took 994 sweeps. A hypothesis: its smaller steps
     are closer to the noise of cold-started SLSQPs, and Anderson differences them.
   - *A line search (T24).* It spends sweeps on trial points.
   - *A plain warm-up (T22).* It only adds its sweeps.
5. **Not Tosserams' algorithm.** The fixed point, and so the equivalence with AAO, is
   unchanged, but the coordination's step is no longer the method of multipliers alone.
   With Anderson the baseline departs from the paper as "Departures" now states.

**Adopted (2026-11-16): T20 is the paper baseline.**
- *The setting.* Anderson at scipy's defaults, w₀ 0.5. Chosen over T28, the fastest,
  because it sits in the middle of the good region, not at a lucky edge of it (finding 3).
- *Where it runs.*
  - `benchmark.py` builds it by default (`build_alc`), and so does `alc.py`
    (`--driver anderson`, the default). `--driver sweeps` gives `alc.Coordinator`.
- *Checks.* AAO from its answer: 2 iterations to its own optimum (R11). The benchmark is
  R12 (section 5).
- *Not done, by decision.* Each discipline is still solved from the file's start in
  every sweep (FR-1). Near the answer that costs 11–20 iterations a solve, where the last
  answer is within 10⁻⁶. A graph-level workaround is possible (FR-12, "Warm starts",
  (b)): the coordination owns the disciplines' start places and moves each to its last
  answer.

## 8. Open points

- **The equivalence check** (AAO started at ALC's answer) is part of
  `python paper_tests/distributed_architectures/alc.py`. The baseline passes it (R11):
  AAO takes 2 iterations from ALC's answer and stays at its own optimum. R8 passed it
  for the earlier configuration, in 3 iterations.
- **Not FR-12: a custom driver, now implemented.** An earlier version of this report
  asked for FR-12 to be extended. It does not need to be. FR-12 is a driver that steps
  a fixed schedule (a weight ladder, a DOE), fixed before the run. ALC's driver decides
  its next point from the model's outputs (`c` after a sweep), as every optimiser and
  root finder does. So it is an ordinary driver of a new kind, ALC's own, not part of
  the standard set: `alc.AugmentedLagrangian`, with `alc.Coordinator` and, as the
  baseline, `anderson.AndersonCoordinator` (section 7).
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
