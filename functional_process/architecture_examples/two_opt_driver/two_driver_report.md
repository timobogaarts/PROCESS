# Two sequential optimisers over the PROCESS graph: a decoupling experiment

*stellarator_helias, functional_process @ eca063ed, cottax @ a3e4c56, 2026-09-16*

## Motivation

PROCESS solves one VMCON over the whole pipeline. The cottax port makes the dependency
structure explicit, so a natural question is whether the single optimisation can be split
into **two `Optimise` problems run in sequence**, each owning a disjoint subset of the
iteration variables and answering a subset of the constraints, with no variable written by
both. The hoped-for gains are smaller SQP problems, cheaper Jacobians and a decomposition
that reads like the engineering (size the magnet, then find the operating point).

## Methodology

1. **Coupling matrix.** For the file's own problem (`ixc = {2,3,4,6,10,56,59,109}`,
   14 constraints, objective 6 = COE), each condition's ancestor cone was walked on the
   assembled graph (constraints and objective inserted, no `Optimise`) to record which
   iteration variables it reaches.
2. **Legality by construction.** Two `Optimise` nodes were inserted and `Blocking.scc`
   asked for the partition. Two drivers are *sequential* iff they land in different
   SCCs; cottax refuses shared ownership at graph build and refuses two problems in one
   SCC unless `Combine`d (= SAND) or `NestInside`d (= bilevel). Each block's pre-existing
   `FixedPoint` cuts were `Residualise`d and `Combine`d into *that* block's problem, i.e.
   `sand_graph` applied per block rather than once.
3. **Execution.** The schedule was seeded as `run_cold_matrix.solve_sand` does (design from
   the cold state, lifted unknowns from a completed MDA), run with `run_schedule(...,
   whole=False)`, and compared against the reference SAND solved in the same process.
4. **Performance.** Cold and warm wall-clock, SQP iteration counts, and jitted
   per-evaluation cost of each block's condition map and `jacfwd`.
5. **Two-schedule warm start.** Can the sequential answer *start* the full SAND? In one
   graph it cannot be a third problem: a `VarPath` has one owner, and a SAND `Optimise`
   owning `.physics.rmajor` beside `OptMagnet` is refused at graph build. cottax's two
   shapes for "one solve's answer starts another" are (a) two `Schedule`s with the second's
   `^guess.*` `Start` ports seeded from the first's output env — the `mdf.restart`
   pattern — and (b) one graph in which a relabelled copy of the stage subgraph owns
   `^stage.*` places and `Supply` points the SAND problem's `Start` ports at them (the
   `answerable` rule admits driver data produced outside the block). Shape (a) was run:
   the two-driver schedule (`b_min` 5.0), then the reference SAND schedule with all 319 of
   its inputs — design, lifted unknowns, every `^guess.*` — taken from that env.

Scripts (scratch): `coupling_probe.py`, `two_drivers2.py`, `two_drivers_solve.py`,
`two_drivers_btmin.py`, `plasma_at_ref.py`, `bt_scan.py`, `channel.py`, `perf.py`,
`sand_warm.py`.
DSMs: `dsm_two_drivers_executing.html` (the driven graph that ran), `xdsm_two_drivers.html`,
`dsm_split_B_refused.html`.

## Results

### Structure

Every constraint reads `b_t` (ixc 2) and `rmajor` (3). The TF-conductor constraints
c32/34/35/65/82/83 read only `{2, 3, 56, 59}`. The COE objective and the net-electric
equality c16 read **all eight**, including `tdmptf` (56) and `fcutfsu` (59).

Consequences:

- **The obvious split is illegal.** "Plasma/machine `{2,3,4,6,10,109}` vs TF conductor
  `{56,59}`" collapses into one 125-node SCC (COE reads 56/59; c32–83 read 2/3). It *is*
  SAND.
- **A legal split exists**, and the graph produces it as two `Drive` steps in order.
  The DSM of the driven graph that executed (`dsm_two_drivers_executing.html`, embedded
  in `two_driver_report.html`) shows the two optimiser blocks on the diagonal with
  everything from the magnet block to the plasma block strictly below it:

| | `^problem.sand_magnet` (25 nodes) | `^problem.sand_plasma` (85 nodes) |
|---|---|---|
| objective | FoM 1: min `rmajor` (proxy) | FoM 6: COE (the file's own) |
| design unknowns | `b_t`, `rmajor`, `tdmptf`, `fcutfsu` | `te`, `dene`, `hfact`, `f_alpha` |
| lifted unknowns | `wp_width_r_min` | `T_ion`, `proton_rate_density`, `fusden_alpha_total`, `f_ster_div_single` |
| consistency (eq) | — | c2 power balance, c16 net electric |
| lifted residuals (eq) | 1 | 4 |
| engineering limits (ineq) | c32, c34, c35, c65, c82, c83 | c8, c17, c18, c24, c62, c67 |

`delta_eta_step` stays a separate 1-unknown Newton after the plasma block. The
reference SAND is one 124-node block, 14 unknowns × 21 conditions.

### Execution

| run | magnet VMCON | plasma VMCON | `b_t` | `rmajor` | `tdmptf`/`fcutfsu` | COE |
|---|---|---|---|---|---|---|
| as designed | converged, 10 it | **QP infeasible, 0 it** | 4.00 (bound) | 25.5 | 2.0 / 0.30 | — |
| + `b_t ≥ 5.0` in magnet | converged, 11 it | converged, 22 it | 5.00 (active) | 26.3 | 4.1 / 0.33 | 1.272 |
| + `b_t ≥ 5.5` | converged, 12 it | converged, 30 it | 5.50 (active) | 26.7 | 5.7 / 0.34 | 1.327 |
| reference SAND | — | 43 it | 4.72 | 26.6 | 31.8 / 0.72 | **1.218** |

Control: the plasma optimiser alone, at the reference SAND's machine, converges in 1
iteration to 1.2184414328 (reference 1.2184414329). The machinery is sound.

### Performance (CPU, same process)

| | two drivers (`b_min` 5.0) | SAND |
|---|---|---|
| cold solve incl. compile | 6.8 s | 7.8–8.5 s |
| warm solve | 0.58–0.63 s | 0.75–0.99 s |
| SQP iterations | 11 + 22 | 43 |
| block evaluation / `jacfwd` (jitted) | 0.21 + 0.55 ms / 0.16 + 0.68 ms | 0.61 / 0.62 ms |

### Two-schedule warm start (compiled programs, 3 repeats)

| stage | SQP iterations | wall-clock | COE |
|---|---|---|---|
| two-driver schedule (`b_min` 5.0) | 11 + 22 | 0.61–0.63 s | 1.2722 |
| full SAND from cold | 43 | 0.73–0.74 s | 1.218441 |
| full SAND from the two-driver answer | **28** | 0.54–0.78 s | 1.218441 (same `x` to 4 digits) |
| pipeline: two-driver **then** SAND | 33 + 28 | **1.15–1.4 s** | 1.218441 |

## Discussion

**Unexpected result 1 — the as-designed split is infeasible for the second stage.**
Structure guarantees the two problems don't overwrite each other and run in order; it says
nothing about whether stage 1 leaves stage 2 a non-empty feasible set. It did not, via two
channels. (a) Minimising `rmajor` under coil/build constraints only sends `b_t` to its lower
bound (thinner winding pack → radial build closes sooner; c83 active). (b) Fast dump and
low copper fraction satisfy c34/c35 cheaply but set the TF turn current at 129 kA instead
of 98 kA, costing ~16 MW of TF power supplies and ~14 MW of cryoplant: same gross 1438 MW,
net 969 instead of 1000. The plasma stage must then meet `P_net ≥ 1000` with β exactly at
its limit (β ∝ p/B², `b_t` fixed) — the linearised set {c16 = 0, c24 ≤ 0, bounds} is
empty and VMCON's first QP fails (`QSPSolverException`). In SAND the 31 MW are bought with
`tdmptf`/`fcutfsu`, which the plasma stage no longer owns. Even at the reference's own
`b_t` and `rmajor`, the magnet driver's conductor choice alone leaves the plasma stage
infeasible (c16 = +0.031, c24 = 0.000).

**Unexpected result 2 — a `b_t` lower bound "fixes" it, but by hand and at a price.**
Adding `b_t ≥ b_min` as a condition node in the magnet problem restores feasibility by
giving β headroom; the magnet stage still picks the expensive conductor and the plasma stage
pays for it in fusion power. COE is 4.4 % worse at 5.0 T and 8.9 % at 5.5 T than the joint
optimum, and `b_min` is a knob whose value matters. Feasibility was recovered by tuning the
interface; optimality is what the decoupling gives up.

**Performance is a modest win for the wrong reason.** ~25–30 % faster warm, ~15 % cold,
but not from the graph: the two blocks together cost slightly *more* per evaluation than the
SAND block (0.76 vs 0.61 ms values; 0.84 vs 0.62 ms Jacobian), since the plasma block
re-evaluates what the magnet block also touches. The saving is fewer and smaller SQP
iterations (33 vs 43; 5×9 and 8×13 QPs vs 14×21), i.e. VMCON's own per-iteration cost. It
is also an unequal comparison — an easier problem, a worse answer, and one failed 7 s solve
before the knob was found that no timing column records. For a `vmap`-over-scan workload the
per-evaluation numbers predict no advantage.

**Unexpected result 3 — the warm start works but the pipeline is net slower.** The
sequential answer is a good start for the joint problem: SAND converges in 28 iterations
instead of 43, to the same design (`b_t` 4.716, `rmajor` 26.64, `tdmptf` 31.81,
`fcutfsu` 0.718), and every one of SAND's 319 schedule inputs — including the lifted
`^hat.*` unknowns — was available in the two-driver env, so no re-seeding logic was
needed beyond `guess_sources`. But the 15 iterations saved are worth ~0.2 s, and producing
the start cost ~0.6 s, so "two drivers then SAND" is 1.2–1.4 s against 0.74 s for SAND
alone. The decomposition is a worse *start-finder* than SAND's own first iterations,
because most of its 33 iterations go into refining a stage-1 answer that SAND then moves
substantially (`tdmptf` 4.1 → 31.8, `fcutfsu` 0.33 → 0.72). The mechanism is the useful
finding: two schedules chain through an env with nothing but `Start` ports, and the
in-graph `Supply` form would make the same chain a single `Schedule` at the price of
duplicating the stage subgraph (roughly doubling compile).

**Two artefacts worth recording.** The graph built directly from
`cut_graph(graph_for(...))` still carries `.vacuum.duct_diameter_root_find`, a disconnected
island (all-zero context) that `mda_harness.EXCLUDED_NODE_NAMES` removes; building off the
harness's `driven` graph is required. And seeding must re-seed every block's lifted unknowns
from an MDA *at that block's incoming context*; seeding from the cold MDA gave a spurious
"violated with zero gradient" reading that was an artefact, not a structural fact.

**What would make the decomposition sound.** As a *solver*, either carry the plasma's need into stage 1
(a recirculating-power proxy on `tdmptf`/`fcutfsu`, or maximise `b_t` at fixed build rather
than minimise `rmajor`), or keep the split but nest it (`NestInside`, bilevel) so the outer
driver sees the inner's answer — at which point the sequential form's cost saving is gone. As a *start-finder* for SAND
it would have to be much cheaper than it is — a stage that stops at a loose tolerance, or a
low-fi stage-1 model, which is exactly what `Supply` was designed for.
The value of the exercise is that all of this was decided by the graph and measured in
minutes: the legal partition was derived, the illegal one refused, and the failure
localised to two named variables and one named constraint.
