# Performance — the headline numbers

**The one file to look at for "how fast is it and where does the time go".** Everything
here is measured; `optimise_design.md` carries the derivations and the failed attempts.
Regenerate rather than trust a stale copy:

```bash
$PY -m functional_process.cottax.run_warm_matrix    # this table
$PY -m functional_process.cottax.run_cold_matrix --native --compare-process
$PY -m functional_process.cottax.run_cold_matrix --native --compare-process --slsqp
```


> **Updated 2026-09-06.** `reference_warm_matrix.txt` has been re-measured: the table it
> held predated the `icc = 11` removal (§52) and the `vacuum.py` conversion (§59).
> `helias_5b`'s two SLSQP rows now **converge** where they read `stopped`. A GPU twin is
> published as `reference_warm_matrix_gpu.txt` -- **the GPU is a median 2.41x slower**,
> up to 9x on the largest configuration, with identical answers on 23 of 24 rows
> (`_audit/optimise_design.md` §68).

## Read the warm matrix, not the cold one, for anything about speed

A cold row measures assembly, tracing, lowering, compilation and the solve. **Compilation
is ~97 % of it** and the arithmetic is ~1 % (§44), so a cold row compares compilers, not
solvers. Worse, its `model` column is not evaluation time at all: of a row's 128 block
calls, two carry the compile at 9 194 ms and 6 662 ms and the other 126 run at 1.4 ms, so
the mean describes no call that happened, and what lands in `model` is mostly tracing and
lowering that `phase_timing`'s two patched entry points cannot see (§44).

The warm matrix solves each configuration **twice**: once to pay the compiler, once to
measure. The second solve compiles nothing, so what is left is what a driver costs.
**Every warm answer — iterations, status, objective — is identical to its cold
counterpart across all 24 rows**, which is the check that this is the same solve and not
a different one.

## The warm matrix (`reference_warm_matrix.txt`)

Seconds. `XLA` sums every block-program call; `host` is the rest — the optimiser's own
cost (`cvxpy`/CLARABEL for VMCON, the Fortran line search for SLSQP, plus the callback
boundary).

| configuration | arm | driver | it | status | wall | XLA | host | calls | ms/call |
|---|---|---|---|---|---|---|---|---|---|
| stellarator_helias | MDF | VMCON | 41 | converged | 0.571 | 0.123 | 0.449 | 81 | 1.500 |
| stellarator_helias | MDF | SLSQP | 27 | converged | **0.177** | 0.076 | 0.101 | 65 | 1.144 |
| stellarator_helias | SAND | VMCON | 24 | converged | 0.423 | 0.059 | 0.364 | 47 | 1.238 |
| stellarator_helias | SAND | SLSQP | 501 | cap(500) | 3.723 | 2.935 | 0.788 | 4019 | 0.701 |
| helias_5b | MDF | VMCON | 4 | converged | 0.109 | 0.010 | 0.099 | 7 | 1.405 |
| helias_5b | MDF | SLSQP | 2 | stopped | 0.086 | 0.003 | 0.083 | 2 | 1.311 |
| helias_5b | SAND | VMCON | 7 | converged | 0.166 | 0.015 | 0.151 | 13 | 1.081 |
| helias_5b | SAND | SLSQP | 2 | stopped | 0.096 | 0.003 | 0.093 | 2 | 1.270 |
| large_tokamak_nof | MDF | VMCON | 7 | converged | 0.328 | 0.119 | 0.209 | 13 | 8.940 |
| large_tokamak_nof | MDF | SLSQP | 8 | converged | 0.248 | 0.089 | 0.159 | 15 | 3.618 |
| large_tokamak_nof | SAND | VMCON | 10 | converged | 0.359 | 0.089 | 0.270 | 19 | 4.441 |
| large_tokamak_nof | SAND | SLSQP | 13 | converged | 0.245 | 0.082 | 0.163 | 26 | 1.754 |
| large_tokamak_eval | MDF | VMCON | 3 | converged | 0.067 | 0.000 | 0.067 | 0 | — |
| large_tokamak_eval | MDF | SLSQP | 3 | converged | 0.098 | 0.000 | 0.098 | 0 | — |
| low_aspect_ratio_DEMO | MDF | VMCON | 11 | converged | 0.577 | 0.300 | 0.277 | 21 | 14.100 |
| low_aspect_ratio_DEMO | MDF | SLSQP | 12 | converged | 0.338 | 0.186 | 0.152 | 22 | 7.513 |
| low_aspect_ratio_DEMO | SAND | VMCON | **79** | converged | **2.341** | 1.217 | 1.124 | 157 | 7.710 |
| low_aspect_ratio_DEMO | SAND | SLSQP | **17** | converged | **0.349** | 0.189 | 0.160 | 52 | 2.142 |
| spherical_tokamak_eval | MDF | VMCON | 2 | converged | 0.064 | 0.000 | 0.064 | 0 | — |
| spherical_tokamak_eval | MDF | SLSQP | 2 | converged | 0.090 | 0.000 | 0.090 | 0 | — |
| st_regression | MDF | VMCON | 10 | converged | 0.272 | 0.063 | 0.210 | 19 | 3.314 |
| st_regression | MDF | SLSQP | 11 | converged | 0.188 | 0.051 | 0.138 | 21 | 2.336 |
| st_regression | SAND | VMCON | 10 | converged | 0.288 | 0.051 | 0.237 | 19 | 2.729 |
| st_regression | SAND | SLSQP | 10 | converged | 0.160 | 0.033 | 0.128 | 19 | 1.420 |

The two `*_eval` rows report 0 calls because they state a `RootFind`, not an `Optimise` —
there is no `host_cache` block for the instrument to time, and their wall is the MDA.

## What it says

**Warm inverts the cold reading of the two drivers.** Cold, SLSQP totals 409.7 s against
VMCON's 369.8 s and looks worse; that is compile time, and SLSQP compiles two programs
where VMCON compiles one fused one (§40). Warm, **SLSQP is faster on every arm where both
converge** — 0.177 vs 0.571, 0.248 vs 0.328, and 0.349 vs 2.341 on `low_aspect_ratio_DEMO`
SAND, a 6.7× gap that is 17 iterations against 79.

**The bottleneck is host, not evaluation, and it is `pyvmcon`'s `cvxpy`.** On
`stellarator_helias` MDF, VMCON spends 0.449 s on the host against 0.123 s in XLA; SLSQP
spends 0.101 s against 0.076 s. Same evaluations, a quarter of the host cost. `pyvmcon`
builds a fresh `cvxpy` problem every SQP iteration — ~11 ms per iteration here — where
scipy's SLSQP is Fortran and allocates nothing. A parametrised (DPP) problem reused
across iterations is the one large lever, and it is upstream of this port.

**Evaluation is close to its floor.** `ms/call` is 1.1-1.5 ms on the stellarator arms and
1.4-8.9 ms on the tokamaks, against a jax bare-dispatch floor of 0.016-0.024 ms — so the
boundary is free and the cost is the program. A forward-mode tangent runs at ~0.4 of a
primal evaluation, which is `vmap` amortising about as well as it can (§41). The remaining
lever is program size: **53 010 emitted MLIR lines for one tokamak MDF block**, which pays
twice, in compile time and per call.

**Where a cold row's time actually goes**, `stellarator_helias`, 13.6 s: ~4.4 s
trace+lower, ~8.3 s compile, **0.13 s arithmetic** (§44). The `trace`/`lower` columns
under-report and `model` over-reports, because `jacfwd` re-enters primitive binding
outside the patched entry points.

## Known bad rows

Both are open items in `next_steps.md`, not defects in this table.

- **`helias_5b` under SLSQP stops at iteration 1 on both arms**, scipy status 6, *"Singular
  matrix C in LSQ subproblem"* — a rank-deficient constraint Jacobian at that
  configuration's cold start, which VMCON's QP survives. Unchanged across every SLSQP run
  since the flag was built.
- **`stellarator_helias` SAND under SLSQP hits the 500-iteration cap**, 4019 calls, where
  VMCON takes 24. Its line search evaluates ~8 points per iteration there.
