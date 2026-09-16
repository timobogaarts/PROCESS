# Performance — the headline numbers

**The one file to look at for "how fast is it and where does the time go".** Everything
here is measured; `optimise_design.md` carries the derivations and the failed attempts.
Regenerate rather than trust a stale copy:

```bash
$PY -m functional_process.cottax.run_warm_matrix    # this table
$PY -m functional_process.cottax.run_cold_matrix --native --compare-process
$PY -m functional_process.cottax.run_cold_matrix --native --compare-process --slsqp
```


> **Updated 2026-09-16 -- all three references re-measured, on a different machine.**
> `reference_cold_matrix.txt`, `reference_slsqp_matrix.txt` and `reference_warm_matrix.txt`
> are now from an **AMD Ryzen 7 3700X** (`PC-Timo`, WSL2); the 2026-09-06 tables were a
> laptop, and neither file said so. Every header now carries a `MACHINE:` line and a
> `COTTAX:` line (the `~/jaxgraph` commit that answered: `a3e4c56`, after the port was
> re-pointed at cottax's moved API in `e84cad95`). **Every one of the 24 warm rows and
> the 12+12 cold rows now converges** -- the two "known bad rows" below are history --
> and every warm answer equals its cold twin, both drivers. The re-port itself is
> numerically inert: the pre-port pair (this tree at `ce19306e`, cottax `e0f22e6`)
> reproduces every row to every digit on this machine, iterations and ms/call included.
> What differs from the 2026-09-06 tables is (a) the 09-07..09-11 port commits on the
> two stellarators (`helias_5b` 0.764215516 -> 0.764242162, `stellarator_helias`
> 1.21848284 -> 1.21844143, both arms, both drivers) and (b) the machine -- see
> "Two machines" below. The GPU twin `reference_warm_matrix_gpu.txt` is still the
> 2026-09-06 laptop measurement (`_audit/optimise_design.md` §68).

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

R7 3700X, 2026-09-16, `--repeats 3`. Seconds. `XLA` sums every block-program call; `host` is the rest — the optimiser's own
cost (`cvxpy`/CLARABEL for VMCON, the Fortran line search for SLSQP, plus the callback
boundary).

| configuration | arm | driver | it | status | wall | XLA | host | calls | ms/call |
|---|---|---|---|---|---|---|---|---|---|
| stellarator_helias | MDF | VMCON | 40 | converged | 0.516 | 0.149 | 0.367 | 79 | 1.899 |
| stellarator_helias | MDF | SLSQP | 27 | converged | 0.141 | 0.078 | 0.063 | 65 | 0.909 |
| stellarator_helias | SAND | VMCON | 43 | converged | 0.800 | 0.158 | 0.642 | 85 | 1.917 |
| stellarator_helias | SAND | SLSQP | 88 | converged | 0.975 | 0.586 | 0.389 | 590 | 0.819 |
| helias_5b | MDF | VMCON | 4 | converged | 0.082 | 0.009 | 0.073 | 7 | 1.331 |
| helias_5b | MDF | SLSQP | 5 | converged | 0.056 | 0.007 | 0.049 | 8 | 0.914 |
| helias_5b | SAND | VMCON | 7 | converged | 0.151 | 0.020 | 0.132 | 13 | 1.558 |
| helias_5b | SAND | SLSQP | 7 | converged | 0.084 | 0.012 | 0.072 | 13 | 0.792 |
| large_tokamak_nof | MDF | VMCON | 7 | converged | 0.533 | 0.329 | 0.203 | 13 | 15.193 |
| large_tokamak_nof | MDF | SLSQP | 8 | converged | 0.309 | 0.191 | 0.119 | 15 | 4.632 |
| large_tokamak_nof | SAND | VMCON | 10 | converged | 0.322 | 0.100 | 0.222 | 19 | 5.285 |
| large_tokamak_nof | SAND | SLSQP | 13 | converged | 0.285 | 0.131 | 0.154 | 26 | 4.787 |
| large_tokamak_eval | MDF | VMCON | 3 | converged | 0.070 | 0.000 | 0.070 | 0 | — |
| large_tokamak_eval | MDF | SLSQP | 3 | converged | 0.076 | 0.000 | 0.076 | 0 | — |
| low_aspect_ratio_DEMO | MDF | VMCON | 11 | converged | 0.963 | 0.718 | 0.245 | 21 | 23.998 |
| low_aspect_ratio_DEMO | MDF | SLSQP | 12 | converged | 0.555 | 0.444 | 0.111 | 22 | 10.195 |
| low_aspect_ratio_DEMO | SAND | VMCON | 79 | converged | 2.192 | 1.194 | 0.998 | 157 | 7.595 |
| low_aspect_ratio_DEMO | SAND | SLSQP | 17 | converged | 0.349 | 0.203 | 0.146 | 52 | 2.451 |
| spherical_tokamak_eval | MDF | VMCON | 2 | converged | 0.082 | 0.000 | 0.082 | 0 | — |
| spherical_tokamak_eval | MDF | SLSQP | 2 | converged | 0.066 | 0.000 | 0.066 | 0 | — |
| st_regression | MDF | VMCON | 10 | converged | 0.243 | 0.074 | 0.169 | 19 | 4.024 |
| st_regression | MDF | SLSQP | 10 | converged | 0.140 | 0.052 | 0.089 | 19 | 2.050 |
| st_regression | SAND | VMCON | 10 | converged | 0.280 | 0.075 | 0.205 | 19 | 3.983 |
| st_regression | SAND | SLSQP | 10 | converged | 0.192 | 0.051 | 0.142 | 19 | 2.319 |

The two `*_eval` rows report 0 calls because they state a `RootFind`, not an `Optimise` —
there is no `host_cache` block for the instrument to time, and their wall is the MDA.

## What it says

**Warm inverts the cold reading of the two drivers.** Cold, SLSQP totals 403 s against
VMCON's 364 s and looks worse (an earlier, quieter pass the same morning: 384 against
355; laptop: 409.7 against 369.8); that is compile time, and
SLSQP compiles two programs where VMCON compiles one fused one (§40). Warm, **SLSQP is
faster on 9 of the 10 optimised arms** (the two `*_eval` rows drive no optimiser) — 0.141 vs 0.516, 0.309 vs 0.533, and 0.349 vs 2.192 on
`low_aspect_ratio_DEMO` SAND, a 6.3× gap that is 17 iterations against 79. The one
exception is `stellarator_helias` SAND, 0.975 vs 0.800, where SLSQP takes 88 iterations
and 590 calls to VMCON's 43 and 85 — and lands on a *lower* objective (1.2183253 against
1.21844143, feasible to 4e-11) closer to PROCESS's own `x` (`worst dx` 7.4e-02 against
1.09e-01). That row capped at 500 on 2026-09-06; something in the 09-07..09-11 port
commits un-stuck it, and which one is not yet measured.

**The bottleneck is host, not evaluation, and it is `pyvmcon`'s `cvxpy`.** On
`stellarator_helias` MDF, VMCON spends 0.367 s on the host against 0.149 s in XLA; SLSQP
spends 0.063 s against 0.078 s. Same evaluations, a sixth of the host cost. `pyvmcon`
builds a fresh `cvxpy` problem every SQP iteration — ~9 ms per iteration here — where
scipy's SLSQP is Fortran and allocates nothing. A parametrised (DPP) problem reused
across iterations is the one large lever, and it is upstream of this port.

**Evaluation is close to its floor.** `ms/call` is 0.8-1.9 ms on the stellarator arms and
2.0-24 ms on the tokamaks, against a jax bare-dispatch floor of 0.016-0.024 ms — so the
boundary is free and the cost is the program. A forward-mode tangent runs at ~0.4 of a
primal evaluation, which is `vmap` amortising about as well as it can (§41). The remaining
lever is program size: **53 010 emitted MLIR lines for one tokamak MDF block**, which pays
twice, in compile time and per call.

**Where a cold row's time actually goes**, `stellarator_helias`, 13.7 s: ~3 s
trace+lower, ~7.6 s compile, **0.15 s arithmetic** (§44). The `trace`/`lower` columns
under-report and `model` over-reports, because `jacfwd` re-enters primitive binding
outside the patched entry points.

## Two machines: what moved with the hardware and what did not

The 2026-09-06 tables were a laptop; these are an R7 3700X (Zen 2, 8C/16T, WSL2). Same
code on the same machine reproduces every digit, so the differences below are the
hardware, not the port.

| quantity | laptop (09-06) | R7 3700X (09-16) |
|---|---|---|
| cold total, VMCON, 12 arms | 369.8 s | 355-364 s |
| cold total, SLSQP, 12 arms | 409.7 s | 384-403 s |
| `compile`, `large_tokamak_nof` MDF / SAND (VMCON) | 31.6 / 38.5 s | 30.7 / 35.2 s |
| warm `ms/call`, stellarator arms | 0.7-1.5 | 0.8-1.9 |
| warm `ms/call`, tokamak SAND arms | 1.4-7.7 | 2.3-7.6 |
| warm `ms/call`, tokamak MDF, SLSQP (two programs) | 2.8-7.5 | 3.9-10.2 |
| warm `ms/call`, tokamak MDF, **VMCON (one fused program)** | 8.9-14.1 | **15-26** |

**Compilation is a wash** (within 5 % on every total): LLVM is single-threaded and the two cores are
of an age. **Small programs are a wash.** The one thing that is slower here, by 1.5-2x,
is the **fused value-plus-Jacobian VMCON program of the two big tokamak MDF blocks** --
and only that. It is not XLA threading: `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false
intra_op_parallelism_threads=1"` gives 19.2 and 28.7 ms on the same two rows, inside
the band, so the program is already effectively serial. SLSQP's two separate programs on
the *same* block run at 3.9-4.6 ms, which matches the laptop. The hypothesis that fits
is instruction footprint: §"the machine code is 6 MB" against Zen 2's 512 KB L2 and the
laptop's presumably larger one, which would make `VmconDriver(fused=True)` a setting
worth re-measuring per machine rather than a fixed win. Not measured further.

**That row is also the noisiest in the table**: `large_tokamak_nof` MDF VMCON read
15.2, 21.8, 22.2 and 26.4 ms across four quiet runs of `--repeats` 2-5, and **469 ms**
once when another process was running a test suite. A warm number from this machine
wants a quiet box and a repeat; the table's `--repeats 3` reports the last.

## Formerly bad rows

Both were open items in `next_steps.md` (§46-§49) and both converge as of 2026-09-16.

- **`helias_5b` under SLSQP** stopped at iteration 1 on both arms, scipy status 6,
  *"Singular matrix C in LSQ subproblem"*: constraint 11, `rbld == rmajor`, is a
  tautology on the stellarator build path and its zero Jacobian row made the equality
  block rank-deficient (§52). `icc = 11` was dropped from `helias_5b.IN.DAT` on
  2026-09-06 (`8cf4abc3`), after the cold SLSQP reference was measured; both arms now
  converge in 5 and 7 iterations to VMCON's answer.
- **`stellarator_helias` SAND under SLSQP** hit the 500-iteration cap, 4019 calls,
  zig-zagging on `^cond.stellarator.wp_width_r_min` against `c62` (§47, §72). As of the
  09-07..09-11 port commits it converges in 88 iterations, 590 calls, to a feasible point
  with a lower objective than the other three arms (see "What it says"). Which commit
  changed the trajectory is not yet isolated; `a56c6f35` (fusion gating) and `ba84ce1d`
  (recovered declarations) are the candidates.
