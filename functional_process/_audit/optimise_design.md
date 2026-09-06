# Optimise / driver design

Current state of the solver, driver and code-generation work. **Everything below is what is
true now**; the investigations that established it are in git history.

This file used to carry a dated sequence of numbered sections (§1 through §94) recording
each measurement as it happened -- 3 900 lines by the end. That narrative is gone, on the
same rule the rest of this directory follows: what is landed lives in the code, what was
refuted lives in `tried_and_rejected.md`, what is still open lives in `next_steps.md`, and
the reasoning stays in git (`git log -p -- functional_process/_audit/optimise_design.md`).
**Section numbers cited from docstrings are unchanged; they resolve into git history rather
than into this file.**

## Drivers

Two, both host-side NumPy libraries reached through `jax.pure_callback`
(`core/solver/drivers.py`). `VmconDriver` (`pyvmcon`, cvxpy/CLARABEL) is production;
`SlsqpDriver` (scipy) is a **second opinion** -- kept so that two solvers disagreeing
localises a problem. `scaled_problem` builds the problem once and both consume it, so a
difference between them is the solver, not the setup.

The published matrices are `cottax/reference_warm_matrix.txt` (CPU) and
`..._gpu.txt`; `_audit/performance.md` is the headline. Read them **warm**: a cold row is
~97 % compilation.

**Open**: `SlsqpDriver` and VMCON settle ~1e-04 apart on `stellarator_helias` SAND, stably
across ten +-ulp draws. Not the interpolation -- it survives both the C1 change and the
resolution alternative. Unexplained.

## Cost, and what actually drives it

- A cold row is **~97 % compilation, ~1 % arithmetic**. The `model` column of the cold
  matrix is mis-attributed trace/lower work and must not be divided by an iteration count.
- The largest block is ~2.3 M StableHLO characters = **28 k ops**, of which **41.6 % is
  pure shape plumbing** (`broadcast_in_dim` alone 27.9 %) -- a scalar-valued graph expressed
  in an array language. `jacfwd` roughly doubles it; XLA's fusion pass then *raises* the
  instruction count to 70 k.
- **Machine code is small**: 5.8 MB, ~87-165 bytes per post-optimisation instruction.
  `LoadedExecutable.serialize()`'s 11 MB is the optimised **HLO**, not code.
- **Per-program resident cost is not a well-defined quantity** -- attempts to measure it
  produced two retractions. Use pass-level peaks. Measured one-program-per-process, peak is
  **~157 MB fixed + ~11 KB per post-optimisation instruction**; of a 867 MB peak, ~54 % is
  transient LLVM workspace, ~30 % is held by the live executable, and the ~142 MB residue is
  **almost entirely glibc free-list** (`mallinfo2` says ~14 MB is live). A tuning problem,
  not a hard cost. `MALLOC_ARENA_MAX=1` is the only real lever: peak -32 %, compile +57 %.
- **Reverse mode** costs ~2.9x the compile time and ~3.3x the peak RSS of forward, for 14 %
  more HLO. Its runtime temp buffer is 5.2x forward's -- and 1.7 MB, four orders below
  compile memory. **`jax.checkpoint` is the wrong tool here**: it cuts the runtime peak 40 %
  and costs 86 MB of compile memory, because it expresses its trade as extra graph.

## GPU

Same answers, **median 2.41x slower** over the warm matrix, worst 9x on the largest
configuration -- launch and per-kernel fixed costs over a scalar graph, plus **float64 at
1/16.4 of float32 on this card** (76.6 against 1254.7 GFLOP/s), and PROCESS is float64
throughout. Treat those numbers as specific to a 35 W laptop Quadro.

**Batching is where the GPU wins**: `vmap` over the block evaluation crosses over at
**~256 points** and reaches 3.4x by 4 096, while the CPU curve *worsens* past 64 as the
working set outgrows cache. That is the shape of `process/core/scan.py`, and it is the
reason to want a jax-native or generated solver at all -- not single-solve speed.

## Code generation (`cottax/warp/`)

The graph is an explicit dataflow IR, so the block can be **emitted** rather than lowered.
Warp is the target: scalar-native (the shape plumbing never exists), source-to-source
adjoints, and `wp.launch(dim=n)` is the scan shape.

**Where it stands.** Of **853** live model functions, **757 (88.7 %) transpile**; 686 emit as
one module; **455 validate bit-identical** against their JAX originals (323 single-return,
132 multi), zero disagreements. `helias_5b`'s SAND Drive resolves **85 of 94** nodes under
the arity invariant, `stellarator_helias` 113 of 123.

**The first real kernel**: the maximal prefix-closed sub-DAG of `helias_5b`'s Drive -- 18 of
89 entries, 1 of 11 conditions, four excluded leaves named -- compiles and is
**bit-identical** to JAX evaluating the identical sub-DAG (`0.000e+00`). Faster at every
batch that fits: **5.8x / 6.2x / 6.6x** on CPU at batch 1 / 16 / 256, and 4.5x the best JAX
number at 4 096 on GPU. **22 registers forward, 244 backward, zero spill.**

**Open**: widening past 18 nodes; whether the adjoint spills as it grows; `stellarator_helias`
and a tokamak; and `jnp.interp`/`searchsorted`, array-valued lookup tables, and the
`Composition` nodes the arity invariant refuses.

## Two rules this file paid for

**Refuse rather than guess.** The transpiler raises on anything it does not recognise and
the function goes to a hand-written registry. A generator that silently mistranslates one
physics formula is worse than one that covers less and says so. The same rule caught three
silent mis-resolutions once the resolver was given an **arity invariant** (a resolved leaf's
return arity must equal its node's declared output count).

**A harness that skips what it cannot drive will report success on the subset it reaches.**
Five separate results in this file's history were wrong that way -- an unconditional return
annotation, a validator calling `float()` on a tuple, a "compiled" claim that was module
loading, a path rewrite pointing at the wrong copy, and `pkgutil.walk_packages` silently
never descending into namespace packages, hiding four subpackages and a third of the
codebase. **Every one inflated confidence by hiding a subset rather than by being wrong
about what it measured.** The measurements that survived enumerate from the graph or the
filesystem directly.
