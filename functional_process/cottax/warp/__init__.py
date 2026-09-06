"""Generate a Warp megakernel from a cottax graph.

**The generator, not the generated.** This package reads the assembled graph and emits
Warp source; the emitted `@wp.func` bodies and kernels are **not committed** -- they are
derived, they would be a five-figure diff on every regeneration, and the generator plus its
self-validation is the thing worth reviewing. Regenerate into `functional_process/warp/`
(gitignored) rather than reading it out of the repository.

Why it lives under `cottax/` rather than beside the models: it **imports cottax** to read
the graph, and `tests/test_cottax_boundary.py` asserts that only things under `cottax/` do.
The generated output imports nothing but `warp`, so it may live outside.

The pieces, in the order they run:

- `resolve.py` -- **the resolver**: each graph node to the leaf function it delegates to,
  with its arguments in signature order and its frozen switch values as literals. Guarded
  by an **arity invariant**: a resolved leaf's return arity must equal the node's declared
  output count, or the node is refused -- unless the wrapper's own code provably selects
  one literal element of a wider return (`_subscript_select_index`), in which case that
  derived index is carried as `output_index`. That check caught three silent
  mis-resolutions (`_audit/optimise_design.md` §94). It also handles a wrapper that
  computes locals before the call that produces the node's outputs (a `PreludeCall`), and
  binds frozen sequence-valued arguments.
- `scalarise.py` -- for a node whose leaf is passed BY VALUE into an array-assembling
  helper, expands its whole `__call__` -- helper, arm, `jax.vmap` and all -- into one flat
  straight-line function of exactly the node's declared parameters. Used by the resolver
  when no existing function's parameter list can be laid against the node's `VarPath`s.
- `leaves.py` / `combined.py` -- a config's SAND Drive to a topologically ordered list of
  `Leaf` and `StructuralOp` entries, via the resolver.
- `transpile.py` -- one `models/**` function to one `@wp.func`. `jnp.X` to `wp.X`, every
  numeric literal to `wp.float64(n)` (Warp is strictly typed and does not promote), an
  annotated signature, and a return annotation **only** when the body returns a single
  value. **It refuses rather than guesses**: an unrecognised construct raises `Unsupported`
  and the function goes to the hand-written registry. A transpiler that silently
  mistranslates one physics formula is worse than one that covers less and says so.
- `leaf_funcs.py` -- **the leaf-function builder**: drives `transpile.py` over every
  distinct function a leaf list needs, plus the transitive same-module helper closure,
  higher-order monomorphisation, list-literal table lookups, sequence-static
  monomorphisation, and the hand-written `REGISTRY` (including `gamma`/`gammaln`, which
  Warp has no builtin for at all).
- `leaf_funcs_arrays.py` -- the same builder extended for array-valued parameters,
  `jnp.interp`/`searchsorted`, per-species 2-D tables, and built (scalarised) leaves. Only
  a leaf `leaf_funcs.py` refuses for an array-shaped reason is retried through it.
- `emit.py` -- **the kernel emitter**: the leaves in topological order, as one kernel,
  with the unknowns and boundary as `wp.array2d` columns and array/table inputs as their
  own parameters.
- `reference.py` -- the same sub-DAG evaluated in JAX. Every emitter concession is
  mirrored here, because an agreement check can only ever say the two engines agree with
  each other; its value rests entirely on their being the same computation.
- `harness.py` -- **the harness**: builds the maximal prefix-closed, fully-emittable
  sub-DAG for one config, compiles it, compares it against `reference.py` and times it.
  Run from the repository root:

      python -m functional_process.cottax.warp.harness helias_5b

- `mapper.py`, `regcheck.py` -- `VarPath`-to-identifier minting, and CUDA register/spill
  measurement via the driver API.

**Self-validating**, which is what makes 700 generated functions reviewable at all: every
emitted `@wp.func` has its JAX original beside it, so the generator is checked per function
on random inputs rather than by reading its output. `harness.AGREEMENT_RTOL` (1e-12) is the
gate; the worst relative difference is reported as a number regardless, because the number
is the evidence and the gate is only a summary of it.
"""
