"""Generate a Warp megakernel from a cottax graph.

**The generator, not the generated.** This package reads the assembled graph and emits
Warp source; the emitted `@wp.func` bodies and kernels are **not committed** -- they are
derived, they would be a five-figure diff on every regeneration, and the generator plus
its self-validation is the thing worth reviewing. Regenerate into
`functional_process/warp/` (gitignored) rather than reading it out of the repository.

Why it lives under `cottax/` rather than beside the models: it **imports cottax** to read
the graph, and `tests/test_cottax_boundary.py` asserts that only things under `cottax/`
do. The generated output imports nothing but `warp`, so it may live outside.

**One node, one `@wp.func`, emitted from the node's own jaxpr.** There used to be a
second way of getting there -- a *resolver* that matched each graph node against "the
function it really is" in `functional_process/models/**`, then transpiled that function's
Python source. It is gone (deleted 2026-09-07). Its whole taxonomy of refusals --
`Composition`, `self.<attr>`, "ambiguous among 4 candidate calls", arity mismatch,
non-literal exponents, switch-valued index bounds -- was a taxonomy of *Python shapes*,
not a property of the physics being ported, and tracing dissolves every one of them by
construction: `jax.make_jaxpr(node.fn)(*reads)` returns straight-line primitives applied
to typed values, with every static decision already resolved against the concrete
configuration. There is nothing left to match, so there is nothing left to get wrong
about matching.

The pieces, in the order they run:

- `assemble.py` -- one configuration's `IN.DAT` to its SAND `Drive`, and to the
  completed MDA run's own output env. That env is not incidental: a `Drive`'s context
  includes variables with no `DataStructure` field and no native answer (the 201-point
  profile grid among them), and everything the graph produces is grounded by the graph.
- `jaxpr_backend.py` -- **the backend**: trace each node at concrete values, walk the
  jaxpr, emit one `@wp.func`. Arrays are SCALARISED (one Warp local per element), so
  every shape primitive is index arithmetic done by the generator and nothing but scalar
  float64 arithmetic survives to runtime; an array crossing a *node boundary* becomes
  one fixed-length `wp.types.vector`. **It refuses rather than guesses**: a primitive
  with no entry, a shape change that is not provably the identity, a `scan` too long to
  unroll, a body too large to compile -- each raises a `Refusal` naming what it is.
  Where a primitive's meaning is intricate rather than absent (`gather`, `scatter`), the
  derived index map is proved against `lax.gather`/`lax.scatter` themselves before a
  line is emitted.
- `emit.py` -- **the kernel emitter**: the nodes in topological order, as one kernel,
  with the unknowns and scalar boundary as `wp.array2d` columns and each array-valued
  boundary as its own `wp.array` parameter.
- `jaxpr_validate.py` -- **per-node validation**: every emitted `@wp.func` against its
  own `defn.fn` in JAX, at the same inputs, swept over several draws. This is the check
  the backend is worth having: a whole-sub-DAG comparison certifies forty functions with
  one number and gives a free pass to any node whose contribution cancels or is swamped.

      python -m functional_process.cottax.warp.jaxpr_validate helias_5b

- `prim_check.py` -- **the primitive-table check**: every scalar primitive the backend
  maps, Warp against XLA, over a hostile argument set (both signed zeros, both
  infinities, a NaN, 1e-300 and 1e300). Per-node validation only ever reaches a
  primitive at arguments that node's physics produces; this reaches the arguments a
  *converging* solver wanders into. It has found three live defects, each of which
  returned a plausible finite number: `wp.max`/`wp.min`/`wp.clamp` discarding a NaN
  XLA propagates, `wp.sign(0.0)` answering `+1` where XLA answers `0`, and
  `wp.asin`/`wp.acos` clamping their argument into [-1, 1] where XLA returns NaN.

      python -m functional_process.cottax.warp.prim_check

- `jaxpr_harness.py` -- **the end-to-end harness**: the maximal prefix-closed sub-DAG of
  one config's Drive, compiled as one kernel and compared against JAX evaluating the
  identical sub-DAG. Weaker than per-node validation, and reported alongside it rather
  than instead of it; its job is to show that the pieces COMPOSE.

      python -m functional_process.cottax.warp.jaxpr_harness helias_5b

- `mapper.py`, `regcheck.py` -- `VarPath`-to-identifier minting, and CUDA register/spill
  measurement via the driver API.

**Self-validating**, which is what makes hundreds of generated functions reviewable at
all: every emitted `@wp.func` has its JAX original beside it, so the generator is checked
per function on random inputs rather than by reading its output. `AGREEMENT_RTOL` (1e-12)
is the gate; the worst relative difference is reported as a number regardless, because
the number is the evidence and the gate is only a summary of it.
"""
