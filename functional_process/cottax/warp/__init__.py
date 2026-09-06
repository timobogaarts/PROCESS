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

- `transpile.py` -- one `models/**` function to one `@wp.func`. `jnp.X` to `wp.X`, every
  numeric literal to `wp.float64(n)` (Warp is strictly typed and does not promote), an
  annotated signature, and a return annotation **only** when the body returns a single
  value. **It refuses rather than guesses**: an unrecognised construct raises `Unsupported`
  and the function goes to the hand-written registry. A transpiler that silently
  mistranslates one physics formula is worse than one that covers less and says so.
- a resolver -- each graph node to the leaf function it delegates to, with its arguments in
  signature order and its frozen switch values as literals. Guarded by an **arity
  invariant**: a resolved leaf's return arity must equal the node's declared output count,
  or the node is refused. That check caught three silent mis-resolutions
  (`_audit/optimise_design.md` §94).
- an emitter -- the leaves in topological order, as one kernel.

**Self-validating**, which is what makes 700 generated functions reviewable at all: every
emitted `@wp.func` has its JAX original beside it, so the generator is checked per function
on random inputs rather than by reading its output.
"""
