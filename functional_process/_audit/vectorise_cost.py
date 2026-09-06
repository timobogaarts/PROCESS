"""What vectorising structurally-identical nodes would buy -- the controlled version.

The measurement behind `optimise_design.md` §51. The port's graph is ~500 nodes each
computing a handful of SCALARS, which is why 41.6 % of its StableHLO is shape plumbing
(§50). This is that shape in isolation: N nodes running the same ten-op scalar recipe,
written first as N separate scalar computations -- exactly how the port emits them today
-- and then as one computation over a length-N array.

    $PY functional_process/_audit/vectorise_cost.py

Measured 2026-09-06 at N = 600: **809 768 chars / 186.2 MB / 2.39 s** against
**1 215 chars / 2.5 MB / 0.03 s**, with `max |diff| = 0.000e+00`. 666x less IR, 74x less
memory, the identical answer bit for bit.

**Read the caveat before quoting the number.** Those N nodes are structurally identical,
which is what `vmap` requires; the port's nodes are mostly *different formulas*, and no
vectorisation merges two different equations. This prices the mechanism, not the port.
The achievable gain is bounded by how much of the graph is genuinely repeated structure,
and that fraction is unmeasured -- see `next_steps.md` item 3.

Also records the MLIR bytecode size, which is what `jax.export` stores (`jax.export`
itself raises `ImportError: Please install flatbuffers` in this env). It is ~21 % of the
text and buys no memory: an `Exported` still holds StableHLO, so loading one compiles it
again.

Not a test. Lives here with `hlo_anatomy.py` and `rss_per_program.py`: it measures the
port rather than being part of it.
"""

import ctypes, io, time
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax._src import compiler

_libc = ctypes.CDLL("libc.so.6")
CAP = {}
_original = compiler.backend_compile_and_load


def rss_kb():
    with open("/proc/self/status") as h:
        for line in h:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return -1


def _hook(*args, **kwargs):
    module = next((a for a in args if hasattr(a, "operation")), None)
    text = ""
    size = -1
    if module is not None:
        try:
            text = module.operation.get_asm()
            buf = io.BytesIO()
            module.operation.write_bytecode(buf)
            size = len(buf.getvalue())
        except Exception:
            pass
    _libc.malloc_trim(0)
    before = rss_kb()
    began = time.perf_counter()
    out = _original(*args, **kwargs)
    CAP.update(
        chars=len(text),
        bytecode=size,
        rss=rss_kb() - before,
        secs=time.perf_counter() - began,
    )
    return out


compiler.backend_compile_and_load = _hook


def recipe(a, b, c):
    """Ten scalar ops -- roughly what one small port node does."""
    t = a * b + c
    t = t / (a + 1.0)
    t = t - jnp.tanh(b)
    t = t * jnp.sqrt(jnp.abs(c) + 1.0)
    return t + a * c


N = 600
key = np.arange(1.0, N + 1.0)
A, B, C = key, key * 0.5, key * 0.25


def scalar_version(a, b, c):
    """N separate scalar computations -- the port's current shape."""
    return jnp.stack([recipe(a[i], b[i], c[i]) for i in range(N)])


def vector_version(a, b, c):
    """The identical arithmetic, once, over a length-N array."""
    return recipe(a, b, c)


for label, fn in (("scalar x600", scalar_version), ("vectorised", vector_version)):
    CAP.clear()
    f = jax.jit(fn)
    out = f(jnp.asarray(A), jnp.asarray(B), jnp.asarray(C))
    out.block_until_ready()
    print(
        f"{label:<14} StableHLO {CAP['chars']:>10,} chars   "
        f"bytecode {CAP['bytecode']:>9,} B   "
        f"compile {CAP['rss'] / 1024:>7.1f} MB / {CAP['secs']:>5.2f} s"
    )
    if label == "scalar x600":
        ref = np.asarray(out)
    else:
        print(
            f"{'':14} same answer: "
            f"max |diff| = {np.max(np.abs(np.asarray(out) - ref)):.3e}"
        )

# What `jax.export` actually stores, for the vectorised one.
try:
    from jax import export as jexport

    exp = jexport.export(jax.jit(vector_version))(
        jnp.asarray(A), jnp.asarray(B), jnp.asarray(C)
    )
    print(
        f"\njax.export of the vectorised version: "
        f"serialize() = {len(exp.serialize()):,} bytes, "
        f"mlir_module() = {len(exp.mlir_module()):,} chars"
    )
    exp2 = jexport.export(jax.jit(scalar_version))(
        jnp.asarray(A), jnp.asarray(B), jnp.asarray(C)
    )
    print(
        f"jax.export of the scalar version    : "
        f"serialize() = {len(exp2.serialize()):,} bytes, "
        f"mlir_module() = {len(exp2.mlir_module()):,} chars"
    )
except Exception as exc:
    print("jax.export:", type(exc).__name__, exc)
