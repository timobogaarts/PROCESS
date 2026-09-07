"""**Standalone**: which Warp constructs have a correct reverse-mode adjoint, and which
do not. No PROCESS, no cottax, no generated kernels -- a few lines of Warp per shape,
`wp.Tape` against `jax.grad` on the identical arithmetic, with a central finite
difference as an independent third opinion.

    JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        python -m functional_process.cottax.warp.adjoint_probe

It exists because `jaxpr_backend`'s loop and fusion docstrings make claims about Warp's
adjoint that decide what this backend is allowed to emit -- `_LOOPS`, `_FUSION`,
`_FUSABLE_REDUCE`, `_scan_loop_ok`, `_while` -- and a claim of that weight should be
reproducible by someone who does not have a PROCESS input file. One Warp compile per
shape per trip count, a couple of minutes cold and seconds warm. It is also the form an
upstream Warp bug report would take.

**Every number it prints is a relative difference against `jax.grad`. There is no
tolerance in this file and nothing to tune.**

What it establishes (Warp 1.17.0, CPU backend, float64):

  1. **`max_unroll` is a correctness cliff, not a performance knob.** A loop Warp
     unrolls has an exact adjoint; the same loop one iteration longer does not. Warp's
     reverse pass for a loop it did not unroll REPLAYS the body against the current
     values of anything carried across the loop -- which after the forward pass are the
     values from the LAST iteration, not from the iteration being reversed. Wrong
     exactly when the body's derivative depends on the carry.

  2. **The `@wp.func` workaround does not fix it.** Warp's docs suggest moving a dynamic
     loop's body into a `@wp.func` so it is replayed. It moves the wrong answer; it does
     not make it right. A `@wp.func`'s adjoint recomputes its forward pass from the
     arguments it is handed, and the argument is the same stale carry.

  3. **Checkpointing the carry in a `wp.array` DOES fix it** -- exactly, and with the
     loop intact. A `wp.vec` is not a substitute.

  4. **`wp::adj_assign_inplace` never zeroes the destination's adjoint**
     (`warp/native/vec.h:936`: `adj_value += adj_v[idx];`). An assignment must CONSUME
     the adjoint of the slot it overwrites; because it does not, a re-assigned `vec`
     component leaks its downstream adjoint past the assignment. Needs no loop at all.

  5. **Anything a non-unrolled loop PRODUCES is zero for every later statement's
     adjoint.** The most consequential one, and it is not the "stale carry" of (1). In
     the generated backward kernel the forward section emits a dynamic loop's HEADER
     and not its body -- the body appears only inside that loop's own reverse block,
     as its replay. So a `vec` filled by loop 1 is still the all-zero `vec_t<>()` when
     loop 2's reverse runs (reverse order puts loop 2 first). `adj_extract` needs no
     value and survives; a `mul`'s adjoint w.r.t. the OTHER factor is
     `adj_other += adj_ret * v[m]` and quietly gets `v[m] == 0`.

     Hence the pair of rows below: one `vec` built by a loop, SUMMED downstream, is
     exact; the identical `vec` MULTIPLIED downstream loses a whole path. Forward
     values are exact in both cases, which is why nothing caught it.
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

MAX_UNROLL_DEFAULT = 16
"""Warp's own default (`warp/_src/codegen.py`, `builder_options["max_unroll"]`). A
`range(n)` with `n` above it is compiled as a DYNAMIC loop, with a warning on stderr."""

X0 = 1.3
FD_H = 1e-6


# ---------------------------------------------------------------------------
# the shapes
# ---------------------------------------------------------------------------

def _recurrence(x, m, n):
    """`c <- sqrt(c*c + x)/(1 + c)`, whose derivative DOES depend on the carry."""
    c = x
    for _ in range(n):
        c = m.sqrt(c * c + x) / (1.0 + c)
    return c


def _linear(x, m, n):
    """`c <- c/2 + x`, whose derivative does NOT depend on the carry."""
    c = x
    for _ in range(n):
        c = 0.5 * c + x
    return c


def _sum_nest(x, m, n):
    a = 0.0
    for k in range(n):
        a += m.sqrt(x * (k + 1))
    return a


def _prod_nest(x, m, n):
    a = 1.0
    for k in range(n):
        a *= 1.0 + x * 0.01 * (k + 1)
    return a


def _square(x, m, n):
    return x * x


def _tri(n):
    return float(sum(k + 1 for k in range(n)))


def _vec_then_sum(x, m, n):
    """v[k] = x*x*(k+1) in one loop, SUMMED in a second: no product downstream."""
    return x * x * _tri(n)


def _vec_then_product(x, m, n):
    """The same vec MULTIPLIED by x in the second loop -- the `_emit_chain` shape that
    `.physics.fusion_rates` has, and the one that fails.
    """
    return x * x * x * _tri(n)


SHAPES = {
    # --- Warp source for the kernel body, given n; and its reference function ---
    "local carry, inline body": ("""
    c = wp.float64(x[0])
    for i in range({n}):
        t = wp.sqrt(c * c + x[0]) / (wp.float64(1.0) + c)
        c = t
    out[0] = c""", _recurrence),
    "local carry, @wp.func body": ("""
    c = wp.float64(x[0])
    for i in range({n}):
        t = step(c, x[0])
        c = t
    out[0] = c""", _recurrence),
    "wp.array checkpoint": ("""
    hist[0] = x[0]
    for i in range({n}):
        c = hist[i]
        t = wp.sqrt(c * c + x[0]) / (wp.float64(1.0) + c)
        hist[i + 1] = t
    out[0] = hist[{n}]""", _recurrence),
    "wp.vec checkpoint": ("""
    v = vecf()
    v[0] = x[0]
    for i in range({n}):
        c = v[i]
        t = wp.sqrt(c * c + x[0]) / (wp.float64(1.0) + c)
        v[i + 1] = t
    out[0] = v[{n}]""", _recurrence),
    "local carry, LINEAR body": ("""
    c = wp.float64(x[0])
    for i in range({n}):
        t = wp.float64(0.5) * c + x[0]
        c = t
    out[0] = c""", _linear),
    # `_emit_chain`'s materialised output: one vec, each slot written once, never read
    "vec write-once in loop": ("""
    v = vecf()
    for k in range({n}):
        t = wp.sqrt(x[0] * wp.float64(k + 1))
        v[k] = t
    s = wp.float64(0.0)
    for j in range({n}):
        s = s + v[j]
    out[0] = s""", _sum_nest),
    # `_emit_chain`'s reduce_sum root: accumulator linear in the carry
    "reduce_sum accumulator": ("""
    a = wp.float64(0.0)
    for k in range({n}):
        t = wp.sqrt(x[0] * wp.float64(k + 1))
        a = (a + t)
    out[0] = a""", _sum_nest),
    # `_emit_chain`'s reduce_prod root: accumulator NONLINEAR in the carry
    "reduce_prod accumulator": ("""
    a = wp.float64(1.0)
    for k in range({n}):
        t = wp.float64(1.0) + x[0] * wp.float64(0.01) * wp.float64(k + 1)
        a = (a * t)
    out[0] = a""", _prod_nest),
    # a vec BUILT by one loop and only SUMMED by the next -- no product downstream
    "vec from loop, summed": ("""
    v = vecf()
    for k in range({n}):
        v[k] = x[0] * x[0] * wp.float64(k + 1)
    a = wp.float64(0.0)
    for m in range({n}):
        a = (a + v[m])
    out[0] = a""", _vec_then_sum),
    # THE SAME vec, MULTIPLIED in the next loop. `.physics.fusion_rates`'s shape.
    "vec from loop, multiplied": ("""
    v = vecf()
    for k in range({n}):
        v[k] = x[0] * x[0] * wp.float64(k + 1)
    a = wp.float64(0.0)
    for m in range({n}):
        a = (a + v[m] * x[0])
    out[0] = a""", _vec_then_product),
    # no loop at all -- `wp::adj_assign_inplace` does not zero the overwritten slot
    "vec slot overwritten (no loop)": ("""
    v = vecf()
    v[0] = x[0]
    v[0] = v[0] * v[0]
    out[0] = v[0]""", _square),
}

MODULE = """
import warp as wp

wp.set_module_options({{"max_unroll": {mu}}})
vecf = wp.types.vector(length={vn}, dtype=wp.float64)


@wp.func
def step(c: wp.float64, x: wp.float64) -> wp.float64:
    return wp.sqrt(c * c + x) / (wp.float64(1.0) + c)


@wp.kernel
def K(x: wp.array(dtype=wp.float64), hist: wp.array(dtype=wp.float64),
      out: wp.array(dtype=wp.float64)):
{body}
"""


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------

def _build(wp, tmp: pathlib.Path, tag: str, body: str, n: int, mu: int):
    src = MODULE.format(mu=mu, vn=max(n + 1, 2), body=body.format(n=n))
    path = tmp / f"_probe_{tag}.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location(f"_probe_{tag}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.K


def _warp_grad(wp, kernel, n: int):
    x = wp.array([X0], dtype=wp.float64, device="cpu", requires_grad=True)
    hist = wp.zeros(n + 1, dtype=wp.float64, device="cpu", requires_grad=True)
    out = wp.array([0.0], dtype=wp.float64, device="cpu", requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.launch(kernel, dim=1, inputs=[x, hist, out], device="cpu")
    val = float(out.numpy()[0])
    tape.backward(grads={out: wp.array([1.0], dtype=wp.float64, device="cpu")})
    return val, float(x.grad.numpy()[0])


def run(trip_counts=(8, 64), max_unroll: int = MAX_UNROLL_DEFAULT):
    """Measure every shape at every trip count; return the rows that disagree.

    A row is reported wrong on the relative difference to `jax.grad` alone. The finite
    difference is not a third vote -- it is a check that `jax.grad` is itself believable
    for that row, and a row where it is not says so and asks to be distrusted.
    """
    import warp as wp

    wp.config.enable_backward = True
    wp.config.quiet = True
    wp.init()

    print(f"warp {wp.__version__}, CPU, float64, max_unroll = {max_unroll}, "
          f"x = {X0}")
    print(f"{'shape':<32} {'range':>7} {'loop':>9} {'wp.Tape':>22} "
          f"{'jax.grad':>22} {'rel':>11}")
    bad = []
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        for i, (name, (body, ref)) in enumerate(SHAPES.items()):
            for n in trip_counts:
                jg = float(jax.grad(lambda z, n=n, f=ref: f(z, jnp, n))(X0))
                lo = float(ref(np.float64(X0 - FD_H), np, n))
                hi = float(ref(np.float64(X0 + FD_H), np, n))
                fd = (hi - lo) / (2 * FD_H)
                kern = _build(wp, tmp, f"{i}_{n}_{max_unroll}", body, n, max_unroll)
                val, g = _warp_grad(wp, kern, n)
                rel = abs(g - jg) / abs(jg) if jg else abs(g - jg)
                vref = float(ref(np.float64(X0), np, n))
                vrel = abs(val - vref) / abs(vref) if vref else abs(val - vref)
                kind = "unrolled" if n <= max_unroll else "DYNAMIC"
                mark = "" if rel < 1e-12 else "   <-- WRONG"
                print(f"{name:<32} {n:>7} {kind:>9} {g:>22.17g} {jg:>22.17g} "
                      f"{rel:>11.3e}{mark}")
                if vrel > 1e-14:
                    print(f"{'':<32} FORWARD VALUE also disagrees: rel {vrel:.3e}")
                if abs(fd - jg) > 1e-6 * max(abs(jg), 1.0):
                    print(f"{'':<32} (finite difference {fd:.14g} does not confirm "
                          f"jax.grad -- distrust this row)")
                if rel >= 1e-12:
                    bad.append((name, n, rel))
    print()
    if bad:
        print("wrong adjoints:")
        for name, n, rel in bad:
            print(f"  range({n}) {name}: rel {rel:.6g}")
    else:
        print("every shape exact")
    return bad


if __name__ == "__main__":
    import sys

    counts = tuple(int(s) for s in sys.argv[1].split(",")) if len(sys.argv) > 1 \
        else (8, 64)
    mu = int(sys.argv[2]) if len(sys.argv) > 2 else MAX_UNROLL_DEFAULT
    run(counts, mu)
