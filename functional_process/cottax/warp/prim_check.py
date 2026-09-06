"""**The primitive-table check**: every scalar primitive the backend maps, Warp against
XLA, over a deliberately hostile set of arguments.

    python -m functional_process.cottax.warp.prim_check

`jaxpr_validate` checks each graph NODE against its own JAX function, which is the
strongest check available -- but it only ever exercises a primitive at the arguments
that node's physics happens to produce. This checks the primitive TABLE itself: does
`wp.X` mean what `lax.X` means at zero, at a negative, at an infinity, at a NaN, at
1e-300 and at 1e300, not just in the middle of the domain? Those are precisely the
arguments a converging solver wanders into and a converged one never visits.

**It has earned its place three times.** All of these were live in the emitted kernels,
all produce a plausible finite number rather than an error, and none is visible at a
node's physical operating point:

- `wp.max`/`wp.min`/`wp.clamp` are `a < b ? b : a` in C++, so every comparison against a
  NaN is false and the NaN is DISCARDED -- where XLA's propagate it. This turned
  `.tokamak.cs_coil.temperature_margin`, a secant root-find, from JAX's honest `nan` off
  its domain into a finite wrong answer at 3 of 8 swept draws.
- `wp.sign(0.0)` is `+1`; `lax.sign(0.0)` is `0`. 66 of 484 swept arguments disagreed,
  none of them involving a NaN -- an ordinary in-domain wrong answer.
- `wp.asin`/`wp.acos` CLAMP their argument into [-1, 1] before calling libm, so
  `wp.asin(2.0)` is pi/2 where `lax.asin(2.0)` is NaN: 264 of 484 arguments, every one
  an invented finite answer outside the domain. In-domain they were already bit-exact.

All three are now emitted through `jaxpr_backend.HELPERS` instead of the Warp builtin,
and this module is what says so.

The Warp side is emitted **by the backend itself** (`emit_node` on a one-primitive
function), not hand-written, so what is compared is what a real node would get.

Three differences are known, measured, and NOT defects in the emitter:

- **Subnormals.** XLA on this CPU backend flushes results below the smallest normal
  double (~2.2e-308) to zero -- eagerly, under `jit` and under `vmap` alike -- while
  Warp returns the correct IEEE subnormal. `1e-8 * 1e-300` is `0.0` in JAX and
  `1e-308` in Warp. It is XLA that departs from IEEE here, it is unreachable by any
  quantity in this graph, and "fixing" it would mean emitting a flush-to-zero after
  every multiply.
- **`lgamma` below x = 0.5.** A DELIBERATE named refusal, not a disagreement to fix:
  XLA's Euler-reflection branch there contains `log(sin(pi*x))`, whose adjoint is +-inf
  at every integer, so `HELPERS["_lgamma"]` returns NaN below 0.5 rather than the
  unreflected Lanczos value. It shows here as 132 one-sided non-finites; in its own
  domain it is bit-exact (worst relative difference 0.000e+00).
- **Transcendentals.** `exp`, `tanh`, `acos`, `sinh` and `cosh` differ from XLA's own
  expansions by a few ulp (worst measured 1.8e-15 on `sinh`/`cosh`), which is three
  orders of magnitude inside the 1e-12 gate. `log1p` does NOT appear in that list: it is
  routed through a bit-exact transcription of XLA's own expansion (`HELPERS`), as is
  `lgamma` in its domain -- those are the two whose 1-ulp error compounded.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax import lax

from . import jaxpr_backend as jb

GEN_DIR = "functional_process/warp"

AGREEMENT_RTOL = 1e-12
"""The same gate the two harnesses use, for the same reason."""

VALUES = (0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 3.7, -3.7, 1e-8, -1e-8,
          1e8, -1e8, 1e-300, 1e300, np.inf, -np.inf, np.nan, 7.0, 0.25, 100.0)
"""Every pair from this set is tried, so a binary primitive sees 484 argument pairs.
Chosen for the awkward cases -- both signed zeros, both infinities, a NaN, both ends of
the exponent range -- rather than for physical plausibility."""

UNARY = ("exp", "log", "sqrt", "sin", "cos", "tan", "tanh", "asin", "acos", "atan",
         "sinh", "cosh", "floor", "ceil", "round", "sign", "cbrt", "erf", "abs",
         "neg", "rsqrt", "logistic", "square", "is_finite", "log1p", "lgamma")
BINARY = ("add", "sub", "mul", "div", "max", "min", "pow", "atan2", "rem",
          "eq", "ne", "lt", "le", "gt", "ge", "nextafter")

_JAX_FN = {
    "neg": lambda x: -x, "abs": jnp.abs, "square": lax.square, "rsqrt": lax.rsqrt,
    "logistic": jax.nn.sigmoid, "is_finite": lax.is_finite, "log1p": lax.log1p,
    "lgamma": lax.lgamma, "sign": lax.sign, "cbrt": lax.cbrt, "erf": lax.erf,
    "round": lax.round, "nextafter": lax.nextafter,
}


def _jax_fn(name):
    return _JAX_FN.get(name) or getattr(lax, name, None) or getattr(jnp, name)


def build(names=None):
    """`(module, checked, refused)` -- one `@wp.func` per primitive, emitted by the
    backend, plus one kernel each. A primitive the backend REFUSES is reported by name
    and skipped; that is a result, not a failure."""
    wanted = set(names) if names else None
    srcs, kernels, checked, refused = [], [], [], []
    helpers: set = set()
    for name, arity in [(n, 1) for n in UNARY] + [(n, 2) for n in BINARY]:
        if wanted is not None and name not in wanted:
            continue
        fn_jax = _jax_fn(name)
        f = (lambda a, _f=fn_jax: _f(a)) if arity == 1 else \
            (lambda a, b, _f=fn_jax: _f(a, b))
        try:
            info = jb.emit_node(f, f"p_{name}", [jnp.float64(1.5)] * arity, 1)
        except Exception as exc:
            refused.append((name, str(exc)[:100]))
            continue
        srcs.append(info.source)
        helpers |= set(info.helpers)
        args = ", ".join(f"a[tid, {i}]" for i in range(arity))
        kernels.append(
            f"@wp.kernel\ndef k_{name}(a: wp.array2d(dtype=wp.float64), "
            f"r: wp.array(dtype=wp.float64)):\n    tid = wp.tid()\n"
            f"    r[tid] = wp.float64(p_{name}({args}))\n")
        checked.append((name, arity, fn_jax))
    pre = "\n".join(jb.HELPERS[h] for h in jb.helper_closure(helpers))
    os.makedirs(GEN_DIR, exist_ok=True)
    path = f"{GEN_DIR}/_prim_check.py"
    with open(path, "w") as fh:
        fh.write('"""GENERATED by functional_process/cottax/warp/prim_check -- do not '
                 'hand-edit."""\nimport warp as wp\n\n' + pre + "\n\n"
                 + "\n".join(srcs) + "\n\n" + "\n".join(kernels))
    spec = importlib.util.spec_from_file_location("_prim_check", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, checked, refused


def check(verbose: bool = True):
    """`(rows, refused)` -- one row per primitive: how many of the swept arguments
    disagree, and by how much relatively.

    A NaN on both sides counts as agreement (it is the same answer); a NaN on one side
    only does not, and shows up as a non-finite worst-case that cannot pass the gate.
    """
    wp = jb.warp_init()
    mod, checked, refused = build()

    a_grid, b_grid = np.meshgrid(np.asarray(VALUES), np.asarray(VALUES), indexing="ij")
    A, B = a_grid.ravel(), b_grid.ravel()
    inp = wp.array(np.stack([A, B], axis=1), dtype=wp.float64, device="cpu")

    rows = []
    for name, arity, fn_jax in checked:
        r = wp.zeros(len(A), dtype=wp.float64, device="cpu")
        wp.launch(getattr(mod, f"k_{name}"), dim=len(A), inputs=[inp, r], device="cpu")
        wp.synchronize()
        warp_out = r.numpy()
        jax_out = np.asarray(
            jax.jit(jax.vmap(fn_jax))(jnp.asarray(A), jnp.asarray(B)) if arity == 2
            else jax.jit(jax.vmap(fn_jax))(jnp.asarray(A))).astype(float)
        both_nan = np.isnan(warp_out) & np.isnan(jax_out)
        same = (warp_out == jax_out) | both_nan
        rel = np.abs(warp_out - jax_out) / np.maximum(np.abs(jax_out), 1e-300)
        rel[both_nan] = 0.0
        one_sided = int((np.isfinite(warp_out) ^ np.isfinite(jax_out)).sum())
        finite = np.isfinite(rel)
        worst = float(rel[finite].max()) if finite.any() else float("nan")
        rows.append({"prim": name, "arity": arity, "n_args": len(A),
                     "n_differing": int((~same).sum()), "worst_rel": worst,
                     "one_sided": one_sided,
                     "verdict": "PASS" if (one_sided == 0 and worst <= AGREEMENT_RTOL)
                     else "OVER-GATE"})
    if verbose:
        _report(rows, refused)
    return rows, refused


def _report(rows, refused):
    exact = [r for r in rows if r["n_differing"] == 0]
    print(f"\n[prim-check] {len(rows)} primitives, {rows[0]['n_args']} argument pairs "
          f"each, against XLA")
    print(f"[prim-check] {len(exact)}/{len(rows)} agree with XLA at EVERY argument")
    for r in sorted((x for x in rows if x["n_differing"]),
                    key=lambda x: -(x["worst_rel"] if np.isfinite(x["worst_rel"]) else 1e9)):
        print(f"    {r['verdict']:<10} {r['prim']:<12} {r['n_differing']:4d} differing, "
              f"worst rel {r['worst_rel']:.3e}, one-sided {r['one_sided']}")
    over = [r for r in rows if r["verdict"] != "PASS"]
    if over:
        known = {"mul", "div", "atan2", "lgamma"}
        unexpected = [r["prim"] for r in over if r["prim"] not in known]
        print(f"[prim-check] {len(over)} over the {AGREEMENT_RTOL:.0e} gate: "
              f"{sorted(r['prim'] for r in over)}")
        print(f"[prim-check] KNOWN and explained in the module docstring: "
              f"{sorted(known & {r['prim'] for r in over})} "
              f"(XLA's subnormal flush-to-zero; `lgamma`'s x < 0.5 refusal)")
        print(f"[prim-check] *** UNEXPECTED: {unexpected or 'none'} ***")
    if refused:
        print(f"[prim-check] refused by the backend ({len(refused)}):")
        for name, why in refused:
            print(f"    {name:<12} {why}")


if __name__ == "__main__":
    check()
    sys.exit(0)
