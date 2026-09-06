"""**The jaxpr backend**: one `@wp.func` per graph node, emitted from the node's own
jaxpr rather than from a source-level match against "the function it really is".

This replaced -- and has now outlived -- `resolve.py`, which was deleted on 2026-09-07
once this path covered more of every configuration. The resolver's taxonomy of refusals
-- `Composition`, `self.<attr>`, "ambiguous among 4 candidate calls", arity mismatch,
non-literal exponents, switch-valued index bounds -- was a taxonomy of *Python shapes*,
not a property of the physics being ported. Tracing dissolves every one of them by
construction: `jax.make_jaxpr(node.fn)(*reads)` returns straight-line primitives applied
to typed values, with every static decision already resolved against the concrete
configuration. There is nothing left to match.

What the backend still has to get right is the primitive table, and it **refuses rather
than guesses**: a primitive with no entry, an array-shaped value in a scalar kernel, or a
shape change that is not provably the identity raises `Refusal` naming the primitive.
That rule is where every correctness result in this port came from.

Public surface:

  `jaxpr_leaves(config)` -> `(entries, refused, drive)`, the drop-in replacement for
  `combined.combined_ordered` -- a topologically ordered list of `JaxprLeaf`
  (duck-typing `combined.Leaf` closely enough for `emit.build_kernel_source`) plus the
  emitted `@wp.func` source for each.

  `emit_node(defn, name, values)` -> `(source, n_returns)` for one node.

Design notes that matter:

- **Parameter names come from nothing at all.** The reads are bound positionally, in
  `defn.reads` order, which is the order `cottax` itself calls `fn` with. The resolver's
  need to recover parameter *names* (and the wall of bogus "unexpected keyword argument"
  failures that came from taking them off the last component of a `VarPath`) is gone with
  the resolver.
- **Every parameter and every return is `wp.float64`**, matching the kernel's convention
  (`emit.py`; Warp does not promote). Concrete values are cast to float64 before tracing
  so the traced jaxpr is the one the kernel will actually run.
- **`integer_pow` reproduces JAX's own binary-exponentiation expansion** (`lax.py`'s
  `_integer_pow`) rather than calling `wp.pow`, so `x**2` is `x*x` on both sides.
- **`squeeze`/`broadcast_in_dim`/`reshape` are the identity only when both sides hold
  exactly one element**, which is a proof, not a convention. A real broadcast refuses.
"""
from __future__ import annotations

import dataclasses
import re
import math

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax.extend.core import ClosedJaxpr, Literal  # noqa: F401  -- `jax.core` no longer
# exports either as of jax 0.11; `jax.extend.core` is the supported spelling.


class Refusal(Exception):
    """A construct this backend will not emit. Always names the primitive (or the
    shape) responsible -- never a generic failure."""


# ---------------------------------------------------------------------------
# entry type -- duck-types `combined.Leaf` for `emit.build_kernel_source`
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class JaxprLeaf:
    """One graph node, emitted from its jaxpr.

    Deliberately the same *shape* `emit.py` already consumes: `order` is empty (so
    `_call_terms` falls back to plain positional binding of `inputs`), `statics` is
    empty (a jaxpr has no statics -- every static decision was resolved during the
    trace), and `output_index` is `None` (tracing the node's own `__call__` returns
    exactly what the node owns, so there is no wider tuple to select out of).
    """

    node: str
    fn: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    module: str = "jaxpr"
    statics: tuple = ()
    order: tuple = ()
    output_index: None = None
    locals_: tuple = ()
    prelude: tuple = ()
    source: str = ""
    """The emitted `@wp.func` text for this node."""

    n_eqns: int = 0

    input_sizes: tuple[int, ...] = ()
    """Flat element count per entry of `inputs`. `1` binds from a float64 column;
    `n > 1` binds from a `vec{n}f` local."""
    output_sizes: tuple[int, ...] = ()
    """Flat element count per entry of `outputs`, same convention."""
    helpers: tuple[str, ...] = ()
    vec_lengths: tuple[int, ...] = ()

    def dependencies(self) -> tuple:
        return self.inputs

    def array_sizes(self) -> dict:
        """`{path: n}` for every VarPath this node touches that is array-valued."""
        out = {}
        for p, n in list(zip(self.inputs, self.input_sizes)) + \
                list(zip(self.outputs, self.output_sizes)):
            if n > 1:
                out[p] = n
        return out


# ---------------------------------------------------------------------------
# values: a jaxpr variable, scalarised
# ---------------------------------------------------------------------------

_F, _I, _B = "f", "i", "b"

MAX_ELEMENTS = 8192
"""Refuse a value with more elements than this. Every array in the emitted code is
SCALARISED -- one Warp local per element -- so a genuinely large array would produce
megabytes of source rather than a wrong answer. The cap turns that into a named refusal
instead of a hang."""


def _kind(aval) -> str:
    dt = np.dtype(aval.dtype)
    if dt == np.dtype(bool):
        return _B
    if np.issubdtype(dt, np.floating):
        return _F
    if np.issubdtype(dt, np.integer):
        return _I
    raise Refusal(f"dtype {dt} is neither float, int nor bool")


def _fmt_float(v: float) -> str:
    v = float(v)
    if math.isnan(v):
        return "(wp.float64(0.0) / wp.float64(0.0))"
    if math.isinf(v):
        return ("(wp.float64(1.0e308) * wp.float64(10.0))" if v > 0
                else "(wp.float64(-1.0e308) * wp.float64(10.0))")
    return f"wp.float64({v!r})"


def _fmt_scalar(x, kind: str) -> str:
    if kind == _B:
        return "True" if x else "False"
    if kind == _I:
        return f"wp.int32({int(x)})"
    return _fmt_float(x)


@dataclasses.dataclass(frozen=True)
class Value:
    """One jaxpr variable, as a flat list of Warp expressions -- one per element, in
    row-major order -- together with its kind and shape.

    **Scalarisation is the whole array strategy.** Warp has arrays, but every array in
    this graph has a shape fixed at trace time, so unrolling one into N locals turns
    every shape primitive (`broadcast_in_dim`, `slice`, `concatenate`, `reduce_sum`,
    `transpose`) into pure index arithmetic performed by the GENERATOR, with nothing
    left at runtime but scalar float64 arithmetic -- which is what the kernel wants
    anyway, and what makes the emitted code identical in form to the hand-scalarised
    code the previous path produced.

    `vals` carries compile-time-known values when there are any (a literal, a jaxpr
    const, or the result of a pure shape rearrangement of one). It is what lets a
    `dynamic_slice` with a constant start index resolve statically instead of refusing.
    """

    exprs: tuple[str, ...]
    kind: str
    shape: tuple[int, ...]
    vals: object = None          # np.ndarray of `shape`, or None

    @property
    def size(self) -> int:
        return len(self.exprs)

    def scalar(self) -> str:
        if self.size != 1:
            raise Refusal(f"expected a scalar, got shape {self.shape}")
        return self.exprs[0]


def _const_value(arr, kind: str) -> Value:
    a = np.asarray(arr)
    if a.size > MAX_ELEMENTS:
        raise Refusal(f"constant with {a.size} elements (cap {MAX_ELEMENTS})")
    flat = a.reshape(-1)
    return Value(tuple(_fmt_scalar(x, kind) for x in flat.tolist()), kind,
                 tuple(a.shape), a)


def _check_size(aval, where: str):
    n = int(np.prod(aval.shape)) if aval.shape else 1
    if n > MAX_ELEMENTS:
        raise Refusal(f"{where}: {n} elements (cap {MAX_ELEMENTS})")
    return n


# ---------------------------------------------------------------------------
# nested-jaxpr primitives
# ---------------------------------------------------------------------------

_NESTING = {
    "pjit": ("jaxpr",),
    "jit": ("jaxpr",),
    "closed_call": ("call_jaxpr",),
    "core_call": ("call_jaxpr",),
    "xla_call": ("call_jaxpr",),
    "custom_jvp_call": ("call_jaxpr",),
    "custom_jvp_call_jaxpr": ("fun_jaxpr",),
    "custom_vjp_call": ("call_jaxpr", "fun_jaxpr"),
    "custom_vjp_call_jaxpr": ("fun_jaxpr",),
    "remat": ("jaxpr",),
    "remat2": ("jaxpr",),
    "checkpoint": ("jaxpr",),
}


def _nested(eqn):
    """`(jaxpr, consts)` for a nesting primitive, or `None`."""
    keys = _NESTING.get(eqn.primitive.name)
    if keys is None:
        return None
    for k in keys:
        j = eqn.params.get(k)
        if j is None:
            continue
        if hasattr(j, "jaxpr"):          # ClosedJaxpr
            return j.jaxpr, list(j.consts)
        if hasattr(j, "eqns"):           # open Jaxpr
            return j, []
    raise Refusal(f"{eqn.primitive.name}: no nested jaxpr found in "
                  f"params {sorted(eqn.params)}")


# ---------------------------------------------------------------------------
# the emitter
# ---------------------------------------------------------------------------

_UNARY = {
    "exp": "wp.exp", "log": "wp.log", "sqrt": "wp.sqrt", "sin": "wp.sin",
    "cos": "wp.cos", "tan": "wp.tan", "tanh": "wp.tanh",
    "atan": "wp.atan", "sinh": "wp.sinh", "cosh": "wp.cosh",
    "floor": "wp.floor", "ceil": "wp.ceil", "round": "wp.round",
    "cbrt": "wp.cbrt", "erf": "wp.erf",
}
"""jax primitive -> a Warp builtin with the same one-argument meaning."""

_BINARY_OP = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
_CMP_OP = {"eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}
_TOTAL_ORDER_CMP = {"lt_to": "_lt_to", "le_to": "_le_to", "eq_to": "_eq_to"}
_BOOL_OP = {"and": "and", "or": "or"}

_IDENTITY_FLAT = {"squeeze", "expand_dims", "copy", "stop_gradient", "device_put",
                  "optimization_barrier", "reshape"}
"""Primitives that permute nothing in row-major order: the flat element list of the
output IS the flat element list of the input. (`reshape` only when its `dimensions`
parameter is `None`; a `reshape` that carries a permutation is handled with
`transpose`.)"""


# ---------------------------------------------------------------------------
# device helpers -- XLA primitives Warp has no builtin for
# ---------------------------------------------------------------------------

HELPERS = {
    # A correctly-rounded fused multiply-add, by Dekker/Veltkamp two-product plus
    # two-sum. Warp does not contract `a*b + c` on the CPU backend (it does on CUDA),
    # so an XLA polynomial evaluated with FMAs has to say so explicitly. Verified in
    # the previous path against a `fractions.Fraction` reference on 4000/4000 random
    # triples, on cpu and cuda:0 alike.
    "_dd_fma": (
        "@wp.func\n"
        "def _dd_fma(a: wp.float64, b: wp.float64, c: wp.float64) -> wp.float64:\n"
        "    s = wp.float64(134217729.0)\n"
        "    p = a * b\n"
        "    ca = s * a\n"
        "    ah = ca - (ca - a)\n"
        "    al = a - ah\n"
        "    cb = s * b\n"
        "    bh = cb - (cb - b)\n"
        "    bl = b - bh\n"
        "    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl\n"
        "    t = p + c\n"
        "    z = t - p\n"
        "    d = (p - (t - z)) + (c - z)\n"
        "    return t + (d + e)\n"
    ),
    # `lax.log1p` is NOT libm's `log1p` (they disagree by 1 ulp on ~20 % of
    # arguments): XLA expands it as Cephes `unity.c` with FMA polynomial steps and
    # Cephes' own `x*x*x*(P/Q)` grouping. Transcribed from the compiled HLO in the
    # previous path and measured 8000/8000 bit-exact on [-0.5, 0.8].
    "_xla_log1p": (
        "@wp.func\n"
        "def _xla_log1p(x: wp.float64) -> wp.float64:\n"
        "    z1 = wp.float64(1.0) + x\n"
        "    xx = x * x\n"
        "    p = wp.float64(4.5270000862445199635215e-5)\n"
        "    p = _dd_fma(p, x, wp.float64(4.9854102823193375972212e-1))\n"
        "    p = _dd_fma(p, x, wp.float64(6.5787325942061044846969e0))\n"
        "    p = _dd_fma(p, x, wp.float64(2.9911919328553073277375e1))\n"
        "    p = _dd_fma(p, x, wp.float64(6.0949667980987787057556e1))\n"
        "    p = _dd_fma(p, x, wp.float64(5.7112963590585538103336e1))\n"
        "    p = _dd_fma(p, x, wp.float64(2.0039553499201281259648e1))\n"
        "    q = wp.float64(1.0)\n"
        "    q = _dd_fma(q, x, wp.float64(1.5062909083469192043167e1))\n"
        "    q = _dd_fma(q, x, wp.float64(8.3047565967967209469434e1))\n"
        "    q = _dd_fma(q, x, wp.float64(2.2176239823732856465394e2))\n"
        "    q = _dd_fma(q, x, wp.float64(3.0909872225312059774938e2))\n"
        "    q = _dd_fma(q, x, wp.float64(2.1642788614495947685003e2))\n"
        "    q = _dd_fma(q, x, wp.float64(6.0118660497603843919306e1))\n"
        "    poly = x + (wp.float64(-0.5) * xx + x * xx * (p / q))\n"
        "    lg = wp.log(z1)\n"
        "    return wp.where(z1 < wp.float64(0.70710678118654752440), lg,\n"
        "                    wp.where(z1 > wp.float64(1.41421356237309504880), lg, poly))\n"
    ),
    # `lax.lgamma` is not libm's either: XLA expands it inline as Lanczos (g = 7,
    # 8 coefficients), and the exact groupings below -- `z*0.13333333333333333` as a
    # multiply, the `(z + 0.5 - t/log_t)*log_t + log(sqrt(2pi))` tail as ONE FMA,
    # `log_t` as `log1p(...) + log(7.5)` -- were read off the compiled HLO in the
    # previous path, operation for operation, and measured 8009/8009 bit-exact on
    # [0.5, 12]. Re-checked here against `lax.lgamma` in this env: 120/120 bit-exact
    # (as is `_xla_log1p` against `lax.log1p`).
    #
    # **Only x >= 0.5 is implemented, and x < 0.5 is a NAMED RUNTIME REFUSAL, not a
    # guess.** XLA's Euler-reflection branch below 0.5 contains `log(sin(pi*x))`,
    # whose adjoint is +-inf at every integer and would poison Warp's reverse pass
    # through `wp.where`. Returning NaN from a compile-time constant (zero adjoint) is
    # loud; returning the unreflected Lanczos value would be quietly wrong.
    "_xla_lgamma": (
        '_XLA_GAMMA_NAN = wp.constant(wp.float64(float("nan")))\n'
        "@wp.func\n"
        "def _xla_lgamma(x: wp.float64) -> wp.float64:\n"
        "    z = x + wp.float64(-1.0)\n"
        "    log_t = _xla_log1p(z * wp.float64(0.13333333333333333)) + "
        "wp.float64(2.0149030205422647)\n"
        "    u = (z + wp.float64(0.5)) - (z + wp.float64(7.5)) / log_t\n"
        "    a = _dd_fma(u, log_t, wp.float64(0.91893853320467267))\n"
        "    s = wp.float64(676.5203681218851) / (z + wp.float64(1.0)) + "
        "wp.float64(0.99999999999980993)\n"
        "    s = s + wp.float64(-1259.1392167224028) / (z + wp.float64(2.0))\n"
        "    s = s + wp.float64(771.32342877765313) / (z + wp.float64(3.0))\n"
        "    s = s + wp.float64(-176.61502916214059) / (z + wp.float64(4.0))\n"
        "    s = s + wp.float64(12.507343278686905) / (z + wp.float64(5.0))\n"
        "    s = s + wp.float64(-0.13857109526572012) / (z + wp.float64(6.0))\n"
        "    s = s + wp.float64(9.9843695780195716e-06) / (z + wp.float64(7.0))\n"
        "    s = s + wp.float64(1.5056327351493116e-07) / (z + wp.float64(8.0))\n"
        "    return a + wp.log(s)\n"
    ),
    "_lgamma": (
        "@wp.func\n"
        "def _lgamma(x: wp.float64) -> wp.float64:\n"
        "    return wp.where(x < wp.float64(0.5), _XLA_GAMMA_NAN, _xla_lgamma(x))\n"
    ),
    # `signbit(x)` -- IEEE-754's sign BIT, which is not `x < 0`: it separates -0.0
    # from +0.0, and that separation is the only reason XLA bit-casts to int64 and
    # shifts. The reciprocal does the same separation exactly and in arithmetic:
    # 1/-0.0 is -inf and 1/+0.0 is +inf, both exact IEEE results, so the predicate is
    # equal to the sign bit for EVERY float64 except a NaN, whose sign bit this
    # returns as 0 regardless of its payload. See `_FuncEmitter._eqn`, which only ever
    # emits this for a `bitcast -> shift_right_arithmetic 63` pair.
    # `wp.max`, `wp.min` and `wp.clamp` are `a < b ? b : a` in C++, and every comparison
    # with a NaN is false -- so they return the OTHER operand and the NaN disappears.
    # XLA's `max`/`min`/`clamp` propagate it. MEASURED, not assumed, by `prim_check`:
    # against `lax.max`/`lax.min`/`lax.clamp` the unguarded builtins disagree on every
    # swept pair containing a NaN and on no other, while `sqrt`, `pow`, `log`, `abs`,
    # `floor`, `rem` and `wp.where` agree at every argument.
    #
    # This is not a corner case here. `.tokamak.cs_coil.temperature_margin` is a secant
    # root-find; off its physical domain the residual goes NaN, JAX returns NaN, and
    # the unguarded Warp version returned a finite number instead -- a plausible wrong
    # answer, which is the one outcome this backend is built to avoid. The per-node
    # sweep found it at 3 of 8 draws.
    "_nanc": '_WARP_NAN = wp.constant(wp.float64(float("nan")))\n',
    "_max": (
        "@wp.func\n"
        "def _max(a: wp.float64, b: wp.float64) -> wp.float64:\n"
        "    return wp.where(wp.isnan(a) or wp.isnan(b), _WARP_NAN, wp.max(a, b))\n"
    ),
    "_min": (
        "@wp.func\n"
        "def _min(a: wp.float64, b: wp.float64) -> wp.float64:\n"
        "    return wp.where(wp.isnan(a) or wp.isnan(b), _WARP_NAN, wp.min(a, b))\n"
    ),
    # `lax.clamp(lo, x, hi)` is XLA's `Clamp`, defined as `min(max(x, lo), hi)`. Written
    # out rather than delegated to `wp.clamp`, so the lo > hi ordering is this file's
    # statement and not an assumption about Warp's.
    "_clamp": (
        "@wp.func\n"
        "def _clamp(x: wp.float64, lo: wp.float64, hi: wp.float64) -> wp.float64:\n"
        "    return wp.where(wp.isnan(x) or wp.isnan(lo) or wp.isnan(hi), _WARP_NAN,\n"
        "                    wp.min(wp.max(x, lo), hi))\n"
    ),
    # `wp.sign(x)` is `x < 0 ? -1 : 1`, so it answers **+1 at zero**; `lax.sign` answers
    # zero there (and NaN at NaN). Measured: 66 of 484 swept arguments disagree, none of
    # them involving a NaN -- an ordinary, in-domain wrong answer, not a corner case.
    # Returning `x` itself at zero also preserves -0.0, which is what `lax.sign` does.
    "_sign": (
        "@wp.func\n"
        "def _sign(x: wp.float64) -> wp.float64:\n"
        "    return wp.where(wp.isnan(x), _WARP_NAN,\n"
        "                    wp.where(x > wp.float64(0.0), wp.float64(1.0),\n"
        "                             wp.where(x < wp.float64(0.0), "
        "wp.float64(-1.0), x)))\n"
    ),
    # Warp CLAMPS `asin`/`acos` into [-1, 1] before calling libm, so `wp.asin(2.0)` is
    # pi/2 where `lax.asin(2.0)` is NaN -- 264 of 484 swept arguments, every one of them
    # a silently invented finite answer outside the domain. In-domain nothing else is
    # wrong: with the guard, `asin` agrees with `lax.asin` at EVERY swept argument and
    # `acos` to 2.1e-16, so the guard is all that is needed.
    "_asin": (
        "@wp.func\n"
        "def _asin(x: wp.float64) -> wp.float64:\n"
        "    return wp.where(wp.isnan(x) or x < wp.float64(-1.0) or "
        "x > wp.float64(1.0),\n"
        "                    _WARP_NAN, wp.asin(x))\n"
    ),
    "_acos": (
        "@wp.func\n"
        "def _acos(x: wp.float64) -> wp.float64:\n"
        "    return wp.where(wp.isnan(x) or x < wp.float64(-1.0) or "
        "x > wp.float64(1.0),\n"
        "                    _WARP_NAN, wp.acos(x))\n"
    ),
    "_sgnbit": (
        "@wp.func\n"
        "def _sgnbit(x: wp.float64) -> wp.bool:\n"
        "    return x < wp.float64(0.0) or (x == wp.float64(0.0) and "
        "wp.float64(1.0) / x < wp.float64(0.0))\n"
    ),
    "_signbit": (
        "@wp.func\n"
        "def _signbit(x: wp.float64) -> wp.int32:\n"
        "    return wp.where(_sgnbit(x), wp.int32(-1), wp.int32(0))\n"
    ),
    # `lt_to`/`le_to`/`eq_to` are StableHLO comparisons with `compare_type =
    # TOTALORDER` (`jax._src.lax.lax`: `lt_to_p` lowers as
    # `_compare_lower_hlo("LT", True)`), i.e. IEEE-754's totalOrder rather than the
    # usual float compare. Two things differ from `<`, `<=`, `==`:
    #
    #   1. **-0.0 sorts strictly below +0.0.** Handled exactly, by `_sgnbit`.
    #   2. **NaN is ordered, not unordered** -- a positive NaN above +inf, a NEGATIVE
    #      NaN below -inf, and distinct payloads ordered among themselves.
    #
    # (2) is where these stop being exact, and the limit is named rather than hidden:
    # every NaN is treated as a single positive NaN, so a negative NaN or two distinct
    # payloads compare differently from XLA. That is not a gap in practice for the one
    # thing that produces these primitives here -- `jnp.searchsorted`, reached through
    # `jnp.interp` -- because searchsorted CANONICALISES both operands immediately
    # before the compare (`select_n(x != x, select_n(x == 0, x, 0.0), nan)`, visible in
    # the jaxpr), mapping every NaN to one literal and -0.0 to +0.0. Those `select_n`s
    # are ordinary primitives and are emitted here too, so the Warp operands reaching
    # these helpers are canonicalised in exactly the same way and the comparison is
    # exact. A `lt_to` arriving from anywhere else would not have that guarantee.
    #
    # MEASURED against `lt_to_p`/`le_to_p`/`eq_to_p` themselves, over every ordered
    # pair from {+-0, +-1, +-2.5, 1e-300, +-inf, 3.0, 7.5, 0.75}: **0 mismatches out of
    # 144** for all three, and `_signbit` likewise 0/144 against `np.signbit`. Adding
    # +-NaN to the set produces mismatches ONLY on pairs containing a NaN (25/196 for
    # `lt_to` and `le_to`, 2/196 for `eq_to`, 14/196 for `_signbit`) -- which is the
    # limitation above, stated as a number rather than a hope. Note that the sweep also
    # showed this platform's `float("nan")` carries a SET sign bit, so the negative-NaN
    # case is real, not hypothetical; it is out of reach here because Warp 1.17 cannot
    # read a float's bits at all.
    # Written as single boolean expressions rather than early returns: a `@wp.func`
    # with more than one `return` is not something to rely on across Warp versions,
    # and the algebra is short enough to read as the case analysis it is.
    "_lt_to": (
        "@wp.func\n"
        "def _lt_to(a: wp.float64, b: wp.float64) -> wp.bool:\n"
        "    return (not wp.isnan(a)) and (wp.isnan(b) or a < b\n"
        "            or (a == b and _sgnbit(a) and not _sgnbit(b)))\n"
    ),
    "_le_to": (
        "@wp.func\n"
        "def _le_to(a: wp.float64, b: wp.float64) -> wp.bool:\n"
        "    return (wp.isnan(a) and wp.isnan(b)) or ((not wp.isnan(a))\n"
        "            and (wp.isnan(b) or a < b\n"
        "                 or (a == b and (_sgnbit(a) or not _sgnbit(b)))))\n"
    ),
    "_eq_to": (
        "@wp.func\n"
        "def _eq_to(a: wp.float64, b: wp.float64) -> wp.bool:\n"
        "    return (wp.isnan(a) and wp.isnan(b)) or ((not wp.isnan(a))\n"
        "            and (not wp.isnan(b)) and a == b\n"
        "            and _sgnbit(a) == _sgnbit(b))\n"
    ),
}
"""Device functions the emitted `@wp.func`s call, keyed by name. Emitted into the
generated module only when some node's body actually names one (`funcs_source`)."""

HELPER_DEPS = {
    "_max": ("_nanc",),
    "_min": ("_nanc",),
    "_clamp": ("_nanc",),
    "_sign": ("_nanc",),
    "_asin": ("_nanc",),
    "_acos": ("_nanc",),
    "_xla_log1p": ("_dd_fma",),
    "_xla_lgamma": ("_dd_fma", "_xla_log1p"),
    "_lgamma": ("_xla_lgamma",),
    "_signbit": ("_sgnbit",),
    "_lt_to": ("_sgnbit",),
    "_le_to": ("_sgnbit",),
    "_eq_to": ("_sgnbit",),
}

HELPER_ORDER = ("_nanc", "_dd_fma", "_xla_log1p", "_xla_lgamma", "_lgamma",
                "_max", "_min", "_clamp", "_sign", "_asin", "_acos", "_sgnbit", "_signbit",
                "_lt_to", "_le_to", "_eq_to")
"""Emission order: a definition must precede its callers, and `_xla_lgamma` calls
`_xla_log1p` while sorting before it."""


def helper_closure(names) -> list[str]:
    """`names` plus everything they transitively call, in a valid emission order."""
    want, queue = set(), list(names)
    while queue:
        n = queue.pop()
        if n in want:
            continue
        want.add(n)
        queue.extend(HELPER_DEPS.get(n, ()))
    return [n for n in HELPER_ORDER if n in want]


# ---------------------------------------------------------------------------
# array-valued VarPaths: one fixed-length Warp vector per length
# ---------------------------------------------------------------------------

MAX_NODE_LINES = 60000
"""Cap on the number of statements in ONE emitted `@wp.func`.

Everything here is fully unrolled, so a node's source is the product of its jaxpr size
and its arrays' lengths -- and that product is not bounded by anything physical.
`.physics.impurity_radiation_totals` interpolates 14 species against 201 profile points
over 200-point tables, and its own binary search unrolls with it: **483,266 statements,
22 MB in one function**, measured. Warp compiles a generated module through clang,
superlinearly.

60,000 is where the three configurations' modules land at 4-5 MB and a minutes-scale
compile with `warp_init`'s default (adjoints off). It is a budget, not a boundary of
what is correct: everything under it emits and validates identically, and the cap is a
REFUSAL, named like any other, rather than a silent truncation or a switch to some
approximate emission. Raising it costs compile time and nothing else -- but note that
compile time here is not linear in the cap, and that with ADJOINTS on the same modules
cost about sixty times as much again (`warp_init`).
"""

MAX_VEC_ELEMENTS = 4096
"""Cap on an array-valued VarPath crossing a NODE BOUNDARY. Inside a node an array is
scalarised and only `MAX_ELEMENTS` applies; crossing a boundary costs one Warp vector
component per element in the caller as well, so the cap is tighter. A larger array is a
named refusal, not a slow compile."""


def warp_init(enable_backward: bool | None = None):
    """`wp.init()`, with Warp's ADJOINT code generation off unless asked for.

    Not a detail. Warp generates a reverse-mode adjoint for every `@wp.func` alongside
    the forward one, and on a fully-unrolled kernel that dominates everything else:
    measured on `stellarator_helias`'s 1.36 MB node module, on this machine,
    **~35 minutes with adjoints and 35.1 seconds without** -- a factor of about sixty,
    for byte-identical forward code. The modules this backend emits are 50x the size of
    the ones the resolver path managed, which is exactly the coverage win, so the
    adjoint cost is new and it is what decides whether a full-coverage kernel can be
    verified at all.

    **What is given up is stated rather than hidden**: a module built this way computes
    values and cannot be differentiated by `wp.Tape`, and a construct that Warp's
    forward codegen accepts but its adjoint codegen rejects would not be caught here.
    Neither affects the forward values these harnesses measure. Set
    `WARP_ENABLE_BACKWARD=1` (or pass `enable_backward=True`) to build the
    differentiable module and pay the sixty-fold compile.
    """
    import os as _os

    import warp as wp

    if enable_backward is None:
        enable_backward = _os.environ.get("WARP_ENABLE_BACKWARD", "0") not in (
            "0", "", "false", "False")
    wp.config.enable_backward = bool(enable_backward)
    wp.init()
    return wp


def vec_name(n: int) -> str:
    """The generated module's type name for a flat length-`n` float64 vector."""
    return f"vec{n}f"


def vec_decls(lengths) -> str:
    """`wp.types.vector` declarations for every length used, one per line."""
    return "\n".join(f"{vec_name(n)} = wp.types.vector(length={n}, dtype=wp.float64)"
                     for n in sorted(set(lengths)))


class _FuncEmitter:
    """Walks one node's jaxpr and produces the body of one `@wp.func`.

    Nested jaxprs (`pjit`/`closed_call`/`custom_jvp_call`/...) are **inlined**, not
    emitted as separate functions: they carry no recursion in this graph, inlining
    keeps every emitted node to exactly one Warp function, and it removes any question
    of name collisions between two nodes' identically-named inner jaxprs.
    """

    def __init__(self):
        self.lines: list[str] = []
        self._n = 0
        self.helpers: set[str] = set()
        """Which `HELPERS` entries this node's body calls."""
        self._uses: dict = {}
        """`jaxpr Var -> number of times it is consumed`, accumulated over every
        (possibly nested) jaxpr this emitter walks. Var objects are unique per jaxpr,
        so one table cannot confuse two of them."""
        self._bits: dict = {}
        """A `bitcast_convert_type` output held back for its sole `>> 63` consumer --
        see `_eqn`. Never read by anything else, by construction of `_uses`."""
        self.vecs: set[int] = set()
        """Fixed-length vector types this BODY needs (a runtime-indexed `gather`
        materialises its operand into one). `emit_node` unions these with the lengths
        the signature needs, and `module_preamble` declares them."""

    def _fresh(self) -> str:
        self._n += 1
        return f"t{self._n}"

    _TRIVIAL = re.compile(r"^(?:[A-Za-z_]\w*|wp\.(?:float64|int32)\([^()]*\)"
                          r"|True|False)$")
    """An expression that is already a name or a literal. Re-binding one to a fresh
    local is pure source bloat -- and source size is not free here: a fully unrolled
    201-point profile node emits tens of thousands of statements and Warp's own codegen
    is what pays for them (a 2.3 MB module measured at 17 minutes to compile). Every
    local this skips is an identical value under a different name; nothing in this
    emitter ever reassigns a name it has bound, and a `@wp.func` parameter is not
    assignable, so substituting the name is exactly equivalent."""

    def _materialise(self, exprs, kind, shape, vals=None) -> Value:
        """Bind each element expression to its own Warp local. Doing this at every
        equation (rather than substituting expressions into each other) keeps the
        emitted source linear in the jaxpr's size instead of exponential in its
        depth."""
        names = []
        for e in exprs:
            if self._TRIVIAL.match(e):
                names.append(e)
                continue
            n = self._fresh()
            self.lines.append(f"    {n} = {e}")
            names.append(n)
        return Value(tuple(names), kind, tuple(shape), vals)

    # -- operand access -----------------------------------------------------

    def _read(self, env, v) -> Value:
        if isinstance(v, Literal):
            _check_size(v.aval, "literal")
            return _const_value(v.val, _kind(v.aval))
        try:
            return env[v]
        except KeyError:
            raise Refusal(f"unbound jaxpr variable {v}") from None

    # -- coercions ----------------------------------------------------------

    def _to_float(self, e: str, k: str) -> str:
        if k == _F:
            return e
        if k == _I:
            return f"wp.float64({e})"
        return f"wp.where({e}, wp.float64(1.0), wp.float64(0.0))"

    def _to_int(self, e: str, k: str) -> str:
        if k == _I:
            return e
        if k == _F:
            return f"wp.int32({e})"
        return f"wp.where({e}, wp.int32(1), wp.int32(0))"

    def _coerce(self, e: str, frm: str, to: str) -> str:
        if frm == to:
            return e
        if to == _F:
            return self._to_float(e, frm)
        if to == _I:
            return self._to_int(e, frm)
        zero = "wp.float64(0.0)" if frm == _F else "wp.int32(0)"
        return f"({e} != {zero})"

    def _bcast(self, v: Value, n: int, kind: str, oshape=None) -> list[str]:
        """`v`'s elements coerced to `kind` and stretched to the result's `oshape`
        (`n` elements).

        Three cases, all of them proofs rather than conventions:
        - equal element counts: no stretching at all;
        - a size-1 operand (a bare scalar literal alongside an array operand): one
          value replicated;
        - **equal RANK with size-1 axes**: the axis map is computed from the two
          DECLARED shapes by `np.broadcast_to`, which is the same map XLA uses. A
          jax 0.11 comparison equation really does carry `bool[1,201]` against
          `bool[14,201]` (`jnp.interp`'s `searchsorted` over 14 species), so refusing
          this was refusing a shape the jaxpr states outright.

        Anything else -- an unequal rank, or an axis that is neither equal nor 1 --
        stays a refusal: THAT is where a numpy-style guess about which axis was meant
        would be plausible and wrong.
        """
        # SHAPE first, element count second, wherever the result's shape is known.
        # Equal element counts do NOT imply the same flat order -- (2,3) and (3,2) hold
        # six elements each in different places -- so an equal-count shortcut taken
        # before the shapes are compared would silently transpose. `lax` never asks for
        # that, but "never asks for it" is the sort of thing that stops being true.
        if oshape is not None and v.size != 1:
            vs, os_ = tuple(v.shape), tuple(oshape)
            if vs != os_:
                if len(vs) != len(os_) or not all(a == b or a == 1
                                                  for a, b in zip(vs, os_)):
                    raise Refusal(f"operand of shape {vs} against a result of shape "
                                  f"{os_} -- neither equal nor a size-1 stretch")
                src = np.broadcast_to(np.arange(v.size).reshape(vs), os_).reshape(-1)
                return [self._coerce(v.exprs[i], v.kind, kind) for i in src]
        if v.size == n:
            return [self._coerce(e, v.kind, kind) for e in v.exprs]
        if v.size == 1:
            return [self._coerce(v.exprs[0], v.kind, kind)] * n
        raise Refusal(f"operand of shape {v.shape} against a result of shape "
                      f"{tuple(oshape) if oshape is not None else '?'} "
                      f"({v.size} against {n} elements)")

    # -- the walk -----------------------------------------------------------

    def emit(self, jaxpr, consts, args: list[Value]) -> list[Value]:
        """Emit `jaxpr` with `jaxpr.invars` bound to `args` and `jaxpr.constvars` to
        `consts`. Returns one `Value` per outvar."""
        env: dict = {}
        if len(consts) != len(jaxpr.constvars):
            raise Refusal(f"jaxpr has {len(jaxpr.constvars)} constvars but "
                          f"{len(consts)} consts")
        for cv, cval in zip(jaxpr.constvars, consts):
            _check_size(cv.aval, "jaxpr const")
            env[cv] = _const_value(cval, _kind(cv.aval))
        if len(args) != len(jaxpr.invars):
            raise Refusal(f"jaxpr takes {len(jaxpr.invars)} arguments but "
                          f"{len(args)} were supplied")
        for iv, a in zip(jaxpr.invars, args):
            env[iv] = a
        # Consumption counts for THIS jaxpr, so `_eqn` can prove a value has exactly
        # one consumer before fusing it into that consumer.
        #
        # Saved and restored rather than accumulated: JAX reuses one inner `Jaxpr`
        # object for every call site of the same traced function, so a `jit[name=
        # signbit]` reached twice would count each of its variables twice and no fusion
        # would ever fire. (It did not, on `parabolic_profile_values`, until this.)
        uses: dict = {}
        for e in jaxpr.eqns:
            for v in e.invars:
                if not isinstance(v, Literal):
                    uses[v] = uses.get(v, 0) + 1
        for o in jaxpr.outvars:
            if not isinstance(o, Literal):
                uses[o] = uses.get(o, 0) + 1
        outer_uses, self._uses = self._uses, uses
        try:
            for eqn in jaxpr.eqns:
                self._eqn(env, eqn)
            return [self._read(env, o) for o in jaxpr.outvars]
        finally:
            self._uses = outer_uses

    def _eqn(self, env, eqn) -> None:
        name = eqn.primitive.name

        # ---- `signbit`: `bitcast_convert_type[int64] x` then `>> 63` ----------
        # XLA has no `signbit` primitive; `jnp.signbit` lowers to exactly this pair,
        # and it is the ONLY producer of `bitcast_convert_type` in this graph. A true
        # bit reinterpretation is not expressible in Warp 1.17 (it has `bit_and`/
        # `bit_or`/`bit_xor` and no cast between a float's and an int's bits), so a
        # bare `bitcast_convert_type` stays a refusal. The PAIR, however, is not a bit
        # operation at all -- an arithmetic right shift by 63 of a two's-complement
        # int64 is 0 or -1 according to the sign bit alone, and `_signbit` computes
        # that sign bit exactly (see `HELPERS`). The fusion is licensed by the use
        # count, not by the shape of the Python that produced it: it fires only when
        # the bitcast's output has exactly ONE consumer and that consumer is this
        # shift, so no other use of the bit pattern can be quietly dropped.
        if name == "bitcast_convert_type":
            a = self._read(env, eqn.invars[0])
            out = eqn.outvars[0]
            if (a.kind == _F and np.dtype(eqn.params["new_dtype"]) == np.dtype(np.int64)
                    and self._uses.get(out, 0) == 1):
                self._bits[out] = a
                return
            raise Refusal("bitcast_convert_type (no bit reinterpretation in Warp; only "
                          "the `signbit` pair `bitcast -> shift_right_arithmetic 63` "
                          "is emitted)")
        if name == "shift_right_arithmetic" and eqn.invars[0] in self._bits:
            a = self._bits.pop(eqn.invars[0])
            shift = eqn.invars[1]
            if not (isinstance(shift, Literal) and int(shift.val) == 63):
                raise Refusal("shift_right_arithmetic on a bit pattern by a shift "
                              "other than 63")
            self.helpers.add("_signbit")
            out = eqn.outvars[0]
            env[out] = self._materialise(
                [f"_signbit({e})" for e in a.exprs], _I, tuple(out.aval.shape))
            return

        if name == "scan":
            self._scan(env, eqn)
            return

        nest = _nested(eqn)
        if nest is not None:
            sub, sub_consts = nest
            outs = self.emit(sub, sub_consts, [self._read(env, v) for v in eqn.invars])
            if len(outs) != len(eqn.outvars):
                raise Refusal(f"{name}: nested jaxpr returned {len(outs)} value(s) "
                              f"for {len(eqn.outvars)} outvar(s)")
            for ov, o in zip(eqn.outvars, outs):
                env[ov] = o
            return

        for ov in eqn.outvars:
            _check_size(ov.aval, f"primitive {name!r} output")
        if len(eqn.outvars) != 1:
            raise Refusal(f"{name}: {len(eqn.outvars)} outputs -- only single-output "
                          f"primitives are emitted")
        out = eqn.outvars[0]
        okind = _kind(out.aval)
        oshape = tuple(out.aval.shape)
        args = [self._read(env, v) for v in eqn.invars]

        exprs, vals = self._expr(name, eqn, args, okind, oshape)
        env[out] = self._materialise(exprs, okind, oshape, vals)

    # -- scan ---------------------------------------------------------------

    MAX_SCAN_STEPS = 512
    """Cap on a scan's trip count. The loop is UNROLLED, so the emitted body appears
    once per step; a longer scan is a named refusal rather than a source explosion."""

    def _scan(self, env, eqn) -> None:
        """`lax.scan`, **unrolled**.

        A scan's trip count is a jaxpr parameter, not a value -- it is fixed the moment
        the node is traced, which is the same fact that lets an array be scalarised.
        Unrolling therefore introduces no approximation and no runtime control flow: it
        emits the body `length` times against the carry the previous step produced,
        which is exactly what the scan computes. `reverse` changes the ORDER the steps
        run in, never where a `ys` row is stored, so both are tracked separately.

        `while` stays refused: its trip count is a value, and nothing here can bound it.
        """
        p = eqn.params
        closed = p["jaxpr"]
        body = closed.jaxpr
        body_consts = list(closed.consts)
        length = int(p["length"])
        # jax <= 0.10 spelled the split as `num_consts`/`num_carry`; 0.11 carries it as
        # the group structure of `ft_in = (consts, carry, xs)` / `ft_out = (carry, ys)`.
        # Read whichever is present rather than assuming a version -- and then CHECK the
        # split against the equation's own arity below, so a third spelling fails loudly
        # instead of silently slicing the operands in the wrong places.
        if "num_consts" in p:
            num_consts, num_carry = int(p["num_consts"]), int(p["num_carry"])
        else:
            # `FlatTree.__len__` counts LEAVES, not groups, and `__repr__` prints the
            # groups -- so `len(ft_in)` is the operand count and `ft_in.elts` is the
            # `(consts, carry, xs)` split. Read the split from `elts`; the shape
            # cross-check below is what actually licenses it.
            ft_in, ft_out = p["ft_in"], p["ft_out"]
            gin = getattr(ft_in, "elts", ft_in)
            gout = getattr(ft_out, "elts", ft_out)
            if len(gin) != 3 or len(gout) != 2:
                raise Refusal(f"scan: unrecognised ft_in/ft_out grouping "
                              f"({len(gin)}/{len(gout)} groups)")
            num_consts, num_carry = len(gin[0]), len(gin[1])
            if (num_consts + num_carry + len(gin[2]) != len(eqn.invars)
                    or len(gout[0]) != num_carry
                    or num_carry + len(gout[1]) != len(eqn.outvars)):
                raise Refusal("scan: ft_in/ft_out groups do not account for the "
                              "equation's operands and results")
        # The body takes the consts, the carry, and ONE slice of each `xs` -- so it has
        # exactly as many parameters as the equation has operands.
        if len(body.invars) != len(eqn.invars):
            raise Refusal(f"scan: body takes {len(body.invars)} argument(s) for "
                          f"{len(eqn.invars)} operand(s)")
        # Cross-check the split against the shapes, so a scan whose grouping this reads
        # wrongly refuses instead of slicing the wrong operands: a const or a carry
        # passes into the body unchanged, and an `xs` loses a leading axis of `length`.
        # Same for the results: a carry comes out at the body's own shape, a `ys`
        # gains that leading axis. The two disagree in RANK, so the check is decisive.
        for i, (v, bv) in enumerate(zip(eqn.invars, body.invars)):
            want = tuple(bv.aval.shape) if i < num_consts + num_carry \
                else (length,) + tuple(bv.aval.shape)
            if tuple(v.aval.shape) != want:
                raise Refusal(f"scan: operand {i} has shape {tuple(v.aval.shape)} "
                              f"where the const/carry/xs split implies {want}")
        for i, (v, bv) in enumerate(zip(eqn.outvars, body.outvars)):
            want = tuple(bv.aval.shape) if i < num_carry \
                else (length,) + tuple(bv.aval.shape)
            if tuple(v.aval.shape) != want:
                raise Refusal(f"scan: result {i} has shape {tuple(v.aval.shape)} "
                              f"where the carry/ys split implies {want}")
        reverse = bool(p.get("reverse", False))
        if length > self.MAX_SCAN_STEPS:
            raise Refusal(f"scan of {length} steps (cap {self.MAX_SCAN_STEPS}) -- the "
                          f"loop is unrolled")

        args = [self._read(env, v) for v in eqn.invars]
        const_args = args[:num_consts]
        carry = list(args[num_consts:num_consts + num_carry])
        xs = args[num_consts + num_carry:]
        xs_vars = eqn.invars[num_consts + num_carry:]
        n_ys = len(eqn.outvars) - num_carry
        ys_slots: list = [[None] * length for _ in range(n_ys)]

        for step in (range(length - 1, -1, -1) if reverse else range(length)):
            sliced = []
            for a, v in zip(xs, xs_vars):
                sh = tuple(v.aval.shape)
                if not sh or sh[0] != length:
                    raise Refusal(f"scan: xs operand of shape {sh} against a trip "
                                  f"count of {length}")
                inner_shape = sh[1:]
                inner = int(np.prod(inner_shape)) if inner_shape else 1
                sliced.append(Value(
                    a.exprs[step * inner:(step + 1) * inner], a.kind, inner_shape,
                    None if a.vals is None else np.asarray(a.vals)[step]))
            outs = self.emit(body, body_consts, const_args + carry + sliced)
            if len(outs) != num_carry + n_ys:
                raise Refusal(f"scan: body returned {len(outs)} value(s) for "
                              f"{num_carry} carry + {n_ys} ys")
            carry = list(outs[:num_carry])
            for k, o in enumerate(outs[num_carry:]):
                ys_slots[k][step] = o

        for i, ov in enumerate(eqn.outvars):
            _check_size(ov.aval, "scan output")
            okind = _kind(ov.aval)
            oshape = tuple(ov.aval.shape)
            if i < num_carry:
                v = carry[i]
                env[ov] = Value(tuple(self._coerce(e, v.kind, okind) for e in v.exprs),
                                okind, oshape, v.vals)
                continue
            rows = ys_slots[i - num_carry]
            exprs = [self._coerce(e, r.kind, okind) for r in rows for e in r.exprs]
            if len(exprs) != (int(np.prod(oshape)) if oshape else 1):
                raise Refusal(f"scan: stacked ys has {len(exprs)} elements for output "
                              f"shape {oshape}")
            env[ov] = self._materialise(exprs, okind, oshape)

    # -- one primitive to a list of element expressions ---------------------

    def _expr(self, name, eqn, args, okind, oshape) -> tuple[list[str], object]:
        """`(one Warp expression per output element, compile-time values or None)`."""
        n_out = int(np.prod(oshape)) if oshape else 1

        # ---------- pure shape rearrangement (the generator does the indexing) ------
        if name in _IDENTITY_FLAT:
            if name == "reshape" and eqn.params.get("dimensions") is not None:
                a = args[0]
                perm = eqn.params["dimensions"]
                src = np.transpose(np.arange(a.size).reshape(a.shape), perm).reshape(-1)
                return ([a.exprs[i] for i in src],
                        None if a.vals is None
                        else np.transpose(a.vals, perm).reshape(oshape))
            a = args[0]
            if a.size != n_out:
                raise Refusal(f"{name}: {a.size} input elements for {n_out} output "
                              f"elements")
            return (list(a.exprs),
                    None if a.vals is None else np.asarray(a.vals).reshape(oshape))

        if name == "transpose":
            a = args[0]
            perm = tuple(eqn.params["permutation"])
            src = np.transpose(np.arange(a.size).reshape(a.shape), perm).reshape(-1)
            return ([a.exprs[i] for i in src],
                    None if a.vals is None else np.transpose(a.vals, perm))

        if name == "broadcast_in_dim":
            a = args[0]
            bdims = tuple(eqn.params["broadcast_dimensions"])
            # Build the index map by broadcasting an index array -- the same
            # arithmetic XLA does, done here once at generation time.
            idx = np.arange(a.size).reshape(a.shape)
            placed = np.expand_dims(idx, tuple(i for i in range(len(oshape))
                                               if i not in bdims))
            # A size-1 axis of the operand that broadcasts to a longer output axis is
            # stretched; any other length mismatch is a refusal.
            reps = []
            for ax in range(len(oshape)):
                have = placed.shape[ax]
                want = oshape[ax]
                if have == want:
                    reps.append(1)
                elif have == 1:
                    reps.append(want)
                else:
                    raise Refusal(f"broadcast_in_dim: axis {ax} of size {have} to "
                                  f"{want}")

            src = np.tile(placed, reps).reshape(-1)
            return ([a.exprs[i] for i in src],
                    None if a.vals is None
                    else np.tile(np.expand_dims(
                        a.vals, tuple(i for i in range(len(oshape))
                                      if i not in bdims)), reps))

        if name == "slice":
            a = args[0]
            starts = tuple(eqn.params["start_indices"])
            limits = tuple(eqn.params["limit_indices"])
            strides = eqn.params.get("strides") or (1,) * len(starts)
            idx = np.arange(a.size).reshape(a.shape)
            sel = idx[tuple(slice(s, l, st)
                            for s, l, st in zip(starts, limits, strides))]
            src = np.asarray(sel).reshape(-1)
            return ([a.exprs[i] for i in src],
                    None if a.vals is None
                    else np.asarray(a.vals)[tuple(slice(s, l, st) for s, l, st
                                                  in zip(starts, limits, strides))])

        if name == "rev":
            a = args[0]
            dims = tuple(eqn.params["dimensions"])
            idx = np.arange(a.size).reshape(a.shape)
            sel = np.flip(idx, dims)
            return ([a.exprs[i] for i in np.asarray(sel).reshape(-1)],
                    None if a.vals is None else np.flip(np.asarray(a.vals), dims))

        if name in ("concatenate", "stack"):
            # `concatenate` joins along an EXISTING axis it names `dimension`; `stack`
            # inserts a NEW one it names `axis`. They are different operations with
            # differently-spelled parameters, and reading one's parameter with the
            # other's name and a default of 0 is a silent wrong answer, not an error:
            # `jnp.stack([...], axis=1)` of six length-22 vectors then comes out
            # transposed, which is exactly the defect the per-node sweep found in
            # `.tokamak.pf_coil.waveform`. Neither parameter has a default here.
            if name == "stack":
                if "axis" not in eqn.params:
                    raise Refusal("stack without an `axis` parameter")
                axis = int(eqn.params["axis"])
                join = lambda bs: np.stack(bs, axis=axis)      # noqa: E731
                shapes = [a.shape for a in args]
            else:
                if "dimension" not in eqn.params:
                    raise Refusal("concatenate without a `dimension` parameter")
                dim = int(eqn.params["dimension"])
                join = lambda bs: np.concatenate(bs, axis=dim)  # noqa: E731
                # A rank-0 operand cannot be concatenated as-is; `lax.concatenate`
                # itself never sees one, but give each a length-1 axis so the index
                # arithmetic is uniform if one ever arrives.
                shapes = [a.shape if a.shape else (1,) for a in args]
            base, blocks, vblocks = 0, [], []
            for a, sh in zip(args, shapes):
                blocks.append(np.arange(base, base + a.size).reshape(sh))
                base += a.size
                vblocks.append(None if a.vals is None
                               else np.asarray(a.vals).reshape(sh))
            joined = join(blocks)
            if tuple(joined.shape) != tuple(oshape):
                raise Refusal(f"{name}: operands join to shape "
                              f"{tuple(joined.shape)}, not the declared {oshape}")
            sel = joined.reshape(-1)
            pool = [e for a in args for e in a.exprs]
            kinds = [k for a in args for k in [a.kind] * a.size]
            vals = (None if any(v is None for v in vblocks) else join(vblocks))
            return ([self._coerce(pool[i], kinds[i], okind) for i in sel], vals)

        if name == "pad":
            a, padv = args[0], args[1]
            cfg = eqn.params["padding_config"]
            if any(interior != 0 for _, _, interior in cfg):
                raise Refusal("pad with interior padding")
            if padv.size != 1:
                raise Refusal("pad with a non-scalar padding value")
            idx = np.full(oshape, -1, dtype=int)
            src_idx = np.arange(a.size).reshape(a.shape)
            region = tuple(slice(lo, lo + d) for (lo, _hi, _i), d
                           in zip(cfg, a.shape))
            if any(s.start < 0 for s in region):
                raise Refusal("pad with negative (trimming) padding")
            idx[region] = src_idx
            flat = idx.reshape(-1)
            fill = self._coerce(padv.exprs[0], padv.kind, okind)
            return ([fill if i < 0 else self._coerce(a.exprs[i], a.kind, okind)
                     for i in flat], None)

        if name == "iota":
            dim = eqn.params["dimension"]
            grid = np.indices(oshape)[dim].reshape(-1)
            return ([_fmt_scalar(v, okind) for v in grid.tolist()],
                    np.indices(oshape)[dim])

        # ---------- reductions -------------------------------------------------
        if name in ("reduce_sum", "reduce_prod", "reduce_max", "reduce_min",
                    "reduce_and", "reduce_or", "argmax", "argmin", "cumsum",
                    "cumlogsumexp", "cumprod", "cummax", "cummin"):
            return self._reduce(name, eqn, args, okind, oshape)

        # ---------- selection --------------------------------------------------
        if name == "select_n":
            pred, *cases = args
            vals = [self._bcast(c, n_out, okind, oshape) for c in cases]
            preds = self._bcast(pred, n_out, pred.kind, oshape)
            if pred.kind == _B:
                if len(vals) != 2:
                    raise Refusal(f"select_n: boolean predicate with {len(vals)} cases")
                return ([f"wp.where({preds[i]}, {vals[1][i]}, {vals[0][i]})"
                         for i in range(n_out)], None)
            if pred.kind != _I:
                raise Refusal(f"select_n: predicate of kind {pred.kind!r}")
            out = []
            for i in range(n_out):
                e = vals[-1][i]
                for c in range(len(vals) - 2, -1, -1):
                    e = f"wp.where({preds[i]} == wp.int32({c}), {vals[c][i]}, {e})"
                out.append(e)
            return (out, None)

        if name == "dynamic_slice":
            return (self._dynamic_slice(eqn, args, okind, oshape), None)

        if name == "gather":
            return self._gather(eqn, args, okind, oshape)

        if name == "scatter":
            return self._scatter(eqn, args, okind, oshape)

        if name in ("dynamic_update_slice", "scatter_add", "sort", "while", "cond",
                    "argsort", "top_k", "dot_general", "conv_general_dilated",
                    "cumsum_p"):
            # `cond` is refusable rather than emitted deliberately. Selecting over every
            # pure branch would be correct, and it was written and measured: it changes
            # coverage on all three configurations by exactly ZERO nodes, because every
            # `cond` in these graphs sits inside a node already refused for something
            # else (`.vacuum.vacuum_old` also contains a `while`, whose trip count is a
            # value). Emission code that no node exercises is emission code that
            # `jaxpr_validate` never checks, and this backend does not ship paths it has
            # no evidence for.
            raise Refusal(name)

        # ---------- element-wise -----------------------------------------------
        return (self._elementwise(name, eqn, args, okind, n_out, oshape), None)

    # -- reductions ---------------------------------------------------------

    def _reduce(self, name, eqn, args, okind, oshape):
        a = args[0]
        # No default: `axes = ()` would silently make a reduction the identity, which
        # is the same class of defect as reading `stack`'s axis under `concatenate`'s
        # name (see above).
        if "axes" not in eqn.params:
            raise Refusal(f"{name} without an `axes` parameter")
        axes = tuple(int(x) for x in eqn.params["axes"])
        n_out = int(np.prod(oshape)) if oshape else 1

        if name in ("cumsum", "cumprod", "cummax", "cummin", "cumlogsumexp"):
            raise Refusal(name)

        idx = np.arange(a.size).reshape(a.shape)
        # Group source elements by their destination: move the reduced axes last, then
        # every row of the reshaped array is one output element's operand list.
        keep = [ax for ax in range(len(a.shape)) if ax not in axes]
        moved = np.transpose(idx, keep + list(axes)).reshape(n_out, -1)

        if name in ("argmax", "argmin"):
            better = "<" if name == "argmax" else ">"
            out = []
            for r in range(n_out):
                grp = [self._to_float(a.exprs[i], a.kind) for i in moved[r]]
                if len(grp) == 1:
                    out.append("wp.int32(0)")
                    continue
                # Sequential scan, first-index-wins on ties -- the same tie rule
                # `lax.argmax` documents.
                best_v, best_i = grp[0], "wp.int32(0)"
                for j in range(1, len(grp)):
                    bi = self._fresh()
                    bv = self._fresh()
                    self.lines.append(
                        f"    {bi} = wp.where({best_v} {better} {grp[j]}, "
                        f"wp.int32({j}), {best_i})")
                    self.lines.append(
                        f"    {bv} = wp.where({best_v} {better} {grp[j]}, "
                        f"{grp[j]}, {best_v})")
                    best_i, best_v = bi, bv
                out.append(best_i)
            return (out, None)

        if name in ("reduce_and", "reduce_or"):
            op = " and " if name == "reduce_and" else " or "
            return ([("(" + op.join(a.exprs[i] for i in moved[r]) + ")")
                     for r in range(n_out)], None)

        if name in ("reduce_max", "reduce_min"):
            if okind == _F:
                fn = "_max" if name == "reduce_max" else "_min"
                self.helpers.add(fn)
            else:
                fn = "wp.max" if name == "reduce_max" else "wp.min"
            out = []
            for r in range(n_out):
                grp = [self._coerce(a.exprs[i], a.kind, okind) for i in moved[r]]
                e = grp[0]
                for g in grp[1:]:
                    e = f"{fn}({e}, {g})"
                out.append(e)
            return (out, None)

        # reduce_sum / reduce_prod.
        #
        # **Known and deliberate**: XLA's reduction is not a left-to-right sequential
        # accumulation beyond a handful of elements -- it splits and re-associates, so
        # a sequential expansion of a long sum is arithmetically right and BITWISE
        # different. Under the 1e-12 relative gate that is tolerable (float64
        # re-association of a well-conditioned sum moves the last bits, ~1e-16), and
        # the per-node sweep in `jaxpr_validate` reports the actual number rather than
        # assuming it. It is recorded here so nobody later reads a 1e-16 disagreement
        # on a long sum as a porting bug.
        op = " + " if name == "reduce_sum" else " * "
        out = []
        for r in range(n_out):
            grp = [self._coerce(a.exprs[i], a.kind, okind) for i in moved[r]]
            e = grp[0]
            for g in grp[1:]:
                e = f"({e}{op}{g})"
            out.append(e)
        return (out, None)

    # -- gather -------------------------------------------------------------

    def _gather_map(self, operand_shape, indices_shape, dnums, slice_sizes, out_shape):
        """The XLA gather index map, as pure integers.

        Returns `(rows, op_strides, hi)`: one `(base, startpos)` pair per FLAT output
        position -- `base[d]` is the compile-time part of the operand index on dim `d`
        (the slice offset, and the batch position for an `operand_batching_dim`), and
        `startpos[d]` is the flat position in the `start_indices` operand supplying the
        runtime start for dim `d`, or `None`. `hi[d]` is XLA's clamp bound
        `operand_shape[d] - slice_sizes[d]`.

        The operand index on dim `d` is therefore
        `base[d] + clamp(start_indices.flat[startpos[d]], 0, hi[d])`, and the flat
        source position is that dotted with `op_strides`. `_check_gather_map` proves
        this against `lax.gather` itself before a line is emitted.
        """
        rank = len(operand_shape)
        ivd = len(indices_shape) - 1        # jax fixes index_vector_dim as the last
        offset_dims = tuple(int(d) for d in dnums.offset_dims)
        collapsed = tuple(int(d) for d in dnums.collapsed_slice_dims)
        sim = tuple(int(d) for d in dnums.start_index_map)
        obd = tuple(int(d) for d in getattr(dnums, "operand_batching_dims", ()) or ())
        sibd = tuple(int(d) for d in
                     getattr(dnums, "start_indices_batching_dims", ()) or ())
        if not indices_shape or indices_shape[ivd] != len(sim):
            raise Refusal(f"gather: start_indices shape {indices_shape} does not end "
                          f"in an index vector of {len(sim)}")
        if len(obd) != len(sibd):
            raise Refusal("gather: batching dimensions do not pair up")
        offset_operand_dims = [d for d in range(rank)
                               if d not in collapsed and d not in obd]
        if len(offset_operand_dims) != len(offset_dims):
            raise Refusal(f"gather: {len(offset_dims)} offset dims for "
                          f"{len(offset_operand_dims)} uncollapsed operand dims")
        batch_out_dims = [d for d in range(len(out_shape)) if d not in offset_dims]
        idx_batch_dims = [d for d in range(len(indices_shape)) if d != ivd]
        if len(batch_out_dims) != len(idx_batch_dims):
            raise Refusal("gather: output batch rank does not match start_indices")

        def _strides(shape):
            st, acc = [0] * len(shape), 1
            for d in range(len(shape) - 1, -1, -1):
                st[d] = acc
                acc *= shape[d]
            return st

        op_strides = _strides(operand_shape)
        idx_strides = _strides(indices_shape)
        hi = [operand_shape[d] - slice_sizes[d] for d in range(rank)]

        rows = []
        for o_flat in range(int(np.prod(out_shape)) if out_shape else 1):
            oi = np.unravel_index(o_flat, out_shape) if out_shape else ()
            batch_pos = [oi[d] for d in batch_out_dims]
            off_pos = [oi[d] for d in offset_dims]
            base = [0] * rank
            startpos: list = [None] * rank
            for j, d in enumerate(offset_operand_dims):
                base[d] = int(off_pos[j])
            for j, d in enumerate(obd):
                base[d] = int(batch_pos[idx_batch_dims.index(sibd[j])])
            for k, d in enumerate(sim):
                full = [0] * len(indices_shape)
                for j, sd in enumerate(idx_batch_dims):
                    full[sd] = int(batch_pos[j])
                full[ivd] = k
                startpos[d] = int(sum(full[i] * idx_strides[i]
                                      for i in range(len(indices_shape))))
            rows.append((base, startpos))
        return rows, op_strides, hi

    @staticmethod
    def _apply_gather_map(rows, op_strides, hi, idx_flat):
        """`rows` applied to one concrete flattened `start_indices` array."""
        out = []
        for base, startpos in rows:
            src = 0
            for d, (b, sp) in enumerate(zip(base, startpos)):
                v = b if sp is None else b + min(max(int(idx_flat[sp]), 0), hi[d])
                src += v * op_strides[d]
            out.append(src)
        return out

    def _check_gather_map(self, rows, op_strides, hi, operand_shape, indices_shape,
                          dnums, slice_sizes, mode, out_shape):
        """Prove the derived map by DIFFERENTIAL TEST against `lax.gather` itself.

        The map above is a transcription of XLA's gather semantics, and a transcription
        is exactly the kind of thing that is plausible and wrong. So it is not trusted:
        the gather is run for real, on an operand whose value at every position IS that
        position, over several random in-bounds index draws (plus deliberately
        out-of-range ones under `CLIP`, where clamping is the defined behaviour). Any
        disagreement is a named refusal, not a warning.
        """
        from jax import lax

        n_op = int(np.prod(operand_shape))
        probe = jnp.arange(n_op, dtype=jnp.float64).reshape(operand_shape)
        rng = np.random.default_rng(0xC0FFEE)
        ivd = len(indices_shape) - 1
        draws = []
        for _ in range(4):
            idx = np.zeros(indices_shape, dtype=np.int64)
            for k, d in enumerate(tuple(int(x) for x in dnums.start_index_map)):
                sl = [slice(None)] * len(indices_shape)
                sl[ivd] = k
                idx[tuple(sl)] = rng.integers(0, hi[d] + 1,
                                              size=idx[tuple(sl)].shape)
            draws.append(idx)
        if str(mode).endswith("CLIP"):
            wild = draws[0].copy()
            wild[...] = 10 ** 6
            draws.extend((wild, -wild))
        for idx in draws:
            got = np.asarray(lax.gather(
                probe, jnp.asarray(idx, dtype=jnp.int32), dnums,
                tuple(slice_sizes), mode=mode)).reshape(-1)
            want = self._apply_gather_map(rows, op_strides, hi, idx.reshape(-1))
            if got.shape[0] != len(want) or np.any(got != np.asarray(want,
                                                                    dtype=float)):
                raise Refusal(
                    "gather: the derived index map disagrees with lax.gather at "
                    f"dimension_numbers={dnums}, slice_sizes={tuple(slice_sizes)}, "
                    f"mode={mode}")

    def _vec_local(self, v: Value) -> str:
        """Materialise `v` into one `vec{n}f` local and return its identifier -- what a
        runtime index needs, since a scalarised array is a set of unrelated locals and
        cannot be subscripted."""
        n = v.size
        if n > MAX_VEC_ELEMENTS:
            raise Refusal(f"runtime-indexed operand of {n} elements "
                          f"(cap {MAX_VEC_ELEMENTS})")
        if v.kind != _F:
            raise Refusal(f"runtime-indexed operand of kind {v.kind!r} -- only float64 "
                          f"vectors are materialised")
        self.vecs.add(n)
        name = self._fresh()
        self.lines.append(f"    {name} = {vec_name(n)}()")
        for i, e in enumerate(v.exprs):
            self.lines.append(f"    {name}[{i}] = {e}")
        return name

    def _gather(self, eqn, args, okind, oshape):
        """`lax.gather`, both statically and dynamically indexed.

        Static indices are resolved by the generator into a plain re-selection of the
        operand's locals -- no runtime work at all. A runtime index needs something
        subscriptable, which a scalarised array is not, so the operand is materialised
        into a `vec{n}f` and indexed natively (Warp indexes a vector with a runtime
        `int32`). `FILL_OR_DROP` is refused: its out-of-bounds behaviour is a fill
        value, not a clamp, and nothing here can prove the indices are in bounds.
        """
        if len(args) != 2:
            raise Refusal(f"gather: {len(args)} operands")
        operand, indices = args
        dnums = eqn.params["dimension_numbers"]
        slice_sizes = tuple(int(s) for s in eqn.params["slice_sizes"])
        mode = eqn.params.get("mode")
        if "FILL_OR_DROP" in str(mode):
            raise Refusal("gather in FILL_OR_DROP mode (out-of-bounds fills rather "
                          "than clamps, and in-boundness is not provable here)")
        operand_shape = tuple(eqn.invars[0].aval.shape)
        indices_shape = tuple(eqn.invars[1].aval.shape)
        rows, op_strides, hi = self._gather_map(
            operand_shape, indices_shape, dnums, slice_sizes, oshape)
        self._check_gather_map(rows, op_strides, hi, operand_shape, indices_shape,
                               dnums, slice_sizes, mode, oshape)

        if indices.vals is not None:
            src = self._apply_gather_map(rows, op_strides, hi,
                                         np.asarray(indices.vals).reshape(-1))
            exprs = [self._coerce(operand.exprs[s], operand.kind, okind) for s in src]
            vals = (None if operand.vals is None else
                    np.asarray(operand.vals).reshape(-1)[np.asarray(src)]
                    .reshape(oshape))
            return exprs, vals

        if okind != _F:
            raise Refusal(f"gather with a runtime index and a {okind!r} result")
        vec = self._vec_local(operand)
        exprs = []
        for base, startpos in rows:
            const = sum(base[d] * op_strides[d] for d in range(len(base)))
            terms = [f"wp.int32({const})"]
            for d, sp in enumerate(startpos):
                if sp is None:
                    continue
                e = self._coerce(indices.exprs[sp], indices.kind, _I)
                terms.append(f"wp.int32({op_strides[d]}) * wp.min(wp.max({e}, "
                             f"wp.int32(0)), wp.int32({hi[d]}))")
            exprs.append(f"{vec}[{' + '.join(terms)}]")
        return exprs, None

    # -- scatter ------------------------------------------------------------

    MAX_SCATTER_PAIRS = 262144
    """Cap on `n_updates * n_operand_elements` for a runtime-indexed scatter, which is
    emitted as one `wp.where` per (update, destination) pair."""

    def _scatter_map(self, operand_shape, indices_shape, updates_shape, dnums):
        """The XLA scatter index map, as pure integers -- the mirror of `_gather_map`.

        Returns `(rows, op_strides, hi)`: one `(base, startpos)` pair per FLAT update
        element, with the same meaning as in `_gather_map`, and `hi[d]` the largest
        legal window start on operand dim `d`. A start outside `[0, hi[d]]` means the
        update is DROPPED (`FILL_OR_DROP`) or clamped (`CLIP`); which of the two is the
        caller's business.
        """
        rank = len(operand_shape)
        ivd = len(indices_shape) - 1
        uwd = tuple(int(d) for d in dnums.update_window_dims)
        iwd = tuple(int(d) for d in dnums.inserted_window_dims)
        sdod = tuple(int(d) for d in dnums.scatter_dims_to_operand_dims)
        obd = tuple(int(d) for d in getattr(dnums, "operand_batching_dims", ()) or ())
        sibd = tuple(int(d) for d in
                     getattr(dnums, "scatter_indices_batching_dims", ()) or ())
        if not indices_shape or indices_shape[ivd] != len(sdod):
            raise Refusal(f"scatter: scatter_indices shape {indices_shape} does not "
                          f"end in an index vector of {len(sdod)}")
        window_operand_dims = [d for d in range(rank)
                               if d not in iwd and d not in obd]
        if len(window_operand_dims) != len(uwd):
            raise Refusal("scatter: update_window_dims do not match the operand's "
                          "un-inserted dimensions")
        batch_upd_dims = [d for d in range(len(updates_shape)) if d not in uwd]
        idx_batch_dims = [d for d in range(len(indices_shape)) if d != ivd]
        if len(batch_upd_dims) != len(idx_batch_dims):
            raise Refusal("scatter: update batch rank does not match scatter_indices")

        def _strides(shape):
            st, acc = [0] * len(shape), 1
            for d in range(len(shape) - 1, -1, -1):
                st[d] = acc
                acc *= shape[d]
            return st

        op_strides = _strides(operand_shape)
        window = [1] * rank
        for j, d in enumerate(window_operand_dims):
            window[d] = updates_shape[uwd[j]]
        hi = [operand_shape[d] - window[d] for d in range(rank)]

        rows = []
        for u_flat in range(int(np.prod(updates_shape)) if updates_shape else 1):
            ui = np.unravel_index(u_flat, updates_shape) if updates_shape else ()
            batch_pos = [ui[d] for d in batch_upd_dims]
            base = [0] * rank
            startpos: list = [None] * rank
            for j, d in enumerate(window_operand_dims):
                base[d] = int(ui[uwd[j]])
            for j, d in enumerate(obd):
                base[d] = int(batch_pos[idx_batch_dims.index(sibd[j])])
            for k, d in enumerate(sdod):
                full = [0] * len(indices_shape)
                for j, sd in enumerate(idx_batch_dims):
                    full[sd] = int(batch_pos[j])
                full[ivd] = k
                idx_strides = _strides(indices_shape)
                startpos[d] = int(sum(full[i] * idx_strides[i]
                                      for i in range(len(indices_shape))))
            rows.append((base, startpos))
        return rows, op_strides, hi

    @staticmethod
    def _apply_scatter_map(rows, op_strides, hi, idx_flat, clip: bool):
        """`(target_flat, valid)` per update element, for one concrete index array."""
        out = []
        for base, startpos in rows:
            tgt, valid = 0, True
            for d, (b, sp) in enumerate(zip(base, startpos)):
                if sp is None:
                    tgt += b * op_strides[d]
                    continue
                s = int(idx_flat[sp])
                if clip:
                    s = min(max(s, 0), hi[d])
                elif not (0 <= s <= hi[d]):
                    valid = False
                tgt += (b + s) * op_strides[d]
            out.append((tgt, valid))
        return out

    def _check_scatter_map(self, rows, op_strides, hi, operand_shape, indices_shape,
                           updates_shape, dnums, mode, clip):
        """Prove the derived map against `lax.scatter` itself, exactly as
        `_check_gather_map` does -- including deliberately out-of-range draws, where
        drop-versus-clamp is the whole question."""
        from jax import lax

        n_up = int(np.prod(updates_shape)) if updates_shape else 1
        operand = jnp.full(operand_shape, -1.0, dtype=jnp.float64)
        updates = jnp.arange(n_up, dtype=jnp.float64).reshape(updates_shape)
        rng = np.random.default_rng(0x5CA77E)
        ivd = len(indices_shape) - 1
        draws = []
        for _ in range(4):
            idx = np.zeros(indices_shape, dtype=np.int64)
            for k, d in enumerate(tuple(int(x) for x in
                                        dnums.scatter_dims_to_operand_dims)):
                sl = [slice(None)] * len(indices_shape)
                sl[ivd] = k
                idx[tuple(sl)] = rng.integers(0, hi[d] + 1, size=idx[tuple(sl)].shape)
            draws.append(idx)
        wild = np.zeros(indices_shape, dtype=np.int64) + 10 ** 6
        draws.extend((wild, -wild))
        for idx in draws:
            got = np.asarray(lax.scatter(
                operand, jnp.asarray(idx, dtype=jnp.int32), updates, dnums,
                mode=mode)).reshape(-1)
            want = np.asarray(operand).reshape(-1).copy()
            for u, (tgt, valid) in enumerate(
                    self._apply_scatter_map(rows, op_strides, hi, idx.reshape(-1),
                                            clip)):
                if valid:
                    want[tgt] = float(u)
            if got.shape[0] != want.shape[0] or np.any(got != want):
                raise Refusal(
                    "scatter: the derived index map disagrees with lax.scatter at "
                    f"dimension_numbers={dnums}, mode={mode}")

    def _scatter(self, eqn, args, okind, oshape):
        """`lax.scatter` (overwrite), for `unique_indices=True` only.

        With unique indices every destination is written at most once, so the result is
        a per-element choice between the operand's value and one update's -- no
        accumulation, and no question about the order two writes to one cell would
        take. `scatter_add` and a non-unique `scatter` are refused rather than given an
        order this backend would be inventing.

        Static indices resolve entirely in the generator. A runtime index becomes one
        `wp.where` per (update, destination) pair, guarded by the same in-bounds test
        XLA uses to drop an out-of-range update -- which is why the flat target alone
        is not enough: an out-of-range component can alias a legal flat position.
        """
        if len(args) != 3:
            raise Refusal(f"scatter: {len(args)} operands")
        if not eqn.params.get("unique_indices"):
            raise Refusal("scatter without unique_indices (two updates to one cell "
                          "would need a write order this backend does not define)")
        if eqn.params.get("update_jaxpr") is not None:
            raise Refusal("scatter with a combining function")
        operand, indices, updates = args
        dnums = eqn.params["dimension_numbers"]
        mode = eqn.params.get("mode")
        clip = "CLIP" in str(mode)
        if not clip and "FILL_OR_DROP" not in str(mode) and \
                "PROMISE_IN_BOUNDS" not in str(mode):
            raise Refusal(f"scatter in mode {mode}")
        operand_shape = tuple(eqn.invars[0].aval.shape)
        indices_shape = tuple(eqn.invars[1].aval.shape)
        updates_shape = tuple(eqn.invars[2].aval.shape)
        rows, op_strides, hi = self._scatter_map(
            operand_shape, indices_shape, updates_shape, dnums)
        self._check_scatter_map(rows, op_strides, hi, operand_shape, indices_shape,
                                updates_shape, dnums, mode, clip)

        n_out = operand.size
        out = [self._coerce(e, operand.kind, okind) for e in operand.exprs]
        if indices.vals is not None:
            for u, (tgt, valid) in enumerate(
                    self._apply_scatter_map(rows, op_strides, hi,
                                            np.asarray(indices.vals).reshape(-1),
                                            clip)):
                if valid:
                    out[tgt] = self._coerce(updates.exprs[u], updates.kind, okind)
            vals = None
            if operand.vals is not None and updates.vals is not None:
                vals = np.asarray(operand.vals, dtype=float).reshape(-1).copy()
                for u, (tgt, valid) in enumerate(
                        self._apply_scatter_map(rows, op_strides, hi,
                                                np.asarray(indices.vals).reshape(-1),
                                                clip)):
                    if valid:
                        vals[tgt] = np.asarray(updates.vals).reshape(-1)[u]
                vals = vals.reshape(oshape)
            return out, vals

        if len(rows) * n_out > self.MAX_SCATTER_PAIRS:
            raise Refusal(f"scatter of {len(rows)} updates into {n_out} elements "
                          f"(cap {self.MAX_SCATTER_PAIRS} pairs)")
        for u, (base, startpos) in enumerate(rows):
            terms = [f"wp.int32({sum(base[d] * op_strides[d] for d in range(len(base)))})"]
            guards = []
            for d, sp in enumerate(startpos):
                if sp is None:
                    continue
                e = self._coerce(indices.exprs[sp], indices.kind, _I)
                s = self._fresh()
                self.lines.append(f"    {s} = {e}")
                if clip:
                    terms.append(f"wp.int32({op_strides[d]}) * wp.min(wp.max({s}, "
                                 f"wp.int32(0)), wp.int32({hi[d]}))")
                else:
                    terms.append(f"wp.int32({op_strides[d]}) * {s}")
                    guards.append(f"({s} >= wp.int32(0) and {s} <= wp.int32({hi[d]}))")
            tgt = self._fresh()
            self.lines.append(f"    {tgt} = {' + '.join(terms)}")
            # Bind the update value and the in-bounds test ONCE: they are the same for
            # every destination, and this chain is `n_out` statements long.
            upd = self._coerce(updates.exprs[u], updates.kind, okind)
            if not self._TRIVIAL.match(upd):
                nm = self._fresh()
                self.lines.append(f"    {nm} = {upd}")
                upd = nm
            ok = None
            if guards:
                ok = self._fresh()
                self.lines.append(f"    {ok} = " + " and ".join(guards))
            for j in range(n_out):
                cond = f"{tgt} == wp.int32({j})"
                if ok is not None:
                    cond = f"{ok} and {cond}"
                out[j] = f"wp.where({cond}, {upd}, {out[j]})"
            # Bind the round's results so the `wp.where` chain stays linear in the
            # number of pairs rather than nesting one expression inside the next.
            bound = self._materialise(out, okind, (n_out,))
            out = list(bound.exprs)
        return out, None

    # -- dynamic_slice ------------------------------------------------------

    def _dynamic_slice(self, eqn, args, okind, oshape):
        """`dynamic_slice(operand, *start_indices)`.

        Two cases, both exact:
        - every start index is compile-time known -> the generator does the indexing
          and the output is a plain re-selection of the operand's locals;
        - a start index is a runtime value -> emit an exhaustive `wp.where` chain over
          every legal start, with XLA's own clamp (`start` is clamped into
          `[0, dim - slice_size]`) applied. The chain is exact because the operand's
          shape is fixed at trace time, so "every legal start" is a finite, known set.
        """
        operand, starts = args[0], args[1:]
        sizes = tuple(eqn.params["slice_sizes"])
        if len(starts) != len(operand.shape):
            raise Refusal("dynamic_slice: start-index count does not match rank")
        idx = np.arange(operand.size).reshape(operand.shape)

        known = [s.vals is not None and np.asarray(s.vals).size == 1 for s in starts]
        if all(known):
            base = []
            for ax, s in enumerate(starts):
                v = int(np.asarray(s.vals).reshape(()))
                base.append(int(np.clip(v, 0, operand.shape[ax] - sizes[ax])))
            sel = idx[tuple(slice(b, b + d) for b, d in zip(base, sizes))]
            return [self._coerce(operand.exprs[i], operand.kind, okind)
                    for i in np.asarray(sel).reshape(-1)]

        # Runtime start(s): enumerate the legal start tuples.
        ranges = [range(0, operand.shape[ax] - sizes[ax] + 1)
                  for ax in range(len(sizes))]
        n_cases = int(np.prod([len(r) for r in ranges]))
        if n_cases > 512:
            raise Refusal(f"dynamic_slice with a runtime index and {n_cases} legal "
                          f"start positions (cap 512)")
        clamped = []
        for ax, s in enumerate(starts):
            hi = operand.shape[ax] - sizes[ax]
            c = self._fresh()
            self.lines.append(
                f"    {c} = wp.clamp({self._to_int(s.exprs[0], s.kind)}, "
                f"wp.int32(0), wp.int32({hi}))")
            clamped.append(c)
        import itertools

        out = []
        n_out = int(np.prod(oshape)) if oshape else 1
        for k in range(n_out):
            expr = None
            for combo in itertools.product(*ranges):
                sel = idx[tuple(slice(b, b + d) for b, d in zip(combo, sizes))]
                src = operand.exprs[int(np.asarray(sel).reshape(-1)[k])]
                src = self._coerce(src, operand.kind, okind)
                if expr is None:
                    expr = src           # innermost fall-through
                    continue
                cond = " and ".join(f"{clamped[ax]} == wp.int32({combo[ax]})"
                                    for ax in range(len(combo)))
                expr = f"wp.where({cond}, {src}, {expr})"
            out.append(expr)
        return out

    # -- element-wise -------------------------------------------------------

    def _elementwise(self, name, eqn, args, okind, n_out, oshape=None) -> list[str]:
        n = len(args)

        if name == "convert_element_type":
            a = args[0]
            return [self._coerce(e, a.kind, okind) for e in self._raw(a, n_out)]

        if name == "reduce_precision":
            raise Refusal("reduce_precision (would silently change rounding)")

        if name in _BINARY_OP:
            if n != 2:
                raise Refusal(f"{name}: expected 2 arguments, got {n}")
            if okind == _B:
                raise Refusal(f"{name}: boolean result")
            L = self._bcast(args[0], n_out, okind, oshape)
            R = self._bcast(args[1], n_out, okind, oshape)
            op = _BINARY_OP[name]
            return [f"({L[i]} {op} {R[i]})" for i in range(n_out)]

        if name == "neg":
            A = self._bcast(args[0], n_out, okind, oshape)
            return [f"(-{a})" for a in A]

        if name == "abs":
            A = self._bcast(args[0], n_out, okind, oshape)
            return [f"wp.abs({a})" for a in A]

        if name in ("max", "min"):
            L = self._bcast(args[0], n_out, okind, oshape)
            R = self._bcast(args[1], n_out, okind, oshape)
            if okind != _F:
                # Integers and booleans have no NaN, so Warp's builtin IS XLA's.
                return [f"wp.{name}({L[i]}, {R[i]})" for i in range(n_out)]
            fn = "_max" if name == "max" else "_min"
            self.helpers.add(fn)
            return [f"{fn}({L[i]}, {R[i]})" for i in range(n_out)]

        if name == "pow":
            L = self._bcast(args[0], n_out, _F, oshape)
            R = self._bcast(args[1], n_out, _F, oshape)
            return [f"wp.pow({L[i]}, {R[i]})" for i in range(n_out)]

        if name == "integer_pow":
            A = self._bcast(args[0], n_out, _F, oshape)
            y = int(eqn.params["y"])
            return [self._integer_pow(a, y) for a in A]

        if name in ("rem", "mod"):
            L = self._bcast(args[0], n_out, okind, oshape)
            R = self._bcast(args[1], n_out, okind, oshape)
            if okind == _F:
                return [f"wp.mod({L[i]}, {R[i]})" for i in range(n_out)]
            return [f"({L[i]} % {R[i]})" for i in range(n_out)]

        if name == "atan2":
            L = self._bcast(args[0], n_out, _F, oshape)
            R = self._bcast(args[1], n_out, _F, oshape)
            return [f"wp.atan2({L[i]}, {R[i]})" for i in range(n_out)]

        if name == "nextafter":
            raise Refusal("nextafter")

        if name == "clamp":
            LO = self._bcast(args[0], n_out, _F, oshape)
            X = self._bcast(args[1], n_out, _F, oshape)
            HI = self._bcast(args[2], n_out, _F, oshape)
            self.helpers.add("_clamp")
            return [f"_clamp({X[i]}, {LO[i]}, {HI[i]})" for i in range(n_out)]

        if name == "square":
            A = self._bcast(args[0], n_out, _F, oshape)
            return [f"({a} * {a})" for a in A]

        if name == "rsqrt":
            A = self._bcast(args[0], n_out, _F, oshape)
            return [f"(wp.float64(1.0) / wp.sqrt({a}))" for a in A]

        if name == "logistic":
            A = self._bcast(args[0], n_out, _F, oshape)
            return [f"(wp.float64(1.0) / (wp.float64(1.0) + wp.exp(-{a})))" for a in A]

        if name in ("asin", "acos"):
            # Warp clamps these into [-1, 1]; XLA does not -- see `HELPERS`.
            helper = "_asin" if name == "asin" else "_acos"
            self.helpers.add(helper)
            A = self._bcast(args[0], n_out, _F, oshape)
            return [f"{helper}({a})" for a in A]

        if name == "sign":
            # `wp.sign` is not `lax.sign` at zero -- see `HELPERS`.
            self.helpers.add("_sign")
            A = self._bcast(args[0], n_out, _F, oshape)
            return [f"_sign({a})" for a in A]

        if name == "log1p":
            # `_xla_log1p`, not `wp.log(1+x)`: `lax.log1p` is a different function
            # near zero AND a different function from libm's `log1p` (~20 % of
            # arguments differ by 1 ulp). See `HELPERS`.
            self.helpers.add("_xla_log1p")
            A = self._bcast(args[0], n_out, _F, oshape)
            return [f"_xla_log1p({a})" for a in A]

        if name == "lgamma":
            # `_lgamma` guards x < 0.5 with a NaN rather than evaluating the
            # unreflected Lanczos sum there -- a named runtime refusal, see `HELPERS`.
            self.helpers.add("_lgamma")
            A = self._bcast(args[0], n_out, _F, oshape)
            return [f"_lgamma({a})" for a in A]

        if name == "expm1":
            raise Refusal("expm1 (Warp has no expm1, and exp(x)-1 is a different "
                          "function near zero)")

        if name in _UNARY:
            if n != 1:
                raise Refusal(f"{name}: expected 1 argument, got {n}")
            A = self._bcast(args[0], n_out, _F, oshape)
            return [f"{_UNARY[name]}({a})" for a in A]

        if name in _TOTAL_ORDER_CMP:
            # IEEE-754 totalOrder comparison -- see `HELPERS`. Float operands only:
            # for an integer operand XLA's TOTALORDER is just the signed compare, but
            # nothing in this graph produces one, so it refuses rather than assumes.
            if args[0].kind != _F or args[1].kind != _F:
                raise Refusal(f"{name}: totalOrder comparison of non-float operands")
            helper = _TOTAL_ORDER_CMP[name]
            self.helpers.add(helper)
            L = self._bcast(args[0], n_out, _F, oshape)
            R = self._bcast(args[1], n_out, _F, oshape)
            return [f"{helper}({L[i]}, {R[i]})" for i in range(n_out)]

        if name in _CMP_OP:
            k = _F if _F in (args[0].kind, args[1].kind) else \
                (_I if _I in (args[0].kind, args[1].kind) else _B)
            L = self._bcast(args[0], n_out, k, oshape)
            R = self._bcast(args[1], n_out, k, oshape)
            op = _CMP_OP[name]
            return [f"({L[i]} {op} {R[i]})" for i in range(n_out)]

        if name in _BOOL_OP:
            if args[0].kind != _B or args[1].kind != _B:
                raise Refusal(f"{name}: bitwise {name} on non-boolean operands")
            L = self._bcast(args[0], n_out, _B, oshape)
            R = self._bcast(args[1], n_out, _B, oshape)
            return [f"({L[i]} {_BOOL_OP[name]} {R[i]})" for i in range(n_out)]

        if name == "not":
            if args[0].kind != _B:
                raise Refusal("not: bitwise not on a non-boolean operand")
            return [f"(not {a})" for a in self._bcast(args[0], n_out, _B, oshape)]

        if name == "xor":
            if args[0].kind != _B or args[1].kind != _B:
                raise Refusal("xor: bitwise xor on non-boolean operands")
            L = self._bcast(args[0], n_out, _B, oshape)
            R = self._bcast(args[1], n_out, _B, oshape)
            return [f"({L[i]} != {R[i]})" for i in range(n_out)]

        if name == "is_finite":
            return [f"wp.isfinite({a})" for a in self._bcast(args[0], n_out, _F, oshape)]

        raise Refusal(name)

    def _raw(self, v: Value, n_out: int) -> list[str]:
        if v.size == n_out:
            return list(v.exprs)
        if v.size == 1:
            return list(v.exprs) * n_out
        raise Refusal(f"operand of {v.size} elements against a result of {n_out}")

    def _integer_pow(self, base: str, y: int) -> str:
        """JAX's own expansion (`lax._integer_pow`): binary exponentiation by repeated
        `mul`, and `1/acc` for a negative exponent. Emitted rather than `wp.pow` so
        `x**2` is `x*x` on both sides -- `wp.pow(x, 2.0)` is a different function and
        rounds differently."""
        if y == 0:
            return "wp.float64(1.0)"
        recip = y < 0
        y = abs(y)
        if y > 64:
            raise Refusal(f"integer_pow with |y| = {y} > 64")
        cur = self._fresh()
        self.lines.append(f"    {cur} = {base}")
        acc = None
        while y > 0:
            if y & 1:
                if acc is None:
                    acc = cur
                else:
                    nxt = self._fresh()
                    self.lines.append(f"    {nxt} = ({acc} * {cur})")
                    acc = nxt
            y >>= 1
            if y > 0:
                nxt = self._fresh()
                self.lines.append(f"    {nxt} = ({cur} * {cur})")
                cur = nxt
        return f"(wp.float64(1.0) / {acc})" if recip else acc


# ---------------------------------------------------------------------------
# one node -> one @wp.func
# ---------------------------------------------------------------------------


def trace_node(fn, values, force_float: bool = True) -> ClosedJaxpr:
    """`jax.make_jaxpr` of `fn` at `values`.

    `force_float` casts every argument to float64 -- the dtype the generated kernel
    will actually pass. Casting BEFORE the trace (rather than after) means the jaxpr
    that is emitted is the jaxpr that will run.

    It is not always possible: a read that is genuinely an integer *index* into an
    array (`jnp.take`, a species selector) rejects a float64 tracer outright with
    "Indexer must have integer or boolean type". `emit_node` falls back to the natural
    dtypes for those, and then re-derives the integer inside the emitted function from
    the float64 kernel column -- which is exact, since the column carries an integer
    value.
    """
    vals = [jnp.asarray(v, dtype=jnp.float64) if force_float else jnp.asarray(v)
            for v in values]
    return jax.make_jaxpr(lambda *a: fn(*a))(*vals)


@dataclasses.dataclass(frozen=True)
class EmitInfo:
    """Everything about one emitted node that its callers (`emit.py`'s kernel,
    `jaxpr_validate`'s per-node kernel) need in order to bind it."""

    source: str
    n_returns: int
    n_eqns: int
    input_sizes: tuple[int, ...]
    """Flat element count per parameter, in `defn.reads` order. `1` is a plain
    `wp.float64` column; `n > 1` is a `vec{n}f`."""
    output_sizes: tuple[int, ...]
    """Flat element count per return, in `defn.owns` order. Same convention."""
    helpers: tuple[str, ...]
    vec_lengths: tuple[int, ...]
    """Every `n > 1` this node's signature mentions -- the `wp.types.vector`
    declarations the generated module must carry."""


def emit_node(fn, func_name: str, values, n_outputs: int) -> EmitInfo:
    """The `@wp.func` text for one node, and how to bind it.

    Raises `Refusal` (naming the primitive or shape) rather than emitting anything it
    cannot emit correctly.

    **Array-valued reads and owned values cross the node boundary as one fixed-length
    Warp vector each** (`wp.types.vector(length=n, dtype=wp.float64)`), flat and
    row-major. The shape is known at trace time, which is what makes this possible at
    all; inside the body nothing changes, because an array was always scalarised into
    one expression per element.

    The alternative -- flattening each array into N float64 columns -- was rejected on
    three counts. It puts **610 parameters and 815 return values** on
    `.physics.fusion_rates` alone (three 201-point profiles in, four out); it spends the
    one thing the kernel ABI is actually short of, columns, on values that never leave
    the kernel; and it gives nothing to index. That last point turned out to decide it:
    a scalarised array is a set of unrelated locals and cannot be subscripted, so a
    runtime-indexed `gather` -- `jnp.interp`, and with it every impurity-radiation and
    critical-surface lookup -- has nowhere to go. A Warp vector indexes natively with a
    runtime `int32`, which is what `_gather` uses. It also costs one identifier in the
    caller regardless of length, so a wrong length is a compile error rather than a
    silent permutation.
    """
    try:
        closed = trace_node(fn, values, force_float=True)
    except Exception:
        # Not a tracing failure to report: some reads are genuinely integer-typed
        # (array indices), and JAX refuses a float64 tracer where an index is
        # required. Re-trace at the values' own dtypes. If THAT fails too, the
        # exception is real and propagates.
        closed = trace_node(fn, values, force_float=False)
    jaxpr, consts = closed.jaxpr, list(closed.consts)

    em = _FuncEmitter()
    params, args, in_sizes, vec_lengths = [], [], [], set()
    for i, iv in enumerate(jaxpr.invars):
        shape = tuple(iv.aval.shape)
        n = int(np.prod(shape)) if shape else 1
        p = f"a{i}"
        k = _kind(iv.aval)
        in_sizes.append(n)
        if n != 1:
            if n > MAX_VEC_ELEMENTS:
                raise Refusal(f"parameter {i} is array-valued with {n} elements, "
                              f"shape {shape} (cap {MAX_VEC_ELEMENTS})")
            if k != _F:
                # A vector parameter is float64 by ABI; an integer-typed array read
                # would need a per-element cast whose exactness depends on values this
                # backend has not proved anything about. Refused rather than cast.
                raise Refusal(f"parameter {i} is an array of kind {k!r}, shape "
                              f"{shape} -- only float64 arrays cross a node boundary")
            vec_lengths.add(n)
            params.append(f"{p}: {vec_name(n)}")
            args.append(Value(tuple(f"{p}[{j}]" for j in range(n)), _F, shape))
            continue
        # Every SCALAR parameter is `wp.float64` regardless of the traced kind: that is
        # the kernel's ABI (`emit.py` binds every scalar argument from a float64
        # column, and Warp does not promote). A parameter the trace typed as an integer
        # is converted back inside the body -- exact, because the column carries an
        # integral value.
        params.append(f"{p}: wp.float64")
        if k == _F:
            args.append(Value((p,), _F, shape))
        elif k == _I:
            args.append(Value((f"wp.int32({p})",), _I, shape))
        else:
            args.append(Value((f"({p} != wp.float64(0.0))",), _B, shape))

    outs = em.emit(jaxpr, consts, args)
    if len(em.lines) > MAX_NODE_LINES:
        raise Refusal(f"{len(em.lines)} statements when fully unrolled "
                      f"(cap {MAX_NODE_LINES}) -- see MAX_NODE_LINES")
    if len(outs) != n_outputs:
        raise Refusal(f"traced to {len(outs)} output(s) but the node owns {n_outputs}")

    ret, out_sizes, ret_types = [], [], []
    for j, o in enumerate(outs):
        if o.size == 1:
            ret.append(em._to_float(o.exprs[0], o.kind))
            ret_types.append("wp.float64")
            out_sizes.append(1)
            continue
        if o.size > MAX_VEC_ELEMENTS:
            raise Refusal(f"output {j} is array-valued with {o.size} elements, shape "
                          f"{o.shape} (cap {MAX_VEC_ELEMENTS})")
        n = o.size
        vec_lengths.add(n)
        name = f"o{j}"
        em.lines.append(f"    {name} = {vec_name(n)}()")
        for e_i, e in enumerate(o.exprs):
            em.lines.append(f"    {name}[{e_i}] = {em._to_float(e, o.kind)}")
        ret.append(name)
        ret_types.append(vec_name(n))
        out_sizes.append(n)

    n_eqns = len(em.lines)

    head = f"@wp.func\ndef {func_name}({', '.join(params)})"
    head += f" -> {ret_types[0]}:\n" if len(ret) == 1 else ":\n"
    body = "\n".join(em.lines) if em.lines else ""
    tail = f"    return {', '.join(ret)}\n"
    src = head + (body + "\n" if body else "") + tail
    return EmitInfo(source=src, n_returns=len(ret), n_eqns=n_eqns,
                    input_sizes=tuple(in_sizes), output_sizes=tuple(out_sizes),
                    helpers=tuple(sorted(em.helpers)),
                    vec_lengths=tuple(sorted(vec_lengths | em.vecs)))


# ---------------------------------------------------------------------------
# a whole config
# ---------------------------------------------------------------------------


def _sanitise(node: str, i: int) -> str:
    keep = "".join(c if c.isalnum() else "_" for c in node).strip("_")
    return f"n{i}_{keep}"[:96]


_DS_DEFAULTS = None


def _datastructure_default(path: str):
    """The PROCESS `DataStructure` field's OWN declared default for `.area.name`, as an
    array, or `None`.

    This is the value-supply fallback, and it is a real component rather than a test
    fixture: a read's SHAPE decides whether its node traces at all, and a scalar handed
    to a genuinely array-valued parameter fails with
    `IndexError: array is 0-dimensional` -- which reads exactly like a tracing failure
    and is not one. The length must come from the field, never be guessed: a guessed
    length traces perfectly well and is then silently wrong.
    """
    global _DS_DEFAULTS
    if _DS_DEFAULTS is None:
        from process.core.model import DataStructure

        _DS_DEFAULTS = DataStructure()
    parts = path.strip(".").split(".")
    if path.startswith("^") or len(parts) != 2:
        return None
    sub = getattr(_DS_DEFAULTS, parts[0], None)
    if sub is None:
        return None
    val = getattr(sub, parts[1], None)
    if val is None or isinstance(val, (str, bytes)):
        return None
    try:
        arr = np.asarray(val, dtype=float)
    except (TypeError, ValueError):
        return None
    return arr if arr.ndim >= 1 else None


def node_values(cold, defn, real_values=None, mda_env=None):
    """One concrete value per `defn.reads`, in order.

    In priority order: ground truth from the PROCESS run (`sand_harness.ground_truth`),
    the **completed MDA run's own output env** (`mda_env`, keyed by `VarPath`), a value
    already propagated through the Drive (`real_values`), the `DataStructure` field's
    own declared default, and only then a scalar `1.0`.

    The value only has to be *representative* -- it fixes the trace's branch selection
    and its shapes -- but the shape has to be RIGHT, so every step that can supply a
    real shape sits ahead of the scalar fallback rather than after it.

    **`mda_env` is the step that closes the profile-array hole, and it is a real
    component rather than a test fixture.** A `Drive`'s context includes variables with
    no `DataStructure` field at all and no native answer -- `.physics.
    radius_plasma_profile_norm` is the 201-point radius grid, produced by a `ProfileGrid`
    node that sits OUTSIDE the drive because nothing in it depends on an unknown. Ground
    truth raises for it, the `DataStructure` has no such field, and the scalar `1.0`
    fallback then reached `plasma_profiles._simpson`, where `y.shape[0]` on a rank-0
    value raises `IndexError` -- which reads exactly like a tracing failure and is not
    one. `sand_harness.mda_env` already computes the value; it was simply being
    discarded (`assemble.py`). Everything the graph produces is grounded by the graph.
    """
    from functional_process.cottax.sand_harness import ground_truth as _gt

    out = []
    for vp in defn.reads:
        path = vp.path_str()
        v = None
        try:
            v = _gt(cold, vp)
        except Exception:
            v = None
        if v is None and mda_env is not None:
            v = mda_env.get(vp)
        if v is None and real_values is not None:
            v = real_values.get(path)
        if v is None:
            v = _datastructure_default(path)
        out.append(jnp.asarray(1.0 if v is None else v))
    return out


def jaxpr_leaves(config: str, drive=None, cold=None, real_values=None, mda_env=None):
    """`(entries, refused, drive)` for one config's SAND Drive.

    `entries` is the topologically ordered list of `JaxprLeaf`; `refused` is
    `[(node, reason), ...]`, every reason naming the primitive or shape responsible.
    Structural nodes (`Compare`/`Pairwise`/`_Negate`) trace like any other node -- they
    need no special case here at all, which is one more thing the resolver had to know
    and this does not.
    """
    if drive is None or cold is None:
        from .assemble import _assemble
        from functional_process.cottax import native as _native

        drive, _, env = _assemble(config)
        if mda_env is None:
            mda_env = env
        cold = _native.native_reference(
            f"tests/regression/input_files/{config}.IN.DAT").cold

    sub = drive.body.subgraph
    entries, refused = [], []
    for i, name in enumerate(sub.topological_order):
        defn = sub[name]
        fn = getattr(defn, "fn", None)
        node = name.path_str()
        if fn is None:
            refused.append((node, "node has no `fn`"))
            continue
        vals = node_values(cold, defn, real_values, mda_env)
        func_name = _sanitise(node, i)
        try:
            info = emit_node(fn, func_name, vals, len(defn.owns))
        except Refusal as exc:
            refused.append((node, f"refused: {exc}"))
            continue
        except Exception as exc:                    # a trace failure, not a refusal
            refused.append((node, f"{type(exc).__name__}: {str(exc)[:120]}"))
            continue
        entries.append(
            JaxprLeaf(
                node=node,
                fn=func_name,
                inputs=tuple(v.path_str() for v in defn.reads),
                outputs=tuple(v.path_str() for v in defn.owns),
                source=info.source,
                n_eqns=info.n_eqns,
                input_sizes=info.input_sizes,
                output_sizes=info.output_sizes,
                helpers=info.helpers,
                vec_lengths=info.vec_lengths,
            )
        )
    return entries, refused, drive


def module_preamble(entries) -> str:
    """The `import`, the `wp.types.vector` declarations every entry's signature needs,
    and the device helpers any entry calls -- everything that must precede the first
    `@wp.func` in the generated module."""
    lengths = sorted({n for e in entries for n in e.vec_lengths})
    helpers = helper_closure({h for e in entries for h in e.helpers})
    parts = ["import warp as wp"]
    if lengths:
        parts.append(vec_decls(lengths))
    parts.extend(HELPERS[h] for h in helpers)
    return "\n\n".join(parts)


def funcs_source(entries) -> str:
    """Every entry's `@wp.func`, in order, de-duplicated by function name."""
    seen, out = set(), []
    for e in entries:
        if e.fn in seen:
            continue
        seen.add(e.fn)
        out.append(e.source)
    return "\n\n".join(out)
