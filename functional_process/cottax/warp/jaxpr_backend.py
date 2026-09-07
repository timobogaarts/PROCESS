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

**Loop nests (`_plan_chain`, `_emit_chain`).** Scalarisation alone emits one statement
per array ELEMENT per equation, and that product is not bounded by anything physical: a
935-equation node over 201-point profiles is 146,530 statements
(`.tokamak.bootstrap_current`, which therefore did not emit at all), and a 227-equation
node over (14, 201) is 435,304. Where a run of equations is provably ALIGNED with one
iteration index -- every output element the same scalar program over one element of each
operand -- the whole run is emitted instead as ONE loop whose body is scalar throughout,
terminating in an accumulator (a reduction) or a write into a vector (a value something
outside the nest reads). Nothing in between is stored, which is the point: the naive fix,
one local array per intermediate, would trade the statement count for 22 KB of
per-thread storage per (14, 201) intermediate and lose more than it won.

Three things make that safe rather than plausible:

- **Alignment is derived, never assumed.** `_template` runs THIS MODULE'S OWN `_expr` on
  sentinel operands and reads the data dependence off the expressions it produced, then
  requires every output element's program to be identical after canonicalisation. So the
  nest emits exactly what the scalarised path would have emitted, and every proof `_expr`
  already makes (`_bcast`'s shape check, `slice`'s index arithmetic,
  `broadcast_in_dim`'s axis map) is inherited rather than restated. A primitive that is
  not element-wise fails the check by construction.
- **Every index expression is proved elementwise.** `_affine_index` fits `c + sum_d t_d *
  k_d` and then compares the prediction against the whole index map. Anything that form
  cannot express is refused, not approximated.
- **A plan that does not fit is retracted, not forced.** `_ChainVeto` drops one equation
  and replans; a `Refusal` inside a nest falls back to emitting the node fully
  scalarised. Fusion can make a node that refused emit; it cannot make a node that
  emitted refuse. `WARP_FUSE=0` runs the whole backend the old way for comparison.

Measured on the three configurations, `.tokamak.bootstrap_current` goes from 146,530
statements (refused) to **779**, `.physics.fusion_rates` from 47,502 to **452**, and the
total emitted statements of a whole configuration fall by 61-74 % on top of what
`_vec_local`'s memo already saves (`helias_5b`: 99,565 -> 71,555 -> 18,709). Per-node
validation is UNCHANGED by fusion wherever the node set is the same: on `helias_5b` and
`stellarator_helias`, `WARP_FUSE=0` and `WARP_FUSE=1` report the same passes, the same
bit-exact nodes and the same worst relative difference to every digit, and on
`large_tokamak_nof` (which gains `bootstrap_current`, so its seeded draw sequence
shifts) the set of nodes that are not bit-exact is identical.

**What a nest still cannot drive, and why `.physics.impurity_radiation_totals` remains
refused.** Its `jnp.interp` alternates chain, table lookup, chain, table lookup: the
lookups are `gather`s whose index is computed by the chain and whose operand is a
(14, 200) table the node itself builds. Driving them inside the nest needs three things
this module does not have -- a `gather` template whose index is a runtime value rather
than an index map, a `scan` unrolled into the flat equation list (its bracket search is
8 steps of a carry batched at the full 14x201 width), and reductions over a SUBSET of
the axes, since it sums over species at each of 201 points. All three were built and
measured: with them the node still refuses, because the interleaving makes the nest and
the equations between its parts mutually dependent, and the plan then cuts itself down
until the reduction it was built for has left it. Splitting one nest into several by
that dependence is the missing piece, and it is a scheduling problem, not a primitive
one. The three mechanisms were removed again rather than left in place unexercised: on
these three configurations they changed the emitted statement count of exactly zero
nodes, and emission code no node exercises is emission code `jaxpr_validate` never
checks.

**Where the remaining size is, measured rather than guessed.** With nests and the vector
memo in place the largest function is `.tokamak.cs_coil.temperature_margin` at 14,382
statements, and NOTHING in it is array-shaped: every one of its 394 equations is a
scalar, and the whole of it is two `lax.scan`s that `_scan` UNROLLS. A nest cannot help
there, because there is no iteration index to fuse along -- the fix is the other
direction, emitting a scan as a runtime `for` loop over a carry held in Warp locals
instead of one copy of the body per step. That is not done here because it would break
the invariant the rest of this emitter rests on -- that a name, once bound, is never
reassigned (`_materialise`, `_TRIVIAL`) -- and a loop-carried variable is exactly a
reassignment. It is the next thing worth doing, and it is worth about 14,000 statements
on `large_tokamak_nof` and 7,000 more on
`.tokamak.cicc_superconducting_tf_coil.tf_superconductor_temperature_margin`.
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

    const_inputs: frozenset = frozenset()
    """Subset of `inputs` this node's own `@wp.func` binds as a `wp.array` GLOBAL
    parameter rather than a `vec{n}f` value, because `_constant_varpaths` proved the
    VarPath unreachable from any unknown. A caller assembling the kernel (`emit.py`)
    must bind these the same way `array_boundary` already binds a boundary array: one
    shared `wp.array`, not a per-thread local -- see `build_kernel_source`'s
    `const_arrays` parameter."""

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

    `array_ident` names a Warp identifier that is ALREADY indexable at runtime --
    either a `wp.array` GLOBAL parameter (a constant array, Change 1) or a `vec{n}f`
    local/parameter -- when `exprs` is exactly `f"{array_ident}[0]", f"{array_ident}[1]",
    ...` in order, i.e. this Value is an untouched read of that identifier and nothing
    has been computed from it yet. It licenses `_vec_local` to hand back `array_ident`
    directly instead of materialising a fresh copy: a runtime-indexed operand needs
    something subscriptable, and this one already is.
    """

    exprs: tuple[str, ...]
    kind: str
    shape: tuple[int, ...]
    vals: object = None          # np.ndarray of `shape`, or None
    array_ident: str | None = None

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


_INT32_MIN, _INT32_MAX = -(2 ** 31), 2 ** 31 - 1


def _fold_exact(eqn, args, okind, oshape):
    """`eqn`'s result as a concrete array -- or `None`.

    Folded ONLY when the result is integer- or boolean-valued and every operand is an
    integer or boolean value this generator already knows (`Value.vals`). Under that
    restriction folding is not an approximation of the emitted code, it is the same
    arithmetic: integer `+`/`*`/`<`/`select` are exact in both `numpy` and Warp, so the
    literal this substitutes is bit-for-bit the value the unfolded expression would
    have computed at runtime. **Float results are deliberately excluded**, because
    `numpy`'s `log`/`exp`/`div` and the device's are allowed to differ in the last
    ulp and a folded float would silently become a DIFFERENT number from the one the
    kernel would have produced.

    The evaluation is `eqn.primitive.bind` -- JAX's own definition of the primitive,
    on concrete arrays -- rather than a second transcription of each primitive's
    semantics here. A primitive `bind` cannot evaluate eagerly (or that returns
    something of the wrong shape or dtype) folds to nothing and the ordinary emission
    path runs.

    Every operand AND the result must fit in `int32`, because that is the integer type
    the emitted code uses. A value outside it is left unfolded, so this can never
    disagree with what the unfolded code computes by folding at a wider width than the
    kernel runs at.

    What this is FOR: `jnp.take`/`arr[idx]` with a computed index array lowers to
    `iota -> lt/add/select_n` (Python's negative-index normalisation) `-> gather`. The
    `iota` is known, but `lt` and `select_n` produced no `vals`, so the index reaching
    `_gather` was opaque and every one of those equations -- and the gather itself --
    was emitted as one runtime statement per element.
    `.tokamak.cicc_superconducting_tf_coil.tf_stress` spent 45,024 of its 158,642
    statements on exactly that, for six lookups of a 3-element per-layer array, and the
    opaque gathers cut the surrounding 1500-element chain into pieces too small for a
    loop nest to be worth planning.
    """
    if okind not in (_I, _B):
        return None
    n = int(np.prod(oshape)) if oshape else 1
    if n < 1 or n > MAX_ELEMENTS:
        return None
    if len(args) != len(eqn.invars) or not args:
        return None
    ops = []
    for v, a in zip(eqn.invars, args):
        if a.vals is None:
            return None
        k = _kind(v.aval)
        if k not in (_I, _B):
            return None
        vshape = tuple(v.aval.shape)
        vsize = int(np.prod(vshape)) if vshape else 1
        arr = np.asarray(a.vals)
        if arr.size != vsize:
            return None
        arr = arr.reshape(vshape)
        if k == _B:
            ops.append(arr.astype(bool))
            continue
        if not np.issubdtype(arr.dtype, np.integer):
            if not np.all(arr == np.rint(arr)):
                return None
        wide = arr.astype(np.int64)
        if wide.size and (int(wide.min()) < _INT32_MIN or int(wide.max()) > _INT32_MAX):
            return None
        ops.append(wide.astype(np.dtype(v.aval.dtype)))
    try:
        res = eqn.primitive.bind(*ops, **eqn.params)
    except Exception:
        return None
    if isinstance(res, (list, tuple)):
        return None
    try:
        res = np.asarray(res)
    except Exception:
        return None
    if res.shape != tuple(oshape):
        return None
    if okind == _B:
        return res.astype(bool) if res.dtype == np.dtype(bool) else None
    if not np.issubdtype(res.dtype, np.integer):
        return None
    wide = res.astype(np.int64)
    if wide.size and (int(wide.min()) < _INT32_MIN or int(wide.max()) > _INT32_MAX):
        return None
    return res


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

MAX_NODE_LINES = 24000
"""Cap on the number of statements in ONE emitted `@wp.func`.

The cap exists because a node's source is the product of its jaxpr size and its arrays'
lengths, and that product is not bounded by anything physical. It USED to be 60,000,
which is where the three configurations' modules landed at 4-5 MB and a minutes-scale
compile, and it refused `.tokamak.bootstrap_current` (146,530 statements) and
`.physics.impurity_radiation_totals` (435,304) outright.

It is 24,000 now because the emitted code is smaller, not because anything was
loosened. With loop nests and one copy of each materialised vector
(`_vec_local`'s memo) the largest function in any of the three configurations is
**14,382 statements** (`.tokamak.cs_coil.temperature_margin`, `large_tokamak_nof`);
`helias_5b` as a whole is 18,709 statements against 99,565 before, and
`.physics.fusion_rates` alone went from 47,502 to 452. 24,000 leaves two thirds again
as much headroom as the largest node needs and still refuses, loudly, anything that
goes back to being unrolled by accident.

Raising it costs compile time and nothing else -- but note that compile time here is not
linear in the cap, and that with ADJOINTS on the same modules cost about sixty times as
much again (`warp_init`).
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


# ---------------------------------------------------------------------------
# the flattened equation list
# ---------------------------------------------------------------------------


MAX_SCAN_STEPS = 512
"""Cap on a scan's trip count. The loop is UNROLLED, so the emitted body appears once
per step; a longer scan is a named refusal rather than a source explosion."""


def _scan_split(eqn):
    """`(body, body_consts, num_consts, num_carry, length, reverse)` for a `scan`,
    every part of it checked rather than assumed.

    One function so that `_FuncEmitter._scan` (which unrolls the scan into scalarised
    equations) and `_flatten` (which unrolls a carry-only scan into the flat equation
    list, where the fusion planner can reach it) read the SAME split. Two transcriptions
    of a version-dependent parameter layout is two chances to slice the operands in the
    wrong places, and a scan sliced wrongly computes a plausible wrong number.
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
    if length > MAX_SCAN_STEPS:
        raise Refusal(f"scan of {length} steps (cap {MAX_SCAN_STEPS}) -- the "
                      f"loop is unrolled")
    return body, body_consts, num_consts, num_carry, length, bool(p.get("reverse",
                                                                        False))


class _FVar:
    """One variable of the flattened equation list. Carries an aval and nothing else;
    identity is the name."""

    __slots__ = ("aval",)

    def __init__(self, aval):
        self.aval = aval

    def __repr__(self):
        return f"<{np.dtype(self.aval.dtype).name}{list(self.aval.shape)}>"


class _FEqn:
    """One equation of the flattened list. Duck-types `jax.extend.core.JaxprEqn`
    closely enough that `_eqn` and every primitive handler take it unchanged."""

    __slots__ = ("primitive", "invars", "outvars", "params")

    def __init__(self, primitive, invars, outvars, params):
        self.primitive = primitive
        self.invars = invars
        self.outvars = outvars
        self.params = params


# ---------------------------------------------------------------------------
# fusion: axis-aligned element-wise chains as loop nests
# ---------------------------------------------------------------------------

_FUSION = __import__("os").environ.get("WARP_FUSE", "1") not in ("0", "", "false",
                                                                "False")
"""`WARP_FUSE=0` emits every node fully scalarised, as this backend did before loop
nests existed. It is here so a measurement can be repeated on both paths in the same
interpreter with nothing else changed -- a fused and an unrolled node are supposed to
be the same arithmetic in a different shape, and that is a claim to check rather than
assert."""

MAX_CHAIN_NESTS = 6
"""How many loop nests one node may carry. Each one is a separate iteration shape, and
past a handful the values crossing between them cost more storage than the unrolling
they save."""

MAX_CHAIN_REPLANS = 8
"""How many equations a node may veto out of its loop nest before fusion is abandoned
for that node. Each veto strictly shrinks the plan, so this only bounds work."""

FUSE_MIN_ELEMENTS = 24
"""Below this many elements a value is cheaper unrolled than looped: the loop nest
costs a few statements of overhead and gives up the constant folding that scalarisation
gets for free. A chain of shorter values is left exactly as it was."""

MIN_FUSED_EQNS = 4
"""Do not build a loop nest for fewer equations than this. A one-equation "chain" is a
loop around a single statement -- strictly worse than the unrolled form it replaces."""

MAX_FUSE_PROBE = 8192
"""Cap on the number of elements `_template` will probe. The probe is O(elements) in
string work per equation, so this bounds generation time; it is the same number as
`MAX_ELEMENTS` because it bounds the same thing."""

_NO_FUSE = frozenset({
    # Primitives an output element of which is NOT a bounded scalar program over a few
    # elements of the operands, or that emit storage of their own. Reductions are the
    # important entries: with per-occurrence slots a whole `reduce_sum` WOULD present
    # as one uniform template (n_out = 1, and every operand element an occurrence), and
    # inlining it into a loop body is exactly the unrolling this transform exists to
    # remove. `gather`/`scatter` are excluded because a static one's index map is
    # usually not affine in the loop index and a dynamic one reads its operand at a
    # position only the running code knows -- see the note at the end of this module's
    # docstring for what a nest would have to do about that, and why the attempt was
    # abandoned rather than half-landed.
    "gather", "scatter", "scatter_add", "dynamic_slice", "dynamic_update_slice",
    "triangular_solve", "dot_general", "sort", "argsort", "top_k",
    "conv_general_dilated", "cumsum", "cumsum_p", "cumprod", "cummax", "cummin",
    "cumlogsumexp", "reduce_sum", "reduce_prod", "reduce_max", "reduce_min",
    "reduce_and", "reduce_or", "argmax", "argmin", "iota", "while", "scan",
    "cond", "lu", "custom_linear_solve", "unstack", "svd", "bitcast_convert_type",
    "shift_right_arithmetic",
})

MAX_TEMPLATE_SLOTS = 32
"""Cap on the operand references ONE output element may make. It is what stops a
primitive whose element really is a small fan-in (`select_n` over a handful of cases)
from being confused with one whose fan-in is the array length -- the latter inlined
into a loop body would reproduce the unrolling, one loop deeper."""

_FUSABLE_REDUCE = frozenset({"reduce_sum", "reduce_prod"})
"""The reductions a loop nest may terminate in. `reduce_sum`/`reduce_prod` accumulate
into one scalar in flat element order, which is EXACTLY the association `_reduce`
already emits for an all-axes reduction -- so a fused sum is bit-identical to the
unrolled one it replaces, not merely close. `reduce_max`/`reduce_min` are excluded
because their identity element interacts with NaN, and `_max`/`_min` reproduce XLA's
NaN rule for a pairwise compare, not for a compare against an initial infinity."""

_TMPNAME = re.compile(r"\bt\d+\b")
_TOK = re.compile(r"§(\d+)·(\d+)|\b(t\d+)\b")
_INST = re.compile(r"§(\d+)|#(\d+)")


def _instantiate(s: str, sub, names) -> str:
    """A template with `§i` replaced by operand `i`'s scalar expression and `#n` by
    the local bound for the template's own `n`-th statement."""
    def rep(m):
        if m.group(1) is not None:
            e = sub[int(m.group(1))]
            if e is None:
                raise Refusal("chain template references an operand the index map "
                              "reported unused")
            return e
        return names[int(m.group(2))]
    return _INST.sub(rep, s)


@dataclasses.dataclass(frozen=True)
class _Tmpl:
    """How ONE equation is computed for ONE element inside a loop nest.

    `lines` and `expr` are a straight-line scalar program, `#n` naming the local bound
    for this template's own n-th statement and `§o` its o-th operand read.

    `slots` is the alignment record: the o-th operand read of output element k comes
    from element `slots[o][1][k]` of operand `slots[o][0][k]`. Everything a chain node
    depends on goes through it, so `_emit_chain`'s discovery never has to know which
    primitive it is looking at.
    """

    slots: tuple
    lines: tuple = ()
    expr: str = ""


class _Bad(Exception):
    pass


class _ChainVeto(Exception):
    """One equation the plan claimed but the loop nest cannot actually drive at the
    elements the roots ask for -- a `concatenate` whose requested range crosses a block
    boundary, an operand whose index map is not affine in the loop indices. It names the
    equation; `emit` replans without it, so the equation is emitted scalarised and its
    output becomes an ordinary operand of the nest. Raised only during the nest's dry
    discovery pass, before anything has been emitted."""

    def __init__(self, ix: int, why: str):
        super().__init__(f"equation {ix}: {why}")
        self.ix = ix


def _template(eqn):
    """`(lines, expr, slots)` proving that EVERY output element of `eqn` is the same
    scalar program over the same number of operand elements -- or `None`.

    The proof is not a table of "primitives that are element-wise". It is obtained by
    running the emitter's own `_expr` on sentinel operands, in a sandbox whose output
    is thrown away, and then reading the data dependence straight off the expressions
    it produced: output element k's program is sliced out of the sandbox's statements
    by its own dependency closure, canonicalised (statement locals renumbered by first
    appearance, every sentinel `§i·j` collapsed to a bare occurrence marker), and
    required to be IDENTICAL to element 0's. What that establishes is **uniformity** --
    every output element runs the same straight-line program, with the same number of
    operand reads in the same places -- so ONE loop body computes all of them. If any
    element's program differs in shape (an `iota` with a different literal, a `select_n`
    whose predicate folded one way here and the other way there) the canonical forms
    differ and the equation is rejected.

    **Alignment is recorded per occurrence, not per operand**: the o-th operand read of
    element k comes from element `slots[o].idx[k]` of operand `slots[o].which[k]`. Two
    things need that generality. `concatenate` reads a DIFFERENT operand at different
    output elements while running the same program -- collapsing it to one map per
    operand would reject it, and rejecting it is what left 11,000 statements outside
    the nest on `.tokamak.bootstrap_current`, because its central-difference gradient
    is built as `concatenate([one_sided_lo, interior, one_sided_hi])` and then sliced
    straight back down to `interior`. And `integer_pow`/`square` read one element
    twice, which per-operand maps could only express by accident.

    Whether a given occurrence is USABLE is not decided here: it depends on which
    elements the consumer actually asks for, and that is `_emit_chain`'s discovery
    (an occurrence whose operand is not constant over the requested elements vetoes the
    equation there). This function only proves the program is one program.

    So this inherits every proof `_expr` already makes (`_bcast`'s shape check,
    `slice`'s index arithmetic, `broadcast_in_dim`'s axis map, `concatenate`'s block
    arithmetic) instead of restating them, and it cannot drift from what the scalarised
    path emits, because it IS what the scalarised path emits.

    An equation that emits storage of its own (`_vec_local`, from a runtime-indexed
    `gather`) is rejected outright: its statements are not per-element and hoisting
    them into the loop body would recompute a whole vector per iteration.
    """
    name = eqn.primitive.name
    if name in _NO_FUSE or len(eqn.outvars) != 1:
        return None
    out = eqn.outvars[0]
    oshape = tuple(out.aval.shape)
    n_out = int(np.prod(oshape)) if oshape else 1
    if n_out < 1 or n_out > MAX_FUSE_PROBE:
        return None
    args = []
    for i, v in enumerate(eqn.invars):
        sz = int(np.prod(v.aval.shape)) if v.aval.shape else 1
        if sz > MAX_FUSE_PROBE:
            return None
        args.append(Value(tuple(f"§{i}·{j}" for j in range(sz)),
                          _kind(v.aval), tuple(v.aval.shape)))
    sb = _FuncEmitter()
    try:
        exprs, _ = sb._expr(name, eqn, args, _kind(out.aval), oshape)
    except Refusal:
        return None
    if sb.vecs or len(exprs) != n_out:
        return None
    defs, order = {}, {}
    for p, ln in enumerate(sb.lines):
        s = ln.strip()
        lhs, sep, rhs = s.partition(" = ")
        if not sep or not _TMPNAME.fullmatch(lhs):
            return None
        defs[lhs] = rhs
        order[lhs] = p

    canon0 = None
    reads: list = []          # per element: [(operand, element), ...] in text order
    for k in range(n_out):
        need, stack = set(), _TMPNAME.findall(exprs[k])
        while stack:
            nm = stack.pop()
            if nm in need:
                continue
            if nm not in defs:
                return None
            need.add(nm)
            stack.extend(_TMPNAME.findall(defs[nm]))
        seq = sorted(need, key=order.__getitem__)
        ren = {nm: f"#{i}" for i, nm in enumerate(seq)}
        seen_reads: list = []

        def rep(m, seen_reads=seen_reads, ren=ren):
            if m.group(1) is not None:
                seen_reads.append((int(m.group(1)), int(m.group(2))))
                return f"§{len(seen_reads) - 1}"
            nm = m.group(3)
            if nm not in ren:
                raise _Bad
            return ren[nm]

        try:
            canon = tuple(_TOK.sub(rep, s) for s in
                          [defs[nm] for nm in seq] + [exprs[k]])
        except _Bad:
            return None
        if len(seen_reads) > MAX_TEMPLATE_SLOTS:
            return None
        if canon0 is None:
            canon0 = canon
        elif canon != canon0:
            return None
        reads.append(seen_reads)
    n_slots = len(reads[0])
    if any(len(r) != n_slots for r in reads):
        return None
    slots = []
    for o in range(n_slots):
        slots.append((np.array([r[o][0] for r in reads], dtype=np.int64),
                      np.array([r[o][1] for r in reads], dtype=np.int64)))
    return _Tmpl(slots=tuple(slots), lines=tuple(canon0[:-1]), expr=canon0[-1])


def _schedule_units(eqns, plans):
    """A topological order of `[("e", eqn_index) | ("p", plan_index)]`, or `None` if
    the units are not a DAG.

    Ties are broken by the smallest equation index in the unit, so the emitted source
    stays as close to the original order as the dependencies allow -- which is what
    makes a diff of the generated module readable.
    """
    import heapq

    unit: dict = {}
    for p_i, p in enumerate(plans):
        for ix in p.group_ixs | set(p.reduce_roots):
            if ix in unit:
                return None                      # two plans claiming one equation
            unit[ix] = ("p", p_i)
    producer = {}
    for ix, e in enumerate(eqns):
        for ov in e.outvars:
            producer[id(ov)] = ix

    def uid(ix):
        return unit.get(ix, ("e", ix))

    keys: dict = {}
    for ix in range(len(eqns)):
        u = uid(ix)
        keys[u] = min(keys.get(u, ix), ix)
    succ: dict = {u: set() for u in keys}
    indeg = {u: 0 for u in keys}
    for ix, e in enumerate(eqns):
        u = uid(ix)
        for v in e.invars:
            pix = producer.get(id(v))
            if pix is None:
                continue
            w = uid(pix)
            if w != u and u not in succ[w]:
                succ[w].add(u)
                indeg[u] += 1
    heap = [(keys[u], u) for u in keys if indeg[u] == 0]
    heapq.heapify(heap)
    out = []
    while heap:
        _, u = heapq.heappop(heap)
        out.append(u)
        for w in sorted(succ[u], key=keys.__getitem__):
            indeg[w] -= 1
            if indeg[w] == 0:
                heapq.heappush(heap, (keys[w], w))
    if len(out) != len(keys):
        return None
    return out


@dataclasses.dataclass
class _ChainPlan:
    shape: tuple
    size: int
    group_ixs: set = dataclasses.field(default_factory=set)
    """Flat-equation indices emitted INSIDE the nest."""
    owner: dict = dataclasses.field(default_factory=dict)
    """`id(var) -> equation index` for every value the nest computes."""
    tmpl: dict = dataclasses.field(default_factory=dict)
    """`equation index -> _Tmpl`."""
    reduce_roots: list = dataclasses.field(default_factory=list)
    """Equation indices of the reductions the nest accumulates."""
    mat_roots: list = dataclasses.field(default_factory=list)
    """Values the nest must write into a vector because something outside reads them."""
    late: set = dataclasses.field(default_factory=set)
    """Equations that consume the nest. Unused since the schedule became a proper
    unit-level topological sort (`_schedule_units`); kept because it is what the
    per-plan cycle check computes on the way to proving the plan schedulable at all."""


def _plan_chain(eqns, outrefs, veto=frozenset()):
    """The fusion plans for one flattened jaxpr: a list, possibly empty.

    One nest per ITERATION SHAPE, taken greedily largest-payoff first, with each nest
    planned against a world in which the ones already taken no longer exist. Greedy is
    sound here rather than merely convenient: `_plan_for_shape` treats every equation
    outside the nest it is building as external, so a value the NEXT nest will want is
    already a materialised output of the previous one -- the plans cannot disagree
    about who stores what.

    A second shape is worth having and this is what it costs. `.physics.fusion_rates`
    is two families at once, a 201-point profile and the 100-interval Simpson rule
    over it, and with one nest only the profile fused: 192 equations in the nest and
    ~80 more left unrolled at 100 elements each. What crosses between the two nests is
    materialised -- but it is exactly the values Simpson's rule integrates, which the
    reduction has to read at 100 separate indices anyway.
    """
    if not eqns or not _FUSION:
        return []
    tmpl: dict = {}
    for ix, e in enumerate(eqns):
        if ix in veto:
            continue
        t = _template(e)
        if t is not None:
            tmpl[ix] = t
    if not tmpl:
        return []

    producer = {}
    for ix, e in enumerate(eqns):
        for ov in e.outvars:
            producer[id(ov)] = ix
    consumers: dict = {}
    for ix, e in enumerate(eqns):
        for v in e.invars:
            consumers.setdefault(id(v), set()).add(ix)
    out_ids = {id(o) for o in outrefs}

    def size_of(v):
        return int(np.prod(v.aval.shape)) if v.aval.shape else 1

    # Candidate reductions: an ALL-AXES float sum/product over a big operand -- one
    # accumulator for the whole nest. A reduction over some of the axes is a different
    # nest shape (its reduced axes would have to be the innermost loops and its
    # accumulator would live one level up); nothing in these three configurations
    # presents one at the head of a chain, so it is not emitted.
    red_by_shape: dict = {}
    for ix, e in enumerate(eqns):
        if e.primitive.name not in _FUSABLE_REDUCE:
            continue
        a = e.invars[0]
        if _kind(a.aval) != _F or _kind(e.outvars[0].aval) != _F:
            continue
        axes = e.params.get("axes")
        if axes is None or tuple(sorted(int(x) for x in axes)) != \
                tuple(range(len(a.aval.shape))):
            continue
        if size_of(a) < FUSE_MIN_ELEMENTS or size_of(a) > MAX_FUSE_PROBE:
            continue
        red_by_shape.setdefault(tuple(a.aval.shape), []).append(ix)

    shapes = set(red_by_shape)
    for o in outrefs:
        if id(o) in producer and producer[id(o)] in tmpl \
                and FUSE_MIN_ELEMENTS <= size_of(o) <= MAX_FUSE_PROBE \
                and _kind(o.aval) == _F:
            shapes.add(tuple(o.aval.shape))

    plans, claimed = [], set()
    for _ in range(MAX_CHAIN_NESTS):
        best = None
        live = {ix: t for ix, t in tmpl.items() if ix not in claimed}
        for shape in shapes:
            reds = [ix for ix in red_by_shape.get(shape, ()) if ix not in claimed]
            plan = _plan_for_shape(shape, eqns, live, producer, consumers,
                                   out_ids, reds, size_of)
            if plan is None or len(plan.group_ixs) < MIN_FUSED_EQNS:
                continue
            if best is None or len(plan.group_ixs) > len(best.group_ixs):
                best = plan
        if best is None:
            break
        plans.append(best)
        claimed |= best.group_ixs | set(best.reduce_roots)
    return plans


def _plan_for_shape(shape, eqns, tmpl, producer, consumers, out_ids, reds, size_of):
    size = int(np.prod(shape)) if shape else 1
    seeds = []
    for ix in reds:
        seeds.append(eqns[ix].invars[0])
    for ix, e in enumerate(eqns):
        ov = e.outvars[0] if len(e.outvars) == 1 else None
        if ov is None or ix not in tmpl:
            continue
        if tuple(ov.aval.shape) == shape and id(ov) in out_ids:
            seeds.append(ov)
    if not seeds:
        return None

    # Backward closure over equations `_template` proved element-wise.
    group: set = set()
    stack = list(seeds)
    while stack:
        v = stack.pop()
        ix = producer.get(id(v))
        if ix is None or ix in group or ix not in tmpl:
            continue
        group.add(ix)
        # An occurrence may read a different operand at different output elements
        # (`concatenate`), so the closure follows EVERY operand any occurrence can
        # reach. Over-reaching here is safe -- `_emit_chain`'s discovery resolves the
        # actual operand per element and vetoes the equation if it is not constant --
        # while under-reaching would leave a producer outside the nest and materialise
        # it for nothing.
        for which, _idx in tmpl[ix].slots:
            for i in np.unique(which).tolist():
                stack.append(eqns[ix].invars[int(i)])
    if not group:
        return None

    mat: dict = {}
    for _ in range(64):
        # A reduction only counts as part of the nest while its operand is still
        # computed there; one whose chain has been pushed out is an ordinary equation
        # again, and treating it as internal would understate who reads what.
        red_set = {ix for ix in reds
                   if producer.get(id(eqns[ix].invars[0])) in group}
        mat = {vid: v for vid, v in mat.items() if producer.get(vid) in group}
        # Anything the nest computes but something outside reads must be written
        # somewhere. If it has the nest's own shape it becomes a materialised output;
        # otherwise the equation leaves the nest and is emitted scalarised, which puts
        # its value back where the outside can read it.
        changed = False
        for ix in sorted(group):
            ov = eqns[ix].outvars[0]
            if id(ov) in mat:
                continue
            ext = (id(ov) in out_ids
                   or any(c not in group and c not in red_set
                          for c in consumers.get(id(ov), ())))
            if not ext:
                continue
            if tuple(ov.aval.shape) == shape and _kind(ov.aval) == _F:
                mat[id(ov)] = ov
            else:
                group.discard(ix)
                changed = True
        if changed:
            continue
        # A unit-level cycle: an equation outside the nest that both consumes the nest
        # and feeds it. Emitting the nest as one statement would need it in two places
        # at once, so the equations that depend on the outside-of-the-nest part are
        # pushed out until the schedule is a DAG again.
        gout = set(mat) | {id(eqns[ix].outvars[0]) for ix in red_set}
        late: set = set()
        work = [c for vid in gout for c in consumers.get(vid, ())
                if c not in group and c not in red_set]
        while work:
            c = work.pop()
            if c in late:
                continue
            late.add(c)
            for ov in eqns[c].outvars:
                work.extend(x for x in consumers.get(id(ov), ())
                            if x not in group and x not in red_set and x not in late)
        bad = {ix for ix in group
               if any(producer.get(id(v)) in late for v in eqns[ix].invars)}
        if bad:
            group -= bad
            for ix in bad:
                mat.pop(id(eqns[ix].outvars[0]), None)
            continue
        if not group:
            return None
        live_mat = [v for vid, v in mat.items() if producer.get(vid) in group]
        if not red_set and not live_mat:
            return None
        # Every equation of `group` is claimed by the nest, including any the pruning
        # left unreachable from a root: `_emit_chain` walks BACK from the roots and so
        # emits only what is reached, and the fixpoint above has already proved that an
        # unreached one has no consumer outside the nest either. Dropping them from
        # `group_ixs` instead would be the bug: an unreached equation emitted
        # scalarised could read a value that only exists inside the nest.
        return _ChainPlan(shape=tuple(shape), size=size, group_ixs=set(group),
                          owner={id(eqns[ix].outvars[0]): ix for ix in group},
                          tmpl={ix: tmpl[ix] for ix in group},
                          reduce_roots=sorted(red_set), mat_roots=live_mat,
                          late=late)
    return None


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
        self._vec_memo: dict = {}
        """`(length, element expressions) -> the vector local already holding them`.
        See `_vec_local`."""
        self._vec_idents: set = set()
        """Identifiers of `vec{n}f` LOCALS this body built. Only these may be returned
        in place of a copy (`emit_node`): a `wp.array` constant-global parameter is
        also `array_ident`-tagged and is not a vector."""
        self.vecs: set[int] = set()
        """Fixed-length vector types this BODY needs (a runtime-indexed `gather`
        materialises its operand into one). `emit_node` unions these with the lengths
        the signature needs, and `module_preamble` declares them."""
        self._concrete: dict = {}
        """`{Var: ndarray}` of jaxpr variables `emit_node` proved data-independent
        (see `_invariant_vars`) -- empty unless this node actually has a `scatter`
        without `unique_indices`. `_scatter`'s only reader."""

    def _fresh(self) -> str:
        self._n += 1
        return f"t{self._n}"

    _TRIVIAL = re.compile(r"^(?:[A-Za-z_]\w*(?:\[\d+\])?|wp\.(?:float64|int32)\([^()]*\)"
                          r"|True|False)$")
    """An expression that is already a name, a literal, or a LITERAL-INDEXED subscript
    of one (`ident[7]`). Re-binding one to a fresh local is pure source bloat -- and
    source size is not free here: a fully unrolled 201-point profile node emits tens of
    thousands of statements and Warp's own codegen is what pays for them (a 2.3 MB
    module measured at 17 minutes to compile). Every local this skips is an identical
    value under a different name; nothing in this emitter ever reassigns a name it has
    bound, and a `@wp.func` parameter is not assignable, so substituting the name is
    exactly equivalent.

    The subscript case matters specifically for a `vec{n}f`/`wp.array` PARAMETER
    (Change 1's constant globals, and any array-valued parameter): before this, a pure
    shape rearrangement of one (`squeeze`, `reshape`, ...) re-bound EVERY element to a
    fresh local regardless -- `ident[0]` through `ident[2799]`, one statement each, for
    no reason but that the string wasn't a bare name. A literal index into a name is
    exactly as cheap to read again as a local is to read once, so it never needs
    binding either."""

    def _materialise(self, exprs, kind, shape, vals=None, array_ident=None) -> Value:
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
        return Value(tuple(names), kind, tuple(shape), vals, array_ident=array_ident)

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
        `consts`. Returns one `Value` per outvar.

        The jaxpr is FLATTENED first (`_flatten`: nesting primitives inlined, literals
        and constvars bound as ordinary values), then a fusion plan is computed over
        the flat equation list (`_plan_chain`), then the flat list is emitted. With no
        plan this is exactly the previous straight walk.

        A `Refusal` raised while emitting a FUSED plan retracts the plan and re-emits
        the same jaxpr scalarised, so fusion can never turn a node that used to emit
        into a node that refuses -- it can only make one that used to refuse emit.
        """
        eqns, env, outrefs = self._flatten(jaxpr, consts, args)
        # Consumption counts for THIS jaxpr, so `_eqn` can prove a value has exactly
        # one consumer before fusing it into that consumer.
        #
        # Saved and restored rather than accumulated: JAX reuses one inner `Jaxpr`
        # object for every call site of the same traced function, so a `jit[name=
        # signbit]` reached twice would count each of its variables twice and no fusion
        # would ever fire. (It did not, on `parabolic_profile_values`, until this.)
        uses: dict = {}
        for e in eqns:
            for v in e.invars:
                uses[v] = uses.get(v, 0) + 1
        for o in outrefs:
            uses[o] = uses.get(o, 0) + 1
        outer_uses, self._uses = self._uses, uses
        outer_bits, self._bits = self._bits, {}
        try:
            mark = (len(self.lines), self._n, set(self.helpers), set(self.vecs),
                    set(self._vec_idents), dict(self._vec_memo))

            def rewind():
                del self.lines[mark[0]:]
                self._n = mark[1]
                self.helpers.clear()
                self.helpers.update(mark[2])
                self.vecs.clear()
                self.vecs.update(mark[3])
                self._vec_idents.clear()
                self._vec_idents.update(mark[4])
                self._vec_memo.clear()
                self._vec_memo.update(mark[5])
                self._bits = {}
                return self._flatten(jaxpr, consts, args)

            veto: set = set()
            for _ in range(MAX_CHAIN_REPLANS):
                plans = _plan_chain(eqns, outrefs, veto)
                if not plans:
                    return self._walk(env, eqns, outrefs, ())
                try:
                    return self._walk(env, eqns, outrefs, plans)
                except _ChainVeto as v:
                    veto.add(v.ix)
                    eqns, env, outrefs = rewind()
                except Refusal:
                    eqns, env, outrefs = rewind()
                    return self._walk(env, eqns, outrefs, ())
            eqns, env, outrefs = rewind()
            return self._walk(env, eqns, outrefs, ())
        finally:
            self._uses = outer_uses
            self._bits = outer_bits

    def _walk(self, env, eqns, outrefs, plans) -> list[Value]:
        """Emit the flat equation list, each plan's equations replaced by one loop
        nest, in an order derived from the dependencies rather than assumed.

        The order is a topological sort of UNITS -- a nest counts as one -- because a
        nest is one statement in the emitted source and every value it reads has to
        exist by then. The original equation order is a topological sort of equations,
        which is not the same thing once a nest collapses many of them into one place;
        scheduling by it is what would put a nest's operand after the nest. A plan that
        cannot be scheduled (two nests that each read the other) is dropped rather than
        forced, and the emitted code is the same code one nest fewer.
        """
        plans = list(plans or ())
        while True:
            if not plans:
                for eqn in eqns:
                    self._eqn(env, eqn)
                return [self._read(env, o) for o in outrefs]
            order = _schedule_units(eqns, plans)
            if order is not None:
                break
            plans.pop()
        for kind, payload in order:
            if kind == "e":
                self._eqn(env, eqns[payload])
            else:
                self._emit_chain(env, eqns, plans[payload])
        return [self._read(env, o) for o in outrefs]

    # -- flattening ---------------------------------------------------------

    def _flatten(self, jaxpr, consts, args: list[Value]):
        """`(eqns, env, outrefs)` -- one linear `_FEqn` list with every nesting
        primitive inlined, every `Literal` and constvar already bound in `env`, and
        every variable a fresh `_FVar` unique to its call site.

        Inlining here rather than during the walk is what makes fusion possible at
        all: a chain of element-wise arithmetic in this graph routinely passes through
        a `jit[name=diff]` or a `jit[name=_where]`, and a chain that stops at every
        such boundary is not a chain. It also makes the walk's own bookkeeping honest
        -- `self._uses` is now counted over the equations that are actually emitted,
        not over a nested jaxpr object JAX may share between two call sites.
        """
        eqns: list = []
        env: dict = {}

        def bind_const(aval, val) -> _FVar:
            _check_size(aval, "jaxpr const")
            fv = _FVar(aval)
            env[fv] = _const_value(val, _kind(aval))
            return fv

        def go(j, cvals, argrefs):
            if len(cvals) != len(j.constvars):
                raise Refusal(f"jaxpr has {len(j.constvars)} constvars but "
                              f"{len(cvals)} consts")
            if len(argrefs) != len(j.invars):
                raise Refusal(f"jaxpr takes {len(j.invars)} arguments but "
                              f"{len(argrefs)} were supplied")
            m: dict = {}
            for cv, cval in zip(j.constvars, cvals):
                m[cv] = bind_const(cv.aval, cval)
            for iv, a in zip(j.invars, argrefs):
                m[iv] = a

            def ref(v):
                if isinstance(v, Literal):
                    _check_size(v.aval, "literal")
                    return bind_const(v.aval, v.val)
                try:
                    return m[v]
                except KeyError:
                    raise Refusal(f"unbound jaxpr variable {v}") from None

            for eqn in j.eqns:
                nest = _nested(eqn)
                if nest is not None:
                    sub, sub_consts = nest
                    outs = go(sub, sub_consts, [ref(v) for v in eqn.invars])
                    if len(outs) != len(eqn.outvars):
                        raise Refusal(
                            f"{eqn.primitive.name}: nested jaxpr returned "
                            f"{len(outs)} value(s) for {len(eqn.outvars)} outvar(s)")
                    for ov, o in zip(eqn.outvars, outs):
                        m[ov] = o
                    continue
                ins = [ref(v) for v in eqn.invars]
                outs = []
                for ov in eqn.outvars:
                    fv = _FVar(ov.aval)
                    m[ov] = fv
                    outs.append(fv)
                eqns.append(_FEqn(eqn.primitive, ins, outs, eqn.params))
            return [ref(o) for o in j.outvars]

        top = []
        if len(args) != len(jaxpr.invars):
            raise Refusal(f"jaxpr takes {len(jaxpr.invars)} arguments but "
                          f"{len(args)} were supplied")
        for iv, a in zip(jaxpr.invars, args):
            fv = _FVar(iv.aval)
            env[fv] = a
            top.append(fv)
        outrefs = go(jaxpr, consts, top)
        return eqns, env, outrefs

    # -- the fused loop nest --------------------------------------------------

    def _affine_index(self, pos, ivars, ls, strides, limit: int):
        """A Warp index expression reproducing `pos` EXACTLY over the whole iteration
        space, or `None`.

        The only form admitted is `c + sum_d t_d * k_d` over the loop nest's own index
        variables -- no division, no modulus, nothing whose Warp integer semantics
        would have to be assumed. The coefficients are READ OFF `pos` (at the flat
        offset where index `d` advances by one) and then the whole predicted map is
        compared against `pos` element by element. That comparison is the proof: a map
        this form cannot express is rejected, never approximated, so an operand that is
        not genuinely aligned with the loop index becomes a refusal rather than a
        plausible wrong number.
        """
        pos = np.asarray(pos)
        if pos.min() < 0 or pos.max() >= limit:
            return None
        c = int(pos[0])
        coef = []
        for d in range(len(ls)):
            coef.append(0 if ls[d] == 1 else int(pos[strides[d]]) - c)
        grid = np.indices(tuple(ls)).reshape(len(ls), -1)
        pred = np.full(pos.shape, c, dtype=np.int64)
        for d, t in enumerate(coef):
            if t:
                pred = pred + t * grid[d]
        if not np.array_equal(pred, pos):
            return None
        terms = []
        for d, t in enumerate(coef):
            if t == 0:
                continue
            terms.append(ivars[d] if t == 1 else f"{t} * {ivars[d]}")
        if c or not terms:
            terms.append(str(c))
        return " + ".join(terms)

    def _emit_chain(self, env, eqns, plan) -> None:
        """Emit `plan` as one loop nest whose body is scalar throughout.

        Nothing inside the nest is stored: every intermediate is one Warp local with a
        live range of a few statements, and the only arrays that exist are the ones
        that must (a value the node returns, a value a later equation indexes at
        runtime, an operand the plan could not prove aligned). That is the whole point
        of the transform -- the alternative, one local per element per intermediate,
        is what put 146,530 statements in `.tokamak.bootstrap_current` and 435,304 in
        `.physics.impurity_radiation_totals`.

        One loop per axis of the iteration shape, in the shape's own order, and every
        reduction the nest terminates in consumes ALL of them -- so there is exactly
        one accumulator per reduction and it lives outside the whole nest.
        """
        shape = list(plan.shape) if plan.shape else [1]
        n = plan.size
        strides, acc_s = [], 1
        for x in reversed(shape):
            strides.insert(0, acc_s)
            acc_s *= x
        ivars = ["k" + self._fresh()[1:] for _ in shape]
        base = np.arange(n, dtype=np.int64)

        # ---- discovery: every (value, index map) the roots need -------------------
        nodes: dict = {}
        post: list = []
        seen: set = set()
        requester: dict = {}
        """Which equation first asked for a given value: the one to veto if that value
        turns out not to be readable at the loop index."""
        stack = []

        def key_of(var, pos):
            return (id(var), pos.tobytes())

        roots = []
        for ix in plan.reduce_roots:
            roots.append((eqns[ix].invars[0], base))
        for v in plan.mat_roots:
            roots.append((v, base))
        for var, pos in roots:
            stack.append((var, pos, False))
        while stack:
            var, pos, expanded = stack.pop()
            k = key_of(var, pos)
            if expanded:
                post.append(k)
                continue
            if k in seen:
                continue
            seen.add(k)
            ix = plan.owner.get(id(var))
            if ix is None:
                nodes[k] = ("leaf", var, pos, requester.get(k))
                post.append(k)
                continue
            deps = []
            for which, idx in plan.tmpl[ix].slots:
                w = which[pos]
                w0 = int(w[0])
                if not bool(np.all(w == w0)):
                    # `concatenate` (or a `pad`) whose requested elements straddle two
                    # of its blocks: one loop body cannot read two different operands
                    # at the same statement. Refused here, not resolved by picking one.
                    raise _ChainVeto(ix, "an operand read that is not the same operand "
                                          "over the elements the nest asks for")
                dv = eqns[ix].invars[w0]
                dpos = idx[pos]
                dk = key_of(dv, dpos)
                requester.setdefault(dk, ix)
                deps.append((dk, dv, dpos))
            nodes[k] = ("eqn", var, pos, (ix, deps))
            stack.append((var, pos, True))
            for d in deps:
                stack.append((d[1], d[2], False))

        # ---- leaves: decide, DRY, how each is read at the loop index ---------------
        # Nothing is emitted in this pass. A leaf that cannot be read at the loop index
        # vetoes the equation that asked for it, and a veto has to be raised before the
        # nest has put a single statement in `self.lines`.
        plan_leaf: dict = {}
        for k in post:
            kind, var, pos, req = nodes[k]
            if kind != "leaf":
                continue
            v = env.get(var)
            if v is None:
                raise Refusal("chain leaf is not bound (scheduling)")
            c0 = int(pos[0])
            if bool(np.all(pos == c0)):
                if c0 >= v.size:
                    raise Refusal("chain leaf index out of range")
                plan_leaf[k] = ("scalar", v.exprs[c0])
                continue
            if v.vals is not None:
                sel = np.asarray(v.vals).reshape(-1)
                if int(pos.max()) < sel.size:
                    picked = sel[pos]
                    if picked.size and bool((picked == picked.reshape(-1)[0]).all()):
                        plan_leaf[k] = ("scalar",
                                        _fmt_scalar(picked.reshape(-1)[0], v.kind))
                        continue
            idx = self._affine_index(pos, ivars, shape, strides, v.size)
            why = None
            if idx is None:
                why = "an operand whose index map is not affine in the loop indices"
            elif v.array_ident is None and v.kind != _F:
                # `_vec_local` materialises float64 vectors only; an integer or boolean
                # operand read at a runtime index has nowhere to live.
                why = f"an operand of kind {v.kind!r} read at a runtime index"
            elif v.size > MAX_VEC_ELEMENTS:
                why = f"an operand of {v.size} elements read at a runtime index"
            if why is not None:
                if req is None:
                    raise Refusal(f"fused loop over {plan.shape}: {why}")
                raise _ChainVeto(req, why)
            plan_leaf[k] = ("index", v, idx)

        # ---- from here on the nest is emitted; no veto may be raised --------------
        leaf_expr: dict = {}
        for k, how in plan_leaf.items():
            leaf_expr[k] = (how[1] if how[0] == "scalar"
                            else f"{self._vec_local(how[1])}[{how[2]}]")

        # ---- accumulators and materialised outputs --------------------------------
        accs: dict = {}
        for ix in plan.reduce_roots:
            nm = self._fresh()
            init = ("wp.float64(0.0)" if eqns[ix].primitive.name == "reduce_sum"
                    else "wp.float64(1.0)")
            self.lines.append(f"    {nm} = {init}")
            accs[ix] = nm
        vecs: dict = {}
        for v in plan.mat_roots:
            size = int(np.prod(v.aval.shape)) if v.aval.shape else 1
            if size > MAX_VEC_ELEMENTS:
                raise Refusal(f"fused output of {size} elements "
                              f"(cap {MAX_VEC_ELEMENTS})")
            self.vecs.add(size)
            nm = self._fresh()
            self._vec_idents.add(nm)
            self.lines.append(f"    {nm} = {vec_name(size)}()")
            vecs[id(v)] = nm

        # ---- the nest -------------------------------------------------------------
        for d in range(len(shape)):
            self.lines.append("    " + "    " * d
                              + f"for {ivars[d]} in range({shape[d]}):")
        body_ind = "    " + "    " * len(shape)

        val: dict = {}
        for k in post:
            kind, var, pos, info = nodes[k]
            if kind == "leaf":
                e = leaf_expr[k]
                if not self._TRIVIAL.match(e):
                    nm = self._fresh()
                    self.lines.append(f"{body_ind}{nm} = {e}")
                    e = nm
                val[k] = e
                continue
            ix, deps = info
            t = plan.tmpl[ix]
            sub = [val[d[0]] for d in deps]
            names: list = []
            for rhs in t.lines:
                nm = self._fresh()
                self.lines.append(f"{body_ind}{nm} = {_instantiate(rhs, sub, names)}")
                names.append(nm)
            e = _instantiate(t.expr, sub, names)
            if not self._TRIVIAL.match(e):
                nm = self._fresh()
                self.lines.append(f"{body_ind}{nm} = {e}")
                e = nm
            val[k] = e

        flat_idx = " + ".join(
            (ivars[d] if strides[d] == 1 else f"{strides[d]} * {ivars[d]}")
            for d in range(len(shape)) if shape[d] != 1) or "0"
        for ix in plan.reduce_roots:
            e = val[key_of(eqns[ix].invars[0], base)]
            op = "+" if eqns[ix].primitive.name == "reduce_sum" else "*"
            self.lines.append(f"{body_ind}{accs[ix]} = ({accs[ix]} {op} {e})")
        for v in plan.mat_roots:
            e = val[key_of(v, base)]
            self.lines.append(f"{body_ind}{vecs[id(v)]}[{flat_idx}] = {e}")

        # ---- bind what the nest produced ------------------------------------------
        for ix in plan.reduce_roots:
            ov = eqns[ix].outvars[0]
            env[ov] = Value((accs[ix],), _F, tuple(ov.aval.shape))
        for v in plan.mat_roots:
            nm = vecs[id(v)]
            size = int(np.prod(v.aval.shape)) if v.aval.shape else 1
            env[v] = Value(tuple(f"{nm}[{j}]" for j in range(size)), _F,
                           tuple(v.aval.shape), array_ident=nm)

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
            # The shift amount is read from its bound VALUE, not from a `Literal`
            # instance check: `_flatten` binds every literal as an ordinary constant
            # value, so `isinstance(..., Literal)` is no longer how a compile-time
            # constant presents itself here. `vals is not None` is the same proof --
            # it is exactly "this value is known to the generator".
            sv = self._read(env, eqn.invars[1])
            if sv.vals is None or sv.size != 1 or \
                    int(np.asarray(sv.vals).reshape(-1)[0]) != 63:
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

        # ---- multi-output primitives -------------------------------------------
        # Everything below `_eqn`'s generic path handles exactly one outvar. These
        # are the multi-output (or multi-jaxpr) primitives this graph is actually
        # observed to need: `.tokamak.pf_coil.inductance`'s `unstack`,
        # `.vacuum.vacuum_old`'s `cond`, and
        # `.tokamak.cicc_superconducting_tf_coil.tf_stress`'s `lu` -- which turned out,
        # once traced, to arrive wrapped in a `custom_linear_solve` (`jnp.linalg.solve`
        # is `lu` + `custom_linear_solve` + `triangular_solve`, not bare `lu`; see
        # `_custom_linear_solve`'s docstring). `svd` (the `.tokamak.pf_coil.*_currents`
        # pair) is refused by name below rather than attempted -- see `_lu`'s docstring
        # for why `lu` is tractable here and `svd` is not.
        if name == "unstack":
            self._unstack(env, eqn)
            return
        if name == "cond":
            self._cond(env, eqn)
            return
        if name == "lu":
            self._lu(env, eqn)
            return
        if name == "custom_linear_solve":
            self._custom_linear_solve(env, eqn)
            return
        if name == "svd":
            raise Refusal(
                "svd (no differentiable SVD is emitted here; the observed call sites "
                "-- .tokamak.pf_coil.initiation_currents/equilibrium_currents -- are "
                "tall-skinny least-squares problems (74x4, 74x2), not a small fixed "
                "matrix a closed-form/unrolled decomposition could handle exactly, and "
                "a general iterative SVD is a real numerical project this backend does "
                "not attempt rather than approximate)")
        if name == "while":
            # `lax.while_loop`'s trip count is a VALUE, not a jaxpr parameter -- unlike
            # `scan`, whose static `length` is exactly what licenses unrolling it
            # (`_scan`'s docstring). If a loop's bound were knowable at trace time, JAX
            # would have lowered it to a `scan`, not a `while`; the mere presence of a
            # `while` primitive is itself the evidence that this backend cannot prove
            # a fixed bound, so this refuses unconditionally rather than trying to
            # unroll to `cond_jaxpr`'s (data-dependent) cap.
            #
            # The one call site this graph is observed to reach, `.vacuum.vacuum_old`'s
            # `solve_duct_diameter` (`functional_process/models/vacuum/vacuum.py`), is
            # a Newton iteration whose own docstring records the trip count taking six
            # different values (6-11) within a single `stellarator_helias` SAND solve,
            # against a 100-iteration cap -- genuinely data-dependent, not merely
            # unproved. Warp differentiates only static-bound loops correctly, so this
            # stays refused rather than guessing a bound.
            raise Refusal("while (data-dependent trip count -- lax.while_loop has no "
                          "static length to unroll, unlike scan; see this branch's "
                          "comment for the call site this was checked against)")

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
        # `array_ident` propagates PROVABLY, not by primitive name: whenever this
        # equation's computed `exprs` are LITERALLY the operand's own `exprs` (a
        # `squeeze`/`reshape`/`expand_dims`/... that only changes the declared shape,
        # never the flat element order -- `_IDENTITY_FLAT` in `_expr`), the operand's
        # own subscriptable identifier is still exactly what every element reads, so a
        # later runtime-indexed use (`_gather`'s `_vec_local`) can keep reading it
        # directly instead of materialising one more copy. A single-input equation is
        # required (an n-ary one has no ONE operand to inherit from), and the equality
        # is checked against the ACTUAL exprs produced, not assumed from the primitive
        # name -- so this is correct for exactly the ops that qualify today and stays
        # correct if the primitive set changes without this comment being updated.
        ident = None
        if len(args) == 1 and args[0].array_ident is not None \
                and tuple(exprs) == args[0].exprs:
            ident = args[0].array_ident
        env[out] = self._materialise(exprs, okind, oshape, vals, array_ident=ident)

    # -- scan ---------------------------------------------------------------

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
        body, body_consts, num_consts, num_carry, length, reverse = _scan_split(eqn)
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

    # -- unstack --------------------------------------------------------------

    def _unstack(self, env, eqn) -> None:
        """`lax.unstack(operand, axis)`: split `operand` into `len(outvars)` slices
        along `axis`, dropping that axis. A fixed split of a known-length value -- the
        trip count of a `for` loop over the axis, in effect -- so, like `slice` and
        `transpose` above, it is pure index arithmetic the GENERATOR does, not
        anything emitted at runtime."""
        for ov in eqn.outvars:
            _check_size(ov.aval, "primitive 'unstack' output")
        a = self._read(env, eqn.invars[0])
        axis = int(eqn.params["axis"])
        n = a.shape[axis] if a.shape else 1
        if n != len(eqn.outvars):
            raise Refusal(f"unstack: axis {axis} has size {n} for "
                          f"{len(eqn.outvars)} outputs")
        idx = np.arange(a.size).reshape(a.shape)
        for i, ov in enumerate(eqn.outvars):
            okind = _kind(ov.aval)
            oshape = tuple(ov.aval.shape)
            sel = np.take(idx, i, axis=axis).reshape(-1)
            exprs = [self._coerce(a.exprs[j], a.kind, okind) for j in sel]
            vals = (None if a.vals is None
                    else np.take(np.asarray(a.vals), i, axis=axis).reshape(oshape))
            env[ov] = self._materialise(exprs, okind, oshape, vals)

    # -- cond -------------------------------------------------------------------

    def _cond(self, env, eqn) -> None:
        """`lax.cond`/`lax.switch`, lowered to `cond_p`: an int32 `index` selecting
        one of `branches` (each a `ClosedJaxpr` closed over the SAME operands --
        `jax.lax.cond(pred, ...)` itself lowers to `branches=(false_jaxpr,
        true_jaxpr)`, index 0/1 from `pred`'s own `int32` cast, so this needs no
        boolean special case of its own).

        Emitted the same way `select_n` already is: EVERY branch is emitted
        unconditionally (there is no runtime control flow inside a `@wp.func` body
        here, any more than there is inside a `scan`), and the branches' outputs are
        combined element-wise by a `wp.where` cascade keyed on the index -- last
        branch as the fallthrough default, matching `select_n`'s own convention for
        an out-of-range predicate.

        This is licensed exactly the way `select_n`'s cascade is: a `cond` branch is a
        pure jaxpr (JAX itself refuses effects other than a fixed allow-list inside
        one), so evaluating every branch and then selecting is the SAME VALUE as
        evaluating only the taken one, not an approximation of it.

        A branch that itself contains a construct this backend refuses (a
        data-dependent `while`, for instance -- see `.vacuum.vacuum_old`) still raises
        `Refusal` from THAT branch, regardless of which branch the runtime predicate
        would actually select at any given draw: nothing here can prove a branch dead
        for every input the caller might supply, so refusing is correct, not a defect
        of eager both-branches emission.
        """
        branches = eqn.params.get("branches")
        if not branches or len(branches) < 2:
            raise Refusal(f"cond: {0 if not branches else len(branches)} branch(es)")
        pred = self._read(env, eqn.invars[0])
        if pred.size != 1:
            raise Refusal(f"cond: predicate of shape {pred.shape}, not scalar")
        if pred.kind not in (_I, _B):
            raise Refusal(f"cond: predicate of kind {pred.kind!r}")
        operands = [self._read(env, v) for v in eqn.invars[1:]]
        n_out = len(eqn.outvars)
        outs_per_branch = []
        for br in branches:
            bjaxpr = br.jaxpr if hasattr(br, "jaxpr") else br
            bconsts = list(br.consts) if hasattr(br, "consts") else []
            outs = self.emit(bjaxpr, bconsts, operands)
            if len(outs) != n_out:
                raise Refusal(f"cond: branch returned {len(outs)} value(s) for "
                              f"{n_out} outvar(s)")
            outs_per_branch.append(outs)

        pv = pred.scalar()
        for j, ov in enumerate(eqn.outvars):
            _check_size(ov.aval, "primitive 'cond' output")
            okind = _kind(ov.aval)
            oshape = tuple(ov.aval.shape)
            n = int(np.prod(oshape)) if oshape else 1
            vals = [self._bcast(outs_per_branch[b][j], n, okind, oshape)
                    for b in range(len(branches))]
            if pred.kind == _B:
                if len(vals) != 2:
                    raise Refusal(f"cond: boolean predicate with {len(vals)} branches")
                out = [f"wp.where({pv}, {vals[1][i]}, {vals[0][i]})"
                       for i in range(n)]
            else:
                out = []
                for i in range(n):
                    e = vals[-1][i]
                    for c in range(len(vals) - 2, -1, -1):
                        e = f"wp.where({pv} == wp.int32({c}), {vals[c][i]}, {e})"
                    out.append(e)
            env[ov] = self._materialise(out, okind, oshape, None)

    # -- lu -----------------------------------------------------------------

    MAX_LU_N = 12
    """Cap on the matrix side `lu` will unroll. The algorithm below is
    `O(n^3)`-many `wp.where` nodes for the pivoting alone, on top of the `O(n^3)`
    elimination itself -- fine at the observed 6x6, a source-size problem well before
    it is a numerical one at anything this graph is likely to present."""

    def _lu(self, env, eqn) -> None:
        """`jax.lax.linalg.lu`: LU decomposition of a square matrix with **partial
        pivoting**, unrolled straight-line for an `N` fixed at trace time.

        **Why this is tractable and `svd` is refused.** The matrix side is fixed at
        trace time (`.tokamak.cicc_superconducting_tf_coil.tf_stress` calls it at
        6x6), so the ALGORITHM -- Doolittle elimination with partial pivoting, the
        textbook one, verified below to be exactly what `lax.linalg.lu` computes, not
        an approximation of it -- can be fully unrolled: `N` outer steps, each a fixed
        amount of straight-line arithmetic. The one thing that is genuinely
        data-dependent is WHICH row is the pivot at each step, and that is handled the
        same way a runtime-indexed `gather` already is elsewhere in this file:
        express "read/write row `k`" as a `wp.where` cascade over the `N - k`
        candidate rows, keyed on an equality test against the (symbolic) pivot index,
        rather than as an actual dynamic memory access Warp's scalarised locals have
        no way to perform. `svd` has no comparable reduction to a fixed amount of
        straight-line arithmetic at any practical matrix size -- it is an iterative
        algorithm with a data-dependent number of iterations -- which is why it is
        refused by name instead.

        **Verified, not assumed.** A pure-Python transcription of the loop below
        matched `jax.lax.linalg.lu`'s `pivots` and `permutation` EXACTLY (bit-for-bit
        integer agreement) and its `lu` matrix to ~1e-14 relative, over 3000 random
        matrices of sizes 3..8 spanning three decades of magnitude -- see the working
        notes for this change. The residual is ordinary float64 reassociation (this
        loop and XLA's own do the same arithmetic in the same order; the ~1e-14 is
        LAPACK's blocked kernel on the reference side, not a difference in method),
        and it is `jaxpr_validate`'s job to report the number here, not this docstring's
        to assert one.

        **Pivot selection is first-occurrence-of-the-max, ties included** -- the same
        convention `_reduce`'s `argmax`/`argmin` already use (`<` as the update test,
        so an EQUAL later value does not displace the earlier index) -- because that
        is what LAPACK's `idamax` (and hence `lax.linalg.lu` on every backend actually
        exercised here) does.
        """
        a = self._read(env, eqn.invars[0])
        if len(a.shape) != 2 or a.shape[0] != a.shape[1]:
            raise Refusal(f"lu: operand of shape {a.shape}, not a square matrix")
        n = a.shape[0]
        if n > self.MAX_LU_N:
            raise Refusal(f"lu: {n}x{n} matrix (cap {self.MAX_LU_N})")
        if a.kind != _F:
            raise Refusal(f"lu: operand of kind {a.kind!r}")
        if len(eqn.outvars) != 3:
            raise Refusal(f"lu: {len(eqn.outvars)} outputs, expected 3 "
                          f"(lu, pivots, permutation)")
        lu_ov, piv_ov, perm_ov = eqn.outvars
        for ov in eqn.outvars:
            _check_size(ov.aval, "primitive 'lu' output")
        if tuple(lu_ov.aval.shape) != (n, n):
            raise Refusal(f"lu: lu output shape {tuple(lu_ov.aval.shape)} for an "
                          f"{n}x{n} operand")
        if tuple(piv_ov.aval.shape) != (n,) or tuple(perm_ov.aval.shape) != (n,):
            raise Refusal("lu: pivots/permutation output shape is not "
                          f"({n},)/({n},)")

        # `M[i][j]` -- the working matrix, one Warp local per element, updated in
        # place exactly as the elimination proceeds (materialising a fresh local at
        # every write, same as everywhere else in this emitter).
        M = [[a.exprs[i * n + j] for j in range(n)] for i in range(n)]
        piv: list[str] = []

        def sel(cands: list[str], idx_expr: str, base: int) -> str:
            """A `wp.where` cascade picking `cands[c - base]` where `idx_expr ==
            c`, `c` ranging over `base .. base + len(cands) - 1`, last candidate as
            the (unreached, by construction of the caller) fallthrough."""
            e = cands[-1]
            for off in range(len(cands) - 2, -1, -1):
                c = base + off
                e = f"wp.where({idx_expr} == wp.int32({c}), {cands[off]}, {e})"
            return e

        for k in range(n):
            # ---- pivot: first-occurrence argmax of |M[i][k]| over i in [k, n) ----
            abs_vals = []
            for i in range(k, n):
                t = self._fresh()
                self.lines.append(f"    {t} = wp.abs({M[i][k]})")
                abs_vals.append(t)
            best_v, best_i = abs_vals[0], f"wp.int32({k})"
            for off in range(1, n - k):
                i = k + off
                nv, ni = self._fresh(), self._fresh()
                self.lines.append(
                    f"    {nv} = wp.where({best_v} < {abs_vals[off]}, "
                    f"{abs_vals[off]}, {best_v})")
                self.lines.append(
                    f"    {ni} = wp.where({best_v} < {abs_vals[off]}, "
                    f"wp.int32({i}), {best_i})")
                best_v, best_i = nv, ni
            piv.append(best_i)

            # ---- swap row k and row `best_i` (symbolic) across every column ----
            row_k = M[k]
            rows_below = [M[i] for i in range(k, n)]     # includes row k itself
            new_rows = [[None] * n for _ in range(k, n)]
            for col in range(n):
                picked = sel([r[col] for r in rows_below], best_i, k)
                p = self._fresh()
                self.lines.append(f"    {p} = {picked}")
                for off, i in enumerate(range(k, n)):
                    t = self._fresh()
                    self.lines.append(
                        f"    {t} = wp.where(wp.int32({i}) == {best_i}, "
                        f"{row_k[col]}, wp.where(wp.int32({i}) == wp.int32({k}), "
                        f"{p}, {rows_below[off][col]}))")
                    new_rows[off][col] = t
            for off, i in enumerate(range(k, n)):
                M[i] = new_rows[off]

            # ---- eliminate below the pivot ----
            pivot_val = M[k][k]
            for i in range(k + 1, n):
                mult = self._fresh()
                self.lines.append(f"    {mult} = {M[i][k]} / {pivot_val}")
                M[i][k] = mult
                for col in range(k + 1, n):
                    t = self._fresh()
                    self.lines.append(
                        f"    {t} = {M[i][col]} - {mult} * {M[k][col]}")
                    M[i][col] = t

        # ---- permutation: the same sequential swap, applied to an identity array ----
        perm = [f"wp.int32({i})" for i in range(n)]
        for k, best_i in enumerate(piv):
            rows_below = perm[k:]
            picked = sel(rows_below, best_i, k)
            p = self._fresh()
            self.lines.append(f"    {p} = {picked}")
            new_below = []
            for off, i in enumerate(range(k, n)):
                t = self._fresh()
                self.lines.append(
                    f"    {t} = wp.where(wp.int32({i}) == {best_i}, "
                    f"{perm[k]}, wp.where(wp.int32({i}) == wp.int32({k}), {p}, "
                    f"{rows_below[off]}))")
                new_below.append(t)
            perm[k:] = new_below

        lu_exprs = [M[i][j] for i in range(n) for j in range(n)]
        env[lu_ov] = self._materialise(lu_exprs, _F, (n, n), None)
        env[piv_ov] = self._materialise(piv, _I, (n,), None)
        env[perm_ov] = self._materialise(perm, _I, (n,), None)

    # -- custom_linear_solve -------------------------------------------------

    def _custom_linear_solve(self, env, eqn) -> None:
        """`jax.lax.custom_linear_solve`: JAX's implicit-differentiation wrapper
        around a linear solve. This is what `jnp.linalg.solve` actually lowers to --
        not a bare `lu` call: `lu` computes the factorisation, and
        `custom_linear_solve` wraps the triangular solves that use it so that
        `jax.grad`/`jax.jvp` through the solve has a custom (and cheaper) rule than
        differentiating the factorisation itself.

        Its primitive carries FOUR closed-over jaxprs (`matvec`/`vecmat`/`solve`/
        `transpose_solve`, `eqn.params["jaxprs"]`), but only `solve` computes the
        FORWARD value -- the other three exist solely to define this primitive's
        custom JVP/transpose rule, which is irrelevant to tracing a node's `fn` for
        its forward jaxpr (autodiff never runs here; `jax.make_jaxpr` traces the
        primal computation only). This mirrors JAX's own forward rule exactly --
        `_custom_linear_solve_impl` in `jax._src.lax.control_flow.solves` is
        `core.jaxpr_as_fun(jaxprs.solve)(*(params.solve + b))` -- not an independent
        reading of what the primitive "ought" to do.

        `eqn.params["const_lengths"]` (a `_LinearSolveTuple`) gives the operand-count
        split: `eqn.invars` is `[*matvec_consts, *vecmat_consts, *solve_consts,
        *transpose_solve_consts, *b]` in that fixed order (verified against
        `_split_linear_solve_args` in the same module) -- `solve_consts` and `b` are
        exactly the two argument groups `solve`'s own jaxpr expects, in order.
        """
        p = eqn.params
        const_lengths = p.get("const_lengths")
        jaxprs = p.get("jaxprs")
        if const_lengths is None or jaxprs is None:
            raise Refusal("custom_linear_solve: no const_lengths/jaxprs params")
        total_consts = sum(const_lengths)
        off = int(const_lengths.matvec) + int(const_lengths.vecmat)
        solve_len = int(const_lengths.solve)
        solve_const_vars = eqn.invars[off:off + solve_len]
        b_vars = eqn.invars[total_consts:]
        solve_closed = jaxprs.solve
        bjaxpr = solve_closed.jaxpr if hasattr(solve_closed, "jaxpr") else solve_closed
        bconsts = list(solve_closed.consts) if hasattr(solve_closed, "consts") else []
        operands = [self._read(env, v) for v in solve_const_vars] + \
                   [self._read(env, v) for v in b_vars]
        outs = self.emit(bjaxpr, bconsts, operands)
        if len(outs) != len(eqn.outvars):
            raise Refusal(f"custom_linear_solve: solve branch returned "
                          f"{len(outs)} value(s) for {len(eqn.outvars)} outvar(s)")
        for ov, o in zip(eqn.outvars, outs):
            env[ov] = o

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

        if name == "scatter-mul":
            return self._scatter_mul(eqn, args, okind, oshape)

        if name == "dot_general":
            return (self._dot_general(eqn, args, okind, oshape), None)

        if name == "triangular_solve":
            return (self._triangular_solve(eqn, args, okind, oshape), None)

        if name in ("dynamic_update_slice", "scatter_add", "sort",
                    "argsort", "top_k", "conv_general_dilated",
                    "cumsum_p"):
            # `cond` USED to be refused here too, on the grounds that no node needed
            # it -- that stopped being true once `.vacuum.vacuum_old` and the `lu` work
            # made it worth emitting (see `_cond`, dispatched earlier in `_eqn`, before
            # any equation reaches `_expr` at all). `while` is dispatched earlier too
            # now, for the same reason `cond` is -- see `_eqn`'s `name == "while"`
            # branch for why it stays refused regardless. `sort` remains refused: it
            # would be
            # tractable to unroll at the sizes this graph presents (a small stable
            # sorting network, the same style as `_lu`'s pivoting), but nothing here
            # currently NEEDS it -- `jnp.linalg.solve`'s forward (`custom_linear_solve`'s
            # `solve` branch) uses `gather`+`select_n` for its negative-index wraparound,
            # not `sort`; `sort` only appears in the `transpose_solve` branch, which is
            # for autodiff and is never traced here. Emission code no node exercises is
            # emission code `jaxpr_validate` never checks, and this backend does not
            # ship paths it has no evidence for.
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
                for j, g in enumerate(grp[1:]):
                    e = f"{fn}({e}, {g})"
                    e = self._break_chain(e, len(grp) - 1, j)
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
            for j, g in enumerate(grp[1:]):
                e = f"({e}{op}{g})"
                e = self._break_chain(e, len(grp) - 1, j)
            out.append(e)
        return (out, None)

    MAX_EXPR_NESTING = 64
    """How deep a left-nested accumulation may get before it is broken by a local.

    Not a style preference. A reduction over n elements builds one expression nested n
    deep, and **CPython's own parser refuses to read more than about 200 levels** --
    `too many nested parentheses`, raised while IMPORTING the generated module, with a
    line number and no function name, so it presents as a whole-module codegen failure
    rather than as anything attributable to a node. The reduction that reaches it is
    `.physics.profiles.parameterisation.pedestal_temperature_profile`'s `reduce_min`
    over 201 profile points, which nests exactly 201 `_min(` deep -- one over the line.
    Breaking the chain into locals changes the NAMES, never the association: the
    partial results are the same partial results in the same order, so the value is
    bit-identical to the nested form."""

    def _break_chain(self, e: str, total: int, j: int) -> str:
        """Bind `e` to a local every `MAX_EXPR_NESTING` steps of an accumulation."""
        if total <= self.MAX_EXPR_NESTING or (j + 1) % self.MAX_EXPR_NESTING:
            return e
        nm = self._fresh()
        self.lines.append(f"    {nm} = {e}")
        return nm

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
    def _apply_gather_map(rows, op_strides, hi, idx_flat, fill: bool = False):
        """`rows` applied to one concrete flattened `start_indices` array.

        Returns one `(src, valid)` pair per row. `fill=False` (CLIP/PROMISE_IN_BOUNDS)
        always clamps and is always `valid` -- the previous behaviour exactly. `fill=
        True` (FILL_OR_DROP) still computes the CLAMPED `src` (so a caller has some
        in-range position to read even when it is about to discard it), but marks the
        row invalid whenever the RAW, unclamped index actually fell outside
        `[0, hi[d]]` on any dimension -- the caller substitutes the primitive's
        `fill_value` there instead of reading `src`.
        """
        out = []
        for base, startpos in rows:
            src, valid = 0, True
            for d, (b, sp) in enumerate(zip(base, startpos)):
                if sp is None:
                    src += b * op_strides[d]
                    continue
                raw = int(idx_flat[sp])
                if fill and not (0 <= raw <= hi[d]):
                    valid = False
                src += (b + min(max(raw, 0), hi[d])) * op_strides[d]
            out.append((src, valid))
        return out

    def _check_gather_map(self, rows, op_strides, hi, operand_shape, indices_shape,
                          dnums, slice_sizes, mode, out_shape, fill_value=None):
        """Prove the derived map by DIFFERENTIAL TEST against `lax.gather` itself.

        The map above is a transcription of XLA's gather semantics, and a transcription
        is exactly the kind of thing that is plausible and wrong. So it is not trusted:
        the gather is run for real, on an operand whose value at every position IS that
        position, over several random in-bounds index draws (plus deliberately
        out-of-range ones under `CLIP` or `FILL_OR_DROP`, where clamping or filling,
        respectively, is the defined behaviour). Any disagreement is a named refusal,
        not a warning.
        """
        from jax import lax

        fill = "FILL_OR_DROP" in str(mode)
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
        if str(mode).endswith("CLIP") or fill:
            wild = draws[0].copy()
            wild[...] = 10 ** 6
            draws.extend((wild, -wild))
        for idx in draws:
            got = np.asarray(lax.gather(
                probe, jnp.asarray(idx, dtype=jnp.int32), dnums,
                tuple(slice_sizes), mode=mode,
                fill_value=fill_value)).reshape(-1)
            pairs = self._apply_gather_map(rows, op_strides, hi, idx.reshape(-1),
                                           fill=fill)
            want = np.asarray(
                [(float(fill_value) if fill and not ok else float(src))
                 for src, ok in pairs])
            mismatch = (got != want) & ~(np.isnan(got) & np.isnan(want))
            if got.shape[0] != want.shape[0] or np.any(mismatch):
                raise Refusal(
                    "gather: the derived index map disagrees with lax.gather at "
                    f"dimension_numbers={dnums}, slice_sizes={tuple(slice_sizes)}, "
                    f"mode={mode}, fill_value={fill_value}")

    def _vec_local(self, v: Value) -> str:
        """Materialise `v` into one `vec{n}f` local and return its identifier -- what a
        runtime index needs, since a scalarised array is a set of unrelated locals and
        cannot be subscripted.

        Unless `v` is ALREADY subscriptable: `array_ident` names a `wp.array`/`vec{n}f`
        identifier this Value is an untouched read of (see `Value`), and that
        identifier is handed back directly -- no fresh local, no copy. This is what
        makes a runtime-indexed lookup against a CONSTANT global table (Change 1) free:
        the table is one `wp.array` kernel parameter regardless of how many places
        index into it."""
        if v.array_ident is not None:
            return v.array_ident
        n = v.size
        if n > MAX_VEC_ELEMENTS:
            raise Refusal(f"runtime-indexed operand of {n} elements "
                          f"(cap {MAX_VEC_ELEMENTS})")
        if v.kind != _F:
            raise Refusal(f"runtime-indexed operand of kind {v.kind!r} -- only float64 "
                          f"vectors are materialised")
        # Two values whose element EXPRESSIONS are the same strings are the same
        # vector, and one copy will do. That is a proof rather than a hope: every
        # expression here is either a literal or a name this emitter bound exactly
        # once and never reassigns, and every vector local lives at function scope, so
        # the earlier one is in scope and holds the same bits.
        #
        # It is not a micro-optimisation. `.physics.plasma_composition` interpolates
        # ELEVEN times against one (14, 200) table and built the table again for each:
        # 11 x vec2800f, 246 KB of per-thread storage and 28,000 statements, for ten
        # copies of one array. It is the largest single per-thread allocation in every
        # configuration measured.
        key = (n, v.exprs)
        hit = self._vec_memo.get(key)
        if hit is not None:
            return hit
        self.vecs.add(n)
        name = self._fresh()
        self._vec_idents.add(name)
        self._vec_memo[key] = name
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
        `int32`).

        `FILL_OR_DROP` is emitted with its ACTUAL semantics -- an out-of-bounds read
        returns `fill_value`, not a clamp -- rather than either refusing or silently
        substituting the CLIP behaviour. This backend does not attempt to prove
        in-boundness from the surrounding computation (that would be reasoning about
        the model, not the primitive); instead every element gets its own runtime
        bounds guard, mirroring `_scatter`'s guarded `wp.where` for the same mode.
        """
        if len(args) != 2:
            raise Refusal(f"gather: {len(args)} operands")
        operand, indices = args
        dnums = eqn.params["dimension_numbers"]
        slice_sizes = tuple(int(s) for s in eqn.params["slice_sizes"])
        mode = eqn.params.get("mode")
        fill = "FILL_OR_DROP" in str(mode)
        fill_value = eqn.params.get("fill_value")
        if fill and fill_value is None:
            raise Refusal("gather in FILL_OR_DROP mode with no fill_value -- refusing "
                          "rather than guessing a default")
        operand_shape = tuple(eqn.invars[0].aval.shape)
        indices_shape = tuple(eqn.invars[1].aval.shape)
        rows, op_strides, hi = self._gather_map(
            operand_shape, indices_shape, dnums, slice_sizes, oshape)
        self._check_gather_map(rows, op_strides, hi, operand_shape, indices_shape,
                               dnums, slice_sizes, mode, oshape, fill_value=fill_value)

        if indices.vals is not None:
            pairs = self._apply_gather_map(rows, op_strides, hi,
                                           np.asarray(indices.vals).reshape(-1),
                                           fill=fill)
            fill_expr = _fmt_scalar(fill_value, okind) if fill else None
            exprs = [(self._coerce(operand.exprs[src], operand.kind, okind) if ok
                      else fill_expr) for src, ok in pairs]
            vals = None
            if operand.vals is not None:
                flat_op = np.asarray(operand.vals, dtype=float).reshape(-1)
                vals = np.asarray(
                    [(flat_op[src] if ok else float(fill_value))
                     for src, ok in pairs]).reshape(oshape)
            return exprs, vals

        if okind != _F:
            raise Refusal(f"gather with a runtime index and a {okind!r} result")
        vec = self._vec_local(operand)
        exprs = []
        for base, startpos in rows:
            const = sum(base[d] * op_strides[d] for d in range(len(base)))
            terms = [f"wp.int32({const})"]
            guards = []
            for d, sp in enumerate(startpos):
                if sp is None:
                    continue
                e = self._coerce(indices.exprs[sp], indices.kind, _I)
                s = self._fresh()
                self.lines.append(f"    {s} = {e}")
                terms.append(f"wp.int32({op_strides[d]}) * wp.min(wp.max({s}, "
                             f"wp.int32(0)), wp.int32({hi[d]}))")
                if fill:
                    guards.append(f"({s} >= wp.int32(0) and {s} <= "
                                  f"wp.int32({hi[d]}))")
            read = f"{vec}[{' + '.join(terms)}]"
            if fill and guards:
                ok = " and ".join(guards)
                exprs.append(f"wp.where({ok}, {read}, {_fmt_scalar(fill_value, okind)})")
            else:
                exprs.append(read)
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

    def _scatter_index_proof(self, eqn, indices, rows, op_strides, hi, clip):
        """`unique_indices=False`, but possibly wrongly conservative: PROVE the actual
        indices are both a compile-time CONSTANT and COLLISION-FREE, rather than
        refusing outright -- see `jnp.diag_indices(n)`, `.tokamak.pf_coil.inductance`'s
        call site. JAX sets `unique_indices` from what the CALLER declared
        (`.at[...].set(...)`'s own `unique_indices` kwarg, default `False`), not from
        anything it proves about the indices -- so `False` here means "not asserted",
        not "actually collides".

        Two independent proofs, both required, neither assumed:

        1. **Constant.** `emit_node` already evaluated this node's jaxpr for real (not
           reasoned about) at its reference point AND at `_INVARIANCE_DRAWS`
           independently jittered ones (`_invariant_vars`); only a variable that came
           out IDENTICAL at every one of them is in `self._concrete`. A value that is
           genuinely independent of every read passes this by construction; one that
           depends on any read generically will not survive even one jitter.
        2. **Collision-free.** A constant array can still repeat an index
           (`[0, 0, 1]`), so the concrete value is run through the SAME
           `_apply_scatter_map`/`clip`-or-drop logic the runtime path uses, and every
           VALID target is checked for a duplicate.

        Returns an `indices` `Value` carrying the proved concrete array as `.vals` (so
        the rest of `_scatter` runs its ordinary static-index path unchanged), or
        `None` if either proof fails -- the caller refuses.
        """
        var = eqn.invars[1]
        cval = np.asarray(var.val) if isinstance(var, Literal) \
            else self._concrete.get(var)
        if cval is None:
            return None
        flat = np.asarray(cval).reshape(-1)
        targets = self._apply_scatter_map(rows, op_strides, hi, flat, clip)
        valid = [t for t, ok in targets if ok]
        if len(valid) != len(set(valid)):
            return None
        return Value(tuple(_fmt_scalar(x, indices.kind) for x in flat.tolist()),
                     indices.kind, indices.shape, flat.reshape(indices.shape))

    def _scatter(self, eqn, args, okind, oshape):
        """`lax.scatter` (overwrite).

        With unique indices every destination is written at most once, so the result is
        a per-element choice between the operand's value and one update's -- no
        accumulation, and no question about the order two writes to one cell would
        take. `scatter_add` and a scatter with a combining function are refused rather
        than given an order this backend would be inventing. A scatter WITHOUT
        `unique_indices` declared is refused too, UNLESS `_scatter_index_proof` can
        prove its actual indices are both constant and collision-free -- see its
        docstring.

        Static indices resolve entirely in the generator. A runtime index becomes one
        `wp.where` per (update, destination) pair, guarded by the same in-bounds test
        XLA uses to drop an out-of-range update -- which is why the flat target alone
        is not enough: an out-of-range component can alias a legal flat position.
        """
        if len(args) != 3:
            raise Refusal(f"scatter: {len(args)} operands")
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

        if not eqn.params.get("unique_indices"):
            proved = self._scatter_index_proof(eqn, indices, rows, op_strides, hi,
                                               clip)
            if proved is None:
                raise Refusal(
                    "scatter without unique_indices (two updates to one cell would "
                    "need a write order this backend does not define, and this call "
                    "site's indices could not be proven both compile-time-constant "
                    "and collision-free)")
            indices = proved

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

    # -- scatter-mul ----------------------------------------------------------

    def _check_scatter_mul_map(self, rows, op_strides, hi, operand_shape,
                               indices_shape, updates_shape, dnums, mode):
        """`_check_scatter_map`'s differential-test discipline, applied to
        `lax.scatter_mul` -- the dimension_numbers/mode index arithmetic `_scatter_map`
        derives is IDENTICAL between plain `scatter` and `scatter-mul` (only the
        combinator differs), so this reuses it and only changes what "applying a row"
        means: multiply into the target rather than overwrite it."""
        from jax import lax

        clip = "CLIP" in str(mode)
        n_up = int(np.prod(updates_shape)) if updates_shape else 1
        operand = jnp.ones(operand_shape, dtype=jnp.float64)
        updates = jnp.arange(2.0, 2.0 + n_up, dtype=jnp.float64).reshape(updates_shape)
        rng = np.random.default_rng(0x5CA11ED)
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
            got = np.asarray(lax.scatter_mul(
                operand, jnp.asarray(idx, dtype=jnp.int32), updates, dnums,
                unique_indices=True, mode=mode)).reshape(-1)
            want = np.ones(int(np.prod(operand_shape)) if operand_shape else 1)
            for u, (tgt, valid) in enumerate(
                    self._apply_scatter_map(rows, op_strides, hi, idx.reshape(-1),
                                            clip)):
                if valid:
                    want[tgt] *= np.asarray(updates).reshape(-1)[u]
            if got.shape[0] != want.shape[0] or np.any(got != want):
                raise Refusal(
                    "scatter-mul: the derived index map disagrees with "
                    f"lax.scatter_mul at dimension_numbers={dnums}, mode={mode}")

    def _scatter_mul(self, eqn, args, okind, oshape):
        """`lax.scatter_mul` -- `x.at[...].multiply(y)`, and `x.at[...].divide(y)`
        (JAX lowers `.divide` to a `scatter-mul` into a `ones_like` array, followed by
        an elementwise divide the ordinary `div` path already handles).

        Refuses everything except the one shape this graph is observed to need: a
        SINGLE shared runtime index tuple (the whole `scatter_indices` operand holds
        exactly one index vector -- no batching) whose window lands on a CONTIGUOUS,
        naturally-ordered run of flat operand positions
        (`.tokamak.cicc_superconducting_tf_coil.tf_stress`'s
        `sig_tf_r.at[wp_slice].multiply(fac_sig_r)`, `wp_slice` a runtime-offset,
        compile-time-SIZED window). That shape is special-cased rather than reusing
        `_scatter`'s per-(update, destination)-pair `wp.where` chain, because here it
        would be `n_updates * n_out` pairs -- 500 * 1500 = 750000 for `tf_stress`, far
        past `MAX_SCATTER_PAIRS`/`MAX_NODE_LINES`. The window shape makes every row's
        RUNTIME target `base(row) + offset` for a SINGLE shared `offset` (batch size
        1), so the map inverts to one `wp.where` per OUTPUT element (`n_out`, not
        `n_updates * n_out`): `offset` is computed once, and which update (if any)
        lands on output `j` is arithmetic (`j - offset - base(0)`), not a search.
        """
        if len(args) != 3:
            raise Refusal(f"scatter-mul: {len(args)} operands")
        if not eqn.params.get("unique_indices"):
            raise Refusal("scatter-mul without unique_indices")
        uj = eqn.params.get("update_jaxpr")
        if uj is None or len(uj.eqns) != 1 or uj.eqns[0].primitive.name != "mul" \
                or len(uj.invars) != 2 or len(uj.outvars) != 1:
            raise Refusal("scatter-mul: update_jaxpr is not a bare binary multiply -- "
                          "refusing rather than assuming what it combines")
        operand, indices, updates = args
        dnums = eqn.params["dimension_numbers"]
        mode = eqn.params.get("mode")
        clip = "CLIP" in str(mode)
        fill = "FILL_OR_DROP" in str(mode)
        if not clip and not fill and "PROMISE_IN_BOUNDS" not in str(mode):
            raise Refusal(f"scatter-mul in mode {mode}")
        operand_shape = tuple(eqn.invars[0].aval.shape)
        indices_shape = tuple(eqn.invars[1].aval.shape)
        updates_shape = tuple(eqn.invars[2].aval.shape)
        if int(np.prod(indices_shape)) != len(dnums.scatter_dims_to_operand_dims):
            raise Refusal("scatter-mul: more than one scatter index tuple (a "
                          "batched scatter-mul is not the shape this backend "
                          "special-cases)")
        rows, op_strides, hi = self._scatter_map(
            operand_shape, indices_shape, updates_shape, dnums)
        self._check_scatter_mul_map(rows, op_strides, hi, operand_shape,
                                    indices_shape, updates_shape, dnums, mode)

        n_updates = len(rows)
        base_flat = [sum(b * s for b, s in zip(base, op_strides))
                    for base, _ in rows]
        if n_updates == 0 or any(base_flat[r] - base_flat[r - 1] != 1
                                 for r in range(1, n_updates)):
            raise Refusal("scatter-mul: update rows do not land on a contiguous, "
                          "naturally-ordered run of operand positions -- refusing "
                          "rather than searching every (update, destination) pair")
        d0 = base_flat[0]
        startpos = rows[0][1]           # identical for every row (batch size 1)
        n_out = operand.size

        if indices.vals is not None:
            off, oob = 0, False
            flat_idx = np.asarray(indices.vals).reshape(-1)
            for d, sp in enumerate(startpos):
                if sp is None:
                    continue
                raw = int(flat_idx[sp])
                if not (0 <= raw <= hi[d]):
                    oob = True
                off += min(max(raw, 0), hi[d]) * op_strides[d]
            exprs = [self._coerce(e, operand.kind, okind) for e in operand.exprs]
            if not (fill and oob):
                for r in range(n_updates):
                    j = d0 + off + r
                    upd = self._coerce(updates.exprs[r], updates.kind, okind)
                    exprs[j] = f"({exprs[j]} * {upd})"
            return exprs, None

        if okind != _F:
            raise Refusal(f"scatter-mul with a runtime index and a {okind!r} result")
        updvec = self._vec_local(updates)
        terms, guards = [], []
        for d, sp in enumerate(startpos):
            if sp is None:
                continue
            e = self._coerce(indices.exprs[sp], indices.kind, _I)
            s = self._fresh()
            self.lines.append(f"    {s} = {e}")
            terms.append(f"wp.int32({op_strides[d]}) * wp.min(wp.max({s}, "
                         f"wp.int32(0)), wp.int32({hi[d]}))")
            if fill:
                guards.append(f"({s} >= wp.int32(0) and {s} <= wp.int32({hi[d]}))")
        off_name = self._fresh()
        self.lines.append(f"    {off_name} = " + " + ".join(terms))
        ok_name = None
        if guards:
            ok_name = self._fresh()
            self.lines.append(f"    {ok_name} = " + " and ".join(guards))
        exprs = []
        for j in range(n_out):
            base_expr = self._coerce(operand.exprs[j], operand.kind, okind)
            lo = f"({off_name} + wp.int32({d0}))"
            hi_excl = f"({off_name} + wp.int32({d0 + n_updates}))"
            in_window = f"(wp.int32({j}) >= {lo} and wp.int32({j}) < {hi_excl})"
            r_expr = f"(wp.int32({j}) - {lo})"
            r_clamped = (f"wp.min(wp.max({r_expr}, wp.int32(0)), "
                        f"wp.int32({n_updates - 1}))")
            cond = in_window if ok_name is None else f"({ok_name} and {in_window})"
            exprs.append(f"wp.where({cond}, ({base_expr} * {updvec}[{r_clamped}]), "
                         f"{base_expr})")
        return exprs, None

    # -- dot_general ------------------------------------------------------------

    def _dot_general_map(self, lhs_shape, rhs_shape, lc, rc, lb, rb, oshape):
        """One list of `(lhs_flat_index, rhs_flat_index)` contraction-term pairs per
        FLAT output position, for `lax.dot_general`'s general (batch dims first, then
        LHS free dims, then RHS free dims) output layout. Shared between `_dot_general`
        (which turns each row into a sum-of-products expression) and
        `_check_dot_general` (which sums the same pairs over CONCRETE probe arrays and
        compares to `lax.dot_general` itself) -- the map is proved once, the same way
        `_gather_map`/`_scatter_map` are."""
        l_free = [d for d in range(len(lhs_shape)) if d not in lc and d not in lb]
        r_free = [d for d in range(len(rhs_shape)) if d not in rc and d not in rb]
        contract_sizes = [lhs_shape[d] for d in lc]
        n_contract = int(np.prod(contract_sizes)) if contract_sizes else 1
        l_idx = np.arange(int(np.prod(lhs_shape))).reshape(lhs_shape)
        r_idx = np.arange(int(np.prod(rhs_shape))).reshape(rhs_shape)
        n_out = int(np.prod(oshape)) if oshape else 1
        nb, nl = len(lb), len(l_free)
        rows = []
        for o_flat in range(n_out):
            oi = np.unravel_index(o_flat, oshape) if oshape else ()
            batch_pos, l_pos, r_pos = oi[:nb], oi[nb:nb + nl], oi[nb + nl:]
            pairs = []
            for c_flat in range(n_contract):
                ci = np.unravel_index(c_flat, contract_sizes) if contract_sizes else ()
                l_full = [0] * len(lhs_shape)
                for j, d in enumerate(lb):
                    l_full[d] = int(batch_pos[j])
                for j, d in enumerate(l_free):
                    l_full[d] = int(l_pos[j])
                for j, d in enumerate(lc):
                    l_full[d] = int(ci[j])
                r_full = [0] * len(rhs_shape)
                for j, d in enumerate(rb):
                    r_full[d] = int(batch_pos[j])
                for j, d in enumerate(r_free):
                    r_full[d] = int(r_pos[j])
                for j, d in enumerate(rc):
                    r_full[d] = int(ci[j])
                pairs.append((int(l_idx[tuple(l_full)]), int(r_idx[tuple(r_full)])))
            rows.append(pairs)
        return rows

    def _check_dot_general(self, lhs_shape, rhs_shape, lc, rc, lb, rb, rows):
        """Prove `_dot_general_map` by DIFFERENTIAL TEST against `lax.dot_general`
        itself, over several random draws -- the same discipline
        `_check_gather_map`/`_check_scatter_map` apply to an index map, applied here
        to a contraction map. A loose tolerance is deliberate: this proves the MAP
        (which elements get multiplied and summed into which output), not bit-exact
        summation order, which `jaxpr_validate`'s whole-kernel agreement number
        (`AGREEMENT_RTOL`) is what actually gates."""
        from jax import lax

        rng = np.random.default_rng(0xD07A6E)
        for _ in range(3):
            a = rng.standard_normal(lhs_shape)
            b = rng.standard_normal(rhs_shape)
            got = np.asarray(lax.dot_general(
                jnp.asarray(a), jnp.asarray(b), ((lc, rc), (lb, rb)))).reshape(-1)
            aflat, bflat = a.reshape(-1), b.reshape(-1)
            want = np.array([sum(aflat[li] * bflat[ri] for li, ri in pairs)
                             for pairs in rows])
            if got.shape[0] != want.shape[0] or \
                    not np.allclose(got, want, rtol=1e-9, atol=1e-9):
                raise Refusal(
                    "dot_general: the derived contraction map disagrees with "
                    f"lax.dot_general at dimension_numbers={((lc, rc), (lb, rb))}")

    def _dot_general(self, eqn, args, okind, oshape):
        """`lax.dot_general`, fully scalarised: every operand element is already its
        own expression, so a contraction is `n_out` independent inner-product sums --
        no runtime work, the same strategy every other shape primitive in this backend
        uses. Handles arbitrary contracting/batch dimensions (`_dot_general_map` is
        general); this graph's own call site (`.power.pf_coil_power`) is a plain
        (7,8)x(8,6)/(7,8)x(8,) matrix-vector product with no batch dims, verified by
        `_check_dot_general` rather than assumed from the shape. `precision` and
        `preferred_element_type` do not affect this: every element is already float64
        and this backend never emits anything but exact float64 arithmetic.
        """
        if len(args) != 2:
            raise Refusal(f"dot_general: {len(args)} operands")
        lhs, rhs = args
        if not lhs.shape or not rhs.shape:
            raise Refusal("dot_general: scalar operand")
        (lc, rc), (lb, rb) = eqn.params["dimension_numbers"]
        lc, rc, lb, rb = tuple(lc), tuple(rc), tuple(lb), tuple(rb)
        lhs_shape, rhs_shape = tuple(lhs.shape), tuple(rhs.shape)
        if [lhs_shape[d] for d in lc] != [rhs_shape[d] for d in rc]:
            raise Refusal("dot_general: contracting dimension sizes do not match")
        if [lhs_shape[d] for d in lb] != [rhs_shape[d] for d in rb]:
            raise Refusal("dot_general: batch dimension sizes do not match")
        l_free = [d for d in range(len(lhs_shape)) if d not in lc and d not in lb]
        r_free = [d for d in range(len(rhs_shape)) if d not in rc and d not in rb]
        want_shape = (tuple(lhs_shape[d] for d in lb) + tuple(lhs_shape[d] for d in l_free)
                     + tuple(rhs_shape[d] for d in r_free))
        if want_shape != tuple(oshape):
            raise Refusal(f"dot_general: derived output shape {want_shape} does not "
                          f"match the declared {tuple(oshape)}")
        rows = self._dot_general_map(lhs_shape, rhs_shape, lc, rc, lb, rb, oshape)
        self._check_dot_general(lhs_shape, rhs_shape, lc, rc, lb, rb, rows)

        out = []
        for pairs in rows:
            terms = [f"({self._coerce(lhs.exprs[li], lhs.kind, okind)} * "
                    f"{self._coerce(rhs.exprs[ri], rhs.kind, okind)})"
                    for li, ri in pairs]
            e = terms[0]
            for t in terms[1:]:
                e = f"({e} + {t})"
            out.append(e)
        return out

    # -- triangular_solve ----------------------------------------------------

    MAX_TRIANGULAR_SOLVE_N = 12
    """Same reasoning as `_lu.MAX_LU_N` -- these two only ever appear together (this
    is `jnp.linalg.solve`'s other half), so the cap is the same size."""

    def _triangular_solve(self, eqn, args, okind, oshape):
        """`jax.lax.linalg.triangular_solve`: solve `op(a) @ x = b` for `x`, `a`
        triangular and fixed-size, by straight-line forward/back substitution --
        the textbook algorithm, unrolled the same way `_lu` unrolls elimination.

        `op(a)` is `a` or `a^T` (`transpose_a`); `a`'s OWN triangle (`lower`) plus
        whether it is transposed together decide which triangle of the linear system
        actually being solved is lower: `eff_lower = lower != transpose_a` (a lower
        matrix transposed is upper, and vice versa). Substitution runs forward over a
        lower system, backward over an upper one; `unit_diagonal` skips the division
        (the diagonal is defined to be 1 and is not read). `conjugate_a` is read from
        `eqn.params` but not branched on: every operand in this graph is float64, and
        conjugation of a real number is the identity, so there is nothing for it to
        change here -- this is a statement about the graph's dtypes, not an
        unconditional assumption about `triangular_solve` in general.

        **Verified against `lax.linalg.triangular_solve` directly**, not assumed: a
        pure-Python transcription of the loop below matched it to ~2e-15 relative
        over 200 draws at each of the 8 (`lower`, `transpose_a`, `unit_diagonal`)
        combinations, sizes 3..8, 1..3 right-hand sides, magnitudes spanning four
        decades -- see the working notes for this change.
        """
        a, b = args
        if len(a.shape) != 2 or a.shape[0] != a.shape[1]:
            raise Refusal(f"triangular_solve: operand `a` of shape {a.shape}, not "
                          f"square")
        n = a.shape[0]
        if n > self.MAX_TRIANGULAR_SOLVE_N:
            raise Refusal(f"triangular_solve: {n}x{n} matrix (cap "
                          f"{self.MAX_TRIANGULAR_SOLVE_N})")
        if len(b.shape) != 2 or b.shape[0] != n:
            raise Refusal(f"triangular_solve: operand `b` of shape {b.shape} against "
                          f"a {n}x{n} `a`")
        if a.kind != _F or b.kind != _F:
            raise Refusal("triangular_solve: non-float64 operand")
        if not eqn.params.get("left_side", True):
            raise Refusal("triangular_solve: left_side=False")
        lower = bool(eqn.params["lower"])
        transpose_a = bool(eqn.params["transpose_a"])
        unit_diagonal = bool(eqn.params["unit_diagonal"])
        k = b.shape[1]

        def A(i, j):
            return a.exprs[(j * n + i) if transpose_a else (i * n + j)]

        eff_lower = lower != transpose_a
        order = range(n) if eff_lower else range(n - 1, -1, -1)
        X = [[None] * k for _ in range(n)]
        for i in order:
            js = range(0, i) if eff_lower else range(i + 1, n)
            for col in range(k):
                acc = b.exprs[i * k + col]
                for j in js:
                    t = self._fresh()
                    self.lines.append(f"    {t} = {acc} - {A(i, j)} * {X[j][col]}")
                    acc = t
                if unit_diagonal:
                    X[i][col] = acc
                else:
                    t = self._fresh()
                    self.lines.append(f"    {t} = {acc} / {A(i, i)}")
                    X[i][col] = t
        return [X[i][col] for i in range(n) for col in range(k)]

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


def _concrete_env(jaxpr, consts, invals) -> dict:
    """`{Var: ndarray}` for every variable `jaxpr` binds, from a REAL, eager
    evaluation at concrete `invals`/`consts` -- `eqn.primitive.bind(...)`, the exact
    mechanism JAX's own eager mode uses, not a hand-derived reinterpretation of any
    primitive's semantics.

    Used ONLY as a proof aid for `_scatter`'s `unique_indices=False` case (see
    `_scatter_index_proof`): it never influences what code gets emitted for a value
    that isn't proved constant, and it is never consulted for code generation itself
    -- `Value.vals`/`.exprs` remain the only things `_materialise` ever sees.

    Best-effort: if any equation's concrete `bind` fails (nothing here is expected
    to, since every primitive this graph traces already runs under ordinary JAX
    execution), evaluation stops and whatever was resolved so far is returned --
    silently narrowing what can be proved, never wrongly claiming a value is
    constant.
    """
    env: dict = {}
    for v, c in zip(jaxpr.constvars, consts):
        env[v] = c
    for v, a in zip(jaxpr.invars, invals):
        env[v] = a

    def read(v):
        return v.val if isinstance(v, Literal) else env[v]

    for eqn in jaxpr.eqns:
        try:
            in_vals = [read(v) for v in eqn.invars]
            out = eqn.primitive.bind(*in_vals, **eqn.params)
        except Exception:
            break
        if not eqn.primitive.multiple_results:
            out = [out]
        for v, o in zip(eqn.outvars, out):
            env[v] = o
    return env


def _jittered(values, draw: int):
    """`values` perturbed for the `draw`-th independence check `_invariant_vars`
    runs. Integer/boolean-valued reads are left untouched -- perturbing what is
    genuinely an index would risk an out-of-range read failing `_concrete_env`
    outright (narrowing the proof) rather than testing anything about data
    dependence; every float read is nudged by a distinct, fixed, non-zero amount so
    a truly data-INDEPENDENT variable comes out identical and a data-dependent one
    (generically) does not."""
    out = []
    for v in values:
        a = np.asarray(v)
        if np.issubdtype(a.dtype, np.integer) or a.dtype == np.bool_:
            out.append(a)
        else:
            out.append(a * (1.0 + 1.0e-3 * (draw + 1)) + 1.0e-6 * (draw + 1))
    return out


_INVARIANCE_DRAWS = 3
"""Independent jittered re-evaluations `_invariant_vars` requires to agree, beyond
the reference point -- the same "several draws, not one" discipline
`_check_gather_map`/`_check_scatter_map` already apply to an index MAP; this applies
it to whether one equation's VALUE is data-dependent at all."""


def _invariant_vars(jaxpr, consts, values, force_float: bool) -> dict:
    """`{Var: ndarray}`, restricted to variables that evaluated to the EXACT SAME
    value at the reference point and at `_INVARIANCE_DRAWS` independently jittered
    ones -- i.e. variables proved data-INDEPENDENT by direct experiment, not by
    reasoning about the jaxpr's shape. See `_concrete_env`."""
    cast = (lambda vs: [jnp.asarray(v, dtype=jnp.float64) for v in vs]) if force_float \
        else (lambda vs: [jnp.asarray(v) for v in vs])
    try:
        envs = [_concrete_env(jaxpr, consts, cast(values))]
        for k in range(_INVARIANCE_DRAWS):
            envs.append(_concrete_env(jaxpr, consts, cast(_jittered(values, k))))
    except Exception:
        return {}
    ref = envs[0]
    others = envs[1:]
    out = {}
    for v, val in ref.items():
        if all(v in e and np.array_equal(np.asarray(e[v]), np.asarray(val))
               for e in others):
            out[v] = np.asarray(val)
    return out


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


def emit_node(fn, func_name: str, values, n_outputs: int,
              const_mask: tuple[bool, ...] | None = None) -> EmitInfo:
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

    `const_mask`, aligned with `values`/`jaxpr.invars` positionally, marks a read as
    CONSTANT across the whole batch (see `_constant_varpaths`): its parameter is a
    `wp.array(dtype=wp.float64)` GLOBAL instead of a `vec{n}f` VALUE. Nothing about the
    body changes -- element access is still `a{i}[j]`, which indexes a `wp.array` just
    as natively as it indexes a vector -- so the only difference this makes is at the
    two boundaries: no per-thread copy is needed to construct the argument, and a
    runtime-indexed use (`_gather`'s `_vec_local`) can read the parameter directly
    rather than materialising one more copy of it. Defaults to "nothing is constant",
    which reduces this to the previous behaviour exactly.
    """
    force_float = True
    try:
        closed = trace_node(fn, values, force_float=True)
    except Exception:
        # Not a tracing failure to report: some reads are genuinely integer-typed
        # (array indices), and JAX refuses a float64 tracer where an index is
        # required. Re-trace at the values' own dtypes. If THAT fails too, the
        # exception is real and propagates.
        force_float = False
        closed = trace_node(fn, values, force_float=False)
    jaxpr, consts = closed.jaxpr, list(closed.consts)
    if const_mask is not None and len(const_mask) != len(jaxpr.invars):
        raise Refusal(f"const_mask has {len(const_mask)} entries for "
                      f"{len(jaxpr.invars)} parameters")

    em = _FuncEmitter()
    # `_scatter`'s only use for this: proving a `unique_indices=False` call site's
    # indices are actually a data-independent constant (see `_invariant_vars`).
    # Computed only when a top-level `scatter` equation without `unique_indices`
    # actually appears -- every other node (including the largest jaxprs this
    # backend handles, tens of thousands of equations) pays nothing for it.
    if any(eqn.primitive.name == "scatter" and not eqn.params.get("unique_indices")
           for eqn in jaxpr.eqns):
        em._concrete = _invariant_vars(jaxpr, consts, values, force_float)
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
            if const_mask is not None and const_mask[i]:
                params.append(f"{p}: wp.array(dtype=wp.float64)")
                args.append(Value(tuple(f"{p}[{j}]" for j in range(n)), _F, shape,
                                  array_ident=p))
                continue
            vec_lengths.add(n)
            params.append(f"{p}: {vec_name(n)}")
            args.append(Value(tuple(f"{p}[{j}]" for j in range(n)), _F, shape,
                              array_ident=p))
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
        if o.kind == _F and o.array_ident in em._vec_idents:
            # The value is ALREADY exactly one `vec{n}f` local this body built (a fused
            # loop nest's output, or a materialised runtime-indexed operand) and
            # nothing has been computed from it since -- that is what `array_ident`
            # asserts. Copying it element by element into a second vector of the same
            # length would be n statements to produce a bit-identical value.
            ret.append(o.array_ident)
            ret_types.append(vec_name(n))
            out_sizes.append(n)
            continue
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


def _constant_varpaths(sub, unknowns: set, boundary: set) -> set:
    """Every VarPath in `sub` (`drive.body.subgraph`) that the unknowns cannot reach --
    computed STRUCTURALLY, by walking `reads`/`owns` on `sub.definitions`, never by
    inspecting a value and never by a path looking table-shaped.

    Walks the FULL graph of node definitions, not just the resolved `entries` a
    particular emission run produced: an earlier version of this analysis
    (`scratchpad/arrays2/hoist.py`) built its producer map from resolved entries alone,
    so a REFUSED node's output had no producer and silently defaulted to "does not
    vary" -- answering the exact question being asked with an artefact of what
    happened to trace. Here a VarPath with no owning definition (an artefact of the
    same kind, or truly nothing in this subgraph produces it) is NOT assumed constant:
    unresolvable provenance is treated as VARYING, because being constant is a claim
    that needs a proof chain back to a boundary or nothing, and the absence of a chain
    is not that proof.
    """
    owner = {}
    for defn in sub.definitions.values():
        for v in getattr(defn, "owns", ()) or ():
            owner[v.path_str()] = defn

    memo: dict = {}

    def varies(var: str) -> bool:
        if var in unknowns:
            return True
        if var in boundary:
            return False
        if var in memo:
            return memo[var]
        memo[var] = True                      # cycle guard: unresolved -> assume varies
        defn = owner.get(var)
        if defn is None:
            r = True                          # no owner found -> cannot prove constant
        else:
            r = any(varies(x.path_str()) for x in (getattr(defn, "reads", ()) or ()))
        memo[var] = r
        return r

    const = set()
    for defn in sub.definitions.values():
        for v in (tuple(getattr(defn, "reads", ()) or ())
                  + tuple(getattr(defn, "owns", ()) or ())):
            p = v.path_str()
            if p not in const and not varies(p):
                const.add(p)
    return const


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
    unknown_paths = {u.path_str() for u in drive.unknowns}
    boundary_paths = {v.path_str() for v in drive.context}
    const_paths = _constant_varpaths(sub, unknown_paths, boundary_paths)

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
        read_paths = tuple(v.path_str() for v in defn.reads)
        const_mask = tuple(p in const_paths for p in read_paths)
        try:
            info = emit_node(fn, func_name, vals, len(defn.owns), const_mask=const_mask)
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
                inputs=read_paths,
                outputs=tuple(v.path_str() for v in defn.owns),
                source=info.source,
                n_eqns=info.n_eqns,
                input_sizes=info.input_sizes,
                output_sizes=info.output_sizes,
                helpers=info.helpers,
                vec_lengths=info.vec_lengths,
                # Only array-valued (n > 1) constant reads actually got a `wp.array`
                # parameter in `info` -- `emit_node` only consults `const_mask` in that
                # branch, a scalar constant stays the ordinary `wp.float64` column. A
                # scalar path here would tell a caller (`_validation_kernel`,
                # `build_kernel_source`) to bind a `wp.array` that was never emitted.
                const_inputs=frozenset(
                    p for p, c, n in zip(read_paths, const_mask, info.input_sizes)
                    if c and n > 1),
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
