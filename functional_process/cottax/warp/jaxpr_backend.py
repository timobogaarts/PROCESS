"""**The jaxpr backend**: one `@wp.func` per graph node, emitted from the node's own
jaxpr rather than from a source-level match against "the function it really is".

This replaces `resolve.py` outright. The resolver's whole taxonomy of refusals --
`Composition`, `self.<attr>`, "ambiguous among 4 candidate calls", arity mismatch,
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

    def dependencies(self) -> tuple:
        return self.inputs


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
    "cos": "wp.cos", "tan": "wp.tan", "tanh": "wp.tanh", "asin": "wp.asin",
    "acos": "wp.acos", "atan": "wp.atan", "sinh": "wp.sinh", "cosh": "wp.cosh",
    "floor": "wp.floor", "ceil": "wp.ceil", "round": "wp.round", "sign": "wp.sign",
    "cbrt": "wp.cbrt", "erf": "wp.erf",
}
"""jax primitive -> a Warp builtin with the same one-argument meaning."""

_BINARY_OP = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
_CMP_OP = {"eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}
_BOOL_OP = {"and": "and", "or": "or"}

_IDENTITY_FLAT = {"squeeze", "expand_dims", "copy", "stop_gradient", "device_put",
                  "optimization_barrier", "reshape"}
"""Primitives that permute nothing in row-major order: the flat element list of the
output IS the flat element list of the input. (`reshape` only when its `dimensions`
parameter is `None`; a `reshape` that carries a permutation is handled with
`transpose`.)"""


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

    def _fresh(self) -> str:
        self._n += 1
        return f"t{self._n}"

    def _materialise(self, exprs, kind, shape, vals=None) -> Value:
        """Bind each element expression to its own Warp local. Doing this at every
        equation (rather than substituting expressions into each other) keeps the
        emitted source linear in the jaxpr's size instead of exponential in its
        depth."""
        names = []
        for e in exprs:
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

    def _bcast(self, v: Value, n: int, kind: str) -> list[str]:
        """`v`'s elements coerced to `kind` and stretched to length `n`.

        Only a size-1 operand stretches: `lax` elementwise primitives require equal
        shapes, but a jaxpr equation may still carry a bare scalar literal alongside an
        array operand. Any other mismatch is a refusal, not a numpy-style broadcast --
        guessing which axis was meant is exactly the kind of plausible-but-wrong the
        refusal rule exists for."""
        if v.size == n:
            return [self._coerce(e, v.kind, kind) for e in v.exprs]
        if v.size == 1:
            return [self._coerce(v.exprs[0], v.kind, kind)] * n
        raise Refusal(f"operand of {v.size} elements against a result of {n} -- "
                      f"only a scalar operand is stretched")

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
        for eqn in jaxpr.eqns:
            self._eqn(env, eqn)
        return [self._read(env, o) for o in jaxpr.outvars]

    def _eqn(self, env, eqn) -> None:
        name = eqn.primitive.name

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
            dim = eqn.params.get("dimension", 0)
            base, blocks, vblocks = 0, [], []
            for a in args:
                # A rank-0 operand cannot be concatenated as-is; `lax.concatenate`
                # itself never sees one, but `stack` reaches here with scalars. Give
                # each a length-1 axis so the index arithmetic below is uniform.
                sh = a.shape if a.shape else (1,)
                blocks.append(np.arange(base, base + a.size).reshape(sh))
                base += a.size
                vblocks.append(None if a.vals is None
                               else np.asarray(a.vals).reshape(sh))
            sel = np.concatenate(blocks, axis=dim).reshape(-1)
            pool = [e for a in args for e in a.exprs]
            kinds = [k for a in args for k in [a.kind] * a.size]
            vals = (None if any(v is None for v in vblocks)
                    else np.concatenate(vblocks, axis=dim))
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
            vals = [self._bcast(c, n_out, okind) for c in cases]
            preds = self._bcast(pred, n_out, pred.kind)
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

        if name in ("gather", "dynamic_update_slice", "scatter", "scatter_add",
                    "sort", "while", "scan", "cond", "argsort", "top_k",
                    "dot_general", "conv_general_dilated", "cumsum_p"):
            raise Refusal(name)

        # ---------- element-wise -----------------------------------------------
        return (self._elementwise(name, eqn, args, okind, n_out), None)

    # -- reductions ---------------------------------------------------------

    def _reduce(self, name, eqn, args, okind, oshape):
        a = args[0]
        axes = tuple(eqn.params.get("axes", ()))
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

    def _elementwise(self, name, eqn, args, okind, n_out) -> list[str]:
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
            L = self._bcast(args[0], n_out, okind)
            R = self._bcast(args[1], n_out, okind)
            op = _BINARY_OP[name]
            return [f"({L[i]} {op} {R[i]})" for i in range(n_out)]

        if name == "neg":
            A = self._bcast(args[0], n_out, okind)
            return [f"(-{a})" for a in A]

        if name == "abs":
            A = self._bcast(args[0], n_out, okind)
            return [f"wp.abs({a})" for a in A]

        if name in ("max", "min"):
            L = self._bcast(args[0], n_out, okind)
            R = self._bcast(args[1], n_out, okind)
            return [f"wp.{name}({L[i]}, {R[i]})" for i in range(n_out)]

        if name == "pow":
            L = self._bcast(args[0], n_out, _F)
            R = self._bcast(args[1], n_out, _F)
            return [f"wp.pow({L[i]}, {R[i]})" for i in range(n_out)]

        if name == "integer_pow":
            A = self._bcast(args[0], n_out, _F)
            y = int(eqn.params["y"])
            return [self._integer_pow(a, y) for a in A]

        if name in ("rem", "mod"):
            L = self._bcast(args[0], n_out, okind)
            R = self._bcast(args[1], n_out, okind)
            if okind == _F:
                return [f"wp.mod({L[i]}, {R[i]})" for i in range(n_out)]
            return [f"({L[i]} % {R[i]})" for i in range(n_out)]

        if name == "atan2":
            L = self._bcast(args[0], n_out, _F)
            R = self._bcast(args[1], n_out, _F)
            return [f"wp.atan2({L[i]}, {R[i]})" for i in range(n_out)]

        if name == "nextafter":
            raise Refusal("nextafter")

        if name == "clamp":
            LO = self._bcast(args[0], n_out, _F)
            X = self._bcast(args[1], n_out, _F)
            HI = self._bcast(args[2], n_out, _F)
            return [f"wp.clamp({X[i]}, {LO[i]}, {HI[i]})" for i in range(n_out)]

        if name == "square":
            A = self._bcast(args[0], n_out, _F)
            return [f"({a} * {a})" for a in A]

        if name == "rsqrt":
            A = self._bcast(args[0], n_out, _F)
            return [f"(wp.float64(1.0) / wp.sqrt({a}))" for a in A]

        if name == "logistic":
            A = self._bcast(args[0], n_out, _F)
            return [f"(wp.float64(1.0) / (wp.float64(1.0) + wp.exp(-{a})))" for a in A]

        if name == "log1p":
            raise Refusal("log1p (Warp has no log1p, and log(1+x) is a different "
                          "function near zero)")
        if name == "expm1":
            raise Refusal("expm1 (Warp has no expm1, and exp(x)-1 is a different "
                          "function near zero)")

        if name in _UNARY:
            if n != 1:
                raise Refusal(f"{name}: expected 1 argument, got {n}")
            A = self._bcast(args[0], n_out, _F)
            return [f"{_UNARY[name]}({a})" for a in A]

        if name in _CMP_OP:
            k = _F if _F in (args[0].kind, args[1].kind) else \
                (_I if _I in (args[0].kind, args[1].kind) else _B)
            L = self._bcast(args[0], n_out, k)
            R = self._bcast(args[1], n_out, k)
            op = _CMP_OP[name]
            return [f"({L[i]} {op} {R[i]})" for i in range(n_out)]

        if name in _BOOL_OP:
            if args[0].kind != _B or args[1].kind != _B:
                raise Refusal(f"{name}: bitwise {name} on non-boolean operands")
            L = self._bcast(args[0], n_out, _B)
            R = self._bcast(args[1], n_out, _B)
            return [f"({L[i]} {_BOOL_OP[name]} {R[i]})" for i in range(n_out)]

        if name == "not":
            if args[0].kind != _B:
                raise Refusal("not: bitwise not on a non-boolean operand")
            return [f"(not {a})" for a in self._bcast(args[0], n_out, _B)]

        if name == "xor":
            if args[0].kind != _B or args[1].kind != _B:
                raise Refusal("xor: bitwise xor on non-boolean operands")
            L = self._bcast(args[0], n_out, _B)
            R = self._bcast(args[1], n_out, _B)
            return [f"({L[i]} != {R[i]})" for i in range(n_out)]

        if name == "is_finite":
            return [f"wp.isfinite({a})" for a in self._bcast(args[0], n_out, _F)]

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


def emit_node(fn, func_name: str, values, n_outputs: int) -> tuple[str, int, int]:
    """`(source, n_returns, n_eqns)` -- the `@wp.func` text for one node.

    Raises `Refusal` (naming the primitive or shape) rather than emitting anything it
    cannot emit correctly.
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
    params = []
    args = []
    for i, iv in enumerate(jaxpr.invars):
        shape = tuple(iv.aval.shape)
        if (int(np.prod(shape)) if shape else 1) != 1:
            # An array-valued READ. Internal arrays are scalarised freely, but a
            # PARAMETER is bound by `emit.py` from one float64 kernel column per
            # VarPath, so there is nowhere to put N values. Refused by name rather
            # than inlined as a constant: freezing a boundary that the kernel is
            # supposed to be parameterised by would compute a plausible wrong answer
            # at every point but the one it was traced at.
            raise Refusal(f"parameter {i} is array-valued, shape {shape} -- the "
                          f"kernel binds one float64 column per read")
        p = f"a{i}"
        # Every parameter is `wp.float64` regardless of the traced kind: that is the
        # kernel's ABI (`emit.py` binds every argument from a float64 column, and Warp
        # does not promote). A parameter the trace typed as an integer is converted
        # back inside the body -- exact, because the column carries an integral value.
        params.append(f"{p}: wp.float64")
        k = _kind(iv.aval)
        if k == _F:
            args.append(Value((p,), _F, shape))
        elif k == _I:
            args.append(Value((f"wp.int32({p})",), _I, shape))
        else:
            args.append(Value((f"({p} != wp.float64(0.0))",), _B, shape))

    outs = em.emit(jaxpr, consts, args)
    if len(outs) != n_outputs:
        raise Refusal(f"traced to {len(outs)} output(s) but the node owns {n_outputs}")
    for j, o in enumerate(outs):
        if o.size != 1:
            raise Refusal(f"output {j} is array-valued, shape {o.shape} -- the kernel "
                          f"binds one float64 column per owned VarPath")

    ret = [em._to_float(o.exprs[0], o.kind) for o in outs]
    n_eqns = len(em.lines)

    head = f"@wp.func\ndef {func_name}({', '.join(params)})"
    head += " -> wp.float64:\n" if len(ret) == 1 else ":\n"
    body = "\n".join(em.lines) if em.lines else ""
    tail = f"    return {', '.join(ret)}\n"
    src = head + (body + "\n" if body else "") + tail
    return src, len(ret), n_eqns


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


def node_values(cold, defn, real_values=None):
    """One concrete value per `defn.reads`, in order.

    In priority order: ground truth from the PROCESS run (`sand_harness.ground_truth`),
    a value already propagated through the Drive (`real_values`), the `DataStructure`
    field's own declared default (which is where PROCESS's real array lengths and its
    fixed per-species tables actually live), and only then a scalar `1.0`.

    The value only has to be *representative* -- it fixes the trace's branch selection
    and its shapes -- but the shape has to be RIGHT, so the `DataStructure` step sits
    ahead of the scalar fallback rather than after it.
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
        if v is None and real_values is not None:
            v = real_values.get(path)
        if v is None:
            v = _datastructure_default(path)
        out.append(jnp.asarray(1.0 if v is None else v))
    return out


def jaxpr_leaves(config: str, drive=None, cold=None, real_values=None):
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

        drive, _ = _assemble(config)
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
        vals = node_values(cold, defn, real_values)
        func_name = _sanitise(node, i)
        try:
            src, n_ret, n_eqns = emit_node(fn, func_name, vals, len(defn.owns))
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
                source=src,
                n_eqns=n_eqns,
            )
        )
    return entries, refused, drive


def funcs_source(entries) -> str:
    """Every entry's `@wp.func`, in order, de-duplicated by function name."""
    seen, out = set(), []
    for e in entries:
        if e.fn in seen:
            continue
        seen.add(e.fn)
        out.append(e.source)
    return "\n\n".join(out)
