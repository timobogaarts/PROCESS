"""Fixed-length array SSA: one straight-line JAX function whose array-valued locals
are N *named scalars*, with every call it makes inlined.

The problem this exists for: `.physics.plasma_composition`'s node body
(`functional_process/cottax/physics/composition.py`) assembles a 14-entry species
array out of twelve scalar leaf-inputs plus two placeholders, hands it to a pure
function that `jax.vmap`s a per-species table interpolation over it, sums slices of
it, overwrites two entries with `.at[i].set()`, and destructures the result. Warp has
no whole-array elementwise arithmetic, no `vmap`, and no first-class functions -- but
N is 14, fixed, and every index in the chain is a Python literal, so the whole thing
is expansible into named scalars with no loss and no guess.

**The output is runnable JAX, not Warp.** That is deliberate: the expansion can then
be checked against the real node function (`node_def.fn(...)`) in pure Python, before
any Warp codegen exists to confuse a disagreement with a codegen bug. The Warp
transpiler consumes the same expanded source afterwards.

Three exactness commitments, each measured rather than assumed (`probe_numerics.py`,
`probe_exp.py`):

- **`jnp.sum` over <= 14 float64 elements is a left-to-right sequential sum, bitwise**
  (0 mismatches in 2000 random draws at n=12 and n=14; at n=64 XLA switches to a
  different association and sequential no longer matches -- so this expansion is
  refused above `MAX_SEQUENTIAL_SUM`).
- **`x ** <int literal>` is expanded into repeated multiplication**, because that is
  what JAX itself does (`lax.integer_pow`) and what `wp.pow` does *not* do. Leaving
  the `**` in place is the one known last-bit divergence the tracked transpiler
  records (§80); expanding it here removes it rather than tolerating it.
- **`jnp.interp(jnp.log(x), jnp.log(<2-D table>[i]), <2-D table>[i])` is left standing
  as an expression**, for the Warp side to pattern-match into a per-row helper. It is
  emitted in that exact shape so the JAX reference and the Warp kernel are reading the
  same three arguments.

What is *not* handled raises `ScalariseError` by name. In particular: any control flow
(`if`/`for`/`while`) inside a function being inlined, a slice with a non-literal bound,
an array whose length two operands disagree on, and any array-valued value reaching a
`return` position this walk did not build.
"""
from __future__ import annotations

import ast
import copy
import inspect
import textwrap

MAX_SEQUENTIAL_SUM = 32
"""Above this length `jnp.sum`'s association is no longer left-to-right sequential
(measured: exact at n=12/14, divergent at n=64) -- refuse rather than emit a sum that
is arithmetically right and bitwise wrong."""


class ScalariseError(Exception):
    """A construct this expansion does not cover -- refused, never guessed at."""


# --------------------------------------------------------------------------- values

class Val:
    pass


class S(Val):
    """A scalar, as an `ast` expression."""

    def __init__(self, expr):
        self.expr = expr


class A(Val):
    """A fixed-length array, as a list of scalar `ast` expressions."""

    def __init__(self, elts):
        self.elts = list(elts)

    def __len__(self):
        return len(self.elts)


class ArrParam(Val):
    """A parameter that stays a real array on the Warp side (a `wp.array` /
    `wp.array2d` kernel argument), indexed rather than expanded.

    `ndim` 1 -> `name[i]` is a scalar; `ndim` 2 -> `name[i]` is a `Row`.
    """

    def __init__(self, name, shape):
        self.name = name
        self.shape = tuple(shape)
        self.ndim = len(self.shape)


class Row(Val):
    """One row of a 2-D `ArrParam` -- `name[i]`, still an array, never expanded."""

    def __init__(self, param: ArrParam, index: int):
        self.param = param
        self.index = index

    def expr(self):
        return ast.Subscript(
            value=ast.Name(self.param.name, ast.Load()),
            slice=ast.Constant(self.index),
            ctx=ast.Load(),
        )


class T(Val):
    """A tuple of values (a multi-return, or a literal tuple)."""

    def __init__(self, items):
        self.items = list(items)


class Lam(Val):
    """A callable bound to a name: a `lambda`, or a function object to inline.

    `bound_kwargs` carries a `functools.partial`'s already-supplied keyword arguments
    -- `PlasmaCompositionNonIgnited` builds its arm that way, and the value it binds
    (`f_nd_beam_electron`) is one of the node's own traced inputs, not a constant, so
    it has to survive as a `Val` rather than be frozen.
    """

    def __init__(self, obj, env=None, fdef=None, globalns=None, bound_kwargs=None):
        self.obj = obj
        self.env = env or {}
        self.fdef = fdef
        self.globalns = globalns
        self.bound_kwargs = dict(bound_kwargs or {})


# ----------------------------------------------------------------------- ast helpers

def _name(n):
    return ast.Name(n, ast.Load())


def _num(v):
    return ast.Constant(v)


def _call(fn_expr, args, keywords=()):
    return ast.Call(func=fn_expr, args=list(args), keywords=list(keywords))


def _attr(base, *parts):
    node = _name(base)
    for p in parts:
        node = ast.Attribute(value=node, attr=p, ctx=ast.Load())
    return node


def _is_jnp_call(node, *attrs):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in ("jnp", "np")
        and node.func.attr in attrs
    )


def _literal_int(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
    ):
        return -node.operand.value
    return None


def _fdef_of(fn):
    src = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(src)
    node = tree.body[0]
    if not isinstance(node, ast.FunctionDef):
        raise ScalariseError(f"{fn!r}: source is not a single `def`")
    body = node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        node.body = body[1:]
    return node


# ------------------------------------------------------------------------- the walk

class Scalariser:
    """Expands one function into straight-line scalar statements.

    `array_params`: `{param_name: length}` for a parameter that becomes N named
    scalars. `keep_array_params`: `{param_name: shape}` for one that stays a real
    array kernel argument (indexed, never expanded).
    """

    def __init__(self, self_obj=None):
        self.stmts: list = []
        self.globals_merged: dict = {}
        self.self_obj = self_obj
        self._tmp = 0

    # -- naming -------------------------------------------------------------

    def _fresh(self, base):
        self._tmp += 1
        return f"_s{self._tmp}_{base}"

    # -- globals ------------------------------------------------------------

    def _merge_globals(self, globalns, fdef):
        """Record every free global name `fdef` uses, refusing a genuine collision
        (the same name bound to two different objects in two inlined functions --
        which would make the merged namespace the transpiler resolves against
        silently wrong for one of them)."""
        bound = set()
        for n in ast.walk(fdef):
            if isinstance(n, ast.arg):
                bound.add(n.arg)
            elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                bound.add(n.id)
        for n in ast.walk(fdef):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in bound:
                if n.id in globalns:
                    prev = self.globals_merged.get(n.id, _MISSING)
                    new = globalns[n.id]
                    if prev is not _MISSING and prev is not new:
                        raise ScalariseError(
                            f"global name {n.id!r} means two different objects in the "
                            f"inlined chain ({prev!r} vs {new!r}) -- refusing to merge"
                        )
                    self.globals_merged[n.id] = new

    # -- emission -----------------------------------------------------------

    def _emit_assign(self, target_name, expr):
        self.stmts.append(
            ast.Assign(targets=[ast.Name(target_name, ast.Store())], value=expr)
        )
        return _name(target_name)

    def _materialise(self, val, base):
        """Bind a value to fresh name(s) so the emitted source stays linear rather
        than one exponentially-nested expression."""
        if isinstance(val, S):
            if isinstance(val.expr, (ast.Name, ast.Constant)):
                return val
            return S(self._emit_assign(self._fresh(base), val.expr))
        if isinstance(val, A):
            out = []
            for i, e in enumerate(val.elts):
                if isinstance(e, (ast.Name, ast.Constant)):
                    out.append(e)
                else:
                    out.append(self._emit_assign(self._fresh(f"{base}__{i}"), e))
            return A(out)
        if isinstance(val, T):
            return T([self._materialise(v, f"{base}_{i}") for i, v in enumerate(val.items)])
        return val

    # -- statements ---------------------------------------------------------

    def run_body(self, body, env):
        """Expand a straight-line body; returns the `Val` of its `return`."""
        for stmt in body:
            if isinstance(stmt, ast.Return):
                return self.ev(stmt.value, env)
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                continue  # a bare docstring/constant expression
            if isinstance(stmt, ast.Pass):
                continue
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                if isinstance(stmt, ast.AnnAssign):
                    targets, value = [stmt.target], stmt.value
                else:
                    targets, value = stmt.targets, stmt.value
                if len(targets) != 1:
                    raise ScalariseError("chained assignment (`a = b = ...`)")
                val = self._materialise(self.ev(value, env), _target_base(targets[0]))
                self._bind(targets[0], val, env)
                continue
            raise ScalariseError(
                f"statement {type(stmt).__name__} is not straight-line "
                f"(`{ast.unparse(stmt)[:70]}`) -- refusing to expand it"
            )
        raise ScalariseError("function body has no `return`")

    def _bind(self, target, val, env):
        if isinstance(target, ast.Name):
            env[target.id] = val
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            if not isinstance(val, T):
                raise ScalariseError(
                    f"tuple-unpacking `{ast.unparse(target)}` from a non-tuple value"
                )
            if len(target.elts) != len(val.items):
                raise ScalariseError(
                    f"tuple-unpacking arity mismatch: {len(target.elts)} names, "
                    f"{len(val.items)} values"
                )
            for t, v in zip(target.elts, val.items):
                self._bind(t, v, env)
            return
        raise ScalariseError(f"assignment target {ast.unparse(target)!r}")

    # -- expressions --------------------------------------------------------

    def ev(self, node, env) -> Val:
        if node is None:
            raise ScalariseError("bare `return` with no value")
        m = getattr(self, f"_ev_{type(node).__name__}", None)
        if m is None:
            raise ScalariseError(f"expression {type(node).__name__} "
                                 f"(`{ast.unparse(node)[:70]}`)")
        return m(node, env)

    def _ev_Constant(self, node, env):
        return S(node)

    def _ev_Name(self, node, env):
        if node.id in env:
            return env[node.id]
        target = self.globals_lookup(node.id)
        if inspect.isfunction(target):
            # A function named but not called -- `self._composition(
            # plasma_composition_ignited, ...)`. It is the thing to monomorphise on,
            # not a value to render.
            return Lam(target)
        return S(_name(node.id))  # a module global -- left for `_collect_globals`

    def _ev_Attribute(self, node, env):
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            return self._self_attr(node.attr)
        # `constants.M_DEUTERON_AMU` and friends: left standing, resolved downstream
        # by the transpiler's own global collector.
        if isinstance(node.value, ast.Name) and node.value.id not in env:
            return S(node)
        raise ScalariseError(f"attribute access `{ast.unparse(node)}`")

    def _self_attr(self, attr):
        """`self.<attr>` on the node instance being expanded.

        A method is something to inline; an `eqx.field(static=True)` scalar is a
        frozen literal; a static *sequence* of numbers is a fixed-length array of
        frozen literals (which is what makes `imp_indices`-shaped fields usable --
        their values are graph-assembly-time facts, not state). Anything else refuses
        by name rather than being rendered on a guess."""
        if self.self_obj is None:
            raise ScalariseError(f"`self.{attr}` with no bound instance to read it from")
        try:
            val = getattr(self.self_obj, attr)
        except AttributeError:
            raise ScalariseError(f"`self.{attr}` does not exist on "
                                 f"{type(self.self_obj).__name__}")
        if _is_bound_method(val) or inspect.isfunction(val):
            return Lam(val)
        if isinstance(val, bool):
            return S(_num(val))
        if isinstance(val, (int, float)):
            return S(_num(val))
        if isinstance(val, (tuple, list)) and val and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in val):
            return A([_num(v) for v in val])
        raise ScalariseError(
            f"`self.{attr}` is {type(val).__name__} -- not a method, a frozen scalar, "
            f"or a frozen numeric sequence; refusing to render it")

    def _ev_UnaryOp(self, node, env):
        v = self.ev(node.operand, env)
        if isinstance(v, S):
            return S(ast.UnaryOp(op=node.op, operand=v.expr))
        if isinstance(v, A):
            return A([ast.UnaryOp(op=node.op, operand=e) for e in v.elts])
        raise ScalariseError(f"unary op on {type(v).__name__}")

    def _ev_BinOp(self, node, env):
        lv, rv = self.ev(node.left, env), self.ev(node.right, env)
        if isinstance(node.op, ast.Pow):
            return self._pow(lv, node.right, rv)
        return self._elementwise(lambda a, b: ast.BinOp(left=a, op=node.op, right=b),
                                 lv, rv, ast.unparse(node))

    def _pow(self, lv, exponent_node, rv):
        """`x ** <int literal>` -> repeated multiplication, exactly what JAX's own
        `lax.integer_pow` lowering does; `wp.pow(x, 2.0)` does not, and that
        divergence is the tracked transpiler's own recorded §80 last-bit issue."""
        k = _literal_int(exponent_node)
        if k is None or not (0 <= k <= 8):
            raise ScalariseError(
                f"`** {ast.unparse(exponent_node)}` -- only a small non-negative "
                f"integer literal exponent is expanded (a float exponent would have "
                f"to go through `wp.pow`, which does not round like JAX)"
            )

        def one(e):
            if k == 0:
                return _num(1.0)
            out = e
            for _ in range(k - 1):
                out = ast.BinOp(left=out, op=ast.Mult(), right=copy.deepcopy(e))
            return out

        if isinstance(lv, S):
            return S(one(lv.expr))
        if isinstance(lv, A):
            return A([one(e) for e in lv.elts])
        raise ScalariseError("`**` on a non-scalar/non-array value")

    def _elementwise(self, build, lv, rv, what):
        if isinstance(lv, S) and isinstance(rv, S):
            return S(build(lv.expr, rv.expr))
        if isinstance(lv, A) and isinstance(rv, A):
            if len(lv) != len(rv):
                raise ScalariseError(f"length mismatch {len(lv)} vs {len(rv)} in `{what}`")
            return A([build(a, b) for a, b in zip(lv.elts, rv.elts)])
        if isinstance(lv, A) and isinstance(rv, S):
            return A([build(a, copy.deepcopy(rv.expr)) for a in lv.elts])
        if isinstance(lv, S) and isinstance(rv, A):
            return A([build(copy.deepcopy(lv.expr), b) for b in rv.elts])
        raise ScalariseError(
            f"`{what}`: operands are {type(lv).__name__}/{type(rv).__name__} -- an "
            f"array kept as a real Warp array cannot take part in whole-array "
            f"arithmetic; index it first"
        )

    def _ev_Compare(self, node, env):
        if len(node.ops) != 1:
            raise ScalariseError("chained comparison")
        lv = self.ev(node.left, env)
        rv = self.ev(node.comparators[0], env)
        return self._elementwise(
            lambda a, b: ast.Compare(left=a, ops=[node.ops[0]], comparators=[b]),
            lv, rv, ast.unparse(node))

    def _ev_Tuple(self, node, env):
        items = []
        for e in node.elts:
            if isinstance(e, ast.Starred):
                inner = self.ev(e.value, env)
                if isinstance(inner, T):
                    items.extend(inner.items)
                elif isinstance(inner, A):
                    items.extend(S(x) for x in inner.elts)
                else:
                    raise ScalariseError(f"`*{ast.unparse(e.value)}` on "
                                         f"{type(inner).__name__}")
            else:
                items.append(self.ev(e, env))
        return T(items)

    _ev_List = _ev_Tuple

    def _ev_Lambda(self, node, env):
        return Lam(None, env=dict(env), fdef=node)

    def _ev_Subscript(self, node, env):
        base = self.ev(node.value, env)
        sl = node.slice
        # A module-level `slice` constant (`IMPURITY_SLICE = slice(2, 14)`) used as an
        # index -- the same thing as a literal `[2:14]`, written once and named.
        if isinstance(sl, ast.Name) and sl.id not in env:
            const = self.globals_lookup(sl.id)
            if isinstance(const, slice):
                if const.step is not None:
                    raise ScalariseError(f"strided slice constant {sl.id!r}")
                sl = ast.Slice(
                    lower=None if const.start is None else _num(const.start),
                    upper=None if const.stop is None else _num(const.stop),
                    step=None,
                )
        if isinstance(sl, ast.Slice):
            lo, hi, step = sl.lower, sl.upper, sl.step
            if step is not None:
                raise ScalariseError(f"strided slice `{ast.unparse(node)}`")
            if isinstance(base, A):
                n = len(base)
            elif isinstance(base, T):
                n = len(base.items)
            elif isinstance(base, ArrParam) and base.ndim == 1:
                n = base.shape[0]
            else:
                raise ScalariseError(f"slice of {type(base).__name__} "
                                     f"`{ast.unparse(node)}`")
            lo_i = 0 if lo is None else self._const_index(lo, env, n)
            hi_i = n if hi is None else self._const_index(hi, env, n)
            if isinstance(base, A):
                return A(base.elts[lo_i:hi_i])
            if isinstance(base, ArrParam):
                return A([ast.Subscript(value=_name(base.name), slice=_num(i),
                                        ctx=ast.Load())
                          for i in range(lo_i, hi_i)])
            return T(base.items[lo_i:hi_i])
        # A GATHER by a fixed index array (`arr[jnp.array(self.imp_indices)]`) -- the
        # index array is itself a frozen numeric sequence this walk already expanded,
        # so every index is a literal and the gather is a permutation/selection known
        # at expansion time.
        gather = self._const_index_vector(sl, env)
        if gather is not None:
            if isinstance(base, A):
                for i in gather:
                    if not (0 <= i < len(base)):
                        raise ScalariseError(
                            f"gather index {i} out of range for length {len(base)}")
                return A([base.elts[i] for i in gather])
            if isinstance(base, ArrParam):
                if list(gather) == list(range(base.shape[0])):
                    # The identity gather over the whole leading axis -- the array is
                    # unchanged, so it stays one real Warp array parameter. A genuine
                    # SUBSET of a 2-D parameter's rows would need a different
                    # representation (a row-index vector carried into the kernel);
                    # refused below rather than silently ignored.
                    return base
                raise ScalariseError(
                    f"gather of {len(gather)} row(s) out of {base.shape[0]} from the "
                    f"array parameter {base.name!r} -- only the identity gather over "
                    f"the whole axis is expanded; a genuine subset would change the "
                    f"parameter's shape and is refused rather than guessed")
            raise ScalariseError(f"gather of {type(base).__name__} "
                                 f"`{ast.unparse(node)}`")
        idx = self._const_index(sl, env, None)
        if isinstance(base, A):
            if not (0 <= idx < len(base)):
                raise ScalariseError(f"index {idx} out of range for length {len(base)}")
            return S(base.elts[idx])
        if isinstance(base, T):
            return base.items[idx]
        if isinstance(base, ArrParam):
            if base.ndim == 1:
                return S(ast.Subscript(value=_name(base.name), slice=_num(idx),
                                       ctx=ast.Load()))
            return Row(base, idx)
        raise ScalariseError(f"subscript of {type(base).__name__} "
                             f"`{ast.unparse(node)}`")

    def _const_index(self, node, env, n):
        """A slice/index bound that must be a Python literal at expansion time. A
        module-level `H_INDEX`/`IMPURITY_SLICE`-style constant counts (it resolves to
        a real `int`/`slice` object); anything runtime-valued does not."""
        k = _literal_int(node)
        if k is not None:
            return k if (n is None or k >= 0) else n + k
        if isinstance(node, ast.Name) and node.id not in env:
            val = self.globals_lookup(node.id)
            if isinstance(val, int) and not isinstance(val, bool):
                return val if (n is None or val >= 0) else n + val
        raise ScalariseError(
            f"index/slice bound `{ast.unparse(node)}` is not a compile-time literal"
        )

    def _const_index_vector(self, node, env):
        """`node` as a list of compile-time integer indices, or `None` if it is not
        one -- an already-expanded frozen sequence (`A` of integer constants), or a
        module/`self` constant that resolves to a tuple/list of ints."""
        val = None
        if isinstance(node, ast.Name):
            if node.id in env:
                val = env[node.id]
            else:
                g = self.globals_lookup(node.id)
                if isinstance(g, (tuple, list)):
                    val = A([_num(v) for v in g])
        elif isinstance(node, ast.Call):
            try:
                val = self.ev(node, env)
            except ScalariseError:
                return None
        if not isinstance(val, A):
            return None
        out = []
        for e in val.elts:
            k = _literal_int(e)
            if k is None:
                return None
            out.append(k)
        return out

    def globals_lookup(self, name):
        return self.globals_merged.get(name, _MISSING)

    # -- calls ---------------------------------------------------------------

    def _ev_Call(self, node, env):
        f = node.func

        # `x.at[i].set(e)` -- an out-of-place element update.
        if (isinstance(f, ast.Attribute) and f.attr == "set"
                and isinstance(f.value, ast.Subscript)
                and isinstance(f.value.value, ast.Attribute)
                and f.value.value.attr == "at"):
            base = self.ev(f.value.value.value, env)
            if not isinstance(base, A):
                raise ScalariseError(
                    f"`.at[...].set(...)` on {type(base).__name__} -- only a "
                    f"fixed-length expanded array supports it"
                )
            idx = self._const_index(f.value.slice, env, len(base))
            new = self.ev(node.args[0], env)
            if not isinstance(new, S):
                raise ScalariseError("`.at[i].set(...)` with a non-scalar value")
            elts = list(base.elts)
            elts[idx] = new.expr
            return A(elts)

        # `jax.vmap(fn, in_axes=...)(...)`
        if isinstance(f, ast.Call) and _is_vmap(f):
            return self._vmap(f, node, env)

        if _is_jnp_call(node, "stack", "array", "asarray", "concatenate"):
            return self._jnp_assemble(node, env)
        if _is_jnp_call(node, "sum"):
            return self._jnp_sum(node, env)
        if _is_jnp_call(node, "zeros_like", "ones_like", "full_like"):
            return self._jnp_like(node, env)

        # `functools.partial(F, kw=<expr>)` -- a callable with some arguments already
        # supplied. Kept as a value, not called here.
        if _is_partial(node):
            inner = self.ev(node.args[0], env)
            if not isinstance(inner, Lam):
                raise ScalariseError(
                    f"`functools.partial` over `{ast.unparse(node.args[0])}`, which "
                    f"this walk cannot resolve to a function")
            if len(node.args) > 1:
                raise ScalariseError("`functools.partial` with positional arguments")
            kw = dict(inner.bound_kwargs)
            for k in node.keywords:
                if k.arg is None:
                    raise ScalariseError("`functools.partial(**kwargs)`")
                kw[k.arg] = self.ev(k.value, env)
            return Lam(inner.obj, env=inner.env, fdef=inner.fdef, bound_kwargs=kw)

        # A callable bound in `env` (a lambda parameter, or a function passed by
        # value) -- inline it.
        if isinstance(f, ast.Name) and isinstance(env.get(f.id), Lam):
            return self._inline(env[f.id], node, env)

        # `self.<method>(...)` on the node instance -- inline it.
        if (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                and f.value.id == "self"):
            return self._inline(self._self_attr(f.attr), node, env)

        # A same-package function this walk can read -- inline it too.
        if isinstance(f, ast.Name):
            target = self.globals_lookup(f.id)
            if inspect.isfunction(target):
                return self._inline(Lam(target), node, env)

        # Anything else: a scalar library call (`jnp.where`, `jnp.maximum`,
        # `jnp.log`, `jnp.interp`, ...). Every argument must already be scalar, with
        # the one exception of `jnp.interp`'s table arguments, which stay arrays.
        return self._scalar_call(node, env)

    def _jnp_assemble(self, node, env):
        if len(node.args) != 1:
            raise ScalariseError(f"`{ast.unparse(node.func)}` with "
                                 f"{len(node.args)} arguments")
        arg = node.args[0]
        if isinstance(arg, (ast.List, ast.Tuple)):
            vals = [self.ev(e, env) for e in arg.elts]
            if all(isinstance(v, S) for v in vals):
                return A([v.expr for v in vals])
            if node.func.attr == "concatenate" and all(isinstance(v, A) for v in vals):
                out = []
                for v in vals:
                    out.extend(v.elts)
                return A(out)
            raise ScalariseError(f"`{ast.unparse(node.func)}` over mixed shapes")
        inner = self.ev(arg, env)
        if isinstance(inner, (A, ArrParam, Row, S)):
            return inner  # `jnp.asarray(x)` on something already the right shape
        raise ScalariseError(f"`{ast.unparse(node.func)}` of {type(inner).__name__}")

    def _jnp_sum(self, node, env):
        if node.keywords:
            raise ScalariseError("`jnp.sum` with keyword arguments (axis/dtype/where)")
        v = self.ev(node.args[0], env)
        if not isinstance(v, A):
            raise ScalariseError(f"`jnp.sum` of {type(v).__name__}")
        if len(v) > MAX_SEQUENTIAL_SUM:
            raise ScalariseError(
                f"`jnp.sum` over {len(v)} elements -- above {MAX_SEQUENTIAL_SUM}, "
                f"XLA's reduction is no longer a left-to-right sequential sum "
                f"(measured) and expanding it that way would be bitwise wrong"
            )
        if not v.elts:
            raise ScalariseError("`jnp.sum` of an empty array")
        # Bind each addend first, then chain: keeps the emitted source linear and the
        # association explicitly left-to-right.
        acc = self._materialise(A(v.elts), "sum_term").elts
        out = acc[0]
        for e in acc[1:]:
            out = ast.BinOp(left=out, op=ast.Add(), right=e)
        return S(out)

    def _jnp_like(self, node, env):
        v = self.ev(node.args[0], env)
        attr = node.func.attr
        if attr == "zeros_like":
            fill = _num(0.0)
        elif attr == "ones_like":
            fill = _num(1.0)
        else:
            f = self.ev(node.args[1], env)
            if not isinstance(f, S):
                raise ScalariseError("`jnp.full_like` with a non-scalar fill")
            fill = f.expr
        if isinstance(v, S):
            return S(copy.deepcopy(fill))
        if isinstance(v, A):
            return A([copy.deepcopy(fill) for _ in v.elts])
        raise ScalariseError(f"`jnp.{attr}` of {type(v).__name__}")

    def _vmap(self, vmap_call, apply_call, env):
        """`jax.vmap(f, in_axes=(...))(a0, a1, ...)` over a fixed axis length ->
        that many explicit calls. `in_axes` entries must be `None` (broadcast) or
        `0` (map over the leading axis); anything else is refused."""
        if len(vmap_call.args) < 1:
            raise ScalariseError("`jax.vmap` with no function argument")
        fn_node = vmap_call.args[0]
        in_axes = None
        for kw in vmap_call.keywords:
            if kw.arg == "in_axes":
                in_axes = kw.value
            elif kw.arg == "out_axes":
                if _literal_int(kw.value) != 0:
                    raise ScalariseError("`jax.vmap(out_axes=...)` other than 0")
            else:
                raise ScalariseError(f"`jax.vmap` keyword {kw.arg!r}")
        if in_axes is None and len(vmap_call.args) > 1:
            in_axes = vmap_call.args[1]
        args = [self.ev(a, env) for a in apply_call.args]
        if apply_call.keywords:
            raise ScalariseError("`jax.vmap(...)` applied with keyword arguments")
        if in_axes is None:
            axes = [0] * len(args)
        elif isinstance(in_axes, (ast.Tuple, ast.List)):
            axes = []
            for e in in_axes.elts:
                if isinstance(e, ast.Constant) and e.value is None:
                    axes.append(None)
                else:
                    k = _literal_int(e)
                    if k != 0:
                        raise ScalariseError(f"`in_axes` entry {ast.unparse(e)}")
                    axes.append(0)
        else:
            k = _literal_int(in_axes)
            if k != 0:
                raise ScalariseError(f"`in_axes={ast.unparse(in_axes)}`")
            axes = [0] * len(args)
        if len(axes) != len(args):
            raise ScalariseError(
                f"`in_axes` has {len(axes)} entries for {len(args)} arguments")

        lengths = set()
        for ax, v in zip(axes, args):
            if ax is None:
                continue
            if isinstance(v, A):
                lengths.add(len(v))
            elif isinstance(v, ArrParam):
                lengths.add(v.shape[0])
            else:
                raise ScalariseError(
                    f"`jax.vmap` mapped argument is {type(v).__name__}, not a "
                    f"fixed-length array")
        if len(lengths) != 1:
            raise ScalariseError(f"`jax.vmap` mapped axes disagree on length: "
                                 f"{sorted(lengths)}")
        n = lengths.pop()

        # Resolve the mapped function once.
        if isinstance(fn_node, ast.Name) and isinstance(env.get(fn_node.id), Lam):
            lam = env[fn_node.id]
        elif isinstance(fn_node, ast.Name):
            target = self.globals_lookup(fn_node.id)
            if not inspect.isfunction(target):
                raise ScalariseError(f"`jax.vmap` over {fn_node.id!r}, which is not a "
                                     f"readable Python function")
            lam = Lam(target)
        elif isinstance(fn_node, ast.Lambda):
            lam = Lam(None, env=dict(env), fdef=fn_node)
        else:
            raise ScalariseError(f"`jax.vmap` over `{ast.unparse(fn_node)}`")

        out = []
        for i in range(n):
            slice_args = []
            for ax, v in zip(axes, args):
                if ax is None:
                    slice_args.append(v)
                elif isinstance(v, A):
                    slice_args.append(S(copy.deepcopy(v.elts[i])))
                else:  # ArrParam
                    slice_args.append(S(ast.Subscript(value=_name(v.name),
                                                      slice=_num(i), ctx=ast.Load()))
                                      if v.ndim == 1 else Row(v, i))
            r = self._inline_values(lam, slice_args, {})
            if not isinstance(r, S):
                raise ScalariseError("`jax.vmap` body did not produce one scalar per "
                                     "element")
            out.append(r.expr)
        return self._materialise(A(out), "vmap")

    def _scalar_call(self, node, env):
        args, keywords = [], []
        for a in node.args:
            v = self.ev(a, env)
            args.append(_as_expr(v, ast.unparse(a)))
        for kw in node.keywords:
            if kw.arg is None:
                raise ScalariseError(f"`**kwargs` in `{ast.unparse(node)[:60]}`")
            v = self.ev(kw.value, env)
            keywords.append(ast.keyword(arg=kw.arg, value=_as_expr(v, ast.unparse(kw.value))))
        return S(ast.Call(func=copy.deepcopy(node.func), args=args, keywords=keywords))

    # -- inlining ------------------------------------------------------------

    def _inline(self, lam, call_node, env):
        args = [self.ev(a, env) for a in call_node.args]
        kwargs = {}
        for kw in call_node.keywords:
            if kw.arg is None:
                raise ScalariseError("`**kwargs` at an inlined call site")
            kwargs[kw.arg] = self.ev(kw.value, env)
        return self._inline_values(lam, args, kwargs)

    def _inline_values(self, lam, args, kwargs):
        if lam.bound_kwargs:
            merged = dict(lam.bound_kwargs)
            clash = set(merged) & set(kwargs)
            if clash:
                raise ScalariseError(f"`functools.partial` argument(s) {sorted(clash)} "
                                     f"supplied twice")
            merged.update(kwargs)
            kwargs = merged
        if lam.fdef is not None and isinstance(lam.fdef, ast.Lambda):
            params = [a.arg for a in lam.fdef.args.args]
            if lam.fdef.args.defaults or lam.fdef.args.kwonlyargs:
                raise ScalariseError("lambda with defaults/keyword-only parameters")
            inner = dict(lam.env)
            inner.update(_bind_params(params, [], args, kwargs, lam.fdef))
            return self.ev(lam.fdef.body, inner)

        fn = lam.obj
        raw = getattr(fn, "__func__", fn)   # `equinox` BoundMethod / plain bound method
        fdef = _fdef_of(raw)
        self._merge_globals(getattr(raw, "__globals__", {}), fdef)
        if _is_bound_method(fn):
            fdef.args.args = [a for a in fdef.args.args if a.arg not in ("self", "cls")]
        pos = [a.arg for a in fdef.args.posonlyargs] + [a.arg for a in fdef.args.args]
        kwonly = [a.arg for a in fdef.args.kwonlyargs]
        if fdef.args.vararg or fdef.args.kwarg:
            raise ScalariseError(f"{fn.__name__}: `*args`/`**kwargs`")
        if fdef.args.defaults or any(d is not None for d in fdef.args.kw_defaults):
            raise ScalariseError(f"{fn.__name__}: default argument value(s) -- a "
                                 f"caller may rely on one, and dropping it is a guess")
        inner = _bind_params(pos, kwonly, args, kwargs, fdef)
        return self.run_body(fdef.body, inner)


_MISSING = object()


def _as_expr(val, what):
    if isinstance(val, S):
        return val.expr
    if isinstance(val, Row):
        return val.expr()
    if isinstance(val, ArrParam):
        return _name(val.name)
    raise ScalariseError(
        f"argument `{what}` is {type(val).__name__} -- a scalar library call cannot "
        f"take an expanded fixed-length array; it would have to be indexed first")


def _bind_params(pos, kwonly, args, kwargs, fdef):
    if len(args) > len(pos):
        raise ScalariseError(
            f"{getattr(fdef, 'name', 'lambda')}: {len(args)} positional arguments for "
            f"{len(pos)} parameters")
    env = dict(zip(pos, args))
    for p in pos[len(args):] + kwonly:
        if p not in kwargs:
            raise ScalariseError(
                f"{getattr(fdef, 'name', 'lambda')}: parameter {p!r} not supplied")
        env[p] = kwargs[p]
    extra = set(kwargs) - set(pos) - set(kwonly)
    if extra:
        raise ScalariseError(f"unexpected keyword argument(s) {sorted(extra)}")
    return env


def _target_base(target):
    if isinstance(target, ast.Name):
        return target.id
    return "t"


def _is_bound_method(obj):
    """A bound method -- Python's own, or `equinox`'s `BoundMethod` wrapper (which is
    a `Module`, so `inspect.ismethod` says no while `__func__`/`__self__` are both
    there and mean exactly what they mean on a plain bound method)."""
    return hasattr(obj, "__func__") and hasattr(obj, "__self__")


def _is_partial(node):
    f = node.func
    if isinstance(f, ast.Name) and f.id == "partial":
        return True
    return (isinstance(f, ast.Attribute) and f.attr == "partial"
            and isinstance(f.value, ast.Name) and f.value.id == "functools")


def _is_vmap(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "vmap"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in ("jax", "jnp")
    )


# --------------------------------------------------------------------------- driver

def scalarise_function(fn, name, param_kinds, bound=None):
    """Expand `fn` into one flat, straight-line function of scalar parameters.

    `param_kinds`: `{param_name: kind}` for each of `fn`'s own parameters --
    `"scalar"`, `("expand", n)` (becomes `n` named scalar parameters
    `<name>__0 .. <name>__{n-1}`), or `("array", shape)` (stays one real array
    parameter, indexed rather than expanded).
    `bound`: `{param_name: <Python function object>}` for a parameter that is a
    function passed by value -- monomorphised away, and dropped from the signature.

    Returns `(fdef, globalns, param_names, n_returns)`. `fdef` is runnable JAX
    (`ast.unparse` it, `exec` it against `globalns`) -- which is how it gets checked
    against the real node before any Warp codegen exists.
    """
    sc = Scalariser(self_obj=getattr(fn, "__self__", None))
    fdef = _fdef_of(fn)
    sc._merge_globals(getattr(fn, "__globals__", {}), fdef)
    bound = bound or {}

    pos = [a.arg for a in fdef.args.posonlyargs] + [a.arg for a in fdef.args.args]
    pos = [p for p in pos if p not in ("self", "cls")]
    kwonly = [a.arg for a in fdef.args.kwonlyargs]

    env, params = {}, []
    for p in pos + kwonly:
        if p in bound:
            env[p] = Lam(bound[p])
            continue
        kind = param_kinds.get(p, "scalar")
        if kind == "scalar":
            env[p] = S(_name(p))
            params.append(p)
        elif isinstance(kind, tuple) and kind[0] == "expand":
            n = kind[1]
            names = [f"{p}__{i}" for i in range(n)]
            env[p] = A([_name(x) for x in names])
            params.extend(names)
        elif isinstance(kind, tuple) and kind[0] == "array":
            env[p] = ArrParam(p, kind[1])
            params.append(p)
        else:
            raise ScalariseError(f"parameter kind {kind!r} for {p!r}")

    result = sc.run_body(fdef.body, env)
    flat = _flatten(result)
    out_fdef = ast.FunctionDef(
        name=name,
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=p) for p in params],
                           vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None,
                           defaults=[]),
        body=sc.stmts + [ast.Return(value=ast.Tuple(elts=flat, ctx=ast.Load())
                                    if len(flat) != 1 else flat[0])],
        decorator_list=[], returns=None, type_params=[],
    )
    ast.fix_missing_locations(out_fdef)
    return out_fdef, dict(sc.globals_merged), params, len(flat)


def _flatten(val):
    if isinstance(val, S):
        return [val.expr]
    if isinstance(val, A):
        return list(val.elts)
    if isinstance(val, T):
        out = []
        for v in val.items:
            out.extend(_flatten(v))
        return out
    raise ScalariseError(f"a {type(val).__name__} reached the return position -- only "
                         f"scalars and expanded fixed-length arrays can be returned")
