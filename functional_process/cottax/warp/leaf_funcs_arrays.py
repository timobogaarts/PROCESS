"""The array-parameter transpiler feature: what stands between the partial SAND kernel
(`leaf_funcs.py`, everything stamped `wp.float64`) and covering the array-valued
lookup-table leaves it currently refuses by name (`§76 Work bucket`).

Built ON TOP of `leaf_funcs.py` (imported, never copied) -- reuses its
`_fdef_of`/`_is_models_fn`/`_collect_globals`/`_RewriteListLookups`/`_RewriteBitwiseBool`/
`_RewriteInComparisons`/`_SpecializeHOF`/`REGISTRY`/`PROVIDED`/`MULTI_RETURN_HELPERS`,
and the tracked `ToWarp`/`Unsupported` it wraps. Only the NEW pieces live here:

1. **Deciding array vs. scalar, per parameter -- derived, not guessed.** A leaf's
   dynamic parameters are bound to real `VarPath`s; `decide_array_params` looks up each
   one's REAL value (from PROCESS's own cold-start ground truth, the same oracle
   `harness.py` already uses for its agreement check) and asks whether it is
   shaped `()`/`(1,)` (scalar) or `(n>1,)` (array). If a bound parameter's real value
   is unavailable, this REFUSES the whole leaf rather than assume -- exactly the
   existing rule (`Unsupported`/`LeafError`), extended to a new question ("what shape is
   this port?") instead of a new guess.

2. **`arr[idx]` for a genuine array parameter.** Warp indexes a `wp.array` with a
   runtime scalar index natively; the only translation work is (a) stripping the
   `jnp.asarray(...)` PROCESS itself wraps these in, and (b) casting the index
   expression to `wp.int32` (Warp's `array.__getitem__` has no float overload; every
   index this codebase produces is a small-magnitude switch stored as an exact
   integral `wp.float64`, so `wp.int32(<float expr>)` is an exact truncation, not a
   rounding guess -- confirmed per-function at validation time, not assumed).

3. **`jnp.interp` -- exact, via a static loop.** `jax._src.numpy.lax_numpy._interp`'s
   own body (read directly, not guessed) is:
       i = clip(searchsorted(xp, x, side='right'), 1, len(xp)-1)
       f = where(|dx|<=eps, fp[i-1], fp[i-1] + (x-xp[i-1])/dx * (fp[i]-fp[i-1]))
       f = where(x < xp[0], fp[0], f); f = where(x > xp[-1], fp[-1], f)
   `searchsorted(..., side='right')` is "count of xp[k] <= x", which is exactly a
   static `for k in range(N)` loop when `N = len(xp)` is known at generation time --
   the ONLY loop shape Warp differentiates correctly (dynamic-length loops are not
   replayed in the backward pass, per the brief). `N` is derived the same way as (1):
   from the real table's length, never assumed. One `@wp.func` (`wp_interp_N`) is
   generated per distinct `N` seen, taking the tables as genuine `wp.array` params --
   so it works identically whether `xp`/`fp` are a call's own array PARAMETERS (none
   of the leaves this file was built for hit that case) or MODULE-LEVEL CONSTANT
   tables baked in at assembly (`calculate_quench_protection_current_density`'s
   `_TEMP_K`/`_Q_HE_ARRAY_SA2M4`/`_Q_CU_ARRAY_SA2M4`) -- the latter are materialised
   as real `wp.array`s once, at kernel-build time, and threaded in as extra trailing
   arguments the emitter must also supply (`extra_table_args`, below).

**What this deliberately does NOT attempt**: `pchip_interp` (the C1 monotone
interpolant `intersect_residual`/`intersect` depend on) is NOT a call to `jnp.interp` --
its `_pchip_slopes` helper is whole-array Fritsch-Carlson slope arithmetic
(`xp[1:] - xp[:-1]`, boolean masks, a `concatenate`) that would need a genuinely
different feature (unrolling whole-array elementwise ops into N scalar temporaries, not
just "index/interp with a known N") to transpile at all -- reusing THIS file's `wp_interp_N`
machinery would silently produce the wrong interpolant (piecewise-LINEAR, not monotone
cubic), which is exactly the kind of guess this design refuses to make. Separately,
its real tables (`.stellarator.wp_width_r`/`.lhs`/`.rhs`) have **no ground-truth value at
all** (`_ground_truth` raises: "no IN.DAT line sets it and it is not in
DATACLASS_DEFAULTS") -- they are computed upstream of this SAND block by nodes outside
it, so `decide_array_params` cannot even ask the question for that leaf. Both blockers
are reported, neither is guessed past.
"""
from __future__ import annotations

import ast
import copy
import math
import re


import numpy as np

from . import leaf_funcs as LF  # the scalar builder this file extends, read-only
from .leaf_funcs import (  # noqa: F401  (re-exported for callers)
    LeafError, PROVIDED, REGISTRY, REGISTRY_ARITY, MULTI_RETURN_HELPERS,
    Unsupported, _fdef_of, _is_models_fn, _local_names, _collect_globals,
    _RewriteListLookups, _RewriteBitwiseBool, _RewriteInComparisons, _SpecializeHOF,
    _Specialized, _const_ident,
)


def _jnp_pi(src: str) -> str:
    """Mirrors `build_leaf_funcs_source`'s own local helper of the same name (it is
    nested, not importable) -- `jnp.pi`/`np.pi`/`math.pi` folded to a literal."""
    for mod_alias in ("jnp", "np", "math"):
        src = re.sub(rf"(?<![\w.]){mod_alias}\.pi(?!\w)",
                      f"wp.float64({math.pi!r})", src)
    return src


SYNTH_LEAVES: dict = {}
"""`{(module, fn): _SynthLeaf}` -- leaves that have no Python source of their own
because they were BUILT (`scalarise.scalarise_function`): a node whose real body
assembles a fixed-length species array, `jax.vmap`s over it and destructures the
result, expanded into one flat straight-line function of named scalars.

Registered by `resolve.py` at resolution time (the same walk that decides the leaf's
identity also decides its expanded body, so the two cannot drift), consumed here
instead of `importlib.import_module(...)`/`getattr`. Keyed exactly like any other
leaf, so `known_bad`, the arity invariant and the round-by-round exclusion loop all
apply to it unchanged."""


class _SynthLeaf:
    """A built leaf: its already-expanded `fdef`, the merged namespace its free names
    resolve against, and `{param: shape}` for every parameter that stays a real Warp
    array (1-D `wp.array`, 2-D `wp.array2d`) rather than a `wp.float64`."""

    def __init__(self, fdef, globalns, array_shapes, n_returns):
        self.fdef = fdef
        self.globalns = globalns
        self.array_shapes = dict(array_shapes)
        self.n_returns = n_returns
        self.__globals__ = globalns


class ShapeUnknown(LeafError):
    """A dynamic parameter's shape could not be found -- refusing rather than
    guessing whether it is scalar or array-shaped."""


import dataclasses as _dataclasses  # noqa: E402
from process.core.model import DataStructure as _DataStructure  # noqa: E402

_DS = _DataStructure()
_AREA_FIELDS = {f.name: getattr(_DS, f.name) for f in _dataclasses.fields(_DS)}
"""`{area_name: the area's own dataclass instance}` -- `DataStructure`'s own top-level
fields (`.costs`, `.physics`, `.tfcoil`, ...), read once. This is the graph's actual
declared port shape: PROCESS types every array-valued state field `list[float]` and
every scalar `float`/`int`/`bool` (`process/data_structure/*_variables.py`) -- e.g.
`CostData.ucsc: list[float]` against `PhysicsData.rminor: float`. Using this instead of
a live numeric sample is what makes the shape decision available for EVERY leaf
parameter, not just the ones a real cold-start run happens to reach (several of this
port's real dependencies -- `.physics.rminor` among them -- have **no** native
ground-truth value at all: they are produced upstream of the SAND block being
transpiled, by nodes outside it, and `_ground_truth` raises for them. The type
annotation needs no live value, so it answers the shape question anyway)."""


def _field_shape_is_array(path: str):
    """`path` (`.costs.ucsc`) -> `True`/`False`/`None` (cannot tell -- not a plain
    two-component `.<area>.<name>` DataStructure path, e.g. a `^cond`/`^hat`-prefixed
    derived node)."""
    parts = path.strip(".").split(".")
    if len(parts) != 2 or path.startswith("^"):
        return None
    area, name = parts
    sub = _AREA_FIELDS.get(area)
    if sub is None:
        return None
    for f in _dataclasses.fields(sub):
        if f.name != name:
            continue
        t = f.type
        t_str = t if isinstance(t, str) else getattr(t, "__name__", str(t))
        if "list" in t_str.lower() or "ndarray" in t_str.lower() or "tuple" in t_str.lower():
            return True
        if t_str in ("float", "int", "bool"):
            return False
        return None  # some other annotation (a nested dataclass, an enum, ...) -- refuse
    return None


def decide_array_params(fn, order: tuple, inputs: tuple, statics: tuple,
                         real_values: dict | None = None,
                         required: set | None = None) -> dict:
    """`{param_name: is_array}` for every one of `fn`'s parameters `required` names
    (default: every dynamic parameter).

    `order`/`inputs`/`statics` are a resolved `Leaf`'s own fields (see
    `leaves.py`'s `Leaf.order` docstring for how they interleave). Primary
    source: the bound `VarPath`'s own DataStructure field type
    (`_field_shape_is_array`) -- static, always available, no live value needed. When
    that path cannot be classified (not a plain `.<area>.<name>` state field) and a
    real ground-truth/propagated value is available in the optional `real_values`
    (`VarPath.path_str() -> real NumPy value`), that is asked instead, as a fallback
    only -- so a genuinely derived/condition-prefixed port can still be typed when its
    concrete value is known. A `statics` name is always scalar by construction
    (`resolve.py`'s `_static_value` only ever returns a plain `float`/`int`/`bool`).

    A name in `required` that neither source can classify makes this raise
    `ShapeUnknown`: refuse, don't assume scalar -- this is the strict path, and it
    is only asked of a parameter whose body actually SUBSCRIPTS it (the caller
    computes `required` that way; see `_transpile_leaf`), because that is the only
    place the answer changes anything. A name NOT in `required` that neither source
    can classify defaults to scalar (`False`) -- exactly what this parameter would
    have been under the plain scalar transpiler, which never asked the question at
    all; asking it of every dynamic parameter regardless of whether the answer is
    used is new, unwarranted strictness (measured: it silently refused a large,
    genuinely-scalar swath of the closure the first time this was tried without the
    `required` restriction).
    """
    real_values = real_values or {}
    statics_names = {n for n, _ in statics}
    input_iter = iter(inputs)
    decision = {}
    for name in order:
        if name in statics_names:
            decision[name] = False
            continue
        path = next(input_iter)
        needed = required is None or name in required
        from_type = _field_shape_is_array(path)
        if from_type is not None:
            decision[name] = from_type
            continue
        if path in real_values:
            val = np.asarray(real_values[path])
            decision[name] = val.ndim >= 1 and val.size > 1
            continue
        if not needed:
            decision[name] = False
            continue
        raise ShapeUnknown(
            f"{fn.__module__}.{fn.__name__}: parameter {name!r} (bound to "
            f"{path!r}) is neither a plain DataStructure field nor available as a "
            f"real value -- cannot derive its shape, refusing rather than guessing"
        )
    return decision


class _RewriteArrayIndex(ast.NodeTransformer):
    """`jnp.asarray(<array param>)[idx]` or bare `<array param>[idx]` ->
    `<array param>[wp.int32(idx)]`. `array_names`: the subset of this function's OWN
    parameter names `decide_array_params` marked array-valued. Anything else standing
    in a `Subscript` after this pass is still refused upstream (unchanged rule)."""

    def __init__(self, array_names: set):
        self.array_names = array_names

    def visit_Subscript(self, node):
        self.generic_visit(node)
        value = node.value
        base_name = None
        if isinstance(value, ast.Name) and value.id in self.array_names:
            base_name = value.id
        elif (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
              and isinstance(value.func.value, ast.Name)
              and value.func.value.id in ("jnp", "np")
              and value.func.attr in ("asarray", "array")
              and len(value.args) == 1 and not value.keywords
              and isinstance(value.args[0], ast.Name)
              and value.args[0].id in self.array_names):
            base_name = value.args[0].id
        if base_name is None:
            return node
        cast = ast.Call(func=ast.Attribute(value=ast.Name("wp", ast.Load()),
                                            attr="int32", ctx=ast.Load()),
                         args=[node.slice], keywords=[])
        return ast.Subscript(value=ast.Name(base_name, ast.Load()), slice=cast,
                              ctx=ast.Load())


def _tuple_from_global(node, globalns):
    """A bare `Name` (module global) or `jnp.asarray(Name)`/`jnp.array(Name)` whose
    global resolves to a fixed tuple/list of plain numbers -> `(tuple, global_name)`.
    `None` otherwise (left alone -- refuses downstream, same as any other unresolved
    case). The `global_name` half is what `_RewriteInterp` needs to tell
    `_collect_globals`'s unresolved-global check "already accounted for, not a
    guess" -- this table's raw values are now baked into `table_registry` instead."""
    target = node
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in ("jnp", "np")
            and node.func.attr in ("asarray", "array")
            and len(node.args) == 1):
        target = node.args[0]
    if not isinstance(target, ast.Name):
        return None
    val = globalns.get(target.id, None)
    if isinstance(val, (tuple, list)) and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in val):
        return tuple(float(v) for v in val), target.id
    return None


class _RewriteInterp(ast.NodeTransformer):
    """`jnp.interp(x, xp, fp)` -> a call to a shared, monomorphised-per-length
    `wp_interp_N` helper, where `xp`/`fp` resolve to a fixed-length tuple of MODULE
    CONSTANTS (the only case any leaf transpiled through this file needs -- a genuine
    per-call dynamic-array `xp`/`fp` is left alone and still refuses, rather than
    being guessed at).

    Records each resolved table (`table_registry`: `identifier -> tuple[float,...]`)
    and, per function, the ORDERED list of extra trailing `wp.array` parameters its
    transpiled signature now needs (`extra_tables`) -- the emitter must supply real
    `wp.array`s for these at every call site, not just at the kernel signature.
    """

    def __init__(self, globalns: dict, table_registry: dict, needed_lengths: set):
        self.globalns = globalns
        self.table_registry = table_registry
        self.needed_lengths = needed_lengths
        self.extra_tables: list[str] = []  # this function's own extra params, in order
        self.consumed_globals: set[str] = set()  # raw global names now baked into tables

    def _register(self, tup: tuple) -> str:
        for ident, existing in self.table_registry.items():
            if existing == tup:
                return ident
        ident = f"TBL_{len(self.table_registry)}_{len(tup)}"
        self.table_registry[ident] = tup
        return ident

    def visit_Call(self, node):
        self.generic_visit(node)
        f = node.func
        is_interp = (
            isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
            and f.value.id == "jnp" and f.attr == "interp"
            and len(node.args) == 3 and not node.keywords
        )
        if not is_interp:
            return node
        x_expr, xp_expr, fp_expr = node.args
        xp_found = _tuple_from_global(xp_expr, self.globalns)
        fp_found = _tuple_from_global(fp_expr, self.globalns)
        if xp_found is None or fp_found is None:
            return node  # not a case this file handles -- leave it, still refuses
        xp_tuple, xp_name = xp_found
        fp_tuple, fp_name = fp_found
        if len(xp_tuple) != len(fp_tuple):
            return node
        n = len(xp_tuple)
        self.needed_lengths.add(n)
        xp_ident = self._register(xp_tuple)
        fp_ident = self._register(fp_tuple)
        self.consumed_globals.update({xp_name, fp_name})
        for ident in (xp_ident, fp_ident):
            if ident not in self.extra_tables:
                self.extra_tables.append(ident)
        return ast.Call(
            func=ast.Name(f"wp_interp_{n}", ast.Load()),
            args=[x_expr, ast.Name(xp_ident, ast.Load()), ast.Name(fp_ident, ast.Load())],
            keywords=[],
        )


class _RewriteInterpLogRow(ast.NodeTransformer):
    """`jnp.interp(jnp.log(x), jnp.log(TBL[i]), FP[i])` -> `wp_interp_log_row_N(x, TBL, FP, i)`.

    This is the exact shape `scalarise.py` emits for one species' `<Z>(T_e)` lookup:
    both tables are 2-D `(species, 200)` kernel parameters and `i` is a Python literal
    the array expansion already resolved, so the row index is known at codegen time and
    the table itself stays one `wp.array2d` argument rather than fourteen separate
    200-length ones. (The alternative -- fourteen 1-D parameters per table, 28 extra
    kernel arguments -- carries the identical data; a 2-D parameter was chosen because
    Warp indexes `a[i, k]` natively, so the row index stays *data* instead of becoming
    part of 28 generated identifiers that a single off-by-one would silently permute.)

    The `jnp.log` on the table is NOT hoisted host-side. It could be -- `np.log` and
    `jnp.log` were measured bit-identical over 5000 float64 points -- but `wp.log` and
    `jnp.log` were measured bit-identical too (4000 points, 0 ulp, CPU), so taking the
    log in the kernel is exact *and* keeps the table parameter the live boundary value
    rather than a derived copy that would silently stop tracking it.
    """

    def __init__(self, array2d_names: set, needed_lengths: set, shapes: dict):
        self.array2d_names = array2d_names
        self.needed_lengths = needed_lengths
        self.shapes = shapes
        self.hits = 0

    @staticmethod
    def _log_of(node):
        """`jnp.log(<expr>)` -> `<expr>`, else `None`."""
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in ("jnp", "np") and node.func.attr == "log"
                and len(node.args) == 1 and not node.keywords):
            return node.args[0]
        return None

    def _row(self, node):
        """`<2-D param>[<int literal>]` -> `(param_name, index)`, else `None`."""
        if not isinstance(node, ast.Subscript):
            return None
        if not (isinstance(node.value, ast.Name) and node.value.id in self.array2d_names):
            return None
        idx = node.slice
        if isinstance(idx, ast.Constant) and isinstance(idx.value, int) \
                and not isinstance(idx.value, bool):
            return node.value.id, idx.value
        return None

    def visit_Call(self, node):
        self.generic_visit(node)
        f = node.func
        if not (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                and f.value.id == "jnp" and f.attr == "interp"
                and len(node.args) == 3 and not node.keywords):
            return node
        x_expr, xp_expr, fp_expr = node.args
        x_inner = self._log_of(x_expr)
        xp_inner = self._log_of(xp_expr)
        if x_inner is None or xp_inner is None:
            return node
        xp_row = self._row(xp_inner)
        fp_row = self._row(fp_expr)
        if xp_row is None or fp_row is None:
            return node
        if xp_row[1] != fp_row[1]:
            raise LeafError(
                f"per-species interpolation reads row {xp_row[1]} of "
                f"{xp_row[0]} against row {fp_row[1]} of {fp_row[0]} -- the abscissa "
                f"and ordinate rows must be the same species; refusing"
            )
        n_xp = self.shapes[xp_row[0]][1]
        n_fp = self.shapes[fp_row[0]][1]
        if n_xp != n_fp:
            raise LeafError(
                f"per-species interpolation: {xp_row[0]} rows are {n_xp} long but "
                f"{fp_row[0]} rows are {n_fp} -- refusing"
            )
        self.needed_lengths.add(n_xp)
        self.hits += 1
        return ast.Call(
            func=ast.Name(f"wp_interp_log_row_{n_xp}", ast.Load()),
            args=[x_inner, ast.Name(xp_row[0], ast.Load()),
                  ast.Name(fp_row[0], ast.Load()),
                  # `wp.int32(...)`, not a bare literal: the tracked `ToWarp` turns
                  # every numeric constant into `wp.float64(...)` (Warp is strictly
                  # typed), and a `wp.float64` row index matches no overload. The
                  # cast is exact -- the species index is a small literal this
                  # expansion resolved itself, not a computed value.
                  ast.Call(func=ast.Attribute(value=ast.Name("wp", ast.Load()),
                                              attr="int32", ctx=ast.Load()),
                           args=[ast.Constant(xp_row[1])], keywords=[])],
            keywords=[],
        )


def emit_wp_interp_log_row(n: int) -> str:
    """One `@wp.func wp_interp_log_row_N(x, xp2, fp2, row)`, computing exactly
    `jnp.interp(jnp.log(x), jnp.log(xp2[row]), fp2[row])`.

    Same body as `emit_wp_interp` (which mirrors `jax._src.numpy.lax_numpy._interp`
    operation for operation), with two differences, both forced by the source
    function: the abscissa is compared in LOG space (`jnp.log` is applied to the whole
    table before `searchsorted`, so the comparison the loop performs must be between
    logs, not between raw values -- monotonicity would make the *bracket* agree, but
    not necessarily at a tie, and a tie is exactly where an off-by-one row would hide),
    and the table is one row of a 2-D parameter.
    """
    lines = []
    lines.append("@wp.func")
    lines.append(f"def wp_interp_log_row_{n}(x: wp.float64, "
                 f"xp2: wp.array2d(dtype=wp.float64), "
                 f"fp2: wp.array2d(dtype=wp.float64), row: wp.int32) -> wp.float64:")
    lines.append("    lx = wp.log(x)")
    lines.append("    idx = wp.int32(0)")
    lines.append(f"    for k in range({n}):")
    lines.append("        if wp.log(xp2[row, k]) <= lx:")
    lines.append("            idx = wp.int32(k) + wp.int32(1)")
    lines.append(f"    i = wp.clamp(idx, wp.int32(1), wp.int32({n} - 1))")
    lines.append("    lo = wp.log(xp2[row, i - wp.int32(1)])")
    lines.append("    hi = wp.log(xp2[row, i])")
    lines.append("    df = fp2[row, i] - fp2[row, i - wp.int32(1)]")
    lines.append("    dx = hi - lo")
    lines.append("    delta = lx - lo")
    eps = float(np.spacing(np.finfo(np.float64).eps))
    lines.append(f"    dx0 = wp.abs(dx) <= wp.float64({eps!r})")
    lines.append("    denom = wp.where(dx0, wp.float64(1.0), dx)")
    # `_dd_fma`, not `a + b*c`: XLA lowers `jnp.interp`'s last step as a FUSED
    # multiply-add and Warp's `+`/`*` are the plain unfused pair, so the plain form
    # disagrees with JAX by 1 ulp whenever the two roundings differ (measured: 1 of
    # 400 sweep points on this node, traced to exactly this line).
    lines.append("    f = wp.where(dx0, fp2[row, i - wp.int32(1)], "
                 "_dd_fma(delta / denom, df, fp2[row, i - wp.int32(1)]))")
    lines.append("    f = wp.where(lx < wp.log(xp2[row, 0]), fp2[row, 0], f)")
    lines.append(f"    f = wp.where(lx > wp.log(xp2[row, {n} - 1]), fp2[row, {n} - 1], f)")
    lines.append("    return f")
    return "\n".join(lines) + "\n"


def emit_wp_interp(n: int) -> str:
    """One `@wp.func wp_interp_N(x, xp: wp.array(dtype=wp.float64), fp: ...)`, exact
    against `jax._src.numpy.lax_numpy._interp`'s own body (module docstring above) --
    same operations, same order. The `for k in range(N)` with `N` a Python int
    literal is a STATIC loop (Warp unrolls it at codegen time and differentiates it
    correctly, per the brief) -- not a `wp.array`-length-derived dynamic one.
    """
    lines = []
    lines.append(f"@wp.func")
    lines.append(f"def wp_interp_{n}(x: wp.float64, xp: wp.array(dtype=wp.float64), "
                 f"fp: wp.array(dtype=wp.float64)) -> wp.float64:")
    lines.append(f"    idx = wp.int32(0)")
    lines.append(f"    for k in range({n}):")
    lines.append(f"        if xp[k] <= x:")
    lines.append(f"            idx = wp.int32(k) + wp.int32(1)")
    lines.append(f"    i = wp.clamp(idx, wp.int32(1), wp.int32({n} - 1))")
    lines.append(f"    df = fp[i] - fp[i - wp.int32(1)]")
    lines.append(f"    dx = xp[i] - xp[i - wp.int32(1)]")
    lines.append(f"    delta = x - xp[i - wp.int32(1)]")
    # np.spacing(np.finfo(np.float64).eps) -- a fixed float64 literal, computed once
    # host-side (identical value every time; not re-derived per emission).
    eps = float(np.spacing(np.finfo(np.float64).eps))
    lines.append(f"    dx0 = wp.abs(dx) <= wp.float64({eps!r})")
    lines.append(f"    denom = wp.where(dx0, wp.float64(1.0), dx)")
    # Same XLA fusion as the per-row variant above -- see `emit_wp_interp_log_row`.
    lines.append(f"    f = wp.where(dx0, fp[i - wp.int32(1)], "
                 f"_dd_fma(delta / denom, df, fp[i - wp.int32(1)]))")
    lines.append(f"    f = wp.where(x < xp[0], fp[0], f)")
    lines.append(f"    f = wp.where(x > xp[{n - 1}], fp[{n - 1}], f)")
    lines.append(f"    return f")
    return "\n".join(lines) + "\n"


def _mirror_transpile_arrays(fdef: ast.FunctionDef, name: str, array_names: set,
                              extra_params: list | None = None,
                              known_arities: dict | None = None,
                              array_ndim: dict | None = None) -> tuple[str, int]:
    """`leaf_funcs._mirror_transpile`, plus: a parameter in `array_names` is annotated
    `wp.array(dtype=wp.float64)` instead of `wp.float64`, and `extra_params` (table
    identifiers this function's body -- or a callee's -- needs; see `_process`) are
    appended as trailing `wp.array(dtype=wp.float64)` formal parameters, so every call
    site (patched by `_AppendExtraCallArgs`) has something real to pass.

    `known_arities`: `{plain_fn_name: n_returns}` for every same-module helper THIS
    call has already transpiled (`_process` recurses into helpers first, so by the
    time a caller is unparsed, every callee it could forward to is already in here).
    Generalises `leaf_funcs`'s hardcoded `MULTI_RETURN_HELPERS` (which only knows
    about `eq`/`leq`/`geq`) to ANY same-module helper a leaf's body plainly forwards
    to (`return _tf_magnet_cost_superconducting(...)`, no literal tuple) -- still
    exact, not a guess: the arity comes from that helper's OWN already-transpiled
    body, the same source of truth the emitter's arity invariant uses."""
    t = LF._tracked.ToWarp()
    fdef = t.visit(fdef)
    if fdef.args.defaults or any(d is not None for d in fdef.args.kw_defaults):
        raise Unsupported("default argument value(s)")
    if fdef.args.vararg or fdef.args.kwarg:
        raise Unsupported("*args/**kwargs")
    f64 = ast.Attribute(value=ast.Name("wp", ast.Load()), attr="float64", ctx=ast.Load())
    arr_t = ast.Call(
        func=ast.Attribute(value=ast.Name("wp", ast.Load()), attr="array", ctx=ast.Load()),
        args=[], keywords=[ast.keyword(arg="dtype", value=f64)],
    )
    arr2_t = ast.Call(
        func=ast.Attribute(value=ast.Name("wp", ast.Load()), attr="array2d", ctx=ast.Load()),
        args=[], keywords=[ast.keyword(arg="dtype", value=f64)],
    )
    array_ndim = array_ndim or {}
    args = fdef.args
    for a in list(args.args) + list(args.kwonlyargs) + list(args.posonlyargs):
        if a.arg not in array_names:
            a.annotation = f64
        elif array_ndim.get(a.arg, 1) == 2:
            a.annotation = copy.deepcopy(arr2_t)
        else:
            a.annotation = copy.deepcopy(arr_t)
    for ident in (extra_params or []):
        args.args.append(ast.arg(arg=ident, annotation=copy.deepcopy(arr_t)))

    known_arities = known_arities or {}

    def _call_arity(value) -> int:
        if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)):
            return 1
        if value.func.id in MULTI_RETURN_HELPERS:
            return 4
        return known_arities.get(value.func.id, 1)

    n_returns = 1
    for n in ast.walk(fdef):
        if isinstance(n, ast.Return):
            if isinstance(n.value, ast.Tuple):
                n_returns = max(n_returns, len(n.value.elts))
            else:
                n_returns = max(n_returns, _call_arity(n.value))
    returns_tuple = n_returns > 1
    if not returns_tuple:
        fdef.returns = f64
    fdef.name = name
    fdef.decorator_list = [ast.Attribute(value=ast.Name("wp", ast.Load()), attr="func", ctx=ast.Load())]
    mod = ast.Module(body=[fdef], type_ignores=[])
    ast.fix_missing_locations(mod)
    return ast.unparse(mod), n_returns


def build_leaf_funcs_source_arrays(leaves, real_values: dict, tolerant: bool = True,
                                    known_bad: dict = None,
                                    already_emitted: set = None):
    """`leaf_funcs.build_leaf_funcs_source`, extended with array-parameter support.

    `already_emitted`: names of `REGISTRY` entries some earlier call has ALREADY put in
    the generated module. The harness runs this builder and the array-aware one over
    the same module and concatenates their output, and both can need the same helper --
    `_dd_fma` is `_xla_lgamma`'s FMA and also both `jnp.interp` helpers' fused final
    step -- so without a shared set it gets defined twice. The set is updated in place
    with everything this call emits.

    Only a LEAF's OWN top-level parameters are ever asked to be array-valued
    (`decide_array_params`, against `leaf.order`/`leaf.inputs`/`leaf.statics`) --
    every leaf this file was built for keeps its array indexing/interp entirely
    within its own body (none of `calculate_indirect_costs`/
    `calculate_reactor_cooling_system_cost`/`calculate_turbine_plant_equipment_cost`/
    `calculate_tf_magnet_cost_superconducting_per_kg`/`_cost_of_electricity`/
    `calculate_quench_protection_current_density` passes its array parameter on to a
    same-module helper), so this does not need to propagate an array decision through
    the closure/specialisation machinery -- a real limitation if a future leaf DOES
    forward an array parameter into a helper, and this raises rather than guess if a
    Subscript survives that it cannot place.

    Returns `(source, fn_names, failed, arities, extra_table_args, table_registry)`.
    `source` is `leaf_funcs`-shaped (consts, registry, then leaf/helper bodies) PLUS
    one `wp_interp_N` per distinct table length used. `extra_table_args`:
    `{(module, fn): [ident, ...]}` -- trailing `wp.array` parameters the emitter must
    append (in order) to every call of that leaf, beyond its normal
    `order`/`inputs`/`statics`. `table_registry`: `{ident: (float, ...)}` -- the real
    data the emitter must materialise as a `wp.array` once and bind at every launch.
    """
    known_bad = known_bad or {}
    already_emitted = already_emitted if already_emitted is not None else set()
    seen: dict = {}
    failed: dict = {}
    arities: dict = {}
    const_defs: dict = {}
    func_srcs: list = []
    registry_used: set = set()
    spec_registry: dict = {}
    spec_queue: dict = {}
    spec_emitted: set = set()
    extra_table_args: dict = {}
    table_registry: dict = {}
    needed_lengths: set = set()
    needed_log_row_lengths: set = set()

    class _AppendExtraCallArgs(ast.NodeTransformer):
        """Once a same-module HELPER has been found to need extra trailing
        `wp.array` table parameters (because ITS body, not the caller's, held the
        `jnp.interp` call), every call site of that helper -- inside the caller's own
        body -- must pass the same identifiers positionally, last. `extra_by_name`:
        `{helper_fn_name: [ident, ...]}`, built bottom-up (children before parents),
        so a caller two levels up from the actual `jnp.interp` site still gets it."""

        def __init__(self, extra_by_name: dict):
            self.extra_by_name = extra_by_name

        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id in self.extra_by_name:
                # AS KEYWORDS, not appended positional args: the call this rewrite
                # patches may itself already be all-keyword (PROCESS's own source
                # calls several helpers that way, `calculate_quench_protection` ->
                # `calculate_quench_protection_current_density` among them) --
                # appending positional args after existing keywords would bind them
                # to the callee's FIRST parameters instead of its new trailing ones.
                # The trailing formal parameter is named exactly `ident`
                # (`_mirror_transpile_arrays`), so `ident=ident` is unambiguous
                # regardless of how the rest of the call is shaped.
                for ident in self.extra_by_name[node.func.id]:
                    node.keywords.append(ast.keyword(arg=ident, value=ast.Name(ident, ast.Load())))
            return node

    def _process(module: str, name: str, fn_obj, node_label: str, array_names: set,
                 array_shapes: dict | None = None) -> list:
        """Transpile one function (a real leaf's own top-level function, or a
        transitively-called same-module helper), recursing into its callees FIRST.

        Returns this function's FULL extra-table-parameter list: its own (if its
        body directly holds a resolvable `jnp.interp`) UNION every callee's full
        list, in first-seen order -- a function that merely forwards to a table-
        needing helper needs those same tables threaded through its OWN signature
        too (a `@wp.func` cannot close over a Python-side `wp.array`; every table it
        touches, directly or transitively, must be an explicit parameter all the way
        up to the leaf the kernel calls directly).
        """
        key = (module, name)
        array_shapes = dict(array_shapes or {})
        if isinstance(fn_obj, _SynthLeaf):
            # A BUILT leaf: its body was expanded by `scalarise.py`, so there is no
            # Python source to read and no higher-order specialisation left to do
            # (the expansion already monomorphised every function-valued argument).
            fdef = copy.deepcopy(fn_obj.fdef)
            globalns = fn_obj.globalns
            array_shapes = dict(fn_obj.array_shapes)
        else:
            fdef = _fdef_of(fn_obj)
            globalns = getattr(fn_obj, "__globals__", {})
            fdef = _SpecializeHOF(globalns, spec_registry, spec_queue).visit(fdef)
        ast.fix_missing_locations(fdef)

        # Collect the callee worklist from the UNMODIFIED body first (specialisation
        # already applied) -- same shape as `leaf_funcs.build_leaf_funcs_source`.
        worklist = []
        for node in ast.walk(fdef):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                callee_name = node.func.id
                if callee_name in PROVIDED or callee_name in REGISTRY:
                    registry_used.add(callee_name)
                    registry_used.update(LF.REGISTRY_DEPS.get(callee_name, ()))
                    continue
                if callee_name in spec_queue and callee_name not in spec_emitted:
                    sfdef, sglobalns, smodule = spec_queue[callee_name]
                    worklist.append((smodule, callee_name, sfdef, sglobalns))
                    continue
                target = globalns.get(callee_name)
                if _is_models_fn(target) and (target.__module__, callee_name) not in seen:
                    worklist.append((target.__module__, callee_name, None, None))

        # Recurse into helpers BEFORE finalising this function's own source, so
        # `extra_by_name` (this function's call sites) is complete before unparsing.
        extra_by_name: dict = {}
        for m, n, sfdef, sglobalns in worklist:
            k = (m, n)
            if k in seen:
                if k in extra_table_args:
                    extra_by_name[n] = extra_table_args[k]
                continue
            helper_obj = _Specialized(sfdef, sglobalns) if sfdef is not None else \
                getattr(__import__(m, fromlist=[n]), n)
            h_full = _process(m, n, helper_obj, f"{node_label} -> {m}.{n}", set())
            if h_full:
                extra_by_name[n] = h_full
            if sfdef is not None:
                spec_emitted.add(n)

        fdef_lk = _RewriteListLookups().visit(copy.deepcopy(fdef))
        fdef_lk = _RewriteBitwiseBool().visit(fdef_lk)
        fdef_lk = _RewriteInComparisons().visit(fdef_lk)
        # The per-species log-interp rewrite runs BEFORE `_RewriteArrayIndex`:
        # its pattern includes `TBL[i]` subscripts on the 2-D table parameters, and
        # the index rewrite would otherwise turn those into `TBL[wp.int32(i)]` and
        # stop the pattern matching. Its output has no bare table subscript left.
        names_2d = {n for n, sh in array_shapes.items() if len(sh) == 2}
        if names_2d:
            logrow_rw = _RewriteInterpLogRow(names_2d, needed_log_row_lengths,
                                             array_shapes)
            fdef_lk = logrow_rw.visit(fdef_lk)
            ast.fix_missing_locations(fdef_lk)
        fdef_lk = _RewriteArrayIndex(array_names - names_2d).visit(fdef_lk)
        interp_rw = _RewriteInterp(globalns, table_registry, needed_lengths)
        fdef_lk = interp_rw.visit(fdef_lk)
        if extra_by_name:
            fdef_lk = _AppendExtraCallArgs(extra_by_name).visit(fdef_lk)
        ast.fix_missing_locations(fdef_lk)

        for node in ast.walk(fdef_lk):
            if isinstance(node, ast.Subscript):
                # A Subscript whose base is `Name(id in array_names)` is exactly
                # `_RewriteArrayIndex`'s own INTENDED output (`arr[wp.int32(idx)]`)
                # -- not a survivor of the rewrite, its result. Anything else still
                # standing here is a genuine refusal.
                if isinstance(node.value, ast.Name) and node.value.id in array_names:
                    continue
                # `wp.array2d` indexing produced by the log-row rewrite is a Tuple
                # subscript on the table parameter -- also intended output.
                if (isinstance(node.value, ast.Name)
                        and node.value.id in array_shapes
                        and len(array_shapes[node.value.id]) == 2):
                    continue
                target_desc = ast.unparse(node.value)
                raise LeafError(
                    f"{node_label} ({module}.{name}): refused -- subscript "
                    f"`{target_desc}[...]` on something not a literal table, a "
                    f"known array parameter, or resolvable -- refusing rather than "
                    f"guessing"
                )

        # Own + every callee's full list, order-preserving, deduplicated.
        full_extra = list(interp_rw.extra_tables)
        for names in extra_by_name.values():
            for ident in names:
                if ident not in full_extra:
                    full_extra.append(ident)

        known_arities_by_name = {n: arities[(m, n)] for (m, n) in seen if (m, n) in arities}
        try:
            src, n_returns = _mirror_transpile_arrays(
                copy.deepcopy(fdef_lk), name, array_names, extra_params=full_extra,
                known_arities=known_arities_by_name,
                array_ndim={n: len(sh) for n, sh in array_shapes.items()})
        except Unsupported as exc:
            raise LeafError(f"{node_label} ({module}.{name}): refused -- {exc}") from exc
        src = _jnp_pi(src)
        arities[key] = n_returns
        if full_extra:
            extra_table_args[key] = full_extra

        resolved, unresolved = _collect_globals(fdef, globalns)
        # A name `_RewriteInterp` already consumed (baked its real values into
        # `table_registry`, threaded in as a `wp.array` parameter) is accounted for,
        # not unresolved -- `_collect_globals` only sees the ORIGINAL body and has no
        # way to know that.
        unresolved = unresolved - interp_rw.consumed_globals
        if unresolved:
            raise LeafError(f"{node_label} ({module}.{name}): unresolved global(s) "
                             f"{sorted(unresolved)} -- not a plain scalar, refusing "
                             f"rather than guessing")
        for text, val in sorted(resolved.items(), key=lambda kv: -len(kv[0])):
            ident = _const_ident(text)
            const_defs.setdefault(ident, f"{ident} = wp.constant(wp.float64({val!r}))")
            src = re.sub(rf"(?<![\w.]){re.escape(text)}(?!\w)", ident, src)
        func_srcs.append(src)
        seen[key] = name
        return full_extra

    def _transpile_leaf(leaf_obj):
        key = (leaf_obj.module, leaf_obj.fn)
        if leaf_obj.fn in REGISTRY:
            registry_used.add(leaf_obj.fn)
            registry_used.update(LF.REGISTRY_DEPS.get(leaf_obj.fn, ()))
            seen[key] = leaf_obj.fn
            arities[key] = REGISTRY_ARITY[leaf_obj.fn]
            return
        if key in known_bad:
            raise LeafError(f"{leaf_obj.node} ({leaf_obj.module}.{leaf_obj.fn}): "
                             f"excluded -- known bad from a previous round: {known_bad[key]}")
        synth = SYNTH_LEAVES.get(key)
        if synth is not None:
            _process(leaf_obj.module, leaf_obj.fn, synth, leaf_obj.node,
                     set(synth.array_shapes), synth.array_shapes)
            return
        import importlib
        mod = importlib.import_module(leaf_obj.module)
        fn = getattr(mod, leaf_obj.fn, None)
        if fn is None:
            raise LeafError(f"{leaf_obj.node}: {leaf_obj.module}.{leaf_obj.fn} not found")

        # The shape question ONLY matters for a parameter this function's body
        # actually SUBSCRIPTS (that is the one place `_mirror_transpile_arrays`'s
        # annotation and `_RewriteArrayIndex`'s rewrite need to know it) -- asking
        # (and possibly refusing on) every OTHER dynamic parameter regardless would
        # be new, unnecessary strictness this leaf never needed under the plain
        # scalar transpiler, and it silently regressed a large chunk of the closure
        # in exactly this way the first time this file's `real_values`-driven
        # decision was wired into the full sub-DAG loop (`.stellarator.coilcurrent`/
        # `.tfcoil.a_tf_wp_with_insulation` and others -- plain scalar arithmetic,
        # never subscripted, but with no DataStructure field of their own since
        # they are graph-derived, not raw state -- were refused for a question
        # their own bodies never ask). A subscript on a LITERAL list (`_RewriteList
        # Lookups`'s job) does not count -- run that rewrite first so only a
        # genuine variable-based subscript is seen.
        raw_fdef = _fdef_of(fn)
        raw_fdef = _RewriteListLookups().visit(raw_fdef)
        param_names = {a.arg for a in raw_fdef.args.args}
        subscripted = set()
        for node in ast.walk(raw_fdef):
            if not isinstance(node, ast.Subscript):
                continue
            v = node.value
            if isinstance(v, ast.Name) and v.id in param_names:
                subscripted.add(v.id)
            elif (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute)
                  and isinstance(v.func.value, ast.Name) and v.func.value.id in ("jnp", "np")
                  and v.func.attr in ("asarray", "array") and len(v.args) == 1
                  and isinstance(v.args[0], ast.Name) and v.args[0].id in param_names):
                subscripted.add(v.args[0].id)

        if subscripted:
            decisions = decide_array_params(
                fn, leaf_obj.order, leaf_obj.inputs, leaf_obj.statics, real_values,
                required=subscripted)
            array_names = {n for n in subscripted if decisions.get(n)}
        else:
            array_names = set()

        _process(leaf_obj.module, leaf_obj.fn, fn, leaf_obj.node, array_names)

    for leaf_obj in leaves:
        key = (leaf_obj.module, leaf_obj.fn)
        if key in seen or key in failed:
            continue
        try:
            _transpile_leaf(leaf_obj)
        except LeafError as exc:
            if tolerant:
                failed[key] = str(exc)
                continue
            raise

    interp_srcs = [emit_wp_interp(n) for n in sorted(needed_lengths)]
    interp_srcs += [emit_wp_interp_log_row(n) for n in sorted(needed_log_row_lengths)]
    if interp_srcs:
        registry_used.add("_dd_fma")   # both interp helpers' fused final step

    parts = []
    if const_defs:
        parts.append("\n".join(const_defs[k] for k in sorted(const_defs)))
    emit_names = [n for n in LF.registry_closure(registry_used)
                  if n not in already_emitted]
    already_emitted.update(emit_names)
    if emit_names:
        parts.append("\n".join(REGISTRY[n] for n in emit_names))
    parts.extend(interp_srcs)
    parts.extend(func_srcs)
    fn_names = {k: v for k, v in seen.items()}
    source = "\n\n".join(parts)
    return source, fn_names, failed, arities, extra_table_args, table_registry
