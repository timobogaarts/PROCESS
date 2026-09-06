"""Leaf `@wp.func` source generation: the tracked transpiler plus what it does not do
(deliberately -- it "refuses rather than guesses", CLAUDE.md's design rule for this
whole effort):

  - resolving module-global scalars to `wp.constant`s (`constants.KILOELECTRON_VOLT`,
    `WARD_KINK_SMOOTHING`);
  - `jnp.pi`/`np.pi`/`math.pi` (a bare Attribute, not a Call -- `ToWarp.visit_Call`
    never sees it);
  - a transitive same-module helper-call closure (a leaf's body calling another
    `functional_process.models` function that is not itself a leaf);
  - **monomorphising higher-order calls.** Warp has no first-class functions. A body
    like `_fast_alpha_beta(_fast_alpha_fraction_ward, ...)` passes a function BY
    REFERENCE as an argument -- found by root-causing why the closure above still left
    `_fast_alpha_fraction_ward` an "undefined symbol" inside `fast_alpha_beta_ward`
    despite being queued: it never needed queueing as a *called* name, because at the
    AST level it is only ever a bare `Name` passed as an *argument*, never itself the
    `.func` of a `Call` -- the closure scan only looks at `.func`, so a function
    reference passed as a value was invisible to it. Fixed by specialising the callee:
    generate one concrete version of `_fast_alpha_beta` per distinct function it is
    called with, with the call to the parameter's name rewritten to the concrete
    function's own name and that parameter dropped. Exact, not an approximation --
    every call site still computes the same thing, just monomorphised the way a C++
    template or Rust generic would be.
  - **list-literal table lookups.** `[a, b, c, d][idx]` (a literal list subscripted by
    a runtime index) is `ast.List`, which Warp's codegen refuses outright. Rewritten to
    an exact chain of `wp.where` on the (integer-valued, 0-based after a `- 1`) index --
    no approximation, and it refuses (does not guess) if the index expression is not of
    the simple `<something> - <int literal>` shape produced by PROCESS's `lsa`-family
    1-based table lookups.
  - a hand-written registry for `PROVIDED` helpers (`safe_sqrt`, `safe_pow`) that the
    transpiler leaves untouched by design.

Built on this package's own `transpile.py` -- the single-function transpiler, imported
read-only. Everything above is what this module adds around it; nothing here edits it.
"""
from __future__ import annotations

import ast
import copy
import importlib
import inspect
import math
import re
import textwrap

from . import transpile as _tracked
from .transpile import PROVIDED, Unsupported

# `transpile.py` is the single-function transpiler this module drives: one
# `models/**` function to one `@wp.func`, refusing rather than guessing. `Unsupported`
# is its refusal; `PROVIDED` names the port-local helpers it emits calls to but does
# not define, which is what `REGISTRY` below supplies bodies for.


class LeafError(Exception):
    """This leaf could not be turned into a `@wp.func` -- refuse, do not guess."""


# ---- the registry: hand-written @wp.func for PROVIDED port-local helpers --------------
REGISTRY: dict[str, str] = {
    # A correctly-rounded fused multiply-add (Dekker splitting), bit-identical to libm
    # `fma` (20000/20000, measured). Needed because **XLA fuses and Warp does not**:
    # `jnp.interp`'s final `fp[i-1] + (delta/dx)*df` comes out of XLA as an FMA, while
    # Warp's `a*b + c` is the plain unfused pair (Warp matches NumPy 20000/20000,
    # libm-fma 15460/20000). Emitting the plain form there disagreed with JAX by 1 ulp
    # on 1 of 400 sweep points -- a genuinely different arithmetic, not just a
    # different rounding, which is why this survives the relaxed agreement gate.
    # `_xla_lgamma`'s polynomial steps are FMAs for the same reason.
    "_dd_fma": (
        '@wp.func\n'
        'def _dd_fma(a: wp.float64, b: wp.float64, c: wp.float64) -> wp.float64:\n'
        '    s = wp.float64(134217729.0)\n'
        '    p = a * b\n'
        '    ca = s * a\n'
        '    ah = ca - (ca - a)\n'
        '    al = a - ah\n'
        '    cb = s * b\n'
        '    bh = cb - (cb - b)\n'
        '    bl = b - bh\n'
        '    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl\n'
        '    t = p + c\n'
        '    z = t - p\n'
        '    d = (p - (t - z)) + (c - z)\n'
        '    return t + (d + e)\n'
    ),
    # `lax.exp` is not libm's `exp`: XLA expands it as Cephes-with-FMA, and the two
    # disagree by 1 ulp on ~13.9 % of arguments (`wp.exp` was measured identical to
    # `np.exp` and `math.exp` -- it is JAX that is the outlier). This reproduces
    # `lax.exp` bit-for-bit **on [-8, 8]**, and NOT outside it: 19999/20000 on
    # [-20, 5], and only 3896/20000 on [-745, -700] near underflow. It is therefore
    # NOT used to route leaf `jnp.exp` calls -- `wp.exp`'s 1 ulp is both smaller and
    # predictable at every magnitude, and passes the agreement gate. Its one caller is
    # `gamma`, whose argument is `lgamma` of a profile index: comfortably inside the
    # range where this is exact.
    "_xla_exp": (
        '@wp.func\n'
        'def _xla_exp(x: wp.float64) -> wp.float64:\n'
        '    n = wp.floor(wp.float64(1.4426950408889634) * x + wp.float64(0.5))\n'
        '    r = x - n * wp.float64(6.93145751953125e-1)\n'
        '    r = r - n * wp.float64(1.42860682030941723212e-6)\n'
        '    rr = r * r\n'
        '    p = wp.float64(1.26177193074810590878e-4)\n'
        '    p = _dd_fma(p, rr, wp.float64(3.02994407707441961300e-2))\n'
        '    p = _dd_fma(p, rr, wp.float64(9.99999999999999999910e-1))\n'
        '    p = r * p\n'
        '    q = wp.float64(3.00198505138664455042e-6)\n'
        '    q = _dd_fma(q, rr, wp.float64(2.52448340349684104192e-3))\n'
        '    q = _dd_fma(q, rr, wp.float64(2.27265548208155028766e-1))\n'
        '    q = _dd_fma(q, rr, wp.float64(2.00000000000000000009e0))\n'
        '    y = p / (q - p)\n'
        '    y = wp.float64(1.0) + wp.float64(2.0) * y\n'
        '    return y * wp.pow(wp.float64(2.0), n)\n'
    ),
    "safe_sqrt": (
        "@wp.func\n"
        "def safe_sqrt(x: wp.float64) -> wp.float64:\n"
        "    return wp.sqrt(wp.max(x, wp.float64(0.0)))\n"
    ),
    "safe_pow": (
        "@wp.func\n"
        "def safe_pow(x: wp.float64, p: wp.float64) -> wp.float64:\n"
        "    return wp.where(x == wp.float64(0.0), wp.float64(0.0), wp.pow(x, p))\n"
    ),
    # `functional_process.cottax.core.solver.constraints.{leq,geq,eq}` -- the four
    # constraint functions that call these (constraint_2/16/24/84) live OUTSIDE
    # `functional_process.models`, so the closure scan's `functional_process.models`
    # prefix rule never picks them up as same-module helpers and they were being
    # flagged "unresolved global" instead. They are trivial, pure, already-verbatim
    # (no jnp constructs beyond `/` and `-`) -- hand-transcribed here exactly rather
    # than generalising the prefix rule, per the coordinator's steer ("likely a small
    # PROVIDED-style entry"). Source: `constraints.py:59-92`.
    "leq": (
        "@wp.func\n"
        "def leq(value: wp.float64, bound: wp.float64):\n"
        "    residual = value - bound\n"
        "    normalised_residual = (value / bound) - wp.float64(1.0)\n"
        "    return residual, normalised_residual, value, bound\n"
    ),
    "geq": (
        "@wp.func\n"
        "def geq(value: wp.float64, bound: wp.float64):\n"
        "    residual = bound - value\n"
        "    normalised_residual = wp.float64(1.0) - (value / bound)\n"
        "    return residual, normalised_residual, value, bound\n"
    ),
    "eq": (
        "@wp.func\n"
        "def eq(value: wp.float64, bound: wp.float64):\n"
        "    residual = value - bound\n"
        "    normalised_residual = wp.float64(1.0) - (value / bound)\n"
        "    return residual, normalised_residual, value, bound\n"
    ),
    # `jax.scipy.special.gamma` -- the Gamma FUNCTION, imported at the top of
    # `functional_process/models/physics/plasma_profiles.py` and called four times by
    # `calculate_parabolic_profile_values` (`gamma(0.5)`, `gamma(alpha + 1)`,
    # `gamma(alpha + 1.5)`).  It is a module global that is not a plain scalar, so
    # `_collect_globals` refused the leaf outright.  Warp has no `lgamma`/`tgamma`
    # builtin, so this is a hand-written device implementation.
    #
    # WHAT IT MIRRORS.  `jax.scipy.special.gamma(x)` is, for real input, literally
    # `gammasgn(x) * exp(lgamma(x))` (jax/_src/scipy/special.py) -- and `gammasgn` is
    # +1 over the whole domain this leaf uses, so `gamma == exp(lgamma)` exactly.
    # `lax.lgamma` is NOT libm's `lgamma`: XLA expands it inline as Lanczos (g=7,
    # 8 coefficients).  The op sequence below was read off the *compiled HLO* of
    # `jax.jit(lax.lgamma)` in this very env, operation for operation and constant for
    # constant -- `z*0.13333333333333333` (a multiply, not `z/7.5`), the
    # `(z + 0.5 - t/log_t)*log_t` grouping (not `(z+0.5)*log_t - t`, which rounds
    # differently), the Lanczos sum accumulated `C0/(z+1) + BASE` first and then one
    # `+ Ci/(z+i+1)` at a time, and `log_t` built as `log1p(...) + log(7.5)` -- not
    # transcribed from memory of the XLA source.
    #
    # WHY THE OTHER THREE HELPERS EXIST.  Naively calling `wp.exp(wp.log(...))` here
    # does NOT reproduce JAX: `wp.exp` compiles to the platform libm `exp`, and
    # `lax.exp` is not libm's -- they disagree on ~13.5 % of arguments by 1 ulp.  Nor
    # is `lax.log1p` libm's log1p (~20 % disagreement).  `lax.log` and `lax.sqrt` ARE
    # libm's, which is why every leaf transpiled before this one read 0.000e+00: none
    # of them used any transcendental but `sqrt`.  Each XLA primitive was therefore
    # reverse-engineered against this env and CHECKED, not assumed:
    #   * `lax.exp`   == Cephes `exp.c` (`floor(log2e*x+0.5)`, C1/C2 range reduction,
    #     the 2/3-term rational, `ldexp`) with the polynomial steps EVALUATED AS FMAs.
    #     8000/8000 bit-exact on [-8, 8].
    #   * `lax.log1p` == Cephes `unity.c` (`log(1+x)` outside 1+x in [sqrt(1/2),
    #     sqrt(2)], the LP/LQ rational inside) with FMA polynomial steps AND the
    #     `x*x*x*(P/Q)` grouping -- `x*((x*x*P)/Q)`, Cephes' own grouping, is 1 ulp out
    #     on ~0.5 % of arguments.  8000/8000 bit-exact on [-0.5, 0.8].
    #   * `lax.lgamma`'s `(z + 0.5 - t/log_t)*log_t + log(sqrt(2pi))` tail is likewise
    #     ONE FMA, not a multiply and an add: that single contraction is worth 8.9e-16
    #     absolute and was the entire residual error in the first version of this
    #     entry.  8009/8009 bit-exact on [0.5, 12].
    # Warp does not contract multiply-add on the CPU backend (it does on CUDA), so the
    # FMAs are done explicitly by `_dd_fma`, an exact Dekker/Veltkamp two-product plus
    # two-sum -- verified equal to a `fractions.Fraction` reference FMA on 4000/4000
    # random triples, on BOTH cpu and cuda:0 (its own multiply-adds are all exact, so
    # CUDA's contraction cannot perturb it).
    #
    # MEASURED, end to end on this leaf against JAX evaluating the same function at one
    # concrete point: 0.000e+00 relative on all five outputs at every point of a 41x41
    # grid of alphan,alphat in [0,3]^2 (1681/1681), and at all three configurations'
    # own alphan/alphat.  Against `jax.jit(jax.vmap(...))` the same grid shows up to
    # 3.2e-15 -- but that is JAX disagreeing with ITSELF: batched XLA takes a different
    # (vectorised) transcendental path, and JAX-scalar vs JAX-vmap differs by exactly
    # the same 3.161e-15 at exactly the same 237 grid points.  `harness.py`'s agreement
    # check evaluates the reference one point at a time, i.e. the bit-exact column.
    #
    # DOMAIN.  Only x >= 0.5 is implemented -- that is the whole range this leaf uses
    # (`gamma(0.5)`, and `gamma(alpha + 1)`/`gamma(alpha + 1.5)` for a non-negative
    # profile index).  XLA's Euler-reflection branch for x < 0.5 is deliberately NOT
    # transcribed: it contains `log(sin(pi*x))`, whose adjoint is +-inf at every
    # integer and would poison Warp's reverse pass through `wp.where`.  x < 0.5
    # returns NaN from a compile-time constant (zero adjoint), loudly, rather than
    # silently returning the unreflected value.
    #
    # `_xla_exp` uses `y * wp.pow(2, n)` for Cephes' `ldexp` (Warp has neither `ldexp`
    # nor `exp2`); `wp.pow(2.0, n)` was checked exact for every integer n in [-20, 20).
    #
    # Differentiable under Warp: straight-line arithmetic, no loops, no data-dependent
    # branching -- only `wp.where`.
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
    "_xla_lgamma": (
        "_XLA_GAMMA_NAN = wp.constant(wp.float64(float(\"nan\")))\n"
        "@wp.func\n"
        "def _xla_lgamma(x: wp.float64) -> wp.float64:\n"
        "    z = x + wp.float64(-1.0)\n"
        "    log_t = _xla_log1p(z * wp.float64(0.13333333333333333)) + wp.float64(2.0149030205422647)\n"
        "    u = (z + wp.float64(0.5)) - (z + wp.float64(7.5)) / log_t\n"
        "    a = _dd_fma(u, log_t, wp.float64(0.91893853320467267))\n"
        "    s = wp.float64(676.5203681218851) / (z + wp.float64(1.0)) + wp.float64(0.99999999999980993)\n"
        "    s = s + wp.float64(-1259.1392167224028) / (z + wp.float64(2.0))\n"
        "    s = s + wp.float64(771.32342877765313) / (z + wp.float64(3.0))\n"
        "    s = s + wp.float64(-176.61502916214059) / (z + wp.float64(4.0))\n"
        "    s = s + wp.float64(12.507343278686905) / (z + wp.float64(5.0))\n"
        "    s = s + wp.float64(-0.13857109526572012) / (z + wp.float64(6.0))\n"
        "    s = s + wp.float64(9.9843695780195716e-06) / (z + wp.float64(7.0))\n"
        "    s = s + wp.float64(1.5056327351493116e-07) / (z + wp.float64(8.0))\n"
        "    return a + wp.log(s)\n"
    ),
    "gamma": (
        "@wp.func\n"
        "def gamma(x: wp.float64) -> wp.float64:\n"
        "    return wp.where(x < wp.float64(0.5), _XLA_GAMMA_NAN, _xla_exp(_xla_lgamma(x)))\n"
    ),
    # `jax.scipy.special.gammaln`, imported alongside `gamma` in the same module and
    # used by `plasma_profiles._beta` (which `calculate_pedestal_on_axis_temperatures`
    # calls) -- the second leaf this refusal blocks, on `large_tokamak_nof`.  It is
    # literally `lax.lgamma` (jax/_src/scipy/special.py: `return lax.lgamma(x)`), so it
    # is `_xla_lgamma` under another name, with the same x >= 0.5 domain restriction.
    "gammaln": (
        "@wp.func\n"
        "def gammaln(x: wp.float64) -> wp.float64:\n"
        "    return wp.where(x < wp.float64(0.5), _XLA_GAMMA_NAN, _xla_lgamma(x))\n"
    ),
}

REGISTRY_ARITY = {"safe_sqrt": 1, "safe_pow": 1, "leq": 4, "geq": 4, "eq": 4,
                  "gamma": 1, "gammaln": 1,
                  "_dd_fma": 1, "_xla_exp": 1, "_xla_log1p": 1, "_xla_lgamma": 1}
"""Return arity of each `REGISTRY` entry -- known by construction (hand-written)."""

REGISTRY_DEPS = {
    "_xla_exp": ("_dd_fma",),
    "_xla_log1p": ("_dd_fma",),
    "_xla_lgamma": ("_dd_fma", "_xla_log1p"),
    "gamma": ("_xla_lgamma", "_xla_exp"),
    "gammaln": ("_xla_lgamma",),
}
"""Which other `REGISTRY` entries an entry's body calls.

The `gamma` block is FOUR functions, not one, and they are separate entries on purpose:
`_dd_fma` is also what both `jnp.interp` helpers use for their fused final step
(`leaf_funcs_arrays.emit_wp_interp`), so a single monolithic `"gamma"` string would
define `_dd_fma` a second time in any kernel that needs both. Splitting them makes each
definition appear exactly once (`registry_used` is a set) at the price of having to say
what calls what -- which is this table, and which `registry_closure` walks.
"""

REGISTRY_ORDER = ("_dd_fma", "_xla_log1p", "_xla_exp", "_xla_lgamma", "gamma", "gammaln")
"""Emission order for entries that call each other -- a definition must precede its
callers in the generated module. Everything not listed sorts after these, by name (the
original `sorted()` behaviour), which is fine because nothing else has a dependency."""


def registry_closure(names) -> list:
    """`names` plus everything they transitively call, in a valid emission order.

    Not `sorted(names)`: `_xla_lgamma` calls `_xla_log1p`, and `"_xla_lgamma" <
    "_xla_log1p"`, so plain alphabetical order would emit a use before its definition.
    """
    want = set()

    def visit(n):
        if n in want:
            return
        want.add(n)
        for d in REGISTRY_DEPS.get(n, ()):
            visit(d)

    for n in names:
        visit(n)
    rank = {n: i for i, n in enumerate(REGISTRY_ORDER)}
    return sorted(want, key=lambda n: (rank.get(n, len(rank)), n))


_MISSING = object()


def _strip_docstring(src: str) -> str:
    return re.sub(r'"""(?:.|\n)*?"""', "", src, count=1)


def _fdef_of(fn) -> ast.FunctionDef:
    """A fresh, docstring-stripped `ast.FunctionDef` for a real function object."""
    src = _strip_docstring(textwrap.dedent(inspect.getsource(fn)))
    return ast.parse(src).body[0]


def _is_models_fn(val) -> bool:
    return val is not None and inspect.isfunction(val) and \
        str(getattr(val, "__module__", "")).startswith("functional_process.models")


def _local_names(fdef: ast.FunctionDef) -> set:
    names = set()
    for a in list(fdef.args.args) + list(fdef.args.posonlyargs) + list(fdef.args.kwonlyargs):
        names.add(a.arg)
    for n in ast.walk(fdef):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                for nm in ast.walk(t):
                    if isinstance(nm, ast.Name):
                        names.add(nm.id)
        if isinstance(n, ast.comprehension):
            for nm in ast.walk(n.target):
                if isinstance(nm, ast.Name):
                    names.add(nm.id)
    return names


def _collect_globals(fdef: ast.FunctionDef, globalns: dict) -> tuple[dict, set]:
    """Free names/attributes `fdef`'s body reads from module scope.

    Returns `(resolved, unresolved)`: `resolved` maps the exact source text
    (`"constants.KILOELECTRON_VOLT"` or `"WARD_KINK_SMOOTHING"`) to its plain Python
    scalar value; `unresolved` holds text for anything that is a global reference but
    not a plain int/float (an enum, a class, a table) -- those make the leaf refuse.
    """
    local = _local_names(fdef)
    resolved: dict[str, float] = {}
    unresolved: set[str] = set()
    skip_bases = {"jnp", "np", "wp"} | set(PROVIDED) | set(REGISTRY)
    attribute_bases = {
        id(n.value)
        for n in ast.walk(fdef)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
    }
    for node in ast.walk(fdef):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            base_name = node.value.id
            if base_name in local or base_name in skip_bases:
                continue
            base = globalns.get(base_name, _MISSING)
            if base is _MISSING:
                continue
            val = getattr(base, node.attr, _MISSING)
            text = f"{base_name}.{node.attr}"
            if val is not _MISSING and isinstance(val, (int, float)) and not isinstance(val, bool):
                resolved[text] = float(val)
            elif val is not _MISSING:
                unresolved.add(text)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if id(node) in attribute_bases:
                continue
            nm = node.id
            if nm in local or nm in skip_bases:
                continue
            if nm not in globalns:
                continue  # a builtin (range/len/...) -- not our concern here
            val = globalns[nm]
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                resolved[nm] = float(val)
            elif _is_models_fn(val):
                continue  # a same-module helper -- the closure scan pulls this in
            else:
                unresolved.add(nm)
    return resolved, unresolved


def _const_ident(text: str) -> str:
    return "K_" + re.sub(r"[^0-9a-zA-Z_]", "_", text)


class _RewriteListLookups(ast.NodeTransformer):
    """`[a, b, c, d][idx]` -> a `wp.where` chain, exact, or refuse.

    Rewrites a `Subscript` whose value is a literal `ast.List`, OR a literal list
    wrapped in `jnp.asarray(...)`/`jnp.array(...)`/`np.asarray(...)`/`np.array(...)`
    (PROCESS's own idiom, e.g. `jnp.asarray([0.5, 0.75, 0.875, 1.0])[lsa - 1]`) -- both
    of scalar-shaped elements. Only handles a `slice` of the form `<expr>`,
    `<expr> - <int literal>`, or a bare non-negative `ast.Constant` int -- PROCESS's own
    `lsa`-family pattern is always "1-based index minus 1"; anything else leaves the
    node untouched (so it still refuses downstream) rather than guess at the offset.
    """

    def visit_Subscript(self, node):
        self.generic_visit(node)
        value = node.value
        if isinstance(value, ast.List):
            elts = value.elts
        elif (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and isinstance(value.func.value, ast.Name)
            and value.func.value.id in ("jnp", "np")
            and value.func.attr in ("asarray", "array")
            and len(value.args) == 1
            and isinstance(value.args[0], ast.List)
        ):
            elts = value.args[0].elts
        else:
            return node
        idx_expr = node.slice
        if isinstance(idx_expr, ast.Constant) and isinstance(idx_expr.value, int):
            offset_expr, base = None, idx_expr.value
        elif isinstance(idx_expr, ast.BinOp) and isinstance(idx_expr.op, ast.Sub) and \
                isinstance(idx_expr.right, ast.Constant) and isinstance(idx_expr.right.value, int):
            offset_expr, base = idx_expr.left, -idx_expr.right.value
        else:
            offset_expr, base = idx_expr, 0
        # index_val == i  <=>  offset_expr == i - base
        chain = elts[-1]
        for i in range(len(elts) - 2, -1, -1):
            target = ast.Constant(i - base)
            cond = ast.Compare(left=copy.deepcopy(offset_expr), ops=[ast.Eq()],
                                comparators=[target])
            chain = ast.Call(
                func=ast.Attribute(value=ast.Name("wp", ast.Load()), attr="where",
                                   ctx=ast.Load()),
                args=[cond, elts[i], chain], keywords=[],
            )
        return chain


class _RewriteBitwiseBool(ast.NodeTransformer):
    """`(a == 1) | (b == 2)` -> `(a == 1) or (b == 2)`, `&` -> `and`.

    JAX code has no vectorised `or`/`and`, so `|`/`&` between comparison results is
    PROCESS's/jax's own idiom for elementwise boolean OR/AND throughout this codebase
    (confirmed at the one site this hit: `(ifueltyp == 1) | (ifueltyp == 2)`) -- never
    integer bitwise arithmetic on a physics quantity. Warp's `|`/`&` (`bit_or`/`bit_and`)
    has no `bool` overload and Warp has no `logical_or`/`logical_and` builtin either,
    but a kernel body is per-thread scalar code, where plain `or`/`and` on two already-
    boolean values is exactly bitwise OR/AND on `{0, 1}` -- Warp supports the Python
    keywords directly, so this rewrite is exact for how the source is actually used
    here, not a guess at unknown intent.
    """

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.BitOr):
            return ast.BoolOp(op=ast.Or(), values=[node.left, node.right])
        if isinstance(node.op, ast.BitAnd):
            return ast.BoolOp(op=ast.And(), values=[node.left, node.right])
        return node


class _RewriteInComparisons(ast.NodeTransformer):
    """`x in (a, b, c)` -> `x == a or x == b or x == c`, `not in` -> the `!=`/`and` dual.

    Warp's codegen has no `In`/`NotIn` comparator. Only rewrites a single-comparator
    `Compare` whose right-hand side is a literal `Tuple`/`List` (the only shape this
    codebase uses it in -- an enum-membership test against a small fixed set) -- exact,
    since `x in (a, b)` and `x == a or x == b` are the same predicate for scalars;
    anything else (a real container variable) is left alone, so it still refuses
    downstream rather than being guessed at.
    """

    def visit_Compare(self, node):
        self.generic_visit(node)
        if len(node.ops) != 1 or not isinstance(node.ops[0], (ast.In, ast.NotIn)):
            return node
        rhs = node.comparators[0]
        if not isinstance(rhs, (ast.Tuple, ast.List)):
            return node
        is_not = isinstance(node.ops[0], ast.NotIn)
        eq_op, bool_op = (ast.NotEq(), ast.And()) if is_not else (ast.Eq(), ast.Or())
        terms = [ast.Compare(left=copy.deepcopy(node.left), ops=[eq_op], comparators=[elt])
                 for elt in rhs.elts]
        return terms[0] if len(terms) == 1 else ast.BoolOp(op=bool_op, values=terms)


class _SpecializeHOF(ast.NodeTransformer):
    """Warp has no first-class functions. A `Call` that passes another
    `functional_process.models` function BY REFERENCE (a bare `Name` argument that
    itself resolves to such a function -- not a call to it) is rewritten to call a
    monomorphised specialisation of the callee instead, with that argument dropped.
    Exact: the specialisation is the callee's body with every call to the
    function-valued parameter's name replaced by the concrete function's own name.
    `spec_registry`/`spec_queue` are shared per `build_leaf_funcs_source` call.
    """

    def __init__(self, globalns: dict, spec_registry: dict, spec_queue: dict):
        self.globalns = globalns
        self.spec_registry = spec_registry
        self.spec_queue = spec_queue

    def visit_Call(self, node):
        self.generic_visit(node)
        if not isinstance(node.func, ast.Name):
            return node
        callee = self.globalns.get(node.func.id)
        if not _is_models_fn(callee):
            return node
        fn_positions = []
        for i, a in enumerate(node.args):
            if isinstance(a, ast.Name):
                target = self.globalns.get(a.id)
                if _is_models_fn(target):
                    fn_positions.append((i, a.id, target))
        if not fn_positions:
            return node
        spec_name = self._get_or_build(callee, fn_positions)
        drop = {i for i, _, _ in fn_positions}
        node.func = ast.Name(id=spec_name, ctx=ast.Load())
        node.args = [a for i, a in enumerate(node.args) if i not in drop]
        return node

    def _get_or_build(self, callee, fn_positions) -> str:
        passed_names = tuple(name for _, name, _ in fn_positions)
        key = (callee.__module__, callee.__name__, passed_names)
        if key in self.spec_registry:
            return self.spec_registry[key]
        spec_name = callee.__name__ + "__spec__" + "_".join(
            re.sub(r"\W", "_", n) for n in passed_names)
        self.spec_registry[key] = spec_name

        callee_fdef = _fdef_of(callee)
        param_names = [a.arg for a in callee_fdef.args.args]
        drop_positions = {i for i, _, _ in fn_positions}
        param_to_concrete = {param_names[i]: name for i, name, _ in fn_positions}

        class _Rewrite(ast.NodeTransformer):
            def visit_Call(self, inner):
                self.generic_visit(inner)
                if isinstance(inner.func, ast.Name) and inner.func.id in param_to_concrete:
                    inner.func = ast.Name(id=param_to_concrete[inner.func.id], ctx=ast.Load())
                return inner

        callee_fdef = _Rewrite().visit(callee_fdef)
        callee_fdef.args.args = [
            a for i, a in enumerate(callee_fdef.args.args) if i not in drop_positions
        ]
        # Recurse: the specialised body may itself pass a function by reference.
        callee_fdef = _SpecializeHOF(callee.__globals__, self.spec_registry,
                                      self.spec_queue).visit(callee_fdef)
        ast.fix_missing_locations(callee_fdef)
        self.spec_queue[spec_name] = (callee_fdef, callee.__globals__, callee.__module__)
        return spec_name


class _SubstituteSequenceStatics(ast.NodeTransformer):
    """Monomorphise a leaf around its SEQUENCE-valued static arguments.

    `resolve._static_sequence_value` resolves an argument like `self.coefficients`
    (`PeakBTfInboardWithRipple16Coils.coefficients`, a four-float class constant) to a
    plain tuple of scalars. Warp has no tuple type, so unlike a scalar static this
    cannot be rendered as an argument at the call site. It is instead baked into the
    function: every load of the parameter's name becomes a literal `ast.List`, and the
    parameter is dropped from the signature (`emit._reconstruct_call_args` drops it
    from the call to match). Exactly the monomorphisation `_SpecializeHOF` performs for
    a function-valued argument, applied to a value-valued one.

    Two body shapes are then made Warp-legal:

    - **`a0, a1, a2, a3 = coefficients`** -- a tuple target assigned a literal list.
      Warp's codegen has no destructuring assignment over a list, so this is expanded
      into one plain assignment per element. Exact, and refuses (leaves the node alone,
      so the pass downstream still rejects it) on a length mismatch or a `Starred`
      target rather than guessing which element goes where.
    - **`coefficients[i]`** -- left as a `Subscript` of a literal `ast.List`, which
      `_RewriteListLookups` (running afterwards) already turns into an exact `wp.where`
      chain.

    Anything else that survives -- the parameter passed on to another function, say --
    leaves an `ast.List` standing in a position Warp cannot compile, and the run's own
    codegen round rejects that leaf by name. It is never silently mistranslated.
    """

    def __init__(self, mapping: dict):
        self.mapping = mapping
        self.substituted: set[str] = set()

    def _literal(self, values):
        return ast.List(
            elts=[ast.Constant(v if isinstance(v, bool) else float(v)) for v in values],
            ctx=ast.Load(),
        )

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id in self.mapping:
            self.substituted.add(node.id)
            return ast.copy_location(self._literal(self.mapping[node.id]), node)
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], (ast.Tuple, ast.List))
            and isinstance(node.value, ast.List)
        ):
            targets = node.targets[0].elts
            values = node.value.elts
            if len(targets) == len(values) and not any(
                isinstance(t, ast.Starred) for t in targets
            ):
                return [
                    ast.copy_location(ast.Assign(targets=[t], value=v), node)
                    for t, v in zip(targets, values)
                ]
        return node


def _apply_sequence_statics(fdef: ast.FunctionDef, mapping: dict, label: str) -> ast.FunctionDef:
    """`fdef` with every sequence static in `mapping` substituted and its parameter
    dropped. Raises `LeafError` if a named parameter is not actually in the signature
    (a resolver/emitter disagreement that must be loud, not silently ignored)."""
    all_params = list(fdef.args.posonlyargs) + list(fdef.args.args) + list(fdef.args.kwonlyargs)
    names = {a.arg for a in all_params}
    missing = sorted(set(mapping) - names)
    if missing:
        raise LeafError(
            f"{label}: sequence static(s) {missing} name no parameter of the leaf "
            f"(signature has {sorted(names)}) -- refusing rather than emitting a call "
            f"whose arity no longer matches"
        )
    sub = _SubstituteSequenceStatics(mapping)
    fdef = sub.visit(fdef)
    fdef.args.posonlyargs = [a for a in fdef.args.posonlyargs if a.arg not in mapping]
    fdef.args.args = [a for a in fdef.args.args if a.arg not in mapping]
    kept_kwonly, kept_defaults = [], []
    for a, d in zip(fdef.args.kwonlyargs, fdef.args.kw_defaults):
        if a.arg not in mapping:
            kept_kwonly.append(a)
            kept_defaults.append(d)
    fdef.args.kwonlyargs = kept_kwonly
    fdef.args.kw_defaults = kept_defaults
    ast.fix_missing_locations(fdef)
    return fdef


MULTI_RETURN_HELPERS = {"eq", "leq", "geq"}
"""Registry entries whose body returns a tuple. `_mirror_transpile`'s (and the tracked
prototype's) `returns_tuple` check only looks for a LITERAL `ast.Tuple` in a `Return`
node -- it does not know that `return geq(a, b)` returns whatever `geq` returns, so
every `constraint_*` function that just forwards to one of these (nearly all of them)
was wrongly annotated `-> wp.float64` and Warp only caught it at kernel-build time
("annotated as float64 but the code returns 4 values"), the same failure mode §85
diagnosed for the transpiler proper. Named here rather than generalised into full
call-graph return-type inference, which is real extra work for a fixed, small set of
pass-through callees."""


"""Transcendental mapping note.

`jnp.<f>` -> `wp.<f>` is not bit-exact for every f: measured in this env, `tanh` agrees
with JAX on 38.5 % of arguments, `sinh` 48.6 %, `cosh` 54.7 %, `asin` 77.0 %, `acos`
92.8 %, `exp` 86.1 % -- always within a few ulp (~1e-15 relative), never more.
`sin cos tan log sqrt atan floor ceil sign round pow` are exact.

These are deliberately NOT refused and NOT rerouted. The agreement gate is
`harness.AGREEMENT_RTOL` (1e-12), some three orders of magnitude above the worst of
them, and refusing them by name costs real coverage --
`calculate_divertor_geometry_conventional` uses `asin`, and refusing it dropped
`large_tokamak_nof` from 40 covered entries to 34. `gamma`/`gammaln` are a different
matter and stay in `REGISTRY`: Warp has no gamma function at all, so those are
load-bearing rather than cosmetic.
"""


def _mirror_transpile(fdef: ast.FunctionDef, name: str) -> str:
    """Mirrors `_tracked.transpile()`'s post-parse logic exactly (jnp->wp renames,
    literal wrapping, wp.float64 annotation, multi-return handling), starting from an
    already-parsed `ast.FunctionDef` (post specialisation/list-lookup rewriting)
    instead of `inspect.getsource(fn)` -- a specialised function has no real Python
    source of its own to re-extract."""
    t = _tracked.ToWarp()
    fdef = t.visit(fdef)
    if fdef.args.defaults or any(d is not None for d in fdef.args.kw_defaults):
        raise Unsupported("default argument value(s)")
    if fdef.args.vararg or fdef.args.kwarg:
        raise Unsupported("*args/**kwargs")
    f64 = ast.Attribute(value=ast.Name("wp", ast.Load()), attr="float64", ctx=ast.Load())
    args = fdef.args
    for a in list(args.args) + list(args.kwonlyargs) + list(args.posonlyargs):
        a.annotation = f64
    def _is_multi_return_call(value) -> bool:
        return isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and \
            value.func.id in MULTI_RETURN_HELPERS

    n_returns = 1
    for n in ast.walk(fdef):
        if isinstance(n, ast.Return):
            if isinstance(n.value, ast.Tuple):
                n_returns = max(n_returns, len(n.value.elts))
            elif _is_multi_return_call(n.value):
                n_returns = max(n_returns, 4)  # eq/leq/geq's fixed arity
    returns_tuple = n_returns > 1
    if not returns_tuple:
        fdef.returns = f64
    fdef.name = name
    fdef.decorator_list = [ast.Attribute(value=ast.Name("wp", ast.Load()), attr="func", ctx=ast.Load())]
    mod = ast.Module(body=[fdef], type_ignores=[])
    ast.fix_missing_locations(mod)
    return ast.unparse(mod), n_returns


class _Specialized:
    """A worklist entry for a monomorphised function: no real Python source, just an
    already-built (and already `_SpecializeHOF`-rewritten) `fdef` + its defining
    module's globals, used for further name resolution (helper scan, global consts)."""

    def __init__(self, fdef: ast.FunctionDef, globalns: dict):
        self.fdef = fdef
        self.globalns = globalns


def build_leaf_funcs_source(leaves, tolerant: bool = False, known_bad: dict = None,
                            already_emitted: set = None) -> tuple[str, dict]:
    """Transpile every distinct `(module, fn)` a leaf list needs.

    `already_emitted`: names of `REGISTRY` entries some earlier call has ALREADY put in
    the generated module. The harness runs this builder and the array-aware one over
    the same module and concatenates their output, and both can need the same helper --
    `_dd_fma` is `_xla_lgamma`'s FMA and also both `jnp.interp` helpers' fused final
    step -- so without a shared set it gets defined twice. The set is updated in place
    with everything this call emits.

    Returns `(source, fn_names)`: `source` is the full text of `@wp.constant`s plus
    `@wp.func` definitions (registry entries, specialisations, then leaves/helpers, in
    a stable order); `fn_names` maps `(module, fn)` -> the callable's name as emitted
    (identical to `Leaf.fn` for a real leaf -- the transpiler does not rename those;
    specialisations get their own minted name, not present in `fn_names`).

    `tolerant=False` (default) raises `LeafError` on the first leaf whose body -- or
    any helper/specialisation its body needs -- cannot be made self-contained. `True`
    instead skips just that leaf (reported in a third return value, `failed`) and
    continues with the rest.

    `known_bad`: `{(module, fn): reason}` for functions already known to fail --
    typically from a PREVIOUS round's Warp *codegen* failure (caught only once a kernel
    is actually built, not at transpile time). Without this, a helper reached only
    transitively (never itself a top-level leaf) that fails codegen gets silently
    re-pulled into the next round's closure by whichever leaf needs it, forever --
    excluding a top-level leaf from `leaves` does nothing to a helper's OWN closure
    resolution. Any leaf/helper whose key is in `known_bad` raises `LeafError`
    immediately, so the exclusion actually propagates to whatever top-level leaf needed
    it, the same as any other transpile-time failure.
    """
    known_bad = known_bad or {}
    already_emitted = already_emitted if already_emitted is not None else set()
    # Sequence-valued statics, per `(module, fn)`: baked into the emitted `@wp.func`'s
    # body (Warp has no tuple type, so they cannot be passed at the call site). Two
    # leaves sharing a function but binding DIFFERENT sequences would need two
    # differently-named specialisations; that does not occur on any configuration here,
    # and rather than silently emit one of the two, both are refused by name.
    seq_statics: dict[tuple, dict] = {}
    seq_conflict: dict[tuple, str] = {}
    for _leaf in leaves:
        if type(_leaf).__name__ == "StructuralOp":
            continue
        mapping = {
            n: tuple(v)
            for n, v in getattr(_leaf, "statics", ())
            if isinstance(v, (tuple, list))
        }
        if not mapping:
            continue
        k = (_leaf.module, _leaf.fn)
        if k in seq_statics and seq_statics[k] != mapping:
            seq_conflict[k] = (
                f"two leaves bind different sequence statics to {k[0]}.{k[1]} "
                f"({seq_statics[k]} vs {mapping}) -- one emitted `@wp.func` cannot be "
                f"both; refusing rather than picking one"
            )
        seq_statics[k] = mapping
    seen: dict[tuple, str] = {}
    failed: dict[tuple, str] = {}
    arities: dict[tuple, int] = {}
    const_defs: dict[str, str] = {}
    func_srcs: list[str] = []
    registry_used: set = set()
    spec_registry: dict = {}
    spec_queue: dict = {}
    spec_emitted: set = set()

    def _jnp_pi(src: str) -> str:
        for mod_alias in ("jnp", "np", "math"):
            src = re.sub(rf"(?<![\w.]){mod_alias}\.pi(?!\w)",
                          f"wp.float64({math.pi!r})", src)
        return src

    def _transpile_and_queue(obj, node_label: str, module: str, fn_name: str, worklist: list):
        if (module, fn_name) in known_bad:
            raise LeafError(
                f"{node_label} ({module}.{fn_name}): excluded -- known bad from a "
                f"previous round: {known_bad[(module, fn_name)]}"
            )
        if (module, fn_name) in seq_conflict:
            raise LeafError(f"{node_label}: {seq_conflict[(module, fn_name)]}")
        if isinstance(obj, _Specialized):
            fdef, globalns = obj.fdef, obj.globalns
        else:
            fdef = _fdef_of(obj)
            globalns = getattr(obj, "__globals__", {})
            mapping = seq_statics.get((module, fn_name))
            if mapping:
                fdef = _apply_sequence_statics(
                    fdef, mapping, f"{node_label} ({module}.{fn_name})"
                )
            fdef = _SpecializeHOF(globalns, spec_registry, spec_queue).visit(fdef)
            ast.fix_missing_locations(fdef)

        fdef_for_lookup = _RewriteListLookups().visit(copy.deepcopy(fdef))
        fdef_for_lookup = _RewriteBitwiseBool().visit(fdef_for_lookup)
        fdef_for_lookup = _RewriteInComparisons().visit(fdef_for_lookup)
        ast.fix_missing_locations(fdef_for_lookup)

        # Any `Subscript` still standing after the literal-list rewrite is indexing
        # something that is NOT a literal table -- almost always a genuinely
        # array-VALUED parameter (`ucsc[i_tf_sc_mat - 1]`, `ucsc` itself an array
        # argument, not a scalar). The transpile spec blanket-annotates every
        # parameter `wp.float64`; silently doing that to an array parameter produces a
        # cryptic Warp codegen error ("no overload for 'extract'") deep inside a
        # 1000-line kernel instead of a clear refusal here. Refuse now, by name.
        for node in ast.walk(fdef_for_lookup):
            if isinstance(node, ast.Subscript):
                target_desc = ast.unparse(node.value)
                raise LeafError(
                    f"{node_label} ({module}.{fn_name}): refused -- subscript "
                    f"`{target_desc}[...]` on something not a literal table -- "
                    f"an array-valued parameter, not a scalar (§76 Work bucket)"
                )

        try:
            src, n_returns = _mirror_transpile(copy.deepcopy(fdef_for_lookup), fn_name)
        except Unsupported as exc:
            raise LeafError(f"{node_label} ({module}.{fn_name}): refused -- {exc}") from exc
        src = _jnp_pi(src)
        arities[(module, fn_name)] = n_returns

        # Helper-call closure: PROVIDED, a newly-specialised name, or another
        # same-module function -- all appear as `Call(func=Name(...))` now that
        # `_SpecializeHOF` has already turned any function-passed-by-reference call
        # into a direct call by concrete name.
        for node in ast.walk(fdef):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                callee_name = node.func.id
                if callee_name in PROVIDED or callee_name in REGISTRY:
                    registry_used.add(callee_name)
                    registry_used.update(REGISTRY_DEPS.get(callee_name, ()))
                    continue
                if callee_name in spec_queue and callee_name not in spec_emitted:
                    sfdef, sglobalns, smodule = spec_queue[callee_name]
                    worklist.append((smodule, callee_name, _Specialized(sfdef, sglobalns)))
                    continue
                target = globalns.get(callee_name)
                if _is_models_fn(target) and (target.__module__, callee_name) not in seen:
                    worklist.append((target.__module__, callee_name, target))

        resolved, unresolved = _collect_globals(fdef, globalns)
        if unresolved:
            raise LeafError(
                f"{node_label} ({module}.{fn_name}): unresolved global(s) "
                f"{sorted(unresolved)} -- not a plain scalar, refusing rather than guessing"
            )
        for text, val in sorted(resolved.items(), key=lambda kv: -len(kv[0])):
            ident = _const_ident(text)
            const_defs.setdefault(ident, f"{ident} = wp.constant(wp.float64({val!r}))")
            src = re.sub(rf"(?<![\w.]){re.escape(text)}(?!\w)", ident, src)
        func_srcs.append(src)
        if isinstance(obj, _Specialized):
            spec_emitted.add(fn_name)

    for leaf in leaves:
        key = (leaf.module, leaf.fn)
        # A "Composition" leaf also needs every function its wrapper's PRELUDE calls
        # (`resolve.PreludeCall`) -- the locals it computes before the call that
        # produces the node's outputs. They go into the SAME per-leaf worklist, so a
        # prelude that cannot be transpiled fails (and rolls back) the whole leaf,
        # rather than leaving a kernel that calls an undefined `@wp.func`.
        pre_keys = [(pc.module, pc.fn) for pc in getattr(leaf, "prelude", ())]
        if key in failed or any(k in failed for k in pre_keys):
            continue
        if key in seen and all(k in seen for k in pre_keys):
            continue
        if leaf.fn in REGISTRY:
            if pre_keys:
                reason = (
                    f"{leaf.fn} is a REGISTRY leaf but node {leaf.node!r} carries a "
                    f"prelude ({[k[1] for k in pre_keys]}) -- the registry path emits "
                    f"no closure for it; refusing rather than emitting a call to an "
                    f"undefined function"
                )
                if tolerant:
                    failed[key] = reason
                    continue
                raise LeafError(reason)
            registry_used.add(leaf.fn)
            registry_used.update(REGISTRY_DEPS.get(leaf.fn, ()))
            seen[key] = leaf.fn
            arities[key] = REGISTRY_ARITY[leaf.fn]
            continue
        try:
            mod = importlib.import_module(leaf.module)
        except Exception as exc:
            reason = f"cannot import module {leaf.module!r}: {exc}"
            if tolerant:
                failed[key] = reason
                continue
            raise LeafError(f"{leaf.node}: {reason}") from exc
        fn = getattr(mod, leaf.fn, None)
        if fn is None:
            reason = f"{leaf.module}.{leaf.fn} not found"
            if tolerant:
                failed[key] = reason
                continue
            raise LeafError(f"{leaf.node}: {reason}")

        local_srcs_before = len(func_srcs)
        local_consts_before = dict(const_defs)
        local_registry_before = set(registry_used)
        local_spec_emitted_before = set(spec_emitted)
        local_arities_before = dict(arities)
        local_seen: dict = {}
        worklist = [(leaf.module, leaf.fn, fn)]
        try:
            for pmodule, pname in pre_keys:
                pmod = importlib.import_module(pmodule)
                pfn = getattr(pmod, pname, None)
                if pfn is None:
                    raise LeafError(f"{leaf.node}: prelude {pmodule}.{pname} not found")
                worklist.append((pmodule, pname, pfn))
        except Exception as exc:
            if tolerant:
                failed[key] = str(exc)
                continue
            raise
        try:
            while worklist:
                m, n, f = worklist.pop()
                k = (m, n)
                if k in seen or k in local_seen:
                    continue
                local_seen[k] = n
                _transpile_and_queue(f, leaf.node, m, n, worklist)
        except LeafError as exc:
            del func_srcs[local_srcs_before:]
            const_defs.clear()
            const_defs.update(local_consts_before)
            registry_used.clear()
            registry_used.update(local_registry_before)
            spec_emitted.clear()
            spec_emitted.update(local_spec_emitted_before)
            arities.clear()
            arities.update(local_arities_before)
            if tolerant:
                failed[key] = str(exc)
                continue
            raise
        seen.update(local_seen)

    parts = []
    if const_defs:
        parts.append("\n".join(const_defs[k] for k in sorted(const_defs)))
    emit_names = [n for n in registry_closure(registry_used) if n not in already_emitted]
    already_emitted.update(emit_names)
    if emit_names:
        parts.append("\n".join(REGISTRY[n] for n in emit_names))
    parts.extend(func_srcs)
    fn_names = {k: v for k, v in seen.items()}
    if tolerant:
        return "\n\n".join(parts), fn_names, failed, arities
    return "\n\n".join(parts), fn_names
