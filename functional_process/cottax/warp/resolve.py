"""Node -> leaf-function resolver, following §76's census approach: AST-parse the
wrapper's signature method (usually `__call__`, sometimes another -- `NodalDeclaration
._signature_of` names which) plus one level of same-module/self-method helper
expansion, rather than a naive walk over `ImplementedFunction.fn`'s own source (which
the §81 spike measured at 4/154 resolved on this graph's naive shape).

Wrapper shapes found on `helias_5b`/`stellarator_helias`'s SAND `Drive`, beyond the
plain `return calculate_x(a, b)` shape §81 already covered:

- **equinox `BoundMethod`.** `ImplicitFunction` nodes (e.g. `Intersect.residual`) hand
  back an equinox `Module` wrapping `__func__`/`__self__` as dataclass fields, not a
  real Python bound method -- `inspect.ismethod` says False and `inspect.getsource`
  refuses it outright. Unwrapped the same way a real bound method already is.
- **A `.fn` field, not a call.** `sand.py`'s `_NormalisedResidual`/`_Metric` (every
  `.ConstraintNN`/`.Objective` node) hold the ported `constraint_<id>`/
  `objective_metric_<id>` function as a *field*, called via `self.fn(**kwargs)` --
  there is no `ast.Call` naming it directly in the wrapper's own source.
- **The leaf passed as a value, not called.** `PlasmaCompositionIgnited.__call__`
  passes `plasma_composition_ignited` itself as the first *argument* to a shared
  `self._composition(arm, *args)` helper, which calls `arm(...)` internally after
  array-assembling its own arguments (`jnp.stack` over 14 impurity-array elements) --
  genuinely array-valued, so even a resolved leaf here would still be a Work-bucket
  function no transpiler covers today (§76). Detected and reported, not chased through
  the array assembly.
- **A same-module dispatch helper as the leaf itself**, not a `functional_process.
  models` function. `StellaratorBetaAndStoredEnergy.__call__` calls
  `select_stellarator_beta_and_stored_energy`, defined in the *same file*
  (`functional_process.cottax.stellarator.plasma_physics`) -- it is not itself further
  expanded (it discards one of three values a genuine models call underneath it
  returns, so treating THAT inner call as the leaf would misstate this node's output
  arity). Accepted as the leaf only when nothing else resolves and its own module
  matches the wrapper's, on the same reasoning `_NormalisedResidual`/`_Metric`'s `.fn`
  is accepted: a small pure function outside `functional_process.models`'s sweep scope
  is still a real, nameable leaf -- it is for the emitter/transpiler-coverage cross
  check to say whether a `@wp.func` exists for it, not this resolver.

`Pairwise`/`_Negate`/bare `operator.sub` are `cottax.rewrites`' own structural
comparison/sign nodes (every `^cond.*` residual and negation) and are classified
`structural`, not `unresolved`: there is no leaf to find, by construction.

The return-arity invariant
---------------------------
**A resolved leaf's return arity must equal the node's own declared output count
(`len(node_def.outputs)`).** This is checked, not assumed: `.buildings.sizing`
resolved to `calculate_shield_height` (4 args, 1 return) for a node declaring 12
outputs, because its wrapper computes `shh = calculate_shield_height(...)` INLINE and
then `return calculate_bldgs(..., shh, ...)` -- an AST walk that takes the first
`functional_process.models` call it finds has no way to prefer the second one, and the
mistake surfaced only as Warp's `TypeError: object of type 'Var' has no len()` when the
emitter tried to destructure 12 values from a 1-return function. That is the exact
"looked fine because nothing checked the thing that was wrong" shape this port has hit
before (§85, §88, §90).

`_find_matching_call`/`_find_via_helper` now restrict every candidate call to ones
whose OWN return arity (`_return_arity`/`_expr_arity`, an AST-only computation, no
tracing) equals the node's declared output count. Zero matches -> `Unresolved`. Exactly
one distinct matching target -> that one, at its textually LAST call site among
matches (`shh = f(...); return g(..., shh)` -- filtering by arity already excludes `f`
here, but the tie-break exists for the case where it would not). MORE than one distinct
target matching arity -> `Composition`, HELD rather than raised, because that message
can be an artifact: see "Composition" below.

**Argument binding was never the bug, and is now provably tied to the right call.** A
misselected leaf's own arguments were always read off ITS OWN call site (self-
consistent, just for the wrong function) -- `.buildings.sizing` never mixed
`calculate_bldgs`'s arguments with `calculate_shield_height`'s. The selection is
arity-correct, `_bind_call_args` reads arguments from that same correctly-selected
call, and an argument that is itself a prelude's local (`shh`) is now bound to the
value the wrapper computes for it -- see below.

Composition: a wrapper that computes locals before its leaf call
-----------------------------------------------------------------
The shape this resolver used to refuse outright. The body is not one leaf but a small
straight-line DAG:

    shh = calculate_shield_height(z, g, u, l)      # a local
    return calculate_bldgs(..., shh, ...)          # the node's leaf

`resolve()` now returns a `prelude` (`PreludeCall`s, in evaluation order) plus
`locals_` (which of the leaf's parameters read which prelude output), and the emitter
renders each prelude as its own `@wp.func` call before the leaf's. Four variants are
covered, each because a real node on these configurations has it:

- a named local (`.buildings.sizing`'s `shh`);
- a tuple-unpacked local (`.tokamak.build.tf_outboard_mid`'s
  `_, r_tf_outboard_midmin = ...`);
- an INLINE call in argument position, including at a `self.<helper>` call site
  (`.availability.electric_production`'s
  `self._production(centrepost_coolant_pump_power_absent(), ...)`);
- the node's OUTPUT itself being a local the `return` merely names, selected out of a
  wider return by its position in the unpack (`.power.delta_eta_step` --
  `_find_returned_local`). This last one is why `Composition` is held rather than
  raised: the arity filter drops the real producing call for being too wide, and the
  innocent single-value helpers left behind then look like an ambiguity. The message
  named four calls precisely BECAUSE it had discarded the right one; "ambiguous" there
  was filter ordering, not a property of the node.

**A local is bound to the value the wrapper actually computes for it, never to
anything else.** That is not a stylistic preference: binding `shh` to a same-named
boundary variable, a default, or a zero computes the wrong thing while a nearly-constant
local makes the error look like float noise. Two consequences are load-bearing:

- **Position matters** (`_Frame.local_at`). `DeltaEtaStep.step` opens with
  `p_fw_blkt_coolant_pump_mw = calculate_p_fw_blkt_coolant_pump_mw(...,
  p_fw_blkt_coolant_pump_mw)` -- the argument is the PARAMETER (a right-hand side is
  evaluated before its target is rebound), while every later reference is the LOCAL.
  Resolving that name one way everywhere is wrong in one direction or the other;
  Python's own rule is the only correct answer. Measured: mis-binding it to the
  parameter moves this node's output by 5.0e+00 relative.
- **Refuse, do not guess.** `_frame_locals` accepts only a straight-line run of
  `local = f(...)` assignments (plus `del`) before one `return`, and names its
  refusals: a conditionally or repeatedly assigned local, a non-call right-hand side,
  a chained assignment, a destructuring into non-names, a producer that is not a
  `functional_process.models` function, or one whose return arity disagrees with the
  unpack it feeds. An unresolvable or array-valued producer leaves the node
  `Unresolved`, with the reason.

Verified bit-exactly against the cottax WRAPPER ITSELF (`node_def.fn(...)`), not
against a second re-derivation of the same binding: every Composition node that
transpiles agrees to 0.000e+00, and deliberately mis-binding a local moves it off zero
(2.4e-02 for `shh`, 5.0e+00 for the shadowing near-miss).

Frozen (static) arguments
--------------------------
A leaf parameter may have no `VarPath` at all -- a switch frozen at assembly
(`ireactor`, `istell`, ...), a node's own field (`self.imp_indices`), or a bare
literal in the wrapper's source (`0.0`). `resolve()` reports these as `statics`
(`(name, value)` pairs, `value` a plain `float`/`int`/`bool` -- an `IntEnum` member is
converted via `int(...)`) rather than leaving the node unresolved, but only when the
value is a scalar a `wp.float64(...)`/`wp.int32(...)` literal can render, **or a
fixed-length sequence of such scalars** (`_static_sequence_value` -- read that
function's docstring for why reading a bound `self.<attr>` is a fact rather than a
guess, and for the constancy evidence for each attribute this actually reaches). A
sequence static is reported the same way but carries a `tuple` value instead of a
scalar; because Warp has no tuple type it is NOT rendered at the call site -- the
emitter monomorphises the leaf around it. An array, a dataclass, a dict or a callable
in that argument position still leaves the node `Unresolved`, with the reason (now
naming the value's actual runtime type), rather than guessing at a rendering.

**Ordering contract for the caller (`leaves.py`/the emitter).** `resolve()` returns
`(leaf_fn, order, inputs, statics, output_index, locals_, prelude)`:

- `order` is every one of `leaf_fn`'s own parameter names, in its signature order --
  the full call the emitter must reconstruct.
- `inputs` is the `VarPath` for each name in `order` that is dynamic, in the same
  relative order those names appear in `order` (so `inputs`'s i-th entry is the
  VarPath for the i-th *dynamic* name in `order`, not for `order[i]` itself).
- `statics` is `(name, value)` for each name in `order` that is frozen, same rule.
- `locals_` is `(name, local identifier)` for each name in `order` bound to a
  `PreludeCall`'s output, same rule.
- `output_index` is `None`, or one position per declared output within `leaf_fn`'s
  WIDER return.
- `prelude` is the `PreludeCall`s, in evaluation order.

The emitter zips `order` against `statics`/`locals_`/`inputs` **by name**: walk
`order`, and for each entry look it up in `dict(statics)` first (present -> render a
literal), then in `dict(locals_)` (present -> the prelude local of that name), else pop
the next value off `inputs` in sequence (present -> read the value already in hand for
that `VarPath`). This is the "add `order`" option from the two the interface offered,
chosen because a name-keyed re-zip is one dict build and a linear walk, with no
sentinel value to define or accidentally collide with a real argument.
"""
import ast
import copy
import dataclasses
import enum
import inspect
import textwrap


class Unresolved(Exception):
    """A node this resolver's rules do not reach -- carries a reason, not a guess."""


class Structural(Exception):
    """A node that is provably not a `functional_process.models` leaf (a comparison,
    residual, or sign-flip primitive from `cottax.rewrites`/`functional_process.cottax.
    sand`) -- expected, and reported separately from a resolver failure."""


class Composition(Exception):
    """Two or more DISTINCT `functional_process.models` (or same-module dispatch)
    calls in one wrapper each return exactly the node's declared output count.
    Reported separately from `Unresolved` so a caller can tell "no candidate found"
    from "more than one, and picking either would be a guess".

    **Raised only as a last resort.** `resolve()` holds it while it tries the
    returned-local path (`_find_returned_local`), because this message is an artifact
    whenever the real producing call was dropped by the arity filter for being wider
    than the node's output count -- the surviving single-value helpers then look
    ambiguous. `.power.delta_eta_step` was exactly that and no longer raises this."""


_STRUCTURAL_MODULES = ("cottax.rewrites",)
_STRUCTURAL_NAMES = {"Pairwise", "_Negate"}


from process.core.model import DataStructure as _DataStructureForShapes  # noqa: E402

_DS_SINGLETON = _DataStructureForShapes()
"""One `DataStructure`, read once, for its declared field defaults -- the source of
truth for whether a bound `VarPath` names a scalar or a fixed-shape table."""


def _is_models_fn(obj):
    return (
        inspect.isfunction(obj)
        and str(getattr(obj, "__module__", "")).startswith("functional_process.models")
    )


def _unwrap_bound(fn):
    """A real function/method as-is; an equinox `BoundMethod` (`__func__`/`__self__`
    dataclass fields, not `inspect.ismethod`-true) rebound into a real one; anything
    else's own `__call__` if that is itself a function/method."""
    if inspect.isfunction(fn) or inspect.ismethod(fn):
        return fn
    func, self_ = getattr(fn, "__func__", None), getattr(fn, "__self__", None)
    if inspect.isfunction(func) and self_ is not None:
        return func.__get__(self_)
    call = getattr(fn, "__call__", None)
    if inspect.isfunction(call) or inspect.ismethod(call):
        return call
    return None


def _source_and_params(fn):
    """`(ast.FunctionDef, [param names excluding self/cls], globalns, definer, module,
    self_instance)`.

    `definer` is the CLASS holding `real` (or `real` itself for a plain function) --
    right for resolving a `self.<method>(...)` call TARGET (an unbound function lives
    on the class). `self_instance` is the actual bound receiver (`None` for a plain
    function) -- required for resolving a `self.<field>` VALUE, since an equinox
    dataclass field is an instance attribute the class itself does not carry. Conflating
    the two silently resolved every `self.<field>` read to `None` (`getattr(the class,
    ...)` on a field with no class-level default) until this was found while adding
    static-argument support -- `self.i_p_coolant_pumping` genuinely IS a plain
    `IntEnum` member at runtime, just not reachable through the class.
    """
    real = _unwrap_bound(fn)
    if real is None:
        raise Unresolved(f"fn {fn!r} is not a function/method and has no __call__")
    holder = getattr(real, "__self__", None)
    definer = type(holder) if holder is not None else real
    src = textwrap.dedent(inspect.getsource(real))
    tree = ast.parse(src)
    fdef = tree.body[0]
    if not isinstance(fdef, ast.FunctionDef):
        raise Unresolved(f"source of {real!r} is not a single def")
    # Keyword-ONLY parameters (`def _masses(self, *, len_tf_coil, ...)`) live in
    # `kwonlyargs`, not `args`. Omitting them made every keyword-only helper look
    # zero-parameter, so `_find_via_helper` built an EMPTY outer->inner map and every
    # one of its arguments then failed to bind -- which is exactly the "argument
    # 'len_tf_coil' ... is neither a traced parameter nor a static scalar" refusal on
    # `.tokamak.cicc_superconducting_tf_coil.superconducting_tf_coil_areas_and_masses`.
    # `inspect.signature` orders positional-then-keyword-only, and so does this.
    params = [
        a.arg
        for a in list(fdef.args.args) + list(fdef.args.kwonlyargs)
        if a.arg not in ("self", "cls")
    ]
    globalns = getattr(real, "__globals__", {})
    module = getattr(real, "__module__", None)
    return fdef, params, globalns, definer, module, holder


def _resolve_name(node, globalns, definer):
    """The object an `ast.Name`/`ast.Attribute` expression names -- `None` if it is
    neither a module global nor a `self.<attr>`."""
    if isinstance(node, ast.Name):
        return globalns.get(node.id)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if node.value.id == "self":
            return getattr(definer, node.attr, None)
        base = globalns.get(node.value.id)
        return getattr(base, node.attr, None) if base is not None else None
    return None


def _calls_in(fdef):
    return [n for n in ast.walk(fdef) if isinstance(n, ast.Call)]


def _direct_returns(fdef):
    """Every `ast.Return` textually inside `fdef`'s own body -- not one belonging to a
    nested `def` (there are none in this port's leaf functions, but the walk should not
    silently attribute one to the wrong function if that ever changes)."""
    returns = []

    class _V(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            if node is fdef:
                self.generic_visit(node)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Lambda(self, node):
            pass  # a lambda cannot contain a `return`; nothing to skip into

        def visit_Return(self, node):
            returns.append(node)

    _V().visit(fdef)
    return returns


_SCALAR_CALL_NAMESPACES = ("jnp", "np", "math", "lax")
"""`return jnp.where(...)`/`np.sqrt(...)`/`lax.select(...)`/etc. -- every function this
transpiler's own `DIRECT`/`RENAME`/`TERNARY` tables know is elementwise-scalar in this
codebase's scalar-typed world (`transpile.py`'s whole design commits every value to
`wp.float64`), so a bare `return jnp.X(...)` is confidently arity 1 without needing to
inspect `jnp.X` itself. `lax` (`from jax import lax`) is here because `safe_sqrt`/
`safe_pow` (`functional_process.models.safe_math`, the `PROVIDED` registry's own two
entries) are built on `lax.select`/`lax.full_like`, not `jnp`."""

_FOUR_TUPLE_HELPERS = {
    ("functional_process.cottax.core.solver.constraints", "eq"),
    ("functional_process.cottax.core.solver.constraints", "leq"),
    ("functional_process.cottax.core.solver.constraints", "geq"),
}
"""`(module, name)` of `eq`/`leq`/`geq` -- confirmed by reading
`functional_process/cottax/core/solver/constraints.py:59-95`: each returns exactly
`(residual, normalised_residual, constraint_value, constraint_bound)`, always in that
order, no branching. Every `constraint_<id>` this port ships whose body is `return
eq(...)`/`leq(...)`/`geq(...)` (`constraint_11`/`32`/`34`/`35`/`65`/`82`/`83` and
others) is therefore confidently arity 4 -- named here the same way
`_SCALAR_CALL_NAMESPACES` names `jnp`/`np`/`math`/`lax`, rather than left as an
unrecognised call that silently defaults `_expr_arity` to `None`
(`leaf_funcs.py`'s independently-maintained `MULTI_RETURN_HELPERS`/`REGISTRY_ARITY`
already encode the identical fact for the Warp transpiler side; this is the resolver's
own copy of the same, small, closed set)."""


def _expr_arity(value, globalns, seen):
    """How many values the expression `value` represents, as a `return`'s value or one
    element of a `return`'s tuple -- `None` if this cannot say a single number.

    Four shapes recognised, each because a real leaf function in this port uses it,
    not speculatively:

    - **`a, b, c`** (`ast.Tuple`) -- `len(elts)`, but a `Starred` element
      (`*_cryo_cool_req_no_aluminium(...)`, `calculate_cryo_plant_loads_active`'s own
      return) is expanded recursively rather than counted as one: `len(elts)` alone
      undercounts a starred unpack.
    - **`jnp.X(...)`/`np.X(...)`/`math.X(...)`** -- confidently `1`, the same
      elementwise-scalar assumption `transpile.py`'s own `DIRECT`/`RENAME` tables make.
    - **`other_leaf(...)`**, `other_leaf` a `functional_process.models` function --
      resolved recursively via `_return_arity` (cycle-guarded by `seen`), since its
      shape is knowable, just not without one more look
      (`calculate_detailed_powerflow_blanket_shield_power_user_input_pumping`'s
      `return _detailed_powerflow_core(...)[:12]` needs this one level down through
      the `Subscript` case below, which needs it again for the `Call` it slices).
    - **`X(...)[a:b]`** (`ast.Subscript` of a `Slice` with literal bounds) -- the
      slice's own length (`b - a`, `lower` defaulting to `0`), independent of the
      sliced value's arity: a fixed-length prefix of a longer tuple is exactly that
      many values regardless of how long the source tuple actually is. A **plain**
      subscript (`results[4]`, not a slice) is one value, not resolved further.

    Anything else -- a bare name, a `BinOp`, an unrecognised call (`constraint_84`'s
    `return geq(...)`, which actually returns a 4-tuple, not "1 value" -- an earlier
    revision of this function wrongly defaulted to 1 here) -- returns `1` only for a
    genuine single expression, `None` for a call this cannot place, so a caller
    comparing against a declared output count never treats "unknown" as "matches".
    """
    if isinstance(value, ast.Tuple):
        total = 0
        for e in value.elts:
            if isinstance(e, ast.Starred):
                inner = _expr_arity(e.value, globalns, seen)
                if inner is None:
                    return None
                total += inner
            else:
                total += 1
        return total
    if isinstance(value, ast.Call):
        f = value.func
        if (
            isinstance(f, ast.Attribute)
            and isinstance(f.value, ast.Name)
            and f.value.id in _SCALAR_CALL_NAMESPACES
        ):
            return 1
        target = _resolve_name(f, globalns, None)
        if (
            target is not None
            and (getattr(target, "__module__", None), getattr(target, "__name__", None))
            in _FOUR_TUPLE_HELPERS
        ):
            return 4
        if _is_models_fn(target) and id(target) not in seen:
            # `seen` is passed through UNCHANGED here, not pre-loaded with
            # `id(target)`: `_return_arity` itself checks `id(fn) in _seen` on entry
            # and adds itself before descending back into `_expr_arity` -- pre-adding
            # the target's id here made every recursive call trip that same guard
            # against ITSELF on the very first line, so every `return other_leaf(...)`
            # silently came back `None` regardless of `other_leaf`'s real shape.
            return _return_arity(target, seen)
        return None
    if isinstance(value, ast.Subscript):
        sl = value.slice
        if isinstance(sl, ast.Slice) and sl.step is None:
            lower = (
                0
                if sl.lower is None
                else (sl.lower.value if isinstance(sl.lower, ast.Constant) else None)
            )
            upper = sl.upper.value if isinstance(sl.upper, ast.Constant) else None
            if isinstance(lower, int) and isinstance(upper, int) and upper >= lower:
                return upper - lower
            return None
        return 1  # a plain index (`x[4]`) is one value, not a slice
    return 1


def _return_arity(fn, _seen=None):
    """How many values `fn` returns -- see `_expr_arity` for what each `return`'s value
    can look like -- or `None` if this cannot say a single number (no source, no
    `return` found, or inconsistent arity across multiple `return` statements/branches).

    **The correctness invariant this resolver exists to enforce**: a candidate call is
    only ever accepted as a node's leaf when this equals the node's own declared output
    count (`len(node_def.outputs)`) -- seeing `TypeError: object of type 'Var' has no
    len()` from Warp is what caught `.buildings.sizing` resolving to
    `calculate_shield_height` (4 args, 1 return) for a node declaring 12 outputs, whose
    real leaf, `calculate_bldgs`, was a second call two lines later in the same wrapper.
    An AST walk that takes the FIRST `functional_process.models` call it finds has no
    way to tell those apart; comparing return arity to declared output count does.
    """
    _seen = _seen or frozenset()
    if id(fn) in _seen:
        return None
    try:
        src = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(src)
        fdef = tree.body[0]
    except (OSError, TypeError, SyntaxError):
        return None
    if not isinstance(fdef, ast.FunctionDef):
        return None
    returns = _direct_returns(fdef)
    if not returns:
        return None
    globalns = getattr(fn, "__globals__", {})
    arities = set()
    for r in returns:
        arities.add(0 if r.value is None else _expr_arity(r.value, globalns, _seen | {id(fn)}))
    return next(iter(arities)) if len(arities) == 1 and None not in arities else None


def _find_matching_call(fdef, globalns, definer, expected_arity, extra_module=None):
    """A call whose `func` names a `functional_process.models` function directly, or
    (only when `extra_module` is given, as a last resort -- see the module docstring's
    "same-module dispatch helper" case) a plain function defined in `extra_module` --
    restricted to candidates whose OWN return arity equals `expected_arity` (the
    node's declared output count).

    Zero matching candidates -> `None` (nothing found at this level; the caller tries
    the next strategy). Exactly one DISTINCT target function -> that target, at its
    textually LAST call site among the matches (`shh = f(...); return g(..., shh)`
    -- `f` is filtered out by arity, but if it were not, the real leaf is the one
    feeding the `return`, not the prelude computation before it). More than one
    distinct target both matching arity -> raises `Composition`: genuinely ambiguous,
    not something to pick between.
    """
    candidates = []
    for call in _calls_in(fdef):
        target = _resolve_name(call.func, globalns, definer)
        is_candidate = _is_models_fn(target) or (
            extra_module is not None
            and inspect.isfunction(target)
            and getattr(target, "__module__", None) == extra_module
            and isinstance(call.func, ast.Name)
        )
        if not is_candidate or _return_arity(target) != expected_arity:
            continue
        candidates.append((target, call))
    if not candidates:
        return None
    distinct = {id(t): t for t, _ in candidates}
    if len(distinct) > 1:
        names = sorted({t.__name__ for t in distinct.values()})
        raise Composition(
            f"{len(distinct)} distinct call(s) -- {names} -- each return "
            f"{expected_arity} value(s), matching the node's declared output count; "
            f"which is the leaf is genuinely ambiguous, not a guess this resolver "
            f"will make"
        )
    target = next(iter(distinct.values()))
    call = max(
        (c for t, c in candidates if t is target), key=lambda c: (c.lineno, c.col_offset)
    )
    return target, call


def _find_value_passed(fdef, globalns, definer):
    """A models function passed as an ARGUMENT (not called) to some helper -- the
    `self._composition(plasma_composition_ignited, ...)` shape. Reported, not chased:
    the helper still has to be inspected to know the leaf's real argument list, and the
    one seen assembles an array from several of the wrapper's own parameters first
    (`jnp.stack`), which is genuinely outside what a positional VarPath list can state
    without re-deriving the array-construction logic.
    """
    for call in _calls_in(fdef):
        for arg in list(call.args) + [kw.value for kw in call.keywords]:
            target = _resolve_name(arg, globalns, definer)
            if _is_models_fn(target):
                return target
    return None


def _array_param_shapes(node_def, params):
    """`{param_name: shape}` for each of the node's own parameters whose bound
    `VarPath` names a genuinely array-valued `DataStructure` field.

    The shape comes from the field's own declared default (`DataStructure()`), which
    is where PROCESS's fixed tables actually live -- `temp_impurity_keV_array` and
    `impurity_arr_zav` are `(14, 200)` there and in every run. A parameter whose bound
    path is not a plain `.<area>.<name>` field, or whose default is not a rectangular
    numeric array, is left scalar -- the same default the scalar transpiler already
    applies; only a body that indexes it can be hurt by that, and that body refuses
    on the surviving `Subscript` rather than guessing.
    """
    import numpy as _np
    shapes = {}
    for name, inp in zip(params, node_def.inputs):
        path = inp.var.path_str()
        parts = path.strip(".").split(".")
        if len(parts) != 2 or path.startswith("^"):
            continue
        area, field = parts
        sub = getattr(_DS_SINGLETON, area, None)
        if sub is None:
            continue
        val = getattr(sub, field, None)
        if val is None:
            continue
        try:
            arr = _np.asarray(val, dtype=float)
        except (TypeError, ValueError):
            continue
        if arr.ndim >= 1 and arr.size > 1:
            shapes[name] = tuple(arr.shape)
    return shapes


def _resolve_built_composition(node_def, fn, params, expected_arity):
    """The leaf a node passes BY VALUE into one of its own helpers.

    `PlasmaCompositionIgnited.__call__` does
    `self._composition(plasma_composition_ignited, *args)`; its sibling passes
    `functools.partial(plasma_composition_non_ignited, f_nd_beam_electron=...)`. The
    function-valued argument is the leaf, but it is never *called* at this level, and
    the helper it goes into assembles a fourteen-entry species array out of twelve of
    the wrapper's own scalar parameters before forwarding -- so there is no single
    existing function whose parameter list the node's `VarPath`s can be laid against.

    Rather than chase the value, this BUILDS the leaf: `scalarise.scalarise_function`
    expands the node's own `__call__` -- helper, arm, `jax.vmap` and all -- into one
    flat straight-line function of exactly the node's declared parameters, with the
    fixed-length arrays expanded to named scalars. The arm is monomorphised by
    construction (the expansion inlines whichever one this occupant names), so one
    occupant gets one built leaf.

    **The return-arity invariant is enforced here, at the earliest point it can be**:
    the built function's return count must equal the node's declared output count. It
    does not hold by construction -- the destructuring the helper performs
    (`results[:4]`, `results[4][H_INDEX]`, `results[5:]`) is exactly where a
    miscounted splice would show -- so it is checked, and a mismatch refuses.

    Returns `(leaf_fn, order, inputs, statics)` or raises `Unresolved`.
    """
    from . import leaf_funcs_arrays as _at
    from . import scalarise as _sc

    self_obj = getattr(fn, "__self__", None)
    if self_obj is None:
        raise Unresolved("leaf is passed as a value, but the node has no bound "
                          "instance to expand from")
    shapes = _array_param_shapes(node_def, params)
    kinds = {name: ("array", sh) for name, sh in shapes.items()}
    module = type(self_obj).__module__
    name = f"_built_{type(self_obj).__name__}"
    try:
        fdef, globalns, built_params, n_returns = _sc.scalarise_function(
            fn, name, kinds)
    except _sc.ScalariseError as exc:
        raise Unresolved(
            f"leaf is passed as a value into an array-assembling helper, and the "
            f"array expansion that would build it refuses: {exc}"
        ) from exc
    if list(built_params) != list(params):
        raise Unresolved(
            f"the built leaf's parameters {built_params} are not the node's own "
            f"{params} -- an input was expanded or dropped, so the VarPath binding "
            f"would no longer be positional; refusing"
        )
    if n_returns != expected_arity:
        raise Unresolved(
            f"the built leaf returns {n_returns} value(s) but the node declares "
            f"{expected_arity} output(s) -- the destructuring this expansion "
            f"reproduced does not match what the node owns; refusing"
        )
    built = _compile_built(fdef, globalns, name, module)
    entry = _at._SynthLeaf(fdef, globalns, shapes, n_returns)
    entry.callable = built
    _at.SYNTH_LEAVES[(module, name)] = entry
    return (built, tuple(params), tuple(i.var for i in node_def.inputs),
            (), None, (), ())


def _compile_built(fdef, globalns, name, module):
    """The expanded body as a REAL, callable Python function.

    Not a stub: this is what the JAX side of the agreement check evaluates, so the
    comparison is Warp-against-JAX over the *identical* expanded body rather than
    against a second, independently-written reference. (That the expansion itself
    reproduces the real node is a separate check --
    `check_scalarise.py`, 18/18 outputs bit-identical for both occupants -- and it has
    to be separate: an agreement check can only ever say the two engines agree with
    each other.)
    """
    ns = dict(globalns)
    exec(compile(ast.Module(body=[copy.deepcopy(fdef)], type_ignores=[]),
                 f"<built {module}.{name}>", "exec"), ns)
    f = ns[name]
    f.__module__ = module
    return f


def _find_via_helper(fdef, outer, ctx, expected_arity):
    """One level of same-module/self-method helper expansion: a call to a `self.
    <method>` that is not itself a models leaf, tried once more from the inside --
    still under the same return-arity restriction (`_find_matching_call`), so a helper
    with its own prelude-then-leaf shape does not repeat `.buildings.sizing`'s bug one
    level down.

    Returns `(leaf_fn, call, inner_frame)`. The inner frame's parameter bindings are
    read off the helper's OWN call site in the outer frame, so an argument there that
    is an inline call (`self._production(centrepost_coolant_pump_power_absent(), ...)`)
    becomes a real `PreludeCall` rather than an unbindable name one level down.

    Outer arguments are handed to the inner frame as DEFERRED bindings, resolved only
    when the leaf actually reads one; a failure there names the outer expression that
    could not be translated instead of a generic "not a traced parameter".
    """
    for call in _calls_in(fdef):
        target = _resolve_name(call.func, outer.globalns, outer.definer)
        if target is None or _is_models_fn(target):
            continue
        if getattr(target, "__module__", None) is None:
            continue
        is_self_method = (
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "self"
        )
        if not is_self_method:
            continue
        try:
            inner_fdef, inner_params, inner_globalns, inner_definer, _mod, inner_self = (
                _source_and_params(target)
            )
        except Unresolved:
            continue
        found = _find_matching_call(
            inner_fdef, inner_globalns, inner_definer, expected_arity
        )
        if found is None:
            if _find_value_passed(inner_fdef, inner_globalns, inner_definer) is not None:
                raise Unresolved(
                    "leaf is passed as a value into an array-assembling helper "
                    "(genuinely array-valued -- §76's Work bucket)"
                )
            continue
        deferred = {}
        positional = list(call.args)
        for i, p in enumerate(inner_params):
            expr = (
                positional[i]
                if i < len(positional)
                else next((kw.value for kw in call.keywords if kw.arg == p), None)
            )
            if expr is not None:
                deferred[p] = (expr, outer)
        inner_frame = _Frame(
            inner_fdef, inner_globalns, inner_definer, inner_self, {}, deferred
        )
        return found[0], found[1], inner_frame
    return None


def _static_value(node, globalns, definer, self_instance=None):
    """`node` as a frozen scalar (`float`/`int`/`bool`), or `None` if it is not one --
    a bare numeric/bool literal (`0.0`, `-1e-5`), or a name/attribute (`self.field`, a
    module global, an `IntEnum` member) resolving to a plain scalar. An `IntEnum`
    member renders as its `int(...)` value: Warp has no enum type, and a switch
    argument is always compared/branched on as an integer in the ported body anyway.

    `self_instance` -- the actual bound receiver, not `definer` (its class) -- is what
    `self.<field>` must resolve against: a dataclass field is an instance attribute,
    which the class itself does not carry (found via `self.i_p_coolant_pumping`, a
    genuine `PumpingPowerModelTypes` member at runtime, resolving to `None` through the
    class and only through the instance). `None` here falls back to `definer`, which is
    right for a plain module-level function (no `self` at all).

    Anything else this cannot classify -- an array, a dataclass, a dict, a callable --
    returns `None` so the caller leaves the node `Unresolved` rather than guessing at a
    rendering (§80: Warp is strictly typed, no promotion).
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool)):
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
        and not isinstance(node.operand.value, bool)
    ):
        return -node.operand.value
    val = _resolve_name(node, globalns, self_instance if self_instance is not None else definer)
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, enum.Enum):
        try:
            return int(val)
        except (TypeError, ValueError):
            return None
    if isinstance(val, (int, float)):
        return val
    return None


@dataclasses.dataclass(frozen=True)
class PreludeCall:
    """One LOCAL computation a wrapper performs before the call that produces the
    node's outputs -- the "Composition" shape this resolver used to refuse outright:

        def __call__(self, ..., z, g, u, l, ...):
            shh = calculate_shield_height(z, g, u, l)   # <- a PreludeCall
            return calculate_bldgs(..., shh, ...)       # <- the node's leaf

    Emitted as its own `@wp.func` call, assigned to `targets`, BEFORE the leaf call --
    so `shh` is the value the wrapper actually computes, never a same-named boundary
    variable, a default, or a zero. Binding a local to anything but its own producer
    is the specific bug this class exists to make impossible: a near-constant local
    bound wrongly shows up as a small relative difference that reads like float noise,
    not as a failure.

    Fields mirror `Leaf`'s ordering contract exactly (`order`/`inputs`/`statics`, plus
    `locals_` for an argument that is ITSELF an earlier prelude's output), so one
    renderer serves both.
    """

    fn: str
    module: str
    order: tuple
    inputs: tuple
    """`VarPath` (not `path_str`) for each dynamic argument, in `order`'s dynamic
    order. `leaves.py` stringifies these the same way it does a `Leaf`'s."""
    statics: tuple
    locals_: tuple
    """`(parameter name, local identifier)` for an argument bound to an EARLIER
    prelude's target."""
    targets: tuple
    """The local identifier(s) this call's return value(s) are assigned to, in return
    order -- length equals the callee's own return arity, checked against
    `_return_arity` at resolve time and against the transpiled arity at emit time."""
    source: str
    """The wrapper's own text for this statement, for reporting."""


class _Ctx:
    """Per-`resolve()` prelude accumulator plus a fresh-identifier mint. Identifiers
    are unique within one node; the emitter namespaces them by node
    (`f"{leaf.node}::{ident}"`) so they cannot collide across nodes or with a real
    `VarPath`."""

    def __init__(self, allow_sequence_static=False):
        self.prelude = []
        self._n = 0
        self.allow_sequence_static = allow_sequence_static
        """Whether a frozen SEQUENCE argument binds as a monomorphisable `statics`
        entry (see `_bind_expr`). Off on the first pass so `resolve()` can offer the
        node to the body expansion first; on for its fallback pass."""

    def ident(self, hint):
        self._n += 1
        h = hint if hint and hint != "_" else "tmp"
        return f"_loc{self._n}_{h}"


def _assigned_names(fdef):
    """Every name BOUND anywhere in `fdef`'s own scope (not a nested `def`/`lambda`/
    comprehension, each of which has its own). Used only to decide whether a `Name`
    argument might be shadowing a parameter -- a cheap pre-check, so a wrapper with no
    locals at all never pays for the strict straight-line analysis below."""
    names = set()

    class _V(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            if node is fdef:
                self.generic_visit(node)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Lambda(self, node):
            pass

        visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = visit_Lambda

        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Store):
                names.add(node.id)

    _V().visit(fdef)
    return names


def _frame_locals(fdef):
    """Ordered `{local name: (targets, call node)}` for a body that is exactly a
    (optional docstring, then) run of `local = f(...)` assignments followed by ONE
    `return` -- the only shape whose locals this resolver will translate.

    Raises `Unresolved`, naming the offender, for every other shape. That refusal list
    is deliberate and is the honest half of this feature:

    - a local assigned inside an `if`/`else`/`for`/`try`/`with` (conditional or looped
      -- the emitted kernel is straight-line, so translating one would mean inventing
      a `wp.select` this resolver has no basis to write);
    - a local assigned more than once (`x = f(); x = g(x)` -- the leaf's `x` is the
      second, and quietly binding the first is exactly the wrong-value bug);
    - a local whose right-hand side is not a call (`shh = a + b` -- a real shape, just
      not one this translates yet; refused rather than guessed);
    - a chained (`a = b = f()`) or destructuring-into-non-names target.

    `_` is exempted from the assigned-twice rule (a discarded tuple slot) and is never
    looked up as a local.
    """
    body = list(fdef.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if not body or not isinstance(body[-1], ast.Return):
        raise Unresolved(
            "the wrapper body does not end in a single `return` -- its locals are not "
            "a straight-line prelude this resolver translates"
        )
    locals_, deleted = {}, set()
    for index, stmt in enumerate(body[:-1]):
        if isinstance(stmt, ast.Delete) and all(
            isinstance(t, ast.Name) for t in stmt.targets
        ):
            # `del delta_eta` (`DeltaEtaStep.step`): the parameter is deliberately
            # unused. Recorded so it can be REMOVED from the frame's bindings -- after
            # a `del` the name is gone, so anything still reading it would be a
            # `NameError` at runtime, and binding it here would be inventing a read
            # the wrapper does not perform.
            deleted.update(t.id for t in stmt.targets)
            continue
        if not isinstance(stmt, ast.Assign):
            raise Unresolved(
                f"the wrapper body contains a `{type(stmt).__name__}` statement before "
                f"its `return` -- only a straight-line run of `local = f(...)` "
                f"assignments is translated; a conditionally or repeatedly assigned "
                f"local is refused, not guessed"
            )
        if len(stmt.targets) != 1:
            raise Unresolved(
                f"chained assignment `{ast.unparse(stmt)}` -- refused, not guessed"
            )
        if not isinstance(stmt.value, ast.Call):
            raise Unresolved(
                f"local `{ast.unparse(stmt.targets[0])}` is assigned "
                f"`{ast.unparse(stmt.value)}`, which is not a call -- refused rather "
                f"than guessing its Warp form"
            )
        tgt = stmt.targets[0]
        if isinstance(tgt, ast.Name):
            targets = (tgt.id,)
        elif isinstance(tgt, ast.Tuple) and all(
            isinstance(e, ast.Name) for e in tgt.elts
        ):
            targets = tuple(e.id for e in tgt.elts)
        else:
            raise Unresolved(
                f"assignment target `{ast.unparse(tgt)}` is not a name or a tuple of "
                f"names -- refused"
            )
        for t in targets:
            if t != "_" and t in locals_:  # noqa: SIM102 -- kept flat for the message
                raise Unresolved(
                    f"local {t!r} is assigned more than once -- which assignment the "
                    f"leaf's argument refers to is not something this resolver will "
                    f"guess"
                )
        for t in targets:
            locals_[t] = (targets, stmt.value, index)
    return locals_, deleted


class _Frame:
    """One function body being read (the wrapper's `__call__`, or a `self.<helper>`
    one level in), with the bindings its parameters already carry.

    A binding is a `(kind, value)` pair: `("var", VarPath)` (a value the graph
    supplies), `("static", scalar)` (frozen at assembly), or `("local", ident)` (an
    earlier `PreludeCall`'s output)."""

    def __init__(self, fdef, globalns, definer, self_instance, binds, deferred=None):
        self.fdef = fdef
        self.globalns = globalns
        self.definer = definer
        self.self_instance = self_instance
        self.binds = dict(binds)
        self.deferred = dict(deferred or {})
        """`param name -> (outer expression, outer frame)` for a helper's parameters,
        bound LAZILY the first time the leaf actually reads one. Lazy on purpose: an
        eager pass would materialise a `PreludeCall` for every argument at the helper's
        call site, including ones the leaf never reads -- dead code in the kernel, and
        a needless way for an untranspilable-but-unused expression to sink the node."""
        self._assigned = None
        self._locals = None
        self._resolved = {}
        self._in_progress = set()

    @property
    def assigned(self):
        if self._assigned is None:
            self._assigned = _assigned_names(self.fdef)
        return self._assigned

    @property
    def locals(self):
        if self._locals is None:
            self._locals, deleted = _frame_locals(self.fdef)  # may raise Unresolved
            for name in deleted:
                self.binds.pop(name, None)
                self.deferred.pop(name, None)
        return self._locals

    def local_at(self, name, pos):
        """The `(targets, call, index)` entry for `name` if the wrapper assigns it
        STRICTLY BEFORE statement `pos`, else `None`.

        Position matters, and getting it wrong is the whole hazard of this node class.
        `DeltaEtaStep.step` opens with

            p_fw_blkt_coolant_pump_mw = calculate_p_fw_blkt_coolant_pump_mw(
                ..., p_fw_blkt_coolant_pump_mw)          # <- the PARAMETER

        and every later statement reading that name means the LOCAL. Resolving the
        name to the local everywhere makes it self-referential (caught, but the node
        is then refused for a reason that is not true); resolving it to the parameter
        everywhere silently feeds `.primary_pumping.p_fw_blkt_coolant_pump_mw` into a
        call that wants the freshly computed value -- a plausible, nearly-equal, WRONG
        number, which is exactly the failure this class was refused rather than
        guessed at. Python's own rule -- the right-hand side is evaluated before the
        target is rebound -- is the only correct answer, and it is what this
        implements.
        """
        if name not in self.assigned:
            return None
        entry = self.locals.get(name)  # may raise Unresolved for a non-straight-line body
        if entry is None or entry[2] >= pos:
            return None
        return entry


_AFTER_ALL = float("inf")
"""Statement position for an expression in the `return` -- after every assignment, so
every local is visible to it."""


def _prelude_from_call(call, targets, frame, ctx, label, pos):
    """Turn `call` into a `PreludeCall` producing `len(targets)` fresh locals, and
    append it to `ctx.prelude` (so it is emitted BEFORE anything that reads it).

    The callee must be a `functional_process.models` function whose OWN return arity
    equals `len(targets)` -- the same return-arity invariant the leaf search enforces,
    applied one level down, so a prelude cannot be silently mis-destructured either.
    """
    target = _resolve_name(call.func, frame.globalns, frame.definer)
    if not _is_models_fn(target):
        raise Unresolved(
            f"{label} is produced by `{ast.unparse(call.func)}`, which is not a "
            f"`functional_process.models` function -- refused rather than guessed"
        )
    arity = _return_arity(target)
    if arity != len(targets):
        raise Unresolved(
            f"{label} unpacks {len(targets)} value(s) from {target.__name__}, whose "
            f"own return arity is {arity!r} -- refused rather than guessed"
        )
    order, inputs, statics, locals_ = _bind_call_args(call, target, frame, ctx, pos)
    idents = tuple(ctx.ident(t) for t in targets)
    ctx.prelude.append(
        PreludeCall(
            fn=target.__name__,
            module=target.__module__,
            order=order,
            inputs=inputs,
            statics=statics,
            locals_=locals_,
            targets=idents,
            source=f"{', '.join(targets)} = {ast.unparse(call)}",
        )
    )
    return idents


def _local_bind(entry, name, frame, ctx):
    """The binding for a name the wrapper assigns locally -- materialising its
    producing `PreludeCall` (and, recursively, anything that one reads) on demand.
    `entry` is `frame.local_at(name, pos)`'s answer, so the caller has already
    established that this reference really does see the local and not a shadowed
    parameter."""
    if name in frame._resolved:
        return frame._resolved[name]
    targets, call, index = entry
    if name in frame._in_progress:
        raise Unresolved(f"local {name!r} is defined in terms of itself")
    frame._in_progress.add(name)
    try:
        # `index`, not the reading position: this call's OWN arguments are evaluated
        # before its target is rebound, so a same-named parameter is still in scope
        # for them.
        idents = _prelude_from_call(call, targets, frame, ctx, f"local {name!r}", index)
    finally:
        frame._in_progress.discard(name)
    for t, ident in zip(targets, idents):
        if t != "_":
            frame._resolved[t] = ("local", ident)
    return frame._resolved[name]


def _bind_expr(expr, frame, ctx, pos):
    """One argument expression -> a `(kind, value)` binding, or `Unresolved` with a
    reason. `pos` is the index of the statement the expression sits in: a `Name` the
    frame assigns EARLIER than `pos` is that local (Python's shadowing rule); the same
    name at or before its own assignment is still the parameter. See
    `_Frame.local_at`."""
    if isinstance(expr, ast.Name):
        entry = frame.local_at(expr.id, pos)
        if entry is not None:
            return _local_bind(entry, expr.id, frame, ctx)
        if expr.id in frame.binds:
            return frame.binds[expr.id]
        if expr.id in frame.deferred:
            outer_expr, outer_frame = frame.deferred[expr.id]
            try:
                bound = _bind_expr(outer_expr, outer_frame, ctx, _AFTER_ALL)
            except Unresolved as exc:
                raise Unresolved(
                    f"comes from `{ast.unparse(outer_expr)}` at the helper's call "
                    f"site, which {exc}"
                ) from exc
            frame.binds[expr.id] = bound
            return bound
    elif isinstance(expr, ast.Call):
        # An inline call in argument position (`self._production(f(), g(a, b), ...)`)
        # -- the same composition, spelled without a name. Materialised as an
        # anonymous prelude local rather than refused.
        idents = _prelude_from_call(
            expr, ("tmp",), frame, ctx, f"inline call `{ast.unparse(expr)}`", pos
        )
        return ("local", idents[0])
    static = _static_value(expr, frame.globalns, frame.definer, frame.self_instance)
    if static is not None:
        return ("static", static)
    if ctx.allow_sequence_static:
        # A frozen SEQUENCE static (`self.coefficients`): reported as a `statics`
        # entry carrying a `tuple`, which has no call-site rendering at all (Warp has
        # no tuple type) -- the emitter monomorphises the leaf around it instead. Off
        # by default so `resolve()` gets first refusal at the richer answer: expanding
        # the node's own body, which renders the sequence as N named scalars AND can
        # follow it through a `jax.vmap`. Only if that refuses is the node
        # monomorphised around the literal instead. See `resolve()`.
        seq = _static_sequence_value(
            expr, frame.globalns, frame.definer, frame.self_instance
        )
        if seq is not None:
            return ("static", seq)
    elif _is_static_sequence(
        expr,
        frame.globalns,
        frame.self_instance if frame.self_instance is not None else frame.definer,
    ):
        # Static, but not *placeable* as one scalar. Distinguished from a genuine
        # refusal so `resolve()` can try the body expansion; `_StaticSequence` is an
        # `Unresolved` subclass, so every other caller still treats it as a refusal.
        raise _StaticSequence(
            "is a frozen numeric sequence -- static, but not placeable as one scalar"
        )
    raise Unresolved(
        "is neither a traced parameter, a local this resolver can translate, nor a "
        "static scalar/enum literal"
        + _refusal_detail(expr, frame.globalns, frame.definer, frame.self_instance)
    )


class _StaticSequence(Unresolved):
    """`_bind_args` met a frozen numeric sequence where it can only place a scalar.

    A subclass of `Unresolved` so every existing caller still treats it as a refusal;
    `resolve` is the one place that catches it specially, to try building the node's
    body instead."""


def _is_static_sequence(node, globalns, receiver):
    """`node` names a frozen tuple/list of plain numbers (`self.imp_indices`, a
    module-level table) -- `True`/`False`. `numpy`/`jax` arrays count: a value stored
    in an `eqx.field(static=True)` is constant by declaration, and the storage type it
    happens to use does not change that. What must NOT count is a live traced array,
    and one never reaches here -- a traced value is a parameter, matched against
    `param_to_var` before this is asked."""
    import numpy as _np
    val = _resolve_name(node, globalns, receiver)
    if val is None or isinstance(val, (str, bytes, dict)):
        return False
    if isinstance(val, (tuple, list)):
        return bool(val) and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in val)
    if isinstance(val, _np.ndarray) or hasattr(val, "__array__"):
        try:
            arr = _np.asarray(val)
        except Exception:
            return False
        return arr.ndim >= 1 and arr.size > 0 and arr.dtype.kind in "fiu"
    return False


def _static_sequence_value(node, globalns, definer, self_instance=None):
    """`node` as a frozen SEQUENCE of scalars (a `tuple`/`list` every element of which
    is an `int`/`float`/`bool`/`IntEnum`), returned as a plain `tuple` of
    `float`/`int`/`bool`, or `None` if it is not one.

    Why this is a legitimate read, and where the argument stops
    -----------------------------------------------------------
    `node_def.fn` is a **bound** wrapper: `self` is a concrete object that already
    exists when the graph is assembled, so `self.<attr>` has an actual value, not a
    guessed one. Every attribute this reaches on this port is constant by
    construction, and that was checked rather than assumed:

    - `PeakBTfInboardWithRipple16Coils.coefficients` (and the 18/20-coil siblings) is a
      **class** attribute, `_RIPPLE_FIT_COEFFICIENTS[16]`, a module-level constant.
    - `ImpurityRadiationTotals.imp_indices`, `TfCoilQuenchHeatCurrentDensity.
      den_helium_at_nodes`/`.cp_helium_at_nodes` are `eqx.field(static=True)` on frozen
      equinox `Module`s -- part of the pytree TREEDEF, not its leaves, so no JAX
      transformation and no residual evaluation can change them; a different value
      would be a different graph.

    A sequence is not renderable as a single Warp argument (Warp has no tuple type), so
    unlike a scalar static it is NOT passed at the call site -- the emitter
    monomorphises the leaf, substituting the literal values into its body and dropping
    the parameter (`leaf_funcs._SubstituteSequenceStatics`). A sequence that is empty,
    ragged, or holds anything but scalars still returns `None` -- refuse, do not guess.

    Note this deliberately does NOT accept a `jnp`/`numpy` array: a traced or
    device-resident array is a value that may legitimately differ per evaluation, and
    conflating the two is exactly the silent-staleness failure this function's
    constancy argument is built to avoid.
    """
    val = _resolve_name(node, globalns, self_instance if self_instance is not None else definer)
    if not isinstance(val, (tuple, list)) or not val:
        return None
    out = []
    for e in val:
        if isinstance(e, bool):
            out.append(e)
        elif isinstance(e, enum.Enum):
            try:
                out.append(int(e))
            except (TypeError, ValueError):
                return None
        elif isinstance(e, (int, float)):
            out.append(e)
        else:
            return None
    return tuple(out)


def _refusal_detail(node, globalns, definer, self_instance):
    """A short ", it is a <type>" clause naming what the argument actually resolved to,
    for the refusal message -- so an unbindable `self.<attr>` says WHICH kind of value
    it is (a dataclass, a callable, an array) instead of only that it is not a scalar.
    """
    try:
        val = _resolve_name(
            node, globalns, self_instance if self_instance is not None else definer
        )
    except Exception:  # a property that raises -- still just an unbindable argument
        return ""
    if val is None:
        return ""
    t = type(val)
    kind = f"{t.__module__}.{t.__qualname__}"
    if callable(val) and (inspect.isfunction(val) or inspect.isbuiltin(val)):
        kind += " (a callable -- a function-valued parameter)"
    elif hasattr(val, "shape"):
        kind += f" (an array, shape {getattr(val, 'shape', None)})"
    elif isinstance(val, (tuple, list)):
        kind += f" (a sequence of length {len(val)} holding non-scalars)"
    return f"; it resolves to {kind}"


def _bind_call_args(call, leaf_fn, frame, ctx, pos=_AFTER_ALL):
    """`(order, inputs, statics, locals_)` -- see the module docstring's "Ordering
    contract", plus `locals_` (`(parameter name, local identifier)`) for an argument
    bound to a `PreludeCall`'s output.

    Reads `call`'s arguments (positional and/or keyword) against `leaf_fn`'s own
    signature so the result is in the LEAF's parameter order regardless of how the
    wrapper happened to call it.
    """
    try:
        sig_params = list(inspect.signature(leaf_fn).parameters)
    except (TypeError, ValueError):
        raise Unresolved(f"could not read {leaf_fn!r}'s signature to order its arguments")
    by_keyword = {kw.arg: kw.value for kw in call.keywords if kw.arg is not None}
    if call.keywords and any(kw.arg is None for kw in call.keywords):
        raise Unresolved("leaf called with **kwargs -- cannot place them positionally")
    exprs = []
    for i, p in enumerate(sig_params):
        if i < len(call.args):
            exprs.append((p, call.args[i]))
        elif p in by_keyword:
            exprs.append((p, by_keyword[p]))
        else:
            raise Unresolved(f"leaf parameter {p!r} not found in the call's arguments")
    order, inputs, statics, locals_ = [], [], [], []
    for name, a in exprs:
        order.append(name)
        try:
            kind, value = _bind_expr(a, frame, ctx, pos)
        except Unresolved as exc:
            # `type(exc)`, not `Unresolved`: a `_StaticSequence` must stay one all the
            # way out to `resolve()`, which is the only caller that treats it
            # differently from an ordinary refusal.
            raise type(exc)(
                f"argument {ast.unparse(a)!r} (parameter {name!r}) {exc}"
            ) from exc
        if kind == "var":
            inputs.append(value)
        elif kind == "static":
            statics.append((name, value))
        elif kind == "local":
            locals_.append((name, value))
        else:  # pragma: no cover -- _bind_expr returns only these three kinds
            raise Unresolved(f"unknown binding kind {kind!r} for parameter {name!r}")
    return tuple(order), tuple(inputs), tuple(statics), tuple(locals_)


def _find_returned_local(fdef, frame, ctx, expected_arity):
    """The node's output(s) ARE local(s) the wrapper computes, named by a bare
    `return` -- `(leaf_fn, call, output_index)`, or `None` if the body is not that
    shape.

    `DeltaEtaStep.step` is the case:

        p_fw_blkt_coolant_pump_mw = calculate_p_fw_blkt_coolant_pump_mw(...)
        p_fw_blkt_heat_deposited_mw = calculate_p_fw_blkt_heat_deposited_mw(..)
        p_shld_heat_deposited_mw = calculate_p_shld_heat_deposited_mw(...)
        p_div_heat_deposited_mw = calculate_p_div_heat_deposited_mw(...)
        _, _, _, _, delta_eta_next = calculate_delta_eta(...)
        return delta_eta_next

    `_find_matching_call`'s arity filter drops `calculate_delta_eta` (5 returns
    against 1 declared output) and then reports the four surviving single-value
    helpers as an ambiguity -- so the `Composition` message there named four calls
    precisely BECAUSE it had discarded the right one. The node is not ambiguous; the
    producing call is the one the `return`'s name comes from, and which of its five
    values this node owns is DERIVED from that name's position in the unpack target
    list, exactly as `_subscript_select_index` derives an index from a literal
    subscript. Position read from the wrapper's own AST is a fact about the code; any
    other way of choosing among five returns would be a guess.

    Refuses (rather than guesses) when the returned names do not all come from one
    call, when the producing call is not a `functional_process.models` function, or
    when its return arity disagrees with the unpack it is destructured into.
    """
    returns = _direct_returns(fdef)
    if len(returns) != 1:
        return None
    value = returns[0].value
    if isinstance(value, ast.Name):
        names = [value.id]
    elif isinstance(value, ast.Tuple) and all(
        isinstance(e, ast.Name) for e in value.elts
    ):
        names = [e.id for e in value.elts]
    else:
        return None
    if len(names) != expected_arity:
        return None
    entries = [frame.local_at(n, _AFTER_ALL) for n in names]
    if any(e is None for e in entries):
        return None
    if len({id(e[1]) for e in entries}) != 1:
        raise Unresolved(
            f"the returned local(s) {names} come from {len({id(e[1]) for e in entries})} "
            f"different calls -- this node's outputs are not one call's return; "
            f"refused rather than guessed"
        )
    targets, call, index = entries[0]
    target = _resolve_name(call.func, frame.globalns, frame.definer)
    if not _is_models_fn(target):
        return None
    arity = _return_arity(target)
    if arity != len(targets):
        raise Unresolved(
            f"the returned local(s) {names} are unpacked into {len(targets)} name(s) "
            f"from {target.__name__}, whose own return arity is {arity!r} -- refused "
            f"rather than guessed"
        )
    output_index = tuple(targets.index(n) for n in names)
    # Guard the producing call against binding one of its own targets (a name that
    # would be self-referential); `local_at` already refuses it by position, this is
    # belt.
    frame._in_progress.update(t for t in targets if t != "_")
    try:
        order, inputs, statics, locals_ = _bind_call_args(call, target, frame, ctx, index)
    finally:
        frame._in_progress.difference_update(t for t in targets if t != "_")
    return target, output_index, (order, inputs, statics, locals_)


def _subscript_select_index(wrapper_type):
    """If every `return` in `wrapper_type.__call__`'s own body is exactly
    `self.fn(**kwargs)[K]` (a literal-integer subscript of a call to the `.fn` field),
    return that literal `K`. `None` for anything else -- a plain `return self.fn(...)`
    with no subscript, an inconsistent index across branches, or a subscript of
    anything other than `self.fn(...)`.

    This is how `_NormalisedResidual` (`functional_process/cottax/sand.py`) is found
    to always select index 1: not asserted from reading it once, but derived here from
    the wrapper class's own AST every time this resolver runs, so a future edit to
    `sand.py` that changed the selected index (or removed the subscript entirely)
    would change this function's answer too, not silently go stale.
    """
    call_method = getattr(wrapper_type, "__call__", None)
    if call_method is None:
        return None
    try:
        src = textwrap.dedent(inspect.getsource(call_method))
        tree = ast.parse(src)
        fdef = tree.body[0]
    except (OSError, TypeError, SyntaxError):
        return None
    if not isinstance(fdef, ast.FunctionDef):
        return None
    returns = _direct_returns(fdef)
    if not returns:
        return None
    idx = None
    for r in returns:
        v = r.value
        is_fn_call_subscript = (
            isinstance(v, ast.Subscript)
            and isinstance(v.value, ast.Call)
            and isinstance(v.value.func, ast.Attribute)
            and isinstance(v.value.func.value, ast.Name)
            and v.value.func.value.id == "self"
            and v.value.func.attr == "fn"
            and isinstance(v.slice, ast.Constant)
            and isinstance(v.slice.value, int)
            and not isinstance(v.slice.value, bool)
        )
        if not is_fn_call_subscript:
            return None
        found = v.slice.value
        if idx is not None and idx != found:
            return None
        idx = found
    return idx


def resolve(node_def):
    """`node_def` (an `ImplementedFunction`) -> `(leaf_fn, order, inputs, statics,
    output_index)`.

    `output_index` is `None` for the ordinary case (the leaf's own return arity
    already equals the node's declared output count, in order) or a tuple of ints --
    one per declared output, in order -- naming which position(s) of the leaf's WIDER
    return tuple this node actually owns. The only source of a non-`None` value today
    is `_subscript_select_index`: a `.fn`-field wrapper (`_NormalisedResidual`/
    `_Metric`) whose own `__call__` provably selects one literal index out of the
    leaf's return tuple (`_audit`'s constraint-arity question -- see
    `_audit/optimise_design.md` for the investigation this settled).

    See the module docstring's "Ordering contract" for how a caller reconstructs the
    real call from `order`/`inputs`/`statics`. Raises `Structural` for a `cottax.
    rewrites` comparison/sign node and `Unresolved` (with a reason) for everything else
    this walk does not reach.
    """
    fn = node_def.fn
    type_name = type(fn).__name__
    fn_module = type(fn).__module__
    if type_name in _STRUCTURAL_NAMES or any(
        fn_module.startswith(m) for m in _STRUCTURAL_MODULES
    ):
        raise Structural(f"{type_name} ({fn_module}) -- a comparison/sign node, no leaf")
    if inspect.isbuiltin(fn) or inspect.isbuiltin(getattr(fn, "__func__", None)):
        raise Structural(f"builtin {fn!r} -- a comparison/sign node, no leaf")

    expected_arity = len(node_def.outputs)

    # `_NormalisedResidual`/`_Metric` (sand.py): the leaf sits in a `.fn` field, called
    # via `self.fn(**kwargs)` rather than named in an `ast.Call` at all.
    inner_fn = getattr(fn, "fn", None)
    if inner_fn is not None and (inspect.isfunction(inner_fn) or inspect.ismethod(inner_fn)):
        names = tuple(getattr(fn, "names", ()))
        switches = dict(getattr(fn, "switches", ()) or ())
        if len(names) != len(node_def.inputs):
            raise Unresolved(
                f"{type_name}.names has {len(names)} entries but the node declares "
                f"{len(node_def.inputs)} input(s)"
            )
        # Same invariant, applied to a `.fn`-field leaf: there is only one candidate
        # here (no search to mis-select from), but the arity check still catches a
        # `.fn` pointed at something whose return shape does not match what the node
        # declares -- belt, not guess.
        arity = _return_arity(inner_fn)
        output_index = None
        if arity is not None and arity != expected_arity:
            # Not necessarily a mismatch: a `.fn`-field wrapper may deliberately keep
            # only ONE element of a wider return (`_NormalisedResidual`, `sand.py`).
            # `_subscript_select_index` reads the WRAPPER CLASS's own `__call__` --
            # not this node's `inner_fn` -- to see whether it provably does exactly
            # `self.fn(**kwargs)[K]`. Only then is the mismatch resolved, and only to
            # the K the wrapper's own code names; anything else stays a refusal.
            select = _subscript_select_index(type(fn))
            if (
                expected_arity == 1
                and select is not None
                and 0 <= select < arity
            ):
                output_index = (select,)
            else:
                raise Unresolved(
                    f"{inner_fn.__name__} returns {arity} value(s) but the node "
                    f"declares {expected_arity} output(s), and {type_name}.__call__ "
                    f"does not provably select a single element of that return "
                    f"(found index {select!r})"
                )
        param_to_var = dict(zip(names, [i.var for i in node_def.inputs]))
        try:
            sig_params = list(inspect.signature(inner_fn).parameters)
        except (TypeError, ValueError):
            raise Unresolved(f"could not read {inner_fn!r}'s signature")
        order, inputs, statics = [], [], []
        for p in sig_params:
            order.append(p)
            if p in switches:
                val = switches[p]
                if isinstance(val, enum.Enum):
                    val = int(val)
                if not isinstance(val, (int, float, bool)):
                    raise Unresolved(
                        f"{inner_fn.__name__}'s switch {p!r} = {val!r} is not a scalar "
                        f"-- no literal rendering"
                    )
                statics.append((p, val))
            elif p in param_to_var:
                inputs.append(param_to_var[p])
            else:
                raise Unresolved(
                    f"{inner_fn.__name__}'s parameter {p!r} is in neither "
                    f"{type_name}.names nor .switches"
                )
        return inner_fn, tuple(order), tuple(inputs), tuple(statics), output_index, (), ()

    fdef, params, globalns, definer, wrapper_module, self_instance = _source_and_params(fn)
    if len(params) != len(node_def.inputs):
        raise Unresolved(
            f"signature has {len(params)} param(s) but node declares "
            f"{len(node_def.inputs)} input(s)"
        )
    param_binds = {
        p: ("var", v) for p, v in zip(params, [i.var for i in node_def.inputs])
    }
    def _attempt(allow_sequence_static):
        """One resolution pass over the wrapper. `allow_sequence_static` is the
        second pass's concession: bind a frozen SEQUENCE argument as a
        monomorphisable `statics` entry instead of asking to build the body.
        """
        ctx = _Ctx(allow_sequence_static)
        outer = _Frame(fdef, globalns, definer, self_instance, param_binds)

        # A `Composition` from the arity-filtered search is HELD, not raised yet: the
        # filter can have discarded the real producing call for having a wider return than
        # the node's output count, leaving several innocent single-value helpers looking
        # like an ambiguity (`DeltaEtaStep.step` -- see `_find_returned_local`). The
        # returned-local path below is tried first; the held exception is re-raised only
        # if that finds nothing either.
        held_composition = None
        try:
            direct = _find_matching_call(fdef, globalns, definer, expected_arity)
        except Composition as exc:
            held_composition, direct = exc, None
        if direct is not None:
            leaf_fn, call, frame = direct[0], direct[1], outer
        else:
            if _find_value_passed(fdef, globalns, definer) is not None:
                # The leaf is handed to one of the wrapper's own helpers as a VALUE,
                # never called at this level, and that helper assembles a
                # fixed-length species array before forwarding -- so no existing
                # function's parameter list can be laid against the node's VarPaths.
                # Build the leaf instead of chasing it.
                return _resolve_built_composition(
                    node_def, _unwrap_bound(fn), params, expected_arity
                )
            found = _find_via_helper(fdef, outer, ctx, expected_arity)
            if found is None:
                # Last resort #1: a same-module dispatch helper as the leaf itself (see the
                # module docstring's "same-module dispatch helper" case) -- still under
                # the same arity restriction.
                try:
                    direct2 = _find_matching_call(
                        fdef, globalns, definer, expected_arity, extra_module=wrapper_module
                    )
                except Composition as exc:
                    held_composition = held_composition or exc
                    direct2 = None
                if direct2 is not None:
                    found = (direct2[0], direct2[1], outer)
            if found is None:
                # Last resort #2: the node's outputs are local(s) the body computes and
                # the `return` merely names -- the producing call is then whatever those
                # names are unpacked from, at the position they occupy in that unpack.
                returned = _find_returned_local(fdef, outer, ctx, expected_arity)
                if returned is not None:
                    leaf_fn, output_index, (order, inputs, statics, locals_) = returned
                    used = {ident for _n, ident in locals_}
                    for pc in ctx.prelude:
                        used.update(ident for _n, ident in pc.locals_)
                    prelude = tuple(
                        pc for pc in ctx.prelude if used.intersection(pc.targets)
                    )
                    return leaf_fn, order, inputs, statics, output_index, locals_, prelude
                if held_composition is not None:
                    raise held_composition
                raise Unresolved(
                    f"no call with return arity {expected_arity} (the node's declared "
                    f"output count) found within one helper level"
                )
            leaf_fn, call, frame = found
        order, inputs, statics, locals_ = _bind_call_args(call, leaf_fn, frame, ctx)
        used = {ident for _n, ident in locals_}
        for pc in ctx.prelude:
            used.update(ident for _n, ident in pc.locals_)
        prelude = tuple(pc for pc in ctx.prelude if used.intersection(pc.targets))
        return leaf_fn, order, inputs, statics, None, locals_, prelude

    # Pass 1 refuses a frozen SEQUENCE argument as `_StaticSequence` rather than
    # binding it, so the richer answer gets first refusal: `_resolve_built_composition`
    # expands the node's own body, rendering the sequence as N named scalars and
    # following it through a `jax.vmap`. Only if that expansion refuses does pass 2
    # bind the sequence as a monomorphisable static (the emitter substitutes its
    # literals into the leaf body and drops the parameter). Both are exact; the
    # expansion simply reaches strictly more shapes, so it is tried first.
    try:
        return _attempt(False)
    except _StaticSequence as exc:
        try:
            return _resolve_built_composition(
                node_def, _unwrap_bound(fn), params, expected_arity
            )
        except Unresolved as inner_exc:
            try:
                return _attempt(True)
            except Unresolved as seq_exc:
                raise Unresolved(
                    f"{exc}; building the node's body instead refuses "
                    f"({inner_exc}); and monomorphising it around the sequence "
                    f"refuses ({seq_exc})"
                ) from inner_exc
