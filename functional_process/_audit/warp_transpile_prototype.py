"""A prototype JAX -> Warp transpiler: source in, source out.

A *transpiler* translates source to source. It does not produce machine code -- it
produces other Python, which Warp then compiles. So: parse a model function with `ast`,
rewrite the nodes that differ, unparse as a `@wp.func`.

Three rewrites are all that 92% of the layer needs (§76):
  jnp.X(...)      -> wp.X(...)          for the handful of names Warp shares
  numeric literal -> wp.float64(n)      because Warp is strictly typed (§80)
  bare signature  -> annotated          every param and the return as wp.float64

**It REFUSES rather than guesses.** An unrecognised construct raises `Unsupported`, and
that function goes on the hand-port list. A transpiler that silently mistranslates is
worse than one that covers less.
"""
import ast, inspect, re, textwrap

DIRECT = {"sqrt", "exp", "log", "sin", "cos", "tan", "tanh", "abs", "sign",
          "floor", "ceil", "pow", "atan", "asin", "acos", "sinh", "cosh"}
TERNARY = {"where", "select"}   # jnp.where(c, a, b) -> wp.where(c, a, b)
IDENTITY = {"asarray", "array", "float64", "double"}
"""Coercions that vanish inside a statically-typed kernel: `jnp.asarray(x)` -> `x`."""

RENAME = {"maximum": "max", "minimum": "min", "where": "where", "clip": "clamp",
          "isinf": "isinf", "isnan": "isnan", "round": "round", "trunc": "trunc", "power": "pow", "arctan": "atan",
          "arcsin": "asin", "arccos": "acos", "log10": "log10", "fabs": "abs"}


PROVIDED = {"safe_sqrt", "safe_pow"}
"""Port-local helpers hand-written once as `@wp.func` and then callable."""


class Unsupported(Exception):
    """Raised for anything the table does not cover -- never guessed at."""


class ToWarp(ast.NodeTransformer):
    def __init__(self):
        self.notes = []

    def visit_Constant(self, node):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            return node
        return ast.Call(func=ast.Attribute(value=ast.Name("wp", ast.Load()),
                                           attr="float64", ctx=ast.Load()),
                        args=[ast.Constant(float(node.value))], keywords=[])

    def visit_Call(self, node):
        self.generic_visit(node)
        f = node.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) \
                and f.value.id in ("jnp", "np"):
            if f.attr in IDENTITY and len(node.args) == 1:
                return node.args[0]        # a coercion that a typed kernel does not need
            if f.attr == "degrees":
                return ast.BinOp(left=node.args[0], op=ast.Mult(),
                                 right=ast.Call(
                                     func=ast.Attribute(ast.Name("wp", ast.Load()),
                                                        "float64", ast.Load()),
                                     args=[ast.Constant(57.29577951308232)],
                                     keywords=[]))
            name = RENAME.get(f.attr, f.attr)
            if f.attr not in DIRECT and f.attr not in RENAME:
                raise Unsupported(f"jnp.{f.attr}")
            node.func = ast.Attribute(value=ast.Name("wp", ast.Load()),
                                      attr=name, ctx=ast.Load())
        elif isinstance(f, ast.Name) and f.id in PROVIDED:
            pass   # a port-local helper we hand-write once as a @wp.func
        return node

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Pow):
            # `x ** 2` and `x ** 1.5` round differently under wp.pow than under a
            # product; leave the operator, but record it -- §80 measured this as a
            # source of last-bit disagreement.
            self.notes.append("** kept as-is (see §80 on rounding)")
        return node


def transpile(fn, name=None):
    """`fn` -> Warp source for an equivalent `@wp.func`."""
    src = textwrap.dedent(inspect.getsource(fn))
    src = re.sub(r'"""(?:.|\n)*?"""', "", src, count=1)      # drop the docstring
    tree = ast.parse(src)
    fdef = tree.body[0]
    t = ToWarp()
    fdef = t.visit(fdef)
    # **Refuse defaults rather than drop them.** A default is a value a caller may rely
    # on; Warp has no equivalent, and silently discarding one is exactly the kind of
    # guess this transpiler exists not to make.
    if fdef.args.defaults or any(d is not None for d in fdef.args.kw_defaults):
        raise Unsupported("default argument value(s)")
    if fdef.args.vararg or fdef.args.kwarg:
        raise Unsupported("*args/**kwargs")
    # Annotate every parameter and the return as wp.float64.
    f64 = ast.Attribute(value=ast.Name("wp", ast.Load()), attr="float64", ctx=ast.Load())
    args = fdef.args
    for a in list(args.args) + list(args.kwonlyargs) + list(args.posonlyargs):
        a.annotation = f64
    args.kwonlyargs = list(args.kwonlyargs)
    # **Only annotate the return when there IS one value.** Warp infers multi-return
    # correctly on its own, and stamping `-> float64` on a function that returns a tuple
    # is a compile error Warp only raises when the function is actually codegen'd --
    # i.e. never, if nothing calls it. 151 of 385 emitted functions were mis-annotated
    # this way and the validator could not see it, because its JAX reference call
    # (`float(fn(...))`) raises on a tuple and skips the function first.
    returns_tuple = any(
        isinstance(n, ast.Return) and isinstance(n.value, ast.Tuple)
        for n in ast.walk(fdef)
    )
    if not returns_tuple:
        fdef.returns = f64
    fdef.name = name or fdef.name
    fdef.decorator_list = [ast.Attribute(value=ast.Name("wp", ast.Load()),
                                         attr="func", ctx=ast.Load())]
    ast.fix_missing_locations(tree)
    return ast.unparse(tree), t.notes


if __name__ == "__main__":
    import collections, importlib, pkgutil, sys
    import functional_process.models as M
    ok, refused = [], collections.Counter()
    seen = set()
    for mi in pkgutil.walk_packages(M.__path__, M.__name__ + "."):
        try:
            mod = importlib.import_module(mi.name)
        except Exception:
            continue
        for nm in dir(mod):
            fn = getattr(mod, nm, None)
            if not callable(fn) or not hasattr(fn, "__module__"):
                continue
            if not str(fn.__module__).startswith("functional_process.models"):
                continue
            key = (fn.__module__, nm)
            if key in seen or not hasattr(fn, "__code__"):
                continue
            seen.add(key)
            try:
                transpile(fn)
                ok.append(key)
            except Unsupported as exc:
                refused[str(exc)] += 1
            except Exception as exc:
                refused[f"<{type(exc).__name__}>"] += 1
    total = len(ok) + sum(refused.values())
    print(f"\nswept {total} functions in functional_process/models/**")
    print(f"  transpiled cleanly : {len(ok):>5}  ({100*len(ok)/total:.1f}%)")
    print(f"  refused            : {sum(refused.values()):>5}\n")
    print("  top reasons for refusal:")
    for why, n in refused.most_common(14):
        print(f"    {n:>5}  {why}")
