"""Census, with `thin` meaning what the split actually needs.

`census.py`'s `thin` accepts any single `return f(...)`. That is a proxy: an argument
list may still carry arithmetic (`f(1e-20 * n, ...)`), which the deferred `fn = <func>`
interface could not express and which is exactly the containment the split removes. This
tightens the test to: one `return`, of one call, every argument a bare parameter name.
"""
import ast, pathlib, collections

# The package this file lives in, NOT an absolute path: a hardcoded root measures
# whichever checkout it names, so running the script from a worktree silently reported
# the main tree's numbers instead (found 2026-09-05, after it had been handed to four
# agents as their gate).
root = pathlib.Path(__file__).resolve().parent.parent
kinds = collections.Counter()
argy = []


def classify(fn, params):
    body = [s for s in fn.body
            if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    if len(body) != 1:
        return f"multi-statement ({len(body)})", None
    s = body[0]
    if not isinstance(s, ast.Return) or s.value is None:
        return "no return", None
    v = s.value
    if isinstance(v, ast.Call) and isinstance(v.func, (ast.Name, ast.Attribute)):
        def ok(node):
            # A bare parameter, or one of the declaration's own fields -- `self.switch`
            # is a legitimate input (a pytree-visible field), not computation.
            if isinstance(node, ast.Name) and node.id in params:
                return True
            return (isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "self")

        bad, fields = [], 0
        for a in v.args:
            if isinstance(a, ast.Starred):
                bad.append("*args")
            elif not ok(a):
                bad.append(ast.unparse(a))
            elif isinstance(a, ast.Attribute):
                fields += 1
        for kw in v.keywords:
            if kw.arg is None:
                bad.append("**kwargs")
            elif not ok(kw.value):
                bad.append(f"{kw.arg}={ast.unparse(kw.value)}")
            elif isinstance(kw.value, ast.Attribute):
                fields += 1
        if bad:
            return "thin-but-computed-args", bad
        return ("thin" if not fields else "thin-plus-own-fields"), None
    if isinstance(v, ast.Tuple):
        return "tuple", None
    if isinstance(v, (ast.Name, ast.Attribute)):
        return "bare-name", None
    return f"expr:{type(v).__name__}", None


for p in sorted(root.rglob("*.py")):
    if "_audit" in p.parts:
        continue
    try:
        tree = ast.parse(p.read_text())
    except SyntaxError:
        continue
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for fn in [n for n in cls.body
                   if isinstance(n, ast.FunctionDef) and n.name == "__call__"]:
            a = fn.args
            defaults = a.defaults + [d for d in a.kw_defaults if d]
            if not any(isinstance(d, ast.Call) and isinstance(d.func, ast.Name)
                       and d.func.id in {"From", "Supply", "Start"} for d in defaults):
                continue
            params = {x.arg for x in a.args + a.kwonlyargs}
            k, bad = classify(fn, params)
            kinds[k] += 1
            if k == "thin-but-computed-args":
                argy.append((f"{p.relative_to(root)}:{fn.lineno}", cls.name, bad))

print("=== declarations with From()-style ports:", sum(kinds.values()))
for k, n in kinds.most_common():
    print(f"  {n:4d}  {k}")
print(f"\n=== thin body but computed arguments: {len(argy)}")
for path, name, bad in argy:
    print(f"  {name:44s} {path}")
    print(f"      {'; '.join(bad[:4])}")
