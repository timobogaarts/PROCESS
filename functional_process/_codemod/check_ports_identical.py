"""Part A one-shot tooling -- the per-file inertness proof. Deletable when Part A closes.

`_audit/path_refactor.md` §A.4. Loads a module twice -- the pre-conversion source from
`git show HEAD:<path>` and the working tree's -- as two separate modules, and for every
`NodalDeclaration` subclass defined in both asserts that the `In`/`Out` `VarPath` tuples
are **identical and in the same order**.

Why this and not `pytest`: the suite cannot see two same-typed reads swapped inside one
node. The values still flow, the graph still assembles, the numbers still land, and
2078 conversion sites is far past the point where reading the diff catches it. Order
matters because a read is bound to a parameter by *position* in the tuple, so a
same-multiset-different-order tuple is a real defect that compares equal as a set.

The §A.5 renames are not special-cased: renaming a parameter is a local change and the
`VarPath` it resolves to must still be identical. A rename that moves a port is a bug,
and this is what says so.

Ports are read off the **class**, not an instance -- 28 registrations carry constructor
configuration and several bases are abstract, and none of that touches the port tuples,
which come from `_declared_outputs_on_cls` and the signature of `__call__`/`residual`/
`step`.

Usage::

    PY=~/miniconda3/envs/process_port/bin/python
    $PY -m functional_process._codemod.check_ports_identical \
        functional_process/models/vacuum.py

Exits nonzero on any difference, on any class that appears in only one of the two
versions, and on any file that fails to load. Run from the repository root.
"""

# ruff: noqa: S404, S603, S607, PLC2701  -- shells out to `git`, and reads cottax
# internals on purpose: `_declared_outputs_on_cls` is where the write ports live.
from __future__ import annotations

import argparse
import importlib.util
import inspect
import subprocess
import sys
import types  # noqa: TC003 -- a runtime return annotation, `from __future__` or not
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)  # before anything makes an array

from cottax.interfaces.pytree_namespace_module import (  # noqa: E402
    NodalDeclaration,
    _declared_outputs_on_cls,
)


def _load(source: str, module_name: str, origin: Path) -> types.ModuleType:
    """Execute `source` as a fresh module named `module_name`.

    The port bans relative imports (`ruff`'s `flake8-tidy-imports`), so a module needs no
    package context; `__package__ = ""` keeps any stray relative import failing loudly
    rather than resolving to the wrong tree. The name is unique per load so the two
    copies never share a `sys.modules` entry -- but they *do* share every module they
    import, which is what makes the comparison about this file and nothing else.
    """
    spec = importlib.util.spec_from_loader(module_name, loader=None, origin=str(origin))
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(origin)
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        exec(compile(source, str(origin), "exec"), module.__dict__)  # noqa: S102
    except Exception:
        del sys.modules[module_name]
        raise
    return module


def _git_show(rev: str, path: Path) -> str:
    return subprocess.run(
        ["git", "show", f"{rev}:{path.as_posix()}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _ports(cls: type) -> tuple[list[str], list[str]]:
    """`(reads, writes)` as rendered `VarPath`s, in declaration order."""
    writes = [
        f"{o.port().var.path_str()}"
        + ("  [static]" if o.port().static else "")
        + (f"  {o.port().tags}" if o.port().tags else "")
        for o in _declared_outputs_on_cls(cls)
    ]
    signature_of = getattr(cls, "_signature_of", None)
    reads: list[str] = []
    if signature_of is not None:
        params = list(inspect.signature(getattr(cls, signature_of)).parameters.values())
        if params and params[0].name in {"self", "cls"}:
            params = params[1:]
        for p in params:
            default = p.default
            if default is inspect.Parameter.empty or not hasattr(default, "port"):
                reads.append(f"{p.name}: <no read declared>")
            else:
                reads.append(default.port(p.name).var.path_str())
    return reads, writes


def _declarations(module: types.ModuleType, module_name: str) -> dict[str, type]:
    return {
        name: obj
        for name, obj in vars(module).items()
        if isinstance(obj, type)
        and issubclass(obj, NodalDeclaration)
        and obj.__module__ == module_name
    }


def check(path: Path, rev: str = "HEAD") -> tuple[bool, int]:
    """Compare one file's declarations before and after. `(ok, total port count)`."""
    stem = path.stem.replace(".", "_")
    before = _load(_git_show(rev, path), f"_partA_before_{stem}", path)
    after = _load(path.read_text(), f"_partA_after_{stem}", path)

    names_before = _declarations(before, f"_partA_before_{stem}")
    names_after = _declarations(after, f"_partA_after_{stem}")

    ok = True
    total = 0
    only_before = sorted(set(names_before) - set(names_after))
    only_after = sorted(set(names_after) - set(names_before))
    for name in only_before:
        print(f"  !! {name}: present at {rev}, gone from the working tree")
        ok = False
    for name in only_after:
        print(f"  !! {name}: new in the working tree, absent at {rev}")
        ok = False

    for name in sorted(set(names_before) & set(names_after)):
        reads_b, writes_b = _ports(names_before[name])
        reads_a, writes_a = _ports(names_after[name])
        total += len(reads_a) + len(writes_a)
        if reads_b == reads_a and writes_b == writes_a:
            print(f"  {name}: {len(reads_a)} in, {len(writes_a)} out -- identical")
            continue
        ok = False
        print(f"  !! {name}: PORTS DIFFER")
        for label, b, a in (("reads", reads_b, reads_a), ("writes", writes_b, writes_a)):
            if b == a:
                continue
            print(f"     {label}:")
            for i in range(max(len(b), len(a))):
                x = b[i] if i < len(b) else "<missing>"
                y = a[i] if i < len(a) else "<missing>"
                flag = "  " if x == y else "<-"
                print(f"       [{i:>3}] {flag} {x}   ->   {y}")
    return ok, total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", help="repo-relative module paths")
    parser.add_argument("--rev", default="HEAD", help="the pre-conversion revision")
    args = parser.parse_args(argv)

    status = 0
    for target in args.paths:
        path = Path(target)
        print(f"{path}:")
        try:
            ok, total = check(path, args.rev)
        except Exception as exc:  # noqa: BLE001 -- a load failure is a check failure
            print(f"  !! failed to load: {type(exc).__name__}: {exc}")
            status = 1
            continue
        print(f"  => {total} ports, " + ("all identical" if ok else "DIFFERENCES ABOVE"))
        if not ok:
            status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
