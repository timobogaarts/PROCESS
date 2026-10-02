"""One module's declarations against the reference, node by node.

Every module-level wrapper (and a tuple, list or dict of them) is built at a name of its
own, and each node it declares is matched to the reference node that owns the same places
(`graphs/*.json`, any configuration). It must read the same places in the same order and,
where the reference body was a physics function, run the same one. A node no reference
configuration holds is listed as unchecked.

    PYTHONPATH=~/PROCESS-port JAX_PLATFORMS=cpu $PY \\
        -m functional_process.tests.port_reference.check_nodes <module> [<module> ...]
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

from cottax.consumers.declaration.interface import Wrapper
from cottax.pytree.path import NodePath
from jax.tree_util import DictKey

from functional_process.tests.port_reference.fingerprint import node

HERE = Path(__file__).parent


def reference() -> dict[str, dict]:
    nodes = {}
    for f in sorted((HERE / "graphs").glob("*.json")):
        for n in json.loads(f.read_text())["nodes"]:
            nodes.setdefault(json.dumps(n["owns"]), n)
    return nodes


def wrappers(module) -> list[tuple[str, Wrapper]]:
    found = []
    for name, value in vars(module).items():
        items = value.items() if isinstance(value, dict) else (
            enumerate(value) if isinstance(value, (tuple, list)) else [(None, value)])
        for key, item in items:
            if isinstance(item, Wrapper):
                found.append((name if key is None else f"{name}[{key!r}]", item))
    return found


def check(module_name: str, ref: dict[str, dict]) -> int:
    try:
        module = importlib.import_module(module_name)
    except Exception as e:  # noqa: BLE001
        print(f"{module_name}: does not import: {type(e).__name__}: {e}")
        return 1
    bad = unchecked = good = 0
    for name, wrapper in wrappers(module):
        for at, definition in wrapper.nodes(NodePath((DictKey(name),))):
            new = node(at, definition)
            old = ref.get(json.dumps(new["owns"]))
            if old is None:
                unchecked += 1
                print(f"  unchecked  {name}: owns {new['owns']}")
                continue
            problems = [f for f in ("reads", "relations", "objectives", "unknowns")
                        if old.get(f) != new.get(f)]
            if old.get("body", "").startswith("functional_process.models") and \
                    old["body"] != new.get("body"):
                problems.append("body")
            if problems:
                bad += 1
                print(f"  DIFFERS    {name}: {problems}")
                for f in problems:
                    print(f"      old {f}: {old.get(f)}\n      new {f}: {new.get(f)}")
            else:
                good += 1
    print(f"{module_name}: {good} match, {bad} differ, {unchecked} unchecked")
    return bad


if __name__ == "__main__":
    ref = reference()
    raise SystemExit(bool(sum(check(m, ref) for m in sys.argv[1:])))
