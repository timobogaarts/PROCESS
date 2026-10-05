"""The port imports cottax through its doors only: a front door, a consumer's
`interface` / `interface_types` / `refusals`, or a feature's answer module by name.
The internals still imported are listed, to be removed; a new one fails."""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DOOR = re.compile(
    r"cottax\.(cx|core|core\.(driven|realisation|frozen)|relational|pytree"
    r"|consumers\.(execution|json|openmdao|declaration|visualization|pickle"
    r"|visualization\.latex)"
    r"|consumers\.(execution|json|openmdao|declaration|visualization)"
    r"\.(interface|interface_types|refusals)"
    r"|(relational|pytree)\.consumers\.(execution|json|openmdao|declaration|visualization)"
    r"|relational\.consumers\.execution\.jax_solvers)"
)

KNOWN = {
    ("cottax/architectures/contracts.py", "cottax.relational.contracts"),
    ("cottax/study.py", "cottax.relational.contracts"),
    ("cottax/architectures/reading.py", "cottax.relational.consumers.canonical_nlp"),
    ("tests/models/stellarator/coils/test_calculate.py", "cottax.consumers.visualization.kinds"),
    ("tests/models/stellarator/coils/test_coils.py", "cottax.consumers.visualization.kinds"),
}
"""Internals no door offers yet: `(file under functional_process/, module)`."""


def cottax_imports(path: Path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module
        elif isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)


def test_cottax_is_imported_through_its_doors():
    found = {
        (path.relative_to(ROOT).as_posix(), module)
        for path in ROOT.rglob("*.py")
        for module in cottax_imports(path)
        if module.startswith("cottax.") and not DOOR.fullmatch(module)
    }
    assert found - KNOWN == set()
