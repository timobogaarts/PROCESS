"""The port imports cottax through its doors only: a front door, a consumer's
`interface` / `interface_types` / `refusals`, or a feature's answer module by name.
The internals still imported are listed, to be removed; a new one fails."""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DOOR = re.compile(
    r"cottax\.(relational_pytree|core|core\.(driven|realisation|frozen|refusal_kinds)|relational|pytree"
    r"|consumers\.(execution|json|openmdao|declaration|visualization|architectures|pickle"
    r"|visualization\.latex)"
    r"|consumers\.(execution|json|openmdao|declaration|visualization|architectures)"
    r"\.(interface|interface_types|refusals)"
    r"|(relational|pytree)\.consumers\.(execution|json|openmdao|declaration|visualization|architectures)"
    r"|relational\.consumers\.execution\.jax_solvers)"
)

KNOWN: set[tuple[str, str]] = set()
"""Internals no door offers yet: `(file under the repo, module)`. None are left."""


def cottax_imports(path: Path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module
        elif isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)


def test_cottax_is_imported_through_its_doors():
    found = {
        (path.relative_to(ROOT.parent).as_posix(), module)
        for scanned in (ROOT, ROOT.parent / "paper_tests")
        for path in scanned.rglob("*.py")
        for module in cottax_imports(path)
        if module.startswith("cottax.") and not DOOR.fullmatch(module)
    }
    assert found - KNOWN == set()
