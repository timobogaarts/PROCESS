"""The split's one enforceable sentence: **nothing outside `cottax/` imports cottax.**

`_audit/formulas_split.md` step 5. Without this the separation lasts exactly until the
first person in a hurry writes `from cottax... import From` in a physics file, and
nothing says so -- the code keeps working, because cottax is installed.

What the separation buys, and why it is worth a test rather than a convention:
`functional_process/models/` is PROCESS's physics as pure JAX functions, usable with no
graph machinery at all. `~/openmdao_process` already imports those bodies, so the
boundary has a consumer rather than only an intention.

**Imports are read with `ast`, not `grep`.** Most apparent hits are docstrings naming a
declaration module (`see \\`functional_process.cottax.stellarator.build\\``), and prose is
not a dependency. A grep-based version of this test would fail on ~30 files that are
perfectly clean.
"""

import ast
import pathlib

PACKAGE = pathlib.Path(__file__).resolve().parent.parent
COTTAX_SUBTREE = PACKAGE / "cottax"
TESTS_SUBTREE = PACKAGE / "tests"


def _imported_modules(tree):
    """Every module name this file imports, `import x` and `from x import y` alike."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            # `from . import x` has no module; a relative import cannot reach cottax
            # from outside the package anyway.
            if node.module and node.level == 0:
                yield node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name


def _is_cottax(module: str) -> bool:
    return module == "cottax" or module.startswith("cottax.")


def test_nothing_outside_the_cottax_subtree_imports_cottax():
    """The boundary. A failure names the file and line, so the fix is obvious: either
    the code belongs under `cottax/`, or it should not be reaching for the graph.
    """
    offenders = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if "_audit" in path.parts or "__pycache__" in path.parts:
            continue
        if COTTAX_SUBTREE in path.parents:
            continue
        # **The tests are exempt, and deliberately.** A test for a declaration has to
        # import the thing it declares with; `tests/models/physics/test_*.py` checks
        # the pure formula *and* the node that wraps it, in one file, against the same
        # PROCESS reference. Forbidding cottax here would either split every unit's
        # tests in two or forbid testing the declarations at all, and neither buys
        # anything: what the split promises is that the *shipped* physics needs no
        # graph machinery, not that nobody may test the graph.
        if TESTS_SUBTREE in path.parents or path.parent == TESTS_SUBTREE:
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover -- a broken file is another test's job
            continue
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                mods = [node.module]
            elif isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            for m in mods:
                if _is_cottax(m):
                    offenders.append(
                        f"{path.relative_to(PACKAGE.parent)}:{node.lineno} imports {m}"
                    )
    assert not offenders, (
        "cottax is imported outside `functional_process/cottax/`:\n  "
        + "\n  ".join(offenders)
    )


def test_the_physics_half_does_not_import_the_declarations_half():
    """Stronger than the boundary above, and it is the one that actually broke.

    `models/` may not import `functional_process.cottax` either. The declarations import
    the physics, never the reverse -- otherwise the physics is not usable standalone even
    though it never says `import cottax`, which is exactly the state the tree was in
    mid-split: `models/` reached into `cottax/` for `safe_math`, the pfcoil constants and
    a handful of all-pure modules that had simply not been moved yet.
    """
    offenders = []
    for path in sorted((PACKAGE / "models").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover
            continue
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                mods = [node.module]
            elif isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            for m in mods:
                if m == "functional_process.cottax" or m.startswith(
                    "functional_process.cottax."
                ):
                    offenders.append(
                        f"{path.relative_to(PACKAGE.parent)}:{node.lineno} imports {m}"
                    )
    assert not offenders, (
        "`functional_process/models/` (the physics) imports the declarations half:\n  "
        + "\n  ".join(offenders)
    )
