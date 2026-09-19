"""Every cottax name the port imports, in one place, asserted importable.

The port tracks a cottax that moves (`../../CLAUDE.md` keeps the drift tables), and a
rename upstream surfaces here as an `ImportError` from whichever module happened to be
imported first -- one name, no list, and the next one only after the first is fixed.
This test is the list: it reads the port's own imports with `ast` (so it cannot fall
behind them) and asks cottax for every name at once, so a re-port begins with the whole
diff rather than one line of it.

`PRIVATE` is the second half: three names the port reaches for under an underscore. Each
is a deliberate reach into cottax's inside and a candidate for an upstream export; a new
one is a decision, so it fails here until it is written down.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

PORT = Path(__file__).resolve().parent.parent
"""`functional_process/` -- every `.py` under it is read."""

PRIVATE: frozenset[str] = frozenset({
    "cottax.evaluation.schedule._run_acyclic",
    "cottax.visualization.sequencing._draws_feedback",
    "cottax.visualization.xdsm._xesc",
})
"""The private cottax names the port uses, and nothing else may be added without
saying so here: `_run_acyclic` runs a nested acyclic body inside SAND's one combined
problem, `_draws_feedback` and `_xesc` are what `visualization/grouping.py` needs to
group a DSM the way cottax draws one."""


def _imports() -> dict[str, set[str]]:
    """`{module: {name, ...}}` -- every `from cottax... import ...` in the port."""
    surface: dict[str, set[str]] = {}
    for path in sorted(PORT.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.split(".")[0] == "cottax":
                    surface.setdefault(module, set()).update(a.name for a in node.names)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] == "cottax":
                        surface.setdefault(alias.name, set())
    return surface


def test_every_imported_cottax_name_is_there():
    """Every name the port imports from cottax resolves, and the failures come as one
    list rather than one `ImportError` at a time.
    """
    missing = []
    for module, names in sorted(_imports().items()):
        try:
            imported = importlib.import_module(module)
        except ImportError as error:  # the module itself moved
            missing.append(f"{module}: {error}")
            continue
        missing += [
            f"{module}.{name}" for name in sorted(names) if not hasattr(imported, name)
        ]
    assert not missing, (
        f"{len(missing)} cottax name(s) the port imports are gone -- re-port against "
        f"the cottax in the env and update ../../CLAUDE.md's drift table:\n  "
        + "\n  ".join(missing)
    )


def test_the_private_reaches_into_cottax_are_the_declared_ones():
    """The port uses exactly `PRIVATE` of cottax's private names."""
    used = {
        f"{module}.{name}"
        for module, names in _imports().items()
        for name in names
        if name.startswith("_") or any(p.startswith("_") for p in module.split(".")[1:])
    }
    assert used == set(PRIVATE), (
        f"new private cottax name(s) {sorted(used - set(PRIVATE))}, gone "
        f"{sorted(set(PRIVATE) - used)} -- each is a reach into cottax's inside and an "
        f"upstream export to ask for; add it to `PRIVATE` with what it is for"
    )
