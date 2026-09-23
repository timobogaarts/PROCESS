"""Every cottax name the port imports, in one place, asserted importable.

The port tracks a cottax that moves (`../../CLAUDE.md` keeps the drift tables), and a
rename upstream surfaces here as an `ImportError` from whichever module happened to be
imported first -- one name, no list, and the next one only after the first is fixed.
This test is the list: it reads the port's own imports with `ast` (so it cannot fall
behind them) and asks cottax for every name at once, so a re-port begins with the whole
diff rather than one line of it. Every `.py` under `functional_process/` and
`paper_tests/`, and every code cell of every notebook under
`architecture_examples/` -- a study written in a notebook is written over the same
surface as one written in a module.

`ALLOWED` is the other half, and the stronger one: the cottax modules a study may be
written over. Two packages -- `cottax.interfaces` (how models, statements and runs are
written) and `cottax.mdao_architectures` (the recipes) -- plus the three things that sit
below them by nature and keep their own names: a **driver** (`execution.driver` and the
kinds), a **name** (`pytree.path`, `pytree.mint`) and a **picture** (`visualization`).
Anything else is a reach past the interface: either it lacks a name (ask for it upstream)
or the port is doing something a study should not.

`PRIVATE` is the last half: the names the port reaches for under an underscore. Each is a
deliberate reach into cottax's inside; a new one is a decision, so it fails here until it
is written down. It is **empty**, and the point is to keep it that way.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import nbformat

PORT = Path(__file__).resolve().parent.parent
"""`functional_process/` -- every `.py` under it, and every notebook code cell."""

PAPER = PORT.parent / "paper_tests"
"""`paper_tests/` -- the thin CLIs over the port, read the same way."""

ALLOWED: tuple[str, ...] = (
    "cottax",
    "cottax.interfaces",
    "cottax.mdao_architectures",
    "cottax.execution.driver",
    "cottax.execution.drivers.kinds",
    "cottax.pytree.path",
    "cottax.pytree.mint",
    "cottax.visualization",
    "cottax.visualization.sequencing",
)
"""The cottax modules the port may import from. `cottax.interfaces` covers its own
submodules (`cottax.interfaces.<module>`); every other entry is exact. `cottax` itself
is the bare `import cottax` the notebooks make to print which checkout answered -- a
path probe, not a name taken off it, and `import cottax.core` is still refused."""

PRIVATE: frozenset[str] = frozenset()
"""The private cottax names the port uses: none, and nothing may be added without
saying so here."""


def _allowed(module: str) -> bool:
    """Whether `module` is one of `ALLOWED` -- `cottax.interfaces` by prefix, the rest
    exactly, so `cottax.execution.drivers` does not ride in on
    `cottax.execution.drivers.kinds`.
    """
    return module in ALLOWED or module.startswith("cottax.interfaces.")


def _sources() -> list[tuple[str, str]]:
    """`(where, source)` per thing to read: every `.py` under the port and
    `paper_tests/`, then every code cell of every notebook, each as its own source so
    one unparseable cell names itself.
    """
    out = [
        (str(path), path.read_text())
        for root in (PORT, PAPER)
        for path in sorted(root.rglob("*.py"))
    ]
    for path in sorted(PORT.rglob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        out += [
            (f"{path}[{i}]", cell.source)
            for i, cell in enumerate(notebook.cells)
            if cell.cell_type == "code"
        ]
    return out


def _imports() -> dict[str, set[str]]:
    """`{module: {name, ...}}` -- every `from cottax... import ...` in the port."""
    surface: dict[str, set[str]] = {}
    for where, source in _sources():
        for node in ast.walk(ast.parse(source, filename=where)):
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


def test_the_port_is_written_over_the_interface_and_nothing_else():
    """Every cottax module the port imports from is one of `ALLOWED`."""
    outside = {
        module: sorted(names)
        for module, names in _imports().items()
        if not _allowed(module)
    }
    assert not outside, (
        f"the port reaches past `cottax.interfaces` into {len(outside)} module(s): "
        + "; ".join(f"{m} ({', '.join(n)})" for m, n in sorted(outside.items()))
        + " -- either the interface lacks the name, which is a request upstream, or "
        "this is something a study should not be doing"
    )
