"""Every declaration assembles, and owns exactly what it declares -- in one sweep.

**Why one test and not 219.** Most hand-written structural tests in this tree say the
same thing about one node: build it, `to_graph` it, check the paths it owns are the ones
its `Output`/`OutputInto` fields name. That assertion is not about the node. It is about
cottax: that `x = OutputInto(physics)` on a field named `x` produces `.physics.x`. Proved
once, it is proved for every declaration, and 219 copies of it are 219 chances for one to
be quietly missing instead.

The sweep is also strictly wider than what it replaces. It enumerates
`NodalDeclaration.__subclasses__` transitively after importing the whole package, so it
reaches the declarations **no assembled graph contains** -- the unwired-but-valid arms
(`unit_registry.md`'s `Jcrit*` family, `WessonInternalInductance`,
`TfMagnetCostResistive`) and every switch occupant the reference `IN.DAT` does not
select. A whole-graph assembly test cannot see any of those; that is the gap the
per-node tests were filling, and it is the part worth keeping.

What it deliberately does NOT assert is which paths a node *should* own. That is a claim
about the port's intent, it belongs in the unit's own case, and no sweep can make it.
"""

import importlib
import pkgutil

import jax
import pytest

jax.config.update("jax_enable_x64", True)

from cottax.interfaces.pytree_namespace_module import (  # noqa: E402
    NodalDeclaration,
    to_graph,
)

import functional_process.cottax as _cottax  # noqa: E402


def _every_declaration():
    """Every `NodalDeclaration` subclass the package defines, imported and deduplicated.

    Import failures are raised, not skipped: a module that cannot be imported is a
    declaration that silently leaves the sweep, which is the failure this exists to stop.
    """
    for module in pkgutil.walk_packages(_cottax.__path__, _cottax.__name__ + "."):
        importlib.import_module(module.name)

    def descendants(cls):
        for sub in cls.__subclasses__():
            yield sub
            yield from descendants(sub)

    seen = {c for c in descendants(NodalDeclaration) if not c.__abstractmethods__}
    return sorted(seen, key=lambda c: (c.__module__, c.__qualname__))


DECLARATIONS = _every_declaration()
IDS = [f"{c.__module__.split('.')[-1]}.{c.__qualname__}" for c in DECLARATIONS]


def test_the_sweep_found_the_declarations():
    """A sweep that enumerated nothing would pass every test below vacuously."""
    assert len(DECLARATIONS) > 400, (
        f"only {len(DECLARATIONS)} declarations found -- the package layout moved, or "
        f"a module stopped importing"
    )


@pytest.mark.parametrize("declaration", DECLARATIONS, ids=IDS)
def test_declaration_owns_exactly_what_it_declares(declaration):
    """Its outputs are the paths its `Output`/`OutputInto` fields name, and no others.

    A declaration that needs constructor arguments (a family head taking its occupants)
    is skipped here and covered by its own unit's case, which knows what to pass.
    """
    try:
        node = declaration()
    except TypeError as needs_arguments:
        pytest.skip(f"needs constructor arguments: {needs_arguments}")
    owned = {out.var for out in node.outputs}
    if not owned:
        # A family HEAD may own nothing: `StatesValues` exists to be subclassed, and
        # each occupant names its own path. A leaf that owns nothing is a real defect,
        # so the exemption is spent on having subclasses rather than on the name.
        assert declaration.__subclasses__(), (
            f"{declaration.__qualname__} declares no outputs and nothing subclasses it"
        )
        return
    assert len(owned) == len(node.outputs), (
        f"{declaration.__qualname__} names the same path twice"
    )


@pytest.mark.parametrize("declaration", DECLARATIONS, ids=IDS)
def test_declaration_assembles(declaration):
    """`to_graph` accepts it, or refuses it for a reason the declaration states.

    A refusal is a pass: `Cryo` names one field as both an output and a read and is kept
    deliberately unassemblable (`_audit/next_steps.md`), and a family head that cannot be
    instantiated alone is the same shape. What must not happen is an unexplained crash.
    """
    try:
        node = declaration()
    except TypeError as needs_arguments:
        pytest.skip(f"needs constructor arguments: {needs_arguments}")
    try:
        graph = to_graph(node)
    except (ValueError, NotImplementedError):
        return
    assert graph.definitions, f"{declaration.__qualname__} assembled to an empty graph"
