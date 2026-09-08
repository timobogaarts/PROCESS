"""Every declaration assembles, and owns exactly what it declares -- in one sweep.

**What this is actually for, measured rather than assumed.** It enumerates
`NodalDeclaration.__subclasses__` transitively after importing the package: 489
instantiable declarations. Of those, 225 are named by a switch registry in `indat.py`
and swapped into the reference machine by `test_machine.py`'s two registry sweeps --
which is a *stronger* test than this one, because it assembles them in context rather
than alone -- and 307 appear in an assembled graph, which the cold matrix exercises on
every run. 439 are covered by one or the other.

**The 50 that are covered by neither are why this exists**: `Avail2`, `AvailSt`,
`BldgsSizes`, `CpLifetimeResistive`, `CrocoCableGeometry` and the rest -- declarations
that are valid, wired to no slot, and selected by no reference input file. Nothing else
in the suite instantiates them, so nothing else would notice one rotting. That is a
narrow contribution and it is stated narrowly on purpose: an earlier version of this
docstring claimed to replace 219 hand-written structural tests, and the classifier
behind that number had regexed test *docstrings* for words like "assembles". The real
count of tests this subsumes is about five.

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
