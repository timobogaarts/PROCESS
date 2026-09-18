"""The graph queries this codebase spells its own way."""

from cottax.abstract import problems
from cottax.graph import Graph
from cottax.names import NodePath
from cottax.plan import Nest, Plan


def declared(graph: Graph) -> tuple[NodePath, ...]:
    """The problems of a graph -- `cottax.abstract.problems`, over the graph's definitions."""
    return problems(graph.definitions)


def nested_inside(graph: Graph, outer: NodePath) -> Graph:
    """`graph` with every other outermost problem on `outer`'s cycle answered inside its
    iteration -- one `Nest` per problem, which is all cottax's old `NestInside` was.
    Every problem already nested somewhere is left where it is, so the innermost pairs
    are stated first and this is called for the level above."""
    component = next(c for c in graph.components if outer in c)
    plan = Plan(graph)
    for name in graph.subgraph(component).outermost_problems:
        if name != outer:
            plan = plan + Nest(name, outer)
    return plan.graph


def interior(blocking, index: int):
    """How block `index` of `blocking` is blocked one level down.

    Spelled here once because cottax moved it: `Blocking.inner[i]` up to `63bae67`,
    `blocking.entries[i].interior` in the working tree after it.
    """
    entries = getattr(blocking, "entries", None)
    if entries is not None:
        return entries[index].interior
    return blocking.inner[index]
