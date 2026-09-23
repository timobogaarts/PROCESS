"""The graph queries this codebase spells its own way."""

from cottax.interfaces import Graph, Nest, Plan, problems
from cottax.pytree.path import NodePath


def declared(graph: Graph) -> tuple[NodePath, ...]:
    """The problems of a graph -- `cottax.interfaces.problems`, over its definitions."""
    return problems(graph.definitions)


def nested_inside(graph: Graph, outer: NodePath) -> Graph:
    """`graph` with every other outermost problem on `outer`'s cycle answered inside its
    iteration -- one `Nest` per problem, which is all cottax's old `NestInside` was.
    Every problem already nested somewhere is left where it is, so the innermost pairs
    are stated first and this is called for the level above.
    """
    component = next(c for c in graph.graph.components if outer in c)
    plan = Plan(graph)
    for name in graph.subgraph(component).outermost_problems:
        if name != outer:
            plan = plan + Nest(name, outer)
    return plan.graph


def component_of(graph: Graph, node: NodePath) -> int:
    """Which component of `graph` (in `graph.graph.components` order -- the order
    `sequencing.entries` and `Schedule.steps` share) `node` is in.

    Spelled here once because cottax moved it: `blocking.index[node]` up to `63bae67`;
    since `4d33cf3` there is no partition value, and a graph's components are its own.

    Raises
    ------
    KeyError
        If `node` is not a node of `graph`.
    """
    for index, component in enumerate(graph.graph.components):
        if node in component:
            return index
    raise KeyError(f"not a node of this graph: {node!r}")
