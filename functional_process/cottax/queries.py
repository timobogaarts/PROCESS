"""The graph queries this codebase spells its own way."""

from cottax.abstract import problems
from cottax.graph import Graph
from cottax.names import NodePath


def declared(graph: Graph) -> tuple[NodePath, ...]:
    """The problems of a graph -- `cottax.abstract.problems`, over the graph's definitions."""
    return problems(graph.definitions)
