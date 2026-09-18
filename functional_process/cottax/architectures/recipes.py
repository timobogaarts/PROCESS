"""The paper's recipe for an executable graph, mechanically: **cut and determine the
coupling variables of every TOSCC component**, then combine and nest.

`mda.cut_graph` cuts nine hand-measured variables (`mda.CUTS`), chosen so that one
Picard iterate is one PROCESS pass. This module derives the cut set from the graph
alone, three ways, so that the architectures the paper describes are produced by the
operations it describes and not by a table:

- **Jacobi** -- every coupling variable of a component is cut. The body a Picard
  iterates is then a DAG in which no coupled node reads another's *current* value, so
  every node of the component can be evaluated from the previous iterate at once.
- **Gauss-Seidel** in a given node order -- only the *backward* reads are cut, so each
  node reads the latest value of everything before it in the order and one Picard
  iterate is one sweep. The order is the "heuristically chosen coupling order" the paper
  mentions; PROCESS's own call order is the graph's binding order and is the default.
- **Gauss-Seidel, minimal** -- the order that cuts the fewest variables, found exactly
  by a subset DP over the component (components here have at most nine nodes).

A coupling variable is one owned in the component and read by a *function* node of the
component other than its owner. A condition node's reads are never cut (a problem reads
the value it constrains); a fixed point's unknown is never cut (that problem is combined
into the component's own iteration, where its unknown is already a previous-iterate
value to every reader); and where the owner is a root find, the read by the model its
solve drives is left alone -- cutting it would only mint a second copy of the problem's
own unknown. Everything else, including a nested solve's output read by another model,
is cut, which is what lets the same cut set serve IDF and SAND.

Only the cycles that survive removing a component's declared problems are cut: a
model's own fixed-point self-loop or root find already closes every cycle through it.
After the cut the component holds one new `FixedPoint` over its copies, and each
declared problem is `Nest`ed inside it -- a local solve converging inside each outer
iterate, as inside a PROCESS pass. (Folding a self-loop into the cut's problem instead
was measured to put PROCESS's `max(u, m)` ratchet into the block's linear solve, where
`I - J` is exactly singular whenever the input wins.) That is the MDF shape; `sand`
and `mdf` take the result exactly as they take `cut_graph`'s.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence

import networkx as nx
from cottax.graph import Graph
from cottax.problem import ConditionalNode, is_fixed_point, is_root_find
from cottax.rewrites import Combine, Cut, FixedPointCut, Nest
from cottax.spec import NodePath, VarPath
from jax.tree_util import GetAttrKey

Component = tuple[NodePath, ...]

RECIPES = ("jacobi", "gauss_seidel", "gauss_seidel_minimal")

MAX_EXACT_ORDER = 18
"""Above this many nodes the exact order search (`2^n n`) is refused, not attempted."""


# ---------------------------------------------------------------- what couples


def coupling_reads(graph: Graph, component: Sequence[NodePath]) -> dict:
    """`{variable: (readers, ...)}` -- every read of a component-owned variable by another
    function node of the same component, in binding order of the owner.
    """
    inside = set(component)
    out: dict[VarPath, tuple[NodePath, ...]] = {}
    for var, owner in graph.graph.owners.items():
        if owner not in inside:
            continue
        readers = tuple(
            reader
            for reader in graph.graph.readers.get(var, ())
            if reader in inside
            and reader != owner
            and not isinstance(graph[reader], ConditionalNode)
        )
        if isinstance(graph[owner], ConditionalNode):
            if is_fixed_point(graph[owner]):
                # A fixed point's own unknown. The recipe combines every fixed point of
                # the component into one Picard, and inside that iteration `u` already
                # *is* a previous-iterate value to every reader -- a copy would be a
                # second unknown equal to the first, and its `Start` port would mint the
                # same `^guess.<place>` as `u`'s (measured: `large_tokamak_nof`'s
                # `dr_tf_plasma_case`). A root find's output is different: that solve is
                # nested and converges inside each iterate, so a reader of it sees the
                # current value unless the read is cut.
                continue
            # A solver output. The model whose value this problem constrains reads the
            # unknown as part of the solve itself; cutting that read would mint a copy
            # of an unknown the problem already owns.
            constrained = graph[owner].reads
            driven = {
                node
                for node in inside
                if any(produced in constrained for produced in graph[node].owns)
            }
            readers = tuple(reader for reader in readers if reader not in driven)
        if readers:
            out[var] = readers
    return out


def backward_reads(graph: Graph, component: Sequence[NodePath], order: Sequence[NodePath]) -> dict:
    """`coupling_reads`, restricted to reads by a node *earlier* in `order` than the
    owner -- what a Gauss-Seidel sweep in that order still has to iterate on.
    """
    position = {node: i for i, node in enumerate(order)}
    missing = [n for n in component if n not in position]
    if missing:
        raise KeyError(f"order does not place {missing!r}")
    out = {}
    for var, readers in coupling_reads(graph, component).items():
        owner = graph.graph.owners[var]
        back = tuple(r for r in readers if position[r] < position[owner])
        if back:
            out[var] = back
    return out


def minimal_feedback_order(graph: Graph, component: Sequence[NodePath]) -> Component:
    """The order of `component`'s nodes that a Gauss-Seidel sweep cuts the fewest
    variables in. Exact: a DP over subsets, cost of placing a node after a set being the
    number of its variables read by that set. Ties fall to binding order.
    """
    nodes = tuple(component)
    n = len(nodes)
    if n > MAX_EXACT_ORDER:
        raise ValueError(
            f"{n} nodes is beyond the exact order search ({MAX_EXACT_ORDER}); pass an "
            f"order to `gauss_seidel` instead"
        )
    index = {node: i for i, node in enumerate(nodes)}
    reads = coupling_reads(graph, component)
    # owned[i]: for node i, a list of bitmasks, one per coupling variable it owns, of
    # the readers of that variable. The variable is cut iff any reader precedes i.
    owned: list[list[int]] = [[] for _ in nodes]
    for var, readers in reads.items():
        mask = 0
        for reader in readers:
            mask |= 1 << index[reader]
        owned[index[graph.graph.owners[var]]].append(mask)
    full = (1 << n) - 1
    best = [None] * (1 << n)  # (cost, last node) per subset placed as a prefix
    best[0] = (0, -1)
    for subset in range(1 << n):
        if best[subset] is None:
            continue
        cost, _ = best[subset]
        for i in range(n):
            bit = 1 << i
            if subset & bit:
                continue
            added = sum(1 for mask in owned[i] if mask & subset)
            candidate = (cost + added, i)
            current = best[subset | bit]
            if current is None or candidate[0] < current[0]:
                best[subset | bit] = candidate
    order = []
    subset = full
    while subset:
        _, i = best[subset]
        order.append(nodes[i])
        subset &= ~(1 << i)
    return tuple(reversed(order))


# ---------------------------------------------------------------- the three cut sets


def jacobi_cuts(graph: Graph, component: Sequence[NodePath]) -> tuple[Cut, ...]:
    """One `Cut` per coupling variable, over all its readers."""
    return tuple(
        Cut(var=var, readers=readers)
        for var, readers in coupling_reads(graph, component).items()
    )


def gauss_seidel_cuts(
    graph: Graph, component: Sequence[NodePath], order: Sequence[NodePath] | None = None
) -> tuple[Cut, ...]:
    """One `Cut` per variable read backward in `order` (default: binding order), over
    exactly its backward readers.
    """
    if order is None:
        order = graph.nodes
    return tuple(
        Cut(var=var, readers=readers)
        for var, readers in backward_reads(graph, component, order).items()
    )


def gauss_seidel_minimal_cuts(graph: Graph, component: Sequence[NodePath]) -> tuple[Cut, ...]:
    """`gauss_seidel_cuts` in the order `minimal_feedback_order` finds."""
    return gauss_seidel_cuts(graph, component, minimal_feedback_order(graph, component))


CUTTERS: dict[str, Callable] = {
    "jacobi": jacobi_cuts,
    "gauss_seidel": gauss_seidel_cuts,
    "gauss_seidel_minimal": gauss_seidel_minimal_cuts,
}


# ---------------------------------------------------------------- one component


def _place_for(graph: Graph, component: Sequence[NodePath]) -> NodePath:
    """Where the component's fixed point is bound, before minting: the first function
    node of the component in binding order, with an `.mda` segment.
    """
    first = next(n for n in graph.nodes if n in set(component) and not isinstance(graph[n], ConditionalNode))
    return NodePath((*first.segments, GetAttrKey("mda")))


@dataclasses.dataclass(frozen=True)
class ComponentCut:
    """What the recipe did to one component, for reporting."""

    component: Component
    cuts: tuple[Cut, ...]
    problem: NodePath | None
    """The one fixed point driving the component afterwards, or `None` if it needed no
    cut and declared no fixed point of its own."""
    combined: tuple[NodePath, ...]
    """Fixed points the component declared already that were folded into `problem`
    (only where nothing was cut)."""
    nested: tuple[NodePath, ...]
    """Declared problems -- self-loop fixed points and root finds -- nested inside
    `problem`."""
    ops: tuple = ()
    """The graph operations this record stands for, in the order they were applied:
    the `FixedPointCut` (or `Combine`), then one `Nest` per nested problem."""

    @property
    def n_variables(self) -> int:
        return len(self.cuts)

    @property
    def n_reads(self) -> int:
        return sum(len(c.readers) for c in self.cuts)


def residual_cycles(graph: Graph, component: Sequence[NodePath]) -> tuple[Component, ...]:
    """The cycles of `component` that remain once its declared problems are taken out
    -- what a cut still has to close. A declared fixed point or root find already closes
    every cycle that passes through it (its unknown is a previous-iterate value to every
    reader), so those cycles need no copy; measured, cutting one anyway and folding the
    declared problem into the copy's block puts PROCESS's `max(u, m)` ratchet
    (`dr_tf_plasma_case`) into a block whose `I - J` is exactly singular whenever the
    input arm wins, and the implicit derivative through the block is NaN.
    """
    keep = [n for n in component if not isinstance(graph[n], ConditionalNode)]
    deps = graph.graph._nx_dependencies.subgraph(keep)
    rank = {n: i for i, n in enumerate(graph.nodes)}
    return tuple(
        tuple(sorted(scc, key=rank.__getitem__))
        for scc in nx.strongly_connected_components(deps)
        if len(scc) > 1
    )


def cut_component(
    graph: Graph, component: Sequence[NodePath], cutter: Callable
) -> tuple[Graph, ComponentCut]:
    """Cut one component with `cutter`, then nest what it declared inside the cut's
    fixed point. Returns the graph and a record of what was done.

    Only the cycles that survive removing the component's declared problems are cut
    (`residual_cycles`); a component whose every cycle runs through a declared problem
    is left as it is, which is `mda.cut_graph`'s guard made mechanical. Where a cut is
    made, each declared problem of the component -- a model's own fixed-point self-loop
    or root find -- is `Nest`ed inside the new fixed point rather than `Combine`d into
    it: a local solve converges inside each outer iterate, exactly as it does inside a
    PROCESS pass, and its Jacobian never enters the outer block's linear solve.
    """
    component = tuple(component)
    declared = [p for p in component if isinstance(graph[p], ConditionalNode)]
    cuts: tuple[Cut, ...] = ()
    for cycle in residual_cycles(graph, component):
        cuts += tuple(cutter(graph, cycle))
    place = _place_for(graph, component)
    if cuts:
        cut = FixedPointCut(cuts, place=place)
        graph = cut.apply(graph)
        problem = cut.problem
        nested = tuple(p for p in declared if _same_component(graph, p, problem))
        nests = tuple(Nest(inner, problem) for inner in nested)
        for op in nests:
            graph = op.apply(graph)
        return graph, ComponentCut(component, cuts, problem, (), nested, (cut, *nests))
    # Nothing to cut: the declared problems close every cycle. Several fixed points on
    # one cycle are folded into one iteration; a root find beside a fixed point is
    # nested in it.
    fixed_points = [p for p in declared if is_fixed_point(graph[p])]
    root_finds = [p for p in declared if is_root_find(graph[p])]
    combined: tuple[NodePath, ...] = ()
    ops: list = []
    problem = fixed_points[0] if fixed_points else None
    if len(fixed_points) > 1:
        join = Combine(place, tuple(fixed_points))
        graph = join.apply(graph)
        problem = join.problem
        combined = tuple(fixed_points)
        ops.append(join)
    nested: tuple[NodePath, ...] = ()
    if problem is not None and root_finds:
        for rf in root_finds:
            op = Nest(rf, problem)
            graph = op.apply(graph)
            ops.append(op)
        nested = tuple(root_finds)
    return graph, ComponentCut(component, cuts, problem, combined, nested, tuple(ops))


def _same_component(graph: Graph, a: NodePath, b: NodePath) -> bool:
    return any(a in c and b in c for c in graph.graph.components)


# ---------------------------------------------------------------- the recipe


@dataclasses.dataclass(frozen=True)
class Recipe:
    """A cut function with `mda.cut_graph`'s signature, and a record of its last run."""

    name: str

    def __post_init__(self) -> None:
        if self.name not in CUTTERS:
            raise ValueError(f"{self.name!r} is not one of {RECIPES}")

    def cut(self, graph: Graph) -> tuple[Graph, tuple[ComponentCut, ...]]:
        """Every cyclic component of `graph` cut, combined and nested."""
        cutter = CUTTERS[self.name]
        records = []
        for component in graph.graph.cycles:
            graph, record = cut_component(graph, component, cutter)
            records.append(record)
        return graph, tuple(records)

    def plan(self, graph: Graph) -> tuple["Plan", tuple[ComponentCut, ...]]:
        """`cut`, as a `Plan`: the same graph, with every op the recipe applied
        recorded in `Plan.ops` -- the recipe made visible, one op per line."""
        from cottax.plan import Plan  # noqa: PLC0415

        _, records = self.cut(graph)
        plan = Plan(graph)
        for record in records:
            for op in record.ops:
                plan = plan + op
        return plan, records

    def __call__(self, graph: Graph) -> Graph:
        return self.cut(graph)[0]


def recipe(name: str) -> Recipe:
    return Recipe(name)


def census(graph: Graph) -> dict:
    """What each recipe cuts on `graph`, per component: the numbers the paper reports."""
    out = {}
    for name in RECIPES:
        _, records = Recipe(name).cut(graph)
        out[name] = tuple(
            {
                "nodes": len(r.component),
                "variables": r.n_variables,
                "reads": r.n_reads,
                "problem": None if r.problem is None else r.problem.spelling,
                "combined": len(r.combined),
                "nested": len(r.nested),
            }
            for r in records
        )
    return out
