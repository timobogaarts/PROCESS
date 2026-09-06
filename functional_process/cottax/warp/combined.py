"""The full topologically-ordered walk of a SAND block: `Leaf`s (via the real
resolver) interleaved with the **structural** `Compare`/`Pairwise`/`_Negate` nodes
`sand_leaves()` deliberately skips (they own no leaf function -- the node itself *is*
the arithmetic, per `cottax.rewrites.Compare`/`Pairwise` and
`functional_process.cottax.sand._Negate`). Emitting a residual vector needs both: the
4 structural nodes on `helias_5b` are (per the coordinator) almost certainly among its
11 conditions.

Reuses `leaves.py`'s `_assemble`/`Leaf` and `resolve.py`'s
`resolve`/`Structural`/`Unresolved` -- read-only, by inserting that directory onto
`sys.path`, never editing it.
"""
from __future__ import annotations

import dataclasses
import operator


from .leaves import Leaf, _assemble, _stringify_prelude
from .resolve import Structural, Unresolved, resolve
try:
    from resolve import Composition  # noqa: E402  -- the resolver's new arity-invariant
    # exception (2026-09-06 19:4x): a node whose declared output count matches several
    # DISTINCT candidate calls, genuinely ambiguous. Treated the same as `Unresolved`
    # here -- reported and skipped, never guessed at.
except ImportError:
    Composition = Unresolved  # older resolve.py without the fix -- degrade gracefully


@dataclasses.dataclass(frozen=True)
class StructuralOp:
    """A `Compare`/`Pairwise`/`_Negate` node: no leaf function, just an inline binary
    subtraction or unary negation between two (or one) `VarPath`s. `op` is `"sub"` or
    `"neg"`; refuses (raises) rather than guess at anything else."""

    node: str
    op: str
    inputs: tuple
    outputs: tuple

    def dependencies(self) -> tuple:
        """Same contract as `Leaf.dependencies` -- a structural node has no prelude,
        so this is just its `inputs`. Present so a closure/ordering walk can call it
        uniformly on either entry type."""
        return self.inputs


def _classify_structural(node_def) -> str:
    """`node_def.fn` for a `Structural`-raising node -> `"sub"` or `"neg"`.

    `Compare.fn` is `operator.sub` itself for one pair, or `Pairwise(operator.sub)`
    (any callable, in general -- `rewrites.py`'s `Compare.compare` defaults to
    `operator.sub` but is not typed to only ever be that) for several; `_Negate` is
    `functional_process.cottax.sand._Negate`, always unary `-metric`. Refuses
    (`ValueError`) on any comparator this port has not actually used, rather than
    guessing its Warp rendering.
    """
    fn = node_def.fn
    type_name = type(fn).__name__
    if type_name == "_Negate":
        return "neg"
    compare = getattr(fn, "compare", fn)  # Pairwise(compare) -> compare; else fn itself
    if compare is operator.sub:
        return "sub"
    raise ValueError(f"structural node with unhandled comparator {compare!r} "
                      f"(type {type_name}) -- refusing rather than guessing its Warp form")


def combined_ordered(config: str):
    """`(entries, unresolved, drive)` for `config`'s SAND `Drive`, walked once in
    topological order. `entries` is a list of `Leaf | StructuralOp`, interleaved
    exactly as the real graph orders them -- the ordering the emitter needs to bind
    each node's inputs from what came before. `unresolved` is `[(node, reason), ...]`
    for anything neither a leaf nor a recognised structural node (should be small and
    named -- report it, do not paper over it)."""
    drive, _report = _assemble(config)
    sub = drive.body.subgraph
    entries = []
    unresolved = []
    for name in sub.topological_order:
        node_def = sub[name]
        try:
            (
                leaf_fn,
                order,
                inputs,
                statics,
                output_index,
                locals_,
                prelude,
            ) = resolve(node_def)
        except Structural:
            try:
                op = _classify_structural(node_def)
            except ValueError as exc:
                unresolved.append((name.path_str(), str(exc)))
                continue
            entries.append(
                StructuralOp(
                    node=name.path_str(),
                    op=op,
                    inputs=tuple(i.var.path_str() for i in node_def.inputs),
                    outputs=tuple(v.path_str() for v in node_def.owns),
                )
            )
            continue
        except (Unresolved, Composition) as exc:
            unresolved.append((name.path_str(), str(exc)))
            continue
        entries.append(
            Leaf(
                node=name.path_str(),
                fn=leaf_fn.__name__,
                inputs=tuple(v.path_str() for v in inputs),
                outputs=tuple(v.path_str() for v in node_def.owns),
                module=leaf_fn.__module__,
                statics=statics,
                order=order,
                output_index=output_index,
                locals_=locals_,
                prelude=_stringify_prelude(prelude),
            )
        )
    return entries, unresolved, drive
