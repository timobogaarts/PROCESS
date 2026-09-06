"""`sand_leaves(config)`: the node -> leaf-function resolver the Warp emitter needs,
walking a configuration's SAND `Drive` in topological order.

Builds the same assembly `run_cold_matrix.build_sand`/`run_sand_harness.py` do (PROCESS
run -> `machine_from_indat` -> `graph_for` -> `mda_env` -> `sand.assemble` ->
`sand.sand_schedule` -> `sand.sand_shape`), then walks `drive.body.subgraph`'s
`topological_order` -- the block with its `Optimise` problem node taken out, which
§79 measured as "pure dataflow" (`Residualise`+`Combine` turn every `FixedPoint` into a
residual condition before scheduling), so it is acyclic and `Graph.topological_order`
applies directly rather than needing `Blocking.scc` first.

Each interior node resolves through `resolve.resolve` (see that module's docstring for
the wrapper shapes it was built to see through). A resolution failure does not stop the
walk -- it is recorded and the caller decides what to do with an unresolved node, the
same "report every phase, do not let one failure hide the rest" discipline the rest of
this port's harnesses use.
"""
import dataclasses

import jax

jax.config.update("jax_enable_x64", True)

from functional_process.cottax import sand
from functional_process.cottax.indat import REFERENCE_INPUT_FILE, graph_for, machine_from_indat
from functional_process.cottax.run_cold_matrix import _resolve
from functional_process.cottax.sand_harness import assemble as sand_assemble
from functional_process.cottax.sand_harness import mda_env, reference_run

from .resolve import Composition, Structural, Unresolved, resolve


@dataclasses.dataclass(frozen=True)
class Leaf:
    node: str
    """The node's `path_str()`."""

    fn: str
    """The leaf function's `__name__`, matching the emitted `@wp.func` name."""

    inputs: tuple[str, ...]
    """`VarPath.path_str()` of each DYNAMIC argument (one this node reads off the
    graph), in the order those arguments appear among `leaf_fn`'s parameters -- may
    skip over static ones interleaved between them. Reconstruct the real call with
    `order`, below."""

    outputs: tuple[str, ...]
    """`VarPath.path_str()` this node owns, in return order."""

    module: str
    """The leaf's `__module__`, to disambiguate a duplicate `fn` name."""

    statics: tuple[tuple[str, object], ...] = ()
    """`(parameter name, frozen value)` for every argument that has no `VarPath`
    because it was fixed at assembly -- a switch (`ireactor`, `istell`), a node's own
    field (`self.imp_indices`), or a bare literal in the wrapper's source (`0.0`).
    `value` is either a plain `float`/`int`/`bool`, ready to render as a Warp literal
    at the CALL SITE (`wp.float64(x)`/`wp.int32(x)`) -- an `IntEnum` member is
    converted to `int(...)` first, since Warp has no enum type -- **or a `tuple` of
    such scalars**, which has no call-site rendering at all (Warp has no tuple type)
    and is instead monomorphised INTO the emitted `@wp.func`'s body, its parameter
    dropped from both the signature and the call (`leaf_funcs.
    _SubstituteSequenceStatics` / `emit._reconstruct_call_args`). An argument this
    resolver cannot classify as either dynamic or one of those two (an array, a
    dataclass, a dict, a callable) leaves the whole node `Unresolved` rather than
    appearing here with a guessed rendering -- see `resolve.py`'s `_static_value` and
    `_static_sequence_value`, the latter of which also records why reading a bound
    `self.<attr>` is a fact about an object that exists rather than a guess.
    """

    order: tuple[str, ...] = ()
    """Every one of `leaf_fn`'s own parameter names, in its signature order -- the
    full call the emitter must reconstruct, `inputs` and `statics` interleaved.

    **Ordering contract**: walk `order`; for each name, look it up in
    `dict(leaf.statics)` first (present -> render that literal), otherwise pop the
    next value off `inputs` in sequence (present -> that argument is this `VarPath`,
    already read). `inputs`'s i-th entry corresponds to the i-th DYNAMIC name in
    `order`, not to `order[i]` itself -- `order` is the only field with one entry per
    parameter; `inputs`/`statics` are each a subsequence of it.
    """

    output_index: tuple[int, ...] | None = None
    """`None` for the ordinary case: calling `fn` returns exactly `len(outputs)`
    values, in order, one per entry of `outputs`. Otherwise a tuple the same length as
    `outputs`, naming which position(s) of `fn`'s WIDER return tuple each output
    actually is -- e.g. `(1,)` for a constraint node, whose leaf (`constraint_<id>`,
    forwarding to `eq`/`leq`/`geq`) returns `(residual, normalised_residual,
    constraint_value, constraint_bound)` but the node owns only the condition
    (`normalised_residual`, index 1 -- see `resolve.py`'s `_subscript_select_index`,
    which derives this from `sand.py`'s `_NormalisedResidual.__call__` rather than
    asserting it). The emitter must call `fn` once, capture its full return, and pick
    these positions -- never assume arity `len(outputs)` when this is set.
    """

    locals_: tuple = ()
    """`(parameter name, local identifier)` for each argument of `fn` bound to a value
    the WRAPPER computes locally rather than to a graph variable or a literal -- see
    `prelude`. Rendered by the same walk over `order` as `statics`: look the name up in
    `dict(statics)` first, then in `dict(locals_)`, and only otherwise consume the next
    `inputs` entry."""

    prelude: tuple = ()
    """`resolve.PreludeCall`s (with `inputs` stringified to `path_str`), in the order
    they must be evaluated, producing every identifier `locals_` -- and any later
    prelude entry -- refers to. Emitted as ordinary `@wp.func` calls immediately BEFORE
    this leaf's own call.

    This is the "Composition" node class the resolver used to refuse outright: a
    wrapper whose body computes one or more locals and then passes them into the call
    that produces the node's outputs. A local is ALWAYS bound to the value the
    wrapper's own code computes for it -- never to a same-named boundary variable, a
    default, or a zero, each of which would compute the wrong thing while looking like
    ordinary float noise whenever the local happens to be nearly constant.

    `dependencies()` includes each prelude's own `inputs`, so a topological/prefix
    closure walk sees them; every entry's `fn`/`module` must be transpiled alongside
    the leaf's own."""

    def dependencies(self) -> tuple:
        """Every graph `VarPath` (as `path_str`) this entry reads -- the leaf call's
        own `inputs` PLUS every prelude's, de-duplicated, preludes first. `inputs` is
        positional (it must stay aligned with `order`), so closure and ordering checks
        must use this instead of `inputs`."""
        seen, out = set(), []
        for pc in self.prelude:
            for p in pc.inputs:
                if p not in seen:
                    seen.add(p)
                    out.append(p)
        for p in self.inputs:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return tuple(out)


def _stringify_prelude(prelude) -> tuple:
    """`resolve.PreludeCall`s with `inputs` turned from `VarPath` into `path_str`,
    matching `Leaf.inputs`'s own convention (the resolver works in `VarPath`s; every
    consumer downstream of `leaves.py` works in strings)."""
    return tuple(
        dataclasses.replace(pc, inputs=tuple(v.path_str() for v in pc.inputs))
        for pc in prelude
    )


def _assemble(config: str):
    """`(drive, report)` for `config` -- the bare stem of a
    `tests/regression/input_files/<config>.IN.DAT`."""
    path = _resolve(f"tests/regression/input_files/{config}.IN.DAT")
    is_reference = path == _resolve(REFERENCE_INPUT_FILE)
    reference = reference_run(str(path))
    machine = machine_from_indat(str(path))
    machine_graph = None if is_reference else graph_for(machine)
    driven, env = mda_env(reference, graph=machine_graph)
    cold = reference.cold
    switch_values = (
        None
        if is_reference
        else sand.switch_values_for(cold, reference.icc, reference.i_figure_merit)
    )
    combined, report = sand_assemble(reference, driven, env, switch_values=switch_values)
    schedule = sand.sand_schedule(combined, None, bounds=reference.bounds)
    shape = sand.sand_shape(schedule)
    return shape["drive"], report


def sand_leaves(config: str) -> list:
    """Topologically ordered leaves of `config`'s SAND `Drive`.

    `config` is a bare stem, e.g. `"helias_5b"`. Returns only the nodes that resolved
    to a `functional_process.models` (or, for a constraint/objective, a ported
    `constraint_<id>`/`objective_metric_<id>`) leaf -- call `sand_report(config)` for
    the full per-node accounting (resolved / structural / unresolved-and-why).
    """
    drive, _report = _assemble(config)
    sub = drive.body.subgraph
    out = []
    for name in sub.topological_order:
        node_def = sub[name]
        try:
            leaf_fn, order, inputs, statics, output_index, locals_, prelude = resolve(
                node_def
            )
        except (Structural, Unresolved, Composition):
            continue
        out.append(
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
    return out


def sand_report(config: str) -> dict:
    """Full per-node accounting for `config`'s SAND `Drive`: every node in topological
    order, classified `resolved` / `structural` / `unresolved` (with the reason for the
    last), plus the ordered `Leaf` list `sand_leaves` returns.

    Kept separate from `sand_leaves` so the emitter's contract (a plain list of `Leaf`)
    never has to change shape to carry reporting detail.
    """
    drive, assemble_report = _assemble(config)
    sub = drive.body.subgraph
    node_order = sub.topological_order
    leaves, structural, unresolved, composition = [], [], [], []
    for name in node_order:
        node_def = sub[name]
        try:
            leaf_fn, arg_order, inputs, statics, output_index, locals_, prelude = resolve(
                node_def
            )
        except Structural as exc:
            structural.append((name.path_str(), str(exc)))
            continue
        except Composition as exc:
            composition.append((name.path_str(), str(exc)))
            continue
        except Unresolved as exc:
            unresolved.append((name.path_str(), str(exc)))
            continue
        leaves.append(
            Leaf(
                node=name.path_str(),
                fn=leaf_fn.__name__,
                inputs=tuple(v.path_str() for v in inputs),
                outputs=tuple(v.path_str() for v in node_def.owns),
                module=leaf_fn.__module__,
                statics=statics,
                order=arg_order,
                output_index=output_index,
                locals_=locals_,
                prelude=_stringify_prelude(prelude),
            )
        )
    return {
        "config": config,
        "drive_nodes": len(node_order),
        "leaves": leaves,
        "structural": structural,
        "unresolved": unresolved,
        "composition": composition,
        "assemble_report": assemble_report,
    }
