"""The kernel emitter: walk a SAND block's leaves in graph order and emit one
`@wp.kernel` that evaluates its residual vector.

Shape (`CLAUDE.md` / the brief):
  - inputs: the block's unknowns and boundary inputs, as `wp.array2d(dtype=wp.float64)`
    (one row per batch point -- `wp.launch(kernel, dim=n_points)`, one thread per point).
  - body: call each leaf's `@wp.func` in order, binding its inputs from previously
    computed values (unknowns, boundary, or an earlier leaf's outputs).
  - output: the block's conditions (the residual vector), written to a third
    `wp.array2d(dtype=wp.float64)`.

A `VarPath` is not a legal identifier (`mapper.IdentifierMapper` handles that,
injectively). Everything is `wp.float64` -- Warp does not promote (§80).
"""
from __future__ import annotations

from .mapper import IdentifierMapper


class EmitError(Exception):
    pass


def _render_static(val) -> str:
    """A frozen `(name, value)` from `Leaf.statics` as a Warp literal. The contract
    (`leaves.py`'s `Leaf.statics` docstring) guarantees `val` is a plain
    `float`/`int`/`bool` -- and every parameter in a transpiled leaf is uniformly
    `wp.float64` (the transpiler annotates every parameter that way, `bool`/`int`
    switches included, since Warp has no enum type and the leaf's own body already
    treats them as float64), so every static renders as `wp.float64(...)`."""
    if isinstance(val, bool):
        return f"wp.float64({1.0 if val else 0.0!r})"
    return f"wp.float64({float(val)!r})"


def _local_ident(node: str, ident: str, mapper: IdentifierMapper) -> str:
    """The kernel-local identifier for one `PreludeCall` target. Namespaced by node so
    two nodes' locals -- and a local and a real `VarPath` -- can never collide, the
    same trick the `::ret{i}` scratch names already use."""
    return mapper.get(f"{node}::{ident}")


def _call_terms(node, order, inputs, statics, locals_, mapper) -> list:
    """One call's arguments, in the CALLEE's own signature order.

    Walk `order`; a name present in `statics` renders as a literal, one present in
    `locals_` renders as the prelude-computed local it names, and anything else
    consumes the next `inputs` VarPath in sequence (the ordering contract in
    `leaves.py`'s `Leaf.order`). Falls back to plain positional `inputs` (the original,
    pre-`statics` contract) when `order` is empty -- true for the hand-built stub and
    for any `Leaf` that predates the extension.
    """
    if not order:
        return [mapper.get(p) for p in inputs]
    statics_map = dict(statics)
    locals_map = dict(locals_)
    input_iter = iter(inputs)
    terms = []
    for name in order:
        if name in statics_map:
            val = statics_map[name]
            if isinstance(val, (tuple, list)):
                # A SEQUENCE static (`self.coefficients`, `self.imp_indices`) has no
                # Warp argument at all: Warp has no tuple type, so `leaf_funcs.
                # _SubstituteSequenceStatics` substituted its literal values into the
                # `@wp.func`'s body and dropped the parameter from its signature. The
                # call site must drop it too, or the arity no longer matches.
                continue
            terms.append(_render_static(val))
        elif name in locals_map:
            terms.append(_local_ident(node, locals_map[name], mapper))
        else:
            terms.append(mapper.get(next(input_iter)))
    return terms


def _reconstruct_call_args(leaf, mapper: IdentifierMapper, extra_table_args=None) -> list:
    """The real call to `leaf.fn`, in its own signature order, plus any trailing TABLE
    identifiers its body (or a callee's) needs.

    A table identifier IS the kernel parameter name -- it comes from
    `leaf_funcs_arrays`'s `table_registry`, is tied to no VarPath, and needs no mapper
    lookup. `extra_table_args` is empty for a purely scalar kernel.
    """
    terms = _call_terms(
        leaf.node,
        getattr(leaf, "order", ()),
        leaf.inputs,
        leaf.statics,
        getattr(leaf, "locals_", ()),
        mapper,
    )
    terms.extend((extra_table_args or {}).get((leaf.module, leaf.fn), ()))
    return terms


def build_kernel_source(
    leaves,
    unknowns: tuple,
    boundary: tuple,
    conditions: tuple,
    kernel_name: str = "sand_residual",
    arities: dict | None = None,
    array_boundary: dict | None = None,
    extra_table_args: dict | None = None,
    table_registry: dict | None = None,
) -> tuple[str, IdentifierMapper]:
    """`arities`: `{(module, fn): return_arity}`, from `leaf_funcs.build_leaf_funcs_
    source` -- the leaf's TRUE transpiled return arity, needed whenever a leaf carries
    `output_index` (its real arity is wider than the node's declared output count, and
    Warp's multi-return functions must be unpacked to their FULL arity, not just up to
    the highest selected index). `None`/missing falls back to `max(output_index) + 1`,
    which under-unpacks whenever the leaf returns more values after the last selected
    one -- only safe when every selected leaf's true arity is already known via
    `arities`.

    `array_boundary`: `{VarPath: shape}` for every boundary path that is genuinely
    array-valued. Each binds as its OWN kernel parameter (`wp.array` for a 1-D field,
    `wp.array2d` for a 2-D one such as `.impurity_radiation.temp_impurity_keV_array`,
    `(14, 200)`) instead of a `p[tid, i]` scalar column -- one array object shared by
    every thread in the batch, exactly like a config constant that `p` replicates
    identically per row today, just not copied N times. `boundary` still lists every
    boundary path; the scalar ones are read from `p` as before.

    `table_registry`: `{identifier: values}` from `leaf_funcs_arrays` -- module-level
    constant tables a transpiled body needed for `jnp.interp`, tied to no VarPath.
    Each contributes one more `wp.array` parameter, named by its own key; the caller
    binds the real data at launch. `extra_table_args` says which leaves take which.

    All three default to empty, which reduces this to the scalar-only kernel exactly.

    Returns `(source, mapper)` -- the mapper is returned so a caller can translate its
    own arrays' column order back to VarPaths."""
    array_boundary = array_boundary or {}
    extra_table_args = extra_table_args or {}
    table_registry = table_registry or {}
    mapper = IdentifierMapper()
    lines: list[str] = []
    lines.append("    tid = wp.tid()")

    for i, path in enumerate(unknowns):
        lines.append(f"    {mapper.get(path)} = x[tid, {i}]")
    scalar_boundary = [p for p in boundary if p not in array_boundary]
    for i, path in enumerate(scalar_boundary):
        lines.append(f"    {mapper.get(path)} = p[tid, {i}]")
    # An array-boundary path's identifier IS the kernel parameter: no per-point read,
    # but `mapper.get` must be called once so `mapper.known(path)` reports true for the
    # availability check below, and so the identifier it mints becomes the parameter
    # name in the signature.
    array_boundary_idents = {path: mapper.get(path) for path in array_boundary}

    for leaf in leaves:
        is_structural = type(leaf).__name__ == "StructuralOp"
        deps = leaf.dependencies() if hasattr(leaf, "dependencies") else leaf.inputs
        missing = [p for p in deps if not mapper.known(p)]
        if missing:
            label = f"{leaf.node!r}" if is_structural else f"{leaf.node!r} ({leaf.module}.{leaf.fn})"
            raise EmitError(
                f"node {label} needs {missing}, which no earlier node/input produced "
                f"-- not in topological order, or an unknown/boundary declaration is "
                f"missing"
            )

        if is_structural:
            # `Compare`/`Pairwise`/`_Negate`: no leaf function, the node itself is the
            # arithmetic (`combined.py`'s `_classify_structural`) -- emitted inline
            # rather than as a call, exactly as the coordinator asked.
            out_idents = [mapper.get(p) for p in leaf.outputs]
            if leaf.op == "neg":
                in_ident = mapper.get(leaf.inputs[0])
                lines.append(f"    {out_idents[0]} = -{in_ident}")
            elif leaf.op == "sub":
                for i, out_ident in enumerate(out_idents):
                    l_ident = mapper.get(leaf.inputs[2 * i])
                    r_ident = mapper.get(leaf.inputs[2 * i + 1])
                    lines.append(f"    {out_ident} = {l_ident} - {r_ident}")
            else:
                raise EmitError(f"structural node {leaf.node!r}: unhandled op {leaf.op!r}")
            continue

        # Prelude (a "Composition" wrapper): the locals the wrapper's own body computes
        # before the call that produces this node's outputs, emitted here as ordinary
        # `@wp.func` calls in dependency order. Each local carries the value the
        # wrapper actually computes for it -- the resolver never binds one to a
        # same-named graph variable or a default (`resolve.PreludeCall`).
        for pc in getattr(leaf, "prelude", ()):
            pre_terms = _call_terms(
                leaf.node, pc.order, pc.inputs, pc.statics, pc.locals_, mapper
            )
            n_returns = (arities or {}).get((pc.module, pc.fn))
            if n_returns is not None and n_returns != len(pc.targets):
                raise EmitError(
                    f"leaf {leaf.node!r}: prelude `{pc.source}` unpacks "
                    f"{len(pc.targets)} value(s) from {pc.fn}, whose transpiled return "
                    f"arity is {n_returns} -- not guessed"
                )
            tgt_idents = [_local_ident(leaf.node, t, mapper) for t in pc.targets]
            lines.append(
                f"    {', '.join(tgt_idents)} = {pc.fn}({', '.join(pre_terms)})"
            )

        arg_terms = _reconstruct_call_args(leaf, mapper, extra_table_args)
        out_idents = [mapper.get(p) for p in leaf.outputs]
        call = f"{leaf.fn}({', '.join(arg_terms)})"
        output_index = getattr(leaf, "output_index", None)
        if len(out_idents) == 0:
            raise EmitError(f"leaf {leaf.node!r} declares no outputs")
        elif output_index is not None:
            # The leaf's own return is WIDER than what this node owns (a
            # `.fn`-field wrapper -- `_NormalisedResidual`/constraint nodes -- that
            # selects one element of `eq`/`leq`/`geq`'s 4-tuple; see
            # `resolve.py`'s `_subscript_select_index`). Call once, bind every
            # returned value to a scratch identifier (never reused, so this cannot
            # collide with a real VarPath's identifier), then assign only the
            # positions `output_index` names to this node's real outputs.
            if len(output_index) != len(out_idents):
                raise EmitError(
                    f"leaf {leaf.node!r}: output_index has {len(output_index)} "
                    f"entries but the node owns {len(out_idents)} output(s)"
                )
            n_returns = (arities or {}).get((leaf.module, leaf.fn))
            if n_returns is None or n_returns < max(output_index) + 1:
                n_returns = max(output_index) + 1
            scratch = [mapper.get(f"{leaf.node}::ret{i}") for i in range(n_returns)]
            lines.append(f"    {', '.join(scratch)} = {call}")
            for out_ident, sel in zip(out_idents, output_index):
                lines.append(f"    {out_ident} = {scratch[sel]}")
        elif len(out_idents) == 1:
            lines.append(f"    {out_idents[0]} = {call}")
        else:
            lines.append(f"    {', '.join(out_idents)} = {call}")

    for i, path in enumerate(conditions):
        if not mapper.known(path):
            raise EmitError(
                f"condition {path!r} is never produced by any leaf or declared as an "
                f"unknown/boundary input"
            )
        lines.append(f"    r[tid, {i}] = {mapper.get(path)}")

    params = ["x: wp.array2d(dtype=wp.float64)"]
    if scalar_boundary:
        params.append("p: wp.array2d(dtype=wp.float64)")
    for path, shape in array_boundary.items():
        # A genuinely 2-D boundary field (`.impurity_radiation.temp_impurity_keV_array`,
        # `(14, 200)` -- one 200-point <Z>(T_e) table per species) binds as ONE
        # `wp.array2d`, not fourteen 1-D parameters: Warp indexes `a[row, k]` natively,
        # so the species index stays data instead of becoming part of fourteen
        # generated identifiers that a single off-by-one would silently permute.
        ndim = len(shape) if isinstance(shape, (tuple, list)) else 1
        if ndim >= 3:
            raise EmitError(
                f"boundary {path!r} is {ndim}-dimensional -- only 1-D and 2-D array "
                f"parameters are bound; refusing"
            )
        kind = "wp.array2d" if ndim == 2 else "wp.array"
        params.append(f"{array_boundary_idents[path]}: {kind}(dtype=wp.float64)")
    for ident in sorted(table_registry):
        params.append(f"{ident}: wp.array(dtype=wp.float64)")
    params.append("r: wp.array2d(dtype=wp.float64)")
    sig = f"@wp.kernel\ndef {kernel_name}({', '.join(params)}):\n"

    src = sig + "\n".join(lines) + "\n"
    return src, mapper


def assemble_module(func_source: str, kernel_source: str) -> str:
    """The full, self-contained `.py` module text: imports, leaf funcs, then the kernel."""
    header = (
        '"""GENERATED by functional_process/cottax/warp -- do not hand-edit. See '
        "`_audit/optimise_design.md` §74/§79/§80 for the design this "
        'implements."""\n'
        "import warp as wp\n\n"
    )
    return header + func_source + "\n\n" + kernel_source
