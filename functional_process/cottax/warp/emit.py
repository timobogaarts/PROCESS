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


def build_kernel_source(
    leaves,
    unknowns: tuple,
    boundary: tuple,
    conditions: tuple,
    kernel_name: str = "sand_residual",
    array_vars: dict | None = None,
    const_arrays: frozenset | None = None,
) -> tuple[str, IdentifierMapper]:
    """`array_vars`: `{VarPath: n}` for every path in this kernel whose value is an
    ARRAY of `n` float64s -- the jaxpr backend's convention, where an array-valued read
    or owned value crosses a node boundary as one `vec{n}f`
    (`jaxpr_backend.emit_node`). A path here that is also a boundary input binds one
    extra `wp.array(dtype=wp.float64)` kernel parameter (`<ident>_buf`).

    `const_arrays`: the subset of `array_vars` (necessarily also boundary -- see
    `jaxpr_backend`'s constant-VarPath classifier) that is provably CONSTANT across the
    whole batch -- unreachable from any unknown. Those bind their `wp.array` kernel
    parameter and are left exactly that: ONE array, in global read-only memory, shared
    by every thread and indexed at the point of use inside whichever node needs an
    element (`jaxpr_backend.emit_node`'s matching `const_mask` gave those nodes a
    `wp.array` parameter too, not a `vec{n}f`, so the identifiers line up with no
    unpacking at all). A path here that is NOT constant is genuinely per-thread and
    still gets the old treatment: unpacked row-major into a `vec{n}f` LOCAL at the top
    of the kernel, which is what every consuming call expects.

    Both default to empty, which reduces this to the scalar-only kernel exactly.

    Returns `(source, mapper)` -- the mapper is returned so a caller can translate its
    own arrays' column order back to VarPaths."""
    array_vars = array_vars or {}
    const_arrays = const_arrays or frozenset()
    mapper = IdentifierMapper()
    lines: list[str] = []
    lines.append("    tid = wp.tid()")

    for i, path in enumerate(unknowns):
        if path in array_vars:
            raise EmitError(f"unknown {path!r} is array-valued -- a design variable is "
                            f"one float64 column by construction; refusing")
        lines.append(f"    {mapper.get(path)} = x[tid, {i}]")
    scalar_boundary = [p for p in boundary if p not in array_vars]
    for i, path in enumerate(scalar_boundary):
        lines.append(f"    {mapper.get(path)} = p[tid, {i}]")

    # CONSTANT array boundary: one wp.array kernel parameter, read directly at the
    # point of use inside each consuming node's own `@wp.func` -- no per-thread copy,
    # no vec{n}f local, nothing at all in this kernel body beyond the parameter itself.
    global_array_boundary = [p for p in boundary if p in array_vars and p in const_arrays]
    global_array_bufs = {p: mapper.get_buf(p) for p in global_array_boundary}

    # VARYING array boundary (unchanged): one `wp.array` parameter, unpacked row-major
    # into the `vec{n}f` local every consuming call expects. The unpack is `n` lines
    # once per kernel, not per use.
    array_var_boundary = [p for p in boundary
                          if p in array_vars and p not in const_arrays]
    array_var_bufs = {p: f"{mapper.get(p)}_buf" for p in array_var_boundary}
    for path in array_var_boundary:
        n = array_vars[path]
        ident = mapper.get(path)
        lines.append(f"    {ident} = vec{n}f()")
        for k in range(n):
            lines.append(f"    {ident}[{k}] = {array_var_bufs[path]}[{k}]")

    for leaf in leaves:
        deps = leaf.dependencies()
        missing = [p for p in deps if not mapper.known(p)]
        if missing:
            raise EmitError(
                f"node {leaf.node!r} ({leaf.fn}) needs {missing}, which no earlier "
                f"node/input produced -- not in topological order, or an "
                f"unknown/boundary declaration is missing"
            )
        arg_terms = [mapper.get(p) for p in leaf.inputs]
        out_idents = [mapper.get(p) for p in leaf.outputs]
        if not out_idents:
            raise EmitError(f"node {leaf.node!r} declares no outputs")
        lines.append(f"    {', '.join(out_idents)} = "
                     f"{leaf.fn}({', '.join(arg_terms)})")

    for i, path in enumerate(conditions):
        if not mapper.known(path):
            raise EmitError(
                f"condition {path!r} is never produced by any leaf or declared as an "
                f"unknown/boundary input"
            )
        if path in array_vars:
            raise EmitError(f"condition {path!r} is array-valued -- a residual entry "
                            f"is one float64 column; refusing")
        lines.append(f"    r[tid, {i}] = {mapper.get(path)}")

    params = ["x: wp.array2d(dtype=wp.float64)"]
    if scalar_boundary:
        params.append("p: wp.array2d(dtype=wp.float64)")
    for path in array_var_boundary:
        params.append(f"{array_var_bufs[path]}: wp.array(dtype=wp.float64)")
    for path in global_array_boundary:
        params.append(f"{global_array_bufs[path]}: wp.array(dtype=wp.float64)")
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
