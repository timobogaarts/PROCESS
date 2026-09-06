"""Evaluate the IDENTICAL sub-DAG the Warp kernel computes, in JAX -- so the agreement
check is genuinely like-for-like.

Not a slice of the full Drive's `condition_map` (that would pull in nodes outside the
emitted subset): literally the same `entries` list, the same underlying functions, the
same prelude calls, the same tuple-element selections, the same built bodies. Whatever
the emitter does to a node, this does too. An agreement check can only ever say the two
engines agree with each other, so the value of it rests entirely on the two sides being
the same computation -- which is why every emitter concession is mirrored here rather
than approximated.

Also provides `real_values_for(config)`: the full env of real ground-truth-seeded
values for every VarPath the whole Drive's topologically-resolved entries produce,
which is what the array-parameter shape decision needs for leaves whose parameters are
intermediate values rather than raw boundary/unknown ones.
"""
from __future__ import annotations

import importlib


import jax.numpy as jnp
import numpy as np

from .combined import combined_ordered
from .leaf_funcs_arrays import SYNTH_LEAVES as _SYNTH


_SNAP_INT_MAX = 2 ** 31
"""`snap_int` exists to keep a switch-like argument (`i_tf_sc_mat = 3.0`) an `int` so
a `[...]`-lookup body sees an index, not a float. A *physical* whole-valued quantity
is not that: `.physics.nd_plasma_electrons_vol_avg` is `7.5e19`, exactly integral as a
float64, and `int(...)` of it is a 67-bit Python int that JAX refuses outright
(`OverflowError ... whose argument path is x`). Bound the snap to values a real switch
could take."""


def _is_whole_scalar(v) -> bool:
    arr = np.asarray(v)
    if arr.ndim != 0:
        return False
    f = float(arr)
    return f.is_integer() and abs(f) < _SNAP_INT_MAX


def _call_kwargs(node, order, inputs, statics, locals_, env) -> dict:
    """One call's arguments as keyword arguments, per `leaves.py`'s ordering contract:
    a name in `statics` is its frozen literal, one in `locals_` is a value an earlier
    `PreludeCall` put in `env` under `f"{node}::{ident}"`, and anything else consumes
    the next `inputs` entry in sequence.

    By NAME, not position: several real leaf functions declare keyword-only parameters
    (`def f(*, a, b, c)`, e.g. `user_input_electron_cyclotron_efficiency`) -- a
    positional `fn(*args)` call raises `TypeError: takes 0 positional arguments` for
    those. Warp's own codegen does not enforce Python's positional/keyword-only
    distinction (its kernel compiled these same calls positionally without complaint),
    so this is a JAX-reference-side-only fix, matching `order`'s names to the real
    function's actual parameter names exactly.

    A SEQUENCE static is passed through as the tuple it is: the Warp side drops it from
    the call because `leaf_funcs._SubstituteSequenceStatics` baked its literals into the
    `@wp.func`, but the real Python function still takes it as an argument, and passing
    it is what makes the two sides the same computation.
    """
    statics_map = dict(statics)
    locals_map = dict(locals_)
    input_iter = iter(inputs)
    kwargs = {}
    for name in order:
        if name in statics_map:
            kwargs[name] = statics_map[name]
        elif name in locals_map:
            kwargs[name] = env[f"{node}::{locals_map[name]}"]
        else:
            kwargs[name] = env[next(input_iter)]
    return kwargs


def eval_subdag_full(entries, raw: dict, snap_int: bool = True) -> dict:
    """`raw`: VarPath -> python/JAX scalar (or, under `jax.vmap`, tracer) for every
    unknown+boundary. Returns VarPath -> value for every VarPath the sub-DAG produces
    (inputs included), plus one `f"{node}::{ident}"` key per prelude local.

    `snap_int=True` (the default, for a single concrete point) leaves a whole-number
    value as a plain Python `int` rather than `jnp.asarray` (float64): PROCESS's own
    source indexes a literal table directly with switch-typed values (`cmlsa[lsa - 1]`),
    and JAX's static indexing refuses a float-dtype index even when its value is whole.
    Warp's kernel tolerates this fine (the list-lookup rewrite compares as float64), so
    it is purely a JAX-side reference-evaluation concession. Pass `snap_int=False` under
    `jax.vmap`/`jax.jit`, where `raw`'s values are tracers and `float(v)`/`int(v)` would
    raise `ConcretizationTypeError`.

    Array-safe: `snap_int` only ever inspects a SCALAR raw value (`_is_whole_scalar`
    checks `.ndim == 0` first), so an array-valued boundary entry passes through as
    `jnp.asarray(v)` unchanged, never routed through `float(v)`.
    """
    if snap_int:
        env = {p: (int(v) if _is_whole_scalar(v) else jnp.asarray(v)) for p, v in raw.items()}
    else:
        env = {p: jnp.asarray(v) for p, v in raw.items()}
    for e in entries:
        is_structural = type(e).__name__ == "StructuralOp"
        if is_structural:
            if e.op == "neg":
                env[e.outputs[0]] = -env[e.inputs[0]]
            elif e.op == "sub":
                for i, out in enumerate(e.outputs):
                    env[out] = env[e.inputs[2 * i]] - env[e.inputs[2 * i + 1]]
            else:
                raise ValueError(f"unhandled structural op {e.op!r}")
            continue
        # Prelude (a "Composition" wrapper -- see `resolve.PreludeCall`): evaluate the
        # locals the wrapper's own body computes, into the SAME namespaced keys the
        # emitted Warp kernel uses for them, before the leaf call that reads them. Both
        # engines therefore evaluate the identical composition, not a leaf call with a
        # local guessed at from somewhere else.
        for pc in getattr(e, "prelude", ()):
            pmod = importlib.import_module(pc.module)
            pfn = getattr(pmod, pc.fn)
            pkwargs = _call_kwargs(e.node, pc.order, pc.inputs, pc.statics, pc.locals_, env)
            presult = pfn(**pkwargs)
            if len(pc.targets) == 1:
                env[f"{e.node}::{pc.targets[0]}"] = presult
            else:
                if len(presult) != len(pc.targets):
                    raise ValueError(
                        f"{e.node}: prelude `{pc.source}` unpacks {len(pc.targets)} "
                        f"value(s) from {pc.fn}, which returned {len(presult)}"
                    )
                for t, v in zip(pc.targets, presult):
                    env[f"{e.node}::{t}"] = v

        built = _SYNTH.get((e.module, e.fn))
        if built is not None:
            # A BUILT leaf (`resolve._resolve_built_composition`): no `getattr` will
            # find it, because it has no source of its own -- it is the node's own
            # body with its fixed-length arrays expanded into named scalars. The JAX
            # reference evaluates that same expanded function, so Warp and JAX are
            # compared over the identical sub-DAG.
            fn = built.callable
        else:
            mod = importlib.import_module(e.module)
            fn = getattr(mod, e.fn)
        if getattr(e, "order", ()):
            # By NAME, not position -- see `_call_kwargs`.
            kwargs = _call_kwargs(
                e.node, e.order, e.inputs, e.statics, getattr(e, "locals_", ()), env
            )
            result = fn(**kwargs)
        else:
            args = [env[p] for p in e.inputs]
            result = fn(*args)
        output_index = getattr(e, "output_index", None)
        if output_index is not None:
            # `result` is WIDER than `e.outputs` (a `.fn`-field wrapper selecting one
            # element of a 4-tuple -- a constraint node forwarding to `eq`/`leq`/`geq`;
            # see `resolve._subscript_select_index`). Applying the SAME selection the
            # emitted kernel applies is what keeps the two sides the same quantity;
            # without it this side stores the whole tuple under the node's single
            # output and the comparison is against a tuple, not a number.
            for out, sel in zip(e.outputs, output_index):
                env[out] = result[sel]
        elif len(e.outputs) == 1:
            env[e.outputs[0]] = result
        else:
            for out, val in zip(e.outputs, result):
                env[out] = val
    return env


def real_values_for(config: str):
    """`(entries, drive, env)` -- `env` is a best-effort full evaluation of `config`'s
    Drive, VarPath -> real NumPy/JAX value, seeded from PROCESS's own cold-start
    ground truth at every unknown/boundary and propagated through every entry that
    evaluates cleanly. An entry whose real inputs are not yet in `env` (should not
    happen -- `combined_ordered`'s `entries` is already topologically ordered) or
    whose function raises is SKIPPED (reported, not silently absorbed) -- its outputs
    then have no entry in `env`, so any leaf depending on them will correctly report
    "no real ground-truth value" rather than a wrong one.
    """
    from functional_process.cottax import native as _native
    from functional_process.cottax.sand_harness import ground_truth as _ground_truth

    entries, unresolved, drive = combined_ordered(config)
    ref = _native.native_reference(f"tests/regression/input_files/{config}.IN.DAT")
    cold = ref.cold

    raw = {}
    for var in list(drive.unknowns) + list(drive.context):
        p = var.path_str()
        try:
            raw[p] = _ground_truth(cold, var)
        except Exception:
            pass  # left out of `raw` -- unresolved downstream, not guessed at

    env = dict(raw)
    skipped = []
    for e in entries:
        is_structural = type(e).__name__ == "StructuralOp"
        if not is_structural and not all(p in env for p in e.inputs):
            skipped.append((e.node, "missing input(s) " + str([p for p in e.inputs if p not in env])))
            continue
        try:
            one_env = eval_subdag_full([e], env, snap_int=True)
        except Exception as exc:
            skipped.append((e.node, f"{type(exc).__name__}: {exc}"))
            continue
        env.update(one_env)
    return entries, drive, env, skipped
