"""The end-to-end harness for the jaxpr backend: build the maximal prefix-closed
sub-DAG of one config's SAND Drive out of `jaxpr_backend`'s emitted nodes, assemble it
with the existing `emit.build_kernel_source`, compile it as one Warp kernel, and compare
it against JAX evaluating the identical sub-DAG.

    python -m functional_process.cottax.warp.jaxpr_harness helias_5b

A node's `@wp.func` comes from tracing the node and emitting its jaxpr. There used to be
a resolver that AST-matched the node against "the function it really is" and transpiled
that function's source; it is gone, along with the concessions the emitter carried for
it (frozen statics, a prelude of wrapper locals, an `output_index` selecting one element
of a wider return). What reaches `emit.py` now is a node name, a function name, its
reads and its owns.

The whole-kernel agreement number here is a WEAKER check than
`jaxpr_validate.validate`, and is reported alongside it rather than instead of it: one
comparison over a whole sub-DAG certifies every function in it at once, so a node whose
contribution cancels or is swamped passes for free. `jaxpr_validate` checks each node
separately, over a sweep. This harness's job is to show the pieces COMPOSE.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from .emit import EmitError, build_kernel_source
from .jaxpr_backend import jaxpr_leaves, module_preamble, warp_init

AGREEMENT_RTOL = 1e-12
GEN_DIR = "functional_process/warp"
MAX_ROUNDS = 30


def prefix_closure(entries, unknowns, boundary, bad):
    """The maximal PREFIX-CLOSED subset in one topological pass: include a node only if
    everything it depends on is already available (an unknown, a boundary, or an
    earlier included node's output) and it is not in `bad`."""
    available = set(unknowns) | set(boundary)
    usable, blocked = [], []
    for e in entries:
        if e.fn not in bad and all(p in available for p in e.dependencies()):
            usable.append(e)
            available.update(e.outputs)
        else:
            blocked.append(e)
    return usable, blocked, available


def _failing_func(err_text: str) -> str | None:
    names = re.findall(r'Error while parsing function "([^"]+)"', err_text)
    return names[-1] if names else None


def eval_subdag_jax(entries, defs_by_node, env):
    """JAX over the IDENTICAL sub-DAG: each node's own `fn`, called on the values
    earlier nodes produced. `env` is `{path_str: value}` seeded with the unknowns and
    boundary. Returns the extended env.

    The node's `fn` called on its `reads` in order is the whole computation -- there is
    no separate reference evaluator to keep in step, which is itself a consequence of
    the resolver's removal: the old path needed one (`reference.py`) precisely because
    its emitter made concessions that had to be mirrored somewhere. An agreement check
    is only worth what the two sides' being the same computation is worth, and this is
    the shortest statement of "the same computation" available.
    """
    env = dict(env)
    for e in entries:
        args = [env[p] for p in e.inputs]
        res = defs_by_node[e.node].fn(*args)
        flat = jax.tree_util.tree_leaves(res)
        if len(flat) != len(e.outputs):
            raise RuntimeError(f"{e.node}: fn returned {len(flat)} values for "
                               f"{len(e.outputs)} outputs")
        for p, v in zip(e.outputs, flat):
            env[p] = v
    return env


def main(config: str):
    wp = warp_init()

    from functional_process.cottax import native as _native
    from functional_process.cottax.sand_harness import ground_truth as _gt
    from .assemble import _assemble

    drive, _, mda_env = _assemble(config)
    cold = _native.native_reference(
        f"tests/regression/input_files/{config}.IN.DAT").cold
    entries, refused, _ = jaxpr_leaves(config, drive=drive, cold=cold,
                                       mda_env=mda_env)
    n_nodes = len(entries) + len(refused)
    print(f"[jaxpr-harness] {config}: {n_nodes} drive nodes, {len(entries)} emitted "
          f"from their jaxpr, {len(refused)} refused")

    sub = drive.body.subgraph
    defs_by_node = {n.path_str(): sub[n] for n in sub.topological_order}

    unknowns = tuple(u.path_str() for u in drive.unknowns)
    boundary = tuple(v.path_str() for v in drive.context)
    probe = {}
    for var in drive.context:
        try:
            probe[var] = jnp.asarray(_gt(cold, var))
        except Exception:
            probe[var] = jnp.asarray(mda_env[var]) if var in mda_env \
                else jnp.asarray(0.0)
    conditions = tuple(c.path_str()
                       for c in drive.condition_map(probe).conditions)
    print(f"[jaxpr-harness] live shape: {len(unknowns)} unknowns, "
          f"{len(boundary)} boundary, {len(conditions)} conditions")

    # Real input values.
    # `native` keeps each value's OWN dtype and shape, which is what the JAX side must
    # be fed. An integer-typed read (an array index, a species selector) is exactly
    # representable in a float64 column and converted back inside the emitted function,
    # but handing float64 to JAX where an index is required is a TypeError -- and it is
    # a bug in the reference harness, not in the generated code.
    # `raw` is what the Warp SCALAR columns carry (float64 throughout -- the kernel's
    # ABI). An array-valued boundary path has no scalar column: it binds its own
    # `wp.array` parameter and is packed into a `vec{n}f` at the top of the kernel
    # (`emit.build_kernel_source`'s `array_vars`).
    # The value comes from PROCESS's own ground truth where there is one and from the
    # completed MDA run's env otherwise -- the profile grid and every other context
    # variable PROCESS never stores. `nan` remains the last resort, so an unsupplied
    # value is loud rather than plausible.
    raw, native, unsupplied = {}, {}, []
    for p, var in list(zip(unknowns, drive.unknowns)) + list(zip(boundary, drive.context)):
        try:
            v = np.asarray(_gt(cold, var))
        except Exception:
            v = np.asarray(mda_env[var]) if var in mda_env else np.asarray(np.nan)
            if var not in mda_env:
                unsupplied.append(p)
        native[p] = v
        raw[p] = float(v.reshape(())) if v.size == 1 else np.nan

    # Which VarPaths are array-valued, and how long -- taken from the emitted
    # signatures, which came from the trace, so this is the graph's own answer rather
    # than a guess. Two nodes disagreeing about one path's length is a hard stop: it
    # would mean one of them was traced at the wrong shape.
    array_vars: dict = {}
    for e in entries:
        for pth, n in e.array_sizes().items():
            if array_vars.setdefault(pth, n) != n:
                raise SystemExit(f"[jaxpr-harness] {pth!r} is length {array_vars[pth]} "
                                 f"in one node's signature and {n} in {e.node!r}")
    # A boundary array's real value must be exactly as long as the signature says.
    array_boundary_vals = {}
    for pth in [b for b in boundary if b in array_vars]:
        v = np.asarray(native[pth], dtype=float).reshape(-1)
        if v.size != array_vars[pth]:
            raise SystemExit(f"[jaxpr-harness] boundary {pth!r} has {v.size} real "
                             f"elements but the kernel binds {array_vars[pth]}")
        array_boundary_vals[pth] = v
    print(f"[jaxpr-harness] {len(array_vars)} array-valued VarPaths "
          f"({len(array_boundary_vals)} of them boundary inputs); "
          f"{len(unsupplied)} inputs with no value at all")

    bad: dict = {}
    kernel = None
    for round_no in range(MAX_ROUNDS):
        usable, blocked, available = prefix_closure(entries, unknowns, boundary, bad)
        # An included node must not read a value nothing could supply.
        for e in usable:
            for p in e.inputs:
                if p in unsupplied:
                    raise SystemExit(
                        f"[jaxpr-harness] node {e.node!r} reads {p!r}, for which "
                        f"neither PROCESS nor the MDA env has a value -- refusing "
                        f"to launch")
        reachable = tuple(c for c in conditions if c in available)

        live_arrays = {pth: n for pth, n in array_vars.items()
                       if any(pth in e.inputs or pth in e.outputs for e in usable)}
        func_src = "\n\n".join(dict.fromkeys(e.source for e in usable))
        try:
            kernel_src, mapper = build_kernel_source(
                usable, unknowns, boundary, reachable,
                kernel_name=f"{config}_jaxpr_subdag", array_vars=live_arrays)
        except EmitError as exc:
            print(f"[jaxpr-harness] round {round_no}: kernel assembly failed: {exc}")
            return
        module_src = ('"""GENERATED by functional_process/cottax/warp/jaxpr_backend '
                      '-- do not hand-edit."""\n' + module_preamble(usable) + "\n\n"
                      + func_src + "\n\n" + kernel_src)
        os.makedirs(GEN_DIR, exist_ok=True)
        path = f"{GEN_DIR}/_jaxpr_subdag_{config}.py"
        with open(path, "w") as f:
            f.write(module_src)

        scalar_boundary = [b for b in boundary if b not in live_arrays]
        live_array_boundary = [b for b in boundary if b in live_arrays]
        spec = importlib.util.spec_from_file_location(
            f"_jaxpr_subdag_{config}_{round_no}", path)
        gen = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(gen)
            k = getattr(gen, f"{config}_jaxpr_subdag")
            x = wp.array(np.ones((2, len(unknowns))), dtype=wp.float64, device="cpu")
            probe_inputs = [x]
            if scalar_boundary:
                probe_inputs.append(wp.array(np.ones((2, len(scalar_boundary))),
                                             dtype=wp.float64, device="cpu"))
            for b in live_array_boundary:
                probe_inputs.append(wp.array(np.ones(live_arrays[b]),
                                             dtype=wp.float64, device="cpu"))
            r = wp.zeros((2, len(reachable)), dtype=wp.float64, device="cpu")
            probe_inputs.append(r)
            wp.launch(k, dim=2, inputs=probe_inputs, device="cpu")
            wp.synchronize()
        except Exception as exc:
            failing = _failing_func(str(exc))
            if failing is None:
                print(f"[jaxpr-harness] round {round_no}: Warp codegen failed and the "
                      f"function is not named in the error -- not guessing which node "
                      f"to drop:\n{str(exc)[-600:]}")
                return
            print(f"[jaxpr-harness] round {round_no}: Warp codegen failed on "
                  f"{failing}; excluding and retrying")
            bad[failing] = str(exc)[:200]
            continue

        kernel = k
        print(f"[jaxpr-harness] round {round_no}: *** COMPILES *** "
              f"{len(usable)}/{n_nodes} entries, {len(reachable)}/{len(conditions)} "
              f"conditions, {len(bad)} Warp-codegen exclusions")
        # What the closure is actually waiting on. Coverage here is limited by which
        # nodes REFUSED, not by the emitter -- so name the first few blocked nodes and
        # the specific dependency each is missing, which is the critical path for the
        # next round of work rather than a number to stare at.
        blockers = []
        for e in blocked[:12]:
            miss = [p for p in e.dependencies() if p not in available]
            blockers.append((e.node, miss[:3]))
        print(f"[jaxpr-harness] {len(blocked)} entries blocked; first few and what "
              f"they are missing:")
        for nd, miss in blockers[:8]:
            print(f"    {nd:<58} missing {miss}")
        break
    else:
        print(f"[jaxpr-harness] gave up after {MAX_ROUNDS} rounds")
        return

    # ------------------------- agreement -------------------------
    env0 = {p: jnp.asarray(native[p]) for p in list(unknowns) + list(boundary)}
    jax_env = eval_subdag_jax(usable, defs_by_node, env0)
    jax_vals = np.array([float(np.asarray(jax_env[c]).reshape(())) for c in reachable])

    x = wp.array(np.array([[raw[p] for p in unknowns]]), dtype=wp.float64, device="cpu")
    launch_inputs = [x]
    if scalar_boundary:
        launch_inputs.append(wp.array(np.array([[raw[p] for p in scalar_boundary]]),
                                      dtype=wp.float64, device="cpu"))
    for b in live_array_boundary:
        launch_inputs.append(wp.array(array_boundary_vals[b], dtype=wp.float64,
                                      device="cpu"))
    r = wp.zeros((1, len(reachable)), dtype=wp.float64, device="cpu")
    launch_inputs.append(r)
    wp.launch(kernel, dim=1, inputs=launch_inputs, device="cpu")
    wp.synchronize()
    warp_vals = r.numpy()[0]

    rel = np.abs(warp_vals - jax_vals) / np.maximum(np.abs(jax_vals), 1e-300)
    print(f"\n[jaxpr-harness] === agreement, {config}, {len(reachable)} conditions ===")
    for c, w, j, rl in zip(reachable, warp_vals, jax_vals, rel):
        print(f"    {c:55s} warp={w: .8e}  jax={j: .8e}  rel={rl:.3e}")
    finite = np.isfinite(rel)
    worst = float(rel[finite].max()) if finite.any() else float("nan")
    verdict = "PASS" if finite.any() and worst <= AGREEMENT_RTOL else "FAIL"
    print(f"[jaxpr-harness] *** worst relative difference (finite only): {worst:.3e} "
          f"-- {verdict} against AGREEMENT_RTOL = {AGREEMENT_RTOL:.0e} ***")

    result = {
        "config": config, "n_nodes": n_nodes, "n_emitted": len(entries),
        "n_entries_covered": len(usable), "n_conditions_total": len(conditions),
        "n_conditions_covered": len(reachable),
        "conditions_covered": list(reachable),
        "worst_rel_diff_finite": worst, "agreement": verdict,
        "agreement_rtol": AGREEMENT_RTOL,
        "refused": [{"node": n, "why": w} for n, w in refused],
        "warp_codegen_excluded": bad,
        "total_eqns_covered": sum(e.n_eqns for e in usable),
        "array_vars": dict(sorted(live_arrays.items())),
        "array_boundary_inputs": list(live_array_boundary),
    }
    with open(f"{GEN_DIR}/jaxpr_subdag_{config}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[jaxpr-harness] wrote jaxpr_subdag_{config}.json "
          f"({result['total_eqns_covered']} jaxpr equations in the covered kernel)")
    return result


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "helias_5b")
