"""The end-to-end harness for the jaxpr backend: build the maximal prefix-closed
sub-DAG of one config's SAND Drive out of `jaxpr_backend`'s emitted nodes, assemble it
with the existing `emit.build_kernel_source`, compile it as one Warp kernel, and compare
it against JAX evaluating the identical sub-DAG.

    python -m functional_process.cottax.warp.jaxpr_harness helias_5b

The emitter is unchanged -- it was always the part that was right. What changed beneath
it is where a node's `@wp.func` comes from: `jaxpr_backend` traces the node and emits its
jaxpr, where `resolve.py` used to AST-match the node against "the function it really
is". The `JaxprLeaf` entries this feeds in are deliberately the shape `emit.py` already
consumed (empty `order` -> positional binding; empty `statics`; `output_index=None`).

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
from .jaxpr_backend import jaxpr_leaves

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

    Deliberately NOT `reference.py`: that module mirrors the resolver's concessions
    (statics, preludes, output_index), and this path has none of them -- the node's
    `fn` called on its `reads` in order is the whole computation. An agreement check is
    only worth what the two sides' being the same computation is worth, and this is the
    shortest statement of "the same computation" available.
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
    import warp as wp

    from functional_process.cottax import native as _native
    from functional_process.cottax.sand_harness import ground_truth as _gt
    from .assemble import _assemble

    drive, _ = _assemble(config)
    cold = _native.native_reference(
        f"tests/regression/input_files/{config}.IN.DAT").cold
    entries, refused, _ = jaxpr_leaves(config, drive=drive, cold=cold)
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
            probe[var] = jnp.asarray(0.0)
    conditions = tuple(c.path_str()
                       for c in drive.condition_map(probe).conditions)
    print(f"[jaxpr-harness] live shape: {len(unknowns)} unknowns, "
          f"{len(boundary)} boundary, {len(conditions)} conditions")

    # Real input values. Every read an emitted node has is scalar by construction
    # (`jaxpr_backend` refuses an array-shaped parameter), so every unknown/boundary
    # column an included node actually reads carries a scalar. An array-valued
    # boundary path nothing reads still needs SOME number in its column; it is given
    # 0.0 and named below, so no unread array is silently flattened into a value a
    # node might later depend on.
    # `raw` is what the Warp columns carry (float64 throughout -- the kernel's ABI);
    # `native` keeps each value's OWN dtype, which is what the JAX side must be fed.
    # An integer-typed read (an array index, a species selector) is exactly
    # representable in a float64 column and converted back inside the emitted
    # function, but handing float64 to JAX where an index is required is a TypeError
    # -- and it is a bug in the reference harness, not in the generated code.
    raw, native, array_boundary_unread = {}, {}, []
    for p, var in list(zip(unknowns, drive.unknowns)) + list(zip(boundary, drive.context)):
        try:
            v = np.asarray(_gt(cold, var))
        except Exception:
            v = np.asarray(np.nan)
        native[p] = v
        if v.size != 1:
            array_boundary_unread.append(p)
            raw[p] = 0.0
        else:
            raw[p] = float(v.reshape(()))

    wp.init()
    bad: dict = {}
    kernel = None
    for round_no in range(MAX_ROUNDS):
        usable, blocked, available = prefix_closure(entries, unknowns, boundary, bad)
        # An included node must not read an array-valued boundary path -- it cannot,
        # since it would have refused, but assert it rather than assume it.
        for e in usable:
            for p in e.inputs:
                if p in array_boundary_unread:
                    raise SystemExit(
                        f"[jaxpr-harness] node {e.node!r} reads array-valued boundary "
                        f"{p!r} through a scalar column -- refusing to launch")
        reachable = tuple(c for c in conditions if c in available)

        func_src = "\n\n".join(dict.fromkeys(e.source for e in usable))
        try:
            kernel_src, mapper = build_kernel_source(
                usable, unknowns, boundary, reachable,
                kernel_name=f"{config}_jaxpr_subdag")
        except EmitError as exc:
            print(f"[jaxpr-harness] round {round_no}: kernel assembly failed: {exc}")
            return
        module_src = ('"""GENERATED by functional_process/cottax/warp/jaxpr_backend '
                      '-- do not hand-edit."""\nimport warp as wp\n\n'
                      + func_src + "\n\n" + kernel_src)
        os.makedirs(GEN_DIR, exist_ok=True)
        path = f"{GEN_DIR}/_jaxpr_subdag_{config}.py"
        with open(path, "w") as f:
            f.write(module_src)

        spec = importlib.util.spec_from_file_location(
            f"_jaxpr_subdag_{config}_{round_no}", path)
        gen = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(gen)
            k = getattr(gen, f"{config}_jaxpr_subdag")
            x = wp.array(np.ones((2, len(unknowns))), dtype=wp.float64, device="cpu")
            p_arr = wp.array(np.ones((2, len(boundary))), dtype=wp.float64, device="cpu")
            r = wp.zeros((2, len(reachable)), dtype=wp.float64, device="cpu")
            wp.launch(k, dim=2, inputs=[x, p_arr, r], device="cpu")
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
    p_arr = wp.array(np.array([[raw[p] for p in boundary]]), dtype=wp.float64,
                     device="cpu")
    r = wp.zeros((1, len(reachable)), dtype=wp.float64, device="cpu")
    wp.launch(kernel, dim=1, inputs=[x, p_arr, r], device="cpu")
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
    }
    with open(f"{GEN_DIR}/jaxpr_subdag_{config}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[jaxpr-harness] wrote jaxpr_subdag_{config}.json "
          f"({result['total_eqns_covered']} jaxpr equations in the covered kernel)")
    return result


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "helias_5b")
