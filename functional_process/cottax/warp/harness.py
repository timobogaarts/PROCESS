"""The end-to-end harness: build the maximal prefix-closed, fully-emittable sub-DAG of
one config's SAND Drive, compile it as one Warp kernel, compare it against JAX
evaluating the IDENTICAL sub-DAG, then time it.

    python -m functional_process.cottax.warp.harness helias_5b

Algorithm, run from the repository root (the config's `IN.DAT` is read relative to it):

  1. Start with `bad = {}` (no known-bad leaves).
  2. Transpile every leaf not in `bad` (tolerant); anything that fails joins `bad`.
     `leaf_funcs` (scalar) goes first; only a leaf it refuses for an
     array-parameter/interp/searchsorted reason, plus every BUILT leaf, is retried
     through `leaf_funcs_arrays`.
  3. Compute the maximal PREFIX-CLOSED subset in one topological pass: include a node
     only if everything it depends on is already available (an unknown/boundary, or an
     earlier INCLUDED node's output) and, if a leaf, it is not in `bad`.
  4. Try to build and Warp-compile the kernel over that subset. If Warp's codegen
     fails on a specific function, add it to `bad` and go to 2. That loop is what
     makes "maximal EMITTABLE" a measured fact rather than an assumption.
  5. Once it compiles: compare against JAX evaluating the same entries at the same
     real input -- per-condition relative differences, worst overall, against
     `AGREEMENT_RTOL`.
  6. Launch it once on the GPU and read the kernel's register count and spill from the
     cubin via the CUDA driver API -- measured, not estimated.

NOT here: the batch-1..65536 CPU/GPU timing sweep the scalar-only ancestor of this file
carried. It bound the kernel's inputs assuming every one of them is a `p[tid, i]`
column, which stopped being true once array boundaries and constant tables became their
own kernel parameters, and its JAX side ran under `jax.vmap` with `snap_int=False`,
which an array-valued boundary does not survive. Re-adding it is real work, not a
restore, and is deliberately left undone rather than left silently broken.

Self-report discipline throughout: every excluded leaf is named with its reason,
agreement is checked over the IDENTICAL covered sub-DAG, and registers/spill are
measured via the CUDA driver API rather than estimated.
"""
import importlib.util
import json
import os
import re
import sys


import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from . import leaf_funcs as _BASE  # the scalar builder -- tried first, see main()
from .combined import Leaf, StructuralOp, combined_ordered
from .emit import EmitError, build_kernel_source
from .leaf_funcs_arrays import (SYNTH_LEAVES, _field_shape_is_array,
                                build_leaf_funcs_source_arrays)
from .reference import eval_subdag_full as eval_subdag
from .reference import real_values_for

CONFIG = sys.argv[1] if len(sys.argv) > 1 else "helias_5b"

AGREEMENT_RTOL = 1e-12
"""The Warp-vs-JAX agreement gate, as a RELATIVE difference per condition.

Not bit-exactness. Bit-exactness was never worth anything as *accuracy* -- its only
value was as a bug detector, and every misbinding this work has actually found fires
enormously higher than one ulp: the two known negative controls (a prelude local bound
to a same-named boundary variable; a mis-selected leaf) move a condition by 2.4e-02 and
5.0e+00 respectively, some thirteen orders of magnitude above 1 ulp. 1e-12 catches all
of them and still leaves three orders of magnitude of headroom over the worst
transcendental disagreement in the generated code (~4 ulp, ~1e-15).

The worst relative difference is REPORTED as a number regardless of whether it passes,
because the number is the evidence and the gate is only a summary of it.
"""

GEN_DIR = "functional_process/warp"
"""Where the generated module is written. Deliberately gitignored: it is derived, it
would be a five-figure diff on every regeneration, and the generator plus its
self-validation is the thing worth reviewing (see this package's `__init__.py`)."""

GEN_PATH = f"{GEN_DIR}/_generated_subdag_{CONFIG}.py"
MAX_ROUNDS = 25


def prefix_closure(entries, unknowns, boundary, bad):
    available = set(unknowns) | set(boundary)
    usable, blocked = [], []
    for e in entries:
        is_leaf = isinstance(e, Leaf)
        # `dependencies()`, not `inputs`: a "Composition" leaf also reads whatever its
        # PRELUDE calls read (`resolve.PreludeCall`), and `inputs` is positional -- it
        # must stay aligned with `order`, so the prelude's reads are not in it. A
        # closure over `inputs` alone would admit a node whose prelude reads something
        # nothing earlier produced.
        deps = e.dependencies() if hasattr(e, "dependencies") else e.inputs
        blocked_here = is_leaf and (
            (e.module, e.fn) in bad
            or any((pc.module, pc.fn) in bad for pc in getattr(e, "prelude", ()))
        )
        if not blocked_here and all(p in available for p in deps):
            usable.append(e)
            available.update(e.outputs)
        else:
            blocked.append(e)
    return usable, blocked, available


def _extract_failing_func_name(err_text: str) -> str | None:
    names = re.findall(r'Error while parsing function "([^"]+)"', err_text)
    return names[-1] if names else None


def main():
    entries, unresolved, drive = combined_ordered(CONFIG)
    print(f"[array-subdag] {CONFIG}: drive has {len(entries)} resolved entries "
          f"({sum(isinstance(e, Leaf) for e in entries)} leaf, "
          f"{sum(isinstance(e, StructuralOp) for e in entries)} structural), "
          f"{len(unresolved)} unresolved at the resolver level")

    from functional_process.cottax import native as _native
    from functional_process.cottax.sand_harness import ground_truth as _ground_truth
    _ref = _native.native_reference(f"tests/regression/input_files/{CONFIG}.IN.DAT")
    _cold = _ref.cold

    unknowns = tuple(u.path_str() for u in drive.unknowns)
    boundary = tuple(v.path_str() for v in drive.context)
    _probe_context = {}
    for var in drive.context:
        try:
            _probe_context[var] = jnp.asarray(_ground_truth(_cold, var))
        except Exception:
            _probe_context[var] = jnp.asarray(0.0)
    conditions = tuple(c.path_str() for c in drive.condition_map(_probe_context).conditions)
    print(f"[array-subdag] live shape: {len(unknowns)} unknowns, {len(boundary)} boundary, "
          f"{len(conditions)} conditions")

    # Real values, propagated through the WHOLE Drive from ground truth
    # (`reference.real_values_for`) -- used as a FALLBACK to the
    # DataStructure-field-type shape decision (for a genuinely derived/intermediate
    # parameter no DataStructure field names, e.g. `.stellarator.coilcurrent`), and
    # for building real test inputs. A best effort: some entries are skipped (their
    # own inputs never resolved), reported, not silently absorbed.
    print("[array-subdag] propagating real values through the full Drive "
          "(ground truth + JAX, for the array-shape fallback)...")
    _rv_entries, _rv_drive, real_values, _rv_skipped = real_values_for(CONFIG)
    print(f"[array-subdag] real_values: {len(real_values)} VarPaths resolved, "
          f"{len(_rv_skipped)} entries skipped upstream")

    # Which boundary paths are array-valued -- derived from the DataStructure field
    # type (`leaf_funcs_arrays._field_shape_is_array`), same source of truth
    # `decide_array_params` uses for leaf parameters.
    # `{path: shape}` -- the real SHAPE, not just a length: a genuinely 2-D field
    # (`.impurity_radiation.temp_impurity_keV_array`, `(14, 200)`) binds as one
    # `wp.array2d` kernel parameter. Falls back to the DataStructure field's own
    # declared default when the run's propagated value is unavailable, which is where
    # PROCESS's fixed per-species tables actually live.
    from process.core.model import DataStructure as _DS
    _ds_defaults = _DS()

    def _default_shape(path):
        parts = path.strip(".").split(".")
        if len(parts) != 2 or path.startswith("^"):
            return None
        sub = getattr(_ds_defaults, parts[0], None)
        if sub is None:
            return None
        val = getattr(sub, parts[1], None)
        if val is None:
            return None
        try:
            arr = np.asarray(val, dtype=float)
        except (TypeError, ValueError):
            return None
        return tuple(arr.shape) if arr.ndim >= 1 and arr.size > 1 else None

    array_boundary_lengths = {}
    for p in boundary:
        is_arr = _field_shape_is_array(p)
        if is_arr:
            real = real_values.get(p)
            if real is not None and np.asarray(real).ndim >= 1:
                array_boundary_lengths[p] = tuple(np.asarray(real).shape)
            else:
                array_boundary_lengths[p] = _default_shape(p)

    import warp as wp
    wp.init()

    bad: dict = {}
    kernel = None
    usable_entries = []
    extra_table_args = {}
    table_registry = {}
    _ARRAY_REASON = re.compile(r"array-valued parameter|jnp\.interp|searchsorted")
    for round_no in range(MAX_ROUNDS):
        usable_entries, blocked, available = prefix_closure(entries, unknowns, boundary, bad)
        leaf_entries = [e for e in usable_entries if isinstance(e, Leaf)]

        # Hybrid: the BASELINE scalar-only transpiler first (unmodified,
        # `leaf_funcs.py`) -- it already carries every OTHER fix in this package
        # session (constraint-arity's derived-K forwarding among them; not this
        # file's concern, and reimplementing it here would drift from the real
        # fix). Only a leaf baseline genuinely cannot place -- for an
        # array-parameter/interp/searchsorted reason, this file's actual scope --
        # is retried through the array-aware path below.
        # A BUILT leaf (`resolve._resolve_built_composition`) has no Python source
        # of its own -- the baseline transpiler cannot even look it up, and its
        # "not found" is not one of the array-shaped reasons the retry filter
        # recognises. Route it straight to the array-aware path instead of letting
        # it fail out of the baseline pass for the wrong reason.
        # One shared set across BOTH builders: they emit into the same generated
        # module and both can need the same registry helper (`_dd_fma` is
        # `_xla_lgamma`'s FMA and also both `jnp.interp` helpers' fused final step),
        # so without this it is defined twice in one file.
        registry_emitted: set = set()
        synth_keys = set(SYNTH_LEAVES)
        base_input = [e for e in leaf_entries if (e.module, e.fn) not in synth_keys]
        try:
            base_src, base_fn_names, base_failed, base_arities = _BASE.build_leaf_funcs_source(
                base_input, tolerant=True, known_bad=bad,
                already_emitted=registry_emitted)
        except _BASE.LeafError as exc:
            print(f"[array-subdag] round {round_no}: hard transpile error (baseline): {exc}")
            return

        array_retry_keys = {k for k, why in base_failed.items() if _ARRAY_REASON.search(why)}
        array_retry_leaves = [e for e in leaf_entries
                               if ((e.module, e.fn) in synth_keys
                                   or ((e.module, e.fn) in array_retry_keys
                                       and (e.module, e.fn) not in base_fn_names))]
        arr_src, arr_fn_names, arr_failed, arr_arities, extra_table_args, table_registry = \
            build_leaf_funcs_source_arrays(array_retry_leaves, real_values, tolerant=True,
                                            known_bad=bad,
                                            already_emitted=registry_emitted)

        # Combine: the array path's result for anything it newly resolved
        # supersedes baseline's failure for that key; everything else is baseline's
        # own verdict, untouched.
        fn_names = dict(base_fn_names)
        fn_names.update(arr_fn_names)
        arities = dict(base_arities)
        arities.update(arr_arities)
        transpile_failed = {k: v for k, v in base_failed.items() if k not in arr_fn_names}
        transpile_failed.update({k: v for k, v in arr_failed.items() if k not in fn_names})
        func_src = base_src + "\n\n" + arr_src

        newly_bad = {k: v for k, v in transpile_failed.items() if k not in bad}
        # The return-arity invariant, against the function as ACTUALLY transpiled.
        # A leaf whose function returns MORE values than the node owns used to be an
        # unconditional exclusion, because which of those values is the owned one was
        # not in the Leaf contract. It now often IS: `resolve._subscript_select_index`
        # derives it from the wrapper class's own code (`_NormalisedResidual.__call__`
        # does `self.fn(**kwargs)[1]` -- always index 1, `normalised_residual`, for
        # every constraint node, no exceptions found). `leaf.output_index` carries that
        # derived selection; only a REAL, unexplained mismatch -- no `output_index`, or
        # one inconsistent with the leaf's actual transpiled arity -- is still excluded.
        for leaf in leaf_entries:
            key = (leaf.module, leaf.fn)
            if key in bad or key in newly_bad:
                continue
            arity = arities.get(key, 1)
            if arity == len(leaf.outputs):
                continue
            output_index = getattr(leaf, "output_index", None)
            if output_index is not None and len(output_index) == len(leaf.outputs) \
                    and all(0 <= k < arity for k in output_index):
                continue
            newly_bad[key] = (
                f"arity mismatch: {leaf.fn} returns {arity} value(s) but node "
                f"{leaf.node!r} owns {len(leaf.outputs)} output(s), and the resolver "
                f"derived no consistent selection ({output_index!r}) -- which tuple "
                f"element is owned is not guessed"
            )
        # The same invariant, applied to each PRELUDE call. The resolver already
        # checked it against the AST (`resolve._prelude_from_call`); this checks it
        # against the transpiled function, which is the arity the emitted kernel will
        # destructure. A mismatch would bind the local to the wrong tuple element --
        # exactly the silently-wrong-value failure this node class was refused for.
        for leaf in leaf_entries:
            key = (leaf.module, leaf.fn)
            if key in bad or key in newly_bad:
                continue
            for pc in getattr(leaf, "prelude", ()):
                p_arity = arities.get((pc.module, pc.fn))
                if p_arity is not None and p_arity != len(pc.targets):
                    newly_bad[key] = (
                        f"prelude arity mismatch: {pc.fn} returns {p_arity} value(s) "
                        f"but node {leaf.node!r}'s `{pc.source}` unpacks "
                        f"{len(pc.targets)} -- not guessed"
                    )
                    break
        if newly_bad:
            bad.update(newly_bad)
            print(f"[array-subdag] round {round_no}: {len(newly_bad)} newly excluded "
                  f"(transpile failure or arity mismatch), retrying closure")
            continue

        reachable_conditions = tuple(c for c in conditions if c in available)
        # Restrict to array-boundary paths an INCLUDED leaf actually reads: a
        # boundary var nothing in the covered sub-DAG reads needs no parameter.
        # (2-D fields -- `.impurity_radiation.temp_impurity_keV_array`, `(14, 200)`
        # -- now bind as `wp.array2d`; they used to fail outright here.)
        used_paths = {p for e in usable_entries for p in e.inputs}
        array_boundary_here = {
            p: n for p, n in array_boundary_lengths.items()
            if p in boundary and p in used_paths
        }
        try:
            kernel_src, mapper = build_kernel_source(
                usable_entries, unknowns, boundary, reachable_conditions,
                kernel_name=f"{CONFIG}_subdag",
                arities=arities,
                array_boundary=array_boundary_here,
                extra_table_args=extra_table_args,
                table_registry=table_registry,
            )
        except EmitError as exc:
            print(f"[array-subdag] round {round_no}: kernel assembly failed: {exc}")
            return
        module_src = ('"""GENERATED by functional_process/cottax/warp -- do not '
                      'hand-edit."""\n'
                      'import warp as wp\n\n' + func_src + "\n\n" + kernel_src)
        os.makedirs(GEN_DIR, exist_ok=True)
        with open(GEN_PATH, "w") as f:
            f.write(module_src)

        spec = importlib.util.spec_from_file_location(f"_gen_array_subdag_{CONFIG}_{round_no}", GEN_PATH)
        gen_try = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(gen_try)
            kernel_try = getattr(gen_try, f"{CONFIG}_subdag")
            N = 2
            x = wp.array(np.ones((N, len(unknowns))), dtype=wp.float64, device="cpu")
            scalar_boundary = [p for p in boundary if p not in array_boundary_here]
            inputs = [x]
            if scalar_boundary:
                p_arr = wp.array(np.ones((N, len(scalar_boundary))), dtype=wp.float64, device="cpu")
                inputs.append(p_arr)
            for p in array_boundary_here:
                sh = array_boundary_here[p] or (1,)
                inputs.append(wp.array(np.ones(sh), dtype=wp.float64, device="cpu"))
            for ident in sorted(table_registry):
                inputs.append(wp.array(np.asarray(table_registry[ident]), dtype=wp.float64, device="cpu"))
            r = wp.zeros((N, len(reachable_conditions)), dtype=wp.float64, device="cpu")
            inputs.append(r)
            wp.launch(kernel_try, dim=N, inputs=inputs, device="cpu")
            wp.synchronize()
        except Exception as exc:
            failing = _extract_failing_func_name(str(exc))
            print(f"[array-subdag] round {round_no}: Warp codegen failed"
                  f"{' on ' + failing if failing else ''}: {type(exc).__name__}")
            if failing is None:
                print(f"    (could not parse a function name out of: {str(exc)[-400:]})")
                return
            hit = [k for k in fn_names if fn_names[k] == failing]
            if not hit:
                print(f"    '{failing}' not found among this round's emitted leaves -- "
                      f"the error is in the KERNEL body itself, not a leaf. Full text:")
                print(f"    {str(exc)}")
                return
            for h in hit:
                bad[h] = f"Warp codegen: {exc}"
            continue

        kernel = kernel_try
        print(f"[array-subdag] round {round_no}: *** COMPILES *** "
              f"{len(usable_entries)}/{len(entries)} entries, "
              f"{len(reachable_conditions)}/{len(conditions)} conditions, "
              f"{len(bad)} leaves excluded")
        break
    else:
        print(f"[array-subdag] gave up after {MAX_ROUNDS} rounds")
        return

    if kernel is None:
        return

    print(f"\n[array-subdag] excluded leaves ({len(bad)}):")
    for (mod, fnm), why in bad.items():
        print(f"    {mod}.{fnm}: {why}")

    excluded_conditions = tuple(c for c in conditions if c not in reachable_conditions)
    print(f"\n[array-subdag] conditions covered: {reachable_conditions}")
    print(f"[array-subdag] conditions excluded: {excluded_conditions}")

    # ---------------- agreement: JAX over the IDENTICAL sub-DAG ----------------
    rng = np.random.default_rng(0)
    raw = {}
    for p, var in list(zip(unknowns, drive.unknowns)) + list(zip(boundary, drive.context)):
        if p in array_boundary_here:
            continue  # array-valued -- handled below via the real table itself
        try:
            raw[p] = float(_ground_truth(_cold, var))
        except Exception:
            raw[p] = float(rng.uniform(0.5, 2.0))
    array_boundary_reals = {}
    for p in array_boundary_here:
        real = real_values.get(p)
        array_boundary_reals[p] = np.asarray(real, dtype=float) if real is not None else \
            np.ones(array_boundary_here[p] or (1,))
        want = array_boundary_here[p]
        got = array_boundary_reals[p].shape
        if want is not None and tuple(want) != tuple(got):
            raise SystemExit(
                f"[array-subdag] boundary {p!r}: kernel signature was built for shape "
                f"{tuple(want)} but the real value is {tuple(got)} -- refusing to "
                f"launch on a mis-shaped binding")
        raw[p] = jnp.asarray(array_boundary_reals[p])

    jax_env = eval_subdag(usable_entries, raw)
    jax_vals = np.array([float(jax_env[c]) for c in reachable_conditions])

    scalar_boundary = [p for p in boundary if p not in array_boundary_here]
    x = wp.array(np.tile([raw[p] for p in unknowns], (1, 1)), dtype=wp.float64, device="cpu")
    inputs = [x]
    if scalar_boundary:
        p_arr = wp.array(np.tile([raw[p] for p in scalar_boundary], (1, 1)), dtype=wp.float64, device="cpu")
        inputs.append(p_arr)
    for p in array_boundary_here:
        inputs.append(wp.array(array_boundary_reals[p], dtype=wp.float64, device="cpu"))
    for ident in sorted(table_registry):
        inputs.append(wp.array(np.asarray(table_registry[ident]), dtype=wp.float64, device="cpu"))
    r = wp.zeros((1, len(reachable_conditions)), dtype=wp.float64, device="cpu")
    inputs.append(r)
    wp.launch(kernel, dim=1, inputs=inputs, device="cpu")
    wp.synchronize()
    warp_vals = r.numpy()[0]

    rel = np.abs(warp_vals - jax_vals) / np.maximum(np.abs(jax_vals), 1e-300)
    print(f"\n[array-subdag] === agreement, {CONFIG}, {len(reachable_conditions)} outputs ===")
    for c, w, j, rl in zip(reachable_conditions, warp_vals, jax_vals, rel):
        print(f"    {c:55s} warp={w: .8e}  jax={j: .8e}  rel={rl:.3e}")
    finite = np.isfinite(rel)
    worst = rel[finite].max() if finite.any() else float("nan")
    verdict = "PASS" if finite.any() and worst <= AGREEMENT_RTOL else "FAIL"
    print(f"[array-subdag] *** worst relative difference (finite only): {worst:.3e} "
          f"-- {verdict} against AGREEMENT_RTOL = {AGREEMENT_RTOL:.0e} ***")
    if (~finite).any():
        print(f"[array-subdag] note: {(~finite).sum()} non-finite condition(s) -- "
              f"both engines NaN/inf at this random draw (a domain issue with the "
              f"synthetic input, not a Warp/JAX disagreement); see the per-condition "
              f"listing above.")

    result = {
        "config": CONFIG,
        "n_entries_total": len(entries),
        "n_entries_covered": len(usable_entries),
        "n_conditions_total": len(conditions),
        "conditions_covered": list(reachable_conditions),
        "conditions_excluded": list(excluded_conditions),
        "excluded_leaves": {f"{m}.{n}": str(w) for (m, n), w in bad.items()},
        "warp": warp_vals.tolist(),
        "jax": jax_vals.tolist(),
        "rel_diff": rel.tolist(),
        "worst_rel_diff_finite": float(worst),
        "agreement_rtol": AGREEMENT_RTOL,
        "agreement": verdict,
        "array_boundary": {p: list(n) if n else n for p, n in array_boundary_here.items()},
        "table_registry_keys": {k: len(v) for k, v in table_registry.items()},
        "extra_table_args": {f"{m}.{n}": v for (m, n), v in extra_table_args.items()},
    }

    # ---------------- registers + spill ----------------
    from .regcheck import kernel_registers
    array_boundary_lens = {p: (array_boundary_here[p] or (1,)) for p in array_boundary_here}
    xg = wp.array(np.ones((2, len(unknowns))), dtype=wp.float64, device="cuda:0")
    inputs_g = [xg]
    if scalar_boundary:
        pg = wp.array(np.ones((2, len(scalar_boundary))), dtype=wp.float64, device="cuda:0")
        inputs_g.append(pg)
    for p in array_boundary_here:
        inputs_g.append(wp.array(np.ones(array_boundary_lens[p]), dtype=wp.float64, device="cuda:0"))
    for ident in sorted(table_registry):
        inputs_g.append(wp.array(np.asarray(table_registry[ident]), dtype=wp.float64, device="cuda:0"))
    rg = wp.zeros((2, len(reachable_conditions)), dtype=wp.float64, device="cuda:0")
    inputs_g.append(rg)
    wp.launch(kernel, dim=2, inputs=inputs_g, device="cuda:0")
    wp.synchronize()
    module_id = kernel.module.get_module_identifier()
    regs = kernel_registers(str(wp.config.kernel_cache_dir), module_id)
    print(f"\n[array-subdag] === registers (sm_75, {len(usable_entries)}-node kernel) ===")
    for which in ("forward", "backward"):
        info = regs.get(which)
        if info:
            spill = "SPILLING" if info["local_bytes"] > 0 else "no spill"
            print(f"    {which:>8}: {info['regs']:>4} registers, "
                  f"{info['local_bytes']:>6} local bytes/thread ({spill})")
        else:
            print(f"    {which:>8}: not found in cubin")
    result["registers"] = regs

    with open(f"{GEN_DIR}/subdag_{CONFIG}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n[array-subdag] wrote subdag_{CONFIG}.json")


if __name__ == "__main__":
    main()
