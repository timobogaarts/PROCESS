"""**Per-node validation**: run each generated `@wp.func` against `defn.fn` in JAX, at
the same inputs, swept over several draws.

This is the check the jaxpr backend is worth having. The previous design could only ever
compare one whole sub-DAG at one input point, so a single number certified forty
functions at once and any node whose contribution happened to cancel, or to be swamped
by a larger term, was certified for free. Here every node is its own experiment: a
disagreement is attributed to the node that caused it, and a node that is never reached
by the assembled kernel is still checked.

**Swept, not sampled.** A previous agent found a real 1-ulp disagreement at 1 of 400
swept points that a single point missed entirely. Each node is evaluated at its ground
truth point plus `N_DRAWS - 1` multiplicative random perturbations of it (log-uniform,
so a positive physical quantity stays positive and stays within its own order of
magnitude -- an additive draw around, say, a 1e20 density produces nothing but noise).

Integer-typed reads are held FIXED across the sweep. An integer read in this graph is an
index or a species selector; perturbing one does not test the emitted arithmetic, it
tests whether an out-of-range index happens to be caught, and that is a different
question.

Run:

    python -m functional_process.cottax.warp.jaxpr_validate helias_5b
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from . import jaxpr_backend as _jb
from .jaxpr_backend import (jaxpr_leaves, module_preamble, node_values,
                            warp_init)

N_DRAWS = 8
"""Ground truth plus 7 perturbations. Enough that a branch-dependent or
argument-order bug shows up; small enough that 150 nodes stay a seconds-scale run."""

SPREAD = 0.35
"""Log-uniform half-width of the multiplicative perturbation, i.e. each float read is
multiplied by `exp(U(-0.35, 0.35))` -- roughly x0.70 to x1.42."""

AGREEMENT_RTOL = 1e-12
"""Same gate as `jaxpr_harness.AGREEMENT_RTOL`, and for the same reason: it is a bug
detector, not an accuracy claim -- every misbinding this work has actually found fires
enormously higher than one ulp, while the worst transcendental disagreement in the
generated code is ~4 ulp. The worst relative difference is reported as a number
regardless, because the number is the evidence and the gate is only a summary of it."""

GEN_DIR = "functional_process/warp"


def draws(base, rng, n=N_DRAWS, spread=SPREAD):
    """`n` input tuples for one node: `base` itself first, then multiplicative
    perturbations of every FLOAT read (integer/bool reads held fixed, see the module
    docstring)."""
    out = [list(base)]
    for _ in range(n - 1):
        row = []
        for v in base:
            a = np.asarray(v)
            if np.issubdtype(a.dtype, np.floating):
                f = np.exp(rng.uniform(-spread, spread, size=a.shape))
                row.append(jnp.asarray(a * f))
            else:
                row.append(jnp.asarray(a))
        out.append(row)
    return out


def flat_layout(sizes):
    """`(offsets, total)` -- where each value starts in a flat float64 row.

    An array-valued read or return occupies its own `size` consecutive columns, in the
    same row-major order the emitted vector uses, so one flat row still carries a whole
    node's inputs (or outputs) whatever their shapes.
    """
    off, total = [], 0
    for n in sizes:
        off.append(total)
        total += n
    return off, total


def _validation_kernel(entry, index: int) -> str:
    """One `@wp.kernel` wrapping one node's `@wp.func`, reading its arguments from a
    row of `inp` and writing its returns to a row of `out`.

    An array-valued read is packed into its `vec{n}f` from its own slice of the row
    before the call, and an array-valued return is unpacked back into its slice
    afterwards -- so a node with array ports is checked by exactly the same sweep as a
    scalar one, rather than being reported unbindable.
    """
    in_sizes = entry.input_sizes or (1,) * len(entry.inputs)
    out_sizes = entry.output_sizes or (1,) * len(entry.outputs)
    in_off, _ = flat_layout(in_sizes)
    out_off, _ = flat_layout(out_sizes)
    lines = ["@wp.kernel",
             f"def check_{index}(inp: wp.array2d(dtype=wp.float64), "
             f"out: wp.array2d(dtype=wp.float64)):",
             "    tid = wp.tid()"]
    args = []
    for i, (n, base) in enumerate(zip(in_sizes, in_off)):
        if n == 1:
            args.append(f"inp[tid, {base}]")
            continue
        v = f"v{i}"
        lines.append(f"    {v} = {_jb.vec_name(n)}()")
        for k in range(n):
            lines.append(f"    {v}[{k}] = inp[tid, {base + k}]")
        args.append(v)
    call = f"{entry.fn}({', '.join(args)})"
    rets = ", ".join(f"r{i}" for i in range(len(out_sizes)))
    lines.append(f"    {rets} = {call}" if len(out_sizes) > 1
                 else f"    r0 = {call}")
    for i, (n, base) in enumerate(zip(out_sizes, out_off)):
        if n == 1:
            lines.append(f"    out[tid, {base}] = r{i}")
        else:
            for k in range(n):
                lines.append(f"    out[tid, {base + k}] = r{i}[{k}]")
    return "\n".join(lines) + "\n"


def _failing_func(err_text: str) -> str | None:
    names = re.findall(r'Error while parsing function "([^"]+)"', err_text)
    return names[-1] if names else None


def build_module(entries, config: str, tag: str):
    """Write and import ONE module holding every node's `@wp.func` plus one validation
    kernel each. One module means one Warp compile for the whole config; a codegen
    failure names the function it came from and the caller drops that node and retries
    (`validate`, below), so a single bad node cannot hide the rest."""
    src = ['"""GENERATED by functional_process/cottax/warp/jaxpr_backend -- do not '
           'hand-edit."""',
           module_preamble(entries), ""]
    for e in entries:
        src.append(e.source)
    for i, e in enumerate(entries):
        src.append(_validation_kernel(e, i))
    text = "\n".join(src)
    os.makedirs(GEN_DIR, exist_ok=True)
    path = f"{GEN_DIR}/_jaxpr_nodes_{config}.py"
    with open(path, "w") as f:
        f.write(text)
    spec = importlib.util.spec_from_file_location(f"_jaxpr_nodes_{config}_{tag}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, path


def validate(config: str, verbose: bool = True):
    """`(entries, results, refused)` -- the emitted entries that COMPILE, one result
    row per node, and everything refused (by the backend or by Warp codegen).

    A result row is
    `{node, n_draws, n_compared, worst_rel, verdict, ...}`; `verdict` is `PASS`,
    `FAIL`, or `NO-FINITE` when both engines produced NaN/inf at every draw (a domain
    issue with the perturbed input, not a disagreement -- reported, never silently
    counted as a pass).
    """
    wp = warp_init()

    from functional_process.cottax import native as _native
    from .assemble import _assemble

    drive, _, mda_env = _assemble(config)
    cold = _native.native_reference(
        f"tests/regression/input_files/{config}.IN.DAT").cold
    entries, refused, _ = jaxpr_leaves(config, drive=drive, cold=cold,
                                       mda_env=mda_env)
    if verbose:
        print(f"[jaxpr] {config}: {len(entries)} emitted, {len(refused)} refused "
              f"out of {len(entries) + len(refused)} nodes")

    sub = drive.body.subgraph
    by_node = {n.path_str(): sub[n] for n in sub.topological_order}

    # Every emitted node is checkable: an array-valued read rides its own slice of the
    # flat float64 row and is packed into the node's `vec{n}f` inside the validation
    # kernel (`_validation_kernel`). What is checked here is exactly what the assembled
    # kernel will call, arrays included.
    checkable, unbindable = [], []
    base_vals = {}
    for e in entries:
        vals = node_values(cold, by_node[e.node], mda_env=mda_env)
        sizes = e.input_sizes or tuple(1 for _ in vals)
        got = tuple(int(np.asarray(v).size) for v in vals)
        if got != tuple(sizes):
            # The trace's own shapes and the values fed back here have to agree, or the
            # sweep is checking a different function from the one that was emitted.
            unbindable.append((e.node, f"read sizes {got} != emitted signature "
                                       f"{tuple(sizes)}"))
            continue
        # Kept at their OWN shapes -- `defn.fn` is called with them -- and flattened
        # row-major only when the Warp row is built, which is the order `emit_node`
        # laid the vector's components out in.
        base_vals[e.node] = [np.asarray(v) if n > 1 else np.asarray(v).reshape(())
                             for v, n in zip(vals, sizes)]
        checkable.append(e)

    # Compile, dropping any node Warp's own codegen rejects, and retry. Measured, not
    # assumed: which constructs Warp accepts is not something to predict from its
    # documentation, and a single rejected function must not hide the other hundred.
    mod = None
    for attempt in range(30):
        if not checkable:
            break
        try:
            mod, path = build_module(checkable, config, f"a{attempt}")
            break
        except Exception as exc:
            bad = _failing_func(str(exc))
            hit = [e for e in checkable if e.fn == bad]
            if not hit:
                if verbose:
                    print(f"[jaxpr] codegen failure not attributable to one function: "
                          f"{str(exc)[-400:]}")
                return checkable, [], refused + unbindable
            for e in hit:
                refused.append((e.node, f"Warp codegen: {str(exc)[:160]}"))
            checkable = [e for e in checkable if e.fn != bad]
    if mod is None:
        return [], [], refused + unbindable

    rng = np.random.default_rng(20260906)
    results = []
    for i, e in enumerate(checkable):
        defn = by_node[e.node]
        rows = draws(base_vals[e.node], rng)
        n_in, n_out = len(e.inputs), len(e.outputs)
        in_sizes = e.input_sizes or (1,) * n_in
        out_sizes = e.output_sizes or (1,) * n_out
        _, n_in_cols = flat_layout(in_sizes)
        _, n_out_cols = flat_layout(out_sizes)

        # Every element of an array-valued return is compared, not just the first --
        # a wrong LENGTH is worse than a refusal, and a wrong element in the middle of
        # a 201-point profile is exactly what a shape-only check would miss.
        jax_out = np.full((len(rows), n_out_cols), np.nan)
        ok_jax = True
        for r, row in enumerate(rows):
            try:
                res = defn.fn(*row)
            except Exception:
                ok_jax = False
                break
            flat = jax.tree_util.tree_leaves(res)
            if len(flat) != n_out or \
                    tuple(int(np.asarray(x).size) for x in flat) != tuple(out_sizes):
                ok_jax = False
                break
            jax_out[r] = np.concatenate(
                [np.asarray(x, dtype=float).reshape(-1) for x in flat])
        if not ok_jax:
            results.append({"node": e.node, "verdict": "JAX-ERROR", "worst_rel": np.nan,
                            "n_compared": 0, "n_out": n_out})
            continue

        inp = wp.array(
            np.array([np.concatenate([np.asarray(v, dtype=float).reshape(-1)
                                      for v in row]) for row in rows]),
            dtype=wp.float64, device="cpu")
        assert inp.shape[1] == n_in_cols
        out = wp.zeros((len(rows), n_out_cols), dtype=wp.float64, device="cpu")
        kern = getattr(mod, f"check_{i}")
        try:
            wp.launch(kern, dim=len(rows), inputs=[inp, out], device="cpu")
            wp.synchronize()
        except Exception as exc:
            results.append({"node": e.node, "verdict": "WARP-ERROR",
                            "worst_rel": np.nan, "n_compared": 0, "n_out": n_out,
                            "detail": str(exc)[:200]})
            continue
        wout = out.numpy()

        rel = np.abs(wout - jax_out) / np.maximum(np.abs(jax_out), 1e-300)
        # Both engines NaN/inf at a draw is a property of the perturbed input, not a
        # disagreement; both engines producing the SAME non-finite value counts as
        # agreement, and only one of them doing so is a real failure.
        both_nan = ~np.isfinite(wout) & ~np.isfinite(jax_out)
        same_sign_inf = np.isinf(wout) & np.isinf(jax_out) & (np.sign(wout) == np.sign(jax_out))
        agree_nonfinite = both_nan | same_sign_inf
        one_sided = (~np.isfinite(wout) ^ ~np.isfinite(jax_out))
        comparable = np.isfinite(rel) & ~agree_nonfinite
        worst = float(rel[comparable].max()) if comparable.any() else np.nan
        if one_sided.any():
            verdict = "FAIL"
        elif not comparable.any():
            verdict = "NO-FINITE"
        else:
            verdict = "PASS" if worst <= AGREEMENT_RTOL else "FAIL"
        results.append({"node": e.node, "fn": e.fn, "verdict": verdict,
                        "worst_rel": worst, "n_compared": int(comparable.sum()),
                        "n_draws": len(rows), "n_out": n_out, "n_in": n_in,
                        "n_eqns": e.n_eqns,
                        "one_sided": int(one_sided.sum())})

    if verbose:
        _report(config, results, refused, unbindable)
    return checkable, results, refused + unbindable


def _report(config, results, refused, unbindable):
    from collections import Counter

    counts = Counter(r["verdict"] for r in results)
    print(f"\n[jaxpr] === per-node validation, {config} "
          f"({N_DRAWS} draws/node) ===")
    for v in ("PASS", "FAIL", "NO-FINITE", "JAX-ERROR", "WARP-ERROR"):
        if counts.get(v):
            print(f"    {v:<11} {counts[v]}")
    fails = [r for r in results if r["verdict"] == "FAIL"]
    if fails:
        print("\n[jaxpr] failing nodes:")
        for r in sorted(fails, key=lambda r: -(r["worst_rel"] if np.isfinite(r["worst_rel"]) else 0)):
            print(f"    {r['node']:<62} worst_rel={r['worst_rel']:.3e} "
                  f"one-sided={r.get('one_sided', 0)}")
    # A non-PASS that is not a FAIL is still not a pass: name it, so "144 PASS" is
    # never read as "145 nodes checked".
    for v in ("NO-FINITE", "JAX-ERROR", "WARP-ERROR"):
        rows = [r for r in results if r["verdict"] == v]
        if rows:
            print(f"\n[jaxpr] {v} (checked, but no comparable value at any draw):")
            for r in rows:
                print(f"    {r['node']}")
    passing = [r for r in results
               if r["verdict"] == "PASS" and np.isfinite(r["worst_rel"])]
    if passing:
        worst_row = max(passing, key=lambda r: r["worst_rel"])
        print(f"\n[jaxpr] worst relative difference among PASSING nodes: "
              f"{worst_row['worst_rel']:.3e}  (gate {AGREEMENT_RTOL:.0e})  at "
              f"{worst_row['node']}")
        inexact = sorted((r for r in passing if r["worst_rel"] > 0),
                         key=lambda r: -r["worst_rel"])
        if inexact:
            print(f"[jaxpr] {len(inexact)} passing node(s) not bit-exact:")
            for r in inexact:
                print(f"    {r['node']:<62} {r['worst_rel']:.3e}")
        passing = [r["worst_rel"] for r in passing]
        nonzero = [p for p in passing if p > 0]
        print(f"[jaxpr] {len(passing) - len(nonzero)}/{len(passing)} passing nodes are "
              f"BIT-EXACT across every draw")
    print(f"\n[jaxpr] refused ({len(refused) + len(unbindable)}):")
    for n, why in list(refused) + list(unbindable):
        print(f"    {n:<62} {why[:100]}")


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "helias_5b"
    validate(cfg)
