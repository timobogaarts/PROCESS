"""Measurement, not generation: what the emitted kernels COST and whether their
reverse-mode adjoint is right.

Two questions, deliberately in one module because they share the whole build path
(`prepare`) and differ only in what they do with the kernel afterwards:

  * ``adjoint <config>`` -- build the module with ``wp.config.enable_backward = True``,
    tape one launch, and read out ``d(conditions)/d(unknowns)`` one reverse pass per
    condition. Compare it, entry by entry, against ``jax.jacfwd`` of the IDENTICAL
    sub-DAG (`jaxpr_harness.eval_subdag_jax`, the same reference the forward agreement
    number uses). The worst relative difference is reported as a number; there is no
    tolerance to tune.

  * ``bench <config> --device {cpu,cuda}`` -- forward kernel timing across batch sizes,
    with COMPILE time separated from steady-state run time, against the equivalent
    ``jax.jit(jax.vmap(...))`` of the same sub-DAG measured the same way.

  * ``regs <config>`` -- CUDA registers and local (spill) bytes for the forward and the
    backward kernel, via `regcheck`.

Nothing here changes codegen. `prepare` regenerates byte-identically what
`jaxpr_harness` writes, under the same module name, so a forward CPU build is a Warp
kernel-cache HIT and only the thing being measured is paid for.

**Batching, on the Warp side and the JAX side, varies the unknowns and holds the
boundary fixed** -- one thread per design point, which is the shape a batched solve
has. The scalar boundary is still bound as a full ``(n, nb)`` column array and read as
``p[tid, i]``, so the kernel does the memory traffic it would really do; the JAX side
is handed the same batched array rather than folded constants, so neither side gets a
constant-folding advantage the other does not.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
import shutil
import sys
import time
import traceback

import jax

jax.config.update("jax_enable_x64", True)

import pathlib

import jax.numpy as jnp
import numpy as np

from .emit import build_kernel_source
from .jaxpr_backend import jaxpr_leaves, module_preamble, warp_init
from .jaxpr_harness import GEN_DIR, eval_subdag_jax, prefix_closure


@dataclasses.dataclass
class Prep:
    config: str
    usable: list
    defs_by_node: dict
    unknowns: tuple
    boundary: tuple
    conditions: tuple          # reachable subset only
    scalar_boundary: list
    array_boundary: list
    live_arrays: dict
    native: dict
    raw: dict
    array_boundary_vals: dict
    module_path: str
    module_name: str
    kernel_name: str
    n_nodes: int
    n_eqns: int


def prepare(config: str, max_entries: int | None = None) -> Prep:
    """Rebuild `jaxpr_harness`'s covered sub-DAG and write its module -- identically,
    so the forward cache hits. Any node the harness had to exclude for Warp codegen is
    read back out of its JSON rather than rediscovered by a compile loop.

    `max_entries` truncates the covered sub-DAG to its first `n` nodes. The entries are
    topologically ordered and the closure is prefix-closed, so a prefix of it is itself
    a valid prefix-closed sub-DAG -- a smaller kernel of the same kind, not a different
    one. It exists because the full-coverage module cannot be built for CUDA on this
    machine at all (NVRTC is OOM-killed at ~12.5 GB), and a reduced-coverage kernel is
    the only way to get any CPU-against-GPU number. It writes its own module file and
    kernel name so it can never be confused with, or overwrite, the full one.
    """
    from functional_process.cottax import native as _native
    from functional_process.cottax.sand_harness import ground_truth as _gt

    from .assemble import _assemble

    drive, _, mda_env = _assemble(config)
    cold = _native.native_reference(
        f"tests/regression/input_files/{config}.IN.DAT").cold
    entries, refused, _ = jaxpr_leaves(config, drive=drive, cold=cold, mda_env=mda_env)
    n_nodes = len(entries) + len(refused)

    sub = drive.body.subgraph
    defs_by_node = {n.path_str(): sub[n] for n in sub.topological_order}

    unknowns = tuple(u.path_str() for u in drive.unknowns)
    boundary = tuple(v.path_str() for v in drive.context)
    probe = {}
    for var in drive.context:
        try:
            probe[var] = jnp.asarray(_gt(cold, var))
        except Exception:
            probe[var] = jnp.asarray(mda_env[var]) if var in mda_env else jnp.asarray(0.0)
    conditions = tuple(c.path_str() for c in drive.condition_map(probe).conditions)

    raw, native = {}, {}
    for p, var in list(zip(unknowns, drive.unknowns)) + list(zip(boundary, drive.context)):
        try:
            v = np.asarray(_gt(cold, var))
        except Exception:
            v = np.asarray(mda_env[var]) if var in mda_env else np.asarray(np.nan)
        native[p] = v
        raw[p] = float(v.reshape(())) if v.size == 1 else np.nan

    array_vars: dict = {}
    for e in entries:
        for pth, n in e.array_sizes().items():
            if array_vars.setdefault(pth, n) != n:
                raise SystemExit(f"[measure] {pth!r} array length disagreement")

    bad = {}
    jpath = f"{GEN_DIR}/jaxpr_subdag_{config}.json"
    if pathlib.Path(jpath).exists():
        bad = json.load(open(jpath)).get("warp_codegen_excluded", {}) or {}

    usable, _blocked, available = prefix_closure(entries, unknowns, boundary, bad)
    suffix = ""
    if max_entries is not None and max_entries < len(usable):
        usable = usable[:max_entries]
        available = set(unknowns) | set(boundary)
        for e in usable:
            available.update(e.outputs)
        suffix = f"_n{max_entries}"
    reachable = tuple(c for c in conditions if c in available)
    live_arrays = {pth: n for pth, n in array_vars.items()
                   if any(pth in e.inputs or pth in e.outputs for e in usable)}
    scalar_boundary = [p for p in boundary if p not in live_arrays]
    array_boundary = [p for p in boundary if p in live_arrays]
    array_boundary_vals = {p: np.asarray(native[p], dtype=float).reshape(-1)
                           for p in array_boundary}

    func_src = "\n\n".join(dict.fromkeys(e.source for e in usable))
    kernel_name = f"{config}_jaxpr_subdag{suffix}"
    kernel_src, _mapper = build_kernel_source(
        usable, unknowns, boundary, reachable, kernel_name=kernel_name,
        array_vars=live_arrays)
    module_src = ('"""GENERATED by functional_process/cottax/warp/jaxpr_backend '
                  '-- do not hand-edit."""\n' + module_preamble(usable) + "\n\n"
                  + func_src + "\n\n" + kernel_src)
    pathlib.Path(GEN_DIR).mkdir(exist_ok=True, parents=True)
    module_path = f"{GEN_DIR}/_jaxpr_subdag_{config}{suffix}.py"
    old = open(module_path).read() if pathlib.Path(module_path).exists() else None
    pathlib.Path(module_path).write_text(module_src)
    if old is not None and old != module_src:
        print("[measure] NOTE: regenerated module differs from the one on disk -- "
              "the forward kernel cache will MISS and a fresh compile is being timed")

    print(f"[measure] {config}: {len(usable)}/{n_nodes} entries, "
          f"{len(reachable)}/{len(conditions)} conditions, {len(unknowns)} unknowns, "
          f"{len(scalar_boundary)} scalar boundary, {len(array_boundary)} array "
          f"boundary; module {len(module_src) / 1e6:.2f} MB")
    return Prep(config, usable, defs_by_node, unknowns, boundary, reachable,
                scalar_boundary, array_boundary, live_arrays, native, raw,
                array_boundary_vals, module_path,
                f"_jaxpr_subdag_{config}{suffix}_0",
                kernel_name, n_nodes, sum(e.n_eqns for e in usable))


# ---------------------------------------------------------------------------
# loading / compiling
# ---------------------------------------------------------------------------


def _cache_dirs(wp, prep: Prep):
    root = wp.config.kernel_cache_dir
    pre = f"wp_{prep.module_name}_"
    if not pathlib.Path(root).is_dir():
        return []
    return sorted((os.path.join(root, d) for d in os.listdir(root)
                   if d.startswith(pre)),
                  key=lambda p: pathlib.Path(p).stat().st_mtime)


def load_kernel(wp, prep: Prep, device: str, fresh: bool = False):
    """Import the generated module and force its Warp compile for `device`, returning
    `(kernel, compile_seconds, cache_dir)`.

    The compile is timed by the FIRST `wp.launch`, not by `Module.load(device)`.
    `Module.load` with no `block_dim` hashes to a different module than the launch path
    does (measured: an explicit load compiles one hash and the following launch
    compiles another, paying twice), so timing the load would time a build nothing
    subsequently uses. `fresh` removes every kernel-cache directory belonging to this
    module first, so what is timed is a real compile rather than a cache hit.
    """
    spec = importlib.util.spec_from_file_location(prep.module_name, prep.module_path)
    gen = importlib.util.module_from_spec(spec)
    t0 = time.perf_counter()
    spec.loader.exec_module(gen)
    t_import = time.perf_counter() - t0
    k = getattr(gen, prep.kernel_name)

    before = _cache_dirs(wp, prep)
    if fresh:
        for d in before:
            shutil.rmtree(d)
            print(f"[measure] removed cache {d} -- timing a cold compile", flush=True)
        before = []
    print(f"[measure] python-exec {t_import:.2f} s; {len(before)} cached build(s) for "
          f"this module; compiling for {device}", flush=True)

    args, _x, _r = _wp_inputs(wp, prep, 1, device)
    t0 = time.perf_counter()
    wp.launch(k, dim=1, inputs=args, device=device)
    wp.synchronize_device(device)
    t_load = time.perf_counter() - t0
    after = _cache_dirs(wp, prep)
    new = [d for d in after if d not in before]
    cache_dir = new[-1] if new else (after[-1] if after else "")
    print(f"[measure] first launch on {device}: {t_load:.2f} s "
          f"({'COLD -- compiled' if new else 'warm -- cache hit'}); cache {cache_dir}",
          flush=True)
    return k, t_load, cache_dir


def _wp_inputs(wp, prep: Prep, n: int, device: str, requires_grad: bool = False):
    """The kernel's argument list at batch `n`, all rows identical to the real point."""
    xrow = np.array([prep.raw[p] for p in prep.unknowns], dtype=np.float64)
    prow = np.array([prep.raw[p] for p in prep.scalar_boundary], dtype=np.float64)
    x = wp.array(np.tile(xrow, (n, 1)), dtype=wp.float64, device=device,
                 requires_grad=requires_grad)
    args = [x]
    if prep.scalar_boundary:
        args.append(wp.array(np.tile(prow, (n, 1)), dtype=wp.float64, device=device))
    for b in prep.array_boundary:
        args.append(wp.array(prep.array_boundary_vals[b], dtype=wp.float64,
                             device=device))
    r = wp.zeros((n, len(prep.conditions)), dtype=wp.float64, device=device,
                 requires_grad=requires_grad)
    args.append(r)
    return args, x, r


# ---------------------------------------------------------------------------
# the JAX side: the same sub-DAG, batched the same way
# ---------------------------------------------------------------------------


def make_jax_single(prep: Prep):
    """`f(xrow, prow) -> conditions`, the identical sub-DAG under JAX.

    `prow` carries the scalar boundary as float64 (the kernel's ABI); an entry whose
    real value is integer- or bool-typed is cast back before use, which is exactly the
    conversion the emitted `@wp.func` does on its own side.
    """
    casts = {}
    for i, p in enumerate(prep.scalar_boundary):
        dt = np.asarray(prep.native[p]).dtype
        if dt.kind in "iub":
            casts[i] = dt

    def f(xrow, prow, *arrs):
        env = {}
        for i, p in enumerate(prep.unknowns):
            env[p] = xrow[i]
        for i, p in enumerate(prep.scalar_boundary):
            v = prow[i]
            env[p] = v.astype(casts[i]) if i in casts else v
        for p, a in zip(prep.array_boundary, arrs):
            env[p] = a
        env = eval_subdag_jax(prep.usable, prep.defs_by_node, env)
        return jnp.stack([jnp.asarray(env[c]).reshape(()) for c in prep.conditions])

    return f


def jax_array_args(prep: Prep):
    """The array boundaries in their NATIVE shape, as ARGUMENTS rather than closed-over
    constants.

    Two reasons, and the second is the load-bearing one. The shape: a node's `fn`
    reshapes or vmaps over the table (the 14x200 impurity pair among them), so the flat
    buffer the kernel binds is a different computation on the JAX side. And the binding:
    the Warp kernel receives each of these as a runtime `wp.array` parameter, so
    closing over them in JAX would let XLA constant-fold whole nodes the kernel
    actually executes -- measured, on the 10-node cut, as a condition folding to a
    literal and the JAX column reading 0.016 us/point for 12,065 jaxpr equations.
    Passing them as arguments is what makes the two sides the same measurement.
    """
    return tuple(jnp.asarray(prep.native[p]) for p in prep.array_boundary)


def _x0_p0(prep: Prep):
    return (jnp.array([prep.raw[p] for p in prep.unknowns], dtype=jnp.float64),
            jnp.array([prep.raw[p] for p in prep.scalar_boundary], dtype=jnp.float64))


def _rel(a, b):
    """Elementwise |a-b| / |b|, with the zero-denominator entries reported apart."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    nz = b != 0.0
    rel = np.full(a.shape, np.nan)
    rel[nz] = np.abs(a[nz] - b[nz]) / np.abs(b[nz])
    finite = np.isfinite(rel)
    worst = float(rel[finite].max()) if finite.any() else float("nan")
    zero_abs = float(np.abs(a[~nz] - b[~nz]).max()) if (~nz).any() else 0.0
    return rel, worst, zero_abs, int(nz.sum()), int((~nz).sum())


# ---------------------------------------------------------------------------
# part 1: adjoint
# ---------------------------------------------------------------------------


def cmd_adjoint(config: str, fresh: bool, max_entries: int | None = None):
    print(f"[measure/adjoint] === {config} ===", flush=True)
    t_start = time.perf_counter()
    wp = warp_init(enable_backward=True)
    print(f"[measure/adjoint] wp.config.enable_backward = "
          f"{wp.config.enable_backward}", flush=True)
    prep = prepare(config, max_entries)
    print(f"[measure/adjoint] prepared at t+{time.perf_counter() - t_start:.1f} s; "
          f"compiling WITH adjoints -- this is the long part", flush=True)

    t_c0 = time.perf_counter()
    try:
        k, t_load, cache_dir = load_kernel(wp, prep, "cpu", fresh=fresh)
    except Exception as exc:
        print(f"[measure/adjoint] *** BACKWARD BUILD FAILED after "
              f"{time.perf_counter() - t_c0:.1f} s ***\n{str(exc)[-4000:]}", flush=True)
        raise SystemExit(1)
    print(f"[measure/adjoint] backward-enabled compile: {t_load:.1f} s", flush=True)

    n = 1
    args, x, r = _wp_inputs(wp, prep, n, "cpu", requires_grad=True)

    tape = wp.Tape()
    t0 = time.perf_counter()
    with tape:
        wp.launch(k, dim=n, inputs=args, device="cpu")
    wp.synchronize()
    t_fwd_taped = time.perf_counter() - t0
    warp_vals = r.numpy()[0].copy()
    print(f"[measure/adjoint] taped forward launch: {t_fwd_taped:.3f} s", flush=True)

    nc, nu = len(prep.conditions), len(prep.unknowns)
    J_warp = np.zeros((nc, nu))
    t_back = []
    for j in range(nc):
        tape.zero()
        seed = np.zeros((n, nc)); seed[0, j] = 1.0
        r.grad.assign(seed)
        t0 = time.perf_counter()
        tape.backward()
        wp.synchronize()
        t_back.append(time.perf_counter() - t0)
        J_warp[j] = x.grad.numpy()[0].copy()
        print(f"[measure/adjoint]   reverse pass {j + 1}/{nc}: {t_back[-1]:.3f} s",
              flush=True)

    f0 = make_jax_single(prep)
    arrs = jax_array_args(prep)

    def f(xv, pv):
        return f0(xv, pv, *arrs)
    x0, p0 = _x0_p0(prep)
    t0 = time.perf_counter()
    jax_vals = np.asarray(f(x0, p0))
    t_jax_fwd = time.perf_counter() - t0
    t0 = time.perf_counter()
    J_jax = np.asarray(jax.jacfwd(f, argnums=0)(x0, p0))
    t_jax_jac = time.perf_counter() - t0
    # JAX's OWN reverse mode, for attribution: Warp's adjoint and `jacfwd` accumulate
    # in opposite orders, so a disagreement at the last few digits could be either
    # side's rounding. `jacrev` against `jacfwd` is the same comparison with Warp taken
    # out of it, and it says how much of the gap is reverse-vs-forward at all.
    t0 = time.perf_counter()
    J_jaxrev = np.asarray(jax.jacrev(f, argnums=0)(x0, p0))
    t_jax_rev = time.perf_counter() - t0
    print(f"[measure/adjoint] jax forward {t_jax_fwd:.1f} s, jacfwd {t_jax_jac:.1f} s, "
          f"jacrev {t_jax_rev:.1f} s", flush=True)

    _, v_worst, v_zero, _, _ = _rel(warp_vals, jax_vals)
    print(f"[measure/adjoint] forward values, worst rel: {v_worst:.3e} "
          f"(zero-valued entries, worst abs: {v_zero:.3e})")

    rel, worst, zero_abs, n_nz, n_z = _rel(J_warp, J_jax)
    print(f"\n[measure/adjoint] === Jacobian d(conditions)/d(unknowns), "
          f"{nc} x {nu} = {nc * nu} entries ===")
    print(f"  {n_nz} entries with a non-zero JAX value; {n_z} exactly zero in JAX")
    print(f"  *** worst relative difference (non-zero entries): {worst:.3e} ***")
    print(f"  *** worst absolute difference where JAX is exactly zero: "
          f"{zero_abs:.3e} ***")
    fin = np.isfinite(rel)
    if fin.any():
        idx = np.unravel_index(np.nanargmax(np.where(fin, rel, -1)), rel.shape)
        print(f"  worst at condition {prep.conditions[idx[0]]} / unknown "
              f"{prep.unknowns[idx[1]]}: warp={J_warp[idx]: .12e} "
              f"jax={J_jax[idx]: .12e}")
        for q in (50, 90, 99):
            print(f"  p{q} rel: {np.percentile(rel[fin], q):.3e}")
    _, wr_worst, wr_zero, _, _ = _rel(J_warp, J_jaxrev)
    _, fr_worst, fr_zero, _, _ = _rel(J_jaxrev, J_jax)
    print(f"  warp-adjoint vs jax.jacrev: worst rel {wr_worst:.3e} "
          f"(abs at zero {wr_zero:.3e})")
    print(f"  jax.jacrev  vs jax.jacfwd:  worst rel {fr_worst:.3e} "
          f"(abs at zero {fr_zero:.3e})  <- reverse-vs-forward alone")
    nrm_w = np.linalg.norm(J_warp)
    nrm_j = np.linalg.norm(J_jax)
    print(f"  ||J_warp||_F = {nrm_w:.8e}   ||J_jax||_F = {nrm_j:.8e}   "
          f"rel = {abs(nrm_w - nrm_j) / max(nrm_j, 1e-300):.3e}")

    out = {
        "config": config, "n_conditions": nc, "n_unknowns": nu,
        "backward_compile_s": t_load, "taped_forward_s": t_fwd_taped,
        "reverse_pass_s": t_back, "jax_jacfwd_s": t_jax_jac,
        "jax_forward_s": t_jax_fwd,
        "forward_worst_rel": v_worst,
        "jacobian_worst_rel_nonzero": worst,
        "jacobian_worst_abs_at_jax_zero": zero_abs,
        "warp_vs_jacrev_worst_rel": wr_worst,
        "jacrev_vs_jacfwd_worst_rel": fr_worst,
        "jax_jacrev_s": t_jax_rev,
        "n_nonzero": n_nz, "n_zero": n_z,
        "frob_warp": nrm_w, "frob_jax": nrm_j,
        "cache_dir": cache_dir,
        "conditions": list(prep.conditions), "unknowns": list(prep.unknowns),
        "J_warp": J_warp.tolist(), "J_jax": J_jax.tolist(),
    }
    atag = "" if max_entries is None else f"_n{max_entries}"
    p = f"{GEN_DIR}/measure_adjoint_{config}{atag}.json"
    with open(p, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[measure/adjoint] wrote {p} (total {time.perf_counter() - t_start:.1f} s)")


# ---------------------------------------------------------------------------
# part 1c: compile cost, forward vs backward
# ---------------------------------------------------------------------------


def cmd_compile(config: str, backward: bool, device: str,
                max_entries: int | None = None):
    """One number: a COLD Warp compile of this config's kernel, adjoints on or off."""
    t_start = time.perf_counter()
    wp = warp_init(enable_backward=backward)
    prep = prepare(config, max_entries)
    t_prep = time.perf_counter() - t_start
    _k, t_load, cache_dir = load_kernel(wp, prep, device, fresh=True)
    cpp = [os.path.join(cache_dir, f) for f in os.listdir(cache_dir)
           if f.endswith((".cpp", ".cu"))]
    src_bytes = sum(pathlib.Path(f).stat().st_size for f in cpp)
    print(f"[measure/compile] {config} device={device} backward={backward}: "
          f"prepare {t_prep:.1f} s, COLD compile {t_load:.1f} s, generated source "
          f"{src_bytes / 1e6:.2f} MB")
    with open(f"{GEN_DIR}/measure_compile_{config}_{device}_"
              f"{'bw' if backward else 'fw'}.json", "w") as fh:
        json.dump({"config": config, "device": device, "backward": backward,
                   "compile_s": t_load, "prepare_s": t_prep,
                   "generated_source_bytes": src_bytes,
                   "python_module_bytes": pathlib.Path(prep.module_path).stat().st_size}, fh,
                  indent=2)


# ---------------------------------------------------------------------------
# part 1d: registers and spill
# ---------------------------------------------------------------------------


def cmd_regs(config: str, backward: bool, max_entries: int | None = None):
    from .regcheck import kernel_registers
    wp = warp_init(enable_backward=backward)
    prep = prepare(config, max_entries)
    k, t_load, cache_dir = load_kernel(wp, prep, "cuda:0", fresh=False)
    # Which cache directory holds THIS build's cubin: the newest one belonging to this
    # module. `load_kernel`'s returned dir is chosen by mtime and a forward and a
    # backward build of the same module both match, so ask the cubins directly.
    cubins = [os.path.join(d, f) for d in _cache_dirs(wp, prep)
              for f in os.listdir(d) if f.endswith(".cubin")]
    if not cubins:
        print("[measure/regs] no .cubin under any cache dir for this module -- the "
              "CUDA build did not produce one; nothing to report")
        return
    newest = max(cubins, key=os.path.getmtime)
    h = os.path.basename(os.path.dirname(newest)).rsplit("_", 1)[-1]
    print(f"[measure/regs] reading {newest}")
    info = kernel_registers(wp.config.kernel_cache_dir, h)
    print(f"[measure/regs] {config} backward={backward} compile {t_load:.1f} s")
    print(f"[measure/regs] {json.dumps(info, indent=2)}")
    rtag = "" if max_entries is None else f"_n{max_entries}"
    with open(f"{GEN_DIR}/measure_regs_{config}{rtag}_"
              f"{'bw' if backward else 'fw'}.json", "w") as fh:
        json.dump({"config": config, "backward": backward,
                   "max_entries": max_entries, "compile_s": t_load,
                   "kernels": info}, fh, indent=2)


# ---------------------------------------------------------------------------
# part 2: forward timing across batch sizes
# ---------------------------------------------------------------------------


def _time_launch(wp, k, args, n, device, repeats):
    wp.launch(k, dim=n, inputs=args, device=device)
    wp.synchronize_device(device)
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(repeats):
            wp.launch(k, dim=n, inputs=args, device=device)
        wp.synchronize_device(device)
        best = min(best, (time.perf_counter() - t0) / repeats)
    return best


def cmd_bench(config: str, device: str, batches, fresh: bool, check_at,
              max_entries: int | None = None):
    print(f"[measure/bench] === {config} device={device} ===", flush=True)
    wp = warp_init(enable_backward=False)
    prep = prepare(config, max_entries)
    wp_dev = "cpu" if device == "cpu" else "cuda:0"
    k, t_compile, _cd = load_kernel(wp, prep, wp_dev, fresh=fresh)
    print(f"[measure/bench] warp compile ({'cold' if fresh else 'cached'}): "
          f"{t_compile:.2f} s", flush=True)

    jf = make_jax_single(prep)
    arrs = jax_array_args(prep)
    jbatch = jax.jit(jax.vmap(jf, in_axes=(0, 0) + (None,) * len(arrs)))
    x0, p0 = _x0_p0(prep)

    rows = []
    for n in batches:
        row = {"n": n}
        try:
            args, x, r = _wp_inputs(wp, prep, n, wp_dev)
            reps = 20 if n <= 4096 else 5
            row["warp_s"] = _time_launch(wp, k, args, n, wp_dev, reps)
            warp_out = r.numpy().copy()
        except Exception as exc:
            row["warp_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            warp_out = None
            print(f"[measure/bench] n={n}: WARP FAILED: {row['warp_error']}", flush=True)

        try:
            xb = jnp.tile(x0, (n, 1))
            pb = jnp.tile(p0, (n, 1))
            t0 = time.perf_counter()
            out = jbatch(xb, pb, *arrs)
            out.block_until_ready()
            row["jax_compile_s"] = time.perf_counter() - t0
            best = float("inf")
            for _ in range(3):
                t0 = time.perf_counter()
                reps = 20 if n <= 4096 else 5
                for _ in range(reps):
                    o = jbatch(xb, pb, *arrs)
                o.block_until_ready()
                best = min(best, (time.perf_counter() - t0) / reps)
            row["jax_s"] = best
            jax_out = np.asarray(out)
        except Exception as exc:
            row["jax_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            jax_out = None
            print(f"[measure/bench] n={n}: JAX FAILED: {row['jax_error']}", flush=True)
            traceback.print_exc()

        if n in check_at and warp_out is not None and jax_out is not None:
            _, worst, zero_abs, _, _ = _rel(warp_out, jax_out)
            row["agree_worst_rel"] = worst
            row["agree_worst_abs_at_zero"] = zero_abs
            print(f"[measure/bench] n={n}: agreement worst rel {worst:.3e}", flush=True)
        rows.append(row)
        print(f"[measure/bench] n={n:>6}: warp {row.get('warp_s', float('nan')):.6f} s"
              f"  jax {row.get('jax_s', float('nan')):.6f} s"
              f"  (jax jit {row.get('jax_compile_s', float('nan')):.1f} s)", flush=True)

    tag = "" if max_entries is None else f"_n{max_entries}"
    print(f"\n[measure/bench] === {config}, device={device}{tag} ===")
    print(f"  warp compile ({'cold' if fresh else 'cached'}): {t_compile:.2f} s")
    print(f"  {'n':>7} {'warp s':>12} {'warp us/pt':>12} {'jax s':>12} "
          f"{'jax us/pt':>12} {'jax jit s':>10} {'worst rel':>11}")
    for row in rows:
        w = row.get("warp_s", float("nan"))
        j = row.get("jax_s", float("nan"))
        print(f"  {row['n']:>7} {w:>12.6f} {w / row['n'] * 1e6:>12.3f} {j:>12.6f} "
              f"{j / row['n'] * 1e6:>12.3f} {row.get('jax_compile_s', float('nan')):>10.1f} "
              f"{row.get('agree_worst_rel', float('nan')):>11.3e}")

    p = f"{GEN_DIR}/measure_bench_{config}_{device}{tag}.json"
    with open(p, "w") as fh:
        json.dump({"config": config, "device": device,
                   "warp_compile_s": t_compile, "warp_compile_cold": fresh,
                   "n_unknowns": len(prep.unknowns),
                   "n_conditions": len(prep.conditions),
                   "n_entries": len(prep.usable), "n_eqns": prep.n_eqns,
                   "rows": rows}, fh, indent=2)
    print(f"[measure/bench] wrote {p}")


def cmd_dump(config: str):
    """Per-entry emitted-source size and the entry index at which each condition first
    becomes reachable -- what a `--max-entries` cut has to be chosen against.
    """
    prep = prepare(config)
    avail = set(prep.unknowns) | set(prep.boundary)
    cum = 0
    first = {}
    print(f"  {'#':>3} {'src KB':>9} {'cum MB':>8} {'eqns':>8}  node")
    for i, e in enumerate(prep.usable):
        cum += len(e.source)
        avail.update(e.outputs)
        for c in prep.conditions:
            if c in avail and c not in first:
                first[c] = i
        print(f"  {i:>3} {len(e.source) / 1e3:>9.1f} {cum / 1e6:>8.2f} {e.n_eqns:>8}  "
              f"{e.node}")
    print("\n  condition first reachable after entry:")
    for c in prep.conditions:
        print(f"    {first.get(c, '-'):>4}  {c}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="measure")
    ap.add_argument("cmd", choices=["adjoint", "bench", "compile", "regs", "dump"])
    ap.add_argument("config", nargs="?", default="helias_5b")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--backward", action="store_true")
    ap.add_argument("--fresh", action="store_true",
                    help="delete the kernel-cache entry first, so a COLD compile is "
                         "what gets timed")
    ap.add_argument("--max-entries", type=int, default=None,
                    help="truncate the covered sub-DAG to its first N nodes")
    ap.add_argument("--batches", default="1,16,256,4096,65536")
    ap.add_argument("--check-at", default="1,256")
    a = ap.parse_args(argv)
    if a.cmd == "adjoint":
        cmd_adjoint(a.config, a.fresh, a.max_entries)
    elif a.cmd == "compile":
        cmd_compile(a.config, a.backward, a.device, a.max_entries)
    elif a.cmd == "regs":
        cmd_regs(a.config, a.backward, a.max_entries)
    elif a.cmd == "dump":
        cmd_dump(a.config)
    else:
        cmd_bench(a.config, a.device,
                  [int(s) for s in a.batches.split(",") if s],
                  a.fresh, {int(s) for s in a.check_at.split(",") if s},
                  a.max_entries)


if __name__ == "__main__":
    main(sys.argv[1:])
