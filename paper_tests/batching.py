"""Does batching pay on the GPU? `jax.vmap` over N independent points of the two
shapes the port can batch, timed per point on the CPU and on the GPU in the same env.

Two shapes:

* **`mda`** -- `process/core/scan.py`'s shape: the hand-cut MDA (`evaluate.
  mda_schedule`) run at N points that differ in two boundary inputs,
  `.physics.rmajor` and `.physics.b_plasma_toroidal_on_axis`, on a sqrt(N) x sqrt(N)
  grid of +-5 % about the reference. Only those two leaves carry a batch axis
  (`in_axes` is a `PathMap` of `0`/`None`); every Picard block is `optimistix`'s
  `fixed_point`, which vmaps into one batched `while`.
* **`sand`** -- the fused value+Jacobian of the SAND block (`host_cache`'s
  `_values_and_jacobian` program: one `jacfwd` with the primal as `aux`), vmapped over
  N flat design vectors drawn as +-1 % perturbations of the design at the
  configuration's own cold values (`session.solve_block` seeds the same start).

Per (shape, configuration, backend, precision, N): the first call's wall (compile
included), the warm wall (min over `--repeats` calls after the first, each
`block_until_ready`), microseconds per point, and peak memory. Every backend and
precision is its own process -- `JAX_PLATFORMS` and `jax_enable_x64` are settled
before any array exists -- and the rows are merged into `out/batching.json`, from
which the `.csv` and the `.tex` are rendered on every run. Outputs at N=64 (float64)
and N=1024 (any precision) are kept under `--arrays` (default `~/.cache/functional_process/batching.arrays/`) so a later process can diff
itself against the CPU float64 answer: `rel_norm` is ||a-b||_inf / ||b||_inf over
every output entry and `rel_elem` the elementwise max over entries with
|b| > 1e-12 ||b||_inf.

    G=~/miniconda3/envs/process_port_gpu/bin/python
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    $G paper_tests/batching.py --backend cpu --shape mda
    $G paper_tests/batching.py --backend gpu --shape mda
    $G paper_tests/batching.py --backend gpu --shape sand --precision f32 --sizes 1024
    $G paper_tests/batching.py --backend gpu --hardware    # matmul / elementwise f64:f32
    $G paper_tests/batching.py --backend cpu --shape sand --recipe jacobi --sizes 1,4,16,64
                                                           # the Jacobi cut's 1193-entry block
    $G paper_tests/batching.py --render                    # tables only
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# The backend is settled before `common` imports jax; `common` only `setdefault`s it.
_argv = sys.argv[1:]
_backend = _argv[_argv.index("--backend") + 1] if "--backend" in _argv else "cpu"
os.environ["JAX_PLATFORMS"] = {"cpu": "cpu", "gpu": "cuda"}[_backend]
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from common import (  # noqa: E402
    OUT,
    cut_for,
    fmt,
    provenance,
    stem,
    tex_name,
    write_csv,
    write_json,
)
from cottax.names import PathMap  # noqa: E402
from jax.flatten_util import ravel_pytree  # noqa: E402

from common import CACHE_DIR  # noqa: E402

from functional_process.cottax.architectures import session  # noqa: E402
from functional_process.cottax.architectures.evaluate import (  # noqa: E402
    cold_state,
    ground_truth,
    mda_env,
    mda_schedule,
    seed_block,
    seed_env,
)
from functional_process.cottax.architectures.host_cache import flat_values  # noqa: E402
from functional_process.cottax.architectures.mda import cut_graph  # noqa: E402
from functional_process.cottax.input.indat import graph_for  # noqa: E402

SIZES = (1, 4, 16, 64, 256, 1024, 4096)
CONFIGS = (
    "tests/regression/input_files/stellarator_helias.IN.DAT",
    "tests/regression/input_files/large_tokamak_nof.IN.DAT",
)
BATCHED = (".physics.rmajor", ".physics.b_plasma_toroidal_on_axis")
"""The two boundary inputs the `mda` shape varies (both are schedule inputs of every
configuration here)."""
SPAN = 0.05
"""Half-width of the `mda` grid, relative."""
PERTURBATION = 0.01
"""Half-width of the `sand` shape's design perturbations, relative, uniform."""
WALL_LIMIT = 60.0
"""A warm call slower than this ends the size ladder for that shape."""
JSON = OUT / "batching.json"
ARRAYS = CACHE_DIR / "batching.arrays"
"""Where the N=64 and N=1024 outputs land for the cross-backend diffs: the port's own
cache directory, not `out/` -- 2.5 GB of `.npy` is not a table."""


# ---------------------------------------------------------------- the two shapes


def mda_shape(live):
    """`(single, batched, point, in_axes)` for the `mda` shape.

    `single(env)` runs one point; `batched(env_b)` runs `env_b`, whose two `BATCHED`
    leaves carry a leading axis. `point` is the seeded reference env.
    """
    graph = live.machine_graph if live.machine_graph is not None else graph_for()
    _driven, runnable, schedule, _run = mda_schedule(graph, cut_graph if live.cut is None else live.cut)
    # A recipe's cut copies start from the cold state, as `mda_convergence.py` seeds them.
    cold = None if live.cut is None else cold_state(live.reference.data, live.machine_graph)
    point = PathMap(seed_env(live.reference.data, schedule, runnable, cold))
    varied = [v for v in point if v.spelling in BATCHED]
    if len(varied) != len(BATCHED):
        raise ValueError(f"{live.name}: {BATCHED} are not all schedule inputs")
    axes = PathMap((v, 0 if v in varied else None) for v in point)
    single = jax.jit(schedule.run)
    batched = jax.jit(jax.vmap(schedule.run, in_axes=(axes,)))
    return single, batched, point, varied


def mda_batch(point, varied, n):
    """`point` with its `varied` leaves on a sqrt(n) x sqrt(n) grid of +-`SPAN`."""
    k = int(round(np.sqrt(n)))
    if k * k != n:
        raise ValueError(f"N={n} is not a square")
    factors = np.linspace(1 - SPAN, 1 + SPAN, k) if k > 1 else np.array([1.0])
    grid = np.meshgrid(factors, factors, indexing="ij")
    values = dict(point.items())
    for var, f in zip(varied, grid, strict=True):
        values[var] = jnp.asarray(np.asarray(point[var]) * f.ravel())
    return PathMap(values)


def sand_shape(live):
    """`(single, batched, x0)` for the `sand` shape: the block's fused value+Jacobian at
    one flat design vector, and its vmap.
    """
    if "SAND" not in live.builds:
        # `session.sand` assembles on first use; assembling without solving is the
        # same call minus the solve, and the block is what this shape times.
        live.builds["SAND"] = session.build_sand(
            live.reference, live.machine_graph, live.switch_values, optimiser=live.optimiser,
            cut=live.cut,
        )
    build = live.builds["SAND"]
    drive = build.drive
    base = live.reference.data
    stage_env = mda_env(live.reference, graph=live.machine_graph,
                        **({} if live.cut is None else {"cut": live.cut}))[1]
    seeded, _borrowed = seed_block(build.solve_schedule, drive, base, stage_env, design=build.design_paths)
    context = {}
    for var in drive.context:
        if var in stage_env:
            context[var] = stage_env[var]
        else:
            try:
                context[var] = jnp.asarray(ground_truth(base, var))
            except (AttributeError, KeyError):
                context[var] = jnp.asarray(0.0)
    conditions = drive.condition_map(context)
    x0, unravel = ravel_pytree(tuple(jnp.asarray(seeded[u]) for u in drive.unknowns))

    def both(flat):
        # `host_cache._values_and_jacobian`'s program: the block traced once, primal
        # returned as `jacfwd`'s aux.
        def stacked_twice(f):
            out = flat_values(conditions(*unravel(f)))
            return out, out

        derivative, primal = jax.jacfwd(stacked_twice, has_aux=True)(flat)
        return primal, derivative

    return jax.jit(both), jax.jit(jax.vmap(both)), x0


def sand_batch(x0, n, seed=0):
    """`n` designs, each `x0 * (1 + PERTURBATION * u)`, `u` uniform in [-1, 1]."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1.0, 1.0, size=(n, x0.shape[0]))
    return jnp.asarray(np.asarray(x0)[None, :] * (1.0 + PERTURBATION * u))


# ---------------------------------------------------------------- measurement


def _flat(out) -> np.ndarray:
    leaves = jax.tree_util.tree_leaves(out)
    return np.concatenate([np.asarray(leaf, dtype=np.float64).ravel() for leaf in leaves])


def _peak_memory(device) -> dict:
    row = {}
    try:
        stats = device.memory_stats() or {}
        if "peak_bytes_in_use" in stats:
            row["peak_device_mb"] = stats["peak_bytes_in_use"] / 2**20
        if "bytes_in_use" in stats:
            row["device_mb"] = stats["bytes_in_use"] / 2**20
    except Exception:  # noqa: BLE001 -- a number, not a result
        pass
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM"):
                row["peak_rss_mb"] = int(line.split()[1]) / 1024
    except OSError:
        pass
    return row


def _time(fn, arg, repeats: int) -> dict:
    """First call (compile) and the min of `repeats` warm calls, each blocked."""
    began = time.perf_counter()
    out = jax.block_until_ready(fn(arg))
    first = time.perf_counter() - began
    walls = []
    for _ in range(repeats):
        began = time.perf_counter()
        out = jax.block_until_ready(fn(arg))
        walls.append(time.perf_counter() - began)
    return {"first_call": first, "warm_wall": min(walls), "warm_walls": walls, "_out": out}


def _rel(a: np.ndarray, b: np.ndarray) -> dict:
    """`a` against `b`, normwise and elementwise (see the module docstring)."""
    finite = np.isfinite(a) & np.isfinite(b)
    scale = float(np.max(np.abs(b[finite]))) if finite.any() else 0.0
    if scale == 0.0:
        return {"rel_norm": float(np.max(np.abs(a - b))) if finite.any() else None, "rel_elem": None}
    diff = np.abs(a - b)
    big = finite & (np.abs(b) > 1e-12 * scale)
    return {
        "rel_norm": float(np.max(diff[finite]) / scale),
        "rel_elem": float(np.max(diff[big] / np.abs(b[big]))) if big.any() else None,
        "nonfinite_a": int((~np.isfinite(a)).sum()),
        "nonfinite_b": int((~np.isfinite(b)).sum()),
    }


def _array_path(arrays: Path, shape, name, backend, precision, n, recipe="hand", tag="") -> Path:
    cut = "" if recipe == "hand" else f"_{recipe}"
    tagged = f"_{tag}" if tag else ""
    return arrays / f"{shape}_{name}{cut}_{backend}_{precision}_{n}{tagged}.npy"


def measure(shape: str, path: str, backend: str, precision: str, sizes, repeats: int,
            arrays: Path, tag: str, recipe: str = "hand") -> list[dict]:
    live = session.open_session(path, cut=cut_for(recipe))
    name = stem(str(path))
    device = jax.devices()[0]
    if shape == "mda":
        single, batched, point, varied = mda_shape(live)
        make = lambda n: mda_batch(point, varied, n)  # noqa: E731
        unbatched_arg = point
        entries = len(point)
    else:
        single, batched, x0 = sand_shape(live)
        make = lambda n: sand_batch(x0, n)  # noqa: E731
        unbatched_arg = x0
        entries = int(x0.shape[0])
    dtype = str(jnp.asarray(1.0).dtype)
    rows = []
    stamp = {"shape": shape, "configuration": name, "backend": backend, "precision": precision,
             "cut": recipe, "dtype": dtype, "tag": tag, "device": str(device.device_kind),
             "jax": jax.__version__, "entries": entries}

    def finish(row, timing, n):
        row.update(first_call=timing["first_call"], warm_wall=timing["warm_wall"],
                   compile_s=timing["first_call"] - timing["warm_wall"],
                   us_per_point=timing["warm_wall"] / n * 1e6,
                   warm_walls=timing["warm_walls"], status="ok", **_peak_memory(device))
        flat = _flat(timing["_out"])
        row["n_outputs"] = int(flat.size / n)
        row["nonfinite"] = int((~np.isfinite(flat)).sum())
        if n in (64, 1024):
            arrays.mkdir(parents=True, exist_ok=True)
            np.save(_array_path(arrays, shape, name, backend, precision, n, recipe, tag), flat)
            reference = _array_path(arrays, shape, name, "cpu", "f64", n, recipe)
            if reference.exists() and (backend, precision, tag) != ("cpu", "f64", ""):
                row.update({f"{k}_vs_cpu_f64": v for k, v in _rel(flat, np.load(reference)).items()})

    # The unbatched call: one point, no `vmap`, the baseline the batched rows divide.
    if 1 in sizes:
        row = {**stamp, "N": 1, "vmap": False}
        try:
            finish(row, _time(single, unbatched_arg, repeats), 1)
        except Exception as failure:  # noqa: BLE001 -- a row, not an exit
            row.update(status="FAILED", note=f"{type(failure).__name__}: {failure}"[:300])
        _print(row)
        rows.append(row)
    for n in sizes:
        row = {**stamp, "N": n, "vmap": True}
        try:
            finish(row, _time(batched, make(n), repeats), n)
        except Exception as failure:  # noqa: BLE001
            text = f"{type(failure).__name__}: {failure}"
            oom = "RESOURCE_EXHAUSTED" in text or "out of memory" in text.lower() or "OOM" in text
            row.update(status="OOM" if oom else "FAILED", note=text[:300])
        _print(row)
        rows.append(row)
        if row["status"] != "ok" or row["warm_wall"] > WALL_LIMIT:
            print(f"  stopping the ladder at N={n}: {row['status']}, "
                  f"warm {row.get('warm_wall')}", flush=True)
            break
    return rows


def _print(row):
    print(f"{row['shape']:4} {row['configuration']:20} {row.get('cut', 'hand')[:4]:4} "
          f"{row['backend']:3} {row['precision']} "
          f"N={row['N']:5d} vmap={row['vmap']!s:5} {row['status']:6} "
          f"first {fmt(row.get('first_call'), 2):>8} s  warm {fmt(row.get('warm_wall'), 4):>9} s  "
          f"{fmt(row.get('us_per_point'), 1):>10} us/pt  "
          f"dev {fmt(row.get('peak_device_mb'), 0):>6} MB  rss {fmt(row.get('peak_rss_mb'), 0):>6} MB  "
          f"{row.get('note', '')}", flush=True)


# ---------------------------------------------------------------- the hardware


def hardware(backend: str, repeats: int = 5) -> list[dict]:
    """The backend's float64 against float32 throughput on two kernels that bracket the
    port: a 2048^2 matmul (dense, what the 1/64 FP64 rate is quoted for) and a chain of
    64 elementwise transcendental ops over 2^20 entries (what a scalar graph under
    `vmap` looks like). GFLOP/s and the f64/f32 wall ratio; x64 on, dtypes explicit.
    """
    device = jax.devices()[0]
    rows = []
    n = 2048
    m = 2**20
    chain = 64

    @jax.jit
    def matmul(a):
        return a @ a

    @jax.jit
    def elementwise(x):
        for _ in range(chain):
            x = jnp.sin(x) * 1.0001 + jnp.exp(-x * x) + 0.5
        return x

    # The dense factorisations the tokamak graph carries per point, vmapped: the SVD of
    # `pfcoil/currents.py`'s `gmat[:, :n_groups]`, whose PROCESS padding is `LROW1 = 74`
    # rows, and the same SVD at 32 rows -- cuSOLVER's batched Jacobi SVD stops at 32x32,
    # and above it XLA calls `gesvd` once per matrix. The 6x6 solve stands for
    # `tfcoil/stress.py`'s. Measured because the tokamak MDA's GPU floor is exactly two
    # of the 74-row SVDs per point (`.tokamak.pf_coil.initiation_currents` and
    # `.equilibrium_currents`, 1.27 s + 0.65 s of a 1.99 s batched call at N=1024).
    batch = 1024
    rng = np.random.default_rng(0)
    solve_shape = (6, 6)
    svd_shapes = ((74, 4), (32, 4))
    svd_in = {shape: rng.standard_normal((batch, *shape)) for shape in svd_shapes}
    solve_in = rng.standard_normal((batch, *solve_shape)) + 6 * np.eye(solve_shape[0])
    solve_rhs = rng.standard_normal((batch, solve_shape[0]))

    @jax.jit
    def svd(a):
        return jax.vmap(lambda x: jnp.linalg.svd(x, full_matrices=False)[1])(a)

    @jax.jit
    def solve(ab):
        return jax.vmap(jnp.linalg.solve)(*ab)

    kernels = {
        "matmul_2048": (matmul, lambda dt: jnp.ones((n, n), dt) * 0.001, 2.0 * n**3),
        f"elementwise_{chain}x2^20": (elementwise, lambda dt: jnp.linspace(0, 1, m, dtype=dt), None),
        **{f"svd_{r}x{c}_x{batch}": (svd, lambda dt, s=(r, c): jnp.asarray(svd_in[s], dt), None)
           for r, c in svd_shapes},
        f"solve_{solve_shape[0]}x{solve_shape[1]}_x{batch}": (
            solve, lambda dt: (jnp.asarray(solve_in, dt), jnp.asarray(solve_rhs, dt)), None),
    }
    for kernel, (fn, make, flops) in kernels.items():
        walls = {}
        for dt in ("float64", "float32"):
            x = make(dt)
            t = _time(fn, x, repeats)
            walls[dt] = t["warm_wall"]
            row = {"shape": "hardware", "configuration": kernel, "backend": backend,
                   "precision": "f64" if dt == "float64" else "f32", "dtype": dt, "tag": "",
                   "device": str(device.device_kind), "jax": jax.__version__, "N": 0, "vmap": False,
                   "status": "ok", "first_call": t["first_call"], "warm_wall": t["warm_wall"],
                   "compile_s": t["first_call"] - t["warm_wall"], "warm_walls": t["warm_walls"]}
            if flops:
                row["gflops"] = flops / t["warm_wall"] / 1e9
            if kernel.endswith(f"_x{batch}"):
                row["N"] = batch
                row["vmap"] = True
                row["us_per_point"] = t["warm_wall"] / batch * 1e6
            rows.append(row)
        for row in rows[-2:]:
            row["f64_over_f32"] = walls["float64"] / walls["float32"]
    return rows


# ---------------------------------------------------------------- bookkeeping


def _key(row):
    return (row["shape"], row["configuration"], row.get("cut", "hand"), row["backend"],
            row["precision"], row.get("tag", ""), int(row["N"]), bool(row["vmap"]))


def merge(new_rows: list[dict]) -> list[dict]:
    import json  # noqa: PLC0415

    rows = json.loads(JSON.read_text()) if JSON.exists() else []
    table = {_key(r): r for r in rows}
    for row in new_rows:
        table[_key(row)] = {k: v for k, v in row.items() if not k.startswith("_")}
    rows = sorted(table.values(), key=_key)
    write_json("batching.py", rows)
    return rows


def render(rows: list[dict]):
    """`batching.csv` (every row) and `batching.tex` (shape x configuration x N: CPU and
    GPU us/point, speedup).
    """
    write_csv("batching.py", [{k: v for k, v in r.items() if k != "warm_walls"} for r in rows])
    main_rows = [r for r in rows if not r.get("tag") and r.get("cut", "hand") == "hand"]

    def find(shape, cfg, backend, precision, n, vmap=True):
        for r in main_rows:
            if (r["shape"], r["configuration"], r["backend"], r["precision"], r["N"], r["vmap"]) == (
                shape, cfg, backend, precision, n, vmap
            ):
                return r
        return None

    def us(r):
        if r is None:
            return "--"
        if r["status"] != "ok":
            return r["status"]
        return fmt(r["us_per_point"], 1 if r["us_per_point"] < 100 else 0)

    def sec(r):
        return "--" if r is None or r["status"] != "ok" else fmt(r["compile_s"], 1)

    tex = []
    for shape in ("mda", "sand"):
        cfgs = list(dict.fromkeys(r["configuration"] for r in main_rows if r["shape"] == shape))
        for cfg in cfgs:
            for precision in ("f64", "f32"):
                sizes = sorted({r["N"] for r in main_rows if r["shape"] == shape
                                and r["configuration"] == cfg and r["precision"] == precision})
                for n in sizes:
                    c = find(shape, cfg, "cpu", precision, n)
                    g = find(shape, cfg, "gpu", precision, n)
                    if c is None and g is None:
                        continue
                    speed = (fmt(c["warm_wall"] / g["warm_wall"], 2)
                             if c and g and c["status"] == "ok" and g["status"] == "ok" else "--")
                    label = shape.upper() if precision == "f64" else f"{shape.upper()} (f32)"
                    tex.append([label, tex_name(cfg), str(n), us(c), us(g), speed, sec(c), sec(g)])
    lines = [
        provenance("batching.py"),
        "% shape x configuration x N: warm microseconds per point (min of repeats after the "
        "first call), GPU/CPU speedup of the batched call, and compile seconds (first call minus "
        "warm); f32 rows are x64 off",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"shape & configuration & $N$ & CPU $\mu$s/pt & GPU $\mu$s/pt & GPU/CPU & CPU compile [s] & GPU compile [s] \\",
        r"\midrule",
        *(" & ".join(row) + r" \\" for row in tex),
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ]
    (OUT / "batching.tex").write_text("\n".join(lines))


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--help" in argv or "-h" in argv:
        print(__doc__)
        return 0
    if "--render" in argv:
        import json  # noqa: PLC0415

        render(json.loads(JSON.read_text()))
        return 0
    backend = _backend
    precision = argv[argv.index("--precision") + 1] if "--precision" in argv else "f64"
    jax.config.update("jax_enable_x64", precision == "f64")
    if "--hardware" in argv:
        rows = hardware(backend)
        for row in rows:
            _print(row)
        render(merge(rows))
        return 0
    if precision == "f32":
        # **float32 does not evaluate PROCESS.** The fusion-rate Picard block
        # (`^problem.physics.proton_rate_density.cycle`) overflows: a reaction rate is a
        # density squared, ~1e40 m^-6, past float32's 3.4e38, and optimistix's
        # `throw=True` then raises on the nonfinite iterate. The f32 rows exist to time
        # the *hardware*, so the drivers are told not to throw; their values are inf/nan
        # (`nonfinite` counts them) and a Picard on nan runs to `max_steps`, so the
        # `mda` f32 wall is not comparable with its f64 row -- only `sand`, which has no
        # loop, is.
        import functools  # noqa: PLC0415

        import optimistix as optx  # noqa: PLC0415

        for name in ("fixed_point", "root_find"):
            setattr(optx, name, functools.partial(getattr(optx, name), throw=False))
    shapes = [argv[i + 1] for i, a in enumerate(argv) if a == "--shape"] or ["mda", "sand"]
    chosen = [argv[i + 1] for i, a in enumerate(argv) if a == "--input"] or list(CONFIGS)
    sizes = ([int(s) for s in argv[argv.index("--sizes") + 1].split(",")]
             if "--sizes" in argv else list(SIZES))
    repeats = int(argv[argv.index("--repeats") + 1]) if "--repeats" in argv else 3
    tag = argv[argv.index("--tag") + 1] if "--tag" in argv else ""
    recipe = argv[argv.index("--recipe") + 1] if "--recipe" in argv else "hand"
    arrays = Path(argv[argv.index("--arrays") + 1]) if "--arrays" in argv else ARRAYS
    print(f"jax {jax.__version__} on {jax.devices()}  x64={jax.config.jax_enable_x64}  "
          f"XLA_PYTHON_CLIENT_PREALLOCATE={os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE')}",
          flush=True)
    rows = []
    for path in chosen:
        path = stem(path)
        for shape in shapes:
            try:
                rows += measure(shape, path, backend, precision, sizes, repeats, arrays, tag, recipe)
            except Exception as failure:  # noqa: BLE001 -- a row, not an exit
                row = {"shape": shape, "configuration": stem(path), "backend": backend,
                       "precision": precision, "cut": recipe, "tag": tag, "N": 0, "vmap": False,
                       "status": "FAILED", "note": f"{type(failure).__name__}: {failure}"[:300]}
                _print(row)
                rows.append(row)
            all_rows = merge(rows)
            render(all_rows)
            jax.clear_caches()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
