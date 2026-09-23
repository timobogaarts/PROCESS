"""Table 5 -- the optimiser's problem over **N designs at once**, on the CPU and on
the GPU: one evaluation and one Jacobian of the block, `vmap`ped, per design.

This is the regime XLA is built for. At N = 1 the block's cost is mostly the runtime
dispatching its hundred-odd kernels (see `iteration.py`); over a batch that cost is
paid once, and what is left is the arithmetic of one pass. The per-design number at
the largest N is therefore the honest cost of the models, and the CPU-to-GPU ratio is
what the card buys on float64 (PROCESS is float64; a consumer GPU runs it at a
fraction of its float32 rate).

    JAX_PLATFORMS=cpu $PY paper_tests/architectures/batched.py --configurations helias_5b
    JAX_PLATFORMS=cuda $PY paper_tests/architectures/batched.py --batches 1 16 256 4096 16384

Each N is its own compilation; the time is the median of `--repeats` warm calls. Past
`--chunk` designs the batch is `lax.map`ped over chunks of one `vmap`, so memory is
bounded by the chunk and N by patience alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402
from cottax.execution.driver import Gaps  # noqa: E402
from cottax.interfaces import PathMap, Schedule  # noqa: E402

from functional_process.cottax.architectures.evaluate import (  # noqa: E402
    ground_truth,
    mda_env,
    seed_block,
)
from functional_process.cottax.architectures.mda import seed_starts  # noqa: E402

BATCHES = (1, 16, 256, 4096)


def block_of(live, arm, build, args):
    """`(f, x)`: the arm's problem as `f(x) -> stacked conditions`, and its start."""
    if live.root_find:
        drive, schedule = build.in_graph.drive, Schedule(build.in_graph.blocking)
    else:
        drive, schedule = build.drive, build.solve_schedule
    cold = live.reference.cold
    stage = mda_env(live.reference, graph=live.machine_graph, data=cold, scheme=bench.SCHEMES[args.scheme])[1]
    seeded, _ = seed_block(schedule, drive, cold, stage, design=set())
    seeded.update(seed_starts(schedule, stage))

    def value(v):
        if v in stage:
            return stage[v]
        if v in seeded:
            return seeded[v]
        try:
            return jnp.asarray(ground_truth(cold, v))
        except (AttributeError, KeyError):
            return jnp.asarray(0.0)

    # As **gaps**, the way the arm's own driver reads the block: the seam hands both
    # sides of every relation and `Gaps` is the level that subtracts.
    cm = Gaps(drive.condition_map(PathMap({v: value(v) for v in drive.context})))
    x, unravel = ravel_pytree(tuple(jnp.asarray(seeded[u]) for u in drive.unknowns))

    def f(flat):
        return ravel_pytree(cm(*unravel(flat)))[0]

    return f, x, len(drive.nodes)


def designs(x, n):
    """`n` designs around `x`: the start scaled by up to a tenth of a percent, so no
    two lanes are identical and every lane stays where the start converges."""
    return x[None, :] * (1.0 + 1e-3 * jnp.linspace(0.0, 1.0, n)[:, None])


def chunked(f, chunk):
    """`vmap(f)` over `chunk` designs at a time, `lax.map`ped over the batch: memory
    bounded by the chunk, the per-design cost that of the vmap, any N."""
    inner = jax.vmap(f)

    def over(X):
        n = X.shape[0]
        if n <= chunk:
            return inner(X)
        full, rest = divmod(n, chunk)
        out = jax.lax.map(inner, X[: full * chunk].reshape(full, chunk, -1))
        out = out.reshape(full * chunk, *out.shape[2:])
        return out if not rest else jnp.concatenate([out, inner(X[full * chunk:])])

    return over


def main():
    args = bench.arguments(__doc__, optimiser=False, batches=True)
    platform = jax.default_backend()
    rows = []
    for name in args.configurations:
        live = bench.open_session(name, args)
        for arm in [a for a in live.arms if a != "MDA"]:
            f, x, block = block_of(live, arm, live.assemble(arm), args)
            evaluate = jax.jit(chunked(f, args.chunk))
            jacobian = jax.jit(chunked(jax.jacfwd(f), args.chunk))
            for n in args.batches:
                X = designs(x, n)
                try:
                    _, compile_e = bench.timed(evaluate, X)
                    _, compile_j = bench.timed(jacobian, X)
                    e = bench.median_seconds(lambda: evaluate(X), args.repeats)
                    j = bench.median_seconds(lambda: jacobian(X), args.repeats)
                except Exception as refusal:  # noqa: BLE001 -- out of memory, most likely
                    print(f"{name} {arm} N={n}: {type(refusal).__name__}: {str(refusal)[:80]}")
                    break
                rows.append({
                    "configuration": name, "arm": arm, "platform": platform,
                    "N": n, "chunk": min(n, args.chunk), "unknowns": int(x.size), "block_nodes": block,
                    "evaluate_us_per_design": 1e6 * e / n,
                    "jacobian_us_per_design": 1e6 * j / n,
                    "evaluate_ms": 1e3 * e, "jacobian_ms": 1e3 * j,
                    "compile_s": compile_e + compile_j,
                })
                print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in rows[-1].items()})
    bench.write("batched.py", rows, f"batched_{args.scheme}_{platform}")


if __name__ == "__main__":
    main()
