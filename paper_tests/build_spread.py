"""How much the four rule-closed build outputs that still vary per sample (the
stage check's remaining violations on the `build` table: blanket / first-wall
lifetime, vacuum pump count and duct diameter, primary heat-exchanger count) actually
spread over the belief samples, and what they are worth in cost: every one of them,
the direct cost and coe, per sample at a design.

    $PY paper_tests/build_spread.py [--n 1024] [--robust out/<run>.json] [--table build]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import ouu  # noqa: E402
from common import write_json  # noqa: E402
from cottax.names import PathMap  # noqa: E402

PLACES = (
    ".fwbs.life_fw_fpy", ".fwbs.life_blkt_fpy", ".fwbs.life_div_fpy",
    ".vacuum.n_vac_pumps_high", ".vacuum.dia_vv_vacuum_ducts",
    ".heat_transport.n_primary_heat_exchangers",
    ".costs.concost", ".costs.coe", ".heat_transport.p_plant_electric_net_mw",
)


def main(argv):
    def option(name, default, cast=int):
        return cast(argv[argv.index(name) + 1]) if name in argv else default

    n = option("--n", 1024)
    table = option("--table", "build", str)
    o = ouu.build(n, alpha=0.9, objective="nominal", table=table, pairing="one")
    model = o.model
    run = model.built.problem.traceable.run
    point = dict(model.point.items())
    guesses = [model.guesses[u] for u in model.unknowns]
    outputs = model.built.problem.eager.subgraph.variables
    var_of = {v.spelling: v for v in outputs}
    places = [var_of[p] for p in PLACES if p in var_of]
    missing = [p for p in PLACES if p not in var_of]
    x = o.x0
    source = "deterministic"
    if "--robust" in argv:
        payload = json.loads(Path(option("--robust", "", str)).read_text())
        x, source = ouu.robust_design_of(payload, o.design)

    def one(x_flat, theta_row, start_row):
        values = dict(point)
        values.update(theta_row.items())
        values.update(zip(o.design, [x_flat[j] for j in range(len(o.design))], strict=True))
        values.update(zip(guesses, [start_row[j] for j in range(len(guesses))], strict=True))
        out = run(PathMap(values))
        return jnp.stack([jnp.asarray(out[c], dtype=jnp.float64).reshape(()) for c in places])

    Y = np.asarray(jax.jit(jax.vmap(one, in_axes=(None, 0, 0)))(jnp.asarray(x), o.theta, jnp.asarray(o.starts0)))
    rows = {}
    for j, p in enumerate(places):
        v = Y[:-1, j]
        ok = np.isfinite(v)
        rows[p.spelling] = {
            "nominal": float(Y[-1, j]), "min": float(v[ok].min()), "p5": float(np.percentile(v[ok], 5)),
            "p50": float(np.percentile(v[ok], 50)), "p95": float(np.percentile(v[ok], 95)), "max": float(v[ok].max()),
            "distinct": int(np.unique(np.round(v[ok], 6)).size),
            "relative_spread_p5_p95": float((np.percentile(v[ok], 95) - np.percentile(v[ok], 5)) / abs(Y[-1, j])) if Y[-1, j] else None,
        }
    result = {"n": n, "table": table, "design": source, "x": {v.spelling: float(xi) for v, xi in zip(o.design, x, strict=True)},
              "missing": missing, "rows": rows}
    write_json("ouu", result, f"build_spread_{table}_{source.split()[0]}")
    for k, r in rows.items():
        print(f"{k:48s} nominal {r['nominal']:12.5g}  p5 {r['p5']:12.5g}  p50 {r['p50']:12.5g}  p95 {r['p95']:12.5g}  distinct {r['distinct']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
