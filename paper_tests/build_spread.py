"""How much the four rule-closed build outputs that still vary per sample (the
stage check's remaining violations on the `build` table: blanket / first-wall
lifetime, vacuum pump count and duct diameter, primary heat-exchanger count) actually
spread over the belief samples, and what they are worth in cost: every one of them,
the direct cost and coe, per sample at a design.

The two-stage problem is `ouu.two_stage` (through `ouu.py`'s `Choice` / `build`, so
the tables and the deterministic start are the OUU study's); the batched program here
is a small local `vmap` of the closed MDA (`TwoStage.whole`) over the sample set,
because the columns wanted -- lifetimes, pump counts -- are not among the ones
`ouu.make`'s `run_batch` returns.

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
import ouu as ouu_cli  # noqa: E402
from common import option, write_json  # noqa: E402
from cottax.names import PathMap  # noqa: E402

PLACES = (
    ".fwbs.life_fw_fpy", ".fwbs.life_blkt_fpy", ".costs.life_div_fpy",
    ".vacuum.n_vac_pumps_high", ".vacuum.dia_vv_vacuum_ducts",
    ".heat_transport.n_primary_heat_exchangers",
    ".costs.concost", ".costs.coe", ".heat_transport.p_plant_electric_net_mw",
)


def main(argv):
    n = option(argv, "--n", 1024)
    table = option(argv, "--table", "build", str)
    choice = ouu_cli.Choice.from_argv(argv, n=n, alpha=0.9, objective="nominal", table=table, pairing="one")
    model = ouu_cli.build(choice)
    run = model.whole.run
    point = dict(model.point.items())
    var_of = {v.spelling: v for v in model.whole.subgraph.graph.variables}
    places = [var_of[p] for p in PLACES if p in var_of]
    missing = [p for p in PLACES if p not in var_of]
    x = model.x0
    source = "deterministic"
    if "--robust" in argv:
        payload = json.loads(Path(option(argv, "--robust", "", str)).read_text())
        x, source = ouu_cli.robust_design_of(payload, model.design)

    def one(x_flat, theta_row, start_row):
        values = dict(point)
        values.update(theta_row.items())
        values.update(zip(model.design, [x_flat[j] for j in range(len(model.design))], strict=True))
        values.update(zip(model.guesses, [start_row[j] for j in range(len(model.guesses))], strict=True))
        out = run(PathMap(values))
        return jnp.stack([jnp.asarray(out[c], dtype=jnp.float64).reshape(()) for c in places])

    y = np.asarray(jax.jit(jax.vmap(one, in_axes=(None, 0, 0)))(jnp.asarray(x), model.theta, jnp.asarray(model.starts0)))
    rows = {}
    for j, p in enumerate(places):
        v = y[:-1, j]
        ok = np.isfinite(v)
        rows[p.spelling] = {
            "nominal": float(y[-1, j]), "min": float(v[ok].min()), "p5": float(np.percentile(v[ok], 5)),
            "p50": float(np.percentile(v[ok], 50)), "p95": float(np.percentile(v[ok], 95)), "max": float(v[ok].max()),
            "distinct": int(np.unique(np.round(v[ok], 6)).size),
            "relative_spread_p5_p95": float((np.percentile(v[ok], 95) - np.percentile(v[ok], 5)) / abs(y[-1, j])) if y[-1, j] else None,
        }
    result = {"n": n, "table": table, "held": list(model.held), "design": source,
              "x": {v.spelling: float(xi) for v, xi in zip(model.design, x, strict=True)},
              "missing": missing, "rows": rows}
    write_json("ouu", result, f"build_spread_{table}_{source.split()[0]}")
    for k, r in rows.items():
        print(f"{k:48s} nominal {r['nominal']:12.5g}  p5 {r['p5']:12.5g}  p50 {r['p50']:12.5g}  p95 {r['p95']:12.5g}  distinct {r['distinct']}")
    return 0


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        raise SystemExit(0)
    raise SystemExit(main(sys.argv[1:]))
