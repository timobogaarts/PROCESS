"""Optimisation under uncertainty (OUU) on the closed stellarator MDA -- the command
line, the file handling and the plots over `architectures.ouu`.

The formulation, the batched program and the outer solve are the port's
(`functional_process/cottax/architectures/ouu.py`, whose docstring states them):
`two_stage` assembles the two-stage problem -- the power balance `c2` closed by the
density per belief sample (`kinds.PAIRINGS[--pairing]`), the graph split at the
sampled leaves and the first stage hoisted out of the batch, the design the file's
`ixc` minus what is closed and what is a belief, one scrambled Sobol' set of N
samples with the nominal appended; `make` compiles it (`--jac fwd|rev`, `--chunk`);
`outer` states the outer problem as a graph -- the statistics node (`f`, the CVaRs,
the failed fraction) and an `Optimise` driven by the `BoxedSlsqpDriver` with move
limits and warm starts through the batched program's memo; `solve` runs it;
`report`, `evaluate`, `summarise`, `per_sample`, `design_table`, `fresh_sample` and
`sensitivity` are what the evidence reads.

Every run starts from PROCESS's own converged design (`common.process_reference`,
`common.deterministic_values`) -- the deterministic optimum the robust design is
compared against -- unless `--start` names another run's JSON.

The belief table: `kinds.BELIEFS`, `--table new` as stated (hfact lognormal
`--hfact-sigma`, 0.10), `--table build` with `kinds.BUILD_LEAVES` held at their
nominal (every sample has one build; `stage_check.py`), `--table old` the 2026-09-16
distributions (`uq.beliefs_for("old")`); `--inputs physics` (the default) holds the
economic rows (`kinds.ECONOMIC`) at their nominal too, `--inputs all` samples them.
`held=` is what `two_stage` is handed.

    PY=~/miniconda3/envs/process_port/bin/python; export JAX_PLATFORMS=cpu
    $PY paper_tests/ouu.py --smoke [--n 256] [--alpha 0.9] [--max-iter 200] [--objective levelised]
                           [--pairing one] [--table build] [--inputs physics] [--with-c16 [--alpha16 0.5]]
                           [--te-recourse K --te-range lo,hi] [--jac fwd] [--chunk C] [--tol 1e-4]
                           [--ftol 1e-6] [--gtol 1e-4] [--move-limit 0.25] [--start RUN.json]
                           [--evidence [--fresh-seed 1]] [--name STEM] [--tag T]
    $PY paper_tests/ouu.py --evidence --robust out/<run>.json [--alpha 0.9] [--fresh-seed 1] [--tag T]
    $PY paper_tests/ouu.py --sweep-te --robust out/<run>.json [--points 16]
    $PY paper_tests/ouu.py --decompose [--n 256] [--robust out/ouu_smoke.json] [--name STEM]
    $PY paper_tests/ouu.py --scaling [--sizes 64,256,1024,4096] [--modes fwd,jacrev,jacfwd] [--fresh]
    $PY paper_tests/ouu.py --measure --n 256 --mode jacrev    # one configuration, one process
    $PY paper_tests/ouu.py --plot [--from paper_tests/out_cluster]

Outputs: `out/<name>.json` (`ouu_smoke` by default; `cluster/ouu_summary.py` reads
`out_cluster/ouu2_alpha*.json` in this layout), `out/ouu_evidence_<alpha><tag>.{json,npz}`,
`out/ouu_sweep_te_<run>.json`, `out/ouu_decomposition.json`, `out/ouu_scaling.{csv,json}`,
`out/ouu_price_of_robustness.png`, `out/ouu_histograms_<alpha>.png`, `out/ouu_logs/*.log`.
"""

from __future__ import annotations

import dataclasses
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)  # before any array: PROCESS is float64

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from common import OUT, deterministic_values, option, write_csv, write_json  # noqa: E402

from functional_process.configurations import kinds  # noqa: E402
from functional_process.cottax.architectures import beliefs as beliefs_  # noqa: E402
from functional_process.cottax.architectures import ouu, session  # noqa: E402
from functional_process.cottax.architectures.drivers import BOXED_CONVERGED  # noqa: E402

NAME = "stellarator_helias"
LOGS = OUT / "ouu_logs"
INPUT_SETS = ("all", "physics")
TABLES = ("new", "old", "build")
HFACT_SIGMA = 0.10
C16 = kinds.C16
TE = kinds.TE
G_TOL = ouu.G_TOL


# ---------------------------------------------------------------- the problem


@dataclasses.dataclass(frozen=True)
class Choice:
    """What the command line chose, in the port's terms."""

    n: int
    alpha: float
    seed: int
    with_c16: bool
    inputs: str
    objective: str
    table: str
    hfact_sigma: float
    pairing: str
    alpha16: float | None
    te_recourse: int
    te_range: tuple | None

    @property
    def held(self) -> tuple[str, ...]:
        """The rows of `kinds.BELIEFS` held at their nominal."""
        held: tuple[str, ...] = ()
        if self.table == "build":
            held += kinds.BUILD_LEAVES
        if self.inputs == "physics":
            held += kinds.ECONOMIC
        return held

    @property
    def beliefs(self) -> tuple:
        """The table, with `hfact`'s sigma as asked."""
        import uq  # noqa: PLC0415 -- beside this file

        rows = uq.beliefs_for("old" if self.table == "old" else "nominal")
        if self.table != "old":
            rows = tuple(
                dataclasses.replace(b, a=self.hfact_sigma) if b.path == ".physics.hfact" else b
                for b in rows
            )
        return rows

    @classmethod
    def from_argv(cls, argv, payload: dict | None = None, **overrides) -> Choice:
        """The choice off `argv`, defaulting to a run's JSON `payload` where given."""
        p = payload or {}

        def pick(name, key, default, cast=int):
            return option(argv, name, p.get(key, default), cast)

        chosen = dict(
            n=pick("--n", "n", 256),
            alpha=pick("--alpha", "alpha", ouu.ALPHA, float),
            seed=pick("--seed", "seed", 0),
            with_c16="--with-c16" in argv or bool(p.get("with_c16", False)),
            inputs=pick("--inputs", "inputs", "physics", str),
            objective=pick("--objective", "objective", "levelised", str),
            table=pick("--table", "table", "new", str),
            hfact_sigma=pick("--hfact-sigma", "hfact_sigma", HFACT_SIGMA, float),
            pairing=pick("--pairing", "pairing", "one", str),
            alpha16=option(argv, "--alpha16", p.get("alpha16"), float) if "--alpha16" in argv else p.get("alpha16"),
            te_recourse=pick("--te-recourse", "te_recourse", 0),
            te_range=(
                tuple(float(v) for v in option(argv, "--te-range", "", str).split(","))
                if "--te-range" in argv
                else ((p["te_grid"][0], p["te_grid"][-1]) if p.get("te_grid") else None)
            ),
        )
        chosen.update(overrides)
        if chosen["inputs"] not in INPUT_SETS:
            raise SystemExit(f"--inputs {chosen['inputs']!r}; one of {INPUT_SETS}")
        if chosen["table"] not in TABLES:
            raise SystemExit(f"--table {chosen['table']!r}; one of {TABLES}")
        return cls(**chosen)


def build(choice: Choice) -> ouu.TwoStage:
    """`ouu.two_stage` on the stellarator at PROCESS's deterministic design."""
    live = session.open_session(NAME)
    design_values, closing_values = deterministic_values(live, choice.pairing)
    return ouu.two_stage(
        live,
        pairing=choice.pairing,
        beliefs=choice.beliefs,
        held=choice.held,
        alpha=choice.alpha,
        objective=choice.objective,
        n=choice.n,
        seed=choice.seed,
        te_recourse=choice.te_recourse,
        te_range=choice.te_range,
        with_c16=choice.with_c16,
        alpha16=choice.alpha16,
        design_values=design_values,
        closing_values=closing_values,
    )


def hfact_sigma_of(model: ouu.TwoStage) -> float:
    return next((b.a for b in model.beliefs if b.path == ".physics.hfact"), 0.0)


def robust_design_of(payload: dict, design: tuple) -> tuple[np.ndarray, str]:
    """The robust design a run's JSON records: `robust` where the run wrote one, else
    its best feasible model call, else the final point."""
    if payload.get("robust"):
        return np.array([payload["robust"]["x"][v.spelling] for v in design]), payload["robust"]["source"]
    if payload.get("best_feasible"):
        return np.array(payload["best_feasible"]["x"], dtype=float), "best_feasible"
    return np.array([payload["final"]["x"][v.spelling] for v in design]), "final"


def nominal_block(model: ouu.TwoStage) -> dict:
    """The deterministic design's answer at the nominal inputs (the primed env)."""
    out, var_of = model.nominal_out, model.var_of
    return {
        "coe_dollar_per_mwh": float(np.asarray(out[var_of[ouu.COE]])),
        "objf": float(np.asarray(out[model.closed.report["objective"]])),
        "x": model.x0.tolist(),
        "inequalities_at_nominal": {
            c.spelling: float(np.asarray(out[c])) for c in model.closed.report["inequalities"]
        },
    }


# ---------------------------------------------------------------- the SLSQP run


def smoke(choice: Choice, max_iter: int, eps: float, delta: float = 0.25, jac: str = "fwd",
          chunks: int | None = None, tol: float = 1e-4, ftol: float = 1e-6, gtol: float = 1e-4,
          name: str = "ouu_smoke", with_evidence: bool = False, fresh_seed: int = 1, tag: str = "",
          max_outer: int = 60, delta_min: float = 1e-3, ftol_outer: float = 1e-5,
          start: Path | None = None) -> dict:
    model = build(choice)
    fns = ouu.make(model, jac=jac, chunks=chunks)
    n, alpha = model.n, model.alpha
    x0, starts = model.x0, model.starts0
    if start is not None:  # a second start, from another run's robust design, warm along the way
        x0, _ = robust_design_of(json.loads(Path(start).read_text()), model.design)
        _y_cold, y_warm, timing = ouu.evaluate(fns, x0, model.theta, model.starts0, x_from=model.x0)
        u0, u1 = fns["layout"]["c_u"]
        conv = ouu.valid_rows(fns["layout"], y_warm)
        starts = np.where(conv[:, None], y_warm[:, u0:u1], model.starts0)
        print(f"start {start}: continuation from the deterministic design in {timing['calls']} calls, "
              f"failed {timing['failed_cold']:.3f} cold, path failed {timing['path_failed']}", flush=True)
    outer_entries: list = []
    built = ouu.outer(
        model, fns, eps=eps, max_iter=max_iter, max_outer=max_outer, delta=delta, delta_min=delta_min,
        tolerance=ftol, tol=tol, ftol_outer=ftol_outer, gtol=gtol, callback=outer_entries.append,
    )
    program = built.program
    n_g = model.n_g
    print(f"built in {model.build_s:.1f} s: N {n} (+ nominal row), alpha {alpha} (m = {model.m}), "
          f"objective {model.objective}, inputs {choice.inputs} ({len(model.beliefs)} rows, "
          f"{len(model.held)} held), table {choice.table} (hfact sigma {hfact_sigma_of(model)}), "
          f"pairing {model.pairing} ({len(model.design)} design places: "
          f"{[v.spelling.rsplit('.', 1)[-1] for v in model.design]}"
          f"{'' if model.te_grid is None else f', T_e recourse on {len(model.te_grid)} points'}), "
          f"first stage {len(model.stages.first)} of {model.stages.n_nodes} nodes, "
          f"jac {jac}, chunks {chunks}, backend {jax.default_backend()}", flush=True)

    # The forward call alone, timed: compile, then warm.
    program.starts = jnp.asarray(starts)
    began = time.perf_counter()
    program.forward(x0)
    forward_compile_s = time.perf_counter() - began
    walls = []
    for _ in range(3):
        began = time.perf_counter()
        program.forward(x0)
        walls.append(time.perf_counter() - began)
    forward_warm_s = min(walls)
    print(f"forward call: compile {forward_compile_s:.1f} s, warm {forward_warm_s * 1e3:.1f} ms "
          f"({1e6 * forward_warm_s / n:.2f} us/sample)", flush=True)

    # The first fused value + Jacobian call: the compile, and the start's statistics.
    first = program.at(x0)
    v0, j0, d0, first_call_s = first.values, first.jacobian, first.diagnostics, first.seconds
    print(f"first fused value+jac{jac} call: {first_call_s:.1f} s (compile); f0 {v0[0]:.4f}, "
          f"max CVaR g {v0[1:1 + n_g].max():.4f}, failed {v0[-1]:.4f}", flush=True)

    def echo(entry):
        print(f"[outer {entry['outer']}] {entry['message']}; nit {entry['nit']}, box +-{entry['move_limit']:.3g}, "
              f"on move limit: {entry['on_move_limit']}, feasible: {entry['feasible']}, moved {entry['moved']:.2e}, "
              f"f {entry['f']:.4f}, max CVaR g {entry['max_ie']:.4f}; {entry['action']}", flush=True)

    outer_entries.clear()
    x, out, wall = ouu.solve(built, x0=x0, starts=starts)
    for entry in outer_entries:
        echo(entry)
    rep = ouu.report(built, out)
    converged = rep["converged"]
    reason = outer_entries[-1]["action"] if outer_entries else ("converged" if converged else "no SLSQP call")
    final = program.at(x)
    vf, df = final.values, final.diagnostics
    names = list(model.names)
    g0, gf = v0[1:1 + n_g], vf[1:1 + n_g]
    final_feasible = bool(gf.max() <= gtol and vf[-1] <= eps) if n_g else bool(vf[-1] <= eps)
    active = [names[i] for i in range(n_g) if abs(gf[i]) < 1e-3]
    violated = [names[i] for i in range(n_g) if gf[i] > 1e-3]
    at_bound = [model.design[i].spelling for i in range(len(x))
                if abs(x[i] - model.lower[i]) < 1e-6 * max(1, abs(model.lower[i]))
                or abs(x[i] - model.upper[i]) < 1e-6 * max(1, abs(model.upper[i]))]
    source = "final" if final_feasible else ("incumbent (best feasible call)" if rep["best_feasible"] else "final (infeasible)")
    model_walls = sorted(t["model_s"] for t in program.trace[1:]) or [first_call_s]
    payload = {
        "n": n, "alpha": alpha, "m_worst": model.m, "alpha16": model.alpha16, "m16": model.m16,
        "te_recourse": choice.te_recourse,
        "te_grid": None if model.te_grid is None else model.te_grid.tolist(), "eps_failed": eps,
        "with_c16": choice.with_c16, "warm_starts": True,
        "objective": model.objective, "inputs": choice.inputs, "n_inputs": len(model.beliefs),
        "table": choice.table, "hfact_sigma": hfact_sigma_of(model),
        "hfact_belief": beliefs_.hfact_belief(hfact_sigma_of(model)), "jac": jac, "chunks": chunks,
        "pairing": model.pairing, "closed": {c.spelling: v.spelling for c, v in model.closed.pairings.items()},
        "uncertain": [b.path for b in model.beliefs], "held": list(model.held), "dropped": list(model.dropped),
        "started_from": str(start) if start is not None else "deterministic", "x_start": np.asarray(x0).tolist(),
        "max_iter": max_iter, "move_limit": delta, "tol": tol, "ftol": ftol, "gtol": gtol, "seed": model.seed,
        "backend": jax.default_backend(),
        "device": str(jax.local_devices()[0].device_kind) if jax.local_devices() else None,
        "design": [v.spelling for v in model.design],
        "ixc": list(model.ixc),
        "bounds": {v.spelling: [float(lo), float(hi)] for v, lo, hi in zip(model.design, model.lower, model.upper, strict=True)},
        "deterministic": nominal_block(model),
        "start": {"f": float(v0[0]), "g_cvar": dict(zip(names, g0.tolist(), strict=True)), "failed_fraction": float(v0[-1]),
                  "diagnostics": d0, "jac_f": j0[0].tolist()},
        "final": {"f": float(vf[0]), "g_cvar": dict(zip(names, gf.tolist(), strict=True)), "failed_fraction": float(vf[-1]),
                  "feasible": final_feasible,
                  "x": dict(zip([v.spelling for v in model.design], x.tolist(), strict=True)),
                  "x_over_x0": (x / model.x0).tolist(), "diagnostics": df, "active": active, "violated": violated,
                  "design_at_bound": at_bound},
        "robust": {"x": rep["x"], "source": source, "f": rep["f"]},
        "converged": converged, "reason": reason,
        "scipy": {"success": converged, "status": rep["status"], "message": reason,
                  "major_iterations": rep["steps"], "outer_calls": len(outer_entries)},
        "outer": outer_entries,
        "best_feasible": rep["best_feasible"],
        "model_calls": rep["model_calls"], "compile_s": first_call_s, "wall_s": wall, "build_s": model.build_s,
        "timing": {"forward_compile_s": forward_compile_s, "forward_warm_s": forward_warm_s,
                   "forward_us_per_sample": 1e6 * forward_warm_s / n,
                   "fused_compile_s": first_call_s, "fused_warm_median_s": float(np.median(model_walls)),
                   "fused_warm_min_s": float(model_walls[0]), "fused_over_forward": float(np.median(model_walls) / forward_warm_s)},
        "trace": program.trace,
    }
    write_json("ouu.py", payload, name=name)
    print(json.dumps({k: v for k, v in payload.items() if k not in ("trace", "bounds", "outer")}, indent=1, default=str), flush=True)
    print(f"[done] converged: {converged} ({reason}; driver status {rep['status']}, {BOXED_CONVERGED} = converged); "
          f"{rep['steps']} major iterations in {len(outer_entries)} outer calls, {rep['model_calls']} model calls, "
          f"{wall:.1f} s; robust design from {source}, f {rep['f']:.4f}", flush=True)
    if with_evidence:
        evidence(model, fns, model.x0, x, fresh_seed, f"ouu_evidence_{alpha:g}{tag}", source, choice,
                 extra={"run": name, "converged": converged, "reason": reason, "robust_f": rep["f"]})
    return payload


# ---------------------------------------------------------------- evidence


def evidence(model: ouu.TwoStage, fns: dict, x_det: np.ndarray, x_rob: np.ndarray, fresh_seed: int,
             name: str, source: str, choice: Choice, extra: dict | None = None) -> dict:
    """Both designs on the fixed sample set and on a fresh Sobol' set: the summaries,
    the design deltas, and the per-sample columns as `<name>.npz`."""
    layout = fns["layout"]
    _theta_f, theta_fresh, starts_fresh = ouu.fresh_sample(model, fresh_seed)
    sets = {"fixed": (model.theta, model.starts0), "fresh": (theta_fresh, starts_fresh)}
    designs = {"deterministic": np.asarray(x_det, float), "robust": np.asarray(x_rob, float)}
    result = {
        "n": model.n, "alpha": model.alpha, "seed": model.seed, "fresh_seed": fresh_seed, "inputs": choice.inputs,
        "objective": model.objective, "table": choice.table, "hfact_sigma": hfact_sigma_of(model),
        "hfact_belief": beliefs_.hfact_belief(hfact_sigma_of(model)),
        "robust_source": source, "backend": jax.default_backend(),
        "constraints": list(model.names),
        "design": ouu.design_table(model, x_det, x_rob),
        "sets": {}, "timing": {},
    }
    if extra:
        result.update(extra)
    columns = {}
    for set_name, (theta, starts0) in sets.items():
        result["sets"][set_name] = {}
        for design_name, x in designs.items():
            y_cold, y_warm, timing = ouu.evaluate(
                fns, x, theta, starts0, x_from=None if design_name == "deterministic" else designs["deterministic"]
            )
            summary = ouu.summarise(model, layout, y_warm)
            summary["failed_fraction_cold_start"] = timing["failed_cold"]
            summary["continuation"] = {"calls": timing["calls"], "path_failed": timing["path_failed"]}
            result["sets"][set_name][design_name] = summary
            result["timing"][f"{set_name}/{design_name}"] = timing
            for key, value in ouu.per_sample(layout, y_warm).items():
                columns[f"{set_name}/{design_name}/{key}"] = value
            print(f"[evidence] {set_name:5s} {design_name:13s}: failed {summary['failed_fraction']:.4f}  "
                  f"feasible {summary['feasible_fraction']:.4f}  any violation {summary['any_violation_fraction']:.4f}  "
                  f"coe|feasible mean {_fmt(summary['coe_where_feasible']['mean'])}  "
                  f"coe|net>0 mean {_fmt(summary['mean_coe_where_converged_and_positive_net'])}  "
                  f"nominal coe {summary['nominal']['coe']:.2f}", flush=True)
    write_json("ouu.py", result, name=name)
    np.savez_compressed(OUT / f"{name}.npz", **columns)
    return result


def sweep_te(model: ouu.TwoStage, fns: dict, x_rob: np.ndarray, name: str, points: int = 16, source: str = "") -> dict:
    """What a per-sample choice of `temp_plasma_electron_vol_avg_kev` would gain at the
    robust design: T_e swept over `points` values between its bounds, every other
    design place at `x_rob`, each batch warm-started from the previous point's roots,
    and per sample the cheapest point that is feasible (converged, net > 0, every g <= 0
    -- the graph's inequalities, and `c16` where it is one). The shared T_e's own row
    is the `robust` reference. Since the shared T_e is a restriction of the recourse,
    the gain reported is a **lower bound** on the value of T_e recourse."""
    run, layout = ouu.jitted(fns, "run_batch"), fns["layout"]
    u0, u1 = layout["c_u"]
    c_coe, c_net, c_conv = layout["c_coe"], layout["c_net"], layout["c_conv"]
    g0, g1 = layout["c_g"]
    te = next((i for i, v in enumerate(model.design) if v.spelling == TE), None)
    if te is None:
        return {"skipped": "T_e is not a design place here"}
    lo, hi = float(model.lower[te]), float(model.upper[te])
    grid = np.linspace(lo, hi, points)
    # The reference: the robust design itself, warm along a continuation from the deterministic one.
    _y_cold, y_ref, timing = ouu.evaluate(fns, x_rob, model.theta, model.starts0, x_from=model.x0)
    y_ref = np.asarray(y_ref)
    names = list(model.names)
    graph_only = [i for i, c in enumerate(model.constraints) if c.spelling != C16]

    def feasible_of(y, cols):
        conv = ouu.valid_rows(layout, y)
        return conv & (y[:, c_net] > 0) & np.all(y[:, g0:g1][:, cols] <= G_TOL, axis=1), conv

    def walk(order):
        """The grid walked in `order` from the reference roots; rows [points, N + 1, columns]."""
        starts = np.where((y_ref[:, c_conv] > 0.5)[:, None], y_ref[:, u0:u1], model.starts0)
        rows = {}
        for k in order:
            x = x_rob.copy()
            x[te] = grid[k]
            y = np.asarray(jax.block_until_ready(run(jnp.asarray(x), model.theta, jnp.asarray(starts))))
            conv = y[:, c_conv] > 0.5
            starts = np.where(conv[:, None], y[:, u0:u1], starts)
            rows[k] = y
        return rows

    # Down from the robust T_e and up from it, so every point is reached by continuation.
    k_rob = int(np.argmin(np.abs(grid - x_rob[te])))
    rows = walk(range(k_rob, -1, -1))
    rows.update(walk(range(k_rob, points)))
    ys = np.stack([rows[k] for k in range(points)])  # [points, N + 1, columns]
    n = ys.shape[1] - 1
    result = {"name": name, "source": source, "grid_kev": grid.tolist(), "t_e_shared_kev": float(x_rob[te]),
              "n": n, "alpha": model.alpha, "constraints": names, "reference_continuation": timing}
    for label, cols in (("all_constraints", list(range(len(names)))), ("graph_inequalities", graph_only)):
        feas_ref, _conv_ref = feasible_of(y_ref[:-1], cols)
        feas = np.stack([feasible_of(ys[k, :-1], cols)[0] for k in range(points)])  # [points, N]
        coe = np.where(feas, ys[:, :-1, c_coe], np.inf)
        best = np.argmin(coe, axis=0)  # per sample
        any_feasible = np.isfinite(coe[best, np.arange(n)])
        coe_best = coe[best, np.arange(n)]
        gained = feas_ref & any_feasible
        result[label] = {
            "feasible_fraction_shared": float(feas_ref.mean()),
            "feasible_fraction_swept": float(any_feasible.mean()),
            "coe_mean_where_feasible_shared": float(y_ref[:-1, c_coe][feas_ref].mean()) if feas_ref.any() else None,
            "coe_mean_where_feasible_swept": float(coe_best[any_feasible].mean()) if any_feasible.any() else None,
            "coe_mean_over_samples_feasible_both": {
                "shared": float(y_ref[:-1, c_coe][gained].mean()) if gained.any() else None,
                "swept": float(coe_best[gained].mean()) if gained.any() else None,
                "count": int(gained.sum()),
            },
            "chosen_t_e_kev": {"p5": float(np.percentile(grid[best[any_feasible]], 5)) if any_feasible.any() else None,
                               "p50": float(np.percentile(grid[best[any_feasible]], 50)) if any_feasible.any() else None,
                               "p95": float(np.percentile(grid[best[any_feasible]], 95)) if any_feasible.any() else None,
                               "at_shared_fraction": float(np.mean(best[any_feasible] == k_rob)) if any_feasible.any() else None},
            "feasible_fraction_per_grid_point": feas.mean(axis=1).tolist(),
            "failed_fraction_per_grid_point": [float(1.0 - feasible_of(ys[k, :-1], cols)[1].mean()) for k in range(points)],
        }
    write_json("ouu", result, name)
    a = result["all_constraints"]
    print(f"[sweep_te] {name}: shared T_e {x_rob[te]:.2f} keV feasible {a['feasible_fraction_shared']:.3f} -> swept {a['feasible_fraction_swept']:.3f}; "
          f"coe|feasible {a['coe_mean_where_feasible_shared']} -> {a['coe_mean_where_feasible_swept']}; "
          f"on samples feasible both ways ({a['coe_mean_over_samples_feasible_both']['count']}): "
          f"{a['coe_mean_over_samples_feasible_both']['shared']} -> {a['coe_mean_over_samples_feasible_both']['swept']}", flush=True)
    return result


def _fmt(v) -> str:
    return "--" if v is None else f"{v:.2f}"


def decompose(choice: Choice, robust: Path, name: str = "ouu_decomposition") -> dict:
    """The deterministic design and a run's robust design on one N-sample set: every
    measure side by side (`out/<name>.json`)."""
    model = build(choice)
    fns = ouu.make(model)
    payload = json.loads(Path(robust).read_text())
    x_rob, source = robust_design_of(payload, model.design)
    rows = {}
    for label, x in (("deterministic", model.x0), ("robust", x_rob)):
        _y_cold, y_warm, timing = ouu.evaluate(fns, x, model.theta, model.starts0, x_from=None if label == "deterministic" else model.x0)
        rows[label] = ouu.summarise(model, fns["layout"], y_warm)
        rows[label]["failed_fraction_cold_start"] = timing["failed_cold"]
        rows[label]["continuation"] = {"calls": timing["calls"], "path_failed": timing["path_failed"]}
    result = {
        "n": model.n, "seed": model.seed, "inputs": choice.inputs, "alpha": model.alpha, "table": choice.table,
        "hfact_sigma": hfact_sigma_of(model), "hfact_belief": beliefs_.hfact_belief(hfact_sigma_of(model)),
        "n_inputs": len(model.beliefs),
        "belief_table": [dataclasses.asdict(b) for b in model.beliefs], "held": list(model.held),
        "robust_from": str(robust), "robust_source": source,
        "robust_f_recorded": payload.get("robust", payload.get("best_feasible") or {}).get("f"),
        "deterministic_coe_at_nominal": nominal_block(model)["coe_dollar_per_mwh"],
        "design": ouu.design_table(model, model.x0, x_rob),
        "rows": rows,
    }
    write_json("ouu.py", result, name=name)
    print(f"table {choice.table}, hfact sigma {hfact_sigma_of(model)}, inputs {choice.inputs} ({len(model.beliefs)} rows), "
          f"N {model.n}, robust design from {robust} ({source})")
    print(decomposition_table(result))
    return result


def decomposition_table(result: dict) -> str:
    d, r = result["rows"]["deterministic"], result["rows"]["robust"]
    lines = [f"{'measure':44s} {'deterministic':>14s} {'robust':>14s}"]

    def row(label, a, b, fmt="{:14.3f}"):
        lines.append(f"{label:44s} {fmt.format(a) if a is not None else '--':>14s} {fmt.format(b) if b is not None else '--':>14s}")

    row("coe at nominal inputs [$/MWh]", d["nominal"]["coe"], r["nominal"]["coe"])
    row("levelised (ratio of expectations)", d["levelised"], r["levelised"])
    row("mean coe | converged, net > 0", d["mean_coe_where_converged_and_positive_net"], r["mean_coe_where_converged_and_positive_net"])
    row("median coe | converged", d["median_coe_where_converged"], r["median_coe_where_converged"])
    row("mean coe | feasible", d["coe_where_feasible"]["mean"], r["coe_where_feasible"]["mean"])
    row("median coe | feasible", d["coe_where_feasible"]["median"], r["coe_where_feasible"]["median"])
    row("coe p5 / p95 | feasible", d["coe_where_feasible"]["p5"], r["coe_where_feasible"]["p5"])
    row("  ", d["coe_where_feasible"]["p95"], r["coe_where_feasible"]["p95"])
    row("failed fraction (warm, continuation)", d["failed_fraction"], r["failed_fraction"], "{:14.4f}")
    row("failed fraction (cold, nominal root)", d["failed_fraction_cold_start"], r["failed_fraction_cold_start"], "{:14.4f}")
    row("P(net power <= 0)", d["p_net_nonpositive"], r["p_net_nonpositive"], "{:14.4f}")
    row("feasible fraction", d["feasible_fraction"], r["feasible_fraction"], "{:14.4f}")
    row("any violation fraction", d["any_violation_fraction"], r["any_violation_fraction"], "{:14.4f}")
    row("net power p5 / p50 / p95 [MW]", d["net_mw_where_converged"]["p5"], r["net_mw_where_converged"]["p5"], "{:14.1f}")
    row("  ", d["net_mw_where_converged"]["p50"], r["net_mw_where_converged"]["p50"], "{:14.1f}")
    row("  ", d["net_mw_where_converged"]["p95"], r["net_mw_where_converged"]["p95"], "{:14.1f}")
    lines.append("")
    lines.append(f"{'constraint':28s} {'P(g>0) det':>11s} {'P(g>0) rob':>11s} {'CVaR0.9 det':>12s} {'CVaR0.9 rob':>12s}")
    for c in d["p_violated"]:
        lines.append(f"{c.replace('^cond.constraints.', ''):28s} {d['p_violated'][c]:11.4f} {r['p_violated'][c]:11.4f} "
                     f"{d['cvar']['0.9'][c]:12.4f} {r['cvar']['0.9'][c]:12.4f}")
    lines.append("")
    lines.append(f"{'ixc':>4s} {'place':44s} {'lower':>8s} {'upper':>8s} {'det':>10s} {'robust':>10s} {'delta':>10s} active")
    for e in result["design"]:
        lines.append(f"{e['ixc']:4d} {e['place']:44s} {e['lower']:8.4g} {e['upper']:8.4g} {e['deterministic']:10.4f} {e['robust']:10.4f} "
                     f"{e['delta']:+10.4f} {e['active_bound'] or ''}")
    return "\n".join(lines)


# ---------------------------------------------------------------- measurement


MODES = ("fwd", "jacrev", "jacfwd", "grad", "jacrev_chunked")
"""`fwd`: `statistics`; `jacrev` / `jacfwd`: its Jacobian (1 + n_g + 1 outputs, 6
inputs); `grad`: `jax.grad` of `f` alone (one cotangent); `jacrev_chunked`:
`jacrev` of the chunked statistics in chunks of `--chunk`."""


def program(fns: dict, mode: str, chunk: int | None):
    if mode == "fwd":
        return jax.jit(fns["statistics_vec"])
    if mode == "jacrev":
        return jax.jit(jax.jacrev(fns["statistics_vec"]))
    if mode == "jacfwd":
        return jax.jit(jax.jacfwd(fns["statistics_vec"]))
    if mode == "grad":
        return jax.jit(jax.grad(lambda x, th, st: fns["statistics"](x, th, st)[0]))
    if mode == "jacrev_chunked":
        def chunked(x, th, st):
            return fns["statistics_chunked_vec"](x, th, st, (st.shape[0] - 1) // chunk)
        return jax.jit(jax.jacrev(chunked))
    raise ValueError(mode)


def executable_memory(fn, *args) -> dict:
    """XLA's own accounting of the compiled executable: the temporaries it holds
    during one call (`temp_mb`, where reverse mode's residuals live), its arguments,
    outputs and code -- separable from the process's RSS, which the compiler's own
    working set dominates on the CPU."""
    try:
        analysis = fn.lower(*args).compile().memory_analysis()
    except Exception as failure:  # noqa: BLE001
        return {"memory_analysis": f"{type(failure).__name__}: {failure}"[:120]}
    if analysis is None:
        return {}
    mb = 2.0**20
    return {
        "temp_mb": analysis.temp_size_in_bytes / mb,
        "argument_mb": analysis.argument_size_in_bytes / mb,
        "output_mb": analysis.output_size_in_bytes / mb,
        "code_mb": analysis.generated_code_size_in_bytes / mb,
    }


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def device_peak_mb() -> float | None:
    try:
        stats = jax.local_devices()[0].memory_stats()
        return None if not stats else stats.get("peak_bytes_in_use", 0) / 2**20
    except Exception:  # noqa: BLE001
        return None


def measure(choice: Choice, mode: str, chunk: int | None, repeats: int = 3) -> dict:
    """One configuration in this process: build, compile, time, peak RSS."""
    n = choice.n
    row = {"N": n, "mode": mode, "backend": jax.default_backend(), "chunk": chunk if mode == "jacrev_chunked" else None}
    model = build(choice)
    fns = ouu.make(model)
    row["build_s"] = model.build_s
    row["rss_after_build_mb"] = rss_mb()
    fn = program(fns, mode, chunk)
    x0 = jnp.asarray(model.x0)
    starts = jnp.asarray(model.starts0)
    began = time.perf_counter()
    out = jax.block_until_ready(fn(x0, model.theta, starts))
    row["first_call_s"] = time.perf_counter() - began
    row["rss_after_compile_mb"] = rss_mb()
    row.update(executable_memory(fn, x0, model.theta, starts))
    walls = []
    for _ in range(repeats):
        began = time.perf_counter()
        out = jax.block_until_ready(fn(x0, model.theta, starts))
        walls.append(time.perf_counter() - began)
    row["warm_s"] = min(walls)
    row["warm_us_per_sample"] = 1e6 * min(walls) / n
    row["rss_peak_mb"] = rss_mb()
    row["rss_increment_mb"] = row["rss_peak_mb"] - row["rss_after_build_mb"]
    row["device_peak_mb"] = device_peak_mb()
    arr = np.asarray(out)
    row["output_shape"] = list(arr.shape)
    row["finite"] = bool(np.all(np.isfinite(arr)))
    if mode == "fwd":
        row["f"] = float(arr[0])
        row["max_g_cvar"] = float(arr[1:-1].max())
        row["failed_fraction"] = float(arr[-1])
    elif mode in ("jacrev", "jacfwd", "jacrev_chunked"):
        row["max_abs_jac"] = float(np.abs(arr).max())
        row["jac_f"] = arr[0].tolist()
    elif mode == "grad":
        row["jac_f"] = arr.tolist()
    return row


def scaling(sizes, modes, chunk: int, timeout_s: float, python: str, fresh: bool = False) -> list[dict]:
    """Every (N, mode) in a fresh subprocess, rows collected into `out/ouu_scaling.csv`.
    Rows already in `out/ouu_scaling.json` with status `ok` are kept and their
    configurations skipped unless `fresh`, so a sweep can be resumed or extended."""
    LOGS.mkdir(parents=True, exist_ok=True)
    rows = []
    previous = OUT / "ouu_scaling.json"
    if not fresh and previous.exists():
        rows = [r for r in json.loads(previous.read_text()) if r.get("status") == "ok"]
    finished = {(r["N"], r["mode"]) for r in rows}
    env = dict(os.environ, JAX_PLATFORMS=os.environ.get("JAX_PLATFORMS", "cpu"), XLA_PYTHON_CLIENT_PREALLOCATE="false")
    for n in sizes:
        for mode in modes:
            if (mode == "jacrev_chunked" and n <= chunk) or (n, mode) in finished:
                continue
            log = LOGS / f"ouu_measure_{n}_{mode}.log"
            cmd = [python, __file__, "--measure", "--n", str(n), "--mode", mode, "--chunk", str(chunk)]
            began = time.perf_counter()
            try:
                done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, env=env, cwd=str(Path(__file__).resolve().parent.parent))  # noqa: S603
                log.write_text(done.stdout + "\n--- stderr ---\n" + done.stderr)
                line = next((ln for ln in reversed(done.stdout.splitlines()) if ln.startswith("OUU_RESULT ")), None)
                if done.returncode != 0 or line is None:
                    tail = (done.stderr.strip().splitlines() or ["?"])[-1][:160]
                    row = {"N": n, "mode": mode, "status": f"exit {done.returncode}: {tail}"}
                else:
                    row = json.loads(line[len("OUU_RESULT "):])
                    row["status"] = "ok"
            except subprocess.TimeoutExpired as failure:
                log.write_text((failure.stdout or b"").decode() if isinstance(failure.stdout, bytes) else (failure.stdout or ""))
                row = {"N": n, "mode": mode, "status": f"timeout after {timeout_s:.0f} s"}
            row["subprocess_s"] = time.perf_counter() - began
            rows.append(row)
            print(json.dumps(row, default=str), flush=True)
            rows.sort(key=lambda r: (r["N"], MODES.index(r["mode"])))
            write_csv("ouu.py", with_ratios(rows), name="ouu_scaling")
            write_json("ouu.py", with_ratios(rows), name="ouu_scaling")
    rows = with_ratios(rows)
    write_csv("ouu.py", rows, name="ouu_scaling")
    write_json("ouu.py", rows, name="ouu_scaling")
    return rows


def with_ratios(rows: list[dict]) -> list[dict]:
    """Reverse / forward ratios for time and memory, per N."""
    by = {(r["N"], r["mode"]): r for r in rows if r.get("status") == "ok"}
    out = []
    for r in rows:
        r = dict(r)
        base = by.get((r["N"], "fwd"))
        if base and r.get("status") == "ok" and r["mode"] != "fwd":
            r["time_over_fwd"] = r["warm_s"] / base["warm_s"]
            r["rss_increment_over_fwd"] = (r["rss_increment_mb"] / base["rss_increment_mb"]) if base["rss_increment_mb"] > 0 else None
            r["rss_peak_over_fwd"] = r["rss_peak_mb"] / base["rss_peak_mb"]
            if "temp_mb" in r and base.get("temp_mb"):
                r["temp_over_fwd"] = r["temp_mb"] / base["temp_mb"]
        out.append(r)
    return out


# ---------------------------------------------------------------- figures


def plot(source: Path, alphas=(0.5, 0.75, 0.9, 0.95, 0.99), histogram_alpha: float = 0.9, tag: str = "") -> None:
    """`out/ouu_price_of_robustness.png` and `out/ouu_histograms_<alpha>.png` from the
    evidence files under `source`."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    det_colour, rob_colour = "#4a4a4a", "#c0504d"
    rows = []
    for alpha in alphas:
        path = source / f"ouu_evidence_{alpha:g}{tag}.json"
        if not path.exists():
            print(f"[plot] no {path}", flush=True)
            continue
        e = json.loads(path.read_text())
        rows.append((alpha, e))
    if rows:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
        for ax, key, title in ((axes[0], "mean_coe_where_converged_and_positive_net", "mean coe | converged, net > 0"),
                               (axes[1], "coe_where_feasible", "mean coe | feasible")):
            def pick(summary):
                return summary[key] if key != "coe_where_feasible" else summary[key]["mean"]

            det_fixed = [pick(e["sets"]["fixed"]["deterministic"]) for _a, e in rows]
            det_fresh = [pick(e["sets"]["fresh"]["deterministic"]) for _a, e in rows]
            ax.axhline(np.nanmean([v for v in det_fixed if v is not None]), color=det_colour, lw=1.2, label="deterministic, fixed set")
            ax.axhline(np.nanmean([v for v in det_fresh if v is not None]), color=det_colour, lw=1.2, ls="--", label="deterministic, fresh set")
            a = [alpha for alpha, _e in rows]
            ax.plot(a, [pick(e["sets"]["fixed"]["robust"]) for _a, e in rows], "o-", color=rob_colour, label="robust, fixed set")
            ax.plot(a, [pick(e["sets"]["fresh"]["robust"]) for _a, e in rows], "s--", color=rob_colour, mfc="white", label="robust, fresh set")
            for alpha, e in rows:
                if not e.get("converged", True):
                    ax.annotate("not converged", (alpha, pick(e["sets"]["fixed"]["robust"])), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
            ax.set_xlabel(r"$\alpha$ (CVaR level)")
            ax.set_ylabel("coe [$/MWh]")
            ax.set_title(title, fontsize=10)
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(OUT / "ouu_price_of_robustness.png", dpi=160)
        plt.close(fig)
        print(f"[plot] wrote {OUT / 'ouu_price_of_robustness.png'}", flush=True)

    path = source / f"ouu_evidence_{histogram_alpha:g}{tag}.json"
    npz = source / f"ouu_evidence_{histogram_alpha:g}{tag}.npz"
    if not (path.exists() and npz.exists()):
        print(f"[plot] no {path} / {npz}", flush=True)
        return
    e = json.loads(path.read_text())
    cols = np.load(npz)
    names = [c.replace("^cond.constraints.", "") for c in e["constraints"]]
    det, rob = e["sets"]["fixed"]["deterministic"], e["sets"]["fixed"]["robust"]
    worst = sorted(range(len(names)), key=lambda i: -max(det["p_violated"][e["constraints"][i]], rob["p_violated"][e["constraints"][i]]))[:3]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4))
    feasible_colour, infeasible_colour = {"deterministic": "#7f7f7f", "robust": "#d98c8a"}, {"deterministic": "#2b2b2b", "robust": "#8b1a1a"}

    def stacked(ax, key, transform, label, bins=40, log=False):
        """coe (or another column) at both designs, each stacked feasible / infeasible
        in one set of bins."""
        values = {d: transform(cols[f"fixed/{d}/{key}"], cols[f"fixed/{d}/converged"]) for d in ("deterministic", "robust")}
        finite = np.concatenate([v[np.isfinite(v)] for v in values.values()])
        if finite.size == 0:
            return
        lo, hi = np.percentile(finite, [0.5, 99.5])
        edges = np.linspace(lo, hi, bins + 1)
        width = (edges[1] - edges[0]) / 2.0
        for j, d in enumerate(("deterministic", "robust")):
            v = np.clip(values[d], lo, hi)
            feasible = cols[f"fixed/{d}/feasible"]
            ok = np.isfinite(values[d])
            h_f, _ = np.histogram(v[ok & feasible], bins=edges)
            h_i, _ = np.histogram(v[ok & ~feasible], bins=edges)
            left = edges[:-1] + j * width
            ax.bar(left, h_f, width=width, align="edge", color=feasible_colour[d], label=f"{d}, feasible")
            ax.bar(left, h_i, width=width, align="edge", bottom=h_f, color=infeasible_colour[d], label=f"{d}, infeasible")
        ax.set_xlabel(label)
        ax.set_ylabel("samples")
        if log:
            ax.set_yscale("log")

    n_samples = e["n"]
    stacked(axes[0, 0], "coe", lambda v, c: np.where(c & (v < 1e6), v, np.nan), "coe [$/MWh] (converged, < 1e6)")
    axes[0, 0].set_title(f"coe, N = {n_samples}, alpha = {histogram_alpha:g}", fontsize=10)
    stacked(axes[0, 1], "net", lambda v, c: np.where(c, v, np.nan), "net electric power [MW]")
    axes[0, 1].set_title("net electric power", fontsize=10)
    axes[0, 2].axis("off")
    text = [f"feasible fraction: det {det['feasible_fraction']:.3f}, robust {rob['feasible_fraction']:.3f}",
            f"failed fraction:   det {det['failed_fraction']:.3f}, robust {rob['failed_fraction']:.3f}",
            f"P(net <= 0):       det {det['p_net_nonpositive']:.3f}, robust {rob['p_net_nonpositive']:.3f}",
            f"coe | feasible mean: det {_fmt(det['coe_where_feasible']['mean'])}, robust {_fmt(rob['coe_where_feasible']['mean'])}",
            f"coe | net > 0 mean:  det {_fmt(det['mean_coe_where_converged_and_positive_net'])}, robust {_fmt(rob['mean_coe_where_converged_and_positive_net'])}",
            f"coe at nominal:      det {det['nominal']['coe']:.1f}, robust {rob['nominal']['coe']:.1f}"]
    axes[0, 2].text(0.0, 0.95, "\n".join(text), va="top", ha="left", fontsize=8, family="monospace", transform=axes[0, 2].transAxes)
    for ax, i in zip(axes[1], worst, strict=False):
        stacked(ax, "g", lambda v, c, i=i: np.where(c, v[:, i], np.nan), f"{names[i]} residual (g <= 0 satisfied)")
        ax.axvline(0.0, color="black", lw=0.8)
        ax.set_title(f"{names[i]}: P(g>0) det {det['p_violated'][e['constraints'][i]]:.3f}, robust {rob['p_violated'][e['constraints'][i]]:.3f}", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / f"ouu_histograms_{histogram_alpha:g}{tag}.png", dpi=160)
    plt.close(fig)
    print(f"[plot] wrote {OUT / f'ouu_histograms_{histogram_alpha:g}{tag}.png'}", flush=True)


# ---------------------------------------------------------------- main


def _run_payload(argv) -> tuple[dict, Path]:
    robust = Path(option(argv, "--robust", "", str))
    if not robust.is_file():
        raise SystemExit("this mode needs --robust <run JSON>")
    return json.loads(robust.read_text()), robust


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--help" in argv or "-h" in argv:
        print(__doc__)
        return 0
    if "--measure" in argv:
        choice = Choice.from_argv(argv, inputs=option(argv, "--inputs", "all", str),
                                  objective=option(argv, "--objective", "levelised", str))
        row = measure(choice, option(argv, "--mode", "fwd", str), option(argv, "--chunk", 256))
        print("OUU_RESULT " + json.dumps(row, default=str), flush=True)
        return 0
    if "--scaling" in argv:
        sizes = tuple(int(s) for s in option(argv, "--sizes", "64,256,1024,4096", str).split(","))
        modes = tuple(option(argv, "--modes", "fwd,jacrev,jacfwd,grad,jacrev_chunked", str).split(","))
        if unknown := set(modes) - set(MODES):
            raise SystemExit(f"--modes: no such mode {sorted(unknown)}; one of {MODES}")
        scaling(sizes, modes, option(argv, "--chunk", 256), option(argv, "--timeout", 1800.0, float), sys.executable, "--fresh" in argv)
        return 0
    if "--smoke" in argv:
        if "--no-warm" in argv:
            raise SystemExit("--no-warm is gone: the port's outer problem warm-starts through the batched program's memo")
        choice = Choice.from_argv(argv)
        smoke(choice, option(argv, "--max-iter", 200), option(argv, "--eps", ouu.EPS_FAILED, float),
              option(argv, "--move-limit", 0.25, float),
              jac=option(argv, "--jac", "fwd", str), chunks=option(argv, "--chunk", None), tol=option(argv, "--tol", 1e-4, float),
              ftol=option(argv, "--ftol", 1e-6, float), gtol=option(argv, "--gtol", 1e-4, float), name=option(argv, "--name", "ouu_smoke", str),
              with_evidence="--evidence" in argv, fresh_seed=option(argv, "--fresh-seed", 1), tag=option(argv, "--tag", "", str),
              start=Path(option(argv, "--start", "", str)) if "--start" in argv else None)
        return 0
    if "--decompose" in argv:
        choice = Choice.from_argv(argv, inputs=option(argv, "--inputs", "all", str), objective="levelised")
        decompose(choice, Path(option(argv, "--robust", str(OUT / "ouu_smoke.json"), str)),
                  name=option(argv, "--name", "ouu_decomposition", str))
        return 0
    if "--evidence" in argv:
        payload, robust = _run_payload(argv)
        choice = Choice.from_argv(argv, payload)
        model = build(choice)
        x_rob, source = robust_design_of(payload, model.design)
        evidence(model, ouu.make(model), model.x0, x_rob, option(argv, "--fresh-seed", 1),
                 f"ouu_evidence_{choice.alpha:g}{option(argv, '--tag', '', str)}", source, choice,
                 extra={"run": str(robust), "converged": payload.get("converged"), "reason": payload.get("reason"),
                        "robust_f": payload.get("robust", {}).get("f")})
        return 0
    if "--sweep-te" in argv:
        payload, robust = _run_payload(argv)
        choice = Choice.from_argv(argv, payload)
        model = build(choice)
        x_rob, source = robust_design_of(payload, model.design)
        sweep_te(model, ouu.make(model), x_rob, option(argv, "--name", f"ouu_sweep_te_{robust.stem}", str),
                 points=option(argv, "--points", 16), source=source)
        return 0
    if "--plot" in argv:
        plot(Path(option(argv, "--from", str(OUT.parent / "out_cluster"), str)), histogram_alpha=option(argv, "--alpha", 0.9, float), tag=option(argv, "--tag", "", str))
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
