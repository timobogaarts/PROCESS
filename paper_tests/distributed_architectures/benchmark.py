"""AAO, the sequential split and ALC on Helias, side by side: the optimum each finds and
what it costs, with only the architecture changed.

What is the same in every case:

- **The problem.** Every case is a recipe on the AAO base (`partition.base`:
  `problem_model`, the file's problem at `Opt`). AAO is cottax's `SAND` on it; the
  sequential split and ALC are primitive-op recipes on it that do inside each of their
  problems what AAO's SAND does to the whole (the same two cuts and closures, the coil
  solve absorbed).
- **The start.** The file's design; every coupling copy from the MDA at it.
- **The optimiser.** SLSQP (`SlsqpDriver`) for every optimisation problem, at its
  defaults (`ftol` 1e-8, scaled `x / x_start`), `max_iter` `study.MAX_ITER` (500), its
  coupling conditions scaled by the MDA (`study.condition_scale`). VMCON is not used:
  ALC's unbounded copies take it to a non-finite model (`alc_report.md`, change 1).

What differs is the architecture: one problem (AAO), two in sequence with the magnet
stage on its own objective and `b_t >= 5` (sequential, not equivalent), three coordinated
by ALC (equivalent at its fixed point).

Each case is built and compiled once; the first solve is reported as **cold** (it
includes compilation), then `runs` more solves of the same compiled graph are timed
(**warm**, mean and standard deviation). Every count is checked to be the same in every
run. Counts:

- **Overarching loop**: ALC's coordination sweeps and multiplier updates; none for the
  others (SAND on this base nests no MDA).
- **Sub-optimisers**: per optimisation problem, its SLSQP solves and their iterations
  in total over a run (ALC solves each discipline once per sweep, and once more when the
  coordination's interior is walked at its answer), and how many solves SLSQP did not
  report as converged.

Writes `out/benchmark.csv` (one row per case), `out/benchmark_problems.csv` (one per
problem) and `out/benchmark.md` (the tables). Run with `run_benchmark.sh`, or

    python paper_tests/distributed_architectures/benchmark.py [--runs 5] [--cases aao sequential alc]
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import partition as C
from functional_process.cottax.architectures.drivers import SlsqpDriver

import alc as A
import sequential as S

OUT = Path(__file__).parent / "out"
OPTIMISER = SlsqpDriver


# ---------------------------------------------------------------- the cases, built


def build_aao(configuration) -> dict:
    a = C.study.assemble(configuration, "SAND", optimiser=OPTIMISER)
    return {"graph": a.graph, "inputs": a.inputs, "traces": {C.TOP: a.trace},
            "problems": {"Opt": C.TOP}, "log": None}


def build_sequential(configuration) -> dict:
    part = C.partition(configuration)
    state, env = C.mda_env(configuration)
    traces: dict = {}
    m, graph, given = S.driven(configuration, part, env, traces, optimiser=OPTIMISER)
    inputs = C.values(graph, state, {**dict(m.given), **given}, env, set(part.design))
    return {"graph": graph, "inputs": inputs, "traces": traces,
            "problems": {"magnet": S.MAGNET, "plasma": S.PLASMA}, "log": None}


def build_alc(configuration) -> dict:
    d = C.disciplines(configuration)
    state, env = C.mda_env(configuration)
    cons = A.consistency(d, env)
    traces: dict = {}
    log: list = []
    m, graph = A.driven(configuration, d, cons, env, traces, log=lambda *row: log.append(row))
    inputs = C.values(graph, state, m.given, env, set(d.design), start=A.starts(cons))
    return {"graph": graph, "inputs": inputs, "traces": traces,
            "problems": {k: A.PROBLEM[k] for k in C.DISCIPLINES}, "log": log}


CASES = {"aao": ("AAO (SAND)", build_aao), "sequential": ("sequential", build_sequential),
         "alc": ("ALC", build_alc)}


# ---------------------------------------------------------------- one run, read


def counts(built) -> dict:
    """Per problem: SLSQP solves, their iterations in total, and the solves SLSQP did not
    call converged, off its callback (one call per iteration, then `-1` per solve with
    `0.0` for success)."""
    out = {}
    for name, label in built["problems"].items():
        trace = built["traces"][label]
        ends = [row for row in trace if row[0] == -1]
        out[name] = {"solves": len(ends), "iterations": sum(1 for row in trace if row[0] >= 1),
                     "failed": sum(1 for row in ends if row[1] != 0.0)}
    return out


def violation(configuration, out) -> float:
    """The largest violation of the file's constraints at the answer, in their own
    normalised residuals: `|c|` for an equality, `max(c, 0)` for an inequality."""
    problem = configuration.problem
    eq = [abs(C.number(out, C.held(c))) for c in problem.equalities]
    ie = [max(C.number(out, C.held(c)), 0.0) for c in problem.inequalities]
    return max(eq + ie)


def read(configuration, built, out) -> dict:
    row = {"coe": C.number(out, C.MERIT), "violation": violation(configuration, out),
           "design": {C.short(x): C.number(out, x) for x in configuration.problem.design},
           "problems": counts(built), "sweeps": None, "updates": None}
    if built["log"] is not None:
        row["sweeps"] = len(built["log"])
        row["updates"] = sum(1 for r in built["log"] if r[3])
    return row


def clear(built) -> None:
    for trace in built["traces"].values():
        trace.clear()
    if built["log"] is not None:
        built["log"].clear()


def measure(configuration, case, runs) -> dict:
    title, build = CASES[case]
    began = time.perf_counter()
    built = build(configuration)
    seconds_build = time.perf_counter() - began
    out, cold = C.timed(built["graph"], built["inputs"])
    first = read(configuration, built, out)
    warm = []
    for i in range(runs):
        clear(built)
        out, seconds = C.timed(built["graph"], built["inputs"])
        again = read(configuration, built, out)
        if (again["problems"], again["sweeps"], again["updates"]) != (
                first["problems"], first["sweeps"], first["updates"]):
            raise RuntimeError(f"{case}: run {i + 1} counted {again} where the first counted {first}")
        warm.append(seconds)
        print(f"  {case} run {i + 1}/{runs}: {seconds:.2f} s", flush=True)
    return {"case": case, "title": title, "build_s": seconds_build, "cold_s": cold,
            "warm_s": warm, **first}


# ---------------------------------------------------------------- tables


def provenance() -> dict:
    def git(path):
        try:
            rev = subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True, check=True).stdout.strip()
            dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain"],
                                   capture_output=True, text=True, check=True).stdout.strip()
            return rev + ("+dirty" if dirty else "")
        except (OSError, subprocess.CalledProcessError):
            return "?"
    import cottax
    import jax
    import scipy
    return {"date": datetime.now().isoformat(timespec="seconds"), "host": platform.node(),
            "python": platform.python_version(), "jax": jax.__version__, "scipy": scipy.__version__,
            "process": git(Path(__file__).resolve().parents[2]),
            "cottax": git(Path(cottax.__file__).resolve().parents[2]),
            "threads": os.environ.get("OMP_NUM_THREADS", "unset")}


def write(rows, runs) -> None:
    OUT.mkdir(exist_ok=True)
    ref = next((r["coe"] for r in rows if r["case"] == "aao"), None)
    with open(OUT / "benchmark.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "coe", "coe_vs_aao", "max_violation", "sweeps", "multiplier_updates",
                    "solves", "iterations", "failed_solves", "build_s", "cold_s",
                    "warm_mean_s", "warm_std_s", "runs", *rows[0]["design"]])
        for r in rows:
            p = r["problems"].values()
            w.writerow([r["case"], f"{r['coe']:.10g}",
                        "" if ref is None else f"{r['coe'] / ref - 1:.3e}", f"{r['violation']:.3e}",
                        r["sweeps"] or "", r["updates"] or "", sum(x["solves"] for x in p),
                        sum(x["iterations"] for x in p), sum(x["failed"] for x in p),
                        f"{r['build_s']:.2f}", f"{r['cold_s']:.2f}", f"{np.mean(r['warm_s']):.3f}",
                        f"{np.std(r['warm_s'], ddof=1) if runs > 1 else 0.0:.3f}", runs,
                        *(f"{v:.8g}" for v in r["design"].values())])
    with open(OUT / "benchmark_problems.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "problem", "solves", "iterations", "iterations_per_solve", "failed_solves"])
        for r in rows:
            for name, x in r["problems"].items():
                w.writerow([r["case"], name, x["solves"], x["iterations"],
                            f"{x['iterations'] / max(x['solves'], 1):.1f}", x["failed"]])
    prov = provenance()
    lines = [f"# AAO, sequential and ALC on Helias", "",
             f"{prov['date']} on `{prov['host']}`; PROCESS `{prov['process']}`, cottax "
             f"`{prov['cottax']}`; jax {prov['jax']}, scipy {prov['scipy']}, Python "
             f"{prov['python']}; `OMP_NUM_THREADS` {prov['threads']}. Every optimisation "
             f"problem: SLSQP, `ftol` 1e-8, `max_iter` {C.study.MAX_ITER}, scaled, from the "
             f"file's start. Warm: mean ± standard deviation over {runs} runs of the compiled "
             f"graph; cold: the first run, compilation included. ALC departs from Tosserams, "
             f"Etman & Rooda (2008) in three choices: the inner loop ends when the copies "
             f"settle (fixed tolerance), the stopping test is stricter, and the weights are "
             f"never raised (β = 1); see `alc_report.md`.", "",
             "| Case | COE | vs AAO | Max violation | Overarching loop | Sub-optimiser iterations (solves) | Cold (s) | Warm (s) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        vs = "—" if ref is None or r["case"] == "aao" else f"{r['coe'] / ref - 1:+.2e}"
        loop = "—" if r["sweeps"] is None else f"{r['sweeps']} sweeps, {r['updates']} multiplier updates"
        subs = "; ".join(f"{n} {x['iterations']} ({x['solves']})" + (f", {x['failed']} failed" if x["failed"] else "")
                         for n, x in r["problems"].items())
        std = np.std(r["warm_s"], ddof=1) if runs > 1 else 0.0
        lines.append(f"| {r['title']} | {r['coe']:.7f} | {vs} | {r['violation']:.1e} | {loop} | "
                     f"{subs} | {r['cold_s']:.1f} | {np.mean(r['warm_s']):.2f} ± {std:.2f} |")
    names = list(rows[0]["design"])
    lines += ["", "| Case | " + " | ".join(f"`{n}`" for n in names) + " |",
              "|---|" + "---|" * len(names)]
    for r in rows:
        lines.append(f"| {r['title']} | " + " | ".join(f"{v:.6g}" for v in r["design"].values()) + " |")
    (OUT / "benchmark.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--cases", nargs="*", default=list(CASES), choices=list(CASES))
    args = ap.parse_args(argv)
    configuration = C.load(C.CONFIGURATION)
    rows = []
    for case in args.cases:
        print(f"== {case}", flush=True)
        rows.append(measure(configuration, case, args.runs))
    write(rows, args.runs)


if __name__ == "__main__":
    main()
