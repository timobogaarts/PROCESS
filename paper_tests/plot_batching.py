"""The batching curves: us per point against N, CPU and GPU, one panel per (shape,
configuration) -- drawn from `out/batching.csv`. Two series per panel, direct-labelled;
log-log, one axis; the same palette slots throughout (slot 1 blue = CPU, slot 2
orange = GPU, validated with the dataviz reference palette).

    $PY paper_tests/plot_batching.py     -> out/batching.png
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "out"
CPU, GPU = "#2a78d6", "#eb6834"
INK, MUTED, RULE = "#0b0b0b", "#52514e", "#e6e5e2"

rows = [r for r in csv.DictReader(open(OUT / "batching.csv"))
        if r["shape"] in ("mda", "sand") and r["precision"] == "f64" and r["status"] == "ok"
        and r["vmap"] == "True" and r["tag"] in ("", "None") and r.get("cut", "hand") in ("hand", "")]
series: dict = defaultdict(dict)   # us per point
walls: dict = defaultdict(dict)    # warm wall per batched call, s
for r in rows:
    key = (r["shape"], r["configuration"], r["backend"])
    n = int(r["N"])
    if float(r["us_per_point"]) < series[key].get(n, 1e30):
        series[key][n] = float(r["us_per_point"])
        walls[key][n] = float(r["warm_wall"])

panels = [("mda", "stellarator_helias"), ("mda", "large_tokamak_nof"),
          ("sand", "stellarator_helias"), ("sand", "large_tokamak_nof")]
titles = {"mda": "MDA evaluation (values)", "sand": "SAND value + Jacobian (hand cut)"}
def draw(data, ylabel, title, name):
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.4), sharex=True)
    for ax, (shape, cfg) in zip(axes.flat, panels):
        for backend, colour, label in (("cpu", CPU, "CPU, Ryzen 7 3700X"), ("gpu", GPU, "GPU, RTX 3080")):
            pts = sorted(data[(shape, cfg, backend)].items())
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=colour, lw=2, marker="o", ms=5, label=label)
            ax.annotate(label, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                        color=INK, fontsize=8, va="center")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"{titles[shape]} — {cfg}", fontsize=9, color=INK, loc="left")
        ax.grid(True, which="major", color=RULE, lw=0.6); ax.grid(False, which="minor")
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): ax.spines[sp].set_color(RULE)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_xlim(0.7, 3e5)
    for ax in axes[1]: ax.set_xlabel("points per batched call, N", color=MUTED, fontsize=9)
    for ax in axes[:, 0]: ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 0.98, 0.96))
    fig.savefig(OUT / f"{name}.png", dpi=150)
    fig.savefig(OUT / f"{name}.pdf")
    print(OUT / f"{name}.png")


# The closed architecture (`close_conditions.py --batch`): plain and closed MDA on
# stellarator_helias, per backend, as a third panel pair.
closed: dict = defaultdict(dict)
closed_walls: dict = defaultdict(dict)
for backend in ("cpu", "gpu"):
    path = OUT / f"close_conditions_batch_{backend}.csv"
    if not path.exists():
        continue
    for r in csv.DictReader(open(path)):
        if r.get("status") or not r.get("us_per_point"):
            continue
        key = (r["shape"], backend)
        closed[key][int(r["N"])] = float(r["us_per_point"])
        closed_walls[key][int(r["N"])] = float(r["warm_s"])


def draw_closed(data, ylabel, title, name):
    fig, ax = plt.subplots(figsize=(6, 4))
    styles = {("plain", "cpu"): (CPU, "-", "plain MDA, CPU"), ("plain", "gpu"): (GPU, "-", "plain MDA, GPU"),
              ("closed", "cpu"): (CPU, "--", "equalities closed inside, CPU"),
              ("closed", "gpu"): (GPU, "--", "equalities closed inside, GPU")}
    for key, (colour, ls, label) in styles.items():
        pts = sorted(data[key].items())
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=colour, ls=ls, lw=2, marker="o", ms=5, label=label)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(True, which="major", color=RULE, lw=0.6); ax.grid(False, which="minor")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color(RULE)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlabel("points per batched call, N", color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=150)
    fig.savefig(OUT / f"{name}.pdf")
    print(OUT / f"{name}.png")


draw_closed(closed, "µs per point (warm, float64)",
            "stellarator_helias MDA, plain vs. the two equalities solved inside", "batching_closed")
draw_closed(closed_walls, "warm wall per batched call, s",
            "stellarator_helias MDA, plain vs. closed: wall per call", "batching_closed_wall")

draw(series, "µs per point (warm, float64)",
     "Batching the port with jax.vmap: cost per point against N (GPU OOM at 65 536 / 4096)", "batching")
draw(walls, "warm wall per batched call, s",
     "The same calls, total wall: what one batched evaluation costs as N grows", "batching_wall")
