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
series: dict = defaultdict(dict)
for r in rows:
    key = (r["shape"], r["configuration"], r["backend"])
    n = int(r["N"])
    series[key][n] = min(float(r["us_per_point"]), series[key].get(n, 1e30))

panels = [("mda", "stellarator_helias"), ("mda", "large_tokamak_nof"),
          ("sand", "stellarator_helias"), ("sand", "large_tokamak_nof")]
titles = {"mda": "MDA evaluation (values)", "sand": "SAND value + Jacobian (hand cut)"}
fig, axes = plt.subplots(2, 2, figsize=(9, 6.4), sharex=True)
for ax, (shape, cfg) in zip(axes.flat, panels):
    for backend, colour, label in (("cpu", CPU, "CPU, Ryzen 7 3700X"), ("gpu", GPU, "GPU, RTX 3080")):
        pts = sorted(series[(shape, cfg, backend)].items())
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=colour, lw=2, marker="o", ms=5, label=label)
        ax.annotate(label, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    color=INK, fontsize=8, va="center")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_title(f"{titles[shape]} — {cfg}", fontsize=9, color=INK, loc="left")
    ax.grid(True, which="major", color=RULE, lw=0.6); ax.grid(False, which="minor")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(RULE)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlim(0.7, 2 ** 18)
for ax in axes[1]: ax.set_xlabel("points per batched call, N", color=MUTED, fontsize=9)
for ax in axes[:, 0]: ax.set_ylabel("µs per point (warm, float64)", color=MUTED, fontsize=9)
axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
fig.suptitle("Batching the port with jax.vmap: cost per point falls with N until memory (GPU OOM at 65 536 / 4096)",
             fontsize=10, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 0.98, 0.96))
fig.savefig(OUT / "batching.png", dpi=150)
fig.savefig(OUT / "batching.pdf")
print(OUT / "batching.png")
