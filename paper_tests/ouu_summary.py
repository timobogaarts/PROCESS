"""One table over the night's two-stage runs: `out_cluster/ouu2_alpha*<tag>.json` with
their evidence (`ouu_evidence_<alpha><tag>.json`) and T_e sweeps
(`ouu_sweep_te_<run>.json`), as `out/ouu2_summary.{csv,md}` and to stdout.

    $PY paper_tests/ouu_summary.py [--from paper_tests/out_cluster]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from common import OUT, write_csv

SHORT = {
    ".physics.b_plasma_toroidal_on_axis": "B", ".physics.rmajor": "R",
    ".physics.temp_plasma_electron_vol_avg_kev": "T_e", ".tfcoil.t_tf_superconductor_quench": "t_q",
    ".tfcoil.f_a_tf_turn_cable_copper": "f_Cu", ".physics.f_nd_alpha_thermal_electron": "f_alpha",
}


def main(argv):
    source = Path(argv[argv.index("--from") + 1]) if "--from" in argv else OUT.parent / "out_cluster"
    rows = []
    for path in sorted(source.glob("ouu2_alpha*.json")):
        if "_summary" in path.name:
            continue
        d = json.loads(path.read_text())
        alpha, tag = d["alpha"], path.stem.split(f"alpha{d['alpha']:g}", 1)[1]
        x = (d.get("robust") or d["final"])["x"]
        row = {
            "run": path.stem, "alpha": alpha, "objective": d["objective"], "pairing": d.get("pairing"),
            "table": d.get("table"), "with_c16": d.get("with_c16"), "alpha16": d.get("alpha16"),
            "te_recourse": d.get("te_recourse", 0), "converged": d.get("converged"), "reason": d.get("reason"),
            "f": (d.get("robust") or d["final"])["f"], "f_det": d["deterministic"].get("coe_dollar_per_mwh"),
            "calls": d.get("model_calls"), "wall_s": d.get("wall_s"),
        }
        row.update({SHORT.get(k, k): v for k, v in x.items()})
        ev = source / f"ouu_evidence_{alpha:g}{tag}.json"
        if ev.is_file():
            e = json.loads(ev.read_text())
            for set_name in ("fixed", "fresh"):
                for design in ("deterministic", "robust"):
                    r = e["sets"].get(set_name, {}).get(design)
                    if not r:
                        continue
                    p = f"{set_name[:2]}_{design[:3]}_"
                    row[p + "feasible"] = r["feasible_fraction"]
                    row[p + "failed"] = r["failed_fraction"]
                    row[p + "levelised"] = r.get("levelised")
                    row[p + "rated"] = r.get("rated")
                    row[p + "coe_nominal"] = r["nominal"]["coe"]
                    row[p + "net_p5"] = r["net_mw_where_converged"]["p5"]
                    row[p + "net_p50"] = r["net_mw_where_converged"]["p50"]
        sw = source / f"ouu_sweep_te_{path.stem}.json"
        if sw.is_file():
            w = json.loads(sw.read_text())
            a = w.get("all_constraints")
            if a:
                row["sweep_feasible_shared"] = a["feasible_fraction_shared"]
                row["sweep_feasible_swept"] = a["feasible_fraction_swept"]
                row["sweep_coe_shared"] = a["coe_mean_where_feasible_shared"]
                row["sweep_coe_swept"] = a["coe_mean_where_feasible_swept"]
                both = a["coe_mean_over_samples_feasible_both"]
                row["sweep_both_shared"], row["sweep_both_swept"], row["sweep_both_n"] = both["shared"], both["swept"], both["count"]
                row["sweep_te_p50"] = a["chosen_t_e_kev"]["p50"]
        rows.append(row)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    write_csv("ouu2_summary", [{k: r.get(k) for k in keys} for r in rows])
    head = ["run", "objective", "converged", "f", "R", "B", "T_e", "f_alpha", "fi_rob_feasible", "fr_rob_feasible",
            "fi_rob_levelised", "fi_det_levelised", "fi_rob_net_p5", "sweep_feasible_shared", "sweep_feasible_swept",
            "sweep_both_shared", "sweep_both_swept"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(k)) for k in head) + " |")
    (OUT / "ouu2_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if "--plot" in argv:
        print(plot(rows))
    return 0


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def plot(rows: list[dict], name: str = "ouu2_price_of_robustness") -> Path:
    """Two panels over alpha for the converged families (B2: shared T_e; D: T_e as
    recourse): the levelised coe of the robust design against the deterministic one
    on the same samples, and the feasible fraction in and out of sample."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    import re  # noqa: PLC0415

    families = {r"_B2$": ("B2: shared $T_e$", "C0"), r"_D(_cont)?$": ("D: $T_e$ recourse", "C1")}
    fig, (ax_f, ax_p) = plt.subplots(1, 2, figsize=(9, 3.4))
    for pattern, (label, colour) in families.items():
        sel = [r for r in rows if re.search(pattern, r["run"]) and r.get("fi_rob_levelised")]
        best = {}
        for r in sel:  # one row per alpha: the converged one, else the cheapest feasible
            key = r["alpha"]
            if key not in best or (r.get("converged") and not best[key].get("converged")) or (
                r.get("converged") == best[key].get("converged") and r["f"] < best[key]["f"]
            ):
                best[key] = r
        sel = sorted(best.values(), key=lambda r: r["alpha"])
        if not sel:
            continue
        alphas = [r["alpha"] for r in sel]
        fill = ["full" if r.get("converged") else "none" for r in sel]
        ax_f.plot(alphas, [r["fi_rob_levelised"] for r in sel], "-", color=colour, label=f"{label}, levelised")
        ax_f.plot(alphas, [r["fi_rob_coe_nominal"] for r in sel], "--", color=colour, label=f"{label}, nominal")
        for a, r, f in zip(alphas, sel, fill, strict=True):
            ax_f.plot([a], [r["fi_rob_levelised"]], "o", color=colour, fillstyle=f)
            ax_f.plot([a], [r["fi_rob_coe_nominal"]], "s", color=colour, fillstyle=f)
        ax_p.plot(alphas, [r["fi_rob_feasible"] for r in sel], "o-", color=colour, label=f"{label}, in sample")
        ax_p.plot(alphas, [r["fr_rob_feasible"] for r in sel], "x:", color=colour, label=f"{label}, fresh samples")
        det = sel[0]
        ax_f.axhline(det["fi_det_levelised"], color=colour, lw=0.8, alpha=0.5, label=f"deterministic design, {label.split(':')[1].strip()}, levelised")
        ax_p.axhline(det["fi_det_feasible"], color=colour, lw=0.8, alpha=0.5, label=f"deterministic design, {label.split(':')[1].strip()}")
    ax_f.text(0.02, 0.98, "hollow: iteration limit, not converged", transform=ax_f.transAxes, va="top", fontsize=7)
    ax_f.set_xlabel(r"$\alpha$"); ax_f.set_ylabel("coe [$/MWh]"); ax_f.legend(fontsize=7, loc="lower right")
    ax_p.set_xlabel(r"$\alpha$"); ax_p.set_ylabel("feasible fraction of samples"); ax_p.set_ylim(0, 1.02); ax_p.legend(fontsize=7)
    fig.tight_layout()
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=160)
    return path


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

