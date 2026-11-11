"""What the sweep found, from `out/` alone, nothing run:

- `out/summary.csv`: one row per variant -- its structure (unknowns, equalities,
  problems, nesting), the solve from the file's start (status, iterations, model calls,
  objective, times), the random starts (valid, converged, successful, success rate,
  median iterations of the successful), and the eval/Jacobian times.
- `out/comparison_snellius.csv`: the corners beside the paper table's own rows
  (`../architectures/out`, Snellius), column by column, with the ratio this/there.
  Where IDF and SAND coincide the corner is set beside both: Snellius ran them in one
  process, so SAND's `cold_s` there is a compile-cache hit on IDF's and only IDF's is a
  cold compile, as every `cold_s` here is.

    $PY paper_tests/architecture_variants/table.py
"""

from __future__ import annotations

import csv

import common

SOLVE = f"solve_{common.SCHEME}_{common.OPTIMISER}"
ITERATION = f"iteration_{common.SCHEME}"
STRUCTURE = f"structure_{common.SCHEME}"

SUMMARY = ("unknowns", "equalities", "inequalities", "problems", "nested",
           "status", "exit_code", "exit_message", "iterations", "values", "jacobians", "fused", "objf",
           "inner", "inner_sizes", "top_models", "inner_steps", "inner_steps_values",
           "model_evals", "model_evals_values", "count_check",
           "assemble_s", "cold_s", "warm_s", "model_s",
           "starts", "starts_valid", "starts_converged", "starts_success", "success_rate",
           "starts_iterations_median", "starts_inner_steps_median", "starts_model_evals_median",
           "starts_exit_codes", "starts_count_check_max", "starts_s",
           "evaluate_ms", "jacobian_ms", "serial_us", "serial_jacobian_us", "compile_s")
COMPARED = {SOLVE: ("iterations", "objf", "cold_s", "warm_s", "model_s", "values", "jacobians", "fused"),
            ITERATION: ("evaluate_ms", "jacobian_ms", "serial_us", "serial_jacobian_us", "compile_s")}


def rows_of(folder: str, root=None) -> dict:
    root = common.OUT if root is None else root
    out = {}
    for path in sorted((root / folder).glob("*.csv")):
        for r in common.read_rows(path):
            out[(r["configuration"], r.get("label", r.get("arm")))] = r
    return out


def ratio(a, b):
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return ""
    return a / b if b else ""


def main():
    structure, solve, iteration = rows_of(STRUCTURE), rows_of(SOLVE), rows_of(ITERATION)
    summary = []
    for key, s in structure.items():
        row = {"configuration": key[0], "label": key[1], "preset": s.get("preset", ""),
               "refused": s.get("refused", "") or solve.get(key, {}).get("refused", "")}
        for source in (s, solve.get(key, {}), iteration.get(key, {})):
            row.update({k: source[k] for k in SUMMARY if k in source and source[k] != ""})
        summary.append(row)
    common.write_all("table.py", common.OUT / "summary.csv", summary)

    snellius = {folder: rows_of(folder, common.ARCH / "out") for folder in COMPARED}
    mine = {SOLVE: solve, ITERATION: iteration}
    comparison = []
    for key, s in structure.items():
        for arm in filter(None, s.get("preset", "").split("/")):
            row = {"configuration": key[0], "label": key[1], "arm": arm}
            for folder, columns in COMPARED.items():
                here, there = mine[folder].get(key, {}), snellius[folder].get((key[0], arm), {})
                for c in columns:
                    row[f"{c}_here"], row[f"{c}_snellius"] = here.get(c, ""), there.get(c, "")
                    row[f"{c}_ratio"] = ratio(here.get(c), there.get(c))
            comparison.append(row)
    common.write_all("table.py", common.OUT / "comparison_snellius.csv", comparison)

    print(f"\n{'machine':24s} {'variant':7s} {'preset':8s} {'it':>4s} {'calls':>6s} {'model evals':>11s} {'success':>8s} {'warm s':>7s}")
    for r in summary:
        calls = sum(int(r.get(k, 0) or 0) for k in ("values", "jacobians", "fused"))
        rate = r.get("success_rate", "")
        rate = f"{float(rate):.0%}" if rate not in ("", "None") else "-"
        print(f"{r['configuration']:24s} {r['label']:7s} {r['preset']:8s} {r.get('iterations', '-'):>4s} "
              f"{calls:>6d} {r.get('model_evals', '-'):>11s} {rate:>8s} {float(r['warm_s']) if r.get('warm_s') else float('nan'):>7.3f}"
              f"{'  ' + r['refused'][:60] if r['refused'] else ''}")


if __name__ == "__main__":
    main()
