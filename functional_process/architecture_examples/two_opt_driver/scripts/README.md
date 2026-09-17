# Scripts behind `../two_driver_report.md`

Run from the `PROCESS/` repo root with the env that imports both `process` and `cottax`:

```bash
cd PROCESS
PYTHONPATH=~/projects/jaxgraph/src:. ~/anaconda3/envs/func_PROCESS_env/bin/python ../two_opt_driver/scripts/<script>.py
```

Outputs (DSM html, report html) are written next to the scripts. Several scripts `exec` an
earlier one's head to share the build (`two_drivers2.py` -> `two_drivers_solve.py` ->
`two_drivers_btmin.py` -> `sand_warm.py` / `perf.py`), so keep them in one directory.

| script | what it does |
|---|---|
| `coupling_probe.py <IN.DAT>` | the condition x iteration-variable coupling matrix |
| `two_drivers2.py` | builds the two-Optimise graphs (legal split A, illegal split B, ownership clash C), folds per block, schedules, renders DSMs |
| `two_drivers_solve.py` | seeds and runs the two-driver schedule |
| `two_drivers_btmin.py` | same with a `b_t >= BT_MIN` condition in the magnet problem (`BT_MIN=5.0` env var) |
| `plasma_at_ref.py` | control: plasma optimiser alone at the reference SAND's machine |
| `bt_scan.py`, `channel.py` | why the plasma stage is infeasible: `b_t` scan, and the `tdmptf`/`fcutfsu` -> net-electric channel |
| `why.py`, `zero_row.py`, `reseed.py` | first-QP diagnosis and the re-seeding artefact |
| `perf.py` | cold/warm timings and per-block evaluation cost, two-driver vs SAND |
| `sand_warm.py` | full SAND warm-started from the two-driver answer (two-schedule chain) |
| `make_report.py` | renders `two_driver_report.md` + the executing-graph DSM into `two_driver_report.html` |
| `reference_cold_matrix_helias.txt` | the reference cold matrix re-run on this tree (2026-09-16) |
