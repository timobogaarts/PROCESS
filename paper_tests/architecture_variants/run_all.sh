#!/bin/bash
# The variant sweep, serially: structure.py over every machine (which refuses to go on if a
# corner is not its preset), then per variant iteration.py and solve.py, one process each, so
# every compile is cold and no process grows XLA's JIT memory past one variant. A variant
# already in its csv is skipped, so a rerun resumes. Everything is logged to out/run.log.
#
#   paper_tests/architecture_variants/run_all.sh              # from anywhere
#   PY=... COTTAX=... paper_tests/architecture_variants/run_all.sh
set -u
PY=${PY:-~/anaconda3/envs/func_PROCESS_env/bin/python}
COTTAX=${COTTAX:-~/projects/jaxgraph/src}
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=$COTTAX
# As the Snellius job (`../cluster/job_tables.sh`): single-threaded BLAS. scipy's OpenBLAS,
# which SLSQP calls, gives last-bit different paths at 1 thread and at 2 or more, and
# st_regression's IDF/SAND count follows them (its optimum is degenerate).
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR=${MPLCONFIGDIR:-${TMPDIR:-/tmp}} XDG_CACHE_HOME=${XDG_CACHE_HOME:-${TMPDIR:-/tmp}}
cd "$(dirname "$0")/../.."                          # the configurations load by relative path
HERE=paper_tests/architecture_variants
OUT=${VARIANTS_OUT:-$HERE/out}
mkdir -p "$OUT"
LOG=$OUT/run.log
# If memory runs out, the kernel kills this sweep first, not the editor.
echo 1000 > /proc/self/oom_score_adj 2>/dev/null || true

quiet() { grep --line-buffered -v -i -E "warn|B matrix|^E0921|LLVM"; }
has_row() { [ -f "$OUT/$1/$2.csv" ] && grep -q "^$2,$3," "$OUT/$1/$2.csv"; }

echo "== $(date -Is) structure" | tee -a "$LOG"
$PY $HERE/structure.py 2>&1 | quiet >> "$LOG"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "structure.py failed: see $LOG" | tee -a "$LOG"; exit 1; fi

total=$(wc -l < "$OUT/variants.txt"); n=0
while read -r name label; do
    n=$((n + 1))
    for step in "iteration iteration_minimal" "solve solve_minimal_slsqp"; do
        set -- $step
        if has_row "$2" "$name" "$label"; then continue; fi
        echo "== $(date -Is) [$n/$total] $1 $name $label" | tee -a "$LOG"
        began=$SECONDS
        $PY $HERE/$1.py --configurations "$name" --variants "$label" 2>&1 < /dev/null | quiet >> "$LOG"
        status=${PIPESTATUS[0]}
        echo "   exit $status, $((SECONDS - began)) s" | tee -a "$LOG"
    done
done < "$OUT/variants.txt"
echo "== $(date -Is) done" | tee -a "$LOG"
