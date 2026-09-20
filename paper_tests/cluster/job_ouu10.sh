#!/bin/bash
#SBATCH --job-name=ouu10_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%A.out
#
# The paper run, with the box opened. `ouu9_alpha0.9_te_design` answered with `rmajor`
# on its 30 m upper bound and the quench time on its 50 s one, so that answer is the
# box edge, not the optimum. PROCESS's own deterministic optimum is interior (26.69 m),
# so widening the bounds cannot move it -- this asks only where the robust design goes
# when nothing stops it.
set -u
source "$HOME/PROCESS/paper_tests/cluster/env.sh"
cd "$HOME/PROCESS" || exit 1
NAME="ouu10_alpha0.9_wide"
LOG="paper_tests/cluster/logs/${NAME}_h100.log"
export JAX_PLATFORMS=cuda
export PYTHONPATH="$HOME/PROCESS/paper_tests${PYTHONPATH:+:$PYTHONPATH}"
{
    echo "job $SLURM_JOB_ID on $(hostname); $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
    echo "== $(date '+%F %T') $NAME"
    python -u paper_tests/ouu.py --smoke --n 8192 --alpha 0.9 \
        --pairing one --table build --objective levelised --closure bracketed \
        --lifts winding_pack --chunk 2 \
        --bounds ".physics.rmajor:10,45;.tfcoil.t_tf_superconductor_quench:1,150" \
        --jac fwd --max-iter 400 --evidence --name "$NAME" --tag "_wide"
    echo "== $(date '+%F %T') exit $?"
} > "$LOG" 2>&1
