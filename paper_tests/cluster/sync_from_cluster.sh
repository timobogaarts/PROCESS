#!/bin/bash
# Run LOCALLY. Pulls the cluster's paper_tests/out/ (the flexibility runs' per-draw
# JSONs) and the job logs down into paper_tests/out/ and paper_tests/cluster/logs/.
# Additive, no --delete (the shape of
# ~/blanket_optimization/scripts/rsync_from_cluster_add.sh). The per-draw JSONs are
# not tracked; `flexibility.py --figures` condenses them to the .npz/_summary pairs
# that are.
set -u
HOST="snellius.surf.nl"
REMOTE="${REMOTE_HOME:-/home/tbogaarts}/PROCESS/paper_tests"
DEST="$HOME/PROCESS/paper_tests/out"
LOGS="$HOME/PROCESS/paper_tests/cluster/logs"
COMMON_OPTS=(-avz --progress --exclude __pycache__)

read -rp "rsync $HOST:$REMOTE/out/ -> $DEST/ and cluster/logs/ -> $LOGS/ ? (y = yes, d = dry-run, anything else = cancel): " choice
case "$choice" in
    y|Y) DRY=() ;;
    d|D) DRY=(--dry-run); echo "Dry-run mode enabled." ;;
    *)   echo "Cancelled."; exit 0 ;;
esac
mkdir -p "$DEST" "$LOGS"
rsync "${COMMON_OPTS[@]}" "${DRY[@]}" "$HOST:$REMOTE/out/" "$DEST/"
rsync "${COMMON_OPTS[@]}" "${DRY[@]}" "$HOST:$REMOTE/cluster/logs/" "$LOGS/"
# The architecture tables' csvs: they replace the laptop's rows of the same name (those
# stay in git history), so every row table.py reads was measured on the cluster.
rsync "${COMMON_OPTS[@]}" "${DRY[@]}" "$HOST:$REMOTE/architectures/out/" "$HOME/PROCESS/paper_tests/architectures/out/"
