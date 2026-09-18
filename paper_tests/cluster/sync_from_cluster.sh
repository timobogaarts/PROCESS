#!/bin/bash
# Run LOCALLY. Pulls the cluster's paper_tests/out/ and the job logs down into
# ~/PROCESS/paper_tests/out_cluster/ (results) and out_cluster/logs/ (logs), so the
# laptop's paper_tests/out/ is never overwritten. Additive, no --delete (the shape of
# ~/blanket_optimization/scripts/rsync_from_cluster_add.sh).
set -u
HOST="snellius.surf.nl"
REMOTE="/home/tbogaarts/PROCESS/paper_tests"
DEST="$HOME/PROCESS/paper_tests/out_cluster"
COMMON_OPTS=(-avz --progress --exclude __pycache__)

read -rp "rsync $HOST:$REMOTE/{out,cluster/logs}/ -> $DEST/ ? (y = yes, d = dry-run, anything else = cancel): " choice
case "$choice" in
    y|Y) DRY=() ;;
    d|D) DRY=(--dry-run); echo "Dry-run mode enabled." ;;
    *)   echo "Cancelled."; exit 0 ;;
esac
mkdir -p "$DEST/logs"
rsync "${COMMON_OPTS[@]}" "${DRY[@]}" "$HOST:$REMOTE/out/" "$DEST/"
rsync "${COMMON_OPTS[@]}" "${DRY[@]}" "$HOST:$REMOTE/cluster/logs/" "$DEST/logs/"
