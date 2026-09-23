#!/bin/bash
# Run LOCALLY. Mirrors ~/jaxgraph and ~/PROCESS to snellius.surf.nl:/home/tbogaarts/
# (same shape as ~/blanket_optimization/scripts/rsync_to_cluster.sh: y / d / cancel).
#
# --delete makes the remote trees match the local ones; excluded paths are protected
# from it, so the remote's .git, its egg-info, paper_tests/out/ (the cluster's own
# results) and paper_tests/cluster/logs/ survive a re-sync. paper_tests/out/ is
# deliberately NOT sent up: batching.py merges into an existing out/batching.json on
# a key without the device, so the H100 rows would overwrite the RTX 3080 rows.
# Note: ~/jaxvenv on Snellius resolves cottax editable from ~/jaxgraph, so this also
# moves the cottax that venv sees.
set -u
HOST="snellius.surf.nl"
# REMOTE_HOME / SYNC_SRC retarget it: the architecture jobs mirror PROCESS alone into a
# fresh ~/arch_run (their cottax is pin.sh's export), leaving the old mirror as it is.
DEST="$HOST:${REMOTE_HOME:-/home/tbogaarts}/"
# shellcheck disable=SC2206
SRC=(${SYNC_SRC:-jaxgraph PROCESS})  # relative to $HOME, so the remote layout matches
COMMON_OPTS=(-avz --progress --delete)
EXCLUDES=(
    --exclude .git --exclude __pycache__ --exclude '*.pyc' --exclude '*.egg-info'
    --exclude .pytest_cache --exclude .ruff_cache --exclude .claude
    --exclude .jupyter_cache --exclude .notebook_runs --exclude .ipynb_checkpoints
    --exclude /jaxgraph/docs/build --exclude /jaxgraph/build
    --exclude /PROCESS/paper_tests/out --exclude /PROCESS/paper_tests/out_cluster
    --exclude /PROCESS/paper_tests/cluster/logs
    --exclude /PROCESS/paper_tests/cluster/pins.env      # pin.sh writes it there
    --exclude /PROCESS/paper_tests/architectures/out     # the cluster's own rows
    --exclude /PROCESS/documentation           # 70 MB of docs the study never reads
    --exclude /PROCESS/process.log --exclude /PROCESS/err.log --exclude '*.process.log'
    --exclude /PROCESS/OUT.DAT --exclude /PROCESS/MFILE.DAT --exclude /PROCESS/tracking
)

cd "$HOME" || exit 1
read -rp "rsync ${SRC[*]} -> $DEST ? (y = yes, d = dry-run, anything else = cancel): " choice
case "$choice" in
    y|Y)
        echo "Running rsync..."
        rsync "${COMMON_OPTS[@]}" "${EXCLUDES[@]}" "${SRC[@]}" "$DEST"
        ;;
    d|D)
        echo "Dry-run mode enabled."
        rsync "${COMMON_OPTS[@]}" "${EXCLUDES[@]}" --dry-run "${SRC[@]}" "$DEST"
        ;;
    *)
        echo "Cancelled."
        ;;
esac
