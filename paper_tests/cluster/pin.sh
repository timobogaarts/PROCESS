#!/bin/bash
# Run LOCALLY, after sync_to_cluster.sh. Pins the cottax the architecture jobs run on:
# exports ~/jaxgraph at COTTAX (default: its HEAD, which must be clean) to
# snellius:~/cottax_<commit>/src, and writes paper_tests/cluster/pins.env on the
# cluster naming both commits. The jobs source pins.env, put that export first on
# PYTHONPATH (ahead of the venv's editable ~/jaxgraph mirror, which moves with every
# sync), and stamp every csv with the two commits (bench.head reads COMMIT_*).
#
#     bash ~/PROCESS/paper_tests/cluster/pin.sh            # jaxgraph HEAD
#     COTTAX=078fb9f bash ~/PROCESS/paper_tests/cluster/pin.sh
set -euo pipefail
HOST="snellius.surf.nl"
ROOT="${REMOTE_HOME:-/home/tbogaarts}/PROCESS"   # the mirror the jobs run from

COTTAX="${COTTAX:-$(git -C ~/jaxgraph rev-parse --short HEAD)}"
if [ -z "${COTTAX_ALLOW_DIRTY:-}" ] && [ -n "$(git -C ~/jaxgraph status --porcelain -- src)" ]; then
    echo "~/jaxgraph/src has uncommitted changes; the export is of $COTTAX, not of them."
    echo "Commit first, or COTTAX_ALLOW_DIRTY=1 to export $COTTAX anyway."
    exit 1
fi
PROCESS="$(git -C ~/PROCESS rev-parse --short HEAD)"
# The mirror carries the working tree, so a dirty PROCESS is what runs: say so.
[ -z "$(git -C ~/PROCESS status --porcelain --untracked-files=no -- functional_process process paper_tests/architectures ':!paper_tests/architectures/out')" ] \
    || PROCESS="${PROCESS}+dirty"

echo "cottax $COTTAX -> $HOST:~/cottax_$COTTAX ; PROCESS $PROCESS"
git -C ~/jaxgraph archive "$COTTAX" src \
    | ssh "$HOST" "rm -rf ~/cottax_$COTTAX && mkdir -p ~/cottax_$COTTAX && tar -x -C ~/cottax_$COTTAX"
ssh "$HOST" "cat > $ROOT/paper_tests/cluster/pins.env" <<EOF
export COMMIT_COTTAX=$COTTAX
export COMMIT_PROCESS=$PROCESS
export COTTAX_SRC=\$HOME/cottax_$COTTAX/src
EOF
ssh "$HOST" "cat $ROOT/paper_tests/cluster/pins.env"
