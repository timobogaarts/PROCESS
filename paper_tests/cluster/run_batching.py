"""`paper_tests/batching.py`, unmodified, with two things the cluster runs need that
its command line does not expose:

* `BATCHING_WALL_LIMIT` (seconds) overrides `batching.WALL_LIMIT`, the warm-call wall
  past which a shape's size ladder stops (60 s in the script -- the tokamak MDA on the
  H100 would end its ladder around N=65536 under that).
* every row is stamped with `host` (the node) and `cpu` (the model, as the `.tex`
  provenance line already records it) before `batching.merge` sees it, so the CPU rows
  can be told from the laptop's; the GPU rows already carry `device`.

Same arguments as `batching.py`; `--backend` is read off `sys.argv` at import, as there.
"""

import os
import socket
import sys
from pathlib import Path

PAPER_TESTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER_TESTS))


def prepare():
    import batching  # noqa: PLC0415 -- settles JAX_PLATFORMS from sys.argv on import
    from common import machine  # noqa: PLC0415

    if "BATCHING_WALL_LIMIT" in os.environ:
        batching.WALL_LIMIT = float(os.environ["BATCHING_WALL_LIMIT"])
    stamp = {"host": socket.gethostname(), "cpu": machine()}
    original_merge = batching.merge

    def merge(new_rows):
        for row in new_rows:
            for key, value in stamp.items():
                row.setdefault(key, value)
        return original_merge(new_rows)

    batching.merge = merge
    return batching


if __name__ == "__main__":
    raise SystemExit(prepare().main())
