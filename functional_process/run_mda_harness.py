"""Run the block-by-block MDA-vs-PROCESS comparison against
`tests/regression/input_files/stellarator_helias.IN.DAT` and print the report.

    $PY functional_process/run_mda_harness.py

Copies the input file (and its `.stella_conf.json` companion, required for
`istell == 6`) into a scratch directory first, since `SingleRun` writes
`OUT.DAT`/`MFILE.DAT` beside its input.

`indat.GRAPH`'s bare default does **not** match this file: it sets
`i_plasma_pedestal = 0`, differing from `graph_for()`'s own default (`1`). Every
other topology switch (`isthtr`, `ipowerflow`, `i_bldgs_size`, the joint
`blktmodel,ipowerflow` switch, `i_tf_sup`) matches PROCESS's own default, confirmed
against the converged run directly, not assumed from the input file's text (several
of these are switches this file's `IN.DAT` never even mentions).
"""

import shutil
import tempfile
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from functional_process.indat import graph_for  # noqa: E402
from functional_process.mda_harness import compare, converged_data  # noqa: E402

INPUT_FILE = (
    Path(__file__).resolve().parent.parent
    / "tests/regression/input_files/stellarator_helias.IN.DAT"
)
STELLA_CONF = INPUT_FILE.with_name("stellarator_helias.stella_conf.json")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp) / INPUT_FILE.name
        shutil.copy(INPUT_FILE, scratch)
        shutil.copy(STELLA_CONF, Path(tmp) / STELLA_CONF.name)

        data = converged_data(str(scratch))

    # `graph_for()` with no argument IS `REFERENCE_MACHINE` -- `machine_from_indat`
    # applied to `stellarator_helias.IN.DAT`, checked against the file itself by
    # `test_machine.py::test_reference_machine_matches_the_input_file`. This used to
    # spell the switch choices out here, which is exactly the arrangement that let five
    # registration bugs through: the harness knew the run's real configuration and
    # nothing else did.
    graph = graph_for()
    report = compare(graph, data)
    print(report.summary())


if __name__ == "__main__":
    main()
