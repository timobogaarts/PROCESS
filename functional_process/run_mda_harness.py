"""Run the block-by-block MDA-vs-PROCESS comparison and print the report.

    $PY -m functional_process.run_mda_harness                      # the stellarator
    $PY -m functional_process.run_mda_harness --input <IN.DAT>     # any other machine
    $PY -m functional_process.run_mda_harness --machine            # the tokamak

With no argument this is exactly what it always was: the Helias stellarator,
`tests/regression/input_files/stellarator_helias.IN.DAT`, whose numbers other records
quote. `--input` names a different `IN.DAT` and `--machine` is shorthand for
`boundary.TOKAMAK_INPUT_FILE`, the conventional large tokamak.

**The smallest clean extension, and deliberately the same one `boundary.py` already
made** (`--machine [<IN.DAT>]`, `_machine_graph`): the two entry points now take the
machine the same way, so a reader who has seen one has seen both. What is *not* shared is
the copy-to-scratch step, which only this one needs.

Copies the input file (and its `.stella_conf.json` companion, where it has one) into a
scratch directory first, since `SingleRun` writes `OUT.DAT`/`MFILE.DAT` beside its input.
The copy keeps the file's own name, which is what `mda_harness._cache_key` hashes -- so
the converged-run cache hits across invocations, and (since `_CACHE_VERSION` v2) two
different input files can no longer share one entry.

**The graph is built from the input file, never described here.** `machine_from_indat` is
the single place an `i_*` integer is read, and `graph_for()` with no argument is
`REFERENCE_MACHINE` -- `machine_from_indat` applied to `stellarator_helias.IN.DAT`,
checked against the file itself by
`test_machine.py::test_reference_machine_matches_the_input_file`. This module used to
spell the stellarator's switch choices out in its own docstring, which is exactly the
arrangement that let five registration bugs through: the harness knew the run's real
configuration and nothing else did.
"""

import shutil
import sys
import tempfile
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from functional_process.boundary import TOKAMAK_INPUT_FILE  # noqa: E402
from functional_process.indat import (  # noqa: E402
    REFERENCE_INPUT_FILE,
    graph_for,
    machine_from_indat,
)
from functional_process.mda_harness import compare, converged_data  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def input_file(argv: list[str]) -> Path:
    """The `IN.DAT` this invocation is about -- see the module docstring.

    `--input` wins over `--machine` if both are given, since it is the more specific of
    the two; neither means the stellarator reference file.

    Raises
    ------
    SystemExit
        If `--input` is given with no path after it -- a harness that silently fell back
        to the stellarator there would print a full report for the wrong machine.
    """
    if "--input" in argv:
        index = argv.index("--input") + 1
        if index >= len(argv) or argv[index].startswith("-"):
            raise SystemExit("--input needs a path to an IN.DAT")
        return _resolve(argv[index])
    if "--machine" in argv:
        return _resolve(TOKAMAK_INPUT_FILE)
    return _resolve(REFERENCE_INPUT_FILE)


def _resolve(name: str) -> Path:
    """`name` as an absolute path, read relative to the repository root when it is not
    already absolute.

    `REFERENCE_INPUT_FILE` and `TOKAMAK_INPUT_FILE` are both spelled repo-relative
    (`tests/regression/input_files/...`), which works from the repo root and nowhere
    else. Anchoring on `__file__` rather than on the working directory is what lets this
    be run from anywhere, and is what the old hard-coded `Path(__file__).parent.parent /
    "tests/..."` was doing by hand.
    """
    path = Path(name)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _sidecars(path: Path) -> tuple[Path, ...]:
    """The companion files a `SingleRun` on `path` needs beside it.

    One shape today, and it is PROCESS's own `output_prefix` convention -- the stem
    before `.IN.DAT`, plus `.stella_conf.json`, which `Stellarator.st_new_config()`
    opens for `istell == 6`. A tokamak has none, and its absence is not an error:
    `mda_harness._cache_key` already hashes the sidecar's absence *as* absence, so the
    two devices cannot collide in the cache through this.
    """
    stem = path.name[: -len(".IN.DAT")] if path.name.endswith(".IN.DAT") else path.stem
    companion = path.parent / f"{stem}.stella_conf.json"
    return (companion,) if companion.is_file() else ()


def main(argv: list[str] | None = None) -> None:
    """Solve `input_file(argv)` with PROCESS, run the graph it describes, print the
    diff. `argv` defaults to this process's own, so importing and calling `main([...])`
    is the same thing as the command line.
    """
    argv = sys.argv[1:] if argv is None else argv
    path = input_file(argv)
    print(f"input file:     {path}")

    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp) / path.name
        shutil.copy(path, scratch)
        for companion in _sidecars(path):
            shutil.copy(companion, Path(tmp) / companion.name)

        data = converged_data(str(scratch))

    # Built from the file, not from the scratch copy: identical bytes either way, and
    # the real path is what an error message should name.
    graph = graph_for(machine_from_indat(str(path)))
    print(f"declared graph: {len(graph.nodes)} nodes")
    report = compare(graph, data)
    print(report.summary())


if __name__ == "__main__":
    main()
