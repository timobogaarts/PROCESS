"""Run the block-by-block MDA-vs-PROCESS comparison and print the report."""

import shutil
import sys
import tempfile
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from functional_process.cottax.boundary import TOKAMAK_INPUT_FILE  # noqa: E402
from functional_process.cottax.indat import (  # noqa: E402
    REFERENCE_INPUT_FILE,
    graph_for,
    machine_from_indat,
)
from functional_process.cottax.mda_harness import compare, converged_data  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent


def input_file(argv: list[str]) -> Path:
    """The `IN.DAT` this invocation is about -- see the module docstring."""
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
    """
    path = Path(name)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _sidecars(path: Path) -> tuple[Path, ...]:
    """The companion files a `SingleRun` on `path` needs beside it."""
    stem = path.name[: -len(".IN.DAT")] if path.name.endswith(".IN.DAT") else path.stem
    companion = path.parent / f"{stem}.stella_conf.json"
    return (companion,) if companion.is_file() else ()


def main(argv: list[str] | None = None) -> None:
    """Solve `input_file(argv)` with PROCESS, run the graph it describes, print the
    diff.
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
