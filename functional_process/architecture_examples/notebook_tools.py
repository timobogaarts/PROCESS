"""Helpers the case notebooks share.

- `print_recipe(plan)`: print the graph operations a plan applied, one per line.
- `execute_notebook(path)`: run a notebook headlessly, in a fresh kernel, and keep its
  outputs. Used by `tests/examples` and by

      python -m functional_process.architecture_examples.notebook_tools <case>.ipynb

  It drives the kernel through `jupyter_client` directly (the env has no `nbclient`),
  on the current interpreter, with the notebook's folder as working directory, as a
  Jupyter front end would.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import nbformat

REPO = Path(__file__).resolve().parents[2]
"""The `PROCESS/` checkout: what the notebooks `chdir` to."""

JAXGRAPH_SRC = REPO.parent.parent / "jaxgraph" / "src"
"""The editable cottax checkout beside this repo, if the layout is the documented one
(`two_opt_driver/scripts/README.md`: `PYTHONPATH=~/projects/jaxgraph/src:.`)."""

DEFAULT_TIMEOUT = 1800
"""Seconds one cell may take: a cold assembly compiles an MDA and solves twice."""


def print_recipe(plan, width: int = 110) -> None:
    """Print the operations `plan` applied to its graph, one per line, in order."""
    for op in plan.ops:
        text = repr(op)
        print("  ", text if len(text) <= width else text[:width] + " ...")


def _output_from(msg) -> nbformat.NotebookNode | None:
    """One iopub message as an `nbformat` output, or `None` for the ones that are not."""
    kind = msg["msg_type"]
    content = msg["content"]
    if kind == "stream":
        return nbformat.v4.new_output("stream", name=content["name"], text=content["text"])
    if kind in ("display_data", "execute_result"):
        out = nbformat.v4.new_output(
            kind, data=content["data"], metadata=content.get("metadata", {})
        )
        if kind == "execute_result":
            out["execution_count"] = content.get("execution_count")
        return out
    if kind == "error":
        return nbformat.v4.new_output(
            "error",
            ename=content["ename"],
            evalue=content["evalue"],
            traceback=content["traceback"],
        )
    return None


def _kernel_env() -> dict:
    """The kernel's environment: this repo on the path, the sibling cottax checkout ahead
    of any installed copy, jax on the CPU."""
    env = dict(os.environ)
    entries = [str(REPO)]
    if JAXGRAPH_SRC.is_dir():
        entries.insert(0, str(JAXGRAPH_SRC))
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(entries + ([existing] if existing else []))
    env.setdefault("JAX_PLATFORMS", "cpu")
    return env


def execute_notebook(
    path,
    output=None,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    user_expressions: dict | None = None,
) -> tuple[nbformat.NotebookNode, dict]:
    """Run every code cell of `path` in order in a fresh kernel. Returns the executed
    notebook and, for each entry of `user_expressions` (`{name: expression}`), the
    `repr` of that expression evaluated after the last cell.

    `output`: where to write the executed notebook (`None`: nowhere; the notebook's own
    path: in place). A cell that raises stops the run; the error is re-raised as
    `NotebookError` with the cell index and traceback.
    """
    from jupyter_client.manager import KernelManager  # noqa: PLC0415

    path = Path(path).resolve()
    nb = nbformat.read(path, as_version=4)
    manager = KernelManager(
        kernel_cmd=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    )
    manager.start_kernel(cwd=str(path.parent), env=_kernel_env())
    client = manager.client()
    client.start_channels()
    answers: dict = {}
    try:
        client.wait_for_ready(timeout=120)
        count = 0
        for index, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            outputs: list = []
            began = time.perf_counter()
            reply = client.execute_interactive(
                cell.source,
                timeout=timeout,
                output_hook=lambda msg: (
                    (out := _output_from(msg)) is not None and outputs.append(out)
                ),
            )
            count += 1
            cell.outputs = outputs
            cell.execution_count = count
            # A string: the schema types every `execution` entry as one.
            cell.metadata.setdefault("execution", {})["seconds"] = (
                f"{time.perf_counter() - began:.2f}"
            )
            if reply["content"]["status"] != "ok":
                if output is not None:
                    nbformat.write(nb, output)
                content = reply["content"]
                raise NotebookError(path, index, content)
        if user_expressions:
            reply = client.execute_interactive(
                "",
                timeout=timeout,
                user_expressions=dict(user_expressions),   # the kernel answers each as its repr
            )
            for name, answer in reply["content"]["user_expressions"].items():
                if answer["status"] != "ok":
                    raise NotebookError(path, None, answer)
                answers[name] = answer["data"]["text/plain"]
    finally:
        client.stop_channels()
        manager.shutdown_kernel(now=True)
    if output is not None:
        nbformat.write(nb, output)
    return nb, answers


class NotebookError(RuntimeError):
    """A cell raised. `cell` is its index in the notebook (`None`: a user expression)."""

    def __init__(self, path, cell, content):
        self.path, self.cell, self.content = path, cell, content
        where = "user expression" if cell is None else f"cell {cell}"
        trace = "\n".join(content.get("traceback", ())) or content.get("evalue", "")
        super().__init__(f"{path.name}, {where}: {content.get('ename')}\n{trace}")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 2
    for name in argv:
        path = Path(name)
        began = time.perf_counter()
        execute_notebook(path, output=path)
        print(f"{path}: executed in place, {time.perf_counter() - began:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
