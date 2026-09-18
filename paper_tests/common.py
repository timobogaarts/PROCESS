"""What every generator here shares: the environment pinned, the configurations, the
provenance line, PROCESS's own converged answer, and LaTeX/CSV writers that land in
`paper_tests/out/`.

Nothing in the paper is typed by hand from a terminal: each script writes a `.csv`
(the numbers, for a reader with a script) and a `.tex` fragment (a `tabular`, for
`\\input`), both named after the script, plus a `.json` where the raw rows are worth
keeping. Every fragment starts with a comment line saying which script, which tree,
which cottax commit and which machine produced it.

The scripts are thin drivers over the port (`functional_process.cottax.architectures`,
`functional_process.configurations`): the physics, the graph operations and the
statistics live there; what lives here is the command line, the rendering, and the
one thing the port deliberately does not carry -- PROCESS itself in the loop
(`process_reference`).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
CACHE_DIR = Path(
    os.environ.get("FP_HARNESS_CACHE_DIR", Path.home() / ".cache/functional_process")
)
"""Where PROCESS's reference runs and the studies' raw samples land -- the same
directory the deleted harness used, so a cluster's `env.sh` still points at it."""

sys.path.insert(0, str(ROOT))

from functional_process import configurations  # noqa: E402
from functional_process.cottax.input import native  # noqa: E402

CONFIGURATIONS = tuple(native.CONFIGURATIONS)
"""The seven reference input files, in the order every table lists them."""

NAMES = tuple(configurations.NAMES)
"""The same seven as stated configurations (`functional_process/configurations/`)."""

STELLARATORS = tuple(c for c in CONFIGURATIONS if "helias" in c)

RECIPES = ("hand", "jacobi", "gauss_seidel", "gauss_seidel_minimal")
"""The cuts a table compares: `mda.CUTS` and the three `recipes.RECIPES`."""

LABEL = {
    "hand": "hand",
    "jacobi": "Jacobi",
    "gauss_seidel": "GS (call order)",
    "gauss_seidel_minimal": "GS (minimal)",
}


def stem(path: str) -> str:
    """`.../helias_5b.IN.DAT` -> `helias_5b`; a configuration name is its own stem."""
    return Path(path).name.replace(".IN.DAT", "")


def input_file(name: str) -> Path:
    """The regression `IN.DAT` a configuration name or path refers to, absolute."""
    path = Path(name)
    if path.suffix == ".DAT" or path.exists():
        return path if path.is_absolute() else (ROOT / path).resolve()
    return (ROOT / "tests/regression/input_files" / f"{stem(name)}.IN.DAT").resolve()


def tex_name(name: str) -> str:
    """A configuration name as LaTeX: underscores escaped."""
    return name.replace("_", r"\_")


def cut_for(name: str):
    """`None` for the hand-measured cuts, else the recipe."""
    from functional_process.cottax.architectures.recipes import recipe  # noqa: PLC0415

    return None if name == "hand" else recipe(name)


# ---------------------------------------------------------------- provenance


def _git(*args, cwd) -> str:
    """One git query, or `""` -- a header line, never a reason a run fails."""
    try:
        return subprocess.run(  # noqa: S603
            ["git", *args],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=20,
            cwd=str(cwd),
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return ""


def machine() -> str:
    """The CPU these rows were timed on. A timing without this is not comparable
    with anything: the 2026-09-06 references were taken on a laptop and re-run on an
    R7 3700X on 2026-09-16, and neither file said so.
    """
    model = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return f"{model or platform.processor() or 'unknown CPU'}, {platform.node()}"


def cottax_tree() -> str:
    """Which `cottax` commit answered: the port tracks an API that moves, and a row is
    a measurement of the pair, not of this tree alone.
    """
    import cottax  # noqa: PLC0415

    src = Path(cottax.__file__).resolve().parent.parent
    head = _git("rev-parse", "--short", "HEAD", cwd=src) or "unknown"
    dirty = " (+ uncommitted edits)" if _git("status", "--porcelain", cwd=src) else ""
    return f"{head}{dirty} at {src}"


def provenance(script: str) -> str:
    """One comment line: script, tree, cottax, machine, time."""
    head = _git("rev-parse", "--short", "HEAD", cwd=ROOT) or "unknown"
    dirty = " (+uncommitted)" if _git("status", "--porcelain", cwd=ROOT) else ""
    return (
        f"% generated by paper_tests/{script} on {time.strftime('%Y-%m-%d %H:%M')}; "
        f"PROCESS {head}{dirty}; cottax {cottax_tree().split(' at ')[0]}; {machine()}"
    )


def return_freed_memory_to_the_os() -> None:
    """`malloc_trim(0)`, because `jax.clear_caches()` frees memory it does not give
    back.
    """
    import ctypes  # noqa: PLC0415

    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):  # not glibc, or no symbol -- not fatal
        pass


# ---------------------------------------------------------------- PROCESS itself

_REFERENCE_VERSION = "process-reference-v1"


def process_reference(name: str = "stellarator_helias", use_cache: bool = True) -> dict:
    """PROCESS's own converged answer for one regression file: `x` (`{spelling:
    value}` over its `ixc`, the port's spellings), `iterations` (VMCON's count),
    `outputs` (net electric power, coe, concost, fusion power, the density) and
    `seconds`. Run in-process (`process.main.SingleRun`, ~1-2 min for the
    stellarator) and cached as JSON under `CACHE_DIR`, keyed by the file's content.

    The port's `session` is deliberately PROCESS-free; a comparison column against
    PROCESS is this directory's business, so the run lives here.
    """
    path = input_file(name)
    digest = hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()  # noqa: S324
    cached = CACHE_DIR / f"{_REFERENCE_VERSION}-{digest}.json"
    if use_cache and cached.exists():
        return json.loads(cached.read_text())
    import shutil  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    from process.core.solver.iteration_variables import (  # noqa: PLC0415
        ITERATION_VARIABLES,
    )
    from process.main import SingleRun  # noqa: PLC0415

    from functional_process.cottax.architectures.sand import (  # noqa: PLC0415
        iteration_variable_path,
    )

    scratch = Path(tempfile.mkdtemp())
    shutil.copy(path, scratch / path.name)
    companion = path.with_name(f"{stem(str(path))}.stella_conf.json")
    if companion.exists():
        shutil.copy(companion, scratch / companion.name)
    run = SingleRun(str(scratch / path.name), "vmcon")
    began = time.perf_counter()
    run.run()
    seconds = time.perf_counter() - began
    data = run.data
    n = int(data.numerics.n_iteration_variables)
    ixc = [int(i) for i in data.numerics.ixc[:n]]

    def value(i):
        iteration_variable = ITERATION_VARIABLES[i]
        area = getattr(data, iteration_variable.module)
        field = getattr(area, iteration_variable.target_name or iteration_variable.name)
        if iteration_variable.array_index is None:
            return float(field)
        return float(field[iteration_variable.array_index])

    result = {
        "file": str(path),
        "ixc": ixc,
        "x": {iteration_variable_path(i).spelling: value(i) for i in ixc},
        "iterations": int(data.numerics.n_solver_iterations),
        "convergence_parameter": float(data.globals.convergence_parameter),
        "outputs": {
            ".heat_transport.p_plant_electric_net_mw": float(
                data.heat_transport.p_plant_electric_net_mw
            ),
            ".costs.coe": float(data.costs.coe),
            ".costs.concost": float(data.costs.concost),
            ".physics.p_fusion_total_mw": float(data.physics.p_fusion_total_mw),
            ".physics.nd_plasma_electrons_vol_avg": float(
                data.physics.nd_plasma_electrons_vol_avg
            ),
            "^cond.numerics.objf": float(data.costs.coe) / 100.0,
        },
        "seconds": seconds,
    }
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(result, indent=1))
    return result


def deterministic_values(live, pairing: str = "one") -> tuple[list, dict]:
    """PROCESS's converged design for `live`'s configuration, as `ouu.two_stage` /
    `closing.seed` take it: `design_values` (one per design place kept, `ixc` order
    minus the closing variables of `kinds.PAIRINGS[pairing]`) and `closing_values`
    (`{closing variable: value}`). The deterministic optimum every study here starts
    from or is compared against.
    """
    from functional_process.configurations import kinds  # noqa: PLC0415
    from functional_process.cottax.architectures.sand import (  # noqa: PLC0415
        iteration_variable_path,
    )

    process = process_reference(live.name)
    closing = set(kinds.PAIRINGS[pairing].values())
    design = [iteration_variable_path(i).spelling for i in live.reference.ixc]
    return (
        [process["x"][v] for v in design if v not in closing],
        {v: process["x"][v] for v in closing},
    )


# ---------------------------------------------------------------- writers


def write_csv(script: str, rows: list[dict], name: str | None = None) -> Path:
    path = OUT / f"{name or script.removesuffix('.py')}.csv"
    if not rows:
        path.write_text("")
        return path
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(row.get(k)) for k in keys})
    return path


def write_json(script: str, payload, name: str | None = None) -> Path:
    path = OUT / f"{name or script.removesuffix('.py')}.json"
    path.write_text(json.dumps(payload, indent=1, default=str))
    return path


def write_tex(script: str, header: list[str], rows: list[list], name: str | None = None,
              align: str | None = None, caption_note: str = "") -> Path:
    """A booktabs `tabular` fragment. `header` and each row are already LaTeX."""
    path = OUT / f"{name or script.removesuffix('.py')}.tex"
    align = align or ("l" + "r" * (len(header) - 1))
    lines = [provenance(script)]
    if caption_note:
        lines.append(f"% {caption_note}")
    lines += [
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(header) + r" \\",
        "\\midrule",
        *(" & ".join(str(c) for c in row) + r" \\" for row in rows),
        "\\bottomrule",
        "\\end{tabular}",
        "",
    ]
    path.write_text("\n".join(lines))
    return path


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.10g}"
    return value


def fmt(value, digits=3) -> str:
    """A number for a table cell, or `--`."""
    if value is None:
        return "--"
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def sci(value) -> str:
    if value is None:
        return "--"
    return f"{value:.2e}"


def option(argv, name, default, cast=int):
    """`--name value` off `argv`, cast, or `default`."""
    return cast(argv[argv.index(name) + 1]) if name in argv else default
