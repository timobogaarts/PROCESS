"""**Assemble once, solve many times** -- the entry point for the interactive regime."""

from __future__ import annotations

import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from functional_process.cottax import native  # noqa: E402
from functional_process.cottax.importer import read_indat  # noqa: E402
from functional_process.cottax.indat import (  # noqa: E402
    REFERENCE_INPUT_FILE,
    graph_for,
    machine_from_indat,
    switch_values_from_indat,
)
from functional_process.cottax.run_cold_matrix import (  # noqa: E402
    MdfBuild,
    SandBuild,
    build_mdf,
    build_sand,
    solve_mdf,
    solve_sand,
)
from functional_process.cottax.run_mda_harness import _resolve  # noqa: E402


@dataclass
class Session:
    """One configuration, assembled once, solvable any number of times."""

    name: str
    path: Path
    reference: object
    cold: object
    machine_graph: object = None
    switch_values: object = None
    root_find: bool = False
    boundary: dict = field(default_factory=dict)
    mdf_build: MdfBuild | None = None
    sand_build: SandBuild | None = None
    optimiser: object = None
    """The driver **class** every `Optimise` in this session is answered by, or `None`
    for `mda.default_drivers`' own choice.
    """

    def mdf(self, cold=None) -> dict:
        """Solve this configuration's MDF arm, assembling it on the first call."""
        if self.mdf_build is None:
            self.mdf_build = build_mdf(
                self.reference,
                self.machine_graph,
                self.switch_values,
                root_find=self.root_find,
            )
        return solve_mdf(
            self.mdf_build,
            self.reference,
            self.cold if cold is None else cold,
            optimiser=self.optimiser,
        )

    def sand(self, cold=None) -> dict:
        """Solve this configuration's SAND arm, assembling it on the first call."""
        if self.root_find:
            raise ValueError(
                f"{self.name} states a root find (`i_process_run_mode = -2`), so it "
                f"has no SAND arm: SAND distributes an optimisation over design AND "
                f"coupling and this file poses no optimisation to distribute. Use "
                f"`.mdf()`, which is PROCESS's own square system"
            )
        if self.sand_build is None:
            self.sand_build = build_sand(
                self.reference,
                self.machine_graph,
                self.switch_values,
                optimiser=self.optimiser,
            )
        return solve_sand(
            self.sand_build,
            self.reference,
            self.machine_graph,
            self.cold if cold is None else cold,
        )


def open_session(path, optimiser=None) -> Session:
    """Everything `run_cold_matrix.run_one` does before it first touches a solver."""
    path = _resolve(str(path))
    name = path.name[: -len(".IN.DAT")] if path.name.endswith(".IN.DAT") else path.stem
    is_reference = path == _resolve(REFERENCE_INPUT_FILE)
    machine = machine_from_indat(str(path))
    machine_graph = None if is_reference else graph_for(machine)
    root_find = read_indat(str(path)).problem.is_evaluation

    reference = native.native_reference(str(path))
    cold = reference.cold
    switch_values = None if is_reference else switch_values_from_indat(str(path))
    boundary: dict = {}
    return Session(
        optimiser=optimiser,
        name=name,
        path=path,
        reference=reference,
        cold=cold,
        machine_graph=machine_graph,
        switch_values=switch_values,
        root_find=root_find,
        boundary=boundary,
    )


def series(live: Session, repeats: int, arm: str = "mdf") -> list[dict]:
    """Solve `live` `repeats` times, timing each -- the instrument for this regime."""
    solve = live.mdf if arm == "mdf" else live.sand
    out = []
    for _ in range(repeats):
        compiles = _Compiles()
        began = time.perf_counter()
        try:
            answer = solve()
        finally:
            counted = compiles.stop()
        out.append({
            "seconds": time.perf_counter() - began,
            "compiles": counted,
            "rss": _rss(),
            **{k: answer.get(k) for k in ("iterations", "objf", "max_eq", "min_ie")},
            "_x": answer.get("_x"),
        })
    return out


class _Compiles:
    """How many XLA modules were compiled between construction and `stop`."""

    def __init__(self):
        try:
            from jax._src import compiler  # noqa: PLC0415, PLC2701
        except ImportError:  # pragma: no cover
            self.compiler = None
            return
        self.compiler = compiler
        self.original = getattr(compiler, "backend_compile_and_load", None)
        if self.original is None:  # pragma: no cover -- jax moved it
            self.compiler = None
            return
        self.n = 0

        def counted(*args, **kwargs):
            self.n += 1
            return self.original(*args, **kwargs)

        compiler.backend_compile_and_load = counted

    def stop(self):
        if self.compiler is None:  # pragma: no cover
            return None
        self.compiler.backend_compile_and_load = self.original
        return self.n


def _rss() -> float:
    """Resident set size in GiB, or `nan` off Linux."""
    try:
        with Path("/proc/self/status").open() as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:  # pragma: no cover -- not Linux
        pass
    return float("nan")


def render(live: Session, measured, arm: str) -> str:
    """The series as a table, plus whether the answer ever moved."""
    first = measured[0]
    lines = [
        "",
        (
            f"REPEATED SOLVE -- {live.name} {arm.upper()}, seeded natively, "
            f"one process, nothing cleared"
        ),
        "",
        "  solve   seconds  compiles   RSS/GiB   SQP it            objf   answer",
        "  " + "-" * 68,
    ]
    for index, row in enumerate(measured):
        same = row["_x"] == first["_x"]
        lines.append(
            f"  {index:5d} {row['seconds']:9.3f} {row['compiles']!s:>9} "
            f"{row['rss']:9.3f} {row['iterations']!s:>8} "
            f"{'-' if row['objf'] is None else format(row['objf'], '.9g'):>15}   "
            f"{'same' if same else 'MOVED'}"
        )
    tail = [row["seconds"] for row in measured[1:]] or [measured[0]["seconds"]]
    moved = [i for i, row in enumerate(measured) if row["_x"] != first["_x"]]
    lines += [
        "",
        (
            f"  first solve {measured[0]['seconds']:.2f} s, "
            f"steady state {statistics.median(tail):.2f} s (median of {len(tail)}), "
            f"min {min(tail):.2f} s"
        ),
        (
            f"  compiles after the first solve: "
            f"{sum(row['compiles'] or 0 for row in measured[1:])}"
        ),
        f"  RSS {measured[0]['rss']:.3f} -> {measured[-1]['rss']:.3f} GiB",
        (
            "  ANSWER STABLE across every repeat"
            if not moved
            else f"  ANSWER MOVED at solve(s) {moved} -- this is a DEFECT, not a timing"
        ),
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    """`--input <path>` (repeatable), `--repeat N`, `--arm mdf|sand|both`."""
    argv = sys.argv[1:] if argv is None else list(argv)
    inputs = [argv[i + 1] for i, a in enumerate(argv) if a == "--input"] or [
        REFERENCE_INPUT_FILE,
        "tests/regression/input_files/large_tokamak_nof.IN.DAT",
    ]
    repeats = int(argv[argv.index("--repeat") + 1]) if "--repeat" in argv else 6
    arm = argv[argv.index("--arm") + 1] if "--arm" in argv else "mdf"
    arms = ("mdf", "sand") if arm == "both" else (arm,)
    for name in inputs:
        live = open_session(name)
        for one in arms:
            if one == "sand" and live.root_find:
                print(f"\n{live.name}: no SAND arm (root find) -- skipped")
                continue
            print(render(live, series(live, repeats, one), one), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
