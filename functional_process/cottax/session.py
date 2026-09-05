"""**Assemble once, solve many times** -- the entry point for the interactive regime.

    $PY -m functional_process.cottax.session --input a.IN.DAT --repeat 8
    $PY -m functional_process.cottax.session --repeat 8 --arm sand

or, from a notebook:

    from functional_process import session
    live = session.open_session("tests/regression/input_files/stellarator_helias.IN.DAT")
    first  = live.mdf()                 # ~25 s: this is where the compiler is paid
    second = live.mdf()                 # ~1 s: nothing is compiled, nothing is traced
    moved  = live.mdf(cold=other_state) # a scan point -- a re-seed, not a rebuild

Why this module exists
----------------------
`run_cold_matrix.run_one` is the discoverable way to solve a configuration, and it is
the **wrong** loop to call twice. It assembles the graph, and a freshly assembled graph
is a structurally equal but *newly allocated* object, so every memo downstream of it
misses:

- `host_cache._BOUND` finds a matching `treedef` and a never-matching `static`, so the
  block is re-traced, re-lowered and re-compiled;
- `sand_harness._SCHEDULE_WHOLE`/`_SCHEDULE_RUNNERS` are keyed on the `Schedule` object;
- jax's own executable cache is keyed on what those two hand it.

**The first of those three is no longer true, and the module's premise is weaker than it
was** (`_audit/optimise_design.md` §37, 2026-09-04). `host_cache._BOUND` does not exist:
the three programs are module level and the block's structure rides as a value-compared
jit static argument, so a *re-assembled* block is a jax cache hit. Measured on
`stellarator_helias` MDF -- assemble, seed, bind and call all three programs, twice, the
second time from scratch: **19.29 s and 3 compiles before, 0.24 s and 0 compiles after.**

**That is the bind-and-call probe over three programs, not a whole solve.** A full
`open_session(...).mdf()` run twice measured **29 compiles / 19.3 s, then 0 / 0.9 s**
(2026-09-05, `stellarator_walkthrough.ipynb`). The 29 was structural: only four of those
programs cost anything -- `host_cache`'s two for the outer `Optimise` block, and the
fused schedule traced once each by `prime` and `solve` -- and the other 25 were scalar
reshapes at ~7 ms.

**The fused schedule's second trace is now gone too** (`_audit/optimise_design.md` §39,
2026-09-05). `mdf.seed`'s env and `mdf.solve`'s re-seeded env disagreed on jax's own
`weak_type` bit at every design variable and every `^guess.*` port -- `mdf.seed` wraps a
bare Python `float` in `jnp.asarray`, which is *weakly* typed, while `VmconDriver`'s
answer comes back through a NumPy array, which never is. Same shape, same dtype,
different abstract value, so `eqx.filter_jit` traced and compiled the schedule a second
time under `solve` even though it is the identical `Schedule` object `prime` already ran.
`mdf.seed` now forces `weak_type` off on every value it hands the schedule
(`mdf._not_weak`), which is exactly the fix `mda._root_find_seed` was for the two
programs behind it: make what differs compare equal. Measured on `stellarator_helias`
MDF: **29 -> 27 compiles, 19-22 s -> ~17.5 s** on the first solve, **0 compiles either
way** on the second, and the answer bit-identical throughout (`_x` and `objf` compared as
`float.hex()`, not rounded). The 25 trivial compiles are *not* `models/stated.py` --
every one of `mdf.seed`'s ~350 inputs, stated or ordinary, shares the same handful of
(shape, dtype) signatures, so they were already deduplicated to a handful of programs
before this fix; of the 25, 8 are `mdf.seed`'s own eager type-staging (now folded into
one XLA op instead of two, at `mdf._not_weak`) and the other 17 are inside
`core/solver/drivers.py` -- `optimistix`'s own trace-time constant-folding the first time
`SeededNewtonDriver` is traced, and `VmconDriver`'s `ravel_pytree`/`_sqp_callback`
setup -- out of this module's reach and, on the numbers above, not worth chasing: each is
a one-time ~7 ms cost paid once per `Session`, not per scan point.

**Two compiles used to survive every re-assembly, and finding out why is worth recording**
because it is `_audit`'s own warning landing in the one place nobody had looked.
`mda._root_find_seed(problem)` returned a closure whose `problem` argument the body never
used, so every assembly minted a fresh function object. `SeededNewtonDriver` compares that
`seed` field by identity, so **one leaf of a ~150-node graph** made the whole re-assembled
`Schedule` compare unequal, missing `sand_harness._SCHEDULE_WHOLE`'s memo and recompiling
the two large programs behind it. Hoisting the body to module level took re-assembly from
2 compiles / 8.0 s to 0 / 0.9 s, with the answer bit-identical throughout.

What remains true is that re-assembly is still *work* -- `machine_from_indat`, the graph,
`mdf.assemble` -- so this module is still the right way to walk a scan; it is no longer
the difference between a second and a minute for the MDF path. The numbers below are
`dcda0769`'s and are kept as the measurement they were:

Measured (`_audit/optimise_design.md` §32.2, 2026-09-03): the naive loop -- `run_one` in
a `for` -- ran **16 XLA compiles on every repeat, for ever**, grew `_BOUND` by two
entries a solve and did not converge on a steady state. The same configuration solved
through this module compiled on solve 0 and **not once afterwards**, and per-solve wall
clock flattened immediately:

| `stellarator_helias` MDF | solve 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| through `run_one` | 51.8 s | 44.9 | 41.1 | 42.0 | -- | -- |
| through this module | 22.5 s | **1.4** | **1.5** | **1.2** | **1.3** | **1.2** |

**The trap is re-assembly, not the caches.** `run_cold_matrix.main`'s per-row
`jax.clear_caches()` is often mistaken for the cause and is not it: those exist because seven *different* configurations do not share a single
cache entry between them, and dropping the previous row's executables is what keeps a
whole pass inside memory (§31.16). A session holds **one** configuration and clears
nothing.

What is and is not repeated
---------------------------
Repeated per solve, and cheap: `mdf.seed` (or SAND's `_seed`) and `mdf.prime`.
Measured warm on `stellarator_helias`, seed plus prime is **10-20 ms with zero
compiles**, and a full re-seed from a *different* `DataStructure` is 25-42 ms, also with
zero compiles. So re-seeding per point is supported and is the intended way to walk a
scan: hand `mdf`/`sand` a different `cold`.

Not repeated: `machine_from_indat`, the graph, `mdf.assemble`/`sand_harness.assemble`,
the SAND solve schedule -- and, therefore, every trace and every compile.

The answers are the matrix's answers, by construction
-----------------------------------------------------
Nothing here computes anything. `open_session` is `run_cold_matrix.run_one`'s own
preparation and `Session.mdf`/`Session.sand` are `run_cold_matrix.solve_mdf`/`solve_sand`
-- the *same* functions `cold_mdf`/`cold_sand` call, which is why a session's row and a
matrix row agree by construction rather than by comparison. That is
`run_cold_matrix.py`'s own rule for why it is not a third harness, applied once more.
"""

from __future__ import annotations

import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from functional_process.cottax import native  # noqa: E402
from functional_process.cottax import sand as sand_module  # noqa: E402
from functional_process.importer import read_indat  # noqa: E402
from functional_process.cottax.indat import (  # noqa: E402
    REFERENCE_INPUT_FILE,
    graph_for,
    machine_from_indat,
    switch_values_from_indat,
)
from functional_process.cottax.run_cold_matrix import (  # noqa: E402
    NATIVE,
    SEED_LABEL,
    MdfBuild,
    SandBuild,
    _boundary_seed,
    build_mdf,
    build_sand,
    solve_mdf,
    solve_sand,
)
from functional_process.cottax.run_mda_harness import _resolve  # noqa: E402
from functional_process.cottax.sand_harness import reference_run  # noqa: E402


@dataclass
class Session:
    """One configuration, assembled once, solvable any number of times.

    Build it with `open_session`. `mdf()` and `sand()` each return exactly the dict
    `run_cold_matrix.cold_mdf`/`cold_sand` return, so anything that renders a matrix row
    renders a session's answer unchanged.

    **Each arm is assembled lazily, on its first solve, and kept.** A caller that only
    ever wants MDF should not pay SAND's assembly, and a caller that wants both should
    pay each once; a field that is `None` is an arm nobody has asked for yet.

    `cold` is the starting `DataStructure` the session was opened with. Passing a
    different one to `mdf`/`sand` re-seeds without rebuilding, which is what a scan
    point is.
    """

    name: str
    path: Path
    reference: object
    cold: object
    machine_graph: object = None
    switch_values: object = None
    root_find: bool = False
    seed_mode: str = NATIVE
    boundary: dict = field(default_factory=dict)
    mdf_build: MdfBuild | None = None
    sand_build: SandBuild | None = None

    def mdf(self, cold=None) -> dict:
        """Solve this configuration's MDF arm, assembling it on the first call.

        `cold` overrides the session's own starting `DataStructure` for this solve only
        -- a re-seed, not a rebuild, and measured at 25-42 ms with zero compiles.
        """
        if self.mdf_build is None:
            self.mdf_build = build_mdf(
                self.reference,
                self.machine_graph,
                self.switch_values,
                root_find=self.root_find,
            )
        return solve_mdf(
            self.mdf_build, self.reference, self.cold if cold is None else cold
        )

    def sand(self, cold=None) -> dict:
        """Solve this configuration's SAND arm, assembling it on the first call.

        Raises
        ------
        ValueError
            If the file states a root find (`i_process_run_mode = -2`). SAND and MDF are
            two ways of distributing an *optimisation* and such a file states none --
            `run_cold_matrix._solve_both`'s docstring carries the argument in full, and
            the matrix prints one row rather than two for exactly this reason.
        """
        if self.root_find:
            raise ValueError(
                f"{self.name} states a root find (`i_process_run_mode = -2`), so it "
                f"has no SAND arm: SAND distributes an optimisation over design AND "
                f"coupling and this file poses no optimisation to distribute. Use "
                f"`.mdf()`, which is PROCESS's own square system"
            )
        if self.sand_build is None:
            self.sand_build = build_sand(
                self.reference, self.machine_graph, self.switch_values
            )
        return solve_sand(
            self.sand_build,
            self.reference,
            self.machine_graph,
            self.cold if cold is None else cold,
        )


def open_session(path, mode: str = NATIVE) -> Session:
    """Everything `run_cold_matrix.run_one` does before it first touches a solver.

    `mode` is that module's seeding axis, unchanged and with the same four values --
    `NATIVE` here rather than `PROVIDER`, because a session is a *repeated* solve and
    `--native` is `run_cold_matrix`'s own intended default table
    (`reference_cold_matrix.txt` is generated with it). The three PROCESS-seeded modes
    run PROCESS once, at open time, and are disk-cached like every other caller's.

    Unlike `run_one` this **raises** rather than recording a row: a session is
    interactive, its caller is present, and a refusal that came back as a dataclass field
    would be read as an answer. The matrix's "a failure is a row, not an exit" rule is
    about an unattended seven-configuration pass.

    Raises
    ------
    NotImplementedError, ValueError
        Straight from `machine_from_indat`, where a configuration this port cannot
        assemble refuses with the arm, the switch and the missing model named.
    """
    path = _resolve(str(path))
    name = path.name[: -len(".IN.DAT")] if path.name.endswith(".IN.DAT") else path.stem
    is_reference = path == _resolve(REFERENCE_INPUT_FILE)
    machine = machine_from_indat(str(path))
    machine_graph = None if is_reference else graph_for(machine)
    root_find = read_indat(str(path)).problem.is_evaluation

    if mode == NATIVE:
        reference = native.native_reference(str(path))
        cold = reference.cold
        switch_values = None if is_reference else switch_values_from_indat(str(path))
        boundary: dict = {"mode": mode}
    else:
        reference = reference_run(str(path))
        cold, boundary, _moved = _boundary_seed(reference, path, mode)
        # Read off the same `DataStructure` the values come from, exactly as
        # `run_cold_matrix.run_one` does: a switch the provider disagreed with would
        # change the graph, not just a number.
        switch_values = (
            None
            if is_reference
            else sand_module.switch_values_for(
                cold, reference.icc, reference.i_figure_merit
            )
        )
    return Session(
        name=name,
        path=path,
        reference=reference,
        cold=cold,
        machine_graph=machine_graph,
        switch_values=switch_values,
        root_find=root_find,
        seed_mode=mode,
        boundary=boundary,
    )


def series(live: Session, repeats: int, arm: str = "mdf") -> list[dict]:
    """Solve `live` `repeats` times, timing each -- the instrument for this regime.

    Returns one dict per solve: `seconds`, `compiles`, `rss`, and the answer's own
    `iterations`/`objf`/`max_eq`/`min_ie`/`_x`. A caller checks that `_x` never moves;
    `main` below prints the series and says where it flattened.
    """
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
    """How many XLA modules were compiled between construction and `stop`.

    Counted by wrapping `jax._src.compiler.backend_compile_and_load`, which is the same
    entry point `phase_timing.install` patches for its `compile` column and the same one
    `_audit/optimise_design.md` counts throughout. A jax that moved it degrades to
    `None`, not to a wrong number.
    """

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
    """Resident set size in GiB, or `nan` off Linux.

    Reported per solve because the leak the naive loop has is a *memory* leak first --
    §31.16's 4.2 GB ceiling arrives at about the seventh repeat of a single
    configuration -- and a series that only reports seconds cannot show it.
    """
    try:
        with Path("/proc/self/status").open() as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:  # pragma: no cover -- not Linux
        pass
    return float("nan")


def render(live: Session, measured, arm: str) -> str:
    """The series as a table, plus whether the answer ever moved.

    The verdict line is the point of the whole exercise: every change this regime was
    made fast by is a caching change, so an answer that moves between repeats is a
    defect and not a measurement.
    """
    first = measured[0]
    lines = [
        "",
        (
            f"REPEATED SOLVE -- {live.name} {arm.upper()}, seed "
            f"{SEED_LABEL.get(live.seed_mode, live.seed_mode)}, one process, "
            f"nothing cleared"
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
    """`--input <path>` (repeatable), `--repeat N`, `--arm mdf|sand|both`, `--seed-mode`.

    Prints one series per configuration per arm. With no `--input` it runs the reference
    stellarator and `large_tokamak_nof`, which are the two configurations
    `_audit/optimise_design.md` §32 measures this regime on.
    """
    argv = sys.argv[1:] if argv is None else list(argv)
    inputs = [argv[i + 1] for i, a in enumerate(argv) if a == "--input"] or [
        REFERENCE_INPUT_FILE,
        "tests/regression/input_files/large_tokamak_nof.IN.DAT",
    ]
    repeats = int(argv[argv.index("--repeat") + 1]) if "--repeat" in argv else 6
    arm = argv[argv.index("--arm") + 1] if "--arm" in argv else "mdf"
    mode = argv[argv.index("--seed-mode") + 1] if "--seed-mode" in argv else NATIVE
    arms = ("mdf", "sand") if arm == "both" else (arm,)
    for name in inputs:
        live = open_session(name, mode=mode)
        for one in arms:
            if one == "sand" and live.root_find:
                print(f"\n{live.name}: no SAND arm (root find) -- skipped")
                continue
            print(render(live, series(live, repeats, one), one), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
