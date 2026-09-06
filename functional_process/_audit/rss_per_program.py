"""Resident cost per compiled program, and per warm-matrix row.

The measurement behind `optimise_design.md` §45 and the answer to `next_steps.md`'s
open item 3: a whole-matrix pass died with `LLVM ERROR: Unable to allocate section
memory` on a **63-byte** request, so the question was never "which program is too big"
but "what stays resident, and does anything give it back".

Patches the one jax entry point `cottax/phase_timing.py` already names for compilation
(`compiler.backend_compile_and_load`) and records, per compiled program, the `VmRSS`
delta across the compile call and the character count of the StableHLO module handed to
it. Then runs `run_warm_matrix.measure` per row with `VmRSS` sampled **before**,
**after**, and **after** `jax.clear_caches()` + `gc.collect()` + `malloc_trim(0)` --
the last of those being the only one that returns anything
(`run_cold_matrix._return_freed_memory_to_the_os`).

Run from the repository root, as a script rather than with `-m`: `_audit` is not a
package (neither is `declaration_census.py`'s home), and the repository root is on
`sys.path` because `process`/`functional_process` are installed editable.

    $PY functional_process/_audit/rss_per_program.py --arm mdf
    $PY functional_process/_audit/rss_per_program.py \
        --input tests/regression/input_files/st_regression.IN.DAT

The `trimmed` column is the one to read: if it climbs without bound the pass leaks, and
if it plateaus the runners' workaround is sufficient. Measured 2026-09-06 it plateaus.

Not a test. It compiles every program in a configuration and takes minutes a row; it
lives here rather than under `cottax/` because it measures the port rather than being
part of it, next to `declaration_census.py` for the same reason.
"""

import argparse
import contextlib
import ctypes
import gc
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

from jax._src import compiler  # noqa: E402, PLC2701


def rss_kb() -> int:
    """Current resident set size in KiB."""
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return -1


def hwm_kb() -> int:
    """Peak resident set size in KiB, for the life of the process."""
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    return -1


PROGRAMS = []  # (module_chars, rss_delta_kb, seconds)

_original = compiler.backend_compile_and_load


def _counting(*args, **kwargs):
    module = None
    for candidate in args:
        if hasattr(candidate, "operation") or hasattr(candidate, "context"):
            module = candidate
            break
    chars = -1
    if module is not None:
        try:
            chars = len(module.operation.get_asm())
        except Exception:  # noqa: BLE001
            chars = -1
    before = rss_kb()
    began = time.perf_counter()
    out = _original(*args, **kwargs)
    PROGRAMS.append((chars, rss_kb() - before, time.perf_counter() - began))
    return out


compiler.backend_compile_and_load = _counting


def trim():
    """`malloc_trim(0)` -- see `run_cold_matrix._return_freed_memory_to_the_os`."""
    with contextlib.suppress(OSError, AttributeError):
        ctypes.CDLL("libc.so.6").malloc_trim(0)


def main() -> int:
    """One row per configuration and arm; read the `trimmed` column."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", default=[])
    parser.add_argument("--arm", action="append", default=[])
    args = parser.parse_args()

    from functional_process.cottax import (  # noqa: PLC0415
        run_cold_matrix,
        run_warm_matrix,
    )

    paths = [run_cold_matrix._resolve(p) for p in args.input] or [
        run_cold_matrix._resolve(p) for p in run_cold_matrix.CONFIGURATIONS
    ]
    arms = args.arm or ["mdf", "sand"]

    print(
        f"{'configuration':<23}{'arm':<5}{'progs':>7}{'MB/row':>9}"
        f"{'before':>9}{'after':>9}{'trimmed':>9}{'HWM':>9}",
        flush=True,
    )
    for path in paths:
        name = path.name.replace(".IN.DAT", "")
        for arm in arms:
            PROGRAMS.clear()
            before = rss_kb()
            try:
                run_warm_matrix.measure(path, arm, None, 1)
            except ValueError:
                continue
            except Exception as failure:  # noqa: BLE001
                print(
                    f"{name:<23}{arm.upper():<5}  FAILED "
                    f"{type(failure).__name__}: {failure}",
                    flush=True,
                )
                continue
            after = rss_kb()
            jax.clear_caches()
            gc.collect()
            trim()
            trimmed = rss_kb()
            grew = sum(delta for _, delta, _ in PROGRAMS)
            print(
                f"{name:<23}{arm.upper():<5}{len(PROGRAMS):>7}{grew / 1024:>9.1f}"
                f"{before / 1024:>9.1f}{after / 1024:>9.1f}"
                f"{trimmed / 1024:>9.1f}{hwm_kb() / 1024:>9.1f}",
                flush=True,
            )
            biggest = sorted(PROGRAMS, key=lambda p: -p[1])[:5]
            for chars, delta, secs in biggest:
                ratio = (delta * 1024 / chars) if chars > 0 else float("nan")
                print(
                    f"    prog {chars:>9} chars  {delta / 1024:>7.1f} MB  "
                    f"{secs:>6.2f} s  {ratio:>6.1f} B/char",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
