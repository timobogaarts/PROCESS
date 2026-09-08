"""The matrix solved **warm**: assembled once, compiled once, then measured."""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

from functional_process.cottax import run_cold_matrix, session  # noqa: E402
from functional_process.cottax.core.solver import (  # noqa: E402
    drivers as _drivers,
)
from functional_process.cottax.core.solver import (  # noqa: E402
    host_cache as _host_cache,
)

CALLS: list[float] = []
_ORIGINAL_BIND = _host_cache.bind


def _timing_bind(conditions, unravel):
    """`host_cache.bind`, with every returned program timed into `CALLS`."""
    programs = _ORIGINAL_BIND(conditions, unravel)

    def timed(program):
        def call(flat_x):
            began = time.perf_counter()
            answer = program(flat_x)
            jax.block_until_ready(answer)
            CALLS.append(time.perf_counter() - began)
            return answer

        return call

    return tuple(timed(p) for p in programs)


_host_cache.bind = _timing_bind
_drivers.bind = _timing_bind


def measure(path, arm: str, optimiser, repeats: int) -> dict:
    """Solve `arm` once to pay the compiler, then `repeats` times, and report the last.
    """
    # **Drop the previous measurement's executables first.** jax caches every program
    # it compiles for the life of the process, and this loop compiles a whole
    # configuration's programs 4 times over (two arms x two drivers). Without this the
    # pass dies partway with `LLVM ERROR: Unable to allocate section memory` -- measured
    # on `large_tokamak_nof` SAND, the fourth configuration in. `run_cold_matrix.main`
    # clears per row for the same reason and records the same failure.
    #
    # Nothing is lost: each measurement re-solves cold before it starts timing, so the
    # cache it would have hit is one it is about to refill anyway.
    jax.clear_caches()
    run_cold_matrix._return_freed_memory_to_the_os()
    live = session.open_session(str(path), optimiser=optimiser)
    solve = getattr(live, arm)
    solve()  # cold: pays trace, lower and compile, and is thrown away
    row: dict = {}
    for _ in range(repeats):
        CALLS.clear()
        began = time.perf_counter()
        row = solve()
        row["_wall"] = time.perf_counter() - began
        row["_xla"] = sum(CALLS)
        row["_calls"] = len(CALLS)
        row["_median_call"] = statistics.median(CALLS) if CALLS else 0.0
    return row


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--out", default=None)
    parser.add_argument("--input", action="append", default=[])
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    paths = [run_cold_matrix._resolve(p) for p in args.input] or [
        run_cold_matrix._resolve(p) for p in run_cold_matrix.CONFIGURATIONS
    ]
    header = (
        f"{'configuration':<23}{'arm':<5}{'driver':<7}"
        f"{'it':>5}{'status':>11}{'objf':>16}"
        f"{'wall':>9}{'XLA':>9}{'host':>9}{'calls':>7}{'ms/call':>9}"
    )
    lines = [
        "WARM MATRIX -- assembled once, compiled once, then measured. Seconds unless "
        "stated.",
        "`XLA` is summed over every block-program call; `host` is the rest of the solve, "
        "which is",
        "the optimiser's own cost. See this module's docstring for why the cold matrix "
        "cannot answer this.",
        "",
        header,
        "-" * len(header),
    ]
    for path in paths:
        name = path.name.replace(".IN.DAT", "")
        for arm in ("mdf", "sand"):
            for label, optimiser in (("VMCON", None), ("SLSQP", _drivers.SlsqpDriver)):
                try:
                    row = measure(path, arm, optimiser, args.repeats)
                except ValueError:
                    continue  # a root-find file states no SAND arm
                except Exception as failure:  # noqa: BLE001 -- a row, not an exit
                    lines.append(
                        f"{name:<23}{arm.upper():<5}{label:<7}"
                        f"{'':>5}{'FAILED':>11}  "
                        f"{type(failure).__name__}: {failure}"
                    )
                    continue
                objf = row.get("objf")
                lines.append(
                    f"{name:<23}{arm.upper():<5}{label:<7}"
                    f"{row.get('iterations', 0):>5}{row.get('status', '?'):>11}"
                    f"{'-' if objf is None else f'{objf:.9g}':>16}"
                    f"{row['_wall']:>9.3f}{row['_xla']:>9.3f}"
                    f"{row['_wall'] - row['_xla']:>9.3f}"
                    f"{row['_calls']:>7}{row['_median_call'] * 1000:>9.3f}"
                )
                print(lines[-1], flush=True)
    text = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
