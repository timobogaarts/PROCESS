"""What is inside a compiled program, and how big is the thing it produces.

The measurement behind `optimise_design.md` §50. Hooks the same entry point
`_audit/rss_per_program.py` does (`compiler.backend_compile_and_load`) but records the
module's *contents* -- an op histogram -- alongside the executable's own numbers.

    $PY functional_process/_audit/hlo_anatomy.py st_regression sand

Three things it is here to establish, each of which corrected a belief:

- **What the characters are.** The largest `st_regression` SAND module is 2.28 M
  characters = 28 106 ops at ~81 characters each, of which **41.6 % is pure shape
  plumbing** (`broadcast_in_dim` alone 27.9 %) -- a scalar-valued graph wrapped into
  tensors, not an enormous model.
- **How big the executable is.** `size_of_generated_code_in_bytes()` returns **0** on the
  CPU backend and is not the field to read; `serialize()` gives 11.0 MB for that module
  and 21.1 MB for all 34 programs. Optimised HLO *text* is 422 % of the input, so it is a
  proxy for nothing.
- **`malloc_trim(0)` before each compile, and this is load-bearing.** Without it an RSS
  delta measures what the allocator did rather than what the program cost: a compile that
  fits in arena an earlier one freed reads as nearly free. That artefact produced, and
  then unproduced, §45's "two classes of module".

Not a test -- it compiles a whole configuration. Lives here for the same reason
`declaration_census.py` and `rss_per_program.py` do: it measures the port rather than
being part of it.
"""

import ctypes, re, sys, time, collections
import jax

jax.config.update("jax_enable_x64", True)
from jax._src import compiler

PROGRAMS = []
_original = compiler.backend_compile_and_load
_libc = ctypes.CDLL("libc.so.6")


def rss_kb():
    with open("/proc/self/status") as h:
        for line in h:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return -1


OP = re.compile(r"=\s*\"?(stablehlo|chlo|mhlo|func|arith|tensor)\.([a-z_0-9A-Z]+)")


def _counting(*args, **kwargs):
    module = next((a for a in args if hasattr(a, "operation")), None)
    text = ""
    if module is not None:
        try:
            text = module.operation.get_asm()
        except Exception:
            text = ""
    _libc.malloc_trim(0)
    before = rss_kb()
    began = time.perf_counter()
    out = _original(*args, **kwargs)
    raw = rss_kb() - before
    secs = time.perf_counter() - began
    exe = out[0] if isinstance(out, tuple) else out
    rec = {
        "i": len(PROGRAMS),
        "chars": len(text),
        "raw": raw,
        "secs": secs,
        "ops": collections.Counter(f"{a}.{b}" for a, b in OP.findall(text)),
    }
    try:
        rec["code"] = exe.size_of_generated_code_in_bytes()
    except Exception as exc:
        rec["code"] = f"n/a({type(exc).__name__})"
    try:
        rec["ser"] = len(exe.serialize())
    except Exception as exc:
        rec["ser"] = f"n/a({type(exc).__name__})"
    try:
        rec["opt"] = sum(len(m.to_string()) for m in exe.hlo_modules())
    except Exception as exc:
        rec["opt"] = f"n/a({type(exc).__name__})"
    _libc.malloc_trim(0)
    PROGRAMS.append(rec)
    return out


compiler.backend_compile_and_load = _counting

name = sys.argv[1] if len(sys.argv) > 1 else "st_regression"
arm = sys.argv[2] if len(sys.argv) > 2 else "sand"
from functional_process.cottax import session

getattr(session.open_session(f"tests/regression/input_files/{name}.IN.DAT"), arm)()

print(
    f"\n{name} {arm.upper()}: {len(PROGRAMS)} programs; those over 100k chars, "
    f"in compile order\n"
)
hdr = (
    f"{'ord':>4}{'HLO chars':>12}{'ops':>9}{'opt HLO':>12}"
    f"{'code MB':>10}{'serial MB':>11}{'RSS MB':>9}{'B/char':>8}{'secs':>7}"
)
print(hdr)
print("-" * len(hdr))
tot_code = tot_ser = 0
for p in PROGRAMS:
    if p["chars"] < 100_000:
        continue
    code = p["code"]
    ser = p["ser"]
    opt = p["opt"]
    if isinstance(code, int):
        tot_code += code
    if isinstance(ser, int):
        tot_ser += ser
    print(
        f"{p['i']:>4}{p['chars']:>12,}{sum(p['ops'].values()):>9,}"
        f"{(f'{opt:,}' if isinstance(opt, int) else opt):>12}"
        f"{(f'{code / 1048576:.1f}' if isinstance(code, int) else code):>10}"
        f"{(f'{ser / 1048576:.1f}' if isinstance(ser, int) else ser):>11}"
        f"{p['raw'] / 1024:>9.1f}"
        f"{p['raw'] * 1024 / max(p['chars'], 1):>8.1f}"
        f"{p['secs']:>7.2f}"
    )
print(
    f"\nall {len(PROGRAMS)} programs: generated code {tot_code / 1048576:.1f} MB, "
    f"serialized {tot_ser / 1048576:.1f} MB"
)
big = max(PROGRAMS, key=lambda p: p["chars"])
print(
    f"\nlargest module op mix ({big['chars']:,} chars, "
    f"{sum(big['ops'].values()):,} ops):"
)
n = sum(big["ops"].values())
for k, v in big["ops"].most_common(12):
    print(f"   {k:<32}{v:>8,}  {100 * v / n:>5.1f}%")
mover = sum(
    v
    for k, v in big["ops"].items()
    if k.split(".")[1]
    in {
        "broadcast_in_dim",
        "reshape",
        "slice",
        "transpose",
        "concatenate",
        "convert",
        "bitcast_convert",
        "dynamic_slice",
        "dynamic_update_slice",
        "pad",
    }
)
print(f"   {'-> pure data movement/shape':<32}{mover:>8,}  {100 * mover / n:>5.1f}%")
