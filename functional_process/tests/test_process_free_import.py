"""The model layer imports with no `process` package -- §23's target, checked.

Run in a **subprocess** with a `sys.meta_path` hook that raises on `import process`,
because there is no other honest way to ask the question from inside a test session that
has already imported PROCESS twenty times over. Blocking beats uninstalling: nothing is
removed, the env stays the co-importable one the harness needs (`CLAUDE.md` § the
environment), and the check is a normal test rather than a manual ritual.

**Two things are asserted and they are not the same claim.**

1. **Import.** Every module under `functional_process/cottax/`,
   `functional_process/cottax/core/` and `functional_process/vocabulary/` imports, and so do
   `paths`, `total_process`, `indat` (which builds `GRAPH` -- the whole reference
   stellarator machine -- at import time), `sand`, `mda`, `boundary` and
   `machine_survey`.
2. **Assembly (§23.6).** `machine_from_indat` + `graph_for` build a **tokamak** and a
   **spherical tokamak** from their real regression inputs with `process` blocked.
   Importing `indat` only ever proved the *stellarator* path, because that is the machine
   `GRAPH` is: the tokamak arm calls `_quench_helium_table`, which was the port's last
   runtime `process` import (`process.core.coolprop_interface`). That wrapper is now
   vendored as `functional_process/_vendor/fluid_properties.py` and equality-tested in
   `test_fluid_properties.py`, so the tokamak arm runs PROCESS-free and this file says so
   by running it rather than by asserting an import list.

**What is knowingly not asserted, and why:**

- `mdf` imports `mda_harness`/`sand_harness`, and the harnesses are *supposed* to import
  PROCESS. Excluded by design, not by defeat.
- **Solving.** These tests assemble a graph and count its nodes; they do not run an MDA,
  a SAND ladder or a cold start. Everything that compares against PROCESS's answer needs
  PROCESS by construction.
- `render_xdsm` fails on a `cottax.visualization` name, which has nothing to do with
  `process`.
"""

import subprocess  # noqa: S404
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

_PROBE = '''
import importlib, pathlib, sys

class _Block:
    """Raise on any `process` import, from anywhere, at any depth."""
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "process" or fullname.startswith("process."):
            raise ImportError("BLOCKED: " + fullname)
        return None

sys.meta_path.insert(0, _Block())

import jax
jax.config.update("jax_enable_x64", True)

root = pathlib.Path("functional_process")
modules = []
for p in sorted(root.rglob("*.py")):
    if "_audit" in p.parts:
        continue
    parts = list(p.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    name = ".".join(parts)
    # **What must import with no PROCESS: the physics, the declarations, the solver
    # core and the vocabulary.** Not the harnesses -- `_harness/`, `*_harness.py`,
    # `run_*.py`, `cold_start`, `provider`, `native` exist to *compare against*
    # PROCESS and are supposed to import it (this module's own docstring says so).
    #
    # Spelled as "a subpackage of cottax, except the harness and the tests" rather than
    # a prefix on `functional_process.cottax`, because after the 2026-09-05 split that
    # prefix covers the harnesses too. A blanket rename turned this list into "every
    # cottax module" and the test started failing on modules it was never about.
    if name.startswith(("functional_process.models", "functional_process.vocabulary")):
        modules.append(name)
    elif name.startswith("functional_process.cottax."):
        rest = name[len("functional_process.cottax."):]
        head = rest.split(".")[0]
        if "." in rest and head not in {"_harness", "tests"}:
            modules.append(name)

modules += ["functional_process.cottax.paths", "functional_process.cottax.total_process",
            "functional_process.cottax.indat", "functional_process.cottax.sand",
            "functional_process.cottax.mda", "functional_process.cottax.boundary",
            "functional_process.cottax.machine_survey"]

failures = []
for m in modules:
    try:
        importlib.import_module(m)
    except BaseException as exc:
        failures.append(m + " -- " + type(exc).__name__ + ": " + str(exc))

assert "process" not in sys.modules, "`process` was imported despite the block"
print("COOLPROP", "CoolProp" in sys.modules)
print("MODULES", len(modules))
print("FAILURES", len(failures))
for f in failures:
    print("FAIL", f)
'''


_ASSEMBLY_CASES = ("large_tokamak_nof", "spherical_tokamak_eval")
"""The two tokamak inputs assembled under the block: a conventional tokamak and a
spherical one. Both reach `indat._quench_helium_table`, which is the only CoolProp call
in the port and was the only runtime `process` import left in it."""

_ASSEMBLY_PROBE = (
    _PROBE.split("root = pathlib.Path", 1)[0]
    + """
import functional_process._vendor.fluid_properties  # the vendored wrapper itself

from functional_process.cottax.indat import graph_for, machine_from_indat

for name in """
    + repr(list(_ASSEMBLY_CASES))
    + """:
    path = "tests/regression/input_files/" + name + ".IN.DAT"
    try:
        graph = graph_for(machine_from_indat(path))
    except BaseException as exc:
        print("FAIL", name, type(exc).__name__ + ": " + str(exc)[:400])
    else:
        print("ASSEMBLED", name, len(graph.nodes))

assert "process" not in sys.modules, "`process` was imported despite the block"
print("DONE")
"""
)
"""Reuses the import block verbatim from `_PROBE` -- same `sys.meta_path` hook, same x64
setup -- so the two probes cannot drift into blocking different things."""


@pytest.fixture(scope="module")
def probe():
    """The blocked-import subprocess, run once."""
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", _PROBE],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture(scope="module")
def assembly_probe():
    """The blocked-import subprocess that builds two tokamak graphs, run once.

    Slower than `probe` by roughly the cost of `import CoolProp` (~3 s) plus two graph
    assemblies -- which is exactly why `quench.py` keeps its lazy import: nothing that
    does not assemble a tokamak pays it.
    """
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", _ASSEMBLY_PROBE],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_probe_ran(probe):
    """The subprocess itself completed -- otherwise the assertions below are vacuous."""
    assert probe.returncode == 0, probe.stderr[-4000:]


def test_no_module_needs_process_to_import(probe):
    """Nothing in the model layer, `indat`, `sand`, `mda` or `paths` imports `process`.

    Including `indat`, which assembles the reference stellarator machine at import: the
    whole graph is built here with PROCESS unavailable.
    """
    lines = probe.stdout.splitlines()
    failures = [line for line in lines if line.startswith("FAIL ")]
    assert not failures, "\n".join(failures)


def test_probe_covered_the_whole_model_layer(probe):
    """The sweep found the modules it claims to have found.

    A probe that silently enumerated nothing would pass the test above for the wrong
    reason. 100+ is a floor, not a pin -- it is there to catch an empty walk, not to
    freeze the file count.
    """
    count = next(
        int(line.split()[1])
        for line in probe.stdout.splitlines()
        if line.startswith("MODULES")
    )
    assert count > 100, count


def test_importing_the_model_layer_does_not_load_coolprop(probe):
    """The lazy import in `quench.py` is still lazy.

    `import CoolProp` costs ~3 s (measured 2026-08-31) and only a tokamak *assembly*
    wants a helium table. Vendoring the wrapper made it possible to import it eagerly;
    this asserts nobody did.
    """
    line = next(
        line for line in probe.stdout.splitlines() if line.startswith("COOLPROP")
    )
    assert line == "COOLPROP False", line


def test_assembly_probe_ran(assembly_probe):
    """The tokamak-assembly subprocess completed."""
    assert assembly_probe.returncode == 0, assembly_probe.stderr[-4000:]


@pytest.mark.parametrize("name", _ASSEMBLY_CASES)
def test_tokamak_assembles_with_process_blocked(assembly_probe, name):
    """§23.6: a tokamak and a spherical tokamak build a full graph with no `process`.

    This is the claim §23.5 could not make. It went through `_quench_helium_table` and
    therefore through CoolProp, live, in a process where `import process` raises.
    """
    lines = assembly_probe.stdout.splitlines()
    assert not [line for line in lines if line.startswith("FAIL " + name)], "\n".join(
        lines
    )
    assembled = next(
        (line for line in lines if line.startswith("ASSEMBLED " + name)), None
    )
    assert assembled is not None, "\n".join(lines)
    assert int(assembled.split()[2]) > 200, assembled
