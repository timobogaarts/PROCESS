"""Every architecture on every checked-in configuration, cold, against a pin.

One row per `(configuration, arm)`: `configurations.NAMES` (every
`tests/regression/input_files/*.IN.DAT` but IFE, stated as a tree) under each of
`study.ARMS` -- the MDA at the configuration's own design, then MDF, IDF and SAND
solved from it. The pin,
`reference_architectures.txt`, holds what each row answered; a row that moves is a
regression in the port, the recipe, or the cottax underneath, and this is the one test
that says so.

Regenerate the pin after a deliberate change with

    $PY -m pytest functional_process/tests/test_architectures.py --fp-write-pin

and read the diff: every changed digit is a claim. `tier4` -- one cold assembly and
solve per row, ten minutes in all.
"""

from __future__ import annotations

import functools
import math
from pathlib import Path

import jax
import pytest

from functional_process import configurations
from functional_process.cottax import study

PIN = Path(__file__).with_name("reference_architectures.txt")
COLUMNS = ("configuration", "arm", "status", "iterations", "objf", "max_eq", "min_ie")
OBJF_RTOL = 1.0e-6

pytestmark = pytest.mark.tier4

ROWS = [
    (name, arm)
    for name in configurations.NAMES
    for arm in study.arms(configurations.load(name))
]


def _cell(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _row(name, arm, result):
    return (
        name,
        arm,
        result["status"],
        result["iterations"],
        result["objf"],
        result["max_eq"],
        result["min_ie"],
    )


def _format(rows):
    lines = [
        "# Every architecture on every regression input file, solved cold. Generated",
        "# by `pytest functional_process/tests/test_architectures.py --fp-write-pin`;",
        "# do not hand-edit. `status`/`objf` are asserted, the rest is context.",
        "# " + " ".join(COLUMNS),
    ]
    lines += [" ".join(_cell(v) for v in row) for row in rows]
    return "\n".join(lines) + "\n"


def read_pin(path=PIN) -> dict:
    """`{(configuration, arm): {column: value}}` off the pin."""
    pinned = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        cells = line.split()
        row = dict(zip(COLUMNS, cells, strict=True))
        for key in ("objf", "max_eq", "min_ie"):
            row[key] = None if row[key] == "-" else float(row[key])
        row["iterations"] = None if row["iterations"] == "-" else int(row["iterations"])
        pinned[row["configuration"], row["arm"]] = row
    return pinned


_COLD: dict = {}
"""The one configuration's cold state -- rows are grouped by configuration, and holding
every configuration's compiled programs at once has crashed XLA's compiler mid-run."""


@functools.cache
def _solve(name, arm):
    configuration = configurations.load(name)
    if name not in _COLD:
        _COLD.clear()
        jax.clear_caches()
        _COLD[name] = study.cold_state(configuration)
    return study.study(configuration, arm, _COLD[name])


def _same_objf(got, pinned):
    if got is None or pinned is None:
        return got is None and pinned is None
    return math.isclose(got, pinned, rel_tol=OBJF_RTOL, abs_tol=0.0)


@pytest.mark.parametrize(("name", "arm"), ROWS, ids=[f"{n}-{a}" for n, a in ROWS])
def test_architecture(name, arm, request):
    """`status` and `objf` of this row are the pinned ones."""
    result = _solve(name, arm)
    if request.config.getoption("--fp-write-pin"):
        return
    pinned = read_pin().get((name, arm))
    assert pinned is not None, f"{name} {arm}: no pin -- regenerate with --fp-write-pin"
    got = _row(name, arm, result)
    assert result["status"] == pinned["status"], (got, pinned)
    assert _same_objf(result["objf"], pinned["objf"]), (got, pinned)


def test_write_pin(request):
    """Writes the pin when asked; otherwise checks it has a row for every case."""
    if request.config.getoption("--fp-write-pin"):
        PIN.write_text(_format([_row(n, a, _solve(n, a)) for n, a in ROWS]))
        return
    missing = [row for row in ROWS if row not in read_pin()]
    assert not missing, f"no pin for {missing} -- regenerate with --fp-write-pin"
