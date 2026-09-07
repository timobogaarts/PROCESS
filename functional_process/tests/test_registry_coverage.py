"""Meta-tests: the harness checked against the unit registry.

With units landing one at a time, the failure mode that matters is not a wrong assertion,
it is a missing one — a unit marked done whose test was never written. That does not show
up as a red test unless something goes looking, which is what this module does.

**Scope shrank on 2026-09-07** when the per-unit audit records were deleted
(`_audit/README.md`). Three of the five checks here existed to keep two
hand-maintained sources of truth — a registry row and a record's own frontmatter
— from drifting apart. With one source left there is nothing to drift, so they
are gone rather than weakened. The registry's `record` column stays as the
unit's IDENTITY: it names the unit's path and is what `_case_for` re-roots, and
it is deliberately no longer required to be a file that exists.
"""

from pathlib import Path

import pytest

import functional_process
from functional_process.cottax._harness.registry import parse_unit_registry

# Anchored on the package, not on this file: the cases live under
# `functional_process/tests/`, mirroring the package layout, so `__file__` points near
# neither the registry nor them and every path here is derived from the package root.
PACKAGE_ROOT = Path(functional_process.__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
REGISTRY = PACKAGE_ROOT / "_audit" / "unit_registry.md"
UNITS_PREFIX = "functional_process/_audit/units/"
CASE_ROOT = REPO_ROOT / "functional_process" / "tests"

# Two rows hold many units in one identity and so mirror onto no single case module.
_AGGREGATE_RECORDS = frozenset({
    UNITS_PREFIX + "core/solver/constraints.md",
    UNITS_PREFIX + "core/solver/switches.md",
})

_VALID_STATUSES = frozenset({"pending", "in-progress", "draft", "reviewed", "final"})


@pytest.fixture(scope="module")
def registry_rows():
    """Every unit-registry row that names a unit."""
    rows = parse_unit_registry(REGISTRY)
    assert rows, f"parsed no rows from {REGISTRY} — the parser or the tables moved"
    return rows


def test_registry_statuses_are_recognised(registry_rows):
    """Every status cell reduces to a known status word."""
    unknown = [(r.record, r.raw_status) for r in registry_rows if r.status == "unknown"]
    assert not unknown, (
        f"unit registry rows whose status cell does not reduce to one of "
        f"{sorted(_VALID_STATUSES)}: {unknown}"
    )


def _case_for(record: str) -> Path:
    """The harness case that mirrors a unit, by path rather than by adjacency.

    The registry's `record` column is a path under `_audit/units/` mirroring the package
    tree, so the case is that path re-rooted onto `functional_process/tests/` with
    `test_` prefixed to the stem. The record file itself no longer exists; only its
    shape is used, which is why this takes a string.
    """
    rel = Path(record[len(UNITS_PREFIX):])
    return CASE_ROOT / rel.parent / f"test_{rel.stem}.py"


def test_final_units_have_a_test_module(registry_rows):
    """A unit whose audit is `final` has a harness case at its mirrored path.

    It exists so that the first unit to reach `final` without a test fails the suite
    instead of passing quietly.
    """
    gaps = [
        f"{row.record} is final but "
        f"{_case_for(row.record).relative_to(REPO_ROOT)} does not exist"
        for row in registry_rows
        if row.status == "final"
        and row.record not in _AGGREGATE_RECORDS
        and row.record.startswith(UNITS_PREFIX)
        and not _case_for(row.record).is_file()
    ]
    assert not gaps, "\n".join(gaps)
