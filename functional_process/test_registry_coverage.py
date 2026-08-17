"""Meta-tests: the harness checked against the unit registry.

With ~20 units landing one at a time, the failure mode that matters is not a wrong
assertion, it is a missing one — a unit marked done whose test was never written, or a
record whose status drifted from the registry's. Neither shows up as a red test unless
something goes looking, which is what this module does.
"""

from pathlib import Path

import pytest

from functional_process._harness.registry import parse_frontmatter, parse_unit_registry

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = Path(__file__).resolve().parent
REGISTRY = PACKAGE_ROOT / "_audit" / "unit_registry.md"

# One record per file, per `_audit/schema.md`. The two aggregate records hold many
# entries in one file and so carry per-section frontmatter instead; they are checked for
# existence only.
_AGGREGATE_RECORDS = frozenset({
    "functional_process/core/solver/constraints.md",
    "functional_process/core/solver/switches.md",
})

_VALID_STATUSES = frozenset({"pending", "in-progress", "draft", "reviewed", "final"})


@pytest.fixture(scope="module")
def registry_rows():
    """Every unit-registry row that names an audit record."""
    rows = parse_unit_registry(REGISTRY)
    assert rows, f"parsed no rows from {REGISTRY} — the parser or the tables moved"
    return rows


def test_registry_statuses_are_recognised(registry_rows):
    """Every status cell reduces to a known status word."""
    unknown = [(r.record, r.raw_status) for r in registry_rows if r.status == "unknown"]
    assert not unknown, (
        f"unrecognised status words (expected one of {sorted(_VALID_STATUSES)}): "
        f"{unknown}"
    )


def test_started_units_have_a_record_file(registry_rows):
    """Any unit past `pending` has the record file its row points at."""
    missing = [
        r.record
        for r in registry_rows
        if r.is_started and not (REPO_ROOT / r.record).is_file()
    ]
    assert not missing, (
        f"registry claims audit work is under way but the record file is absent: "
        f"{sorted(set(missing))}"
    )


def test_record_frontmatter_agrees_with_registry(registry_rows):
    """A record's own `status:` matches the registry row pointing at it.

    Two hand-maintained sources of truth for the same fact will drift; this is the
    cheapest possible guard against acting on the stale one.
    """
    disagreements = []
    for row in registry_rows:
        path = REPO_ROOT / row.record
        if not path.is_file() or row.record in _AGGREGATE_RECORDS:
            continue
        recorded = parse_frontmatter(path).get("status")
        if recorded != row.status:
            disagreements.append(
                f"{row.record}: record says {recorded!r}, registry says "
                f"{row.status!r} (from {row.raw_status!r})"
            )
    assert not disagreements, "\n".join(disagreements)


def test_final_records_have_a_test_module(registry_rows):
    """A unit whose audit is `final` has a harness case sitting next to it.

    Passes vacuously today — nothing is `final` yet. It exists so that the first unit
    to reach `final` without a test fails the suite instead of passing quietly.
    """
    gaps = []
    for row in registry_rows:
        if row.status != "final" or row.record in _AGGREGATE_RECORDS:
            continue
        record = REPO_ROOT / row.record
        expected = record.with_name(f"test_{record.stem}.py")
        if not expected.is_file():
            gaps.append(f"{row.record} is final but {expected.name} does not exist")
    assert not gaps, "\n".join(gaps)


def test_every_record_file_is_in_the_registry():
    """No audit record exists that the registry does not know about."""
    known = {r.record for r in parse_unit_registry(REGISTRY)}
    orphans = [
        str(p.relative_to(REPO_ROOT))
        for p in PACKAGE_ROOT.rglob("*.md")
        if p.parent.name != "_audit"
        and p.name != "README.md"
        and str(p.relative_to(REPO_ROOT)) not in known
    ]
    assert not orphans, f"audit records not listed in unit_registry.md: {orphans}"
