"""Parse `_audit/unit_registry.md` so the harness can check itself against it.

The registry is the master list of what is in scope and how far along it is. It is
maintained by hand, in prose, which means it drifts — a unit gets ported and the row
still says `pending`, or a record's own frontmatter says `draft` while the registry says
`reviewed`. `test_registry_coverage.py` turns that drift into a test failure.
"""

import re
from dataclasses import dataclass

_STATUS_WORDS = ("pending", "in-progress", "draft", "reviewed", "final")
_RECORD_RE = re.compile(r"functional_process/[\w/]+\.md")


@dataclass(frozen=True)
class RegistryRow:
    """One row of a unit-registry table that names an audit record."""

    record: str
    """Path relative to the repository root."""
    status: str
    """Normalised status word."""
    raw_status: str
    """The cell as written, which usually carries the reason and a confidence note."""

    @property
    def is_started(self):
        """Whether any audit work is claimed for this row."""
        return self.status != "pending"


def _normalise_status(cell):
    """Reduce a free-text status cell to one of the known status words."""
    plain = cell.replace("*", "").replace("`", "").strip().lower()
    for word in _STATUS_WORDS:
        if plain.startswith(word):
            return word
    return "unknown"


def parse_unit_registry(path):
    """Extract every row of `unit_registry.md` that names an audit record.

    Rows using a ditto mark (the switch table repeats one record across ten rows) are
    skipped rather than guessed at — they carry no record path of their own.

    Parameters
    ----------
    path :
        Path to `_audit/unit_registry.md`.

    Returns
    -------
    :
        List of `RegistryRow`, one per row naming a record, in file order.
    """
    rows = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or set(stripped) <= set("|- "):
            continue

        cells = [c.strip() for c in stripped.strip("|").split("|")]
        record_cells = [c for c in cells if _RECORD_RE.search(c)]
        if not record_cells:
            continue

        # A row may name two candidate records (the i_cost_model split); the status
        # cell is the last one either way.
        rows.extend(
            RegistryRow(
                record=match.group(0),
                status=_normalise_status(cells[-1]),
                raw_status=cells[-1],
            )
            for match in _RECORD_RE.finditer(" ".join(record_cells))
        )
    return rows


def parse_frontmatter(path):
    """Read the YAML-ish frontmatter block from an audit record.

    Deliberately a five-line parser rather than a YAML dependency: the schema is
    `key: value` only (`_audit/schema.md`), and anything richer appearing in a record
    is itself worth noticing.

    Parameters
    ----------
    path :
        Path to a record file.

    Returns
    -------
    :
        Mapping of key to value; empty if the file has no frontmatter block.
    """
    lines = path.read_text().splitlines()
    try:
        start = lines.index("---")
    except ValueError:
        return {}

    out = {}
    for line in lines[start + 1 :]:
        if line.strip() == "---":
            break
        if ":" in line:
            key, _, value = line.partition(":")
            out[key.strip()] = value.strip()
    return out
