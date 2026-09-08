"""Parse `_audit/unit_registry.md` so the harness can check itself against it."""

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
    """Extract every row of `unit_registry.md` that names an audit record."""
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
    """Read the YAML-ish frontmatter block from an audit record."""
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
