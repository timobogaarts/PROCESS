"""The checked-in configurations are the regression input files, converted.

`configurations/<name>.py` is `input.indat.configuration_from_indat` written out; this
pins that it still is, for every `IN.DAT` under `tests/regression/input_files/` the
port converts. A row that differs means the file moved, the converter moved, or the
tree was hand-edited -- regenerate with `input.indat.write_configuration` and read the
diff.
"""

from __future__ import annotations

import pytest

from functional_process import configurations
from functional_process.cottax.input.indat import (
    REFERENCE_STELLA_CONF,
    configuration_from_indat,
)

CONTEXT = {"stella_conf": str(REFERENCE_STELLA_CONF)}


@pytest.mark.parametrize("name", configurations.NAMES)
def test_configuration_is_its_input_file_converted(name):
    """`spell` of the converted file equals `spell` of the checked-in tree."""
    converted = configuration_from_indat(f"tests/regression/input_files/{name}.IN.DAT")
    checked_in = configurations.load(name)
    assert configurations.spell(converted, context=CONTEXT) == configurations.spell(
        checked_in, context=CONTEXT
    )


def test_every_checked_in_configuration_is_listed():
    """`NAMES` names every module beside `configurations/__init__.py`."""
    from pathlib import Path  # noqa: PLC0415

    modules = {
        p.stem
        for p in Path(configurations.__file__).parent.glob("*.py")
        if p.stem not in {"__init__", "defaults"}
    }
    assert modules == set(configurations.NAMES)
