"""`vocabulary.fields` is PROCESS's own `DataStructure`, field for field."""

import dataclasses

import pytest

from functional_process.vocabulary.fields import FIELDS


def test_fields_are_the_live_data_structure():
    model = pytest.importorskip("process.core.model")
    data = model.DataStructure()
    live = {}
    for area in dataclasses.fields(model.DataStructure):
        obj = getattr(data, area.name)
        live[area.name] = (
            tuple(f.name for f in dataclasses.fields(obj))
            if dataclasses.is_dataclass(obj)
            else tuple(vars(obj))
        )
    assert FIELDS == live
