"""One adapter for PROCESS's mutable-object API, in place of 484 hand-written ones."""

import dataclasses
import functools


@functools.cache
def _index() -> dict[str, str]:
    """`field name -> area name`, for every unambiguous field of `DataStructure`."""
    from process.core.model import DataStructure  # noqa: PLC0415

    data = DataStructure()
    seen: dict[str, str | None] = {}
    for field in dataclasses.fields(data):
        area = getattr(data, field.name)
        if not dataclasses.is_dataclass(area):
            continue
        for sub in dataclasses.fields(area):
            seen[sub.name] = None if sub.name in seen else field.name
    return {k: v for k, v in seen.items() if v is not None}


def _resolve(name: str) -> tuple[str, str]:
    """`name` (or `"area.field"`) as `(area, field)`."""
    if "." in name:
        area, field = name.split(".", 1)
        return area, field
    area = _index().get(name)
    if area is None:
        raise KeyError(
            f"{name!r} is not an unambiguous DataStructure field — write it as "
            f"'area.{name}' (it exists in more than one area, or in none)"
        )
    return area, name


def _poke(data, kwargs, areas):
    """Set each of `kwargs` onto the area that owns it."""
    for name, value in kwargs.items():
        area, field = _resolve(areas.get(name, name))
        setattr(getattr(data, area), field, value)


def data_reference(call, *, areas=None):
    """A `reference` callable that pokes a fresh `DataStructure` and hands it to `call`.
    """
    areas = dict(areas or {})

    def reference(**kwargs):
        from process.core.model import DataStructure  # noqa: PLC0415

        data = DataStructure()
        _poke(data, kwargs, areas)
        return call(data)

    return reference


def process_reference(factory, method: str, outputs, *, call_args=(), areas=None):
    """A `reference` callable: keyword arguments in, `outputs` out."""
    outputs = (outputs,) if isinstance(outputs, str) else tuple(outputs)
    areas = dict(areas or {})

    def reference(**kwargs):
        model = factory()
        _poke(model.data, kwargs, areas)
        getattr(model, method)(*call_args)
        read = []
        for name in outputs:
            area, field = _resolve(name)
            read.append(getattr(getattr(model.data, area), field))
        return read[0] if len(read) == 1 else tuple(read)

    reference.__name__ = f"reference_{method}"
    reference.__doc__ = f"PROCESS's `{method}`, called with keyword arguments."
    return reference
