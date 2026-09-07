"""One adapter for PROCESS's mutable-object API, in place of 484 hand-written ones.

A tier contract wants a `reference` it can call with the port's own keyword arguments.
PROCESS's models do not work that way: they read and write fields of a `DataStructure`,
so every unit used to carry a `_reference_*` function that made a model, poked each
argument onto `data.<area>.<field>`, called the method, and returned a tuple read back
off `data`. Those were 10,556 lines across the test tree — one line per argument in and
one per result out, times 484 units — and every one of them said the same thing.

**A field name is enough to find its area.** `DataStructure` has 2,284 distinct field
names across 36 areas and exactly three of them are ambiguous (`first_call`,
`nu_star`, `copper_rrr`), so the poke target can be *derived* rather than written down.
Ambiguous names must be written `area.field`; an unqualified one raises at call time
rather than picking an area, because silently poking the wrong `first_call` is the kind
of wrong answer this harness exists to catch.

Errors here are loud by construction: the reference is the oracle, so a mis-resolved
field makes PROCESS compute something the port disagrees with, and the unit's own value
test fails. There is no way for this to be quietly wrong.
"""

import dataclasses
import functools


@functools.cache
def _index() -> dict[str, str]:
    """`field name -> area name`, for every unambiguous field of `DataStructure`.

    Built once, lazily: importing `process` at module scope would make `_harness` — and
    so every module that touches the harness — unimportable without PROCESS installed
    (`test_process_free_import`).
    """
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
    """`name` (or `"area.field"`) as `(area, field)`.

    Raises
    ------
    KeyError
        If `name` names no `DataStructure` field, or names one in more than one area.
    """
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

    The shape PROCESS's constraint equations want: there is no model to construct and no
    field to read back, only `f(data)`.

    Parameters
    ----------
    call :
        Callable taking the populated `DataStructure` and returning what the port
        returns.
    areas :
        Overrides for arguments whose name does not identify its area — the three
        ambiguous fields, and anything the caller wants to pin. Maps argument name to
        `"area.field"`.

    Returns
    -------
    :
        The callable.
    """
    areas = dict(areas or {})

    def reference(**kwargs):
        from process.core.model import DataStructure  # noqa: PLC0415

        data = DataStructure()
        _poke(data, kwargs, areas)
        return call(data)

    return reference


def process_reference(factory, method: str, outputs, *, call_args=(), areas=None):
    """A `reference` callable: keyword arguments in, `outputs` out.

    Parameters
    ----------
    factory :
        Zero-argument callable returning the PROCESS model, already constructed with
        whatever sub-models it needs.
    method :
        Name of the method to call on it, after the arguments are in place.
    outputs :
        Field names to read back, in order. A single name returns a bare value; two or
        more return a tuple, matching what the port returns.
    call_args :
        Positional arguments for `method`, for the few that take any.
    areas :
        Overrides for arguments whose name does not identify its area; see
        `data_reference`.

    Returns
    -------
    :
        The callable.
    """
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
