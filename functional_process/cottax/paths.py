"""Where the port's variables live: `data.<area>.<field>`, the way PROCESS spells it."""

import difflib
from collections.abc import Iterable

from cottax.interfaces.pytree_namespace_module import Area
from jax.tree_util import GetAttrKey

# `AREAS` -- every area PROCESS has, 36 of them. Was
# `tuple(f.name for f in dataclasses.fields(DataStructure))`, read live off PROCESS; the
# list is vocabulary, not physics, so §23.2 vendors it and
# `functional_process/tests/test_vocabulary.py` asserts the vendored tuple still equals
# that expression, order included. Re-exported from here because every caller in the port
# imports it from `paths`.
from functional_process.vocabulary import AREAS


class _Root:
    """The areas of a `DataStructure`, as something whole paths hang off."""

    __slots__ = ("_names",)

    def __init__(self, names):
        object.__setattr__(self, "_names", frozenset(names))

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(
                f"{name!r}: a path is built from ordinary attribute access, not a "
                f"private or dunder name"
            )
        if name not in self._names:
            close = difflib.get_close_matches(name, sorted(self._names), n=3)
            hint = f" Did you mean {', '.join(close)}?" if close else ""
            raise AttributeError(
                f"{name!r} is not an area of this data structure.{hint}"
            )
        return Area((GetAttrKey(name),))

    def __repr__(self):
        return f"<PROCESS data structure: {len(self._names)} areas>"


data = _Root(AREAS)
"""The whole namespace, for the escape hatch:
`FromExactly(data.impurity_radiation.arr[2])`.
"""

globals().update({name: getattr(data, name) for name in AREAS})


def written(names: Iterable) -> list[str]:
    """Names as they are written, sorted -- what a message lists.

    Was `cottax.names.written` until cottax dropped its spelling helper (2026-09-12,
    `af5b2f9`): a place has no order of its own, so the spelling orders them and every
    message that lists names is deterministic.
    """
    return sorted(name.spelling for name in names)


__all__ = ["AREAS", "data", "written", *AREAS]
