"""Where the port's variables live: `.<area>.<field>`, the way PROCESS spells it. Each
area is a scope (`Area`): `physics.function(fn, 'rminor')` writes `.physics.rminor`, and a
parameter is read at its own name, in the area PROCESS keeps that field in
(`vocabulary.fields`), the scope's own first. `reads=` is left for what that does not
say: a place the port names itself in another area, or a renamed read."""

import difflib
from collections.abc import Iterable

from cottax.cx import DeclarationSurface, Shipped
from cottax.pytree.consumers.declaration import PytreeNames
from cottax.pytree import VarPath
from jax.tree_util import GetAttrKey

# `AREAS` -- every area PROCESS has, 36 of them. Was
# `tuple(f.name for f in dataclasses.fields(DataStructure))`, read live off PROCESS; the
# list is vocabulary, not physics, so §23.2 vendors it and
# `functional_process/tests/test_vocabulary.py` asserts the vendored tuple still equals
# that expression, order included. Re-exported from here because every caller in the port
# imports it from `paths`.
from functional_process.vocabulary import AREAS
from functional_process.vocabulary.fields import AREAS_OF


PORT_PLACES = {
    "a_tf_wp_no_insulation": "tfcoil",
    "a_tf_wp_with_insulation": "tfcoil",
}
"""Quantities the port keeps in an area of its own choosing where PROCESS has a field of
the same name elsewhere: the stellarator winding pack's two areas, locals of
`winding_pack_total_size` PROCESS never stores (`superconducting_tfcoil` holds the
tokamak's)."""


class Area(PytreeNames):
    """One area of the data structure, reading a name where PROCESS keeps it: in this
    area if PROCESS has it here or has no such field (a quantity the port names), else
    in the one area PROCESS has it in."""

    area: str

    def name(self, x: str) -> VarPath:
        areas = AREAS_OF.get(x, ())
        if areas and self.area not in areas and PORT_PLACES.get(x) != self.area:
            elsewhere = PORT_PLACES.get(x) or (areas[0] if len(areas) == 1 else None)
            if elsewhere is None:
                raise ValueError(
                    f"{x!r} is a field of {areas}, not of {self.area!r}: name its place"
                )
            return VarPath((GetAttrKey(elsewhere), GetAttrKey(x)))
        return VarPath((GetAttrKey(self.area), GetAttrKey(x)))


class _Root:
    """The areas of a `DataStructure`, each a `Scope`; an area PROCESS does not have is
    refused, with the closest it does."""

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
        return DeclarationSurface(Area(name), Shipped())

    def __repr__(self):
        return f"<PROCESS data structure: {len(self._names)} areas>"


data = _Root(AREAS)
"""Every area, by name."""

globals().update({name: getattr(data, name) for name in AREAS})


def written(names: Iterable) -> list[str]:
    """Names as they are written, sorted -- what a message lists.

    Was `cottax.pytree.names.written` until cottax dropped its spelling helper (2026-09-12,
    `af5b2f9`): a place has no order of its own, so the spelling orders them and every
    message that lists names is deterministic.
    """
    return sorted(name.spelling for name in names)


__all__ = ["AREAS", "data", "written", *AREAS]
