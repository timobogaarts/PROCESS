"""Outputs a node **states** rather than derives: each read at `^stated.<the place it is
for>` and written, unchanged, to that place.

    TfCryoplantEfficiency = states(tfcoil, "eff_tf_cryo")

The body passes its values through. There is one per arity, since a read binds a
parameter by position and a variadic parameter reads nothing.
"""

from __future__ import annotations

from cottax.cx import Explicit, Scope, absolute
from cottax.pytree.mint import Minted

__all__ = ["STATED", "states"]

STATED = Minted("stated")
"""The namespace a stated value is read from: `^stated.<the place it is for>`."""


def _pass_1(a):
    return a


def _pass_2(a, b):
    return a, b


def _pass_3(a, b, c):
    return a, b, c


def _pass_4(a, b, c, d):
    return a, b, c, d


def _pass_5(a, b, c, d, e):
    return a, b, c, d, e


def _pass_6(a, b, c, d, e, f):
    return a, b, c, d, e, f


def _pass_7(a, b, c, d, e, f, g):
    return a, b, c, d, e, f, g


def _pass_8(a, b, c, d, e, f, g, h):
    return a, b, c, d, e, f, g, h


_PASS = (_pass_1, _pass_2, _pass_3, _pass_4, _pass_5, _pass_6, _pass_7, _pass_8)


def states(scope: Scope, *outputs: str) -> Explicit:
    """A node writing `outputs` under `scope`, each read from `^stated.<output>`."""
    places = tuple(scope.place(o) for o in outputs)
    body = _PASS[len(places) - 1]
    names = body.__code__.co_varnames[: len(places)]
    return absolute.function(
        body, *places, reads=dict(zip(names, map(STATED, places), strict=True))
    )
