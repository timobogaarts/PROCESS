"""Build cottax `VarPath`s from PROCESS's own `.area.field` spelling."""

from cottax import VarPath
from jax.tree_util import GetAttrKey


def path(dotted):
    """`".physics.rmajor"` -> `VarPath` rooted at `physics`."""
    if not dotted.startswith("."):
        raise ValueError(
            f"expected a leading '.', got {dotted!r} (see naming_convention.md)"
        )
    return VarPath(tuple(GetAttrKey(k) for k in dotted[1:].split(".")))
