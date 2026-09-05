"""Build cottax `VarPath`s from PROCESS's own `.area.field` spelling.

`_audit/naming_convention.md`: a `VarPath` is rooted at the PROCESS data-structure area
name and spelled exactly as PROCESS spells the field -- no new names invented. This
module is the one place that spelling turns into a `cottax.VarPath`, so every node wrap
in this tree constructs its ports the same way rather than hand-building `GetAttrKey`
tuples per unit.
"""

from cottax import VarPath
from jax.tree_util import GetAttrKey


def path(dotted):
    """`".physics.rmajor"` -> `VarPath` rooted at `physics`.

    Parameters
    ----------
    dotted :
        A PROCESS `VarPath` spelling, `.area.field`, leading dot required -- the same
        rendering `naming_convention.md` and every audit record already use.

    Returns
    -------
    :
        The `VarPath`.

    Raises
    ------
    ValueError
        If `dotted` has no leading dot.
    """
    if not dotted.startswith("."):
        raise ValueError(
            f"expected a leading '.', got {dotted!r} (see naming_convention.md)"
        )
    return VarPath(tuple(GetAttrKey(k) for k in dotted[1:].split(".")))
