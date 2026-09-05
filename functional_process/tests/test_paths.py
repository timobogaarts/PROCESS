"""`paths.data` -- the area namespace, and the check that is its whole point.

`_Root` was `cottax`'s until that package deleted it; these pin the behaviour the port
depends on, which it never had a test of its own for while the class lived elsewhere.
"""

import pytest
from cottax.interfaces.pytree_namespace_module import Area

from functional_process.cottax.paths import AREAS, data


def test_every_area_of_the_data_structure_is_reachable():
    """`AREAS` is read off `DataStructure`'s fields, so this is really a check that
    nothing in the namespace layer drops or renames one.
    """
    assert len(AREAS) == 36
    assert all(isinstance(getattr(data, name), Area) for name in AREAS)


def test_a_misspelled_area_is_refused_with_a_suggestion():
    """The reason this class exists at all. A typo'd area is otherwise a boundary input
    that nothing ever writes, and **no value test can see it** -- the field reads as its
    default, so every number still agrees. It is caught at declaration time or not at
    all.
    """
    typo = "physcis"
    with pytest.raises(AttributeError, match=r"physcis.*Did you mean physics"):
        getattr(data, typo)


def test_a_private_name_is_not_a_path():
    private = "_not_an_area"
    with pytest.raises(AttributeError, match="ordinary attribute access"):
        getattr(data, private)
