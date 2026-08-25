"""Marks `tests/` as a package so test module names are fully qualified.

`tests/functional_process/**` mirrors the `functional_process/` package, and fifteen of
its basenames (`test_availability.py`, `test_vacuum.py`, ...) also exist under
`tests/unit/**`. In pytest's default `prepend` import mode a test module in a directory
with no `__init__.py` is imported under its bare basename, so those pairs would collide
the moment `pytest tests` collected both. The `__init__.py` chain from here down through
`tests/functional_process/**` gives that half of the tree fully qualified module names
(`tests.functional_process.models.test_vacuum`) and leaves every other subdirectory --
`unit`, `integration`, `regression`, `examples` -- importing exactly as it did before.
"""
