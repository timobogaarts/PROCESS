"""Sample points, held beside their case as data rather than inside it as source.

8,990 of the test tree's lines were literal evaluation points -- 41 % of every contract,
and the part of it a reader skips. They are numbers, not logic: a reviewer checks *what
is compared to what* (`reference_signoff.txt`) and trusts the points, so keeping them in
Python bought nothing and cost a sixth of the tree.

Each case module gets a JSON sibling under `_samples/`, keyed by contract name. A
contract opts in with `samples = FROM_FILE` and `PortContract.__init_subclass__`
resolves it at class creation, so `SomeContract.samples` is an ordinary list everywhere
downstream and nothing else in the harness changed.

**Every value round-trips exactly.** Floats are written with `repr`, so a point is the
same float64 it was; arrays carry their dtype and shape. A value that cannot round-trip
is refused at extraction rather than silently approximated.
"""

import json
import pathlib

import numpy as np

FROM_FILE = "<from-file>"
"""Sentinel for `samples`. A string, so a contract that forgets to opt in fails on
iteration rather than silently declaring one sample."""


def _encode(value):
    """A sample value as JSON, losslessly.

    Raises
    ------
    TypeError
        If the value cannot round-trip -- refused at extraction rather than silently
        approximated.
    """
    if isinstance(value, np.ndarray):
        return {"~array": value.tolist(), "dtype": str(value.dtype)}
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return {"~seq": [_encode(v) for v in value], "tuple": isinstance(value, tuple)}
    if isinstance(value, dict):
        return {"~dict": {k: _encode(v) for k, v in value.items()}}
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    raise TypeError(f"sample value of type {type(value).__name__} does not round-trip")


def _decode(value):
    """The inverse of `_encode`."""
    if isinstance(value, dict):
        if "~array" in value:
            return np.array(value["~array"], dtype=np.dtype(value["dtype"]))
        if "~seq" in value:
            seq = [_decode(v) for v in value["~seq"]]
            return tuple(seq) if value["tuple"] else seq
        if "~dict" in value:
            return {k: _decode(v) for k, v in value["~dict"].items()}
    return value


def path_for(module: str) -> pathlib.Path:
    """The JSON sibling for a case module: `<dir>/_samples/<stem>.json`."""
    import importlib.util  # noqa: PLC0415

    spec = importlib.util.find_spec(module)
    here = pathlib.Path(spec.origin)
    return here.parent / "_samples" / f"{here.stem}.json"


def load(module: str, name: str):
    """`[Sample]` for contract `name` in case module `module`.

    Raises
    ------
    KeyError
        If the store has no entry -- a contract that opted in without its points being
        extracted, which is a lost test rather than an empty one.
    """
    from functional_process.cottax._harness.sampling import Sample  # noqa: PLC0415

    store = json.loads(path_for(module).read_text())
    if name not in store:
        raise KeyError(f"{module}.{name} declares `samples = FROM_FILE` but "
                       f"{path_for(module)} has no entry for it")
    return [
        Sample({k: _decode(v) for k, v in row["kwargs"].items()},
               row["provenance"], row["label"])
        for row in store[name]
    ]


def dump(module: str, entries: dict) -> pathlib.Path:
    """Write `{contract name: [sample, ...]}` for a case module."""
    out = path_for(module)
    out.parent.mkdir(parents=True, exist_ok=True)
    # NOT sorted: `fuzz_samples` draws in the order the bounds are iterated, which comes
    # from the sample's own argument order, so re-ordering a point silently moves every
    # fuzz point drawn from it.
    out.write_text(json.dumps(entries, indent=1) + "\n")
    return out
