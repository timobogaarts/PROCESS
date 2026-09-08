"""A sample point read off the reference machine, instead of copied into the contract.

Most contracts carry their evaluation point as a literal dict: one line per argument,
per unit, 8,990 lines across the tree. For 58 % of them every argument is an unambiguous
`DataStructure` field, so the point can be *read* from a converged reference state
instead of written down.

**The oracle is unaffected**, which is what makes this safe. A `legacy_sample` is not
checked against a recorded answer -- the harness calls PROCESS at that point and compares
live -- so a point derived from the reference machine is exactly as good an oracle, and
is an operating point the device actually reaches rather than one someone typed. What is
lost is the regimes a hand-chosen point reached and the reference machine does not, which
is what the shared fuzz domain (`fuzz_domain.py`) now covers.

A unit whose derived point does not suit it keeps its literal sample and says why.
"""

import functools
import inspect

from functional_process.cottax._harness.process_reference import _resolve
from functional_process.cottax._harness.sampling import Sample

REFERENCE_INPUT_FILE = "tests/regression/input_files/stellarator_helias.IN.DAT"
"""The machine every derived point is read from."""


@functools.cache
def _state():
    """The reference machine's **converged** `DataStructure`, once per process.

    Converged, not cold: `init_process` leaves every derived field at `0.0`
    (`rminor`, `vol_plasma`, ...), and a point made of zeros is degenerate for most
    units. `reference_run` is disk-cached, so the first import pays a solve and the
    rest read a pickle.

    This is PROCESS's own answer being used to *choose the point*, never to supply the
    expected value -- the contract still calls PROCESS at that point and compares live.
    """
    from functional_process.cottax.sand_harness import reference_run  # noqa: PLC0415

    return reference_run(REFERENCE_INPUT_FILE).data


def reference_point(fn, label="reference-machine", **overrides):
    """`[Sample]` holding `fn`'s arguments, read off the reference machine.

    Parameters
    ----------
    fn :
        The ported callable. Its signature names the arguments to read.
    label :
        Sample label, for the test id.
    **overrides :
        Values to use instead of the state's -- for a switch the point must pin, or a
        field the reference machine leaves at a value this unit cannot use.

    Returns
    -------
    :
        A one-element list, so it drops straight into `samples`.

    Raises
    ------
    KeyError
        If an argument is not an unambiguous `DataStructure` field and is not overridden.
        The contract should keep a literal sample instead.
    """
    state = _state()
    kwargs = {}
    for name in inspect.signature(fn).parameters:
        if name in overrides:
            kwargs[name] = overrides[name]
            continue
        area, field = _resolve(name)
        kwargs[name] = getattr(getattr(state, area), field)
    return [Sample(kwargs, "reference", label)]
