"""Outputs a node **states** rather than derives: read as ports, never held as fields.
"""

from __future__ import annotations

from cottax.interfaces.pytree_namespace_module import ExplicitFunction
from cottax.spec import In
from cottax.tools.minting import MintKey, prefix_path

__all__ = ["STATED", "StatesValues", "stated_port"]

STATED = MintKey("stated")
"""The namespace a stated value is read from: `^stated.<the place it is for>`."""


def stated_port(out) -> In:
    """The read that supplies one declared output: `^stated.<out.var>`."""
    return In(prefix_path(out.var, STATED))


class StatesValues(ExplicitFunction):
    """A node whose outputs are stated at assembly and read from the env."""

    @property
    def _params(self) -> list:
        """No parameter names a read here, so there is none to check."""
        return []

    @property
    def inputs(self) -> tuple[In, ...]:
        """One read per declared output, at the output's own place under `^stated`."""
        return tuple(stated_port(out) for out in self.outputs)

    def __call__(self, *values):
        """The stated values, unchanged -- a tuple where several are declared."""
        return values[0] if len(values) == 1 else values
