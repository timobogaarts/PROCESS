"""Outputs a node **states** rather than derives: read as ports, never held as fields.

This is `models/carried.py`'s successor, and the difference is where the value lives.
`carried.py` held it as a `jax.Array` field of the declaration and made the declaration
reachable through `fn` (a `jtu.Partial`) so the field would reach the trace as an
argument. `cottax` now **refuses an array in the graph outright** --
`graph._check_bindings` asks `hash(node)` of every binding -- and refuses a
`jtu.Partial` body besides, so that route is closed. The reason is the one
`plans/closures_and_undeclared_inputs.md` states: a value carried in the graph is an
input the graph does not name, and the repair is to name it.

So it is named. A `StatesValues` declaration has **one read per output**, at the output's
own place under the `^stated` mint:

    .initialisation.tf_cryoplant_efficiency
        reads  ^stated.tfcoil.eff_tf_cryo
        owns   .tfcoil.eff_tf_cryo

and the body is the identity. Nothing is invented: the read's name is the write's name,
one namespace out, so a declaration says only *which* variables it states and never what
they are called. The value reaches a run through the env, like every other boundary
input, and `sand_harness.KNOWN_MINT_VALUES` is where each one's value is stated
(`STATED_VALUES` below is registered into it).

Why the value may not stay a Python scalar either
-------------------------------------------------
The obvious reading of the array ban is "make it a `float` again and the graph hashes".
It does, and it reinstates both defects `_audit/optimise_design.md` §25/§28 measured:

- **A baked constant is not merely frozen, it is deleted.** §25's Arm C found XLA folds a
  literal `0.0` and removes the subexpressions it multiplies --
  `NoDiamagneticCurrent`'s own docstring: *"a constant this node's readers multiply by is
  one XLA deletes the readers of"*. That is why the `None`/`Absent` arms carry a field at
  all; it has nothing to do with the value varying.
- **The module gets specialised to one IN.DAT.** `eqx.filter_jit` keys its cache on the
  non-array half of its arguments, so a value resolved at assembly is part of the key and
  `Scan` recompiles the whole module at every point (38 635 HLO lines for the
  stellarator, 132 125 for the tokamak -- §31.1).

An env value is a traced argument, so it is neither folded nor in the key. That is the
whole of the fix, and it is uniform: there is no member of this family for which a
specialised scalar would be right.

What each spelling costs, and why the ports are minted rather than spelled out
-----------------------------------------------------------------------------
The read has to be a name no `DataStructure` field answers by accident, and it has to
survive `sand_harness.ground_truth`'s fallback chain. A mint is both: `is_minted` says
`boundary.py` should count it apart from a physical input, and `unminted` gives the
place the value belongs to, so a missing `STATED_VALUES` row degrades to *"whatever
PROCESS holds at the output path"* rather than to a silent `0.0`. `test_stated.py`
refuses that degradation rather than relying on it -- every stated port must have a row.
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
    """A node whose outputs are stated at assembly and read from the env.

    One overridden query and one body, and neither is per-declaration: `inputs` is one
    `^stated.*` read per declared output, and `__call__` is the identity on them. A
    subclass therefore declares **only its `OutputInto`s** -- no field, no literal, no
    body -- which is what makes the family auditable: there is nowhere left to put a
    number.

    **The arity check is live again.** `carried.py` gave up
    `ImplementedFunction.__check_init__`'s "n input(s) declared, but ..." because a
    `jtu.Partial` reports `(*args)`; `__call__` here reports `(*values)`, so it is still
    inert -- but the shape it is protecting against cannot arise, since `inputs` is
    *derived* from `outputs` rather than written by hand.
    """

    @property
    def _params(self) -> list:
        """No parameter names a read here, so there is none to check.

        `NodalDeclaration.__check_init__` walks `_params` asking each for its
        `FromExactly` default -- right for a declaration whose *signature* names its
        reads, and wrong for one whose reads are a function of its outputs: `(*values)`
        has no defaults to find and is refused. Overriding `_params` rather than
        `__check_init__` because equinox runs every `__check_init__` in the MRO, so an
        override cannot suppress the base's; emptying what it walks can, and it says the
        true thing besides. What the check buys -- every read named -- is bought here by
        construction, since `inputs` is derived and there is no second place a name could
        come from.
        """
        return []

    @property
    def inputs(self) -> tuple[In, ...]:
        """One read per declared output, at the output's own place under `^stated`."""
        return tuple(stated_port(out) for out in self.outputs)

    def __call__(self, *values):
        """The stated values, unchanged -- a tuple where several are declared."""
        return values[0] if len(values) == 1 else values
