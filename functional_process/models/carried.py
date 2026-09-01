"""Values a node **carries** rather than computes, kept where the compiler can see them.

`_audit/optimise_design.md` §25 censused the compiled XLA modules and found node
outputs -- not boundary inputs -- baked in as compile-time constants: nine values on
`stellarator_helias` and twelve on `large_tokamak_nof`. Every one came from a node whose
body hands back a Python scalar, either a literal (`StellaratorPulseTimes`' `return 0.0,
0.0, 3.15576e7, 0.0`) or a value resolved from the input file at assembly
(`TfCryoplantEfficiency`'s `return self.value`). §28 re-censused it from the graph rather
than from the module and found **fourteen** such nodes over the seven tracked
configurations, twenty-six output paths between them.

Both halves of that cost something:

- **XLA folds arithmetic on a constant at compile time.** §25's Arm D found `1 /
  eyoung_ins`, `4 * eyoung_ins` and a series stiffness computed on the host while
  lowering `tfcoil/stress.py:188`; its Arm C found that a baked `0.0` is not merely
  frozen but *deleted*, taking the subexpressions it multiplies with it -- the
  `big_q_plasma` defect (§26) one level down, at the compiler rather than at the port.
- **The compiled module is specialised to one IN.DAT.** `Scan` -- PROCESS's core
  workflow, one input variable swept across a range -- therefore recompiles the whole
  38,635-line stellarator module (132,125 for the tokamak) at every scan point, because
  the carried value is part of `eqx.filter_jit`'s cache key.

Where the fix has to go, measured rather than reasoned
-----------------------------------------------------
The obvious repair -- `return jnp.asarray(self.value)` -- **buys nothing at all**, and
§28 measured that before changing anything. Two facts compose:

1. An array built *inside* the traced body is a jaxpr constant exactly as the Python
   float was. On a minimal probe the optimised HLO is identical (2 parameters, 4
   constants) for `x * 0.13` and for `x * jnp.asarray(0.13)`.
2. `cottax`'s `ExplicitFunction.node_definition` builds `CallableNode(fn=self.__call__)`,
   and a **plain bound method is not a pytree**. `jax.tree_util.tree_leaves` of a real
   graph node definition returns *no* array leaves even when the declaration holds one,
   so `eqx.filter_jit` never sees the field and cannot trace it.

What does work is making the declaration itself reachable through `fn`, which
`jax.tree_util.Partial` does: its `args` are pytree children, so the declaration's array
leaves become **arguments** of the compiled program and its non-array leaves stay in the
cache key where they belong. Measured on the same probe: 4 parameters, 3 constants, and
two declarations differing only in the carried value share one compiled program (second
call, 0 compiles).

`fn=self` -- the declaration used directly as the callable, which is a pytree without any
wrapper -- has the same effect on the trace and was rejected: `Graph.__hash__` hashes
`tuple(self.definitions.items())`, `mdf._hashable` and `sand_harness._SCHEDULE_WHOLE`
depend on that hash, and an `eqx.Module` holding a `jax.Array` is unhashable. A
`Partial` hashes by identity and keeps every one of those working.

**This belongs upstream, not here.** The general statement is "a node's own array fields
are data the graph should trace, not configuration it should specialise on", which is a
property of `CallableNode` and not of these fourteen declarations. Until `cottax` says
it, `CarriesValues` says it for the declarations that need it; the day
`ExplicitFunction.node_definition` builds a pytree `fn` itself, this class becomes an
empty subclass and can go.
"""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from cottax.interfaces.pytree_namespace_module import ExplicitFunction
from cottax.spec import CallableNode
from cottax.tools.cache import cached_query

__all__ = ["CarriesValues", "carried", "carried_all"]


def carried(default=dataclasses.MISSING, **kwargs):
    """A declaration field holding one carried value, as an array and never as a scalar.

    `kw_only` because every carried value is passed by name at assembly (`indat`), and
    `converter=jnp.asarray` because the conversion has to happen when the declaration is
    *built* -- doing it in the body is the repair that does nothing (see the module
    docstring). `jnp.asarray` of a Python float keeps `weak_type=True`, so the value
    promotes exactly as the literal it replaces did.

    The conversion runs at assembly, which is after every entry point has set
    `jax_enable_x64`; a caller that builds a machine before enabling it would get
    `float32` here, the same way it already would for every other array in this port
    (`CLAUDE.md`, `traceability_policy.md` §Precision).
    """
    if default is not dataclasses.MISSING:
        kwargs["default"] = default
    return eqx.field(kw_only=True, converter=jnp.asarray, **kwargs)


def carried_all(**kwargs):
    """`carried`, for the several values one ported function hands back together.

    `CentrepostNeutronicsAbsent`'s four zeros come out of
    `calculate_centrepost_neutronics_absent` as one tuple, and splitting them into four
    fields would put four literals here that the registered unit already states.
    """
    return eqx.field(
        kw_only=True,
        converter=lambda values: tuple(jnp.asarray(v) for v in values),
        **kwargs,
    )


def _apply(declaration, *args):
    """`declaration(*args)`, as a module-level function.

    `jtu.Partial`'s `func` is static aux data and its `args` are children, so this has to
    be one function shared by every node -- a per-instance closure would put a distinct
    object in the cache key and lose the sharing the whole scheme is for.
    """
    return declaration(*args)


class CarriesValues(ExplicitFunction):
    """An `ExplicitFunction` whose own array fields reach the trace as arguments.

    One overridden query, and nothing else: the reads, the writes, the body and the name
    are `ExplicitFunction`'s. See the module docstring for why `fn` has to be a pytree
    and why it is a `Partial` rather than the declaration itself.

    **What it costs.** `CallableNode.__check_init__` binds `len(inputs)` positional
    arguments against `inspect.signature(self.fn)`; a `Partial` reports `(*args)`, so the
    arity check that catches "three inputs declared, two parameters written" is inert for
    these nodes. Acceptable here because every one of them declares *no* inputs -- there
    is no arity to get wrong -- and it is one more reason the general form belongs in
    `cottax`, where the signature is still in hand.
    """

    @cached_query
    def node_definition(self) -> CallableNode:
        """`ExplicitFunction.node_definition`, with the declaration held in a pytree."""
        return CallableNode(
            inputs=self.inputs,
            outputs=self.outputs,
            fn=jtu.Partial(_apply, self),
        )
