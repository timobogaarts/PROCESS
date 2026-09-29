"""The port's reading of a posed problem, and the level its drivers implement.

cottax hands a driver implementation the problem (`Posed`: the condition node and a
condition map, both static) and the context (the values the problem's block reads from
outside it). The port's drivers were written against a reading of the two together:
`gaps(*unknowns)` gives the objectives and one gap per relation, `lhs - rhs`, an absent
side being zero; `gaps.symbols` says what each relation is held to. That reading is
`Gaps` here, and `NextValues` its fixed-point counterpart.

A reading is a pytree whose children are the context's values and whose static part is
the posed problem, which compares by value: so `host_cache` keys a compiled program on
the problem and hands it the values, as cottax's own implementations do with `args`.

A bound (`lb <= u`, `u <= ub` against a parameter or zero) is not a gap: it is the box
(`box`), as in `canonical`.
"""

import operator
from abc import abstractmethod
from functools import cached_property
from typing import Any

import jax
from cottax.core.frozen import AbstractClassVar, Frozen
from cottax.driven import Driver
from cottax.execution import ByRole, Context, DriverImplementation, Posed, Unknowns
from cottax.jax_execution.solvers import NotSquare
from cottax.jax_execution.solvers.canonical_nlp import Box, box
from cottax.relational import Le, Max, Relation
from jax.tree_util import GetAttrKey, register_pytree_with_keys

__all__ = [
    "Gaps",
    "GapsDriver",
    "IterationDriver",
    "NextValues",
    "PortDriver",
    "require_square",
]


def _is_bound(posed: Posed, r: Relation) -> bool:
    """`side <= u` or `u <= side`, the side a parameter or zero: the box's, not a gap."""
    condition = posed.condition
    fixed = lambda side: side is None or side in condition.parameters  # noqa: E731
    return r.op is Le and (
        (r.rhs in condition.unknowns and fixed(r.lhs))
        or (r.lhs in condition.unknowns and fixed(r.rhs))
    )


def _gap(lhs: Any, rhs: Any) -> Any:
    """`lhs - rhs` leaf-wise, an absent side being zero."""
    if rhs is None:
        return lhs
    if lhs is None:
        return jax.tree.map(operator.neg, rhs)
    return jax.tree.map(operator.sub, lhs, rhs)


class Reading(Frozen):
    """The level: a posed problem at one context. A pytree: the context is its child."""

    posed: Posed
    context: Context
    order: tuple[int, ...] = ()
    """The relations in this order (indices into those read), or as stated if empty: a
    driver partitioning by position reorders the reading, never the problem."""

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        register_pytree_with_keys(
            cls,
            lambda reading: (
                ((GetAttrKey("context"), reading.context),),
                (reading.posed, reading.order),
            ),
            lambda static, children: cls(static[0], children[0], static[1]),
        )

    @cached_property
    def relations(self) -> tuple[Relation, ...]:
        """The relations read, the bounds left to the box, in `order`."""
        read = tuple(
            r for r in self.posed.condition.relations if not _is_bound(self.posed, r)
        )
        return tuple(read[i] for i in self.order) if self.order else read

    @property
    def symbols(self) -> tuple:
        """What each relation is held to, `Eq` or `Le`, parallel to the gaps."""
        return tuple(r.op for r in self.relations)

    @property
    def unknowns(self) -> tuple:
        return self.posed.condition.unknowns

    @property
    def objectives(self) -> tuple:
        """The objectives' places."""
        return tuple(o.place for o in self.posed.condition.objectives)

    def values(self, unknowns: Unknowns) -> dict:
        """Every place the condition reads or determines, at `unknowns`."""
        condition = self.posed.condition
        read = self.posed.conditions(tuple(unknowns), self.context)
        return dict(zip(condition.unknowns, unknowns, strict=True)) | dict(
            zip(condition.reads, read, strict=True)
        )

    def box(self, unknowns: Unknowns) -> Box:
        """Each unknown's lower and upper bound, `None` where it has none.

        A bound is a parameter, fixed while the unknowns move, so its value is read off
        the context where the condition map says where each context value goes; only a
        map that does not say is evaluated, which walks the whole interior.
        """
        condition = self.posed.condition
        places = getattr(self.posed.conditions, "context_places", ())
        at = dict(zip(places, self.context, strict=True)) if places else {}
        if all(p in at for p in condition.parameters):
            values = tuple(at.get(r) for r in condition.reads)
        else:
            values = self.posed.conditions(tuple(unknowns), self.context)
        return box(condition, tuple(unknowns), values)

    @abstractmethod
    def __call__(self, *unknowns: Any) -> Any:
        """This reading of the problem, at one point."""


class Gaps(Reading):
    """`(objectives, one gap per relation)`, the bounds aside; a maximised objective
    negated, so every driver minimises."""

    def __call__(self, *unknowns: Any) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        at = self.values(unknowns)
        value = lambda p: None if p is None else at[p]  # noqa: E731
        return (
            tuple(
                -at[o.place] if o.sense is Max else at[o.place]
                for o in self.posed.condition.objectives
            ),
            tuple(_gap(value(r.lhs), value(r.rhs)) for r in self.relations),
        )


class NextValues(Reading):
    """Where each unknown goes next, `u = g(u)` read as `u <- g(u)`, in their order."""

    def __call__(self, *unknowns: Any) -> tuple[Any, ...]:
        at = self.values(unknowns)
        update = {r.lhs: r.rhs for r in self.relations}
        return tuple(at[update[u]] for u in self.unknowns)


class PortDriver[D: Driver](DriverImplementation[D]):
    """
    The level the port's drivers implement: a reading of the posed problem (`read`), then
    `solve(reading, data)`, which returns the unknowns and then one value per role in
    `computes`. What the contract asks back (`posed.reports`) is taken from those.
    """

    computes: AbstractClassVar[tuple]
    """The report roles `solve` returns, in order, after the unknowns."""

    @abstractmethod
    def read(self, posed: Posed, context: Context) -> Reading:
        """The reading `solve` is handed."""

    @abstractmethod
    def solve(self, reading: Reading, data: ByRole) -> tuple[Any, ...]:
        """The unknowns' values in order, then one per role in `computes`."""

    def __call__(self, posed: Posed, context: Context, data: ByRole):
        answer = tuple(self.solve(self.read(posed, tuple(context)), data))
        n = len(posed.condition.unknowns)
        reported = dict(zip(self.computes, answer[n:], strict=True))
        return answer[:n], {role: (reported[role],) for role in posed.reports}


class GapsDriver[D: Driver](PortDriver[D]):
    """A port driver reading every relation as a gap."""

    def read(self, posed: Posed, context: Context) -> Gaps:
        return Gaps(posed, context)


class IterationDriver[D: Driver](PortDriver[D]):
    """A port driver reading `u = g(u)` as where each unknown goes next."""

    def read(self, posed: Posed, context: Context) -> NextValues:
        return NextValues(posed, context)


def require_square(driver: str, gaps: Gaps, at) -> None:
    """Refuse a problem with not as many numbers in its gaps as in its unknowns, counted
    in leaves at `at`: a driver inverting the gaps needs them square."""
    _objectives, at_gaps = gaps(*at)
    got = sum(jax.numpy.size(leaf) for leaf in jax.tree.leaves(at_gaps))
    unknowns = sum(jax.numpy.size(leaf) for leaf in jax.tree.leaves(tuple(at)))
    if got != unknowns:
        raise NotSquare(driver, got, unknowns).raised
