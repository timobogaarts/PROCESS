"""Graph-assembly-time selection of switch-gated node alternatives.

`_audit/naming_convention.md` § "Switches are not ports" already fixed the policy: a
topology-changing switch "is **not** a `VarPath` on any node at all -- it's consumed by
the Python code that assembles the `Graph`". This module is that Python code, made
explicit and checkable instead of being a hand-edited import list in `total_process.py`.

## Why assembly time is the *only* correct answer here

Not a preference. Every switch PROCESS has is a constant for the whole solve:

    grep -n "\\"i_\\|'i_" process/core/solver/iteration_variables.py   -> no matches
    grep -n "\\"i_\\|'i_\\|istell" process/core/scan.py                -> no matches

No switch is an iteration variable, and no switch is a scan variable, so no switch can
change between two evaluations of one assembled graph. `Scan` re-solves from scratch per
point anyway. A switch therefore carries no derivative, participates in no edge, and has
nothing to contribute to a `Graph` -- which is exactly cottax's position that graph
structure is decided by the caller once, not re-read per evaluation.

The rejected alternative was a single node owning the union of every variant's ports and
branching internally on the switch value. That would make the node read
`eta_ecrh_injector_wall_plug` *and* `eta_lowhyb_injector_wall_plug` regardless of which
one is live, inventing graph edges that do not exist in the run being modelled, and would
put a non-differentiable integer on a port. It also loses the result below.

## What the switch actually changes

More than which formula runs. Compare the two `ipowerflow` arms of `build.py`:
`AFwTotalWithPowerflow` reads `.fwbs.f_ster_div_single`, which `Divertor` owns, while
`Divertor` reads `.first_wall.a_fw_total`, which both arms own -- so `ipowerflow != 0`
has a genuine two-node SCC and `ipowerflow == 0` is acyclic. **The switch decides whether
the graph has a cycle**, which is not a fact any single fused node could express, and is
directly the question `_audit/next_steps.md` § 5 exists to ask. `test_configuration.py`
asserts both halves so the claim stays checked rather than narrated.

## Scope

Only *topology-changing* switches belong here -- the first of the two categories in
`naming_convention.md`. The second (a formula-changing switch with a provably identical
reads-set, kept as a static kwarg on one node's `fn`) stays where it is: that is
`EcrhDensityLimit(i_plasma_pedestal=0)` in `total_process.py`, and it is deliberately
*not* modelled as a `Switch` here, because nothing about the graph's shape depends on it.
"""

from dataclasses import dataclass, field

from cottax.interfaces.pytree_namespace_module import node_and_names, to_graph


@dataclass(frozen=True)
class Alternative:
    """One arm of a switch: the value that selects it, and what it contributes.

    An arm that is known to exist in PROCESS but has not been ported declares itself with
    `unported` set and no `declarations`. That is the point of listing it at all: a
    requested-but-unported value then fails with the reason it is missing, instead of
    silently assembling a graph with a hole where the arm's outputs should be.

    `unproduced` is the third state, and it is deliberately *not* the same as `unported`
    even though both mean "no nodes were written for this arm". The difference is what
    happens when a caller asks for the value:

    - `unported` **raises**. Use it when assembling anyway would hand the caller a graph
      that looks complete but is wrong -- e.g. `.stellarator.istell in 1..5`, where the
      machine-preset tables the arm needs are missing but every *other* node still
      assembles, so the graph would silently run with the wrong preset data.
    - `unproduced` **assembles nothing**, leaving the fields that arm would own with no
      producer at all. Use it when the honest answer is "this configuration's graph does
      not compute these values", so a consumer surfaces as an unowned (boundary) input
      rather than being silently satisfied by the *other* arm's formula. That is a
      weaker guarantee than `unported`'s, and it is only correct when the arm's outputs
      have no other producer in the assembled graph -- which `build_graph` cannot check
      for you, since "no producer" is exactly what it is being told to produce.

    The first real instance is `.costs.i_cost_model == 1` (`KOVARI_2014`, PROCESS's own
    default): `costs_2015.py` has no `cottax` nodes at all, so its arm contributes
    nothing, and the alternative -- registering `costs.py`'s 1990-model nodes
    unconditionally instead -- would make the default graph compute `.costs.coe` by the
    *wrong* cost model, the `EcrhDensityLimit` bug class this whole module exists to
    make impossible. See `total_process.py`'s `.costs.i_cost_model` switch for the full
    write-up, including the two rejected alternatives.

    Attributes
    ----------
    value :
        The switch value in `data` that selects this arm.
    declarations :
        The `NodalDeclaration` classes/instances this arm contributes.
    unported :
        Why this arm has no declarations *and must be refused*, or `None`.
    unproduced :
        Why this arm has no declarations *and assembles as empty*, or `None`.
    """

    value: int
    declarations: tuple = ()
    unported: str | None = None
    unproduced: str | None = None

    def __post_init__(self):
        states = [
            name
            for name, given in (
                ("declarations", bool(self.declarations)),
                ("unported", self.unported is not None),
                ("unproduced", self.unproduced is not None),
            )
            if given
        ]
        if len(states) > 1:
            raise ValueError(
                f"alternative {self.value} declares {sorted(states)} together -- an arm "
                f"is exactly one of ported (declarations), refused (unported) or empty "
                f"(unproduced)"
            )
        if not states:
            raise ValueError(
                f"alternative {self.value} declares nothing at all -- say which of "
                f"declarations/unported/unproduced it is, so a reader can tell an "
                f"empty arm from an oversight"
            )

    @property
    def is_ported(self):
        """Whether this arm contributes nodes. `unported`/`unproduced` arms do not."""
        return bool(self.declarations)


@dataclass(frozen=True)
class Switch:
    """A topology-changing switch and the graph arms its values select.

    Attributes
    ----------
    path :
        The switch's `.area.field` string. Not a `VarPath` -- deliberately, per
        `naming_convention.md`: it is a cross-reference to `core/solver/switches.md` and
        to `process/data_structure/`, not a port. Kept as a string so nothing is tempted
        to bind it into a graph.
    default :
        PROCESS's own default, read from the `data_structure` dataclass field (cited in
        `switches.md`). Assembling with no explicit choice reproduces the run PROCESS
        would do with a silent IN.DAT.
    alternatives :
        One per known value of the switch, ported or not.
    """

    path: str
    default: int
    alternatives: tuple[Alternative, ...]

    def __post_init__(self):
        values = [alternative.value for alternative in self.alternatives]
        if len(values) != len(set(values)):
            raise ValueError(f"{self.path}: duplicate alternative values {values}")
        if self.default not in values:
            raise ValueError(
                f"{self.path}: default {self.default} is not among the declared "
                f"alternatives {sorted(values)}"
            )

    def choose(self, value):
        """The declarations for `value`, or a loud failure naming what is available."""
        for alternative in self.alternatives:
            if alternative.value == value:
                if alternative.unported is not None:
                    raise NotImplementedError(
                        f"{self.path} == {value} is a real PROCESS branch but is not "
                        f"ported: {alternative.unported}"
                    )
                return alternative.declarations  # `()` for an `unproduced` arm
        raise ValueError(
            f"{self.path} == {value} is not a known alternative; declared values are "
            f"{sorted(a.value for a in self.alternatives)}"
        )

    def check_arms_are_exclusive(self):
        """Every ported pair of arms must actually collide on an owned output.

        Two nodes filed under one switch that own *disjoint* outputs are not alternatives
        at all -- they could coexist in one graph, and filing them here would wrongly
        delete one of them from every assembled graph. Since the whole reason this module
        exists is that `Graph` raises on duplicate ownership, the arms are checked to be
        the thing that would have raised.

        A single-ported-arm switch (the rest `unported` or `unproduced`) has no pair to
        check and passes.
        """
        ported = [a for a in self.alternatives if a.is_ported]
        owned = {
            alternative.value: {
                output
                for _, definition in node_and_names(alternative.declarations)
                for output in definition.outputs
            }
            for alternative in ported
        }
        for i, left in enumerate(ported):
            for right in ported[i + 1 :]:
                shared = owned[left.value] & owned[right.value]
                if not shared:
                    raise ValueError(
                        f"{self.path}: alternatives {left.value} and {right.value} own "
                        f"no output in common, so they are not mutually exclusive and "
                        f"do not belong under one switch -- both could be assembled "
                        f"into one graph"
                    )


@dataclass(frozen=True)
class Configuration:
    """A choice of value for each topology switch; PROCESS's defaults where unstated.

    Keyed by the switch's `.area.field` path -- the same string `switches.md` sections
    and `process/data_structure/` field lookups use -- so a configuration is readable
    against an IN.DAT without a translation table.
    """

    choices: dict[str, int] = field(default_factory=dict)

    def value_for(self, switch):
        return self.choices.get(switch.path, switch.default)


def declarations_for(configuration, common, switches):
    """Everything that belongs in the graph `configuration` describes.

    Splitting this out from `to_graph` is what lets a test inspect the selection without
    paying to build a graph, and what lets an unknown switch path be reported as such --
    a typo'd key in `choices` would otherwise be silently ignored, which is the failure
    mode this whole module is meant to remove.
    """
    known = {switch.path: switch for switch in switches}
    unknown = set(configuration.choices) - set(known)
    if unknown:
        raise ValueError(
            f"configuration sets unknown switch(es) {sorted(unknown)}; declared "
            f"topology switches are {sorted(known)}"
        )
    selected = [switch.choose(configuration.value_for(switch)) for switch in switches]
    return (*common, *(d for arm in selected for d in arm))


def build_graph(configuration, common, switches):
    """`to_graph` of exactly the arms `configuration` selects."""
    for switch in switches:
        switch.check_arms_are_exclusive()
    return to_graph(*declarations_for(configuration, common, switches))
