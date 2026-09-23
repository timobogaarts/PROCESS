"""PROCESS's figures of merit as a model node.

The metric is a ported function in `functional_process.models.objectives`, declared here
the way a constraint is: reads resolved against the graph the run holds, switches frozen
at assembly. The node owns `^cond.numerics.objf`, and a **maximised** merit is negated
in the body -- PROCESS's own `np.sign(i_figure_merit)`, applied where the value is
computed rather than by a second node standing for no computation.
"""

import equinox as eqx
from cottax.interfaces import Function, Implemented
from cottax.pytree.mint import prefix_path
from cottax.pytree.path import GetAttrKey, NodePath, VarPath

from functional_process.cottax.models.constraints import (
    COND,
    REFERENCE_SWITCH_VALUES,
    Resolver,
    bind,
)


class ObjectiveSelection(eqx.Module):
    """Which figure of merit this run states, and in which direction -- resolved once,
    at the input-parsing boundary, and never re-derived at assembly.
    """

    metric: object
    """The ported `objective_metric_<id>`, already selected."""

    maximise: bool = eqx.field(static=True)
    """`i_figure_merit < 0` in PROCESS's spelling."""


class Metric(eqx.Module):
    """`objective_metric(*args, **switches)`, negated where the merit is maximised."""

    fn: object
    names: tuple = ()
    switches: tuple = ()
    """`((parameter, value), ...)`, the switch arguments frozen at assembly."""
    negate: bool = eqx.field(static=True, default=False)

    def __call__(self, *args):
        """The metric at `args`, negated where the merit is maximised."""
        arguments = dict(zip(self.names, args, strict=True))
        arguments.update(self.switches)
        value = self.fn(**arguments)
        return -value if self.negate else value


def objective_place(label: str = "") -> VarPath:
    """Where this run's figure of merit is computed: `^cond.numerics.objf<label>`."""
    return prefix_path(
        VarPath((GetAttrKey("numerics"), GetAttrKey(f"objf{label}"))), COND
    )


def objective_node(graph_variables, selection, switches=None, label: str = ""):
    """`(name, definition)` for this run's figure of merit over a graph's variables.

    The node is bound at `Objective<label>` and owns `^cond.numerics.objf<label>`.
    `label` suffixes both, for a graph stating more than one objective -- two sequential
    optimisers, say.
    """
    switches = REFERENCE_SWITCH_VALUES if switches is None else switches
    static, read, reads = bind(
        selection.metric, Resolver(graph_variables), switches
    )
    return (
        NodePath((GetAttrKey(f"Objective{label}"),)),
        Implemented(
            Function(reads, (objective_place(label),)),
            Metric(selection.metric, tuple(read), static, selection.maximise),
        ),
    )
