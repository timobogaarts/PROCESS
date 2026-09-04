"""The vacuum subsystem's namespace.

Beside the nodes it names (`model_tree_design.md` §11).
"""

from cottax.interfaces.pytree_namespace_module import ModelNamespace

from functional_process.models.vacuum.vacuum import DuctDiameterRootFind, VacuumOld


class Vacuum(ModelNamespace):
    """Vacuum pumping and the duct sizing problem."""

    # unit #16, vacuum.py -- `"old"` branch only, matching PROCESS's own default
    # (`.vacuum.i_vacuum_pumping = "old"`, `vacuum_variables.py:18`). Not gated by a
    # `Switch`: the `"simple"` alternative (`VacuumPumpingSimple`) owns a disjoint
    # output, so this switch fails `check_arms_are_exclusive` -- see
    # `TOPOLOGY_SWITCHES`'s docstring above. `VacuumPumpingSimple` stays
    # ported-but-unregistered.
    vacuum_old: VacuumOld = VacuumOld()
    # `DuctDiameterRootFind` -- registered as a deliberate island: every `VarPath` it
    # reads/owns is minted and unique to it (`.vacuum.d_duct`/`l1`/`l2`/`l3`/`xmult_i`/
    # `ceff_i`), so it has no producer/consumer edge to any other node registered here
    # today, the same shape `coils.py`'s unregistered `Jcrit*` nodes are flagged with
    # (see this module's own docstring). Registered anyway, on explicit instruction, as
    # a perfectly valid undriven `RootFind` problem sitting in the graph -- see that
    # class's own docstring. `vacuum.py`'s own `DuctFeasibility` (a bare `Feasibility`
    # `ProblemNode`, not a `NodalDeclaration` -- see its docstring for why it cannot be
    # passed to `to_graph()`/listed here the same way) is *not* registered: joining it
    # with this node into one combined block is demonstrated in `test_vacuum.py`, not
    # asserted by this graph.
    duct_diameter_root_find: DuctDiameterRootFind = DuctDiameterRootFind()
