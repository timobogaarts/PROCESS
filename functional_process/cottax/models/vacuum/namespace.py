"""The vacuum subsystem's namespace."""

from cottax.interfaces.pytree_namespace_module import ModelNamespace

from functional_process.cottax.models.vacuum.vacuum import VacuumOld


class Vacuum(ModelNamespace):
    """Vacuum pumping and the duct sizing problem."""

    # unit #16, vacuum.py -- `"old"` branch only, matching PROCESS's own default
    # (`.vacuum.i_vacuum_pumping = "old"`, `vacuum_variables.py:18`). Not gated by a
    # `Switch`: the `"simple"` alternative (`VacuumPumpingSimple`) owns a disjoint
    # output, so this switch fails `check_arms_are_exclusive` -- see
    # `TOPOLOGY_SWITCHES`'s docstring above. `VacuumPumpingSimple` stays
    # ported-but-unregistered.
    vacuum_old: VacuumOld = VacuumOld()
    # `DuctDiameterRootFind` is **not registered**, and that is the point of it.
    # Every `VarPath` it reads or owns is minted and unique to it
    # (`.vacuum.d_duct`/`l1`/`l2`/`l3`/`xmult_i`/`ceff_i`), so registering it put an
    # island in every machine's graph: a node with no producer or consumer edge to
    # anything else, answering a question no other node asks. The duct sizing the
    # machine actually uses is `VacuumOld`'s -- `solve_duct_geometry` run eagerly --
    # and that has not changed.
    #
    # It was registered for a while on explicit instruction, to have a valid undriven
    # `RootFind` sitting in the graph. The cost was paid twice over: `evaluate`'s
    # `EXCLUDED_NODE_NAMES` existed solely to delete it again before any architecture
    # was assembled, so the declared graph and the graph that ran disagreed by one
    # solve for a reason no graph operation accounts for. A demonstration that the
    # shape is real and drivable belongs where it is demonstrated -- the class's own
    # docstring in `vacuum.py`, and `test_vacuum.py`'s test-only driver -- not in
    # every machine a user assembles.
