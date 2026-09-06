"""`_assemble(config)`: a configuration's `IN.DAT` to its SAND `Drive`.

Split out of the old `leaves.py` so the **jaxpr backend imported none of the resolver**,
and kept when the resolver went: the assembly (PROCESS run -> `machine_from_indat` ->
`graph_for` -> `mda_env` -> `sand.assemble` -> `sand.sand_schedule` ->
`sand.sand_shape`) has nothing to do with how a node becomes a `@wp.func`, and it is
what the whole package starts from.
"""
import jax

jax.config.update("jax_enable_x64", True)

from functional_process.cottax import sand
from functional_process.cottax.indat import REFERENCE_INPUT_FILE, graph_for, machine_from_indat
from functional_process.cottax.run_cold_matrix import _resolve
from functional_process.cottax.sand_harness import assemble as sand_assemble
from functional_process.cottax.sand_harness import mda_env, reference_run


def _assemble(config: str):
    """`(drive, report, env)` for `config` -- the bare stem of a
    `tests/regression/input_files/<config>.IN.DAT`.

    `env` is the completed MDA run's own output env (`VarPath -> value`), which
    `mda_env` computes on the way to the `Drive` and this used to throw away. It is the
    only source of a real value for a context variable that has neither a native answer
    nor a `DataStructure` field -- the 201-point profile grid among them -- so the
    jaxpr backend traces at the shapes the graph actually produces rather than at a
    scalar placeholder (`jaxpr_backend.node_values`).
    """
    path = _resolve(f"tests/regression/input_files/{config}.IN.DAT")
    is_reference = path == _resolve(REFERENCE_INPUT_FILE)
    reference = reference_run(str(path))
    machine = machine_from_indat(str(path))
    machine_graph = None if is_reference else graph_for(machine)
    driven, env = mda_env(reference, graph=machine_graph)
    cold = reference.cold
    switch_values = (
        None
        if is_reference
        else sand.switch_values_for(cold, reference.icc, reference.i_figure_merit)
    )
    combined, report = sand_assemble(reference, driven, env, switch_values=switch_values)
    schedule = sand.sand_schedule(combined, None, bounds=reference.bounds)
    shape = sand.sand_shape(schedule)
    return shape["drive"], report, env
