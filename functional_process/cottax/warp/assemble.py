"""`_assemble(config)`: a configuration's `IN.DAT` to its SAND `Drive`.

Split out of `leaves.py` so the **jaxpr backend imports none of the resolver**. The
assembly (PROCESS run -> `machine_from_indat` -> `graph_for` -> `mda_env` ->
`sand.assemble` -> `sand.sand_schedule` -> `sand.sand_shape`) is shared by both
back-ends and has nothing to do with how a node becomes a `@wp.func`; `leaves.py`
re-exports it so its own callers are unaffected.
"""
import jax

jax.config.update("jax_enable_x64", True)

from functional_process.cottax import sand
from functional_process.cottax.indat import REFERENCE_INPUT_FILE, graph_for, machine_from_indat
from functional_process.cottax.run_cold_matrix import _resolve
from functional_process.cottax.sand_harness import assemble as sand_assemble
from functional_process.cottax.sand_harness import mda_env, reference_run


def _assemble(config: str):
    """`(drive, report)` for `config` -- the bare stem of a
    `tests/regression/input_files/<config>.IN.DAT`."""
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
    return shape["drive"], report
