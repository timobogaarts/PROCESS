"""Where the port's variables live: `data.<area>.<field>`, the way PROCESS spells it.

PROCESS keeps every physics and engineering variable at `data.<area>.<name>`
(`process/core/model.py`'s `DataStructure`, one sub-dataclass per
`process/data_structure/*_variables.py`). This module exposes that same address as
something a declaration can name:

    from functional_process.paths import data

    class Cryo(ExplicitFunction):
        helpow = Output(data.heat_transport)          # -> .heat_transport.helpow

        def __call__(self,
            tfcryoarea = data.tfcoil,                 # -> .tfcoil.tfcryoarea
            qnuc       = data.fwbs,                   # -> .fwbs.qnuc
            q95        = data.physics.q,              # -> .physics.q  (spelling differs)
        ): ...

**Three forms, one rule each.** A bare **area** (`data.tfcoil`) is completed by the name
being declared -- the parameter's, or the class attribute's. A **path** off an area
(`data.physics.q`) is used verbatim. A **callable** (`Input(lambda s: s.physics.q)`) is
also verbatim and stays available for everything; cottax supplies its root, so it is the
one form that cannot be built against the wrong data structure.

An area never yields another area. That is what makes `data.physics` and `data.physics.q`
unambiguous rather than a convention to remember: the moment there is a second dot, the
path is written out in full and nothing is appended to it.

The `Input(...)` wrapper is optional on a **parameter** -- `qnuc=data.fwbs` is enough,
because cottax requires every parameter to carry a read, so that slot has no competing
meaning. It is **not** optional on an `Output`, which is found by scanning class
attributes, where a bare area bound as an alias could not be told from a declared output.

`data` knows the real area names (taken from `DataStructure` itself, so it cannot drift),
and refuses anything else at declaration time with a suggestion -- `data.physcis` raises
rather than silently naming a place that is never read.

The areas, and what they hold -- PROCESS's own names, which are not always self-evident:

| area | |
|---|---|
| `blanket`, `fwbs` | blanket; first wall, blanket and shield together |
| `first_wall`, `divertor` | first wall; divertor |
| `ccfe_hcpb`, `dcll` | the two blanket models (CCFE-HCPB, dual-coolant lithium-lead) |
| `build`, `buildings` | radial/vertical machine build; site buildings |
| `tfcoil`, `superconducting_tfcoil`, `pf_coil`, `pf_power`, `cs_fatigue`, `rebco` | magnets |
| `physics`, `impurity_radiation`, `neoclassics`, `reinke` | plasma |
| `current_drive`, `heat_transport`, `power`, `primary_pumping`, `vacuum` | plant systems |
| `costs`, `costs_2015` | the two costing models |
| `structure`, `times`, `pulse`, `water_use` | support structure; timing; water |
| `stellarator`, `stellarator_config`, `ife` | device-type specifics |
| `numerics`, `constraints`, `scan`, `globals` | the solve itself, not the machine |
"""

import dataclasses

from cottax.interfaces.pytree_namespace_module import Root

from process.core.model import DataStructure

AREAS = tuple(f.name for f in dataclasses.fields(DataStructure))
"""Every area PROCESS has, straight from `DataStructure` -- 36 of them."""

data = Root(AREAS)
"""The whole namespace, for the escape hatch: `Input(data.impurity_radiation.arr[2])`.

Also what makes a misspelled area fail at declaration -- `data.physcis` raises with a
suggestion rather than quietly naming a place nothing ever writes.
"""

globals().update({name: getattr(data, name) for name in AREAS})

__all__ = ["AREAS", "data", *AREAS]
