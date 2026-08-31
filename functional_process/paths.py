"""Where the port's variables live: `data.<area>.<field>`, the way PROCESS spells it.

PROCESS keeps every physics and engineering variable at `data.<area>.<name>`
(`process/core/model.py`'s `DataStructure`, one sub-dataclass per
`process/data_structure/*_variables.py`). This module exposes that same address as
something a declaration can name:

    from functional_process.paths import fwbs, heat_transport, impurity_radiation, tfcoil

    class Cryo(ExplicitFunction):
        helpow = OutputInto(heat_transport)           # -> .heat_transport.helpow

        def __call__(self,
            tfcryoarea = From(tfcoil),                # -> .tfcoil.tfcryoarea
            qnuc       = From(fwbs),                  # -> .fwbs.qnuc
            helium     = FromExactly(impurity_radiation.f_nd_impurity_electron_array[2]),
        ): ...

**Two calls, and which one you call is the whole difference.**

- `From(area)` / `OutputInto(area)` take an **area** and complete it with the name being
  declared -- the parameter's, or the class attribute's. The parameter name *is* the
  field name; a body wanting a shorter local name renames in the body. This is the
  ordinary way and covers almost everything.
- `FromExactly(place)` / `Output(place)` take a **whole place** and use it verbatim.
  This is the escape hatch, for the places no declaring name can spell -- one element of
  `f_nd_impurity_electron_array`, a dict key. *Exactly* because nothing is appended,
  which is the whole difference from `From`; the shared stem is so that every read reads
  as one, with the suffix saying which rule applies.

They are not interchangeable spellings, and cottax enforces that in both directions:
`From`/`OutputInto` refuse a path, and `FromExactly`/`Output` refuse a bare area. That
second guard matters more than it looks -- `FromExactly(physics)` on `rmajor` would
complete to exactly the port `From(physics)` declares, so the two would produce
*identical* ports and no graph, value test or gradient test could ever tell them apart.

An area never yields another area. That is what makes `physics` and `physics.q` different
kinds of thing rather than two lengths of the same thing: the moment there is a second
dot, the path is written out in full and nothing is appended to it.

**The escape hatch does not need a lambda.** A recorded chain off an area is a whole
place already, so `FromExactly(impurity_radiation.f_nd_impurity_electron_array[2])` is
the address it reads as. A callable (`FromExactly(lambda root: root.physics.q)`) is still
accepted and is not deprecated -- cottax supplies its root, so it cannot be built
against the wrong data structure -- but the recorder form is built off `data` below,
which refuses a misspelled area, so it carries the same guarantee with less ceremony.

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

import difflib

from cottax.interfaces.pytree_namespace_module import Area
from jax.tree_util import GetAttrKey

# `AREAS` -- every area PROCESS has, 36 of them. Was
# `tuple(f.name for f in dataclasses.fields(DataStructure))`, read live off PROCESS; the
# list is vocabulary, not physics, so §23.2 vendors it and
# `tests/functional_process/test_vocabulary.py` asserts the vendored tuple still equals
# that expression, order included. Re-exported from here because every caller in the port
# imports it from `paths`.
from functional_process.vocabulary import AREAS


class _Root:
    """The areas of a `DataStructure`, as something whole paths hang off.

    Was `cottax.interfaces.pytree_namespace_module.Root` until cottax deleted it (no
    replacement anywhere in that package; its own suite green at the time, so the removal
    was deliberate rather than in-flight). Reimplemented here rather than re-imported
    because it was never really cottax's: the area list comes from PROCESS's own
    `DataStructure`, and every area name it accepts or rejects is a PROCESS fact. What is
    cottax's is `Area`, which this still returns, so nothing downstream changes -- a
    `From`/`FromExactly` cannot tell the difference.

    The name check is the part worth keeping, and it earns its place for a reason
    specific to this port: a misspelled area is otherwise a boundary input nothing ever
    writes, and **no value test can see it**, because the field reads as its
    default. It is caught here, at declaration time, or not at all.
    """

    __slots__ = ("_names",)

    def __init__(self, names):
        object.__setattr__(self, "_names", frozenset(names))

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(
                f"{name!r}: a path is built from ordinary attribute access, not a "
                f"private or dunder name"
            )
        if name not in self._names:
            close = difflib.get_close_matches(name, sorted(self._names), n=3)
            hint = f" Did you mean {', '.join(close)}?" if close else ""
            raise AttributeError(
                f"{name!r} is not an area of this data structure.{hint}"
            )
        return Area((GetAttrKey(name),))

    def __repr__(self):
        return f"<PROCESS data structure: {len(self._names)} areas>"


data = _Root(AREAS)
"""The whole namespace, for the escape hatch:
`FromExactly(data.impurity_radiation.arr[2])`.

Also what makes a misspelled area fail at declaration -- `data.physcis` raises with a
suggestion rather than quietly naming a place nothing ever writes.
"""

globals().update({name: getattr(data, name) for name in AREAS})

__all__ = ["AREAS", "data", *AREAS]
