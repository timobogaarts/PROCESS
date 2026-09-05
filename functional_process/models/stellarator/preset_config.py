"""Pure-functional port of `load_stellarator_config` (registry unit #8).

Audit record: `functional_process/_audit/units/models/stellarator/preset_config.md`.
Source: `process/models/stellarator/preset_config.py` (the five machine-preset dict
literals and `load_stellarator_config`).

`load_stellarator_config` is called by `Stellarator.st_new_config()` before anything
else in the stellarator pipeline and fills the ~35 `.stellarator_config.stella_config_*`
fields the whole device design scales from. It is not arithmetic -- it is a table lookup
(`istell` in 1..5) or a file read (`istell == 6`) followed by a reflective
`hasattr`/`setattr` copy onto `StellaratorConfigData`.

**The shape chosen here, and why** (argued at length in the record's "proposed
signature(s)"): the *selection and reading* of a machine config stays outside the graph,
at graph-assembly time, where every other topology-switch decision already lives
(`functional_process/configuration.py`); the *resulting values* enter the graph as one
zero-input node, `StellaratorMachineConfig`, that owns all 34 numeric
`.stellarator_config.*` `VarPath`s. That splits the two halves of PROCESS's function
along the line `_audit/traceability_policy.md` already draws:

- which table/file is read is an `istell` fact that changes nothing about *what* is
  computed, only the numbers -- so it is a static parameter of the node, not a switch
  over node topology and not a port;
- the 34 scalars themselves are inputs *nothing else in the graph produces*, and leaving
  them as unowned boundary inputs is precisely what made the graph unrunnable from a
  cold `DataStructure`: every consumer got the `StellaratorConfigData` dataclass default
  of `0.0`, and the first division by `.tfcoil.n_tf_coils` (itself
  `stella_config_coilspermodule * stella_config_symmetry`) turned the whole downstream
  pipeline into `nan`.

The reflective copy is replaced by a **statically enumerated** field list
(`STELLA_CONFIG_SCALAR_FIELDS`, checked against `StellaratorConfigData` by the unit's
test), which is what makes a fixed, declarable `outputs` tuple possible at all -- the
record's original "not portable as a node" verdict rested on the write set being
knowable only by cross-referencing two files at run time.

**PROCESS behaviour reproduced, not fixed.** A config key that matches no
`StellaratorConfigData` field is dropped silently -- no error, no log. The reference
`stellarator_helias.stella_conf.json` drops three of them
(`number_nu_star`, `D11_star_mono_input`, `nu_star_mono_input`), and *nothing anywhere
in `process/` reads a field of those names*, so a JSON author supplying a neoclassics
mono-energetic transport table gets it discarded in silence. `dropped_config_keys` makes
that visible without changing it; the port drops exactly the same keys.
"""

import json  # noqa: F401

import equinox as eqx
import jax.numpy as jnp  # noqa: F401
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    OutputInto,
)

from functional_process.paths import (
    stellarator_config,
)
from functional_process.stellarator.preset_config import (
    STELLA_CONFIG_DEFAULT,  # noqa: F401
    STELLA_CONFIG_SCALAR_FIELDS,  # noqa: F401
    STELLARATOR_MACHINE_PRESETS,  # noqa: F401
    dropped_config_keys,  # noqa: F401
    machine_config_for_istell,  # noqa: F401
    read_stellarator_config_file,  # noqa: F401
    select_stellarator_config_scalars,
)
from functional_process.vocabulary import (
    HELIAS3,  # noqa: F401
    HELIAS4,  # noqa: F401
    HELIAS5B,  # noqa: F401
    W7X30,  # noqa: F401
    W7X50,  # noqa: F401
)


class StellaratorMachineConfig(ExplicitFunction):
    """cottax node: the selected machine config's 34 scalars, owned by the graph.

    **A node with no inputs**, which is the point: these `VarPath`s used to be unowned
    boundary inputs that every consumer read and nothing produced, so a graph stepped
    from a cold `DataStructure` read `0.0` for all of them. Giving them a producer makes
    the configuration -> design dependency an edge in the DAG instead of a seeding
    convention, and it introduces no cycle -- this node is strictly upstream of
    everything.

    `machine_config` is a **static** field, not a port. Which machine is being designed
    is fixed for a whole solve (no `istell`, and no `stella_config_*`, is an iteration
    variable or a scan variable -- see `configuration.py`'s docstring for the greps), so
    it is graph-assembly-time information by the same argument every topology switch
    already uses. A run that wants a different machine assembles a different graph, which
    is exactly what PROCESS's own `Scan` does per point.
    """

    machine_config: tuple = eqx.field(static=True)

    stella_config_symmetry = OutputInto(stellarator_config)
    stella_config_coilspermodule = OutputInto(stellarator_config)
    stella_config_rmajor_ref = OutputInto(stellarator_config)
    stella_config_rminor_ref = OutputInto(stellarator_config)
    stella_config_coil_rmajor = OutputInto(stellarator_config)
    stella_config_coil_rminor = OutputInto(stellarator_config)
    stella_config_aspect_ref = OutputInto(stellarator_config)
    stella_config_bt_ref = OutputInto(stellarator_config)
    stella_config_wp_area = OutputInto(stellarator_config)
    stella_config_wp_bmax = OutputInto(stellarator_config)
    stella_config_i0 = OutputInto(stellarator_config)
    stella_config_a1 = OutputInto(stellarator_config)
    stella_config_a2 = OutputInto(stellarator_config)
    stella_config_dmin = OutputInto(stellarator_config)
    stella_config_inductance = OutputInto(stellarator_config)
    stella_config_coilsurface = OutputInto(stellarator_config)
    stella_config_coillength = OutputInto(stellarator_config)
    stella_config_max_portsize_width = OutputInto(stellarator_config)
    stella_config_maximal_coil_height = OutputInto(stellarator_config)
    stella_config_min_plasma_coil_distance = OutputInto(stellarator_config)
    stella_config_derivative_min_lcfs_coils_dist = OutputInto(stellarator_config)
    stella_config_vol_plasma = OutputInto(stellarator_config)
    stella_config_plasma_surface = OutputInto(stellarator_config)
    stella_config_wp_ratio = OutputInto(stellarator_config)
    stella_config_max_force_density = OutputInto(stellarator_config)
    stella_config_max_force_density_mnm = OutputInto(stellarator_config)
    stella_config_min_bend_radius = OutputInto(stellarator_config)
    stella_config_epseff = OutputInto(stellarator_config)
    stella_config_max_lateral_force_density = OutputInto(stellarator_config)
    stella_config_max_radial_force_density = OutputInto(stellarator_config)
    stella_config_centering_force_max_mn = OutputInto(stellarator_config)
    stella_config_centering_force_min_mn = OutputInto(stellarator_config)
    stella_config_centering_force_avg_mn = OutputInto(stellarator_config)
    stella_config_neutron_peakfactor = OutputInto(stellarator_config)

    def __call__(self):
        return select_stellarator_config_scalars(dict(self.machine_config))
