"""Pure-functional port of `load_stellarator_config` (registry unit #8).

Audit record: `functional_process/models/stellarator/preset_config.md`. Source:
`process/models/stellarator/preset_config.py` (the five machine-preset dict literals and
`load_stellarator_config`).

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

import json

import equinox as eqx
import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, OutputInto

from functional_process.paths import stellarator_config

STELLA_CONFIG_SCALAR_FIELDS = (
    "stella_config_symmetry",
    "stella_config_coilspermodule",
    "stella_config_rmajor_ref",
    "stella_config_rminor_ref",
    "stella_config_coil_rmajor",
    "stella_config_coil_rminor",
    "stella_config_aspect_ref",
    "stella_config_bt_ref",
    "stella_config_wp_area",
    "stella_config_wp_bmax",
    "stella_config_i0",
    "stella_config_a1",
    "stella_config_a2",
    "stella_config_dmin",
    "stella_config_inductance",
    "stella_config_coilsurface",
    "stella_config_coillength",
    "stella_config_max_portsize_width",
    "stella_config_maximal_coil_height",
    "stella_config_min_plasma_coil_distance",
    "stella_config_derivative_min_lcfs_coils_dist",
    "stella_config_vol_plasma",
    "stella_config_plasma_surface",
    "stella_config_wp_ratio",
    "stella_config_max_force_density",
    "stella_config_max_force_density_mnm",
    "stella_config_min_bend_radius",
    "stella_config_epseff",
    "stella_config_max_lateral_force_density",
    "stella_config_max_radial_force_density",
    "stella_config_centering_force_max_mn",
    "stella_config_centering_force_min_mn",
    "stella_config_centering_force_avg_mn",
    "stella_config_neutron_peakfactor",
)
"""The numeric `StellaratorConfigData` fields, in that dataclass's declaration order.

Static enumeration of what PROCESS's `hasattr`/`setattr` loop discovers by reflection --
the substitution that turns this unit from "output set only knowable at run time" into a
node with a fixed `outputs` tuple. `stella_config_name` is deliberately absent: it is a
`str`, so it is neither a graph variable nor something the harness can compare
numerically. Kept in sync with `process/data_structure/stellarator_configuration.py` by
`test_preset_config.py`'s reference adapter, which reads the fields back off a real
`StellaratorConfigData` by these very names.
"""

_STELLA_CONFIG_PREFIX = "stella_config_"

STELLA_CONFIG_DEFAULT = 0.0
"""`StellaratorConfigData`'s own default for every numeric field.

What a field keeps when the selected machine config has no key for it -- PROCESS never
raises for a missing key, it simply never assigns, so the dataclass default survives.
No config shipped with PROCESS (the five presets or the reference JSON) leaves any of
the 34 unset, so this is a faithfulness detail rather than a live path.
"""


def _by_lowered_key(machine_config):
    """`machine_config` re-keyed the way PROCESS's loop keys it: `key.lower()`.

    PROCESS writes `stella_config_{variable_name.lower()}`, so two keys differing only in
    case collide and the later one wins. Building a dict in iteration order reproduces
    that same last-one-wins resolution rather than inventing a rule.
    """
    return {str(key).lower(): value for key, value in machine_config.items()}


def select_stellarator_config_scalars(machine_config):
    """The 34 numeric `.stellarator_config.*` values a machine config supplies.

    Ports `load_stellarator_config`'s copy loop, inverted: PROCESS iterates the config's
    keys and asks `hasattr` whether a field of that name exists, this iterates the
    (statically known) field names and asks whether the config has a key for it. The two
    agree exactly on the intersection -- which is all either one writes -- and the
    inversion is what makes the write set declarable in advance.

    Parameters
    ----------
    machine_config :
        The selected machine's config mapping: one of `preset_config.py`'s five preset
        dicts, or the parsed contents of an `istell == 6` `stella_conf.json`. Keys are
        matched case-insensitively, as PROCESS matches them.

    Returns
    -------
    :
        One value per entry of `STELLA_CONFIG_SCALAR_FIELDS`, in that order. A field the
        config has no key for takes `STELLA_CONFIG_DEFAULT`, silently, as PROCESS does.
    """
    lowered = _by_lowered_key(machine_config)
    return tuple(
        jnp.asarray(
            float(
                lowered.get(
                    field.removeprefix(_STELLA_CONFIG_PREFIX), STELLA_CONFIG_DEFAULT
                )
            )
        )
        for field in STELLA_CONFIG_SCALAR_FIELDS
    )


def dropped_config_keys(machine_config):
    """The config's keys that no `StellaratorConfigData` field will ever receive.

    Not part of the port's computation and deliberately not an error: PROCESS discards
    these silently and this unit reproduces PROCESS. It exists so the silence is at least
    *inspectable* -- see the module docstring for the three keys the reference
    `stellarator_helias.stella_conf.json` loses this way, none of which any file in
    `process/` reads under any name.

    `name` is excluded from the result: it does land on `StellaratorConfigData`
    (`stella_config_name`), it is simply not a numeric graph variable.

    Parameters
    ----------
    machine_config :
        The machine config mapping, as passed to `select_stellarator_config_scalars`.

    Returns
    -------
    :
        The dropped keys, lower-cased and sorted.
    """
    wanted = {
        field.removeprefix(_STELLA_CONFIG_PREFIX)
        for field in STELLA_CONFIG_SCALAR_FIELDS
    }
    wanted.add("name")
    return tuple(sorted(set(_by_lowered_key(machine_config)) - wanted))


def read_stellarator_config_file(config_file):
    """Parse an `istell == 6` `stella_conf.json`, as a hashable mapping.

    **Assembly-time I/O, deliberately outside any node body.** `istell == 6` is a file
    read (`_audit/traceability_policy.md` § "Non-traceable external calls"); doing it
    once, where the graph is assembled, is what keeps it out of the traced computation
    entirely rather than needing a `pure_callback` or a custom primitive.

    Parameters
    ----------
    config_file :
        Path to the JSON file PROCESS would open, i.e.
        `f"{data.globals.output_prefix}stella_conf.json"`.

    Returns
    -------
    :
        `(key, value)` pairs, suitable as `StellaratorMachineConfig(machine_config=...)`
        -- a tuple rather than a dict so the node stays a hashable `eqx.Module` static
        field. Non-scalar values (the reference file's two arrays) are kept as-is; they
        match no field and are dropped downstream exactly as PROCESS drops them.
    """
    with open(config_file) as stream:
        return tuple(json.load(stream).items())


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
