"""Pure-functional port of `Stellarator.st_strc` (chunk 1D of unit #1).

Audit record:
`functional_process/_audit/units/models/stellarator/structure.md`. The
record proposes splitting the source method in two: the real structural masses (this
module's `calculate_structure_masses`), and the "previous scaling law, kept for
comparison, not fully trusted" reporting value
(`calculate_intercoil_mass_scaling_reference`), which never feeds
`aintmass`/`clgsmass`/`coldmass` and is only ever printed. Splitting it out drops one
otherwise-unused argument (`e_tf_magnetic_stored_total_gj`) from the real function's
signature.

`fncmass` and `gsmass` are not ported: both are unconditional literal `0.0` in the
source (open question 1 in the record -- whether a constant "producer" should be a graph
node at all is a policy question, not resolved here, so neither is wrapped in a
`ImplementedFunction`).

`StructureMasses` below is the `cottax` node -- a thin `ExplicitFunction` declaration
wrapping `calculate_structure_masses` unchanged. See `_audit/schema.md`'s "cottax node"
section for why the pytree-namespace surface
(`cottax.interfaces.pytree_namespace_module`) rather than a hand-built `ImplementedFunction`:
`Output`/`FromExactly` read like the PROCESS `data.<area>.<field>` path they name, instead of a
`VarPath` built from a string one node at a time.
"""

from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    OutputInto,
)

from functional_process.models.safe_math import (
    safe_pow,  # noqa: F401
)
from functional_process.paths import (
    fwbs,
    physics,
    stellarator,
    stellarator_config,
    structure,
    tfcoil,
)
from functional_process.stellarator.structure import (
    calculate_intercoil_mass_scaling_reference,  # noqa: F401
    calculate_structure_masses,
)


class StructureMasses(ExplicitFunction):
    """cottax node: `calculate_structure_masses`, unchanged, with its ports declared.

    `calculate_intercoil_mass_scaling_reference` has no node: it feeds nothing else in
    the graph (reporting-only, per the module docstring), so there is nowhere for it to
    sit as a node with no reader.
    """

    aintmass = OutputInto(structure)
    clgsmass = OutputInto(structure)
    coldmass = OutputInto(structure)

    def __call__(
        self,
        stella_config_coilsurface=From(stellarator_config),
        f_st_rmajor=From(stellarator),
        r_coil_minor=From(stellarator),
        stella_config_coil_rminor=From(stellarator_config),
        dx_tf_inboard_out_toroidal=From(tfcoil),
        len_tf_coil=From(tfcoil),
        n_tf_coils=From(tfcoil),
        b_plasma_toroidal_on_axis=From(physics),
        den_steel=From(fwbs),
        m_tf_coils_total=From(tfcoil),
        dewmkg=From(fwbs),
    ):
        return calculate_structure_masses(
            stella_config_coilsurface,
            f_st_rmajor,
            r_coil_minor,
            stella_config_coil_rminor,
            dx_tf_inboard_out_toroidal,
            len_tf_coil,
            n_tf_coils,
            b_plasma_toroidal_on_axis,
            den_steel,
            m_tf_coils_total,
            dewmkg,
        )
