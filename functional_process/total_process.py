"""The running graph assembly of every ported stellarator unit.

Imports each ported unit's `cottax` node declaration and assembles them into one
`Graph` via `to_graph`. Run directly for a smoke check (builds the graph, prints its
node/variable count); `render_xdsm.py` imports `GRAPH` from here to draw it.

This is the whole graph **as it currently exists**, not a claim that the stellarator MDA
is assembled: most nodes here are still islands with unowned (external) reads, since
their producers haven't been ported yet. It exists so there is always one place the next
ported unit joins, and one place to point a visual inspection at. See
`_audit/unit_registry.md`'s "Ported so far" for what is and isn't in it.

**There is no single graph.** A topology-changing switch selects which nodes exist, so
what this module exports is `build_graph(configuration)`; `GRAPH` is the one PROCESS's
own switch defaults produce, kept as a module-level name because `render_xdsm.py` and the
smoke check want a default to point at. `TOPOLOGY_SWITCHES` below enumerates the arms,
and `configuration.py` explains why assembly time is the only correct place to resolve
them (short version: no switch in PROCESS is ever an iteration variable or a scan
variable, so no switch can change between two evaluations of one assembled graph).

`EcrhDensityLimit(i_plasma_pedestal=0)` is deliberately *not* a `Switch` here. It is
`naming_convention.md`'s other category -- a formula-changing switch kept as a static
kwarg on one node's `fn` -- because `i_plasma_pedestal != 0` has no formula at all in
`density_limits.py` and no node's existence depends on it.

Not included here despite being ported: `coils/coils.py` -- no node written; its own
audit found no basis yet for the wiring a node would assert. `coils/calculate.py`'s
`winding_pack_total_size` and `st_coil` itself -- real tier-2/tier-3 units, not
self-contained, still audit-only.
"""

from functional_process.configuration import (
    Alternative,
    Configuration,
    Switch,
    build_graph,
)
from functional_process.models.physics.plasma_profiles import ProfileFactors
from functional_process.models.stellarator.build import (
    AFwTotalNoPowerflow,
    AFwTotalWithPowerflow,
    BlktmodelBlanketThickness,
    Build,
)
from functional_process.models.stellarator.coils.calculate import (
    CoilCasing,
    CoilCoilToroidalGap,
    CoilCrossSectionalArea,
    CoilCurrent,
    CoilHalfWidths,
    CoilRadialThickness,
    CoilsSummaryVariables,
    CoilToroidalThickness,
    HorizontalPorts,
    PlasmaFacingCoilArea,
    StoredMagneticEnergy,
    VerticalPorts,
    WindingPackGeometry,
)
from functional_process.models.stellarator.coils.forces import (
    MaxForceDensity,
    MaximumStress,
)
from functional_process.models.stellarator.coils.mass import CoilsMass
from functional_process.models.stellarator.coils.quench import QuenchProtection
from functional_process.models.stellarator.density_limits import (
    EcrhDensityLimit,
    SudoDensityLimit,
)
from functional_process.models.stellarator.divertor import Divertor
from functional_process.models.stellarator.heating import (
    BeamCurrent,
    EcrhHeating,
    FusionGain,
    InjectedPowerTotal,
    LowhybHeating,
)
from functional_process.models.stellarator.initialization import PulseDurations
from functional_process.models.stellarator.neoclassics import (
    EffectiveThermalDiffusivity,
    ProfileValues,
)
from functional_process.models.stellarator.stellarator_D_structure import (
    StructureMasses,
)
from functional_process.models.stellarator.stellarator_F_tf_nuclear_heating import (
    ScTfCoilNuclearHeating,
)

TOPOLOGY_SWITCHES = (
    Switch(
        path=".stellarator.isthtr",
        default=1,  # `stellarator_variables.py:87`
        alternatives=(
            Alternative(value=1, declarations=(EcrhHeating,)),
            Alternative(value=2, declarations=(LowhybHeating,)),
            Alternative(
                value=3,
                unported=(
                    "the NBI branch of `st_heat` calls `current_drive.culnbi()`, a "
                    "model that is not audited yet (registry unit #5)"
                ),
            ),
        ),
    ),
    Switch(
        path=".heat_transport.ipowerflow",
        default=1,  # `heat_transport_variables.py:94`
        alternatives=(
            Alternative(value=0, declarations=(AFwTotalNoPowerflow,)),
            Alternative(value=1, declarations=(AFwTotalWithPowerflow,)),
        ),
    ),
)
"""Switches whose value decides which nodes exist. See `configuration.py`.

Grows as ported units bring more arms: `build.py`'s `blktmodel`, the blanket
CCFE-HCPB/DCLL split and `costs`' `i_cost_model` are all known to belong here
(`core/solver/switches.md`), but none of their arms is ported yet, so declaring them now
would be a switch with one arm and no choice to make.
"""

COMMON = (
    # unit #1 chunks
    SudoDensityLimit,
    EcrhDensityLimit(i_plasma_pedestal=0),  # static kwarg, not a topology switch
    StructureMasses,
    ScTfCoilNuclearHeating,
    # unit #2, build.py
    BlktmodelBlanketThickness,
    Build,
    # unit #4, divertor.py
    Divertor,
    # unit #5, heating.py
    InjectedPowerTotal,
    BeamCurrent,
    FusionGain,
    # unit #6, initialization.py
    PulseDurations,
    # unit #12, physics/plasma_profiles.py
    ProfileFactors,
    # unit #7, neoclassics.py (scalar-argument functions only, see module docstring)
    ProfileValues,
    EffectiveThermalDiffusivity,
    # unit #9, coils/calculate.py
    CoilToroidalThickness,
    CoilRadialThickness,
    CoilCrossSectionalArea,
    CoilHalfWidths,
    PlasmaFacingCoilArea,
    CoilCoilToroidalGap,
    CoilsSummaryVariables,
    StoredMagneticEnergy,
    WindingPackGeometry,
    CoilCurrent,
    CoilCasing,
    VerticalPorts,
    HorizontalPorts,
    # unit #12, coils/mass.py
    CoilsMass,
    # unit #11, coils/forces.py
    MaxForceDensity,
    MaximumStress,
    # unit #14, coils/quench.py
    QuenchProtection,
)
"""Nodes present in every configuration -- everything no topology switch gates."""


def graph_for(configuration=None):
    """The assembled graph for one configuration; PROCESS's defaults if unstated."""
    return build_graph(configuration or Configuration(), COMMON, TOPOLOGY_SWITCHES)


GRAPH = graph_for()
"""The default configuration's graph: `isthtr = 1` (ECRH), `ipowerflow = 1`."""

if __name__ == "__main__":
    n_vars = sum(len(node.inputs) + len(node.outputs) for node in GRAPH.definitions.values())
    print(f"{len(GRAPH.definitions)} nodes, {n_vars} ports (inputs + outputs, unmerged)")
    for name, node in GRAPH.definitions.items():
        print(f"  {name.path_str()}: {len(node.inputs)} in, {len(node.outputs)} out")

    print("\ncycles, per configuration:")
    for path, value in ((None, None), (".heat_transport.ipowerflow", 0)):
        configuration = Configuration({path: value} if path else {})
        cycles = graph_for(configuration).cycles
        label = f"{path} = {value}" if path else "PROCESS defaults"
        print(f"  {label}: {[[n.path_str() for n in c] for c in cycles] or 'acyclic'}")
