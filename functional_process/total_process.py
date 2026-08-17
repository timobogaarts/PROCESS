"""The running graph assembly of every ported stellarator unit.

Imports each ported unit's `cottax` node declaration and assembles them into one
`Graph` via `to_graph`. Run directly for a smoke check (builds the graph, prints its
node/variable count); `render_xdsm.py` imports `GRAPH` from here to draw it.

This is the whole graph **as it currently exists**, not a claim that the stellarator MDA
is assembled: most nodes here are still islands with unowned (external) reads, since
their producers haven't been ported yet, and the couplings that do exist haven't been
checked for cycles (no `Blocking` has been run over this). It exists so there is always
one place the next ported unit joins, and one place to point a visual inspection at. See
`_audit/unit_registry.md`'s "Ported so far" for what is and isn't in it.

**Switch-selected alternatives, resolved here by PROCESS's own default value** (per
`naming_convention.md`'s "switches are not ports" -- this is graph-assembly-time
selection, not a design decision made by any one node):
- `EcrhDensityLimit(i_plasma_pedestal=0)` -- the only value PROCESS itself exercises
  (see `density_limits.py`); not really a switch pick, `i_plasma_pedestal != 0` has no
  formula at all.
- `heating.EcrhHeating` over `heating.LowhybHeating` -- both mint the same four
  `.current_drive.*` fields (mutually exclusive per `isthtr`); `isthtr`'s PROCESS
  default is `1` (ECRH), so that's the one wired in. `LowhybHeating` exists, ported and
  tested, just not registered here.
- `build.AFwTotalWithPowerflow` over `build.AFwTotalNoPowerflow` -- both mint
  `.first_wall.a_fw_total` (mutually exclusive per `ipowerflow`); PROCESS's default is
  `ipowerflow = 1` ("with"), so that's the one wired in. `AFwTotalNoPowerflow` exists,
  ported and tested, just not registered here.

Not included here despite being ported: `neoclassics.py`'s 10 array-argument functions
(`calculate_kt` etc.) -- ported but flagged by their own audit as **not test-validated**
(`Tier1Contract`'s gradient check only supports scalar arguments; a harness gap, not a
property of the functions), so wrapping them as nodes here would put untested code in
the graph. `coils/coils.py` -- no node written; its own audit found no basis yet for the
wiring a node would assert. `coils/calculate.py`'s `winding_pack_total_size` and
`st_coil` itself -- real tier-2/tier-3 units, not self-contained, still audit-only.
"""

from cottax.interfaces.pytree_namespace_module import to_graph

from functional_process.models.stellarator.build import (
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

GRAPH = to_graph(
    # unit #1 chunks
    SudoDensityLimit,
    EcrhDensityLimit(i_plasma_pedestal=0),
    StructureMasses,
    ScTfCoilNuclearHeating,
    # unit #2, build.py
    BlktmodelBlanketThickness,
    Build,
    AFwTotalWithPowerflow,
    # unit #4, divertor.py
    Divertor,
    # unit #5, heating.py
    EcrhHeating,
    InjectedPowerTotal,
    BeamCurrent,
    FusionGain,
    # unit #6, initialization.py
    PulseDurations,
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

if __name__ == "__main__":
    n_vars = sum(len(node.inputs) + len(node.outputs) for node in GRAPH.definitions.values())
    print(f"{len(GRAPH.definitions)} nodes, {n_vars} ports (inputs + outputs, unmerged)")
    for name, node in GRAPH.definitions.items():
        print(f"  {name.path_str()}: {len(node.inputs)} in, {len(node.outputs)} out")
