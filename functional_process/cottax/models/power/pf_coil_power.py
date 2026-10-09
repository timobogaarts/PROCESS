"""Pure-functional port of the PF-coil power-supply sub-unit of
`process/models/power.py` (registry unit #14, chunk D).
"""

import jax.numpy as jnp
from cottax.assemblies.jax import V

from functional_process.cottax.models.pfcoil import (
    N_COILS_IN_GROUP,
    N_PF_GROUPS,
    PLASMA_INDEX,
    REFERENCE_TOPOLOGY,
    SPHERICAL_TOKAMAK_TOPOLOGY,
    PFCoilTopology,
)
from functional_process.cottax.paths import (
    pf_power,
)
from functional_process.models.power.pf_coil_power import (
    COILS_IN_GROUP_WITH_CS,
    GROUP_CIRCUIT_INDEX,
    MIN_INTERVAL_S,
    N_PF_ACTIVE_INTERVALS,
    N_PF_ACTIVE_POINTS,
    N_PF_CS_PLASMA_CIRCUITS,
    N_PF_GROUPS_WITH_CS,
    PF_BUS_CURRENT_DENSITY_A_PER_CM2,
    PFCKTS_SPARE_CIRCUITS,
    POLOIDAL_POWER_SENTINEL_W,
    VPFSKV_KV,
    calculate_pf_coil_power_supplies,
    calculate_pf_coil_power_supplies_no_central_solenoid,
    calculate_pf_coil_power_supplies_reference,
    coils_in_group_with_cs,
    group_circuit_index,
)

# Step 2 of `_audit/formulas_split.md` moved the bodies below to
# `functional_process.models.power.pf_coil_power`; these names are imported above purely to
# keep every public name this module resolved before the move still resolving now
# (`models/**` modules must not lose any name -- see the split's invariant), not
# because the declaration below reads them itself.
__all__ = [
    "COILS_IN_GROUP_WITH_CS",
    "GROUP_CIRCUIT_INDEX",
    "MIN_INTERVAL_S",
    "N_COILS_IN_GROUP",
    "N_PF_ACTIVE_INTERVALS",
    "N_PF_ACTIVE_POINTS",
    "N_PF_CS_PLASMA_CIRCUITS",
    "N_PF_GROUPS",
    "N_PF_GROUPS_WITH_CS",
    "PFCKTS_SPARE_CIRCUITS",
    "PF_BUS_CURRENT_DENSITY_A_PER_CM2",
    "PLASMA_INDEX",
    "POLOIDAL_POWER_SENTINEL_W",
    "REFERENCE_TOPOLOGY",
    "SPHERICAL_TOKAMAK_TOPOLOGY",
    "VPFSKV_KV",
    "PFCoilTopology",
    "PfCoilPowerSuppliesNoCentralSolenoid",
    "PfCoilPowerSuppliesReference",
    "calculate_pf_coil_power_supplies",
    "calculate_pf_coil_power_supplies_no_central_solenoid",
    "calculate_pf_coil_power_supplies_reference",
    "coils_in_group_with_cs",
    "group_circuit_index",
    "jnp",
]


# The PfCoilPowerSupplies family:
# cottax node: `.power.pf_coil_power` -- `Power.pfpwr`, eleven owned fields.
#
# Bodiless family base: `topology` used to be a static field (`PFCoilTopology`,
# defaulting to `REFERENCE_TOPOLOGY`) -- `_pf_coil_topology` only ever hands this
# slot one of two values, so it is now two occupants instead, following the arm
# pattern `models/pfcoil/`'s own topology-bound nodes already established (e.g.
# `PFCoilEquilibriumCurrents`/`PFCoilEquilibriumCurrentsNoCentralSolenoid`). The
# read set is identical between the two arms -- only the loop bounds baked into
# `topology` differ -- so both share this base's reads and outputs and add only
# `fn`.


# `.build.iohcl != 0` -- `REFERENCE_TOPOLOGY`, a machine with a central
# solenoid.
PfCoilPowerSuppliesReference = pf_power.function(
    calculate_pf_coil_power_supplies_reference,
    "srcktpm",
    "poloidalpower",
    "ensxpfm",
    "peakpoloidalpower",
    V.heat_transport.peakmva,
    "vpfskv",
    "pfckts",
    "spfbusl",
    "acptmax",
    "spsmva",
    V.pf_coil.p_pf_electric_supplies_mw,
)


# `.build.iohcl == 0` -- `SPHERICAL_TOKAMAK_TOPOLOGY`, no central solenoid.
PfCoilPowerSuppliesNoCentralSolenoid = pf_power.function(
    calculate_pf_coil_power_supplies_no_central_solenoid,
    "srcktpm",
    "poloidalpower",
    "ensxpfm",
    "peakpoloidalpower",
    V.heat_transport.peakmva,
    "vpfskv",
    "pfckts",
    "spfbusl",
    "acptmax",
    "spsmva",
    V.pf_coil.p_pf_electric_supplies_mw,
)
