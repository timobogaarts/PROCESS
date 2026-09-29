"""Pure-functional port of the PF-coil power-supply sub-unit of
`process/models/power.py` (registry unit #14, chunk D).
"""

import jax.numpy as jnp
from cottax.interfaces import V

from functional_process.cottax.models.pfcoil import (
    N_COILS_IN_GROUP,
    N_PF_GROUPS,
    PLASMA_INDEX,
    REFERENCE_TOPOLOGY,
    SPHERICAL_TOKAMAK_TOPOLOGY,
    PFCoilTopology,
)
from functional_process.cottax.paths import (
    pf_coil,
    pf_power,
    physics,
    times,
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
    "PfCoilPowerSupplies",
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
    reads={
        "rmajor": physics,
        "c_pf_coil_turn_peak_input": pf_coil,
        "rhopfbus": pf_coil,
        "rho_pf_coil": pf_coil,
        "r_pf_coil_middle": pf_coil,
        "j_pf_coil_wp_peak": pf_coil,
        "f_a_pf_coil_void": pf_coil,
        "c_pf_cs_coils_peak_ma": pf_coil,
        "c_pf_cs_coil_pulse_end_ma": pf_coil,
        "n_pf_coil_turns": pf_coil,
        "c_pf_coil_turn": pf_coil,
        "ind_pf_cs_plasma_mutual": pf_coil,
        "etapsu": pf_coil,
        "p_plasma_ohmic_mw": physics,
        "t_plant_pulse_coil_precharge": times,
        "t_plant_pulse_plasma_current_ramp_up": times,
        "t_plant_pulse_fusion_ramp": times,
        "t_plant_pulse_burn": times,
        "t_plant_pulse_plasma_current_ramp_down": times,
    },
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
    reads={
        "rmajor": physics,
        "c_pf_coil_turn_peak_input": pf_coil,
        "rhopfbus": pf_coil,
        "rho_pf_coil": pf_coil,
        "r_pf_coil_middle": pf_coil,
        "j_pf_coil_wp_peak": pf_coil,
        "f_a_pf_coil_void": pf_coil,
        "c_pf_cs_coils_peak_ma": pf_coil,
        "c_pf_cs_coil_pulse_end_ma": pf_coil,
        "n_pf_coil_turns": pf_coil,
        "c_pf_coil_turn": pf_coil,
        "ind_pf_cs_plasma_mutual": pf_coil,
        "etapsu": pf_coil,
        "p_plasma_ohmic_mw": physics,
        "t_plant_pulse_coil_precharge": times,
        "t_plant_pulse_plasma_current_ramp_up": times,
        "t_plant_pulse_fusion_ramp": times,
        "t_plant_pulse_burn": times,
        "t_plant_pulse_plasma_current_ramp_down": times,
    },
)
