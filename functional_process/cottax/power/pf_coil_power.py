"""Pure-functional port of the PF-coil power-supply sub-unit of
`process/models/power.py` (registry unit #14, chunk D).
"""

import equinox as eqx
import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.cottax.pfcoil import (
    N_COILS_IN_GROUP,
    N_PF_GROUPS,
    PLASMA_INDEX,
    REFERENCE_TOPOLOGY,
    PFCoilTopology,
)
from functional_process.cottax.paths import (
    heat_transport,
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
    "VPFSKV_KV",
    "PFCoilTopology",
    "PfCoilPowerSupplies",
    "calculate_pf_coil_power_supplies",
    "coils_in_group_with_cs",
    "group_circuit_index",
    "jnp",
]


class PfCoilPowerSupplies(ExplicitFunction):
    """cottax node: `.power.pf_coil_power` -- `Power.pfpwr`, eleven owned fields."""

    topology: PFCoilTopology = eqx.field(static=True, default=REFERENCE_TOPOLOGY)
    """Static, and the same object the PF coil package's own nodes carry: `pfpwr`'s four
    loop bounds are the coil topology's (module docstring), so a machine with no central
    solenoid loops over its groups and not over a fifth that does not exist.
    """

    srcktpm = OutputInto(pf_power)
    poloidalpower = OutputInto(pf_power)
    ensxpfm = OutputInto(pf_power)
    peakpoloidalpower = OutputInto(pf_power)
    peakmva = OutputInto(heat_transport)
    vpfskv = OutputInto(pf_power)
    pfckts = OutputInto(pf_power)
    spfbusl = OutputInto(pf_power)
    acptmax = OutputInto(pf_power)
    spsmva = OutputInto(pf_power)
    p_pf_electric_supplies_mw = OutputInto(pf_coil)

    def __call__(
        self,
        rmajor=From(physics),
        c_pf_coil_turn_peak_input=From(pf_coil),
        rhopfbus=From(pf_coil),
        rho_pf_coil=From(pf_coil),
        r_pf_coil_middle=From(pf_coil),
        j_pf_coil_wp_peak=From(pf_coil),
        f_a_pf_coil_void=From(pf_coil),
        c_pf_cs_coils_peak_ma=From(pf_coil),
        c_pf_cs_coil_pulse_end_ma=From(pf_coil),
        n_pf_coil_turns=From(pf_coil),
        c_pf_coil_turn=From(pf_coil),
        ind_pf_cs_plasma_mutual=From(pf_coil),
        f_p_pf_energy_store_loss=From(pf_power),
        f_p_pf_psu_loss=From(pf_power),
        etapsu=From(pf_coil),
        p_plasma_ohmic_mw=From(physics),
        t_plant_pulse_coil_precharge=From(times),
        t_plant_pulse_plasma_current_ramp_up=From(times),
        t_plant_pulse_fusion_ramp=From(times),
        t_plant_pulse_burn=From(times),
        t_plant_pulse_plasma_current_ramp_down=From(times),
    ):
        return calculate_pf_coil_power_supplies(
            rmajor=rmajor,
            c_pf_coil_turn_peak_input=c_pf_coil_turn_peak_input,
            rhopfbus=rhopfbus,
            rho_pf_coil=rho_pf_coil,
            r_pf_coil_middle=r_pf_coil_middle,
            j_pf_coil_wp_peak=j_pf_coil_wp_peak,
            f_a_pf_coil_void=f_a_pf_coil_void,
            c_pf_cs_coils_peak_ma=c_pf_cs_coils_peak_ma,
            c_pf_cs_coil_pulse_end_ma=c_pf_cs_coil_pulse_end_ma,
            n_pf_coil_turns=n_pf_coil_turns,
            c_pf_coil_turn=c_pf_coil_turn,
            ind_pf_cs_plasma_mutual=ind_pf_cs_plasma_mutual,
            f_p_pf_energy_store_loss=f_p_pf_energy_store_loss,
            f_p_pf_psu_loss=f_p_pf_psu_loss,
            etapsu=etapsu,
            p_plasma_ohmic_mw=p_plasma_ohmic_mw,
            t_plant_pulse_coil_precharge=t_plant_pulse_coil_precharge,
            t_plant_pulse_plasma_current_ramp_up=(t_plant_pulse_plasma_current_ramp_up),
            t_plant_pulse_fusion_ramp=t_plant_pulse_fusion_ramp,
            t_plant_pulse_burn=t_plant_pulse_burn,
            t_plant_pulse_plasma_current_ramp_down=(
                t_plant_pulse_plasma_current_ramp_down
            ),
            topology=self.topology,
        )
