"""The PF/CS volt-second accounting: per-turn current waveforms and `PFCoil.vsec`."""

import equinox as eqx
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.cottax.pfcoil import (
    REFERENCE_TOPOLOGY,
    SPHERICAL_TOKAMAK_TOPOLOGY,
    PFCoilTopology,
)
from functional_process.cottax.paths import pf_coil, physics
from functional_process.models.pfcoil.volt_seconds import (
    calculate_pf_coil_turn_currents,
    calculate_pf_cs_volt_seconds,
    calculate_pf_volt_seconds_no_central_solenoid,
)


class PFCoilTurnCurrents(ExplicitFunction):
    """cottax node: `.tokamak.pf_coil.turn_currents`."""

    topology: PFCoilTopology = eqx.field(static=True, default=REFERENCE_TOPOLOGY)
    """Static, and the only thing that changes between the two machines: which row is
    the plasma's.
    """

    c_pf_coil_turn = OutputInto(pf_coil)

    def __call__(
        self,
        f_c_pf_cs_peak_time_array=From(pf_coil),
        c_pf_coil_turn_peak_input=From(pf_coil),
        c_pf_cs_coils_peak_ma=From(pf_coil),
        plasma_current=From(physics),
    ):
        return calculate_pf_coil_turn_currents(
            f_c_pf_cs_peak_time_array=f_c_pf_cs_peak_time_array,
            c_pf_coil_turn_peak_input=c_pf_coil_turn_peak_input,
            c_pf_cs_coils_peak_ma=c_pf_cs_coils_peak_ma,
            plasma_current=plasma_current,
            topology=self.topology,
        )


class PFCoilVoltSeconds(ExplicitFunction):
    """cottax node: `.tokamak.pf_coil.volt_seconds`."""

    vs_cs_pf_total_burn = OutputInto(pf_coil)
    vs_cs_pf_total_pulse = OutputInto(pf_coil)

    def __call__(
        self,
        ind_pf_cs_plasma_mutual=From(pf_coil),
        c_pf_coil_turn=From(pf_coil),
    ):
        return calculate_pf_cs_volt_seconds(
            ind_pf_cs_plasma_mutual=ind_pf_cs_plasma_mutual,
            c_pf_coil_turn=c_pf_coil_turn,
        )


class PFCoilVoltSecondsNoCentralSolenoid(PFCoilVoltSeconds):
    """cottax node: `.tokamak.pf_coil.volt_seconds`, the `iohcl = 0` occupant."""

    topology: PFCoilTopology = eqx.field(static=True, default=SPHERICAL_TOKAMAK_TOPOLOGY)

    def __call__(
        self,
        ind_pf_cs_plasma_mutual=From(pf_coil),
        c_pf_coil_turn=From(pf_coil),
    ):
        return calculate_pf_volt_seconds_no_central_solenoid(
            ind_pf_cs_plasma_mutual=ind_pf_cs_plasma_mutual,
            c_pf_coil_turn=c_pf_coil_turn,
            topology=self.topology,
        )
