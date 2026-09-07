"""The seed's own writes, as nodes: `process/core/init.py` and `st_init`."""

import dataclasses

from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    ModelNamespace,
    OutputInto,
)

from functional_process.cottax.stated import StatesValues
from functional_process.cottax.paths import (
    build,
    buildings,
    pf_coil,
    physics,
    tfcoil,
    times,
)


class TfCryoplantEfficiency(StatesValues):
    """cottax node: `.tfcoil.eff_tf_cryo`, `init.py:933-940`'s sentinel resolved."""

    eff_tf_cryo = OutputInto(tfcoil)
    """Stated at `^stated.tfcoil.eff_tf_cryo`, from `indat.resolve_eff_tf_cryo`."""


class TfInsulationYoungsModulus(StatesValues):
    """cottax node: `.tfcoil.eyoung_ins`, `init.py:961-975`'s material table."""

    eyoung_ins = OutputInto(tfcoil)
    """Insulation Young's modulus (Pa), from `indat.resolve_eyoung_ins`."""


class TfConductorYoungsModulus(StatesValues):
    """cottax node: `.tfcoil.eyoung_cond_axial` and `.tfcoil.eyoung_cond_trans`,
    `init.py:992-1034`.
    """

    eyoung_cond_axial = OutputInto(tfcoil)
    eyoung_cond_trans = OutputInto(tfcoil)
    """Both from one call to `indat.resolve_eyoung_cond`, stated at
    `^stated.tfcoil.eyoung_cond_axial` and `^stated.tfcoil.eyoung_cond_trans`.
    """


class PfCoilResistivity(StatesValues):
    """cottax node: `.pf_coil.rho_pf_coil`, `init.py:1140`."""

    rho_pf_coil = OutputInto(pf_coil)
    """PF coil winding resistivity (ohm-m), from `indat.resolve_rho_pf_coil`."""


class BeamElectronDensityFraction(StatesValues):
    """cottax node: `.physics.f_nd_beam_electron`, `init.py:1145-1147`."""

    f_nd_beam_electron = OutputInto(physics)
    """Hot beam density as a fraction of the electron density, from
    `indat.resolve_f_nd_beam_electron`.
    """


class EnergyStorageBuildingVolume(StatesValues):
    """cottax node: `.buildings.esbldgm3`, `init.py:827`."""

    esbldgm3 = OutputInto(buildings)
    """Energy storage building volume (m^3). `0.0` wherever this node exists."""


class DoubleNullUpperBuild(ExplicitFunction):
    """cottax node: the upper vertical build forced to match the lower,
    `init.py:610-612`.
    """

    dz_shld_upper = OutputInto(build)
    dz_vv_upper = OutputInto(build)

    def __call__(self, dz_shld_lower=From(build), dz_vv_lower=From(build)):
        return dz_shld_lower, dz_vv_lower


class StellaratorSolenoidAbsent(StatesValues):
    """cottax node: `.build.dr_cs` and `.build.dr_cs_tf_gap`, `st_init:23,26`."""

    dr_cs = OutputInto(build)
    """`st_init:23`'s literal: the solenoid a stellarator does not have is 0 m thick."""
    dr_cs_tf_gap = OutputInto(build)
    """`st_init:26`'s literal: and the gap to the TF coil it does not have is 0 m."""


class StellaratorPulseTimes(StatesValues):
    """cottax node: the four pulse phase durations `st_init:43-46` forces."""

    t_plant_pulse_coil_precharge = OutputInto(times)
    t_plant_pulse_plasma_current_ramp_up = OutputInto(times)
    t_plant_pulse_burn = OutputInto(times)
    """`st_init:45`'s own comment: one year, 3.15576e7 s."""
    t_plant_pulse_plasma_current_ramp_down = OutputInto(times)


class Initialisation(ModelNamespace):
    """The seed's writes, one slot per resolved field."""

    tf_cryoplant_efficiency: TfCryoplantEfficiency | None = dataclasses.field(
        kw_only=True
    )
    """`init.py:933-940`."""

    tf_insulation_youngs_modulus: TfInsulationYoungsModulus | None = dataclasses.field(
        kw_only=True
    )
    """`init.py:961-975`."""

    tf_conductor_youngs_modulus: TfConductorYoungsModulus | None = dataclasses.field(
        kw_only=True
    )
    """`init.py:992-1034`. `None` on a stellarator, for the same reason."""

    pf_coil_resistivity: PfCoilResistivity | None = dataclasses.field(kw_only=True)
    """`init.py:1140`. `None` on a stellarator, whose PF nodes do not read it."""

    beam_electron_density_fraction: BeamElectronDensityFraction | None = (
        dataclasses.field(kw_only=True)
    )
    """`init.py:1145-1147`."""

    energy_storage_building_volume: EnergyStorageBuildingVolume | None = (
        dataclasses.field(kw_only=True)
    )
    """`init.py:827`. `None` on a pulsed plant, which keeps its own volume."""

    double_null_upper_build: DoubleNullUpperBuild | None = dataclasses.field(
        kw_only=True
    )
    """`init.py:610-612`. `None` on a single-null machine and on a stellarator."""

    stellarator_solenoid_absent: StellaratorSolenoidAbsent | None = dataclasses.field(
        kw_only=True
    )
    """`st_init:23,26`."""

    stellarator_pulse_times: StellaratorPulseTimes | None = dataclasses.field(
        kw_only=True
    )
    """`st_init:43-46`."""


__all__ = [
    "BeamElectronDensityFraction",
    "DoubleNullUpperBuild",
    "EnergyStorageBuildingVolume",
    "Initialisation",
    "PfCoilResistivity",
    "StellaratorPulseTimes",
    "StellaratorSolenoidAbsent",
    "TfConductorYoungsModulus",
    "TfCryoplantEfficiency",
    "TfInsulationYoungsModulus",
]
