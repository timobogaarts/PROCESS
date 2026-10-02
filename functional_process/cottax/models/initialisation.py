"""The seed's own writes, as nodes: `process/core/init.py` and `st_init`."""

import dataclasses

import equinox as eqx
from cottax.cx import Explicit

from functional_process.cottax.paths import (
    build,
    buildings,
    pf_coil,
    physics,
    tfcoil,
    times,
)
from functional_process.cottax.stated import states


# cottax node: `.tfcoil.eff_tf_cryo`, `init.py:933-940`'s sentinel resolved.
# `eff_tf_cryo`: Stated at `^stated.tfcoil.eff_tf_cryo`, from `indat.resolve_eff_tf_cryo`.
TfCryoplantEfficiency = states(tfcoil, "eff_tf_cryo")


# cottax node: `.tfcoil.eyoung_ins`, `init.py:961-975`'s material table.
# `eyoung_ins`: Insulation Young's modulus (Pa), from `indat.resolve_eyoung_ins`.
TfInsulationYoungsModulus = states(tfcoil, "eyoung_ins")


# cottax node: `.tfcoil.eyoung_cond_axial` and `.tfcoil.eyoung_cond_trans`,
# `init.py:992-1034`.
# `eyoung_cond_trans`: Both from one call to `indat.resolve_eyoung_cond`, stated at `^stated.tfcoil.eyoung_cond_axial` and `^stated.tfcoil.eyoung_cond_trans`.
TfConductorYoungsModulus = states(tfcoil, "eyoung_cond_axial", "eyoung_cond_trans")


# cottax node: `.pf_coil.rho_pf_coil`, `init.py:1140`.
# `rho_pf_coil`: PF coil winding resistivity (ohm-m), from `indat.resolve_rho_pf_coil`.
PfCoilResistivity = states(pf_coil, "rho_pf_coil")


# cottax node: `.physics.f_nd_beam_electron`, `init.py:1145-1147`.
# `f_nd_beam_electron`: Hot beam density as a fraction of the electron density, from `indat.resolve_f_nd_beam_electron`.
BeamElectronDensityFraction = states(physics, "f_nd_beam_electron")


# cottax node: `.buildings.esbldgm3`, `init.py:827`.
# `esbldgm3`: Energy storage building volume (m^3). `0.0` wherever this node exists.
EnergyStorageBuildingVolume = states(buildings, "esbldgm3")


# cottax node: the upper vertical build forced to match the lower,
# `init.py:610-612`.
def _double_null_upper_build(dz_shld_lower, dz_vv_lower):
    return dz_shld_lower, dz_vv_lower


DoubleNullUpperBuild = build.function(
    _double_null_upper_build, "dz_shld_upper", "dz_vv_upper"
)


# cottax node: `.build.dr_cs` and `.build.dr_cs_tf_gap`, `st_init:23,26`.
# `dr_cs`: `st_init:23`'s literal: the solenoid a stellarator does not have is 0 m thick.
# `dr_cs_tf_gap`: `st_init:26`'s literal: and the gap to the TF coil it does not have is 0 m.
StellaratorSolenoidAbsent = states(build, "dr_cs", "dr_cs_tf_gap")


# cottax node: the four pulse phase durations `st_init:43-46` forces.
# `t_plant_pulse_burn`: `st_init:45`'s own comment: one year, 3.15576e7 s.
StellaratorPulseTimes = states(
    times,
    "t_plant_pulse_coil_precharge",
    "t_plant_pulse_plasma_current_ramp_up",
    "t_plant_pulse_burn",
    "t_plant_pulse_plasma_current_ramp_down",
)


class Initialisation(eqx.Module):
    """The seed's writes, one slot per resolved field."""

    tf_cryoplant_efficiency: Explicit | None = dataclasses.field(kw_only=True)
    """`init.py:933-940`."""

    tf_insulation_youngs_modulus: Explicit | None = dataclasses.field(kw_only=True)
    """`init.py:961-975`."""

    tf_conductor_youngs_modulus: Explicit | None = dataclasses.field(kw_only=True)
    """`init.py:992-1034`. `None` on a stellarator, for the same reason."""

    pf_coil_resistivity: Explicit | None = dataclasses.field(kw_only=True)
    """`init.py:1140`. `None` on a stellarator, whose PF nodes do not read it."""

    beam_electron_density_fraction: Explicit | None = dataclasses.field(kw_only=True)
    """`init.py:1145-1147`."""

    energy_storage_building_volume: Explicit | None = dataclasses.field(kw_only=True)
    """`init.py:827`. `None` on a pulsed plant, which keeps its own volume."""

    double_null_upper_build: Explicit | None = dataclasses.field(kw_only=True)
    """`init.py:610-612`. `None` on a single-null machine and on a stellarator."""

    stellarator_solenoid_absent: Explicit | None = dataclasses.field(kw_only=True)
    """`st_init:23,26`."""

    stellarator_pulse_times: Explicit | None = dataclasses.field(kw_only=True)
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
