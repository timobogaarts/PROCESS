"""Pure-functional port of `process/models/vacuum.py` (registry unit #16)."""

import inspect

import jax  # noqa: F401
import jax.numpy as jnp  # noqa: F401
from cottax.interfaces import V

from functional_process.cottax.paths import (
    blanket,
    build,
    divertor,
    physics,
    tfcoil,
    times,
    vacuum,
)
from functional_process.models.engineering.ivc_functions import (
    dshellvol,  # noqa: F401
    eshellvol,  # noqa: F401
)
from functional_process.models.vacuum.vacuum import (
    XMULT,  # noqa: F401
    _solve_vacuum_pumping_old,  # noqa: F401
    _solve_vacuum_pumping_old_from_fields,  # noqa: F401
    calculate_dshaped_vessel_volumes,  # noqa: F401
    calculate_duct_feasibility_conditions,
    calculate_elliptical_vessel_volumes,  # noqa: F401
    calculate_vacuum_pumping_old_cryopump_dwell,
    calculate_vacuum_pumping_old_cryopump_dwell_and_precharge,
    calculate_vacuum_pumping_old_cryopump_precharge,
    calculate_vacuum_pumping_old_turbomolecular_dwell,
    calculate_vacuum_pumping_old_turbomolecular_dwell_and_precharge,
    calculate_vacuum_pumping_old_turbomolecular_precharge,
    calculate_vacuum_pumping_simple,
    calculate_vacuum_vessel_mass,  # noqa: F401
    calculate_vacuum_vessel_outputs,
    calculate_vacuum_vessel_outputs_double_null,
    calculate_vacuum_vessel_outputs_dshaped_double_null,
    calculate_vessel_half_height,  # noqa: F401
    calculate_vessel_half_height_double_null,  # noqa: F401
    duct_conductance,  # noqa: F401
    duct_diameter_residual,
    solve_duct_diameter,  # noqa: F401
    solve_duct_geometry,  # noqa: F401
)


def _restricted(fn, reads):
    """`reads`, less the parameters `fn` does not take."""
    taken = inspect.signature(fn).parameters
    return {name: place for name, place in reads.items() if name in taken}


# cottax node: `calculate_vacuum_pumping_simple`'s combined pump count.
VacuumPumpingSimple = vacuum.function(
    calculate_vacuum_pumping_simple,
    "n_iter_vacuum_pumps",
)


# cottax node: `duct_diameter_residual` as a genuine `RootFind` implicit model.
def _duct_diameter_root_find(d_duct, l1, l2, l3, xmult_i, ceff_i):
    return duct_diameter_residual(d_duct, l1, l2, l3, xmult_i, ceff_i)


DuctDiameterRootFind = vacuum.residual(_duct_diameter_root_find, "d_duct")


# cottax node: the two inequality residuals, each held `<= 0`.
DuctFeasibilityConditions = vacuum.inequality_constraint(
    calculate_duct_feasibility_conditions,
    "duct_fits_residual",
    "pumping_speed_floor_residual",
)


DuctFeasibility = vacuum.satisfy(
    design=["ceff_i"], subject_to=[DuctFeasibilityConditions]
)
"""The declared problem itself: "find a feasible `ceff_i`", no objective."""


# cottax node: `calculate_vacuum_pumping_old`'s five real outputs, one occupant per
# `.vacuum.i_vacuum_pump_type` x `.vacuum.i_vac_pump_dwell` arm (the pump type decides
# the pump speeds, the dwell switch the pumping time).
_VACUUM_OLD_READS = {
    "p_fusion_total_mw": physics,
    "rmajor": physics,
    "rminor": physics,
    "dr_fw_plasma_gap_inboard": build,
    "dr_fw_plasma_gap_outboard": build,
    "a_plasma_surface": physics,
    "vol_plasma": physics,
    "dr_shld_outboard": build,
    "dr_shld_inboard": build,
    "dr_tf_inboard": build,
    "r_shld_inboard_inner": build,
    "dr_shld_vv_gap_inboard": build,
    "dr_vv_inboard": build,
    "n_tf_coils": tfcoil,
    "t_plant_pulse_dwell": times,
    "n_divertors": divertor,
    "molflow_plasma_fuelling_required": physics,
    "m_fuel_amu": physics,
    "t_plant_pulse_coil_precharge": times,
}

VacuumOldTurbomolecularDwell = vacuum.function(
    calculate_vacuum_pumping_old_turbomolecular_dwell,
    "n_vac_pumps_high",
    "n_vv_vacuum_ducts",
    "dlscal",
    "m_vv_vacuum_duct_shield",
    "dia_vv_vacuum_ducts",
    reads=_restricted(
        calculate_vacuum_pumping_old_turbomolecular_dwell, _VACUUM_OLD_READS
    ),
)

VacuumOldTurbomolecularPrecharge = vacuum.function(
    calculate_vacuum_pumping_old_turbomolecular_precharge,
    "n_vac_pumps_high",
    "n_vv_vacuum_ducts",
    "dlscal",
    "m_vv_vacuum_duct_shield",
    "dia_vv_vacuum_ducts",
    reads=_restricted(
        calculate_vacuum_pumping_old_turbomolecular_precharge, _VACUUM_OLD_READS
    ),
)

VacuumOldTurbomolecularDwellAndPrecharge = vacuum.function(
    calculate_vacuum_pumping_old_turbomolecular_dwell_and_precharge,
    "n_vac_pumps_high",
    "n_vv_vacuum_ducts",
    "dlscal",
    "m_vv_vacuum_duct_shield",
    "dia_vv_vacuum_ducts",
    reads=_restricted(
        calculate_vacuum_pumping_old_turbomolecular_dwell_and_precharge,
        _VACUUM_OLD_READS,
    ),
)

VacuumOldCryopumpDwell = vacuum.function(
    calculate_vacuum_pumping_old_cryopump_dwell,
    "n_vac_pumps_high",
    "n_vv_vacuum_ducts",
    "dlscal",
    "m_vv_vacuum_duct_shield",
    "dia_vv_vacuum_ducts",
    reads=_restricted(calculate_vacuum_pumping_old_cryopump_dwell, _VACUUM_OLD_READS),
)

VacuumOldCryopumpPrecharge = vacuum.function(
    calculate_vacuum_pumping_old_cryopump_precharge,
    "n_vac_pumps_high",
    "n_vv_vacuum_ducts",
    "dlscal",
    "m_vv_vacuum_duct_shield",
    "dia_vv_vacuum_ducts",
    reads=_restricted(
        calculate_vacuum_pumping_old_cryopump_precharge, _VACUUM_OLD_READS
    ),
)

VacuumOldCryopumpDwellAndPrecharge = vacuum.function(
    calculate_vacuum_pumping_old_cryopump_dwell_and_precharge,
    "n_vac_pumps_high",
    "n_vv_vacuum_ducts",
    "dlscal",
    "m_vv_vacuum_duct_shield",
    "dia_vv_vacuum_ducts",
    reads=_restricted(
        calculate_vacuum_pumping_old_cryopump_dwell_and_precharge, _VACUUM_OLD_READS
    ),
)


# The VacuumVesselElliptical family:
# The family that occupies `.tokamak.vacuum_vessel`: one occupant per cell of the
# shape x divertor-count grid (see the module comment above for the grid).


# cottax node: `.tokamak.vacuum_vessel` at `.divertor.n_divertors == 1` -- the
# combination live on `large_tokamak_eval.IN.DAT` (see module comment above).
VacuumVesselEllipticalSingleNull = blanket.function(
    calculate_vacuum_vessel_outputs,
    "dz_vv_half",
    "vol_vv_inboard",
    "vol_vv_outboard",
    V.fwbs.vol_vv,
    V.fwbs.m_vv,
)


# cottax node: `.tokamak.vacuum_vessel` at `.divertor.n_divertors == 2` -- the
# value `spherical_tokamak_eval.IN.DAT` and `st_regression.IN.DAT` derive from
# `i_single_null = 0`.
VacuumVesselEllipticalDoubleNull = blanket.function(
    calculate_vacuum_vessel_outputs_double_null,
    "dz_vv_half",
    "vol_vv_inboard",
    "vol_vv_outboard",
    V.fwbs.vol_vv,
    V.fwbs.m_vv,
)


# cottax node: `.tokamak.vacuum_vessel` at `.divertor.n_divertors == 2` **and** the
# D-shaped shape arm -- the configuration live on `spherical_tokamak_eval.IN.DAT` and
# `st_regression.IN.DAT` (`i_single_null = 0`; `itart = 1` and `i_fw_blkt_vv_shape =
# 1`, either of which alone selects the D-shaped arm).
VacuumVesselDShapedDoubleNull = blanket.function(
    calculate_vacuum_vessel_outputs_dshaped_double_null,
    "dz_vv_half",
    "vol_vv_inboard",
    "vol_vv_outboard",
    V.fwbs.vol_vv,
    V.fwbs.m_vv,
)
