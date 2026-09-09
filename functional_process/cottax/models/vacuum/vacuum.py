"""Pure-functional port of `process/models/vacuum.py` (registry unit #16)."""

import jax  # noqa: F401
import jax.numpy as jnp  # noqa: F401
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    ImplicitFunction,
    OutputInto,
    resolve,
)
from cottax.problem import Feasibility
from cottax.spec import In, Out, VarPath

from functional_process.models.engineering.ivc_functions import (
    dshellvol,  # noqa: F401
    eshellvol,  # noqa: F401
)
from functional_process.cottax.paths import (
    blanket,
    build,
    divertor,
    fwbs,
    physics,
    tfcoil,
    times,
    vacuum,
)
from functional_process.models.vacuum.vacuum import (
    XMULT,  # noqa: F401
    _solve_vacuum_pumping_old,  # noqa: F401
    _solve_vacuum_pumping_old_from_fields,  # noqa: F401
    calculate_dshaped_vessel_volumes,  # noqa: F401
    calculate_duct_feasibility_conditions,
    calculate_elliptical_vessel_volumes,  # noqa: F401
    calculate_vacuum_pumping_old,
    calculate_vacuum_pumping_simple,
    calculate_vacuum_vessel_mass,  # noqa: F401
    calculate_vacuum_vessel_outputs,
    calculate_vacuum_vessel_outputs_double_null,
    calculate_vacuum_vessel_outputs_dshaped_double_null,
    calculate_vessel_half_height,  # noqa: F401
    calculate_vessel_half_height_double_null,  # noqa: F401
    duct_conductance,  # noqa: F401
    duct_diameter_residual,
    duct_fits_residual,
    pumping_speed_floor_residual,
    solve_duct_diameter,  # noqa: F401
    solve_duct_geometry,  # noqa: F401
)


class VacuumPumpingSimple(ExplicitFunction):
    """cottax node: `calculate_vacuum_pumping_simple`'s combined pump count."""

    n_iter_vacuum_pumps = OutputInto(vacuum)

    def __call__(
        self,
        molflow_plasma_fuelling_required=From(physics),
        molflow_vac_pumps=From(vacuum),
        volflow_vac_pumps_max=From(vacuum),
        f_a_vac_pump_port_plasma_surface=From(vacuum),
        f_volflow_vac_pumps_impedance=From(vacuum),
        a_plasma_surface=From(physics),
        n_tf_coils=From(tfcoil),
        outgasfactor=From(vacuum),
        pres_vv_chamber_base=From(vacuum),
        outgasindex=From(vacuum),
        t_plant_pulse_dwell=From(times),
    ):
        return calculate_vacuum_pumping_simple(
            molflow_plasma_fuelling_required,
            molflow_vac_pumps,
            volflow_vac_pumps_max,
            f_a_vac_pump_port_plasma_surface,
            f_volflow_vac_pumps_impedance,
            a_plasma_surface,
            n_tf_coils,
            outgasfactor,
            pres_vv_chamber_base,
            outgasindex,
            t_plant_pulse_dwell,
        )


class DuctDiameterRootFind(ImplicitFunction):
    """cottax node: `duct_diameter_residual` as a genuine `RootFind` implicit model."""

    d_duct = OutputInto(vacuum)

    def residual(
        self,
        d_duct=From(vacuum),
        l1=From(vacuum),
        l2=From(vacuum),
        l3=From(vacuum),
        xmult_i=From(vacuum),
        ceff_i=From(vacuum),
    ):
        return duct_diameter_residual(d_duct, l1, l2, l3, xmult_i, ceff_i)


class DuctFeasibilityConditions(ExplicitFunction):
    """cottax node: the two inequality residuals `DuctFeasibility` (below) reads."""

    duct_fits_residual = OutputInto(vacuum)
    pumping_speed_floor_residual = OutputInto(vacuum)

    def __call__(
        self,
        d_duct=From(vacuum),
        a1max=From(vacuum),
        ceff_i=From(vacuum),
        s_i=From(vacuum),
    ):
        return calculate_duct_feasibility_conditions(d_duct, a1max, ceff_i, s_i)


DuctFeasibility = Feasibility(
    design=(Out(resolve(vacuum.ceff_i, VarPath)),),
    inequalities=(
        In(resolve(vacuum.duct_fits_residual, VarPath)),
        In(resolve(vacuum.pumping_speed_floor_residual, VarPath)),
    ),
)
"""The declared problem itself: "find a feasible `ceff_i`", no objective."""


class VacuumOld(ExplicitFunction):
    """cottax node: `calculate_vacuum_pumping_old`'s five real outputs."""

    n_vac_pumps_high = OutputInto(vacuum)
    n_vv_vacuum_ducts = OutputInto(vacuum)
    dlscal = OutputInto(vacuum)
    m_vv_vacuum_duct_shield = OutputInto(vacuum)
    dia_vv_vacuum_ducts = OutputInto(vacuum)

    def __call__(
        self,
        p_fusion_total_mw=From(physics),
        rmajor=From(physics),
        rminor=From(physics),
        dr_fw_plasma_gap_inboard=From(build),
        dr_fw_plasma_gap_outboard=From(build),
        a_plasma_surface=From(physics),
        vol_plasma=From(physics),
        dr_shld_outboard=From(build),
        dr_shld_inboard=From(build),
        dr_tf_inboard=From(build),
        r_shld_inboard_inner=From(build),
        dr_shld_vv_gap_inboard=From(build),
        dr_vv_inboard=From(build),
        n_tf_coils=From(tfcoil),
        t_plant_pulse_dwell=From(times),
        n_divertors=From(divertor),
        molflow_plasma_fuelling_required=From(physics),
        m_fuel_amu=From(physics),
        i_vac_pump_dwell=From(vacuum),
        i_vacuum_pump_type=From(vacuum),
        pres_vv_chamber_base=From(vacuum),
        pres_div_chamber_burn=From(vacuum),
        outgrat_fw=From(vacuum),
        t_plant_pulse_coil_precharge=From(times),
    ):
        return calculate_vacuum_pumping_old(
            p_fusion_total_mw,
            rmajor,
            rminor,
            dr_fw_plasma_gap_inboard,
            dr_fw_plasma_gap_outboard,
            a_plasma_surface,
            vol_plasma,
            dr_shld_outboard,
            dr_shld_inboard,
            dr_tf_inboard,
            r_shld_inboard_inner,
            dr_shld_vv_gap_inboard,
            dr_vv_inboard,
            n_tf_coils,
            t_plant_pulse_dwell,
            n_divertors,
            0.0,
            molflow_plasma_fuelling_required,
            m_fuel_amu,
            i_vac_pump_dwell,
            i_vacuum_pump_type,
            pres_vv_chamber_base,
            pres_div_chamber_burn,
            outgrat_fw,
            t_plant_pulse_coil_precharge,
        )


class VacuumVesselElliptical(ExplicitFunction):
    """The family that occupies `.tokamak.vacuum_vessel`: one occupant per cell of the
    shape x divertor-count grid (see the module comment above for the grid).
    """


class VacuumVesselEllipticalSingleNull(VacuumVesselElliptical):
    """cottax node: `.tokamak.vacuum_vessel` at `.divertor.n_divertors == 1` -- the
    combination live on `large_tokamak_eval.IN.DAT` (see module comment above).
    """

    dz_vv_half = OutputInto(blanket)
    vol_vv_inboard = OutputInto(blanket)
    vol_vv_outboard = OutputInto(blanket)
    vol_vv = OutputInto(fwbs)
    m_vv = OutputInto(fwbs)

    def __call__(
        self,
        z_tf_inside_half=From(build),
        dz_shld_vv_gap=From(build),
        dz_vv_lower=From(build),
        dz_blkt_upper=From(build),
        dz_shld_upper=From(build),
        z_plasma_xpoint_upper=From(build),
        dr_fw_plasma_gap_inboard=From(build),
        dr_fw_plasma_gap_outboard=From(build),
        dr_fw_inboard=From(build),
        dr_fw_outboard=From(build),
        rmajor=From(physics),
        rminor=From(physics),
        triang=From(physics),
        r_shld_inboard_inner=From(build),
        r_shld_outboard_outer=From(build),
        dr_vv_inboard=From(build),
        dr_vv_outboard=From(build),
        dz_vv_upper=From(build),
        fvoldw=From(fwbs),
        den_steel=From(fwbs),
    ):
        return calculate_vacuum_vessel_outputs(
            z_tf_inside_half=z_tf_inside_half,
            dz_shld_vv_gap=dz_shld_vv_gap,
            dz_vv_lower=dz_vv_lower,
            dz_blkt_upper=dz_blkt_upper,
            dz_shld_upper=dz_shld_upper,
            z_plasma_xpoint_upper=z_plasma_xpoint_upper,
            dr_fw_plasma_gap_inboard=dr_fw_plasma_gap_inboard,
            dr_fw_plasma_gap_outboard=dr_fw_plasma_gap_outboard,
            dr_fw_inboard=dr_fw_inboard,
            dr_fw_outboard=dr_fw_outboard,
            rmajor=rmajor,
            rminor=rminor,
            triang=triang,
            r_shld_inboard_inner=r_shld_inboard_inner,
            r_shld_outboard_outer=r_shld_outboard_outer,
            dr_vv_inboard=dr_vv_inboard,
            dr_vv_outboard=dr_vv_outboard,
            dz_vv_upper=dz_vv_upper,
            fvoldw=fvoldw,
            den_steel=den_steel,
        )


class VacuumVesselEllipticalDoubleNull(VacuumVesselElliptical):
    """cottax node: `.tokamak.vacuum_vessel` at `.divertor.n_divertors == 2` -- the
    value `spherical_tokamak_eval.IN.DAT` and `st_regression.IN.DAT` derive from
    `i_single_null = 0`.
    """

    dz_vv_half = OutputInto(blanket)
    vol_vv_inboard = OutputInto(blanket)
    vol_vv_outboard = OutputInto(blanket)
    vol_vv = OutputInto(fwbs)
    m_vv = OutputInto(fwbs)

    def __call__(
        self,
        z_tf_inside_half=From(build),
        dz_shld_vv_gap=From(build),
        dz_vv_lower=From(build),
        rmajor=From(physics),
        rminor=From(physics),
        triang=From(physics),
        r_shld_inboard_inner=From(build),
        r_shld_outboard_outer=From(build),
        dr_vv_inboard=From(build),
        dr_vv_outboard=From(build),
        dz_vv_upper=From(build),
        fvoldw=From(fwbs),
        den_steel=From(fwbs),
    ):
        return calculate_vacuum_vessel_outputs_double_null(
            z_tf_inside_half=z_tf_inside_half,
            dz_shld_vv_gap=dz_shld_vv_gap,
            dz_vv_lower=dz_vv_lower,
            rmajor=rmajor,
            rminor=rminor,
            triang=triang,
            r_shld_inboard_inner=r_shld_inboard_inner,
            r_shld_outboard_outer=r_shld_outboard_outer,
            dr_vv_inboard=dr_vv_inboard,
            dr_vv_outboard=dr_vv_outboard,
            dz_vv_upper=dz_vv_upper,
            fvoldw=fvoldw,
            den_steel=den_steel,
        )


class VacuumVesselDShapedDoubleNull(VacuumVesselElliptical):
    """cottax node: `.tokamak.vacuum_vessel` at `.divertor.n_divertors == 2` **and** the
    D-shaped shape arm -- the configuration live on `spherical_tokamak_eval.IN.DAT` and
    `st_regression.IN.DAT` (`i_single_null = 0`; `itart = 1` and `i_fw_blkt_vv_shape =
    1`, either of which alone selects the D-shaped arm).
    """

    dz_vv_half = OutputInto(blanket)
    vol_vv_inboard = OutputInto(blanket)
    vol_vv_outboard = OutputInto(blanket)
    vol_vv = OutputInto(fwbs)
    m_vv = OutputInto(fwbs)

    def __call__(
        self,
        z_tf_inside_half=From(build),
        dz_shld_vv_gap=From(build),
        dz_vv_lower=From(build),
        r_shld_inboard_inner=From(build),
        r_shld_outboard_outer=From(build),
        dr_vv_inboard=From(build),
        dr_vv_outboard=From(build),
        dz_vv_upper=From(build),
        fvoldw=From(fwbs),
        den_steel=From(fwbs),
    ):
        return calculate_vacuum_vessel_outputs_dshaped_double_null(
            z_tf_inside_half=z_tf_inside_half,
            dz_shld_vv_gap=dz_shld_vv_gap,
            dz_vv_lower=dz_vv_lower,
            r_shld_inboard_inner=r_shld_inboard_inner,
            r_shld_outboard_outer=r_shld_outboard_outer,
            dr_vv_inboard=dr_vv_inboard,
            dr_vv_outboard=dr_vv_outboard,
            dz_vv_upper=dz_vv_upper,
            fvoldw=fvoldw,
            den_steel=den_steel,
        )
