"""Pure-functional port of `process/models/blankets/hcpb.py`'s `CCFE_HCPB`."""

import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.models.blankets.hcpb import (
    calculate_centrepost_angle_fraction,
    calculate_centrepost_fast_neutron_flux_superconducting,
    calculate_centrepost_neutronics_absent,
    calculate_centrepost_neutronics_spherical_tokamak_superconducting,
    calculate_centrepost_nuclear_heating_superconducting,
    calculate_component_masses,
    calculate_divertor_surface_and_plate_mass_double_null,
    calculate_divertor_surface_and_plate_mass_single_null,
    calculate_first_wall_radiation_powers,
    calculate_fw_coolant_void_fractions,
    calculate_nuclear_heating_magnets_conventional,
    calculate_nuclear_heating_magnets_spherical_tokamak,
    calculate_nuclear_heating_renormalisation_double_null_conventional,
    calculate_nuclear_heating_renormalisation_double_null_spherical_tokamak,
    calculate_nuclear_heating_renormalisation_single_null_conventional,
    calculate_nuclear_heating_renormalisation_single_null_spherical_tokamak,
    calculate_pumping_power_mechanical_with_pressure_drop,
    nuclear_heating_blanket,
    nuclear_heating_fw,
    nuclear_heating_shield_conventional,
    nuclear_heating_shield_spherical_tokamak,
)
from functional_process.models.safe_math import safe_pow, safe_sqrt
from functional_process.cottax.stated import StatesValues
from functional_process.cottax.paths import (
    build,
    ccfe_hcpb,
    current_drive,
    divertor,
    first_wall,
    fwbs,
    heat_transport,
    physics,
    primary_pumping,
    tfcoil,
)
from functional_process.vocabulary import constants

# ruff's docstring rules treat `__all__` membership as the definition of "public" once
# one is present, so this lists every public name this module resolved before step 2 of
# `_audit/formulas_split.md` moved the pure functions out -- not just `jnp`/`constants`/
# `safe_pow`/`safe_sqrt` and the four functions no surviving declaration calls directly
# (`calculate_centrepost_neutronics_absent`, `calculate_centrepost_angle_fraction`,
# `calculate_centrepost_fast_neutron_flux_superconducting`,
# `calculate_centrepost_nuclear_heating_superconducting` -- the last three are called
# only from inside `calculate_centrepost_neutronics_spherical_tokamak_superconducting`,
# not from any node), which are the only names actually unused here (see
# `power/electric_production.py`'s commit for why a partial list is the wrong move).
__all__ = [
    "CentrepostNeutronics",
    "CentrepostNeutronicsAbsent",
    "CentrepostNeutronicsSphericalTokamakSuperconducting",
    "ComponentMasses",
    "DivertorSurfaceAndPlateMass",
    "DivertorSurfaceAndPlateMassDoubleNull",
    "DivertorSurfaceAndPlateMassSingleNull",
    "ExplicitFunction",
    "FirstWallCoolantVoidFractions",
    "FirstWallRadiationPowers",
    "From",
    "NuclearHeatingBlanket",
    "NuclearHeatingFw",
    "NuclearHeatingMagnets",
    "NuclearHeatingMagnetsConventional",
    "NuclearHeatingMagnetsSphericalTokamak",
    "NuclearHeatingRenormalisation",
    "NuclearHeatingRenormalisationDoubleNullConventional",
    "NuclearHeatingRenormalisationDoubleNullSphericalTokamak",
    "NuclearHeatingRenormalisationSingleNullConventional",
    "NuclearHeatingRenormalisationSingleNullSphericalTokamak",
    "NuclearHeatingShield",
    "NuclearHeatingShieldConventional",
    "NuclearHeatingShieldSphericalTokamak",
    "OutputInto",
    "PumpingPowerMechanicalWithPressureDrop",
    "StatesValues",
    "build",
    "calculate_centrepost_angle_fraction",
    "calculate_centrepost_fast_neutron_flux_superconducting",
    "calculate_centrepost_neutronics_absent",
    "calculate_centrepost_neutronics_spherical_tokamak_superconducting",
    "calculate_centrepost_nuclear_heating_superconducting",
    "calculate_component_masses",
    "calculate_divertor_surface_and_plate_mass_double_null",
    "calculate_divertor_surface_and_plate_mass_single_null",
    "calculate_first_wall_radiation_powers",
    "calculate_fw_coolant_void_fractions",
    "calculate_nuclear_heating_magnets_conventional",
    "calculate_nuclear_heating_magnets_spherical_tokamak",
    "calculate_nuclear_heating_renormalisation_double_null_conventional",
    "calculate_nuclear_heating_renormalisation_double_null_spherical_tokamak",
    "calculate_nuclear_heating_renormalisation_single_null_conventional",
    "calculate_nuclear_heating_renormalisation_single_null_spherical_tokamak",
    "calculate_pumping_power_mechanical_with_pressure_drop",
    "ccfe_hcpb",
    "constants",
    "current_drive",
    "divertor",
    "first_wall",
    "fwbs",
    "heat_transport",
    "jnp",
    "nuclear_heating_blanket",
    "nuclear_heating_fw",
    "nuclear_heating_shield_conventional",
    "nuclear_heating_shield_spherical_tokamak",
    "physics",
    "primary_pumping",
    "safe_pow",
    "safe_sqrt",
    "tfcoil",
]


class FirstWallCoolantVoidFractions(ExplicitFunction):
    """cottax node: `calculate_fw_coolant_void_fractions`."""

    f_a_fw_coolant_inboard = OutputInto(fwbs)
    f_a_fw_coolant_outboard = OutputInto(fwbs)

    def __call__(
        self,
        radius_fw_channel=From(fwbs),
        dx_fw_module=From(fwbs),
        dr_fw_inboard=From(build),
    ):
        return calculate_fw_coolant_void_fractions(
            radius_fw_channel, dx_fw_module, dr_fw_inboard
        )


class DivertorSurfaceAndPlateMass(ExplicitFunction):
    """The family that owns `.divertor.a_div_surface_total` and `.divertor.m_div_plate`:
    one occupant per `n_divertors` arm of `component_masses`' `hcpb.py:353-367`.
    """


class DivertorSurfaceAndPlateMassSingleNull(DivertorSurfaceAndPlateMass):
    """cottax node: `calculate_divertor_surface_and_plate_mass_single_null`."""

    a_div_surface_total = OutputInto(divertor)
    m_div_plate = OutputInto(divertor)

    def __call__(
        self,
        fdiva=From(divertor),
        rmajor=From(physics),
        rminor=From(physics),
        den_div_structure=From(divertor),
        f_vol_div_coolant=From(divertor),
        dx_div_plate=From(divertor),
    ):
        return calculate_divertor_surface_and_plate_mass_single_null(
            fdiva,
            rmajor,
            rminor,
            den_div_structure,
            f_vol_div_coolant,
            dx_div_plate,
        )


class DivertorSurfaceAndPlateMassDoubleNull(DivertorSurfaceAndPlateMass):
    """cottax node: `calculate_divertor_surface_and_plate_mass_double_null`."""

    a_div_surface_total = OutputInto(divertor)
    m_div_plate = OutputInto(divertor)

    def __call__(
        self,
        fdiva=From(divertor),
        rmajor=From(physics),
        rminor=From(physics),
        den_div_structure=From(divertor),
        f_vol_div_coolant=From(divertor),
        dx_div_plate=From(divertor),
    ):
        return calculate_divertor_surface_and_plate_mass_double_null(
            fdiva,
            rmajor,
            rminor,
            den_div_structure,
            f_vol_div_coolant,
            dx_div_plate,
        )


class ComponentMasses(ExplicitFunction):
    """cottax node: `calculate_component_masses`."""

    m_fw_blkt_div_coolant_total = OutputInto(fwbs)
    fwclfr = OutputInto(fwbs)
    whtshld = OutputInto(fwbs)
    wpenshld = OutputInto(fwbs)
    vol_fw_total = OutputInto(fwbs)
    m_fw_total = OutputInto(fwbs)
    fw_armour_vol = OutputInto(fwbs)
    fw_armour_mass = OutputInto(fwbs)
    f_vol_blkt_li4sio4 = OutputInto(fwbs)
    f_vol_blkt_tibe12 = OutputInto(fwbs)
    m_blkt_tibe12 = OutputInto(fwbs)
    m_blkt_li4sio4 = OutputInto(fwbs)
    m_blkt_beryllium = OutputInto(fwbs)
    m_blkt_li2o = OutputInto(fwbs)
    f_vol_blkt_steel = OutputInto(fwbs)
    m_blkt_steel_total = OutputInto(fwbs)
    m_blkt_total = OutputInto(fwbs)
    armour_fw_bl_mass = OutputInto(fwbs)

    def __call__(
        self,
        a_div_surface_total=From(divertor),
        f_vol_div_coolant=From(divertor),
        dx_div_plate=From(divertor),
        vol_blkt_total=From(fwbs),
        f_a_blkt_cooling_channels=From(fwbs),
        vol_shld_total=From(fwbs),
        vfshld=From(fwbs),
        a_fw_inboard=From(first_wall),
        a_fw_outboard=From(first_wall),
        a_fw_total=From(first_wall),
        dr_fw_inboard=From(build),
        dr_fw_outboard=From(build),
        f_a_fw_coolant_inboard=From(fwbs),
        f_a_fw_coolant_outboard=From(fwbs),
        den_steel=From(fwbs),
        a_plasma_surface=From(physics),
        fw_armour_thickness=From(fwbs),
        breeder_f=From(fwbs),
        breeder_multiplier=From(fwbs),
        vfcblkt=From(fwbs),
        vfpblkt=From(fwbs),
    ):
        return calculate_component_masses(
            a_div_surface_total,
            f_vol_div_coolant,
            dx_div_plate,
            vol_blkt_total,
            f_a_blkt_cooling_channels,
            vol_shld_total,
            vfshld,
            a_fw_inboard,
            a_fw_outboard,
            a_fw_total,
            dr_fw_inboard,
            dr_fw_outboard,
            f_a_fw_coolant_inboard,
            f_a_fw_coolant_outboard,
            den_steel,
            a_plasma_surface,
            fw_armour_thickness,
            breeder_f,
            breeder_multiplier,
            vfcblkt,
            vfpblkt,
        )


class NuclearHeatingMagnets(ExplicitFunction):
    """The family that owns the nine `nuclear_heating_magnets` outputs: one occupant per
    value of `.physics.itart` (`hcpb.py:495-575`).
    """


class NuclearHeatingMagnetsConventional(NuclearHeatingMagnets):
    """cottax node: `calculate_nuclear_heating_magnets_conventional`. `itart == 0`."""

    armour_density = OutputInto(ccfe_hcpb)
    fw_density = OutputInto(ccfe_hcpb)
    blanket_density = OutputInto(ccfe_hcpb)
    shield_density = OutputInto(ccfe_hcpb)
    vv_density = OutputInto(ccfe_hcpb)
    x_blanket = OutputInto(ccfe_hcpb)
    x_shield = OutputInto(ccfe_hcpb)
    tfc_nuc_heating = OutputInto(ccfe_hcpb)
    p_tf_nuclear_heat_mw_unnormalised = OutputInto(ccfe_hcpb)
    """Minted."""

    def __call__(
        self,
        radius_fw_channel=From(fwbs),
        dx_fw_module=From(fwbs),
        dr_fw_inboard=From(build),
        dr_fw_outboard=From(build),
        den_steel=From(fwbs),
        m_blkt_total=From(fwbs),
        vol_blkt_total=From(fwbs),
        whtshld=From(fwbs),
        vol_shld_total=From(fwbs),
        dr_vv_inboard=From(build),
        dr_vv_outboard=From(build),
        m_vv=From(fwbs),
        vol_vv=From(fwbs),
        dr_blkt_outboard=From(build),
        dr_blkt_inboard=From(build),
        dr_shld_outboard=From(build),
        dr_shld_inboard=From(build),
        fw_armour_thickness=From(fwbs),
        m_tf_coils_total=From(tfcoil),
        p_fusion_total_mw=From(physics),
    ):
        return calculate_nuclear_heating_magnets_conventional(
            radius_fw_channel,
            dx_fw_module,
            dr_fw_inboard,
            dr_fw_outboard,
            den_steel,
            m_blkt_total,
            vol_blkt_total,
            whtshld,
            vol_shld_total,
            dr_vv_inboard,
            dr_vv_outboard,
            m_vv,
            vol_vv,
            dr_blkt_outboard,
            dr_blkt_inboard,
            dr_shld_outboard,
            dr_shld_inboard,
            fw_armour_thickness,
            m_tf_coils_total,
            p_fusion_total_mw,
        )


class NuclearHeatingMagnetsSphericalTokamak(NuclearHeatingMagnets):
    """cottax node: `calculate_nuclear_heating_magnets_spherical_tokamak`."""

    armour_density = OutputInto(ccfe_hcpb)
    fw_density = OutputInto(ccfe_hcpb)
    blanket_density = OutputInto(ccfe_hcpb)
    shield_density = OutputInto(ccfe_hcpb)
    vv_density = OutputInto(ccfe_hcpb)
    x_blanket = OutputInto(ccfe_hcpb)
    x_shield = OutputInto(ccfe_hcpb)
    tfc_nuc_heating = OutputInto(ccfe_hcpb)
    p_tf_nuclear_heat_mw_unnormalised = OutputInto(ccfe_hcpb)

    def __call__(
        self,
        radius_fw_channel=From(fwbs),
        dx_fw_module=From(fwbs),
        dr_fw_inboard=From(build),
        dr_fw_outboard=From(build),
        den_steel=From(fwbs),
        m_blkt_total=From(fwbs),
        vol_blkt_total=From(fwbs),
        whtshld=From(fwbs),
        vol_shld_total=From(fwbs),
        dr_vv_inboard=From(build),
        dr_vv_outboard=From(build),
        m_vv=From(fwbs),
        vol_vv=From(fwbs),
        dr_blkt_outboard=From(build),
        dr_shld_outboard=From(build),
        fw_armour_thickness=From(fwbs),
        whttflgs=From(tfcoil),
        p_fusion_total_mw=From(physics),
    ):
        return calculate_nuclear_heating_magnets_spherical_tokamak(
            radius_fw_channel,
            dx_fw_module,
            dr_fw_inboard,
            dr_fw_outboard,
            den_steel,
            m_blkt_total,
            vol_blkt_total,
            whtshld,
            vol_shld_total,
            dr_vv_inboard,
            dr_vv_outboard,
            m_vv,
            vol_vv,
            dr_blkt_outboard,
            dr_shld_outboard,
            fw_armour_thickness,
            whttflgs,
            p_fusion_total_mw,
        )


class NuclearHeatingFw(ExplicitFunction):
    """cottax node: `nuclear_heating_fw`, unchanged."""

    p_fw_nuclear_heat_total_mw_unnormalised = OutputInto(ccfe_hcpb)
    """Minted; `.fwbs.p_fw_nuclear_heat_total_mw` is the renormalised value."""

    def __call__(
        self,
        m_fw_total=From(fwbs),
        fw_armour_u_nuc_heating=From(ccfe_hcpb),
        p_fusion_total_mw=From(physics),
    ):
        return nuclear_heating_fw(m_fw_total, fw_armour_u_nuc_heating, p_fusion_total_mw)


class NuclearHeatingBlanket(ExplicitFunction):
    """cottax node: `nuclear_heating_blanket`, unchanged."""

    p_blkt_nuclear_heat_total_mw_unnormalised = OutputInto(ccfe_hcpb)
    """Minted; `.fwbs.p_blkt_nuclear_heat_total_mw` is the renormalised value."""
    exp_blanket = OutputInto(ccfe_hcpb)

    def __call__(
        self,
        m_blkt_total=From(fwbs),
        p_fusion_total_mw=From(physics),
    ):
        return nuclear_heating_blanket(m_blkt_total, p_fusion_total_mw)


class NuclearHeatingShield(ExplicitFunction):
    """The family that owns the four `nuclear_heating_shield` outputs: one occupant per
    value of `.physics.itart` (`hcpb.py:748-769`).
    """


class NuclearHeatingShieldConventional(NuclearHeatingShield):
    """cottax node: `nuclear_heating_shield_conventional`."""

    p_shld_nuclear_heat_mw_unnormalised = OutputInto(ccfe_hcpb)
    """Minted; `.fwbs.p_shld_nuclear_heat_mw` is the renormalised value."""
    exp_shield1 = OutputInto(ccfe_hcpb)
    exp_shield2 = OutputInto(ccfe_hcpb)
    shld_u_nuc_heating = OutputInto(ccfe_hcpb)

    def __call__(
        self,
        dr_shld_outboard=From(build),
        dr_shld_inboard=From(build),
        shield_density=From(ccfe_hcpb),
        whtshld=From(fwbs),
        x_blanket=From(ccfe_hcpb),
        p_fusion_total_mw=From(physics),
    ):
        return nuclear_heating_shield_conventional(
            dr_shld_outboard,
            dr_shld_inboard,
            shield_density,
            whtshld,
            x_blanket,
            p_fusion_total_mw,
        )


class NuclearHeatingShieldSphericalTokamak(NuclearHeatingShield):
    """cottax node: `nuclear_heating_shield_spherical_tokamak`."""

    p_shld_nuclear_heat_mw_unnormalised = OutputInto(ccfe_hcpb)
    exp_shield1 = OutputInto(ccfe_hcpb)
    exp_shield2 = OutputInto(ccfe_hcpb)
    shld_u_nuc_heating = OutputInto(ccfe_hcpb)

    def __call__(
        self,
        dr_shld_outboard=From(build),
        shield_density=From(ccfe_hcpb),
        whtshld=From(fwbs),
        x_blanket=From(ccfe_hcpb),
        p_fusion_total_mw=From(physics),
    ):
        return nuclear_heating_shield_spherical_tokamak(
            dr_shld_outboard,
            shield_density,
            whtshld,
            x_blanket,
            p_fusion_total_mw,
        )


class CentrepostNeutronics(ExplicitFunction):
    """The family that owns `run()`'s centrepost block (`hcpb.py:103-148`): one occupant
    per cell of the joint `(itart, i_tf_sup)` arm PROCESS's three `st_*` routines cut
    between them.
    """


class CentrepostNeutronicsAbsent(CentrepostNeutronics, StatesValues):
    """cottax node: `calculate_centrepost_neutronics_absent`."""

    pnuc_cp_tf = OutputInto(fwbs)
    p_cp_shield_nuclear_heat_mw = OutputInto(fwbs)
    pnuc_cp = OutputInto(fwbs)
    neut_flux_cp = OutputInto(fwbs)


class CentrepostNeutronicsSphericalTokamakSuperconducting(CentrepostNeutronics):
    """cottax node: `calculate_centrepost_neutronics_spherical_tokamak_superconducting`.
    """

    f_geom_cp = OutputInto(ccfe_hcpb)
    """Minted. A local of `run()` in PROCESS, read by the renormalisation."""
    neut_flux_cp = OutputInto(fwbs)
    pnuc_cp_tf = OutputInto(fwbs)
    p_cp_shield_nuclear_heat_mw_fit = OutputInto(ccfe_hcpb)
    """Minted."""
    pnuc_cp = OutputInto(fwbs)

    def __call__(
        self,
        rmajor=From(physics),
        rminor=From(physics),
        triang=From(physics),
        dr_fw_plasma_gap_inboard=From(build),
        z_plasma_xpoint_upper=From(build),
        r_sh_inboard_out=From(build),
        p_neutron_total_mw=From(physics),
        dr_shld_inboard=From(build),
    ):
        return calculate_centrepost_neutronics_spherical_tokamak_superconducting(
            rmajor,
            rminor,
            triang,
            dr_fw_plasma_gap_inboard,
            z_plasma_xpoint_upper,
            r_sh_inboard_out,
            p_neutron_total_mw,
            dr_shld_inboard,
        )


class NuclearHeatingRenormalisation(ExplicitFunction):
    """The family that owns the four renormalised nuclear-heating powers: one occupant
    per cell of the `(n_divertors, itart)` pair `hcpb.py:195-276` branches on.
    """


class NuclearHeatingRenormalisationSingleNullConventional(NuclearHeatingRenormalisation):
    """cottax node:
    `calculate_nuclear_heating_renormalisation_single_null_conventional`.
    """

    pnuc_tot_blk_sector = OutputInto(ccfe_hcpb)
    p_fw_nuclear_heat_total_mw = OutputInto(fwbs)
    p_blkt_nuclear_heat_total_mw = OutputInto(fwbs)
    p_shld_nuclear_heat_mw = OutputInto(fwbs)
    p_tf_nuclear_heat_mw = OutputInto(fwbs)
    p_blkt_multiplication_mw = OutputInto(fwbs)

    def __call__(
        self,
        p_fw_nuclear_heat_total_mw_unnormalised=From(ccfe_hcpb),
        p_blkt_nuclear_heat_total_mw_unnormalised=From(ccfe_hcpb),
        p_shld_nuclear_heat_mw_unnormalised=From(ccfe_hcpb),
        p_tf_nuclear_heat_mw_unnormalised=From(ccfe_hcpb),
        f_ster_div_single=From(fwbs),
        f_p_blkt_multiplication=From(fwbs),
        p_neutron_total_mw=From(physics),
    ):
        return calculate_nuclear_heating_renormalisation_single_null_conventional(
            p_fw_nuclear_heat_total_mw_unnormalised,
            p_blkt_nuclear_heat_total_mw_unnormalised,
            p_shld_nuclear_heat_mw_unnormalised,
            p_tf_nuclear_heat_mw_unnormalised,
            f_ster_div_single,
            f_p_blkt_multiplication,
            p_neutron_total_mw,
        )


class NuclearHeatingRenormalisationDoubleNullConventional(NuclearHeatingRenormalisation):
    """cottax node:
    `calculate_nuclear_heating_renormalisation_double_null_conventional`.
    """

    pnuc_tot_blk_sector = OutputInto(ccfe_hcpb)
    p_fw_nuclear_heat_total_mw = OutputInto(fwbs)
    p_blkt_nuclear_heat_total_mw = OutputInto(fwbs)
    p_shld_nuclear_heat_mw = OutputInto(fwbs)
    p_tf_nuclear_heat_mw = OutputInto(fwbs)
    p_blkt_multiplication_mw = OutputInto(fwbs)

    def __call__(
        self,
        p_fw_nuclear_heat_total_mw_unnormalised=From(ccfe_hcpb),
        p_blkt_nuclear_heat_total_mw_unnormalised=From(ccfe_hcpb),
        p_shld_nuclear_heat_mw_unnormalised=From(ccfe_hcpb),
        p_tf_nuclear_heat_mw_unnormalised=From(ccfe_hcpb),
        f_ster_div_single=From(fwbs),
        f_p_blkt_multiplication=From(fwbs),
        p_neutron_total_mw=From(physics),
    ):
        return calculate_nuclear_heating_renormalisation_double_null_conventional(
            p_fw_nuclear_heat_total_mw_unnormalised,
            p_blkt_nuclear_heat_total_mw_unnormalised,
            p_shld_nuclear_heat_mw_unnormalised,
            p_tf_nuclear_heat_mw_unnormalised,
            f_ster_div_single,
            f_p_blkt_multiplication,
            p_neutron_total_mw,
        )


class NuclearHeatingRenormalisationSingleNullSphericalTokamak(
    NuclearHeatingRenormalisation
):
    """cottax node:
    `calculate_nuclear_heating_renormalisation_single_null_spherical_tokamak`.
    """

    pnuc_tot_blk_sector = OutputInto(ccfe_hcpb)
    p_fw_nuclear_heat_total_mw = OutputInto(fwbs)
    p_blkt_nuclear_heat_total_mw = OutputInto(fwbs)
    p_shld_nuclear_heat_mw = OutputInto(fwbs)
    p_tf_nuclear_heat_mw = OutputInto(fwbs)
    p_cp_shield_nuclear_heat_mw = OutputInto(fwbs)
    p_blkt_multiplication_mw = OutputInto(fwbs)

    def __call__(
        self,
        p_fw_nuclear_heat_total_mw_unnormalised=From(ccfe_hcpb),
        p_blkt_nuclear_heat_total_mw_unnormalised=From(ccfe_hcpb),
        p_shld_nuclear_heat_mw_unnormalised=From(ccfe_hcpb),
        p_tf_nuclear_heat_mw_unnormalised=From(ccfe_hcpb),
        f_ster_div_single=From(fwbs),
        f_p_blkt_multiplication=From(fwbs),
        p_neutron_total_mw=From(physics),
        f_geom_cp=From(ccfe_hcpb),
        pnuc_cp_tf=From(fwbs),
    ):
        return calculate_nuclear_heating_renormalisation_single_null_spherical_tokamak(
            p_fw_nuclear_heat_total_mw_unnormalised,
            p_blkt_nuclear_heat_total_mw_unnormalised,
            p_shld_nuclear_heat_mw_unnormalised,
            p_tf_nuclear_heat_mw_unnormalised,
            f_ster_div_single,
            f_p_blkt_multiplication,
            p_neutron_total_mw,
            f_geom_cp,
            pnuc_cp_tf,
        )


class NuclearHeatingRenormalisationDoubleNullSphericalTokamak(
    NuclearHeatingRenormalisation
):
    """cottax node:
    `calculate_nuclear_heating_renormalisation_double_null_spherical_tokamak`.
    """

    pnuc_tot_blk_sector = OutputInto(ccfe_hcpb)
    p_fw_nuclear_heat_total_mw = OutputInto(fwbs)
    p_blkt_nuclear_heat_total_mw = OutputInto(fwbs)
    p_shld_nuclear_heat_mw = OutputInto(fwbs)
    p_tf_nuclear_heat_mw = OutputInto(fwbs)
    p_cp_shield_nuclear_heat_mw = OutputInto(fwbs)
    p_blkt_multiplication_mw = OutputInto(fwbs)

    def __call__(
        self,
        p_fw_nuclear_heat_total_mw_unnormalised=From(ccfe_hcpb),
        p_blkt_nuclear_heat_total_mw_unnormalised=From(ccfe_hcpb),
        p_shld_nuclear_heat_mw_unnormalised=From(ccfe_hcpb),
        p_tf_nuclear_heat_mw_unnormalised=From(ccfe_hcpb),
        f_ster_div_single=From(fwbs),
        f_p_blkt_multiplication=From(fwbs),
        p_neutron_total_mw=From(physics),
        f_geom_cp=From(ccfe_hcpb),
        pnuc_cp_tf=From(fwbs),
    ):
        return calculate_nuclear_heating_renormalisation_double_null_spherical_tokamak(
            p_fw_nuclear_heat_total_mw_unnormalised,
            p_blkt_nuclear_heat_total_mw_unnormalised,
            p_shld_nuclear_heat_mw_unnormalised,
            p_tf_nuclear_heat_mw_unnormalised,
            f_ster_div_single,
            f_p_blkt_multiplication,
            p_neutron_total_mw,
            f_geom_cp,
            pnuc_cp_tf,
        )


class FirstWallRadiationPowers(ExplicitFunction):
    """cottax node: `calculate_first_wall_radiation_powers`."""

    p_fw_hcd_rad_total_mw = OutputInto(fwbs)
    p_fw_rad_total_mw = OutputInto(fwbs)
    psurffwo = OutputInto(fwbs)
    psurffwi = OutputInto(fwbs)

    def __call__(
        self,
        p_plasma_rad_mw=From(physics),
        f_a_fw_outboard_hcd=From(fwbs),
        p_div_rad_total_mw=From(fwbs),
        a_fw_outboard=From(first_wall),
        a_fw_total=From(first_wall),
        p_beam_orbit_loss_mw=From(current_drive),
        p_fw_alpha_mw=From(physics),
    ):
        return calculate_first_wall_radiation_powers(
            p_plasma_rad_mw,
            f_a_fw_outboard_hcd,
            p_div_rad_total_mw,
            a_fw_outboard,
            a_fw_total,
            p_beam_orbit_loss_mw,
            p_fw_alpha_mw,
        )


class PumpingPowerMechanicalWithPressureDrop(ExplicitFunction):
    """cottax node: `calculate_pumping_power_mechanical_with_pressure_drop`."""

    p_fw_blkt_coolant_pump_mw = OutputInto(primary_pumping)
    p_shld_coolant_pump_mw = OutputInto(heat_transport)
    p_div_coolant_pump_mw = OutputInto(heat_transport)

    def __call__(
        self,
        p_he=From(primary_pumping),
        dp_he=From(primary_pumping),
        gamma_he=From(primary_pumping),
        t_in_bb=From(primary_pumping),
        t_out_bb=From(primary_pumping),
        etaiso=From(fwbs),
        f_p_fw_blkt_pump=From(primary_pumping),
        p_fw_nuclear_heat_total_mw=From(fwbs),
        psurffwi=From(fwbs),
        psurffwo=From(fwbs),
        p_blkt_nuclear_heat_total_mw=From(fwbs),
        f_p_shld_coolant_pump_total_heat=From(heat_transport),
        p_shld_nuclear_heat_mw=From(fwbs),
        p_cp_shield_nuclear_heat_mw=From(fwbs),
        f_p_div_coolant_pump_total_heat=From(heat_transport),
        p_plasma_separatrix_mw=From(physics),
        p_div_nuclear_heat_total_mw=From(fwbs),
        p_div_rad_total_mw=From(fwbs),
    ):
        return calculate_pumping_power_mechanical_with_pressure_drop(
            p_he,
            dp_he,
            gamma_he,
            t_in_bb,
            t_out_bb,
            etaiso,
            f_p_fw_blkt_pump,
            p_fw_nuclear_heat_total_mw,
            psurffwi,
            psurffwo,
            p_blkt_nuclear_heat_total_mw,
            f_p_shld_coolant_pump_total_heat,
            p_shld_nuclear_heat_mw,
            p_cp_shield_nuclear_heat_mw,
            f_p_div_coolant_pump_total_heat,
            p_plasma_separatrix_mw,
            p_div_nuclear_heat_total_mw,
            p_div_rad_total_mw,
        )
