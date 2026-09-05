"""Pure-functional port of `process/models/blankets/hcpb.py`'s `CCFE_HCPB`.

Registry unit #13, **extended for the tokamak** (`_audit/tokamak_boundary.md`
§`.tokamak.ccfe_hcpb`, the single biggest attributable gap at 16 boundary reads). Read
`functional_process/_audit/units/models/blankets/hcpb.md` first -- it carries the
evidence for every read/write attribution below.

**What changed relative to the original three-function port.** Unit #13 ported
`nuclear_heating_blanket`, `nuclear_heating_shield` and `nuclear_heating_magnets` because
they were the sole blocker on the *stellarator*'s `st_fwbs` S2, where they are reachable
only through `blanket_neutronics()` -- a path with a live PROCESS `TypeError` (see the
record's open question #1). A tokamak reaches them directly from
`CCFE_HCPB.run()` (`caller.py:345`), so this file now ports the whole tokamak call path:
`run()`'s own renormalisation block, `component_masses`, `nuclear_heating_fw` and
`powerflow_calc` alongside the original three. Three things about the originals moved:

1. `.fwbs.f_a_fw_coolant_inboard`/`_outboard` are **no longer owned by the magnets
   node** -- `FirstWallCoolantVoidFractions` owns them. That is what dissolves the one
   real cycle in this model; see that class's docstring.
2. The four `nuclear_heating_*` nodes now own **minted `_unnormalised` names under
   `.ccfe_hcpb`**, not the real `.fwbs` fields. `run()` overwrites all four
   (`hcpb.py:220-264`) and the overwritten value is what every downstream consumer reads,
   so the raw and the final are two different quantities that PROCESS happens to store in
   one slot. The original record already flagged this for `p_tf_nuclear_heat_mw`; it is
   true of all four.
3. `itart` is no longer a traced argument selecting a branch with `jnp.where`. Under
   `next_steps.md` §14.2 a switch value selects an occupant class, so the shield and
   magnets functions are one function per arm.

**Switches met on this path, and what this port answers** (values from
`tests/regression/input_files/large_tokamak_eval.IN.DAT`, read out of the assembled
`DataStructure`, `tokamak_call_surface.md` §"The reference run"):

| switch | value here | ported | note |
|---|---|---|---|
| `.physics.itart` | 0 | **both arms** | `itart == 1` written 2026-08-27, see below |
| `.tfcoil.i_tf_sup` | 1 | the superconducting cell | new on this path 2026-08-27: the
centrepost chain reads it, and cuts it two different ways |
| `.divertor.n_divertors` | 1 | **both arms** | `== 2` written 2026-08-27, see below |
| `.fwbs.i_p_coolant_pumping` | 3 | `MECHANICAL_WITH_PRESSURE_DROP` | `0`/`1`
unported; `2` is CoolProp-bound |
| `.fwbs.i_blkt_coolant_type` | 1 (`HELIUM`) | only arm reachable | `run()` assigns
`HELIUM` unconditionally at `hcpb.py:45`, so `powerflow_calc`'s `WATER` arm (`:793`,
CoolProp) is **dead code** for `CCFE_HCPB`, not merely dormant |
| `.fwbs.i_blanket_type` | 1 (`CCFE_HCPB`) | this whole file | `== 5` routes to
`blankets/dcll.py` |

Nothing here reaches CoolProp: the only CoolProp site inside `CCFE_HCPB` is
`powerflow_calc:794`, behind the dead `WATER` arm above (`tokamak_call_surface.md` §D
records the same three modules as "dormant"; for this one it is stronger than dormant).

2026-08-27 (the double-null wave): this file's two `n_divertors` slots gained their
`== 2` occupants -- `DivertorSurfaceAndPlateMassDoubleNull` (`hcpb.py:360-361`, the
factor of two on `a_div_surface_total`) and
`NuclearHeatingRenormalisationDoubleNullConventional` (`hcpb.py:213-217`, the different
`f_geom_blanket`).

2026-08-27 (the centrepost wave): `itart == 1` is ported and registered. The chain
`hcpb.py:1008-1287` -- `st_cp_angle_fraction`, `st_tf_centrepost_fast_neut_flux`,
`st_centrepost_nuclear_heating` -- becomes one occupant,
`CentrepostNeutronicsSphericalTokamakSuperconducting`, and the renormalisation's
`(n_divertors, itart)` square gains its two `itart == 1` cells. Three consequences worth
carrying:

1. **`.tfcoil.i_tf_sup` is a switch of this file now.** Two of the three centrepost
   routines read it and *partition it differently* -- `{1}` vs `{0, 2}` for the neutron
   flux, `{2}` vs `{0, 1}` for the nuclear heating -- so the slot is keyed on a joint
   `(itart, i_tf_sup)` arm. Only the `(1, 1)` cell is written; both input files select
   it.
2. **Two mints.** `.ccfe_hcpb.f_geom_cp` is a `run()` local that crosses a node
   boundary; `.ccfe_hcpb.p_cp_shield_nuclear_heat_mw_fit` is the MCNP fit value PROCESS
   stores in `.fwbs.p_cp_shield_nuclear_heat_mw` at `:137` and overwrites at `:267`.
3. **`.build.r_sh_inboard_out` was a boundary input with no producer** when this
   cluster landed -- `build.py:1858` accumulates it outwards from the bore, a chain the
   port did not own. It does now: `models/build.py`'s
   `VacuumVesselAndShieldRadiiTfOutsideCs` (2026-08-29) owns those three lines, so this
   read has a real edge and the note stands only as history.
"""

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
    """cottax node: `calculate_fw_coolant_void_fractions`.

    The node that makes the rest of this file acyclic -- see the function's docstring.
    """

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

    `.divertor.a_div_surface_total` is read by `.costs.divertor_cost` -- one of the
    sixteen boundary variables this slot owes. Both arms are written (2026-08-27); the
    slot is total.
    """


class DivertorSurfaceAndPlateMassSingleNull(DivertorSurfaceAndPlateMass):
    """cottax node: `calculate_divertor_surface_and_plate_mass_single_null`.
    `n_divertors == 1`.
    """

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
    """cottax node: `calculate_divertor_surface_and_plate_mass_double_null`.
    `n_divertors == 2` -- live on `spherical_tokamak_eval.IN.DAT` and
    `st_regression.IN.DAT`.
    """

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
    """cottax node: `calculate_component_masses`.

    Unswitched: nothing left in this body branches once the divertor pair is elsewhere.
    Owns four of the slot's sixteen boundary variables -- `.fwbs.m_blkt_beryllium`,
    `.fwbs.m_blkt_li2o`, `.fwbs.m_blkt_steel_total` (all read by `.costs.blanket_cost`),
    `.fwbs.whtshld` (`.costs.shield_cost`, `.buildings.sizing`) and `.fwbs.wpenshld`
    (`.costs.shield_cost`).
    """

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

    Both arms are written and, since 2026-08-27, both are registered. They own the same
    nine fields and read unequal sets -- the conventional arm reads
    `.build.dr_blkt_inboard`, `.build.dr_shld_inboard` and `.tfcoil.m_tf_coils_total`;
    the spherical one reads `.tfcoil.whttflgs` instead of the last of those and
    neither of the first two, because a spherical machine's inboard blanket and shield
    are the centrepost's business.
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
    """Minted. `.fwbs.p_tf_nuclear_heat_mw` is the *renormalised* value
    (`hcpb.py:255-264`) and is owned by
    `NuclearHeatingRenormalisationSingleNullConventional`."""

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
    """cottax node: `calculate_nuclear_heating_magnets_spherical_tokamak`. `itart == 1`.

    Registered 2026-08-27, once the centrepost chain this machine also needs existed.
    """

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

    The arms own the same four fields and differ by one read:
    `.build.dr_shld_inboard` enters the average shield thickness on a conventional
    machine and does not on a spherical one, where the inboard shield is the centrepost's
    and is accounted separately. Both registered since 2026-08-27.
    """


class NuclearHeatingShieldConventional(NuclearHeatingShield):
    """cottax node: `nuclear_heating_shield_conventional`. `itart == 0`.

    `shield_density`/`x_blanket` are `NuclearHeatingMagnetsConventional`'s own outputs --
    an ordinary graph edge, magnets before shield, matching the call order
    `CCFE_HCPB.run()` uses (`hcpb.py:155` then `:174`).
    """

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
    """cottax node: `nuclear_heating_shield_spherical_tokamak`. `itart == 1`.

    Registered 2026-08-27, with the centrepost chain.
    """

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

    **The arms do not own the same set, and the difference is one field.**
    `CentrepostNeutronicsAbsent` owns `.fwbs.p_cp_shield_nuclear_heat_mw`, because on the
    conventional arm the two writes PROCESS makes to it (`:146` and `:267`) are both
    `0.0` -- a `redundant-duplicate-write` resolved by picking one owner. On a spherical
    machine they differ, `:267` wins, and the field belongs to the renormalisation
    occupant instead; this family's spherical member mints the earlier value as
    `.ccfe_hcpb.p_cp_shield_nuclear_heat_mw_fit`. Partial overlap by construction, the
    same shape `.fwbs.i_p_coolant_pumping`'s arms have in this file.

    **Why the arm is joint.** `st_tf_centrepost_fast_neut_flux` splits `i_tf_sup` as
    `{1}` against `{0, 2}` (`hcpb.py:1114`); `st_centrepost_nuclear_heating` splits it as
    `{2}` against `{0, 1}` (`:1192`). Two different partitions of one switch inside one
    straight-line block, so no single integer names the occupant and
    `indat._centrepost_neutronics_arm` derives one from the pair.
    """


class CentrepostNeutronicsAbsent(CentrepostNeutronics, StatesValues):
    """cottax node: `calculate_centrepost_neutronics_absent`. `itart == 0`.

    Reads nothing -- the same shape as `i_pulsed_plant`'s unpulsed occupant
    (`next_steps.md` §14.4), and legitimate for the same reason: on this arm PROCESS's
    own source is four literal assignments.

    The four are **stated** rather than produced in the body: a zero built during the
    trace is a compile-time constant, and §25 measured XLA deleting the subexpressions
    such a zero multiplies (`models/stated.py`, `_audit/optimise_design.md` §28, §34).
    The unit (`calculate_centrepost_neutronics_absent`) still states them, through
    `indat.STATED_VALUES`.

    `carried_all` used to hold the four together, because the ported function hands them
    back as one tuple and four fields would have restated four literals the unit already
    states. Stating them keeps that: the four are one `default_factory` row in
    `indat.STATED_VALUES`, zipped onto the four outputs in declaration order.
    """

    pnuc_cp_tf = OutputInto(fwbs)
    p_cp_shield_nuclear_heat_mw = OutputInto(fwbs)
    pnuc_cp = OutputInto(fwbs)
    neut_flux_cp = OutputInto(fwbs)


class CentrepostNeutronicsSphericalTokamakSuperconducting(CentrepostNeutronics):
    """cottax node: `calculate_centrepost_neutronics_spherical_tokamak_superconducting`.
    `itart == 1` and `i_tf_sup == 1`.

    Owns the mint `.ccfe_hcpb.f_geom_cp` -- a *local* in PROCESS's `run()` (`:120`) that
    two later statements read (`:216`, `:268`). Once those statements are a different
    node, the local crosses a node boundary and has to be named.

    `.build.r_sh_inboard_out` is a **boundary input with no producer in this port**; see
    the pure function's docstring for why `.build.r_shld_inboard_inner` is not a
    substitute for it.
    """

    f_geom_cp = OutputInto(ccfe_hcpb)
    """Minted. A local of `run()` in PROCESS, read by the renormalisation."""
    neut_flux_cp = OutputInto(fwbs)
    pnuc_cp_tf = OutputInto(fwbs)
    p_cp_shield_nuclear_heat_mw_fit = OutputInto(ccfe_hcpb)
    """Minted. `.fwbs.p_cp_shield_nuclear_heat_mw` is `run():267`'s *later* value and is
    owned by the spherical renormalisation; this is the MCNP fit PROCESS overwrites."""
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

    Owns four of the slot's sixteen boundary variables:
    `.fwbs.p_fw_nuclear_heat_total_mw`, `.fwbs.p_blkt_nuclear_heat_total_mw`,
    `.fwbs.p_shld_nuclear_heat_mw` and `.fwbs.p_tf_nuclear_heat_mw`.

    **`.fwbs.p_tf_nuclear_heat_mw` has a second producer in the tree**, and it is worth
    naming rather than discovering later: `models/stellarator/tf_nuclear_heating.py`'s
    `ScTfCoilNuclearHeating` owns the same `VarPath`. There is no conflict on a
    tokamak -- that node is a slot of `Stellarator`'s `blktmodel`/`ipowerflow` switch
    (`models/stellarator/namespace.py:186`) and a `TokamakProcess` has no `Stellarator`
    namespace at all -- but the two are alternative producers of one field on two
    different devices, and any future machine that assembled both would have to choose.

    **All four cells are written** since 2026-08-27 (the centrepost wave). The two
    `itart == 1` cells own a fifth field the conventional two do not,
    `.fwbs.p_cp_shield_nuclear_heat_mw`, and read two more,
    `.ccfe_hcpb.f_geom_cp` and `.fwbs.pnuc_cp_tf`; see
    `calculate_nuclear_heating_renormalisation_single_null_spherical_tokamak` for why
    that is the same fact three times over.
    """


class NuclearHeatingRenormalisationSingleNullConventional(NuclearHeatingRenormalisation):
    """cottax node: `calculate_nuclear_heating_renormalisation_single_null_conventional`.
    `n_divertors == 1` and `itart == 0`.
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
    """cottax node: `calculate_nuclear_heating_renormalisation_double_null_conventional`.
    `n_divertors == 2` and `itart == 0`.

    Written 2026-08-27 with the rest of the double-null wave. Note that the two
    spherical-tokamak input files that motivated that wave do **not** select it: they
    set `itart = 1`, so this slot refuses on `('itart_hcpb', 1)` before `n_divertors` is
    consulted. A conventional-aspect-ratio double-null machine is what reaches it.
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
    `n_divertors == 1` and `itart == 1`.

    Owns `.fwbs.p_cp_shield_nuclear_heat_mw`, which the conventional arms leave to
    `CentrepostNeutronicsAbsent`, and reads `.ccfe_hcpb.f_geom_cp` and
    `.fwbs.pnuc_cp_tf` from `CentrepostNeutronicsSphericalTokamakSuperconducting`.

    Written with its double-null sibling; neither spherical-tokamak input file selects
    *this* cell (both are `i_single_null = 0`), but `hcpb.py:215`'s
    `n_divertors * f_ster_div_single` is a multiplication rather than a branch, so
    writing one cell of the row and not the other would leave a hole with no argument
    behind it.
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
    `n_divertors == 2` and `itart == 1`.

    **The cell `spherical_tokamak_eval.IN.DAT` and `st_regression.IN.DAT` select.**
    Both set `itart = 1` and `i_single_null = 0`.
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
    """cottax node: `calculate_first_wall_radiation_powers`. Unswitched.

    Owns two of the slot's sixteen boundary variables, `.fwbs.p_fw_hcd_rad_total_mw` and
    `.fwbs.p_fw_rad_total_mw`.
    """

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
    """cottax node: `calculate_pumping_power_mechanical_with_pressure_drop`.

    `.fwbs.i_p_coolant_pumping == 3` (`MECHANICAL_WITH_PRESSURE_DROP`). Owns two of the
    slot's sixteen boundary variables (`.heat_transport.p_shld_coolant_pump_mw`,
    `.heat_transport.p_div_coolant_pump_mw`) and
    `.primary_pumping.p_fw_blkt_coolant_pump_mw`, which is what this arm produces
    *instead of* the other two the boundary table asks for.
    """

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
