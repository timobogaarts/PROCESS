"""Pure-functional port of a self-contained subset of `process/models/costs/costs.py`
(the 1990-style cost model, registry unit #18).

**Scope note**: `unit_registry.md` row 18 nominally scopes this unit to the whole of
`Costs.run()`/`.output()` (43 methods, ~3000 lines). Per this dispatch's own guidance
("costs code is likely to be extensively entangled... it's entirely plausible most of
your output this pass is audit records, not ported code") and the standing practice of
porting what is genuinely self-contained rather than an entire file at once (see
`physics.py`'s chunked treatment), this file ports 23 of the 43 methods: every one that is
loop-free (or has only a Python-level, compile-time-constant-length loop — none of the
23 do), calls no other `Model`, and needs no `scipy`. See `costs.md` for the full
per-method audit, including the remaining 20 methods (audit-only: TF/PF magnet costs,
power injection, first-wall/blanket/shield costs, thermal storage, `coelc`, and the
dynamic-length `n_cs_pf_coils` loop in `acc2222`, none of which are blockers of the
finding this dispatch was mainly sent to resolve — see the switches note below).

**`i_cost_model` finding (the main ask of this dispatch)**: `.costs.i_cost_model` is
**never read inside this file or `costs_2015.py`** (confirmed by grep — zero hits in
either file). It is resolved one layer up, in `process/main.py`'s `Models.costs`
`@property`, which picks a whole `Model` instance (`Costs()`/`Costs2015()`/a custom
model) *before* any model runs — exactly the precedent `_audit/schema.md`'s own
`## switches touched` template names ("a `@property` on `Models` picking a model
instance before any model runs -- see `i_cost_model` / `Models.costs` in
`process/main.py` for the precedent"). `stellarator.py` itself never branches on
`i_cost_model` either: it calls `self.costs.run()`/`.output()` on whatever was already
injected. So this **is** a genuine topology-changing switch, and the two arms are
**disjoint subgraphs, not a shared-body-with-a-branch case**: `costs.py` writes 114
distinct `.costs.*` fields, `costs_2015.py` writes only `.costs_2015.s_cost`/`s_cref`/
`s_k`/`s_kref`/`s_cost_factor` (a 100-slot array) plus a handful of `.costs_2015.*`
scalars -- the *only* two `VarPath`s both files write are `.costs.coe` and
`.costs.concost`, PROCESS's own two "final" cost outputs that feed the objective
function and every other unit's cost-dependent read. That is exactly the shape
`configuration.py`'s `Switch.check_arms_are_exclusive` wants from a real pair of
`Alternative`s (they must own at least one output in common, or they are not
alternatives at all) -- so this is the third real `TOPOLOGY_SWITCHES` entry after
`isthtr`/`ipowerflow`/`i_plasma_pedestal`, confirming `next_steps.md` §4c's prediction.
Not wired into `total_process.TOPOLOGY_SWITCHES` here -- that is reserved for the
consolidation pass, per this dispatch's boundary.
"""

import equinox as eqx
import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    FromExactly,
    OutputInto,
)

from functional_process.models.costs.costs import (
    calculate_atmospheric_recovery_cost,
    calculate_auxiliary_component_cooling_cost,
    calculate_auxiliary_component_cooling_cost_magnetic_confinement,
    calculate_auxiliary_facility_power_cost,
    calculate_blanket_cost,
    calculate_blanket_cost_magnetic_confinement,
    calculate_constructed_cost,
    calculate_cost_of_electricity,
    calculate_cost_of_electricity_conventional_aspect_ratio,
    calculate_cost_of_electricity_spherical_tokamak,
    calculate_cryogenic_system_cost,
    calculate_diesel_generators_cost,
    calculate_divertor_cost,
    calculate_electric_plant_equipment_cost,
    calculate_energy_storage_cost,
    calculate_energy_storage_cost_electrowatt_option_1,
    calculate_energy_storage_cost_electrowatt_option_2,
    calculate_first_wall_cost,
    calculate_first_wall_cost_magnetic_confinement,
    calculate_fuel_handling_cost,
    calculate_fuel_processing_cost,
    calculate_fuel_processing_cost_magnetic_confinement,
    calculate_fuelling_system_cost,
    calculate_fusion_power_island_cost,
    calculate_heat_rejection_cost,
    calculate_heat_transport_system_cost,
    calculate_indirect_costs,
    calculate_instrumentation_and_control_cost,
    calculate_low_voltage_cost,
    calculate_magnets_cost,
    calculate_maintenance_equipment_cost,
    calculate_misc_plant_equipment_cost,
    calculate_nuclear_building_ventilation_cost,
    calculate_pf_coil_power_conditioning_cost,
    calculate_pf_magnet_cost,
    calculate_pf_magnet_cost_per_kam,
    calculate_pf_magnet_cost_per_kam_no_central_solenoid,
    calculate_pf_magnet_cost_per_kg,
    calculate_pf_magnet_cost_per_kg_no_central_solenoid,
    calculate_power_conditioning_cost,
    calculate_power_injection_cost,
    calculate_power_injection_cost_magnetic_confinement,
    calculate_reactor_cooling_system_cost,
    calculate_reactor_cost,
    calculate_reactor_structure_cost,
    calculate_shield_cost,
    calculate_shield_cost_magnetic_confinement,
    calculate_structures_cost,
    calculate_switchyard_cost,
    calculate_tf_coil_power_conditioning_cost,
    calculate_tf_magnet_cost_resistive,
    calculate_tf_magnet_cost_superconducting,
    calculate_tf_magnet_cost_superconducting_per_kam,
    calculate_tf_magnet_cost_superconducting_per_kg,
    calculate_total_plant_direct_cost,
    calculate_transformers_cost,
    calculate_turbine_plant_equipment_cost,
    calculate_vacuum_system_cost,
    calculate_vacuum_vessel_assembly_cost,
    convert_fpy_to_calendar,
)
from functional_process.cottax.pfcoil.masses import (
    I_CS_SUPERCONDUCTOR,
    I_CS_SUPERCONDUCTOR_WST_NB3SN,
    I_PF_SUPERCONDUCTOR,
    I_PF_SUPERCONDUCTOR_HAZELTON_ZHAI_REBCO,
)
from functional_process.models.safe_math import safe_pow, safe_sqrt
from functional_process.cottax.stated import StatesValues
from functional_process.cottax.paths import (
    buildings,
    costs,
    current_drive,
    divertor,
    first_wall,
    fwbs,
    heat_transport,
    ife,
    pf_coil,
    pf_power,
    physics,
    structure,
    tfcoil,
    times,
    vacuum,
)
from functional_process.vocabulary import PFConductorModel

# ruff's docstring rules treat `__all__` membership as the definition of "public"
# once one is present, so this lists every public name this module resolved before
# step 2 of `_audit/formulas_split.md` moved the pure functions out -- not just the
# handful actually unused here (see `power/electric_production.py`'s commit for why
# a partial list is the wrong move).

__all__ = [
    "I_CS_SUPERCONDUCTOR",
    "I_CS_SUPERCONDUCTOR_WST_NB3SN",
    "I_PF_SUPERCONDUCTOR",
    "I_PF_SUPERCONDUCTOR_HAZELTON_ZHAI_REBCO",
    "AtmosphericRecoveryCost",
    "AuxiliaryComponentCoolingCost",
    "AuxiliaryFacilityPowerCost",
    "BlanketCost",
    "ConstructedCost",
    "ConvertFpyToCalendar",
    "CostOfElectricity",
    "CostOfElectricityConventionalAspectRatio",
    "CostOfElectricitySphericalTokamak",
    "CryogenicSystemCost",
    "DieselGeneratorsCost",
    "DivertorCost",
    "ElectricPlantEquipmentCost",
    "EnergyStorageCost",
    "EnergyStorageCostPulsed",
    "EnergyStorageCostPulsedElectrowattOption1",
    "EnergyStorageCostPulsedElectrowattOption2",
    "EnergyStorageCostUnpulsed",
    "ExplicitFunction",
    "FirstWallCost",
    "From",
    "FromExactly",
    "FuelHandlingCost",
    "FuelProcessingCost",
    "FuellingSystemCost",
    "FusionPowerIslandCost",
    "HeatRejectionCost",
    "HeatTransportSystemCost",
    "IndirectCosts",
    "InstrumentationAndControlCost",
    "LowVoltageCost",
    "MagnetsCost",
    "MaintenanceEquipmentCost",
    "MiscPlantEquipmentCost",
    "NuclearBuildingVentilationCost",
    "OutputInto",
    "PFConductorModel",
    "PfCoilPowerConditioningCost",
    "PfMagnetCost",
    "PfMagnetCostPerKam",
    "PfMagnetCostPerKamNoCentralSolenoid",
    "PfMagnetCostPerKg",
    "PfMagnetCostPerKgCsWstNb3Sn",
    "PfMagnetCostPerKgNoCentralSolenoid",
    "PfMagnetCostPerKgWithCentralSolenoid",
    "PowerConditioningCost",
    "PowerInjectionCost",
    "ReactorCoolingSystemCost",
    "ReactorCost",
    "ReactorStructureCost",
    "ShieldCost",
    "StatesValues",
    "StructuresCost",
    "SwitchyardCost",
    "TfCoilPowerConditioningCost",
    "TfMagnetCostResistive",
    "TfMagnetCostSuperconducting",
    "TfMagnetCostSuperconductingPerKam",
    "TfMagnetCostSuperconductingPerKg",
    "TotalPlantDirectCost",
    "TransformersCost",
    "TurbinePlantEquipmentCost",
    "VacuumSystemCost",
    "VacuumVesselAssemblyCost",
    "buildings",
    "calculate_atmospheric_recovery_cost",
    "calculate_auxiliary_component_cooling_cost",
    "calculate_auxiliary_component_cooling_cost_magnetic_confinement",
    "calculate_auxiliary_facility_power_cost",
    "calculate_blanket_cost",
    "calculate_blanket_cost_magnetic_confinement",
    "calculate_constructed_cost",
    "calculate_cost_of_electricity",
    "calculate_cost_of_electricity_conventional_aspect_ratio",
    "calculate_cost_of_electricity_spherical_tokamak",
    "calculate_cryogenic_system_cost",
    "calculate_diesel_generators_cost",
    "calculate_divertor_cost",
    "calculate_electric_plant_equipment_cost",
    "calculate_energy_storage_cost",
    "calculate_energy_storage_cost_electrowatt_option_1",
    "calculate_energy_storage_cost_electrowatt_option_2",
    "calculate_first_wall_cost",
    "calculate_first_wall_cost_magnetic_confinement",
    "calculate_fuel_handling_cost",
    "calculate_fuel_processing_cost",
    "calculate_fuel_processing_cost_magnetic_confinement",
    "calculate_fuelling_system_cost",
    "calculate_fusion_power_island_cost",
    "calculate_heat_rejection_cost",
    "calculate_heat_transport_system_cost",
    "calculate_indirect_costs",
    "calculate_instrumentation_and_control_cost",
    "calculate_low_voltage_cost",
    "calculate_magnets_cost",
    "calculate_maintenance_equipment_cost",
    "calculate_misc_plant_equipment_cost",
    "calculate_nuclear_building_ventilation_cost",
    "calculate_pf_coil_power_conditioning_cost",
    "calculate_pf_magnet_cost",
    "calculate_pf_magnet_cost_per_kam",
    "calculate_pf_magnet_cost_per_kam_no_central_solenoid",
    "calculate_pf_magnet_cost_per_kg",
    "calculate_pf_magnet_cost_per_kg_no_central_solenoid",
    "calculate_power_conditioning_cost",
    "calculate_power_injection_cost",
    "calculate_power_injection_cost_magnetic_confinement",
    "calculate_reactor_cooling_system_cost",
    "calculate_reactor_cost",
    "calculate_reactor_structure_cost",
    "calculate_shield_cost",
    "calculate_shield_cost_magnetic_confinement",
    "calculate_structures_cost",
    "calculate_switchyard_cost",
    "calculate_tf_coil_power_conditioning_cost",
    "calculate_tf_magnet_cost_resistive",
    "calculate_tf_magnet_cost_superconducting",
    "calculate_tf_magnet_cost_superconducting_per_kam",
    "calculate_tf_magnet_cost_superconducting_per_kg",
    "calculate_total_plant_direct_cost",
    "calculate_transformers_cost",
    "calculate_turbine_plant_equipment_cost",
    "calculate_vacuum_system_cost",
    "calculate_vacuum_vessel_assembly_cost",
    "convert_fpy_to_calendar",
    "costs",
    "current_drive",
    "divertor",
    "eqx",
    "first_wall",
    "fwbs",
    "heat_transport",
    "ife",
    "jnp",
    "pf_coil",
    "pf_power",
    "physics",
    "safe_pow",
    "safe_sqrt",
    "structure",
    "tfcoil",
    "times",
    "vacuum",
]


class ConvertFpyToCalendar(ExplicitFunction):
    """cottax node: `convert_fpy_to_calendar`."""

    life_blkt = OutputInto(fwbs)
    cdrlife_cal = OutputInto(costs)
    life_div = OutputInto(costs)
    cplife_cal = OutputInto(costs)

    def __call__(
        self,
        life_blkt_fpy=From(fwbs),
        life_plant=From(costs),
        f_t_plant_available=From(costs),
        life_div_fpy=From(costs),
        itart=From(physics),
        cplife=From(costs),
    ):
        return convert_fpy_to_calendar(
            life_blkt_fpy, life_plant, f_t_plant_available, life_div_fpy, itart, cplife
        )


class StructuresCost(ExplicitFunction):
    """cottax node: `calculate_structures_cost` (Account 21)."""

    c211 = OutputInto(costs)
    c212 = OutputInto(costs)
    c213 = OutputInto(costs)
    c2141 = OutputInto(costs)
    c2142 = OutputInto(costs)
    c214 = OutputInto(costs)
    c215 = OutputInto(costs)
    c216 = OutputInto(costs)
    c2171 = OutputInto(costs)
    c2172 = OutputInto(costs)
    c2173 = OutputInto(costs)
    c2174 = OutputInto(costs)
    c217 = OutputInto(costs)
    c21 = OutputInto(costs)

    def __call__(
        self,
        csi=From(costs),
        lsa=From(costs),
        cland=From(costs),
        ucrb=From(costs),
        rbvol=From(buildings),
        UCMB=From(costs),
        rmbvol=From(buildings),
        UCWS=From(costs),
        wsvol=From(buildings),
        UCTR=From(costs),
        triv=From(buildings),
        UCEL=From(costs),
        elevol=From(buildings),
        UCAD=From(costs),
        admvol=From(buildings),
        UCCO=From(costs),
        convol=From(buildings),
        UCSH=From(costs),
        shovol=From(buildings),
        UCCR=From(costs),
        cryvol=From(buildings),
        ireactor=From(costs),
        cturbb=From(costs),
    ):
        return calculate_structures_cost(
            csi,
            lsa,
            cland,
            ucrb,
            rbvol,
            UCMB,
            rmbvol,
            UCWS,
            wsvol,
            UCTR,
            triv,
            UCEL,
            elevol,
            UCAD,
            admvol,
            UCCO,
            convol,
            UCSH,
            shovol,
            UCCR,
            cryvol,
            ireactor,
            cturbb,
        )


class IndirectCosts(ExplicitFunction):
    """cottax node: `calculate_indirect_costs` (Account 9)."""

    cindrt = OutputInto(costs)
    ccont = OutputInto(costs)

    def __call__(
        self,
        cfind=From(costs),
        lsa=From(costs),
        cdirt=From(costs),
        cowner=From(costs),
        fcontng=From(costs),
    ):
        return calculate_indirect_costs(cfind, lsa, cdirt, cowner, fcontng)


class ReactorStructureCost(ExplicitFunction):
    """cottax node: `calculate_reactor_structure_cost` (Account 221.4)."""

    c2214 = OutputInto(costs)

    def __call__(
        self,
        gsmass=From(structure),
        UCGSS=From(costs),
        lsa=From(costs),
        fkind=From(costs),
    ):
        return calculate_reactor_structure_cost(gsmass, UCGSS, lsa, fkind)


class VacuumVesselAssemblyCost(ExplicitFunction):
    """cottax node: `calculate_vacuum_vessel_assembly_cost` (Account 222.3)."""

    c2223 = OutputInto(costs)

    def __call__(
        self,
        m_vv=From(fwbs),
        uccryo=From(costs),
        lsa=From(costs),
        fkind=From(costs),
    ):
        return calculate_vacuum_vessel_assembly_cost(m_vv, uccryo, lsa, fkind)


class DivertorCost(ExplicitFunction):
    """cottax node: `calculate_divertor_cost` (Account 221.5)."""

    c2215 = OutputInto(costs)
    divcst = OutputInto(costs)

    def __call__(
        self,
        ife=From(ife),
        a_div_surface_total=From(divertor),
        ucdiv=From(costs),
        fkind=From(costs),
        ifueltyp=From(costs),
    ):
        return calculate_divertor_cost(ife, a_div_surface_total, ucdiv, fkind, ifueltyp)


class VacuumSystemCost(ExplicitFunction):
    """cottax node: `calculate_vacuum_system_cost` (Account 224)."""

    c2241 = OutputInto(costs)
    c2242 = OutputInto(costs)
    c2243 = OutputInto(costs)
    c2244 = OutputInto(costs)
    c2245 = OutputInto(costs)
    c2246 = OutputInto(costs)
    c224 = OutputInto(costs)

    def __call__(
        self,
        i_vacuum_pump_type=From(vacuum),
        n_vac_pumps_high=From(vacuum),
        UCCPMP=From(costs),
        UCTPMP=From(costs),
        n_vv_vacuum_ducts=From(vacuum),
        UCBPMP=From(costs),
        dlscal=From(vacuum),
        UCDUCT=From(costs),
        dia_vv_vacuum_ducts=From(vacuum),
        UCVALV=From(costs),
        m_vv_vacuum_duct_shield=From(vacuum),
        UCVDSH=From(costs),
        UCVIAC=From(costs),
        fkind=From(costs),
    ):
        return calculate_vacuum_system_cost(
            i_vacuum_pump_type,
            n_vac_pumps_high,
            UCCPMP,
            UCTPMP,
            n_vv_vacuum_ducts,
            UCBPMP,
            dlscal,
            UCDUCT,
            dia_vv_vacuum_ducts,
            UCVALV,
            m_vv_vacuum_duct_shield,
            UCVDSH,
            UCVIAC,
            fkind,
        )


class TfCoilPowerConditioningCost(ExplicitFunction):
    """cottax node: `calculate_tf_coil_power_conditioning_cost` (Account 225.1)."""

    c22511 = OutputInto(costs)
    c22512 = OutputInto(costs)
    c22513 = OutputInto(costs)
    c22514 = OutputInto(costs)
    c22515 = OutputInto(costs)
    c2251 = OutputInto(costs)

    def __call__(
        self,
        uctfps=From(costs),
        tfckw=From(tfcoil),
        tfcmw=From(tfcoil),
        i_tf_sup=From(tfcoil),
        uctfbr=From(costs),
        n_tf_coils=From(tfcoil),
        c_tf_turn=From(tfcoil),
        v_tf_coil_dump_quench_kv=From(tfcoil),
        uctfsw=From(costs),
        UCTFDR=From(costs),
        e_tf_magnetic_stored_total_gj=From(tfcoil),
        UCTFGR=From(costs),
        UCTFIC=From(costs),
        uctfbus=From(costs),
        m_tf_bus=From(tfcoil),
        ucbus=From(costs),
        len_tf_bus=From(tfcoil),
        fkind=From(costs),
    ):
        return calculate_tf_coil_power_conditioning_cost(
            uctfps,
            tfckw,
            tfcmw,
            i_tf_sup,
            uctfbr,
            n_tf_coils,
            c_tf_turn,
            v_tf_coil_dump_quench_kv,
            uctfsw,
            UCTFDR,
            e_tf_magnetic_stored_total_gj,
            UCTFGR,
            UCTFIC,
            uctfbus,
            m_tf_bus,
            ucbus,
            len_tf_bus,
            fkind,
        )


class PfCoilPowerConditioningCost(ExplicitFunction):
    """cottax node: `calculate_pf_coil_power_conditioning_cost` (Account 225.2)."""

    c22521 = OutputInto(costs)
    c22522 = OutputInto(costs)
    c22523 = OutputInto(costs)
    c22524 = OutputInto(costs)
    c22525 = OutputInto(costs)
    c22526 = OutputInto(costs)
    c22527 = OutputInto(costs)
    c2252 = OutputInto(costs)

    def __call__(
        self,
        ucpfps=From(costs),
        peakmva=From(heat_transport),
        ucpfic=From(costs),
        pfckts=From(pf_power),
        ucpfb=From(costs),
        spfbusl=From(pf_power),
        acptmax=From(pf_power),
        ucpfbs=From(costs),
        srcktpm=From(pf_power),
        ucpfbk=From(costs),
        vpfskv=From(pf_power),
        ucpfdr1=From(costs),
        ensxpfm=From(pf_power),
        ucpfcb=From(costs),
        fkind=From(costs),
    ):
        return calculate_pf_coil_power_conditioning_cost(
            ucpfps,
            peakmva,
            ucpfic,
            pfckts,
            ucpfb,
            spfbusl,
            acptmax,
            ucpfbs,
            srcktpm,
            ucpfbk,
            vpfskv,
            ucpfdr1,
            ensxpfm,
            ucpfcb,
            fkind,
        )


class ReactorCoolingSystemCost(ExplicitFunction):
    """cottax node: `calculate_reactor_cooling_system_cost` (Account 2261)."""

    cpp = OutputInto(costs)
    chx = OutputInto(costs)
    c2261 = OutputInto(costs)

    def __call__(
        self,
        uchts=From(costs),
        i_blkt_coolant_type=From(fwbs),
        p_fw_div_heat_deposited_mw=From(heat_transport),
        p_blkt_nuclear_heat_total_mw=From(fwbs),
        p_shld_nuclear_heat_mw=From(fwbs),
        lsa=From(costs),
        fkind=From(costs),
        UCPHX=From(costs),
        n_primary_heat_exchangers=From(heat_transport),
        p_plant_primary_heat_mw=From(heat_transport),
    ):
        return calculate_reactor_cooling_system_cost(
            uchts,
            i_blkt_coolant_type,
            p_fw_div_heat_deposited_mw,
            p_blkt_nuclear_heat_total_mw,
            p_shld_nuclear_heat_mw,
            lsa,
            fkind,
            UCPHX,
            n_primary_heat_exchangers,
            p_plant_primary_heat_mw,
        )


class FuellingSystemCost(ExplicitFunction):
    """cottax node: `calculate_fuelling_system_cost` (Account 2271)."""

    c2271 = OutputInto(costs)

    def __call__(
        self,
        ucf1=From(costs),
        fkind=From(costs),
    ):
        return calculate_fuelling_system_cost(ucf1, fkind)


class NuclearBuildingVentilationCost(ExplicitFunction):
    """cottax node: `calculate_nuclear_building_ventilation_cost` (Account 2274)."""

    c2274 = OutputInto(costs)

    def __call__(
        self,
        UCNBV=From(costs),
        volrci=From(buildings),
        wsvol=From(buildings),
        fkind=From(costs),
    ):
        return calculate_nuclear_building_ventilation_cost(UCNBV, volrci, wsvol, fkind)


class InstrumentationAndControlCost(ExplicitFunction):
    """cottax node: `calculate_instrumentation_and_control_cost` (Account 228)."""

    c228 = OutputInto(costs)

    def __call__(
        self,
        uciac=From(costs),
        fkind=From(costs),
    ):
        return calculate_instrumentation_and_control_cost(uciac, fkind)


class MaintenanceEquipmentCost(ExplicitFunction):
    """cottax node: `calculate_maintenance_equipment_cost` (Account 229)."""

    c229 = OutputInto(costs)

    def __call__(
        self,
        ucme=From(costs),
        fkind=From(costs),
    ):
        return calculate_maintenance_equipment_cost(ucme, fkind)


class TurbinePlantEquipmentCost(ExplicitFunction):
    """cottax node: `calculate_turbine_plant_equipment_cost` (Account 23)."""

    c23 = OutputInto(costs)

    def __call__(
        self,
        ireactor=From(costs),
        ucturb=From(costs),
        i_blkt_coolant_type=From(fwbs),
        p_plant_electric_gross_mw=From(heat_transport),
    ):
        return calculate_turbine_plant_equipment_cost(
            ireactor, ucturb, i_blkt_coolant_type, p_plant_electric_gross_mw
        )


class SwitchyardCost(ExplicitFunction):
    """cottax node: `calculate_switchyard_cost` (Account 241)."""

    c241 = OutputInto(costs)

    def __call__(
        self,
        UCSWYD=From(costs),
        lsa=From(costs),
    ):
        return calculate_switchyard_cost(UCSWYD, lsa)


class TransformersCost(ExplicitFunction):
    """cottax node: `calculate_transformers_cost` (Account 242)."""

    c242 = OutputInto(costs)

    def __call__(
        self,
        UCPP=From(costs),
        pacpmw=From(heat_transport),
        UCAP=From(costs),
        p_plant_electric_base_total_mw=From(heat_transport),
        lsa=From(costs),
    ):
        return calculate_transformers_cost(
            UCPP, pacpmw, UCAP, p_plant_electric_base_total_mw, lsa
        )


class LowVoltageCost(ExplicitFunction):
    """cottax node: `calculate_low_voltage_cost` (Account 243)."""

    c243 = OutputInto(costs)

    def __call__(
        self,
        UCLV=From(costs),
        tlvpmw=From(heat_transport),
        lsa=From(costs),
    ):
        return calculate_low_voltage_cost(UCLV, tlvpmw, lsa)


class DieselGeneratorsCost(ExplicitFunction):
    """cottax node: `calculate_diesel_generators_cost` (Account 244)."""

    c244 = OutputInto(costs)

    def __call__(
        self,
        UCDGEN=From(costs),
        lsa=From(costs),
    ):
        return calculate_diesel_generators_cost(UCDGEN, lsa)


class AuxiliaryFacilityPowerCost(ExplicitFunction):
    """cottax node: `calculate_auxiliary_facility_power_cost` (Account 245)."""

    c245 = OutputInto(costs)

    def __call__(
        self,
        UCAF=From(costs),
        lsa=From(costs),
    ):
        return calculate_auxiliary_facility_power_cost(UCAF, lsa)


class ElectricPlantEquipmentCost(ExplicitFunction):
    """cottax node: `calculate_electric_plant_equipment_cost` (Account 24, total).

    Reads the five sub-account nodes' own outputs -- an ordinary graph edge, matching
    `Costs.acc24`'s own call order (`acc241`..`acc245` before `acc24`).
    """

    c24 = OutputInto(costs)

    def __call__(
        self,
        c241=From(costs),
        c242=From(costs),
        c243=From(costs),
        c244=From(costs),
        c245=From(costs),
    ):
        return calculate_electric_plant_equipment_cost(c241, c242, c243, c244, c245)


class MiscPlantEquipmentCost(ExplicitFunction):
    """cottax node: `calculate_misc_plant_equipment_cost` (Account 25)."""

    c25 = OutputInto(costs)

    def __call__(
        self,
        ucmisc=From(costs),
        lsa=From(costs),
    ):
        return calculate_misc_plant_equipment_cost(ucmisc, lsa)


class HeatRejectionCost(ExplicitFunction):
    """cottax node: `calculate_heat_rejection_cost` (Account 26)."""

    c26 = OutputInto(costs)

    def __call__(
        self,
        ireactor=From(costs),
        p_fusion_total_mw=From(physics),
        p_hcd_electric_total_mw=From(heat_transport),
        tfcmw=From(tfcoil),
        p_plant_primary_heat_mw=From(heat_transport),
        p_plant_electric_gross_mw=From(heat_transport),
        uchrs=From(costs),
        lsa=From(costs),
    ):
        return calculate_heat_rejection_cost(
            ireactor,
            p_fusion_total_mw,
            p_hcd_electric_total_mw,
            tfcmw,
            p_plant_primary_heat_mw,
            p_plant_electric_gross_mw,
            uchrs,
            lsa,
        )


# --------------------------------------------------------------------------------------
# Second porting wave's nodes: the `.costs.coe` chain. Registered in `total_process.py`
# under `.costs.i_cost_model == 0`; see that switch's own comment block.
# --------------------------------------------------------------------------------------


class FirstWallCost(ExplicitFunction):
    """cottax node: `calculate_first_wall_cost` (Account 221.1)."""

    c2211 = OutputInto(costs)
    fwallcst = OutputInto(costs)

    def __call__(
        self,
        lsa=From(costs),
        UCFWA=From(costs),
        UCFWS=From(costs),
        a_fw_total=From(first_wall),
        UCFWPS=From(costs),
        fkind=From(costs),
        ifueltyp=From(costs),
    ):
        return calculate_first_wall_cost_magnetic_confinement(
            lsa, UCFWA, UCFWS, a_fw_total, UCFWPS, fkind, ifueltyp
        )


class BlanketCost(ExplicitFunction):
    """cottax node: `calculate_blanket_cost` (Account 221.2)."""

    c22121 = OutputInto(costs)
    c22122 = OutputInto(costs)
    c22123 = OutputInto(costs)
    c22124 = OutputInto(costs)
    c22125 = OutputInto(costs)
    c22126 = OutputInto(costs)
    c22127 = OutputInto(costs)
    c2212 = OutputInto(costs)
    blkcst = OutputInto(costs)

    def __call__(
        self,
        lsa=From(costs),
        m_blkt_beryllium=From(fwbs),
        ucblbe=From(costs),
        m_blkt_li2o=From(fwbs),
        ucblli2o=From(costs),
        m_blkt_steel_total=From(fwbs),
        ucblss=From(costs),
        m_blkt_vanadium=From(fwbs),
        ucblvd=From(costs),
        fkind=From(costs),
        ifueltyp=From(costs),
    ):
        return calculate_blanket_cost_magnetic_confinement(
            lsa,
            m_blkt_beryllium,
            ucblbe,
            m_blkt_li2o,
            ucblli2o,
            m_blkt_steel_total,
            ucblss,
            m_blkt_vanadium,
            ucblvd,
            fkind,
            ifueltyp,
        )


class ShieldCost(ExplicitFunction):
    """cottax node: `calculate_shield_cost` (Account 221.3)."""

    c22131 = OutputInto(costs)
    c22132 = OutputInto(costs)
    c2213 = OutputInto(costs)

    def __call__(
        self,
        lsa=From(costs),
        whtshld=From(fwbs),
        ucshld=From(costs),
        wpenshld=From(fwbs),
        ucpens=From(costs),
        fkind=From(costs),
    ):
        return calculate_shield_cost_magnetic_confinement(
            lsa, whtshld, ucshld, wpenshld, ucpens, fkind
        )


class ReactorCost(ExplicitFunction):
    """cottax node: `calculate_reactor_cost` (Account 221 total)."""

    c221 = OutputInto(costs)

    def __call__(
        self,
        c2211=From(costs),
        c2212=From(costs),
        c2213=From(costs),
        c2214=From(costs),
        c2215=From(costs),
    ):
        return calculate_reactor_cost(c2211, c2212, c2213, c2214, c2215)


class TfMagnetCostSuperconducting(ExplicitFunction):
    """The Account 222.1 superconducting-TF family (`.tfcoil.i_tf_sup == 1`). One
    occupant per `.costs.supercond_cost_model` value.

    The switch was an `eqx.field(static=True)` here until `_audit/next_steps.md`
    §14.2's binding policy. Its two arms are two one-line strand-cost formulas over
    **disjoint** fields -- `.costs.ucsc` + `.tfcoil.m_tf_coil_superconductor` against
    `.costs.sc_mat_cost_0` + `.tfcoil.j_crit_str_0` + `.tfcoil.j_crit_str_tf` -- so one
    node carrying the switch declared three edges the reference run does not make
    (`_audit/switch_kwarg_survey.md` §3, `live (3)`).
    """

    c22211 = OutputInto(costs)
    c22212 = OutputInto(costs)
    c22213 = OutputInto(costs)
    c22214 = OutputInto(costs)
    c22215 = OutputInto(costs)
    c2221 = OutputInto(costs)


class TfMagnetCostSuperconductingPerKg(TfMagnetCostSuperconducting):
    """`.costs.supercond_cost_model == PER_KG` (0) -- PROCESS's own default
    (`cost_variables.py:552`) and the reference run's.

    **Three reads leave with this occupant**: `.costs.sc_mat_cost_0`,
    `.tfcoil.j_crit_str_0`, `.tfcoil.j_crit_str_tf`.
    """

    def __call__(
        self,
        lsa=From(costs),
        ucsc=From(costs),
        i_tf_sc_mat=From(tfcoil),
        m_tf_coil_superconductor=From(tfcoil),
        len_tf_coil=From(tfcoil),
        n_tf_coil_turns=From(tfcoil),
        uccu=From(costs),
        m_tf_coil_copper=From(tfcoil),
        cconshtf=From(costs),
        cconfix=From(costs),
        n_tf_coils=From(tfcoil),
        ucwindtf=From(costs),
        m_tf_coil_case=From(tfcoil),
        uccase=From(costs),
        aintmass=From(structure),
        UCINT=From(costs),
        clgsmass=From(structure),
        UCGSS=From(costs),
        fkind=From(costs),
    ):
        return calculate_tf_magnet_cost_superconducting_per_kg(
            lsa,
            ucsc,
            i_tf_sc_mat,
            m_tf_coil_superconductor,
            len_tf_coil,
            n_tf_coil_turns,
            uccu,
            m_tf_coil_copper,
            cconshtf,
            cconfix,
            n_tf_coils,
            ucwindtf,
            m_tf_coil_case,
            uccase,
            aintmass,
            UCINT,
            clgsmass,
            UCGSS,
            fkind,
        )


class TfMagnetCostSuperconductingPerKam(TfMagnetCostSuperconducting):
    """`.costs.supercond_cost_model == PER_KAM` (1) -- strand cost scaled by critical
    current density.

    **Two reads leave with this occupant**: `.costs.ucsc` and
    `.tfcoil.m_tf_coil_superconductor`.
    """

    def __call__(
        self,
        lsa=From(costs),
        sc_mat_cost_0=From(costs),
        i_tf_sc_mat=From(tfcoil),
        j_crit_str_0=From(tfcoil),
        j_crit_str_tf=From(tfcoil),
        len_tf_coil=From(tfcoil),
        n_tf_coil_turns=From(tfcoil),
        uccu=From(costs),
        m_tf_coil_copper=From(tfcoil),
        cconshtf=From(costs),
        cconfix=From(costs),
        n_tf_coils=From(tfcoil),
        ucwindtf=From(costs),
        m_tf_coil_case=From(tfcoil),
        uccase=From(costs),
        aintmass=From(structure),
        UCINT=From(costs),
        clgsmass=From(structure),
        UCGSS=From(costs),
        fkind=From(costs),
    ):
        return calculate_tf_magnet_cost_superconducting_per_kam(
            lsa,
            sc_mat_cost_0,
            i_tf_sc_mat,
            j_crit_str_0,
            j_crit_str_tf,
            len_tf_coil,
            n_tf_coil_turns,
            uccu,
            m_tf_coil_copper,
            cconshtf,
            cconfix,
            n_tf_coils,
            ucwindtf,
            m_tf_coil_case,
            uccase,
            aintmass,
            UCINT,
            clgsmass,
            UCGSS,
            fkind,
        )


class TfMagnetCostResistive(ExplicitFunction):
    """cottax node: `calculate_tf_magnet_cost_resistive` (Account 222.1,
    `.tfcoil.i_tf_sup != 1`). Ported but **not registered** -- see the function's own
    docstring and `total_process.py`'s `.costs.i_cost_model` switch comment.
    """

    c22211 = OutputInto(costs)
    c22212 = OutputInto(costs)
    c2221 = OutputInto(costs)
    cpstcst = OutputInto(costs)

    def __call__(
        self,
        lsa=From(costs),
        whtcp=From(tfcoil),
        uccpcl1=From(costs),
        whttflgs=From(tfcoil),
        uccpclb=From(costs),
        itart=From(physics),
        ifueltyp=From(costs),
        fkind=From(costs),
    ):
        return calculate_tf_magnet_cost_resistive(
            lsa, whtcp, uccpcl1, whttflgs, uccpclb, itart, ifueltyp, fkind
        )


class PfMagnetCost(ExplicitFunction):
    """The Account 222.2 PF-magnet family. **Four occupants** -- one per
    `.costs.supercond_cost_model` value times one per `.build.iohcl` value.

    `supercond_cost_model` was an `eqx.field(static=True)` here and the node was
    unregistered because of it (`_audit/cost_boundary_inputs.md` §13.2): its two arms
    read **disjoint** strand-cost fields -- `.costs.ucsc` + `.tfcoil.dcond` against
    `.costs.sc_mat_cost_0` + `.tfcoil.j_crit_str_0` + `.pf_coil.j_crit_str_pf` +
    `.pf_coil.j_crit_str_cs` -- so one node carrying the switch declared four edges the
    reference run does not make, and one of those four (`.pf_coil.j_crit_str_pf`) had no
    producer at all. Both are fixed: the family is here and
    `models/pfcoil/superconductor.py::PFStrandCriticalCurrentDensity` owns the field.

    **`iohcl` was the second such kwarg and it was live, not hypothetical**
    (`_audit/switch_consultation_audit.md` §2). It was pinned to
    `CentralSolenoidConfiguration.PRESENT` while both tracked spherical tokamaks set
    `iohcl = 0`, so the same assembled machine held a PF coil *system* with eight coils
    and no solenoid (`indat._pf_coil_system_deviations` reads the switch correctly) and
    a PF magnet *cost* for six coils plus a solenoid. Splitting the family on it is
    `next_steps.md` §14.2's rule -- no switch is a static kwarg -- and the reads agree
    independently: four fields (`.pf_coil.i_cs_superconductor`,
    `.pf_coil.a_cs_cable_space`, `.pf_coil.f_a_cs_void`, `.pf_coil.fcuohsu`, and on the
    `PER_KAM` arm `.pf_coil.j_crit_str_cs` as a fifth) are read *only* inside the CS
    block, and `a_cs_cable_space` is `unwritten` on both ST files -- a declared edge to
    a field the run never fills.

    `n_cs_pf_coils` stays a static field and is **not** a switch: it is the topology's
    coil count, `naming_convention.md`'s static-kwarg category, threaded from the same
    `PFCoilTopology` the PF coil system is built from (7 with a solenoid, 8 without).
    `i_pf_conductor` also stays -- it branches inside *every* arm rather than between
    them.
    """

    n_cs_pf_coils: int = eqx.field(static=True)
    i_pf_conductor: PFConductorModel = eqx.field(static=True)

    c22221 = OutputInto(costs)
    c22222 = OutputInto(costs)
    c22223 = OutputInto(costs)
    c22224 = OutputInto(costs)
    c2222 = OutputInto(costs)


class PfMagnetCostPerKgWithCentralSolenoid(PfMagnetCost):
    """`supercond_cost_model == PER_KG` (0) with a central solenoid -- the whole
    calculation, and **no ports**: the two conductor densities are the only thing its
    occupants differ in, and each declares its own.

    Abstract in use rather than by `ABC`: `ExplicitFunction._signature_of` reads
    `__call__`'s signature only, so a class with no `__call__` declares no reads and is
    never bound. The shape is `PFCoilMasses._masses`'s and `CoilsMass`'s.

    **Four reads leave with the `PER_KAM` sibling**: `.costs.sc_mat_cost_0`,
    `.tfcoil.j_crit_str_0`, `.pf_coil.j_crit_str_pf`, `.pf_coil.j_crit_str_cs`.
    """

    def _cost(
        self,
        lsa,
        r_pf_coil_middle,
        n_pf_coil_turns,
        cconshpf,
        ucsc,
        i_pf_superconductor,
        fcupfsu,
        f_a_pf_coil_void,
        c_pf_cs_coils_peak_ma,
        j_pf_coil_wp_peak,
        den_pf_conductor,
        den_cs_conductor,
        uccu,
        cconfix,
        i_cs_superconductor,
        a_cs_cable_space,
        f_a_cs_void,
        fcuohsu,
        ucwindpf,
        uccase,
        m_pf_coil_structure_total,
        ucfnc,
        fncmass,
        fkind,
    ):
        """The account, given this occupant's two conductor densities.

        Not a port surface -- `_params` reads `__call__`'s signature only, so what each
        occupant declares is still its own parameter list, and the only entries that
        differ between them are the two `.tfcoil.dcond[k]` elements.
        """
        return calculate_pf_magnet_cost_per_kg(
            self.n_cs_pf_coils,
            self.i_pf_conductor,
            lsa,
            r_pf_coil_middle,
            n_pf_coil_turns,
            cconshpf,
            ucsc,
            i_pf_superconductor,
            fcupfsu,
            f_a_pf_coil_void,
            c_pf_cs_coils_peak_ma,
            j_pf_coil_wp_peak,
            den_pf_conductor,
            den_cs_conductor,
            uccu,
            cconfix,
            i_cs_superconductor,
            a_cs_cable_space,
            f_a_cs_void,
            fcuohsu,
            ucwindpf,
            uccase,
            m_pf_coil_structure_total,
            ucfnc,
            fncmass,
            fkind,
        )


class PfMagnetCostPerKg(PfMagnetCostPerKgWithCentralSolenoid):
    """`supercond_cost_model == PER_KG` (0) with a central solenoid, on the
    `(i_pf_superconductor, i_cs_superconductor) = (3, 1)` pair -- NbTi PF coils, ITER
    Nb3Sn CS. `indat._pf_coil_system_arm` arm `0`, and PROCESS's own default
    (`cost_variables.py:552`) with the reference tokamaks' materials
    (`large_tokamak_nof`, `large_tokamak_eval`).

    **`.tfcoil.dcond` is read by element here, not whole** (2026-08-31), which is what
    makes this class a material occupant at all. `pfcoil/masses.py` and
    `tfcoil/superconducting.py` already read the same array by element, so a node
    reading it whole named one storage location two ways and
    `cottax.tools.pytree.check_antichain` refuses to write such a pytree back --
    measured as 2 violations on `large_tokamak_nof`/`large_tokamak_eval`, 2 on
    `low_aspect_ratio_DEMO` and 1 on `spherical_tokamak_eval`, and it was the entire
    blocker to running the MDA through `cottax.boundary.run`.
    `models/physics/radiation_power.py`'s `f_nd_impurity_electron_array` is the
    standing precedent.

    **`.costs.ucsc` stays whole and both `i_*_superconductor` switches stay ordinary
    reads.** Nothing reads `.costs.ucsc` by element, so no antichain question arises
    there, and the switches still index that cost table -- the case
    `naming_convention.md` leaves as a read, exactly as `.tfcoil.i_tf_sc_mat` is for
    Account 222.1. The occupant split here is forced by *where a value is stored*, not
    by what a switch selects, and saying so is why the reads stay.
    """

    def __call__(
        self,
        lsa=From(costs),
        r_pf_coil_middle=From(pf_coil),
        n_pf_coil_turns=From(pf_coil),
        cconshpf=From(costs),
        ucsc=From(costs),
        i_pf_superconductor=From(pf_coil),
        fcupfsu=From(pf_coil),
        f_a_pf_coil_void=From(pf_coil),
        c_pf_cs_coils_peak_ma=From(pf_coil),
        j_pf_coil_wp_peak=From(pf_coil),
        den_pf_conductor=FromExactly(tfcoil.dcond[I_PF_SUPERCONDUCTOR - 1]),
        den_cs_conductor=FromExactly(tfcoil.dcond[I_CS_SUPERCONDUCTOR - 1]),
        uccu=From(costs),
        cconfix=From(costs),
        i_cs_superconductor=From(pf_coil),
        a_cs_cable_space=From(pf_coil),
        f_a_cs_void=From(pf_coil),
        fcuohsu=From(pf_coil),
        ucwindpf=From(costs),
        uccase=From(costs),
        m_pf_coil_structure_total=From(pf_coil),
        ucfnc=From(costs),
        fncmass=From(structure),
        fkind=From(costs),
    ):
        return self._cost(
            lsa,
            r_pf_coil_middle,
            n_pf_coil_turns,
            cconshpf,
            ucsc,
            i_pf_superconductor,
            fcupfsu,
            f_a_pf_coil_void,
            c_pf_cs_coils_peak_ma,
            j_pf_coil_wp_peak,
            den_pf_conductor,
            den_cs_conductor,
            uccu,
            cconfix,
            i_cs_superconductor,
            a_cs_cable_space,
            f_a_cs_void,
            fcuohsu,
            ucwindpf,
            uccase,
            m_pf_coil_structure_total,
            ucfnc,
            fncmass,
            fkind,
        )


class PfMagnetCostPerKgCsWstNb3Sn(PfMagnetCostPerKgWithCentralSolenoid):
    """The same arm on the `(3, 5)` pair -- NbTi PF coils, WST Nb3Sn CS,
    `low_aspect_ratio_DEMO.IN.DAT`'s materials, `indat._pf_coil_system_arm` arm `1`.

    One read differs from `PfMagnetCostPerKg` and nothing else does: the CS conductor
    density is `.tfcoil.dcond[4]` instead of `.tfcoil.dcond[0]`. That is exactly the
    difference between `PFCoilMassesCsWstNb3Sn` and `PFCoilMasses`, and it is the same
    switch pair resolved by the same predicate -- the two nodes read the same storage
    and now name it the same way. The two elements hold the same 6080 kg/m^3 today and
    the split is still the point (`pfcoil/masses.py`'s
    `I_CS_SUPERCONDUCTOR_WST_NB3SN`).
    """

    def __call__(
        self,
        lsa=From(costs),
        r_pf_coil_middle=From(pf_coil),
        n_pf_coil_turns=From(pf_coil),
        cconshpf=From(costs),
        ucsc=From(costs),
        i_pf_superconductor=From(pf_coil),
        fcupfsu=From(pf_coil),
        f_a_pf_coil_void=From(pf_coil),
        c_pf_cs_coils_peak_ma=From(pf_coil),
        j_pf_coil_wp_peak=From(pf_coil),
        den_pf_conductor=FromExactly(tfcoil.dcond[I_PF_SUPERCONDUCTOR - 1]),
        den_cs_conductor=FromExactly(tfcoil.dcond[I_CS_SUPERCONDUCTOR_WST_NB3SN - 1]),
        uccu=From(costs),
        cconfix=From(costs),
        i_cs_superconductor=From(pf_coil),
        a_cs_cable_space=From(pf_coil),
        f_a_cs_void=From(pf_coil),
        fcuohsu=From(pf_coil),
        ucwindpf=From(costs),
        uccase=From(costs),
        m_pf_coil_structure_total=From(pf_coil),
        ucfnc=From(costs),
        fncmass=From(structure),
        fkind=From(costs),
    ):
        return self._cost(
            lsa,
            r_pf_coil_middle,
            n_pf_coil_turns,
            cconshpf,
            ucsc,
            i_pf_superconductor,
            fcupfsu,
            f_a_pf_coil_void,
            c_pf_cs_coils_peak_ma,
            j_pf_coil_wp_peak,
            den_pf_conductor,
            den_cs_conductor,
            uccu,
            cconfix,
            i_cs_superconductor,
            a_cs_cable_space,
            f_a_cs_void,
            fcuohsu,
            ucwindpf,
            uccase,
            m_pf_coil_structure_total,
            ucfnc,
            fncmass,
            fkind,
        )


class PfMagnetCostPerKgNoCentralSolenoid(PfMagnetCost):
    """`supercond_cost_model == PER_KG` (0) on a machine with `.build.iohcl == 0` --
    both tracked spherical tokamaks.

    **Four reads leave with the sibling above**: `.pf_coil.i_cs_superconductor`,
    `.pf_coil.a_cs_cable_space`, `.pf_coil.f_a_cs_void` and `.pf_coil.fcuohsu`. All four
    are read only inside `acc2222`'s `iohcl == 1` block, and `a_cs_cable_space` has no
    producer on a machine with no solenoid at all.

    `i_pf_superconductor = 9` (Hazelton/Zhai REBCO tape) on both tracked spherical
    tokamaks, so the one conductor density this arm needs is `.tfcoil.dcond[8]` --
    read by element for `PfMagnetCostPerKg`'s reason, and the same element
    `PFCoilMassesNoCentralSolenoid` reads. There is no CS density: with no solenoid
    the whole CS block is skipped, so this occupant is not split further.
    """

    def __call__(
        self,
        lsa=From(costs),
        r_pf_coil_middle=From(pf_coil),
        n_pf_coil_turns=From(pf_coil),
        cconshpf=From(costs),
        ucsc=From(costs),
        i_pf_superconductor=From(pf_coil),
        fcupfsu=From(pf_coil),
        f_a_pf_coil_void=From(pf_coil),
        c_pf_cs_coils_peak_ma=From(pf_coil),
        j_pf_coil_wp_peak=From(pf_coil),
        den_pf_conductor=FromExactly(
            tfcoil.dcond[I_PF_SUPERCONDUCTOR_HAZELTON_ZHAI_REBCO - 1]
        ),
        uccu=From(costs),
        cconfix=From(costs),
        ucwindpf=From(costs),
        uccase=From(costs),
        m_pf_coil_structure_total=From(pf_coil),
        ucfnc=From(costs),
        fncmass=From(structure),
        fkind=From(costs),
    ):
        return calculate_pf_magnet_cost_per_kg_no_central_solenoid(
            self.n_cs_pf_coils,
            self.i_pf_conductor,
            lsa,
            r_pf_coil_middle,
            n_pf_coil_turns,
            cconshpf,
            ucsc,
            i_pf_superconductor,
            fcupfsu,
            f_a_pf_coil_void,
            c_pf_cs_coils_peak_ma,
            j_pf_coil_wp_peak,
            den_pf_conductor,
            uccu,
            cconfix,
            ucwindpf,
            uccase,
            m_pf_coil_structure_total,
            ucfnc,
            fncmass,
            fkind,
        )


class PfMagnetCostPerKam(PfMagnetCost):
    """`supercond_cost_model == PER_KAM` (1) with a central solenoid -- strand cost
    scaled by critical current density.

    **Two reads leave with this occupant**: `.costs.ucsc` and `.tfcoil.dcond`.

    Written although no tracked file selects it, for
    `CSCriticalCurrentDensitiesWstNb3Sn`'s reason: the registry this slot is keyed on is
    `.costs.supercond_cost_model`'s, `TF_MAGNET_COST_SUPERCONDUCTING` is already total
    over both its values, and a `PF_MAGNET_COST` that refused arm `1` would make
    `supercond_cost_model = 1` assemble a TF coil and then fail on the PF coils -- a new
    slot narrowing the set of files the port accepts. The same argument makes the
    `iohcl` split total: all four combinations exist.
    """

    def __call__(
        self,
        lsa=From(costs),
        r_pf_coil_middle=From(pf_coil),
        n_pf_coil_turns=From(pf_coil),
        cconshpf=From(costs),
        i_pf_superconductor=From(pf_coil),
        fcupfsu=From(pf_coil),
        f_a_pf_coil_void=From(pf_coil),
        c_pf_cs_coils_peak_ma=From(pf_coil),
        j_pf_coil_wp_peak=From(pf_coil),
        sc_mat_cost_0=From(costs),
        j_crit_str_0=From(tfcoil),
        j_crit_str_pf=From(pf_coil),
        uccu=From(costs),
        cconfix=From(costs),
        i_cs_superconductor=From(pf_coil),
        a_cs_cable_space=From(pf_coil),
        f_a_cs_void=From(pf_coil),
        fcuohsu=From(pf_coil),
        j_crit_str_cs=From(pf_coil),
        ucwindpf=From(costs),
        uccase=From(costs),
        m_pf_coil_structure_total=From(pf_coil),
        ucfnc=From(costs),
        fncmass=From(structure),
        fkind=From(costs),
    ):
        return calculate_pf_magnet_cost_per_kam(
            self.n_cs_pf_coils,
            self.i_pf_conductor,
            lsa,
            r_pf_coil_middle,
            n_pf_coil_turns,
            cconshpf,
            i_pf_superconductor,
            fcupfsu,
            f_a_pf_coil_void,
            c_pf_cs_coils_peak_ma,
            j_pf_coil_wp_peak,
            sc_mat_cost_0,
            j_crit_str_0,
            j_crit_str_pf,
            uccu,
            cconfix,
            i_cs_superconductor,
            a_cs_cable_space,
            f_a_cs_void,
            fcuohsu,
            j_crit_str_cs,
            ucwindpf,
            uccase,
            m_pf_coil_structure_total,
            ucfnc,
            fncmass,
            fkind,
        )


class PfMagnetCostPerKamNoCentralSolenoid(PfMagnetCost):
    """`supercond_cost_model == PER_KAM` (1) on a machine with `.build.iohcl == 0`.

    **Five reads leave with the sibling above** -- the `PER_KG` pair's four plus
    `.pf_coil.j_crit_str_cs`, which only this arm's CS strand cost needs. Written for
    the same totality reason `PfMagnetCostPerKam` is: refusing one of the four
    combinations would let a file assemble its TF coils and then fail on its PF coils.
    """

    def __call__(
        self,
        lsa=From(costs),
        r_pf_coil_middle=From(pf_coil),
        n_pf_coil_turns=From(pf_coil),
        cconshpf=From(costs),
        i_pf_superconductor=From(pf_coil),
        fcupfsu=From(pf_coil),
        f_a_pf_coil_void=From(pf_coil),
        c_pf_cs_coils_peak_ma=From(pf_coil),
        j_pf_coil_wp_peak=From(pf_coil),
        sc_mat_cost_0=From(costs),
        j_crit_str_0=From(tfcoil),
        j_crit_str_pf=From(pf_coil),
        uccu=From(costs),
        cconfix=From(costs),
        ucwindpf=From(costs),
        uccase=From(costs),
        m_pf_coil_structure_total=From(pf_coil),
        ucfnc=From(costs),
        fncmass=From(structure),
        fkind=From(costs),
    ):
        return calculate_pf_magnet_cost_per_kam_no_central_solenoid(
            self.n_cs_pf_coils,
            self.i_pf_conductor,
            lsa,
            r_pf_coil_middle,
            n_pf_coil_turns,
            cconshpf,
            i_pf_superconductor,
            fcupfsu,
            f_a_pf_coil_void,
            c_pf_cs_coils_peak_ma,
            j_pf_coil_wp_peak,
            sc_mat_cost_0,
            j_crit_str_0,
            j_crit_str_pf,
            uccu,
            cconfix,
            ucwindpf,
            uccase,
            m_pf_coil_structure_total,
            ucfnc,
            fncmass,
            fkind,
        )


class MagnetsCost(ExplicitFunction):
    """cottax node: `calculate_magnets_cost` (Account 222 total)."""

    c222 = OutputInto(costs)

    def __call__(
        self,
        ife=From(ife),
        c2221=From(costs),
        c2222=From(costs),
        c2223=From(costs),
    ):
        return calculate_magnets_cost(ife, c2221, c2222, c2223)


class PowerInjectionCost(ExplicitFunction):
    """cottax node: `calculate_power_injection_cost` (Account 223)."""

    c2231 = OutputInto(costs)
    c2232 = OutputInto(costs)
    c2233 = OutputInto(costs)
    c223 = OutputInto(costs)
    cdcost = OutputInto(costs)

    def __call__(
        self,
        ucech=From(costs),
        p_hcd_ecrh_injected_total_mw=From(current_drive),
        i_hcd_primary=From(current_drive),
        uclh=From(costs),
        ucich=From(costs),
        p_hcd_lowhyb_injected_total_mw=From(current_drive),
        ucnbi=From(costs),
        p_beam_injected_mw=From(current_drive),
        ifueltyp=From(costs),
        fcdfuel=From(costs),
        fkind=From(costs),
    ):
        return calculate_power_injection_cost_magnetic_confinement(
            ucech,
            p_hcd_ecrh_injected_total_mw,
            i_hcd_primary,
            uclh,
            ucich,
            p_hcd_lowhyb_injected_total_mw,
            ucnbi,
            p_beam_injected_mw,
            ifueltyp,
            fcdfuel,
            fkind,
        )


class EnergyStorageCost(ExplicitFunction):
    """The family owning `.costs.c2253` (Account 225.3): one occupant per arm.

    `i_pulsed_plant` and `istore` were `eqx.field(static=True)` kwargs on one node
    declaring two reads. **At the tree's own value both reads are dead**: with
    `i_pulsed_plant == 0` the body sets `c2253 = 0.0`, scales zero by
    `p_plant_electric_net_mw`, multiplies by `fkind` and returns zero -- so the graph
    carried a `.heat_transport -> .costs` edge and a `.costs.fkind` edge that no run
    makes. The unpulsed occupant below reads nothing at all, which is the same statement
    made structurally.
    """

    c2253 = OutputInto(costs)


class EnergyStorageCostUnpulsed(EnergyStorageCost, StatesValues):
    """`i_pulsed_plant == 0`: no storage, so `.costs.c2253` is zero and nothing is read.

    A zero-input node, the same shape as `StellaratorMachineConfig` -- and the same
    open policy question (`next_steps.md` §2, "this node always/only produces
    literals"). It is a node here because something must own the field: an unowned
    `.costs.c2253` would be a boundary input read from the `DataStructure`, which is the
    defect this port exists to remove, not a simplification.

    The zero is **stated** and not a literal in the body, so it reaches the compiled
    program as an argument rather than as a constant XLA is free to fold the readers of
    (`models/stated.py`, `_audit/optimise_design.md` §28, §34): `.costs.c2253` -- account
    225.3 (M$) -- is read at `^stated.costs.c2253` and supplied through the env.
    """


class EnergyStorageCostPulsed(EnergyStorageCost):
    """`i_pulsed_plant == 1`: an ELECTROWATT thermal-storage design, scaled by net
    electric power. One occupant per `.pulse.istore` value.

    **`istore` was a static kwarg here and is not any more.** `switch_kwarg_survey.md`
    band (c) argued it should stay one, because options 1 and 2 are two itemised literal
    sums (`costs.py:2617-2643` and `:2645-2682`) over the *same* two variables, so
    splitting invents no edge and the two occupants are indistinguishable by ports.
    `_audit/next_steps.md` §14.2 withdrew that position: a switch value selects an
    occupant whatever its reads, and `test_occupants_of_one_slot_differ` now asserts a
    distinct occupant **class** rather than distinct ports. This slot is the case that
    policy was restated for, and the gap it leaves is named in that test's docstring --
    nothing catches a family whose occupants differ only in a literal, because from
    outside they do not differ at all.

    The literal moved out of the node with the switch: each occupant calls its own arm
    function, so no integer and no module constant decides anything inside a body.

    Option 3 is the contrast case and is `UNPORTED`: it reads three variables the others
    do not, so it is a different occupant, not a different literal.
    """


class EnergyStorageCostPulsedElectrowattOption1(EnergyStorageCostPulsed):
    """`.pulse.istore == ELECTROWATT_OPTION_1` (1) -- PROCESS's own default
    (`pulse_variables.py:16`).
    """

    def __call__(
        self,
        p_plant_electric_net_mw=From(heat_transport),
        fkind=From(costs),
    ):
        return calculate_energy_storage_cost_electrowatt_option_1(
            p_plant_electric_net_mw, fkind
        )


class EnergyStorageCostPulsedElectrowattOption2(EnergyStorageCostPulsed):
    """`.pulse.istore == ELECTROWATT_OPTION_2` (2)."""

    def __call__(
        self,
        p_plant_electric_net_mw=From(heat_transport),
        fkind=From(costs),
    ):
        return calculate_energy_storage_cost_electrowatt_option_2(
            p_plant_electric_net_mw, fkind
        )


class PowerConditioningCost(ExplicitFunction):
    """cottax node: `calculate_power_conditioning_cost` (Account 225 total)."""

    c225 = OutputInto(costs)

    def __call__(
        self,
        ife=From(ife),
        c2251=From(costs),
        c2252=From(costs),
        c2253=From(costs),
    ):
        return calculate_power_conditioning_cost(ife, c2251, c2252, c2253)


class AuxiliaryComponentCoolingCost(ExplicitFunction):
    """cottax node: `calculate_auxiliary_component_cooling_cost` (Account 2262)."""

    cppa = OutputInto(costs)
    c2262 = OutputInto(costs)

    def __call__(
        self,
        lsa=From(costs),
        UCAHTS=From(costs),
        p_hcd_electric_loss_mw=From(heat_transport),
        p_cryo_plant_electric_mw=From(heat_transport),
        vachtmw=From(heat_transport),
        p_tritium_plant_electric_mw=From(heat_transport),
        fachtmw=From(heat_transport),
        fkind=From(costs),
    ):
        return calculate_auxiliary_component_cooling_cost_magnetic_confinement(
            lsa,
            UCAHTS,
            p_hcd_electric_loss_mw,
            p_cryo_plant_electric_mw,
            vachtmw,
            p_tritium_plant_electric_mw,
            fachtmw,
            fkind,
        )


class CryogenicSystemCost(ExplicitFunction):
    """cottax node: `calculate_cryogenic_system_cost` (Account 2263)."""

    c2263 = OutputInto(costs)

    def __call__(
        self,
        lsa=From(costs),
        uccry=From(costs),
        temp_tf_cryo=From(tfcoil),
        helpow=From(heat_transport),
        fkind=From(costs),
    ):
        return calculate_cryogenic_system_cost(lsa, uccry, temp_tf_cryo, helpow, fkind)


class HeatTransportSystemCost(ExplicitFunction):
    """cottax node: `calculate_heat_transport_system_cost` (Account 226 total)."""

    c226 = OutputInto(costs)

    def __call__(
        self,
        c2261=From(costs),
        c2262=From(costs),
        c2263=From(costs),
    ):
        return calculate_heat_transport_system_cost(c2261, c2262, c2263)


class FuelProcessingCost(ExplicitFunction):
    """cottax node: `calculate_fuel_processing_cost` (Account 2272). Sole producer of
    `.physics.wtgpd`.
    """

    wtgpd = OutputInto(physics)
    c2272 = OutputInto(costs)

    def __call__(
        self,
        rndfuel=From(physics),
        m_fuel_amu=From(physics),
        UCFPR=From(costs),
        fkind=From(costs),
    ):
        return calculate_fuel_processing_cost_magnetic_confinement(
            rndfuel, m_fuel_amu, UCFPR, fkind
        )


class AtmosphericRecoveryCost(ExplicitFunction):
    """cottax node: `calculate_atmospheric_recovery_cost` (Account 2273)."""

    c2273 = OutputInto(costs)

    def __call__(
        self,
        f_plasma_fuel_tritium=From(physics),
        UCDTC=From(costs),
        volrci=From(buildings),
        wsvol=From(buildings),
        fkind=From(costs),
    ):
        return calculate_atmospheric_recovery_cost(
            f_plasma_fuel_tritium, UCDTC, volrci, wsvol, fkind
        )


class FuelHandlingCost(ExplicitFunction):
    """cottax node: `calculate_fuel_handling_cost` (Account 227 total)."""

    c227 = OutputInto(costs)

    def __call__(
        self,
        c2271=From(costs),
        c2272=From(costs),
        c2273=From(costs),
        c2274=From(costs),
    ):
        return calculate_fuel_handling_cost(c2271, c2272, c2273, c2274)


class FusionPowerIslandCost(ExplicitFunction):
    """cottax node: `calculate_fusion_power_island_cost` (Account 22 total)."""

    crctcore = OutputInto(costs)
    c22 = OutputInto(costs)

    def __call__(
        self,
        c221=From(costs),
        c222=From(costs),
        c223=From(costs),
        c224=From(costs),
        c225=From(costs),
        c226=From(costs),
        c227=From(costs),
        c228=From(costs),
        c229=From(costs),
    ):
        return calculate_fusion_power_island_cost(
            c221, c222, c223, c224, c225, c226, c227, c228, c229
        )


class TotalPlantDirectCost(ExplicitFunction):
    """cottax node: `calculate_total_plant_direct_cost`."""

    cdirt = OutputInto(costs)

    def __call__(
        self,
        c21=From(costs),
        c22=From(costs),
        c23=From(costs),
        c24=From(costs),
        c25=From(costs),
        c26=From(costs),
    ):
        return calculate_total_plant_direct_cost(c21, c22, c23, c24, c25, c26)


class ConstructedCost(ExplicitFunction):
    """cottax node: `calculate_constructed_cost`."""

    concost = OutputInto(costs)

    def __call__(
        self,
        cdirt=From(costs),
        cindrt=From(costs),
        ccont=From(costs),
    ):
        return calculate_constructed_cost(cdirt, cindrt, ccont)


class CostOfElectricity(ExplicitFunction):
    """The `Costs.coelc` family -- sole producer of `.costs.coe`, the
    `i_figure_merit == 6` objective. One occupant per `.physics.itart` value.

    **Four switches were static kwargs on this one class and none is now**
    (`_audit/next_steps.md` §14.2). Three of them decided nothing this node computes and
    are answered by the slot instead: `ireactor`/`ipnet` jointly decide whether the node
    *exists* (`_cost_of_electricity_arm`, and `costs.py:82-83` is the reason), and `ife`
    is refused once for all seven Account-22x nodes at assembly
    (`indat.py`'s `_ife_cost_accounts_arm`). The fourth, `itart`, is a real branch: the
    centrepost replacement cost of `costs.py:2769-2783` exists only on a spherical
    tokamak, and one class carrying the switch had to declare
    `.costs.cplife_cal`/`.cpstcst`/`.cplife` on a machine that reads none of them.

    The `__check_init__` that used to assert `ireactor == 1 and ipnet == 0` is gone with
    the fields: arm 1 of the slot exists only where they hold, so there is nothing left
    to contradict. That containment is the same one `EcrhDensityLimit` has inside
    `ProfileParameterisationParabolic`.
    """

    moneyint = OutputInto(costs)
    capcost = OutputInto(costs)
    coecap = OutputInto(costs)
    coeoam = OutputInto(costs)
    coefuelt = OutputInto(costs)
    coe = OutputInto(costs)


class CostOfElectricityConventionalAspectRatio(CostOfElectricity):
    """`.physics.itart == 0` -- the reference run's, and PROCESS's own default
    (`physics_variables.py:994`).

    **Three reads leave with this occupant**: `.costs.cplife_cal`, `.costs.cpstcst` and
    `.costs.cplife`. The last is the one that mattered structurally -- it is owned by
    `availability.cplife_avail`'s `FixedPoint`, which is the identity map on this
    machine (`_audit/switch_kwarg_survey.md` §4.7), so the cost of electricity was
    declared to depend on a driven quantity that determines nothing.
    """

    def __call__(
        self,
        p_plant_electric_net_mw=From(heat_transport),
        f_t_plant_available=From(costs),
        t_plant_pulse_burn=From(times),
        t_plant_pulse_total=From(times),
        concost=From(costs),
        fcap0=From(costs),
        fcr0=From(costs),
        discount_rate=From(costs),
        life_blkt=From(fwbs),
        fwallcst=From(costs),
        blkcst=From(costs),
        cfind=From(costs),
        lsa=From(costs),
        fcap0cp=From(costs),
        ifueltyp=From(costs),
        life_blkt_fpy=From(fwbs),
        life_plant=From(costs),
        life_div=From(costs),
        divcst=From(costs),
        life_div_fpy=From(costs),
        cdrlife_cal=From(costs),
        cdcost=From(costs),
        fcdfuel=From(costs),
        ucoam=From(costs),
        ucfuel=From(costs),
        f_plasma_fuel_helium3=From(physics),
        wtgpd=From(physics),
        uche3=From(costs),
        ucwst=From(costs),
        decomf=From(costs),
        dintrt=From(costs),
        dtlife=From(costs),
    ):
        return calculate_cost_of_electricity_conventional_aspect_ratio(
            p_plant_electric_net_mw,
            f_t_plant_available,
            t_plant_pulse_burn,
            t_plant_pulse_total,
            concost,
            fcap0,
            fcr0,
            discount_rate,
            life_blkt,
            fwallcst,
            blkcst,
            cfind,
            lsa,
            fcap0cp,
            ifueltyp,
            life_blkt_fpy,
            life_plant,
            life_div,
            divcst,
            life_div_fpy,
            cdrlife_cal,
            cdcost,
            fcdfuel,
            ucoam,
            ucfuel,
            f_plasma_fuel_helium3,
            wtgpd,
            uche3,
            ucwst,
            decomf,
            dintrt,
            dtlife,
        )


class CostOfElectricitySphericalTokamak(CostOfElectricity):
    """`.physics.itart == 1` -- the spherical tokamak, which additionally pays to
    replace its centrepost (`costs.py:2769-2783`).

    **The three reads its sibling does not make** are `.costs.cplife_cal`,
    `.costs.cpstcst` and `.costs.cplife`. Written and registered although no tracked
    input assembles it yet: without it, dropping those three reads from the conventional
    arm would have deleted a branch PROCESS has rather than filed it.
    """

    def __call__(
        self,
        p_plant_electric_net_mw=From(heat_transport),
        f_t_plant_available=From(costs),
        t_plant_pulse_burn=From(times),
        t_plant_pulse_total=From(times),
        concost=From(costs),
        fcap0=From(costs),
        fcr0=From(costs),
        discount_rate=From(costs),
        life_blkt=From(fwbs),
        fwallcst=From(costs),
        blkcst=From(costs),
        cfind=From(costs),
        lsa=From(costs),
        fcap0cp=From(costs),
        ifueltyp=From(costs),
        life_blkt_fpy=From(fwbs),
        life_plant=From(costs),
        life_div=From(costs),
        divcst=From(costs),
        life_div_fpy=From(costs),
        cplife_cal=From(costs),
        cpstcst=From(costs),
        cplife=From(costs),
        cdrlife_cal=From(costs),
        cdcost=From(costs),
        fcdfuel=From(costs),
        ucoam=From(costs),
        ucfuel=From(costs),
        f_plasma_fuel_helium3=From(physics),
        wtgpd=From(physics),
        uche3=From(costs),
        ucwst=From(costs),
        decomf=From(costs),
        dintrt=From(costs),
        dtlife=From(costs),
    ):
        return calculate_cost_of_electricity_spherical_tokamak(
            p_plant_electric_net_mw,
            f_t_plant_available,
            t_plant_pulse_burn,
            t_plant_pulse_total,
            concost,
            fcap0,
            fcr0,
            discount_rate,
            life_blkt,
            fwallcst,
            blkcst,
            cfind,
            lsa,
            fcap0cp,
            ifueltyp,
            life_blkt_fpy,
            life_plant,
            life_div,
            divcst,
            life_div_fpy,
            cplife_cal,
            cpstcst,
            cplife,
            cdrlife_cal,
            cdcost,
            fcdfuel,
            ucoam,
            ucfuel,
            f_plasma_fuel_helium3,
            wtgpd,
            uche3,
            ucwst,
            decomf,
            dintrt,
            dtlife,
        )
