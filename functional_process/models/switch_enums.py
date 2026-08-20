"""`IntEnum`s for PROCESS model-selection switches that PROCESS itself does not name.

`_audit/model_tree_design.md` §4 ("Settings stay on the occupant, enum-typed") requires
every kind-(a) model-selection switch carried as an `eqx.field(static=True)` on a
declaration to be typed with an `IntEnum`, never a bare `int` -- so that `PROCESS_1990`
cannot typo into `KOVARI_2014` the way `0` typos into `1`
(`_audit/switch_elimination_design.md` §5(A): five defects of exactly that shape).

**Upstream is the first choice.** 51 `IntEnum`s already exist across `process/`
(`switch_elimination_design.md` §4) and every switch that has one is typed with *that*
enum, imported directly -- `ConfinementTimeModel`, `PumpingPowerModelTypes`,
`TFConductorModel`, `SuperconductorModel`, `PFConductorModel`, `BlktModelTypes`,
`ElectricConversionModelTypes`, `PlasmaIgnitionModel`,
`ConfinementRadiationLossModel`, `PlasmaProfileShapeType`, `CurrentDriveModel`.
This module holds only the remainder: switches this port carries for which PROCESS
declares no enum at all.

**Member names are PROCESS's own words, not invented ones.** Each enum below quotes the
`process/data_structure/*_variables.py` docstring it was derived from, so the mapping
back to PROCESS's documentation is checkable by reading, not by trust. Where the
docstring gives a name (`Frances Fox model`, `option 1 of Electrowatt report`) that name
is used; where it gives only a description the member name paraphrases it minimally.

**These are candidates to be pushed upstream**, at which point the definition here is
deleted and the import re-pointed at `process/` -- exactly what happened for every enum
in the first list. Nothing else about the declarations changes when that happens,
because `IntEnum` members compare and hash equal to their `int` values: arithmetic,
`jax.jit` cache keys and `switch_audit`'s comparisons against a converged
`DataStructure` all see the same numbers they saw before.

Kinds (b) (shape/resolution ints such as `n_plasma_profile_elements`, `n_cs_pf_coils`,
`n_vac_pumps_high`, `redun_vac`) and (c) (set membership, `imp_indices`) are
deliberately **not** here: they are numbers and sets, not choices, and calling them
models would be the category error `switch_elimination_design.md` §3 warns about.
"""

from enum import IntEnum

__all__ = [
    "BlanketDualCoolantModel",
    "BlanketLifetimeModel",
    "CentralSolenoidConfiguration",
    "CoilNuclearHeatingModel",
    "CostOfElectricityModel",
    "FastAlphaPressureModel",
    "IFEModel",
    "NetElectricPowerModel",
    "NeutronWallLoadModel",
    "PFEnergyStorageSource",
    "PlantOperationModel",
    "PowerFlowModel",
    "SphericalTokamakModel",
    "SuperconductorCostModel",
    "ThermalStorageModel",
]


class FastAlphaPressureModel(IntEnum):
    """`.physics.i_beta_fast_alpha` -- switch for fast alpha pressure calculation.

    `process/data_structure/physics_variables.py:875-879`: "=0 ITER physics rules
    (Uckan) fit", "=1 Modified fit (D. Ward) - better at high temperature".
    """

    ITER_PHYSICS_RULES = 0
    WARD = 1


class NeutronWallLoadModel(IntEnum):
    """`.physics.i_pflux_fw_neutron` -- switch for neutron wall load calculation.

    `process/data_structure/physics_variables.py:1006-1010`: "=1 use scaled plasma
    surface area", "=2 use first wall area directly".
    """

    SCALED_PLASMA_SURFACE_AREA = 1
    FIRST_WALL_AREA = 2


class SphericalTokamakModel(IntEnum):
    """`.physics.itart` -- switch for spherical tokamak (ST) models.

    `process/data_structure/physics_variables.py:994-998`: "=0 use conventional aspect
    ratio models", "=1 use spherical tokamak models".
    """

    CONVENTIONAL_ASPECT_RATIO = 0
    SPHERICAL_TOKAMAK = 1


class PowerFlowModel(IntEnum):
    """`.heat_transport.ipowerflow` -- switch for power flow model.

    `process/data_structure/heat_transport_variables.py:94-98`: "=0 pre-2014 version",
    "=1 comprehensive 2014 model".
    """

    PRE_2014 = 0
    COMPREHENSIVE_2014 = 1


class IFEModel(IntEnum):
    """`.ife.ife` -- switch for the inertial fusion energy option.

    `process/data_structure/ife_variables.py:253-257`: "=0 use tokamak, RFP or
    stellarator model", "=1 use IFE model".
    """

    MAGNETIC_CONFINEMENT = 0
    INERTIAL_CONFINEMENT = 1


class CostOfElectricityModel(IntEnum):
    """`.costs.ireactor` -- switch for net electric power and cost-of-electricity
    calculations.

    `process/data_structure/cost_variables.py:521-525`: "=0 do not calculate
    MW(electric) or c-o-e", "=1 calculate MW(electric) and c-o-e".
    """

    NOT_CALCULATED = 0
    CALCULATED = 1


class NetElectricPowerModel(IntEnum):
    """`.costs.ipnet` -- switch for net electric power calculation.

    `process/data_structure/cost_variables.py:515-519`: "=0 scale so that always > 0",
    "=1 let go < 0 (no c-o-e)".
    """

    SCALED_POSITIVE = 0
    MAY_GO_NEGATIVE = 1


class PlantOperationModel(IntEnum):
    """`.pulse.i_pulsed_plant` -- switch for reactor operation model.

    `process/data_structure/pulse_variables.py:30-34`: "=0 continuous operation",
    "=1 pulsed operation".
    """

    CONTINUOUS = 0
    PULSED = 1


class ThermalStorageModel(IntEnum):
    """`.pulse.istore` -- switch for thermal storage method.

    `process/data_structure/pulse_variables.py:16-21`: "=1 option 1 of Electrowatt
    report, AEA FUS 205", "=2 option 2 of Electrowatt report, AEA FUS 205",
    "=3 stainless steel block".
    """

    ELECTROWATT_OPTION_1 = 1
    ELECTROWATT_OPTION_2 = 2
    STAINLESS_STEEL_BLOCK = 3


class BlanketLifetimeModel(IntEnum):
    """`.costs.ibkt_life` -- switch for fw/blanket lifetime calculation in the
    availability module.

    `process/data_structure/cost_variables.py:416-418`: "=0 use neutron fluence model",
    "=1 use fusion power model (DEMO only)".
    """

    NEUTRON_FLUENCE = 0
    FUSION_POWER = 1


class CoilNuclearHeatingModel(IntEnum):
    """`.fwbs.inuclear` -- switch for nuclear heating in the coils.

    `process/data_structure/fwbs_variables.py:81-85`: "=0 Frances Fox model (default)",
    "=1 Fixed by user (qnuc)".
    """

    FRANCES_FOX = 0
    USER_INPUT = 1


class BlanketDualCoolantModel(IntEnum):
    """`.fwbs.i_blkt_dual_coolant` -- switch for single- vs dual-coolant breeding
    blanket.

    `process/data_structure/fwbs_variables.py:526-533`: "=0 Single coolant used for FW
    and Blanket (H2O or He). Solid Breeder.", "=1 Single coolant used for FW and Blanket
    (H2O or He). Liquid metal breeder circulted for tritium extraction.", "=2 Dual
    coolant: primary coolant (H2O or He) for FW and blanket structure; secondary coolant
    is self-cooled liquid metal breeder."
    """

    SINGLE_COOLANT_SOLID_BREEDER = 0
    SINGLE_COOLANT_LIQUID_BREEDER = 1
    DUAL_COOLANT = 2


class PFEnergyStorageSource(IntEnum):
    """`.pf_power.i_pf_energy_storage_source` -- switch for the PF coil energy storage
    option.

    `process/data_structure/pf_power_variables.py:18-24`: "=1 all power from MGF
    (motor-generator flywheel) units", "=2 all pulsed power from line", "=3 PF power
    from MGF, heating from line". PROCESS's own note that "options 1 and 3 are not
    treated differently" is a property of the arithmetic, not of the vocabulary, and is
    preserved by keeping all three members.
    """

    MGF = 1
    LINE = 2
    MGF_PF_LINE_HEATING = 3


class CentralSolenoidConfiguration(IntEnum):
    """`.build.iohcl` -- switch for the existence of a central solenoid.

    `process/data_structure/build_variables.py:177-181`: "=0 central solenoid not
    present", "=1 central solenoid exists". Spelled `Configuration` rather than `Model`
    to match `build_variables.py`'s own two neighbouring enums
    (`CSPrecompressionConfiguration`, `InboardBlanketConfiguration`), which name the
    same kind of presence/absence topology fact.
    """

    NOT_PRESENT = 0
    PRESENT = 1


class SuperconductorCostModel(IntEnum):
    """`.costs.supercond_cost_model` -- switch for superconductor cost model.

    `process/data_structure/cost_variables.py:552-554`: "=0 use $/kg", "=1 use $/kAm".
    """

    PER_KG = 0
    PER_KAM = 1
