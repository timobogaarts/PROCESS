"""The CCFE HCPB blanket's namespace -- the sixteen slots of `.tokamak.ccfe_hcpb`."""

import dataclasses

from cottax.interfaces.pytree_namespace_module import ModelNamespace

from functional_process.cottax.blankets.blanket_library import (
    BlanketAreas,
    BlanketCoverageFactors,
    BlanketHalfHeight,
    BlanketInboardPoloidalAngle,
    BlanketVolumes,
)
from functional_process.cottax.blankets.hcpb import (
    CentrepostNeutronics,
    ComponentMasses,
    DivertorSurfaceAndPlateMass,
    FirstWallCoolantVoidFractions,
    FirstWallRadiationPowers,
    NuclearHeatingBlanket,
    NuclearHeatingFw,
    NuclearHeatingMagnets,
    NuclearHeatingRenormalisation,
    NuclearHeatingShield,
    PumpingPowerMechanicalWithPressureDrop,
)


class CcfeHcpb(ModelNamespace):
    """The CCFE helium-cooled pebble-bed blanket: geometry, masses, neutronics, pumping.
    """

    # ---- blanket_library.py: the geometry `component_masses` runs on ----------------

    blanket_half_height: BlanketHalfHeight = dataclasses.field(kw_only=True)
    """`.divertor.n_divertors` -- **both** arms are written (2026-08-27)."""

    blanket_areas: BlanketAreas = dataclasses.field(kw_only=True)
    """`component_volumes`' shape decision -- **both** arms written (2026-08-27)."""

    blanket_volumes: BlanketVolumes = dataclasses.field(kw_only=True)
    """The same joint arm as `blanket_areas`; one input value filling two slots."""

    blanket_coverage_factors: BlanketCoverageFactors = dataclasses.field(kw_only=True)
    """`.divertor.n_divertors` -- both arms written (2026-08-27)."""

    # ---- hcpb.py: the poloidal angles run() computes next ---------------------------

    inboard_poloidal_angle: BlanketInboardPoloidalAngle = BlanketInboardPoloidalAngle()
    """`.blanket.deg_blkt_inboard_poloidal_plasma` (`hcpb.py:64-69`, calling the base
    class's `blanket_library.py:3771-3797`).
    """

    # ---- hcpb.py: masses ------------------------------------------------------------

    first_wall_coolant_void_fractions: FirstWallCoolantVoidFractions = (
        FirstWallCoolantVoidFractions()
    )
    """The node that makes the rest of this file acyclic (`hcpb.md` §"the two cycles").
    """

    divertor_surface_and_plate_mass: DivertorSurfaceAndPlateMass = dataclasses.field(
        kw_only=True
    )
    """`.divertor.n_divertors` -- `hcpb.py:360-361` doubles `a_div_surface_total` on the
    double-null arm; both arms written (2026-08-27).
    """

    component_masses: ComponentMasses = ComponentMasses()
    """Unswitched, once the divertor pair above is a slot of its own."""

    # ---- hcpb.py: the four nuclear-heating routines, in `run()`'s order -------------

    nuclear_heating_magnets: NuclearHeatingMagnets = dataclasses.field(kw_only=True)
    """`.physics.itart` -- **both** arms registered (2026-08-27)."""

    nuclear_heating_fw: NuclearHeatingFw = NuclearHeatingFw()
    nuclear_heating_blanket: NuclearHeatingBlanket = NuclearHeatingBlanket()
    """Unswitched."""

    nuclear_heating_shield: NuclearHeatingShield = dataclasses.field(kw_only=True)
    """`.physics.itart` -- both arms registered (2026-08-27)."""

    centrepost_neutronics: CentrepostNeutronics = dataclasses.field(kw_only=True)
    """The joint `(.physics.itart, .tfcoil.i_tf_sup)` arm -- `hcpb.py:103-148`."""

    nuclear_heating_renormalisation: NuclearHeatingRenormalisation = dataclasses.field(
        kw_only=True
    )
    """`.divertor.n_divertors` **and** `.physics.itart` together -- a 2x2, **total**
    since 2026-08-27.
    """

    # ---- hcpb.py: powerflow_calc ----------------------------------------------------

    first_wall_radiation_powers: FirstWallRadiationPowers = FirstWallRadiationPowers()
    """`powerflow_calc`'s unconditional prologue -- **not** behind
    `i_p_coolant_pumping`.
    """

    pumping_power: PumpingPowerMechanicalWithPressureDrop = dataclasses.field(
        kw_only=True
    )
    """`.fwbs.i_p_coolant_pumping`, and the clearest instance in this port of a switch
    whose arms **do not own the same set**: arm 1 owns four
    `.heat_transport.p_*_pump_mw` fields, arm 3 owns two of them plus
    `.primary_pumping.p_fw_blkt_coolant_pump_mw`.
    """
