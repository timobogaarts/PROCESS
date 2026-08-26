"""The CCFE HCPB blanket's namespace -- the fifteen slots of `.tokamak.ccfe_hcpb`.

Beside the nodes it names (`model_tree_design.md` §11), and spanning two modules for the
same reason `tfcoil/namespace.py` spans three: `blankets/blanket_library.py` is reached
purely as a **base class** of `CCFE_HCPB` (`hcpb.py:25`), never by any call in
`caller.py` (`tokamak_call_surface.md` §A row 10). It is this occupant's body, not its
sibling, which is why its four nodes are slots here rather than a namespace of their own.

**Fifteen slots, not seventeen.** `hcpb.py` and `blanket_library.py` carry seventeen
ported node classes; the two `*SphericalTokamak` occupants
(`NuclearHeatingMagnetsSphericalTokamak`, `NuclearHeatingShieldSphericalTokamak`) are
written, harness-tested and **deliberately unregistered**. A machine at
`.physics.itart == 1` needs the centrepost neutronics chain
(`hcpb.py:1008-1287`, unported) and `blanket_library`'s D-shaped geometry as well, so
filling the two `itart` slots without the rest would assemble a graph that looks
complete and is wrong -- the `EcrhDensityLimit` bug class. `indat.py`'s `UNPORTED`
carries the refusal; `hcpb.md` open question 3 asked for exactly this.

**Three switches decide five of the slots**, and all three are answered in `indat.py`:
`.physics.itart`, `.divertor.n_divertors` (which the factory *derives* from
`.physics.i_single_null`, as `init.py:606-616` does) and
`.fwbs.i_p_coolant_pumping`. A fourth, the joint
`.physics.itart` x `.fwbs.i_fw_blkt_vv_shape` shape decision at
`blanket_library.py:90-93`, is a joint arm index in the same shape as
`_blanket_shield_power_arm`.
"""

import dataclasses

from cottax.interfaces.pytree_namespace_module import ModelNamespace

from functional_process.models.blankets.blanket_library import (
    BlanketCoverageFactorsSingleNull,
    BlanketHalfHeightSingleNull,
    EllipticalBlanketAreas,
    EllipticalBlanketVolumes,
)
from functional_process.models.blankets.hcpb import (
    CentrepostNeutronicsAbsent,
    ComponentMasses,
    DivertorSurfaceAndPlateMassSingleNull,
    FirstWallCoolantVoidFractions,
    FirstWallRadiationPowers,
    NuclearHeatingBlanket,
    NuclearHeatingFw,
    NuclearHeatingMagnetsConventional,
    NuclearHeatingRenormalisationSingleNullConventional,
    NuclearHeatingShieldConventional,
    PumpingPowerMechanicalWithPressureDrop,
)


class CcfeHcpb(ModelNamespace):
    """The CCFE helium-cooled pebble-bed blanket: geometry, masses, neutronics, pumping.

    Named for `process/models/blankets/hcpb.py::CCFE_HCPB`, which `caller.py:345` runs at
    `.fwbs.i_blanket_type == 1`. That switch has a second value in scope --  `== 5`
    routes to `blankets/dcll.py`, a different occupant of this same slot -- which is why
    `.tokamak.ccfe_hcpb` is a slot at all and not simply a subsystem.

    Written in PROCESS's own call order: `blanket_library`'s geometry first (it runs
    inside `component_masses`, `hcpb.py:306` onwards), then the masses, then the four
    nuclear-heating routines in the order `run()` calls them, then the renormalisation
    that scales all four, then `powerflow_calc`'s two nodes.
    """

    # ---- blanket_library.py: the geometry `component_masses` runs on ----------------

    blanket_half_height: BlanketHalfHeightSingleNull = dataclasses.field(kw_only=True)
    """`.divertor.n_divertors` -- the single-null arm is written; the double-null arm
    (`blanket_library.py:169-232`'s other half) is not."""

    blanket_areas: EllipticalBlanketAreas = dataclasses.field(kw_only=True)
    """The elliptical arm of `component_volumes`' shape decision.

    A **joint** switch: `blanket_library.py:90-93` tests
    `itart == 1 or i_fw_blkt_vv_shape == D_SHAPED`, so neither integer decides it alone
    and the factory turns the pair into an arm index -- the shape
    `_blanket_shield_power_arm` and `_energy_storage_arm` already use. Only the
    elliptical arm (`itart == 0` **and** `ELLIPTICAL_SHAPED`) is written."""

    blanket_volumes: EllipticalBlanketVolumes = dataclasses.field(kw_only=True)
    """The same joint arm as `blanket_areas`; one input value filling two slots."""

    blanket_coverage_factors: BlanketCoverageFactorsSingleNull = dataclasses.field(
        kw_only=True
    )
    """`.divertor.n_divertors`. Owns `.fwbs.vol_blkt_total`, which is what the whole of
    `blanket_library.py` exists to reach."""

    # ---- hcpb.py: masses ------------------------------------------------------------

    first_wall_coolant_void_fractions: FirstWallCoolantVoidFractions = (
        FirstWallCoolantVoidFractions()
    )
    """The node that makes the rest of this file acyclic (`hcpb.md` §"the two cycles").
    Unswitched."""

    divertor_surface_and_plate_mass: DivertorSurfaceAndPlateMassSingleNull = (
        dataclasses.field(kw_only=True)
    )
    """`.divertor.n_divertors` -- `hcpb.py:360-361` doubles `a_div_surface_total` on the
    double-null arm. Owns `.divertor.a_div_surface_total`, which `.costs.divertor_cost`
    reads."""

    component_masses: ComponentMasses = ComponentMasses()
    """Unswitched, once the divertor pair above is a slot of its own. Owns four of this
    slot's sixteen boundary variables."""

    # ---- hcpb.py: the four nuclear-heating routines, in `run()`'s order -------------

    nuclear_heating_magnets: NuclearHeatingMagnetsConventional = dataclasses.field(
        kw_only=True
    )
    """`.physics.itart`. The spherical-tokamak occupant exists and is not registered --
    see this module's docstring."""

    nuclear_heating_fw: NuclearHeatingFw = NuclearHeatingFw()
    nuclear_heating_blanket: NuclearHeatingBlanket = NuclearHeatingBlanket()
    """Unswitched. **This is the node whose stellarator registration was blocked** by
    `blanket_neutronics()`'s live PROCESS call-site bug (`unit_registry.md` row 13,
    `indat.py`'s `("blktmodel_ipowerflow", 0)` refusal). The tokamak reaches it directly,
    from `run()`, so the port's three original hcpb nodes are registered here for the
    first time."""

    nuclear_heating_shield: NuclearHeatingShieldConventional = dataclasses.field(
        kw_only=True
    )
    """`.physics.itart`; reads two of `nuclear_heating_magnets`' own outputs, which is
    an ordinary graph edge in the same order `run()` calls them."""

    centrepost_neutronics: CentrepostNeutronicsAbsent = dataclasses.field(kw_only=True)
    """`.physics.itart` -- the `else` arm, four literal zeros and no reads. A node that
    reads nothing is legitimate for the reason `next_steps.md` §14.4 gives for
    `i_pulsed_plant`'s unpulsed occupant: PROCESS's own source on this arm *is* four
    literal assignments, and the fields have readers in `power.py`, `tfcoil/base.py` and
    `availability.py`."""

    nuclear_heating_renormalisation: NuclearHeatingRenormalisationSingleNullConventional = dataclasses.field(
        kw_only=True
    )  # noqa: E501
    """`.divertor.n_divertors` **and** `.physics.itart` together. Owns four more boundary
    variables, one of which -- `.fwbs.p_tf_nuclear_heat_mw` -- has a second producer on
    the *stellarator* tree; see the class's own docstring for why that is a record and
    not a conflict."""

    # ---- hcpb.py: powerflow_calc ----------------------------------------------------

    first_wall_radiation_powers: FirstWallRadiationPowers = FirstWallRadiationPowers()
    """`powerflow_calc`'s unconditional prologue -- **not** behind
    `i_p_coolant_pumping`."""

    pumping_power: PumpingPowerMechanicalWithPressureDrop = dataclasses.field(
        kw_only=True
    )
    """`.fwbs.i_p_coolant_pumping`, and the clearest instance in this port of a switch
    whose arms **do not own the same set**: arm 1 owns four `.heat_transport.p_*_pump_mw`
    fields, arm 3 owns two of them plus `.primary_pumping.p_fw_blkt_coolant_pump_mw`.
    `next_steps.md` §12.2's "alternatives are keyed on output -- nearly" is this."""
