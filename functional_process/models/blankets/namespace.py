"""The CCFE HCPB blanket's namespace -- the sixteen slots of `.tokamak.ccfe_hcpb`.

Beside the nodes it names (`model_tree_design.md` §11), and spanning two modules for the
same reason `tfcoil/namespace.py` spans three: `blankets/blanket_library.py` is reached
purely as a **base class** of `CCFE_HCPB` (`hcpb.py:25`), never by any call in
`caller.py` (`tokamak_call_surface.md` §A row 10). It is this occupant's body, not its
sibling, which is why its four nodes are slots here rather than a namespace of their own.

**Sixteen slots, and every one of them is now total.** (The sixteenth,
`inboard_poloidal_angle`, arrived 2026-08-30 as a missing producer -- see its own
docstring.) The two `*SphericalTokamak`
occupants (`NuclearHeatingMagnetsSphericalTokamak`,
`NuclearHeatingShieldSphericalTokamak`) were written, harness-tested and deliberately
*unregistered* until 2026-08-27, because a machine at `.physics.itart == 1` also needs
the centrepost neutronics chain (`hcpb.py:1008-1287`) and `blanket_library`'s D-shaped
geometry, and filling the two `itart` slots without the rest would assemble a graph that
looks complete and is wrong -- the `EcrhDensityLimit` bug class. Both preconditions were
supplied that day, the D-shaped geometry by its own wave and the centrepost chain by
this one, and the refusal is gone rather than moved. `hcpb.md` open question 3 asked for
the reason to live in `indat.py`'s `UNPORTED`; it lived there, and then it was answered.

**Four switches decide six of the slots**, and all four are answered in `indat.py`:
`.physics.itart`, `.divertor.n_divertors` (which the factory *derives* from
`.physics.i_single_null`, as `init.py:606-616` does), `.tfcoil.i_tf_sup` -- new to this
namespace on 2026-08-27, read by the centrepost chain -- and `.fwbs.i_p_coolant_pumping`.
Three of the six slots are keyed on a **joint** arm rather than on one integer: the
`.physics.itart` x `.fwbs.i_fw_blkt_vv_shape` shape decision at
`blanket_library.py:90-93`, the `(n_divertors, itart)` renormalisation square, and
the `(itart, i_tf_sup)` centrepost cell -- the last because `hcpb.py`'s two
`i_tf_sup`-reading routines cut that switch in two *different* places.
"""

import dataclasses

from cottax.interfaces.pytree_namespace_module import ModelNamespace

from functional_process.models.blankets.blanket_library import (
    BlanketAreas,
    BlanketCoverageFactors,
    BlanketHalfHeight,
    BlanketInboardPoloidalAngle,
    BlanketVolumes,
)
from functional_process.models.blankets.hcpb import (
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

    blanket_half_height: BlanketHalfHeight = dataclasses.field(kw_only=True)
    """`.divertor.n_divertors` -- **both** arms are written (2026-08-27). They differ by
    five reads (`blanket_library.py:169-232`), which is why they are two occupants."""

    blanket_areas: BlanketAreas = dataclasses.field(kw_only=True)
    """`component_volumes`' shape decision -- **both** arms written (2026-08-27).

    A **joint** switch: `blanket_library.py:90-93` tests
    `itart == 1 or i_fw_blkt_vv_shape == D_SHAPED`, so neither integer decides it alone
    and the factory turns the pair into an arm index -- the shape
    `_blanket_shield_power_arm` and `_energy_storage_arm` already use. The two arms read
    overlapping but unequal sets (the D-shaped one reads no `triang` and no outboard
    build radius), so they are occupants and not a parameter."""

    blanket_volumes: BlanketVolumes = dataclasses.field(kw_only=True)
    """The same joint arm as `blanket_areas`; one input value filling two slots. Also
    total since 2026-08-27."""

    blanket_coverage_factors: BlanketCoverageFactors = dataclasses.field(kw_only=True)
    """`.divertor.n_divertors` -- both arms written (2026-08-27). Owns
    `.fwbs.vol_blkt_total`, which is what the whole of `blanket_library.py` exists to
    reach."""

    # ---- hcpb.py: the poloidal angles run() computes next ---------------------------

    inboard_poloidal_angle: BlanketInboardPoloidalAngle = BlanketInboardPoloidalAngle()
    """`.blanket.deg_blkt_inboard_poloidal_plasma` (`hcpb.py:64-69`, calling the base
    class's `blanket_library.py:3771-3797`). Unswitched, so a default.

    Added 2026-08-30 as a missing producer: `.tokamak.divertor.heat_flux_split` reads
    this angle and is the only reader of it, so with nothing owning it the divertor sized
    itself on `(180 - 0)/2 = 90` degrees against PROCESS's `26.1`. It sits between
    `component_volumes`' four slots and the masses because that is where `run()` computes
    it, and it needs `.blanket.dz_blkt_half` from `blanket_half_height` above.

    The outboard sibling is deliberately absent and would have to be a separate slot --
    see the class's own docstring for the cycle that folding them together would close.
    """

    # ---- hcpb.py: masses ------------------------------------------------------------

    first_wall_coolant_void_fractions: FirstWallCoolantVoidFractions = (
        FirstWallCoolantVoidFractions()
    )
    """The node that makes the rest of this file acyclic (`hcpb.md` §"the two cycles").
    Unswitched."""

    divertor_surface_and_plate_mass: DivertorSurfaceAndPlateMass = dataclasses.field(
        kw_only=True
    )
    """`.divertor.n_divertors` -- `hcpb.py:360-361` doubles `a_div_surface_total` on the
    double-null arm; both arms written (2026-08-27). Owns
    `.divertor.a_div_surface_total`, which `.costs.divertor_cost` reads."""

    component_masses: ComponentMasses = ComponentMasses()
    """Unswitched, once the divertor pair above is a slot of its own. Owns four of this
    slot's sixteen boundary variables."""

    # ---- hcpb.py: the four nuclear-heating routines, in `run()`'s order -------------

    nuclear_heating_magnets: NuclearHeatingMagnets = dataclasses.field(kw_only=True)
    """`.physics.itart` -- **both** arms registered (2026-08-27). They own the same nine
    fields and read unequal sets; see the family base's docstring."""

    nuclear_heating_fw: NuclearHeatingFw = NuclearHeatingFw()
    nuclear_heating_blanket: NuclearHeatingBlanket = NuclearHeatingBlanket()
    """Unswitched. **This is the node whose stellarator registration was blocked** by
    `blanket_neutronics()`'s live PROCESS call-site bug (`unit_registry.md` row 13,
    `indat.py`'s `("blktmodel_ipowerflow", 0)` refusal). The tokamak reaches it directly,
    from `run()`, so the port's three original hcpb nodes are registered here for the
    first time."""

    nuclear_heating_shield: NuclearHeatingShield = dataclasses.field(kw_only=True)
    """`.physics.itart` -- both arms registered (2026-08-27). Reads two of
    `nuclear_heating_magnets`' own outputs, which is an ordinary graph edge in the same
    order `run()` calls them."""

    centrepost_neutronics: CentrepostNeutronics = dataclasses.field(kw_only=True)
    """The joint `(.physics.itart, .tfcoil.i_tf_sup)` arm -- `hcpb.py:103-148`.

    The conventional occupant reads nothing and writes four literal zeros, which is
    legitimate for the reason `next_steps.md` §14.4 gives for `i_pulsed_plant`'s unpulsed
    occupant: PROCESS's own source on that arm *is* four literal assignments, and the
    fields have readers in `power.py`, `tfcoil/base.py` and `availability.py`. The
    spherical occupant (2026-08-27) runs the three `st_*` routines of
    `hcpb.py:1008-1287`, owns two mints, and is the reason `itart == 1` assembles at
    all. Only the `i_tf_sup == 1` cell of the spherical row is written; see the family
    base for why `i_tf_sup` cannot be dropped from the key."""

    nuclear_heating_renormalisation: NuclearHeatingRenormalisation = dataclasses.field(
        kw_only=True
    )
    """`.divertor.n_divertors` **and** `.physics.itart` together -- a 2x2, **total**
    since 2026-08-27. Owns four boundary variables on every arm and a fifth
    (`.fwbs.p_cp_shield_nuclear_heat_mw`) on the two spherical ones. One of the four --
    `.fwbs.p_tf_nuclear_heat_mw` -- has a second producer on the *stellarator* tree; see
    the class's own docstring for why that is a record and not a conflict."""

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
