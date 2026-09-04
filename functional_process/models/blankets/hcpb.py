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

from functional_process.models.stated import StatesValues
from functional_process.models.safe_math import safe_pow, safe_sqrt
from functional_process.paths import (
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


def calculate_fw_coolant_void_fractions(radius_fw_channel, dx_fw_module, dr_fw_inboard):
    """First-wall coolant void fractions, inboard and outboard.

    Ports `nuclear_heating_magnets`' first four statements (`hcpb.py:483-490`) as a node
    of its own. **This extraction is the point of the whole decomposition here**, so the
    reason is worth stating rather than assuming:

    `CCFE_HCPB.run()` calls `component_masses()` at `:150` and
    `nuclear_heating_magnets()` at `:155`. `component_masses` *reads*
    `.fwbs.f_a_fw_coolant_inboard`/`_outboard` (`:317`, `:320`, `:332`, `:335`, `:383`,
    `:386`) and `nuclear_heating_magnets` *writes* them (`:483`, `:490`) -- five lines
    later. The magnets routine in turn reads `.fwbs.m_blkt_total`/`whtshld`/
    `vol_blkt_total`/`vol_shld_total`, which `component_masses` writes. That is a genuine
    two-node cycle, and PROCESS resolves it by accident: `Caller.call_models` re-runs the
    entire pipeline up to ten times until the objective and constraints stop moving
    (`caller.py:100-133`), so the value `component_masses` reads is the *previous*
    pass's.

    It dissolves on inspection, which is the fourth-plus instance of the pattern
    `next_steps.md` §5 records. These three lines depend on **nothing** either routine
    writes -- `radius_fw_channel` and `dx_fw_module` have no producer anywhere under
    `process/models/` (verified by grep: no assignment to either field outside a
    stellarator local), and `dr_fw_inboard` is `build.py`'s. Lift them out and the graph
    is acyclic, with `FirstWallCoolantVoidFractions -> ComponentMasses ->
    NuclearHeatingMagnetsConventional` in that order, and the answer is PROCESS's
    *converged* answer in one pass rather than after N.

    `calculate_nuclear_heating_magnets_conventional` still recomputes the same expression
    as a local (`vffwm`), because PROCESS does and because keeping its signature 1:1 with
    the PROCESS method is what makes its harness case a real diff rather than a
    transcription. Two nodes, one formula, one owner -- the duplicate is a *local*, not a
    second write.

    Parameters
    ----------
    radius_fw_channel :
        First-wall coolant channel radius (m). `.fwbs.radius_fw_channel`.
    dx_fw_module :
        First-wall module toroidal width (m). `.fwbs.dx_fw_module`.
    dr_fw_inboard :
        Inboard first-wall radial thickness (m). `.build.dr_fw_inboard`.

    Returns
    -------
    tuple
        `(f_a_fw_coolant_inboard, f_a_fw_coolant_outboard)` -- numerically identical, the
        source sets the second from the first (`hcpb.py:490`).
    """
    f_a_fw_coolant_inboard = (
        jnp.pi * radius_fw_channel**2 / (dx_fw_module * dr_fw_inboard)
    )
    return f_a_fw_coolant_inboard, f_a_fw_coolant_inboard


def calculate_divertor_surface_and_plate_mass_single_null(
    fdiva,
    rmajor,
    rminor,
    den_div_structure,
    f_vol_div_coolant,
    dx_div_plate,
):
    """Divertor surface area and plate mass, single-null arm (`n_divertors == 1`).

    Ports `component_masses`' `hcpb.py:353-367`, split out of the rest of that method for
    the same reason as the void fractions above: `component_masses` **reads**
    `.divertor.a_div_surface_total` at `:299` (the divertor coolant volume) and
    **writes** it at `:353`, so a single node covering the whole method would read what
    it owns -- a cottax hard error, and in PROCESS a silent one-pass-stale read that only
    the outer Gauss-Seidel loop repairs.

    The write depends on nothing the read's block produces (`fdiva`, `rmajor`, `rminor`
    are all upstream), so ordering this node *before* `ComponentMasses` is both legal and
    exactly PROCESS's converged answer.

    The `n_divertors == 2` arm (`hcpb.py:360-361`, a factor of two) is
    `calculate_divertor_surface_and_plate_mass_double_null` below.

    Parameters
    ----------
    fdiva :
        Divertor area scaling factor. `.divertor.fdiva`.
    rmajor, rminor :
        Plasma major/minor radius (m). `.physics.rmajor`, `.physics.rminor`.
    den_div_structure :
        Divertor structure density (kg/m3). `.divertor.den_div_structure`.
    f_vol_div_coolant :
        Divertor coolant volume fraction. `.divertor.f_vol_div_coolant`.
    dx_div_plate :
        Divertor plate thickness (m). `.divertor.dx_div_plate`.

    Returns
    -------
    tuple
        `(a_div_surface_total, m_div_plate)` -- m2 and kg.
    """
    a_div_surface_total = fdiva * 2.0 * jnp.pi * rmajor * rminor

    m_div_plate = (
        a_div_surface_total
        * den_div_structure
        * (1.0 - f_vol_div_coolant)
        * dx_div_plate
    )

    return a_div_surface_total, m_div_plate


def calculate_divertor_surface_and_plate_mass_double_null(
    fdiva,
    rmajor,
    rminor,
    den_div_structure,
    f_vol_div_coolant,
    dx_div_plate,
):
    """Divertor surface area and plate mass, double-null arm (`n_divertors == 2`).

    Ports `component_masses`' `hcpb.py:353-367` with the `if n_divertors == 2:
    a_div_surface_total *= 2.0` at `:360-361` taken: two divertors, twice the plate
    area, and twice the plate mass that follows from it.

    Reads exactly what the single-null sibling reads -- the arms differ by one literal
    factor, not by a field -- but under `next_steps.md` §14.2 (the `istore` precedent) a
    literal is enough to make this an occupant rather than a `n_divertors` parameter
    multiplied in. See `calculate_divertor_surface_and_plate_mass_single_null` for the
    read/own conflict that puts this node before `ComponentMasses` in the first place;
    it applies unchanged here.

    Parameters
    ----------
    fdiva :
        Divertor area scaling factor. `.divertor.fdiva`.
    rmajor, rminor :
        Plasma major/minor radius (m). `.physics.rmajor`, `.physics.rminor`.
    den_div_structure :
        Divertor structure density (kg/m3). `.divertor.den_div_structure`.
    f_vol_div_coolant :
        Divertor coolant volume fraction. `.divertor.f_vol_div_coolant`.
    dx_div_plate :
        Divertor plate thickness (m). `.divertor.dx_div_plate`.

    Returns
    -------
    tuple
        `(a_div_surface_total, m_div_plate)` -- m2 and kg.
    """
    a_div_surface_total = fdiva * 2.0 * jnp.pi * rmajor * rminor
    # `hcpb.py:360-361`, spelled as PROCESS spells it.
    a_div_surface_total *= 2.0

    m_div_plate = (
        a_div_surface_total
        * den_div_structure
        * (1.0 - f_vol_div_coolant)
        * dx_div_plate
    )

    return a_div_surface_total, m_div_plate


def calculate_component_masses(
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
):
    """Everything `component_masses` computes except the divertor pair.

    Ports `hcpb.py:285-461` minus `:353-367` (the divertor surface area and plate mass,
    which `calculate_divertor_surface_and_plate_mass_single_null` above owns). Closes the
    method's `self.data` back-door.

    **Deviation: `.fwbs.breeder_f` is not owned by this node.** PROCESS clamps it in
    place -- `breeder_f = max(breeder_f, 1e-10)` then `min(breeder_f, 1.0)`
    (`hcpb.py:404-405`) -- a read of a field followed by a write to the same field, which
    is a cottax self-loop. It dissolves rather than needing a `FixedPointFunction`, for
    two measured reasons: the clamp is idempotent, and `.fwbs.breeder_f`'s **only reader
    anywhere under `process/models/` is `hcpb.py:411`, two lines later** (grep:
    `\\.breeder_f` returns `:404`, `:405`, `:411` and nothing else). So the clamp is a
    local, and nothing outside this node can tell the difference. It is also inert on any
    run where `breeder_f` is the solver's -- iteration variable 108 declares bounds
    `(0.060, 1.0)` (`iteration_variables.py:102`), strictly inside the clamp.

    Parameters
    ----------
    a_div_surface_total :
        Total divertor surface area (m2). `.divertor.a_div_surface_total`, from
        `DivertorSurfaceAndPlateMassSingleNull`.
    f_vol_div_coolant, dx_div_plate :
        Divertor coolant volume fraction, plate thickness (m).
        `.divertor.f_vol_div_coolant`, `.divertor.dx_div_plate`.
    vol_blkt_total :
        Blanket volume (m3). `.fwbs.vol_blkt_total`, from
        `blanket_library.BlanketCoverageFactorsSingleNull`.
    f_a_blkt_cooling_channels :
        Blanket coolant channel area fraction. `.fwbs.f_a_blkt_cooling_channels`.
    vol_shld_total, vfshld :
        Shield volume (m3) and void fraction. `.fwbs.vol_shld_total` (`shield.py:138`),
        `.fwbs.vfshld`.
    a_fw_inboard, a_fw_outboard, a_fw_total :
        First-wall areas (m2). `.first_wall.a_fw_inboard`/`_outboard`/`a_fw_total`
        (`fw.py:89-91`).
    dr_fw_inboard, dr_fw_outboard :
        First-wall radial thicknesses (m). `.build.dr_fw_inboard`/`_outboard`.
    f_a_fw_coolant_inboard, f_a_fw_coolant_outboard :
        First-wall coolant void fractions. `.fwbs.f_a_fw_coolant_inboard`/`_outboard`,
        from `FirstWallCoolantVoidFractions`.
    den_steel :
        Steel density (kg/m3). `.fwbs.den_steel`.
    a_plasma_surface :
        Plasma surface area (m2). `.physics.a_plasma_surface`.
    fw_armour_thickness :
        First-wall armour thickness (m). `.fwbs.fw_armour_thickness`.
    breeder_f, breeder_multiplier :
        Lithium-orthosilicate fraction of the breeder, and the breeder/multiplier volume
        fraction. `.fwbs.breeder_f` (iteration variable 108), `.fwbs.breeder_multiplier`.
    vfcblkt, vfpblkt :
        Blanket coolant and purge-gas void fractions. `.fwbs.vfcblkt`, `.fwbs.vfpblkt`.

    Returns
    -------
    tuple
        Eighteen values in PROCESS's own write order -- see the node class below for the
        `VarPath` each lands in.
    """
    # --- coolant volume, in PROCESS's own accumulation order (hcpb.py:298-321) --------
    coolvol = a_div_surface_total * f_vol_div_coolant * dx_div_plate
    coolvol += vol_blkt_total * f_a_blkt_cooling_channels
    coolvol += vol_shld_total * vfshld
    a_fw_coolant_volume = (
        a_fw_inboard * dr_fw_inboard * f_a_fw_coolant_inboard
        + a_fw_outboard * dr_fw_outboard * f_a_fw_coolant_outboard
    )
    coolvol += a_fw_coolant_volume

    # Mass of He coolant at typical coolant temperature and pressure (kg).
    m_fw_blkt_div_coolant_total = coolvol * 1.517

    fwclfr = a_fw_coolant_volume / (a_fw_total * 0.5 * (dr_fw_inboard + dr_fw_outboard))

    # --- shield (hcpb.py:370-377) ----------------------------------------------------
    whtshld = vol_shld_total * den_steel * (1.0 - vfshld)
    # Penetration shield mass is set equal to the internal shield mass.
    wpenshld = whtshld

    # --- first wall and armour (hcpb.py:380-402) -------------------------------------
    vol_fw_total = a_fw_inboard * dr_fw_inboard * (
        1.0 - f_a_fw_coolant_inboard
    ) + a_fw_outboard * dr_fw_outboard * (1.0 - f_a_fw_coolant_outboard)
    m_fw_total = den_steel * vol_fw_total

    fw_armour_vol = a_plasma_surface * fw_armour_thickness
    fw_armour_mass = fw_armour_vol * constants.DEN_TUNGSTEN

    # --- breeder materials (hcpb.py:404-454) -----------------------------------------
    # The clamp stays a local: see the deviation note in this function's docstring.
    breeder_f_clamped = jnp.minimum(jnp.maximum(breeder_f, 1.0e-10), 1.0)

    f_vol_blkt_li4sio4 = breeder_f_clamped * breeder_multiplier
    f_vol_blkt_tibe12 = breeder_multiplier - f_vol_blkt_li4sio4

    m_blkt_tibe12 = vol_blkt_total * f_vol_blkt_tibe12 * 2260.0
    m_blkt_li4sio4 = vol_blkt_total * f_vol_blkt_li4sio4 * 2400.0

    # PROCESS issue #327: the pre-CCFE names are kept as aliases for the cost model.
    m_blkt_beryllium = m_blkt_tibe12
    m_blkt_li2o = m_blkt_li4sio4

    f_vol_blkt_steel = 1.0 - f_vol_blkt_li4sio4 - f_vol_blkt_tibe12 - vfcblkt - vfpblkt
    m_blkt_steel_total = vol_blkt_total * f_vol_blkt_steel * den_steel

    m_blkt_total = m_blkt_tibe12 + m_blkt_li4sio4 + m_blkt_steel_total

    armour_fw_bl_mass = fw_armour_mass + m_fw_total + m_blkt_total

    return (
        m_fw_blkt_div_coolant_total,
        fwclfr,
        whtshld,
        wpenshld,
        vol_fw_total,
        m_fw_total,
        fw_armour_vol,
        fw_armour_mass,
        f_vol_blkt_li4sio4,
        f_vol_blkt_tibe12,
        m_blkt_tibe12,
        m_blkt_li4sio4,
        m_blkt_beryllium,
        m_blkt_li2o,
        f_vol_blkt_steel,
        m_blkt_steel_total,
        m_blkt_total,
        armour_fw_bl_mass,
    )


def calculate_nuclear_heating_magnets_conventional(
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
):
    """Nuclear heating in the magnets, conventional-aspect-ratio arm (`itart == 0`).

    Ports `nuclear_heating_magnets` (`hcpb.py:463-609`), taking the `else` side of its
    three `if self.data.physics.itart == 1` branches (`:515`, `:552`). Closes the
    method's `self.data` back-door -- unlike two of its siblings it is not a
    `@staticmethod`, so this is a genuine extraction (`calculate_` prefix per
    `_audit/naming_convention.md`).

    **Two changes from unit #13's original version**, both required by policy rather than
    by physics:

    - `itart` is gone. The two arms' reads-sets genuinely differ (`itart == 1` reads
      `.tfcoil.whttflgs` and neither `dr_blkt_inboard` nor `dr_shld_inboard` nor
      `.tfcoil.m_tf_coils_total`), so the old single `jnp.where` version declared three
      edges a spherical tokamak does not have and one a conventional one does not. The
      spherical arm is `calculate_nuclear_heating_magnets_spherical_tokamak` below.
    - the two first-wall void fractions are no longer returned. They are
      `FirstWallCoolantVoidFractions`' outputs; `vffwm` here is the same expression kept
      as a local, exactly as PROCESS keeps it.

    `vv_density`'s division is guarded. PROCESS's `if d_vv_all > 1e-6` short-circuits it
    entirely (`:509-512`), and `vol_vv` may legitimately be `0.0` on the untaken side; a
    traced `jnp.where` evaluates both branches, so the denominator is substituted rather
    than the result selected.

    Parameters
    ----------
    radius_fw_channel, dx_fw_module :
        First-wall coolant channel radius (m) and module toroidal width (m).
        `.fwbs.radius_fw_channel`, `.fwbs.dx_fw_module`.
    dr_fw_inboard, dr_fw_outboard :
        First-wall radial thicknesses (m). `.build.dr_fw_inboard`/`_outboard`.
        `dr_fw_outboard` is easy to miss -- it appears only in `x_blanket`'s FW term.
    den_steel :
        Steel density (kg/m3). `.fwbs.den_steel`.
    m_blkt_total, vol_blkt_total :
        Blanket mass (kg) and volume (m3). `.fwbs.m_blkt_total` (`ComponentMasses`),
        `.fwbs.vol_blkt_total`.
    whtshld, vol_shld_total :
        Shield mass (kg) and volume (m3). `.fwbs.whtshld` (`ComponentMasses`),
        `.fwbs.vol_shld_total`.
    dr_vv_inboard, dr_vv_outboard, m_vv, vol_vv :
        Vacuum-vessel thicknesses (m), mass (kg), volume (m3).
        `.build.dr_vv_inboard`/`_outboard`, `.fwbs.m_vv`, `.fwbs.vol_vv`
        (`vacuum.py:796-799`).
    dr_blkt_outboard, dr_blkt_inboard, dr_shld_outboard, dr_shld_inboard :
        Blanket and shield radial thicknesses (m). `.build.*`.
    fw_armour_thickness :
        First-wall armour thickness (m). `.fwbs.fw_armour_thickness`.
    m_tf_coils_total :
        Total TF coil mass (kg). `.tfcoil.m_tf_coils_total`.
    p_fusion_total_mw :
        Total fusion power (MW). `.physics.p_fusion_total_mw`.

    Returns
    -------
    tuple
        `(armour_density, fw_density, blanket_density, shield_density, vv_density,
        x_blanket, x_shield, tfc_nuc_heating, p_tf_nuclear_heat_mw_unnormalised)`.
    """
    # Model factors and coefficients
    a = 2.830  # Exponential factor (m2/tonne)
    b = 0.583  # Exponential factor (m2/tonne)
    e = 9.062  # Pre-factor (1/kg), corrected -- PROCESS issue #272

    # Mean FW coolant void fraction. Same expression as
    # `calculate_fw_coolant_void_fractions`, kept local because that node owns the field.
    vffwm = jnp.pi * radius_fw_channel**2 / (dx_fw_module * dr_fw_inboard)

    # Smeared densities of the blanket sections; the gaseous He coolant mass is neglected
    armour_density = constants.DEN_TUNGSTEN * (1.0 - vffwm)
    fw_density = den_steel * (1.0 - vffwm)
    blanket_density = m_blkt_total / vol_blkt_total
    shield_density = whtshld / vol_shld_total

    d_vv_all = jnp.maximum(dr_vv_inboard, dr_vv_outboard)
    vv_density = jnp.where(
        d_vv_all > 1.0e-6,
        m_vv / jnp.where(vol_vv == 0.0, 1.0, vol_vv),  # noqa: RUF069
        0.0,
    )

    # Average breeding-blanket and neutronic-shield thickness (m)
    th_blanket_av = 0.5 * (dr_blkt_outboard + dr_blkt_inboard)
    th_shield_av = 0.5 * (dr_shld_outboard + dr_shld_inboard)

    # Exponents (tonne/m2); /1000 converts kg to tonnes
    x_blanket = (
        armour_density * fw_armour_thickness
        + fw_density * (dr_fw_inboard + dr_fw_outboard) / 2.0
        + blanket_density * th_blanket_av
    ) / 1000.0

    x_shield = (
        shield_density * th_shield_av
        + vv_density * (dr_vv_inboard + dr_vv_outboard) / 2.0
    ) / 1000.0

    # Unit heating (W/kg/GW of fusion power) x total TF coil mass (kg)
    tfc_nuc_heating = (
        e * jnp.exp(-a * x_blanket) * jnp.exp(-b * x_shield) * m_tf_coils_total
    )

    p_tf_nuclear_heat_mw_unnormalised = (
        tfc_nuc_heating * (p_fusion_total_mw / 1000.0) / 1.0e6
    )

    return (
        armour_density,
        fw_density,
        blanket_density,
        shield_density,
        vv_density,
        x_blanket,
        x_shield,
        tfc_nuc_heating,
        p_tf_nuclear_heat_mw_unnormalised,
    )


def calculate_nuclear_heating_magnets_spherical_tokamak(
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
):
    """Nuclear heating in the magnets, spherical-tokamak arm (`itart == 1`).

    The `itart == 1` side of `hcpb.py:515-569`. A tight-aspect-ratio machine has no
    inboard blanket and its centrepost shield is a separate calculation, so the average
    thicknesses are the outboard values alone and the heated mass is the **outboard TF
    leg mass** `.tfcoil.whttflgs` rather than the whole coil set. Written to preserve
    unit #13's coverage of this arm; **not registerable on the current tokamak tree**,
    because at `itart == 1` `run()` also takes the centrepost-neutronics branch
    (`hcpb.py:103-141`) and `component_volumes` takes the D-shaped branch, neither of
    which is ported. See the audit record's UNPORTED list.

    Parameters
    ----------
    whttflgs :
        Mass of the outboard TF coil legs (kg). `.tfcoil.whttflgs`.
    radius_fw_channel, dx_fw_module, dr_fw_inboard, dr_fw_outboard, den_steel,
    m_blkt_total, vol_blkt_total, whtshld, vol_shld_total, dr_vv_inboard,
    dr_vv_outboard, m_vv, vol_vv, dr_blkt_outboard, dr_shld_outboard,
    fw_armour_thickness, p_fusion_total_mw :
        As the conventional arm above. Note the absence of `dr_blkt_inboard`,
        `dr_shld_inboard` and `m_tf_coils_total`: this arm does not read them.

    Returns
    -------
    tuple
        The same nine values as the conventional arm.
    """
    a = 2.830
    b = 0.583
    e = 9.062

    vffwm = jnp.pi * radius_fw_channel**2 / (dx_fw_module * dr_fw_inboard)

    armour_density = constants.DEN_TUNGSTEN * (1.0 - vffwm)
    fw_density = den_steel * (1.0 - vffwm)
    blanket_density = m_blkt_total / vol_blkt_total
    shield_density = whtshld / vol_shld_total

    d_vv_all = jnp.maximum(dr_vv_inboard, dr_vv_outboard)
    vv_density = jnp.where(
        d_vv_all > 1.0e-6,
        m_vv / jnp.where(vol_vv == 0.0, 1.0, vol_vv),  # noqa: RUF069
        0.0,
    )

    # No inboard blanket on a TART; the CP shield is a separate calculation.
    th_blanket_av = dr_blkt_outboard
    th_shield_av = dr_shld_outboard

    x_blanket = (
        armour_density * fw_armour_thickness
        + fw_density * (dr_fw_inboard + dr_fw_outboard) / 2.0
        + blanket_density * th_blanket_av
    ) / 1000.0

    x_shield = (
        shield_density * th_shield_av
        + vv_density * (dr_vv_inboard + dr_vv_outboard) / 2.0
    ) / 1000.0

    tfc_nuc_heating = e * jnp.exp(-a * x_blanket) * jnp.exp(-b * x_shield) * whttflgs

    p_tf_nuclear_heat_mw_unnormalised = (
        tfc_nuc_heating * (p_fusion_total_mw / 1000.0) / 1.0e6
    )

    return (
        armour_density,
        fw_density,
        blanket_density,
        shield_density,
        vv_density,
        x_blanket,
        x_shield,
        tfc_nuc_heating,
        p_tf_nuclear_heat_mw_unnormalised,
    )


def nuclear_heating_fw(m_fw_total, fw_armour_u_nuc_heating, p_fusion_total_mw):
    """Nuclear heating in the first wall (MW), before renormalisation.

    Ports the `@staticmethod` of the same name (`hcpb.py:611-651`) -- already pure in the
    source.

    PROCESS raises `ProcessValueError` when the result is negative (`:646-650`). A traced
    function cannot raise on a data-dependent condition, so the port returns `nan` there
    instead, which is what `_harness.contracts.Tier1Contract.reference_domain_errors`
    asserts. The untaken branch is a *constant*, so its derivative is zero and no NaN
    leaks into the tangent.

    Parameters
    ----------
    m_fw_total :
        Total first-wall mass, excluding armour (kg). `.fwbs.m_fw_total`, from
        `ComponentMasses`.
    fw_armour_u_nuc_heating :
        Unit nuclear heating of FW and armour (W/kg per W of fusion power).
        `.ccfe_hcpb.fw_armour_u_nuc_heating` -- an input; nothing under `process/models/`
        writes it.
    p_fusion_total_mw :
        Total fusion power (MW). `.physics.p_fusion_total_mw`.

    Returns
    -------
    :
        `p_fw_nuclear_heat_total_mw_unnormalised` (MW).
    """
    p_fw_nuclear_heat_total_mw = m_fw_total * fw_armour_u_nuc_heating * p_fusion_total_mw

    return jnp.where(
        p_fw_nuclear_heat_total_mw < 0.0, jnp.nan, p_fw_nuclear_heat_total_mw
    )


def nuclear_heating_blanket(m_blkt_total, p_fusion_total_mw):
    """Nuclear heating in the blanket (MW), before renormalisation.

    Ports the `@staticmethod` of the same name (`hcpb.py:653-698`) verbatim -- already
    pure in the source, no `self` to close.

    The source's `logger.error(...)` diagnostic (gated on
    `p_blkt_nuclear_heat_total_mw < 1`) is dropped: a Python-level side effect
    conditioned on a traced value, with no effect on the return value.

    Parameters
    ----------
    m_blkt_total :
        Total mass of the blanket (kg). `.fwbs.m_blkt_total`, from `ComponentMasses`.
    p_fusion_total_mw :
        Total fusion power (MW). `.physics.p_fusion_total_mw`.

    Returns
    -------
    tuple
        `(p_blkt_nuclear_heat_total_mw_unnormalised, exp_blanket)`.
    """
    a = 0.764
    b = 2.476e-3  # 1/tonne

    m_blkt_total_tonnes = m_blkt_total / 1000

    exp_blanket = 1 - jnp.exp(-b * m_blkt_total_tonnes)
    p_blkt_nuclear_heat_total_mw = p_fusion_total_mw * a * exp_blanket

    return p_blkt_nuclear_heat_total_mw, exp_blanket


def nuclear_heating_shield_conventional(
    dr_shld_outboard,
    dr_shld_inboard,
    shield_density,
    whtshld,
    x_blanket,
    p_fusion_total_mw,
):
    """Nuclear heating in the shield (MW), conventional arm (`itart == 0`).

    The `else` side of `nuclear_heating_shield`'s `if itart == 1` (`hcpb.py:751-756`).
    `itart` is no longer a parameter: the switch selected this occupant, so it is gone
    from the body, and with it the invented dependence on `dr_shld_inboard` that a
    spherical tokamak does not have.

    Parameters
    ----------
    dr_shld_outboard, dr_shld_inboard :
        Outboard/inboard shield thickness (m). `.build.dr_shld_outboard`/`_inboard`.
    shield_density :
        Shield smeared density (kg/m3). `.ccfe_hcpb.shield_density`, from
        `NuclearHeatingMagnetsConventional`.
    whtshld :
        Shield mass (kg). `.fwbs.whtshld`, from `ComponentMasses`.
    x_blanket :
        Blanket line density (tonne/m2). `.ccfe_hcpb.x_blanket`, same producer as
        `shield_density`.
    p_fusion_total_mw :
        Total fusion power (MW). `.physics.p_fusion_total_mw`.

    Returns
    -------
    tuple
        `(p_shld_nuclear_heat_mw_unnormalised, exp_shield1, exp_shield2,
        shld_u_nuc_heating)`.
    """
    return _nuclear_heating_shield(
        dr_shld_average=0.5 * (dr_shld_outboard + dr_shld_inboard),
        shield_density=shield_density,
        whtshld=whtshld,
        x_blanket=x_blanket,
        p_fusion_total_mw=p_fusion_total_mw,
    )


def nuclear_heating_shield_spherical_tokamak(
    dr_shld_outboard,
    shield_density,
    whtshld,
    x_blanket,
    p_fusion_total_mw,
):
    """Nuclear heating in the shield (MW), spherical-tokamak arm (`itart == 1`).

    The `itart == 1` side of `hcpb.py:751-753`: the centrepost shield is a separate
    calculation, so the average shield thickness is the outboard value alone and
    `dr_shld_inboard` is not read. Written to preserve unit #13's coverage of this arm;
    not registerable until the centrepost chain is ported (see
    `calculate_nuclear_heating_magnets_spherical_tokamak`).

    Parameters
    ----------
    dr_shld_outboard, shield_density, whtshld, x_blanket, p_fusion_total_mw :
        As the conventional arm above, less `dr_shld_inboard`.

    Returns
    -------
    tuple
        The same four values as the conventional arm.
    """
    return _nuclear_heating_shield(
        dr_shld_average=dr_shld_outboard,
        shield_density=shield_density,
        whtshld=whtshld,
        x_blanket=x_blanket,
        p_fusion_total_mw=p_fusion_total_mw,
    )


def _nuclear_heating_shield(
    dr_shld_average, shield_density, whtshld, x_blanket, p_fusion_total_mw
):
    """The part of `nuclear_heating_shield` both `itart` arms share (`hcpb.py:758-769`).

    A private helper rather than a node: the two occupants differ only in how
    `dr_shld_average` is formed, and duplicating twelve lines to make that point would
    add a second place for the coefficients to drift.

    Parameters
    ----------
    dr_shld_average :
        Average neutronic shield thickness (m), formed by the calling arm.
    shield_density, whtshld, x_blanket, p_fusion_total_mw :
        As the public arms.

    Returns
    -------
    tuple
        `(p_shld_nuclear_heat_mw_unnormalised, exp_shield1, exp_shield2,
        shld_u_nuc_heating)`.
    """
    f = 6.88e2  # Shield nuclear heating coefficient (W/kg/W)
    g = 2.723  # Shield nuclear heating exponent (m2/tonne)
    h = 0.798  # Shield nuclear heating exponent (m2/tonne)

    # Decay length (m^-2)
    y = (shield_density / 1000) * dr_shld_average

    exp_shield1 = jnp.exp(-g * x_blanket)
    exp_shield2 = jnp.exp(-h * y)
    shld_u_nuc_heating = whtshld * f * exp_shield1 * exp_shield2

    p_shld_nuclear_heat_mw = shld_u_nuc_heating * (p_fusion_total_mw / 1000) / 1.0e6

    return p_shld_nuclear_heat_mw, exp_shield1, exp_shield2, shld_u_nuc_heating


def calculate_centrepost_neutronics_absent():
    """The four centrepost fields a machine without a centrepost still has to have.

    Ports the `else` arm of `run()`'s `if self.data.physics.itart == 1`
    (`hcpb.py:143-148`): four literal zeros. They are a node rather than nothing, because
    three of the four have real readers elsewhere in the tokamak surface and would
    otherwise surface as boundary inputs served by a dataclass default --
    `.fwbs.pnuc_cp_tf` at `power.py:1095` and `tfcoil/base.py:1264`, and
    `.fwbs.neut_flux_cp` at `availability.py:1553`/`:1557`.

    `.fwbs.p_cp_shield_nuclear_heat_mw` is written twice by `run()` on this arm -- `:146`
    and again at `:267` as `f_geom_cp * p_neutron_total_mw - pnuc_cp_tf`, which is
    `0 * x - 0` -- so it is owned here once rather than by the renormalisation node. That
    is a `redundant-duplicate-write` in `schema.md`'s sense, resolved by picking one
    owner rather than by reproducing both writes.

    Returns
    -------
    tuple
        `(pnuc_cp_tf, p_cp_shield_nuclear_heat_mw, pnuc_cp, neut_flux_cp)`, all `0.0`.
    """
    return 0.0, 0.0, 0.0, 0.0


def calculate_centrepost_angle_fraction(z_cp_top, r_cp_mid, r_cp_top, rmajor):
    """Solid-angle fraction of the plasma's neutrons intercepted by the centrepost.

    Ports `CCFE_HCPB.st_cp_angle_fraction` (`hcpb.py:1008-1080`) verbatim, including its
    parameter order -- which does **not** match its own docstring's order, and does not
    have to: `rho_maj` reads `r_cp_mid + r_cp_top` symmetrically, so the two radii are
    interchangeable in the arithmetic and only the names would mislead.

    Equations (1-3) of P. Guest, *Rev. Sci. Instrum.* **32** (2), 1960, integrated over
    the toroidal half-angle the centrepost subtends by a ten-panel trapezoid rule. The
    trip count is a literal `10` in the source, so the loop is unrolled at trace time and
    nothing here is data-dependent.

    Two transcription notes:

    - PROCESS's `max(int_calc_3, 0.0)` (`:1059`, `:1069`) is `jnp.maximum`. It is not a
      guard against bad input but against *rounding*: the integral's last panel sits at
      `phy_cp = arcsin(1 / rho_maj)`, where `1 - rho_maj**2 * sin(phy_cp)**2` is
      analytically zero and numerically either sign.
    - Because that radicand really is zero at the last panel, its square root is taken
      through `safe_sqrt`. `jnp.sqrt` is value-correct there and has an infinite
      derivative, which is `next_steps.md` §9's defect exactly -- and this is the first
      site in the port where the zero is reached on the *reference operating point*
      rather than only at a fuzzed edge.

    Parameters
    ----------
    z_cp_top :
        Centrepost shield half height (m). `run():113` passes
        `.build.z_plasma_xpoint_upper`.
    r_cp_mid :
        Centrepost mid-plane radius (m). `run():122` passes `.build.r_sh_inboard_out`.
    r_cp_top :
        Centrepost top radius (m). `run():106-110` forms it as
        `rmajor - rminor * triang - 3 * dr_fw_plasma_gap_inboard`.
    rmajor :
        Plasma major radius (m). `.physics.rmajor`.

    Returns
    -------
    :
        Solid-angle fraction covered by the centrepost (-). A local in PROCESS's
        `run()`; the port mints it as `.ccfe_hcpb.f_geom_cp` because it crosses a node
        boundary into the renormalisation.
    """
    n_integral = 10

    # Major radius normalised to the CP average radius [-]
    rho_maj = 2.0 * rmajor / (r_cp_mid + r_cp_top)

    # Average CP extent in the toroidal plane [rad]
    phy_cp = jnp.arcsin(1.0 / rho_maj)

    # toroidal plane infinitesimal angle used in the integral [rad]
    d_phy_cp = phy_cp / n_integral

    # CP solid angle integral using trapezoidal method
    phy_cp_calc = 0.0
    cp_sol_angle = 0.0

    for _ in range(n_integral):
        # Little tricks to avoild NaNs due to rounding
        int_calc_3 = 1.0 - rho_maj**2 * jnp.sin(phy_cp_calc) ** 2
        int_calc_3 = jnp.maximum(int_calc_3, 0.0)

        int_calc_1 = 1.0 / jnp.sqrt(
            z_cp_top**2 + (rho_maj * jnp.cos(phy_cp_calc) - safe_sqrt(int_calc_3)) ** 2
        )

        phy_cp_calc += d_phy_cp

        # Little tricks to avoild NaNs due to rounding
        int_calc_3 = 1.0 - rho_maj**2 * jnp.sin(phy_cp_calc) ** 2
        int_calc_3 = jnp.maximum(int_calc_3, 0.0)

        int_calc_2 = 1.0 / jnp.sqrt(
            z_cp_top**2 + (rho_maj * jnp.cos(phy_cp_calc) - safe_sqrt(int_calc_3)) ** 2
        )

        cp_sol_angle += d_phy_cp * 0.5 * (int_calc_1 + int_calc_2)

    cp_sol_angle = cp_sol_angle * 4.0 * z_cp_top

    # Solid angle fraction covered by the CP (OUTPUT) [-]
    return 0.25 * cp_sol_angle / jnp.pi


def calculate_centrepost_fast_neutron_flux_superconducting(
    p_neutron_total_mw, sh_width, rmajor
):
    """Fast neutron flux (E > 0.1 MeV) reaching the TF at the centrepost.

    Ports the `i_tf_sup == SUPERCONDUCTING` arm of
    `CCFE_HCPB.st_tf_centrepost_fast_neut_flux` (`hcpb.py:1082-1134`). The other two
    conductor models take no branch at all: `neut_flux_cp` is initialised to `0`
    (`:1112`) and the `if` at `:1114` never fires, so a water-cooled-copper or
    aluminium centrepost returns the literal zero. That is a *different* occupant, not
    this one with a zeroed input, which is why the slot is keyed on the joint
    `(itart, i_tf_sup)` arm rather than on `itart` alone.

    The fit is a CP-only MCNP scan with a variable tungsten-carbide shield at 13% water
    cooling; the shielding length per decade is 16.6 cm (e-folding 7.22 cm), close to the
    "15 - 16 cm" of Menard et al. 2016.

    Parameters
    ----------
    p_neutron_total_mw :
        Neutron fusion power (MW). `.physics.p_neutron_total_mw`.
    sh_width :
        Neutron shield width (m). `run():130` passes `.build.dr_shld_inboard`.
    rmajor :
        Plasma major radius (m). `.physics.rmajor`.

    Returns
    -------
    :
        Centrepost fast neutron flux (m^-2 s^-1). `.fwbs.neut_flux_cp`.
    """
    # Fraction of fast neutrons originating from the outer wall reflection [-]
    f_neut_flux_out_wall = 1

    # Tungsten density may vary with different manufacturing processes.
    f_wc_density = 2

    # Fraction of steel structures
    f_steel_struct = 0.1

    # Effecting shield width, removing steel structures
    sh_width_eff = sh_width * (1.0 - f_steel_struct)

    # Fit [10^{-13}.cm^{-2}]
    neut_flux_cp = 5.835 * jnp.exp(-15.392 * sh_width_eff) + 39.70 * (
        sh_width_eff / rmajor
    ) * jnp.exp(-24.722 * sh_width_eff)

    # Units conversion [10^{-13}.cm^{-2}] -> [m^{-2}]
    neut_flux_cp *= 1.0e17

    # Scaling to the actual plasma neutron power
    return (
        f_wc_density * f_neut_flux_out_wall * neut_flux_cp * (p_neutron_total_mw / 800)
    )


def calculate_centrepost_nuclear_heating_superconducting(pneut, sh_width, rmajor):
    """Nuclear heat deposited in a superconducting or copper spherical centrepost.

    Ports the `else` arm of `CCFE_HCPB.st_centrepost_nuclear_heating`
    (`hcpb.py:1200-1285`). **That arm is `i_tf_sup in {0, 1}`**, not `i_tf_sup == 1`:
    PROCESS branches on `HELIUM_COOLED_ALUMINIUM` alone (`:1192`) and the comment at
    `:1202-1204` says why -- the MCNP model's winding pack is large enough to be mostly
    copper, so the same fit serves the superconducting and the water-cooled-copper
    centrepost. This partition is **not** the fast-neutron-flux routine's, which splits
    `{1}` from `{0, 2}`; the two functions read the same switch and cut it in different
    places, which is the whole reason the port keys their shared slot on a joint arm.

    The shielding length per decade is 15.5 cm (e-folding 6.72 cm), again within Menard
    et al. 2016's "15 - 16 cm". Six fits -- winding pack and steel case, each by gammas
    and by neutrons, plus the two shield terms -- are evaluated for an 800 MW plasma
    neutron source and rescaled.

    Parameters
    ----------
    pneut :
        14 MeV plasma neutron power (MW). `.physics.p_neutron_total_mw`.
    sh_width :
        Centrepost neutron shield thickness (m). `.build.dr_shld_inboard`.
    rmajor :
        Plasma major radius (m). `.physics.rmajor` -- read off `self.data` in the source
        (`:1216`, `:1223`, `:1230`, `:1237`) rather than passed, which is the one
        `self`-bound read this extraction closes.

    Returns
    -------
    tuple
        `(pnuc_cp_tf, p_cp_shield_nuclear_heat_mw, pnuc_cp)`. The middle value is
        **overwritten** by `run():267` on this arm, so the port mints it as
        `.ccfe_hcpb.p_cp_shield_nuclear_heat_mw_fit`; see
        `calculate_nuclear_heating_renormalisation_single_null_spherical_tokamak`.
    """
    # Outer wall reflection TF nuclear heating enhancement factor [-]
    f_pnuc_cp_tf = 1

    # Outer wall reflection shield nuclear heating enhancement factor [-]
    f_pnuc_cp_sh = 1.7

    # Tungsten density may vary with different manufacturing processes.
    f_wc_density = 2

    # Fraction of steel structures
    f_steel_struct = 0.1

    # Steel support structure effective WC shield thickness reduction
    sh_width_eff = sh_width * (1 - f_steel_struct)

    # Nuclear power deposited in the CP winding pack by gammas [MW]
    pnuc_cp_wp_gam = 16.3 * jnp.exp(-14.63 * sh_width_eff) + 143.08 * sh_width_eff * (
        sh_width / rmajor
    ) * jnp.exp(-21.747 * sh_width_eff)

    # Nuclear power deposited in the CP winding pack by neutrons [MW]
    pnuc_cp_wp_n = 1.403 * jnp.exp(-16.535 * sh_width_eff) + 3.812 * sh_width_eff * (
        sh_width / rmajor
    ) * jnp.exp(-23.631 * sh_width_eff)

    # Nuclear power deposited in the CP steel case by gammas [MW]
    pnuc_cp_case_gam = 1.802 * jnp.exp(-13.993 * sh_width_eff) + 38.592 * sh_width * (
        sh_width_eff / rmajor
    ) * jnp.exp(-27.051 * sh_width_eff)

    # Nuclear power deposited in the CP steel case by neutrons [MW]
    pnuc_cp_case_n = 0.158 * jnp.exp(-55.046 * sh_width_eff) + 2.0742 * sh_width_eff * (
        sh_width / rmajor
    ) * jnp.exp(-24.401 * sh_width_eff)

    # Nuclear power density deposited in the tungsten carbide shield by photons [MW]
    pnuc_cp_sh_gam = sh_width_eff * (
        596 * jnp.exp(-4.130 * sh_width_eff) + 90.586 * jnp.exp(0.6837 * sh_width_eff)
    )

    # Nuclear power density deposited in the tungsten carbide shield by neutrons [MW]
    pnuc_cp_sh_n = sh_width_eff * (
        202.10 * jnp.exp(-10.533 * sh_width_eff)
        + 80.510 * jnp.exp(-0.9801 * sh_width_eff)
    )

    # Correction for the actual 14 MeV plasma neutron power
    pnuc_cp_wp_gam = (pneut / 800) * pnuc_cp_wp_gam
    pnuc_cp_wp_n = (pneut / 800) * pnuc_cp_wp_n
    pnuc_cp_case_gam = (pneut / 800) * pnuc_cp_case_gam
    pnuc_cp_case_n = (pneut / 800) * pnuc_cp_case_n
    pnuc_cp_sh_gam = (pneut / 800) * pnuc_cp_sh_gam
    pnuc_cp_sh_n = (pneut / 800) * pnuc_cp_sh_n

    # Correction for neutron reflected by the outer wall hitting the CP
    pnuc_cp_wp_gam = f_pnuc_cp_tf * pnuc_cp_wp_gam
    pnuc_cp_wp_n = f_pnuc_cp_tf * pnuc_cp_wp_n
    pnuc_cp_case_gam = f_pnuc_cp_tf * pnuc_cp_case_gam
    pnuc_cp_case_n = f_pnuc_cp_tf * pnuc_cp_case_n
    pnuc_cp_sh_gam = f_pnuc_cp_sh * pnuc_cp_sh_gam
    pnuc_cp_sh_n = f_pnuc_cp_sh * pnuc_cp_sh_n

    # TF nuclear heat [MW]
    pnuc_cp_tf = pnuc_cp_wp_gam + pnuc_cp_wp_n + pnuc_cp_case_gam + pnuc_cp_case_n

    # Tungsten density correction
    pnuc_cp_tf *= f_wc_density

    # Shield nuclear heat [MW]
    p_cp_shield_nuclear_heat_mw = pnuc_cp_sh_gam + pnuc_cp_sh_n

    # Total CP nuclear heat [MW]
    pnuc_cp = pnuc_cp_tf + p_cp_shield_nuclear_heat_mw

    return pnuc_cp_tf, p_cp_shield_nuclear_heat_mw, pnuc_cp


def calculate_centrepost_neutronics_spherical_tokamak_superconducting(
    rmajor,
    rminor,
    triang,
    dr_fw_plasma_gap_inboard,
    z_plasma_xpoint_upper,
    r_sh_inboard_out,
    p_neutron_total_mw,
    dr_shld_inboard,
):
    """`run()`'s `itart == 1` centrepost block (`hcpb.py:103-141`), at `i_tf_sup == 1`.

    The spherical counterpart of `calculate_centrepost_neutronics_absent`, and the
    occupant that makes `.physics.itart == 1` assemblable at all. One node rather than
    three, because PROCESS's `:103-141` is one straight-line block with no branch inside
    it once the conductor model is known, and because the three routines' outputs have
    a single consumer between them (the renormalisation). Splitting it would buy finer
    SCC granularity and nothing else -- measured acyclic here, so nothing else is what it
    would buy.

    The two locals `run()` forms before calling out are kept as locals:
    `r_sh_inboard_out_top` (`:106-110`, the CP radius at the X-point, where the shield is
    widest) and `h_sh_max_r` (`:113`, a pure alias of `.build.z_plasma_xpoint_upper`).

    **`.build.r_sh_inboard_out` has no producer in this port.** `build.py:1858` writes
    it, accumulating *outwards* from the central-solenoid bore, and that chain is
    deliberately outside the ported closure -- `models/namespace.py` records why
    `.build.r_shld_inboard_inner`, built inwards from the plasma, is the radius the port
    produces. The two are equal on a self-consistent build and are **not** the same
    expression, so this port declares a boundary input rather than substituting one for
    the other.

    Parameters
    ----------
    rmajor, rminor, triang :
        Plasma major radius (m), minor radius (m) and triangularity (-). `.physics.*`.
    dr_fw_plasma_gap_inboard :
        Inboard first-wall-to-plasma gap (m). `.build.dr_fw_plasma_gap_inboard`.
    z_plasma_xpoint_upper :
        Height of the upper plasma X-point (m). `.build.z_plasma_xpoint_upper`.
    r_sh_inboard_out :
        Plasma-facing radius of the inboard neutronic shield (m).
        `.build.r_sh_inboard_out` -- the boundary input above.
    p_neutron_total_mw :
        Total neutron power (MW). `.physics.p_neutron_total_mw`.
    dr_shld_inboard :
        Inboard neutronic shield thickness (m). `.build.dr_shld_inboard`.

    Returns
    -------
    tuple
        `(f_geom_cp, neut_flux_cp, pnuc_cp_tf, p_cp_shield_nuclear_heat_mw_fit,
        pnuc_cp)`.
    """
    # CP radius at the point of maximum shield radius [m]. The maximum shield radius is
    # assumed to be at the X-point.
    r_sh_inboard_out_top = rmajor - rminor * triang - 3 * dr_fw_plasma_gap_inboard

    # Half height of the CP at the largest shield radius [m]
    h_sh_max_r = z_plasma_xpoint_upper

    f_geom_cp = calculate_centrepost_angle_fraction(
        h_sh_max_r, r_sh_inboard_out, r_sh_inboard_out_top, rmajor
    )

    neut_flux_cp = calculate_centrepost_fast_neutron_flux_superconducting(
        p_neutron_total_mw, dr_shld_inboard, rmajor
    )

    (
        pnuc_cp_tf,
        p_cp_shield_nuclear_heat_mw_fit,
        pnuc_cp,
    ) = calculate_centrepost_nuclear_heating_superconducting(
        p_neutron_total_mw, dr_shld_inboard, rmajor
    )

    return (
        f_geom_cp,
        neut_flux_cp,
        pnuc_cp_tf,
        p_cp_shield_nuclear_heat_mw_fit,
        pnuc_cp,
    )


def calculate_nuclear_heating_renormalisation_single_null_conventional(
    p_fw_nuclear_heat_total_mw_unnormalised,
    p_blkt_nuclear_heat_total_mw_unnormalised,
    p_shld_nuclear_heat_mw_unnormalised,
    p_tf_nuclear_heat_mw_unnormalised,
    f_ster_div_single,
    f_p_blkt_multiplication,
    p_neutron_total_mw,
):
    """Renormalise the four nuclear heating powers onto the neutron power budget.

    Ports `CCFE_HCPB.run()`'s `hcpb.py:195-276` -- the block that turns the four
    `nuclear_heating_*` routines' raw answers into the values every downstream consumer
    reads. It is not a method in PROCESS, only lines inside `run()`, which is why its
    harness case drives PROCESS's own `run()` through a subclass whose other steps are
    stubbed rather than transcribing the arithmetic (see `test_hcpb.py`).

    **Two switch values are baked in, and that is what removes two dead edges:**

    - `n_divertors == 1`, so `f_geom_blanket = 1 - f_ster_div_single - f_geom_cp`
      (`hcpb.py:213-217`) rather than carrying a traced count.
    - `itart == 0`, so `f_geom_cp` is the literal `0` PROCESS assigns at `:144` and
      `.fwbs.pnuc_cp_tf` the literal `0` it assigns at `:145`. Neither is read here as an
      edge: `+ pnuc_cp_tf` at `:263` is provably `+ 0` on this arm, and
      `f_geom_cp * p_neutron_total_mw` at `:268` is provably `0`. On the `itart == 1` arm
      both are live and `p_cp_shield_nuclear_heat_mw` is a real quantity -- which is
      exactly why the arms are different occupants.

    Parameters
    ----------
    p_fw_nuclear_heat_total_mw_unnormalised :
        First-wall nuclear heating before renormalisation (MW).
        `.ccfe_hcpb.p_fw_nuclear_heat_total_mw_unnormalised`, from `NuclearHeatingFw`.
    p_blkt_nuclear_heat_total_mw_unnormalised :
        Blanket, same. From `NuclearHeatingBlanket`.
    p_shld_nuclear_heat_mw_unnormalised :
        Shield, same. From `NuclearHeatingShieldConventional`.
    p_tf_nuclear_heat_mw_unnormalised :
        TF coils, same. From `NuclearHeatingMagnetsConventional`.
    f_ster_div_single :
        Divertor solid-angle fraction per divertor. `.fwbs.f_ster_div_single`
        (`divertor.py:42`).
    f_p_blkt_multiplication :
        Blanket neutron energy multiplication factor. `.fwbs.f_p_blkt_multiplication`.
    p_neutron_total_mw :
        Total neutron power (MW). `.physics.p_neutron_total_mw`.

    Returns
    -------
    tuple
        `(pnuc_tot_blk_sector, p_fw_nuclear_heat_total_mw,
        p_blkt_nuclear_heat_total_mw, p_shld_nuclear_heat_mw, p_tf_nuclear_heat_mw,
        p_blkt_multiplication_mw)`.
    """
    # Total nuclear power deposited in the blanket sector (MW)
    pnuc_tot_blk_sector = (
        p_fw_nuclear_heat_total_mw_unnormalised
        + p_blkt_nuclear_heat_total_mw_unnormalised
        + p_shld_nuclear_heat_mw_unnormalised
        + p_tf_nuclear_heat_mw_unnormalised
    )

    # Solid angle fraction taken by the breeding blankets/shields. `f_geom_cp` is 0 on
    # this arm and `n_divertors` is 1, so PROCESS's
    # `1 - n_divertors * f_ster_div_single - f_geom_cp` reduces to this.
    f_geom_blanket = 1 - f_ster_div_single

    normalisation = f_p_blkt_multiplication * f_geom_blanket * p_neutron_total_mw

    p_fw_nuclear_heat_total_mw = (
        p_fw_nuclear_heat_total_mw_unnormalised / pnuc_tot_blk_sector
    ) * normalisation
    p_blkt_nuclear_heat_total_mw = (
        p_blkt_nuclear_heat_total_mw_unnormalised / pnuc_tot_blk_sector
    ) * normalisation
    # The power deposited in the CP shield is added back in `powerflow_calc`.
    p_shld_nuclear_heat_mw = (
        p_shld_nuclear_heat_mw_unnormalised / pnuc_tot_blk_sector
    ) * normalisation
    # `+ pnuc_cp_tf` in the source; it is 0 on this arm.
    p_tf_nuclear_heat_mw = (
        p_tf_nuclear_heat_mw_unnormalised / pnuc_tot_blk_sector
    ) * normalisation

    p_blkt_multiplication_mw = (
        (f_p_blkt_multiplication - 1) * f_geom_blanket * p_neutron_total_mw
    )

    return (
        pnuc_tot_blk_sector,
        p_fw_nuclear_heat_total_mw,
        p_blkt_nuclear_heat_total_mw,
        p_shld_nuclear_heat_mw,
        p_tf_nuclear_heat_mw,
        p_blkt_multiplication_mw,
    )


def calculate_nuclear_heating_renormalisation_double_null_conventional(
    p_fw_nuclear_heat_total_mw_unnormalised,
    p_blkt_nuclear_heat_total_mw_unnormalised,
    p_shld_nuclear_heat_mw_unnormalised,
    p_tf_nuclear_heat_mw_unnormalised,
    f_ster_div_single,
    f_p_blkt_multiplication,
    p_neutron_total_mw,
):
    """The renormalisation of `hcpb.py:195-276` at `n_divertors == 2`, `itart == 0`.

    Identical to
    `calculate_nuclear_heating_renormalisation_single_null_conventional` except in one
    place: `f_geom_blanket`. PROCESS writes it as
    `1 - n_divertors * f_ster_div_single - f_geom_cp` (`hcpb.py:213-217`) -- **not** as
    an `if`, but as a multiplication by the divertor count, which is exactly why the
    two arms are the same shape with a different constant. With `n_divertors == 2` and
    `f_geom_cp` the literal `0` of the conventional arm (`:144`), it is
    `1 - 2 * f_ster_div_single`.

    The `itart == 0` half of the bake is unchanged and carries the same two
    non-declarations: `+ pnuc_cp_tf` at `:263` is provably `+ 0` here, and
    `f_geom_cp * p_neutron_total_mw` at `:268` provably `0`, so neither is a read. The
    `itart == 1` flavour of this arm is UNPORTED for the same reason its single-null
    counterpart is -- see `indat.py`'s `('itart_hcpb', 1)`.

    Parameters
    ----------
    p_fw_nuclear_heat_total_mw_unnormalised :
        First-wall nuclear heating before renormalisation (MW).
        `.ccfe_hcpb.p_fw_nuclear_heat_total_mw_unnormalised`, from `NuclearHeatingFw`.
    p_blkt_nuclear_heat_total_mw_unnormalised :
        Blanket, same. From `NuclearHeatingBlanket`.
    p_shld_nuclear_heat_mw_unnormalised :
        Shield, same. From `NuclearHeatingShieldConventional`.
    p_tf_nuclear_heat_mw_unnormalised :
        TF coils, same. From `NuclearHeatingMagnetsConventional`.
    f_ster_div_single :
        Divertor solid-angle fraction **per divertor**. `.fwbs.f_ster_div_single`
        (`divertor.py:42`).
    f_p_blkt_multiplication :
        Blanket neutron energy multiplication factor. `.fwbs.f_p_blkt_multiplication`.
    p_neutron_total_mw :
        Total neutron power (MW). `.physics.p_neutron_total_mw`.

    Returns
    -------
    tuple
        `(pnuc_tot_blk_sector, p_fw_nuclear_heat_total_mw,
        p_blkt_nuclear_heat_total_mw, p_shld_nuclear_heat_mw, p_tf_nuclear_heat_mw,
        p_blkt_multiplication_mw)`.
    """
    # Total nuclear power deposited in the blanket sector (MW)
    pnuc_tot_blk_sector = (
        p_fw_nuclear_heat_total_mw_unnormalised
        + p_blkt_nuclear_heat_total_mw_unnormalised
        + p_shld_nuclear_heat_mw_unnormalised
        + p_tf_nuclear_heat_mw_unnormalised
    )

    # `1 - n_divertors * f_ster_div_single - f_geom_cp` at `n_divertors == 2` and
    # `f_geom_cp == 0` (the conventional arm's literal, `hcpb.py:144`).
    f_geom_blanket = 1 - 2 * f_ster_div_single

    normalisation = f_p_blkt_multiplication * f_geom_blanket * p_neutron_total_mw

    p_fw_nuclear_heat_total_mw = (
        p_fw_nuclear_heat_total_mw_unnormalised / pnuc_tot_blk_sector
    ) * normalisation
    p_blkt_nuclear_heat_total_mw = (
        p_blkt_nuclear_heat_total_mw_unnormalised / pnuc_tot_blk_sector
    ) * normalisation
    # The power deposited in the CP shield is added back in `powerflow_calc`.
    p_shld_nuclear_heat_mw = (
        p_shld_nuclear_heat_mw_unnormalised / pnuc_tot_blk_sector
    ) * normalisation
    # `+ pnuc_cp_tf` in the source; it is 0 on this arm.
    p_tf_nuclear_heat_mw = (
        p_tf_nuclear_heat_mw_unnormalised / pnuc_tot_blk_sector
    ) * normalisation

    p_blkt_multiplication_mw = (
        (f_p_blkt_multiplication - 1) * f_geom_blanket * p_neutron_total_mw
    )

    return (
        pnuc_tot_blk_sector,
        p_fw_nuclear_heat_total_mw,
        p_blkt_nuclear_heat_total_mw,
        p_shld_nuclear_heat_mw,
        p_tf_nuclear_heat_mw,
        p_blkt_multiplication_mw,
    )


def calculate_nuclear_heating_renormalisation_single_null_spherical_tokamak(
    p_fw_nuclear_heat_total_mw_unnormalised,
    p_blkt_nuclear_heat_total_mw_unnormalised,
    p_shld_nuclear_heat_mw_unnormalised,
    p_tf_nuclear_heat_mw_unnormalised,
    f_ster_div_single,
    f_p_blkt_multiplication,
    p_neutron_total_mw,
    f_geom_cp,
    pnuc_cp_tf,
):
    """The renormalisation of `hcpb.py:195-276` at `n_divertors == 1`, `itart == 1`.

    **This arm owns one field more and reads two more than its conventional sibling**,
    and both differences are the same fact: on a spherical tokamak the centrepost is
    real, so the two terms the conventional occupants drop as provably inert come back.

    - `f_geom_cp` (`:216`) is `CentrepostNeutronicsSphericalTokamakSuperconducting`'s
      minted `.ccfe_hcpb.f_geom_cp`, not the literal `0` of `:144`, so the blanket's
      solid-angle share is `1 - f_ster_div_single - f_geom_cp`.
    - `.fwbs.pnuc_cp_tf` (`:263`) is that node's real output, so the TF coils' share
      gains the heat deposited in the centrepost conductor.
    - `.fwbs.p_cp_shield_nuclear_heat_mw` (`:267-269`) becomes a live quantity --
      `f_geom_cp * p_neutron_total_mw - pnuc_cp_tf`, the centrepost's whole neutron
      budget less what the conductor took. It is **owned here**, and this is the write
      that makes the centrepost node's own third return value dead: PROCESS stores that
      value in the same field at `:137` and then overwrites it here, before the only
      readers (`powerflow_calc:834`/`:852`/`:908` and `power.py:940`, all downstream of
      `run():279`) ever see it. The port therefore mints the earlier value as
      `.ccfe_hcpb.p_cp_shield_nuclear_heat_mw_fit`, the same treatment the four
      `_unnormalised` powers get and for the same reason: two quantities, one PROCESS
      slot.

    On the conventional arm the second write is `0 * x - 0` and `CentrepostNeutronics-
    Absent` keeps ownership; here the two writes differ and the later one wins. That is
    why this occupant and its conventional sibling do not own the same set -- the shape
    `next_steps.md` §12.2 names, already live in this file at `i_p_coolant_pumping`.

    Parameters
    ----------
    p_fw_nuclear_heat_total_mw_unnormalised :
        First-wall nuclear heating before renormalisation (MW).
        `.ccfe_hcpb.p_fw_nuclear_heat_total_mw_unnormalised`, from `NuclearHeatingFw`.
    p_blkt_nuclear_heat_total_mw_unnormalised :
        Blanket, same. From `NuclearHeatingBlanket`.
    p_shld_nuclear_heat_mw_unnormalised :
        Shield, same. From `NuclearHeatingShieldSphericalTokamak`.
    p_tf_nuclear_heat_mw_unnormalised :
        TF coils, same. From `NuclearHeatingMagnetsSphericalTokamak`.
    f_ster_div_single :
        Divertor solid-angle fraction per divertor. `.fwbs.f_ster_div_single`.
    f_p_blkt_multiplication :
        Blanket neutron energy multiplication factor. `.fwbs.f_p_blkt_multiplication`.
    p_neutron_total_mw :
        Total neutron power (MW). `.physics.p_neutron_total_mw`.
    f_geom_cp :
        Centrepost solid-angle fraction (-). `.ccfe_hcpb.f_geom_cp`.
    pnuc_cp_tf :
        Nuclear heat in the centrepost TF conductor (MW). `.fwbs.pnuc_cp_tf`.

    Returns
    -------
    tuple
        `(pnuc_tot_blk_sector, p_fw_nuclear_heat_total_mw,
        p_blkt_nuclear_heat_total_mw, p_shld_nuclear_heat_mw, p_tf_nuclear_heat_mw,
        p_cp_shield_nuclear_heat_mw, p_blkt_multiplication_mw)`.
    """
    return _nuclear_heating_renormalisation_spherical_tokamak(
        p_fw_nuclear_heat_total_mw_unnormalised,
        p_blkt_nuclear_heat_total_mw_unnormalised,
        p_shld_nuclear_heat_mw_unnormalised,
        p_tf_nuclear_heat_mw_unnormalised,
        # `1 - n_divertors * f_ster_div_single - f_geom_cp` at `n_divertors == 1`.
        f_geom_blanket=1 - f_ster_div_single - f_geom_cp,
        f_p_blkt_multiplication=f_p_blkt_multiplication,
        p_neutron_total_mw=p_neutron_total_mw,
        f_geom_cp=f_geom_cp,
        pnuc_cp_tf=pnuc_cp_tf,
    )


def calculate_nuclear_heating_renormalisation_double_null_spherical_tokamak(
    p_fw_nuclear_heat_total_mw_unnormalised,
    p_blkt_nuclear_heat_total_mw_unnormalised,
    p_shld_nuclear_heat_mw_unnormalised,
    p_tf_nuclear_heat_mw_unnormalised,
    f_ster_div_single,
    f_p_blkt_multiplication,
    p_neutron_total_mw,
    f_geom_cp,
    pnuc_cp_tf,
):
    """The renormalisation of `hcpb.py:195-276` at `n_divertors == 2`, `itart == 1`.

    **The cell both spherical-tokamak input files actually select.**
    `spherical_tokamak_eval.IN.DAT` and `st_regression.IN.DAT` set `itart = 1` and
    `i_single_null = 0`, so `_n_divertors` derives `2` and this is the occupant the
    factory builds -- the fourth and last cell of the `(n_divertors, itart)` square, and
    the one that closes it.

    Identical to its single-null sibling except in `f_geom_blanket`: PROCESS spells it as
    `1 - n_divertors * f_ster_div_single - f_geom_cp` (`hcpb.py:213-217`), a
    multiplication rather than an `if`, so the two arms are the same shape with a
    different divertor count. See the single-null docstring for why `f_geom_cp`,
    `.fwbs.pnuc_cp_tf` and `.fwbs.p_cp_shield_nuclear_heat_mw` appear on the spherical
    arms and not on the conventional ones.

    Parameters
    ----------
    p_fw_nuclear_heat_total_mw_unnormalised :
        First-wall nuclear heating before renormalisation (MW).
    p_blkt_nuclear_heat_total_mw_unnormalised :
        Blanket, same.
    p_shld_nuclear_heat_mw_unnormalised :
        Shield, same.
    p_tf_nuclear_heat_mw_unnormalised :
        TF coils, same.
    f_ster_div_single :
        Divertor solid-angle fraction **per divertor**. `.fwbs.f_ster_div_single`.
    f_p_blkt_multiplication :
        Blanket neutron energy multiplication factor.
    p_neutron_total_mw :
        Total neutron power (MW).
    f_geom_cp :
        Centrepost solid-angle fraction (-). `.ccfe_hcpb.f_geom_cp`.
    pnuc_cp_tf :
        Nuclear heat in the centrepost TF conductor (MW). `.fwbs.pnuc_cp_tf`.

    Returns
    -------
    tuple
        The same seven values as the single-null spherical arm.
    """
    return _nuclear_heating_renormalisation_spherical_tokamak(
        p_fw_nuclear_heat_total_mw_unnormalised,
        p_blkt_nuclear_heat_total_mw_unnormalised,
        p_shld_nuclear_heat_mw_unnormalised,
        p_tf_nuclear_heat_mw_unnormalised,
        # `1 - n_divertors * f_ster_div_single - f_geom_cp` at `n_divertors == 2`.
        f_geom_blanket=1 - 2 * f_ster_div_single - f_geom_cp,
        f_p_blkt_multiplication=f_p_blkt_multiplication,
        p_neutron_total_mw=p_neutron_total_mw,
        f_geom_cp=f_geom_cp,
        pnuc_cp_tf=pnuc_cp_tf,
    )


def _nuclear_heating_renormalisation_spherical_tokamak(
    p_fw_nuclear_heat_total_mw_unnormalised,
    p_blkt_nuclear_heat_total_mw_unnormalised,
    p_shld_nuclear_heat_mw_unnormalised,
    p_tf_nuclear_heat_mw_unnormalised,
    *,
    f_geom_blanket,
    f_p_blkt_multiplication,
    p_neutron_total_mw,
    f_geom_cp,
    pnuc_cp_tf,
):
    """The part of `hcpb.py:195-276` both spherical arms share.

    A private helper rather than two transcriptions, on the same argument
    `_nuclear_heating_shield` makes: the arms differ in exactly one expression, and this
    block has enough coefficients in it that a second copy is a second place for them to
    drift. The two *conventional* arms predate this and are written out in full; they are
    left as they are rather than refactored in a wave about `itart`.

    Parameters
    ----------
    p_fw_nuclear_heat_total_mw_unnormalised :
        First-wall nuclear heating before renormalisation (MW).
    p_blkt_nuclear_heat_total_mw_unnormalised :
        Blanket, same.
    p_shld_nuclear_heat_mw_unnormalised :
        Shield, same.
    p_tf_nuclear_heat_mw_unnormalised :
        TF coils, same.
    f_geom_blanket :
        Solid-angle fraction taken by the breeding blankets and shields (-), formed by
        the calling arm from its own divertor count.
    f_p_blkt_multiplication :
        Blanket neutron energy multiplication factor (-).
    p_neutron_total_mw :
        Total neutron power (MW).
    f_geom_cp :
        Centrepost solid-angle fraction (-).
    pnuc_cp_tf :
        Nuclear heat in the centrepost TF conductor (MW).

    Returns
    -------
    tuple
        The seven values both spherical arms return.
    """
    # Total nuclear power deposited in the blanket sector (MW)
    pnuc_tot_blk_sector = (
        p_fw_nuclear_heat_total_mw_unnormalised
        + p_blkt_nuclear_heat_total_mw_unnormalised
        + p_shld_nuclear_heat_mw_unnormalised
        + p_tf_nuclear_heat_mw_unnormalised
    )

    normalisation = f_p_blkt_multiplication * f_geom_blanket * p_neutron_total_mw

    p_fw_nuclear_heat_total_mw = (
        p_fw_nuclear_heat_total_mw_unnormalised / pnuc_tot_blk_sector
    ) * normalisation
    p_blkt_nuclear_heat_total_mw = (
        p_blkt_nuclear_heat_total_mw_unnormalised / pnuc_tot_blk_sector
    ) * normalisation
    # The power deposited in the CP shield is added back in `powerflow_calc`.
    p_shld_nuclear_heat_mw = (
        p_shld_nuclear_heat_mw_unnormalised / pnuc_tot_blk_sector
    ) * normalisation
    # The power deposited in the CP conductor is added back here (`hcpb.py:263`).
    p_tf_nuclear_heat_mw = (
        p_tf_nuclear_heat_mw_unnormalised / pnuc_tot_blk_sector
    ) * normalisation + pnuc_cp_tf

    # Power deposited in the CP shield (`hcpb.py:267-269`), overwriting the MCNP fit
    # value `st_centrepost_nuclear_heating` wrote at `:137`.
    p_cp_shield_nuclear_heat_mw = f_geom_cp * p_neutron_total_mw - pnuc_cp_tf

    p_blkt_multiplication_mw = (
        (f_p_blkt_multiplication - 1) * f_geom_blanket * p_neutron_total_mw
    )

    return (
        pnuc_tot_blk_sector,
        p_fw_nuclear_heat_total_mw,
        p_blkt_nuclear_heat_total_mw,
        p_shld_nuclear_heat_mw,
        p_tf_nuclear_heat_mw,
        p_cp_shield_nuclear_heat_mw,
        p_blkt_multiplication_mw,
    )


def calculate_first_wall_radiation_powers(
    p_plasma_rad_mw,
    f_a_fw_outboard_hcd,
    p_div_rad_total_mw,
    a_fw_outboard,
    a_fw_total,
    p_beam_orbit_loss_mw,
    p_fw_alpha_mw,
):
    """Radiation and surface heat flux incident on the first wall.

    Ports `powerflow_calc`'s unconditional prologue (`hcpb.py:780-814`) -- everything
    before the `i_p_coolant_pumping` dispatch. It is **not** behind that switch and is
    therefore an ordinary unswitched node, which matters: the pumping occupants
    downstream read `psurffwi`/`psurffwo`, and folding this into each of them would
    duplicate an owner four ways.

    The `i_blkt_coolant_type == WATER` branch between them (`:793-801`, the only CoolProp
    call inside `CCFE_HCPB`) is **not ported and is dead code on this model's own path**:
    `run()` assigns `self.data.fwbs.i_blkt_coolant_type = CoolantType.HELIUM` at `:45`
    and nothing between there and `:793` changes it, so `powerflow_calc` reached through
    `CCFE_HCPB.run()` can never take it. That is stronger than
    `tokamak_call_surface.md` §D's "dormant behind this run's switch values" -- for this
    one site no input file revives it.

    Parameters
    ----------
    p_plasma_rad_mw :
        Total plasma radiation power (MW). `.physics.p_plasma_rad_mw`.
    f_a_fw_outboard_hcd :
        Fraction of the outboard first-wall area taken by HCD apparatus.
        `.fwbs.f_a_fw_outboard_hcd`.
    p_div_rad_total_mw :
        Radiation power incident on the divertor (MW). `.fwbs.p_div_rad_total_mw`
        (`divertor.py:52`).
    a_fw_outboard, a_fw_total :
        Outboard and total first-wall area (m2). `.first_wall.a_fw_outboard`,
        `.first_wall.a_fw_total` (`fw.py:89-91`).
    p_beam_orbit_loss_mw :
        Neutral-beam orbit loss power (MW). `.current_drive.p_beam_orbit_loss_mw`.
    p_fw_alpha_mw :
        Alpha power incident on the first wall (MW). `.physics.p_fw_alpha_mw`.

    Returns
    -------
    tuple
        `(p_fw_hcd_rad_total_mw, p_fw_rad_total_mw, psurffwo, psurffwi)` -- PROCESS's own
        write order.
    """
    p_fw_hcd_rad_total_mw = p_plasma_rad_mw * f_a_fw_outboard_hcd

    p_fw_rad_total_mw = p_plasma_rad_mw - p_div_rad_total_mw - p_fw_hcd_rad_total_mw

    # All of the fast particle losses go to the outer wall.
    psurffwo = (
        p_fw_rad_total_mw * a_fw_outboard / a_fw_total
        + p_beam_orbit_loss_mw
        + p_fw_alpha_mw
    )
    psurffwi = p_fw_rad_total_mw * (1 - a_fw_outboard / a_fw_total)

    return p_fw_hcd_rad_total_mw, p_fw_rad_total_mw, psurffwo, psurffwi


def calculate_pumping_power_mechanical_with_pressure_drop(
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
):
    """Coolant pumping powers at `i_p_coolant_pumping == 3`.

    Ports `powerflow_calc`'s `MECHANICAL_WITH_PRESSURE_DROP` arm (`hcpb.py:864-918`,
    PROCESS issue #503): the first-wall/blanket loop's mechanical pumping power from a
    specified pressure drop across the loop including heat exchanger and pipes, plus the
    shield and divertor powers as fractions of the thermal power their coolant removes.

    **This arm does not produce `.heat_transport.p_fw_coolant_pump_mw` or
    `.heat_transport.p_blkt_coolant_pump_mw`** -- it produces
    `.primary_pumping.p_fw_blkt_coolant_pump_mw`, one combined loop, and PROCESS's own
    consumer knows it: `power.component_thermal_powers` forms
    `p_fw_blkt_coolant_pump_mw` from the two `heat_transport` fields only when
    `i_p_coolant_pumping not in {MECHANICAL, MECHANICAL_WITH_PRESSURE_DROP}`
    (`power.py:821-829`). Measured on the reference run after four
    `_call_models_once` passes, both `heat_transport` fields are still exactly `0.0`,
    their `heat_transport_variables.py:73`/`:85` defaults. `tokamak_boundary.md` lists
    them under this slot because its attribution is a mechanical `ast` walk over
    `Assign` targets, which sweeps in arms 1 and 2; on this run they have no producer at
    all. See the audit record's "two boundary variables with no producer".

    `pfactor`'s power law goes through `safe_pow`. The exponent is
    `(gamma_he - 1) / gamma_he ~= 0.4`, squarely in the `0 < p < 1` band where `x ** p`
    has an infinite derivative at `x == 0` (`models/safe_math.py`); the base is far from
    zero on any realistic input, but the guard is free and the defect class it closes is
    invisible to every value test.

    Parameters
    ----------
    p_he, dp_he, gamma_he :
        Helium pressure at the blanket inlet / pump outlet (Pa), pressure drop across the
        FW+blanket circuit (Pa), and ratio of specific heats. `.primary_pumping.p_he`,
        `.primary_pumping.dp_he`, `.primary_pumping.gamma_he`.
    t_in_bb, t_out_bb :
        Blanket coolant inlet/outlet temperature (K). `.primary_pumping.t_in_bb`,
        `.primary_pumping.t_out_bb`.
    etaiso :
        Isentropic efficiency of the coolant pumps. `.fwbs.etaiso`.
    f_p_fw_blkt_pump :
        Multiplier on the FW+blanket pumping power.
        `.primary_pumping.f_p_fw_blkt_pump`.
    p_fw_nuclear_heat_total_mw, p_blkt_nuclear_heat_total_mw :
        Renormalised first-wall and blanket nuclear heating (MW), from
        `NuclearHeatingRenormalisationSingleNullConventional`.
    psurffwi, psurffwo :
        Inboard/outboard first-wall surface heat flux (MW), from
        `FirstWallRadiationPowers`.
    f_p_shld_coolant_pump_total_heat, f_p_div_coolant_pump_total_heat :
        Pumping power as a fraction of the heat removed, shield and divertor.
        `.heat_transport.*`.
    p_shld_nuclear_heat_mw :
        Renormalised shield nuclear heating (MW), same producer as the first two.
    p_cp_shield_nuclear_heat_mw :
        Centrepost shield nuclear heating (MW). `.fwbs.p_cp_shield_nuclear_heat_mw`, from
        `CentrepostNeutronicsAbsent` (`0.0` on this arm).
    p_plasma_separatrix_mw :
        Power crossing the separatrix (MW). `.physics.p_plasma_separatrix_mw`.
    p_div_nuclear_heat_total_mw, p_div_rad_total_mw :
        Divertor nuclear and radiative heating (MW). `.fwbs.*` (`divertor.py:46`, `:52`).

    Returns
    -------
    tuple
        `(p_fw_blkt_coolant_pump_mw, p_shld_coolant_pump_mw, p_div_coolant_pump_mw)`.
    """
    pfactor = safe_pow(p_he / (p_he - dp_he), (gamma_he - 1) / gamma_he)

    t_in_compressor = t_in_bb / pfactor
    dt_he = t_out_bb - t_in_bb
    fpump = t_in_compressor / (etaiso * dt_he) * (pfactor - 1)

    p_plasma = (
        p_fw_nuclear_heat_total_mw + psurffwi + psurffwo + p_blkt_nuclear_heat_total_mw
    )

    p_fw_blkt_coolant_pump_mw = f_p_fw_blkt_pump * fpump / (1 - fpump) * p_plasma

    p_shld_coolant_pump_mw = f_p_shld_coolant_pump_total_heat * (
        p_shld_nuclear_heat_mw + p_cp_shield_nuclear_heat_mw
    )
    p_div_coolant_pump_mw = f_p_div_coolant_pump_total_heat * (
        p_plasma_separatrix_mw + p_div_nuclear_heat_total_mw + p_div_rad_total_mw
    )

    return p_fw_blkt_coolant_pump_mw, p_shld_coolant_pump_mw, p_div_coolant_pump_mw


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
