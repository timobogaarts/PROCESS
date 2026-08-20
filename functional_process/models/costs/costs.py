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
    FromExactly,
    Output,
)

from functional_process.models.safe_math import safe_pow, safe_sqrt
from functional_process.models.switch_enums import (
    CentralSolenoidConfiguration,
    CostOfElectricityModel,
    IFEModel,
    NetElectricPowerModel,
    PlantOperationModel,
    SphericalTokamakModel,
    SuperconductorCostModel,
    ThermalStorageModel,
)
from process.data_structure.pfcoil_variables import PFConductorModel


def convert_fpy_to_calendar(
    life_blkt_fpy,
    life_plant,
    f_t_plant_available,
    life_div_fpy,
    itart,
    cplife,
):
    """Convert component lifetimes from full-power-years to calendar years. Ports
    `Costs.convert_fpy_to_calendar` verbatim (already a straight-line function, no
    branching beyond the three independent 2-way thresholds below).

    Three independent `life_x_fpy < life_plant` thresholds (FW/blanket+HCD, divertor,
    centrepost), each ported as a `jnp.where` -- see `costs.md`'s JAX-difficulty flags.
    **`cdrlife_cal` and `cplife_cal` are only written by PROCESS on the "fast" branch of
    their own threshold** (source leaves them at whatever `.costs.cdrlife_cal`/
    `cplife_cal` already held otherwise, i.e. history from a *previous* solver
    iteration on the same persistent `DataStructure` -- not a fixed default). The port
    cannot reproduce that history dependence (a pure function has no notion of "the
    value before this call"), so it returns `0.0` on the untaken branch instead --
    flagged as a real behavioural difference, not silently normalised away, see
    `costs.md`'s open questions.

    Parameters
    ----------
    life_blkt_fpy :
        FW/blanket lifetime (full-power-years). `.fwbs.life_blkt_fpy`.
    life_plant :
        Plant lifetime (calendar years). `.costs.life_plant`.
    f_t_plant_available :
        Plant availability fraction, FPY -> calendar-year conversion factor.
        `.costs.f_t_plant_available`.
    life_div_fpy :
        Divertor lifetime (full-power-years). `.costs.life_div_fpy`.
    itart :
        Spherical-tokamak indicator (1 if ST, else 0). `.physics.itart`.
    cplife :
        Centrepost lifetime (full-power-years), ST only. `.costs.cplife`.

    Returns
    -------
    tuple
        `(life_blkt, cdrlife_cal, life_div, cplife_cal)`.
    """
    blkt_is_fast = life_blkt_fpy < life_plant
    life_blkt = jnp.where(
        blkt_is_fast, life_blkt_fpy * f_t_plant_available, life_blkt_fpy
    )
    cdrlife_cal = jnp.where(blkt_is_fast, life_blkt, 0.0)

    div_is_fast = life_div_fpy < life_plant
    life_div = jnp.where(div_is_fast, life_div_fpy * f_t_plant_available, life_div_fpy)

    cp_is_fast = cplife < life_plant
    cplife_cal = jnp.where(
        itart == 1, jnp.where(cp_is_fast, cplife * f_t_plant_available, cplife), 0.0
    )

    return life_blkt, cdrlife_cal, life_div, cplife_cal


def calculate_structures_cost(
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
):
    """Account 21: structures and site facilities. Ports `Costs.acc21` verbatim -- one
    straight-line function in the source, no sub-methods to compose.

    `lsa` (level of safety assurance, 1-4) indexes four `cmlsa` multiplier tables that
    differ *per account* (hardcoded literals inline in the source, not a shared
    constant) -- kept as literal arrays here too, indexed the same way
    (`cmlsa[lsa - 1]`). Not a switch in the topology sense: it selects a scaling
    coefficient, never which formula runs -- see `costs.md`'s switches touched.

    `c213` (turbine building) is only computed by PROCESS when `ireactor == 1`; the port
    returns `0.0` on the other branch, matching the dataclass field's own default (unlike
    `convert_fpy_to_calendar`'s `cdrlife_cal`, `ireactor` is a run-configuration constant,
    not an iteration variable, so this default cannot be stale mid-run -- see `costs.md`).

    Parameters
    ----------
    csi, cland :
        Site improvements cost, land cost (M$). `.costs.csi`, `.costs.cland`.
    lsa :
        Level of safety assurance, 1-4. `.costs.lsa`.
    ucrb, rbvol :
        Reactor building unit cost ($/m^3), volume (m^3). `.costs.ucrb`,
        `.buildings.rbvol`.
    UCMB, rmbvol :
        Reactor maintenance building unit cost, volume. `.costs.UCMB`,
        `.buildings.rmbvol`.
    UCWS, wsvol :
        Warm shop unit cost, volume. `.costs.UCWS`, `.buildings.wsvol`.
    UCTR, triv :
        Tritium building unit cost, volume. `.costs.UCTR`, `.buildings.triv`.
    UCEL, elevol :
        Electrical equipment building unit cost, volume. `.costs.UCEL`,
        `.buildings.elevol`.
    UCAD, admvol :
        Admin building unit cost, volume. `.costs.UCAD`, `.buildings.admvol`.
    UCCO, convol :
        Control room building unit cost, volume. `.costs.UCCO`, `.buildings.convol`.
    UCSH, shovol :
        Shop/warehouse unit cost, volume. `.costs.UCSH`, `.buildings.shovol`.
    UCCR, cryvol :
        Cryogenic building unit cost, volume. `.costs.UCCR`, `.buildings.cryvol`.
    ireactor :
        1 if this run represents a reactor (net electricity), else 0. `.costs.ireactor`.
    cturbb :
        Turbine building base cost (M$). `.costs.cturbb`.

    Returns
    -------
    tuple
        `(c211, c212, c213, c2141, c2142, c214, c215, c216, c2171, c2172, c2173, c2174,
        c217, c21)`.
    """
    cmlsa = jnp.asarray([0.6800e0, 0.8400e0, 0.9200e0, 1.0000e0])[lsa - 1]
    exprb = 1.0e0

    c211 = csi * cmlsa + cland
    c212 = 1.0e-6 * ucrb * rbvol**exprb * cmlsa
    c213 = jnp.where(ireactor == 1, cturbb * cmlsa, 0.0)
    c2141 = 1.0e-6 * UCMB * rmbvol**exprb * cmlsa
    c2142 = 1.0e-6 * UCWS * wsvol**exprb * cmlsa
    c214 = c2141 + c2142
    c215 = 1.0e-6 * UCTR * triv**exprb * cmlsa
    c216 = 1.0e-6 * UCEL * elevol**exprb * cmlsa
    c2171 = 1.0e-6 * UCAD * admvol**exprb * cmlsa
    c2172 = 1.0e-6 * UCCO * convol**exprb * cmlsa
    c2173 = 1.0e-6 * UCSH * shovol**exprb * cmlsa
    c2174 = 1.0e-6 * UCCR * cryvol**exprb * cmlsa
    c217 = c2171 + c2172 + c2173 + c2174
    c21 = c211 + c212 + c213 + c214 + c215 + c216 + c217

    return (
        c211,
        c212,
        c213,
        c2141,
        c2142,
        c214,
        c215,
        c216,
        c2171,
        c2172,
        c2173,
        c2174,
        c217,
        c21,
    )


def calculate_indirect_costs(cfind, lsa, cdirt, cowner, fcontng):
    """Account 9: indirect cost and project contingency. Ports `Costs.acc9` verbatim.

    Parameters
    ----------
    cfind :
        Indirect cost factor by level of safety assurance, length-4. `.costs.cfind`.
    lsa :
        Level of safety assurance, 1-4. `.costs.lsa`.
    cdirt :
        Total plant direct cost (M$). `.costs.cdirt`.
    cowner :
        Owner cost factor. `.costs.cowner`.
    fcontng :
        Project contingency factor. `.costs.fcontng`.

    Returns
    -------
    tuple
        `(cindrt, ccont)`.
    """
    cindrt = jnp.asarray(cfind)[lsa - 1] * cdirt * (1.0e0 + cowner)
    ccont = fcontng * (cdirt + cindrt)
    return cindrt, ccont


def calculate_reactor_structure_cost(gsmass, UCGSS, lsa, fkind):
    """Account 221.4: reactor structure. Ports `Costs.acc2214` verbatim.

    Parameters
    ----------
    gsmass :
        Gravity support structure mass (kg). `.structure.gsmass`.
    UCGSS :
        Unit cost of structural steel ($/kg). `.costs.UCGSS`.
    lsa :
        Level of safety assurance, 1-4. `.costs.lsa`.
    fkind :
        Nth-of-a-kind multiplier. `.costs.fkind`.

    Returns
    -------
    float
        `c2214`.
    """
    cmlsa = jnp.asarray([0.6700e0, 0.8350e0, 0.9175e0, 1.0000e0])[lsa - 1]
    return fkind * (1.0e-6 * gsmass * UCGSS * cmlsa)


def calculate_vacuum_vessel_assembly_cost(m_vv, uccryo, lsa, fkind):
    """Account 222.3: vacuum vessel assembly (part of the magnets account, distinct
    from Account 224's "vacuum system"). Ports `Costs.acc2223` verbatim.

    Parameters
    ----------
    m_vv :
        Vacuum vessel mass (kg). `.fwbs.m_vv`.
    uccryo :
        Unit cost of vacuum vessel material ($/kg). `.costs.uccryo`.
    lsa :
        Level of safety assurance, 1-4. `.costs.lsa`.
    fkind :
        Nth-of-a-kind multiplier. `.costs.fkind`.

    Returns
    -------
    float
        `c2223`.
    """
    cmlsa = jnp.asarray([0.6900e0, 0.8450e0, 0.9225e0, 1.0000e0])[lsa - 1]
    return fkind * (1.0e-6 * m_vv * uccryo) * cmlsa


def calculate_divertor_cost(ife, a_div_surface_total, ucdiv, fkind, ifueltyp):
    """Account 221.5: divertor. Ports `Costs.acc2215` verbatim.

    `ife` (`.ife.ife == 1`) zeroes both outputs -- IFE devices have no divertor plates in
    this model. Kept as a plain traced `jnp.where` per the `itart` precedent
    (`hcpb.md`'s switches-touched section): a device-mode flag this file's own PROCESS
    code does not itself split into separate functions.

    Parameters
    ----------
    ife :
        1 if this is an IFE device, else 0. `.ife.ife`.
    a_div_surface_total :
        Divertor surface area (m^2). `.divertor.a_div_surface_total`.
    ucdiv :
        Unit cost of divertor ($/m^2). `.costs.ucdiv`.
    fkind :
        Nth-of-a-kind multiplier. `.costs.fkind`.
    ifueltyp :
        1: divertor cost is a fuel cost; 2: capital cost, replacements are fuel cost;
        else: pure capital cost. `.costs.ifueltyp`.

    Returns
    -------
    tuple
        `(c2215, divcst)`.
    """
    is_ife = ife == 1
    c2215_raw = fkind * (1.0e-6 * a_div_surface_total * ucdiv)

    divcst = jnp.where(
        is_ife,
        0.0,
        jnp.where(ifueltyp == 1, c2215_raw, jnp.where(ifueltyp == 2, c2215_raw, 0.0)),
    )
    c2215 = jnp.where(is_ife, 0.0, jnp.where(ifueltyp == 1, 0.0, c2215_raw))
    return c2215, divcst


def calculate_vacuum_system_cost(
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
):
    """Account 224: vacuum system. Ports `Costs.acc224` verbatim.

    `i_vacuum_pump_type` selects between two unit-cost formulas for account 224.1
    (`VacuumPumpType.COMPOUND_CRYOPUMP == 1` vs. `TURBOMOLECULAR == 0`,
    `process/data_structure/vacuum_variables.py`) -- compared against the plain int
    value rather than importing the enum, so this module has no dependency on
    `process.*` at import time (see `costs.md`'s JAX-difficulty flags for why: the
    enum's own values are the contract, not its class identity).

    Parameters
    ----------
    i_vacuum_pump_type :
        0 (turbomolecular) or 1 (compound cryopump). `.vacuum.i_vacuum_pump_type`.
    n_vac_pumps_high :
        Number of high-vacuum pumps. `.vacuum.n_vac_pumps_high`.
    UCCPMP, UCTPMP :
        Unit cost of cryopump / turbomolecular pump ($). `.costs.UCCPMP`, `.costs.UCTPMP`.
    n_vv_vacuum_ducts :
        Number of vacuum ducts. `.vacuum.n_vv_vacuum_ducts`.
    UCBPMP :
        Unit cost of backing pump ($). `.costs.UCBPMP`.
    dlscal :
        Duct length scaling factor (m). `.vacuum.dlscal`.
    UCDUCT :
        Unit cost of duct ($/m). `.costs.UCDUCT`.
    dia_vv_vacuum_ducts :
        Duct diameter (m). `.vacuum.dia_vv_vacuum_ducts`.
    UCVALV :
        Unit cost of valve ($). `.costs.UCVALV`.
    m_vv_vacuum_duct_shield :
        Duct shielding mass (kg). `.vacuum.m_vv_vacuum_duct_shield`.
    UCVDSH :
        Unit cost of duct shielding ($/kg). `.costs.UCVDSH`.
    UCVIAC :
        Vacuum instrumentation cost ($). `.costs.UCVIAC`.
    fkind :
        Nth-of-a-kind multiplier. `.costs.fkind`.

    Returns
    -------
    tuple
        `(c2241, c2242, c2243, c2244, c2245, c2246, c224)`.
    """
    c2241 = fkind * (
        1.0e-6 * n_vac_pumps_high * jnp.where(i_vacuum_pump_type == 1, UCCPMP, UCTPMP)
    )
    c2242 = fkind * (1.0e-6 * n_vv_vacuum_ducts * UCBPMP)
    c2243 = fkind * (1.0e-6 * n_vv_vacuum_ducts * dlscal * UCDUCT)
    c2244 = fkind * (
        1.0e-6
        * 2.0e0
        * n_vv_vacuum_ducts
        * (dia_vv_vacuum_ducts * 1.2e0) ** 1.4e0
        * UCVALV
    )
    c2245 = fkind * (1.0e-6 * n_vv_vacuum_ducts * m_vv_vacuum_duct_shield * UCVDSH)
    c2246 = fkind * (1.0e-6 * UCVIAC)
    c224 = c2241 + c2242 + c2243 + c2244 + c2245 + c2246
    return c2241, c2242, c2243, c2244, c2245, c2246, c224


def calculate_tf_coil_power_conditioning_cost(
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
):
    """Account 225.1: TF coil power conditioning. Ports `Costs.acc2251` verbatim.

    `i_tf_sup` (`process.models.tfcoil.base.TFConductorModel`, values 0/1/2) gates two
    2-way formula choices (breakers, bussing) -- compared against the plain int `1`
    (`SUPERCONDUCTING`) rather than importing the enum, same reasoning as
    `calculate_vacuum_system_cost`'s `i_vacuum_pump_type`.

    Parameters
    ----------
    uctfps :
        TF power supply unit cost. `.costs.uctfps`.
    tfckw, tfcmw :
        TF coil power supply rating (kW, MW). `.tfcoil.tfckw`, `.tfcoil.tfcmw`.
    i_tf_sup :
        0 (water-cooled copper), 1 (superconducting), 2 (helium-cooled aluminium).
        `.tfcoil.i_tf_sup`.
    uctfbr :
        Unit cost of TF coil breakers. `.costs.uctfbr`.
    n_tf_coils :
        Number of TF coils. `.tfcoil.n_tf_coils`.
    c_tf_turn :
        TF coil turn current (A). `.tfcoil.c_tf_turn`.
    v_tf_coil_dump_quench_kv :
        TF coil quench dump voltage (kV). `.tfcoil.v_tf_coil_dump_quench_kv`.
    uctfsw :
        Unit cost of TF coil switches. `.costs.uctfsw`.
    UCTFDR :
        Unit cost of TF coil dump resistors ($/GJ). `.costs.UCTFDR`.
    e_tf_magnetic_stored_total_gj :
        Total TF coil stored magnetic energy (GJ). `.tfcoil.e_tf_magnetic_stored_total_gj`.
    UCTFGR :
        Ground protection unit cost per coil. `.costs.UCTFGR`.
    UCTFIC :
        Unit cost of TF coil instrumentation, per 30 coils. `.costs.UCTFIC`.
    uctfbus :
        Unit cost of TF coil bussing (resistive). `.costs.uctfbus`.
    m_tf_bus :
        TF bus mass (kg), resistive case. `.tfcoil.m_tf_bus`.
    ucbus :
        Unit cost of TF coil bussing conductor (superconducting case). `.costs.ucbus`.
    len_tf_bus :
        TF bus length (m), superconducting case. `.tfcoil.len_tf_bus`.
    fkind :
        Nth-of-a-kind multiplier. `.costs.fkind`.

    Returns
    -------
    tuple
        `(c22511, c22512, c22513, c22514, c22515, c2251)`.
    """
    expel = 0.7e0
    is_sc = i_tf_sup == 1

    c22511 = fkind * (1.0e-6 * uctfps * (tfckw * 1.0e3 + tfcmw * 1.0e6) ** expel)
    c22512 = fkind * jnp.where(
        is_sc,
        1.0e-6
        * (
            uctfbr * n_tf_coils * (c_tf_turn * v_tf_coil_dump_quench_kv * 1.0e3) ** expel
            + uctfsw * c_tf_turn
        ),
        0.0,
    )
    c22513 = fkind * (
        1.0e-6
        * (1.0e9 * UCTFDR * e_tf_magnetic_stored_total_gj + UCTFGR * 0.5e0 * n_tf_coils)
    )
    c22514 = fkind * (1.0e-6 * UCTFIC * (30.0e0 * n_tf_coils))
    c22515 = fkind * jnp.where(
        is_sc,
        1.0e-6 * ucbus * c_tf_turn * len_tf_bus,
        1.0e-6 * uctfbus * m_tf_bus,
    )
    c2251 = c22511 + c22512 + c22513 + c22514 + c22515
    return c22511, c22512, c22513, c22514, c22515, c2251


def calculate_pf_coil_power_conditioning_cost(
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
):
    """Account 225.2: PF coil power conditioning. Ports `Costs.acc2252` verbatim.

    `pfckts != 0.0` guards account 225.2.4's `(srcktpm / pfckts) ** 0.7` against a
    literal `0/0` -- ported as `jnp.where` with a guarded denominator (see
    `costs.md`'s JAX-difficulty flags), not dropped, since PROCESS's own branch
    genuinely changes the formula, not just guards a domain edge.

    Parameters
    ----------
    ucpfps :
        PF power supply unit cost. `.costs.ucpfps`.
    peakmva :
        Peak MVA rating. `.heat_transport.peakmva`.
    ucpfic :
        PF coil instrumentation unit cost, per 30 circuits. `.costs.ucpfic`.
    pfckts :
        Number of PF coil circuits. `.pf_power.pfckts`.
    ucpfb :
        PF coil bussing unit cost. `.costs.ucpfb`.
    spfbusl :
        PF coil bus length (m). `.pf_power.spfbusl`.
    acptmax :
        Max PF coil circuit current (A). `.pf_power.acptmax`.
    ucpfbs :
        PF coil burn power supply unit cost. `.costs.ucpfbs`.
    srcktpm :
        PF coil circuit power (kVA). `.pf_power.srcktpm`.
    ucpfbk :
        PF coil breaker unit cost. `.costs.ucpfbk`.
    vpfskv :
        PF coil breaker voltage (kV). `.pf_power.vpfskv`.
    ucpfdr1 :
        PF coil dump resistor unit cost. `.costs.ucpfdr1`.
    ensxpfm :
        Maximum PF coil stored energy (MJ). `.pf_power.ensxpfm`.
    ucpfcb :
        PF coil AC breaker unit cost. `.costs.ucpfcb`.
    fkind :
        Nth-of-a-kind multiplier. `.costs.fkind`.

    Returns
    -------
    tuple
        `(c22521, c22522, c22523, c22524, c22525, c22526, c22527, c2252)`.
    """
    c22521 = fkind * (1.0e-6 * ucpfps * peakmva)
    c22522 = fkind * (1.0e-6 * ucpfic * pfckts * 30.0e0)
    c22523 = fkind * (1.0e-6 * ucpfb * spfbusl * acptmax)

    safe_pfckts = jnp.where(pfckts == 0.0, 1.0, pfckts)
    c22524 = fkind * jnp.where(
        pfckts != 0.0,
        1.0e-6 * ucpfbs * pfckts * safe_pow(srcktpm / safe_pfckts, 0.7e0),
        0.0,
    )

    c22525 = fkind * (1.0e-6 * ucpfbk * pfckts * safe_pow(acptmax * vpfskv, 0.7e0))
    c22526 = fkind * (1.0e-6 * ucpfdr1 * ensxpfm)
    c22527 = fkind * (1.0e-6 * ucpfcb * pfckts)
    c2252 = c22521 + c22522 + c22523 + c22524 + c22525 + c22526 + c22527
    return c22521, c22522, c22523, c22524, c22525, c22526, c22527, c2252


def calculate_reactor_cooling_system_cost(
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
):
    """Account 2261: reactor cooling system. Ports `Costs.acc2261` verbatim.

    `uchts` indexed by `i_blkt_coolant_type - 1` (a 2-element table, water/helium) --
    the source's own comment flags this as "a slight inconsistency" under
    `blktmodel > 0` (the shield is water-cooled even when the blanket is helium-cooled,
    but only one coolant-type index is used for both) -- a pre-existing PROCESS
    modelling note, not a porting concern, reproduced verbatim.

    Parameters
    ----------
    uchts :
        Heat transport system unit cost by coolant type, length-2.
        `.costs.uchts`.
    i_blkt_coolant_type :
        1 (water) or 2 (helium). `.fwbs.i_blkt_coolant_type`.
    p_fw_div_heat_deposited_mw :
        FW/divertor heat deposited (MW). `.heat_transport.p_fw_div_heat_deposited_mw`.
    p_blkt_nuclear_heat_total_mw :
        Blanket nuclear heating (MW). `.fwbs.p_blkt_nuclear_heat_total_mw`.
    p_shld_nuclear_heat_mw :
        Shield nuclear heating (MW). `.fwbs.p_shld_nuclear_heat_mw`.
    lsa :
        Level of safety assurance, 1-4. `.costs.lsa`.
    fkind :
        Nth-of-a-kind multiplier. `.costs.fkind`.
    UCPHX :
        Primary heat exchanger unit cost. `.costs.UCPHX`.
    n_primary_heat_exchangers :
        Number of primary heat exchangers. `.heat_transport.n_primary_heat_exchangers`.
    p_plant_primary_heat_mw :
        Total primary heat (MW). `.heat_transport.p_plant_primary_heat_mw`.

    Returns
    -------
    tuple
        `(cpp, chx, c2261)`.
    """
    cmlsa = jnp.asarray([0.4000e0, 0.7000e0, 0.8500e0, 1.0000e0])[lsa - 1]
    exphts = 0.7e0

    cpp = fkind * (
        1.0e-6
        * jnp.asarray(uchts)[i_blkt_coolant_type - 1]
        * (
            safe_pow(1.0e6 * p_fw_div_heat_deposited_mw, exphts)
            + safe_pow(1.0e6 * p_blkt_nuclear_heat_total_mw, exphts)
            + safe_pow(1.0e6 * p_shld_nuclear_heat_mw, exphts)
        )
        * cmlsa
    )
    chx = fkind * (
        1.0e-6
        * UCPHX
        * n_primary_heat_exchangers
        * safe_pow(1.0e6 * p_plant_primary_heat_mw / n_primary_heat_exchangers, exphts)
        * cmlsa
    )
    c2261 = chx + cpp
    return cpp, chx, c2261


def calculate_fuelling_system_cost(ucf1, fkind):
    """Account 2271: fuelling system. Ports `Costs.acc2271` verbatim.

    Parameters
    ----------
    ucf1 :
        Unit cost of fuelling system. `.costs.ucf1`.
    fkind :
        Nth-of-a-kind multiplier. `.costs.fkind`.

    Returns
    -------
    float
        `c2271`.
    """
    return fkind * (1.0e-6 * ucf1)


def calculate_nuclear_building_ventilation_cost(UCNBV, volrci, wsvol, fkind):
    """Account 2274: nuclear building ventilation. Ports `Costs.acc2274` verbatim.

    Parameters
    ----------
    UCNBV :
        Unit cost of nuclear building ventilation. `.costs.UCNBV`.
    volrci :
        Reactor building volume, non-shielded-plus-shielded interior (m^3).
        `.buildings.volrci`.
    wsvol :
        Warm shop volume (m^3). `.buildings.wsvol`.
    fkind :
        Nth-of-a-kind multiplier. `.costs.fkind`.

    Returns
    -------
    float
        `c2274`.
    """
    return fkind * (1.0e-6 * UCNBV * safe_pow(volrci + wsvol, 0.8e0))


def calculate_instrumentation_and_control_cost(uciac, fkind):
    """Account 228: instrumentation and control. Ports `Costs.acc228` verbatim.

    Parameters
    ----------
    uciac :
        Unit cost of instrumentation and control. `.costs.uciac`.
    fkind :
        Nth-of-a-kind multiplier. `.costs.fkind`.

    Returns
    -------
    float
        `c228`.
    """
    return fkind * (1.0e-6 * uciac)


def calculate_maintenance_equipment_cost(ucme, fkind):
    """Account 229: maintenance equipment. Ports `Costs.acc229` verbatim.

    Parameters
    ----------
    ucme :
        Unit cost of maintenance equipment. `.costs.ucme`.
    fkind :
        Nth-of-a-kind multiplier. `.costs.fkind`.

    Returns
    -------
    float
        `c229`.
    """
    return fkind * (1.0e-6 * ucme)


def calculate_turbine_plant_equipment_cost(
    ireactor, ucturb, i_blkt_coolant_type, p_plant_electric_gross_mw
):
    """Account 23: turbine plant equipment. Ports `Costs.acc23` verbatim.

    **PROCESS itself only computes `c23` when `ireactor == 1`**; on the other branch the
    source leaves `.costs.c23` untouched, i.e. at the dataclass default `0.0` (`ireactor`
    is a run-configuration constant per `calculate_structures_cost`'s note on `c213`, not
    an iteration variable, so `0.0` is a safe port-side default here too, unlike
    `convert_fpy_to_calendar`'s `cdrlife_cal`).

    Parameters
    ----------
    ireactor :
        1 if this run represents a reactor (net electricity), else 0. `.costs.ireactor`.
    ucturb :
        Turbine unit cost by coolant type, length-2. `.costs.ucturb`.
    i_blkt_coolant_type :
        1 (water) or 2 (helium). `.fwbs.i_blkt_coolant_type`.
    p_plant_electric_gross_mw :
        Gross electric power (MW). `.heat_transport.p_plant_electric_gross_mw`.

    Returns
    -------
    float
        `c23`.
    """
    exptpe = 0.83e0
    computed = (
        1.0e-6
        * jnp.asarray(ucturb)[i_blkt_coolant_type - 1]
        * safe_pow(p_plant_electric_gross_mw / 1200.0e0, exptpe)
    )
    return jnp.where(ireactor == 1, computed, 0.0)


def calculate_switchyard_cost(UCSWYD, lsa):
    """Account 241: switchyard. Ports `Costs.acc241` verbatim.

    Parameters
    ----------
    UCSWYD :
        Unit cost of switchyard. `.costs.UCSWYD`.
    lsa :
        Level of safety assurance, 1-4. `.costs.lsa`.

    Returns
    -------
    float
        `c241`.
    """
    cmlsa = jnp.asarray([0.5700e0, 0.7850e0, 0.8925e0, 1.0000e0])[lsa - 1]
    return 1.0e-6 * UCSWYD * cmlsa


def calculate_transformers_cost(UCPP, pacpmw, UCAP, p_plant_electric_base_total_mw, lsa):
    """Account 242: transformers. Ports `Costs.acc242` verbatim.

    Parameters
    ----------
    UCPP :
        Unit cost of primary power transformer ($/kW). `.costs.UCPP`.
    pacpmw :
        AC power requirement (MW). `.heat_transport.pacpmw`.
    UCAP :
        Unit cost of auxiliary power transformer ($/kW). `.costs.UCAP`.
    p_plant_electric_base_total_mw :
        Total base electric load (MW). `.heat_transport.p_plant_electric_base_total_mw`.
    lsa :
        Level of safety assurance, 1-4. `.costs.lsa`.

    Returns
    -------
    float
        `c242`.
    """
    cmlsa = jnp.asarray([0.5700e0, 0.7850e0, 0.8925e0, 1.0000e0])[lsa - 1]
    expepe = 0.9e0
    c242 = 1.0e-6 * (
        UCPP * safe_pow(pacpmw * 1.0e3, expepe)
        + UCAP * (p_plant_electric_base_total_mw * 1.0e3)
    )
    return c242 * cmlsa


def calculate_low_voltage_cost(UCLV, tlvpmw, lsa):
    """Account 243: low voltage. Ports `Costs.acc243` verbatim.

    Parameters
    ----------
    UCLV :
        Unit cost of low voltage equipment. `.costs.UCLV`.
    tlvpmw :
        Low-voltage power requirement (MW). `.heat_transport.tlvpmw`.
    lsa :
        Level of safety assurance, 1-4. `.costs.lsa`.

    Returns
    -------
    float
        `c243`.
    """
    cmlsa = jnp.asarray([0.5700e0, 0.7850e0, 0.8925e0, 1.0000e0])[lsa - 1]
    return 1.0e-6 * UCLV * tlvpmw * 1.0e3 / 0.8e0 * cmlsa


def calculate_diesel_generators_cost(UCDGEN, lsa):
    """Account 244: diesel generators (4 x 8 MW assumed). Ports `Costs.acc244`
    verbatim.

    Parameters
    ----------
    UCDGEN :
        Unit cost of one diesel generator. `.costs.UCDGEN`.
    lsa :
        Level of safety assurance, 1-4. `.costs.lsa`.

    Returns
    -------
    float
        `c244`.
    """
    cmlsa = jnp.asarray([0.5700e0, 0.7850e0, 0.8925e0, 1.0000e0])[lsa - 1]
    return 1.0e-6 * UCDGEN * 4.0e0 * cmlsa


def calculate_auxiliary_facility_power_cost(UCAF, lsa):
    """Account 245: auxiliary facility power. Ports `Costs.acc245` verbatim.

    Parameters
    ----------
    UCAF :
        Unit cost of auxiliary facility power equipment. `.costs.UCAF`.
    lsa :
        Level of safety assurance, 1-4. `.costs.lsa`.

    Returns
    -------
    float
        `c245`.
    """
    cmlsa = jnp.asarray([0.5700e0, 0.7850e0, 0.8925e0, 1.0000e0])[lsa - 1]
    return 1.0e-6 * UCAF * cmlsa


def calculate_electric_plant_equipment_cost(c241, c242, c243, c244, c245):
    """Account 24: electric plant equipment, total. Ports `Costs.acc24` verbatim -- a
    bare sum of the five sub-accounts above, kept as its own node since it is a real
    graph edge (`.costs.c24` feeds `run()`'s `cdirt` sum) rather than inlined.

    Parameters
    ----------
    c241, c242, c243, c244, c245 :
        The five sub-account costs (M$), this file's own outputs above.

    Returns
    -------
    float
        `c24`.
    """
    return c241 + c242 + c243 + c244 + c245


def calculate_misc_plant_equipment_cost(ucmisc, lsa):
    """Account 25: miscellaneous plant equipment. Ports `Costs.acc25` verbatim.

    Parameters
    ----------
    ucmisc :
        Unit cost of miscellaneous plant equipment. `.costs.ucmisc`.
    lsa :
        Level of safety assurance, 1-4. `.costs.lsa`.

    Returns
    -------
    float
        `c25`.
    """
    cmlsa = jnp.asarray([0.7700e0, 0.8850e0, 0.9425e0, 1.0000e0])[lsa - 1]
    return 1.0e-6 * ucmisc * cmlsa


def calculate_heat_rejection_cost(
    ireactor,
    p_fusion_total_mw,
    p_hcd_electric_total_mw,
    tfcmw,
    p_plant_primary_heat_mw,
    p_plant_electric_gross_mw,
    uchrs,
    lsa,
):
    """Account 26: heat rejection system. Ports `Costs.acc26` verbatim.

    Parameters
    ----------
    ireactor :
        1 if this run represents a reactor (net electricity), else 0. `.costs.ireactor`.
    p_fusion_total_mw :
        Total fusion power (MW), non-reactor branch. `.physics.p_fusion_total_mw`.
    p_hcd_electric_total_mw :
        Total H&CD electric power (MW), non-reactor branch.
        `.heat_transport.p_hcd_electric_total_mw`.
    tfcmw :
        TF coil power (MW), non-reactor branch. `.tfcoil.tfcmw`.
    p_plant_primary_heat_mw :
        Total primary heat (MW), reactor branch. `.heat_transport.p_plant_primary_heat_mw`.
    p_plant_electric_gross_mw :
        Gross electric power (MW), reactor branch.
        `.heat_transport.p_plant_electric_gross_mw`.
    uchrs :
        Reference cost of heat rejection system ($). `.costs.uchrs`.
    lsa :
        Level of safety assurance, 1-4. `.costs.lsa`.

    Returns
    -------
    float
        `c26`.
    """
    cmlsa = jnp.asarray([0.8000e0, 0.9000e0, 0.9500e0, 1.0000e0])[lsa - 1]
    pwrrej = jnp.where(
        ireactor == 0,
        p_fusion_total_mw + p_hcd_electric_total_mw + tfcmw,
        p_plant_primary_heat_mw - p_plant_electric_gross_mw,
    )
    return 1.0e-6 * uchrs * pwrrej / 2300.0e0 * cmlsa


# --------------------------------------------------------------------------------------
# Second porting wave: everything on `.costs.coe`'s transitive dependency chain.
#
# The first wave (above) ported 23 self-contained leaf accounts. `.costs.coe` needs 18
# more of `Costs`'s 43 methods plus the two accumulations `Costs.run()` performs inline
# (`cdirt`, `concost`) -- see `costs.md`'s "coverage map for `.costs.coe`" for the
# function-by-function derivation of that list, and for why the remaining two methods
# (`run`, `output`) are orchestration/reporting rather than computation.
# --------------------------------------------------------------------------------------

_UMASS = 1.660538921e-27
"""kg. `process.core.constants.UMASS` (`process/core/constants.py:280`), inlined rather
than imported so this module stays free of any `process.*` import at module scope -- the
same treatment `models/vacuum.py:53` already uses for the same constant."""

_DEN_COPPER = 8900.0
"""kg/m^3. `process.core.constants.DEN_COPPER` (`process/core/constants.py:289`)."""

_N_DAY_YEAR = 365.2425
"""days. `process.core.constants.N_DAY_YEAR` (`process/core/constants.py:307`)."""


def calculate_first_wall_cost(
    ife, lsa, UCFWA, UCFWS, a_fw_total, UCFWPS, fkind, ifueltyp
):
    """Account 221.1: first wall. Ports `Costs.acc2211` (`process/models/costs/
    costs.py:1145-1207`), magnetic-confinement arm only.

    `ife` is a **static** argument here, not a traced one (unlike
    `calculate_divertor_cost`'s, whose IFE arm is a plain zero): PROCESS's `ife == 1`
    arm reads four columns of the 2-D array `.ife.fwmatm` plus `.ife.uccarb`/`.ucconc`
    (`costs.py:1168-1195`) -- a genuinely different reads-set, which is
    `_audit/traceability_policy.md`'s own criterion for splitting rather than keeping a
    static kwarg. Only the `ife != 1` arm is ported; the other raises, so a request for
    it fails loudly instead of silently returning a magnetic-confinement number. The
    stellarator pipeline this project scopes to always has `.ife.ife == 0` (confirmed on
    the converged reference run, `costs.md` § coverage map).

    Parameters
    ----------
    ife :
        1 if this is an inertial-fusion device, else 0. `.ife.ife`. Static.
    lsa :
        Level of safety assurance, 1-4. `.costs.lsa`.
    UCFWA, UCFWS :
        First wall armour and structure unit costs ($/m^2). `.costs.UCFWA`,
        `.costs.UCFWS`.
    a_fw_total :
        First wall total surface area (m^2). `.first_wall.a_fw_total`.
    UCFWPS :
        First wall passive stabiliser cost ($). `.costs.UCFWPS`.
    fkind :
        Nth-of-a-kind multiplier. `.costs.fkind`.
    ifueltyp :
        1: first wall cost is a fuel cost; 2: capital cost, replacements are fuel cost;
        else: pure capital cost. `.costs.ifueltyp`.

    Returns
    -------
    tuple
        `(c2211, fwallcst)`.
    """
    if ife == 1:
        raise NotImplementedError(
            "acc2211's ife == 1 arm reads .ife.fwmatm (a 2-D array) plus .ife.uccarb/"
            ".ucconc -- a different reads-set, and the .ife.* subsystem is entirely "
            "unported (unit_registry.md has no `ife` unit). Only ife != 1 is ported."
        )
    cmlsa = jnp.asarray([0.5000e0, 0.7500e0, 0.8750e0, 1.0000e0])[lsa - 1]
    c2211_raw = fkind * (1.0e-6 * cmlsa * ((UCFWA + UCFWS) * a_fw_total + UCFWPS))

    fwallcst = jnp.where(
        (ifueltyp == 1) | (ifueltyp == 2),
        c2211_raw,
        0.0,
    )
    c2211 = jnp.where(ifueltyp == 1, 0.0, c2211_raw)
    return c2211, fwallcst


def calculate_blanket_cost(
    ife,
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
):
    """Account 221.2: blanket. Ports `Costs.acc2212` (`costs.py:1208-1327`),
    magnetic-confinement arm only.

    `ife` static for the same reason as `calculate_first_wall_cost`: the `ife == 1` arm
    reads `.ife.blmatm` (2-D), `.ife.uccarb`/`.ucconc`/`.ucflib`/`.mflibe`,
    `.costs.ucblli` and `.fwbs.m_blkt_lithium` (`costs.py:1298-1320`).

    `c22125`/`c22126`/`c22127` are identically zero on this arm (PROCESS assigns them
    `0.0` at `costs.py:1233-1235` and then scales them, so they stay zero) but are
    returned anyway because they are real `DataStructure` fields PROCESS writes on every
    call. **`c22128` is deliberately not returned**: `costs.py:1321-1323` writes it only
    inside the `ife == 1` arm and it is not a term of `c2212` in either arm, so this
    port never owns it.

    Parameters
    ----------
    ife :
        1 if this is an inertial-fusion device, else 0. `.ife.ife`. Static.
    lsa :
        Level of safety assurance, 1-4. `.costs.lsa`.
    m_blkt_beryllium, ucblbe :
        Blanket beryllium mass (kg) and unit cost ($/kg). `.fwbs.m_blkt_beryllium`,
        `.costs.ucblbe`.
    m_blkt_li2o, ucblli2o :
        Blanket Li2O mass (kg) and unit cost ($/kg). `.fwbs.m_blkt_li2o`,
        `.costs.ucblli2o`.
    m_blkt_steel_total, ucblss :
        Blanket stainless steel mass (kg) and unit cost ($/kg).
        `.fwbs.m_blkt_steel_total`, `.costs.ucblss`.
    m_blkt_vanadium, ucblvd :
        Blanket vanadium mass (kg) and unit cost ($/kg). `.fwbs.m_blkt_vanadium`,
        `.costs.ucblvd`.
    fkind :
        Nth-of-a-kind multiplier. `.costs.fkind`.
    ifueltyp :
        1: blanket cost is a fuel cost; 2: capital cost, replacements are fuel cost;
        else: pure capital cost. `.costs.ifueltyp`.

    Returns
    -------
    tuple
        `(c22121, c22122, c22123, c22124, c22125, c22126, c22127, c2212, blkcst)`.
    """
    if ife == 1:
        raise NotImplementedError(
            "acc2212's ife == 1 arm reads .ife.blmatm (a 2-D array), .ife.uccarb/"
            ".ucconc/.ucflib/.mflibe, .costs.ucblli and .fwbs.m_blkt_lithium -- a "
            "different reads-set over an entirely unported subsystem. Only ife != 1 "
            "is ported."
        )
    cmlsa = jnp.asarray([0.5000e0, 0.7500e0, 0.8750e0, 1.0000e0])[lsa - 1]
    scale = fkind * cmlsa

    c22121 = scale * (1.0e-6 * m_blkt_beryllium * ucblbe)
    c22122 = scale * (1.0e-6 * m_blkt_li2o * ucblli2o)
    c22123 = scale * (1.0e-6 * m_blkt_steel_total * ucblss)
    c22124 = scale * (1.0e-6 * m_blkt_vanadium * ucblvd)
    c22125 = scale * 0.0
    c22126 = scale * 0.0
    c22127 = scale * 0.0

    c2212_raw = c22121 + c22122 + c22123 + c22124 + c22125 + c22126 + c22127
    blkcst = jnp.where((ifueltyp == 1) | (ifueltyp == 2), c2212_raw, 0.0)
    c2212 = jnp.where(ifueltyp == 1, 0.0, c2212_raw)
    return c22121, c22122, c22123, c22124, c22125, c22126, c22127, c2212, blkcst


def calculate_shield_cost(ife, lsa, whtshld, ucshld, wpenshld, ucpens, fkind):
    """Account 221.3: shield. Ports `Costs.acc2213` (`costs.py:1328-1389`),
    magnetic-confinement arm only.

    `ife` static: the `ife == 1` arm reads `.ife.shmatm` (2-D) plus `.ife.uccarb`/
    `.ucconc`/`.costs.ucblli2o` (`costs.py:1345-1370`), and additionally zeroes
    `c22132` rather than computing it.

    Parameters
    ----------
    ife :
        1 if this is an inertial-fusion device, else 0. `.ife.ife`. Static.
    lsa :
        Level of safety assurance, 1-4. `.costs.lsa`.
    whtshld, ucshld :
        Shield mass (kg) and unit cost ($/kg). `.fwbs.whtshld`, `.costs.ucshld`.
    wpenshld, ucpens :
        Penetration shield mass (kg) and unit cost ($/kg). `.fwbs.wpenshld`,
        `.costs.ucpens`.
    fkind :
        Nth-of-a-kind multiplier. `.costs.fkind`.

    Returns
    -------
    tuple
        `(c22131, c22132, c2213)`.
    """
    if ife == 1:
        raise NotImplementedError(
            "acc2213's ife == 1 arm reads .ife.shmatm (a 2-D array) plus .ife.uccarb/"
            ".ucconc -- a different reads-set over an unported subsystem. Only "
            "ife != 1 is ported."
        )
    cmlsa = jnp.asarray([0.5000e0, 0.7500e0, 0.8750e0, 1.0000e0])[lsa - 1]
    c22131 = fkind * (1.0e-6 * whtshld * ucshld * cmlsa)
    c22132 = fkind * (1.0e-6 * wpenshld * ucpens * cmlsa)
    return c22131, c22132, c22131 + c22132


def calculate_reactor_cost(c2211, c2212, c2213, c2214, c2215):
    """Account 221 (total): reactor. Ports `Costs.acc221`'s own accumulation
    (`costs.py:970-972`) -- the five `self.accNNNN()` calls above it are separate nodes.

    Parameters
    ----------
    c2211, c2212, c2213, c2214, c2215 :
        First wall, blanket, shield, reactor structure and divertor costs (M$).
        `.costs.c2211` .. `.costs.c2215`.

    Returns
    -------
    float
        `c221`.
    """
    return c2211 + c2212 + c2213 + c2214 + c2215


def calculate_tf_magnet_cost_superconducting(
    supercond_cost_model,
    lsa,
    ucsc,
    i_tf_sc_mat,
    m_tf_coil_superconductor,
    len_tf_coil,
    n_tf_coil_turns,
    sc_mat_cost_0,
    j_crit_str_0,
    j_crit_str_tf,
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
):
    """Account 222.1: TF magnet assemblies, **superconducting arm**
    (`.tfcoil.i_tf_sup == 1`). Ports the `else` half of `Costs.acc2221`
    (`costs.py:1484-1590`).

    Split into two functions/nodes rather than one node with a static `i_tf_sup` kwarg,
    per `_audit/traceability_policy.md`'s split default: the two arms share no
    computation and read disjoint sets (this one reads
    `.tfcoil.m_tf_coil_superconductor`/`m_tf_coil_copper`/`m_tf_coil_case`/`len_tf_coil`/
    `n_tf_coil_turns`/`i_tf_sc_mat` and `.structure.aintmass`/`clgsmass`; the resistive
    arm reads `.tfcoil.whtcp`/`whttflgs` and `.physics.itart` and nothing else). Keeping
    one node would make it declare the union, inventing graph edges that do not exist in
    the run being modelled -- exactly what `configuration.py`'s own module docstring
    rejects. **Not wired as a `Switch` either**, because it would have to nest inside
    `.costs.i_cost_model`'s arm, and nested switches are a still-open gap
    (`_audit/next_steps.md` §1); only the superconducting node is registered, matching
    both PROCESS's own default (`tfcoil_variables.py:261`) and the reference run.

    `supercond_cost_model` **is** kept as a static kwarg rather than split: it selects
    between two one-line cost-per-metre formulas whose reads differ by three scalar
    fields (`.costs.sc_mat_cost_0`, `.tfcoil.j_crit_str_0`, `.tfcoil.j_crit_str_tf`
    versus `.costs.ucsc`, `.tfcoil.m_tf_coil_superconductor`), inside an otherwise
    shared ~100-line body -- the size-aware exception `_audit/next_steps.md` §1 already
    tracks with three prior instances. Its value is checked automatically against the
    run by `mda_harness.switch_audit`.

    Parameters
    ----------
    supercond_cost_model :
        0: legacy `$/kg`-style superconductor costing; else: `sc_mat_cost_0`-based.
        `.costs.supercond_cost_model`. Static.
    lsa :
        Level of safety assurance, 1-4. `.costs.lsa`.
    ucsc :
        Superconductor unit costs by material, length-9 ($/kg). `.costs.ucsc`.
    i_tf_sc_mat :
        TF superconductor material index, 1-9. `.tfcoil.i_tf_sc_mat`.
    m_tf_coil_superconductor :
        Superconductor mass per TF coil (kg). `.tfcoil.m_tf_coil_superconductor`.
    len_tf_coil :
        TF coil length (m). `.tfcoil.len_tf_coil`.
    n_tf_coil_turns :
        Turns per TF coil. `.tfcoil.n_tf_coil_turns`.
    sc_mat_cost_0 :
        Reference superconductor strand costs by material, length-9.
        `.costs.sc_mat_cost_0`.
    j_crit_str_0 :
        Reference critical strand current densities by material, length-9 (A/m^2).
        `.tfcoil.j_crit_str_0`.
    j_crit_str_tf :
        TF critical strand current density (A/m^2). `.tfcoil.j_crit_str_tf`.
    uccu, m_tf_coil_copper :
        Copper unit cost ($/kg) and mass per TF coil (kg). `.costs.uccu`,
        `.tfcoil.m_tf_coil_copper`.
    cconshtf, cconfix :
        Conduit/sheath and fixed conductor costs per metre ($/m). `.costs.cconshtf`,
        `.costs.cconfix`.
    n_tf_coils :
        Number of TF coils. `.tfcoil.n_tf_coils`.
    ucwindtf :
        TF winding unit cost ($/m). `.costs.ucwindtf`.
    m_tf_coil_case, uccase :
        TF coil case mass (kg) and unit cost ($/kg). `.tfcoil.m_tf_coil_case`,
        `.costs.uccase`.
    aintmass, UCINT :
        Intercoil structure mass (kg) and unit cost ($/kg). `.structure.aintmass`,
        `.costs.UCINT`.
    clgsmass, UCGSS :
        Gravity support structure mass (kg) and unit cost ($/kg).
        `.structure.clgsmass`, `.costs.UCGSS`.
    fkind :
        Nth-of-a-kind multiplier. `.costs.fkind`.

    Returns
    -------
    tuple
        `(c22211, c22212, c22213, c22214, c22215, c2221)`.
    """
    cmlsa = jnp.asarray([0.6900e0, 0.8450e0, 0.9225e0, 1.0000e0])[lsa - 1]

    if supercond_cost_model == 0:
        costtfsc = (
            jnp.asarray(ucsc)[i_tf_sc_mat - 1]
            * m_tf_coil_superconductor
            / (len_tf_coil * n_tf_coil_turns)
        )
    else:
        costtfsc = (
            jnp.asarray(sc_mat_cost_0)[i_tf_sc_mat - 1]
            * jnp.asarray(j_crit_str_0)[i_tf_sc_mat - 1]
            / j_crit_str_tf
        )

    costtfcu = uccu * m_tf_coil_copper / (len_tf_coil * n_tf_coil_turns)
    ctfconpm = costtfsc + costtfcu + cconshtf + cconfix

    winding_length = n_tf_coils * len_tf_coil * n_tf_coil_turns
    c22211 = fkind * (1.0e-6 * ctfconpm * winding_length) * cmlsa
    c22212 = fkind * (1.0e-6 * ucwindtf * winding_length) * cmlsa
    c22213 = fkind * (1.0e-6 * (m_tf_coil_case * uccase) * n_tf_coils) * cmlsa
    c22214 = fkind * (1.0e-6 * aintmass * UCINT) * cmlsa
    c22215 = fkind * (1.0e-6 * clgsmass * UCGSS) * cmlsa

    c2221 = c22211 + c22212 + c22213 + c22214 + c22215
    return c22211, c22212, c22213, c22214, c22215, c2221


def calculate_tf_magnet_cost_resistive(
    lsa, whtcp, uccpcl1, whttflgs, uccpclb, itart, ifueltyp, fkind
):
    """Account 222.1: TF magnet assemblies, **resistive arm**
    (`.tfcoil.i_tf_sup != 1`). Ports the `if` half of `Costs.acc2221`
    (`costs.py:1447-1483`). See `calculate_tf_magnet_cost_superconducting`'s docstring
    for why the two arms are separate functions, and why only that one is registered.

    Parameters
    ----------
    lsa :
        Level of safety assurance, 1-4. `.costs.lsa`.
    whtcp, uccpcl1 :
        Inboard TF leg (centrepost) mass (kg) and unit cost ($/kg). `.tfcoil.whtcp`,
        `.costs.uccpcl1`.
    whttflgs, uccpclb :
        Outboard TF leg mass (kg) and unit cost ($/kg). `.tfcoil.whttflgs`,
        `.costs.uccpclb`.
    itart :
        1 if this is a spherical tokamak (has a centrepost), else 0. `.physics.itart`.
    ifueltyp :
        1: centrepost cost is a fuel cost; 2: capital cost, replacements are fuel cost;
        else: pure capital cost. `.costs.ifueltyp`.
    fkind :
        Nth-of-a-kind multiplier. `.costs.fkind`.

    Returns
    -------
    tuple
        `(c22211, c22212, c2221, cpstcst)`.
    """
    cmlsa = jnp.asarray([0.6900e0, 0.8450e0, 0.9225e0, 1.0000e0])[lsa - 1]

    c22211_raw = fkind * (1.0e-6 * whtcp * uccpcl1 * cmlsa)
    is_tart = itart == 1
    cpstcst = jnp.where(is_tart & ((ifueltyp == 1) | (ifueltyp == 2)), c22211_raw, 0.0)
    c22211 = jnp.where(is_tart & (ifueltyp == 1), 0.0, c22211_raw)

    c22212 = fkind * (1.0e-6 * whttflgs * uccpclb * cmlsa)
    return c22211, c22212, c22211 + c22212, cpstcst


def calculate_pf_magnet_cost(
    n_cs_pf_coils,
    iohcl,
    i_pf_conductor,
    supercond_cost_model,
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
    dcond,
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
):
    """Account 222.2: PF magnet assemblies. Ports `Costs.acc2222`
    (`costs.py:1591-1840`) in full, including its central-solenoid block.

    **`costs.md`'s original "dynamic-length loop is a structural JAX blocker" finding is
    resolved, not worked around.** The loop bound is `.pf_coil.n_cs_pf_coils`, a
    run-configuration count fixed before the solve starts: it is not an iteration
    variable (`grep -n n_cs_pf_coils process/core/solver/iteration_variables.py` -> no
    match) and not a scan variable (`process/core/scan.py` -> no match), so it is
    exactly `naming_convention.md`'s static-kwarg category -- the same move
    `ImpurityRadiationTotals.imp_indices` already makes for "which impurity species
    exist". With `n_cs_pf_coils` and `.build.iohcl` static, both loops are ordinary
    Python `range`s unrolled at trace time and no `lax.fori_loop` or padding is needed.
    `mda_harness.switch_audit` checks both values against the run automatically.

    `i_pf_conductor` and `supercond_cost_model` are static for the same reason as their
    counterparts elsewhere in this file (four nested `if`/`elif` chains selecting among
    formulas whose per-branch reads differ by a handful of scalars inside one shared
    ~200-line body).

    On the reference stellarator run every output of this function is exactly zero
    (`n_cs_pf_coils == 0`, `iohcl == 0`, and both `.pf_coil.m_pf_coil_structure_total`
    and `.structure.fncmass` are `0.0`) -- PROCESS never runs its PF coil model for a
    stellarator. That is reproduced, not special-cased.

    Parameters
    ----------
    n_cs_pf_coils :
        Number of PF coils including the central solenoid. `.pf_coil.n_cs_pf_coils`.
        Static (loop bound).
    iohcl :
        1 if a central solenoid is present, else 0. `.build.iohcl`. Static (loop bound).
    i_pf_conductor :
        0: superconducting PF coils; 1: resistive. `.pf_coil.i_pf_conductor`. Static.
    supercond_cost_model :
        0: legacy `$/kg`-style superconductor costing; else: `sc_mat_cost_0`-based.
        `.costs.supercond_cost_model`. Static.
    lsa :
        Level of safety assurance, 1-4. `.costs.lsa`.
    r_pf_coil_middle :
        PF coil mid-plane radii (m), one per coil. `.pf_coil.r_pf_coil_middle`.
    n_pf_coil_turns :
        Turns per PF coil. `.pf_coil.n_pf_coil_turns`.
    cconshpf :
        Sheath/conduit cost per metre for superconducting PF cable ($/m).
        `.costs.cconshpf`.
    ucsc :
        Superconductor unit costs by material, length-9 ($/kg). `.costs.ucsc`.
    i_pf_superconductor :
        PF superconductor material index, 1-9. `.pf_coil.i_pf_superconductor`.
    fcupfsu :
        Copper fraction of the PF superconducting strand. `.pf_coil.fcupfsu`.
    f_a_pf_coil_void :
        Void fraction of each PF coil winding pack. `.pf_coil.f_a_pf_coil_void`.
    c_pf_cs_coils_peak_ma :
        Peak current in each PF/CS coil (MA). `.pf_coil.c_pf_cs_coils_peak_ma`.
    j_pf_coil_wp_peak :
        Peak winding-pack current density in each PF coil (A/m^2).
        `.pf_coil.j_pf_coil_wp_peak`.
    dcond :
        Superconductor densities by material, length-9 (kg/m^3). `.tfcoil.dcond`.
    sc_mat_cost_0, j_crit_str_0 :
        Reference strand cost and critical current density by material, length-9.
        `.costs.sc_mat_cost_0`, `.tfcoil.j_crit_str_0`.
    j_crit_str_pf, j_crit_str_cs :
        PF and CS critical strand current densities (A/m^2). `.pf_coil.j_crit_str_pf`,
        `.pf_coil.j_crit_str_cs`.
    uccu :
        Copper unit cost ($/kg). `.costs.uccu`.
    cconfix :
        Fixed conductor cost per metre ($/m). `.costs.cconfix`.
    i_cs_superconductor :
        CS superconductor material index, 1-9. `.pf_coil.i_cs_superconductor`.
    a_cs_cable_space :
        CS cable space cross-sectional area (m^2). `.pf_coil.a_cs_cable_space`.
    f_a_cs_void, fcuohsu :
        CS void fraction and copper fraction. `.pf_coil.f_a_cs_void`,
        `.pf_coil.fcuohsu`.
    ucwindpf :
        PF winding unit cost ($/m). `.costs.ucwindpf`.
    uccase, m_pf_coil_structure_total :
        Steel case unit cost ($/kg) and total PF coil structure mass (kg).
        `.costs.uccase`, `.pf_coil.m_pf_coil_structure_total`.
    ucfnc, fncmass :
        Outer PF coil support unit cost ($/kg) and mass (kg). `.costs.ucfnc`,
        `.structure.fncmass`.
    fkind :
        Nth-of-a-kind multiplier. `.costs.fkind`.

    Returns
    -------
    tuple
        `(c22221, c22222, c22223, c22224, c2222)`.
    """
    cmlsa = jnp.asarray([0.6900e0, 0.8450e0, 0.9225e0, 1.0000e0])[lsa - 1]
    scale = fkind * cmlsa

    r_pf_coil_middle = jnp.asarray(r_pf_coil_middle)
    n_pf_coil_turns = jnp.asarray(n_pf_coil_turns)
    f_a_pf_coil_void = jnp.asarray(f_a_pf_coil_void)
    c_pf_cs_coils_peak_ma = jnp.asarray(c_pf_cs_coils_peak_ma)
    j_pf_coil_wp_peak = jnp.asarray(j_pf_coil_wp_peak)

    # Total length of PF coil windings (m).
    pfwndl = 0.0
    for i in range(n_cs_pf_coils):
        pfwndl += 2.0 * jnp.pi * r_pf_coil_middle[i] * n_pf_coil_turns[i]

    # `costpfsh` is the cost per metre of the steel conduit/sheath around each
    # superconducting cable, so zero for resistive coils (`i_pf_conductor == 1`,
    # `PFConductorModel.RESISTIVE`, `pfcoil_variables.py`).
    costpfsh = 0.0 if i_pf_conductor == 1 else cconshpf
    is_superconducting = i_pf_conductor == 0

    npf = n_cs_pf_coils - 1 if iohcl == 1 else n_cs_pf_coils

    c22221 = 0.0
    for i in range(npf):
        if supercond_cost_model == 0:
            costpfsc = (
                (
                    jnp.asarray(ucsc)[i_pf_superconductor - 1]
                    * (1.0e0 - fcupfsu)
                    * (1.0e0 - f_a_pf_coil_void[i])
                    * abs(c_pf_cs_coils_peak_ma[i] / n_pf_coil_turns[i])
                    * 1.0e6
                    / j_pf_coil_wp_peak[i]
                    * jnp.asarray(dcond)[i_pf_superconductor - 1]
                )
                if is_superconducting
                else 0.0
            )
        elif is_superconducting:
            costpfsc = (
                jnp.asarray(sc_mat_cost_0)[i_pf_superconductor - 1]
                * jnp.asarray(j_crit_str_0)[i_pf_superconductor - 1]
                / j_crit_str_pf
            )
        else:
            costpfsc = 0.0

        copper_fraction = fcupfsu if is_superconducting else 1.0
        costpfcu = (
            uccu
            * copper_fraction
            * (1.0e0 - f_a_pf_coil_void[i])
            * abs(c_pf_cs_coils_peak_ma[i] / n_pf_coil_turns[i])
            * 1.0e6
            / j_pf_coil_wp_peak[i]
            * _DEN_COPPER
        )

        cpfconpm = costpfsc + costpfcu + costpfsh + cconfix
        c22221 += (
            1.0e-6 * 2.0 * jnp.pi * r_pf_coil_middle[i] * n_pf_coil_turns[i] * cpfconpm
        )

    if iohcl == 1:
        cs = n_cs_pf_coils - 1
        if supercond_cost_model == 0:
            #  Issue #328: use the CS conductor cross-sectional area (m^2).
            costpfsc = (
                (
                    jnp.asarray(ucsc)[i_cs_superconductor - 1]
                    * a_cs_cable_space
                    * (1 - f_a_cs_void)
                    * (1 - fcuohsu)
                    / n_pf_coil_turns[cs]
                    * jnp.asarray(dcond)[i_cs_superconductor - 1]
                )
                if is_superconducting
                else 0.0
            )
        elif is_superconducting:
            costpfsc = (
                jnp.asarray(sc_mat_cost_0)[i_cs_superconductor - 1]
                * jnp.asarray(j_crit_str_0)[i_cs_superconductor - 1]
                / j_crit_str_cs
            )
        else:
            costpfsc = 0.0

        # PROCESS's own comment on the resistive branch: "MDK I don't know if this is
        # ccorrect as we never use the resistive model" (`costs.py:1758-1759`). The
        # only difference is the missing `fcuohsu` factor; reproduced as written.
        cs_copper_fraction = fcuohsu if is_superconducting else 1.0
        costpfcu = (
            uccu
            * a_cs_cable_space
            * (1 - f_a_cs_void)
            * cs_copper_fraction
            / n_pf_coil_turns[cs]
            * _DEN_COPPER
        )

        cpfconpm = costpfsc + costpfcu + costpfsh + cconfix
        c22221 += (
            1.0e-6 * 2.0 * jnp.pi * r_pf_coil_middle[cs] * n_pf_coil_turns[cs] * cpfconpm
        )

    c22221 = scale * c22221
    c22222 = scale * (1.0e-6 * ucwindpf * pfwndl)
    c22223 = scale * (1.0e-6 * uccase * m_pf_coil_structure_total)
    c22224 = scale * (1.0e-6 * ucfnc * fncmass)

    c2222 = c22221 + c22222 + c22223 + c22224
    return c22221, c22222, c22223, c22224, c2222


def calculate_magnets_cost(ife, c2221, c2222, c2223):
    """Account 222 (total): magnets, including the vacuum vessel. Ports `Costs.acc222`'s
    own accumulation and its `ife == 1` early return (`costs.py:974-999`).

    `ife` stays a plain traced argument here (unlike `calculate_first_wall_cost`'s): the
    IFE arm is a bare `c222 = 0.0`, no different reads at all -- the same shape
    `calculate_divertor_cost` already handles with `jnp.where`.

    Parameters
    ----------
    ife :
        1 if this is an inertial-fusion device, else 0. `.ife.ife`.
    c2221, c2222, c2223 :
        TF magnet, PF magnet and vacuum vessel costs (M$). `.costs.c2221`,
        `.costs.c2222`, `.costs.c2223`.

    Returns
    -------
    float
        `c222`.
    """
    return jnp.where(ife == 1, 0.0, c2221 + c2222 + c2223)


def calculate_power_injection_cost(
    ife,
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
):
    """Account 223: power injection. Ports `Costs.acc223` (`costs.py:1854-1979`),
    magnetic-confinement arm only.

    **Reproduces two real PROCESS defects rather than fixing them**, per this project's
    standing policy:

    1. **`c2233` (neutral beam) is only ever computed when `ifueltyp == 1`.** Its
       assignment (`costs.py:1909-1915`) is nested inside the `if ifueltyp == 1:` block
       that finishes `c2232`, not inside its own top-level "Account 223.3" section the
       comment above it claims to start. On any run with `ifueltyp != 1` the field is
       never written at all and keeps `cost_variables.py:165`'s dataclass default,
       `0.0`. Because `ifueltyp` is a run-configuration constant (not an iteration
       variable, not a scan variable), "never written in *this* call" is equivalent to
       "never written in *any* call of the run", so the port's `0.0` on that branch is
       exact, not an approximation -- unlike `convert_fpy_to_calendar`'s `cdrlife_cal`
       (open question 2), whose gating value genuinely can change between iterations.
       Confirmed on the reference run: `.costs.c2233 == 0.0` with `ifueltyp == 0`.
    2. **`fkind` is applied only on the `ifueltyp == 1` branch.** For all three
       sub-accounts PROCESS writes `cNNNN = ...` unconditionally and then applies both
       `(1 - fcdfuel)` *and* `fkind` inside `if ifueltyp == 1:` (`costs.py:1877-1881`,
       `1899-1903`, `1917-1921`), so the Nth-of-a-kind multiplier silently does not
       apply to Account 223 on any other run. Every other account in this file applies
       `fkind` unconditionally. Reproduced as written.

    `.costs.c2234` (a fourth term of `c223`) is written **only** inside the IFE
    `ifueltyp == 1` branch, where PROCESS sets it to `0.0` (`costs.py:1968`); nothing
    anywhere in `process/` ever gives it a non-zero value (grepped). It is therefore the
    literal `0.0` here, and is not an output of this node.

    `ife` static: the IFE arm is a four-way driver-cost dispatch reading
    `.ife.ifedrv`/`.dcdrv0..2`/`.cdriv0..3`/`.edrive`/`.etadrv`/`.mcdriv`
    (`costs.py:1924-1969`) -- an entirely different reads-set over an unported
    subsystem.

    Parameters
    ----------
    ife :
        1 if this is an inertial-fusion device, else 0. `.ife.ife`. Static.
    ucech, p_hcd_ecrh_injected_total_mw :
        ECH unit cost ($/W) and injected ECRH power (MW). `.costs.ucech`,
        `.current_drive.p_hcd_ecrh_injected_total_mw`.
    i_hcd_primary :
        Primary heating/current-drive model index; `!= 2` selects lower hybrid, `== 2`
        ion cyclotron. `.current_drive.i_hcd_primary`.
    uclh, ucich :
        Lower hybrid and ICH unit costs ($/W). `.costs.uclh`, `.costs.ucich`.
    p_hcd_lowhyb_injected_total_mw :
        Injected lower-hybrid power (MW).
        `.current_drive.p_hcd_lowhyb_injected_total_mw`.
    ucnbi, p_beam_injected_mw :
        Neutral beam unit cost ($/W) and injected beam power (MW). `.costs.ucnbi`,
        `.current_drive.p_beam_injected_mw`.
    ifueltyp :
        1: the fraction `fcdfuel` of the current drive cost is a recurring fuel cost.
        `.costs.ifueltyp`.
    fcdfuel :
        Fraction of current drive cost treated as fuel. `.costs.fcdfuel`.
    fkind :
        Nth-of-a-kind multiplier. `.costs.fkind`.

    Returns
    -------
    tuple
        `(c2231, c2232, c2233, c223, cdcost)`.
    """
    if ife == 1:
        raise NotImplementedError(
            "acc223's ife == 1 arm is a four-way IFE driver-cost dispatch reading "
            ".ife.ifedrv/.cdriv0..3/.dcdrv0..2/.edrive/.etadrv/.mcdriv -- a different "
            "reads-set over an unported subsystem. Only ife != 1 is ported."
        )
    exprf = 1.0e0
    is_fuel = ifueltyp == 1
    fuel_scale = fkind * (1.0e0 - fcdfuel)

    c2231_raw = 1.0e-6 * ucech * (1.0e6 * p_hcd_ecrh_injected_total_mw) ** exprf
    c2231 = jnp.where(is_fuel, fuel_scale * c2231_raw, c2231_raw)

    c2232_raw = (
        1.0e-6
        * jnp.where(i_hcd_primary != 2, uclh, ucich)
        * (1.0e6 * p_hcd_lowhyb_injected_total_mw) ** exprf
    )
    c2232 = jnp.where(is_fuel, fuel_scale * c2232_raw, c2232_raw)

    c2233_raw = 1.0e-6 * ucnbi * (1.0e6 * p_beam_injected_mw) ** exprf
    c2233 = jnp.where(is_fuel, fuel_scale * c2233_raw, 0.0)

    c2234 = 0.0
    c223 = c2231 + c2232 + c2233 + c2234
    return c2231, c2232, c2233, c223, c223


def calculate_energy_storage_cost(
    i_pulsed_plant, istore, p_plant_electric_net_mw, fkind
):
    """Account 225.3: energy storage. Ports `Costs.acc2253` (`costs.py:2598-2702`).

    `i_pulsed_plant` and `istore` are static: `istore == 3` reads three fields the other
    options do not (`.heat_transport.p_plant_primary_heat_mw`,
    `.times.t_plant_pulse_no_burn`, `.pulse.dtstor`), and options 1/2 are pure literal
    sums, so this is `traceability_policy.md`'s split default expressed as one function
    with a Python branch on a static value plus a loud refusal on the unported arm.
    Options 1 and 2 are the ELECTROWATT report's two thermal-storage designs (AEA FUS
    205); their itemised sums are folded to the literals `37.2` and `358.7` M$ (1992
    pounds) here, each term kept as a comment against `costs.py`'s own listing.

    PROCESS raises `ProcessValueError` for `istore` outside 1-3 when
    `i_pulsed_plant == 1`; this port raises `ValueError` on the same condition.

    Parameters
    ----------
    i_pulsed_plant :
        1 if this is a pulsed plant, else 0. `.pulse.i_pulsed_plant`. Static.
    istore :
        Thermal storage option, 1-3. `.pulse.istore`. Static.
    p_plant_electric_net_mw :
        Net electric power (MW). `.heat_transport.p_plant_electric_net_mw`.
    fkind :
        Nth-of-a-kind multiplier. `.costs.fkind`.

    Returns
    -------
    float
        `c2253`.
    """
    if i_pulsed_plant == 1:
        if istore == 1:
            # 0.1 condensate tank + 0.8 feedpump + 4.0 turbine-generator duty
            # + 0.5 auxiliary transformer + 2.8 drum + 29.0 externally fired
            # superheater (`costs.py:2617-2643`).
            c2253 = 0.1e0 + 0.8e0 + 4.0e0 + 0.5e0 + 2.8e0 + 29.0e0
        elif istore == 2:
            # 0.1 + 0.8 + 2.8 + 4.0 + 330.0 fired boiler + 1.0 steam bypass
            # + 2.0 dump condenser + 18.0 cooling water (`costs.py:2645-2682`).
            c2253 = 0.1e0 + 0.8e0 + 2.8e0 + 4.0e0 + 330.0e0 + 1.0e0 + 2.0e0 + 18.0e0
        elif istore == 3:
            raise NotImplementedError(
                "acc2253's istore == 3 arm (a stainless-steel thermal storage block) "
                "reads .heat_transport.p_plant_primary_heat_mw, "
                ".times.t_plant_pulse_no_burn and .pulse.dtstor, which options 1/2 do "
                "not -- a different reads-set. Not ported; the reference run has "
                ".pulse.i_pulsed_plant == 0, so no istore arm is reached at all."
            )
        else:
            raise ValueError(f"Illegal value for istore: {istore}")
    else:
        c2253 = 0.0e0

    if istore < 3:
        #  Scale with net electric power, then convert 1992 pounds to 1990 dollars
        #  (inflation 5%/yr + 1.5 $/pound exchange rate).
        c2253 = c2253 * p_plant_electric_net_mw / 1200.0e0
        c2253 *= 1.36e0

    return fkind * c2253


def calculate_power_conditioning_cost(ife, c2251, c2252, c2253):
    """Account 225 (total): power conditioning. Ports `Costs.acc225`'s own accumulation
    and its `ife == 1` zero branch (`costs.py:1000-1024`). `ife` traced, same shape as
    `calculate_magnets_cost`.

    Parameters
    ----------
    ife :
        1 if this is an inertial-fusion device, else 0. `.ife.ife`.
    c2251, c2252, c2253 :
        TF coil, PF coil and energy storage power conditioning costs (M$).
        `.costs.c2251`, `.costs.c2252`, `.costs.c2253`.

    Returns
    -------
    float
        `c225`.
    """
    return jnp.where(ife == 1, 0.0, c2251 + c2252 + c2253)


def calculate_auxiliary_component_cooling_cost(
    ife,
    lsa,
    UCAHTS,
    p_hcd_electric_loss_mw,
    p_cryo_plant_electric_mw,
    vachtmw,
    p_tritium_plant_electric_mw,
    fachtmw,
    fkind,
):
    """Account 2262: auxiliary component cooling. Ports `Costs.acc2262`
    (`costs.py:2263-2300`), magnetic-confinement arm only.

    `ife` static: the IFE arm *adds* two further terms reading `.ife.tdspmw` and
    `.ife.tfacmw` (`costs.py:2285-2293`) -- a different reads-set over an unported
    subsystem, unlike `calculate_magnets_cost`'s plain zero.

    Parameters
    ----------
    ife :
        1 if this is an inertial-fusion device, else 0. `.ife.ife`. Static.
    lsa :
        Level of safety assurance, 1-4. `.costs.lsa`.
    UCAHTS :
        Auxiliary heat transport system unit cost ($/W). `.costs.UCAHTS`.
    p_hcd_electric_loss_mw :
        Heating/current drive electrical loss (MW).
        `.heat_transport.p_hcd_electric_loss_mw`.
    p_cryo_plant_electric_mw :
        Cryoplant electric power (MW). `.heat_transport.p_cryo_plant_electric_mw`.
    vachtmw :
        Vacuum pump power (MW). `.heat_transport.vachtmw`.
    p_tritium_plant_electric_mw :
        Tritium plant electric power (MW).
        `.heat_transport.p_tritium_plant_electric_mw`.
    fachtmw :
        Facility power (MW). `.heat_transport.fachtmw`.
    fkind :
        Nth-of-a-kind multiplier. `.costs.fkind`.

    Returns
    -------
    tuple
        `(cppa, c2262)`.
    """
    if ife == 1:
        raise NotImplementedError(
            "acc2262's ife == 1 arm adds terms reading .ife.tdspmw/.ife.tfacmw -- a "
            "different reads-set over an unported subsystem. Only ife != 1 is ported."
        )
    cmlsa = jnp.asarray([0.4000e0, 0.7000e0, 0.8500e0, 1.0000e0])[lsa - 1]
    exphts = 0.7e0

    cppa = (
        1.0e-6
        * UCAHTS
        * (
            safe_pow(1.0e6 * p_hcd_electric_loss_mw, exphts)
            + safe_pow(1.0e6 * p_cryo_plant_electric_mw, exphts)
            + safe_pow(1.0e6 * vachtmw, exphts)
            + safe_pow(1.0e6 * p_tritium_plant_electric_mw, exphts)
            + safe_pow(1.0e6 * fachtmw, exphts)
        )
    )
    cppa = fkind * cppa * cmlsa
    return cppa, cppa


def calculate_cryogenic_system_cost(lsa, uccry, temp_tf_cryo, helpow, fkind):
    """Account 2263: cryogenic system. Ports `Costs.acc2263` (`costs.py:2301-2322`) --
    branch-free in the source.

    Parameters
    ----------
    lsa :
        Level of safety assurance, 1-4. `.costs.lsa`.
    uccry :
        Cryoplant unit cost at 4.5 K ($/W). `.costs.uccry`.
    temp_tf_cryo :
        TF coil cryogenic operating temperature (K). `.tfcoil.temp_tf_cryo`.
    helpow :
        Cryogenic heat load (W). `.heat_transport.helpow`.
    fkind :
        Nth-of-a-kind multiplier. `.costs.fkind`.

    Returns
    -------
    float
        `c2263`.
    """
    cmlsa = jnp.asarray([0.4000e0, 0.7000e0, 0.8500e0, 1.0000e0])[lsa - 1]
    expcry = 0.67e0
    c2263 = 1.0e-6 * uccry * 4.5e0 / temp_tf_cryo * safe_pow(helpow, expcry)
    return fkind * c2263 * cmlsa


def calculate_heat_transport_system_cost(c2261, c2262, c2263):
    """Account 226 (total): heat transport system. Ports `Costs.acc226`
    (`costs.py:2210-2218`).

    Parameters
    ----------
    c2261, c2262, c2263 :
        Reactor cooling, auxiliary component cooling and cryogenic system costs (M$).
        `.costs.c2261`, `.costs.c2262`, `.costs.c2263`.

    Returns
    -------
    float
        `c226`.
    """
    return c2261 + c2262 + c2263


def calculate_fuel_processing_cost(ife, rndfuel, m_fuel_amu, UCFPR, fkind):
    """Account 2272: fuel processing and purification. Ports `Costs.acc2272`
    (`costs.py:2344-2382`), magnetic-confinement arm only.

    This is the one method in `costs.py` that writes **outside** `.costs.*`: it owns
    `.physics.wtgpd` (fuel throughput, g/day), which it then consumes itself and which
    `coelc` reads for the He3 fuel cost (`costs.py:2915`). Nothing else in `process/`
    writes `.physics.wtgpd` (grepped), so this node is its sole producer.

    `ife` static: the IFE arm computes `wtgpd` from `.ife.gain`/`.edrive`/`.fburn`/
    `.reprat` instead (`costs.py:2364-2372`) -- a different reads-set.

    Parameters
    ----------
    ife :
        1 if this is an inertial-fusion device, else 0. `.ife.ife`. Static.
    rndfuel :
        Fuel burn rate (reactions/s). `.physics.rndfuel`.
    m_fuel_amu :
        Average mass of the fuel nuclei (amu). `.physics.m_fuel_amu`.
    UCFPR :
        Fuel processing unit cost ($). `.costs.UCFPR`.
    fkind :
        Nth-of-a-kind multiplier. `.costs.fkind`.

    Returns
    -------
    tuple
        `(wtgpd, c2272)`.
    """
    if ife == 1:
        raise NotImplementedError(
            "acc2272's ife == 1 arm computes wtgpd from .ife.gain/.edrive/.fburn/"
            ".reprat -- a different reads-set over an unported subsystem. Only "
            "ife != 1 is ported."
        )
    #  2 nuclei * reactions/sec * kg/nucleus * g/kg * sec/day.
    wtgpd = 2.0e0 * rndfuel * m_fuel_amu * _UMASS * 1000.0e0 * 86400.0e0

    #  Assumes He3 costs the same as tritium to process.
    c2272 = 1.0e-6 * UCFPR * (0.5e0 + 0.5e0 * safe_pow(wtgpd / 60.0e0, 0.67e0))
    return wtgpd, fkind * c2272


def calculate_atmospheric_recovery_cost(
    f_plasma_fuel_tritium, UCDTC, volrci, wsvol, fkind
):
    """Account 2273: atmospheric recovery systems. Ports `Costs.acc2273`
    (`costs.py:2383-2403`). The tritium threshold is a plain traced `jnp.where`: both
    branches read the same fields.

    Parameters
    ----------
    f_plasma_fuel_tritium :
        Tritium fraction of the fuel; no detritiation is needed below 1e-3 (a pure
        D-He3 reaction). `.physics.f_plasma_fuel_tritium`.
    UCDTC :
        Detritiation unit cost ($/(10 m^3/min)). `.costs.UCDTC`.
    volrci, wsvol :
        Reactor building and warm shop volumes (m^3). `.buildings.volrci`,
        `.buildings.wsvol`.
    fkind :
        Nth-of-a-kind multiplier. `.costs.fkind`.

    Returns
    -------
    float
        `c2273`.
    """
    cfrht = 1.0e5
    c2273 = jnp.where(
        f_plasma_fuel_tritium > 1.0e-3,
        1.0e-6 * UCDTC * (safe_pow(cfrht / 1.0e4, 0.6e0) * (volrci + wsvol)),
        0.0,
    )
    return fkind * c2273


def calculate_fuel_handling_cost(c2271, c2272, c2273, c2274):
    """Account 227 (total): fuel handling. Ports `Costs.acc227` (`costs.py:2323-2334`).

    Parameters
    ----------
    c2271, c2272, c2273, c2274 :
        Fuelling system, fuel processing, atmospheric recovery and nuclear building
        ventilation costs (M$). `.costs.c2271` .. `.costs.c2274`.

    Returns
    -------
    float
        `c227`.
    """
    return c2271 + c2272 + c2273 + c2274


def calculate_fusion_power_island_cost(
    c221, c222, c223, c224, c225, c226, c227, c228, c229
):
    """Account 22 (total): fusion power island. Ports `Costs.acc22`'s own accumulation
    (`costs.py:917-933`) -- the sub-account calls above it are separate nodes.

    Parameters
    ----------
    c221, c222, c223, c224, c225, c226, c227, c228, c229 :
        Reactor, magnets, power injection, vacuum system, power conditioning, heat
        transport, fuel handling, instrumentation and control, and maintenance
        equipment costs (M$). `.costs.c221` .. `.costs.c229`.

    Returns
    -------
    tuple
        `(crctcore, c22)`.
    """
    crctcore = c221 + c222 + c223
    c22 = c221 + c222 + c223 + c224 + c225 + c226 + c227 + c228 + c229
    return crctcore, c22


def calculate_total_plant_direct_cost(c21, c22, c23, c24, c25, c26):
    """Total plant direct cost. Ports the inline accumulation in `Costs.run`
    (`costs.py:64-71`) -- not a method of its own in PROCESS, but a real owned field.

    Parameters
    ----------
    c21, c22, c23, c24, c25, c26 :
        Structures/site, fusion power island, turbine plant, electric plant,
        miscellaneous plant and heat rejection costs (M$). `.costs.c21` ..
        `.costs.c26`.

    Returns
    -------
    float
        `cdirt`.
    """
    return c21 + c22 + c23 + c24 + c25 + c26


def calculate_constructed_cost(cdirt, cindrt, ccont):
    """Constructed cost. Ports the inline accumulation in `Costs.run`
    (`costs.py:77-79`).

    `.costs.concost` is one of exactly two fields both `costs.py` and `costs_2015.py`
    write (the other being `.costs.coe`) -- see `costs.md`'s `i_cost_model` finding.

    Parameters
    ----------
    cdirt :
        Total plant direct cost (M$). `.costs.cdirt`.
    cindrt, ccont :
        Indirect cost and project contingency (M$). `.costs.cindrt`, `.costs.ccont`.

    Returns
    -------
    float
        `concost`.
    """
    return cdirt + cindrt + ccont


def calculate_cost_of_electricity(
    ife,
    itart,
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
):
    """Cost of electricity. Ports `Costs.coelc` (`costs.py:2703-2996`),
    magnetic-confinement arm only. **This is the function `.costs.coe` -- the objective
    of `i_figure_merit == 6` -- comes from.**

    PROCESS calls this only when `.costs.ireactor == 1 and .costs.ipnet == 0`
    (`costs.py:82-83`); both are run-configuration constants and both hold by PROCESS's
    own defaults (`cost_variables.py:521`, `:515`) and on the reference run, so the node
    wrapping this function carries them as static kwargs that refuse any other value
    rather than silently producing a `.costs.coe` PROCESS would have left untouched.

    `ife` static: the IFE arm drops the burn-fraction factor from `kwhpy`
    (`costs.py:2713-2720`), zeroes the divertor replacement cost, and computes the fuel
    cost from `.ife.uctarg`/`.reprat` (`costs.py:2916-2923`) -- a different reads-set
    over an unported subsystem.

    `itart` is static but **both** arms are implemented, so this node reads the union
    (`.costs.cpstcst`/`cplife_cal`/`cplife` are read even when `itart == 0`). That is a
    deliberate deviation from `traceability_policy.md`'s split default, on the
    size-aware grounds `_audit/next_steps.md` §1 is already tracking: the centrepost
    term is 15 lines of a ~290-line function whose other ~275 lines are shared, and
    splitting would duplicate all of them. Contrast
    `calculate_tf_magnet_cost_superconducting`/`_resistive` in this same file, which
    *were* split because they share no body at all.

    `.costs.coefuelt`'s six components (`coefwbl`, `coediv`, `coecdr`, `coecp`,
    `coefuel`, `coewst`) are Python locals in PROCESS, not `DataStructure` fields
    (`costs.py:2782`, `2809`, `2872`, `2839`, `2926`, `2938`), so they are not outputs
    here either -- only their sum is.

    Parameters
    ----------
    ife :
        1 if this is an inertial-fusion device, else 0. `.ife.ife`. Static.
    itart :
        1 if this is a spherical tokamak with a replaceable centrepost, else 0.
        `.physics.itart`. Static.
    p_plant_electric_net_mw :
        Net electric power (MW). `.heat_transport.p_plant_electric_net_mw`.
    f_t_plant_available :
        Plant availability fraction. `.costs.f_t_plant_available`.
    t_plant_pulse_burn, t_plant_pulse_total :
        Burn time and total pulse time (s). `.times.t_plant_pulse_burn`,
        `.times.t_plant_pulse_total`.
    concost :
        Constructed cost (M$). `.costs.concost`.
    fcap0, fcap0cp :
        Capital cost multipliers for the whole plant and for replaceable components.
        `.costs.fcap0`, `.costs.fcap0cp`.
    fcr0 :
        Fixed charge rate for the capital cost of the first plant.  `.costs.fcr0`.
    discount_rate :
        Effective cost of money (real discount rate). `.costs.discount_rate`.
    life_blkt :
        FW/blanket lifetime in calendar years. `.fwbs.life_blkt`.
    fwallcst, blkcst, divcst :
        First wall, blanket and divertor direct costs treated as fuel-like (M$).
        `.costs.fwallcst`, `.costs.blkcst`, `.costs.divcst`.
    cfind :
        Indirect cost factor by level of safety assurance, length-4. `.costs.cfind`.
    lsa :
        Level of safety assurance, 1-4. `.costs.lsa`.
    ifueltyp :
        Fuel-cost treatment switch; `== 2` prorates each replacement annuity by the
        fraction of plant life the component survives. `.costs.ifueltyp`.
    life_blkt_fpy, life_div_fpy :
        FW/blanket and divertor lifetimes in full-power-years. `.fwbs.life_blkt_fpy`,
        `.costs.life_div_fpy`.
    life_plant :
        Plant lifetime in calendar years. `.costs.life_plant`.
    life_div :
        Divertor lifetime in calendar years. `.costs.life_div`.
    cplife_cal, cplife :
        Centrepost lifetime in calendar years and in full-power-years.
        `.costs.cplife_cal`, `.costs.cplife`.
    cpstcst :
        Centrepost direct cost (M$). `.costs.cpstcst`.
    cdrlife_cal :
        Current drive system lifetime in calendar years. `.costs.cdrlife_cal`.
    cdcost, fcdfuel :
        Current drive system cost (M$) and the fraction of it treated as fuel.
        `.costs.cdcost`, `.costs.fcdfuel`.
    ucoam, ucwst :
        Annual operation-and-maintenance and waste disposal costs by level of safety
        assurance, length-4 (M$/yr). `.costs.ucoam`, `.costs.ucwst`.
    ucfuel :
        D-T fuel cost (M$/yr at 1.2 GW). `.costs.ucfuel`.
    f_plasma_fuel_helium3 :
        He3 fraction of the fuel. `.physics.f_plasma_fuel_helium3`.
    wtgpd :
        Fuel throughput (g/day). `.physics.wtgpd`.
    uche3 :
        He3 unit cost ($/kg). `.costs.uche3`.
    decomf :
        Fraction of the construction cost set aside for decommissioning.
        `.costs.decomf`.
    dintrt :
        Difference between borrowing and saving interest rates. `.costs.dintrt`.
    dtlife :
        Years before the end of plant life at which the decommissioning fund is
        complete. `.costs.dtlife`.

    Returns
    -------
    tuple
        `(moneyint, capcost, coecap, coeoam, coefuelt, coe)`.
    """
    if ife == 1:
        raise NotImplementedError(
            "coelc's ife == 1 arm changes kwhpy, zeroes the divertor replacement cost "
            "and reads .ife.uctarg/.reprat for the fuel cost -- a different reads-set "
            "over an unported subsystem. Only ife != 1 is ported."
        )

    kwhpy = (
        1.0e3
        * p_plant_electric_net_mw
        * (24.0e0 * _N_DAY_YEAR)
        * f_t_plant_available
        * t_plant_pulse_burn
        / t_plant_pulse_total
    )

    #  Costs due to reactor plant.
    moneyint = concost * (fcap0 - 1.0e0)
    capcost = concost + moneyint
    anncap = capcost * fcr0

    #  SJP Issue #836: guard against kwhpy == 0.
    kwhpy = jnp.maximum(kwhpy, 1.0e-10)
    coecap = 1.0e9 * anncap / kwhpy

    cfind_lsa = jnp.asarray(cfind)[lsa - 1]
    prorate = ifueltyp == 2

    #  Costs due to first wall and blanket renewal.
    feffwbl = (1.0e0 + discount_rate) ** life_blkt
    crffwbl = (feffwbl * discount_rate) / (feffwbl - 1.0e0)
    annfwbl = (fwallcst + blkcst) * (1.0e0 + cfind_lsa) * fcap0cp * crffwbl
    annfwbl = jnp.where(prorate, annfwbl * (1.0e0 - life_blkt_fpy / life_plant), annfwbl)
    coefwbl = 1.0e9 * annfwbl / kwhpy

    #  Costs due to divertor renewal.
    fefdiv = (1.0e0 + discount_rate) ** life_div
    crfdiv = (fefdiv * discount_rate) / (fefdiv - 1.0e0)
    anndiv = divcst * (1.0e0 + cfind_lsa) * fcap0cp * crfdiv
    anndiv = jnp.where(prorate, anndiv * (1.0e0 - life_div_fpy / life_plant), anndiv)
    coediv = 1.0e9 * anndiv / kwhpy

    #  Costs due to centrepost renewal.
    if itart == 1:
        fefcp = (1.0e0 + discount_rate) ** cplife_cal
        crfcp = (fefcp * discount_rate) / (fefcp - 1.0e0)
        anncp = cpstcst * (1.0e0 + cfind_lsa) * fcap0cp * crfcp
        anncp = jnp.where(prorate, anncp * (1.0e0 - cplife / life_plant), anncp)
        coecp = 1.0e9 * anncp / kwhpy
    else:
        coecp = 0.0e0

    #  Costs due to partial current drive system renewal.
    fefcdr = (1.0e0 + discount_rate) ** cdrlife_cal
    crfcdr = (fefcdr * discount_rate) / (fefcdr - 1.0e0)
    anncdr = jnp.where(
        ifueltyp == 0,
        0.0e0,
        cdcost * fcdfuel / (1.0e0 - fcdfuel) * (1.0e0 + cfind_lsa) * fcap0cp * crfcdr,
    )
    coecdr = 1.0e9 * anncdr / kwhpy

    #  Costs due to operation and maintenance. PROCESS clamps a negative net electric
    #  power to zero here rather than taking the square root of a negative number
    #  (`costs.py:2874-2888`). Written as a *double* `jnp.where` rather than the obvious
    #  `jnp.sqrt(jnp.maximum(p, 0.0))`: `sqrt` has an infinite derivative at zero, so
    #  the single-clamp form is value-correct but returns `nan` from `jacfwd` on the
    #  clamped branch -- caught by `test_gradient_finite`, which exists for exactly this
    #  class of leak. The inner `where` keeps the untaken branch's argument away from
    #  zero; the outer one discards its value. PROCESS's own finite difference here is
    #  identically zero (a further negative perturbation stays clamped), which is what
    #  this form differentiates to.
    is_negative = p_plant_electric_net_mw < 0.0
    sqrt_p_plant_electric_net_mw_1200 = jnp.where(
        is_negative,
        0.0,
        safe_sqrt(jnp.where(is_negative, 1.0, p_plant_electric_net_mw) / 1200.0e0),
    )
    annoam = jnp.asarray(ucoam)[lsa - 1] * sqrt_p_plant_electric_net_mw_1200
    coeoam = 1.0e9 * annoam / kwhpy

    #  Costs due to reactor fuel: D-T fuel cost plus He3 fuel cost.
    annfuel = (
        ucfuel * p_plant_electric_net_mw / 1200.0e0
        + 1.0e-6
        * f_plasma_fuel_helium3
        * wtgpd
        * 1.0e-3
        * uche3
        * _N_DAY_YEAR
        * f_t_plant_available
    )
    coefuel = 1.0e9 * annfuel / kwhpy

    #  Costs due to waste disposal.
    annwst = jnp.asarray(ucwst)[lsa - 1] * sqrt_p_plant_electric_net_mw_1200
    coewst = 1.0e9 * annwst / kwhpy

    #  Costs due to the decommissioning fund.
    anndecom = (
        decomf
        * concost
        * fcr0
        / (1.0e0 + discount_rate - dintrt) ** (life_plant - dtlife)
    )
    coedecom = 1.0e9 * anndecom / kwhpy

    coefuelt = coefwbl + coediv + coecdr + coecp + coefuel + coewst
    coe = coecap + coefuelt + coeoam + coedecom
    return moneyint, capcost, coecap, coeoam, coefuelt, coe


class ConvertFpyToCalendar(ExplicitFunction):
    """cottax node: `convert_fpy_to_calendar`."""

    life_blkt = Output(lambda s: s.fwbs.life_blkt)
    cdrlife_cal = Output(lambda s: s.costs.cdrlife_cal)
    life_div = Output(lambda s: s.costs.life_div)
    cplife_cal = Output(lambda s: s.costs.cplife_cal)

    def __call__(
        self,
        life_blkt_fpy=FromExactly(lambda s: s.fwbs.life_blkt_fpy),
        life_plant=FromExactly(lambda s: s.costs.life_plant),
        f_t_plant_available=FromExactly(lambda s: s.costs.f_t_plant_available),
        life_div_fpy=FromExactly(lambda s: s.costs.life_div_fpy),
        itart=FromExactly(lambda s: s.physics.itart),
        cplife=FromExactly(lambda s: s.costs.cplife),
    ):
        return convert_fpy_to_calendar(
            life_blkt_fpy, life_plant, f_t_plant_available, life_div_fpy, itart, cplife
        )


class StructuresCost(ExplicitFunction):
    """cottax node: `calculate_structures_cost` (Account 21)."""

    c211 = Output(lambda s: s.costs.c211)
    c212 = Output(lambda s: s.costs.c212)
    c213 = Output(lambda s: s.costs.c213)
    c2141 = Output(lambda s: s.costs.c2141)
    c2142 = Output(lambda s: s.costs.c2142)
    c214 = Output(lambda s: s.costs.c214)
    c215 = Output(lambda s: s.costs.c215)
    c216 = Output(lambda s: s.costs.c216)
    c2171 = Output(lambda s: s.costs.c2171)
    c2172 = Output(lambda s: s.costs.c2172)
    c2173 = Output(lambda s: s.costs.c2173)
    c2174 = Output(lambda s: s.costs.c2174)
    c217 = Output(lambda s: s.costs.c217)
    c21 = Output(lambda s: s.costs.c21)

    def __call__(
        self,
        csi=FromExactly(lambda s: s.costs.csi),
        lsa=FromExactly(lambda s: s.costs.lsa),
        cland=FromExactly(lambda s: s.costs.cland),
        ucrb=FromExactly(lambda s: s.costs.ucrb),
        rbvol=FromExactly(lambda s: s.buildings.rbvol),
        UCMB=FromExactly(lambda s: s.costs.UCMB),
        rmbvol=FromExactly(lambda s: s.buildings.rmbvol),
        UCWS=FromExactly(lambda s: s.costs.UCWS),
        wsvol=FromExactly(lambda s: s.buildings.wsvol),
        UCTR=FromExactly(lambda s: s.costs.UCTR),
        triv=FromExactly(lambda s: s.buildings.triv),
        UCEL=FromExactly(lambda s: s.costs.UCEL),
        elevol=FromExactly(lambda s: s.buildings.elevol),
        UCAD=FromExactly(lambda s: s.costs.UCAD),
        admvol=FromExactly(lambda s: s.buildings.admvol),
        UCCO=FromExactly(lambda s: s.costs.UCCO),
        convol=FromExactly(lambda s: s.buildings.convol),
        UCSH=FromExactly(lambda s: s.costs.UCSH),
        shovol=FromExactly(lambda s: s.buildings.shovol),
        UCCR=FromExactly(lambda s: s.costs.UCCR),
        cryvol=FromExactly(lambda s: s.buildings.cryvol),
        ireactor=FromExactly(lambda s: s.costs.ireactor),
        cturbb=FromExactly(lambda s: s.costs.cturbb),
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

    cindrt = Output(lambda s: s.costs.cindrt)
    ccont = Output(lambda s: s.costs.ccont)

    def __call__(
        self,
        cfind=FromExactly(lambda s: s.costs.cfind),
        lsa=FromExactly(lambda s: s.costs.lsa),
        cdirt=FromExactly(lambda s: s.costs.cdirt),
        cowner=FromExactly(lambda s: s.costs.cowner),
        fcontng=FromExactly(lambda s: s.costs.fcontng),
    ):
        return calculate_indirect_costs(cfind, lsa, cdirt, cowner, fcontng)


class ReactorStructureCost(ExplicitFunction):
    """cottax node: `calculate_reactor_structure_cost` (Account 221.4)."""

    c2214 = Output(lambda s: s.costs.c2214)

    def __call__(
        self,
        gsmass=FromExactly(lambda s: s.structure.gsmass),
        UCGSS=FromExactly(lambda s: s.costs.UCGSS),
        lsa=FromExactly(lambda s: s.costs.lsa),
        fkind=FromExactly(lambda s: s.costs.fkind),
    ):
        return calculate_reactor_structure_cost(gsmass, UCGSS, lsa, fkind)


class VacuumVesselAssemblyCost(ExplicitFunction):
    """cottax node: `calculate_vacuum_vessel_assembly_cost` (Account 222.3)."""

    c2223 = Output(lambda s: s.costs.c2223)

    def __call__(
        self,
        m_vv=FromExactly(lambda s: s.fwbs.m_vv),
        uccryo=FromExactly(lambda s: s.costs.uccryo),
        lsa=FromExactly(lambda s: s.costs.lsa),
        fkind=FromExactly(lambda s: s.costs.fkind),
    ):
        return calculate_vacuum_vessel_assembly_cost(m_vv, uccryo, lsa, fkind)


class DivertorCost(ExplicitFunction):
    """cottax node: `calculate_divertor_cost` (Account 221.5)."""

    c2215 = Output(lambda s: s.costs.c2215)
    divcst = Output(lambda s: s.costs.divcst)

    def __call__(
        self,
        ife=FromExactly(lambda s: s.ife.ife),
        a_div_surface_total=FromExactly(lambda s: s.divertor.a_div_surface_total),
        ucdiv=FromExactly(lambda s: s.costs.ucdiv),
        fkind=FromExactly(lambda s: s.costs.fkind),
        ifueltyp=FromExactly(lambda s: s.costs.ifueltyp),
    ):
        return calculate_divertor_cost(ife, a_div_surface_total, ucdiv, fkind, ifueltyp)


class VacuumSystemCost(ExplicitFunction):
    """cottax node: `calculate_vacuum_system_cost` (Account 224)."""

    c2241 = Output(lambda s: s.costs.c2241)
    c2242 = Output(lambda s: s.costs.c2242)
    c2243 = Output(lambda s: s.costs.c2243)
    c2244 = Output(lambda s: s.costs.c2244)
    c2245 = Output(lambda s: s.costs.c2245)
    c2246 = Output(lambda s: s.costs.c2246)
    c224 = Output(lambda s: s.costs.c224)

    def __call__(
        self,
        i_vacuum_pump_type=FromExactly(lambda s: s.vacuum.i_vacuum_pump_type),
        n_vac_pumps_high=FromExactly(lambda s: s.vacuum.n_vac_pumps_high),
        UCCPMP=FromExactly(lambda s: s.costs.UCCPMP),
        UCTPMP=FromExactly(lambda s: s.costs.UCTPMP),
        n_vv_vacuum_ducts=FromExactly(lambda s: s.vacuum.n_vv_vacuum_ducts),
        UCBPMP=FromExactly(lambda s: s.costs.UCBPMP),
        dlscal=FromExactly(lambda s: s.vacuum.dlscal),
        UCDUCT=FromExactly(lambda s: s.costs.UCDUCT),
        dia_vv_vacuum_ducts=FromExactly(lambda s: s.vacuum.dia_vv_vacuum_ducts),
        UCVALV=FromExactly(lambda s: s.costs.UCVALV),
        m_vv_vacuum_duct_shield=FromExactly(lambda s: s.vacuum.m_vv_vacuum_duct_shield),
        UCVDSH=FromExactly(lambda s: s.costs.UCVDSH),
        UCVIAC=FromExactly(lambda s: s.costs.UCVIAC),
        fkind=FromExactly(lambda s: s.costs.fkind),
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

    c22511 = Output(lambda s: s.costs.c22511)
    c22512 = Output(lambda s: s.costs.c22512)
    c22513 = Output(lambda s: s.costs.c22513)
    c22514 = Output(lambda s: s.costs.c22514)
    c22515 = Output(lambda s: s.costs.c22515)
    c2251 = Output(lambda s: s.costs.c2251)

    def __call__(
        self,
        uctfps=FromExactly(lambda s: s.costs.uctfps),
        tfckw=FromExactly(lambda s: s.tfcoil.tfckw),
        tfcmw=FromExactly(lambda s: s.tfcoil.tfcmw),
        i_tf_sup=FromExactly(lambda s: s.tfcoil.i_tf_sup),
        uctfbr=FromExactly(lambda s: s.costs.uctfbr),
        n_tf_coils=FromExactly(lambda s: s.tfcoil.n_tf_coils),
        c_tf_turn=FromExactly(lambda s: s.tfcoil.c_tf_turn),
        v_tf_coil_dump_quench_kv=FromExactly(lambda s: s.tfcoil.v_tf_coil_dump_quench_kv),
        uctfsw=FromExactly(lambda s: s.costs.uctfsw),
        UCTFDR=FromExactly(lambda s: s.costs.UCTFDR),
        e_tf_magnetic_stored_total_gj=FromExactly(
            lambda s: s.tfcoil.e_tf_magnetic_stored_total_gj
        ),
        UCTFGR=FromExactly(lambda s: s.costs.UCTFGR),
        UCTFIC=FromExactly(lambda s: s.costs.UCTFIC),
        uctfbus=FromExactly(lambda s: s.costs.uctfbus),
        m_tf_bus=FromExactly(lambda s: s.tfcoil.m_tf_bus),
        ucbus=FromExactly(lambda s: s.costs.ucbus),
        len_tf_bus=FromExactly(lambda s: s.tfcoil.len_tf_bus),
        fkind=FromExactly(lambda s: s.costs.fkind),
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

    c22521 = Output(lambda s: s.costs.c22521)
    c22522 = Output(lambda s: s.costs.c22522)
    c22523 = Output(lambda s: s.costs.c22523)
    c22524 = Output(lambda s: s.costs.c22524)
    c22525 = Output(lambda s: s.costs.c22525)
    c22526 = Output(lambda s: s.costs.c22526)
    c22527 = Output(lambda s: s.costs.c22527)
    c2252 = Output(lambda s: s.costs.c2252)

    def __call__(
        self,
        ucpfps=FromExactly(lambda s: s.costs.ucpfps),
        peakmva=FromExactly(lambda s: s.heat_transport.peakmva),
        ucpfic=FromExactly(lambda s: s.costs.ucpfic),
        pfckts=FromExactly(lambda s: s.pf_power.pfckts),
        ucpfb=FromExactly(lambda s: s.costs.ucpfb),
        spfbusl=FromExactly(lambda s: s.pf_power.spfbusl),
        acptmax=FromExactly(lambda s: s.pf_power.acptmax),
        ucpfbs=FromExactly(lambda s: s.costs.ucpfbs),
        srcktpm=FromExactly(lambda s: s.pf_power.srcktpm),
        ucpfbk=FromExactly(lambda s: s.costs.ucpfbk),
        vpfskv=FromExactly(lambda s: s.pf_power.vpfskv),
        ucpfdr1=FromExactly(lambda s: s.costs.ucpfdr1),
        ensxpfm=FromExactly(lambda s: s.pf_power.ensxpfm),
        ucpfcb=FromExactly(lambda s: s.costs.ucpfcb),
        fkind=FromExactly(lambda s: s.costs.fkind),
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

    cpp = Output(lambda s: s.costs.cpp)
    chx = Output(lambda s: s.costs.chx)
    c2261 = Output(lambda s: s.costs.c2261)

    def __call__(
        self,
        uchts=FromExactly(lambda s: s.costs.uchts),
        i_blkt_coolant_type=FromExactly(lambda s: s.fwbs.i_blkt_coolant_type),
        p_fw_div_heat_deposited_mw=FromExactly(
            lambda s: s.heat_transport.p_fw_div_heat_deposited_mw
        ),
        p_blkt_nuclear_heat_total_mw=FromExactly(
            lambda s: s.fwbs.p_blkt_nuclear_heat_total_mw
        ),
        p_shld_nuclear_heat_mw=FromExactly(lambda s: s.fwbs.p_shld_nuclear_heat_mw),
        lsa=FromExactly(lambda s: s.costs.lsa),
        fkind=FromExactly(lambda s: s.costs.fkind),
        UCPHX=FromExactly(lambda s: s.costs.UCPHX),
        n_primary_heat_exchangers=FromExactly(
            lambda s: s.heat_transport.n_primary_heat_exchangers
        ),
        p_plant_primary_heat_mw=FromExactly(
            lambda s: s.heat_transport.p_plant_primary_heat_mw
        ),
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

    c2271 = Output(lambda s: s.costs.c2271)

    def __call__(
        self,
        ucf1=FromExactly(lambda s: s.costs.ucf1),
        fkind=FromExactly(lambda s: s.costs.fkind),
    ):
        return calculate_fuelling_system_cost(ucf1, fkind)


class NuclearBuildingVentilationCost(ExplicitFunction):
    """cottax node: `calculate_nuclear_building_ventilation_cost` (Account 2274)."""

    c2274 = Output(lambda s: s.costs.c2274)

    def __call__(
        self,
        UCNBV=FromExactly(lambda s: s.costs.UCNBV),
        volrci=FromExactly(lambda s: s.buildings.volrci),
        wsvol=FromExactly(lambda s: s.buildings.wsvol),
        fkind=FromExactly(lambda s: s.costs.fkind),
    ):
        return calculate_nuclear_building_ventilation_cost(UCNBV, volrci, wsvol, fkind)


class InstrumentationAndControlCost(ExplicitFunction):
    """cottax node: `calculate_instrumentation_and_control_cost` (Account 228)."""

    c228 = Output(lambda s: s.costs.c228)

    def __call__(
        self,
        uciac=FromExactly(lambda s: s.costs.uciac),
        fkind=FromExactly(lambda s: s.costs.fkind),
    ):
        return calculate_instrumentation_and_control_cost(uciac, fkind)


class MaintenanceEquipmentCost(ExplicitFunction):
    """cottax node: `calculate_maintenance_equipment_cost` (Account 229)."""

    c229 = Output(lambda s: s.costs.c229)

    def __call__(
        self,
        ucme=FromExactly(lambda s: s.costs.ucme),
        fkind=FromExactly(lambda s: s.costs.fkind),
    ):
        return calculate_maintenance_equipment_cost(ucme, fkind)


class TurbinePlantEquipmentCost(ExplicitFunction):
    """cottax node: `calculate_turbine_plant_equipment_cost` (Account 23)."""

    c23 = Output(lambda s: s.costs.c23)

    def __call__(
        self,
        ireactor=FromExactly(lambda s: s.costs.ireactor),
        ucturb=FromExactly(lambda s: s.costs.ucturb),
        i_blkt_coolant_type=FromExactly(lambda s: s.fwbs.i_blkt_coolant_type),
        p_plant_electric_gross_mw=FromExactly(
            lambda s: s.heat_transport.p_plant_electric_gross_mw
        ),
    ):
        return calculate_turbine_plant_equipment_cost(
            ireactor, ucturb, i_blkt_coolant_type, p_plant_electric_gross_mw
        )


class SwitchyardCost(ExplicitFunction):
    """cottax node: `calculate_switchyard_cost` (Account 241)."""

    c241 = Output(lambda s: s.costs.c241)

    def __call__(
        self,
        UCSWYD=FromExactly(lambda s: s.costs.UCSWYD),
        lsa=FromExactly(lambda s: s.costs.lsa),
    ):
        return calculate_switchyard_cost(UCSWYD, lsa)


class TransformersCost(ExplicitFunction):
    """cottax node: `calculate_transformers_cost` (Account 242)."""

    c242 = Output(lambda s: s.costs.c242)

    def __call__(
        self,
        UCPP=FromExactly(lambda s: s.costs.UCPP),
        pacpmw=FromExactly(lambda s: s.heat_transport.pacpmw),
        UCAP=FromExactly(lambda s: s.costs.UCAP),
        p_plant_electric_base_total_mw=FromExactly(
            lambda s: s.heat_transport.p_plant_electric_base_total_mw
        ),
        lsa=FromExactly(lambda s: s.costs.lsa),
    ):
        return calculate_transformers_cost(
            UCPP, pacpmw, UCAP, p_plant_electric_base_total_mw, lsa
        )


class LowVoltageCost(ExplicitFunction):
    """cottax node: `calculate_low_voltage_cost` (Account 243)."""

    c243 = Output(lambda s: s.costs.c243)

    def __call__(
        self,
        UCLV=FromExactly(lambda s: s.costs.UCLV),
        tlvpmw=FromExactly(lambda s: s.heat_transport.tlvpmw),
        lsa=FromExactly(lambda s: s.costs.lsa),
    ):
        return calculate_low_voltage_cost(UCLV, tlvpmw, lsa)


class DieselGeneratorsCost(ExplicitFunction):
    """cottax node: `calculate_diesel_generators_cost` (Account 244)."""

    c244 = Output(lambda s: s.costs.c244)

    def __call__(
        self,
        UCDGEN=FromExactly(lambda s: s.costs.UCDGEN),
        lsa=FromExactly(lambda s: s.costs.lsa),
    ):
        return calculate_diesel_generators_cost(UCDGEN, lsa)


class AuxiliaryFacilityPowerCost(ExplicitFunction):
    """cottax node: `calculate_auxiliary_facility_power_cost` (Account 245)."""

    c245 = Output(lambda s: s.costs.c245)

    def __call__(
        self,
        UCAF=FromExactly(lambda s: s.costs.UCAF),
        lsa=FromExactly(lambda s: s.costs.lsa),
    ):
        return calculate_auxiliary_facility_power_cost(UCAF, lsa)


class ElectricPlantEquipmentCost(ExplicitFunction):
    """cottax node: `calculate_electric_plant_equipment_cost` (Account 24, total).

    Reads the five sub-account nodes' own outputs -- an ordinary graph edge, matching
    `Costs.acc24`'s own call order (`acc241`..`acc245` before `acc24`).
    """

    c24 = Output(lambda s: s.costs.c24)

    def __call__(
        self,
        c241=FromExactly(lambda s: s.costs.c241),
        c242=FromExactly(lambda s: s.costs.c242),
        c243=FromExactly(lambda s: s.costs.c243),
        c244=FromExactly(lambda s: s.costs.c244),
        c245=FromExactly(lambda s: s.costs.c245),
    ):
        return calculate_electric_plant_equipment_cost(c241, c242, c243, c244, c245)


class MiscPlantEquipmentCost(ExplicitFunction):
    """cottax node: `calculate_misc_plant_equipment_cost` (Account 25)."""

    c25 = Output(lambda s: s.costs.c25)

    def __call__(
        self,
        ucmisc=FromExactly(lambda s: s.costs.ucmisc),
        lsa=FromExactly(lambda s: s.costs.lsa),
    ):
        return calculate_misc_plant_equipment_cost(ucmisc, lsa)


class HeatRejectionCost(ExplicitFunction):
    """cottax node: `calculate_heat_rejection_cost` (Account 26)."""

    c26 = Output(lambda s: s.costs.c26)

    def __call__(
        self,
        ireactor=FromExactly(lambda s: s.costs.ireactor),
        p_fusion_total_mw=FromExactly(lambda s: s.physics.p_fusion_total_mw),
        p_hcd_electric_total_mw=FromExactly(
            lambda s: s.heat_transport.p_hcd_electric_total_mw
        ),
        tfcmw=FromExactly(lambda s: s.tfcoil.tfcmw),
        p_plant_primary_heat_mw=FromExactly(
            lambda s: s.heat_transport.p_plant_primary_heat_mw
        ),
        p_plant_electric_gross_mw=FromExactly(
            lambda s: s.heat_transport.p_plant_electric_gross_mw
        ),
        uchrs=FromExactly(lambda s: s.costs.uchrs),
        lsa=FromExactly(lambda s: s.costs.lsa),
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

    ife: IFEModel = eqx.field(static=True)

    c2211 = Output(lambda s: s.costs.c2211)
    fwallcst = Output(lambda s: s.costs.fwallcst)

    def __call__(
        self,
        lsa=FromExactly(lambda s: s.costs.lsa),
        UCFWA=FromExactly(lambda s: s.costs.UCFWA),
        UCFWS=FromExactly(lambda s: s.costs.UCFWS),
        a_fw_total=FromExactly(lambda s: s.first_wall.a_fw_total),
        UCFWPS=FromExactly(lambda s: s.costs.UCFWPS),
        fkind=FromExactly(lambda s: s.costs.fkind),
        ifueltyp=FromExactly(lambda s: s.costs.ifueltyp),
    ):
        return calculate_first_wall_cost(
            self.ife, lsa, UCFWA, UCFWS, a_fw_total, UCFWPS, fkind, ifueltyp
        )


class BlanketCost(ExplicitFunction):
    """cottax node: `calculate_blanket_cost` (Account 221.2)."""

    ife: IFEModel = eqx.field(static=True)

    c22121 = Output(lambda s: s.costs.c22121)
    c22122 = Output(lambda s: s.costs.c22122)
    c22123 = Output(lambda s: s.costs.c22123)
    c22124 = Output(lambda s: s.costs.c22124)
    c22125 = Output(lambda s: s.costs.c22125)
    c22126 = Output(lambda s: s.costs.c22126)
    c22127 = Output(lambda s: s.costs.c22127)
    c2212 = Output(lambda s: s.costs.c2212)
    blkcst = Output(lambda s: s.costs.blkcst)

    def __call__(
        self,
        lsa=FromExactly(lambda s: s.costs.lsa),
        m_blkt_beryllium=FromExactly(lambda s: s.fwbs.m_blkt_beryllium),
        ucblbe=FromExactly(lambda s: s.costs.ucblbe),
        m_blkt_li2o=FromExactly(lambda s: s.fwbs.m_blkt_li2o),
        ucblli2o=FromExactly(lambda s: s.costs.ucblli2o),
        m_blkt_steel_total=FromExactly(lambda s: s.fwbs.m_blkt_steel_total),
        ucblss=FromExactly(lambda s: s.costs.ucblss),
        m_blkt_vanadium=FromExactly(lambda s: s.fwbs.m_blkt_vanadium),
        ucblvd=FromExactly(lambda s: s.costs.ucblvd),
        fkind=FromExactly(lambda s: s.costs.fkind),
        ifueltyp=FromExactly(lambda s: s.costs.ifueltyp),
    ):
        return calculate_blanket_cost(
            self.ife,
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

    ife: IFEModel = eqx.field(static=True)

    c22131 = Output(lambda s: s.costs.c22131)
    c22132 = Output(lambda s: s.costs.c22132)
    c2213 = Output(lambda s: s.costs.c2213)

    def __call__(
        self,
        lsa=FromExactly(lambda s: s.costs.lsa),
        whtshld=FromExactly(lambda s: s.fwbs.whtshld),
        ucshld=FromExactly(lambda s: s.costs.ucshld),
        wpenshld=FromExactly(lambda s: s.fwbs.wpenshld),
        ucpens=FromExactly(lambda s: s.costs.ucpens),
        fkind=FromExactly(lambda s: s.costs.fkind),
    ):
        return calculate_shield_cost(
            self.ife, lsa, whtshld, ucshld, wpenshld, ucpens, fkind
        )


class ReactorCost(ExplicitFunction):
    """cottax node: `calculate_reactor_cost` (Account 221 total)."""

    c221 = Output(lambda s: s.costs.c221)

    def __call__(
        self,
        c2211=FromExactly(lambda s: s.costs.c2211),
        c2212=FromExactly(lambda s: s.costs.c2212),
        c2213=FromExactly(lambda s: s.costs.c2213),
        c2214=FromExactly(lambda s: s.costs.c2214),
        c2215=FromExactly(lambda s: s.costs.c2215),
    ):
        return calculate_reactor_cost(c2211, c2212, c2213, c2214, c2215)


class TfMagnetCostSuperconducting(ExplicitFunction):
    """cottax node: `calculate_tf_magnet_cost_superconducting` (Account 222.1,
    `.tfcoil.i_tf_sup == 1`)."""

    supercond_cost_model: SuperconductorCostModel = eqx.field(static=True)

    c22211 = Output(lambda s: s.costs.c22211)
    c22212 = Output(lambda s: s.costs.c22212)
    c22213 = Output(lambda s: s.costs.c22213)
    c22214 = Output(lambda s: s.costs.c22214)
    c22215 = Output(lambda s: s.costs.c22215)
    c2221 = Output(lambda s: s.costs.c2221)

    def __call__(
        self,
        lsa=FromExactly(lambda s: s.costs.lsa),
        ucsc=FromExactly(lambda s: s.costs.ucsc),
        i_tf_sc_mat=FromExactly(lambda s: s.tfcoil.i_tf_sc_mat),
        m_tf_coil_superconductor=FromExactly(lambda s: s.tfcoil.m_tf_coil_superconductor),
        len_tf_coil=FromExactly(lambda s: s.tfcoil.len_tf_coil),
        n_tf_coil_turns=FromExactly(lambda s: s.tfcoil.n_tf_coil_turns),
        sc_mat_cost_0=FromExactly(lambda s: s.costs.sc_mat_cost_0),
        j_crit_str_0=FromExactly(lambda s: s.tfcoil.j_crit_str_0),
        j_crit_str_tf=FromExactly(lambda s: s.tfcoil.j_crit_str_tf),
        uccu=FromExactly(lambda s: s.costs.uccu),
        m_tf_coil_copper=FromExactly(lambda s: s.tfcoil.m_tf_coil_copper),
        cconshtf=FromExactly(lambda s: s.costs.cconshtf),
        cconfix=FromExactly(lambda s: s.costs.cconfix),
        n_tf_coils=FromExactly(lambda s: s.tfcoil.n_tf_coils),
        ucwindtf=FromExactly(lambda s: s.costs.ucwindtf),
        m_tf_coil_case=FromExactly(lambda s: s.tfcoil.m_tf_coil_case),
        uccase=FromExactly(lambda s: s.costs.uccase),
        aintmass=FromExactly(lambda s: s.structure.aintmass),
        UCINT=FromExactly(lambda s: s.costs.UCINT),
        clgsmass=FromExactly(lambda s: s.structure.clgsmass),
        UCGSS=FromExactly(lambda s: s.costs.UCGSS),
        fkind=FromExactly(lambda s: s.costs.fkind),
    ):
        return calculate_tf_magnet_cost_superconducting(
            self.supercond_cost_model,
            lsa,
            ucsc,
            i_tf_sc_mat,
            m_tf_coil_superconductor,
            len_tf_coil,
            n_tf_coil_turns,
            sc_mat_cost_0,
            j_crit_str_0,
            j_crit_str_tf,
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
    docstring and `total_process.py`'s `.costs.i_cost_model` switch comment."""

    c22211 = Output(lambda s: s.costs.c22211)
    c22212 = Output(lambda s: s.costs.c22212)
    c2221 = Output(lambda s: s.costs.c2221)
    cpstcst = Output(lambda s: s.costs.cpstcst)

    def __call__(
        self,
        lsa=FromExactly(lambda s: s.costs.lsa),
        whtcp=FromExactly(lambda s: s.tfcoil.whtcp),
        uccpcl1=FromExactly(lambda s: s.costs.uccpcl1),
        whttflgs=FromExactly(lambda s: s.tfcoil.whttflgs),
        uccpclb=FromExactly(lambda s: s.costs.uccpclb),
        itart=FromExactly(lambda s: s.physics.itart),
        ifueltyp=FromExactly(lambda s: s.costs.ifueltyp),
        fkind=FromExactly(lambda s: s.costs.fkind),
    ):
        return calculate_tf_magnet_cost_resistive(
            lsa, whtcp, uccpcl1, whttflgs, uccpclb, itart, ifueltyp, fkind
        )


class PfMagnetCost(ExplicitFunction):
    """cottax node: `calculate_pf_magnet_cost` (Account 222.2)."""

    n_cs_pf_coils: int = eqx.field(static=True)
    iohcl: CentralSolenoidConfiguration = eqx.field(static=True)
    i_pf_conductor: PFConductorModel = eqx.field(static=True)
    supercond_cost_model: SuperconductorCostModel = eqx.field(static=True)

    c22221 = Output(lambda s: s.costs.c22221)
    c22222 = Output(lambda s: s.costs.c22222)
    c22223 = Output(lambda s: s.costs.c22223)
    c22224 = Output(lambda s: s.costs.c22224)
    c2222 = Output(lambda s: s.costs.c2222)

    def __call__(
        self,
        lsa=FromExactly(lambda s: s.costs.lsa),
        r_pf_coil_middle=FromExactly(lambda s: s.pf_coil.r_pf_coil_middle),
        n_pf_coil_turns=FromExactly(lambda s: s.pf_coil.n_pf_coil_turns),
        cconshpf=FromExactly(lambda s: s.costs.cconshpf),
        ucsc=FromExactly(lambda s: s.costs.ucsc),
        i_pf_superconductor=FromExactly(lambda s: s.pf_coil.i_pf_superconductor),
        fcupfsu=FromExactly(lambda s: s.pf_coil.fcupfsu),
        f_a_pf_coil_void=FromExactly(lambda s: s.pf_coil.f_a_pf_coil_void),
        c_pf_cs_coils_peak_ma=FromExactly(lambda s: s.pf_coil.c_pf_cs_coils_peak_ma),
        j_pf_coil_wp_peak=FromExactly(lambda s: s.pf_coil.j_pf_coil_wp_peak),
        dcond=FromExactly(lambda s: s.tfcoil.dcond),
        sc_mat_cost_0=FromExactly(lambda s: s.costs.sc_mat_cost_0),
        j_crit_str_0=FromExactly(lambda s: s.tfcoil.j_crit_str_0),
        j_crit_str_pf=FromExactly(lambda s: s.pf_coil.j_crit_str_pf),
        uccu=FromExactly(lambda s: s.costs.uccu),
        cconfix=FromExactly(lambda s: s.costs.cconfix),
        i_cs_superconductor=FromExactly(lambda s: s.pf_coil.i_cs_superconductor),
        a_cs_cable_space=FromExactly(lambda s: s.pf_coil.a_cs_cable_space),
        f_a_cs_void=FromExactly(lambda s: s.pf_coil.f_a_cs_void),
        fcuohsu=FromExactly(lambda s: s.pf_coil.fcuohsu),
        j_crit_str_cs=FromExactly(lambda s: s.pf_coil.j_crit_str_cs),
        ucwindpf=FromExactly(lambda s: s.costs.ucwindpf),
        uccase=FromExactly(lambda s: s.costs.uccase),
        m_pf_coil_structure_total=FromExactly(lambda s: s.pf_coil.m_pf_coil_structure_total),
        ucfnc=FromExactly(lambda s: s.costs.ucfnc),
        fncmass=FromExactly(lambda s: s.structure.fncmass),
        fkind=FromExactly(lambda s: s.costs.fkind),
    ):
        return calculate_pf_magnet_cost(
            self.n_cs_pf_coils,
            self.iohcl,
            self.i_pf_conductor,
            self.supercond_cost_model,
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
            dcond,
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


class MagnetsCost(ExplicitFunction):
    """cottax node: `calculate_magnets_cost` (Account 222 total)."""

    c222 = Output(lambda s: s.costs.c222)

    def __call__(
        self,
        ife=FromExactly(lambda s: s.ife.ife),
        c2221=FromExactly(lambda s: s.costs.c2221),
        c2222=FromExactly(lambda s: s.costs.c2222),
        c2223=FromExactly(lambda s: s.costs.c2223),
    ):
        return calculate_magnets_cost(ife, c2221, c2222, c2223)


class PowerInjectionCost(ExplicitFunction):
    """cottax node: `calculate_power_injection_cost` (Account 223)."""

    ife: IFEModel = eqx.field(static=True)

    c2231 = Output(lambda s: s.costs.c2231)
    c2232 = Output(lambda s: s.costs.c2232)
    c2233 = Output(lambda s: s.costs.c2233)
    c223 = Output(lambda s: s.costs.c223)
    cdcost = Output(lambda s: s.costs.cdcost)

    def __call__(
        self,
        ucech=FromExactly(lambda s: s.costs.ucech),
        p_hcd_ecrh_injected_total_mw=FromExactly(
            lambda s: s.current_drive.p_hcd_ecrh_injected_total_mw
        ),
        i_hcd_primary=FromExactly(lambda s: s.current_drive.i_hcd_primary),
        uclh=FromExactly(lambda s: s.costs.uclh),
        ucich=FromExactly(lambda s: s.costs.ucich),
        p_hcd_lowhyb_injected_total_mw=FromExactly(
            lambda s: s.current_drive.p_hcd_lowhyb_injected_total_mw
        ),
        ucnbi=FromExactly(lambda s: s.costs.ucnbi),
        p_beam_injected_mw=FromExactly(lambda s: s.current_drive.p_beam_injected_mw),
        ifueltyp=FromExactly(lambda s: s.costs.ifueltyp),
        fcdfuel=FromExactly(lambda s: s.costs.fcdfuel),
        fkind=FromExactly(lambda s: s.costs.fkind),
    ):
        return calculate_power_injection_cost(
            self.ife,
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
    """cottax node: `calculate_energy_storage_cost` (Account 225.3)."""

    i_pulsed_plant: PlantOperationModel = eqx.field(static=True)
    istore: ThermalStorageModel = eqx.field(static=True)

    c2253 = Output(lambda s: s.costs.c2253)

    def __call__(
        self,
        p_plant_electric_net_mw=FromExactly(
            lambda s: s.heat_transport.p_plant_electric_net_mw
        ),
        fkind=FromExactly(lambda s: s.costs.fkind),
    ):
        return calculate_energy_storage_cost(
            self.i_pulsed_plant, self.istore, p_plant_electric_net_mw, fkind
        )


class PowerConditioningCost(ExplicitFunction):
    """cottax node: `calculate_power_conditioning_cost` (Account 225 total)."""

    c225 = Output(lambda s: s.costs.c225)

    def __call__(
        self,
        ife=FromExactly(lambda s: s.ife.ife),
        c2251=FromExactly(lambda s: s.costs.c2251),
        c2252=FromExactly(lambda s: s.costs.c2252),
        c2253=FromExactly(lambda s: s.costs.c2253),
    ):
        return calculate_power_conditioning_cost(ife, c2251, c2252, c2253)


class AuxiliaryComponentCoolingCost(ExplicitFunction):
    """cottax node: `calculate_auxiliary_component_cooling_cost` (Account 2262)."""

    ife: IFEModel = eqx.field(static=True)

    cppa = Output(lambda s: s.costs.cppa)
    c2262 = Output(lambda s: s.costs.c2262)

    def __call__(
        self,
        lsa=FromExactly(lambda s: s.costs.lsa),
        UCAHTS=FromExactly(lambda s: s.costs.UCAHTS),
        p_hcd_electric_loss_mw=FromExactly(lambda s: s.heat_transport.p_hcd_electric_loss_mw),
        p_cryo_plant_electric_mw=FromExactly(
            lambda s: s.heat_transport.p_cryo_plant_electric_mw
        ),
        vachtmw=FromExactly(lambda s: s.heat_transport.vachtmw),
        p_tritium_plant_electric_mw=FromExactly(
            lambda s: s.heat_transport.p_tritium_plant_electric_mw
        ),
        fachtmw=FromExactly(lambda s: s.heat_transport.fachtmw),
        fkind=FromExactly(lambda s: s.costs.fkind),
    ):
        return calculate_auxiliary_component_cooling_cost(
            self.ife,
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

    c2263 = Output(lambda s: s.costs.c2263)

    def __call__(
        self,
        lsa=FromExactly(lambda s: s.costs.lsa),
        uccry=FromExactly(lambda s: s.costs.uccry),
        temp_tf_cryo=FromExactly(lambda s: s.tfcoil.temp_tf_cryo),
        helpow=FromExactly(lambda s: s.heat_transport.helpow),
        fkind=FromExactly(lambda s: s.costs.fkind),
    ):
        return calculate_cryogenic_system_cost(lsa, uccry, temp_tf_cryo, helpow, fkind)


class HeatTransportSystemCost(ExplicitFunction):
    """cottax node: `calculate_heat_transport_system_cost` (Account 226 total)."""

    c226 = Output(lambda s: s.costs.c226)

    def __call__(
        self,
        c2261=FromExactly(lambda s: s.costs.c2261),
        c2262=FromExactly(lambda s: s.costs.c2262),
        c2263=FromExactly(lambda s: s.costs.c2263),
    ):
        return calculate_heat_transport_system_cost(c2261, c2262, c2263)


class FuelProcessingCost(ExplicitFunction):
    """cottax node: `calculate_fuel_processing_cost` (Account 2272). Sole producer of
    `.physics.wtgpd`."""

    ife: IFEModel = eqx.field(static=True)

    wtgpd = Output(lambda s: s.physics.wtgpd)
    c2272 = Output(lambda s: s.costs.c2272)

    def __call__(
        self,
        rndfuel=FromExactly(lambda s: s.physics.rndfuel),
        m_fuel_amu=FromExactly(lambda s: s.physics.m_fuel_amu),
        UCFPR=FromExactly(lambda s: s.costs.UCFPR),
        fkind=FromExactly(lambda s: s.costs.fkind),
    ):
        return calculate_fuel_processing_cost(
            self.ife, rndfuel, m_fuel_amu, UCFPR, fkind
        )


class AtmosphericRecoveryCost(ExplicitFunction):
    """cottax node: `calculate_atmospheric_recovery_cost` (Account 2273)."""

    c2273 = Output(lambda s: s.costs.c2273)

    def __call__(
        self,
        f_plasma_fuel_tritium=FromExactly(lambda s: s.physics.f_plasma_fuel_tritium),
        UCDTC=FromExactly(lambda s: s.costs.UCDTC),
        volrci=FromExactly(lambda s: s.buildings.volrci),
        wsvol=FromExactly(lambda s: s.buildings.wsvol),
        fkind=FromExactly(lambda s: s.costs.fkind),
    ):
        return calculate_atmospheric_recovery_cost(
            f_plasma_fuel_tritium, UCDTC, volrci, wsvol, fkind
        )


class FuelHandlingCost(ExplicitFunction):
    """cottax node: `calculate_fuel_handling_cost` (Account 227 total)."""

    c227 = Output(lambda s: s.costs.c227)

    def __call__(
        self,
        c2271=FromExactly(lambda s: s.costs.c2271),
        c2272=FromExactly(lambda s: s.costs.c2272),
        c2273=FromExactly(lambda s: s.costs.c2273),
        c2274=FromExactly(lambda s: s.costs.c2274),
    ):
        return calculate_fuel_handling_cost(c2271, c2272, c2273, c2274)


class FusionPowerIslandCost(ExplicitFunction):
    """cottax node: `calculate_fusion_power_island_cost` (Account 22 total)."""

    crctcore = Output(lambda s: s.costs.crctcore)
    c22 = Output(lambda s: s.costs.c22)

    def __call__(
        self,
        c221=FromExactly(lambda s: s.costs.c221),
        c222=FromExactly(lambda s: s.costs.c222),
        c223=FromExactly(lambda s: s.costs.c223),
        c224=FromExactly(lambda s: s.costs.c224),
        c225=FromExactly(lambda s: s.costs.c225),
        c226=FromExactly(lambda s: s.costs.c226),
        c227=FromExactly(lambda s: s.costs.c227),
        c228=FromExactly(lambda s: s.costs.c228),
        c229=FromExactly(lambda s: s.costs.c229),
    ):
        return calculate_fusion_power_island_cost(
            c221, c222, c223, c224, c225, c226, c227, c228, c229
        )


class TotalPlantDirectCost(ExplicitFunction):
    """cottax node: `calculate_total_plant_direct_cost`."""

    cdirt = Output(lambda s: s.costs.cdirt)

    def __call__(
        self,
        c21=FromExactly(lambda s: s.costs.c21),
        c22=FromExactly(lambda s: s.costs.c22),
        c23=FromExactly(lambda s: s.costs.c23),
        c24=FromExactly(lambda s: s.costs.c24),
        c25=FromExactly(lambda s: s.costs.c25),
        c26=FromExactly(lambda s: s.costs.c26),
    ):
        return calculate_total_plant_direct_cost(c21, c22, c23, c24, c25, c26)


class ConstructedCost(ExplicitFunction):
    """cottax node: `calculate_constructed_cost`."""

    concost = Output(lambda s: s.costs.concost)

    def __call__(
        self,
        cdirt=FromExactly(lambda s: s.costs.cdirt),
        cindrt=FromExactly(lambda s: s.costs.cindrt),
        ccont=FromExactly(lambda s: s.costs.ccont),
    ):
        return calculate_constructed_cost(cdirt, cindrt, ccont)


class CostOfElectricity(ExplicitFunction):
    """cottax node: `calculate_cost_of_electricity` (`Costs.coelc`). Sole producer of
    `.costs.coe`, the `i_figure_merit == 6` objective.

    `ireactor`/`ipnet` are static preconditions, not ports: PROCESS calls `coelc()` only
    when `ireactor == 1 and ipnet == 0` (`process/models/costs/costs.py:82-83`), and
    both are run-configuration constants. Declaring them makes the graph say which
    precondition it is relying on, the same move `EcrhDensityLimit(i_plasma_pedestal=0)`
    makes, and `mda_harness.switch_audit` then checks them against the real run.
    """

    ife: IFEModel = eqx.field(static=True)
    itart: SphericalTokamakModel = eqx.field(static=True)
    ireactor: CostOfElectricityModel = eqx.field(static=True)
    ipnet: NetElectricPowerModel = eqx.field(static=True)

    moneyint = Output(lambda s: s.costs.moneyint)
    capcost = Output(lambda s: s.costs.capcost)
    coecap = Output(lambda s: s.costs.coecap)
    coeoam = Output(lambda s: s.costs.coeoam)
    coefuelt = Output(lambda s: s.costs.coefuelt)
    coe = Output(lambda s: s.costs.coe)

    def __check_init__(self):  # noqa: PLW3201 -- equinox's own validation hook
        if self.ireactor != 1 or self.ipnet != 0:
            raise ValueError(
                "Costs.coelc is only called when .costs.ireactor == 1 and "
                f".costs.ipnet == 0 (costs.py:82-83); got ireactor={self.ireactor}, "
                f"ipnet={self.ipnet}. On any other configuration PROCESS leaves "
                ".costs.coe at whatever it already held, so this node must not exist."
            )

    def __call__(
        self,
        p_plant_electric_net_mw=FromExactly(
            lambda s: s.heat_transport.p_plant_electric_net_mw
        ),
        f_t_plant_available=FromExactly(lambda s: s.costs.f_t_plant_available),
        t_plant_pulse_burn=FromExactly(lambda s: s.times.t_plant_pulse_burn),
        t_plant_pulse_total=FromExactly(lambda s: s.times.t_plant_pulse_total),
        concost=FromExactly(lambda s: s.costs.concost),
        fcap0=FromExactly(lambda s: s.costs.fcap0),
        fcr0=FromExactly(lambda s: s.costs.fcr0),
        discount_rate=FromExactly(lambda s: s.costs.discount_rate),
        life_blkt=FromExactly(lambda s: s.fwbs.life_blkt),
        fwallcst=FromExactly(lambda s: s.costs.fwallcst),
        blkcst=FromExactly(lambda s: s.costs.blkcst),
        cfind=FromExactly(lambda s: s.costs.cfind),
        lsa=FromExactly(lambda s: s.costs.lsa),
        fcap0cp=FromExactly(lambda s: s.costs.fcap0cp),
        ifueltyp=FromExactly(lambda s: s.costs.ifueltyp),
        life_blkt_fpy=FromExactly(lambda s: s.fwbs.life_blkt_fpy),
        life_plant=FromExactly(lambda s: s.costs.life_plant),
        life_div=FromExactly(lambda s: s.costs.life_div),
        divcst=FromExactly(lambda s: s.costs.divcst),
        life_div_fpy=FromExactly(lambda s: s.costs.life_div_fpy),
        cplife_cal=FromExactly(lambda s: s.costs.cplife_cal),
        cpstcst=FromExactly(lambda s: s.costs.cpstcst),
        cplife=FromExactly(lambda s: s.costs.cplife),
        cdrlife_cal=FromExactly(lambda s: s.costs.cdrlife_cal),
        cdcost=FromExactly(lambda s: s.costs.cdcost),
        fcdfuel=FromExactly(lambda s: s.costs.fcdfuel),
        ucoam=FromExactly(lambda s: s.costs.ucoam),
        ucfuel=FromExactly(lambda s: s.costs.ucfuel),
        f_plasma_fuel_helium3=FromExactly(lambda s: s.physics.f_plasma_fuel_helium3),
        wtgpd=FromExactly(lambda s: s.physics.wtgpd),
        uche3=FromExactly(lambda s: s.costs.uche3),
        ucwst=FromExactly(lambda s: s.costs.ucwst),
        decomf=FromExactly(lambda s: s.costs.decomf),
        dintrt=FromExactly(lambda s: s.costs.dintrt),
        dtlife=FromExactly(lambda s: s.costs.dtlife),
    ):
        return calculate_cost_of_electricity(
            self.ife,
            self.itart,
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
