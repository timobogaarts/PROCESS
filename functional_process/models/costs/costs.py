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

import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, Input, Output


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
    life_blkt = jnp.where(blkt_is_fast, life_blkt_fpy * f_t_plant_available, life_blkt_fpy)
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

    return c211, c212, c213, c2141, c2142, c214, c215, c216, c2171, c2172, c2173, c2174, c217, c21


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
        1.0e-6 * 2.0e0 * n_vv_vacuum_ducts * (dia_vv_vacuum_ducts * 1.2e0) ** 1.4e0 * UCVALV
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
        1.0e-6 * ucpfbs * pfckts * (srcktpm / safe_pfckts) ** 0.7e0,
        0.0,
    )

    c22525 = fkind * (1.0e-6 * ucpfbk * pfckts * (acptmax * vpfskv) ** 0.7e0)
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
            (1.0e6 * p_fw_div_heat_deposited_mw) ** exphts
            + (1.0e6 * p_blkt_nuclear_heat_total_mw) ** exphts
            + (1.0e6 * p_shld_nuclear_heat_mw) ** exphts
        )
        * cmlsa
    )
    chx = fkind * (
        1.0e-6
        * UCPHX
        * n_primary_heat_exchangers
        * (1.0e6 * p_plant_primary_heat_mw / n_primary_heat_exchangers) ** exphts
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
    return fkind * (1.0e-6 * UCNBV * (volrci + wsvol) ** 0.8e0)


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
        * (p_plant_electric_gross_mw / 1200.0e0) ** exptpe
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
        UCPP * (pacpmw * 1.0e3) ** expepe + UCAP * (p_plant_electric_base_total_mw * 1.0e3)
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


class ConvertFpyToCalendar(ExplicitFunction):
    """cottax node: `convert_fpy_to_calendar`."""

    life_blkt = Output(lambda s: s.fwbs.life_blkt)
    cdrlife_cal = Output(lambda s: s.costs.cdrlife_cal)
    life_div = Output(lambda s: s.costs.life_div)
    cplife_cal = Output(lambda s: s.costs.cplife_cal)

    def __call__(
        self,
        life_blkt_fpy=Input(lambda s: s.fwbs.life_blkt_fpy),
        life_plant=Input(lambda s: s.costs.life_plant),
        f_t_plant_available=Input(lambda s: s.costs.f_t_plant_available),
        life_div_fpy=Input(lambda s: s.costs.life_div_fpy),
        itart=Input(lambda s: s.physics.itart),
        cplife=Input(lambda s: s.costs.cplife),
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
        csi=Input(lambda s: s.costs.csi),
        lsa=Input(lambda s: s.costs.lsa),
        cland=Input(lambda s: s.costs.cland),
        ucrb=Input(lambda s: s.costs.ucrb),
        rbvol=Input(lambda s: s.buildings.rbvol),
        UCMB=Input(lambda s: s.costs.UCMB),
        rmbvol=Input(lambda s: s.buildings.rmbvol),
        UCWS=Input(lambda s: s.costs.UCWS),
        wsvol=Input(lambda s: s.buildings.wsvol),
        UCTR=Input(lambda s: s.costs.UCTR),
        triv=Input(lambda s: s.buildings.triv),
        UCEL=Input(lambda s: s.costs.UCEL),
        elevol=Input(lambda s: s.buildings.elevol),
        UCAD=Input(lambda s: s.costs.UCAD),
        admvol=Input(lambda s: s.buildings.admvol),
        UCCO=Input(lambda s: s.costs.UCCO),
        convol=Input(lambda s: s.buildings.convol),
        UCSH=Input(lambda s: s.costs.UCSH),
        shovol=Input(lambda s: s.buildings.shovol),
        UCCR=Input(lambda s: s.costs.UCCR),
        cryvol=Input(lambda s: s.buildings.cryvol),
        ireactor=Input(lambda s: s.costs.ireactor),
        cturbb=Input(lambda s: s.costs.cturbb),
    ):
        return calculate_structures_cost(
            csi, lsa, cland, ucrb, rbvol, UCMB, rmbvol, UCWS, wsvol, UCTR, triv, UCEL,
            elevol, UCAD, admvol, UCCO, convol, UCSH, shovol, UCCR, cryvol, ireactor,
            cturbb,
        )


class IndirectCosts(ExplicitFunction):
    """cottax node: `calculate_indirect_costs` (Account 9)."""

    cindrt = Output(lambda s: s.costs.cindrt)
    ccont = Output(lambda s: s.costs.ccont)

    def __call__(
        self,
        cfind=Input(lambda s: s.costs.cfind),
        lsa=Input(lambda s: s.costs.lsa),
        cdirt=Input(lambda s: s.costs.cdirt),
        cowner=Input(lambda s: s.costs.cowner),
        fcontng=Input(lambda s: s.costs.fcontng),
    ):
        return calculate_indirect_costs(cfind, lsa, cdirt, cowner, fcontng)


class ReactorStructureCost(ExplicitFunction):
    """cottax node: `calculate_reactor_structure_cost` (Account 221.4)."""

    c2214 = Output(lambda s: s.costs.c2214)

    def __call__(
        self,
        gsmass=Input(lambda s: s.structure.gsmass),
        UCGSS=Input(lambda s: s.costs.UCGSS),
        lsa=Input(lambda s: s.costs.lsa),
        fkind=Input(lambda s: s.costs.fkind),
    ):
        return calculate_reactor_structure_cost(gsmass, UCGSS, lsa, fkind)


class VacuumVesselAssemblyCost(ExplicitFunction):
    """cottax node: `calculate_vacuum_vessel_assembly_cost` (Account 222.3)."""

    c2223 = Output(lambda s: s.costs.c2223)

    def __call__(
        self,
        m_vv=Input(lambda s: s.fwbs.m_vv),
        uccryo=Input(lambda s: s.costs.uccryo),
        lsa=Input(lambda s: s.costs.lsa),
        fkind=Input(lambda s: s.costs.fkind),
    ):
        return calculate_vacuum_vessel_assembly_cost(m_vv, uccryo, lsa, fkind)


class DivertorCost(ExplicitFunction):
    """cottax node: `calculate_divertor_cost` (Account 221.5)."""

    c2215 = Output(lambda s: s.costs.c2215)
    divcst = Output(lambda s: s.costs.divcst)

    def __call__(
        self,
        ife=Input(lambda s: s.ife.ife),
        a_div_surface_total=Input(lambda s: s.divertor.a_div_surface_total),
        ucdiv=Input(lambda s: s.costs.ucdiv),
        fkind=Input(lambda s: s.costs.fkind),
        ifueltyp=Input(lambda s: s.costs.ifueltyp),
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
        i_vacuum_pump_type=Input(lambda s: s.vacuum.i_vacuum_pump_type),
        n_vac_pumps_high=Input(lambda s: s.vacuum.n_vac_pumps_high),
        UCCPMP=Input(lambda s: s.costs.UCCPMP),
        UCTPMP=Input(lambda s: s.costs.UCTPMP),
        n_vv_vacuum_ducts=Input(lambda s: s.vacuum.n_vv_vacuum_ducts),
        UCBPMP=Input(lambda s: s.costs.UCBPMP),
        dlscal=Input(lambda s: s.vacuum.dlscal),
        UCDUCT=Input(lambda s: s.costs.UCDUCT),
        dia_vv_vacuum_ducts=Input(lambda s: s.vacuum.dia_vv_vacuum_ducts),
        UCVALV=Input(lambda s: s.costs.UCVALV),
        m_vv_vacuum_duct_shield=Input(lambda s: s.vacuum.m_vv_vacuum_duct_shield),
        UCVDSH=Input(lambda s: s.costs.UCVDSH),
        UCVIAC=Input(lambda s: s.costs.UCVIAC),
        fkind=Input(lambda s: s.costs.fkind),
    ):
        return calculate_vacuum_system_cost(
            i_vacuum_pump_type, n_vac_pumps_high, UCCPMP, UCTPMP, n_vv_vacuum_ducts,
            UCBPMP, dlscal, UCDUCT, dia_vv_vacuum_ducts, UCVALV, m_vv_vacuum_duct_shield,
            UCVDSH, UCVIAC, fkind,
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
        uctfps=Input(lambda s: s.costs.uctfps),
        tfckw=Input(lambda s: s.tfcoil.tfckw),
        tfcmw=Input(lambda s: s.tfcoil.tfcmw),
        i_tf_sup=Input(lambda s: s.tfcoil.i_tf_sup),
        uctfbr=Input(lambda s: s.costs.uctfbr),
        n_tf_coils=Input(lambda s: s.tfcoil.n_tf_coils),
        c_tf_turn=Input(lambda s: s.tfcoil.c_tf_turn),
        v_tf_coil_dump_quench_kv=Input(lambda s: s.tfcoil.v_tf_coil_dump_quench_kv),
        uctfsw=Input(lambda s: s.costs.uctfsw),
        UCTFDR=Input(lambda s: s.costs.UCTFDR),
        e_tf_magnetic_stored_total_gj=Input(
            lambda s: s.tfcoil.e_tf_magnetic_stored_total_gj
        ),
        UCTFGR=Input(lambda s: s.costs.UCTFGR),
        UCTFIC=Input(lambda s: s.costs.UCTFIC),
        uctfbus=Input(lambda s: s.costs.uctfbus),
        m_tf_bus=Input(lambda s: s.tfcoil.m_tf_bus),
        ucbus=Input(lambda s: s.costs.ucbus),
        len_tf_bus=Input(lambda s: s.tfcoil.len_tf_bus),
        fkind=Input(lambda s: s.costs.fkind),
    ):
        return calculate_tf_coil_power_conditioning_cost(
            uctfps, tfckw, tfcmw, i_tf_sup, uctfbr, n_tf_coils, c_tf_turn,
            v_tf_coil_dump_quench_kv, uctfsw, UCTFDR, e_tf_magnetic_stored_total_gj,
            UCTFGR, UCTFIC, uctfbus, m_tf_bus, ucbus, len_tf_bus, fkind,
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
        ucpfps=Input(lambda s: s.costs.ucpfps),
        peakmva=Input(lambda s: s.heat_transport.peakmva),
        ucpfic=Input(lambda s: s.costs.ucpfic),
        pfckts=Input(lambda s: s.pf_power.pfckts),
        ucpfb=Input(lambda s: s.costs.ucpfb),
        spfbusl=Input(lambda s: s.pf_power.spfbusl),
        acptmax=Input(lambda s: s.pf_power.acptmax),
        ucpfbs=Input(lambda s: s.costs.ucpfbs),
        srcktpm=Input(lambda s: s.pf_power.srcktpm),
        ucpfbk=Input(lambda s: s.costs.ucpfbk),
        vpfskv=Input(lambda s: s.pf_power.vpfskv),
        ucpfdr1=Input(lambda s: s.costs.ucpfdr1),
        ensxpfm=Input(lambda s: s.pf_power.ensxpfm),
        ucpfcb=Input(lambda s: s.costs.ucpfcb),
        fkind=Input(lambda s: s.costs.fkind),
    ):
        return calculate_pf_coil_power_conditioning_cost(
            ucpfps, peakmva, ucpfic, pfckts, ucpfb, spfbusl, acptmax, ucpfbs, srcktpm,
            ucpfbk, vpfskv, ucpfdr1, ensxpfm, ucpfcb, fkind,
        )


class ReactorCoolingSystemCost(ExplicitFunction):
    """cottax node: `calculate_reactor_cooling_system_cost` (Account 2261)."""

    cpp = Output(lambda s: s.costs.cpp)
    chx = Output(lambda s: s.costs.chx)
    c2261 = Output(lambda s: s.costs.c2261)

    def __call__(
        self,
        uchts=Input(lambda s: s.costs.uchts),
        i_blkt_coolant_type=Input(lambda s: s.fwbs.i_blkt_coolant_type),
        p_fw_div_heat_deposited_mw=Input(
            lambda s: s.heat_transport.p_fw_div_heat_deposited_mw
        ),
        p_blkt_nuclear_heat_total_mw=Input(
            lambda s: s.fwbs.p_blkt_nuclear_heat_total_mw
        ),
        p_shld_nuclear_heat_mw=Input(lambda s: s.fwbs.p_shld_nuclear_heat_mw),
        lsa=Input(lambda s: s.costs.lsa),
        fkind=Input(lambda s: s.costs.fkind),
        UCPHX=Input(lambda s: s.costs.UCPHX),
        n_primary_heat_exchangers=Input(
            lambda s: s.heat_transport.n_primary_heat_exchangers
        ),
        p_plant_primary_heat_mw=Input(lambda s: s.heat_transport.p_plant_primary_heat_mw),
    ):
        return calculate_reactor_cooling_system_cost(
            uchts, i_blkt_coolant_type, p_fw_div_heat_deposited_mw,
            p_blkt_nuclear_heat_total_mw, p_shld_nuclear_heat_mw, lsa, fkind, UCPHX,
            n_primary_heat_exchangers, p_plant_primary_heat_mw,
        )


class FuellingSystemCost(ExplicitFunction):
    """cottax node: `calculate_fuelling_system_cost` (Account 2271)."""

    c2271 = Output(lambda s: s.costs.c2271)

    def __call__(
        self,
        ucf1=Input(lambda s: s.costs.ucf1),
        fkind=Input(lambda s: s.costs.fkind),
    ):
        return calculate_fuelling_system_cost(ucf1, fkind)


class NuclearBuildingVentilationCost(ExplicitFunction):
    """cottax node: `calculate_nuclear_building_ventilation_cost` (Account 2274)."""

    c2274 = Output(lambda s: s.costs.c2274)

    def __call__(
        self,
        UCNBV=Input(lambda s: s.costs.UCNBV),
        volrci=Input(lambda s: s.buildings.volrci),
        wsvol=Input(lambda s: s.buildings.wsvol),
        fkind=Input(lambda s: s.costs.fkind),
    ):
        return calculate_nuclear_building_ventilation_cost(UCNBV, volrci, wsvol, fkind)


class InstrumentationAndControlCost(ExplicitFunction):
    """cottax node: `calculate_instrumentation_and_control_cost` (Account 228)."""

    c228 = Output(lambda s: s.costs.c228)

    def __call__(
        self,
        uciac=Input(lambda s: s.costs.uciac),
        fkind=Input(lambda s: s.costs.fkind),
    ):
        return calculate_instrumentation_and_control_cost(uciac, fkind)


class MaintenanceEquipmentCost(ExplicitFunction):
    """cottax node: `calculate_maintenance_equipment_cost` (Account 229)."""

    c229 = Output(lambda s: s.costs.c229)

    def __call__(
        self,
        ucme=Input(lambda s: s.costs.ucme),
        fkind=Input(lambda s: s.costs.fkind),
    ):
        return calculate_maintenance_equipment_cost(ucme, fkind)


class TurbinePlantEquipmentCost(ExplicitFunction):
    """cottax node: `calculate_turbine_plant_equipment_cost` (Account 23)."""

    c23 = Output(lambda s: s.costs.c23)

    def __call__(
        self,
        ireactor=Input(lambda s: s.costs.ireactor),
        ucturb=Input(lambda s: s.costs.ucturb),
        i_blkt_coolant_type=Input(lambda s: s.fwbs.i_blkt_coolant_type),
        p_plant_electric_gross_mw=Input(
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
        UCSWYD=Input(lambda s: s.costs.UCSWYD),
        lsa=Input(lambda s: s.costs.lsa),
    ):
        return calculate_switchyard_cost(UCSWYD, lsa)


class TransformersCost(ExplicitFunction):
    """cottax node: `calculate_transformers_cost` (Account 242)."""

    c242 = Output(lambda s: s.costs.c242)

    def __call__(
        self,
        UCPP=Input(lambda s: s.costs.UCPP),
        pacpmw=Input(lambda s: s.heat_transport.pacpmw),
        UCAP=Input(lambda s: s.costs.UCAP),
        p_plant_electric_base_total_mw=Input(
            lambda s: s.heat_transport.p_plant_electric_base_total_mw
        ),
        lsa=Input(lambda s: s.costs.lsa),
    ):
        return calculate_transformers_cost(
            UCPP, pacpmw, UCAP, p_plant_electric_base_total_mw, lsa
        )


class LowVoltageCost(ExplicitFunction):
    """cottax node: `calculate_low_voltage_cost` (Account 243)."""

    c243 = Output(lambda s: s.costs.c243)

    def __call__(
        self,
        UCLV=Input(lambda s: s.costs.UCLV),
        tlvpmw=Input(lambda s: s.heat_transport.tlvpmw),
        lsa=Input(lambda s: s.costs.lsa),
    ):
        return calculate_low_voltage_cost(UCLV, tlvpmw, lsa)


class DieselGeneratorsCost(ExplicitFunction):
    """cottax node: `calculate_diesel_generators_cost` (Account 244)."""

    c244 = Output(lambda s: s.costs.c244)

    def __call__(
        self,
        UCDGEN=Input(lambda s: s.costs.UCDGEN),
        lsa=Input(lambda s: s.costs.lsa),
    ):
        return calculate_diesel_generators_cost(UCDGEN, lsa)


class AuxiliaryFacilityPowerCost(ExplicitFunction):
    """cottax node: `calculate_auxiliary_facility_power_cost` (Account 245)."""

    c245 = Output(lambda s: s.costs.c245)

    def __call__(
        self,
        UCAF=Input(lambda s: s.costs.UCAF),
        lsa=Input(lambda s: s.costs.lsa),
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
        c241=Input(lambda s: s.costs.c241),
        c242=Input(lambda s: s.costs.c242),
        c243=Input(lambda s: s.costs.c243),
        c244=Input(lambda s: s.costs.c244),
        c245=Input(lambda s: s.costs.c245),
    ):
        return calculate_electric_plant_equipment_cost(c241, c242, c243, c244, c245)


class MiscPlantEquipmentCost(ExplicitFunction):
    """cottax node: `calculate_misc_plant_equipment_cost` (Account 25)."""

    c25 = Output(lambda s: s.costs.c25)

    def __call__(
        self,
        ucmisc=Input(lambda s: s.costs.ucmisc),
        lsa=Input(lambda s: s.costs.lsa),
    ):
        return calculate_misc_plant_equipment_cost(ucmisc, lsa)


class HeatRejectionCost(ExplicitFunction):
    """cottax node: `calculate_heat_rejection_cost` (Account 26)."""

    c26 = Output(lambda s: s.costs.c26)

    def __call__(
        self,
        ireactor=Input(lambda s: s.costs.ireactor),
        p_fusion_total_mw=Input(lambda s: s.physics.p_fusion_total_mw),
        p_hcd_electric_total_mw=Input(
            lambda s: s.heat_transport.p_hcd_electric_total_mw
        ),
        tfcmw=Input(lambda s: s.tfcoil.tfcmw),
        p_plant_primary_heat_mw=Input(lambda s: s.heat_transport.p_plant_primary_heat_mw),
        p_plant_electric_gross_mw=Input(
            lambda s: s.heat_transport.p_plant_electric_gross_mw
        ),
        uchrs=Input(lambda s: s.costs.uchrs),
        lsa=Input(lambda s: s.costs.lsa),
    ):
        return calculate_heat_rejection_cost(
            ireactor, p_fusion_total_mw, p_hcd_electric_total_mw, tfcmw,
            p_plant_primary_heat_mw, p_plant_electric_gross_mw, uchrs, lsa,
        )
