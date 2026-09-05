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

from functional_process.models.safe_math import safe_pow, safe_sqrt


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
    return calculate_first_wall_cost_magnetic_confinement(
        lsa,
        UCFWA,
        UCFWS,
        a_fw_total,
        UCFWPS,
        fkind,
        ifueltyp,
    )


def calculate_first_wall_cost_magnetic_confinement(
    lsa,
    UCFWA,
    UCFWS,
    a_fw_total,
    UCFWPS,
    fkind,
    ifueltyp,
):
    """Account 221.1's magnetic-confinement arm -- `calculate_first_wall_cost`'s
    body at `ife != 1`, and the only arm this port has an occupant for.

    Split out of the composite above rather than left inside it because `ife` is a
    model-selection switch, and under `_audit/next_steps.md` §14.2 no switch is a static
    kwarg: the node calls this function and never sees the integer. The composite keeps
    the refusal -- it is what `functional_process/tests/models/costs/test_costs.py`
    diffs against PROCESS's own method, sample by sample, `ife` included -- and
    `machine_from_indat` refuses `ife == 1` at assembly instead, once, for all seven
    Account-22x nodes at once (`indat.py`'s `_ife_cost_accounts_arm`).

    Parameters and returns are the composite's, less `ife`.
    """
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


def calculate_blanket_cost_magnetic_confinement(
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
    """Account 221.2's magnetic-confinement arm -- `calculate_blanket_cost`'s body
    at `ife != 1`, and the only arm this port has an occupant for.

    Split out of the composite above rather than left inside it because `ife` is a
    model-selection switch, and under `_audit/next_steps.md` §14.2 no switch is a static
    kwarg: the node calls this function and never sees the integer. The composite keeps
    the refusal -- it is what `functional_process/tests/models/costs/test_costs.py`
    diffs against PROCESS's own method, sample by sample, `ife` included -- and
    `machine_from_indat` refuses `ife == 1` at assembly instead, once, for all seven
    Account-22x nodes at once (`indat.py`'s `_ife_cost_accounts_arm`).

    Parameters and returns are the composite's, less `ife`.
    """
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
    return calculate_shield_cost_magnetic_confinement(
        lsa,
        whtshld,
        ucshld,
        wpenshld,
        ucpens,
        fkind,
    )


def calculate_shield_cost_magnetic_confinement(
    lsa,
    whtshld,
    ucshld,
    wpenshld,
    ucpens,
    fkind,
):
    """Account 221.3's magnetic-confinement arm -- `calculate_shield_cost`'s body at
    `ife != 1`, and the only arm this port has an occupant for.

    Split out of the composite above rather than left inside it because `ife` is a
    model-selection switch, and under `_audit/next_steps.md` §14.2 no switch is a static
    kwarg: the node calls this function and never sees the integer. The composite keeps
    the refusal -- it is what `functional_process/tests/models/costs/test_costs.py`
    diffs against PROCESS's own method, sample by sample, `ife` included -- and
    `machine_from_indat` refuses `ife == 1` at assembly instead, once, for all seven
    Account-22x nodes at once (`indat.py`'s `_ife_cost_accounts_arm`).

    Parameters and returns are the composite's, less `ife`.
    """
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
    if supercond_cost_model == 0:
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


def calculate_tf_magnet_cost_superconducting_per_kg(
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
):
    """Account 222.1's `supercond_cost_model == PER_KG` arm -- the legacy `$/kg`
    superconductor costing (`costs.py:1550-1554`).

    The strand cost per metre is the superconductor mass per coil times its unit cost
    per kilogram, divided by the winding length; **`.costs.sc_mat_cost_0`,
    `.tfcoil.j_crit_str_0` and `.tfcoil.j_crit_str_tf` are not read at all**, which is
    the three-edge difference that made this a family rather than a static kwarg
    (`_audit/next_steps.md` §14.2).

    `.tfcoil.i_tf_sc_mat` stays an ordinary declared read on **both** arms: it indexes
    a cost table rather than selecting a model here, which is the case the policy
    leaves as a read.

    Parameters and returns are `calculate_tf_magnet_cost_superconducting`'s, less
    `supercond_cost_model` and the other arm's three fields.
    """
    return _tf_magnet_cost_superconducting(
        jnp.asarray(ucsc)[i_tf_sc_mat - 1]
        * m_tf_coil_superconductor
        / (len_tf_coil * n_tf_coil_turns),
        lsa,
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


def calculate_tf_magnet_cost_superconducting_per_kam(
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
):
    """Account 222.1's `supercond_cost_model == PER_KAM` arm -- strand cost scaled by
    critical current density (`costs.py:1556-1560`).

    **`.costs.ucsc` and `.tfcoil.m_tf_coil_superconductor` are not read at all** on this
    arm; see its sibling above.

    Parameters and returns are `calculate_tf_magnet_cost_superconducting`'s, less
    `supercond_cost_model` and the other arm's two fields.
    """
    return _tf_magnet_cost_superconducting(
        jnp.asarray(sc_mat_cost_0)[i_tf_sc_mat - 1]
        * jnp.asarray(j_crit_str_0)[i_tf_sc_mat - 1]
        / j_crit_str_tf,
        lsa,
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


def _tf_magnet_cost_superconducting(
    costtfsc,
    lsa,
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
):
    """The ~15 lines both `supercond_cost_model` arms share, given the superconductor
    strand cost per metre the arm computed.

    `costtfsc` is data, not a switch: which of the two expressions produced it follows
    from which arm function called this one.
    """
    cmlsa = jnp.asarray([0.6900e0, 0.8450e0, 0.9225e0, 1.0000e0])[lsa - 1]

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
    exist". With `n_cs_pf_coils` static and the central-solenoid arm chosen by which
    occupant was selected, both loops are ordinary Python `range`s unrolled at trace
    time and no `lax.fori_loop` or padding is needed.
    `mda_harness.switch_audit` checks the count against the run automatically.

    **`supercond_cost_model` is a dispatcher argument, not a static kwarg, since
    2026-08-30, and `iohcl` joined it on 2026-08-31**; this function is now a 2x2
    dispatcher over `calculate_pf_magnet_cost_per_kg`/`_per_kam` and their
    `_no_central_solenoid` siblings, all four sharing `_pf_magnet_cost`. It survives
    only as the shape the tier-1 contract exercises -- no node carries either switch.
    The two `supercond_cost_model` arms read **disjoint** strand-cost fields -- arm `0`
    reads
    `.costs.ucsc` and `.tfcoil.dcond`, arm `1` reads `.costs.sc_mat_cost_0`,
    `.tfcoil.j_crit_str_0`, `.pf_coil.j_crit_str_pf` and `.pf_coil.j_crit_str_cs` -- so
    one node carrying the switch declared four edges the reference run does not make.
    That is the defect `TfMagnetCostSuperconducting` was split into a
    `supercond_cost_model` slot to remove one account earlier in this same file
    (`_audit/next_steps.md` §14.2, `_audit/switch_kwarg_survey.md` §3), and the split
    here is the same shape. Unlike the TF coil's, this one is interleaved: the switch is
    branched on **twice**, once inside the PF loop and once inside the CS block, so the
    arms compute two strand costs each and `_pf_magnet_cost` takes both as data.

    `i_pf_conductor` stays static, and remains a real branch: it selects the conduit cost
    and the copper fraction, and zeroes the superconductor strand cost on both arms.

    On the reference stellarator run every output of this function is exactly zero
    (`n_cs_pf_coils == 0`, `iohcl == 0`, and both `.pf_coil.m_pf_coil_structure_total`
    and `.structure.fncmass` are `0.0`) -- PROCESS never runs its PF coil model for a
    stellarator. That is reproduced, not special-cased, and it is why the *slot* is
    `None` on that device (`models/costs/namespace.py::pf_magnet_cost`): the arithmetic
    is faithful, but a node whose ports assert a dependence on a subsystem the device
    does not have is the `EcrhDensityLimit` bug class. The function stays as written
    because it is the tokamak's, and the tokamak's answer on `large_tokamak_nof` is
    `c2222 = 591.8489361585899`, bit-for-bit PROCESS's.

    Parameters
    ----------
    n_cs_pf_coils :
        Number of PF coils including the central solenoid. `.pf_coil.n_cs_pf_coils`.
        Static (loop bound).
    iohcl :
        1 if a central solenoid is present, else 0. `.build.iohcl`. A dispatcher
        argument: it selects one of the two `_no_central_solenoid` pairs and reaches no
        occupant.
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
    if supercond_cost_model == 0:
        if iohcl == 1:
            return calculate_pf_magnet_cost_per_kg(
                n_cs_pf_coils,
                i_pf_conductor,
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
                jnp.asarray(dcond)[i_pf_superconductor - 1],
                jnp.asarray(dcond)[i_cs_superconductor - 1],
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
        return calculate_pf_magnet_cost_per_kg_no_central_solenoid(
            n_cs_pf_coils,
            i_pf_conductor,
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
            jnp.asarray(dcond)[i_pf_superconductor - 1],
            uccu,
            cconfix,
            ucwindpf,
            uccase,
            m_pf_coil_structure_total,
            ucfnc,
            fncmass,
            fkind,
        )
    if iohcl == 1:
        return calculate_pf_magnet_cost_per_kam(
            n_cs_pf_coils,
            i_pf_conductor,
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
    return calculate_pf_magnet_cost_per_kam_no_central_solenoid(
        n_cs_pf_coils,
        i_pf_conductor,
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


def _pf_strand_costs_per_kg(
    n_pf_coils_costed,
    is_superconducting,
    ucsc,
    i_pf_superconductor,
    fcupfsu,
    f_a_pf_coil_void,
    c_pf_cs_coils_peak_ma,
    n_pf_coil_turns,
    j_pf_coil_wp_peak,
    den_pf_conductor,
):
    """One PF coil's superconductor strand cost per metre, per costed coil
    (`costs.py:1639-1655`).

    `n_pf_coils_costed` is the loop bound `acc2222`'s first loop walks, and it is the
    *only* thing `.build.iohcl` used to decide here: `n_cs_pf_coils - 1` when a central
    solenoid occupies the last slot, all of them when there is none. The two callers
    each pass the count their own arm implies, so the switch is answered by which
    occupant was selected rather than re-read inside the formula.

    **`den_pf_conductor` is the density itself, not `.tfcoil.dcond` and an index into
    it** (2026-08-31). PROCESS writes `dcond(i_pf_superconductor)`; the node above now
    reads that one element as its own port, for the reason `pfcoil/masses.py` already
    does -- see `PfMagnetCostPerKg`.
    """
    n_pf_coil_turns = jnp.asarray(n_pf_coil_turns)
    f_a_pf_coil_void = jnp.asarray(f_a_pf_coil_void)
    c_pf_cs_coils_peak_ma = jnp.asarray(c_pf_cs_coils_peak_ma)
    j_pf_coil_wp_peak = jnp.asarray(j_pf_coil_wp_peak)
    return [
        (
            jnp.asarray(ucsc)[i_pf_superconductor - 1]
            * (1.0e0 - fcupfsu)
            * (1.0e0 - f_a_pf_coil_void[i])
            * abs(c_pf_cs_coils_peak_ma[i] / n_pf_coil_turns[i])
            * 1.0e6
            / j_pf_coil_wp_peak[i]
            * den_pf_conductor
        )
        if is_superconducting
        else 0.0
        for i in range(n_pf_coils_costed)
    ]


def calculate_pf_magnet_cost_per_kg(
    n_cs_pf_coils,
    i_pf_conductor,
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
    """Account 222.2's `supercond_cost_model == PER_KG` arm on a machine **with** a
    central solenoid -- the legacy `$/kg` superconductor costing (`costs.py:1639-1655`
    for the PF coils, `:1770-1790` for the central solenoid).

    Each coil's superconductor strand cost per metre is the volume of non-copper,
    non-void conductor in one turn times its density times its unit cost per kilogram:
    for a PF coil, from the turn current and the winding-pack current density; for the
    CS, from the cable-space cross-section directly (PROCESS issue #328). **`.costs.
    sc_mat_cost_0`, `.tfcoil.j_crit_str_0`, `.pf_coil.j_crit_str_pf` and
    `.pf_coil.j_crit_str_cs` are not read at all**, which is the four-edge difference
    that made this a family rather than a static kwarg.

    `.pf_coil.i_pf_superconductor` and `.pf_coil.i_cs_superconductor` stay ordinary
    declared reads on **both** arms: they index a cost table rather than selecting a
    model here, which is the case the policy leaves as a read -- exactly as
    `.tfcoil.i_tf_sc_mat` does for Account 222.1.

    **The two conductor *densities* are scalars here, not `.tfcoil.dcond` and an index
    into it** (2026-08-31). The array read moved out to the caller for a reason that is
    not about this arithmetic at all: `pfcoil/masses.py` and
    `tfcoil/superconducting.py` already read the same array **by element**
    (`FromExactly(tfcoil.dcond[k])`), so a node reading it whole named the same storage
    both ways -- and a pytree cannot be written back at a path named both whole and by
    element (the graph library's antichain check on write paths). Measured on
    2026-08-31: two such
    pairs on `large_tokamak_nof`/`large_tokamak_eval`, two on `low_aspect_ratio_DEMO`,
    one on `spherical_tokamak_eval`, none on either stellarator (which has no PF magnet
    cost node). `radiation_power.py`'s `f_nd_impurity_electron_array` is the standing
    precedent for the fix; `.costs.ucsc` stays whole because nothing reads *it* by
    element.

    Parameters and returns are `calculate_pf_magnet_cost`'s, less
    `supercond_cost_model` and the other arm's four fields, and with `dcond` replaced by
    the two elements the dispatcher selects out of it.
    """
    is_superconducting = i_pf_conductor == 0
    costpfsc = _pf_strand_costs_per_kg(
        n_cs_pf_coils - 1,
        is_superconducting,
        ucsc,
        i_pf_superconductor,
        fcupfsu,
        f_a_pf_coil_void,
        c_pf_cs_coils_peak_ma,
        n_pf_coil_turns,
        j_pf_coil_wp_peak,
        den_pf_conductor,
    )
    #  Issue #328: use the CS conductor cross-sectional area (m^2).
    costpfsc_cs = (
        (
            jnp.asarray(ucsc)[i_cs_superconductor - 1]
            * a_cs_cable_space
            * (1 - f_a_cs_void)
            * (1 - fcuohsu)
            / jnp.asarray(n_pf_coil_turns)[n_cs_pf_coils - 1]
            * den_cs_conductor
        )
        if is_superconducting
        else 0.0
    )
    return _pf_magnet_cost(
        costpfsc,
        costpfsc_cs,
        n_cs_pf_coils,
        i_pf_conductor,
        lsa,
        r_pf_coil_middle,
        n_pf_coil_turns,
        cconshpf,
        fcupfsu,
        f_a_pf_coil_void,
        c_pf_cs_coils_peak_ma,
        j_pf_coil_wp_peak,
        uccu,
        cconfix,
        ucwindpf,
        uccase,
        m_pf_coil_structure_total,
        ucfnc,
        fncmass,
        fkind,
        a_cs_cable_space=a_cs_cable_space,
        f_a_cs_void=f_a_cs_void,
        fcuohsu=fcuohsu,
    )


def calculate_pf_magnet_cost_per_kg_no_central_solenoid(
    n_cs_pf_coils,
    i_pf_conductor,
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
):
    """The same arm on a machine with **no** central solenoid (`.build.iohcl == 0`).

    Two differences from the sibling above, both of them `costs.py`'s own: the first
    loop walks all `n_cs_pf_coils` slots rather than `n_cs_pf_coils - 1`
    (`costs.py:1630-1633`), and neither of the two CS conductor terms exists at all
    (`:1770-1790` and `:1745-1760` are both inside `if iohcl == 1`).

    **Four reads leave with this occupant**: `.pf_coil.i_cs_superconductor`,
    `.pf_coil.a_cs_cable_space`, `.pf_coil.f_a_cs_void` and `.pf_coil.fcuohsu` -- which
    is why `iohcl` is a family here and not a static kwarg. On both tracked spherical
    tokamaks PROCESS never writes `a_cs_cable_space` at all (`provider`'s pin records it
    as `unwritten`), so the sibling's port would read a field the run does not produce.

    `den_pf_conductor` is the one `.tfcoil.dcond` element this arm needs, taken as a
    scalar for the antichain reason `calculate_pf_magnet_cost_per_kg` states.
    """
    costpfsc = _pf_strand_costs_per_kg(
        n_cs_pf_coils,
        i_pf_conductor == 0,
        ucsc,
        i_pf_superconductor,
        fcupfsu,
        f_a_pf_coil_void,
        c_pf_cs_coils_peak_ma,
        n_pf_coil_turns,
        j_pf_coil_wp_peak,
        den_pf_conductor,
    )
    return _pf_magnet_cost(
        costpfsc,
        None,
        n_cs_pf_coils,
        i_pf_conductor,
        lsa,
        r_pf_coil_middle,
        n_pf_coil_turns,
        cconshpf,
        fcupfsu,
        f_a_pf_coil_void,
        c_pf_cs_coils_peak_ma,
        j_pf_coil_wp_peak,
        uccu,
        cconfix,
        ucwindpf,
        uccase,
        m_pf_coil_structure_total,
        ucfnc,
        fncmass,
        fkind,
    )


def calculate_pf_magnet_cost_per_kam(
    n_cs_pf_coils,
    i_pf_conductor,
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
):
    """Account 222.2's `supercond_cost_model == PER_KAM` arm on a machine **with** a
    central solenoid -- strand cost scaled by critical current density
    (`costs.py:1656-1666` for the PF coils, `:1791-1801` for the central solenoid).

    **`.costs.ucsc` and `.tfcoil.dcond` are not read at all** on this arm; see its
    sibling above. Both strand costs here are constants over the loop -- neither reads
    any per-coil quantity -- so this arm's `costpfsc` is one number repeated, which is
    PROCESS's own behaviour and not a simplification.

    Parameters and returns are `calculate_pf_magnet_cost`'s, less
    `supercond_cost_model` and the other arm's two fields.
    """
    is_superconducting = i_pf_conductor == 0
    costpfsc_pf = _pf_strand_cost_per_kam(
        is_superconducting,
        sc_mat_cost_0,
        j_crit_str_0,
        i_pf_superconductor,
        j_crit_str_pf,
    )
    costpfsc_cs = _pf_strand_cost_per_kam(
        is_superconducting,
        sc_mat_cost_0,
        j_crit_str_0,
        i_cs_superconductor,
        j_crit_str_cs,
    )
    return _pf_magnet_cost(
        [costpfsc_pf] * (n_cs_pf_coils - 1),
        costpfsc_cs,
        n_cs_pf_coils,
        i_pf_conductor,
        lsa,
        r_pf_coil_middle,
        n_pf_coil_turns,
        cconshpf,
        fcupfsu,
        f_a_pf_coil_void,
        c_pf_cs_coils_peak_ma,
        j_pf_coil_wp_peak,
        uccu,
        cconfix,
        ucwindpf,
        uccase,
        m_pf_coil_structure_total,
        ucfnc,
        fncmass,
        fkind,
        a_cs_cable_space=a_cs_cable_space,
        f_a_cs_void=f_a_cs_void,
        fcuohsu=fcuohsu,
    )


def calculate_pf_magnet_cost_per_kam_no_central_solenoid(
    n_cs_pf_coils,
    i_pf_conductor,
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
):
    """The `PER_KAM` arm with no central solenoid -- the same two differences
    `calculate_pf_magnet_cost_per_kg_no_central_solenoid` documents.

    **Five reads leave with this occupant**, one more than the `PER_KG` pair, because
    this arm's CS strand cost also needs `.pf_coil.j_crit_str_cs`.
    """
    costpfsc_pf = _pf_strand_cost_per_kam(
        i_pf_conductor == 0,
        sc_mat_cost_0,
        j_crit_str_0,
        i_pf_superconductor,
        j_crit_str_pf,
    )
    return _pf_magnet_cost(
        [costpfsc_pf] * n_cs_pf_coils,
        None,
        n_cs_pf_coils,
        i_pf_conductor,
        lsa,
        r_pf_coil_middle,
        n_pf_coil_turns,
        cconshpf,
        fcupfsu,
        f_a_pf_coil_void,
        c_pf_cs_coils_peak_ma,
        j_pf_coil_wp_peak,
        uccu,
        cconfix,
        ucwindpf,
        uccase,
        m_pf_coil_structure_total,
        ucfnc,
        fncmass,
        fkind,
    )


def _pf_strand_cost_per_kam(
    is_superconducting, sc_mat_cost_0, j_crit_str_0, i_superconductor, j_crit_str
):
    """The `PER_KAM` strand cost per metre for one conductor family
    (`costs.py:1656-1666`, and the byte-identical `:1791-1801` for the CS).

    One helper for both sites because the two differ only in which superconductor index
    and which critical current density they read -- the shape the source repeats and
    `_pf_magnet_cost`'s docstring already names.
    """
    return (
        (
            jnp.asarray(sc_mat_cost_0)[i_superconductor - 1]
            * jnp.asarray(j_crit_str_0)[i_superconductor - 1]
            / j_crit_str
        )
        if is_superconducting
        else 0.0
    )


def _pf_magnet_cost(
    costpfsc,
    costpfsc_cs,
    n_cs_pf_coils,
    i_pf_conductor,
    lsa,
    r_pf_coil_middle,
    n_pf_coil_turns,
    cconshpf,
    fcupfsu,
    f_a_pf_coil_void,
    c_pf_cs_coils_peak_ma,
    j_pf_coil_wp_peak,
    uccu,
    cconfix,
    ucwindpf,
    uccase,
    m_pf_coil_structure_total,
    ucfnc,
    fncmass,
    fkind,
    *,
    a_cs_cable_space=None,
    f_a_cs_void=None,
    fcuohsu=None,
):
    """The ~40 lines all four `Account 222.2` occupants share, given the superconductor
    strand cost per metre each arm computed for each PF coil and for the CS.

    `costpfsc`/`costpfsc_cs` are data, not switches: which of the two expressions
    produced them follows from which arm function called this one. **Two of them rather
    than one**, unlike `_tf_magnet_cost_superconducting`'s single `costtfsc`, because
    `acc2222` branches on `supercond_cost_model` twice -- once per PF coil and once for
    the central solenoid -- with the copper, conduit and winding-length arithmetic
    interleaved between the two sites.

    **`iohcl` is not a parameter.** `costpfsc_cs is None` marks a machine with no
    central solenoid -- absence, not a variant, the same reading `_pf_coil_system_arm`
    gives the switch -- and the three CS-only fields are keyword-only with no default
    value so that a no-solenoid caller cannot supply them. `len(costpfsc)` is the first
    loop's bound, so the caller's own arm decides how many coils are costed rather than
    a switch re-read here.
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

    c22221 = 0.0
    for i in range(len(costpfsc)):
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

        cpfconpm = costpfsc[i] + costpfcu + costpfsh + cconfix
        c22221 += (
            1.0e-6 * 2.0 * jnp.pi * r_pf_coil_middle[i] * n_pf_coil_turns[i] * cpfconpm
        )

    if costpfsc_cs is not None:
        cs = n_cs_pf_coils - 1
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

        cpfconpm = costpfsc_cs + costpfcu + costpfsh + cconfix
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


def calculate_power_injection_cost_magnetic_confinement(
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
    """Account 223's magnetic-confinement arm -- `calculate_power_injection_cost`'s
    body at `ife != 1`, and the only arm this port has an occupant for.

    Split out of the composite above rather than left inside it because `ife` is a
    model-selection switch, and under `_audit/next_steps.md` §14.2 no switch is a static
    kwarg: the node calls this function and never sees the integer. The composite keeps
    the refusal -- it is what `functional_process/tests/models/costs/test_costs.py`
    diffs against PROCESS's own method, sample by sample, `ife` included -- and
    `machine_from_indat` refuses `ife == 1` at assembly instead, once, for all seven
    Account-22x nodes at once (`indat.py`'s `_ife_cost_accounts_arm`).

    Parameters and returns are the composite's, less `ife`.
    """
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
    if i_pulsed_plant != 1:
        # `c2253 = 0.0`, then PROCESS's own `if istore < 3` still scales it -- kept
        # exactly, so the composite stays bit-identical to what it was before the split.
        if istore < 3:
            return _energy_storage_cost_scaled(0.0e0, p_plant_electric_net_mw, fkind)
        return fkind * 0.0e0
    if istore == 1:
        return calculate_energy_storage_cost_electrowatt_option_1(
            p_plant_electric_net_mw, fkind
        )
    if istore == 2:
        return calculate_energy_storage_cost_electrowatt_option_2(
            p_plant_electric_net_mw, fkind
        )
    if istore == 3:
        raise NotImplementedError(
            "acc2253's istore == 3 arm (a stainless-steel thermal storage block) "
            "reads .heat_transport.p_plant_primary_heat_mw, "
            ".times.t_plant_pulse_no_burn and .pulse.dtstor, which options 1/2 do "
            "not -- a different reads-set. Not ported; the reference run has "
            ".pulse.i_pulsed_plant == 0, so no istore arm is reached at all."
        )
    raise ValueError(f"Illegal value for istore: {istore}")


def calculate_energy_storage_cost_electrowatt_option_1(p_plant_electric_net_mw, fkind):
    """Account 225.3's `istore == ELECTROWATT_OPTION_1` arm (`costs.py:2617-2643`).

    0.1 condensate tank + 0.8 feedpump + 4.0 turbine-generator duty + 0.5 auxiliary
    transformer + 2.8 drum + 29.0 externally fired superheater, in 1992 pounds.
    """
    return _energy_storage_cost_scaled(
        0.1e0 + 0.8e0 + 4.0e0 + 0.5e0 + 2.8e0 + 29.0e0,
        p_plant_electric_net_mw,
        fkind,
    )


def calculate_energy_storage_cost_electrowatt_option_2(p_plant_electric_net_mw, fkind):
    """Account 225.3's `istore == ELECTROWATT_OPTION_2` arm (`costs.py:2645-2682`).

    0.1 + 0.8 + 2.8 + 4.0 + 330.0 fired boiler + 1.0 steam bypass + 2.0 dump condenser
    + 18.0 cooling water, in 1992 pounds.
    """
    return _energy_storage_cost_scaled(
        0.1e0 + 0.8e0 + 2.8e0 + 4.0e0 + 330.0e0 + 1.0e0 + 2.0e0 + 18.0e0,
        p_plant_electric_net_mw,
        fkind,
    )


def _energy_storage_cost_scaled(c2253_1992_pounds, p_plant_electric_net_mw, fkind):
    """Scale an ELECTROWATT itemised sum with net electric power and convert 1992
    pounds to 1990 dollars (inflation 5%/yr + 1.5 $/pound exchange rate).

    Shared by options 1 and 2, which differ **only** in the literal handed in -- which
    is why `_audit/switch_kwarg_survey.md` band (c) argued they should stay one node
    carrying an `istore` kwarg. `_audit/next_steps.md` §14.2 withdrew that: a switch
    value selects an occupant whatever its reads, and the two literals live in the two
    arm functions above rather than behind an integer.
    """
    return fkind * (c2253_1992_pounds * p_plant_electric_net_mw / 1200.0e0 * 1.36e0)


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


def calculate_auxiliary_component_cooling_cost_magnetic_confinement(
    lsa,
    UCAHTS,
    p_hcd_electric_loss_mw,
    p_cryo_plant_electric_mw,
    vachtmw,
    p_tritium_plant_electric_mw,
    fachtmw,
    fkind,
):
    """Account 2262's magnetic-confinement arm --
    `calculate_auxiliary_component_cooling_cost`'s body at `ife != 1`, and the
    only arm this port has an occupant for.

    Split out of the composite above rather than left inside it because `ife` is a
    model-selection switch, and under `_audit/next_steps.md` §14.2 no switch is a static
    kwarg: the node calls this function and never sees the integer. The composite keeps
    the refusal -- it is what `functional_process/tests/models/costs/test_costs.py`
    diffs against PROCESS's own method, sample by sample, `ife` included -- and
    `machine_from_indat` refuses `ife == 1` at assembly instead, once, for all seven
    Account-22x nodes at once (`indat.py`'s `_ife_cost_accounts_arm`).

    Parameters and returns are the composite's, less `ife`.
    """
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
    return calculate_fuel_processing_cost_magnetic_confinement(
        rndfuel,
        m_fuel_amu,
        UCFPR,
        fkind,
    )


def calculate_fuel_processing_cost_magnetic_confinement(
    rndfuel,
    m_fuel_amu,
    UCFPR,
    fkind,
):
    """Account 2272's magnetic-confinement arm --
    `calculate_fuel_processing_cost`'s body at `ife != 1`, and the only arm
    this port has an occupant for.

    Split out of the composite above rather than left inside it because `ife` is a
    model-selection switch, and under `_audit/next_steps.md` §14.2 no switch is a static
    kwarg: the node calls this function and never sees the integer. The composite keeps
    the refusal -- it is what `functional_process/tests/models/costs/test_costs.py`
    diffs against PROCESS's own method, sample by sample, `ife` included -- and
    `machine_from_indat` refuses `ife == 1` at assembly instead, once, for all seven
    Account-22x nodes at once (`indat.py`'s `_ife_cost_accounts_arm`).

    Parameters and returns are the composite's, less `ife`.
    """
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
    if itart == 1:
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


def calculate_cost_of_electricity_conventional_aspect_ratio(
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
):
    """`coelc`'s conventional-aspect-ratio arm (`.physics.itart == 0`) -- the whole
    account, with **no centrepost replacement cost**.

    `costs.py:2769-2783` computes `coecp` only on a spherical tokamak; on every other
    machine it is a literal `0.0` and the three centrepost fields
    `.costs.cplife_cal`/`.cpstcst`/`.costs.cplife` are never read. Declaring them anyway
    -- which one node carrying an `itart` static kwarg had to -- invented three edges
    the reference run does not make, one of them onto `.costs.cplife`, whose only other
    reader is the `FixedPoint` that owns it (`_audit/switch_kwarg_survey.md` §4.7/§4.8).

    Parameters and returns are `calculate_cost_of_electricity`'s, less `ife`, `itart`
    and the three centrepost fields.
    """
    return _cost_of_electricity(
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
        centrepost=None,
    )


def calculate_cost_of_electricity_spherical_tokamak(
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
    """`coelc`'s spherical-tokamak arm (`.physics.itart == 1`) -- the whole account
    **plus** the centrepost replacement cost of `costs.py:2769-2783`.

    The three reads its sibling does not make are `.costs.cplife_cal`, `.costs.cpstcst`
    and `.costs.cplife`. Registered as an occupant rather than reachable, since no
    tracked input assembles a spherical tokamak's cost model yet; it exists so the
    conventional arm can drop those three reads without losing the branch PROCESS has.

    Parameters and returns are `calculate_cost_of_electricity`'s, less `ife` and `itart`.
    """
    return _cost_of_electricity(
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
        centrepost=(cplife_cal, cpstcst, cplife),
    )


def _cost_of_electricity(
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
    centrepost,
):
    """The body both `itart` arms share -- everything except the centrepost
    replacement term.

    `centrepost` is **data, not a switch**: `None` on the conventional arm, where
    PROCESS's `coecp` is a literal zero, and the `(cplife_cal, cpstcst, cplife)` triple
    on the spherical one. Which of the two a node gets follows from which arm function
    it calls, so the integer never reaches a port and the two arms' declared reads
    differ by exactly those three fields.
    """

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
    if centrepost is None:
        coecp = 0.0e0
    else:
        cplife_cal, cpstcst, cplife = centrepost
        fefcp = (1.0e0 + discount_rate) ** cplife_cal
        crfcp = (fefcp * discount_rate) / (fefcp - 1.0e0)
        anncp = cpstcst * (1.0e0 + cfind_lsa) * fcap0cp * crfcp
        anncp = jnp.where(prorate, anncp * (1.0e0 - cplife / life_plant), anncp)
        coecp = 1.0e9 * anncp / kwhpy

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
