"""Pure physics functions extracted from
`functional_process.models.stellarator.stellarator_fwbs_s2`, per `_audit/formulas_split.md`
step 2 phase A. The graph declarations that wrap these stay in that sibling
module.
"""

import jax.numpy as jnp


def calculate_exponential_attenuation_blanket_shield_power(
    p_neutron_total_mw,
    pnucloss,
    f_p_blkt_multiplication,
    f_a_blkt_cooling_channels,
    fblli2o,
    fblbe,
    dr_blkt_outboard,
):
    """S2 arm 2 (`blktmodel != 1 & ipowerflow == 0`, `stellarator.py:684-715`).

    The "old model": one exponential-attenuation formula for blanket nuclear heating,
    shield heating as the remainder. Ports `stellarator.py:686-714` verbatim (the arm's
    own local arithmetic only -- the arm's tail, `self.sc_tf_coil_nuclear_heating_iter90()`
    at 716-728, is a separate already-ported node, see module docstring).

    Parameters
    ----------
    p_neutron_total_mw :
        Total neutron power (MW). `.physics.p_neutron_total_mw`.
    pnucloss :
        Neutron power lost through first-wall holes (MW). `.fwbs.pnucloss` -- an S1
        (`fw_blanket_shield_geometry_setup`) output, unconditionally
        `p_neutron_total_mw * fhole` on every real call, but read here as an ordinary
        upstream value, not re-derived.

    Note: the source also reads `.fwbs.pnuc_cp` here, but `stellarator.py:681` sets it
    to the literal `0.0` unconditionally, in the same straight-line scope, immediately
    before either `blktmodel != 1` arm runs -- `local-intermediate` per
    `_audit/schema.md`, not a real upstream edge, so it is inlined as `0.0` below rather
    than exposed as a parameter (see the audit record).
    f_p_blkt_multiplication :
        Blanket energy multiplication factor. `.fwbs.f_p_blkt_multiplication`.
    f_a_blkt_cooling_channels :
        Blanket coolant channel void fraction. `.fwbs.f_a_blkt_cooling_channels`.
    fblli2o, fblbe :
        Li2O / beryllium blanket volume fractions. `.fwbs.fblli2o`, `.fwbs.fblbe`.
    dr_blkt_outboard :
        Outboard blanket thickness (m). `.build.dr_blkt_outboard`.

    Returns
    -------
    :
        `(p_blkt_multiplication_mw, p_blkt_nuclear_heat_total_mw, p_shld_nuclear_heat_mw)`.
    """
    pnuc_cp = 0.0  # local-intermediate, see docstring
    pneut2 = (p_neutron_total_mw - pnucloss - pnuc_cp) * f_p_blkt_multiplication
    p_blkt_multiplication_mw = pneut2 - (p_neutron_total_mw - pnucloss - pnuc_cp)

    decaybl = 0.075 / (1.0 - f_a_blkt_cooling_channels - fblli2o - fblbe)
    p_blkt_nuclear_heat_total_mw = pneut2 * (1.0 - jnp.exp(-dr_blkt_outboard / decaybl))

    p_shld_nuclear_heat_mw = pneut2 - p_blkt_nuclear_heat_total_mw

    return p_blkt_multiplication_mw, p_blkt_nuclear_heat_total_mw, p_shld_nuclear_heat_mw


def _detailed_powerflow_core(
    p_neutron_total_mw,
    f_ster_div_single,
    f_a_fw_outboard_hcd,
    pnucloss,
    a_fw_inboard,
    a_fw_outboard,
    a_fw_total,
    p_plasma_rad_mw,
    fhole,
    dr_fw_inboard,
    dr_fw_outboard,
    radius_fw_channel,
    declfw,
    dr_blkt_inboard,
    dr_blkt_outboard,
    declblkt,
    f_p_blkt_coolant_pump_total_heat,
    f_p_blkt_multiplication,
    declshld,
    dr_shld_inboard,
    dr_shld_outboard,
):
    """S2 arm 3's `i_p_coolant_pumping`-independent arithmetic
    (`blktmodel != 1 & ipowerflow == 1`, `stellarator.py:730-1029`).

    The "new model": inboard/outboard power-flow accounting through first wall, blanket
    and shield. **Everything in the arm except the four coolant-pumping powers**, which
    `stellarator.py:901-928`/`:996-1013` gate on `.fwbs.i_p_coolant_pumping` and which
    the two public wrappers below own one arm each. Not a node itself -- the two
    wrappers are what `indat.py`'s `BLANKET_SHIELD_POWER` registry holds.

    Ports the arm's arithmetic in full **except**:

    - the CoolProp/`irefprop`-gated `temp_blkt_coolant_out` block (`stellarator.py:
      803-823`) -- excluded entirely (not computed, not returned). It is not read again
      anywhere else in this arm (confirmed by direct read of 824-1029); its only readers
      are outside `st_fwbs` (`power.py`, `blankets/blanket_library.py`), out of S2's
      scope. `non-traceable-external-call` (CoolProp), see the audit record.
    - **the confirmed `p_div_rad_total_mw` bug, reproduced not fixed**:
      `.fwbs.p_div_rad_total_mw` is read at two sites in this arm (`stellarator.py:792`,
      feeding `p_fw_rad_total_mw`; and `stellarator.py:1013`, feeding
      `p_div_coolant_pump_mw` -- the second site is a new finding, not flagged by
      `stellarator_E_fwbs_synthesis.md` section 6) but is never written anywhere on this
      call path (`blktmodel != 1` means `blanket_neutronics()`, the only other writer,
      never runs) -- its value is deterministically the dataclass default `0.0` for the
      run's whole lifetime. Both read sites are reproduced as a literal `0.0`, not
      as a function parameter, matching PROCESS's own actual behaviour exactly (not an
      approximation of it). The first site is here; the second is in the
      `FRACTION_OF_HEAT` wrapper, the only arm that reaches it.
    - the redundant duplicate write of `.fwbs.p_fw_hcd_rad_total_mw`
      (`stellarator.py:770-780` computes the identical expression twice under two
      different comments) -- collapsed to a single computation, per
      `_audit/schema.md`'s `redundant-duplicate-write` classification.
    - `i_tf_sup != SUPERCONDUCTING` -- dropped, see module docstring and the audit
      record's switches-touched section. This function always computes
      `p_tf_nuclear_heat_mw` as if `i_tf_sup == SUPERCONDUCTING`.

    Parameters
    ----------
    p_neutron_total_mw :
        Total neutron power (MW). `.physics.p_neutron_total_mw`.
    f_ster_div_single :
        Divertor solid-angle fraction. `.fwbs.f_ster_div_single` (an S1/`Divertor`
        output, upstream of S2).
    f_a_fw_outboard_hcd :
        HCD-apparatus area fraction. `.fwbs.f_a_fw_outboard_hcd`.
    pnucloss :
        Neutron power lost through holes (MW). `.fwbs.pnucloss` (S1 output).

    Note: as in the sibling arm, `.fwbs.pnuc_cp` is also read here but is a
    `local-intermediate` (`stellarator.py:681` sets it to the literal `0.0`
    unconditionally, in the same straight-line scope, immediately before this arm runs)
    -- inlined as `0.0` below, not exposed as a parameter.
    a_fw_inboard, a_fw_outboard, a_fw_total :
        First-wall areas (m2). `.first_wall.a_fw_inboard`/`a_fw_outboard`/`a_fw_total`
        (S1 outputs; S1 always sets the first two to exactly half of the third, but this
        arm reads all three as independent values, matching the source).
    p_plasma_rad_mw :
        Plasma radiated power (MW). `.physics.p_plasma_rad_mw`.
    fhole :
        First-wall hole area fraction. `.fwbs.fhole`.
    dr_fw_inboard, dr_fw_outboard :
        First-wall thicknesses (m). `.build.dr_fw_inboard`/`dr_fw_outboard`.
    radius_fw_channel :
        FW coolant channel radius (m). `.fwbs.radius_fw_channel`.
    declfw :
        First-wall power decay length (m). `.fwbs.declfw`.
    dr_blkt_inboard, dr_blkt_outboard :
        Blanket thicknesses (m). `.build.dr_blkt_inboard`/`dr_blkt_outboard`.
    declblkt :
        Blanket power decay length (m). `.fwbs.declblkt`.
    f_p_blkt_coolant_pump_total_heat :
        Blanket coolant pumping power fraction.
        `.heat_transport.f_p_blkt_coolant_pump_total_heat`. Read here even though no
        pumping power is computed here: `stellarator.py:929-933`'s
        `p_blkt_multiplication_mw` accumulation is **outside** the
        `i_p_coolant_pumping` dispatch and reads it unconditionally, on every arm.
    f_p_blkt_multiplication :
        Blanket energy multiplication factor. `.fwbs.f_p_blkt_multiplication`.
    declshld :
        Shield power decay length (m). `.fwbs.declshld`.
    dr_shld_inboard, dr_shld_outboard :
        Shield thicknesses (m). `.build.dr_shld_inboard`/`dr_shld_outboard`.

    Returns
    -------
    :
        The twelve arm outputs that do not depend on `i_p_coolant_pumping` --
        `(p_div_nuclear_heat_total_mw, p_fw_hcd_nuclear_heat_mw, p_fw_hcd_rad_total_mw,
        pradloss, p_fw_rad_total_mw, f_a_fw_coolant_inboard, f_a_fw_coolant_outboard,
        p_fw_nuclear_heat_total_mw, p_blkt_multiplication_mw,
        p_blkt_nuclear_heat_total_mw, p_shld_nuclear_heat_mw, p_tf_nuclear_heat_mw)` --
        followed by the eight source locals the pumping formulas need
        (`p_fw_inboard_nuclear_heat_mw`, `p_fw_outboard_nuclear_heat_mw`, `psurffwi`,
        `psurffwo`, `pnucbzi`, `pnucbzo`, `pnucshldi`, `pnucshldo`). The trailing eight
        are Python locals in the source, not `data` fields; they cross this boundary
        only because the function was split at the switch, and neither wrapper exposes
        them.
    """
    pnuc_cp = 0.0  # local-intermediate, see docstring
    p_div_nuclear_heat_total_mw = p_neutron_total_mw * f_ster_div_single
    p_fw_hcd_nuclear_heat_mw = p_neutron_total_mw * f_a_fw_outboard_hcd
    pnucfwbs = (
        p_neutron_total_mw
        - p_div_nuclear_heat_total_mw
        - pnucloss
        - pnuc_cp
        - p_fw_hcd_nuclear_heat_mw
    )
    pnucfwbsi = pnucfwbs * a_fw_inboard / a_fw_total
    pnucfwbso = pnucfwbs * a_fw_outboard / a_fw_total

    # Redundant-duplicate-write in the source (identical expression, two comments) --
    # collapsed to one computation.
    p_fw_hcd_rad_total_mw = p_plasma_rad_mw * f_a_fw_outboard_hcd
    pradloss = p_plasma_rad_mw * fhole

    # `.fwbs.p_div_rad_total_mw` bug: never written on this call path, deterministically
    # 0.0. Reproduced as a literal, not a parameter -- see docstring.
    p_div_rad_total_mw = 0.0
    p_fw_rad_total_mw = (
        p_plasma_rad_mw - p_div_rad_total_mw - pradloss - p_fw_hcd_rad_total_mw
    )

    bfwi = 0.5 * dr_fw_inboard
    bfwo = 0.5 * dr_fw_outboard

    f_a_fw_coolant_inboard = radius_fw_channel * radius_fw_channel / (bfwi * bfwi)
    f_a_fw_coolant_outboard = radius_fw_channel * radius_fw_channel / (bfwo * bfwo)

    decayfwi = declfw
    decayfwo = declfw

    psurffwi = p_fw_rad_total_mw * a_fw_inboard / a_fw_total
    psurffwo = p_fw_rad_total_mw * a_fw_outboard / a_fw_total

    p_fw_inboard_nuclear_heat_mw = pnucfwbsi * (1.0 - jnp.exp(-2.0 * bfwi / decayfwi))
    p_fw_outboard_nuclear_heat_mw = pnucfwbso * (1.0 - jnp.exp(-2.0 * bfwo / decayfwo))

    pnucbsi = pnucfwbsi - p_fw_inboard_nuclear_heat_mw
    pnucbso = pnucfwbso - p_fw_outboard_nuclear_heat_mw

    decaybzi = declblkt
    decaybzo = declblkt

    pnucbzi = pnucbsi * (1.0 - jnp.exp(-dr_blkt_inboard / decaybzi))
    pnucbzo = pnucbso * (1.0 - jnp.exp(-dr_blkt_outboard / decaybzo))

    # Unconditional in the source (not gated by i_p_coolant_pumping) -- two sequential
    # accumulations onto the same field, combined here into one expression.
    p_blkt_multiplication_mw = f_p_blkt_coolant_pump_total_heat * (
        pnucbzi * f_p_blkt_multiplication + pnucbzo
    ) * (f_p_blkt_multiplication - 1.0) + (pnucbzi + pnucbzo) * (
        f_p_blkt_multiplication - 1.0
    )

    p_fw_nuclear_heat_total_mw = (
        p_fw_inboard_nuclear_heat_mw + p_fw_outboard_nuclear_heat_mw
    )
    p_blkt_nuclear_heat_total_mw = (pnucbzi + pnucbzo) * f_p_blkt_multiplication

    pnucsi = pnucbsi - pnucbzi + (pnucloss + pradloss) * a_fw_inboard / a_fw_total
    pnucso = pnucbso - pnucbzo + (pnucloss + pradloss) * a_fw_outboard / a_fw_total

    decayshldi = declshld
    decayshldo = declshld

    pnucshldi = pnucsi * (1.0 - jnp.exp(-dr_shld_inboard / decayshldi))
    pnucshldo = pnucso * (1.0 - jnp.exp(-dr_shld_outboard / decayshldo))

    p_shld_nuclear_heat_mw = pnucshldi + pnucshldo

    # SUPERCONDUCTING branch only.
    p_tf_nuclear_heat_mw = pnucsi + pnucso - pnucshldi - pnucshldo

    return (
        p_div_nuclear_heat_total_mw,
        p_fw_hcd_nuclear_heat_mw,
        p_fw_hcd_rad_total_mw,
        pradloss,
        p_fw_rad_total_mw,
        f_a_fw_coolant_inboard,
        f_a_fw_coolant_outboard,
        p_fw_nuclear_heat_total_mw,
        p_blkt_multiplication_mw,
        p_blkt_nuclear_heat_total_mw,
        p_shld_nuclear_heat_mw,
        p_tf_nuclear_heat_mw,
        p_fw_inboard_nuclear_heat_mw,
        p_fw_outboard_nuclear_heat_mw,
        psurffwi,
        psurffwo,
        pnucbzi,
        pnucbzo,
        pnucshldi,
        pnucshldo,
    )


def calculate_detailed_powerflow_blanket_shield_power(
    p_neutron_total_mw,
    f_ster_div_single,
    f_a_fw_outboard_hcd,
    pnucloss,
    a_fw_inboard,
    a_fw_outboard,
    a_fw_total,
    p_plasma_rad_mw,
    fhole,
    dr_fw_inboard,
    dr_fw_outboard,
    radius_fw_channel,
    declfw,
    dr_blkt_inboard,
    dr_blkt_outboard,
    declblkt,
    f_p_fw_coolant_pump_total_heat,
    p_beam_orbit_loss_mw,
    f_p_blkt_coolant_pump_total_heat,
    f_p_blkt_multiplication,
    declshld,
    dr_shld_inboard,
    dr_shld_outboard,
    f_p_shld_coolant_pump_total_heat,
    p_plasma_separatrix_mw,
    f_p_div_coolant_pump_total_heat,
):
    """S2 arm 3 at `i_p_coolant_pumping == FRACTION_OF_HEAT` (1).

    `_detailed_powerflow_core` plus the four pumping powers PROCESS computes on this
    arm and this arm only (`stellarator.py:907-923` for the first-wall and blanket pair,
    `:1000-1013` for the shield and divertor pair): each is a fixed fraction `fpump_i`
    of the non-pumping thermal power reaching that component's coolant.

    **The arithmetic below is bit-for-bit what this function computed before the
    `i_p_coolant_pumping` split** -- the switch used to be dropped as "the absence of a
    computation," which was true only of the files that existed then; see the module
    docstring.

    Parameters
    ----------
    f_p_fw_coolant_pump_total_heat :
        FW coolant pumping power fraction. `.heat_transport.f_p_fw_coolant_pump_total_heat`.
    p_beam_orbit_loss_mw :
        NBI orbit loss power (MW). `.current_drive.p_beam_orbit_loss_mw`.
    f_p_shld_coolant_pump_total_heat :
        Shield coolant pumping power fraction.
        `.heat_transport.f_p_shld_coolant_pump_total_heat`.
    p_plasma_separatrix_mw :
        Power to the separatrix (MW). `.physics.p_plasma_separatrix_mw`.
    f_p_div_coolant_pump_total_heat :
        Divertor coolant pumping power fraction.
        `.heat_transport.f_p_div_coolant_pump_total_heat`.

    Every other parameter is `_detailed_powerflow_core`'s; see its docstring.

    Returns
    -------
    :
        `(p_div_nuclear_heat_total_mw, p_fw_hcd_nuclear_heat_mw, p_fw_hcd_rad_total_mw,
        pradloss, p_fw_rad_total_mw, f_a_fw_coolant_inboard, f_a_fw_coolant_outboard,
        p_fw_nuclear_heat_total_mw, p_blkt_multiplication_mw,
        p_blkt_nuclear_heat_total_mw, p_fw_coolant_pump_mw, p_blkt_coolant_pump_mw,
        p_shld_nuclear_heat_mw, p_shld_coolant_pump_mw, p_div_coolant_pump_mw,
        p_tf_nuclear_heat_mw)`. `f_a_fw_coolant_inboard`/`f_a_fw_coolant_outboard` are
        Python locals in the source (never written to `.fwbs.*` in this arm -- see the
        audit record's cross-boundary ledger), consumed downstream by S4's FW-mass
        `else` branch, out of this unit's scope.
    """
    (
        p_div_nuclear_heat_total_mw,
        p_fw_hcd_nuclear_heat_mw,
        p_fw_hcd_rad_total_mw,
        pradloss,
        p_fw_rad_total_mw,
        f_a_fw_coolant_inboard,
        f_a_fw_coolant_outboard,
        p_fw_nuclear_heat_total_mw,
        p_blkt_multiplication_mw,
        p_blkt_nuclear_heat_total_mw,
        p_shld_nuclear_heat_mw,
        p_tf_nuclear_heat_mw,
        p_fw_inboard_nuclear_heat_mw,
        p_fw_outboard_nuclear_heat_mw,
        psurffwi,
        psurffwo,
        pnucbzi,
        pnucbzo,
        pnucshldi,
        pnucshldo,
    ) = _detailed_powerflow_core(
        p_neutron_total_mw,
        f_ster_div_single,
        f_a_fw_outboard_hcd,
        pnucloss,
        a_fw_inboard,
        a_fw_outboard,
        a_fw_total,
        p_plasma_rad_mw,
        fhole,
        dr_fw_inboard,
        dr_fw_outboard,
        radius_fw_channel,
        declfw,
        dr_blkt_inboard,
        dr_blkt_outboard,
        declblkt,
        f_p_blkt_coolant_pump_total_heat,
        f_p_blkt_multiplication,
        declshld,
        dr_shld_inboard,
        dr_shld_outboard,
    )

    # FRACTION_OF_HEAT branch only -- see docstring/module docstring.
    p_fw_coolant_pump_mw = f_p_fw_coolant_pump_total_heat * (
        p_fw_inboard_nuclear_heat_mw
        + p_fw_outboard_nuclear_heat_mw
        + psurffwi
        + psurffwo
        + p_beam_orbit_loss_mw
    )
    p_blkt_coolant_pump_mw = f_p_blkt_coolant_pump_total_heat * (
        pnucbzi * f_p_blkt_multiplication + pnucbzo * f_p_blkt_multiplication
    )

    p_shld_coolant_pump_mw = f_p_shld_coolant_pump_total_heat * (pnucshldi + pnucshldo)
    # Second read site of the buggy `p_div_rad_total_mw` field -- see the core's
    # docstring. Reached on this arm only.
    p_div_rad_total_mw = 0.0
    p_div_coolant_pump_mw = f_p_div_coolant_pump_total_heat * (
        p_plasma_separatrix_mw + p_div_nuclear_heat_total_mw + p_div_rad_total_mw
    )

    return (
        p_div_nuclear_heat_total_mw,
        p_fw_hcd_nuclear_heat_mw,
        p_fw_hcd_rad_total_mw,
        pradloss,
        p_fw_rad_total_mw,
        f_a_fw_coolant_inboard,
        f_a_fw_coolant_outboard,
        p_fw_nuclear_heat_total_mw,
        p_blkt_multiplication_mw,
        p_blkt_nuclear_heat_total_mw,
        p_fw_coolant_pump_mw,
        p_blkt_coolant_pump_mw,
        p_shld_nuclear_heat_mw,
        p_shld_coolant_pump_mw,
        p_div_coolant_pump_mw,
        p_tf_nuclear_heat_mw,
    )


def calculate_detailed_powerflow_blanket_shield_power_user_input_pumping(
    p_neutron_total_mw,
    f_ster_div_single,
    f_a_fw_outboard_hcd,
    pnucloss,
    a_fw_inboard,
    a_fw_outboard,
    a_fw_total,
    p_plasma_rad_mw,
    fhole,
    dr_fw_inboard,
    dr_fw_outboard,
    radius_fw_channel,
    declfw,
    dr_blkt_inboard,
    dr_blkt_outboard,
    declblkt,
    f_p_blkt_coolant_pump_total_heat,
    f_p_blkt_multiplication,
    declshld,
    dr_shld_inboard,
    dr_shld_outboard,
):
    """S2 arm 3 at `i_p_coolant_pumping == USER_INPUT` (0).

    Exactly `_detailed_powerflow_core`, and that is the whole content of the arm:
    `stellarator.py:904-906` is

    ```
    if i_p_coolant_pumping == PumpingPowerModelTypes.USER_INPUT:
        #   Use input
        pass
    ```

    and the second dispatch (`:1000`) is an `if FRACTION_OF_HEAT` with no `else`, so
    the shield and divertor pumping powers are not written either. All four of
    `.heat_transport.p_fw_coolant_pump_mw`, `p_blkt_coolant_pump_mw`,
    `p_shld_coolant_pump_mw` and `p_div_coolant_pump_mw` keep the values the input file
    gave them -- **boundary inputs on this arm, not outputs of any node**. That is why
    this is a second occupant rather than a `jnp.where` inside the first: a node that
    owns a `VarPath` on one switch value and must not own it on another is two nodes
    (`_audit/next_steps.md` §14.2), and declaring the ownership unconditionally is what
    made this port answer `15.6 MW` where `helias_5b.IN.DAT` says `176.0`.

    Parameters are `_detailed_powerflow_core`'s twenty-one; the five that only feed the
    pumping formulas (`f_p_fw_coolant_pump_total_heat`, `p_beam_orbit_loss_mw`,
    `f_p_shld_coolant_pump_total_heat`, `p_plasma_separatrix_mw`,
    `f_p_div_coolant_pump_total_heat`) are genuinely unread on this arm and are
    therefore absent from the signature rather than accepted and discarded.

    Returns
    -------
    :
        `(p_div_nuclear_heat_total_mw, p_fw_hcd_nuclear_heat_mw, p_fw_hcd_rad_total_mw,
        pradloss, p_fw_rad_total_mw, f_a_fw_coolant_inboard, f_a_fw_coolant_outboard,
        p_fw_nuclear_heat_total_mw, p_blkt_multiplication_mw,
        p_blkt_nuclear_heat_total_mw, p_shld_nuclear_heat_mw, p_tf_nuclear_heat_mw)` --
        the `FRACTION_OF_HEAT` sibling's sixteen minus the four pumping powers, in the
        same relative order.
    """
    return _detailed_powerflow_core(
        p_neutron_total_mw,
        f_ster_div_single,
        f_a_fw_outboard_hcd,
        pnucloss,
        a_fw_inboard,
        a_fw_outboard,
        a_fw_total,
        p_plasma_rad_mw,
        fhole,
        dr_fw_inboard,
        dr_fw_outboard,
        radius_fw_channel,
        declfw,
        dr_blkt_inboard,
        dr_blkt_outboard,
        declblkt,
        f_p_blkt_coolant_pump_total_heat,
        f_p_blkt_multiplication,
        declshld,
        dr_shld_inboard,
        dr_shld_outboard,
    )[:12]
