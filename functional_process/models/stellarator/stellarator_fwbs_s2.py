"""Pure-functional port of S2 (`blanket_shield_tf_nuclear_power`), the `blktmodel` x
`ipowerflow` dispatch inside `Stellarator.st_fwbs` (registry unit #1's `stellarator.py`).

Audit record: `functional_process/models/stellarator/stellarator_fwbs_s2.md` (read it
first) -- this file only implements the two of S2's three arms that are self-contained
tier-1 and do not touch the buggy `blanket_neutronics()` call site. The third arm
(`blktmodel == 1`, i.e. `blanket_neutronics()` + its `ipowerflow`-nested tail) is
audit-only, per the record's "arm 1" section -- it calls into `hcpb.py`'s already-ported
functions through a call site with two live PROCESS bugs, and porting it would mean
choosing how to route around both, not reproducing them.

Two ported arms, both `blktmodel != 1` (`stellarator.py:680`'s `else` branch):

- **Arm 2** (`ipowerflow == 0`, `stellarator.py:684-728`, the "old model"):
  `calculate_exponential_attenuation_blanket_shield_power` below. Small, self-contained,
  no bug. The arm also calls `self.sc_tf_coil_nuclear_heating_iter90()` (chunk 1F,
  `stellarator_F_tf_nuclear_heating.py`'s `calculate_sc_tf_coil_nuclear_heating`,
  already ported) for `flu_tf_neutron_fast_peak`/`p_tf_nuclear_heat_mw` -- that call is
  a tier-3 composition edge onto an already-validated node, not reproduced here.
- **Arm 3** (`ipowerflow == 1`, `stellarator.py:730-1029`, the "new model"):
  `calculate_detailed_powerflow_blanket_shield_power` below. Self-contained (no
  cross-model calls), but two things are carved out of the port, both documented in the
  audit record: the CoolProp/`irefprop`-gated `temp_blkt_coolant_out` computation
  (803-823, not consumed anywhere else inside this arm, `non-traceable-external-call`),
  and the confirmed `p_div_rad_total_mw` bug (`.fwbs.p_div_rad_total_mw` is read at two
  sites, 792 and 1013, but never written anywhere on this call path -- deterministically
  the dataclass default `0.0` for the run's whole lifetime, per the record's "latent
  bugs" section) -- reproduced here as a literal `0.0`, not as an input, matching
  PROCESS's own actual runtime behaviour.

Both arms drop the trivial branches of two switches they read, matching the precedent
`stellarator_F_tf_nuclear_heating.py` already set for `i_tf_sup` on a sibling field (see
this file's own docstring): `i_tf_sup != SUPERCONDUCTING` and
`i_p_coolant_pumping != FRACTION_OF_HEAT` are both "the absence of the computation," not
a second formula to port -- see the audit record's "switches touched" section.
"""

import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    FromExactly,
    Output,
)


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


class ExponentialAttenuationBlanketShieldPower(ExplicitFunction):
    """cottax node: `calculate_exponential_attenuation_blanket_shield_power`.

    Not registered in `total_process.py` -- reserved for the consolidation pass, per
    this unit's boundary (see the audit record's switches-touched section: this arm is
    one of three `blktmodel`x`ipowerflow` alternatives sharing output fields with the
    other two arms, an `Alternative`/`Switch` design decision out of this audit's scope).
    """

    p_blkt_multiplication_mw = Output(lambda s: s.fwbs.p_blkt_multiplication_mw)
    p_blkt_nuclear_heat_total_mw = Output(lambda s: s.fwbs.p_blkt_nuclear_heat_total_mw)
    p_shld_nuclear_heat_mw = Output(lambda s: s.fwbs.p_shld_nuclear_heat_mw)

    def __call__(
        self,
        p_neutron_total_mw=FromExactly(lambda s: s.physics.p_neutron_total_mw),
        pnucloss=FromExactly(lambda s: s.fwbs.pnucloss),
        f_p_blkt_multiplication=FromExactly(lambda s: s.fwbs.f_p_blkt_multiplication),
        f_a_blkt_cooling_channels=FromExactly(lambda s: s.fwbs.f_a_blkt_cooling_channels),
        fblli2o=FromExactly(lambda s: s.fwbs.fblli2o),
        fblbe=FromExactly(lambda s: s.fwbs.fblbe),
        dr_blkt_outboard=FromExactly(lambda s: s.build.dr_blkt_outboard),
    ):
        return calculate_exponential_attenuation_blanket_shield_power(
            p_neutron_total_mw,
            pnucloss,
            f_p_blkt_multiplication,
            f_a_blkt_cooling_channels,
            fblli2o,
            fblbe,
            dr_blkt_outboard,
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
    """S2 arm 3 (`blktmodel != 1 & ipowerflow == 1`, `stellarator.py:730-1029`).

    The "new model": inboard/outboard power-flow accounting through first wall, blanket
    and shield, with per-component coolant pumping power. Ports the arm's arithmetic in
    full **except**:

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
      run's whole lifetime. Both read sites are reproduced below as a literal `0.0`, not
      as a function parameter, matching PROCESS's own actual behaviour exactly (not an
      approximation of it).
    - the redundant duplicate write of `.fwbs.p_fw_hcd_rad_total_mw`
      (`stellarator.py:770-780` computes the identical expression twice under two
      different comments) -- collapsed to a single computation, per
      `_audit/schema.md`'s `redundant-duplicate-write` classification.
    - `i_p_coolant_pumping != FRACTION_OF_HEAT` (`USER_INPUT`, or the two values that
      raise `ProcessValueError`) and `i_tf_sup != SUPERCONDUCTING` -- both dropped, see
      module docstring and the audit record's switches-touched section. This function
      always computes as if `i_p_coolant_pumping == FRACTION_OF_HEAT` and
      `i_tf_sup == SUPERCONDUCTING`.

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
    f_p_fw_coolant_pump_total_heat :
        FW coolant pumping power fraction. `.heat_transport.f_p_fw_coolant_pump_total_heat`.
    p_beam_orbit_loss_mw :
        NBI orbit loss power (MW). `.current_drive.p_beam_orbit_loss_mw`.
    f_p_blkt_coolant_pump_total_heat :
        Blanket coolant pumping power fraction.
        `.heat_transport.f_p_blkt_coolant_pump_total_heat`.
    f_p_blkt_multiplication :
        Blanket energy multiplication factor. `.fwbs.f_p_blkt_multiplication`.
    declshld :
        Shield power decay length (m). `.fwbs.declshld`.
    dr_shld_inboard, dr_shld_outboard :
        Shield thicknesses (m). `.build.dr_shld_inboard`/`dr_shld_outboard`.
    f_p_shld_coolant_pump_total_heat :
        Shield coolant pumping power fraction.
        `.heat_transport.f_p_shld_coolant_pump_total_heat`.
    p_plasma_separatrix_mw :
        Power to the separatrix (MW). `.physics.p_plasma_separatrix_mw`.
    f_p_div_coolant_pump_total_heat :
        Divertor coolant pumping power fraction.
        `.heat_transport.f_p_div_coolant_pump_total_heat`.

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

    # Unconditional in the source (not gated by i_p_coolant_pumping) -- two sequential
    # accumulations onto the same field, combined here into one expression.
    p_blkt_multiplication_mw = (
        f_p_blkt_coolant_pump_total_heat
        * (pnucbzi * f_p_blkt_multiplication + pnucbzo)
        * (f_p_blkt_multiplication - 1.0)
        + (pnucbzi + pnucbzo) * (f_p_blkt_multiplication - 1.0)
    )

    p_fw_nuclear_heat_total_mw = p_fw_inboard_nuclear_heat_mw + p_fw_outboard_nuclear_heat_mw
    p_blkt_nuclear_heat_total_mw = (pnucbzi + pnucbzo) * f_p_blkt_multiplication

    pnucsi = pnucbsi - pnucbzi + (pnucloss + pradloss) * a_fw_inboard / a_fw_total
    pnucso = pnucbso - pnucbzo + (pnucloss + pradloss) * a_fw_outboard / a_fw_total

    decayshldi = declshld
    decayshldo = declshld

    pnucshldi = pnucsi * (1.0 - jnp.exp(-dr_shld_inboard / decayshldi))
    pnucshldo = pnucso * (1.0 - jnp.exp(-dr_shld_outboard / decayshldo))

    p_shld_nuclear_heat_mw = pnucshldi + pnucshldo

    # FRACTION_OF_HEAT branch only.
    p_shld_coolant_pump_mw = f_p_shld_coolant_pump_total_heat * (pnucshldi + pnucshldo)
    # Second read site of the buggy `p_div_rad_total_mw` field -- see docstring.
    p_div_coolant_pump_mw = f_p_div_coolant_pump_total_heat * (
        p_plasma_separatrix_mw + p_div_nuclear_heat_total_mw + p_div_rad_total_mw
    )

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
        p_fw_coolant_pump_mw,
        p_blkt_coolant_pump_mw,
        p_shld_nuclear_heat_mw,
        p_shld_coolant_pump_mw,
        p_div_coolant_pump_mw,
        p_tf_nuclear_heat_mw,
    )


class DetailedPowerflowBlanketShieldPower(ExplicitFunction):
    """cottax node: `calculate_detailed_powerflow_blanket_shield_power`.

    `f_a_fw_coolant_inboard`/`f_a_fw_coolant_outboard` are given best-effort `VarPath`s
    under `.fwbs.*` (matching their PROCESS field names) even though the source never
    actually writes them there in this arm (they stay Python-locals, consumed by S4
    within the same call frame) -- same treatment `stellarator_F_tf_nuclear_heating.py`
    gives its own best-effort output paths, flagged here for whoever wires S4.

    Not registered in `total_process.py` -- same reservation as the sibling arm above.
    """

    p_div_nuclear_heat_total_mw = Output(lambda s: s.fwbs.p_div_nuclear_heat_total_mw)
    p_fw_hcd_nuclear_heat_mw = Output(lambda s: s.fwbs.p_fw_hcd_nuclear_heat_mw)
    p_fw_hcd_rad_total_mw = Output(lambda s: s.fwbs.p_fw_hcd_rad_total_mw)
    pradloss = Output(lambda s: s.fwbs.pradloss)
    p_fw_rad_total_mw = Output(lambda s: s.fwbs.p_fw_rad_total_mw)
    f_a_fw_coolant_inboard = Output(lambda s: s.fwbs.f_a_fw_coolant_inboard)
    f_a_fw_coolant_outboard = Output(lambda s: s.fwbs.f_a_fw_coolant_outboard)
    p_fw_nuclear_heat_total_mw = Output(lambda s: s.fwbs.p_fw_nuclear_heat_total_mw)
    p_blkt_multiplication_mw = Output(lambda s: s.fwbs.p_blkt_multiplication_mw)
    p_blkt_nuclear_heat_total_mw = Output(lambda s: s.fwbs.p_blkt_nuclear_heat_total_mw)
    p_fw_coolant_pump_mw = Output(lambda s: s.heat_transport.p_fw_coolant_pump_mw)
    p_blkt_coolant_pump_mw = Output(lambda s: s.heat_transport.p_blkt_coolant_pump_mw)
    p_shld_nuclear_heat_mw = Output(lambda s: s.fwbs.p_shld_nuclear_heat_mw)
    p_shld_coolant_pump_mw = Output(lambda s: s.heat_transport.p_shld_coolant_pump_mw)
    p_div_coolant_pump_mw = Output(lambda s: s.heat_transport.p_div_coolant_pump_mw)
    p_tf_nuclear_heat_mw = Output(lambda s: s.fwbs.p_tf_nuclear_heat_mw)

    def __call__(
        self,
        p_neutron_total_mw=FromExactly(lambda s: s.physics.p_neutron_total_mw),
        f_ster_div_single=FromExactly(lambda s: s.fwbs.f_ster_div_single),
        f_a_fw_outboard_hcd=FromExactly(lambda s: s.fwbs.f_a_fw_outboard_hcd),
        pnucloss=FromExactly(lambda s: s.fwbs.pnucloss),
        a_fw_inboard=FromExactly(lambda s: s.first_wall.a_fw_inboard),
        a_fw_outboard=FromExactly(lambda s: s.first_wall.a_fw_outboard),
        a_fw_total=FromExactly(lambda s: s.first_wall.a_fw_total),
        p_plasma_rad_mw=FromExactly(lambda s: s.physics.p_plasma_rad_mw),
        fhole=FromExactly(lambda s: s.fwbs.fhole),
        dr_fw_inboard=FromExactly(lambda s: s.build.dr_fw_inboard),
        dr_fw_outboard=FromExactly(lambda s: s.build.dr_fw_outboard),
        radius_fw_channel=FromExactly(lambda s: s.fwbs.radius_fw_channel),
        declfw=FromExactly(lambda s: s.fwbs.declfw),
        dr_blkt_inboard=FromExactly(lambda s: s.build.dr_blkt_inboard),
        dr_blkt_outboard=FromExactly(lambda s: s.build.dr_blkt_outboard),
        declblkt=FromExactly(lambda s: s.fwbs.declblkt),
        f_p_fw_coolant_pump_total_heat=FromExactly(
            lambda s: s.heat_transport.f_p_fw_coolant_pump_total_heat
        ),
        p_beam_orbit_loss_mw=FromExactly(lambda s: s.current_drive.p_beam_orbit_loss_mw),
        f_p_blkt_coolant_pump_total_heat=FromExactly(
            lambda s: s.heat_transport.f_p_blkt_coolant_pump_total_heat
        ),
        f_p_blkt_multiplication=FromExactly(lambda s: s.fwbs.f_p_blkt_multiplication),
        declshld=FromExactly(lambda s: s.fwbs.declshld),
        dr_shld_inboard=FromExactly(lambda s: s.build.dr_shld_inboard),
        dr_shld_outboard=FromExactly(lambda s: s.build.dr_shld_outboard),
        f_p_shld_coolant_pump_total_heat=FromExactly(
            lambda s: s.heat_transport.f_p_shld_coolant_pump_total_heat
        ),
        p_plasma_separatrix_mw=FromExactly(lambda s: s.physics.p_plasma_separatrix_mw),
        f_p_div_coolant_pump_total_heat=FromExactly(
            lambda s: s.heat_transport.f_p_div_coolant_pump_total_heat
        ),
    ):
        return calculate_detailed_powerflow_blanket_shield_power(
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
        )
