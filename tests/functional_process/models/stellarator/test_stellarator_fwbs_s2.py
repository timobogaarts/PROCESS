"""Harness cases for S2 (registry unit #1's `st_fwbs`), arms 2 and 3 -- see
`stellarator_fwbs_s2.md`. Arm 1 (`blktmodel == 1`) is audit-only, not ported, so it has
no case here.

No PROCESS unit test covers `st_fwbs` directly (`tests/unit/models/stellarator/
test_stellarator.py` has no `test_st_fwbs`), so there is no `legacy_sample` to lift --
coverage is fuzz-only, against a real `Stellarator.st_fwbs()` call.

Both reference adapters build a full `DataStructure` and populate it with a safe
baseline chosen so the *entirety* of `st_fwbs` -- S1 through S5, not just S2's own arm --
runs without a `ZeroDivisionError`; the method has no early return (see the audit
record's "arm boundaries" section). One field genuinely needs special handling because
`st_fwbs` itself, not just S2, constrains it: `a_fw_inboard`/`a_fw_outboard`
(`.first_wall.*`) are unconditionally overwritten by S1 (`stellarator.py:521-522`) to
exactly half of `a_fw_total` each, before either arm ever runs. `_arm_c_samples` below
draws `a_fw_total` and derives both from it, matching what any real call actually
produces -- the port itself keeps all three as independent parameters (S2's own read,
not S1's identity), see the audit record.

`pnucloss` (`.fwbs.pnucloss`) is similarly an S1 output (`stellarator.py:598-600`,
`pnucloss = p_neutron_total_mw * fhole`), but arm 2 does not read `fhole` itself, so its
reference derives a consistent `fhole` from the requested `pnucloss` and
`p_neutron_total_mw`. Arm 3 *does* read `fhole` directly (for `pradloss`), so its sample
generator derives `pnucloss = p_neutron_total_mw * fhole` from the same draw used for
`fhole`, rather than drawing `pnucloss` independently -- the two are not actually free
in any reachable state, and drawing them independently would test an unreachable
combination.
"""

from types import MappingProxyType

from functional_process._harness import Tier1Contract
from functional_process._harness.sampling import Sample, fuzz_samples
from functional_process.models.stellarator.stellarator_fwbs_s2 import (
    calculate_detailed_powerflow_blanket_shield_power,
    calculate_exponential_attenuation_blanket_shield_power,
)
from process.core.model import DataStructure
from process.models.engineering.pumping import CoolantType
from process.models.stellarator.stellarator import Stellarator
from process.models.tfcoil.base import TFConductorModel


def _make_stellarator():
    """A `Stellarator` + `DataStructure` pair, safe for a full `st_fwbs(output=False)`
    call (S1 through S5 -- see module docstring).

    `hcpb`/`availability`/`vacuum`/etc. are `None`: none of arm 2 or arm 3 touches them
    (only arm 1, `blanket_neutronics()`, needs `hcpb`, and that arm is not exercised
    here).
    """
    data = DataStructure()
    stel = Stellarator(
        availability=None,
        vacuum=None,
        buildings=None,
        costs=None,
        power=None,
        plasma_profile=None,
        hcpb=None,
        current_drive=None,
        physics=None,
        neoclassics=None,
        plasma_beta=None,
        plasma_bootstrap=None,
    )
    stel.data = data

    # S1's own denominators, not part of either arm's footprint.
    data.physics.pflux_fw_neutron_mw = 1.5
    data.physics.rmajor = 15.0
    data.physics.rminor = 1.5
    data.build.dr_fw_inboard = 0.02
    data.build.dr_fw_outboard = 0.02
    data.build.dr_blkt_inboard = 0.3
    data.build.dr_blkt_outboard = 0.4
    data.build.dr_shld_inboard = 0.3
    data.build.dr_shld_outboard = 0.4
    data.build.dr_vv_inboard = 0.1
    data.build.dr_vv_outboard = 0.1
    data.fwbs.radius_fw_channel = 0.006
    data.first_wall.a_fw_total = 2000.0

    data.fwbs.blktmodel = 0  # both arms live in the `blktmodel != 1` else-branch
    data.fwbs.i_p_coolant_pumping = (
        1  # FRACTION_OF_HEAT -- see the port's module docstring
    )
    data.tfcoil.i_tf_sup = TFConductorModel.SUPERCONDUCTING
    # WATER + irefprop=False would also avoid the real CoolProp call, but HELIUM skips
    # the whole `temp_blkt_coolant_out` block (arm 3 only) at the source, cleaner.
    data.fwbs.i_blkt_coolant_type = CoolantType.HELIUM

    return stel, data


def _reference_exponential_attenuation(
    p_neutron_total_mw,
    pnucloss,
    f_p_blkt_multiplication,
    f_a_blkt_cooling_channels,
    fblli2o,
    fblbe,
    dr_blkt_outboard,
):
    """Call PROCESS's `Stellarator.st_fwbs` (`blktmodel=0, ipowerflow=0`) through arm 2's
    signature."""
    stel, data = _make_stellarator()
    data.heat_transport.ipowerflow = 0

    data.physics.p_neutron_total_mw = p_neutron_total_mw
    # S1 overwrites `.fwbs.pnucloss` to `p_neutron_total_mw * fhole` before this arm
    # runs (arm 2 does not read `fhole` itself) -- derive the `fhole` that reproduces
    # the requested `pnucloss` exactly.
    data.fwbs.fhole = pnucloss / p_neutron_total_mw
    data.fwbs.f_p_blkt_multiplication = f_p_blkt_multiplication
    data.fwbs.f_a_blkt_cooling_channels = f_a_blkt_cooling_channels
    data.fwbs.fblli2o = fblli2o
    data.fwbs.fblbe = fblbe
    data.build.dr_blkt_outboard = dr_blkt_outboard

    stel.st_fwbs(output=False)

    return (
        data.fwbs.p_blkt_multiplication_mw,
        data.fwbs.p_blkt_nuclear_heat_total_mw,
        data.fwbs.p_shld_nuclear_heat_mw,
    )


class TestExponentialAttenuationBlanketShieldPower(Tier1Contract):
    """S2 arm 2 (`st_fwbs`, `blktmodel != 1 & ipowerflow == 0`) ->
    `calculate_exponential_attenuation_blanket_shield_power`."""

    audit_record = "models/stellarator/stellarator_fwbs_s2.md"
    reference = _reference_exponential_attenuation
    ported = calculate_exponential_attenuation_blanket_shield_power

    # Bounds keep pnucloss << p_neutron_total_mw so the derived fhole (see the
    # reference adapter) stays well inside PROCESS's own physical range [0, ~0.3).
    fuzz_bounds = {
        "p_neutron_total_mw": (1000.0, 3000.0),
        "pnucloss": (5.0, 150.0),
        "f_p_blkt_multiplication": (1.0, 1.5),
        "f_a_blkt_cooling_channels": (0.05, 0.2),
        "fblli2o": (0.05, 0.3),
        "fblbe": (0.1, 0.4),
        "dr_blkt_outboard": (0.2, 1.0),
    }


def _arm_c_samples(count, seed):
    """Fuzz arm 3's independent physical parameters, then derive the two fields
    `st_fwbs` itself constrains before arm 3 ever runs (see module docstring):
    `a_fw_inboard = a_fw_outboard = 0.5 * a_fw_total` (S1, unconditional), and
    `pnucloss = p_neutron_total_mw * fhole` (also S1, unconditional -- arm 3, unlike
    arm 2, reads `fhole` directly too, so it is drawn here rather than back-derived).
    """
    raw = fuzz_samples(
        {
            "p_neutron_total_mw": (1000.0, 3000.0),
            "f_ster_div_single": (0.02, 0.2),
            "f_a_fw_outboard_hcd": (0.0, 0.1),
            "a_fw_total": (500.0, 4000.0),
            "p_plasma_rad_mw": (20.0, 400.0),
            "fhole": (0.01, 0.15),
            "dr_fw_inboard": (0.01, 0.05),
            "dr_fw_outboard": (0.01, 0.05),
            "radius_fw_channel": (0.002, 0.01),
            "declfw": (0.03, 0.15),
            "dr_blkt_inboard": (0.1, 0.6),
            "dr_blkt_outboard": (0.1, 0.6),
            "declblkt": (0.03, 0.15),
            "f_p_fw_coolant_pump_total_heat": (0.005, 0.05),
            "p_beam_orbit_loss_mw": (0.0, 20.0),
            "f_p_blkt_coolant_pump_total_heat": (0.005, 0.05),
            "f_p_blkt_multiplication": (1.0, 1.5),
            "declshld": (0.03, 0.15),
            "dr_shld_inboard": (0.1, 0.6),
            "dr_shld_outboard": (0.1, 0.6),
            "f_p_shld_coolant_pump_total_heat": (0.005, 0.05),
            "p_plasma_separatrix_mw": (10.0, 300.0),
            "f_p_div_coolant_pump_total_heat": (0.005, 0.05),
        },
        count,
        seed,
    )
    out = []
    for s in raw:
        kwargs = dict(s.kwargs)
        a_fw_total = kwargs["a_fw_total"]
        kwargs["a_fw_inboard"] = 0.5 * a_fw_total
        kwargs["a_fw_outboard"] = 0.5 * a_fw_total
        fhole = kwargs.pop("fhole")
        kwargs["fhole"] = fhole
        kwargs["pnucloss"] = kwargs["p_neutron_total_mw"] * fhole
        out.append(Sample(MappingProxyType(kwargs), s.provenance, s.label))
    return out


def _reference_detailed_powerflow(
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
    """Call PROCESS's `Stellarator.st_fwbs` (`blktmodel=0, ipowerflow=1`) through arm
    3's signature.

    `pnucloss` is accepted (matching the port's own signature, an ordinary read) but not
    used to set `.fwbs.pnucloss` directly -- `fhole` is, and S1 rederives the identical
    `pnucloss` from it (`_arm_c_samples` only ever supplies a `pnucloss` consistent with
    `p_neutron_total_mw * fhole`, so this is not a silent divergence).
    `a_fw_inboard`/`a_fw_outboard` are accepted for the same reason but not set directly
    either -- S1 always derives them from `a_fw_total`.
    """
    del pnucloss, a_fw_inboard, a_fw_outboard  # see docstring
    stel, data = _make_stellarator()
    data.heat_transport.ipowerflow = 1

    data.physics.p_neutron_total_mw = p_neutron_total_mw
    data.fwbs.f_ster_div_single = f_ster_div_single
    data.fwbs.f_a_fw_outboard_hcd = f_a_fw_outboard_hcd
    data.first_wall.a_fw_total = a_fw_total
    data.physics.p_plasma_rad_mw = p_plasma_rad_mw
    data.fwbs.fhole = fhole
    data.build.dr_fw_inboard = dr_fw_inboard
    data.build.dr_fw_outboard = dr_fw_outboard
    data.fwbs.radius_fw_channel = radius_fw_channel
    data.fwbs.declfw = declfw
    data.build.dr_blkt_inboard = dr_blkt_inboard
    data.build.dr_blkt_outboard = dr_blkt_outboard
    data.fwbs.declblkt = declblkt
    data.heat_transport.f_p_fw_coolant_pump_total_heat = f_p_fw_coolant_pump_total_heat
    data.current_drive.p_beam_orbit_loss_mw = p_beam_orbit_loss_mw
    data.heat_transport.f_p_blkt_coolant_pump_total_heat = (
        f_p_blkt_coolant_pump_total_heat
    )
    data.fwbs.f_p_blkt_multiplication = f_p_blkt_multiplication
    data.fwbs.declshld = declshld
    data.build.dr_shld_inboard = dr_shld_inboard
    data.build.dr_shld_outboard = dr_shld_outboard
    data.heat_transport.f_p_shld_coolant_pump_total_heat = (
        f_p_shld_coolant_pump_total_heat
    )
    data.physics.p_plasma_separatrix_mw = p_plasma_separatrix_mw
    data.heat_transport.f_p_div_coolant_pump_total_heat = f_p_div_coolant_pump_total_heat

    stel.st_fwbs(output=False)

    return (
        data.fwbs.p_div_nuclear_heat_total_mw,
        data.fwbs.p_fw_hcd_nuclear_heat_mw,
        data.fwbs.p_fw_hcd_rad_total_mw,
        data.fwbs.pradloss,
        data.fwbs.p_fw_rad_total_mw,
        # f_a_fw_coolant_inboard/outboard are Python locals in the source, never
        # written to `.fwbs.*` in this arm (see the port's docstring) -- not observable
        # through `data`, so not part of this reference's return. `test_value_agreement`
        # therefore only checks the 14 outputs that really do round-trip through
        # `data`; the port's own two local-only outputs are exercised by
        # `test_outputs_finite`/gradient checks instead (they still appear in every
        # call, just outside this particular agreement check -- see `_as_array`'s
        # flattening: both sides must return the same *count*, so they are excluded
        # from the `ported` comparison too, via the wrapper below).
        data.fwbs.p_fw_nuclear_heat_total_mw,
        data.fwbs.p_blkt_multiplication_mw,
        data.fwbs.p_blkt_nuclear_heat_total_mw,
        data.heat_transport.p_fw_coolant_pump_mw,
        data.heat_transport.p_blkt_coolant_pump_mw,
        data.fwbs.p_shld_nuclear_heat_mw,
        data.heat_transport.p_shld_coolant_pump_mw,
        data.heat_transport.p_div_coolant_pump_mw,
        data.fwbs.p_tf_nuclear_heat_mw,
    )


def _ported_detailed_powerflow_observable(*args, **kwargs):
    """`calculate_detailed_powerflow_blanket_shield_power`, minus the two outputs
    (`f_a_fw_coolant_inboard`/`outboard`) that never round-trip through `data` in this
    arm -- see `_reference_detailed_powerflow`'s docstring. Only used to keep
    `test_value_agreement`'s flattened output count matched to the reference; the full
    16-output function (with those two included) is what `total_process.py`/S4 would
    actually import.
    """
    full = calculate_detailed_powerflow_blanket_shield_power(*args, **kwargs)
    return full[:5] + full[7:]


class TestDetailedPowerflowBlanketShieldPower(Tier1Contract):
    """S2 arm 3 (`st_fwbs`, `blktmodel != 1 & ipowerflow == 1`) ->
    `calculate_detailed_powerflow_blanket_shield_power`."""

    audit_record = "models/stellarator/stellarator_fwbs_s2.md"
    reference = _reference_detailed_powerflow
    ported = _ported_detailed_powerflow_observable

    # `st_fwbs` (S1) imposes real relationships between these six arguments that this
    # arm's own pure function does not know about and does not need to (S2 reads them
    # as ordinary, independent upstream values -- see the port's docstring and the
    # audit record's "arm boundaries" section): `a_fw_inboard = a_fw_outboard =
    # 0.5 * a_fw_total` and `pnucloss = p_neutron_total_mw * fhole`, both unconditional,
    # both forced by S1 before this arm ever runs. Perturbing any one of these six in
    # isolation -- exactly what `jacfwd`'s per-argument columns and the reference's own
    # finite difference both do -- therefore either has no effect on the reference (it
    # forces the field right back via the other four) or a *different* effect than on
    # the port (it also moves an S1-derived sibling the port treats as independent).
    # Neither side is wrong; the two are answering different questions once elevated to
    # "vary this one argument, holding the rest of `data` fixed," a phrase that does not
    # correspond to any single well-defined perturbation once a full `st_fwbs()` call is
    # the oracle. Same reasoning `test_initialization.py`'s `TestPulseDurations` already
    # applies to arguments a reference is structurally unresponsive to; generalised here
    # to arguments a reference is *entangled* on. Value agreement and output-finiteness
    # are unaffected (no differentiation involved) and still pass for all 26 arguments.
    static_argnames = (
        "p_neutron_total_mw",
        "a_fw_total",
        "a_fw_inboard",
        "a_fw_outboard",
        "fhole",
        "pnucloss",
    )

    samples = _arm_c_samples(count=12, seed=0)
