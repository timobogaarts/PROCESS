"""Harness cases for the ported fusion-reaction rates (registry unit #19).

Follows `test_density_limits.py`'s shape (the project's worked example). The
`_fusion_reaction_rate` adapter mirrors `test_plasma_profiles.py`'s `_plasma_profile`
stub: a real `FusionReactionRate` bound to real `NeProfile`/`TeProfile` objects whose
`run()` is disabled and whose `profile_x`/`profile_y` are set directly, since those two
profile objects belong to a separate, still-only-partially-ported unit (#12) and this
unit's audit only needs the *array values*, not a live profile solve.

`beam_fusion`/`beam_reaction_rate_coefficient` have no cases here at all -- see
`fusion_reactions.md`'s "tier signal": they are not ported, so there is nothing to
contract-test. The rest of `beam_fusion`'s dependency chain (everything not touching
`scipy.integrate.quad`) is tested below against PROCESS's own module-level functions,
even though it has no cottax node yet.
"""

import numpy as np

from functional_process._harness import Tier1Contract, legacy_sample
from functional_process.cottax.physics.fusion_reactions import (
    alpha_power_beam,
    beam_fusion_cross_section,
    beam_slowing_down_state,
    beam_target_reaction_rate,
    bosch_hale_reactivity,
    calculate_deuterium_branching_trit,
    calculate_fusion_rates,
    fast_ion_pressure_integral,
    hot_beam_fusion_reaction_rate_integrand,
    set_fusion_powers,
)
from process.core.model import DataStructure
from process.models.physics import fusion_reactions as reactions
from process.models.physics.plasma_profiles import PlasmaProfile
from process.models.physics.profiles import NeProfile, TeProfile

# 11 points -> 10 intervals (even), matching `test_plasma_profiles.py`'s choice: the
# non-uniform composite Simpson's rule PROCESS actually uses (`_simpson`, imported from
# that unit's port) applies for an even interval count, same as PROCESS's own default of
# 201 points / 200 intervals, and keeps `jacfwd` over the array arguments cheap.
_N_POINTS = 11
_RHO = np.linspace(0.0, 1.0, _N_POINTS)


def _fusion_reaction_rate(profile_x, ne_profile_y, te_profile_y, **physics):
    """A real `FusionReactionRate` bound to stubbed `NeProfile`/`TeProfile` objects.

    Same closure-of-the-`data`-backdoor technique `test_plasma_profiles.py` already used
    for the same underlying objects: `run()` is disabled and `profile_x`/`profile_y` are
    set directly, so the reference calls exactly the arithmetic under audit and nothing
    from the (separately scoped, still-pending) profile solve.
    """
    ne_profile, te_profile = NeProfile(), TeProfile()
    data = DataStructure()

    for obj in (ne_profile, te_profile):
        obj.data = data
        obj.run = lambda: None
        obj.profile_x = np.asarray(profile_x, dtype=float)
        obj.profile_dx = obj.profile_x[1] - obj.profile_x[0]

    ne_profile.profile_y = np.asarray(ne_profile_y, dtype=float)
    te_profile.profile_y = np.asarray(te_profile_y, dtype=float)

    plasma_profile = PlasmaProfile(ne_profile, te_profile)
    plasma_profile.data = data
    for name, value in physics.items():
        setattr(data.physics, name, value)

    return reactions.FusionReactionRate(plasma_profile, data)


def _reference_deuterium_branching_trit(ion_temperature):
    """Call PROCESS's `.deuterium_branching()` through the port's signature."""
    frr = _fusion_reaction_rate(_RHO, np.ones(_N_POINTS), np.ones(_N_POINTS))
    frr.deuterium_branching(ion_temperature)
    return frr.f_dd_branching_trit


def _reference_bosch_hale_reactivity(ion_temperature_profile, reaction_constants):
    """Call PROCESS's `bosch_hale_reactivity` through the port's signature."""
    return reactions.bosch_hale_reactivity(
        np.asarray(ion_temperature_profile, dtype=float),
        reactions.BoschHaleConstants(**reaction_constants),
    )


def _reference_calculate_fusion_rates(
    profile_x,
    te_profile_y,
    ne_profile_y,
    temp_plasma_ion_vol_avg_kev,
    temp_plasma_electron_vol_avg_kev,
    f_plasma_fuel_deuterium,
    f_plasma_fuel_tritium,
    f_plasma_fuel_helium3,
    nd_plasma_fuel_ions_vol_avg,
    nd_plasma_electrons_vol_avg,
    f_dd_branching_trit,
):
    """Call PROCESS's `.calculate_fusion_rates()` + `.set_physics_variables()` together,
    matching the port's fused signature (see `fusion_reactions.md`'s "cottax node").
    """
    frr = _fusion_reaction_rate(
        profile_x,
        ne_profile_y,
        te_profile_y,
        temp_plasma_ion_vol_avg_kev=temp_plasma_ion_vol_avg_kev,
        temp_plasma_electron_vol_avg_kev=temp_plasma_electron_vol_avg_kev,
        f_plasma_fuel_deuterium=f_plasma_fuel_deuterium,
        f_plasma_fuel_tritium=f_plasma_fuel_tritium,
        f_plasma_fuel_helium3=f_plasma_fuel_helium3,
        nd_plasma_fuel_ions_vol_avg=nd_plasma_fuel_ions_vol_avg,
        nd_plasma_electrons_vol_avg=nd_plasma_electrons_vol_avg,
    )
    # `f_dd_branching_trit` is normally set by a prior `.deuterium_branching()` call on
    # the same instance; the port takes it as an explicit argument instead (see
    # `fusion_reactions.md`), so the reference sets the instance attribute directly to
    # match, rather than calling `.deuterium_branching()` again with a possibly
    # inconsistent temperature.
    frr.f_dd_branching_trit = f_dd_branching_trit
    frr.calculate_fusion_rates()
    frr.set_physics_variables()
    physics = frr.data.physics
    return (
        physics.pden_plasma_alpha_mw,
        physics.pden_non_alpha_charged_mw,
        physics.pden_plasma_neutron_mw,
        physics.fusden_plasma,
        physics.fusden_plasma_alpha,
        physics.proton_rate_density,
        physics.sigmav_dt_average,
        physics.dt_power_density_plasma,
        physics.dhe3_power_density,
        physics.dd_power_density,
        physics.f_dd_branching_trit,
        physics.fusrat_plasma_dt_profile,
        physics.fusrat_plasma_dhe3_profile,
        physics.fusrat_plasma_dd_helion_profile,
        physics.fusrat_plasma_dd_triton_profile,
    )


def _reference_set_fusion_powers(**kwargs):
    """Call PROCESS's `set_fusion_powers` through the port's signature."""
    return reactions.set_fusion_powers(**kwargs)


def _reference_alpha_power_beam(beam_target_reaction_rate_value):
    """Call PROCESS's `alpha_power_beam` through the port's (renamed) argument name."""
    return reactions.alpha_power_beam(beam_target_reaction_rate_value)


def _reference_beam_slowing_down_state(
    e_beam_kev,
    critical_energy_deuterium,
    critical_energy_tritium,
    t_beam_slow,
    f_beam_tritium,
    c_beam_total,
    vol_plasma,
):
    """Call PROCESS's `beam_slowing_down_state` through the port's signature,
    flattening its `BeamSlowingDownState` dataclass return into a plain tuple.
    """
    state = reactions.beam_slowing_down_state(
        e_beam_kev,
        critical_energy_deuterium,
        critical_energy_tritium,
        t_beam_slow,
        f_beam_tritium,
        c_beam_total,
        vol_plasma,
    )
    return (
        state.deuterium_beam_density,
        state.tritium_beam_density,
        state.deuterium_critical_energy_speed,
        state.tritium_critical_energy_speed,
        state.nd_beam_hot,
        state.e_beam_deposited_kev,
    )


class TestDeuteriumBranchingTrit(Tier1Contract):
    """`.deuterium_branching()` -> `calculate_deuterium_branching_trit`."""

    audit_record = "models/physics/fusion_reactions.md"
    reference = _reference_deuterium_branching_trit
    ported = calculate_deuterium_branching_trit

    samples = [
        legacy_sample("branching-mid-temperature", ion_temperature=11.0),
        # 55.73 keV is the ion temperature `tests/unit/models/physics/
        # test_fusion_reactions.py::test_bosch_hale` uses for the same species of
        # calculation (a different function, but a PROCESS-validated temperature value).
        legacy_sample(
            "branching-bosch-hale-reference-temperature", ion_temperature=55.73
        ),
    ]

    fuzz_bounds = {"ion_temperature": (0.5, 200.0)}


class TestBoschHaleReactivity(Tier1Contract):
    """`bosch_hale_reactivity` -> the same, unchanged.

    Samples are `tests/unit/models/physics/test_fusion_reactions.py::test_bosch_hale`'s
    four cases verbatim -- genuinely legacy, one per reaction species.
    """

    audit_record = "models/physics/fusion_reactions.md"
    reference = _reference_bosch_hale_reactivity
    ported = bosch_hale_reactivity

    static_argnames = ("reaction_constants",)

    samples = [
        legacy_sample(
            "bosch-hale-dt",
            ion_temperature_profile=np.array([55.73]),
            reaction_constants=reactions.REACTION_CONSTANTS_DT,
        ),
        legacy_sample(
            "bosch-hale-dhe3",
            ion_temperature_profile=np.array([55.73]),
            reaction_constants=reactions.REACTION_CONSTANTS_DHE3,
        ),
        legacy_sample(
            "bosch-hale-dd1",
            ion_temperature_profile=np.array([55.73]),
            reaction_constants=reactions.REACTION_CONSTANTS_DD1,
        ),
        legacy_sample(
            "bosch-hale-dd2",
            ion_temperature_profile=np.array([55.73]),
            reaction_constants=reactions.REACTION_CONSTANTS_DD2,
        ),
        # Exercises the `t == 0.0` mask branch (source: `sigmav[t_mask] = 0.0`), which
        # none of PROCESS's own test points do.
        legacy_sample(
            "bosch-hale-dt-zero-temperature",
            ion_temperature_profile=np.array([55.73, 0.0]),
            reaction_constants=reactions.REACTION_CONSTANTS_DT,
        ),
    ]

    # Array-shaped bounds, not scalar: PROCESS's own `bosch_hale_reactivity` does
    # in-place mask assignment (`sigmav[t_mask] = 0.0`), which raises on a bare scalar --
    # a scalar draw would fail the *reference* call, not the port. Shape (3,) keeps the
    # gradient check (one `jacfwd` column per component) cheap.
    fuzz_bounds = {
        "ion_temperature_profile": (np.full(3, 0.2), np.full(3, 100.0)),
    }
    fuzz_fixed = {"reaction_constants": reactions.REACTION_CONSTANTS_DT}


class TestFusionRates(Tier1Contract):
    """`calculate_fusion_rates`: `.calculate_fusion_rates()` fused with
    `.set_physics_variables()`.

    The two source methods fuse into one port function -- see `fusion_reactions.md`'s
    "cottax node" section for why. The profile arrays are the same shapes
    `test_plasma_profiles.py` already uses for its own reference-point sample --
    plausible stellarator-core-like profiles, not independently re-derived. Verified by
    hand against a live `FusionReactionRate` instance while writing the port (see
    `fusion_reactions.md`'s tier signal): every field matched PROCESS at full float64
    precision before this case existed.
    """

    audit_record = "models/physics/fusion_reactions.md"
    reference = _reference_calculate_fusion_rates
    ported = calculate_fusion_rates

    samples = [
        legacy_sample(
            "fusion-rates-reference-point",
            profile_x=_RHO,
            te_profile_y=12.0 * (1.0 - 0.9 * _RHO**2) + 0.5,
            ne_profile_y=8.0e19 * (1.0 - 0.8 * _RHO**2),
            temp_plasma_ion_vol_avg_kev=11.0,
            temp_plasma_electron_vol_avg_kev=12.0,
            f_plasma_fuel_deuterium=0.5,
            f_plasma_fuel_tritium=0.5,
            f_plasma_fuel_helium3=0.0,
            nd_plasma_fuel_ions_vol_avg=7.0e19,
            nd_plasma_electrons_vol_avg=8.0e19,
            f_dd_branching_trit=0.478,
        ),
    ]

    fuzz_bounds = {
        # Array-shaped bounds, matching `profile_x`'s fixed (11,) grid -- a scalar draw
        # would break `_simpson` (`y.shape[0]`) and PROCESS's own mask assignment alike.
        "te_profile_y": (np.full(_N_POINTS, 1.0), np.full(_N_POINTS, 30.0)),
        "ne_profile_y": (np.full(_N_POINTS, 1.0e19), np.full(_N_POINTS, 2.0e20)),
        "temp_plasma_ion_vol_avg_kev": (1.0, 40.0),
        "temp_plasma_electron_vol_avg_kev": (1.0, 40.0),
        "nd_plasma_fuel_ions_vol_avg": (1.0e19, 1.5e20),
        "nd_plasma_electrons_vol_avg": (1.0e19, 2.0e20),
        "f_dd_branching_trit": (0.3, 0.5),
    }
    # `profile_x` is a fixed grid, not fuzzed (`_simpson` requires an even interval
    # count, and PROCESS never varies the grid independent of `n_plasma_profile_elements`
    # -- see `plasma_profiles.md`); fuel fractions are held physical (sum <= 1).
    fuzz_fixed = {
        "profile_x": _RHO,
        "f_plasma_fuel_deuterium": 0.5,
        "f_plasma_fuel_tritium": 0.5,
        "f_plasma_fuel_helium3": 0.0,
    }


class TestSetFusionPowers(Tier1Contract):
    """`set_fusion_powers` -> the same, unchanged.

    Both samples are `tests/unit/models/physics/test_fusion_reactions.py::
    test_set_fusion_powers`'s parametrised cases verbatim -- genuinely legacy, one with
    no beam contribution and one with.
    """

    audit_record = "models/physics/fusion_reactions.md"
    reference = _reference_set_fusion_powers
    ported = set_fusion_powers

    samples = [
        legacy_sample(
            "set-fusion-powers-no-beam",
            f_p_alpha_plasma_deposited=0.95,
            f_alpha_electron=0.68,
            f_alpha_ion=0.32,
            p_beam_alpha_mw=0.0,
            pden_non_alpha_charged_mw=0.00066,
            vol_plasma=2426.25,
            pden_plasma_alpha_mw=0.163,
            pden_plasma_neutron_mw=0.654,
        ),
        legacy_sample(
            "set-fusion-powers-with-beam",
            f_p_alpha_plasma_deposited=0.95,
            f_alpha_electron=0.68,
            f_alpha_ion=0.32,
            p_beam_alpha_mw=100.5,
            pden_non_alpha_charged_mw=0.00066,
            vol_plasma=2426.25,
            pden_plasma_alpha_mw=0.163,
            pden_plasma_neutron_mw=0.654,
        ),
    ]

    fuzz_bounds = {
        "p_beam_alpha_mw": (0.0, 500.0),
        "pden_non_alpha_charged_mw": (1.0e-5, 1.0),
        "vol_plasma": (100.0, 5000.0),
        "pden_plasma_alpha_mw": (1.0e-3, 1.0),
        "pden_plasma_neutron_mw": (1.0e-3, 5.0),
    }
    fuzz_fixed = {
        "f_p_alpha_plasma_deposited": 0.95,
        "f_alpha_electron": 0.68,
        "f_alpha_ion": 0.32,
    }


class TestBeamSlowingDownState(Tier1Contract):
    """`beam_slowing_down_state` -> the same, tuple-flattened.

    Sample is `test_beam_slowing_down_state`'s case verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/fusion_reactions.md"
    reference = _reference_beam_slowing_down_state
    ported = beam_slowing_down_state

    samples = [
        legacy_sample(
            "beam-slowing-down-reference",
            e_beam_kev=1000.0,
            critical_energy_deuterium=276.7,
            critical_energy_tritium=415.0,
            t_beam_slow=1.42,
            f_beam_tritium=1e-06,
            c_beam_total=130.0,
            vol_plasma=1888.0,
        ),
    ]

    fuzz_bounds = {
        "e_beam_kev": (100.0, 2000.0),
        "critical_energy_deuterium": (50.0, 500.0),
        "critical_energy_tritium": (50.0, 700.0),
        "t_beam_slow": (0.1, 5.0),
        "f_beam_tritium": (0.0, 1.0),
        "c_beam_total": (10.0, 300.0),
        "vol_plasma": (100.0, 5000.0),
    }


class TestFastIonPressureIntegral(Tier1Contract):
    """`fast_ion_pressure_integral` -> the same, unchanged.

    Sample is `test__fast_ion_pressure_integral`'s case verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/fusion_reactions.md"
    reference = reactions.fast_ion_pressure_integral
    ported = fast_ion_pressure_integral

    samples = [
        legacy_sample(
            "fast-ion-pressure-reference", e_beam_kev=1000.0, critical_energy=276.7
        ),
    ]

    fuzz_bounds = {"e_beam_kev": (100.0, 2000.0), "critical_energy": (50.0, 700.0)}


class TestBeamTargetReactionRate(Tier1Contract):
    """`beam_target_reaction_rate` -> the same, unchanged.

    Sample is `test_beam_target_reaction_rate`'s case verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/fusion_reactions.md"
    reference = reactions.beam_target_reaction_rate
    ported = beam_target_reaction_rate

    samples = [
        legacy_sample(
            "beam-target-rate-reference",
            nd_beam_ion=3.16e11,
            nd_target_ion=3.3e19,
            sigv_beam=7.5e-22,
            vol_plasma=1888.0,
        ),
    ]

    fuzz_bounds = {
        "nd_beam_ion": (1.0e9, 1.0e13),
        "nd_target_ion": (1.0e18, 1.0e20),
        "sigv_beam": (1.0e-23, 1.0e-20),
        "vol_plasma": (100.0, 5000.0),
    }


class TestAlphaPowerBeam(Tier1Contract):
    """`alpha_power_beam` -> the same, unchanged (positional arg renamed for clarity;
    same value, see the port's docstring).

    Sample is `test_alpha_power_beam`'s case verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/fusion_reactions.md"
    reference = _reference_alpha_power_beam
    ported = alpha_power_beam

    samples = [
        legacy_sample(
            "alpha-power-beam-reference", beam_target_reaction_rate_value=1.0e13
        ),
    ]

    fuzz_bounds = {"beam_target_reaction_rate_value": (1.0e10, 1.0e15)}


class TestBeamFusionCrossSection(Tier1Contract):
    """`beam_fusion_cross_section` -> ports `_beam_fusion_cross_section`, renamed.

    Samples span all three arms of the clamp (`< 10 keV`, `> 1e4 keV`, and the smooth
    middle) -- constructed points, not lifted from a PROCESS test (none exercises this
    function directly), verified against `reactions._beam_fusion_cross_section` by hand
    while writing the port (see `fusion_reactions.md`). Fuzzing stays inside the smooth
    middle arm only, since the two clamp regions are locally flat (zero gradient almost
    everywhere) and the kink itself is a genuine non-smooth point in PROCESS's own
    function -- not something a wider fuzz range would usefully exercise beyond the three
    fixed samples already covering it.
    """

    audit_record = "models/physics/fusion_reactions.md"
    reference = staticmethod(reactions._beam_fusion_cross_section)
    ported = beam_fusion_cross_section

    samples = [
        # e_beam_kev = 0.5 * M_DEUTERON_AMU * vrelsq; M_DEUTERON_AMU ~= 2.0136.
        legacy_sample("cross-section-low-clamp", vrelsq=1.0),  # e_beam_kev ~ 1.0
        legacy_sample("cross-section-mid", vrelsq=5000.0),  # e_beam_kev ~ 5034
        legacy_sample("cross-section-high-clamp", vrelsq=50000.0),  # e_beam_kev ~ 50340
    ]

    fuzz_bounds = {"vrelsq": (2000.0, 8000.0)}  # e_beam_kev in [~2013, ~8054], mid arm


class TestHotBeamFusionReactionRateIntegrand(Tier1Contract):
    """`hot_beam_fusion_reaction_rate_integrand` -> ports
    `_hot_beam_fusion_reaction_rate_integrand`, renamed.

    Constructed points (no direct PROCESS unit test), verified by hand against
    `reactions._hot_beam_fusion_reaction_rate_integrand` while writing the port.
    """

    audit_record = "models/physics/fusion_reactions.md"
    reference = staticmethod(reactions._hot_beam_fusion_reaction_rate_integrand)
    ported = hot_beam_fusion_reaction_rate_integrand

    samples = [
        legacy_sample("integrand-mid", velocity_ratio=0.5, critical_velocity=1.2e6),
        legacy_sample(
            "integrand-above-one", velocity_ratio=2.0, critical_velocity=1.2e6
        ),
    ]

    fuzz_bounds = {"velocity_ratio": (0.05, 3.0), "critical_velocity": (1.0e5, 1.0e7)}
