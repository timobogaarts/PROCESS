"""Harness cases for the ported plasma-profile arithmetic (registry unit #12).

Every reference adapter here has to solve the same problem: the methods under test live
on `PlasmaProfile`, whose two branch entry points call `self.neprofile.run()` /
`self.teprofile.run()` for effect before doing any arithmetic. Those calls belong to
`process/models/physics/profiles.py` -- a **separate, unaudited, unported unit** that the
audit record's scope correction adds to the registry -- and they overwrite the very
profile arrays a case needs to control.

So `_plasma_profile` stubs those two `run()` methods to no-ops and sets `profile_y`
directly. That is the point of the exercise rather than a dodge: it is exactly the
`data`-back-door closure the audit demands, performed against real PROCESS objects, and
it is what proves the ported functions depend on the profile *arrays* and nothing else
about the profile objects. If a ported function secretly needed something else off
`neprofile`, these cases would fail rather than quietly agreeing.
"""

import numpy as np
from cottax.interfaces.pytree_namespace_module import resolve, to_graph
from cottax.spec import VarPath

from functional_process._harness import Tier1Contract, legacy_sample
from functional_process.cottax.physics.plasma_profiles import (
    PedestalProfileValues,
    calculate_ion_vol_avg_temperature,
    calculate_parabolic_gradient_lengths,
    calculate_parabolic_profile_values,
    calculate_pedestal_profile_values,
    calculate_profile_factors,
    lmode_profile_reset,
)
from functional_process.paths import divertor, physics
from process.core.exceptions import ProcessValueError
from process.core.model import DataStructure
from process.models.physics.plasma_profiles import PlasmaProfile
from process.models.physics.profiles import NeProfile, TeProfile

# 11 points -> 10 intervals (even), so `scipy.integrate.simpson` uses the plain
# composite rule, the one `_simpson_uniform` implements. PROCESS's own default is 201;
# 11 keeps `jacfwd` over the array arguments cheap without changing which rule applies.
_N_POINTS = 11
_RHO = np.linspace(0.0, 1.0, _N_POINTS)


def _plasma_profile(ne_profile_y=None, te_profile_y=None, **physics):
    """A real `PlasmaProfile` with its two sub-model `run()`s disabled.

    See the module docstring. `profile_x` is set on both objects because
    `pedestal_parameterisation` reads it off `neprofile` only, but a case that got them
    out of step would be testing something PROCESS never does.
    """
    ne_profile, te_profile = NeProfile(), TeProfile()
    data = DataStructure()

    for obj in (ne_profile, te_profile):
        obj.data = data
        obj.run = lambda: None
        obj.profile_x = _RHO.copy()
        obj.profile_dx = _RHO[1] - _RHO[0]

    if ne_profile_y is not None:
        ne_profile.profile_y = np.asarray(ne_profile_y, dtype=float)
    if te_profile_y is not None:
        te_profile.profile_y = np.asarray(te_profile_y, dtype=float)

    model = PlasmaProfile(ne_profile, te_profile)
    model.data = data
    for name, value in physics.items():
        setattr(data.physics, name, value)
    return model


def _reference_ion_vol_avg_temperature(
    f_temp_plasma_ion_electron,
    temp_plasma_electron_vol_avg_kev,
    temp_plasma_ion_vol_avg_kev,
):
    """`parameterise_plasma`'s conditional ion-temperature write.

    Calls the real method, which then dispatches into a full branch -- harmless here,
    since only the one field is read back, and both sub-model `run()`s are stubbed.
    `i_plasma_pedestal = 0` picks the parabolic arm, the cheaper of the two.
    """
    model = _plasma_profile(
        ne_profile_y=np.ones(_N_POINTS),
        te_profile_y=np.ones(_N_POINTS),
        i_plasma_pedestal=0,
        f_temp_plasma_ion_electron=f_temp_plasma_ion_electron,
        temp_plasma_electron_vol_avg_kev=temp_plasma_electron_vol_avg_kev,
        temp_plasma_ion_vol_avg_kev=temp_plasma_ion_vol_avg_kev,
        alphan=1.0,
        alphat=1.5,
        nd_plasma_electrons_vol_avg=1.0e20,
        nd_plasma_ions_total_vol_avg=9.0e19,
        # Unused by the field read back, but `parameterise_plasma` runs the whole
        # parabolic branch including `calculate_profile_factors`, which divides by
        # `a_plasma_poloidal`. Left at its 0.0 default it warns on every sample.
        a_plasma_poloidal=35.0,
        alphaj=2.0,
    )
    model.parameterise_plasma()
    return model.data.physics.temp_plasma_ion_vol_avg_kev


def _reference_parabolic_profile_values(
    alphan,
    alphat,
    nd_plasma_electrons_vol_avg,
    temp_plasma_electron_vol_avg_kev,
    temp_plasma_ion_vol_avg_kev,
):
    """`parabolic_parameterisation`'s arithmetic tail, read back off `data`."""
    model = _plasma_profile(
        ne_profile_y=np.ones(_N_POINTS),
        te_profile_y=np.ones(_N_POINTS),
        i_plasma_pedestal=0,
        alphan=alphan,
        alphat=alphat,
        nd_plasma_electrons_vol_avg=nd_plasma_electrons_vol_avg,
        temp_plasma_electron_vol_avg_kev=temp_plasma_electron_vol_avg_kev,
        temp_plasma_ion_vol_avg_kev=temp_plasma_ion_vol_avg_kev,
        nd_plasma_ions_total_vol_avg=0.9 * nd_plasma_electrons_vol_avg,
    )
    model.parabolic_parameterisation()
    physics = model.data.physics
    return (
        physics.f_temp_plasma_electron_density_vol_avg,
        physics.nd_plasma_electron_line,
        physics.temp_plasma_electron_line_avg_kev,
        physics.temp_plasma_electron_density_weighted_kev,
        physics.temp_plasma_ion_density_weighted_kev,
    )


def _reference_pedestal_profile_values(
    profile_x,
    ne_profile_y,
    te_profile_y,
    ne_profile_integ,
    te_profile_integ,
    temp_plasma_ion_vol_avg_kev,
    temp_plasma_electron_vol_avg_kev,
    nd_plasma_separatrix_electron,
    nd_plasma_electrons_vol_avg,
):
    """`pedestal_parameterisation`, read back off `data` (including `.divertor.prn1`).

    `ne_profile_integ`/`te_profile_integ` are set as the profile objects' `profile_integ`
    attributes, since the source copies them straight through rather than computing them.
    """
    model = _plasma_profile(
        ne_profile_y=ne_profile_y,
        te_profile_y=te_profile_y,
        i_plasma_pedestal=1,
        temp_plasma_ion_vol_avg_kev=temp_plasma_ion_vol_avg_kev,
        temp_plasma_electron_vol_avg_kev=temp_plasma_electron_vol_avg_kev,
        nd_plasma_separatrix_electron=nd_plasma_separatrix_electron,
        nd_plasma_electrons_vol_avg=nd_plasma_electrons_vol_avg,
    )
    model.neprofile.profile_x = np.asarray(profile_x, dtype=float)
    model.teprofile.profile_x = np.asarray(profile_x, dtype=float)
    model.neprofile.profile_integ = ne_profile_integ
    model.teprofile.profile_integ = te_profile_integ

    model.pedestal_parameterisation()
    physics = model.data.physics
    return (
        physics.temp_plasma_electron_density_weighted_kev,
        physics.temp_plasma_ion_density_weighted_kev,
        physics.f_temp_plasma_electron_density_vol_avg,
        physics.nd_plasma_electron_line,
        physics.temp_plasma_electron_line_avg_kev,
        model.data.divertor.prn1,
    )


def _reference_lmode_profile_reset(
    radius_plasma_pedestal_temp_norm,
    radius_plasma_pedestal_density_norm,
    temp_plasma_pedestal_kev,
    temp_plasma_separatrix_kev,
    nd_plasma_pedestal_electron,
    nd_plasma_separatrix_electron,
    tbeta,
):
    """`parabolic_parameterisation`'s L-mode reset, read back off `data`.

    Runs the *whole* branch, not just the `if` block, which is the point: the reset is
    an ordinary side effect of taking the parabolic arm and this adapter asserts nothing
    about where inside that arm it happens. The remaining `physics` values are the ones
    the branch's arithmetic tail needs so that the call completes; none of them is read
    back.
    """
    model = _plasma_profile(
        ne_profile_y=np.ones(_N_POINTS),
        te_profile_y=np.ones(_N_POINTS),
        i_plasma_pedestal=0,
        radius_plasma_pedestal_temp_norm=radius_plasma_pedestal_temp_norm,
        radius_plasma_pedestal_density_norm=radius_plasma_pedestal_density_norm,
        temp_plasma_pedestal_kev=temp_plasma_pedestal_kev,
        temp_plasma_separatrix_kev=temp_plasma_separatrix_kev,
        nd_plasma_pedestal_electron=nd_plasma_pedestal_electron,
        nd_plasma_separatrix_electron=nd_plasma_separatrix_electron,
        tbeta=tbeta,
        alphan=1.0,
        alphat=1.45,
        nd_plasma_electrons_vol_avg=8.0e19,
        nd_plasma_ions_total_vol_avg=7.2e19,
        temp_plasma_electron_vol_avg_kev=12.0,
        temp_plasma_ion_vol_avg_kev=11.0,
    )
    model.parabolic_parameterisation()
    physics = model.data.physics
    return (
        physics.radius_plasma_pedestal_temp_norm,
        physics.radius_plasma_pedestal_density_norm,
        physics.temp_plasma_pedestal_kev,
        physics.temp_plasma_separatrix_kev,
        physics.nd_plasma_pedestal_electron,
        physics.nd_plasma_separatrix_electron,
        physics.tbeta,
    )


def _reference_profile_factors(
    ne_profile_y,
    te_profile_y,
    nd_plasma_electron_on_axis,
    temp_plasma_electron_on_axis_kev,
    nd_plasma_ions_on_axis,
    temp_plasma_ion_on_axis_kev,
    nd_plasma_ions_total_vol_avg,
    nd_plasma_electrons_vol_avg,
    nd_plasma_fuel_ions_vol_avg,
    f_temp_plasma_ion_electron,
    temp_plasma_electron_density_weighted_kev,
    temp_plasma_ion_density_weighted_kev,
    alphan,
    alphat,
    alphaj,
    plasma_current,
    a_plasma_poloidal,
):
    """`calculate_profile_factors`, read back off `data`."""
    model = _plasma_profile(
        ne_profile_y=ne_profile_y,
        te_profile_y=te_profile_y,
        nd_plasma_electron_on_axis=nd_plasma_electron_on_axis,
        temp_plasma_electron_on_axis_kev=temp_plasma_electron_on_axis_kev,
        nd_plasma_ions_on_axis=nd_plasma_ions_on_axis,
        temp_plasma_ion_on_axis_kev=temp_plasma_ion_on_axis_kev,
        nd_plasma_ions_total_vol_avg=nd_plasma_ions_total_vol_avg,
        nd_plasma_electrons_vol_avg=nd_plasma_electrons_vol_avg,
        nd_plasma_fuel_ions_vol_avg=nd_plasma_fuel_ions_vol_avg,
        f_temp_plasma_ion_electron=f_temp_plasma_ion_electron,
        temp_plasma_electron_density_weighted_kev=(
            temp_plasma_electron_density_weighted_kev
        ),
        temp_plasma_ion_density_weighted_kev=temp_plasma_ion_density_weighted_kev,
        alphan=alphan,
        alphat=alphat,
        alphaj=alphaj,
        plasma_current=plasma_current,
        a_plasma_poloidal=a_plasma_poloidal,
    )
    model.calculate_profile_factors()
    physics = model.data.physics
    return (
        physics.pres_plasma_thermal_on_axis,
        physics.pres_plasma_electron_profile,
        physics.pres_plasma_ion_total_profile,
        physics.pres_plasma_thermal_total_profile,
        physics.pres_plasma_fuel_profile,
        physics.alphap,
        physics.pres_plasma_thermal_vol_avg,
        physics.j_plasma_on_axis,
    )


def _reference_parabolic_gradient_lengths(
    alphat,
    alphan,
    temp_plasma_electron_on_axis_kev,
    nd_plasma_electron_on_axis,
    rminor,
):
    """`calculate_parabolic_profile_factors`, read back off `data`.

    `i_plasma_pedestal = 0` is set because the source method re-checks it internally --
    the dead guard the port drops (only one caller, in the parabolic branch). Setting it
    keeps the reference on the path the port models.
    """
    model = _plasma_profile(
        i_plasma_pedestal=0,
        alphat=alphat,
        alphan=alphan,
        temp_plasma_electron_on_axis_kev=temp_plasma_electron_on_axis_kev,
        nd_plasma_electron_on_axis=nd_plasma_electron_on_axis,
        rminor=rminor,
    )
    model.calculate_parabolic_profile_factors()
    return (
        model.data.physics.gradient_length_te,
        model.data.physics.gradient_length_ne,
    )


class TestIonVolAvgTemperature(Tier1Contract):
    """`parameterise_plasma`'s conditional write -> `calculate_ion_vol_avg_temperature`.

    Both arms are sampled: a positive ratio (the field is computed) and a zero one (the
    input passes through untouched). The zero-ratio case is the one that would catch a
    port that unconditionally multiplied.
    """

    audit_record = "models/physics/plasma_profiles.md"
    reference = _reference_ion_vol_avg_temperature
    ported = calculate_ion_vol_avg_temperature

    samples = [
        legacy_sample(
            "ratio-active",
            f_temp_plasma_ion_electron=0.9,
            temp_plasma_electron_vol_avg_kev=12.0,
            temp_plasma_ion_vol_avg_kev=1.0,
        ),
        legacy_sample(
            "ratio-zero-passthrough",
            f_temp_plasma_ion_electron=0.0,
            temp_plasma_electron_vol_avg_kev=12.0,
            temp_plasma_ion_vol_avg_kev=11.5,
        ),
    ]

    fuzz_bounds = {
        "f_temp_plasma_ion_electron": (0.1, 1.5),
        "temp_plasma_electron_vol_avg_kev": (1.0, 40.0),
        "temp_plasma_ion_vol_avg_kev": (1.0, 40.0),
    }


class TestParabolicProfileValues(Tier1Contract):
    """`parabolic_parameterisation`'s tail -> `calculate_parabolic_profile_values`."""

    audit_record = "models/physics/plasma_profiles.md"
    reference = _reference_parabolic_profile_values
    ported = calculate_parabolic_profile_values

    samples = [
        legacy_sample(
            "parabolic-reference-point",
            alphan=1.0,
            alphat=1.45,
            nd_plasma_electrons_vol_avg=8.0e19,
            temp_plasma_electron_vol_avg_kev=12.0,
            temp_plasma_ion_vol_avg_kev=11.0,
        ),
    ]

    # `alphan`/`alphat` bounds are PROCESS's own (iteration variables 6 and 5); the
    # densities and temperatures are given plausible operating ranges since the
    # volume-averaged forms are not themselves iteration variables.
    fuzz_bounds = {
        "alphan": (0.1, 1.0),
        "alphat": (0.5, 2.5),
        "nd_plasma_electrons_vol_avg": (1.0e19, 2.0e20),
        "temp_plasma_electron_vol_avg_kev": (1.0, 40.0),
        "temp_plasma_ion_vol_avg_kev": (1.0, 40.0),
    }


class TestLModeProfileReset(Tier1Contract):
    """`parabolic_parameterisation`'s L-mode reset -> `lmode_profile_reset`.

    The claim under test is that the reset's post-condition is **unconditional**: the
    seven fields hold `(1, 1, 0, 0, 0, 0, 2)` on exit whatever they held on entry, so the
    `if` guard governs only the `logger.error`. The samples take the guard both ways --
    `reset-fires` is the reference run's own cold input file, `already-l-mode` is the
    values PROCESS's converged `DataStructure` carries, where the guard is false and the
    body never runs -- and the fuzz sweep covers all seven arguments at once. A port that
    quietly kept an incoming value would fail `reset-fires`; one that reset only when the
    guard fires would still pass every sample, because both branches leave the same
    state, which is exactly why the port drops the guard.
    """

    audit_record = "models/physics/plasma_profiles.md"
    reference = _reference_lmode_profile_reset
    ported = lmode_profile_reset

    samples = [
        legacy_sample(
            # `tests/regression/input_files/stellarator_helias.IN.DAT` after
            # `init_process` and before any model has run: four of the seven differ from
            # their L-mode values, so PROCESS's guard fires.
            "reset-fires",
            radius_plasma_pedestal_temp_norm=1.0,
            radius_plasma_pedestal_density_norm=1.0,
            temp_plasma_pedestal_kev=1.0,
            temp_plasma_separatrix_kev=0.1,
            nd_plasma_pedestal_electron=4.0e19,
            nd_plasma_separatrix_electron=3.0e19,
            tbeta=2.0,
        ),
        legacy_sample(
            "already-l-mode",
            radius_plasma_pedestal_temp_norm=1.0,
            radius_plasma_pedestal_density_norm=1.0,
            temp_plasma_pedestal_kev=0.0,
            temp_plasma_separatrix_kev=0.0,
            nd_plasma_pedestal_electron=0.0,
            nd_plasma_separatrix_electron=0.0,
            tbeta=2.0,
        ),
    ]

    # PROCESS's own input ranges for the pedestal fields; the bounds matter only in that
    # they must straddle the L-mode values so both sides of the guard are sampled.
    fuzz_bounds = {
        "radius_plasma_pedestal_temp_norm": (0.8, 1.0),
        "radius_plasma_pedestal_density_norm": (0.8, 1.0),
        "temp_plasma_pedestal_kev": (0.0, 6.0),
        "temp_plasma_separatrix_kev": (0.0, 0.5),
        "nd_plasma_pedestal_electron": (0.0, 8.0e19),
        "nd_plasma_separatrix_electron": (0.0, 6.0e19),
        "tbeta": (1.0, 2.0),
    }


class TestPedestalProfileValues(Tier1Contract):
    """`pedestal_parameterisation` -> `calculate_pedestal_profile_values`.

    The array arguments exercise the harness's component-wise gradient support: `rho`,
    `dens` and `temp` are each 11 entries, differentiated one at a time against PROCESS's
    own finite difference.
    """

    audit_record = "models/physics/plasma_profiles.md"
    reference = _reference_pedestal_profile_values
    ported = calculate_pedestal_profile_values

    samples = [
        legacy_sample(
            "pedestal-reference-point",
            profile_x=_RHO,
            ne_profile_y=8.0e19 * (1.0 - 0.8 * _RHO**2),
            te_profile_y=12.0 * (1.0 - 0.9 * _RHO**2) + 0.5,
            ne_profile_integ=7.2e19,
            te_profile_integ=9.4,
            temp_plasma_ion_vol_avg_kev=11.0,
            temp_plasma_electron_vol_avg_kev=12.0,
            nd_plasma_separatrix_electron=2.0e19,
            nd_plasma_electrons_vol_avg=8.0e19,
        ),
        legacy_sample(
            # `prn1` is floored at 0.01; this point sits below the floor so the
            # `jnp.maximum` is the branch actually taken.
            "pedestal-prn1-floored",
            profile_x=_RHO,
            ne_profile_y=8.0e19 * (1.0 - 0.8 * _RHO**2),
            te_profile_y=12.0 * (1.0 - 0.9 * _RHO**2) + 0.5,
            ne_profile_integ=7.2e19,
            te_profile_integ=9.4,
            temp_plasma_ion_vol_avg_kev=11.0,
            temp_plasma_electron_vol_avg_kev=12.0,
            nd_plasma_separatrix_electron=1.0e17,
            nd_plasma_electrons_vol_avg=8.0e19,
        ),
    ]


def test_pedestal_profile_values_assembles_alone():
    """`PedestalProfileValues` must assemble as its own one-node graph, the same
    minimal structural check `test_composition.py` runs for `PlasmaComposition`.
    """
    graph = to_graph(PedestalProfileValues())
    assert graph.definitions


def test_pedestal_profile_values_owns_cross_area_prn1():
    """`.divertor.prn1` is this unit's only cross-area write (audit record's open
    question 5). Checking `to_graph` succeeds is only half the proof -- a node could
    assemble while silently declaring `prn1` as a read, or omitting it, or landing it
    under `.physics` by a naming slip. This checks the exact `VarPath` the node owns
    lands in `.divertor`, alongside the four fields `_audit/tokamak_boundary.md` §
    "The four that are a shared subsystem's gap" lists, which must land in `.physics`.
    """
    node = PedestalProfileValues()
    owned = {out.var for out in node.outputs}
    read = {inp.var for inp in node.inputs}

    prn1_path = resolve(divertor.prn1, VarPath)
    assert prn1_path in owned
    assert prn1_path not in read

    for name in (
        "f_temp_plasma_electron_density_vol_avg",
        "nd_plasma_electron_line",
        "temp_plasma_electron_density_weighted_kev",
        "temp_plasma_ion_density_weighted_kev",
        "temp_plasma_electron_line_avg_kev",
    ):
        path = resolve(getattr(physics, name), VarPath)
        assert path in owned, name
        assert path not in read, name


class TestProfileFactors(Tier1Contract):
    """`calculate_profile_factors` -> the same, with the profile arrays as arguments.

    Four of the eight outputs are arrays, so this case also checks `_as_array`'s
    flattening: reference and port return the same leaves in the same order or the
    shape assertion fires.
    """

    audit_record = "models/physics/plasma_profiles.md"
    reference = _reference_profile_factors
    ported = calculate_profile_factors

    samples = [
        legacy_sample(
            "profile-factors-reference-point",
            ne_profile_y=8.0e19 * (1.0 - 0.8 * _RHO**2),
            te_profile_y=12.0 * (1.0 - 0.9 * _RHO**2) + 0.5,
            nd_plasma_electron_on_axis=1.6e20,
            temp_plasma_electron_on_axis_kev=29.4,
            nd_plasma_ions_on_axis=1.44e20,
            temp_plasma_ion_on_axis_kev=27.0,
            nd_plasma_ions_total_vol_avg=7.2e19,
            nd_plasma_electrons_vol_avg=8.0e19,
            nd_plasma_fuel_ions_vol_avg=6.0e19,
            f_temp_plasma_ion_electron=0.9,
            temp_plasma_electron_density_weighted_kev=13.5,
            temp_plasma_ion_density_weighted_kev=12.2,
            alphan=1.0,
            alphat=1.45,
            alphaj=2.0,
            plasma_current=1.5e7,
            a_plasma_poloidal=35.0,
        ),
    ]


class TestParabolicGradientLengths(Tier1Contract):
    """`calculate_parabolic_profile_factors` -> `calculate_parabolic_gradient_lengths`.

    Both live arms of the three-way index branch are sampled -- `alpha > 1` (the analytic
    steepest point) and `0 < alpha <= 1` (the 'boxy' 0.9 approximation) -- including one
    point that takes a *different* arm for temperature than for density, which is the
    case a port sharing one branch decision between the two would get wrong.

    `reference_domain_errors` covers the third arm: PROCESS raises `ProcessValueError`
    on a non-positive index, and the port returns NaN there instead.
    """

    audit_record = "models/physics/plasma_profiles.md"
    reference = _reference_parabolic_gradient_lengths
    ported = calculate_parabolic_gradient_lengths
    reference_domain_errors = (ProcessValueError,)

    samples = [
        legacy_sample(
            "both-steep",
            alphat=1.45,
            alphan=1.2,
            temp_plasma_electron_on_axis_kev=29.4,
            nd_plasma_electron_on_axis=1.6e20,
            rminor=1.8,
        ),
        legacy_sample(
            "both-boxy",
            alphat=0.6,
            alphan=0.4,
            temp_plasma_electron_on_axis_kev=29.4,
            nd_plasma_electron_on_axis=1.6e20,
            rminor=1.8,
        ),
        legacy_sample(
            "steep-temperature-boxy-density",
            alphat=2.0,
            alphan=0.5,
            temp_plasma_electron_on_axis_kev=29.4,
            nd_plasma_electron_on_axis=1.6e20,
            rminor=1.8,
        ),
    ]

    fuzz_bounds = {
        "alphat": (0.1, 2.5),
        "alphan": (0.1, 2.5),
        "temp_plasma_electron_on_axis_kev": (5.0, 60.0),
        "nd_plasma_electron_on_axis": (1.0e19, 3.0e20),
        "rminor": (0.5, 4.0),
    }
