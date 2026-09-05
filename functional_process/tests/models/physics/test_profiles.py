"""Harness cases for the ported profile arithmetic (registry unit #21).

Every reference adapter builds a **real** `NeProfile` / `TeProfile`, gives it a real
`DataStructure`, and calls the real method -- then reads the answer back off `data` or off
the object attribute PROCESS wrote it to. Nothing here is a transcription of the source.

Two things make these adapters less obvious than unit #12's:

- **`Profile.run()` is overridden.** `NeProfile.run` / `TeProfile.run` chain
  `super().run()` -> `normalise_profile_x` -> `calculate_profile_dx` ->
  `set_physics_variables` -> `calculate_profile_y` -> `integrate_profile_y`, so calling
  `run()` on the subclass would run the whole unit. Where only the grid is wanted,
  `Profile.run(obj)` is called unbound, which is exactly what `super().run()` does.
- **`calculate_profile_y` returns `None`** and writes `self.profile_y` in place, so the
  adapters set `profile_y` to zeros first and read it back afterwards. See the record's
  § open questions: six call sites in `current_drive.py` use that `None` as a number.

`i_plasma_pedestal` is a topology switch, not an argument, so it never appears in a
sample's kwargs. Where a function has two arms, each arm gets its own contract class whose
adapter pins the switch. `TestDensityProfileParabolicSwitch` is the exception that proves
the finding: it runs the *same* port against a reference pinned to the *other* switch
value, because `NeProfile.calculate_profile_y`'s parabolic branch is dead code.
"""

import numpy as np

from functional_process.cottax._harness import Tier1Contract, legacy_sample
from functional_process.cottax.physics.profiles import (
    calculate_density_profile,
    calculate_greenwald_density_fractions,
    calculate_parabolic_on_axis_densities,
    calculate_parabolic_on_axis_temperatures,
    calculate_parabolic_temperature_profile,
    calculate_pedestal_on_axis_densities,
    calculate_pedestal_on_axis_temperatures,
    calculate_pedestal_separatrix_densities,
    calculate_pedestal_temperature_profile,
    calculate_profile_grid,
    integrate_profile_y,
    ncore,
    tcore,
)
from process.core.exceptions import ProcessValueError
from process.core.model import DataStructure
from process.models.physics.profiles import NeProfile, Profile, TeProfile

_N_POINTS = 11
"""11 points -> 10 intervals (even), so `_simpson`'s composite rule applies.

PROCESS's own default is 201; 11 keeps `jacfwd` over the array arguments cheap without
changing which quadrature rule is in force.
"""

_RHO = np.linspace(0.0, 1.0, _N_POINTS)
"""The real grid, endpoint included -- what `calculate_profile_grid` produces."""

_RHO_INTERIOR = np.linspace(0.0, 0.9, _N_POINTS)
"""A grid stopping short of the axis edge.

Used only by `TestParabolicTemperatureProfileInterior`. At `rho == 1` the parabolic form's
base `1 - rho**2` is exactly zero, and PROCESS's finite difference steps `rho` to 1.001,
where `(-0.002) ** 1.45` is NaN in `numpy` -- PROCESS leaving its own domain, not a
disagreement with the port. Differentiating on an interior grid checks `d/d(rho[i])`
honestly instead.
"""

_LEGACY_RHO_10 = np.arange(10, dtype=float) / 9.0
"""`Profile.run()`'s grid at `n_plasma_profile_elements = 10`.

The size PROCESS's own `tests/unit/models/physics/test_plasma_profiles.py` monkeypatches
in for `test_neprofile` / `test_teprofile`.
"""


def _profile(cls, **physics):
    """A real `Profile` subclass instance with a real `DataStructure` attached."""
    obj = cls()
    data = DataStructure()
    obj.data = data
    for name, value in physics.items():
        setattr(data.physics, name, value)
    return obj


# --------------------------------------------------------------------- grid


def _reference_profile_grid(n_plasma_profile_elements):
    """`Profile.run` + `normalise_profile_x` + `calculate_profile_dx`, in that order.

    `Profile.run` is called unbound: `NeProfile.run` would run the entire unit, and only
    the base class's three-line grid construction is under test here.
    """
    obj = _profile(NeProfile, n_plasma_profile_elements=n_plasma_profile_elements)
    Profile.run(obj)
    obj.normalise_profile_x()
    obj.calculate_profile_dx()
    return obj.profile_x, obj.profile_dx


class TestProfileGrid(Tier1Contract):
    """`Profile.run`/`normalise_profile_x`/`calculate_profile_dx` -> `calculate_profile_grid`.

    `n_plasma_profile_elements` is `static`: it is a shape, never an iteration variable,
    and there is nothing to differentiate. So this case checks values only -- which is the
    whole content of the claim, that the composed three steps are `linspace(0, 1, n)` and
    `1 / (n - 1)`.
    """

    audit_record = "models/physics/profiles.md"
    reference = _reference_profile_grid
    ported = calculate_profile_grid
    static_argnames = ("n_plasma_profile_elements",)

    samples = [
        legacy_sample("default-201", n_plasma_profile_elements=201),
        legacy_sample("harness-11", n_plasma_profile_elements=_N_POINTS),
        # The size PROCESS's own profile unit tests monkeypatch in.
        legacy_sample("baseline-2018-10", n_plasma_profile_elements=10),
    ]


# --------------------------------------------------------------------- integral


def _reference_integrate_profile_y(profile_y, profile_x):
    """`Profile.integrate_profile_y`, read back off `profile_integ`.

    `profile_dx` is set to the true spacing rather than left at its `__init__` value of 0:
    the source passes it to `sp.integrate.simpson` as `dx=`, and while `scipy` ignores
    `dx` whenever `x` is given, handing it a wrong value would make the test's agreement
    an accident of that behaviour rather than a check of it.
    """
    obj = _profile(NeProfile)
    obj.profile_x = np.asarray(profile_x, dtype=float)
    obj.profile_y = np.asarray(profile_y, dtype=float)
    obj.profile_dx = float(obj.profile_x[1] - obj.profile_x[0])
    obj.integrate_profile_y()
    return obj.profile_integ


class TestIntegrateProfileY(Tier1Contract):
    """`Profile.integrate_profile_y` -> `integrate_profile_y`.

    A direct check of `plasma_profiles._simpson` against `sp.integrate.simpson` at the
    call site that actually uses it, including `d/d(profile_x[i])` -- the derivative the
    uniform-grid shortcut gets wrong while agreeing in value. Unit #12 exercises the same
    helper only through `calculate_pedestal_profile_values`.
    """

    audit_record = "models/physics/profiles.md"
    reference = _reference_integrate_profile_y
    ported = integrate_profile_y

    samples = [
        legacy_sample(
            "density-like",
            profile_y=8.0e19 * (1.0 - 0.8 * _RHO**2),
            profile_x=_RHO,
        ),
        legacy_sample(
            # Non-uniform grid: the case where the general and the uniform rules
            # disagree in value as well as in derivative.
            "non-uniform-grid",
            profile_y=12.0 * (1.0 - 0.9 * _RHO**2) + 0.5,
            profile_x=np.sort(_RHO**1.5),
        ),
    ]


# --------------------------------------------------------------------- density profile


def _density_profile_reference(i_plasma_pedestal):
    """Build a `NeProfile.calculate_profile_y` adapter pinned to one switch value."""

    def reference(
        rho,
        radius_plasma_pedestal_density_norm,
        nd_plasma_electron_on_axis,
        nd_plasma_pedestal_electron,
        nd_plasma_separatrix_electron,
        alphan,
    ):
        obj = _profile(NeProfile, i_plasma_pedestal=i_plasma_pedestal)
        obj.profile_y = np.zeros(np.shape(rho))
        obj.calculate_profile_y(
            np.asarray(rho, dtype=float),
            radius_plasma_pedestal_density_norm,
            nd_plasma_electron_on_axis,
            nd_plasma_pedestal_electron,
            nd_plasma_separatrix_electron,
            alphan,
        )
        return obj.profile_y

    return reference


_DENSITY_PEDESTAL_SAMPLES = [
    legacy_sample(
        # `NeProfileParam(baseline_2018)` from PROCESS's own
        # `tests/unit/models/physics/test_plasma_profiles.py`, on that test's 10-point
        # grid. The central density is the value that test's run actually reaches.
        # (That test declares these parameters and then never applies them -- see the
        # record. They are still a real, PROCESS-authored operating point.)
        "baseline-2018-pedestal",
        rho=_LEGACY_RHO_10,
        radius_plasma_pedestal_density_norm=0.94,
        nd_plasma_electron_on_axis=1.125e20,
        nd_plasma_pedestal_electron=6.1916268627398164e19,
        nd_plasma_separatrix_electron=3.6421334486704804e19,
        alphan=1.0,
    ),
    legacy_sample(
        # Pedestal off the grid on purpose: at `rped` exactly on a grid point the
        # core/edge mask boundary sits on a sample, and PROCESS's own finite difference
        # steps that point across the branch.
        "off-grid-pedestal",
        rho=_RHO,
        radius_plasma_pedestal_density_norm=0.85,
        nd_plasma_electron_on_axis=1.0e20,
        nd_plasma_pedestal_electron=5.0e19,
        nd_plasma_separatrix_electron=2.0e19,
        alphan=1.3,
    ),
]


class TestDensityProfile(Tier1Contract):
    """`NeProfile.calculate_profile_y` -> `calculate_density_profile`, pedestal switch."""

    audit_record = "models/physics/profiles.md"
    reference = _density_profile_reference(1)
    ported = calculate_density_profile

    samples = _DENSITY_PEDESTAL_SAMPLES


class TestDensityProfileParabolicSwitch(Tier1Contract):
    """The same port, against a reference pinned to `i_plasma_pedestal == 0`.

    This is the unit's central finding made into a test. `NeProfile.calculate_profile_y`
    opens with a parabolic assignment and -- unlike its `TeProfile` twin -- has no
    `return`, so the pedestal assignment immediately overwrites every element. If that
    dead branch ever became live, these cases would fail while `TestDensityProfile`
    passed, which is precisely the signal wanted.

    Same samples as `TestDensityProfile`, deliberately: the claim is that the two switch
    values are indistinguishable at the *same* point, not merely that each is
    self-consistent.
    """

    audit_record = "models/physics/profiles.md"
    reference = _density_profile_reference(0)
    ported = calculate_density_profile

    samples = _DENSITY_PEDESTAL_SAMPLES


class TestDensityProfileLModeLimit(Tier1Contract):
    """The L-mode configuration: `rped = 1`, `nped = nsep = 0`.

    What `plasma_profiles.parabolic_parameterisation`'s input-validation reset produces,
    and the configuration in which the pedestal formula degenerates to
    `n0 * (1 - rho**2) ** alphan`.

    `rho` and `radius_plasma_pedestal_density_norm` are **static here and nowhere else**.
    At `rped == 1` the core/edge mask boundary coincides with the grid's last point, so
    PROCESS's finite difference in either of those two directions straddles a branch
    change *and* a division by `1 - rped`: it reports a large spurious derivative that
    halving the step does not settle. That is the reference being ill-posed at this point,
    not the port disagreeing. The remaining four arguments are still differentiated.
    """

    audit_record = "models/physics/profiles.md"
    reference = _density_profile_reference(0)
    ported = calculate_density_profile
    static_argnames = ("rho", "radius_plasma_pedestal_density_norm")

    samples = [
        legacy_sample(
            "l-mode-reset-values",
            rho=_RHO,
            radius_plasma_pedestal_density_norm=1.0,
            nd_plasma_electron_on_axis=1.6e20,
            nd_plasma_pedestal_electron=0.0,
            nd_plasma_separatrix_electron=0.0,
            alphan=1.0,
        ),
    ]


# --------------------------------------------------------------------- temperature profile


def _temperature_profile_reference(i_plasma_pedestal, pedestal_arguments):
    """Build a `TeProfile.calculate_profile_y` adapter pinned to one switch value.

    The two arms take different argument lists, which is why `i_plasma_pedestal` is a
    topology switch here rather than a static kwarg: `pedestal_arguments` says which of
    the two signatures the adapter presents.
    """
    if pedestal_arguments:

        def reference(
            rho,
            radius_plasma_pedestal_temp_norm,
            temp_plasma_electron_on_axis_kev,
            temp_plasma_pedestal_kev,
            temp_plasma_separatrix_kev,
            alphat,
            tbeta,
        ):
            obj = _profile(TeProfile, i_plasma_pedestal=i_plasma_pedestal)
            obj.profile_y = np.zeros(np.shape(rho))
            obj.calculate_profile_y(
                np.asarray(rho, dtype=float),
                radius_plasma_pedestal_temp_norm,
                temp_plasma_electron_on_axis_kev,
                temp_plasma_pedestal_kev,
                temp_plasma_separatrix_kev,
                alphat,
                tbeta,
            )
            return obj.profile_y

        return reference

    def reference(rho, temp_plasma_electron_on_axis_kev, alphat):
        obj = _profile(TeProfile, i_plasma_pedestal=i_plasma_pedestal)
        obj.profile_y = np.zeros(np.shape(rho))
        obj.calculate_profile_y(
            np.asarray(rho, dtype=float),
            # The parabolic arm reads none of these; PROCESS's L-mode reset values are
            # passed so the call is the one a parabolic run actually makes.
            1.0,
            temp_plasma_electron_on_axis_kev,
            0.0,
            0.0,
            alphat,
            2.0,
        )
        return obj.profile_y

    return reference


class TestParabolicTemperatureProfile(Tier1Contract):
    """`TeProfile.calculate_profile_y`, parabolic arm, on the real endpoint-inclusive grid.

    `rho` is static here: at `rho == 1` the base `1 - rho**2` is exactly 0, and PROCESS's
    finite difference steps to 1.001 where `(-0.002) ** alphat` is NaN in `numpy`. That is
    the reference leaving its own domain. `TestParabolicTemperatureProfileInterior` checks
    `d/d(rho[i])` on a grid where the question is well-posed; here `t0` and `alphat` are
    still differentiated, including at the floored endpoint where both derivatives must be
    exactly zero.
    """

    audit_record = "models/physics/profiles.md"
    reference = _temperature_profile_reference(0, pedestal_arguments=False)
    ported = calculate_parabolic_temperature_profile
    static_argnames = ("rho",)

    samples = [
        legacy_sample(
            "baseline-2018-parabolic",
            rho=_RHO,
            temp_plasma_electron_on_axis_kev=27.370104119511087,
            alphat=1.45,
        ),
        legacy_sample(
            "flat-profile",
            rho=_RHO,
            temp_plasma_electron_on_axis_kev=12.0,
            alphat=0.5,
        ),
    ]


class TestParabolicTemperatureProfileInterior(Tier1Contract):
    """The parabolic arm again, on an interior grid, with `rho` differentiated.

    See `_RHO_INTERIOR`. Splitting this out rather than dropping the endpoint case keeps
    both claims tested: the values PROCESS actually computes (endpoint included) and the
    radial derivative (endpoint excluded).
    """

    audit_record = "models/physics/profiles.md"
    reference = _temperature_profile_reference(0, pedestal_arguments=False)
    ported = calculate_parabolic_temperature_profile

    samples = [
        legacy_sample(
            "baseline-2018-parabolic-interior",
            rho=_RHO_INTERIOR,
            temp_plasma_electron_on_axis_kev=27.370104119511087,
            alphat=1.45,
        ),
    ]


class TestPedestalTemperatureProfile(Tier1Contract):
    """`TeProfile.calculate_profile_y`, pedestal arm -> `calculate_pedestal_temperature_profile`.

    No endpoint problem: at `rho == 1` the edge arm applies, and it is linear. Both
    pedestal positions are placed off the grid so the mask boundary is never sampled.

    `reference_domain_errors` covers the source's `ProcessValueError("Negative
    temperature in plasma profile")`; the port returns an all-NaN profile there instead.
    """

    audit_record = "models/physics/profiles.md"
    reference = _temperature_profile_reference(1, pedestal_arguments=True)
    ported = calculate_pedestal_temperature_profile
    reference_domain_errors = (ProcessValueError,)

    samples = [
        legacy_sample(
            # `TeProfileParam(baseline_2018)`, on that test's 10-point grid, with the
            # central temperature the corresponding `PlasmaProfilesParam` run reaches.
            "baseline-2018-pedestal",
            rho=_LEGACY_RHO_10,
            radius_plasma_pedestal_temp_norm=0.94,
            temp_plasma_electron_on_axis_kev=27.370104119511087,
            temp_plasma_pedestal_kev=5.5,
            temp_plasma_separatrix_kev=0.1,
            alphat=1.45,
            tbeta=2.0,
        ),
        legacy_sample(
            # `tbeta != 2` is the one thing the density profile has no analogue for --
            # `NeProfile` hard-codes the exponent 2.
            "tbeta-not-two",
            rho=_RHO,
            radius_plasma_pedestal_temp_norm=0.72,
            temp_plasma_electron_on_axis_kev=22.0,
            temp_plasma_pedestal_kev=4.0,
            temp_plasma_separatrix_kev=0.15,
            alphat=1.1,
            tbeta=1.5,
        ),
    ]


# --------------------------------------------------------------------- core values


def _reference_ncore(
    radius_plasma_pedestal_density_norm,
    nd_plasma_pedestal_electron,
    nd_plasma_separatrix_electron,
    nd_plasma_electrons_vol_avg,
    alphan,
):
    """`NeProfile.ncore`, renamed.

    The source's parameters are `nped`/`nsep`/`nav`; the port spells them with the
    `data.physics` field names they are read from, per `naming_convention.md`. This
    adapter is the rename and nothing else -- positional, so a reordering in either
    signature shows up as a value mismatch rather than passing silently.
    """
    return NeProfile.ncore(
        radius_plasma_pedestal_density_norm,
        nd_plasma_pedestal_electron,
        nd_plasma_separatrix_electron,
        nd_plasma_electrons_vol_avg,
        alphan,
    )


def _reference_tcore(
    radius_plasma_pedestal_temp_norm,
    temp_plasma_pedestal_kev,
    temp_plasma_separatrix_kev,
    temp_plasma_electron_vol_avg_kev,
    alphat,
    tbeta,
):
    """`TeProfile.tcore`, renamed. See `_reference_ncore`; `tav` is the one renamed here."""
    return TeProfile.tcore(
        radius_plasma_pedestal_temp_norm,
        temp_plasma_pedestal_kev,
        temp_plasma_separatrix_kev,
        temp_plasma_electron_vol_avg_kev,
        alphat,
        tbeta,
    )


class TestNcore(Tier1Contract):
    """`NeProfile.ncore` -> `ncore`.

    The second sample sits in the floored region, where the source substitutes `1e-6` and
    logs. That arm is the one a port could silently drop: it agrees in value only if the
    `jnp.where` is there, and its derivative must be identically zero.
    """

    audit_record = "models/physics/profiles.md"
    reference = _reference_ncore
    ported = ncore

    samples = [
        legacy_sample(
            # `tests/unit/models/physics/test_plasma_profiles.py::test_ncore`, which
            # asserts 9.7756974320342041e19.
            "baseline-2018-ncore",
            radius_plasma_pedestal_density_norm=0.94,
            nd_plasma_pedestal_electron=5.8300851381352219e19,
            nd_plasma_separatrix_electron=3.4294618459618943e19,
            nd_plasma_electrons_vol_avg=7.4321e19,
            alphan=1.0,
        ),
        legacy_sample(
            "floored-negative",
            radius_plasma_pedestal_density_norm=0.94,
            nd_plasma_pedestal_electron=2.0e20,
            nd_plasma_separatrix_electron=1.0e20,
            nd_plasma_electrons_vol_avg=1.0e19,
            alphan=1.0,
        ),
    ]

    fuzz_bounds = {
        "radius_plasma_pedestal_density_norm": (0.8, 0.99),
        "nd_plasma_pedestal_electron": (1.0e19, 4.0e19),
        "nd_plasma_separatrix_electron": (1.0e18, 3.0e19),
        "nd_plasma_electrons_vol_avg": (6.0e19, 2.0e20),
        "alphan": (0.1, 1.0),
    }
    """Bounds chosen to stay clear of the floor.

    `alphan`'s range is PROCESS's own (iteration variable 6). The three densities are given
    ranges in which the closed form stays positive for every `rped` in range -- straddling
    the floor would put fuzz points on a kink, where the reference's finite difference has
    no meaningful error bar.
    """


class TestTcore(Tier1Contract):
    """`TeProfile.tcore` -> `tcore`.

    Also the check on `plasma_profiles._beta` standing in for `sp.special.beta`, at a real
    argument pair rather than a synthetic one.
    """

    audit_record = "models/physics/profiles.md"
    reference = _reference_tcore
    ported = tcore

    samples = [
        legacy_sample(
            # `tests/unit/models/physics/test_plasma_profiles.py::test_tcore`, which
            # asserts 28.09093632260765.
            "baseline-2018-tcore",
            radius_plasma_pedestal_temp_norm=0.94,
            temp_plasma_pedestal_kev=3.7775374842470044,
            temp_plasma_separatrix_kev=0.1,
            temp_plasma_electron_vol_avg_kev=12.33,
            alphat=1.45,
            tbeta=2.0,
        ),
    ]

    fuzz_bounds = {
        "radius_plasma_pedestal_temp_norm": (0.8, 0.99),
        "temp_plasma_pedestal_kev": (1.0, 8.0),
        "temp_plasma_separatrix_kev": (0.05, 1.0),
        "temp_plasma_electron_vol_avg_kev": (5.0, 25.0),
        "alphat": (0.5, 2.5),
        "tbeta": (1.0, 3.0),
    }
    """`alphat`'s range is PROCESS's own (iteration variable 5); the rest are operating
    ranges. `tbeta` is kept away from 0, where `2 / tbeta` diverges."""


# --------------------------------------------------------------------- on-axis values


def _on_axis_densities_reference(i_plasma_pedestal, pedestal_arguments):
    """`NeProfile.set_physics_variables`, one arm, read back off `data`."""
    if pedestal_arguments:

        def reference(
            radius_plasma_pedestal_density_norm,
            nd_plasma_pedestal_electron,
            nd_plasma_separatrix_electron,
            nd_plasma_electrons_vol_avg,
            nd_plasma_ions_total_vol_avg,
            alphan,
        ):
            obj = _profile(
                NeProfile,
                i_plasma_pedestal=i_plasma_pedestal,
                radius_plasma_pedestal_density_norm=(
                    radius_plasma_pedestal_density_norm
                ),
                nd_plasma_pedestal_electron=nd_plasma_pedestal_electron,
                nd_plasma_separatrix_electron=nd_plasma_separatrix_electron,
                nd_plasma_electrons_vol_avg=nd_plasma_electrons_vol_avg,
                nd_plasma_ions_total_vol_avg=nd_plasma_ions_total_vol_avg,
                alphan=alphan,
            )
            obj.set_physics_variables()
            return (
                obj.data.physics.nd_plasma_electron_on_axis,
                obj.data.physics.nd_plasma_ions_on_axis,
            )

        return reference

    def reference(nd_plasma_electrons_vol_avg, nd_plasma_ions_total_vol_avg, alphan):
        obj = _profile(
            NeProfile,
            i_plasma_pedestal=i_plasma_pedestal,
            nd_plasma_electrons_vol_avg=nd_plasma_electrons_vol_avg,
            nd_plasma_ions_total_vol_avg=nd_plasma_ions_total_vol_avg,
            alphan=alphan,
        )
        obj.set_physics_variables()
        return (
            obj.data.physics.nd_plasma_electron_on_axis,
            obj.data.physics.nd_plasma_ions_on_axis,
        )

    return reference


class TestParabolicOnAxisDensities(Tier1Contract):
    """`NeProfile.set_physics_variables`, parabolic arm.

    These two fields are the ones `plasma_profiles.parabolic_parameterisation` rewrites
    forty lines later; unit #12's port drops that rewrite in favour of this producer.
    """

    audit_record = "models/physics/profiles.md"
    reference = _on_axis_densities_reference(0, pedestal_arguments=False)
    ported = calculate_parabolic_on_axis_densities

    samples = [
        legacy_sample(
            "baseline-2018",
            nd_plasma_electrons_vol_avg=7.983e19,
            nd_plasma_ions_total_vol_avg=6.9461125748017857e19,
            alphan=1.0,
        ),
    ]

    fuzz_bounds = {
        "nd_plasma_electrons_vol_avg": (1.0e19, 2.0e20),
        "nd_plasma_ions_total_vol_avg": (1.0e19, 2.0e20),
        "alphan": (0.1, 1.0),
    }


class TestPedestalOnAxisDensities(Tier1Contract):
    """`NeProfile.set_physics_variables`, pedestal arm (via `ncore`).

    The legacy point is the one `PlasmaProfilesParam(baseline_2018)` asserts:
    `ne0 = 1.0585658890823703e20`, `ni0 = 9.210720071916929e19`.
    """

    audit_record = "models/physics/profiles.md"
    reference = _on_axis_densities_reference(1, pedestal_arguments=True)
    ported = calculate_pedestal_on_axis_densities

    samples = [
        legacy_sample(
            "baseline-2018",
            radius_plasma_pedestal_density_norm=0.94,
            nd_plasma_pedestal_electron=6.1916268627398164e19,
            nd_plasma_separatrix_electron=3.6421334486704804e19,
            nd_plasma_electrons_vol_avg=7.983e19,
            nd_plasma_ions_total_vol_avg=6.9461125748017857e19,
            alphan=1.0,
        ),
    ]

    fuzz_bounds = {
        "radius_plasma_pedestal_density_norm": (0.8, 0.99),
        "nd_plasma_pedestal_electron": (1.0e19, 4.0e19),
        "nd_plasma_separatrix_electron": (1.0e18, 3.0e19),
        "nd_plasma_electrons_vol_avg": (6.0e19, 2.0e20),
        "nd_plasma_ions_total_vol_avg": (1.0e19, 2.0e20),
        "alphan": (0.1, 1.0),
    }
    """Same clear-of-the-floor reasoning as `TestNcore`."""


def _on_axis_temperatures_reference(i_plasma_pedestal, pedestal_arguments):
    """`TeProfile.set_physics_variables`, one arm, read back off `data`."""
    if pedestal_arguments:

        def reference(
            radius_plasma_pedestal_temp_norm,
            temp_plasma_pedestal_kev,
            temp_plasma_separatrix_kev,
            temp_plasma_electron_vol_avg_kev,
            temp_plasma_ion_vol_avg_kev,
            alphat,
            tbeta,
        ):
            obj = _profile(
                TeProfile,
                i_plasma_pedestal=i_plasma_pedestal,
                radius_plasma_pedestal_temp_norm=radius_plasma_pedestal_temp_norm,
                temp_plasma_pedestal_kev=temp_plasma_pedestal_kev,
                temp_plasma_separatrix_kev=temp_plasma_separatrix_kev,
                temp_plasma_electron_vol_avg_kev=temp_plasma_electron_vol_avg_kev,
                temp_plasma_ion_vol_avg_kev=temp_plasma_ion_vol_avg_kev,
                alphat=alphat,
                tbeta=tbeta,
            )
            obj.set_physics_variables()
            return (
                obj.data.physics.temp_plasma_electron_on_axis_kev,
                obj.data.physics.temp_plasma_ion_on_axis_kev,
            )

        return reference

    def reference(temp_plasma_electron_vol_avg_kev, temp_plasma_ion_vol_avg_kev, alphat):
        obj = _profile(
            TeProfile,
            i_plasma_pedestal=i_plasma_pedestal,
            temp_plasma_electron_vol_avg_kev=temp_plasma_electron_vol_avg_kev,
            temp_plasma_ion_vol_avg_kev=temp_plasma_ion_vol_avg_kev,
            alphat=alphat,
        )
        obj.set_physics_variables()
        return (
            obj.data.physics.temp_plasma_electron_on_axis_kev,
            obj.data.physics.temp_plasma_ion_on_axis_kev,
        )

    return reference


class TestParabolicOnAxisTemperatures(Tier1Contract):
    """`TeProfile.set_physics_variables`, parabolic arm."""

    audit_record = "models/physics/profiles.md"
    reference = _on_axis_temperatures_reference(0, pedestal_arguments=False)
    ported = calculate_parabolic_on_axis_temperatures

    samples = [
        legacy_sample(
            "baseline-2018",
            temp_plasma_electron_vol_avg_kev=13.07,
            temp_plasma_ion_vol_avg_kev=13.07,
            alphat=1.45,
        ),
    ]

    fuzz_bounds = {
        "temp_plasma_electron_vol_avg_kev": (1.0, 40.0),
        "temp_plasma_ion_vol_avg_kev": (1.0, 40.0),
        "alphat": (0.5, 2.5),
    }


class TestPedestalOnAxisTemperatures(Tier1Contract):
    """`TeProfile.set_physics_variables`, pedestal arm (via `tcore`).

    The legacy point is `PlasmaProfilesParam(baseline_2018)`'s
    `te0 = ti0 = 27.370104119511087`.
    """

    audit_record = "models/physics/profiles.md"
    reference = _on_axis_temperatures_reference(1, pedestal_arguments=True)
    ported = calculate_pedestal_on_axis_temperatures

    samples = [
        legacy_sample(
            "baseline-2018",
            radius_plasma_pedestal_temp_norm=0.94,
            temp_plasma_pedestal_kev=5.5,
            temp_plasma_separatrix_kev=0.1,
            temp_plasma_electron_vol_avg_kev=13.07,
            temp_plasma_ion_vol_avg_kev=13.07,
            alphat=1.45,
            tbeta=2.0,
        ),
    ]

    fuzz_bounds = {
        "radius_plasma_pedestal_temp_norm": (0.8, 0.99),
        "temp_plasma_pedestal_kev": (1.0, 8.0),
        "temp_plasma_separatrix_kev": (0.05, 1.0),
        "temp_plasma_electron_vol_avg_kev": (5.0, 25.0),
        "temp_plasma_ion_vol_avg_kev": (5.0, 25.0),
        "alphat": (0.5, 2.5),
        "tbeta": (1.0, 3.0),
    }


# --------------------------------------------------------- pedestal/separatrix densities


def _reference_greenwald_density_fractions(
    nd_plasma_pedestal_electron,
    nd_plasma_separatrix_electron,
    plasma_current,
    rminor,
):
    """`NeProfile.set_pedestal_and_separatrix_values`, `USER_INPUT` arm.

    Goes through the real method, so the call into
    `PlasmaDensityLimit.calculate_greenwald_density_limit` is the real one -- which is
    what makes this case the check on `profiles._greenwald_limit` being a faithful
    inline of it.
    """
    obj = _profile(
        NeProfile,
        i_nd_plasma_pedestal_separatrix=0,
        nd_plasma_pedestal_electron=nd_plasma_pedestal_electron,
        nd_plasma_separatrix_electron=nd_plasma_separatrix_electron,
        plasma_current=plasma_current,
        rminor=rminor,
    )
    obj.set_pedestal_and_separatrix_values()
    return (
        obj.data.physics.f_nd_plasma_pedestal_greenwald,
        obj.data.physics.f_nd_plasma_separatrix_greenwald,
    )


def _reference_pedestal_separatrix_densities(
    f_nd_plasma_pedestal_greenwald,
    f_nd_plasma_separatrix_greenwald,
    plasma_current,
    rminor,
):
    """`NeProfile.set_pedestal_and_separatrix_values`, `GREENWALD_FRACTION` arm."""
    obj = _profile(
        NeProfile,
        i_nd_plasma_pedestal_separatrix=1,
        f_nd_plasma_pedestal_greenwald=f_nd_plasma_pedestal_greenwald,
        f_nd_plasma_separatrix_greenwald=f_nd_plasma_separatrix_greenwald,
        plasma_current=plasma_current,
        rminor=rminor,
    )
    obj.set_pedestal_and_separatrix_values()
    return (
        obj.data.physics.nd_plasma_pedestal_electron,
        obj.data.physics.nd_plasma_separatrix_electron,
    )


class TestGreenwaldDensityFractions(Tier1Contract):
    """`set_pedestal_and_separatrix_values`, `i_nd_plasma_pedestal_separatrix == 0`."""

    audit_record = "models/physics/profiles.md"
    reference = _reference_greenwald_density_fractions
    ported = calculate_greenwald_density_fractions

    samples = [
        legacy_sample(
            "baseline-2018",
            nd_plasma_pedestal_electron=6.1916268627398164e19,
            nd_plasma_separatrix_electron=3.6421334486704804e19,
            plasma_current=1.5e7,
            rminor=2.9264516129032256,
        ),
    ]

    fuzz_bounds = {
        "nd_plasma_pedestal_electron": (1.0e19, 1.0e20),
        "nd_plasma_separatrix_electron": (1.0e18, 5.0e19),
        # PROCESS's own bounds for iteration variable 2 (`plasma_current`) and 3
        # (`rminor`) are far wider than any real device; these are operating ranges.
        "plasma_current": (5.0e6, 3.0e7),
        "rminor": (0.5, 4.0),
    }


class TestPedestalSeparatrixDensities(Tier1Contract):
    """`set_pedestal_and_separatrix_values`, `i_nd_plasma_pedestal_separatrix == 1`.

    PROCESS's default arm, and the exact inverse of `TestGreenwaldDensityFractions`.
    """

    audit_record = "models/physics/profiles.md"
    reference = _reference_pedestal_separatrix_densities
    ported = calculate_pedestal_separatrix_densities

    samples = [
        legacy_sample(
            "defaults",
            f_nd_plasma_pedestal_greenwald=0.85,
            f_nd_plasma_separatrix_greenwald=0.5,
            plasma_current=1.5e7,
            rminor=2.9264516129032256,
        ),
    ]

    fuzz_bounds = {
        "f_nd_plasma_pedestal_greenwald": (0.1, 1.2),
        "f_nd_plasma_separatrix_greenwald": (0.05, 0.9),
        "plasma_current": (5.0e6, 3.0e7),
        "rminor": (0.5, 4.0),
    }
