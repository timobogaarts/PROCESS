"""Pure-functional port of `process/models/physics/profiles.py`.

Registry unit #21. Audit record:
`functional_process/_audit/units/models/physics/profiles.md`.

This is the unit `plasma_profiles.py` (unit #12) is blocked on. `PlasmaProfile` holds two
injected `Profile` sub-models -- `NeProfile()` / `TeProfile()`, built in
`process/main.py:674-676` -- and both of its branches call `.run()` on them **for
effect**, then read four `.physics` on-axis fields and six object attributes those calls
wrote. Everything in that back-door is ported here as explicit arguments and returns.

Thirteen tier-1 functions, covering every arithmetic path in the source file. Nothing
here iterates, and nothing calls another `Model`.

Three things about the source are worth reading before the code:

- **`NeProfile.calculate_profile_y`'s parabolic branch is dead code.** Unlike
  `TeProfile.calculate_profile_y`, it has no `return` after the parabolic assignment, so
  the pedestal formula immediately overwrites it. Verified: both `i_plasma_pedestal`
  values produce bit-identical arrays for identical inputs. So the density profile needs
  *one* function, not two arms -- see `calculate_density_profile`.
- **`calculate_profile_y` returns `None` on both classes**, and `current_drive.py` uses
  its return value at six call sites. Those paths raise `TypeError`. See the record's
  § open questions; the port returns the profile, which is what those callers want.
- **`profile_x` has no differentiable producer.** It is `linspace(0, 1, n)` for a static
  `n_plasma_profile_elements`, so the grid is graph-assembly-time data, not a flowing
  value. It is still passed explicitly (never read off an object) and is still
  differentiated by the harness wherever that is well-posed.

`_simpson` and `_beta` are imported from `plasma_profiles.py` rather than reimplemented:
`_simpson` in particular carries a correctness argument (`scipy` uses the general
non-uniform rule whenever `x` is passed, and the uniform shortcut agrees in value while
being wrong in `d/dx[i]`) that must not be re-derived per unit.

**Minted `VarPath`s.** Five values here have no PROCESS storage location -- they live only
as attributes of the two injected `Profile` objects. Each is a real graph edge, so each is
minted into `.physics`, following the precedent set by `coils/calculate.py`'s
`.stellarator.coilcurrent` and by `plasma_profiles.py`'s `ProfileFactors`:

    .physics.radius_plasma_profile_norm              <- Profile.profile_x
    .physics.dradius_plasma_profile_norm             <- Profile.profile_dx
    .physics.nd_plasma_electron_profile              <- neprofile.profile_y   [1]
    .physics.temp_plasma_electron_profile_kev        <- teprofile.profile_y   [1]
    .physics.nd_plasma_electron_profile_integral     <- neprofile.profile_integ
    .physics.temp_plasma_electron_profile_integral_kev <- teprofile.profile_integ

[1] These two were already minted, with the same spelling, by `ProfileFactors` in
`plasma_profiles.py` -- which *reads* them and had no producer. The nodes here are that
producer, so unit #12's largest dangling edge closes on this unit landing.
"""

import equinox as eqx
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    OutputInto,
)

from functional_process.paths import physics
from functional_process.physics.profiles import (
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

__all__ = [
    "ncore",
    "tcore",
]


class ProfileGrid(ExplicitFunction):
    """cottax node: `calculate_profile_grid`, unchanged, ports declared.

    A **source node**: it has no reads at all, because its only argument is a static
    shape. That is the honest shape of it -- the radius grid is decided when the graph is
    assembled and never flows.

    Both outputs are minted (see module docstring). `dradius_plasma_profile_norm` is
    declared even though no consumer reads it, so that the graph shows what the source
    computes; `Graph.prune` will drop it.

    Not blocked: `Profile.run` runs identically in both `i_plasma_pedestal` branches.
    """

    n_plasma_profile_elements: int = eqx.field(static=True)

    radius_plasma_profile_norm = OutputInto(physics)
    dradius_plasma_profile_norm = OutputInto(physics)

    def __call__(self):
        return calculate_profile_grid(self.n_plasma_profile_elements)


class NeProfileIntegral(ExplicitFunction):
    """cottax node: `integrate_profile_y` on the **density** profile.

    Two node classes for one function because PROCESS runs it on two objects with two
    separate stores. Both outputs and both array inputs are minted (module docstring).

    Not blocked: `integrate_profile_y` runs in both `i_plasma_pedestal` branches. Its
    only *consumer*, `plasma_profiles.pedestal_parameterisation`, is pedestal-only -- but
    that is the consumer's gating, not this node's, and an unread output is pruned rather
    than wrong.
    """

    nd_plasma_electron_profile_integral = OutputInto(physics)

    def __call__(
        self,
        nd_plasma_electron_profile=From(physics),
        radius_plasma_profile_norm=From(physics),
    ):
        return integrate_profile_y(
            nd_plasma_electron_profile, radius_plasma_profile_norm
        )


class TeProfileIntegral(ExplicitFunction):
    """cottax node: `integrate_profile_y` on the **temperature** profile.

    See `NeProfileIntegral`. Its `temp_plasma_electron_profile_kev` input is owned by
    whichever arm of the `i_plasma_pedestal` switch is selected, which is the arm's
    problem, not this node's.
    """

    temp_plasma_electron_profile_integral_kev = OutputInto(physics)

    def __call__(
        self,
        temp_plasma_electron_profile_kev=From(physics),
        radius_plasma_profile_norm=From(physics),
    ):
        return integrate_profile_y(
            temp_plasma_electron_profile_kev, radius_plasma_profile_norm
        )


class DensityProfile(ExplicitFunction):
    """cottax node: `calculate_density_profile`, unchanged, ports declared.

    **Not** an arm of the `i_plasma_pedestal` switch, and that is the point: the source's
    parabolic branch is dead, so one node covers both configurations. In the parabolic
    configuration its three pedestal inputs carry the L-mode values that
    `plasma_profiles.parabolic_parameterisation`'s input-validation reset supplies -- a
    graph-assembly-time coercion nothing currently performs (unit #12's open question 2).

    Owns `.physics.nd_plasma_electron_profile`, the array `ProfileFactors` in
    `plasma_profiles.py` already reads and had no producer for.

    Ready to register in `total_process.COMMON` -- with the caveat that until the
    L-mode-reset question is settled, a parabolic configuration relies on the input file
    supplying `rped = 1`, `nped = nsep = 0` rather than on any node enforcing it.
    """

    nd_plasma_electron_profile = OutputInto(physics)

    def __call__(
        self,
        radius_plasma_profile_norm=From(physics),
        radius_plasma_pedestal_density_norm=From(physics),
        nd_plasma_electron_on_axis=From(physics),
        nd_plasma_pedestal_electron=From(physics),
        nd_plasma_separatrix_electron=From(physics),
        alphan=From(physics),
    ):
        return calculate_density_profile(
            radius_plasma_profile_norm,
            radius_plasma_pedestal_density_norm,
            nd_plasma_electron_on_axis,
            nd_plasma_pedestal_electron,
            nd_plasma_separatrix_electron,
            alphan,
        )


class ParabolicTemperatureProfile(ExplicitFunction):
    """cottax node: `calculate_parabolic_temperature_profile`.

    **Blocked, do not register yet.** One of two `Alternative`s under
    `.physics.i_plasma_pedestal` -- it and `PedestalTemperatureProfile` both own
    `.physics.temp_plasma_electron_profile_kev`, so they cannot both be in one graph.
    The switch is the same one `density_limits.EcrhDensityLimit` consumes as a *static
    kwarg*, and nothing reconciles the two roles (`plasma_profiles.md` § open questions 1).
    Registering either arm before that decision bakes in a wiring it may change.
    """

    temp_plasma_electron_profile_kev = OutputInto(physics)

    def __call__(
        self,
        radius_plasma_profile_norm=From(physics),
        temp_plasma_electron_on_axis_kev=From(physics),
        alphat=From(physics),
    ):
        return calculate_parabolic_temperature_profile(
            radius_plasma_profile_norm, temp_plasma_electron_on_axis_kev, alphat
        )


class PedestalTemperatureProfile(ExplicitFunction):
    """cottax node: `calculate_pedestal_temperature_profile`.

    **Blocked, do not register yet** -- the other `Alternative` to
    `ParabolicTemperatureProfile`; see that class's docstring for the reason.
    """

    temp_plasma_electron_profile_kev = OutputInto(physics)

    def __call__(
        self,
        radius_plasma_profile_norm=From(physics),
        radius_plasma_pedestal_temp_norm=From(physics),
        temp_plasma_electron_on_axis_kev=From(physics),
        temp_plasma_pedestal_kev=From(physics),
        temp_plasma_separatrix_kev=From(physics),
        alphat=From(physics),
        tbeta=From(physics),
    ):
        return calculate_pedestal_temperature_profile(
            radius_plasma_profile_norm,
            radius_plasma_pedestal_temp_norm,
            temp_plasma_electron_on_axis_kev,
            temp_plasma_pedestal_kev,
            temp_plasma_separatrix_kev,
            alphat,
            tbeta,
        )


class ParabolicOnAxisDensities(ExplicitFunction):
    """cottax node: `calculate_parabolic_on_axis_densities`.

    **Blocked, do not register yet.** `Alternative` to `PedestalOnAxisDensities` under
    `.physics.i_plasma_pedestal`; same unresolved two-roles question as
    `ParabolicTemperatureProfile`.
    """

    nd_plasma_electron_on_axis = OutputInto(physics)
    nd_plasma_ions_on_axis = OutputInto(physics)

    def __call__(
        self,
        nd_plasma_electrons_vol_avg=From(physics),
        nd_plasma_ions_total_vol_avg=From(physics),
        alphan=From(physics),
    ):
        return calculate_parabolic_on_axis_densities(
            nd_plasma_electrons_vol_avg, nd_plasma_ions_total_vol_avg, alphan
        )


class PedestalOnAxisDensities(ExplicitFunction):
    """cottax node: `calculate_pedestal_on_axis_densities`.

    **Blocked, do not register yet** -- `Alternative` to `ParabolicOnAxisDensities`.
    """

    nd_plasma_electron_on_axis = OutputInto(physics)
    nd_plasma_ions_on_axis = OutputInto(physics)

    def __call__(
        self,
        radius_plasma_pedestal_density_norm=From(physics),
        nd_plasma_pedestal_electron=From(physics),
        nd_plasma_separatrix_electron=From(physics),
        nd_plasma_electrons_vol_avg=From(physics),
        nd_plasma_ions_total_vol_avg=From(physics),
        alphan=From(physics),
    ):
        return calculate_pedestal_on_axis_densities(
            radius_plasma_pedestal_density_norm,
            nd_plasma_pedestal_electron,
            nd_plasma_separatrix_electron,
            nd_plasma_electrons_vol_avg,
            nd_plasma_ions_total_vol_avg,
            alphan,
        )


class ParabolicOnAxisTemperatures(ExplicitFunction):
    """cottax node: `calculate_parabolic_on_axis_temperatures`.

    **Blocked, do not register yet** -- `Alternative` to `PedestalOnAxisTemperatures`.
    """

    temp_plasma_electron_on_axis_kev = OutputInto(physics)
    temp_plasma_ion_on_axis_kev = OutputInto(physics)

    def __call__(
        self,
        temp_plasma_electron_vol_avg_kev=From(physics),
        temp_plasma_ion_vol_avg_kev=From(physics),
        alphat=From(physics),
    ):
        return calculate_parabolic_on_axis_temperatures(
            temp_plasma_electron_vol_avg_kev, temp_plasma_ion_vol_avg_kev, alphat
        )


class PedestalOnAxisTemperatures(ExplicitFunction):
    """cottax node: `calculate_pedestal_on_axis_temperatures`.

    **Blocked, do not register yet** -- `Alternative` to `ParabolicOnAxisTemperatures`.
    """

    temp_plasma_electron_on_axis_kev = OutputInto(physics)
    temp_plasma_ion_on_axis_kev = OutputInto(physics)

    def __call__(
        self,
        radius_plasma_pedestal_temp_norm=From(physics),
        temp_plasma_pedestal_kev=From(physics),
        temp_plasma_separatrix_kev=From(physics),
        temp_plasma_electron_vol_avg_kev=From(physics),
        temp_plasma_ion_vol_avg_kev=From(physics),
        alphat=From(physics),
        tbeta=From(physics),
    ):
        return calculate_pedestal_on_axis_temperatures(
            radius_plasma_pedestal_temp_norm,
            temp_plasma_pedestal_kev,
            temp_plasma_separatrix_kev,
            temp_plasma_electron_vol_avg_kev,
            temp_plasma_ion_vol_avg_kev,
            alphat,
            tbeta,
        )


class GreenwaldDensityFractions(ExplicitFunction):
    """cottax node: `calculate_greenwald_density_fractions`.

    The `.physics.i_nd_plasma_pedestal_separatrix == USER_INPUT` (`0`) occupant of
    `ProfileParameterisationPedestal.pedestal_separatrix`. **Written, registered in the
    registry, and not reached by either reference machine** -- both select the default
    `GREENWALD_FRACTION` arm below.

    Two things about this node were recorded as blockers until 2026-08-27 and both are
    resolved rather than worked around:

    1. Its sibling `PedestalSeparatrixDensities` owns the *inputs* this one reads and
       reads the *outputs* this one owns. That is not a conflict: the two arms are
       inverses of one another, and only one is ever in a graph. The slot's docstring
       in `models/physics/namespace.py` records why that makes a default *wrong*
       rather than merely unnecessary.
    2. `set_pedestal_and_separatrix_values` is called only from `physics.py:363-368`,
       inside `if i_plasma_pedestal == PEDESTAL_PROFILE` -- a switch nested under
       another switch, which `configuration.TOPOLOGY_SWITCHES`' flat list cannot
       express. The slot mechanism does not need it to: the nested switch is a slot of
       the outer arm's occupant, so the parabolic occupant simply has no such slot.
    """

    f_nd_plasma_pedestal_greenwald = OutputInto(physics)
    f_nd_plasma_separatrix_greenwald = OutputInto(physics)

    def __call__(
        self,
        nd_plasma_pedestal_electron=From(physics),
        nd_plasma_separatrix_electron=From(physics),
        plasma_current=From(physics),
        rminor=From(physics),
    ):
        return calculate_greenwald_density_fractions(
            nd_plasma_pedestal_electron,
            nd_plasma_separatrix_electron,
            plasma_current,
            rminor,
        )


class PedestalSeparatrixDensities(ExplicitFunction):
    """cottax node: `calculate_pedestal_separatrix_densities`.

    The `.physics.i_nd_plasma_pedestal_separatrix == GREENWALD_FRACTION` (`1`, PROCESS's
    default) occupant of `ProfileParameterisationPedestal.pedestal_separatrix`, and the
    live arm on `large_tokamak_eval.IN.DAT`. See `GreenwaldDensityFractions` above for
    the two blockers this pair carried until 2026-08-27 and how the slot resolves both.

    **A genuine upstream producer, not a leaf.** It owns
    `.physics.nd_plasma_pedestal_electron` and `.physics.nd_plasma_separatrix_electron`,
    which `DensityProfile`, `PedestalOnAxisDensities` and unit #12's pedestal arm all
    read -- and which `optimise_design.md` §11.5 found frozen at their input values,
    `0.5e20`/`0.2e20` against PROCESS's converged `6.12e19`/`3.60e19`.
    """

    nd_plasma_pedestal_electron = OutputInto(physics)
    nd_plasma_separatrix_electron = OutputInto(physics)

    def __call__(
        self,
        f_nd_plasma_pedestal_greenwald=From(physics),
        f_nd_plasma_separatrix_greenwald=From(physics),
        plasma_current=From(physics),
        rminor=From(physics),
    ):
        return calculate_pedestal_separatrix_densities(
            f_nd_plasma_pedestal_greenwald,
            f_nd_plasma_separatrix_greenwald,
            plasma_current,
            rminor,
        )
