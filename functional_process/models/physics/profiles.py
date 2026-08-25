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
import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    OutputInto,
)

from functional_process.models.physics.plasma_profiles import _beta, _simpson
from functional_process.paths import physics

GREENWALD_COEFFICIENT = 1.0e14
"""`PlasmaDensityLimit.calculate_greenwald_density_limit`'s coefficient.

`n_GW = 1e14 * I_p / (pi * a^2)` with `I_p` in A and `a` in m. See `_greenwald_limit`.
"""

NCORE_FLOOR = 1.0e-6
"""What `NeProfile.ncore` substitutes when its closed form goes negative (source L283)."""

TEMPERATURE_FLOOR_KEV = 1.0e-8
"""`TeProfile.calculate_profile_y`'s parabolic-branch floor (source L432).

The source's comment: profile values of 0 cause divide-by-zero errors downstream, and
1e-8 keV is small enough not to change any calculation.
"""


def _greenwald_limit(plasma_current, rminor):
    """`PlasmaDensityLimit.calculate_greenwald_density_limit`, inlined.

    `set_pedestal_and_separatrix_values` is the one place in this file that calls out to
    another model -- `process/models/physics/density_limit.py`, which is **not a
    registered unit** (only `models/stellarator/density_limits.py` is). The callee is a
    pure `@staticmethod` one-liner with no `self`/`data` access, so inlining it keeps the
    two ported functions tier 1 and keeps their signatures equal to the source's actual
    reads-set (`plasma_current`, `rminor`).

    This is duplicated arithmetic and is flagged as such in the record: when
    `physics/density_limit.py` is audited, this should become a graph edge from that
    unit's Greenwald node rather than a second copy of the formula.
    """
    return GREENWALD_COEFFICIENT * plasma_current / (jnp.pi * rminor**2)


def calculate_profile_grid(n_plasma_profile_elements):
    """The normalised radius grid and its spacing.

    Ports `Profile.run` + `Profile.normalise_profile_x` + `Profile.calculate_profile_dx`
    (source L59-95), which PROCESS always runs as that exact three-call sequence at the
    top of `NeProfile.run`/`TeProfile.run`. The composition is `arange(n) / max(arange(n))`
    followed by `max(x) / (n - 1)`, i.e. `linspace(0, 1, n)` and `1 / (n - 1)`; it is
    written out below in the source's own two steps rather than collapsed, so the port
    reads against the source line by line.

    `n_plasma_profile_elements` is a **shape, not a value** (default 201). It is a static
    argument: it sets the length of every profile array, it is never an iteration
    variable, and differentiating with respect to it is meaningless.

    `profile_dx` is returned because the source computes and stores it, **not** because
    anything needs it. Every consumer (`plasma_profiles.pedestal_parameterisation`,
    `fusion_reactions.py`, `impurity_radiation.py`) passes it to `sp.integrate.simpson`
    as `dx=` alongside `x=`, and `scipy` ignores `dx` entirely whenever `x` is given
    (verified). It is a dead output; see the record.

    Parameters
    ----------
    n_plasma_profile_elements :
        Number of points in every profile array. Static.

    Returns
    -------
    tuple
        `(profile_x, profile_dx)`.
    """
    profile_x = jnp.arange(n_plasma_profile_elements, dtype=float)
    profile_x = profile_x / jnp.max(profile_x)
    profile_dx = jnp.max(profile_x) / (n_plasma_profile_elements - 1)
    return profile_x, profile_dx


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


def integrate_profile_y(profile_y, profile_x):
    """Simpson integral of a profile over the normalised radius.

    Ports `Profile.integrate_profile_y` (source L103-112), which is
    `sp.integrate.simpson(profile_y, x=profile_x, dx=profile_dx)`. `profile_dx` is not an
    argument: `scipy` ignores `dx` whenever `x` is given, so taking it would declare a
    dependence the computation does not have.

    Delegates to `plasma_profiles._simpson` rather than reimplementing the rule -- read
    that function's docstring, it is where the non-uniform-vs-uniform trap (identical
    values, wrong `d/d(profile_x[i])`) is argued out.

    Parameters
    ----------
    profile_y :
        Profile values on `profile_x`.
    profile_x :
        Normalised radius grid.

    Returns
    -------
    :
        `profile_integ` -- the line-averaged value over rho in [0, 1].
    """
    return _simpson(profile_y, profile_x)


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


def calculate_density_profile(
    rho,
    radius_plasma_pedestal_density_norm,
    nd_plasma_electron_on_axis,
    nd_plasma_pedestal_electron,
    nd_plasma_separatrix_electron,
    alphan,
):
    """Electron density at each normalised minor radius; HELIOS pedestal profile.

    Ports `NeProfile.calculate_profile_y` (source L161-211) -- **all of it, in one
    function with no switch arm**, which is the finding this port rests on.

    The source opens with `if i_plasma_pedestal == PARABOLIC_PROFILE: self.profile_y =
    n0 * (1 - rho**2) ** alphan` and then, unlike its `TeProfile` twin forty lines later,
    **does not return**. Execution falls straight into the pedestal assignment, which
    overwrites every element. Verified empirically: for the same arguments the two switch
    values produce bit-identical arrays. The parabolic branch is dead code.

    It is dead *benignly* only because `plasma_profiles.parabolic_parameterisation`
    resets `radius_plasma_pedestal_density_norm = 1`, `nd_plasma_pedestal_electron = 0`
    and `nd_plasma_separatrix_electron = 0` before calling `neprofile.run()`, at which
    point this formula degenerates to exactly `n0 * (1 - rho**2) ** alphan`. That reset is
    input coercion PROCESS performs elsewhere, not physics this function does; if it were
    ever dropped, a parabolic run would silently get a pedestal profile. Flagged in the
    record, not fixed here.

    Two guards the source does not need and a traced port does:

    - `(1 - (rho / rped) ** 2) ** alphan` has a negative base wherever `rho > rped`.
      PROCESS never evaluates it there (it indexes with a boolean mask); `jnp.where`
      evaluates both arms, so the base is clamped and the result substituted. At
      `rho == rped` the base is exactly 0 and `0 ** alphan == 0`, giving `nped` -- which
      is also what the edge arm gives there, so the profile is continuous.
    - `(1 - rho) / (1 - rped)` divides by zero at `rped == 1`, the parabolic
      configuration. PROCESS's mask is empty there so the expression never runs; here the
      denominator is substituted when it vanishes. The arm is unselected at `rped == 1`,
      so the substitution changes no value.

    The source's `logger.info` when `n0 < nped` is not ported: it is a diagnostic, it is
    data-dependent, and it changes nothing.

    Parameters
    ----------
    rho :
        Normalised minor radius grid.
    radius_plasma_pedestal_density_norm :
        Normalised minor radius of the pedestal.
    nd_plasma_electron_on_axis :
        Central electron density (m^-3); `n0` in the source.
    nd_plasma_pedestal_electron :
        Pedestal electron density (m^-3).
    nd_plasma_separatrix_electron :
        Separatrix electron density (m^-3).
    alphan :
        Density peaking parameter.

    Returns
    -------
    :
        `profile_y` -- electron density (m^-3) at each point of `rho`.
    """
    inside = rho <= radius_plasma_pedestal_density_norm

    base = 1.0 - (rho / radius_plasma_pedestal_density_norm) ** 2
    safe_base = jnp.where(base > 0.0, base, 1.0)
    shape = jnp.where(base > 0.0, safe_base**alphan, 0.0)
    core = (
        nd_plasma_pedestal_electron
        + (nd_plasma_electron_on_axis - nd_plasma_pedestal_electron) * shape
    )

    span = 1.0 - radius_plasma_pedestal_density_norm
    safe_span = jnp.where(span != 0.0, span, 1.0)
    edge = (
        nd_plasma_separatrix_electron
        + (nd_plasma_pedestal_electron - nd_plasma_separatrix_electron)
        * (1.0 - rho)
        / safe_span
    )

    return jnp.where(inside, core, edge)


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


def calculate_parabolic_temperature_profile(
    rho, temp_plasma_electron_on_axis_kev, alphat
):
    """Electron temperature profile for `i_plasma_pedestal == 0`.

    Ports `TeProfile.calculate_profile_y`'s parabolic branch (source L425-433), which
    -- unlike its `NeProfile` counterpart -- does `return`, so it is a live arm.

    `(1 - rho**2)` is exactly 0 at the outermost grid point, where `0 ** alphat` is 0 in
    value but has a NaN derivative with respect to `alphat` (`x**y * log x` at `x = 0`).
    The base is therefore substituted and the result put back, giving derivative 0 there
    -- which is what PROCESS's own finite difference reports, since the floor pins the
    value at `TEMPERATURE_FLOOR_KEV` on both sides.

    That endpoint is also why the harness differentiates this function with respect to
    `rho` only on an interior grid: PROCESS's finite difference steps `rho` past 1, where
    `(negative) ** 1.45` is NaN in `numpy`. That is PROCESS leaving its own domain, not a
    disagreement.

    Parameters
    ----------
    rho :
        Normalised minor radius grid.
    temp_plasma_electron_on_axis_kev :
        Central electron temperature (keV); `t0` in the source.
    alphat :
        Temperature peaking parameter.

    Returns
    -------
    :
        `profile_y` -- electron temperature (keV), floored at `TEMPERATURE_FLOOR_KEV`.
    """
    base = 1.0 - rho**2
    safe_base = jnp.where(base > 0.0, base, 1.0)
    shape = jnp.where(base > 0.0, safe_base**alphat, 0.0)
    return jnp.maximum(temp_plasma_electron_on_axis_kev * shape, TEMPERATURE_FLOOR_KEV)


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


def calculate_pedestal_temperature_profile(
    rho,
    radius_plasma_pedestal_temp_norm,
    temp_plasma_electron_on_axis_kev,
    temp_plasma_pedestal_kev,
    temp_plasma_separatrix_kev,
    alphat,
    tbeta,
):
    """Electron temperature profile for `i_plasma_pedestal != 0`.

    Ports `TeProfile.calculate_profile_y`'s pedestal branch (source L435-456). Same shape
    as `calculate_density_profile` with two differences the source really does have: the
    core exponent is `tbeta` rather than a hard-coded 2, and a negative value anywhere in
    the profile is fatal.

    Three guards a traced port needs and the masked source does not:

    - `(rho / rped) ** tbeta` at `rho == 0` is 0 in value but NaN in `d/d(tbeta)`.
      Substituted, then put back.
    - `(1 - (rho/rped)**tbeta) ** alphat` has a negative base wherever `rho > rped`, which
      PROCESS never evaluates. Substituted, then put back.
    - `(1 - rho) / (1 - rped)` divides by zero at `rped == 1`. Substituted.

    **Domain guard.** PROCESS raises `ProcessValueError("Negative temperature in plasma
    profile")` if any point comes out negative (source L455-456). A traced function cannot
    raise on a data-dependent condition, so the whole profile is returned non-finite
    there instead -- the harness's declared policy, and what
    `reference_domain_errors = (ProcessValueError,)` on the contract asserts.

    The source's `logger.info` when `t0 < tped` is not ported (diagnostic only).

    Parameters
    ----------
    rho :
        Normalised minor radius grid.
    radius_plasma_pedestal_temp_norm :
        Normalised minor radius of the temperature pedestal.
    temp_plasma_electron_on_axis_kev :
        Central electron temperature (keV); `t0` in the source.
    temp_plasma_pedestal_kev :
        Pedestal temperature (keV).
    temp_plasma_separatrix_kev :
        Separatrix temperature (keV).
    alphat :
        Temperature peaking parameter.
    tbeta :
        Second temperature exponent.

    Returns
    -------
    :
        `profile_y` -- electron temperature (keV) at each point of `rho`, or all-NaN
        where PROCESS would raise.
    """
    inside = rho <= radius_plasma_pedestal_temp_norm

    ratio = rho / radius_plasma_pedestal_temp_norm
    safe_ratio = jnp.where(ratio > 0.0, ratio, 1.0)
    ratio_pow = jnp.where(ratio > 0.0, safe_ratio**tbeta, 0.0)

    base = 1.0 - ratio_pow
    safe_base = jnp.where(base > 0.0, base, 1.0)
    shape = jnp.where(base > 0.0, safe_base**alphat, 0.0)
    core = (
        temp_plasma_pedestal_kev
        + (temp_plasma_electron_on_axis_kev - temp_plasma_pedestal_kev) * shape
    )

    span = 1.0 - radius_plasma_pedestal_temp_norm
    safe_span = jnp.where(span != 0.0, span, 1.0)
    edge = (
        temp_plasma_separatrix_kev
        + (temp_plasma_pedestal_kev - temp_plasma_separatrix_kev)
        * (1.0 - rho)
        / safe_span
    )

    profile_y = jnp.where(inside, core, edge)

    # PROCESS raises here; a traced port returns non-finite instead. Poisoning the whole
    # array rather than the offending points mirrors the source, whose exception discards
    # the entire profile.
    return jnp.where(jnp.min(profile_y) < 0.0, jnp.nan, profile_y)


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


def ncore(
    radius_plasma_pedestal_density_norm,
    nd_plasma_pedestal_electron,
    nd_plasma_separatrix_electron,
    nd_plasma_electrons_vol_avg,
    alphan,
):
    """Central electron density of a pedestalised profile (m^-3).

    Ports `NeProfile.ncore` (source L213-284), name kept. The closed form is the
    pedestal profile integrated against the torus volume element `rho drho` over the core
    and edge regions, rearranged for `n_0`.

    **The floor is part of the model, not error handling.** When the closed form goes
    negative the source substitutes `1e-6` and logs an error saying the run may not have
    converged -- deliberately, "allows solver to continue". Ported as a `jnp.where`, which
    reproduces both the value and its derivative (identically zero in the floored region,
    which is what PROCESS's own finite difference reports there too).

    Parameters
    ----------
    radius_plasma_pedestal_density_norm :
        Normalised minor radius of the pedestal.
    nd_plasma_pedestal_electron :
        Pedestal density (m^-3).
    nd_plasma_separatrix_electron :
        Separatrix density (m^-3).
    nd_plasma_electrons_vol_avg :
        Volume-averaged electron density (m^-3); `nav` in the source.
    alphan :
        Density peaking parameter.

    Returns
    -------
    :
        `nd_plasma_electron_on_axis`.

    References
    ----------
    Jean, J. (2011). HELIOS: A Zero-Dimensional Tool for Next Step and Reactor Studies.
    Fusion Science and Technology, 59(2), 308-349.
    """
    rped = radius_plasma_pedestal_density_norm
    value = (
        1.0
        / (3.0 * rped**2)
        * (
            3.0 * nd_plasma_electrons_vol_avg * (1.0 + alphan)
            + nd_plasma_separatrix_electron * (1.0 + alphan) * (-2.0 + rped + rped**2)
            - nd_plasma_pedestal_electron
            * ((1.0 + alphan) * (1.0 + rped) + (alphan - 2.0) * rped**2)
        )
    )
    return jnp.where(value < 0.0, NCORE_FLOOR, value)


def tcore(
    radius_plasma_pedestal_temp_norm,
    temp_plasma_pedestal_kev,
    temp_plasma_separatrix_kev,
    temp_plasma_electron_vol_avg_kev,
    alphat,
    tbeta,
):
    """Central electron temperature of a pedestalised profile (keV).

    Ports `TeProfile.tcore` (source L458-529), name kept. Same construction as `ncore`:
    the pedestal profile integrated against `rho drho` and rearranged for `T_0`. The
    `beta(1 + alphat, 2 / tbeta)` factor is where the core region's integral closes.

    Unlike `ncore` there is **no floor** -- `tcore` can return a negative or absurd
    temperature and the source does not check. The check happens one call later, in
    `calculate_pedestal_temperature_profile`, which raises if the resulting profile has a
    negative point. Noted rather than corrected.

    `sp.special.beta` has no `jax.scipy.special` equivalent; `plasma_profiles._beta`
    supplies it via `gammaln`, agreeing to ~2e-15 relative.

    Parameters
    ----------
    radius_plasma_pedestal_temp_norm :
        Normalised minor radius of the temperature pedestal.
    temp_plasma_pedestal_kev :
        Pedestal temperature (keV).
    temp_plasma_separatrix_kev :
        Separatrix temperature (keV).
    temp_plasma_electron_vol_avg_kev :
        Volume-averaged temperature (keV); `tav` in the source.
    alphat :
        Temperature peaking parameter.
    tbeta :
        Second temperature exponent.

    Returns
    -------
    :
        `temp_plasma_electron_on_axis_kev`.

    References
    ----------
    Jean, J. (2011). HELIOS: A Zero-Dimensional Tool for Next Step and Reactor Studies.
    Fusion Science and Technology, 59(2), 308-349.
    """
    rped = radius_plasma_pedestal_temp_norm
    return temp_plasma_pedestal_kev + (
        tbeta
        * (
            3.0 * temp_plasma_electron_vol_avg_kev
            + temp_plasma_separatrix_kev * (-2.0 + rped + rped**2)
            - temp_plasma_pedestal_kev * (1.0 + rped + rped**2)
        )
    ) / (6.0 * rped**2 * _beta(1.0 + alphat, 2.0 / tbeta))


def calculate_parabolic_on_axis_densities(
    nd_plasma_electrons_vol_avg, nd_plasma_ions_total_vol_avg, alphan
):
    """On-axis electron and ion densities for `i_plasma_pedestal == 0`.

    Ports `NeProfile.set_physics_variables`'s parabolic arm (source L335-342) plus the
    unconditional ion line (L354-358) that follows both arms.

    **This unit is the real owner of these two fields.**
    `plasma_profiles.parabolic_parameterisation` writes them again forty lines later, from
    `nd_vol * (1 + alphan)` and `nd_ion_vol * (1 + alphan)` -- algebraically identical
    after substituting the ion line, but a `redundant-duplicate-write`, and unlike this
    one it has no pedestal counterpart. Unit #12's port drops that rewrite; this function
    is what replaces it.

    Parameters
    ----------
    nd_plasma_electrons_vol_avg :
        Volume-averaged electron density (m^-3).
    nd_plasma_ions_total_vol_avg :
        Volume-averaged total ion density (m^-3).
    alphan :
        Density peaking parameter.

    Returns
    -------
    tuple
        `(nd_plasma_electron_on_axis, nd_plasma_ions_on_axis)`.
    """
    nd_plasma_electron_on_axis = nd_plasma_electrons_vol_avg * (1.0 + alphan)
    nd_plasma_ions_on_axis = (
        nd_plasma_ions_total_vol_avg
        / nd_plasma_electrons_vol_avg
        * nd_plasma_electron_on_axis
    )
    return nd_plasma_electron_on_axis, nd_plasma_ions_on_axis


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


def calculate_pedestal_on_axis_densities(
    radius_plasma_pedestal_density_norm,
    nd_plasma_pedestal_electron,
    nd_plasma_separatrix_electron,
    nd_plasma_electrons_vol_avg,
    nd_plasma_ions_total_vol_avg,
    alphan,
):
    """On-axis electron and ion densities for `i_plasma_pedestal != 0`.

    Ports `NeProfile.set_physics_variables`'s pedestal arm (source L343-353) plus the
    unconditional ion line. The electron value comes from `ncore`, floor included.

    Parameters
    ----------
    radius_plasma_pedestal_density_norm, nd_plasma_pedestal_electron,
    nd_plasma_separatrix_electron, nd_plasma_electrons_vol_avg, alphan :
        As for `ncore`.
    nd_plasma_ions_total_vol_avg :
        Volume-averaged total ion density (m^-3).

    Returns
    -------
    tuple
        `(nd_plasma_electron_on_axis, nd_plasma_ions_on_axis)`.
    """
    nd_plasma_electron_on_axis = ncore(
        radius_plasma_pedestal_density_norm,
        nd_plasma_pedestal_electron,
        nd_plasma_separatrix_electron,
        nd_plasma_electrons_vol_avg,
        alphan,
    )
    nd_plasma_ions_on_axis = (
        nd_plasma_ions_total_vol_avg
        / nd_plasma_electrons_vol_avg
        * nd_plasma_electron_on_axis
    )
    return nd_plasma_electron_on_axis, nd_plasma_ions_on_axis


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


def calculate_parabolic_on_axis_temperatures(
    temp_plasma_electron_vol_avg_kev, temp_plasma_ion_vol_avg_kev, alphat
):
    """On-axis electron and ion temperatures for `i_plasma_pedestal == 0` (keV).

    Ports `TeProfile.set_physics_variables`'s parabolic arm (source L533-540) plus the
    unconditional ion line (L554-558). The same `redundant-duplicate-write` note as
    `calculate_parabolic_on_axis_densities` applies: unit #12's
    `parabolic_parameterisation` rewrites both fields, this unit owns them.

    Parameters
    ----------
    temp_plasma_electron_vol_avg_kev :
        Volume-averaged electron temperature (keV).
    temp_plasma_ion_vol_avg_kev :
        Volume-averaged ion temperature (keV).
    alphat :
        Temperature peaking parameter.

    Returns
    -------
    tuple
        `(temp_plasma_electron_on_axis_kev, temp_plasma_ion_on_axis_kev)`.
    """
    temp_plasma_electron_on_axis_kev = temp_plasma_electron_vol_avg_kev * (1.0 + alphat)
    temp_plasma_ion_on_axis_kev = (
        temp_plasma_ion_vol_avg_kev
        / temp_plasma_electron_vol_avg_kev
        * temp_plasma_electron_on_axis_kev
    )
    return temp_plasma_electron_on_axis_kev, temp_plasma_ion_on_axis_kev


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


def calculate_pedestal_on_axis_temperatures(
    radius_plasma_pedestal_temp_norm,
    temp_plasma_pedestal_kev,
    temp_plasma_separatrix_kev,
    temp_plasma_electron_vol_avg_kev,
    temp_plasma_ion_vol_avg_kev,
    alphat,
    tbeta,
):
    """On-axis electron and ion temperatures for `i_plasma_pedestal != 0` (keV).

    Ports `TeProfile.set_physics_variables`'s pedestal arm (source L541-552) plus the
    unconditional ion line. The electron value comes from `tcore`.

    Parameters
    ----------
    radius_plasma_pedestal_temp_norm, temp_plasma_pedestal_kev,
    temp_plasma_separatrix_kev, temp_plasma_electron_vol_avg_kev, alphat, tbeta :
        As for `tcore`.
    temp_plasma_ion_vol_avg_kev :
        Volume-averaged ion temperature (keV).

    Returns
    -------
    tuple
        `(temp_plasma_electron_on_axis_kev, temp_plasma_ion_on_axis_kev)`.
    """
    temp_plasma_electron_on_axis_kev = tcore(
        radius_plasma_pedestal_temp_norm,
        temp_plasma_pedestal_kev,
        temp_plasma_separatrix_kev,
        temp_plasma_electron_vol_avg_kev,
        alphat,
        tbeta,
    )
    temp_plasma_ion_on_axis_kev = (
        temp_plasma_ion_vol_avg_kev
        / temp_plasma_electron_vol_avg_kev
        * temp_plasma_electron_on_axis_kev
    )
    return temp_plasma_electron_on_axis_kev, temp_plasma_ion_on_axis_kev


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


def calculate_greenwald_density_fractions(
    nd_plasma_pedestal_electron,
    nd_plasma_separatrix_electron,
    plasma_current,
    rminor,
):
    """Pedestal and separatrix densities as fractions of the Greenwald limit.

    Ports `NeProfile.set_pedestal_and_separatrix_values`'s
    `i_nd_plasma_pedestal_separatrix == USER_INPUT` arm (source L294-313): the densities
    are the input, the fractions are derived.

    Parameters
    ----------
    nd_plasma_pedestal_electron :
        Pedestal electron density (m^-3).
    nd_plasma_separatrix_electron :
        Separatrix electron density (m^-3).
    plasma_current :
        Plasma current (A).
    rminor :
        Plasma minor radius (m).

    Returns
    -------
    tuple
        `(f_nd_plasma_pedestal_greenwald, f_nd_plasma_separatrix_greenwald)`.
    """
    limit = _greenwald_limit(plasma_current, rminor)
    return nd_plasma_pedestal_electron / limit, nd_plasma_separatrix_electron / limit


class GreenwaldDensityFractions(ExplicitFunction):
    """cottax node: `calculate_greenwald_density_fractions`.

    **Blocked, do not register yet**, for two reasons that compound:

    1. It is one arm of `.physics.i_nd_plasma_pedestal_separatrix`, whose other arm
       (`PedestalSeparatrixDensities`) owns the *inputs* this one reads and reads the
       *outputs* this one owns. The two arms are inverses, not competing producers.
    2. `set_pedestal_and_separatrix_values` is itself called only from
       `physics.py:365-368`, **inside** `if i_plasma_pedestal == PEDESTAL_PROFILE`. So the
       arm is nested under a second switch. `configuration.TOPOLOGY_SWITCHES` is a flat
       list of independent choices with no way to express "this switch only exists when
       that one has this value".

    The signature is settled; only the wiring is not.
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


def calculate_pedestal_separatrix_densities(
    f_nd_plasma_pedestal_greenwald,
    f_nd_plasma_separatrix_greenwald,
    plasma_current,
    rminor,
):
    """Pedestal and separatrix densities from their Greenwald fractions (m^-3).

    Ports `NeProfile.set_pedestal_and_separatrix_values`'s
    `i_nd_plasma_pedestal_separatrix == GREENWALD_FRACTION` arm (source L314-331), the
    exact inverse of `calculate_greenwald_density_fractions`. This is PROCESS's default
    (`i_nd_plasma_pedestal_separatrix = 1`).

    Parameters
    ----------
    f_nd_plasma_pedestal_greenwald :
        Pedestal density as a fraction of the Greenwald limit.
    f_nd_plasma_separatrix_greenwald :
        Separatrix density as a fraction of the Greenwald limit.
    plasma_current :
        Plasma current (A).
    rminor :
        Plasma minor radius (m).

    Returns
    -------
    tuple
        `(nd_plasma_pedestal_electron, nd_plasma_separatrix_electron)`.
    """
    limit = _greenwald_limit(plasma_current, rminor)
    return (
        f_nd_plasma_pedestal_greenwald * limit,
        f_nd_plasma_separatrix_greenwald * limit,
    )


class PedestalSeparatrixDensities(ExplicitFunction):
    """cottax node: `calculate_pedestal_separatrix_densities`.

    **Blocked, do not register yet** -- the other arm of
    `.physics.i_nd_plasma_pedestal_separatrix`; see `GreenwaldDensityFractions` for both
    reasons.

    Worth noting for whoever unblocks it: this arm *writes*
    `.physics.nd_plasma_pedestal_electron` and `.physics.nd_plasma_separatrix_electron`,
    which `DensityProfile`, `PedestalOnAxisDensities` and unit #12's pedestal arm all
    read. Under the default switch value it is therefore a genuine upstream producer, not
    a leaf.
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
