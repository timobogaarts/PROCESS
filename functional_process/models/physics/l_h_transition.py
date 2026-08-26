"""Pure-functional port of `process/models/physics/l_h_transition.py`.

Audit record: `functional_process/_audit/units/models/physics/l_h_transition.md`. Read
it first, especially "the computes-then-selects shape" and "switches touched" before
trusting which arm is live on any given run.

`PlasmaConfinementTransition.l_h_threshold_power` unconditionally evaluates all 21
published L-H/L-I threshold scalings and returns them as a list; `run()` then picks one
element with `i_l_h_threshold - 1` and writes it to `.physics.p_l_h_threshold_mw`. This
is a "computes-then-selects" family in the sense `next_steps.md` §14.2 and
`bootstrap_current.md` use the term: the 21 formulas share no dispatch logic (there is
no `if`/`elif` inside `l_h_threshold_power` at all -- every arm is a plain function
call), so there is nothing to "split" in the sense `plasma_geometry.md`'s 13-branch
dispatcher needed splitting. What the settled policy (`next_steps.md` §14.2, restated in
the wave-1 brief) actually decides here is which single element the *selection* wires as
an occupant: one class per switch value, not a family node computing all 21 plus an
index.

All 21 pure formulas are ported below (cheap: each is already a `@staticmethod` with no
`self.data` access, and `tests/unit/models/physics/test_l_h_transition.py` already
validates every one of them against PROCESS with a legacy sample -- the same free-oracle
situation `confinement_time.md` records for its 48 scaling laws). Only the six occupant
node classes needed to answer `i_l_h_threshold` on `large_tokamak_eval.IN.DAT`
(`i_l_h_threshold = 19`, PROCESS's own default, `physics_variables.py:1234` -- the file
never sets it) are wired below: the reference arm (19, Martin08 aspect-corrected
nominal) plus its five siblings in the same family (6, 7, 8, 20, 21), whose reads-sets
are a validated subset/superset of arm 19's (see the audit record's "the Martin family"
table). The other 15 arms (ITER-1996 x3, Snipes 1997 x2, Snipes 2000 x3, Snipes 2000
closed divertor x3, Hubbard 2012 x3, Hubbard 2017 x1) have their pure formulas ported and
Tier-1-tested here, but are **not** wired as occupants in this pass -- each has a
reads-set that has not been independently checked against another live arm, so wiring
them is future, not-yet-done work, not a defect. See the audit record's UNPORTED table.

`dnla20 = 1e-20 * nd_plasma_electron_line` (`l_h_transition.py:131`) is the one derived
local every arm consumes. Unlike `confinement_time.py`'s `ConfinementScalingInputs`
(which earns its own node because ~40 laws share it across a much larger family), here
it is a single line consumed only by the six wired occupants in this file, so each
occupant's `__call__` computes it inline rather than through a shared node -- there is
no second consumer to justify one.

Every fractional exponent `0 < p < 1` uses `safe_pow` (`_audit/next_steps.md` §9's
`x ** p` derivative trap at `x == 0`); integer or `> 1` exponents keep the bare `**`,
matching `confinement_time.py`'s convention.
"""

import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.models.safe_math import safe_pow
from functional_process.paths import physics

# ---------------------------------------------------------------------------
# The 21 scaling laws. Direct ports of `PlasmaConfinementTransition`'s
# `@staticmethod`s, source order preserved -- matches
# `PlasmaConfinementTransitionModel`'s declaration order and
# `tests/unit/models/physics/test_l_h_transition.py`'s parametrisation, which this
# unit's legacy samples are lifted from verbatim.
# ---------------------------------------------------------------------------


def calculate_iter1996_nominal(dnla20, b_plasma_toroidal_on_axis, rmajor):
    """Nominal ITER-1996 L-H threshold. `PlasmaConfinementTransitionModel.
    ITER1996_NOMINAL` (1). `process/models/physics/l_h_transition.py:541-570`.
    """
    return 0.45 * safe_pow(dnla20, 0.75) * b_plasma_toroidal_on_axis * rmajor**2


def calculate_iter1996_upper(dnla20, b_plasma_toroidal_on_axis, rmajor):
    """Upper ITER-1996 L-H threshold. `PlasmaConfinementTransitionModel.
    ITER1996_UPPER` (2). `process/models/physics/l_h_transition.py:572-602`.
    """
    return 0.3960502816 * dnla20 * b_plasma_toroidal_on_axis * rmajor**2.5


def calculate_iter1996_lower(dnla20, b_plasma_toroidal_on_axis, rmajor):
    """Lower ITER-1996 L-H threshold. `PlasmaConfinementTransitionModel.
    ITER1996_LOWER` (3). `process/models/physics/l_h_transition.py:604-634`.
    """
    return 0.5112987149 * safe_pow(dnla20, 0.5) * b_plasma_toroidal_on_axis * rmajor**1.5


def calculate_snipes1997_iter(dnla20, b_plasma_toroidal_on_axis, rmajor):
    """Snipes 1997 ITER L-H threshold. `PlasmaConfinementTransitionModel.
    SNIPES1997_ITER` (4). `process/models/physics/l_h_transition.py:637-665`.
    """
    return (
        0.65
        * safe_pow(dnla20, 0.93)
        * safe_pow(b_plasma_toroidal_on_axis, 0.86)
        * rmajor**2.15
    )


def calculate_snipes1997_kappa(dnla20, b_plasma_toroidal_on_axis, rmajor, kappa):
    """Snipes 1997 ITER L-H threshold with a kappa factor.
    `PlasmaConfinementTransitionModel.SNIPES1997_KAPPA` (5).
    `process/models/physics/l_h_transition.py:667-705`.
    """
    return (
        0.42
        * safe_pow(dnla20, 0.80)
        * safe_pow(b_plasma_toroidal_on_axis, 0.90)
        * rmajor**1.99
        * safe_pow(kappa, 0.76)
    )


def calculate_martin08_nominal(
    dnla20, b_plasma_toroidal_on_axis, a_plasma_surface, m_ions_total_amu
):
    """Nominal Martin 2008 L-H threshold. `PlasmaConfinementTransitionModel.
    MARTIN08_NOMINAL` (6). `process/models/physics/l_h_transition.py:707-755`.
    """
    return (
        0.0488
        * safe_pow(dnla20, 0.717)
        * safe_pow(b_plasma_toroidal_on_axis, 0.803)
        * safe_pow(a_plasma_surface, 0.941)
        * (2.0 / m_ions_total_amu)
    )


def calculate_martin08_upper(
    dnla20, b_plasma_toroidal_on_axis, a_plasma_surface, m_ions_total_amu
):
    """Upper Martin 2008 L-H threshold. `PlasmaConfinementTransitionModel.
    MARTIN08_UPPER` (7). `process/models/physics/l_h_transition.py:757-804`.
    """
    return (
        0.05166240355
        * safe_pow(dnla20, 0.752)
        * safe_pow(b_plasma_toroidal_on_axis, 0.835)
        * safe_pow(a_plasma_surface, 0.96)
        * (2.0 / m_ions_total_amu)
    )


def calculate_martin08_lower(
    dnla20, b_plasma_toroidal_on_axis, a_plasma_surface, m_ions_total_amu
):
    """Lower Martin 2008 L-H threshold. `PlasmaConfinementTransitionModel.
    MARTIN08_LOWER` (8). `process/models/physics/l_h_transition.py:806-853`.
    """
    return (
        0.04609619059
        * safe_pow(dnla20, 0.682)
        * safe_pow(b_plasma_toroidal_on_axis, 0.771)
        * safe_pow(a_plasma_surface, 0.922)
        * (2.0 / m_ions_total_amu)
    )


def calculate_snipes2000_nominal(
    dnla20, b_plasma_toroidal_on_axis, rmajor, rminor, m_ions_total_amu
):
    """Nominal Snipes 2000 L-H threshold. `PlasmaConfinementTransitionModel.
    SNIPES2000_NOMINAL` (9). `process/models/physics/l_h_transition.py:855-905`.
    """
    return (
        1.42
        * safe_pow(dnla20, 0.58)
        * safe_pow(b_plasma_toroidal_on_axis, 0.82)
        * rmajor
        * safe_pow(rminor, 0.81)
        * (2.0 / m_ions_total_amu)
    )


def calculate_snipes2000_upper(
    dnla20, b_plasma_toroidal_on_axis, rmajor, rminor, m_ions_total_amu
):
    """Upper Snipes 2000 L-H threshold. `PlasmaConfinementTransitionModel.
    SNIPES2000_UPPER` (10). `process/models/physics/l_h_transition.py:907-958`.
    """
    return (
        1.547
        * safe_pow(dnla20, 0.615)
        * safe_pow(b_plasma_toroidal_on_axis, 0.851)
        * rmajor**1.089
        * safe_pow(rminor, 0.876)
        * (2.0 / m_ions_total_amu)
    )


def calculate_snipes2000_lower(
    dnla20, b_plasma_toroidal_on_axis, rmajor, rminor, m_ions_total_amu
):
    """Lower Snipes 2000 L-H threshold. `PlasmaConfinementTransitionModel.
    SNIPES2000_LOWER` (11). `process/models/physics/l_h_transition.py:960-1011`.
    """
    return (
        1.293
        * safe_pow(dnla20, 0.545)
        * safe_pow(b_plasma_toroidal_on_axis, 0.789)
        * safe_pow(rmajor, 0.911)
        * safe_pow(rminor, 0.744)
        * (2.0 / m_ions_total_amu)
    )


def calculate_snipes2000_closed_divertor_nominal(
    dnla20, b_plasma_toroidal_on_axis, rmajor, m_ions_total_amu
):
    """Nominal Snipes 2000 (closed divertor) L-H threshold.
    `PlasmaConfinementTransitionModel.SNIPES2000_CLOSED_DIVERTOR_NOMINAL` (12).
    `process/models/physics/l_h_transition.py:1013-1061`.
    """
    return (
        0.8
        * safe_pow(dnla20, 0.5)
        * safe_pow(b_plasma_toroidal_on_axis, 0.53)
        * rmajor**1.51
        * (2.0 / m_ions_total_amu)
    )


def calculate_snipes2000_closed_divertor_upper(
    dnla20, b_plasma_toroidal_on_axis, rmajor, m_ions_total_amu
):
    """Upper Snipes 2000 (closed divertor) L-H threshold.
    `PlasmaConfinementTransitionModel.SNIPES2000_CLOSED_DIVERTOR_UPPER` (13).
    `process/models/physics/l_h_transition.py:1063-1111`.
    """
    return (
        0.867
        * safe_pow(dnla20, 0.561)
        * safe_pow(b_plasma_toroidal_on_axis, 0.588)
        * rmajor**1.587
        * (2.0 / m_ions_total_amu)
    )


def calculate_snipes2000_closed_divertor_lower(
    dnla20, b_plasma_toroidal_on_axis, rmajor, m_ions_total_amu
):
    """Lower Snipes 2000 (closed divertor) L-H threshold.
    `PlasmaConfinementTransitionModel.SNIPES2000_CLOSED_DIVERTOR_LOWER` (14).
    `process/models/physics/l_h_transition.py:1113-1161`.
    """
    return (
        0.733
        * safe_pow(dnla20, 0.439)
        * safe_pow(b_plasma_toroidal_on_axis, 0.472)
        * rmajor**1.433
        * (2.0 / m_ions_total_amu)
    )


def calculate_hubbard2012_nominal(plasma_current, dnla20):
    """Nominal Hubbard 2012 L-I threshold. `PlasmaConfinementTransitionModel.
    HUBBARD2012_NOMINAL` (15). `process/models/physics/l_h_transition.py:1163-1187`.
    """
    return 2.11 * safe_pow(plasma_current / 1.0e6, 0.94) * safe_pow(dnla20, 0.65)


def calculate_hubbard2012_upper(plasma_current, dnla20):
    """Upper Hubbard 2012 L-I threshold. `PlasmaConfinementTransitionModel.
    HUBBARD2012_UPPER` (17). `process/models/physics/l_h_transition.py:1189-1213`.
    `plasma_current`'s exponent (`1.18`) is `> 1`, so it keeps the bare `**`.
    """
    return 2.11 * (plasma_current / 1.0e6) ** 1.18 * safe_pow(dnla20, 0.83)


def calculate_hubbard2012_lower(plasma_current, dnla20):
    """Lower Hubbard 2012 L-I threshold. `PlasmaConfinementTransitionModel.
    HUBBARD2012_LOWER` (16). `process/models/physics/l_h_transition.py:1216-1239`.
    """
    return 2.11 * safe_pow(plasma_current / 1.0e6, 0.7) * safe_pow(dnla20, 0.47)


def calculate_hubbard2017(dnla20, a_plasma_surface, b_plasma_toroidal_on_axis):
    """Hubbard 2017 L-I threshold. `PlasmaConfinementTransitionModel.
    HUBBARD2017_I_MODE` (18). `process/models/physics/l_h_transition.py:1241-1272`.
    """
    return 0.162 * dnla20 * a_plasma_surface * safe_pow(b_plasma_toroidal_on_axis, 0.26)


def _martin08_aspect_correction(aspect):
    """The aspect-ratio correction shared by the three `martin08_aspect_*` laws.

    `process/models/physics/l_h_transition.py:1328-1331` (and the two siblings' identical
    copies): `if aspect <= 2.7: ... else: 1.0` on `aspect`, a plain differentiable
    argument here, not a switch -- `needs-lax-cond-or-where` (see the audit record's JAX
    flags). `jnp.where` evaluates both branches; the "then" branch's own denominator,
    `1.0 - safe_pow(2.0 / (1.0 + aspect), 0.5)`, is singular only at `aspect == 1.0`,
    outside every physically meaningful aspect ratio and outside
    `ITERATION_VARIABLES[1]`'s declared bounds `(1.1, 10.0)` -- documented, not guarded,
    matching `plasma_geometry.md`'s convention for a singularity outside the sampled
    domain.
    """
    correction = 0.098 * aspect / (1.0 - safe_pow(2.0 / (1.0 + aspect), 0.5))
    return jnp.where(aspect <= 2.7, correction, 1.0)


def calculate_martin08_aspect_nominal(
    dnla20, b_plasma_toroidal_on_axis, a_plasma_surface, m_ions_total_amu, aspect
):
    """Nominal Martin 2008 L-H threshold, aspect-ratio corrected.
    `PlasmaConfinementTransitionModel.MARTIN08_ASPECT_NOMINAL` (19) -- PROCESS's own
    default (`physics_variables.py:1234`) and the live arm on
    `large_tokamak_eval.IN.DAT`, which never sets `i_l_h_threshold`.
    `process/models/physics/l_h_transition.py:1274-1340`.
    """
    return (
        0.0488
        * safe_pow(dnla20, 0.717)
        * safe_pow(b_plasma_toroidal_on_axis, 0.803)
        * safe_pow(a_plasma_surface, 0.941)
        * (2.0 / m_ions_total_amu)
        * _martin08_aspect_correction(aspect)
    )


def calculate_martin08_aspect_upper(
    dnla20, b_plasma_toroidal_on_axis, a_plasma_surface, m_ions_total_amu, aspect
):
    """Upper Martin 2008 L-H threshold, aspect-ratio corrected.
    `PlasmaConfinementTransitionModel.MARTIN08_ASPECT_UPPER` (20).
    `process/models/physics/l_h_transition.py:1342-1408`.
    """
    return (
        0.05166240355
        * safe_pow(dnla20, 0.752)
        * safe_pow(b_plasma_toroidal_on_axis, 0.835)
        * safe_pow(a_plasma_surface, 0.96)
        * (2.0 / m_ions_total_amu)
        * _martin08_aspect_correction(aspect)
    )


def calculate_martin08_aspect_lower(
    dnla20, b_plasma_toroidal_on_axis, a_plasma_surface, m_ions_total_amu, aspect
):
    """Lower Martin 2008 L-H threshold, aspect-ratio corrected.
    `PlasmaConfinementTransitionModel.MARTIN08_ASPECT_LOWER` (21).
    `process/models/physics/l_h_transition.py:1410-1476`.
    """
    return (
        0.04609619059
        * safe_pow(dnla20, 0.682)
        * safe_pow(b_plasma_toroidal_on_axis, 0.771)
        * safe_pow(a_plasma_surface, 0.922)
        * (2.0 / m_ions_total_amu)
        * _martin08_aspect_correction(aspect)
    )


# ---------------------------------------------------------------------------
# cottax nodes -- the Martin family: the reference arm (19) plus its five siblings
# whose reads-sets are a validated subset/equal set of arm 19's. Each owns
# `.physics.p_l_h_threshold_mw` directly, per `naming_convention.md` § "switches are not
# ports": `i_l_h_threshold` is read once at graph-build time to pick which one of these
# occupies the `.tokamak.l_h_transition` slot, and is not itself a `VarPath` on any of
# them. `l_h_threshold_powers` (the full 21-element reporting array) is not ported --
# only `output()` and `core/io/plot/summary.py` read it, both reporting paths outside
# this unit's scope.
# ---------------------------------------------------------------------------


class LHThresholdPower(ExplicitFunction):
    """The family that owns `.physics.p_l_h_threshold_mw`: one occupant per arm.

    `i_l_h_threshold` selects among 21 arms in PROCESS; only the six-member Martin
    family is wired here (see the module docstring for which and why). Consumed by
    constraints 15, 22 and 73 (`process/core/solver/constraints.py`), not by any other
    graph node -- `_audit/tokamak_boundary.md` records zero boundary reads for this slot
    today because that measurement only counts the plain compute graph, not constraints.
    """


class Martin08NominalLHThresholdPower(LHThresholdPower):
    """`i_l_h_threshold == 6`. No `aspect` read -- the aspect-ratio correction is not
    part of this arm.
    """

    p_l_h_threshold_mw = OutputInto(physics)

    def __call__(
        self,
        nd_plasma_electron_line=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
        a_plasma_surface=From(physics),
        m_ions_total_amu=From(physics),
    ):
        dnla20 = 1.0e-20 * nd_plasma_electron_line
        return calculate_martin08_nominal(
            dnla20, b_plasma_toroidal_on_axis, a_plasma_surface, m_ions_total_amu
        )


class Martin08UpperLHThresholdPower(LHThresholdPower):
    """`i_l_h_threshold == 7`."""

    p_l_h_threshold_mw = OutputInto(physics)

    def __call__(
        self,
        nd_plasma_electron_line=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
        a_plasma_surface=From(physics),
        m_ions_total_amu=From(physics),
    ):
        dnla20 = 1.0e-20 * nd_plasma_electron_line
        return calculate_martin08_upper(
            dnla20, b_plasma_toroidal_on_axis, a_plasma_surface, m_ions_total_amu
        )


class Martin08LowerLHThresholdPower(LHThresholdPower):
    """`i_l_h_threshold == 8`."""

    p_l_h_threshold_mw = OutputInto(physics)

    def __call__(
        self,
        nd_plasma_electron_line=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
        a_plasma_surface=From(physics),
        m_ions_total_amu=From(physics),
    ):
        dnla20 = 1.0e-20 * nd_plasma_electron_line
        return calculate_martin08_lower(
            dnla20, b_plasma_toroidal_on_axis, a_plasma_surface, m_ions_total_amu
        )


class Martin08AspectNominalLHThresholdPower(LHThresholdPower):
    """`i_l_h_threshold == 19` -- the reference arm on `large_tokamak_eval.IN.DAT`.

    One read more than `Martin08NominalLHThresholdPower`: `.physics.aspect`, for the
    correction factor. That is the whole difference between the two, matching
    `confinement_time.md`'s `PlasmaPowerLoss` siblings' shape.
    """

    p_l_h_threshold_mw = OutputInto(physics)

    def __call__(
        self,
        nd_plasma_electron_line=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
        a_plasma_surface=From(physics),
        m_ions_total_amu=From(physics),
        aspect=From(physics),
    ):
        dnla20 = 1.0e-20 * nd_plasma_electron_line
        return calculate_martin08_aspect_nominal(
            dnla20,
            b_plasma_toroidal_on_axis,
            a_plasma_surface,
            m_ions_total_amu,
            aspect,
        )


class Martin08AspectUpperLHThresholdPower(LHThresholdPower):
    """`i_l_h_threshold == 20`."""

    p_l_h_threshold_mw = OutputInto(physics)

    def __call__(
        self,
        nd_plasma_electron_line=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
        a_plasma_surface=From(physics),
        m_ions_total_amu=From(physics),
        aspect=From(physics),
    ):
        dnla20 = 1.0e-20 * nd_plasma_electron_line
        return calculate_martin08_aspect_upper(
            dnla20,
            b_plasma_toroidal_on_axis,
            a_plasma_surface,
            m_ions_total_amu,
            aspect,
        )


class Martin08AspectLowerLHThresholdPower(LHThresholdPower):
    """`i_l_h_threshold == 21`."""

    p_l_h_threshold_mw = OutputInto(physics)

    def __call__(
        self,
        nd_plasma_electron_line=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
        a_plasma_surface=From(physics),
        m_ions_total_amu=From(physics),
        aspect=From(physics),
    ):
        dnla20 = 1.0e-20 * nd_plasma_electron_line
        return calculate_martin08_aspect_lower(
            dnla20,
            b_plasma_toroidal_on_axis,
            a_plasma_surface,
            m_ions_total_amu,
            aspect,
        )
