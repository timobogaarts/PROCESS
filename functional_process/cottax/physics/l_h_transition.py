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

`dnla20 = 1e-20 * nd_plasma_electron_line` is the one derived local every arm consumes.
Unlike `confinement_time.py`'s `ConfinementScalingInputs` (which earns its own node
because ~40 laws share it across a much larger family), here it is a single line
consumed only by the six wired occupants in this file, so it does **not** get a node --
there is no second consumer to justify one. It used to be computed inline in each
occupant's `__call__`; since 2026-09-05 it lives in a per-arm
`calculate_martin08_*_threshold_power` function instead, so that a declaration names its
implementation rather than containing it (`_audit/formulas_split.md`). That is a
different question from whether it deserves a node, and the answer to that one is still
no.

Every fractional exponent `0 < p < 1` uses `safe_pow` (`_audit/next_steps.md` §9's
`x ** p` derivative trap at `x == 0`); integer or `> 1` exponents keep the bare `**`,
matching `confinement_time.py`'s convention.
"""

from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.paths import physics
from functional_process.models.physics.l_h_transition import (
    calculate_hubbard2012_lower,
    calculate_hubbard2012_nominal,
    calculate_hubbard2012_upper,
    calculate_hubbard2017,
    calculate_iter1996_lower,
    calculate_iter1996_nominal,
    calculate_iter1996_upper,
    calculate_martin08_aspect_lower,
    calculate_martin08_aspect_lower_threshold_power,
    calculate_martin08_aspect_nominal,
    calculate_martin08_aspect_nominal_threshold_power,
    calculate_martin08_aspect_upper,
    calculate_martin08_aspect_upper_threshold_power,
    calculate_martin08_lower,
    calculate_martin08_lower_threshold_power,
    calculate_martin08_nominal,
    calculate_martin08_nominal_threshold_power,
    calculate_martin08_upper,
    calculate_martin08_upper_threshold_power,
    calculate_snipes1997_iter,
    calculate_snipes1997_kappa,
    calculate_snipes2000_closed_divertor_lower,
    calculate_snipes2000_closed_divertor_nominal,
    calculate_snipes2000_closed_divertor_upper,
    calculate_snipes2000_lower,
    calculate_snipes2000_nominal,
    calculate_snipes2000_upper,
)

__all__ = [
    "calculate_hubbard2012_lower",
    "calculate_hubbard2012_nominal",
    "calculate_hubbard2012_upper",
    "calculate_hubbard2017",
    "calculate_iter1996_lower",
    "calculate_iter1996_nominal",
    "calculate_iter1996_upper",
    "calculate_martin08_aspect_lower",
    "calculate_martin08_aspect_nominal",
    "calculate_martin08_aspect_upper",
    "calculate_martin08_lower",
    "calculate_martin08_nominal",
    "calculate_martin08_upper",
    "calculate_snipes1997_iter",
    "calculate_snipes1997_kappa",
    "calculate_snipes2000_closed_divertor_lower",
    "calculate_snipes2000_closed_divertor_nominal",
    "calculate_snipes2000_closed_divertor_upper",
    "calculate_snipes2000_lower",
    "calculate_snipes2000_nominal",
    "calculate_snipes2000_upper",
]


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
        return calculate_martin08_nominal_threshold_power(
            nd_plasma_electron_line,
            b_plasma_toroidal_on_axis,
            a_plasma_surface,
            m_ions_total_amu,
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
        return calculate_martin08_upper_threshold_power(
            nd_plasma_electron_line,
            b_plasma_toroidal_on_axis,
            a_plasma_surface,
            m_ions_total_amu,
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
        return calculate_martin08_lower_threshold_power(
            nd_plasma_electron_line,
            b_plasma_toroidal_on_axis,
            a_plasma_surface,
            m_ions_total_amu,
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
        return calculate_martin08_aspect_nominal_threshold_power(
            nd_plasma_electron_line,
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
        return calculate_martin08_aspect_upper_threshold_power(
            nd_plasma_electron_line,
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
        return calculate_martin08_aspect_lower_threshold_power(
            nd_plasma_electron_line,
            b_plasma_toroidal_on_axis,
            a_plasma_surface,
            m_ions_total_amu,
            aspect,
        )
