"""Pure-functional port of `process/models/physics/density_limit.py`
(`PlasmaDensityLimit` -- the **tokamak** density limit).

Audit record: `functional_process/_audit/units/models/physics/density_limit.md`. Read it
first, especially "the eight are a computes-then-selects family" and "not the
stellarator unit" before trusting which formula is live on any given run.

**Not the same unit as `functional_process/cottax/stellarator/density_limits.py`**
(plural, already ported/registered as unit #3 of the stellarator scope). That module
ports `process/models/stellarator/density_limits.py` (`st_sudo_density_limit` and
friends) -- an entirely different PROCESS source file, a different physics model (the
Sudo scaling), and a different `DataStructure` producer for the *same* field,
`.physics.nd_plasma_electrons_max`. `CLAUDE.md`'s "difficulties" section names
`density_limit.py`'s pure/impure split as its exemplar of the extraction seam this
project is built on; this file is that exemplar's port, singular, tokamak only.

**Scope: `large_tokamak_eval.IN.DAT`'s `i_density_limit = 7` (GREENWALD) reference arm**
(`tests/regression/input_files/large_tokamak_eval.IN.DAT:289`; PROCESS's own default is
`8`, ASDEX_NEW, `process/data_structure/physics_variables.py:863` -- the reference file
overrides it). One occupant class per switch value, per `next_steps.md` §14.2 and the
wave coordinator's settled policy for computes-then-selects families
(`bootstrap_current.md`'s "## deviations" item 2, restated in the wave-1 brief): no
`i_density_limit` integer appears as a kwarg or inside any body here.

**`PlasmaDensityLimit.calculate_density_limit` computes all eight formulas
unconditionally** (`process/models/physics/density_limit.py:517-616`) and only then
indexes the chosen one via `get_density_limit_value`. All eight one-liner statics are
ported and Tier-1-tested below against PROCESS's own staticmethods directly (free,
strong oracle: `tests/unit/models/physics/test_physics.py::test_calculate_density_limit`
already records all eight expected array elements from one `large_tokamak_nof.IN.DAT`
input point -- "trivially cheap validated siblings" per the wave-1 brief, the same
situation `l_h_transition.md` records for its own 21-formula family). **Only the
GREENWALD arm is wired as an occupant node**; the other seven have no reader anywhere in
`process/` outside `PlasmaDensityLimit.output` and the model-selection dispatch itself
(measured by `grep`, see the audit record) -- dead work, matching
`bootstrap_current.md`'s treatment of its own thirteen unselected scalings.

**`.physics.nd_plasma_electron_max_array[6]` (the Greenwald element) is not actually
switch-gated in PROCESS** -- it is computed on every call regardless of
`i_density_limit`, and three real non-reporting readers depend on it whatever the switch
says: this unit's own `f_nd_plasma_greenwald` bookkeeping (below), constraint 76
(`Eich critical separatrix density`), and `.tokamak.bootstrap_current`'s Sauter arm
(`n_greenwald`, `bootstrap_current.py:245` -- another agent's ported unit, which already
declares this same array element as a boundary read). So `GreenwaldDensityLimit` below is
**not** gated on `i_density_limit == 7` the way a "one occupant per switch value" node
normally would be -- it is unconditional, and `EnforcedDensityLimitGreenwald` (the actual
`i_density_limit == 7` occupant) is a separate, trivial node that only *selects* it,
mirroring `get_density_limit_value`'s own two-step shape (compute the array, then index
into it) exactly rather than collapsing the two into one node. See the audit record's
"why two nodes, not one" for the alternative considered and rejected.

`.physics.f_nd_plasma_greenwald` (the Greenwald *fraction*) is also unconditional --
`PlasmaDensityLimit.run` assigns it from the array's Greenwald element regardless of
`i_density_limit` (`process/models/physics/density_limit.py:106-109`).
"""

import dataclasses

from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    FromExactly,
    ModelNamespace,
    Output,
    OutputInto,
)

from functional_process.cottax.paths import physics
from functional_process.models.physics.density_limit import (
    calculate_asdex_density_limit,
    calculate_asdex_new_density_limit,
    calculate_borrass_iter_i_density_limit,
    calculate_borrass_iter_ii_density_limit,
    calculate_greenwald_density_limit,
    calculate_greenwald_fraction,
    calculate_hugill_murakami_density_limit,
    calculate_jet_edge_radiation_density_limit,
    calculate_jet_simple_density_limit,
    select_enforced_density_limit_greenwald,
)

__all__ = [
    "calculate_asdex_density_limit",
    "calculate_asdex_new_density_limit",
    "calculate_borrass_iter_i_density_limit",
    "calculate_borrass_iter_ii_density_limit",
    "calculate_hugill_murakami_density_limit",
    "calculate_jet_edge_radiation_density_limit",
    "calculate_jet_simple_density_limit",
]


class GreenwaldDensityLimit(ExplicitFunction):
    """Unconditional producer of `.physics.nd_plasma_electron_max_array[6]`.

    Present in the graph regardless of `i_density_limit`'s value -- see the module
    docstring's "not actually switch-gated". `nd_plasma_electron_max_array_7` (the
    attribute name, not the PROCESS field) follows `constraints.py`'s own convention
    for this exact element: the PROCESS docstring label is Fortran 1-indexed
    (`nd_plasma_electron_max_array(7)`), the storage index is 0-indexed (`[6]`) --
    record both, `_audit/naming_convention.md` § "Array elements".
    """

    nd_plasma_electron_max_array_7 = Output(physics.nd_plasma_electron_max_array[6])

    def __call__(self, plasma_current=From(physics), rminor=From(physics)):
        return calculate_greenwald_density_limit(c_plasma=plasma_current, rminor=rminor)


class EnforcedDensityLimitGreenwald(ExplicitFunction):
    """The `i_density_limit == 7` (GREENWALD) occupant: selects the array element
    `GreenwaldDensityLimit` already produced. Owns `.physics.nd_plasma_electrons_max`,
    the field constraint 5's body reads (`process/core/solver/constraints.py:444-476`
    -- a constraint body, not a graph node; see the audit record).
    """

    nd_plasma_electrons_max = OutputInto(physics)

    def __call__(
        self,
        nd_plasma_electron_max_array_7=FromExactly(
            physics.nd_plasma_electron_max_array[6]
        ),
    ):
        return select_enforced_density_limit_greenwald(nd_plasma_electron_max_array_7)


class GreenwaldFraction(ExplicitFunction):
    """Unconditional producer of `.physics.f_nd_plasma_greenwald`."""

    f_nd_plasma_greenwald = OutputInto(physics)

    def __call__(
        self,
        nd_plasma_electron_line=From(physics),
        nd_plasma_electron_max_array_7=FromExactly(
            physics.nd_plasma_electron_max_array[6]
        ),
    ):
        return calculate_greenwald_fraction(
            nd_plasma_electron_line, nd_plasma_electron_max_array_7
        )


class TokamakDensityLimit(ModelNamespace):
    """`.tokamak.density_limit` -- three slots, one switched.

    `GreenwaldDensityLimit` and `GreenwaldFraction` are unconditional (the module
    docstring's "not actually switch-gated": PROCESS fills the whole
    `nd_plasma_electron_max_array` family regardless of `i_density_limit`); only the
    *enforced* limit -- the element constraint 5 reads -- answers the switch.
    """

    greenwald_density_limit: GreenwaldDensityLimit = GreenwaldDensityLimit()
    """Unconditional producer of `.physics.nd_plasma_electron_max_array[6]`."""

    enforced_density_limit: EnforcedDensityLimitGreenwald = dataclasses.field(
        kw_only=True
    )
    """`.physics.i_density_limit` -- eight values, one occupant. `7` (GREENWALD,
    `large_tokamak_eval.IN.DAT:289`) is written; `1`-`6` and `8` are UNPORTED (each
    formula is ported and Tier-1-tested, no occupant node -- `density_limit.md`
    "## UNPORTED")."""

    greenwald_fraction: GreenwaldFraction = GreenwaldFraction()
    """Unconditional producer of `.physics.f_nd_plasma_greenwald`."""
