"""Pure-functional port of `process/models/cs_fatigue.py` (partial -- see "not ported").

Audit record: `functional_process/_audit/units/models/cs_fatigue.md`.

**Scope.** `surface_stress_intensity_factor` (2026-08-26) and `ncycle` (2026-08-30).
`embedded_stress_intensity_factor` is dead in `process/` (no caller outside its own
PROCESS unit test -- confirmed by `grep -rn "embedded_stress_intensity_factor"
process/`) and is not needed by `ncycle` either, so it is not ported (same "don't port
dead code" instruction the wave brief states for whole files, applied here to one
function).

**`ncycle` was the record's stop item for four days, and it is ported on the terms that
record's open question 1 settled** (DECIDED-DEFERRED, 2026-08-27): an eager
`lax.while_loop`, Tier-1 value agreement, and the gradient checks structurally excused
because `n_cycle` is a count. (The loop is a **masked `lax.scan`** since 2026-09-02 --
see `_MAX_CRACK_STEPS`; the tier and the value contract are unchanged.) What changed is only the "deferred" half -- that decision
rested on *"no reader needs `n_cycle` yet (constraint 90 is not active on any tracked
input and the CS stress chain feeding it is UNPORTED)"*, and both clauses have since
stopped being true. `stresses.py` landed the CS stress chain on 2026-08-27, and
`low_aspect_ratio_DEMO.IN.DAT` activates constraint 90 -- which, reading a
`.cs_fatigue.n_cycle` no node owned, evaluated `1 - 0 / n_cycle_min` = exactly
`+1.000000` with an identically zero gradient row, and stopped both of that
configuration's SAND cells at zero iterations. A constraint that is violated by a
constant cannot be steered, so nothing else about that machine could be measured either.

**The loop terminates, and not by luck.** `k_max` is `max(k_a, k_c)`, so one of the two
ratios `(k_a / k_max)`, `(k_c / k_max)` is exactly `1` on every pass and the
corresponding crack dimension advances by the full `delta`. `a` or `c` therefore
strictly increases towards a fixed bound each iteration, which bounds the trip count at
`(bound - start) / delta` -- a few hundred passes on the tracked inputs. A non-finite
input exits immediately instead, because `nan <= x` is `False`.

**That argument is now load-bearing rather than reassuring**: it is what licenses the
static bound in `_MAX_CRACK_STEPS`, which turned this from a `lax.while_loop` into a
masked `lax.scan` so that the whole graph can be differentiated in reverse mode at all
(`_audit/optimise_design.md` §31.9, §31.12). It also removes the hazard the original
form carried -- `lax.while_loop` has no iteration cap, so a loop that could stall would
stall the whole graph evaluation and not just this node.

The stop item's other two candidate resolutions (a bespoke Tier-2 `residual`, or a new
tier) are not revisited -- see the record for why the tier question was decided the way
it was.
"""

import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto
from jax import lax  # noqa: F401

from functional_process.models.cs_fatigue import (
    calculate_n_cycle,
    surface_stress_intensity_factor,  # noqa: F401
)
from functional_process.models.safe_math import safe_pow, safe_sqrt  # noqa: F401
from functional_process.cottax.paths import cs_fatigue, pf_coil, physics


def calculate_cs_fatigue_n_cycle_gated(
    f_c_plasma_inductive,
    stress_hoop_cs_inner,
    residual_sig_hoop,
    t_crack_vertical,
    dz_cs_turn_conduit,
    dr_cs_turn_conduit,
    paris_coefficient,
    paris_power_law,
    walker_coefficient,
    sf_vertical_crack,
    sf_radial_crack,
    fracture_toughness,
    sf_fast_fracture,
):
    """`CsFatigue`'s own `f_c_plasma_inductive > 0.0` gate around `calculate_n_cycle`
    (see that class's docstring for why the guard belongs to the binding), moved out of
    the declaration and into a named function -- `_audit/formulas_split.md` step 1.
    """
    return jnp.where(
        f_c_plasma_inductive > 0.0,
        calculate_n_cycle(
            stress_hoop_cs_inner,
            residual_sig_hoop,
            t_crack_vertical,
            dz_cs_turn_conduit,
            dr_cs_turn_conduit,
            paris_coefficient,
            paris_power_law,
            walker_coefficient,
            sf_vertical_crack,
            sf_radial_crack,
            fracture_toughness,
            sf_fast_fracture,
        ),
        0.0,
    )


class CsFatigue(ExplicitFunction):
    """cottax node: `.tokamak.cs_fatigue`. Owns `.cs_fatigue.n_cycle`, constraint 90's
    operand.

    **The `f_c_plasma_inductive` guard is this node's, not `calculate_n_cycle`'s.**
    PROCESS calls `ncycle` only when `.physics.f_c_plasma_inductive > 0.0e-4`
    (`pfcoil.py:3488`, "this is only valid for pulsed reactor design"), and leaves
    `.cs_fatigue.n_cycle` at its entering value otherwise. That guard lives at the call
    site, in `pfcoil.py`, so it belongs to the binding rather than to the ported
    function -- and it is a *computed* condition, not a switch, so `machine_from_indat`
    cannot resolve it the way it resolves `conditional-ownership-by-run-config` (a node
    either owns this field or does not, and which it is cannot be known until the
    physics has run). It is therefore a `jnp.where`, with `0.0` as the ungated value:
    `.cs_fatigue.n_cycle` is an output field with dataclass default `0.0`
    (`cs_fatigue_variables.py:15`) and no `IN.DAT` entry, so `0.0` *is* what PROCESS
    leaves there, exactly, rather than a stand-in for it. `t_crack_radial` could not be
    treated this way -- see `calculate_n_cycle`'s docstring for why it is not owned at
    all.
    """

    n_cycle = OutputInto(cs_fatigue)

    def __call__(
        self,
        stress_hoop_cs_inner=From(pf_coil),
        residual_sig_hoop=From(cs_fatigue),
        t_crack_vertical=From(cs_fatigue),
        dz_cs_turn_conduit=From(cs_fatigue),
        dr_cs_turn_conduit=From(cs_fatigue),
        paris_coefficient=From(cs_fatigue),
        paris_power_law=From(cs_fatigue),
        walker_coefficient=From(cs_fatigue),
        sf_vertical_crack=From(cs_fatigue),
        sf_radial_crack=From(cs_fatigue),
        fracture_toughness=From(cs_fatigue),
        sf_fast_fracture=From(cs_fatigue),
        f_c_plasma_inductive=From(physics),
    ):
        return calculate_cs_fatigue_n_cycle_gated(
            f_c_plasma_inductive,
            stress_hoop_cs_inner,
            residual_sig_hoop,
            t_crack_vertical,
            dz_cs_turn_conduit,
            dr_cs_turn_conduit,
            paris_coefficient,
            paris_power_law,
            walker_coefficient,
            sf_vertical_crack,
            sf_radial_crack,
            fracture_toughness,
            sf_fast_fracture,
        )
