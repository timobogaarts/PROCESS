"""Pure-functional port of `process/models/stellarator/density_limits.py`.

Audit record: `functional_process/_audit/units/models/stellarator/density_limits.md`.
Only the two tier-1 functions are ported here. `power_at_ignition_point` is tier 2 and
remains blocked on registry unit #1 (`st_phys`), and `output` is a reporting shell.

Signatures follow the record's `## proposed signature(s)` verbatim, including its
argument spellings, so that the record stays the thing you read to know what a port
should look like.

`SudoDensityLimit`/`EcrhDensityLimit` are the `cottax` nodes. `EcrhDensityLimit`'s
`.stellarator.dlimit_ecrh`/`.bt_max_ecrh` outputs are invented names, not existing
PROCESS storage: `st_d_limit_ecrh`'s return values are locals in its caller
(`st_density_limits`), clamped and passed straight to `output()`, never written to
`data` (confirmed by grep, see the audit record's data-footprint table).
`i_plasma_pedestal` is a precondition (open question 2 in the record), not a `VarPath`
-- a static field on the node, per `naming_convention.md`'s "switches are not ports".
"""

import jax.numpy as jnp  # noqa: F401
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    OutputInto,
)

from functional_process.models.safe_math import (
    safe_sqrt,  # noqa: F401
)
from functional_process.paths import (
    physics,
    stellarator,
)
from functional_process.models.stellarator.density_limits import (
    calculate_ecrh_density_limit,  # noqa: F401
    calculate_ecrh_density_limit_parabolic,
    calculate_sudo_density_limit,
)


class SudoDensityLimit(ExplicitFunction):
    """cottax node: `calculate_sudo_density_limit`, unchanged, ports declared."""

    nd_plasma_electrons_max = OutputInto(physics)

    def __call__(
        self,
        b_plasma_toroidal_on_axis=From(physics),
        p_plasma_loss_mw=From(physics),
        rmajor=From(physics),
        rminor=From(physics),
        nd_plasma_electrons_vol_avg=From(physics),
        nd_plasma_electron_line=From(physics),
    ):
        return calculate_sudo_density_limit(
            b_plasma_toroidal_on_axis,
            p_plasma_loss_mw,
            rmajor,
            rminor,
            nd_plasma_electrons_vol_avg,
            nd_plasma_electron_line,
        )


class EcrhDensityLimit(ExplicitFunction):
    """cottax node: `calculate_ecrh_density_limit_parabolic`, ports declared.

    **`i_plasma_pedestal` was an `eqx.field(static=True)` here and is gone**
    (`_audit/next_steps.md` §14.2). It was the one static switch in the tree that could
    not disagree with anything: this node is a slot of `ProfileParameterisationParabolic`
    and nowhere else, so its container already *is* the answer, and `switch_kwarg_survey.md`
    band (c) said so -- *"should simply be deleted: its container occupant already encodes
    it and cannot disagree"*. There is no family here and no second occupant: PROCESS
    computes no ECRH density limit at any other value (`st_d_limit_ecrh` raises
    `UnboundLocalError`), which is why the pedestal occupant has no `ecrh_density_limit`
    slot at all rather than an `UNPORTED` entry.
    """

    dlimit_ecrh = OutputInto(stellarator)
    bt_max_ecrh = OutputInto(stellarator)

    def __call__(
        self,
        max_gyrotron_frequency=From(stellarator),
        b_plasma_toroidal_on_axis=From(physics),
    ):
        return calculate_ecrh_density_limit_parabolic(
            max_gyrotron_frequency, b_plasma_toroidal_on_axis
        )
