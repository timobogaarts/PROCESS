"""Pure-functional port of the TF coil power conversion sub-unit of
`process/models/power.py` (registry unit #14, chunk A).

Audit record: `functional_process/_audit/units/models/power/tf_coil_power.md`. Covers
`Power.tfpwr` (2117-2287), `Power.tfpwcall` (2291-2330) and `Power.tfcpwr`
(2332-2629) -- see the audit record's data-footprint table for the full trace.

`Power.tfpwr` dispatches on the topology-changing switch `.tfcoil.i_tf_sup` to one of
two essentially disjoint computations, same shape as `vacuum.py`'s
`.vacuum.i_vacuum_pumping` dispatch (see `vacuum.md`):

- **`i_tf_sup != 1`** (resistive TF coil) -- straight-line algebra, no calls.
  `calculate_tf_power_resistive` below.
- **`i_tf_sup == 1`** (superconducting TF coil) -- `tfpwcall` folds `ettfmj`/`itfka`
  into `tfcpwr`'s call, itself straight-line algebra (one real branch, on whether the
  TF leg resistance `rptfc` is exactly zero -- see that function's docstring).
  `calculate_tf_power_superconducting` below.

Both are tier-1: no `self.data` access once ported, no internal iteration, no calls
into any other model.
"""

import jax.numpy as jnp
import numpy as np
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.models.safe_math import safe_pow, safe_sqrt
from functional_process.paths import buildings, heat_transport, physics, tfcoil
from functional_process.power.tf_coil_power import (
    calculate_tf_power_resistive,
    calculate_tf_power_superconducting,
)
from functional_process.vocabulary import constants

# ruff's docstring rules treat `__all__` membership as the definition of "public" once
# one is present, so this lists every public name this module resolved before step 2 of
# `_audit/formulas_split.md` moved the pure functions out -- not just `jnp`/`np`/
# `safe_pow`/`safe_sqrt`/`constants`, which are unused now that their real uses left with
# the functions (see `power/electric_production.py`'s commit for why a partial list is
# the wrong move; this file's own first pass over-trimmed the import block and dropped
# these five without checking, a gap `_audit/formulas_split.md`'s name-preservation gate
# should have caught and did not, until a later cluster-wide re-check with clean caches).
__all__ = [
    "ExplicitFunction",
    "From",
    "OutputInto",
    "TfPowerResistive",
    "TfPowerSuperconducting",
    "buildings",
    "calculate_tf_power_resistive",
    "calculate_tf_power_superconducting",
    "constants",
    "heat_transport",
    "jnp",
    "np",
    "physics",
    "safe_pow",
    "safe_sqrt",
    "tfcoil",
]


class TfPowerResistive(ExplicitFunction):
    """cottax node: `calculate_tf_power_resistive`.

    Only reached when `.tfcoil.i_tf_sup != 1` -- a topology-changing switch resolved
    at graph-assembly time, per `_audit/naming_convention.md`, not represented as a
    field on this node (same convention as `vacuum.py`'s `VacuumPumpingSimple`/
    `calculate_vacuum_pumping_old` split on `.vacuum.i_vacuum_pumping`).
    """

    m_tf_bus = OutputInto(tfcoil)
    vtfkv = OutputInto(tfcoil)
    p_cp_resistive_mw = OutputInto(tfcoil)
    p_tf_leg_resistive_mw = OutputInto(tfcoil)
    p_tf_joints_resistive_mw = OutputInto(tfcoil)
    tfcmw = OutputInto(tfcoil)
    p_tf_electric_supplies_mw = OutputInto(heat_transport)

    def __call__(
        self,
        c_tf_turn=From(tfcoil),
        j_tf_bus=From(tfcoil),
        rho_tf_bus=From(tfcoil),
        len_tf_bus=From(tfcoil),
        n_tf_coils=From(tfcoil),
        res_tf_leg=From(tfcoil),
        p_cp_resistive=From(tfcoil),
        c_tf_total=From(tfcoil),
        p_tf_joints_resistive=From(tfcoil),
        p_tf_leg_resistive=From(tfcoil),
        etatf=From(heat_transport),
    ):
        return calculate_tf_power_resistive(
            c_tf_turn,
            j_tf_bus,
            rho_tf_bus,
            len_tf_bus,
            n_tf_coils,
            res_tf_leg,
            p_cp_resistive,
            c_tf_total,
            p_tf_joints_resistive,
            p_tf_leg_resistive,
            etatf,
        )


class TfPowerSuperconducting(ExplicitFunction):
    """cottax node: `calculate_tf_power_superconducting`.

    Only reached when `.tfcoil.i_tf_sup == 1` -- see `TfPowerResistive`'s docstring.
    """

    tfckw = OutputInto(tfcoil)
    len_tf_bus = OutputInto(tfcoil)
    drarea = OutputInto(tfcoil)
    tfcbv = OutputInto(buildings)
    p_tf_electric_supplies_mw = OutputInto(heat_transport)

    def __call__(
        self,
        c_tf_turn=From(tfcoil),
        e_tf_magnetic_stored_total_gj=From(tfcoil),
        n_tf_coils=From(tfcoil),
        rmajor=From(physics),
        v_tf_coil_dump_quench_kv=From(tfcoil),
        res_tf_leg=From(tfcoil),
        rho_tf_bus=From(tfcoil),
        etatf=From(heat_transport),
    ):
        return calculate_tf_power_superconducting(
            c_tf_turn,
            e_tf_magnetic_stored_total_gj,
            n_tf_coils,
            rmajor,
            v_tf_coil_dump_quench_kv,
            res_tf_leg,
            rho_tf_bus,
            etatf,
        )
