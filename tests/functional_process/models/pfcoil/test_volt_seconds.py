"""Harness cases for `functional_process/models/pfcoil/volt_seconds.py`.

Audit record: `functional_process/_audit/units/models/pfcoil/volt_seconds.md`.

One tier-1 contract here: `calculate_pf_cs_volt_seconds` against `PFCoil.vsec`
itself -- unlike most of the package, `vsec` *is* a separable PROCESS callable (an
instance method whose whole read set is two `.pf_coil` arrays plus the baked
topology), so the adapter seeds a `DataStructure` and calls the real method.

`calculate_pf_coil_turn_currents` is an inline block of `pfcoil()`
(`process/models/pfcoil.py:1082-1111`) with no separable PROCESS callable, so its
oracle is `pfcoil()` in `test_masses.py`'s whole-chain contract, which returns
`c_pf_coil_turn` on both sides since 2026-08-27 -- the same disposition
`test_currents.py`'s module docstring records for the other inline blocks.

The legacy point is `large_tokamak_eval.IN.DAT` at convergence, read off a live
in-process `SingleRun`: the plasma row of the mutual-inductance matrix and the seven
circuits' turn currents, whose volt-second totals are the
`vs_cs_pf_total_burn = -280.188` / `vs_cs_pf_total_pulse = -578.333` that
`cold_boundary.md` records (negative -- `calculate_burn_time` takes `abs()`).
"""

import jax.numpy as jnp
import numpy as np

from functional_process._harness import Tier1Contract, legacy_sample
from functional_process.models.pfcoil import N_CS_PF_COILS, NGC2, PLASMA_INDEX
from functional_process.models.pfcoil.volt_seconds import (
    calculate_pf_cs_volt_seconds,
)
from process.core.model import DataStructure
from process.models.pfcoil import PFCoil

_IND_PLASMA_ROW = np.array([
    0.00079175263827604,
    0.0007304101505631,
    0.00149602285088694,
    0.00149602285088694,
    0.0007308677607818,
    0.0007308677607818,
    0.00428048977060111,
])
"""`.pf_coil.ind_pf_cs_plasma_mutual[7, :7]` at convergence."""

_C_TURN_ROWS = np.array([
    [0.0, 3.2777833392831351e04, 4.0e04, 4.0e04, -1.1721446774755054e03, 0.0],
    [0.0, 3.7263325444208589e04, 4.0e04, 4.0e04, -6.8063586743457181e03, 0.0],
    [
        0.0,
        2.9666739966214186e03,
        -3.6273569105805662e04,
        -3.6273569105805662e04,
        -4.0e04,
        0.0,
    ],
    [
        0.0,
        2.9666739966214186e03,
        -3.6273569105805662e04,
        -3.6273569105805662e04,
        -4.0e04,
        0.0,
    ],
    [
        0.0,
        6.4414371268627804e02,
        -3.9190892887459559e04,
        -3.9190892887459559e04,
        -4.0e04,
        0.0,
    ],
    [
        0.0,
        6.4414371268627804e02,
        -3.9190892887459559e04,
        -3.9190892887459559e04,
        -4.0e04,
        0.0,
    ],
    [
        0.0,
        3.7396475861864797e04,
        6.9736085408608951e03,
        6.9736085408608951e03,
        -4.0e04,
        0.0,
    ],
])
"""`.pf_coil.c_pf_coil_turn[:7, :]` at convergence -- six PF circuits then the CS."""


def _around(base, fraction):
    """Elementwise `(lower, upper)` at `+-fraction` of `base`, sign-safe."""
    base = np.asarray(base, dtype=float)
    low = base * (1.0 - fraction)
    high = base * (1.0 + fraction)
    return np.minimum(low, high), np.maximum(low, high)


def _reference_pf_cs_volt_seconds(ind_plasma_row, c_pf_coil_turn_rows):
    """PROCESS's real `PFCoil.vsec`, through the `data` back-door.

    Only the fields `vsec` reads are seeded: the topology integers the package bakes
    (`iohcl = 1`, eight circuits, seven coils) and the two arrays, embedded at the
    indices the port slices. `cs_fatigue`/`cs_coil` are `None` because `vsec` never
    touches either injected sub-model.
    """
    data = DataStructure()
    p = data.pf_coil
    data.build.iohcl = 1
    p.n_pf_cs_plasma_circuits = PLASMA_INDEX + 1
    p.n_cs_pf_coils = N_CS_PF_COILS
    p.ind_pf_cs_plasma_mutual = np.zeros((NGC2, NGC2))
    p.ind_pf_cs_plasma_mutual[PLASMA_INDEX, :N_CS_PF_COILS] = ind_plasma_row
    p.c_pf_coil_turn = np.zeros((NGC2, 6))
    p.c_pf_coil_turn[:N_CS_PF_COILS, :] = c_pf_coil_turn_rows

    model = PFCoil(cs_fatigue=None, cs_coil=None)
    model.data = data
    model.vsec()
    return p.vs_cs_pf_total_burn, p.vs_cs_pf_total_pulse


def _ported_pf_cs_volt_seconds(ind_plasma_row, c_pf_coil_turn_rows):
    """`calculate_pf_cs_volt_seconds` with the free rows embedded at storage width."""
    ind = jnp.zeros((NGC2, NGC2)).at[PLASMA_INDEX, :N_CS_PF_COILS].set(ind_plasma_row)
    c_turn = jnp.zeros((NGC2, 6)).at[:N_CS_PF_COILS, :].set(c_pf_coil_turn_rows)
    return calculate_pf_cs_volt_seconds(
        ind_pf_cs_plasma_mutual=ind,
        c_pf_coil_turn=c_turn,
    )


class TestCalculatePfCsVoltSeconds(Tier1Contract):
    """`calculate_pf_cs_volt_seconds` -> `PFCoil.vsec`
    (`process/models/pfcoil.py:1615-1720`), `iohcl = 1` arm. Added 2026-08-27
    (`cold_boundary.md` producer 4).
    """

    audit_record = "models/pfcoil/volt_seconds.md"
    reference = _reference_pf_cs_volt_seconds
    ported = _ported_pf_cs_volt_seconds

    samples = [
        legacy_sample(
            "large_tokamak_eval-converged",
            ind_plasma_row=_IND_PLASMA_ROW,
            c_pf_coil_turn_rows=_C_TURN_ROWS,
        ),
    ]

    fuzz_bounds = {
        "ind_plasma_row": _around(_IND_PLASMA_ROW, 0.20),
        "c_pf_coil_turn_rows": _around(_C_TURN_ROWS, 0.20),
    }
