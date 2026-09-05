"""Harness cases for `functional_process/cottax/pfcoil/volt_seconds.py`.

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
from functional_process.cottax.pfcoil import (
    N_CS_PF_COILS,
    NGC2,
    PLASMA_INDEX,
    SPHERICAL_TOKAMAK_TOPOLOGY,
)
from functional_process.cottax.pfcoil.volt_seconds import (
    calculate_pf_cs_volt_seconds,
    calculate_pf_volt_seconds_no_central_solenoid,
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


# ---------------------------------------------------------------------------------
# The no-central-solenoid arm (`iohcl = 0`), 2026-08-31.
# ---------------------------------------------------------------------------------
_ST_IND_PLASMA_ROW = np.array([
    2.8361866096271888e-04,
    2.8361866096271888e-04,
    6.2887552328768379e-04,
    6.2887552328768379e-04,
    2.1581211129976063e-04,
    2.1581211129976063e-04,
    2.5003488482242812e-05,
    2.5003488482242812e-05,
])
"""`.pf_coil.ind_pf_cs_plasma_mutual[8, :8]` -- the plasma row of the eight-coil
spherical-tokamak matrix, produced by running
`test_inductance.py::TestCalculatePfPlasmaInductancesNoCentralSolenoid`'s own reference
(`PFCoil.induct(False)` at `iohcl = 0`) on that contract's legacy geometry. Eight
entries, not seven-plus-a-CS: index 7 is the last PF coil."""

_ST_C_TURN_ROWS = np.array([
    [-0.0, -0.0, -40000.0, -40000.0, -40000.0, -0.0],
    [-0.0, -0.0, -40000.0, -40000.0, -40000.0, -0.0],
    [-0.0, -0.0, -40000.0, -40000.0, -40000.0, -0.0],
    [-0.0, -0.0, -40000.0, -40000.0, -40000.0, -0.0],
    [-0.0, -0.0, -40000.0, -40000.0, -40000.0, -0.0],
    [-0.0, -0.0, -40000.0, -40000.0, -40000.0, -0.0],
    [-0.0, -0.0, -40000.0, -40000.0, -40000.0, -0.0],
    [-0.0, -0.0, -40000.0, -40000.0, -40000.0, -0.0],
])
"""`.pf_coil.c_pf_coil_turn[:8, :]` at the same point -- `pfcoil()`'s waveform fractions
times `copysign(c_pf_coil_turn_peak_input, c_pf_cs_coils_peak_ma)`.

**Degenerate in the burn term, and the reason is worth recording rather than tuning
away.** On `TestPFCoilChainSphericalTokamak`'s point the plasma-initiation solve returns
`ccl0 = 0` exactly, and `calculate_time_point_currents_no_central_solenoid`
(`currents.py:672-680`) builds both the flat-top and the end-of-burn currents as
`ccls - ccl0 * k`, so with `ccl0 = 0` the two are **equal**. The waveform fractions are
then `(0, -0, 1, 1, 1, 0)`, columns 2 and 4 coincide, and `vs_cs_pf_total_burn` is
exactly `0.0` on both sides -- a true value of this machine at this point, not a
cancellation the port introduced. `vs_cs_pf_total_pulse` is the ramp term and is not
degenerate. The fuzz draws move every entry of the matrix independently, which is what
separates columns 2 and 4 and puts a non-zero burn term under test."""


def _reference_pf_volt_seconds_no_central_solenoid(ind_plasma_row, c_pf_coil_turn_rows):
    """`PFCoil.vsec` at `iohcl = 0`, through the same `data` back-door.

    `nef = n_pf_cs_plasma_circuits - 1 = 8` here rather than `- 2`, so the PF loop
    covers every circuit but the plasma. `vs_cs_ramp`/`vs_cs_burn` are never assigned on
    this arm (`pfcoil.py:1647`, `:1677`, both guarded) and reach the totals at their
    `pfcoil_variables.py` storage default of `0.0` -- which is what the port reproduces
    by leaving them out of the sums rather than by adding a zero.
    """
    data = DataStructure()
    p = data.pf_coil
    n = SPHERICAL_TOKAMAK_TOPOLOGY.n_pf_coils
    plasma = SPHERICAL_TOKAMAK_TOPOLOGY.plasma_index

    data.build.iohcl = 0
    p.n_pf_cs_plasma_circuits = plasma + 1
    p.n_cs_pf_coils = SPHERICAL_TOKAMAK_TOPOLOGY.n_cs_pf_coils
    p.ind_pf_cs_plasma_mutual = np.zeros((NGC2, NGC2))
    p.ind_pf_cs_plasma_mutual[plasma, :n] = ind_plasma_row
    p.c_pf_coil_turn = np.zeros((NGC2, 6))
    p.c_pf_coil_turn[:n, :] = c_pf_coil_turn_rows

    model = PFCoil(cs_fatigue=None, cs_coil=None)
    model.data = data
    model.vsec()
    return p.vs_cs_pf_total_burn, p.vs_cs_pf_total_pulse


def _ported_pf_volt_seconds_no_central_solenoid(ind_plasma_row, c_pf_coil_turn_rows):
    """`calculate_pf_volt_seconds_no_central_solenoid` at storage width."""
    n = SPHERICAL_TOKAMAK_TOPOLOGY.n_pf_coils
    plasma = SPHERICAL_TOKAMAK_TOPOLOGY.plasma_index
    ind = jnp.zeros((NGC2, NGC2)).at[plasma, :n].set(ind_plasma_row)
    c_turn = jnp.zeros((NGC2, 6)).at[:n, :].set(c_pf_coil_turn_rows)
    return calculate_pf_volt_seconds_no_central_solenoid(
        ind_pf_cs_plasma_mutual=ind,
        c_pf_coil_turn=c_turn,
    )


class TestCalculatePfVoltSecondsNoCentralSolenoid(Tier1Contract):
    """`calculate_pf_volt_seconds_no_central_solenoid` -> `PFCoil.vsec` at `iohcl = 0`.

    Owed since 2026-08-30 (`_audit/next_steps.md` §20.5 item 1); verified bit-exact in a
    scratch script with nothing in the test tree holding it.

    **The absent CS is absence here too.** `vsec`'s two CS statements are guarded, and
    the port's answer is the PF sums alone -- so a port that had read the CS's row of the
    matrix as zeros would agree on the number and disagree on the read set. What
    separates the two is the *index range*: `nef` is `n_pf_cs_plasma_circuits - 1` on
    this arm, one larger than the conventional arm's, so the last PF coil is inside the
    sum where on `large_tokamak_eval` index 7 is the plasma. A port that had kept the
    conventional `- 2` would silently drop coil 7 from both totals, and this contract is
    what catches that.
    """

    audit_record = "models/pfcoil/volt_seconds.md"
    reference = _reference_pf_volt_seconds_no_central_solenoid
    ported = _ported_pf_volt_seconds_no_central_solenoid

    samples = [
        legacy_sample(
            "spherical-tokamak-plausible",
            ind_plasma_row=_ST_IND_PLASMA_ROW,
            c_pf_coil_turn_rows=_ST_C_TURN_ROWS,
        ),
    ]

    fuzz_bounds = {
        "ind_plasma_row": _around(_ST_IND_PLASMA_ROW, 0.20),
        "c_pf_coil_turn_rows": _around(_ST_C_TURN_ROWS, 0.20),
    }
