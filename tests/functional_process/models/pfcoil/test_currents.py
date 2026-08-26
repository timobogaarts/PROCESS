"""Harness cases for `functional_process/models/pfcoil/currents.py`.

Audit record: `functional_process/_audit/units/models/pfcoil/currents.md`. One tier-1
contract: `calculate_efc_currents` against `PFCoil.efc`
(`process/models/pfcoil.py:1403-1506`), the only unit in this module that PROCESS exposes
as a callable. It carries `fixb`, `mtrx`, `PFCoil.solv` and `rsid` with it, so the SVD
and the whole matrix assembly are compared here rather than only through the chain.

The sample is the **equilibrium** call -- one field point at `(rmajor, 0)`, two
fixed-current divertor filaments, two two-coil groups to solve for. The
plasma-initiation call is the same function with 32 field points and 14 fixed filaments;
it is exercised by `test_masses.py`'s whole-`pfcoil()` chain contract instead, because
its `rpts`/`zpts` are not free inputs (PROCESS builds them from `rmajor`/`rminor`,
`pfcoil.py:377-384`) and differentiating 32 dependent components against PROCESS's finite
difference would be measuring a function of a point that cannot occur.

The other four units here -- `calculate_plasma_initiation_currents`,
`calculate_equilibrium_currents`, `calculate_cs_flux_swing`,
`calculate_time_point_currents` -- are inline blocks of `pfcoil()` with no separable
PROCESS callable, so their oracle is `pfcoil()` itself in `test_masses.py`. See
`currents.md` § tier signal.

**Value tolerance.** `TestCalculateEfcCurrents` loosens tier 1's default `rtol` from
`1e-12` to `5e-12`, with the reason attached to the `Tolerance` object below. Both sides
compute the same pseudo-inverse of the same matrix; they differ only in which LAPACK
driver call produces the decomposition (`scipy.linalg.svd` against
`jnp.linalg.svd`). Measured on the reference point, the group currents agree to
`2.3e-13` relative -- inside the default -- but that margin is a property of this
matrix's conditioning, not of the port, and a fuzz draw that squeezes the smallest
singular value has every right to use more of it.
"""

import numpy as np

from functional_process._harness import Tier1Contract, Tolerance, legacy_sample
from functional_process.models.pfcoil import LROW1
from functional_process.models.pfcoil.currents import calculate_efc_currents
from process.data_structure.pfcoil_variables import (
    N_PF_GROUPS_MAX,
    NFIXMX,
    NPTSMX,
)
from process.models.pfcoil import PFCoil

_R_FIX = np.array([5.566666666666666, 5.566666666666666])
_Z_FIX = np.array([9.644333333333332, -10.878217164127493])
_C_FIX = np.array([15720143.18611007, 17587384.305304807])
_R_GROUP = np.array([
    [16.868931216641325, 16.868931216641325],
    [15.359714853856374, 15.359714853856374],
])
_Z_GROUP = np.array([
    [2.6666666666666665, -2.6666666666666665],
    [7.466666666666666, -7.466666666666666],
])


def _around(base, fraction):
    """Elementwise `(lower, upper)` at `+-fraction` of `base`, sign-safe."""
    base = np.asarray(base, dtype=float)
    low = base * (1.0 - fraction)
    high = base * (1.0 + fraction)
    return np.minimum(low, high), np.maximum(low, high)


def _reference_efc_currents(
    rpts, zpts, brin, bzin, r_fix, z_fix, c_fix, r_group, z_group, alfa, n_in_group
):
    """`PFCoil.efc`, with the port's trimmed arrays padded back to PROCESS's shapes.

    `efc` reads no `self.data` at all -- it takes every array it touches as an argument
    and calls three module-level `@numba.njit` functions plus the `@staticmethod`
    `solv` (`process/models/pfcoil.py:1476-1506`) -- so the instance below carries no
    state and exists only because `efc` and `solv` are declared on the class.

    The three "work arrays" `efc` takes are pure output: `fixb` and `mtrx` both allocate
    fresh arrays and return them, so what is passed in is used only for its *shape*
    (`lrow1 = bfix.shape[0]`, `lcol1 = gmat.shape[1]`). Zeros of the right shape are
    therefore the whole contract.
    """
    npts = len(rpts)
    nfix = len(r_fix)
    n_groups = len(n_in_group)

    def _pad(values, width):
        out = np.zeros(width)
        out[: len(values)] = np.asarray(values, dtype=float)
        return out

    group_shape = (N_PF_GROUPS_MAX, r_group.shape[1])
    r_group_full = np.zeros(group_shape)
    z_group_full = np.zeros(group_shape)
    r_group_full[:n_groups] = r_group
    z_group_full[:n_groups] = z_group

    n_in_group_full = np.zeros(N_PF_GROUPS_MAX + 2, dtype=int)
    n_in_group_full[:n_groups] = np.asarray(n_in_group, dtype=int)

    model = PFCoil(cs_fatigue=None, cs_coil=None)
    ssq, ccls = model.efc(
        npts,
        _pad(rpts, NPTSMX),
        _pad(zpts, NPTSMX),
        _pad(brin, NPTSMX),
        _pad(bzin, NPTSMX),
        nfix,
        _pad(r_fix, NFIXMX),
        _pad(z_fix, NFIXMX),
        _pad(c_fix, NFIXMX),
        n_groups,
        n_in_group_full,
        r_group_full,
        z_group_full,
        alfa,
        np.zeros(LROW1),
        np.zeros((LROW1, N_PF_GROUPS_MAX), order="F"),
        np.zeros(LROW1),
    )
    return ssq, ccls


class TestCalculateEfcCurrents(Tier1Contract):
    """`calculate_efc_currents` -> `PFCoil.efc` (with `fixb`, `mtrx`, `solv`, `rsid`)."""

    audit_record = "models/pfcoil/currents.md"
    reference = _reference_efc_currents
    ported = calculate_efc_currents
    static_argnames = ("n_in_group",)

    value_tolerance = Tolerance(
        rtol=5e-12,
        atol=0.0,
        reason=(
            "the two sides build the same pseudo-inverse of the same matrix but get "
            "the decomposition from different LAPACK driver calls (scipy.linalg.svd "
            "against jnp.linalg.svd); measured 2.3e-13 relative on the reference "
            "point, which is a fact about this matrix's conditioning rather than about "
            "the port, so the bar is set a factor of ~20 above it rather than at it"
        ),
    )

    # The equilibrium solve on `large_tokamak_eval.IN.DAT`, read off a converged
    # in-process run: `ccls` comes back as `[-7.065e6, -4.845e6, 0, ...]`.
    samples = [
        legacy_sample(
            "large-tokamak-equilibrium",
            rpts=np.array([8.0]),
            zpts=np.array([0.0]),
            brin=np.array([0.0]),
            bzin=np.array([-0.7310770137585806]),
            r_fix=_R_FIX,
            z_fix=_Z_FIX,
            c_fix=_C_FIX,
            r_group=_R_GROUP,
            z_group=_Z_GROUP,
            alfa=5e-10,
            n_in_group=(2, 2),
        ),
    ]

    fuzz_fixed = {"n_in_group": (2, 2)}
    fuzz_bounds = {
        "rpts": (np.array([6.0]), np.array([11.0])),
        "zpts": (np.array([-1.0]), np.array([1.0])),
        "brin": (np.array([-0.2]), np.array([0.2])),
        "bzin": (np.array([-1.2]), np.array([-0.3])),
        "r_fix": _around(_R_FIX, 0.15),
        "z_fix": _around(_Z_FIX, 0.15),
        "c_fix": _around(_C_FIX, 0.20),
        "r_group": _around(_R_GROUP, 0.10),
        "z_group": _around(_Z_GROUP, 0.15),
        "alfa": (1e-10, 1e-9),
    }
