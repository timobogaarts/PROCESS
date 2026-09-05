"""Pure functions for the mutual and self inductances of the PF coils, the CS and
the plasma, extracted from `functional_process/cottax/pfcoil/inductance.py`.

That module still holds the graph declarations (`ExplicitFunction` occupants) that wire
these functions to `VarPath`s; read its module docstring for the cycle this closes and
the `noh` discontinuity discussion. The audit record is
`functional_process/_audit/units/models/pfcoil/inductance.md` and mirrors these
functions, not the declarations that call them.
"""

import jax
import jax.numpy as jnp

from functional_process.models.pfcoil import (
    CS_INDEX,
    N_COILS_IN_GROUP,
    N_CS_PF_COILS,
    N_PF_COILS,
    N_PF_GROUPS,
    NGC2,
    PLASMA_INDEX,
    SPHERICAL_TOKAMAK_TOPOLOGY,
)
from functional_process.models.pfcoil.fields import calculate_b_field_at_point
from functional_process.vocabulary import constants

RMU0 = constants.RMU0
"""Vacuum permeability (H/m), `process/core/constants.py:277`."""


NOH_PAD = 64
"""Length of the CS's segment arrays.

`induct` splits the solenoid into `noh = ceil(2 * z_cs_half / dr_cs)` pancake segments
(`pfcoil.py:1758-1765`), a count that moves with the design. A traced `noh` cannot size
an array, so the arrays are always `NOH_PAD` long and an `active` mask keeps the
arithmetic to exactly `noh` of them. `64` is the smallest power of two comfortably above
every `noh` the tracked configurations reach (27 to 32) and stands in for PROCESS's own
`nohmax = 200` clamp (`pfcoil.py:1733`), tightened; the clamp is applied in
`_cs_segments`, so a design that ran past the pad is pinned at `NOH_PAD` rather than
silently truncated."""


_DELTAR_FLOOR = 1.0e-6
"""`pfcoil.py:1819` -- what `deltar` becomes when the CS is thinner than one segment is
tall, so the Rosa-Grover two-filament split would need the square root of a negative
number. PROCESS's comment: "allows solver to continue and hopefully be constrained away
from this point"."""


_PF_SELF_INDUCTANCE_OFFSET = 1.75
"""The `- 1.75` in the PF coil self-inductance (`pfcoil.py:1978`), the standard
thin-ring result for a circular cross-section of equivalent radius `rl`."""


def _mutual_inductances(r_loop, z_loop, r_test, z_test):
    """Mutual inductance between each loop and a filament at the test point (H).

    `calculate_b_field_at_point`'s first return, with the currents set to zero because
    only the inductances are wanted -- which is exactly how `induct` calls it
    (`pfcoil.py:1827`, `cc` is never written and stays `np.zeros`). The field components
    it also returns are identically zero for the same reason, and are discarded here
    rather than computed and thrown away in the caller.

    **The loop and test roles are swapped relative to `induct` at two call sites**, so
    that thirty separate one-loop calls become one thirty-loop call. That is exact, not
    an approximation: `ind_mutual_array` depends on the pair only through
    `(r_test + r_loop)**2`, `(z_test - z_loop)**2` and `4 * r_test * r_loop`, each
    symmetric under the swap, and the leading `4.0 *` is a power of two so the product
    reassociates without changing a bit. The two kludges that are *not* symmetric
    (`dr = r_test - r_loop`) touch only `br`/`bz`, which are discarded.
    """
    ind, _br, _bz, _psi = calculate_b_field_at_point(
        r_current_loop=r_loop,
        z_current_loop=z_loop,
        c_current_loop=jnp.zeros_like(jnp.asarray(r_loop)),
        r_test_point=r_test,
        z_test_point=z_test,
    )
    return ind


_mutual_inductances_over_test_points = jax.vmap(
    _mutual_inductances, in_axes=(None, None, 0, 0)
)
"""`_mutual_inductances` batched over the test point, one loop set held fixed.

`induct` evaluates the same loop set at several test points -- the thirty CS filaments
against each PF group's representative coil, and every PF coil against every other -- and
each of those was a separate trace of the field kernel. There is no reduction inside
`_mutual_inductances` (it returns the per-loop array; the caller sums or indexes it), so
batching is only a leading axis."""


_mutual_inductances_over_loop_sets = jax.vmap(
    _mutual_inductances, in_axes=(0, None, None, None)
)
"""`_mutual_inductances` batched over the *loop set*, one test point held fixed -- the
inner and outer CS filament radii against the plasma filament."""


def calculate_solenoid_self_inductance(a, b, c, n):
    """Bunet's formula for the self inductance of a multi-layer solenoid (H).

    Ports `PFCoil.selfinductance`, `process/models/pfcoil.py:2837-2867`, unchanged.
    Renamed from `selfinductance` per `_audit/naming_convention.md`
    § "Function/module naming": the PROCESS name says neither what it is the
    self-inductance *of* nor that it is a fit, and it is the only name in this package
    that is not already derivable from the `VarPath` it produces.

    Parameters
    ----------
    a :
        Mean radius of the coil (m).
    b :
        Length of the coil (m).
    c :
        Radial winding thickness (m).
    n :
        Number of turns.

    Returns
    -------
    :
        Self inductance (H).

    Notes
    -----
    The mean radius appears in the denominator as `3.2 * c * b / a`, so `a == 0` is an
    unguarded division by zero: the value survives it (`0 / inf` is `0`, which is also
    the limit) but the derivative comes back `nan`, and
    `test_gradient_finite_at_zero` catches exactly that. The `jnp.where` pair below is
    `models/safe_math.py`'s idiom applied to a division rather than to a fractional
    power -- the inner `where` keeps the untaken branch's *argument* away from zero, so
    no `inf` is formed to be multiplied by a zero tangent. **Bit-identical for every
    `a != 0`**, since that branch evaluates the same expression; at `a == 0` it returns
    the same `0.0` with a derivative of `0`, which is the true limit
    (`f ~ k a^3 n^2 / (3.2 c b)` for small `a`, so `f' -> 0`).
    """
    at_zero = a == 0.0  # noqa: RUF069 -- the exact boundary is the point of the guard
    safe_a = jnp.where(at_zero, 1.0, a)
    inductance = (
        (1.0e-6 / 0.0254)
        * safe_a**2
        * n**2
        / (9.0 * safe_a + 10.0 * b + 8.4 * c + 3.2 * c * b / safe_a)
    )
    return jnp.where(at_zero, 0.0, inductance)


def _cs_segments(z_cs_half, dr_cs_edges, r_cs_middle):
    """The CS's pancake segments: `(noh, delzoh, zoh, roh, active)`.

    `noh = ceil(2 * z / dr)` as `induct` computes it (`pfcoil.py:1758-1765`), clipped
    into `[1, NOH_PAD]` -- PROCESS's own `min(noh, nohmax)`/`max(noh, 0)` guards
    (`:1774-1778`) with a tighter cap, and a lower clip of `1` rather than `0` because
    `xohpl / noh` divides by it.

    `noh` stays a **float**: `jnp.ceil` of a traced quantity is traceable, and the value
    is only ever divided by, never used as a length. The arrays are `NOH_PAD` long and
    `active` masks the tail out of both sums that consume them, so the trace has one
    shape for every design while the *arithmetic* sees exactly `noh` segments.

    `jnp.ceil` has a derivative of zero almost everywhere, so the tangent of everything
    downstream is the tangent of the piece the design is standing on. The discontinuity
    is entirely in the value, and only where a step is crossed -- measured at `3.6e-07`
    relative on `M[CS, plasma]` and `6.1e-06` on `M[PF, CS]`. See the module docstring
    for why that is the right trade against pinning the count.
    """
    ratio = 2.0 * z_cs_half / dr_cs_edges
    noh = jnp.clip(jnp.ceil(ratio), 1.0, float(NOH_PAD))
    delzoh = 2.0 * z_cs_half / noh
    index = jnp.arange(NOH_PAD)
    return (
        noh,
        delzoh,
        z_cs_half - delzoh * (0.5 + index),
        jnp.full(NOH_PAD, r_cs_middle),
        index < noh,
    )


def calculate_pf_cs_plasma_inductances(
    rmajor,
    ind_plasma,
    dr_cs,
    r_cs_middle,
    r_pf_coil_middle,
    z_pf_coil_middle,
    r_pf_coil_inner,
    r_pf_coil_outer,
    z_pf_coil_upper,
    z_pf_coil_lower,
    n_pf_coil_turns,
):
    """The mutual/self inductance matrix of the six PF coils, the CS and the plasma.

    Ports `PFCoil.induct`, `process/models/pfcoil.py:1721-1984`, for `iohcl = 1` and this
    run's `n_pf_coils_in_group = (1, 1, 2, 2)`. The `if not output: return` reporting
    block (`:1986-2019`) is dropped, as is `noh`'s `nohmax` clamp and its
    `logger.error`/`max(noh, 0)` guards (`:1767-1778`) -- with `noh` a graph-assembly
    constant of 30 there is nothing left to clamp.

    Four blocks, in the source's order, each writing a symmetric pair of entries:

    1. CS/plasma (`:1808-1856`). Each CS segment is split into two filaments at
       `reqv +- deltar` and the two mutual inductances averaged -- Rosa and Grover 1916
       p. 33, as the source's comment says.
    2. Plasma self (`:1859-1862`): `.physics.ind_plasma`, read straight through.
    3. PF/plasma (`:1867-1891`) and CS/PF (`:1911-1941`). Both evaluate the field point
       at the **last** coil of each group (`r_pf_coil_middle[ncoils - 1]`) and then give
       every coil in that group that same geometric factor scaled by its own turns. On
       this run each `i_pf_location = 3` group is an up/down symmetric pair, so the two
       coils have the same mutual inductance with anything on the midplane anyway; on a
       group that was *not* symmetric this would be an approximation PROCESS makes
       silently. Reproduced, and recorded.
    4. PF/PF (`:1949-1983`). Off-diagonal from the pairwise mutual inductances, diagonal
       from the thin-ring self-inductance of a coil whose cross-section is replaced by a
       circle of equal area (`rl = |dz| / sqrt(pi)`).

    Parameters
    ----------
    rmajor :
        Plasma major radius (m) -- the plasma filament's radius. `.physics.rmajor`.
    ind_plasma :
        Plasma self inductance (H). `.physics.ind_plasma`, produced by
        `models/physics/plasma_inductance.py::PlasmaVoltSecondRequirements`.
    dr_cs :
        CS radial thickness (m). `.build.dr_cs`. Read only by `deltar`; `noh` uses
        `r_pf_coil_outer[CS] - r_pf_coil_inner[CS]` for the same quantity
        (`pfcoil.py:1762-1764` vs `:1814`), which is a second spelling of it.
    r_cs_middle :
        CS mean radius (m). `.pf_coil.r_cs_middle`.
    r_pf_coil_middle, z_pf_coil_middle :
        Coil centres (m), seven entries -- six PF coils then the CS.
    r_pf_coil_inner, r_pf_coil_outer, z_pf_coil_upper, z_pf_coil_lower :
        Coil edges (m), seven entries.
    n_pf_coil_turns :
        Turns in each coil, seven entries.

    Returns
    -------
    :
        `.pf_coil.ind_pf_cs_plasma_mutual`, `(NGC2, NGC2)`. Entries outside the
        `8 x 8` circuit block are structural zeros, as PROCESS's own
        `ind_pf_cs_plasma_mutual[:, :] = 0.0` (`:1750`) leaves them.
    """
    ind = jnp.zeros((NGC2, NGC2))

    z_cs_half = z_pf_coil_upper[CS_INDEX]
    noh, delzoh, zoh, roh, active = _cs_segments(
        z_cs_half=z_cs_half,
        dr_cs_edges=r_pf_coil_outer[CS_INDEX] - r_pf_coil_inner[CS_INDEX],
        r_cs_middle=r_cs_middle,
    )

    # --- 1. Central Solenoid / plasma -------------------------------------------------
    # `deltar` needs `dr_cs >= delzoh`; below that PROCESS substitutes a small positive
    # number. The double `jnp.where` keeps the untaken branch's radicand positive, so no
    # `nan` leaks into the tangent (`models/safe_math.py`'s idiom, same reason).
    wide_enough = dr_cs >= delzoh
    radicand = (dr_cs**2 - delzoh**2) / 12.0
    deltar = jnp.where(
        wide_enough,
        jnp.sqrt(jnp.where(wide_enough, radicand, 1.0)),
        _DELTAR_FLOOR,
    )

    reqv = roh * (1.0 + delzoh**2 / (24.0 * roh**2))

    # Loop/test roles swapped -- see `_mutual_inductances`. The "loops" are the 2 x NOH
    # equivalent CS filaments; the test point is the single plasma filament.
    xc_in, xc_out = _mutual_inductances_over_loop_sets(
        jnp.stack([reqv - deltar, reqv + deltar]), zoh, rmajor, 0.0
    )
    xohpl = jnp.sum(jnp.where(active, 0.5 * (xc_in + xc_out), 0.0))

    ind_cs_plasma = xohpl / noh * n_pf_coil_turns[CS_INDEX]
    ind = ind.at[PLASMA_INDEX, CS_INDEX].set(ind_cs_plasma)
    ind = ind.at[CS_INDEX, PLASMA_INDEX].set(ind_cs_plasma)

    # --- 2. Plasma self ---------------------------------------------------------------
    ind = ind.at[PLASMA_INDEX, PLASMA_INDEX].set(ind_plasma)

    # --- 3a. PF coil / plasma ---------------------------------------------------------
    last_of_group = _last_coil_of_each_group()
    xpfpl = _mutual_inductances(
        jnp.stack([r_pf_coil_middle[c] for c in last_of_group]),
        jnp.stack([z_pf_coil_middle[c] for c in last_of_group]),
        rmajor,
        0.0,
    )
    plasma_column = (
        jnp.stack([
            xpfpl[group] for group in range(N_PF_GROUPS) for _ in _coils_of_group(group)
        ])
        * n_pf_coil_turns[:N_PF_COILS]
    )
    ind = ind.at[:N_PF_COILS, PLASMA_INDEX].set(plasma_column)
    ind = ind.at[PLASMA_INDEX, :N_PF_COILS].set(plasma_column)

    # --- 3b. Central Solenoid self, then Central Solenoid / PF coil -------------------
    ind = ind.at[CS_INDEX, CS_INDEX].set(
        calculate_solenoid_self_inductance(
            a=r_cs_middle,
            b=2.0 * z_cs_half,
            c=r_pf_coil_outer[CS_INDEX] - r_pf_coil_inner[CS_INDEX],
            n=n_pf_coil_turns[CS_INDEX],
        )
    )

    # The thirty CS filaments seen from each group's representative coil: one loop set,
    # four test points, so one batched call.
    targets = jnp.asarray(last_of_group)
    xohpf = jnp.sum(
        jnp.where(
            active[None, :],
            _mutual_inductances_over_test_points(
                roh, zoh, r_pf_coil_middle[targets], z_pf_coil_middle[targets]
            ),
            0.0,
        ),
        axis=1,
    )
    xohpf_of_coil = jnp.stack([
        xohpf[group] for group in range(N_PF_GROUPS) for _ in _coils_of_group(group)
    ])
    cs_column = (
        xohpf_of_coil * n_pf_coil_turns[:N_PF_COILS] * n_pf_coil_turns[CS_INDEX] / noh
    )
    ind = ind.at[:N_PF_COILS, CS_INDEX].set(cs_column)
    ind = ind.at[CS_INDEX, :N_PF_COILS].set(cs_column)

    # --- 4. PF coil / PF coil ---------------------------------------------------------
    # Every ordered pair at once. The self pairs are computed too and then overwritten
    # by the thin-ring diagonal below -- `calculate_b_field_at_point`'s `_S_MAX` clamp
    # keeps a coil against itself finite and finitely differentiable, and the `.set`
    # discards it either way, so it costs one column of arithmetic and saves five traces
    # of the kernel.
    r_coils = r_pf_coil_middle[:N_PF_COILS]
    z_coils = z_pf_coil_middle[:N_PF_COILS]
    turns = n_pf_coil_turns[:N_PF_COILS]
    xc = _mutual_inductances_over_test_points(r_coils, z_coils, r_coils, z_coils)

    # Diagonal: thin ring of equivalent circular cross-section.
    rl = jnp.abs(z_pf_coil_upper[:N_PF_COILS] - z_pf_coil_lower[:N_PF_COILS]) / jnp.sqrt(
        jnp.pi
    )
    block = (
        (xc * turns[None, :] * turns[:, None])
        .at[jnp.diag_indices(N_PF_COILS)]
        .set(
            RMU0
            * turns**2
            * r_coils
            * (jnp.log(8.0 * r_coils / rl) - _PF_SELF_INDUCTANCE_OFFSET)
        )
    )
    return ind.at[:N_PF_COILS, :N_PF_COILS].set(block)


def _coils_of_group(group):
    """The flattened coil indices belonging to one group, on the reference topology."""
    start = sum(N_COILS_IN_GROUP[:group])
    return range(start, start + N_COILS_IN_GROUP[group])


def _last_coil_of_each_group():
    """`ncoils - 1` per group -- the coil `induct` evaluates each group's field at."""
    return [sum(N_COILS_IN_GROUP[: group + 1]) - 1 for group in range(N_PF_GROUPS)]


def calculate_pf_plasma_inductances_no_central_solenoid(
    rmajor,
    ind_plasma,
    r_pf_coil_middle,
    z_pf_coil_middle,
    z_pf_coil_upper,
    z_pf_coil_lower,
    n_pf_coil_turns,
    *,
    topology=SPHERICAL_TOKAMAK_TOPOLOGY,
):
    """`calculate_pf_cs_plasma_inductances` on a machine with no central solenoid.

    Ports `PFCoil.induct`, `process/models/pfcoil.py:1721-1984`, at `iohcl = 0`.
    **Three of the four blocks survive and one does not**: `induct` guards the CS/plasma
    block (`:1812`), the CS self-inductance and the CS/PF block (`:1893`) on
    `iohcl != 0`, and sets `nef = n_cs_pf_coils` rather than `n_cs_pf_coils - 1`
    (`:1943-1947`) so that the PF/PF block covers every coil. What is left is the plasma
    self-inductance, PF/plasma and PF/PF.

    Six reads disappear with those blocks: `dr_cs` and `r_cs_middle` outright, and
    `r_pf_coil_inner`/`r_pf_coil_outer`, whose only use in `induct` is the CS's radial
    winding thickness (`:1896-1899`). **`noh` disappears too**, and that is worth its own
    sentence: it is the module docstring's "graph-assembly constant computed from solved
    geometry", and on this arm `roh`/`zoh` are never filled (`:1783-1791` is guarded) so
    no inductance depends on it at all. The piecewise-constant discontinuity in
    `dr_cs` that the reference occupant carries is simply not present here.

    Parameters
    ----------
    rmajor :
        Plasma major radius (m) -- the plasma filament's radius. `.physics.rmajor`.
    ind_plasma :
        Plasma self inductance (H). `.physics.ind_plasma`.
    r_pf_coil_middle, z_pf_coil_middle :
        Coil centres (m), `topology.n_cs_pf_coils` entries -- all PF coils.
    z_pf_coil_upper, z_pf_coil_lower :
        Coil upper/lower edges (m); their difference gives the equivalent circular
        cross-section of the diagonal's thin-ring self-inductance.
    n_pf_coil_turns :
        Turns in each coil.

    Returns
    -------
    :
        `.pf_coil.ind_pf_cs_plasma_mutual`, `(NGC2, NGC2)`.
    """
    ind = jnp.zeros((NGC2, NGC2))
    plasma = topology.plasma_index
    n_pf_coils = topology.n_pf_coils

    # --- Plasma self ------------------------------------------------------------------
    ind = ind.at[plasma, plasma].set(ind_plasma)

    # --- PF coil / plasma -------------------------------------------------------------
    last_of_group = [
        topology.last_coil_of_group(group) for group in range(topology.n_pf_coil_groups)
    ]
    xpfpl = _mutual_inductances(
        jnp.stack([r_pf_coil_middle[c] for c in last_of_group]),
        jnp.stack([z_pf_coil_middle[c] for c in last_of_group]),
        rmajor,
        0.0,
    )
    plasma_column = (
        jnp.stack([
            xpfpl[group]
            for group in range(topology.n_pf_coil_groups)
            for _ in topology.coils_of_group(group)
        ])
        * n_pf_coil_turns[:n_pf_coils]
    )
    ind = ind.at[:n_pf_coils, plasma].set(plasma_column)
    ind = ind.at[plasma, :n_pf_coils].set(plasma_column)

    # --- PF coil / PF coil ------------------------------------------------------------
    # Every ordered pair at once; see `_ind_pf_cs_plasma_mutual` for why the self pairs
    # are computed and then overwritten rather than excluded.
    r_coils = r_pf_coil_middle[:n_pf_coils]
    z_coils = z_pf_coil_middle[:n_pf_coils]
    turns = n_pf_coil_turns[:n_pf_coils]
    xc = _mutual_inductances_over_test_points(r_coils, z_coils, r_coils, z_coils)

    rl = jnp.abs(z_pf_coil_upper[:n_pf_coils] - z_pf_coil_lower[:n_pf_coils]) / jnp.sqrt(
        jnp.pi
    )
    return ind.at[:n_pf_coils, :n_pf_coils].set(
        (xc * turns[None, :] * turns[:, None])
        .at[jnp.diag_indices(n_pf_coils)]
        .set(
            RMU0
            * turns**2
            * r_coils
            * (jnp.log(8.0 * r_coils / rl) - _PF_SELF_INDUCTANCE_OFFSET)
        )
    )


def calculate_pf_cs_plasma_inductances_at_reference_width(
    rmajor,
    ind_plasma,
    dr_cs,
    r_cs_middle,
    r_pf_coil_middle,
    z_pf_coil_middle,
    r_pf_coil_inner,
    r_pf_coil_outer,
    z_pf_coil_upper,
    z_pf_coil_lower,
    n_pf_coil_turns,
):
    """`PFCoilInductance`: trims every coil array to `N_CS_PF_COILS` entries."""
    return calculate_pf_cs_plasma_inductances(
        rmajor=rmajor,
        ind_plasma=ind_plasma,
        dr_cs=dr_cs,
        r_cs_middle=r_cs_middle,
        r_pf_coil_middle=r_pf_coil_middle[:N_CS_PF_COILS],
        z_pf_coil_middle=z_pf_coil_middle[:N_CS_PF_COILS],
        r_pf_coil_inner=r_pf_coil_inner[:N_CS_PF_COILS],
        r_pf_coil_outer=r_pf_coil_outer[:N_CS_PF_COILS],
        z_pf_coil_upper=z_pf_coil_upper[:N_CS_PF_COILS],
        z_pf_coil_lower=z_pf_coil_lower[:N_CS_PF_COILS],
        n_pf_coil_turns=n_pf_coil_turns[:N_CS_PF_COILS],
    )


def calculate_pf_plasma_inductances_no_central_solenoid_for_topology(
    rmajor,
    ind_plasma,
    r_pf_coil_middle,
    z_pf_coil_middle,
    z_pf_coil_upper,
    z_pf_coil_lower,
    n_pf_coil_turns,
    *,
    topology,
):
    """`PFCoilInductanceNoCentralSolenoid`: trims every coil array to
    `topology.n_cs_pf_coils` entries.
    """
    n = topology.n_cs_pf_coils
    return calculate_pf_plasma_inductances_no_central_solenoid(
        rmajor=rmajor,
        ind_plasma=ind_plasma,
        r_pf_coil_middle=r_pf_coil_middle[:n],
        z_pf_coil_middle=z_pf_coil_middle[:n],
        z_pf_coil_upper=z_pf_coil_upper[:n],
        z_pf_coil_lower=z_pf_coil_lower[:n],
        n_pf_coil_turns=n_pf_coil_turns[:n],
        topology=topology,
    )
