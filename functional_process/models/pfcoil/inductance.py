"""Mutual and self inductances of the PF coils, the CS and the plasma.

Audit record: `functional_process/_audit/units/models/pfcoil/inductance.md`.

One unit, `calculate_pf_cs_plasma_inductances`, porting `PFCoil.induct`
(`process/models/pfcoil.py:1721-1984`, everything before its `if not output: return`)
and `PFCoil.selfinductance` (`:2837-2867`). One cottax node, `PFCoilInductance`
(`.tokamak.pf_coil.inductance`), owning `.pf_coil.ind_pf_cs_plasma_mutual` whole.

**This closes the cycle rather than joining it.** Before this module,
`.pf_coil.ind_pf_cs_plasma_mutual` was a boundary input of
`currents.py::CSFluxSwing` -- nothing in the graph wrote it, because `induct` was
unported. It now has a producer, and that producer reads `.pf_coil.n_pf_coil_turns` and
the coil geometry, which `masses.py::PFCoilSizes` owns. So the SCC `currents.py`
describes grows from three nodes to four, and PROCESS's `first_call` bootstrap
(`pfcoil.py:605-608`, `ind_pf_cs_plasma_mutual[:, :] = 1.0`) is revealed for what it is:
**the cycle's initial guess**, not an external input. See § "The cycle, one node larger"
in the record.

**`noh` is a graph-assembly constant, and that is a finding, not a convenience.**
`induct` chooses how many pancake segments to split the CS into as

    noh = ceil(2 * z_pf_coil_upper[CS] / (r_pf_coil_outer[CS] - r_pf_coil_inner[CS]))

(`pfcoil.py:1758-1765`) -- an integer computed from *solved* geometry, on a run where
the CS radial thickness is an iteration variable. Every inductance this routine returns
depends on it, so `ind_pf_cs_plasma_mutual` is a **piecewise-constant-discontinuous**
function of `dr_cs`: PROCESS's answer steps whenever `dz_cs_full / dr_cs` crosses an
integer, and the derivative its own finite difference reports is the derivative of the
piece it happens to be sitting on. On `large_tokamak_eval.IN.DAT` the ratio is `29.027`,
so `noh = 30` and the nearest step is `0.9 %` away in `dr_cs`. This port fixes
`noh = 30`: a different `noh` is a different occupant, the same way a different
`i_pf_location` pattern is. Flagged in the record's open questions -- a structural
switch whose value moves with the solve is not something the conventions cover.
"""

import equinox as eqx
import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.models.pfcoil import (
    CS_INDEX,
    N_COILS_IN_GROUP,
    N_CS_PF_COILS,
    N_PF_COILS,
    N_PF_GROUPS,
    NGC2,
    PLASMA_INDEX,
    SPHERICAL_TOKAMAK_TOPOLOGY,
    PFCoilTopology,
)
from functional_process.models.pfcoil.fields import calculate_b_field_at_point
from functional_process.paths import build, pf_coil, physics
from functional_process.vocabulary import constants

RMU0 = constants.RMU0
"""Vacuum permeability (H/m), `process/core/constants.py:277`."""

NOH = 30
"""Number of pancake segments the CS is split into, on the reference arm. See the module
docstring: `ceil(2 * 7.936395 / 0.546817) = ceil(29.027) = 30`."""

NPLAS = 1
"""`nplas`, a literal `1` in `induct` (`pfcoil.py:1734`) -- the plasma is one filament,
at `(rmajor, 0)`. Every `for ii in range(nplas)` in the source is therefore a one-trip
loop, and every `/ nplas` a division by one; both are dropped here."""

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
    delzoh = 2.0 * z_cs_half / NOH
    zoh = z_cs_half - delzoh * (0.5 + jnp.arange(NOH))
    roh = jnp.full(NOH, r_cs_middle)

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
    xc_in = _mutual_inductances(reqv - deltar, zoh, rmajor, 0.0)
    xc_out = _mutual_inductances(reqv + deltar, zoh, rmajor, 0.0)
    xohpl = jnp.sum(0.5 * (xc_in + xc_out))

    ind_cs_plasma = xohpl / NOH * n_pf_coil_turns[CS_INDEX]
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
    for group in range(N_PF_GROUPS):
        for coil in _coils_of_group(group):
            value = xpfpl[group] * n_pf_coil_turns[coil]
            ind = ind.at[coil, PLASMA_INDEX].set(value)
            ind = ind.at[PLASMA_INDEX, coil].set(value)

    # --- 3b. Central Solenoid self, then Central Solenoid / PF coil -------------------
    ind = ind.at[CS_INDEX, CS_INDEX].set(
        calculate_solenoid_self_inductance(
            a=r_cs_middle,
            b=2.0 * z_cs_half,
            c=r_pf_coil_outer[CS_INDEX] - r_pf_coil_inner[CS_INDEX],
            n=n_pf_coil_turns[CS_INDEX],
        )
    )

    for group in range(N_PF_GROUPS):
        target = last_of_group[group]
        xohpf = jnp.sum(
            _mutual_inductances(
                roh, zoh, r_pf_coil_middle[target], z_pf_coil_middle[target]
            )
        )
        for coil in _coils_of_group(group):
            value = xohpf * n_pf_coil_turns[coil] * n_pf_coil_turns[CS_INDEX] / NOH
            ind = ind.at[coil, CS_INDEX].set(value)
            ind = ind.at[CS_INDEX, coil].set(value)

    # --- 4. PF coil / PF coil ---------------------------------------------------------
    for i in range(N_PF_COILS):
        others = [k for k in range(N_PF_COILS) if k != i]
        xc = _mutual_inductances(
            jnp.stack([r_pf_coil_middle[k] for k in others]),
            jnp.stack([z_pf_coil_middle[k] for k in others]),
            r_pf_coil_middle[i],
            z_pf_coil_middle[i],
        )
        for slot, k in enumerate(others):
            ind = ind.at[i, k].set(xc[slot] * n_pf_coil_turns[k] * n_pf_coil_turns[i])

        # Diagonal: thin ring of equivalent circular cross-section.
        rl = jnp.abs(z_pf_coil_upper[i] - z_pf_coil_lower[i]) / jnp.sqrt(jnp.pi)
        ind = ind.at[i, i].set(
            RMU0
            * n_pf_coil_turns[i] ** 2
            * r_pf_coil_middle[i]
            * (jnp.log(8.0 * r_pf_coil_middle[i] / rl) - _PF_SELF_INDUCTANCE_OFFSET)
        )

    return ind


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
    for group in range(topology.n_pf_coil_groups):
        for coil in topology.coils_of_group(group):
            value = xpfpl[group] * n_pf_coil_turns[coil]
            ind = ind.at[coil, plasma].set(value)
            ind = ind.at[plasma, coil].set(value)

    # --- PF coil / PF coil ------------------------------------------------------------
    for i in range(n_pf_coils):
        others = [k for k in range(n_pf_coils) if k != i]
        xc = _mutual_inductances(
            jnp.stack([r_pf_coil_middle[k] for k in others]),
            jnp.stack([z_pf_coil_middle[k] for k in others]),
            r_pf_coil_middle[i],
            z_pf_coil_middle[i],
        )
        for slot, k in enumerate(others):
            ind = ind.at[i, k].set(xc[slot] * n_pf_coil_turns[k] * n_pf_coil_turns[i])

        rl = jnp.abs(z_pf_coil_upper[i] - z_pf_coil_lower[i]) / jnp.sqrt(jnp.pi)
        ind = ind.at[i, i].set(
            RMU0
            * n_pf_coil_turns[i] ** 2
            * r_pf_coil_middle[i]
            * (jnp.log(8.0 * r_pf_coil_middle[i] / rl) - _PF_SELF_INDUCTANCE_OFFSET)
        )

    return ind


class PFCoilInductance(ExplicitFunction):
    """cottax node: `.tokamak.pf_coil.inductance`.

    Occupant for `iohcl = 1`, `n_pf_coils_in_group = (1, 1, 2, 2)` and `noh = 30`.

    **Owns `.pf_coil.ind_pf_cs_plasma_mutual` whole, not per index**, and the evidence
    for that is on both sides. Producer side: `induct` zeroes the entire matrix
    (`pfcoil.py:1750`) and then fills every entry of the eight-circuit block from one
    shared set of geometry reads -- there is no slice of it this node does not compute,
    unlike `b_pf_coil_peak`, whose index 6 comes from an unported routine and which is
    therefore owned per index. Consumer side: `currents.py::CSFluxSwing` reads column
    `[0:6, 7]`, `process/models/pulse.py:228,235` reads `[n_cs_pf_coils - 1, ...]`, and
    `process/models/power.py:320-539` reads the whole matrix; owning six entries would
    leave the rest of what this node computes unowned. `_audit/naming_convention.md`
    § "Array elements" asks for per-index addressing where read and write ranges
    *differ*, and here they do not.

    `.pf_coil.nef` (`:1944-1947`) is **not** owned: it is `n_cs_pf_coils - 1` on this
    arm, loop bookkeeping of the same kind as `n_cs_pf_coils` itself, which
    `__init__.py` records as graph-assembly data rather than a port.
    """

    ind_pf_cs_plasma_mutual = OutputInto(pf_coil)

    def __call__(
        self,
        rmajor=From(physics),
        ind_plasma=From(physics),
        dr_cs=From(build),
        r_cs_middle=From(pf_coil),
        r_pf_coil_middle=From(pf_coil),
        z_pf_coil_middle=From(pf_coil),
        r_pf_coil_inner=From(pf_coil),
        r_pf_coil_outer=From(pf_coil),
        z_pf_coil_upper=From(pf_coil),
        z_pf_coil_lower=From(pf_coil),
        n_pf_coil_turns=From(pf_coil),
    ):
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


class PFCoilInductanceNoCentralSolenoid(ExplicitFunction):
    """cottax node: `.tokamak.pf_coil.inductance`, the `iohcl = 0` occupant.

    Owns `.pf_coil.ind_pf_cs_plasma_mutual` whole, on the same producer-side argument as
    `PFCoilInductance`: `induct` zeroes the matrix (`pfcoil.py:1750`) and fills every
    entry of the circuit block that exists on this arm.

    Not a subclass of `PFCoilInductance`, because it declares **four reads fewer**
    (`dr_cs`, `r_cs_middle`, `r_pf_coil_inner`, `r_pf_coil_outer`) and a subclass may
    only widen a signature, not narrow it. See
    `calculate_pf_plasma_inductances_no_central_solenoid` for which blocks of `induct`
    those reads belong to and why the `noh` discontinuity is not present here.
    """

    topology: PFCoilTopology = eqx.field(static=True, default=SPHERICAL_TOKAMAK_TOPOLOGY)

    ind_pf_cs_plasma_mutual = OutputInto(pf_coil)

    def __call__(
        self,
        rmajor=From(physics),
        ind_plasma=From(physics),
        r_pf_coil_middle=From(pf_coil),
        z_pf_coil_middle=From(pf_coil),
        z_pf_coil_upper=From(pf_coil),
        z_pf_coil_lower=From(pf_coil),
        n_pf_coil_turns=From(pf_coil),
    ):
        n = self.topology.n_cs_pf_coils
        return calculate_pf_plasma_inductances_no_central_solenoid(
            rmajor=rmajor,
            ind_plasma=ind_plasma,
            r_pf_coil_middle=r_pf_coil_middle[:n],
            z_pf_coil_middle=z_pf_coil_middle[:n],
            z_pf_coil_upper=z_pf_coil_upper[:n],
            z_pf_coil_lower=z_pf_coil_lower[:n],
            n_pf_coil_turns=n_pf_coil_turns[:n],
            topology=self.topology,
        )
