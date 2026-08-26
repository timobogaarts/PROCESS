"""What current each PF coil group carries, at each of the three time points.

Audit record: `functional_process/_audit/units/models/pfcoil/currents.md`.

This is the half of `PFCoil.pfcoil()` between the coil placement and the coil sizing:

- `calculate_efc_currents` -- `PFCoil.efc` (`process/models/pfcoil.py:1403-1506`) with
  its three helpers `fixb` (`:5133-5183`), `mtrx` (`:5186-5285`) and `PFCoil.solv`
  (`:1567-1613`), plus the residual norm `rsid` (`:5063-5130`). A damped least-squares
  fit, by SVD, of the group currents that reproduce a wanted field at a set of points.
- `calculate_plasma_initiation_currents` -- `pfcoil()`'s "Flux swing coils" block
  (`:366-405`): the currents that null the field across the plasma midplane at
  breakdown, `.pf_coil.ccl0`.
- `calculate_equilibrium_currents` -- `pfcoil()`'s `i_pf_current = 1`, conventional
  aspect ratio arm (`:456-598`): the divertor-coil currents are fixed analytically and
  the outside-TF groups' currents are solved for against the required vertical field.
- `calculate_cs_flux_swing` -- `:600-661`, the CS current-density ratio that supplies
  whatever volt-seconds the PF set does not.
- `calculate_time_point_currents` -- `:663-728`, each coil's current at beginning of
  pulse, beginning of flat-top and end of flat-top.

**A three-node cycle lives here, and it is real.** `calculate_cs_flux_swing` reads
`.pf_coil.n_pf_coil_turns`, which `masses.py`'s `PFCoilSizes` owns; `PFCoilSizes` reads
`.pf_coil.c_pf_cs_coils_peak_ma`, which `fields.py`'s `PFCoilCurrentWaveform` owns; and
that reads `.pf_coil.c_pf_cs_coil_*_ma`, which `calculate_time_point_currents` owns from
`.pf_coil.f_j_cs_start_end_flat_top`. PROCESS closes the loop by *bootstrapping*: the
first visit to `pfcoil()` sets `ind_pf_cs_plasma_mutual` to all ones and
`n_pf_coil_turns` to a flat 100 (`:605-608`, `first_call`) and then relies on
`Caller.call_models` re-running the whole pipeline up to ten times until nothing moves.
That is a Gauss-Seidel iteration over an undeclared SCC, and it is exactly what this
port is meant to make visible. **Nothing here is a `FixedPointFunction`**: no node reads
a `VarPath` it owns, so this is a genuine multi-node SCC for `Blocking` to find and for
a `Drive` to solve, not a self-loop. See `currents.md` § "The cycle" and this wave's
report -- the assembler has to decide which algorithm drives it.

`.pf_coil.ind_pf_cs_plasma_mutual` **was** a boundary input here and is not one any
more: `inductance.py::PFCoilInductance` ports its producer, `PFCoil.induct`
(`:1721-1984`). The cycle above therefore has four nodes, not three, and the matrix is
an internal edge of the block rather than something supplied from outside -- which also
means PROCESS's `first_call` seeding of it is the iteration's initial guess. See
`inductance.md` § "The cycle, one node larger".
"""

import jax
import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.models.pfcoil import (
    CS_INDEX,
    LROW1,
    N_COILS_IN_GROUP,
    N_PF_GROUPS,
    N_PF_GROUPS_MAX,
    NGC2,
    NPTS,
    PLASMA_INDEX,
)
from functional_process.models.pfcoil.fields import calculate_b_field_at_point
from functional_process.models.pfcoil.geometry import place_cs_filaments
from functional_process.paths import build, pf_coil, physics

_SIGMA_FLOOR = 1.0e-10
"""`PFCoil.solv`'s `sigma[j] > 1.0e-10` guard (`pfcoil.py:1608`) -- singular values below
this do not contribute to the solution."""

FIXED_CURRENT_GROUPS = (0, 1)
"""The `i_pf_location = 2` divertor-coil groups, whose current PROCESS fixes
analytically and then hands to the equilibrium solve as fixed-current filaments
(`pfcoil.py:485-511`). Each holds one coil on this run, so `nfxf0 = 2`."""

EQUILIBRIUM_GROUPS = (2, 3)
"""The `i_pf_location = 3` groups, whose current the SVD solves for, in `pcls0`'s order
(`pfcoil.py:519-532`)."""


def _fixb(rpts, zpts, r_fix, z_fix, c_fix):
    """Field at each test point from the fixed-current loops.

    Ports `fixb`, `process/models/pfcoil.py:5133-5183`. PROCESS's `nfix <= 0` early
    return is not reproduced as a branch: both call sites on this arm pass a positive
    `nfix`, and a zero-length `r_fix` makes the sum below zero anyway.
    """
    npts = rpts.shape[0]
    br = []
    bz = []
    for i in range(npts):
        _, brw, bzw, _ = calculate_b_field_at_point(
            r_current_loop=r_fix,
            z_current_loop=z_fix,
            c_current_loop=c_fix,
            r_test_point=rpts[i],
            z_test_point=zpts[i],
        )
        br.append(brw)
        bz.append(bzw)
    return (
        jnp
        .zeros(LROW1)
        .at[:npts]
        .set(jnp.stack(br))
        .at[npts : 2 * npts]
        .set(jnp.stack(bz))
    )


def _mtrx(rpts, zpts, brin, bzin, r_group, z_group, n_in_group, alfa, bfix):
    """The damped least-squares matrix and right-hand side.

    Ports `mtrx`, `process/models/pfcoil.py:5186-5285`. `n_in_group` is a Python tuple:
    it selects how many coils of each group's row are summed, which is a shape, not a
    value. `gmat` keeps PROCESS's full `(LROW1, N_PF_GROUPS_MAX)` padding; `_solv` trims
    the unused columns before decomposing, for a reason given there.

    Returns
    -------
    tuple
        `(nrws, gmat, bvec)` -- the number of rows in use (a Python int), the matrix and
        the right-hand side.
    """
    npts = rpts.shape[0]
    n_groups = len(n_in_group)

    bvec = jnp.zeros(LROW1)
    gmat = jnp.zeros((LROW1, N_PF_GROUPS_MAX))

    for i in range(npts):
        bvec = bvec.at[i].set(brin[i] - bfix[i])
        bvec = bvec.at[i + npts].set(bzin[i] - bfix[i + npts])
        for j in range(n_groups):
            nc = n_in_group[j]
            _, br, bz, _ = calculate_b_field_at_point(
                r_current_loop=r_group[j, :nc],
                z_current_loop=z_group[j, :nc],
                c_current_loop=jnp.ones(nc),
                r_test_point=rpts[i],
                z_test_point=zpts[i],
            )
            gmat = gmat.at[i, j].set(br)
            gmat = gmat.at[i + npts, j].set(bz)

    # Smoothing constraint rows: one per group, on the diagonal.
    for j in range(n_groups):
        gmat = gmat.at[2 * npts + j, j].set(n_in_group[j] * alfa)

    return 2 * npts + n_groups, gmat, bvec


def _solv(gmat, bvec, n_groups):
    """Group currents from the matrix equation, by singular value decomposition.

    Ports `PFCoil.solv`, `process/models/pfcoil.py:1567-1613`. Two details of the
    original are load-bearing and are reproduced rather than tidied:

    - `zvec` is **carried** across the inner loop. It is reset to zero before each output
      component and only updated when `sigma[j] > 1e-10`, so a singular value below the
      floor makes that term reuse the *previous* `j`'s ratio instead of contributing
      zero. Reproduced by forward-filling the ratio along `j` (`jax.lax.cummax` over
      the index of the last admissible `j`), which is what that loop computes.
    - The sums run over `j < n_pf_coil_groups` only, so the null-space directions of the
      padded matrix never enter. Since the rank is `n_groups` and the singular values
      come back sorted, those are exactly the admissible directions.

    `scipy.linalg.svd`'s default full `U` is replaced by the thin one; only its first
    `min(rows, cols)` columns are ever indexed, and those two agree column for column.
    A per-column sign flip of the decomposition changes neither, because `U` and `V`
    flip together and this is a pseudo-inverse.

    **The decomposition is taken of `gmat[:, :n_groups]`, not of the padded matrix**, and
    that is a gradient fix rather than an optimisation. `mtrx` writes only the first
    `n_groups` of `gmat`'s `N_PF_GROUPS_MAX` columns, so the rest are structurally zero
    and contribute `N_PF_GROUPS_MAX - n_groups` *repeated zero* singular values. The
    values are unaffected -- for `A = [A_n | 0]`, `A`'s nonzero singular values, their
    left vectors and the leading block of their right vectors are exactly `A_n`'s, which
    is precisely what the sums below index -- but SVD's JVP divides by
    `sigma_i^2 - sigma_j^2`, so a repeated singular value makes the *whole* tangent
    `nan`. It does so only when the perturbation direction reaches that block, which is
    why the defect shows up at `alfa == 0` and nowhere else: the smoothing rows are the
    one thing whose tangent is nonzero while its value is not. Trimming the columns
    removes the degenerate block, and with it the `0/0`. Same numbers, finite
    derivative -- the `safe_math.py` bargain, in linear algebra rather than in `sqrt`.
    """
    umat, sigma, vmat = jnp.linalg.svd(gmat[:, :n_groups], full_matrices=False)

    work2 = umat.T @ bvec

    admissible = sigma > _SIGMA_FLOOR
    ratio = jnp.where(admissible, work2 / jnp.where(admissible, sigma, 1.0), 0.0)

    # Index of the last admissible `j` at or before each `j`; -1 where there is none.
    last = jax.lax.cummax(jnp.where(admissible, jnp.arange(n_groups), -1))
    zvec = jnp.where(last >= 0, ratio[jnp.maximum(last, 0)], 0.0)

    ccls = vmat.T @ zvec
    return jnp.zeros(N_PF_GROUPS_MAX).at[:n_groups].set(ccls)


def _rsid(brin, bzin, ccls, bfix, gmat, n_groups):
    """Sum of squares of the residual, normalised by the target field's own norm.

    Ports `rsid`, `process/models/pfcoil.py:5063-5130`. PROCESS's `if nfix > 0` guard
    around reading `bfix` is dropped: both call sites here have `nfix > 0`, and `bfix` is
    zero where they would not.
    """
    npts = brin.shape[0]
    g = gmat[:, :n_groups]
    svec_r = bfix[:npts] + g[:npts] @ ccls[:n_groups]
    svec_z = bfix[npts : 2 * npts] + g[npts : 2 * npts] @ ccls[:n_groups]

    brssq = jnp.sum((svec_r - brin) ** 2)
    brnrm = jnp.sum(brin**2)
    bzssq = jnp.sum((svec_z - bzin) ** 2)
    bznrm = jnp.sum(bzin**2)
    return brssq / (1.0 + brnrm) + bzssq / (1.0 + bznrm)


def calculate_efc_currents(
    rpts, zpts, brin, bzin, r_fix, z_fix, c_fix, r_group, z_group, alfa, n_in_group
):
    """Currents in a group of ring coils that best produce a wanted field.

    Ports `PFCoil.efc`, `process/models/pfcoil.py:1403-1506`, i.e. `fixb` then `mtrx`
    then `solv` then `rsid`, unchanged.

    Parameters
    ----------
    rpts, zpts :
        Coordinates of the points where the field is prescribed (m).
    brin, bzin :
        Wanted radial and vertical field at those points (T).
    r_fix, z_fix, c_fix :
        Coordinates (m) and currents (A) of the loops whose current is *not* being
        solved for.
    r_group, z_group :
        Coil centres by `(group, coil)`, `(n_groups, 2)`.
    alfa :
        Smoothing parameter. `.pf_coil.alfapf`.
    n_in_group :
        Python tuple of coils per group -- a shape, resolved at trace time.

    Returns
    -------
    tuple
        `(ssq, ccls)` -- the normalised sum of squared residuals, and the current in each
        group (A), padded to `N_PF_GROUPS_MAX`.
    """
    n_groups = len(n_in_group)
    bfix = _fixb(rpts, zpts, r_fix, z_fix, c_fix)
    _nrws, gmat, bvec = _mtrx(
        rpts, zpts, brin, bzin, r_group, z_group, n_in_group, alfa, bfix
    )
    ccls = _solv(gmat, bvec, n_groups)
    ssq = _rsid(brin, bzin, ccls, bfix, gmat, n_groups)
    return ssq, ccls


def calculate_plasma_initiation_currents(
    rmajor,
    rminor,
    r_pf_coil_middle_group_array,
    z_pf_coil_middle_group_array,
    r_cs_middle,
    dz_cs_full,
    a_cs_poloidal,
    j_cs_flat_top_end,
    f_j_cs_start_pulse_end_flat_top,
    alfapf,
):
    """Group currents that null the poloidal field across the plasma midplane.

    Ports `pfcoil()`'s "Flux swing coils" block, `process/models/pfcoil.py:366-405`, plus
    the CS filament placement it feeds on (`:206-234`) and the CS total current at end of
    flat-top (`:209-211`). PROCESS guards the whole block with
    `if j_cs_pulse_start != 0.0`; on this arm the CS is present and its current density
    is an iteration variable, never zero, so the guard is not reproduced -- recorded in
    `currents.md` as a dropped branch, not an oversight.

    The 32 test points span the plasma midplane from `rmajor - rminor` outward
    (`:377-384`), with zero wanted field at each.

    Returns
    -------
    tuple
        `(ssq0, ccl0)` -- `.pf_coil.ssq0` and `.pf_coil.ccl0`.
    """
    c_cs_flat_top_end = -(a_cs_poloidal * j_cs_flat_top_end)
    r_fix, z_fix, c_fix = place_cs_filaments(
        r_cs_middle=r_cs_middle,
        z_cs_inside_half=dz_cs_full / 2.0,
        c_cs_flat_top_end=c_cs_flat_top_end,
        f_j_cs_start_pulse_end_flat_top=f_j_cs_start_pulse_end_flat_top,
    )

    drpt = 2.0 * rminor / (NPTS - 1)
    rpt0 = rmajor - rminor
    rpts = rpt0 + jnp.arange(NPTS) * drpt
    zpts = jnp.zeros(NPTS)
    brin = jnp.zeros(NPTS)
    bzin = jnp.zeros(NPTS)

    return calculate_efc_currents(
        rpts=rpts,
        zpts=zpts,
        brin=brin,
        bzin=bzin,
        r_fix=r_fix,
        z_fix=z_fix,
        c_fix=c_fix,
        r_group=r_pf_coil_middle_group_array,
        z_group=z_pf_coil_middle_group_array,
        alfa=alfapf,
        n_in_group=N_COILS_IN_GROUP,
    )


def calculate_equilibrium_currents(
    rmajor,
    rminor,
    kappa,
    aspect,
    plasma_current,
    beta_poloidal_vol_avg,
    ind_plasma_internal_norm,
    r_pf_coil_middle_group_array,
    z_pf_coil_middle_group_array,
    alfapf,
):
    """Group currents at flat-top: divertor coils fixed, equilibrium coils solved for.

    Ports `pfcoil()`'s `i_pf_current = 1` conventional-aspect-ratio arm,
    `process/models/pfcoil.py:456-598`. Groups 0 and 1 are `i_pf_location = 2` divertor
    coils whose current is set analytically from the plasma's elongation and the coil's
    own height (`:489-499`) and then handed to the solve as *fixed* filaments
    (`:501-511`); groups 2 and 3 are the equilibrium coils, solved against the vertical
    field the plasma needs at `(rmajor, 0)` (`:556-595`).

    The `i_pf_location = 1` arm (a coil above the CS, current forced to zero) is not
    reachable on this run and is not ported.

    Returns
    -------
    tuple
        `(ccls, b_plasma_vertical_required)` -- the current in each group (A), padded to
        `N_PF_GROUPS_MAX`, and `.physics.b_plasma_vertical_required` (T).
    """
    # Divertor coils: fixed current, RK 07/12 (`pfcoil.py:489-499`).
    fixed = [
        plasma_current
        * 2.0
        * (1.0 - (kappa * rminor) / jnp.abs(z_pf_coil_middle_group_array[group, 0]))
        for group in FIXED_CURRENT_GROUPS
    ]
    c_fix = jnp.stack(fixed)
    r_fix = jnp.stack([r_pf_coil_middle_group_array[g, 0] for g in FIXED_CURRENT_GROUPS])
    z_fix = jnp.stack([z_pf_coil_middle_group_array[g, 0] for g in FIXED_CURRENT_GROUPS])

    # Vertical field required to hold the plasma in equilibrium (`:563-575`).
    b_plasma_vertical_required = (
        -1.0e-7
        * plasma_current
        / rmajor
        * (
            jnp.log(8.0 * aspect)
            + beta_poloidal_vol_avg
            + (ind_plasma_internal_norm / 2.0)
            - 1.5
        )
    )

    r_equilibrium = jnp.stack([
        r_pf_coil_middle_group_array[g] for g in EQUILIBRIUM_GROUPS
    ])
    z_equilibrium = jnp.stack([
        z_pf_coil_middle_group_array[g] for g in EQUILIBRIUM_GROUPS
    ])

    _ssq, solved = calculate_efc_currents(
        rpts=jnp.atleast_1d(rmajor),
        zpts=jnp.zeros(1),
        brin=jnp.zeros(1),
        bzin=jnp.atleast_1d(b_plasma_vertical_required),
        r_fix=r_fix,
        z_fix=z_fix,
        c_fix=c_fix,
        r_group=r_equilibrium,
        z_group=z_equilibrium,
        alfa=alfapf,
        n_in_group=tuple(N_COILS_IN_GROUP[g] for g in EQUILIBRIUM_GROUPS),
    )

    ccls = jnp.zeros(N_PF_GROUPS_MAX)
    for slot, group in enumerate(FIXED_CURRENT_GROUPS):
        ccls = ccls.at[group].set(fixed[slot])
    for slot, group in enumerate(EQUILIBRIUM_GROUPS):
        ccls = ccls.at[group].set(solved[slot])
    return ccls, b_plasma_vertical_required


def calculate_cs_flux_swing(
    ccls,
    ind_pf_cs_plasma_mutual_column,
    n_pf_coil_turns,
    vs_plasma_ramp_required,
    dr_cs_bore,
    dr_cs,
    dz_cs_full,
    a_cs_poloidal,
    j_cs_flat_top_end,
    f_j_cs_start_pulse_end_flat_top,
):
    """CS current-density ratio between beginning and end of flat-top.

    Ports `pfcoil()`'s flux-swing block, `process/models/pfcoil.py:600-657`, for
    `iohcl = 1`. The `iohcl = 0` arm (`:658-661`, ratio forced to 1 and an error logged)
    is a different occupant and is UNPORTED. The `|f_j_cs_start_end_flat_top| > 1`
    warning (`:652-657`) is pure reporting and is dropped.

    Parameters
    ----------
    ccls :
        Current in each PF coil group (A), `N_PF_GROUPS` entries.
    ind_pf_cs_plasma_mutual_column :
        `.pf_coil.ind_pf_cs_plasma_mutual[:6, 7]` -- the mutual inductance (H) between
        each PF coil and the plasma circuit. A **boundary input**: produced by
        `PFCoil.induct`, which this pass does not port.
    n_pf_coil_turns :
        Turns in each PF coil, six entries. Owned by `masses.py`'s `PFCoilSizes` -- this
        read is one edge of the SCC described in the module docstring.
    vs_plasma_ramp_required :
        Volt-seconds the plasma needs during the current ramp (Wb).
        `.physics.vs_plasma_ramp_required`.
    dr_cs_bore, dr_cs :
        CS bore radius and radial thickness (m). `.build.*`.
    dz_cs_full, a_cs_poloidal :
        Full CS height (m) and poloidal cross-section (m^2). `.pf_coil.*`.
    j_cs_flat_top_end :
        CS overall current density at end of flat-top (A/m^2).
    f_j_cs_start_pulse_end_flat_top :
        Ratio of CS current density at beginning of pulse to end of flat-top.

    Returns
    -------
    :
        `.pf_coil.f_j_cs_start_end_flat_top`.
    """
    per_group = jnp.stack([
        ccls[group] for group, n in enumerate(N_COILS_IN_GROUP) for _ in range(n)
    ])
    pfflux = jnp.sum(per_group * ind_pf_cs_plasma_mutual_column / n_pf_coil_turns)

    csflux = -vs_plasma_ramp_required - pfflux

    ddics = (
        4.0e-7
        * jnp.pi
        * jnp.pi
        * (
            (dr_cs_bore * dr_cs_bore)
            + (dr_cs * dr_cs) / 6.0
            + (dr_cs * dr_cs_bore) / 2.0
        )
        / dz_cs_full
    )
    dics = csflux / ddics

    c_cs_flat_top_end = -(a_cs_poloidal * j_cs_flat_top_end)
    return (
        (-c_cs_flat_top_end * f_j_cs_start_pulse_end_flat_top) + dics
    ) / c_cs_flat_top_end


def calculate_time_point_currents(
    ccl0,
    ccls,
    a_cs_poloidal,
    j_cs_flat_top_end,
    f_j_cs_start_pulse_end_flat_top,
    f_j_cs_start_end_flat_top,
):
    """Each coil's current at the three time points the rest of the model cares about.

    Ports `pfcoil()`'s `ncl` loop (`process/models/pfcoil.py:663-716`) and the CS's own
    three entries (`:718-728`), for `i_pf_current != 0` -- currents computed, not read
    from `ccl0_ma`/`ccls_ma`. The `i_pf_current = 0` arm is a different occupant and is
    UNPORTED.

    Returns
    -------
    tuple
        `(c_pf_cs_coil_pulse_start_ma, c_pf_cs_coil_flat_top_ma,
        c_pf_cs_coil_pulse_end_ma)`, seven entries each (MA) -- six PF coils then the CS.
    """
    group_of_coil = [group for group, n in enumerate(N_COILS_IN_GROUP) for _ in range(n)]
    start = [1.0e-6 * ccl0[g] for g in group_of_coil]
    flat = [
        1.0e-6
        * (
            ccls[g]
            - (ccl0[g] * f_j_cs_start_end_flat_top / f_j_cs_start_pulse_end_flat_top)
        )
        for g in group_of_coil
    ]
    end = [
        1.0e-6 * (ccls[g] - (ccl0[g] * (1.0 / f_j_cs_start_pulse_end_flat_top)))
        for g in group_of_coil
    ]

    c_cs_flat_top_end = -(a_cs_poloidal * j_cs_flat_top_end)
    start.append(-1.0e-6 * c_cs_flat_top_end * f_j_cs_start_pulse_end_flat_top)
    flat.append(1.0e-6 * c_cs_flat_top_end * f_j_cs_start_end_flat_top)
    end.append(1.0e-6 * c_cs_flat_top_end)

    return jnp.stack(start), jnp.stack(flat), jnp.stack(end)


class CSCurrentDensityPulseStart(ExplicitFunction):
    """cottax node: `.tokamak.cs_coil.current_density_pulse_start`.

    One line of `pfcoil()` (`process/models/pfcoil.py:161-164`), given its own node
    because three separate downstream nodes read it and none of them owns it.
    """

    j_cs_pulse_start = OutputInto(pf_coil)

    def __call__(
        self,
        j_cs_flat_top_end=From(pf_coil),
        f_j_cs_start_pulse_end_flat_top=From(pf_coil),
    ):
        return j_cs_flat_top_end * f_j_cs_start_pulse_end_flat_top


class PFCoilInitiationCurrents(ExplicitFunction):
    """cottax node: `.tokamak.pf_coil.initiation_currents`. Occupant for `iohcl = 1`.

    Owns `.pf_coil.ccl0` and `.pf_coil.ssq0`. The `iohcl = 0` arm (no CS, so no filaments
    and `c_cs_flat_top_end = 0`, `pfcoil.py:202-204`) is UNPORTED.

    The CS filament placement is computed inside rather than read: PROCESS stores the
    filament arrays in `.pf_coil.r/z/c_pf_cs_current_filaments`, but *overwrites* parts
    of them twice more within the same routine (see `fields.py`'s module docstring), so
    those three `VarPath`s have no single owner and are deliberately not claimed by any
    node in this package.
    """

    ssq0 = OutputInto(pf_coil)
    ccl0 = OutputInto(pf_coil)

    def __call__(
        self,
        rmajor=From(physics),
        rminor=From(physics),
        r_pf_coil_middle_group_array=From(pf_coil),
        z_pf_coil_middle_group_array=From(pf_coil),
        r_cs_middle=From(pf_coil),
        dz_cs_full=From(pf_coil),
        a_cs_poloidal=From(pf_coil),
        j_cs_flat_top_end=From(pf_coil),
        f_j_cs_start_pulse_end_flat_top=From(pf_coil),
        alfapf=From(pf_coil),
    ):
        return calculate_plasma_initiation_currents(
            rmajor=rmajor,
            rminor=rminor,
            r_pf_coil_middle_group_array=r_pf_coil_middle_group_array[:N_PF_GROUPS],
            z_pf_coil_middle_group_array=z_pf_coil_middle_group_array[:N_PF_GROUPS],
            r_cs_middle=r_cs_middle,
            dz_cs_full=dz_cs_full,
            a_cs_poloidal=a_cs_poloidal,
            j_cs_flat_top_end=j_cs_flat_top_end,
            f_j_cs_start_pulse_end_flat_top=f_j_cs_start_pulse_end_flat_top,
            alfapf=alfapf,
        )


class PFCoilEquilibriumCurrents(ExplicitFunction):
    """cottax node: `.tokamak.pf_coil.equilibrium_currents`.

    Occupant for `i_pf_current = 1` with `itart = 0` and
    `i_pf_location = (2, 2, 3, 3)`. Owns `.pf_coil.ccls` and
    `.physics.b_plasma_vertical_required` -- the latter is written by both arms of the
    `itart` branch (`pfcoil.py:444-454` and `:575`) from the same expression, so it
    belongs to whichever occupant is instantiated.

    UNPORTED arms: `i_pf_current = 0` (currents read from `ccls_ma` instead of solved
    for) and `itart = 1, itartpf = 0` (the spherical-tokamak scaling that bypasses the
    SVD entirely, `:411-454`).
    """

    ccls = OutputInto(pf_coil)
    b_plasma_vertical_required = OutputInto(physics)

    def __call__(
        self,
        rmajor=From(physics),
        rminor=From(physics),
        kappa=From(physics),
        aspect=From(physics),
        plasma_current=From(physics),
        beta_poloidal_vol_avg=From(physics),
        ind_plasma_internal_norm=From(physics),
        r_pf_coil_middle_group_array=From(pf_coil),
        z_pf_coil_middle_group_array=From(pf_coil),
        alfapf=From(pf_coil),
    ):
        return calculate_equilibrium_currents(
            rmajor=rmajor,
            rminor=rminor,
            kappa=kappa,
            aspect=aspect,
            plasma_current=plasma_current,
            beta_poloidal_vol_avg=beta_poloidal_vol_avg,
            ind_plasma_internal_norm=ind_plasma_internal_norm,
            r_pf_coil_middle_group_array=r_pf_coil_middle_group_array[:N_PF_GROUPS],
            z_pf_coil_middle_group_array=z_pf_coil_middle_group_array[:N_PF_GROUPS],
            alfapf=alfapf,
        )


class CSFluxSwing(ExplicitFunction):
    """cottax node: `.tokamak.cs_coil.flux_swing`. Occupant for `iohcl = 1`.

    **One edge of the SCC** described in this module's docstring: it reads
    `.pf_coil.n_pf_coil_turns`, which `masses.py`'s `PFCoilSizes` owns. Checked against
    the brief's Shape-A/Shape-B rule before declaring it -- this is not an apparent
    self-loop that dissolves on inspection. The producer really is a different node, the
    value really is the sizing pass's output, and PROCESS really does bootstrap it
    (`pfcoil.py:605-608`).
    """

    f_j_cs_start_end_flat_top = OutputInto(pf_coil)

    def __call__(
        self,
        ccls=From(pf_coil),
        ind_pf_cs_plasma_mutual=From(pf_coil),
        n_pf_coil_turns=From(pf_coil),
        vs_plasma_ramp_required=From(physics),
        dr_cs_bore=From(build),
        dr_cs=From(build),
        dz_cs_full=From(pf_coil),
        a_cs_poloidal=From(pf_coil),
        j_cs_flat_top_end=From(pf_coil),
        f_j_cs_start_pulse_end_flat_top=From(pf_coil),
    ):
        return calculate_cs_flux_swing(
            ccls=ccls[:N_PF_GROUPS],
            ind_pf_cs_plasma_mutual_column=ind_pf_cs_plasma_mutual[
                :CS_INDEX, PLASMA_INDEX
            ],
            n_pf_coil_turns=n_pf_coil_turns[:CS_INDEX],
            vs_plasma_ramp_required=vs_plasma_ramp_required,
            dr_cs_bore=dr_cs_bore,
            dr_cs=dr_cs,
            dz_cs_full=dz_cs_full,
            a_cs_poloidal=a_cs_poloidal,
            j_cs_flat_top_end=j_cs_flat_top_end,
            f_j_cs_start_pulse_end_flat_top=f_j_cs_start_pulse_end_flat_top,
        )


class PFCoilTimePointCurrents(ExplicitFunction):
    """cottax node: `.tokamak.pf_coil.time_point_currents`. Occupant for
    `i_pf_current != 0` and `iohcl = 1`.

    Owns the three `.pf_coil.c_pf_cs_coil_*_ma` arrays at full `NGC2` width, plus
    `.pf_coil.ccl0_ma`/`.pf_coil.ccls_ma`, which on this arm are a pure unit conversion
    of `ccl0`/`ccls` (`pfcoil.py:678-680`) rather than inputs.
    """

    c_pf_cs_coil_pulse_start_ma = OutputInto(pf_coil)
    c_pf_cs_coil_flat_top_ma = OutputInto(pf_coil)
    c_pf_cs_coil_pulse_end_ma = OutputInto(pf_coil)
    ccl0_ma = OutputInto(pf_coil)
    ccls_ma = OutputInto(pf_coil)

    def __call__(
        self,
        ccl0=From(pf_coil),
        ccls=From(pf_coil),
        a_cs_poloidal=From(pf_coil),
        j_cs_flat_top_end=From(pf_coil),
        f_j_cs_start_pulse_end_flat_top=From(pf_coil),
        f_j_cs_start_end_flat_top=From(pf_coil),
    ):
        start, flat, end = calculate_time_point_currents(
            ccl0=ccl0,
            ccls=ccls,
            a_cs_poloidal=a_cs_poloidal,
            j_cs_flat_top_end=j_cs_flat_top_end,
            f_j_cs_start_pulse_end_flat_top=f_j_cs_start_pulse_end_flat_top,
            f_j_cs_start_end_flat_top=f_j_cs_start_end_flat_top,
        )
        pad = jnp.zeros(NGC2)
        return (
            pad.at[: CS_INDEX + 1].set(start),
            pad.at[: CS_INDEX + 1].set(flat),
            pad.at[: CS_INDEX + 1].set(end),
            1.0e-6 * ccl0,
            1.0e-6 * ccls,
        )
