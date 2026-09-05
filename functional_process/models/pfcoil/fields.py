"""Pure functions for the PF/CS Green's-function field kernel and per-group peak
fields, extracted from `functional_process/cottax/pfcoil/fields.py`.

That module still holds the graph declarations (`ExplicitFunction` occupants) that wire
these functions to `VarPath`s; read its module docstring (and
`functional_process/cottax/pfcoil/__init__.py`'s package docstring) for scope and the
reference-arm switch table. The audit record is
`functional_process/_audit/units/models/pfcoil/fields.md` and mirrors these functions,
not the declarations that call them.
"""

import jax
import jax.numpy as jnp

from functional_process.models.pfcoil import (
    CS_INDEX,
    N_CS_FILAMENTS,
    N_PF_COILS,
    N_PF_GROUPS,
    NFXF,
    NGC2,
    REFERENCE_TOPOLOGY,
    SPHERICAL_TOKAMAK_TOPOLOGY,
    PFLocation,
)
from functional_process.vocabulary import constants

RMU0 = constants.RMU0
"""Vacuum permeability (H/m), `process/core/constants.py:277`. Imported rather than
re-declared so a change there cannot silently diverge from the reference."""


_A = (1.38629436112, 0.09666344259, 0.03590092383, 0.03742563713, 0.01451196212)


_B = (0.5, 0.12498593597, 0.06880248576, 0.03328355346, 0.00441787012)


_C = (0.44325141463, 0.06260601220, 0.04757383546, 0.01736506451)


_D = (0.24998368310, 0.09200180037, 0.04069697526, 0.00526449639)


_S_MAX = 0.999999
"""`pfcoil.py:5002`'s kludge -- `s` is clamped below 1 so `a = log(1/(1-s))` stays
finite when the test point sits on a current loop."""


_DR_FLOOR = 1e-6
"""`pfcoil.py:5012-5014`'s kludge -- a test point at exactly a loop's radius would
divide by `dr**2 + zs = 0` when it is also at the loop's height."""


def calculate_b_field_at_point(
    r_current_loop,
    z_current_loop,
    c_current_loop,
    r_test_point,
    z_test_point,
):
    """Field and mutual inductance at one point from a set of circular current loops.

    Ports `calculate_b_field_at_point`, `process/models/pfcoil.py:4926-5060`, arithmetic
    unchanged. PROCESS's `for i in range(n_current_loops)` becomes one vectorised
    expression over the leading axis; the two in-loop kludges become `jnp.minimum` and
    `jnp.where` (see `_S_MAX`, `_DR_FLOOR`).

    Parameters
    ----------
    r_current_loop :
        R coordinate of each current loop (m).
    z_current_loop :
        Z coordinate of each current loop (m).
    c_current_loop :
        Current in each loop (A).
    r_test_point :
        R coordinate of the point the field is wanted at (m).
    z_test_point :
        Z coordinate of the point the field is wanted at (m).

    Returns
    -------
    tuple
        `(ind_mutual_array, b_test_point_radial, b_test_point_vertical,
        web_test_point_poloidal)` -- per-loop mutual inductance (H), radial and vertical
        field (T), poloidal flux (Wb). PROCESS's four-way return, unchanged.
    """
    d = (r_test_point + r_current_loop) ** 2 + (z_test_point - z_current_loop) ** 2
    s = jnp.minimum(4.0 * r_test_point * r_current_loop / d, _S_MAX)

    t = 1.0 - s
    a = jnp.log(1.0 / t)

    dz = z_test_point - z_current_loop
    zs = dz**2
    dr = r_test_point - r_current_loop
    sd = jnp.sqrt(d)

    dr = jnp.where(dr == 0.0, _DR_FLOOR, dr)  # noqa: RUF069 -- `pfcoil.py:5012`'s own

    # Elliptic integrals K and E.
    xk = (
        _A[0]
        + t * (_A[1] + t * (_A[2] + t * (_A[3] + _A[4] * t)))
        + a * (_B[0] + t * (_B[1] + t * (_B[2] + t * (_B[3] + _B[4] * t))))
    )
    xe = (
        1.0
        + t * (_C[0] + t * (_C[1] + t * (_C[2] + _C[3] * t)))
        + a * t * (_D[0] + t * (_D[1] + t * (_D[2] + _D[3] * t)))
    )

    ind_mutual_array = 0.5 * RMU0 * sd * ((2.0 - s) * xk - 2.0 * xe)

    brx = (
        RMU0
        * c_current_loop
        * dz
        / (2.0 * jnp.pi * r_test_point * sd)
        * (-xk + (r_current_loop**2 + r_test_point**2 + zs) / (dr**2 + zs) * xe)
    )
    bzx = (
        RMU0
        * c_current_loop
        / (2.0 * jnp.pi * sd)
        * (xk + (r_current_loop**2 - r_test_point**2 - zs) / (dr**2 + zs) * xe)
    )

    return (
        ind_mutual_array,
        jnp.sum(brx),
        jnp.sum(bzx),
        jnp.sum(ind_mutual_array * c_current_loop),
    )


_b_field_over_test_points = jax.vmap(
    calculate_b_field_at_point, in_axes=(None, None, None, 0, 0)
)
"""`calculate_b_field_at_point` batched over the *test point*, the loops held fixed.

**A batching axis, not a rewrite.** The kernel already takes the current loops as arrays
and reduces over them with `jnp.sum`; `vmap` adds a leading axis to the two scalar
test-point arguments and leaves every expression inside untouched, that reduction
included, so each batch element computes exactly what its own separate call computed.

What it buys is jaxpr size, which is what XLA's compile time scales superlinearly in.
Every caller ran this kernel once per test point *at trace time*, so its ~110 equations
were emitted 32 times in `currents._fixb`, 32 times per group in `currents._mtrx`, and
twice per group here -- 171 copies on `large_tokamak_nof`, roughly half that
configuration's entire program, against a stellarator that reaches none of them.
Batched, the kernel is emitted once per call site."""


_b_field_over_filament_sets = jax.vmap(
    _b_field_over_test_points, in_axes=(0, 0, 0, 0, 0)
)
"""`_b_field_over_test_points` batched again over *independent filament sets*.

Two axes: a set of current loops, and for each set its own test points.
`peak_b_field_at_pf_coil` has exactly that shape -- one filament set per PF group, each
evaluated at its own coil's inner and outer edge -- and unbatched it traced the kernel
once per group. The sets have to be padded to a common width for this, which
`_peak_fields_from_loops` does with zero-current filaments; see the comment there for why
that is exact."""


def calculate_coil_current_waveform(
    c_pf_cs_coil_pulse_start_ma,
    c_pf_cs_coil_flat_top_ma,
    c_pf_cs_coil_pulse_end_ma,
):
    """Peak current per coil and the current waveform normalised to it.

    Ports `PFCoil.waveform`, `process/models/pfcoil.py:2869-2940`, for the seven real
    coils (six PF plus the CS). The plasma row's all-ones waveform (`:2877-2879`) is not
    produced here -- it is a constant, and nothing in this file's closure reads it.

    PROCESS's peak selection is **three consecutive `if`s, not `if`/`elif`**, so a later
    one overrides an earlier one, and the third's first clause compares the end-of-flat-
    top current to itself (`:2911-2913`) and is therefore vacuously true. Resolving that
    exactly: the end-of-flat-top current wins whenever its magnitude is at least the
    beginning-of-flat-top one's; otherwise the larger of the beginning-of-flat-top and
    beginning-of-pulse currents wins, ties going to the former. That is what the two
    nested `jnp.where`s below say, and it leaves **no** unassigned case -- which matters,
    because PROCESS's fall-through would silently keep the previous iteration's value.

    Parameters
    ----------
    c_pf_cs_coil_pulse_start_ma :
        Coil currents at the beginning of the pulse (MA), one per coil.
    c_pf_cs_coil_flat_top_ma :
        Coil currents at the beginning of flat-top (MA).
    c_pf_cs_coil_pulse_end_ma :
        Coil currents at the end of flat-top (MA).

    Returns
    -------
    tuple
        `(c_pf_cs_coils_peak_ma, f_c_pf_cs_peak_time_array)` -- the peak current in each
        coil (MA, signed) and its six-point normalised waveform.
    """
    start = c_pf_cs_coil_pulse_start_ma
    flat = c_pf_cs_coil_flat_top_ma
    end = c_pf_cs_coil_pulse_end_ma

    flat_or_start = jnp.where(jnp.abs(flat) >= jnp.abs(start), flat, start)
    peak = jnp.where(jnp.abs(end) >= jnp.abs(flat), end, flat_or_start)

    zero = jnp.zeros_like(peak)
    waveform = jnp.stack(
        [zero, start / peak, flat / peak, flat / peak, end / peak, zero], axis=1
    )
    return peak, waveform


def _peak_time_column(
    c_pf_cs_coil_pulse_start_ma,
    c_pf_cs_coil_flat_top_ma,
    c_pf_cs_coil_pulse_end_ma,
    c_pf_cs_coils_peak_ma,
):
    """Which waveform column each coil's peak current came from (0-based).

    `peak_b_field_at_pf_coil` (`pfcoil.py:4459-4487`) recovers the time point by
    comparing each of the three currents to the peak within `1e-12` and raises if none
    matches. `calculate_coil_current_waveform` above makes the peak a bitwise copy of one
    of the three, so exactly one comparison is an exact equality and the raise is
    unreachable. PROCESS's `t_b_field_peak` of 2/4/5 is a 1-based column index into the
    six-point waveform; this returns the 0-based 1/3/4.
    """
    return jnp.where(
        jnp.abs(c_pf_cs_coil_pulse_start_ma - c_pf_cs_coils_peak_ma) < 1.0e-12,
        1,
        jnp.where(
            jnp.abs(c_pf_cs_coil_flat_top_ma - c_pf_cs_coils_peak_ma) < 1.0e-12, 3, 4
        ),
    )


def _cs_filament_positions(
    r_cs_middle,
    dz_cs_full,
    r_pf_coil_middle_group_array,
    z_pf_coil_middle_group_array,
    topology=REFERENCE_TOPOLOGY,
):
    """The `nfxf` "CS" filament positions as `peak_b_field_at_pf_coil` finds them.

    `CSCoil.place_cs_filaments` (`pfcoil.py:3151-3226`) puts filament `k` of the upper
    half at `z = (dz_cs_full/2) / n * (k + 0.5)` and mirrors it into the lower half, all
    at `r = r_cs_middle`. `pfcoil()`'s equilibrium branch then overwrites the leading
    entries with the *fixed-current* groups' PF coil positions (`:474-479`, `nocoil`
    counting one per coil of every `i_pf_location = 2` group) and never restores them.
    See this module's docstring; this helper reproduces the state that is actually read.
    """
    z_cs_inside_half = dz_cs_full / 2.0
    upper = z_cs_inside_half / N_CS_FILAMENTS * (jnp.arange(N_CS_FILAMENTS) + 0.5)
    z_filaments = jnp.concatenate([upper, -upper])
    r_filaments = jnp.full(NFXF, r_cs_middle)

    clobbered = [
        (group, coil)
        for group in topology.groups_at(PFLocation.ABOVE_TF)
        for coil in range(topology.n_pf_coils_in_group[group])
    ]
    for slot, (group, coil) in enumerate(clobbered):
        r_filaments = r_filaments.at[slot].set(r_pf_coil_middle_group_array[group, coil])
        z_filaments = z_filaments.at[slot].set(z_pf_coil_middle_group_array[group, coil])
    return r_filaments, z_filaments


def _peak_fields_from_loops(
    c_peak,
    waveform,
    time_column,
    r_pf_coil_middle,
    z_pf_coil_middle,
    r_pf_coil_inner,
    r_pf_coil_outer,
    z_pf_coil_upper,
    z_pf_coil_lower,
    rmajor,
    plasma_current,
    cs_filaments,
    topology,
):
    """One `peak_b_field_at_pf_coil` call per group, at that group's first coil.

    The body `calculate_pf_coil_peak_fields` and its `iohcl = 0` sibling share; both
    docstrings carry the argument.

    `cs_filaments` is `(r, z, current_scale)` when the machine has a central solenoid
    and `None` when it does not -- `peak_b_field_at_pf_coil` sets `kk = 0` outright at
    `pfcoil.py:4487-4489` in that case, so there is no filament to omit the current of.
    """
    r_per_group = []
    z_per_group = []
    c_per_group = []
    r_test_per_group = []
    z_test_per_group = []
    for group in range(topology.n_pf_coil_groups):
        target = topology.first_coil_of_group(group)

        column = time_column[target]
        f_at_time = jnp.take(waveform, column, axis=1)

        r_parts = []
        z_parts = []
        c_parts = []
        if cs_filaments is not None:
            r_cs, z_cs, current_scale = cs_filaments
            r_parts.append(r_cs)
            z_parts.append(z_cs)
            c_parts.append(
                jnp.full(r_cs.shape, f_at_time[topology.cs_index] * current_scale)
            )

        for coil in range(topology.n_pf_coils):
            current = c_peak[coil] * f_at_time[coil]
            if topology.group_of_coil(coil) == group:
                # Self field, Lyle's method: four filaments at +-1/8 and +-3/8 of the
                # coil height, each carrying a quarter of the current (`:4520-4573`).
                dzpf = z_pf_coil_upper[coil] - z_pf_coil_lower[coil]
                offsets = (0.125, 0.375, -0.125, -0.375)
                r_parts.append(jnp.full(4, r_pf_coil_middle[coil]))
                z_parts.append(
                    jnp.stack([z_pf_coil_middle[coil] + dzpf * o for o in offsets])
                )
                c_parts.append(jnp.full(4, current * 0.25e6))
            else:
                # Field from a different coil: one filament at its centre (`:4575-4588`).
                r_parts.append(r_pf_coil_middle[coil][None])
                z_parts.append(z_pf_coil_middle[coil][None])
                c_parts.append((current * 1.0e6)[None])

        # Plasma filament -- see the docstring: present with a zero current when
        # PROCESS's `t_b_field_peak > 2` does not hold.
        r_parts.append(jnp.asarray(rmajor)[None])
        z_parts.append(jnp.zeros(1))
        c_parts.append(jnp.where(column >= 2, plasma_current, 0.0)[None])

        r_per_group.append(jnp.concatenate(r_parts))
        z_per_group.append(jnp.concatenate(z_parts))
        c_per_group.append(jnp.concatenate(c_parts))
        r_test_per_group.append(
            jnp.stack([r_pf_coil_inner[target], r_pf_coil_outer[target]])
        )
        z_test_per_group.append(jnp.full(2, z_pf_coil_middle[target]))

    # One call for the whole calculation: `n_pf_coil_groups` filament sets crossed with
    # the two test points (inner and outer edge) each is evaluated at. A group whose
    # target coil expands into Lyle's four self-field filaments carries three loops more
    # than one whose does not, so the sets are padded to a common width with filaments at
    # `r = 1 m` carrying **no current** -- `brx`/`bzx` are linear in the current, so a
    # zero-current loop contributes exactly zero, and a unit radius keeps every
    # denominator in the kernel away from zero. Same trick, and the same justification,
    # as the plasma filament's masked current above.
    width = max(r.shape[0] for r in r_per_group)

    def _padded(parts, fill):
        return jnp.stack([
            jnp.concatenate([part, jnp.full(width - part.shape[0], fill)])
            for part in parts
        ])

    _, br, bz, _ = _b_field_over_filament_sets(
        _padded(r_per_group, 1.0),
        _padded(z_per_group, 0.0),
        _padded(c_per_group, 0.0),
        jnp.stack(r_test_per_group),
        jnp.stack(z_test_per_group),
    )
    b_edge = jnp.sqrt(br**2 + bz**2)

    coil_of_row = [
        group
        for group in range(topology.n_pf_coil_groups)
        for _ in range(topology.n_pf_coils_in_group[group])
    ]
    return (
        jnp.stack([b_edge[group, 0] for group in coil_of_row]),
        jnp.stack([b_edge[group, 1] for group in coil_of_row]),
    )


def calculate_pf_coil_peak_fields(
    c_pf_cs_coil_pulse_start_ma,
    c_pf_cs_coil_flat_top_ma,
    c_pf_cs_coil_pulse_end_ma,
    r_pf_coil_middle,
    z_pf_coil_middle,
    r_pf_coil_inner,
    r_pf_coil_outer,
    z_pf_coil_upper,
    z_pf_coil_lower,
    r_pf_coil_middle_group_array,
    z_pf_coil_middle_group_array,
    r_cs_middle,
    dz_cs_full,
    a_cs_poloidal,
    j_cs_pulse_start,
    j_cs_flat_top_end,
    rmajor,
    plasma_current,
    *,
    topology=REFERENCE_TOPOLOGY,
):
    """Peak field at the inner and outer edge of each of the six PF coils.

    Ports `peak_b_field_at_pf_coil` (`process/models/pfcoil.py:4414-4638`) for the four
    PF groups, plus `PFCoil.waveform` folded in (module docstring). Specialised to the
    reference run: `iohcl = 1`, `n_pf_coils_in_group = (1, 1, 2, 2)`, so the target coil
    is never the CS and the `kk = 0` arm at `:4452-4456` is not taken.

    PROCESS calls the routine once per group, for the group's *first* coil, and copies
    the answer to every coil in that group (`:4629-4631`); the four calls are independent
    given the coil geometry, so the Python loop over groups below is a trace-time
    unrolling, not a sequential dependence.

    **The plasma filament is always present, with a zero current when PROCESS omits
    it.** PROCESS appends a loop at `(rmajor, 0, plasma_current)` only when the peak time
    point is after the current ramp (`:4591`, `t_b_field_peak > 2`). That would make the
    traced filament *count* depend on data. A loop carrying zero current contributes
    exactly zero to both field components (`brx`/`bzx` are linear in the current), so it
    is included unconditionally with the current masked to zero instead -- identical
    values, fixed shapes.

    Parameters
    ----------
    c_pf_cs_coil_pulse_start_ma, c_pf_cs_coil_flat_top_ma, c_pf_cs_coil_pulse_end_ma :
        Coil currents (MA) at the three time points, seven entries -- six PF coils then
        the CS. `.pf_coil.c_pf_cs_coil_{pulse_start,flat_top,pulse_end}_ma[:7]`.
    r_pf_coil_middle, z_pf_coil_middle :
        Coil centre coordinates (m), seven entries.
    r_pf_coil_inner, r_pf_coil_outer :
        Coil inner/outer radii (m), seven entries -- the field is evaluated at these.
    z_pf_coil_upper, z_pf_coil_lower :
        Coil upper/lower edge heights (m), seven entries; their difference is the coil
        height Lyle's four-filament self-field model needs.
    r_pf_coil_middle_group_array, z_pf_coil_middle_group_array :
        Coil centres by `(group, coil)`, shape `(4, 2)`. Read for the two clobbered CS
        filament slots only.
    r_cs_middle :
        CS mean radius (m), where the remaining twelve CS filaments sit.
    dz_cs_full :
        Full CS height (m).
    a_cs_poloidal :
        CS poloidal cross-section (m^2).
    j_cs_pulse_start, j_cs_flat_top_end :
        CS overall current density at beginning of pulse / end of flat-top (A/m^2).
        Only their *order* is used, to pick the CS filament current's sign (`:4493`).
    rmajor :
        Plasma major radius (m) -- the plasma filament's radius.
    plasma_current :
        Plasma current (A).

    Returns
    -------
    tuple
        `(b_pf_coil_peak, bpf2)`, each six entries -- the field magnitude at the inner
        and at the outer edge of each PF coil (T).
    """
    c_peak, waveform = calculate_coil_current_waveform(
        c_pf_cs_coil_pulse_start_ma,
        c_pf_cs_coil_flat_top_ma,
        c_pf_cs_coil_pulse_end_ma,
    )
    time_column = _peak_time_column(
        c_pf_cs_coil_pulse_start_ma,
        c_pf_cs_coil_flat_top_ma,
        c_pf_cs_coil_pulse_end_ma,
        c_peak,
    )

    r_cs_filaments, z_cs_filaments = _cs_filament_positions(
        r_cs_middle,
        dz_cs_full,
        r_pf_coil_middle_group_array,
        z_pf_coil_middle_group_array,
        topology,
    )

    # Sign of the CS filament current: positive when the CS runs harder at the beginning
    # of the pulse than at the end of flat-top (`pfcoil.py:4493-4497`).
    sgn = jnp.where(j_cs_pulse_start > j_cs_flat_top_end, 1.0, -1.0)

    return _peak_fields_from_loops(
        c_peak=c_peak,
        waveform=waveform,
        time_column=time_column,
        r_pf_coil_middle=r_pf_coil_middle,
        z_pf_coil_middle=z_pf_coil_middle,
        r_pf_coil_inner=r_pf_coil_inner,
        r_pf_coil_outer=r_pf_coil_outer,
        z_pf_coil_upper=z_pf_coil_upper,
        z_pf_coil_lower=z_pf_coil_lower,
        rmajor=rmajor,
        plasma_current=plasma_current,
        cs_filaments=(
            r_cs_filaments,
            z_cs_filaments,
            j_cs_flat_top_end * sgn * a_cs_poloidal / topology.nfxf,
        ),
        topology=topology,
    )


def calculate_pf_coil_peak_fields_no_central_solenoid(
    c_pf_cs_coil_pulse_start_ma,
    c_pf_cs_coil_flat_top_ma,
    c_pf_cs_coil_pulse_end_ma,
    r_pf_coil_middle,
    z_pf_coil_middle,
    r_pf_coil_inner,
    r_pf_coil_outer,
    z_pf_coil_upper,
    z_pf_coil_lower,
    rmajor,
    plasma_current,
    *,
    topology=SPHERICAL_TOKAMAK_TOPOLOGY,
):
    """`calculate_pf_coil_peak_fields` on a machine with no central solenoid.

    Ports the same routine, `peak_b_field_at_pf_coil`
    (`process/models/pfcoil.py:4414-4638`), at `iohcl = 0`. One difference and it is
    an early one: `:4487-4489` sets `kk = 0` -- **no CS filaments contribute at all**,
    so the fourteen loops, the `sgn` comparison and the five CS fields that feed them
    (`r_cs_middle`, `dz_cs_full`, `a_cs_poloidal`, `j_cs_pulse_start`,
    `j_cs_flat_top_end`) are not read. Nor are the two group arrays, whose only use in
    the conventional arm is to reproduce the filament-clobbering defect this module's
    docstring records -- with no filaments there is nothing to clobber.

    The time-point re-derivation, Lyle's four-filament self-field expansion and the
    plasma loop are unchanged; see `calculate_pf_coil_peak_fields` for all three.

    Returns
    -------
    tuple
        `(b_pf_coil_peak, bpf2)`, `topology.n_pf_coils` entries each -- the field
        magnitude at the inner and at the outer edge of each PF coil (T).
    """
    c_peak, waveform = calculate_coil_current_waveform(
        c_pf_cs_coil_pulse_start_ma,
        c_pf_cs_coil_flat_top_ma,
        c_pf_cs_coil_pulse_end_ma,
    )
    time_column = _peak_time_column(
        c_pf_cs_coil_pulse_start_ma,
        c_pf_cs_coil_flat_top_ma,
        c_pf_cs_coil_pulse_end_ma,
        c_peak,
    )
    return _peak_fields_from_loops(
        c_peak=c_peak,
        waveform=waveform,
        time_column=time_column,
        r_pf_coil_middle=r_pf_coil_middle,
        z_pf_coil_middle=z_pf_coil_middle,
        r_pf_coil_inner=r_pf_coil_inner,
        r_pf_coil_outer=r_pf_coil_outer,
        z_pf_coil_upper=z_pf_coil_upper,
        z_pf_coil_lower=z_pf_coil_lower,
        rmajor=rmajor,
        plasma_current=plasma_current,
        cs_filaments=None,
        topology=topology,
    )


# ---------------------------------------------------------------------------
# Wrapped occupant bodies, `_audit/formulas_split.md` step 1: each of these used to
# slice the coil arrays to a fixed width (and, for `PFCoilCurrentWaveform`, pad and
# fill the plasma row) inline in a declaration's `__call__` before delegating. The
# `calculate_*` signatures above are untouched.
# ---------------------------------------------------------------------------


def calculate_pf_coil_peak_fields_reference_arm(
    c_pf_cs_coil_pulse_start_ma,
    c_pf_cs_coil_flat_top_ma,
    c_pf_cs_coil_pulse_end_ma,
    r_pf_coil_middle,
    z_pf_coil_middle,
    r_pf_coil_inner,
    r_pf_coil_outer,
    z_pf_coil_upper,
    z_pf_coil_lower,
    r_pf_coil_middle_group_array,
    z_pf_coil_middle_group_array,
    r_cs_middle,
    dz_cs_full,
    a_cs_poloidal,
    j_cs_pulse_start,
    j_cs_flat_top_end,
    rmajor,
    plasma_current,
):
    """`PFCoilPeakField`: trims each coil array to the seven-entry (six PF + CS) width
    and flattens the inner/outer 6-vectors to twelve per-index outputs.
    """
    b_inner, b_outer = calculate_pf_coil_peak_fields(
        topology=REFERENCE_TOPOLOGY,
        c_pf_cs_coil_pulse_start_ma=c_pf_cs_coil_pulse_start_ma[: CS_INDEX + 1],
        c_pf_cs_coil_flat_top_ma=c_pf_cs_coil_flat_top_ma[: CS_INDEX + 1],
        c_pf_cs_coil_pulse_end_ma=c_pf_cs_coil_pulse_end_ma[: CS_INDEX + 1],
        r_pf_coil_middle=r_pf_coil_middle[: CS_INDEX + 1],
        z_pf_coil_middle=z_pf_coil_middle[: CS_INDEX + 1],
        r_pf_coil_inner=r_pf_coil_inner[: CS_INDEX + 1],
        r_pf_coil_outer=r_pf_coil_outer[: CS_INDEX + 1],
        z_pf_coil_upper=z_pf_coil_upper[: CS_INDEX + 1],
        z_pf_coil_lower=z_pf_coil_lower[: CS_INDEX + 1],
        r_pf_coil_middle_group_array=r_pf_coil_middle_group_array[:N_PF_GROUPS, :],
        z_pf_coil_middle_group_array=z_pf_coil_middle_group_array[:N_PF_GROUPS, :],
        r_cs_middle=r_cs_middle,
        dz_cs_full=dz_cs_full,
        a_cs_poloidal=a_cs_poloidal,
        j_cs_pulse_start=j_cs_pulse_start,
        j_cs_flat_top_end=j_cs_flat_top_end,
        rmajor=rmajor,
        plasma_current=plasma_current,
    )
    return (*b_inner, *b_outer)


def calculate_pf_coil_peak_fields_no_central_solenoid_for_topology(
    c_pf_cs_coil_pulse_start_ma,
    c_pf_cs_coil_flat_top_ma,
    c_pf_cs_coil_pulse_end_ma,
    r_pf_coil_middle,
    z_pf_coil_middle,
    r_pf_coil_inner,
    r_pf_coil_outer,
    z_pf_coil_upper,
    z_pf_coil_lower,
    rmajor,
    plasma_current,
    *,
    topology,
):
    """`PFCoilPeakFieldNoCentralSolenoid`: trims each coil array to `topology`'s width
    and pads the two field vectors back out to `NGC2`.
    """
    n = topology.n_pf_coils
    b_inner, b_outer = calculate_pf_coil_peak_fields_no_central_solenoid(
        c_pf_cs_coil_pulse_start_ma=c_pf_cs_coil_pulse_start_ma[:n],
        c_pf_cs_coil_flat_top_ma=c_pf_cs_coil_flat_top_ma[:n],
        c_pf_cs_coil_pulse_end_ma=c_pf_cs_coil_pulse_end_ma[:n],
        r_pf_coil_middle=r_pf_coil_middle[:n],
        z_pf_coil_middle=z_pf_coil_middle[:n],
        r_pf_coil_inner=r_pf_coil_inner[:n],
        r_pf_coil_outer=r_pf_coil_outer[:n],
        z_pf_coil_upper=z_pf_coil_upper[:n],
        z_pf_coil_lower=z_pf_coil_lower[:n],
        rmajor=rmajor,
        plasma_current=plasma_current,
        topology=topology,
    )
    pad = jnp.zeros(NGC2)
    return pad.at[:n].set(b_inner), pad.at[:n].set(b_outer)


def calculate_coil_current_waveform_for_topology(
    c_pf_cs_coil_pulse_start_ma,
    c_pf_cs_coil_flat_top_ma,
    c_pf_cs_coil_pulse_end_ma,
    *,
    topology,
):
    """`PFCoilCurrentWaveform`: trims the three current arrays to `topology`'s width,
    pads the peak/waveform outputs back out to `NGC2`, and fills the plasma row with
    ones.
    """
    coils = topology.n_cs_pf_coils
    plasma = topology.plasma_index
    peak, waveform = calculate_coil_current_waveform(
        c_pf_cs_coil_pulse_start_ma[:coils],
        c_pf_cs_coil_flat_top_ma[:coils],
        c_pf_cs_coil_pulse_end_ma[:coils],
    )
    peak_full = jnp.zeros(NGC2).at[:coils].set(peak)
    waveform_full = (
        jnp.zeros((NGC2, 6)).at[:coils, :].set(waveform).at[plasma, :].set(1.0)
    )
    return peak_full, waveform_full


_T_B_FIELD_PEAK_FLAT_TOP_END = 5
"""`ohcalc`'s `timepoint = 5` for the end-of-flat-top field (`pfcoil.py:3345`), a
1-based column index into the six-point waveform."""


_T_B_FIELD_PEAK_PULSE_START = 2
"""`ohcalc`'s `timepoint = 2` for the beginning-of-pulse field (`pfcoil.py:3383`)."""


def calculate_cs_bore_magnetic_field(j_cs, r_cs_inner, r_cs_outer, dz_cs_half):
    """Field at the centre of a rectangular-section solenoid's bore (T).

    Ports `CSCoil.calculate_cs_bore_magnetic_field`,
    `process/models/pfcoil.py:3707-3748`, unchanged (M. N. Wilson,
    *Superconducting Magnets*).

    Parameters
    ----------
    j_cs :
        Overall current density (A/m^2).
    r_cs_inner, r_cs_outer :
        Solenoid inner and outer radii (m).
    dz_cs_half :
        Solenoid half-height (m).

    Returns
    -------
    :
        Field at the bore centre (T).
    """
    beta = dz_cs_half / r_cs_inner
    alpha = r_cs_outer / r_cs_inner
    return (
        j_cs
        * RMU0
        * dz_cs_half
        * jnp.log(
            (alpha + jnp.sqrt(alpha**2 + beta**2)) / (1.0 + jnp.sqrt(1.0 + beta**2))
        )
    )


def calculate_cs_self_peak_magnetic_field(j_cs, r_cs_inner, r_cs_outer, dz_cs_half):
    """Peak self-field of a rectangular-section solenoid (T).

    Ports `CSCoil.calculate_cs_self_peak_magnetic_field`, `pfcoil.py:3750-3822` --
    Wilson's five-branch fit to the ratio of peak field to bore-centre field, keyed on
    `beta = dz_cs_half / r_cs_inner`.

    **A five-way `if`/`elif` chain on traced data, so a nested `jnp.where`.** The five
    arms are genuinely different expressions of the same two shape ratios and PROCESS
    selects among them by value, not by a switch: this is a formula branch, not an
    occupant question. The chain is transcribed in PROCESS's own order and the `else`
    arm is the innermost default, so a `beta` exactly on a boundary lands where PROCESS
    puts it (the comparisons are strict `>` throughout).

    Every arm's expression is finite for every `beta > 0`, so no untaken branch can leak
    a NaN into the gradient -- the `where`-guard `exhaust.calculate_radiation_fraction`
    needs is not needed here, and that is a property of the fit rather than an oversight.

    Parameters
    ----------
    j_cs :
        Overall current density (A/m^2).
    r_cs_inner, r_cs_outer :
        Solenoid inner and outer radii (m).
    dz_cs_half :
        Solenoid half-height (m).

    Returns
    -------
    :
        Peak field of the solenoid (T).
    """
    beta = dz_cs_half / r_cs_inner
    alpha = r_cs_outer / r_cs_inner

    b_cs_bore_centre = calculate_cs_bore_magnetic_field(
        j_cs, r_cs_inner, r_cs_outer, dz_cs_half
    )

    # `beta > 3`: a blend between the fit and the infinite-solenoid field `b1`.
    b1 = RMU0 * j_cs * (r_cs_outer - r_cs_inner)
    f = (3.0 / beta) ** 2
    tall = f * b_cs_bore_centre * (1.007 + (alpha - 1.0) * 0.0055) + (1.0 - f) * b1

    rat_2_3 = (1.025 - (beta - 2.0) * 0.018) + (alpha - 1.0) * (
        0.01 - (beta - 2.0) * 0.0045
    )
    rat_1_2 = (1.117 - (beta - 1.0) * 0.092) + (alpha - 1.0) * (beta - 1.0) * 0.01
    rat_075_1 = (1.30 - 0.732 * (beta - 0.75)) + (alpha - 1.0) * (
        0.2 * (beta - 0.75) - 0.05
    )
    rat_short = (1.65 - 1.4 * (beta - 0.5)) + (alpha - 1.0) * (0.6 * (beta - 0.5) - 0.20)

    return jnp.where(
        beta > 3.0,
        tall,
        b_cs_bore_centre
        * jnp.where(
            beta > 2.0,
            rat_2_3,
            jnp.where(
                beta > 1.0,
                rat_1_2,
                jnp.where(beta > 0.75, rat_075_1, rat_short),
            ),
        ),
    )


def _cs_external_field(
    c_peak,
    waveform,
    column,
    r_pf_coil_middle,
    z_pf_coil_middle,
    r_cs_inner,
    r_cs_outer,
    z_cs_middle,
    rmajor,
    plasma_current,
):
    """Vertical field at the CS's inner and outer edge from everything *except* the CS.

    `peak_b_field_at_pf_coil`'s `n_coil == n_cs_pf_coils` path
    (`process/models/pfcoil.py:4451-4456`, `:4513-4600`), which is a genuinely different
    routine from the per-group path `calculate_pf_coil_peak_fields` above ports:

    - **`kk = 0`**: the fourteen CS filaments are omitted outright, because the CS's own
      contribution is `calculate_cs_self_peak_magnetic_field`'s job (`:4451-4456`). So
      none of the filament-clobbering defect this module's docstring records applies to
      this call at all.
    - **`n_coil_group = 99`** (`:3346`, `:3384`): no group index can equal `98`, so the
      `iii == n_coil_group - 1` test is false for every coil and Lyle's four-filament
      self-field expansion is never taken. Every one of the six PF coils contributes a
      *single* filament at its centre.
    - **`t_b_field_peak` is passed in rather than derived.** The re-derivation at
      `:4459-4487` sits inside the `else` of the `kk = 0` branch, so the caller's `5`
      and `2` are used as given -- which is why this helper takes `column` and
      `_peak_time_column` is not consulted.

    The plasma filament is present only when `t_b_field_peak > 2` (`:4591`), i.e. for
    the end-of-flat-top call and not the beginning-of-pulse one. As in
    `calculate_pf_coil_peak_fields` it is included unconditionally with the current
    masked to zero, for the same reason: a loop carrying zero current contributes
    exactly zero and the traced filament count stays fixed.

    Returns
    -------
    tuple
        `(b_pf_inner_vertical, b_pf_outer_vertical)` (T) -- PROCESS's third and fourth
        return values, the only two `ohcalc` reads.
    """
    f_at_time = jnp.take(waveform, column, axis=1)
    currents = c_peak[:N_PF_COILS] * f_at_time[:N_PF_COILS] * 1.0e6

    r_loops = jnp.concatenate([
        r_pf_coil_middle[:N_PF_COILS],
        jnp.asarray(rmajor)[None],
    ])
    z_loops = jnp.concatenate([z_pf_coil_middle[:N_PF_COILS], jnp.zeros(1)])
    c_loops = jnp.concatenate([
        currents,
        jnp.where(column >= 2, plasma_current, 0.0)[None],
    ])

    # Two test points, one identical loop set -- one batched call, not two traces of the
    # kernel. See `_b_field_over_test_points`.
    _, _, bz_edges, _ = _b_field_over_test_points(
        r_loops,
        z_loops,
        c_loops,
        jnp.stack([r_cs_inner, r_cs_outer]),
        jnp.stack([z_cs_middle, z_cs_middle]),
    )
    return bz_edges[0], bz_edges[1]


def calculate_cs_peak_fields(
    c_pf_cs_coil_pulse_start_ma,
    c_pf_cs_coil_flat_top_ma,
    c_pf_cs_coil_pulse_end_ma,
    r_pf_coil_middle,
    z_pf_coil_middle,
    r_cs_inner,
    r_cs_outer,
    z_cs_middle,
    z_cs_upper,
    j_cs_flat_top_end,
    j_cs_pulse_start,
    rmajor,
    plasma_current,
):
    """The CS's peak field at the end of flat-top and at the beginning of pulse (T).

    Ports `ohcalc`'s field block, `process/models/pfcoil.py:3327-3396`: two self-field
    evaluations, two `peak_b_field_at_pf_coil` calls at time points 5 and 2, and the
    four combinations PROCESS forms from them.

    **The two combinations differ in sign, and PROCESS says why in a comment**
    (`:3324-3326`, `:3379-3381`): at the end of flat-top the CS self-field and the
    external vertical field are of *opposite* sign at the coil's inner edge, so the
    magnitude is `|b_pf_inner_vertical - b_self|`; at the beginning of pulse they are of
    the *same* sign and it is `|b_self + b_pf_inner_vertical|`. Transcribed as written --
    the asymmetry is a physical claim about the pulse, not an algebraic slip.

    **`dz_cs_half` is `z_cs_upper`, not `dz_cs_full / 2`.** PROCESS passes
    `z_pf_coil_upper[n_cs_pf_coils - 1]` to both self-field calls (`:3336`, `:3372`) and
    `dz_cs_full / 2.0` to the *stress* calls a few lines later (`:3428`). The two are the
    same number on a CS centred on the midplane, which this one is
    (`calculate_cs_geometry` puts `z_cs_upper = -z_cs_lower`), so nothing here can
    distinguish them -- transcribed from the source rather than from the equality, the
    same call `exhaust.EuDemoReAttachmentMetric` records for its mint.

    Parameters
    ----------
    c_pf_cs_coil_pulse_start_ma, c_pf_cs_coil_flat_top_ma, c_pf_cs_coil_pulse_end_ma :
        Coil currents (MA) at the three time points, seven entries -- six PF coils then
        the CS. Taken as the three arrays rather than the derived peak/waveform pair,
        for the reason this module's docstring gives.
    r_pf_coil_middle, z_pf_coil_middle :
        Coil centre coordinates (m), six entries or more; only the six PF coils are read.
    r_cs_inner, r_cs_outer :
        CS inner and outer radii (m) -- the field is evaluated at these.
    z_cs_middle :
        CS mid-height (m), the test points' height.
    z_cs_upper :
        CS upper edge height (m), used as the solenoid half-height.
    j_cs_flat_top_end, j_cs_pulse_start :
        CS overall current density at the two time points (A/m^2).
    rmajor :
        Plasma major radius (m).
    plasma_current :
        Plasma current (A).

    Returns
    -------
    tuple
        `(b_cs_peak_flat_top_end, b_cs_peak_pulse_start, b_cs_self_outer_midplane,
        b_pf_coil_peak_cs, bpf2_cs)` -- the two peak fields (T), the outboard self-field
        (identically `0.0`, PROCESS's long-solenoid approximation, `:3360`), and the CS
        entries of the two whole-array peak fields (T).
    """
    c_peak, waveform = calculate_coil_current_waveform(
        c_pf_cs_coil_pulse_start_ma,
        c_pf_cs_coil_flat_top_ma,
        c_pf_cs_coil_pulse_end_ma,
    )

    b_self_flat_top_end = calculate_cs_self_peak_magnetic_field(
        j_cs=j_cs_flat_top_end,
        r_cs_inner=r_cs_inner,
        r_cs_outer=r_cs_outer,
        dz_cs_half=z_cs_upper,
    )
    bz_in_eof, bz_out_eof = _cs_external_field(
        c_peak,
        waveform,
        _T_B_FIELD_PEAK_FLAT_TOP_END - 1,
        r_pf_coil_middle,
        z_pf_coil_middle,
        r_cs_inner,
        r_cs_outer,
        z_cs_middle,
        rmajor,
        plasma_current,
    )
    b_cs_peak_flat_top_end = jnp.abs(bz_in_eof - b_self_flat_top_end)
    bohco = jnp.abs(bz_out_eof)

    b_self_pulse_start = calculate_cs_self_peak_magnetic_field(
        j_cs=j_cs_pulse_start,
        r_cs_inner=r_cs_inner,
        r_cs_outer=r_cs_outer,
        dz_cs_half=z_cs_upper,
    )
    bz_in_bop, bz_out_bop = _cs_external_field(
        c_peak,
        waveform,
        _T_B_FIELD_PEAK_PULSE_START - 1,
        r_pf_coil_middle,
        z_pf_coil_middle,
        r_cs_inner,
        r_cs_outer,
        z_cs_middle,
        rmajor,
        plasma_current,
    )
    b_cs_peak_pulse_start = jnp.abs(b_self_pulse_start + bz_in_bop)

    return (
        b_cs_peak_flat_top_end,
        b_cs_peak_pulse_start,
        0.0 * b_cs_peak_flat_top_end,
        jnp.maximum(b_cs_peak_flat_top_end, jnp.abs(b_cs_peak_pulse_start)),
        jnp.maximum(bohco, jnp.abs(bz_out_bop)),
    )


def calculate_cs_peak_fields_reference_widths(
    c_pf_cs_coil_pulse_start_ma,
    c_pf_cs_coil_flat_top_ma,
    c_pf_cs_coil_pulse_end_ma,
    r_pf_coil_middle,
    z_pf_coil_middle,
    r_cs_inner,
    r_cs_outer,
    z_cs_middle,
    z_cs_upper,
    j_cs_flat_top_end,
    j_cs_pulse_start,
    rmajor,
    plasma_current,
):
    """`CSCoilPeakField`: trims the three current arrays to the seven-entry (six PF +
    CS) width and the two coordinate arrays to the six PF coils.
    """
    return calculate_cs_peak_fields(
        c_pf_cs_coil_pulse_start_ma=c_pf_cs_coil_pulse_start_ma[: CS_INDEX + 1],
        c_pf_cs_coil_flat_top_ma=c_pf_cs_coil_flat_top_ma[: CS_INDEX + 1],
        c_pf_cs_coil_pulse_end_ma=c_pf_cs_coil_pulse_end_ma[: CS_INDEX + 1],
        r_pf_coil_middle=r_pf_coil_middle[:N_PF_COILS],
        z_pf_coil_middle=z_pf_coil_middle[:N_PF_COILS],
        r_cs_inner=r_cs_inner,
        r_cs_outer=r_cs_outer,
        z_cs_middle=z_cs_middle,
        z_cs_upper=z_cs_upper,
        j_cs_flat_top_end=j_cs_flat_top_end,
        j_cs_pulse_start=j_cs_pulse_start,
        rmajor=rmajor,
        plasma_current=plasma_current,
    )
