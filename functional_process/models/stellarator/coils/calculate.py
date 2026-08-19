"""Pure-functional port of the tier-1 functions in `coils/calculate.py` (registry unit #9).

Audit record: `functional_process/models/stellarator/coils/calculate.md`. That file's
`st_coil` (593 lines) is the orchestrator called directly from `Stellarator.run()`; most
of its body is 12 short, independent helper functions, 10 of which are tier-1 (pure
arithmetic, no internal solve, no calls into any other file) and were ported first. The
other two -- `winding_pack_total_size` (a 200-point sampled curve fed into `intersect`, a
Newton-Raphson root-find in `coils/coils.py`) and `st_coil` itself (the orchestrator,
which also calls `coils/mass.py`, `coils/quench.py`, `coils/forces.py`,
`coils/output.py`) -- were blocked on registry units #10-14. Units #10 (partially, see
below), #11, #12 and #14 are now ported, and #13 is confirmed pure reporting with nothing
to port, which unblocked both: `winding_pack_total_size` is ported below (tier-2, a
`Tier2Contract`, same pattern as `coils.py`'s own `intersect`), and `st_coil` is ported as
a plain composed function (tier-3; see the record for why it gets no `cottax` node of its
own).

`winding_pack_total_size` calls `intersect`/`bmax_from_awp` (`coils/coils.py`, already
ported) directly, and needs `jcrit_from_material`'s dispatch on `i_tf_sc_mat` -- which
itself is **not** ported (`coils.py` remains out of this unit's boundary; see
`coils.md`). `_critical_current_density_by_material` below is therefore a local
restatement of that dispatch, scoped to this unit's own solve, calling the real ported
material models in `functional_process/models/physics/superconductors.py` directly. It is
not itself the audited port of `jcrit_from_material` -- that stays unit #10's to do,
likely split one node per `i_tf_sc_mat` branch per `switches.md`'s guidance -- see the
record's "switches touched" section.

Every function below keeps its original name (already `calculate_*`-shaped in the
source, so nothing to rename per `naming_convention.md`) and takes exactly the fields it
reads as explicit arguments -- no `data: DataStructure` parameter anywhere, unlike the
source (`_audit/traceability_policy.md`: closing the `data` back-door is the whole
point).
"""

import equinox as eqx
import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    Input,
    Output,
)

from functional_process.models.physics.superconductors import (
    bi2212,
    gl_nbti,
    gl_rebco,
    itersc,
    jcrit_nbti,
    jcrit_rebco,
    western_superconducting_nb3sn,
)
from functional_process.models.stellarator.coils.coils import bmax_from_awp, intersect
from functional_process.models.stellarator.coils.forces import (
    calculate_centering_force_avg_mn,
    calculate_centering_force_max_mn,
    calculate_centering_force_min_mn,
    calculate_max_force_density,
    calculate_max_force_density_mnm,
    calculate_max_lateral_force_density,
    calculate_max_radial_force_density,
    calculate_maximum_stress,
)
from functional_process.models.stellarator.coils.mass import calculate_coils_mass
from functional_process.models.stellarator.coils.quench import (
    calculate_quench_protection,
)


def calculate_coil_toroidal_thickness(
    dx_tf_wp_primary_toroidal, dx_tf_side_case_min, dx_tf_wp_insulation
):
    """Thickness of the inboard coil leg in the toroidal direction (m).

    Ports `calculate_coil_toroidal_thickness`.

    Returns
    -------
    :
        `dx_tf_inboard_out_toroidal`.
    """
    return (
        dx_tf_wp_primary_toroidal + 2.0 * dx_tf_side_case_min + 2.0 * dx_tf_wp_insulation
    )


class CoilToroidalThickness(ExplicitFunction):
    dx_tf_inboard_out_toroidal = Output(lambda s: s.tfcoil.dx_tf_inboard_out_toroidal)

    def __call__(
        self,
        dx_tf_wp_primary_toroidal=Input(lambda s: s.tfcoil.dx_tf_wp_primary_toroidal),
        dx_tf_side_case_min=Input(lambda s: s.tfcoil.dx_tf_side_case_min),
        dx_tf_wp_insulation=Input(lambda s: s.tfcoil.dx_tf_wp_insulation),
    ):
        return calculate_coil_toroidal_thickness(
            dx_tf_wp_primary_toroidal, dx_tf_side_case_min, dx_tf_wp_insulation
        )


def calculate_coil_radial_thickness(
    dr_tf_nose_case, dr_tf_wp_with_insulation, dr_tf_plasma_case, dx_tf_wp_insulation
):
    """Thickness of the inboard/outboard coil leg in the radial direction (m).

    Ports `calculate_coil_radial_thickness`. The source writes the identical value to
    both `.build.dr_tf_inboard` and `.build.dr_tf_outboard` ("same as inboard") --
    ported as one value, returned once; the port's caller is responsible for the
    duplicate write if both fields are still wanted downstream, same treatment as any
    other `redundant-duplicate-write`.

    Returns
    -------
    :
        `dr_tf_inboard` (== `dr_tf_outboard`).
    """
    return (
        dr_tf_nose_case
        + dr_tf_wp_with_insulation
        + dr_tf_plasma_case
        + 2.0 * dx_tf_wp_insulation
    )


class CoilRadialThickness(ExplicitFunction):
    dr_tf_inboard = Output(lambda s: s.build.dr_tf_inboard)

    def __call__(
        self,
        dr_tf_nose_case=Input(lambda s: s.tfcoil.dr_tf_nose_case),
        dr_tf_wp_with_insulation=Input(lambda s: s.tfcoil.dr_tf_wp_with_insulation),
        dr_tf_plasma_case=Input(lambda s: s.tfcoil.dr_tf_plasma_case),
        dx_tf_wp_insulation=Input(lambda s: s.tfcoil.dx_tf_wp_insulation),
    ):
        return calculate_coil_radial_thickness(
            dr_tf_nose_case,
            dr_tf_wp_with_insulation,
            dr_tf_plasma_case,
            dx_tf_wp_insulation,
        )


def calculate_coil_cross_sectional_area(
    a_tf_wp_with_insulation, dr_tf_inboard, dx_tf_inboard_out_toroidal
):
    """Overall coil leg and surrounding-case cross-sectional areas (m2).

    Ports `calculate_coil_cross_sectional_area`. Assumes inboard and outboard legs are
    identical, per the source.

    Returns
    -------
    :
        `(a_tf_leg_outboard, a_tf_coil_inboard_case)`.
    """
    a_tf_leg_outboard = dr_tf_inboard * dx_tf_inboard_out_toroidal
    a_tf_coil_inboard_case = a_tf_leg_outboard - a_tf_wp_with_insulation
    return a_tf_leg_outboard, a_tf_coil_inboard_case


class CoilCrossSectionalArea(ExplicitFunction):
    a_tf_leg_outboard = Output(lambda s: s.tfcoil.a_tf_leg_outboard)
    a_tf_coil_inboard_case = Output(lambda s: s.tfcoil.a_tf_coil_inboard_case)

    def __call__(
        self,
        a_tf_wp_with_insulation=Input(lambda s: s.tfcoil.a_tf_wp_with_insulation),
        dr_tf_inboard=Input(lambda s: s.build.dr_tf_inboard),
        dx_tf_inboard_out_toroidal=Input(lambda s: s.tfcoil.dx_tf_inboard_out_toroidal),
    ):
        return calculate_coil_cross_sectional_area(
            a_tf_wp_with_insulation, dr_tf_inboard, dx_tf_inboard_out_toroidal
        )


def calculate_coil_half_widths(dx_tf_inboard_out_toroidal):
    """Half-widths of the coil side nearest the torus centreline / plasma (m).

    Ports `calculate_coil_half_widths` -- both outputs are the same formula in the
    source (`0.5 * dx_tf_inboard_out_toroidal`), kept as two return values since they
    are two distinct `data` fields with independent readers downstream, not a
    redundant write of one quantity.

    Returns
    -------
    :
        `(tfocrn, tficrn)`.
    """
    half = 0.5 * dx_tf_inboard_out_toroidal
    return half, half


class CoilHalfWidths(ExplicitFunction):
    tfocrn = Output(lambda s: s.tfcoil.tfocrn)
    tficrn = Output(lambda s: s.tfcoil.tficrn)

    def __call__(
        self,
        dx_tf_inboard_out_toroidal=Input(lambda s: s.tfcoil.dx_tf_inboard_out_toroidal),
    ):
        return calculate_coil_half_widths(dx_tf_inboard_out_toroidal)


def calculate_plasma_facing_coil_area(
    n_tf_coils, dx_tf_inboard_out_toroidal, len_tf_coil
):
    """Total surface area of the coil side facing the plasma, inboard/outboard (m2).

    Ports `calculate_plasma_facing_coil_area` -- outboard is identical to inboard in
    the source ("same as inboard"), same treatment as `calculate_coil_half_widths`.

    Returns
    -------
    :
        `(tfsai, tfsao)`.
    """
    area = n_tf_coils * dx_tf_inboard_out_toroidal * 0.5 * len_tf_coil
    return area, area


class PlasmaFacingCoilArea(ExplicitFunction):
    tfsai = Output(lambda s: s.tfcoil.tfsai)
    tfsao = Output(lambda s: s.tfcoil.tfsao)

    def __call__(
        self,
        n_tf_coils=Input(lambda s: s.tfcoil.n_tf_coils),
        dx_tf_inboard_out_toroidal=Input(lambda s: s.tfcoil.dx_tf_inboard_out_toroidal),
        len_tf_coil=Input(lambda s: s.tfcoil.len_tf_coil),
    ):
        return calculate_plasma_facing_coil_area(
            n_tf_coils, dx_tf_inboard_out_toroidal, len_tf_coil
        )


def calculate_coil_coil_toroidal_gap(
    stella_config_dmin,
    r_coil_major,
    r_coil_minor,
    stella_config_coil_rmajor,
    stella_config_coil_rminor,
    dx_tf_inboard_out_toroidal,
):
    """Toroidal gap between two stellarator coils, and the leftover gap (m).

    Ports `calculate_coil_coil_toroidal_gap`.

    Returns
    -------
    :
        `(coilcoilgap, toroidalgap)`.
    """
    toroidalgap = (
        stella_config_dmin
        * (r_coil_major - r_coil_minor)
        / (stella_config_coil_rmajor - stella_config_coil_rminor)
    )
    coilcoilgap = toroidalgap - dx_tf_inboard_out_toroidal
    return coilcoilgap, toroidalgap


class CoilCoilToroidalGap(ExplicitFunction):
    toroidalgap = Output(lambda s: s.tfcoil.toroidalgap)

    def __call__(
        self,
        stella_config_dmin=Input(lambda s: s.stellarator_config.stella_config_dmin),
        r_coil_major=Input(lambda s: s.stellarator.r_coil_major),
        r_coil_minor=Input(lambda s: s.stellarator.r_coil_minor),
        stella_config_coil_rmajor=Input(
            lambda s: s.stellarator_config.stella_config_coil_rmajor
        ),
        stella_config_coil_rminor=Input(
            lambda s: s.stellarator_config.stella_config_coil_rminor
        ),
        dx_tf_inboard_out_toroidal=Input(lambda s: s.tfcoil.dx_tf_inboard_out_toroidal),
    ):
        # `coilcoilgap` is a local in the source (returned to the caller, never
        # written to `data`) -- only `toroidalgap` is a node output here.
        _coilcoilgap, toroidalgap = calculate_coil_coil_toroidal_gap(
            stella_config_dmin,
            r_coil_major,
            r_coil_minor,
            stella_config_coil_rmajor,
            stella_config_coil_rminor,
            dx_tf_inboard_out_toroidal,
        )
        return toroidalgap


def calculate_coils_summary_variables(
    n_tf_coils, a_tf_leg_outboard, coilcurrent, r_coil_major, r_coil_minor, awp_rad
):
    """Aggregate quantities over all coils (area, current, current density, peak-field radius).

    Ports `calculate_coils_summary_variables`.

    Returns
    -------
    :
        `(a_tf_inboard_total, c_tf_total, j_tf_coil_full_area,
        r_b_tf_inboard_peak_symmetric)`.
    """
    a_tf_inboard_total = n_tf_coils * a_tf_leg_outboard
    c_tf_total = n_tf_coils * coilcurrent * 1.0e6
    j_tf_coil_full_area = c_tf_total / a_tf_inboard_total
    r_b_tf_inboard_peak_symmetric = r_coil_major - r_coil_minor + awp_rad
    return (
        a_tf_inboard_total,
        c_tf_total,
        j_tf_coil_full_area,
        r_b_tf_inboard_peak_symmetric,
    )


class CoilsSummaryVariables(ExplicitFunction):
    a_tf_inboard_total = Output(lambda s: s.tfcoil.a_tf_inboard_total)
    c_tf_total = Output(lambda s: s.tfcoil.c_tf_total)
    j_tf_coil_full_area = Output(lambda s: s.tfcoil.j_tf_coil_full_area)
    r_b_tf_inboard_peak_symmetric = Output(
        lambda s: s.tfcoil.r_b_tf_inboard_peak_symmetric
    )

    def __call__(
        self,
        n_tf_coils=Input(lambda s: s.tfcoil.n_tf_coils),
        a_tf_leg_outboard=Input(lambda s: s.tfcoil.a_tf_leg_outboard),
        coilcurrent=Input(lambda s: s.stellarator.coilcurrent),
        r_coil_major=Input(lambda s: s.stellarator.r_coil_major),
        r_coil_minor=Input(lambda s: s.stellarator.r_coil_minor),
        awp_rad=Input(lambda s: s.tfcoil.dr_tf_wp_with_insulation),
    ):
        return calculate_coils_summary_variables(
            n_tf_coils,
            a_tf_leg_outboard,
            coilcurrent,
            r_coil_major,
            r_coil_minor,
            awp_rad,
        )


def calculate_inductance(
    stella_config_inductance,
    f_st_rmajor,
    r_coil_minor,
    stella_config_coil_rminor,
    f_st_n_coils,
):
    """Coil inductance (units as PROCESS's `stella_config_inductance`), scaled a2/R.

    Ports `calculate_inductance`. Reporting-only in the source (printed by `write()`,
    never stored to `data`) -- no `ExplicitFunction` wrap, same treatment as
    `calculate_intercoil_mass_scaling_reference` in `stellarator_D_structure.py`.

    Returns
    -------
    :
        `inductance`.
    """
    return (
        stella_config_inductance
        / f_st_rmajor
        * (r_coil_minor / stella_config_coil_rminor) ** 2
        * f_st_n_coils**2
    )


def calculate_stored_magnetic_energy(
    stella_config_inductance,
    f_st_rmajor,
    r_coil_minor,
    stella_config_coil_rminor,
    f_st_n_coils,
    c_tf_total,
    n_tf_coils,
):
    """Total magnetic energy stored in the TF coil set (GJ).

    Ports `calculate_stored_magnetic_energy`. Recomputes `calculate_inductance`'s
    formula inline rather than taking it as an argument -- matches the source, which
    does the same (`data.tfcoil.e_tf_magnetic_stored_total_gj` is derived independently
    of the `inductance` local `st_coil` also computes for the report).

    Returns
    -------
    :
        `e_tf_magnetic_stored_total_gj`.
    """
    inductance = calculate_inductance(
        stella_config_inductance,
        f_st_rmajor,
        r_coil_minor,
        stella_config_coil_rminor,
        f_st_n_coils,
    )
    return 0.5 * inductance * (c_tf_total / n_tf_coils) ** 2 * 1.0e-9


class StoredMagneticEnergy(ExplicitFunction):
    e_tf_magnetic_stored_total_gj = Output(
        lambda s: s.tfcoil.e_tf_magnetic_stored_total_gj
    )

    def __call__(
        self,
        stella_config_inductance=Input(
            lambda s: s.stellarator_config.stella_config_inductance
        ),
        f_st_rmajor=Input(lambda s: s.stellarator.f_st_rmajor),
        r_coil_minor=Input(lambda s: s.stellarator.r_coil_minor),
        stella_config_coil_rminor=Input(
            lambda s: s.stellarator_config.stella_config_coil_rminor
        ),
        f_st_n_coils=Input(lambda s: s.stellarator.f_st_n_coils),
        c_tf_total=Input(lambda s: s.tfcoil.c_tf_total),
        n_tf_coils=Input(lambda s: s.tfcoil.n_tf_coils),
    ):
        return calculate_stored_magnetic_energy(
            stella_config_inductance,
            f_st_rmajor,
            r_coil_minor,
            stella_config_coil_rminor,
            f_st_n_coils,
            c_tf_total,
            n_tf_coils,
        )


def calculate_winding_pack_geometry(
    dx_tf_turn_general, dx_tf_turn_steel, dx_tf_turn_insulation
):
    """Cross-sectional area of the cable space and conduit case, per turn (m2).

    Ports `calculate_winding_pack_geometry`. The source logs a warning (not an error)
    when the cable-space dimension goes negative -- dropped here: a side-effecting log
    on a data-dependent condition inside a traced function isn't representable, and the
    caller has the same information from a non-finite/negative result without it (see
    the audit record's JAX-difficulty flags).

    Returns
    -------
    :
        `(a_tf_turn_cable_space_no_void, a_tf_turn_steel)`.
    """
    dx_tf_turn_cable_space_average = dx_tf_turn_general - 2.0 * (
        dx_tf_turn_steel + dx_tf_turn_insulation
    )
    a_tf_turn_cable_space_no_void = 0.9 * dx_tf_turn_cable_space_average**2
    a_tf_turn_steel = (
        dx_tf_turn_cable_space_average + 2.0 * dx_tf_turn_steel
    ) ** 2 - a_tf_turn_cable_space_no_void
    return a_tf_turn_cable_space_no_void, a_tf_turn_steel


class WindingPackGeometry(ExplicitFunction):
    a_tf_turn_cable_space_no_void = Output(
        lambda s: s.tfcoil.a_tf_turn_cable_space_no_void
    )
    a_tf_turn_steel = Output(lambda s: s.tfcoil.a_tf_turn_steel)

    def __call__(
        self,
        dx_tf_turn_general=Input(lambda s: s.tfcoil.dx_tf_turn_general),
        dx_tf_turn_steel=Input(lambda s: s.tfcoil.dx_tf_turn_steel),
        dx_tf_turn_insulation=Input(lambda s: s.tfcoil.dx_tf_turn_insulation),
    ):
        return calculate_winding_pack_geometry(
            dx_tf_turn_general, dx_tf_turn_steel, dx_tf_turn_insulation
        )


def calculate_current(f_st_b, stella_config_i0, f_st_rmajor, f_st_n_coils):
    """Total coil current, and its ratio to the reference current.

    Ports `calculate_current`. `f_st_i_total` is a `data`-back-door write in the
    source (`data.stellarator.f_st_i_total = ...`), alongside the return value
    `coilcurrent` -- both are real, independent outputs (not a redundant duplicate: two
    different fields, two different values), so both become return values here.

    Returns
    -------
    :
        `(coilcurrent, f_st_i_total)`.
    """
    coilcurrent = f_st_b * stella_config_i0 * f_st_rmajor / f_st_n_coils
    f_st_i_total = coilcurrent / stella_config_i0
    return coilcurrent, f_st_i_total


class CoilCurrent(ExplicitFunction):
    """`coilcurrent` has no PROCESS storage location -- it is a local in `st_coil`,
    threaded manually into `winding_pack_total_size` and
    `calculate_coils_summary_variables`. `.stellarator.coilcurrent` is an invented
    `VarPath` (per `naming_convention.md`: port the existing name where one exists;
    mint one where it doesn't), needed because `CoilsSummaryVariables` below reads it --
    without minting it, that node would have no way to source this input at all.
    """

    coilcurrent = Output(lambda s: s.stellarator.coilcurrent)
    f_st_i_total = Output(lambda s: s.stellarator.f_st_i_total)

    def __call__(
        self,
        f_st_b=Input(lambda s: s.stellarator.f_st_b),
        stella_config_i0=Input(lambda s: s.stellarator_config.stella_config_i0),
        f_st_rmajor=Input(lambda s: s.stellarator.f_st_rmajor),
        f_st_n_coils=Input(lambda s: s.stellarator.f_st_n_coils),
    ):
        return calculate_current(f_st_b, stella_config_i0, f_st_rmajor, f_st_n_coils)


def _critical_current_density_by_material(
    b_max,
    t_helium,
    i_tf_sc_mat,
    b_crit_upper_nbti,
    b_crit_sc,
    f_a_tf_turn_cable_copper,
    f_hts,
    t_crit_nbti,
    t_crit_sc,
    f_a_tf_turn_cable_space_extra_void,
    j_wp,
):
    """Local restatement of `jcrit_from_material`'s dispatch (`coils/coils.py`, unit
    #10, not itself ported -- see the module and record docstrings). `i_tf_sc_mat` is a
    graph-build-time switch (`_audit/naming_convention.md` "switches are not ports"), so
    branch selection is ordinary Python control flow, not `jnp.where` -- only the one
    material formula actually selected gets traced. Mirrors the source's 8 branches
    exactly, calling the already-ported material models in
    `functional_process/models/physics/superconductors.py` directly.

    Branch 6 (REBCO) calls `jcrit_rebco(t_helium, b_max)` with the ported function's real
    2-argument signature -- the source's own call site (`coils.py:136`,
    `jcrit_rebco(t_helium, b_max, 0)`) passes an extra positional argument that
    `jcrit_rebco` does not accept and would raise `TypeError` if ever executed (confirmed
    by directly running PROCESS's `winding_pack_total_size` with `i_tf_sc_mat=6` at a
    realistic operating point while building this port -- see the record's "real PROCESS
    bugs found"). Not reproduced here: this dispatcher is not `coils.py`'s
    `jcrit_from_material` and has no call site to be faithful to; it exists so
    `winding_pack_total_size` below has *a* working REBCO branch rather than none.

    Returns
    -------
    :
        Critical current density in the superconductor, MA/m2 (matches
        `jcrit_from_material`'s `j_crit_sc * 1e-6` scaling).
    """
    strain = -0.005

    if i_tf_sc_mat == 1:  # ITER Nb3Sn critical surface parameterization
        bc20m, tc0m = 32.97, 16.06
        j_crit_sc, _bcrit, _tcrit = itersc(t_helium, b_max, strain, bc20m, tc0m)
        j_crit_sc = jnp.where(b_max > bc20m, 1.0e-9, j_crit_sc)
        j_crit_sc = jnp.maximum(1.0e-9, j_crit_sc)
    elif i_tf_sc_mat == 2:  # Bi-2212 high temperature superconductor
        f_he = f_a_tf_turn_cable_space_extra_void
        jstrand = j_wp / (1.0 - f_he)
        j_crit_cable, _tmarg = bi2212(b_max, jstrand, t_helium, f_hts)
        j_crit_sc = j_crit_cable / (1.0 - f_a_tf_turn_cable_copper)
    elif i_tf_sc_mat == 3:  # NbTi data (Lubell scaling)
        bc20m, tc0m, c0 = 15.0, 9.3, 1.0
        j_crit_sc, _tcrit = jcrit_nbti(t_helium, b_max, c0, bc20m, tc0m)
        j_crit_sc = jnp.where(b_max > bc20m, 1.0e-9, j_crit_sc)
        j_crit_sc = jnp.maximum(1.0e-9, j_crit_sc)
    elif i_tf_sc_mat == 4:  # As (1), but user-defined bc20m/tc0m
        j_crit_sc, _bcrit, _tcrit = itersc(t_helium, b_max, strain, b_crit_sc, t_crit_sc)
    elif i_tf_sc_mat == 5:  # WST Nb3Sn parameterisation
        bc20m, tc0m = 32.97, 16.06
        j_crit_sc, _bcrit, _tcrit = western_superconducting_nb3sn(
            t_helium, b_max, strain, bc20m, tc0m
        )
    elif i_tf_sc_mat == 6:  # REBCO 2nd generation HTS superconductor
        j_crit_sc, _validity, _, _ = jcrit_rebco(t_helium, b_max)
        j_crit_sc = jnp.maximum(1.0e-9, j_crit_sc)
    elif i_tf_sc_mat == 7:  # Durham Ginzburg-Landau Nb-Ti parameterisation
        j_crit_sc, _bcrit, _tcrit = gl_nbti(
            t_helium, b_max, strain, b_crit_upper_nbti, t_crit_nbti
        )
    elif i_tf_sc_mat == 8:  # Durham Ginzburg-Landau REBCO parameterisation
        bc20m, tc0m = 429.0, 185.0
        j_crit_sc, _bcrit, _tcrit = gl_rebco(t_helium, b_max, strain, bc20m, tc0m)
    else:
        raise ValueError(f"i_tf_sc_mat={i_tf_sc_mat!r} is not in range [1, 8]")

    return j_crit_sc * 1.0e-6


_N_WINDING_PACK_SAMPLES = 200


def winding_pack_curves(
    r_coil_major,
    r_coil_minor,
    coilcurrent,
    n_tf_coils,
    i_tf_sc_mat,
    stella_config_a1,
    stella_config_a2,
    stella_config_wp_ratio,
    tftmp,
    tmargmin,
    b_crit_upper_nbti,
    bcritsc,
    f_a_tf_turn_cable_copper,
    fhts,
    t_crit_nbti,
    tcritsc,
    f_a_tf_turn_cable_space_extra_void,
    j_tf_wp,
    f_j_tf_wp_critical_max,
    a_tf_turn_cable_space_no_void,
    dx_tf_turn_general,
):
    """The `(wp_width_r, lhs, rhs)` curves `winding_pack_total_size` finds the crossing
    of, plus the superconductor area fraction they're built from.

    Split out from `winding_pack_total_size` below so the same sampling can be rebuilt
    from a `Tier2Contract` sample's kwargs by the harness's residual function
    (`test_calculate.py`), the same way `intersect_residual` lets `coils.md`'s
    `TestIntersect` re-check `intersect`'s own defining equation. Not independently
    audited/tested -- an internal seam, not a second port.

    Returns
    -------
    :
        `(wp_width_r, lhs, rhs, fraction_area_superconductor_of_wp)`.
    """
    n_it = _N_WINDING_PACK_SAMPLES
    k = jnp.arange(n_it, dtype=float)
    lo = r_coil_minor / (150.0 if i_tf_sc_mat == 6 else 40.0)
    hi = r_coil_minor / 1.0
    wp_width_r = lo + (k / (n_it - 1.0)) * (hi - lo)

    b_max_k = bmax_from_awp(
        wp_width_r,
        coilcurrent,
        n_tf_coils,
        r_coil_major,
        r_coil_minor,
        stella_config_a1,
        stella_config_a2,
    )

    t_helium = tftmp + tmargmin
    jcrit_vector = _critical_current_density_by_material(
        b_max_k,
        t_helium,
        i_tf_sc_mat,
        b_crit_upper_nbti,
        bcritsc,
        f_a_tf_turn_cable_copper,
        fhts,
        t_crit_nbti,
        tcritsc,
        f_a_tf_turn_cable_space_extra_void,
        j_tf_wp,
    )

    lhs = f_j_tf_wp_critical_max * jcrit_vector

    fraction_area_superconductor_of_wp = (
        (a_tf_turn_cable_space_no_void * (1.0 - f_a_tf_turn_cable_space_extra_void))
        * (1.0 - f_a_tf_turn_cable_copper)
        / (dx_tf_turn_general**2)
    )

    rhs = coilcurrent / (
        wp_width_r**2 / stella_config_wp_ratio * fraction_area_superconductor_of_wp
    )

    return wp_width_r, lhs, rhs, fraction_area_superconductor_of_wp


def winding_pack_pre_intersect(
    r_coil_major,
    r_coil_minor,
    coilcurrent,
    n_tf_coils,
    i_tf_sc_mat,
    stella_config_a1,
    stella_config_a2,
    stella_config_wp_ratio,
    tftmp,
    tmargmin,
    b_crit_upper_nbti,
    bcritsc,
    f_a_tf_turn_cable_copper,
    fhts,
    t_crit_nbti,
    tcritsc,
    f_a_tf_turn_cable_space_extra_void,
    j_tf_wp,
    f_j_tf_wp_critical_max,
    a_tf_turn_cable_space_no_void,
    dx_tf_turn_general,
):
    """The half of `winding_pack_total_size` that runs *before* `intersect`: builds the
    sampled `(wp_width_r, lhs, rhs)` curves (`winding_pack_curves`, unchanged) plus
    `intersect`'s own starting guess (`wp_width_r_min_guess`).

    Split out so `coils.py`'s `Intersect` (an `ImplicitFunction`/`RootFind` pair, see
    that class's own docstring) can sit structurally between this function and
    `winding_pack_post_intersect` below, instead of `intersect` being called eagerly in
    the middle of one large function -- exactly `winding_pack_total_size`'s own
    docstring note on why this split exists. Not independently audited/tested on its
    own, same internal-seam status as `winding_pack_curves` itself.

    Returns
    -------
    :
        `(wp_width_r, lhs, rhs, fraction_area_superconductor_of_wp,
        wp_width_r_min_guess)`.
    """
    wp_width_r, lhs, rhs, fraction_area_superconductor_of_wp = winding_pack_curves(
        r_coil_major,
        r_coil_minor,
        coilcurrent,
        n_tf_coils,
        i_tf_sc_mat,
        stella_config_a1,
        stella_config_a2,
        stella_config_wp_ratio,
        tftmp,
        tmargmin,
        b_crit_upper_nbti,
        bcritsc,
        f_a_tf_turn_cable_copper,
        fhts,
        t_crit_nbti,
        tcritsc,
        f_a_tf_turn_cable_space_extra_void,
        j_tf_wp,
        f_j_tf_wp_critical_max,
        a_tf_turn_cable_space_no_void,
        dx_tf_turn_general,
    )
    wp_width_r_min_guess = (r_coil_minor / (20.0 if i_tf_sc_mat == 6 else 10.0)) ** 2
    return wp_width_r, lhs, rhs, fraction_area_superconductor_of_wp, wp_width_r_min_guess


def winding_pack_post_intersect(
    wp_width_r_min,
    r_coil_major,
    r_coil_minor,
    coilcurrent,
    n_tf_coils,
    stella_config_a1,
    stella_config_a2,
    stella_config_wp_ratio,
    f_a_tf_turn_cable_space_extra_void,
    a_tf_turn_cable_space_no_void,
    dx_tf_turn_general,
    dx_tf_wp_insulation,
    a_tf_turn_steel,
):
    """The half of `winding_pack_total_size` that runs *after* `intersect`: the
    turn-size-floor clamp on the resolved crossing point (`wp_width_r_min`, `intersect`'s
    raw, unclamped answer here) and everything downstream of it.

    Split out for the same reason as `winding_pack_pre_intersect` above -- see that
    function's docstring and `coils.py`'s `Intersect`. Unlike the pre-intersect half,
    nothing here depends on `i_tf_sc_mat` at all (the material dispatch is entirely
    upstream, inside `winding_pack_curves`), so this function takes no such argument.

    Returns
    -------
    :
        `(b_tf_inboard_peak_symmetric, dx_tf_wp_primary_toroidal,
        dx_tf_wp_secondary_toroidal, dr_tf_wp_with_insulation, j_tf_wp,
        n_tf_coil_turns, c_tf_turn, a_tf_wp_conductor, a_tf_wp_extra_void,
        a_tf_coil_wp_turn_insulation, a_tf_wp_steel, a_tf_wp_no_insulation,
        a_tf_wp_with_insulation)` -- same as `winding_pack_total_size`'s own return,
        minus `fraction_area_superconductor_of_wp` (that one belongs to the
        pre-intersect half, see its own docstring).
    """
    # Maximum field at superconductor surface is achieved at this minimum WP width --
    # source comment, kept verbatim; the clamp itself is the turn-size floor.
    wp_width_r_min = jnp.maximum(dx_tf_turn_general**2, wp_width_r_min)

    b_tf_inboard_peak_symmetric = bmax_from_awp(
        wp_width_r_min,
        coilcurrent,
        n_tf_coils,
        r_coil_major,
        r_coil_minor,
        stella_config_a1,
        stella_config_a2,
    )

    awp_tor = wp_width_r_min / stella_config_wp_ratio
    dx_tf_wp_primary_toroidal = awp_tor
    dx_tf_wp_secondary_toroidal = awp_tor
    dr_tf_wp_with_insulation = wp_width_r_min

    a_tf_wp_with_insulation = (dr_tf_wp_with_insulation + 2.0 * dx_tf_wp_insulation) * (
        dx_tf_wp_primary_toroidal + 2.0 * dx_tf_wp_insulation
    )
    a_tf_wp_no_insulation = awp_tor * wp_width_r_min

    j_tf_wp_new = coilcurrent * 1.0e6 / a_tf_wp_no_insulation
    n_tf_coil_turns = a_tf_wp_no_insulation / dx_tf_turn_general**2
    c_tf_turn = coilcurrent * 1.0e6 / n_tf_coil_turns
    a_tf_wp_conductor = (
        a_tf_turn_cable_space_no_void
        * n_tf_coil_turns
        * (1.0 - f_a_tf_turn_cable_space_extra_void)
    )
    a_tf_wp_extra_void = (
        a_tf_turn_cable_space_no_void
        * n_tf_coil_turns
        * f_a_tf_turn_cable_space_extra_void
    )
    a_tf_coil_wp_turn_insulation = n_tf_coil_turns * (
        dx_tf_turn_general**2 - a_tf_turn_steel - a_tf_turn_cable_space_no_void
    )
    a_tf_wp_steel = n_tf_coil_turns * a_tf_turn_steel

    return (
        b_tf_inboard_peak_symmetric,
        dx_tf_wp_primary_toroidal,
        dx_tf_wp_secondary_toroidal,
        dr_tf_wp_with_insulation,
        j_tf_wp_new,
        n_tf_coil_turns,
        c_tf_turn,
        a_tf_wp_conductor,
        a_tf_wp_extra_void,
        a_tf_coil_wp_turn_insulation,
        a_tf_wp_steel,
        a_tf_wp_no_insulation,
        a_tf_wp_with_insulation,
    )


def winding_pack_total_size(
    r_coil_major,
    r_coil_minor,
    coilcurrent,
    n_tf_coils,
    i_tf_sc_mat,
    stella_config_a1,
    stella_config_a2,
    stella_config_wp_ratio,
    tftmp,
    tmargmin,
    b_crit_upper_nbti,
    bcritsc,
    f_a_tf_turn_cable_copper,
    fhts,
    t_crit_nbti,
    tcritsc,
    f_a_tf_turn_cable_space_extra_void,
    j_tf_wp,
    f_j_tf_wp_critical_max,
    a_tf_turn_cable_space_no_void,
    dx_tf_turn_general,
    dx_tf_wp_insulation,
    a_tf_turn_steel,
):
    """Winding pack total size: the coil-current-carrying-capacity crossing point.

    Ports `winding_pack_total_size`. Samples 200 points of the operating-current curve
    (`lhs`, the critical-current-density-derived limit) against the geometric-current
    curve (`rhs`) over a swept winding-pack radial width, then finds where they cross
    with `intersect` (`coils/coils.py`, already ported, tier-2 internal solve) -- see
    the record for why this makes the whole function tier-2 too, `Tier2Contract`, same
    reasoning as `coils.md`'s `TestIntersect`.

    `j_tf_wp` is a genuine finding, not a modelling choice of this port: the source reads
    `data.tfcoil.j_tf_wp` here (only used by the `i_tf_sc_mat == 2` branch, inside
    `_critical_current_density_by_material`) *before* this same function overwrites it
    near the end (`data.tfcoil.j_tf_wp = coilcurrent * 1e6 / a_tf_wp_no_insulation`) --
    a self-referential, cross-call read of last call's own output, not this call's. Kept
    faithful here as two independent things: `j_tf_wp` (an explicit input, the stale
    prior value) and the new `j_tf_wp` in the return tuple (this call's fresh value) --
    see the record's data-footprint table and JAX-difficulty flags; a single `cottax`
    node cannot own and read the same `VarPath` (`spec.py`: "a node may not read what it
    owns"). Resolved at the node level, not by changing this pure function (which keeps
    returning both, faithfully) -- **and resolved as an ordinary cross-node cycle, not a
    single-node `FixedPointFunction`**: `WindingPackIntersectInputs` (this function's own
    pre-intersect node) reads the real `.tfcoil.j_tf_wp` as a plain, non-owning `Input`,
    and `WindingPackTotalSizePost` (the post-intersect node) owns it as an ordinary
    `Output`. Since the two are connected through `coils.py`'s `Intersect` in between,
    this is a real multi-node cycle (`WindingPackIntersectInputs -> Intersect ->
    WindingPackTotalSizePost -> WindingPackIntersectInputs`), the same "Shape A" shape as
    `Divertor`/`AFwTotalWithPowerflow` -- `Blocking`/`to_graph()` finds the SCC on its
    own, no `Cut`/`FixedPointFunction` wrapper needed (`_audit/next_steps.md` §5). An
    earlier pass wrote a `WindingPackJTfWp` `FixedPointFunction` instead, duplicating
    this whole function's computation a second time just to isolate `j_tf_wp` alone;
    deleted now that the split nodes carry the same self-reference without duplication.
    See `WindingPackIntersectInputs`/`WindingPackTotalSizePost`'s own docstrings.

    **Internally split around `intersect`, this pass** (`_audit/next_steps.md` §7, see
    `coils.py`'s `Intersect` docstring for why): `winding_pack_pre_intersect` builds the
    `(wp_width_r, lhs, rhs)` curves and `intersect`'s own starting guess,
    `winding_pack_post_intersect` takes the converged crossing point and finishes the
    rest. This function still calls `intersect` eagerly, exactly as before -- the split
    changes nothing about what this function computes or how it is called, only how the
    computation is organised internally, so that a `cottax` graph can also assemble the
    same three pieces (`WindingPackIntersectInputs`, `coils.py`'s `Intersect`,
    `WindingPackTotalSizePost`) as separate, driven nodes instead.

    Returns
    -------
    :
        `(b_tf_inboard_peak_symmetric, dx_tf_wp_primary_toroidal,
        dx_tf_wp_secondary_toroidal, dr_tf_wp_with_insulation, j_tf_wp,
        n_tf_coil_turns, c_tf_turn, a_tf_wp_conductor, a_tf_wp_extra_void,
        a_tf_coil_wp_turn_insulation, a_tf_wp_steel, a_tf_wp_no_insulation,
        a_tf_wp_with_insulation, fraction_area_superconductor_of_wp)`. The first entry
        (`awp_rad` in the source) and `dr_tf_wp_with_insulation` are the same value --
        the source returns and writes it separately; kept as one entry here
        (`redundant-duplicate-write`, see `_audit/schema.md`).
    """
    wp_width_r, lhs, rhs, fraction_area_superconductor_of_wp, wp_width_r_min_guess = (
        winding_pack_pre_intersect(
            r_coil_major,
            r_coil_minor,
            coilcurrent,
            n_tf_coils,
            i_tf_sc_mat,
            stella_config_a1,
            stella_config_a2,
            stella_config_wp_ratio,
            tftmp,
            tmargmin,
            b_crit_upper_nbti,
            bcritsc,
            f_a_tf_turn_cable_copper,
            fhts,
            t_crit_nbti,
            tcritsc,
            f_a_tf_turn_cable_space_extra_void,
            j_tf_wp,
            f_j_tf_wp_critical_max,
            a_tf_turn_cable_space_no_void,
            dx_tf_turn_general,
        )
    )

    wp_width_r_min = intersect(wp_width_r, lhs, wp_width_r, rhs, wp_width_r_min_guess)

    (
        b_tf_inboard_peak_symmetric,
        dx_tf_wp_primary_toroidal,
        dx_tf_wp_secondary_toroidal,
        dr_tf_wp_with_insulation,
        j_tf_wp_new,
        n_tf_coil_turns,
        c_tf_turn,
        a_tf_wp_conductor,
        a_tf_wp_extra_void,
        a_tf_coil_wp_turn_insulation,
        a_tf_wp_steel,
        a_tf_wp_no_insulation,
        a_tf_wp_with_insulation,
    ) = winding_pack_post_intersect(
        wp_width_r_min,
        r_coil_major,
        r_coil_minor,
        coilcurrent,
        n_tf_coils,
        stella_config_a1,
        stella_config_a2,
        stella_config_wp_ratio,
        f_a_tf_turn_cable_space_extra_void,
        a_tf_turn_cable_space_no_void,
        dx_tf_turn_general,
        dx_tf_wp_insulation,
        a_tf_turn_steel,
    )

    return (
        b_tf_inboard_peak_symmetric,
        dx_tf_wp_primary_toroidal,
        dx_tf_wp_secondary_toroidal,
        dr_tf_wp_with_insulation,
        j_tf_wp_new,
        n_tf_coil_turns,
        c_tf_turn,
        a_tf_wp_conductor,
        a_tf_wp_extra_void,
        a_tf_coil_wp_turn_insulation,
        a_tf_wp_steel,
        a_tf_wp_no_insulation,
        a_tf_wp_with_insulation,
        fraction_area_superconductor_of_wp,
    )


class WindingPackIntersectInputs(ExplicitFunction):
    """cottax node: the *pre*-`intersect` half of `winding_pack_total_size` -- the
    sampled `(wp_width_r, lhs, rhs)` curves `coils.py`'s `Intersect`
    (`ImplicitFunction`/`RootFind`) needs as its own `Input`s.

    This, together with `coils.py`'s `Intersect` and `WindingPackTotalSizePost` below,
    replaces the single `WindingPackTotalSize` node an earlier pass wrote (which called
    `intersect` eagerly, in the middle of its own `__call__`) -- see
    `_audit/next_steps.md` §7 and `coils.py`'s `Intersect` docstring for why splitting
    the *structural* declaration around `intersect` is worth doing even though nothing
    else in the graph needs `intersect`'s internal unknowns visible (§7's own test for
    that, unchanged): it makes the root-find's solver algorithm a first-class, swappable
    `Drive` choice, not something hardcoded inside one node's body.

    Mints `.stellarator.wp_width_r`/`.lhs`/`.rhs` at exactly the `VarPath`s `Intersect`
    reads -- `coils.md`'s own sketch of this split already proposed these names for this
    exact call site, not a fresh invention here. `i_tf_sc_mat` is a precondition, not a
    port, same treatment as `WindingPackTotalSize`'s (now `WindingPackTotalSizePost`'s)
    original.

    `fraction_area_superconductor_of_wp` (return-only, reporting) and
    `wp_width_r_min_guess` (`intersect`'s own `xin`) are both computed by
    `winding_pack_pre_intersect` but not wired through as declared `Output`s here --
    `xin` has no port in the `ImplicitFunction`/`RootFind` shape at all (see
    `Intersect`'s own docstring: a `RootFind`'s starting guess comes from whatever
    `Drive`s the block, not from a graph edge), and `fraction_area_superconductor_of_wp`
    was already discarded by the pre-split `WindingPackTotalSize` for the same
    reporting-only reason.
    """

    i_tf_sc_mat: int = eqx.field(static=True)

    wp_width_r = Output(lambda s: s.stellarator.wp_width_r)
    lhs = Output(lambda s: s.stellarator.lhs)
    rhs = Output(lambda s: s.stellarator.rhs)

    def __call__(
        self,
        r_coil_major=Input(lambda s: s.stellarator.r_coil_major),
        r_coil_minor=Input(lambda s: s.stellarator.r_coil_minor),
        coilcurrent=Input(lambda s: s.stellarator.coilcurrent),
        n_tf_coils=Input(lambda s: s.tfcoil.n_tf_coils),
        stella_config_a1=Input(lambda s: s.stellarator_config.stella_config_a1),
        stella_config_a2=Input(lambda s: s.stellarator_config.stella_config_a2),
        stella_config_wp_ratio=Input(
            lambda s: s.stellarator_config.stella_config_wp_ratio
        ),
        tftmp=Input(lambda s: s.tfcoil.tftmp),
        tmargmin=Input(lambda s: s.tfcoil.tmargmin),
        b_crit_upper_nbti=Input(lambda s: s.tfcoil.b_crit_upper_nbti),
        bcritsc=Input(lambda s: s.tfcoil.bcritsc),
        f_a_tf_turn_cable_copper=Input(lambda s: s.tfcoil.f_a_tf_turn_cable_copper),
        fhts=Input(lambda s: s.tfcoil.fhts),
        t_crit_nbti=Input(lambda s: s.tfcoil.t_crit_nbti),
        tcritsc=Input(lambda s: s.tfcoil.tcritsc),
        f_a_tf_turn_cable_space_extra_void=Input(
            lambda s: s.tfcoil.f_a_tf_turn_cable_space_extra_void
        ),
        j_tf_wp=Input(lambda s: s.tfcoil.j_tf_wp),
        f_j_tf_wp_critical_max=Input(lambda s: s.constraints.f_j_tf_wp_critical_max),
        a_tf_turn_cable_space_no_void=Input(
            lambda s: s.tfcoil.a_tf_turn_cable_space_no_void
        ),
        dx_tf_turn_general=Input(lambda s: s.tfcoil.dx_tf_turn_general),
    ):
        (
            wp_width_r,
            lhs,
            rhs,
            _fraction_area_superconductor_of_wp,
            _wp_width_r_min_guess,
        ) = winding_pack_pre_intersect(
            r_coil_major,
            r_coil_minor,
            coilcurrent,
            n_tf_coils,
            self.i_tf_sc_mat,
            stella_config_a1,
            stella_config_a2,
            stella_config_wp_ratio,
            tftmp,
            tmargmin,
            b_crit_upper_nbti,
            bcritsc,
            f_a_tf_turn_cable_copper,
            fhts,
            t_crit_nbti,
            tcritsc,
            f_a_tf_turn_cable_space_extra_void,
            j_tf_wp,
            f_j_tf_wp_critical_max,
            a_tf_turn_cable_space_no_void,
            dx_tf_turn_general,
        )
        return wp_width_r, lhs, rhs


class WindingPackTotalSizePost(ExplicitFunction):
    """cottax node: the *post*-`intersect` half of `winding_pack_total_size` --
    everything downstream of the resolved crossing point.

    Reads `.stellarator.wp_width_r_min` as a plain, ordinary `Input` -- `coils.py`'s
    `Intersect` (its `RootFind` problem, specifically) owns that `VarPath`, not this
    node, so this is a genuine cross-node edge, not a self-loop (see `Intersect`'s own
    docstring for why the pair below it is *not* a self-loop either). Together with
    `WindingPackIntersectInputs` above and `coils.py`'s `Intersect`, this is
    `WindingPackTotalSize`'s (an earlier pass's node) replacement -- see that class'
    removal note and `_audit/next_steps.md` §7 for why the split is worth doing now.

    `.tfcoil.a_tf_wp_with_insulation`/`.tfcoil.a_tf_wp_no_insulation` are minted here,
    at the same `VarPath`s the pre-split `WindingPackTotalSize` already minted them at
    (unchanged by this split) -- `coils/mass.py`'s `CoilsMass` and `coils/forces.py`'s
    `MaxForceDensity` (etc.) already declared `Input`s at exactly these two paths; this
    node is still their producer. See that removed class' own docstring (preserved
    below in this module's history/`calculate.md`) for the full reasoning, including the
    real port bug (`CoilCrossSectionalArea`'s `a_tf_wp_with_insulation` `Input`) that
    discovering this producer's correct path fixed.

    **Owns `.tfcoil.j_tf_wp`.** Unlike the pre-intersect half, nothing in
    `winding_pack_post_intersect` *reads* `j_tf_wp` (the material dispatch that does is
    entirely upstream, in `WindingPackIntersectInputs`), but it does *produce* the fresh
    `j_tf_wp_new` value -- previously discarded here because an earlier pass gave sole
    ownership of `.tfcoil.j_tf_wp` to a separate `WindingPackJTfWp` `FixedPointFunction`
    that duplicated this entire computation just to isolate that one value. That class is
    gone; this node now declares `j_tf_wp` as an ordinary `Output` instead, and
    `WindingPackIntersectInputs` reads the real `.tfcoil.j_tf_wp` as an ordinary `Input`
    -- together with `coils.py`'s `Intersect` sitting between them, this closes a genuine
    multi-node cycle (see `winding_pack_total_size`'s own docstring), not a self-loop on
    one node, so no `FixedPointFunction`/`Cut` is needed here either.
    """

    b_tf_inboard_peak_symmetric = Output(lambda s: s.tfcoil.b_tf_inboard_peak_symmetric)
    dx_tf_wp_primary_toroidal = Output(lambda s: s.tfcoil.dx_tf_wp_primary_toroidal)
    dx_tf_wp_secondary_toroidal = Output(lambda s: s.tfcoil.dx_tf_wp_secondary_toroidal)
    dr_tf_wp_with_insulation = Output(lambda s: s.tfcoil.dr_tf_wp_with_insulation)
    j_tf_wp = Output(lambda s: s.tfcoil.j_tf_wp)
    n_tf_coil_turns = Output(lambda s: s.tfcoil.n_tf_coil_turns)
    c_tf_turn = Output(lambda s: s.tfcoil.c_tf_turn)
    a_tf_wp_conductor = Output(lambda s: s.tfcoil.a_tf_wp_conductor)
    a_tf_wp_extra_void = Output(lambda s: s.tfcoil.a_tf_wp_extra_void)
    a_tf_coil_wp_turn_insulation = Output(
        lambda s: s.tfcoil.a_tf_coil_wp_turn_insulation
    )
    a_tf_wp_steel = Output(lambda s: s.tfcoil.a_tf_wp_steel)
    a_tf_wp_no_insulation = Output(lambda s: s.tfcoil.a_tf_wp_no_insulation)
    a_tf_wp_with_insulation = Output(lambda s: s.tfcoil.a_tf_wp_with_insulation)

    def __call__(
        self,
        wp_width_r_min=Input(lambda s: s.stellarator.wp_width_r_min),
        r_coil_major=Input(lambda s: s.stellarator.r_coil_major),
        r_coil_minor=Input(lambda s: s.stellarator.r_coil_minor),
        coilcurrent=Input(lambda s: s.stellarator.coilcurrent),
        n_tf_coils=Input(lambda s: s.tfcoil.n_tf_coils),
        stella_config_a1=Input(lambda s: s.stellarator_config.stella_config_a1),
        stella_config_a2=Input(lambda s: s.stellarator_config.stella_config_a2),
        stella_config_wp_ratio=Input(
            lambda s: s.stellarator_config.stella_config_wp_ratio
        ),
        f_a_tf_turn_cable_space_extra_void=Input(
            lambda s: s.tfcoil.f_a_tf_turn_cable_space_extra_void
        ),
        a_tf_turn_cable_space_no_void=Input(
            lambda s: s.tfcoil.a_tf_turn_cable_space_no_void
        ),
        dx_tf_turn_general=Input(lambda s: s.tfcoil.dx_tf_turn_general),
        dx_tf_wp_insulation=Input(lambda s: s.tfcoil.dx_tf_wp_insulation),
        a_tf_turn_steel=Input(lambda s: s.tfcoil.a_tf_turn_steel),
    ):
        (
            b_tf_inboard_peak_symmetric,
            dx_tf_wp_primary_toroidal,
            dx_tf_wp_secondary_toroidal,
            dr_tf_wp_with_insulation,
            j_tf_wp_new,
            n_tf_coil_turns,
            c_tf_turn,
            a_tf_wp_conductor,
            a_tf_wp_extra_void,
            a_tf_coil_wp_turn_insulation,
            a_tf_wp_steel,
            a_tf_wp_no_insulation,
            a_tf_wp_with_insulation,
        ) = winding_pack_post_intersect(
            wp_width_r_min,
            r_coil_major,
            r_coil_minor,
            coilcurrent,
            n_tf_coils,
            stella_config_a1,
            stella_config_a2,
            stella_config_wp_ratio,
            f_a_tf_turn_cable_space_extra_void,
            a_tf_turn_cable_space_no_void,
            dx_tf_turn_general,
            dx_tf_wp_insulation,
            a_tf_turn_steel,
        )
        return (
            b_tf_inboard_peak_symmetric,
            dx_tf_wp_primary_toroidal,
            dx_tf_wp_secondary_toroidal,
            dr_tf_wp_with_insulation,
            j_tf_wp_new,
            n_tf_coil_turns,
            c_tf_turn,
            a_tf_wp_conductor,
            a_tf_wp_extra_void,
            a_tf_coil_wp_turn_insulation,
            a_tf_wp_steel,
            a_tf_wp_no_insulation,
            a_tf_wp_with_insulation,
        )


def calculate_casing(dr_tf_nose_case):
    """Coil case thickness, radial and toroidal (m).

    Ports `calculate_casing`. Both outputs equal the single input in the source
    (docstring: "assumed to be constant until something better comes up") -- kept as
    two return values since they are two distinct `data` fields, not a redundant write
    of the same field.

    Returns
    -------
    :
        `(dr_tf_plasma_case, dx_tf_side_case_min)`.
    """
    return dr_tf_nose_case, dr_tf_nose_case


class CoilCasing(ExplicitFunction):
    dr_tf_plasma_case = Output(lambda s: s.tfcoil.dr_tf_plasma_case)
    dx_tf_side_case_min = Output(lambda s: s.tfcoil.dx_tf_side_case_min)

    def __call__(self, dr_tf_nose_case=Input(lambda s: s.tfcoil.dr_tf_nose_case)):
        return calculate_casing(dr_tf_nose_case)


def calculate_vertical_ports(
    stella_config_max_portsize_width, f_st_rmajor, f_st_n_coils
):
    """Maximal vertical port size and clearance area (m, m2).

    Ports `calculate_vertical_ports`.

    Returns
    -------
    :
        `(vporttmax, vportpmax, vportamax)`.
    """
    vporttmax = 0.4 * stella_config_max_portsize_width * f_st_rmajor / f_st_n_coils
    vportpmax = 2.0 * vporttmax
    vportamax = vporttmax * vportpmax
    return vporttmax, vportpmax, vportamax


class VerticalPorts(ExplicitFunction):
    vporttmax = Output(lambda s: s.stellarator.vporttmax)
    vportpmax = Output(lambda s: s.stellarator.vportpmax)
    vportamax = Output(lambda s: s.stellarator.vportamax)

    def __call__(
        self,
        stella_config_max_portsize_width=Input(
            lambda s: s.stellarator_config.stella_config_max_portsize_width
        ),
        f_st_rmajor=Input(lambda s: s.stellarator.f_st_rmajor),
        f_st_n_coils=Input(lambda s: s.stellarator.f_st_n_coils),
    ):
        return calculate_vertical_ports(
            stella_config_max_portsize_width, f_st_rmajor, f_st_n_coils
        )


def calculate_horizontal_ports(
    stella_config_max_portsize_width, f_st_rmajor, f_st_n_coils
):
    """Maximal horizontal port size and clearance area (m, m2).

    Ports `calculate_horizontal_ports`.

    Returns
    -------
    :
        `(hporttmax, hportpmax, hportamax)`.
    """
    hporttmax = 0.8 * stella_config_max_portsize_width * f_st_rmajor / f_st_n_coils
    hportpmax = 2.0 * hporttmax
    hportamax = hporttmax * hportpmax
    return hporttmax, hportpmax, hportamax


class HorizontalPorts(ExplicitFunction):
    hporttmax = Output(lambda s: s.stellarator.hporttmax)
    hportpmax = Output(lambda s: s.stellarator.hportpmax)
    hportamax = Output(lambda s: s.stellarator.hportamax)

    def __call__(
        self,
        stella_config_max_portsize_width=Input(
            lambda s: s.stellarator_config.stella_config_max_portsize_width
        ),
        f_st_rmajor=Input(lambda s: s.stellarator.f_st_rmajor),
        f_st_n_coils=Input(lambda s: s.stellarator.f_st_n_coils),
    ):
        return calculate_horizontal_ports(
            stella_config_max_portsize_width, f_st_rmajor, f_st_n_coils
        )


def calculate_z_tf_inside_half(
    stella_config_maximal_coil_height, r_coil_minor, stella_config_coil_rminor
):
    """TF coil inside half-height, `st_coil`'s formula.

    Extracted from `st_coil`'s own inline geometry block (`calculate.md`'s open
    question #2) into its own function -- previously left inline because it had
    exactly one call site; it now has two (`st_coil` itself, and `ZTfInsideHalf`
    below), so one shared source of truth replaced the duplicate. See `ZTfInsideHalf`'s
    own docstring for why this formula, not `build.py`'s `st_build`-derived one, is
    the one that owns `.build.z_tf_inside_half` in this port's graph.

    Parameters
    ----------
    stella_config_maximal_coil_height :
        Reference-configuration maximal coil height (m). `.stellarator_config.
        stella_config_maximal_coil_height`.
    r_coil_minor :
        Coil minor radius (m). `.stellarator.r_coil_minor`.
    stella_config_coil_rminor :
        Reference-configuration coil minor radius (m). `.stellarator_config.
        stella_config_coil_rminor`.

    Returns
    -------
    :
        `z_tf_inside_half` (m).
    """
    return (
        0.5 * stella_config_maximal_coil_height * (r_coil_minor / stella_config_coil_rminor)
    )


class ZTfInsideHalf(ExplicitFunction):
    """cottax node: `calculate_z_tf_inside_half`, owning `.build.z_tf_inside_half`.

    **Why this node, not `build.py`'s `Build`, owns this field**: real PROCESS has two
    independent writers of `.build.z_tf_inside_half` -- `st_build`'s formula (what
    `Build` computes) and `st_coil`'s formula (what this node computes, ported here).
    `stellarator.py`'s `run()` calls them in opposite order depending on the `output`
    flag; every real run ends with an `output=True` report pass that runs `st_build`
    then `st_coil`, so `st_coil`'s value is what survives into the converged answer --
    confirmed directly against a real run via the block-by-block MDA-vs-PROCESS
    comparison harness (`functional_process/mda_harness.py`), which caught `Build`
    claiming this field under the wrong (transient, `st_build`) formula. See
    `build.py`'s `calculate_build`/`Build` docstrings for the fuller account, and
    `_audit/next_steps.md` §5 for this session's other "ordering artifact" findings --
    same shape: two producers, one wins by call order, not represented structurally
    until now.
    """

    z_tf_inside_half = Output(lambda s: s.build.z_tf_inside_half)

    def __call__(
        self,
        stella_config_maximal_coil_height=Input(
            lambda s: s.stellarator_config.stella_config_maximal_coil_height
        ),
        r_coil_minor=Input(lambda s: s.stellarator.r_coil_minor),
        stella_config_coil_rminor=Input(
            lambda s: s.stellarator_config.stella_config_coil_rminor
        ),
    ):
        return calculate_z_tf_inside_half(
            stella_config_maximal_coil_height, r_coil_minor, stella_config_coil_rminor
        )


def calculate_len_tf_coil(
    stella_config_coillength, r_coil_minor, stella_config_coil_rminor, n_tf_coils
):
    """Estimated average length of one coil (m), `st_coil`'s formula
    (`process/models/stellarator/coils/calculate.py:87-91`).

    Extracted for the same reason `calculate_z_tf_inside_half` and
    `calculate_tfcryoarea` were: it now has two call sites (`st_coil` itself and
    `LenTfCoil` below) instead of one.

    Parameters
    ----------
    stella_config_coillength :
        Reference-configuration total coil length (m).
        `.stellarator_config.stella_config_coillength`.
    r_coil_minor :
        Coil minor radius (m). `.stellarator.r_coil_minor`.
    stella_config_coil_rminor :
        Reference-configuration coil minor radius (m).
        `.stellarator_config.stella_config_coil_rminor`.
    n_tf_coils :
        Number of coils. `.tfcoil.n_tf_coils`.

    Returns
    -------
    :
        `len_tf_coil` (m).
    """
    return stella_config_coillength * (r_coil_minor / stella_config_coil_rminor) / n_tf_coils


def calculate_tfcryoarea(
    stella_config_coilsurface, f_st_rmajor, r_coil_minor, stella_config_coil_rminor
):
    """Total surface area of the toroidal shells covering the coils, `st_coil`'s
    formula (`process/models/stellarator/coils/calculate.py:92-101`).

    Extracted from `st_coil`'s inline geometry block for exactly the reason
    `calculate_z_tf_inside_half` (above) was: it now has two call sites (`st_coil`
    itself and `TfCryoArea` below) rather than one, so a shared source of truth
    replaces a duplicate formula.

    The trailing `1.1` is PROCESS's own, with PROCESS's own comment: *"1.1 to scale
    it out a bit, as the shell must be bigger than WP"*.

    **Faithfulness note on `r_coil_minor`.** PROCESS's line reads
    `data.stellarator.r_coil_minor` here, while the two formulas immediately above it
    (`z_tf_inside_half`, `len_tf_coil`) use `st_coil`'s local `r_coil_minor`. They are
    the same value -- the local is bound from the field at
    `process/models/stellarator/coils/calculate.py:41` and nothing between writes it --
    so the single parameter here is faithful, not a simplification.

    Parameters
    ----------
    stella_config_coilsurface :
        Reference-configuration total coil surface area (m2).
        `.stellarator_config.stella_config_coilsurface`.
    f_st_rmajor :
        Major-radius scaling factor of this machine against the reference
        configuration. `.stellarator.f_st_rmajor`.
    r_coil_minor :
        Coil minor radius (m). `.stellarator.r_coil_minor`.
    stella_config_coil_rminor :
        Reference-configuration coil minor radius (m).
        `.stellarator_config.stella_config_coil_rminor`.

    Returns
    -------
    :
        `tfcryoarea` (m2).
    """
    return (
        stella_config_coilsurface
        * f_st_rmajor
        * (r_coil_minor / stella_config_coil_rminor)
        * 1.1
    )


class LenTfCoil(ExplicitFunction):
    """cottax node: `calculate_len_tf_coil`, owning `.tfcoil.len_tf_coil`.

    Carved out of `st_coil`'s inline geometry block like `ZTfInsideHalf` and
    `TfCryoArea`. Four registered nodes read `.tfcoil.len_tf_coil` -- `StructureMasses`,
    `PlasmaFacingCoilArea`, `CoilsMass`, `TfMagnetCostSuperconducting` -- and until this
    landed it was a **boundary input** with no producer, so all four consumed a frozen
    seed. Cold, that seed is `0.0`, which is what made
    `TfMagnetCostSuperconducting`'s `.costs.c22211`/`.c2221` come out `nan`
    (`costs.md`'s cold-start finding): every coil mass in `coils/mass.py` is
    proportional to `len_tf_coil`, so `costtfcu = uccu * m_tf_coil_copper /
    (len_tf_coil * n_tf_coil_turns)` is `0.0 / 0.0` there.

    **The stale-vs-fresh question this node was held back for, resolved.**
    `st_coil` calls `calculate_plasma_facing_coil_area` at
    `process/models/stellarator/coils/calculate.py:68`, **19 lines before** `:87` writes
    `len_tf_coil` -- so within one `Caller` round `PlasmaFacingCoilArea` reads the
    *previous* round's value, and the eager port preserves that faithfully with a
    separate `len_tf_coil_stale` parameter (`calculate.md:124`). Giving the field a
    producer switches the declared `PlasmaFacingCoilArea` node from stale to fresh, and
    the question was whether that needs modelling as a `FixedPointFunction` self-loop.

    **It does not, and the reason is structural rather than numerical.** There is no
    feedback path: `len_tf_coil`'s own inputs are two `stellarator_config` boundary
    values plus `.stellarator.r_coil_minor`/`.tfcoil.n_tf_coils`, owned by
    `StellaratorScalingFactors`, which is **not reachable from any of the four readers**
    (measured -- `_audit/boundary_inputs_audit.md` §4c (c1)). So the loop equation would
    be `x = g()` with `g` not depending on `x`: a degenerate fixed point, which
    `sand.degenerate_fixed_points` drops on sight, exactly as it already drops
    `EtaTurbineStep` and `CplifeAvail`. Modelling PROCESS's read-before-write as a cycle
    would not be more faithful -- it would add a block that is deleted for being an
    identity. The staleness is a property of PROCESS's Gauss-Seidel *schedule*, not of
    the dependency structure, and this port does not model PROCESS's round structure at
    all.

    The honest caveat: `Caller.call_models` checks idempotence on the objective and
    constraints at `rtol=1e-6`, not per field, so stale and fresh can differ
    *transiently* while upstream is still moving. They cannot differ at a converged
    point, which is what every harness here compares.
    """

    len_tf_coil = Output(lambda s: s.tfcoil.len_tf_coil)

    def __call__(
        self,
        stella_config_coillength=Input(
            lambda s: s.stellarator_config.stella_config_coillength
        ),
        r_coil_minor=Input(lambda s: s.stellarator.r_coil_minor),
        stella_config_coil_rminor=Input(
            lambda s: s.stellarator_config.stella_config_coil_rminor
        ),
        n_tf_coils=Input(lambda s: s.tfcoil.n_tf_coils),
    ):
        return calculate_len_tf_coil(
            stella_config_coillength,
            r_coil_minor,
            stella_config_coil_rminor,
            n_tf_coils,
        )


class TfCryoArea(ExplicitFunction):
    """cottax node: `calculate_tfcryoarea`, owning `.tfcoil.tfcryoarea`.

    Carved out of `st_coil`'s inline geometry block exactly as `ZTfInsideHalf` (above)
    was, and for the same structural reason: the eager `st_coil` orchestrator is
    deliberately not registered, so a formula that lives only inside it has no owner
    in the graph and its output stays a boundary input.

    **Why it was worth carving out now.** `.tfcoil.tfcryoarea` is an input of
    `power_B_thermal_cryo.py`'s cryogenic-load nodes (`CryoQLoadsStep`, via
    `Power.cryo`'s `qss` term). Registering those without this node would have traded
    two boundary inputs (`.heat_transport.helpow`,
    `.heat_transport.p_cryo_plant_electric_mw`) for one new one -- see
    `_audit/boundary_inputs_audit.md` §4c (c1)'s "sibling gap in the same three lines"
    and §7 items 4 and 7.

    **Its two siblings in the same block are deliberately left alone**:
    `.tfcoil.len_tf_coil` carries an unresolved stale-vs-fresh design decision
    (`PlasmaFacingCoilArea` reads it 19 lines before `st_coil` writes it, and the
    eager port preserves that with a separate `len_tf_coil_stale` parameter --
    `calculate.md:124`), and `min_bending_radius` has no reader at all. `tfcryoarea`
    has neither complication: nothing reads it before `st_coil` writes it.
    """

    tfcryoarea = Output(lambda s: s.tfcoil.tfcryoarea)

    def __call__(
        self,
        stella_config_coilsurface=Input(
            lambda s: s.stellarator_config.stella_config_coilsurface
        ),
        f_st_rmajor=Input(lambda s: s.stellarator.f_st_rmajor),
        r_coil_minor=Input(lambda s: s.stellarator.r_coil_minor),
        stella_config_coil_rminor=Input(
            lambda s: s.stellarator_config.stella_config_coil_rminor
        ),
    ):
        return calculate_tfcryoarea(
            stella_config_coilsurface,
            f_st_rmajor,
            r_coil_minor,
            stella_config_coil_rminor,
        )


def st_coil(
    *,
    r_coil_major,
    r_coil_minor,
    n_tf_coils,
    i_tf_sc_mat,
    dx_tf_turn_general,
    dx_tf_turn_steel,
    dx_tf_turn_insulation,
    f_st_b,
    stella_config_i0,
    f_st_rmajor,
    f_st_n_coils,
    stella_config_a1,
    stella_config_a2,
    stella_config_wp_ratio,
    tftmp,
    tmargmin,
    b_crit_upper_nbti,
    bcritsc,
    f_a_tf_turn_cable_copper,
    fhts,
    t_crit_nbti,
    tcritsc,
    f_a_tf_turn_cable_space_extra_void,
    j_tf_wp,
    f_j_tf_wp_critical_max,
    dx_tf_wp_insulation,
    dr_tf_nose_case,
    stella_config_max_portsize_width,
    stella_config_dmin,
    stella_config_coil_rmajor,
    stella_config_coil_rminor,
    stella_config_inductance,
    len_tf_coil_stale,
    stella_config_maximal_coil_height,
    stella_config_coillength,
    stella_config_coilsurface,
    stella_config_min_bend_radius,
    den_tf_coil_case,
    den_tf_wp_turn_insulation,
    a_tf_wp_coolant_channels,
    den_tf_sc_material,
    den_steel,
    rmajor,
    rminor,
    dr_fw_plasma_gap_inboard,
    dr_fw_inboard,
    dr_blkt_inboard,
    dr_shld_blkt_gap,
    dr_shld_inboard,
    dr_fw_plasma_gap_outboard,
    dr_fw_outboard,
    dr_blkt_outboard,
    dr_shld_outboard,
    b_plasma_toroidal_on_axis,
    t_tf_superconductor_quench,
    dr_vv_inboard,
    dr_vv_outboard,
    t_tf_quench_detection,
    stella_config_max_force_density,
    stella_config_wp_bmax,
    stella_config_wp_area,
    stella_config_max_force_density_mnm,
    stella_config_max_lateral_force_density,
    stella_config_max_radial_force_density,
    stella_config_centering_force_max_mn,
    stella_config_centering_force_min_mn,
    stella_config_centering_force_avg_mn,
):
    """Coil properties for a stellarator power plant, composed from the already-ported
    pieces of registry units #9-#14.

    Ports `st_coil` (`process/models/stellarator/coils/calculate.py`), minus the
    reporting call (`coils/output.py`'s `write`, confirmed pure reporting with nothing
    to port -- `output` is not a parameter here, matching the source's `output: bool`
    flag being a print-vs-don't-print switch with no effect on any returned value).

    **Real ordering bug, reproduced faithfully, not fixed**: the source calls
    `calculate_plasma_facing_coil_area(data)` -- which reads `data.tfcoil.len_tf_coil`
    -- several lines *before* it (re)computes `data.tfcoil.len_tf_coil` itself (the
    "Coil dimensions" block near the end of the source function). So `tfsai`/`tfsao`
    (the plasma-facing coil area) are always computed from whatever `len_tf_coil` held
    on entry to this call -- typically the *previous* outer solver round's value, not
    this round's -- not the fresh geometry `st_coil` itself just derived. Confirmed by
    running PROCESS's own `st_coil` directly on a fresh `DataStructure` (`len_tf_coil`
    defaults to `0.0`): `tfsai`/`tfsao` come out exactly `0.0`. This is the same class
    of bug as `winding_pack_total_size`'s `j_tf_wp` self-read -- an implicit,
    undeclared cross-call fixed point that PROCESS's outer `Caller.call_models`
    Gauss-Seidel loop (`_audit/traceability_policy.md`, `../../../CLAUDE.md` "Implicit
    cycles are hidden, not declared") happens to wash out after enough rounds, but which
    is genuinely wrong on any round where `len_tf_coil` hasn't yet stabilised (e.g. the
    very first call). `len_tf_coil_stale` is kept as its own explicit parameter here,
    separate from the freshly-computed `len_tf_coil` used everywhere else in this
    function (mass, forces), precisely so this port does not silently "fix" the bug by
    accidentally wiring the fresh value through instead.

    Returns
    -------
    :
        A large namedtuple-shaped plain tuple of every value `st_coil` computes; see the
        record's data-footprint table for the full field-by-field mapping. Not
        individually itemised in this docstring -- read the source alongside this
        function's body, which follows its call order exactly.
    """
    a_tf_turn_cable_space_no_void, a_tf_turn_steel = calculate_winding_pack_geometry(
        dx_tf_turn_general, dx_tf_turn_steel, dx_tf_turn_insulation
    )

    coilcurrent, f_st_i_total = calculate_current(
        f_st_b, stella_config_i0, f_st_rmajor, f_st_n_coils
    )

    (
        b_tf_inboard_peak_symmetric,
        dx_tf_wp_primary_toroidal,
        dx_tf_wp_secondary_toroidal,
        dr_tf_wp_with_insulation,
        j_tf_wp_new,
        n_tf_coil_turns,
        c_tf_turn,
        a_tf_wp_conductor,
        a_tf_wp_extra_void,
        a_tf_coil_wp_turn_insulation,
        a_tf_wp_steel,
        a_tf_wp_no_insulation,
        a_tf_wp_with_insulation,
        fraction_area_superconductor_of_wp,
    ) = winding_pack_total_size(
        r_coil_major,
        r_coil_minor,
        coilcurrent,
        n_tf_coils,
        i_tf_sc_mat,
        stella_config_a1,
        stella_config_a2,
        stella_config_wp_ratio,
        tftmp,
        tmargmin,
        b_crit_upper_nbti,
        bcritsc,
        f_a_tf_turn_cable_copper,
        fhts,
        t_crit_nbti,
        tcritsc,
        f_a_tf_turn_cable_space_extra_void,
        j_tf_wp,
        f_j_tf_wp_critical_max,
        a_tf_turn_cable_space_no_void,
        dx_tf_turn_general,
        dx_tf_wp_insulation,
        a_tf_turn_steel,
    )

    dr_tf_plasma_case, dx_tf_side_case_min = calculate_casing(dr_tf_nose_case)

    vporttmax, vportpmax, vportamax = calculate_vertical_ports(
        stella_config_max_portsize_width, f_st_rmajor, f_st_n_coils
    )
    hporttmax, hportpmax, hportamax = calculate_horizontal_ports(
        stella_config_max_portsize_width, f_st_rmajor, f_st_n_coils
    )

    dx_tf_inboard_out_toroidal = calculate_coil_toroidal_thickness(
        dx_tf_wp_primary_toroidal, dx_tf_side_case_min, dx_tf_wp_insulation
    )
    dr_tf_inboard = calculate_coil_radial_thickness(
        dr_tf_nose_case, dr_tf_wp_with_insulation, dr_tf_plasma_case, dx_tf_wp_insulation
    )

    a_tf_leg_outboard, a_tf_coil_inboard_case = calculate_coil_cross_sectional_area(
        a_tf_wp_with_insulation, dr_tf_inboard, dx_tf_inboard_out_toroidal
    )

    tfocrn, tficrn = calculate_coil_half_widths(dx_tf_inboard_out_toroidal)

    # See docstring: `len_tf_coil_stale`, not the fresh `len_tf_coil` computed below --
    # faithful to the source's own ordering bug.
    tfsai, tfsao = calculate_plasma_facing_coil_area(
        n_tf_coils, dx_tf_inboard_out_toroidal, len_tf_coil_stale
    )

    coilcoilgap, toroidalgap = calculate_coil_coil_toroidal_gap(
        stella_config_dmin,
        r_coil_major,
        r_coil_minor,
        stella_config_coil_rmajor,
        stella_config_coil_rminor,
        dx_tf_inboard_out_toroidal,
    )

    (
        a_tf_inboard_total,
        c_tf_total,
        j_tf_coil_full_area,
        r_b_tf_inboard_peak_symmetric,
    ) = calculate_coils_summary_variables(
        n_tf_coils,
        a_tf_leg_outboard,
        coilcurrent,
        r_coil_major,
        r_coil_minor,
        dr_tf_wp_with_insulation,
    )

    inductance = calculate_inductance(
        stella_config_inductance,
        f_st_rmajor,
        r_coil_minor,
        stella_config_coil_rminor,
        f_st_n_coils,
    )
    e_tf_magnetic_stored_total_gj = calculate_stored_magnetic_energy(
        stella_config_inductance,
        f_st_rmajor,
        r_coil_minor,
        stella_config_coil_rminor,
        f_st_n_coils,
        c_tf_total,
        n_tf_coils,
    )

    # Coil dimensions -- source's own inline geometry block (calculate.md's open
    # question #2). `z_tf_inside_half` now shares `calculate_z_tf_inside_half` with
    # `ZTfInsideHalf` (this file, above) rather than duplicating the formula -- see
    # that function's own docstring for why it has two call sites now. `tfcryoarea`
    # is now split out the same way (`calculate_tfcryoarea`/`TfCryoArea`).
    # `len_tf_coil` is now split out the same way too (`calculate_len_tf_coil`/
    # `LenTfCoil`); only `min_bending_radius` still stays inline, for want of a reader.
    z_tf_inside_half = calculate_z_tf_inside_half(
        stella_config_maximal_coil_height, r_coil_minor, stella_config_coil_rminor
    )
    len_tf_coil = calculate_len_tf_coil(
        stella_config_coillength, r_coil_minor, stella_config_coil_rminor, n_tf_coils
    )
    tfcryoarea = calculate_tfcryoarea(
        stella_config_coilsurface, f_st_rmajor, r_coil_minor, stella_config_coil_rminor
    )
    min_bending_radius = (
        stella_config_min_bend_radius
        * f_st_rmajor
        / (1.0 - dr_tf_wp_with_insulation / (2.0 * r_coil_minor))
    )

    (
        m_tf_coil_case,
        m_tf_coil_wp_insulation,
        m_tf_coil_superconductor,
        m_tf_coil_copper,
        m_tf_wp_steel_conduit,
        m_tf_coil_wp_turn_insulation,
        m_tf_coil_conductor,
        m_tf_coils_total,
    ) = calculate_coils_mass(
        a_tf_wp_with_insulation,
        a_tf_wp_no_insulation,
        len_tf_coil,
        a_tf_coil_inboard_case,
        den_tf_coil_case,
        den_tf_wp_turn_insulation,
        n_tf_coil_turns,
        a_tf_turn_cable_space_no_void,
        f_a_tf_turn_cable_space_extra_void,
        f_a_tf_turn_cable_copper,
        a_tf_wp_coolant_channels,
        den_tf_sc_material,
        a_tf_turn_steel,
        den_steel,
        a_tf_coil_wp_turn_insulation,
        n_tf_coils,
    )

    (
        f_vv_actual,
        vv_stress_quench,
        j_tf_wp_quench_heat_max,
        coppera_m2,
        v_tf_coil_dump_quench_kv,
    ) = calculate_quench_protection(
        rmajor,
        rminor,
        dr_fw_plasma_gap_inboard,
        dr_fw_inboard,
        dr_blkt_inboard,
        dr_shld_blkt_gap,
        dr_shld_inboard,
        dr_fw_plasma_gap_outboard,
        dr_fw_outboard,
        dr_blkt_outboard,
        dr_shld_outboard,
        b_plasma_toroidal_on_axis,
        c_tf_total,
        t_tf_superconductor_quench,
        dr_vv_inboard,
        dr_vv_outboard,
        t_tf_quench_detection,
        f_a_tf_turn_cable_copper,
        f_a_tf_turn_cable_space_extra_void,
        tftmp,
        a_tf_turn_cable_space_no_void,
        dx_tf_turn_general,
        a_tf_wp_conductor,
        e_tf_magnetic_stored_total_gj,
        n_tf_coils,
        c_tf_turn,
    )

    max_force_density = calculate_max_force_density(
        a_tf_wp_no_insulation,
        stella_config_max_force_density,
        f_st_i_total,
        f_st_n_coils,
        b_tf_inboard_peak_symmetric,
        stella_config_wp_bmax,
        stella_config_wp_area,
    )
    sig_tf_wp = calculate_maximum_stress(max_force_density, dr_tf_wp_with_insulation)

    max_force_density_mnm = calculate_max_force_density_mnm(
        stella_config_max_force_density_mnm,
        f_st_i_total,
        f_st_n_coils,
        b_tf_inboard_peak_symmetric,
        stella_config_wp_bmax,
    )
    max_lateral_force_density = calculate_max_lateral_force_density(
        a_tf_wp_no_insulation,
        stella_config_max_lateral_force_density,
        f_st_i_total,
        f_st_n_coils,
        b_tf_inboard_peak_symmetric,
        stella_config_wp_bmax,
        stella_config_wp_area,
    )
    max_radial_force_density = calculate_max_radial_force_density(
        a_tf_wp_no_insulation,
        stella_config_max_radial_force_density,
        f_st_i_total,
        f_st_n_coils,
        b_tf_inboard_peak_symmetric,
        stella_config_wp_bmax,
        stella_config_wp_area,
    )
    centering_force_max_mn = calculate_centering_force_max_mn(
        stella_config_centering_force_max_mn,
        f_st_i_total,
        f_st_n_coils,
        b_tf_inboard_peak_symmetric,
        stella_config_wp_bmax,
        stella_config_coillength,
        n_tf_coils,
        len_tf_coil,
    )
    centering_force_min_mn = calculate_centering_force_min_mn(
        stella_config_centering_force_min_mn,
        f_st_i_total,
        f_st_n_coils,
        b_tf_inboard_peak_symmetric,
        stella_config_wp_bmax,
        stella_config_coillength,
        n_tf_coils,
        len_tf_coil,
    )
    centering_force_avg_mn = calculate_centering_force_avg_mn(
        stella_config_centering_force_avg_mn,
        f_st_i_total,
        f_st_n_coils,
        b_tf_inboard_peak_symmetric,
        stella_config_wp_bmax,
        stella_config_coillength,
        n_tf_coils,
        len_tf_coil,
    )

    return {
        "coilcurrent": coilcurrent,
        "f_st_i_total": f_st_i_total,
        "a_tf_turn_cable_space_no_void": a_tf_turn_cable_space_no_void,
        "a_tf_turn_steel": a_tf_turn_steel,
        "b_tf_inboard_peak_symmetric": b_tf_inboard_peak_symmetric,
        "dx_tf_wp_primary_toroidal": dx_tf_wp_primary_toroidal,
        "dx_tf_wp_secondary_toroidal": dx_tf_wp_secondary_toroidal,
        "dr_tf_wp_with_insulation": dr_tf_wp_with_insulation,
        "j_tf_wp": j_tf_wp_new,
        "n_tf_coil_turns": n_tf_coil_turns,
        "c_tf_turn": c_tf_turn,
        "a_tf_wp_conductor": a_tf_wp_conductor,
        "a_tf_wp_extra_void": a_tf_wp_extra_void,
        "a_tf_coil_wp_turn_insulation": a_tf_coil_wp_turn_insulation,
        "a_tf_wp_steel": a_tf_wp_steel,
        "a_tf_wp_no_insulation": a_tf_wp_no_insulation,
        "a_tf_wp_with_insulation": a_tf_wp_with_insulation,
        "fraction_area_superconductor_of_wp": fraction_area_superconductor_of_wp,
        "dr_tf_plasma_case": dr_tf_plasma_case,
        "dx_tf_side_case_min": dx_tf_side_case_min,
        "vporttmax": vporttmax,
        "vportpmax": vportpmax,
        "vportamax": vportamax,
        "hporttmax": hporttmax,
        "hportpmax": hportpmax,
        "hportamax": hportamax,
        "dx_tf_inboard_out_toroidal": dx_tf_inboard_out_toroidal,
        "dr_tf_inboard": dr_tf_inboard,
        "dr_tf_outboard": dr_tf_inboard,
        "a_tf_leg_outboard": a_tf_leg_outboard,
        "a_tf_coil_inboard_case": a_tf_coil_inboard_case,
        "tfocrn": tfocrn,
        "tficrn": tficrn,
        "tfsai": tfsai,
        "tfsao": tfsao,
        "coilcoilgap": coilcoilgap,
        "toroidalgap": toroidalgap,
        "a_tf_inboard_total": a_tf_inboard_total,
        "c_tf_total": c_tf_total,
        "j_tf_coil_full_area": j_tf_coil_full_area,
        "r_b_tf_inboard_peak_symmetric": r_b_tf_inboard_peak_symmetric,
        "inductance": inductance,
        "e_tf_magnetic_stored_total_gj": e_tf_magnetic_stored_total_gj,
        "z_tf_inside_half": z_tf_inside_half,
        "len_tf_coil": len_tf_coil,
        "tfcryoarea": tfcryoarea,
        "min_bending_radius": min_bending_radius,
        "m_tf_coil_case": m_tf_coil_case,
        "m_tf_coil_wp_insulation": m_tf_coil_wp_insulation,
        "m_tf_coil_superconductor": m_tf_coil_superconductor,
        "m_tf_coil_copper": m_tf_coil_copper,
        "m_tf_wp_steel_conduit": m_tf_wp_steel_conduit,
        "m_tf_coil_wp_turn_insulation": m_tf_coil_wp_turn_insulation,
        "m_tf_coil_conductor": m_tf_coil_conductor,
        "m_tf_coils_total": m_tf_coils_total,
        "f_vv_actual": f_vv_actual,
        "vv_stress_quench": vv_stress_quench,
        "j_tf_wp_quench_heat_max": j_tf_wp_quench_heat_max,
        "coppera_m2": coppera_m2,
        "v_tf_coil_dump_quench_kv": v_tf_coil_dump_quench_kv,
        "max_force_density": max_force_density,
        "sig_tf_wp": sig_tf_wp,
        "max_force_density_mnm": max_force_density_mnm,
        "max_lateral_force_density": max_lateral_force_density,
        "max_radial_force_density": max_radial_force_density,
        "centering_force_max_mn": centering_force_max_mn,
        "centering_force_min_mn": centering_force_min_mn,
        "centering_force_avg_mn": centering_force_avg_mn,
    }
