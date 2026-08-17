"""Pure-functional port of the tier-1 functions in `coils/calculate.py` (registry unit #9).

Audit record: `functional_process/models/stellarator/coils/calculate.md`. That file's
`st_coil` (593 lines) is the orchestrator called directly from `Stellarator.run()`; most
of its body is 12 short, independent helper functions, 10 of which are tier-1 (pure
arithmetic, no internal solve, no calls into any other file) and ported here. The other
two -- `winding_pack_total_size` (a 200-point sampled curve fed into `intersect`, a
Newton-Raphson root-find in `coils/coils.py`) and `st_coil` itself (the orchestrator,
which also calls `coils/mass.py`, `coils/quench.py`, `coils/forces.py`,
`coils/output.py`) -- are **not** self-contained: both call into registry units #10-14,
none of which are ported yet. They stay audit-only until those land; see the record.

Every function below keeps its original name (already `calculate_*`-shaped in the
source, so nothing to rename per `naming_convention.md`) and takes exactly the fields it
reads as explicit arguments -- no `data: DataStructure` parameter anywhere, unlike the
source (`_audit/traceability_policy.md`: closing the `data` back-door is the whole
point).
"""

from cottax.interfaces.pytree_namespace_module import ExplicitFunction, Input, Output


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
        dx_tf_wp_primary_toroidal
        + 2.0 * dx_tf_side_case_min
        + 2.0 * dx_tf_wp_insulation
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
        a_tf_wp_with_insulation=Input(lambda s: s.tfcoil.dr_tf_wp_with_insulation),
        dr_tf_inboard=Input(lambda s: s.build.dr_tf_inboard),
        dx_tf_inboard_out_toroidal=Input(
            lambda s: s.tfcoil.dx_tf_inboard_out_toroidal
        ),
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
        dx_tf_inboard_out_toroidal=Input(
            lambda s: s.tfcoil.dx_tf_inboard_out_toroidal
        ),
    ):
        return calculate_coil_half_widths(dx_tf_inboard_out_toroidal)


def calculate_plasma_facing_coil_area(n_tf_coils, dx_tf_inboard_out_toroidal, len_tf_coil):
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
        dx_tf_inboard_out_toroidal=Input(
            lambda s: s.tfcoil.dx_tf_inboard_out_toroidal
        ),
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
        dx_tf_inboard_out_toroidal=Input(
            lambda s: s.tfcoil.dx_tf_inboard_out_toroidal
        ),
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
            n_tf_coils, a_tf_leg_outboard, coilcurrent, r_coil_major, r_coil_minor, awp_rad
        )


def calculate_inductance(
    stella_config_inductance, f_st_rmajor, r_coil_minor, stella_config_coil_rminor, f_st_n_coils
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
        stella_config_inductance, f_st_rmajor, r_coil_minor, stella_config_coil_rminor, f_st_n_coils
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


def calculate_vertical_ports(stella_config_max_portsize_width, f_st_rmajor, f_st_n_coils):
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


def calculate_horizontal_ports(stella_config_max_portsize_width, f_st_rmajor, f_st_n_coils):
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
