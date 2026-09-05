"""Pure-functional port of `process/models/fw.py` (`FirstWall`, `.tokamak.first_wall`)
-- the minimal closure for `.first_wall.a_fw_total`, `.physics.p_fw_alpha_mw` and
`.physics.pflux_fw_neutron_mw`, `tokamak_boundary.md`'s three reads of this slot.

Audit record: `functional_process/_audit/units/models/fw.md`.

2026-08-27: `set_fw_geometry` (`fw.py:347-352`) joined the closure as its own node,
`FirstWallGeometry` -- `cold_boundary.md`'s producer 1, the two boundary zeros behind
7 of the cold MDA's 11 non-finite roots. See that class's docstring for why it is a
second node beside `FirstWall` rather than two more outputs of it.

`fw.py` imports `FluidProperties` (CoolProp) but reaches it **zero** times on the live
run: every CoolProp site is inside `FirstWall.fw_temp` (thermal-hydraulics), reached
only via `.fwbs.i_p_coolant_pumping == 2` (`MECHANICAL`), and
`tests/regression/input_files/large_tokamak_eval.IN.DAT:172` sets `i_p_coolant_pumping =
3`. `fw_temp` also produces nothing on this slot's boundary (`n_fw_*_channels`,
`radius_fw_channel_*_bend`, thermal quantities -- none of them read by anything in
`tokamak_boundary.md`), so it is UNPORTED, dormant rather than absent -- a second
tokamak input file with mechanical pumping would wake it, and a CoolProp wrapping
policy would be needed first (`_audit/next_steps.md` §5).

Three switches are baked into the one occupant below, each at the value live on the
reference input (none of the three appears in `large_tokamak_eval.IN.DAT`, so all three
take their PROCESS default):

- `.physics.itart` (default `0`) and `.fwbs.i_fw_blkt_vv_shape` (default `2`,
  `ELLIPTICAL_SHAPED` -- `process/models/build.py:26-30`) jointly select the elliptical
  area formula over the D-shaped one (`fw.py:58-86`).
- `.divertor.n_divertors` -- **not** the `DataStructure` field's own default of `2`
  (`divertor_variables.py:94`), but `1`, derived by `process/core/init.py:606-616` from
  `.physics.i_single_null = 1` (`large_tokamak_eval.IN.DAT:307`); see
  `divertor.md`/`structure.md` for the same correction. Selects the single-null branch
  of both `calculate_first_wall_half_height` and `apply_first_wall_coverage_factors`.
- `.physics.i_pflux_fw_neutron` (default `1`, `physics_variables.py:1006`) selects the
  `ffwal`-scaled formula for `pflux_fw_neutron_mw` over the `a_fw_total`-normalised one.

All four are read-to-branch switches under the wave-1 policy ("no switch is a static
kwarg"), so none of the port functions below takes them as a parameter at all -- each
function *is* the occupant for its live value (see `fw.md` § switches touched for the
per-switch reads-set evidence).

2026-08-27 (the double-null wave): `n_divertors` is no longer one of the unported arms.
`FirstWall` became a family of two -- `FirstWallSingleNull` (unchanged, still the
`large_tokamak_eval.IN.DAT` occupant) and `FirstWallDoubleNull` -- because the two
`n_divertors` branches this file takes differ in *reads*: a double-null first wall never
looks at `.build.z_plasma_xpoint_upper` or `.build.dz_fw_plasma_gap`, so folding the arms
into one node would declare two edges a double-null machine does not have. The other two
switches are untouched and still refused; a double-null machine that is also spherical or
D-shaped stops at `('fw_blkt_vv_shape_arm', 0)` instead.

2026-08-27 (the D-shaped wave): that last sentence no longer holds for the *double-null*
D-shaped machine. `spherical_tokamak_eval.IN.DAT` and `st_regression.IN.DAT` are both
D-shaped (`i_fw_blkt_vv_shape = 1`) **and** spherical (`itart = 1`) **and** double-null
(`i_single_null = 0`), so `FirstWallDShapedDoubleNull` joins the family and the shape
refusal moves off that cell.

**The shape and the divertor count are a genuine product here, and this is one of only
two slots where they are.** `FirstWall.run()` (`process/models/fw.py:44-149`) branches on
`n_divertors` twice -- once for the half-height (`:46-56`) and once for the coverage
factors (`:103-109`) -- with the shape branch (`:58-86`) sitting *between* them. The two
predicates are independent, but because this port keeps `run()` as **one** composite node
rather than three, the occupant grid is 2 (shape) x 2 (divertor count) = four cells, of
which three are written:

| | single null | double null |
|---|---|---|
| **elliptical** | `FirstWallSingleNull` | `FirstWallDoubleNull` |
| **D-shaped** | UNPORTED, `('first_wall_arm', -2)` | `FirstWallDShapedDoubleNull` |

`models/vacuum/vacuum.py` pays the same product for the same reason;
`blankets/blanket_library.py` and `models/shield.py` do **not**, because wave 1 had
already split their `run()`s into one slot per branch, so each of their slots is keyed on
exactly one predicate. The difference is in the decomposition, not in PROCESS.

The unwritten cell is refused rather than written because no input file in this
repository selects it -- the wave's reachability-first discipline. Its every
ingredient exists (`calculate_dshaped_first_wall_areas` below, and the single-null
half-height and coverage
arms), so it is one composite function and one class away, with no new arithmetic;
that is what `('first_wall_arm', -2)` now says.
"""

from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.models.fw import (
    apply_first_wall_coverage_factors,  # noqa: F401
    apply_first_wall_coverage_factors_double_null,  # noqa: F401
    calculate_dshaped_first_wall_areas,  # noqa: F401
    calculate_elliptical_first_wall_areas,  # noqa: F401
    calculate_first_wall_half_height,  # noqa: F401
    calculate_first_wall_half_height_double_null,  # noqa: F401
    calculate_first_wall_outputs,
    calculate_first_wall_outputs_double_null,
    calculate_first_wall_outputs_dshaped_double_null,
    calculate_p_fw_alpha_mw,  # noqa: F401
    calculate_pflux_fw_neutron_mw_ffwal,  # noqa: F401
    calculate_radiated_wall_load_scaled_plasma_surface,
    set_fw_geometry,
)
from functional_process.models.engineering.ivc_functions import (
    dshellarea,  # noqa: F401
    eshellarea,  # noqa: F401
)
from functional_process.paths import (
    build,
    constraints,
    divertor,
    first_wall,
    fwbs,
    physics,
)


class FirstWall(ExplicitFunction):
    """The family that occupies `.tokamak.first_wall`: one occupant per cell of the
    shape x divertor-count grid, all at `.physics.i_pflux_fw_neutron == 1`.

    Each occupant owns all three of `.tokamak.first_wall`'s declared boundary outputs
    (`.first_wall.a_fw_total`, `.physics.p_fw_alpha_mw`, `.physics.pflux_fw_neutron_mw`)
    plus `a_fw_inboard`/`a_fw_outboard`, siblings of `a_fw_total` from the same source
    function and read elsewhere in `process/` (`blankets/dcll.py`, `blankets/hcpb.py`,
    `stellarator.py`).

    Three of the grid's four cells are written -- see the module docstring's table.
    `FirstWallSingleNull` and `FirstWallDoubleNull` are the **elliptical** pair (their
    names predate the shape becoming a family axis and are left as they are, since
    `indat.py` and two audit records name them); `FirstWallDShapedDoubleNull` is the
    D-shaped one. D-shaped single null refuses at `('first_wall_arm', -2)`.

    `.physics.i_pflux_fw_neutron` is still **not** a family axis: its other arm divides
    by a field this same occupant owns and would need a fixed point, not another class
    (`('first_wall_arm', -3)`).
    """


class FirstWallSingleNull(FirstWall):
    """cottax node: `.tokamak.first_wall` at `.divertor.n_divertors == 1`, elliptical --
    the combination live on `large_tokamak_eval.IN.DAT` (see module docstring). Thin wrap
    of `calculate_first_wall_outputs`, no arithmetic of its own.
    """

    a_fw_inboard = OutputInto(first_wall)
    a_fw_outboard = OutputInto(first_wall)
    a_fw_total = OutputInto(first_wall)
    p_fw_alpha_mw = OutputInto(physics)
    pflux_fw_neutron_mw = OutputInto(physics)

    def __call__(
        self,
        z_plasma_xpoint_lower=From(build),
        dz_xpoint_divertor=From(build),
        dz_divertor=From(divertor),
        dz_blkt_upper=From(build),
        z_plasma_xpoint_upper=From(build),
        dz_fw_plasma_gap=From(build),
        dr_fw_inboard=From(build),
        dr_fw_outboard=From(build),
        rmajor=From(physics),
        rminor=From(physics),
        triang=From(physics),
        dr_fw_plasma_gap_inboard=From(build),
        dr_fw_plasma_gap_outboard=From(build),
        f_ster_div_single=From(fwbs),
        f_a_fw_outboard_hcd=From(fwbs),
        p_alpha_total_mw=From(physics),
        f_p_alpha_plasma_deposited=From(physics),
        ffwal=From(physics),
        pflux_plasma_surface_neutron_avg_mw=From(physics),
    ):
        return calculate_first_wall_outputs(
            z_plasma_xpoint_lower=z_plasma_xpoint_lower,
            dz_xpoint_divertor=dz_xpoint_divertor,
            dz_divertor=dz_divertor,
            dz_blkt_upper=dz_blkt_upper,
            z_plasma_xpoint_upper=z_plasma_xpoint_upper,
            dz_fw_plasma_gap=dz_fw_plasma_gap,
            dr_fw_inboard=dr_fw_inboard,
            dr_fw_outboard=dr_fw_outboard,
            rmajor=rmajor,
            rminor=rminor,
            triang=triang,
            dr_fw_plasma_gap_inboard=dr_fw_plasma_gap_inboard,
            dr_fw_plasma_gap_outboard=dr_fw_plasma_gap_outboard,
            f_ster_div_single=f_ster_div_single,
            f_a_fw_outboard_hcd=f_a_fw_outboard_hcd,
            p_alpha_total_mw=p_alpha_total_mw,
            f_p_alpha_plasma_deposited=f_p_alpha_plasma_deposited,
            ffwal=ffwal,
            pflux_plasma_surface_neutron_avg_mw=pflux_plasma_surface_neutron_avg_mw,
        )


class FirstWallDoubleNull(FirstWall):
    """cottax node: `.tokamak.first_wall` at `.divertor.n_divertors == 2`, elliptical.
    Thin wrap of `calculate_first_wall_outputs_double_null`.

    Owns the same five fields as its single-null sibling and reads two fewer:
    `.build.z_plasma_xpoint_upper` and `.build.dz_fw_plasma_gap` are absent from the
    signature below, which is the structural difference the split exists to record.

    Written for the two ST files, which turned out to be D-shaped as well and so select
    `FirstWallDShapedDoubleNull` instead; no input file in this repository currently
    reaches this cell. It stays registered: `n_divertors == 2` with an elliptical
    cross-section is an ordinary PROCESS configuration and the occupant is
    harness-tested against a real `FirstWall.run()`.
    """

    a_fw_inboard = OutputInto(first_wall)
    a_fw_outboard = OutputInto(first_wall)
    a_fw_total = OutputInto(first_wall)
    p_fw_alpha_mw = OutputInto(physics)
    pflux_fw_neutron_mw = OutputInto(physics)

    def __call__(
        self,
        z_plasma_xpoint_lower=From(build),
        dz_xpoint_divertor=From(build),
        dz_divertor=From(divertor),
        dz_blkt_upper=From(build),
        dr_fw_inboard=From(build),
        dr_fw_outboard=From(build),
        rmajor=From(physics),
        rminor=From(physics),
        triang=From(physics),
        dr_fw_plasma_gap_inboard=From(build),
        dr_fw_plasma_gap_outboard=From(build),
        f_ster_div_single=From(fwbs),
        f_a_fw_outboard_hcd=From(fwbs),
        p_alpha_total_mw=From(physics),
        f_p_alpha_plasma_deposited=From(physics),
        ffwal=From(physics),
        pflux_plasma_surface_neutron_avg_mw=From(physics),
    ):
        return calculate_first_wall_outputs_double_null(
            z_plasma_xpoint_lower=z_plasma_xpoint_lower,
            dz_xpoint_divertor=dz_xpoint_divertor,
            dz_divertor=dz_divertor,
            dz_blkt_upper=dz_blkt_upper,
            dr_fw_inboard=dr_fw_inboard,
            dr_fw_outboard=dr_fw_outboard,
            rmajor=rmajor,
            rminor=rminor,
            triang=triang,
            dr_fw_plasma_gap_inboard=dr_fw_plasma_gap_inboard,
            dr_fw_plasma_gap_outboard=dr_fw_plasma_gap_outboard,
            f_ster_div_single=f_ster_div_single,
            f_a_fw_outboard_hcd=f_a_fw_outboard_hcd,
            p_alpha_total_mw=p_alpha_total_mw,
            f_p_alpha_plasma_deposited=f_p_alpha_plasma_deposited,
            ffwal=ffwal,
            pflux_plasma_surface_neutron_avg_mw=pflux_plasma_surface_neutron_avg_mw,
        )


class FirstWallDShapedDoubleNull(FirstWall):
    """cottax node: `.tokamak.first_wall` at `.divertor.n_divertors == 2` **and** the
    D-shaped shape arm -- the configuration live on `spherical_tokamak_eval.IN.DAT` and
    `st_regression.IN.DAT` (`i_single_null = 0`; `itart = 1` and
    `i_fw_blkt_vv_shape = 1`, either of which alone selects the D-shaped arm). Thin wrap
    of `calculate_first_wall_outputs_dshaped_double_null`.

    Owns the same five fields as the other two occupants and reads **three** fewer than
    the elliptical single-null one: no `.physics.triang` (the D-shaped area formula does
    not use it) on top of the double-null arm's two.
    """

    a_fw_inboard = OutputInto(first_wall)
    a_fw_outboard = OutputInto(first_wall)
    a_fw_total = OutputInto(first_wall)
    p_fw_alpha_mw = OutputInto(physics)
    pflux_fw_neutron_mw = OutputInto(physics)

    def __call__(
        self,
        z_plasma_xpoint_lower=From(build),
        dz_xpoint_divertor=From(build),
        dz_divertor=From(divertor),
        dz_blkt_upper=From(build),
        dr_fw_inboard=From(build),
        dr_fw_outboard=From(build),
        rmajor=From(physics),
        rminor=From(physics),
        dr_fw_plasma_gap_inboard=From(build),
        dr_fw_plasma_gap_outboard=From(build),
        f_ster_div_single=From(fwbs),
        f_a_fw_outboard_hcd=From(fwbs),
        p_alpha_total_mw=From(physics),
        f_p_alpha_plasma_deposited=From(physics),
        ffwal=From(physics),
        pflux_plasma_surface_neutron_avg_mw=From(physics),
    ):
        return calculate_first_wall_outputs_dshaped_double_null(
            z_plasma_xpoint_lower=z_plasma_xpoint_lower,
            dz_xpoint_divertor=dz_xpoint_divertor,
            dz_divertor=dz_divertor,
            dz_blkt_upper=dz_blkt_upper,
            dr_fw_inboard=dr_fw_inboard,
            dr_fw_outboard=dr_fw_outboard,
            rmajor=rmajor,
            rminor=rminor,
            dr_fw_plasma_gap_inboard=dr_fw_plasma_gap_inboard,
            dr_fw_plasma_gap_outboard=dr_fw_plasma_gap_outboard,
            f_ster_div_single=f_ster_div_single,
            f_a_fw_outboard_hcd=f_a_fw_outboard_hcd,
            p_alpha_total_mw=p_alpha_total_mw,
            f_p_alpha_plasma_deposited=f_p_alpha_plasma_deposited,
            ffwal=ffwal,
            pflux_plasma_surface_neutron_avg_mw=pflux_plasma_surface_neutron_avg_mw,
        )


class FirstWallGeometry(ExplicitFunction):
    """cottax node: `.tokamak.first_wall_geometry`. Ports `FirstWall.set_fw_geometry`
    (`process/models/fw.py:347-352`); no switch.

    A separate node rather than two more outputs of `FirstWall`, because `FirstWall`
    *reads* `dr_fw_inboard`/`dr_fw_outboard` (for the half-height) and a node must not
    read what it owns. The split also preserves a measured PROCESS quirk without
    reproducing it: `FirstWall.run()` computes the half-height from the *entering*
    `dr_fw_*` values and only then calls `set_fw_geometry` (`fw.py:46-56` before
    `:110`), so PROCESS's first cold pass sees `0.0` where every later pass -- and this
    graph, which reads the produced value -- sees `2*0.006 + 2*0.003 = 0.018`. The two
    fields are constants of two run inputs, so the fixed point is reached after one
    PROCESS pass and the port's fresh read *is* the converged value; no cycle exists
    (this node reads only `.fwbs` inputs) and none is created (measured,
    `Blocking.scc` on both reference machines, 2026-08-27).
    """

    dr_fw_inboard = OutputInto(build)
    dr_fw_outboard = OutputInto(build)

    def __call__(
        self,
        radius_fw_channel=From(fwbs),
        dr_fw_wall=From(fwbs),
    ):
        return set_fw_geometry(
            radius_fw_channel=radius_fw_channel,
            dr_fw_wall=dr_fw_wall,
        )


class RadiatedWallLoad(ExplicitFunction):
    """cottax node: `calculate_radiated_wall_load_scaled_plasma_surface`, ports
    declared. `.tokamak.radiated_wall_load`.

    A node of its own rather than two more outputs of `FirstWall`, on exactly
    `FirstWallGeometry`'s grounds and with a sharper consequence: the *other*
    `i_pflux_fw_neutron` arm of these same two lines divides by
    `.first_wall.a_fw_total`, which `FirstWall` owns, so folding them in would make the
    live arm's node read a field the dead arm makes it own. Split, the two arms are two
    occupants of one slot and neither reads what it owns.

    **`.constraints.pflux_fw_rad_max_mw` was a frozen `0.0` on both tracked spherical
    tokamaks** (`optimise_design.md` §26.2, rank 5), where PROCESS reads `0.36324`
    (`st_regression`) and `0.49896` (`spherical_tokamak_eval`) against constraint 67's
    bound of `1.2`. Unlike constraint 56's path this one is **not** binding at
    PROCESS's own answer, so what the freeze cost was a live Jacobian row and an
    honest report, not a wrong optimum -- which is why §26.3 ranks it last of the five
    live rows and why §29 predicts, and measures, a smaller move.

    Unswitched *in this port*: see the ported function's docstring for why the
    `i_pflux_fw_neutron` refusal lives in `_first_wall_arm` and is not repeated here.
    """

    pflux_fw_rad_mw = OutputInto(physics)
    pflux_fw_rad_max_mw = OutputInto(constraints)

    def __call__(
        self,
        ffwal=From(physics),
        p_plasma_rad_mw=From(physics),
        a_plasma_surface=From(physics),
        f_fw_rad_max=From(constraints),
    ):
        return calculate_radiated_wall_load_scaled_plasma_surface(
            ffwal=ffwal,
            p_plasma_rad_mw=p_plasma_rad_mw,
            a_plasma_surface=a_plasma_surface,
            f_fw_rad_max=f_fw_rad_max,
        )
