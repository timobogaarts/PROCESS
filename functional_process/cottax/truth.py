"""What PROCESS's own data says a place holds: the ground truth a solve starts from.

A place the configuration's state holds is read off it (`native.state_of`: the
configuration's values over PROCESS's defaults). A place PROCESS never stores is either
a `^stated` value (`indat.STATED_VALUES`) or one of `KNOWN_MINT_VALUES`, each an
identity read off PROCESS's own source. A copy (`^hat.x`, `^start.x`) reads the place it
copies.
"""

from cottax.pytree.crossings import read_at
from cottax.pytree.mint import unminted

from functional_process.cottax.input.indat import STATED_VALUES

KNOWN_MINT_VALUES = {
    # --- the three incoming-value places that replaced three self-loops ---
    #
    # `IonVolAvgTemperature`, `DrTfPlasmaCaseFromInput` and `DeltaEtaStep` each used to
    # read the field they own, which made each of them a `FixedPointFunction` with a
    # cut, a minted `^cond.` copy and a driver. None of the three ever read a previous
    # iterate: the value is PROCESS's *incoming* field -- the IN.DAT value for the
    # first two ("use the input directly" is what `f_temp_plasma_ion_electron <= 0`
    # means, and the second clamps the input against a geometric floor), and an inert
    # read for the third. Each now reads a free `..._in` place instead, and each
    # resolves here to the very field it used to read, which is why no value moves.
    #
    # Nothing new is asked of PROCESS: these are identities, not inversions, and no
    # IN.DAT name, `INPUT_VARIABLES` entry or `DataStructure` field is added.
    ".physics.temp_plasma_ion_vol_avg_kev_in": (
        lambda d: d.physics.temp_plasma_ion_vol_avg_kev
    ),
    ".tfcoil.dr_tf_plasma_case_in": (lambda d: d.tfcoil.dr_tf_plasma_case),
    ".power.delta_eta_in": (lambda d: d.power.delta_eta),
    # `.stellarator.coilcurrent` -- a local in `st_coil` (`process/models/stellarator/
    # coils/calculate.py:46,378`), never stored, but exactly recoverable from two real
    # fields: `calculate.py:276` writes `data.tfcoil.c_tf_total = data.tfcoil.n_tf_coils
    # * coilcurrent * 1.0e6`, so this inverts that same line. This port already
    # implements the identity in the same direction at `coils/quench.py:201`.
    ".stellarator.coilcurrent": (
        lambda d: d.tfcoil.c_tf_total / (d.tfcoil.n_tf_coils * 1.0e6)
    ),
    # `.impurity_radiation.pden_impurity_rad_total_mw` -- an `ImpurityRadiation`
    # instance attribute (`process/models/physics/impurity_radiation.py:667-668,737`),
    # never stored, but inverted from `process/models/physics/radiation_power.py:132`
    # (`pden_plasma_rad_mw = imp_rad.pden_impurity_rad_total_mw + pden_plasma_sync_mw`).
    #
    # **The `pden_impurity_core_rad_total_mw` sibling deliberately gets no entry.** Its
    # analogous inverse (`radiation_power.py:128-129`) is *not* valid: PROCESS clips
    # `.physics.pden_plasma_core_rad_mw` at zero after storing it
    # (`process/models/stellarator/stellarator.py:2153-2155`), so the stored value is
    # not recoverable back through the sum. `pden_plasma_rad_mw` itself is written
    # unclipped (`ibid.:2151`), which is what makes the entry above sound.
    ".impurity_radiation.pden_impurity_rad_total_mw": (
        lambda d: d.physics.pden_plasma_rad_mw - d.physics.pden_plasma_sync_mw
    ),
    # `.physics.pden_plasma_core_rad_mw_unclipped` / `..._outer_...` --
    # `PlasmaRadiationPowers`'s outputs *before* `st_phys`'s zero-clip
    # (`stellarator.py:2152-2159`), which `ClippedRadiationPowers` applies. PROCESS
    # overwrites the field in place, so only the post-clip value is ever stored: these
    # two mints equal the stored field **exactly whenever the clip is inactive**, and
    # are a lower bound on it otherwise -- the same discipline, and the same caveat, as
    # `.stellarator.wp_width_r_min` below. Measured on this run: core `0.0575`, outer
    # `0.0553`, both positive, so both are exact here. A run that clipped would show up
    # as a disagreement on these two mints and *not* on the real fields, which is the
    # right way round: the real fields would still be right.
    ".physics.pden_plasma_core_rad_mw_unclipped": (
        lambda d: d.physics.pden_plasma_core_rad_mw
    ),
    ".physics.pden_plasma_outer_rad_mw_unclipped": (
        lambda d: d.physics.pden_plasma_outer_rad_mw
    ),
    # --- the three entries that let the coil island come out of EXCLUDED_NODE_NAMES ---
    #
    # `.stellarator.wp_width_r_min` -- `Intersect`'s unknown, the raw crossing point of
    # `intersect(wp_width_r, lhs, wp_width_r, rhs, ...)`. A local in
    # `winding_pack_total_size` (`process/models/stellarator/coils/
    # calculate.py:452-465`), never stored. But `calculate.py:481,489` write
    # `awp_rad = wp_width_r_min` straight into the real field
    # `data.tfcoil.dr_tf_wp_with_insulation`, *after* the turn-size
    # clamp `wp_width_r_min = max(dx_tf_turn_general**2, wp_width_r_min)`
    # (`calculate.py:465`, ported at `coils/calculate.py:778`). So this identity is
    # exact whenever the clamp is inactive, and a lower bound otherwise. Measured on
    # this run: `dx_tf_turn_general**2 = 3.136e-03` against
    # `dr_tf_wp_with_insulation = 7.170e-01`, i.e. the clamp is inactive by 228x, so the
    # seed is exact here. It is in any case only a `RootFind`'s **starting guess** --
    # `Intersect` re-solves its own residual from it, and the solved answer is then
    # compared back against this same value like any other output. A clamped run would
    # start the Newton slightly low and still converge to the true crossing.
    ".stellarator.wp_width_r_min": (lambda d: d.tfcoil.dr_tf_wp_with_insulation),
    # `.tfcoil.a_tf_wp_no_insulation` / `.tfcoil.a_tf_wp_with_insulation` -- Python
    # locals in `winding_pack_total_size` (`calculate.py:494`/`:498`; the source's own
    # comment on `:493` says "(not global)"), never stored, but written *from* real
    # fields on those same lines:
    #
    #   a_tf_wp_no_insulation   = awp_tor * awp_rad
    #                           = .tfcoil.dx_tf_wp_primary_toroidal
    #                             * .tfcoil.dr_tf_wp_with_insulation
    #   a_tf_wp_with_insulation = (dr_tf_wp_with_insulation + 2*dx_tf_wp_insulation)
    #                             * (dx_tf_wp_primary_toroidal + 2*dx_tf_wp_insulation)
    #
    # -- `calculate.py:483-491` assign `awp_tor`/`awp_rad` to those two real fields
    # immediately before, so all three right-hand sides are stored values. Independently
    # confirmed on the converged run rather than read off the source alone: PROCESS's own
    # `data.tfcoil.j_tf_wp = coilcurrent * 1e6 / a_tf_wp_no_insulation`
    # (`calculate.py:499`) is reproduced to the last printed digit
    # (`3.0620392270788945e7`) by dividing `.tfcoil.c_tf_total / n_tf_coils` by the
    # reconstruction above.
    #
    # Not circular as a *comparison*: the right-hand sides are PROCESS's stored numbers,
    # so scoring the port's `a_tf_wp_*` against them checks that the port's own resolved
    # `wp_width_r_min` matches PROCESS's, which is exactly what `Intersect` is on the
    # hook for.
    ".tfcoil.a_tf_wp_no_insulation": (
        lambda d: d.tfcoil.dx_tf_wp_primary_toroidal * d.tfcoil.dr_tf_wp_with_insulation
    ),
    ".tfcoil.a_tf_wp_with_insulation": (
        lambda d: (
            (d.tfcoil.dr_tf_wp_with_insulation + 2.0 * d.tfcoil.dx_tf_wp_insulation)
            * (d.tfcoil.dx_tf_wp_primary_toroidal + 2.0 * d.tfcoil.dx_tf_wp_insulation)
        )
    ),
    # --- the two `.tokamak.build` mints (`_audit/units/models/build.md`) --------------
    #
    # Both are values PROCESS computes and then **overwrites in place**, so neither has a
    # `DataStructure` field of its own -- the same shape as the two `_unclipped`
    # radiation mints above, and the same resolution: each is reconstructible from stored
    # fields by an identity read straight off PROCESS's own source.
    #
    # `.tfcoil.dx_tf_wp_conductor_max` in particular is one edge of the build/winding-
    # pack cycle `mda.CUTS` closes.
    ".tfcoil.dx_tf_wp_conductor_max": (
        lambda d: (
            d.tfcoil.dx_tf_wp_primary_toroidal
            - 2.0 * (d.tfcoil.dx_tf_wp_insulation + d.tfcoil.dx_tf_wp_insertion_gap)
        )
    ),
    # `.build.r_tf_outboard_mid_unrippled` -- `process/models/build.py:1901-1909`, the
    # outboard stack-up *before* `:1939` may raise it to satisfy the ripple constraint.
    # Exact whenever the ripple constraint is inactive and a lower bound otherwise,
    # which is the same caveat `.stellarator.wp_width_r_min` carries and the same way
    # round: a run where the constraint bit would disagree on this mint and not on
    # `.build.r_tf_outboard_mid` itself. `13.988666666666669` on
    # `large_tokamak_eval.IN.DAT`.
    ".build.r_tf_outboard_mid_unrippled": (
        lambda d: (
            d.build.r_shld_outboard_outer
            + d.build.dr_shld_blkt_gap
            + d.build.dr_vv_outboard
            + d.build.gapomin
            + d.build.dr_shld_thermal_outboard
            + d.build.dr_tf_shld_gap
            + 0.5 * d.build.dr_tf_outboard
        )
    ),
}

KNOWN_MINT_VALUES.update(STATED_VALUES)

UNWRITTEN_BY_PROCESS = float("nan")
"""What `ground_truth` seeds for a field PROCESS itself leaves `None`."""


def ground_truth(data, var):
    """`data`'s own value at `var`: a `KNOWN_MINT_VALUES` rule where one exists, else
    the field the place `var` copies names. `None` there is PROCESS's own "never
    written" and comes back as `UNWRITTEN_BY_PROCESS`.

    Raises
    ------
    AttributeError, KeyError
        If `data` holds nothing at that place.
    """
    known = KNOWN_MINT_VALUES.get(var.spelling)
    if known is not None:
        return known(data)
    value = read_at(data, unminted(var).segments)
    return UNWRITTEN_BY_PROCESS if value is None else value
