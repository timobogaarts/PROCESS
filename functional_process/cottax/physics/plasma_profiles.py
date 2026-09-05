"""Pure-functional port of `process/models/physics/plasma_profiles.py`.

Registry unit #12.

Audit record: `functional_process/_audit/units/models/physics/plasma_profiles.md`. Read
its **scope correction** first: `PlasmaProfile.run()` is not portable end-to-end from
this file alone: both of its branches call `neprofile.run()`/`teprofile.run()` for
effect and then read four `.physics` fields those calls wrote. Those classes live in
`process/models/physics/profiles.py` (558 LOC), which is **not in the unit registry** —
the same class of scoping miss as `coils/` and `rether`.

What *is* portable, and is ported here, is everything downstream of the profile arrays:
five tier-1 functions covering all the arithmetic in the file. The profile arrays
(`profile_x`/`profile_y`) arrive as explicit array arguments rather than being read off
an injected object -- exactly the `data`-back-door closure the audit exists to force.

**All five now have `cottax` nodes**, and both `i_plasma_pedestal` arms of the
topology switch are covered: `ParabolicProfileValues` (`i_plasma_pedestal == 0`) and
`PedestalProfileValues` (`i_plasma_pedestal == 1`) wrap `calculate_parabolic_profile_
values` and `calculate_pedestal_profile_values` respectively; `IonVolAvgTemperature`
(a `FixedPointFunction`, since the field it owns is conditionally read as well) and
`ProfileFactors`/`ParabolicGradientLengths` cover the rest. The "static kwarg on
`density_limits.EcrhDensityLimit`" reconciliation this docstring used to say was
unresolved (record § open question 1) is settled in practice -- see
`ParabolicProfileValues`'s own docstring -- and `PedestalProfileValues` closes the last
gap `_audit/tokamak_boundary.md` § "The four that are a shared subsystem's gap" found:
a tokamak is the first machine to select the pedestal arm, and until this node existed
four of its fields had no producer at all even though the underlying function was
already ported and tested.
"""

from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    FixedPointFunction,
    From,
    OutputInto,
)

from functional_process.cottax.stated import StatesValues
from functional_process.paths import divertor, physics
from functional_process.models.physics.plasma_profiles import (
    calculate_ion_vol_avg_temperature,
    calculate_parabolic_gradient_lengths,
    calculate_parabolic_profile_values,
    calculate_pedestal_profile_values,
    calculate_profile_factors,
    lmode_profile_reset,
)

__all__ = [
    "lmode_profile_reset",
]


class ProfileFactors(ExplicitFunction):
    """cottax node: `calculate_profile_factors`, unchanged, ports declared.

    In `COMMON`, not under a switch: it runs in both `i_plasma_pedestal` branches, and
    its on-axis inputs come from `NeProfile`/`TeProfile` in both.

    The four `.physics.pres_*_profile` outputs are arrays of
    `n_plasma_profile_elements`, as are the two `profile_y` inputs.

    **Two minted `VarPath`s**, same situation and same justification as
    `coils/calculate.py`'s `.stellarator.coilcurrent`: the density and temperature
    profile arrays have **no PROCESS storage location**. They live only as
    `neprofile.profile_y` / `teprofile.profile_y`, attributes of the two injected
    `Profile` objects, and are read straight off those objects here. Verified absent from
    `data_structure/physics_variables.py`. They are real graph edges -- produced by the
    (unaudited, unported) `profiles.py` unit and consumed here -- so without minting them
    this node could not source its two largest inputs at all:

        .physics.nd_plasma_electron_profile        <- neprofile.profile_y
        .physics.temp_plasma_electron_profile_kev  <- teprofile.profile_y

    Minted into `.physics` rather than a new root because that is where every other
    profile-shaped array in this file already lives -- `pres_plasma_electron_profile`,
    which this node *owns*, is a real `physics_variables.py` field of exactly the same
    shape. The names follow `standards.md`'s `<type>_<system>_<description>_<units>`
    scheme so they read as siblings of it rather than as imports.
    """

    pres_plasma_thermal_on_axis = OutputInto(physics)
    pres_plasma_electron_profile = OutputInto(physics)
    pres_plasma_ion_total_profile = OutputInto(physics)
    pres_plasma_thermal_total_profile = OutputInto(physics)
    pres_plasma_fuel_profile = OutputInto(physics)
    alphap = OutputInto(physics)
    pres_plasma_thermal_vol_avg = OutputInto(physics)
    j_plasma_on_axis = OutputInto(physics)

    def __call__(
        self,
        nd_plasma_electron_profile=From(physics),
        temp_plasma_electron_profile_kev=From(physics),
        nd_plasma_electron_on_axis=From(physics),
        temp_plasma_electron_on_axis_kev=From(physics),
        nd_plasma_ions_on_axis=From(physics),
        temp_plasma_ion_on_axis_kev=From(physics),
        nd_plasma_ions_total_vol_avg=From(physics),
        nd_plasma_electrons_vol_avg=From(physics),
        nd_plasma_fuel_ions_vol_avg=From(physics),
        f_temp_plasma_ion_electron=From(physics),
        temp_plasma_electron_density_weighted_kev=From(physics),
        temp_plasma_ion_density_weighted_kev=From(physics),
        alphan=From(physics),
        alphat=From(physics),
        alphaj=From(physics),
        plasma_current=From(physics),
        a_plasma_poloidal=From(physics),
    ):
        return calculate_profile_factors(
            nd_plasma_electron_profile,
            temp_plasma_electron_profile_kev,
            nd_plasma_electron_on_axis,
            temp_plasma_electron_on_axis_kev,
            nd_plasma_ions_on_axis,
            temp_plasma_ion_on_axis_kev,
            nd_plasma_ions_total_vol_avg,
            nd_plasma_electrons_vol_avg,
            nd_plasma_fuel_ions_vol_avg,
            f_temp_plasma_ion_electron,
            temp_plasma_electron_density_weighted_kev,
            temp_plasma_ion_density_weighted_kev,
            alphan,
            alphat,
            alphaj,
            plasma_current,
            a_plasma_poloidal,
        )


class ParabolicGradientLengths(ExplicitFunction):
    """cottax node: `calculate_parabolic_gradient_lengths`, unchanged, ports declared.

    Parabolic-only, but **not** an `Alternative`: the pedestal branch has no counterpart
    node that owns `.physics.gradient_length_*`, so there is nothing for it to be
    mutually exclusive with. Registering it in `COMMON` would be wrong for a different
    reason -- it would claim these fields are produced in the pedestal configuration too.
    Left out of `total_process.py` until `i_plasma_pedestal`'s two roles are reconciled
    (record § open questions 1), which is the same blocker as the two branch arms.
    """

    gradient_length_te = OutputInto(physics)
    gradient_length_ne = OutputInto(physics)

    def __call__(
        self,
        alphat=From(physics),
        alphan=From(physics),
        temp_plasma_electron_on_axis_kev=From(physics),
        nd_plasma_electron_on_axis=From(physics),
        rminor=From(physics),
    ):
        return calculate_parabolic_gradient_lengths(
            alphat,
            alphan,
            temp_plasma_electron_on_axis_kev,
            nd_plasma_electron_on_axis,
            rminor,
        )


class IonVolAvgTemperature(FixedPointFunction):
    """cottax node: `calculate_ion_vol_avg_temperature`, as a fixed point.

    **Why a `FixedPointFunction` and not an `ExplicitFunction`.** PROCESS writes
    `.physics.temp_plasma_ion_vol_avg_kev` **only** when
    `f_temp_plasma_ion_electron > 0`, and otherwise leaves the input value in place
    (`process/models/physics/plasma_profiles.py:64-68`). The ported pure function is
    faithful to that -- it takes the incumbent value as an argument -- which makes the
    field a read *and* a write of the same node, and `to_graph` refuses that outright.
    This is `plasma_profiles.md`'s own `conditional-ownership-by-data` classification,
    and it gets the same treatment as the six `conditional-ownership` fields in
    `thermal_cryo.py`: the node owns the field and reads the minted copy, so the
    "keep the incumbent" arm is a **fixed point** (`u = g(u)`, converging in one Picard
    step from anywhere) rather than a self-loop.

    That shape is not a formality here, it carries the switch correctly in both
    directions:

    - `f_temp_plasma_ion_electron > 0` -- `g` does not depend on the unknown at all, so
      the residual `g(u) - u` has derivative `-1` and the fixed point is well-posed and
      SAND-solvable.
    - `f_temp_plasma_ion_electron <= 0` -- `g` is the exact identity, so the residual is
      structurally zero. `functional_process.sand.degenerate_fixed_points` detects that
      by differentiation and **drops the problem**, which reverts
      `.physics.temp_plasma_ion_vol_avg_kev` to an ordinary boundary input -- exactly
      PROCESS's "use the input value" semantics, recovered from structure rather than
      declared.

    Registered in `COMMON`, not under `.physics.i_plasma_pedestal`: PROCESS writes this
    field in `parameterise_plasma` **before** the branch, so it runs in both arms.

    **What registering it fixes.** Before this node existed
    `.physics.temp_plasma_ion_vol_avg_kev` was a boundary input, so ion temperature was
    structurally disconnected from `.physics.temp_plasma_electron_vol_avg_kev` --
    iteration variable 4. Every derivative with respect to that variable was therefore
    wrong by construction while every *value* was right, which is precisely the defect
    class only a gradient comparison can find (`_audit/optimise_design.md` §5.2's
    "x4 column").
    """

    temp_plasma_ion_vol_avg_kev = OutputInto(physics)

    def step(
        self,
        f_temp_plasma_ion_electron=From(physics),
        temp_plasma_electron_vol_avg_kev=From(physics),
        temp_plasma_ion_vol_avg_kev=From(physics),
    ):
        return calculate_ion_vol_avg_temperature(
            f_temp_plasma_ion_electron,
            temp_plasma_electron_vol_avg_kev,
            temp_plasma_ion_vol_avg_kev,
        )


class ParabolicProfileValues(ExplicitFunction):
    """cottax node: `calculate_parabolic_profile_values`, the `i_plasma_pedestal == 0`
    arm of `parameterise_plasma`'s line-average/density-weighted tail.

    An `Alternative` under `.physics.i_plasma_pedestal`, alongside
    `ParabolicTemperatureProfile`/`ParabolicOnAxisTemperatures`/
    `ParabolicGradientLengths`, which are already there. `plasma_profiles.md`'s "cottax
    node" section deferred this one pending open question 1 ("`i_plasma_pedestal` holds
    two different switch roles across two units"); that question is **settled in
    practice** -- `total_process.py`'s value-0 arm already co-locates
    `EcrhDensityLimit(i_plasma_pedestal=0)` with the parabolic profile nodes, so there is
    exactly one place the value is written and nothing left to reconcile.

    **`calculate_pedestal_profile_values` now has a node too, `PedestalProfileValues`
    below** -- this class's counterpart under `i_plasma_pedestal == 1`. `profiles.py` (the
    unit it needed) has since been ported and registered, closing the sequencing
    constraint that used to block it.

    **What registering this node fixes**, together with `IonVolAvgTemperature` above: the two
    density-weighted temperatures were boundary inputs, so `.physics.beta_total_vol_avg`
    (constraint 24) and every fusion-reactivity quantity had **zero** sensitivity to
    iteration variable 4. See `IonVolAvgTemperature`'s docstring.
    """

    f_temp_plasma_electron_density_vol_avg = OutputInto(physics)
    nd_plasma_electron_line = OutputInto(physics)
    temp_plasma_electron_line_avg_kev = OutputInto(physics)
    temp_plasma_electron_density_weighted_kev = OutputInto(physics)
    temp_plasma_ion_density_weighted_kev = OutputInto(physics)

    def __call__(
        self,
        alphan=From(physics),
        alphat=From(physics),
        nd_plasma_electrons_vol_avg=From(physics),
        temp_plasma_electron_vol_avg_kev=From(physics),
        temp_plasma_ion_vol_avg_kev=From(physics),
    ):
        return calculate_parabolic_profile_values(
            alphan,
            alphat,
            nd_plasma_electrons_vol_avg,
            temp_plasma_electron_vol_avg_kev,
            temp_plasma_ion_vol_avg_kev,
        )


class LModeProfileReset(StatesValues):
    """cottax node: `lmode_profile_reset`, the `i_plasma_pedestal == 0` arm's
    input-validation reset, as the producer it always was.

    An `Alternative` under `.physics.i_plasma_pedestal` alongside
    `ParabolicProfileValues` and the rest of the parabolic arm. This closes
    `plasma_profiles.md`'s open question 2 and `profiles.md`'s open question about
    `DensityProfile`: the answer is **not** graph-assembly-time input coercion, it is an
    ordinary node, because PROCESS performs the reset inside the pipeline and the result
    is a plain post-condition of selecting the parabolic arm.

    **No inputs of its own, by construction.** `lmode_profile_reset` ignores its
    arguments, so declaring the seven fields as `From`s as well as `OutputInto`s would be
    a seven-way self-loop stating a dependence the computation does not have.

    **The seven constants are `stated`, not returned from the body** (`models/stated.py`,
    `_audit/optimise_design.md` §34). This node was not in §28.1's fourteen -- it landed
    after that census and holds no field, so the array ban does not reach it -- but it is
    the same defect: seven literals handed back by an input-less body are seven
    compile-time constants, and §25's Arm C measured XLA deleting the readers of one.
    `indat.STATED_VALUES` calls `lmode_profile_reset` for them, so the unit is still the
    source of the numbers.

    **What registering it fixes -- measured, on the reference stellarator run.** Before
    this node the four of these seven fields the graph touches were unowned boundary
    inputs, so a cold solve carried the input file's own `nd_plasma_pedestal_electron =
    4e19` / `nd_plasma_separatrix_electron = 3e19` into `profiles.DensityProfile`, whose
    single formula only degenerates to the parabolic profile once they are zero. A warm
    solve seeded from PROCESS's converged `DataStructure` got the reset for free (the
    fields are already `0` there) and a cold one did not, so the two were solving
    **different problems**: SAND finished at `objf` 1.217757 warm and 1.215038 cold, a
    gap read for a while as evidence of several local minima. With this node the cold
    solve lands on the warm one's answer to nine digits and the median distance to
    PROCESS's converged `x` falls from `1.4e-02` to `8.6e-03`.
    """

    radius_plasma_pedestal_temp_norm = OutputInto(physics)
    radius_plasma_pedestal_density_norm = OutputInto(physics)
    temp_plasma_pedestal_kev = OutputInto(physics)
    temp_plasma_separatrix_kev = OutputInto(physics)
    nd_plasma_pedestal_electron = OutputInto(physics)
    nd_plasma_separatrix_electron = OutputInto(physics)
    tbeta = OutputInto(physics)


class PedestalProfileValues(ExplicitFunction):
    """cottax node: `calculate_pedestal_profile_values`, the `i_plasma_pedestal == 1`
    arm of `parameterise_plasma`'s line-average/density-weighted tail --
    `ParabolicProfileValues`' pedestal-arm counterpart.

    An `Alternative` under `.physics.i_plasma_pedestal`, alongside
    `PedestalTemperatureProfile`/`PedestalOnAxisDensities`/`PedestalOnAxisTemperatures`
    (`functional_process/cottax/physics/profiles.py`), the other three nodes of
    `ProfileParameterisationPedestal`. `calculate_pedestal_profile_values` itself has
    been ported and tested since this unit's earliest audit pass (`TestPedestalProfileValues`
    in the test module); what was missing was only this wrapper. Found by
    `_audit/tokamak_boundary.md` § "The four that are a shared subsystem's gap": a
    conventional tokamak is the first machine this port assembles that selects
    `i_plasma_pedestal == 1`
    (`tests/regression/input_files/large_tokamak_eval.IN.DAT:291`), and a slot with a
    registered occupant on both switch arms can still have one arm that produces less
    than the other -- only the boundary saw it, because every *value* test for the
    underlying function already passed.

    **Six outputs, not four.** `tokamak_boundary.md`'s table lists only
    `f_temp_plasma_electron_density_vol_avg`, `nd_plasma_electron_line`,
    `temp_plasma_electron_density_weighted_kev` and
    `temp_plasma_ion_density_weighted_kev` -- the four fields the graph, as assembled so
    far, actually reads. `calculate_pedestal_profile_values`
    (`process/models/physics/plasma_profiles.py:217-247`) also writes
    `temp_plasma_electron_line_avg_kev` and `.divertor.prn1`, and this node owns both
    faithfully even though nothing in the current graph reads them -- an unread output
    is pruned by `Graph.prune`, not wrong, the same reasoning `NeProfileIntegral`'s
    docstring (`functional_process/cottax/physics/profiles.py`) gives for its own
    always-computed, sometimes-unread output.

    **`nd_plasma_electron_line`/`temp_plasma_electron_line_avg_kev` are pass-throughs on
    this arm, not recomputations.** PROCESS stores `neprofile.profile_integ` /
    `teprofile.profile_integ` straight into these two fields
    (`process/models/physics/plasma_profiles.py:234,236-238`), where the parabolic arm
    computes closed-form gamma-function expressions instead
    (`plasma_profiles.py:136-150`, `ParabolicProfileValues` above) -- exactly the
    identity `next_steps.md` §8.1 rows 9/10 record. `calculate_pedestal_profile_values`
    already takes `ne_profile_integ`/`te_profile_integ` as arguments and returns them
    unmodified for this reason (see its own docstring); this node reads them off
    `NeProfileIntegral`/`TeProfileIntegral`
    (`functional_process/cottax/physics/profiles.py`), which are registered in `COMMON`
    and run on both switch arms, so the pass-through is a real graph edge, not a
    special-cased identity.

    **`.divertor.prn1` is a cross-area write**, this file's only one (the audit record's
    open question 5): the parabolic arm never writes this field -- PROCESS's own comment
    at `plasma_profiles.py:240-241` says the input value is used instead when
    `i_plasma_pedestal == 0` -- so `.divertor.prn1` is owned by this node alone, not by
    both arms of the switch.

    **Every read this node declares is already produced somewhere in the graph.**
    `radius_plasma_profile_norm`/`nd_plasma_electron_profile` by the `COMMON`
    `ProfileGrid`/`DensityProfile`; `temp_plasma_electron_profile_kev` by this arm's own
    `PedestalTemperatureProfile`; `nd_plasma_electron_profile_integral`/
    `temp_plasma_electron_profile_integral_kev` by the `COMMON` `NeProfileIntegral`/
    `TeProfileIntegral`; `temp_plasma_ion_vol_avg_kev` by the `COMMON`
    `IonVolAvgTemperature`; and the remaining three are ordinary boundary inputs
    (`temp_plasma_electron_vol_avg_kev` is iteration variable 4;
    `nd_plasma_separatrix_electron` is one of the pedestal arm's own seven input
    fields, `_audit/tokamak_boundary.md` § "Seven pedestal-arm inputs";
    `nd_plasma_electrons_vol_avg` is a boundary input on both arms). So registering
    this node adds **zero** new boundary reads, matching `tokamak_boundary.md`'s own
    count of the pedestal gap as four variables, not four variables plus new inputs.
    """

    temp_plasma_electron_density_weighted_kev = OutputInto(physics)
    temp_plasma_ion_density_weighted_kev = OutputInto(physics)
    f_temp_plasma_electron_density_vol_avg = OutputInto(physics)
    nd_plasma_electron_line = OutputInto(physics)
    temp_plasma_electron_line_avg_kev = OutputInto(physics)
    prn1 = OutputInto(divertor)

    def __call__(
        self,
        radius_plasma_profile_norm=From(physics),
        nd_plasma_electron_profile=From(physics),
        temp_plasma_electron_profile_kev=From(physics),
        nd_plasma_electron_profile_integral=From(physics),
        temp_plasma_electron_profile_integral_kev=From(physics),
        temp_plasma_ion_vol_avg_kev=From(physics),
        temp_plasma_electron_vol_avg_kev=From(physics),
        nd_plasma_separatrix_electron=From(physics),
        nd_plasma_electrons_vol_avg=From(physics),
    ):
        return calculate_pedestal_profile_values(
            radius_plasma_profile_norm,
            nd_plasma_electron_profile,
            temp_plasma_electron_profile_kev,
            nd_plasma_electron_profile_integral,
            temp_plasma_electron_profile_integral_kev,
            temp_plasma_ion_vol_avg_kev,
            temp_plasma_electron_vol_avg_kev,
            nd_plasma_separatrix_electron,
            nd_plasma_electrons_vol_avg,
        )
