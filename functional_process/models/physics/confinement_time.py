"""Pure-functional port of `process/models/physics/confinement_time.py`.

Registry unit #10. Audit record:
`functional_process/_audit/units/models/physics/confinement_time.md`. Read it first,
especially "A latent PROCESS bug, ported faithfully" (the `KAYE_GOLDSTON` branch) and "A
dead branch" (`PAZ_SOLDAN_NT`) before trusting any single scaling law's numbers against
`calculate_confinement_time`'s dispatch.

In scope: `calculate_confinement_time` and `calculate_double_and_triple_product`
(registry's stated method list), plus everything they call transitively within this same
file -- 48 individual `<name>_confinement_time` scaling-law statics, all already pure
(no `self.data` access of their own). Also ported here, out of nominal file scope but
needed for closure: `calculate_iter_physics_basis_elongation`, a one-line pure formula
`calculate_confinement_time` calls into `process/models/physics/plasma_geometry.py`
for -- see the audit record's "calls into other models".

Every scaling law keeps its PROCESS parameter names and formula verbatim, translated
`np.` -> `jnp.`, `min`/`max` -> `jnp.minimum`/`jnp.maximum` (JAX cannot trace a Python
`min`/`max` over a differentiable argument -- see the audit record's JAX-difficulty
flags). `menard_nstx_petty08_hybrid_confinement_time`'s three-way `if`/`elif`/`else` on
`1/aspect` is replaced with the equivalent clipped linear blend (verified continuous:
the "else" branch already reduces to the two boundary values exactly at the two
thresholds), since `aspect` is a differentiable argument here, not a switch.

`i_confinement_time` and `i_rad_loss` are switches (`_audit/naming_convention.md` §
"switches are not ports"): plain Python ints used for ordinary branching in
`calculate_confinement_time`, never traced. The harness marks them
`static_argnames` so `jacfwd` never differentiates through the dispatch itself.
"""

from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    FromExactly,
    OutputInto,
)

from functional_process.paths import current_drive, physics, stellarator
from functional_process.physics.confinement_time import (
    calculate_confinement_time,
    calculate_double_and_triple_product,
    calculate_iter_physics_basis_elongation,
    christiansen_confinement_time,
    confinement_from_scaling,
    confinement_scaling_inputs,
    ds03_confinement_time,
    goldston_confinement_time,
    gyro_reduced_bohm_confinement_time,
    hubbard_lower_confinement_time,
    hubbard_nominal_confinement_time,
    hubbard_upper_confinement_time,
    iss04_stellarator_confinement_time,
    iss95_stellarator_confinement_time,
    iter_89_0_confinement_time,
    iter_89p_confinement_time,
    iter_93h_confinement_time,
    iter_96p_confinement_time,
    iter_h90_p_amended_confinement_time,
    iter_h90_p_confinement_time,
    iter_h97p_confinement_time,
    iter_h97p_elmy_confinement_time,
    iter_ipb98y1_confinement_time,
    iter_ipb98y2_confinement_time,
    iter_ipb98y3_confinement_time,
    iter_ipb98y4_confinement_time,
    iter_ipb98y_confinement_time,
    iter_pb98py_confinement_time,
    itpa20_confinement_time,
    itpa20_il_confinement_time,
    jaeri_confinement_time,
    kaye_big_confinement_time,
    kaye_confinement_time,
    kaye_goldston_confinement_time,
    lackner_gottardi_confinement_time,
    lackner_gottardi_stellarator_confinement_time,
    lang_high_density_confinement_time,
    menard_nstx_confinement_time,
    menard_nstx_petty08_hybrid_confinement_time,
    merezhkin_muhkovatov_confinement_time,
    mirnov_confinement_time,
    murari_confinement_time,
    ncst_confinement_time,
    neo_alcator_confinement_time,
    neo_kaye_confinement_time,
    nstx_gyro_bohm_confinement_time,
    paz_soldan_nt_confinement_time,
    petty08_confinement_time,
    plasma_power_loss_mw,
    rebut_lallia_confinement_time,
    riedel_h_confinement_time,
    riedel_l_confinement_time,
    shimomura_confinement_time,
    sudo_et_al_confinement_time,
    t10_confinement_time,
    valovic_elmy_confinement_time,
)
from functional_process.vocabulary import (
    ConfinementRadiationLossModel,
    PlasmaIgnitionModel,
)

__all__ = [
    "calculate_confinement_time",
    "christiansen_confinement_time",
    "ds03_confinement_time",
    "goldston_confinement_time",
    "gyro_reduced_bohm_confinement_time",
    "hubbard_lower_confinement_time",
    "hubbard_nominal_confinement_time",
    "hubbard_upper_confinement_time",
    "iss95_stellarator_confinement_time",
    "iter_89_0_confinement_time",
    "iter_89p_confinement_time",
    "iter_93h_confinement_time",
    "iter_96p_confinement_time",
    "iter_h90_p_amended_confinement_time",
    "iter_h90_p_confinement_time",
    "iter_h97p_confinement_time",
    "iter_h97p_elmy_confinement_time",
    "iter_ipb98y1_confinement_time",
    "iter_ipb98y3_confinement_time",
    "iter_ipb98y4_confinement_time",
    "iter_ipb98y_confinement_time",
    "iter_pb98py_confinement_time",
    "itpa20_confinement_time",
    "itpa20_il_confinement_time",
    "jaeri_confinement_time",
    "kaye_big_confinement_time",
    "kaye_confinement_time",
    "kaye_goldston_confinement_time",
    "lackner_gottardi_confinement_time",
    "lackner_gottardi_stellarator_confinement_time",
    "lang_high_density_confinement_time",
    "menard_nstx_confinement_time",
    "menard_nstx_petty08_hybrid_confinement_time",
    "merezhkin_muhkovatov_confinement_time",
    "mirnov_confinement_time",
    "murari_confinement_time",
    "ncst_confinement_time",
    "neo_alcator_confinement_time",
    "neo_kaye_confinement_time",
    "nstx_gyro_bohm_confinement_time",
    "paz_soldan_nt_confinement_time",
    "petty08_confinement_time",
    "rebut_lallia_confinement_time",
    "riedel_h_confinement_time",
    "riedel_l_confinement_time",
    "shimomura_confinement_time",
    "sudo_et_al_confinement_time",
    "t10_confinement_time",
    "valovic_elmy_confinement_time",
]


class IterPhysicsBasisElongation(ExplicitFunction):
    """cottax node: `calculate_iter_physics_basis_elongation`, ports declared."""

    kappa_ipb = OutputInto(physics)

    def __call__(
        self,
        vol_plasma=From(physics),
        rmajor=From(physics),
        rminor=From(physics),
    ):
        return calculate_iter_physics_basis_elongation(vol_plasma, rmajor, rminor)


class ConfinementScalingInputs(ExplicitFunction):
    """The unit conversions every scaling law takes as arguments.

    PROCESS computes these inline at the head of `calculate_confinement_time` and stores
    none of them, so `.physics.nd_plasma_electron_line_19` and `.physics.cur_plasma_ma`
    have no backing `DataStructure` field. That is PROCESS's omission, not a reason to
    invent a namespace for them: they are values one node computes and several others
    consume, which is what a graph variable *is*. The consequence is bookkeeping and is
    stated where it lands -- the MDA harness cannot compare them against PROCESS's
    converged state, so they join its not-data-backed category.

    Owning them here is what lets a scaling node's signature be **exactly** its law's:
    no argument preparation in the node body, so the node is callable as the function it
    declares and the harness can diff the node itself against PROCESS's own staticmethod.
    """

    nd_plasma_electron_line_19 = OutputInto(physics)
    cur_plasma_ma = OutputInto(physics)

    def __call__(
        self,
        nd_plasma_electron_line=From(physics),
        plasma_current=From(physics),
    ):
        return confinement_scaling_inputs(nd_plasma_electron_line, plasma_current)


class PlasmaPowerLoss(ExplicitFunction):
    """The family that owns `.physics.p_plasma_loss_mw`: the head, one occupant per arm.

    Two switches decide it -- `i_plasma_ignited` (whether injected heating counts) and
    `i_rad_loss` (which radiation term is subtracted) -- and both change the *reads*, so
    both are occupants rather than static kwargs (`traceability_policy.md`'s
    split-by-default). Only the arm this port supports is written; the rest are
    `UNPORTED` entries in `indat.py`, which is `switch_kwarg_survey.md` band (d)'s rule:
    an occupant per value *this port supports*, not per value PROCESS has.
    """


class PlasmaPowerLossIgnitedCoreRadiation(PlasmaPowerLoss):
    """`i_plasma_ignited == IGNITED` and `i_rad_loss == CORE_ONLY` -- both runs' arm.

    **This arm is the measured case for two invented edges.** Ignited means the
    `p_hcd_injected_total_mw` term is not taken, and core-only radiation means
    `pden_plasma_rad_mw` is not the term subtracted -- yet the composite node declared
    both, so the graph claimed a `.current_drive -> .physics` dependency this run does
    not have. Declaring the arm removes them: this class reads neither.

    It calls `plasma_power_loss_mw` with those two arguments at `0.0` rather than
    inlining the arithmetic, so there stays exactly one source of truth for the formula
    -- the one `calculate_confinement_time` is diffed against PROCESS through. A dead
    argument passed as zero is not a read: it never reaches a port.
    """

    p_plasma_loss_mw = OutputInto(physics)

    def __call__(
        self,
        f_p_alpha_plasma_deposited=From(physics),
        p_alpha_total_mw=From(physics),
        p_non_alpha_charged_mw=From(physics),
        p_plasma_ohmic_mw=From(physics),
        pden_plasma_core_rad_mw=From(physics),
        vol_plasma=From(physics),
    ):
        return plasma_power_loss_mw(
            f_p_alpha_plasma_deposited=f_p_alpha_plasma_deposited,
            p_alpha_total_mw=p_alpha_total_mw,
            p_non_alpha_charged_mw=p_non_alpha_charged_mw,
            p_plasma_ohmic_mw=p_plasma_ohmic_mw,
            p_hcd_injected_total_mw=0.0,
            pden_plasma_rad_mw=0.0,
            pden_plasma_core_rad_mw=pden_plasma_core_rad_mw,
            vol_plasma=vol_plasma,
            i_plasma_ignited=PlasmaIgnitionModel.IGNITED,
            i_rad_loss=ConfinementRadiationLossModel.CORE_ONLY,
        )


class PlasmaPowerLossNonIgnitedCoreRadiation(PlasmaPowerLoss):
    """`i_plasma_ignited == NON_IGNITED` and `i_rad_loss == CORE_ONLY`.

    **The conventional tokamak's arm, and the one `large_tokamak_eval.IN.DAT` needs.**
    Neither switch appears in that file, so both take PROCESS's own defaults --
    `i_plasma_ignited = 0` (`physics_variables.py:881`) and `i_rad_loss = 1`
    (`physics_variables.py:954`) -- and the sibling above, written for the arm both
    stellarator runs use (`stellarator_helias.IN.DAT:126` sets `i_plasma_ignited = 1`),
    does not fit. That refusal is what
    `_audit/tokamak_boundary.md` § "What blocked the real file" records; this class is
    the one occupant it says the file was blocked on.

    The difference from `PlasmaPowerLossIgnitedCoreRadiation` is one term and one read:
    a non-ignited plasma is heated by its injection system, so
    `p_hcd_injected_total_mw` enters the loss power (`process/models/physics/
    confinement_time.py:143-144`, guarded by `i_plasma_ignited` and nothing else) and
    the node declares the `.current_drive -> .physics` edge that the ignited arm
    correctly does not have. `.current_drive.p_hcd_injected_total_mw` was already on the
    tokamak boundary before this class existed (`tokamak_boundary.md` §
    `.tokamak.current_drive`); as of this pass it is produced, by
    `models/physics/current_drive.py::HcdInjectedPowerTotal`.

    `pden_plasma_rad_mw` is still passed as `0.0` for the same reason the ignited arm
    passes it: `CORE_ONLY` subtracts `pden_plasma_core_rad_mw`, so the full-radiation
    density never reaches a port.
    """

    p_plasma_loss_mw = OutputInto(physics)

    def __call__(
        self,
        f_p_alpha_plasma_deposited=From(physics),
        p_alpha_total_mw=From(physics),
        p_non_alpha_charged_mw=From(physics),
        p_plasma_ohmic_mw=From(physics),
        p_hcd_injected_total_mw=From(current_drive),
        pden_plasma_core_rad_mw=From(physics),
        vol_plasma=From(physics),
    ):
        return plasma_power_loss_mw(
            f_p_alpha_plasma_deposited=f_p_alpha_plasma_deposited,
            p_alpha_total_mw=p_alpha_total_mw,
            p_non_alpha_charged_mw=p_non_alpha_charged_mw,
            p_plasma_ohmic_mw=p_plasma_ohmic_mw,
            p_hcd_injected_total_mw=p_hcd_injected_total_mw,
            pden_plasma_rad_mw=0.0,
            pden_plasma_core_rad_mw=pden_plasma_core_rad_mw,
            vol_plasma=vol_plasma,
            i_plasma_ignited=PlasmaIgnitionModel.NON_IGNITED,
            i_rad_loss=ConfinementRadiationLossModel.CORE_ONLY,
        )


class ConfinementTimeScaling(ExplicitFunction):
    """The family that owns `.physics.t_electron_confinement`: one occupant per law.

    This is what `i_confinement_time` was: ~40 scaling laws behind one static kwarg on
    one node, which therefore declared the union of all their reads -- 32, where a law
    needs 6 to 8. Each law is already a separate, separately-validated pure function in
    this module; an occupant is that function with its own ports, and nothing else.

    **The device rebinding disappears with it.** `StellaratorConfinementTime` existed
    solely to rebind one parameter that PROCESS's own caller passes differently in
    stellarator mode: the source calls its 20th argument `q95` and hands ISS04 the
    rotational transform. With one class per law that is not a rebinding at all --
    `iss04_stellarator_confinement_time`'s own parameter *is* `iotabar`, so the occupant
    reads `.stellarator.iotabar` because that is what the law takes. The read follows
    from the law, not from the device, and `CONFINEMENT_TIME` keyed on `istell` has
    nothing left to decide.
    """


class Iss04ConfinementTime(ConfinementTimeScaling):
    """ISS04 stellarator scaling. `ConfinementTimeModel.ISS04_STELLARATOR` (38)."""

    t_electron_confinement = OutputInto(physics)

    def __call__(
        self,
        rminor=From(physics),
        rmajor=From(physics),
        nd_plasma_electron_line_19=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
        p_plasma_loss_mw=From(physics),
        iotabar=FromExactly(stellarator.iotabar),
    ):
        return iss04_stellarator_confinement_time(
            rminor,
            rmajor,
            nd_plasma_electron_line_19,
            b_plasma_toroidal_on_axis,
            p_plasma_loss_mw,
            iotabar,
        )


class IterIpb98y2ConfinementTime(ConfinementTimeScaling):
    """IPB98(y,2) ELMy H-mode scaling. `ConfinementTimeModel.ITER_IPB98Y2` (34).

    The conventional tokamak's law, and the reason this family exists before there is a
    tokamak to use it: `large_tokamak_eval.IN.DAT` sets `i_confinement_time = 34` where
    the tree pinned `38`, which is one of the four contradictions
    `_audit/tokamak_scope.md` names as the first tokamak deliverable.
    """

    t_electron_confinement = OutputInto(physics)

    def __call__(
        self,
        cur_plasma_ma=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
        nd_plasma_electron_line_19=From(physics),
        p_plasma_loss_mw=From(physics),
        rmajor=From(physics),
        kappa_ipb=From(physics),
        aspect=From(physics),
        afuel=FromExactly(physics.m_fuel_amu),
    ):
        return iter_ipb98y2_confinement_time(
            cur_plasma_ma,
            b_plasma_toroidal_on_axis,
            nd_plasma_electron_line_19,
            p_plasma_loss_mw,
            rmajor,
            kappa_ipb,
            aspect,
            afuel,
        )


class ConfinementTail(ExplicitFunction):
    """The family that owns everything downstream of the chosen law.

    Identical for all ~40 laws, which is why keeping it inside the dispatching node was
    what forced that node to declare 32 reads. `i_rad_loss` decides it a second time,
    and here the three arms read genuinely different variables.
    """


class ConfinementTailCoreRadiation(ConfinementTail):
    """`i_rad_loss == CORE_ONLY`: `hstar` degrades on synchrotron plus inner radiation.

    Reads `pden_plasma_sync_mw` and `p_plasma_inner_rad_mw` and **not**
    `pden_plasma_rad_mw`, which is the `FULL_RADIATION` arm's read.
    """

    pden_electron_transport_loss_mw = OutputInto(physics)
    pden_ion_transport_loss_mw = OutputInto(physics)
    t_electron_energy_confinement = OutputInto(physics)
    t_ion_energy_confinement = OutputInto(physics)
    t_energy_confinement = OutputInto(physics)
    hstar = OutputInto(physics)
    t_energy_confinement_beta = OutputInto(physics)

    def __call__(
        self,
        t_electron_confinement=From(physics),
        hfact=From(physics),
        p_plasma_loss_mw=From(physics),
        pden_plasma_sync_mw=From(physics),
        p_plasma_inner_rad_mw=From(physics),
        vol_plasma=From(physics),
        eden_plasma_ions_thermal_vol_avg=From(physics),
        eden_plasma_electrons_thermal_vol_avg=From(physics),
        e_plasma_beta=From(physics),
    ):
        return confinement_from_scaling(
            t_electron_confinement=t_electron_confinement,
            hfact=hfact,
            p_plasma_loss_mw=p_plasma_loss_mw,
            i_rad_loss=ConfinementRadiationLossModel.CORE_ONLY,
            pden_plasma_sync_mw=pden_plasma_sync_mw,
            p_plasma_inner_rad_mw=p_plasma_inner_rad_mw,
            pden_plasma_rad_mw=0.0,
            vol_plasma=vol_plasma,
            eden_plasma_ions_thermal_vol_avg=eden_plasma_ions_thermal_vol_avg,
            eden_plasma_electrons_thermal_vol_avg=eden_plasma_electrons_thermal_vol_avg,
            e_plasma_beta=e_plasma_beta,
        )


class DoubleAndTripleProduct(ExplicitFunction):
    """cottax node: `calculate_double_and_triple_product`, ports declared."""

    ntau = OutputInto(physics)
    nTtau = OutputInto(physics)

    def __call__(
        self,
        nd_plasma_electrons_vol_avg=From(physics),
        temp_plasma_electron_vol_avg_kev=From(physics),
        t_energy_confinement=From(physics),
    ):
        return calculate_double_and_triple_product(
            nd_plasma_electrons_vol_avg,
            temp_plasma_electron_vol_avg_kev,
            t_energy_confinement,
        )
