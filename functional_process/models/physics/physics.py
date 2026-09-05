"""Pure-functional port of the **tokamak arm** of `process/models/physics/physics.py`.

Audit record: `functional_process/_audit/units/models/physics/physics.md`. Read it
first -- especially "the two radiation call sites disagree" and "`p_plasma_separatrix_mw`
is written twice in one pass", both of which decide the node split here.

**Not the file.** `process/models/physics/physics.py` is 6931 lines and five `Model`
classes. What is ported here is the *minimal closure* that produces the eight variables
`_audit/tokamak_boundary.md` attributes to the `.tokamak.physics` slot, traced write by
write (`file:line` for each is in the record's "data footprint"):

| output | write site | who computes it |
|---|---|---|
| `.physics.b_plasma_surface_poloidal_average` | `:313-324` | `plasma_fields.py:27-93` |
| `.physics.pden_plasma_core_rad_mw` | `:751` | `radiation_power.py`, *no clip* |
| `.physics.p_plasma_inner_rad_mw` | `:758-760` | inline |
| `.physics.p_plasma_rad_mw` | `:764-766` | inline |
| `.physics.p_plasma_separatrix_mw` | `:800-809`, `:843-845` | `exhaust.py`, inline |
| `.times.t_plant_pulse_plasma_present` | `:516` | `pulse.py:71-79` |
| `.times.t_plant_pulse_total` | `:521` | `pulse.py:92-95` |
| `.physics.e_plasma_beta` | `:3912-3916` | `physics.py:4153-4176` |

(the write sites are all in `physics.py`)

Three of those eight are produced by code that does **not** live in `physics.py`:
`calculate_surface_averaged_poloidal_field` (`plasma_fields.py`),
`calculate_separatrix_power` (`exhaust.py`) and `PulseTimings` (`pulse.py`). They are
ported here rather than deferred, the same call `confinement_time.py` made for
`calculate_iter_physics_basis_elongation` (out of nominal file scope, one small pure
function, needed for closure) -- except for the pulse sums, which are **not re-ported at
all**: `models/stellarator/initialization.py::PulseDurations` is already exactly
`PulseTimings.plasma_present`/`no_burn`/`total` and is reused. See the record's
"already-ported sub-calls".

**Two switch families, one occupant each, per the wave's binding policy** (no switch is
a static kwarg; an occupant answers the values this port supports and the rest are
`UNPORTED`):

- `i_plasma_current` decides `b_plasma_surface_poloidal_average`
  (`plasma_fields.py:83`). Value `2` (`PENG_DIVERTOR_SCALING`) reads `q95`, `aspect`,
  `b_plasma_toroidal_on_axis`, `kappa`, `triang` and calls `plascar_bpol`; every other
  value reads `plasma_current` and `len_plasma_poloidal` and nothing else. Two genuinely
  disjoint reads-sets, so two occupants; only the Ampere one is written.
- `(i_pulsed_plant, pulsetimings, i_t_current_ramp_up)` decides the ramp times
  (`physics.py:464-498`). `pulsetimings` has its **only read in all of
  `process/models/**`** here. Four arms, two written -- the `(1, 0, --)` arm
  `large_tokamak_eval.IN.DAT` uses and the `(0, --, 0)` arm the two spherical-tokamak
  files select.

`i_plasma_ignited` decides `p_plasma_separatrix_mw`'s injected-heating term
(`physics.py:793-798`) -- the `NON_IGNITED` occupant is the live one here and is written;
note this is a *different* node from `confinement_time.py`'s `PlasmaPowerLoss` family,
which answers the same switch for a different variable and whose live arm is `IGNITED`
on the stellarator runs. The two are not in conflict: they are two nodes, each declaring
its own arm.
"""

from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    OutputInto,
)

from functional_process.paths import current_drive, physics, times
from functional_process.physics.physics import (
    calculate_beta_limit_from_norm,
    calculate_beta_norm_max_wesson,
    calculate_continuous_plant_ramp_times,
    calculate_coulomb_logarithm_ion_electron,
    calculate_pflux_plasma_surface_neutron_avg_mw,
    calculate_plasma_energy_from_beta,
    calculate_poloidal_beta,
    calculate_pulsed_plant_ramp_times,
    calculate_separatrix_power,
    calculate_surface_averaged_poloidal_field_amperes,
    calculate_thermal_beta,
    calculate_toroidal_beta,
    calculate_total_radiation_power,
    calculate_unclipped_radiation_powers,
    force_positive_separatrix_power,
    plasma_ohmic_heating,
)


class SurfaceAveragedPoloidalField(ExplicitFunction):
    """The family that owns `.physics.b_plasma_surface_poloidal_average`.

    `i_plasma_current` decides it (`plasma_fields.py:83`), and the two arms are as far
    apart as a switch's arms get: the Ampere arm reads two variables, the Peng arm reads
    five different ones and calls `PlasmaCurrent.plascar_bpol`. One occupant per arm; the
    Peng arm is `UNPORTED` because `plascar_bpol` belongs to `plasma_current.py`, which
    is not ported.
    """


class SurfaceAveragedPoloidalFieldAmperes(SurfaceAveragedPoloidalField):
    """`i_plasma_current != PENG_DIVERTOR_SCALING`: Ampere's law over the perimeter.

    Answers every `PlasmaCurrentModel` value except `PENG_DIVERTOR_SCALING` (2) --
    PROCESS's own test at `plasma_fields.py:83` is `!= 2`, so the eight remaining values
    are one arm in the source, not eight, and declaring them as one occupant states what
    the source states rather than inventing a distinction. `large_tokamak_eval.IN.DAT`
    uses `4` (`IPDG89_SCALING`).
    """

    b_plasma_surface_poloidal_average = OutputInto(physics)

    def __call__(
        self,
        plasma_current=From(physics),
        len_plasma_poloidal=From(physics),
    ):
        return calculate_surface_averaged_poloidal_field_amperes(
            plasma_current,
            len_plasma_poloidal,
        )


class UnclippedRadiationPowers(ExplicitFunction):
    """cottax node: `calculate_unclipped_radiation_powers`, ports declared.

    The tokamak's counterpart to
    `models/stellarator/plasma_physics.py::ClippedRadiationPowers`, and the two are the
    reason `PlasmaRadiationPowers` owns `_unclipped` mints rather than the real fields:
    the same PROCESS function feeds both, and only one of its callers clips. Owns the
    real `.physics.pden_plasma_core_rad_mw`/`pden_plasma_outer_rad_mw` on a tokamak, and
    gives `.physics.p_plasma_inner_rad_mw` its tokamak producer.
    """

    pden_plasma_core_rad_mw = OutputInto(physics)
    pden_plasma_outer_rad_mw = OutputInto(physics)
    p_plasma_inner_rad_mw = OutputInto(physics)
    p_plasma_outer_rad_mw = OutputInto(physics)

    def __call__(
        self,
        pden_plasma_core_rad_mw_unclipped=From(physics),
        pden_plasma_outer_rad_mw_unclipped=From(physics),
        vol_plasma=From(physics),
    ):
        return calculate_unclipped_radiation_powers(
            pden_plasma_core_rad_mw_unclipped,
            pden_plasma_outer_rad_mw_unclipped,
            vol_plasma,
        )


class TotalRadiationPower(ExplicitFunction):
    """cottax node: `calculate_total_radiation_power`, ports declared."""

    p_plasma_rad_mw = OutputInto(physics)

    def __call__(
        self,
        pden_plasma_rad_mw=From(physics),
        vol_plasma=From(physics),
    ):
        return calculate_total_radiation_power(pden_plasma_rad_mw, vol_plasma)


class SeparatrixPower(ExplicitFunction):
    """The family that owns `.physics.p_plasma_separatrix_mw_raw`.

    `i_plasma_ignited` decides it (`physics.py:793-798`): an ignited plasma does not
    count injected heating towards the power crossing the separatrix, so the two arms
    differ by exactly one read -- `.current_drive.p_hcd_injected_total_mw`. That is a
    reads-set difference, so it is a split, and declaring it removes a
    `.current_drive -> .physics` edge from whichever arm is not live.
    """


class SeparatrixPowerNonIgnited(SeparatrixPower):
    """`i_plasma_ignited == NON_IGNITED`: injected heating crosses the separatrix.

    The arm `large_tokamak_eval.IN.DAT` uses (`i_plasma_ignited` unset, default `0`).
    Note this is the *opposite* arm from `confinement_time.py`'s
    `PlasmaPowerLossIgnitedCoreRadiation`, which answers the same switch for
    `.physics.p_plasma_loss_mw` on the stellarator runs -- two nodes, two configurations,
    no conflict.

    Owns the **mint**, not `.physics.p_plasma_separatrix_mw`: see
    `force_positive_separatrix_power`'s docstring for why the field has two producers in
    one PROCESS pass and how they are told apart here.
    """

    p_plasma_separatrix_mw_raw = OutputInto(physics)

    def __call__(
        self,
        f_p_alpha_plasma_deposited=From(physics),
        p_alpha_total_mw=From(physics),
        p_non_alpha_charged_mw=From(physics),
        p_hcd_injected_total_mw=From(current_drive),
        p_plasma_ohmic_mw=From(physics),
        p_plasma_rad_mw=From(physics),
    ):
        return calculate_separatrix_power(
            f_p_alpha_plasma_deposited=f_p_alpha_plasma_deposited,
            p_alpha_total_mw=p_alpha_total_mw,
            p_non_alpha_charged_mw=p_non_alpha_charged_mw,
            p_hcd_injected_total_mw=p_hcd_injected_total_mw,
            p_plasma_ohmic_mw=p_plasma_ohmic_mw,
            p_plasma_rad_mw=p_plasma_rad_mw,
        )


class PositiveSeparatrixPower(ExplicitFunction):
    """cottax node: `force_positive_separatrix_power`, ports declared.

    Owns the real `.physics.p_plasma_separatrix_mw` -- the value every consumer after
    `physics.py:845` sees, which is the one `.power.component_thermal_powers`,
    `.power.delta_eta_step` and `.power.p_fw_div_heat_deposited_mw_step` read.
    """

    p_plasma_separatrix_mw = OutputInto(physics)

    def __call__(
        self,
        p_plasma_separatrix_mw_raw=From(physics),
    ):
        return force_positive_separatrix_power(p_plasma_separatrix_mw_raw)


class PulseRampTimes(ExplicitFunction):
    """The family that owns the plasma-current ramp times, `physics.py:463-498`.

    Two switches decide it jointly and neither decides it alone, so this is one family
    with a joint arm index rather than two slots -- the same shape as `indat.py`'s
    `_energy_storage_arm` (`i_pulsed_plant`/`istore`) and `_plasma_power_loss_arm`:

    - **arm 0** -- `i_pulsed_plant != 1` and `i_t_current_ramp_up == 0`
      (`physics.py:465-474`): ramp-up, precharge and ramp-down, all from
      `plasma_current / 5e5`.
    - **arm 1** -- `i_pulsed_plant != 1` and `i_t_current_ramp_up != 0`: writes
      nothing; the three times are inputs.
    - **arm 2** *(live)* -- `i_pulsed_plant == 1` and `pulsetimings == 0`
      (`physics.py:476-483`): ramp-up `= plasma_current / 1e5`, ramp-down `= ramp-up`.
    - **arm 3** -- `i_pulsed_plant == 1` and `pulsetimings != 0`
      (`physics.py:485-498`): precharge `= max(precharge, ramp-up)`, ramp-down
      `= ramp-up`.

    `pulsetimings` is read **here and nowhere else in all of `process/models/**`**
    (`physics.py:476`), so this family is the whole of that decision.

    The fourth arm is not merely unwritten: `physics.py:489-492` reads
    `.times.t_plant_pulse_coil_precharge` and writes it back, which is a node reading
    what it owns and needs the `FixedPointFunction` treatment or a producer split before
    it can exist at all. Recorded as an open item, not improvised.
    """


class PulseRampTimesPulsedDefault(PulseRampTimes):
    """`i_pulsed_plant == 1` and `pulsetimings == 0` -- `large_tokamak_eval`'s arm.

    Reads one variable and writes two. The arm that would be its static-kwarg twin
    (`i_pulsed_plant != 1`, `i_t_current_ramp_up == 0`) differs by the literal `5e5` vs
    `1e5` *and* by owning `.times.t_plant_pulse_coil_precharge` as a third output, so
    even the literal-only reading of the policy does not apply here.
    """

    t_plant_pulse_plasma_current_ramp_up = OutputInto(times)
    t_plant_pulse_plasma_current_ramp_down = OutputInto(times)

    def __call__(
        self,
        plasma_current=From(physics),
    ):
        return calculate_pulsed_plant_ramp_times(plasma_current)


class PulseRampTimesContinuousDefault(PulseRampTimes):
    """`i_pulsed_plant != 1` and `i_t_current_ramp_up == 0` -- the spherical tokamaks'.

    The arm both `spherical_tokamak_eval.IN.DAT` (`:312`) and `st_regression.IN.DAT`
    (`:2979`) select: `i_pulsed_plant = 0` in the file, `i_t_current_ramp_up` left at
    PROCESS's own default `0` (`times_variables.py:44`).

    Reads one variable and writes **three**: on this arm
    `.times.t_plant_pulse_coil_precharge` is owned (`physics.py:469-471`), where on the
    pulsed-default arm it is an input (the source's own comment at `:477`). That
    write-set difference -- not the `5e5`-vs-`1e5` literal -- is why the two arms are
    separate occupants; see the family docstring.
    """

    t_plant_pulse_plasma_current_ramp_up = OutputInto(times)
    t_plant_pulse_coil_precharge = OutputInto(times)
    t_plant_pulse_plasma_current_ramp_down = OutputInto(times)

    def __call__(
        self,
        plasma_current=From(physics),
    ):
        return calculate_continuous_plant_ramp_times(plasma_current)


class PlasmaEnergyFromBeta(ExplicitFunction):
    """cottax node: `calculate_plasma_energy_from_beta`, total-beta binding.

    `PlasmaBeta.run` calls the same static twice (`physics.py:3905-3916`), once with
    `beta_thermal_vol_avg` for `.physics.e_plasma_beta_thermal` and once with
    `beta_total_vol_avg` for `.physics.e_plasma_beta`. Only the second is in this slot's
    eight outputs, so only the second has a node here; the thermal binding is a
    two-line follow-up whenever `.physics.e_plasma_beta_thermal` gains a consumer
    (today only `outplas` reports it, `physics.py:4666-4675`).

    The stellarator's counterpart is `StellaratorBetaAndStoredEnergy`, which owns
    `.physics.e_plasma_beta` there as one of three outputs of a fused expression. This
    node is the tokamak's, and it is the narrower one: PROCESS's tokamak arm has the
    formula factored out as its own `@staticmethod` already.
    """

    e_plasma_beta = OutputInto(physics)

    def __call__(
        self,
        beta_total_vol_avg=From(physics),
        b_plasma_total=From(physics),
        vol_plasma=From(physics),
    ):
        return calculate_plasma_energy_from_beta(
            beta_total_vol_avg,
            b_plasma_total,
            vol_plasma,
        )


class PlasmaOhmicHeating(ExplicitFunction):
    """cottax node: `plasma_ohmic_heating`. No switch -- `Physics.run` computes it
    unconditionally (`physics.py:768-778`), whatever the machine's arms.

    Owns all four of the writeback's fields. `.physics.res_plasma` is the one
    `cold_boundary.md` names (read by `plasma_inductance.volt_seconds`, whose zero
    made `v_plasma_loop_burn` zero and the cold burn time nan);
    `.physics.p_plasma_ohmic_mw` closes the boundary read of `separatrix_power` and
    `PlasmaPowerLoss`; the other two are the same source function's remaining stores,
    owned so the writeback is whole rather than sliced.
    """

    pden_plasma_ohmic_mw = OutputInto(physics)
    p_plasma_ohmic_mw = OutputInto(physics)
    f_res_plasma_neo = OutputInto(physics)
    res_plasma = OutputInto(physics)

    def __call__(
        self,
        f_c_plasma_inductive=From(physics),
        kappa95=From(physics),
        plasma_current=From(physics),
        rmajor=From(physics),
        rminor=From(physics),
        temp_plasma_electron_density_weighted_kev=From(physics),
        vol_plasma=From(physics),
        n_charge_plasma_effective_vol_avg=From(physics),
        plasma_res_factor=From(physics),
    ):
        return plasma_ohmic_heating(
            f_c_plasma_inductive=f_c_plasma_inductive,
            kappa95=kappa95,
            plasma_current=plasma_current,
            rmajor=rmajor,
            rminor=rminor,
            temp_plasma_electron_density_weighted_kev=(
                temp_plasma_electron_density_weighted_kev
            ),
            vol_plasma=vol_plasma,
            zeff=n_charge_plasma_effective_vol_avg,
            plasma_res_factor=plasma_res_factor,
        )


class CoulombLogarithmIonElectron(ExplicitFunction):
    """cottax node: `calculate_coulomb_logarithm_ion_electron`. Unswitched --
    `Physics.run` computes it on every tokamak pass, whatever the machine's arms.

    Owns `.physics.dlamie` only; see the function's docstring for why `.physics.dlamee`
    beside it is UNPORTED, and for why this node is a tokamak slot even though both
    devices read the field.
    """

    dlamie = OutputInto(physics)

    def __call__(
        self,
        nd_plasma_electrons_vol_avg=From(physics),
        temp_plasma_electron_vol_avg_kev=From(physics),
    ):
        return calculate_coulomb_logarithm_ion_electron(
            nd_plasma_electrons_vol_avg, temp_plasma_electron_vol_avg_kev
        )


class PlasmaSurfaceNeutronFlux(ExplicitFunction):
    """cottax node: `calculate_pflux_plasma_surface_neutron_avg_mw`. Unswitched.

    Both reads are produced on the tokamak path -- `.physics.p_neutron_total_mw` by
    `.physics.set_fusion_powers` and `.physics.a_plasma_surface` by
    `.tokamak.plasma_geom.geometry` -- so this node adds no boundary input at all,
    only takes one away.
    """

    pflux_plasma_surface_neutron_avg_mw = OutputInto(physics)

    def __call__(
        self,
        p_neutron_total_mw=From(physics),
        a_plasma_surface=From(physics),
    ):
        return calculate_pflux_plasma_surface_neutron_avg_mw(
            p_neutron_total_mw, a_plasma_surface
        )


class BetaNormMaxWesson(ExplicitFunction):
    """cottax node: `calculate_beta_norm_max_wesson`.

    Occupant of the `.physics.i_beta_norm_max` slot for value `1` (`WESSON`,
    `large_tokamak_eval.IN.DAT:287`'s default -- the file sets `i_beta_component` but
    not this one, and `physics_variables.py`'s default is `1`).

    **The slot exists because `get_beta_norm_max_value` is a five-way select over five
    separately computed fields** (`physics.py:3723-3743`), and the wave's binding policy
    is one occupant class per switch value, each declaring only its own arm's reads.
    PROCESS itself computes all five scalings unconditionally and then picks
    (`physics.py:3766-3800`); the four unselected ones are dead work and are not
    computed here. `.physics.beta_norm_max_wesson` and its four siblings are therefore
    still boundary inputs -- only the *selected* value is owned, under the name every
    consumer actually reads.

    The `USER_INPUT` arm (`0`) has **no occupant and cannot have one**: PROCESS's
    `model_map` returns `physics_data.beta_norm_max` itself, so the node would own what
    it reads. `indat.py`'s `BETA_NORM_MAX` registry leaves that value out on purpose and
    `.physics.beta_norm_max` stays a boundary input there, which is exactly what PROCESS
    leaves it as.
    """

    beta_norm_max = OutputInto(physics)

    def __call__(self, ind_plasma_internal_norm=From(physics)):
        return calculate_beta_norm_max_wesson(ind_plasma_internal_norm)


class BetaLimitFromNorm(ExplicitFunction):
    """cottax node: `calculate_beta_limit_from_norm`. Unswitched."""

    beta_vol_avg_max = OutputInto(physics)

    def __call__(
        self,
        b_plasma_toroidal_on_axis=From(physics),
        beta_norm_max=From(physics),
        plasma_current=From(physics),
        rminor=From(physics),
    ):
        return calculate_beta_limit_from_norm(
            b_plasma_toroidal_on_axis,
            beta_norm_max,
            plasma_current,
            rminor,
        )


class ToroidalBeta(ExplicitFunction):
    """cottax node: `calculate_toroidal_beta`. Unswitched."""

    beta_toroidal_vol_avg = OutputInto(physics)

    def __call__(
        self,
        beta_total_vol_avg=From(physics),
        b_plasma_total=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
    ):
        return calculate_toroidal_beta(
            beta_total_vol_avg,
            b_plasma_total,
            b_plasma_toroidal_on_axis,
        )


class PoloidalBeta(ExplicitFunction):
    """cottax node: `calculate_poloidal_beta`. Unswitched.

    Both reads are produced on the tokamak path -- `b_plasma_total` by
    `.tokamak.plasma_fields`' `TotalMagneticField`, `b_plasma_surface_poloidal_average`
    by this file's own `SurfaceAveragedPoloidalField` family -- and `beta_total_vol_avg`
    is iteration variable 5 on both tracked tokamaks, i.e. a design input by intent.
    """

    beta_poloidal_vol_avg = OutputInto(physics)

    def __call__(
        self,
        b_plasma_total=From(physics),
        b_plasma_surface_poloidal_average=From(physics),
        beta_total_vol_avg=From(physics),
    ):
        return calculate_poloidal_beta(
            b_plasma_total,
            b_plasma_surface_poloidal_average,
            beta_total_vol_avg,
        )


class ThermalBeta(ExplicitFunction):
    """cottax node: `calculate_thermal_beta`. Unswitched.

    `.physics.beta_beam` is a boundary input on this machine and stays one: its producer
    is `beam_fusion`, which `unit_registry.md`'s constraint-7 row already records as an
    unported conditional producer. It is zero here (no neutral beam), so the thermal
    beta is exact anyway -- but the read is declared, not folded away, because the
    edge is real the moment a beam-heated file is assembled.
    """

    beta_thermal_vol_avg = OutputInto(physics)

    def __call__(
        self,
        beta_total_vol_avg=From(physics),
        beta_fast_alpha=From(physics),
        beta_beam=From(physics),
    ):
        return calculate_thermal_beta(beta_total_vol_avg, beta_fast_alpha, beta_beam)
