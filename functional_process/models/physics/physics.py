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
- `(i_pulsed_plant, pulsetimings)` decides the ramp times (`physics.py:464-498`).
  `pulsetimings` has its **only read in all of `process/models/**`** here. Three arms,
  one written -- the `(1, 0)` arm `large_tokamak_eval.IN.DAT` uses.

`i_plasma_ignited` decides `p_plasma_separatrix_mw`'s injected-heating term
(`physics.py:793-798`) -- the `NON_IGNITED` occupant is the live one here and is written;
note this is a *different* node from `confinement_time.py`'s `PlasmaPowerLoss` family,
which answers the same switch for a different variable and whose live arm is `IGNITED`
on the stellarator runs. The two are not in conflict: they are two nodes, each declaring
its own arm.
"""

import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    OutputInto,
)

from functional_process.paths import current_drive, physics, times
from process.core import constants

# ---------------------------------------------------------------------------
# `plasma_fields.py::PlasmaFields.calculate_surface_averaged_poloidal_field`
# (`process/models/physics/plasma_fields.py:27-93`), Ampere's-law arm only.
# Not this file's own method; called unconditionally from `Physics.run`
# (`physics.py:313-324`) and the sole producer of one of this slot's eight outputs.
# ---------------------------------------------------------------------------


def calculate_surface_averaged_poloidal_field_amperes(cur_plasma, len_plasma_poloidal):
    """Surface-averaged poloidal field from Ampere's law, <Bp(a)> [T].

    Ports `PlasmaFields.calculate_surface_averaged_poloidal_field`'s
    `i_plasma_current != PENG_DIVERTOR_SCALING` arm, `plasma_fields.py:83-84`, unchanged.

    The whole arm is one line, and the point of declaring it separately is what it does
    **not** read: the source's signature takes eight arguments and this branch uses two
    of them. `q95`, `aspect`, `b_plasma_toroidal_on_axis`, `kappa` and `triang` are
    read only by the `PENG_DIVERTOR_SCALING` arm (`plasma_fields.py:86-93`), so a node
    declaring the union would claim five edges this configuration does not have.

    Parameters
    ----------
    cur_plasma :
        Plasma current (A). `.physics.plasma_current`.
    len_plasma_poloidal :
        Plasma poloidal perimeter (m). `.physics.len_plasma_poloidal`.

    Returns
    -------
    :
        Surface-averaged poloidal field (T).
    """
    return constants.RMU0 * cur_plasma / len_plasma_poloidal


# ---------------------------------------------------------------------------
# `Physics.run`'s radiation block (`physics.py:750-766`).
# ---------------------------------------------------------------------------


def calculate_unclipped_radiation_powers(
    pden_plasma_core_rad_mw_unclipped,
    pden_plasma_outer_rad_mw_unclipped,
    vol_plasma,
):
    """The tokamak's core/outer radiation densities and their volume integrals.

    Ports `physics.py:751-752` (two bare assignments off `calculate_radiation_powers`'s
    `RadpwrData`) and `physics.py:758-763` (the two products).

    **This is the tokamak half of a divergence between `calculate_radiation_powers`'s
    two callers, and the reason `PlasmaRadiationPowers` mints `_unclipped` names.**
    `stellarator.py:2153-2158` clips both densities at zero before forming the products;
    `physics.py:751-752` does not clip at all. The clip is therefore a property of one
    caller, not of the radiation model, and this function is the other caller: it is
    `models/stellarator/plasma_physics.py::calculate_clipped_radiation_powers` with the
    two `max(..., 0.0)` removed and nothing else changed. Confirmed by reading
    `physics.py:750-766` in full -- there is no `max`, no `jnp.maximum`, and no guard of
    any kind between the assignment and the products.

    A negative `pden_plasma_core_rad_mw` therefore propagates on a tokamak where the
    stellarator would have floored it, including into `.physics.p_plasma_inner_rad_mw`
    and (through `confinement_time.py`'s `power_loss`) into `.physics.p_plasma_loss_mw`.
    Ported faithfully; flagged in the record as a suspected PROCESS defect (**D1**),
    not fixed.

    Parameters
    ----------
    pden_plasma_core_rad_mw_unclipped :
        Core radiation power density (MW/m^3), as `PlasmaRadiationPowers` produces it.
        `.physics.pden_plasma_core_rad_mw_unclipped`.
    pden_plasma_outer_rad_mw_unclipped :
        Edge radiation power density (MW/m^3).
        `.physics.pden_plasma_outer_rad_mw_unclipped`.
    vol_plasma :
        Plasma volume (m^3). `.physics.vol_plasma`.

    Returns
    -------
    :
        `(pden_plasma_core_rad_mw, pden_plasma_outer_rad_mw, p_plasma_inner_rad_mw,
        p_plasma_outer_rad_mw)`, the first two in MW/m^3 and the last two in MW.
    """
    pden_plasma_core_rad_mw = pden_plasma_core_rad_mw_unclipped
    pden_plasma_outer_rad_mw = pden_plasma_outer_rad_mw_unclipped
    return (
        pden_plasma_core_rad_mw,
        pden_plasma_outer_rad_mw,
        pden_plasma_core_rad_mw * vol_plasma,
        pden_plasma_outer_rad_mw * vol_plasma,
    )


def calculate_total_radiation_power(pden_plasma_rad_mw, vol_plasma):
    """Total radiated power from the plasma (MW). Ports `physics.py:764-766`.

    A separate node from `calculate_unclipped_radiation_powers` above even though
    PROCESS writes all four products in one straight-line block, because its input is a
    different variable: `.physics.pden_plasma_rad_mw` is a **real** `DataStructure`
    field that `PlasmaRadiationPowers` owns directly (PROCESS never clips it -- see that
    node's docstring), where the core/outer pair arrive through the two `_unclipped`
    mints. Bundling them would give every consumer of `p_plasma_rad_mw` an edge from the
    mints it does not depend on.

    Parameters
    ----------
    pden_plasma_rad_mw :
        Total radiation power density (MW/m^3). `.physics.pden_plasma_rad_mw`.
    vol_plasma :
        Plasma volume (m^3). `.physics.vol_plasma`.

    Returns
    -------
    :
        Total radiated power (MW).
    """
    return pden_plasma_rad_mw * vol_plasma


# ---------------------------------------------------------------------------
# `exhaust.py::PlasmaExhaust.calculate_separatrix_power` plus `Physics.run`'s
# positivity transform (`physics.py:793-809` and `:839-845`).
# ---------------------------------------------------------------------------


def calculate_separatrix_power(
    f_p_alpha_plasma_deposited,
    p_alpha_total_mw,
    p_non_alpha_charged_mw,
    p_hcd_injected_total_mw,
    p_plasma_ohmic_mw,
    p_plasma_rad_mw,
):
    """Power crossing the separatrix, before the positivity transform (MW).

    Ports `PlasmaExhaust.calculate_separatrix_power`,
    `process/models/physics/exhaust.py:88-127`, unchanged. Out of `physics.py`'s own
    file scope and out of `exhaust.py`'s ported scope (`exhaust.md` records the other
    three `PlasmaExhaust` statics as deliberately not ported there), but it is the sole
    producer of one of this slot's eight outputs, so it is ported here. If `exhaust.py`'s
    scope is ever widened to cover it, one of the two copies must go -- there is no
    reason for both.

    Parameters
    ----------
    f_p_alpha_plasma_deposited :
        Fraction of alpha power deposited in the plasma.
        `.physics.f_p_alpha_plasma_deposited`.
    p_alpha_total_mw :
        Total alpha power (MW). `.physics.p_alpha_total_mw`.
    p_non_alpha_charged_mw :
        Non-alpha charged-particle power (MW). `.physics.p_non_alpha_charged_mw`.
    p_hcd_injected_total_mw :
        Injected heating and current-drive power (MW).
        `.current_drive.p_hcd_injected_total_mw`, or `0.0` on the ignited arm.
    p_plasma_ohmic_mw :
        Ohmic heating power (MW). `.physics.p_plasma_ohmic_mw`.
    p_plasma_rad_mw :
        Total radiated power (MW). `.physics.p_plasma_rad_mw`.

    Returns
    -------
    :
        Power crossing the separatrix (MW), which may be negative.
    """
    return (
        f_p_alpha_plasma_deposited * p_alpha_total_mw
        + p_non_alpha_charged_mw
        + p_hcd_injected_total_mw
        + p_plasma_ohmic_mw
        - p_plasma_rad_mw
    )


def force_positive_separatrix_power(p_plasma_separatrix_mw_raw):
    """PROCESS's positivity transform on the separatrix power (MW).

    Ports `physics.py:839-845` verbatim -- the source's own label is *"KLUDGE: Ensure
    p_plasma_separatrix_mw is continuously positive (physical, rather than negative
    potential power), as required by other models"*:

        p_plasma_separatrix_mw /= 1 - exp(-p_plasma_separatrix_mw)

    It is a smooth `softplus`-like map, not a clip: it is the identity to within
    `exp(-x)` for `x` of order a few, and maps negative `x` to a small positive number.

    **This is the second write to `.physics.p_plasma_separatrix_mw` in a single pass**,
    and it is why the pre-transform value is minted as
    `.physics.p_plasma_separatrix_mw_raw` rather than the two writes sharing one name.
    Three PROCESS call sites read the *pre*-transform value between the two writes --
    `calculate_psep_over_r_metric` (`physics.py:811-816`),
    `calculate_eu_demo_re_attachment_metric` (`:818-826`) and `ScrapeOffLayer.run`
    (`:832`) -- and every consumer after line 845 reads the post-transform one. A single
    node owning `.physics.p_plasma_separatrix_mw` and applying the transform inside would
    be correct for its own output and would silently hand the wrong one of the two values
    to those three. Same precedent, same shape, and the same reason as
    `radiation_power.py`'s `pden_plasma_core_rad_mw_unclipped` mint.

    Not a `jnp.where`-guarded domain: at exactly `x == 0` the source evaluates `0.0/0.0`
    and returns `nan` (a `RuntimeWarning`, not a raise), and the port reproduces that
    rather than inventing a limit PROCESS does not take.

    Parameters
    ----------
    p_plasma_separatrix_mw_raw :
        Separatrix power before the transform (MW).
        `.physics.p_plasma_separatrix_mw_raw`.

    Returns
    -------
    :
        Separatrix power after the transform (MW).
    """
    return p_plasma_separatrix_mw_raw / (1 - jnp.exp(-p_plasma_separatrix_mw_raw))


# ---------------------------------------------------------------------------
# `Physics.run`'s PF-coil ramp-time block (`physics.py:463-498`).
# ---------------------------------------------------------------------------


def calculate_pulsed_plant_ramp_times(plasma_current):
    """Plasma-current ramp-up and ramp-down times for a pulsed plant (s).

    Ports `physics.py:476-483`, the `i_pulsed_plant == 1 and pulsetimings == 0` arm:

        t_plant_pulse_plasma_current_ramp_up   = plasma_current / 1.0e5
        t_plant_pulse_plasma_current_ramp_down = t_plant_pulse_plasma_current_ramp_up

    `.times.t_plant_pulse_coil_precharge` is **not** written on this arm -- the source's
    own comment at `:477` says it is an input -- which is exactly what distinguishes it
    from the other two arms and why the switch is a split rather than a static kwarg.

    Parameters
    ----------
    plasma_current :
        Plasma current (A). `.physics.plasma_current`.

    Returns
    -------
    :
        `(t_plant_pulse_plasma_current_ramp_up,
        t_plant_pulse_plasma_current_ramp_down)`, both in seconds.
    """
    t_plant_pulse_plasma_current_ramp_up = plasma_current / 1.0e5
    return (
        t_plant_pulse_plasma_current_ramp_up,
        t_plant_pulse_plasma_current_ramp_up,
    )


# ---------------------------------------------------------------------------
# `PlasmaBeta.calculate_plasma_energy_from_beta` (`physics.py:4153-4176`).
# ---------------------------------------------------------------------------


def calculate_plasma_energy_from_beta(beta, b_field, vol_plasma):
    """Plasma stored energy derived from beta (J).

    Ports `PlasmaBeta.calculate_plasma_energy_from_beta`, `physics.py:4153-4176`,
    unchanged:

        E = 1.5 * beta * B^2 / (2 * mu_0) * V

    Parameters
    ----------
    beta :
        Plasma beta (dimensionless).
    b_field :
        Magnetic field (T).
    vol_plasma :
        Plasma volume (m^3). `.physics.vol_plasma`.

    Returns
    -------
    :
        Plasma energy (J).
    """
    return (1.5e0 * beta * b_field**2) / (2.0e0 * constants.RMU0) * vol_plasma


# ---------------------------------------------------------------------------
# cottax nodes
# ---------------------------------------------------------------------------


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
