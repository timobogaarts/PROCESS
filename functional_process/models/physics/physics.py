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


def calculate_continuous_plant_ramp_times(plasma_current):
    """Plasma-current ramp times for a continuous (non-pulsed) plant (s).

    Ports `physics.py:465-474`, the `i_pulsed_plant != 1 and i_t_current_ramp_up == 0`
    arm, unchanged:

        t_plant_pulse_plasma_current_ramp_up   = plasma_current / 5.0e5
        t_plant_pulse_coil_precharge           = t_plant_pulse_plasma_current_ramp_up
        t_plant_pulse_plasma_current_ramp_down = t_plant_pulse_plasma_current_ramp_up

    Unlike the pulsed-default arm (`calculate_pulsed_plant_ramp_times`, `:476-483`),
    this arm *does* write `.times.t_plant_pulse_coil_precharge` -- the third output --
    which is exactly why the two are separate occupants rather than one function with a
    literal swapped (`5e5` vs `1e5`): the write-sets differ, not just a constant.

    Parameters
    ----------
    plasma_current :
        Plasma current (A). `.physics.plasma_current`.

    Returns
    -------
    :
        `(t_plant_pulse_plasma_current_ramp_up, t_plant_pulse_coil_precharge,
        t_plant_pulse_plasma_current_ramp_down)`, all in seconds, all equal --
        PROCESS's write order at `physics.py:466-474`.
    """
    t_plant_pulse_plasma_current_ramp_up = plasma_current / 5.0e5
    return (
        t_plant_pulse_plasma_current_ramp_up,
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
# `Physics.plasma_ohmic_heating` (`physics.py:1605-1697`), written back at
# `Physics.run` `:768-778`. Added 2026-08-27: `cold_boundary.md` producer 3 --
# `.physics.res_plasma` was the boundary zero that (with `vs_cs_pf_total_burn`)
# made the cold `pulse.burn_time` nan, via `v_plasma_loop_burn = plasma_current *
# res_plasma * f_c_plasma_inductive` (`plasma_inductance.volt_seconds`).
# ---------------------------------------------------------------------------


def plasma_ohmic_heating(
    f_c_plasma_inductive,
    kappa95,
    plasma_current,
    rmajor,
    rminor,
    temp_plasma_electron_density_weighted_kev,
    vol_plasma,
    zeff,
    plasma_res_factor,
):
    """Ohmic heating power and plasma resistance (IPDG89).

    Ports `Physics.plasma_ohmic_heating`, `physics.py:1605-1697`, term for term --
    **including its live defect**. PROCESS's neo-classical enhancement guard is the
    chained comparison `1.0 if 2.5 >= rmajor / rminor <= 4.0 else 4.3 - 0.6 * rmajor /
    rminor` (`physics.py:1675`), which Python reads as `(2.5 >= A) and (A <= 4.0)`,
    i.e. `A <= 2.5` -- **not** the documented "aspect ratios in the range 2.5 to 4.0".
    Reproduced exactly as `jnp.where(A <= 2.5, ...)`; on `large_tokamak_eval`
    (`A = 3.1`) the enhancement arm is taken either way.

    Two shells are dropped, neither of them arithmetic: the `aspect` parameter (read
    only by the negative-resistance `logger.error`, `physics.py:1682-1685` -- a traced
    function cannot log on a data-dependent condition, and the message's own value is
    unused), and that logger itself.

    Parameters
    ----------
    f_c_plasma_inductive :
        Fraction of plasma current driven inductively.
        `.physics.f_c_plasma_inductive`.
    kappa95 :
        Plasma elongation at the 95% surface. `.physics.kappa95`.
    plasma_current :
        Plasma current (A). `.physics.plasma_current`.
    rmajor, rminor :
        Plasma major/minor radius (m). `.physics.rmajor`, `.physics.rminor`.
    temp_plasma_electron_density_weighted_kev :
        Density-weighted average electron temperature (keV).
        `.physics.temp_plasma_electron_density_weighted_kev`.
    vol_plasma :
        Plasma volume (m^3). `.physics.vol_plasma`.
    zeff :
        Plasma effective charge (the staticmethod's own spelling; the storage field is
        `.physics.n_charge_plasma_effective_vol_avg`, `Physics.run` `:782`, and the
        node port uses that name per the declaration-surface rule).
    plasma_res_factor :
        Plasma resistivity pre-factor. `.physics.plasma_res_factor`.

    Returns
    -------
    tuple
        `(pden_plasma_ohmic_mw, p_plasma_ohmic_mw, f_res_plasma_neo, res_plasma)` --
        MW/m^3, MW, dimensionless, ohm; `.physics.` all four (`physics.py:768-773`).
    """
    t10 = temp_plasma_electron_density_weighted_kev / 10.0

    res_plasma = (
        plasma_res_factor * 2.15e-9 * zeff * rmajor / (kappa95 * rminor**2 * t10**1.5)
    )

    # PROCESS's chained comparison, reproduced -- see the docstring.
    f_res_plasma_neo = jnp.where(
        rmajor / rminor <= 2.5, 1.0, 4.3 - 0.6 * rmajor / rminor
    )

    res_plasma = res_plasma * f_res_plasma_neo

    pden_plasma_ohmic_mw = (
        f_c_plasma_inductive * plasma_current**2 * res_plasma * 1.0e-6 / vol_plasma
    )

    p_plasma_ohmic_mw = pden_plasma_ohmic_mw * vol_plasma

    return pden_plasma_ohmic_mw, p_plasma_ohmic_mw, f_res_plasma_neo, res_plasma


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


# ---------------------------------------------------------------------------
# `PlasmaBeta.run`'s beta-limit block (`physics.py:3789-3835`).
#
# Added 2026-08-27 for `optimise_design.md` §11.5's constraint-24 rows: all three
# of `.physics.beta_thermal_vol_avg`, `.beta_toroidal_vol_avg` and
# `.beta_vol_avg_max` were boundary zeros that PROCESS's own solve moves, so
# `constraint_24` compared a frozen 0 against a frozen 0 and its whole gradient
# row was structurally dead.
# ---------------------------------------------------------------------------


def calculate_beta_norm_max_wesson(ind_plasma_internal_norm):
    """Wesson's normalised beta upper limit, beta_N_max.

    Ports `PlasmaBeta.calculate_beta_norm_max_wesson`, `physics.py:3941-3974`,
    unchanged -- the whole body is `4 * l_i`.

    Parameters
    ----------
    ind_plasma_internal_norm :
        Plasma normalised internal inductance. `.physics.ind_plasma_internal_norm`.

    Returns
    -------
    :
        Wesson normalised beta upper limit.
    """
    return 4 * ind_plasma_internal_norm


def calculate_beta_limit_from_norm(
    b_plasma_toroidal_on_axis,
    beta_norm_max,
    plasma_current,
    rminor,
):
    """Maximum allowed volume-averaged beta, from the normalised limit.

    Ports `PlasmaBeta.calculate_beta_limit_from_norm`, `physics.py:4180-4235`,
    unchanged (AEA FUS 172). The `0.01` converts the Troyon coefficient from per-cent
    to a fraction.

    **This node owns `.physics.beta_vol_avg_max` and nothing selects among components
    here.** `.physics.i_beta_component` chooses which beta the *constraint* compares
    against the limit (`constraint_24`), not which limit is computed -- PROCESS
    computes exactly this one limit whatever the switch says, and the switch is already
    a static kwarg of the ported `constraint_24`.

    Parameters
    ----------
    b_plasma_toroidal_on_axis :
        Toroidal field on the plasma axis (T).
    beta_norm_max :
        Troyon-like g coefficient. `.physics.beta_norm_max`.
    plasma_current :
        Plasma current (A).
    rminor :
        Plasma minor radius (m).

    Returns
    -------
    :
        Volume-averaged beta limit (dimensionless).
    """
    return (
        0.01
        * beta_norm_max
        * (plasma_current / 1.0e6)
        / (rminor * b_plasma_toroidal_on_axis)
    )


def calculate_toroidal_beta(
    beta_total_vol_avg,
    b_plasma_total,
    b_plasma_toroidal_on_axis,
):
    """Volume-averaged beta referred to the toroidal field alone.

    Ports `physics.py:3818-3822` -- an inline assignment in `PlasmaBeta.run` with no
    `@staticmethod` of its own, transcribed term for term.

    Parameters
    ----------
    beta_total_vol_avg :
        Volume-averaged total beta, referred to the total field.
    b_plasma_total :
        Total field on axis (T).
    b_plasma_toroidal_on_axis :
        Toroidal field on axis (T).

    Returns
    -------
    :
        Toroidal beta (dimensionless).
    """
    return beta_total_vol_avg * b_plasma_total**2 / b_plasma_toroidal_on_axis**2


def calculate_poloidal_beta(b_plasma_total, b_plasma_poloidal_average, beta):
    """Volume-averaged beta referred to the poloidal field alone.

    Ports `Physics.calculate_poloidal_beta` (`physics.py:4239-4263`), called from
    `physics.py:3825` -- the one line of `PlasmaBeta.run`'s 3818-3835 block this slot
    skipped, sitting between `ToroidalBeta` (3818-3822) and `ThermalBeta` (3831-3835).

    **This was a known hole and it was load-bearing.** `constraint_48`'s docstring has
    recorded since `batch5.md` that "`beta_poloidal_vol_avg`'s real producer
    (`Physics.calculate_poloidal_beta`, `physics.py:3825`) is not yet ported anywhere in
    `functional_process`", and ported the constraint over the unproduced read anyway.
    The read is not only constraint 48's: `models/pfcoil/currents.py::
    calculate_equilibrium_currents` puts it inside `log(8*aspect) + beta_poloidal_vol_avg
    + l_i/2 - 1.5`, the bracket that sets the **equilibrium PF coil currents**. With no
    producer the term was `0.0` against PROCESS's `1.0874` on `large_tokamak_nof` -- an
    O(1) error in an O(1) bracket, propagating through the coil flux to the volt-second
    balance, the burn time (55x), the CS field and finally `stress_shear_cs_peak` (708x),
    which is constraint 72 and which is *active* at PROCESS's optimum. See
    `_audit/optimise_design.md` §16.

    References
    ----------
    - J.P. Freidberg, "Plasma physics and fusion energy", Cambridge University Press
      (2007) Page 270 ISBN 0521851076

    Parameters
    ----------
    b_plasma_total :
        Total field on axis (T).
    b_plasma_poloidal_average :
        Surface-averaged poloidal field (T).
    beta :
        Volume-averaged total beta, referred to the total field.

    Returns
    -------
    :
        Poloidal beta (dimensionless).
    """
    return beta * (b_plasma_total / b_plasma_poloidal_average) ** 2


def calculate_thermal_beta(beta_total_vol_avg, beta_fast_alpha, beta_beam):
    """Volume-averaged thermal beta: the total less both fast-ion contributions.

    Ports `physics.py:3831-3835`, an inline assignment in `PlasmaBeta.run`.

    Parameters
    ----------
    beta_total_vol_avg :
        Volume-averaged total beta.
    beta_fast_alpha :
        Fast-alpha beta contribution. `.physics.beta_fast_alpha`.
    beta_beam :
        Neutral-beam fast-ion beta contribution. `.physics.beta_beam`.

    Returns
    -------
    :
        Thermal beta (dimensionless).
    """
    return beta_total_vol_avg - beta_fast_alpha - beta_beam


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
