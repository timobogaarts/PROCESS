"""Pure-functional port of `process/models/physics/current_drive.py`'s `CurrentDrive`.

Audit record: `functional_process/_audit/units/models/physics/current_drive.md`. Read it
first -- especially "A live PROCESS bug in two sibling arms" (`calculate_profile_y`
returns `None`) and "What this unit does *not* port and why", which between them say
exactly which of the eleven `i_hcd_primary` values this file can answer.

**Scope: the minimal closure that produces `_audit/tokamak_boundary.md`'s three
`.tokamak.current_drive` boundary reads** --
`.current_drive.p_hcd_ecrh_injected_total_mw`, `.current_drive.p_hcd_injected_total_mw`
and `.heat_transport.p_hcd_electric_total_mw` -- for the combinations the tracked
tokamak input files actually hold. `large_tokamak_eval.IN.DAT`'s: `i_hcd_primary = 10`
(`USER_INPUT_ELECTRON_CYCLOTRON`, the file's line 124), `i_hcd_secondary = 0`
(`NO_CURRENT_DRIVE`, PROCESS's default at `current_drive_variables.py:206` -- the file
never sets it), `i_hcd_calculations = 1` (default, `:223`) and `i_plasma_ignited = 0`
(`NON_IGNITED`, `physics_variables.py:881`). And, since 2026-08-27, the two spherical
tokamak files' (`spherical_tokamak_eval.IN.DAT:133`, `st_regression.IN.DAT:2522`):
`i_hcd_primary = 13` (`FREETHY_ELECTRON_CYCLOTRON`) with its nested `i_ecrh_wave_mode =
0` (O-mode -- both files set it explicitly, and it is also PROCESS's default at
`current_drive_variables.py:116`), same secondary/calculations/ignition values. The two
arms share every stage but the first: `CurrentDriveModel(13).method` is
`ELECTRON_CYCLOTRON`, so the wall-plug block is the one already ported. Every other
heating-and-current-drive scheme is UNPORTED; see the audit record for the per-value
reason.

`CurrentDrive.current_drive` (`process/models/physics/current_drive.py:1651-2309`) is one
660-line method in which **four** switches interleave -- `i_hcd_calculations` gates the
whole body, `i_hcd_primary` picks both an efficiency formula and a wall-plug block,
`i_hcd_secondary` picks the same two things again for the second system, and
`i_plasma_ignited` decides whether the wall-plug total survives. Splitting it is the same
move `confinement_time.py` made: one class per switch value this port supports, each
declaring only the reads its own arm makes, so the graph stops claiming edges the run
does not have. Two of the seven stages below turn out to be switch-*independent* pure
algebra once the arms are peeled off, and they are single nodes.

`i_plasma_ignited` is a switch (`_audit/naming_convention.md` § "switches are not
ports"): a plain Python int used for ordinary branching in the composite
`calculate_current_drive_ecrh_primary_no_secondary` below, never traced. The harness
marks it `static_argnames`.
"""

import jax
import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    OutputInto,
)

from functional_process.models.stated import StatesValues
from functional_process.paths import current_drive, heat_transport, physics
from functional_process.vocabulary import constants
from functional_process.vocabulary import PlasmaIgnitionModel

# ---------------------------------------------------------------------------
# Stage 1 -- the current drive efficiency of the primary heating system.
# `i_hcd_primary` selects one of eleven formulas; `hcd_models` in the source
# (`current_drive.py:1697-1771`) is a dict of eleven lambdas, of which exactly one is
# ever called. Each reads a different set of variables, which is why each is its own
# function here rather than one function behind a switch argument.
# ---------------------------------------------------------------------------


def user_input_electron_cyclotron_efficiency(
    *, eta_cd_norm_ecrh, nd_plasma_electrons_vol_avg, rmajor
):
    """ECRH current drive efficiency from a user-supplied normalised gamma.

    Ports `hcd_models[10]`, `process/models/physics/current_drive.py:1744-1747`, plus the
    `dene20` conversion its enclosing scope computes once at `:1690`.
    `CurrentDriveModel.USER_INPUT_ELECTRON_CYCLOTRON` (10), the only `i_hcd_primary`
    value this port answers.

    Three reads, and that is the whole model -- no plasma profile, no Coulomb logarithm,
    no local temperature. That is worth stating because it is *why* this value is the one
    ported first: the ten other arms of `hcd_models` reach `culecd`/`cullhy`/`culnbi`/
    `iternb` and through them the profile machinery, and two of those are broken in
    PROCESS itself (see the audit record).
    """
    dene20 = nd_plasma_electrons_vol_avg * 1.0e-20
    return eta_cd_norm_ecrh / (dene20 * rmajor)


def freethy_electron_cyclotron_efficiency(
    *,
    temp_plasma_electron_vol_avg_kev,
    n_charge_plasma_effective_vol_avg,
    rmajor,
    nd_plasma_electrons_vol_avg,
    b_plasma_toroidal_on_axis,
    n_ecrh_harmonic,
    feffcd,
    i_ecrh_wave_mode,
):
    """ECCD efficiency from the Freethy model (PROCESS issue #2994, GRAY-tuned).

    Ports `ElectronCyclotron.electron_cyclotron_freethy`
    (`process/models/physics/current_drive.py:992-1088`) plus the `* feffcd` factor its
    enclosing lambda applies (`hcd_models[13]`, `:1759-1770`).
    `CurrentDriveModel.FREETHY_ELECTRON_CYCLOTRON` (13),
    `spherical_tokamak_eval.IN.DAT:133` and `st_regression.IN.DAT:2522`'s value.

    The parameter names are the `VarPath` leaves the lambda reads, not the
    staticmethod's abbreviations: its `te` is `.physics.temp_plasma_electron_vol_avg_kev`
    and its `zeff` is `.physics.n_charge_plasma_effective_vol_avg` (`:1760-1767`).

    `i_ecrh_wave_mode` is the **nested switch** (`:1074-1079`): it selects the cut-off
    frequency formula -- O-mode (`0`) takes the plasma frequency, X-mode (`1`) the
    right-hand cut-off -- and an invalid value raises `ValueError`, exactly as the
    reference does. It is a static Python int, never traced, on the same terms as
    `i_plasma_ignited` in `hcd_electric_total_mw` below: the occupant pins it, and the
    two branches read **provably identical** variable sets (both cut-offs are formed
    from `fc` and `fp`, which are already computed from the same two reads), which is
    `traceability_policy.md`'s static-kwarg exception -- the one switch in this unit
    that does *not* split into per-value functions, and the reads-set identity is the
    required evidence, asserted in `test_current_drive.py`. Only the O-mode branch has
    an occupant; X-mode is live on no tracked input (both files that select model 13
    set `0`, which is also the default, `current_drive_variables.py:116`) and its
    binding is refused in `indat.py`'s `UNPORTED` -- the branch is transcribed here
    because splitting one line out of a shared twenty would be the duplication the
    policy's deviation clause exists for, and it is value-checked against the
    reference, not assumed.

    Note the X-mode cut-off is transcribed defect-preservingly: the reference puts
    `n_ecrh_harmonic` on `fc**2` *inside* the square root (`:1077`), i.e.
    `sqrt(n * fc**2 + 4 * fp**2)` and not `sqrt((n * fc)**2 + 4 * fp**2)`, which for
    the right-hand cut-off one would expect. Ported as written.

    Raises
    ------
    ValueError
        If the wave mode is invalid (not 0 for O-mode or 1 for X-mode) -- the
        reference's own error, transcribed.
    """
    # Cyclotron frequency (`:1046-1052`).
    fc = (
        1
        / (2 * jnp.pi)
        * constants.ELECTRON_CHARGE
        * b_plasma_toroidal_on_axis
        / constants.ELECTRON_MASS
    )

    # Plasma frequency (`:1055-1064`).
    fp = (
        1
        / (2 * jnp.pi)
        * jnp.sqrt(
            nd_plasma_electrons_vol_avg
            * constants.ELECTRON_CHARGE**2
            / (constants.ELECTRON_MASS * constants.EPSILON0)
        )
    )

    # Scaling factor for ECCD efficiency: 0.18 tuned to a GRAY study, then the Zeff
    # correction (`:1067-1068`).
    xi_cd = 0.18e0 * (4.8e0 / (2 + n_charge_plasma_effective_vol_avg))

    # ECCD efficiency before the coupling factor (`:1071`).
    eta_cd = (
        xi_cd
        * temp_plasma_electron_vol_avg_kev
        / (3.27e0 * rmajor * (nd_plasma_electrons_vol_avg / 1.0e19))
    )

    # The nested switch: cut-off frequency by wave mode (`:1074-1079`).
    if i_ecrh_wave_mode == 0:  # O-mode case
        f_cutoff = fp
    elif i_ecrh_wave_mode == 1:  # X-mode case
        f_cutoff = 0.5 * (fc + jnp.sqrt(n_ecrh_harmonic * fc**2 + 4 * fp**2))
    else:
        raise ValueError("Invalid wave mode. Use 0 for O-mode or 1 for X-mode.")

    # Plasma coupling factor: no coupling unless the cut-off is below the cyclotron
    # harmonic, smoothed by a tanh of sharpness `a` (`:1081-1085`).
    a = 0.1
    cutoff_factor = 0.5 * (
        1 + jnp.tanh((2 / a) * ((n_ecrh_harmonic * fc - f_cutoff) / fp - a))
    )

    return eta_cd * cutoff_factor * feffcd


# ---------------------------------------------------------------------------
# Stage 2 -- the secondary heating system's contribution, `i_hcd_secondary`.
# ---------------------------------------------------------------------------


def hcd_secondary_driven_current(
    *, eta_cd_hcd_secondary, p_hcd_secondary_injected_mw, plasma_current
):
    """Current driven by the secondary system, and its share of the plasma current.

    Ports `process/models/physics/current_drive.py:1821-1831`, unchanged. **Switch
    independent**: the source computes these two lines outside every `if`, from
    `eta_cd_hcd_secondary` -- which is what the `i_hcd_secondary` arms decide -- so once
    that efficiency is a variable the arms have already produced, this is ordinary
    algebra with no arm of its own.

    Returns
    -------
    tuple
        `(c_hcd_secondary_driven, f_c_plasma_hcd_secondary)`.
    """
    c_hcd_secondary_driven = eta_cd_hcd_secondary * p_hcd_secondary_injected_mw * 1.0e6
    return c_hcd_secondary_driven, c_hcd_secondary_driven / plasma_current


# ---------------------------------------------------------------------------
# Stage 3 -- the primary system's injected power. Switch independent.
# ---------------------------------------------------------------------------


def hcd_primary_injected_power_mw(
    *,
    f_c_plasma_auxiliary,
    f_c_plasma_hcd_secondary,
    plasma_current,
    eta_cd_hcd_primary,
):
    """Injected power the primary system needs to drive the auxiliary current fraction.

    Ports `process/models/physics/current_drive.py:1834-1842`, unchanged. The primary
    system is the *residual*: it supplies whatever share of the auxiliary current
    fraction the secondary did not, which is why this reads
    `f_c_plasma_hcd_secondary` and not the secondary's power directly.
    """
    return (
        1.0e-6
        * (f_c_plasma_auxiliary - f_c_plasma_hcd_secondary)
        * plasma_current
        / eta_cd_hcd_primary
    )


# ---------------------------------------------------------------------------
# Stage 4 -- the primary system's wall-plug block. Selected by the primary model's
# *method* (`CurrentDriveModel.method`), not by its value: `current_drive.py:2131`
# branches on `primary_cdm.method == ELECTRON_CYCLOTRON`, so one block serves all four
# ECRH models (3, 7, 10, 13). The ECRH injected-power accumulator it feeds is also
# written by the *secondary* ECRH block (`:1955`), which is why the occupant below is
# keyed on the pair and not on `i_hcd_primary` alone.
# ---------------------------------------------------------------------------


def electron_cyclotron_primary_powers(
    *,
    p_hcd_ecrh_injected_secondary_mw,
    p_hcd_primary_injected_mw,
    p_hcd_primary_extra_heat_mw,
    eta_ecrh_injector_wall_plug,
):
    """The ECRH primary block: wall-plug power and the ECRH injected-power total.

    Ports `process/models/physics/current_drive.py:2131-2156`. The source forms the sum
    `p_hcd_primary_injected_mw + p_hcd_primary_extra_heat_mw` three separate times
    (`:2132`, `:2139`, `:2148`) from the same two unchanged values; it is computed once
    here, which is the same number in float64 and not an approximation.

    `p_hcd_ecrh_injected_secondary_mw` is what the secondary system already added to
    `.current_drive.p_hcd_ecrh_injected_total_mw` before this block's `+=` at `:2147`
    (the accumulator is zeroed at `:1663`). It has no `DataStructure` field of its own --
    PROCESS accumulates in place -- so it is a plain argument, and the only occupant
    written below passes `0.0` for it because its arm's secondary system is
    `NO_CURRENT_DRIVE`.

    Returns
    -------
    tuple
        `(p_hcd_ecrh_injected_total_mw, p_hcd_ecrh_electric_mw,
        eta_hcd_primary_injector_wall_plug, p_hcd_primary_electric_mw)`.
    """
    p_hcd_primary_total_mw = p_hcd_primary_injected_mw + p_hcd_primary_extra_heat_mw

    p_hcd_primary_electric_mw = p_hcd_primary_total_mw / eta_ecrh_injector_wall_plug
    p_hcd_ecrh_injected_total_mw = (
        p_hcd_ecrh_injected_secondary_mw + p_hcd_primary_total_mw
    )
    p_hcd_ecrh_electric_mw = p_hcd_ecrh_injected_total_mw / eta_ecrh_injector_wall_plug
    return (
        p_hcd_ecrh_injected_total_mw,
        p_hcd_ecrh_electric_mw,
        eta_ecrh_injector_wall_plug,
        p_hcd_primary_electric_mw,
    )


# ---------------------------------------------------------------------------
# Stage 5 and 6 -- the totals. Switch independent given the arms' own outputs, except
# for `i_plasma_ignited`, which decides the wall-plug total outright.
# ---------------------------------------------------------------------------


def hcd_injected_power_total_mw(
    *,
    p_hcd_primary_injected_mw,
    p_hcd_primary_extra_heat_mw,
    p_hcd_secondary_injected_mw,
    p_hcd_secondary_extra_heat_mw,
):
    """Total injected power that contributed to heating.

    Ports `process/models/physics/current_drive.py:2265-2270`, unchanged. This is the
    variable `_audit/tokamak_boundary.md` § "What blocked the real file" names as the one
    the NON_IGNITED confinement head also reads -- see
    `confinement_time.py::PlasmaPowerLossNonIgnitedCoreRadiation`.
    """
    return (
        p_hcd_primary_injected_mw
        + p_hcd_primary_extra_heat_mw
        + p_hcd_secondary_injected_mw
        + p_hcd_secondary_extra_heat_mw
    )


def hcd_electric_total_mw(
    *, p_hcd_primary_electric_mw, p_hcd_secondary_electric_mw, i_plasma_ignited
):
    """Total wall plug power for all heating systems.

    Ports `process/models/physics/current_drive.py:2289-2299`, including the
    `i_plasma_ignited` reset the source's own comment calls a *"fudge"*: on an ignited
    plasma the whole electrical total is discarded and replaced by zero, so the two
    reads above are not reads at all on that arm. That is exactly the invented edge the
    occupant split exists to remove, and here it removes two -- see
    `HcdElectricTotalIgnited`, which reads nothing.
    """
    if PlasmaIgnitionModel(int(i_plasma_ignited)) == PlasmaIgnitionModel.IGNITED:
        return 0.0
    return p_hcd_primary_electric_mw + p_hcd_secondary_electric_mw


def fusion_gain(
    *,
    p_fusion_total_mw,
    p_hcd_injected_total_mw,
    p_beam_orbit_loss_mw,
    p_plasma_ohmic_mw,
):
    """Fusion gain `Q`: fusion power over input (injection + orbit loss + ohmic) power.

    Ports `process/models/physics/current_drive.py:2301-2308`, unchanged and unguarded
    -- the source divides straight, and the stellarator's own `st_heat` (ported at
    `models/stellarator/heating.py::calculate_fusion_gain`) is the one that carries a
    `< 1e-6 -> 1e18` degenerate guard. **The two devices' formulas are otherwise
    identical**, which is why this is a two-line function and not a model: what was
    missing on the tokamak graph was the *node*, not the physics.

    The last statement of `CurrentDrive.current_drive`, and the last line of the
    `i_hcd_calculations != 0` body -- so on the `= 0` arm PROCESS does not write it at
    all, exactly as `TokamakCurrentDrive`'s docstring says of every slot there.

    `_audit/units/models/physics/current_drive.md`'s data-footprint table has carried
    the two `.physics` reads as *"outside this unit's closure -- feed only
    `big_q_plasma`"* since the unit was written; this closes that. The reason it was
    left out then was that no node in the graph read `.current_drive.big_q_plasma`;
    `st_regression.IN.DAT`'s `i_figure_merit = -5` (`FUSION_GAIN_Q`) is the reader that
    arrived, and its absence is what `_audit/optimise_design.md` §26/§27.4 is about.

    Parameters
    ----------
    p_fusion_total_mw :
        Total fusion power (MW). `.physics.p_fusion_total_mw`.
    p_hcd_injected_total_mw :
        Total injected heating/current-drive power (MW).
        `.current_drive.p_hcd_injected_total_mw`.
    p_beam_orbit_loss_mw :
        Neutral-beam orbit-loss power (MW). `.current_drive.p_beam_orbit_loss_mw`.
        Zeroed by the source at `:1668-1669` on every arm this unit ports (no neutral
        beam), and a boundary input of this graph rather than an output of any node --
        the audit record's "a node declaring them would be asserting five switch values
        at once" still holds.
    p_plasma_ohmic_mw :
        Ohmic heating power (MW). `.physics.p_plasma_ohmic_mw`.

    Returns
    -------
    :
        `big_q_plasma`.
    """
    return p_fusion_total_mw / (
        p_hcd_injected_total_mw + p_beam_orbit_loss_mw + p_plasma_ohmic_mw
    )


# ---------------------------------------------------------------------------
# The composite -- one function reproducing the whole of `CurrentDrive.current_drive`
# for the arm this port supports, so there is a boundary PROCESS itself has to diff
# against. `TestCurrentDriveEcrhPrimaryNoSecondary` calls it against a real
# `CurrentDrive` bound to a `DataStructure`, sample by sample, values and gradients.
# The node split below is finer than anything PROCESS exposes; this is where the two
# meet, the same trade `confinement_time.py::plasma_power_loss_mw` records.
# ---------------------------------------------------------------------------


def calculate_current_drive_ecrh_primary_no_secondary(
    *,
    eta_cd_norm_ecrh,
    nd_plasma_electrons_vol_avg,
    rmajor,
    plasma_current,
    f_c_plasma_auxiliary,
    p_hcd_primary_extra_heat_mw,
    p_hcd_secondary_injected_mw,
    eta_ecrh_injector_wall_plug,
    i_plasma_ignited,
):
    """`CurrentDrive.current_drive` for `i_hcd_primary = 10`, `i_hcd_secondary = 0`.

    Ports `process/models/physics/current_drive.py:1651-2309` restricted to the arm
    `large_tokamak_eval.IN.DAT` selects, with `i_hcd_calculations = 1`. Written as a
    composition of the stage functions above so there is exactly one source of truth for
    each formula and the node occupants cannot drift from what this is diffed against.

    Three of the source's writes are **not** recomputed here because on this arm PROCESS
    does not compute them either -- `.current_drive.eta_cd_hcd_secondary`,
    `.current_drive.p_hcd_secondary_extra_heat_mw` (forced to `0.0` at `:1682`) and
    `.heat_transport.p_hcd_secondary_electric_mw` are all zero, and
    `HcdSecondaryHeatingNone` below is the node that says so. They appear here as
    literals, and the audit record's data footprint table carries the evidence for each.

    Returns
    -------
    tuple
        `(eta_cd_hcd_primary, c_hcd_secondary_driven, f_c_plasma_hcd_secondary,
        p_hcd_primary_injected_mw, p_hcd_ecrh_injected_total_mw, p_hcd_ecrh_electric_mw,
        eta_hcd_primary_injector_wall_plug, p_hcd_primary_electric_mw,
        p_hcd_injected_total_mw, p_hcd_electric_total_mw)`.
    """
    eta_cd_hcd_primary = user_input_electron_cyclotron_efficiency(
        eta_cd_norm_ecrh=eta_cd_norm_ecrh,
        nd_plasma_electrons_vol_avg=nd_plasma_electrons_vol_avg,
        rmajor=rmajor,
    )

    c_hcd_secondary_driven, f_c_plasma_hcd_secondary = hcd_secondary_driven_current(
        eta_cd_hcd_secondary=0.0,
        p_hcd_secondary_injected_mw=p_hcd_secondary_injected_mw,
        plasma_current=plasma_current,
    )

    p_hcd_primary_injected_mw = hcd_primary_injected_power_mw(
        f_c_plasma_auxiliary=f_c_plasma_auxiliary,
        f_c_plasma_hcd_secondary=f_c_plasma_hcd_secondary,
        plasma_current=plasma_current,
        eta_cd_hcd_primary=eta_cd_hcd_primary,
    )

    (
        p_hcd_ecrh_injected_total_mw,
        p_hcd_ecrh_electric_mw,
        eta_hcd_primary_injector_wall_plug,
        p_hcd_primary_electric_mw,
    ) = electron_cyclotron_primary_powers(
        p_hcd_ecrh_injected_secondary_mw=0.0,
        p_hcd_primary_injected_mw=p_hcd_primary_injected_mw,
        p_hcd_primary_extra_heat_mw=p_hcd_primary_extra_heat_mw,
        eta_ecrh_injector_wall_plug=eta_ecrh_injector_wall_plug,
    )

    p_hcd_injected_total_mw = hcd_injected_power_total_mw(
        p_hcd_primary_injected_mw=p_hcd_primary_injected_mw,
        p_hcd_primary_extra_heat_mw=p_hcd_primary_extra_heat_mw,
        p_hcd_secondary_injected_mw=p_hcd_secondary_injected_mw,
        p_hcd_secondary_extra_heat_mw=0.0,
    )

    p_hcd_electric_total_mw = hcd_electric_total_mw(
        p_hcd_primary_electric_mw=p_hcd_primary_electric_mw,
        p_hcd_secondary_electric_mw=0.0,
        i_plasma_ignited=i_plasma_ignited,
    )

    return (
        eta_cd_hcd_primary,
        c_hcd_secondary_driven,
        f_c_plasma_hcd_secondary,
        p_hcd_primary_injected_mw,
        p_hcd_ecrh_injected_total_mw,
        p_hcd_ecrh_electric_mw,
        eta_hcd_primary_injector_wall_plug,
        p_hcd_primary_electric_mw,
        p_hcd_injected_total_mw,
        p_hcd_electric_total_mw,
    )


def calculate_current_drive_freethy_ecrh_primary_no_secondary(
    *,
    temp_plasma_electron_vol_avg_kev,
    n_charge_plasma_effective_vol_avg,
    nd_plasma_electrons_vol_avg,
    rmajor,
    b_plasma_toroidal_on_axis,
    n_ecrh_harmonic,
    feffcd,
    plasma_current,
    f_c_plasma_auxiliary,
    p_hcd_primary_extra_heat_mw,
    p_hcd_secondary_injected_mw,
    eta_ecrh_injector_wall_plug,
    i_plasma_ignited,
    i_ecrh_wave_mode,
):
    """`CurrentDrive.current_drive` for `i_hcd_primary = 13`, `i_hcd_secondary = 0`.

    The spherical tokamak files' arm (`spherical_tokamak_eval.IN.DAT:133`,
    `st_regression.IN.DAT:2522`), with `i_hcd_calculations = 1`. Identical to
    `calculate_current_drive_ecrh_primary_no_secondary` in every stage but the first --
    `CurrentDriveModel(13).method` is `ELECTRON_CYCLOTRON` like model 10's, so the
    wall-plug block, the totals and the ignition fudge are the *same* functions, called
    here so the diff against PROCESS crosses them on this arm's numbers too. Only the
    efficiency differs: Freethy's model reads seven variables (six data reads plus
    `feffcd`, which model 10 conspicuously does not read) where model 10 reads three.

    `i_plasma_ignited` and `i_ecrh_wave_mode` are static switches, never traced; the
    harness marks both `static_argnames`.

    Returns
    -------
    tuple
        The same 10-tuple as `calculate_current_drive_ecrh_primary_no_secondary`.
    """
    eta_cd_hcd_primary = freethy_electron_cyclotron_efficiency(
        temp_plasma_electron_vol_avg_kev=temp_plasma_electron_vol_avg_kev,
        n_charge_plasma_effective_vol_avg=n_charge_plasma_effective_vol_avg,
        rmajor=rmajor,
        nd_plasma_electrons_vol_avg=nd_plasma_electrons_vol_avg,
        b_plasma_toroidal_on_axis=b_plasma_toroidal_on_axis,
        n_ecrh_harmonic=n_ecrh_harmonic,
        feffcd=feffcd,
        i_ecrh_wave_mode=i_ecrh_wave_mode,
    )

    c_hcd_secondary_driven, f_c_plasma_hcd_secondary = hcd_secondary_driven_current(
        eta_cd_hcd_secondary=0.0,
        p_hcd_secondary_injected_mw=p_hcd_secondary_injected_mw,
        plasma_current=plasma_current,
    )

    p_hcd_primary_injected_mw = hcd_primary_injected_power_mw(
        f_c_plasma_auxiliary=f_c_plasma_auxiliary,
        f_c_plasma_hcd_secondary=f_c_plasma_hcd_secondary,
        plasma_current=plasma_current,
        eta_cd_hcd_primary=eta_cd_hcd_primary,
    )

    (
        p_hcd_ecrh_injected_total_mw,
        p_hcd_ecrh_electric_mw,
        eta_hcd_primary_injector_wall_plug,
        p_hcd_primary_electric_mw,
    ) = electron_cyclotron_primary_powers(
        p_hcd_ecrh_injected_secondary_mw=0.0,
        p_hcd_primary_injected_mw=p_hcd_primary_injected_mw,
        p_hcd_primary_extra_heat_mw=p_hcd_primary_extra_heat_mw,
        eta_ecrh_injector_wall_plug=eta_ecrh_injector_wall_plug,
    )

    p_hcd_injected_total_mw = hcd_injected_power_total_mw(
        p_hcd_primary_injected_mw=p_hcd_primary_injected_mw,
        p_hcd_primary_extra_heat_mw=p_hcd_primary_extra_heat_mw,
        p_hcd_secondary_injected_mw=p_hcd_secondary_injected_mw,
        p_hcd_secondary_extra_heat_mw=0.0,
    )

    p_hcd_electric_total_mw = hcd_electric_total_mw(
        p_hcd_primary_electric_mw=p_hcd_primary_electric_mw,
        p_hcd_secondary_electric_mw=0.0,
        i_plasma_ignited=i_plasma_ignited,
    )

    return (
        eta_cd_hcd_primary,
        c_hcd_secondary_driven,
        f_c_plasma_hcd_secondary,
        p_hcd_primary_injected_mw,
        p_hcd_ecrh_injected_total_mw,
        p_hcd_ecrh_electric_mw,
        eta_hcd_primary_injector_wall_plug,
        p_hcd_primary_electric_mw,
        p_hcd_injected_total_mw,
        p_hcd_electric_total_mw,
    )


# ---------------------------------------------------------------------------
# cottax nodes
# ---------------------------------------------------------------------------


class HcdPrimaryEfficiency(ExplicitFunction):
    """The family that owns `.current_drive.eta_cd_hcd_primary`: one occupant per model.

    This is what `hcd_models` was: eleven lambdas in a dict, indexed by
    `i_hcd_primary`, of which one is called (`current_drive.py:1795-1798`). Each reads a
    different set -- model 10 reads three variables, model 13 reads six plus two more
    switches, models 6/7/8 reach the plasma profile machinery entirely -- so declaring
    the union would be the invented-edge defect at its widest in this file.
    """


class HcdPrimaryEfficiencyUserInputEcrh(HcdPrimaryEfficiency):
    """`i_hcd_primary == 10` (`USER_INPUT_ELECTRON_CYCLOTRON`).

    `large_tokamak_eval.IN.DAT:124`'s value, and the only one this port answers.
    """

    eta_cd_hcd_primary = OutputInto(current_drive)

    def __call__(
        self,
        eta_cd_norm_ecrh=From(current_drive),
        nd_plasma_electrons_vol_avg=From(physics),
        rmajor=From(physics),
    ):
        return user_input_electron_cyclotron_efficiency(
            eta_cd_norm_ecrh=eta_cd_norm_ecrh,
            nd_plasma_electrons_vol_avg=nd_plasma_electrons_vol_avg,
            rmajor=rmajor,
        )


class HcdPrimaryEfficiencyFreethyEcrhOMode(HcdPrimaryEfficiency):
    """`i_hcd_primary == 13` (`FREETHY_ELECTRON_CYCLOTRON`), O-mode.

    `spherical_tokamak_eval.IN.DAT:133` and `st_regression.IN.DAT:2522`'s value, the
    second occupant of this family. The nested `i_ecrh_wave_mode` is pinned to `0`
    (O-mode) -- both files set it explicitly (`:130` / `:2665`) and it is PROCESS's
    default (`current_drive_variables.py:116`). It is pinned rather than read because it
    is a switch, not a port (`_audit/naming_convention.md` § "switches are not ports"),
    and it stays a static kwarg of the shared pure function rather than splitting it
    because the two wave modes read identical variable sets -- see
    `freethy_electron_cyclotron_efficiency`'s docstring for the evidence. An X-mode
    occupant would be these same seven reads over the other branch; it is not written
    (`indat.py`'s `UNPORTED[("i_ecrh_wave_mode", 1)]`).

    Seven reads where model 10 has three -- including `feffcd`, the fudge factor model
    10's lambda conspicuously omits (`current_drive.py:1744-1747` has no `feffcd`;
    `:1759-1770` does).
    """

    eta_cd_hcd_primary = OutputInto(current_drive)

    def __call__(
        self,
        temp_plasma_electron_vol_avg_kev=From(physics),
        n_charge_plasma_effective_vol_avg=From(physics),
        rmajor=From(physics),
        nd_plasma_electrons_vol_avg=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
        n_ecrh_harmonic=From(current_drive),
        feffcd=From(current_drive),
    ):
        return freethy_electron_cyclotron_efficiency(
            temp_plasma_electron_vol_avg_kev=temp_plasma_electron_vol_avg_kev,
            n_charge_plasma_effective_vol_avg=n_charge_plasma_effective_vol_avg,
            rmajor=rmajor,
            nd_plasma_electrons_vol_avg=nd_plasma_electrons_vol_avg,
            b_plasma_toroidal_on_axis=b_plasma_toroidal_on_axis,
            n_ecrh_harmonic=n_ecrh_harmonic,
            feffcd=feffcd,
            i_ecrh_wave_mode=0,
        )


class HcdSecondaryHeating(ExplicitFunction):
    """The family that owns what the *secondary* heating system contributes.

    `i_hcd_secondary` chooses among the same eleven efficiency models plus a twelfth
    value, `0`, meaning there is no secondary system at all.
    """


class HcdSecondaryHeatingNone(HcdSecondaryHeating, StatesValues):
    """`i_hcd_secondary == 0` (`NO_CURRENT_DRIVE`): the secondary contributes zero.

    PROCESS's default (`current_drive_variables.py:206`) and
    `large_tokamak_eval.IN.DAT`'s value, since that file never sets the switch.

    **A node with no reads, and that is the finding, not an accident.** Of the three
    fields it owns, PROCESS explicitly assigns only one -- `p_hcd_secondary_extra_heat_mw
    = 0.0` at `current_drive.py:1682`, guarded by `if i_hcd_secondary == 0` and by
    nothing else. The other two it simply never writes on this arm:
    `eta_cd_hcd_secondary` is skipped because `0` is not a key of `hcd_models`
    (`:1784-1787`) and `p_hcd_secondary_electric_mw` because every block that assigns it
    is guarded on a `secondary_cdm.method` that `NO_CURRENT_DRIVE` does not have. Both
    therefore hold their `DataStructure` defaults -- `0.0` at
    `current_drive_variables.py:98` and `heat_transport_variables.py:127` -- for the
    whole run, and no other model writes either (checked across `process/`).

    Declaring the zeros is what keeps two computed quantities off the boundary. The
    alternative is to leave them as inputs nothing produces, which would be a boundary
    entry standing for "PROCESS did not run this code", and `_audit/tokamak_boundary.md`
    § "The twelve that are simply inputs" is explicit that the boundary is for variables
    PROCESS *computes nowhere*, not for ones a switch happened to skip.

    All three zeros are **stated** rather than literals in the body, so they reach the
    compiled program as arguments (`models/stated.py`, `_audit/optimise_design.md` §28,
    §34): each output is read at `^stated.<its place>` and supplied through the env.
    """

    eta_cd_hcd_secondary = OutputInto(current_drive)
    p_hcd_secondary_extra_heat_mw = OutputInto(current_drive)
    p_hcd_secondary_electric_mw = OutputInto(heat_transport)


class HcdSecondaryDrivenCurrent(ExplicitFunction):
    """cottax node: `hcd_secondary_driven_current`, ports declared.

    Switch independent.
    """

    c_hcd_secondary_driven = OutputInto(current_drive)
    f_c_plasma_hcd_secondary = OutputInto(current_drive)

    def __call__(
        self,
        eta_cd_hcd_secondary=From(current_drive),
        p_hcd_secondary_injected_mw=From(current_drive),
        plasma_current=From(physics),
    ):
        return hcd_secondary_driven_current(
            eta_cd_hcd_secondary=eta_cd_hcd_secondary,
            p_hcd_secondary_injected_mw=p_hcd_secondary_injected_mw,
            plasma_current=plasma_current,
        )


class HcdPrimaryInjectedPower(ExplicitFunction):
    """cottax node: `hcd_primary_injected_power_mw`, ports declared.

    Switch independent.
    """

    p_hcd_primary_injected_mw = OutputInto(current_drive)

    def __call__(
        self,
        f_c_plasma_auxiliary=From(physics),
        f_c_plasma_hcd_secondary=From(current_drive),
        plasma_current=From(physics),
        eta_cd_hcd_primary=From(current_drive),
    ):
        return hcd_primary_injected_power_mw(
            f_c_plasma_auxiliary=f_c_plasma_auxiliary,
            f_c_plasma_hcd_secondary=f_c_plasma_hcd_secondary,
            plasma_current=plasma_current,
            eta_cd_hcd_primary=eta_cd_hcd_primary,
        )


class HcdPrimaryPowers(ExplicitFunction):
    """The family that owns the primary system's wall-plug and per-technology powers.

    Two switches decide this one, which is why its occupants are named for a pair the
    way `PlasmaPowerLossIgnitedCoreRadiation` is. `i_hcd_primary`'s *method* picks the
    block (`current_drive.py:2068`/`2099`/`2131`/`2162`/`2191`, one per technology), and
    `i_hcd_secondary`'s method decides how much that technology's injected-power
    accumulator already held when the block's `+=` ran -- the accumulators are zeroed at
    `:1663-1667` and the secondary blocks at `:1885-2063` add to them first. A
    per-technology "secondary contribution" variable would let the two be independent
    nodes, but PROCESS has no such field: it accumulates in place, so the pair is the
    honest key until someone adds one.
    """


class HcdPrimaryPowersElectronCyclotronNoSecondary(HcdPrimaryPowers):
    """Primary method `ELECTRON_CYCLOTRON` (`i_hcd_primary` 3, 7, 10, 13), secondary 0.

    Only `i_hcd_primary == 10` is reachable end to end in this port, because it is the
    only value with an `HcdPrimaryEfficiency` occupant -- but the block itself is keyed
    on the method, so the class is stated for the four values that share it rather than
    narrowed to the one whose upstream happens to be written.

    Reads `eta_ecrh_injector_wall_plug` and **not** the four sibling injector
    efficiencies (`eta_lowhyb_*`, `eta_icrh_*`, `eta_ebw_*`, `eta_beam_*`) that a node
    covering all five methods would have had to declare, nor `e_beam_kev`,
    `f_p_beam_orbit_loss`, `f_p_beam_shine_through` or `f_p_beam_injected_ions`, which
    only the neutral-beam block at `:2191-2260` reads.
    """

    p_hcd_ecrh_injected_total_mw = OutputInto(current_drive)
    p_hcd_ecrh_electric_mw = OutputInto(current_drive)
    eta_hcd_primary_injector_wall_plug = OutputInto(current_drive)
    p_hcd_primary_electric_mw = OutputInto(heat_transport)

    def __call__(
        self,
        p_hcd_primary_injected_mw=From(current_drive),
        p_hcd_primary_extra_heat_mw=From(current_drive),
        eta_ecrh_injector_wall_plug=From(current_drive),
    ):
        return electron_cyclotron_primary_powers(
            p_hcd_ecrh_injected_secondary_mw=0.0,
            p_hcd_primary_injected_mw=p_hcd_primary_injected_mw,
            p_hcd_primary_extra_heat_mw=p_hcd_primary_extra_heat_mw,
            eta_ecrh_injector_wall_plug=eta_ecrh_injector_wall_plug,
        )


class HcdInjectedPowerTotal(ExplicitFunction):
    """cottax node: `hcd_injected_power_total_mw`, ports declared. Switch independent."""

    p_hcd_injected_total_mw = OutputInto(current_drive)

    def __call__(
        self,
        p_hcd_primary_injected_mw=From(current_drive),
        p_hcd_primary_extra_heat_mw=From(current_drive),
        p_hcd_secondary_injected_mw=From(current_drive),
        p_hcd_secondary_extra_heat_mw=From(current_drive),
    ):
        return hcd_injected_power_total_mw(
            p_hcd_primary_injected_mw=p_hcd_primary_injected_mw,
            p_hcd_primary_extra_heat_mw=p_hcd_primary_extra_heat_mw,
            p_hcd_secondary_injected_mw=p_hcd_secondary_injected_mw,
            p_hcd_secondary_extra_heat_mw=p_hcd_secondary_extra_heat_mw,
        )


class HcdElectricTotal(ExplicitFunction):
    """The family that owns `.heat_transport.p_hcd_electric_total_mw`.

    `i_plasma_ignited` decides it, and the two arms could hardly differ more: one is the
    sum of two reads, the other is the literal `0.0` and reads nothing at all.
    """


class HcdElectricTotalNonIgnited(HcdElectricTotal):
    """`i_plasma_ignited == 0` (`NON_IGNITED`): the sum of the two systems' wall plugs.

    PROCESS's default (`physics_variables.py:881`) and `large_tokamak_eval.IN.DAT`'s
    value, since that file never sets the switch -- the same discovery
    `_audit/tokamak_boundary.md` § "What blocked the real file" makes about the
    confinement head.
    """

    p_hcd_electric_total_mw = OutputInto(heat_transport)

    def __call__(
        self,
        p_hcd_primary_electric_mw=From(heat_transport),
        p_hcd_secondary_electric_mw=From(heat_transport),
    ):
        return hcd_electric_total_mw(
            p_hcd_primary_electric_mw=p_hcd_primary_electric_mw,
            p_hcd_secondary_electric_mw=p_hcd_secondary_electric_mw,
            i_plasma_ignited=PlasmaIgnitionModel.NON_IGNITED,
        )


class HcdElectricTotalIgnited(HcdElectricTotal):
    """`i_plasma_ignited == 1` (`IGNITED`): zero, and two reads that are not reads.

    `current_drive.py:2294-2299` computes the sum and then overwrites it with `0.0`,
    under a comment that calls the reset a *"fudge"*. Declared as one node branching
    internally, the graph would carry
    `.heat_transport.p_hcd_primary_electric_mw` and `.p_hcd_secondary_electric_mw` as
    dependencies of a constant. Declared as an occupant, it reads nothing -- which is
    also the honest statement about `stellarator_helias.IN.DAT:126`, the reference run
    that sets this switch to `1`.
    """

    p_hcd_electric_total_mw = OutputInto(heat_transport)

    def __call__(self):
        return hcd_electric_total_mw(
            p_hcd_primary_electric_mw=0.0,
            p_hcd_secondary_electric_mw=0.0,
            i_plasma_ignited=PlasmaIgnitionModel.IGNITED,
        )


class FusionGain(ExplicitFunction):
    """cottax node: `fusion_gain`, ports declared. Switch independent.

    Owns `.current_drive.big_q_plasma` on a **tokamak** graph. The stellarator's
    counterpart is `models/stellarator/heating.py::FusionGain`, registered in
    `models/stellarator/namespace.py`; the two never coexist, for the reason
    `TokamakCurrentDrive.electric_total`'s docstring gives about
    `.heat_transport.p_hcd_electric_total_mw` -- ownership is per-graph and the two
    device graphs are never assembled together.

    Nothing inside the model graph reads this path: its only reader is the problem
    layer -- `core/solver/objectives.py::objective_metric_5` (`i_figure_merit = -5`,
    `FUSION_GAIN_Q`, `st_regression.IN.DAT`'s) and
    `core/solver/constraints.py::constraint_28`. That is exactly why the node was
    missing for as long as it was: `boundary.py`'s pins are measured on the model graph,
    where an unread output is invisible.
    """

    big_q_plasma = OutputInto(current_drive)

    def __call__(
        self,
        p_fusion_total_mw=From(physics),
        p_hcd_injected_total_mw=From(current_drive),
        p_beam_orbit_loss_mw=From(current_drive),
        p_plasma_ohmic_mw=From(physics),
    ):
        return fusion_gain(
            p_fusion_total_mw=p_fusion_total_mw,
            p_hcd_injected_total_mw=p_hcd_injected_total_mw,
            p_beam_orbit_loss_mw=p_beam_orbit_loss_mw,
            p_plasma_ohmic_mw=p_plasma_ohmic_mw,
        )
