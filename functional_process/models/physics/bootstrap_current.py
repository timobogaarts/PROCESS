"""Pure-functional port of the tokamak bootstrap-current chain.

Audit record: `functional_process/_audit/units/models/physics/bootstrap_current.md` --
read it first, especially "## the chain is not one file" and "## the invented
`triang` edge".

**Scope of this pass: the `large_tokamak_eval.IN.DAT` reference arm.** That file sets
`i_bootstrap_current = 4` (Sauter, line 286) and leaves `i_diamagnetic_current` and
`i_pfirsch_schluter_current` at their `physics_variables.py` defaults of `0`
(lines 856, 895; the `OUT.DAT` confirms `i_diamagnetic_current = 0` at line 970). One
occupant class per switch value, per `next_steps.md` §14.2 -- no `i_*` integer appears as
a kwarg or inside any body here. The other thirteen `i_bootstrap_current` values, the two
non-zero `i_diamagnetic_current` values and the one non-zero `i_pfirsch_schluter_current`
value were UNPORTED; the audit record's "switches touched" tables carry each one's reads
and reason.

**The ST closing wave (2026-08-29) added the two SCENE fits.**
`i_diamagnetic_current = 2` (`SceneDiamagneticCurrent`) and
`i_pfirsch_schluter_current = 1` (`ScenePfirschSchluterCurrent`) are what both
`spherical_tokamak_eval.IN.DAT` and `st_regression.IN.DAT` set, and they are the first
two occupants that make `.current_drive.f_c_plasma_diamagnetic` and
`.current_drive.f_c_plasma_pfirsch_schluter` non-zero anywhere in this port -- so they
are also the first live test of `PlasmaCurrentFractions`' three-term sum. Only
`i_diamagnetic_current == 1` (Hender) remains UNPORTED in the two families.

**Source spans two PROCESS files, and that is deliberate.** The unit's nominal source is
`process/models/physics/bootstrap_current.py` (`PlasmaBootstrapCurrent`,
`SauterBootstrapCurrent`), but the *bookkeeping* that turns a bootstrap fraction into the
auxiliary and inductive fractions is written inline in `Physics.run()`,
**`process/models/physics/physics.py:543-588`**, immediately after
`self.plasma_bootstrap_current.run()`. They are one chain:

    f_c_plasma_bootstrap_sauter  ->  f_c_plasma_bootstrap (capped)
                                       |
                                       +-> f_c_plasma_internal
                                             |
                                             +-> f_c_plasma_auxiliary
                                                 f_c_plasma_inductive

`.physics.f_c_plasma_auxiliary` is a declared boundary read of
`functional_process/models/physics/current_drive.py`'s `HcdPrimaryInjectedPower`, and
`physics.py:585-588` is its **only** producer anywhere in `process/`. Leaving it on the
boundary while porting the thing it is computed from would have been the invented-edge
defect this project exists to remove, so the six-statement block is ported here with
`file:line` attribution on every function that came from `physics.py`. **If a later pass
gives `physics.py`'s current-fraction bookkeeping its own unit,
`PlasmaCurrentFractions` and `calculate_plasma_current_fractions` move there wholesale**
-- flagged, not decided. The oracle weakness this creates is real and is stated in the
test module: `Physics.run()` has no callable sub-shell, so that one contract's reference
is a second reading of the source rather than PROCESS's own execution, and it is
anchored separately on `large_tokamak_eval`'s recorded MFILE numbers.

**The whole `i_bootstrap_current` family is computed by PROCESS on every run.**
`PlasmaBootstrapCurrent.run` (`bootstrap_current.py:81-262`) evaluates all fourteen
scalings into fourteen `.current_drive.f_c_plasma_bootstrap_*` fields and only then
indexes the family by the switch (`get_bootstrap_current_fraction_value`, `:264-298`).
`_audit/tokamak_boundary.md`'s note on this slot reads that as "one node producing the
family plus an index, not an occupant per arm". **This port takes the other reading**,
because §14.2's binding policy binds: the thirteen unselected scalings are dead work
whose only consumers are `output()` and the MFILE, and computing them would give this
node the union of fourteen arms' reads -- twenty-odd `.physics` fields against the
seventeen the Sauter arm actually needs, including `alphaj`, `alphap`,
`beta_thermal_poloidal_vol_avg`, `ind_plasma_internal_norm`, `kappa`, `eps` and
`nd_plasma_electron_max_array[6]`, none of which the live arm touches. The disagreement
is recorded in the audit record rather than smoothed over.

**Not ported, and why:**

- Thirteen of the fourteen `i_bootstrap_current` values -- see the audit record.
  Value `0` (`USER_INPUT`) is an **empty slot** rather than an unported model:
  `get_bootstrap_current_fraction_value` selects `.current_drive.f_c_plasma_bootstrap`
  from itself (`bootstrap_current.py:283`) and `physics.py:549-551` exempts it from the
  cap, so under it the field is a boundary input with no producer.
- The thirteen `.current_drive.f_c_plasma_bootstrap_*` sibling fields and
  `.current_drive.bscf_gi_i`/`bscf_gi_ii` -- **reporting-only**. Measured: no reader
  anywhere in `process/` outside `PlasmaBootstrapCurrent.output`
  (`bootstrap_current.py:1271-1445`) and `core/io/plot/summary.py:8934-8946`.
  `.current_drive.f_c_plasma_bootstrap_sauter` *is* carried, because it is the live arm's
  own value and PROCESS stores it (`bootstrap_current.py:138`) before the cap is applied.
- `.physics.err242` / `.physics.err243` -- reporting-only flags, read only at
  `bootstrap_current.py:1406` and `:1410`, both inside `output()`. Same call
  `plasma_current.py`'s port made for `.physics.alphaj_wesson`.
- `.current_drive.f_c_plasma_pfirsch_schluter_scene` (`physics.py:534`) and
  `.current_drive.f_c_plasma_diamagnetic_hender`/`_scene`
  (`plasma_current.py:1068-1079`) -- computed unconditionally by PROCESS, selected only
  at `i_pfirsch_schluter_current == 1` / `i_diamagnetic_current in (1, 2)`, and otherwise
  read only by `output()`. Reporting-only on this arm.
- `_trapped_particle_fraction_sauter`'s `fit = 1` (Sauter 2002) and `fit = 2` (Sauter
  2016) branches (`bootstrap_current.py:2509-2527`). `fit` is a **method-choice static
  kwarg with no `DataStructure` field behind it** and no call site anywhere passes it, so
  the port drops the parameter entirely and carries the ASTRA branch alone. See "## the
  invented `triang` edge" -- this is what makes `.physics.triang` unread by the live
  chain.
- `PlasmaBootstrapCurrent.output` and `SauterBootstrapCurrent.run`/`output` -- reporting,
  and the latter two are empty (`bootstrap_current.py:1450-1454`).
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto
from jax import lax

from functional_process.models.stated import StatesValues
from functional_process.models.safe_math import safe_sqrt
from functional_process.paths import current_drive, physics

# ---------------------------------------------------------------------------
# Numerics shared with PROCESS
# ---------------------------------------------------------------------------


def _profile_at(profile, radial_elements):
    """`profile[radial_elements]`, emitted as a **slice** when the index set allows it.

    Pure expression rewrite -- same elements, same order, same value. It exists only to
    keep the Sauter scaling's jaxpr small. `radial_elements` is
    `np.arange(2, n_plasma_profile_elements)` at the one live call site
    (`bootstrap_fraction_sauter`), so every index expression in this module is a
    *contiguous, statically known* run. Written as `profile[indices]`, `jnp` cannot know
    that: it must emit the general fancy-index lowering, which is **seven** jaxpr
    equations per site (`lt`/`add`/`select_n` for negative-index wraparound,
    `convert_element_type`, `broadcast_in_dim`, then a `gather`). `lax.slice_in_dim` is
    one.

    Measured on `large_tokamak_nof`'s profile shape (`n = 201`): 188 gathers, and the
    whole node's program falls from 2,095 equations to 902. It pays twice, because the
    MDA solve also compiles `jacfwd`, whose XLA pass scales with the forward program
    (`_audit/next_steps.md` §24.10).

    The fallback is not decoration: the tier-1 contracts drive the coefficient functions
    directly and a future caller could pass a traced or non-contiguous index array, for
    which the gather is the *correct* lowering. Contiguity is checked, not assumed.
    """
    indices = (
        np.asarray(radial_elements)
        if isinstance(radial_elements, np.ndarray)
        else radial_elements
    )
    if isinstance(indices, np.ndarray) and indices.ndim == 1 and indices.size > 0:
        start = int(indices[0])
        if start >= 0 and np.array_equal(
            indices, np.arange(start, start + indices.size)
        ):
            return lax.slice_in_dim(profile, start, start + indices.size)
    return profile[radial_elements]


def _gradient(profile_y, coordinate):
    """`numpy.gradient(profile_y, coordinate)` for a 1-D, possibly non-uniform grid.

    `SauterBootstrapCurrent.bootstrap_fraction_sauter` takes three logarithmic profile
    derivatives with `np.gradient(..., rho)` (`bootstrap_current.py:1541-1543`), passing
    a **coordinate array** rather than a spacing. That selects `numpy`'s non-uniform
    second-order interior stencil and its `edge_order = 1` one-sided ends, which is a
    different expression from the uniform `(f[i+1] - f[i-1]) / (2h)` shortcut even when
    the grid is uniform -- same value in exact arithmetic, different rounding, and a
    different `d/d(rho[i])`.

    `jnp.gradient` accepts the coordinate array and agrees to `2.2e-16`, which is not
    agreement at the tolerance this harness asserts (`MACHINE_PRECISION`), so the
    stencil is written out instead. Measured: this function is **bit-identical** to
    `np.gradient` on the unit's own grid, which is the same reasoning
    `plasma_profiles._simpson` records for `scipy.integrate.simpson` -- and the same
    trap, a uniform-grid shortcut that is right in value and wrong in derivative, is the
    one this harness already caught once (`_audit/test_harness.md`, "the
    `scipy.integrate.simpson` bug").
    """
    spacing = jnp.diff(coordinate)
    backward, forward = spacing[:-1], spacing[1:]
    interior = (
        -(forward / (backward * (backward + forward))) * profile_y[:-2]
        + ((forward - backward) / (backward * forward)) * profile_y[1:-1]
        + (backward / (forward * (backward + forward))) * profile_y[2:]
    )
    first = (profile_y[1] - profile_y[0]) / spacing[0]
    last = (profile_y[-1] - profile_y[-2]) / spacing[-1]
    return jnp.concatenate([first[jnp.newaxis], interior, last[jnp.newaxis]])


# ---------------------------------------------------------------------------
# The Sauter scaling's collisionality chain
# ---------------------------------------------------------------------------


def _coulomb_logarithm_sauter(radial_elements, tempe, ne):
    """Coulomb logarithm on the profile grid.

    Ports `SauterBootstrapCurrent._coulomb_logarithm_sauter`,
    `process/models/physics/bootstrap_current.py:1610-1651`, unchanged (`np.log` ->
    `jnp.log`, `@nb.njit` dropped).

    `ne` is in units of `1e19 m^-3` here, not `m^-3` -- the caller scales it at
    `bootstrap_current.py:1500`.

    References
    ----------
    C. A. Ordonez, M. I. Molina, Phys. Plasmas 1 (1994) 2515.
    """
    return (
        15.9
        - 0.5 * jnp.log(_profile_at(ne, radial_elements - 1))
        + jnp.log(_profile_at(tempe, radial_elements - 1))
    )


def _electron_collisions_sauter(radial_elements, tempe, ne):
    """Electron-electron collision frequency (Hz).

    Ports `SauterBootstrapCurrent._electron_collisions_sauter`,
    `bootstrap_current.py:1653-1682`, unchanged.
    """
    return (
        670.0
        * _coulomb_logarithm_sauter(radial_elements, tempe, ne)
        * _profile_at(ne, radial_elements - 1)
        / (
            _profile_at(tempe, radial_elements - 1)
            * jnp.sqrt(_profile_at(tempe, radial_elements - 1))
        )
    )


def _electron_collisionality_sauter(
    radial_elements, rmajor, zeff, inverse_q, sqeps, tempe, ne
):
    """Electron collisionality `nu_e*`.

    Ports `SauterBootstrapCurrent._electron_collisionality_sauter`,
    `bootstrap_current.py:1684-1734`, unchanged (`np.abs`/`np.sqrt` -> `jnp`).
    """
    return (
        _electron_collisions_sauter(radial_elements, tempe, ne)
        * 1.4
        * _profile_at(zeff, radial_elements - 1)
        * rmajor
        / jnp.abs(
            _profile_at(inverse_q, radial_elements - 1)
            * (_profile_at(sqeps, radial_elements - 1) ** 3)
            * jnp.sqrt(_profile_at(tempe, radial_elements - 1))
            * 1.875e7
        )
    )


def _ion_collisions_sauter(radial_elements, zeff, ni, tempi, amain):
    """Ion collision frequency (Hz), at a fixed Coulomb logarithm of 15.

    Ports `SauterBootstrapCurrent._ion_collisions_sauter`,
    `bootstrap_current.py:1736-1777`, unchanged.

    **The `zeff` parameter is not the effective charge at the live call site.**
    `_ion_collisionality_sauter` is called with `zmain` in the `zeff` position
    (`bootstrap_current.py:2231-2233` against the signature at `:1779-1788`), so what is
    raised to the fourth power here is the main-ion charge `1 + f_plasma_fuel_helium3`,
    not `n_charge_plasma_effective_vol_avg`. Kept verbatim; see the record's **D1**.
    """
    return (
        _profile_at(zeff, radial_elements - 1) ** 4
        * _profile_at(ni, radial_elements - 1)
        * 322.0
        / (
            _profile_at(tempi, radial_elements - 1)
            * jnp.sqrt(
                _profile_at(tempi, radial_elements - 1)
                * _profile_at(amain, radial_elements - 1)
            )
        )
    )


def _ion_collisionality_sauter(
    radial_elements, rmajor, inverse_q, sqeps, tempi, amain, zeff, ni
):
    """Ion collisionality `nu_i*`.

    Ports `SauterBootstrapCurrent._ion_collisionality_sauter`,
    `bootstrap_current.py:1779-1826`, unchanged. See `_ion_collisions_sauter` on what
    `zeff` actually carries here.
    """
    return (
        3.2e-6
        * _ion_collisions_sauter(radial_elements, zeff, ni, tempi, amain)
        * rmajor
        / (
            jnp.abs(_profile_at(inverse_q, radial_elements - 1) + 1.0e-4)
            * _profile_at(sqeps, radial_elements - 1) ** 3
            * jnp.sqrt(
                _profile_at(tempi, radial_elements - 1)
                / _profile_at(amain, radial_elements - 1)
            )
        )
    )


def _trapped_particle_fraction_sauter(radial_elements, sqeps):
    """Trapped particle fraction, ASTRA fit (Emiliano Fable, private communication).

    Ports `SauterBootstrapCurrent._trapped_particle_fraction_sauter`'s `fit == 0` branch,
    `bootstrap_current.py:2444-2507`, unchanged.

    **`triang` and `fit` are both dropped from the signature**, and that is the finding
    rather than a simplification. `fit` defaults to `0` and no call site in `process/`
    passes anything else, so branches `1` (Sauter 2002) and `2` (Sauter 2016) are
    unreachable; `triang` is read **only** by branch `2` (`:2522`). PROCESS threads it
    down from `bootstrap_fraction_sauter` through all three `_calculate_l*` coefficient
    functions to reach a branch that never runs. See the module docstring and the audit
    record's "## the invented `triang` edge".

    References
    ----------
    O. Sauter, C. Angioni, Y. R. Lin-Liu, Phys. Plasmas 6 (1999) 2834.
    """
    sqeps_reduced = _profile_at(sqeps, radial_elements - 1)
    eps = sqeps_reduced**2
    zz = 1.0 - eps
    return 1.0 - zz * jnp.sqrt(zz) / (1.0 + 1.46 * sqeps_reduced)


# ---------------------------------------------------------------------------
# Local poloidal betas (Fable's 2015 corrections)
# ---------------------------------------------------------------------------


def _beta_poloidal_sauter(
    radial_elements,
    nr,
    rmajor,
    b_plasma_toroidal_on_axis,
    ne,
    tempe,
    inverse_q,
    rho,
):
    """Local poloidal beta from the electron profiles alone.

    Ports `SauterBootstrapCurrent._beta_poloidal_sauter`,
    `bootstrap_current.py:2307-2364`, unchanged (`np.where` -> `jnp.where`).

    **The `radial_elements == nr` branch is dead.** `radial_elements` is
    `arange(2, n_plasma_profile_elements)` (`:1532`), so its largest value is `nr - 1`
    and the second arm of the `where` is never selected. It is carried anyway, evaluated
    on both sides as `jnp.where` requires: both arms are finite everywhere the first one
    is, so nothing leaks a NaN through the untaken branch. Audit record **D2**.
    """
    return (
        jnp.where(
            radial_elements != nr,
            1.6e-4
            * jnp.pi
            * (_profile_at(ne, radial_elements) + _profile_at(ne, radial_elements - 1))
            * (
                _profile_at(tempe, radial_elements)
                + _profile_at(tempe, radial_elements - 1)
            ),
            6.4e-4
            * jnp.pi
            * _profile_at(ne, radial_elements - 1)
            * _profile_at(tempe, radial_elements - 1),
        )
        * (
            rmajor
            / (
                b_plasma_toroidal_on_axis
                * _profile_at(rho, radial_elements - 1)
                * jnp.abs(_profile_at(inverse_q, radial_elements - 1) + 1.0e-4)
            )
        )
        ** 2
    )


def _beta_poloidal_total_sauter(
    radial_elements,
    nr,
    rmajor,
    b_plasma_toroidal_on_axis,
    ne,
    ni,
    tempe,
    tempi,
    inverse_q,
    rho,
):
    """Local poloidal beta including the ion pressure.

    Ports `SauterBootstrapCurrent._beta_poloidal_total_sauter`,
    `bootstrap_current.py:2366-2442`, unchanged. Same dead `where` arm as
    `_beta_poloidal_sauter`.
    """
    return (
        jnp.where(
            radial_elements != nr,
            1.6e-4
            * jnp.pi
            * (
                (
                    (
                        _profile_at(ne, radial_elements)
                        + _profile_at(ne, radial_elements - 1)
                    )
                    * (
                        _profile_at(tempe, radial_elements)
                        + _profile_at(tempe, radial_elements - 1)
                    )
                )
                + (
                    (
                        _profile_at(ni, radial_elements)
                        + _profile_at(ni, radial_elements - 1)
                    )
                    * (
                        _profile_at(tempi, radial_elements)
                        + _profile_at(tempi, radial_elements - 1)
                    )
                )
            ),
            6.4e-4
            * jnp.pi
            * (
                _profile_at(ne, radial_elements - 1)
                * _profile_at(tempe, radial_elements - 1)
                + _profile_at(ni, radial_elements - 1)
                * _profile_at(tempi, radial_elements - 1)
            ),
        )
        * (
            rmajor
            / (
                b_plasma_toroidal_on_axis
                * _profile_at(rho, radial_elements - 1)
                * jnp.abs(_profile_at(inverse_q, radial_elements - 1) + 1.0e-4)
            )
        )
        ** 2
    )


# ---------------------------------------------------------------------------
# The three Sauter transport coefficients
# ---------------------------------------------------------------------------


def _calculate_l31_coefficient(
    radial_elements,
    number_of_elements,
    rmajor,
    b_plasma_toroidal_on_axis,
    ne,
    ni,
    tempe,
    tempi,
    inverse_q,
    rho,
    zeff,
    sqeps,
):
    """`L31` -- the coefficient of `grad(ln n_e)`, times the total poloidal beta.

    Ports `SauterBootstrapCurrent._calculate_l31_coefficient`,
    `bootstrap_current.py:1828-1931`, unchanged except that the unused `triang`
    parameter is gone (see `_trapped_particle_fraction_sauter`).

    `f31_teff` is Eq. 14b of Sauter 1999 and the polynomial is Eq. 14a; the trailing
    multiplication by `_beta_poloidal_total_sauter` is the correction Fable suggested on
    15/05/2015 (`bootstrap_current.py:1919`).
    """
    charge_profile = _profile_at(zeff, radial_elements - 1)
    f_trapped = _trapped_particle_fraction_sauter(radial_elements, sqeps)
    electron_collisionality = _electron_collisionality_sauter(
        radial_elements, rmajor, zeff, inverse_q, sqeps, tempe, ne
    )

    f31_teff = f_trapped / (
        (1.0 + (1.0 - 0.1 * f_trapped) * jnp.sqrt(electron_collisionality))
        + (0.5 * (1.0 - f_trapped) * electron_collisionality) / charge_profile
    )

    l31_coefficient = (
        ((1.0 + 1.4 / (charge_profile + 1.0)) * f31_teff)
        - (1.9 / (charge_profile + 1.0) * f31_teff**2)
        + ((0.3 * f31_teff**3 + 0.2 * f31_teff**4) / (charge_profile + 1.0))
    )

    return l31_coefficient * _beta_poloidal_total_sauter(
        radial_elements,
        number_of_elements,
        rmajor,
        b_plasma_toroidal_on_axis,
        ne,
        ni,
        tempe,
        tempi,
        inverse_q,
        rho,
    )


def _calculate_l31_32_coefficient(
    radial_elements,
    number_of_elements,
    rmajor,
    b_plasma_toroidal_on_axis,
    ne,
    ni,
    tempe,
    tempi,
    inverse_q,
    rho,
    zeff,
    sqeps,
):
    """`L31 + L32` -- the coefficient of `grad(ln T_e)`.

    Ports `SauterBootstrapCurrent._calculate_l31_32_coefficient`,
    `bootstrap_current.py:1933-2120`, unchanged except for the dropped `triang`.
    Eqs. 15b-15e of Sauter 1999, plus Fable's 2015 correction.
    """
    charge_profile = _profile_at(zeff, radial_elements - 1)
    f_trapped = _trapped_particle_fraction_sauter(radial_elements, sqeps)
    electron_collisionality = _electron_collisionality_sauter(
        radial_elements, rmajor, zeff, inverse_q, sqeps, tempe, ne
    )

    f32ee_teff = f_trapped / (
        1.0
        + 0.26 * (1.0 - f_trapped) * jnp.sqrt(electron_collisionality)
        + (
            0.18
            * (1.0 - 0.37 * f_trapped)
            * electron_collisionality
            / jnp.sqrt(charge_profile)
        )
    )

    f32ei_teff = f_trapped / (
        (1.0 + (1.0 + 0.6 * f_trapped) * jnp.sqrt(electron_collisionality))
        + (
            0.85
            * (1.0 - 0.37 * f_trapped)
            * electron_collisionality
            * (1.0 + charge_profile)
        )
    )

    big_f32ee_teff = (
        (
            (0.05 + 0.62 * charge_profile)
            / charge_profile
            / (1.0 + 0.44 * charge_profile)
            * (f32ee_teff - f32ee_teff**4)
        )
        + (
            (f32ee_teff**2 - f32ee_teff**4 - 1.2 * (f32ee_teff**3 - f32ee_teff**4))
            / (1.0 + 0.22 * charge_profile)
        )
        + (1.2 / (1.0 + 0.5 * charge_profile) * f32ee_teff**4)
    )

    big_f32ei_teff = (
        (
            -(0.56 + 1.93 * charge_profile)
            / charge_profile
            / (1.0 + 0.44 * charge_profile)
            * (f32ei_teff - f32ei_teff**4)
        )
        + (
            4.95
            / (1.0 + 2.48 * charge_profile)
            * (f32ei_teff**2 - f32ei_teff**4 - 0.55 * (f32ei_teff**3 - f32ei_teff**4))
        )
        - (1.2 / (1.0 + 0.5 * charge_profile) * f32ei_teff**4)
    )

    return _beta_poloidal_sauter(
        radial_elements,
        number_of_elements,
        rmajor,
        b_plasma_toroidal_on_axis,
        ne,
        tempe,
        inverse_q,
        rho,
    ) * (big_f32ee_teff + big_f32ei_teff) + _calculate_l31_coefficient(
        radial_elements,
        number_of_elements,
        rmajor,
        b_plasma_toroidal_on_axis,
        ne,
        ni,
        tempe,
        tempi,
        inverse_q,
        rho,
        zeff,
        sqeps,
    ) * _beta_poloidal_sauter(
        radial_elements,
        number_of_elements,
        rmajor,
        b_plasma_toroidal_on_axis,
        ne,
        tempe,
        inverse_q,
        rho,
    ) / _beta_poloidal_total_sauter(
        radial_elements,
        number_of_elements,
        rmajor,
        b_plasma_toroidal_on_axis,
        ne,
        ni,
        tempe,
        tempi,
        inverse_q,
        rho,
    )


def _calculate_l34_alpha_31_coefficient(
    radial_elements,
    number_of_elements,
    rmajor,
    b_plasma_toroidal_on_axis,
    inverse_q,
    sqeps,
    tempi,
    tempe,
    amain,
    zmain,
    ni,
    ne,
    rho,
    zeff,
):
    """`L34 * alpha + L31` -- the coefficient of `grad(ln T_i)`.

    Ports `SauterBootstrapCurrent._calculate_l34_alpha_31_coefficient`,
    `bootstrap_current.py:2122-2305`, unchanged except for the dropped `triang`.
    Eqs. 16a, 16b, 17a and 17b of Sauter 1999, plus Fable's 2015 correction.

    The `zmain` parameter is named for what the call site passes
    (`bootstrap_current.py:2231-2233`), where PROCESS's own signature calls it `zeff` --
    see `_ion_collisions_sauter` and audit record **D1**.
    """
    charge_profile = _profile_at(zeff, radial_elements - 1)
    f_trapped = _trapped_particle_fraction_sauter(radial_elements, sqeps)
    electron_collisionality = _electron_collisionality_sauter(
        radial_elements, rmajor, zeff, inverse_q, sqeps, tempe, ne
    )

    f34_teff = f_trapped / (
        (1.0 + (1.0 - 0.1 * f_trapped) * jnp.sqrt(electron_collisionality))
        + 0.5 * (1.0 - 0.5 * f_trapped) * electron_collisionality / charge_profile
    )

    l34_coefficient = (
        ((1.0 + (1.4 / (charge_profile + 1.0))) * f34_teff)
        - ((1.9 / (charge_profile + 1.0)) * f34_teff**2)
        + ((0.3 / (charge_profile + 1.0)) * f34_teff**3)
        + ((0.2 / (charge_profile + 1.0)) * f34_teff**4)
    )

    alpha_0 = (-1.17 * (1.0 - f_trapped)) / (
        1.0 - (0.22 * f_trapped) - 0.19 * f_trapped**2
    )

    ion_collisionality = _ion_collisionality_sauter(
        radial_elements, rmajor, inverse_q, sqeps, tempi, amain, zmain, ni
    )

    # `safe_sqrt`, not `jnp.sqrt`: `ion_collisionality` is exactly `0` whenever the ion
    # density is (`_ion_collisions_sauter` is linear in `ni`), and `sqrt` there is
    # value-correct with an `inf` derivative -- `_audit/next_steps.md` §9's class, found
    # by `test_gradient_finite_at_zero` on `nd_plasma_ions_total_vol_avg`. Identical
    # for every non-zero radicand; see `models/safe_math.py`.
    sqrt_ion_collisionality = safe_sqrt(ion_collisionality)

    alpha = (
        (alpha_0 + (0.25 * (1.0 - f_trapped**2)) * sqrt_ion_collisionality)
        / (1.0 + (0.5 * sqrt_ion_collisionality))
        + (0.315 * ion_collisionality**2 * f_trapped**6)
    ) / (1.0 + (0.15 * ion_collisionality**2 * f_trapped**6))

    return (
        _beta_poloidal_total_sauter(
            radial_elements,
            number_of_elements,
            rmajor,
            b_plasma_toroidal_on_axis,
            ne,
            ni,
            tempe,
            tempi,
            inverse_q,
            rho,
        )
        - _beta_poloidal_sauter(
            radial_elements,
            number_of_elements,
            rmajor,
            b_plasma_toroidal_on_axis,
            ne,
            tempe,
            inverse_q,
            rho,
        )
    ) * (l34_coefficient * alpha) + _calculate_l31_coefficient(
        radial_elements,
        number_of_elements,
        rmajor,
        b_plasma_toroidal_on_axis,
        ne,
        ni,
        tempe,
        tempi,
        inverse_q,
        rho,
        zeff,
        sqeps,
    ) * (
        1.0
        - _beta_poloidal_sauter(
            radial_elements,
            number_of_elements,
            rmajor,
            b_plasma_toroidal_on_axis,
            ne,
            tempe,
            inverse_q,
            rho,
        )
        / _beta_poloidal_total_sauter(
            radial_elements,
            number_of_elements,
            rmajor,
            b_plasma_toroidal_on_axis,
            ne,
            ni,
            tempe,
            tempi,
            inverse_q,
            rho,
        )
    )


# ---------------------------------------------------------------------------
# The Sauter bootstrap fraction
# ---------------------------------------------------------------------------


def bootstrap_fraction_sauter(
    *,
    n_plasma_profile_elements,
    radius_plasma_profile_norm,
    nd_plasma_electron_profile,
    temp_plasma_electron_profile_kev,
    a_plasma_poloidal,
    rminor,
    rmajor,
    nd_plasma_ions_total_vol_avg,
    nd_plasma_electrons_vol_avg,
    temp_plasma_ion_vol_avg_kev,
    temp_plasma_electron_vol_avg_kev,
    n_charge_plasma_effective_vol_avg,
    q0,
    q95,
    m_ions_total_amu,
    f_plasma_fuel_helium3,
    b_plasma_toroidal_on_axis,
    plasma_current,
):
    """Bootstrap current fraction and profile from the Sauter et al. scaling.

    Ports `SauterBootstrapCurrent.bootstrap_fraction_sauter`,
    `process/models/physics/bootstrap_current.py:1456-1608`. The one structural change is
    the signature: PROCESS takes a `PlasmaProfile` object and reads sixteen
    `self.data.physics.*` fields off the back door, and every one of those is an explicit
    keyword argument here. `.physics.triang` is **not** among them -- see
    `_trapped_particle_fraction_sauter`.

    The three profile arrays are the ported profile nodes' outputs
    (`functional_process/models/physics/profiles.py`): `radius_plasma_profile_norm` is
    `ProfileGrid`'s, and the density/temperature pair is
    `plasma_profiles.py`'s `PedestalProfileValues` on this input
    (`i_plasma_pedestal = 1`, `large_tokamak_eval.IN.DAT:291`). Nothing is re-derived
    here.

    `n_plasma_profile_elements` is a **shape, not a value** -- it fixes the length of
    every profile array and the loop bound `arange(2, n)`, it is never an iteration
    variable, and differentiating with respect to it is meaningless. Static.

    **The radial integral is PROCESS's own rectangle sum, not Simpson's rule.**
    `bootstrap_current.py:1608` is `np.sum(da * jboot) / plasma_current` with
    `da = 2 pi rho[j-1] (rho[j] - rho[j-1])`, evaluated on the annulus *inner* radius.
    That is a first-order rule and it is kept verbatim; `plasma_profiles._simpson` is
    deliberately **not** substituted, because the two do not agree in value and PROCESS's
    answer is the oracle here.

    Returns
    -------
    tuple
        `(f_c_plasma_bootstrap_sauter, j_plasma_bootstrap_sauter_profile)` -- the
        fraction (before `cboot`) and the bootstrap current density profile (A/m^2) on
        the `arange(2, n)` sub-grid, exactly PROCESS's return.

    References
    ----------
    O. Sauter, C. Angioni, Y. R. Lin-Liu, Phys. Plasmas 6 (1999) 2834;
    Erratum, Phys. Plasmas 9 (2002) 5140. Code supplied by Emiliano Fable, IPP Garching.
    """
    roa = radius_plasma_profile_norm

    # Local circularised minor radius (`:1494`).
    rho = jnp.sqrt(a_plasma_poloidal / jnp.pi) * roa

    # Square root of local aspect ratio (`:1497`).
    #
    # `safe_sqrt`, not `jnp.sqrt`: `roa[0]` is exactly `0` -- the profile grid starts on
    # axis -- so the radicand is `0` at element 0, where `sqrt` is `0` in value and `inf`
    # in derivative. Element 0 is never read (every use is
    # `_profile_at(sqeps, radial_elements - 1)`, i.e. `arange(1, n-1)`), and a slice
    # *discards* its tangent in forward mode but *transposes to a zero pad* in reverse,
    # where `0 * inf` is `nan`. So the unguarded spelling is finite under `jacfwd` and
    # non-finite under `jacrev` at the same point -- `_audit/optimise_design.md` §33.
    # Value-identical: `safe_sqrt` returns `0.0` at `0` and is bit-identical elsewhere.
    sqeps = safe_sqrt(roa * (rminor / rmajor))

    # Electron and ion density profiles, in 1e19 m^-3 (`:1500-1504`).
    ne = nd_plasma_electron_profile * 1e-19
    ni = (nd_plasma_ions_total_vol_avg / nd_plasma_electrons_vol_avg) * ne

    # Electron and ion temperature profiles, keV (`:1507-1511`).
    tempe = temp_plasma_electron_profile_kev
    tempi = (temp_plasma_ion_vol_avg_kev / temp_plasma_electron_vol_avg_kev) * tempe

    # Flat Zeff profile assumed (`:1515`).
    zeff = jnp.full_like(tempi, n_charge_plasma_effective_vol_avg)

    # Parabolic q profile assumed (`:1519-1522`).
    inverse_q = 1 / (q0 + (q95 - q0) * roa**2)

    # Flat main-ion mass and charge profiles (`:1524-1527`).
    amain = jnp.full_like(inverse_q, m_ions_total_amu)
    zmain = jnp.full_like(inverse_q, 1.0 + f_plasma_fuel_helium3)

    # From 2 because the coefficient functions should return 0 at j == 1 (`:1530-1532`).
    radial_elements = np.arange(2, n_plasma_profile_elements)

    drho = _profile_at(rho, radial_elements) - _profile_at(rho, radial_elements - 1)

    # Area of annulus, assuming a circular plasma cross-section (`:1538`).
    da = 2 * jnp.pi * _profile_at(rho, radial_elements - 1) * drho

    dlogte_drho = _profile_at(_gradient(jnp.log(tempe), rho), radial_elements - 1)
    dlogti_drho = _profile_at(_gradient(jnp.log(tempi), rho), radial_elements - 1)
    dlogne_drho = _profile_at(_gradient(jnp.log(ne), rho), radial_elements - 1)

    jboot = (
        0.5
        * (
            _calculate_l31_coefficient(
                radial_elements,
                n_plasma_profile_elements,
                rmajor,
                b_plasma_toroidal_on_axis,
                ne,
                ni,
                tempe,
                tempi,
                inverse_q,
                rho,
                zeff,
                sqeps,
            )
            * dlogne_drho
            + _calculate_l31_32_coefficient(
                radial_elements,
                n_plasma_profile_elements,
                rmajor,
                b_plasma_toroidal_on_axis,
                ne,
                ni,
                tempe,
                tempi,
                inverse_q,
                rho,
                zeff,
                sqeps,
            )
            * dlogte_drho
            + _calculate_l34_alpha_31_coefficient(
                radial_elements,
                n_plasma_profile_elements,
                rmajor,
                b_plasma_toroidal_on_axis,
                inverse_q,
                sqeps,
                tempi,
                tempe,
                amain,
                zmain,
                ni,
                ne,
                rho,
                zeff,
            )
            * dlogti_drho
        )
        * 1.0e6
        * (
            -b_plasma_toroidal_on_axis
            / (0.2 * jnp.pi * rmajor)
            * _profile_at(rho, radial_elements - 1)
            * _profile_at(inverse_q, radial_elements - 1)
        )
    )  # A/m2

    return jnp.sum(da * jboot, axis=0) / plasma_current, jboot


def enforce_bootstrap_current_fraction_max(
    f_c_plasma_bootstrap, f_c_plasma_bootstrap_max
):
    """The bootstrap fraction, capped at `f_c_plasma_bootstrap_max`.

    Ports **`process/models/physics/physics.py:545-556`**, the block `Physics.run()`
    executes immediately after `self.plasma_bootstrap_current.run()`. PROCESS writes the
    capped value back into the same field it read (`physics.py:552-555`) and raises
    `.physics.err242` alongside; the flag is reporting-only (read at
    `bootstrap_current.py:1406`, inside `output()`) and is not carried.

    **The cap is exempted for `i_bootstrap_current == USER_INPUT`**
    (`physics.py:549-551`)
    and applies to all thirteen other values, which is why it lives inside each scaling's
    occupant rather than in a node of its own: making it a separate node would answer
    `i_bootstrap_current` a second time, and the user-input arm has no node at all.

    `min` is `jnp.minimum` -- the `if`/`min` pair in PROCESS is a data-dependent test on
    a traced value, which `_audit/traceability_policy.md` flags
    `needs-lax-cond-or-where`. Both arms are finite everywhere, so the `where` inside
    `jnp.minimum` leaks nothing; the derivative at the kink is `jnp.minimum`'s, i.e. it
    picks the first argument on a tie, matching `min(a, b)`.
    """
    return jnp.minimum(f_c_plasma_bootstrap, f_c_plasma_bootstrap_max)


def diamagnetic_fraction_scene(beta, q95, q0):
    """The diamagnetic current fraction, SCENE fit (Tim Hender).

    Ports `PlasmaDiamagneticCurrent.diamagnetic_fraction_scene`,
    **`process/models/physics/plasma_current.py:1158-1179`**, unchanged (`@nb.njit`
    dropped, as everywhere in this port). Selected by `i_diamagnetic_current == 2`
    at `plasma_current.py:1088-1094`.

    Pure arithmetic -- no `jnp` call at all, so it traces on whatever the caller hands
    it. The `q95 / q0` quotient is PROCESS's and is left unguarded: `q0` is an input,
    PROCESS neither clamps nor checks it, and at `q0 == 0` both the value and the tangent
    are non-finite, which is the domain edge rather than the value-finite/tangent-`nan`
    class `safe_math.py` exists for.
    """
    return beta * (0.1 * q95 / q0 + 0.44) * 0.414


def ps_fraction_scene(beta):
    """The Pfirsch-Schlüter current fraction, SCENE fit (Tim Hender, 2019).

    Ports the module-level `ps_fraction_scene`,
    **`process/models/physics/physics.py:161-179`**, unchanged (`@nb.jit` dropped).
    Selected by `i_pfirsch_schluter_current == 1` at `physics.py:538-541`.

    Negative by construction: the Pfirsch-Schlüter current opposes the plasma current,
    and `calculate_plasma_current_fractions` above adds it to the bootstrap and
    diamagnetic fractions with that sign, exactly as `physics.py:558-562` does.
    """
    return -9e-2 * beta


def calculate_plasma_current_fractions(
    f_c_plasma_bootstrap,
    f_c_plasma_diamagnetic,
    f_c_plasma_pfirsch_schluter,
    f_c_plasma_non_inductive,
):
    """The plasma-driven, auxiliary and inductive current fractions.

    Ports **`process/models/physics/physics.py:558-588`** -- six statements written
    inline in `Physics.run()`, with no callable sub-shell of their own:

        f_c_plasma_internal  = bootstrap + diamagnetic + pfirsch_schluter   (:558-562)
        f_c_plasma_internal  = min(f_c_plasma_internal, non_inductive)      (:570-578)
        f_c_plasma_inductive = max(1e-10, 1 - non_inductive)                (:581-583)
        f_c_plasma_auxiliary = non_inductive - f_c_plasma_internal          (:585-588)

    `.physics.err243` (`:569`, `:578`) is reporting-only, read at
    `bootstrap_current.py:1410` inside `output()`, and is not carried.

    This is the **only** producer of `.physics.f_c_plasma_auxiliary` in `process/`, and
    that path is a declared boundary read of
    `functional_process/models/physics/current_drive.py`'s `HcdPrimaryInjectedPower`.

    Both clamps are `jnp` elementwise forms of PROCESS's `if`/`min` and `max`. The
    `1e-10` floor on the inductive fraction is PROCESS's guard against a divide-by-zero
    in `v_plasma_loop_burn` (`physics.py:4878`) and is kept.

    Returns
    -------
    tuple
        `(f_c_plasma_internal, f_c_plasma_auxiliary, f_c_plasma_inductive)`.
    """
    f_c_plasma_internal = (
        f_c_plasma_bootstrap + f_c_plasma_diamagnetic + f_c_plasma_pfirsch_schluter
    )
    f_c_plasma_internal = jnp.minimum(f_c_plasma_internal, f_c_plasma_non_inductive)
    f_c_plasma_inductive = jnp.maximum(1.0e-10, 1.0e0 - f_c_plasma_non_inductive)
    f_c_plasma_auxiliary = f_c_plasma_non_inductive - f_c_plasma_internal
    return f_c_plasma_internal, f_c_plasma_auxiliary, f_c_plasma_inductive


# ---------------------------------------------------------------------------
# cottax nodes
# ---------------------------------------------------------------------------


class BootstrapCurrentFractionScaling(ExplicitFunction):
    """The family that owns `.current_drive.f_c_plasma_bootstrap` under
    `i_bootstrap_current` (`bootstrap_current.py:250-262`, `:264-298`).

    Fourteen values. **This pass ports only `SAUTER` (4)**, the value
    `large_tokamak_eval.IN.DAT:286` sets; `USER_INPUT` (0) is an **empty slot** under
    which the field is a boundary input with no producer, and the other twelve are
    UNPORTED (audit record's "switches touched").

    Each occupant owns the *capped* fraction as well as its own scaling value, because
    the cap at `physics.py:546-556` is exempted for `USER_INPUT` alone -- so the cap is
    per-occupant behaviour, and putting it in a node of its own would answer
    `i_bootstrap_current` twice (`model_tree_design.md` §8 step 4d).

    Ragged in a way worth naming: value `4` is the only arm that integrates over the
    plasma profiles at all. The other thirteen are closed-form expressions in a dozen
    volume-averaged scalars, so an occupant for any of them will not share this one's
    profile reads.
    """


class SauterBootstrapCurrentFraction(BootstrapCurrentFractionScaling):
    """`i_bootstrap_current == SAUTER` (4) -- the arm `large_tokamak_eval` takes.

    Seventeen reads plus one static shape, against the twenty-odd `.physics` fields
    PROCESS's `PlasmaBootstrapCurrent.run` touches to fill the whole fourteen-member
    family (module docstring). `.physics.triang` is not read -- see
    `_trapped_particle_fraction_sauter` and the audit record's "## the invented `triang`
    edge".

    Owns three `VarPath`s:

    - `.current_drive.f_c_plasma_bootstrap_sauter` -- the scaling's own value, scaled by
      `cboot` (`bootstrap_current.py:141-143`). PROCESS stores it; it is the field the
      MFILE reports.
    - `.physics.j_plasma_bootstrap_sauter_profile` -- the bootstrap current density
      profile (`bootstrap_current.py:139`). No reader outside `output()`; declared
      anyway, so the graph shows what the source computes, following `ProfileGrid`'s
      precedent in `profiles.py`. `Graph.prune` drops it.
    - `.current_drive.f_c_plasma_bootstrap` -- the selected, capped fraction
      (`bootstrap_current.py:255-257` then `physics.py:552-555`).
    """

    n_plasma_profile_elements: int = eqx.field(static=True)

    f_c_plasma_bootstrap_sauter = OutputInto(current_drive)
    j_plasma_bootstrap_sauter_profile = OutputInto(physics)
    f_c_plasma_bootstrap = OutputInto(current_drive)

    def __call__(
        self,
        radius_plasma_profile_norm=From(physics),
        nd_plasma_electron_profile=From(physics),
        temp_plasma_electron_profile_kev=From(physics),
        a_plasma_poloidal=From(physics),
        rminor=From(physics),
        rmajor=From(physics),
        nd_plasma_ions_total_vol_avg=From(physics),
        nd_plasma_electrons_vol_avg=From(physics),
        temp_plasma_ion_vol_avg_kev=From(physics),
        temp_plasma_electron_vol_avg_kev=From(physics),
        n_charge_plasma_effective_vol_avg=From(physics),
        q0=From(physics),
        q95=From(physics),
        m_ions_total_amu=From(physics),
        f_plasma_fuel_helium3=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
        plasma_current=From(physics),
        cboot=From(current_drive),
        f_c_plasma_bootstrap_max=From(current_drive),
    ):
        fraction, j_plasma_bootstrap_sauter_profile = bootstrap_fraction_sauter(
            n_plasma_profile_elements=self.n_plasma_profile_elements,
            radius_plasma_profile_norm=radius_plasma_profile_norm,
            nd_plasma_electron_profile=nd_plasma_electron_profile,
            temp_plasma_electron_profile_kev=temp_plasma_electron_profile_kev,
            a_plasma_poloidal=a_plasma_poloidal,
            rminor=rminor,
            rmajor=rmajor,
            nd_plasma_ions_total_vol_avg=nd_plasma_ions_total_vol_avg,
            nd_plasma_electrons_vol_avg=nd_plasma_electrons_vol_avg,
            temp_plasma_ion_vol_avg_kev=temp_plasma_ion_vol_avg_kev,
            temp_plasma_electron_vol_avg_kev=temp_plasma_electron_vol_avg_kev,
            n_charge_plasma_effective_vol_avg=n_charge_plasma_effective_vol_avg,
            q0=q0,
            q95=q95,
            m_ions_total_amu=m_ions_total_amu,
            f_plasma_fuel_helium3=f_plasma_fuel_helium3,
            b_plasma_toroidal_on_axis=b_plasma_toroidal_on_axis,
            plasma_current=plasma_current,
        )
        f_c_plasma_bootstrap_sauter = cboot * fraction
        return (
            f_c_plasma_bootstrap_sauter,
            j_plasma_bootstrap_sauter_profile,
            enforce_bootstrap_current_fraction_max(
                f_c_plasma_bootstrap_sauter, f_c_plasma_bootstrap_max
            ),
        )


class PlasmaDiamagneticCurrentFraction(ExplicitFunction):
    """The family that owns `.current_drive.f_c_plasma_diamagnetic` under
    `i_diamagnetic_current` (`plasma_current.py:1081-1094`).

    Three values: `NONE` (0, below), `HENDER_ST_FIT` (1) and `SCENE_FIT` (2, below --
    added by the ST closing wave, 2026-08-29, for the two tracked spherical tokamaks).
    `HENDER_ST_FIT` reads only `.physics.beta_total_vol_avg` and stays UNPORTED: it is
    not live on any tracked input, and it is a genuinely smaller read set than SCENE's,
    so it is its own occupant rather than a constant in this one.
    """


class NoDiamagneticCurrent(PlasmaDiamagneticCurrentFraction, StatesValues):
    """`i_diamagnetic_current == NONE` (0) -- the default, and this input's value.

    **A node that computes nothing, and that is the finding rather than an accident.**
    (It reads its own output's statement and nothing else -- `models/stated.py` derives
    that read, so the claim is unchanged from when it read literally nothing.) PROCESS
    never assigns `.current_drive.f_c_plasma_diamagnetic` on this arm
    (`plasma_current.py:1081-1094` has no `else`), the field is not settable from
    `IN.DAT` (`process/core/input.py` has no entry for it), and no other model writes it
    -- so it holds its `current_drive_variables.py:77` default of `0.0` for the whole
    run. Declaring the zero is what keeps a computed quantity off the boundary, exactly
    as `current_drive.py`'s `NoSecondaryHcd` argues: `_audit/tokamak_boundary.md` §"The
    twelve that are simply inputs" is explicit that the boundary is for variables PROCESS
    computes *nowhere*, not for ones a switch happened to skip.

    The zero is **stated** and not a literal in the body: a constant this node's readers
    multiply by is one XLA deletes the readers of (`models/stated.py`,
    `_audit/optimise_design.md` §28, §34).
    """

    f_c_plasma_diamagnetic = OutputInto(current_drive)
    """The diamagnetic current fraction PROCESS never assigns on this arm, read at
    `^stated.current_drive.f_c_plasma_diamagnetic`."""


class SceneDiamagneticCurrent(PlasmaDiamagneticCurrentFraction):
    """`i_diamagnetic_current == SCENE_FIT` (2) -- both tracked spherical tokamaks.

    Ports `PlasmaDiamagneticCurrent.diamagnetic_fraction_scene`
    (`process/models/physics/plasma_current.py:1158-1179`, `@nb.njit` dropped) and the
    `SCENE_FIT` limb of the selection at `:1088-1094`.

    **Owns `.current_drive.f_c_plasma_diamagnetic` directly, not
    `f_c_plasma_diamagnetic_scene` and then a copy.** PROCESS computes *both* fits
    unconditionally (`:1068-1079`) and then selects; the Hender and SCENE sibling fields
    are reporting-only, measured (this module's docstring, and
    `bootstrap_current.md` "## data footprint"). One owned `VarPath`, no intermediate --
    the same call `WessonCurrentProfileIndex` made for `.physics.alphaj_wesson`.

    The `q95 / q0` quotient is unguarded, exactly as PROCESS writes it: at `q0 == 0` the
    value is `inf` and the tangent is not finite. `q0` is a plasma input (`2.0` on
    `spherical_tokamak_eval.IN.DAT:287`) and PROCESS neither guards nor clamps it; the
    zero-boundary probe therefore steps aside on its own, because zeroing `q0` makes the
    *value* non-finite too.
    """

    f_c_plasma_diamagnetic = OutputInto(current_drive)

    def __call__(
        self,
        beta_total_vol_avg=From(physics),
        q95=From(physics),
        q0=From(physics),
    ):
        return diamagnetic_fraction_scene(beta=beta_total_vol_avg, q95=q95, q0=q0)


class PlasmaPfirschSchluterCurrentFraction(ExplicitFunction):
    """The family that owns `.current_drive.f_c_plasma_pfirsch_schluter` under
    `i_pfirsch_schluter_current` (`physics.py:538-541`).

    Two values: `0` (below) and `1`, which copies
    `.current_drive.f_c_plasma_pfirsch_schluter_scene` -- `ps_fraction_scene(
    beta_total_vol_avg)`, `physics.py:534-536`. Value `1` is `ScenePfirschSchluterCurrent`
    below, added by the ST closing wave (2026-08-29); it is not live on
    `large_tokamak_eval`, whose MFILE records the enforced fraction as `0.0`
    (line 14427) while the SCENE value it would have copied is `-2.9e-3` (line 6900).
    """


class NoPfirschSchluterCurrent(PlasmaPfirschSchluterCurrentFraction, StatesValues):
    """`i_pfirsch_schluter_current == 0` -- the default, and this input's value.

    Same shape and same argument as `NoDiamagneticCurrent`: `physics.py:538-541` has no
    `else`, the field is not an `IN.DAT` variable, nothing else writes it, and it holds
    its `current_drive_variables.py:283` default of `0.0`. The zero is **stated** for the
    reason `NoDiamagneticCurrent`'s is.
    """

    f_c_plasma_pfirsch_schluter = OutputInto(current_drive)
    """The Pfirsch-Schluter current fraction PROCESS never assigns on this arm, read at
    `^stated.current_drive.f_c_plasma_pfirsch_schluter`."""


class ScenePfirschSchluterCurrent(PlasmaPfirschSchluterCurrentFraction):
    """`i_pfirsch_schluter_current == 1` -- both tracked spherical tokamaks.

    Ports the module-level `ps_fraction_scene`
    (`process/models/physics/physics.py:161-179`, `@nb.jit` dropped) and the `== 1` limb
    of `physics.py:538-541`.

    Owns `.current_drive.f_c_plasma_pfirsch_schluter` directly rather than
    `.f_c_plasma_pfirsch_schluter_scene` and a copy, for the reason
    `SceneDiamagneticCurrent` gives: the `_scene` sibling is reporting-only, measured.

    **The fraction is negative** (`-0.09 * beta`), and that is PROCESS's, not a sign
    slip: the Pfirsch-Schlüter current opposes the plasma current, and
    `PlasmaCurrentFractions` sums it into `.current_drive.f_c_plasma_internal` with the
    bootstrap and diamagnetic fractions as written.
    """

    f_c_plasma_pfirsch_schluter = OutputInto(current_drive)

    def __call__(self, beta_total_vol_avg=From(physics)):
        return ps_fraction_scene(beta=beta_total_vol_avg)


class PlasmaCurrentFractions(ExplicitFunction):
    """cottax node: `calculate_plasma_current_fractions`, ports declared.

    Unconditional -- `physics.py:558-588` runs on every device path with no switch on it.

    `.physics.f_c_plasma_non_inductive` is iteration variable 44
    (`core/solver/iteration_variables.py:72`) and a boundary input to this node;
    `large_tokamak_eval.IN.DAT:283` seeds it at `0.4242184436680697`.

    Owns the three fields `physics.py` writes here.
    `.current_drive.f_c_plasma_internal` has no reader outside this block and
    `output()`, but it is the clamped intermediate the other two are defined against, so
    it is declared rather than inlined; `Graph.prune` drops it if nothing wants it.
    """

    f_c_plasma_internal = OutputInto(current_drive)
    f_c_plasma_auxiliary = OutputInto(physics)
    f_c_plasma_inductive = OutputInto(physics)

    def __call__(
        self,
        f_c_plasma_bootstrap=From(current_drive),
        f_c_plasma_diamagnetic=From(current_drive),
        f_c_plasma_pfirsch_schluter=From(current_drive),
        f_c_plasma_non_inductive=From(physics),
    ):
        return calculate_plasma_current_fractions(
            f_c_plasma_bootstrap=f_c_plasma_bootstrap,
            f_c_plasma_diamagnetic=f_c_plasma_diamagnetic,
            f_c_plasma_pfirsch_schluter=f_c_plasma_pfirsch_schluter,
            f_c_plasma_non_inductive=f_c_plasma_non_inductive,
        )
