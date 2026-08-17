"""Pure-functional port of `process/models/stellarator/density_limits.py`.

Audit record: `functional_process/models/stellarator/density_limits.md`. Only the two
tier-1 functions are ported here. `power_at_ignition_point` is tier 2 and remains
blocked on registry unit #1 (`st_phys`), and `output` is a reporting shell.

Signatures follow the record's `## proposed signature(s)` verbatim, including its
argument spellings, so that the record stays the thing you read to know what a port
should look like.

`SudoDensityLimit`/`EcrhDensityLimit` are the `cottax` nodes. `EcrhDensityLimit`'s
`.stellarator.dlimit_ecrh`/`.bt_max_ecrh` outputs are invented names, not existing
PROCESS storage: `st_d_limit_ecrh`'s return values are locals in its caller
(`st_density_limits`), clamped and passed straight to `output()`, never written to
`data` (confirmed by grep, see the audit record's data-footprint table).
`i_plasma_pedestal` is a precondition (open question 2 in the record), not a `VarPath`
-- a static field on the node, per `naming_convention.md`'s "switches are not ports".
"""

import equinox as eqx
import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, Input, Output

_SUDO_COEFFICIENT = 0.25e20
"""Sudo et al. (Nucl. Fusion 30, 11, 1990) line-averaged density limit prefactor."""

_ECRH_CUTOFF_COEFFICIENT = 3.142077e-4
"""me*e0/e^2 — cutoff density per squared angular frequency."""

_ELECTRON_GYRO_COEFFICIENT = 1.76e11
"""Electron gyrofrequency per tesla (rad/s/T)."""


def calculate_sudo_density_limit(
    b_plasma_toroidal_on_axis,
    p_plasma_loss_mw,
    rmajor,
    rminor,
    nd_plasma_electrons_vol_avg,
    nd_plasma_electron_line,
):
    """Sudo volume-averaged density limit for a stellarator.

    Ports `st_sudo_density_limit`. Two differences from the original, both from the
    audit record rather than invented here:

    - the `data` back-door is closed: the two fields it read off `DataStructure`
      (`nd_plasma_electrons_vol_avg`, `nd_plasma_electron_line`) are explicit arguments,
      and the internal write to `.physics.nd_plasma_electrons_max` is dropped as a
      redundant duplicate of the return value;
    - the `ProcessValueError` on a non-positive square-root argument becomes a NaN. A
      traced function cannot raise on a data-dependent condition, so the domain guard
      has to be a value. The double-`where` below is not redundant: the inner one keeps
      `sqrt` away from the invalid argument so that the *gradient* stays finite on the
      valid branch, which a single outer `where` would not do.

    Parameters
    ----------
    b_plasma_toroidal_on_axis :
        Toroidal field on axis (T).
    p_plasma_loss_mw :
        Absorbed heating power (MW). `powht` in the original.
    rmajor :
        Plasma major radius (m).
    rminor :
        Plasma minor radius (m).
    nd_plasma_electrons_vol_avg :
        Volume-averaged electron density (/m3).
    nd_plasma_electron_line :
        Line-averaged electron density (/m3).

    Returns
    -------
    :
        Maximum volume-averaged plasma density (/m3), or NaN where the original would
        have raised.
    """
    arg = p_plasma_loss_mw * b_plasma_toroidal_on_axis / (rmajor * rminor * rminor)

    valid = arg > 0.0
    safe_arg = jnp.where(valid, arg, 1.0)

    # Maximum line-averaged electron density.
    dnlamx = _SUDO_COEFFICIENT * jnp.sqrt(safe_arg)

    # Scale to the volume-averaged electron density.
    limit = dnlamx * nd_plasma_electrons_vol_avg / nd_plasma_electron_line

    return jnp.where(valid, limit, jnp.nan)


def calculate_ecrh_density_limit(
    gyro_frequency_max,
    b_plasma_toroidal_on_axis,
    i_plasma_pedestal=0,
):
    """Density limit imposed by an on-axis ECRH heating scheme.

    Ports `st_d_limit_ecrh`. `i_plasma_pedestal` is a **precondition, not a switch**:
    the original assigns `dlimit_ecrh` only under `i_plasma_pedestal == 0` and merely
    logs on the `else` branch, so a nonzero value raises `UnboundLocalError` there. The
    audit record recommends encoding that as an explicit precondition rather than
    splitting the function or keeping a static branch (open question 2, flagged for
    review rather than decided). It is a static argument, so raising here is legitimate
    — nothing is traced.

    Parameters
    ----------
    gyro_frequency_max :
        Maximum available gyrotron frequency (1/s, not rad/s).
        `.stellarator.max_gyrotron_frequency`.
    b_plasma_toroidal_on_axis :
        Field on axis (T). `bt_input` in the original.
    i_plasma_pedestal :
        Pedestal profile switch; must be 0.

    Returns
    -------
    :
        `(dlimit_ecrh, bt_max)` — maximum peak plasma density (/m3) and the maximum
        field allowing ECRH (T).

    Raises
    ------
    ValueError
        If `i_plasma_pedestal` is not 0.
    """
    if i_plasma_pedestal != 0:
        raise ValueError(
            "calculate_ecrh_density_limit requires i_plasma_pedestal == 0; PROCESS's "
            "st_d_limit_ecrh has no formula for any other value (it raises "
            "UnboundLocalError). See density_limits.md open question 2"
        )

    gyro_frequency_max_rad = gyro_frequency_max * 2.0 * jnp.pi

    gyro_frequency = jnp.minimum(
        _ELECTRON_GYRO_COEFFICIENT * b_plasma_toroidal_on_axis,
        gyro_frequency_max_rad,
    )

    # Restrict the field to the maximum available gyrotron frequency.
    bt_max = gyro_frequency_max_rad / _ELECTRON_GYRO_COEFFICIENT

    dlimit_ecrh = jnp.maximum(
        0.0, _ECRH_CUTOFF_COEFFICIENT * gyro_frequency * gyro_frequency
    )

    return dlimit_ecrh, bt_max


class SudoDensityLimit(ExplicitFunction):
    """cottax node: `calculate_sudo_density_limit`, unchanged, ports declared."""

    nd_plasma_electrons_max = Output(lambda s: s.physics.nd_plasma_electrons_max)

    def __call__(
        self,
        b_plasma_toroidal_on_axis=Input(lambda s: s.physics.b_plasma_toroidal_on_axis),
        p_plasma_loss_mw=Input(lambda s: s.physics.p_plasma_loss_mw),
        rmajor=Input(lambda s: s.physics.rmajor),
        rminor=Input(lambda s: s.physics.rminor),
        nd_plasma_electrons_vol_avg=Input(
            lambda s: s.physics.nd_plasma_electrons_vol_avg
        ),
        nd_plasma_electron_line=Input(lambda s: s.physics.nd_plasma_electron_line),
    ):
        return calculate_sudo_density_limit(
            b_plasma_toroidal_on_axis,
            p_plasma_loss_mw,
            rmajor,
            rminor,
            nd_plasma_electrons_vol_avg,
            nd_plasma_electron_line,
        )


class EcrhDensityLimit(ExplicitFunction):
    """cottax node: `calculate_ecrh_density_limit`, unchanged, ports declared.

    `i_plasma_pedestal` is a precondition, not a port -- see module docstring. No
    default: the graph-assembly step must say explicitly which precondition it is
    relying on, rather than silently inheriting one.
    """

    i_plasma_pedestal: int = eqx.field(static=True)

    dlimit_ecrh = Output(lambda s: s.stellarator.dlimit_ecrh)
    bt_max_ecrh = Output(lambda s: s.stellarator.bt_max_ecrh)

    def __call__(
        self,
        gyro_frequency_max=Input(lambda s: s.stellarator.max_gyrotron_frequency),
        b_plasma_toroidal_on_axis=Input(lambda s: s.physics.b_plasma_toroidal_on_axis),
    ):
        return calculate_ecrh_density_limit(
            gyro_frequency_max, b_plasma_toroidal_on_axis, self.i_plasma_pedestal
        )
