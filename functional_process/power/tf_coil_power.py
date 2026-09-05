"""Pure-functional port of the TF coil power conversion sub-unit of
`process/models/power.py` (registry unit #14, chunk A).

Audit record: `functional_process/_audit/units/models/power/tf_coil_power.md`. Covers
`Power.tfpwr` (2117-2287), `Power.tfpwcall` (2291-2330) and `Power.tfcpwr`
(2332-2629) -- see the audit record's data-footprint table for the full trace.

`Power.tfpwr` dispatches on the topology-changing switch `.tfcoil.i_tf_sup` to one of
two essentially disjoint computations, same shape as `vacuum.py`'s
`.vacuum.i_vacuum_pumping` dispatch (see `vacuum.md`):

- **`i_tf_sup != 1`** (resistive TF coil) -- straight-line algebra, no calls.
  `calculate_tf_power_resistive` below.
- **`i_tf_sup == 1`** (superconducting TF coil) -- `tfpwcall` folds `ettfmj`/`itfka`
  into `tfcpwr`'s call, itself straight-line algebra (one real branch, on whether the
  TF leg resistance `rptfc` is exactly zero -- see that function's docstring).
  `calculate_tf_power_superconducting` below.

Both are tier-1: no `self.data` access once ported, no internal iteration, no calls
into any other model.
"""

import jax.numpy as jnp
import numpy as np

from functional_process.models.safe_math import safe_pow, safe_sqrt
from functional_process.vocabulary import constants


def calculate_tf_power_resistive(
    c_tf_turn,
    j_tf_bus,
    rho_tf_bus,
    len_tf_bus,
    n_tf_coils,
    res_tf_leg,
    p_cp_resistive,
    c_tf_total,
    p_tf_joints_resistive,
    p_tf_leg_resistive,
    etatf,
):
    """`Power.tfpwr`'s `i_tf_sup != 1` (resistive TF coil) branch.

    Ports `process/models/power.py:2131-2198` verbatim -- straight-line arithmetic, no
    branches. The reactive-power term `tfreacmw` is hardcoded to `0.0` in the source
    ("Set reactive power to 0, since ramp up can be long", #199 #847) and is dropped
    here rather than carried as a dead `+ 0.0`.

    Parameters
    ----------
    c_tf_turn :
        Current per TF coil turn (A). `.tfcoil.c_tf_turn`.
    j_tf_bus :
        Bus current density (A/m2). `.tfcoil.j_tf_bus`.
    rho_tf_bus :
        Bus resistivity (ohm m). `.tfcoil.rho_tf_bus`.
    len_tf_bus :
        Total TF system bus length (m). `.tfcoil.len_tf_bus` (read here; written by
        the superconducting branch instead -- the two branches never both touch it).
    n_tf_coils :
        Number of TF coils. `.tfcoil.n_tf_coils`.
    res_tf_leg :
        Resistance per TF coil leg (ohm). `.tfcoil.res_tf_leg`.
    p_cp_resistive :
        Resistive power in the centrepost/inboard legs (W). `.tfcoil.p_cp_resistive`.
    c_tf_total :
        Total TF coil current (A). `.tfcoil.c_tf_total`.
    p_tf_joints_resistive :
        Resistive power in TF coil joints (W). `.tfcoil.p_tf_joints_resistive`.
    p_tf_leg_resistive :
        Resistive power in the outboard legs (W). `.tfcoil.p_tf_leg_resistive`.
    etatf :
        TF coil power supply efficiency. `.heat_transport.etatf`.

    Returns
    -------
    :
        `(m_tf_bus, vtfkv, p_cp_resistive_mw, p_tf_leg_resistive_mw,
        p_tf_joints_resistive_mw, tfcmw, p_tf_electric_supplies_mw)` -- the seven
        `VarPath`s `Power.tfpwr` writes in this branch (`.tfcoil.m_tf_bus`,
        `.tfcoil.vtfkv`, `.tfcoil.p_cp_resistive_mw`, `.tfcoil.p_tf_leg_resistive_mw`,
        `.tfcoil.p_tf_joints_resistive_mw`, `.tfcoil.tfcmw`,
        `.heat_transport.p_tf_electric_supplies_mw`). `tfbusres`/`res_tf_system_total`/
        `tfbusmw` are locals with no `VarPath` of their own (used only to build the
        above, or for display in `tfpwr`'s out-of-scope output section) and are
        dropped from the return.
    """
    a_tf_bus = c_tf_turn / j_tf_bus
    tfbusres = rho_tf_bus * len_tf_bus / a_tf_bus
    m_tf_bus = len_tf_bus * a_tf_bus * constants.DEN_COPPER

    res_tf_system_total = (
        n_tf_coils * res_tf_leg + (p_cp_resistive / c_tf_total**2) + tfbusres
    )
    vtfkv = 1.0e-3 * res_tf_system_total * c_tf_turn / n_tf_coils

    p_cp_resistive_mw = 1.0e-6 * p_cp_resistive
    p_tf_leg_resistive_mw = 1.0e-6 * p_tf_leg_resistive
    p_tf_joints_resistive_mw = 1.0e-6 * p_tf_joints_resistive
    tfbusmw = 1.0e-6 * c_tf_turn**2 * tfbusres

    tfcmw = (
        p_cp_resistive_mw + p_tf_leg_resistive_mw + tfbusmw + p_tf_joints_resistive_mw
    )
    p_tf_electric_supplies_mw = tfcmw / etatf

    return (
        m_tf_bus,
        vtfkv,
        p_cp_resistive_mw,
        p_tf_leg_resistive_mw,
        p_tf_joints_resistive_mw,
        tfcmw,
        p_tf_electric_supplies_mw,
    )


def calculate_tf_power_superconducting(
    c_tf_turn,
    e_tf_magnetic_stored_total_gj,
    n_tf_coils,
    rmajor,
    v_tf_coil_dump_quench_kv,
    res_tf_leg,
    rho_tf_bus,
    etatf,
):
    """`Power.tfpwr`'s `i_tf_sup == 1` (superconducting TF coil) branch.

    Fuses `Power.tfpwcall` (2306-2330, computes `ettfmj`/`itfka` and calls `tfcpwr`)
    and `Power.tfcpwr` (2332-2629, the actual power-conversion-system sizing) into one
    pure function -- `tfpwcall` does nothing but that one call, so there is no
    intermediate `VarPath` to preserve.

    Only the arithmetic feeding the five returned/written values is kept; PROCESS's
    `tfcpwr` computes several more locals (`lptfcs`, `r1dump`, `ttfsec`, `r1ppmw`,
    `xpower`/`xpwrmw`, `tfpsv`, `tfpska`, `vtfbus`, `albuswt`) that feed only its
    out-of-scope output section, not any of the five returns -- dropped here, same
    convention as `vacuum.py`'s pruning of display-only locals.

    Parameters
    ----------
    c_tf_turn :
        Current per TF coil turn (A). `.tfcoil.c_tf_turn`.
    e_tf_magnetic_stored_total_gj :
        Total TF coil magnetic stored energy (GJ). `.tfcoil.e_tf_magnetic_stored_total_gj`.
    n_tf_coils :
        Number of TF coils. `.tfcoil.n_tf_coils`.
    rmajor :
        Plasma major radius (m). `.physics.rmajor`.
    v_tf_coil_dump_quench_kv :
        Voltage across a TF coil during quench (kV). `.tfcoil.v_tf_coil_dump_quench_kv`.
    res_tf_leg :
        Resistance per TF coil leg (ohm), `tfcpwr`'s `rptfc`. `.tfcoil.res_tf_leg`.
        `rptfc == 0.0` selects PROCESS's "superconducting" sub-case
        (`nsptfc = 1`, 4-hour charge time) inside `tfcpwr` itself; any nonzero value
        selects its "resistive" sub-case (`nsptfc = 0`, 10-minute charge time) --
        note the naming collision with this function's own "superconducting" branch
        of `tfpwr`: `tfcpwr` is shared code with its own, independent notion of
        "superconducting" keyed off a different field. Ported as `jnp.where(res_tf_leg
        == 0.0, ...)`, an exact float equality test carried over unchanged from PROCESS
        (`# noqa: RUF069` at `power.py:2365`).
    rho_tf_bus :
        Bus resistivity (ohm m). `.tfcoil.rho_tf_bus`.
    etatf :
        TF coil power supply efficiency. `.heat_transport.etatf`.

    Returns
    -------
    :
        `(tfckw, len_tf_bus, drarea, tfcbv, p_tf_electric_supplies_mw)` -- the five
        `VarPath`s `Power.tfpwcall` writes (`.tfcoil.tfckw`, `.tfcoil.len_tf_bus`,
        `.tfcoil.drarea`, `.buildings.tfcbv`, `.heat_transport.p_tf_electric_supplies_mw`).
    """
    ettfmj = e_tf_magnetic_stored_total_gj / n_tf_coils * 1.0e3
    itfka = 1.0e-3 * c_tf_turn

    ncpbkr = 1.0e0
    djmka = 0.125e0
    rtfps = 1.05e0
    fspc1 = 0.15e0
    fspc2 = 0.8e0
    fspc3 = 0.4e0

    is_resistive_leg = res_tf_leg != 0.0e0
    tchghr = jnp.where(is_resistive_leg, 0.16667e0, 4.0e0)
    nsptfc = jnp.where(is_resistive_leg, 0.0e0, 1.0e0)

    ettfc = n_tf_coils * ettfmj
    ltfth = 2.0e0 * ettfc / itfka**2
    ntfbkr = n_tf_coils / ncpbkr
    albusa = itfka / djmka
    len_tf_bus = (
        8.0e0 * np.pi * rmajor
        + (1.0e0 + ntfbkr) * (12.0e0 * rmajor + 80.0e0)
        + 0.2e0 * itfka * safe_sqrt(n_tf_coils * res_tf_leg * 1000.0e0)
    )
    rtfbus = rho_tf_bus * len_tf_bus / (albusa / 10000)
    rcoils = n_tf_coils * res_tf_leg
    ztotal = rtfbus + rcoils + ltfth / (3600.0e0 * tchghr)
    tfcv = 1000.0e0 * itfka * ztotal

    ntfpm = (itfka * (1.0e0 + nsptfc)) / 5.0e0
    tfpmv = rtfps * tfcv / (1.0e0 + nsptfc)
    tfpmka = rtfps * itfka / (ntfpm / (1.0e0 + nsptfc))
    tfpmkw = tfpmv * tfpmka
    tfckw = tfpmkw * ntfpm
    tfackw = tfckw / 0.9e0

    r1emj = nsptfc * ettfc / (ntfbkr * 4.0e0 + 0.0001e0)
    rpower = (n_tf_coils * res_tf_leg + rtfbus) * itfka**2

    part1 = fspc1 * ntfpm * safe_pow(tfpmkw, 0.667e0)
    part2 = fspc2 * ntfbkr * safe_pow(v_tf_coil_dump_quench_kv * itfka, 0.667e0)
    part3 = fspc3 * safe_pow(
        tfackw / (2.4e0 * nsptfc + 13.8e0 * (1.0e0 - nsptfc)), 0.667e0
    )
    tfcfsp = part1 + part2 + part3
    drarea = 0.5e0 * (ntfbkr * 4.0e0) * safe_pow(1.0e0 + r1emj, 0.667e0)
    tfcbv = 6.0e0 * tfcfsp

    p_tf_electric_supplies_mw = rpower / etatf

    return (tfckw, len_tf_bus, drarea, tfcbv, p_tf_electric_supplies_mw)
