"""Pure-functional port of the tokamak TF quench chain --
`CICCSuperconductingTFCoil.quench_heat_protection_current_density`
(`process/models/tfcoil/superconducting.py:1298-1379`) and the material physics it calls
into (`process/models/tfcoil/quench.py`).

Audit record: `functional_process/_audit/units/models/tfcoil/quench.md`.

## The headline: the dump voltage never reaches CoolProp

`_audit/tokamak_boundary.md` § `.tokamak.cicc_superconducting_tf_coil` schedules this
whole slot behind the unresolved CoolProp wrapping policy on the grounds that
*"`.tfcoil.v_tf_coil_dump_quench_kv` above is on that chain"*. **It is not**, and the
evidence is one function body. `quench_heat_protection_current_density` returns a pair
and the two halves are computed independently:

```python
# process/models/tfcoil/superconducting.py:1357-1360
v_tf_dump_voltage_peak = 2.0e0 * e_tf_coil_magnetic_stored / (t_tf_quench_dump * c_tf_turn)

# process/models/tfcoil/superconducting.py:1362-1377
j_tf_wp_quench_protection_max = (a_tf_turn_cable_space / a_tf_turn
    * quench.calculate_quench_protection_current_density(...))   # <- the only CoolProp path
```

`.tfcoil.v_tf_coil_dump_quench_kv` is `v_tf_dump_voltage_peak / 1.0e3`
(`superconducting.py:2793-2795`) and depends on exactly three variables, none of them a
material property. It is fully ported below, with no seam and no approximation.

`.tfcoil.j_tf_wp_quench_heat_max` is the one that reaches helium, and it is **not** on
this slot's boundary list at all -- nothing in the currently-assembled tokamak graph
reads it.

## The seam, for whoever resolves the CoolProp policy

Everything up to the boundary is ported: the five copper-resistivity pieces, the copper
and Nb3Sn heat capacities, the integrand, the Gauss-Legendre quadrature and the
hotspot-criterion formula itself. **The helium properties enter as ordinary arguments**
-- `den_helium` and `cp_helium`, evaluated at the quadrature nodes -- so the port is a
pure, traceable function of everything except two numbers per node, and those two
numbers are the entire CoolProp surface.

Measured on the reference configuration (`tests/regression/input_files/
large_tokamak_eval.IN.DAT` plus `tfcoil_variables.py` defaults; probe in the audit
record), one call to `calculate_quench_protection_current_density` makes exactly:

| | |
|---|---|
| fluid | `He`, only |
| properties | `C` (isobaric specific heat, J/kg/K) and `D` (density, kg/m3) -- nothing else |
| state variables | `("T", T, "P", P)`, always |
| pressure | `600000.0` Pa exactly, a hardcoded literal (`process/models/tfcoil/quench.py:301,382,448`), never an input |
| temperatures | the 75 Gauss-Legendre nodes of `[tftmp, temp_tf_conductor_quench_max]` = `[4.75, 150.0]` K, i.e. `4.7868...` to `149.9631...` K |
| calls | 150 `PropsSI` (75 x 2) on the **first** evaluation, **0** on every later one |

The zero is the important number. `process/core/coolprop_interface.py` memoises each
property on its input tuple (`@cache`), and the quadrature grid depends only on `tftmp`
and `temp_tf_conductor_quench_max` -- **neither is written by any model and neither is an
iteration variable** (`grep` over `process/models/**` finds no assignment to either;
`process/core/solver/iteration_variables.py` lists neither; only `t_tf_superconductor_quench`,
ID 56, is an unknown, and it enters through `1/(0.5*tau + t_detect)` *outside* the
integrals). So on this run the helium property surface is **a frozen table of 150
constants**, not a function the optimiser moves along.

That makes the resolution cheap and removes the need to choose between the three
options the dispatch brief listed: no fit and no interpolation table is needed for
`large_tokamak_eval`, because a 75-node lookup evaluated once is already exact. A
JAX-traceable fit or an interpolation in `T` only becomes necessary if a future
configuration makes `tftmp` or `temp_tf_conductor_quench_max` an unknown, and a
`P`-dependence is never needed while the pressure stays a literal.

**This module deliberately declares no node for the current density.** `den_helium`/
`cp_helium` have no `VarPath` -- they are not `DataStructure` fields -- so a node would
have to mint two, which is a policy call for the orchestrator, not for this port
(`traceability_policy.md` § "Non-traceable external calls" defers exactly this). The
pure functions are here, tested against PROCESS, ready for whichever binding is chosen.

## Two more things worth knowing before wrapping it

- **`process/models/tfcoil/quench.py:18` sets `COPPER_DENSITY = 8960.0`, while
  `process/core/constants.py:289` sets `DEN_COPPER = 8900.0`**, and the mass chain in
  `functional_process/models/tfcoil/superconducting.py` uses the latter. Two different
  copper densities in one coil model. Ported faithfully (each formula keeps its own),
  recorded as defect **D3**.
- `calculate_quench_protection_current_density` **clips the fluence** to `[0, 1.5e23]`
  with a warning (`quench.py:533-537`), because *"default fluence is too high for this
  model"*. The reference run's `flu_tf_neutron_fast_max = 1e22`
  (`large_tokamak_eval.IN.DAT:387`) is inside the range, so the clip is inert there --
  but it is a real kink in the derivative and is ported as `jnp.clip`, not dropped.
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.models.safe_math import safe_sqrt
from functional_process.paths import constraints, superconducting_tfcoil, tfcoil
from process.core.coolprop_interface import FluidProperties

QUENCH_HELIUM_PRESSURE_PA = 6.0e5
"""ITER TF coolant pressure, hardcoded in three places in `process/models/tfcoil/
quench.py` (lines 301, 382, 448) with the source's own comment *"no plans to make
input"*. The `P` argument of every CoolProp call this chain makes."""

N_QUENCH_GAUSS_LEGENDRE_NODES = 75
"""`np.polynomial.legendre.leggauss(75)`, `process/models/tfcoil/quench.py:293`. A
quadrature resolution (kind (b)), not a model choice."""

QUENCH_GAUSS_LEGENDRE_NODES, QUENCH_GAUSS_LEGENDRE_WEIGHTS = (
    np.polynomial.legendre.leggauss(N_QUENCH_GAUSS_LEGENDRE_NODES)
)
"""The same nodes and weights PROCESS builds at import
(`process/models/tfcoil/quench.py:293`), taken from the same `numpy` call rather than
re-derived, so the two quadratures are identical by construction."""

COPPER_DENSITY_QUENCH = 8960.0
"""`process/models/tfcoil/quench.py:18`. **Not** `constants.DEN_COPPER` (8900.0) -- see
the module docstring, defect D3."""

NB3SN_DENSITY = 8040.0
"""`process/models/tfcoil/quench.py:19`."""

FLUENCE_MODEL_RANGE = (0.0, 1.5e23)
"""`process/models/tfcoil/quench.py:533`: the range the irradiation-resistivity fit is
valid over, outside which PROCESS clips."""

_CU_CP_POLY = (1.131, -9.454, 12.99, -5.501, 0.7637)
"""NIST Monograph 177 eq. 7-1, `process/models/tfcoil/quench.py:48`."""

_CU_MAGNETORESISTANCE_POLY = (-2.662, 0.3168, 0.6229, -0.1839, 0.01827)
"""NIST Monograph 177 eq. 8-7, `process/models/tfcoil/quench.py:185`."""

_CU_RHO_REF = 1.553e-8
"""`p9` in both `_copper_rrr_resistivity` and `_copper_magneto_resistivity`
(`process/models/tfcoil/quench.py:92,180`)."""

_CU_MAGNETORESISTANCE_FIELD_FLOOR = 1e-2
"""`process/models/tfcoil/quench.py:184`: below this field the magnetoresistance
correction is skipped entirely. The source's own comment calls the cut-off *"strange,
but necessary"*."""


def copper_specific_heat_capacity(temperature):
    """Cryogenic copper specific heat capacity, J/(kg K).

    Ports `_copper_specific_heat_capacity`, `process/models/tfcoil/quench.py:22-51`.
    NIST Monograph 177 eq. 7-1.
    """
    logt = jnp.log10(temperature)
    logcp = sum(c * logt**i for i, c in enumerate(_CU_CP_POLY))
    return 10**logcp


def copper_rrr_resistivity(temperature, rrr):
    """Copper electrical resistivity from temperature and RRR, ohm m.

    Ports `_copper_rrr_resistivity`, `process/models/tfcoil/quench.py:54-105`, constants
    and the `p4 = -1.14` sign the source flags as *"a typo in the original papers"*
    included unchanged.
    """
    p1 = 1.171e-17
    p2 = 4.49
    p3 = 3.841e10
    p4 = -1.14
    p5 = 50.0
    p6 = 6.428
    p7 = 0.4531
    rho_c = 0.0

    t = temperature

    # `exp(-(p5 / t) ** p6)` is `exp(-inf) == 0` at `t == 0`, which is the value PROCESS
    # produces and is right -- but its *derivative* there is `inf * 0 == nan`, and the
    # `t ** (p2 + p4)` factor in front is `0`, so the whole term's tangent comes out
    # `nan` while its value is exactly `0`. The `_audit/next_steps.md` §9 trap, one
    # `exp` away from its usual `x ** p` shape. Guarded by substitution (the double-
    # `where` idiom of `models/safe_math.py`): at `t == 0` the exponential is evaluated
    # at a dummy `1.0` and multiplied by a `t ** 3.35` that is exactly `0`, so the value
    # is unchanged and the tangent is finite.
    at_zero = t == 0.0
    t_safe = jnp.where(at_zero, 1.0, t)

    rho_o = _CU_RHO_REF / rrr
    numerator = p1 * t**p2
    denominator = 1.0 + p1 * p3 * t ** (p2 + p4) * jnp.exp(-((p5 / t_safe) ** p6))
    rho_i = numerator / denominator + rho_c
    rho_io = p7 * rho_i * rho_o / (rho_i + rho_o)

    return rho_o + rho_i + rho_io


def copper_irradiation_resistivity(fluence):
    """Radiation-induced copper resistivity, ohm m.

    Ports `_copper_irradiation_resistivity`, `process/models/tfcoil/quench.py:108-150`.
    """
    c1 = 0.00283
    c2 = -0.0711
    c3 = 0.77982
    res_scale = 1e-9
    flu_scale = 1e-22

    fluence_norm = flu_scale * fluence
    return res_scale * (c1 * fluence_norm**3 + c2 * fluence_norm**2 + c3 * fluence_norm)


def copper_magneto_resistivity(resistivity, field):
    """Copper resistivity with the magnetoresistance correction, ohm m.

    Ports `_copper_magneto_resistivity`, `process/models/tfcoil/quench.py:153-191`.

    The source's `if field > 1e-2` is a branch on a *traced* quantity (the peak field is
    downstream of the plasma current), so it becomes `jnp.where`, and the `log10` inside
    the taken branch is guarded by substitution rather than by clipping -- with `field`
    at or below the floor the argument would be zero or negative and would leak a NaN
    into the gradient of the arm that is not selected.
    """
    above_floor = field > _CU_MAGNETORESISTANCE_FIELD_FLOOR
    safe_field = jnp.where(above_floor, field, 1.0)

    x = jnp.log10(_CU_RHO_REF * safe_field / resistivity)
    a = sum(c * x**i for i, c in enumerate(_CU_MAGNETORESISTANCE_POLY))

    return jnp.where(above_floor, resistivity * (1.0 + 10**a), resistivity)


def copper_electrical_resistivity(temperature, field, rrr, fluence):
    """Copper resistivity including RRR, irradiation and magnetoresistance, ohm m.

    Ports `_copper_electrical_resistivity`, `process/models/tfcoil/quench.py:194-240`.
    """
    rho_rrr = copper_rrr_resistivity(temperature, rrr)
    rho_irr = copper_irradiation_resistivity(fluence)
    return copper_magneto_resistivity(rho_rrr + rho_irr, field)


def nb3sn_specific_heat_capacity(temperature):
    """Nb3Sn specific heat capacity, J/(kg K), normal state.

    Ports `_nb3sn_specific_heat_capacity`, `process/models/tfcoil/quench.py:243-286`.
    """
    gamma = 0.1
    beta = 0.001
    cp_300 = 210.0

    cp_low = beta * temperature**3 + gamma * temperature

    # `1 / (1 / cp_300 + 1 / cp_low)` is `1 / inf == 0` at `cp_low == 0` -- the value
    # PROCESS produces -- with a `nan` tangent, the same §9 trap as
    # `copper_rrr_resistivity` above. Guarded by substitution rather than by
    # re-associating to `cp_300 * cp_low / (cp_300 + cp_low)`, which would be finite at
    # zero but would also change the last bits everywhere else.
    at_zero = cp_low == 0.0
    return jnp.where(
        at_zero, 0.0, 1.0 / (1.0 / cp_300 + 1.0 / jnp.where(at_zero, 1.0, cp_low))
    )


def quench_integrands_at_temperature(
    *, temperature, field, rrr, fluence, den_helium, cp_helium
):
    """The three per-material quench integrands at one temperature, J s/(m3 ohm m K).

    Ports `_quench_integrand_at_temperature`, `process/models/tfcoil/quench.py:296-338`
    -- **this is the CoolProp seam**. The source calls
    `FluidProperties.of("He", temperature=temperature, pressure=6e5)` inline and reads
    `.specific_heat_const_p` and `.density` off it; here those two numbers are ordinary
    arguments, so everything else in the chain is pure and traceable.

    Parameters
    ----------
    temperature :
        Temperature, K.
    field :
        Magnetic field, T.
    rrr :
        Copper residual resistance ratio.
    fluence :
        Fast-neutron fluence, 1/m2.
    den_helium :
        Helium density at `(temperature, QUENCH_HELIUM_PRESSURE_PA)`, kg/m3 --
        CoolProp `PropsSI("D", "T", temperature, "P", 6e5, "He")`.
    cp_helium :
        Helium isobaric specific heat at the same state, J/(kg K) --
        CoolProp `PropsSI("C", ...)`.

    Returns
    -------
    :
        `(helium, copper, superconductor)` integrands, each `rho * cp / rho_cu`.
    """
    nu_cu = copper_electrical_resistivity(temperature, field, rrr, fluence)

    ihe_integrand = cp_helium * den_helium / nu_cu
    icu_integrand = (
        copper_specific_heat_capacity(temperature) * COPPER_DENSITY_QUENCH / nu_cu
    )
    isc_integrand = nb3sn_specific_heat_capacity(temperature) * NB3SN_DENSITY / nu_cu

    return ihe_integrand, icu_integrand, isc_integrand


def quench_quadrature_temperatures(*, temp_he_peak, temp_quench_max):
    """The 75 temperatures the quadrature evaluates at, K.

    `process/models/tfcoil/quench.py:389`. Exposed on its own because it is exactly the
    list of states CoolProp has to be asked about -- a caller resolving the property
    boundary evaluates helium once per entry of this array and passes the two resulting
    vectors to `quench_integrals` below.
    """
    nodes = jnp.asarray(QUENCH_GAUSS_LEGENDRE_NODES)
    return 0.5 * (nodes + 1.0) * (temp_quench_max - temp_he_peak) + temp_he_peak


def quench_integrals(
    *,
    temp_he_peak,
    temp_quench_max,
    field,
    rrr,
    fluence,
    den_helium_at_nodes,
    cp_helium_at_nodes,
):
    """The three material-property integrals over `[temp_he_peak, temp_quench_max]`.

    Ports `_quench_integrals`, `process/models/tfcoil/quench.py:341-400`. The source's
    Python loop over `zip(GAUSS_LEG_NODES, GAUSS_LEG_WEIGHTS)` becomes a vectorised
    weighted sum; the nodes and weights are the same `numpy` array PROCESS builds.

    Parameters
    ----------
    den_helium_at_nodes, cp_helium_at_nodes :
        Helium density and isobaric specific heat at
        `quench_quadrature_temperatures(...)`, in that order -- shape
        `(N_QUENCH_GAUSS_LEGENDRE_NODES,)`. See the module docstring for what this costs
        in CoolProp calls (150 once, 0 thereafter) and why.

    Returns
    -------
    :
        `(i_he, i_cu, i_sc)`.
    """
    weights = jnp.asarray(QUENCH_GAUSS_LEGENDRE_WEIGHTS)
    temperatures = quench_quadrature_temperatures(
        temp_he_peak=temp_he_peak, temp_quench_max=temp_quench_max
    )
    dti = 0.5 * weights * (temp_quench_max - temp_he_peak)

    ihe, icu, isc = quench_integrands_at_temperature(
        temperature=temperatures,
        field=field,
        rrr=rrr,
        fluence=fluence,
        den_helium=den_helium_at_nodes,
        cp_helium=cp_helium_at_nodes,
    )
    return jnp.sum(dti * ihe), jnp.sum(dti * icu), jnp.sum(dti * isc)


def calculate_quench_protection_current_density(
    *,
    tau_discharge,
    b_peak,
    f_a_cable_copper,
    f_a_cable_space_helium,
    temp_he_peak,
    temp_quench_max,
    cu_rrr,
    t_quench_detection,
    fluence,
    den_helium_at_nodes,
    cp_helium_at_nodes,
):
    """Hotspot-criterion current density limit, A/m2 (of cable cross-section).

    Ports `calculate_quench_protection_current_density`,
    `process/models/tfcoil/quench.py:474-551`. The fluence clip
    (`quench.py:533-537`) is kept as `jnp.clip`; PROCESS's accompanying `logger.warning`
    is dropped.

    See the module docstring for the two helium arrays.
    """
    fluence = jnp.clip(fluence, *FLUENCE_MODEL_RANGE)

    i_he, i_cu, i_sc = quench_integrals(
        temp_he_peak=temp_he_peak,
        temp_quench_max=temp_quench_max,
        field=b_peak,
        rrr=cu_rrr,
        fluence=fluence,
        den_helium_at_nodes=den_helium_at_nodes,
        cp_helium_at_nodes=cp_helium_at_nodes,
    )

    f_cu_cable = (1.0 - f_a_cable_space_helium) * f_a_cable_copper
    f_sc_cable = (1.0 - f_a_cable_space_helium) * (1.0 - f_a_cable_copper)

    factor = 1.0 / (0.5 * tau_discharge + t_quench_detection)
    total_integral = (
        f_a_cable_space_helium * i_he + f_cu_cable * i_cu + f_sc_cable * i_sc
    )

    return safe_sqrt(factor * f_cu_cable * total_integral)


def j_tf_wp_quench_heat_max(
    *,
    a_tf_turn_cable_space,
    a_tf_turn,
    tau_discharge,
    b_peak,
    f_a_cable_copper,
    f_a_cable_space_helium,
    temp_he_peak,
    temp_quench_max,
    cu_rrr,
    t_quench_detection,
    fluence,
    den_helium_at_nodes,
    cp_helium_at_nodes,
):
    """Winding-pack current density limited by the hotspot criterion, A/m2.

    Ports `quench_heat_protection_current_density`'s second half,
    `process/models/tfcoil/superconducting.py:1362-1377` -- one cable-to-turn area ratio
    over `calculate_quench_protection_current_density` above.
    """
    return (
        a_tf_turn_cable_space
        / a_tf_turn
        * calculate_quench_protection_current_density(
            tau_discharge=tau_discharge,
            b_peak=b_peak,
            f_a_cable_copper=f_a_cable_copper,
            f_a_cable_space_helium=f_a_cable_space_helium,
            temp_he_peak=temp_he_peak,
            temp_quench_max=temp_quench_max,
            cu_rrr=cu_rrr,
            t_quench_detection=t_quench_detection,
            fluence=fluence,
            den_helium_at_nodes=den_helium_at_nodes,
            cp_helium_at_nodes=cp_helium_at_nodes,
        )
    )


def helium_properties_at_quench_nodes(*, temp_he_peak, temp_quench_max):
    """`(density, specific heat)` of helium at the 75 quadrature nodes -- **the whole
    CoolProp surface of the tokamak scope, called once, outside every traced region.**

    This is the function `quench.md` OQ1 was asking for and deliberately did not write.
    It is `numpy`, not `jnp`, and it is called by `indat.py` while the machine is being
    assembled, never by a node body: `TfCoilQuenchHeatCurrentDensity` receives its two
    return values as a **static** field. See that class for the decision and its guard.

    150 `PropsSI` calls the first time a `(temp_he_peak, temp_quench_max)` pair is seen
    and none afterwards -- `process/core/coolprop_interface.py` memoises on
    `(T, P, fluid)` with `functools.cache`, and the grid is a pure function of those two
    numbers.
    """
    temperatures = np.asarray(
        quench_quadrature_temperatures(
            temp_he_peak=temp_he_peak, temp_quench_max=temp_quench_max
        )
    )
    states = [
        FluidProperties.of(
            "He", temperature=float(t), pressure=QUENCH_HELIUM_PRESSURE_PA
        )
        for t in temperatures
    ]
    return (
        tuple(float(s.density) for s in states),
        tuple(float(s.specific_heat_const_p) for s in states),
    )


class TfCoilQuenchHeatCurrentDensity(ExplicitFunction):
    """cottax node: `.tfcoil.j_tf_wp_quench_heat_max`, constraint 35's read.

    ## The CoolProp policy call, made (2026-08-27)

    `quench.md` OQ1 listed three resolutions for the helium property boundary and
    declined to pick one. This class picks **(a) -- the property table is a constant of
    the run, resolved at graph-assembly time and carried as a static field** -- and the
    reason is the measurement that record already contains rather than a preference:

    - The quadrature grid is a function of `.tfcoil.tftmp` and
      `.tfcoil.temp_tf_conductor_quench_max` and nothing else.
    - **Neither is written by any model** (`grep` over `process/models/**` finds no
      assignment to either; the only site is the read at `superconducting.py:2785`) and
      **neither is an iteration variable**. So the states CoolProp is asked about do not
      move during a solve, a scan point, or a gradient sweep.
    - Therefore the 150 numbers are constants of the machine in the same sense that
      `models/stellarator/preset_config.py`'s `machine_config` is -- and they are
      carried the same way, as an `eqx.field(static=True)` filled by `indat.py` from
      the file's own values.

    What that buys, against the two options not taken: a `pure_callback` would put a
    host round-trip inside a `jit` region to fetch numbers that cannot change, and a fit
    or interpolation would introduce an approximation where an exact table is already
    available at zero marginal cost. Nothing is approximated here, and no CoolProp call
    happens inside any traced or differentiated region.

    ## What the decision costs, stated rather than hidden

    `tftmp` and `temp_tf_conductor_quench_max` are **not reads of this node**, so the
    graph carries no edge from them and `jax` sees no derivative with respect to either.
    That is exactly right while they are inputs and exactly wrong the moment one is not,
    which is why `indat.py` **refuses to assemble** a machine whose `ixc` contains either
    (`_quench_helium_table`). The refusal is the thing that makes the static field safe;
    without it this would be the same shape of defect as the `dcond[0]` bake
    `superconducting.md` records for the mass slot.

    The harness contracts already excluded both from differentiation for this same
    reason (`tests/functional_process/models/tfcoil/test_quench.py`'s `_SEAM_STATICS`),
    so the node's declared reads and the case's differentiated arguments now say the
    same thing.
    """

    tftmp: float = eqx.field(static=True)
    """The helium peak-field temperature, as a static value rather than a read.

    **It is not only the property grid that depends on it** -- the quadrature interval
    `[tftmp, temp_tf_conductor_quench_max]` sets the nodes and the weights too, so
    freezing the table without freezing the interval would evaluate the integrand at
    temperatures the properties do not belong to. Both endpoints are static together, or
    neither.

    Named for the `DataStructure` field rather than for the pure function's parameter
    (`temp_he_peak`) **on purpose**: `mda_harness.switch_audit` resolves a static
    kwarg's backing field by name, so spelling it `tftmp` makes the frozen value
    value-checked against the converged run automatically, on every harness run. A
    static field with no resolvable name is a value nothing can check."""

    temp_tf_conductor_quench_max: float = eqx.field(static=True)
    """The hotspot temperature limit. See `tftmp`, including on the naming."""

    den_helium_at_nodes: tuple = eqx.field(static=True)
    """Helium density at `quench_quadrature_temperatures(...)`, from
    `helium_properties_at_quench_nodes`. A `tuple` and not an array because a static
    field has to be hashable -- it is a jit cache key."""

    cp_helium_at_nodes: tuple = eqx.field(static=True)
    """Helium isobaric specific heat at the same nodes."""

    j_tf_wp_quench_heat_max = OutputInto(tfcoil)

    def __call__(
        self,
        a_tf_turn_cable_space_no_void=From(tfcoil),
        a_tf_turn=From(tfcoil),
        t_tf_superconductor_quench=From(tfcoil),
        b_tf_inboard_peak_with_ripple=From(tfcoil),
        f_a_tf_turn_cable_copper=From(tfcoil),
        f_a_tf_turn_cable_space_cooling=From(superconducting_tfcoil),
        rrr_tf_cu=From(tfcoil),
        t_tf_quench_detection=From(tfcoil),
        flu_tf_neutron_fast_max=From(constraints),
    ):
        return j_tf_wp_quench_heat_max(
            a_tf_turn_cable_space=a_tf_turn_cable_space_no_void,
            a_tf_turn=a_tf_turn,
            tau_discharge=t_tf_superconductor_quench,
            b_peak=b_tf_inboard_peak_with_ripple,
            f_a_cable_copper=f_a_tf_turn_cable_copper,
            f_a_cable_space_helium=f_a_tf_turn_cable_space_cooling,
            temp_he_peak=self.tftmp,
            temp_quench_max=self.temp_tf_conductor_quench_max,
            cu_rrr=rrr_tf_cu,
            t_quench_detection=t_tf_quench_detection,
            fluence=flu_tf_neutron_fast_max,
            den_helium_at_nodes=jnp.asarray(self.den_helium_at_nodes),
            cp_helium_at_nodes=jnp.asarray(self.cp_helium_at_nodes),
        )


# ---------------------------------------------------------------------------
# The half that has no property dependence at all
# ---------------------------------------------------------------------------


def tf_dump_voltage_peak(*, e_tf_coil_magnetic_stored, t_tf_quench_dump, c_tf_turn):
    """Peak discharge voltage imposed on one TF coil, V.

    Ports `quench_heat_protection_current_density`'s first half,
    `process/models/tfcoil/superconducting.py:1357-1360`, in full. No material
    property, no CoolProp, no approximation -- see the module docstring.
    """
    return 2.0 * e_tf_coil_magnetic_stored / (t_tf_quench_dump * c_tf_turn)


def v_tf_coil_dump_quench_kv(
    *, e_tf_coil_magnetic_stored, t_tf_superconductor_quench, c_tf_turn
):
    """TF coil quench dump voltage, kV. `superconducting.py:2793-2795`.

    The unit conversion is a separate line in `run` rather than part of the returning
    function, so it is kept separate here too -- `tf_dump_voltage_peak` is what
    PROCESS's own unit test (`tests/unit/models/tfcoil/test_sctfcoil.py::test_protect`,
    `expected_vd`) checks.
    """
    return (
        tf_dump_voltage_peak(
            e_tf_coil_magnetic_stored=e_tf_coil_magnetic_stored,
            t_tf_quench_dump=t_tf_superconductor_quench,
            c_tf_turn=c_tf_turn,
        )
        / 1.0e3
    )


class TfCoilDumpQuenchVoltage(ExplicitFunction):
    """cottax node: `.tfcoil.v_tf_coil_dump_quench_kv`, one of the slot's ten reads.

    Three reads, no switch, no property boundary. The companion output of the same
    PROCESS function, `.tfcoil.j_tf_wp_quench_heat_max`, gets **no node here**: it needs
    helium density and specific heat, which have no `VarPath`, and minting two is the
    orchestrator's call. See the module docstring.
    """

    v_tf_coil_dump_quench_kv = OutputInto(tfcoil)

    def __call__(
        self,
        e_tf_coil_magnetic_stored=From(tfcoil),
        t_tf_superconductor_quench=From(tfcoil),
        c_tf_turn=From(tfcoil),
    ):
        return v_tf_coil_dump_quench_kv(
            e_tf_coil_magnetic_stored=e_tf_coil_magnetic_stored,
            t_tf_superconductor_quench=t_tf_superconductor_quench,
            c_tf_turn=c_tf_turn,
        )
