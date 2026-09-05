"""Pure functions for the TF coil quench chain, extracted from
`functional_process/cottax/tfcoil/quench.py`.

That module still holds the graph declarations (`ExplicitFunction` occupants) that wire
these functions to `VarPath`s; read its module docstring for the CoolProp boundary and
scope notes. The audit record is
`functional_process/_audit/units/models/tfcoil/quench.md` and mirrors these functions,
not the declarations that call them.
"""

import jax.numpy as jnp
import numpy as np

from functional_process.models.safe_math import safe_sqrt

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
    and none afterwards -- `functional_process/fluid_properties.py` memoises on
    `(T, P, fluid)` with `functools.cache`, and the grid is a pure function of those two
    numbers.
    """
    temperatures = np.asarray(
        quench_quadrature_temperatures(
            temp_he_peak=temp_he_peak, temp_quench_max=temp_quench_max
        )
    )
    # The wrapper is vendored (`functional_process/fluid_properties.py`, a verbatim copy
    # of `process/core/coolprop_interface.py`, equality-tested against it in
    # `tests/functional_process/test_fluid_properties.py`), so this call no longer needs
    # `process` -- §23.6, the last runtime PROCESS import in the port.
    #
    # **The deferral stays, for a second reason.** It was here so that importing this
    # module needed no `process`; it is here now because `import CoolProp` costs ~3 s
    # (measured) and only a tokamak assembly ever wants the table. Do not lift it to
    # module scope.
    from functional_process.fluid_properties import FluidProperties  # noqa: PLC0415

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


def calculate_tf_coil_quench_heat_current_density(
    a_tf_turn_cable_space_no_void,
    a_tf_turn,
    t_tf_superconductor_quench,
    b_tf_inboard_peak_with_ripple,
    f_a_tf_turn_cable_copper,
    f_a_tf_turn_cable_space_cooling,
    tftmp,
    temp_tf_conductor_quench_max,
    rrr_tf_cu,
    t_tf_quench_detection,
    flu_tf_neutron_fast_max,
    den_helium_at_nodes,
    cp_helium_at_nodes,
):
    """`.tfcoil.j_tf_wp_quench_heat_max`'s own arm: the two static helium property
    tables are `tuple`s (hashable, for jit caching -- see `TfCoilQuenchHeatCurrentDensity`
    docstring), converted to `jnp` arrays here rather than at the call site, then handed
    unchanged to `j_tf_wp_quench_heat_max`.
    """
    return j_tf_wp_quench_heat_max(
        a_tf_turn_cable_space=a_tf_turn_cable_space_no_void,
        a_tf_turn=a_tf_turn,
        tau_discharge=t_tf_superconductor_quench,
        b_peak=b_tf_inboard_peak_with_ripple,
        f_a_cable_copper=f_a_tf_turn_cable_copper,
        f_a_cable_space_helium=f_a_tf_turn_cable_space_cooling,
        temp_he_peak=tftmp,
        temp_quench_max=temp_tf_conductor_quench_max,
        cu_rrr=rrr_tf_cu,
        t_quench_detection=t_tf_quench_detection,
        fluence=flu_tf_neutron_fast_max,
        den_helium_at_nodes=jnp.asarray(den_helium_at_nodes),
        cp_helium_at_nodes=jnp.asarray(cp_helium_at_nodes),
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
