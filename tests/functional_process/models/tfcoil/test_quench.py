"""Harness cases for the ported TF quench chain
(`functional_process/models/tfcoil/quench.py`).

Two things are being checked here and they are not the same thing.

1. **The dump voltage is a complete port.** `TestTfDumpVoltagePeak` calls PROCESS's own
   `quench_heat_protection_current_density` -- the function
   `tokamak_boundary.md` says puts `.tfcoil.v_tf_coil_dump_quench_kv` behind CoolProp --
   and diffs its *second* return against a three-argument pure function. That the diff
   is exact, and that the port takes only three of PROCESS's thirteen arguments, is the
   evidence for the port module's finding: the voltage does not depend on any material
   property.

2. **The current density is a port up to a declared seam.** The helium density and
   isobaric specific heat are handed to the port as arrays evaluated at the quadrature
   nodes; the reference computes them itself, from CoolProp. Agreement in value with the
   *same* numbers is what says the rest of the chain -- copper resistivity,
   heat capacities, Gauss-Legendre quadrature, the hotspot formula -- is faithfully
   ported, which is exactly the claim that survives whatever the property boundary is
   eventually resolved to.

**Four arguments are `static_argnames` on the seam contracts, and the reason is
structural rather than a convenience.** `den_helium_at_nodes`/`cp_helium_at_nodes` are
inputs the *reference does not have*: differentiating them would compare a real
derivative against a structurally-zero one. `temp_he_peak`/`temp_quench_max` are worse
than that -- they set the quadrature grid, so perturbing them moves the states the
helium properties should have been evaluated at, and the harness has no way to move the
supplied arrays with them. The port's derivative with respect to those two is therefore
**not defined until the property boundary is resolved**, and saying so by excluding them
is more honest than differentiating a frozen table and calling the answer a gradient.
Every other argument is differentiated normally. Recorded in `quench.md` as the one open
item this unit hands back.
"""

import numpy as np

from functional_process._harness import Tier1Contract, legacy_sample
from functional_process.models.tfcoil.quench import (
    QUENCH_HELIUM_PRESSURE_PA,
    calculate_quench_protection_current_density,
    copper_electrical_resistivity,
    copper_irradiation_resistivity,
    copper_magneto_resistivity,
    copper_rrr_resistivity,
    copper_specific_heat_capacity,
    nb3sn_specific_heat_capacity,
    quench_integrals,
    quench_quadrature_temperatures,
    tf_dump_voltage_peak,
    v_tf_coil_dump_quench_kv,
)
from process.core.coolprop_interface import FluidProperties
from process.models.tfcoil import quench as process_quench
from process.models.tfcoil.superconducting import CICCSuperconductingTFCoil

_SEAM_STATICS = (
    "temp_he_peak",
    "temp_quench_max",
    "den_helium_at_nodes",
    "cp_helium_at_nodes",
)
"""See the module docstring. Not a convenience: two of the four have no reference
counterpart at all, and the other two cannot be perturbed without moving the supplied
property table with them."""


def _helium_properties(temp_he_peak, temp_quench_max):
    """Helium density and isobaric specific heat at the 75 quadrature nodes.

    This function **is** the CoolProp call surface of the whole tokamak scope: one
    fluid, two properties, `(T, P)` with `P` fixed at
    `QUENCH_HELIUM_PRESSURE_PA`. Written out here rather than hidden inside the port,
    so the boundary is visible in the test as well as in the module docstring.
    """
    temperatures = np.asarray(
        quench_quadrature_temperatures(
            temp_he_peak=temp_he_peak, temp_quench_max=temp_quench_max
        )
    )
    density = np.array([
        FluidProperties.of(
            "He", temperature=float(t), pressure=QUENCH_HELIUM_PRESSURE_PA
        ).density
        for t in temperatures
    ])
    specific_heat = np.array([
        FluidProperties.of(
            "He", temperature=float(t), pressure=QUENCH_HELIUM_PRESSURE_PA
        ).specific_heat_const_p
        for t in temperatures
    ])
    return density, specific_heat


_QUENCH_POINT = {
    "tau_discharge": 23.0,
    "b_peak": 11.0,
    "f_a_cable_copper": 0.7,
    "f_a_cable_space_helium": 0.2,
    "temp_he_peak": 4.75,
    "temp_quench_max": 150.0,
    "cu_rrr": 100.0,
    "t_quench_detection": 1.0,
    "fluence": 3e21,
}
"""`tests/unit/models/tfcoil/test_quench.py:158`'s
`test_calculate_quench_protection_intuitive_gradient` point, verbatim. That test checks
the *signs* of nine derivatives with `scipy.optimize.approx_fprime`; this harness checks
their values against PROCESS's own difference scheme, which is strictly the stronger
claim at the same point."""

_QUENCH_HELIUM = dict(
    zip(
        ("den_helium_at_nodes", "cp_helium_at_nodes"),
        _helium_properties(
            _QUENCH_POINT["temp_he_peak"], _QUENCH_POINT["temp_quench_max"]
        ),
        strict=True,
    )
)


# ---------------------------------------------------------------------------
# Material properties
# ---------------------------------------------------------------------------


class TestCopperSpecificHeatCapacity(Tier1Contract):
    """`_copper_specific_heat_capacity` -> the same, `np.` -> `jnp.`.

    Sample temperatures are `test_copper_specific_heat_capacity`'s NIST table
    (`tests/unit/models/tfcoil/test_quench.py:97-133`) -- there, compared against
    measured data at `rtol=0.24`; here, against PROCESS's own evaluation at machine
    precision, which is what a port has to reproduce.
    """

    audit_record = "models/tfcoil/quench.md"
    reference = staticmethod(process_quench._copper_specific_heat_capacity)
    ported = copper_specific_heat_capacity

    samples = [
        legacy_sample(
            "copper-cp-nist-table",
            temperature=np.array([
                4.0,
                6.0,
                8.0,
                10.0,
                14.0,
                20.0,
                30.0,
                50.0,
                80.0,
                100.0,
                120.0,
                160.0,
                200.0,
                260.0,
            ]),
        ),
    ]


class TestCopperRrrResistivity(Tier1Contract):
    """`_copper_rrr_resistivity` -> the same.

    `test_copper_rrr_resistivity` checks a ratio at `t = 4.0` for three RRRs; the same
    three points are used here, as one vector sample.
    """

    audit_record = "models/tfcoil/quench.md"
    reference = staticmethod(process_quench._copper_rrr_resistivity)
    ported = copper_rrr_resistivity

    samples = [
        legacy_sample(
            "copper-rrr-4K",
            temperature=np.array([4.0, 4.0, 4.0]),
            rrr=np.array([100.0, 300.0, 1000.0]),
        ),
        legacy_sample(
            "copper-rrr-quench-range",
            temperature=np.array([4.75, 40.0, 150.0]),
            rrr=np.array([100.0, 100.0, 100.0]),
        ),
    ]


class TestCopperIrradiationResistivity(Tier1Contract):
    """`_copper_irradiation_resistivity` -> the same.

    Sample is the Nakagawa fluence series `test_copper_irradiation` reads off Figure 6
    (`tests/unit/models/tfcoil/test_quench.py:26-55`), verbatim.
    """

    audit_record = "models/tfcoil/quench.md"
    reference = staticmethod(process_quench._copper_irradiation_resistivity)
    ported = copper_irradiation_resistivity

    samples = [
        legacy_sample(
            "copper-irradiation-nakagawa",
            fluence=1e20
            * np.array([
                0.24,
                0.45,
                0.60,
                0.79,
                1.06,
                1.26,
                1.56,
                1.89,
                2.23,
                2.68,
                2.92,
                3.13,
                3.30,
                3.59,
                3.90,
                4.20,
                4.71,
                5.04,
                5.34,
                5.56,
                5.82,
                6.03,
                6.23,
                6.58,
                6.86,
                7.38,
                7.75,
                8.11,
                8.40,
                8.51,
            ]),
        ),
    ]


class TestCopperMagnetoResistivity(Tier1Contract):
    """`_copper_magneto_resistivity` -> the same, with the field cut-off as `jnp.where`.

    Both sides of the `field > 1e-2` cut-off are sampled -- `test_copper_magneto_
    resistivity` uses `0.0`, `10.0` and `20.0` T, and the zero-field case is the one
    that would expose a `jnp.where` guarding the wrong way round.
    """

    audit_record = "models/tfcoil/quench.md"
    reference = staticmethod(process_quench._copper_magneto_resistivity)
    ported = copper_magneto_resistivity

    samples = [
        legacy_sample(
            "copper-magneto-10T",
            resistivity=process_quench._copper_rrr_resistivity(4.0, 100.0),
            field=10.0,
        ),
        legacy_sample(
            "copper-magneto-20T",
            resistivity=process_quench._copper_rrr_resistivity(4.0, 100.0),
            field=20.0,
        ),
        legacy_sample(
            "copper-magneto-below-cutoff",
            resistivity=process_quench._copper_rrr_resistivity(4.0, 100.0),
            field=0.0,
        ),
    ]


class TestCopperElectricalResistivity(Tier1Contract):
    """`_copper_electrical_resistivity` -> the same: the three pieces composed.

    **Scalar samples, one per temperature, and that is forced by the reference.**
    `_copper_magneto_resistivity`'s `if field > 1e-2`
    (`process/models/tfcoil/quench.py:184`)
    raises `ValueError: The truth value of an array ... is ambiguous` on any array
    `field`, so PROCESS cannot be called vectorised here at all. The port *is*
    vectorised (the branch is a `jnp.where`), which is why its own caller
    `quench_integrands_at_temperature` evaluates all 75 quadrature temperatures in one
    go where PROCESS loops. Recorded rather than worked around silently: the ported
    function is strictly more general than the one it is diffed against, and the diff
    can only be taken on the intersection.
    """

    audit_record = "models/tfcoil/quench.md"
    reference = staticmethod(process_quench._copper_electrical_resistivity)
    ported = copper_electrical_resistivity

    samples = [
        legacy_sample(
            f"copper-resistivity-{temperature}K",
            temperature=temperature,
            field=11.0,
            rrr=100.0,
            fluence=3e21,
        )
        for temperature in (4.75, 40.0, 150.0)
    ]


class TestNb3snSpecificHeatCapacity(Tier1Contract):
    """`_nb3sn_specific_heat_capacity` -> the same.

    PROCESS has no unit test for it; the sample is the quench temperature range the
    quadrature actually spans, which is the only place the function is ever evaluated.
    """

    audit_record = "models/tfcoil/quench.md"
    reference = staticmethod(process_quench._nb3sn_specific_heat_capacity)
    ported = nb3sn_specific_heat_capacity

    samples = [
        legacy_sample(
            "nb3sn-cp-quench-range",
            temperature=np.array([4.75, 10.0, 40.0, 80.0, 150.0]),
        ),
    ]

    fuzz_bounds = {"temperature": (4.0, 300.0)}


# ---------------------------------------------------------------------------
# The quadrature and the hotspot criterion -- across the seam
# ---------------------------------------------------------------------------


def _reference_quench_integrals(
    temp_he_peak,
    temp_quench_max,
    field,
    rrr,
    fluence,
    den_helium_at_nodes,
    cp_helium_at_nodes,
):
    """`_quench_integrals`, ignoring the supplied helium table.

    PROCESS looks the properties up itself; the arrays are accepted and dropped so that
    the two signatures line up. `_SEAM_STATICS` keeps them out of every differentiation,
    so no derivative is ever taken through a dropped argument.
    """
    del den_helium_at_nodes, cp_helium_at_nodes
    return process_quench._quench_integrals(
        temp_he_peak, temp_quench_max, field, rrr, fluence
    )


class TestQuenchIntegrals(Tier1Contract):
    """The 75-node Gauss-Legendre quadrature, vectorised.

    PROCESS loops in Python over `zip(GAUSS_LEG_NODES, GAUSS_LEG_WEIGHTS)`; the port
    evaluates the integrand at all 75 nodes at once and takes a weighted sum. Same
    nodes, same weights (imported from the same `numpy` call), so any disagreement here
    is the integrand, not the quadrature.
    """

    audit_record = "models/tfcoil/quench.md"
    reference = _reference_quench_integrals
    ported = quench_integrals
    static_argnames = _SEAM_STATICS

    samples = [
        legacy_sample(
            "quench-integrals-intuitive-gradient-point",
            temp_he_peak=_QUENCH_POINT["temp_he_peak"],
            temp_quench_max=_QUENCH_POINT["temp_quench_max"],
            field=_QUENCH_POINT["b_peak"],
            rrr=_QUENCH_POINT["cu_rrr"],
            fluence=_QUENCH_POINT["fluence"],
            **_QUENCH_HELIUM,
        ),
    ]


def _reference_quench_current_density(den_helium_at_nodes, cp_helium_at_nodes, **kwargs):
    """`calculate_quench_protection_current_density`, ignoring the helium table."""
    del den_helium_at_nodes, cp_helium_at_nodes
    return process_quench.calculate_quench_protection_current_density(**kwargs)


class TestCalculateQuenchProtectionCurrentDensity(Tier1Contract):
    """The hotspot criterion, end to end across the seam.

    Sample is `test_calculate_quench_protection_intuitive_gradient`'s point. That test
    asserts the signs of nine derivatives; this contract asserts their values against
    PROCESS's own finite difference for the five it can move without disturbing the
    quadrature grid (see the module docstring for the other four).
    """

    audit_record = "models/tfcoil/quench.md"
    reference = _reference_quench_current_density
    ported = calculate_quench_protection_current_density
    static_argnames = _SEAM_STATICS

    samples = [
        legacy_sample(
            "quench-j-intuitive-gradient-point", **_QUENCH_POINT, **_QUENCH_HELIUM
        )
    ]


# ---------------------------------------------------------------------------
# The dump voltage -- no seam at all
# ---------------------------------------------------------------------------


def _reference_dump_voltage(e_tf_coil_magnetic_stored, t_tf_quench_dump, c_tf_turn):
    """PROCESS's `quench_heat_protection_current_density`, second return only.

    Deliberately the *whole* PROCESS function rather than a transcription of its first
    two lines: the point being made is that its second output is independent of the ten
    arguments the port does not take, and the only way to make that point is to call it
    with those ten held fixed and watch the answer follow the three that are left.
    The ten fixed values are `test_protect`'s second case
    (`tests/unit/models/tfcoil/test_sctfcoil.py:100-110`).
    """
    return CICCSuperconductingTFCoil.quench_heat_protection_current_density(
        c_tf_turn=c_tf_turn,
        e_tf_coil_magnetic_stored=e_tf_coil_magnetic_stored,
        a_tf_turn_cable_space=0.001293323051622732,
        a_tf_turn=0.0032012300777680192,
        t_tf_quench_dump=t_tf_quench_dump,
        f_a_tf_turn_cable_space_cooling=1 - 0.63927285511442711,
        f_a_tf_turn_cable_copper=0.80884,
        temp_tf_coolant_peak_field=4.75,
        temp_tf_conductor_quench_max=150,
        b_tf_inboard_peak=11.0,
        cu_rrr=200.0,
        t_tf_quench_detection=3.0,
        flu_tf_neutron_fast_max=3.2e21,
    )[1]


def _reference_dump_voltage_kv(
    e_tf_coil_magnetic_stored, t_tf_superconductor_quench, c_tf_turn
):
    """The same, divided by 1000.

    `process/models/tfcoil/superconducting.py:2793-2795`.
    """
    return (
        _reference_dump_voltage(
            e_tf_coil_magnetic_stored=e_tf_coil_magnetic_stored,
            t_tf_quench_dump=t_tf_superconductor_quench,
            c_tf_turn=c_tf_turn,
        )
        / 1.0e3
    )


_DUMP_VOLTAGE_SAMPLE = {
    "e_tf_coil_magnetic_stored": 9561415368.8360519,
    "c_tf_turn": 74026.751437500003,
}
"""`test_protect`'s `tfes` and `aio`, from `baseline_2018_IN.DAT`."""


class TestTfDumpVoltagePeak(Tier1Contract):
    """`v_tf_dump_voltage_peak`: three reads, no material property, no CoolProp."""

    audit_record = "models/tfcoil/quench.md"
    reference = _reference_dump_voltage
    ported = tf_dump_voltage_peak

    samples = [
        legacy_sample(
            "dump-voltage-baseline2018",
            t_tf_quench_dump=25.829000000000001,
            **_DUMP_VOLTAGE_SAMPLE,
        ),
    ]

    fuzz_bounds = {
        "e_tf_coil_magnetic_stored": (1e9, 5e10),
        "t_tf_quench_dump": (0.1, 100.0),
        "c_tf_turn": (1e4, 1e5),
    }


class TestVTfCoilDumpQuenchKv(Tier1Contract):
    """`.tfcoil.v_tf_coil_dump_quench_kv` itself -- the boundary variable, in kV."""

    audit_record = "models/tfcoil/quench.md"
    reference = _reference_dump_voltage_kv
    ported = v_tf_coil_dump_quench_kv

    samples = [
        legacy_sample(
            "v_tf_coil_dump_quench_kv-baseline2018",
            t_tf_superconductor_quench=25.829000000000001,
            **_DUMP_VOLTAGE_SAMPLE,
        ),
    ]

    fuzz_bounds = {
        "e_tf_coil_magnetic_stored": (1e9, 5e10),
        "t_tf_superconductor_quench": (0.1, 100.0),
        "c_tf_turn": (1e4, 1e5),
    }
