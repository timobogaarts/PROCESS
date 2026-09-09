"""Pure-functional port of the tokamak TF quench chain --
`CICCSuperconductingTFCoil.quench_heat_protection_current_density`
(`process/models/tfcoil/superconducting.py:1298-1379`) and the material physics it calls
into (`process/models/tfcoil/quench.py`).
"""

import equinox as eqx
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.cottax.paths import constraints, superconducting_tfcoil, tfcoil
from functional_process.models.tfcoil.quench import (
    QUENCH_HELIUM_PRESSURE_PA,  # noqa: F401 -- re-exported for tests
    calculate_quench_protection_current_density,  # noqa: F401 -- re-exported for tests
    calculate_tf_coil_quench_heat_current_density,
    copper_electrical_resistivity,  # noqa: F401 -- re-exported for tests
    copper_irradiation_resistivity,  # noqa: F401 -- re-exported for tests
    copper_magneto_resistivity,  # noqa: F401 -- re-exported for tests
    copper_rrr_resistivity,  # noqa: F401 -- re-exported for tests
    copper_specific_heat_capacity,  # noqa: F401 -- re-exported for tests
    helium_properties_at_quench_nodes,  # noqa: F401 -- re-exported for indat.py / tests
    j_tf_wp_quench_heat_max,  # noqa: F401 -- re-exported for tests
    nb3sn_specific_heat_capacity,  # noqa: F401 -- re-exported for tests
    quench_integrals,  # noqa: F401 -- re-exported for tests
    quench_quadrature_temperatures,  # noqa: F401 -- re-exported for tests
    tf_dump_voltage_peak,  # noqa: F401 -- re-exported for tests
    v_tf_coil_dump_quench_kv,
)


class TfCoilQuenchHeatCurrentDensity(ExplicitFunction):
    """cottax node: `.tfcoil.j_tf_wp_quench_heat_max`, constraint 35's read."""

    tftmp: float = eqx.field(static=True)
    """The helium peak-field temperature, as a static value rather than a read."""

    temp_tf_conductor_quench_max: float = eqx.field(static=True)
    """The hotspot temperature limit. See `tftmp`, including on the naming."""

    den_helium_at_nodes: tuple = eqx.field(static=True)
    """Helium density at `quench_quadrature_temperatures(...)`, from
    `helium_properties_at_quench_nodes`.
    """

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
        return calculate_tf_coil_quench_heat_current_density(
            a_tf_turn_cable_space_no_void,
            a_tf_turn,
            t_tf_superconductor_quench,
            b_tf_inboard_peak_with_ripple,
            f_a_tf_turn_cable_copper,
            f_a_tf_turn_cable_space_cooling,
            self.tftmp,
            self.temp_tf_conductor_quench_max,
            rrr_tf_cu,
            t_tf_quench_detection,
            flu_tf_neutron_fast_max,
            self.den_helium_at_nodes,
            self.cp_helium_at_nodes,
        )


class TfCoilDumpQuenchVoltage(ExplicitFunction):
    """cottax node: `.tfcoil.v_tf_coil_dump_quench_kv`, one of the slot's ten reads."""

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
