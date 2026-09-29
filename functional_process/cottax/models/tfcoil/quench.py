"""Pure-functional port of the tokamak TF quench chain --
`CICCSuperconductingTFCoil.quench_heat_protection_current_density`
(`process/models/tfcoil/superconducting.py:1298-1379`) and the material physics it calls
into (`process/models/tfcoil/quench.py`).
"""

from functional_process.cottax.paths import tfcoil
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

# cottax node: `.tfcoil.j_tf_wp_quench_heat_max`, constraint 35's read.
#
# `tftmp` (the helium peak-field temperature) and `temp_tf_conductor_quench_max` (the
# hotspot temperature limit) are read at their PROCESS inputs. The helium density and
# isobaric specific heat at `quench_quadrature_temperatures(...)` are read at
# `.tfcoil.den_helium_at_nodes` / `.tfcoil.cp_helium_at_nodes`: tuples of floats from
# `helium_properties_at_quench_nodes`, the one CoolProp call of the tokamak scope, made
# while the machine is stated (static under jit, a non-array leaf).
TfCoilQuenchHeatCurrentDensity = tfcoil.function(
    calculate_tf_coil_quench_heat_current_density,
    "j_tf_wp_quench_heat_max",
)


# cottax node: `.tfcoil.v_tf_coil_dump_quench_kv`, one of the slot's ten reads.
TfCoilDumpQuenchVoltage = tfcoil.function(
    v_tf_coil_dump_quench_kv, "v_tf_coil_dump_quench_kv"
)
