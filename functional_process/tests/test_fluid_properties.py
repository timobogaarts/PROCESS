"""`functional_process/_vendor/fluid_properties.py` equals `process/core/coolprop_interface.py`.

§23.2's rule -- **vendor for runtime, assert equality in tests** -- applied to the one
piece of vendored *behaviour* in the port. The wrapper is a verbatim copy, so the
agreement asserted here is `==`, not `approx`: both call the same installed `CoolProp`
through the same `PropsSI` entry point, and any difference at all would mean the copy had
drifted rather than that floating point had.

**The range is the one `models/tfcoil/quench.py` actually queries**, read off that module
rather than invented:

- fluid `He`, always; pressure `QUENCH_HELIUM_PRESSURE_PA = 6.0e5` Pa, a literal in three
  places in `process/models/tfcoil/quench.py` and never an input;
- temperature: the 75 Gauss-Legendre nodes of `[tftmp, temp_tf_conductor_quench_max]`.
  `tftmp` is `4.2`, `4.5`, `4.75` or `20.0` across the seven regression inputs and
  `temp_tf_conductor_quench_max` is `150.0` on every one of them, so
  `_REAL_QUENCH_INTERVALS` covers all four real grids -- 300 states -- and
  `_SWEEP_TEMPERATURES` covers a denser 4-200 K span on top, so a future input file
  moving either endpoint is inside already-checked territory.

`test_all_nine_properties_agree` goes past what `quench.py` asks for on purpose: the
vendored file is the whole class, not the `D`/`C` subset, and a copy is only honestly a
copy if everything in it is checked. `test_vendored_source_is_verbatim` is the cheap
belt-and-braces version of the same claim.
"""

import inspect
from pathlib import Path

import numpy as np
import pytest

from functional_process._vendor import fluid_properties as ported
from functional_process.cottax.tfcoil.quench import (
    QUENCH_HELIUM_PRESSURE_PA,
    helium_properties_at_quench_nodes,
    quench_quadrature_temperatures,
)
from process.core import coolprop_interface as reference

_REAL_QUENCH_INTERVALS = (
    (4.2, 150.0),  # helias_5b.IN.DAT
    (4.5, 150.0),  # stellarator_helias.IN.DAT
    (4.75, 150.0),  # large_tokamak_eval / large_tokamak_nof / low_aspect_ratio_DEMO
    (20.0, 150.0),  # spherical_tokamak_eval / st_regression
)
"""Every `(tftmp, temp_tf_conductor_quench_max)` pair the seven regression inputs
produce. `temp_tf_conductor_quench_max` is never set by any of them, so it is
`tfcoil_variables.py`'s default `150.0` throughout; `st_regression.IN.DAT:1410` has the
name present but commented out."""

_SWEEP_TEMPERATURES = tuple(np.linspace(4.0, 200.0, 61))
"""A denser span than any real grid, over the same pressure. Helium at 6 bar is
supercritical (critical point 5.2 K / 2.27 bar), so there is no phase boundary inside
this interval for the two copies to land on different sides of."""

_NINE_PROPERTIES = (
    "temperature",
    "pressure",
    "density",
    "enthalpy",
    "entropy",
    "specific_heat_const_p",
    "specific_heat_const_v",
    "viscosity",
    "thermal_conductivity",
)
"""Every property `FluidProperties` exposes -- asserted below to *be* every property, so
one added upstream cannot slip past this file's coverage."""


def _states(temperature):
    """The same helium state, built through each copy's own `of` classmethod."""
    kwargs = {"temperature": float(temperature), "pressure": QUENCH_HELIUM_PRESSURE_PA}
    return (
        ported.FluidProperties.of("He", **kwargs),
        reference.FluidProperties.of("He", **kwargs),
    )


def test_the_nine_properties_are_the_whole_surface():
    """`_NINE_PROPERTIES` names every property on both classes.

    Without this, adding a property upstream would leave the sweep below quietly
    incomplete -- the §23.5 shape of defect, where an equality test passes because it
    never looked.
    """
    for cls in (ported.FluidProperties, reference.FluidProperties):
        exposed = {
            name
            for name, member in vars(cls).items()
            if isinstance(member, property) and not name.startswith("_")
        }
        assert exposed == set(_NINE_PROPERTIES), cls


@pytest.mark.parametrize("temperature", _SWEEP_TEMPERATURES)
def test_density_and_specific_heat_agree_over_the_queried_range(temperature):
    """The entire surface `quench.py` uses -- `PropsSI("D", ...)`, `PropsSI("C", ...)`.

    Exact equality, over 4-200 K at 6 bar helium.
    """
    port, ref = _states(temperature)
    assert port.density == ref.density
    assert port.specific_heat_const_p == ref.specific_heat_const_p


@pytest.mark.parametrize("temperature", [4.2, 20.0, 77.0, 150.0, 200.0])
def test_all_nine_properties_agree(temperature):
    """The vendored file is the whole class, so the whole class is checked."""
    port, ref = _states(temperature)
    for name in _NINE_PROPERTIES:
        assert getattr(port, name) == getattr(ref, name), name


@pytest.mark.parametrize(("temp_he_peak", "temp_quench_max"), _REAL_QUENCH_INTERVALS)
def test_quench_node_table_is_identical(temp_he_peak, temp_quench_max):
    """The shipped table -- 75 nodes x 2 properties -- against PROCESS's own.

    This is the function `indat._quench_helium_table` calls at tokamak-assembly time, so
    what is compared here is exactly what the assembled machine carries as a static
    field, not a re-derivation of it.
    """
    den, cp = helium_properties_at_quench_nodes(
        temp_he_peak=temp_he_peak, temp_quench_max=temp_quench_max
    )
    temperatures = np.asarray(
        quench_quadrature_temperatures(
            temp_he_peak=temp_he_peak, temp_quench_max=temp_quench_max
        )
    )
    expected = [
        reference.FluidProperties.of(
            "He", temperature=float(t), pressure=QUENCH_HELIUM_PRESSURE_PA
        )
        for t in temperatures
    ]
    assert len(den) == len(cp) == len(expected) == 75
    assert list(den) == [float(s.density) for s in expected]
    assert list(cp) == [float(s.specific_heat_const_p) for s in expected]


def test_other_state_specifications_agree():
    """`of`'s pressure/entropy and pressure/vapor-quality arms.

    `quench.py` never takes either, but the vendored copy still carries them.
    """
    ref_ps = reference.FluidProperties.of("He", pressure=1.0e5, entropy=2000.0)
    port_ps = ported.FluidProperties.of("He", pressure=1.0e5, entropy=2000.0)
    assert port_ps.temperature == ref_ps.temperature
    assert port_ps.density == ref_ps.density

    ref_pq = reference.FluidProperties.of("He", pressure=1.0e5, vapor_quality=0.5)
    port_pq = ported.FluidProperties.of("He", pressure=1.0e5, vapor_quality=0.5)
    assert port_pq.temperature == ref_pq.temperature
    assert port_pq.enthalpy == ref_pq.enthalpy


def test_water_agrees_too():
    """Not a helium-only wrapper. `fluid_name.title()` and the whole path are shared."""
    port = ported.FluidProperties.of("water", temperature=300.0, pressure=1.0e5)
    ref = reference.FluidProperties.of("water", temperature=300.0, pressure=1.0e5)
    for name in _NINE_PROPERTIES:
        assert getattr(port, name) == getattr(ref, name), name


def test_vendored_source_is_verbatim():
    """The copy is byte-for-byte, module docstring aside.

    `constants.py` and `stellarator_presets.py` were vendored this way and the value
    tests are what actually guard them -- but for a module whose values come out of a C
    library, "the source is identical" is a second, independent statement, and it is one
    line to make.
    """
    ported_body = Path(inspect.getfile(ported)).read_text()
    reference_body = Path(inspect.getfile(reference)).read_text()
    marker = "from functools import cache"
    assert (
        ported_body[ported_body.index(marker) :]
        == (reference_body[reference_body.index(marker) :])
    )
