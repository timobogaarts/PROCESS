"""PROCESS's CoolProp wrapper, vendored so the port runs with no `process` present.

**Deliberately not under `models/`.** That package is PROCESS's physics as pure
JAX -- importable with no graph machinery and no external library -- and CoolProp
is an opaque C extension, the one dependency `CLAUDE.md` names as not
JAX-traceable. Moving this module into `models/` was tried on 2026-09-05 and
`test_importing_the_model_layer_does_not_load_coolprop` failed immediately, which
is exactly what that test is for: importing the physics must not drag in a C
library. Its one consumer, `models/tfcoil/quench.py`, imports it lazily.

§23.5 named this the last runtime `process` import: `models/tfcoil/quench.py`'s
`helium_properties_at_quench_nodes` reached into `process.core.coolprop_interface`, and
because `indat.py` makes exactly one such lookup while assembling a **tokamak**, no
tokamak machine could be built PROCESS-free. Stellarators never call it. This module is
that import, cut.

**Why here and not in `vocabulary/`.** §23.1 sorted the model layer's PROCESS imports
into constants, enums, tables and presets -- *declarations* -- and §23.2's vendoring rule
was written for those. A CoolProp wrapper is **behaviour**: it calls a C library and
returns a number that no test can read off a source line. So it sits beside
`vocabulary/`, not inside it, and the §23.2 rule is applied to it anyway --
`functional_process/tests/test_fluid_properties.py` asserts this module agrees with
`process.core.coolprop_interface` **exactly** (`==`, not `approx`) over the temperatures
the quench chain actually queries.

**Verbatim.** The class and the nine memoised property functions below are a byte-for-
byte copy of `process/core/coolprop_interface.py` (`constants.py` and
`stellarator_presets.py` were vendored the same way). Nothing is renamed, reordered or
narrowed to the two properties `quench.py` uses: a delegating copy that stays a copy is
cheaper to keep honest than a subset that has to justify what it dropped, and the
equality test can then check all nine. `PropsSI` is deterministic and both copies call
the same installed `CoolProp` 8.0.0, so agreement is bit-identical rather than
approximate.

**`CoolProp` costs ~3 s to import** (measured, 2026-08-31), which is why `quench.py`
imports *this module* lazily rather than at module scope. The deferral it already had is
kept for the new reason as well as the old one: importing `functional_process` must not
pay three seconds for a table only a tokamak assembly ever asks for.

Not traceable, and deliberately not made so: CoolProp is an opaque C library and
`CLAUDE.md` records wrapping it in a JAX primitive as an open design question. This
change cuts an *import*; it changes no number.
"""

from functools import cache

from CoolProp.CoolProp import PropsSI


class FluidProperties:
    """Fluid properties setup"""

    def __init__(self, coolprop_inputs: list[str | float]):
        self._coolprop_inputs = tuple(coolprop_inputs)

    @property
    def temperature(self):
        """Fluid temperature [K]"""
        return _temperature(self._coolprop_inputs)

    @property
    def pressure(self):
        """Fluid pressure [Pa]"""
        return _pressure(self._coolprop_inputs)

    @property
    def density(self):
        """Fluid density [kg/m3]"""
        return _density(self._coolprop_inputs)

    @property
    def enthalpy(self):
        """Fluid specific enthalpy [J/kg]"""
        return _enthalpy(self._coolprop_inputs)

    @property
    def entropy(self):
        """Fluid entropy [J/kg/K]"""
        return _entropy(self._coolprop_inputs)

    @property
    def specific_heat_const_p(self):
        """Fluid specific heat capacity at constant pressure [J/kg/K]"""
        return _specific_heat_const_p(self._coolprop_inputs)

    @property
    def specific_heat_const_v(self):
        """Fluid specific heat capacity at constant volume [J/kg/K]"""
        return _specific_heat_const_v(self._coolprop_inputs)

    @property
    def viscosity(self):
        """Fluid viscosity [Pa.s]"""
        return _viscosity(self._coolprop_inputs)

    @property
    def thermal_conductivity(self):
        """Fluid thermal conductivity [W/m/K]"""
        return _thermal_conductivity(self._coolprop_inputs)

    @classmethod
    def of(
        cls,
        fluid_name: str,
        *,
        temperature: float | None = None,
        pressure: float | None = None,
        entropy: float | None = None,
        vapor_quality: float | None = None,
    ):
        """Calculates the fluid properties of a fluid given its temperature and pressure.

        Parameters
        ----------
        fluid_name :
            the name of the fluid to calculate properties for, e.g. 'Helium' or 'Water'.
        temperature :
            the current temperature [K] of the fluid to calculate the properties.
        pressure :
            the current pressure [Pa] of the fluid to calculate the properties.
        entropy :
            the current entropy [J/kg/K] of the fluid to calculate the properties.
        vapor_quality :
            the molar vapor quality [mol/mol] of the fluid to calculate the properties.
            `[0, 1]`, where `0` is a saturated liquid and `1` is a saturated vapor.
        """
        coolprop_inputs = []

        if temperature is not None:
            coolprop_inputs += ["T", temperature]

        if pressure is not None:
            coolprop_inputs += ["P", pressure]

        if entropy is not None:
            coolprop_inputs += ["S", entropy]

        if vapor_quality is not None:
            coolprop_inputs += ["Q", vapor_quality]

        coolprop_inputs.append(fluid_name.title())

        return cls(coolprop_inputs)


@cache
def _temperature(coolprop_inputs: tuple[str | float]):
    return PropsSI("T", *coolprop_inputs)


@cache
def _pressure(coolprop_inputs: tuple[str | float]):
    return PropsSI("P", *coolprop_inputs)


@cache
def _density(coolprop_inputs: tuple[str | float]):
    return PropsSI("D", *coolprop_inputs)


@cache
def _enthalpy(coolprop_inputs: tuple[str | float]):
    return PropsSI("H", *coolprop_inputs)


@cache
def _entropy(coolprop_inputs: tuple[str | float]):
    return PropsSI("S", *coolprop_inputs)


@cache
def _specific_heat_const_p(coolprop_inputs: tuple[str | float]):
    return PropsSI("C", *coolprop_inputs)


@cache
def _specific_heat_const_v(coolprop_inputs: tuple[str | float]):
    return PropsSI("CVMASS", *coolprop_inputs)


@cache
def _viscosity(coolprop_inputs: tuple[str | float]):
    return PropsSI("V", *coolprop_inputs)


@cache
def _thermal_conductivity(coolprop_inputs: tuple[str | float]):
    return PropsSI("CONDUCTIVITY", *coolprop_inputs)
