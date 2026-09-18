"""The solve environment, built from a configuration alone -- no `DataStructure`.

`state_of(configuration)` layers the configuration's own values over
`configurations.defaults.VALUES` and applies `DERIVATIONS`, `init.py`'s own rules;
`reference_of(configuration)` adds the problem. The `*_from_indat` route is the
converter's: the same, read off an `IN.DAT` (`input.indat.configuration_from_indat`).
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from functional_process.configurations import Configuration, Problem, defaults
from functional_process.configurations.defaults import Full
from functional_process.cottax.input.importer import ArrayInput, Imported, read_indat
from functional_process.vocabulary.iteration_variables import ITERATION_VARIABLES

DATACLASS_DEFAULTS: dict[tuple[str, str], Any] = {
    (area, name): value
    for area, fields in defaults.VALUES.items()
    for name, value in fields.items()
}
"""`configurations.defaults.VALUES`, keyed `(area, field)`."""


def _expand(value):
    """A `Full` as its array; anything else unchanged."""
    return (
        np.full(value.shape, value.value, dtype=float) if type(value) is Full else value
    )


@dataclass
class _Area:
    """One `data.<area>`. Answers the fields it holds; records the ones it does not."""

    name: str
    values: dict[str, Any]
    missing: list[tuple[str, str]]

    def __getattr__(self, field_name: str):
        """The field, or an `AttributeError` **and a row in `missing`**."""
        if field_name.startswith("__"):
            raise AttributeError(field_name)
        if field_name in self.values:
            return self.values[field_name]
        self.missing.append((self.name, field_name))
        raise AttributeError(
            f"`.{self.name}.{field_name}` has no native answer -- no IN.DAT line sets "
            f"it and it is not in `DATACLASS_DEFAULTS`"
        )


@dataclass
class NativeState:
    """What a solve reads instead of PROCESS's `DataStructure`."""

    areas: dict[str, _Area]
    values: dict[tuple[str, str], Any]
    missing: list[tuple[str, str]] = field(default_factory=list)
    sources: dict[tuple[str, str], str] = field(default_factory=dict)
    """`indat` or `defaults` per answered place -- §22.6's `source` column, minus the
    `process` row that no longer exists here.
    """

    def __getattr__(self, area_name: str):
        """The area, or an `AttributeError`."""
        if area_name.startswith("__"):
            raise AttributeError(area_name)
        if area_name in self.areas:
            return self.areas[area_name]
        raise AttributeError(f"no area `{area_name}`")

    def get(self, area: str, name: str, default=None):
        """One place's value, without going through the area objects."""
        return self.values.get((area, name), default)


def _array_from(imported_value: ArrayInput, default):
    """One `IN.DAT` array assignment applied to its dataclass default."""
    base = np.asarray(_expand(default), dtype=float)
    out = np.zeros_like(base) if imported_value.zero_filled else base.copy()
    flat = out.reshape(-1)
    for index, value in imported_value.elements:
        if index < flat.size:
            flat[index] = value
    return out


# ------------------------------------------------------------------- the derivations

_Places = dict[tuple[str, str], Any]
"""`(area, field) -> value`."""

#
# The third source, and the last one. `_audit/init_audit.md` counts **zero** genuine
# parse-time inputs among `init.py`'s 35 writes: every one of them is a sentinel
# resolved, a presence flag, or a *derivation* from something the file already states.
# So a place `init.py` writes is not a place a native state has to be told about -- it is
# a place it has to *work out*, and answering it with the dataclass default is answering
# a field PROCESS's own initialisation has already overwritten before any model runs.
#
# These run **after** the file's values, because that is where they run in PROCESS:
# `SingleRun.initialise` calls `initialise_imprad` and then `init_process`, and
# `init_process` parses the `IN.DAT` before applying any of the rules below.


def _initialise_imprad(values: _Places, out: _Places) -> None:
    """`process/main.py:430`'s four tables -- `init_audit.md` §5's fifth source."""
    from functional_process.models.physics.impurity_radiation import (  # noqa: PLC0415
        M_IMPURITY_AMU_ARRAY,
        impurity_tables,
    )

    for name, table in impurity_tables().items():
        out["impurity_radiation", name] = table.copy()
    out["impurity_radiation", "m_impurity_amu_array"] = np.asarray(
        M_IMPURITY_AMU_ARRAY, dtype=float
    )


def _alias_impurity_fractions(values: _Places, out: _Places) -> None:
    """`init.py:381-384` -- `f_nd_impurity_electron_array[i]` from
    `f_nd_impurity_electrons[i]`.
    """
    declared = values.get(("impurity_radiation", "f_nd_impurity_electrons"))
    if declared is None:
        return
    out["impurity_radiation", "f_nd_impurity_electron_array"] = np.asarray(
        declared, dtype=float
    ).copy()


def _single_or_double_null(values: _Places, out: _Places) -> None:
    """`init.py:606-617` -- `.physics.i_single_null` decides four fields, not one."""
    from functional_process.cottax.input.indat import _n_divertors  # noqa: PLC0415
    from functional_process.vocabulary.enums import DivertorNumberModels  # noqa: PLC0415

    i_single_null = values.get(("physics", "i_single_null"))
    if i_single_null is None:
        return
    out["divertor", "n_divertors"] = _n_divertors(int(i_single_null))
    if DivertorNumberModels(int(i_single_null)) is not DivertorNumberModels.DOUBLE_NULL:
        return
    for upper, lower in (
        ("dz_fw_plasma_gap", "dz_xpoint_divertor"),
        ("dz_shld_upper", "dz_shld_lower"),
        ("dz_vv_upper", "dz_vv_lower"),
    ):
        if ("build", lower) in values:
            out["build", upper] = values["build", lower]


def _deprecated_temperature_margin_alias(values: _Places, out: _Places) -> None:
    """`init.py:1171-1190` -- `tmargmin` is a deprecated alias for two fields."""
    tmargmin = values.get(("tfcoil", "tmargmin"))
    if tmargmin is None or float(tmargmin) <= 0.0001:
        return
    out["tfcoil", "temp_tf_superconductor_margin_min"] = float(tmargmin)
    out["tfcoil", "temp_cs_superconductor_margin_min"] = float(tmargmin)


DERIVATIONS = (
    _initialise_imprad,
    _alias_impurity_fractions,
    _single_or_double_null,
    _deprecated_temperature_margin_alias,
)
"""The rules a native state applies over the file's own values, in PROCESS's own order.
"""


def stated_values(input_file: str | Imported) -> tuple[dict[str, dict[str, Any]], list]:
    """`({area: {field: value}}, unshapeable)`: the file's **own** numbers, an array
    assignment applied to its default -- what a configuration states as `values`.
    """
    imported = input_file if isinstance(input_file, Imported) else read_indat(input_file)
    values: dict[str, dict[str, Any]] = {}
    unshapeable: list[tuple[str, str]] = []
    for (area, name), stated in imported.values.items():
        if area == "numerics":
            # The problem's, not the machine's: bounds, figure of merit, run mode,
            # tolerances. `Problem` carries what a solve reads of it; no node does.
            continue
        if isinstance(stated, ArrayInput):
            default = DATACLASS_DEFAULTS.get((area, name))
            if default is None:
                unshapeable.append((area, name))
                continue
            value = _array_from(stated, default)
        else:
            value = stated
        values.setdefault(area, {})[name] = value
    return values, unshapeable


def layered_values(stated: dict[str, dict[str, Any]]) -> tuple[dict, dict]:
    """`(values, sources)`: the defaults, then `stated`, then `DERIVATIONS`."""
    values: dict[tuple[str, str], Any] = {}
    sources: dict[tuple[str, str], str] = {}
    for place, default in DATACLASS_DEFAULTS.items():
        values[place] = _expand(default)
        sources[place] = "defaults"
    for area, fields in stated.items():
        for name, value in fields.items():
            values[area, name] = value
            sources[area, name] = "stated"
    derived: _Places = {}
    for derive in DERIVATIONS:
        derive(values, derived)
    values.update(derived)
    for place in derived:
        sources[place] = "derived"
    return values, sources


def native_values(input_file: str | Imported) -> tuple[dict, dict, list]:
    """`(values, sources, unshapeable)` for one file -- `stated_values` layered."""
    stated, unshapeable = stated_values(input_file)
    values, sources = layered_values(stated)
    return values, sources, unshapeable


def state_of(configuration: Configuration) -> NativeState:
    """The `DataStructure` stand-in for one configuration. **The interface.**"""
    values, sources = layered_values(configuration.values)
    return _state(values, sources, [])


def _state(values, sources, missing) -> NativeState:
    areas: dict[str, _Area] = {}
    for (area, name), value in values.items():
        areas.setdefault(area, _Area(area, {}, missing)).values[name] = value
    return NativeState(areas=areas, values=values, missing=missing, sources=sources)


def native_state(input_file: str | Imported) -> NativeState:
    """`state_of`, read off an `IN.DAT`."""
    values, sources, unshapeable = native_values(input_file)
    return _state(values, sources, list(unshapeable))


# ------------------------------------------------------------------- the problem side


def _pedestal_temperature_bound(ixc, state, low: float, high: float):
    """`init.py:444-459` -- iteration variable 4's lower bound, raised off the pedestal.
    """
    from functional_process.cottax.input.indat import (
        ST_INIT_I_PLASMA_PEDESTAL,
    )

    if 4 not in {int(i) for i in ixc}:
        return low, high
    if int(state.get("ife", "ife", 0)) == 1:
        return low, high
    if int(state.get("numerics", "i_process_run_mode", 1)) != 1:  # OPTIMISATION
        return low, high
    pedestal = int(state.get("physics", "i_plasma_pedestal", 1))
    if int(state.get("stellarator", "istell", 0)) != 0:
        pedestal = ST_INIT_I_PLASMA_PEDESTAL
    if pedestal != 1:
        return low, high
    raised = float(state.get("physics", "temp_plasma_pedestal_kev", 1.0)) * 1.001
    return (low, high) if low >= raised else (raised, max(high, raised))


def native_bounds(ixc, imported, state=None):
    """`((VarPath, lower, upper), ...)` -- `ReferenceRun.bounds`, from the file."""
    from functional_process.cottax.architectures.sand import (
        iteration_variable_path,  # noqa: PLC0415
    )

    lower = imported.get("numerics", "boundl")
    upper = imported.get("numerics", "boundu")
    out = []
    for i in ixc:
        variable = ITERATION_VARIABLES[int(i)]
        low, high = float(variable.lower_bound), float(variable.upper_bound)
        if isinstance(lower, ArrayInput):
            low = float(lower.as_dict().get(int(i) - 1, low))
        if isinstance(upper, ArrayInput):
            high = float(upper.as_dict().get(int(i) - 1, high))
        if int(i) == 4 and state is not None:
            low, high = _pedestal_temperature_bound(ixc, state, low, high)
        out.append((iteration_variable_path(int(i)), low, high))
    return tuple(out)


@dataclass
class NativeReference:
    """Everything a cold solve reads about the problem a file states -- `ixc`, `icc`, the
    bounds, the figure of merit, the cold state -- with nothing from PROCESS in it.
    """

    data: object
    cold: object
    ixc: list
    icc: list
    n_equality: int
    i_figure_merit: int
    bounds: tuple
    solver_iterations: int | None = None
    convergence_parameter: float | None = None
    solve_seconds: float = 0.0


def reference_of(configuration: Configuration) -> NativeReference:
    """Everything a cold solve needs for one configuration."""
    from functional_process.cottax.architectures.sand import (
        iteration_variable_path,  # noqa: PLC0415
    )

    problem = configuration.problem
    state = state_of(configuration)
    return NativeReference(
        data=state,
        cold=state,
        ixc=list(problem.ixc),
        icc=list(problem.icc),
        n_equality=int(problem.n_equality),
        i_figure_merit=int(problem.i_figure_merit),
        bounds=tuple(
            (iteration_variable_path(i), *problem.bounds[i]) for i in problem.ixc
        ),
    )


def problem_from_indat(input_file: str | Imported, switches: dict[str, int]) -> Problem:
    """The `Problem` one file states: `ixc` sorted as PROCESS loads it, the
    equalities counted, the bounds resolved (`native_bounds`), `switches` carried.
    """
    from functional_process.cottax.input.indat import (  # noqa: PLC0415
        problem_from_indat as stated_problem,
    )

    imported = input_file if isinstance(input_file, Imported) else read_indat(input_file)
    stated = stated_problem(imported)
    ixc = tuple(sorted(int(i) for i in stated.ixc))
    n_equality = stated.n_equality_constraints
    if n_equality is None:
        # `init.py`'s `-1` sentinel: the equalities are what is left over.
        n_equality = len(stated.icc) - (stated.n_inequality_constraints or 0)
    state = native_state(imported)
    bounds = {
        int(i): (low, high)
        for (_path, low, high), i in zip(
            native_bounds(ixc, imported, state), ixc, strict=True
        )
    }
    return Problem(
        ixc=ixc,
        icc=tuple(int(i) for i in stated.icc),
        n_equality=int(n_equality),
        i_figure_merit=int(stated.i_figure_merit or 7),
        bounds=bounds,
        switches=dict(switches),
        root_find=stated.is_evaluation,
    )


def native_reference(input_file: str) -> NativeReference:
    """`reference_of`, read off an `IN.DAT` -- through the converter."""
    from functional_process.cottax.input.indat import (  # noqa: PLC0415
        configuration_from_indat,
    )

    return reference_of(configuration_from_indat(input_file))


CONFIGURATIONS = (
    "tests/regression/input_files/stellarator_helias.IN.DAT",
    "tests/regression/input_files/helias_5b.IN.DAT",
    "tests/regression/input_files/large_tokamak_nof.IN.DAT",
    "tests/regression/input_files/large_tokamak_eval.IN.DAT",
    "tests/regression/input_files/low_aspect_ratio_DEMO.IN.DAT",
    "tests/regression/input_files/spherical_tokamak_eval.IN.DAT",
    "tests/regression/input_files/st_regression.IN.DAT",
)
"""Every `tests/regression/input_files/*.IN.DAT` except `IFE.IN.DAT`."""


def stem(input_file: str) -> str:
    """`.../helias_5b.IN.DAT` -> `helias_5b`, the name a configuration is known by."""
    return pathlib.PurePath(input_file).name.removesuffix(".IN.DAT")
