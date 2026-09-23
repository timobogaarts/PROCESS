"""A configuration, stated: the machine, its values, and the problem it poses.

PROCESS reads all three out of one `IN.DAT`: integer switches select the models, the
numbers seed the boundary, and `ixc`/`icc`/`minmax` state the optimisation. Here each
is a declared tree in a Python module beside this file:

- **`machine`**: a `StellaratorProcess`/`TokamakProcess` with every slot's occupant
  written out -- the models, chosen directly, no integer switch anywhere.
- **`values`**: `{area: {field: value}}`, the configuration's own numbers. Everything
  it does not state is `defaults.VALUES`, PROCESS's own dataclass defaults; the two
  layered, plus `init.py`'s derivation rules (`input.native.DERIVATIONS`), are the
  boundary a solve starts from (`input.native.state_of`).
- **`problem`**: the design variables, the constraints, the figure of merit, the
  bounds, and the static switch values the constraint nodes are bound with.

`input.indat.configuration_from_indat` converts an `IN.DAT` into one, and `spell`
writes one back out as the module it is read from -- `tests/test_configurations.py`
checks the two agree for every regression file, so a checked-in tree cannot drift from
the file it was made from without saying so.
"""

from __future__ import annotations

import dataclasses
import enum
import importlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from cottax.interfaces.pytree_namespace_module import ModelNamespace

NAMES = (
    "stellarator_helias",
    "helias_5b",
    "large_tokamak_nof",
    "large_tokamak_eval",
    "low_aspect_ratio_DEMO",
    "spherical_tokamak_eval",
    "st_regression",
)
"""Every configuration checked in here: `tests/regression/input_files/*.IN.DAT` but
`IFE`, converted."""


@dataclass(frozen=True)
class Problem:
    """What a configuration asks to be solved, in PROCESS's own terms."""

    ixc: tuple[int, ...]
    """The design variables, as iteration-variable ids
    (`vocabulary.iteration_variables.ITERATION_VARIABLES`), in PROCESS's order."""
    icc: tuple[int, ...]
    """The active constraints, as constraint ids -- the first `n_equality` are
    equalities (`models.constraints`)."""
    n_equality: int
    i_figure_merit: int
    """The figure of merit (`models.objectives`), negative to maximise."""
    bounds: dict[int, tuple[float, float]]
    """`ixc id -> (lower, upper)`, the file's own where it states one."""
    switches: dict[str, int] = field(default_factory=dict)
    """The static switch arguments the constraint and objective nodes are bound with
    (`cottax.models.constraints.SWITCH_PARAMETER_NAMES`)."""
    root_find: bool = False
    """A root find over the equalities (`i_process_run_mode = -2`) rather than an
    optimisation: no objective, the inequalities evaluated once at the answer."""


@dataclass(frozen=True)
class Configuration:
    """One machine, its values, and its problem."""

    name: str
    machine: ModelNamespace
    values: dict[str, dict[str, Any]]
    problem: Problem


def load(name: str) -> Configuration:
    """The configuration module `name` declares."""
    return importlib.import_module(
        f"functional_process.configurations.{name}"
    ).CONFIGURATION


def all_configurations() -> tuple[Configuration, ...]:
    """Every checked-in configuration, in `NAMES` order."""
    return tuple(load(name) for name in NAMES)


# ------------------------------------------------------------------- spelling

_INDENT = "    "


def _default_of(f: dataclasses.Field):
    """The field's default, or `MISSING`."""
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:
        return f.default_factory()
    return dataclasses.MISSING


def _same(a, b) -> bool:
    """Whether `a` is `b`'s value -- arrays elementwise, anything else by `==`."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    try:
        return bool(a == b)
    except Exception:  # noqa: BLE001 -- an object that cannot say is not the default
        return False


class Speller:
    """Spells a value as Python source, collecting the names it needs imported."""

    def __init__(self, context: dict | None = None):
        self.imports: dict[str, set[str]] = {}
        self.context = context or {}

    def name(self, obj) -> str:
        """The class's name, its import recorded."""
        cls = obj if isinstance(obj, type) else type(obj)
        self.imports.setdefault(cls.__module__, set()).add(cls.__name__)
        return cls.__name__

    def spell(self, value, depth=0) -> str:  # noqa: PLR0911
        """`value` as source at `depth` levels of indentation.

        Raises
        ------
        TypeError
            If `value` is of a kind this has no spelling for.
        """
        pad, inner = _INDENT * depth, _INDENT * (depth + 1)
        if isinstance(value, enum.Enum):
            return f"{self.name(value)}.{value.name}"
        if isinstance(value, str):
            return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
        if isinstance(value, (bool, int, float)) or value is None:
            return repr(value)
        if isinstance(value, np.ndarray):
            self.imports.setdefault("numpy", set())
            return f"np.array({value.tolist()!r})"
        if isinstance(value, (tuple, list)):
            body = ", ".join(self.spell(v, depth + 1) for v in value)
            if isinstance(value, list):
                return f"[{body}]"
            return f"({body},)" if len(value) == 1 else f"({body})"
        if isinstance(value, dict):
            lines = [
                f"{inner}{self.spell(k)}: {self.spell(v, depth + 1)},"
                for k, v in value.items()
            ]
            return "{\n" + "\n".join(lines) + f"\n{pad}}}"
        special = _SPECIAL.get(type(value).__name__)
        if special is not None:
            return special(self, value, depth)
        if dataclasses.is_dataclass(value):
            return self._dataclass(value, depth)
        raise TypeError(f"no spelling for {type(value).__name__}: {value!r}")

    def _dataclass(self, value, depth) -> str:
        pad, inner = _INDENT * depth, _INDENT * (depth + 1)
        name = self.name(value)
        parts = []
        for f in dataclasses.fields(value):
            if not f.init:
                continue
            got = getattr(value, f.name)
            if got is None:
                # An empty slot, stated as such where the namespace has no default
                # for it: the tree says which slots this device leaves empty.
                if (
                    isinstance(value, ModelNamespace)
                    and _default_of(f) is dataclasses.MISSING
                ):
                    parts.append(f"{inner}{f.name}=None,")
                continue
            if not isinstance(value, ModelNamespace) and _same(got, _default_of(f)):
                continue
            parts.append(f"{inner}{f.name}={self.spell(got, depth + 1)},")
        if not parts:
            return f"{name}()"
        return f"{name}(\n" + "\n".join(parts) + f"\n{pad})"

    def import_lines(self) -> str:
        """The import block every name spelled so far needs."""
        lines = []
        if "numpy" in self.imports:
            lines.append("import numpy as np\n")
        for module in sorted(m for m in self.imports if m != "numpy"):
            names = sorted(self.imports[module])
            lines.append(f"from {module} import {', '.join(names)}")
        return "\n".join(lines)


def _spell_quench(speller: Speller, value, depth) -> str:
    """`TfCoilQuenchHeatCurrentDensity.at(...)`: the helium table from its two inputs."""
    name = speller.name(value)
    return (
        f"{name}.at(tftmp={value.tftmp!r}, "
        f"temp_tf_conductor_quench_max={value.temp_tf_conductor_quench_max!r})"
    )


def _spell_machine_config(speller: Speller, value, depth) -> str:
    """`StellaratorMachineConfig(machine_config=machine_config_for_istell(...))`.

    Raises
    ------
    ValueError
        If the table is neither a preset nor the `stella_conf` file in `context`.
    """
    from functional_process.models.stellarator.preset_config import (  # noqa: PLC0415
        machine_config_for_istell,
    )

    name = speller.name(value)
    speller.imports.setdefault(machine_config_for_istell.__module__, set()).add(
        "machine_config_for_istell"
    )
    for istell in range(1, 6):
        if machine_config_for_istell(istell) == value.machine_config:
            return f"{name}(machine_config=machine_config_for_istell({istell}))"
    config_file = speller.context.get("stella_conf")
    if config_file is None or (
        machine_config_for_istell(6, config_file=config_file) != value.machine_config
    ):
        raise ValueError(
            "a stellarator machine config that is no preset and not the `stella_conf` "
            "file the speller was told about -- pass `context={'stella_conf': path}`"
        )
    return (
        f"{name}(\n{_INDENT * (depth + 1)}machine_config=machine_config_for_istell("
        f"6, config_file={speller.spell(config_file)})\n{_INDENT * depth})"
    )


_SPECIAL = {
    "TfCoilQuenchHeatCurrentDensity": _spell_quench,
    "StellaratorMachineConfig": _spell_machine_config,
}
"""Occupants whose state is data made from a few inputs: spelled as the call."""


def spell(configuration: Configuration, source: str = "", context=None) -> str:
    """`configuration` as the module `load` reads it back from."""
    speller = Speller(context)
    machine = speller.spell(configuration.machine)
    values = speller.spell(configuration.values)
    problem = speller.spell(configuration.problem)
    speller.name(Configuration)
    origin = (
        f"Converted from `{source}` by\n`input.indat.configuration_from_indat`"
        if source
        else "Generated"
    )
    header = (
        f'"""`{configuration.name}`, stated.\n\n{origin}; regenerate rather than '
        f"hand-edit --\n`tests/test_configurations.py` checks the two agree.\n"
        f'"""\n'
    )
    return (
        header
        + "\n"
        + speller.import_lines()
        + "\n\n"
        + f"machine = {machine}\n\n"
        + f"values = {values}\n\n"
        + f"problem = {problem}\n\n"
        + "CONFIGURATION = Configuration(\n"
        + f'    name="{configuration.name}",\n'
        + "    machine=machine,\n    values=values,\n    problem=problem,\n"
        + ")\n"
    )
