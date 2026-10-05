"""A configuration, stated: the machine, its values, and the problem it poses.

PROCESS reads all three out of one `IN.DAT`: integer switches select the models and
the constraint variants, the numbers seed the boundary, and `ixc`/`icc`/`minmax` state
the optimisation. Here each
is a declared tree in a Python module beside this file:

- **`machine`**: a `StellaratorProcess`/`TokamakProcess` with every slot's occupant
  written out -- the models, chosen directly, no integer switch anywhere.
- **`values`**: `{area: {field: value}}`, the configuration's own numbers. Everything
  it does not state is `defaults.VALUES`, PROCESS's own dataclass defaults; the two
  layered, plus `init.py`'s derivation rules (`input.native.DERIVATIONS`), are the
  boundary a solve starts from (`input.native.state_of`).
- **`problem`**: an `Optimisation` or a `RootFinding` -- the design variables with
  their bounds, the constraints held, and the figure of merit, each chosen directly by
  name as the machine's models are; no switch.

`input.indat.configuration_from_indat` converts an `IN.DAT` into one, and `spell`
writes one back out as the module it is read from -- `tests/test_configurations.py`
checks the two agree for every regression file, so a checked-in tree cannot drift from
the file it was made from without saying so.
"""

from __future__ import annotations

import dataclasses
import enum
import importlib
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import numpy as np
from cottax.consumers.declaration.interface_types import Wrapper
from cottax.pytree import VarPath
from jax.tree_util import GetAttrKey, SequenceKey

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
class Optimisation:
    """An optimisation: `minimise` or `maximise` one figure of merit over `design`
    within its bounds, every equality held `= 0` and every inequality `<= 0`. The
    constraints and the merit are the declarations in `cottax.models.constraints`."""

    design: dict[Any, tuple[float, float]]
    """Each design variable's place, and its `(lower, upper)` bounds."""
    equalities: tuple[Wrapper, ...]
    inequalities: tuple[Wrapper, ...]
    minimise: Wrapper | None = None
    maximise: Wrapper | None = None


@dataclass(frozen=True)
class RootFinding:
    """A root find: every equality held `= 0` over `design`, one per variable; the
    inequalities are computed, and held to nothing, as PROCESS evaluates them once at
    the answer."""

    design: tuple[Any, ...]
    equalities: tuple[Wrapper, ...]
    evaluated: tuple[Wrapper, ...]


@dataclass(frozen=True)
class Configuration:
    """One machine, its values, and its problem."""

    name: str
    machine: eqx.Module
    values: dict[str, dict[str, Any]]
    problem: Optimisation | RootFinding


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

    def place(self, path: VarPath) -> str:
        """`path` as `place(V...)`, its imports recorded."""
        self.imports.setdefault("cottax.relational_pytree", set()).update({"V", "place"})
        keys = []
        for key in path.segments:
            if isinstance(key, GetAttrKey):
                keys.append(f".{key.name}")
            elif isinstance(key, SequenceKey):
                keys.append(f"[{key.idx}]")
            else:
                raise TypeError(f"no spelling for the key {key!r} of {path!r}")
        return f"place(V{''.join(keys)})"

    def variant(self, wrapper: Wrapper) -> str:
        """The name a declared node is bound at in its module, its import recorded."""
        import sys  # noqa: PLC0415

        for module_name, module in list(sys.modules.items()):
            if module_name.startswith("functional_process.cottax.models"):
                for name, bound in vars(module).items():
                    if bound is wrapper and not name.startswith("_"):
                        self.imports.setdefault(module_name, set()).add(name)
                        return name
        raise TypeError(f"{wrapper!r} is bound at no name in the port's models")

    def spell(self, value, depth=0) -> str:  # noqa: PLR0911
        """`value` as source at `depth` levels of indentation.

        Raises
        ------
        TypeError
            If `value` is of a kind this has no spelling for.
        """
        pad, inner = _INDENT * depth, _INDENT * (depth + 1)
        if isinstance(value, _Source):
            self.imports.setdefault(value.module, set()).update(value.names)
            return value.text
        if isinstance(value, Wrapper):
            return self.variant(value)
        if isinstance(value, VarPath):
            return self.place(value)
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
                    isinstance(value, eqx.Module)
                    and _default_of(f) is dataclasses.MISSING
                ):
                    parts.append(f"{inner}{f.name}=None,")
                continue
            if not isinstance(value, eqx.Module) and _same(got, _default_of(f)):
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


@dataclass(frozen=True)
class _Source:
    """A value spelled as the call that makes it, with what the call imports."""

    text: str
    module: str
    names: tuple[str, ...]


def _machine_config_source(machine_config, context) -> _Source:
    """`.stellarator_config.machine_config` as the `machine_config_for_istell(...)` call
    that makes it: a preset by its number, or the `stella_conf` file in `context`.

    Raises
    ------
    ValueError
        If the table is neither a preset nor the `stella_conf` file in `context`.
    """
    from functional_process.models.stellarator.preset_config import (  # noqa: PLC0415
        machine_config_for_istell,
    )

    call = _Source(
        "", machine_config_for_istell.__module__, ("machine_config_for_istell",)
    )
    for istell in range(1, 6):
        if machine_config_for_istell(istell) == machine_config:
            return dataclasses.replace(call, text=f"machine_config_for_istell({istell})")
    config_file = context.get("stella_conf")
    if config_file is None or (
        machine_config_for_istell(6, config_file=config_file) != machine_config
    ):
        raise ValueError(
            "a stellarator machine config that is no preset and not the `stella_conf` "
            "file the speller was told about -- pass `context={'stella_conf': path}`"
        )
    return dataclasses.replace(
        call, text=f'machine_config_for_istell(6, config_file="{config_file}")'
    )


def spell(configuration: Configuration, source: str = "", context=None) -> str:
    """`configuration` as the module `load` reads it back from."""
    speller = Speller(context)
    machine = speller.spell(configuration.machine)
    stated = configuration.values
    if "machine_config" in stated.get("stellarator_config", {}):
        stated = stated | {
            "stellarator_config": stated["stellarator_config"]
            | {
                "machine_config": _machine_config_source(
                    stated["stellarator_config"]["machine_config"], speller.context
                )
            }
        }
    values = speller.spell(stated)
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
