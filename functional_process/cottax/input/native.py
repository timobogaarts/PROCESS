"""The solve environment, built from a configuration alone -- no `DataStructure`.

`state_of(configuration)` layers the configuration's own values over
`configurations.defaults.VALUES` and applies `DERIVATIONS`, `init.py`'s own rules. The
`*_from_indat` route is the converter's: the same, read off an `IN.DAT`
(`input.indat.configuration_from_indat`), and `problem_from_indat` the problem it
states, where PROCESS's switches choose the constraint variants.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cottax.interfaces import to_graph
from cottax.pytree.path import VarPath
from jax.tree_util import GetAttrKey, SequenceKey

from functional_process.configurations import (
    Configuration,
    Optimisation,
    RootFinding,
    defaults,
)
from functional_process.configurations.defaults import Full
from functional_process.cottax.input.importer import ArrayInput, Imported, read_indat
from functional_process.vocabulary import (
    AvailabilityModel,
    BetaComponentLimits,
    DensityLimitModel,
    FiguresOfMerit,
    TFConductorModel,
    TFCSRadialConfiguration,
)
from functional_process.vocabulary.iteration_variables import ITERATION_VARIABLES

DATACLASS_DEFAULTS: dict[tuple[str, str], Any] = {
    (area, name): value
    for area, fields in defaults.VALUES.items()
    for name, value in fields.items()
}
"""`configurations.defaults.VALUES`, keyed `(area, field)`."""


PORT_CONSTANTS: dict[tuple[str, str], Any] = {
    ("impurity_radiation", "imp_indices"): tuple(range(14)),
    ("neoclassics", "rho"): 0.6,
}
"""Places the port reads that PROCESS does not hold as fields: which of the 14 impurity
species the radiation totals sum over (all of them, in every configuration), and the
normalised radius the neoclassical profiles are evaluated at (PROCESS's own literal at
its one call site)."""


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


def _pf_coil_counts(values: _Places, out: _Places) -> None:
    """`.pf_coil.n_cs_pf_coils`: the coil count of the PF topology `.build.iohcl` selects,
    which the PF magnet cost reads."""
    from functional_process.cottax.input.indat import _pf_coil_topology  # noqa: PLC0415

    out["pf_coil", "n_cs_pf_coils"] = _pf_coil_topology(
        values.get(("build", "iohcl"), 1)
    ).n_cs_pf_coils


def _quench_helium_tables(values: _Places, out: _Places) -> None:
    """The helium density and specific heat at the quench integral's 75 nodes, from
    `.tfcoil.tftmp` and `.tfcoil.temp_tf_conductor_quench_max`: CoolProp, which runs
    before the run and never inside it."""
    from functional_process.models.tfcoil.quench import (  # noqa: PLC0415
        helium_properties_at_quench_nodes,
    )

    den, cp = helium_properties_at_quench_nodes(
        temp_he_peak=float(values["tfcoil", "tftmp"]),
        temp_quench_max=float(values["tfcoil", "temp_tf_conductor_quench_max"]),
    )
    out["tfcoil", "den_helium_at_nodes"] = den
    out["tfcoil", "cp_helium_at_nodes"] = cp


def _entry(values: _Places, table, index) -> float | None:
    """`table[index - 1]`, both places of `values`; `None` if either is not valued."""
    if table not in values or index not in values:
        return None
    return float(np.asarray(_expand(values[table]), dtype=float)[int(values[index]) - 1])


def _level_of_safety_assurance(values: _Places, out: _Places) -> None:
    """The cost multipliers at `.costs.lsa`: `.costs.cmlsa_<name>` for each of
    `models.costs.costs.SAFETY_ASSURANCE_TABLES`, and `.costs.cfind_lsa`,
    `.costs.ucoam_lsa`, `.costs.ucwst_lsa`. `lsa` only indexes tables, so the lookup
    happens here, once, and no node reads it."""
    from functional_process.models.costs.costs import (  # noqa: PLC0415
        SAFETY_ASSURANCE_TABLES,
    )

    lsa = int(values["costs", "lsa"])
    for name, table in SAFETY_ASSURANCE_TABLES.items():
        out["costs", f"cmlsa_{name}"] = float(table[lsa - 1])
    for name in ("cfind", "ucoam", "ucwst"):
        out["costs", f"{name}_lsa"] = _entry(values, ("costs", name), ("costs", "lsa"))


TABLE_ENTRIES: tuple[tuple[tuple[str, str], tuple[str, str], tuple[str, str]], ...] = (
    (
        ("costs", "uchts"),
        ("fwbs", "i_blkt_coolant_type"),
        ("costs", "uchts_blkt_coolant"),
    ),
    (
        ("costs", "ucturb"),
        ("fwbs", "i_blkt_coolant_type"),
        ("costs", "ucturb_blkt_coolant"),
    ),
    (("costs", "ucsc"), ("pf_coil", "i_pf_superconductor"), ("costs", "ucsc_pf")),
    (("costs", "ucsc"), ("pf_coil", "i_cs_superconductor"), ("costs", "ucsc_cs")),
    (("costs", "ucsc"), ("tfcoil", "i_tf_sc_mat"), ("costs", "ucsc_tf")),
    (
        ("tfcoil", "dcond"),
        ("pf_coil", "i_pf_superconductor"),
        ("pf_coil", "den_pf_conductor"),
    ),
    (
        ("tfcoil", "dcond"),
        ("pf_coil", "i_cs_superconductor"),
        ("pf_coil", "den_cs_conductor"),
    ),
    (("tfcoil", "dcond"), ("tfcoil", "i_tf_sc_mat"), ("tfcoil", "den_tf_sc_material")),
    (
        ("costs", "sc_mat_cost_0"),
        ("pf_coil", "i_pf_superconductor"),
        ("costs", "sc_mat_cost_0_pf"),
    ),
    (
        ("costs", "sc_mat_cost_0"),
        ("pf_coil", "i_cs_superconductor"),
        ("costs", "sc_mat_cost_0_cs"),
    ),
    (
        ("costs", "sc_mat_cost_0"),
        ("tfcoil", "i_tf_sc_mat"),
        ("costs", "sc_mat_cost_0_tf"),
    ),
    (
        ("tfcoil", "j_crit_str_0"),
        ("pf_coil", "i_pf_superconductor"),
        ("pf_coil", "j_crit_str_0_pf"),
    ),
    (
        ("tfcoil", "j_crit_str_0"),
        ("pf_coil", "i_cs_superconductor"),
        ("pf_coil", "j_crit_str_0_cs"),
    ),
    (
        ("tfcoil", "j_crit_str_0"),
        ("tfcoil", "i_tf_sc_mat"),
        ("tfcoil", "j_crit_str_0_tf"),
    ),
)
"""`(table, index, entry)`: a table the models read only at an index a configuration
states, and the place its entry is derived at. The blanket coolant indexes the turbine
and heat transport unit costs; the superconductor materials index the unit costs,
densities and critical current densities."""


def _table_entries(values: _Places, out: _Places) -> None:
    """Every `TABLE_ENTRIES` entry whose table and index are valued. The `PER_KAM`
    tables `.costs.sc_mat_cost_0` and `.tfcoil.j_crit_str_0` are valued by no
    configuration (none takes that arm), so their entries are derived only where they
    are."""
    for table, index, entry in TABLE_ENTRIES:
        value = _entry(values, table, index)
        if value is not None:
            out[entry] = value


DERIVATIONS = (
    _initialise_imprad,
    _alias_impurity_fractions,
    _single_or_double_null,
    _deprecated_temperature_margin_alias,
    _pf_coil_counts,
    _quench_helium_tables,
    _level_of_safety_assurance,
    _table_entries,
)
"""The rules a native state applies over the file's own values, in PROCESS's own order;
then the table entries at the indices the configuration states, which PROCESS looks up
inside its models."""


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
            # tolerances. The problem carries what a solve reads of it; no node does.
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
    for place, default in (DATACLASS_DEFAULTS | PORT_CONSTANTS).items():
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
    """`init.py:444-459` -- iteration variable 4's lower bound, raised off the pedestal."""
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


def design_place(ixc_id: int) -> VarPath:
    """Where PROCESS iteration variable `ixc_id` lives."""
    variable = ITERATION_VARIABLES[ixc_id]
    keys = (
        GetAttrKey(variable.module),
        GetAttrKey(variable.target_name or variable.name),
    )
    if variable.array_index is not None:
        keys = (*keys, SequenceKey(variable.array_index))
    return VarPath(keys)


def native_bounds(ixc, imported, state=None) -> dict:
    """`{place: (lower, upper)}` per iteration variable, the file's own where it states
    one."""
    lower = imported.get("numerics", "boundl")
    upper = imported.get("numerics", "boundu")
    out = {}
    for i in ixc:
        variable = ITERATION_VARIABLES[int(i)]
        low, high = float(variable.lower_bound), float(variable.upper_bound)
        if isinstance(lower, ArrayInput):
            low = float(lower.as_dict().get(int(i) - 1, low))
        if isinstance(upper, ArrayInput):
            high = float(upper.as_dict().get(int(i) - 1, high))
        if int(i) == 4 and state is not None:
            low, high = _pedestal_temperature_bound(ixc, state, low, high)
        out[design_place(int(i))] = (low, high)
    return out


class NoVariant(ValueError):
    """A switch value no declared variant answers."""


def constraint_declaration(cid: int, switches):
    """The declaration of PROCESS constraint `cid` that `switches` choose.

    Raises
    ------
    NoVariant
        If `cid` is not declared, or no declared variant answers `switches`.
    """
    from functional_process.cottax.models import constraints  # noqa: PLC0415

    heated = "Ignited" if switches["i_plasma_ignited"] != 0 else "Heated"
    radiation = {0: "TotalRadiation", 1: "CoreRadiation"}
    beta = {
        BetaComponentLimits.TOTAL: "TotalBeta",
        BetaComponentLimits.THERMAL: "ThermalBeta",
        BetaComponentLimits.THERMAL_AND_BEAM: "ThermalAndBeamBeta",
        BetaComponentLimits.TOROIDAL: "ToroidalBeta",
    }
    lifetime = {0: "Input", 1: "Divertor", 2: "Blanket", 3: "Plant"}
    try:
        suffix = {
            2: lambda: heated + radiation[switches["i_rad_loss"]],
            3: lambda: heated,
            4: lambda: heated + radiation[switches["i_rad_loss"]],
            5: lambda: (
                "Greenwald"
                if switches["i_density_limit"] == DensityLimitModel.GREENWALD
                else "VolumeAverage"
            ),
            7: lambda: "" if heated == "Heated" else None,
            17: lambda: "Stellarator" if switches["istell"] != 0 else "Tokamak",
            24: lambda: (
                "TotalBeta"
                if switches["istell"] != 0
                else beta[BetaComponentLimits(switches["i_beta_component"])]
            ),
            28: lambda: "" if heated == "Heated" else None,
            43: lambda: (
                "Copper"
                if switches["i_tf_sup"] == TFConductorModel.WATER_COOLED_COPPER
                else "Other"
            ),
            44: lambda: (
                "Copper"
                if switches["i_tf_sup"] == TFConductorModel.WATER_COOLED_COPPER
                else "Other"
            ),
            45: lambda: "" if switches["itart"] != 0 else None,
            46: lambda: "" if switches["itart"] != 0 else None,
            68: lambda: "FixedQ95" if switches["i_q95_fixed"] == 1 else "ComputedQ95",
            72: lambda: (
                "Bucked"
                if switches["i_tf_bucking"] >= 2
                and switches["i_tf_inside_cs"] == TFCSRadialConfiguration.TF_OUTSIDE_CS
                else "Free"
            ),
            85: lambda: lifetime[switches["i_cp_lifetime"]] + "Lifetime",
            90: lambda: (
                "BlanketCycles"
                if switches["ibkt_life"] == 1 and switches["bkt_life_csf"] == 1
                else "MinimumCycles"
            ),
            91: lambda: heated,
        }.get(cid, lambda: "")()
    except KeyError as e:
        raise NoVariant(f"constraint {cid}: no declared variant for {e}") from None
    name = None if suffix is None else f"Constraint{cid}{suffix}"
    if name is None or not hasattr(constraints, name):
        raise NoVariant(f"constraint {cid} has no declared variant for these switches")
    return getattr(constraints, name)


def merit_declaration(i_figure_merit: int, switches):
    """The declaration of the figure of merit `abs(i_figure_merit)` that `switches`
    choose.

    Raises
    ------
    NoVariant
        If no declared variant answers `switches`.
    """
    from functional_process.cottax.models import constraints  # noqa: PLC0415

    merit = FiguresOfMerit(abs(int(i_figure_merit)))
    if merit is FiguresOfMerit.CAPITAL_COST:
        name = "CapitalCostDirect" if switches["ireactor"] == 0 else "CapitalCostReactor"
    elif merit is FiguresOfMerit.PLANT_AVAILABILITY_FACTOR:
        if switches["i_plant_availability"] == AvailabilityModel.USER_INPUT:
            raise NoVariant("the plant availability factor is an input here")
        name = "PlantAvailabilityFactor"
    else:
        name = "".join(w.capitalize() for w in merit.name.split("_"))
    return getattr(constraints, name)


def problem_from_indat(input_file: str | Imported, switches: dict[str, int], machine):
    """The problem one file states over `machine`: an `Optimisation`, or a
    `RootFinding` in PROCESS's evaluation mode. `ixc` in PROCESS's load order, the
    first `neqns` of `icc` the equalities, the bounds resolved (`native_bounds`), and
    every constraint and the merit the variant `switches` choose.

    An inequality that reads nothing the design reaches is constant over it, so no
    optimiser moves it: it is left out. An equality like it is refused.

    Raises
    ------
    ValueError
        If an equality reads nothing the design reaches, or a root find is not square.
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
    icc = tuple(int(i) for i in stated.icc)
    declared = [constraint_declaration(c, switches) for c in icc]
    equalities = tuple(declared[:n_equality])
    inequalities = tuple(declared[n_equality:])
    design = native_bounds(ixc, imported, native_state(imported))
    if stated.is_evaluation:
        if len(equalities) != len(design):
            raise ValueError(
                f"a root find needs one equality per iteration variable: "
                f"{len(equalities)} against {len(design)}"
            )
        return RootFinding(tuple(design), equalities, inequalities)
    graph = to_graph(machine).graph
    reached = {p for n in graph.reach(tuple(design)) for p in graph[n].owns}
    reached |= set(design)

    def constant(c) -> bool:
        return not reached & set(c.reads.values())

    if refused := [c for c in equalities if constant(c)]:
        raise ValueError(
            f"equality constraint(s) {[c.fn.__name__ for c in refused]} read nothing "
            f"the design reaches: leaving one out changes what feasible means"
        )
    merit = merit_declaration(stated.i_figure_merit or 7, switches)
    sense = "maximise" if (stated.i_figure_merit or 7) < 0 else "minimise"
    return Optimisation(
        design=design,
        equalities=equalities,
        inequalities=tuple(c for c in inequalities if not constant(c)),
        **{sense: merit},
    )


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
