"""The optimisation problem each of the seven reference machines poses, as tables.

One table per configuration (`functional_process.configurations.NAMES`): the design
variables (`ixc`, with the file's bounds), the figure of merit, the equality
constraints and the inequality constraints (`icc`, split at `n_equality`), each by its
PROCESS number and its PROCESS label (`NumericsData.lablcc`,
`FiguresOfMerit.description`). A last table sets the machines side by side on the
integer model switches that tell them apart, each value with the meaning PROCESS's
own docstring gives it.

The problem and the values are read off each configuration module's source, not
imported from it: the tables need no machine tree, so they need no cottax, and do not
move when cottax does.

    $PY paper_tests/reference_problems.py          # -> out/reference_problems.{md,tex}
    $PY paper_tests/reference_problems.py --paper  # and the paper's copy, which
                                                   # main.tex \\inputs in its appendix
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import numpy as np

from functional_process.configurations import NAMES, Problem
from functional_process.configurations.defaults import VALUES
from functional_process.vocabulary.iteration_variables import ITERATION_VARIABLES
from process.data_structure.numerics import FiguresOfMerit, NumericsData
from process.data_structure.impurity_radiation_variables import ImpurityRadiationData
from process.data_structure.physics_variables import ConfinementTimeModel
from process.models.physics.current_drive import CurrentDriveModel
from process.models.tfcoil.superconducting import SuperconductingTFTurnType

ROOT = Path(__file__).resolve().parent.parent
CONFIGURATIONS = ROOT / "functional_process" / "configurations"
DATA_STRUCTURE = ROOT / "process" / "data_structure"
OUT = Path(__file__).resolve().parent / "out"
PAPER = Path.home() / "graph_paper" / "listings" / "process_cases"
"""Where `--paper` puts the `.tex`: the paper `\\input`s it from its appendix."""

CONSTRAINT_LABELS = [" ".join(s.split()) for s in NumericsData().lablcc]
for _number, _label in {
    81: "Central electron density above pedestal",
    82: "Stellarator toroidal build consistency",
    83: "Stellarator radial build consistency",
}.items():
    CONSTRAINT_LABELS[_number - 1] = _label
"""PROCESS's `lablcc`, but where it spells a constraint in variable names, the words
of that constraint's own docstring (`process/core/solver/constraints.py`)."""

NOT_SWITCHES = {"maxcal", "n_pf_coil_groups", "n_tf_wp_layers", "n_tf_wp_pancakes"}
"""Integers in `values` that are counts or limits, not model choices."""

SWITCH_ENUMS = {
    "i_confinement_time": ConfinementTimeModel,
    "i_hcd_primary": CurrentDriveModel,
    "i_tf_turn_type": SuperconductingTFTurnType,
}
"""Switches whose values PROCESS names in an enum rather than in the docstring."""


def stated(name: str) -> tuple[dict, Problem]:
    """`values` and `problem` of configuration `name`, evaluated from its source."""
    source = (CONFIGURATIONS / f"{name}.py").read_text()
    got = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and node.targets[0].id in {"values", "problem"}:
            segment = ast.get_source_segment(source, node.value)
            got[node.targets[0].id] = eval(segment, {"np": np, "Problem": Problem})
    return got["values"], got["problem"]


def field_docs() -> dict[str, str]:
    """`field name -> docstring` over every `process/data_structure` dataclass."""
    docs = {}
    for path in sorted(DATA_STRUCTURE.glob("*.py")):
        for cls in ast.walk(ast.parse(path.read_text())):
            if not isinstance(cls, ast.ClassDef):
                continue
            for field, doc in zip(cls.body, cls.body[1:], strict=False):
                if (
                    isinstance(field, ast.AnnAssign)
                    and isinstance(field.target, ast.Name)
                    and isinstance(doc, ast.Expr)
                    and isinstance(doc.value, ast.Constant)
                    and isinstance(doc.value.value, str)
                ):
                    docs.setdefault(field.target.id, " ".join(doc.value.value.split()))
    return docs


_NOT_DESCRIPTION = re.compile(
    r"iteration variable|calculated|sweep variable|issue #|input"
)


def variable_description(i: int, docs: dict[str, str]) -> str:
    """Plain words for iteration variable `i`: its field's docstring, bookkeeping cut.

    The parentheticals that say how PROCESS uses the field (`iteration variable 3`,
    "calculated for stellarators", ...) go, and so does everything after the first
    sentence. An array element is named by the element (`f_nd_impurity_electrons(13)`
    is xenon's).
    """
    variable = ITERATION_VARIABLES[i]
    if variable.target_name == "f_nd_impurity_electron_array":
        element = ImpurityRadiationData().imp_label[variable.array_index].strip("_")
        return f"{element} impurity density fraction (n_{element}/nₑ)"
    doc = docs[variable.target_name or variable.name]
    doc = re.sub(
        r"\s*\((?:[^()]|\([^()]*\))*\)",
        lambda m: "" if _NOT_DESCRIPTION.search(m.group(0)) else m.group(0),
        doc,
    )
    doc = re.split(r"(?<=[a-z)\]])\.\s|(?<=\))\s+(?=[A-Z][a-z])", doc)[0].strip(" .")
    return doc[0].upper() + doc[1:]


def switch_meaning(doc: str) -> tuple[str, dict[int, str]]:
    """A switch docstring's headline and its per-value meanings.

    PROCESS writes the options either as `- =N meaning` or as `N : meaning`.
    """
    doc = re.sub(r"</?\w+>?", "", doc)
    pattern = r"\s-\s*=\s*" if re.search(r"\s-\s*=", doc) else r"\s(?=-?\d+\s*:\s)"
    head, *options = re.split(pattern, " " + doc)
    meanings = {}
    for option in options:
        match = re.match(r"(-?\d+(?:\s*,\s*-?\d+)*)\s*[:,]?\s*(.*)", option)
        if match:
            text = match.group(2).strip(" ;.:")
            text = text if len(text) <= 60 else text[:57].rstrip() + "..."
            for value in re.findall(r"-?\d+", match.group(1)):
                meanings[int(value)] = text
    head = re.split(r"(?<=[a-z)])[.:]\s", head.strip())[0].rstrip(":. ")
    head = re.sub(r"\s*\(`constraint equation \d+`\)", "", head)
    return head, meanings


def value_meaning(key: str, value: int, meanings: dict[int, str]) -> str | None:
    """What `value` of switch `key` means: the enum's name for it, else the docstring's."""
    enum = SWITCH_ENUMS.get(key)
    if enum is not None:
        return " ".join(enum(value).full_name.split())
    return meanings.get(value)


def switches(configurations: dict) -> tuple[list, list]:
    """`(varying, constant)` rows of `(area, name, {config: value})`.

    Every integer switch any configuration states, its value in each configuration
    (the stated one, else PROCESS's default); `varying` are the ones that differ between
    machines, `constant` the ones all seven set alike but away from the default.
    """
    keys = sorted({
        (area, key)
        for values, _ in configurations.values()
        for area, fields in values.items()
        for key, value in fields.items()
        if type(value) is int and key not in NOT_SWITCHES
    })
    varying, constant = [], []
    for area, key in keys:
        default = VALUES.get(area, {}).get(key)
        row = {
            name: values.get(area, {}).get(key, default)
            for name, (values, _) in configurations.items()
        }
        if len(set(row.values())) > 1:
            varying.append((area, key, row))
        elif default != next(iter(row.values())):
            constant.append((area, key, row))
    return varying, constant


def columns(problem: Problem) -> dict[str, list]:
    """The four columns: design variables, objective, equalities, inequalities."""
    objective = [None] if problem.root_find else [problem.i_figure_merit]
    return {
        "dv": list(problem.ixc),
        "obj": objective,
        "eq": sorted(problem.icc[: problem.n_equality]),
        "ineq": sorted(problem.icc[problem.n_equality :]),
    }


def objective_text(fom: int | None) -> str:
    """Plain text for figure of merit `fom` (negative maximises)."""
    if fom is None:
        return "none -- a root find"
    return f"{short_objective(fom)} ({fom})"


# ------------------------------------------------------------------- markdown


def markdown(configurations: dict, docs: dict) -> str:
    """Every table as Markdown."""
    lines = [
        "# The seven reference problems",
        "",
        "Generated by `paper_tests/reference_problems.py` from "
        "`functional_process/configurations/*.py`; regenerate rather than hand-edit. "
        "Design variables are PROCESS iteration variables (`ixc`) with the file's "
        "bounds; constraints are PROCESS constraint equations (`icc`) with PROCESS's "
        "label (`lablcc`), in number order. A root find (`i_process_run_mode = -2`) "
        "has no objective and evaluates its inequalities once at the answer. Where "
        "PROCESS's label is a sentence the tables print a symbol or the inequality "
        "itself; the symbols are:",
        "",
        "| symbol | meaning |",
        "|---|---|",
        *(f"| {symbol} | {meaning} |" for symbol, meaning in GLOSSARY),
        "",
    ]
    for name, (_, problem) in configurations.items():
        cols = columns(problem)
        cells = {
            "dv": [
                f"**{i}** {short_variable(i, docs, problem.bounds[i], tex=False)}"
                for i in cols["dv"]
            ],
            "obj": [objective_text(f) for f in cols["obj"]],
            "eq": [f"**{i}** {short_constraint(i)}" for i in cols["eq"]],
            "ineq": [f"**{i}** {short_constraint(i)}" for i in cols["ineq"]],
        }
        n = max(len(c) for c in cells.values())
        lines += [
            f"## `{name}`",
            "",
            f"{len(cols['dv'])} design variables, {len(cols['eq'])} equalities, "
            f"{len(cols['ineq'])} inequalities.",
            "",
            "| design variables (ixc) | objective | equality constraints (icc) "
            "| inequality constraints (icc) |",
            "|---|---|---|---|",
        ]
        for r in range(n):
            row = [c[r] if r < len(c) else "" for c in cells.values()]
            lines.append("| " + " | ".join(s.replace("|", "\\|") for s in row) + " |")
        lines.append("")

    varying, constant = switches(configurations)
    names = list(configurations)
    lines += [
        "## Model switches",
        "",
        "The integer switches whose value differs between the machines (the stated "
        "value, else PROCESS's default), with the meaning PROCESS's docstring gives "
        "each value where it gives one. A switch that does not apply to a machine (a "
        "tokamak switch on a stellarator) shows PROCESS's default, which that machine "
        "never reads. In the port these are not numbers: each one is already the "
        "choice of model in the configuration's `machine` tree.",
        "",
        "| switch | " + " | ".join(f"`{n}`" for n in names) + " | values |",
        "|---|" + "---|" * len(names) + "---|",
    ]
    for _, key, row in varying:
        head, meanings = switch_meaning(docs.get(key, ""))
        used = "; ".join(
            f"{v} {text}"
            for v in sorted(set(row.values()))
            if (text := value_meaning(key, v, meanings))
        )
        cells = [str(row[n]) for n in names]
        lines.append(f"| `{key}`: {head} | " + " | ".join(cells) + f" | {used} |")
    lines += [
        "",
        "Set alike on all seven, away from PROCESS's default: "
        + ", ".join(f"`{key} = {next(iter(row.values()))}`" for _, key, row in constant)
        + ".",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------- LaTeX

_SUBSCRIPTS = dict(
    zip("ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ₀₁₂₃₄₅₆₇₈₉", "aehijklmnoprstuvx0123456789", strict=True)
)
_SUBSCRIPTS["ᵧ"] = r"\gamma"
_SUBSCRIPTS["ᴛ"] = "T"
_SUPERSCRIPTS = {"³": "3", "²": "2"}
_SYMBOLS = {
    "β": r"\beta",
    "ε": r"\varepsilon",
    "τ": r"\tau",
    "α": r"\alpha",
    "γ": r"\gamma",
    "⟨": r"\langle ",
    "⟩": r"\rangle ",
}
_ESCAPE = {"_": r"\_", "&": r"\&", "%": r"\%", "#": r"\#", "<": "$<$", ">": "$>$"}


def tex_text(s: str) -> str:
    """ASCII text, escaped for LaTeX."""
    return "".join(_ESCAPE.get(c, c) for c in s)


def tex_tt(s: str) -> str:
    """A variable name in typewriter, breakable after each underscore."""
    return r"\texttt{" + s.replace("_", r"\_\allowbreak{}") + "}"


def _tex_math_token(token: str) -> str:
    out, i = [], 0
    while i < len(token):
        c = token[i]
        if c in _SUBSCRIPTS:
            j = i
            while j < len(token) and token[j] in _SUBSCRIPTS:
                j += 1
            sub = r"\mathrm{" + "".join(_SUBSCRIPTS[k] for k in token[i:j]) + "}"
            if out and out[-1].startswith("_{"):  # n_αₜₕ: one subscript, not two
                out[-1] = out[-1][:-1] + sub + "}"
            else:
                out.append("_{" + sub + "}")
            i = j
            continue
        if c == "_" and i + 1 < len(token) and token[i + 1] in _SYMBOLS:
            out.append("_{" + _SYMBOLS[token[i + 1]] + "}")
            i += 2
            continue
        if c == "_" and i + 1 < len(token) and token[i + 1].isascii():
            j = i + 1
            while j < len(token) and token[j].isascii() and token[j].isalpha():
                j += 1
            if j > i + 1:  # n_Xe: an ASCII subscript; nₑ_pedestal: n_{e,pedestal}
                sub = r"\mathrm{" + token[i + 1 : j] + "}"
                if out and out[-1].startswith("_{"):
                    out[-1] = out[-1][:-1] + "," + sub + "}"
                else:
                    out.append("_{" + sub + "}")
                i = j
                continue
        if c in _SYMBOLS:
            out.append(_SYMBOLS[c])
        elif c in _SUPERSCRIPTS:
            out.append("^{" + _SUPERSCRIPTS[c] + "}")
        elif c.isascii() and c.isalpha():
            j = i
            while j < len(token) and token[j].isascii() and token[j].isalpha():
                j += 1
            word = token[i:j]
            out.append(word if len(word) == 1 else r"\mathrm{" + word + "}")
            i = j
            continue
        elif c == "_":
            out.append(r"\_")
        elif c.isascii():
            out.append(c)
        else:
            raise ValueError(f"no LaTeX for {c!r} in {token!r}")
        i += 1
    return "$" + "".join(out) + "$"


def tex_label(s: str) -> str:
    """A PROCESS label, its Unicode maths set as LaTeX maths."""
    return " ".join(
        tex_text(t) if t.isascii() else _tex_math_token(t) for t in s.split(" ")
    )


# ---------------------------------------------------------------- short forms
#
# The tables print a symbol or a formula where PROCESS's label is a sentence, and the
# glossary (`GLOSSARY`) says what each symbol is. Written as LaTeX, used by both outputs
# (the Markdown as `$...$`). Anything not here prints PROCESS's own words.


def _m(tex: str) -> str:
    return f"${tex}$"


VARIABLE_SHORT = {
    2: (_m(r"B_\mathrm{t}(R_0)"), "T"),
    3: (_m("R_0"), "m"),
    4: (_m(r"\langle T_\mathrm{e}\rangle"), "keV"),
    5: (_m(r"\langle\beta\rangle"), ""),
    6: (_m(r"\langle n_\mathrm{e}\rangle"), "m$^{-3}$"),
    10: (_m("H"), ""),
    13: ("TF inboard leg thickness", "m"),
    16: ("CS radial thickness", "m"),
    18: (_m("q_{95}"), ""),
    29: ("Bore radius", "m"),
    37: (_m(r"j_\mathrm{CS,EOF}"), r"A\,m$^{-2}$"),
    41: (_m(r"j_\mathrm{CS,BOP}/j_\mathrm{CS,EOF}"), ""),
    44: (_m(r"f_\mathrm{NI}"), ""),
    56: (_m(r"\tau_\mathrm{Q}"), "s"),
    57: ("TF nose case thickness", "m"),
    58: ("TF turn conduit thickness", "m"),
    59: (_m(r"f_\mathrm{Cu}"), ""),
    60: (_m(r"I_\mathrm{turn}"), "A"),
    93: ("Inboard shield thickness", "m"),
    109: (_m(r"\langle n_{\alpha,\mathrm{th}}\rangle/\langle n_\mathrm{e}\rangle"), ""),
    122: ("CS turn steel fraction", ""),
    135: (_m(r"n_\mathrm{Xe}/n_\mathrm{e}"), ""),
    140: ("TF winding pack radial thickness", "m"),
    145: (_m(r"f_\mathrm{GW,ped}"), ""),
    152: (_m(r"f_\mathrm{GW,sep}"), ""),
}
"""`ixc -> (short form, unit)`: the table prints `form ∈ [lower, upper] unit`."""

OBJECTIVE_SHORT = {
    1: _m("R_0"),
    5: _m("Q"),
    6: "CoE",
    7: "capital cost",
    14: _m(r"t_\mathrm{burn}"),
}
"""`|i_figure_merit| -> short form`; the sign says min or max."""

_LE, _GE = r"\le\mathrm{max}", r"\ge\mathrm{min}"

CONSTRAINT_SHORT = {
    1: _m(r"\langle\beta\rangle") + " consistency",
    2: "Power balance",
    5: _m(r"\langle n_\mathrm{e}\rangle \le n_\mathrm{lim}"),
    8: _m(rf"q_\mathrm{{n}} {_LE}"),
    9: _m(rf"P_\mathrm{{fus}} {_LE}"),
    11: "Radial build",
    13: _m(rf"t_\mathrm{{burn}} {_GE}"),
    15: _m(r"P_\mathrm{sep} \ge P_\mathrm{LH}"),
    16: _m(rf"P_\mathrm{{net}} {_GE}"),
    17: _m(rf"f_\mathrm{{rad}} {_LE}"),
    18: _m(rf"q_\mathrm{{div}} {_LE}"),
    24: _m(rf"\langle\beta\rangle {_LE}"),
    25: _m(rf"B_\mathrm{{max}} {_LE}"),
    26: _m(r"j_\mathrm{CS} \le j_\mathrm{crit}") + " (EOF)",
    27: _m(r"j_\mathrm{CS} \le j_\mathrm{crit}") + " (BOP)",
    30: _m(rf"P_\mathrm{{aux}} {_LE}"),
    31: _m(rf"\sigma_\mathrm{{TF,case}} {_LE}"),
    32: _m(rf"\sigma_\mathrm{{TF,conduit}} {_LE}"),
    33: _m(r"j_\mathrm{TF,WP} \le j_\mathrm{crit}"),
    34: _m(rf"V_\mathrm{{TF}} {_LE}"),
    35: _m(r"j_\mathrm{TF,WP} \le j_\mathrm{quench}"),
    36: _m(rf"\Delta T_\mathrm{{TF}} {_GE}"),
    46: _m(rf"I_\mathrm{{p}}/I_\mathrm{{rod}} {_LE}"),
    56: _m(rf"P_\mathrm{{sep}}/R_0 {_LE}"),
    60: _m(rf"\Delta T_\mathrm{{CS}} {_GE}"),
    62: _m(rf"\tau_\alpha/\tau_\mathrm{{E}} {_GE}"),
    65: _m(rf"\sigma_\mathrm{{VV}} {_LE}") + " (quench)",
    67: _m(rf"q_\mathrm{{FW,rad}} {_LE}"),
    68: _m(rf"P_\mathrm{{sep}} B_\mathrm{{T}}/(q_{{95}} A R_0) {_LE}"),
    72: _m(rf"\sigma_\mathrm{{CS,Tresca}} {_LE}"),
    81: _m(r"n_\mathrm{e0} \ge n_\mathrm{e,ped}"),
    82: _m(r"d_\mathrm{min} \ge w_\mathrm{coil}"),
    83: _m(r"d_\mathrm{pc} \ge \textstyle\sum d"),
    84: _m(rf"\langle\beta\rangle {_GE}"),
    90: _m(rf"N_\mathrm{{cycles,CS}} {_GE}"),
}
"""`icc -> short form`: the constraint as the inequality it states."""

GLOSSARY = (
    (_m(r"B_\mathrm{t}(R_0)"), "toroidal field on axis"),
    (_m("R_0"), "plasma major radius"),
    (_m("A"), "aspect ratio"),
    (_m(r"\langle T_\mathrm{e}\rangle"), "volume-averaged electron temperature"),
    (_m(r"\langle n_\mathrm{e}\rangle"), "volume-averaged electron density"),
    (
        _m(r"n_\mathrm{e0}, n_\mathrm{e,ped}"),
        "electron density on axis, at the pedestal",
    ),
    (_m(r"n_\mathrm{lim}"), "density limit"),
    (_m(r"\langle n_{\alpha,\mathrm{th}}\rangle"), "thermal alpha density"),
    (_m(r"n_\mathrm{Xe}"), "xenon impurity density"),
    (
        _m(r"f_\mathrm{GW,ped}, f_\mathrm{GW,sep}"),
        "Greenwald fraction of the pedestal, separatrix density",
    ),
    (_m(r"\langle\beta\rangle"), "volume-averaged total beta"),
    (
        _m("H"),
        r"confinement enhancement factor (hfact; $f_\mathrm{ren}$ in \cite{lion2021general})",
    ),
    (_m("q_{95}"), "safety factor at the 95\\% flux surface"),
    (_m(r"f_\mathrm{NI}"), "non-inductive fraction of the plasma current"),
    (_m(r"I_\mathrm{p}, I_\mathrm{rod}"), "plasma current, centre-post current"),
    (_m(r"\tau_\alpha, \tau_\mathrm{E}"), "alpha particle, energy confinement time"),
    (_m("Q"), "fusion gain"),
    (_m(r"t_\mathrm{burn}"), "burn time (pulse length)"),
    ("CoE", "cost of electricity"),
    (_m(r"P_\mathrm{fus}, P_\mathrm{net}"), "fusion power, net electric power"),
    (
        _m(r"P_\mathrm{sep}, P_\mathrm{LH}"),
        "power across the separatrix, L--H threshold power",
    ),
    (_m(r"P_\mathrm{aux}"), "injected auxiliary power"),
    (_m(r"f_\mathrm{rad}"), "radiated power fraction"),
    (_m(r"q_\mathrm{n}"), "neutron wall load"),
    (
        _m(r"q_\mathrm{div}, q_\mathrm{FW,rad}"),
        "divertor heat load, first-wall radiation load",
    ),
    (_m(r"B_\mathrm{max}"), "peak field on the coil"),
    (_m(r"j_\mathrm{CS}, j_\mathrm{TF,WP}"), "CS, TF winding pack current density"),
    (
        _m(r"j_\mathrm{crit}, j_\mathrm{quench}"),
        "critical current density, quench (hot-spot) limit",
    ),
    ("EOF, BOP", "end of flat-top, beginning of pulse"),
    (_m(r"\sigma"), "stress (TF case, TF conduit, vacuum vessel, CS Tresca)"),
    (
        _m(r"V_\mathrm{TF}, \tau_\mathrm{Q}"),
        "TF quench dump voltage, discharge time",
    ),
    (
        _m(r"\Delta T_\mathrm{TF}, \Delta T_\mathrm{CS}"),
        "superconductor temperature margin",
    ),
    (_m(r"I_\mathrm{turn}"), "TF current per turn"),
    (_m(r"f_\mathrm{Cu}"), "copper fraction of the TF cable"),
    (_m(r"N_\mathrm{cycles,CS}"), "achievable CS stress cycles"),
    (
        _m(r"d_\mathrm{min}, w_\mathrm{coil}"),
        "minimal distance between coils, coil toroidal width",
    ),
    (
        _m(r"d_\mathrm{pc}, \textstyle\sum d"),
        "plasma--coil distance, required radial build",
    ),
)
"""Every symbol the short forms use, and what it is."""


def short_variable(
    i: int, docs: dict[str, str], bounds: tuple[float, float], *, tex: bool = True
) -> str:
    """Design variable `i` and its bounds: `form ∈ [lower, upper] unit`, the form
    PROCESS's own words where there is no short one; the numbers `\\num` for LaTeX."""
    form, unit = VARIABLE_SHORT.get(i, (tex_label(variable_description(i, docs)), ""))
    lower, upper = (rf"\num{{{b:g}}}" if tex else f"{b:g}" for b in bounds)
    interval = f"[{lower}, {upper}]"
    return rf"{form} $\in$ {interval}" + (f" {unit}" if unit else "")


def short_objective(fom: int) -> str:
    """Figure of merit `fom` as LaTeX, `min`/`max` first."""
    sense = "max" if fom < 0 else "min"
    short = OBJECTIVE_SHORT.get(abs(fom))
    return f"{sense} " + (short or tex_label(FiguresOfMerit(abs(fom)).description))


def short_constraint(i: int) -> str:
    """Constraint `i` as LaTeX: its inequality, else PROCESS's label."""
    return CONSTRAINT_SHORT.get(i) or tex_label(CONSTRAINT_LABELS[i - 1])


_UNBROKEN = r"\setlength{\aboverulesep}{0pt}\setlength{\belowrulesep}{0pt}"
"""booktabs pads its horizontal rules with space a `\vrule` cannot cross; the tables
with vertical rules drop that padding and put it back inside the rows as struts."""
_TOP = r"\rule{0pt}{2.6ex}"
_BOTTOM = r"\rule[-1.2ex]{0pt}{0pt}"


def _tex_row(cells, strut: str = "") -> str:
    """One row of ragged-right paragraph cells (no `array` package needed), `strut`
    in the first to set the row's height or depth."""
    cells = [rf"\raggedright {c}" if c else "" for c in cells]
    cells[0] = strut + cells[0]
    return " & ".join(cells) + r" \tabularnewline"


_PAD = 5
"""Points either side of a column separator -- stated, since REVTeX sets `\tabcolsep`
to 2pt, which leaves a rule touching the text."""
_GAP = rf"@{{\hspace{{{2 * _PAD}pt}}}}"
_RULE = rf"@{{\hspace{{{_PAD}pt}}\vrule width 0.2pt\relax\hspace{{{_PAD}pt}}}}"
"""Column separators: a plain gap, and the same gap split by a thin rule."""


def _widths(fractions, separators) -> list[str]:
    """Paragraph-column widths that, with `separators` between them, fill
    `\textwidth` exactly: each column its fraction, less an equal share of the gaps."""
    rules = sum(sep == _RULE for sep in separators)
    gaps = 2 * _PAD * len(separators) + 0.2 * rules
    share = f"{gaps / len(fractions):.3f}pt"
    return [rf"\dimexpr {f}\textwidth-{share}\relax" for f in fractions]


def _spec(fractions, separators) -> str:
    """A `tabular` preamble of `p` columns joined by `separators`, edge to edge."""
    columns = [f"p{{{w}}}" for w in _widths(fractions, separators)]
    body = columns[0] + "".join(
        sep + col for sep, col in zip(separators, columns[1:], strict=True)
    )
    return "@{}" + body + "@{}"


def _tex_list(entries, width: str) -> str:
    """A column's entries as their own top-aligned list: numbers right-aligned in a
    column of their own, each text wrapping beside its number. Each of the four columns
    is one such list, so no column waits on a tall entry in another."""
    last = len(entries) - 1
    rows = "\n".join(
        rf"{number} & \raggedright {_TOP if r == 0 else ''}{text}"
        rf"{_BOTTOM if r == last else ''} \tabularnewline"
        for r, (number, text) in enumerate(entries)
    )
    return (
        r"\begin{tabular}[t]{@{}r@{\hspace{0.6em}}"
        + rf"p{{{width}-2.6em}}@{{}}}}"
        + "\n"
        + rows
        + "\n"
        + r"\end{tabular}"
    )


WIDTHS = (0.37, 0.15, 0.19, 0.29)
"""The four columns' shares of `\textwidth`, thin rules between them."""
SEPARATORS = (_RULE, _RULE, _RULE)


def _tex_glossary() -> list[str]:
    """`GLOSSARY` as one float, its entries in two side-by-side column pairs."""
    half = (len(GLOSSARY) + 1) // 2
    left, right = GLOSSARY[:half], GLOSSARY[half:]
    rows = [
        _tex_row(
            (*left[r], *(right[r] if r < len(right) else ("", ""))),
            (_TOP if r == 0 else "") + (_BOTTOM if r == half - 1 else ""),
        )
        for r in range(half)
    ]
    return [
        r"\begin{table*}[htbp]",
        r"\centering\scriptsize",
        r"\caption{}  % TODO: caption -- the symbols in the problem tables",
        r"\label{tab:problem-symbols}",
        _UNBROKEN,
        r"\begin{tabular}{" + _spec((0.15, 0.35, 0.15, 0.35), (_GAP, _RULE, _GAP)) + "}",
        r"\toprule",
        _tex_row(("symbol", "meaning", "symbol", "meaning"), _TOP + _BOTTOM),
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ]


def latex(configurations: dict, docs: dict) -> str:
    """The same tables as LaTeX floats, the switch matrix without its meanings."""
    parts = [
        "% Generated by PROCESS's paper_tests/reference_problems.py -- regenerate,",
        "% do not edit. Needs booktabs, graphicx and siunitx.",
        "",
        *_tex_glossary(),
    ]
    for name, (_, problem) in configurations.items():
        cols = columns(problem)
        lists = [
            [
                (
                    i,
                    short_variable(i, docs, problem.bounds[i]),
                )
                for i in cols["dv"]
            ],
            [
                ("", "none (root find)")
                if fom is None
                else (
                    fom,
                    short_objective(fom),
                )
                for fom in cols["obj"]
            ],
            [(i, short_constraint(i)) for i in cols["eq"]],
            [(i, short_constraint(i)) for i in cols["ineq"]],
        ]
        parts += [
            r"\begin{table*}[htbp]",
            r"\centering\scriptsize",
            rf"\caption{{\texttt{{{tex_text(name)}}}}}  % TODO: caption",
            rf"\label{{tab:problem-{name.replace('_', '-')}}}",
            _UNBROKEN,
            r"\begin{tabular}{" + _spec(WIDTHS, SEPARATORS) + "}",
            r"\toprule",
            _tex_row(
                (
                    r"design variables (\texttt{ixc})",
                    "objective",
                    r"equality constraints (\texttt{icc})",
                    r"inequality constraints (\texttt{icc})",
                ),
                _TOP + _BOTTOM,
            ),
            r"\midrule",
            " &\n".join(
                _tex_list(entries, w)
                for entries, w in zip(lists, _widths(WIDTHS, SEPARATORS), strict=True)
            )
            + r" \tabularnewline",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]

    varying, _ = switches(configurations)
    names = list(configurations)
    parts += [
        r"\begin{table*}[htbp]",
        r"\centering\scriptsize\setlength{\tabcolsep}{2.5pt}",
        r"\caption{}  % TODO: caption -- the integer switches that differ between "
        r"the machines, stated value else PROCESS's default",
        r"\label{tab:problem-switches}",
        r"\begin{tabular}{@{}l" + "c" * len(names) + "@{}}",
        r"\toprule",
        "switch & "
        + " & ".join(rf"\rotatebox{{70}}{{\texttt{{{tex_text(n)}}}}}" for n in names)
        + r" \\",
        r"\midrule",
        *(
            tex_tt(key) + " & " + " & ".join(str(row[n]) for n in names) + r" \\"
            for _, key, row in varying
        ),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ]
    return "\n".join(parts)


def main() -> None:
    """Write `out/reference_problems.{md,tex}`, and with `--paper` the paper's copy."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--paper", action="store_true", help=f"also write the .tex to {PAPER}"
    )
    args = parser.parse_args()
    configurations = {name: stated(name) for name in NAMES}
    docs = field_docs()
    OUT.mkdir(exist_ok=True)
    tex = latex(configurations, docs)
    (OUT / "reference_problems.md").write_text(markdown(configurations, docs))
    (OUT / "reference_problems.tex").write_text(tex)
    print(f"wrote {OUT / 'reference_problems.md'} and .tex")
    if args.paper:
        PAPER.mkdir(parents=True, exist_ok=True)
        (PAPER / "reference_problems.tex").write_text(tex)
        print(f"wrote {PAPER / 'reference_problems.tex'}")


if __name__ == "__main__":
    main()
