"""Part A one-shot tooling: the declaration-surface codemod. Delete when Part A closes.

`_audit/path_refactor.md` §A.3. Rewrites the port's declaration surface from the
`lambda s: s.<area>.<field>` escape hatch to cottax's sugar, per file, by exact
source-span replacement driven by `ast`:

    param=FromExactly(lambda s: s.A.param)   ->  param=From(A)
    attr = Output(lambda s: s.A.attr)        ->  attr = OutputInto(A)

    FromExactly(lambda s: s.A.arr[2])        ->  FromExactly(A.arr[2])
    attr = Output(lambda s: s.A.arr[2])      ->  attr = Output(A.arr[2])

The third and fourth are §A.6's escape hatches: the *lambda* dies, the escape-hatch
call stays, because no declaring name can spell one element of an array.

**Sites where the field is spelled differently from the declaring name are not
converted.** Those are §A.5's 36 renames -- the parameter has to be renamed to the field
name, in the signature and in the body, and that is a human's call about the body's
readability, not a codemod's. They are listed instead (`--census`, and on every run).

`libcst` is not installed in `process_port` and must not be added (§A.3). Plain `ast`
plus span replacement is sufficient: every target is a single expression with no
interior comments, and `ruff format` normalises the result.

Usage::

    PY=~/miniconda3/envs/process_port/bin/python

    # census only, nothing written -- the §0 figures, per file and in total
    $PY -m functional_process._codemod.convert_declarations --census

    # convert one file (codemod + ruff format + ruff check on that file only)
    $PY -m functional_process._codemod.convert_declarations \
        functional_process/models/vacuum.py

    # see the diff without writing it
    $PY -m functional_process._codemod.convert_declarations --dry-run <path> ...

Run from the repository root; paths are repo-relative.
"""

# ruff: noqa: S404, S603  -- this tool shells out to `ruff` on purpose
from __future__ import annotations

import argparse
import ast
import difflib
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

COTTAX_MODULE = "cottax.interfaces.pytree_namespace_module"
PATHS_MODULE = "functional_process.paths"

READ_EXACT, READ_SUGAR = "FromExactly", "From"
WRITE_EXACT, WRITE_SUGAR = "Output", "OutputInto"

MECHANICAL, RENAME, ESCAPE, CONVERTED, OTHER = (
    "mechanical",
    "rename",
    "escape",
    "converted",
    "other",
)
DECLARATION_CALLS = frozenset({READ_EXACT, READ_SUGAR, WRITE_EXACT, WRITE_SUGAR})
SUGAR_CALLS = frozenset({READ_SUGAR, WRITE_SUGAR})

ESCAPE_PENDING = "escape_pending"
"""An escape hatch that still carries a `lambda s:` -- §A.6 work, inside the 43."""


# --------------------------------------------------------------------- the sites


@dataclass
class Site:
    """One declaration-surface call, classified."""

    category: str
    call: ast.Call
    kind: str  # "read" or "write"
    decl_name: str | None  # the parameter name, or the class attribute name
    class_name: str | None
    area: str | None
    field: str | None
    replacement: str | None  # the new source text, or None if not convertible

    @property
    def line(self) -> int:
        return self.call.lineno

    @property
    def pending(self) -> bool:
        """Still carries a `lambda s:`, so there is work here."""
        return bool(self.call.args) and isinstance(self.call.args[0], ast.Lambda)


def _attribute_chain(node: ast.expr) -> tuple[str, list[str]] | None:
    """`s.physics.rmajor` -> `("s", ["physics", "rmajor"])`; None if not such a chain."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.reverse()
    return node.id, parts


def _classify_place(src: str, call: ast.Call) -> tuple[str | None, str | None, bool]:
    """`(area, field, subscripted)` for the single argument of a `FromExactly`/`Output`.

    `area`/`field` are None when the argument is not a `lambda s: s.<area>.<field>`
    chain (with any number of trailing subscripts) -- such a site is reported and left
    alone rather than guessed at.
    """
    if len(call.args) != 1 or call.keywords:
        return None, None, False
    arg = call.args[0]
    if not isinstance(arg, ast.Lambda):
        return None, None, False  # already the recorder form, or something else
    if len(arg.args.args) != 1 or arg.args.posonlyargs or arg.args.kwonlyargs:
        return None, None, False
    root_name = arg.args.args[0].arg

    body = arg.body
    subscripted = False
    while isinstance(body, ast.Subscript):
        subscripted = True
        body = body.value

    chain = _attribute_chain(body)
    if chain is None:
        return None, None, subscripted
    base, parts = chain
    if base != root_name or len(parts) != 2:
        return None, None, subscripted
    return parts[0], parts[1], subscripted


def _delambda(src: str, call: ast.Call) -> str:
    """`FromExactly(lambda s: s.A.arr[2])` -> `FromExactly(A.arr[2])`, verbatim body."""
    lam = call.args[0]
    assert isinstance(lam, ast.Lambda)
    root_name = lam.args.args[0].arg
    body_src = ast.get_source_segment(src, lam.body)
    if body_src is None:  # pragma: no cover -- defensive
        body_src = ast.unparse(lam.body)
    body_src = " ".join(body_src.split())
    prefix = root_name + "."
    assert body_src.startswith(prefix), body_src
    return f"{call.func.id}({body_src[len(prefix) :]})"


def _param_defaults(fn: ast.FunctionDef) -> list[tuple[str, ast.expr]]:
    """`(parameter name, default)` for every parameter that has one."""
    a = fn.args
    positional = a.posonlyargs + a.args
    pairs: list[tuple[str, ast.expr]] = []
    if a.defaults:
        for arg, dflt in zip(positional[-len(a.defaults) :], a.defaults, strict=True):
            pairs.append((arg.arg, dflt))
    for arg, dflt in zip(a.kwonlyargs, a.kw_defaults, strict=True):
        if dflt is not None:
            pairs.append((arg.arg, dflt))
    return pairs


def collect_sites(src: str, tree: ast.Module) -> list[Site]:
    """Every `From`/`FromExactly`/`Output`/`OutputInto` call in the file, classified."""
    declaration_calls: dict[int, tuple[str, str | None, str | None]] = {}
    # id(call) -> (kind, declaring name, enclosing class)

    def note(call: ast.expr, kind: str, name: str, cls: str | None) -> None:
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id in DECLARATION_CALLS
        ):
            declaration_calls[id(call)] = (kind, name, cls)

    class Walker(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                    target = stmt.targets[0]
                    if isinstance(target, ast.Name):
                        note(stmt.value, "write", target.id, node.name)
                elif isinstance(stmt, ast.AnnAssign) and isinstance(
                    stmt.target, ast.Name
                ):
                    if stmt.value is not None:
                        note(stmt.value, "write", stmt.target.id, node.name)
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def _function(self, node: ast.FunctionDef) -> None:
            cls = self.stack[-1] if self.stack else None
            for name, dflt in _param_defaults(node):
                note(dflt, "read", name, cls)
            self.generic_visit(node)

        visit_FunctionDef = _function
        visit_AsyncFunctionDef = _function

    Walker().visit(tree)

    sites: list[Site] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        fname = node.func.id
        if fname not in DECLARATION_CALLS:
            continue

        if fname in SUGAR_CALLS:
            kind, decl, cls = declaration_calls.get(
                id(node), ("read" if fname == READ_SUGAR else "write", None, None)
            )
            area = (
                node.args[0].id
                if node.args and isinstance(node.args[0], ast.Name)
                else None
            )
            sites.append(Site(CONVERTED, node, kind, decl, cls, area, decl, None))
            continue

        area, fld, subscripted = _classify_place(src, node)
        declared = declaration_calls.get(id(node))
        kind = declared[0] if declared else ("read" if fname == READ_EXACT else "write")
        decl = declared[1] if declared else None
        cls = declared[2] if declared else None

        if declared is None:
            # Not a parameter default and not a class attribute: `_rebound_signature`'s
            # keyword rebind, a test's `Output(...).port()`. Out of the §0 census; the
            # lambda can still be shed (--include-other), the call cannot be sugared.
            repl = _delambda(src, node) if area or subscripted else None
            sites.append(Site(OTHER, node, kind, decl, cls, area, fld, repl))
            continue

        if subscripted or area is None:
            repl = _delambda(src, node) if isinstance(node.args[0], ast.Lambda) else None
            sites.append(Site(ESCAPE, node, kind, decl, cls, area, fld, repl))
            continue

        if fld == decl:
            sugar = READ_SUGAR if kind == "read" else WRITE_SUGAR
            sites.append(
                Site(MECHANICAL, node, kind, decl, cls, area, fld, f"{sugar}({area})")
            )
        else:
            sites.append(Site(RENAME, node, kind, decl, cls, area, fld, None))

    sites.sort(key=lambda s: (s.call.lineno, s.call.col_offset))
    return sites


# ------------------------------------------------------------------ source spans


def _line_starts(src: str) -> list[int]:
    starts, pos = [0], 0
    for line in src.splitlines(keepends=True):
        pos += len(line)
        starts.append(pos)
    return starts


def _span(starts: list[int], node: ast.AST) -> tuple[int, int]:
    return (
        starts[node.lineno - 1] + node.col_offset,
        starts[node.end_lineno - 1] + node.end_col_offset,
    )


def _apply(src: str, edits: list[tuple[int, int, str]]) -> str:
    out = src
    for begin, end, text in sorted(edits, reverse=True):
        out = out[:begin] + text + out[end:]
    return out


# ---------------------------------------------------------------------- imports


def _member_sort_key(name: str) -> tuple[int, str]:
    """ruff-isort's `order-by-type` member order: constants, classes, then functions."""
    if name.isupper() and len(name) > 1:
        rank = 0
    elif name[:1].isupper():
        rank = 1
    else:
        rank = 2
    return rank, name.lower()


def _render_import(module: str, names: list[str]) -> str:
    ordered = sorted(set(names), key=_member_sort_key)
    return f"from {module} import ({', '.join(ordered)})"


def _still_uses(tree: ast.Module, name: str) -> bool:
    """Is `name` referenced anywhere outside its own import statement?"""
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == name:
            return True
        if isinstance(node, ast.Attribute) and node.attr == name:
            return True
    return False


def rewrite_imports(src: str, added_areas: set[str], added_sugar: set[str]) -> str:
    """Add `From`/`OutputInto` and the areas; drop `FromExactly`/`Output` if now unused.

    Raises
    ------
    RuntimeError
        If sugar has to be added but the file has no cottax import to add it to.
    """
    tree = ast.parse(src)
    starts = _line_starts(src)
    edits: list[tuple[int, int, str]] = []

    cottax_import = paths_import = None
    top_imports: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            top_imports.append(node)
        if isinstance(node, ast.ImportFrom):
            if node.module == COTTAX_MODULE:
                cottax_import = node
            elif node.module == PATHS_MODULE:
                paths_import = node

    if cottax_import is not None:
        names = [a.name for a in cottax_import.names]
        names.extend(added_sugar)
        for exact in (READ_EXACT, WRITE_EXACT):
            if exact in names and not _still_uses(tree, exact):
                names.remove(exact)
        begin, end = _span(starts, cottax_import)
        edits.append((begin, end, _render_import(COTTAX_MODULE, names)))
    elif added_sugar:
        raise RuntimeError(
            f"no `from {COTTAX_MODULE} import ...` to add {sorted(added_sugar)} to"
        )

    if added_areas:
        if paths_import is not None:
            names = [a.name for a in paths_import.names] + sorted(added_areas)
            begin, end = _span(starts, paths_import)
            edits.append((begin, end, _render_import(PATHS_MODULE, names)))
        else:
            anchor = _paths_import_anchor(top_imports)
            begin = starts[anchor.lineno - 1]
            edits.append((
                begin,
                begin,
                _render_import(PATHS_MODULE, sorted(added_areas)) + "\n",
            ))

    return _apply(src, edits)


def _paths_import_anchor(top_imports: list[ast.stmt]) -> ast.stmt:
    """The statement a new `functional_process.paths` import sorts before."""
    for node in top_imports:
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module > PATHS_MODULE
            and not node.module.startswith("cottax")
        ):
            return node
    return top_imports[-1]


# ------------------------------------------------------------------- conversion


@dataclass
class FileReport:
    """What one file holds, and what a conversion did to it."""

    path: Path
    counts: Counter = field(default_factory=Counter)
    renames: list[Site] = field(default_factory=list)
    others: list[Site] = field(default_factory=list)
    unparseable: list[Site] = field(default_factory=list)
    changed: bool = False

    @property
    def unconverted(self) -> int:
        """Sites still to convert -- an already-de-lambda'd escape hatch is *done*."""
        return (
            self.counts[MECHANICAL] + self.counts[RENAME] + self.counts[ESCAPE_PENDING]
        )


def analyse(path: Path) -> FileReport:
    """Classify every declaration in one file. Reads only."""
    src = path.read_text()
    report = FileReport(path=path)
    if not any(tok in src for tok in (READ_EXACT, WRITE_EXACT, READ_SUGAR, WRITE_SUGAR)):
        return report
    for site in collect_sites(src, ast.parse(src)):
        report.counts[site.category] += 1
        if site.category == ESCAPE and site.pending:
            report.counts[ESCAPE_PENDING] += 1
        if site.category == RENAME:
            report.renames.append(site)
        elif site.category == OTHER:
            report.others.append(site)
        elif site.category == ESCAPE and site.replacement is None:
            report.unparseable.append(site)
    return report


def convert(path: Path, *, include_other: bool, dry_run: bool) -> FileReport:
    """Convert one file in place (§A.3 steps 1-2). Returns what it found and did."""
    src = path.read_text()
    tree = ast.parse(src)
    sites = collect_sites(src, tree)
    starts = _line_starts(src)

    report = FileReport(path=path)
    edits: list[tuple[int, int, str]] = []
    added_areas: set[str] = set()
    added_sugar: set[str] = set()

    for site in sites:
        report.counts[site.category] += 1
        if site.category == ESCAPE and site.pending:
            report.counts[ESCAPE_PENDING] += 1
        if site.category == RENAME:
            report.renames.append(site)
            continue
        if site.category == OTHER:
            report.others.append(site)
            if not include_other or site.replacement is None:
                continue
        if site.category == ESCAPE and site.replacement is None:
            report.unparseable.append(site)
            continue
        if site.replacement is None or site.category == CONVERTED:
            continue

        begin, end = _span(starts, site.call)
        edits.append((begin, end, site.replacement))
        if site.area:
            added_areas.add(site.area)
        if site.category == MECHANICAL:
            added_sugar.add(READ_SUGAR if site.kind == "read" else WRITE_SUGAR)

    if not edits:
        return report

    out = _apply(src, edits)
    out = rewrite_imports(out, added_areas, added_sugar)
    ast.parse(out)  # the codemod must not produce something unparseable

    report.changed = True
    if dry_run:
        sys.stdout.writelines(
            difflib.unified_diff(
                src.splitlines(keepends=True),
                out.splitlines(keepends=True),
                fromfile=str(path),
                tofile=str(path) + " (converted)",
            )
        )
    else:
        path.write_text(out)
    return report


# ------------------------------------------------------------------------ ruff


def _ruff() -> str:
    return str(Path(sys.executable).parent / "ruff")


def run_ruff(path: Path) -> int:
    """§A.3 step 3, on this file only -- never on files the codemod did not touch."""
    ruff = _ruff()
    subprocess.run([ruff, "format", str(path)], check=False)
    # The codemod is what disturbed the import block; let isort put it back in order.
    subprocess.run(
        [ruff, "check", "--select", "I", "--fix", "--quiet", str(path)], check=False
    )
    subprocess.run([ruff, "format", str(path)], check=False)
    return subprocess.run([ruff, "check", str(path)], check=False).returncode


# ---------------------------------------------------------------------- reports


def _iter_python(targets: list[str]) -> list[Path]:
    paths: list[Path] = []
    for target in targets:
        p = Path(target)
        paths.extend(sorted(p.rglob("*.py")) if p.is_dir() else [p])
    return paths


def print_census(reports: list[FileReport]) -> None:
    """The §0 table: per file, then the repo total."""
    width = max((len(str(r.path)) for r in reports if r.counts), default=10)
    header = (
        f"{'file':<{width}}  {'mech':>5} {'rename':>6} {'escape':>6} "
        f"{'(todo)':>6} {'done':>5}"
    )
    print(header)
    print("-" * len(header))
    total = Counter()
    for report in sorted(reports, key=lambda r: -r.unconverted):
        if not report.counts:
            continue
        total.update(report.counts)
        print(
            f"{report.path!s:<{width}}  "
            f"{report.counts[MECHANICAL]:>5} {report.counts[RENAME]:>6} "
            f"{report.counts[ESCAPE]:>6} {report.counts[ESCAPE_PENDING]:>6} "
            f"{report.counts[CONVERTED]:>5}"
        )
    print("-" * len(header))
    print(
        f"{'TOTAL':<{width}}  "
        f"{total[MECHANICAL]:>5} {total[RENAME]:>6} "
        f"{total[ESCAPE]:>6} {total[ESCAPE_PENDING]:>6} {total[CONVERTED]:>5}"
    )
    declarations = total[MECHANICAL] + total[RENAME] + total[ESCAPE] + total[CONVERTED]
    print(
        f"\n{declarations} declarations: {total[MECHANICAL]} mechanical, "
        f"{total[RENAME]} body-rename, {total[ESCAPE]} escape-hatch, "
        f"{total[CONVERTED]} already converted"
    )
    files = sum(1 for r in reports if r.unconverted)
    todo = total[MECHANICAL] + total[RENAME] + total[ESCAPE_PENDING]
    print(f"{files} files hold the {todo} sites still to convert")
    if total[OTHER]:
        print(
            f"\n{total[OTHER]} `lambda s:` uses outside the declaration surface "
            f"(not declarations, not in the census above -- see --include-other)"
        )


def print_renames(reports: list[FileReport]) -> None:
    """§A.5: the sites a human converts, grouped by file."""
    rows = [(r, s) for r in reports for s in r.renames]
    if not rows:
        return
    print(
        f"\n{len(rows)} §A.5 rename(s) -- parameter/attribute spelled unlike the field:"
    )
    current = None
    for report, site in rows:
        if report.path != current:
            current = report.path
            print(f"  {report.path}")
        print(
            f"    L{site.line:<5} {site.class_name}.{site.decl_name}"
            f"  ->  .{site.area}.{site.field}"
        )


def print_others(reports: list[FileReport]) -> None:
    rows = [(r, s) for r in reports for s in r.others]
    if not rows:
        return
    print(f"\n{len(rows)} non-declaration `lambda s:` use(s) (out of census):")
    for report, site in rows:
        place = f".{site.area}.{site.field}" if site.area else "?"
        print(f"  {report.path}:{site.line}  {site.call.func.id}({place})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "targets",
        nargs="*",
        default=["functional_process"],
        help="files or directories, repo-relative (default: functional_process)",
    )
    parser.add_argument(
        "--census", action="store_true", help="count only; write nothing"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the diff instead of writing it"
    )
    parser.add_argument(
        "--include-other",
        action="store_true",
        help="also shed the lambda at non-declaration sites (rebinds, tests)",
    )
    parser.add_argument(
        "--no-ruff", action="store_true", help="skip §A.3 step 3 (format + check)"
    )
    args = parser.parse_args(argv)

    paths = _iter_python(args.targets or ["functional_process"])

    if args.census:
        reports = [analyse(p) for p in paths]
        print_census(reports)
        print_renames(reports)
        print_others(reports)
        return 0

    reports: list[FileReport] = []
    status = 0
    for path in paths:
        report = convert(path, include_other=args.include_other, dry_run=args.dry_run)
        reports.append(report)
        if report.changed and not args.dry_run and not args.no_ruff:
            print(f"\n--- ruff: {path}")
            status |= run_ruff(path)

    for report in reports:
        if report.changed:
            print(
                f"\n{report.path}: {report.counts[MECHANICAL]} mechanical, "
                f"{report.counts[ESCAPE]} escape, {report.counts[RENAME]} rename(s) left"
            )
    print_renames(reports)
    print_others(reports)
    for report in reports:
        for site in report.unparseable:
            print(f"  !! {report.path}:{site.line} unrecognised place, left alone")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
