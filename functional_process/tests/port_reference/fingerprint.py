"""A model graph's structure as plain JSON, comparable across cottax versions.

Phase 0 of porting the port onto current cottax: every configuration's model graph is
written here from the old cottax (`ea3abbe`), and Phase 1 must reproduce it from the new
one. A node is keyed by what it owns, not by its name, since naming changes with the
surface; its name is kept only to read the diff.

Run from ~/PROCESS with the cottax under test first on the path:

    PYTHONPATH=<jaxgraph checkout>/src JAX_PLATFORMS=cpu $PY \\
        -m functional_process.tests.port_reference.fingerprint [--out DIR]

and compare two directories with `--compare OLD NEW`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).parent


def segment(seg) -> list:
    """A key as `[kind, value]`: `DictKey(0)` and `SequenceKey(0)` stay apart."""
    for attr in ("name", "key", "idx", "ns"):
        if hasattr(seg, attr):
            return [type(seg).__name__, repr(getattr(seg, attr))]
    return [type(seg).__name__, repr(seg)]


def place(p) -> list:
    return [segment(s) for s in p.segments]


def body(impl) -> str:
    """The physics function a node runs: a declaration's `fn` if it has one, else what
    the implementation is, by module and qualified name."""
    target = getattr(type(impl), "fn", None) or getattr(impl, "fn", None) or impl
    kind = target if isinstance(target, type) or callable(target) and hasattr(
        target, "__qualname__") else type(target)
    return f"{getattr(kind, '__module__', '?')}:{getattr(kind, '__qualname__', repr(kind))}"


def settings(impl) -> dict:
    """What an implementation holds besides its reads: its dataclass fields, by repr."""
    if not dataclasses.is_dataclass(impl):
        return {}
    return {f.name: repr(getattr(impl, f.name))[:200] for f in dataclasses.fields(impl)}


def node(name, d) -> dict:
    out = {"name": place(name), "reads": [place(p) for p in d.reads],
           "owns": [place(p) for p in d.owns], "levels": []}
    while True:
        out["levels"].append(type(d).__name__)
        if hasattr(d, "implementation"):
            out["body"] = body(d.implementation)
            out["settings"] = settings(d.implementation)
        if hasattr(d, "driver"):
            out["driver"] = type(d.driver).__name__
            out["data"] = [[place(p) for p in slot] for slot in d.data]
            out["reports"] = [[place(p) for p in slot] for slot in d.reports]
        inner = next((getattr(d, f) for f in ("node", "function", "condition")
                      if dataclasses.is_dataclass(d) and f in {x.name for x in dataclasses.fields(d)}),
                     None)
        if inner is None:
            break
        d = inner
    if hasattr(d, "relations"):
        out["relations"] = [[None if r.lhs is None else place(r.lhs),
                             None if r.rhs is None else place(r.rhs), r.op.__name__]
                            for r in d.relations]
        out["objectives"] = [place(p) for p in d.objectives]
        out["unknowns"] = [place(p) for p in d.unknowns]
        out["parameters"] = [place(p) for p in getattr(d, "parameters", ())]
    return out


def fingerprint(graph) -> dict:
    nodes = [node(n, d) for n, d in graph.definitions.items()]
    nodes.sort(key=lambda x: json.dumps(x["owns"]))
    return {"nodes": nodes,
            "within": sorted([place(i), place(o)] for i, o in getattr(graph, "within", {}).items())}


def cottax_version() -> str:
    import cottax

    root = Path(cottax.__file__).resolve().parents[2]
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "src"],
                           capture_output=True, text=True).stdout.strip()
    return f"{root} {head}{' +dirty' if dirty else ''}"


def write(out: Path) -> None:
    from functional_process import configurations
    from functional_process.cottax.input.indat import GRAPH, graph_for

    out.mkdir(parents=True, exist_ok=True)
    graphs = {"REFERENCE_MACHINE": GRAPH}
    graphs |= {name: graph_for(configurations.load(name).machine) for name in configurations.NAMES}
    version = cottax_version()
    for name, graph in graphs.items():
        data = {"cottax": version, "configuration": name} | fingerprint(graph)
        (out / f"{name}.json").write_text(json.dumps(data, indent=1) + "\n")
        print(f"{name}: {len(data['nodes'])} nodes, {len(data['within'])} nesting pairs")


IGNORED = {"name", "levels", "settings"}
"""Not compared: a node's name and wrapper levels change with the surface, and a setting
on a declaration is a read now, which the reads show."""


def compare(old: Path, new: Path) -> int:
    """Per configuration, the nodes whose owns are only in one side, and the fields that
    differ where both have a node. Names are shown, never compared."""
    differences = 0
    for a in sorted(old.glob("*.json")):
        b = new / a.name
        if not b.exists():
            print(f"{a.stem}: missing in {new}")
            differences += 1
            continue
        left, right = (json.loads(p.read_text()) for p in (a, b))
        by = lambda doc: {json.dumps(n["owns"]): n for n in doc["nodes"]}  # noqa: E731
        lnodes, rnodes = by(left), by(right)
        for key in lnodes.keys() - rnodes.keys():
            print(f"{a.stem}: only old  {lnodes[key]['name']} owns {key}")
        for key in rnodes.keys() - lnodes.keys():
            print(f"{a.stem}: only new  {rnodes[key]['name']} owns {key}")
        for key in lnodes.keys() & rnodes.keys():
            for field in sorted((lnodes[key].keys() | rnodes[key].keys()) - IGNORED):
                if field == "body" and not lnodes[key].get("body", "").startswith(
                        "functional_process.models"):
                    continue            # a body written in a declaration class is renamed
                if lnodes[key].get(field) != rnodes[key].get(field):
                    print(f"{a.stem}: {lnodes[key]['name']} {field}: "
                          f"{lnodes[key].get(field)} -> {rnodes[key].get(field)}")
                    differences += 1
        differences += len(lnodes.keys() ^ rnodes.keys())
        if left["within"] != right["within"]:
            print(f"{a.stem}: nesting differs")
            differences += 1
    print(f"{differences} differences")
    return differences


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "graphs")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("OLD", "NEW"))
    args = parser.parse_args()
    raise SystemExit(bool(compare(*args.compare)) if args.compare else write(args.out))
