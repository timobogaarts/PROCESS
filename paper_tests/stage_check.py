"""Stage check for OUU: which build quantities actually vary per belief sample.

Parses `decision_kinds.md` and `output_kinds.md`'s hand classification of every
boundary input and every claimed rule-closed build output, builds the stellarator
graph, and uses `reach` to find every claimed build output whose producer depends
on a belief -- a non-anticipativity violation.

    $PY paper_tests/stage_check.py
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from common import OUT, write_json

from functional_process.cottax import indat, mda

HERE = Path(__file__).resolve().parent
SECTION_KIND = {"3.1": "belief", "3.2": "build", "3.3": "operating", "3.4": "numerics", "3.5": "derived"}

# ---------------------------------------------------------------- markdown parsing


def parse_tables(text: str):
    """Yield `(section, header, rows)` for every markdown table, `section` the nearest
    preceding `##`/`###` heading's leading number (`'3.1'`, `'1a'`, ...), each row a
    dict of column name -> cell text."""
    lines = text.splitlines()
    section = None
    i = 0
    while i < len(lines):
        line = lines[i]
        heading = re.match(r"^#{2,3}\s+(.*)", line)
        if heading:
            m = re.match(r"^(\d+(?:\.\d+)?[a-z]?)\.?\s", heading.group(1))
            if m:
                section = m.group(1)
        if line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:-]+\|", lines[i + 1]):
            header = [c.strip().lower() for c in line.strip().strip("|").split("|")]
            i += 2
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if len(cells) == len(header):
                    rows.append(dict(zip(header, cells)))
                i += 1
            yield section, header, rows
            continue
        i += 1


def backticks(cell: str) -> list[str]:
    return re.findall(r"`([^`]+)`", cell)


def unbold(cell: str) -> str:
    return cell.strip().strip("*").strip().lower()


def parse_decision_kinds(text: str):
    """`kind_of_input`: spelling -> 'belief'/'build'/'operating'/'numerics'/'derived',
    plus the §3.6 owned-place claims (spelling -> [(file, section, '§3.6')]), and a
    per-section parse count."""
    kind_of_input: dict[str, str] = {}
    section_36: dict[str, list] = {}
    counts: dict[str, int] = {}
    for section, header, rows in parse_tables(text):
        if section == "1" and "place" in header and "kind" in header:
            for r in rows:
                for p in backticks(r["place"]):
                    kind_of_input[p] = unbold(r["kind"])
            counts["§1 (ixc)"] = counts.get("§1 (ixc)", 0) + len(rows)
        elif section in SECTION_KIND and "path" in header:
            for r in rows:
                for p in backticks(r["path"]):
                    kind_of_input[p] = SECTION_KIND[section]
            counts[f"§{section}"] = counts.get(f"§{section}", 0) + len(rows)
        elif section == "3.6" and "owned place" in header:
            for r in rows:
                for p in backticks(r["owned place"]):
                    section_36.setdefault(p, []).append(("decision_kinds.md", "§3.6", "§3.6"))
            counts["§3.6"] = counts.get("§3.6", 0) + len(rows)
    return kind_of_input, section_36, counts


def parse_output_kinds(text: str):
    """`claimed_build_outputs` from §1a-1d: spelling -> [(file, section, kind)]."""
    claims: dict[str, list] = {}
    counts: dict[str, int] = {}
    for section, header, rows in parse_tables(text):
        if section and section.startswith("1") and section != "1" and "place" in header and "kind" in header:
            for r in rows:
                kind = unbold(r["kind"])
                if kind not in ("definition", "sizing choice"):
                    continue
                for p in backticks(r["place"]):
                    claims.setdefault(p, []).append(("output_kinds.md", f"§{section}", kind))
            counts[f"§{section}"] = counts.get(f"§{section}", 0) + len(rows)
    return claims, counts


# ---------------------------------------------------------------- the check


def stage_of(node, second_stage: dict[str, set]) -> str:
    hits = [name for name, reached in second_stage.items() if node in reached]
    return "+".join(hits) if hits else "first"


def analyse(g, label: str, beliefs: tuple, operating: tuple) -> dict:
    second_stage = set(g.reach(beliefs))
    recourse = set(g.reach(operating))
    stages = {"first": {"nodes": 0, "owned": 0}, "second": {"nodes": 0, "owned": 0}, "recourse": {"nodes": 0, "owned": 0}}
    for n in g.nodes:
        if n in second_stage:
            key = "second"
        elif n in recourse:
            key = "recourse"
        else:
            key = "first"
        stages[key]["nodes"] += 1
        stages[key]["owned"] += len(g[n].owns)
    total_nodes = len(g.nodes)
    # A build quantity must be the same in every sample, so it may vary with neither
    # the beliefs nor the operating point the sample settles at: both sets are leaves
    # a claimed build output can be responsible to.
    reach_of_leaf = {leaf: set(g.reach([leaf])) for leaf in (*beliefs, *operating)}
    return {
        "label": label,
        "n_leaves": len(beliefs),
        "leaves": sorted(v.spelling for v in beliefs),
        "stages": stages,
        "first_stage_fraction_nodes": stages["first"]["nodes"] / total_nodes if total_nodes else 0.0,
        "second_stage": second_stage | recourse,
        "reach_of_leaf": reach_of_leaf,
    }


def violations_for(result: dict, claims: dict[str, list], spelling_of_var: dict, owners) -> tuple[list, list]:
    """Every claimed build output whose owner is in the second stage (with its
    responsible belief leaves), and every claimed output confirmed first-stage."""
    second_stage = result["second_stage"]
    reach_of_leaf = result["reach_of_leaf"]
    hits, clean = [], []
    for spelling, provenance in sorted(claims.items()):
        var = spelling_of_var.get(spelling)
        if var is None or var not in owners:
            continue  # not an owned place of this graph -- shorthand cell, or absent in this config
        owner = owners[var]
        source_file, section, kind = provenance[0]
        if owner in second_stage:
            responsible = sorted(
                (leaf.spelling for leaf, reached in reach_of_leaf.items() if owner in reached),
            )
            hits.append({
                "place": spelling, "kind": kind, "source": f"{source_file} {section}",
                "producing_node": owner.spelling, "n_responsible": len(responsible), "responsible": responsible,
            })
        else:
            clean.append({
                "place": spelling, "kind": kind, "source": f"{source_file} {section}", "producing_node": owner.spelling,
            })
    hits.sort(key=lambda r: -r["n_responsible"])
    return hits, clean


# ---------------------------------------------------------------- report


def truncate(items, n=8):
    if len(items) <= n:
        return ", ".join(items)
    return ", ".join(items[:n]) + f" +{len(items) - n} more"


def main():
    began = time.perf_counter()
    decision_text = (HERE / "decision_kinds.md").read_text()
    output_text = (HERE / "output_kinds.md").read_text()

    kind_of_input, section_36, dcounts = parse_decision_kinds(decision_text)
    output_claims, ocounts = parse_output_kinds(output_text)
    claimed_build_outputs: dict[str, list] = {}
    for spelling, prov in section_36.items():
        claimed_build_outputs.setdefault(spelling, []).extend(prov)
    for spelling, prov in output_claims.items():
        claimed_build_outputs.setdefault(spelling, []).extend(prov)

    print("parsed per section (decision_kinds.md):")
    for section, n in dcounts.items():
        print(f"  {section}: {n} rows")
    print("parsed per section (output_kinds.md):")
    for section, n in ocounts.items():
        print(f"  {section}: {n} rows")
    print(f"kind_of_input: {len(kind_of_input)} classified boundary spellings")
    print(f"claimed_build_outputs: {len(claimed_build_outputs)} spellings ({len(section_36)} from §3.6, {len(output_claims)} from output_kinds §1)")

    print(f"building the graph ({time.perf_counter() - began:.0f}s so far)...")
    g = mda.driven_graph(indat.GRAPH)
    print(f"graph: {len(g.nodes)} nodes, {len(g.boundary_inputs)} boundary inputs ({time.perf_counter() - began:.0f}s)")

    spelling_of_var = {v.spelling: v for v in g.variables}
    boundary_spellings = {v.spelling for v in g.boundary_inputs}
    classified_spellings = set(kind_of_input)
    unclassified = sorted(boundary_spellings - classified_spellings)
    extra_classified = sorted(classified_spellings - boundary_spellings)
    print(f"coverage: {len(unclassified)} boundary inputs unclassified, {len(extra_classified)} classified spellings not in the graph")

    belief_all = tuple(v for v in g.boundary_inputs if kind_of_input.get(v.spelling) == "belief")
    operating = tuple(v for v in g.boundary_inputs if kind_of_input.get(v.spelling) == "operating")

    import ouu  # noqa: PLC0415 -- paper_tests-local, imported after common's sys.path setup

    sampled_spellings = {i.path for i in ouu.belief_table("new") if i.path != "dummy"}
    belief_sampled = tuple(v for v in g.boundary_inputs if v.spelling in sampled_spellings)
    sampled_misclassified = {
        v.spelling: kind_of_input.get(v.spelling, "UNCLASSIFIED")
        for v in belief_sampled if kind_of_input.get(v.spelling) != "belief"
    }
    unmatched_sampled = sorted(sampled_spellings - {v.spelling for v in g.boundary_inputs})

    print(f"belief sets: all={len(belief_all)}, sampled={len(belief_sampled)} (of {len(sampled_spellings)} sampled paths, {len(unmatched_sampled)} not in graph)")
    print(f"operating set: {len(operating)}")
    if sampled_misclassified:
        print(f"sampled paths not classified 'belief': {sampled_misclassified}")

    # `all` is the graph's own question -- may a build quantity depend on any belief or
    # any operating leaf -- and the OUU sets ask what actually varies per sample in
    # `ouu.py`: the sampled beliefs and the recourse, the closing variables of
    # `ouu.PAIRINGS["two"]`; every other operating leaf is held at its nominal there.
    recourse_spellings = set(ouu.PAIRINGS["two"].values())
    recourse = tuple(v for v in g.boundary_inputs if v.spelling in recourse_spellings)
    print(f"recourse (per-sample operating point): {[v.spelling for v in recourse]}")
    results = {}
    belief_build = tuple(v for v in belief_sampled if v.spelling not in ouu.BUILD_LEAVES)
    for label, beliefs, varying in (("all", belief_all, operating), ("sampled", belief_sampled, recourse),
                                    ("build", belief_build, recourse)):
        results[label] = analyse(g, label, beliefs, varying)

    owners = g.owners
    summary_rows = []
    violation_tables = {}
    clean_tables = {}
    for label, result in results.items():
        hits, clean = violations_for(result, claimed_build_outputs, spelling_of_var, owners)
        violation_tables[label] = hits
        clean_tables[label] = clean
        stages = result["stages"]
        summary_rows.append({
            "belief_set": label, "n_leaves": result["n_leaves"],
            "nodes_first": stages["first"]["nodes"], "nodes_second": stages["second"]["nodes"],
            "nodes_recourse": stages["recourse"]["nodes"],
            "owned_first": stages["first"]["owned"], "owned_second": stages["second"]["owned"],
            "owned_recourse": stages["recourse"]["owned"],
            "first_stage_fraction_nodes": result["first_stage_fraction_nodes"],
            "n_violations": len(hits), "n_confirmed_first_stage": len(clean),
        })

    print("\nsummary:")
    for row in summary_rows:
        print(f"  {row['belief_set']:>8}: {row['n_leaves']:3d} leaves | nodes 1st/2nd/recourse "
              f"{row['nodes_first']}/{row['nodes_second']}/{row['nodes_recourse']} "
              f"| owned 1st/2nd/recourse {row['owned_first']}/{row['owned_second']}/{row['owned_recourse']} "
              f"| 1st-stage frac {row['first_stage_fraction_nodes']:.2f} "
              f"| violations {row['n_violations']} | confirmed-first {row['n_confirmed_first_stage']}")

    for label in ("sampled", "build"):
        print(f"\nviolations, belief set {label!r} ({len(violation_tables[label])}):")
        for v in violation_tables[label]:
            print(f"  {v['place']:45s} {v['kind']:14s} {v['source']:25s} <- {v['producing_node']:35s} "
                  f"({v['n_responsible']} leaves: {truncate(v['responsible'])})")

    if unclassified:
        print(f"\nunclassified boundary inputs ({len(unclassified)}):")
        for s in unclassified:
            print(f"  {s}")

    # ---------------------------------------------------------- write out

    payload = {
        "parsed_counts": {"decision_kinds.md": dcounts, "output_kinds.md": ocounts},
        "n_classified_inputs": len(kind_of_input),
        "n_claimed_build_outputs": len(claimed_build_outputs),
        "unclassified_boundary_inputs": unclassified,
        "classified_but_not_in_graph": extra_classified,
        "sampled_misclassified": sampled_misclassified,
        "unmatched_sampled_paths": unmatched_sampled,
        "summary": summary_rows,
        "violations": {label: rows for label, rows in violation_tables.items()},
        "confirmed_first_stage": {label: rows for label, rows in clean_tables.items()},
    }
    write_json("stage_check", payload)

    md = ["# Stage check: non-anticipativity of claimed build outputs\n"]
    md.append("## Summary\n")
    md.append("| belief set | leaves | nodes 1st | nodes 2nd | owned 1st | owned 2nd | 1st-stage frac | violations |")
    md.append("|---|---|---|---|---|---|---|---|")
    for row in summary_rows:
        md.append(f"| {row['belief_set']} | {row['n_leaves']} | {row['nodes_first']} | {row['nodes_second']} | "
                   f"{row['owned_first']} | {row['owned_second']} | {row['first_stage_fraction_nodes']:.2f} | {row['n_violations']} |")
    for label in ("all", "sampled", "build"):
        md.append(f"\n## Violations -- belief set '{label}' ({len(violation_tables[label])})\n")
        md.append("| place | kind | source | producing node | # leaves | leaves |")
        md.append("|---|---|---|---|---|---|")
        for v in violation_tables[label]:
            md.append(f"| `{v['place']}` | {v['kind']} | {v['source']} | `{v['producing_node']}` | "
                       f"{v['n_responsible']} | {truncate(v['responsible'])} |")
        md.append(f"\n## Confirmed first-stage -- belief set '{label}' ({len(clean_tables[label])})\n")
        md.append("| place | kind | source | producing node |")
        md.append("|---|---|---|---|")
        for v in clean_tables[label]:
            md.append(f"| `{v['place']}` | {v['kind']} | {v['source']} | `{v['producing_node']}` |")
    if unclassified:
        md.append(f"\n## Unclassified boundary inputs ({len(unclassified)})\n")
        for s in unclassified:
            md.append(f"- `{s}`")
    (OUT / "stage_check.md").write_text("\n".join(md) + "\n")

    print(f"\nwrote {OUT / 'stage_check.json'} and {OUT / 'stage_check.md'} ({time.perf_counter() - began:.0f}s total)")


if __name__ == "__main__":
    main()
