import os
HERE = os.path.dirname(os.path.abspath(__file__))
import html, os, re
T = os.path.dirname(HERE)   # two_opt_driver/: report .md/.html and the DSM live there
md = open(f"{T}/two_driver_report.md").read()
dsm = open(f"{T}/dsm_two_drivers_executing.html").read()
def inline(s):
    s = html.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*])\*(?!\*)(.+?)\*(?!\*)", r"<em>\1</em>", s)
    return s
out, i, lines = [], 0, md.split("\n")
while i < len(lines):
    l = lines[i]
    if l.startswith("|"):
        rows = []
        while i < len(lines) and lines[i].startswith("|"):
            rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")]); i += 1
        rows = [r for r in rows if not all(re.fullmatch(r"-+", c) for c in r)]
        out.append("<table><tr>" + "".join(f"<th>{inline(c)}</th>" for c in rows[0]) + "</tr>" + "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in rows[1:]) + "</table>"); continue
    m = re.match(r"^(#+) (.*)", l)
    if m:
        out.append(f"<h{len(m[1])}>{inline(m[2])}</h{len(m[1])}>")
        if m[2].startswith("Structure"):
            out.append('<figure><iframe srcdoc="' + html.escape(dsm, quote=True) + '" style="width:100%;height:820px;border:1px solid #ccc;border-radius:6px;background:#fff"></iframe><figcaption>DSM of the driven graph that executed: <code>Blocking.scc(assigned)</code> in SCC/run order. Two optimiser blocks on the diagonal (<code>sand_magnet</code> ~step 25, <code>sand_plasma</code> ~step 47); everything from the first to the second lies below the diagonal. Hover a cell for the variables crossing it; click a block to fold/unfold. Interactive copy: <code>dsm_two_drivers_executing.html</code>.</figcaption></figure>')
        i += 1; continue
    if re.match(r"^\s*(\d+\.|-) ", l):
        tag = "ol" if re.match(r"^\s*\d+\.", l) else "ul"; items = []
        while i < len(lines) and (re.match(r"^\s*(\d+\.|-) ", lines[i]) or (lines[i].startswith("   ") and items)):
            if re.match(r"^\s*(\d+\.|-) ", lines[i]): items.append(re.sub(r"^\s*(\d+\.|-) ", "", lines[i]))
            else: items[-1] += " " + lines[i].strip()
            i += 1
        out.append(f"<{tag}>" + "".join(f"<li>{inline(x)}</li>" for x in items) + f"</{tag}>"); continue
    if l.strip() == "": i += 1; continue
    para = []
    while i < len(lines) and lines[i].strip() and not lines[i].startswith(("|", "#")) and not re.match(r"^\s*(\d+\.|-) ", lines[i]):
        para.append(lines[i].strip()); i += 1
    out.append(f"<p>{inline(' '.join(para))}</p>")
page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Two sequential optimisers over the PROCESS graph</title>
<style>body{{max-width:1100px;margin:2rem auto;padding:0 16px;font:15px/1.5 system-ui,sans-serif;color:#1a1a1a;background:#fff}}
h1{{font-size:1.6rem}} h2{{margin-top:2rem;border-bottom:1px solid #ddd}} table{{border-collapse:collapse;margin:1rem 0;font-size:.92em}}
th,td{{border:1px solid #ccc;padding:4px 8px;text-align:left;vertical-align:top}} th{{background:#f4f4f4}} code{{background:#f3f3f3;padding:0 3px;border-radius:3px;font-size:.92em}}
figure{{margin:1.5rem 0}} figcaption{{font-size:.9em;color:#555;margin-top:.4rem}}
@media (prefers-color-scheme: dark){{body{{background:#161616;color:#e6e6e6}} th{{background:#262626}} th,td{{border-color:#444}} code{{background:#2a2a2a}} h2{{border-color:#444}} figcaption{{color:#aaa}}}}</style>
<body>{''.join(out)}</body></html>"""
open(f"{T}/two_driver_report.html", "w").write(page)
print("wrote", f"{T}/two_driver_report.html", len(page), "bytes")
