"""Render `total_process.GRAPH` as a self-contained, interactive XDSM HTML page.

Run directly:

    ~/miniconda/envs/process_port/bin/python -m functional_process.render_xdsm

writes `functional_process/xdsm.html` (pan/zoom, hover a step or a coupling to
highlight its row/column; self-contained, opens in a plain browser). Re-run after
porting a new unit to `total_process.py` to see it join the diagram.
"""

from pathlib import Path

from cottax.visualization import render_xdsm_html# render_dsm_html

from functional_process.total_process import GRAPH

OUTDIR = Path(__file__).parent


def main():
    """Write `xdsm.html` next to this file and return its path."""
    from cottax import Blocking 
#    render_xdsm_canvas(Blocking.fused(GRAPH), file_name="xdsm_c", outdir=str(OUTDIR), write=True, collapse_names = True, blocks=True, collapse_models=True)    
    render_xdsm_html(Blocking.fused(GRAPH), file_name="xdsm", outdir=str(OUTDIR), write=True, collapse_names = True, blocks=True, collapse_models=True)
    #    render_dsm_html(Blocking.fused(GRAPH), file_name="dsm", outdir=str(OUTDIR), write=True)
    return OUTDIR / "xdsm.html"


if __name__ == "__main__":
    path = main()
    print(f"wrote {path}")
