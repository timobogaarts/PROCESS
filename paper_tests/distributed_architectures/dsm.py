"""The XDSM and N2 of a case, as cottax's own interactive page (`render_html`), the solves
ringed as the executable graph has them: `out/<case>.html`. Built and driven, not solved;
the ALC case solves one MDA, for the reference magnitudes its consistency is scaled by.

    python paper_tests/distributed_architectures/dsm.py [sequential] [alc]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cottax.consumers.visualization import render_html

import partition as C

OUT = Path(__file__).parent / "out"


def sequential_graph(configuration):
    import sequential as S
    return S.driven(configuration, C.partition(configuration), {}, {})[1]


def alc_graph(configuration):
    import alc as A
    d = C.disciplines(configuration)
    _, env = C.mda_env(configuration)
    return A.driven(configuration, d, A.consistency(d, env), env, {})[1]


CASES = {"sequential": sequential_graph, "alc": alc_graph}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cases", nargs="*", default=list(CASES))
    args = ap.parse_args(argv)
    OUT.mkdir(exist_ok=True)
    configuration = C.load(C.CONFIGURATION)
    for case in args.cases:
        graph = CASES[case](configuration)
        render_html(graph, views=("xdsm", "n2"), blocks=C.study.proved(graph), write=True,
                    file_name=f"{C.CONFIGURATION}_{case}", outdir=str(OUT))
        print("wrote", OUT / f"{C.CONFIGURATION}_{case}.html")


if __name__ == "__main__":
    main()
