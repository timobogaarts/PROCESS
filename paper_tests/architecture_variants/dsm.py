"""The XDSM and N2 of variants, as cottax's own interactive page
(`cottax.consumers.visualization.render_html`), the solves ringed as the executable
graph has them: `out/dsm/<machine>_<label>.html`. Built, not solved; nothing compiled.

    $PY paper_tests/architecture_variants/dsm.py --configurations stellarator_helias --variants FO-O FO-L
"""

from __future__ import annotations

import common

from cottax.consumers.visualization import render_html  # noqa: E402
from functional_process import configurations  # noqa: E402
from functional_process.cottax import study  # noqa: E402

import variants as V  # noqa: E402


def main():
    args = common.arguments(__doc__)
    out = common.OUT / "dsm"
    out.mkdir(parents=True, exist_ok=True)
    for name in args.configurations:
        configuration = configurations.load(name)
        for v in V.variants(configuration):
            if args.variants is not None and v.label not in args.variants:
                continue
            graph = V.variant_plan(configuration, v, {}, []).graph
            render_html(graph, views=("xdsm", "n2"), blocks=study.proved(graph), write=True,
                        file_name=f"{name}_{v.label}", outdir=str(out))
            print(f"wrote {common.shown(out / f'{name}_{v.label}.html')}", flush=True)


if __name__ == "__main__":
    main()
