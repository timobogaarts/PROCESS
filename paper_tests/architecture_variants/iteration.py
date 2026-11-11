"""The cost of one optimiser iteration per variant, on the CPU, warm: one evaluation and
one Jacobian of the problem the top holds, at the file's start, each compiled once and
then timed as the median of `--repeats` calls -- `../architectures/iteration.py`'s own
`optimiser_row`, called on the variant, so the columns are the paper table's.

Not measured here: the eager columns (`eager_walk_ms`, `eager_ms`), left out of this
sweep, and the finite-difference ones (`--epsfcn` there). As there, the top is assigned
VMCON, which is never run: an evaluation and a Jacobian do not involve the driver, and
the assembly is then the one the Snellius rows were measured on.

One process per variant (`run_all.sh`), so `compile_s` is a cold compile every time;
on Snellius it was one process per machine, all arms in it.

    $PY paper_tests/architecture_variants/iteration.py --configurations NAME --variants LABEL
"""

from __future__ import annotations

import types

import common
from common import bench

from functional_process import configurations  # noqa: E402
from functional_process.cottax import study  # noqa: E402
from functional_process.cottax.architectures.drivers import VmconDriver  # noqa: E402

import variants as V  # noqa: E402

arch_iteration = common.load("architectures_iteration", common.ARCH / "iteration.py")
arch_iteration.eager_columns = lambda assembled, args: {}     # not in this sweep

FOLDER = f"iteration_{common.SCHEME}"


def main():
    args = common.arguments(__doc__)
    measured = types.SimpleNamespace(repeats=args.repeats, epsfcn=None)
    for name in args.configurations:
        configuration = configurations.load(name)
        todo = [v for v in V.variants(configuration) if args.variants is None or v.label in args.variants]
        todo = [v for v in todo if args.force or not common.done(FOLDER, name, v.label)]
        if not todo:
            continue
        cold = study.cold_state(configuration)
        for v in todo:
            row = {"configuration": name, "label": v.label, "preset": v.preset}
            try:
                assembled = V.assemble_variant(configuration, v, cold, optimiser=VmconDriver)
                row.update({"refused": "", **arch_iteration.optimiser_row(assembled, measured)})
            except Exception as e:  # noqa: BLE001 -- a refusal is a result here
                row.update({"refused": f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"})
            print(row, flush=True)
            common.upsert("iteration.py", FOLDER, row)


if __name__ == "__main__":
    main()
