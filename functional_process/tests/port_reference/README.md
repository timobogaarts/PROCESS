# Port reference: the old world, fixed before porting

The port runs against cottax at **jaxgraph `ea3abbe`** (2026-09-25 13:19, the commit before
`5837a1e` enforced abstract-or-final on methods, which the port's drivers break). At that
commit every name `test_cottax_surface.py` collects from `.py` files resolves; the two it
reports missing (`interfaces.bare_conditions`, `mdao_architectures.coupled`) are used only
in the `architecture_examples/` notebooks, which were already stale.

To reproduce the old world:

    git -C ~/jaxgraph worktree add --detach /tmp/jaxgraph_ref ea3abbe
    cd ~/PROCESS && PYTHONPATH=/tmp/jaxgraph_ref/src JAX_PLATFORMS=cpu $PY -m pytest ...

`graphs/` holds `fingerprint.py`'s record of every configuration's model graph
(`REFERENCE_MACHINE` and `configurations.NAMES`): per node what it reads and owns in
order, its levels, its body (the physics function, or the declaration class that holds a
body), its settings, and for a condition node its relations. The Phase 1 gate is
`fingerprint.py --compare graphs <new>` reporting no difference except names. The
numeric gates are the unit tiers and `tests/reference_architectures.txt`.

`timings/` holds the MDA's timing per configuration on this machine (`timings.py`: the
paper's own `iteration.mda_row`, compile once then the median of 5 warm runs, one process
per configuration, ~80 s in all). The later phases compare against it on the same
machine: an evaluation or compile time that moves beyond noise is framework overhead the
port introduced.
