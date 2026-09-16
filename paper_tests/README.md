# paper_tests

Generators for the PROCESS section of *Managing complexity for integrated design in
fusion using a graph-based functional framework* (`~/graph_paper`). Each script measures
one thing over the seven reference input files and writes it to `out/` as a `.csv`
(numbers), a `.tex` fragment (a `tabular`, for `\input`) and, where the raw rows are
worth keeping, a `.json`. Every fragment's first line says which script, tree, cottax
commit and machine produced it.

```bash
PY=~/miniconda3/envs/process_port/bin/python      # see ../CLAUDE.md for the env
export JAX_PLATFORMS=cpu
$PY paper_tests/graph_census.py        # ~1 min   structure: nodes, SCCs, cuts, body depth
$PY paper_tests/timings.py             # instant  renders the three reference matrices
$PY paper_tests/mda_convergence.py     # ~5 min   Picard steps per block under each cut
$PY paper_tests/architectures.py       # ~1 h     MDF/SAND x cut x optimiser, warm and cold
```

If `~/jaxgraph`'s working tree is mid-edit, pin cottax with
`PYTHONPATH=<a worktree of its committed HEAD>/src` (`../CLAUDE.md`, "The port tracks a
cottax that moves").

## What each table says

**`graph_census`** -- the raw graph per configuration (nodes, boundary inputs, the cyclic
components and their sizes, the problems the models declare themselves), then per cut
recipe the copies minted and the deepest coupled-block body. The body depth is the
structural parallelism number: a Jacobi body is one layer (every node of the component
evaluates from the previous iterate), a Gauss-Seidel body is as deep as its sweep.

**`timings`** -- one row per configuration and arm, under VMCON and SLSQP: trace+lower,
compile, cold total, SQP iterations, warm wall, model ms/call. Read straight off
`functional_process/cottax/reference_*_matrix.txt`; regenerate those first
(`functional_process/_audit/performance.md`).

**`mda_convergence`** -- the inner analysis alone, from one cold state
(`sand_harness.cold_state`: the graph after one pass in call order, which is where
PROCESS starts too): Picard steps per coupled block, their sum, the sum weighted by body
depth (sequential node evaluations), and the warm wall of one converged MDA.

**`architectures`** -- MDF and SAND assembled on each cut and answered by each optimiser:
iterations, verdict, objective, design entries (a Jacobi SAND carries whole profiles),
cold wall and warm wall. One `tabular` per optimiser.

## The cuts

| name | what is cut | where |
|---|---|---|
| `hand` | nine variables measured so that one Picard iterate is one PROCESS pass | `functional_process/cottax/mda.py` (`CUTS`) |
| `jacobi` | every coupling variable of every component | `functional_process/cottax/recipes.py` |
| `gauss_seidel` | the backward reads in the graph's binding order (PROCESS's call order) | same |
| `gauss_seidel_minimal` | the backward reads in the order that cuts fewest variables (exact, subset DP) | same |

`recipes.py`'s docstring is the definition; `functional_process/tests/test_recipes.py`
pins the census on the stellarator and that every recipe reaches the hand cut's fixed
point and optimum.
