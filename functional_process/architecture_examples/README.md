# Architecture examples

One folder per solution architecture. Each notebook says what the architecture is,
builds it as a short list of operations on the graph of PROCESS's models (printed, so
the recipe is visible), shows the run order it produces as a DSM, and runs it on the
Helias stellarator (`tests/regression/input_files/stellarator_helias.IN.DAT`). They are
written for a PROCESS user; the cottax terms they use are the ones the cottax examples
introduce (graph, node, cut, fixed point, problem, driver, nest, combine).

| folder | architecture | the recipe |
|---|---|---|
| [`mda_gauss_seidel/`](mda_gauss_seidel/mda_gauss_seidel.ipynb) | the analysis alone, converged Gauss-Seidel style as PROCESS's idempotence loop does | cut the feedback loops (`recipes.gauss_seidel`), nest the models' own solves, iterate |
| [`mdf/`](mdf/mdf.ipynb) | MDF, PROCESS's own architecture | cut, insert the optimisation problem, nest everything inside it |
| [`idf/`](idf/idf.ipynb) | IDF | cut, insert, residualise and combine the coupling copies, nest the models' own solves |
| [`sand/`](sand/sand.ipynb) | SAND | cut, insert, residualise and combine every problem |
| [`two_opt_driver/`](two_opt_driver/two_opt_driver.ipynb) | two optimisers in sequence | cut, insert two problems, residualise, combine per loop; the graph decides whether the split is legal. Full experiment in [`two_driver_report.md`](two_opt_driver/two_driver_report.md) |
| [`paper/`](paper/README.md) | -- | where the cases the paper shows go |

The notebooks call the library (`mda.cut_ops`, `recipes`, `sand.optimise_graph`,
`sand_harness.assemble`, `mdf.assemble`, `sand.sand_schedule`, `run_cold_matrix`'s
seeding) and check that the spelled-out recipe builds the same graph the library does.

## Running them

The env has to import `process`, `cottax` and `functional_process` together
(`../../CLAUDE.md`, "The environment"). The notebooks put the repo and, if it is laid out
as `two_opt_driver/scripts/README.md` describes, the sibling `jaxgraph/src` on
`sys.path` themselves and print which `cottax` answered. Open them in Jupyter or VS
Code, or run headlessly:

```bash
cd PROCESS
PYTHONPATH=~/projects/jaxgraph/src:. $PY -m functional_process.architecture_examples.notebook_tools \
    functional_process/architecture_examples/mdf/mdf.ipynb        # in place, outputs kept
```

Each notebook ends in a `RESULT` dict; `functional_process/tests/examples/` runs every
notebook and checks that dict (`tier4`, about two minutes in all):

```bash
$PY -m pytest functional_process/tests/examples -m tier4
```

A run writes the DSM page (`dsm_<case>.html`) next to its notebook. The pages are
self-contained and embedded in the notebooks, so a notebook with outputs reads without
being re-run.
