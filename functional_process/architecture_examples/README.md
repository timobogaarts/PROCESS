# Architecture examples

One folder per solution architecture. Each notebook says what the architecture is,
builds it as a short list of operations on the graph of PROCESS's models (printed, so
the recipe is visible), shows the run order it produces as a DSM, and runs it on the
Helias stellarator (`tests/regression/input_files/stellarator_helias.IN.DAT`). They are
written for a PROCESS user; the cottax terms they use are the ones the cottax examples
introduce (graph, node, cut, fixed point, requirement, problem, driver, nest, absorb).

MDF, IDF and SAND share the first two steps -- cut the feedback loops with a scheme,
state the file's problem -- and differ only in the last, which is one operation:

| folder | architecture | the recipe |
|---|---|---|
| [`mda_gauss_seidel/`](mda_gauss_seidel/mda_gauss_seidel.ipynb) | the analysis alone, converged Gauss-Seidel style as PROCESS's idempotence loop does | cut the feedback loops (`GaussSeidel`), iterate; the three schemes compared |
| [`mdf/`](mdf/mdf.ipynb) | MDF, PROCESS's own architecture | cut, state, `MDF()`: every problem nested inside the optimiser |
| [`idf/`](idf/idf.ipynb) | IDF | cut, state, `IDF()`: the coupling absorbed into the optimiser, the models' own solves nested in it |
| [`sand/`](sand/sand.ipynb) | SAND | cut, state, `SAND()`: every problem on the optimiser's cycle absorbed into it |
| [`two_opt_driver/`](two_opt_driver/two_opt_driver.ipynb) | two optimisers in sequence | cut, state two problems, absorb per loop; the graph decides whether the split is legal |

The notebooks call the library (`mda.SCHEME` and `mda.cut_graph`,
`sand.condition_nodes` / `requirement_nodes` / `problem_graph`, `sand.assemble`,
`idf.idf_graph`, `sand.sand_schedule`, `evaluate`'s seeding) and check that the
spelled-out recipe builds the same graph the library does.
`session.open_session(path)` is the one-line form of each.

## Running them

The env has to import `process`, `cottax` and `functional_process` together
(`../../CLAUDE.md`, "The environment"). A notebook uses whichever `cottax` already
imports and falls back to the sibling `jaxgraph/src` only when none does; it prints
which one answered, and `notebook_tools` hands the kernel the same one. Open them in Jupyter or VS
Code, or run headlessly:

```bash
cd PROCESS
PYTHONPATH=~/projects/jaxgraph/src:. $PY -m functional_process.architecture_examples.notebook_tools \
    functional_process/architecture_examples/mdf/mdf.ipynb        # in place, outputs kept
```

Each notebook ends in a `RESULT` dict; `functional_process/tests/examples/` runs every
notebook and checks that dict (`tier4`, about three minutes in all):

```bash
$PY -m pytest functional_process/tests/examples -m tier4
```

A run writes the DSM page (`dsm_<case>.html`) next to its notebook. The pages are
self-contained and embedded in the notebooks, so a notebook with outputs reads without
being re-run.
