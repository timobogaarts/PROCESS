# functional_process

A pure-functional, cottax-shaped port of PROCESS's models. `process/` is never
modified; everything here is new JAX code written from it, validated against it.

Read `../CLAUDE.md` first for the `process_port` env (the one interpreter where
`process` and `cottax` import together) and the cottax vocabulary. `x64` must be on
before any array exists -- every entry point here does that itself.

## Layout

```
functional_process/
  models/                 the ported models: pure functions, one file per PROCESS model,
                          same tree as process/models/, plus constraints.py and
                          objectives.py (PROCESS's constraint equations and figures of
                          merit are models too). safe_math.py and switch_enums.py are
                          what they share.
  vocabulary/             PROCESS's enums, constants, input and iteration-variable
  _vendor/                tables, and coolprop_interface -- vendored so that models/
  data/                   imports without process. (Goes away when merged upstream.)

  configurations/         the regression input files, stated: one module per file with
                          the machine (every slot's occupant, no integer switch), its
                          own values, and its problem; defaults.py is what a
                          configuration does not state. Generated from the IN.DAT by
                          input/indat.py, and checked against it.

  cottax/                 the models as cottax nodes, and the architectures over them
    paths.py wraps.py stated.py queries.py   the declaration vocabulary: how a node
                          spells its variables, declares its reads, states a value
    models/               one node per ported function, thin; mirrors models/. Plus
                          <subsystem>/namespace.py (the subsystem's slots),
                          total_process.py (StellaratorProcess / TokamakProcess: one
                          slot per subsystem) and initialisation.py (the seed's writes)
    input/                importer.py (read an IN.DAT), indat.py (the converter: its
                          switches -> the machine, and configuration_from_indat),
                          native.py (a configuration's solve state, no DataStructure)
    architectures/        mda.py recipes.py (cutting the cycles), mdf.py idf.py sand.py
                          (the optimisation architectures, as graph ops), evaluate.py
                          (seeding and running a schedule), session.py (one input file,
                          every architecture), drivers.py host_cache.py (the algorithms)
    visualization/        grouping.py render_xdsm.py: the grouped DSM the notebooks draw

  tests/
    models/               one case per ported unit, mirroring models/; sample points
                          beside each case under _samples/
    architectures/        the drivers' own unit tests
    test_architectures.py every architecture on every configuration, cold, against
    reference_architectures.txt   this pin
    test_configurations.py  every configuration is still its input file, converted
    examples/             the notebooks run and their RESULT has not moved
    _harness/             the unit-case contracts (tiers, sampling, tolerances, FD)
    conftest.py           --fp-fuzz, --fp-gradients, --fp-write-pin

  architecture_examples/  one notebook per architecture: the recipe, its DSM, a solve
  deliberate_divergences.md   every place the port knowingly differs from PROCESS
```

## The two kinds of test

```bash
$PY -m pytest functional_process/tests -m "not tier4"      # unit cases, ~2 min
$PY -m pytest functional_process/tests/test_architectures.py   # the pin, ~10 min
$PY -m pytest functional_process/tests/examples            # the notebooks, ~3 min
```

**A unit case** is two files sharing a stem at the same relative path in two trees:
`density_limits.py` under `functional_process/` and `test_density_limits.py` under
`tests/`. The case declares the PROCESS reference, the port and its sample points, then
subclasses `Tier1Contract` (value and gradient against PROCESS, at every point) or
`Tier2Contract` (a solver: residual-based, no value agreement by construction) -- it
writes no test functions. Gradients are checked against PROCESS's own finite difference
with a per-point error bar from Richardson extrapolation; `--fp-gradients` turns that
on, `--fp-fuzz N` draws N random points per fuzzable case from the shared domain.
Copy `tests/models/stellarator/test_density_limits.py` to write one.

**The architecture pin** solves every checked-in configuration under MDA, MDF, IDF and
SAND from its own cold values and asserts each row's `status` and `objf` against
`tests/reference_architectures.txt`. After a deliberate change, regenerate it with
`--fp-write-pin` and read the diff. `test_configurations.py` guards the other end:
each `configurations/<name>.py` still equals `configuration_from_indat` of the
regression file it was made from (`input.indat.write_configuration` regenerates one).

## A configuration is three stated trees

`configurations.Configuration(machine, values, problem)`: the machine is a
`StellaratorProcess`/`TokamakProcess` with every slot's occupant written out (the models,
chosen directly -- PROCESS's integer switches exist only in the converter); `values` is
`{area: {field: value}}`, the configuration's own numbers over `defaults.VALUES`;
`problem` is the design variables, constraints, figure of merit, bounds and the static
switch values the constraint nodes are bound with. `native.state_of` layers the values
and applies `init.py`'s derivation rules; `native.reference_of` adds the problem.

## An architecture is a recipe

The models are nodes; `input.indat.graph_for(machine)` is their graph, with PROCESS's own
cycles in it. An architecture is a short list of cottax ops on that graph, then a
driver per problem:

| | ops | in |
|---|---|---|
| MDA | `FixedPointCut` per loop-carried variable, `Nest` the models' own solves, Picard/Newton on each | `architectures.mda.cut_graph`, `architectures.recipes` |
| MDF | the cut graph plus the constraint and objective nodes and one `Optimise`, the MDA nested inside it | `architectures.mdf.assemble` |
| IDF | as MDF, then `Residualise` and `Combine` the cut copies into the optimiser; the models' own solves stay nested | `architectures.idf.idf_graph` |
| SAND | `Residualise` every fixed point, `Combine` every problem into one | `architectures.sand.assemble` |

`architectures.session.open_session(name)` assembles each once and solves it from the configuration
(`.mda()`, `.mdf()`, `.idf()`, `.sand()`); the notebooks under `architecture_examples/`
spell the same recipes out step by step and check they build the same graph.

## Conventions that hold

- **Precision.** PROCESS is float64 throughout; so is the port. A diff that looks like a
  porting bug is usually `x64` off.
- **Switches.** A model never reads an integer switch, and neither does a solve: a
  configuration names its occupants. Which node a switch value selects is the
  converter's knowledge (`input/indat.py`), consulted once, when an IN.DAT is converted;
  a slot that only exists for some values is annotated with the union of its occupants
  and left without a default.
- **External calls** that JAX cannot trace (CoolProp) stay outside any differentiated
  block; `_vendor/fluid_properties.py` is the traced replacement where one exists.
- **Divergences** from PROCESS are deliberate, few, and each is listed in
  `deliberate_divergences.md` with what it changes and by how much.
- **Derivations live in commits.** Why a port reads what it reads, which branch was
  dead, what the reads-set turned out to be: in the commit that made it, not in a file
  beside it. Docstrings that cite `_audit/<record>.md` resolve through git history
  (`git log --diff-filter=D -- functional_process/_audit/`).
