# functional_process

A pure-functional, cottax-shaped port of PROCESS's models. `process/` is never
modified; everything here is new JAX code written from it, validated against it.

Read `../CLAUDE.md` first for the `process_port` env (the one interpreter where
`process` and `cottax` import together). `x64` must be on before any array exists --
every entry point here does that itself.

**The cottax surface.** Everything here is written over `cottax.interfaces` (how a
model, a statement and a run are written -- the declaration modules, and the `ops` /
`runtime` / `queries` facades re-exported from the package root) and
`cottax.mdao_architectures` (the recipes). Three things sit below those by nature and
are imported by their own names: a **driver** (`cottax.execution.driver` and the namings
in `cottax.execution.drivers.kinds`), a **name** (`cottax.pytree.path`,
`cottax.pytree.mint`) and a **picture** (`cottax.visualization`,
`cottax.visualization.sequencing`). `tests/test_cottax_surface.py` asserts that list
over the port's own imports -- every `.py` here and under `paper_tests/`, and every code
cell of every notebook -- and keeps `PRIVATE` empty: a name the interface lacks is a
request upstream, not a reach past it.

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
    paths.py stated.py    the declaration vocabulary: how a node spells its variables,
                          states a value
    problem.py study.py truth.py   a configuration's problem as a model; every
                          architecture on it, and its runs; the state a solve starts from
    models/               one node per ported function, thin; mirrors models/. Plus
                          <subsystem>/namespace.py (the subsystem's slots),
                          total_process.py (StellaratorProcess / TokamakProcess: one
                          slot per subsystem) and initialisation.py (the seed's writes)
    input/                importer.py (read an IN.DAT), indat.py (the converter: its
                          switches -> the machine, and configuration_from_indat),
                          native.py (a configuration's solve state, no DataStructure)
    architectures/        contracts.py drivers.py reading.py host_cache.py (the port's
                          drivers, as cottax contracts and their implementations)

  tests/
    models/               one case per ported unit, mirroring models/; sample points
                          beside each case under _samples/
    architectures/        the drivers' own unit tests
    test_architectures.py every architecture on every configuration, cold, against
    reference_architectures.txt   this pin
    test_configurations.py  every configuration is still its input file, converted
    _harness/             the unit-case contracts (tiers, sampling, tolerances, FD)
    conftest.py           --fp-fuzz, --fp-gradients, --fp-write-pin

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
and applies `init.py`'s derivation rules.

## An architecture is a recipe

The models are nodes; `input.indat.graph_for(machine)` is their graph, with PROCESS's own
cycles in it. An architecture is a short list of cottax ops on that graph, then a
driver per problem.

**Every driver here sits on one of cottax's two readings of a statement.** The seam
(`Conditions`) hands the two *sides* of each relation and subtracts nothing, so a driver
says which reading it wants by which level it is a final of: a `GapDriver` gets a `Gaps`
and `solve(gaps, data)`, where `gaps(*x)` is the objectives and one gap (`lhs - rhs`)
per relation -- the SQPs (`VmconDriver`, `SlsqpDriver`, `BoxedSlsqpDriver`) and the Newtons
(`SeededNewtonDriver`, `SafeguardedNewtonDriver`, `BracketedRootDriver`); an
`IterateDriver` gets a `NextValues` and `iterate(next_values, data)`, where a step *is*
the right side of each relation -- `PicardDriver` and `SweepDriver`. A Newton that
inverts the gaps sets `square = True`, and cottax counts the gaps against the unknowns
once per solve. Nothing in the port subtracts a side for itself.

Each active constraint is **one declaration** (`cottax/models/constraints.py`, cottax's
`ConstraintFunction`): a body owning `.constraints.c<id>` at `.Constraint<id>`, and
beside it the requirement that holds it against zero, `^require.Constraint<id>` -- `= 0`
for the first `n_equality` of the file's `icc`, `<= 0` for the rest. The requirement is
the declaration's own, so a constraint is stated where it is computed and an
architecture absorbs it into the optimiser its conditions reach. The figure of merit is
an ordinary node owning `.numerics.objf`. Neither carries a mint: a model's output is a
place in the port's own namespace, and `^` is for what a rewrite fabricates.

`study.py` is every architecture on a configuration: `problem.problem_model` (the
machine, each constraint and the merit, and the problem at `problem.TOP`) under
`cx.mda` / `cx.mdf` / `cx.idf` / `cx.sand`, every cycle cut by `study.RULE`, every
problem driven by the port's algorithm for its shape; `study.study(configuration, arm)`
solves one arm cold. The decision kinds (belief / build / operating per boundary input,
the model outputs that are really build decisions) are data in `configurations/kinds.py`.

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
