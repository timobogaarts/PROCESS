# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this file is for

This is **not** a general PROCESS contributor guide — see `README.md` and
`CONTRIBUTING.md` for that, and `documentation/source/development/standards.md` for the
variable naming convention. This file exists for one purpose: **preparing to rewrite
PROCESS models in the `cottax` style** (repo `~/jaxgraph`, package `src/cottax/` — read
`~/jaxgraph/CLAUDE.md` before working here, it is the vocabulary this file assumes).
cottax declares computation as a graph of named nodes with typed ports, decomposes it
into SCCs (`Blocking`), and drives each coupled block with an explicit, autodiff-visible
algorithm (`Square`/`RootFind`/`FixedPoint`/`Optimise`) rather than one opaque
finite-difference optimiser over everything.

**Do not rewrite models yet.** This file records the current architecture, the mapping
from PROCESS's concepts to cottax's, and the concrete obstacles a rewrite will hit. Model
rewrites are future work that will consume this document.

## The environment: `process_port`

**`process_port` is the env for this work** — the one place where
`process` and `cottax` are importable in the *same* interpreter. That co-importability is
the whole point of it: the test harness can call a PROCESS reference function and its JAX
port in one process and diff them directly, with no serialise-to-golden-file boundary
between two envs. Do not assume any other env: `stellsim`/`jaxsn`/`beerpy` resolve
`process` off the repo path but are not installed (`importlib.metadata.version("process")`
raises `PackageNotFoundError`), and only `beerpy` has `cottax` at all.

```bash
PY=~/miniconda3/envs/process_port/bin/python   # or: conda activate process_port
# The conda root differs per machine: if the path above does not exist, try
#   PY=~/miniconda/envs/process_port/bin/python   (no "3") — this is the live path on
# at least one machine. Check with `ls -d ~/miniconda*/envs/process_port` before
# assuming the env is missing; a wrong root looks exactly like a lost env.
```

Python 3.12 (cottax needs ≥3.12, PROCESS ≥3.10). Built with:

```bash
conda create -n process_port python=3.12 -y
$PY -m pip install -e "$HOME/PROCESS[test]"        # editable — tracks this working tree
$PY -m pip install -e "$HOME/jaxgraph[dev,viz]"    # editable cottax, same
```

Both installs are **editable**, so edits to `process/`, `functional_process/` and
`~/jaxgraph/src/cottax/` are live with no reinstall. **Check that this is still true of
`cottax` before trusting it** — on 2026-09-01 the env was found resolving `cottax` from a
*snapshot copy*, not from `~/jaxgraph/src`, with no `"editable": true` in its
`direct_url.json`; every cottax edit made for days had been invisible to the port, and
nothing failed to say so. One line settles it:
`$PY -c "import cottax; print(cottax.__file__)"` must print a path under
`~/jaxgraph/src`. Re-editable with `$PY -m pip install -e ~/jaxgraph --no-deps`. This
matters most for bit-level work: a silent switch of cottax underneath a measurement is
the kind of thing that produces an irreproducible number and a confident wrong
explanation. Verified state at creation
(rebuilt 2026-08-18 after the env was lost): `process 0.0.1.dev1186+g769950de1`,
`cottax 0.1.0`, `jax` **0.11.0** (recorded here as 0.11.1 until 2026-09-06, when the env
was checked directly and read 0.11.0; CPU — no CUDA jaxlib, and jax warns about that on every
import; harmless), `numpy 2.5.2`, `pytest 9.1.1`. **`tests/unit` → 846 passed;
`~/jaxgraph` → 740 passed, 3 skipped.** If either number moves without you having
changed something, suspect the env before the code. The rebuild reproduced both numbers exactly, so the
`jax` 0.11.0 → 0.11.1 drift is inert as far as either suite can see.

### `process_port_gpu` — the env for the Warp/GPU work

**`process_port` cannot run any of it**: it has no `warp`, and its `jax` is CPU-only. The
Warp transpiler, the emitted kernels and every GPU timing number live in a *second* env:

```bash
G=~/miniconda/envs/process_port_gpu/bin/python
```

Verified 2026-09-06: `warp 1.17.0`, `jax 0.11.1` **with** a CUDA jaxlib
(`jax.devices()` -> `CudaDevice(id=0)`, a Quadro T1000, 4 GiB, sm_75), `process`
`0.0.1.dev1585+g7291828db` and `cottax 0.1.0` **both editable** against `~/PROCESS` and
`~/jaxgraph` -- so the same "is cottax still editable?" check above applies here too, and
for the same reason.

Two things that have each cost real time:

- **Read the sentence at the top of this section before debugging a `ModuleNotFoundError:
  No module named 'warp'`.** It is not a broken env; it is the wrong one. This file
  previously said only "`process_port` is the env for this work", which is what sent
  someone looking for a lost install.
- **`export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false` for correctness runs.**
  JAX preallocates most of the card by default, and on a 4 GiB device that produced an OOM
  which was then confidently misattributed to memory fragmentation.

`graphviz` is deliberately **not** installed: the Python binding came in via cottax's
`viz` extra but the `dot` executable did not, so `cottax.visualization.render` raises.
`to_dot` still works and `draw` falls back to its own layered layout — see
`~/jaxgraph/pyproject.toml`'s comment. Install `conda install -c conda-forge graphviz`
only if rendering is actually wanted.

`x64` is **not** on by default and PROCESS is float64 throughout — every entry point in
this env must `jax.config.update("jax_enable_x64", True)` before any array is created, or
diffs against PROCESS show precision loss that reads like a porting bug
(`functional_process/_audit/traceability_policy.md` §Precision).

### `process_port_gpu` -- the same env on CUDA

Built 2026-09-06, on request, to measure whether the GPU helps. **It does not, for a
single solve** -- see below and `_audit/optimise_design.md` §68.

```bash
PYG=~/miniconda/envs/process_port_gpu/bin/python
conda create -n process_port_gpu python=3.12 -y
$PYG -m pip install -e "$HOME/PROCESS[test]"
$PYG -m pip install -e "$HOME/jaxgraph[dev,viz]"
$PYG -m pip install --upgrade "jax[cuda12]"     # last, so nothing downgrades it
```

**Intended to match `process_port`'s `jax`/`jaxlib` so a CPU/GPU comparison is a
comparison of backends, and it does NOT, quite**: `jax[cuda12]` pulled **0.11.1** while
`process_port` is actually on **0.11.0** -- which is also a correction to this file, which
claimed 0.11.1 for the CPU env (verified 2026-09-06: `process_port` is 0.11.0). The prior
evidence is that this drift is inert -- the rebuild note below records both suites
reproducing their counts exactly across it -- but a strict comparison should pin the GPU
env with `pip install "jax[cuda12]==0.11.0"` first. Both installs editable, same as above;
validated at creation with `tests/unit` -> **846 passed**, the identical count `CLAUDE.md`
records for the CPU env, and `cottax.__file__` under `~/jaxgraph/src`. 6.5 GB on disk.

Hardware: **Quadro T1000, 4 GB VRAM** with ~0.5 GB already taken by the display. Always
run with `XLA_PYTHON_CLIENT_PREALLOCATE=false` -- XLA grabs 75 % of VRAM by default, which
on this card leaves nothing.

**Do not expect a speedup on this card, and know why.** The port's blocks are ~28k
*scalar* operations with ~100 KB of runtime buffers and essentially no arithmetic
intensity, and a cold row is ~97 % compilation. On top of that, **this GPU runs float64 at
1/16 the rate of float32** (measured: 76.6 against 1254.7 GFLOP/s on a 1024x1024 matmul),
and PROCESS is float64 throughout -- so every kernel pays a 16x penalty a data-centre card
(1/2 rate) would not. Treat the numbers below as specific to a 35 W laptop Quadro, not as a
verdict on GPUs (`_audit/optimise_design.md` §70). Measured on `helias_5b` MDF, identical
answer to every digit: cold 13.55 s (CPU) -> 15.57 s (GPU), warm 0.136 s -> 0.227 s. The
plausible win is **batching** -- `vmap` over many independent solves, which is the shape of
`process/core/scan.py` -- and that is untested.

### Commands

```bash
$PY -m pytest functional_process/tests         # the port's validation harness — 3752
                                            # passed + 3347 skipped, ~60 s. The cases
                                            # mirror `functional_process/`; the audit
                                            # records stay next to the port. See below.
$PY -m pytest tests/unit                    # unit tests (models, core) — 846, ~4 s
$PY -m pytest tests/unit/models -k density_limit
$PY -m pytest tests/unit/models/stellarator # the in-scope subset — 16, <1 s
$PY -m pytest tests/integration
$PY -m pytest tests/regression -k large_tokamak   # tracked reference output; clones
                                            # process-tracking-data into a user cache
cd ~/jaxgraph && $PY -m pytest              # cottax — 740, 3 skipped
$(dirname $PY)/ruff check && $(dirname $PY)/ruff format          # style; see
                                            # standards.md for naming rules
```

`ruff` is pinned by the repo's `lint` extra (0.16.1) and installed in `process_port`
only — it is not on `PATH`.

## The validation harness (`functional_process/`)

Built and green; the design and the reasoning behind every choice live in
`functional_process/_audit/test_harness.md` (§ As built), which is the file to read
before touching it. The short version:

- **A unit's three files share a stem at the same relative path in three trees**:
  `density_limits.py` (the port) in `functional_process/`, `density_limits.md` (the audit
  record) under `functional_process/_audit/units/`, and `test_density_limits.py` (the
  case) under `functional_process/tests/`. What binds a record to its unit is its row in
  `_audit/unit_registry.md`, which names the path explicitly and is enforced by the
  meta-tests in `functional_process/tests/test_registry_coverage.py` — not adjacency.
- **Tier is a base class.** A case declares `audit_record`/`reference`/`ported`/`samples`
  and subclasses `Tier1Contract` or `Tier2Contract`; it writes no test functions. Tier 2
  has no value-agreement test *by construction*, because PROCESS's answer is not ground
  truth for a unit whose loop never converged.
- **Gradient agreement is checked against PROCESS's own finite difference, with a
  per-point error bar** derived by Richardson extrapolation rather than a fixed `rtol`.
  This is the check the rewrite is being bought: a `stop_gradient` injected into the
  pilot port failed 10 gradient tests while every value test still passed.
- `--fp-fuzz N` / `--fp-fuzz-seed S` control random sampling; `-k legacy` / `-k fuzz`
  select by sample provenance.

`_audit/unit_registry.md` is the authoritative per-unit status and
`_audit/next_steps.md` the priority-ordered punch list — read those, not this paragraph,
for what is ported. Roughly: most of `models/stellarator/**` and three `models/physics/`
units are ported and harness-tested; the rest is audit records or still pending.

`hatch` envs are declared in `pyproject.toml` (`tests`, `tests-unit`, `tests-regression`,
`tests-integration`, `tests-examples`) but `hatch` is not on `PATH` — invoke `pytest`
directly through `$PY` as above.

## The current architecture (what is being replaced)

```
IN.DAT --init.init_process--> DataStructure (one big mutable object, ~40 dataclass fields)
                                    |
                    Caller._call_models_once(xc)   <- fixed, hand-written call order
                    for model in [plasma_geom, build, physics, tfcoil*, pfcoil, ...,
                                  power, vacuum, buildings, availability, costs]:
                        model.run()     # reads and writes DataStructure in place
                                    |
                    objective_function(data) -> float     (numerics.i_figure_merit)
                    constraint_eqns(m, data) -> residuals  (numerics.icc, ConstraintManager)
                                    |
                    VMCON (SQP) over xc = iteration variables (numerics.ixc)
                    gradients by finite differences (Evaluators.fcnvmc2, epsfcn perturbation)
```

- **`DataStructure`** (`process/core/model.py`) is a `dataclass` of ~40 sub-dataclasses
  (`PhysicsData`, `BuildData`, `TFData`, `ConstraintData`, `NumericsData`, ...), one per
  `process/data_structure/*_variables.py` module. This is PROCESS's only namespace: every
  physics/engineering variable lives at `data.<area>.<name>`, e.g.
  `data.physics.rmajor`, `data.build.dr_tf_inboard`. There is no encapsulation — any model
  may read or write any field of any area.
- **`Model`** (`process/core/model.py`) is the node-shaped abstraction that already
  exists: `abc.ABC` with `run()` (compute, mutate `self.data`) and `output()` (write
  formatted results). Every class in `process/models/**` implements it. A `Model`
  constructor may take other `Model`/sub-model instances (see `Models.__init__` in
  `process/main.py`) — this constructor graph is PROCESS's only existing notion of a
  dependency DAG, and it is **assembled by hand**, not derived from reads/writes.
- **`Caller._call_models_once`** (`process/core/caller.py`) is the entire computation
  graph, flattened into one hard-coded imperative call sequence with `if`/`elif` branches
  selecting among alternative sub-models (costs 1990/2015/custom, blanket
  CCFE-HCPB/DCLL, TF coil resistive/superconducting/aluminium, tokamak/stellarator/IFE).
  There is no declared graph object anywhere — the call order **is** the only
  serialization of dependency structure, and it was authored, not derived.
- **Implicit cycles are hidden, not declared.** `Caller.call_models` (not
  `_call_models_once`) calls the whole pipeline **up to 10 times** per optimiser
  evaluation and stops when the objective and constraints stop changing
  (`check_agreement`, `rtol=1e-6`). This *is* a fixed-point (Gauss–Seidel) iteration —
  PROCESS has genuine feedback loops among models (e.g. plasma physics ↔ TF coil ↔
  build), but nothing says so structurally; it is discovered empirically at runtime by
  re-running everything and checking idempotence, and a `RuntimeError` is raised after 10
  rounds if it never settles. There is no SCC, no blocking, no declared "this converges in
  k rounds."
- **Iteration variables** (`process/core/solver/iteration_variables.py`,
  `ITERATION_VARIABLES: dict[int, IterationVariable]`) are the unknowns: a numeric ID
  (1, 2, 3, ...) maps to `(name, module, lower_bound, upper_bound, target_name?,
  array_index?)`. `numerics.ixc` holds the *active* IDs for a given run (arbitrary subset
  and order, chosen per input file). `target_name`/`array_index` let one ID address one
  slot of an array field (e.g. `f_nd_impurity_electrons(03)` → element 2 of
  `impurity_radiation.f_nd_impurity_electron_array`). This ID indirection is PROCESS's
  closest analogue to cottax's `VarPath` — except addressed by an arbitrary integer, not
  by structural name.
- **Constraints** (`process/core/solver/constraints.py`) are similarly ID-indexed:
  `ConstraintManager` is a decorator-based registry, `numerics.icc` lists active
  constraint IDs (equalities first, then inequalities — position, not type, marks the
  split; `n_equality_constraints`/`n_inequality_constraints` record the boundary). Each
  registered function reads `data` directly and returns a `ConstraintResult` via one of
  `eq`/`leq`/`geq` (residual + normalised residual + value + bound). **Many constraint
  bodies already call a model's pure `calculate_*` staticmethod and compare its result to
  a value elsewhere in `data`** — e.g. constraint 1 calls
  `PlasmaBeta.calculate_plasma_beta(...)` and compares it to `data.physics.beta_total_vol_avg`
  via `eq(...)`. That shape — a pure function's output compared against another node's
  output — **is** cottax's `Compare`, already present in embryonic form.
- **The objective function** (`process/core/solver/objectives.py`) is a single `if/elif`
  chain keyed by `numerics.i_figure_merit`, each branch a linear/scaled read of one or two
  `data` fields (e.g. `0.2 * data.physics.rmajor`, `data.costs.coe / 100.0`). Sign encodes
  minimise vs. maximise. This is a *query* over the graph's outputs, not a node — the
  cottax analogue is `Graph.prune(wanted)`/an `Optimise` problem's objective condition,
  selected per run rather than baked into structure.
- **The solver** (`process/core/solver/solver.py`, `evaluators.py`) treats the *entire*
  pipeline above as one opaque function `xc -> (objf, conf)` (`Evaluators.fcnvmc1`) and
  differentiates it by forward/backward finite differences per iteration variable
  (`fcnvmc2`, perturbation `epsfcn`) — `O(n)` extra full pipeline evaluations per
  gradient step, no autodiff. VMCON (a Fortran-derived SQP solver, now via `PyVMCON`) then
  takes one step. This is functionally cottax's `Optimise` problem type, but with the
  *whole graph* as a single undifferentiated block — there is no blocking, so there is
  nothing smaller than "everything" to hand to a driver, and no way to `jacfwd` any part
  of it.
- **`Scan`** (`process/core/scan.py`) is an outer loop that re-solves the whole system
  while sweeping one input variable (`ScanVariable`, addressed via an `Area` enum that is
  literally a short-code alias for `DataStructure` field names — the closest thing to a
  root/namespace list in the current code). Each scan point is an independent solve; no
  state or Jacobian is reused between points.
- **Model bodies are already split into a pure core and a stateful shell** in many places:
  `Model.run()` reads `self.data.*`, calls a `@staticmethod`/pure `calculate_*` method
  with explicit keyword arguments, and writes the result(s) back to `self.data.*` (see
  `PlasmaDensityLimit.run` in `process/models/physics/density_limit.py`, which calls
  `self.calculate_density_limit(b_plasma_toroidal_on_axis=..., ...)`). **This is the
  extraction seam** — the `calculate_*` staticmethod is close to being a cottax
  `CallableNode.fn` already; `run()` is the `In`/`Out` binding a rewrite would make
  explicit and structural instead of textual.

## Logical mapping: PROCESS concept → cottax concept

| PROCESS | cottax | Notes |
|---|---|---|
| `data.<area>.<name>` field | `VarPath` (`.area.name`) | The dataclass-field dotted path is already exactly cottax's `Path`/`VarPath` shape — `data_structure/*_variables.py` module names are natural root namespaces. Array elements (`f_nd_impurity_electron_array[2]`) need a `SequenceKey`/`FlattenedIndexKey` component. |
| `Model` subclass (or a `calculate_*` staticmethod within one) | `CallableNode` (or `DeclaredNode` if it's a problem) | The pure `calculate_*` core is the `fn`; the surrounding `run()` is the `In`-read/`Out`-write PROCESS currently writes by hand and untyped. |
| Hard-coded order in `Caller._call_models_once` | binding order in a `Graph` + `scc_order_graph` | The call order is a *witness* of a topological sort someone worked out by hand; a real `Graph` would derive it (and expose where it *isn't* a DAG). |
| "Call everything up to 10x until idempotent" (`Caller.call_models`) | `Blocking` (SCCs) + `Drive` steps solved by an explicit algorithm | The current code is Gauss–Seidel-by-accident on the *whole* graph. A cottax graph would isolate just the genuinely coupled nodes into one or more SCCs and drive each with a declared `Square`/`FixedPoint`/`RootFind`, leaving everything else as ordinary acyclic `Call` steps. **The rewrite's case does not rest on how much of PROCESS turns out to be genuinely cyclic** — that is an open empirical question the audit is tracking (`functional_process/_audit/next_steps.md`; one confirmed SCC among 44 ported nodes so far, with more expected once the orchestration layer, currently unported, is reached). The actual thesis is structural: making the dependency graph explicit is what makes decomposition, reordering, and per-block algorithm choice *possible* at all — whatever coupling genuinely exists gets isolated and driven by an explicit, autodiff-visible algorithm chosen for that block, instead of every evaluation blindly re-running the *entire* pipeline regardless of which parts depend on which. That is the robustness/efficiency case, independent of the eventual cyclic/acyclic split. |
| `ITERATION_VARIABLES[id]` + `numerics.ixc` | `unknowns` (`owns`) of a `DeclaredNode` | The integer ID and its `(module, name, array_index)` triple is exactly a `VarPath`; the ID itself is throwaway indirection once names are structural. |
| `ConstraintManager` registry, `numerics.icc` | `reads`/conditions of a `DeclaredNode`, or a `Compare` node | Constraint bodies of the shape "call a `calculate_*`, compare to a `data` field" are already `Compare`-shaped; ones that just threshold one `data` field against a bound are more like a bare residual read with no interesting node. |
| `numerics.i_figure_merit` branch in `objective_function` | the objective condition of an `Optimise` problem, chosen by `Graph.prune`-style query | Not a node — a per-run selection of which existing output is "wanted", same as cottax's refusal to have an `OutputNode`. |
| VMCON over the whole pipeline, finite-difference gradients | `Optimise` (`problem.py`) `Drive`, ideally over a much smaller block, differentiated by `jax.jacfwd` once bodies are JAX-traceable | The prize: the current design pays for one opaque global finite-difference Jacobian on every solver iteration; if most of the graph is acyclic and JAX-traceable, only the actual coupled blocks need an iterative driver at all, and even those can use forward-mode AD internally. |
| Model constructor injection (`Models.__init__` wiring in `process/main.py`) | nothing yet — closest is a hand-assembled `Insert(nodes)`/`Graph([...])` | This is evidence of *intended* dependency structure (e.g. `PFCoil(cs_fatigue=..., cs_coil=...)`) that could seed which nodes to declare, but it wires *model objects*, not variables — it says nothing about which fields are read vs. owned. |
| `if data.<switch> == <Enum member>: model_a.run() else: model_b.run()` (costs/blanket/TF coil/tokamak-stellarator-IFE variants) | no direct cottax equivalent — see Difficulties | Runtime variant dispatch on an enum switch. cottax's graph is a fixed structure per `Graph` instance; a switch like this means either building a different `Graph` per configuration, or finding a modelling convention cottax doesn't have yet. |
| `Area` enum in `scan.py` | a manual, incomplete root-namespace list | Independent evidence that `data.<area>` is the natural unit of namespacing, from a completely different part of the codebase. |
| `documentation/source/development/standards.md` naming scheme (`<type>_<system>_<description>_<units>`) | the spelling convention `Path.path_str()`/`written()` would render | PROCESS variable names already encode almost everything cottax asks a name to carry (what kind of quantity, which subsystem, units) — a real asset for readable minted names (`^cond.*`, `^hat.*`) and for choosing node/root names that read naturally. |

## Difficulties specific to this rewrite (not yet resolved, don't paper over these)

- **No model declares its read set or write set.** `Model.run()` bodies reach into
  `self.data.*` freely and without restriction; the only way to know what a given
  model reads is to read its source (or the `calculate_*` signature, where one exists).
  Recovering `In`/`Out` ports for even one model requires a careful read of that model's
  `run()`, not a mechanical extraction — and some models' `run()` methods are hundreds of
  lines with many branches (`bootstrap_current.py` is 2500 lines; `physics.py` nearly
  7000), each potentially reading different fields depending on switches.
- **Implicit cycles are currently discovered by brute-force re-evaluation, not declared.**
  There is no source of truth for *which* nodes are mutually coupled — that structure has
  to be reverse-engineered from read/write sets once they exist, and cross-checked against
  the empirical "stabilises within 10 full-pipeline passes" behaviour, which today is the
  only thing verifying convergence at all.
- **Variant dispatch (switches) has no clean node-graph shape.** Costs
  (1990/2015/custom), blanket (CCFE-HCPB/DCLL), TF coil conductor
  (resistive/superconducting/aluminium, further split by turn type), and whole-device mode
  (tokamak/stellarator/IFE) are chosen by integer switches read from input, each
  routing to an entirely different subgraph of models with different reads/writes. A
  cottax `Graph` is a fixed binding of names to definitions; representing "one of several
  possible subgraphs, chosen at input-parse time" cleanly (rather than unioning every
  variant's ports into one node) is an open design question, not solved by anything in
  `~/jaxgraph` today.
- **The pure/impure split is uneven across the codebase.** Some models cleanly separate a
  `calculate_*` staticmethod from the `run()` shell (`density_limit.py` is a good
  example); many others interleave `data` reads, branching on switches, and computation
  throughout `run()` with no separable pure core. Extracting a `CallableNode.fn` is cheap
  for the former and a real rewrite for the latter.
- **Not everything is JAX-traceable.** At least the CoolProp calls
  (`process/core/coolprop_interface.py`, used for thermophysical properties) are opaque
  external C-library calls, not JAX primitives — the tracing half of cottax (`primitive.py`,
  currently parked, see `~/jaxgraph/CLAUDE.md` §"The tracing half") would need either a
  custom JAX primitive wrapping CoolProp or these nodes staying non-traced (`Out.static`-like,
  or simply outside any differentiated block).
- **Array-valued and per-element iteration variables complicate `VarPath`.** Some
  `IterationVariable`s address one element of an array field via `array_index` and a
  `target_name` that differs from the reported `name` (see ID 125/126 in
  `iteration_variables.py`, addressing `f_nd_impurity_electron_array[2]`/`[3]` under the
  display name `f_nd_impurity_electrons(03)`/`(04)`). Mapping this onto a `Path` is
  mechanical but must preserve both the storage location and the human-facing label.
- **Constraint equality/inequality split is positional, not typed**, and constraints are
  independently addressable by ID even when several are logically about the same
  quantity — unlike cottax's `Compare`, which mints one node per `place` for however many
  pairs share it. A faithful port needs a policy for when several PROCESS constraint IDs
  collapse into one cottax `Compare`/problem versus staying separate prunable sinks.
- **Regression tests are the ground truth for numerical behaviour**, tracked externally
  (see `documentation/source/development/testing.md`, reference data in a separate
  `process-tracking-data` repo) and compared with a percentage tolerance, not exact
  equality. Any rewritten model must reproduce existing outputs within that tolerance —
  there is no simpler oracle.
- **`~/jaxgraph`'s own caveat applies doubly here**: "Validate before porting" — cottax
  itself is pre-Phase-4 and still a second opinion against `jax_sn`/`jaxmdo`, not
  production. A PROCESS port would be building on a foundation that is itself still
  settling (see "Rules that must not be quietly undone" in `~/jaxgraph/CLAUDE.md`).
