# paper_tests

The uncertainty study of the PROCESS section of *Managing complexity for integrated
design in fusion using a graph-based functional framework* (`~/graph_paper`): what
happens to PROCESS's deterministic stellarator optimum when the beliefs behind it are
sampled. Three drivers, one shared module, one job script. Everything else this
directory once held (the census, timings, architectures, batching, close-conditions,
Sobol' and OUU studies and their outputs) was removed on 2026-09-20 and is in git
history at `eae0ae45`.

**The scripts are thin command-line drivers over the port.** The physics, the graph
operations and the batched programs live in `functional_process/cottax/architectures/`
(`session`, `closing`, `lift`, `stages`, `beliefs`, `ouu`) and
`functional_process/configurations/kinds.py` (the decision kinds and the belief
table); what lives here is the command line, the statistics, the figures, and the one
thing the port deliberately does not carry -- PROCESS itself in the loop
(`common.process_reference`: one converged VMCON run, cached under
`~/.cache/functional_process/`, which is the deterministic design every study starts
from and is compared against).

| file | what |
|---|---|
| `flexibility.py` | **the study.** Fix the machine, let the operator re-optimise per belief draw; the coe histogram against the deterministic optimum. Its docstring is the recipe and the answer table. |
| `flexibility_dsm.py` | the graph that study solves, as two interactive DSM pages (`out/dsm/`); its docstring is the graph recipe from the uncut base graph, step by step with node counts |
| `ga.py` | the converse question: a genetic algorithm over the machine, scored by the batched MDA across sampled worlds -- is there a machine that fares better? |
| `one_dial.py` | **the minimal showcase**, on `stellarator_helias` and `low_aspect_ratio_DEMO`. `hfact` becomes a given; the graph refuses, one variable is named to absorb the power balance, the build is frozen, and the dial is swept -- `redesign` (PROCESS's own question at each value) against `built` (the design held). `report(plan)` prints the recipe, one named graph operation at a time. `functional_process/tests/test_one_dial.py` pins it. |
| `common.py` | the env pin, `process_reference`, `deterministic_values` |
| `decision_kinds.md`, `output_kinds.md` | the census behind `kinds.py`: every build decision (design variable or fixed), every belief, and the guarantee that no build decision is left in the graph |
| `cluster/` | `job_flex.sh` (N = 262144 on one Snellius `rome` node in ~6 min) and its environment |
| `out/` | the one run kept: `flex_n262144_s{0.10,0.05,0.02}_hfact{,_req982.4}.npz` + `_summary.json`, `flex_coe.png`, `flex_histograms.png`, `ga_run.json` (penalty 1000), `ga_run_penalty250.json`, `ga_best.json`, `ga.png`, `dsm/` (regenerated, not tracked) |

```bash
PY=~/miniconda3/envs/process_port/bin/python      # see ../CLAUDE.md for the env
SC=<scratch>; mkdir -p $SC/cottax_head && git -C ~/jaxgraph archive 6b1d540 src | tar -x -C $SC/cottax_head
export JAX_PLATFORMS=cpu PYTHONPATH=~/PROCESS:$SC/cottax_head/src:~/PROCESS/paper_tests
cd ~/PROCESS
$PY paper_tests/flexibility.py --n 1024 --workers 8 --order hfact                     # ~3 min
$PY paper_tests/flexibility.py --n 1024 --workers 8 --order hfact --require-net 982.4
$PY paper_tests/flexibility.py --figures
$PY paper_tests/flexibility_dsm.py
$PY paper_tests/ga.py --pop 32 --n 512 --gens 40 --certify 4096                       # ~10 min
```

The `PYTHONPATH` line pins cottax at the commit the port tracks (`6b1d540`);
`~/jaxgraph`'s working tree has moved past it (`Schedule()` takes an
`ExecutableGraph` there) and the editable install would pick that up.

## The answer (N = 262144, `flexibility.py`)

With the build held -- five numbers, the winding-pack width among them once
`lift.lift_winding_pack` stops the coil from being re-sized per sample -- and the
operator free to set `T_e` and the helium fraction in each world, PROCESS's machine
can be run in 43 % of worlds (hfact sigma 0.10; 40 % / 38 % at 0.05 / 0.02), and in
those worlds its cost of electricity is centred **1.29x the deterministic value**
(median 159.5 against 123.6 $/MWh; 1.19x / 1.12x at the smaller sigmas). The
deterministic optimum sits at the 10th percentile of its own distribution: only one
operable world in ten is cheaper than the one it was designed in. Every failure is an
operating failure (the seven build constraints hold in every world), the binding set is
c83 / c35 / the pack rule at ~100 %, all fixed by the build, so the knobs buy almost
nothing. Rated power is a reference line, not a criterion: 10 % of operable worlds reach
it (`flex_coe.png`, lower panel).


## One dial in, one dial out (`one_dial.py`)

```bash
PY=~/miniconda3/envs/process_port/bin/python
export JAX_PLATFORMS=cpu PYTHONPATH=~/PROCESS:~/jaxgraph/src
cd ~/PROCESS
$PY paper_tests/one_dial.py                                   # both, ~5 min
$PY paper_tests/one_dial.py --machine stellarator_helias      # ignited: the density absorbs
$PY paper_tests/one_dial.py --machine low_aspect_ratio_DEMO   # driven: the heating absorbs
```

**Not the `6b1d540` pin quoted above**: that commit predates `cottax.pytree`, which the
port now imports. `one_dial.py` runs against `~/jaxgraph` at `6488a1c` or later -- the
editable install, or a worktree of it where that repository is being edited elsewhere.
Both machines in one process clears jax's caches between them; without that, two
configurations' worth of jitted schedules is already tight on a 15 GB laptop.

**Two machines, the two genuinely optimised configurations** (`SHOWCASE`), and they
reach the same place by different routes -- which is why both are shown. The recipe is
four named procedures, each `Plan -> Plan`, each recording the graph operations it
applied:

0. **read the input file** -- PROCESS's question verbatim, no operation.
1. **make `hfact` a given** -- it leaves `chosen`. On `stellarator_helias` `hfact` is
   `ixc = 10`, active and converged to 1.056 *inside* its bound, so giving it away is
   what over-determines the problem and the graph refuses:
   `.Close.c2 holds (^cond.constraints.c2,) against nothing: it has no unknowns, so no
   driver can move it -- Determine it by a variable on its cycle, or drop it`. On
   `low_aspect_ratio_DEMO` `hfact` is fixed in the file at 1.1 and was never an `ixc`,
   so **this step is honestly a no-op** and says so; nothing refuses yet.
2. **name what absorbs it** -- one `Insert(RootFind((c2,), (var,)))`, nested on its
   cycle: the density on the ignited machine (a 23-node cycle) and the heating power on
   the driven one (6 nodes, and never an `ixc` in any file -- PROCESS cannot be asked
   this without restating the problem). `CLOSURE` is that decision, per configuration,
   in a table a reader can argue with.
3. **freeze the build** -- every design variable held at PROCESS's answer, and every
   claimed build output still downstream of the dial (`stages.split` with `hfact` the
   one varying leaf) `Cut` to `^built.*` and checked against what this world asks for.
   On `low_aspect_ratio_DEMO` **this** is the step that breaks the equation count, and
   it draws the same refusal there. That is the sharper form of the point: it is
   freezing the design, not the dial, that breaks the arithmetic a systems code rests
   on -- the dial is only what makes it visible.

`report(plan)` prints all four, and the step that drew the refusal carries it on its own
`REFUSED` line, so which step it was is visible in the printed recipe.

### The numbers

| | `stellarator_helias` | `low_aspect_ratio_DEMO` |
|---|---|---|
| `hfact` | `ixc = 10`, converged to 1.0556 | fixed at 1.1 in the file |
| refusal drawn at | step 1, giving the dial away | step 3, freezing the build |
| closed by | `.physics.nd_plasma_electrons_vol_avg` (23-node cycle) | `.current_drive.p_hcd_primary_extra_heat_mw` (6) |
| at the nominal | 1.74596e20 m^-3, PROCESS's own to 3e-9 | 10.0 MW, the file's own input |
| frozen | 6 build outputs, 12 ops | 5 build outputs, 10 ops |
| conditions constant given the build | 6 of 13 | 18 of 24 |
| margin | 10.38 % | 0.00 % |
| binds first | `c8`, neutron wall load (held by 4.9e-01) | `c16`, net electric power (held by 3.8e-07) |
| tight at the nominal | `c16`, `c24`, `c35` | `c11`, `c30`, `c5`, `c31`, `c34`, `c60`, `c62`, `c68`, `c90` |

**Both say the same thing: an optimum sits on its constraints, so the machine that was
built has next to no room for the world to differ.** On `low_aspect_ratio_DEMO` every
condition the dial can reach but one is already active at PROCESS's answer, and the one
that is not -- the net electric power -- holds by 4e-07, so the operable range is a
point: the heating power climbs from 10 MW to 197 MW by -15 % and the plant's net output
goes from 350 MW to -59 MW. The stellarator has a genuine 10 % before the neutron wall
load goes, but its beta limit `c24` and its net-electric equality `c16` were tight from
the start, so what the sweep buys is bought outside them.

`margin` is how far `hfact` falls before a condition that held **with slack** at
PROCESS's answer stops holding, interpolated between the bracketing sweep points. A
condition already at its limit there cannot break; a frozen capacity check is at its
limit *by construction* the day it is frozen, so those are reported apart
(`frozen_tight_by_construction`) rather than counted as tight conditions.

Two things the sweep does not hide. The heating closure is answered by
`closing.bracketed()` in `[-25, 1000]` MW rather than the default Newton, because a
Newton scales its step by the unknown it moves and a file may install no heating at all;
the bracket's bottom is negative because the root is then *at* zero and a bracket has to
contain it. And the control curve is one cold VMCON per point (warm-started from the
last point that converged, as `Scan` does): on `low_aspect_ratio_DEMO` it converges only
down to `hfact = 1.089` and reports `no-step` below that, so the figure says where the
control stops instead of drawing a curve that is not there.

`out/one_dial_<machine>.{json,png}` per machine: the recipe as text, both curves, every
condition at every point, and the three panels (the objective under both questions, the
absorbing variable, and what the plant then delivers, with the wall marked on all
three). `functional_process/tests/test_one_dial.py` pins the recipe's shape, which step
draws the refusal, and the two nominal numbers.
