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
