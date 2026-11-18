# paper_tests/distributed_architectures

Distributed architectures on `stellarator_helias`, each a recipe of primitive graph
operations on the same base as AAO: the file's problem as `problem_model` declares it.
They follow the three steps of jaxgraph's `examples/sellar_plans.py` (remove, assemble,
add). Inside each of their problems they do what AAO's SAND does to the whole problem:
the same cuts and closures, and the coil's implicit solve absorbed.

- **Sequential**: a magnet stage (`b_t`, `rmajor`, `tdmptf`, `fcutfsu`; the coil
  constraints; minimises `0.2 rmajor` with `b_t ≥ 5`), then a plasma stage (the other
  four variables and constraints; minimises COE). Not equivalent to the file's problem.
- **ALC**: augmented Lagrangian coordination (Tosserams, Etman & Rooda 2008) of build,
  operation and economics, split by design context. There are 43 consistency
  constraints, each discipline is an SLSQP solve, and the coordination is a fixed point
  over the three. Equivalent at its fixed point. The paper baseline runs the coordination
  with `scipy.optimize.anderson` at scipy's defaults (`anderson.py`), every weight 0.5;
  `alc.Coordinator` is the plain sweep of the earlier runs. It departs from the paper's
  stated algorithm in three choices: no inner loop, with Anderson mixing of the sweeps;
  the weights are never raised (β = 1); and the stopping test is on the copies' movement
  where the paper's is on `c`'s. What it took to converge, and how each choice stands
  against the paper, is in `alc_report.md`.

## The partition

Both cases split the same file problem: minimise COE (figure of merit 6) over eight
iteration variables, with 2 equality and 12 inequality constraints. ALC has three
disciplines. The sequential split has two stages: its magnet stage is ALC's build, and
its plasma stage is ALC's operation and economics together.

| | **Build** | **Operation** | **Economics** |
|---|---|---|---|
| Iteration variables | `b_plasma_toroidal_on_axis` (b_t), `rmajor`, `t_tf_superconductor_quench` (tdmptf), `f_a_tf_turn_cable_copper` (fcutfsu) | `temp_plasma_electron_vol_avg_kev` (te), `nd_plasma_electrons_vol_avg` (dene), `hfact`, `f_nd_alpha_thermal_electron` (f_alpha) | none |
| Internal unknowns (absorbed as AAO does) | `wp_width_r_min`, the coil cross-section's implicit model | 3 coupling states of the plasma: fuel and total ion density, divertor solid-angle fraction | none |
| Constraints | c32 conduit stress, c34 dump voltage, c35 winding-pack current density against quench protection, c65 vessel stress at quench, c82 coil toroidal gap, c83 radial build | c2 power balance (=), c8 neutron wall load, c17 radiation fraction, c18 divertor heat load, c24 beta, c62 alpha confinement ratio, c67 radiation wall load | c16 net electric output (=) |
| Objective, ALC | none: only the price of disagreeing with the others | none: only the price of disagreeing with the others | COE, at its own copies |
| Objective, sequential | figure of merit 1, `0.2 rmajor`, with `b_t ≥ 5` (magnet stage) | COE (plasma stage, with economics) | COE (plasma stage, with operation) |
| Models | 22: TF coil geometry, cross-section, current, forces, stresses, quench protection, stored energy, length, masses and winding pack; radial build; vessel and cryostat geometry; plasma geometry and scaling factors (`rminor`, volume, surface from `rmajor`); structure masses; coil cryostat area | 33: profiles, plasma composition, fusion rates and powers, confinement scaling and power loss, radiation (impurity, synchrotron, fraction, clipping), beta and stored energy, toroidal and poloidal field, heating power, divertor, first-wall area, neutron and radiation wall loads | 48: costs (31, COE among them), plant power balance (8: AC power, thermal powers, cryogenic loads, coolant pumping, TF power supply), blanket and shield (4), buildings (2), availability (2), vacuum (1) |
| ALC: copies it holds | none | 6 of build's: `b_t`, `rmajor`, `rminor`, plasma volume, plasma surface, unadjusted first-wall area | 25 of build's (radial build and cryostat dimensions, coil, vessel and structure masses, stored energy, dump voltage, turns, current per turn, coil length, coil cryostat area, `rmajor`, `rminor`, volume, surface) and 12 of operation's (fusion, neutron, radiated, alpha and separatrix powers, wall and divertor loads, divertor area and coverage, first-wall area, fuelling) |

- **No back flow.** Nothing operation or economics computes is read by build, and nothing
  economics computes is read by operation. Every model feedback loop lies inside one
  discipline: two in operation (the fusion-power and first-wall-area loops) and the coil
  cross-section's implicit model in build.
- **Five crossing quantities never move with the design**, checked by moving each
  variable by 1 %: the first-wall thicknesses, `eps`, the fuel mass and the coil count.
  They are read directly, with no copy.
- **Not in any problem:** 25 models depend on no iteration variable and run once before
  (initialisation, fixed costs, heating, machine configuration). 21 models depend on the
  design but feed no constraint and not COE (profile integrals, ports, fusion gain,
  `nu_star`, …), and run after.
- **Sequential only.** 15 models depend on the magnet's variables alone, but neither its
  constraints nor its objective read them. They run once, between the stages, from the
  magnet's answer, and the plasma stage reads them as given:
  - 6 of build's: coil masses, coil length, `z_tf_inside_half`, vessel and cryostat
    geometry, structure masses, coil cryostat area;
  - the 7 coil and vessel cost models (TF magnet, superconductor, vessel assembly, power
    conditioning, TF power supply, coil building envelope);
  - operation's toroidal and poloidal field models.
- **How each partition is computed.** The sequential split takes its partition by reach
  (`partition.partition`): what the coil constraints depend on. ALC takes its partition by
  design context (`partition.disciplines`). The two agree on every iteration variable and
  constraint. They differ only on those 7 cost models and the coil cryostat area, which
  ALC places by context.

<details><summary>Every model, by discipline, with its sequential stage</summary>

- **Build** (magnet stage): `.stellarator.build`; `.stellarator.coils.`
  `coil_coil_toroidal_gap`, `coil_cross_sectional_area`, `coil_current`,
  `coil_radial_thickness`, `coil_toroidal_thickness`, `coils_mass` (between),
  `coils_summary_variables`, `intersect` (its implicit solve absorbed into the problem, as in AAO), `len_tf_coil`
  (between), `max_force_density`, `maximum_stress`, `quench_protection`,
  `stored_magnetic_energy`, `tf_cryo_area` (between), `winding_pack_intersect_inputs`,
  `winding_pack_total_size_post`, `z_tf_inside_half` (between);
  `.stellarator.fwbs.cryostat_and_vv_geometry` (between);
  `.stellarator.stellarator_plasma_geometry`, `.stellarator.stellarator_scaling_factors`,
  `.stellarator.structure_masses` (between); constraint bodies c32, c34, c35, c65, c82, c83.
- **Operation** (plasma stage): `.physics.` `auxiliary_physics_quantities`,
  `confinement_time.{inputs, power_loss, scaling, tail}`, `electron_thermal_energy`,
  `fast_alpha_beta`, `fusion_power_totals_mw`, `fusion_rates`, `fusion_totals_no_beam`,
  `impurity_radiation_totals`, `ion_thermal_energy`, `plasma_composition`,
  `plasma_radiation_powers`, `profiles.{density_profile, ion_vol_avg_temperature}`,
  `profiles.parameterisation.{parabolic_on_axis_densities, parabolic_on_axis_temperatures,
  parabolic_profile_values, parabolic_temperature_profile}`, `radiation_fraction`,
  `set_fusion_powers`, `synchrotron_radiation_power`, `total_plasma_heating_power`;
  `.stellarator.` `clipped_radiation_powers`, `divertor`, `fw_area`,
  `heating_and_radiation_power`, `neutron_wall_load`,
  `poloidal_field_from_rotational_transform` (between), `radiated_wall_load_and_fraction`,
  `stellarator_beta_and_stored_energy`, `total_field` (between); constraint bodies c2,
  c8, c17, c18, c24, c62, c67.
- **Economics** (plasma stage): `.costs.` `atmospheric_recovery_cost`,
  `auxiliary_component_cooling_cost`, `blanket_cost`, `constructed_cost`,
  `convert_fpy_to_calendar`, `cost_of_electricity`, `cryogenic_system_cost`,
  `divertor_cost`, `electric_plant_equipment_cost`, `first_wall_cost`,
  `fuel_handling_cost`, `fuel_processing_cost`, `fusion_power_island_cost`,
  `heat_rejection_cost`, `heat_transport_system_cost`, `indirect_costs`,
  `low_voltage_cost`, `magnets_cost` (between), `nuclear_building_ventilation_cost`,
  `power_conditioning_cost` (between), `reactor_cooling_system_cost`, `reactor_cost`,
  `shield_cost`, `structures_cost`, `tf_coil_power_conditioning_cost` (between),
  `tf_magnet_cost_superconducting` (between), `total_plant_direct_cost`,
  `transformers_cost`, `turbine_plant_equipment_cost`, `vacuum_system_cost`,
  `vacuum_vessel_assembly_cost` (between); `.power.` `acpow`,
  `component_thermal_powers`, `cryo_loads`, `cryo_q_loads`, `cryo_q_nuc`,
  `p_fw_blkt_coolant_pump_mw`, `p_fw_div_heat_deposited_mw`, `tf_power` (between);
  `.stellarator.fwbs.` `blanket_masses`, `blanket_shield_power`,
  `fw_blanket_shield_geometry`, `shield_mass`; `.buildings.` `sizing`,
  `tf_coil_envelope` (between); `.availability.` `avail`, `electric_production`;
  `.vacuum.vacuum_old`; the constraint body c16 and the merit (COE).

"Between" marks the 15 models the sequential split runs once, between its stages, from
the magnet's answer.
</details>

## ALC in plain terms

PROCESS solves this machine with one optimiser. VMCON (here SLSQP, as AAO) moves all
eight iteration variables at once. Every iteration, every model is evaluated, and every
constraint and the figure of merit are checked. Everything sees everything else's
current values.

ALC hands the same problem to three teams instead:
- **build** sizes the coils and the machine;
- **operation** picks the plasma's operating point;
- **economics** costs the plant and holds the net-electric target.

Each team runs its own optimiser over its own iteration variables, with its own
constraints. What makes this work:

- **Each team keeps its own working copy of what it needs from the others.** Operation
  needs the plasma volume, `rmajor` and the field, so it holds its own values for them.
  Economics holds its own coil masses, stored energy, fusion power and so on: 37 numbers
  in all. A team is free to move its copies like iteration variables. That is how
  economics, which owns no PROCESS iteration variable, still has something to optimise:
  it asks for the plant that would minimise COE.
- **Disagreement costs.** The gap between a copy and the value the owning team actually
  computes is an inconsistency: 43 of them, each relative to its starting value. Every
  inconsistency carries a price per unit (a multiplier `v`) and a quadratic penalty (a
  weight `w`). Economics minimises COE plus what its disagreements cost. Build and
  operation have no figure of merit of their own: they minimise only what their
  disagreements cost, inside their own constraints. So build tries to deliver the machine
  the others assume, and operation the plasma economics assumes, at the price each
  mismatch currently carries.
- **The coordinator sets the prices.** One sweep solves build, then operation (with
  build's new answer), then economics (with both). The coordinator then reads every
  inconsistency and raises the price of each that persists (`v ← v + 2w²c`). This is the
  method of multipliers, like a market finding the price at which supply meets demand.
  The weights stay at 0.5.
- **It also looks ahead.** Left alone, the sweeps creep: each moves the copies and prices
  only a small part of the way still to go. So the coordinator keeps the last five sweeps,
  sees where they are heading, and jumps there (Anderson acceleration). It needs no
  derivatives: the history stands in for them.
- **At the end** every copy agrees with what it copies (`c = 0`) and the prices have
  stopped moving. Each price is then the marginal value of its quantity to COE: how much
  COE would gain if, say, the coil mass could be 1 % lower. The three teams' optimality
  conditions together are then exactly those of PROCESS's single problem. That holds even
  though the problem is not convex. Here the answer is AAO's optimum: AAO started at
  ALC's answer stays there.
- **What it costs, and what it buys.** 262 sweeps, about 790 small optimisations and
  about 23 s on a laptop, against AAO's one 16-iteration solve in 0.07 s. In exchange, no team ever needs another team's
  models, variables or constraints, only the numbers that cross between them. So each
  could be a different code, solver or group of people.

The sequential split, for contrast, has no prices. The magnet stage decides the machine
first, on its own objective (`0.2 rmajor`, with `b_t ≥ 5`). The plasma stage then takes
that machine as given. Nothing tells the magnet what its choices cost the plant, and COE
ends up 4.4 % higher.

## Files

| file | writes to `out/` | what |
|---|---|---|
| `partition.py` | -- | the reach partition (sequential), the build/operation/economics assignment (ALC), the AAO base every recipe starts from (`base`), and several problems driven and valued as `study` drives and values one |
| `sequential.py` | -- | the sequential recipe; run alone, its result and the refusal of a split whose problems share a cycle |
| `alc.py` | -- | the ALC recipe, its coordination driver, and a run checked against AAO (AAO started at ALC's answer) |
| `benchmark.py` | `benchmark.{csv,md}`, `benchmark_problems.csv` | AAO, sequential and ALC side by side: optimum, constraint violation, overarching-loop and per-problem iterations, cold and warm time (mean of `--runs`) |
| `anderson.py` | -- | ALC's coordination by `scipy.optimize.anderson` on the sweep map, a host callback as `SlsqpDriver` wraps scipy's SLSQP (`AndersonCoordinator`) |
| `run_benchmark.sh` | `benchmark.log` (`alc.log`) | `benchmark.py` (`SCRIPT=alc`: `alc.py`) with the environment pinned (single-threaded BLAS, as the Snellius jobs) |
| `dsm.py` | `stellarator_helias_{sequential,alc}.html` | the XDSM and N2 of a case, cottax's interactive page |
| `alc_report.md` | -- | the ALC implementation, every change convergence needed, every run |

```bash
cd ~/projects/functional_PROCESS/PROCESS
export COTTAX=<cottax>/src                  # see below
paper_tests/distributed_architectures/run_benchmark.sh --runs 5     # ~4 min on one CPU, ALC most of it
SCRIPT=alc paper_tests/distributed_architectures/run_benchmark.sh   # ALC alone, with the AAO checks
```

**cottax.** The results so far (R9–R12, T1–T34) used cottax's old
`distributed_architectures` branch, `0ffbd84` and then its tip `42fd44e`, which differs
only in visualisation. The `~/projects/jaxgraph` checkout that `run_benchmark.sh` still
defaults to no longer exists. `~/projects/cottax` is on another branch with a different
history, and these scripts have not been run on it. To reproduce the results, snapshot
that branch and point `COTTAX` at it:
`git -C ~/projects/cottax archive 42fd44e src | tar -x -C <dir>`, then
`COTTAX=<dir>/src`. The branch is `remotes/jaxgraph/distributed_architectures`.

All three cases use SLSQP at the same settings (`ftol` 1e-8, `max_iter` 500, scaled
coordinates, coupling conditions scaled by the MDA) and start from the file's design.
VMCON is not used, because ALC's unbounded copies take it to a non-finite model.
