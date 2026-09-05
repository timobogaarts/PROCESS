# Cold boundary — where every input's value comes from, and why 11 roots went non-finite

**Closed.** The cold tokamak MDA ran 183/183 schedule steps with 0 failures and 0
ungrounded inputs, yet produced non-finite values at 11 roots (a non-finite output whose
declared reads are all finite), 104 owned outputs affected. **The failure mode of a graph
like this is silently finite-looking wrong numbers plus loud non-finite ones, not
exceptions** — 0 failures/0 ungrounded is not evidence of correctness on a cold start.

## What it was: one defect class, four missing producers

All 11 roots traced to exactly **six boundary values stuck at their bare dataclass
default (`0.0`)** because their PROCESS producer was unported, flowing into a division or
exponent (`x/0`, `0/0`, `exp(nan)`). Not a PROCESS cold singularity — in every case
PROCESS's own producer runs earlier in the same evaluation pass and its own inputs are
ordinary file literals, verified by running the full evaluation and confirming all six
become finite and nonzero. Four unported producers accounted for all six zeros:
`Build.calculate_radial_build`'s TF radii slice, `Fw.set_fw_geometry`,
`Physics.plasma_ohmic_heating`, and `PFCoil.vsec`.

A companion measurement — diffing the cold and converged `DataStructure` for a full
`sr.run()` — found **29 boundary inputs a PROCESS run actually overwrites** (2 of them
the run's own iteration variables), of which the six degenerate zeros were a subset; the
other 23 are the same class without the alarm, silently stale-but-finite on a cold start.
Two of the 29 are **file literals a model overwrites anyway** (the pedestal density
pair) — the same defect shape as an earlier stellarator finding (the L-mode profile
reset), found independently on the tokamak by the same before/after diff method: **no
check based on the input file alone can see a value the model is entitled to overwrite.**

## Fixed, same day: all four producers landed, in payoff order

`FirstWallGeometry` (11→4 roots), the CS-to-TF radial slice (4→1), `PlasmaOhmicHeating`,
then `PFCoil.vsec` (which also merged the PF ring and volt-second/burn-time ring into one
nine-node SCC, cut sufficiently and minimally by the standing `mda.CUTS` trio). Final
state: **185/185 steps, 0 failures, 0 ungrounded, 0 non-finite, 0 roots.** Warm harness
gained exactly the 14 new owned outputs as agreements, with no other row moving.

## One number worth keeping: how close to PROCESS-free this boundary already was

Of the reference tokamak's 349 pinned boundary inputs, **98% (342) were already pure data
on a cold start** — 90 file literals plus 252 dataclass defaults surviving
`init_process` untouched, needing no PROCESS machinery at all. The entire irreducible
`init_process` dependency at this boundary was **seven variables**: four are impurity
data tables loaded from packaged files (a data asset to ship, not init logic to port) and
three are one-line switch-derived rules (`i_single_null`→`n_divertors`, the
`eff_tf_cryo` sentinel, `f_nd_beam_electron` zeroing) that belong in the machine factory
next to the switches they already read.
