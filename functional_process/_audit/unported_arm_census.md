# Which `UNPORTED` refusals a file that says nothing would hit

**Analysis only, not yet acted on.** `indat.UNPORTED` holds 219 rows over 50 switch axes,
all shaped by the same eight regression files (checked against every other `IN.DAT` in
the repo: only 4 new `(switch, value)` pairs anywhere, none of them a new physics arm —
the bias is real and this repo can't sample its way out of it). Since a file is silent
about most switches and gets PROCESS's *default* for each, the question that predicts
what an unseen file actually hits is: **for each `UNPORTED` row, is that value the
switch's default?**

Of 88 dispatched axes, 50 carry at least one `UNPORTED` row. Splitting those 50: **8 are
default-and-unported** (a silent file hits these unconditionally — tier 1), **42 are
non-default-and-unported** (only a file that explicitly asks — tier 2, 211 rows), and 38
axes are fully ported with no `UNPORTED` row at all (tier 3).

## Tier 1 — what a silent file actually needs, ranked by real cost

| axis | unported default | cost to port |
|---|---|---|
| `i_cost_model` | `1` `KOVARI_2014` | whole `Model` package — `costs_2015.py`, ~1227 LOC, zero cottax nodes today |
| `i_hcd_primary` | `5` `ITER_NEUTRAL_BEAM` | ~160 LOC of formula (`NeutralBeam.iternb` + the beam wall-plug block), not a new `Model` |
| `pf_coil_system_arm` | `-2` (the default coil topology) | a third `PFCoilTopology` + a third set of 13 node instances — largest of the eight |
| `i_bootstrap_current` | `3` `WILSON` | one `@staticmethod` formula, ~130 LOC, needs its own occupant and harness contract |
| `i_p_coolant_pumping` | `2` `MECHANICAL` | in flight elsewhere — do not re-analyse |
| `i_density_limit` | `8` `ASDEX_NEW` | near-zero: formula already ported and Tier-1-tested; needs a node class + one registry line |
| `hcd_primary_powers_arm` | derived from row 2 | not independent work, same fact as `i_hcd_primary` at the joint slot |
| `blktmodel_ipowerflow_i_p_coolant_pumping` = 4 | derived, stellarator-only | zero — PROCESS itself refuses this cell too; the port's refusal is already correct |

**The list that actually costs anything is `i_cost_model`, `i_hcd_primary` and
`i_bootstrap_current`** — everything else is nearly free or already in flight. (One doc
slip found in passing, not fixed: `_blanket_shield_power_arm`'s docstring misstates
`i_p_coolant_pumping`'s dataclass default as `1`; it's `2`. Owed to whoever owns
`indat.py`.)

## Blind spots of this method, stated rather than hidden

1. **A default is not a usage probability.** This measures what a *silent* file gets, not
   which switches real users set — tier 2's ranking (by regression-file precedent, doc
   recommendation, unit-test count) is a proxy, not a measurement.
2. **Being keyed on `UNPORTED` misses partial enforcement.** An axis that refuses one bad
   value but silently accepts a different, also-wrong one (the `i_tf_sup` LIVE defect
   found elsewhere) is invisible to any table indexed by refusal rows. This document ranks
   the refusals; it does not find the missing checks.
3. **Only one configuration was evaluated** (conventional superconducting tokamak,
   `itart=0, i_tf_sup=1`); a spherical or resistive machine resolves several sentinels
   differently and may have a different tier-1 list. Not re-run.

Tier 2/3's full row-by-row listings and the evaluation method are in git history; the
current authoritative unported-switch list is `indat.UNPORTED` in code, not this file.
