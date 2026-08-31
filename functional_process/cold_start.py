"""Does the port's graph compute what PROCESS computes **when nothing hands it the
answer**?

Every other stage of this harness seeds the port from a run PROCESS has already solved.
`mda_harness.compare` seeds boundary inputs from the converged `DataStructure` and diffs
against that same structure; `sand_harness`'s Stage A and Stage C2 do the same through
`ground_truth`. That is the right seed for the question those stages ask -- *"does the
graph reproduce an answer PROCESS already found?"* -- and it makes one whole defect class
**structurally invisible**: a variable no node in the graph owns is a boundary input, so
the seed hands it PROCESS's own value, the graph reads it back, and the comparison passes
on a number the port never computed. That is how the harness reported **983 of 1039
variables agreeing to 1e-9** on `large_tokamak_nof` while twenty-two producers were
missing outright and every cold tokamak solve was broken by them
(`_audit/optimise_design.md` §16.3(b), §16.5).

**A check performed where the seed supplies the answer is not a check.** This module is
the one that does not: it seeds from the `DataStructure` as `init_process` left it --
before any model has run -- and compares against PROCESS's own state after one pipeline
evaluation at the same cold `x`. Every genuine `IN.DAT` input is identical in the two
structures, because PROCESS's pipeline does not write them; that is what makes them
inputs. The only thing the split changes is that anything PROCESS *computes* is seeded
with its uninitialised default instead of with the answer, so the port has to produce it
or disagree.

Three questions, three answers, and the third is the hard one
------------------------------------------------------------
**1. Did the port produce it at all?** `boundary.unproduced_but_computed` already answers
this from the graph's declaration alone and pins the result. This module is its
value-side twin: the same defect seen as a wrong number rather than as a missing owner.

**2. Is there anything to compare against?** `cold_state` records PROCESS's own write set
for the pass, so a port output PROCESS's *solve* pass never writes can be told apart from
one it writes differently. The standing case is `Physics.outplas`, which computes
`.physics.nu_star`, `.rho_star` and `.beta_mcdonald` and is called only from
`Physics.output()` (`physics.py:219-223`) -- never from `run()`. In a converged
`DataStructure` those three are filled by the final report pass, so `mda_harness.compare`
sees them agree; at the cold point PROCESS holds `0.0` and the port holds its own correct
answer. Neither side is wrong and the comparison is empty, so these are counted in
`output_pass_only` and not scored as defects. **The category is derived from PROCESS's
measured write set, never from a hand-written list** -- the same discipline
`boundary.computed_by_process` applies to the converse question.

**3. Is a real disagreement the port's fault, or has PROCESS not settled?**
`Caller.call_models` runs the pipeline at most ten times and stops when the objective and
the constraints stop moving (`caller.py:96-126`, `check_agreement`, `rtol = 1e-6`), which
says nothing about whether the *model* loops converged. So PROCESS's cold state may be
unconverged, and the port -- which drives each declared block to its own tolerance -- may
be the more consistent side. That is a real possibility, and **arguing it either way is
what §16.3 records three wrong answers for**, so it is measured instead:
`ColdState.unsettled` runs PROCESS's own `_call_models_once` `EXTRA_PASSES` further times
past where `check_agreement` stopped it and reports every field that still moves, and
`ColdState.drift` is the largest such motion.

The answer is a *size*, not a yes/no, and it has to be read per configuration:

| configuration | passes | fields still moving | worst drift | smallest disagreement |
|---|---|---|---|---|
| `stellarator_helias` | 5 | 88 | `7.00e-07` | `1.21e-04` |
| `large_tokamak_nof` | 6 | 15 | `2.74e-08` | `1.17e-06` |
| `low_aspect_ratio_DEMO` | 5 | 0 | `0` | `1.04e-06` |
| `large_tokamak_eval` | 6 | 0 | `0` | `1.10e-06` |

PROCESS's cold state is an *exact* fixed point on two of the four and creeps at `7e-07`
and `2.7e-08` on the others -- in every case at least 40x below the smallest disagreement
reported for that same configuration, and six to seven orders below the largest.
Unconvergedness is therefore ruled out as the explanation for anything in the pin, on a
measurement re-taken on every run rather than on this paragraph.

Two of the three findings below were then confirmed by substitution rather than by
argument -- the discipline `_audit/optimise_design.md` §16.3 asks for. See
`_audit/optimise_design.md` §17, which is this stage's record. (There is no
`_audit/units/` record: that tree is the per-*model-unit* one and `test_registry_
coverage.py` requires a `unit_registry.md` row for everything in it, which a harness
stage has no business having -- `boundary.py` and `mda_harness.py` are documented the
same way, in `_audit/`'s flat design documents.)

The pin
-------
`reference_cold_start.txt` holds, per configuration, the agreement count, the error
count and every disagreeing and output-pass-only `VarPath`. Every pinned disagreement must
carry a reason in `ACCEPTED` -- a pin without one is refused by `check_reasons`, because a
silently pinned disagreement is indistinguishable from a defect nobody looked at, which is
the failure this whole module exists to end. Regenerate (never hand-edit) with::

    $PY -m functional_process.cold_start --write
"""

from __future__ import annotations

import os
import pickle
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

CONFIGURATIONS = (
    "tests/regression/input_files/stellarator_helias.IN.DAT",
    "tests/regression/input_files/large_tokamak_nof.IN.DAT",
    "tests/regression/input_files/low_aspect_ratio_DEMO.IN.DAT",
    "tests/regression/input_files/large_tokamak_eval.IN.DAT",
)
"""The reference configurations this stage is measured on.

**Not "every one that assembles" any more**, and the gap is now three files rather than
the two switch refusals this used to name. `IFE.IN.DAT` is out of scope entirely
(`ife == 1`, a whole unported device). The other three assemble and are absent for
reasons that are *not* assembly:

- `spherical_tokamak_eval` and `st_regression` were refused on
  `tf_stress_arm == (0, 1, 0)` (`extended_plane_strain`) until the port of it landed on
  2026-08-31; `machine_survey.assembly_verdict` reports **ASSEMBLES** for both as of that
  day. They are absent here only because this stage has never been run on them -- adding
  them is a measurement nobody has taken, not a refusal. **This paragraph has now been
  wrong twice in two days for the same reason** (it said `i_tf_turn_type == 2` after the
  CroCo wave closed that, and `tf_stress_arm` after the stress port closed that), which
  is why it now names `assembly_verdict` as the authority instead of a switch: a refusal
  is only valid against the tree that was current when it was written.
- `helias_5b`, below.

`helias_5b` assembles (`_audit/next_steps.md` §20) and is deliberately not here yet,
and unlike the two above it has been measured and has a reason.
Measured, on 2026-08-31, rather than assumed either way: its cold report is **74
disagreements**, and 49 of those are exactly the reference stellarator's own two
accepted causes (`STELLARATOR_ARM_ORDER_ROWS`, `VACUUM_DUCT_ROWS`) -- a strict subset, so
those would cost one `_because` row each. The other **25 are one new chain and it is a
port defect, not an accepted disagreement**: `helias_5b.IN.DAT:121` sets
`i_p_coolant_pumping = 0` (`USER_INPUT`) with the pump powers given directly
(`120 + 56 = 176` MW for FW+blanket, `24` MW for the divertor), while
`stellarator_helias.IN.DAT:198` sets `1` (`FRACTION_OF_HEAT`) --- and
`models/stellarator/stellarator_fwbs_s2.py` **always computes as if the value were
`FRACTION_OF_HEAT`**, a drop its own docstring records as *"the absence of the
computation, not a second formula to port"*, which was true of every machine that
existed when it was written. So the port returns `15.58` MW where PROCESS reads `176.0`,
and the error runs downstream through `.power.*` and `.heat_transport.*` into
`.costs.concost` -- **this file's own objective** (`i_figure_merit = 7`), off by `+3.1 %`
at a fixed cold design, and into `c16`, one of its five active constraints, through
`p_plant_electric_net_mw` (off by `+11.1 %`).

Adding the row therefore needs the `USER_INPUT` arm ported, not 25 `ACCEPTED` entries:
an entry here is a reason a disagreement is *understood*, and pinning a live unported
switch arm as acceptable is the exact ambiguity `ACCEPTED`'s own docstring exists to
end. Recorded on the list rather than papered over.

Named as repository-relative paths, resolved by `_resolve`, exactly as
`boundary.TOKAMAK_INPUT_FILE` is.
"""

PIN = os.path.join(os.path.dirname(__file__), "reference_cold_start.txt")
"""The pinned cold-point agreement -- see `rows` for the four kinds of line it holds.

Generated by `$PY -m functional_process.cold_start --write`; never hand-edited.
"""

EXTRA_PASSES = 3
"""How many further `Caller._call_models_once` passes `cold_state` runs past the point
`check_agreement` stopped PROCESS, to decide whether its cold state is actually settled.

Three rather than one because a two-cycle would be invisible to one extra pass, and
rather than thirty because thirty buys nothing: measured on `large_tokamak_nof`, thirty
further passes move the burn time, the TF temperature margin, `dlscal` and `.costs.coe` by
**exactly zero** to eight significant figures, and the first extra pass already says so. A
pass is ~0.1 s on a tokamak and ~0.8 s on the stellarator, which re-reads its
`.stella_conf.json` inside every model call, so three is also what keeps the stellarator's
whole cold evaluation under seven seconds.
"""

SETTLED_RTOL = 1e-9
"""What counts as "still moving" in `ColdState.unsettled`.

Three orders tighter than `check_agreement`'s own `rtol = 1e-6`, deliberately: the
question is not whether PROCESS would stop here (it did) but whether the state it stopped
at is a fixed point of its own map, and a `1e-6` test could not distinguish "settled"
from "drifting slowly enough to pass its own convergence check". Loose enough to
ignore the last-bit noise of a `CoolProp` table lookup.

**The list is not empty on two of the four configurations, and that is the point of
reporting the size of the drift rather than a yes/no.** See the table in this module's
docstring for the measured numbers; the short of it is `7.00e-07` at worst on the
stellarator, `2.74e-08` on `large_tokamak_nof`, exactly zero on the other two, against
smallest reported disagreements of `1.21e-04` and `1.17e-06`. "PROCESS has not converged"
is therefore ruled out as the explanation for any row in the pin -- not asserted,
measured, and re-measured on every run.
"""

BOOKKEEPING = frozenset({("numerics", "n_model_calls")})
"""Fields excluded from `ColdState.unsettled` that the *measurement itself* moves.

`.numerics.n_model_calls` is a counter `Caller._call_models_once` increments
(`caller.py`), so the `EXTRA_PASSES` probe raises it by exactly `EXTRA_PASSES` every time
and it appeared as the one "still moving" field on the two configurations that are
otherwise exact fixed points. Excluding an observer effect is not the same as excluding
an inconvenient result, and the distinction is worth keeping narrow: this set holds
*only* fields whose motion is caused by the probe, and it holds one.
"""


@dataclass
class ColdState:
    """PROCESS's own cold start on one input file: the seed, the answer, and two
    measurements about the answer.

    Held as one object because every part of it comes from the same single pipeline
    evaluation and re-deriving any of it means paying for that evaluation again.
    """

    seed: object
    """The `DataStructure` after `init_process` and **before any model has run**.

    `SingleRun.__init__` already performs that init, so an un-run instance *is* the input
    file's own starting state; this is a `deepcopy` of it taken before the evaluation,
    for the same reason `sand_harness.ReferenceRun.cold` exists -- there is no way to
    recover it afterwards, because `set_scaled_iteration_variable` overwrites every
    iteration variable in place on the first model call.
    """

    process: object
    """The same `DataStructure` after `load_iteration_variables` and one
    `Evaluators.fcnvmc1` at the cold `x` -- PROCESS's own answer at the design the input
    file starts from, computed by PROCESS's own pipeline and nothing else."""

    written: frozenset
    """`{(area, field)}` the pass moved -- PROCESS's measured write set.

    The same measurement `boundary.computed_by_process` returns, and now the same code:
    that function delegates here rather than running its own `SingleRun`, so the two
    halves of the missing-producer question (declaration and value) cannot drift apart or
    be measured at two different points.
    """

    passes: int
    """How many times `Caller._call_models_once` ran inside the evaluation -- PROCESS's
    Gauss-Seidel pass count at the cold `x`, capped at ten by `caller.py:99`."""

    unsettled: tuple
    """`(area, field, before, after, rel)` per field that still moves when PROCESS's
    pipeline is run `EXTRA_PASSES` further times, worst-relative-motion first.

    **This is the discriminator, and it is a size rather than a yes/no.** If PROCESS's
    cold state still moves by `d`, then no disagreement of order `d` can be attributed to
    the port -- `check_agreement` stopped on the objective and the constraints while the
    model loops were still going, so PROCESS's number is not an answer yet. A
    disagreement orders of magnitude *above* `d` cannot be explained that way: it is a
    difference between two nearly-converged states of two different maps, i.e. a
    difference in the maps.

    `drift` is that `d`. Measured: `1.30e-08` on the stellarator, `2.59e-09` on
    `large_tokamak_nof`, exactly zero on the other two -- against a smallest reported
    disagreement of `1.1e-06`. See this module's docstring, question 3.
    """

    @property
    def drift(self) -> float:
        """The largest relative motion in `unsettled` -- how far PROCESS's cold state
        still is from being a fixed point of its own pipeline. `0.0` when it is one.
        """
        return max((row[4] for row in self.unsettled), default=0.0)


def _resolve(name: str) -> str:
    """`name` as an absolute path, read relative to the repository root when relative --
    `run_mda_harness._resolve`'s rule, restated so this module has no import of it.
    """
    from pathlib import Path

    path = Path(name)
    root = Path(__file__).resolve().parent.parent
    return str(path if path.is_absolute() else (root / path).resolve())


def _scratch_copy(input_file: str) -> str:
    """`input_file` copied into a fresh directory with its `.stella_conf.json`, so the
    `OUT.DAT`/`MFILE.DAT` a `SingleRun` writes do not land in the repository.

    Not a `TemporaryDirectory` context manager, for `sand_harness._scratch_copy`'s
    reason: PROCESS re-reads the stellarator preset file on every model call, so the
    directory has to outlive any `with` block.
    """
    import shutil
    import tempfile
    from pathlib import Path

    source = Path(input_file)
    directory = Path(tempfile.mkdtemp())
    shutil.copy(source, directory / source.name)
    stem = (
        source.name[: -len(".IN.DAT")]
        if source.name.endswith(".IN.DAT")
        else source.stem
    )
    companion = source.parent / f"{stem}.stella_conf.json"
    if companion.exists():
        shutil.copy(companion, directory / companion.name)
    return str(directory / source.name)


def _snapshot(data) -> dict:
    """Every numeric field of every area of `data`, as float arrays.

    Lifted from `boundary.computed_by_process`, which now calls this through
    `cold_state`. Fields that will not convert (strings, `None`, objects) are skipped
    rather than reported: they cannot move numerically, so they cannot be part of a write
    set measured by numeric change.
    """
    out = {}
    for area_name in dir(data):
        if area_name.startswith("_"):
            continue
        area = getattr(data, area_name)
        if not hasattr(area, "__dataclass_fields__"):
            continue
        for field_name in area.__dataclass_fields__:
            try:
                out[area_name, field_name] = np.array(
                    getattr(area, field_name), dtype=float, copy=True
                )
            except (TypeError, ValueError, AttributeError):  # noqa: PERF203
                continue
    return out


def _moved(before: dict, after: dict, rtol: float) -> frozenset:
    """Keys whose value differs between the two snapshots, shape changes included.

    `nan`/`inf` are mapped to `0.0` on both sides before comparison: a field that is
    `nan` in both snapshots has not moved, and `np.allclose`'s `equal_nan` would have to
    be set per call to say so.
    """
    clean = {"nan": 0.0, "posinf": 0.0, "neginf": 0.0}
    moved = set()
    for key, was in before.items():
        now = after.get(key)
        if now is None or now.shape != was.shape:
            moved.add(key)
            continue
        if not np.allclose(
            np.nan_to_num(was, **clean),
            np.nan_to_num(now, **clean),
            rtol=rtol,
            atol=0.0,
        ):
            moved.add(key)
    return frozenset(moved)


CACHE_VERSION = "cold-v1"
"""Bumped when `ColdState`'s *contents* change, so an old pickle can never be read back
under a key whose meaning has moved -- `mda_harness._CACHE_VERSION`'s discipline, and its
docstring records what a stale cross-read costs."""


def cold_state(input_file: str, use_cache: bool = True) -> ColdState:
    """Run PROCESS's pipeline **once**, at the input file's own starting design, and
    return everything the cold comparison needs.

    The evaluation is `Evaluators.fcnvmc1` rather than a bare `Caller._call_models_once`
    on purpose: `fcnvmc1` is what the optimiser itself calls, so this is PROCESS's cold
    state as PROCESS's own solver sees it on iteration zero -- including
    `load_iteration_variables`/`set_scaled_iteration_variable`'s write-back of the design
    vector, and including `call_models`' up-to-ten-pass idempotence loop.

    Cached on disk beside `mda_harness`'s converged runs and keyed the same way (the
    input files plus the state of `process/`). The saving is real but modest, and worth
    stating rather than assuming: measured uncached, `stellarator_helias` costs **6.3 s**
    -- PROCESS re-reads its `.stella_conf.json` on every model call -- and the three
    tokamaks 0.3-0.6 s each. What the cache actually buys is that
    `boundary.computed_by_process` and this stage no longer pay separately for the same
    evaluation. `FP_HARNESS_NO_CACHE=1` forces the run.
    """
    from pathlib import Path

    from functional_process.mda_harness import CACHE_DIR, _cache_key

    input_file = _resolve(input_file)
    use_cache = use_cache and not os.environ.get("FP_HARNESS_NO_CACHE")
    cached = (
        Path(CACHE_DIR) / f"{CACHE_VERSION}-{_cache_key(input_file)}.pkl"
        if use_cache
        else None
    )
    if cached is not None and cached.exists():
        with cached.open("rb") as handle:
            return pickle.load(handle)  # noqa: S301 -- our own file, written just below

    state = _measure(input_file)
    if cached is not None:
        Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
        partial = cached.with_suffix(".partial")
        with partial.open("wb") as handle:
            pickle.dump(state, handle)
        partial.replace(cached)
    return state


def _measure(input_file: str) -> ColdState:
    """`cold_state`'s uncached body: one `SingleRun`, one `fcnvmc1`, the extra passes.

    Imports are deferred because `process.main` pulls in the whole of PROCESS and this
    module is imported by tests that never run it.
    """
    import copy

    import process.core.caller as caller_module
    from process.core.solver.evaluators import Evaluators
    from process.core.solver.iteration_variables import load_iteration_variables
    from process.main import SingleRun

    run = SingleRun(_scratch_copy(input_file), "vmcon")
    data = run.data
    seed = copy.deepcopy(data)
    before = _snapshot(data)

    n = int(data.numerics.n_iteration_variables)
    m = int(data.numerics.n_equality_constraints) + int(
        data.numerics.n_inequality_constraints
    )
    load_iteration_variables(data)
    x = np.array(data.numerics.xcm[:n], dtype=float)

    # Counted by wrapping `_call_models_once` rather than by reading a field, because
    # PROCESS does not record its own pass count anywhere -- `call_models`' loop variable
    # is discarded. Restored in `finally`: leaving a patched method behind would make
    # every later measurement in the same interpreter wrong in a way nothing would catch.
    original = caller_module.Caller._call_models_once
    passes = [0]

    def counted(self, xc):
        passes[0] += 1
        return original(self, xc)

    caller_module.Caller._call_models_once = counted
    try:
        Evaluators(run.models, data, x).fcnvmc1(n, m, x, 1)
    finally:
        caller_module.Caller._call_models_once = original

    after = _snapshot(data)
    written = _moved(before, after, rtol=1e-12)

    # Has PROCESS actually settled, or did `check_agreement` merely stop looking? Run
    # its own map further from where it stopped and see what moves. `Caller` is
    # constructed fresh rather than reached through `run.models`, which does not hold
    # one -- `Evaluators` builds its own and discards it.
    caller = caller_module.Caller(run.models, data)
    for _ in range(EXTRA_PASSES):
        caller._call_models_once(x)
    settled = _snapshot(data)
    unsettled = tuple(
        sorted(
            (
                (
                    area,
                    name,
                    float(np.ravel(after[area, name])[0]),
                    float(np.ravel(settled[area, name])[0]),
                    _relative(after[area, name], settled[area, name]),
                )
                for area, name in _moved(after, settled, rtol=SETTLED_RTOL) - BOOKKEEPING
                if (area, name) in settled
                and settled[area, name].shape == after[area, name].shape
                and settled[area, name].size
            ),
            # Worst first: `drift` reads row 0's relative motion and a reader who sees
            # only the truncated list in `summary` must see the largest, not the
            # alphabetically first.
            key=lambda row: -row[4],
        )
    )
    return ColdState(
        seed=seed, process=data, written=written, passes=passes[0], unsettled=unsettled
    )


def _relative(was, now) -> float:
    """Largest elementwise relative change between two snapshots of one field."""
    was, now = np.nan_to_num(np.asarray(was)), np.nan_to_num(np.asarray(now))
    if was.size == 0:
        return 0.0
    denom = np.where(was == 0.0, 1.0, np.abs(was))
    return float(np.max(np.abs(now - was) / denom))


# --------------------------------------------------------------- the comparison


@dataclass
class ColdReport:
    """One configuration's cold-point result: `mda_harness.ComparisonReport` plus the
    split only a cold measurement can make.
    """

    input_file: str
    state: ColdState
    comparison: object
    """The `ComparisonReport` from `mda_harness.compare(graph, state.process,
    seed=state.seed)` -- every bucket that stage defines, unchanged."""

    output_pass_only: list = field(default_factory=list)
    """Disagreements on a field PROCESS's **solve** pass never writes, so its "expected"
    value is `init_process`' default and not an answer. Split out of `disagreements`, not
    counted as either agreement or defect -- see this module's docstring, question 2."""

    @property
    def real(self) -> list:
        """The disagreements that are actually a comparison -- what the pin holds."""
        return [
            d
            for d in self.comparison.disagreements
            if d not in self.output_pass_only  # identity, not value: same objects
        ]

    def summary(self) -> str:
        name = os.path.basename(self.input_file)
        lines = [
            f"=== {name}",
            f"  PROCESS Gauss-Seidel passes at the cold x: {self.state.passes}",
            f"  fields PROCESS writes in that pass: {len(self.state.written)}",
            f"  still moving after {EXTRA_PASSES} further passes: "
            f"{len(self.state.unsettled)} field(s), worst {self.state.drift:.2e}"
            + (
                ""
                if self.state.unsettled
                else "  (an exact fixed point of PROCESS's own map)"
            ),
            f"  agreements: {self.comparison.agreements}"
            f" (both-sides-zero: {len(self.comparison.trivial_agreements)})",
            f"  disagreements: {len(self.real)}",
            f"  output-pass-only (nothing to compare): {len(self.output_pass_only)}",
            f"  errors: {len(self.comparison.errors)}",
        ]
        for area, name_, was, now, rel in self.state.unsettled[:5]:
            lines.append(
                f"    still moving: .{area}.{name_} {was!r} -> {now!r} {rel:.2e}"
            )
        for d in sorted(self.real, key=lambda d: -d.rel_diff):
            reason = ACCEPTED.get((name, d.var.path_str()))
            lines.append(
                f"    {d.rel_diff:11.3e}  {d.var.path_str()}  got={d.got!r} "
                f"expected={d.expected!r}{d.where}"
                + ("" if reason else "   <-- NO REASON PINNED")
            )
        for d in sorted(self.output_pass_only, key=lambda d: -d.rel_diff):
            lines.append(
                f"    (output-pass-only) {d.var.path_str()} port={d.got!r} "
                f"PROCESS's solve pass leaves {d.expected!r}"
            )
        return "\n".join(lines)


def cold_report(input_file: str, state: ColdState | None = None) -> ColdReport:
    """Assemble the machine `input_file` describes, run its MDA from the cold seed, and
    diff every variable it owns against PROCESS's own cold answer.

    The graph is built by `machine_from_indat` from the file itself, never described
    here -- `run_mda_harness`'s own docstring records what spelling a machine's switches
    out in a harness cost the last time it was done.
    """
    from functional_process.indat import graph_for, machine_from_indat
    from functional_process.mda_harness import compare

    input_file = _resolve(input_file)
    state = cold_state(input_file) if state is None else state
    graph = graph_for(machine_from_indat(input_file))
    comparison = compare(graph, state.process, seed=state.seed)
    report = ColdReport(input_file=input_file, state=state, comparison=comparison)
    report.output_pass_only = [
        d
        for d in comparison.disagreements
        if _area_field(d.var) is not None and _area_field(d.var) not in state.written
    ]
    return report


def _area_field(var) -> tuple[str, str] | None:
    """`(area, field)` for a plain `.area.field` `VarPath`, `None` for anything else.

    A minted path (`^hat.*`, `^cond.*`) and a per-element path have no single
    `DataStructure` field of that shape, so neither can be looked up in a write set
    measured over `dataclass` fields; returning `None` keeps them in the ordinary
    disagreement bucket rather than silently exempting them.
    """
    keys = var.path_str().lstrip(".").split(".")
    return (keys[0], keys[1]) if len(keys) == 2 and "[" not in keys[1] else None


# --------------------------------------------------------------- accepted disagreements


STELLARATOR = "stellarator_helias.IN.DAT"
TOKAMAK_NOF = "large_tokamak_nof.IN.DAT"
TOKAMAK_DEMO = "low_aspect_ratio_DEMO.IN.DAT"
TOKAMAK_EVAL = "large_tokamak_eval.IN.DAT"
"""The four `CONFIGURATIONS`, by the base name a pin row and an `ACCEPTED` key use."""

TF_STRESS_LANDED = (
    "**`stresscl` landed the same day this stage was written, and closed its own "
    "entry.** This block used to hold seven rows -- the CICC critical-surface chain "
    "and both temperature margins -- caused by `.tfcoil.str_wp` having no producer and "
    "the cold seed handing it `DataStructure()`'s `0.0`, the *peak* of the Nb3Sn fit. "
    "Registry row 55 (`models/tfcoil/stress.py`) owns it now and all seven agree; the "
    "prediction recorded here (substituting PROCESS's cold `0.0018442328` removes "
    "exactly those seven and adds none) was measured, and the producer landing "
    "reproduced it. `large_tokamak_nof` cold went 631 -> 646 agreements.\n\n"
    "**What it left is one row, and the port is the correct side.** "
    "`.tfcoil.insstrain` is a new output of the landed node. At PROCESS's converged "
    "design port and PROCESS agree to nine digits (`-0.00591260699` both). Cold they "
    "differ by 6.2e-03 relative: port `-0.00775040112` against PROCESS's own "
    "`-0.007703004533833493` after one pipeline pass. That is the ordinary "
    "two-fixed-points-of-two-maps case this module's `drift` measurement settles for "
    "the rest of the file -- PROCESS's cold state is settled here (worst motion "
    "2.74e-08 over three further Gauss-Seidel passes, against this row's 6.2e-03) -- "
    "and it is a *smaller* disagreement than the seven it replaced."
)

TF_STRESS_ROWS = (".tfcoil.insstrain",)
"""What survived `stresscl` landing. The seven rows this used to name now agree."""

NOH_WRONG = (
    "**A real port defect this stage found, and it is not fixed here.** "
    "`PFCoil.induct` splits the CS into "
    "`noh = ceil(2 * z_pf_coil_upper[CS] / (r_pf_coil_outer[CS] - "
    "r_pf_coil_inner[CS]))` pancake segments (`pfcoil.py:1758-1765`), and every "
    "inductance it returns depends "
    "on that integer -- so `ind_pf_cs_plasma_mutual` is piecewise constant and "
    "discontinuous "
    "in the CS geometry. `models/pfcoil/inductance.py` pins `NOH = 30`, measured on "
    "`large_tokamak_eval`, where the ratio is `29.028`. Measured on the other two:\n\n"
    "| configuration | ratio cold | `noh` cold | ratio converged | `noh` converged |\n"
    "|---|---|---|---|---|\n"
    "| `large_tokamak_eval` | 29.028 | **30** | 29.028 | **30** |\n"
    "| `large_tokamak_nof` | 31.746 | 32 | 26.867 | 27 |\n"
    "| `low_aspect_ratio_DEMO` | 27.010 | 28 | 26.407 | 27 |\n\n"
    "So the port is on the wrong piece at *both* the cold and the converged design of "
    "both configurations, and **no single constant is right at both** -- which is the "
    "point, and why the fix is not a different number. Confirmed by substitution: with "
    "`NOH = 32` on `large_tokamak_nof` the cold result goes **631/82 to 662/51**, 31 of "
    "these rows disappear outright and the other 34 fall by one to two orders "
    "(`.times.t_plant_pulse_burn` `3.76e-04 -> 8.89e-06`, "
    "`.pf_coil.ind_pf_cs_plasma_mutual` `5.10e-04 -> 6.55e-05`) to at most `6.55e-05`, "
    "below `PicardDriver`'s own `1e-4` tolerance. `_audit/units/models/pfcoil/"
    "inductance.md` § 'noh is a step function of the CS geometry' already files 'a "
    "structural integer that the solve moves' as an open convention question; this is "
    "the measurement of what it costs."
)

NOH_ROWS_NOF = (
    ".buildings.cryvol",
    ".costs.bktcycles",
    ".costs.c217",
    ".costs.c2174",
    ".costs.c2252",
    ".costs.c22521",
    ".costs.c22526",
    ".costs.c2253",
    ".costs.c226",
    ".costs.c2262",
    ".costs.c2263",
    ".costs.c24",
    ".costs.c242",
    ".costs.c243",
    ".costs.coe",
    ".costs.coecap",
    ".costs.coefuelt",
    ".costs.coeoam",
    ".costs.cpfact",
    ".costs.cppa",
    ".heat_transport.f_p_plant_electric_recirc",
    ".heat_transport.helpow",
    ".heat_transport.p_cryo_plant_electric_mw",
    ".heat_transport.p_plant_electric_net_mw",
    ".heat_transport.p_plant_electric_recirc_mw",
    ".heat_transport.p_plant_secondary_heat_mw",
    ".heat_transport.pacpmw",
    ".heat_transport.peakmva",
    ".heat_transport.tlvpmw",
    ".pf_coil.b_pf_coil_peak[0]",
    ".pf_coil.b_pf_coil_peak[1]",
    ".pf_coil.bpf2[0]",
    ".pf_coil.bpf2[1]",
    ".pf_coil.c_pf_coil_turn",
    ".pf_coil.c_pf_cs_coil_flat_top_ma",
    ".pf_coil.c_pf_cs_coils_peak_ma",
    ".pf_coil.f_c_pf_cs_peak_time_array",
    ".pf_coil.f_j_cs_start_end_flat_top",
    ".pf_coil.ind_pf_cs_plasma_mutual",
    ".pf_coil.m_pf_coil_conductor",
    ".pf_coil.m_pf_coil_structure",
    ".pf_coil.p_pf_electric_supplies_mw",
    ".pf_coil.pfcaseth",
    ".pf_coil.vs_cs_pf_total_burn",
    ".pf_coil.vs_cs_pf_total_pulse",
    ".pf_power.ensxpfm",
    ".pf_power.peakpoloidalpower",
    ".pf_power.poloidalpower",
    ".pf_power.spsmva",
    ".physics.vs_plasma_burn_required",
    ".physics.vs_plasma_total_required",
    ".power.e_plant_net_electric_pulse_kwh",
    ".power.e_plant_net_electric_pulse_mj",
    ".power.p_cryo_plant_electric_profile_mw",
    ".power.p_pf_electric_supplies_profile_mw",
    ".power.p_plant_core_systems_elec_mw",
    ".power.p_plant_electric_net_profile_mw",
    ".power.qac",
    ".power.qmisc",
    ".tfcoil.cryo_cool_req",
    ".times.t_plant_pulse_burn",
    ".times.t_plant_pulse_plasma_present",
    ".times.t_plant_pulse_total",
    "^hat.pf_coil.ind_pf_cs_plasma_mutual",
    "^hat.times.t_plant_pulse_burn",
)
"""`large_tokamak_nof`'s `noh` chain: the inductance matrix, everything the volt-second
balance carries from it (burn time, pulse durations), the PF power conversion the peak
currents size, and the cost and cryogenic accounts below those. Sixty-five rows from one
integer."""

NOH_ROWS_DEMO = (
    ".costs.bktcycles",
    ".costs.c22",
    ".costs.c22521",
    ".costs.c22526",
    ".costs.c242",
    ".costs.c243",
    ".costs.coecap",
    ".heat_transport.pacpmw",
    ".heat_transport.peakmva",
    ".heat_transport.tlvpmw",
    ".pf_coil.c_pf_coil_turn",
    ".pf_coil.c_pf_cs_coil_flat_top_ma",
    ".pf_coil.f_c_pf_cs_peak_time_array",
    ".pf_coil.f_j_cs_start_end_flat_top",
    ".pf_coil.ind_pf_cs_plasma_mutual",
    ".pf_coil.p_pf_electric_supplies_mw",
    ".pf_coil.vs_cs_pf_total_burn",
    ".pf_power.ensxpfm",
    ".pf_power.peakpoloidalpower",
    ".pf_power.poloidalpower",
    ".pf_power.spsmva",
    ".physics.vs_plasma_burn_required",
    ".power.e_plant_net_electric_pulse_kwh",
    ".power.e_plant_net_electric_pulse_mj",
    ".power.p_pf_electric_supplies_profile_mw",
    ".times.t_plant_pulse_burn",
    ".times.t_plant_pulse_plasma_present",
    ".times.t_plant_pulse_total",
    "^hat.pf_coil.ind_pf_cs_plasma_mutual",
    "^hat.times.t_plant_pulse_burn",
)
"""The same chain on `low_aspect_ratio_DEMO`, and shorter for the reason the table in
`NOH_WRONG` predicts: its `noh` is off by one (28 against 30) where
`large_tokamak_nof`'s is off by two, so the same chain is an order smaller and half of
it lands inside `compare`'s `rtol`."""

STELLARATOR_ARM_ORDER = (
    "**Not a port defect: PROCESS's solve pass and its report pass compute different "
    "geometries, and the port models the report pass.** `Stellarator.run(output=False)` "
    "runs `st_coil` then `st_build`; `Stellarator.run(output=True)` runs `st_build` "
    "then "
    "`st_coil` (`stellarator.py:141-146` against `:159-165`), and `.build.z_tf_inside_"
    "half` is written by one and read by the other. Measured directly rather than "
    "inferred: at the cold design PROCESS's solve pass leaves `3.611990999471611` and "
    "**PROCESS's own output-pass order leaves `5.513665371874896`, which is the port's "
    "answer to sixteen digits**; `.buildings.a_plant_floor_effective`, which "
    "`Buildings.run` computes from it, is `378222.11` and `424256.91004` against the "
    "port's `424256.91005`. Everything below is that one geometry through the buildings "
    "volumes, the site accounts and the AC power and cost chain.\n\n"
    "`mda_harness.EXPLAINED_DISAGREEMENTS`'s "
    "`.heat_transport.p_plant_electric_base_total_mw` entry is the same finding seen "
    "from the other side and is the authority on it: at the *converged* point PROCESS "
    "stores the report-pass geometry, so the port agrees on `z_tf_inside_half` and "
    "disagrees only on the one field the report pass does not recompute. Cold, PROCESS "
    "stores the solve-pass geometry, so the disagreement is the whole chain. **PROCESS "
    "is not self-consistent between its two arms and the port is self-consistent with "
    "one of them**, which is why neither point can be made to agree everywhere and why "
    "this is pinned rather than chased."
)

STELLARATOR_ARM_ORDER_ROWS = (
    ".build.z_tf_inside_half",
    ".buildings.a_plant_floor_effective",
    ".buildings.rbvol",
    ".buildings.rmbvol",
    ".buildings.volnucb",
    ".buildings.volrci",
    ".buildings.wsvol",
    ".costs.c21",
    ".costs.c212",
    ".costs.c214",
    ".costs.c2141",
    ".costs.c2142",
    ".costs.c22",
    ".costs.c226",
    ".costs.c2262",
    ".costs.c227",
    ".costs.c2273",
    ".costs.c2274",
    ".costs.c24",
    ".costs.c242",
    ".costs.c243",
    ".costs.capcost",
    ".costs.ccont",
    ".costs.cdirt",
    ".costs.cindrt",
    ".costs.coe",
    ".costs.coecap",
    ".costs.coefuelt",
    ".costs.coeoam",
    ".costs.concost",
    ".costs.cppa",
    ".costs.moneyint",
    ".heat_transport.f_p_plant_electric_recirc",
    ".heat_transport.fachtmw",
    ".heat_transport.p_plant_electric_base_total_mw",
    ".heat_transport.p_plant_electric_net_mw",
    ".heat_transport.p_plant_electric_recirc_mw",
    ".heat_transport.p_plant_secondary_heat_mw",
    ".heat_transport.tlvpmw",
    ".power.e_plant_net_electric_pulse_kwh",
    ".power.e_plant_net_electric_pulse_mj",
    ".power.p_plant_core_systems_elec_mw",
    ".power.p_plant_electric_base_total_profile_mw",
    ".power.p_plant_electric_net_profile_mw",
)
"""The 44 rows `.build.z_tf_inside_half` reaches on the stellarator, worst `5.27e-01` on
the geometry itself and `1.56e-01` on `.buildings.volrci`."""

PF_TURNS_DEAD_TAIL = (
    "The array's dead tail, exactly as at the converged point -- "
    "`mda_harness.EXPLAINED_DISAGREEMENTS`'s entry for this path is the authority. "
    "`PFCoilSizes` writes a structural `0.0` past the plasma circuit where PROCESS "
    "keeps `pfcoil.py:605-608`'s `first_call` bootstrap residue of `100.0`, which "
    "nothing ever "
    "overwrites at indices no coil occupies. The live entries agree. The `^hat.*` twin "
    "is the same array as the PF cycle's minted unknown: one cause, two rows."
)

PF_TURNS_ROWS = (".pf_coil.n_pf_coil_turns", "^hat.pf_coil.n_pf_coil_turns")

VACUUM_DUCT_SOLVE = (
    "`VacuumOld`'s duct-diameter Newton solve, a deliberate solver-tolerance difference "
    "documented at `models/vacuum.py:262-271` and in "
    "`mda_harness.EXPLAINED_DISAGREEMENTS`, which is the authority on it. PROCESS stops "
    "the same iteration at a 1 % relative step (`process/models/vacuum.py:469-477`), so "
    "**PROCESS's own stopping rule admits ~1e-2 relative error in this field -- twenty "
    "to thirty times larger than anything measured here** -- and PROCESS is not ground "
    "truth for it. `dlscal` follows as `1.4x` the diameter's error (`dlscal ∝ d**1.4`), "
    "checked cold on `large_tokamak_nof`: `3.222e-04` and `4.510e-04`, and "
    "`1.4 * 3.222e-04 = 4.511e-04`. Accounts 224.3/224.4 are linear in the two, and "
    "224 is their sum. One cause, five rows, present on every machine that registers "
    "`VacuumOld` -- which is all four."
)

VACUUM_DUCT_ROWS = (
    ".costs.c224",
    ".costs.c2243",
    ".costs.c2244",
    ".vacuum.dia_vv_vacuum_ducts",
    ".vacuum.dlscal",
)

DRIVER_TOLERANCE = (
    "**Below the driver's own convergence tolerance, so it is not evidence about the "
    "model.** `PicardDriver` (`cottax.drivers`) stops when its iterate moves by less "
    "than `rtol = atol = 1e-4`, and `^hat.pf_coil.ind_pf_cs_plasma_mutual` is the "
    "loop-carried unknown of the PF cycle: it agrees to `1.49e-06`, **two orders better "
    "than the driver promises**, on 480 of its 484 entries exactly. `compare`'s "
    "`rtol = 1e-6` is tighter than any driven block's own tolerance, so a residue at "
    "`1e-6` inside or downstream of a `Drive` is the algorithm's convergence criterion "
    "showing through, not a difference in the arithmetic. The cost accounts below it "
    "carry the same `1.1e-06` through `PfMagnetCost` into `c22` and the capital-cost "
    "sum. `large_tokamak_eval` is the configuration where `NOH = 30` is *right* "
    "(see `NOH_WRONG`'s table), which is why this is all that is left of its PF chain."
)

DRIVER_TOLERANCE_ROWS_EVAL = (
    ".costs.c22",
    ".costs.capcost",
    ".costs.ccont",
    ".costs.cdirt",
    ".costs.cindrt",
    ".costs.coe",
    ".costs.coecap",
    ".costs.concost",
    ".costs.moneyint",
    "^hat.pf_coil.ind_pf_cs_plasma_mutual",
)


def _because(reason: str, mapping) -> dict:
    """`{(configuration, path): reason}` from `{configuration: paths}` -- so one cause is
    written once and still covers every row it explains, on every machine it explains it
    on.
    """
    return {
        (configuration, path): reason
        for configuration, paths in mapping.items()
        for path in paths
    }


ACCEPTED = {
    **_because(
        TF_STRESS_LANDED,
        {
            TOKAMAK_NOF: TF_STRESS_ROWS,
            TOKAMAK_DEMO: TF_STRESS_ROWS,
            TOKAMAK_EVAL: TF_STRESS_ROWS,
        },
    ),
    **_because(NOH_WRONG, {TOKAMAK_NOF: NOH_ROWS_NOF, TOKAMAK_DEMO: NOH_ROWS_DEMO}),
    **_because(STELLARATOR_ARM_ORDER, {STELLARATOR: STELLARATOR_ARM_ORDER_ROWS}),
    **_because(
        PF_TURNS_DEAD_TAIL,
        {
            TOKAMAK_NOF: PF_TURNS_ROWS,
            TOKAMAK_DEMO: PF_TURNS_ROWS,
            TOKAMAK_EVAL: PF_TURNS_ROWS,
        },
    ),
    **_because(
        VACUUM_DUCT_SOLVE,
        {
            STELLARATOR: VACUUM_DUCT_ROWS,
            TOKAMAK_NOF: VACUUM_DUCT_ROWS,
            TOKAMAK_DEMO: VACUUM_DUCT_ROWS,
            TOKAMAK_EVAL: VACUUM_DUCT_ROWS,
        },
    ),
    **_because(DRIVER_TOLERANCE, {TOKAMAK_EVAL: DRIVER_TOLERANCE_ROWS_EVAL}),
}
"""`{(configuration, written path): why it is pinned}`. **A pin with no entry is
refused**, by `check_reasons` and by the test that reads the pin.

The rule this table enforces is the one `_audit/optimise_design.md` §16 exists to
establish: in a bare list of paths, a disagreement somebody chased and a disagreement
nobody looked at are indistinguishable, and the twenty-two missing producers lived for
weeks inside exactly that ambiguity. Every entry names which of the three questions in
this module's docstring it answers and cites the measurement it rests on -- and two of
the six were settled by substituting the suspected cause and re-measuring, rather than
by argument, because §16.3 records three persuasive arguments that were all wrong.

**Keyed on `(configuration, path)`, not on the path alone**, because the *cause* is
per-machine and merging them would lie about it: `.costs.coe` is off by `3.4e-02` on the
stellarator through the report-pass geometry and by `2.2e-04` on `large_tokamak_nof`
through `noh`, and one entry covering both would have to be vague enough to cover a
future third cause too.

**Two of the six entries name a defect rather than excusing one.** `NOH_WRONG` is the
port being wrong on two of the four configurations, and `TF_STRESS_UNPORTED` is the cost
of the one producer still missing. Pinning them is how they stay visible and bounded;
it is not a claim that they are acceptable.
"""


def check_reasons(report: ColdReport) -> tuple[str, ...]:
    """Disagreeing paths in `report` with no `ACCEPTED` entry: what a caller refuses on.

    Returned rather than raised so a caller can report every configuration's missing
    reasons at once instead of stopping at the first.
    """
    name = os.path.basename(report.input_file)
    return tuple(
        sorted({
            d.var.path_str()
            for d in report.real
            if (name, d.var.path_str()) not in ACCEPTED
        })
    )


# --------------------------------------------------------------- the pin


def rows(report: ColdReport) -> tuple[str, ...]:
    """`report` as pin lines, in a stable order.

    Four kinds, each prefixed by the configuration's file name so one file can hold
    every machine (unlike `boundary.py`'s two separate pins, which exist because a
    *boundary* is a property of one graph; a cold-agreement count is a property of one
    run and the four are read together):

    - `agree <n>`   -- how many owned variables reproduced PROCESS's cold answer.
    - `errors <n>`  -- how many could not be compared at all.
    - `off <path>`  -- one per disagreement, sorted by written name so an unrelated
      node landing elsewhere cannot reorder the file.
    - `nocompare <path>` -- one per output-pass-only variable.

    **`errors` is pinned as a count and that is the point of including it.** An entry
    there is a variable that was neither passed nor failed -- a mint with no
    `DataStructure` field, or a shape mismatch -- so it is the one bucket in which a
    regression is *silent*, which is the failure mode `ComparisonReport.unaccounted`'s
    docstring records costing a real wrong answer its visibility. The cold tokamaks sit
    at 22 against the warm run's 20, and the two extra are the same `Physics.outplas`
    cause as `nocompare`, seen as a shape rather than a value: PROCESS's solve pass never
    calls `calculate_effective_charge_ionisation_profiles`, so
    `.physics.n_charge_plasma_effective_profile` is `(0,)` in the cold structure against
    the port's `(201,)`, and a shape mismatch is not comparable rather than wrong.

    The relative difference is deliberately **not** pinned. It is reported by `summary`
    and it moves in the last digits with any upstream change; a pin that held it would
    have to be regenerated for reasons that are not regressions, and a pin regenerated
    routinely is a pin nobody reads.
    """
    name = os.path.basename(report.input_file)
    lines = [
        f"{name} agree {report.comparison.agreements}",
        f"{name} errors {len(report.comparison.errors)}",
    ]
    lines += sorted(f"{name} off {d.var.path_str()}" for d in report.real)
    lines += sorted(
        f"{name} nocompare {d.var.path_str()}" for d in report.output_pass_only
    )
    return tuple(lines)


def write_pin(all_rows: Iterable[str], path: str = PIN) -> None:
    """Regenerate the pin. Generated, never typed -- `boundary.write_pin`'s rule."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            "# Cold-point agreement: what each machine's graph computes for itself,\n"
            "# with nothing seeded from an answer PROCESS already found. Generated by\n"
            "# `$PY -m functional_process.cold_start --write`; do not hand-edit.\n"
            "# `agree` may only go up; `off`, `nocompare` and `errors` may only shrink;\n"
            "# every `off` path must carry a reason in\n"
            "# functional_process/cold_start.py's ACCEPTED.\n"
        )
        handle.writelines(f"{line}\n" for line in all_rows)


def read_pin(path: str = PIN) -> tuple[str, ...]:
    """The pin's rows, comments and blanks dropped."""
    with open(path, encoding="utf-8") as handle:
        return tuple(
            line.strip() for line in handle if line.strip() and not line.startswith("#")
        )


def _main(argv: list[str]) -> int:
    import jax

    jax.config.update("jax_enable_x64", True)

    only = None
    if "--input" in argv:
        only = argv[argv.index("--input") + 1]
    files = [only] if only else list(CONFIGURATIONS)

    all_rows: list[str] = []
    missing: list[str] = []
    for input_file in files:
        report = cold_report(input_file)
        print(report.summary())
        all_rows += rows(report)
        missing += [
            f"{os.path.basename(input_file)} {name}" for name in check_reasons(report)
        ]
    if missing:
        print("\ndisagreements with no reason in ACCEPTED  <-- MUST BE EMPTY:")
        for name in missing:
            print(f"  {name}")
    if "--write" not in argv:
        return 1 if missing else 0
    write_pin(all_rows)
    print(f"\nwrote {PIN}")
    return 0


if __name__ == "__main__":
    import sys

    # Re-imported under its real name rather than run out of `__main__`: `cold_state`
    # pickles a `ColdState`, and a class defined in `__main__` pickles as
    # `__main__.ColdState`, which no other process can unpickle. Found by a cache
    # written from the command line and read back from a test.
    from functional_process.cold_start import _main as main

    raise SystemExit(main(sys.argv[1:]))
