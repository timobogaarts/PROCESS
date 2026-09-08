"""Does the port's graph compute what PROCESS computes **when nothing hands it the
answer**?
"""

from __future__ import annotations

import os
import pickle
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

CONFIGURATIONS = (
    "tests/regression/input_files/stellarator_helias.IN.DAT",
    "tests/regression/input_files/helias_5b.IN.DAT",
    "tests/regression/input_files/large_tokamak_nof.IN.DAT",
    "tests/regression/input_files/low_aspect_ratio_DEMO.IN.DAT",
    "tests/regression/input_files/large_tokamak_eval.IN.DAT",
    "tests/regression/input_files/spherical_tokamak_eval.IN.DAT",
    "tests/regression/input_files/st_regression.IN.DAT",
)
"""The reference configurations this stage is measured on."""

PIN = os.path.join(os.path.dirname(__file__), "reference_cold_start.txt")
"""The pinned cold-point agreement -- see `rows` for the four kinds of line it holds."""

EXTRA_PASSES = 3
"""How many further `Caller._call_models_once` passes `cold_state` runs past the point
`check_agreement` stopped PROCESS, to decide whether its cold state is actually settled.
"""

SETTLED_RTOL = 1e-9
"""What counts as "still moving" in `ColdState.unsettled`."""

BOOKKEEPING = frozenset({("numerics", "n_model_calls")})
"""Fields excluded from `ColdState.unsettled` that the *measurement itself* moves."""


@dataclass
class ColdState:
    """PROCESS's own cold start on one input file: the seed, the answer, and two
    measurements about the answer.
    """

    seed: object
    """The `DataStructure` after `init_process` and **before any model has run**."""

    process: object
    """The same `DataStructure` after `load_iteration_variables` and one
    `Evaluators.fcnvmc1` at the cold `x` -- PROCESS's own answer at the design the input
    file starts from, computed by PROCESS's own pipeline and nothing else.
    """

    written: frozenset
    """`{(area, field)}` the pass moved -- PROCESS's measured write set."""

    passes: int
    """How many times `Caller._call_models_once` ran inside the evaluation -- PROCESS's
    Gauss-Seidel pass count at the cold `x`, capped at ten by `caller.py:99`.
    """

    unsettled: tuple
    """`(area, field, before, after, rel)` per field that still moves when PROCESS's
    pipeline is run `EXTRA_PASSES` further times, worst-relative-motion first.
    """

    @property
    def drift(self) -> float:
        """The largest relative motion in `unsettled` -- how far PROCESS's cold state
        still is from being a fixed point of its own pipeline.
        """
        return max((row[4] for row in self.unsettled), default=0.0)


def _resolve(name: str) -> str:
    """`name` as an absolute path, read relative to the repository root when relative --
    `run_mda_harness._resolve`'s rule, restated so this module has no import of it.
    """
    from pathlib import Path

    path = Path(name)
    root = Path(__file__).resolve().parent.parent.parent
    return str(path if path.is_absolute() else (root / path).resolve())


def _scratch_copy(input_file: str) -> str:
    """`input_file` copied into a fresh directory with its `.stella_conf.json`, so the
    `OUT.DAT`/`MFILE.DAT` a `SingleRun` writes do not land in the repository.
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
    """Every numeric field of every area of `data`, as float arrays."""
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
    """Keys whose value differs between the two snapshots, shape changes included."""
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


CACHE_VERSION = "cold-v2"
"""Bumped when `ColdState`'s *contents* change, so an old pickle can never be read back
under a key whose meaning has moved -- `mda_harness._CACHE_VERSION`'s discipline, and
its docstring records what a stale cross-read costs.
"""


def cold_state(input_file: str, use_cache: bool = True) -> ColdState:
    """Run PROCESS's pipeline **once**, at the input file's own starting design, and
    return everything the cold comparison needs.
    """
    from pathlib import Path

    from functional_process.cottax.mda_harness import CACHE_DIR, _cache_key

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
    seed=state.seed)` -- every bucket that stage defines, unchanged.
    """

    output_pass_only: list = field(default_factory=list)
    """Disagreements on a field PROCESS's **solve** pass never writes, so its "expected"
    value is `init_process`' default and not an answer.
    """

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
    """
    from functional_process.cottax.indat import graph_for, machine_from_indat
    from functional_process.cottax.mda_harness import compare

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
    """`(area, field)` for a plain `.area.field` `VarPath`, `None` for anything else."""
    keys = var.path_str().lstrip(".").split(".")
    return (keys[0], keys[1]) if len(keys) == 2 and "[" not in keys[1] else None


# --------------------------------------------------------------- accepted disagreements


STELLARATOR = "stellarator_helias.IN.DAT"
TOKAMAK_NOF = "large_tokamak_nof.IN.DAT"
TOKAMAK_DEMO = "low_aspect_ratio_DEMO.IN.DAT"
TOKAMAK_EVAL = "large_tokamak_eval.IN.DAT"
HELIAS_5B = "helias_5b.IN.DAT"
SPHERICAL_EVAL = "spherical_tokamak_eval.IN.DAT"
ST_REGRESSION = "st_regression.IN.DAT"
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

PF_COIL_SIX_RESIDUAL = (
    "**What was left when `noh` was fixed, and it was never a `noh` row.** These five "
    "sat inside `NOH_ROWS_DEMO` until `models/pfcoil/inductance.py` stopped pinning the "
    "CS segment count and computed it; eighty of that table's eighty-five rows retired "
    "outright and these did not move *at all* -- byte-identical in "
    "`reference_cold_start.txt` across the change, which is the evidence that the old "
    "table over-claimed them. They are one PF coil, index `6`: `2/132` elements of "
    "`.pf_coil.c_pf_coil_turn` and `f_c_pf_cs_peak_time_array`, `1/22` of "
    "`c_pf_cs_coil_flat_top_ma`, and the `1.3e-06` those carry into `.costs.c22`. All "
    "at `5.742e-05`, all with both sides nonzero (`-34.40651` against `-34.40849`), so "
    "coil `6` is **live** and this is not `PF_TURNS_DEAD_TAIL` leaking either -- that "
    "row's own worst index is `8`, in the tail, and a wrong turns count would quantise "
    "rather than land at five parts in a hundred thousand.\n\n"
    "**The cause is not diagnosed**, and this entry says so rather than guessing. It is "
    "pinned to stay bounded and visible: at `5.742e-05` it is below `PicardDriver`'s "
    "`1e-4` tolerance and an order under the `3.86e-04` vacuum-duct rows above it, and "
    "it is now the largest unexplained PF disagreement on any tokamak. Chasing it is "
    "`_audit/next_steps.md`'s item, not this table's."
)

PF_COIL_SIX_ROWS_DEMO = (
    ".costs.c22",
    ".pf_coil.c_pf_coil_turn",
    ".pf_coil.c_pf_cs_coil_flat_top_ma",
    ".pf_coil.f_c_pf_cs_peak_time_array",
    ".pf_coil.f_j_cs_start_end_flat_top",
)
"""The five, on `low_aspect_ratio_DEMO` only."""


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
the geometry itself and `1.56e-01` on `.buildings.volrci`.
"""

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

PF_TURNS_ROWS_SPHERICAL = (".pf_coil.n_pf_coil_turns",)
"""The same cause on the two spherical files, **without the `^hat` twin.** Those
machines set `iohcl = 0`, so the PF cycle's minted copy does not carry the dead tail the
way it does where a CS exists -- the array itself still does.
"""

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

VACUUM_DUCT_ROWS_SPHERICAL = (
    ".vacuum.dia_vv_vacuum_ducts",
    ".vacuum.dlscal",
    ".costs.c224",
    ".costs.c2243",
    ".costs.c2244",
    # The cost sums below `c224`, traced rather than assumed: `.costs.c224` is the only
    # disagreeing input of `.costs.fusion_power_island_cost`, `c22` the only one of
    # `.costs.total_plant_direct_cost`, and `concost` the only one of
    # `.costs.cost_of_electricity`. So this is `c224` propagating and **not** the
    # stellarator's `z_tf_inside_half` chain, whose row list happens to contain the same
    # aggregate names because the same sums sit below it there.
    ".costs.c22",
    ".costs.capcost",
    ".costs.ccont",
    ".costs.cdirt",
    ".costs.cindrt",
    ".costs.coe",
    ".costs.coecap",
    ".costs.concost",
    ".costs.moneyint",
)
"""`VACUUM_DUCT_SOLVE`'s rows on the two `i_pulsed_plant = 0` spherical files."""

VACUUM_DUCT_ROWS = (
    ".costs.c224",
    ".costs.c2243",
    ".costs.c2244",
    ".vacuum.dia_vv_vacuum_ducts",
    ".vacuum.dlscal",
)

DRIVER_TOLERANCE = (
    "**Below the driver's own convergence tolerance, so it is not evidence about the "
    "model.** `PicardDriver` is `cottax.drivers.PicardDriver` at this port's tolerances "
    "-- `rtol = 1e-6`, `atol = 1e-8`, `max_steps = 256` -- and a residue at `compare`'s "
    "own `rtol = 1e-6` inside or downstream of a `Drive` is the algorithm's convergence "
    "criterion showing through, not a difference in the arithmetic. The cost accounts "
    "below the PF cycle carry that residue through `PfMagnetCost` into `c22` and the "
    "capital-cost sum. `large_tokamak_eval` was the one configuration the retired "
    "`noh` pin happened to be right about, which is why this is all that is left of "
    "its PF chain -- and why it did not move when `noh` became computed.\n\n"
    "**`^hat.pf_coil.ind_pf_cs_plasma_mutual` was pinned here and is not any more** "
    "(2026-09-02). It used to agree only to `1.49e-06`, which this reason called *two "
    "orders better than the driver promises* -- against `cottax`'s own defaults of "
    "`rtol = atol = 1e-4`. This port's subclass holds the fixed point two and four "
    "orders tighter than that, on an implicit adjoint and from a starting guess that is "
    "no longer a dataclass zero (`mda.GIVEN_STARTS`), so the loop-carried unknown now "
    "agrees inside `compare`'s own tolerance and has nothing left to explain."
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
)


def _because(reason: str, mapping) -> dict:
    """`{(configuration, path): reason}` from `{configuration: paths}` -- so one cause
    is written once and still covers every row it explains, on every machine it explains
    it on.
    """
    return {
        (configuration, path): reason
        for configuration, paths in mapping.items()
        for path in paths
    }


WARD_KINK_SMOOTHED = (
    "**Not a defect on either side: the port deliberately does not compute PROCESS's "
    "expression here.** `_fast_alpha_fraction_ward` "
    "(`models/physics/pure_formulas.py`) regularises PROCESS's "
    "`sqrt(temp_sum_20 - 0.65)` threshold at `WARD_KINK_SMOOTHING = 1e-3`, because its "
    "unbounded derivative made `stellarator_helias`'s converged/stopped outcome turn on "
    "the last bit of a Jacobian cell -- 46 % of SQP steps crossed it and each crossing "
    "moved the `c24` Jacobian row by 339x (`_audit/optimise_design.md` §31.36). The "
    "same departure is declared at the unit level by `TestFastAlphaBetaWard`'s "
    "`declared_deviation`, which is the authority on its size.\n\n"
    "**Why it shows at the cold point, and only on these two files.** Both stellarators "
    "sit *below* the threshold cold (`temp_sum_20 = 0.6449` against `0.65`), where "
    "PROCESS returns exactly `0.0` and the regularisation returns a small positive tail "
    "-- `0.13 r^2 eps / sqrt(|a|)`, linear in `eps`, `~4e-05` at this `eps`. A "
    "disagreement against an exact zero has no relative size, which is why it appears "
    "here as a pinned row rather than inside a tolerance. The tokamaks sit above the "
    "threshold and are perturbed as `eps**2` instead, four orders smaller, so they do "
    "not reach this table (§31.41).\n\n"
    "Accepted deliberately, with `eps = 5e-4` measured as the alternative and rejected: "
    "see `WARD_KINK_SMOOTHING`'s own docstring for that trade."
)
"""Why `.physics.beta_fast_alpha` disagrees cold on the two stellarators."""

WARD_KINK_ROWS = (".physics.beta_fast_alpha",)
"""`WARD_KINK_SMOOTHED`'s one row."""


C1_INTERPOLANT = (
    "`intersect_residual` interpolates with a monotone cubic (C1) where PROCESS "
    "interpolates linearly. A deliberate divergence -- see "
    "`_audit/deliberate_divergences.md` and `optimise_design.md` §89 for the receipt: "
    "the piecewise-linear residual is only piecewise smooth, its derivative jumps at "
    "every one of ~200 tabulated breakpoints, and SAND exposes that residual to the "
    "outer SQP. Measured, ten +-ulp draws: 8/10 converging across 235-429 iterations "
    "with two hard caps becomes 10/10 across 83-101, while VMCON takes exactly 43 on "
    "every draw either way. A different interpolant through the same points is a "
    "different function, so the crossing moves 8.1e-05 relative and propagates through "
    "the winding-pack geometry. **Stellarator-only**: `intersect` has no tokamak "
    "caller, and no tokamak row moved."
)

C1_INTERPOLANT_ROWS_STELLARATOR = frozenset({
    ".build.dr_bore",
    ".build.dr_tf_inboard",
    ".build.dr_tf_outboard",
    ".build.required_radial_space",
    ".buildings.cryvol",
    ".buildings.elevol",
    ".buildings.tfcbv",
    ".buildings.wrbi",
    ".costs.c216",
    ".costs.c217",
    ".costs.c2174",
    ".costs.c222",
    ".costs.c2221",
    ".costs.c22211",
    ".costs.c22212",
    ".costs.c22213",
    ".costs.c22214",
    ".costs.c22215",
    ".costs.c225",
    ".costs.c2251",
    ".costs.c22511",
    ".costs.c22515",
    ".costs.c2263",
    ".costs.crctcore",
    ".fwbs.r_cryostat_inboard",
    ".fwbs.vol_cryostat",
    ".heat_transport.helpow",
    ".heat_transport.p_cryo_plant_electric_mw",
    ".heat_transport.p_tf_electric_supplies_mw",
    ".heat_transport.pacpmw",
    ".power.p_cryo_plant_electric_profile_mw",
    ".power.p_tf_electric_supplies_profile_mw",
    ".power.qcl",
    ".power.qmisc",
    ".power.qss",
    ".rebco.coppera_m2",
    ".structure.aintmass",
    ".structure.clgsmass",
    ".structure.coldmass",
    ".tfcoil.a_tf_coil_inboard_case",
    ".tfcoil.a_tf_coil_wp_turn_insulation",
    ".tfcoil.a_tf_inboard_total",
    ".tfcoil.a_tf_leg_outboard",
    ".tfcoil.a_tf_wp_conductor",
    ".tfcoil.a_tf_wp_extra_void",
    ".tfcoil.a_tf_wp_steel",
    ".tfcoil.b_tf_inboard_peak_symmetric",
    ".tfcoil.c_tf_turn",
    ".tfcoil.cryo_cool_req",
    ".tfcoil.dr_tf_wp_with_insulation",
    ".tfcoil.dx_tf_inboard_out_toroidal",
    ".tfcoil.dx_tf_wp_primary_toroidal",
    ".tfcoil.dx_tf_wp_secondary_toroidal",
    ".tfcoil.j_tf_coil_full_area",
    ".tfcoil.j_tf_wp",
    ".tfcoil.m_tf_coil_case",
    ".tfcoil.m_tf_coil_conductor",
    ".tfcoil.m_tf_coil_copper",
    ".tfcoil.m_tf_coil_superconductor",
    ".tfcoil.m_tf_coil_wp_insulation",
    ".tfcoil.m_tf_coil_wp_turn_insulation",
    ".tfcoil.m_tf_coils_total",
    ".tfcoil.m_tf_wp_steel_conduit",
    ".tfcoil.max_force_density",
    ".tfcoil.n_tf_coil_turns",
    ".tfcoil.r_b_tf_inboard_peak_symmetric",
    ".tfcoil.sig_tf_wp",
    ".tfcoil.tfckw",
    ".tfcoil.tficrn",
    ".tfcoil.tfocrn",
    ".tfcoil.tfsai",
    ".tfcoil.tfsao",
    ".tfcoil.v_tf_coil_dump_quench_kv",
})

C1_INTERPOLANT_ROWS_HELIAS_5B = frozenset({
    ".build.dr_bore",
    ".build.dr_tf_inboard",
    ".build.dr_tf_outboard",
    ".build.r_tf_outboard_mid",
    ".build.required_radial_space",
    ".buildings.cryvol",
    ".buildings.elevol",
    ".buildings.tfcbv",
    ".buildings.wrbi",
    ".costs.c216",
    ".costs.c217",
    ".costs.c2174",
    ".costs.c222",
    ".costs.c2221",
    ".costs.c22211",
    ".costs.c22212",
    ".costs.c22213",
    ".costs.c22214",
    ".costs.c22215",
    ".costs.c225",
    ".costs.c2251",
    ".costs.c22511",
    ".costs.c22512",
    ".costs.c22515",
    ".costs.c2263",
    ".costs.crctcore",
    ".fwbs.dewmkg",
    ".fwbs.r_cryostat_inboard",
    ".fwbs.vol_cryostat",
    ".heat_transport.helpow",
    ".heat_transport.p_cryo_plant_electric_mw",
    ".heat_transport.p_tf_electric_supplies_mw",
    ".heat_transport.pacpmw",
    ".power.p_cryo_plant_electric_profile_mw",
    ".power.p_tf_electric_supplies_profile_mw",
    ".power.qcl",
    ".power.qmisc",
    ".power.qss",
    ".rebco.coppera_m2",
    ".structure.aintmass",
    ".structure.clgsmass",
    ".structure.coldmass",
    ".tfcoil.a_tf_coil_inboard_case",
    ".tfcoil.a_tf_coil_wp_turn_insulation",
    ".tfcoil.a_tf_inboard_total",
    ".tfcoil.a_tf_leg_outboard",
    ".tfcoil.a_tf_wp_conductor",
    ".tfcoil.a_tf_wp_extra_void",
    ".tfcoil.a_tf_wp_steel",
    ".tfcoil.b_tf_inboard_peak_symmetric",
    ".tfcoil.c_tf_turn",
    ".tfcoil.cryo_cool_req",
    ".tfcoil.dr_tf_wp_with_insulation",
    ".tfcoil.dx_tf_inboard_out_toroidal",
    ".tfcoil.dx_tf_wp_primary_toroidal",
    ".tfcoil.dx_tf_wp_secondary_toroidal",
    ".tfcoil.j_tf_coil_full_area",
    ".tfcoil.j_tf_wp",
    ".tfcoil.m_tf_coil_case",
    ".tfcoil.m_tf_coil_conductor",
    ".tfcoil.m_tf_coil_copper",
    ".tfcoil.m_tf_coil_superconductor",
    ".tfcoil.m_tf_coil_wp_insulation",
    ".tfcoil.m_tf_coil_wp_turn_insulation",
    ".tfcoil.m_tf_coils_total",
    ".tfcoil.m_tf_wp_steel_conduit",
    ".tfcoil.max_force_density",
    ".tfcoil.n_tf_coil_turns",
    ".tfcoil.r_b_tf_inboard_peak_symmetric",
    ".tfcoil.sig_tf_wp",
    ".tfcoil.tfckw",
    ".tfcoil.tficrn",
    ".tfcoil.tfocrn",
    ".tfcoil.tfsai",
    ".tfcoil.tfsao",
    ".tfcoil.v_tf_coil_dump_quench_kv",
})


ACCEPTED = {
    **_because(
        C1_INTERPOLANT,
        {
            STELLARATOR: C1_INTERPOLANT_ROWS_STELLARATOR,
            HELIAS_5B: C1_INTERPOLANT_ROWS_HELIAS_5B,
        },
    ),
    **_because(
        TF_STRESS_LANDED,
        {
            TOKAMAK_NOF: TF_STRESS_ROWS,
            TOKAMAK_DEMO: TF_STRESS_ROWS,
            TOKAMAK_EVAL: TF_STRESS_ROWS,
        },
    ),
    **_because(PF_COIL_SIX_RESIDUAL, {TOKAMAK_DEMO: PF_COIL_SIX_ROWS_DEMO}),
    **_because(
        STELLARATOR_ARM_ORDER,
        {STELLARATOR: STELLARATOR_ARM_ORDER_ROWS, HELIAS_5B: STELLARATOR_ARM_ORDER_ROWS},
    ),
    **_because(
        PF_TURNS_DEAD_TAIL,
        {
            TOKAMAK_NOF: PF_TURNS_ROWS,
            TOKAMAK_DEMO: PF_TURNS_ROWS,
            TOKAMAK_EVAL: PF_TURNS_ROWS,
            SPHERICAL_EVAL: PF_TURNS_ROWS_SPHERICAL,
            ST_REGRESSION: PF_TURNS_ROWS_SPHERICAL,
        },
    ),
    **_because(
        VACUUM_DUCT_SOLVE,
        {
            STELLARATOR: VACUUM_DUCT_ROWS,
            HELIAS_5B: VACUUM_DUCT_ROWS,
            TOKAMAK_NOF: VACUUM_DUCT_ROWS,
            TOKAMAK_DEMO: VACUUM_DUCT_ROWS,
            TOKAMAK_EVAL: VACUUM_DUCT_ROWS,
            SPHERICAL_EVAL: VACUUM_DUCT_ROWS_SPHERICAL,
            ST_REGRESSION: VACUUM_DUCT_ROWS_SPHERICAL,
        },
    ),
    **_because(DRIVER_TOLERANCE, {TOKAMAK_EVAL: DRIVER_TOLERANCE_ROWS_EVAL}),
    **_because(
        WARD_KINK_SMOOTHED,
        {STELLARATOR: WARD_KINK_ROWS, HELIAS_5B: WARD_KINK_ROWS},
    ),
}
"""`{(configuration, written path): why it is pinned}`."""


def check_reasons(report: ColdReport) -> tuple[str, ...]:
    """Disagreeing paths in `report` with no `ACCEPTED` entry: what a caller refuses on.
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
    """`report` as pin lines, in a stable order."""
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
            "# `$PY -m functional_process.cottax.cold_start --write`; do not hand-edit.\n"
            "# `agree` may only go up; `off`, `nocompare` and `errors` may only shrink;\n"
            "# every `off` path must carry a reason in\n"
            "# functional_process/cottax/cold_start.py's ACCEPTED.\n"
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
    from functional_process.cottax.cold_start import _main as main

    raise SystemExit(main(sys.argv[1:]))
