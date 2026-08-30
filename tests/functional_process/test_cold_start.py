"""The cold-point stage, pinned.

What is tested here is the *policy* and the *number*, in that order of importance:

- the seed and the expectation come from two different points, which is the whole
  mechanism (`test_the_seed_and_the_expectation_are_different_structures`);
- PROCESS's own cold state is settled far below anything the stage reports, so
  "PROCESS has not converged" cannot explain a pinned row
  (`test_process_s_cold_state_is_settled_far_below_every_disagreement`);
- every pinned disagreement carries a reason
  (`test_every_pinned_disagreement_has_a_reason`);
- and the four configurations' agreement counts and disagreement sets are exactly what
  `functional_process/reference_cold_start.txt` says
  (`test_the_cold_pin_is_exact`).

The pin itself is generated, never typed --
`$PY -m functional_process.cold_start --write`.

**Cost: ~59 s for the module**, dominated by four graph assemblies and four MDA runs,
not by PROCESS -- a cold PROCESS evaluation is 0.3-6.3 s per configuration and
`cold_state` caches it on disk anyway (keyed on the input files and the `process/` tree,
`mda_harness._cache_key`), so a re-run after editing `functional_process/` pays only for
the graph. Not marked `slow` and not skipped by default, deliberately: the reason this
check did not exist for the first year of the port is that nobody ran it, and a stage
that only runs when asked is a stage that does not run.
"""

from pathlib import Path

import jax
import pytest

jax.config.update("jax_enable_x64", True)

from functional_process.cold_start import (  # noqa: E402
    ACCEPTED,
    BOOKKEEPING,
    CONFIGURATIONS,
    EXTRA_PASSES,
    PIN,
    STELLARATOR,
    TOKAMAK_EVAL,
    TOKAMAK_NOF,
    _area_field,
    check_reasons,
    cold_report,
    cold_state,
    read_pin,
    rows,
)
from functional_process.mda_harness import EXPLAINED_DISAGREEMENTS  # noqa: E402


def _path_of(base: str) -> str:
    """The `CONFIGURATIONS` entry whose file name is `base`.

    `cold_start`'s `STELLARATOR`/`TOKAMAK_*` constants are the *base names* an
    `ACCEPTED` key and a pin row use; `cold_state` wants a path. One lookup rather than
    a second table, so the two cannot disagree about which file a name means.
    """
    return next(name for name in CONFIGURATIONS if Path(name).name == base)


@pytest.fixture(scope="module")
def reports():
    """One `ColdReport` per assembling configuration, built once for the module.

    Module-scoped because the four together are the expensive part of this file and
    every test below wants all four; `cold_state`'s own on-disk cache makes a *second*
    session cheap, but not a second call inside one.
    """
    return {Path(name).name: cold_report(name) for name in CONFIGURATIONS}


# ============================================================== the mechanism
def test_the_seed_and_the_expectation_are_different_structures():
    """The cold stage's entire content, in one assertion.

    `mda_harness.compare(graph, data)` seeds from the same `data` it diffs against, so a
    variable no node owns is handed PROCESS's own answer through the boundary and the
    comparison passes without the port computing anything. `cold_state` returns *two*
    structures -- `seed`, as `init_process` left it, and `process`, after one pipeline
    pass -- and they differ on exactly the fields PROCESS computes.

    Checked on `.physics.beta_poloidal_vol_avg`, which is the field that made this
    concrete: it was the first of the twenty-two missing producers to be found and land
    (`_audit/optimise_design.md` §16.6), and it is `0.0` in the seed against `1.087` in
    PROCESS's cold answer. A future refactor that quietly seeded from `process` would
    make this equality hold and every cold check vacuous.
    """
    state = cold_state(_path_of(TOKAMAK_NOF))
    assert state.seed is not state.process
    assert float(state.seed.physics.beta_poloidal_vol_avg) == 0.0
    assert float(state.process.physics.beta_poloidal_vol_avg) == pytest.approx(
        1.0874279219157827
    )
    # And a genuine `IN.DAT` input is identical in both -- which is what makes the split
    # safe: only computed fields move, so only they are un-seeded.
    assert float(state.seed.tfcoil.tftmp) == float(state.process.tfcoil.tftmp)


def test_the_write_set_is_what_the_boundary_audit_reads(reports):
    """`boundary.computed_by_process` is now this measurement, not a second one.

    The declaration-side check (*"does anything own it?"*) and the value-side check
    (*"does the port compute it?"*) are two views of one defect, and they are only
    comparable if they are taken at the same point from the same run. Asserted rather
    than trusted because the delegation is a one-line import that a later edit could
    silently un-do.
    """
    from functional_process.boundary import computed_by_process

    nof = reports["large_tokamak_nof.IN.DAT"]
    assert computed_by_process(nof.input_file) == nof.state.written
    # The one field the whole `str_wp` chain below rests on is in the write set: PROCESS
    # computes it every pass, so nothing in the port owning it is a missing producer and
    # not a genuine input.
    assert ("tfcoil", "str_wp") in nof.state.written


# ============================================================== the discriminator
def test_process_s_cold_state_is_settled_far_below_every_disagreement(reports):
    """The measurement that makes a cold disagreement attributable at all.

    `Caller.call_models` stops when the objective and the constraints stop moving
    (`caller.py:96-126`, `rtol = 1e-6`), which says nothing about the model loops. So
    before any row of the pin can be called the port's fault, PROCESS's own cold state
    has to be shown to be *finished* -- otherwise the port, which drives its blocks, is
    simply the more converged of the two.

    `ColdState.drift` runs PROCESS's own map `EXTRA_PASSES` further times and reports
    the largest relative motion. Measured 2026-08-30: exactly zero on
    `low_aspect_ratio_DEMO` and `large_tokamak_eval`, `7.00e-07` on `stellarator_helias`
    and `2.74e-08` on `large_tokamak_nof`, against smallest reported disagreements of
    `1.21e-04` and `1.17e-06`. The assertion is the *relation*, not those numbers: a
    tenfold margin per configuration, which is what licenses reading the pin as being
    about the port.
    """
    for name, report in reports.items():
        smallest = min((d.rel_diff for d in report.real), default=1.0)
        assert report.state.drift * 10 < smallest, (
            f"{name}: PROCESS's cold state still moves by {report.state.drift:.2e} "
            f"after {EXTRA_PASSES} further passes, which is not comfortably below its "
            f"smallest reported disagreement ({smallest:.2e}). Until that gap is "
            f"restored, no row of this configuration's pin can be attributed to the "
            f"port -- read cold_start.py's docstring, question 3."
        )


def test_the_pass_counter_is_not_mistaken_for_a_moving_model(reports):
    """`.numerics.n_model_calls` is excluded, and only it.

    The probe that measures whether PROCESS has settled increments PROCESS's own model
    call counter, so the counter always "moves" -- an observer effect, not a result. It
    was the single reported motion on the two configurations that are exact fixed
    points, which is exactly how a measurement artefact hides a null result.

    The narrow guard is that `BOOKKEEPING` stays narrow: excluding a field because the
    probe moves it is legitimate, excluding one because its motion is inconvenient is
    not, and there is no way to tell the two apart in a set that has grown.
    """
    assert frozenset({("numerics", "n_model_calls")}) == BOOKKEEPING
    for report in reports.values():
        assert all(
            (area, field) not in BOOKKEEPING
            for area, field, *_ in report.state.unsettled
        )


# ============================================================== output-pass-only
def test_a_variable_process_s_solve_pass_never_writes_is_not_scored(reports):
    """`Physics.outplas` runs only from `Physics.output()` -- so its three outputs have
    no cold answer to be compared against.

    `physics.py:219-223`: `output()` calls `outplas()`, `run()` does not. In a converged
    `DataStructure` the final report pass has filled `.physics.nu_star`, `.rho_star` and
    `.beta_mcdonald`, so `mda_harness.compare` sees them agree; at the cold point
    PROCESS holds `DataStructure()`'s `0.0` and the port holds its own answer. Neither
    side is wrong and there is nothing to compare, which is a third outcome and not a
    defect.

    **Derived from PROCESS's measured write set, never from a list**, which is why the
    assertion is stated that way round: any variable the port owns and PROCESS's solve
    pass does not write lands here automatically, so a *new* one appears as a count
    moving rather than as a silent pass.
    """
    for name, report in reports.items():
        for d in report.output_pass_only:
            assert _area_field(d.var) not in report.state.written, name
    tokamak = reports["large_tokamak_eval.IN.DAT"]
    assert {d.var.path_str() for d in tokamak.output_pass_only} == {
        ".physics.beta_mcdonald",
        ".physics.nu_star",
        ".physics.rho_star",
    }


def test_the_same_cause_reaches_the_error_bucket_as_a_shape(reports):
    """The one place the output-pass split shows up as an error rather than a value.

    `Physics.output()` also calls `calculate_effective_charge_ionisation_profiles`
    (`physics.py:220`), which `run()` does not, so at the cold point
    `.physics.n_charge_plasma_effective_profile` is still `(0,)` in PROCESS's structure
    while the port has computed all 201 elements. `_diff` reports a shape mismatch as
    *not comparable*, which is the honest third outcome and is why the cold tokamaks
    carry 22 errors against the warm run's 20.

    Worth its own test because `errors` is the one bucket where a regression is silent --
    an entry there is neither a pass nor a fail. The pin holds its count; this holds the
    reason the count is what it is, so a *different* error arriving is visible as more
    than a number changing.
    """
    messages = "\n".join(reports["large_tokamak_nof.IN.DAT"].comparison.errors)
    assert "shape mismatch for .physics.n_charge_plasma_effective_profile" in messages
    assert "port (201,) vs data (0,)" in messages


# ============================================================== the reasons
def test_every_pinned_disagreement_has_a_reason(reports):
    """**A pinned disagreement with no explanation is the failure this stage exists to
    end.**

    In a bare list of paths, a row somebody chased and a row nobody looked at read
    identically -- which is how twenty-two missing producers survived weeks of a harness
    reporting 983 of 1039 variables agreeing (`_audit/optimise_design.md` §16.3(b)).
    `ACCEPTED` is keyed on `(configuration, path)` because the cause is per-machine:
    `.costs.coe` is off by `3.4e-02` on the stellarator through the report-pass geometry
    and by `2.2e-04` on `large_tokamak_nof` through `noh`.
    """
    unexplained = {
        name: check_reasons(report)
        for name, report in reports.items()
        if check_reasons(report)
    }
    assert not unexplained, (
        "cold-point disagreements with no entry in cold_start.ACCEPTED. Establish which "
        "side is wrong -- by substituting the suspected cause and re-measuring, not by "
        "argument -- and record it there before pinning: " + repr(unexplained)
    )


def test_the_reason_table_has_no_rows_that_no_longer_disagree(reports):
    """`ACCEPTED` may not outlive what it explains.

    A reason for a row that now agrees is the same hazard as `boundary.py`'s stale
    exclusions and as `constraint_48`'s hole: a justification whose cause has expired,
    still standing, still read as current. Landing `stresscl` should make seven of these
    entries fail this test, which is the point.
    """
    live = {
        (name, d.var.path_str()) for name, report in reports.items() for d in report.real
    }
    assert set(ACCEPTED) == live, (
        "cold_start.ACCEPTED and the measured disagreements have diverged. Rows in "
        f"ACCEPTED that no longer disagree: {sorted(set(ACCEPTED) - live)}"
    )


def test_the_two_shared_causes_defer_to_the_warm_harness_s_record():
    """The dead PF-turn tail and the vacuum duct solve are the *same* two findings
    `mda_harness.EXPLAINED_DISAGREEMENTS` records at the converged point, and this
    module says so rather than restating them.

    Two tables of reasons that can drift apart is one table too many. The cold entries
    name the warm one as the authority; this asserts that the authority still exists.
    """
    for path in (
        ".pf_coil.n_pf_coil_turns",
        ".vacuum.dlscal",
        ".vacuum.dia_vv_vacuum_ducts",
    ):
        assert path in EXPLAINED_DISAGREEMENTS
        assert ("large_tokamak_eval.IN.DAT", path) in ACCEPTED or (
            STELLARATOR,
            path,
        ) in ACCEPTED


# ============================================================== the pin
def test_the_cold_pin_is_exact(reports):
    """The four configurations' cold agreement is exactly what the pin says.

    Equality, not a one-sided bound, for `test_boundary.py`'s reason: a count that
    *improved* is a producer landing or a defect fixed, and it must show up as a pin to
    regenerate at the moment it is cheap, rather than as a silent drift that makes the
    number stop meaning anything.

    Measured 2026-08-30, at the state this file was written in:

    | configuration | agreements | disagreements | output-pass-only |
    |---|---|---|---|
    | `stellarator_helias` | 453 | 49 | 2 |
    | `large_tokamak_nof` | 631 | 79 | 3 |
    | `low_aspect_ratio_DEMO` | 672 | 44 | 3 |
    | `large_tokamak_eval` | 688 | 24 | 3 |

    **`large_tokamak_nof`'s 631/79 against the warm harness's 682/33 is the measurement
    this whole module was built to take**, and the gap is not noise: 7 of the 46 extra
    cold rows are `.tfcoil.str_wp` sitting at its uninitialised `0.0` because
    `stresscl` is unported, and 65 are one wrong integer in `inductance.NOH`. Neither is
    visible warm, because warm the seed supplies the answer.
    """
    found = [line for report in reports.values() for line in rows(report)]
    assert found == list(read_pin(PIN)), (
        "cold-point agreement has moved. If a producer landed or a defect was fixed, "
        f"regenerate: $PY -m functional_process.cold_start --write ({PIN})"
    )


def test_the_pin_never_lists_a_configuration_that_does_not_assemble():
    """`helias_5b`, `spherical_tokamak_eval` and `st_regression` are refused by
    `machine_from_indat` before a graph exists, so there is nothing to cold-start.

    Named here so the four are a decision rather than an accident: a fifth appearing in
    `CONFIGURATIONS` without a graph would fail deep inside `cold_report` with an
    assembly error rather than saying what it is.
    """
    pinned = {line.split(" ", 1)[0] for line in read_pin(PIN)}
    assert pinned == {Path(name).name for name in CONFIGURATIONS}
    assert not pinned & {
        "helias_5b.IN.DAT",
        "spherical_tokamak_eval.IN.DAT",
        "st_regression.IN.DAT",
        "IFE.IN.DAT",
    }


# ============================================================== the two defects
def test_the_missing_stresscl_producer_costs_seven_variables_per_tokamak(reports):
    """`.tfcoil.str_wp` is `boundary.MISSING_PRODUCERS_PIN`'s one remaining row, and
    this is what it costs.

    Zero strain is the *peak* of the Nb3Sn critical-current fit, so seeding it at
    `DataStructure()`'s `0.0` makes every critical current and the TF temperature margin
    come out high -- an optimistic error on the quantity constraints 33/36 read. On
    `large_tokamak_nof` the margin is `1.58` against PROCESS's `1.24`, +27 %.

    **The warm harness reports none of this**, because at PROCESS's converged design the
    seed supplies `str_wp` itself. Confirmed by substitution on `large_tokamak_eval`:
    replacing the seeded `0.0` with PROCESS's own cold `0.0018442328` removes exactly
    these seven rows and adds none (688/27 -> 695/20, measured on the raw `compare`,
    before the output-pass split).

    Narrow guard rather than trusting the pin's row count: this is the row the
    missing-producer audit and this stage agree on, and it is the demonstration that the
    two checks measure one defect.
    """
    from functional_process.boundary import MISSING_PRODUCERS_PIN

    assert Path(MISSING_PRODUCERS_PIN).read_text().split() == [".tfcoil.str_wp"]
    for name in (
        "large_tokamak_nof.IN.DAT",
        "low_aspect_ratio_DEMO.IN.DAT",
        "large_tokamak_eval.IN.DAT",
    ):
        off = {d.var.path_str(): d for d in reports[name].real}
        assert ".tfcoil.temp_tf_superconductor_margin" in off, name
        assert off[".tfcoil.temp_tf_superconductor_margin"].got > (
            off[".tfcoil.temp_tf_superconductor_margin"].expected
        ), (
            f"{name}: the missing strain should make the margin optimistic, "
            f"not pessimistic"
        )


def test_the_pinned_noh_is_right_on_one_configuration_and_wrong_on_two():
    """The defect this stage found: `inductance.NOH` is a per-machine, per-*design*
    quantity pinned as a module constant.

    `PFCoil.induct` (`pfcoil.py:1758-1765`) splits the CS into
    `ceil(2 * z_pf_coil_upper[CS] / (r_pf_coil_outer[CS] - r_pf_coil_inner[CS]))`
    segments and every inductance depends on that integer, so the port's answer is a
    piecewise-constant function of the CS geometry and `NOH = 30` selects one piece.
    Measured from PROCESS's own cold `DataStructure`, which is the honest place to ask,
    since `.build.dr_cs` is an iteration variable on two of these three files:

    | configuration | ratio cold | `noh` cold |
    |---|---|---|
    | `large_tokamak_eval` | 29.028 | 30 |
    | `large_tokamak_nof` | 31.746 | 32 |
    | `low_aspect_ratio_DEMO` | 27.010 | 28 |

    So the constant is right on exactly the configuration it was measured on. It is also
    wrong at those two files' *converged* designs, and by a different amount (27 for
    both), which is the finding that makes this not a matter of picking a better number:
    `noh` moves with the solve. `_audit/units/models/pfcoil/inductance.md` files that as
    an open convention question; this test is the standing measurement of it, and it
    will start failing the day the question is answered.
    """
    import math

    from functional_process.models.pfcoil.inductance import NOH

    assert NOH == 30
    expected = {
        "large_tokamak_eval.IN.DAT": 30,
        "large_tokamak_nof.IN.DAT": 32,
        "low_aspect_ratio_DEMO.IN.DAT": 28,
    }
    for name in CONFIGURATIONS:
        if Path(name).name not in expected:
            continue
        pf = cold_state(name).process.pf_coil
        cs = int(pf.n_cs_pf_coils) - 1
        ratio = (
            2.0
            * pf.z_pf_coil_upper[cs]
            / (pf.r_pf_coil_outer[cs] - pf.r_pf_coil_inner[cs])
        )
        assert math.ceil(ratio) == expected[Path(name).name], Path(name).name


def test_the_stellarator_chain_is_process_s_own_other_arm(reports):
    """The stellarator's 44-row cold chain is not the port disagreeing with PROCESS --
    it is PROCESS disagreeing with itself.

    `Stellarator.run(output=False)` runs `st_coil` then `st_build`;
    `Stellarator.run(output=True)` runs `st_build` then `st_coil`
    (`stellarator.py:141-146` against `:159-165`), and `.build.z_tf_inside_half` is
    written by one and read by the other. Measured at the cold design: PROCESS's solve
    pass leaves `3.611990999471611`, PROCESS's own output-pass order leaves
    `5.513665371874896`, and the port computes `5.513665371874896` -- **the report-pass
    arm, to sixteen digits.**

    The port is self-consistent with one of PROCESS's two arms and PROCESS is not
    self-consistent between them, which is why this cannot be made to agree at both
    points and is pinned instead. `mda_harness.EXPLAINED_DISAGREEMENTS`'s
    `.heat_transport.p_plant_electric_base_total_mw` entry is the same finding at the
    converged point, from the other side.
    """
    stellarator = reports["stellarator_helias.IN.DAT"]
    off = {d.var.path_str(): d for d in stellarator.real}
    z = off[".build.z_tf_inside_half"]
    assert z.expected == pytest.approx(3.611990999471611, rel=1e-12)
    assert z.got == pytest.approx(5.513665371874896, rel=1e-12)
    # No tokamak shows it: `caller.py:272-275` never reaches `Stellarator.run`.
    for name in (TOKAMAK_NOF, TOKAMAK_EVAL):
        other = reports[Path(name).name]
        assert ".build.z_tf_inside_half" not in {d.var.path_str() for d in other.real}, (
            name
        )
