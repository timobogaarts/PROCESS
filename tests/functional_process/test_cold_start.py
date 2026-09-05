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
    before a row of the pin can be called the port's fault, PROCESS's own cold state has
    to be shown to be *finished* -- otherwise the port, which drives its blocks, is
    simply the more converged of the two.

    `ColdState.drift` runs PROCESS's own map `EXTRA_PASSES` further times and reports
    the largest relative motion. Measured 2026-08-30: exactly zero on
    `low_aspect_ratio_DEMO` and `large_tokamak_eval`, `7.00e-07` on `stellarator_helias`
    and `2.74e-08` on `large_tokamak_nof`.

    **Tightened 2026-09-04, and it is stricter than what it replaced.** This used to
    test the *smallest* disagreement per configuration -- `drift * 10 < min(rel_diff)` --
    which has two faults. It checked only one row and let every row above it ride free;
    and one small, deliberate, fully explained disagreement made the whole
    configuration's pin uninterpretable, including rows a hundred times above the drift
    that were perfectly attributable. `stellarator_helias` was already thin on the old
    rule (its smallest was `1.17e-06` against a `7.00e-07` drift, a 1.7x margin), so the
    fault was latent rather than hypothetical.

    Now **every** row must individually clear `drift * 10` **or** carry an `ACCEPTED`
    reason. Rows that clear it keep the original guarantee unchanged; rows that do not
    have to be *explained*, which is what `ACCEPTED` is for and what `check_reasons`
    already enforces separately. Nothing is exempted that was checked before -- the
    per-row rule is strictly stronger, because the old one only ever bound the minimum.
    """
    for name, report in reports.items():
        floor = report.state.drift * 10
        unattributable = [
            d
            for d in report.real
            if d.rel_diff <= floor and (name, d.var.path_str()) not in ACCEPTED
        ]
        assert not unattributable, (
            f"{name}: PROCESS's cold state moves by {report.state.drift:.2e} after "
            f"{EXTRA_PASSES} further passes, and these rows are not ten times above "
            f"that and carry no `ACCEPTED` reason, so they cannot be attributed to the "
            f"port -- read cold_start.py's docstring, question 3: "
            + ", ".join(f"{d.var.path_str()} ({d.rel_diff:.2e})" for d in unattributable)
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
    """Every configuration that assembles is measured; only `IFE` is out.

    **This test twice named files as "refused by `machine_from_indat` before a graph
    exists" that assembled fine**, and both times the refusal was stale rather than
    wrong-at-the-time. It now asserts the positive -- every tracked file is in -- so a
    file dropping out is the failure rather than a docstring quietly going out of date.

    `helias_5b` joined on 2026-09-02: its 49 real disagreements were covered exactly by
    the two causes the reference stellarator already carried, with nothing left over.

    `spherical_tokamak_eval` and `st_regression` joined the same day, and only after two
    port defects they alone exposed were **fixed rather than pinned** -- the PF topology
    never threaded into `PfCoilPowerSupplies` (`pfckts` 12 against 13, and ten rows), and
    `burn_time` running on an `i_pulsed_plant = 0` machine where `Pulse.run` does not
    (thirteen rows, `-10` against `1000`). Their remaining 15 are the vacuum-duct solve
    and the PF turns' dead tail.

    `IFE.IN.DAT` stays out: `ife == 1`, a whole unported device.
    """
    pinned = {line.split(" ", 1)[0] for line in read_pin(PIN)}
    assert pinned == {Path(name).name for name in CONFIGURATIONS}
    assert pinned >= {
        "stellarator_helias.IN.DAT",
        "helias_5b.IN.DAT",
        "large_tokamak_nof.IN.DAT",
        "low_aspect_ratio_DEMO.IN.DAT",
        "large_tokamak_eval.IN.DAT",
        "spherical_tokamak_eval.IN.DAT",
        "st_regression.IN.DAT",
    }
    assert "IFE.IN.DAT" not in pinned


# ============================================================== the two defects
def test_the_landed_stresscl_producer_closed_its_own_seven_rows(reports):
    """`stresscl` landed the same day this stage was written, and this is the guard that
    it stayed landed.

    This test used to assert the opposite. `.tfcoil.str_wp` was
    `boundary.MISSING_PRODUCERS_PIN`'s last row, seeded at `DataStructure()`'s `0.0` --
    the *peak* of the Nb3Sn critical-current fit -- so every critical current and both
    temperature margins came out high, an **optimistic** error on exactly what
    constraints 33 and 36 read (`1.58` against PROCESS's `1.24` on `large_tokamak_nof`,
    +27 %). The prediction recorded with it was that substituting PROCESS's own cold
    `0.0018442328` would remove exactly seven rows and add none. Registry row 55 landed
    the producer hours later and reproduced that: the seven agree, the pin is empty, and
    `large_tokamak_nof` went 631 -> 646 cold agreements.

    **The warm harness could not have seen any of it**, because at PROCESS's converged
    design the seed supplies `str_wp` itself -- which is this stage's whole argument.

    What survived is one row, `.tfcoil.insstrain`, a new output of the landed node, and
    the port is the correct side of it: nine digits of agreement at PROCESS's converged
    design and 6.2e-03 relative cold, an ordinary two-fixed-points disagreement smaller
    than any of the seven it replaced. It carries its own reason in `ACCEPTED`.
    """
    from functional_process.boundary import MISSING_PRODUCERS_PIN

    assert Path(MISSING_PRODUCERS_PIN).read_text().split() == []
    closed = (
        ".tfcoil.temp_tf_superconductor_margin",
        ".tfcoil.j_tf_wp_critical",
        ".superconducting_tfcoil.c_tf_turn_cables_critical",
    )
    for name in (
        "large_tokamak_nof.IN.DAT",
        "low_aspect_ratio_DEMO.IN.DAT",
        "large_tokamak_eval.IN.DAT",
    ):
        off = {d.var.path_str() for d in reports[name].real}
        for row in closed:
            assert row not in off, f"{name}: {row} disagrees again -- stresscl unwired?"


def test_the_port_computes_the_same_noh_process_does_on_every_configuration():
    """The answer to the question its predecessor was the standing measurement of.

    `PFCoil.induct` (`pfcoil.py:1758-1765`) splits the CS into
    `ceil(2 * z_pf_coil_upper[CS] / (r_pf_coil_outer[CS] - r_pf_coil_inner[CS]))`
    segments and every inductance depends on that integer, so the port's answer is a
    piecewise-constant function of the CS geometry. The port used to pin `NOH = 30`,
    which selected one piece -- right on `large_tokamak_eval`, where it was measured,
    and wrong on the other two:

    | configuration | ratio cold | `noh` cold | old pin |
    |---|---|---|---|
    | `large_tokamak_eval` | 29.028 | 30 | 30, right |
    | `large_tokamak_nof` | 31.746 | 32 | 30, wrong by two |
    | `low_aspect_ratio_DEMO` | 27.010 | 28 | 30, wrong by two |

    `_cs_segments` now computes it, so this test asks the question the other way round:
    the port's `noh` must equal the one PROCESS's own cold `DataStructure` implies, on
    **every** configuration that has a solenoid, not just the one the constant came
    from. `.build.dr_cs` is an iteration variable on two of these files, which is why
    the cold state is the honest place to ask.

    That change retired eighty of `cold_start`'s eighty-five `NOH_ROWS_*` rows;
    `PF_COIL_SIX_RESIDUAL` records what stayed, and why it was never a `noh` row.
    """
    import math

    from functional_process.cottax.pfcoil.inductance import NOH_PAD, _cs_segments

    seen = {}
    for name in CONFIGURATIONS:
        state = cold_state(name)
        if not int(state.process.build.iohcl):
            # No central solenoid: `pfcoil()` skips `ohcalc` outright
            # (`pfcoil.py:1048-1050`), `induct` never reaches the segment split, and the
            # port's occupant is `PFCoilInductanceNoCentralSolenoid`, which has no
            # `_cs_segments` call to check. The CS slot of the geometry arrays is unset
            # on these files -- one of them reads a *negative* ratio -- so asking here
            # would be measuring uninitialised memory, not the port.
            continue
        pf = state.process.pf_coil
        n = int(pf.n_cs_pf_coils) - 1
        dr = pf.r_pf_coil_outer[n] - pf.r_pf_coil_inner[n]
        ratio = 2.0 * pf.z_pf_coil_upper[n] / dr
        ported, *_ = _cs_segments(
            z_cs_half=pf.z_pf_coil_upper[n],
            dr_cs_edges=dr,
            r_cs_middle=pf.r_pf_coil_middle[n],
        )
        assert int(ported) == math.ceil(ratio), Path(name).name
        seen[Path(name).name] = int(ported)

    # The three the retired pin was measured against, and its two errors.
    assert seen["large_tokamak_eval.IN.DAT"] == 30
    assert seen["large_tokamak_nof.IN.DAT"] == 32
    assert seen["low_aspect_ratio_DEMO.IN.DAT"] == 28
    # Every count stays inside the pad, which is the assumption `NOH_PAD` encodes.
    assert max(seen.values()) <= NOH_PAD


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
