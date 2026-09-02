"""`render_xdsm.grouped_uncut`: the provenance/SCC pair on the declared graph, uncut.

Not a numerical test -- nothing here checks a value against PROCESS, and nothing here
solves anything. What is pinned is the structural claim `_audit/uncut_graph.md` records:
`Blocking.scc` on `machine_graph()`'s graph (no `driven_graph`, no cut, no assigned
driver) needs no problem declared anywhere to be computed at all, so it never hits the
"coupled block [...] declares no problem" refusal `mda.py`'s module docstring and
`grouped`'s own docstring warn a raw, undriven cycle would trigger for anything that
*runs* it. It only partitions and orders; running is a different question this module
never asks.
"""

from cottax.blocking import Blocking

from functional_process.boundary import TOKAMAK_INPUT_FILE
from functional_process.indat import graph_for, machine_from_indat
from functional_process.render_xdsm import (
    MODES,
    OUTDIR,
    grouped,
    grouped_uncut,
    machine_graph,
    mode,
)
from functional_process.visualization.grouping import grouping_report


def test_grouped_uncut_is_registered_as_a_mode():
    assert MODES["grouped_uncut"] is grouped_uncut
    assert mode(["grouped_uncut"]) is grouped_uncut


def test_grouped_uncut_takes_the_machine_flag_like_grouped_does():
    chosen = mode(["grouped_uncut", "--machine"])
    assert getattr(chosen, "func", chosen) is grouped_uncut
    assert chosen.keywords["input_file"] == TOKAMAK_INPUT_FILE


def test_blocking_scc_needs_no_driven_graph_at_all(monkeypatch):
    """The likely obstacle the brief warned about does not occur.

    `mda.driven_graph` is patched to raise if it is ever called; `grouped_uncut` still
    runs to completion (writes both files), which is only possible if nothing on its
    call path -- `Blocking.scc`, `grouping_report`, `structure_order`,
    `render_grouped_dsm_html` -- reaches for a driver or a declared problem.
    """
    import functional_process.mda as mda

    def _boom(*a, **k):
        raise AssertionError("grouped_uncut must not call mda.driven_graph")

    monkeypatch.setattr(mda, "driven_graph", _boom)
    # `render_xdsm` imported `driven_graph` by name into its own module for `grouped`;
    # patching `mda`'s copy alone would not catch a call through that binding, so both
    # are patched -- proving neither path is taken, not just one of them.
    import functional_process.render_xdsm as render_xdsm

    if hasattr(render_xdsm, "driven_graph"):
        monkeypatch.setattr(render_xdsm, "driven_graph", _boom)

    grouped_uncut()  # must not raise


def test_grouped_uncut_writes_files_distinct_from_grouped(monkeypatch, tmp_path):
    import functional_process.render_xdsm as render_xdsm

    monkeypatch.setattr(render_xdsm, "OUTDIR", tmp_path)
    path = grouped_uncut()
    assert path == tmp_path / "dsm_provenance_uncut.html"
    assert (tmp_path / "dsm_provenance_uncut.html").exists()
    assert (tmp_path / "dsm_scc_uncut.html").exists()
    # Neither collides with `grouped`'s own pair -- both can be written to the same
    # directory without either overwriting the other.
    assert not (tmp_path / "dsm_provenance.html").exists()
    grouped()
    assert (tmp_path / "dsm_provenance.html").exists()
    assert (tmp_path / "dsm_scc.html").exists()
    # `grouped`'s run did not touch the uncut pair.
    assert (tmp_path / "dsm_provenance_uncut.html").exists()
    assert (tmp_path / "dsm_scc_uncut.html").exists()


def test_grouped_uncut_writes_the_machine_suffixed_pair(monkeypatch, tmp_path):
    import functional_process.render_xdsm as render_xdsm

    monkeypatch.setattr(render_xdsm, "OUTDIR", tmp_path)
    grouped_uncut(input_file=TOKAMAK_INPUT_FILE)
    assert (tmp_path / "dsm_provenance_uncut_tokamak.html").exists()
    assert (tmp_path / "dsm_scc_uncut_tokamak.html").exists()


def test_the_reference_stellarator_has_exactly_two_genuinely_coupled_uncut_sccs():
    """Pinned so a future edit to `physics`/`stellarator` membership is forced to
    re-check rather than silently leave `_audit/uncut_graph.md`'s census stale.

    Measured 2026-09-01: 144 total blocks, 2 with more than one *real* (non-minted)
    member -- the 6-node density/fusion/composition cycle inside `physics`
    (`fusion_power_totals_mw`, `fusion_rates`, `fusion_totals_no_beam`,
    `plasma_composition`, `density_profile`, `parabolic_on_axis_densities`) and the
    2-node `stellarator.divertor`/`stellarator.fw_area` cycle -- exactly the two
    `mda.CUTS` was built to close (`proton_rate_density` + `fusden_alpha_total` for the
    first, `f_ster_div_single` for the second).
    """
    declared, _ = machine_graph()
    blocking = Blocking.scc(declared)
    report = grouping_report(blocking)
    assert len(blocking.blocks) == 144
    assert len(report.coupled) == 2
    sizes = sorted(len(b.members) for b in report.coupled)
    assert sizes == [2, 6]


def test_the_reference_tokamak_has_exactly_three_genuinely_coupled_uncut_sccs():
    """The tokamak counterpart -- three raw cycles, none of them the stellarator's
    pair verbatim: an 8-node density/fusion/pedestal cycle (the stellarator's 6-node
    one, enlarged by the pedestal profile arm), a 4-node TF build/winding-pack cycle,
    and a 9-node PF-coil/volt-second/burn-time cycle -- matching `mda.CUTS`'s own
    accounting of which cuts land on `large_tokamak_eval.IN.DAT`.

    The block **total** moved `223 -> 226` on 2026-09-02 and the coupled structure did
    not: the three nodes that arrived since it was pinned (`.tokamak.build.r_cp_top`,
    `.tokamak.physics.psep_over_r_metric`, `.tokamak.radiated_wall_load`) are acyclic, so
    each is its own singleton block. The assertions this test is *about* -- three coupled
    SCCs of sizes `[4, 8, 9]` -- never moved, which is the reason to keep the total here
    rather than drop it: it is the line that notices a port landing.
    """
    declared = graph_for(machine_from_indat(TOKAMAK_INPUT_FILE))
    blocking = Blocking.scc(declared)
    report = grouping_report(blocking)
    assert len(blocking.blocks) == 226
    assert len(report.coupled) == 3
    sizes = sorted(len(b.members) for b in report.coupled)
    assert sizes == [4, 8, 9]


def test_cutting_changes_no_block_count_only_the_coupled_blocks_own_size():
    """`mda.driven_graph` lands every `FixedPointCut` *inside* an SCC that already
    existed on the uncut graph, so the total block count is unchanged and each coupled
    block gains exactly the one minted problem node its cut declared.
    """
    from functional_process.mda import driven_graph

    declared, _ = machine_graph()
    driven = driven_graph(declared)
    uncut_blocking = Blocking.scc(declared)
    driven_blocking = Blocking.scc(driven)
    assert len(uncut_blocking.blocks) == len(driven_blocking.blocks) == 144

    uncut_report = grouping_report(uncut_blocking)
    driven_report = grouping_report(driven_blocking)
    assert len(uncut_report.coupled) == len(driven_report.coupled) == 2
    uncut_sizes = sorted(len(b.members) for b in uncut_report.coupled)
    driven_sizes = sorted(len(b.members) for b in driven_report.coupled)
    assert driven_sizes == [s + 1 for s in uncut_sizes]
    # `real` (non-minted membership) is exactly what cutting must not change: the cut
    # adds a problem, not a model.
    uncut_real = sorted(b.real for b in uncut_report.coupled)
    driven_real = sorted(b.real for b in driven_report.coupled)
    assert uncut_real == driven_real
