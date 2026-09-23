"""`configurations.kinds` still says what its sources say.

The counts are `decision_kinds.md`'s headline with the one 2026-09-17 move applied
(`f_j_tf_wp_critical_max`, belief -> build): 147 / 76 / 16 / 62 / 13 / 10 becomes
146 / 77 / 16 / 62 / 13 / 10, boundary 324. NUMERICS has since moved twice more, neither
re-pinned at the time: `48cc2c14` ("a solver start is numerics by kind") took it
62 -> 56 by reclassifying entries the outer-driven MDF path's removal made numerics
elsewhere; `df050df3` ("an incoming field value is an input, not a fixed point") then
added two new boundary inputs as `Kind.NUMERICS`
(`.physics.temp_plasma_ion_vol_avg_kev_in`, `.power.delta_eta_in`), 56 -> 58. Net
62 -> 58, boundary 324 -> 320, every other kind unchanged. The cross-references are the
invariants the two-stage architecture relies on: every sizing choice is a claimed build
output, every sampled leaf is a boundary input, the design variables sort 4 / 3 / 1, and
the settled pairing closes the power balance by the density.
"""

from __future__ import annotations

from collections import Counter

import pytest

from functional_process.configurations.kinds import (
    BELIEFS,
    BUILD_LEAVES,
    C16,
    CLAIMED_BUILD_OUTPUTS,
    DESIGN_KINDS,
    DESIGN_PLACES,
    ECONOMIC,
    HISTORIC_PAIRING,
    KINDS,
    LIMIT_OF,
    LIMIT_ON,
    PAIRINGS,
    SIZING_CHOICES,
    TE,
    Decision,
    Kind,
)

COUNTS = {
    Kind.BELIEF: 146,
    Kind.BUILD: 77,
    Kind.OPERATING: 16,
    Kind.NUMERICS: 58,
    Kind.DERIVED: 13,
    Kind.LIMIT: 10,
}
"""Per kind, after the reclassification."""


def test_counts_per_kind():
    """The headline counts, after the reclassification; boundary 320."""
    assert Counter(KINDS.values()) == COUNTS
    assert len(KINDS) == 320


def test_the_one_reclassified_row():
    """The designer's current margin is a build decision."""
    assert KINDS[".constraints.f_j_tf_wp_critical_max"] is Kind.BUILD


def test_spellings_are_the_ports():
    """Every key is `.area.field` or a `^stated` / `^guess` port of one."""
    for spelling in KINDS:
        assert spelling.startswith((".", "^stated.", "^guess.")), spelling
        assert spelling.count(".") >= 2, spelling


def test_design_variables():
    """The eight `ixc`: 4 build, 3 operating, 1 belief, each at its place's kind."""
    assert len(DESIGN_KINDS) == 8
    assert set(DESIGN_KINDS) == set(DESIGN_PLACES)
    assert Counter(DESIGN_KINDS.values()) == {
        Kind.BUILD: 4,
        Kind.OPERATING: 3,
        Kind.BELIEF: 1,
    }
    for ixc, place in DESIGN_PLACES.items():
        assert KINDS[place] is DESIGN_KINDS[ixc], (ixc, place)


def test_limits():
    """Ten constraints name a limit input; each is a `LIMIT` with a kind it is on."""
    assert set(LIMIT_OF) == set(LIMIT_ON)
    assert len(LIMIT_OF) == 10
    for spelling in LIMIT_OF.values():
        assert KINDS[spelling] is Kind.LIMIT, spelling
    assert Counter(LIMIT_ON.values()) == {
        Kind.BUILD: 7,
        Kind.BELIEF: 2,
        Kind.NUMERICS: 1,
    }


def test_sizing_choices_are_claimed_build_outputs():
    """The fourteen sizing choices are claimed outputs, decided 3 / 3 / 4 / 4."""
    assert len(SIZING_CHOICES) == 14
    assert set(SIZING_CHOICES) <= set(CLAIMED_BUILD_OUTPUTS)
    assert Counter(SIZING_CHOICES.values()) == {
        Decision.LIFT: 3,
        Decision.RECOURSE: 3,
        Decision.QUANTILE: 4,
        Decision.NONE: 4,
    }
    assert set(CLAIMED_BUILD_OUTPUTS.values()) == {"1a", "1b", "1c", "1d", "4"}
    assert not set(CLAIMED_BUILD_OUTPUTS) & set(KINDS), "an output is not an input"


def test_beliefs_are_boundary_inputs():
    """Every sampled path bar `dummy`, and the two subsets, is a boundary input."""
    paths = [b.path for b in BELIEFS]
    assert "dummy" in paths
    assert len(paths) == len(set(paths))
    for path in paths:
        if path != "dummy":
            assert path in KINDS, path
    for path in (*BUILD_LEAVES, *ECONOMIC):
        assert path in KINDS, path
        assert path in paths, path
    assert {b.kind for b in BELIEFS} <= {"uniform", "relative", "lognormal", "factor"}


def test_the_canonical_table_is_the_new_one():
    """`hfact` lognormal(0.10); tungsten and ripple relative 20 %."""
    by_path = {b.path: b for b in BELIEFS}
    assert by_path[".physics.hfact"].kind == "lognormal"
    assert by_path[".physics.hfact"].a == pytest.approx(0.10)
    for path in (
        ".impurity_radiation.f_nd_impurity_electron_array[13]",
        ".stellarator.bmn",
    ):
        assert by_path[path].kind == "relative"
        assert by_path[path].a == pytest.approx(0.20)


def test_build_leaves_are_build():
    """What the `build` table holds at nominal is build."""
    for path in BUILD_LEAVES:
        assert KINDS[path] is Kind.BUILD, path


def test_pairings():
    """`one` closes c2 by the density; the historic pairing by `hfact`, a belief."""
    density = ".physics.nd_plasma_electrons_vol_avg"
    assert PAIRINGS["one"] == {".constraints.c2": density}
    assert PAIRINGS["two"][".constraints.c2"] == density
    assert PAIRINGS["te"] == {".constraints.c2": density, C16: TE}
    assert KINDS[density] is Kind.OPERATING
    assert KINDS[TE] is Kind.OPERATING
    assert HISTORIC_PAIRING[".constraints.c2"] == ".physics.hfact"
    assert KINDS[".physics.hfact"] is Kind.BELIEF
