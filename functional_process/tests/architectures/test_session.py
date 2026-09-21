"""`architectures.session`: an arm assembled without a solve (`Session.assemble`)
is the build the solve then runs.
"""

from __future__ import annotations

import pytest

from functional_process.cottax.architectures import session


def test_assemble_is_what_solve_uses(monkeypatch):
    """`assemble("MDA")` builds once, under its own key; a solve afterwards hands
    that very object to the arm's solver, and assembling again returns it. `MDF` is
    a block of its own on an optimiser configuration.
    """
    live = session.open_session("helias_5b")
    assert live.builds == {}
    build = live.assemble("MDA")
    assert isinstance(build, session.MdfBuild)
    assert live.builds == {"MDA": build}
    seen = []

    def solve_mda(build, cold):
        seen.append(build)
        return {"status": "stubbed"}

    monkeypatch.setattr(session, "solve_mda", solve_mda)
    assert live.mda() == {"status": "stubbed"}
    assert seen == [build]
    assert live.assemble("MDA") is build
    block = live.assemble("MDF")
    assert isinstance(block, session.BlockBuild) and block.nested
    assert live.builds == {"MDA": build, "MDF": block}
    with pytest.raises(ValueError, match="the arms are"):
        live.assemble("MDO")


def test_a_root_find_configuration_shares_its_build_between_mda_and_mdf():
    live = session.open_session("large_tokamak_eval")
    assert live.arms == ("MDA", "MDF")
    assert live.assemble("MDF") is live.assemble("MDA")
    assert isinstance(live.assemble("MDF"), session.MdfBuild)
