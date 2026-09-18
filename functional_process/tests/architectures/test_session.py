"""`architectures.session`: an arm assembled without a solve (`Session.assemble`)
is the build the solve then runs.
"""

from __future__ import annotations

import pytest

from functional_process.cottax.architectures import session


def test_assemble_is_what_solve_uses(monkeypatch):
    """`assemble("MDF")` builds once, under the key `MDA` and `MDF` share; a solve
    afterwards hands that very object to the arm's solver, and assembling again
    returns it.
    """
    live = session.open_session("helias_5b")
    assert live.builds == {}
    build = live.assemble("MDF")
    assert isinstance(build, session.MdfBuild)
    assert live.builds == {"MDF": build}
    assert live.assemble("MDA") is build
    seen = []

    def solve_mda(build, cold):
        seen.append(build)
        return {"status": "stubbed"}

    monkeypatch.setattr(session, "solve_mda", solve_mda)
    assert live.mda() == {"status": "stubbed"}
    assert seen == [build]
    assert live.assemble("MDF") is build
    with pytest.raises(ValueError, match="the arms are"):
        live.assemble("MDO")
