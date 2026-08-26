"""Harness cases for the ported subset of `process/models/physics/scrape_off_layer.py`
(`ScrapeOffLayer`, would-be `.tokamak.scrape_off_layer`).

Audit record: `functional_process/_audit/units/models/physics/scrape_off_layer.md`.

- Four `TestCalculate*` classes -- `ScrapeOffLayer`'s own `@staticmethod`s, diffed
  directly against them (no adapter needed: they take plain floats, no `self.data`
  access at all). Legacy samples are
  `tests/unit/models/physics/test_scrape_off_layer.py`'s own exact-value cases, verbatim.
- `TestScrapeOffLayer` -- the composite `calculate_scrape_off_layer`, diffed against a
  `DataStructure`-backed run of the real `ScrapeOffLayer.run()`, same "close the data
  backdoor" technique `test_confinement_time.py`'s `TestConfinementTime` uses for its own
  composite dispatcher.
- Plain structural tests confirming the RAW-separatrix-power caution actually landed in
  the node declarations (`.physics.p_plasma_separatrix_mw_raw`, never the transformed
  `.physics.p_plasma_separatrix_mw`), and that the composite's declared UNPORTED switch
  value (`USER_INPUT`) raises rather than silently mis-answering.
"""

import pytest

from functional_process._harness import Tier1Contract, legacy_sample
from functional_process.models.physics.scrape_off_layer import (
    Eich2013SOLPowerDecayLength,
    Mast2014SOLPowerDecayLength1,
    Mast2014SOLPowerDecayLength2,
    OutboardSOLEich13ParallelPowerFlux,
    OutboardSOLParallelPowerFlux,
    OutboardSOLPowerDecayLengthEich2013,
    UpstreamSOLOutboardEich13ParallelArea,
    UpstreamSOLOutboardParallelArea,
    calculate_eich2013_sol_power_decay_length,
    calculate_mast2014_sol_power_decay_length_1,
    calculate_mast2014_sol_power_decay_length_2,
    calculate_scrape_off_layer,
    calculate_upstream_sol_outboard_parallel_area,
)
from process.core.model import DataStructure
from process.models.physics.scrape_off_layer import ScrapeOffLayer


class TestCalculateEich2013SolPowerDecayLength(Tier1Contract):
    """`calculate_eich2013_sol_power_decay_length` -> the same, unchanged but for
    `safe_pow`.

    Samples are `tests/unit/models/physics/test_scrape_off_layer.py`'s own
    parametrised and exact-value cases, verbatim.
    """

    audit_record = "models/physics/scrape_off_layer.md"
    reference = staticmethod(ScrapeOffLayer.calculate_eich2013_sol_power_decay_length)
    ported = calculate_eich2013_sol_power_decay_length

    samples = [
        legacy_sample(
            "exact",
            p_plasma_separatrix_mw=100.0,
            rmajor=3.0,
            b_plasma_surface_poloidal_average=0.5,
            aspect=3.0,
        ),
        legacy_sample(
            "low-psep",
            p_plasma_separatrix_mw=10.0,
            rmajor=3.0,
            b_plasma_surface_poloidal_average=0.5,
            aspect=3.0,
        ),
        legacy_sample(
            "high-psep",
            p_plasma_separatrix_mw=500.0,
            rmajor=3.0,
            b_plasma_surface_poloidal_average=0.5,
            aspect=3.0,
        ),
        legacy_sample(
            "large-rmajor",
            p_plasma_separatrix_mw=100.0,
            rmajor=10.0,
            b_plasma_surface_poloidal_average=0.5,
            aspect=3.0,
        ),
        legacy_sample(
            "small-rmajor",
            p_plasma_separatrix_mw=100.0,
            rmajor=1.0,
            b_plasma_surface_poloidal_average=0.5,
            aspect=3.0,
        ),
    ]

    fuzz_bounds = {
        "p_plasma_separatrix_mw": (10.0, 500.0),
        "rmajor": (2.0, 20.0),
        "b_plasma_surface_poloidal_average": (0.05, 1.5),
        "aspect": (1.5, 4.0),
    }


class TestCalculateMast2014SolPowerDecayLength1(Tier1Contract):
    """`calculate_mast2014_sol_power_decay_length_1` -> the same, unchanged but for
    `safe_pow`.
    """

    audit_record = "models/physics/scrape_off_layer.md"
    reference = staticmethod(ScrapeOffLayer.calculate_mast2014_sol_power_decay_length_1)
    ported = calculate_mast2014_sol_power_decay_length_1

    samples = [
        legacy_sample(
            "exact", p_plasma_separatrix_mw=100.0, b_plasma_surface_poloidal_average=0.5
        ),
        legacy_sample(
            "low-psep",
            p_plasma_separatrix_mw=10.0,
            b_plasma_surface_poloidal_average=0.5,
        ),
        legacy_sample(
            "high-psep",
            p_plasma_separatrix_mw=500.0,
            b_plasma_surface_poloidal_average=0.5,
        ),
        legacy_sample(
            "high-bpol",
            p_plasma_separatrix_mw=100.0,
            b_plasma_surface_poloidal_average=2.0,
        ),
    ]

    fuzz_bounds = {
        "p_plasma_separatrix_mw": (10.0, 500.0),
        "b_plasma_surface_poloidal_average": (0.05, 1.5),
    }


class TestCalculateMast2014SolPowerDecayLength2(Tier1Contract):
    """`calculate_mast2014_sol_power_decay_length_2` -> the same, unchanged but for
    `safe_pow`. `cur_plasma_ma` is the already-converted (A -> MA) value, matching
    PROCESS's own `@staticmethod` signature -- see the port module's docstring.
    """

    audit_record = "models/physics/scrape_off_layer.md"
    reference = staticmethod(ScrapeOffLayer.calculate_mast2014_sol_power_decay_length_2)
    ported = calculate_mast2014_sol_power_decay_length_2

    samples = [
        legacy_sample("exact", p_plasma_separatrix_mw=100.0, cur_plasma_ma=1.0),
        legacy_sample("low-psep", p_plasma_separatrix_mw=10.0, cur_plasma_ma=1.0),
        legacy_sample("high-psep", p_plasma_separatrix_mw=500.0, cur_plasma_ma=1.0),
        legacy_sample("high-current", p_plasma_separatrix_mw=100.0, cur_plasma_ma=3.0),
    ]

    fuzz_bounds = {
        "p_plasma_separatrix_mw": (10.0, 500.0),
        "cur_plasma_ma": (1.0, 20.0),
    }


class TestCalculateUpstreamSolOutboardParallelArea(Tier1Contract):
    """`calculate_upstream_sol_outboard_parallel_area` -> the same, unchanged.

    `b_plasma_outboard_total == 0` is an unguarded division that makes the *value*
    itself non-finite -- see the port module's docstring for why that needs no
    `_harness/boundary.py` registration (the register is only for a value that stays
    finite while its gradient does not).
    """

    audit_record = "models/physics/scrape_off_layer.md"
    reference = staticmethod(
        ScrapeOffLayer.calculate_upstream_sol_outboard_parallel_area
    )
    ported = calculate_upstream_sol_outboard_parallel_area

    samples = [
        legacy_sample(
            "exact",
            rmajor=6.0,
            rminor=2.0,
            len_plasma_sol_power_decay=0.001,
            b_plasma_outboard_total=4.0,
            b_plasma_surface_poloidal_average=0.5,
        ),
    ]

    fuzz_bounds = {
        "rmajor": (2.0, 20.0),
        "rminor": (0.5, 5.0),
        "len_plasma_sol_power_decay": (1.0e-4, 1.0e-2),
        "b_plasma_outboard_total": (1.0, 12.0),
        "b_plasma_surface_poloidal_average": (0.05, 1.5),
    }


def _reference_scrape_off_layer(
    p_plasma_separatrix_mw_raw,
    rmajor,
    rminor,
    b_plasma_surface_poloidal_average,
    b_plasma_outboard_total,
    aspect,
    plasma_current,
    i_len_sol_outboard_power_decay,
):
    """Call the real `ScrapeOffLayer.run()` through the composite's own signature,
    closing the `self.data` back door -- same technique as `test_divertor.py`'s
    `_reference_divertor_heat_load_wade` and `test_confinement_time.py`'s
    `_reference_calculate_confinement_time`.

    `p_plasma_separatrix_mw_raw` is written onto `data.physics.p_plasma_separatrix_mw`
    directly -- PROCESS's own field has no "_raw" spelling; that is a port-side mint
    name introduced one node earlier (`functional_process/models/physics/physics.py`'s
    `SeparatrixPowerNonIgnited`), not a `DataStructure` field. `ScrapeOffLayer.run()`
    reads exactly this field, before `physics.py`'s own positivity kludge would have
    transformed it -- see the port module's docstring.
    """
    data = DataStructure()
    data.physics.p_plasma_separatrix_mw = p_plasma_separatrix_mw_raw
    data.physics.rmajor = rmajor
    data.physics.rminor = rminor
    data.physics.b_plasma_surface_poloidal_average = b_plasma_surface_poloidal_average
    data.physics.b_plasma_outboard_total = b_plasma_outboard_total
    data.physics.aspect = aspect
    data.physics.plasma_current = plasma_current
    data.physics.i_len_sol_outboard_power_decay = i_len_sol_outboard_power_decay

    sol = ScrapeOffLayer()
    sol.data = data
    sol.run()

    return (
        data.physics.len_plasma_sol_eich13_power_decay,
        data.physics.len_plasma_sol_mast14_power_decay_1,
        data.physics.len_plasma_sol_mast14_power_decay_2,
        data.physics.len_sol_outboard_power_decay,
        data.physics.a_plasma_outboard_sol_parallel,
        data.physics.a_plasma_outboard_sol_eich13_parallel,
        data.physics.pflux_plasma_outboard_sol_parallel_mw,
        data.physics.pflux_plasma_outboard_sol_eich13_parallel_mw,
    )


class TestScrapeOffLayer(Tier1Contract):
    """`calculate_scrape_off_layer` -> the composite `ScrapeOffLayer.run()`, all eight
    outputs.

    Legacy points at one realistic operating point, swept over the three switch values
    the composite supports (`EICH_2013`, `MAST_2014_1`, `MAST_2014_2` -- `USER_INPUT`
    is not a computation in PROCESS at all, see the port module's docstring, and is
    covered separately by `test_user_input_is_not_ported` below rather than by a value
    test). `fuzz_bounds` holds the switch fixed at `EICH_2013` (1) -- the value live on
    `large_tokamak_eval.IN.DAT` -- and sweeps the continuous physics arguments.
    """

    audit_record = "models/physics/scrape_off_layer.md"
    reference = _reference_scrape_off_layer
    ported = calculate_scrape_off_layer

    static_argnames = ("i_len_sol_outboard_power_decay",)

    samples = [
        legacy_sample(
            "eich2013-reference-arm",
            p_plasma_separatrix_mw_raw=150.0,
            rmajor=8.0,
            rminor=2.5,
            b_plasma_surface_poloidal_average=0.6,
            b_plasma_outboard_total=5.0,
            aspect=3.2,
            plasma_current=15.0e6,
            i_len_sol_outboard_power_decay=1,
        ),
        legacy_sample(
            "mast14-1",
            p_plasma_separatrix_mw_raw=150.0,
            rmajor=8.0,
            rminor=2.5,
            b_plasma_surface_poloidal_average=0.6,
            b_plasma_outboard_total=5.0,
            aspect=3.2,
            plasma_current=15.0e6,
            i_len_sol_outboard_power_decay=2,
        ),
        legacy_sample(
            "mast14-2",
            p_plasma_separatrix_mw_raw=150.0,
            rmajor=8.0,
            rminor=2.5,
            b_plasma_surface_poloidal_average=0.6,
            b_plasma_outboard_total=5.0,
            aspect=3.2,
            plasma_current=15.0e6,
            i_len_sol_outboard_power_decay=3,
        ),
    ]

    fuzz_fixed = {"i_len_sol_outboard_power_decay": 1}
    fuzz_bounds = {
        "p_plasma_separatrix_mw_raw": (10.0, 500.0),
        "rmajor": (2.0, 20.0),
        "rminor": (0.5, 5.0),
        "b_plasma_surface_poloidal_average": (0.05, 1.5),
        "b_plasma_outboard_total": (1.0, 12.0),
        "aspect": (1.5, 4.0),
        "plasma_current": (1.0e6, 2.0e7),
    }


def test_user_input_switch_value_is_not_ported():
    """`i_len_sol_outboard_power_decay == USER_INPUT` (0) raises in the port.

    A declared divergence from PROCESS, which does not raise there -- `run()`'s
    `if`/`elif`/`elif` simply matches nothing and leaves `len_sol_outboard_power_decay`
    at whatever it already was. There is no "answer" for the composite to reproduce for
    that arm, so it raises rather than silently returning an unrelated value. See the
    port module's and audit record's own notes on this switch.
    """
    with pytest.raises(ValueError, match="USER_INPUT"):
        calculate_scrape_off_layer(
            p_plasma_separatrix_mw_raw=150.0,
            rmajor=8.0,
            rminor=2.5,
            b_plasma_surface_poloidal_average=0.6,
            b_plasma_outboard_total=5.0,
            aspect=3.2,
            plasma_current=15.0e6,
            i_len_sol_outboard_power_decay=0,
        )


def test_raw_separatrix_power_nodes_read_the_raw_mint():
    """Every node reading the separatrix power reads `.physics.p_plasma_separatrix_mw_
    raw`, never the transformed `.physics.p_plasma_separatrix_mw` -- the caution the
    port module's docstring and the audit record both carry, pinned as a structural
    test so a future edit cannot silently rebind it to the wrong mint.
    """
    raw_reading_nodes = [
        Eich2013SOLPowerDecayLength(),
        Mast2014SOLPowerDecayLength1(),
        Mast2014SOLPowerDecayLength2(),
        OutboardSOLParallelPowerFlux(),
        OutboardSOLEich13ParallelPowerFlux(),
    ]
    for node in raw_reading_nodes:
        paths = [i.var.path_str() for i in node.inputs]
        assert ".physics.p_plasma_separatrix_mw_raw" in paths, (
            f"{type(node).__name__} does not read the raw separatrix-power mint"
        )
        assert ".physics.p_plasma_separatrix_mw" not in paths, (
            f"{type(node).__name__} reads the transformed (post-kludge) separatrix "
            "power instead of the raw mint"
        )


def test_eich13_area_node_reads_the_eich_length_directly():
    """`UpstreamSOLOutboardEich13ParallelArea` reads `len_plasma_sol_eich13_power_
    decay` directly, not `len_sol_outboard_power_decay` -- see its own docstring for
    why: PROCESS computes this area for the Eich length unconditionally, regardless of
    the switch, so reusing the switch-selected length would only be correct when the
    switch happens to select `EICH_2013`.
    """
    eich13_area = [
        i.var.path_str() for i in UpstreamSOLOutboardEich13ParallelArea().inputs
    ]
    selected_area = [i.var.path_str() for i in UpstreamSOLOutboardParallelArea().inputs]

    assert ".physics.len_plasma_sol_eich13_power_decay" in eich13_area
    assert ".physics.len_sol_outboard_power_decay" not in eich13_area
    assert ".physics.len_sol_outboard_power_decay" in selected_area


def test_outboard_sol_power_decay_length_occupant_reads_only_its_own_candidate():
    """`OutboardSOLPowerDecayLengthEich2013` reads only the Eich candidate -- the
    computes-then-selects shape means the switch decides which single already-computed
    value is passed through, not which formula runs, so the occupant's own reads-set is
    exactly one `VarPath`.
    """
    reads = [
        i.var.path_str() for i in OutboardSOLPowerDecayLengthEich2013().inputs
    ]
    assert reads == [".physics.len_plasma_sol_eich13_power_decay"]
    outputs = [
        o.var.path_str() for o in OutboardSOLPowerDecayLengthEich2013().outputs
    ]
    assert outputs == [".physics.len_sol_outboard_power_decay"]
