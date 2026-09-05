"""Harness case for the ported machine-config selection (registry unit #8).

The reference adapter drives PROCESS's *real* `load_stellarator_config` -- including the
`istell == 6` file read and the reflective `hasattr`/`setattr` loop -- and reads the 34
numeric fields straight back off a fresh `StellaratorConfigData`. So this case checks the
one thing the port actually changed: that a **statically enumerated** field list
reproduces, exactly, what reflection discovers at run time, on every machine config
PROCESS ships.

Every sample is routed through the `istell == 6` JSON path, the five preset dicts
included. That is deliberate rather than convenient: it puts the presets and the file
through one identical reference code path, so a value difference can only come from the
config data itself, and it exercises the `json` round-trip the reference run really does.
`json.dumps` uses `repr` for floats, so the round-trip is exact in float64.

**No differentiable argument exists here**, and `static_argnames` says so: the only input
is *which machine*, which is a graph-assembly fact, not a quantity. The gradient
tests therefore have nothing to compare and pass vacuously -- correctly, since the node
this unit backs has no inputs at all (see the port's `StellaratorMachineConfig`).
"""

import json
import tempfile
from pathlib import Path

import functional_process
from functional_process.cottax._harness import Tier1Contract, legacy_sample
from functional_process.cottax.stellarator.preset_config import (
    STELLA_CONFIG_SCALAR_FIELDS,
    select_stellarator_config_scalars,
)
from process.data_structure.stellarator_configuration import StellaratorConfigData
from process.models.stellarator.preset_config import (
    HELIAS3,
    HELIAS4,
    HELIAS5B,
    W7X30,
    W7X50,
    load_stellarator_config,
)


class _ConfigOnly:
    """The only attribute `load_stellarator_config` touches on its `data` argument.

    A whole `DataStructure` would work too; this makes it visible that the function's
    entire footprint is `data.stellarator_config`, which is what the audit record's
    data-footprint table claims.
    """

    def __init__(self):
        self.stellarator_config = StellaratorConfigData()


REFERENCE_STELLA_CONF = (
    Path(functional_process.__file__).resolve().parents[1]
    / "tests/regression/input_files/stellarator_helias.stella_conf.json"
)
"""`stellarator_helias.IN.DAT`'s companion config -- the run this whole port validates
against (`indat.REFERENCE_INPUT_FILE`)."""


def _reference_config_scalars(machine_config):
    """PROCESS's own `load_stellarator_config`, read back field by field.

    Parameters
    ----------
    machine_config :
        The machine config mapping to load.

    Returns
    -------
    :
        One value per entry of `STELLA_CONFIG_SCALAR_FIELDS`, in that order.
    """
    data = _ConfigOnly()
    with tempfile.TemporaryDirectory() as scratch:
        config_file = Path(scratch) / "stella_conf.json"
        config_file.write_text(json.dumps(dict(machine_config)))
        load_stellarator_config(6, config_file, data)

    return tuple(
        getattr(data.stellarator_config, field) for field in STELLA_CONFIG_SCALAR_FIELDS
    )


class TestStellaratorMachineConfig(Tier1Contract):
    """`load_stellarator_config`'s copy loop -> `select_stellarator_config_scalars`."""

    audit_record = "models/stellarator/preset_config.md"
    reference = _reference_config_scalars
    ported = select_stellarator_config_scalars

    static_argnames = ("machine_config",)

    samples = [
        legacy_sample(
            "stellarator-helias-stella-conf-json",
            machine_config=json.loads(REFERENCE_STELLA_CONF.read_text()),
        ),
        legacy_sample("preset-helias5b", machine_config=HELIAS5B),
        legacy_sample("preset-helias4", machine_config=HELIAS4),
        legacy_sample("preset-helias3", machine_config=HELIAS3),
        legacy_sample("preset-w7x30", machine_config=W7X30),
        legacy_sample("preset-w7x50", machine_config=W7X50),
    ]
