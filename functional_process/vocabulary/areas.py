"""The `DataStructure` area names, vendored from `process/core/model.py` (§23.2).

`paths.py` needed only the *list* of field names, not the class -- so this is that list.
`tests/functional_process/test_vocabulary.py` asserts it equals
`[f.name for f in dataclasses.fields(DataStructure)]`, order included.
"""

AREAS = (
    "water_use",
    "costs_2015",
    "cs_fatigue",
    "vacuum",
    "costs",
    "first_wall",
    "fwbs",
    "blanket",
    "structure",
    "times",
    "reinke",
    "ccfe_hcpb",
    "pulse",
    "build",
    "primary_pumping",
    "buildings",
    "constraints",
    "dcll",
    "current_drive",
    "heat_transport",
    "ife",
    "divertor",
    "pf_coil",
    "power",
    "stellarator",
    "stellarator_config",
    "pf_power",
    "neoclassics",
    "impurity_radiation",
    "physics",
    "rebco",
    "tfcoil",
    "superconducting_tfcoil",
    "globals",
    "scan",
    "numerics",
)
"""Every area PROCESS has, straight from `DataStructure` -- 36 of them."""
