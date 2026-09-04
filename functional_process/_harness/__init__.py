"""Validation harness for the pure-functional port.

Design: `functional_process/_audit/test_harness.md` (the four tiers and their pass
criteria). This package is the machinery those tiers are built from; the per-unit cases
live next to their audit records, not here.

Importing this package enables JAX's x64 mode as a side effect. That is deliberate and
it is why every harness module imports `_harness` (directly or transitively) before it
touches `jax`: PROCESS is float64 throughout, and a value diff run under JAX's float32
default shows precision loss that reads exactly like a porting bug. A session fixture
would be too late — collection imports the port modules first, and any array built
during import would already be float32.
"""

import jax

jax.config.update("jax_enable_x64", True)

from functional_process._harness.contracts import (  # noqa: E402
    PortContract,
    Tier1Contract,
    Tier2Contract,
)
from functional_process._harness.finite_difference import (  # noqa: E402
    PROCESS_EPSFCN,
    ZeroPerturbationError,
    central_difference,
    fd_gradient_with_error,
)
from functional_process._harness.sampling import (  # noqa: E402
    Sample,
    bounds_from_iteration_variables,
    fuzz_samples,
    legacy_sample,
)
from functional_process._harness.tolerance import (  # noqa: E402
    MACHINE_PRECISION,
    DeclaredDeviation,
    Tolerance,
)
from functional_process._harness.varpath import path  # noqa: E402

__all__ = [
    "MACHINE_PRECISION",
    "PROCESS_EPSFCN",
    "PortContract",
    "Sample",
    "Tier1Contract",
    "Tier2Contract",
    "DeclaredDeviation",
    "Tolerance",
    "ZeroPerturbationError",
    "bounds_from_iteration_variables",
    "central_difference",
    "fd_gradient_with_error",
    "fuzz_samples",
    "legacy_sample",
    "path",
]
