"""Validation harness for the pure-functional port."""

import jax

jax.config.update("jax_enable_x64", True)

from functional_process.cottax._harness.contracts import (  # noqa: E402
    PortContract,
    Tier1Contract,
    Tier2Contract,
)
from functional_process.cottax._harness.finite_difference import (  # noqa: E402
    PROCESS_EPSFCN,
    ZeroPerturbationError,
    central_difference,
    fd_gradient_with_error,
)
from functional_process.cottax._harness.sampling import (  # noqa: E402
    Sample,
    bounds_from_iteration_variables,
    fuzz_samples,
    legacy_sample,
)
from functional_process.cottax._harness.tolerance import (  # noqa: E402
    MACHINE_PRECISION,
    DeclaredDeviation,
    Tolerance,
)
from functional_process.cottax._harness.varpath import path  # noqa: E402

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
