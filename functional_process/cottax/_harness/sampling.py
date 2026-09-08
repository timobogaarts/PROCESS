"""Where the input points come from."""

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

_LOG_UNIFORM_SPAN = 100.0
"""Bounds spanning more than this factor are sampled log-uniformly."""


@dataclass(frozen=True)
class Sample:
    """One evaluation point: keyword arguments, plus where they came from."""

    kwargs: MappingProxyType
    provenance: str
    label: str

    @property
    def id(self):
        """Test id for this sample, provenance included."""
        return f"{self.provenance}-{self.label}"


def legacy_sample(label, **kwargs):
    """Build a `Sample` from a literal point lifted from PROCESS's own unit tests."""
    return Sample(MappingProxyType(dict(kwargs)), "legacy", label)


def bounds_from_iteration_variables(*names):
    """Look up declared bounds for PROCESS variables by name."""
    by_name = {}
    # Deferred: this is the only thing in `_harness` that needs PROCESS installed, and
    # importing it at module scope made `_harness/__init__` -- and so every module that
    # touches the harness -- unimportable without it (`test_process_free_import`).
    from process.core.solver.iteration_variables import (  # noqa: PLC0415
        ITERATION_VARIABLES,
    )

    for var in ITERATION_VARIABLES.values():
        key = var.target_name or var.name
        by_name.setdefault(key, (var.lower_bound, var.upper_bound))

    missing = [n for n in names if n not in by_name]
    if missing:
        raise KeyError(
            f"not declared iteration variables: {missing}. Give explicit bounds for "
            f"these instead — they have no PROCESS-sanctioned range to borrow"
        )
    return {n: by_name[n] for n in names}


def _draw(rng, low, high):
    """One draw from `(low, high)`, log-uniform per component where the span demands it.
    """
    low, high = np.broadcast_arrays(
        np.asarray(low, dtype=float), np.asarray(high, dtype=float)
    )
    unit = rng.uniform(size=low.shape)

    positive = low > 0.0
    safe_low = np.where(positive, low, 1.0)
    safe_high = np.where(positive, high, 1.0)
    log_span = positive & (safe_high / safe_low > _LOG_UNIFORM_SPAN)

    value = np.where(
        log_span,
        np.exp(np.log(safe_low) + unit * (np.log(safe_high) - np.log(safe_low))),
        low + unit * (high - low),
    )
    return float(value) if value.shape == () else value


def fuzz_samples(bounds, count, seed, fixed=None):
    """Draw `count` random points from `bounds`."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(count):
        kwargs = dict(fixed or {})
        for name, (low, high) in bounds.items():
            kwargs[name] = _draw(rng, low, high)
        out.append(Sample(MappingProxyType(kwargs), "fuzz", f"seed{seed}-{i:03d}"))
    return out


def converged_sample(*_args, **_kwargs):
    """Sample read off a solved operating point."""
    raise NotImplementedError(
        "converged-point sampling needs a solved DataStructure; see "
        "functional_process/_audit/test_harness.md, tier 1, sampling"
    )
