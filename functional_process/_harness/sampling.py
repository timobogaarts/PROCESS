"""Where the input points come from.

Three provenances, with genuinely different jobs (`_audit/test_harness.md` § Tier 1):

- `legacy` — literal points lifted from PROCESS's own `tests/unit`. These are already
  labelled input/output pairs someone validated, and most were generated from real
  stellarator input files, so they are both realistic *and* independently oracled. A
  port that breaks one of these breaks visibly.
- `fuzz` — randomised within the bounds PROCESS already declares for its iteration
  variables. Finds the points nobody thought to write a test for.
- `converged` — read off a solved operating point. Not implemented yet; it needs a full
  solve, and the units that will need it (anything with a narrow physical domain) do not
  exist yet. `converged_sample` is the seam where it lands.

Provenance is part of the test id, so `-k legacy` and `-k fuzz` select between them.
"""

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from process.core.solver.iteration_variables import ITERATION_VARIABLES

_LOG_UNIFORM_SPAN = 100.0
"""Bounds spanning more than this factor are sampled log-uniformly.

Densities run 2e19..1e21; sampling that linearly would put essentially every draw in the
top decade and never exercise the low end.
"""


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
    """Build a `Sample` from a literal point lifted from PROCESS's own unit tests.

    Parameters
    ----------
    label :
        Short identifier, ideally naming the source test and input file.
    **kwargs :
        The port function's arguments.

    Returns
    -------
    :
        The sample.
    """
    return Sample(MappingProxyType(dict(kwargs)), "legacy", label)


def bounds_from_iteration_variables(*names):
    """Look up declared bounds for PROCESS variables by name.

    Uses the bounds PROCESS itself declares in
    `process/core/solver/iteration_variables.py` rather than inventing a range, so the
    fuzzing domain is the one the solver is actually allowed to explore.

    Parameters
    ----------
    *names :
        PROCESS variable names, as spelled in `ITERATION_VARIABLES`.

    Returns
    -------
    :
        Mapping of name to `(lower_bound, upper_bound)`.

    Raises
    ------
    KeyError
        If a name is not a declared iteration variable.
    """
    by_name = {}
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

    Either bound may be an array; the draw takes the shape the two broadcast to. Both
    branches are evaluated and selected between, so the guard against `log(0)` is a
    substituted bound rather than a branch — `np.where` computes what it does not choose.
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
    """Draw `count` random points from `bounds`.

    Parameters
    ----------
    bounds :
        Mapping of argument name to `(lower, upper)`. Either bound may be an array, in
        which case the argument is drawn with the shape the two broadcast to and each
        component from its own interval. **An array-valued argument declares its shape by
        declaring its bounds** — there is no separate shape field to keep in sync, and
        per-component bounds are what a species array needs anyway (alpha densities do
        not live in the electron density's decade).
    count :
        Number of samples to draw.
    seed :
        PRNG seed. Recorded in the label so a failure is reproducible from its test id
        alone.
    fixed :
        Arguments held constant across every draw (switches, and anything whose value
        is a precondition rather than a domain).

    Returns
    -------
    :
        List of samples.
    """
    rng = np.random.default_rng(seed)
    out = []
    for i in range(count):
        kwargs = dict(fixed or {})
        for name, (low, high) in bounds.items():
            kwargs[name] = _draw(rng, low, high)
        out.append(Sample(MappingProxyType(kwargs), "fuzz", f"seed{seed}-{i:03d}"))
    return out


def converged_sample(*_args, **_kwargs):
    """Sample read off a solved operating point. Not implemented.

    Deliberately a hard failure rather than a silent skip: this is the provenance that
    matters for units whose domain is narrow enough that fuzzing mostly produces NaN
    (sqrt/log restrictions), and a unit that reaches for it needs to know it is not
    there yet rather than quietly testing nothing.

    Raises
    ------
    NotImplementedError
        Always.
    """
    raise NotImplementedError(
        "converged-point sampling needs a solved DataStructure; see "
        "functional_process/_audit/test_harness.md, tier 1, sampling"
    )
