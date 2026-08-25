"""pytest configuration for the pure-functional port's validation harness.

Lives at the root of `tests/functional_process/`, not inside the `functional_process/`
package: pytest only applies a `conftest.py` at or above the tests it collects, so once
the cases moved under `tests/` this file had to follow them. Everything it registers --
`--fp-fuzz`, `--fp-fuzz-seed`, `--fp-gradients`, the tier markers, the `sample`
parametrisation, `audit_root` -- is scoped to this subtree, exactly as before.

`tests/conftest.py` still applies above it (it is an ancestor now, which it was not when
this file sat in the package). That one configures PROCESS's own suite -- matplotlib
backends, a cwd guard, a `Models` fixture -- none of which this harness needs, but none
of which it conflicts with either.
"""

import logging
from pathlib import Path

import pytest

# Imported first and for its side effect: enables JAX x64 before any array exists.
# See functional_process/_harness/__init__.py.
import functional_process
import functional_process._harness  # noqa: F401
from functional_process._harness.sampling import fuzz_samples

_FUZZ_DEFAULT = 1

PACKAGE_ROOT = Path(functional_process.__file__).resolve().parent

AUDIT_ROOT = PACKAGE_ROOT / "_audit" / "units"
"""Audit-record paths in contracts are relative to this.

Records live in their own mirrored tree under `_audit/units/`, cases in another under
`tests/functional_process/`; both mirror the package layout, so a contract's
`audit_record` stays the package-relative path it always was and only the root it is
resolved against moved. Resolved off the *package*, not off this file -- anchoring on
`__file__` would silently resolve every `audit_record` under `tests/`, where no record
exists, and `test_audit_record_exists` would fail on every contract at once.
"""


def pytest_configure(config):
    """Register the harness's markers.

    The repo runs with `--strict-markers`, so these must be declared. Declaring them
    here rather than in `pyproject.toml` keeps the harness self-contained.
    """
    config.addinivalue_line("markers", "tier1: explicit pure function, value + gradient")
    config.addinivalue_line(
        "markers", "tier2: internal solver, residual-based criterion"
    )
    config.addinivalue_line("markers", "tier3: acyclic composition of ported units")
    config.addinivalue_line("markers", "tier4: full coupled MDA")
    config.addinivalue_line(
        "markers", "gradient: compares against PROCESS's finite difference; opt-in"
    )
    # jax announces the missing CUDA jaxlib on every import in this env. It is expected
    # (the harness is CPU-only) and it is emitted through `logging`, not `warnings`, so
    # it needs silencing here rather than via `filterwarnings` — otherwise it is
    # captured and replayed in the output of every failing test.
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)


def pytest_addoption(parser):
    """Add the harness's CLI options."""
    group = parser.getgroup("functional_process")
    group.addoption(
        "--fp-fuzz",
        type=int,
        default=_FUZZ_DEFAULT,
        help=(
            "number of random samples per fuzzable contract "
            f"(default {_FUZZ_DEFAULT}; 0 disables fuzzing)"
        ),
    )
    group.addoption(
        "--fp-fuzz-seed",
        type=int,
        default=0,
        help="PRNG seed for fuzz sampling; recorded in each test id",
    )
    group.addoption(
        "--fp-gradients",
        action="store_true",
        default=False,
        help=(
            "run the gradient-agreement checks against PROCESS's finite difference "
            "(off by default: they cost ~4 reference calls per argument component, "
            "which for an array argument is the bulk of the run)"
        ),
    )


def pytest_collection_modifyitems(config, items):
    """Skip the finite-difference gradient comparison unless it was asked for.

    Autodiff of an explicit pure function is hard to get wrong once the *value* agrees,
    and the check is the expensive one -- per argument component, against a PROCESS
    reference that has to be re-evaluated four times. So it is opt-in, and skipped rather
    than deselected: a check that is not running says so on every run.
    """
    if config.getoption("--fp-gradients"):
        return
    skip = pytest.mark.skip(reason="gradient checks are opt-in: pass --fp-gradients")
    for item in items:
        if "gradient" in item.keywords:
            item.add_marker(skip)


def pytest_generate_tests(metafunc):
    """Parametrize the `sample` fixture from the contract class's declaration.

    This is what lets a ported unit declare points instead of writing test functions:
    every inherited test taking `sample` is expanded once per point, with the
    provenance in the test id so `-k legacy` / `-k fuzz` select between them.
    """
    if "sample" not in metafunc.fixturenames or metafunc.cls is None:
        return

    samples = list(getattr(metafunc.cls, "samples", ()))

    bounds = getattr(metafunc.cls, "fuzz_bounds", None)
    count = metafunc.config.getoption("--fp-fuzz")
    if bounds and count:
        samples += fuzz_samples(
            bounds,
            count,
            metafunc.config.getoption("--fp-fuzz-seed"),
            fixed=getattr(metafunc.cls, "fuzz_fixed", None),
        )

    if not samples:
        pytest.fail(
            f"{metafunc.cls.__name__} declares no samples and no fuzz_bounds — it "
            f"would pass without checking anything"
        )

    metafunc.parametrize("sample", samples, ids=[s.id for s in samples])


@pytest.fixture(scope="session")
def audit_root():
    """Directory audit-record paths are resolved against."""
    return AUDIT_ROOT
