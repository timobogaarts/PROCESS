"""Check every ported constraint/objective against one real, converged PROCESS run."""

import dataclasses
import inspect
import math

from functional_process.cottax.core.solver import constraints as ported_constraints
from functional_process.cottax.core.solver import objectives as ported_objectives
from functional_process.cottax.mda_harness import converged_data
from process.core.solver.constraints import ConstraintManager
from process.core.solver.objectives import objective_function


def _close(g, e, rtol, atol) -> bool:
    """`np.isclose`'s own rule, plus an explicit same-signed-infinity case --
    `math.isclose`/`np.isclose` both already treat `inf == inf` as close, but `abs(inf -
    inf)` is `nan` and a naive `abs(diff) <= tol` check would wrongly call that a
    disagreement (`nan <= anything` is always `False`).
    """
    g, e = float(g), float(e)
    if math.isinf(g) or math.isinf(e):
        return g == e
    return abs(g - e) <= atol + rtol * abs(e)


def _areas(data):
    return [f.name for f in dataclasses.fields(data)]


def _resolve_args(fn, data):
    """`{param name: value}` for every parameter of `fn`, read off `data` by searching
    every area for an unambiguous match.
    """
    areas = _areas(data)
    kwargs = {}
    for name in inspect.signature(fn).parameters:
        hits = [area for area in areas if hasattr(getattr(data, area), name)]
        if len(hits) == 0:
            raise ValueError(f"{fn.__name__}: no area has a field named {name!r}")
        if len(hits) > 1:
            raise ValueError(
                f"{fn.__name__}: {name!r} is ambiguous, found in areas {hits}"
            )
        kwargs[name] = getattr(getattr(data, hits[0]), name)
    return kwargs


def _all_ported_constraints():
    """`{id: function}` for every `constraint_N` in `ported_constraints`."""
    out = {}
    for name, fn in vars(ported_constraints).items():
        if name.startswith("constraint_") and callable(fn):
            out[int(name.removeprefix("constraint_"))] = fn
    return out


def check_constraints(data, rtol=1e-9, atol=1e-9):
    """Every ported constraint, called with `data`'s own real values, compared against
    PROCESS's own real evaluation of the same constraint on the same `data`.
    """
    agreements, disagreements, errors = 0, [], []
    for cid, fn in sorted(_all_ported_constraints().items()):
        try:
            kwargs = _resolve_args(fn, data)
        except ValueError as e:
            errors.append((cid, str(e)))
            continue
        try:
            got = fn(**kwargs)
        except Exception as e:  # noqa: BLE001 -- report, don't crash the harness
            errors.append((cid, f"{fn.__name__} raised {type(e).__name__}: {e}"))
            continue
        registration = ConstraintManager.get_constraint(cid)
        try:
            expected_result = registration.constraint_equation(registration, data)
        except Exception as e:  # noqa: BLE001 -- e.g. constraint 44's own itart precondition
            errors.append((
                cid,
                f"PROCESS constraint_equation raised {type(e).__name__}: {e}",
            ))
            continue
        expected = (
            expected_result.residual,
            expected_result.normalised_residual,
            expected_result.constraint_value,
            expected_result.constraint_bound,
        )
        if all(_close(g, e, rtol, atol) for g, e in zip(got, expected, strict=True)):
            agreements += 1
        else:
            disagreements.append((cid, f"got={got} expected={expected}"))
    return agreements, disagreements, errors


def _all_ported_objectives():
    """`{FiguresOfMerit member: function}` -- reuses `OBJECTIVE_METRICS`, the same
    lookup `objectives.py` itself is built around.
    """
    return dict(ported_objectives.OBJECTIVE_METRICS)


def check_objectives(data, rtol=1e-9, atol=1e-9):
    """Every ported objective branch, called with `data`'s own real values, compared
    against PROCESS's own real `objective_function` for the same `data`.
    """
    agreements, disagreements, errors = 0, [], []
    for merit, fn in sorted(
        _all_ported_objectives().items(), key=lambda kv: kv[0].value
    ):
        try:
            kwargs = _resolve_args(fn, data)
        except ValueError as e:
            errors.append((merit.name, str(e)))
            continue
        try:
            got = fn(**kwargs)
        except Exception as e:  # noqa: BLE001
            errors.append((merit.name, f"{fn.__name__} raised {type(e).__name__}: {e}"))
            continue
        try:
            expected = objective_function(merit.value, data)
        except Exception as e:  # noqa: BLE001 -- e.g. objective_metric_15's own precondition
            errors.append((
                merit.name,
                f"PROCESS objective_function raised {type(e).__name__}: {e}",
            ))
            continue
        if _close(got, expected, rtol, atol):
            agreements += 1
        else:
            disagreements.append((merit.name, f"got={got} expected={expected}"))
    return agreements, disagreements, errors


def report(data) -> str:
    """A printable summary of `check_constraints`/`check_objectives` on `data`."""
    c_agree, c_dis, c_err = check_constraints(data)
    o_agree, o_dis, o_err = check_objectives(data)
    lines = [
        f"constraints: {c_agree} agree, {len(c_dis)} disagree, {len(c_err)} error",
        f"objectives:  {o_agree} agree, {len(o_dis)} disagree, {len(o_err)} error",
    ]
    if c_dis:
        lines.append("\nconstraint disagreements:")
        lines += [f"  constraint_{cid}: {detail}" for cid, detail in c_dis]
    if c_err:
        lines.append("\nconstraint errors:")
        lines += [f"  constraint_{cid}: {detail}" for cid, detail in c_err]
    if o_dis:
        lines.append("\nobjective disagreements:")
        lines += [f"  {name}: {detail}" for name, detail in o_dis]
    if o_err:
        lines.append("\nobjective errors:")
        lines += [f"  {name}: {detail}" for name, detail in o_err]
    return "\n".join(lines)


if __name__ == "__main__":
    data = converged_data("tests/regression/input_files/stellarator_helias.IN.DAT")
    print(report(data))
