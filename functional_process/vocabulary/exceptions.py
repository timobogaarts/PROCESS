"""`ProcessValueError`, vendored from `process/core/exceptions.py` (§23.2).

Kept structurally identical (`ProcessError` base, `**kwargs` diagnostics rendered into
the message) so a caller cannot tell the two apart by behaviour -- but it is a *distinct
class*, so `except process.core.exceptions.ProcessValueError` will not catch this one.
Nothing in the port or its tests does that today; both are `ValueError` subclasses, which
is what `models/power/thermal_cryo.py`'s only raise site relies on.
"""


class ProcessError(Exception):
    """A base Exception to derive other PROCESS exceptions from"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self._diagnostics = kwargs

    def __str__(self):
        """Exception message for ProcessError"""
        exception_message = super().__str__()
        diagnostics_message = "\n".join([
            f"\t{d}: {v!r}" for d, v in self._diagnostics.items()
        ])

        if diagnostics_message:
            return f"{exception_message}\n{diagnostics_message}"

        return exception_message


class ProcessValueError(ProcessError, ValueError):
    """A ValueError in a PROCESS model."""
