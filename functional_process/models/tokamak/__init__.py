"""The conventional tokamak's own model units. Empty: nothing here is ported yet.

The namespace beside this file (`namespace.py`) names the slots those units will fill,
so the boundary check can say what is missing by name before anything is written. Mirrors
`process/models/` -- unlike `models/stellarator/`, the tokamak's models are scattered
across the top level of `process/models/` rather than gathered in one subpackage, which
is why `tokamak_call_surface.md` §A had to be traced rather than globbed.
"""
