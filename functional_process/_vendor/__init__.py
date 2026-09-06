"""Verbatim copies of `process` modules the port needs without importing `process`.

**Not ported code.** Everything here is a transcription of a `process/**` module, kept
byte-equivalent in behaviour and equality-tested against its original, so the model layer
can run with `process` unavailable (`tests/test_process_free_import.py`). Ported code lives
in `functional_process/models/`; this is the seam where it does not.

Deliberately **not** under `models/`: `tests/test_process_free_import.py` imports every
module there to prove the layer needs no `process`, and `fluid_properties` pulls in
CoolProp -- a C extension costing ~3 s -- which would defeat
`test_importing_the_model_layer_does_not_load_coolprop`. Kept out of the package's top
level too, so `functional_process/` reads as port, harness and audit rather than as a
grab-bag.
"""
