"""VarPath -> valid Warp identifier, injective by construction.

VarPaths look like `.physics.rmajor` or `^stated.buildings.esbldgm3` (a `^`-minted name
from `cottax`'s scc-collapse naming). Neither is a legal Python/Warp identifier: leading
`.`/`^`, and internal `.`. This maps each *distinct* VarPath string to a *distinct*
identifier, deterministically within one `IdentifierMapper` instance (fresh state per
kernel build -- do not share one across configurations).
"""
from __future__ import annotations

import keyword
import re

_BAD = re.compile(r"[^0-9a-zA-Z_]")


class IdentifierMapper:
    def __init__(self):
        self._path_to_id: dict[str, str] = {}
        self._used: set[str] = set()

    def _sanitize(self, path: str) -> str:
        base = _BAD.sub("_", path)
        base = re.sub(r"_+", "_", base).strip("_")
        if not base:
            base = "v"
        if base[0].isdigit():
            base = "v_" + base
        if keyword.iskeyword(base) or base in ("wp", "tid"):
            base = base + "_"
        return "v_" + base

    def get(self, path: str) -> str:
        """Return the identifier for `path`, minting one on first sight.

        Injective: two distinct paths never receive the same identifier (collisions
        after sanitisation are broken with a numeric suffix); the same path always
        returns the same identifier within this instance's lifetime.
        """
        if path in self._path_to_id:
            return self._path_to_id[path]
        base = self._sanitize(path)
        ident = base
        n = 0
        while ident in self._used:
            n += 1
            ident = f"{base}__{n}"
        self._used.add(ident)
        self._path_to_id[path] = ident
        return ident

    def known(self, path: str) -> bool:
        return path in self._path_to_id

    def items(self):
        return self._path_to_id.items()
