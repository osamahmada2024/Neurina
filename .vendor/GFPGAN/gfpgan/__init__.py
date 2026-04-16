from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _load_real_gfpganer():
    current_init = Path(__file__).resolve()
    vendor_root = current_init.parents[1]
    wrapper_module = sys.modules.get("gfpgan")
    removed_entries = []

    for entry in list(sys.path):
        try:
            if Path(entry).resolve() == vendor_root:
                sys.path.remove(entry)
                removed_entries.append(entry)
        except Exception:
            continue

    sys.modules.pop("gfpgan", None)
    try:
        module = importlib.import_module("gfpgan")
        return module.GFPGANer
    finally:
        for entry in reversed(removed_entries):
            if entry not in sys.path:
                sys.path.insert(0, entry)
        if wrapper_module is not None and "gfpgan" not in sys.modules:
            sys.modules["gfpgan"] = wrapper_module


class GFPGANer:
    def __init__(self, *args, **kwargs):
        real_cls = _load_real_gfpganer()
        self._impl = real_cls(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._impl, name)


__all__ = ["GFPGANer"]
