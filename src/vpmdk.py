"""Compatibility wrapper for the vpmdk command and legacy imports."""
from __future__ import annotations

import importlib
import sys


_core = importlib.import_module("vpmdk_core")

if __name__ == "__main__":
    _core.main()
else:
    sys.modules[__name__] = _core
