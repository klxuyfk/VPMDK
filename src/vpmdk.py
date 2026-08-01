"""Compatibility wrapper for the vpmdk command and legacy imports."""
from __future__ import annotations

import importlib
import sys


if __name__ == "__main__":
    from vpmdk_entry import main

    raise SystemExit(main())
else:
    _core = importlib.import_module("vpmdk_core")
    sys.modules[__name__] = _core
