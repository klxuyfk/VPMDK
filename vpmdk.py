"""Compatibility wrapper for the vpmdk command and legacy imports."""
from __future__ import annotations

import importlib
import os
import sys
from types import ModuleType


def _load_core() -> ModuleType:
    root = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(root, "src")
    if os.path.isdir(src_path) and src_path not in sys.path:
        sys.path.insert(0, src_path)
    return importlib.import_module("vpmdk_core")


if __name__ == "__main__":
    root = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from vpmdk_entry import main

    raise SystemExit(main())
else:
    _core = _load_core()
    sys.modules[__name__] = _core
