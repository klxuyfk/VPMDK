"""Compatibility-oriented helpers layered on top of the core API."""

from __future__ import annotations

import sys


sys.modules.setdefault("vpmdk.compat", sys.modules[__name__])
