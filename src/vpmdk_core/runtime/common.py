"""Helpers shared across execution modes."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterable


def _resolve_calculator(calculator):
    if hasattr(calculator, "get_potential_energy"):
        return calculator
    inner_calculator = getattr(calculator, "calculator", None)
    if inner_calculator is not None and hasattr(inner_calculator, "get_potential_energy"):
        return inner_calculator
    return calculator


def _extract_numeric_attribute(obj, names: Iterable[str]) -> float:
    """Return first numeric attribute or method result from ``names``."""

    for name in names:
        value = getattr(obj, name, None)
        if callable(value):
            try:
                result = value()
            except Exception:
                continue
            try:
                return float(result)
            except (TypeError, ValueError):
                continue
        else:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 0.0


@contextmanager
def _working_directory(path: str):
    """Temporarily change the current working directory."""

    original_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original_cwd)
