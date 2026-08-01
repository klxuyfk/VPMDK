"""Helpers shared across execution modes."""

from __future__ import annotations

import os
from contextlib import contextmanager

import numpy as np
from typing import Any, Iterable


def _resolve_calculator(calculator):
    if hasattr(calculator, "get_potential_energy"):
        return calculator
    inner_calculator = getattr(calculator, "calculator", None)
    if inner_calculator is not None and hasattr(inner_calculator, "get_potential_energy"):
        return inner_calculator
    return calculator


def _calculator_candidates(calculator: Any) -> list[Any]:
    """Return a calculator wrapper and its distinct resolved calculator."""

    candidates = [calculator]
    try:
        resolved = _resolve_calculator(calculator)
    except Exception:
        resolved = None
    if resolved is not None and resolved is not calculator:
        candidates.append(resolved)
    return candidates


def _thermostat_energy_terms(dyn) -> tuple[float, float]:
    """Return the thermostat's (potential, kinetic) energy in eV.

    VASP reports these as OSZICAR ``SP=``/``SK=`` (OUTCAR "nose potential"/"nose
    kinetic", vasprun.xml ``nosepot``/``nosekinetic``) and its ``E=`` is the
    CONSERVED quantity ``F + EK + SP + SK``. Probing for attribute names alone
    silently returned 0.0 for every thermostat ASE actually ships: ase 3.29's
    NoseHooverChainNVT keeps the chain state on a private ``_thermostat`` object and
    exposes only the SUM via ``get_thermostat_energy()``, and Bussi exposes only
    ``transferred_energy``. The reported total energy was therefore not conserved --
    measured drift of ~0.5-1.1 eV on runs whose true conserved energy was flat to a
    few meV -- so the standard "is ETOTAL conserved / is POTIM small enough" check
    rejected healthy trajectories.

    ``dyn.get_conserved_energy()`` is deliberately NOT used: it asks the calculator
    for ``free_energy``, which several MLIP calculators (CHGNet among them) do not
    implement, so it raises rather than returning a number.
    """

    potential = _extract_numeric_attribute(
        dyn,
        (
            "thermostat_potential_energy",
            "thermostat_potential",
            "nose_potential_energy",
            "nhc_potential_energy",
        ),
    )
    kinetic = _extract_numeric_attribute(
        dyn,
        (
            "thermostat_kinetic_energy",
            "thermostat_kinetic",
            "nose_kinetic_energy",
            "nhc_kinetic_energy",
        ),
    )
    if potential or kinetic:
        return potential, kinetic

    thermostat = getattr(dyn, "_thermostat", None)
    if thermostat is not None:
        try:
            # ase.md.nose_hoover_chain.NoseHooverChainThermostat.get_thermostat_energy
            # is exactly this sum, and its two halves ARE VASP's SP and SK:
            #   SP = 3 N kT eta[0] + kT * sum(eta[1:])   (chain "position" term)
            #   SK = sum(p_eta**2 / (2 Q))               (chain kinetic term)
            eta = np.asarray(thermostat._eta, dtype=float)
            p_eta = np.asarray(thermostat._p_eta, dtype=float)
            chain_masses = np.asarray(thermostat._Q, dtype=float)
            kT = float(thermostat._kT)
            num_atoms = float(thermostat._num_atoms_global)
            potential = 3.0 * num_atoms * kT * float(eta[0]) + kT * float(
                np.sum(eta[1:])
            )
            kinetic = float(np.sum(0.5 * p_eta**2 / chain_masses))
            return potential, kinetic
        except Exception:
            # A future ASE layout: keep the conserved TOTAL right even when the
            # split is not recoverable, rather than reporting zero for both.
            total = _extract_numeric_attribute(
                thermostat, ("get_thermostat_energy",)
            )
            if total:
                return total, 0.0

    # ase.md.bussi.Bussi accumulates the energy it has handed to the bath, and the
    # conserved quantity is Epot + Ekin - transferred_energy. There is no
    # potential/kinetic split to report for a stochastic rescaling thermostat.
    transferred = _extract_numeric_attribute(dyn, ("transferred_energy",))
    if transferred:
        return -transferred, 0.0

    return 0.0, 0.0


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
