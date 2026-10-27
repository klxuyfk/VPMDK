"""Opt-in energy, force, and stress checks for both BAM checkpoint variants."""

import os
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk

import vpmdk


@pytest.mark.integration
@pytest.mark.parametrize("model_variable", ["VPMDK_BAM_MODEL", "VPMDK_BAM_DENS_MODEL"])
def test_bam_checkpoint_properties(model_variable: str) -> None:
    model = os.environ.get(model_variable)
    if not model:
        pytest.skip(f"Set {model_variable} to run this BAM integration.")
    if not Path(model).is_file():
        pytest.skip(f"{model_variable} checkpoint is unavailable: {model}")
    pytest.importorskip("bam_torch")
    atoms = bulk("Si", "diamond", a=5.43)
    device = os.environ.get("VPMDK_BAM_DEVICE", "cpu")
    atoms.calc = vpmdk._build_bam_calculator({
        "MLP": "BAM",
        "MODEL": model,
        "DEVICE": device,
    })
    if device.lower().startswith("cuda"):
        # Upstream can fall back to CPU; a requested GPU check must catch that.
        assert str(atoms.calc.device).startswith("cuda")
        assert all(param.is_cuda for param in atoms.calc.model.parameters())
    assert np.isfinite(atoms.get_potential_energy())
    forces = atoms.get_forces()
    assert forces.shape == (len(atoms), 3)
    assert np.isfinite(forces).all()
    stress = atoms.get_stress()
    assert stress.shape == (6,)
    assert np.isfinite(stress).all()
