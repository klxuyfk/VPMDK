"""Real MACE-SOYO checks, enabled explicitly with VPMDK_MACE_SOYO_MODEL."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from ase.io import read, write

import vpmdk
from test_integration_md import _run_vpmdk, _assert_outputs

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def calculator():
    model = os.environ.get("VPMDK_MACE_SOYO_MODEL")
    if not model:
        pytest.skip("Set VPMDK_MACE_SOYO_MODEL to a MACE-SOYO .pth checkpoint or .pt2 package")
    if os.environ.get("VPMDK_REQUIRE_MACE_SOYO_AOTI") == "1":
        assert Path(model).suffix.lower() == ".pt2", "AOTI verification requires an actual .pt2 package"
    pytest.importorskip("mace_soyo")
    device = os.environ.get("VPMDK_MACE_SOYO_DEVICE", "cuda")
    calc = vpmdk.build_calculator(vpmdk.BackendConfig(
        mlp="MACE_SOYO", model=model, device=device,
        options={"MACE_SOYO_CUEQ_METHOD": os.environ.get("VPMDK_MACE_SOYO_CUEQ_METHOD", "native")}
    ))
    # Verify the actual model placement, including an explicitly requested GPU.
    assert calc.device.type == "cuda"
    if Path(model).suffix.lower() == ".pt2":
        assert hasattr(calc, "aoti_model"), "Expected the compiled AOTI runtime"

    return calc


def test_energy_force_stress_and_gradients(calculator):
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.positions[1, 0] += 0.03
    atoms.calc = calculator
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()
    assert np.isfinite(energy)
    assert forces.shape == (2, 3) and np.isfinite(forces).all()
    assert stress.shape == (6,) and np.isfinite(stress).all()
    h = 1e-3
    energies = []
    for delta in [h, -h]:
        displaced = atoms.copy()
        displaced.positions[1, 0] += delta
        displaced.calc = calculator
        energies.append(displaced.get_potential_energy())
    assert forces[1, 0] == pytest.approx(-(energies[0] - energies[1]) / (2*h), abs=2e-3)
    # Float32 energy differences need a larger strain than displacements.
    h = 3e-3
    energies = []
    for delta in [h, -h]:
        strained = atoms.copy()
        cell = strained.cell.copy()
        cell[:, 0] *= 1 + delta
        strained.set_cell(cell, scale_atoms=True)
        strained.calc = calculator
        energies.append(strained.get_potential_energy())
    assert stress[0] == pytest.approx((energies[0] - energies[1]) / (2*h*atoms.get_volume()), abs=5e-5)


@pytest.mark.parametrize("ibrion", [-1, 2, 0])
def test_cli_workflows(calculator, tmp_path: Path, ibrion):
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.positions[1, 0] += 0.03
    write(tmp_path / "POSCAR", atoms, format="vasp")
    (tmp_path / "BCAR").write_text(
        f"MLP=MACE_SOYO\nMODEL={os.environ['VPMDK_MACE_SOYO_MODEL']}\n"
        f"DEVICE={os.environ.get('VPMDK_MACE_SOYO_DEVICE', 'cuda')}\n"
        f"MACE_SOYO_CUEQ_METHOD={os.environ.get('VPMDK_MACE_SOYO_CUEQ_METHOD', 'native')}\n"
    )
    (tmp_path / "INCAR").write_text(
        f"IBRION={ibrion}\nNSW=2\nISIF=2\nEDIFFG=-0.01\n"
        "POTIM=0.5\nTEBEG=100\nTEEND=100\nMDALGO=0\n"
    )
    _run_vpmdk(tmp_path)
    for name in ["OUTCAR", "OSZICAR", "CONTCAR", "vasprun.xml"]:
        assert (tmp_path / name).stat().st_size > 0
    final = read(tmp_path / "CONTCAR", format="vasp")
    assert np.isfinite(final.positions).all()
    if ibrion == 0:
        _assert_outputs(tmp_path)


def test_resident_reuses_calculator(calculator, tmp_path, monkeypatch):
    import threading
    import time
    from vpmdk_core.client import VPMDKClient, ServerConnectionError, RemoteBackendMismatch
    from vpmdk_core.server import VPMDKServer

    tags = {"MLP": "MACE_SOYO", "MODEL": os.environ["VPMDK_MACE_SOYO_MODEL"],
            "DEVICE": os.environ.get("VPMDK_MACE_SOYO_DEVICE", "cuda"),
            "MACE_SOYO_CUEQ_METHOD": os.environ.get("VPMDK_MACE_SOYO_CUEQ_METHOD", "native")}
    directories = []
    for index in range(2):
        directory = tmp_path / str(index)
        directory.mkdir()
        write(directory / "POSCAR", bulk("Si", "diamond", a=5.43), format="vasp")
        (directory / "INCAR").write_text("IBRION=-1\nNSW=0\n")
        (directory / "BCAR").write_text("\n".join(f"{k}={v}" for k,v in tags.items()))
        directories.append(directory)
    server = VPMDKServer(str(tmp_path / "server.sock"), calculator, tags,
                         backend_base_dir=str(tmp_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = VPMDKClient(str(tmp_path / "server.sock"))
    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                client.status()
                break
            except ServerConnectionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.02)
        def reject_reload(*args, **kwargs):
            pytest.fail("Resident request rebuilt the calculator")
        monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", reject_reload)
        for directory in directories:
            client.run(str(directory))
        assert client.status()["jobs_completed"] == 2
        assert server.backend["effective_configuration"]["MACE_SOYO_HEAD"] == calculator.head
        (directories[0] / "BCAR").write_text("MACE_SOYO_HEAD=wrong_head\n")
        with pytest.raises(RemoteBackendMismatch):
            client.run(str(directories[0]))
        # GPU scatter reductions can differ at the last float32 digit.
        energies = []
        for directory in directories:
            text = (directory / "OSZICAR").read_text()
            energies.append(float(text.split("F=", 1)[1].split()[0]))
        assert energies[0] == pytest.approx(energies[1], abs=1e-5, rel=0)
        assert read(directories[0] / "CONTCAR").positions == pytest.approx(
            read(directories[1] / "CONTCAR").positions, abs=1e-10
        )
    finally:
        client.stop(timeout=10)
        thread.join(timeout=10)
    assert not thread.is_alive()




def test_aoti_matches_eager_for_multiple_sizes(calculator):
    if Path(os.environ["VPMDK_MACE_SOYO_MODEL"]).suffix.lower() != ".pt2":
        pytest.skip("Requires a compiled .pt2 model")
    reference = os.environ.get("VPMDK_MACE_SOYO_REFERENCE_MODEL")
    if not reference:
        pytest.skip("Set VPMDK_MACE_SOYO_REFERENCE_MODEL to the original checkpoint")
    eager = vpmdk.build_calculator(vpmdk.BackendConfig(
        mlp="MACE_SOYO", model=reference,
        device=os.environ.get("VPMDK_MACE_SOYO_DEVICE", "cuda"),
        options={"MACE_SOYO_CUEQ_METHOD": os.environ.get("VPMDK_MACE_SOYO_REFERENCE_METHOD", "native")},
    ))
    for repeat in [(1, 1, 1), (2, 1, 1), (2, 2, 1)]:
        atoms = bulk("Si", "diamond", a=5.43).repeat(repeat)
        atoms.positions[1, 0] += 0.03
        atoms.calc = eager
        energy, forces, stress = atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress()
        atoms.calc = calculator
        assert (atoms.get_potential_energy() - energy) / len(atoms) == pytest.approx(0, abs=2e-5)
        assert atoms.get_forces() == pytest.approx(forces, abs=8e-4, rel=2e-3)
        assert atoms.get_stress() == pytest.approx(stress, abs=1e-4, rel=2e-3)
