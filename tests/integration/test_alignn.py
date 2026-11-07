"""Real ALIGNN-FF checks, enabled explicitly with VPMDK_ALIGNN_MODEL."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.io import read, write

import vpmdk
from test_integration_md import _run_vpmdk, _assert_outputs

pytestmark = pytest.mark.integration


def _assert_energy_derivatives(
    atoms: Atoms, *, step: float = 3e-3,
) -> None:
    """Check forces and finite-strain stress from the same reported energy."""
    assert np.isfinite(atoms.get_potential_energy())
    forces, stress = atoms.get_forces(), atoms.get_stress()
    assert forces.shape == (len(atoms), 3) and np.isfinite(forces).all()
    assert stress.shape == (6,) and np.isfinite(stress).all()
    energies = []
    h = step
    for delta in [h, -h]:
        displaced = atoms.copy()
        displaced.positions[1, 0] += delta
        displaced.calc = atoms.calc
        energies.append(displaced.get_potential_energy())
    assert forces[1, 0] == pytest.approx(
        -(energies[0] - energies[1]) / (2*h), abs=2e-3
    )
    energies = []
    for delta in [h, -h]:
        strained = atoms.copy()
        cell = strained.cell.copy()
        cell[:, 0] *= 1 + delta
        strained.set_cell(cell, scale_atoms=True)
        strained.calc = atoms.calc
        energies.append(strained.get_potential_energy())
    assert stress[0] == pytest.approx(
        (energies[0] - energies[1]) / (2*h*atoms.get_volume()), abs=5e-5
    )


@pytest.fixture(scope="module")
def calculator():
    model = os.environ.get("VPMDK_ALIGNN_MODEL")
    if not model:
        pytest.skip("Set VPMDK_ALIGNN_MODEL to an ALIGNN-FF directory, file, or model name")
    pytest.importorskip("alignn.ff.calculators")
    device = os.environ.get("VPMDK_ALIGNN_DEVICE", "cpu")
    calc = vpmdk.build_calculator(vpmdk.BackendConfig(
        mlp="ALIGNN", model=model, device=device
    ))
    # Verify the actual model placement, including an explicitly requested GPU.
    assert next(calc.model.parameters()).device.type == device.split(":")[0]
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


@pytest.mark.parametrize("pbc", [
    False, (False, True, True), (True, False, True), (True, True, False),
    (True, False, False), (False, True, False), (False, False, True),
])
def test_nonperiodic_inputs_rejected_before_upstream_conversion(calculator, monkeypatch, pbc):
    import alignn.ff.calculators as upstream

    atoms = bulk("Si", "diamond", a=5.43)
    atoms.pbc = pbc
    original_pbc = atoms.pbc.copy()
    atoms.calc = calculator
    def forbidden_conversion(structure):
        pytest.fail("Nonperiodic atoms reached the PBC-dropping upstream conversion")
    monkeypatch.setattr(upstream, "ase_to_atoms", forbidden_conversion)
    for evaluate in [atoms.get_potential_energy, atoms.get_forces, atoms.get_stress,
                     lambda: vpmdk.single_point(atoms, calculator=calculator)]:
        with pytest.raises(ValueError, match="ALIGNN requires fully periodic"):
            evaluate()
    np.testing.assert_array_equal(atoms.pbc, original_pbc)
    assert not calculator.model._forward_hooks


@pytest.mark.parametrize("ibrion", [-1, 2, 0])
def test_cli_workflows(calculator, tmp_path: Path, ibrion):
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.positions[1, 0] += 0.03
    write(tmp_path / "POSCAR", atoms, format="vasp")
    (tmp_path / "BCAR").write_text(
        f"MLP=ALIGNN\nMODEL={os.environ['VPMDK_ALIGNN_MODEL']}\n"
        f"DEVICE={os.environ.get('VPMDK_ALIGNN_DEVICE', 'cpu')}\n"
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


@pytest.mark.parametrize("energy_mult_natoms,force_mult_natoms", [
    (False, False), (False, True), (True, False), (True, True),
])
@pytest.mark.parametrize("include_pos_deriv", [False, True])
def test_atom_count_normalization_variants(
    calculator, tmp_path: Path, energy_mult_natoms, force_mult_natoms, include_pos_deriv
):
    """Check real-model force/stress derivatives and reject unusable branches."""
    import json
    from vpmdk_core.backends.alignn import _alignn_model_files

    reference = vpmdk._resolve_backend_model_reference("ALIGNN", os.environ["VPMDK_ALIGNN_MODEL"])
    if reference.kind is not vpmdk.ModelReferenceKind.LOCAL_PATH:
        pytest.skip("Set VPMDK_ALIGNN_MODEL to a local checkpoint for config variants")
    weights, config_path = _alignn_model_files(reference.value)
    config = json.loads(config_path.read_text())
    model = config["model"]
    name = model["name"]
    if name not in {"alignn_atomwise_pure", "alignn_atomwise"}:
        pytest.skip("Test uses the validated pure/DGL checkpoint architectures")
    model.update(energy_mult_natoms=energy_mult_natoms,
                 force_mult_natoms=force_mult_natoms, include_pos_deriv=include_pos_deriv)
    if name == "alignn_atomwise_pure":
        model["use_zbl"] = False  # Isolate the learned energy's normalization.
    (tmp_path / "best_model.pt").symlink_to(weights.resolve())
    (tmp_path / "config.json").write_text(json.dumps(config))
    backend = vpmdk.BackendConfig(mlp="ALIGNN", model=str(tmp_path),
                                  device=os.environ.get("VPMDK_ALIGNN_DEVICE", "cpu"))
    allowed = {(True, False), (False, True)}
    if name == "alignn_atomwise" and include_pos_deriv:
        error = "force normalization" if energy_mult_natoms else "Cartesian.*stress"
        with pytest.raises(ValueError, match=error):
            vpmdk.build_calculator(backend)
        return
    if (energy_mult_natoms, force_mult_natoms) not in allowed:
        with pytest.raises(ValueError, match="force normalization"):
            vpmdk.build_calculator(backend)
        return
    variant = vpmdk.build_calculator(backend)
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.positions[1, 0] += 0.1
    atoms.calc = variant
    _assert_energy_derivatives(atoms)


@pytest.mark.parametrize("updates,error", [
    ({"lg_on_fly": False}, "lg_on_fly"),
    ({"grad_multiplier": 1}, "grad_multiplier"),
    ({"stress_multiplier": 2}, "stress_multiplier"),
    ({"link": "log"}, "link=identity"),
    ({"link": "logit"}, "link=identity"),
    ({"classification": True}, "classification=false"),
    ({"output_features": 2}, "output_features=1"),
    ({"stresswise_weight": "omitted"}, None),
    ({"stresswise_weight": 0.}, None),
    ({"stresswise_weight": .5}, None),
])
def test_profile_configuration_variants(calculator, tmp_path: Path, updates, error):
    import json
    from vpmdk_core.backends.alignn import _alignn_model_files

    reference = vpmdk._resolve_backend_model_reference("ALIGNN", os.environ["VPMDK_ALIGNN_MODEL"])
    if reference.kind is not vpmdk.ModelReferenceKind.LOCAL_PATH:
        pytest.skip("Set VPMDK_ALIGNN_MODEL to a local checkpoint for config variants")
    weights, config_path = _alignn_model_files(reference.value)
    config = json.loads(config_path.read_text())
    config["model"].update(updates)
    if config["model"].get("stresswise_weight") == "omitted":
        del config["model"]["stresswise_weight"]
    (tmp_path / "best_model.pt").symlink_to(weights.resolve())
    selected_config = tmp_path / "config.json"
    selected_config.write_text(json.dumps(config))
    original = selected_config.read_bytes()
    backend = vpmdk.BackendConfig(mlp="ALIGNN", model=str(tmp_path),
                                  device=os.environ.get("VPMDK_ALIGNN_DEVICE", "cpu"))
    if error is not None:
        with pytest.raises(ValueError, match=error):
            vpmdk.build_calculator(backend)
    else:
        variant = vpmdk.build_calculator(backend)
        assert variant.model.config.stresswise_weight != 0
        atoms = bulk("Si", "diamond", a=5.43)
        atoms.positions[1, 0] += 0.1
        atoms.calc = variant
        _assert_energy_derivatives(atoms)
    assert selected_config.read_bytes() == original


@pytest.mark.parametrize("name,layers,error", [
    ("alignn_atomwise", 0, None),
    ("alignn_atomwise_pure", 0, "alignn_layers>0"),
    ("ealignn_atomwise", 1, "ealignn_atomwise.*torque"),
])
def test_small_checkpoint_profile_branches(calculator, tmp_path: Path, name, layers, error):
    """Use small genuine state dicts to test the extra architecture branches."""
    import json
    torch = pytest.importorskip("torch")
    pytest.importorskip("dgl")
    from alignn.models.alignn_atomwise import ALIGNNAtomWise, ALIGNNAtomWiseConfig
    from alignn.models.alignn_atomwise_pure import ALIGNNAtomWisePure, ALIGNNAtomWisePureConfig
    from alignn.models.ealignn_atomwise import eALIGNNAtomWise, eALIGNNAtomWiseConfig

    constructors = {
        "alignn_atomwise": (ALIGNNAtomWise, ALIGNNAtomWiseConfig),
        "alignn_atomwise_pure": (ALIGNNAtomWisePure, ALIGNNAtomWisePureConfig),
        "ealignn_atomwise": (eALIGNNAtomWise, eALIGNNAtomWiseConfig),
    }
    model_cls, config_cls = constructors[name]
    settings = dict(name=name, alignn_layers=layers, gcn_layers=1, atom_input_features=92,
                    embedding_features=16, hidden_features=16, stresswise_weight=.1,
                    lg_on_fly=False if layers == 0 else True)
    if name == "ealignn_atomwise":
        settings["energy_mult_natoms"] = False
    if name == "alignn_atomwise_pure":
        settings["use_zbl"] = False
    model_config = config_cls(**settings)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(2026)
        torch.save(model_cls(model_config).state_dict(), tmp_path / "best_model.pt")
    config = {"neighbor_strategy": "radius_graph", "cutoff": 4., "max_neighbors": 12,
              "atom_features": "cgcnn", "use_canonize": True, "model": model_config.model_dump()}
    (tmp_path / "config.json").write_text(json.dumps(config))
    backend = vpmdk.BackendConfig(mlp="ALIGNN", model=str(tmp_path),
                                  device=os.environ.get("VPMDK_ALIGNN_DEVICE", "cpu"))
    if error is not None:
        with pytest.raises(ValueError, match=error):
            vpmdk.build_calculator(backend)
    else:
        variant = vpmdk.build_calculator(backend)
        atoms = Atoms("SiGeSi", positions=[[0, 0, 0], [2, .2, .3], [.4, 2.2, .1]],
                      cell=[6, 6, 6], pbc=True)
        atoms.calc = variant
        assert np.max(np.abs(atoms.get_forces())) > 1e-5
        _assert_energy_derivatives(atoms)


@pytest.mark.parametrize("use_cutoff,multiply_cutoff,energy_mult_natoms,use_penalty", [
    (False, False, True, True),
    (True, False, True, True),
    (True, True, True, True),
    (True, False, False, True),
    (True, False, True, False),
])
def test_cutoff_penalty_energy_gradients(
    calculator, tmp_path: Path, use_cutoff, multiply_cutoff, energy_mult_natoms, use_penalty
):
    """Exercise cutoff/penalty branches with full force and strain derivatives."""
    import json
    import torch
    from vpmdk_core.backends.alignn import _alignn_model_files

    reference = vpmdk._resolve_backend_model_reference("ALIGNN", os.environ["VPMDK_ALIGNN_MODEL"])
    if reference.kind is not vpmdk.ModelReferenceKind.LOCAL_PATH:
        pytest.skip("Set VPMDK_ALIGNN_MODEL to a local checkpoint for config variants")
    weights, config_path = _alignn_model_files(reference.value)
    config = json.loads(config_path.read_text())
    config["model"].update(use_cutoff_function=use_cutoff, multiply_cutoff=multiply_cutoff,
                           energy_mult_natoms=energy_mult_natoms,
                           force_mult_natoms=not energy_mult_natoms, use_penalty=use_penalty)
    if config["model"]["name"] == "alignn_atomwise_pure":
        config["model"]["use_zbl"] = False
    (tmp_path / "best_model.pt").symlink_to(weights.resolve())
    selected_config = tmp_path / "config.json"
    selected_config.write_text(json.dumps(config))
    original = selected_config.read_bytes()
    # Cutoff-transformed inputs can produce large float32 neural energies.
    # Upcast the same checkpoint to resolve the smaller penalty derivatives.
    # The radius builder still quantizes coordinates to float32: a 3e-4 step
    # balances that roundoff against the transformed network's curvature.
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        variant = vpmdk.build_calculator(vpmdk.BackendConfig(
            mlp="ALIGNN", model=str(tmp_path), device=os.environ.get("VPMDK_ALIGNN_DEVICE", "cpu")
        ))
        atoms = Atoms("SiGeSi", positions=[[0, 0, 0], [2, .2, .3], [.4, 2.2, .1]],
                      cell=[6, 6, 6], pbc=True)
        atoms.calc = variant
        _assert_energy_derivatives(atoms, step=3e-4)
    finally:
        torch.set_default_dtype(previous_dtype)
    assert selected_config.read_bytes() == original


@pytest.mark.parametrize("name", ["alignn_atomwise_pure", "alignn_atomwise"])
def test_default_layer_checkpoint_first_evaluation(calculator, tmp_path: Path, name):
    """A genuine two-layer checkpoint must evaluate when its config omits the key."""
    import json
    import torch

    if name == "alignn_atomwise":
        pytest.importorskip("dgl")
        from alignn.models.alignn_atomwise import ALIGNNAtomWise as Model
        from alignn.models.alignn_atomwise import ALIGNNAtomWiseConfig as Config
    else:
        from alignn.models.alignn_atomwise_pure import ALIGNNAtomWisePure as Model
        from alignn.models.alignn_atomwise_pure import ALIGNNAtomWisePureConfig as Config
    settings = dict(name=name, gcn_layers=1, atom_input_features=92,
                    embedding_features=16, hidden_features=16, stresswise_weight=.1)
    if name == "alignn_atomwise_pure":
        settings["use_zbl"] = False
    model_config = Config(**settings)
    assert model_config.alignn_layers == 2
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(2026)
        torch.save(Model(model_config).state_dict(), tmp_path / "best_model.pt")
    raw_model = model_config.model_dump()
    del raw_model["alignn_layers"]
    config = {"model": raw_model, "neighbor_strategy": "radius_graph" if name == "alignn_atomwise"
              else "pure_torch", "cutoff": 4., "max_neighbors": 12,
              "atom_features": "cgcnn", "use_canonize": True}
    selected_config = tmp_path / "config.json"
    selected_config.write_text(json.dumps(config))
    original = selected_config.read_bytes()
    variant = vpmdk.build_calculator(vpmdk.BackendConfig(
        mlp="ALIGNN", model=str(tmp_path), device=os.environ.get("VPMDK_ALIGNN_DEVICE", "cpu")
    ))
    assert variant.config["model"]["alignn_layers"] == 2
    assert len(variant.model.alignn_layers) == 2
    atoms = Atoms("SiGeSi", positions=[[0, 0, 0], [2, .2, .3], [.4, 2.2, .1]],
                  cell=[6, 6, 6], pbc=True)
    atoms.calc = variant
    _assert_energy_derivatives(atoms)
    assert selected_config.read_bytes() == original


def test_resident_reuses_calculator(calculator, tmp_path, monkeypatch):
    import threading
    import time
    from vpmdk_core.client import VPMDKClient, ServerConnectionError
    from vpmdk_core.server import VPMDKServer

    tags = {"MLP": "ALIGNN", "MODEL": os.environ["VPMDK_ALIGNN_MODEL"],
            "DEVICE": os.environ.get("VPMDK_ALIGNN_DEVICE", "cpu")}
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



def test_gpu_matches_cpu(calculator):
    if next(calculator.model.parameters()).device.type != "cuda":
        pytest.skip("CPU/GPU comparison requires VPMDK_ALIGNN_DEVICE=cuda")
    cpu = vpmdk.build_calculator(vpmdk.BackendConfig(
        mlp="ALIGNN", model=os.environ["VPMDK_ALIGNN_MODEL"], device="cpu"
    ))
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.positions[1, 0] += 0.03
    atoms.calc = cpu
    energy, forces, stress = atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress()
    atoms.calc = calculator
    # Upstream float32 CUDA scatter reductions vary between evaluations;
    # the unwrapped calculator also differs by slightly over 1e-5 eV on Si2.
    assert atoms.get_potential_energy() == pytest.approx(energy, abs=2e-5, rel=0)
    assert atoms.get_forces() == pytest.approx(forces, abs=5e-5, rel=0)
    assert atoms.get_stress() == pytest.approx(stress, abs=1e-5, rel=0)


def test_single_atom_fcc_copper_static(calculator, tmp_path: Path):
    """Upstream returns (3,) for a one-atom cell; ASE requires (1, 3)."""
    atoms = bulk("Cu", "fcc", a=3.61)
    assert len(atoms) == 1
    atoms.calc = calculator
    assert np.isfinite(atoms.get_potential_energy())
    forces = atoms.get_forces()
    assert forces.shape == (1, 3)
    assert np.isfinite(forces).all()
    assert calculator.results["forces"].shape == (1, 3)
    write(tmp_path / "POSCAR", atoms, format="vasp")
    (tmp_path / "BCAR").write_text(
        f"MLP=ALIGNN\nMODEL={os.environ['VPMDK_ALIGNN_MODEL']}\n"
        f"DEVICE={os.environ.get('VPMDK_ALIGNN_DEVICE', 'cpu')}\n"
    )
    (tmp_path / "INCAR").write_text("IBRION=-1\nNSW=0\n")
    _run_vpmdk(tmp_path)
    for name in ["OUTCAR", "OSZICAR", "CONTCAR", "vasprun.xml"]:
        assert (tmp_path / name).stat().st_size > 0
    assert len(read(tmp_path / "CONTCAR", format="vasp")) == 1


def test_resident_rejects_different_sibling_config(calculator, tmp_path: Path):
    """Sharing weight bytes does not make different graph settings equivalent."""
    import json
    import threading
    import time
    from vpmdk_core.backends.alignn import _alignn_model_files
    from vpmdk_core.client import VPMDKClient, ServerConnectionError, RemoteBackendMismatch
    from vpmdk_core.server import VPMDKServer

    reference = vpmdk._resolve_backend_model_reference("ALIGNN", os.environ["VPMDK_ALIGNN_MODEL"])
    if reference.kind is not vpmdk.ModelReferenceKind.LOCAL_PATH:
        pytest.skip("Set VPMDK_ALIGNN_MODEL to a local checkpoint for the symlink regression")
    weights, config_path = _alignn_model_files(reference.value)
    original = json.loads(config_path.read_text())
    resident_model = tmp_path / "resident-model"
    request_model = tmp_path / "request-model"
    for directory in [resident_model, request_model]:
        directory.mkdir()
        (directory / "best_model.pt").symlink_to(weights.resolve())
    (resident_model / "config.json").write_bytes(config_path.read_bytes())
    changed = dict(original)
    changed["cutoff"] = float(original["cutoff"]) - 1
    (request_model / "config.json").write_text(json.dumps(changed))
    workdir = tmp_path / "calculation"
    workdir.mkdir()
    write(workdir / "POSCAR", bulk("Si", "diamond", a=5.43), format="vasp")
    (workdir / "INCAR").write_text("IBRION=-1\nNSW=0\n")
    tags = {"MLP": "ALIGNN", "MODEL": str(resident_model),
            "DEVICE": os.environ.get("VPMDK_ALIGNN_DEVICE", "cpu")}
    server = VPMDKServer(str(tmp_path / "config.sock"), calculator, tags, backend_base_dir=str(tmp_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = VPMDKClient(str(tmp_path / "config.sock"))
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
        (workdir / "BCAR").write_text(f"MODEL={resident_model / 'best_model.pt'}\n")
        client.run(str(workdir))
        (workdir / "BCAR").write_text(f"MODEL={request_model / 'best_model.pt'}\n")
        with pytest.raises(RemoteBackendMismatch):
            client.run(str(workdir))
        assert client.status()["jobs_completed"] == 1
    finally:
        client.stop(timeout=10)
        thread.join(timeout=10)
    assert not thread.is_alive()


def test_short_range_penalty_energy_matches_force_and_stress(calculator):
    """The default directed short-bond penalty must be present in total energy."""
    from ase import Atoms

    config = calculator.model.config
    if not getattr(config, "use_penalty", False):
        pytest.skip("Selected checkpoint does not enable the short-range penalty")
    if not 0.8 < config.penalty_threshold < 3:
        pytest.skip("Test geometry requires a short-bond threshold between 0.8 and 3 Å")
    if not config.energy_mult_natoms:
        pytest.skip("Test uses the default total-energy gradient convention")
    atoms = Atoms("Si2", positions=[[0, 0, 0], [0.8, 0, 0]], cell=[6, 6, 6], pbc=True)
    atoms.calc = calculator
    energy_on = atoms.get_potential_energy()
    forces_on = atoms.get_forces()
    stress_on = atoms.get_stress()
    factor = config.penalty_factor
    threshold = config.penalty_threshold
    try:
        config.use_penalty = False
        calculator.reset()
        energy_off = atoms.get_potential_energy()
        forces_off = atoms.get_forces()
        stress_off = atoms.get_stress()
    finally:
        config.use_penalty = True
        calculator.reset()
    # One short edge in each direction; preserve upstream's directed counting.
    assert energy_on - energy_off == pytest.approx(
        2 * factor * (threshold - 0.8), abs=2e-6
    )
    assert forces_on[1, 0] - forces_off[1, 0] == pytest.approx(2 * factor, abs=2e-5)
    assert stress_on[0] - stress_off[0] == pytest.approx(
        -2 * factor * 0.8 / atoms.get_volume(), abs=2e-6
    )

    def penalty_energy(structure: Atoms) -> float:
        # Subtract the same neural energy to resolve the small penalty gradient
        # despite float32 noise in this strongly repulsive configuration.
        values = []
        try:
            for enabled in [True, False]:
                config.use_penalty = enabled
                calculator.reset()
                structure.calc = calculator
                values.append(structure.get_potential_energy())
        finally:
            config.use_penalty = True
            calculator.reset()
        return values[0] - values[1]

    h = 1e-3
    energies = []
    for delta in [h, -h]:
        moved = atoms.copy()
        moved.positions[1, 0] += delta
        energies.append(penalty_energy(moved))
    assert forces_on[1, 0] - forces_off[1, 0] == pytest.approx(
        -(energies[0] - energies[1]) / (2*h), abs=2e-5
    )
    energies = []
    for delta in [h, -h]:
        strained = atoms.copy()
        cell = strained.cell.copy()
        cell[:, 0] *= 1 + delta
        strained.set_cell(cell, scale_atoms=True)
        energies.append(penalty_energy(strained))
    assert stress_on[0] - stress_off[0] == pytest.approx(
        (energies[0] - energies[1]) / (2*h*atoms.get_volume()), abs=2e-6
    )
