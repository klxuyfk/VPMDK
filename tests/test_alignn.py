"""Model selection and optional-dependency regressions for ALIGNN-FF."""
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

import vpmdk
from vpmdk_core.backends import alignn


def _checkpoint(path, name="best_model.pt"):
    path.mkdir(exist_ok=True)
    (path / name).write_bytes(b"checkpoint")
    (path / "config.json").write_text(json.dumps({
        "model": {"name": "alignn_atomwise_pure", "calculate_gradient": True}
    }))
    return path / name


@pytest.mark.parametrize("as_file", [False, True])
def test_local_model_and_device(tmp_path, monkeypatch, as_file):
    checkpoint = _checkpoint(tmp_path, "custom.pt" if as_file else "best_model.pt")
    monkeypatch.setattr(alignn, "_get_alignn_calculator_cls", lambda: lambda **kw: kw)
    result = vpmdk.build_calculator(vpmdk.BackendConfig(
        mlp="ALIGNN", model=str(checkpoint if as_file else tmp_path), device="cpu"
    ))
    assert result == dict(path=str(tmp_path), model_filename=checkpoint.name,
                          device="cpu", include_stress=True, intensive=True,
                          config={"model": {"name": "alignn_atomwise_pure",
                                            "calculate_gradient": True,
                                            "alignn_layers": 2,
                                            "stresswise_weight": 0.1}},
                          batch_stress=True, force_multiplier=1,
                          force_mult_natoms=False, force_mult_batchsize=False, stress_wt=1.)


def test_missing_path_rejected_before_import(tmp_path, monkeypatch):
    monkeypatch.setattr(alignn, "_get_alignn_calculator_cls", lambda: pytest.fail("imported"))
    with pytest.raises(FileNotFoundError):
        alignn._build_alignn_calculator({"MODEL": str(tmp_path / "missing.pt")})


@pytest.mark.parametrize("missing", ["config.json", "best_model.pt"])
def test_incomplete_directory(tmp_path, monkeypatch, missing):
    _checkpoint(tmp_path)
    (tmp_path / missing).unlink()
    monkeypatch.setattr(alignn, "_get_alignn_calculator_cls", lambda: lambda **kw: kw)
    with pytest.raises(FileNotFoundError, match="checkpoint and config.json"):
        alignn._build_alignn_calculator({"MODEL": str(tmp_path)})


def test_scalar_model_rejected(tmp_path, monkeypatch):
    _checkpoint(tmp_path)
    (tmp_path / "config.json").write_text('{"model": {"name": "alignn"}}')
    monkeypatch.setattr(alignn, "_get_alignn_calculator_cls", lambda: lambda **kw: kw)
    with pytest.raises(ValueError, match="force field"):
        alignn._build_alignn_calculator({"MODEL": str(tmp_path)})


def test_missing_dependency(monkeypatch):
    monkeypatch.setattr(alignn, "_get_alignn_calculator_cls", lambda: None)
    with pytest.raises(RuntimeError, match="Install alignn"):
        alignn._build_alignn_calculator({})


@pytest.mark.parametrize("name", [None, "other_ff", "unknown"])
def test_named_model_selection(tmp_path, monkeypatch, name):
    _checkpoint(tmp_path)
    module = ModuleType("alignn.ff.calculators")
    module.get_all_models = lambda: {"alignn_ff_mptrj": "url", "other_ff": "url"}
    calls = []
    def download(model_name):
        calls.append(model_name)
        return str(tmp_path)
    module.get_figshare_model_ff = download
    monkeypatch.setitem(sys.modules, "alignn.ff.calculators", module)
    monkeypatch.setattr(alignn, "_get_alignn_calculator_cls", lambda: lambda **kw: kw)
    tags = {} if name is None else {"MODEL": name}
    if name == "unknown":
        with pytest.raises(ValueError, match="Unsupported ALIGNN"):
            alignn._build_alignn_calculator(tags)
        assert calls == []
    else:
        alignn._build_alignn_calculator(tags)
        assert calls == [name or "alignn_ff_mptrj"]


@pytest.mark.parametrize("available", [False, True])
def test_backend_discovery(monkeypatch, available):
    monkeypatch.setattr(vpmdk, "_get_alignn_calculator_cls", lambda: object if available else None)
    spec = next(spec for spec in vpmdk.list_backends() if spec.name == "ALIGNN")
    assert spec.available is available
    assert spec.default_model == "alignn_ff_mptrj"
    assert spec.capabilities.energy and spec.capabilities.forces and spec.capabilities.stress
    assert not spec.capabilities.spin



def test_inconsistent_legacy_gradients_are_rejected(tmp_path, monkeypatch):
    _checkpoint(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"model": {
        "name": "alignn_atomwise", "calculate_gradient": True,
        "energy_mult_natoms": False, "force_mult_natoms": False,
    }}))
    original = config_path.read_bytes()
    monkeypatch.setattr(alignn, "_get_alignn_calculator_cls", lambda: lambda **kw: kw)
    with pytest.raises(ValueError, match="force normalization"):
        alignn._build_alignn_calculator({"MODEL": str(tmp_path), "DEVICE": "cpu"})
    assert config_path.read_bytes() == original


@pytest.mark.parametrize("model_name,include_pos_deriv,allowed", [
    ("alignn_atomwise_pure", False, {(True, False), (False, True)}),
    ("alignn_atomwise_pure", True, {(True, False), (False, True)}),
    ("alignn_atomwise", False, {(True, False), (False, True)}),
    ("alignn_atomwise", True, set()),  # Cartesian branch has no usable stress.
])
@pytest.mark.parametrize("energy_mult_natoms,force_mult_natoms", [
    (False, False), (False, True), (True, False), (True, True),
])
def test_force_normalization_combinations(
    tmp_path, monkeypatch, model_name, include_pos_deriv, allowed,
    energy_mult_natoms, force_mult_natoms,
):
    _checkpoint(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"model": {
        "name": model_name, "calculate_gradient": True,
        "energy_mult_natoms": energy_mult_natoms,
        "force_mult_natoms": force_mult_natoms,
        "include_pos_deriv": include_pos_deriv,
        "use_zbl": False,
    }}))
    original = config_path.read_bytes()
    calls = []
    def calculator(**kwargs):
        calls.append(kwargs)
        return kwargs
    monkeypatch.setattr(alignn, "_get_alignn_calculator_cls", lambda: calculator)
    if (energy_mult_natoms, force_mult_natoms) in allowed:
        assert alignn._build_alignn_calculator({"MODEL": str(tmp_path)})["intensive"]
        assert len(calls) == 1
    else:
        error = "force normalization"
        if model_name == "alignn_atomwise" and include_pos_deriv and not energy_mult_natoms:
            error = "Cartesian.*stress"
        with pytest.raises(ValueError, match=error):
            alignn._build_alignn_calculator({"MODEL": str(tmp_path)})
        assert not calls
    assert config_path.read_bytes() == original


@pytest.mark.parametrize("model_name,include_pos_deriv,accepted", [
    ("alignn_atomwise_pure", False, True),
    ("alignn_atomwise_pure", True, True),
    ("alignn_atomwise", False, True),
    ("alignn_atomwise", True, False),
])
def test_force_normalization_uses_upstream_atom_count_defaults(
    tmp_path, monkeypatch, model_name, include_pos_deriv, accepted
):
    _checkpoint(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"model": {
        "name": model_name, "calculate_gradient": True,
        "include_pos_deriv": include_pos_deriv,
    }}))
    monkeypatch.setattr(alignn, "_get_alignn_calculator_cls", lambda: lambda **kw: kw)
    if accepted:
        assert alignn._build_alignn_calculator({"MODEL": str(tmp_path)})["intensive"]
    else:
        with pytest.raises(ValueError, match="force normalization"):
            alignn._build_alignn_calculator({"MODEL": str(tmp_path)})


@pytest.mark.parametrize("zbl", ["omitted", None, True, False])
def test_alternate_pure_normalization_requires_disabled_zbl(tmp_path, monkeypatch, zbl):
    _checkpoint(tmp_path)
    config = {"model": {
        "name": "alignn_atomwise_pure", "calculate_gradient": True,
        "energy_mult_natoms": False, "force_mult_natoms": True,
    }}
    if zbl != "omitted":
        config["model"]["use_zbl"] = zbl
    (tmp_path / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(alignn, "_get_alignn_calculator_cls", lambda: lambda **kw: kw)
    if zbl is False:
        assert alignn._build_alignn_calculator({"MODEL": str(tmp_path)})["intensive"]
    else:
        with pytest.raises(ValueError, match="ZBL.*force normalization"):
            alignn._build_alignn_calculator({"MODEL": str(tmp_path)})


@pytest.mark.parametrize("key", [
    "energy_mult_natoms", "force_mult_natoms", "include_pos_deriv", "add_reverse_forces",
    "lg_on_fly", "classification", "calculate_gradient",
])
@pytest.mark.parametrize("value", ["false", 0])
def test_normalization_flags_require_json_booleans(tmp_path, monkeypatch, key, value):
    _checkpoint(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config["model"][key] = value
    config_path.write_text(json.dumps(config))
    def forbidden_calculator(**kwargs):
        pytest.fail("Invalid normalization flag reached model loading")
    monkeypatch.setattr(alignn, "_get_alignn_calculator_cls", lambda: forbidden_calculator)
    with pytest.raises(ValueError, match=f"{key} must be a JSON boolean"):
        alignn._build_alignn_calculator({"MODEL": str(tmp_path)})


@pytest.mark.parametrize("selection", ["directory", "file", "named"])
@pytest.mark.parametrize("model_name,include_pos_deriv", [
    ("alignn_atomwise", False),
    ("alignn_atomwise_pure", False),
    ("alignn_atomwise_pure", True),  # Pure PyTorch ignores include_pos_deriv.
])
def test_missing_reverse_forces_rejected_before_model_loading(
    tmp_path, monkeypatch, selection, model_name, include_pos_deriv
):
    checkpoint = _checkpoint(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"model": {
        "name": model_name, "calculate_gradient": True,
        "energy_mult_natoms": False, "force_mult_natoms": True,
        "include_pos_deriv": include_pos_deriv, "add_reverse_forces": False,
    }}))
    original = config_path.read_bytes()
    def forbidden_calculator(**kwargs):
        pytest.fail("Inconsistent checkpoint was loaded")
    monkeypatch.setattr(alignn, "_get_alignn_calculator_cls", lambda: forbidden_calculator)
    selections = {"directory": str(tmp_path), "file": str(checkpoint), "named": "alignnff_fmult"}
    if selection == "named":
        module = ModuleType("alignn.ff.calculators")
        module.get_all_models = lambda: {"alignnff_fmult": "url"}
        module.get_figshare_model_ff = lambda model_name: str(tmp_path)
        monkeypatch.setitem(sys.modules, "alignn.ff.calculators", module)
    with pytest.raises(ValueError, match="add_reverse_forces=false"):
        alignn._build_alignn_calculator({"MODEL": selections[selection], "DEVICE": "cpu"})
    assert config_path.read_bytes() == original


@pytest.mark.parametrize("model_name", ["alignn_atomwise", "alignn_atomwise_pure"])
@pytest.mark.parametrize("reverse_forces", [None, True])
def test_complete_reverse_force_convention_remains_accepted(
    tmp_path, monkeypatch, model_name, reverse_forces
):
    _checkpoint(tmp_path)
    config = {"model": {"name": model_name, "calculate_gradient": True,
                        "energy_mult_natoms": True}}
    if reverse_forces is not None:
        config["model"]["add_reverse_forces"] = reverse_forces
    (tmp_path / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(alignn, "_get_alignn_calculator_cls", lambda: lambda **kw: kw)
    result = alignn._build_alignn_calculator({"MODEL": str(tmp_path), "DEVICE": "cpu"})
    assert result["path"] == str(tmp_path)


@pytest.mark.parametrize("selection", ["directory", "file", "named"])
@pytest.mark.parametrize("force_mult_natoms", [False, True])
@pytest.mark.parametrize("stresswise_weight", [0.0, 0.1])
def test_cartesian_stress_rejected_before_model_loading(
    tmp_path, monkeypatch, selection, force_mult_natoms, stresswise_weight
):
    checkpoint = _checkpoint(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"model": {
        "name": "alignn_atomwise", "calculate_gradient": True,
        "energy_mult_natoms": False, "force_mult_natoms": force_mult_natoms,
        "include_pos_deriv": True, "stresswise_weight": stresswise_weight,
    }}))
    original = config_path.read_bytes()
    def forbidden_calculator(**kwargs):
        pytest.fail("Cartesian model without usable stress was loaded")
    monkeypatch.setattr(alignn, "_get_alignn_calculator_cls", lambda: forbidden_calculator)
    selections = {"directory": str(tmp_path), "file": str(checkpoint), "named": "cartesian_ff"}
    if selection == "named":
        module = ModuleType("alignn.ff.calculators")
        module.get_all_models = lambda: {"cartesian_ff": "url"}
        module.get_figshare_model_ff = lambda model_name: str(tmp_path)
        monkeypatch.setitem(sys.modules, "alignn.ff.calculators", module)
    with pytest.raises(ValueError, match="Cartesian.*stress"):
        alignn._build_alignn_calculator({"MODEL": selections[selection], "DEVICE": "cpu"})
    assert config_path.read_bytes() == original


@pytest.mark.parametrize("model_name,include_pos_deriv", [
    ("alignn_atomwise", True),
])
def test_unused_reverse_force_flag_is_not_reported_as_missing_derivatives(
    tmp_path, monkeypatch, model_name, include_pos_deriv
):
    _checkpoint(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"model": {
        "name": model_name, "calculate_gradient": True,
        "energy_mult_natoms": False, "force_mult_natoms": True,
        "include_pos_deriv": include_pos_deriv, "add_reverse_forces": False,
    }}))
    monkeypatch.setattr(alignn, "_get_alignn_calculator_cls", lambda: lambda **kw: kw)
    if model_name == "alignn_atomwise":
        with pytest.raises(ValueError, match="Cartesian.*stress"):
            alignn._build_alignn_calculator({"MODEL": str(tmp_path)})
    else:
        assert alignn._build_alignn_calculator({"MODEL": str(tmp_path)})["path"] == str(tmp_path)


@pytest.mark.parametrize("selection", ["directory", "file", "named"])
@pytest.mark.parametrize("name", ["alignn_atomwise_pure", "alignn_atomwise"])
@pytest.mark.parametrize("updates,error", [
    ({"lg_on_fly": False}, "lg_on_fly"),
    ({"grad_multiplier": 1}, "grad_multiplier"),
    ({"stress_multiplier": 2}, "stress_multiplier"),
    ({"link": "log"}, "link=identity"),
    ({"link": "logit"}, "link=identity"),
    ({"classification": True}, "classification=false"),
    ({"output_features": 2}, "output_features=1"),
])
def test_nonconservative_profiles_rejected_before_loading(
    tmp_path, monkeypatch, selection, name, updates, error
):
    checkpoint = _checkpoint(tmp_path)
    model = {"name": name, "calculate_gradient": True, **updates}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"model": model}))
    original = config_path.read_bytes()
    def forbidden_calculator(**kwargs):
        pytest.fail("Nonconservative model reached weight loading")
    monkeypatch.setattr(alignn, "_get_alignn_calculator_cls", lambda: forbidden_calculator)
    if selection == "named":
        module = ModuleType("alignn.ff.calculators")
        module.get_all_models = lambda: {"custom_ff": "url"}
        module.get_figshare_model_ff = lambda model_name: str(tmp_path)
        monkeypatch.setitem(sys.modules, "alignn.ff.calculators", module)
    reference = {"directory": str(tmp_path), "file": str(checkpoint), "named": "custom_ff"}[selection]
    with pytest.raises(ValueError, match=error):
        alignn._build_alignn_calculator({"MODEL": reference})
    assert config_path.read_bytes() == original


@pytest.mark.parametrize("torque", [None, True, False])
def test_unvalidated_ealignn_architecture_is_rejected(tmp_path, monkeypatch, torque):
    _checkpoint(tmp_path)
    model = {"name": "ealignn_atomwise", "calculate_gradient": True,
             "energy_mult_natoms": False}
    if torque is not None:
        model["remove_torque"] = torque
    (tmp_path / "config.json").write_text(json.dumps({"model": model}))
    def forbidden_calculator(**kwargs):
        pytest.fail("Unsupported eALIGNN reached weight loading")
    monkeypatch.setattr(alignn, "_get_alignn_calculator_cls", lambda: forbidden_calculator)
    with pytest.raises(ValueError, match="ealignn_atomwise.*torque"):
        alignn._build_alignn_calculator({"MODEL": str(tmp_path)})


@pytest.mark.parametrize("key", ["grad_multiplier", "stress_multiplier", "stresswise_weight"])
@pytest.mark.parametrize("value", ["1", True, float("nan"), float("inf")])
def test_numeric_profile_settings_are_finite_json_numbers(tmp_path, monkeypatch, key, value):
    _checkpoint(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config["model"][key] = value
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(alignn, "_get_alignn_calculator_cls", lambda: lambda **kw: kw)
    with pytest.raises(ValueError, match=f"{key} must be a finite JSON number"):
        alignn._build_alignn_calculator({"MODEL": str(tmp_path)})


@pytest.mark.parametrize("name,layers,accepted", [
    ("alignn_atomwise", 0, True),
    ("alignn_atomwise", 1, False),
    ("alignn_atomwise_pure", 0, False),
    ("alignn_atomwise_pure", 1, False),
])
def test_angle_validation_uses_actual_layer_branch(tmp_path, monkeypatch, name, layers, accepted):
    _checkpoint(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"model": {
        "name": name, "calculate_gradient": True, "alignn_layers": layers, "lg_on_fly": False,
    }}))
    monkeypatch.setattr(alignn, "_get_alignn_calculator_cls", lambda: lambda **kw: kw)
    if accepted:
        assert alignn._build_alignn_calculator({"MODEL": str(tmp_path)})["include_stress"]
    else:
        with pytest.raises(ValueError, match="alignn_layers"):
            alignn._build_alignn_calculator({"MODEL": str(tmp_path)})


@pytest.mark.parametrize("weight", [None, 0., 0.5])
def test_stress_is_enabled_in_private_runtime_configuration(tmp_path, monkeypatch, weight):
    _checkpoint(tmp_path)
    config_path = tmp_path / "config.json"
    original = json.loads(config_path.read_text())
    if weight is not None:
        original["model"]["stresswise_weight"] = weight
    config_path.write_text(json.dumps(original))
    original_bytes = config_path.read_bytes()
    monkeypatch.setattr(alignn, "_get_alignn_calculator_cls", lambda: lambda **kw: kw)
    result = alignn._build_alignn_calculator({"MODEL": str(tmp_path)})
    expected = json.loads(original_bytes)
    expected["model"]["stresswise_weight"] = weight or .1
    expected["model"].setdefault("alignn_layers", 2)
    assert result["config"] == expected
    assert result["include_stress"] and result["batch_stress"]
    result["config"]["model"]["stresswise_weight"] = 99  # Upstream mutates its own config.
    assert config_path.read_bytes() == original_bytes


@pytest.mark.parametrize("name", ["alignn_atomwise", "alignn_atomwise_pure"])
@pytest.mark.parametrize("selection", ["directory", "file", "named"])
def test_default_layer_count_is_forwarded_without_editing_config(
    tmp_path, monkeypatch, name, selection
):
    checkpoint = _checkpoint(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"model": {"name": name, "calculate_gradient": True}}))
    original = config_path.read_bytes()
    monkeypatch.setattr(alignn, "_get_alignn_calculator_cls", lambda: lambda **kw: kw)
    if selection == "named":
        module = ModuleType("alignn.ff.calculators")
        module.get_all_models = lambda: {"custom_ff": "url"}
        module.get_figshare_model_ff = lambda model_name: str(tmp_path)
        monkeypatch.setitem(sys.modules, "alignn.ff.calculators", module)
    reference = {"directory": str(tmp_path), "file": str(checkpoint), "named": "custom_ff"}[selection]
    result = alignn._build_alignn_calculator({"MODEL": reference})
    assert result["config"]["model"]["alignn_layers"] == 2
    assert config_path.read_bytes() == original


@pytest.mark.parametrize("natoms", [1, 2])
def test_upstream_squeezed_forces_are_normalized_and_cached(tmp_path, monkeypatch, natoms):
    import numpy as np
    from ase import Atoms
    from ase.calculators.calculator import Calculator

    expected = np.arange(3 * natoms, dtype=np.float64).reshape(natoms, 3)
    class UpstreamCalculator(Calculator):
        implemented_properties = ["energy", "forces", "stress"]
        def __init__(self, **kwargs):
            super().__init__()
            self.calls = 0
        def calculate(self, atoms, properties=None, system_changes=None):
            super().calculate(atoms, properties, system_changes)
            self.calls += 1
            self.results = {"energy": -1.25, "forces": expected.squeeze().copy(),
                            "stress": np.zeros(6)}

    module = ModuleType("alignn.ff.calculators")
    module.AlignnAtomwiseCalculator = UpstreamCalculator
    monkeypatch.setitem(sys.modules, "alignn.ff.calculators", module)
    _checkpoint(tmp_path)
    calc = alignn._build_alignn_calculator({"MODEL": str(tmp_path), "DEVICE": "cpu"})
    atoms = Atoms("Cu" * natoms, positions=np.zeros((natoms, 3)), cell=[4, 4, 4], pbc=True)
    atoms.calc = calc
    assert atoms.get_potential_energy() == -1.25
    forces = vpmdk._safe_get_forces(atoms, strict=True)
    assert forces.shape == (natoms, 3)
    np.testing.assert_array_equal(forces, expected)
    np.testing.assert_array_equal(calc.results["forces"], expected)
    assert calc.calls == 1
    atoms.get_forces()
    atoms.get_stress()
    assert calc.calls == 1
    atoms.positions[0, 0] += 0.01
    np.testing.assert_array_equal(atoms.get_forces(), expected)
    assert calc.calls == 2


@pytest.fixture
def pbc_calculator(tmp_path, monkeypatch):
    import numpy as np
    from ase.calculators.calculator import Calculator

    calls = []
    class UpstreamCalculator(Calculator):
        implemented_properties = ["energy", "forces", "stress"]
        def __init__(self, **kwargs):
            super().__init__()
        def calculate(self, atoms, properties=None, system_changes=None):
            if not np.all(atoms.pbc):
                pytest.fail("Nonperiodic input reached upstream calculation")
            super().calculate(atoms, properties, system_changes)
            calls.append(atoms.pbc.copy())
            self.results = {"energy": 42., "forces": np.zeros((len(atoms), 3)),
                            "stress": np.ones(6)}

    module = ModuleType("alignn.ff.calculators")
    module.AlignnAtomwiseCalculator = UpstreamCalculator
    monkeypatch.setitem(sys.modules, "alignn.ff.calculators", module)
    _checkpoint(tmp_path)
    backend = vpmdk.BackendConfig(mlp="ALIGNN", model=str(tmp_path), device="cpu")
    return backend, vpmdk.build_calculator(backend), calls


@pytest.mark.parametrize("pbc", [
    True, False, (False, True, True), (True, False, True), (True, True, False),
    (True, False, False), (False, True, False), (False, False, True),
])
@pytest.mark.parametrize("entry", ["energy", "forces", "stress", "single_point_calc", "single_point_backend"])
def test_pbc_contract_at_public_calculation_entry(pbc_calculator, pbc, entry):
    import numpy as np
    from ase import Atoms

    backend, calculator, calls = pbc_calculator
    atoms = Atoms("Si2", positions=[[0, 0, 0], [2, 0, 0]], cell=[6, 6, 6], pbc=pbc)
    original_pbc = atoms.pbc.copy()
    atoms.calc = calculator
    def evaluate():
        if entry == "single_point_calc":
            return vpmdk.single_point(atoms, calculator=calculator)
        if entry == "single_point_backend":
            return vpmdk.single_point(atoms, backend=backend)
        return getattr(atoms, {"energy": "get_potential_energy", "forces": "get_forces",
                               "stress": "get_stress"}[entry])()
    if np.all(original_pbc):
        evaluate()
        assert len(calls) == 1
    else:
        with pytest.raises(ValueError, match="ALIGNN requires fully periodic"):
            evaluate()
        assert not calls
    np.testing.assert_array_equal(atoms.pbc, original_pbc)


def test_pbc_changes_cannot_reuse_a_periodic_result(pbc_calculator):
    import numpy as np
    from ase import Atoms

    _, calculator, calls = pbc_calculator
    atoms = Atoms("Si2", cell=[6, 6, 6], pbc=True)
    atoms.calc = calculator
    assert atoms.get_potential_energy() == 42.
    atoms.pbc[2] = False
    with pytest.raises(ValueError, match="ALIGNN requires fully periodic"):
        atoms.get_potential_energy()
    assert len(calls) == 1
    np.testing.assert_array_equal(atoms.pbc, [True, True, False])
    atoms.pbc = True
    assert atoms.get_potential_energy() == 42.
    assert all(np.all(pbc) for pbc in calls)


def test_pbc_guard_checks_stored_atoms_when_argument_is_none(pbc_calculator):
    from ase import Atoms

    _, calculator, calls = pbc_calculator
    calculator.atoms = Atoms("Si2", cell=[6, 6, 6], pbc=[True, True, False])
    with pytest.raises(ValueError, match="ALIGNN requires fully periodic"):
        calculator.calculate(None)
    assert not calls


def test_invalid_upstream_force_count_is_not_silently_replaced(monkeypatch):
    import numpy as np
    from ase import Atoms
    from ase.calculators.calculator import Calculator

    class UpstreamCalculator(Calculator):
        implemented_properties = ["forces"]
        def calculate(self, atoms, properties=None, system_changes=None):
            super().calculate(atoms, properties, system_changes)
            self.results = {"forces": np.array([1., 2.])}

    module = ModuleType("alignn.ff.calculators")
    module.AlignnAtomwiseCalculator = UpstreamCalculator
    monkeypatch.setitem(sys.modules, "alignn.ff.calculators", module)
    atoms = Atoms("Cu", cell=[4, 4, 4], pbc=True)
    atoms.calc = alignn._get_alignn_calculator_cls()()
    with pytest.raises(ValueError):
        atoms.get_forces()


def test_resident_rejects_shared_weights_with_different_sibling_config(tmp_path, monkeypatch):
    from vpmdk_core.server import backend_identity, validate_request_backend, BackendConfigurationMismatch

    weights = tmp_path / "shared.pt"
    weights.write_bytes(b"shared weights")
    paths = []
    for name, cutoff in [("resident", 5), ("request", 4)]:
        directory = tmp_path / name
        _checkpoint(directory)
        (directory / "best_model.pt").unlink()
        (directory / "best_model.pt").symlink_to(weights)
        config_path = directory / "config.json"
        config = json.loads(config_path.read_text())
        config["cutoff"] = cutoff
        config_path.write_text(json.dumps(config))
        paths.append(directory / "best_model.pt")
    def calculator(**kwargs):
        return json.loads((Path(kwargs["path"]) / "config.json").read_text())
    monkeypatch.setattr(alignn, "_get_alignn_calculator_cls", lambda: calculator)
    assert alignn._build_alignn_calculator({"MODEL": str(paths[0])})["cutoff"] == 5
    assert alignn._build_alignn_calculator({"MODEL": str(paths[1])})["cutoff"] == 4
    resident = backend_identity({"MLP": "ALIGNN", "MODEL": str(paths[0]), "DEVICE": "cpu"}, base_dir=str(tmp_path))
    with pytest.raises(BackendConfigurationMismatch):
        validate_request_backend(resident, {"MODEL": str(paths[1])}, request_base_dir=str(tmp_path))


def test_resident_accepts_weight_and_config_aliases_and_directory_model(tmp_path):
    from vpmdk_core.server import backend_identity, validate_request_backend

    directory = tmp_path / "resident"
    weights = _checkpoint(directory)
    aliases = tmp_path / "aliases"
    aliases.mkdir()
    (aliases / "weights.pt").symlink_to(weights)
    (aliases / "config.json").symlink_to(directory / "config.json")
    resident = backend_identity({"MLP": "ALIGNN", "MODEL": str(directory), "DEVICE": "cpu"}, base_dir=str(tmp_path))
    assert resident["model"] == str(weights.resolve())
    assert resident["effective_configuration"]["ALIGNN_CONFIG"] == str((directory / "config.json").resolve())
    validate_request_backend(resident, {"MODEL": "weights.pt"}, request_base_dir=str(aliases))
    validate_request_backend(resident, {"MODEL": str(weights)}, request_base_dir=str(tmp_path))
    validate_request_backend(resident, {}, request_base_dir=str(aliases))


def test_resident_rejects_changed_or_missing_sibling_config(tmp_path):
    from vpmdk_core.server import backend_identity, validate_request_backend, BackendConfigurationMismatch

    weights = _checkpoint(tmp_path)
    resident = backend_identity({"MLP": "ALIGNN", "MODEL": str(weights), "DEVICE": "cpu"}, base_dir=str(tmp_path))
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config["cutoff"] = 4
    config_path.write_text(json.dumps(config))
    with pytest.raises(BackendConfigurationMismatch, match="ALIGNN_CONFIG_SHA256"):
        validate_request_backend(resident, {"MODEL": str(weights)}, request_base_dir=str(tmp_path))
    config_path.unlink()
    with pytest.raises(BackendConfigurationMismatch):
        validate_request_backend(resident, {"MODEL": str(weights)}, request_base_dir=str(tmp_path))
    # Omitted MODEL intentionally inherits the already loaded resident model.
    validate_request_backend(resident, {}, request_base_dir=str(tmp_path))


def test_named_resident_identity_does_not_load_or_download(monkeypatch, tmp_path):
    from vpmdk_core.server import backend_identity, validate_request_backend

    monkeypatch.setattr(alignn, "_get_alignn_calculator_cls", lambda: pytest.fail("optional runtime imported"))
    resident = backend_identity({"MLP": "ALIGNN", "DEVICE": "cpu"}, base_dir=str(tmp_path))
    validate_request_backend(resident, {"MODEL": "alignn_ff_mptrj"}, request_base_dir=str(tmp_path))


@pytest.fixture
def penalty_calculators(monkeypatch):
    """Represent both upstream output conventions on a chosen inference graph."""
    import numpy as np
    from types import SimpleNamespace
    from ase.calculators.calculator import Calculator

    torch = pytest.importorskip("torch")

    def cutoff_transform(lengths, *, inner_cutoff, exponent):
        # Distinct, analytically simple stand-in for the upstream transformation.
        assert inner_cutoff == 3. and exponent == 5
        return lengths / 2

    utils = ModuleType("alignn.models.utils")
    utils.cutoff_function_based_edges = cutoff_transform
    monkeypatch.setitem(sys.modules, "alignn.models.utils", utils)

    class Model(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config

        def forward(self, inputs):
            graph = inputs[0]
            r = graph.edata["r"].requires_grad_(True)
            # Include a separate energy contribution that must remain intact.
            out = r.square().sum() / 3 + 0.7
            en = out * 3 if self.config.energy_mult_natoms else out
            if self.config.use_penalty:
                lengths = r.norm(dim=1)
                if (self.config.name == "alignn_atomwise" and self.config.use_cutoff_function
                        and not self.config.multiply_cutoff):
                    lengths = cutoff_transform(lengths, inner_cutoff=self.config.inner_cutoff,
                                               exponent=self.config.exponent)
                penalty = 0.1 * torch.relu(1 - lengths).sum()
                if self.config.name == "alignn_atomwise_pure":
                    en = en + penalty
                else:
                    en += penalty  # DGL aliases `out` when energy_mult_natoms=False.
            pair_forces = -torch.autograd.grad(en, r)[0]
            return {"out": out, "grad": pair_forces[:3]}

    class UpstreamCalculator(Calculator):
        implemented_properties = ["energy", "forces", "stress"]

        def __init__(self, name, energy_mult_natoms, use_penalty, intensive):
            super().__init__()
            self.model = Model(SimpleNamespace(
                name=name, energy_mult_natoms=energy_mult_natoms,
                use_penalty=use_penalty, penalty_factor=0.1, penalty_threshold=1.,
                use_cutoff_function=False, multiply_cutoff=False, inner_cutoff=3., exponent=5,
            ))
            self.intensive = intensive
            # Include extra short periodic-image edges and an edge at threshold.
            # These graph edges cannot be recovered from the isolated positions.
            self.edges = [[.8, 0, 0], [-.8, 0, 0], [.6, 0, 0], [-.6, 0, 0],
                          [2., 0, 0], [1., 0, 0]]
            self.calls = 0

        def calculate(self, atoms, properties=None, system_changes=None):
            super().calculate(atoms, properties, system_changes)
            self.calls += 1
            graph = SimpleNamespace(edata={"r": torch.tensor(self.edges, dtype=torch.float64)})
            result = self.model((graph, None, None))
            self.results = {
                "energy": result["out"].item() * (len(atoms) if self.intensive else 1),
                "forces": result["grad"].detach().numpy(),
                "stress": np.arange(6, dtype=float),
            }

    module = ModuleType("alignn.ff.calculators")
    module.AlignnAtomwiseCalculator = UpstreamCalculator
    monkeypatch.setitem(sys.modules, "alignn.ff.calculators", module)
    return UpstreamCalculator, alignn._get_alignn_calculator_cls()


@pytest.mark.parametrize("name", ["alignn_atomwise_pure", "alignn_atomwise", "ealignn_atomwise"])
@pytest.mark.parametrize("energy_mult_natoms", [True, False])
@pytest.mark.parametrize("intensive", [True, False])
@pytest.mark.parametrize("use_penalty", [True, False])
def test_penalty_energy_preserves_upstream_gradients_and_cache(
    penalty_calculators, name, energy_mult_natoms, intensive, use_penalty
):
    import numpy as np
    from ase import Atoms

    original_cls, adapted_cls = penalty_calculators
    args = (name, energy_mult_natoms, use_penalty, intensive)
    original, adapted = original_cls(*args), adapted_cls(*args)
    atoms = Atoms("Si3", positions=[[0, 0, 0], [2, 0, 0], [4, 0, 0]], pbc=True)
    atoms.calc = original
    raw_energy = atoms.get_potential_energy()
    raw_forces, raw_stress = atoms.get_forces(), atoms.get_stress()
    # Four directed short edges contribute 0.12 eV before output normalization.
    missing = use_penalty and (energy_mult_natoms or name == "alignn_atomwise_pure")
    correction = 0.12 * (3 if intensive else 1) / (3 if energy_mult_natoms else 1)
    atoms.calc = adapted
    energy = atoms.get_potential_energy()
    assert energy == pytest.approx(raw_energy + (correction if missing else 0))
    np.testing.assert_array_equal(atoms.get_forces(), raw_forces)
    np.testing.assert_array_equal(atoms.get_stress(), raw_stress)
    assert atoms.get_potential_energy() == energy
    assert adapted.calls == 1
    assert not adapted.model._forward_hooks
    # An independent calculation with no short edges must not reuse a correction.
    adapted.edges = [[2., 0, 0]] * 6
    original.edges = adapted.edges
    atoms.positions[1, 0] += 0.01
    adapted.calculate(atoms)
    original.calculate(atoms)
    assert adapted.results["energy"] == original.results["energy"]
    assert not adapted.model._forward_hooks


@pytest.mark.parametrize("name", ["alignn_atomwise", "alignn_atomwise_pure"])
@pytest.mark.parametrize("energy_mult_natoms", [False, True])
@pytest.mark.parametrize("use_cutoff,multiply_cutoff", [
    (False, False), (False, True), (True, False), (True, True),
])
def test_penalty_uses_the_actual_cutoff_branch(
    penalty_calculators, name, energy_mult_natoms, use_cutoff, multiply_cutoff
):
    import numpy as np
    from ase import Atoms

    original_cls, adapted_cls = penalty_calculators
    original = original_cls(name, energy_mult_natoms, True, True)
    adapted = adapted_cls(name, energy_mult_natoms, True, True)
    for calculator in [original, adapted]:
        calculator.model.config.use_cutoff_function = use_cutoff
        calculator.model.config.multiply_cutoff = multiply_cutoff
    atoms = Atoms("Si3", pbc=True)
    atoms.calc = original
    raw_energy = atoms.get_potential_energy()
    forces, stress = atoms.get_forces(), atoms.get_stress()
    atoms.calc = adapted
    # Six directed edges: the test transform produces lengths .4,.4,.3,.3,1,.5.
    penalty = .31 if name == "alignn_atomwise" and use_cutoff and not multiply_cutoff else .12
    missing = energy_mult_natoms or name == "alignn_atomwise_pure"
    correction = penalty * (1 if energy_mult_natoms else 3) if missing else 0
    assert atoms.get_potential_energy() == pytest.approx(raw_energy + correction)
    np.testing.assert_array_equal(atoms.get_forces(), forces)
    np.testing.assert_array_equal(atoms.get_stress(), stress)
    assert atoms.get_potential_energy() == pytest.approx(raw_energy + correction)
    assert adapted.calls == 1
    assert not adapted.model._forward_hooks


def test_penalty_hook_is_removed_after_failed_inference(penalty_calculators, monkeypatch):
    from ase import Atoms

    _, adapted_cls = penalty_calculators
    calculator = adapted_cls("alignn_atomwise_pure", True, True, True)
    def fail(inputs):
        raise RuntimeError("inference failed")
    monkeypatch.setattr(calculator.model, "forward", fail)
    with pytest.raises(RuntimeError, match="inference failed"):
        calculator.calculate(Atoms("Si3", pbc=True))
    assert not calculator.model._forward_hooks
