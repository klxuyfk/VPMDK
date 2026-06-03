"""FAIRChem backend builders."""

from __future__ import annotations

import sys
from typing import Dict, Iterable

import numpy as np
from ase.calculators.calculator import Calculator, all_changes


def _root():
    return sys.modules["vpmdk_core"]


_EQUIFORMER_V3_MODEL_NAME = "equiformer_v3"
_EQUIFORMER_V3_IMPORT_PATHS = (
    "fairchem.experimental.models.equiformer_v3.equiformer_v3",
    "experimental.models.equiformer_v3.equiformer_v3",
)


def _split_equiformer_v3_modules(value: str | None) -> list[str]:
    if not value:
        return []
    return [module.strip() for module in value.replace(";", ",").split(",") if module.strip()]


def _equiformer_v3_import_candidates(bcar_tags: Dict[str, str]) -> list[str]:
    explicit: list[str] = []
    for tag in ("EQUIFORMER_V3_MODULE", "EQUIFORMER_V3_IMPORT_MODULE"):
        explicit.extend(_split_equiformer_v3_modules(bcar_tags.get(tag)))

    candidates: list[str] = []
    seen: set[str] = set()
    for module_name in [*explicit, *_EQUIFORMER_V3_IMPORT_PATHS]:
        if module_name in seen:
            continue
        seen.add(module_name)
        candidates.append(module_name)
    return candidates


def _fairchem_model_registered(model_name: str) -> bool:
    root = _root()
    try:
        module = root.importlib.import_module("fairchem.core.common.registry")
        registry = getattr(module, "registry", None)
        get_model_class = getattr(registry, "get_model_class", None)
        if not callable(get_model_class):
            return False
        get_model_class(model_name)
        return True
    except Exception:
        return False


def _import_equiformer_v3_model(bcar_tags: Dict[str, str]):
    """Import EquiformerV3 so the FAIRChem v1 registry can construct it."""

    root = _root()
    if root._fairchem_model_registered(_EQUIFORMER_V3_MODEL_NAME):
        return None

    errors: list[str] = []
    last_error: Exception | None = None
    for module_name in root._equiformer_v3_import_candidates(bcar_tags):
        try:
            module = root.importlib.import_module(module_name)
        except Exception as exc:
            last_error = exc
            errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
            continue
        if root._fairchem_model_registered(_EQUIFORMER_V3_MODEL_NAME):
            return module
        errors.append(f"{module_name}: imported but did not register {_EQUIFORMER_V3_MODEL_NAME!r}")

    details = "; ".join(errors) if errors else "no import candidates configured"
    raise RuntimeError(
        "EquiformerV3 model registration is not available. Install the official "
        "atomicarchitects/equiformer_v3 code, put its src directory on PYTHONPATH, "
        "or set EQUIFORMER_V3_MODULE to a module that registers 'equiformer_v3'. "
        f"Import attempts: {details}"
    ) from last_error


def _is_equiformer_v3_available() -> bool:
    root = _root()
    if root._get_fairchem_v1_calculator_cls() is None:
        return False
    if root._fairchem_model_registered(_EQUIFORMER_V3_MODEL_NAME):
        return True
    for module_name in _EQUIFORMER_V3_IMPORT_PATHS:
        try:
            if root.importlib.util.find_spec(module_name) is not None:
                return True
        except Exception:
            continue
    return False


def _get_fairchem_v1_calculator_cls():
    """Return FAIRChem v1 calculator class if installed."""

    root = _root()
    if root.FAIRChemV1Calculator is not None:
        return root.FAIRChemV1Calculator

    for module_name in root._FAIRCHEM_V1_IMPORT_PATHS:
        try:
            spec = root.importlib.util.find_spec(module_name)
        except Exception:
            continue
        if spec is None:
            continue
        try:
            module = root.importlib.import_module(module_name)
        except Exception:
            continue
        candidate = getattr(module, "OCPCalculator", None)
        if candidate is not None:
            root.FAIRChemV1Calculator = candidate
            return candidate
    return None


def _get_fairchem_v1_predictor_cls():
    """Return FAIRChem v1 predictor class if installed."""

    root = _root()
    if root.FAIRChemV1Predictor is not None:
        return root.FAIRChemV1Predictor

    for module_name in root._FAIRCHEM_V1_PREDICTOR_IMPORT_PATHS + root._FAIRCHEM_V1_IMPORT_PATHS:
        try:
            spec = root.importlib.util.find_spec(module_name)
        except Exception:
            continue
        if spec is None:
            continue
        try:
            module = root.importlib.import_module(module_name)
        except Exception:
            continue
        for class_name in root._FAIRCHEM_V1_PREDICTOR_CLASS_NAMES:
            candidate = getattr(module, class_name, None)
            if candidate is not None:
                root.FAIRChemV1Predictor = candidate
                return candidate
    return None


def _build_fairchem_calculator(bcar_tags: Dict[str, str]):
    """Create the FAIRChem ASE calculator configured from BCAR tags."""

    root = _root()
    if root.FAIRChemCalculator is None:
        raise RuntimeError("FAIRChemCalculator not available. Install fairchem and dependencies.")

    model_name = bcar_tags.get("MODEL") or root.DEFAULT_FAIRCHEM_MODEL
    task_name = bcar_tags.get("FAIRCHEM_TASK")
    if task_name is None and model_name == root.DEFAULT_FAIRCHEM_MODEL:
        task_name = root.DEFAULT_FAIRCHEM_TASK
    inference_settings = bcar_tags.get("FAIRCHEM_INFERENCE_SETTINGS") or "default"
    device = bcar_tags.get("DEVICE")

    return root.FAIRChemCalculator.from_model_checkpoint(
        model_name,
        task_name=task_name,
        inference_settings=inference_settings,
        device=device,
    )


def _build_equiformer_v3_calculator(bcar_tags: Dict[str, str]):
    """Create an EquiformerV3 ASE calculator through the FAIRChem v1 runtime."""

    root = _root()
    model_path = bcar_tags.get("MODEL")
    if not model_path:
        raise ValueError("EquiformerV3 requires MODEL pointing to a checkpoint file.")
    if not root.os.path.exists(model_path):
        raise FileNotFoundError(f"EquiformerV3 model not found: {model_path}")

    root._import_equiformer_v3_model(bcar_tags)
    return root._build_fairchem_v1_calculator(bcar_tags)


def _pick_fairchem_prediction_value(prediction, keys: Iterable[str]):
    if isinstance(prediction, dict):
        for key in keys:
            if key in prediction:
                return prediction[key]
    for key in keys:
        if hasattr(prediction, key):
            return getattr(prediction, key)
    return None


def _as_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _normalize_fairchem_prediction(prediction, atoms):
    if isinstance(prediction, (list, tuple)):
        if not prediction:
            prediction = {}
        else:
            prediction = prediction[0]

    energy_value = _pick_fairchem_prediction_value(
        prediction, ("energy", "energies", "y_energy", "y")
    )
    forces_value = _pick_fairchem_prediction_value(
        prediction, ("forces", "force", "y_force", "y_forces")
    )
    stress_value = _pick_fairchem_prediction_value(
        prediction, ("stress", "stresses", "virial")
    )

    energy = _as_numpy(energy_value)
    if energy is None:
        energy_float = 0.0
    else:
        energy_float = float(np.asarray(energy).reshape(-1)[0])

    forces = _as_numpy(forces_value)
    if forces is None:
        forces_array = np.zeros((len(atoms), 3))
    else:
        forces_array = np.asarray(forces)
        if forces_array.size == len(atoms) * 3:
            forces_array = forces_array.reshape((len(atoms), 3))

    stress = _as_numpy(stress_value)
    if stress is None:
        stress_array = np.zeros(6)
    else:
        stress_array = np.asarray(stress).reshape(-1)
        if stress_array.size == 9:
            stress_matrix = stress_array.reshape(3, 3)
            stress_array = np.array(
                [
                    stress_matrix[0, 0],
                    stress_matrix[1, 1],
                    stress_matrix[2, 2],
                    stress_matrix[1, 2],
                    stress_matrix[0, 2],
                    stress_matrix[0, 1],
                ]
            )
        elif stress_array.size != 6:
            stress_array = np.zeros(6)

    return energy_float, forces_array, stress_array


def _run_fairchem_v1_prediction(predictor, atoms):
    for method_name in ("predict_atoms", "predict", "__call__"):
        method = getattr(predictor, method_name, None)
        if not callable(method):
            continue
        try:
            return method(atoms)
        except TypeError:
            try:
                return method([atoms])
            except Exception:
                continue
    raise RuntimeError("FAIRChem v1 predictor does not expose a usable prediction method.")


class _FairChemV1PredictorCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, predictor):
        super().__init__()
        self._predictor = predictor

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if atoms is None:
            self.results = {"energy": 0.0, "forces": [], "stress": [0.0] * 6}
            return
        prediction = _run_fairchem_v1_prediction(self._predictor, atoms)
        energy, forces, stress = _normalize_fairchem_prediction(prediction, atoms)
        self.results = {"energy": energy, "forces": forces, "stress": stress}


def _build_fairchem_v1_predictor(bcar_tags: Dict[str, str]):
    """Create a FAIRChem v1 predictor-backed ASE calculator."""

    root = _root()
    predictor_cls = root._get_fairchem_v1_predictor_cls()
    if predictor_cls is None:
        raise RuntimeError(
            "FAIRChem v1 predictor not available. Install fairchem v1 (OCP) dependencies."
        )

    model_path = bcar_tags.get("MODEL")
    if not model_path:
        raise ValueError("FAIRChem v1 requires MODEL pointing to a checkpoint file.")

    config_path = bcar_tags.get("FAIRCHEM_CONFIG")
    device = bcar_tags.get("DEVICE")
    cpu_flag = device is not None and device.lower() == "cpu"

    kwargs: Dict[str, object] = {"checkpoint_path": model_path, "cpu": cpu_flag}
    if config_path:
        kwargs["config_yml"] = config_path
    if device and not cpu_flag:
        kwargs["device"] = device

    predictor = predictor_cls(**kwargs)
    return _FairChemV1PredictorCalculator(predictor)


def _build_fairchem_v1_calculator(bcar_tags: Dict[str, str]):
    """Create the FAIRChem v1 OCPCalculator configured from BCAR tags."""

    root = _root()
    predictor_tag = bcar_tags.get("FAIRCHEM_V1_PREDICTOR")
    if predictor_tag is not None and root._coerce_bool_tag(
        predictor_tag, "FAIRCHEM_V1_PREDICTOR"
    ):
        return root._build_fairchem_v1_predictor(bcar_tags)

    calculator_cls = root._get_fairchem_v1_calculator_cls()
    if calculator_cls is None:
        raise RuntimeError(
            "FAIRChem v1 calculator not available. Install fairchem v1 (OCP) dependencies."
        )

    model_path = bcar_tags.get("MODEL")
    if not model_path:
        raise ValueError("FAIRChem v1 requires MODEL pointing to a checkpoint file.")

    config_path = bcar_tags.get("FAIRCHEM_CONFIG")
    device = bcar_tags.get("DEVICE")
    cpu_flag = device is not None and device.lower() == "cpu"

    kwargs: Dict[str, object] = {"checkpoint_path": model_path, "cpu": cpu_flag}
    if config_path:
        kwargs["config_yml"] = config_path

    calculator = calculator_cls(**kwargs)
    return root._attach_fallback_calculator(calculator, bcar_tags)
