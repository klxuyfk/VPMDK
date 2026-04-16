"""Standalone ChargE3Net inference runner used by the main process."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
import warnings
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import numpy as np


_MODEL_CONFIG_FIELDS = (
    "num_interactions",
    "num_neighbors",
    "mul",
    "lmax",
    "cutoff",
    "basis",
    "num_basis",
    "spin",
)
_DEFAULT_MODEL_CONFIG = {
    "num_interactions": 3,
    "num_neighbors": 20.0,
    "mul": 500,
    "lmax": 4,
    "cutoff": 4.0,
    "basis": "gaussian",
    "num_basis": 20,
    "spin": False,
}
_MODEL_CONFIG_ALIASES = {
    "num_interactions": ("num_interactions",),
    "num_neighbors": ("num_neighbors",),
    "mul": ("mul",),
    "lmax": ("lmax",),
    "cutoff": ("cutoff",),
    "basis": ("basis",),
    "num_basis": ("num_basis", "number_of_basis"),
    "spin": ("spin",),
}
_MODEL_SHAPE_KEYS = (
    "atom_model.convolutions.0.lin1.weight",
    "atom_model.convolutions.0.lin2.weight",
    "atom_model.convolutions.0.lin3.weight",
    "probe_model.readout.weight",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ChargE3Net inference on one structure/grid.")
    parser.add_argument("--input", required=True, help="Input NPZ path.")
    parser.add_argument("--output", required=True, help="Output NPY path.")
    parser.add_argument("--source-dir", default=None, help="charge3net source checkout.")
    parser.add_argument("--model-path", required=True, help="ChargE3Net checkpoint path.")
    parser.add_argument("--device", default=None, help="Torch device string.")
    parser.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help="Probe/atom cutoff in Angstrom. Overrides checkpoint/default cutoff when set.",
    )
    parser.add_argument(
        "--max-probes-per-batch",
        type=int,
        default=2500,
        help="Maximum number of probe points to process per slice.",
    )
    parser.add_argument("--num-interactions", type=int, default=None, help="Model interaction layers.")
    parser.add_argument("--num-neighbors", type=float, default=None, help="Model neighbor normalization.")
    parser.add_argument("--mul", type=int, default=None, help="Model multiplicity parameter.")
    parser.add_argument("--lmax", type=int, default=None, help="Maximum angular momentum.")
    parser.add_argument("--basis", default=None, help="Radial basis family.")
    parser.add_argument("--num-basis", type=int, default=None, help="Number of radial basis functions.")
    parser.add_argument("--spin", type=int, choices=(0, 1), default=None, help="Spin-density model flag.")
    return parser.parse_args()


def _move_batch_to_device(batch: dict[str, object], device):
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def _split_probe_output(predictions, *, spin: bool) -> tuple[np.ndarray, np.ndarray | None]:
    prediction_array = predictions[0].detach().cpu().numpy()
    if not spin:
        return np.asarray(prediction_array, dtype=np.float32), None
    if prediction_array.ndim != 2 or prediction_array.shape[1] != 2:
        raise RuntimeError(
            "Spin-enabled ChargE3Net probe head must return shape (n_probe, 2), "
            f"got {prediction_array.shape!r}."
        )
    spin_up = prediction_array[:, 0]
    spin_down = prediction_array[:, 1]
    charge_density = spin_up + spin_down
    spin_density = spin_up - spin_down
    return np.asarray(charge_density, dtype=np.float32), np.asarray(spin_density, dtype=np.float32)


def _grid_positions_for_slice(
    grid_shape: tuple[int, int, int],
    cell: np.ndarray,
    start: int,
    stop: int,
) -> np.ndarray:
    flat_indices = np.arange(start, stop, dtype=np.int64)
    grid_indices = np.stack(np.unravel_index(flat_indices, grid_shape), axis=1)
    divisors = np.asarray(grid_shape, dtype=float)
    fractional = grid_indices / divisors[None, :]
    return fractional @ cell


def _resolve_device_argument(requested_device: str | None, torch_module) -> str:
    if requested_device:
        return requested_device
    return "cuda" if torch_module.cuda.is_available() else "cpu"


def _ensure_torch_safe_globals(torch_module) -> None:
    serialization = getattr(torch_module, "serialization", None)
    add_safe_globals = getattr(serialization, "add_safe_globals", None)
    if callable(add_safe_globals):
        add_safe_globals([slice])


def _ensure_package_module(name: str, path: Path) -> ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = ModuleType(name)
        module.__path__ = [str(path)]  # type: ignore[attr-defined]
        sys.modules[name] = module
        return module

    package_path = getattr(module, "__path__", None)
    if package_path is None:
        module.__path__ = [str(path)]  # type: ignore[attr-defined]
    elif str(path) not in package_path:
        package_path.append(str(path))
    return module


def _load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to build import spec for {module_name} from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_checkout_modules(prefix: str, package_root: Path):
    models_dir = package_root / "models"
    data_dir = package_root / "data"
    e3_path = models_dir / "e3.py"
    graph_path = data_dir / "graph_construction.py"
    collate_path = data_dir / "collate.py"
    if not all(path.exists() for path in (e3_path, graph_path, collate_path)):
        raise FileNotFoundError(f"Missing ChargE3Net modules under {package_root}.")

    package_parts = prefix.split(".")
    current_path = package_root.parents[len(package_parts) - 1]
    current_name_parts: list[str] = []
    for part in package_parts:
        current_path = current_path / part
        current_name_parts.append(part)
        current_name = ".".join(current_name_parts)
        _ensure_package_module(current_name, current_path)
    _ensure_package_module(f"{prefix}.models", models_dir)
    _ensure_package_module(f"{prefix}.data", data_dir)

    e3_module = _load_module_from_path(f"{prefix}.models.e3", e3_path)
    graph_module = _load_module_from_path(f"{prefix}.data.graph_construction", graph_path)
    collate_module = _load_module_from_path(f"{prefix}.data.collate", collate_path)
    return (
        e3_module.E3DensityModel,
        graph_module.KdTreeGraphConstructor,
        collate_module.collate_list_of_dicts,
    )


def _find_installed_package_root(prefix: str) -> Path:
    package_spec = importlib.util.find_spec(prefix)
    if package_spec is None or not package_spec.submodule_search_locations:
        raise ModuleNotFoundError(f"Installed package {prefix!r} not found.")
    return Path(next(iter(package_spec.submodule_search_locations))).resolve()


def _coerce_mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        return value
    return None


def _coerce_model_config_value(field: str, value: object):
    if value is None:
        return None
    if field in {"num_interactions", "mul", "lmax", "num_basis"}:
        return int(value)
    if field in {"num_neighbors", "cutoff"}:
        return float(value)
    if field == "spin":
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    return str(value)


def _extract_model_config_from_mapping(mapping: Mapping[str, object]) -> dict[str, object]:
    config: dict[str, object] = {}
    for field, aliases in _MODEL_CONFIG_ALIASES.items():
        for alias in aliases:
            if alias in mapping and mapping[alias] is not None:
                config[field] = _coerce_model_config_value(field, mapping[alias])
                break
    return config


def _extract_model_config_from_checkpoint(checkpoint: object) -> dict[str, object]:
    root_mapping = _coerce_mapping(checkpoint)
    if root_mapping is None:
        return {}

    candidates: list[Mapping[str, object]] = [root_mapping]
    for key in ("model_kwargs", "model_config", "hyper_parameters", "hparams", "config", "cfg"):
        nested = _coerce_mapping(root_mapping.get(key))
        if nested is None:
            continue
        candidates.append(nested)
        model_nested = _coerce_mapping(nested.get("model"))
        if model_nested is not None:
            candidates.append(model_nested)
            nested_model = _coerce_mapping(model_nested.get("model"))
            if nested_model is not None:
                candidates.append(nested_model)

    merged: dict[str, object] = {}
    for candidate in candidates:
        merged.update(_extract_model_config_from_mapping(candidate))
    return merged


def _normalize_state_dict(checkpoint: Mapping[str, object]) -> Mapping[str, object]:
    if "model" in checkpoint and isinstance(checkpoint["model"], Mapping):
        return checkpoint["model"]
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], Mapping):
        return {
            key.replace("network.", ""): value
            for key, value in checkpoint["state_dict"].items()
        }
    raise RuntimeError("ChargE3Net checkpoint does not contain a supported state_dict.")


def _infer_num_interactions(state_dict: Mapping[str, object]) -> int | None:
    prefixes = {
        int(key.split(".")[2])
        for key in state_dict
        if key.startswith("atom_model.convolutions.")
    }
    if not prefixes:
        return None
    return max(prefixes) + 1


def _infer_num_basis(state_dict: Mapping[str, object]) -> int | None:
    basis_mean = state_dict.get("atom_model.basis.mean")
    if hasattr(basis_mean, "shape") and len(getattr(basis_mean, "shape", ())) == 1:
        return int(basis_mean.shape[0])
    fc_weight = state_dict.get("atom_model.convolutions.0.fc.layer0.weight")
    if hasattr(fc_weight, "shape") and len(getattr(fc_weight, "shape", ())) == 2:
        return int(fc_weight.shape[0])
    return None


def _infer_spin(state_dict: Mapping[str, object]) -> bool | None:
    output_mask = state_dict.get("probe_model.readout.output_mask")
    if hasattr(output_mask, "numel"):
        return int(output_mask.numel()) == 2
    return None


def _extract_state_dict_shape_signature(
    state_dict: Mapping[str, object],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple(
        (key, tuple(state_dict[key].shape))
        for key in _MODEL_SHAPE_KEYS
        if key in state_dict and hasattr(state_dict[key], "shape")
    )


@lru_cache(maxsize=256)
def _candidate_shape_signature(
    model_cls,
    *,
    num_interactions: int,
    num_basis: int,
    spin: bool,
    lmax: int,
    mul: int,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = model_cls(
            num_interactions=num_interactions,
            num_neighbors=20,
            mul=mul,
            lmax=lmax,
            cutoff=4.0,
            basis="gaussian",
            num_basis=num_basis,
            spin=spin,
        )
    state = model.state_dict()
    signature = []
    for key in _MODEL_SHAPE_KEYS:
        if key in state:
            signature.append((key, tuple(state[key].shape)))
    return tuple(signature)


def _infer_mul_lmax_from_state_dict(
    state_dict: Mapping[str, object],
    model_cls,
    *,
    num_interactions: int,
    num_basis: int,
    spin: bool,
) -> dict[str, int]:
    target_signature = _extract_state_dict_shape_signature(state_dict)
    if not target_signature:
        return {}

    target_readout = next(
        (int(np.prod(shape)) for key, shape in target_signature if key == "probe_model.readout.weight"),
        None,
    )
    max_lmax = 8
    max_mul = 4096
    for lmax in range(0, max_lmax + 1):
        candidate_range: range
        if target_readout is None:
            candidate_range = range(1, min(max_mul, 1024) + 1)
        else:
            high = 1
            while high < max_mul:
                current = dict(
                    _candidate_shape_signature(
                        model_cls,
                        num_interactions=num_interactions,
                        num_basis=num_basis,
                        spin=spin,
                        lmax=lmax,
                        mul=high,
                    )
                )
                current_readout = int(np.prod(current.get("probe_model.readout.weight", (0,))))
                if current_readout >= target_readout:
                    break
                high *= 2
            low = max(1, high // 2)
            candidate_range = range(max(1, low - 32), min(max_mul, high + 32) + 1)

        for mul in candidate_range:
            if (
                _candidate_shape_signature(
                    model_cls,
                    num_interactions=num_interactions,
                    num_basis=num_basis,
                    spin=spin,
                    lmax=lmax,
                    mul=mul,
                )
                == target_signature
            ):
                return {"lmax": lmax, "mul": mul}
    return {}


def _infer_model_config_from_state_dict(
    state_dict: Mapping[str, object],
    model_cls,
) -> dict[str, object]:
    config: dict[str, object] = {}
    num_interactions = _infer_num_interactions(state_dict)
    if num_interactions is not None:
        config["num_interactions"] = num_interactions
    num_basis = _infer_num_basis(state_dict)
    if num_basis is not None:
        config["num_basis"] = num_basis
    spin = _infer_spin(state_dict)
    if spin is not None:
        config["spin"] = spin
    if num_interactions is not None and num_basis is not None and spin is not None:
        target_signature = _extract_state_dict_shape_signature(state_dict)
        default_signature = _candidate_shape_signature(
            model_cls,
            num_interactions=num_interactions,
            num_basis=num_basis,
            spin=spin,
            lmax=int(_DEFAULT_MODEL_CONFIG["lmax"]),
            mul=int(_DEFAULT_MODEL_CONFIG["mul"]),
        )
        if target_signature and target_signature == default_signature:
            config["mul"] = int(_DEFAULT_MODEL_CONFIG["mul"])
            config["lmax"] = int(_DEFAULT_MODEL_CONFIG["lmax"])
        else:
            config.update(
                _infer_mul_lmax_from_state_dict(
                    state_dict,
                    model_cls,
                    num_interactions=num_interactions,
                    num_basis=num_basis,
                    spin=spin,
                )
            )
    return config


def _resolve_model_config(
    checkpoint: Mapping[str, object],
    *,
    explicit_config: Mapping[str, object],
    model_cls,
) -> dict[str, object]:
    state_dict = _normalize_state_dict(checkpoint)
    config = _extract_model_config_from_checkpoint(checkpoint)
    inferred = _infer_model_config_from_state_dict(state_dict, model_cls)
    for key, value in inferred.items():
        config.setdefault(key, value)
    for key, value in explicit_config.items():
        if value is not None:
            config[key] = value
    for key, value in _DEFAULT_MODEL_CONFIG.items():
        config.setdefault(key, value)
    return config


def _load_charge3net_modules(source_dir: str | None):
    if source_dir:
        resolved_source_dir = Path(source_dir).resolve()
        last_error: Exception | None = None
        for prefix, package_root in (
            ("src.charge3net", resolved_source_dir / "src" / "charge3net"),
            ("charge3net", resolved_source_dir / "charge3net"),
        ):
            try:
                return _load_checkout_modules(prefix, package_root)
            except Exception as exc:  # pragma: no cover - exercised indirectly in subprocess
                last_error = exc
        details = f": {last_error}" if last_error is not None else ""
        raise RuntimeError(
            "Unable to load ChargE3Net modules from CHARGE_SOURCE_DIR"
            f"{details}"
        )

    last_error: Exception | None = None
    for prefix in ("src.charge3net", "charge3net"):
        try:
            return _load_checkout_modules(prefix, _find_installed_package_root(prefix))
        except Exception as exc:  # pragma: no cover - exercised indirectly in subprocess
            last_error = exc

    details = f": {last_error}" if last_error is not None else ""
    raise RuntimeError(
        "Unable to import ChargE3Net modules. Install ChargE3Net in CHARGE_PYTHON "
        "or set CHARGE_SOURCE_DIR to a checkout path"
        f"{details}"
    )


def _coalesce_edge_groups(
    edge_groups,
    displacement_groups,
    *,
    float_dtype=float,
) -> tuple[np.ndarray, np.ndarray, int]:
    valid_edges = [np.asarray(edges, dtype=np.int64) for edges in edge_groups if len(edges) > 0]
    valid_displacements = [
        np.asarray(displacements, dtype=float)
        for displacements in displacement_groups
        if len(displacements) > 0
    ]
    if not valid_edges:
        return (
            np.zeros((0, 2), dtype=np.int64),
            np.zeros((0, 3), dtype=float_dtype),
            0,
        )
    return (
        np.concatenate(valid_edges, axis=0),
        np.concatenate(valid_displacements, axis=0).astype(float_dtype, copy=False),
        int(sum(len(edges) for edges in valid_edges)),
    )


def main() -> int:
    args = _parse_args()

    import torch
    from ase import Atoms

    _ensure_torch_safe_globals(torch)

    E3DensityModel, KdTreeGraphConstructor, collate_list_of_dicts = _load_charge3net_modules(
        args.source_dir
    )

    payload = np.load(args.input)
    numbers = np.asarray(payload["numbers"], dtype=np.int64)
    positions = np.asarray(payload["positions"], dtype=float)
    cell = np.asarray(payload["cell"], dtype=float)
    pbc = np.asarray(payload["pbc"], dtype=bool)
    grid_shape = tuple(int(value) for value in np.asarray(payload["grid_shape"], dtype=np.int64))

    atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=pbc)
    device = torch.device(_resolve_device_argument(args.device, torch))
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model_config = _resolve_model_config(
        checkpoint,
        explicit_config={
            "num_interactions": args.num_interactions,
            "num_neighbors": args.num_neighbors,
            "mul": args.mul,
            "lmax": args.lmax,
            "cutoff": args.cutoff,
            "basis": args.basis,
            "num_basis": args.num_basis,
            "spin": None if args.spin is None else bool(args.spin),
        },
        model_cls=E3DensityModel,
    )
    resolved_cutoff = float(model_config.get("cutoff", _DEFAULT_MODEL_CONFIG["cutoff"]))

    model = E3DensityModel(
        num_interactions=int(model_config.get("num_interactions", 3)),
        num_neighbors=float(model_config.get("num_neighbors", 20)),
        mul=int(model_config.get("mul", 500)),
        lmax=int(model_config.get("lmax", 4)),
        cutoff=resolved_cutoff,
        basis=str(model_config.get("basis", "gaussian")),
        num_basis=int(model_config.get("num_basis", 20)),
        spin=bool(model_config.get("spin", False)),
    )
    state_dict = _normalize_state_dict(checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    constructor = KdTreeGraphConstructor(cutoff=resolved_cutoff, num_probes=None)
    atom_edges, atom_edges_displacement, _, _ = constructor.atoms_to_graph(atoms)
    default_type = torch.get_default_dtype()
    atom_edge_array, atom_edge_displacement_array, num_atom_edges = _coalesce_edge_groups(
        atom_edges,
        atom_edges_displacement,
        float_dtype=float,
    )
    atom_graph = {
        "nodes": torch.tensor(atoms.get_atomic_numbers()),
        "atom_edges": torch.tensor(atom_edge_array),
        "atom_edges_displacement": torch.tensor(atom_edge_displacement_array, dtype=default_type),
        "num_nodes": torch.tensor(len(atoms)),
        "num_atom_edges": torch.tensor(num_atom_edges),
        "atom_xyz": torch.tensor(atoms.get_positions(), dtype=default_type),
        "cell": torch.tensor(np.array(atoms.get_cell()), dtype=default_type),
    }
    atom_batch = collate_list_of_dicts([atom_graph], pin_memory=False)
    atom_batch = _move_batch_to_device(atom_batch, device)

    with torch.no_grad():
        atom_representation = model.atom_model(atom_batch)
        total_probes = int(np.prod(grid_shape))
        density_parts: list[np.ndarray] = []
        spin_density_parts: list[np.ndarray] = []
        for start in range(0, total_probes, int(args.max_probes_per_batch)):
            stop = min(start + int(args.max_probes_per_batch), total_probes)
            probe_positions = _grid_positions_for_slice(grid_shape, cell, start, stop)
            try:
                probe_edges, probe_edges_displacement = constructor.probes_to_graph(
                    atoms,
                    probe_positions,
                )
            except ValueError:
                probe_edges = np.zeros((0, 2), dtype=np.int64)
                probe_edges_displacement = np.zeros((0, 3), dtype=float)

            probe_batch = {
                **atom_batch,
                "probe_edges": torch.tensor(probe_edges, dtype=torch.long, device=device).unsqueeze(0),
                "probe_edges_displacement": torch.tensor(
                    probe_edges_displacement,
                    dtype=torch.get_default_dtype(),
                    device=device,
                ).unsqueeze(0),
                "num_probe_edges": torch.tensor([len(probe_edges)], dtype=torch.long, device=device),
                "num_probes": torch.tensor([len(probe_positions)], dtype=torch.long, device=device),
                "probe_xyz": torch.tensor(
                    probe_positions,
                    dtype=torch.get_default_dtype(),
                    device=device,
                ).unsqueeze(0),
                "probe_target": torch.zeros(
                    (1, len(probe_positions)),
                    dtype=torch.get_default_dtype(),
                    device=device,
                ),
            }
            predictions = model.probe_model(probe_batch, atom_representation)
            density_chunk, spin_density_chunk = _split_probe_output(
                predictions,
                spin=bool(model_config.get("spin", False)),
            )
            density_parts.append(density_chunk)
            if spin_density_chunk is not None:
                spin_density_parts.append(spin_density_chunk)

    density = np.concatenate(density_parts, axis=0).reshape(grid_shape)
    output_payload = {"density": density.astype(np.float32)}
    if spin_density_parts:
        output_payload["spin_density"] = (
            np.concatenate(spin_density_parts, axis=0).reshape(grid_shape).astype(np.float32)
        )
    np.savez(args.output, **output_payload)
    return 0


if __name__ == "__main__":  # pragma: no cover - subprocess entrypoint
    raise SystemExit(main())
