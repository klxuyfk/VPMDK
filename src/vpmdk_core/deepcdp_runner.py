"""Standalone DeepCDP inference runner used by the main process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepCDP inference on one structure/grid.")
    parser.add_argument("--input", required=True, help="Path to input npz file.")
    parser.add_argument("--output", required=True, help="Path to output npz file.")
    parser.add_argument("--model-path", required=True, help="Path to a DeepCDP .pt checkpoint.")
    parser.add_argument(
        "--metadata-path",
        help="Optional JSON file with SOAP and activation metadata.",
    )
    parser.add_argument("--device", help="Torch device override.")
    parser.add_argument("--probe-count", type=int, default=2500, help="Probe batch size.")
    parser.add_argument("--species", help="Comma-separated SOAP species list.")
    parser.add_argument("--soap-rcut", type=float, help="SOAP cutoff radius.")
    parser.add_argument("--soap-nmax", type=int, help="SOAP radial basis count.")
    parser.add_argument("--soap-lmax", type=int, help="SOAP angular basis count.")
    parser.add_argument("--soap-sigma", type=float, help="SOAP Gaussian width.")
    parser.add_argument(
        "--soap-periodic",
        choices=("0", "1"),
        help="Whether to build periodic SOAP descriptors.",
    )
    parser.add_argument("--activation", help="Hidden activation: relu, tanh, silu, gelu.")
    parser.add_argument("--weighting-function", help="DScribe SOAP weighting function.")
    parser.add_argument("--weighting-r0", type=float, help="DScribe SOAP weighting r0.")
    parser.add_argument("--weighting-c", type=float, help="DScribe SOAP weighting c.")
    parser.add_argument("--weighting-m", type=float, help="DScribe SOAP weighting m.")
    parser.add_argument("--weighting-d", type=float, help="DScribe SOAP weighting d.")
    return parser.parse_args()


def _load_input(path: str) -> tuple[Atoms, tuple[int, int, int]]:
    with np.load(path) as payload:
        atoms = Atoms(
            numbers=np.asarray(payload["numbers"], dtype=np.int64),
            positions=np.asarray(payload["positions"], dtype=float),
            cell=np.asarray(payload["cell"], dtype=float),
            pbc=np.asarray(payload["pbc"], dtype=bool),
        )
        grid_shape = tuple(int(value) for value in np.asarray(payload["grid_shape"], dtype=np.int64))
    if len(grid_shape) != 3 or any(value <= 0 for value in grid_shape):
        raise ValueError(f"Invalid grid_shape in input: {grid_shape!r}")
    return atoms, grid_shape


def _resolve_device_argument(device_arg: str | None, torch_module) -> str:
    if device_arg:
        return str(device_arg)
    if getattr(torch_module, "cuda", None) and torch_module.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_metadata(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with Path(path).expanduser().open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"DeepCDP metadata must be a JSON object: {path}")
    return payload


def _override(value: Any, fallback: Any) -> Any:
    return fallback if value is None else value


def _resolve_weighting(args: argparse.Namespace, metadata: dict[str, Any]) -> dict[str, Any] | None:
    weighting = {}
    metadata_weighting = metadata.get("weighting", {})
    if metadata_weighting and not isinstance(metadata_weighting, dict):
        raise ValueError("DeepCDP metadata field 'weighting' must be a JSON object.")
    for key in ("function", "r0", "c", "m", "d"):
        cli_value = getattr(args, f"weighting_{key}", None)
        value = _override(cli_value, metadata_weighting.get(key) if metadata_weighting else None)
        if value is not None:
            weighting[key] = value
    return weighting or None


def _resolve_species(args: argparse.Namespace, metadata: dict[str, Any], atoms) -> list[str]:
    raw = _override(args.species, metadata.get("species"))
    if raw is None:
        unique_species: list[str] = []
        for symbol in atoms.get_chemical_symbols():
            if symbol not in unique_species:
                unique_species.append(symbol)
        return unique_species
    if isinstance(raw, str):
        result = [token.strip() for token in raw.split(",") if token.strip()]
    else:
        result = [str(token).strip() for token in raw if str(token).strip()]
    if not result:
        raise ValueError("DeepCDP SOAP species list must not be empty.")
    return result


def _resolve_activation(args: argparse.Namespace, metadata: dict[str, Any], model_path: Path) -> str:
    raw = _override(args.activation, metadata.get("activation"))
    if raw is None:
        raw = "tanh" if "tanh" in model_path.name.lower() else "relu"
    normalized = str(raw).strip().lower().replace("-", "_")
    aliases = {
        "relu": "relu",
        "tanh": "tanh",
        "silu": "silu",
        "swish": "silu",
        "gelu": "gelu",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported DeepCDP activation: {raw!r}")
    return aliases[normalized]


def _resolve_config(args: argparse.Namespace, metadata: dict[str, Any], atoms) -> dict[str, Any]:
    soap_periodic = _override(args.soap_periodic, metadata.get("periodic"))
    if soap_periodic is None:
        periodic = bool(np.any(atoms.get_pbc()))
    elif isinstance(soap_periodic, str):
        periodic = soap_periodic == "1"
    else:
        periodic = bool(soap_periodic)

    config = {
        "species": _resolve_species(args, metadata, atoms),
        "rcut": _override(args.soap_rcut, metadata.get("rcut")),
        "nmax": _override(args.soap_nmax, metadata.get("nmax")),
        "lmax": _override(args.soap_lmax, metadata.get("lmax")),
        "sigma": _override(args.soap_sigma, metadata.get("sigma", 0.5)),
        "periodic": periodic,
        "activation": _resolve_activation(args, metadata, Path(args.model_path)),
        "weighting": _resolve_weighting(args, metadata),
    }
    missing = [key for key in ("rcut", "nmax", "lmax") if config[key] is None]
    if missing:
        raise ValueError(
            "DeepCDP requires SOAP configuration for "
            + ", ".join(missing)
            + ". Provide CHARGE_DEEPCDP_* tags or metadata JSON."
        )
    return config


def _load_modules():
    import torch  # type: ignore
    from dscribe.descriptors import SOAP  # type: ignore

    return torch, SOAP


def _activation_module(name: str, torch_module):
    mapping = {
        "relu": torch_module.nn.ReLU,
        "tanh": torch_module.nn.Tanh,
        "silu": torch_module.nn.SiLU,
        "gelu": torch_module.nn.GELU,
    }
    return mapping[name]


class _DeepCDPNetwork:
    def __init__(self, layer_sizes: list[int], activation: str, torch_module):
        modules: list[Any] = []
        activation_cls = _activation_module(activation, torch_module)
        for index, (in_features, out_features) in enumerate(
            zip(layer_sizes[:-1], layer_sizes[1:], strict=False)
        ):
            modules.append(torch_module.nn.Linear(in_features, out_features))
            if index < len(layer_sizes) - 2:
                modules.append(activation_cls())
        self.layers = torch_module.nn.Sequential(*modules)

    def __call__(self, inputs):
        return self.layers(inputs)


def _extract_state_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict"):
            candidate = payload.get(key)
            if isinstance(candidate, dict):
                return candidate
        if all(isinstance(key, str) for key in payload):
            return payload
    raise ValueError("Unsupported DeepCDP checkpoint format.")


def _normalize_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    if any(key.startswith("layers.") for key in state_dict):
        return {
            key.removeprefix("layers."): value
            for key, value in state_dict.items()
        }
    return state_dict


def _build_network(state_dict: dict[str, Any], activation: str, torch_module):
    state_dict = _normalize_state_dict_keys(state_dict)
    weight_keys = sorted(
        (
            key
            for key in state_dict
            if key.endswith(".weight")
        ),
        key=lambda key: int(key.split(".")[0]),
    )
    if not weight_keys:
        raise ValueError("DeepCDP checkpoint does not contain sequential layer weights.")
    layer_sizes = [int(state_dict[weight_keys[0]].shape[1])]
    layer_sizes.extend(int(state_dict[key].shape[0]) for key in weight_keys)
    model = _DeepCDPNetwork(layer_sizes, activation, torch_module)
    model.layers.load_state_dict(state_dict)
    return model.layers


def _grid_positions(cell: np.ndarray, grid_shape: tuple[int, int, int]) -> np.ndarray:
    fractional = np.mgrid[
        0 : grid_shape[0],
        0 : grid_shape[1],
        0 : grid_shape[2],
    ].reshape((3, -1)).T
    fractional = fractional / np.asarray(grid_shape, dtype=float)
    return fractional @ cell


def _predict_density(
    atoms,
    *,
    grid_shape: tuple[int, int, int],
    checkpoint_path: Path,
    config: dict[str, Any],
    probe_count: int,
    device_arg: str | None,
) -> np.ndarray:
    if probe_count <= 0:
        raise ValueError(f"Invalid --probe-count value: {probe_count!r}")

    torch_module, soap_cls = _load_modules()
    device = _resolve_device_argument(device_arg, torch_module)
    checkpoint = torch_module.load(checkpoint_path, map_location=device)
    state_dict = _extract_state_dict(checkpoint)
    model = _build_network(state_dict, config["activation"], torch_module)
    model.to(device)
    model.eval()

    soap_kwargs = {
        "species": config["species"],
        "r_cut": float(config["rcut"]),
        "n_max": int(config["nmax"]),
        "l_max": int(config["lmax"]),
        "sigma": float(config["sigma"]),
        "periodic": bool(config["periodic"]),
        "sparse": False,
    }
    if config["weighting"] is not None:
        soap_kwargs["weighting"] = dict(config["weighting"])
    try:
        soap = soap_cls(**soap_kwargs)
    except TypeError:
        soap_kwargs["rcut"] = soap_kwargs.pop("r_cut")
        soap_kwargs["nmax"] = soap_kwargs.pop("n_max")
        soap_kwargs["lmax"] = soap_kwargs.pop("l_max")
        soap = soap_cls(**soap_kwargs)

    flat_grid = _grid_positions(np.asarray(atoms.get_cell(), dtype=float), grid_shape)
    predicted: list[np.ndarray] = []
    with torch_module.no_grad():
        for start in range(0, flat_grid.shape[0], probe_count):
            stop = min(start + probe_count, flat_grid.shape[0])
            current_positions = flat_grid[start:stop]
            try:
                descriptor_values = soap.create(atoms, positions=current_positions)
            except TypeError:
                descriptor_values = soap.create(atoms, centers=current_positions)
            features = np.asarray(descriptor_values, dtype=np.float32)
            feature_tensor = torch_module.as_tensor(features, device=device)
            values = model(feature_tensor).detach().cpu().numpy().reshape(-1)
            predicted.append(np.asarray(values, dtype=np.float32))
    return np.concatenate(predicted, axis=0).reshape(grid_shape)


def main() -> int:
    args = _parse_args()
    atoms, grid_shape = _load_input(args.input)
    metadata = _load_metadata(args.metadata_path)
    config = _resolve_config(args, metadata, atoms)
    density = _predict_density(
        atoms,
        grid_shape=grid_shape,
        checkpoint_path=Path(args.model_path).expanduser(),
        config=config,
        probe_count=int(args.probe_count),
        device_arg=args.device,
    )
    np.savez(args.output, density=density)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
