"""Standalone DeepDFT inference runner used by the main process."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepDFT inference on one structure/grid.")
    parser.add_argument("--input", required=True, help="Path to input npz file.")
    parser.add_argument("--output", required=True, help="Path to output npz file.")
    parser.add_argument(
        "--model-dir",
        required=True,
        help="DeepDFT model directory containing arguments.json and best_model.pth.",
    )
    parser.add_argument(
        "--source-dir",
        help="Optional DeepDFT checkout root used to import dataset/densitymodel modules.",
    )
    parser.add_argument("--device", help="Torch device override.")
    parser.add_argument(
        "--probe-count",
        type=int,
        default=2500,
        help="Maximum number of probe points per inference batch.",
    )
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
    if torch_module.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_deepdft_modules(source_dir: str | None):
    if source_dir:
        source_root = str(Path(source_dir).expanduser().resolve())
        if source_root not in sys.path:
            sys.path.insert(0, source_root)
    try:
        import densitymodel  # type: ignore
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Unable to import DeepDFT modules. Install DeepDFT in CHARGE_PYTHON "
            "or set CHARGE_SOURCE_DIR to a DeepDFT checkout path with densitymodel.py."
        ) from exc
    return densitymodel, torch


def _load_model(model_dir: Path, device: str, densitymodel_module, torch_module):
    arguments_path = model_dir / "arguments.json"
    checkpoint_path = model_dir / "best_model.pth"
    if not arguments_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(
            "DeepDFT model directory must contain arguments.json and best_model.pth: "
            f"{model_dir}"
        )

    with arguments_path.open() as handle:
        runner_args = argparse.Namespace(**json.load(handle))

    if getattr(runner_args, "use_painn_model", False):
        model = densitymodel_module.PainnDensityModel(
            runner_args.num_interactions,
            runner_args.node_size,
            runner_args.cutoff,
        )
    else:
        model = densitymodel_module.DensityModel(
            runner_args.num_interactions,
            runner_args.node_size,
            runner_args.cutoff,
        )

    torch_device = torch_module.device(device)
    model.to(torch_device)
    state_dict = torch_module.load(checkpoint_path, map_location=torch_device)
    parameters = state_dict["model"] if isinstance(state_dict, dict) and "model" in state_dict else state_dict
    model.load_state_dict(parameters)
    model.eval()
    return model, float(runner_args.cutoff)


def _pad_and_stack_tensors(tensors, torch_module):
    if tensors[0].shape:
        return torch_module.nn.utils.rnn.pad_sequence(
            tensors,
            batch_first=True,
            padding_value=0,
        )
    return torch_module.stack(tensors)


def _collate_graph_dicts(graphs, torch_module):
    keys = graphs[0].keys()
    collated = {}
    for key in keys:
        collated[key] = _pad_and_stack_tensors([graph[key] for graph in graphs], torch_module)
    return collated


def _atoms_to_graph(atoms, cutoff: float):
    src, dst, shifts = neighbor_list("ijS", atoms, cutoff)
    if len(src):
        edges = np.stack((dst, src), axis=1).astype(np.int64, copy=False)
        displacements = np.asarray(shifts, dtype=np.float32)
    else:
        edges = np.zeros((0, 2), dtype=np.int64)
        displacements = np.zeros((0, 3), dtype=np.float32)
    return edges, displacements


def _build_atom_graph(atoms, cutoff: float, torch_module):
    atom_edges, atom_edges_displacement = _atoms_to_graph(atoms, cutoff)
    graph = {
        "nodes": torch_module.tensor(np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)),
        "atom_edges": torch_module.tensor(atom_edges, dtype=torch_module.int64),
        "atom_edges_displacement": torch_module.tensor(
            atom_edges_displacement,
            dtype=torch_module.float32,
        ),
        "num_nodes": torch_module.tensor(len(atoms), dtype=torch_module.int64),
        "num_atom_edges": torch_module.tensor(atom_edges.shape[0], dtype=torch_module.int64),
        "atom_xyz": torch_module.tensor(np.asarray(atoms.get_positions(), dtype=float), dtype=torch_module.float32),
        "cell": torch_module.tensor(np.asarray(atoms.get_cell(), dtype=float), dtype=torch_module.float32),
    }
    return graph


def _grid_points(cell: np.ndarray, grid_shape: tuple[int, int, int]) -> np.ndarray:
    fractional = np.mgrid[
        0 : grid_shape[0],
        0 : grid_shape[1],
        0 : grid_shape[2],
    ].reshape((3, -1)).T
    fractional = fractional / np.asarray(grid_shape, dtype=float)
    return fractional @ cell


def _probes_to_graph(atoms, probe_positions: np.ndarray, cutoff: float):
    probe_atoms = Atoms(
        numbers=np.zeros(len(probe_positions), dtype=np.int64),
        positions=np.asarray(probe_positions, dtype=float),
        cell=atoms.get_cell(),
        pbc=atoms.get_pbc(),
    )
    atoms_with_probes = atoms.copy()
    atoms_with_probes.extend(probe_atoms)
    src, dst, shifts = neighbor_list("ijS", atoms_with_probes, cutoff)
    atom_count = len(atoms)
    mask = (src >= atom_count) & (dst < atom_count)
    if np.any(mask):
        edges = np.stack((dst[mask], src[mask] - atom_count), axis=1).astype(np.int64, copy=False)
        displacements = np.asarray(shifts[mask], dtype=np.float32)
    else:
        edges = np.zeros((0, 2), dtype=np.int64)
        displacements = np.zeros((0, 3), dtype=np.float32)
    return edges, displacements


def _probe_batch(probe_positions: np.ndarray, atoms, cutoff: float, torch_module):
    edges, displacements = _probes_to_graph(atoms, probe_positions, cutoff)
    return {
        "probe_edges": torch_module.tensor(edges, dtype=torch_module.int64),
        "probe_edges_displacement": torch_module.tensor(displacements, dtype=torch_module.float32),
        "num_probe_edges": torch_module.tensor(edges.shape[0], dtype=torch_module.int64),
        "num_probes": torch_module.tensor(len(probe_positions), dtype=torch_module.int64),
        "probe_xyz": torch_module.tensor(probe_positions, dtype=torch_module.float32),
    }


def _predict_density(
    atoms,
    *,
    grid_shape: tuple[int, int, int],
    model_dir: Path,
    source_dir: str | None,
    device_arg: str | None,
    probe_count: int,
) -> np.ndarray:
    if probe_count <= 0:
        raise ValueError(f"Invalid --probe-count value: {probe_count!r}")

    densitymodel_module, torch_module = _load_deepdft_modules(source_dir)
    device = _resolve_device_argument(device_arg, torch_module)
    model, cutoff = _load_model(model_dir, device, densitymodel_module, torch_module)
    atom_graph = _build_atom_graph(atoms, cutoff, torch_module)
    pin_memory = device.startswith("cuda")
    atom_batch = _collate_graph_dicts([atom_graph], torch_module)
    device_batch = {
        key: value.to(device=device, non_blocking=pin_memory)
        for key, value in atom_batch.items()
    }

    with torch_module.no_grad():
        if isinstance(model, densitymodel_module.PainnDensityModel):
            atom_representation_scalar, atom_representation_vector = model.atom_model(device_batch)
        else:
            atom_representation = model.atom_model(device_batch)

        flat_grid = _grid_points(np.asarray(atoms.get_cell(), dtype=float), grid_shape)
        predicted: list[np.ndarray] = []
        for start in range(0, flat_grid.shape[0], probe_count):
            stop = min(start + probe_count, flat_grid.shape[0])
            probe_graph = _probe_batch(
                flat_grid[start:stop],
                atoms,
                cutoff,
                torch_module,
            )
            probe_batch = _collate_graph_dicts([probe_graph], torch_module)
            probe_batch = {
                key: value.to(device=device, non_blocking=pin_memory)
                for key, value in probe_batch.items()
            }
            current_batch = dict(device_batch)
            current_batch.update(probe_batch)
            if isinstance(model, densitymodel_module.PainnDensityModel):
                values = model.probe_model(
                    current_batch,
                    atom_representation_scalar,
                    atom_representation_vector,
                )
            else:
                values = model.probe_model(current_batch, atom_representation)
            predicted.append(np.asarray(values.detach().cpu(), dtype=np.float32).reshape(-1))

    density = np.concatenate(predicted, axis=0)
    return density.reshape(grid_shape)


def main() -> int:
    args = _parse_args()
    atoms, grid_shape = _load_input(args.input)
    density = _predict_density(
        atoms,
        grid_shape=grid_shape,
        model_dir=Path(args.model_dir).expanduser(),
        source_dir=args.source_dir,
        device_arg=args.device,
        probe_count=int(args.probe_count),
    )
    np.savez(args.output, density=density)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
