"""Standalone ChargE3Net inference runner used by the main process."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ChargE3Net inference on one structure/grid.")
    parser.add_argument("--input", required=True, help="Input NPZ path.")
    parser.add_argument("--output", required=True, help="Output NPY path.")
    parser.add_argument("--source-dir", required=True, help="charge3net source checkout.")
    parser.add_argument("--model-path", required=True, help="ChargE3Net checkpoint path.")
    parser.add_argument("--device", default=None, help="Torch device string.")
    parser.add_argument("--cutoff", type=float, default=4.0, help="Probe/atom cutoff in Angstrom.")
    parser.add_argument(
        "--max-probes-per-batch",
        type=int,
        default=2500,
        help="Maximum number of probe points to process per slice.",
    )
    return parser.parse_args()


def _move_batch_to_device(batch: dict[str, object], device):
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


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


def main() -> int:
    args = _parse_args()

    source_dir = Path(args.source_dir).resolve()
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

    import torch
    from torch.serialization import add_safe_globals
    from ase import Atoms

    add_safe_globals([slice])

    from src.charge3net.models.e3 import E3DensityModel
    from src.charge3net.data.graph_construction import KdTreeGraphConstructor
    from src.charge3net.data.collate import collate_list_of_dicts

    payload = np.load(args.input)
    numbers = np.asarray(payload["numbers"], dtype=np.int64)
    positions = np.asarray(payload["positions"], dtype=float)
    cell = np.asarray(payload["cell"], dtype=float)
    pbc = np.asarray(payload["pbc"], dtype=bool)
    grid_shape = tuple(int(value) for value in np.asarray(payload["grid_shape"], dtype=np.int64))

    atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=pbc)
    device = torch.device(_resolve_device_argument(args.device, torch))

    model = E3DensityModel(
        num_interactions=3,
        num_neighbors=20,
        mul=500,
        lmax=4,
        cutoff=float(args.cutoff),
        basis="gaussian",
        num_basis=20,
    )
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint["state_dict"]
    if "model" not in checkpoint:
        state_dict = {key.replace("network.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    constructor = KdTreeGraphConstructor(cutoff=float(args.cutoff), num_probes=None)
    atom_edges, atom_edges_displacement, _, _ = constructor.atoms_to_graph(atoms)
    default_type = torch.get_default_dtype()
    atom_graph = {
        "nodes": torch.tensor(atoms.get_atomic_numbers()),
        "atom_edges": torch.tensor(np.concatenate(atom_edges, axis=0)),
        "atom_edges_displacement": torch.tensor(
            np.concatenate(atom_edges_displacement, axis=0),
            dtype=default_type,
        ),
        "num_nodes": torch.tensor(len(atoms)),
        "num_atom_edges": torch.tensor(sum(len(edges) for edges in atom_edges)),
        "atom_xyz": torch.tensor(atoms.get_positions(), dtype=default_type),
        "cell": torch.tensor(np.array(atoms.get_cell()), dtype=default_type),
    }
    atom_batch = collate_list_of_dicts([atom_graph], pin_memory=False)
    atom_batch = _move_batch_to_device(atom_batch, device)

    with torch.no_grad():
        atom_representation = model.atom_model(atom_batch)
        total_probes = int(np.prod(grid_shape))
        density_parts: list[np.ndarray] = []
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
            density_parts.append(predictions[0].detach().cpu().numpy())

    density = np.concatenate(density_parts, axis=0).reshape(grid_shape)
    np.save(args.output, density.astype(np.float32))
    return 0


if __name__ == "__main__":  # pragma: no cover - subprocess entrypoint
    raise SystemExit(main())
