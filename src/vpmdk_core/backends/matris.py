"""MatRIS backend builder."""

from __future__ import annotations

import os
import sys
from typing import Dict


def _root():
    return sys.modules["vpmdk_core"]


def _load_matris_checkpoint_model(checkpoint_path: str, *, device: str | None):
    """Load a MatRIS model directly from a checkpoint file."""

    root = _root()
    if root.MatRISModel is None:
        raise RuntimeError("MatRIS model loader not available. Install matris and dependencies.")

    import torch

    checkpoint_state = torch.load(
        checkpoint_path,
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    model = root.MatRISModel.from_dict(checkpoint_state)
    return model.to(device or "cpu")


def _ensure_matris_named_model_checkpoint(model_name: str) -> str | None:
    """Download a known MatRIS named model into the standard cache when needed."""

    root = _root()
    download_info = root._MATRIS_NAMED_MODEL_DOWNLOADS.get(model_name.lower())
    if download_info is None:
        return None

    checkpoint_filename, url = download_info
    cache_dir = root.os.path.expanduser("~/.cache/matris")
    root.os.makedirs(cache_dir, exist_ok=True)
    checkpoint_path = root.os.path.join(cache_dir, checkpoint_filename)
    if not root.os.path.exists(checkpoint_path) or root.os.path.getsize(checkpoint_path) == 0:
        print(f"MatRIS checkpoint not found, downloading to {checkpoint_path} ...")
        root._download_file_to_path(url, checkpoint_path)
    return checkpoint_path


def _instantiate_matris_calculator(*, model, task: str, device: str | None):
    """Create a MatRIS ASE calculator from a preloaded model instance."""

    root = _root()
    if root.MatRISCalculator is None:
        raise RuntimeError("MatRIS calculator not available. Install matris and dependencies.")

    calculator = root.MatRISCalculator.__new__(root.MatRISCalculator)
    root.Calculator.__init__(calculator)
    calculator.task = task
    calculator.device = device or "cpu"
    calculator.model = model
    calculator.stress_unit = root.units.GPa
    calculator.key = {"atoms_per_graph", "ref_energy", *task}
    return calculator


def _build_matris_calculator(bcar_tags: Dict[str, str]):
    """Create the MatRIS ASE calculator configured from BCAR tags."""

    root = _root()
    if root.MatRISCalculator is None:
        raise RuntimeError("MatRIS calculator not available. Install matris and dependencies.")

    device = root._resolve_device(bcar_tags.get("DEVICE"))
    task = (bcar_tags.get("MATRIS_TASK") or "efs").lower()
    model_value = bcar_tags.get("MODEL") or root.DEFAULT_MATRIS_MODEL
    graph_converter_algorithm = root._resolve_graph_converter_algorithm(
        bcar_tags,
        backend_tag="MATRIS",
    )

    if os.path.exists(model_value):
        model = root._load_matris_checkpoint_model(model_value, device=device)
        if graph_converter_algorithm is not None:
            model = root._override_model_graph_converter_algorithm(
                model,
                algorithm=graph_converter_algorithm,
                backend_name="MatRIS",
            )
        return root._instantiate_matris_calculator(model=model, task=task, device=device)

    if root._looks_like_filesystem_path(
        model_value,
        suffixes=(".ckpt", ".pt", ".pth", ".pth.tar", ".tar"),
    ):
        raise FileNotFoundError(f"MatRIS model not found: {model_value}")

    checkpoint_path = root._ensure_matris_named_model_checkpoint(model_value)
    if checkpoint_path is not None:
        model = root._load_matris_checkpoint_model(checkpoint_path, device=device)
        if graph_converter_algorithm is not None:
            model = root._override_model_graph_converter_algorithm(
                model,
                algorithm=graph_converter_algorithm,
                backend_name="MatRIS",
            )
        return root._instantiate_matris_calculator(model=model, task=task, device=device)

    kwargs: Dict[str, object] = {}
    if graph_converter_algorithm is not None and root._callable_supports_parameter(
        root.MatRISCalculator,
        "graph_converter_algorithm",
    ):
        kwargs["graph_converter_algorithm"] = graph_converter_algorithm
        return root.MatRISCalculator(model=model_value, task=task, device=device, **kwargs)

    calculator = root.MatRISCalculator(model=model_value, task=task, device=device, **kwargs)
    if graph_converter_algorithm is not None:
        calculator.model = root._override_model_graph_converter_algorithm(
            calculator.model,
            algorithm=graph_converter_algorithm,
            backend_name="MatRIS",
        )
    return calculator
