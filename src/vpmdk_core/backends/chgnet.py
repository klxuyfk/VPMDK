"""CHGNet backend builder."""

from __future__ import annotations

import os
import sys
from typing import Dict


def _root():
    return sys.modules["vpmdk_core"]


def _load_chgnet_model(
    *,
    model_path: str | None,
    device: str | None,
    graph_converter_algorithm: str | None,
):
    """Load a CHGNet model with optional graph-converter override."""

    root = _root()
    if root.CHGNetModel is None:
        raise RuntimeError("CHGNet model loader not available. Install chgnet.")

    if model_path and os.path.exists(model_path):
        if graph_converter_algorithm is not None:
            try:
                return root.CHGNetModel.from_file(
                    model_path,
                    graph_converter_algorithm=graph_converter_algorithm,
                )
            except TypeError:
                model = root.CHGNetModel.from_file(model_path)
                return root._override_model_graph_converter_algorithm(
                    model,
                    algorithm=graph_converter_algorithm,
                    backend_name="CHGNet",
                )
        return root.CHGNetModel.from_file(model_path)

    named_model = None
    if model_path and not root._looks_like_filesystem_path(
        model_path,
        suffixes=(".pt", ".pth", ".ckpt", ".tar"),
    ):
        named_model = model_path

    load_attempts: list[tuple[tuple[object, ...], dict[str, object]]] = []
    if named_model is not None:
        load_attempts.extend(
            [
                ((), {"model_name": named_model, "use_device": device, "verbose": False}),
                ((), {"model_name": named_model, "use_device": device}),
                ((), {"model_name": named_model, "verbose": False}),
                ((named_model,), {"use_device": device, "verbose": False}),
                ((named_model,), {"use_device": device}),
                ((named_model,), {"verbose": False}),
                ((named_model,), {}),
            ]
        )
    else:
        load_attempts.extend(
            [
                ((), {"verbose": False, "use_device": device}),
                ((), {"use_device": device}),
                ((), {"verbose": False}),
                ((), {}),
            ]
        )

    model = None
    for args, kwargs in load_attempts:
        filtered_kwargs = {key: value for key, value in kwargs.items() if value is not None}
        try:
            model = root.CHGNetModel.load(*args, **filtered_kwargs)
            break
        except TypeError:
            continue
    if model is None:
        raise RuntimeError("CHGNet model loader does not expose a compatible load() signature.")

    if graph_converter_algorithm is not None:
        model = root._override_model_graph_converter_algorithm(
            model,
            algorithm=graph_converter_algorithm,
            backend_name="CHGNet",
        )
    return model


def _build_chgnet_calculator(bcar_tags: Dict[str, str]):
    """Create a CHGNet calculator with optional DEVICE hint."""

    root = _root()
    if root.CHGNetCalculator is None:
        raise RuntimeError("CHGNetCalculator not available. Install chgnet.")

    model_path = bcar_tags.get("MODEL")
    device = root._resolve_device(bcar_tags.get("DEVICE"))
    graph_converter_algorithm = root._resolve_graph_converter_algorithm(
        bcar_tags,
        backend_tag="CHGNET",
    )
    kwargs = {"use_device": device} if device is not None else {}

    if graph_converter_algorithm is not None:
        model = root._load_chgnet_model(
            model_path=model_path,
            device=device,
            graph_converter_algorithm=graph_converter_algorithm,
        )
        return root.CHGNetCalculator(model=model, **kwargs)

    if model_path and os.path.exists(model_path):
        from_file = getattr(root.CHGNetCalculator, "from_file", None)
        if callable(from_file):
            try:
                return from_file(model_path, **kwargs)
            except TypeError:
                return from_file(model_path)
        try:
            return root.CHGNetCalculator(model_path, **kwargs)
        except TypeError:
            return root.CHGNetCalculator(model_path)

    try:
        return root.CHGNetCalculator(**kwargs)
    except TypeError:
        return root.CHGNetCalculator()
