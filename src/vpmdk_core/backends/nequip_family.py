"""NequIP-family backend builders."""

from __future__ import annotations

import importlib
import os
import sys
from typing import Any, Dict


def _root():
    return sys.modules["vpmdk_core"]


def _build_nequip_family_calculator(
    bcar_tags: Dict[str, str],
    *,
    require_allegro: bool = False,
    missing_message: str,
):
    """Create NequIP-based calculators that require deployed model files."""

    root = _root()
    if require_allegro and importlib.util.find_spec("allegro") is None:
        raise RuntimeError(
            "Allegro calculator not available. Install allegro and dependencies."
        )
    if root.NequIPCalculator is None:
        raise RuntimeError(missing_message)

    model_path = bcar_tags.get("MODEL")
    model_name = "Allegro" if require_allegro else "NequIP"
    if not model_path:
        raise ValueError(f"{model_name} requires MODEL pointing to a deployed model file.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_name} model not found: {model_path}")

    device = bcar_tags.get("DEVICE")
    if hasattr(root.NequIPCalculator, "from_deployed_model"):
        try:
            if device:
                return root.NequIPCalculator.from_deployed_model(model_path, device=device)
            return root.NequIPCalculator.from_deployed_model(model_path)
        except Exception as exc:
            ext = os.path.splitext(model_path)[1].lower()
            if ext not in {".pt", ".pth"}:
                raise
            if not hasattr(root.NequIPCalculator, "from_compiled_model"):
                raise
            try:
                if device:
                    return root.NequIPCalculator.from_compiled_model(model_path, device=device)
                return root.NequIPCalculator.from_compiled_model(model_path)
            except Exception as compiled_exc:
                raise compiled_exc from exc

    if hasattr(root.NequIPCalculator, "from_compiled_model"):
        if device:
            return root.NequIPCalculator.from_compiled_model(model_path, device=device)
        return root.NequIPCalculator.from_compiled_model(model_path)

    raise RuntimeError(
        f"{model_name} calculator does not expose from_deployed_model or from_compiled_model."
    )


def _build_nequip_calculator(bcar_tags: Dict[str, str], *, structure=None):
    """Create a NequIP calculator from a deployed model."""

    return _build_nequip_family_calculator(
        bcar_tags,
        missing_message="NequIPCalculator not available. Install nequip and dependencies.",
    )


def _build_allegro_calculator(bcar_tags: Dict[str, str], *, structure=None):
    """Create an Allegro calculator from a deployed model."""

    return _build_nequip_family_calculator(
        bcar_tags,
        require_allegro=True,
        missing_message="NequIPCalculator not available. Install nequip and dependencies.",
    )


def _resolve_graph_converter_algorithm(
    bcar_tags: Dict[str, str], *, backend_tag: str
) -> str | None:
    """Return an optional fast/legacy graph-converter selection from BCAR."""

    root = _root()
    for tag_name in (
        f"{backend_tag}_GRAPH_CONVERTER_ALGORITHM",
        f"{backend_tag}_GRAPH_CONVERTER",
        "GRAPH_CONVERTER_ALGORITHM",
        "GRAPH_CONVERTER",
    ):
        raw_value = bcar_tags.get(tag_name)
        if raw_value is None:
            continue
        algorithm = str(raw_value).strip().lower()
        if algorithm in root._GRAPH_CONVERTER_ALGORITHMS:
            return algorithm
        supported = ", ".join(sorted(root._GRAPH_CONVERTER_ALGORITHMS))
        raise ValueError(
            f"Invalid {tag_name} value: {raw_value!r}. Expected one of: {supported}."
        )
    return None


def _override_model_graph_converter_algorithm(model, *, algorithm: str, backend_name: str):
    """Replace a model's graph converter with the requested algorithm."""

    root = _root()
    if algorithm not in root._GRAPH_CONVERTER_ALGORITHMS:
        supported = ", ".join(sorted(root._GRAPH_CONVERTER_ALGORITHMS))
        raise ValueError(
            f"Unsupported {backend_name} graph converter algorithm {algorithm!r}. "
            f"Expected one of: {supported}."
        )

    graph_converter = getattr(model, "graph_converter", None)
    if graph_converter is None:
        raise RuntimeError(
            f"{backend_name} model does not expose graph_converter; cannot set "
            f"{algorithm!r}."
        )

    if getattr(graph_converter, "algorithm", None) == algorithm:
        return model

    try:
        signature = root.inspect.signature(type(graph_converter))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{backend_name} graph converter cannot be reconfigured dynamically."
        ) from exc

    kwargs: Dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        if name == "algorithm":
            kwargs[name] = algorithm
            continue
        if hasattr(graph_converter, name):
            kwargs[name] = getattr(graph_converter, name)
            continue
        if parameter.default is root.inspect.Signature.empty:
            raise RuntimeError(
                f"{backend_name} graph converter requires {name!r}; cannot set "
                f"{algorithm!r} from the loaded model."
            )

    converter_cls = type(graph_converter)
    module = root.inspect.getmodule(converter_cls)
    make_graph = getattr(module, "make_graph", None) if module is not None else None

    try:
        if "algorithm" in signature.parameters:
            kwargs["algorithm"] = algorithm
            new_converter = converter_cls(**kwargs)
        elif module is not None and hasattr(module, "make_graph"):
            original_make_graph = make_graph
            if algorithm == "legacy":
                module.make_graph = None
            else:
                if make_graph is None:
                    package = getattr(module, "__package__", None)
                    if package:
                        try:
                            cygraph_module = importlib.import_module(f"{package}.cygraph")
                            module.make_graph = getattr(cygraph_module, "make_graph", None)
                        except Exception:
                            module.make_graph = None
                if module.make_graph is None:
                    raise RuntimeError(
                        f"{backend_name} fast graph converter is not available in this "
                        f"environment."
                    )
            try:
                new_converter = converter_cls(**kwargs)
            finally:
                module.make_graph = original_make_graph
        else:
            raise RuntimeError(
                f"{backend_name} graph converter does not accept an algorithm selector."
            )
    except Exception as exc:
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(
            f"Failed to build {backend_name} graph converter with algorithm={algorithm!r}."
        ) from exc

    isolated_atoms_response = getattr(graph_converter, "on_isolated_atoms", None)
    if isolated_atoms_response is not None and hasattr(
        new_converter, "set_isolated_atom_response"
    ):
        new_converter.set_isolated_atom_response(isolated_atoms_response)

    actual_algorithm = getattr(new_converter, "algorithm", algorithm)
    if actual_algorithm != algorithm:
        raise RuntimeError(
            f"{backend_name} graph converter requested {algorithm!r} but initialized "
            f"{actual_algorithm!r}."
        )

    model.graph_converter = new_converter
    return model
