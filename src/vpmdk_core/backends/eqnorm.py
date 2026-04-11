"""Eqnorm backend builder."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Any, Dict


def _root():
    return sys.modules["vpmdk_core"]


def _normalize_eqnorm_key(value: str) -> str:
    """Return a separator-insensitive Eqnorm lookup key."""

    return _root().re.sub(r"[^a-z0-9]+", "", str(value).strip().casefold())


def _match_eqnorm_variant(value: str | None) -> str | None:
    """Return a canonical Eqnorm variant when a value matches one."""

    root = _root()
    if not value:
        return None

    normalized = _normalize_eqnorm_key(value)
    for variant, aliases in root._EQNORM_VARIANT_ALIASES.items():
        if normalized in {_normalize_eqnorm_key(alias) for alias in aliases}:
            return variant
    return None


def _normalize_eqnorm_variant(value: str) -> str:
    """Validate and normalize an Eqnorm model variant."""

    root = _root()
    variant = _match_eqnorm_variant(value)
    if variant is not None:
        return variant

    supported = ", ".join(sorted(root._EQNORM_VARIANT_ALIASES))
    raise ValueError(f"Invalid EQNORM_VARIANT value: {value!r}. Available: {supported}")


def _resolve_eqnorm_named_model_spec(model_name: str) -> Dict[str, Any] | None:
    """Return Eqnorm named-model metadata for a model key or alias."""

    root = _root()
    normalized = _normalize_eqnorm_key(model_name)
    for spec in root._EQNORM_NAMED_MODELS.values():
        aliases = [
            spec["display_name"],
            spec.get("model_variant", ""),
            spec.get("model_name", ""),
            *spec.get("aliases", []),
        ]
        if normalized in {_normalize_eqnorm_key(alias) for alias in aliases if alias}:
            return spec
    return None


def _resolve_eqnorm_download_url(spec: Dict[str, Any]) -> str:
    """Return the best available download URL for an Eqnorm named model."""

    root = _root()
    article_api_url = spec.get("article_api_url")
    expected_filename = spec["checkpoint_filename"]
    if article_api_url:
        request = root.urllib.request.Request(article_api_url, headers={"User-Agent": "vpmdk"})
        try:
            with root.urllib.request.urlopen(request) as response:
                payload = root.json.load(response)
            for file_info in payload.get("files", []):
                if file_info.get("name") == expected_filename and file_info.get("download_url"):
                    return str(file_info["download_url"])
        except Exception:
            pass

    return spec["checkpoint_url"]


def _ensure_eqnorm_named_model_checkpoint(model_name: str) -> tuple[Dict[str, Any], str]:
    """Download a known Eqnorm named model into the standard cache when needed."""

    root = _root()
    spec = _resolve_eqnorm_named_model_spec(model_name)
    if spec is None:
        supported = ", ".join(
            sorted(named_spec["display_name"] for named_spec in root._EQNORM_NAMED_MODELS.values())
        )
        raise ValueError(f"Unsupported Eqnorm model '{model_name}'. Available: {supported}")

    cache_dir = root.os.path.expanduser("~/.cache/eqnorm")
    root.os.makedirs(cache_dir, exist_ok=True)
    checkpoint_path = root.os.path.join(cache_dir, spec["checkpoint_filename"])
    if not root.os.path.exists(checkpoint_path) or root.os.path.getsize(checkpoint_path) == 0:
        print(f"Eqnorm checkpoint not found, downloading to {checkpoint_path} ...")
        root._download_file_to_path(_resolve_eqnorm_download_url(spec), checkpoint_path)
    return spec, checkpoint_path


def _resolve_eqnorm_variant(
    model_value: str,
    bcar_tags: Dict[str, str],
    *,
    named_variant: str | None = None,
) -> str:
    """Resolve the Eqnorm architecture variant from BCAR tags or a checkpoint path."""

    root = _root()
    explicit_variant = bcar_tags.get("EQNORM_VARIANT")
    if explicit_variant:
        resolved = _normalize_eqnorm_variant(explicit_variant)
        if named_variant is not None and resolved != named_variant:
            raise ValueError(
                f"EQNORM_VARIANT={explicit_variant!r} does not match named model variant "
                f"{named_variant!r}."
            )
        return resolved

    if named_variant is not None:
        return named_variant

    candidate = root.os.path.basename(model_value)
    while candidate:
        inferred = _match_eqnorm_variant(candidate)
        if inferred is not None:
            return inferred
        stem, ext = root.os.path.splitext(candidate)
        if not ext:
            break
        candidate = stem

    supported = ", ".join(sorted(root._EQNORM_VARIANT_ALIASES))
    raise ValueError(
        "Eqnorm local checkpoints require EQNORM_VARIANT set to one of "
        f"{supported} when the variant cannot be inferred from the filename."
    )


def _stage_eqnorm_checkpoint(checkpoint_path: str, variant: str) -> str:
    """Expose a checkpoint at the cache path expected by the upstream calculator."""

    root = _root()
    cache_dir = root.os.path.expanduser("~/.cache/eqnorm")
    root.os.makedirs(cache_dir, exist_ok=True)
    staged_path = root.os.path.join(cache_dir, f"{variant}.pt")
    source_path = root.os.path.abspath(checkpoint_path)

    if root.os.path.abspath(staged_path) == source_path:
        return staged_path

    try:
        if root.os.path.exists(staged_path) and root.os.path.samefile(staged_path, source_path):
            return staged_path
    except FileNotFoundError:
        pass

    if root.os.path.lexists(staged_path):
        root.os.remove(staged_path)

    try:
        root.os.symlink(source_path, staged_path)
    except Exception:
        root.shutil.copy2(source_path, staged_path)

    return staged_path


@contextmanager
def _temporarily_stage_eqnorm_local_checkpoint(checkpoint_path: str, variant: str):
    """Temporarily expose a local Eqnorm checkpoint without poisoning the named cache."""

    root = _root()
    cache_dir = root.os.path.expanduser("~/.cache/eqnorm")
    root.os.makedirs(cache_dir, exist_ok=True)

    staged_path = root.os.path.join(cache_dir, f"{variant}.pt")
    source_path = root.os.path.abspath(checkpoint_path)
    if root.os.path.abspath(staged_path) == source_path:
        yield staged_path
        return

    try:
        if root.os.path.exists(staged_path) and root.os.path.samefile(staged_path, source_path):
            yield staged_path
            return
    except FileNotFoundError:
        pass

    backup_path = None
    if root.os.path.lexists(staged_path):
        backup_path = root.os.path.join(
            cache_dir,
            f".{variant}.vpmdk-backup-{root.time.time_ns()}.pt",
        )
        root.os.replace(staged_path, backup_path)

    staged_path = _stage_eqnorm_checkpoint(source_path, variant)
    try:
        yield staged_path
    finally:
        if root.os.path.lexists(staged_path):
            root.os.remove(staged_path)
        if backup_path is not None:
            root.os.replace(backup_path, staged_path)


def _ensure_eqnorm_torch_safe_globals() -> None:
    """Allowlist globals needed by e3nn constants on PyTorch 2.6+."""

    try:
        import torch.serialization

        torch.serialization.add_safe_globals([slice])
    except Exception:
        pass


def _build_eqnorm_calculator(bcar_tags: Dict[str, str]):
    """Create the Eqnorm ASE calculator configured from BCAR tags."""

    root = _root()
    if root.EqnormCalculator is None:
        raise RuntimeError("Eqnorm calculator not available. Install eqnorm and dependencies.")

    model_value = bcar_tags.get("MODEL") or root.DEFAULT_EQNORM_MODEL
    device = root._resolve_device(bcar_tags.get("DEVICE")) or "cpu"
    compile_flag = False
    compile_value = bcar_tags.get("EQNORM_COMPILE")
    if compile_value is not None:
        compile_flag = root._coerce_bool_tag(compile_value, "EQNORM_COMPILE")

    root._ensure_eqnorm_torch_safe_globals()

    if os.path.exists(model_value):
        variant = _resolve_eqnorm_variant(model_value, bcar_tags)
        with root._temporarily_stage_eqnorm_local_checkpoint(model_value, variant):
            return root.EqnormCalculator(
                model_name="eqnorm",
                model_variant=variant,
                device=device,
                compile=compile_flag,
            )
    if root._looks_like_filesystem_path(model_value, suffixes=(".pt", ".pth", ".ckpt")):
        raise FileNotFoundError(f"Eqnorm model not found: {model_value}")

    spec, _ = root._ensure_eqnorm_named_model_checkpoint(model_value)
    variant = _resolve_eqnorm_variant(
        model_value,
        bcar_tags,
        named_variant=spec["model_variant"],
    )
    return root.EqnormCalculator(
        model_name="eqnorm",
        model_variant=variant,
        device=device,
        compile=compile_flag,
    )
