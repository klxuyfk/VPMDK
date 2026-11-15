"""Export and verify a MACE-SOYO AOTI package using the upstream exporter."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Sequence
import zipfile


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_metadata(path: Path) -> dict[str, str]:
    """Read the single-model metadata without executing compiled GPU code."""
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith(".wrapper_metadata.json")]
        if len(names) != 1:
            raise ValueError("Expected one AOTI model with embedded wrapper metadata.")
        metadata = json.loads(archive.read(names[0]))
    if metadata.get("mace_soyo_aoti_outputs") != "energy forces stress":
        raise ValueError("Exported package does not declare MACE-SOYO energy/forces/stress.")
    return metadata


def _runtime_info(device: str) -> dict[str, Any]:
    import torch

    target = torch.device(device)
    packages = {}
    for name in ("MACE-SOYO", "torch", "cuequivariance", "cuequivariance-torch",
                 "cuequivariance-ops-cu12", "cuequivariance-ops-torch-cu12"):
        packages[name] = importlib.metadata.version(name)
    return {"packages": packages, "python": sys.version.split()[0], "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(target),
            "compute_capability": list(torch.cuda.get_device_capability(target))}


def _verify_native_ops(device: str) -> None:
    """Fail before compilation if a missing extension would select fallback ops."""
    import torch
    import cuequivariance as cue
    import cuequivariance_torch as cuet
    import cuequivariance_ops_torch  # noqa: F401 -- load errors must propagate
    from cuequivariance_torch.primitives.segmented_polynomial import HAS_CUE_OPS

    if not HAS_CUE_OPS:
        raise RuntimeError("Native cuEquivariance ops are unavailable; export will not use a fallback.")
    polynomial = cue.descriptors.spherical_harmonics(cue.SO3(1), [0, 1]).polynomial
    operation = cuet.SegmentedPolynomial(polynomial, method="uniform_1d")
    if "Naive" in type(operation.m).__name__:
        raise RuntimeError("cuEquivariance selected a fallback evaluator instead of native kernels.")
    with torch.cuda.device(device):
        xyz = torch.tensor([[1., 0., 0.], [0., 1., 0.]], device=device)
        result = operation([xyz])[0]
        if not bool(torch.isfinite(result).all()):
            raise RuntimeError("Native cuEquivariance probe returned nonfinite values.")
        torch.cuda.synchronize(device)


def _run_upstream_export(arguments: list[str]) -> None:
    """Upstream verifies compiled versus eager E/F/S before returning."""
    from mace_soyo.export.AOTI_export import main as upstream_main

    _verify_native_ops(arguments[arguments.index("--device") + 1])
    previous = sys.argv
    try:
        sys.argv = ["mace-soyo-aoti-export", *arguments]
        from .backends.mace_soyo_common import _cueq_checkpoint_compat

        with _cueq_checkpoint_compat():
            upstream_main()
    finally:
        sys.argv = previous


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-path", type=Path, required=True)
    parser.add_argument("--structure-file", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--max-nodes", type=int)
    parser.add_argument("--max-edges", type=int)
    parser.add_argument("--max-graphs", type=int, default=20000)
    parser.add_argument("--inductor-config", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    args.ckpt_path = args.ckpt_path.expanduser()
    args.structure_file = args.structure_file.expanduser()
    for path in (args.ckpt_path, args.structure_file):
        if not path.is_file():
            parser.error(f"Input file not found: {path}")
    if args.output_path.suffix.lower() != ".pt2":
        parser.error("--output-path must end in .pt2")
    if args.max_graphs < 2 or any(value is not None and value < 2
                                 for value in (args.max_nodes, args.max_edges)):
        parser.error("Export dimension bounds must be at least 2")
    if args.device.split(":", 1)[0] != "cuda":
        parser.error("MACE-SOYO export requires a CUDA device")
    output = args.output_path.expanduser().absolute()
    report_path = output.with_suffix(output.suffix + ".json")
    inputs = {args.ckpt_path.resolve(), args.structure_file.resolve()}
    if any(destination.resolve() in inputs for destination in (output, report_path)):
        parser.error("Output package and report must differ from input files")
    if not args.overwrite and (output.exists() or report_path.exists()):
        parser.error("Output or report already exists; use --overwrite to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    # Never publish the unchecked package produced before upstream's numerical
    # checks. A failed export also preserves a previously verified destination.
    with tempfile.TemporaryDirectory(prefix=f".{output.stem}-", dir=output.parent) as directory:
        temporary = Path(directory) / output.name
        arguments = ["--ckpt-path", str(args.ckpt_path.resolve()),
                     "--structure-file", str(args.structure_file.resolve()),
                     "--output-path", str(temporary), "--device", args.device,
                     "--dtype", args.dtype, "--max-graphs", str(args.max_graphs)]
        for tag, value in (("--max-nodes", args.max_nodes), ("--max-edges", args.max_edges)):
            if value is not None:
                arguments.extend([tag, str(value)])
        for config in args.inductor_config:
            arguments.extend(["--inductor-config", config])
        _run_upstream_export(arguments)
        metadata = _package_metadata(temporary)
        report = {"schema": 1, "verified": True, "cueq_method": "native",
                  "checkpoint_sha256": _sha256(args.ckpt_path),
                  "structure_sha256": _sha256(args.structure_file),
                  "package_sha256": _sha256(temporary), "metadata": metadata,
                  "runtime": _runtime_info(args.device),
                  "elapsed_seconds": time.monotonic() - start}
        temporary_report = Path(directory) / report_path.name
        temporary_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, output)
        os.replace(temporary_report, report_path)
    print(f"Verified AOTI package: {output}")
    print(f"Export report: {report_path}")


if __name__ == "__main__":
    main()
