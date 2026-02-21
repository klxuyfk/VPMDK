from __future__ import annotations

"""Integration tests that run short MD trajectories with real backends.

Run explicitly with: pytest -m integration

Required:
- CHGNet (no model path required, but chgnet + CUDA are required)
- MACE (set VPMDK_MACE_MODEL)

Optional backends (skipped unless env vars are set):
- MatGL: VPMDK_MATGL_MODEL
- GRACE: VPMDK_GRACE_MODEL
- DeePMD: VPMDK_DEEPMD_MODEL, optional VPMDK_DEEPMD_HEAD
- NequIP: VPMDK_NEQUIP_MODEL
- Allegro: VPMDK_ALLEGRO_MODEL
- ORB: VPMDK_ORB_MODEL
- FAIRChem v2: VPMDK_FAIRCHEM_MODEL, optional VPMDK_FAIRCHEM_TASK
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

import vpmdk


INCAR_MD = """IBRION = 0
NSW = 5
POTIM = 1.0
TEBEG = 300
TEEND = 300
MDALGO = 0
"""


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _require_cuda() -> None:
    if not _module_available("torch"):
        pytest.skip("PyTorch is not installed; CUDA checks unavailable.")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available in this environment.")


def _write_inputs(tmp_path: Path, data_dir: Path, bcar_text: str) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "POSCAR").write_text((data_dir / "POSCAR").read_text())
    (tmp_path / "INCAR").write_text(INCAR_MD)
    (tmp_path / "BCAR").write_text(bcar_text)


def _run_vpmdk(calc_dir: Path) -> None:
    argv = ["vpmdk.py", "--dir", str(calc_dir)]
    original_argv = sys.argv[:]
    original_cwd = Path.cwd()
    sys.argv[:] = argv
    try:
        os.chdir(calc_dir)
        vpmdk.main()
    finally:
        sys.argv[:] = original_argv
        os.chdir(original_cwd)


def _assert_outputs(calc_dir: Path) -> None:
    for name in ("CONTCAR", "OUTCAR", "XDATCAR"):
        path = calc_dir / name
        assert path.exists(), f"Missing output file: {name}"
        assert path.stat().st_size > 0, f"Empty output file: {name}"


@pytest.mark.integration
def test_md_chgnet_required(tmp_path: Path, data_dir: Path) -> None:
    if vpmdk.CHGNetCalculator is None:
        pytest.skip("chgnet is not installed.")
    _require_cuda()
    bcar = "MLP=CHGNET\nDEVICE=cuda\n"
    _write_inputs(tmp_path, data_dir, bcar)
    _run_vpmdk(tmp_path)
    _assert_outputs(tmp_path)


@pytest.mark.integration
def test_md_mace_required(tmp_path: Path, data_dir: Path) -> None:
    _require_cuda()
    model_path = os.environ.get("VPMDK_MACE_MODEL")
    if not model_path:
        pytest.skip("Set VPMDK_MACE_MODEL to run MACE integration.")
    if not Path(model_path).exists():
        pytest.fail(f"MACE model not found: {model_path}")
    bcar = f"MLP=MACE\nMODEL={model_path}\nDEVICE=cuda\n"
    _write_inputs(tmp_path, data_dir, bcar)
    _run_vpmdk(tmp_path)
    _assert_outputs(tmp_path)


@pytest.mark.integration
def test_md_matgl_optional(tmp_path: Path, data_dir: Path) -> None:
    if not _module_available("matgl"):
        pytest.skip("matgl is not installed.")
    _require_cuda()
    model_path = os.environ.get("VPMDK_MATGL_MODEL")
    if not model_path:
        pytest.skip("Set VPMDK_MATGL_MODEL to run MatGL integration.")
    if not Path(model_path).exists():
        pytest.fail(f"MatGL model not found: {model_path}")
    bcar = f"MLP=MATGL\nMODEL={model_path}\nDEVICE=cuda\n"
    _write_inputs(tmp_path, data_dir, bcar)
    _run_vpmdk(tmp_path)
    _assert_outputs(tmp_path)


@pytest.mark.integration
def test_md_grace_optional(tmp_path: Path, data_dir: Path) -> None:
    if not _module_available("tensorpotential"):
        pytest.skip("grace-tensorpotential is not installed.")
    _require_cuda()
    model_path = os.environ.get("VPMDK_GRACE_MODEL")
    if not model_path:
        pytest.skip("Set VPMDK_GRACE_MODEL to run GRACE integration.")
    if not Path(model_path).exists():
        pytest.fail(f"GRACE model not found: {model_path}")
    bcar = f"MLP=GRACE\nMODEL={model_path}\nDEVICE=cuda\n"
    _write_inputs(tmp_path, data_dir, bcar)
    _run_vpmdk(tmp_path)
    _assert_outputs(tmp_path)


@pytest.mark.integration
def test_md_deepmd_optional(tmp_path: Path, data_dir: Path) -> None:
    if not _module_available("deepmd"):
        pytest.skip("deepmd-kit is not installed.")
    _require_cuda()
    model_path = os.environ.get("VPMDK_DEEPMD_MODEL")
    if not model_path:
        pytest.skip("Set VPMDK_DEEPMD_MODEL to run DeePMD integration.")
    if not Path(model_path).exists():
        pytest.fail(f"DeePMD model not found: {model_path}")
    head = os.environ.get("VPMDK_DEEPMD_HEAD", "")
    bcar_lines = ["MLP=DEEPMD", f"MODEL={model_path}", "DEVICE=cuda"]
    if head:
        bcar_lines.append(f"DEEPMD_HEAD={head}")
    bcar = "\n".join(bcar_lines) + "\n"
    _write_inputs(tmp_path, data_dir, bcar)
    _run_vpmdk(tmp_path)
    _assert_outputs(tmp_path)


@pytest.mark.integration
def test_md_nequip_optional(tmp_path: Path, data_dir: Path) -> None:
    if not _module_available("nequip"):
        pytest.skip("nequip is not installed.")
    _require_cuda()
    model_path = os.environ.get("VPMDK_NEQUIP_MODEL")
    if not model_path:
        pytest.skip("Set VPMDK_NEQUIP_MODEL to run NequIP integration.")
    if not Path(model_path).exists():
        pytest.fail(f"NequIP model not found: {model_path}")
    bcar = f"MLP=NEQUIP\nMODEL={model_path}\nDEVICE=cuda\n"
    _write_inputs(tmp_path, data_dir, bcar)
    _run_vpmdk(tmp_path)
    _assert_outputs(tmp_path)


@pytest.mark.integration
def test_md_allegro_optional(tmp_path: Path, data_dir: Path) -> None:
    if not _module_available("allegro"):
        pytest.skip("allegro is not installed.")
    _require_cuda()
    model_path = os.environ.get("VPMDK_ALLEGRO_MODEL")
    if not model_path:
        pytest.skip("Set VPMDK_ALLEGRO_MODEL to run Allegro integration.")
    if not Path(model_path).exists():
        pytest.fail(f"Allegro model not found: {model_path}")
    bcar = f"MLP=ALLEGRO\nMODEL={model_path}\nDEVICE=cuda\n"
    _write_inputs(tmp_path, data_dir, bcar)
    _run_vpmdk(tmp_path)
    _assert_outputs(tmp_path)


@pytest.mark.integration
def test_md_orb_optional(tmp_path: Path, data_dir: Path) -> None:
    if not _module_available("orb_models"):
        pytest.skip("orb-models is not installed.")
    _require_cuda()
    model_path = os.environ.get("VPMDK_ORB_MODEL")
    if not model_path:
        pytest.skip("Set VPMDK_ORB_MODEL to run ORB integration.")
    if not Path(model_path).exists():
        pytest.fail(f"ORB model not found: {model_path}")
    bcar = f"MLP=ORB\nMODEL={model_path}\nDEVICE=cuda\n"
    _write_inputs(tmp_path, data_dir, bcar)
    _run_vpmdk(tmp_path)
    _assert_outputs(tmp_path)


@pytest.mark.integration
def test_md_fairchem_v2_optional(tmp_path: Path, data_dir: Path) -> None:
    if not _module_available("fairchem"):
        pytest.skip("fairchem is not installed.")
    _require_cuda()
    model_value = os.environ.get("VPMDK_FAIRCHEM_MODEL")
    if not model_value:
        pytest.skip("Set VPMDK_FAIRCHEM_MODEL to run FAIRChem v2 integration.")
    if os.path.sep in model_value and not Path(model_value).exists():
        pytest.fail(f"FAIRChem model not found: {model_value}")
    task = os.environ.get("VPMDK_FAIRCHEM_TASK", "")
    bcar_lines = ["MLP=FAIRCHEM_V2", f"MODEL={model_value}", "DEVICE=cuda"]
    if task:
        bcar_lines.append(f"FAIRCHEM_TASK={task}")
    bcar = "\n".join(bcar_lines) + "\n"
    _write_inputs(tmp_path, data_dir, bcar)
    _run_vpmdk(tmp_path)
    _assert_outputs(tmp_path)
