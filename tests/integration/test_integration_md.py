from __future__ import annotations

"""Integration tests that run short MD trajectories with real backends.

Run explicitly with: pytest -m integration

Required:
- CHGNet (no model path required, but chgnet + CUDA are required)
- MACE (set VPMDK_MACE_MODEL)

Optional backends (skipped unless env vars are set):
- SevenNet: VPMDK_SEVENNET_MODEL, optional VPMDK_SEVENNET_MODAL / VPMDK_SEVENNET_FILE_TYPE
- FlashTP: VPMDK_FLASHTP_MODEL, optional VPMDK_FLASHTP_MODAL
- EquFlash: VPMDK_EQUFLASH_MODEL
- MatGL: VPMDK_MATGL_MODEL
- Eqnorm: VPMDK_EQNORM_MODEL, optional VPMDK_EQNORM_VARIANT
- MatRIS: VPMDK_MATRIS_MODEL, optional VPMDK_MATRIS_TASK
- AlphaNet: VPMDK_ALPHANET_MODEL, optional VPMDK_ALPHANET_CONFIG / VPMDK_ALPHANET_PRECISION
- HIENet: VPMDK_HIENET_MODEL, optional VPMDK_HIENET_FILE_TYPE
- Nequix: VPMDK_NEQUIX_MODEL, optional VPMDK_NEQUIX_BACKEND / VPMDK_NEQUIX_USE_KERNEL
- GRACE: VPMDK_GRACE_MODEL
- DeePMD: VPMDK_DEEPMD_MODEL, optional VPMDK_DEEPMD_HEAD
- NequIP: VPMDK_NEQUIP_MODEL
- Allegro: VPMDK_ALLEGRO_MODEL
- ORB: VPMDK_ORB_MODEL
- UPET: VPMDK_UPET_MODEL, optional VPMDK_UPET_VERSION
- TACE: VPMDK_TACE_MODEL, optional VPMDK_TACE_FIDELITY_IDX
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
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def _any_module_available(*names: str) -> bool:
    return any(_module_available(name) for name in names)


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


def _load_test_atoms(data_dir: Path):
    structure = vpmdk.read_structure(str(data_dir / "POSCAR"), None)
    atoms = vpmdk.AseAtomsAdaptor.get_atoms(structure)
    atoms.wrap()
    return atoms


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
def test_md_sevennet_optional(tmp_path: Path, data_dir: Path) -> None:
    if not _any_module_available("sevenn", "sevennet"):
        pytest.skip("SevenNet is not installed.")
    _require_cuda()
    model_value = os.environ.get("VPMDK_SEVENNET_MODEL")
    if not model_value:
        pytest.skip("Set VPMDK_SEVENNET_MODEL to run SevenNet integration.")
    looks_like_path = os.path.sep in model_value or model_value.endswith(
        (".ckpt", ".pth", ".pt", ".jit", ".ts")
    )
    if looks_like_path and not Path(model_value).exists():
        pytest.fail(f"SevenNet model not found: {model_value}")
    modal = os.environ.get("VPMDK_SEVENNET_MODAL", "")
    file_type = os.environ.get("VPMDK_SEVENNET_FILE_TYPE", "")
    bcar_lines = ["MLP=SEVENNET", f"MODEL={model_value}", "DEVICE=cuda"]
    if modal:
        bcar_lines.append(f"SEVENNET_MODAL={modal}")
    if file_type:
        bcar_lines.append(f"SEVENNET_FILE_TYPE={file_type}")
    bcar = "\n".join(bcar_lines) + "\n"
    _write_inputs(tmp_path, data_dir, bcar)
    _run_vpmdk(tmp_path)
    _assert_outputs(tmp_path)


@pytest.mark.integration
def test_md_flashtp_optional(tmp_path: Path, data_dir: Path) -> None:
    if vpmdk._SEVENNET_PACKAGE != "sevenn":
        pytest.skip("FlashTP requires the modern sevenn backend.")
    if not _module_available("flashTP_e3nn"):
        pytest.skip("flashTP_e3nn is not installed.")
    _require_cuda()
    model_value = os.environ.get("VPMDK_FLASHTP_MODEL")
    if not model_value:
        pytest.skip("Set VPMDK_FLASHTP_MODEL to run FlashTP integration.")
    looks_like_path = os.path.sep in model_value or model_value.endswith(
        (".ckpt", ".pth", ".pt")
    )
    if looks_like_path and not Path(model_value).exists():
        pytest.fail(f"FlashTP/SevenNet model not found: {model_value}")
    modal = os.environ.get("VPMDK_FLASHTP_MODAL", "")
    bcar_lines = ["MLP=FLASHTP", f"MODEL={model_value}", "DEVICE=cuda"]
    if modal:
        bcar_lines.append(f"SEVENNET_MODAL={modal}")
    bcar = "\n".join(bcar_lines) + "\n"
    _write_inputs(tmp_path, data_dir, bcar)
    _run_vpmdk(tmp_path)
    _assert_outputs(tmp_path)


@pytest.mark.integration
def test_md_equflash_optional(tmp_path: Path, data_dir: Path) -> None:
    if not _any_module_available("GGNN.common.calculator", "ggnn.common.calculator"):
        pytest.skip("EquFlash calculator package is not installed.")
    _require_cuda()
    model_value = os.environ.get("VPMDK_EQUFLASH_MODEL")
    if not model_value:
        pytest.skip("Set VPMDK_EQUFLASH_MODEL to run EquFlash integration.")
    if not Path(model_value).exists():
        pytest.fail(f"EquFlash model not found: {model_value}")
    bcar = f"MLP=EQUFLASH\nMODEL={model_value}\nDEVICE=cuda\n"
    _write_inputs(tmp_path, data_dir, bcar)
    _run_vpmdk(tmp_path)
    _assert_outputs(tmp_path)


@pytest.mark.integration
@pytest.mark.parametrize("algorithm", ["legacy", "fast"])
def test_chgnet_graph_converter_algorithms_available(
    data_dir: Path, algorithm: str
) -> None:
    if vpmdk.CHGNetCalculator is None or getattr(vpmdk, "CHGNetModel", None) is None:
        pytest.skip("chgnet is not installed.")
    _require_cuda()

    calculator = vpmdk._build_chgnet_calculator(
        {
            "MLP": "CHGNET",
            "DEVICE": "cuda",
            "CHGNET_GRAPH_CONVERTER_ALGORITHM": algorithm,
        }
    )

    assert getattr(calculator.model.graph_converter, "algorithm", None) == algorithm

    atoms = _load_test_atoms(data_dir)
    atoms.calc = calculator
    float(atoms.get_potential_energy())


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
def test_md_eqnorm_optional(tmp_path: Path, data_dir: Path) -> None:
    if not _module_available("eqnorm"):
        pytest.skip("eqnorm is not installed.")
    _require_cuda()
    model_value = os.environ.get("VPMDK_EQNORM_MODEL")
    if not model_value:
        pytest.skip("Set VPMDK_EQNORM_MODEL to run Eqnorm integration.")
    looks_like_path = os.path.sep in model_value or model_value.endswith((".pt", ".pth", ".ckpt"))
    if looks_like_path and not Path(model_value).exists():
        pytest.fail(f"Eqnorm model not found: {model_value}")
    variant = os.environ.get("VPMDK_EQNORM_VARIANT", "")
    bcar_lines = ["MLP=EQNORM", f"MODEL={model_value}", "DEVICE=cuda"]
    if variant:
        bcar_lines.append(f"EQNORM_VARIANT={variant}")
    bcar = "\n".join(bcar_lines) + "\n"
    _write_inputs(tmp_path, data_dir, bcar)
    _run_vpmdk(tmp_path)
    _assert_outputs(tmp_path)


@pytest.mark.integration
def test_md_matris_optional(tmp_path: Path, data_dir: Path) -> None:
    if not _module_available("matris"):
        pytest.skip("matris is not installed.")
    _require_cuda()
    model_value = os.environ.get("VPMDK_MATRIS_MODEL")
    if not model_value:
        pytest.skip("Set VPMDK_MATRIS_MODEL to run MatRIS integration.")
    looks_like_path = os.path.sep in model_value or model_value.endswith(
        (".ckpt", ".pt", ".pth", ".pth.tar", ".tar")
    )
    if looks_like_path and not Path(model_value).exists():
        pytest.fail(f"MatRIS model not found: {model_value}")
    task = os.environ.get("VPMDK_MATRIS_TASK", "")
    bcar_lines = ["MLP=MATRIS", f"MODEL={model_value}", "DEVICE=cuda"]
    if task:
        bcar_lines.append(f"MATRIS_TASK={task}")
    bcar = "\n".join(bcar_lines) + "\n"
    _write_inputs(tmp_path, data_dir, bcar)
    _run_vpmdk(tmp_path)
    _assert_outputs(tmp_path)


@pytest.mark.integration
@pytest.mark.parametrize("algorithm", ["legacy", "fast"])
def test_matris_graph_converter_algorithms_optional(
    data_dir: Path, algorithm: str
) -> None:
    if not _module_available("matris"):
        pytest.skip("matris is not installed.")
    _require_cuda()
    model_value = os.environ.get("VPMDK_MATRIS_MODEL")
    if not model_value:
        pytest.skip("Set VPMDK_MATRIS_MODEL to run MatRIS integration.")
    looks_like_path = os.path.sep in model_value or model_value.endswith(
        (".ckpt", ".pt", ".pth", ".pth.tar", ".tar")
    )
    if looks_like_path and not Path(model_value).exists():
        pytest.fail(f"MatRIS model not found: {model_value}")

    tags = {
        "MLP": "MATRIS",
        "MODEL": model_value,
        "DEVICE": "cuda",
        "MATRIS_GRAPH_CONVERTER_ALGORITHM": algorithm,
    }
    task = os.environ.get("VPMDK_MATRIS_TASK", "")
    if task:
        tags["MATRIS_TASK"] = task

    calculator = vpmdk._build_matris_calculator(tags)
    atoms = _load_test_atoms(data_dir)
    atoms.calc = calculator
    float(atoms.get_potential_energy())

    graph_converter = getattr(getattr(calculator, "model", None), "graph_converter", None)
    if graph_converter is not None:
        assert getattr(graph_converter, "algorithm", None) == algorithm


@pytest.mark.integration
def test_md_alphanet_optional(tmp_path: Path, data_dir: Path) -> None:
    if not _module_available("alphanet"):
        pytest.skip("alphanet is not installed.")
    _require_cuda()
    model_value = os.environ.get("VPMDK_ALPHANET_MODEL")
    if not model_value:
        pytest.skip("Set VPMDK_ALPHANET_MODEL to run AlphaNet integration.")
    looks_like_path = os.path.sep in model_value or model_value.endswith((".ckpt", ".pt", ".pth"))
    if looks_like_path and not Path(model_value).exists():
        pytest.fail(f"AlphaNet model not found: {model_value}")
    config_path = os.environ.get("VPMDK_ALPHANET_CONFIG", "")
    if config_path and not Path(config_path).exists():
        pytest.fail(f"AlphaNet config not found: {config_path}")
    precision = os.environ.get("VPMDK_ALPHANET_PRECISION", "")
    bcar_lines = ["MLP=ALPHANET", f"MODEL={model_value}", "DEVICE=cuda"]
    if config_path:
        bcar_lines.append(f"ALPHANET_CONFIG={config_path}")
    if precision:
        bcar_lines.append(f"ALPHANET_PRECISION={precision}")
    bcar = "\n".join(bcar_lines) + "\n"
    _write_inputs(tmp_path, data_dir, bcar)
    _run_vpmdk(tmp_path)
    _assert_outputs(tmp_path)


@pytest.mark.integration
def test_md_hienet_optional(tmp_path: Path, data_dir: Path) -> None:
    if not _module_available("hienet"):
        pytest.skip("hienet is not installed.")
    _require_cuda()
    model_value = os.environ.get("VPMDK_HIENET_MODEL")
    if not model_value:
        pytest.skip("Set VPMDK_HIENET_MODEL to run HIENet integration.")
    looks_like_path = os.path.sep in model_value or model_value.endswith(
        (".ckpt", ".pt", ".pth", ".jit", ".ts")
    )
    if looks_like_path and not Path(model_value).exists():
        pytest.fail(f"HIENet model not found: {model_value}")
    file_type = os.environ.get("VPMDK_HIENET_FILE_TYPE", "")
    bcar_lines = ["MLP=HIENET", f"MODEL={model_value}", "DEVICE=cuda"]
    if file_type:
        bcar_lines.append(f"HIENET_FILE_TYPE={file_type}")
    bcar = "\n".join(bcar_lines) + "\n"
    _write_inputs(tmp_path, data_dir, bcar)
    _run_vpmdk(tmp_path)
    _assert_outputs(tmp_path)


@pytest.mark.integration
def test_md_nequix_optional(tmp_path: Path, data_dir: Path) -> None:
    if not _module_available("nequix"):
        pytest.skip("nequix is not installed.")
    _require_cuda()
    model_value = os.environ.get("VPMDK_NEQUIX_MODEL")
    if not model_value:
        pytest.skip("Set VPMDK_NEQUIX_MODEL to run Nequix integration.")
    looks_like_path = os.path.sep in model_value or model_value.endswith(
        (".nqx", ".pt", ".pth", ".ckpt")
    )
    if looks_like_path and not Path(model_value).exists():
        pytest.fail(f"Nequix model not found: {model_value}")
    backend = os.environ.get("VPMDK_NEQUIX_BACKEND", "")
    use_kernel = os.environ.get("VPMDK_NEQUIX_USE_KERNEL", "")
    bcar_lines = ["MLP=NEQUIX", f"MODEL={model_value}", "DEVICE=cuda"]
    if backend:
        bcar_lines.append(f"NEQUIX_BACKEND={backend}")
    if use_kernel:
        bcar_lines.append(f"NEQUIX_USE_KERNEL={use_kernel}")
    bcar = "\n".join(bcar_lines) + "\n"
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
def test_md_upet_optional(tmp_path: Path, data_dir: Path) -> None:
    if not _module_available("upet"):
        pytest.skip("upet is not installed.")
    _require_cuda()
    model_value = os.environ.get("VPMDK_UPET_MODEL")
    if not model_value:
        pytest.skip("Set VPMDK_UPET_MODEL to run UPET integration.")
    looks_like_path = os.path.sep in model_value or model_value.endswith((".ckpt", ".pt", ".pth"))
    if looks_like_path and not Path(model_value).exists():
        pytest.fail(f"UPET model not found: {model_value}")
    version = os.environ.get("VPMDK_UPET_VERSION", "")
    bcar_lines = ["MLP=UPET", f"MODEL={model_value}", "DEVICE=cuda"]
    if version:
        bcar_lines.append(f"UPET_VERSION={version}")
    bcar = "\n".join(bcar_lines) + "\n"
    _write_inputs(tmp_path, data_dir, bcar)
    _run_vpmdk(tmp_path)
    _assert_outputs(tmp_path)


@pytest.mark.integration
def test_md_tace_optional(tmp_path: Path, data_dir: Path) -> None:
    if not _module_available("tace"):
        pytest.skip("tace is not installed.")
    _require_cuda()
    model_value = os.environ.get("VPMDK_TACE_MODEL")
    if not model_value:
        pytest.skip("Set VPMDK_TACE_MODEL to run TACE integration.")
    looks_like_path = os.path.sep in model_value or model_value.endswith((".ckpt", ".pt", ".pth"))
    if looks_like_path and not Path(model_value).exists():
        pytest.fail(f"TACE model not found: {model_value}")
    fidelity_idx = os.environ.get("VPMDK_TACE_FIDELITY_IDX", os.environ.get("VPMDK_TACE_LEVEL", ""))
    bcar_lines = ["MLP=TACE", f"MODEL={model_value}", "DEVICE=cuda"]
    if fidelity_idx:
        bcar_lines.append(f"TACE_FIDELITY_IDX={fidelity_idx}")
    bcar = "\n".join(bcar_lines) + "\n"
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
