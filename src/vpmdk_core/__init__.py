"""vpmdk: Run machine-learning potentials using VASP style inputs.

The utility consumes VASP-style inputs (POSCAR, INCAR, POTCAR, BCAR) and
executes single-point, relaxation, or molecular dynamics runs with the selected
neural-network potential. Multiple ASE calculators are supported (CHGNet,
M3GNet/MatGL, MACE, MatterSim, Matlantis, Eqnorm, MatRIS, AlphaNet, HIENet,
Nequix, SevenNet, FlashTP, EquFlash, UPET, TACE) and the expected VASP outputs
such as CONTCAR and OUTCAR-style energy logs are produced.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import os
import re
import shutil
import sys
import time
import urllib.request
from typing import Any, Callable, Dict, Iterable, List

import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp import Incar, Poscar, Potcar

try:
    from chgnet.model import CHGNet as CHGNetModel
    from chgnet.model import CHGNetCalculator
except Exception:
    CHGNetModel = None  # type: ignore
    CHGNetCalculator = None  # type: ignore

LegacyM3GNet = None
LegacyM3GNetPotential = None
MatGLLoadModel = None

try:
    from matgl.ext.ase import M3GNetCalculator  # type: ignore

    try:
        import matgl

        MatGLLoadModel = getattr(matgl, "load_model", None)
    except Exception:
        MatGLLoadModel = None

    _USING_LEGACY_M3GNET = False
except Exception:
    try:
        from m3gnet.models import M3GNet as LegacyM3GNet  # type: ignore
        from m3gnet.models import M3GNetCalculator  # type: ignore
        from m3gnet.models import Potential as LegacyM3GNetPotential  # type: ignore

        _USING_LEGACY_M3GNET = True
    except Exception:
        M3GNetCalculator = None  # type: ignore
        LegacyM3GNet = None  # type: ignore
        LegacyM3GNetPotential = None  # type: ignore
        _USING_LEGACY_M3GNET = False

try:
    from mace.calculators import MACECalculator
except Exception:
    MACECalculator = None  # type: ignore

try:
    from mattersim.forcefield import MatterSimCalculator
except Exception:
    MatterSimCalculator = None  # type: ignore

try:
    from pfp_api_client.pfp.calculators.ase_calculator import (
        ASECalculator as MatlantisASECalculator,
    )
    from pfp_api_client.pfp.estimator import Estimator as MatlantisEstimator
    from pfp_api_client.pfp.estimator import EstimatorCalcMode
except Exception:
    MatlantisEstimator = None  # type: ignore
    MatlantisASECalculator = None  # type: ignore
    EstimatorCalcMode = None  # type: ignore

try:
    from orb_models.forcefield.calculator import ORBCalculator
    from orb_models.forcefield.pretrained import ORB_PRETRAINED_MODELS
except Exception:
    ORBCalculator = None  # type: ignore
    ORB_PRETRAINED_MODELS = None  # type: ignore

try:
    from matris.applications.base import MatRISCalculator
    from matris.model.model import MatRIS as MatRISModel
except Exception:
    MatRISCalculator = None  # type: ignore
    MatRISModel = None  # type: ignore

try:
    from eqnorm.calculator import EqnormCalculator
except Exception:
    EqnormCalculator = None  # type: ignore

try:
    from alphanet.config import All_Config as AlphaNetAllConfig
    from alphanet.infer.calc import AlphaNetCalculator
except Exception:
    AlphaNetCalculator = None  # type: ignore
    AlphaNetAllConfig = None  # type: ignore

try:
    from hienet.hienet_calculator import HIENetCalculator
except Exception:
    HIENetCalculator = None  # type: ignore

try:
    from nequix.calculator import NequixCalculator
except Exception:
    NequixCalculator = None  # type: ignore

try:
    from upet.calculator import UPETCalculator
except Exception:
    UPETCalculator = None  # type: ignore

try:
    from tace.foundations import tace_foundations
except Exception:
    tace_foundations = None  # type: ignore

try:
    from tace.interface.ase import TACEAseCalc
except Exception:
    TACEAseCalc = None  # type: ignore

try:
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator  # type: ignore
except Exception:
    FAIRChemCalculator = None  # type: ignore

FAIRChemV1Calculator = None  # type: ignore
FAIRChemV1Predictor = None  # type: ignore

try:
    from tensorpotential.calculator.asecalculator import TPCalculator
except Exception:
    TPCalculator = None  # type: ignore

try:
    from tensorpotential.calculator.foundation_models import (
        MODELS_NAME_LIST as GRACE_MODEL_NAMES,
        grace_fm,
    )
except Exception:
    GRACE_MODEL_NAMES: List[str] = []
    grace_fm = None  # type: ignore

try:
    from deepmd.calculator import DP as DeePMDCalculator
except Exception:
    DeePMDCalculator = None  # type: ignore

SevenNetCalculator = None  # type: ignore
_SEVENNET_PACKAGE = None
try:
    from sevenn.calculator import SevenNetCalculator

    _SEVENNET_PACKAGE = "sevenn"
except Exception:
    _sevennet_spec = importlib.util.find_spec("sevennet")
    if _sevennet_spec is not None:
        try:
            from sevennet.ase import SevenNetCalculator

            _SEVENNET_PACKAGE = "sevennet"
        except Exception:
            SevenNetCalculator = None  # type: ignore
            _SEVENNET_PACKAGE = None

from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
from ase.io.lammpsdata import Prism
from ase.io.vasp import write_vasp_xdatcar
from ase.optimize import BFGS

try:
    from ase.constraints import FixAtoms, StrainFilter, UnitCellFilter
except ImportError:
    from ase.constraints import FixAtoms
    from ase.filters import StrainFilter, UnitCellFilter  # type: ignore

from ase.md import velocitydistribution
from ase.md.verlet import VelocityVerlet

try:
    import resource
except Exception:
    resource = None  # type: ignore

try:
    from ase.md.andersen import Andersen
except Exception:
    Andersen = None  # type: ignore

try:
    from ase.md.langevin import Langevin
except Exception:
    Langevin = None  # type: ignore

try:
    from ase.md.bussi import Bussi
except Exception:
    Bussi = None  # type: ignore

try:
    from ase.md.nose_hoover_chain import NoseHooverChainNVT
except Exception:
    NoseHooverChainNVT = None  # type: ignore

_nequip_spec = importlib.util.find_spec("nequip")
_nequip_ase_spec = importlib.util.find_spec("nequip.ase") if _nequip_spec else None
if _nequip_ase_spec is not None:
    from nequip.ase import NequIPCalculator
else:
    NequIPCalculator = None  # type: ignore

DEFAULT_ORB_MODEL = "orb-v3-conservative-20-omat"
DEFAULT_SEVENNET_MODEL = "7net-0"
DEFAULT_EQNORM_MODEL = "eqnorm-mptrj"
DEFAULT_MATRIS_MODEL = "matris_10m_oam"
DEFAULT_ALPHANET_MODEL = "AlphaNet-MATPES-r2scan"
DEFAULT_HIENET_MODEL = "HIENet-0"
DEFAULT_NEQUIX_MODEL = "nequix-mp-1"
DEFAULT_FAIRCHEM_MODEL = "esen-sm-direct-all-oc25"
DEFAULT_GRACE_MODEL = "GRACE-2L-MP-r6"

_SEVENNET_FILE_TYPES = frozenset({"checkpoint", "torchscript"})
_GRAPH_CONVERTER_ALGORITHMS = frozenset({"fast", "legacy"})
_EQNORM_VARIANT_ALIASES: Dict[str, List[str]] = {
    DEFAULT_EQNORM_MODEL: [DEFAULT_EQNORM_MODEL, "eqnorm", "eqnormmptrj"],
    "eqnorm-omat": ["eqnorm-omat", "eqnormomat", "omat"],
    "eqnorm-max-mptrj": ["eqnorm-max-mptrj", "eqnormmaxmptrj", "max-mptrj", "maxmptrj"],
}
_EQNORM_NAMED_MODELS: Dict[str, Dict[str, Any]] = {
    DEFAULT_EQNORM_MODEL.casefold(): {
        "display_name": DEFAULT_EQNORM_MODEL,
        "aliases": ["eqnorm"],
        "model_name": "eqnorm",
        "model_variant": DEFAULT_EQNORM_MODEL,
        "checkpoint_filename": f"{DEFAULT_EQNORM_MODEL}.pt",
        "checkpoint_url": "https://ndownloader.figshare.com/files/55429685",
        "article_api_url": "https://api.figshare.com/v2/articles/29153315",
    },
}
_MATRIS_NAMED_MODEL_DOWNLOADS: Dict[str, tuple[str, str]] = {
    DEFAULT_MATRIS_MODEL: (
        "MatRIS_10M_OAM.pth.tar",
        "https://api.figshare.com/v2/file/download/59142728",
    ),
    "matris_10m_mp": (
        "MatRIS_10M_MP.pth.tar",
        "https://api.figshare.com/v2/file/download/59143058",
    ),
}
_ALPHANET_NAMED_MODELS: Dict[str, Dict[str, Any]] = {
    DEFAULT_ALPHANET_MODEL.casefold(): {
        "display_name": DEFAULT_ALPHANET_MODEL,
        "aliases": ["matpes"],
        "checkpoint_filename": "r2scan_1021.ckpt",
        "checkpoint_url": (
            "https://raw.githubusercontent.com/zmyybc/AlphaNet/jax_and_zbl/pretrained/"
            "MATPES/r2scan_1021.ckpt"
        ),
        "config_filename": "matpes.json",
        "config_url": (
            "https://raw.githubusercontent.com/zmyybc/AlphaNet/jax_and_zbl/pretrained/"
            "MATPES/matpes.json"
        ),
    },
    "alphanet-aqcat25".casefold(): {
        "display_name": "AlphaNet-AQCAT25",
        "aliases": ["aqcat25"],
        "checkpoint_filename": "aqcat_1021.ckpt",
        "checkpoint_url": (
            "https://raw.githubusercontent.com/zmyybc/AlphaNet/jax_and_zbl/pretrained/"
            "AQCAT25/aqcat_1021.ckpt"
        ),
        "config_filename": "aqcat.json",
        "config_url": (
            "https://raw.githubusercontent.com/zmyybc/AlphaNet/jax_and_zbl/pretrained/"
            "AQCAT25/aqcat.json"
        ),
    },
    "alphanet-mptrj-v1".casefold(): {
        "display_name": "AlphaNet-MPtrj-v1",
        "aliases": ["mptrj", "mptrj-v1"],
        "checkpoint_filename": "alphanet_mptrj_v1.ckpt",
        "checkpoint_url": "https://api.figshare.com/v2/file/download/53851133",
        "config_filename": "mp.json",
        "config_url": (
            "https://raw.githubusercontent.com/zmyybc/AlphaNet/jax_and_zbl/pretrained/"
            "MPtrj/mp.json"
        ),
    },
    "alphanet-oma-v1".casefold(): {
        "display_name": "AlphaNet-oma-v1",
        "aliases": ["oma", "oma-v1"],
        "checkpoint_filename": "alphanet_oma_v1.ckpt",
        "checkpoint_url": "https://api.figshare.com/v2/file/download/53851139",
        "config_filename": "oma.json",
        "config_url": (
            "https://raw.githubusercontent.com/zmyybc/AlphaNet/jax_and_zbl/pretrained/"
            "OMA/oma.json"
        ),
    },
}
_HIENET_NAMED_MODELS: Dict[str, Dict[str, Any]] = {
    DEFAULT_HIENET_MODEL.casefold(): {
        "display_name": DEFAULT_HIENET_MODEL,
        "aliases": ["hienet", "hienet-0", "hienet-v3", "v3"],
        "checkpoint_filename": "HIENet-V3.pth",
        "checkpoint_url": (
            "https://raw.githubusercontent.com/divelab/AIRS/"
            "f7b0bde44400e2be8de0488009ee9f46925d6885/OpenMat/HIENet/checkpoints/HIENet-V3.pth"
        ),
    },
}

_FAIRCHEM_V1_IMPORT_PATHS = (
    "fairchem_core.common.relaxation.ase_utils",
    "ocpmodels.common.relaxation.ase_utils",
    "fairchem.core.common.relaxation.ase_utils",
    "fairchem.common.relaxation.ase_utils",
)
_FAIRCHEM_V1_PREDICTOR_IMPORT_PATHS = (
    "fairchem_core.common.relaxation.predictor",
    "ocpmodels.common.relaxation.predictor",
    "fairchem.core.common.relaxation.predictor",
    "fairchem.common.relaxation.predictor",
)
_FAIRCHEM_V1_PREDICTOR_CLASS_NAMES = (
    "Predictor",
    "OCPredictor",
    "OCPPredictor",
)

from .backend_common import (
    _callable_declares_parameter,
    _callable_supports_parameter,
    _coerce_bool_tag,
    _coerce_int_tag,
    _download_file_to_path,
    _looks_like_filesystem_path,
    _parse_optional_bool_tag,
    _resolve_device,
)
from .api import (
    get_backend_capabilities,
    list_backends,
    md,
    relax,
    single_point,
)
from .backends.alphanet import (
    _build_alphanet_calculator,
    _ensure_alphanet_named_model_files,
    _load_alphanet_config,
)
from .backends.chgnet import _build_chgnet_calculator, _load_chgnet_model
from .backends.eqnorm import (
    _build_eqnorm_calculator,
    _ensure_eqnorm_named_model_checkpoint,
    _ensure_eqnorm_torch_safe_globals,
    _match_eqnorm_variant,
    _normalize_eqnorm_key,
    _normalize_eqnorm_variant,
    _resolve_eqnorm_download_url,
    _resolve_eqnorm_named_model_spec,
    _resolve_eqnorm_variant,
    _stage_eqnorm_checkpoint,
    _temporarily_stage_eqnorm_local_checkpoint,
)
from .backends.fairchem import (
    _FairChemV1PredictorCalculator,
    _as_numpy,
    _build_fairchem_calculator,
    _build_fairchem_v1_calculator,
    _build_fairchem_v1_predictor,
    _get_fairchem_v1_calculator_cls,
    _get_fairchem_v1_predictor_cls,
    _normalize_fairchem_prediction,
    _pick_fairchem_prediction_value,
    _run_fairchem_v1_prediction,
)
from .backends.hienet import (
    _build_hienet_calculator,
    _ensure_hienet_named_model_checkpoint,
)
from .backends.m3gnet import _build_m3gnet_calculator, _build_mace_calculator
from .backends.matris import (
    _build_matris_calculator,
    _ensure_matris_named_model_checkpoint,
    _instantiate_matris_calculator,
    _load_matris_checkpoint_model,
)
from .backends.misc import (
    _build_deepmd_calculator,
    _build_equflash_calculator,
    _build_grace_calculator,
    _build_matlantis_calculator,
    _build_orb_calculator,
    _build_tace_calculator,
    _build_upet_calculator,
    _get_equflash_calculator_cls,
    _list_matlantis_calc_modes,
    _resolve_matlantis_calc_mode,
)
from .backends.nequix import (
    _build_nequix_calculator,
    _list_nequix_named_models,
    _normalize_nequix_backend,
    _resolve_nequix_model_name,
)
from .backends.nequip_family import (
    _build_allegro_calculator,
    _build_nequip_calculator,
    _build_nequip_family_calculator,
    _override_model_graph_converter_algorithm,
    _resolve_graph_converter_algorithm,
)
from .backends.sevennet_family import (
    _build_flashtp_calculator,
    _build_sevennet_calculator,
    _build_sevennet_family_calculator,
    _is_sevennet_flash_available,
)
from .io.inputs import (
    _apply_initial_magnetization,
    _expand_magmom_to_atoms,
    _flatten,
    _infer_type_map,
    _normalize_species_labels,
    _parse_magmom_values,
    _resolve_mlp_tag,
    parse_key_value_file,
    read_structure,
)
from .io.trajectories import (
    _XDATCAR_STATE,
    _append_variable_cell_configuration,
    _append_xdatcar_configuration,
    _count_symbols_in_order,
    _initialize_xdatcar_state,
    _rewrite_first_xdatcar_frame,
    _write_lammps_trajectory_step,
    _write_xdatcar_step,
)
from .io.vasp_compat import (
    _ACTIVE_PSEUDO_SCF_SETTINGS,
    _ACTIVE_VASP_INPUT_PATHS,
    _NebChainApproximation,
    _NebImageResult,
    _PSEUDO_SCF_INCAR_TAGS,
    _PseudoScfSettings,
    _VaspCompatRecorder,
    _VaspInputPaths,
    _VasprunStep,
    _active_pseudo_scf_settings,
    _active_vasp_input_paths,
    _append_kpoints_xml,
    _append_oszicar_compat_step,
    _append_outcar_compat_step,
    _append_outcar_footer,
    _append_outcar_metadata_header,
    _append_pseudo_scf_xml_step,
    _append_structure_xml,
    _build_atominfo_xml,
    _coerce_neb_reference_positions,
    _estimate_neb_chain_approximation,
    _extract_potcar_titles,
    _format_outcar_ediff,
    _full_to_voigt_stress,
    _initialize_vasp_compat_outputs,
    _matrix_to_nested_list,
    _pseudo_scf_settings_from_incar,
    _read_non_comment_lines,
    _record_vasp_compat_step,
    _resolve_pseudo_scf_settings,
    _safe_get_forces,
    _safe_get_stress_matrix,
    _selected_incar_path,
    _stress_mode_from_isif,
    _voigt_to_full_stress,
    _write_vasprun_xml,
)
from .models import (
    BackendCapabilities,
    BackendConfig,
    BackendSpec,
    CalculationResult,
    MDConfig,
    MDResult,
    RelaxConfig,
    RelaxResult,
    RunContext,
    RunStep,
    SinglePointConfig,
    SinglePointResult,
    VaspCompatConfig,
    coerce_backend_config,
    normalize_thermostat_name,
    run_steps_to_energy_rows,
    thermostat_to_mdalgo,
)
from .observers import (
    CompositeObserver,
    PrintProgressObserver,
    RunObserver,
    VaspCompatObserver,
    coerce_observer,
)
from .runtime.common import _extract_numeric_attribute, _resolve_calculator, _working_directory
from .runtime.md import _estimate_tdamp, _rescale_velocities, _select_md_dynamics, run_md
from .runtime.neb import (
    _collect_neb_image_results,
    _discover_neb_image_directories,
    _parse_vasprun_varray_rows,
    _read_last_vasprun_step,
    _resolve_neb_image_structure_path,
    _write_neb_parent_aggregate_outputs,
    run_neb_images,
)
from .runtime.registry import (
    _CALCULATOR_BUILDERS,
    _SIMPLE_CALCULATORS,
    _attach_fallback_calculator,
    _build_calculator_from_tags,
    _build_calculator_from_init_factory,
    build_calculator,
    get_calculator,
)
from .runtime.relax import (
    _EnergyConvergenceMonitor,
    _make_relaxation_builder,
    _temporarily_freeze_atoms,
    run_relaxation,
)
from .runtime.single import run_single_point
from .settings.incar import (
    KBAR_TO_EV_PER_A3,
    IncarSettings,
    SUPPORTED_INCAR_TAGS,
    SUPPORTED_ISIF_VALUES,
    _get_lammps_trajectory_interval,
    _is_neb_like_incar,
    _is_truthy_flag,
    _load_incar,
    _load_incar_settings,
    _normalize_isif,
    _parse_neb_image_count,
    _parse_optional_float,
    _should_write_energy_csv,
    _should_write_lammps_trajectory,
    _should_write_oszicar_pseudo_scf,
    _should_write_pseudo_scf,
    _warn_for_unsupported_incar_tags,
)
from .cli import main


def _format_energy_value(value: float) -> str:
    """Return energy in VASP-like ``E`` notation with mantissa < 1."""

    if value == 0:
        return "+.00000000E+00"

    mantissa_str, exponent_str = f"{value:.8e}".split("e")
    mantissa = float(mantissa_str) / 10.0
    exponent = int(exponent_str) + 1
    formatted = f"{mantissa:+.8f}".replace("+0.", "+.").replace("-0.", "-.")
    return f"{formatted}E{exponent:+03d}"


def _format_oszicar_energy(value: float) -> str:
    """Return right-aligned OSZICAR energy token."""

    return f"{_format_energy_value(value):>15s}"


def _format_oszicar_residual(value: float) -> str:
    """Return VASP-like residual notation used in electronic step lines."""

    return f"{value:+.3E}".replace("+0.", "+.").replace("-0.", "-.")
